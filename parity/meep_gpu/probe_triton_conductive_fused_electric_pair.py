"""Device gate for the CONDUCTIVE PML fused ELECTRIC D/E pair on Triton.

DEVICE STATUS: **UNRUN.** This module is written to be run on a CUDA host with an
    idle device; nothing in this tree may cite it as a release until an artifact
    exists with ``device_status: RUN`` and ``release.released: true``.

Conductive ``step_D`` welded into the ORDINARY ``update_E`` -- the last buildable
non-folded D->E cell on the Triton fusion board.

===========================================================================
WHAT THE PRODUCT CLAIMS, AND WHAT EACH LEG MEASURES
===========================================================================

The claim is narrow and checkable, and it is a claim about FOUR shipped bodies
rather than two:

* the head is ``conductivity.conductive_pml_curl_step``'s, which is itself
  ``kernels.pml_curl_step``'s;
* the three recurrences are ``conductivity._conductive_component``, called through
  a helper that is that function with ONE statement moved out;
* the wall clear, the D store and the constitutive half are
  ``kernels.fused_curl_constitutive_D``'s.

1. **TRANSCRIPTION (no device).** FOUR exact statement-list equalities against the
   shipped sources, read from the FILES so they bite on the merge bar as well as
   here. Nothing is ``ast.unparse``d -- what is being compared includes the
   PARENTHESISATION, and unparsing re-derives minimal parentheses, which would
   silently equate ``((c * cf) - curl) * ci`` with a different float32 grouping.
2. **REDUCTION (device).** ``COND0 = COND1 = COND2 = 0`` must reproduce the shipped
   ``kernels.fused_curl_constitutive_D`` BIT FOR BIT over a complete step, on the
   same volumes, while BOTH differ from the conductive array path. This is the
   transcription claim measured on silicon instead of in source text, and it is the
   leg that would catch a mis-ordered argument in the plan's ``run`` that no amount
   of source diffing can see.
3. **IDENTITY (device).** One launch of the fused kernel leaves the engine
   bit-identical, as uint32 words, per COMPLETE driver step, to BOTH the CuPy array
   path AND the two separately certified Triton products it replaces (``conductive
   PML`` on ``step_D``, ``ordinary`` on ``update_E``), at one launch and at ~60,
   over every allocated volume, in three value classes.
4. **THE CARRY (device), and it is what makes this cell worth anything at all.**
   The single corpus row declares an ELECTRIC source, which the driver injects
   BETWEEN the two halves -- and on a conductive row it injects through the
   ``condinv``-scaled path (driver.py:3295-3296). Every CARRY case is run TWICE --
   once with the shipped :class:`~meep_gpu.deposit_repair.LeadingRepairPlan` /
   :class:`~meep_gpu.deposit_repair.TrailingRepairPlan` pair and once WITHOUT it --
   and the unbracketed run MUST DIVERGE. A carry family whose null control agreed
   would be measuring a seam that carried no deposit.

===========================================================================
THE SEAM
===========================================================================

``driver.step`` runs, in order (driver.py:3292-3304)::

    step_D -> ELECTRIC SOURCES -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

On the configurations this arm admits the two fill passes are DEAD -- the predicate
refuses every fold by name -- so the launch spans ``step_D``, ``zero_metal_D`` and
``update_E``, with the electric injection carried across it by the shipped repair.

WHAT THE CORPUS SAYS, from ``results/fusion_matrix_triton_2026-08-31_electrics``
at the D->E cell (``conductive PML``, ``ordinary``): ONE seam-instance,
``tests:TestAdjointSolver.test_damping``, shape (150, 150, 1), metallic in x and y,
``in_seam_source == true`` and ``in_seam_source_blocks == false``. The
``corpus_admission`` leg re-derives that from the census rather than citing it.

THAT ROW IS WALLED, so the inline ``zero_metal_D`` carry is LIVE on it -- unlike
every other 08-31 electric cell, where the wall clear had to be exercised on a case
the corpus does not itself contain. The all-periodic case is kept anyway, as the
control in which the flags compile to False.

WHY THE RECURRENCE IS THE INTERESTING PART. ``_conductive_component`` selects among
FOUR cases with two EXACT float comparisons (``km != 1.0``, ``sinv != 1.0``), and
writes ``fu_D`` only where ``dsigu`` and ``f_cond_D`` only where ``dsig``. A weld
that widened either store, or that flattened one of the five branch expressions'
parentheses, is smooth and plausible and wrong; the mutation table below arms one
rewrite per branch and per store mask.

Usage::

    # laptop, no CUDA, no Triton -- the legs that need neither
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_conductive_fused_electric_pair.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>

    # CUDA host, verified-empty device -- the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_conductive_fused_electric_pair.py \\
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
PRODUCT_MODULE = "meep_gpu.triton_kernels.conductive_fused_electric_pair"
KERNEL_FILE = os.path.join(PACKAGE_DIR, "conductive_fused_electric_pair.py")
KERNEL_NAME = "conductive_fused_curl_constitutive_D"

#: 2d_cond_pml / TestAdjointSolver.test_damping's own sigma scale.
SIGMA_CORPUS = 0.4

#: The ONE statement ``_conductive_component_registers`` may drop, and the ONE it
#: may add. Written out so a second deletion cannot hide behind a set difference.
MOVED_STORE = "tl.store(f_ptr + idx, v, mask=live)"
ADDED_RETURN = "return v"

#: The helper the product calls three times, and the shipped function it is.
HELPER_NAME = "_conductive_component_registers"
SHIPPED_HELPER = "_conductive_component"

#: The device cases. ``(name, cell, boundaries, steps, components)``.
#:
#: ``components`` is the conductivity map: ``None`` installs one volume on every D
#: component (all three ``COND`` constexprs 1), a tuple installs it on those
#: components only. Z STAYS PERIODIC in every case: on a 2-D cell it is the
#: INVARIANT axis and ``Grid`` refuses a PEC there by name (grid.py:842-877).
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any, int, Any, Any], ...] = (
    # THE CORPUS GEOMETRY: metallic in x and y, which is what
    # tests:TestAdjointSolver.test_damping declares. ZM_X and ZM_Y fire and both
    # metallic ghost arms are entered. This is the mutation case, for that reason.
    ("wall_xy", (1.4, 1.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 60, None, {"x": 2, "y": 2}),
    # ALL PERIODIC: ZM_* all compile to False and every ghost arm is the periodic
    # wrap. The control in which the wall clear is dead.
    ("periodic", (1.4, 1.0, 0.0), "periodic", 60, None, {"x": 2, "y": 2}),
    # ONE WALLED AXIS ONLY: the case where a ROTATED wall table still clears the
    # same NUMBER of planes and clears the wrong ones.
    ("wall_x", (1.4, 1.0, 0.0),
     {"x": "metallic", "y": "periodic", "z": "periodic"}, 60, None, {"x": 2, "y": 2}),
    # A MIXED GRID: one lossy component beside two lossless ones, which is the
    # configuration the three COND constexprs exist for. `_conductive_component`'s
    # COND == 0 arm must be byte-identical to the plain recurrence ON THE SAME
    # LAUNCH as its COND == 1 arm, and only a mixed case can measure that.
    ("mixed_cond_z", (1.4, 1.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 60, ("Dz",),
     {"x": 2, "y": 2}),
    # A DEEP LAYER, and its only purpose is the size of ONE region. Case A of the
    # four-case table -- `u_split` -- reaches a store only where `dsig AND dsigu`
    # hold, which for target 2 is the INTERSECTION of the x-graded and y-graded
    # bands. On the 2-cell layers above that intersection is NINE cells, counted
    # from the coefficients, and run c measured an associativity rewrite of
    # `u_split` as UNCAUGHT across all nine over four steps. Eight-cell layers make
    # it 225. The mutation was not the problem; the case it was scored on was.
    ("deep_pml", (3.0, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 20, None,
     {"x": 8, "y": 8}),
    # ONE LAUNCH: a divergence here is attributable to a single launch rather than
    # to an accumulation.
    ("wall_xy_single_launch", (1.4, 1.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 1, None, {"x": 2, "y": 2}),
)

CASES_BY_NAME: Dict[str, Tuple[Any, ...]] = {case[0]: case for case in CASES}

#: The case every kernel mutation is scored on unless it names another. ``wall_xy``
#: IS the corpus geometry here, and it is also the only one where the wall-clear
#: lines are live.
DEFAULT_MUTATION_CASE = "wall_xy"

#: Per mutation, the case it is scored on. TRAP: a mutation scored on a grid that
#: never ENTERS the branch it rewrites reports UNCAUGHT while measuring nothing.
MUTATION_CASE: Dict[str, str] = {
    # THE ONLY REWRITE OF THE `COND == 0` ARM, and it is DEAD on every all-lossy
    # case: with all three flags at 1, `COND == 0` is already False and changing the
    # literal it compares against changes nothing. Measured UNCAUGHT on run b for
    # exactly that reason, and re-pointed at the mixed grid, where two of the three
    # components take that arm.
    "m_lossless_arm_takes_the_conductive_one": "mixed_cond_z",
    # SCORED ON THE DEEP LAYER, because the region it rewrites is nine cells on the
    # default case and 225 there. See the `deep_pml` entry in CASES.
    "m_case_a_regrouped": "deep_pml",
}

#: Mutations that rewrite a line reached ONLY in case A of the four-case table, with
#: the number of cells the scored case must give them. A rewrite scored on a case
#: whose case-A region is a handful of cells reports UNCAUGHT while measuring almost
#: nothing, which is what run c did; :func:`mutation_case_for` counts the region from
#: the case's own coefficients and refuses a pairing below the floor.
MUTATION_REQUIRES_CASE_A_CELLS: Dict[str, int] = {
    "m_case_a_regrouped": 64,
}

#: Per mutation, axes that must carry a live WALL on the scored case.
MUTATION_REQUIRES_WALL: Dict[str, Tuple[str, ...]] = {
    "m_wall_clear_dropped": ("x",),
    "m_wall_clear_is_the_magnetic_familys_one_row": ("x",),
    "m_wall_clear_after_the_store": ("x",),
}

#: Per mutation, components that must be LOSSY on the scored case. A rewrite inside
#: the ``COND == 1`` arm scored on a lossless component is a dead branch, which is
#: the same trap ``MUTATION_REQUIRES_WALL`` closes one rung up.
MUTATION_REQUIRES_CONDUCTIVE: Dict[str, Tuple[str, ...]] = {}

#: Per mutation, components that must be LOSSLESS on the scored case -- the mirror
#: of the table above, for a rewrite of the ``COND == 0`` arm.
MUTATION_REQUIRES_LOSSLESS: Dict[str, Tuple[str, ...]] = {
    "m_lossless_arm_takes_the_conductive_one": ("Dx", "Dy"),
}

#: The two mutations this gate scores as CONFIRMED NULLS, with the measurement that
#: confirms each. A null is a rewrite whose defect is real at the WORD layer and
#: whose effect on this engine's coefficients is provably empty -- not a rewrite
#: nobody could be bothered to make bite. The ``coefficient_predicates`` no-device
#: leg re-measures both on every run, so a change to `pml._split_field_coefficients`
#: that made either one live would turn this gate red rather than quietly widen it.
MUTATION_EVIDENCE: Dict[str, str] = {
    "m_dsig_becomes_a_tolerance":
        "`kms` and `sinv` are both 1.0 EXACTLY where sigma is zero and both below 1 "
        "everywhere else -- measured over every axis of every case grid by the "
        "`coefficient_predicates` leg: max(kms) == 1.0, no entry above 1, and the "
        "tolerance form `(km < 0.999999) | (si < 0.999999)` selects the SAME set as "
        "`(km != 1.0) | (si != 1.0)` at every entry. A tolerance can only differ "
        "where a coefficient sits within it of 1 without being 1, and this engine "
        "constructs none",
    "m_dsig_uses_or_instead_of_and":
        "`kms != 1.0` and `sinv != 1.0` are the SAME SET on every axis of every case "
        "grid -- both are derived from the one graded sigma, so neither departs from "
        "1 without the other. `|` and `&` therefore select identically, measured "
        "entry by entry by the `coefficient_predicates` leg",
}

#: The CARRY family: the same grids with a real ELECTRIC deposit IN THIS SEAM.
#: THIS IS THE PRODUCT, not an extra — the cell's only corpus row sources
#: electrically. One all-periodic grid (the corpus geometry) and one walled grid,
#: where the deposit and the wall clear can interact.
CARRY_CASES: Tuple[str, ...] = ("wall_xy", "periodic", "mixed_cond_z")

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

#: THE CONDUCTIVE HISTORY IS PART OF THE COMPARISON, and it is required PER
#: COMPONENT rather than in :data:`REQUIRED`, because ``Fields`` allocates
#: ``f_cond_D<c>`` only for a component that carries a conductivity
#: (fields.py:738, :825) -- which is the whole point of the mixed case. Demanding
#: all three unconditionally failed the mixed leg on run a while measuring nothing
#: about the weld. What IS required is derived from the live object below: every
#: component ``condfac_for`` answers for must have its history in the scan, because
#: that volume is written under a MASK (``live & dsig``) and read on the next step,
#: so a weld that widened the mask would be invisible in D and E on the step it
#: happened.
def required_history(fields) -> Tuple[str, ...]:
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        return ()
    out = []
    for component in ("Dx", "Dy", "Dz"):
        try:
            if reader(component) is not None:
                out.append("f_cond_" + component)
        except Exception:  # noqa: BLE001 - an unanswerable component is not conductive
            continue
    return tuple(out)

MATERIAL = ("eps", "inv_eps")

#: The volumes THIS SEAM writes. A leg in which none of them moves measured nothing
#: about this launch, whatever else moved.
SEAM_OUTPUTS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
    "f_cond_Dx", "f_cond_Dy", "f_cond_Dz",
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
    # WHICH BYTES THIS PROCESS IMPORTED, and the one readable verdict beside the
    # gate's own spelling. Measured on the 2026-09-09 artifact: this probe was the
    # only one of the four in its round that wrote NEITHER — `imported_source_sha256`
    # empty and `canonical_verdict` absent — so `seed_triton_welds.py` refused it as
    # recording no imported digests, and a family with a released gate stayed
    # unweldable for want of one call. The sibling probes have carried it since
    # 2026-08-19 (probe_triton_cylindrical_real_fused_electric_pair.py:202).
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
        "meep_gpu/triton_kernels/conductive_fused_electric_pair.py",
        "meep_gpu/triton_kernels/conductivity.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_conductive_fused_electric_pair.py",
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


def _function_source(text: str, tree: ast.Module, name: str) -> str:
    """One function as source text, DECORATOR INCLUDED.

    THE DECORATOR IS PREPENDED, and it is not cosmetic. ``ast.get_source_segment``
    on a ``FunctionDef`` EXCLUDES the decorator list, so a mutant compiled from the
    bare segment is a PLAIN PYTHON FUNCTION with no ``[grid]`` launcher and the first
    mutation leg dies with "'function' object is not subscriptable" -- measured
    2026-08-20 on a sibling gate, and again 2026-08-31 on two more.
    """
    import textwrap  # noqa: PLC0415

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            decorators = [ast.get_source_segment(text, decorator)
                          for decorator in node.decorator_list]
            if any(segment is None for segment in decorators):
                raise AssertionError(f"{name}'s decorator source is unavailable")
            return "\n".join(
                [f"@{segment}" for segment in decorators]
                + [textwrap.dedent(ast.get_source_segment(text, node))])
    raise AssertionError(f"{name} is not defined in {KERNEL_FILE}")


def shipped_source() -> str:
    """The HELPER and the kernel, in that order -- the mutation harness's input.

    BOTH, and that is what makes the helper mutable. The kernel CALLS
    ``_conductive_component_registers``, so a mutant module carrying only the kernel
    would not import; and a mutation table that could not reach the helper could not
    arm a single rewrite of the four-case recurrence, which is where this product's
    only new arithmetic decision lives.
    """
    text = open(KERNEL_FILE, encoding="utf-8").read()
    tree = ast.parse(text)
    return (_function_source(text, tree, HELPER_NAME) + "\n\n"
            + _function_source(text, tree, KERNEL_NAME))


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription
# ---------------------------------------------------------------------------

def _calls(path: str, function: str, callee: str) -> List[List[str]]:
    """Every call to ``callee`` inside ``function``, as argument-name lists.

    Read through the AST rather than off the text, because a call wrapped across two
    source lines is ONE statement and a line-based reader would see two.
    """
    text = open(path, encoding="utf-8").read()
    node = next((n for n in ast.walk(ast.parse(text))
                 if isinstance(n, ast.FunctionDef) and n.name == function), None)
    if node is None:
        raise AssertionError(f"{function} is not defined in {path}")
    return [[ast.unparse(a) for a in call.args]
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == callee]


def _slice(statements: Sequence[str], first: str, last: str) -> List[str]:
    """The statements from the one starting with ``first`` to the one before ``last``.

    Both anchors are asserted PRESENT AND UNIQUE by the caller through
    :func:`_anchor`; a silent ``next(..., default)`` here would let a renamed anchor
    compare an empty slice against an empty slice and pass.
    """
    low = _anchor(statements, first)
    high = _anchor(statements, last)
    return list(statements[low:high])


def _anchor(statements: Sequence[str], prefix: str) -> int:
    hits = [n for n, s in enumerate(statements) if s.startswith(prefix)]
    if len(hits) != 1:
        raise AssertionError(
            f"the anchor {prefix!r} matches {len(hits)} statements, not exactly one: "
            f"a slice taken from it would compare the wrong region")
    return hits[0]


def transcription_leg() -> Dict[str, Any]:
    """The claim this whole product rests on, made mechanical.

    FOUR equalities, all EXACT and all in order:

    * the HEAD (down to and including the six coefficient loads) IS
      ``conductivity.conductive_pml_curl_step``'s head;
    * that head is in turn ``kernels.pml_curl_step``'s -- the chain the module
      docstring claims, checked rather than asserted, so a change upstream that
      broke it would surface here and not in a reader's assumption;
    * the MIDDLE is exactly three calls to :data:`HELPER_NAME`, with the SAME
      ``(km, si)`` pairs and the SAME per-component pointer rows that
      ``conductive_pml_curl_step``'s own three calls use;
    * the TAIL (the wall clear, the D store and the constitutive half) IS
      ``kernels.fused_curl_constitutive_D``'s tail, minus its three ``fu_D`` stores,
      which did not disappear: they moved INTO the helper, where the shipped
      function already performs them under the four-case table's masks.

    And one more, about the helper itself: it is ``_conductive_component`` minus
    exactly :data:`MOVED_STORE` and plus exactly :data:`ADDED_RETURN`.

    Together they say: nothing in this body was written here. The last check states
    that positively, over the union of the shipped bodies.
    """
    kernels_py = os.path.join(PACKAGE_DIR, "kernels.py")
    conductivity_py = os.path.join(PACKAGE_DIR, "conductivity.py")
    fused = body_statements(KERNEL_FILE, KERNEL_NAME)
    helper = body_statements(KERNEL_FILE, HELPER_NAME)
    shipped_helper = body_statements(conductivity_py, SHIPPED_HELPER)
    conductive_curl = body_statements(conductivity_py, "conductive_pml_curl_step")
    ordinary_curl = body_statements(kernels_py, "pml_curl_step")
    ordinary_fused = body_statements(kernels_py, "fused_curl_constitutive_D")

    findings: List[str] = []
    head_end, tail_start = "si_z = tl.load", "if ZM_X"

    mine_head = fused[:_anchor(fused, head_end) + 1]
    theirs_head = conductive_curl[:_anchor(conductive_curl, head_end) + 1]
    if mine_head != theirs_head:
        index = next((i for i, (a, b) in enumerate(zip(mine_head, theirs_head))
                      if a != b), min(len(mine_head), len(theirs_head)))
        findings.append(
            f"the head is NOT conductive_pml_curl_step's; first difference at "
            f"statement {index}: "
            f"{mine_head[index:index + 1]} vs {theirs_head[index:index + 1]}")

    plain_head = ordinary_curl[:_anchor(ordinary_curl, head_end) + 1]
    if theirs_head[:len(plain_head)] != plain_head:
        findings.append(
            "conductive_pml_curl_step's head is no longer pml_curl_step's, so this "
            "product's head is not a transcription of both and the module docstring "
            "is wrong about the chain")

    mine_tail = fused[_anchor(fused, tail_start):]
    u_stores = {f"tl.store(u{n} + idx, n{n}, mask=live)" for n in (0, 1, 2)}
    theirs_tail = [s for s in ordinary_fused[_anchor(ordinary_fused, tail_start):]
                   if s not in u_stores]
    if mine_tail != theirs_tail:
        index = next((i for i, (a, b) in enumerate(zip(mine_tail, theirs_tail))
                      if a != b), min(len(mine_tail), len(theirs_tail)))
        findings.append(
            f"the tail is NOT fused_curl_constitutive_D's; first difference at "
            f"statement {index}: "
            f"{mine_tail[index:index + 1]} vs {theirs_tail[index:index + 1]}")

    mine_calls = _calls(KERNEL_FILE, KERNEL_NAME, HELPER_NAME)
    theirs_calls = _calls(conductivity_py, "conductive_pml_curl_step", SHIPPED_HELPER)
    if len(mine_calls) != 3 or len(theirs_calls) != 3:
        findings.append(
            f"the recurrence is called {len(mine_calls)} times here and "
            f"{len(theirs_calls)} times in the shipped curl; three components need "
            f"three")
    else:
        pairs = [call[-5:-1] for call in mine_calls]
        if pairs != [call[-5:-1] for call in theirs_calls]:
            findings.append(
                f"the (km, si) pairs are {pairs}, not the shipped curl's "
                f"{[call[-5:-1] for call in theirs_calls]}: a rotated pair is a "
                f"wrong absorber profile that no shape check would see")
        if [call[-6] for call in mine_calls] != ["curl0", "curl1", "curl2"]:
            findings.append(
                f"a component consumes another's curl: "
                f"{[call[-6] for call in mine_calls]}")
        rows = [call[:5] for call in mine_calls]
        expected = [[f"f{n}", f"u{n}", f"c{n}", f"cf{n}", f"ci{n}"] for n in range(3)]
        if rows != expected:
            findings.append(f"the pointer rows are {rows}, not {expected}")

    without_return = [s for s in helper if s != ADDED_RETURN]
    without_store = [s for s in shipped_helper if s != MOVED_STORE]
    if without_return != without_store:
        index = next((i for i, (a, b) in enumerate(zip(without_return,
                                                       without_store)) if a != b),
                     min(len(without_return), len(without_store)))
        findings.append(
            f"{HELPER_NAME} is NOT {SHIPPED_HELPER} with one statement moved; first "
            f"difference at statement {index}: "
            f"{without_return[index:index + 1]} vs {without_store[index:index + 1]}")
    if helper.count(ADDED_RETURN) != 1:
        findings.append(f"{HELPER_NAME} returns {helper.count(ADDED_RETURN)} times")
    if MOVED_STORE in helper:
        findings.append(
            f"{HELPER_NAME} still performs {MOVED_STORE!r}; the caller would then "
            f"write the wall-cleared value over a value already stored")
    if MOVED_STORE not in shipped_helper:
        findings.append(
            f"{SHIPPED_HELPER} no longer performs {MOVED_STORE!r}, so the statement "
            f"this weld claims to have MOVED is not there to move")

    # THE TWO MASKED STORES KEEP THEIR MASKS. They are the "unchanged" column of the
    # four-case table; widening either writes a component the array path leaves
    # alone, in a volume no E comparison reads on the step it happens.
    for line in ("tl.store(u_ptr + idx, u_new, mask=live & dsigu)",
                 "tl.store(c_ptr + idx, c_new, mask=live & dsig)",
                 "tl.store(u_ptr + idx, n, mask=live)"):
        if line not in helper:
            findings.append(f"the helper no longer performs {line!r}")

    invented = [s for s in fused
                if s not in set(conductive_curl) | set(ordinary_fused)
                and HELPER_NAME not in s]
    if invented:
        findings.append(f"statements that appear in NEITHER shipped body: {invented}")

    # THE E SIDE'S OWN MULTIPLY, named rather than left to the tail equality: a tail
    # taken from the MAGNETIC fused pair would satisfy a shorter comparison.
    for component in range(3):
        line = (f"src{component} = v{component} * tl.load("
                f"ie{component} + idx, mask=live, other=0.0)")
        if line not in fused:
            findings.append(f"the E side's inverse-permittivity multiply is missing "
                            f"for component {component}: {line!r}")
    if "v2 = tl.where(at_x, 0.0, v2)" not in fused:
        findings.append("zero_metal_D's second tangential row on the x wall is "
                        "missing; the D family clears TWO components per axis")

    return {"leg": "transcription", "device": False,
            "statements_in_the_product": len(fused),
            "statements_in_the_helper": len(helper),
            "head_statements": len(mine_head),
            "tail_statements": len(mine_tail),
            "recurrence_calls": mine_calls,
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
    sigma = keywords.pop("sigma", SIGMA_CORPUS)
    components = keywords.pop("components", None)
    grid = Grid(resolution=10.0, cell_size=keywords.pop("cell_size", (1.2, 1.0, 0.0)),
                dimensions=2, courant=0.35,
                boundaries=keywords.pop("boundaries", {"x": "metallic",
                                                       "y": "metallic",
                                                       "z": "periodic"}),
                k_point=keywords.pop("k_point", (0.0, 0.0, 0.0)), **keywords)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    if sigma is not None:
        volume = np.full(grid.shape, sigma, dtype=np.float32)
        fields.set_d_conductivity({name: volume for name in components}
                                  if components is not None else volume)
    rng = np.random.default_rng(17)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, size=grid.shape).astype(np.float32)
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


class _Magnetic:
    field_type = "B"


class _Electric:
    """An electric source that publishes the index the injection writes.

    NOT INTEGRATED, which is MEEP's default and the corpus row's: since the
    2026-09-01 lift the driver replays its condinv rescale SPARSELY at exactly
    the deposit cells this table names, so the clause that used to refuse it is
    quiet and the product carries it.
    """

    field_type = "D"
    component = "Ez"
    is_integrated = False

    def __init__(self, index=(1, 1, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = index


class _IntegratedElectric(_Electric):
    """The electric source a conductive run could ALWAYS carry -- the driver
    injects it point-wise and skips the rescale (MEEP step.cpp:300)."""

    is_integrated = True


class _ScaledElectricNoTable:
    """A scaled source with NO deposit table at all — the driver's dense fallback.

    ``hasattr(source, "_point_ix")`` is False, the partition
    ``_inject_electric_through_conductivity`` uses; the surviving clause refuses
    exactly this duck-typed stranger, and no in-tree source class is one.
    """

    field_type = "D"
    component = "Ez"
    is_integrated = False


class _ElectricWithoutIndex:
    field_type = "D"


#: One-clause perturbations off the corpus family. Every one must leave the weld no
#: wider than the narrower of its two halves.
EQUIVALENCE_CASES: Tuple[Tuple[str, Dict[str, Any], Any], ...] = (
    ("corpus_row_source_admitted_after_the_lift", {}, (_Electric(),)),
    ("scaled_source_without_a_table_is_refused", {}, (_ScaledElectricNoTable(),)),
    ("integrated_electric", {}, (_IntegratedElectric(),)),
    ("magnetic_source", {}, (_Magnetic(),)),
    ("undeclared_sources", {}, None),
    ("electric_without_an_index", {}, (_ElectricWithoutIndex(),)),
    ("lossless", {"sigma": None}, (_IntegratedElectric(),)),
    ("mixed_conductivity", {"components": ("Dz",)}, (_IntegratedElectric(),)),
    ("complex_storage", {"complex_storage": True}, (_IntegratedElectric(),)),
    ("inactive_absorber", {"pml_thickness": 0}, (_IntegratedElectric(),)),
    ("periodic", {"boundaries": "periodic"}, (_IntegratedElectric(),)),
    # A Bloch phase and a metallic wall are mutually exclusive on ONE axis -- the
    # driver refuses the pair by name, and so does MEEP (boundaries.cpp) -- so this
    # perturbation carries the periodic boundary with it. It is still ONE clause off
    # the corpus family, because the `periodic` case above isolates the other half.
    ("bloch", {"k_point": (0.2, 0.0, 0.0), "boundaries": "periodic"},
     (_IntegratedElectric(),)),
)


def equivalence_leg() -> Dict[str, Any]:
    """``weld => spine curl AND spine E``, measured over one-clause perturbations."""
    from meep_gpu.grid import Mirror  # noqa: PLC0415
    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import (  # noqa: PLC0415
        constitutive_coverage,
    )

    product = importlib.import_module(PRODUCT_MODULE)
    cases = list(EQUIVALENCE_CASES) + [
        # A folded axis cannot carry a LOW-face layer: cell 0 there is the mirror
        # plane and pml._resolve_mirror_faces RAISES on a per-side request naming it.
        ("folded", {"symmetry": (Mirror("Y", 1),),
                    "boundaries": {"x": "metallic", "y": "periodic",
                                   "z": "periodic"},
                    "thickness": ((2, 2), (0, 2), (0, 0))},
         (_IntegratedElectric(),)),
    ]
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    admitted = 0
    for name, keywords, sources in cases:
        fields, pml = _laptop_fixture(**dict(keywords))
        weld = product.conductive_fused_electric_pair_coverage(fields, pml, sources)
        curl = conductivity.conductive_pml_curl_coverage(fields, pml, "step_D")
        electric = constitutive_coverage(fields, pml, "E")
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

    # WHAT THE FLAG DECIDES, measured rather than declared. It is NOT what makes
    # this cell reachable -- the corpus row's source is refused for the whole-volume
    # rescale whatever the flag says -- so what is asserted is the narrower true
    # thing: with the flag True an INTEGRATED electric source is admitted, and with
    # it False the same source is refused by name.
    fields, pml = _laptop_fixture()
    integrated = (_IntegratedElectric(),)
    before = [r for r in product.conductive_fused_electric_pair_coverage(
        fields, pml, integrated).reasons if "array module" not in r]
    original = product.CARRIES_DEPOSIT_REPAIR
    try:
        product.CARRIES_DEPOSIT_REPAIR = False
        after = [r for r in product.conductive_fused_electric_pair_coverage(
            fields, pml, integrated).reasons if "array module" not in r]
    finally:
        product.CARRIES_DEPOSIT_REPAIR = original
    if before:
        findings.append(f"an INTEGRATED electric source is refused WITH the flag: "
                        f"{before}")
    if not any("is electric" in reason for reason in after):
        findings.append(
            "flipping CARRIES_DEPOSIT_REPAIR to False did NOT refuse the integrated "
            "electric source: the flag decides nothing and the bracket this product "
            "installs is not load-bearing")

    # AND WHAT THE 2026-09-01 LIFT DECIDES, in both directions. The corpus row's
    # NON-integrated source (deposit table published) must be ADMITTED modulo the
    # backend clause — the driver replays the rescale sparsely at exactly those
    # cells — while a scaled source with NO table must still be refused for the
    # dense whole-volume fallback, by name. Otherwise this product's corpus count
    # would be a different claim from the one its docstring makes.
    scaled = [r for r in product.conductive_fused_electric_pair_coverage(
        fields, pml, (_Electric(),)).reasons if "array module" not in r]
    if scaled:
        findings.append(
            f"the corpus row's NON-integrated source is still refused after the "
            f"lift: {scaled}")
    tableless = [r for r in product.conductive_fused_electric_pair_coverage(
        fields, pml, (_ScaledElectricNoTable(),)).reasons
        if "array module" not in r]
    if not any("publishes NO deposit table" in reason for reason in tableless):
        findings.append(
            f"a scaled source with NO deposit table is NOT refused for the dense "
            f"whole-volume fallback: {tableless}")

    return {"leg": "equivalence", "device": False, "cases": rows,
            "admitted_modulo_backend": admitted,
            "reasons_with_the_flag": before,
            "reasons_with_the_flag_off": after,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — what the corpus says, re-derived from the census
# ---------------------------------------------------------------------------

CENSUS = os.path.join(HERE, "results",
                      "predicate_coverage_triton_2026-08-31_electrics")
MATRIX = os.path.join(HERE, "results",
                      "fusion_matrix_triton_2026-08-31_electrics",
                      "fusion_matrix.json")
CELL = ("D->E", "conductive PML", "ordinary")


def corpus_admission_leg() -> Dict[str, Any]:
    """The rows this arm can serve, counted rather than cited.

    Read from the census's own configuration block: a row is in this cell when it
    declares a D-side CONDUCTIVITY, an active absorber, real storage, no fold, no
    cylindrical axis, k = 0, zero beta, no BFAST, no dispersion, no nonlinearity and
    no off-diagonal epsilon. It is REACHABLE when its in-seam ELECTRIC source is one
    the repair can carry, which is the board's own ``in_seam_source_blocks`` and is
    read from there.

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
        if not configuration.get("has_conductivity"):
            continue
        if float(configuration.get("beta") or 0.0) != 0.0:
            continue
        if not configuration.get("pml_active"):
            continue
        if configuration.get("force_complex_fields"):
            continue
        if configuration.get("has_symmetry") or any(
                configuration.get("mirrored") or ()):
            continue
        if configuration.get("cylindrical") or configuration.get("bfast_active"):
            continue
        if configuration.get("has_bloch") or configuration.get("has_nonlinearity"):
            continue
        if configuration.get("has_offdiagonal_epsilon"):
            continue
        # DISPERSION IS THE `ordinary` ARM'S OWN REFUSAL, not this weld's addition:
        # `constitutive_coverage` refuses a run with poles by name, because the
        # consumer would be `(D - sum P) * inv_eps` and not the register expression
        # the tail carries. A row with poles is in the DISPERSIVE cell, not this one.
        if int(configuration.get("n_polarizations") or 0):
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
    # WHAT THIS PRODUCT SERVES OF THE CELL, recorded so the artifact carries the
    # number rather than leaving a reader to infer it. Since the 2026-09-01 lift the
    # electric rows are SERVED, not refused: the driver replays its condinv rescale
    # sparsely at the published deposit cells, and the retired whole-volume
    # composition — whose signed-zero arithmetic the `whole_volume_rescale` leg
    # still executes — survives only as the dense fallback for a tableless source.
    served = sorted(in_cell)
    return {"leg": "corpus_admission", "device": False, "rows_scanned": len(record),
            "cell": list(CELL), "rows_in_the_cell": sorted(in_cell),
            "rows_this_product_can_serve": served,
            "how_the_electric_rows_became_servable": (
                "the driver used to rescale a NON-INTEGRATED electric injection "
                "over the WHOLE volume by difference, which rewrote -0.0 to +0.0 at "
                "every cell of the target component at cells no deposit repair can "
                "name; since 2026-09-01 it replays the same composition per "
                "PUBLISHED deposit cell (identity everywhere else), the clause was "
                "lifted on the re-measured signed-zero walk (`lifted_refusal` leg), "
                "and the `whole_volume_rescale` leg keeps the retired arithmetic "
                "in the artifact"),
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


def coefficient_predicates_leg() -> Dict[str, Any]:
    """Are ``km != 1.0`` and ``sinv != 1.0`` the same predicate on this engine?

    THE EVIDENCE BEHIND TWO CONFIRMED NULLS, re-measured on every run rather than
    written down once. ``_conductive_component`` selects among its four cases with

        dsig  = (km1 != 1.0) | (si1 != 1.0)
        dsigu = (km2 != 1.0) | (si2 != 1.0)

    and two obvious rewrites of that -- softening the exact comparison to a
    tolerance, and narrowing ``|`` to ``&`` -- are real defects at the word layer
    that this engine's coefficients cannot express. Both are scored ``null``, and
    both entries in :data:`MUTATION_EVIDENCE` cite THIS leg. If
    ``pml._split_field_coefficients`` ever produced a ``kms`` above 1, or a graded
    ``sinv`` where ``kms`` is exactly 1, this leg fails and the two mutations must be
    re-scored -- which is the whole point of measuring rather than asserting.

    NO DEVICE AND NO TRITON: the PML's coefficient arrays are NumPy on this host.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    for case_name, cell, boundaries, _steps, _components, pml_cells in CASES:
        grid = Grid(resolution=12.0, cell_size=cell, dimensions=2, courant=0.35,
                    boundaries=boundaries)
        fields = Fields(grid=grid, force_complex_fields=False)
        fields.enable_pml_storage()
        depth = dict(pml_cells or {"x": 2, "y": 2})
        thickness = tuple(
            (depth.get("xyz"[axis], 0), depth.get("xyz"[axis], 0))
            if grid.shape[axis] >= 6 else (0, 0) for axis in range(3))
        pml = PML(grid=grid, thickness=thickness)
        for axis in "xyz":
            km = np.asarray(getattr(pml, f"kms_{axis}")).ravel()
            si = np.asarray(getattr(pml, f"sinv_{axis}")).ravel()
            exact_km, exact_si = km != 1.0, si != 1.0
            tolerance = (km < 0.999999) | (si < 0.999999)
            row = {
                "case": case_name, "axis": axis, "entries": int(km.size),
                "km_not_one": int(exact_km.sum()),
                "sinv_not_one": int(exact_si.sum()),
                "same_set": bool((exact_km == exact_si).all()),
                "or_selects": int((exact_km | exact_si).sum()),
                "and_selects": int((exact_km & exact_si).sum()),
                "max_km": float(km.max()), "any_km_above_one": bool((km > 1).any()),
                "tolerance_matches_exact": bool(
                    (tolerance == (exact_km | exact_si)).all()),
                "tolerance_disagrees_at": int(
                    (tolerance != (exact_km | exact_si)).sum()),
            }
            rows.append(row)
            if not row["same_set"]:
                findings.append(
                    f"{case_name}/{axis}: kms != 1 and sinv != 1 select DIFFERENT "
                    f"sets, so m_dsig_uses_or_instead_of_and is no longer a null and "
                    f"must be re-scored as a live mutation")
            if not row["tolerance_matches_exact"]:
                findings.append(
                    f"{case_name}/{axis}: the tolerance form disagrees with the exact "
                    f"one at {row['tolerance_disagrees_at']} entries, so "
                    f"m_dsig_becomes_a_tolerance is no longer a null")
            if row["any_km_above_one"]:
                findings.append(
                    f"{case_name}/{axis}: kms exceeds 1 (max {row['max_km']}), which "
                    f"a `< 0.999999` tolerance cannot see")
    # A leg that measured NOTHING would agree with itself. At least one axis must
    # actually be graded, or every equality above is between two empty sets.
    if not any(row["or_selects"] for row in rows):
        findings.append(
            "NO axis of any case grid is graded at all: every predicate above "
            "selects the empty set and the two nulls rest on nothing")
    return {"leg": "coefficient_predicates", "device": False, "axes": rows,
            "findings": findings, "passed": not findings}


def whole_volume_rescale_leg() -> Dict[str, Any]:
    """The RETIRED whole-volume passes' arithmetic, executed on the words it acts on.

    THE MEASUREMENT BEHIND THE REFUSAL THIS PRODUCT SHIPPED WITH — and, since the
    2026-09-01 lift, behind the two clauses that survive it. The driver used to
    snapshot the target D component, inject, and rescale BY DIFFERENCE over the
    whole volume; it now replays that composition per PUBLISHED deposit cell
    (identity everywhere else) and keeps the whole-volume form only as the dense
    fallback for a scaled source publishing no deposit table::

        array -= before[name]
        array *= condinv
        array += before[name]

    Run here on a lattice of signed zeros and finite values, so the artifact carries
    the arithmetic rather than a citation to it:

    * for every FINITE value the round trip is EXACT -- which is why the sparse
      replay is byte-identical AT the deposit cells and why the dense fallback is
      still correct physics;
    * for ``-0.0`` it is not: ``(-0.0) - (-0.0)`` is ``+0.0``, ``+0.0 * condinv`` is
      ``+0.0``, and ``+0.0 + (-0.0)`` is ``+0.0``. Every ``-0.0`` under the passes
      becomes ``+0.0`` — at every cell of the volume in the dense form, which is
      why the tableless source stays refused; at ONLY the deposit cells in the
      sparse form, where the array path makes the same rewrite and the routes agree.

    The device half of this story is the gate's ``lifted_refusal`` leg: the carried
    scaled-source case must be byte-identical, and the retired dense composition,
    replayed as the armed control, must still diverge.
    """
    findings: List[str] = []
    rng = np.random.default_rng(7)
    picks = rng.integers(0, 3, size=4096)
    values = np.where(picks == 0, np.float32(0.0),
                      np.where(picks == 1, np.float32(-0.0),
                               rng.uniform(-1, 1, size=4096))).astype(np.float32)
    condinv = np.float32(0.9235)
    before = values.copy()
    array = values.copy()
    array -= before
    array *= condinv
    array += before
    moved = np.flatnonzero(array.view(np.uint32) != before.view(np.uint32))
    negative_zero = np.flatnonzero(before.view(np.uint32) == np.uint32(0x80000000))
    finite = np.flatnonzero((before != 0.0))
    if not np.array_equal(np.sort(moved), np.sort(negative_zero)):
        findings.append(
            f"the rescale moved {moved.size} words and there are "
            f"{negative_zero.size} negative zeros: the two sets are not the same, so "
            f"this product's refusal names the wrong mechanism")
    if negative_zero.size == 0:
        findings.append("the fixture drew no negative zeros; this leg measured nothing")
    if not np.array_equal(array[finite].view(np.uint32),
                          before[finite].view(np.uint32)):
        findings.append(
            "the rescale changed a FINITE value, which would make this a defect in "
            "the driver rather than a signed-zero property")
    return {"leg": "whole_volume_rescale", "device": False,
            "words": int(before.size), "negative_zeros": int(negative_zero.size),
            "finite_values": int(finite.size),
            "words_the_rescale_moved": int(moved.size),
            "moved_set_is_exactly_the_negative_zeros": bool(
                np.array_equal(np.sort(moved), np.sort(negative_zero))),
            "where": "driver.py:3363-3370",
            "findings": findings, "passed": not findings}


NO_DEVICE_LEGS = (transcription_leg, equivalence_leg, corpus_admission_leg,
                  mutation_arming_leg, coefficient_predicates_leg,
                  whole_volume_rescale_leg)


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
            # convention, INCLUDING the four-case recurrence's `(c * cf) - curl`.
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
    # single-launch leg non-vacuous: without a nonzero H the D curl is exactly zero
    # on step 1.
    #
    # THE CONDUCTIVE HISTORY IS SEEDED, and that is not thoroughness. `f_cond_D` is
    # read by `c_new = ((c * cf) - curl) * ci` and by case A's `u_split` and case C's
    # `f_first`; at `c == 0` all three lose their leading term and every mutation of
    # the history load or its store becomes a null.
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                 "f_cond_Dx", "f_cond_Dy", "f_cond_Dz",
                 "f_cond_Bx", "f_cond_By", "f_cond_Bz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(draw())


def build_driver(cp, cell, boundaries, seed: int, value_class: str,
                 electric: bool, sigma: float = SIGMA_CORPUS,
                 components: Any = None, integrated: bool = True,
                 pml_cells: Any = None):
    """One 2-D CONDUCTIVE PML driver, seeded identically for every route.

    ``components`` is the conductivity map: ``None`` puts one volume on every D
    component, a tuple puts it on those only -- which is the mixed configuration the
    three ``COND`` constexprs exist for and the only one that measures the
    ``COND == 0`` arm beside a live ``COND == 1`` arm in the SAME launch.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=2,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
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
    # PER AXIS, NOT SCALAR. On a 2-D cell z is translationally invariant and the
    # driver refuses a layer there BY NAME (driver.py:2206).
    driver.setup_pml(dict(pml_cells or {"x": 2, "y": 2}))
    # THE CONDUCTIVITY, and it is INSTALLED AFTER THE LAYER deliberately: the
    # absorber's own graded sigma and a material sigma are the two independent
    # sources the engine keeps apart (driver.py:1010-1028), and a material
    # conductivity declared here is not `absorber_included`.
    if sigma is not None:
        # A VARYING sigma, for the reason the epsilon above varies: a constant makes
        # `condfac` and `condinv` uniform, and every one of the four cases'
        # coefficient reads becomes a scaling a wrong index reproduces.
        values = np.ascontiguousarray(
            (float(sigma) * (0.6 + 0.4 * np.sin(index * np.float32(0.023)))
             ).astype(np.float32))
        payload = ({name: cp.asarray(values) for name in components}
                   if components is not None else cp.asarray(values))
        driver.set_conductivity(payload)
    if electric:
        # THE DEPOSIT THIS PRODUCT EXISTS FOR. Injected BETWEEN the two halves
        # (driver.py:3294-3299), so the fused launch consumes a pre-injection D and
        # the shipped repair is what puts the difference back.
        #
        # INTEGRATED BY DEFAULT HERE, and that default is a MEASUREMENT rather than a
        # convenience: on a conductive run a NON-integrated source takes
        # driver.py:3363-3370's three whole-volume passes, which rewrite -0.0 to
        # +0.0 at every cell of the target component and which a point repair cannot
        # follow. The product refuses that case BY NAME and the refusal family below
        # measures the refusal; every carry leg therefore uses the case the product
        # actually claims. `integrated=False` remains reachable, for exactly that
        # refusal row.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4,
                           "is_integrated": bool(integrated)})
    else:
        # A MAGNETIC source is admitted with no repair at all: the driver injects it
        # in the B/H seam, not this one. Carrying one is what stops the source clause
        # from being tested only in its refusing direction.
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    seed_values(cp, driver, seed, value_class)
    # THE FAST PATH IS DISABLED, EXPLICITLY. `FdtdDriver.step` consults `fastpath`
    # BEFORE the module-level sub-step functions this harness substitutes, and the
    # `conductive PML` / `ordinary` arms are WIRED — so a driver left alone would
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
    history = required_history(fields)
    if not history:
        raise AssertionError(
            "no D component carries a conductivity on this driver, so the four-case "
            "recurrence this gate exists for never runs")
    missing = [name for name in tuple(REQUIRED) + history if name not in found]
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

    They are the arms ``launch.plan_step`` really selects on a conductive PML row
    today -- ``conductive PML`` on ``step_D`` and ``ordinary`` on ``update_E`` -- so
    this route is not a second model of the seam, it is the shipped one.

    The three in-seam passes stay on the ARRAY PATH here: no Triton product owns the
    wall clear, and both symmetry fills are no-ops without a mirror plane. All three
    are COUNTED. THE INJECTION ALSO STAYS ON THE ARRAY PATH, which is what makes
    this the right oracle for the carry family: the certified kernels with the
    driver's own deposit between them.
    """
    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import plan_constitutive  # noqa: PLC0415

    curl = conductivity.plan_conductive_pml_curl(driver.fields, driver.pml, "step_D")
    constitutive = plan_constitutive(driver.fields, driver.pml, "E")
    missing = [name for name, plan in
               (("conductive PML curl", curl),
                ("ordinary constitutive", constitutive)) if plan is None]
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
        ("h_condfac_and_condinv_exchanged",
         "condfac and condinv bound to each other's slots; both are per-cell "
         "volumes of the same shape and dtype, so nothing but the arithmetic can "
         "tell them apart",
         "caught", lambda driver, plan: _swap_conductive_slots(plan)),
        ("h_conductive_history_rotated",
         "f_cond_Dx/Dy/Dz rotated by one component; the history each recurrence "
         "reads and writes is then its neighbour's",
         "caught", lambda driver, plan: _rotate_slot(plan, "_history")),
        ("h_cond_flags_all_off",
         "COND0/1/2 forced to 0 on a grid whose components ARE lossy: the plan "
         "would step the plain recurrence and grow no f_cond history at all",
         "caught", lambda driver, plan: _clear_cond_flags(plan)),
    )


def _swap_conductive_slots(plan):
    plan._condfac, plan._condinv = plan._condinv, plan._condfac  # noqa: SLF001
    return plan


def _rotate_slot(plan, slot: str):
    values = list(getattr(plan, slot))
    setattr(plan, slot, tuple(values[1:] + values[:1]))
    return plan


def _clear_cond_flags(plan):
    plan.cond = (0, 0, 0)
    return plan


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, steps: int, product,
            value_class: str = "uniform", mutant: Any = None,
            host_mutation: Any = None, install_fused: bool = True,
            freeze_electric: bool = False, electric: bool = False,
            bracket: bool = True, components: Any = None,
            integrated: bool = True, pml_cells: Any = None) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    seed = case_seed(name, str(cell), str(boundaries), value_class, str(components),
                     str(pml_cells))
    made = [build_driver(cp, cell, boundaries, seed, value_class, electric,
                         components=components, integrated=integrated,
                         pml_cells=pml_cells)
            for _ in range(3)]
    reference, separate, fused = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.conductive_fused_curl_constitutive_D_kernel())
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
        plan = product.plan_conductive_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), kernel=kernel)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.conductive_fused_electric_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if host_mutation is not None:
            row["host_mutation"] = host_mutation[0]
            plan = host_mutation[3](fused, plan)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["zero_metal_flags"] = list(plan.zero_metal)
        row["cond_flags"] = list(plan.cond)
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
        # these functions and both arms are WIRED, so a missing count here means
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
# THE REDUCTION LEG — COND=(0,0,0) must BE the shipped ordinary kernel
# ---------------------------------------------------------------------------

def reduction_leg(cp, product, cell, boundaries, steps: int,
                  value_class: str = "uniform",
                  pml_cells: Any = None) -> Dict[str, Any]:
    """``COND = (0,0,0)`` reproduces ``kernels.fused_curl_constitutive_D`` bit for bit.

    The transcription claim, measured on silicon instead of in source text. Three
    drivers, seeded identically, on a live CONDUCTIVE run:

    * route A launches THIS product's kernel with all three ``COND`` flags at 0;
    * route B launches the SHIPPED ``kernels.fused_curl_constitutive_D`` over the
      same volumes, through ``launch.plan_fused_pair_from_arrays`` -- the bare-array
      route, because ``coverage.fused_pair_coverage`` refuses a conductive run by
      name and would return ``None``;
    * route C is the array path, and is the DISCRIMINATION precondition.

    NEITHER A NOR B IS COMPARED AGAINST THE ARRAY PATH FOR IDENTITY, deliberately:
    the array path really does take the four-case recurrence, so both routes must
    DIFFER from it. What this leg measures is that the two KERNELS agree, which is
    the reduction.

    THE FLOOR IS TWO-SIDED, and that shape is a 2026-08-31 correction paid for on a
    sibling gate. At ``COND == 0`` the conductive branch is not compiled at all, so
    ``f_cond_D*`` is never written -- BY DESIGN, and it is this leg's own claim. An
    absolute "every array moved" floor therefore asserts the NEGATION of what the leg
    measures and fails a correct kernel; the sibling's run c did exactly that. What
    replaces it is stronger everywhere else and pins the direction here:

    * the conductive history must be INERT on the reduced route -- that IS the
      reduction;
    * it must MOVE on the ARRAY PATH -- without that the two routes differ by a pair
      of zeros and the discrimination is worthless;
    * every OTHER array the array path moves must move here too -- a pass this weld
      swallowed is still a failure, which is what the absolute floor was for.

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
    made = [build_driver(cp, cell, boundaries, seed, value_class, electric=False,
                         pml_cells=pml_cells)
            for _ in range(3)]
    reference, shipped, reduced = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    row: Dict[str, Any] = {"leg": "reduction:cond_all_zero", "device": True,
                           "steps_budget": steps, "value_class": value_class,
                           "seed": seed, "shape": list(reference.shape),
                           "boundaries": boundaries}
    try:
        from meep_gpu.triton_kernels.conductivity import (  # noqa: PLC0415
            CONDUCTIVE_SUB_STEPS,
        )
        from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
        from meep_gpu.stepping import _boundary_kinds as resolve  # noqa: PLC0415

        reduced_plan = product.plan_conductive_fused_electric_pair(
            reduced.fields, reduced.pml, tuple(reduced._sources))
        if reduced_plan is None:
            raise AssertionError("the product refused the reduction case")
        # KEPT BEFORE IT IS CLEARED, in a LOCAL rather than on the plan: which
        # components this configuration makes lossy is what decides where the array
        # path's history can move, and the flags are about to be zeroed. The plan
        # carries `__slots__`, so an attribute stashed on it would be an
        # AttributeError rather than a record.
        cond_of_record = tuple(reduced_plan.cond)
        row["cond_of_record"] = list(cond_of_record)
        reduced_plan.cond = (0, 0, 0)

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
        shipped_opening = snapshot(cp, shipped)
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
                "cond_zero_vs_shipped_fused_pair": versus_shipped,
                "cond_zero_vs_array_path": versus_array,
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

        # The conductive history is this leg's LEVER, not its floor -- see the
        # docstring. The names come from the SUB-STEP TABLE, so renaming the volumes
        # without renaming them here turns the assertion red rather than vacuous.
        history = tuple("f_cond_" + name for name in
                        CONDUCTIVE_SUB_STEPS[product.CURL_SUB_STEP])
        row["conductive_history"] = list(history)

        # WHICH history volumes the ARRAY PATH can write on THIS configuration,
        # derived rather than assumed. `_conductive_component` stores `c_new` under
        # `live & dsig`, and `dsig` reads the component's FIRST coefficient axis --
        # target 0 takes y, target 1 takes z, target 2 takes x (vec.hpp's
        # cycle_direction, and the three call sites above). On a 2-D cell z is
        # invariant, so component 1's `dsig` is False everywhere and `f_cond_Dy` is
        # NEVER written; demanding it move was run b's failure, and demanding "at
        # least one" instead would have let a two-of-three defect through.
        first_axis = {0: "y", 1: "z", 2: "x"}
        expected_movers = set()
        for index, name in enumerate(CONDUCTIVE_SUB_STEPS[product.CURL_SUB_STEP]):
            if not cond_of_record[index]:
                continue
            axis = first_axis[index]
            km = np.asarray(cp.asnumpy(getattr(reduced.pml, f"kms_{axis}"))).ravel()
            si = np.asarray(cp.asnumpy(getattr(reduced.pml, f"sinv_{axis}"))).ravel()
            if bool(((km != 1.0) | (si != 1.0)).any()):
                expected_movers.add("f_cond_" + name)
        row["history_the_coefficients_allow"] = sorted(expected_movers)
        row["history_moved_here"] = sorted(set(ever_moved) & set(history))
        row["history_moved_on_the_array_path"] = sorted(
            set(reference_moved) & set(history))
        # SWALLOWED-PASS IS MEASURED AGAINST ROUTE B, NOT THE ARRAY PATH, for the
        # same reason the seam-output comparison below is: the reduced route runs a
        # DIFFERENT recurrence from the conductive array path by construction, so an
        # array moving on one and not the other is the reduction, not a swallowed
        # pass. Route B is the kernel this leg claims identity with, and it is the
        # only route whose movement the reduced route must match.
        shipped_moved = moved(shipped_opening, snapshot(cp, shipped))
        row["inert_here_but_moving_on_the_shipped_pair"] = sorted(
            (set(final) - set(ever_moved) - set(material)) & set(shipped_moved))
        row["seam_outputs_moved"] = sorted(set(ever_moved) & set(SEAM_OUTPUTS))
        row["reference_seam_outputs_moved"] = sorted(
            set(reference_moved) & set(SEAM_OUTPUTS))
        row["launches"] = dict(counter)
        failures: List[str] = []
        if row["first_divergence"] is not None:
            failures.append(
                f"COND=(0,0,0) is NOT the shipped fused pair: "
                f"{row['first_divergence']}")
        if not history:
            failures.append(
                "the sub-step table declares NO conductive history, so this leg has "
                "no lever and its reduction claim is untestable")
        if row["history_moved_here"]:
            failures.append(
                f"COND=(0,0,0) still wrote {row['history_moved_here']}: the branch "
                f"this flag removes was compiled anyway")
        # THE LEVER IS A PROPERTY OF THE VALUE CLASS, recorded rather than demanded
        # of each one. On the signed-zero lattice the array path's history can hold
        # its words -- every case-A expression on +-0 inputs lands on a zero of the
        # same sign -- so the reduced route's stillness is not a measurable
        # difference THERE. Measured on run d, where the class moved none of the two
        # volumes its coefficients allow. Demanding it per class would fail a correct
        # kernel for a property of the seeding; the gate-level floor below demands
        # that at least ONE class carry the lever, which is what makes the reduction
        # a measurement rather than a tautology.
        row["history_lever"] = bool(row["history_moved_on_the_array_path"])
        if not expected_movers:
            failures.append(
                "VACUOUS: no component of this configuration can write its "
                "conductive history at all, so the reduced route's stillness could "
                "not be a measurable difference on ANY value class")
        if (row["history_lever"]
                and set(row["history_moved_on_the_array_path"]) != set(expected_movers)):
            failures.append(
                f"the ARRAY PATH moved {row['history_moved_on_the_array_path']} of "
                f"the history, and this configuration's coefficients say it can move "
                f"only {sorted(expected_movers)}: the array path is writing a volume "
                f"its own dsig forbids")
        if not row["history_lever"]:
            row["no_lever_because"] = (
                "the ARRAY PATH moved none of the conductive history on this value "
                "class, so the reduced route's stillness is not a measurable "
                "difference here; the gate-level floor requires another class to "
                "carry it")
        if row["inert_here_but_moving_on_the_shipped_pair"]:
            failures.append(
                f"the reduced route left "
                f"{row['inert_here_but_moving_on_the_shipped_pair']} INERT while the "
                f"SHIPPED fused pair moves them: a pass this weld swallowed")
        if not row["reference_seam_outputs_moved"]:
            failures.append(
                "VACUOUS: the ARRAY PATH itself moves nothing this seam writes on "
                "this configuration")
        # THE MOVEMENT COMPARISON IS AGAINST ROUTE B, NOT THE ARRAY PATH, and that is
        # not a softening -- it is the only comparison that is TRUE. The reduced
        # route runs the ORDINARY recurrence, which stores `fu` unconditionally,
        # where the conductive array path stores it only under `dsigu`; so the
        # reduced route legitimately moves MORE than the array path, and route B --
        # the shipped `kernels.fused_curl_constitutive_D`, the kernel this leg claims
        # identity with -- is what it must match. Measured on run b, where the array
        # path moved `fu_Dy, fu_Dz` and the reduced route moved all three.
        row["shipped_seam_outputs_moved"] = sorted(
            set(shipped_moved) & set(SEAM_OUTPUTS))
        if set(row["seam_outputs_moved"]) != set(row["shipped_seam_outputs_moved"]):
            failures.append(
                f"the reduced route's seam-output movement "
                f"{row['seam_outputs_moved']} is not the SHIPPED fused pair's "
                f"{row['shipped_seam_outputs_moved']}")
        # THE DISCRIMINATION PRECONDITION, recorded rather than assumed. If
        # COND=(0,0,0) also matches the ARRAY PATH then this value class cannot tell
        # the two kernels apart at all and its agreement is a tautology. That is a
        # PROPERTY OF THE VALUE CLASS, not a defect, so it is reported as
        # `discriminating: false` with its reason and the gate-level floor demands
        # that at least one class discriminate.
        row["discriminating"] = row.get("divergence_from_the_array_path") is not None
        if not row["discriminating"]:
            row["not_discriminating_because"] = (
                "COND=(0,0,0) matched the ARRAY PATH too, so this value class cannot "
                "separate the conductive kernel from the ordinary one")
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
        "w", suffix="_mutated_conductive_electric_pair.py", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_conductive_electric_pair_" + str(len(_TEMPORARY)), handle.name)
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
    """(id, why it is armed, expectation, rewrite).

    THE FIRST ELEVEN REWRITE THE FOUR-CASE RECURRENCE, which is the only arithmetic
    decision this product makes that the ordinary fused pair does not. Each of the
    five branch expressions gets one, each of the two exact float comparisons gets
    one, and each of the two masked stores gets one -- because those masks are the
    "unchanged" column of the case table and a widened one is invisible in D and E
    on the step it happens.

    THE REST rewrite the wall clear and the constitutive half, and are the same
    rewrites the sibling electric welds carry, on the same lines.
    """

    def m_history_load_dropped(source: str) -> Tuple[str, int]:
        """``f_cond`` read as zero -- cases A and C lose their leading term."""
        return _replace_all(source, "c = tl.load(c_ptr + idx, mask=live, other=0.0)",
                            "c = tl.load(c_ptr + idx, mask=live & (idx < 0), "
                            "other=0.0)")

    def m_history_store_dropped(source: str) -> Tuple[str, int]:
        """``f_cond`` never written; the next step reads a stale history."""
        return _replace_all(source,
                            "tl.store(c_ptr + idx, c_new, mask=live & dsig)",
                            "tl.store(c_ptr + idx, c_new, mask=live & dsig & "
                            "(idx < 0))")

    def m_history_store_mask_widened(source: str) -> Tuple[str, int]:
        """``f_cond`` written EVERYWHERE, not only where ``dsig``.

        The four-case table's "unchanged" column: outside ``dsig`` the array path
        leaves ``f_cond`` at exactly its previous value, and a widened store puts
        ``c_new`` there. Invisible in D and E on the step it happens, which is why
        the comparison is over every allocated volume.
        """
        return _replace_all(source,
                            "tl.store(c_ptr + idx, c_new, mask=live & dsig)",
                            "tl.store(c_ptr + idx, c_new, mask=live)")

    def m_aux_store_mask_widened(source: str) -> Tuple[str, int]:
        """``fu_D`` written everywhere, not only where ``dsigu``."""
        return _replace_all(source,
                            "tl.store(u_ptr + idx, u_new, mask=live & dsigu)",
                            "tl.store(u_ptr + idx, u_new, mask=live)")

    def m_dsig_becomes_a_tolerance(source: str) -> Tuple[str, int]:
        """``km != 1.0`` softened to a tolerance.

        stepping.py:2055-2058 is an EXACT float comparison. A tolerance takes the
        graded cells nearest the layer's inner face down the WRONG case -- smooth,
        plausible, and a different recurrence.
        """
        return _replace_all(source, "dsig = (km1 != 1.0) | (si1 != 1.0)",
                            "dsig = (km1 < 0.999999) | (si1 < 0.999999)")

    def m_dsig_and_dsigu_swapped(source: str) -> Tuple[str, int]:
        """The two predicates exchanged; each case then tests the other axis."""
        return _rewrite_block(
            source,
            ["dsig = (km1 != 1.0) | (si1 != 1.0)",
             "dsigu = (km2 != 1.0) | (si2 != 1.0)"],
            ["dsig = (km2 != 1.0) | (si2 != 1.0)",
             "dsigu = (km1 != 1.0) | (si1 != 1.0)"])

    def m_dsig_uses_or_instead_of_and(source: str) -> Tuple[str, int]:
        """``|`` narrowed to ``&``: a cell whose km is graded but whose sinv is 1
        falls out of the split-field case the array path puts it in."""
        return _replace_all(source, "dsig = (km1 != 1.0) | (si1 != 1.0)",
                            "dsig = (km1 != 1.0) & (si1 != 1.0)")

    def m_case_a_regrouped(source: str) -> Tuple[str, int]:
        """Case A regrouped as ``(u*km1) + (c_new - c)``.

        A GENUINELY DIFFERENT float32 grouping, and the second spelling of this
        mutation. The first simply DROPPED the parentheses --
        ``u * km1 + c_new - c`` -- which Python parses as ``((u*km1) + c_new) - c``,
        the shipped expression exactly; it was measured UNCAUGHT on run b because it
        was not a mutation at all. Recorded rather than replaced silently: the same
        left-associativity makes ``c_new = ((c * cf) - curl) * ci`` and
        ``u_cond = ((u * cf) - curl) * ci`` un-flattenable too, so the curl's
        ``dtdx * ((c_y - c) + (b - b_z))`` is the only place in this kernel where
        dropping parentheses changes the arithmetic, and `m_curl_parens_flattened`
        is where that is armed.
        """
        return _replace_all(source, "u_split = (((u * km1) + c_new) - c) * si1",
                            "u_split = ((u * km1) + (c_new - c)) * si1")

    def m_case_c_takes_the_second_axis(source: str) -> Tuple[str, int]:
        """Case C built from ``(km2, si2)`` instead of its own ``(km1, si1)``."""
        return _replace_all(source, "f_first = (((f * km1) + c_new) - c) * si1",
                            "f_first = (((f * km2) + c_new) - c) * si2")

    def m_case_d_uses_the_history(source: str) -> Tuple[str, int]:
        """Case D -- the no-PML arm -- built from ``c`` instead of ``f``.

        Case D is the one the array path reaches where neither axis is graded; it
        reads the FIELD, not the history.
        """
        return _replace_all(source, "f_direct = ((f * cf) - curl) * ci",
                            "f_direct = ((c * cf) - curl) * ci")

    def m_case_selection_order_swapped(source: str) -> Tuple[str, int]:
        """``dsig`` tested before ``dsigu``; the nesting order IS the case table."""
        return _replace_all(
            source,
            "v = tl.where(dsigu, f_split, tl.where(dsig, f_first, f_direct))",
            "v = tl.where(dsig, f_first, tl.where(dsigu, f_split, f_direct))")

    def m_condfac_and_condinv_exchanged(source: str) -> Tuple[str, int]:
        """``(1 - sigma*dt/2)`` and ``1/(1 + sigma*dt/2)`` exchanged in case B."""
        return _replace_all(source, "u_cond = ((u * cf) - curl) * ci",
                            "u_cond = ((u * ci) - curl) * cf")

    def m_lossless_arm_takes_the_conductive_one(source: str) -> Tuple[str, int]:
        """``COND == 0`` routed through the conductive branch.

        On a MIXED grid this dereferences the placeholder pointer -- the component's
        own target, bound where a lossless component has no ``f_cond`` -- and grows a
        spurious history from it. The module docstrings call this the over-covering
        failure by name; here it is a measurement.
        """
        return _replace_all(source, "if COND == 0:", "if COND == 2:")

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
        the D volume alone -- which is exactly why the comparison is over EVERY
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
             "tl.store(f0 + idx, v0, mask=live)",
             "tl.store(f1 + idx, v1, mask=live)",
             "tl.store(f2 + idx, v2, mask=live)"],
            ["tl.store(f0 + idx, v0, mask=live)",
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
        """``f_w`` read AFTER it is overwritten -- the one ordering the constitutive
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
        """``dtdx * (c_y - c + b - b_z)`` -- a different float32 grouping."""
        return _replace_all(source, "curl0 = dtdx * ((c_y - c) + (b - b_z))",
                            "curl0 = dtdx * (c_y - c + b - b_z)")

    def m_d_store_dropped(source: str) -> Tuple[str, int]:
        """The stepped displacement never written back; the next step's curl reads
        it."""
        return _replace_all(source, "tl.store(f0 + idx, v0, mask=live)",
                            "tl.store(f0 + idx, v0, mask=live & (idx < 0))")

    def m_recurrence_component_rotated(source: str) -> Tuple[str, int]:
        """Component 0's recurrence handed component 1's pointer row.

        The three calls are the ONLY place this kernel decides which volume each
        component's history lives in, and the five pointers move together -- so a
        rotation is a defect no per-pointer check would see.
        """
        return _rewrite_block(
            source,
            ["v0 = _conductive_component_registers(f0, u0, c0, cf0, ci0, idx, live, curl0,",
             "km_y, si_y, km_z, si_z, COND0)"],
            ["v0 = _conductive_component_registers(f0, u1, c1, cf1, ci1, idx, live, curl0,",
             "                                     km_y, si_y, km_z, si_z, COND0)"])

    return (
        ("m_history_load_dropped", "f_cond read as zero", "caught",
         m_history_load_dropped),
        ("m_history_store_dropped", "f_cond never written back", "caught",
         m_history_store_dropped),
        ("m_history_store_mask_widened",
         "f_cond written outside dsig, where the array path leaves it alone",
         "caught", m_history_store_mask_widened),
        ("m_aux_store_mask_widened",
         "fu_D written outside dsigu, where the array path leaves it alone",
         "caught", m_aux_store_mask_widened),
        ("m_dsig_becomes_a_tolerance",
         "an EXACT float comparison softened to a tolerance", "null",
         m_dsig_becomes_a_tolerance),
        ("m_dsig_and_dsigu_swapped", "the two case predicates exchanged", "caught",
         m_dsig_and_dsigu_swapped),
        ("m_dsig_uses_or_instead_of_and",
         "the case predicate narrowed from OR to AND", "null",
         m_dsig_uses_or_instead_of_and),
        ("m_case_a_regrouped", "a different float32 grouping in case A", "caught",
         m_case_a_regrouped),
        ("m_case_c_takes_the_second_axis",
         "case C built from the second axis's coefficients", "caught",
         m_case_c_takes_the_second_axis),
        ("m_case_d_uses_the_history", "case D built from f_cond instead of D",
         "caught", m_case_d_uses_the_history),
        ("m_case_selection_order_swapped",
         "the tl.where nesting order, which IS the case table", "caught",
         m_case_selection_order_swapped),
        ("m_condfac_and_condinv_exchanged",
         "condfac and condinv exchanged in case B", "caught",
         m_condfac_and_condinv_exchanged),
        ("m_lossless_arm_takes_the_conductive_one",
         "COND == 0 routed through the conductive branch, dereferencing the "
         "placeholder pointer", "caught", m_lossless_arm_takes_the_conductive_one),
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
        ("m_recurrence_component_rotated",
         "component 0's recurrence handed component 1's pointer row", "caught",
         m_recurrence_component_rotated),
    )


def case_a_cells(cell, boundaries, pml_cells, components) -> int:
    """How many cells of this case reach CASE A's store, counted from the coefficients.

    ``u_split`` is selected where ``dsig`` and stored where ``dsigu``, so the cells
    that can carry it are the INTERSECTION of the two graded bands -- per target, on
    that target's own pair of axes (0 takes y then z, 1 takes z then x, 2 takes x then
    y; vec.hpp's cycle_direction and the kernel's three call sites). On a 2-D cell the
    invariant axis is graded nowhere, so two of the three targets contribute zero and
    the count is target 2's alone.

    NO DEVICE AND NO TRITON: the coefficients are the ``PML`` object's own arrays.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=12.0, cell_size=cell, dimensions=2, courant=0.35,
                boundaries=boundaries)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    depth = dict(pml_cells or {"x": 2, "y": 2})
    thickness = tuple((depth.get("xyz"[axis], 0), depth.get("xyz"[axis], 0))
                      if grid.shape[axis] >= 6 else (0, 0) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)

    def graded(axis: str) -> np.ndarray:
        km = np.asarray(getattr(pml, f"kms_{axis}")).ravel()
        si = np.asarray(getattr(pml, f"sinv_{axis}")).ravel()
        return (km != 1.0) | (si != 1.0)

    live = ("Dx", "Dy", "Dz") if components is None else tuple(components)
    first, second = {0: "y", 1: "z", 2: "x"}, {0: "z", 1: "x", 2: "y"}
    total = 0
    for index, target in enumerate(("Dx", "Dy", "Dz")):
        if target not in live:
            continue
        total += int(graded(first[index]).sum()) * int(graded(second[index]).sum())
    return total


def mutation_case_for(name: str) -> Tuple[str, Tuple[Any, ...]]:
    """The case a mutation is scored on, and it must really carry what it needs."""
    case_name = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    case = CASES_BY_NAME[case_name]
    _name, cell, boundaries, _steps, components, pml_cells = case

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
    # THE SAME TRAP, ONE RUNG DOWN. A rewrite inside the `COND == 1` arm scored on a
    # case whose components are lossless touches a branch the launch never compiles.
    live = ("Dx", "Dy", "Dz") if components is None else tuple(components)
    for component in MUTATION_REQUIRES_CONDUCTIVE.get(name, ()):
        if component not in live:
            raise AssertionError(
                f"mutation {name} rewrites the conductive arm for {component}, and "
                f"case {case_name!r} makes it lossless: the rewritten lines would be "
                f"a DEAD BRANCH and the leg would report 'uncaught' while measuring "
                f"nothing")
    floor = MUTATION_REQUIRES_CASE_A_CELLS.get(name)
    if floor is not None:
        cells = case_a_cells(cell, boundaries, pml_cells, components)
        if cells < floor:
            raise AssertionError(
                f"mutation {name} rewrites a line reached only in CASE A, and case "
                f"{case_name!r} gives it {cells} cells against a floor of {floor}: "
                f"an UNCAUGHT verdict there would be a statement about the case, not "
                f"about the kernel")
    for component in MUTATION_REQUIRES_LOSSLESS.get(name, ()):
        if component in live:
            raise AssertionError(
                f"mutation {name} rewrites the COND == 0 arm for {component}, and "
                f"case {case_name!r} makes it LOSSY: that arm is not compiled there, "
                f"so the rewritten lines would be a DEAD BRANCH and the leg would "
                f"report 'uncaught' while measuring nothing")
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
        kernel_name = f"mutant_{index}_conductive_electric_D"
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
                      product, value_class="uniform", mutant=mutant, electric=True,
                      components=case[4], pml_cells=case[5])
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
                      electric=True, components=case[4], pml_cells=case[5])
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
    _name, cell, boundaries, _steps, _components, pml_cells = CASES_BY_NAME["wall_xy"]
    for name, sigma, components, sources_of, needle, admit in (
        ("integrated_electric_source_with_a_deposit_index", SIGMA_CORPUS, None,
         lambda d: (_deposit(d, "Ez", integrated=True),), None, True),
        ("magnetic_source_only", SIGMA_CORPUS, None,
         lambda d: (_deposit(d, "Hz"),), None, True),
        ("mixed_conductivity_is_admitted", SIGMA_CORPUS, ("Dz",),
         lambda d: (_deposit(d, "Ez", integrated=True),), None, True),
        # THE CORPUS ROW'S OWN SOURCE — ADMITTED since the 2026-09-01 lift: not
        # integrated, but its deposit table is published and the driver replays the
        # condinv rescale sparsely at exactly those cells.
        ("non_integrated_electric_source_on_a_conductive_run_is_admitted",
         SIGMA_CORPUS, None,
         lambda d: (_deposit(d, "Ez", integrated=False),), None, True),
        # WHAT SURVIVES THE LIFT: a scaled source with NO deposit table takes the
        # driver's dense whole-volume fallback and stays refused by name.
        ("scaled_source_publishing_no_deposit_table", SIGMA_CORPUS, None,
         lambda d: (_ScaledElectricNoTable(),),
         "publishes NO deposit table", False),
        ("electric_source_without_a_deposit_index", SIGMA_CORPUS, None,
         lambda d: (_ElectricWithoutIndex(),), "does not publish the index", False),
        ("undeclared_source_list", SIGMA_CORPUS, None, lambda d: None,
         "was not declared", False),
        ("lossless_grid", None, None,
         lambda d: (_deposit(d, "Ez", integrated=True),),
         "no D component carries a conductivity", False),
    ):
        driver = build_driver(cp, cell, boundaries, 11, "uniform", electric=False,
                              sigma=sigma, components=components,
                              pml_cells=pml_cells)
        try:
            sources = sources_of(driver)
            verdict = product.conductive_fused_electric_pair_coverage(
                driver.fields, driver.pml, sources)
            built = product.plan_conductive_fused_electric_pair(
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


def _deposit(driver, component: str, integrated: bool = False):
    """A REAL source that publishes the index the injection writes.

    ``integrated`` is the flag the conductive seam turns on: driver.py:3355-3356
    injects an integrated source point-wise, where :3363-3370 rescales the whole
    volume for a scaled one. IT RIDES ON THE ENVELOPE, not on the source
    (sources.py:1443, :1467, :1498, :1548), and ``VolumeSource.is_integrated`` reads
    through to it -- constructing the source with the keyword is a TypeError, which
    is what run c measured.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    source = VolumeSource(grid=driver.fields.grid, component=component,
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2,
                                                    is_integrated=bool(integrated)),
                          amplitude=1.0)
    if bool(getattr(source, "is_integrated", False)) is not bool(integrated):
        raise AssertionError(
            f"the fixture asked for is_integrated={integrated!r} and the source "
            f"reports {getattr(source, 'is_integrated', None)!r}: the flag did not "
            f"reach the object the predicate reads")
    if not source._n_source_points:
        raise AssertionError("the fixture source deposits nothing")
    if deposit_repair._deposit_index(source) is None:
        raise AssertionError("the fixture source publishes no deposit index")
    return source


# ---------------------------------------------------------------------------
# The lifted refusal — measured in both directions, not asserted
# ---------------------------------------------------------------------------

def _retired_whole_volume_inject(self, electric, source_time):
    """The RETIRED composition of ``_inject_electric_through_conductivity``, verbatim.

    The whole-volume difference passes the driver replaced on 2026-09-01: snapshot
    the target component, inject, then ``array -= before; array *= condinv;
    array += before`` over the WHOLE volume. Kept HERE, in the gate that measures
    the lift, as the armed control's body — the composition whose ``-0.0``
    canonicalisation (and, on device, whose subnormal flushing) the retired refusal
    priced at 14 Ez + 74 f_w_Ez words on this product's own wall_xy grid. The
    control MUST reproduce a divergence or the lift measured nothing. Integrated
    sources are injected point-wise first, exactly as both driver generations do.
    """
    integrated = [source for source in electric if source.is_integrated]
    scaled = [source for source in electric if not source.is_integrated]
    for source in integrated:
        source.inject(self.fields, source_time)
    if not scaled:
        return
    by_target: Dict[str, list] = {}
    for source in scaled:
        by_target.setdefault("D" + source.component[1], []).append(source)
    before = {}
    for name in sorted(by_target):
        if self.fields.condinv_for(name) is None:
            continue
        before[name] = getattr(self.fields, name).copy()
    for source in scaled:
        source.inject(self.fields, source_time)
    for name, prev in before.items():
        array = getattr(self.fields, name)
        array -= prev
        array *= self.fields.condinv_for(name)
        array += prev


def _retired_passes_control(cp) -> Dict[str, Any]:
    """Replay the retired whole-volume composition beside today's array path.

    TWO ARRAY-PATH DRIVERS, identical except for the injection composition: no
    fused plan on either, so what diverges is the driver change ALONE — the exact
    delta the clause was lifted on. The divergence's word classes are recorded
    (signed-zero pairs, and on device the subnormals the whole-volume passes
    flush), so the artifact says WHAT the retired composition destroyed.
    """
    import types  # noqa: PLC0415

    _name, cell, boundaries, _steps, components, pml_cells = CASES_BY_NAME["wall_xy"]
    seed = case_seed("lifted_refusal_control", str(cell))
    made = [build_driver(cp, cell, boundaries, seed, "signed_zero_lattice",
                         electric=True, components=components,
                         integrated=False, pml_cells=pml_cells)
            for _ in range(2)]
    reference, retired = made
    row: Dict[str, Any] = {"leg": "retired_control", "device": True, "armed": True}
    try:
        retired._inject_electric_through_conductivity = types.MethodType(
            _retired_whole_volume_inject, retired)
        row["per_step"] = []
        for step in range(1, CARRY_STEPS + 1):
            reference.step()
            retired.step()
            cp.cuda.runtime.deviceSynchronize()
            divergence = first_divergence(snapshot(cp, retired),
                                          snapshot(cp, reference))
            row["per_step"].append({"step": step, "retired_vs_array": divergence})
            if divergence is not None and "diverged_at_step" not in row:
                row["diverged_at_step"] = step
                row["divergence"] = divergence
                row["word_classes"] = _divergence_word_classes(cp, retired,
                                                               reference)
        row["diverges_at_some_step"] = any(
            entry["retired_vs_array"] is not None for entry in row["per_step"])
        return row
    except Exception as exc:  # noqa: BLE001
        row["error"] = repr(exc)
        return row
    finally:
        for target in made:
            try:
                target.close()
            except Exception:  # noqa: BLE001
                pass
        cp.get_default_memory_pool().free_all_blocks()


def _divergence_word_classes(cp, retired, reference) -> Dict[str, Any]:
    """Classify every differing word pair across the two drivers' stored volumes.

    The retired passes' two known casualties are the SIGNED ZERO (``x - x`` is
    ``+0.0``) and, on device arithmetic that flushes, the SUBNORMAL. Anything else
    is recorded as ``other`` — a finding about the retired composition, not a
    pass/fail input: the control's job is to diverge.
    """
    classes = {"signed_zero_pair": 0, "subnormal_flush": 0, "other": 0}
    left_state = snapshot(cp, retired)
    right_state = snapshot(cp, reference)
    for name, left in left_state.items():
        right = right_state.get(name)
        if right is None or left.shape != right.shape:
            continue
        differ = np.flatnonzero(left != right)
        for index in differ[:4096]:
            lw = int(left.reshape(-1)[index])
            rw = int(right.reshape(-1)[index])
            if {lw, rw} <= {0x00000000, 0x80000000}:
                classes["signed_zero_pair"] += 1
            elif (lw & 0x7F800000) == 0 or (rw & 0x7F800000) == 0:
                classes["subnormal_flush"] += 1
            else:
                classes["other"] += 1
    return classes


def lifted_refusal_leg(cp, product) -> Dict[str, Any]:
    """The FORMERLY refused configuration, carried — with the retired passes armed.

    THE RECORDED THREE-ROUTE SIGNED-ZERO PROTOCOL, re-run against the lifted
    clause. The 2026-08-31 measurement (the GPU host GPU 6, this product's own wall_xy
    grid) found the bracketed route's Ez differing in 14 words and f_w_Ez in 74 —
    every one a ``-0.0``/``+0.0`` pair — under the driver's then whole-volume
    rescale; the clause refused the cell's one corpus row on it. The driver now
    replays the rescale sparsely at the published deposit cells, and the burden of
    proof inverts:

    1. the lifted case (non-integrated electric deposit, conductive PML) must be
       ADMITTED by the shipped predicate and BYTE-IDENTICAL to the array path at
       every step, through the full carry protocol;
    2. the INTEGRATED companion of the recorded protocol must stay byte-identical
       exactly as it always was;
    3. the RETIRED whole-volume composition, replayed verbatim on a third driver
       against today's array path, MUST still diverge on the same signed-zero
       seed — the control separating "the driver change made the routes agree"
       from "this seed stopped discriminating".
    """
    row: Dict[str, Any] = {"leg": "lifted_refusal:non_integrated_conductive",
                           "device": True, "armed": True, "legs": {}}
    _name, cell, boundaries, _steps, components, pml_cells = CASES_BY_NAME["wall_xy"]
    try:
        carried = run_leg(cp, "lifted:non_integrated_conductive", cell, boundaries,
                          CARRY_STEPS, product, value_class="signed_zero_lattice",
                          electric=True, integrated=False, components=components,
                          pml_cells=pml_cells)
        passed, failures = verdict_of(carried, require_launches=CARRY_STEPS,
                                      require_repairs=True)
        carried["passed"], carried["failures"] = passed, failures
        row["legs"]["lifted_non_integrated"] = carried

        integrated = run_leg(cp, "lifted:integrated_companion", cell, boundaries,
                             CARRY_STEPS, product, value_class="signed_zero_lattice",
                             electric=True, integrated=True, components=components,
                             pml_cells=pml_cells)
        passed, failures = verdict_of(integrated, require_launches=CARRY_STEPS,
                                      require_repairs=True)
        integrated["passed"], integrated["failures"] = passed, failures
        row["legs"]["integrated_companion"] = integrated

        row["legs"]["retired_control"] = _retired_passes_control(cp)

        failures = []
        for name in ("lifted_non_integrated", "integrated_companion"):
            if not row["legs"][name].get("passed"):
                failures.append(f"{name}: {row['legs'][name].get('failures')}")
        control = row["legs"]["retired_control"]
        if not control.get("diverges_at_some_step"):
            failures.append(
                "the RETIRED whole-volume composition did NOT diverge from the "
                "sparse array path on any step: the control is disarmed and the "
                "lift's byte-identity requirement was vacuous on this seed")
        if control.get("error"):
            failures.append(f"retired_control: {control['error']}")
        row["failures"] = failures
        row["passed"] = not failures
        log(f"  lifted refusal: carried={row['legs']['lifted_non_integrated'].get('passed')} "
            f"integrated={row['legs']['integrated_companion'].get('passed')} "
            f"retired_control_diverges={control.get('diverges_at_some_step')} "
            f"at step {control.get('diverged_at_step')}")
        return row
    except Exception as exc:  # noqa: BLE001 - a raised leg is a recorded leg
        row["error"] = repr(exc)
        row["passed"] = False
        return row


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
            properties = cp.cuda.runtime.getDeviceProperties(device.id)
            # THE DOTTED SPELLING, from the same properties the device name comes
            # from. `Device().compute_capability` is CuPy's undotted "86"; the weld
            # contract tests a SUBSTRING of the seeded host line against
            # `validated_compute_capabilities` (["8.6"]), so the undotted form seeds
            # a weld that reports "names no validated cc" for a run that was in fact
            # on a validated architecture. One reader, one spelling.
            out["compute_capability"] = f"{properties['major']}.{properties['minor']}"
            out["device"] = properties["name"].decode()
        except Exception as exc:  # noqa: BLE001
            out["cupy"] = f"unavailable: {exc!r}"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_conductive_fused_electric_pair"),
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

    payload: Dict[str, Any] = {
        "gate": "triton_conductive_fused_electric_pair",
        "product": PRODUCT_MODULE,
        "kernel": KERNEL_NAME,
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": environment(),
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
    # THE MACHINE AND THE DEVICE, so this artifact can be SEEDED into a weld. The
    # weld contract binds a host string to the record's validated Triton and
    # capability lists; `environment()` above records the device but no hostname, and
    # a seeding tool that supplied the missing half would be typing a fact instead of
    # reading one. Resolved by name off the parity directory, as gate_provenance is.
    import triton_device_identity  # noqa: PLC0415

    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "quiet_cases": {case[0]: case[3] for case in CASES},
        "conductive_components": {case[0]: (list(case[4]) if case[4] else "all")
                                  for case in CASES},
        "pml_cells": {case[0]: case[5] for case in CASES},
        "carry_cases": list(CARRY_CASES),
        "steps_per_carry": CARRY_STEPS,
        "steps_per_mutation": MUTATION_STEPS,
        "value_classes": list(VALUE_CLASSES),
    }
    save(payload, artifact)

    log("\n=== device legs: the QUIET family (a magnetic source, no seam deposit) ===")
    for name, cell, boundaries, steps, components, pml_cells in CASES:
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"quiet:{name}", cell, boundaries, steps, product,
                          value_class=value_class, components=components,
                          pml_cells=pml_cells)
            passed, failures = verdict_of(row, require_launches=steps)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== device legs: the CARRY family (a real electric deposit in the seam) ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, _steps, components, pml_cells = CASES_BY_NAME[case_name]
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"carry:{name}", cell, boundaries, CARRY_STEPS, product,
                          value_class=value_class, electric=True,
                          components=components, pml_cells=pml_cells)
            passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                          require_repairs=True)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, _steps, components, pml_cells = CASES_BY_NAME[case_name]
        row = run_leg(cp, f"null_control:{name}", cell, boundaries, CARRY_STEPS,
                      product, electric=True, bracket=False, components=components,
                      pml_cells=pml_cells)
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

    log("\n=== the REDUCTION leg: COND=(0,0,0) is the shipped ordinary kernel ===")
    reductions = []
    for value_class in VALUE_CLASSES:
        name, cell, boundaries, _steps, _components, pml_cells = CASES_BY_NAME["wall_xy"]
        row = reduction_leg(cp, product, cell, boundaries, 4, value_class,
                            pml_cells=pml_cells)
        reductions.append(row)
        payload["device_legs"].append(row)
        save(payload, artifact)
    if not any(row.get("discriminating") for row in reductions):
        payload["device_legs"].append({
            "leg": "reduction:discrimination_floor", "device": True,
            "passed": False,
            "failures": ["NO value class separated COND=(0,0,0) from the array "
                         "path, so every reduction agreement above is a tautology"],
        })
        save(payload, artifact)
    # THE SECOND GATE-LEVEL FLOOR, and it is the other half of the same statement:
    # at least one value class must have found the array path WRITING the conductive
    # history, or the reduced route's stillness is a difference between two volumes
    # nothing ever touched.
    if not any(row.get("history_lever") for row in reductions):
        payload["device_legs"].append({
            "leg": "reduction:history_lever_floor", "device": True,
            "passed": False,
            "failures": ["NO value class moved the conductive history on the ARRAY "
                         "PATH, so `COND=(0,0,0) writes no history` is a statement "
                         "about volumes nothing wrote either way"],
        })
        save(payload, artifact)

    log("\n=== armed harness mutations ===")
    (name, cell, boundaries, _steps, components,
     pml_cells) = CASES_BY_NAME[DEFAULT_MUTATION_CASE]
    row = run_leg(cp, "armed:no_substitution", cell, boundaries, 3, product,
                  install_fused=False, electric=True, components=components,
                  pml_cells=pml_cells)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, artifact)

    row = run_leg(cp, "armed:frozen_electric_seam", cell, boundaries, 3, product,
                  freeze_electric=True, electric=True, components=components,
                  pml_cells=pml_cells)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's electric seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.conductive_fused_curl_constitutive_D)
    payload["mutations"] = run_mutations(cp, product, pristine)
    for _row in payload["mutations"]:
        if _row["expectation"] != "caught":
            _row["evidence"] = MUTATION_EVIDENCE.get(_row["mutation"])
    save(payload, artifact)

    log("\n=== armed host mutations ===")
    payload["host_mutations"] = run_host_mutations(cp, product)
    save(payload, artifact)

    log("\n=== the LIFTED REFUSAL: the recorded protocol + the retired-passes control ===")
    payload["lifted_refusal"] = lifted_refusal_leg(cp, product)
    save(payload, artifact)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, artifact)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])
    lifted_ok = bool(payload["lifted_refusal"].get("passed"))

    #: The only three verdicts a mutation may declare, and what each costs to use.
    #: `caught` needs no justification; `null` and `unreached` each need an entry in
    #: MUTATION_EVIDENCE, so a rewrite cannot be excused by relabelling it.
    VOCABULARY = ("caught", "null", "unreached")

    def _mutation_ok(row: Dict[str, Any]) -> bool:
        if row.get("error") is not None:
            return False
        if not row.get("rewrite_hits", 1):
            return False
        expectation = row["expectation"]
        if expectation not in VOCABULARY:
            return False
        if expectation != "caught" and not MUTATION_EVIDENCE.get(row["mutation"]):
            return False
        return bool(row.get("caught")) is (expectation == "caught")

    mutation_ok = all(_mutation_ok(row) for row in payload["mutations"])
    host_ok = all(bool(row.get("caught")) is (row["expectation"] == "caught")
                  for row in payload["host_mutations"])
    payload["passed"] = bool(
        device_ok and refusal_ok and lifted_ok and mutation_ok and host_ok
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
                     f"the lifted refusal ok: {lifted_ok}",
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
