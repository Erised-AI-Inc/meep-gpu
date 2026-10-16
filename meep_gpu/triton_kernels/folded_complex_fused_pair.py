"""The FOLDED COMPLEX electric seam in one Triton launch: ``step_D`` into ``update_E``.

THE D-SIDE TWIN of :mod:`.folded_complex_fused_magnetic_pair`, and the mirror image of
what :mod:`.folded_fused_pair` is to :mod:`.folded_fused_magnetic_pair` on the real
board. It claims the cell ``(folded complex PML, folded complex)`` at ``D->E``, which
no shipped Triton weld can stand in for: :mod:`.folded_fused_pair` refuses complex
storage (its fold parity is a compile-time ``+-1.0`` multiply, exact in float32 and
NOT exact as a complex64 product), and :mod:`.complex_fused_electric_pair` refuses a
mirror plane BY NAME through both of its halves.

THE SIBLING IS :mod:`..metal_kernels.folded_complex_fused_pair`, which serves the same
cell on Metal. This module is that product's Triton twin: the same three certified
halves, the same ownership restructure, the same ordered parity chain with the wall
clear inside it. Where the two backends differ is only in how the arithmetic is
spelled — MSL there, ``triton.jit`` here — and each region below is transcribed from
ITS OWN backend's certified body, never from the sibling's.

===========================================================================
WHAT IT IS WORTH: +2 SEAM-INSTANCES OF 387, AND THE FLAG THAT BUYS THEM
===========================================================================

MEASURED, from the board this product was written against —
``parity/meep_gpu/results/fusion_matrix_triton_2026-08-31_folded_disp/`` (305/387
served). Exactly two reachable D->E seam-instances sit in a cell no shipped Triton
product occupies and that this one does::

    curl arm              constitutive arm      rows  reachable  ceiling
    folded complex PML    folded complex          2        2        2

and they are::

    tests:TestEigCoeffs.test_binary_grating_special_kz_2_21_2   fill_D + fill_folded_far_ghosts_D
    tests:TestModeDecomposition.test_triangular_lattice_oblique  fill_D + fill_folded_far_ghosts_D

BOTH ROWS DECLARE AN ELECTRIC SOURCE (``source_field_types == ['D']``), so under the
source clause with :data:`CARRIES_DEPOSIT_REPAIR` at False this arm would admit
NOTHING AT ALL — the same polarity that put :mod:`.complex_fused_electric_pair` at one
row of sixteen before its flag flipped, and the exact reverse of the MAGNETIC twin's
situation, where the same two rows are electric-only and the clause costs it nothing.
The ceiling of 2 is the count with the deposit repair carried, and this module
declares the flag in the same change as the wiring that brackets its launch. The
Metal sibling declares it True for the same two rows, which is the independent check.

BOTH ROWS NEED THE FAR CARRY, so this family carries it from the first cut rather
than shipping a refusal it would have to retire.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

``driver.step`` runs five passes between the two halves (driver.py:3292-3304)::

    step_D -> ELECTRIC SOURCES -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — CARRIED, through the SHIPPED deposit repair.** The
  injection lands between the two halves, so the fused launch consumes a
  pre-injection ``D``; :class:`.deposit_repair.LeadingRepairPlan` saves the two
  stateful arrays at the deposit points (and, on a folded seam, at every cell the
  two fills image them into — :func:`.deposit_repair.repair_cells`) before the
  launch, and :class:`.deposit_repair.TrailingRepairPlan` recomputes them after, in
  the driver's own order. IGNORANCE IS STILL NEVER AN EMPTY SET: ``Fields`` does not
  hold the source list, so an undeclared ``sources`` is a REFUSAL, and a source that
  cannot publish the index it writes is refused by name;
* **``fill_symmetry_bc_D`` — CARRIED INLINE, and it is NOT dead.** It writes
  ``cell 0 = (+phase) (x) cell 2`` on every folded axis for every component whose Yee
  shift there is 0; ``update_E`` then reads exactly those cells;
* **``zero_metal_D`` — CARRIED INLINE** (driver.py:3301; ``stepping._zero_metal``
  :2206-2247), through :func:`.coverage.zero_metal_axes`, which is IMPORTED rather
  than re-spelled. It writes complex zero — ``array[face] = 0`` on a complex64
  volume — so the register clear is applied to BOTH word planes, and it is SIX rows
  on this family against the magnetic twin's three;
* **``fill_folded_far_ghosts_D`` — CARRIED INLINE.** It runs only on a folded
  PERIODIC axis (``stepping._stored_past_owned``:1454-1469) and images the LAST
  stored slot from a RUNTIME reflect row. Carrying it brings the folded complex
  curl's top-plane mask with it (``folded_complex.folded_bloch_pml_curl_step``'s
  DELTA 3, its ``BACKWARD == 1`` arm), which is transcribed below.

So :data:`REPLACES` names FIVE passes, and every one of them does work on both
corpus rows.

===========================================================================
THE FILL GEOMETRY IS THE D GEOMETRY, AND IT IS THE EXACT INVERSE OF THE B TWIN'S
===========================================================================

``fields.IYEE_SHIFTS`` (fields.py:214-219) gives ``Dx (1,0,0)``, ``Dy (0,1,0)``,
``Dz (0,0,1)`` — a D component's Yee shift is ONE on its own axis and ZERO on the
other two. The near fill touches component ``m`` on axis ``a`` exactly when
``iyee[m][a] == 0``, so on the D half that is the two axes that are NOT its own (up to
TWO destinations, and a corner where both fire); the FAR fill is the exact complement
and images along the component's OWN axis only (at most ONE). Both sets are DERIVED
from :data:`.folded_complex.TARGET_IYEE` at import rather than written out, and pinned
by test against ``fields.IYEE_SHIFTS``. They are the same two tables
:mod:`.folded_fused_pair` derives, and the same D-side answers the Metal sibling
imports rather than re-deriving.

Two consequences, inherited from the real folded electric pair and re-derived here
from the same table rather than assumed:

1. **THE COEFFICIENT INDEX MOVES FOR THE FAR HALF ONLY** — the exact reverse of the
   magnetic twin. ``update_E`` indexes ``kps``/``kms`` on the component's own axis
   (``stepping.E_CONSTITUTIVE_TERMS`` :227), which is exactly the axis the NEAR fill
   never images along and exactly the one the FAR fill does. So a near destination
   sits at the SAME coefficient index as the lane that owns it and reuses that lane's
   pair — a reload there would reload the identical word — while a far destination
   reads ``kps``/``kms`` at the TOP row rather than at the reflect row.
2. **THE WALL CLEAR AND THE FILLS CANNOT MEET.** ``_zero_metal`` clears component
   ``m`` at stored cell 0 of axis ``a`` when ``iyee[m][a] == 0`` — the SAME shift-0
   axes the near fill images — and it SKIPS a folded axis (:2237-2239), while the
   near fill runs only on a folded one. The predicate CHECKS that disjointness rather
   than inferring it.

**THE WALL CLEAR SITS INSIDE THE PARITY CHAIN HERE, WHICH IT NEVER DOES ON THE B
SIDE.** The B family's near destination is stored 0 of a folded axis and its far
destinations are axes ``_zero_metal`` never touches, so no ghost there takes a clear
line at all. On the D side the clear's rows ARE the near fill's axes, and the driver
clears BETWEEN the two fills (:3300 near, :3301 clear, :3302 far). So a destination
that carries near passes takes the clear AFTER them and any far parity AFTER the
clear. :func:`parity_chain` returns the ordered parity applications and
:data:`CLEAR_AFTER_NEAR` states where the clear lands; the kernel emits both in that
order.

MEASURED, NOT ARGUED. ``parity/meep_gpu/measure_d_side_composition_law.py`` scores
this composition rule and four transcription errors against the array path's own three
passes over 156 grid declarations: this rule 156/156, the four alternatives caught on
84, 36, 36 and 70 cases, with no vacuous case. Reversing either order, or flattening
the composite, is one of those four.

===========================================================================
THE OWNERSHIP RESTRUCTURE — one launch, no barrier, no race
===========================================================================

The fills read cells THIS LAUNCH WRITES. Triton has no device-wide barrier, so
ownership moves, exactly as it does on the real folded electric pair:

* a lane on ANY destination plane — stored 0 of each near axis, the top plane of the
  far one — computes the curl and the split-field ``n`` and stores ``fu_D`` (the array
  path's ``step_D`` writes ``fu`` at every cell and neither fill touches it,
  stepping.py:1497-1498), and then STOPS: it never loads ``D``, never stores ``D``,
  and never touches ``E`` or ``f_w_E``. Its ``D`` load is MASKED OFF rather than
  merely unused, which is what makes that a statement about memory traffic and not
  about dead registers;
* the lane at stored 2 on each near axis and at the reflect row on the far one is the
  SOURCE. It does its own cell in full and then writes every destination that
  back-substitutes to it — up to SEVEN for a D component (two near planes, their
  corner, the far plane, and every composition) — each with its own ORDERED parity
  chain, the displacement stored, and the constitutive update performed there.

EVERY CARRY MASK IS ANDED WITH THE COMPONENT'S OWNERSHIP MASK. That is a defect the
real board's device gate FOUND rather than a precaution (``run_farcarry2``,
2026-08-20: one word of ``Hx`` and one of ``Hy``, 1-2 ULP, on the two-folded-axis rows
alone): with two fills a lane can be the SOURCE of one and the DESTINATION of the
other, and it would then write a ghost from a ``v`` built on a load its own mask
zeroed.

===========================================================================
THE ONE THING THIS FAMILY MAY NOT INHERIT FROM THE REAL FOLDED PAIR
===========================================================================

:func:`.folded_fused_pair.folded_fused_curl_constitutive_D` applies its parity as a
compile-time ``PHY * v0`` sign and composes a corner as ``PHZ * (PHY * v0)``.

THE PARITY IS A REAL SCALE PER STORED PART. MEEP's connection multiplies each
``realnum`` part (a ``-1`` connection is a unary minus per part,
boundaries.cpp:359-366), and the array path's fills form ``parity * plane`` through
``part_products.scale``: the host-rounded ``+-1.0`` word times each part, exact on
every word, signed zeros and subnormals included. The previous whole-complex product
(a ``(+-1, +0)`` coefficient with its zero cross terms) diverged from it at signed
zeros, which is what once made the order of a parity chain observable; per part the
applications commute and fold exactly, and the gate records the re-order and the
fold as declared nulls, measured at the word layer.

The kernel still applies ONE word PER PASS, in the driver's own order, and never a
folded word, because that is what the array path does:
the near applications innermost and ascending, then the far one, with the wall
clear between them (the D family's clear rows ARE the near fill's axes). THE PARITY
WORDS ARE HOST-ROUNDED AND PASSED, never synthesised in-kernel —
:func:`.folded_complex.mirror_parity_coefficients`' rule; the plan keeps each axis's
``(re, im)`` pair and launches the real word. That is also why this kernel needs no
mirror-phase constexpr at all: the phase rides in the words.

===========================================================================
WHY THERE IS NO STENCIL PROBLEM HERE, and why that is not general
===========================================================================

The fusion matrix refuses EIGHTEEN D->E seam-instances as STRUCTURALLY UNFUSABLE on
any backend: ``stepping._offdiagonal_terms`` (stepping.py:1235-1253) reads each
partner component's D volume at FOUR indices while ``step_D`` writes those volumes in
place in the first half of the same launch, and no grid-wide barrier exists inside one
launch. THIS CONSTITUTIVE IS NOT THAT ONE. The certified folded complex constitutive
reads its OWN component at its OWN index — the word the same program computed in the
curl half, or the word this same lane just carried into a ghost — so no lane reads a
word another lane writes. An off-diagonal ``chi1inv`` row is refused by the E half
already, and RESTATED below by name because THIS is the reason the cell is buildable
at all.

**THE OFF-DIAGONAL ADMISSION THE MAGNETIC TWIN CARRIES IS DELIBERATELY ABSENT HERE.**
That twin admits BOTH ``folded complex`` and ``folded complex off-diagonal`` because
on the B->H seam the two arms are the same kernel and the same plan class — the
off-diagonal split is an E-side distinction. On THIS seam it is not a label: the
off-diagonal ``update_E`` is the stencil above, it has its own product
(``complex folded off-diagonal``), and a weld may not claim it. One admission, and the
row is refused by name.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; there is no slot it
can claim without a composition rule nothing has measured. Nothing in ``launch.py``
names this module, ``fastpath.plan_fast_path`` is unchanged, and dispatch stays
disabled — the same deferral :mod:`.folded_complex_fused_magnetic_pair`,
:mod:`.folded_fused_pair` and :mod:`.complex_fused_electric_pair` ship under.

Import contract: importable WITHOUT Triton — the predicate and the plan builder (to
``None``) must answer on the laptop that is the merge bar.

DEVICE STATUS: see :data:`DEVICE_STATUS`. A weld licenses a claim, not a dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    _word_view,
    bloch_phase_table,
    connection_arguments,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .folded_complex import (
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    CODE_PERIODIC,
    MIRROR_SOURCE_INDEX,
    TARGET_IYEE,
    _far_reflect_rows,
    _stored_past_owned_reader,
    folded_axis_kinds,
    folded_complex_composition_curl_coverage,
    folded_complex_constitutive_coverage,
    folded_mirror_ghost_fill_complex_coverage,
    mirror_parity_coefficients,
)
from .launch import SUB_STEPS, CupyPointer, _flat
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between two seam-instances and zero: BOTH corpus
#: rows declare an ELECTRIC source, which the driver injects between the two halves
#: (driver.py:3294-3299). The flag is a claim about the PLAN this module builds --
#: that the leading slot saves and the trailing slot restores -- and is only ever
#: changed in the same edit as that wiring. The MAGNETIC twin keeps the flag at False
#: for the mirror-image reason: on its seam the same two rows are electric-only.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``,
#: which goes on refusing BY NAME every seam the repair cannot invert -- an
#: off-diagonal constitutive (a stencil over the curl's own in-place output), a
#: nonlinear one, a cylindrical r = 0 axis, a fold too short to hold the near fill's
#: source row, and an absorber whose split-field recurrence never ran. On a FOLDED
#: seam the repair also extends its saved and restored cell set to every image the two
#: fills write (``deposit_repair.repair_cells``), which is the clause that makes the
#: flag legitimate on this family rather than merely convenient.
CARRIES_DEPOSIT_REPAIR = True

try:  # pragma: no cover - CUDA host only
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name. All five do
#: work on both corpus rows of this cell, which fold a PERIODIC axis. THE INJECTION
#: IS NOT ONE OF THE FIVE: it is carried by the deposit repair bracket around the
#: launch, not by the launch.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding is
#: visible without importing :mod:`launch`; the test suite pins the two equal.
BACKWARD = 1

#: The two codes a folded axis can take. The NEAR fill runs on either
#: (``stepping._fill_symmetry_ghost_cells`` gates on the mirror phases alone); the FAR
#: fill only on ``MIRROR_PERIODIC`` (``stepping._stored_past_owned``:1454-1469), which
#: is the whole of what separates the two carries.
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: The :data:`.folded_complex.GHOST_FILL_FAMILIES` key this product's fill half is
#: asked about. Spelled out rather than derived from ``CURL_SUB_STEP``, and pinned by
#: test against ``GHOST_FILL_FAMILIES``' own keys.
FILL_FAMILY: str = "D"

#: Which folded axes the NEAR fill images for each D component. Derived from
#: ``TARGET_IYEE`` at import rather than written out, and pinned by test against
#: ``fields.IYEE_SHIFTS``: axis ``a`` fills exactly the components whose Yee shift
#: there is 0, which for D is every component EXCEPT ``a``.
NEAR_FILL_AXES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(axis for axis in range(3) if TARGET_IYEE[name][axis] == 0)
    for name in ("Dx", "Dy", "Dz")
)

#: Which folded PERIODIC axes the FAR fill images for each D component. The exact
#: complement of :data:`NEAR_FILL_AXES` -- ``stepping._fill_folded_far_ghosts``
#: (:1489-1532) writes axis ``a`` for every component whose Yee shift there is ONE --
#: and derived from the same table for the same reason. For D that is the component's
#: OWN axis and only it, so the two sets never overlap.
FAR_FILL_AXES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(axis for axis in range(3) if TARGET_IYEE[name][axis] == 1)
    for name in ("Dx", "Dy", "Dz")
)

#: Does the wall clear land AFTER the near parities and BEFORE the far one? TRUE on
#: this family, and FALSE on the magnetic twin, which is why that module has no such
#: constant. The driver runs fill_symmetry_bc_D (:3300), zero_metal_D (:3301) and
#: fill_folded_far_ghosts_D (:3302) in that order, and on the D family the clear's
#: rows ARE the near fill's axes (both are the component's shift-0 axes), so a near
#: ghost takes the clear after its parity and any far parity lands after the clear.
#: Stated as data so the host suite can assert the kernel's emission order against it
#: rather than reading the order out of the same body it is checking.
CLEAR_AFTER_NEAR: bool = True

#: The complex-multiply device functions this kernel is allowed to call, and the only
#: ones: the partner connection. Every other product has a real operand (a PML
#: coefficient, ``dtdx``, ``inv_eps`` or a +-1 parity word) and is formed part by
#: part, inline; the test scans the shipped body for any other helper.
MULTIPLY_HELPERS: Tuple[str, ...] = ("_connect",)

#: What has and has not been executed on a device. Edited only by a released gate.
DEVICE_STATUS: str = (
    "RELEASED 2026-08-31 on the GPU host GPU 6 (verified physically empty before the "
    "run), RE-RUN in place after this paragraph was written -- the gate pins this "
    "module in source_sha256, so the run that certifies it has to be the one that "
    "saw the paragraph. "
    "run: 3 MiB, no compute process), NVIDIA RTX A6000, cc 8.6, Triton 3.1.0 / CuPy "
    "13.5.1, under the 'keep' float32 subnormal policy (ieee_keep_ftz_stripped) "
    "through an ftz-stripped cache, on the FMA_V1 expansion arm licensed by "
    "results/unified_expansion_2026-08-27/keep/gate.json: "
    "parity/meep_gpu/results/triton_folded_cplx_2026-08-31f/"
    "folded_complex_fused_pair/gate.json, release.released = true. "
    "63 device legs -- 15 configurations x 3 value classes, plus the carry family, "
    "the null controls and two armed harness mutations -- 619 COMPLETE driver steps, "
    "613 fused launches, every one of 26 allocated volumes BIT-IDENTICAL on the "
    "uint32 view against BOTH oracles: the CuPy array path and the three separately "
    "certified Triton products this launch replaces (the folded complex curl on "
    "step_D, the complex mirror ghost fill on family D run as its TWO passes, and "
    "the folded complex constitutive on side E). "
    "12 CARRY legs, each repairing 6 in-seam electric deposits through the SHIPPED "
    "deposit_repair bracket, and 4 NULL CONTROLS with the bracket removed that ALL "
    "diverge as they must. 22 kernel mutations: 18 CAUGHT, 2 confirmed NULL and 2 "
    "recorded UNREACHED with their evidence; 5/5 refusals and 8/8 no-device legs. "
    "The branch-reachability leg counts every constexpr guard entered by some case "
    "and measures the deepest composition at 7 ghost cells per source lane -- the "
    "triple-composite blocks are executed rather than merely shipped. "
    "|| WHAT THE RUN DID NOT ESTABLISH, stated here and not only in the artifact: "
    "the PARITY CHAIN's ORDER and GROUPING are held by CONSTRUCTION and not by a "
    "device leg. The parity_chain_associativity leg measures both defects REAL at "
    "the word layer -- re-ordering a two-parity chain moves 32 of 2048 uint32 words "
    "and folding it into one coefficient moves 34, identically on BOTH expansion "
    "arms, every moved word at a signed zero -- and THREE device configurations "
    "(uniform with the carry, and the +-0 lattice QUIET at one step, which is the "
    "only shape in which every ghost SOURCE lane still holds a seeded signed zero "
    "when step_D runs) left every compared volume byte-identical. So the two "
    "mutations that re-order and fold the chain come back UNREACHED rather than "
    "null. The MAGNETIC twin reached the same position independently on 2026-08-21, "
    "which is the cross-check that this is a property of the arithmetic rather than "
    "of this harness. A synthetic bare-array leg with planted words at the source "
    "lane is what would close it. "
    "|| TWO CONFIRMED NULLS, both about the OWNERSHIP masks and both measured: "
    "dropping the mask on the D LOAD changes memory traffic and no byte, because "
    "every consumer of the register it feeds is masked by the same predicate; and "
    "dropping the ownership conjunction on a carry mask corrupts a cell that a LATER "
    "composite block rewrites correctly in the same program. Both masks STAY -- the "
    "second nullity rests on the emission order and on BLOCK, and the real folded "
    "pair's own gate measured that defect class REAL on its body. "
    "|| STILL NOT WIRED. A weld licenses a claim, not a dispatch: nothing in a "
    "default run reaches this plan. This module has no fingerprints.json entry -- "
    "its licensed bytes are a function of a PROBE-BOUND arm, so a checked-in hash "
    "would record a choice rather than a measurement; the gate artifact's own "
    "source_sha256 map is the binding, and the fusion matrix reads it.")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CLEAR_AFTER_NEAR", "CONSTITUTIVE_SIDE",
    "CURL_SUB_STEP", "DEVICE_STATUS", "FAR_FILL_AXES", "FILL_FAMILY",
    "MULTIPLY_HELPERS", "MIRROR_CODES", "NEAR_FILL_AXES",
    "REPLACES",
    "FoldedComplexFusedPairPlan",
    "carried_destinations",
    "folded_complex_fused_curl_constitutive_D",
    "folded_complex_fused_curl_constitutive_D_kernel",
    "folded_complex_fused_pair_coverage",
    "parity_chain",
    "parity_coefficient_words",
    "plan_folded_complex_fused_pair",
    "plan_folded_complex_fused_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The ownership rule and the parity chain — host side, so they can be READ
# ---------------------------------------------------------------------------

def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """Every ghost cell ONE source lane owns, as ``(near axes, far axes)``.

    THE OWNERSHIP RULE. The driver runs the near fill, the wall clear and the far fill
    in that order (driver.py:3300-3302), and the near fill loops its axes in ascending
    order, so a cell the fills leave at stored 0 of SEVERAL near axes and at the top of
    the far one is written more than once. Back-substituting every pass gives ONE
    source cell per ghost — the cell at stored :data:`.folded_complex.MIRROR_SOURCE_INDEX`
    on each near axis and at the reflect row on the far one — so one lane computes the
    displacement every one of those ghosts carries and writes them all.

    THE D SHAPE IS THE INVERSE OF THE B ONE: up to TWO near axes and at most ONE far
    axis, against the magnetic twin's one near and up to two far. With both near axes
    and the far axis folded a component owns SEVEN ghosts.

    Returned smallest-subset-first for a stable emission order; the order the blocks
    are emitted in is unobservable (the cells are distinct) and a stable one keeps a
    source diff readable.
    """
    near = tuple(int(axis) for axis in near)
    far = tuple(int(axis) for axis in far)
    combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for near_size in range(len(near) + 1):
        for near_subset in itertools.combinations(near, near_size):
            for far_size in range(len(far) + 1):
                for far_subset in itertools.combinations(far, far_size):
                    if not near_subset and not far_subset:
                        continue  # the lane's OWN cell, not a ghost
                    combos.append((near_subset, far_subset))
    return tuple(sorted(
        combos, key=lambda item: (len(item[0]) + len(item[1]), item[0], item[1])))


def parity_chain(near_subset: Sequence[int], far_subset: Sequence[int]
                 ) -> Tuple[Tuple[int, str], ...]:
    """The ORDERED ``(axis, pass)`` parity applications one ghost's value carries.

    THE ORDER IS THE DRIVER'S AND IS TRANSCRIBED, NEVER CHOSEN. Each pass reads the
    plane the previous one wrote, so back-substituting a corner gives

        c_far[f] (x) ( CLEAR ( c_near[a2] (x) ( c_near[a1] (x) v ) ) ),  a1 < a2

    — the near applications INNERMOST and ASCENDING (``_fill_symmetry_ghost_cells``
    loops its axes in x, y, z order, stepping.py:1441-1447), then the wall clear
    (:3301, and see :data:`CLEAR_AFTER_NEAR`), then the far axis.

    THE CLEAR IS NOT IN THE RETURNED CHAIN, because it is not a parity: it is a
    ``where`` on the value between the two groups. :data:`CLEAR_AFTER_NEAR` states
    where it lands and the host suite asserts the kernel emits it there. A FAR-ONLY
    ghost takes no clear of its own -- the value it images is the owned register,
    which has already taken the clear -- and a NEAR ghost takes one after its parity.

    A single-entry chain is the common case and the only one the two corpus rows this
    carry admits ever reach — ``special_kz_2_21_2`` and ``triangular_lattice_oblique``
    each fold ONE axis, which gives two components a near ghost and one a far ghost.
    The longer chains are reachable configuration space all the same and the gate
    carries those cases rather than leaving the ordering untested because the corpus
    does not press on it.
    """
    chain: List[Tuple[int, str]] = []
    previous = -1
    for axis in near_subset:
        axis = int(axis)
        if axis <= previous:
            raise AssertionError(
                f"the near axis subset {tuple(near_subset)!r} is not strictly "
                f"ascending; stepping._fill_symmetry_ghost_cells applies its axes in "
                f"ascending order (:1441-1447) and this chain transcribes that order")
        previous = axis
        chain.append((axis, "near"))
    previous = -1
    for axis in far_subset:
        axis = int(axis)
        if axis <= previous:
            raise AssertionError(
                f"the far axis subset {tuple(far_subset)!r} is not strictly "
                f"ascending; stepping._fill_folded_far_ghosts applies its axes in "
                f"ascending order (:1518) and this chain transcribes that order")
        previous = axis
        chain.append((axis, "far"))
    return tuple(chain)


def parity_coefficient_words(grid: Any) -> Tuple[Tuple[float, float], ...]:
    """The six ``(re, im)`` word pairs the kernel takes — near x/y/z then far x/y/z.

    Every pair is :func:`.folded_complex.mirror_parity_coefficients`' output for that
    axis's declared phase, HOST-ROUNDED through ``numpy.complex64`` exactly once, and
    ``(+0.0, +0.0)`` on an unfolded axis, where no block that reads it is emitted.

    ``mirror_parity_coefficients`` returns ``(+phase, -phase)`` — the shift-0 parity
    and the shift-1 one (``fields.mirror_parity`` :117 is ``phase * (1 - 2 * iyee)``).
    On the D family the NEAR fill is the shift-0 pass and the FAR fill the shift-1
    one, exactly as on the B family, so the same pair serves both sides and the
    near/far split is about WHICH AXIS, never about which word.

    A zero pair on a FOLDED axis would be a whole plane of exact zeros — loud — rather
    than a plausible field, which is the same choice
    :func:`.folded_fused_pair.mirror_phases` makes for its constexpr.
    """
    near: List[Tuple[float, float]] = []
    far: List[Tuple[float, float]] = []
    for axis in range(3):
        if not bool(_call(grid, "is_mirrored", axis, default=False)):
            near.append((0.0, 0.0))
            far.append((0.0, 0.0))
            continue
        phase = _call(grid, "mirror_phase", axis, default=None)
        if phase not in (1, -1):
            near.append((0.0, 0.0))
            far.append((0.0, 0.0))
            continue
        near_words, far_words = mirror_parity_coefficients(int(phase))
        near.append(near_words)
        far.append(far_words)
    return tuple(near) + tuple(far)


if triton is not None:

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

    # The forms, the record layout and the connection, read by the kernel below:
    # its curl half is folded_complex.folded_bloch_pml_curl_step's body and its
    # constitutive half complex_fields.bloch_constitutive_step's, per part.
    from .complex_fields import _connect  # noqa: E402
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: E402
                        curl_stencil, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

    @triton.jit
    def _carry_ghost_complex_E(f, w, e, ie, dst, ghost_re, ghost_im, kp_d, km_d, dw_d,
                               mask):
        """Write ONE imaged ghost cell of ``D``, both parts, and run ``update_E`` there.

        :func:`.folded_fused_pair._carry_ghost_E`'s statements once per stored part, at
        the cell's two word offsets, with ``src = ghost * inv_eps`` where ``inv_eps``
        is one float32 word per COMPLEX cell (``+ dst``, never word-doubled): the
        displacement store, then the workspace read, the workspace write and the
        ``E`` accumulate, ``prev`` read BEFORE the write. ``dw_d`` is the
        destination's membership.
        """
        us = tl.load(ie + dst, mask=mask, other=0.0)
        tl.store(f + 2 * dst, ghost_re, mask=mask)
        tl.store(f + 2 * dst + 1, ghost_im, mask=mask)
        prev_re = tl.load(w + 2 * dst, mask=mask & dw_d, other=0.0)
        prev_im = tl.load(w + 2 * dst + 1, mask=mask & dw_d, other=0.0)
        src_re = ghost_re * us
        src_im = ghost_im * us
        tl.store(w + 2 * dst, src_re, mask=mask & dw_d)
        tl.store(w + 2 * dst + 1, src_im, mask=mask & dw_d)
        acc_re = tl.load(e + 2 * dst, mask=mask & dw_d, other=0.0)
        acc_im = tl.load(e + 2 * dst + 1, mask=mask & dw_d, other=0.0)
        tl.store(e + 2 * dst, constitutive_form(acc_re, prev_re, src_re, kp_d, km_d, dw_d),
                 mask=mask)
        tl.store(e + 2 * dst + 1,
                 constitutive_form(acc_im, prev_im, src_im, kp_d, km_d, dw_d), mask=mask)

    @triton.jit
    def folded_complex_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx,Dy,Dz (complex64 as words)
        u0, u1, u2,                       # curl auxiliaries: fu_Dx,fu_Dy,fu_Dz
        g0, g1, g2,                       # curl sources: Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        e0, e1, e2,                       # constitutive targets: Ex,Ey,Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex,f_w_Ey,f_w_Ez
        ie0, ie1, ie2,                    # inverse epsilon, float32 VOLUMES
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER lattice
        cb,                               # chunk record: both halves' forms
        nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded
        cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        n0r, n1r, n2r,                    # NEAR parity words, per axis (+phase)
        d0r, d1r, d2r,                    # FAR parity words, per axis (-phase)
        BACKWARD: tl.constexpr,           # bound to 1 by every builder
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Folded complex ``step_D`` + both fills + ``zero_metal_D`` + ``update_E``.

        :func:`.folded_fused_pair.folded_fused_curl_constitutive_D` run once per stored
        part, its curl half :func:`.folded_complex.folded_bloch_pml_curl_step` (the
        Bloch connection of an unfolded phased axis included). ``BACKWARD`` is carried
        and must be 1. ``NEAR_a`` / ``FAR_a`` are derived from ``bc`` by the plan;
        ``rx``/``ry``/``rz`` are ``stepping._far_reflect_rows``' rows; ``n*r`` /
        ``d*r`` the near (+phase) and far (-phase) parity words, host-rounded. ``cb``
        serves both halves.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- DELTA 1: the ghost rule, per axis (stepping._shift_up / _shift_down) ---
        # PERIODIC wraps (a plain roll, phased or not); EVERY other rule here serves an
        # exact (+0.0, +0.0) past the face, which `tl.load`'s `other=` delivers on both
        # words. On a folded axis that zero stands in for a value nothing reads (the two
        # masks below zero every cell that reads it).
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vx, vy, vz = live, live, live
        if BCX == PERIODIC:
            si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
        else:
            vx = live & (si >= 0) & (si < nx)
        if BCY == PERIODIC:
            sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        else:
            vy = live & (sj >= 0) & (sj < ny)
        if BCZ == PERIODIC:
            sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
        else:
            vz = live & (sk >= 0) & (sk < nz)

        ox = si * nyz + j * nz + k
        oy = i * nyz + sj * nz + k
        oz = i * nyz + j * nz + sk

        # --- loads: two words per operand ------------------------------------------
        a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
        b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
        b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
        c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
        c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
        a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
        a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)
        a_z_re = tl.load(g0 + 2 * oz, mask=vz, other=0.0)
        a_z_im = tl.load(g0 + 2 * oz + 1, mask=vz, other=0.0)
        b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)
        b_x_im = tl.load(g1 + 2 * ox + 1, mask=vx, other=0.0)
        b_z_re = tl.load(g1 + 2 * oz, mask=vz, other=0.0)
        b_z_im = tl.load(g1 + 2 * oz + 1, mask=vz, other=0.0)
        c_x_re = tl.load(g2 + 2 * ox, mask=vx, other=0.0)
        c_x_im = tl.load(g2 + 2 * ox + 1, mask=vx, other=0.0)
        c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)
        c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)

        # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
        # Target 0 reads c along y and b along z, target 1 a along z and c along x,
        # target 2 b along x and a along y (stepping's term table). `tl.where` is a
        # bitwise select, so every cell off plane 0 keeps its loaded words.
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
        b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
        a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
        c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
        b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
        a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
        if CNX:
            r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
            c1_re = tl.where(at_x, r_re, c_re)
            c1_im = tl.where(at_x, r_im, c_im)
            r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
            b2_re = tl.where(at_x, r_re, b_re)
            b2_im = tl.where(at_x, r_im, b_im)
        if CNY:
            r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
            c0_re = tl.where(at_y, r_re, c_re)
            c0_im = tl.where(at_y, r_im, c_im)
            r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
            a2_re = tl.where(at_y, r_re, a_re)
            a2_im = tl.where(at_y, r_im, a_im)
        if CNZ:
            r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
            b0_re = tl.where(at_z, r_re, b_re)
            b0_im = tl.where(at_z, r_im, b_im)
            r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
            a1_re = tl.where(at_z, r_re, a_re)
            a1_im = tl.where(at_z, r_im, a_im)

        # --- the curl (stepping._curl_from_operands), per part ---------------------
        # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
        # association with both partners, the one-partner form where the record says the
        # other partner's derivative axis is invariant, an exact +0 with neither. The
        # dtdx scale is the real scalar times each part (part_products.scale).
        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
        curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
        curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
        curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
        curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
        curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

        # --- DELTA 2: ownership mask, cell 0 (stepping._mask_non_owned_cells) -------
        # Every target whose Yee shift is 0 on a non-periodic axis (``is_mirrored or
        # is_metallic``): the unfolded body's ``== METALLIC`` widened to ``!= PERIODIC``.
        # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
        if BACKWARD:
            if BCY != PERIODIC:
                curl0_re = tl.where(at_y, 0.0, curl0_re)
                curl0_im = tl.where(at_y, 0.0, curl0_im)
            if BCZ != PERIODIC:
                curl0_re = tl.where(at_z, 0.0, curl0_re)
                curl0_im = tl.where(at_z, 0.0, curl0_im)
            if BCX != PERIODIC:
                curl1_re = tl.where(at_x, 0.0, curl1_re)
                curl1_im = tl.where(at_x, 0.0, curl1_im)
            if BCZ != PERIODIC:
                curl1_re = tl.where(at_z, 0.0, curl1_re)
                curl1_im = tl.where(at_z, 0.0, curl1_im)
            if BCX != PERIODIC:
                curl2_re = tl.where(at_x, 0.0, curl2_re)
                curl2_im = tl.where(at_x, 0.0, curl2_im)
            if BCY != PERIODIC:
                curl2_re = tl.where(at_y, 0.0, curl2_re)
                curl2_im = tl.where(at_y, 0.0, curl2_im)
        else:
            if BCX != PERIODIC:
                curl0_re = tl.where(at_x, 0.0, curl0_re)
                curl0_im = tl.where(at_x, 0.0, curl0_im)
            if BCY != PERIODIC:
                curl1_re = tl.where(at_y, 0.0, curl1_re)
                curl1_im = tl.where(at_y, 0.0, curl1_im)
            if BCZ != PERIODIC:
                curl2_re = tl.where(at_z, 0.0, curl2_re)
                curl2_im = tl.where(at_z, 0.0, curl2_im)

        # --- DELTA 3: ownership mask, the TOP plane of a folded PERIODIC axis --------
        # Every target whose Yee shift is 1 there: that slot sits past MEEP's owned
        # window and the far fill, not the curl, writes it (symmetry.py's block, both
        # parts).
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BACKWARD:
            # Dx:(1,0,0)  Dy:(0,1,0)  Dz:(0,0,1) -- shift 1 on its OWN axis only.
            if BCX == MIRROR_PERIODIC:
                curl0_re = tl.where(last_x, 0.0, curl0_re)
                curl0_im = tl.where(last_x, 0.0, curl0_im)
            if BCY == MIRROR_PERIODIC:
                curl1_re = tl.where(last_y, 0.0, curl1_re)
                curl1_im = tl.where(last_y, 0.0, curl1_im)
            if BCZ == MIRROR_PERIODIC:
                curl2_re = tl.where(last_z, 0.0, curl2_re)
                curl2_im = tl.where(last_z, 0.0, curl2_im)
        else:
            # Bx:(0,1,1)  By:(1,0,1)  Bz:(1,1,0) -- shift 1 on the two OTHER axes.
            if BCY == MIRROR_PERIODIC:
                curl0_re = tl.where(last_y, 0.0, curl0_re)
                curl0_im = tl.where(last_y, 0.0, curl0_im)
            if BCZ == MIRROR_PERIODIC:
                curl0_re = tl.where(last_z, 0.0, curl0_re)
                curl0_im = tl.where(last_z, 0.0, curl0_im)
            if BCX == MIRROR_PERIODIC:
                curl1_re = tl.where(last_x, 0.0, curl1_re)
                curl1_im = tl.where(last_x, 0.0, curl1_im)
            if BCZ == MIRROR_PERIODIC:
                curl1_re = tl.where(last_z, 0.0, curl1_re)
                curl1_im = tl.where(last_z, 0.0, curl1_im)
            if BCX == MIRROR_PERIODIC:
                curl2_re = tl.where(last_x, 0.0, curl2_re)
                curl2_im = tl.where(last_x, 0.0, curl2_im)
            if BCY == MIRROR_PERIODIC:
                curl2_re = tl.where(last_y, 0.0, curl2_re)
                curl2_im = tl.where(last_y, 0.0, curl2_im)

        # --- ownership of the fill destinations (folded_fused_pair's) --------------
        # Component m is a NEAR destination at stored cell 0 of each OTHER folded axis and
        # a FAR destination at the top plane of its OWN folded periodic axis (Dx (1,0,0),
        # Dy (0,1,0), Dz (0,0,1)). Those lanes do NOT load or store D, E or f_w_E, either
        # part: the source lane writes all three for them.
        own0, own1, own2 = live, live, live
        if NEAR_Y:
            own0 = own0 & (j != 0)
            own2 = own2 & (j != 0)
        if NEAR_Z:
            own0 = own0 & (k != 0)
            own1 = own1 & (k != 0)
        if NEAR_X:
            own1 = own1 & (i != 0)
            own2 = own2 & (i != 0)
        if FAR_X:
            own0 = own0 & (i != nx - 1)
        if FAR_Y:
            own1 = own1 & (j != ny - 1)
        if FAR_Z:
            own2 = own2 & (k != nz - 1)

        # --- the PML forms (stepping._apply_curl_regions), per part ----------------
        # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). One record
        # select per target serves both parts; fu is read and written on the
        # recurrence's cells only (the auxiliary contract), both words.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)
        p_im = tl.load(u0 + 2 * idx + 1, mask=live & du0, other=0.0)
        v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=own0, other=0.0), p_re, curl0_re,
                                 km_y, si_y, km_z, si_z, du0)
        v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=own0, other=0.0), p_im,
                                 curl0_im, km_y, si_y, km_z, si_z, du0)

        p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
        p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
        v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=own1, other=0.0), p_re, curl1_re,
                                 km_z, si_z, km_x, si_x, du1)
        v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=own1, other=0.0), p_im,
                                 curl1_im, km_z, si_z, km_x, si_x, du1)

        p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
        p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
        v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=own2, other=0.0), p_re, curl2_re,
                                 km_x, si_x, km_y, si_y, du2)
        v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=own2, other=0.0), p_im,
                                 curl2_im, km_x, si_x, km_y, si_y, du2)

        # --- the seam: stepping.zero_metal_D, on the registers -----------------------
        # The TANGENTIAL two components per walled axis (the electric twin's map).
        if ZM_X:
            v1_re = tl.where(at_x, 0.0, v1_re)
            v1_im = tl.where(at_x, 0.0, v1_im)
            v2_re = tl.where(at_x, 0.0, v2_re)
            v2_im = tl.where(at_x, 0.0, v2_im)
        if ZM_Y:
            v0_re = tl.where(at_y, 0.0, v0_re)
            v0_im = tl.where(at_y, 0.0, v0_im)
            v2_re = tl.where(at_y, 0.0, v2_re)
            v2_im = tl.where(at_y, 0.0, v2_im)
        if ZM_Z:
            v0_re = tl.where(at_z, 0.0, v0_re)
            v0_im = tl.where(at_z, 0.0, v0_im)
            v1_re = tl.where(at_z, 0.0, v1_re)
            v1_im = tl.where(at_z, 0.0, v1_im)

        # --- stores: fu on the recurrence's cells, the fields on owned cells -------
        tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
        tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
        tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
        tl.store(f0 + 2 * idx, v0_re, mask=own0)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=own0)
        tl.store(f1 + 2 * idx, v1_re, mask=own1)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=own1)
        tl.store(f2 + 2 * idx, v2_re, mask=own2)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=own2)

        # ==================== the constitutive half (update_E) ======================
        # bloch_constitutive_step's E arm per part, `src` the register pair times inv_eps
        # (one float32 word per COMPLEX cell) and `live` the component's ownership; dsigw
        # lines on the half-integer lattice.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        dw0 = record_in(cb, RECORD_HALF_INTEGER + 0, i)
        dw1 = record_in(cb, RECORD_HALF_INTEGER + 4, j)
        dw2 = record_in(cb, RECORD_HALF_INTEGER + 8, k)

        # --- component 0 ---------------------------------------------------------
        prev_re = tl.load(w0 + 2 * idx, mask=own0 & dw0, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w0 + 2 * idx + 1, mask=own0 & dw0, other=0.0)
        us = tl.load(ie0 + idx, mask=own0, other=0.0)
        src_re = v0_re * us
        src_im = v0_im * us
        tl.store(w0 + 2 * idx, src_re, mask=own0 & dw0)
        tl.store(w0 + 2 * idx + 1, src_im, mask=own0 & dw0)
        a_re = tl.load(e0 + 2 * idx, mask=own0 & dw0, other=0.0)
        a_im = tl.load(e0 + 2 * idx + 1, mask=own0 & dw0, other=0.0)
        tl.store(e0 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_0, km_0, dw0),
                 mask=own0)
        tl.store(e0 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_0, km_0, dw0), mask=own0)

        # --- component 1 ---------------------------------------------------------
        prev_re = tl.load(w1 + 2 * idx, mask=own1 & dw1, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w1 + 2 * idx + 1, mask=own1 & dw1, other=0.0)
        us = tl.load(ie1 + idx, mask=own1, other=0.0)
        src_re = v1_re * us
        src_im = v1_im * us
        tl.store(w1 + 2 * idx, src_re, mask=own1 & dw1)
        tl.store(w1 + 2 * idx + 1, src_im, mask=own1 & dw1)
        a_re = tl.load(e1 + 2 * idx, mask=own1 & dw1, other=0.0)
        a_im = tl.load(e1 + 2 * idx + 1, mask=own1 & dw1, other=0.0)
        tl.store(e1 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_1, km_1, dw1),
                 mask=own1)
        tl.store(e1 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_1, km_1, dw1), mask=own1)

        # --- component 2 ---------------------------------------------------------
        prev_re = tl.load(w2 + 2 * idx, mask=own2 & dw2, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w2 + 2 * idx + 1, mask=own2 & dw2, other=0.0)
        us = tl.load(ie2 + idx, mask=own2, other=0.0)
        src_re = v2_re * us
        src_im = v2_im * us
        tl.store(w2 + 2 * idx, src_re, mask=own2 & dw2)
        tl.store(w2 + 2 * idx + 1, src_im, mask=own2 & dw2)
        a_re = tl.load(e2 + 2 * idx, mask=own2 & dw2, other=0.0)
        a_im = tl.load(e2 + 2 * idx + 1, mask=own2 & dw2, other=0.0)
        tl.store(e2 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_2, km_2, dw2),
                 mask=own2)
        tl.store(e2 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_2, km_2, dw2), mask=own2)

        # ========= the mirror fills, carried by the SOURCE lane =====================
        # folded_fused_pair's carries, per part, with the fills' parity words (host-rounded
        # +-1.0, near ``n*r`` = +phase, far ``d*r`` = -phase): a NEAR ghost (written before
        # the wall clear) takes the clear, a FAR ghost (written after it) does not, and
        # each fill is a real scale per part (part_products.scale), exact, applied in the
        # driver's order (near x, y, z, then far). A far destination reads its coefficient
        # pair and membership at the TOP row of the component's own axis; a near-only one
        # keeps the source lane's.
        top_x = idx * 0 + (nx - 1)
        top_y = idx * 0 + (ny - 1)
        top_z = idx * 0 + (nz - 1)
        kp_f0 = tl.load(kp0 + top_x, mask=live, other=0.0)
        km_f0 = tl.load(km0 + top_x, mask=live, other=0.0)
        kp_f1 = tl.load(kp1 + top_y, mask=live, other=0.0)
        km_f1 = tl.load(km1 + top_y, mask=live, other=0.0)
        kp_f2 = tl.load(kp2 + top_z, mask=live, other=0.0)
        km_f2 = tl.load(km2 + top_z, mask=live, other=0.0)
        dw_f0 = record_in(cb, RECORD_HALF_INTEGER + 0, top_x)
        dw_f1 = record_in(cb, RECORD_HALF_INTEGER + 4, top_y)
        dw_f2 = record_in(cb, RECORD_HALF_INTEGER + 8, top_z)
        near_i = live & (i == 2)
        near_j = live & (j == 2)
        near_k = live & (k == 2)
        far_i = live & (i == rx)
        far_j = live & (j == ry)
        far_k = live & (k == rz)
        dn_x = -2 * nyz
        dn_y = -2 * nz
        dn_z = -2
        df_x = (nx - 1 - rx) * nyz
        df_y = (ny - 1 - ry) * nz
        df_z = (nz - 1 - rz)

        # --- component 0 (Dx): near on y and z, far on x ---
        if NEAR_Y:
            gv_0y_re = n1r * v0_re
            gv_0y_im = n1r * v0_im
            if ZM_Y:
                gv_0y_re = tl.where(at_y, 0.0, gv_0y_re)
                gv_0y_im = tl.where(at_y, 0.0, gv_0y_im)
            if ZM_Z:
                gv_0y_re = tl.where(at_z, 0.0, gv_0y_re)
                gv_0y_im = tl.where(at_z, 0.0, gv_0y_im)
            _carry_ghost_complex_E(f0, w0, e0, ie0, idx + dn_y, gv_0y_re, gv_0y_im,
                                   kp_0, km_0, dw0, own0 & near_j)
        if NEAR_Z:
            gv_0z_re = n2r * v0_re
            gv_0z_im = n2r * v0_im
            if ZM_Y:
                gv_0z_re = tl.where(at_y, 0.0, gv_0z_re)
                gv_0z_im = tl.where(at_y, 0.0, gv_0z_im)
            if ZM_Z:
                gv_0z_re = tl.where(at_z, 0.0, gv_0z_re)
                gv_0z_im = tl.where(at_z, 0.0, gv_0z_im)
            _carry_ghost_complex_E(f0, w0, e0, ie0, idx + dn_z, gv_0z_re, gv_0z_im,
                                   kp_0, km_0, dw0, own0 & near_k)
        if NEAR_Y and NEAR_Z:
            gv_0yz_re = n2r * (n1r * v0_re)
            gv_0yz_im = n2r * (n1r * v0_im)
            if ZM_Y:
                gv_0yz_re = tl.where(at_y, 0.0, gv_0yz_re)
                gv_0yz_im = tl.where(at_y, 0.0, gv_0yz_im)
            if ZM_Z:
                gv_0yz_re = tl.where(at_z, 0.0, gv_0yz_re)
                gv_0yz_im = tl.where(at_z, 0.0, gv_0yz_im)
            _carry_ghost_complex_E(f0, w0, e0, ie0, idx + dn_y + dn_z,
                                   gv_0yz_re, gv_0yz_im, kp_0, km_0, dw0,
                                   own0 & near_j & near_k)
        if FAR_X:
            _carry_ghost_complex_E(f0, w0, e0, ie0, idx + df_x, d0r * v0_re, d0r * v0_im,
                                   kp_f0, km_f0, dw_f0, own0 & far_i)
        if FAR_X and NEAR_Y:
            _carry_ghost_complex_E(f0, w0, e0, ie0, idx + df_x + dn_y, d0r * gv_0y_re, d0r * gv_0y_im,
                                   kp_f0, km_f0, dw_f0, own0 & far_i & near_j)
        if FAR_X and NEAR_Z:
            _carry_ghost_complex_E(f0, w0, e0, ie0, idx + df_x + dn_z, d0r * gv_0z_re, d0r * gv_0z_im,
                                   kp_f0, km_f0, dw_f0, own0 & far_i & near_k)
        if FAR_X and (NEAR_Y and NEAR_Z):
            _carry_ghost_complex_E(f0, w0, e0, ie0, idx + df_x + dn_y + dn_z, d0r * gv_0yz_re, d0r * gv_0yz_im,
                                   kp_f0, km_f0, dw_f0, own0 & far_i & near_j & near_k)

        # --- component 1 (Dy): near on x and z, far on y ---
        if NEAR_X:
            gv_1x_re = n0r * v1_re
            gv_1x_im = n0r * v1_im
            if ZM_X:
                gv_1x_re = tl.where(at_x, 0.0, gv_1x_re)
                gv_1x_im = tl.where(at_x, 0.0, gv_1x_im)
            if ZM_Z:
                gv_1x_re = tl.where(at_z, 0.0, gv_1x_re)
                gv_1x_im = tl.where(at_z, 0.0, gv_1x_im)
            _carry_ghost_complex_E(f1, w1, e1, ie1, idx + dn_x, gv_1x_re, gv_1x_im,
                                   kp_1, km_1, dw1, own1 & near_i)
        if NEAR_Z:
            gv_1z_re = n2r * v1_re
            gv_1z_im = n2r * v1_im
            if ZM_X:
                gv_1z_re = tl.where(at_x, 0.0, gv_1z_re)
                gv_1z_im = tl.where(at_x, 0.0, gv_1z_im)
            if ZM_Z:
                gv_1z_re = tl.where(at_z, 0.0, gv_1z_re)
                gv_1z_im = tl.where(at_z, 0.0, gv_1z_im)
            _carry_ghost_complex_E(f1, w1, e1, ie1, idx + dn_z, gv_1z_re, gv_1z_im,
                                   kp_1, km_1, dw1, own1 & near_k)
        if NEAR_X and NEAR_Z:
            gv_1xz_re = n2r * (n0r * v1_re)
            gv_1xz_im = n2r * (n0r * v1_im)
            if ZM_X:
                gv_1xz_re = tl.where(at_x, 0.0, gv_1xz_re)
                gv_1xz_im = tl.where(at_x, 0.0, gv_1xz_im)
            if ZM_Z:
                gv_1xz_re = tl.where(at_z, 0.0, gv_1xz_re)
                gv_1xz_im = tl.where(at_z, 0.0, gv_1xz_im)
            _carry_ghost_complex_E(f1, w1, e1, ie1, idx + dn_x + dn_z,
                                   gv_1xz_re, gv_1xz_im, kp_1, km_1, dw1,
                                   own1 & near_i & near_k)
        if FAR_Y:
            _carry_ghost_complex_E(f1, w1, e1, ie1, idx + df_y, d1r * v1_re, d1r * v1_im,
                                   kp_f1, km_f1, dw_f1, own1 & far_j)
        if FAR_Y and NEAR_X:
            _carry_ghost_complex_E(f1, w1, e1, ie1, idx + df_y + dn_x, d1r * gv_1x_re, d1r * gv_1x_im,
                                   kp_f1, km_f1, dw_f1, own1 & far_j & near_i)
        if FAR_Y and NEAR_Z:
            _carry_ghost_complex_E(f1, w1, e1, ie1, idx + df_y + dn_z, d1r * gv_1z_re, d1r * gv_1z_im,
                                   kp_f1, km_f1, dw_f1, own1 & far_j & near_k)
        if FAR_Y and (NEAR_X and NEAR_Z):
            _carry_ghost_complex_E(f1, w1, e1, ie1, idx + df_y + dn_x + dn_z, d1r * gv_1xz_re, d1r * gv_1xz_im,
                                   kp_f1, km_f1, dw_f1, own1 & far_j & near_i & near_k)

        # --- component 2 (Dz): near on x and y, far on z ---
        if NEAR_X:
            gv_2x_re = n0r * v2_re
            gv_2x_im = n0r * v2_im
            if ZM_X:
                gv_2x_re = tl.where(at_x, 0.0, gv_2x_re)
                gv_2x_im = tl.where(at_x, 0.0, gv_2x_im)
            if ZM_Y:
                gv_2x_re = tl.where(at_y, 0.0, gv_2x_re)
                gv_2x_im = tl.where(at_y, 0.0, gv_2x_im)
            _carry_ghost_complex_E(f2, w2, e2, ie2, idx + dn_x, gv_2x_re, gv_2x_im,
                                   kp_2, km_2, dw2, own2 & near_i)
        if NEAR_Y:
            gv_2y_re = n1r * v2_re
            gv_2y_im = n1r * v2_im
            if ZM_X:
                gv_2y_re = tl.where(at_x, 0.0, gv_2y_re)
                gv_2y_im = tl.where(at_x, 0.0, gv_2y_im)
            if ZM_Y:
                gv_2y_re = tl.where(at_y, 0.0, gv_2y_re)
                gv_2y_im = tl.where(at_y, 0.0, gv_2y_im)
            _carry_ghost_complex_E(f2, w2, e2, ie2, idx + dn_y, gv_2y_re, gv_2y_im,
                                   kp_2, km_2, dw2, own2 & near_j)
        if NEAR_X and NEAR_Y:
            gv_2xy_re = n1r * (n0r * v2_re)
            gv_2xy_im = n1r * (n0r * v2_im)
            if ZM_X:
                gv_2xy_re = tl.where(at_x, 0.0, gv_2xy_re)
                gv_2xy_im = tl.where(at_x, 0.0, gv_2xy_im)
            if ZM_Y:
                gv_2xy_re = tl.where(at_y, 0.0, gv_2xy_re)
                gv_2xy_im = tl.where(at_y, 0.0, gv_2xy_im)
            _carry_ghost_complex_E(f2, w2, e2, ie2, idx + dn_x + dn_y,
                                   gv_2xy_re, gv_2xy_im, kp_2, km_2, dw2,
                                   own2 & near_i & near_j)
        if FAR_Z:
            _carry_ghost_complex_E(f2, w2, e2, ie2, idx + df_z, d2r * v2_re, d2r * v2_im,
                                   kp_f2, km_f2, dw_f2, own2 & far_k)
        if FAR_Z and NEAR_X:
            _carry_ghost_complex_E(f2, w2, e2, ie2, idx + df_z + dn_x, d2r * gv_2x_re, d2r * gv_2x_im,
                                   kp_f2, km_f2, dw_f2, own2 & far_k & near_i)
        if FAR_Z and NEAR_Y:
            _carry_ghost_complex_E(f2, w2, e2, ie2, idx + df_z + dn_y, d2r * gv_2y_re, d2r * gv_2y_im,
                                   kp_f2, km_f2, dw_f2, own2 & far_k & near_j)
        if FAR_Z and (NEAR_X and NEAR_Y):
            _carry_ghost_complex_E(f2, w2, e2, ie2, idx + df_z + dn_x + dn_y, d2r * gv_2xy_re, d2r * gv_2xy_im,
                                   kp_f2, km_f2, dw_f2, own2 & far_k & near_i & near_j)

else:  # pragma: no cover - laptop path
    folded_complex_fused_curl_constitutive_D = None  # type: ignore[assignment]


def folded_complex_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if folded_complex_fused_curl_constitutive_D is None:
        raise ImportError(
            "the folded complex fused electric D/E kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return folded_complex_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def folded_complex_fused_pair_coverage(fields: Any, pml: Any,
                                       sources: Any = None) -> Coverage:
    """May ONE launch span folded complex ``step_D`` -> both fills -> ``update_E``?

    A conjunction of the THREE halves' own certified predicates plus the seam
    clauses. Nothing is weakened: a configuration any half refuses is refused here
    with that half's reasons, prefixed so a reader can tell which side said it — the
    same construction
    :func:`.folded_complex_fused_magnetic_pair.folded_complex_fused_magnetic_pair_coverage`
    and :func:`.folded_fused_pair.folded_fused_pair_coverage` use.

    ONE ADMISSION ONLY, unlike the magnetic twin. That family admits the off-diagonal
    label as well, because on B->H the two arms are the same kernel; on THIS seam the
    off-diagonal ``update_E`` is a stencil over the D volumes ``step_D`` writes in
    place and belongs to a different product entirely, so the row is refused by name.
    """
    reasons: List[str] = []

    curl = folded_complex_composition_curl_coverage(
        fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"folded complex curl half: {reason}"
                       for reason in curl.reasons)
    constitutive = folded_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"folded complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    fill = folded_mirror_ghost_fill_complex_coverage(fields, FILL_FAMILY)
    if not fill.covered:
        reasons.extend(f"folded complex fill half: {reason}"
                       for reason in fill.reasons)

    # THE STENCIL, RESTATED BY NAME. The eighteen structurally unfusable D->E
    # seam-instances on the Triton board are the OFF-DIAGONAL constitutive reading
    # partner D volumes at four indices (stepping.py:1235-1253) while step_D writes
    # them in place, with no grid-wide barrier inside one launch. The E half refuses
    # that row already; it is named again because THIS is the fact that makes the
    # present cell buildable at all, and because this module's rule is that its
    # coverage may not be read off another module's guard.
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append(
            "an off-diagonal chi1inv row makes update_E a STENCIL over the D "
            "volumes step_D writes in place (stepping.py:1235-1253), and no "
            "grid-wide barrier exists inside one launch; that row belongs to the "
            "complex folded off-diagonal product and this weld may not claim it")

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM, and on this cell it is the clause that decides everything. An
    # ELECTRIC source is injected BETWEEN the two halves (driver.py:3294-3299), so a
    # fused pair would consume a pre-injection D. BOTH corpus rows of this cell
    # declare exactly that, so with CARRIES_DEPOSIT_REPAIR False the arm would admit
    # NOTHING; it is True, and the plan below brackets its launch with the repair.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so an
    # undeclared `sources` is itself a REFUSAL. A MAGNETIC source is injected in the
    # B/H half and does not disqualify this pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    if codes is not None:
        # THE FAR CARRY'S OWNERSHIP MOVE. `fill_folded_far_ghosts_D` (driver.py:3302)
        # images the top stored slot from stepping._far_reflect_rows' row, and the
        # lane that owns that destination is the one AT the reflect row — so a row
        # outside the allocation is an out-of-range write from a lane that owns
        # neither cell, not a soft error on a whole-plane assignment. The three row
        # checks are symmetry.mirror_ghost_fill_coverage's own (symmetry.py:842-895),
        # restated here for the reason both fused folded families restate them.
        shape = tuple(getattr(grid, "shape", ()))
        rows = _far_reflect_rows(grid)
        for axis, code in enumerate(codes):
            periodic = int(code) == CODE_MIRROR_PERIODIC
            reader = _stored_past_owned_reader()
            if reader is None:  # pragma: no cover - folded_axis_kinds refused first
                reasons.append(
                    "stepping._stored_past_owned is not importable; the far carry "
                    "cannot be checked against the array path's own predicate")
                break
            past_owned = bool(reader(grid, axis))
            if periodic != past_owned:
                reasons.append(
                    f"axis {axis} code {int(code)} and "
                    f"stepping._stored_past_owned {past_owned} disagree; the kernel "
                    f"would mask the top plane and image the far ghost on different "
                    f"axes than the array path")
            if not periodic or len(shape) != 3:
                continue
            row = rows[axis] if rows is not None else None
            if row is None:
                reasons.append(
                    f"axis {axis} is a folded PERIODIC axis with no reflect row "
                    f"from stepping._far_reflect_rows; the far carry has nothing "
                    f"to image")
                continue
            row, extent = int(row), int(shape[axis])
            if not (0 <= row < extent - 1):
                reasons.append(
                    f"axis {axis} reflect row {row} is outside [0, {extent - 1}); "
                    f"the far carry would image the plane it writes, or read "
                    f"outside the allocation")
            if row == 0:
                reasons.append(
                    f"axis {axis} reflect row is 0, the plane the NEAR fill writes: "
                    f"the far carry would image a ghost rather than an owned cell")
            if extent - 1 == MIRROR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} stores {extent} cells, so the far carry's write "
                    f"plane IS the near fill's read plane "
                    f"(cell {MIRROR_SOURCE_INDEX})")

        # THE NEAR FILL IMAGES STORED CELL 2 FROM THAT CELL'S OWN LANE. The folded
        # complex curl predicate already refuses a folded axis storing two cells or
        # fewer; restated because THIS family writes the destination from a different
        # lane and a missing source plane is an out-of-range write, not a soft error.
        # BOTH mirror codes: the near fill runs on either.
        if len(shape) == 3:
            for axis, code in enumerate(codes):
                if (int(code) in MIRROR_CODES
                        and int(shape[axis]) <= MIRROR_SOURCE_INDEX):
                    reasons.append(
                        f"axis {axis} is folded with {int(shape[axis])} stored "
                        f"cells; the near fill images stored cell "
                        f"{MIRROR_SOURCE_INDEX} and this kernel images it from that "
                        f"cell's own lane")

        # THE WALL CLEAR AND THE FILLS MUST NOT MEET. On this family the clear's rows
        # ARE the near fill's axes — both are the component's shift-0 axes — and
        # `_zero_metal` skips a folded axis (:2237-2239), which is the ONLY thing
        # keeping them apart. CHECKED rather than inferred: an overlap would be a
        # plane of wrong values, and on the D side there is no second reason it could
        # not happen.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall "
                    f"there; stepping._zero_metal skips a folded axis, so the two "
                    f"passes have drifted and the carry's disjointness no longer "
                    f"holds")

        # THE PARITY WORDS THE KERNEL WILL BE PASSED must be readable on every folded
        # axis. `folded_axis_kinds` already refuses a folded axis whose phase is not
        # +-1; this re-derives the WORDS through the builder's own function so a pair
        # that came back (0.0, 0.0) on a folded axis is caught here and not on the
        # device, where it would write an exact zero plane.
        words = parity_coefficient_words(grid)
        for axis, code in enumerate(codes):
            if int(code) not in MIRROR_CODES:
                continue
            if words[axis] == (0.0, 0.0) or words[axis + 3] == (0.0, 0.0):
                reasons.append(
                    f"axis {axis} is folded but its parity coefficient words "
                    f"resolve to near={words[axis]!r} far={words[axis + 3]!r}; the "
                    f"carry would image an exact zero plane")

    # The wall clear is carried inline, so the grid must be able to answer which axes
    # are walled; an unanswerable one compiles to ZM=False and silently skips a plane
    # the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL at every owned cell AND at
    # every ghost, so `Fields` must be able to hand one over per E component. An
    # absent accessor would be a TypeError inside the builder rather than a refusal,
    # which is the wrong way for an uncovered configuration to fail. The MAGNETIC
    # twin needs no such clause: update_H is `H = B` with mu = 1 baked in and binds
    # no inverse volume at all.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half "
            "cannot bind the inv_eps volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedComplexFusedPairPlan:
    """One allocation-free launch for five of the driver's electric call sites.

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_fields.ComplexPmlCurlPlan` makes, for the same reason. The
    ``inv_eps`` volumes are the exception and are bound RAW: each is float32 with one
    real coefficient per complex cell, indexed at ``+ idx``.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal", "connections",
        "connection_values", "parity", "reflect", "near", "far",
        "chunk_bounds", "_cb",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients",
        "_grid", "_kernel",
    )

    #: The five driver call sites one launch performs. Declared, so a composition can
    #: be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, connections,
                 connection_values, parity, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 reflect: Any = (None, None, None),
                 kernel: Any = None, num_warps: Optional[int] = 1, *,
                 chunk_bounds: Any) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # and Triton types a Python float argument as fp32, so the two are the same
        # bits (complex_fields.ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        # SIX word pairs, near x/y/z then far x/y/z, host-rounded once.
        self.parity = tuple((float(pair[0]), float(pair[1])) for pair in parity)
        if len(self.parity) != 6:
            raise ValueError(
                f"the parity table is six (re, im) pairs — near x/y/z then far "
                f"x/y/z — got {len(self.parity)}")
        # DERIVED FROM `bc`, NEVER TAKEN AS AN ARGUMENT. The kernel branches its curl
        # half on the four codes and its two carries on these booleans; if they could
        # be passed independently they could disagree, and a kernel that masked the
        # top plane on one axis set and imaged the far ghost on another is a plane of
        # wrong values rather than a crash.
        self.near = tuple(code in MIRROR_CODES for code in self.bc)
        self.far = tuple(code == CODE_MIRROR_PERIODIC for code in self.bc)
        # -1 where no far ghost exists. A sentinel rather than 0: no lane reads it
        # (the guard that would is emitted only under FAR_a), and an out-of-range
        # index is loud where imaging row 0 would be plausible and wrong.
        self.reflect = tuple(-1 if value is None else int(value)
                             for value in reflect)
        for axis, code in enumerate(self.bc):
            folded = code in MIRROR_CODES
            if folded and self.parity[axis] == (0.0, 0.0):
                raise ValueError(
                    f"axis {axis} is folded but its NEAR parity words are "
                    f"(0.0, 0.0); the carry would image an exact zero plane")
            if folded and self.parity[axis + 3] == (0.0, 0.0):
                raise ValueError(
                    f"axis {axis} is folded but its FAR parity words are "
                    f"(0.0, 0.0); the carry would image an exact zero plane")
            if folded and self.connections[axis]:
                raise ValueError(
                    f"axis {axis} carries both a fold and a Bloch phase; "
                    f"driver._require_bloch_is_representable refuses that "
                    f"configuration outright and a kernel cannot lift what the "
                    f"array path will not run")
            if folded and self.zero_metal[axis]:
                raise ValueError(
                    f"axis {axis} carries both a fold and a wall clear; "
                    f"stepping._zero_metal skips a folded axis")
            if code == CODE_MIRROR_PERIODIC and not (
                    0 <= self.reflect[axis] < int(self.shape[axis]) - 1):
                raise ValueError(
                    f"axis {axis} is folded PERIODIC with reflect row "
                    f"{self.reflect[axis]}, which is outside "
                    f"[0, {int(self.shape[axis]) - 1}); the far carry would image "
                    f"the plane it writes, or read outside the allocation")
            if code != CODE_MIRROR_PERIODIC and self.reflect[axis] != -1:
                raise ValueError(
                    f"axis {axis} is not folded PERIODIC but carries reflect row "
                    f"{self.reflect[axis]}; stepping._far_reflect_rows answers None "
                    f"there and the kernel would read a row nothing wrote")
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._e_targets = tuple(CupyPointer(_word_view(a)) for a in e_targets)
        self._e_aux = tuple(CupyPointer(_word_view(a)) for a in e_aux)
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder "
                "binding would be read as a coefficient")
        # RAW, not word views: inv_eps stays float32 under complex storage
        # (stepping.py:37-38, fields.py:1203-1204) with one entry per COMPLEX cell.
        self._inverse_epsilon = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(a)) for a in e_coefficients)
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in int32,
        # so the complex cell count must leave room for the doubling — the same
        # halving the complex predicates apply. Refused here as well because the
        # from-arrays route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "FoldedComplexFusedPairPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which compile
        # deliberately broken copies of this kernel. Dropping it is not a slowdown,
        # it is a DISARMING — every mutation leg would then launch the shipped kernel
        # and report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_complex_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        # The parity is a real scale per part: each pair's real word (+-1.0); the
        # imaginary word is +0 and is not read.
        parity_words = tuple(pair[0] for pair in self.parity)
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.connection_values,
            self.reflect[0], self.reflect[1], self.reflect[2],
            *parity_words,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            NEAR_X=self.near[0], NEAR_Y=self.near[1], NEAR_Z=self.near[2],
            FAR_X=self.far[0], FAR_Y=self.far[1], FAR_Z=self.far[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            CNX=self.connections[0], CNY=self.connections[1],
            CNZ=self.connections[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"FoldedComplexFusedPairPlan(shape={self.shape}, "
                f"bc={self.bc}, zero_metal={self.zero_metal}, "
                f"connections={self.connections}, near={self.near}, far={self.far}, "
                f"reflect={self.reflect}, parity={self.parity}, block={self.block}, "
                f"num_warps={self.num_warps}, chunk_bounds={self.chunk_bounds.tolist()})")


def plan_folded_complex_fused_pair(
        fields: Any, pml: Any, sources: Any = None,
        block: Optional[int] = None, num_warps: Optional[int] = 1,
        kernel: Any = None, chunk_layout: Any = None,
) -> Optional[FoldedComplexFusedPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default
    (``launch.chunk_bounds_for``).
    """
    if not folded_complex_fused_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .launch import chunk_bounds_for  # noqa: PLC0415

    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    connections, values = connection_arguments(bloch_phase_table(grid, kinds),
                                               backward=bool(curl_spec["backward"]))
    return FoldedComplexFusedPairPlan(
        grid.shape, grid.dt / grid.dx, codes, zero_metal_axes(grid),
        connections, values, parity_coefficient_words(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        [fields.inverse_epsilon_for(name) for name in side_spec["targets"]],
        # The curl takes the INTEGER lattice on step_D and the constitutive the
        # HALF-INTEGER one on E (stepping.py:948 vs :1015). The kernel takes both and
        # never asks which is which, so a swap here is a silent half-cell error in
        # the absorber profile; the gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        # The far carry's image rows, from the engine's own function rather than
        # recomputed: `n_full - stored + 2` is `stored - 2` at an even full count and
        # `stored - 3` at an odd one (stepping._far_reflect_rows:1661).
        reflect=_far_reflect_rows(grid) or (None, None, None),
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_folded_complex_fused_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        phases: Sequence[Optional[complex]], parity, dtdx: float,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1,
        reflect: Any = (None, None, None), *, chunk_bounds: Any) -> FoldedComplexFusedPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``phases`` is
    the per-axis ``Optional[complex]`` Bloch table (None = unphased) and ``parity``
    the six ``(re, im)`` word pairs, near x/y/z then far x/y/z; ``arrays``
    additionally carries ``inv_eps_Ex``... as float32 volumes. The conjugation for a
    backward sub-step is applied HERE, exactly as the engine route applies it — this
    product is backward-only, so it always is.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(curl_spec["backward"]))
    return FoldedComplexFusedPairPlan(
        shape, dtdx, codes, zero_metal, connections, values, parity,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        reflect=reflect, kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
