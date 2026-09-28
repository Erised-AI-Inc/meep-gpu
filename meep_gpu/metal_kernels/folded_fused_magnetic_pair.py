"""The FOURTH fused Metal kernel: the folded ``step_B`` welded into ``update_H``.

THE PRODUCT THE MATRIX POINTS AT. Two facts, both measured, pick this arm and this
seam out of the twenty-five permutations:

* **the folded arm is the dominant one** — 53 of the 186 census rows
  (``parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19``), against 26 for
  PML/ordinary;
* **B->H is the productive seam** — the driver injects the ELECTRIC current
  between ``step_D`` and ``update_E`` (driver.py:3295-3299) and the MAGNETIC one
  between ``step_B`` and ``update_H`` (driver.py:3283-3284), and 157 of the 186
  rows declare an electric source against 58 a magnetic one. Closing the D/E seam
  on a folded grid admits **2** rows; closing the B/H seam on the same grids admits
  **45**.

Those two numbers are the same derivation over the same census, run as this family's
own admission count rather than quoted from :mod:`.folded_fused_pair`'s
counterfactual — ``parity/meep_gpu/results/metal_folded_fused_magnetic_pair_2026-08-19/``
``count_corpus_admission.py`` / ``corpus_admission.json``. The funnel, clause by
clause::

    folded B->H   77 curl -> 77 constitutive -> 54 no-magnetic-source
                  -> 54 fold-stores-row                             =  54
    folded D->E   77 curl -> 53 constitutive ->  4 no-electric-source
                  ->  2 every-fold-metallic ->  2 fold-stores-row   =   2

54 rows is twenty-seven times what the folded D/E product reaches and more than
double the 24 the unfolded :mod:`.fused_magnetic_pair` reaches, which is the whole
reason this module exists. THE +9 IS THE FAR CARRY OF 2026-08-20, recomputed rather
than claimed: ``results/fusion_matrix_metal_2026-08-20_farcarry/`` re-runs the
CLOSED cut with this one clause retired and lands on 127/387 against its 118, and it
re-evaluates the PRE-CARRY ladder on the same rows to reproduce the standing 45
exactly, so the delta is attributable to the clause and not to the method.

Dropping the magnetic-source clause entirely would reach 65, so THAT clause — not
the fold, not either fill, not the binding ceiling — is now what caps the family,
and it is reported beside the gate rather than left to be discovered.

ONE DISPATCH FOR ALL THREE COMPONENTS: 27 pointers plus one packed
``constant Params&``, 28 of the 31 bindings the platform allows. That is
:mod:`.fused_magnetic_pair`'s signature EXACTLY — the fold adds no argument, because
the stored extent is what carries it and the PML coefficient vectors are already
built at that extent — so :data:`SEPARATE_SCALAR_BINDINGS`,
:data:`PACKED_BINDINGS` and the refuted 32-binding source are IMPORTED from that
module rather than re-spelled here.

EVERYTHING HERE IS TRANSCRIBED, and the two halves are not retyped at all — they
are LIFTED FROM THE CERTIFIED EMITTERS' OWN OUTPUT and reassembled:

* the folded B curl — :func:`.symmetry.folded_curl_source` with ``backward=False``,
  whose head (guard, decode, ghost gather, curl grouping, cell-0 mask, top-plane
  slot) is spliced verbatim by :func:`certified_curl_head` and whose recurrence
  statements are pulled line by line by :func:`certified_curl_statements`;
* the wall clear — ``stepping._zero_metal`` (stepping.py:2206-2247) restricted to
  ``B_COMPONENTS``, through :data:`.fused_magnetic_pair._ZERO_METAL_ROWS`, which is
  IMPORTED: that table is already the B DIAGONAL and is already pinned;
* the near mirror fill — ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1426-1452)
  and ``stepping._write_mirror_ghost`` (:1451), with the parity spelled by
  :func:`.symmetry._parity_spelling`, the ONE measured place a mirror parity becomes
  Metal text;
* the constitutive half — :func:`.fused_magnetic_pair.certified_constitutive_body`,
  which is ``shaders.constitutive_source("H")``'s own body with the targets renamed
  ``f`` -> ``h``. This module re-emits it PER COMPONENT AT A PARAMETERISED INDEX
  (the owned cell and the imaged ghost cell take the identical seven statements at
  two different flat indices), and :func:`constitutive_transcription` COMPARES the
  re-emission against that lifted body statement for statement. The comparison is a
  measurement the gate runs, not a claim this docstring makes.

===========================================================================
THE SEAM, AND THE FOUR PASSES INSIDE IT — from the driver, not assumed
===========================================================================

``driver.step`` runs FIVE passes between ``step_B`` and ``update_H``
(driver.py:3281-3289, mirrored in :data:`.coverage.RESIDENCY_ORDER`)::

    step_B -> MAGNETIC SOURCES -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — REFUSED BY NAME**, and this is the clause that caps the
  family: 54 of the 77 rows that clear both halves declare no magnetic source.
  Ignorance is never an empty set — ``Fields`` does not hold the source list, so an
  undeclared ``sources`` is a REFUSAL, never an assumed ``()``. An ELECTRIC source
  is injected in the D/E half and does NOT disqualify this pair, which is the whole
  asymmetry the matrix rests on.
* **``fill_symmetry_bc_B`` — CARRIED INLINE, and it is NOT dead.** It writes
  ``cell 0 = parity * cell 2`` on every folded axis for every component whose Yee
  shift there is 0, ``update_H`` then reads exactly those cells, and nothing between
  the fill and the read touches them (next bullet). A pair that skipped it would
  feed the constitutive half a pre-fill B on one component of every folded axis.
* **``zero_metal_B`` — CARRIED INLINE**, through the imported B diagonal.
  ``_zero_metal`` SKIPS a folded axis (stepping.py:2284-2286:
  ``is_metallic(axis) and not is_mirrored(axis)``), so a walled axis is never a
  folded one.
* **``fill_folded_far_ghosts_B`` — CARRIED INLINE as of 2026-08-20.** It runs
  only on a folded PERIODIC axis (``stepping._stored_past_owned``:1454-1469) and
  images the LAST stored slot from a RUNTIME reflect row. It was REFUSED BY NAME
  until this round — that refusal is what capped this family on a folded PERIODIC
  axis — and carrying it brings the folded curl's top-plane block with it
  (``symmetry.folded_top_plane_mask`` now emits lines), which the lift picks up
  unchanged. The reflect rows ride in the packed ``Params``, so the binding count
  is UNCHANGED at 28 of the platform's 31.

===========================================================================
THE B GEOMETRY IS NOT THE D GEOMETRY — the whole of what is new here
===========================================================================

``fields.IYEE_SHIFTS`` (fields.py:214-219): ``Bx (0,1,1)``, ``By (1,0,1)``,
``Bz (1,1,0)`` against ``Dx (1,0,0)``, ``Dy (0,1,0)``, ``Dz (0,0,1)``. The near fill
touches component ``m`` on axis ``a`` exactly when ``iyee[m][a] == 0``, so

    D family:  a != m   (the two axes that are NOT the component's own)
    B family:  a == m   (the component's OWN axis, and only it)

THREE CONSEQUENCES, each moving in a different direction:

1. **THE COEFFICIENT INDEX MOVES**, and this is the one property
   :mod:`.folded_fused_pair` is built on that is FALSE here. ``update_H`` indexes
   ``kps``/``kms`` on the component's own axis (``shaders._CONSTITUTIVE_TEMPLATE``,
   ``stepping.H_CONSTITUTIVE_TERMS``:226), and on the B half that is exactly the
   axis the fill images along — so the fill's SOURCE (stored 2) and DESTINATION
   (stored 0) take DIFFERENT coefficient entries. Not fatal: the source thread loads
   the destination's pair explicitly at index 0, two extra scalar loads
   (:func:`_ghost_coefficient_lines`).

   **AND IT IS USUALLY UNOBSERVABLE, WHICH IS A MEASURED FINDING AND NOT A
   REASSURANCE.** A folded axis carries no absorber at the mirror plane —
   ``stepping._require_consistent_pml`` admits the HIGH face only — so a layer has to
   be deep enough to reach stored cell 2 before the two entries differ at all. On the
   shared fixture's 2-cell layer they are the same word to the bit
   (``kps_x = [1, 1, ..., 1.567]`` on an 11-cell folded axis), and a kernel that
   reused the source's pair would be BYTE-IDENTICAL there. The gate therefore does
   not merely arm that mutation: leg ``moved_coefficient`` censuses the coefficient
   vectors, PREDICTS observability from them, and requires ``caught == observable``
   on a case of each kind — measured uncaught at 2 cells and caught at 4 cells on a
   5-cell folded axis. A mutation leg that simply reported "uncaught" here would have
   left a reader unable to tell a blind fixture from a dead line.
2. **NO PARITY PRODUCTS FROM THE NEAR FILL — THE FAR ONE MAKES THEM.** A B
   component is a NEAR destination on at most ONE axis, so
   ``_fill_symmetry_ghost_cells``' X/Y/Z application order (:1441-1447) is
   unobservable on that half; :func:`near_fill_axes` returns at most one axis and
   the source builder REFUSES a longer answer rather than assuming it cannot happen.
   :func:`far_fill_axes` is that set's exact COMPLEMENT and returns up to two, so a
   cell at the top of both far planes carries the PRODUCT of two parities and a cell
   that is a near destination on one axis and a far one on another carries the
   product of theirs. :func:`carried_destinations` is the closed form, and it is
   MEASURED against the array path's own three passes over nine configurations —
   zero differing words, including the three-axis fold where one source thread owns
   SEVEN ghost cells for one component
   (``results/metal_folded_far_carry_2026-08-20/ownership.json``).
3. **THE WALL CLEAR AND THE NEAR FILL CANNOT MEET; THE FAR GHOST INHERITS IT.**
   ``_zero_metal`` clears component ``m`` at stored cell 0 of axis ``a`` when
   ``iyee[m][a] == 0`` — ``a == m`` again, the same cell set the NEAR fill writes —
   and it skips a folded axis. So on that half the two passes are mutually exclusive
   PER COMPONENT, the D side's "parity-then-clear" ordering question is VACUOUS, and
   the near ghost takes no wall line at all. A FAR ghost moves along an axis that is
   NOT the component's own, and ``_ZERO_METAL_ROWS`` is the B DIAGONAL, so it sits
   at the same coordinate on every axis that could clear it as the thread that owns
   it: its clear status is that thread's, INHERITED by reading ``v`` after the clear
   lines rather than re-emitted, which is the driver's own order (:3286 before
   :3287). The source builder ASSERTS that emptiness
   rather than inferring it from the table's construction.

===========================================================================
THE OWNERSHIP RESTRUCTURE — one dispatch, no barrier, no race
===========================================================================

The near fill reads a cell THIS DISPATCH WRITES. Metal has no device-wide barrier,
so the naive carry — the thread at stored cell 0 reading stored cell 2 — races on
``B``. Ownership moves instead. For component ``m`` on folded axis ``m``:

* the thread at coordinate 0 is the DESTINATION. It computes the curl and the
  split-field ``n`` and stores ``u`` (the array path's ``step_B`` writes ``fu`` at
  every cell and the fill does not touch it), and then STOPS: the whole per-component
  tail — forming ``v``, reading ``f``, writing ``f``, and every touch of ``H`` and
  ``f_w_H`` — sits inside ``if (!(coord == 0)) { ... }``, so no load of a word another
  thread writes is issued;
* the thread at coordinate 2 is the SOURCE. It does its own cell in full and then
  writes the destination as well: ``parity * v``, the displacement stored, and the
  constitutive update performed there AT THE DESTINATION'S OWN COEFFICIENT INDEX.

Every destination word is written by exactly one thread, no thread reads a word
another thread writes, and no barrier is needed.

===========================================================================
THE TWO YEE SUB-LATTICES, AND WHY THEY ARE CROSSED HERE
===========================================================================

The B curl reads the HALF-INTEGER split-field coefficients ``kms_a_h``/``sinv_a_h``
and the H constitutive reads the INTEGER ``kps_a``/``kms_a``
(``SUB_STEPS['step_B']['suffix'] == '_h'``; ``CONSTITUTIVE_SIDES['H']['half_integer']
is False``). That is the OPPOSITE pairing to :mod:`.folded_fused_pair`. The kernel
takes twelve coefficient pointers and never asks which lattice they came from, so a
swap on either group is a silent half-cell error in the absorber profile —
converged, smooth and wrong — and the gate carries one host mutation for each group.

DEVICE STATUS: GATED ON THIS HOST, RE-CUT 2026-08-20 FOR THE FAR CARRY. The gate is
``parity/meep_gpu/gate_metal_folded_fused_magnetic_pair.py`` and it runs end to end
on the local Apple GPU: SEVENTEEN configurations x twelve COMPLETE driver steps
compared as uint32 words against an array-path oracle running the identical live
pass list — nine of them folded PERIODIC, including two folded axes at BOTH
full-count parities and a three-axis fold where one source thread owns seven ghost
cells — four separate-composition controls, the carry leg, two observability legs
whose predictions must hold, thirty-nine armed shader defects on BOTH fold
terminations, two host-binding defects, and a disarm run per mutation case. The
release verdict is shown to FLIP against two planted defects by
``results/metal_folded_far_carry_2026-08-20T2/flip_the_release_verdict.py``. The
artifact is ``results/metal_folded_far_carry_2026-08-20T2/gate.json``; the
``...2026-08-20/`` directory beside it holds the FIRST cut's run (a PASS on the
bytes before this docstring was corrected) and the two host measurements the carry
rests on, ``ownership.json`` and ``grid_reflect_row_law.json``. Each run gets its
own directory because the campaign manifest binds a directory to one set of source
digests — a second run into the first directory is refused, and that refusal is how
this one was found. No claim in this docstring rests on an unrun measurement.

WHAT THE TRITON TWIN'S GATE FOUND AND THIS ONE STRUCTURALLY CANNOT, recorded here
because it is the same product on another board: there the carry masks are separate
lane predicates and the first cut forgot to AND them with the ownership mask, so a
lane that is the SOURCE of one fill and the DESTINATION of the other raced the
legitimate owner. On this backend the carries sit INSIDE the owned-cell guard
(``if (!(i == 0 || last_y))``), so the same slip is not expressible.

STILL ``wired=False``, AND SINCE 2026-08-27 ROUTED ANYWAY — two different questions,
and this paragraph used to answer only the first. ``STEP_ORDER`` assigns at most one
ARM per slot and this product spans FIVE of them (``step_B``, ``fill_B``,
``zero_metal_B``, ``fill_folded_far_ghosts_B``, ``update_H``), so it can claim no
slot through the arm table and goes on registering ``wired=False``: ``arms.arms_for``
skips it and ``_select_slot`` cannot pick it, while the disjointness sweep can still
enumerate it through ``arms.registered``.

What it now has is the OTHER route. ``launch._install_fused_pairs``
(metal_kernels/launch.py:725-800) walks ``arms.registered("step_B")`` for ``is_weld``
rows whose ``replaces`` covers the partner slot, and ``launch.FUSED_PAIR_ARMS``
(launch.py:660-664) now carries the row ``folded_fused_magnetic_pair: ("folded",
"folded")`` — the declaration of which arm each of the two slots implements, read off
this module's own predicate, which is exactly the ``folded`` curl arm's AND the
``folded`` constitutive arm's. Under ``plan_step(..., fuse=True)`` on a folded grid
this product therefore takes ``step_B`` and ``update_H``. The row licenses the
SUBSTITUTION and nothing else: the deposit repair is declared separately by
:data:`CARRIES_DEPOSIT_REPAIR`, which became ``True`` on 2026-08-28 once the repair
gained the closure of the cells a post-injection fill images a deposit into.

NOTHING ON THIS BACKEND DISPATCHES. ``meep_gpu.fastpath.plan_fast_path`` consults
``triton_kernels.plan_step`` alone and names no Metal module on any branch, so a
Metal plan — fused or not — is still built for measurement rather than for a run.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..fields import mirror_parity
from ..stepping import _far_reflect_rows, _stored_past_owned
from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    MAGNETIC_FIELD_TYPE,
    _call,
    zero_metal_axes,
)
from ..triton_kernels.launch import SUB_STEPS
from ..triton_kernels.symmetry import (
    # CODE_MIRROR_METALLIC is deliberately NOT imported here. It was, until
    # 2026-08-21, and nothing in this module referenced it: the metallic
    # termination is recognised through `MIRROR_CODES` from `.symmetry` below,
    # which is the set both terminations belong to, and only CODE_MIRROR_PERIODIC
    # is ever compared against on its own (it is what gates the far carry). An
    # unused name imported from the module a predicate reads reads as a second
    # place the termination is decided.
    CODE_MIRROR_PERIODIC,
    MIRROR_SOURCE_INDEX,
    TARGET_IYEE,
    folded_axis_kinds,
)
from . import shaders, templates
from .device import Residency, compile_source
from .fused_magnetic_pair import (
    PACKED_BINDINGS,
    SEPARATE_SCALAR_BINDINGS,
    _ZERO_METAL_ROWS,
    certified_constitutive_body,
    refuted_separate_scalar_source,
)
from .plans import KernelPlan
from .symmetry import (
    MIRROR_CODES,
    _parity_spelling,
    folded_composition_curl_coverage,
    folded_constitutive_coverage,
    folded_curl_source,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
#:
#: TRUE SINCE 2026-08-28, AND WHAT HAD TO BE BUILT FIRST. This is a FOLDED seam, so it
#: also runs ``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` AFTER the injection
#: (driver.py:3293-3295) -- and a repair that wrote only the
#: deposit index left every MIRROR IMAGE of that index carrying the constitutive result
#: the launch computed from the pre-injection field. That was measured, not assumed: on
#: a corpus row with the deposit one cell off the plane, the image cell came back
#: byte-equal to a run with no source at all. The flag moved because
#: ``deposit_repair.repair_cells`` now hands ``save``/``apply`` the CLOSURE of the cells
#: the two fills image each deposit point into -- the inverse of the forward carry
#: :func:`carried_destinations` already implements -- and the seam is byte-identical to
#: the array path over ``step_B`` -> ``update_H`` with it and diverges without it.
#: What is still refused by name: a source that publishes no deposit index, and a folded
#: grid whose fill map ``deposit_repair`` cannot read (the cylindrical r = 0 axis, a folded axis too short to hold the near fill's source row).
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_fused_magnetic_pair"

#: The sub-step slot this arm holds a row on. It spans four; a refusal is NAMED on
#: the slot the fusion starts at rather than being invisible to the table.
SLOT = "step_B"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
#: ``BACKWARD`` is the direction ``SUB_STEPS[CURL_SUB_STEP]`` declares, restated as
#: a bool so the one legal binding is visible without importing :mod:`launch`; a
#: test pins the two equal.
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"
BACKWARD = False

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3281-3289) and in :data:`.coverage.RESIDENCY_ORDER`'s spelling. Read
#: by the whole-step gate's ``covered_passes``; declared, never inferred from the
#: slot name.
REPLACES: Tuple[str, ...] = ("step_B", "fill_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: The stored index the near fill images, re-exported so a reader of this module
#: does not have to chase ``triton_kernels.symmetry`` for the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX

#: The reflect row's runtime name per axis inside the kernel, and the top-plane
#: predicate the lifted curl head already declares. Both are needed by the far
#: carry and neither is a new binding: the rows ride in the packed ``Params``.
_REFLECT: Tuple[str, str, str] = ("reflect_x", "reflect_y", "reflect_z")

#: The TOP STORED INDEX per axis and the boolean the lifted curl head declares for
#: it. ``_LAST`` is the integer the far destination's flat index is built from;
#: ``_LAST_FLAG`` is ``symmetry._TOP_PLANE_FLAGS``' own name, which the head already
#: declares for the top-plane curl mask and which this family reuses rather than
#: writing a second comparison against the same extent.
_LAST: Tuple[str, str, str] = ("(nxi - 1)", "(nyi - 1)", "(nzi - 1)")
_LAST_FLAG: Tuple[str, str, str] = ("last_x", "last_y", "last_z")

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY",
    "NEAR_SOURCE_INDEX", "PACKED_BINDINGS", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "MetalFoldedFusedMagneticPairPlan",
    "carried_destinations", "certified_curl_head", "certified_curl_statements",
    "compile_folded_fused_magnetic_pair", "constitutive_transcription",
    "far_fill_axes", "folded_fused_magnetic_pair_source",
    "metal_folded_fused_magnetic_pair_coverage", "near_fill_axes",
    "plan_metal_folded_fused_magnetic_pair", "refuted_separate_scalar_source",
    "zero_metal_lines",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

// EIGHT SCALARS, ONE BINDING, AND THE BINDING COUNT IS UNCHANGED AT 28. The three
// reflect rows the far carry needs (stepping._far_reflect_rows:1661) ride inside
// this record rather than as three more buffers, so carrying
// fill_folded_far_ghosts_B costs nothing against the platform's 31-binding ceiling
// (device.py:72). They stay RUNTIME for the certified separate fill's own reason
// (symmetry._MIRROR_FILL_TEMPLATE): the row is per-axis and integer, so unlike the
// parity — which is a SOURCE specialisation because a runtime float weight flushes
// this backend's subnormals — nothing about it can move a bit.
struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;
                int rx; int ry; int rz; };

kernel void folded_fused_magnetic_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device float*       h0      [[buffer(9)]],
    device float*       h1      [[buffer(10)]],
    device float*       h2      [[buffer(11)]],
    device float*       w0      [[buffer(12)]],
    device float*       w1      [[buffer(13)]],
    device float*       w2      [[buffer(14)]],
    device const float* kmx     [[buffer(15)]],
    device const float* sinvx   [[buffer(16)]],
    device const float* kmy     [[buffer(17)]],
    device const float* sinvy   [[buffer(18)]],
    device const float* kmz     [[buffer(19)]],
    device const float* sinvz   [[buffer(20)]],
    device const float* kp0     [[buffer(21)]],
    device const float* km0     [[buffer(22)]],
    device const float* kp1     [[buffer(23)]],
    device const float* km1     [[buffer(24)]],
    device const float* kp2     [[buffer(25)]],
    device const float* km2     [[buffer(26)]],
    constant Params&    prm     [[buffer(27)]],
    uint idx [[thread_position_in_grid]])
{
    // THE FIVE SCALARS ARE UNPACKED INTO THE LIFTED BODIES' OWN NAMES, once,
    // before any of the spliced text runs. Everything below this line is then
    // character-for-character what the folded curl emitter and the certified H
    // constitutive emitter produce, plus the wall clear and the mirror carry.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    int reflect_x = prm.rx, reflect_y = prm.ry, reflect_z = prm.rz;
    (void)reflect_x; (void)reflect_y; (void)reflect_z;

__BODY__
}
"""

#: The marker that separates a certified shader's signature from its body. Every
#: template in this package ends its parameter list with this exact line, so ONE
#: anchor lifts any body and a template that stopped carrying it raises here rather
#: than splicing a truncated kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The comment line the folded curl body opens its split-field block with. This is
#: the CUT POINT of the lift: everything above it is spliced verbatim, everything
#: below is re-emitted per component around the ownership carve-out. Matched as a
#: PREFIX so the block's trailing dash run is not a second thing to keep in step.
_RECURRENCE_MARK = "    // --- split-field recurrence"

#: The flat-index coordinate name per axis, and the stride expression per axis, in
#: the lifted decode's own spelling. ``nyz`` and ``nzi`` are that block's variables;
#: a second decode here would be a second place to get the layout wrong.
_COORDINATE: Tuple[str, str, str] = ("i", "j", "k")
_STRIDE: Tuple[str, str, str] = ("nyz", "nzi", "1")

#: The seven statements ``_apply_constitutive_pml`` runs for one component, as they
#: appear in the certified H body once the targets are renamed ``f`` -> ``h``.
#: Parameterised on the flat index, the source expression, the coefficient pair and
#: a register prefix — the ghost cell takes the identical seven at a different index
#: with a DIFFERENT coefficient pair, which is this family's whole delta from
#: :mod:`.folded_fused_pair`.
#:
#: NOT A RETYPING: :func:`constitutive_transcription` renders these with the
#: certified names and compares them statement for statement against
#: :func:`.fused_magnetic_pair.certified_constitutive_body`'s own text, and
#: :func:`folded_fused_magnetic_pair_source` runs that comparison on every build.
_CONSTITUTIVE_STATEMENTS: Tuple[str, ...] = (
    "float {prefix}prev = w{target}[{index}];",
    "float {prefix}src = {source};",
    "w{target}[{index}] = {prefix}src;",
    "float {prefix}acc = h{target}[{index}];",
    "{prefix}acc = {prefix}acc + {kp} * {prefix}src;",
    "{prefix}acc = {prefix}acc - {km} * {prefix}prev;",
    "h{target}[{index}] = {prefix}acc;",
)

#: How the certified H body spells each of the seven for component ``t``: the
#: register names ``prev{t}`` / ``src{t}`` / ``a{t}``, the flat index ``ii``, the
#: source ``g{t}[ii]`` and the coefficient pair ``kp_{t}`` / ``km_{t}``. Rendering
#: :data:`_CONSTITUTIVE_STATEMENTS` with these must reproduce that body exactly.
_CERTIFIED_PREFIX = "@"
_CERTIFIED_NAMES: Dict[str, str] = {
    "@prev": "prev{target}", "@src": "src{target}", "@acc": "a{target}",
}


def near_fill_axes(codes: Sequence[int], target: int) -> Tuple[int, ...]:
    """Which folded axes ``fill_symmetry_bc_B`` writes for ONE B component.

    ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1441-1447) fills axis ``a``
    for component ``m`` exactly when ``a`` is mirrored and ``iyee[m][a] == 0``. Read
    off ``TARGET_IYEE`` rather than spelled: a literal here would be another
    transcription of the Yee table, and this predicate decides which threads own
    which cells.

    For the B family the answer is at most ONE axis — the component's own — and
    :func:`folded_fused_magnetic_pair_source` refuses a longer one rather than
    trusting that. BOTH mirror codes are matched: the near fill runs on any folded
    axis (``stepping._fill_symmetry_ghost_cells`` gates on ``_mirror_phases``
    alone), which is exactly what separates it from :func:`far_fill_axes`.
    """
    names = SUB_STEPS[CURL_SUB_STEP]["targets"]
    return tuple(axis for axis, code in enumerate(codes)
                 if int(code) in MIRROR_CODES
                 and TARGET_IYEE[names[target]][axis] == 0)


def far_fill_axes(codes: Sequence[int], target: int) -> Tuple[int, ...]:
    """Which axes ``fill_folded_far_ghosts_B`` images for ONE B component.

    ``stepping._fill_folded_far_ghosts`` (stepping.py:1516-1534) writes axis ``a``
    for component ``m`` exactly when ``_stored_past_owned(grid, a)`` — a folded
    PERIODIC axis, at either full-count parity — and ``iyee[m][a] == 1``. The two
    conditions are read here off ``CODE_MIRROR_PERIODIC`` (which
    ``folded_axis_kinds`` derives from ``_stored_past_owned`` itself, and
    cross-checks against ``grid.is_metallic``) and off ``TARGET_IYEE``, so neither
    the fold split nor the Yee table is transcribed a second time.

    THE EXACT COMPLEMENT OF :func:`near_fill_axes` ON THIS FAMILY, and that is the
    whole geometric delta the far carry adds. A B component's Yee shift is 0 on its
    OWN axis and 1 on the other two, so the near fill touches component ``m`` on
    axis ``m`` alone while the far fill touches it on the two axes that are NOT its
    own — up to TWO destination planes per component, against the near fill's one,
    and a corner where both fire carrying the PRODUCT of the two parities.
    """
    names = SUB_STEPS[CURL_SUB_STEP]["targets"]
    return tuple(axis for axis, code in enumerate(codes)
                 if int(code) == CODE_MIRROR_PERIODIC
                 and TARGET_IYEE[names[target]][axis] == 1)


def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], bool], ...]:
    """Every ghost cell ONE source thread owns, as ``(far axes at top, near?)``.

    THE OWNERSHIP RULE, AND IT IS MEASURED RATHER THAN ARGUED
    (``parity/meep_gpu/results/metal_folded_far_carry_2026-08-20/ownership.json``).
    The driver runs the near fill, the wall clear and the far fill in that order
    (driver.py:3285-3287), and the far fill is applied axis by axis, so a cell the
    fills leave at the top of SEVERAL folded periodic axes is written more than
    once. Composing the passes gives one closed form: for component ``m`` the seam
    leaves, at the cell that sits at ``last`` on the far-axis set ``T`` and at
    stored 0 on the near axis when ``z``,

        (prod over a in T of mirror_parity(m, a, phase_a))
            * (z ? mirror_parity(m, near, phase_near) : 1)
            * v_m(that cell with every a in T moved to reflect_a, near moved to 2)

    — independent of the order the axes are applied in, which is why nothing here
    has to model ``_fill_folded_far_ghosts``' X/Y/Z loop. So ONE thread, the one at
    the fully back-substituted cell, computes the displacement every one of those
    ghosts carries, and it writes them all. Every destination word is written by
    exactly one thread and no thread reads a word another writes, which is what
    makes the carry legal inside a single Metal dispatch with no device-wide
    barrier.

    Returned smallest-subset-first for a stable emission order; the order is
    unobservable (the cells are distinct) and a stable one keeps a source diff
    readable.
    """
    far = tuple(int(axis) for axis in far)
    combos: List[Tuple[Tuple[int, ...], bool]] = []
    for size in range(len(far) + 1):
        for subset in itertools.combinations(far, size):
            for carries_near in ((False, True) if near else (False,)):
                if not subset and not carries_near:
                    continue  # the thread's OWN cell, not a ghost
                combos.append((subset, carries_near))
    return tuple(sorted(combos, key=lambda item: (len(item[0]), item[0], item[1])))


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    The lift's whole safety property. A missing anchor is a certified emitter that
    has changed under this family, and splicing around it would produce a kernel
    that compiles and is quietly not the certified arithmetic; TWO matches would
    mean the anchor no longer identifies a single statement.
    """
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this family LIFTS that line rather than "
            f"retyping it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the "
            f"lift of {what} would take an arbitrary one")
    return matches[0]


def _body_of(source: str, what: str) -> str:
    """One certified shader's body: everything between its signature and its brace."""
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            f"the certified {what} source no longer carries the body anchor; this "
            f"family lifts that body and would otherwise splice a truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"the certified {what} source does not end with '}}'")
    return body[: -len("}\n")]


def certified_curl_head(codes: Sequence[int],
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The folded ``step_B`` curl body DOWN TO the split-field recurrence, verbatim.

    Not a transcription: this is :func:`.symmetry.folded_curl_source`'s own output
    with the ``#include``/signature preamble removed and the cut taken at
    :data:`_RECURRENCE_MARK`. It carries the guard, the index decode, the ghost
    gather (``templates.ghost`` through ``symmetry._reduced_codes``), the curl
    grouping, the cell-0 ownership mask (``templates.ownership_mask``, the B
    DIAGONAL at ``backward=False``) and the top-plane slot — which
    :func:`.symmetry.folded_top_plane_mask` leaves as a comment here, because this
    family refuses ``MIRROR_PERIODIC``.

    ``backward`` is :data:`BACKWARD` and never a parameter: this pair is the B seam,
    and ``step_D``'s negated strides are a different product with a different fill
    geometry entirely.
    """
    body = _body_of(folded_curl_source(codes, BACKWARD, contract), "folded curl")
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(_RECURRENCE_MARK):
            head = "".join(lines[:index])
            if not head.strip():
                raise AssertionError("the lifted folded curl head is empty")
            return head
    raise AssertionError(
        f"the certified folded curl body no longer opens its split-field block "
        f"with {_RECURRENCE_MARK!r}; this family cuts the lift there")


def certified_curl_statements(codes: Sequence[int],
                              contract: str = shaders.CONTRACT_OFF,
                              ) -> Dict[str, Any]:
    """The split-field recurrence's own lines, pulled out of the certified body.

    Every arithmetic string this family emits below the cut comes from HERE rather
    than from a keyboard: the coefficient loads, the three ``p``/``n``/``v``
    statements and the two store lines are the certified folded curl's, matched by
    an anchor that must identify exactly one line each.

    ``stores`` is the ``f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;`` triple SPLIT into
    its three statements, because this family emits them one per component inside
    that component's ownership block rather than as one line.
    """
    body = _body_of(folded_curl_source(codes, BACKWARD, contract), "folded curl")
    coefficients = [_line_starting(body, f"    float km_{axis} = ",
                                   f"the {axis} split-field coefficient load")
                    for axis in "xyz"]
    recurrence = []
    for target in range(3):
        recurrence.append((
            _line_starting(body, f"    float p{target} = ",
                           f"target {target}'s previous split field"),
            _line_starting(body, f"    float n{target} = ",
                           f"target {target}'s split-field recurrence"),
            _line_starting(body, f"    float v{target} = ",
                           f"target {target}'s stepped displacement"),
        ))
    aux_store = _line_starting(body, "    u0[ii] = ", "the three fu stores")
    flux_store = _line_starting(body, "    f0[ii] = ", "the three flux stores")
    stores = tuple(f"{part.strip()};" for part in flux_store.strip().split(";")
                   if part.strip())
    if len(stores) != 3:
        raise AssertionError(
            f"the certified folded curl no longer stores three targets on one "
            f"line: {flux_store!r}")
    return {"coefficients": tuple(coefficients), "recurrence": tuple(recurrence),
            "aux_store": aux_store, "stores": stores}


def _constitutive_coefficient_lines(contract: str = shaders.CONTRACT_OFF
                                    ) -> Tuple[str, ...]:
    """``kp_t``/``km_t`` on the component's OWN axis, lifted from the certified body.

    ``shaders._CONSTITUTIVE_TEMPLATE`` indexes target 0 on ``i``, 1 on ``j`` and 2
    on ``k`` (``stepping.H_CONSTITUTIVE_TERMS``:226) — MEEP's ``dsigw``, NOT the
    dsig/dsigu cycle the curl recurrence uses. On THIS family the imaged ghost cell
    does NOT reuse these, which is the delta :func:`_ghost_coefficient_lines`
    carries.
    """
    body = certified_constitutive_body(contract)
    return tuple(_line_starting(body, f"    float kp_{target} = ",
                                f"target {target}'s constitutive coefficient pair")
                 for target in range(3))


def _render_constitutive(target: int, index: str, source: str, kp: str, km: str,
                         prefix: str) -> Tuple[str, ...]:
    """The seven constitutive statements for one component at one cell, unindented."""
    return tuple(line.format(target=target, index=index, source=source,
                             kp=kp, km=km, prefix=prefix)
                 for line in _CONSTITUTIVE_STATEMENTS)


def constitutive_transcription(contract: str = shaders.CONTRACT_OFF
                               ) -> Dict[str, Any]:
    """Is :data:`_CONSTITUTIVE_STATEMENTS` the certified H body, statement for statement?

    THE MEASUREMENT THAT MAKES THE PARAMETERISATION HONEST. This family re-emits the
    certified constitutive arithmetic twice per folded component — once at the owned
    cell and once at the imaged ghost — so it cannot splice that body wholesale the
    way :mod:`.fused_magnetic_pair` does. What it can do is render its own template
    with the CERTIFIED spellings and require the result to be exactly the lifted
    body's statements, which is what this returns.

    ``identical`` false is a build-time failure, not a diagnostic:
    :func:`folded_fused_magnetic_pair_source` calls this and raises.
    """
    body = certified_constitutive_body(contract)
    statements = [line.strip() for line in body.splitlines()
                  if line.strip() and not line.strip().startswith("//")]
    certified: List[List[str]] = []
    emitted: List[List[str]] = []
    for target in range(3):
        rendered = list(_render_constitutive(
            target, "ii", f"g{target}[ii]", f"kp_{target}", f"km_{target}",
            _CERTIFIED_PREFIX))
        for token, spelling in _CERTIFIED_NAMES.items():
            rendered = [line.replace(token, spelling.format(target=target))
                        for line in rendered]
        emitted.append(rendered)
        certified.append([line for line in statements if _touches(line, target)])
    return {
        "emitted": [list(rows) for rows in emitted],
        "certified": [list(rows) for rows in certified],
        "identical": all(emitted[t] == certified[t] for t in range(3)),
    }


def _touches(statement: str, target: int) -> bool:
    """Does one certified constitutive statement belong to component ``target``?

    The certified H body is three coefficient loads followed by three seven-line
    component blocks, and every line of a block names its component in a register
    (``prev0``, ``src0``, ``a0``) or in a pointer (``w0``, ``h0``). The coefficient
    loads name TWO registers each and are excluded by the ``float kp_`` prefix,
    which is where this family lifts them separately.
    """
    if statement.startswith("float kp_"):
        return False
    marks = (f"prev{target}", f"src{target}", f"a{target} ", f"w{target}[",
             f"h{target}[")
    return any(mark in statement for mark in marks)


def zero_metal_lines(target: int, zero_metal: Sequence[bool], name: str,
                     indent: str = "    ", zero: str = "0.0f") -> List[str]:
    """``stepping.zero_metal_B`` for ONE target, on one named register.

    The rows are :data:`.fused_magnetic_pair._ZERO_METAL_ROWS`, IMPORTED rather than
    re-derived: that table is already the B family's Yee DIAGONAL (Bx on an x wall,
    By on y, Bz on z) and is already pinned against the array path.

    The select spelling is :func:`.templates.ownership_mask`'s, ``flag ? 0.0f : v``,
    the form measured to deliver an exact ``+0.0`` on this platform.
    """
    return [f"{indent}{name} = {flag} ? {zero} : {name};"
            for row_target, axis, flag in _ZERO_METAL_ROWS
            if row_target == target and bool(zero_metal[axis])]


def _ghost_coefficient_lines(target: int, axis: int, tag: str,
                             indent: str) -> List[str]:
    """The NEAR destination's constitutive coefficient pair, read at stored index 0.

    THE ONE THING :mod:`.folded_fused_pair` DOES NOT HAVE TO DO. There the fill's
    source and destination differ only along an axis that is never the component's
    own, so they share a coefficient entry; the NEAR fill here images along the
    component's OWN axis (``iyee[Bm][a] == 0`` iff ``a == m``), which is exactly the
    axis ``update_H`` indexes on. Reusing the source thread's ``kp_t``/``km_t`` would
    apply the absorber profile of stored cell 2 to stored cell 0 — smooth, converged
    and wrong inside the PML, invisible outside it — so the pair is loaded again at
    index 0. The gate arms that reuse as a mutation.

    THE FAR FILL NEEDS NO SUCH LINE, and that asymmetry is derived rather than
    asserted: ``shaders._CONSTITUTIVE_TEMPLATE`` indexes target ``t`` on axis ``t``
    (``stepping.H_CONSTITUTIVE_TERMS``:226) and :func:`far_fill_axes` returns only
    axes ``a != t``, so a far destination sits at the SAME coordinate on the
    indexed axis as the thread that owns it and takes that thread's own pair
    unchanged. A carry that reloaded there would be reloading the identical word.
    """
    if axis != target:
        raise AssertionError(
            f"component {target} images its NEAR ghost on axis {axis}; for the B "
            f"family the near fill only ever touches a component's OWN axis "
            f"(fields.IYEE_SHIFTS)")
    name = _COORDINATE[axis]
    return [
        f"{indent}// The DESTINATION's own coefficient entry: stored index 0 on "
        f"{name}, NOT the",
        f"{indent}// source thread's at {NEAR_SOURCE_INDEX} "
        f"(shaders._CONSTITUTIVE_TEMPLATE indexes target "
        f"{target} on {name}).",
        f"{indent}float {tag}_kp = kp{target}[0], "
        f"{tag}_km = km{target}[0];",
    ]


def _constitutive_block(target: int, index: str, source: str, kp: str, km: str,
                        prefix: str, indent: str) -> List[str]:
    """``_apply_constitutive_pml`` for one component at one cell, indented."""
    return [indent + line for line in
            _render_constitutive(target, index, source, kp, km, prefix)]


def _component_parity(target: int, axis: int, phase: int) -> int:
    """``fields.mirror_parity`` for one B component about one plane — the ONE table.

    ``stepping._symmetry_phase`` (stepping.py:2457-2471) is exactly this call, and
    both fills weight their image with it. Not spelled as ``+phase`` for the near
    fill and ``-phase`` for the far one: that collapse is TRUE (``mirror_parity``
    is ``phase * (1 - 2 * iyee)`` for a B component about its own axis,
    fields.py:117) but writing it out here would be a second copy of the Yee table
    with a sign in it, and the far carry is the change that makes both signs live
    in the same kernel.
    """
    name = SUB_STEPS[CURL_SUB_STEP]["targets"][target]
    return int(mirror_parity(name, axis, int(phase)))


def _carry_blocks(target: int, near: Sequence[int], far: Sequence[int],
                  phases: Sequence[int], zero_metal: Sequence[bool],
                  indent: str) -> List[str]:
    """Every imaged ghost this thread owns, then ``update_H`` at each of them.

    ONE BLOCK PER DESTINATION IN :func:`carried_destinations`. Each is guarded on
    the flags that say this thread is the source of that destination — stored
    ``NEAR_SOURCE_INDEX`` on the near axis, the runtime reflect row on each far
    axis — and writes ``parity * v`` at the destination's flat index, followed by
    the certified constitutive at that index.

    ``v{target}`` IS READ AFTER THE WALL CLEAR, deliberately, and the driver order
    is why: ``zero_metal_B`` (driver.py:3286) runs BEFORE
    ``fill_folded_far_ghosts_B`` (:3287), so a far image of a cleared row must
    carry the cleared value. The near fill runs BEFORE the clear (:3285), but its
    destination (stored 0 of a FOLDED axis) and the clear's (stored 0 of a WALLED
    axis) are disjoint per component, so reading the same post-clear register for
    both is exact — measured over nine configurations against the array path's own
    three passes.

    NO WALL CLEAR IS EMITTED AT A GHOST, and the two fills reach that conclusion by
    DIFFERENT routes, both checked below rather than assumed:

    * the NEAR ghost moves the component to stored 0 of its OWN axis — the exact
      cell ``_zero_metal`` clears — but that axis is FOLDED here, and
      ``_zero_metal`` skips a folded axis (stepping.py:2284-2286), so no line can
      apply. A drift in either table would put one there, so it is asserted;
    * the FAR ghost moves the component along an axis that is NOT its own, and
      ``_ZERO_METAL_ROWS`` is the B DIAGONAL, so the ghost sits at the SAME
      coordinate on every axis that could clear it as the thread that owns it. Its
      clear status is therefore its source's, and it is INHERITED rather than
      re-emitted: this block reads ``v{target}`` AFTER the owned cell's clear lines
      have run on that register, which is the driver's own order
      (``zero_metal_B`` at :3286 before ``fill_folded_far_ghosts_B`` at :3287).
      What is asserted is the disjointness that makes the inheritance exact.
    """
    walls = tuple(axis for _, axis, _ in _ZERO_METAL_ROWS
                  if _ == target and bool(zero_metal[axis]))
    if near and zero_metal_lines(target, zero_metal, "unused", indent):
        raise AssertionError(
            f"component {target} images a NEAR ghost on folded axis {near[0]} AND "
            f"zero_metal_B clears it on axes {walls}. stepping._zero_metal skips a "
            f"folded axis, so the two tables have drifted")
    overlap = sorted(set(walls) & set(int(axis) for axis in far))
    if overlap:
        raise AssertionError(
            f"component {target} images a FAR ghost along axes {tuple(far)} that "
            f"zero_metal_B also clears on {overlap}; _ZERO_METAL_ROWS is the B "
            f"diagonal and far_fill_axes excludes the component's own axis, so the "
            f"ghost would no longer inherit its source thread's clear")
    inner = indent + "    "
    lines: List[str] = []
    for subset, carries_near in carried_destinations(near, far):
        tag = (f"g{target}_"
               + "".join(_COORDINATE[axis] for axis in subset)
               + ("n" if carries_near else ""))
        flags = [f"{_COORDINATE[axis]} == {_REFLECT[axis]}" for axis in subset]
        terms = [f"+ ({_LAST[axis]} - {_REFLECT[axis]}) * {_STRIDE[axis]}"
                 for axis in subset]
        weight = 1
        for axis in subset:
            weight *= _component_parity(target, axis, phases[axis])
        described = [f"the far fill on {_COORDINATE[axis]}: "
                     f"{_COORDINATE[axis]} = {_LAST[axis]} imaged from "
                     f"{_REFLECT[axis]} (stepping._fill_folded_far_ghosts:1516)"
                     for axis in subset]
        if carries_near:
            axis = int(near[0])
            flags.append(f"{_COORDINATE[axis]} == {NEAR_SOURCE_INDEX}")
            terms.append(f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}")
            weight *= _component_parity(target, axis, phases[axis])
            described.append(
                f"the near fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = 0 "
                f"imaged from {NEAR_SOURCE_INDEX} "
                f"(stepping._fill_symmetry_ghost_cells:1441, "
                f"_write_mirror_ghost:1451)")
        lines.append(f"{indent}if ({' && '.join(flags)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(f"{inner}// composed parity {weight:+d}.")
        lines.append(f"{inner}int {tag}_i = ii {' '.join(terms)};")
        lines.append(f"{inner}float {tag}_v = "
                     f"{_parity_spelling(weight, f'v{target}')};")
        lines.append(f"{inner}f{target}[{tag}_i] = {tag}_v;")
        if carries_near:
            lines.extend(_ghost_coefficient_lines(target, int(near[0]), tag, inner))
            kp, km = f"{tag}_kp", f"{tag}_km"
        else:
            # The far destination shares the indexed axis with its source thread,
            # so the certified pair this thread already loaded IS the destination's.
            kp, km = f"kp_{target}", f"km_{target}"
        lines.extend(_constitutive_block(
            target, f"{tag}_i", f"{tag}_v", kp, km, f"{tag}_", inner))
        lines.append(f"{indent}}}")
    return lines


def folded_fused_magnetic_pair_source(codes: Sequence[int],
                                      phases: Sequence[int],
                                      zero_metal: Sequence[bool],
                                      contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded boundary quadruple, parities, walls, mode).

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC /
    MIRROR_PERIODIC quadruple :func:`.symmetry.folded_axis_kinds` resolves — never a
    hand-built triple, because the MIRROR_METALLIC / MIRROR_PERIODIC split is this
    family's single point of failure and getting it backwards on one axis is a plane
    of wrong values rather than a crash.

    ``phases`` is ``grid.mirror_phase(axis)`` per axis, +1 or -1 on a folded axis and
    ignored elsewhere. It is a SOURCE specialisation and not a runtime scalar, which
    is the opposite of the Triton twin's rule and is MEASURED: a runtime weight
    flushes every subnormal on this backend at BOTH signs
    (:func:`.symmetry._parity_spelling`).

    Every constant Triton would bake into a ``tl.constexpr`` is baked into the
    string, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes
    a source and nothing else.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if not any(code in MIRROR_CODES for code in codes):
        raise ValueError(
            "no axis is folded: this product exists to carry the mirror fill "
            "inside the B seam, and an unfolded grid belongs to fused_magnetic_pair")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    phases = tuple(int(value) if value is not None else 0 for value in phases)
    if len(phases) != 3:
        raise ValueError(f"phases must be a per-axis triple, got {phases!r}")
    for axis, code in enumerate(codes):
        if code in MIRROR_CODES and phases[axis] not in (1, -1):
            raise ValueError(
                f"axis {axis} is folded but its mirror phase is {phases[axis]!r}; "
                f"a plane's parity is +1 or -1 and the even-mirror default standing "
                f"in for a plane that declared otherwise is a run wrong by twice "
                f"the field wherever the parity mattered")
    # THE DISJOINTNESS THE CARRY RESTS ON, asserted rather than trusted:
    # `zero_metal_axes` is `is_metallic and not is_mirrored` (stepping.py:2284-2286),
    # so a folded axis can never be a walled one. If that changed, the imaged ghost
    # would need a wall line this kernel does not emit.
    for axis, code in enumerate(codes):
        if code in MIRROR_CODES and zero_metal[axis]:
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction "
                f"(stepping.py:2284-2286) and this kernel's ghost carry relies on "
                f"the two sets being disjoint")

    transcription = constitutive_transcription(contract)
    if not transcription["identical"]:
        raise AssertionError(
            "this family's parameterised constitutive statements no longer "
            "reproduce shaders.constitutive_source('H'); the fused arithmetic "
            "would silently stop being the certified arithmetic. emitted="
            f"{transcription['emitted']!r} certified={transcription['certified']!r}")

    statements = certified_curl_statements(codes, contract)
    body: List[str] = [certified_curl_head(codes, contract).rstrip("\n")]
    periodic = tuple(axis for axis, code in enumerate(codes)
                     if code == CODE_MIRROR_PERIODIC)
    body.extend([
        "",
        ("    // The three top-plane flags the lifted head declares are read below "
         "by the far"
         if periodic else
         "    // The three top-plane flags the lifted head declares are unread here:"),
        ("    // carry's ownership guard, and by folded_top_plane_mask's own lines "
         "above."
         if periodic else
         "    // no folded PERIODIC axis, so folded_top_plane_mask emits no line "
         "and the"),
        ("    // A (void) keeps the unfolded axes' flags from warning."
         if periodic else "    // block above is a comment."),
        "    (void)last_x; (void)last_y; (void)last_z;",
        "",
        "    // --- split-field recurrence (stepping._apply_pml_update:1905) ------",
    ])
    body.extend(statements["coefficients"])
    body.extend([
        "",
        "    // --- the constitutive coefficient, on the component's OWN axis -----",
        "    // Read HERE for the owned cell only. On this family the imaged ghost",
        "    // cell does NOT share it: the fill images along the component's own",
        "    // axis, which is the axis this is indexed on.",
    ])
    body.extend(_constitutive_coefficient_lines(contract))
    body.append("")

    # The three recurrences and the three `fu` stores, in the certified body's own
    # order. `u` IS WRITTEN AT EVERY CELL, ghost cells included: `step_B` updates
    # `fu` everywhere and the fill touches B only (stepping.py:1451 writes `field`,
    # never `fu_field`).
    for target in range(3):
        previous, recurrence, _ = statements["recurrence"][target]
        body.extend([previous, recurrence])
    body.append("")
    body.append(statements["aux_store"])

    for target in range(3):
        near = near_fill_axes(codes, target)
        if len(near) > 1:
            raise AssertionError(
                f"component {target} is a near-fill destination on {near}; for the "
                f"B family that set is at most one axis (the component's own) and "
                f"this carry images one near ghost cell per component")
        far = far_fill_axes(codes, target)
        if len(far) > 2 or (near and near[0] in far):
            raise AssertionError(
                f"component {target} is a far-fill destination on {far} against a "
                f"near axis {near}; for the B family the two sets are COMPLEMENTARY "
                f"(iyee is 0 on the component's own axis and 1 on the other two, "
                f"fields.IYEE_SHIFTS), so far holds at most two axes and never the "
                f"near one")
        _, _, displacement = statements["recurrence"][target]
        body.append("")
        # A cell any fill writes is OWNED BY ITS SOURCE THREAD. This thread stops
        # after fu there: the array path's fill overwrites the displacement it
        # would store, and forming v would read f at a word another thread writes.
        owned: List[str] = []
        if near:
            owned.append(f"{_COORDINATE[near[0]]} == 0")
        owned.extend(_LAST_FLAG[axis] for axis in far)
        if owned:
            body.extend([
                f"    // component {target}: the fills image "
                + ", ".join(
                    [f"stored cell 0 on {_COORDINATE[axis]} (near)"
                     for axis in near]
                    + [f"the top plane on {_COORDINATE[axis]} (far)"
                       for axis in far]) + ",",
                "    // so those cells are OWNED BY THEIR SOURCE THREADS. This one "
                "stops after fu:",
                "    // the fill overwrites the displacement it would store, and "
                "forming v here",
                "    // would read a word another thread writes.",
                f"    if (!({' || '.join(owned)})) {{",
            ])
            indent = "        "
        else:
            body.append(f"    // component {target}: no folded axis images a cell "
                        f"of this component")
            indent = "    "
        body.append(indent + displacement.strip())
        # `zero_metal_B` on the cell this thread owns, then the store, then the
        # constitutive read of the SAME register — the seam this family closes.
        cleared = zero_metal_lines(target, zero_metal, f"v{target}", indent)
        body.extend(cleared or [f"{indent}// no walled axis clears this target"])
        body.append(indent + statements["stores"][target])
        body.extend(_constitutive_block(
            target, "ii", f"v{target}", f"kp_{target}", f"km_{target}",
            f"o{target}_", indent))
        if owned:
            body.append("")
            body.extend(_carry_blocks(target, near, far, phases, zero_metal,
                                      indent))
            body.append("    }")

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__BODY__": "\n".join(body),
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   reflect: Sequence[Optional[int]], device: str) -> Any:
    """The eight scalars as one 32-byte device record.

    :func:`.fused_magnetic_pair._params_tensor` WITH THE THREE REFLECT ROWS, and
    that is the only reason this family does not import it: the far carry needs
    ``stepping._far_reflect_rows``' answer per axis, and putting them here rather
    than in three new buffers is what keeps the signature at
    :data:`.fused_magnetic_pair.PACKED_BINDINGS` — 28 of the platform's 31
    (device.py:72) — instead of 31.

    ``-1`` stands for an axis with no reflect row (unfolded, or folded METALLIC,
    where the stored array stops at ``big_corner`` and no far ghost exists). It is
    a value the kernel never reads: the guard that would read it is emitted only
    for an axis :func:`far_fill_axes` returned, and that requires
    ``CODE_MIRROR_PERIODIC``. A sentinel rather than 0, so a source that read it
    anyway would index out of the volume and be caught, not silently image row 0.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for
    :meth:`.Residency.mirror`'s reason: it binds float32 and complex64 volumes and
    refuses anything else BY NAME, and this record is a packed struct of four
    uints, a float and three ints. Built once, at plan time; nothing on the launch
    path allocates.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype(
        [("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
         ("dtdx", "<f4"), ("rx", "<i4"), ("ry", "<i4"), ("rz", "<i4")]))
    nx, ny, nz = (int(n) for n in shape)
    rows = tuple(-1 if value is None else int(value) for value in reflect)
    if len(rows) != 3:
        raise ValueError(f"reflect must be a per-axis triple, got {reflect!r}")
    record[0] = (nx, ny, nz, nx * ny * nz, np.float32(dtdx)) + rows
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def compile_folded_fused_magnetic_pair(codes: Sequence[int],
                                       phases: Sequence[int],
                                       zero_metal: Sequence[bool],
                                       contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, parities, walls, mode)."""
    return compile_source(folded_fused_magnetic_pair_source(
        codes, phases, zero_metal, contract)).folded_fused_magnetic_pair_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _mirror_phase_reasons(grid: Any, codes: Optional[Sequence[int]]) -> List[str]:
    """Every folded axis must declare a readable +1/-1 parity.

    ``stepping._symmetry_phase`` RAISES on a ``None`` phase (stepping.py:2463-2469)
    rather than folding with the even-mirror default, and this kernel bakes the
    parity into the SOURCE. A grid that cannot answer is refused here, before a
    compile, for the same reason.
    """
    reasons: List[str] = []
    if codes is None:
        return reasons
    for axis, code in enumerate(codes):
        if int(code) not in MIRROR_CODES:
            continue
        phase = _call(grid, "mirror_phase", axis, default=None)
        if phase is None or int(phase) not in (1, -1):
            reasons.append(
                f"axis {axis} is folded but grid.mirror_phase({axis}) is {phase!r}; "
                f"the parity is a compile-time specialisation here and cannot be "
                f"defaulted")
    return reasons


def _far_carry_reasons(grid: Any, codes: Optional[Sequence[int]]) -> List[str]:
    """What a folded PERIODIC axis must satisfy for the far carry to be legal.

    THE CLAUSE THIS FAMILY USED TO REFUSE BY NAME. ``fill_folded_far_ghosts_B``
    runs inside this seam (driver.py:3287) on a folded PERIODIC axis and images the
    top stored slot from ``stepping._far_reflect_rows``' row; the kernel now
    carries it, and these are the conditions the carry's OWNERSHIP MOVE rests on.

    The three row checks are :func:`.symmetry.mirror_ghost_fill_coverage`'s own
    (symmetry.py:1272-1307), reused rather than re-derived, and the reasons are
    restated in this product's terms because here a violated row is an
    OUT-OF-RANGE WRITE from a thread that is not the destination's, not a soft
    error on a whole-plane assignment.

    TWO FURTHER CLAUSES ARE THIS PRODUCT'S:

    * ``_stored_past_owned`` must AGREE with the code, both ways. The code is what
      decides whether the kernel emits a far carry at all AND whether the lifted
      curl head masks the top plane, so a disagreement is a plane the kernel steps
      and images inconsistently with the array path. ``folded_axis_kinds`` already
      derives the code from that predicate; asked again here because reading
      coverage off another module's derivation is what this package's rule forbids.
    * the reflect row must not be the near fill's SOURCE plane (stored
      ``NEAR_SOURCE_INDEX``) on an axis that also carries a near fill. It cannot
      be — the near axis of a B component is its own and the far axes are the other
      two — but that is a property of ``fields.IYEE_SHIFTS``, and the source
      builder asserts the complementarity rather than assuming it, so the predicate
      names the same fact before a compile.
    """
    reasons: List[str] = []
    if codes is None:
        return reasons
    rows = _far_reflect_rows(grid)
    shape = tuple(getattr(grid, "shape", ()))
    for axis, code in enumerate(codes):
        periodic = int(code) == CODE_MIRROR_PERIODIC
        try:
            past_owned = bool(_stored_past_owned(grid, axis))
        except Exception as error:  # pragma: no cover - a grid that cannot answer
            reasons.append(
                f"axis {axis}: stepping._stored_past_owned raised {error!r}; the "
                f"far carry and the curl's top-plane mask are both keyed on it")
            continue
        if periodic != past_owned:
            reasons.append(
                f"axis {axis} code {int(code)} and "
                f"stepping._stored_past_owned {past_owned} disagree; the kernel "
                f"would mask the top plane and image the far ghost on different "
                f"axes than the array path")
        if not periodic:
            continue
        row = rows[axis] if rows is not None else None
        if row is None:
            reasons.append(
                f"axis {axis} is a folded PERIODIC axis with no reflect row from "
                f"stepping._far_reflect_rows; the far carry has nothing to image")
            continue
        row = int(row)
        if len(shape) != 3:
            reasons.append("grid.shape is not a per-axis triple; the far carry's "
                           "destination index cannot be built")
            continue
        extent = int(shape[axis])
        if not (0 <= row < extent - 1):
            reasons.append(
                f"axis {axis} reflect row {row} is outside [0, {extent - 1}); the "
                f"far carry would image the plane it writes, or read outside the "
                f"allocation from a thread that owns neither cell")
        if row == 0:
            reasons.append(
                f"axis {axis} reflect row is 0, the plane the NEAR fill writes: "
                f"the far carry would image a ghost rather than an owned cell")
        if extent - 1 == NEAR_SOURCE_INDEX:
            reasons.append(
                f"axis {axis} stores {extent} cells, so the far carry's write "
                f"plane IS the near fill's read plane (cell {NEAR_SOURCE_INDEX}); "
                f"refused rather than made order-dependent, exactly as "
                f"symmetry.mirror_ghost_fill_coverage refuses it")
        if row == NEAR_SOURCE_INDEX and any(
                near_fill_axes(codes, target) == (axis,) for target in range(3)):
            reasons.append(  # pragma: no cover - the B geometry excludes it
                f"axis {axis} carries BOTH a near fill and a far carry whose "
                f"reflect row is the near source plane {NEAR_SOURCE_INDEX}; the B "
                f"family's Yee shifts make those sets complementary, so this means "
                f"fields.IYEE_SHIFTS has drifted")
    return reasons


def metal_folded_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                              sources: Any = None,
                                              residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_B`` -> near fill -> wall -> ``update_H``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it. That is
    the construction :func:`.folded_fused_pair.metal_folded_fused_pair_coverage`
    uses on the electric half.
    """
    reasons: List[str] = []

    curl = folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP, residency)
    if not curl.covered:
        reasons.extend(f"folded curl half: {reason}" for reason in curl.reasons)
    magnetic = folded_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE,
                                            residency)
    if not magnetic.covered:
        reasons.extend(f"folded constitutive half: {reason}"
                       for reason in magnetic.reasons)

    # THE SOURCE SEAM, and on the measured corpus it is the binding clause of this
    # family — 54 of the 77 rows that clear both halves survive it. A MAGNETIC
    # source is injected BETWEEN the two halves (driver.py:3283-3284), so a fused
    # pair would consume a pre-injection B. IGNORANCE IS NOT AN EMPTY SET: `Fields`
    # does not hold the source list, so a predicate that inferred "no sources" from
    # not being told would be the over-covering refusal this clause exists to
    # prevent. An ELECTRIC source is injected in the D/E half and does NOT
    # disqualify this pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283-3284), which is work inside the seam this "
            f"kernel closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # ASKED AFTER THE SOURCE CLAUSE, deliberately: that clause reads no grid, and a
    # degenerate object should still get its polarity named rather than only "no
    # grid". Everything below this line does need one.
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    reasons.extend(_far_carry_reasons(grid, codes))
    reasons.extend(_mirror_phase_reasons(grid, codes))

    # THE WALL CLEAR IS CARRIED INLINE, so the grid must be able to answer which
    # axes are walled. A grid that cannot would silently be treated as unwalled,
    # which is a plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        # THE NEAR FILL IMAGES STORED CELL 2, and the source thread must exist. The
        # folded curl predicate already refuses a folded axis storing too few cells;
        # restated because THIS family writes the destination from a DIFFERENT
        # thread and a missing source plane is an out-of-range write, not a soft
        # error.
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; "
                    f"the near fill images stored cell {NEAR_SOURCE_INDEX} and this "
                    f"kernel images it from that cell's own thread")

        # THE WALL CLEAR AND THE FILL MUST NOT MEET. On this family both act on
        # component m at stored cell 0 of axis m, and `_zero_metal` skips a folded
        # axis so they are disjoint by construction. CHECKED rather than inferred:
        # reading coverage off another module's guard is what this package's rule
        # forbids, and an overlap would be a plane of wrong values.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall "
                    f"there; stepping._zero_metal skips a folded axis, so the two "
                    f"passes have drifted and the carry's disjointness no longer "
                    f"holds")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs four driver passes, B never leaving a register.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no
    special case for it.

    ``runs`` is an alias of ``launches`` rather than a second counter, because with
    one dispatch per run the two ARE the same number and a second counter would be a
    second thing to get out of step.
    """

    __slots__ = ("residency", "volumes", "codes", "phases", "zero_metal", "shape",
                 "dtdx", "params", "carried_axes", "carried_far_axes", "reflect")

    family = "folded fused B-curl/mirror-fill/stored-H pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phases", "zero_metal", "carried_axes",
                   "carried_far_axes", "reflect")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phases: Sequence[int],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 reflect: Sequence[Optional[int]],
                 params: Any, pointers: Sequence[Any],
                 functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phases = tuple(int(value) for value in phases)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # float(): the array path multiplies a float32 volume by a Python float and
        # the packed struct holds numpy.float32 of the same value, so the two
        # scalars are the same bits.
        self.dtdx = float(dtdx)
        self.params = params
        self.reflect = tuple(None if value is None else int(value)
                             for value in reflect)
        self.carried_axes = tuple(
            near_fill_axes(self.codes, target) for target in range(3))
        # The far-fill destinations this launch carries, per component. SEPARATE
        # from `carried_axes` rather than merged with it: they are the two fills'
        # COMPLEMENTARY axis sets (near on the component's own axis, far on the
        # other two), and a gate row that could not tell them apart could not say
        # which pass a byte came from.
        self.carried_far_axes = tuple(
            far_fill_axes(self.codes, target) for target in range(3))
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def plan_metal_folded_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalFoldedFusedMagneticPairPlan]:
    """Build the fused folded B/H plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.symmetry.plan_folded_pml_curl`
    gives: a configuration this kernel does not carry must fall back to the array
    path, never raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy
    of the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING — every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not metal_folded_fused_magnetic_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    walls = zero_metal_axes(grid)
    # The far carry's image rows, from the engine's own function. Read here rather
    # than recomputed: `n_full - stored + 2` is `stored - 2` at an even full count
    # and `stored - 3` at an odd one, and a fixed `n - 2` is a whole cell wrong on
    # every odd-count run (stepping._far_reflect_rows:1661).
    reflect = _far_reflect_rows(grid)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step,
    # so the groups are built in the order the kernel declares them. There is NO
    # inverse-mu volume: the H product is `source = B` with mu = 1 baked into the
    # array path too (stepping.update_H passes fields.Bx), which is what makes this
    # signature 27 pointers rather than 30.
    pointers = (
        [bind(name, getattr(fields, name)) for name in curl_spec["targets"]]
        + [bind("fu_" + name, getattr(fields, "fu_" + name))
           for name in curl_spec["targets"]]
        + [bind(name, getattr(fields, name)) for name in curl_spec["sources"]]
        + [bind(name, getattr(fields, name)) for name in side_spec["targets"]]
        + [bind(name, getattr(fields, name)) for name in side_spec["aux"]]
        # The B curl takes the HALF-INTEGER lattice and the H constitutive the
        # INTEGER one (SUB_STEPS['step_B']['suffix'] == '_h';
        # CONSTITUTIVE_SIDES['H']['half_integer'] is False). That is the OPPOSITE
        # pairing to the folded D/E pair, and the kernel cannot tell — the gate
        # carries a host mutation for each group.
        + [bind(f"pml:{stem}_{axis}{curl_spec['suffix']}",
                getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}"), constant=True)
           for axis in "xyz" for stem in ("kms", "sinv")]
        + [bind(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"), constant=True)
           for axis in "xyz" for stem in ("kps", "kms")])
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_fused_magnetic_pair(
                codes, phases, walls, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor would
    # read "mps:0" where the mirrors were built with "mps".
    dtdx = grid.dt / grid.dx
    return MetalFoldedFusedMagneticPairPlan(
        residency, volumes, codes, phases, walls, grid.shape, dtdx, reflect,
        _params_tensor(grid.shape, dtdx, reflect, residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — wired=False (no arm selection), routed by the absorb table
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False,
                        (f"folded fused magnetic pair cannot fill {slot}",))
    return metal_folded_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    THE FLAG IS ONE HALF OF THE COMPOSITION STORY, and until 2026-08-27 it was the
    whole of it. ``plan_step`` assigns at most one ARM per slot and this product
    spans four, so there is no slot it could claim through the arm table.
    Registering unwired keeps it ENUMERABLE for the disjointness sweep
    (``arms.registered``) while ``arms.arms_for`` skips it, so ``_select_slot``
    cannot select it and no existing arm's selection changes.

    THE OTHER HALF IS THE ABSORB TABLE, which is what routing this family means.
    ``launch._install_fused_pairs`` reaches this row through ``arms.registered``
    precisely BECAUSE it is registered — ``is_weld`` with ``replaces`` covering
    ``update_H`` — and ``launch.FUSED_PAIR_ARMS["folded_fused_magnetic_pair"]``
    declares which arm each of the two slots implements. So ``wired=False`` still
    says what it always said (no arm selection), and it no longer implies that
    nothing composes this product.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "folded fused B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded fused B/H pair: ",
                          noun="folded PML B-curl/mirror-fill/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
