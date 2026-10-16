"""The SECOND fused Metal kernel: the folded ``step_D`` welded into ``update_E``.

ONE DISPATCH FOR ALL THREE COMPONENTS, which is what separates this product from
:mod:`.fused_dispersive_pair`. That family dispatches per component because each
component carries up to eight pole buffers; the folded PLAIN constitutive carries
none, so the whole D half fits in a single launch — 30 pointers plus one packed
``constant Params&``, the 31-binding shape the platform measurement licenses (see
"THE SIGNATURE" below). The separate composition for this configuration is TWO
device dispatches (the folded curl, then the certified constitutive) plus one to
three host fill passes; this is ONE, and the displacement never leaves a register.

EVERYTHING HERE IS LIFTED OR TRANSCRIBED, and every arithmetic line is either a
certified emitter's own text or carries the file:line it came from:

* the folded D curl — LIFTED from :func:`.symmetry.folded_curl_source` at
  ``backward=True``: its head (guard, decode, the chunk record's bounds block,
  ghost gather, the curl and its absent-partner forms, the cell-0 mask and the
  top-plane mask of :func:`.symmetry.folded_top_plane_mask`) verbatim by
  :func:`certified_curl_head`, and its per-chunk form statements line by line by
  :func:`certified_curl_statements`, two of them through the declared edits of
  :data:`LIFT_EDITS`;
* the wall clear — ``stepping._zero_metal`` (stepping.py:2206-2247) restricted to
  ``D_COMPONENTS``, carried inline through :data:`.fused_dispersive_pair._ZERO_METAL_ROWS`,
  which is IMPORTED rather than re-spelled;
* the near mirror fill — ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1426-1452)
  and ``stepping._write_mirror_ghost`` (stepping.py:1497-1498), with the parity
  spelled by :func:`.symmetry._parity_spelling` (symmetry.py:657-686), the ONE
  measured place a mirror parity becomes Metal text;
* the constitutive half — ``shaders.constitutive_source("E")``'s own body
  (:func:`certified_constitutive_body`, renamed for this kernel's pointers), whose
  six statements per component are re-emitted at the owned cell and at every imaged
  ghost and checked statement for statement by :func:`constitutive_transcription`:
  the E product ``D * inv_eps`` with D LEFT, the accumulation on a cell whose chunk
  carries the component's own layer and the output select ``E = D * inv_eps``
  elsewhere (``stepping.update_E`` through ``stepping._apply_constitutive_regions``).

===========================================================================
THE PML FORMS ARE CHOSEN PER STRUCTURE CHUNK — the chunk record, in Params
===========================================================================

Both halves take the array path's per-chunk forms: the curl steps the split-field
recurrence on a cell whose chunk carries the target's dsigu layer and the dsig-only
form elsewhere, and ``update_E`` accumulates on a cell whose chunk carries the
component's own layer and is the output select elsewhere. ``fu_D`` is read and
written only on the recurrence's cells and ``f_w_E`` only on the accumulation's.
Membership is the chunk record (:mod:`.chunk_bounds`), which rides as one member of
the packed ``Params`` and costs no binding — the family stays at 31, the ceiling.

Each select is evaluated WHERE THE CELL IS. A destination thread steps its own curl
form and ``fu`` at its own coordinate. The E half's membership is on the
component's OWN axis, which only the FAR fill moves: a far ghost's flag is evaluated
at its destination (stored ``last``), beside the coefficient pair it reloads there,
and a near ghost shares its owner's flag as it shares its owner's pair. The
record's lines are ghost-adjusted (``chunk_map.ChunkLayout.membership``), so a ghost
slot reads its image row's membership, as the array path's line does.

ONE LIFTED LINE PAIR IS EDITED, BY DECLARATION. The certified curl reads
``p = du ? fu : f`` at every cell, which on a destination thread is a load of a word
the source thread writes. :data:`LIFT_EDITS` moves the dsig-only form's read of
``f`` into the owned block — the same operations in the same order, so the same
word.

===========================================================================
THE SEAM, AND THE FILL THAT SITS INSIDE IT — the whole design problem
===========================================================================

The driver runs FOUR passes between ``step_D`` and ``update_E``
(driver.py:3292-3304, mirrored in :data:`.coverage.RESIDENCY_ORDER`):
``step_D`` -> electric sources -> ``fill_symmetry_bc_D`` -> ``zero_metal_D`` ->
``fill_folded_far_ghosts_D`` -> ``update_E``. On an UNFOLDED grid the two fills
return immediately (stepping.py:1482-1483) and the first fused pair refuses the
fold outright. HERE THE FOLD IS MANDATORY, so both fills are live questions and
each is answered by name:

* **the electric sources — REFUSED BY NAME**, as in the first fused pair, and this
  is what caps the family on the measured corpus rather than anything about the
  fold. The injection lands between the two halves, so a fused pair would consume
  a pre-injection D. Ignorance is never an empty set: an undeclared source list is
  a refusal.
* **``zero_metal_D`` — CARRIED INLINE.** ``_zero_metal`` writes zero into stored
  cell 0 of every D component whose Yee shift on a walled axis is 0, and it
  DELIBERATELY SKIPS A FOLDED AXIS (stepping.py:2282-2284: ``is_metallic(axis) and
  not is_mirrored(axis)``) because a folded axis's stored cell 0 holds the parity
  ghost rather than a wall — measured on the array path at 1.28e+00 complex
  relative L2 against CPU MEEP with that ghost zeroed, against 2.0e-07 with the
  skip. THE WALLED AXES AND THE FOLDED AXES ARE THEREFORE DISJOINT BY
  CONSTRUCTION, which is the fact that makes the fill carry below exact.
* **``fill_symmetry_bc_D`` — CARRIED INLINE, and it is NOT dead.** This was the
  first hypothesis this module tested and it is FALSE: the near fill writes
  ``cell 0 = parity * cell 2`` on every folded axis for every component whose Yee
  shift there is 0, ``update_E`` then reads exactly those cells, and the wall clear
  that runs after it does not touch them (previous bullet). A fused pair that
  skipped the fill would consume a pre-fill D on two of the three components of
  every folded grid. See "THE OWNERSHIP RESTRUCTURE" below for how one dispatch
  carries a pass whose source cell another thread computes.
* **``fill_folded_far_ghosts_D`` — CARRIED INLINE as of 2026-08-21.** It runs only
  on a folded PERIODIC axis (``_stored_past_owned``, stepping.py:1503-1518) and it
  images the last stored slot from a RUNTIME reflect row on the component's OWN
  axis. It was REFUSED BY NAME until this round, on the ground that source and
  destination then take different constitutive coefficient indices; they do, and
  the carry pays for it with two extra scalar loads
  (:func:`_far_coefficient_lines`) exactly as the magnetic twin pays for its NEAR
  ghost. Carrying it brings the folded curl's TOP-PLANE MASK with it
  (``symmetry.folded_top_plane_mask`` now emits a line), which the body already
  calls and which the refusal used to make unreachable — the second obligation the
  clause discharged, and a carry that took the fill and left the mask would have
  left a plane of unmasked cells behind. The reflect rows ride in the packed
  ``Params``, so the binding count is UNCHANGED at 31.

===========================================================================
THE D GEOMETRY IS NOT THE B GEOMETRY — and every fact INVERTS
===========================================================================

``fields.IYEE_SHIFTS`` (fields.py:214-219): ``Dx (1,0,0)``, ``Dy (0,1,0)``,
``Dz (0,0,1)`` against ``Bx (0,1,1)``, ``By (1,0,1)``, ``Bz (1,1,0)``. A fill
touches component ``m`` on axis ``a`` when ``iyee[m][a] == 0`` (near) or ``== 1``
(far), so

    D family:  near = the two axes that are NOT the component's own; far = its OWN
    B family:  near = the component's OWN axis alone;   far = the other two

which is the exact swap, and it is not a fact about this module — it is the same
complementarity ``symmetry._TOP_PLANE_FLAGS`` is derived from and
``test_metal_symmetry`` already pins. THREE CONSEQUENCES, each the mirror image of
:mod:`.folded_fused_magnetic_pair`'s:

1. **THE COEFFICIENT INDEX MOVES ON THE FAR GHOST AND NOT ON THE NEAR ONE**, which
   is the OPPOSITE of the magnetic twin. ``update_E`` indexes ``kps``/``kms`` on
   the component's own axis (shaders.py:310-312), the near fill images along an
   axis that is never the component's own, and the far fill images along exactly
   that axis. So a near destination takes its source thread's pair unchanged and a
   FAR destination loads its own at stored index ``last``
   (:func:`_far_coefficient_lines`). **AND HERE IT IS OBSERVABLE RATHER THAN
   USUALLY BLIND**, which is the second half of the inversion: a folded axis carries
   an absorber on its HIGH face only (``stepping._require_consistent_pml``), the
   near ghost sits at the mirror plane where there is none, and the FAR ghost sits
   at the top plane, which is inside the layer. The gate censuses the coefficient
   vectors and predicts observability from them rather than asserting it either way.
2. **THE NEAR FILL MAKES THE PARITY PRODUCTS AND THE FAR ONE NEVER DOES.**
   :func:`near_fill_axes` returns up to TWO axes for a D component, so the X/Y/Z
   application order (stepping.py:1441-1447) is live and a corner unowned on both
   planes carries the PRODUCT of the two parities; :func:`far_fill_axes` returns AT
   MOST ONE — the component's own — so no far corner exists. One source thread
   therefore owns up to ``2^2 * 2 - 1 = 7`` ghost cells for one component, the same
   count the magnetic twin reaches from the other direction.
   :func:`carried_destinations` is the closed form, and it is MEASURED against the
   array path's own three passes over thirteen configurations — zero differing
   words, and five planted variants of the same form each shown to produce them
   (``results/folded_far_carry_d_2026-08-21/d_ownership.json``,
   ``d_ownership_flip.json``).
3. **THE WALL CLEAR MEETS THE NEAR GHOST AND IS INHERITED BY THE FAR ONE.**
   ``_zero_metal`` clears component ``m`` at stored 0 of a walled axis ``b`` with
   ``iyee[m][b] == 0`` — ``b != m``, the same shift condition the NEAR fill uses —
   and a walled axis is never a folded one, so ``b`` is never a near axis but CAN
   be a second non-own axis. The near ghost therefore takes the clear
   (``mask(parity * v)``, the driver's fill-then-clear order at :3300 before :3301),
   and on the magnetic twin that question is vacuous. The FAR ghost moves along the
   component's own axis, which ``_ZERO_METAL_ROWS`` never clears, so it sits at the
   same coordinate on every axis that could clear it as the thread that owns it: its
   clear status is INHERITED by reading ``v`` after the clear lines, which is the
   driver's order (:3301 before :3302). The two orders are therefore OPPOSITE within
   one kernel — ``mask(parity * v)`` at a near ghost, ``parity * mask(v)`` at a far
   one — and on a cleared plane at an odd parity they differ in the sign bit of a
   zero. Measured: 12 and 27 differing words respectively when each is written the
   other way round (``d_ownership_flip.json``).

TWO OF THE MAGNETIC TWIN'S REFUSAL CLAUSES DO NOT TRANSFER, and they are named here
rather than copied. ``_far_carry_reasons`` there refuses a reflect row of 0 ("the
far carry would image a ghost rather than an owned cell") and an extent whose top
plane IS the near fill's read plane. Both are properties of the B geometry, where
the near and far axes are the same axis or share one: on the D side the near
destination is stored 0 of an axis that is NEVER the far axis, so stored 0 of the
component's own axis is an OWNED, stepped cell and the two planes are always
orthogonal. Copying either clause would have been a refusal with no hazard behind
it.

===========================================================================
THE OWNERSHIP RESTRUCTURE — one dispatch, no barrier, no race
===========================================================================

Both fills read a cell THIS DISPATCH WRITES. Metal has no device-wide barrier,
so the naive carry — the thread at stored cell 0 reading stored cell 2, or the
thread at the top plane reading the reflect row — is a race on both ``f`` and
``u``. THE FIX IS TO MOVE THE OWNERSHIP: for a component the fills write on folded
axis ``a``,

* the thread whose coordinate on a NEAR axis ``a`` is 0, or whose coordinate on the
  FAR axis is ``last``, is a DESTINATION. It computes the curl and, where its own
  chunk carries the target's dsigu layer, the split-field recurrence's ``n`` and
  stores ``u`` (the array path's ``step_D`` writes ``fu`` on the recurrence's cells
  and neither fill touches it), and then STOPS: it never forms ``v``,
  never reads ``f``, never writes ``f``, and never touches ``E`` or ``f_w_E``. Its
  stepped displacement is discarded by the array path too — the fill overwrites it,
  and at the top plane ``_mask_non_owned_cells`` has already zeroed its curl;
* the thread at stored 2 on every near axis it images and at the runtime reflect row
  on the far axis is the SOURCE. It does its own cell in full and then writes each
  destination as well: the composed parity, the wall clear where the near fill's
  order calls for it, the displacement stored, and the constitutive update performed
  there at the DESTINATION's coefficient index.

Every destination cell is written by exactly one thread, no thread reads a word
another thread writes, and no barrier is needed. The wall clear's flags do not move
at all: a walled axis is never a folded axis (``stepping._zero_metal``:2235-2237),
and the clear reads only axes that are neither a near axis (folded) nor the far one
(the component's own, which ``_ZERO_METAL_ROWS`` never clears), so the destination's
``at_x``/``at_y``/``at_z`` ARE the source's.

MULTIPLE FOLDED AXES COMPOSE, and the composition is the array path's: the near
fills run in X, Y, Z order (stepping.py:1441-1447), so a cell unowned on two planes
carries the PRODUCT of both parities imaged from the doubly-shifted source, and the
far fill then images the whole top plane of the own axis from the reflect row —
which is why the closed form is independent of the order the array path applies its
axes in and nothing here has to model that loop. A thread at coordinate 2 on both
near axes AND at the reflect row on its own axis writes SEVEN destinations, one per
nonempty (near subset, far?) pair; a thread that is itself a destination propagates
nothing, because the cell it would image from is imaged by the source that owns all
of them.

===========================================================================
THE SIGNATURE — 30 pointers plus one packed struct, MEASURED
===========================================================================

Counting an all-three-component fused folded D/E kernel's bindings:

    3 D + 3 fu_D + 3 H + 3 E + 3 f_w_E + 3 inverse epsilon      = 18
  + 6 curl coefficients (kms/sinv per axis)                     = 24
  + 6 constitutive coefficients (kps_h/kms_h per axis)          = 30
  + 8 scalars (nx, ny, nz, n_elem, dtdx, rx, ry, rz)            = 38

and :data:`.device.MAX_BUFFER_BINDINGS` is 31. Measured on this host (torch
2.10.0, metalfe-32023.850.10) on 2026-08-19 at five scalars and re-measured
2026-08-21 at eight, and the second row is what makes this family one dispatch
instead of three:

    38 separate bindings                    -> FAILED to compile, "'buffer'
                                               attribute parameter is out of
                                               bounds: must be between 0 and 30"
    30 pointers + one constant Params&      -> COMPILED, LAUNCHED, and every
                                               struct field read back correctly
                                               (nx/ny/nz/n_elem as uints, dtdx as
                                               a float and the three reflect rows
                                               as ints, from a 32-byte packed
                                               record bound as a device buffer)

THE THREE REFLECT ROWS COST NOTHING AGAINST THE CEILING because they ride inside
that record rather than as three more buffers: 31 before the far carry and 31
after. They stay RUNTIME for the certified separate fill's own reason
(``symmetry._MIRROR_FILL_TEMPLATE``) — the row is per-axis and integer, so unlike
the parity, which is a SOURCE specialisation because a runtime float weight flushes
this backend's subnormals, nothing about it can move a bit.

The launch half of that is this module's own measurement: the fused-dispersive
pair's docstring records the COMPILE result and stops there. The gate's
``binding_ceiling`` leg re-runs both rows so the shape of the product rests on a
measurement rather than on this paragraph.

``dtdx`` rides in the struct as float32. The array path multiplies a float32
volume by a Python float and NumPy's weak promotion casts that double to float32;
``constant float&`` does the same; ``numpy.float32(dtdx)`` in the struct is the
same bits again. One value, three spellings, no conversion difference.

DEVICE STATUS: GATED ON THIS HOST, RE-CUT 2026-08-21 FOR THE FAR CARRY. The gate is
``parity/meep_gpu/gate_metal_folded_fused_pair.py`` and it runs end to end on the
local Apple GPU: both fold terminations side by side, complete driver steps compared
as uint32 words against an array-path oracle running the identical live pass list,
the separate-composition control, the carry's own legs, the armed shader and host
defects and a disarm run. The artifact is
``results/folded_far_carry_d_2026-08-21/gate.json``, beside the two host
measurements the carry rests on — ``d_ownership.json`` and ``d_ownership_flip.json``.

STILL ``wired=False``, AND SINCE 2026-08-27 ROUTED ANYWAY — two different questions,
and this paragraph used to answer only the first. ``STEP_ORDER`` assigns at most one
ARM per slot and this product spans FIVE of them (``step_D``, ``fill_D``,
``zero_metal_D``, ``fill_folded_far_ghosts_D``, ``update_E``), so it can claim no
slot through the arm table and goes on registering ``wired=False``: ``arms.arms_for``
skips it and ``_select_slot`` cannot pick it, while the disjointness sweep can still
enumerate it through ``arms.registered``.

What it now has is the OTHER route. ``launch._install_fused_pairs``
(metal_kernels/launch.py:725-800) walks ``arms.registered("step_D")`` for ``is_weld``
rows whose ``replaces`` covers the partner slot, and ``launch.FUSED_PAIR_ARMS``
(launch.py:660-664) now carries the row ``folded_fused_pair: ("folded", "folded")`` —
the declaration of which arm each of the two slots implements, read off this module's
own predicate, which is exactly the ``folded`` curl arm's AND the ``folded``
constitutive arm's. Under ``plan_step(..., fuse=True)`` on a folded grid this product
therefore takes ``step_D`` and ``update_E``. The row licenses the SUBSTITUTION and
nothing else: the deposit repair is declared separately by
:data:`CARRIES_DEPOSIT_REPAIR`, which became ``True`` on 2026-08-28 once the repair
gained the closure of the cells a post-injection fill images a deposit into.

NOTHING ON THIS BACKEND DISPATCHES. ``meep_gpu.fastpath.plan_fast_path`` consults
``triton_kernels.plan_step`` alone and names no Metal module on any branch, so a
Metal plan — fused or not — is still built for measurement rather than for a run.

WHAT THE CORPUS SAYS THIS IS WORTH, measured rather than argued, from the 186-row
census (``parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19``): 53 rows
admit BOTH halves of this seam (``folded_pml_curl@step_D`` and
``folded_constitutive@update_E``), and 4 of those 53 declare no electric source. The
far carry is what takes this family from 2 of those 4 to all 4 — the other two are
``examples:binary_grating_phasemap.py`` and ``examples:diffracted_planewave.py``,
which the 2026-08-21 matrix lists by name under ``carry_gap`` on BOTH boards. The
electric-source clause — not the fold, not either fill and not the binding ceiling —
is what caps this family at 4, and that number is reported beside the gate rather
than left to be discovered.
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
    CODE_MIRROR_PERIODIC,
    MIRROR_SOURCE_INDEX,
    TARGET_IYEE,
    folded_axis_kinds,
)
from . import shaders, templates
# Bound as `_chunk_bounds`: the plan builder takes the RECORD as `chunk_bounds`.
from . import chunk_bounds as _chunk_bounds
from .device import Residency, compile_source
from .fused_dispersive_pair import _ZERO_METAL_ROWS
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
#: also runs ``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` AFTER the injection
#: (driver.py:3308-3311) -- and a repair that wrote only the
#: deposit index left every MIRROR IMAGE of that index carrying the constitutive result
#: the launch computed from the pre-injection field. That was measured, not assumed: on
#: a corpus row with the deposit one cell off the plane, the image cell came back
#: byte-equal to a run with no source at all. The flag moved because
#: ``deposit_repair.repair_cells`` now hands ``save``/``apply`` the CLOSURE of the cells
#: the two fills image each deposit point into -- the inverse of the forward carry
#: :func:`carried_destinations` already implements -- and the seam is byte-identical to
#: the array path over ``step_D`` -> ``update_E`` with it and diverges without it.
#: What is still refused by name: a source that publishes no deposit index, and a folded
#: grid whose fill map ``deposit_repair`` cannot read (the cylindrical r = 0 axis, a folded axis too short to hold the near fill's source row).
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_fused_pair"

#: The sub-step slot this arm holds a row on. It spans four; a refusal is NAMED on
#: the slot the fusion starts at rather than being invisible to the table.
SLOT = "step_D"

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304) and in :data:`.coverage.RESIDENCY_ORDER`'s spelling. Read
#: by the whole-step gate's ``covered_passes``; declared, never inferred from the
#: slot name.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: How many bindings an all-three-component fused folded D/E kernel needs with the
#: scalars bound SEPARATELY, and the ceiling it breaks. Spelled as data so the gate
#: compiles the refuted signature from the same number the docstring argues from.
#: EIGHT scalars since the far carry of 2026-08-21 — the three reflect rows are
#: scalars like the rest, and a count that stayed at 35 would be arguing from a
#: signature this module no longer has.
SEPARATE_SCALAR_BINDINGS = 38

#: How many it needs with the eight scalars PACKED into one ``constant Params&``.
#: This is the shipped signature and it is exactly the measured ceiling — UNCHANGED
#: by the far carry, which is why that carry costs no binding.
PACKED_BINDINGS = 31

#: The stored index the near fill images, re-exported so a reader of this module
#: does not have to chase ``triton_kernels.symmetry`` for the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX

#: The reflect row's runtime name per axis inside the kernel. Needed by the far
#: carry and NOT a new binding: the rows ride in the packed ``Params``.
_REFLECT: Tuple[str, str, str] = ("reflect_x", "reflect_y", "reflect_z")

#: The TOP STORED INDEX per axis and the boolean the curl head already declares for
#: it. ``_LAST`` is the integer the far destination's flat index is built from;
#: ``_LAST_FLAG`` is ``symmetry._TOP_PLANE_FLAGS``' own name, which the body already
#: declares for the top-plane curl mask and which the far carry reuses rather than
#: writing a second comparison against the same extent.
_LAST: Tuple[str, str, str] = ("(nxi - 1)", "(nyi - 1)", "(nzi - 1)")
_LAST_FLAG: Tuple[str, str, str] = ("last_x", "last_y", "last_z")

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY", "LIFT_EDITS",
    "NEAR_SOURCE_INDEX", "PACKED_BINDINGS", "REPLACES", "SLOT",
    "SEPARATE_SCALAR_BINDINGS", "MetalFoldedFusedPairPlan",
    "carried_destinations", "certified_constitutive_body", "certified_curl_head",
    "certified_curl_statements", "compile_folded_fused_pair",
    "constitutive_transcription", "far_fill_axes", "folded_fused_pair_source",
    "metal_folded_fused_pair_coverage", "near_fill_axes", "params_record_dtype",
    "plan_metal_folded_fused_pair", "refuted_separate_scalar_source",
    "zero_metal_lines",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

// EIGHT SCALARS, ONE BINDING, AND THE BINDING COUNT IS UNCHANGED AT 31. The three
// reflect rows the far carry needs (stepping._far_reflect_rows:1661) ride inside
// this record rather than as three more buffers, so carrying
// fill_folded_far_ghosts_D costs nothing against the platform's 31-binding ceiling
// (device.py:72) — which this family sits EXACTLY on and could not have paid. THE
// CHUNK RECORD rides here too, as one member after the scalars, for the same
// reason: the per-chunk forms read it as runtime data at no binding.
__RECORD_STRUCT__

struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;
                int rx; int ry; int rz;
__RECORD_MEMBER__
};

kernel void folded_fused_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device float*       e0      [[buffer(9)]],
    device float*       e1      [[buffer(10)]],
    device float*       e2      [[buffer(11)]],
    device float*       w0      [[buffer(12)]],
    device float*       w1      [[buffer(13)]],
    device float*       w2      [[buffer(14)]],
    device const float* ie0     [[buffer(15)]],
    device const float* ie1     [[buffer(16)]],
    device const float* ie2     [[buffer(17)]],
    device const float* kmx     [[buffer(18)]],
    device const float* sinvx   [[buffer(19)]],
    device const float* kmy     [[buffer(20)]],
    device const float* sinvy   [[buffer(21)]],
    device const float* kmz     [[buffer(22)]],
    device const float* sinvz   [[buffer(23)]],
    device const float* kp0     [[buffer(24)]],
    device const float* km0     [[buffer(25)]],
    device const float* kp1     [[buffer(26)]],
    device const float* km1     [[buffer(27)]],
    device const float* kp2     [[buffer(28)]],
    device const float* km2     [[buffer(29)]],
    constant Params&    prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE EIGHT SCALARS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES, once,
    // and the record is aliased as the `cb` both lifted bounds blocks read, before
    // any of the lifted text runs. Everything below this line is then
    // character-for-character what the folded curl and the certified constitutive
    // emit, rather than the same arithmetic re-spelled around a struct field.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    int reflect_x = prm.rx, reflect_y = prm.ry, reflect_z = prm.rz;
    (void)reflect_x; (void)reflect_y; (void)reflect_z;
__RECORD_ALIAS__

__BODY__
}
"""

#: Per target: the two coefficient axes its split-field recurrence takes.
#: TRANSCRIBED from symmetry.py:432-448 — ``vec.hpp``'s cycle_direction, the same
#: triple on both curl sides: target 0 takes (y, z), target 1 (z, x), target 2
#: (x, y). Spelled as data because three components in one body would otherwise
#: repeat the pairing three times in string literals.
_RECURRENCE_AXES: Tuple[Tuple[str, str], ...] = (("y", "z"), ("z", "x"), ("x", "y"))

#: The flat-index coordinate name per axis, and the stride expression per axis, in
#: the lifted decode's own spelling. ``nyz`` and ``nzi`` are that block's variables;
#: a second decode here would be a second place to get the C-contiguous layout
#: wrong.
_COORDINATE: Tuple[str, str, str] = ("i", "j", "k")
_STRIDE: Tuple[str, str, str] = ("nyz", "nzi", "1")

#: The curl sub-step this product starts at, its direction, and the constitutive
#: side it ends at. ``BACKWARD`` is ``SUB_STEPS['step_D']['backward']``, restated so
#: the one legal binding is visible here; the test file pins the two equal.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"
BACKWARD = True

#: The Yee sub-lattice the E half's membership reads, per ``shaders``' own table:
#: half-integer, on the component's own axis. The curl half's lattice is the lifted
#: head's (integer, on each target's dsigu axis), so the two blocks declare disjoint
#: names (``cbi_*``/``in_i_*`` against ``cbh_*``/``in_h_*``).
CONSTITUTIVE_HALF_INTEGER = shaders.CONSTITUTIVE_HALF_INTEGER[CONSTITUTIVE_SIDE]

#: The marker that separates a certified shader's signature from its body. Every
#: template in this package ends its parameter list with this exact line.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The comment line the folded curl body opens its per-chunk form block with: the
#: CUT POINT of the lift. Everything above it is spliced verbatim; everything below
#: is re-emitted per component around the ownership carve-out.
_FORMS_MARK = "    // --- the two PML forms"

#: The decode line the certified constitutive body's prologue ends on. The lift of
#: that body starts AFTER it, because the curl head has already decoded the index.
_CONSTITUTIVE_PROLOGUE_END = "    int i   = plane / nyi;\n"

#: THE TWO DECLARED EDITS of the lifted curl statements, per target ``t``:
#: ``(statement prefix, old fragment, new fragment)``. Everything else below the cut
#: is the certified folded curl's text verbatim.
#:
#: The certified ``p`` statement reads ``f`` off the recurrence's cells (the
#: dsig-only form is the recurrence's first stage with ``fu := f``). Fused, a
#: DESTINATION thread runs that statement too, and ``f`` at a destination is a word
#: its source thread writes in the same dispatch, so ``p`` takes ``0.0f`` there and
#: the read moves into the owned ``v``: ``{dsig_only}`` is the certified ``n``
#: statement's own expression with ``p{t}`` replaced by ``f{t}[ii]`` — the same
#: operations in the same order, so the same word as the certified ``v``.
LIFT_EDITS: Tuple[Tuple[str, str, str], ...] = (
    ("    float p{t} = ", ": f{t}[ii];", ": 0.0f;"),
    ("    float v{t} = ", ": n{t};", ": {dsig_only};"),
)

#: The six statements ``_apply_constitutive_regions`` runs for one E component, as
#: they appear in the certified E body once its pointers are renamed for this kernel
#: (the inverse permittivity ``e`` -> ``ie``, then the target ``f`` -> ``e``): the
#: accumulation ``f + ((kps*src) - (kms*prev))`` with ``f_w = src`` where the
#: component's own layer holds, and the output select ``e = src`` elsewhere, ``f_w``
#: neither read nor written there. Parameterised on the flat index, the displacement
#: register, the coefficient pair, the membership flag and a register prefix — the
#: FAR ghost takes the identical six at a different index with its own pair and its
#: own flag, which is the D geometry's one asymmetry.
#:
#: NOT A RETYPING: :func:`constitutive_transcription` renders these with the
#: certified names and compares them statement for statement against
#: :func:`certified_constitutive_body`, and :func:`folded_fused_pair_source` runs
#: that comparison on every build.
_CONSTITUTIVE_STATEMENTS: Tuple[str, ...] = (
    "float {prefix}prev = {dw} ? w{target}[{index}] : 0.0f;",
    "float {prefix}src = {value} * ie{target}[{index}];",
    "if ({dw}) {{ w{target}[{index}] = {prefix}src; }}",
    "float {prefix}acc = {prefix}src;",
    "if ({dw}) {{ {prefix}acc = e{target}[{index}] + "
    "(({kp} * {prefix}src) - ({km} * {prefix}prev)); }}",
    "e{target}[{index}] = {prefix}acc;",
)

#: How the certified E body spells each of the six for component ``t``: the register
#: names ``prev{t}`` / ``src{t}`` / ``a{t}``, the flat index ``ii``, the displacement
#: ``g{t}[ii]``, the pair ``kp_{t}`` / ``km_{t}`` and the flag ``dw{t}``.
_CERTIFIED_PREFIX = "@"
_CERTIFIED_NAMES: Dict[str, str] = {
    "@prev": "prev{target}", "@src": "src{target}", "@acc": "a{target}",
}


def near_fill_axes(codes: Sequence[int], target: int) -> Tuple[int, ...]:
    """Which folded axes ``fill_symmetry_bc_D`` writes for ONE D component.

    ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1441-1447) fills axis
    ``a`` for component ``m`` exactly when ``a`` is mirrored and ``iyee[m][a] == 0``.
    Read off ``TARGET_IYEE`` rather than spelled: a literal here would be another
    transcription of the Yee table, and this predicate decides which threads own
    which cells.

    BOTH mirror codes are matched: the near fill runs on any folded axis
    (``stepping._fill_symmetry_ghost_cells`` gates on ``_mirror_phases`` alone),
    which is exactly what separates it from :func:`far_fill_axes`.

    For the D family the answer is up to TWO axes — the two that are NOT the
    component's own — and :func:`folded_fused_pair_source` refuses a longer one
    rather than trusting that.
    """
    names = SUB_STEPS["step_D"]["targets"]
    return tuple(axis for axis, code in enumerate(codes)
                 if int(code) in MIRROR_CODES
                 and TARGET_IYEE[names[target]][axis] == 0)


def far_fill_axes(codes: Sequence[int], target: int) -> Tuple[int, ...]:
    """Which axes ``fill_folded_far_ghosts_D`` images for ONE D component.

    ``stepping._fill_folded_far_ghosts`` (stepping.py:1516-1534) writes axis ``a``
    for component ``m`` exactly when ``_stored_past_owned(grid, a)`` — a folded
    PERIODIC axis, at either full-count parity — and ``iyee[m][a] == 1``. The two
    conditions are read here off ``CODE_MIRROR_PERIODIC`` (which
    ``folded_axis_kinds`` derives from ``_stored_past_owned`` itself, and
    cross-checks against ``grid.is_metallic``) and off ``TARGET_IYEE``, so neither
    the fold split nor the Yee table is transcribed a second time.

    THE EXACT COMPLEMENT OF :func:`near_fill_axes` ON THIS FAMILY, AND IT IS THE
    MIRROR IMAGE OF THE MAGNETIC TWIN'S. A D component's Yee shift is 1 on its OWN
    axis and 0 on the other two, so the near fill touches component ``m`` on the two
    axes that are NOT its own while the far fill touches it on axis ``m`` alone — at
    most ONE destination plane per component, against the near fill's two, and
    therefore no far corner and no far parity product. On the B family the two sets
    are the other way round, which is why :func:`.folded_fused_magnetic_pair.far_fill_axes`
    can return two and this one cannot.
    """
    names = SUB_STEPS["step_D"]["targets"]
    return tuple(axis for axis, code in enumerate(codes)
                 if int(code) == CODE_MIRROR_PERIODIC
                 and TARGET_IYEE[names[target]][axis] == 1)


def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], bool], ...]:
    """Every ghost cell ONE source thread owns, as ``(near axes at 0, far?)``.

    THE OWNERSHIP RULE, MEASURED RATHER THAN ARGUED
    (``parity/meep_gpu/results/folded_far_carry_d_2026-08-21/d_ownership.json``,
    thirteen configurations, zero differing words, with five planted variants of the
    same form each shown to produce them in ``d_ownership_flip.json``). The driver
    runs the near fill, the wall clear and the far fill in that order
    (driver.py:3300-3302), the near fill is applied axis by axis, and the far fill
    then images the whole top plane of the component's own axis. Composing the three
    gives one closed form: for component ``m`` the seam leaves, at the cell that sits
    at stored 0 on the near-axis subset ``S`` and at ``last`` on its own axis when
    ``z``,

        (z ? mirror_parity(m, m, phase_m) : 1)
            * mask( (prod over a in S of mirror_parity(m, a, phase_a))
                    * v_m(that cell with every a in S moved to stored 2 and, when z,
                          the own axis moved to reflect_m) )

    — independent of the order the array path applies its near axes in, which is why
    nothing here has to model ``_fill_symmetry_ghost_cells``' X/Y/Z loop. So ONE
    thread, the one at the fully back-substituted cell, computes the displacement
    every one of those ghosts carries, and it writes them all. Every destination word
    is written by exactly one thread and no thread reads a word another writes, which
    is what makes the carry legal inside a single Metal dispatch with no device-wide
    barrier.

    THE MASK SITS INSIDE THE FAR PARITY AND OUTSIDE THE NEAR ONE, and that is the
    driver's order rather than a convention: the fill at :3300 runs BEFORE the clear
    at :3301 and the far fill at :3302 runs after it. On a cleared plane at an odd
    parity the two spellings differ in the sign bit of a zero, measured at 12 and 27
    differing words when each is written the other way round.

    Returned smallest-subset-first for a stable emission order; the order is
    unobservable (the cells are distinct) and a stable one keeps a source diff
    readable.
    """
    near = tuple(int(axis) for axis in near)
    combos: List[Tuple[Tuple[int, ...], bool]] = []
    for size in range(len(near) + 1):
        for subset in itertools.combinations(near, size):
            for carries_far in ((False, True) if far else (False,)):
                if not subset and not carries_far:
                    continue  # the thread's OWN cell, not a ghost
                combos.append((subset, carries_far))
    return tuple(sorted(combos, key=lambda item: (len(item[0]), item[0], item[1])))


def _component_parity(target: int, axis: int, phase: int) -> int:
    """``fields.mirror_parity`` for one D component about one plane — the ONE table.

    ``stepping._symmetry_phase`` (stepping.py:2457-2471) is exactly this call, and
    both fills weight their image with it. NOT spelled as ``+phase`` for the near
    fill and ``-phase`` for the far one: that collapse is TRUE (``mirror_parity`` is
    ``phase * (1 - 2 * iyee)``, fields.py:117, and a D component's shift is 1 on its
    own axis alone) but writing it out here would be a second copy of the Yee table
    with a sign in it, and the far carry is the change that makes both signs live in
    the same kernel.
    """
    name = SUB_STEPS["step_D"]["targets"][target]
    return int(mirror_parity(name, axis, int(phase)))


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    The lift's whole safety property: a missing anchor is a certified emitter that
    has changed under this family, and TWO matches would mean the anchor no longer
    identifies a single statement.
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
    """The folded ``step_D`` curl body DOWN TO the per-chunk PML forms, verbatim.

    :func:`.symmetry.folded_curl_source`'s own output at ``backward=True`` with the
    ``#include``/signature preamble removed and the cut taken at
    :data:`_FORMS_MARK`: the guard, the index decode, the chunk record's bounds
    block (integer-lattice membership and partner presence), the ghost gather
    (``templates.ghost`` through ``symmetry._reduced_codes``), the curl and its
    absent-partner forms, the cell-0 ownership mask and the top-plane mask that
    :func:`.symmetry.folded_top_plane_mask` emits on a folded PERIODIC axis.
    """
    body = _body_of(folded_curl_source(codes, BACKWARD, contract), "folded curl")
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(_FORMS_MARK):
            head = "".join(lines[:index])
            if not head.strip():
                raise AssertionError("the lifted folded curl head is empty")
            return head
    raise AssertionError(
        f"the certified folded curl body no longer opens its per-chunk form block "
        f"with {_FORMS_MARK!r}; this family cuts the lift there")


def _lift_edit(line: str, target: int, old: str, new: str) -> str:
    """One declared edit of one lifted statement, refused unless it lands exactly once."""
    old = old.format(t=target)
    if line.count(old) != 1 or not line.rstrip().endswith(old):
        raise AssertionError(
            f"the declared lift edit {old!r} does not end the certified statement "
            f"{line!r} exactly once; the certified folded curl has changed under it")
    return line[: line.rindex(old)] + new


def certified_curl_statements(codes: Sequence[int],
                              contract: str = shaders.CONTRACT_OFF,
                              ) -> Dict[str, Any]:
    """The per-chunk form's own lines, pulled out of the certified folded D curl.

    The coefficient loads, the dsigu membership flags, the three ``p``/``n``/``v``
    statements (the ``p``/``v`` pair through :data:`LIFT_EDITS`), the three guarded
    ``fu`` stores and the flux store split per target, each matched by an anchor that
    must identify exactly one line.
    """
    body = _body_of(folded_curl_source(codes, BACKWARD, contract), "folded curl")
    coefficients = [_line_starting(body, f"    float km_{axis} = ",
                                   f"the {axis} split-field coefficient load")
                    for axis in "xyz"]
    flags = _line_starting(body, "    const bool du0 = ",
                           "the three dsigu membership flags")
    (p_prefix, p_old, p_new), (v_prefix, v_old, v_new) = LIFT_EDITS
    recurrence = []
    for target in range(3):
        previous = _line_starting(body, p_prefix.format(t=target),
                                  f"target {target}'s previous split field")
        stage = _line_starting(body, f"    float n{target} = ",
                               f"target {target}'s split-field recurrence")
        stepped = _line_starting(body, v_prefix.format(t=target),
                                 f"target {target}'s stepped displacement")
        expression = stage.split(" = ", 1)[1].rstrip()
        if not expression.endswith(";") or expression.count(f"p{target}") != 1:
            raise AssertionError(
                f"target {target}'s recurrence statement {stage!r} no longer reads "
                f"p{target} once; the dsig-only form cannot be derived from it")
        dsig_only = "(" + expression[:-1].replace(f"p{target}", f"f{target}[ii]") + ")"
        recurrence.append((
            _lift_edit(previous, target, p_old, p_new),
            stage,
            _lift_edit(stepped, target, v_old, v_new.format(dsig_only=dsig_only)),
        ))
    aux_stores = tuple(_line_starting(body, f"    if (du{target}) {{ u{target}[ii] = ",
                                      f"target {target}'s guarded fu store")
                       for target in range(3))
    flux_store = _line_starting(body, "    f0[ii] = ", "the three flux stores")
    stores = tuple(f"{part.strip()};" for part in flux_store.strip().split(";")
                   if part.strip())
    if len(stores) != 3:
        raise AssertionError(
            f"the certified folded curl no longer stores three targets on one "
            f"line: {flux_store!r}")
    return {"coefficients": tuple(coefficients), "flags": flags,
            "recurrence": tuple(recurrence), "aux_stores": aux_stores,
            "stores": stores}


def certified_constitutive_body(contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``update_E`` constitutive kernel's BODY, lifted and renamed for this kernel.

    ``shaders.constitutive_source("E")``'s own body with the decode prologue dropped
    (the curl head already decoded the index) and two pointer renames, in this
    order: the inverse permittivity ``e{t}`` -> ``ie{t}``, then the target ``f{t}``
    -> ``e{t}`` (fused, ``f`` is D). No arithmetic is edited.
    """
    body = _body_of(shaders.constitutive_source(CONSTITUTIVE_SIDE, contract),
                    "constitutive")
    if _CONSTITUTIVE_PROLOGUE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer carries its decode prologue; "
            "the lift would redeclare the curl half's indices")
    body = body.split(_CONSTITUTIVE_PROLOGUE_END, 1)[1]
    for old_stem, new_stem in (("e", "ie"), ("f", "e")):
        for target in range(3):
            old, new = f"{old_stem}{target}[ii]", f"{new_stem}{target}[ii]"
            if old not in body:
                raise AssertionError(f"the certified constitutive body has no {old}")
            body = body.replace(old, new)
    return body


def _constitutive_bounds_lines(contract: str = shaders.CONTRACT_OFF
                               ) -> Tuple[str, ...]:
    """The certified E body's chunk-record block, BEGIN to END, verbatim."""
    lines = certified_constitutive_body(contract).splitlines()
    begin = [index for index, line in enumerate(lines)
             if line.strip() == _chunk_bounds.BLOCK_BEGIN]
    end = [index for index, line in enumerate(lines)
           if line.strip() == _chunk_bounds.BLOCK_END]
    if len(begin) != 1 or len(end) != 1 or end[0] <= begin[0]:
        raise AssertionError(
            "the certified E body no longer carries exactly one chunk-record block; "
            "this family lifts it rather than retyping it")
    return tuple(lines[begin[0]: end[0] + 1])


def _constitutive_flag_line(contract: str = shaders.CONTRACT_OFF) -> str:
    """``const bool dw0 = in_h_x, dw1 = in_h_y, dw2 = in_h_z;``, lifted."""
    return _line_starting(certified_constitutive_body(contract), "    const bool dw0 = ",
                          "the three dsigw membership flags")


def _constitutive_coefficient_lines(contract: str = shaders.CONTRACT_OFF
                                    ) -> Tuple[str, ...]:
    """``kp_t``/``km_t`` on the component's OWN axis, lifted from the certified body."""
    body = certified_constitutive_body(contract)
    return tuple(_line_starting(body, f"    float kp_{target} = ",
                                f"target {target}'s constitutive coefficient pair")
                 for target in range(3))


def _render_constitutive(target: int, index: str, value: str, prefix: str,
                         kp: str, km: str, dw: str) -> Tuple[str, ...]:
    """The six constitutive statements for one component at one cell, unindented."""
    return tuple(line.format(target=target, index=index, value=value, kp=kp, km=km,
                             prefix=prefix, dw=dw)
                 for line in _CONSTITUTIVE_STATEMENTS)


def _touches(statement: str, target: int) -> bool:
    """Does one certified constitutive statement belong to component ``target``?

    The block, the coefficient loads and the flags are declarations (``const``,
    ``float kp_``) naming several components and are lifted separately; every other
    line names its component in a register or a pointer.
    """
    if statement.startswith(("float kp_", "const ")):
        return False
    marks = (f"prev{target}", f"src{target}", f"a{target} ", f"w{target}[",
             f"e{target}[")
    return any(mark in statement for mark in marks)


def constitutive_transcription(contract: str = shaders.CONTRACT_OFF
                               ) -> Dict[str, Any]:
    """Is :data:`_CONSTITUTIVE_STATEMENTS` the certified E body, statement for statement?

    This family re-emits the certified E arithmetic at the owned cell and at every
    imaged ghost, so it cannot splice the body wholesale; it renders its own
    template with the CERTIFIED spellings and requires exactly the lifted body's
    statements back. ``identical`` false is a build-time failure:
    :func:`folded_fused_pair_source` raises on it.
    """
    body = certified_constitutive_body(contract)
    statements = [line.strip() for line in body.splitlines()
                  if line.strip() and not line.strip().startswith("//")]
    certified: List[List[str]] = []
    emitted: List[List[str]] = []
    for target in range(3):
        rendered = list(_render_constitutive(
            target, "ii", f"g{target}[ii]", _CERTIFIED_PREFIX, f"kp_{target}",
            f"km_{target}", f"dw{target}"))
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


def zero_metal_lines(target: int, zero_metal: Sequence[bool], name: str,
                     indent: str = "    ", zero: str = "0.0f") -> List[str]:
    """``stepping.zero_metal_D`` for ONE target, on one named register.

    The rows are :data:`.fused_dispersive_pair._ZERO_METAL_ROWS`, IMPORTED rather
    than re-derived: that table is already the complement of the D family's Yee
    diagonal and is already pinned against the array path. Only the variable name
    is a parameter here, because this family applies the same clear to two
    registers — the cell the thread owns and the ghost cell it images.

    The select spelling is :func:`.templates.ownership_mask`'s, ``flag ? 0.0f : v``,
    which is the form measured to deliver an exact ``+0.0`` on this platform.
    """
    return [f"{indent}{name} = {flag} ? {zero} : {name};"
            for row_target, axis, flag in _ZERO_METAL_ROWS
            if row_target == target and bool(zero_metal[axis])]


def _constitutive_block(target: int, index: str, value: str, indent: str,
                        prefix: str, kp: Optional[str] = None,
                        km: Optional[str] = None,
                        dw: Optional[str] = None) -> List[str]:
    """``_apply_constitutive_regions`` for one component at one cell.

    The certified E body's six statements (:data:`_CONSTITUTIVE_STATEMENTS`, checked
    against ``shaders.constitutive_source('E')`` by :func:`constitutive_transcription`):
    on a cell whose chunk carries the component's own layer, ``prev`` is read BEFORE
    ``fw`` is written and the right-hand side is formed first, ``e + ((kps*src) -
    (kms*prev))``; elsewhere ``e = src``, an output select, and ``fw`` is not
    touched. The source is ``D * inv_eps`` with D on the LEFT.

    ``kp``/``km``/``dw`` default to the thread's own pair and flag, which is what a
    NEAR ghost takes: it differs from its source only along an axis that is never
    the component's own, and both are indexed on the component's own axis. A FAR
    ghost moves ALONG that axis and passes its own (:func:`_far_coefficient_lines`,
    :func:`_far_membership`) — the one asymmetry the D geometry forces and the
    exact inverse of the magnetic twin's.
    """
    kp = f"kp_{target}" if kp is None else kp
    km = f"km_{target}" if km is None else km
    dw = f"dw{target}" if dw is None else dw
    return [indent + line
            for line in _render_constitutive(target, index, value, prefix, kp, km, dw)]


def _far_coefficient_lines(target: int, axis: int, tag: str,
                           indent: str) -> List[str]:
    """The FAR destination's constitutive coefficient pair, read at stored ``last``.

    THE ONE THING :mod:`.folded_fused_magnetic_pair` NEEDS FOR ITS NEAR GHOST AND
    THIS FAMILY NEEDS FOR ITS FAR ONE. ``shaders._CONSTITUTIVE_TEMPLATE`` indexes
    target ``t`` on axis ``t`` (shaders.py:310-312, ``stepping.E_CONSTITUTIVE_TERMS``)
    and :func:`far_fill_axes` returns axis ``t`` ALONE, so the far destination sits
    at a DIFFERENT coordinate on the indexed axis from the thread that owns it.
    Reusing the source thread's ``kp_t``/``km_t`` would apply the absorber profile of
    the reflect row to the top plane; the pair is therefore loaded again at
    ``last``. The gate arms that reuse as a mutation.

    AND UNLIKE THE MAGNETIC TWIN'S NEAR CASE THIS IS NORMALLY OBSERVABLE, which is
    the second half of the inversion: a folded axis carries an absorber on its HIGH
    face only (``stepping._require_consistent_pml``), so the mirror plane at stored 0
    is outside the layer and its ``kps`` entries are the identity, while the top
    plane the far ghost lands on is INSIDE it. Observability is still censused and
    predicted by the gate rather than asserted here — an axis with no absorber at all
    makes the two entries the same word again.

    THE NEAR GHOST NEEDS NO SUCH LINE, and that asymmetry is derived rather than
    asserted: :func:`near_fill_axes` returns only axes ``a != t``, so a near
    destination sits at the SAME coordinate on the indexed axis as its owner and
    takes that thread's pair unchanged. A carry that reloaded there would be
    reloading the identical word.
    """
    if axis != target:
        raise AssertionError(
            f"component {target} images its FAR ghost on axis {axis}; for the D "
            f"family the far fill only ever touches a component's OWN axis "
            f"(fields.IYEE_SHIFTS)")
    return [
        f"{indent}// The DESTINATION's own coefficient entry: stored index "
        f"{_LAST[axis]} on",
        f"{indent}// {_COORDINATE[axis]}, NOT the source thread's at "
        f"{_REFLECT[axis]} (shaders._CONSTITUTIVE_TEMPLATE",
        f"{indent}// indexes target {target} on {_COORDINATE[axis]}).",
        f"{indent}float {tag}_kp = kp{target}[{_LAST[axis]}], "
        f"{tag}_km = km{target}[{_LAST[axis]}];",
    ]


def _far_membership(target: int, axis: int, tag: str,
                    indent: str) -> Tuple[List[str], str]:
    """The FAR destination's own dsigw membership, evaluated at stored ``last``.

    The E half selects its form on the component's OWN axis, the axis the far fill
    moves, so the select is evaluated where the destination cell is — beside the
    coefficient pair :func:`_far_coefficient_lines` reloads there and for the same
    reason. Emitted by :func:`.chunk_bounds.membership_lines` with the destination's
    coordinate on that axis and a per-ghost suffix, reading the ``cbh_*`` locals the
    lifted block declares. The record's line is ghost-adjusted, so the far ghost
    slot reads its image row's membership, as the array path's line does.

    Returns the declaration lines and the flag's name.
    """
    if axis != target:
        raise AssertionError(
            f"component {target} images its FAR ghost on axis {axis}; the E half's "
            f"membership is on the component's OWN axis (fields.IYEE_SHIFTS)")
    coordinates = list(_COORDINATE)
    coordinates[axis] = _LAST[axis]
    suffix = f"_{tag}"
    lines = _chunk_bounds.membership_lines(
        [(CONSTITUTIVE_HALF_INTEGER, axis)], coordinates, suffix, indent).splitlines()
    return lines, _chunk_bounds.membership_name(CONSTITUTIVE_HALF_INTEGER, axis, suffix)


def _carry_blocks(target: int, near: Sequence[int], far: Sequence[int],
                  phases: Sequence[int], zero_metal: Sequence[bool],
                  indent: str) -> List[str]:
    """Every imaged ghost this thread owns, then ``update_E`` at each of them.

    ONE BLOCK PER DESTINATION IN :func:`carried_destinations`. Each is guarded on the
    flags that say this thread is the source of that destination — stored
    ``NEAR_SOURCE_INDEX`` on each near axis it images, the runtime reflect row on the
    far axis — and writes the composed value at the destination's flat index,
    followed by the certified constitutive at that index.

    ``v{target}`` IS READ AFTER THE WALL CLEAR, deliberately, and the driver order is
    why: ``zero_metal_D`` (driver.py:3301) runs BEFORE
    ``fill_folded_far_ghosts_D`` (:3302), so a far image of a cleared row must carry
    the cleared value. ``stepping._write_mirror_ghost`` (stepping.py:1497-1498) writes
    ``mirror_parity * cell 2`` and the near fills compose in X, Y, Z order
    (stepping.py:1441-1447), so a cell unowned on both near planes carries the PRODUCT
    of their parities imaged from the doubly shifted cell. Every parity is a
    COMPILE-TIME sign — :func:`.symmetry._parity_spelling`'s measured rule, ``-x`` for
    odd and a plain copy for even, never a runtime weight.

    THE TWO FILLS TAKE OPPOSITE ORDERS AROUND THE WALL CLEAR, and both are emitted
    here rather than collapsed:

    * the NEAR ghost is ``mask(parity * v)``. The driver fills at :3300 BEFORE it
      clears at :3301, and the near destination — stored 0 of a folded axis that is
      not the component's own — CAN also sit at stored 0 of a walled axis, because
      ``_ZERO_METAL_ROWS`` is the complement of the D diagonal and reaches exactly
      the same shift-0 axes the near fill does. The clear is therefore emitted at the
      near ghost;
    * the FAR ghost is ``parity * mask(v)``, with NO clear line of its own. It moves
      along the component's OWN axis, which ``_ZERO_METAL_ROWS`` never clears, so it
      sits at the same coordinate on every axis that could clear it as the thread
      that owns it — its clear status is that thread's, INHERITED by reading
      ``v{target}`` after the clear lines rather than re-emitted, which is the
      driver's own order. What is asserted below is the disjointness that makes the
      inheritance exact.

    On a cleared plane at an odd parity those two spellings differ in the sign bit of
    a zero, which is a byte difference and nothing else; each is a gate mutation, not
    a comment.
    """
    walls = tuple(axis for row_target, axis, _ in _ZERO_METAL_ROWS
                  if row_target == target and bool(zero_metal[axis]))
    overlap = sorted(set(walls) & set(int(axis) for axis in far))
    if overlap:
        raise AssertionError(
            f"component {target} images a FAR ghost along axes {tuple(far)} that "
            f"zero_metal_D also clears on {overlap}; _ZERO_METAL_ROWS is the "
            f"complement of the D diagonal and far_fill_axes returns the component's "
            f"OWN axis, so the ghost would no longer inherit its source thread's "
            f"clear")
    if set(walls) & set(int(axis) for axis in near):
        raise AssertionError(
            f"component {target} images a NEAR ghost on folded axes {tuple(near)} "
            f"that zero_metal_D also clears on {walls}; stepping._zero_metal skips a "
            f"folded axis (stepping.py:2282-2284), so the two tables have drifted")
    inner = indent + "    "
    lines: List[str] = []
    for subset, carries_far in carried_destinations(near, far):
        tag = ("g" + "".join(_COORDINATE[axis] for axis in subset)
               + ("f" if carries_far else ""))
        flags = [f"({_COORDINATE[axis]} == {NEAR_SOURCE_INDEX})" for axis in subset]
        terms = [f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}" for axis in subset]
        weight = 1
        for axis in subset:
            weight *= _component_parity(target, axis, phases[axis])
        described = [
            f"the near fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = 0 imaged "
            f"from {NEAR_SOURCE_INDEX} (stepping._fill_symmetry_ghost_cells:1441, "
            f"_write_mirror_ghost:1449)" for axis in subset]
        if carries_far:
            axis = int(far[0])
            flags.append(f"({_COORDINATE[axis]} == {_REFLECT[axis]})")
            terms.append(f"+ ({_LAST[axis]} - {_REFLECT[axis]}) * {_STRIDE[axis]}")
            described.append(
                f"the far fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = "
                f"{_LAST[axis]} imaged from {_REFLECT[axis]} "
                f"(stepping._fill_folded_far_ghosts:1516)")
        far_weight = (_component_parity(target, int(far[0]), phases[int(far[0])])
                      if carries_far else 1)
        # `zero_metal_D` on the NEAR image: the driver clears AFTER the near fill
        # (:3301 after :3300), so the mask sits OUTSIDE the near parity. A
        # destination with no near axis is written by the far fill, which runs
        # AFTER the clear, and takes no clear line at all.
        cleared = (zero_metal_lines(target, zero_metal, f"{tag}_v", inner)
                   if subset else [])
        lines.append(f"{indent}if ({' && '.join(flags)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(f"{inner}// near parity {weight:+d}"
                     + (f", far parity {far_weight:+d}" if carries_far else "")
                     + (", clear between them (driver.py:3301 between :3300 and "
                        ":3302)." if cleared and carries_far
                        else ", clear applied AFTER it." if cleared
                        else ", no wall clears this target so the two compose."
                        if carries_far
                        else ", no wall clears this target."))
        lines.append(f"{inner}int {tag}_i = ii {' '.join(terms)};")
        if cleared:
            lines.append(f"{inner}float {tag}_v = "
                         f"{_parity_spelling(weight, f'v{target}')};")
            lines.extend(cleared)
            if far_weight == -1:
                lines.append(
                    f"{inner}// the far parity, applied to the ALREADY cleared "
                    f"value (driver.py:3302 after :3301).")
                lines.append(f"{inner}{tag}_v = -{tag}_v;")
        else:
            # The mask is the identity for this component, so the two parities
            # compose into ONE sign — the same collapse the near fill already makes
            # across its own axes, and bit-exact because a parity is a sign flip.
            lines.append(f"{inner}float {tag}_v = "
                         f"{_parity_spelling(weight * far_weight, f'v{target}')};")
        lines.append(f"{inner}f{target}[{tag}_i] = {tag}_v;")
        if carries_far:
            lines.extend(_far_coefficient_lines(target, int(far[0]), tag, inner))
            kp, km = f"{tag}_kp", f"{tag}_km"
            membership, dw = _far_membership(target, int(far[0]), tag, inner)
            lines.extend(membership)
        else:
            # The near destination shares the indexed axis with its source thread,
            # so the pair and the flag this thread already holds ARE the
            # destination's.
            kp, km, dw = f"kp_{target}", f"km_{target}", f"dw{target}"
        lines.extend(_constitutive_block(target, f"{tag}_i", f"{tag}_v", inner,
                                         f"{tag}_", kp, km, dw))
        lines.append(f"{indent}}}")
    return lines


def folded_fused_pair_source(codes: Sequence[int], phases: Sequence[int],
                             zero_metal: Sequence[bool],
                             contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded boundary quadruple, parities, walls, mode).

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC /
    MIRROR_PERIODIC quadruple :func:`.symmetry.folded_axis_kinds` resolves — never a
    hand-built triple, because the MIRROR_METALLIC / MIRROR_PERIODIC split is this
    family's single point of failure and getting it backwards on one axis is a
    plane of wrong values rather than a crash.

    ``phases`` is ``grid.mirror_phase(axis)`` per axis, +1 or -1 on a folded axis
    and ignored elsewhere. It is a SOURCE specialisation and not a runtime scalar,
    which is the opposite of the Triton folded families' rule and is measured:
    a runtime weight flushes every subnormal on this backend at BOTH signs
    (symmetry.py:657-686).

    Every constant Triton would bake into a ``tl.constexpr`` is baked into the
    string, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes
    a source and nothing else.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if not any(code in MIRROR_CODES for code in codes):
        raise ValueError(
            "no axis is folded: this product exists to carry the mirror fills "
            "inside the D seam, and an unfolded grid belongs to the certified "
            "curl and constitutive pair")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    phases = tuple(int(value) if value is not None else 0 for value in phases)
    if len(phases) != 3:
        raise ValueError(f"phases must be a per-axis triple, got {phases!r}")
    for axis, code in enumerate(codes):
        if int(code) in MIRROR_CODES and phases[axis] not in (1, -1):
            raise ValueError(
                f"axis {axis} is folded but its mirror phase is {phases[axis]!r}; "
                f"a plane's parity is +1 or -1 and the even-mirror default standing "
                f"in for a plane that declared otherwise is a run wrong by twice "
                f"the field wherever the parity mattered")
    # THE DISJOINTNESS THE CARRY RESTS ON, asserted rather than trusted:
    # `zero_metal_axes` is `is_metallic and not is_mirrored` (stepping.py:2282-2284),
    # so a folded axis can never be a walled one. If that ever changed, the imaged
    # ghost's wall flags would stop being its source's and this kernel would be
    # silently wrong on one edge rather than loudly wrong anywhere.
    for axis, code in enumerate(codes):
        if int(code) in MIRROR_CODES and zero_metal[axis]:
            raise ValueError(
                f"axis {axis} is reported both folded and walled; stepping._zero_metal "
                f"excludes a folded axis by construction (stepping.py:2282-2284) and "
                f"this kernel's ghost carry relies on the two sets being disjoint")

    transcription = constitutive_transcription(contract)
    if not transcription["identical"]:
        raise AssertionError(
            "this family's parameterised constitutive statements no longer "
            "reproduce shaders.constitutive_source('E'); the fused arithmetic "
            "would silently stop being the certified arithmetic. emitted="
            f"{transcription['emitted']!r} certified={transcription['certified']!r}")

    statements = certified_curl_statements(codes, contract)
    body: List[str] = [certified_curl_head(codes, contract).rstrip("\n")]
    body.extend([
        "",
        "    // The top-plane flags the lifted head declares are read below by the far",
        "    // carry's ownership guard where a folded PERIODIC axis exists; a (void)",
        "    // keeps the others from warning.",
        "    (void)last_x; (void)last_y; (void)last_z;",
        "",
        "    // --- the E half's chunk record: each component's own-axis membership on",
        "    // the HALF-INTEGER lattice, lifted from the certified E body (cbh_*/in_h_*,",
        "    // beside the curl head's integer cbi_*/in_i_*).",
    ])
    body.extend(_constitutive_bounds_lines(contract))
    body.extend([
        "",
        "    // --- the two PML forms, per structure chunk (stepping._apply_curl_regions)",
        "    // The certified folded curl's own statements; a target whose dsigu",
        "    // membership holds steps the recurrence, any other the dsig-only form, and",
        "    // fu is read and written only on the recurrence's cells.",
    ])
    body.extend(statements["coefficients"])
    body.append(statements["flags"])
    body.extend([
        "",
        "    // --- the constitutive coefficient and membership, on the OWN axis ----",
        "    // shaders._CONSTITUTIVE_TEMPLATE's own lines. The NEAR imaged cell takes",
        "    // the SAME entry and flag: it differs from its source only on a folded",
        "    // axis whose iyee is 0, which is never the component's own. The FAR",
        "    // imaged cell does NOT: it moves along exactly the axis these are indexed",
        "    // on, and loads its own pair and evaluates its own flag at `last`",
        "    // (_far_coefficient_lines, _far_membership). That inversion is this",
        "    // family's whole delta from folded_fused_magnetic_pair.",
    ])
    body.extend(_constitutive_coefficient_lines(contract))
    body.append(_constitutive_flag_line(contract))
    body.append("")

    # The three first stages and the three guarded `fu` stores, in the certified
    # body's own order, every thread: `step_D` updates `fu` on the recurrence's cells
    # whatever the fills later do to D, and the fills touch D only
    # (stepping._write_mirror_ghost writes `field`, never `fu_field`). The `p`
    # statement carries LIFT_EDITS' first edit, so a destination thread never loads
    # the `f` its source thread writes.
    for target in range(3):
        previous, recurrence, _ = statements["recurrence"][target]
        body.extend([previous, recurrence])
    body.append("")
    body.extend(statements["aux_stores"])

    for target in range(3):
        near = near_fill_axes(codes, target)
        far = far_fill_axes(codes, target)
        if len(near) > 2:
            raise AssertionError(
                f"component {target} is a near-fill destination on {near}; for the D "
                f"family that set is at most two axes (the two that are NOT the "
                f"component's own, fields.IYEE_SHIFTS) and this carry images one "
                f"ghost per nonempty subset")
        if len(far) > 1 or set(far) & set(near):
            raise AssertionError(
                f"component {target} is a far-fill destination on {far} against near "
                f"axes {near}; for the D family the two sets are COMPLEMENTARY (iyee "
                f"is 1 on the component's own axis and 0 on the other two, "
                f"fields.IYEE_SHIFTS), so far holds at most ONE axis and never a "
                f"near one")
        _, _, stepped = statements["recurrence"][target]
        body.append("")
        owned = ([f"({_COORDINATE[axis]} == 0)" for axis in near]
                 + [_LAST_FLAG[axis] for axis in far])
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
                "    // the array path's fill overwrites the displacement it would "
                "store, and",
                "    // forming v here would read a word another thread writes.",
                f"    if (!({' || '.join(owned)})) {{",
            ])
            indent = "        "
        else:
            body.append(f"    // component {target}: no folded axis images a cell "
                        f"of this component")
            indent = "    "
        body.append(indent + stepped.strip())
        # `zero_metal_D` on the cell this thread owns, then the store, then the
        # constitutive read of the SAME register — that is the seam this family
        # closes, exactly as the first fused pair closes it.
        cleared = zero_metal_lines(target, zero_metal, f"v{target}", indent)
        body.extend(cleared or [f"{indent}// no walled axis clears this target"])
        body.append(indent + statements["stores"][target])
        body.extend(_constitutive_block(target, "ii", f"v{target}", indent, "o"))
        if owned:
            body.append("")
            body.extend(_carry_blocks(target, near, far, phases, zero_metal,
                                      indent))
            body.append("    }")

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_MEMBER__": _chunk_bounds.params_member(),
        "__RECORD_ALIAS__": _chunk_bounds.params_alias(),
        "__BODY__": "\n".join(body),
    })


def compile_folded_fused_pair(codes: Sequence[int], phases: Sequence[int],
                              zero_metal: Sequence[bool],
                              contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, parities, walls, mode)."""
    return compile_source(folded_fused_pair_source(
        codes, phases, zero_metal, contract)).folded_fused_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 38-binding signature this platform REFUSES — scalars bound one each.

    Not shipped and not launchable: it exists so the gate can compile it and
    require the failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from
    an argument into a measurement. The body touches every buffer, because a body
    the compiler could drop would let dead-code elimination decide the answer.

    THE THREE REFLECT ROWS ARE IN THE COUNT since the far carry of 2026-08-21. They
    are scalars like the other five, and a refuted signature that omitted them would
    be refuting a shape this module no longer has.
    """
    writes = ["f0", "f1", "f2", "u0", "u1", "u2", "e0", "e1", "e2", "w0", "w1", "w2"]
    reads = ["g0", "g1", "g2", "ie0", "ie1", "ie2",
             "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
             "kp0", "km0", "kp1", "km1", "kp2", "km2"]
    lines: List[str] = []
    slot = 0
    for name in writes:
        lines.append(f"    device float*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in reads:
        lines.append(f"    device const float* {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("nx", "ny", "nz", "n_elem"):
        lines.append(f"    constant uint&      {name:<8}[[buffer({slot})]],")
        slot += 1
    lines.append(f"    constant float&     dtdx    [[buffer({slot})]],")
    slot += 1
    for name in ("rx", "ry", "rz"):
        lines.append(f"    constant int&       {name:<8}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    touch = " + ".join(f"{name}[0]" for name in reads)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        templates.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalars(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {touch};",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz)"
        " + float(rx + ry + rz);",
        *(f"    {name}[idx] = {name}[idx] * dtdx;" for name in writes),
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _mirror_phase_reasons(grid: Any, codes: Optional[Sequence[int]]) -> List[str]:
    """Every folded axis must declare a readable +1/-1 parity.

    ``stepping._symmetry_phase`` RAISES on a ``None`` phase (stepping.py:2465-2469)
    rather than folding with the even-mirror default, and this kernel bakes the
    parity into the source. A grid that cannot answer is refused here, before a
    compile, for the same reason: an even fold run for a run asked to be odd is
    wrong by twice the field wherever the parity mattered.
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
    """What a folded PERIODIC axis must satisfy for the D far carry to be legal.

    THE CLAUSE THIS FAMILY USED TO REFUSE BY NAME. ``fill_folded_far_ghosts_D`` runs
    inside this seam (driver.py:3302) on a folded PERIODIC axis and images the top
    stored slot from ``stepping._far_reflect_rows``' row; the kernel now carries it,
    and these are the conditions the carry's OWNERSHIP MOVE rests on.

    ``_stored_past_owned`` must AGREE with the code, both ways. The code decides
    whether the kernel emits a far carry at all AND whether ``folded_top_plane_mask``
    masks the top plane, so a disagreement is a plane the kernel steps and images
    inconsistently with the array path. ``folded_axis_kinds`` already derives the
    code from that predicate; asked again here because reading coverage off another
    module's derivation is what this package's rule forbids.

    TWO OF THE MAGNETIC TWIN'S CLAUSES ARE DELIBERATELY ABSENT, and their absence is
    derived rather than an oversight. :func:`.folded_fused_magnetic_pair._far_carry_reasons`
    refuses a reflect row of 0 and an extent whose top plane is the near fill's read
    plane. Both are B-geometry hazards: there the near and far axes are the same axis
    or share one, so stored 0 of the far axis IS a near destination. On the D side
    the near destination sits at stored 0 of an axis that is never the far axis
    (``iyee`` is 1 on the component's own axis and 0 on the other two), so stored 0
    of the far axis is an OWNED, stepped cell and the two fills' planes are always
    orthogonal. Copying either clause would be a refusal with no hazard behind it.
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
                f"axis {axis}: stepping._stored_past_owned raised {error!r}; the far "
                f"carry and the curl's top-plane mask are both keyed on it")
            continue
        if periodic != past_owned:
            reasons.append(
                f"axis {axis} code {int(code)} and stepping._stored_past_owned "
                f"{past_owned} disagree; the kernel would mask the top plane and "
                f"image the far ghost on different axes than the array path")
        if not periodic:
            continue
        row = rows[axis] if rows is not None else None
        if row is None:
            reasons.append(
                f"axis {axis} is a folded PERIODIC axis with no reflect row from "
                f"stepping._far_reflect_rows; the far carry has nothing to image")
            continue
        if len(shape) != 3:
            reasons.append("grid.shape is not a per-axis triple; the far carry's "
                           "destination index cannot be built")
            continue
        row, extent = int(row), int(shape[axis])
        if not (0 <= row < extent - 1):
            reasons.append(
                f"axis {axis} reflect row {row} is outside [0, {extent - 1}); the far "
                f"carry would image the plane it writes, or read outside the "
                f"allocation from a thread that owns neither cell")
    return reasons


def metal_folded_fused_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                     residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_D`` -> both fills -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it. That is the construction :func:`.fused_dispersive_pair.metal_fused_dispersive_pair_coverage`
    uses and the one :mod:`.complex_dispersive_spine` uses to conjoin a curl body
    with its constitutive companion.
    """
    reasons: List[str] = []

    curl = folded_composition_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"folded curl half: {reason}" for reason in curl.reasons)
    electric = folded_constitutive_coverage(fields, pml, "E", residency)
    if not electric.covered:
        reasons.extend(f"folded constitutive half: {reason}"
                       for reason in electric.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. An electric source is injected BETWEEN the two halves
    # (driver.py:3292-3299), so a fused pair would consume a pre-injection D.
    # IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. A MAGNETIC source is
    # injected in the B/H half and does not disqualify the pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    # THE FAR FILL is CARRIED, not refused; these are the conditions the carry's
    # OWNERSHIP MOVE rests on.
    reasons.extend(_far_carry_reasons(grid, codes))

    reasons.extend(_mirror_phase_reasons(grid, codes))

    # THE WALL CLEAR IS CARRIED INLINE, so the grid must be able to answer which
    # axes are walled. A grid that cannot would silently be treated as unwalled,
    # which is a plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # THE NEAR FILL IMAGES STORED CELL 2, and the source thread must exist. The
    # folded curl predicate already refuses a folded axis storing three cells or
    # fewer; restated because THIS family reads that cell from a different thread
    # and a missing source plane would be an out-of-range write, not a soft error.
    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; the "
                    f"near fill images stored cell {NEAR_SOURCE_INDEX} and this "
                    f"kernel images it from that cell's own thread")

    # A polarization would make the constitutive source (D - sum P) rather than D.
    # `folded_constitutive_coverage` already refuses it on the E side; restated
    # because this family's kernel bakes the plain product and a reader should not
    # have to chase the other predicate to learn that.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedFusedPairPlan(KernelPlan):
    """ONE dispatch that performs four driver passes, D never leaving a register.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no
    special case for it.

    ``runs`` is an alias of ``launches`` rather than a second counter, because with
    one dispatch per run the two ARE the same number and a second counter would be
    a second thing to get out of step.
    """

    __slots__ = ("residency", "volumes", "codes", "phases", "zero_metal", "shape",
                 "dtdx", "params", "reflect", "carried_axes", "carried_far_axes",
                 "chunk_bounds")

    family = "folded fused D-curl/mirror-fill/far-fill/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phases", "zero_metal", "carried_axes",
                   "carried_far_axes")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phases: Sequence[int],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 params: Any, reflect: Sequence[Optional[int]],
                 pointers: Sequence[Any],
                 functions: Mapping[str, Any], *, chunk_bounds: Any) -> None:
        # The chunk record's host words, kept for inspection: the device copy
        # rides inside ``params``.
        self.chunk_bounds = chunk_bounds
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phases = tuple(int(value) for value in phases)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        self.params = params
        self.reflect = tuple(None if value is None else int(value)
                             for value in reflect)
        self.carried_axes = tuple(
            near_fill_axes(self.codes, target) for target in range(3))
        self.carried_far_axes = tuple(
            far_fill_axes(self.codes, target) for target in range(3))
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def params_record_dtype() -> Any:
    """The packed ``Params`` record: the eight scalars, then the chunk record.

    ONE HOME FOR THE LAYOUT. Every member is four bytes and Metal aligns ``uint``,
    ``int``, ``float`` and the all-4-byte ``ChunkBounds`` struct to four, so the
    natural record IS the struct: 32 bytes of scalars, then the record's 100 at
    offset 32 (:func:`.chunk_bounds.record_dtype_fields`).
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
                     ("dtdx", "<f4"), ("rx", "<i4"), ("ry", "<i4"), ("rz", "<i4")]
                    + _chunk_bounds.record_dtype_fields("cb"))


def _params_tensor(shape: Sequence[int], dtdx: float,
                   reflect: Sequence[Optional[int]], record: Any,
                   device: str) -> Any:
    """The eight scalars and the chunk record as one 132-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, deliberately and for a stated
    reason: :meth:`.Residency.mirror` binds float32 and complex64 volumes and
    refuses anything else BY NAME, because a wider element silently reinterprets.
    This record is neither — it is a packed struct of four uints, a float and three
    ints that nothing on the host reads back and nothing on the device writes — so
    it is built here, once, at plan time, and held by the plan. Nothing on the
    launch path allocates.

    ``numpy.float32(dtdx)`` is the same value the array path uses: NumPy's weak
    promotion casts the Python float to the array's float32 when it multiplies a
    float32 volume by it, and ``constant float&`` would do the same.

    ``-1`` stands for an axis with no reflect row (unfolded, or folded METALLIC,
    where the stored array stops at ``big_corner`` and no far ghost exists). It is a
    value the kernel never reads: the guard that would read it is emitted only for
    an axis :func:`far_fill_axes` returned, and that requires ``CODE_MIRROR_PERIODIC``.
    A sentinel rather than 0, so a source that read it anyway would index outside the
    volume and be caught, not silently image row 0.

    ``record`` is the chunk record's checked words (:func:`.chunk_bounds.plan_words`),
    packed as the ``cb`` member both lifted bounds blocks read.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    params = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    rows = tuple(-1 if value is None else int(value) for value in reflect)
    if len(rows) != 3:
        raise ValueError(f"reflect must be a per-axis triple, got {reflect!r}")
    params["nx"], params["ny"], params["nz"] = nx, ny, nz
    params["n_elem"] = nx * ny * nz
    params["dtdx"] = np.float32(dtdx)
    params["rx"], params["ry"], params["rz"] = rows
    params["cb"] = _chunk_bounds.pack(record)
    words = np.frombuffer(params.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_folded_fused_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        *, chunk_bounds: Any,
        ) -> Optional[MetalFoldedFusedPairPlan]:
    """Build the fused folded D/E plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.symmetry.plan_folded_pml_curl`
    gives: a configuration this kernel does not carry must fall back to the array
    path, never raise into a caller that would otherwise have stepped correctly.

    ``chunk_bounds`` is the record of the division the run steps with — the
    composer's ``StepContext.chunk_bounds`` — and has no default, as
    :func:`.launch.plan_pml_curl` takes it. An admitted configuration given no
    record, or one that disagrees with the grid, is refused BY NAME
    (:func:`.chunk_bounds.plan_words`).

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy
    of the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING — every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not metal_folded_fused_pair_coverage(fields, pml, sources, residency).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    walls = zero_metal_axes(grid)
    record = _chunk_bounds.plan_words(chunk_bounds, grid.shape,
                                      "MetalFoldedFusedPairPlan")
    # The runtime reflect rows the far carry images from — stepping's OWN answer,
    # never a local `n - 2`: the formula is `n_full - stored + 2`, which is
    # `stored - 2` at an even full count and `stored - 3` at an odd one.
    reflect = _far_reflect_rows(grid)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    curl_spec = SUB_STEPS["step_D"]
    electric_spec = CONSTITUTIVE_SIDES["E"]
    pointers = (
        [bind(name, getattr(fields, name)) for name in curl_spec["targets"]]
        + [bind("fu_" + name, getattr(fields, "fu_" + name))
           for name in curl_spec["targets"]]
        + [bind(name, getattr(fields, name)) for name in curl_spec["sources"]]
        + [bind(name, getattr(fields, name)) for name in electric_spec["targets"]]
        + [bind(name, getattr(fields, name)) for name in electric_spec["aux"]]
        + [bind("inv_eps_" + name, fields.inverse_epsilon_for(name), constant=True)
           for name in electric_spec["targets"]]
        # The curl takes the INTEGER lattice on step_D and the constitutive the
        # HALF-INTEGER one on E (stepping.py:948 vs :1015). The kernel takes both and
        # never asks which is which, so a swap here is a silent half-cell error in
        # the absorber profile; the gate carries a mutation for exactly it.
        + [bind(f"pml:{stem}_{axis}{curl_spec['suffix']}",
                getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}"), constant=True)
           for axis in "xyz" for stem in ("kms", "sinv")]
        + [bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"),
                constant=True)
           for axis in "xyz" for stem in ("kps", "kms")])

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_fused_pair(codes, phases, walls, mode)

    return MetalFoldedFusedPairPlan(
        residency, volumes, codes, phases, walls, grid.shape, grid.dt / grid.dx,
        _params_tensor(grid.shape, grid.dt / grid.dx, reflect, record,
                       residency.device),
        reflect, pointers, selected, chunk_bounds=record)


# ---------------------------------------------------------------------------
# Registration — wired=False (no arm selection), routed by the absorb table
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"folded fused pair cannot fill {slot}",))
    return metal_folded_fused_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalFoldedFusedPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_fused_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, chunk_bounds=context.chunk_bounds)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG IS ONE HALF OF THE COMPOSITION STORY, and until 2026-08-27 it was the
    whole of it. ``plan_step`` assigns at most one ARM per slot and this product
    spans five, so there is no slot it could claim through the arm table.
    Registering unwired keeps it ENUMERABLE for the disjointness sweep
    (``arms.registered``) while ``arms.arms_for`` skips it, so ``_select_slot``
    cannot select it and no existing arm's selection changes — including the six
    the folded family already wins.

    THE OTHER HALF IS THE ABSORB TABLE, which is what routing this family means.
    ``launch._install_fused_pairs`` reaches this row through ``arms.registered``
    precisely BECAUSE it is registered — ``is_weld`` with ``replaces`` covering
    ``update_E`` — and ``launch.FUSED_PAIR_ARMS["folded_fused_pair"]`` declares
    which arm each of the two slots implements. So ``wired=False`` still says what
    it always said (no arm selection), and it no longer implies that nothing
    composes this product.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "folded fused D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded fused D/E pair: ",
                          noun="folded PML D-curl/mirror-fill/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
