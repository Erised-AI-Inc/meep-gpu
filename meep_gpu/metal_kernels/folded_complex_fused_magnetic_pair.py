"""The FOLDED COMPLEX fused Metal pair: a folded Bloch ``step_B`` welded into ``update_H``.

THE INTERSECTION OF THE TWO SHIPPED B/H PAIRS, and it is the one cell on this seam
where neither of them can stand in for the other. :mod:`.folded_fused_magnetic_pair`
refuses complex storage; :mod:`.complex_fused_magnetic_pair` refuses a fold BY NAME
("A FOLDED complex B/H seam is a different product and is NOT built here"). This
module is that product.

===========================================================================
WHAT IT IS WORTH: +2 SEAM-INSTANCES OF 387, RECOMPUTED
===========================================================================

MEASURED, not inferred from the ranked-gap table's reachable ceiling.
``results/fusion_matrix_metal_2026-08-20_foldedcomplex/`` re-runs the NAMED Metal
baseline (``results/fusion_matrix_metal_2026-08-20``, 83/387) with this ONE product
added to its ``PRODUCTS`` table and nothing else changed, and lands on **85/387**:
B->H 81 -> 83, D->E and E->P unmoved.

The cell ``(folded complex, folded complex)`` at B->H is driven by FIVE corpus rows,
and the ladder from five to two is where the honest number lives. **THE LADDER AND
THE 85/387 ARE THE PRE-CARRY MEASUREMENT AND ARE LEFT AS THEY WERE MEASURED**: the
``every_fold_is_mirror_metallic`` clause was RETIRED on 2026-08-21 by carrying
``fill_folded_far_ghosts_B``, so the two rows it cost are no longer refused BY IT — but
a new cell count is a matrix RUN, not an edit to this table, and none has been made
since::

    folded_complex_pml_curl@step_B          5
    folded_complex_constitutive@update_H    5
    no_magnetic_source                      4   <- the driver's clause; the CEILING
    every_fold_is_mirror_metallic           2   <- the far fill; RETIRED 2026-08-21
    fold_stores_the_source_row              2
    fold_and_wall_are_disjoint              2

* ``tests:TestHoleyWvgBands.test_fields_at_kx`` declares a MAGNETIC source, which the
  driver deposits between ``step_B`` and ``update_H`` (driver.py:3283-3284). No launch
  on any backend can straddle it. That is the clause the ranked-gap table's
  ``reach 4 (of 5 rows)`` already accounts for, and it is a driver fact rather than
  this kernel's scope.
* ``special_kz_2_21_2`` (``mirrored=[F,T,F]``, ``metallic=[F,F,F]``) and
  ``triangular_lattice_oblique`` (``mirrored=[T,F,F]``, ``metallic=[F,F,F]``) fold over
  a PERIODIC outer declaration, which puts ``fill_folded_far_ghosts_B``
  (driver.py:3287) inside the seam; it images the LAST stored slot from a RUNTIME
  reflect row. This carry TAKES it as of 2026-08-21 and the clause that refused them
  is retired — the same round retired the twin clause in :mod:`.folded_fused_pair`.
  **The row count above is NOT re-derived here.** These two rows are no longer
  refused BY THIS CLAUSE; whether each is admitted end to end is a matrix question
  (every other clause still applies to them) and belongs to a matrix re-run, not to
  a module docstring.

ROWS SERVED: ``examples:solve-cw.py`` and
``tests:TestArrayMetadata.test_array_metadata``. **Both carry an off-diagonal
``chi1inv`` row**, which is why the gate builds that material on two of its cases —
``update_H`` reads ``B`` and ``mu`` and never an inverse epsilon, and this family
DROPS the certified complex constitutive's three inverse-epsilon volumes to bind 27
pointers rather than 30. If that were wrong, those are the rows on which it would
show.

===========================================================================
THE TWO CENSUSES LOOKED LIKE THEY DISAGREED. THEY DO NOT — ADJUDICATED
===========================================================================

``results/fusion_matrix_triton_2026-08-20/`` puts only TWO of those five rows at this
cell and scatters the other three. That was measured out rather than smoothed over:
``results/folded_complex_seam_row_adjudication_2026-08-20/``.

* **The corpus is not in dispute.** All 23 shared configuration keys agree, on all
  five rows, between the Triton battery (2026-08-16) and the Metal one (2026-08-19).
* **The difference is an ARM LABEL.** Triton registers a second family name,
  ``folded complex off-diagonal``, over the SAME kernel and the SAME plan class for
  rows carrying an off-diagonal ``chi1inv`` row — its own builders say so
  (``triton_kernels/folded_complex.py``: *"The kernel and the plan class are K1's,
  untouched; only the ADMISSION is new"*; the constitutive one returns a
  ``ComplexConstitutivePlan``, the plain arm's class). The split exists because an arm
  table holds at most one row per (family, slot) and two admitting arms FAIL CLOSED.
  Metal ships no off-diagonal MAGNETIC constitutive arm and so admits the plain one.
  Both admissions are right, for the reason the pointer count already states.
* **THE CONSEQUENCE FOR A TRITON TWIN IS NOT COSMETIC, and it is measured**
  (``triton_twin_counterfactual.json``, over all 186 rows): a twin conjoining only the
  PLAIN folded-complex halves admits ``special_kz_2_21_2`` and
  ``triangular_lattice_oblique`` — the two the far-fill clause then refuses — and is
  worth **ZERO**. Admitting through EITHER arm name is worth **TWO**, the same two
  rows this product serves. A Triton twin is NOT BUILT here, and that specification is
  the deliverable in its place.

Three ``special_kz`` rows sit in Triton's ``folded complex`` constitutive bucket with
NO admitted curl arm at all (nonzero beta, refused by name in both directions), where
Metal routes them to ``(folded beta complex, folded beta complex)``. One of the three
declares an in-seam magnetic source, so that gap is worth at most 2 more. It is
FOLDED-BETA-COMPLEX CURL ARM work, not fusion work, and is out of this family's scope.

ONE DISPATCH FOR ALL THREE COMPONENTS: 27 pointers plus one packed
``constant Params&``, 28 of the 31 bindings the platform allows. That is
:mod:`.complex_fused_magnetic_pair`'s signature EXACTLY — the fold adds no POINTER,
because the stored extent is what carries it and the PML coefficient vectors are
already built at that extent — so :data:`PACKED_BINDINGS` and
:data:`SEPARATE_SCALAR_BINDINGS` are IMPORTED from that module rather than re-spelled.
What the fold does add is SIX ``float2`` members INSIDE the packed struct (the NEAR
mirror parity and the FAR one, one of each per axis) and three ``int`` reflect rows,
which costs no binding at all — the record grows from 72 to
:data:`PARAMS_ITEMSIZE` bytes and the signature does not move. The measured fitness
verdict the matrix recorded for this cell — ``pointers 15 + 18 -> 27..27 (ceiling
30)`` — is therefore met exactly, and :func:`plan_metal_folded_complex_fused_magnetic_pair`
asserts the pointer count rather than trusting it.

===========================================================================
EVERYTHING HERE IS TRANSCRIBED — the two halves are LIFTED, not retyped
===========================================================================

* the folded Bloch B curl — :func:`.folded_complex.folded_bloch_curl_source` with
  ``backward=False``, whose head (guard, decode, ghost gather, Bloch phase block,
  curl grouping, cell-0 ownership mask, top-plane slot) is spliced verbatim by
  :func:`certified_curl_head` and whose split-field recurrence statements are pulled
  line by line by :func:`certified_curl_statements`. The cut is taken at
  :data:`_RECURRENCE_MARK`, exactly where :mod:`.folded_fused_magnetic_pair` takes it
  on the real fold;
* the wall clear — ``stepping._zero_metal`` (stepping.py:2206-2247) restricted to
  ``B_COMPONENTS`` (stepping.py:183), through
  :data:`.complex_fused_magnetic_pair._ZERO_METAL_ROWS`, which is IMPORTED: that
  table is already the B DIAGONAL and is already pinned. The zero written is
  :data:`.templates.COMPLEX_ZERO`, both planes, because the array path assigns a
  complex zero (stepping.py:1943, :1949);
* the near mirror fill — ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1426-1452)
  and ``stepping._write_mirror_ghost`` (:1451), carried as the CERTIFIED FOLDED
  COMPLEX FILL'S OWN ARITHMETIC: ``c_mul(coefficient, plane)`` with the coefficient
  on the LEFT and its words HOST-ROUNDED through ``numpy.complex64`` by
  :func:`.folded_complex.mirror_parity_coefficients`. :func:`near_fill_transcription`
  re-runs :func:`.folded_complex.folded_mirror_fill_complex_source` and requires this
  module's carry line to be that emitter's own line under a stated rename;
* the constitutive half — :func:`.complex_fused_magnetic_pair.certified_constitutive_body`,
  which is ``complex_fields.bloch_constitutive_source("H")``'s own body with the
  targets renamed ``f`` -> ``h``. This module re-emits it PER COMPONENT AT A
  PARAMETERISED INDEX (the owned cell and the imaged ghost cell take the identical
  seven statements at two different flat indices), and
  :func:`constitutive_transcription` COMPARES the re-emission against that lifted
  body statement for statement, on every build.

===========================================================================
WHY THE PARITY IS A RUNTIME WORD HERE AND A COMPILE-TIME SIGN ON THE REAL FOLD
===========================================================================

:mod:`.folded_fused_magnetic_pair` bakes ``phase`` into the SOURCE as ``-v`` or a
plain copy, because :func:`.symmetry._parity_spelling` MEASURED that a runtime float
multiply flushes every subnormal on this backend at both signs. This family does the
opposite, and for the reason :mod:`.folded_complex` records at its own fill: the
complex parity is already inside a ``c_mul`` — an arithmetic expression that flushes
anyway — so making it compile-time would buy nothing and would LOSE the property that
matters, that the coefficient words are host-rounded and passed rather than
synthesised in-kernel. Keeping them passed is also what keeps "``PHASE * word``" and
"``phase == +1`` is a plain copy" refutable mutations rather than unreachable ones.

The consequence for the signature is nil: the parity words ride INSIDE the packed
``Params`` struct beside the three Bloch phases — SIX of them since the 2026-08-21 far
carry, the three NEAR and the three FAR, plus three int reflect rows — so this kernel
still binds the same 28 things the unfolded complex pair binds.

===========================================================================
THE SEAM, AND THE FOUR PASSES INSIDE IT — from the driver, not assumed
===========================================================================

``driver.step`` runs four passes between ``step_B`` and ``update_H``
(driver.py:3281-3289)::

    step_B -> MAGNETIC SOURCES -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — REFUSED BY NAME**, and on this cell that clause is
  exactly one row of five. Ignorance is never an empty set: ``Fields`` does not hold
  the source list, so an undeclared ``sources`` is a REFUSAL and not an assumed
  ``()``. An ELECTRIC source is injected in the D/E half and does NOT disqualify this
  pair.
* **``fill_symmetry_bc_B`` — CARRIED INLINE, and it is NOT dead.** It writes
  ``cell 0 = parity (x) cell 2`` on every folded axis for every component whose Yee
  shift there is 0, ``update_H`` then reads exactly those cells, and nothing between
  the fill and the read touches them.
* **``zero_metal_B`` — CARRIED INLINE**, through the imported B diagonal.
  ``_zero_metal`` SKIPS a folded axis (stepping.py:2284-2286:
  ``is_metallic(axis) and not is_mirrored(axis)``), so a walled axis is never a
  folded one and the carry's disjointness holds by construction — asserted in
  :func:`_carry_block` rather than trusted.
* **``fill_folded_far_ghosts_B`` — CARRIED INLINE as of 2026-08-21.** It runs only on
  a folded PERIODIC axis (``stepping._stored_past_owned``:1454-1469) and images the
  LAST stored slot from a RUNTIME reflect row. It was REFUSED BY NAME until this
  round; the refusal ALSO retired the folded complex curl's top-plane block
  (``symmetry.folded_top_plane_mask`` emitted nothing), so a carry that took the fill
  and stopped there would have left that plane unmasked. Both moved together: the
  lifted head's ``last_*`` flags are READ again, the reflect rows ride in the packed
  ``Params`` as ``rx``/``ry``/``rz``, and the FAR parity words ride beside the near
  ones as ``d0``/``d1``/``d2``. Under complex storage the two parities compose as a
  CHAIN of ``c_mul`` calls whose ORDER is the driver's and is transcribed
  (:func:`parity_chain`), never chosen — re-ordering it moves bytes
  (``results/complex_parity_chain_order_2026-08-21/parity_chain_order.json``).

===========================================================================
THE B GEOMETRY IS NOT THE D GEOMETRY — inherited, and it still holds
===========================================================================

``fields.IYEE_SHIFTS`` (fields.py:214-219): ``Bx (0,1,1)``, ``By (1,0,1)``,
``Bz (1,1,0)``. The near fill touches component ``m`` on axis ``a`` exactly when
``iyee[m][a] == 0``, so for the B family that is the component's OWN axis and only
it. Two consequences carry over from :mod:`.folded_fused_magnetic_pair` unchanged,
and both are re-derived here from the same table rather than assumed:

1. **THE COEFFICIENT INDEX MOVES.** ``update_H`` indexes ``kps``/``kms`` on the
   component's own axis (``stepping.H_CONSTITUTIVE_TERMS``:226), which is exactly the
   axis the fill images along — so the fill's SOURCE (stored 2) and DESTINATION
   (stored 0) take DIFFERENT coefficient entries, and the source thread loads the
   destination's pair explicitly at index 0 (:func:`_ghost_coefficient_lines`). On a
   shallow layer the two entries are the same word to the bit and the defect is
   UNOBSERVABLE, which is why the gate predicts observability from a coefficient
   census rather than reporting a catch.
2. **ONE GHOST PER COMPONENT.** A component is a near-fill destination on at most one
   axis, so there is no doubly-unowned corner on this side and no ordering question
   between two carries. :func:`folded_complex_fused_magnetic_pair_source` refuses a
   longer axis set rather than trusting it.

===========================================================================
NOT WIRED
===========================================================================

``register_arms`` registers this product with ``wired=False``, as all six other
Metal fused pairs are. ``arms.arms_for`` skips an unwired arm, so ``plan_step``
cannot select it and no existing arm's selection changes. The four seam-instances
this family reaches are a PREDICATE VERDICT; the number actually stepped by a fused
kernel in production remains zero.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from . import complex_fields, shaders, templates
from .complex_fused_magnetic_pair import (
    PACKED_BINDINGS,
    SEPARATE_SCALAR_BINDINGS,
    _ZERO_METAL_ROWS,
    certified_constitutive_body,
    refuted_separate_scalar_source,
)
from .device import Residency, compile_source
from .folded_complex import (
    expansion_from_probe,
    folded_bloch_curl_source,
    folded_complex_composition_curl_coverage,
    folded_complex_constitutive_coverage,
    load_expansion_probe,
    mirror_parity_coefficients,
)
from .plans import KernelPlan
from .symmetry import MIRROR_CODES
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
#:
#: TRUE SINCE 2026-08-30, in the same edit as the ``launch.FUSED_PAIR_ARMS`` row that
#: lets the seam loop bracket this family (the ``('folded complex', 'folded complex')``
#: entry). Until that row existed this family had NO route into
#: ``_install_fused_pair`` at all -- the seam loop refused it by name on every
#: configuration, "has no absorb declaration" -- so the flag could not truthfully be
#: anything but False, and the fusion board recorded the cell as
#: ``product_exists_but_refuses`` on its ``in_seam_magnetic_deposit_clears`` clause.
#: Both halves of the claim now hold and neither is new here:
#:
#: * THE WIRING. ``launch._install_fused_pair`` puts a ``LeadingRepairPlan`` in
#:   ``step_B`` and a ``TrailingRepairPlan`` in ``update_H`` whenever the seam carries
#:   a deposit, and it reaches this family through the absorb row above.
#: * THE FOLDED SEAM. What held every folded family at ``False`` was that a folded
#:   seam runs ``fill_symmetry_bc_B``/``fill_folded_far_ghosts_B`` AFTER the injection
#:   and a POINT repair never visits the MIRROR IMAGE of a deposit index. That is
#:   retired by ``deposit_repair.repair_cells`` (deposit_repair.py:217-258), which
#:   extends the saved and restored set to the CLOSURE of the cells those two fills
#:   image each deposit point into -- the same machinery that flipped
#:   ``folded_fused_magnetic_pair`` and ``folded_fused_pair`` on 2026-08-28. It is
#:   storage-agnostic: it indexes and accumulates element-wise over one array per
#:   component, which under complex storage is one ``complex64`` array (fields.py:573).
#:   ``repairable`` still refuses BY NAME the two folds whose fill map it cannot read
#:   -- the cylindrical r = 0 axis and a folded axis storing no more than
#:   ``MIRROR_SOURCE_INDEX`` cells -- and this family inherits both refusals rather
#:   than restating them.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_complex_fused_magnetic_pair"

#: The sub-step slot this arm holds a row on. It spans four; a refusal is NAMED on
#: the slot the fusion starts at rather than being invisible to the table.
SLOT = "step_B"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
#: ``BACKWARD`` is the direction ``SUB_STEPS[CURL_SUB_STEP]`` declares, restated as a
#: bool so the one legal binding is visible without importing :mod:`launch`; a test
#: pins the two equal.
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"
BACKWARD = False

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3281-3289) and in :data:`.coverage.RESIDENCY_ORDER`'s spelling. Read by
#: the whole-step gate's ``covered_passes``; declared, never inferred from the slot
#: name.
REPLACES: Tuple[str, ...] = ("step_B", "fill_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: The stored index the near fill images, re-exported so a reader of this module does
#: not have to chase ``triton_kernels.symmetry`` for the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX

#: The packed struct's size in bytes. NINE ``float2`` members (three Bloch phases,
#: three NEAR mirror parities, three FAR ones), then four uints, one float and three
#: ints: 72 + 20 + 12 = 104, which is already a multiple of the strictest member's
#: 8-byte alignment so Metal adds no tail padding. Stated rather than derived because
#: a record that agreed with Metal only by accident is the failure mode this constant
#: exists for — leg ``binding_ceiling`` LAUNCHES the struct and reads every field back
#: rather than trusting the arithmetic.
PARAMS_ITEMSIZE = 104

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY",
    "NEAR_SOURCE_INDEX", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalFoldedComplexFusedMagneticPairPlan",
    "carried_destinations", "certified_curl_head", "certified_curl_statements",
    "compile_folded_complex_fused_magnetic_pair", "constitutive_transcription",
    "far_fill_axes", "far_fill_transcription", "far_parity_words",
    "folded_complex_fused_magnetic_pair_source",
    "metal_folded_complex_fused_magnetic_pair_coverage", "near_fill_axes",
    "near_fill_transcription", "params_record_dtype", "parity_chain",
    "parity_words", "plan_metal_folded_complex_fused_magnetic_pair",
    "refuted_separate_scalar_source", "zero_metal_lines",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
#
# TWENTY-EIGHT BINDINGS: 15 float2 volumes + 12 float32 coefficient vectors + 1
# packed Params&. The coefficients stay float32 under complex storage
# (stepping.py:41-50), which is what keeps this at twelve buffers rather than
# twenty-four and the whole product buildable at all.

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE NINE float2 MEMBERS COME FIRST, for the reason `complex_fused_magnetic_pair`
// measured on this toolchain: Metal aligns float2 to 8 bytes, so with the scalars
// first the struct needs internal padding and the natural host record puts every
// phase one word early — which reads as a plausible complex number rather than as
// garbage. Phases and parities first, no internal padding.
//
// `m0`/`m1`/`m2` are the NEAR mirror parity on axes x/y/z and `d0`/`d1`/`d2` the FAR
// one, as complex64 words rounded on the host by
// `folded_complex.mirror_parity_coefficients` and passed. The two are DIFFERENT words
// — `mirror_parity` is `phase * (1 - 2*iyee)` (fields.py:180-182), so a shift-0 near
// destination takes `+phase` and a shift-1 far one `-phase` — and both must be
// present because one kernel now carries both fills. An axis that carries neither
// takes the zero word and no line reads it.
//
// TWELVE SCALARS, ONE BINDING, AND THE BINDING COUNT IS UNCHANGED AT 28. The three
// far parities and the three reflect rows (stepping._far_reflect_rows:1661) ride
// inside this record rather than as six more buffers, so carrying
// fill_folded_far_ghosts_B costs nothing against the platform's 31-binding ceiling
// (device.py:72) — which matters here more than on the real board, because complex
// storage already spent the headroom on float2 volumes (complex_fields.py:38).
struct Params {
    float2 px; float2 py; float2 pz;
    float2 m0; float2 m1; float2 m2;
    float2 d0; float2 d1; float2 d2;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
    int rx; int ry; int rz;
};

kernel void folded_complex_fused_magnetic_pair_step(
    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device float2*       u0      [[buffer(3)]],
    device float2*       u1      [[buffer(4)]],
    device float2*       u2      [[buffer(5)]],
    device const float2* g0      [[buffer(6)]],
    device const float2* g1      [[buffer(7)]],
    device const float2* g2      [[buffer(8)]],
    device float2*       h0      [[buffer(9)]],
    device float2*       h1      [[buffer(10)]],
    device float2*       h2      [[buffer(11)]],
    device float2*       w0      [[buffer(12)]],
    device float2*       w1      [[buffer(13)]],
    device float2*       w2      [[buffer(14)]],
    device const float*  kmx     [[buffer(15)]],
    device const float*  sinvx   [[buffer(16)]],
    device const float*  kmy     [[buffer(17)]],
    device const float*  sinvy   [[buffer(18)]],
    device const float*  kmz     [[buffer(19)]],
    device const float*  sinvz   [[buffer(20)]],
    device const float*  kp0     [[buffer(21)]],
    device const float*  km0     [[buffer(22)]],
    device const float*  kp1     [[buffer(23)]],
    device const float*  km1     [[buffer(24)]],
    device const float*  kp2     [[buffer(25)]],
    device const float*  km2     [[buffer(26)]],
    constant Params&     prm     [[buffer(27)]],
    uint idx [[thread_position_in_grid]])
{
    // THE ELEVEN PACKED ARGUMENTS ARE UNPACKED INTO THE LIFTED BODIES' OWN NAMES,
    // once, before any of the spliced text runs. Everything below this line is then
    // character-for-character what the folded Bloch curl emitter and the certified
    // complex H constitutive emitter produce, plus the wall clear and the mirror
    // carry.
    //
    // The parity registers are named `mp0`/`mp1`/`mp2` (near) and `fp0`/`fp1`/`fp2`
    // (far) and NOT `c`: the lifted curl head already declares `float2 c = g2[ii];`,
    // and the certified fill's own spelling of the coefficient is `c`. The rename is
    // the whole difference between the fill's line and this one, and
    // `near_fill_transcription` / `far_fill_transcription` measure exactly that
    // rather than asserting it. `n0`/`n1`/`n2` are NOT available as parity names:
    // the lifted split-field recurrence owns them.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float2 px = prm.px, py = prm.py, pz = prm.pz;
    float2 mp0 = prm.m0, mp1 = prm.m1, mp2 = prm.m2;
    float2 fp0 = prm.d0, fp1 = prm.d1, fp2 = prm.d2;
    int reflect_x = prm.rx, reflect_y = prm.ry, reflect_z = prm.rz;
    (void)fp0; (void)fp1; (void)fp2;
    (void)reflect_x; (void)reflect_y; (void)reflect_z;

__BODY__
}
"""

#: The marker that separates a certified shader's signature from its body. Every
#: template in this package ends its parameter list with this exact line, so ONE
#: anchor lifts any body and a template that stopped carrying it raises here rather
#: than splicing a truncated kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The comment line the folded Bloch curl body opens its split-field block with. This
#: is the CUT POINT of the lift: everything above it is spliced verbatim, everything
#: below is re-emitted per component around the ownership carve-out. Matched as a
#: PREFIX so the block's trailing dash run is not a second thing to keep in step.
_RECURRENCE_MARK = "    // --- split-field recurrence"

#: The flat-index coordinate name per axis, and the stride expression per axis, in
#: the lifted decode's own spelling. ``nyz`` and ``nzi`` are that block's variables; a
#: second decode here would be a second place to get the layout wrong.
_COORDINATE: Tuple[str, str, str] = ("i", "j", "k")
_STRIDE: Tuple[str, str, str] = ("nyz", "nzi", "1")

#: The kernel register holding the NEAR mirror parity for each axis. Unpacked from
#: :data:`Params` in the template above; see the note there for why not ``c``.
_PARITY_REGISTER: Tuple[str, str, str] = ("mp0", "mp1", "mp2")

#: The kernel register holding the FAR mirror parity for each axis. A SEPARATE word
#: rather than a negation of the near one: ``mirror_parity`` is
#: ``phase * (1 - 2*iyee)`` (fields.py:180-182) and
#: :func:`.folded_complex.mirror_parity_coefficients` rounds BOTH through
#: ``numpy.complex64`` on the host and returns them together, so synthesising one from
#: the other in-kernel would be exactly the "never synthesised in-kernel" rule that
#: function exists to keep.
_FAR_PARITY_REGISTER: Tuple[str, str, str] = ("fp0", "fp1", "fp2")

#: The kernel register holding ``stepping._far_reflect_rows``' answer per axis. A
#: RUNTIME int, not a source specialisation, for the certified separate fill's own
#: reason: the row is per-axis and integer, so unlike the parity nothing about it can
#: move a bit.
_REFLECT: Tuple[str, str, str] = ("reflect_x", "reflect_y", "reflect_z")

#: The TOP STORED INDEX per axis, and the boolean the lifted curl head declares for
#: it. ``_LAST`` is the integer the far destination's flat index is built from;
#: ``_LAST_FLAG`` is the head's own name, which it already declares for the top-plane
#: curl mask and which this family reuses rather than writing a second comparison
#: against the same extent.
_LAST: Tuple[str, str, str] = ("(nxi - 1)", "(nyi - 1)", "(nzi - 1)")
_LAST_FLAG: Tuple[str, str, str] = ("last_x", "last_y", "last_z")

#: The seven statements ``_apply_constitutive_pml`` runs for one component under
#: COMPLEX storage, as they appear in the certified complex H body once the targets
#: are renamed ``f`` -> ``h``. Parameterised on the flat index, the source expression,
#: the coefficient pair and a register prefix — the ghost cell takes the identical
#: seven at a different index with a DIFFERENT coefficient pair.
#:
#: NOT A RETYPING: :func:`constitutive_transcription` renders these with the certified
#: names and compares them statement for statement against
#: :func:`.complex_fused_magnetic_pair.certified_constitutive_body`'s own text, and
#: :func:`folded_complex_fused_magnetic_pair_source` runs that comparison on every
#: build. The two products' accumulations are the SAME SHAPE and DIFFERENT
#: ARITHMETIC: the real fold's ``acc + kp * src`` becomes the zero-imaginary complex
#: product with the COEFFICIENT ON THE LEFT (stepping.py:2133-2134, :2140-2142), which
#: is a different rounding and is why this table is spelled here rather than imported
#: from the real twin.
_CONSTITUTIVE_STATEMENTS: Tuple[str, ...] = (
    "float2 {prefix}prev = w{target}[{index}];",
    "float2 {prefix}src = {source};",
    "w{target}[{index}] = {prefix}src;",
    "float2 {prefix}acc = h{target}[{index}];",
    "{prefix}acc = {prefix}acc + c_mul_coefficient_left({kp}, {prefix}src);",
    "{prefix}acc = {prefix}acc - c_mul_coefficient_left({km}, {prefix}prev);",
    "h{target}[{index}] = {prefix}acc;",
)

#: How the certified complex H body spells each of the seven for component ``t``: the
#: register names ``prev{t}`` / ``src{t}`` / ``a{t}``, the flat index ``ii``, the
#: source ``g{t}[ii]`` and the coefficient pair ``kp_{t}`` / ``km_{t}``. Rendering
#: :data:`_CONSTITUTIVE_STATEMENTS` with these must reproduce that body exactly.
_CERTIFIED_PREFIX = "@"
_CERTIFIED_NAMES: Dict[str, str] = {
    "@prev": "prev{target}", "@src": "src{target}", "@acc": "a{target}",
}

#: The certified folded complex fill's own near-pass line, as
#: :func:`.folded_complex.folded_mirror_fill_complex_source` emits it for slot ``t``.
#: :func:`near_fill_transcription` builds that source and requires this template,
#: rendered with the fill's names, to be one of its lines.
_CERTIFIED_FILL_LINE = (
    "    f{slot}[base] = c_mul(c, f{slot}[base + {index} * stride]);")

#: The same emitter's FAR-pass line. Two templates rather than one parameterised on
#: the index expression, because the far pass moves the DESTINATION too — it writes
#: ``last`` and reads ``reflect_row``, where the near pass writes ``base`` and reads a
#: fixed stored 2 — and a single template hiding that would be the retyping this
#: family exists to avoid.
_CERTIFIED_FAR_FILL_LINE = (
    "    f{slot}[base + last * stride] = "
    "c_mul(c, f{slot}[base + reflect_row * stride]);")


def near_fill_axes(codes: Sequence[int], target: int) -> Tuple[int, ...]:
    """Which folded axes ``fill_symmetry_bc_B`` writes for ONE B component.

    ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1441-1447) fills axis ``a``
    for component ``m`` exactly when ``a`` is mirrored and ``iyee[m][a] == 0``. Read
    off ``TARGET_IYEE`` rather than spelled: a literal here would be another
    transcription of the Yee table, and this predicate decides which threads own which
    cells.

    For the B family the answer is at most ONE axis — the component's own — and
    :func:`folded_complex_fused_magnetic_pair_source` refuses a longer one rather than
    trusting that. BOTH mirror codes are matched: the near fill runs on any folded
    axis (``stepping._fill_symmetry_ghost_cells`` gates on ``_mirror_phases`` alone),
    which is exactly what separates it from :func:`far_fill_axes`.
    """
    names = SUB_STEPS[CURL_SUB_STEP]["targets"]
    return tuple(axis for axis, code in enumerate(codes)
                 if int(code) in MIRROR_CODES
                 and TARGET_IYEE[names[target]][axis] == 0)


def far_fill_axes(codes: Sequence[int], target: int) -> Tuple[int, ...]:
    """Which axes ``fill_folded_far_ghosts_B`` images for ONE B component.

    ``stepping._fill_folded_far_ghosts`` (stepping.py:1518-1534) writes axis ``a`` for
    component ``m`` exactly when ``_stored_past_owned(grid, a)`` — a folded PERIODIC
    axis, at either full-count parity — and ``iyee[m][a] == 1``. The two conditions
    are read here off ``CODE_MIRROR_PERIODIC`` (which ``folded_axis_kinds`` derives
    from ``_stored_past_owned`` itself, and cross-checks against ``grid.is_metallic``)
    and off ``TARGET_IYEE``, so neither the fold split nor the Yee table is
    transcribed a second time.

    THE EXACT COMPLEMENT OF :func:`near_fill_axes` ON THIS FAMILY. A B component's Yee
    shift is 0 on its OWN axis and 1 on the other two, so the near fill touches
    component ``m`` on axis ``m`` alone while the far fill touches it on the two axes
    that are NOT its own — up to TWO destination planes per component against the near
    fill's one, and a corner where both fire carrying a CHAIN of both parities. On the
    real board that chain collapses to a compile-time sign; here it does not, and
    :func:`parity_chain` is where the measurement that says so is spent.
    """
    names = SUB_STEPS[CURL_SUB_STEP]["targets"]
    return tuple(axis for axis, code in enumerate(codes)
                 if int(code) == CODE_MIRROR_PERIODIC
                 and TARGET_IYEE[names[target]][axis] == 1)


def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], bool], ...]:
    """Every ghost cell ONE source thread owns, as ``(far axes at top, near?)``.

    THE OWNERSHIP RULE. The driver runs the near fill, the wall clear and the far fill
    in that order (driver.py:3285-3287) and the far fill is applied axis by axis, so a
    cell the fills leave at the top of SEVERAL folded periodic axes is written more
    than once. Back-substituting every pass gives one source cell per ghost — the cell
    at stored ``NEAR_SOURCE_INDEX`` on the near axis and at the reflect row on each far
    axis — so ONE thread computes the displacement every one of those ghosts carries
    and writes them all. Every destination word is written by exactly one thread and no
    thread reads a word another writes, which is what makes the carry legal inside a
    single Metal dispatch with no device-wide barrier.

    WHAT THIS FAMILY DOES NOT INHERIT FROM THE REAL BOARD IS THE WEIGHT.
    :func:`.folded_fused_magnetic_pair.carried_destinations` composes the parities into
    ONE compile-time sign and states that the result is "independent of the order the
    axes are applied in". Under COMPLEX storage that is FALSE and it is measured false:
    ``results/complex_parity_chain_order_2026-08-21/parity_chain_order.json`` re-orders
    a two-parity chain over an exhaustive signed-zero/subnormal table and moves 8 of
    128 uint32 words, and folding the chain into a single word moves up to 23 of 128 —
    at every sign combination, all-``+1`` included. So this function returns the CELLS
    only, and :func:`parity_chain` returns the ORDERED applications separately.

    Returned smallest-subset-first for a stable emission order; the order the blocks
    are EMITTED in is unobservable (the cells are distinct), and a stable one keeps a
    source diff readable.
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


def parity_chain(near: Sequence[int], subset: Sequence[int], carries_near: bool
                 ) -> Tuple[Tuple[int, str], ...]:
    """The ORDERED ``(axis, pass)`` applications one ghost's value carries.

    THE ORDER IS THE DRIVER'S AND IS TRANSCRIBED, NEVER CHOSEN. ``driver.step`` runs
    ``fill_symmetry_bc_B`` (:3285) to completion before ``fill_folded_far_ghosts_B``
    (:3287), and the far pass loops ``for axis in axes`` with ``axes`` in ASCENDING
    axis order (``stepping._fill_folded_far_ghosts``:1518, :1528). Each pass reads the
    plane the previous one wrote, so back-substituting a corner gives

        c_far[a2] (x) ( c_far[a1] (x) ( c_near[n] (x) v ) ),    a1 < a2

    — the near application INNERMOST because its whole pass precedes the far one, then
    the far axes ascending. This function returns that list, and the emitter applies
    one ``c_mul`` per entry in exactly this sequence.

    A single-entry chain is the common case and the only one the two corpus rows this
    carry admits ever reach: ``special_kz_2_21_2`` and ``triangular_lattice_oblique``
    each fold ONE axis, so each component takes either one near or one far parity and
    the ordering question is null on both. The chain is reachable configuration space
    all the same — two folded periodic axes give a two-entry chain and three give a
    three-entry one — and the gate carries those cases rather than leaving the
    ordering untested because the corpus does not press on it.
    """
    chain: List[Tuple[int, str]] = []
    if carries_near:
        if not near:
            raise AssertionError(
                "a destination that carries the near fill was enumerated for a "
                "component with no near axis; carried_destinations and "
                "near_fill_axes have drifted")
        chain.append((int(near[0]), "near"))
    previous = -1
    for axis in subset:
        axis = int(axis)
        if axis <= previous:
            raise AssertionError(
                f"the far axis subset {tuple(subset)!r} is not strictly ascending; "
                f"stepping._fill_folded_far_ghosts applies its axes in ascending "
                f"order (:1518) and this chain transcribes that order")
        previous = axis
        chain.append((axis, "far"))
    return tuple(chain)


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    The lift's whole safety property. A missing anchor is a certified emitter that has
    changed under this family, and splicing around it would produce a kernel that
    compiles and is quietly not the certified arithmetic; TWO matches would mean the
    anchor no longer identifies a single statement.
    """
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this family LIFTS that line rather than retyping "
            f"it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the lift "
            f"of {what} would take an arbitrary one")
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


def certified_curl_head(codes: Sequence[int], phased: Sequence[int], expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The folded Bloch ``step_B`` curl body DOWN TO the split-field recurrence, verbatim.

    Not a transcription: this is :func:`.folded_complex.folded_bloch_curl_source`'s own
    output with the ``#include``/helpers/signature preamble removed and the cut taken
    at :data:`_RECURRENCE_MARK`. It carries the guard, the index decode, the ghost
    gather, the per-axis Bloch phase block (which SKIPS an unphased axis rather than
    multiplying by ``1+0j`` — the bit-identity of ``k = 0``), the curl grouping, the
    cell-0 ownership mask and the top-plane slot, which
    :func:`.symmetry.folded_top_plane_mask` leaves as a comment here because this
    family refuses ``MIRROR_PERIODIC``.

    ``backward`` is :data:`BACKWARD` and never a parameter: this pair is the B seam,
    and ``step_D``'s negated strides and conjugated phase are a different product with
    a different fill geometry entirely.
    """
    body = _body_of(
        folded_bloch_curl_source(codes, BACKWARD, phased, expansion, contract),
        "folded complex curl")
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(_RECURRENCE_MARK):
            head = "".join(lines[:index])
            if not head.strip():
                raise AssertionError("the lifted folded complex curl head is empty")
            return head
    raise AssertionError(
        f"the certified folded complex curl body no longer opens its split-field "
        f"block with {_RECURRENCE_MARK!r}; this family cuts the lift there")


def certified_curl_statements(codes: Sequence[int], phased: Sequence[int],
                              expansion: str,
                              contract: str = shaders.CONTRACT_OFF,
                              ) -> Dict[str, Any]:
    """The split-field recurrence's own lines, pulled out of the certified body.

    Every arithmetic string this family emits below the cut comes from HERE rather
    than from a keyboard: the coefficient loads, the three ``p``/``n``/``v``
    statements and the two store lines are the certified folded Bloch curl's, matched
    by an anchor that must identify exactly one line each. The ``float2`` prefixes are
    what separate these anchors from the real fold's.

    ``stores`` is the ``f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;`` triple SPLIT into its
    three statements, because this family emits them one per component inside that
    component's ownership block rather than as one line.
    """
    body = _body_of(
        folded_bloch_curl_source(codes, BACKWARD, phased, expansion, contract),
        "folded complex curl")
    coefficients = [_line_starting(body, f"    float km_{axis} = ",
                                   f"the {axis} split-field coefficient load")
                    for axis in "xyz"]
    recurrence = []
    for target in range(3):
        recurrence.append((
            _line_starting(body, f"    float2 p{target} = ",
                           f"target {target}'s previous split field"),
            _line_starting(body, f"    float2 n{target} = ",
                           f"target {target}'s split-field recurrence"),
            _line_starting(body, f"    float2 v{target} = ",
                           f"target {target}'s stepped displacement"),
        ))
    aux_store = _line_starting(body, "    u0[ii] = ", "the three fu stores")
    flux_store = _line_starting(body, "    f0[ii] = ", "the three flux stores")
    stores = tuple(f"{part.strip()};" for part in flux_store.strip().split(";")
                   if part.strip())
    if len(stores) != 3:
        raise AssertionError(
            f"the certified folded complex curl no longer stores three targets on one "
            f"line: {flux_store!r}")
    return {"coefficients": tuple(coefficients), "recurrence": tuple(recurrence),
            "aux_store": aux_store, "stores": stores}


def _constitutive_coefficient_lines(expansion: str,
                                    contract: str = shaders.CONTRACT_OFF
                                    ) -> Tuple[str, ...]:
    """``kp_t``/``km_t`` on the component's OWN axis, lifted from the certified body.

    ``complex_fields``' constitutive template indexes target 0 on ``i``, 1 on ``j`` and
    2 on ``k`` (``stepping.H_CONSTITUTIVE_TERMS``:226) — MEEP's ``dsigw``, NOT the
    dsig/dsigu cycle the curl recurrence uses. On THIS family the imaged ghost cell
    does NOT reuse these, which is the delta :func:`_ghost_coefficient_lines` carries.
    """
    body = certified_constitutive_body(expansion, contract)
    return tuple(_line_starting(body, f"    float kp_{target} = ",
                                f"target {target}'s constitutive coefficient pair")
                 for target in range(3))


def _render_constitutive(target: int, index: str, source: str, kp: str, km: str,
                         prefix: str) -> Tuple[str, ...]:
    """The seven constitutive statements for one component at one cell, unindented."""
    return tuple(line.format(target=target, index=index, source=source,
                             kp=kp, km=km, prefix=prefix)
                 for line in _CONSTITUTIVE_STATEMENTS)


def _touches(statement: str, target: int) -> bool:
    """Does one certified constitutive statement belong to component ``target``?

    The certified complex H body is three coefficient loads followed by three
    seven-line component blocks, and every line of a block names its component in a
    register (``prev0``, ``src0``, ``a0``) or in a pointer (``w0``, ``h0``). The
    coefficient loads name TWO registers each and are excluded by the ``float kp_``
    prefix, which is where this family lifts them separately.
    """
    if statement.startswith("float kp_"):
        return False
    marks = (f"prev{target}", f"src{target}", f"a{target} ", f"w{target}[",
             f"h{target}[")
    return any(mark in statement for mark in marks)


def constitutive_transcription(expansion: str,
                               contract: str = shaders.CONTRACT_OFF
                               ) -> Dict[str, Any]:
    """Is :data:`_CONSTITUTIVE_STATEMENTS` the certified complex H body, statement for statement?

    THE MEASUREMENT THAT MAKES THE PARAMETERISATION HONEST. This family re-emits the
    certified constitutive arithmetic twice per folded component — once at the owned
    cell and once at the imaged ghost — so it cannot splice that body wholesale the way
    :mod:`.complex_fused_magnetic_pair` does. What it can do is render its own template
    with the CERTIFIED spellings and require the result to be exactly the lifted body's
    statements, which is what this returns.

    ``identical`` false is a build-time failure, not a diagnostic:
    :func:`folded_complex_fused_magnetic_pair_source` calls this and raises.
    """
    body = certified_constitutive_body(expansion, contract)
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


#: The Yee shift a fill pass images: the NEAR pass writes the shift-0 components and
#: the FAR pass the shift-1 ones (``folded_complex.folded_mirror_fill_complex_source``
#: reads exactly this off the same table). Named rather than spelled twice.
_PASS_SHIFT: Dict[str, int] = {"near": 0, "far": 1}


def _certified_fill_line(target: int, pass_name: str) -> str:
    """The certified folded complex fill's own line for one component and one pass."""
    if pass_name == "near":
        return _CERTIFIED_FILL_LINE.format(slot=target, index=NEAR_SOURCE_INDEX)
    return _CERTIFIED_FAR_FILL_LINE.format(slot=target)


def _certified_fill_operand(target: int, pass_name: str) -> Tuple[str, str]:
    """That line's ``(destination, source operand)`` expressions, as it spells them."""
    if pass_name == "near":
        return (f"f{target}[base]",
                f"f{target}[base + {NEAR_SOURCE_INDEX} * stride]")
    return (f"f{target}[base + last * stride]",
            f"f{target}[base + reflect_row * stride]")


def _chain_step(target: int, tag: str, axis: int, pass_name: str,
                first: bool, last: bool) -> str:
    """ONE ``c_mul`` of the parity chain — THE ONE PLACE it is spelled.

    The emitter and both transcription checks call this, so a line that stopped being
    the certified fill's cannot be one in the kernel and another in the measurement.

    The LAST application writes straight to the ghost cell, which reproduces the
    certified fill's one-line shape exactly; the earlier ones land in a register,
    because the array path's earlier passes land in a memory cell the later pass then
    reads and this kernel threads that value through a register instead. A chain of
    length one therefore emits precisely what this family emitted before the far
    carry: ``f{t}[g{t}_..._i] = c_mul(<parity>, v{t});``.
    """
    coefficient = (_PARITY_REGISTER[axis] if pass_name == "near"
                   else _FAR_PARITY_REGISTER[axis])
    source = f"v{target}" if first else f"{tag}_v"
    if last:
        return f"f{target}[{tag}_i] = c_mul({coefficient}, {source});"
    declaration = "float2 " if first else ""
    return f"{declaration}{tag}_v = c_mul({coefficient}, {source});"


def _fill_transcription(axis: int, pass_name: str, expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """Is this family's ghost write the CERTIFIED folded complex fill's own line?

    THE SECOND MEASUREMENT, and it covers the one piece of arithmetic that is neither
    curl nor constitutive. :func:`.folded_complex.folded_mirror_fill_complex_source`
    emits

        near:  ``f{slot}[base] = c_mul(c, f{slot}[base + 2 * stride]);``
        far:   ``f{slot}[base + last * stride] =
                     c_mul(c, f{slot}[base + reflect_row * stride]);``

    — the coefficient on the LEFT, the whole complex product, no shortcut for
    ``phase == +1``, and a DIFFERENT destination and source row per pass. This family
    writes the same calls at different flat indices with the coefficient register
    renamed, because the lifted curl head already owns the name ``c``.

    So the comparison is made under a STATED rename rather than on the raw strings:
    the fill's own emitted lines are read back, the names this family changes are
    applied to them, and the result must be the line this family emits. Both roles a
    chain link can take are checked — the LAST application, which writes the ghost
    cell, and an EARLIER one, which writes the carry register because the array path's
    corresponding pass writes a memory cell the next pass reads.
    ``identical`` false is a build-time failure.
    """
    from .folded_complex import GHOST_FILL_FAMILIES  # noqa: PLC0415
    from .folded_complex import folded_mirror_fill_complex_source  # noqa: PLC0415

    if pass_name not in _PASS_SHIFT:
        raise ValueError(f"pass_name must be near or far, got {pass_name!r}")
    axis = int(axis)
    names = tuple(GHOST_FILL_FAMILIES["B"]["targets"])
    shifts = tuple(TARGET_IYEE[name][axis] for name in names)
    source = folded_mirror_fill_complex_source(axis, pass_name, shifts, expansion,
                                               contract)
    lines = [line for line in source.splitlines()
             if line.strip().startswith("f") and "c_mul(c," in line]
    coefficient = (_PARITY_REGISTER[axis] if pass_name == "near"
                   else _FAR_PARITY_REGISTER[axis])
    tag = f"g{{target}}_t"
    rows: List[Dict[str, Any]] = []
    for target, shift in enumerate(shifts):
        if int(shift) != _PASS_SHIFT[pass_name]:
            continue
        certified = _certified_fill_line(target, pass_name)
        if certified not in lines:
            raise AssertionError(
                f"the certified folded complex {pass_name} fill no longer emits "
                f"{certified!r} on axis {axis}; this family carries that line inline "
                f"and cannot rename around its absence. emitted={lines!r}")
        destination, operand = _certified_fill_operand(target, pass_name)
        marker = tag.format(target=target)
        base = certified.replace("c_mul(c,", f"c_mul({coefficient},")
        for role, first, last, expected in (
                ("last", True, True,
                 _chain_step(target, marker, axis, pass_name, True, True)),
                ("earlier", True, False,
                 _chain_step(target, marker, axis, pass_name, True, False)),
                ("continued", False, False,
                 _chain_step(target, marker, axis, pass_name, False, False))):
            renamed = base.replace(
                operand, f"v{target}" if first else f"{marker}_v")
            renamed = renamed.replace(
                destination,
                f"f{target}[{marker}_i]" if last
                else (("float2 " if first else "") + f"{marker}_v"))
            rows.append({"target": target, "role": role,
                         "certified": certified.strip(),
                         "renamed": renamed.strip(), "emitted": expected})
    if not rows:
        raise AssertionError(
            f"axis {axis} images no B component on the {pass_name} pass; "
            f"_fill_transcription would measure nothing and an empty comparison "
            f"reports as a pass")
    return {"axis": axis, "pass": pass_name, "rows": rows,
            "identical": all(row["renamed"] == row["emitted"] for row in rows)}


def near_fill_transcription(axis: int, expansion: str,
                            contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """:func:`_fill_transcription` for the NEAR pass — ``cell 0 = parity (x) cell 2``."""
    return _fill_transcription(axis, "near", expansion, contract)


def far_fill_transcription(axis: int, expansion: str,
                           contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """:func:`_fill_transcription` for the FAR pass — ``last = parity (x) reflect row``.

    The pass this family refused BY NAME until the far carry. It exists as its own
    entry point because the far line is NOT the near line with an index moved: it
    changes the destination as well as the source, and it takes the OTHER of
    :func:`.folded_complex.mirror_parity_coefficients`' two words.
    """
    return _fill_transcription(axis, "far", expansion, contract)


def zero_metal_lines(target: int, zero_metal: Sequence[bool], name: str,
                     indent: str = "    ",
                     zero: str = templates.COMPLEX_ZERO) -> List[str]:
    """``stepping.zero_metal_B`` for ONE target, on one named register.

    The rows are :data:`.complex_fused_magnetic_pair._ZERO_METAL_ROWS`, IMPORTED
    rather than re-derived: that table is already the B family's Yee DIAGONAL (Bx on
    an x wall, By on y, Bz on z) and is already pinned against the array path.

    The zero is a COMPLEX zero — BOTH planes — because the array path assigns one
    (stepping.py:1943, :1949); a real ``0.0f`` here would not even compile, but the
    plane-wise question it stands for is live and the gate carries a mutation that
    clears only the real plane. The select spelling is
    :func:`.templates.ownership_mask`'s, ``flag ? zero : v``, the form measured to
    deliver an exact ``+0.0`` on this platform.
    """
    return [f"{indent}{name} = {flag} ? {zero} : {name};"
            for row_target, axis, flag in _ZERO_METAL_ROWS
            if row_target == target and bool(zero_metal[axis])]


def _ghost_coefficient_lines(target: int, axis: int, tag: str,
                             indent: str) -> List[str]:
    """The NEAR destination's constitutive coefficient pair, read at stored index 0.

    THE LINE NOTHING ELSE ON THIS SEAM HAS TO WRITE. The NEAR fill images along the
    component's OWN axis (``iyee[Bm][a] == 0`` iff ``a == m``), which is exactly the
    axis ``update_H`` indexes on, so the fill's source (stored 2) and destination
    (stored 0) take DIFFERENT coefficient entries. Reusing the source thread's
    ``kp_t``/``km_t`` would apply the absorber profile of stored cell 2 to stored cell
    0 — smooth, converged and wrong inside the PML, invisible outside it — so the pair
    is loaded again at index 0. The coefficients are float32 in BOTH storage modes
    (stepping.py:41-50), so these two loads are the real fold's two loads exactly.

    THE FAR FILL NEEDS NO SUCH LINE, and that asymmetry is derived rather than
    asserted: ``complex_fields``' constitutive template indexes target ``t`` on axis
    ``t`` (``stepping.H_CONSTITUTIVE_TERMS``:226) and :func:`far_fill_axes` returns
    only axes ``a != t``, so a far destination sits at the SAME coordinate on the
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
        f"(complex_fields' constitutive template indexes target {target} on {name}).",
        f"{indent}float {tag}_kp = kp{target}[0], "
        f"{tag}_km = km{target}[0];",
    ]


def _constitutive_block(target: int, index: str, source: str, kp: str, km: str,
                        prefix: str, indent: str) -> List[str]:
    """``_apply_constitutive_pml`` for one component at one cell, indented."""
    return [indent + line for line in
            _render_constitutive(target, index, source, kp, km, prefix)]


def _carry_blocks(target: int, near: Sequence[int], far: Sequence[int],
                  zero_metal: Sequence[bool], indent: str) -> List[str]:
    """Every imaged ghost this thread owns, then ``update_H`` at each of them.

    ONE BLOCK PER DESTINATION IN :func:`carried_destinations`. Each is guarded on the
    flags that say this thread is the source of that destination — stored
    ``NEAR_SOURCE_INDEX`` on the near axis, the runtime reflect row on each far axis —
    and writes the parity CHAIN applied to ``v``, followed by the certified
    constitutive at that index.

    THE CHAIN IS EMITTED ONE ``c_mul`` PER PASS, IN :func:`parity_chain`'S ORDER, and
    it is not folded into a single word. That is this family's whole delta from
    :mod:`.folded_fused_magnetic_pair`, and it is a MEASUREMENT rather than a
    preference: under complex storage, re-ordering a two-parity chain moves 8 of 128
    uint32 words on an exhaustive signed-zero/subnormal table and folding it into one
    word moves up to 23 of 128, at every sign combination
    (``results/complex_parity_chain_order_2026-08-21/parity_chain_order.json``). Each
    chain starts from ``v{target}`` again rather than from a shorter chain's register:
    the value is a pure function of ``v`` and the ordered coefficient list, so the
    recomputation is the same expression, and the prefix registers live inside their
    own guards.

    ``v{target}`` IS READ AFTER THE WALL CLEAR, deliberately, and the driver order is
    why: ``zero_metal_B`` (driver.py:3286) runs BEFORE ``fill_folded_far_ghosts_B``
    (:3287), so a far image of a cleared row must carry the cleared value. The near
    fill runs BEFORE the clear (:3285), but its destination (stored 0 of a FOLDED
    axis) and the clear's (stored 0 of a WALLED axis) are disjoint per component, so
    reading the same post-clear register for both is exact.

    NO WALL CLEAR IS EMITTED AT A GHOST, and the two fills reach that conclusion by
    DIFFERENT routes, both checked below rather than assumed:

    * the NEAR ghost moves the component to stored 0 of its OWN axis — the exact cell
      ``_zero_metal`` clears — but that axis is FOLDED here, and ``_zero_metal`` skips
      a folded axis (stepping.py:2284-2286), so no line can apply. A drift in either
      table would put one there, so it is asserted;
    * the FAR ghost moves the component along an axis that is NOT its own, and
      :data:`._ZERO_METAL_ROWS` is the B DIAGONAL, so the ghost sits at the SAME
      coordinate on every axis that could clear it as the thread that owns it. Its
      clear status is therefore its source's, INHERITED by reading ``v{target}`` after
      the clear lines rather than re-emitted.
    """
    walls = tuple(axis for row_target, axis, _ in _ZERO_METAL_ROWS
                  if row_target == target and bool(zero_metal[axis]))
    if near and zero_metal_lines(target, zero_metal, "unused", indent):
        raise AssertionError(
            f"component {target} images a NEAR ghost on folded axis {near[0]} AND "
            f"zero_metal_B clears it on axes {walls}. stepping._zero_metal skips a "
            f"folded axis, so the two tables have drifted")
    overlap = sorted(set(walls) & {int(axis) for axis in far})
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
        # THE GUARD, THE INDEX AND THE COMMENTS ALL FOLLOW THE CHAIN, so a reader sees
        # the passes in the order the driver runs them rather than in an order chosen
        # here. The guard's conjunction and the index's sum are order-INDEPENDENT; the
        # multiplies below are not, which is the whole point of driving all three off
        # one list.
        chain = parity_chain(near, subset, carries_near)
        flags: List[str] = []
        terms: List[str] = []
        described: List[str] = []
        for axis, pass_name in chain:
            if pass_name == "near":
                flags.append(f"{_COORDINATE[axis]} == {NEAR_SOURCE_INDEX}")
                terms.append(f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}")
                described.append(
                    f"the near fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = 0 "
                    f"imaged from {NEAR_SOURCE_INDEX}, parity word "
                    f"{_PARITY_REGISTER[axis]} "
                    f"(stepping._fill_symmetry_ghost_cells:1441, "
                    f"_write_mirror_ghost:1451)")
            else:
                flags.append(f"{_COORDINATE[axis]} == {_REFLECT[axis]}")
                terms.append(f"+ ({_LAST[axis]} - {_REFLECT[axis]}) * {_STRIDE[axis]}")
                described.append(
                    f"the far fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = "
                    f"{_LAST[axis]} imaged from {_REFLECT[axis]}, parity word "
                    f"{_FAR_PARITY_REGISTER[axis]} "
                    f"(stepping._fill_folded_far_ghosts:1518)")
        lines.append(f"{indent}if ({' && '.join(flags)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(
            f"{inner}// the parity CHAIN, in the driver's own pass order "
            f"(near first, then far ascending):")
        lines.append(
            f"{inner}// "
            + " then ".join(f"{pass_name} on {_COORDINATE[axis]}"
                            for axis, pass_name in chain)
            + ". NOT folded into one word: order and grouping both move bytes here")
        lines.append(
            f"{inner}// (results/complex_parity_chain_order_2026-08-21). "
            f"folded_complex.folded_mirror_fill_complex_source is the emitter every")
        lines.append(f"{inner}// line below is lifted from.")
        lines.append(f"{inner}int {tag}_i = ii {' '.join(terms)};")
        for step, (axis, pass_name) in enumerate(chain):
            lines.append(inner + _chain_step(
                target, tag, axis, pass_name,
                first=step == 0, last=step == len(chain) - 1))
        lines.append(f"{inner}float2 {tag}_b = f{target}[{tag}_i];")
        if carries_near:
            lines.extend(
                _ghost_coefficient_lines(target, int(near[0]), tag, inner))
            kp, km = f"{tag}_kp", f"{tag}_km"
        else:
            # The far destination shares the indexed axis with its source thread,
            # so the certified pair this thread already loaded IS the destination's.
            kp, km = f"kp_{target}", f"km_{target}"
        lines.extend(_constitutive_block(
            target, f"{tag}_i", f"{tag}_b", kp, km, f"{tag}_", inner))
        lines.append(f"{indent}}}")
    return lines


def folded_complex_fused_magnetic_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded quadruple, Bloch flags, walls, arm, mode).

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC / MIRROR_PERIODIC
    quadruple :func:`.symmetry.folded_axis_kinds` resolves — never a hand-built triple,
    because the MIRROR_METALLIC / MIRROR_PERIODIC split is this family's single point
    of failure and getting it backwards on one axis is a plane of wrong values rather
    than a crash.

    ``phased`` is the per-axis Bloch flag :func:`.complex_fields.phase_arguments`
    resolves. THE PARITY IS NOT HERE: unlike the real fold it is a runtime word, so it
    does not specialise the source and two grids differing only in a mirror parity
    compile to the SAME kernel. That is a property this module states because it is
    the opposite of :mod:`.folded_fused_magnetic_pair`'s, and a gate that assumed
    otherwise would arm a parity mutation on a source that cannot carry one.

    Every constant Triton would bake into a ``tl.constexpr`` is baked into the string,
    for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a source
    and nothing else.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if not any(code in MIRROR_CODES for code in codes):
        raise ValueError(
            "no axis is folded: this product exists to carry the mirror fill inside "
            "the B seam, and an unfolded complex grid belongs to "
            "complex_fused_magnetic_pair")
    phased = tuple(int(flag) for flag in phased)
    if len(phased) != 3:
        raise ValueError(f"phased must be a per-axis triple, got {phased!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    # THE DISJOINTNESS THE CARRY RESTS ON, asserted rather than trusted:
    # `zero_metal_axes` is `is_metallic and not is_mirrored` (stepping.py:2284-2286),
    # so a folded axis can never be a walled one. If that changed, the imaged ghost
    # would need a wall line this kernel does not emit.
    for axis, code in enumerate(codes):
        if code in MIRROR_CODES and zero_metal[axis]:
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction "
                f"(stepping.py:2284-2286) and this kernel's ghost carry relies on the "
                f"two sets being disjoint")
    # A phased FOLDED or METALLIC axis is refused by the certified emitter itself
    # (folded_complex.folded_bloch_curl_source, mirroring `_reachable_phase_flags`).
    # It is not re-checked here: one raise, in the emitter that owns the rule.

    transcription = constitutive_transcription(expansion, contract)
    if not transcription["identical"]:
        raise AssertionError(
            "this family's parameterised constitutive statements no longer reproduce "
            "complex_fields.bloch_constitutive_source('H'); the fused arithmetic "
            "would silently stop being the certified arithmetic. emitted="
            f"{transcription['emitted']!r} certified={transcription['certified']!r}")

    statements = certified_curl_statements(codes, phased, expansion, contract)
    body: List[str] = [
        certified_curl_head(codes, phased, expansion, contract).rstrip("\n")]
    periodic = tuple(axis for axis, code in enumerate(codes)
                     if code == CODE_MIRROR_PERIODIC)
    body.extend([
        "",
        ("    // The three top-plane flags the lifted head declares are read below by "
         "the far"
         if periodic else
         "    // The three top-plane flags the lifted head declares are unread here:"),
        ("    // carry's ownership guard, and by folded_top_plane_mask's own lines "
         "above."
         if periodic else
         "    // no folded PERIODIC axis, so folded_top_plane_mask emits no line and "
         "the"),
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
    body.extend(_constitutive_coefficient_lines(expansion, contract))
    body.append("")

    # The three recurrences and the three `fu` stores, in the certified body's own
    # order. `u` IS WRITTEN AT EVERY CELL, ghost cells included: `step_B` updates `fu`
    # everywhere and the fill touches B only (stepping.py:1451 writes `field`, never
    # `fu_field`).
    for target in range(3):
        previous, recurrence, _ = statements["recurrence"][target]
        body.extend([previous, recurrence])
    body.append("")
    body.append(statements["aux_store"])

    for target in range(3):
        near = near_fill_axes(codes, target)
        if len(near) > 1:
            raise AssertionError(
                f"component {target} is a near-fill destination on {near}; for the B "
                f"family that set is at most one axis (the component's own) and this "
                f"carry images one near ghost cell per component")
        far = far_fill_axes(codes, target)
        if len(far) > 2 or (near and near[0] in far):
            raise AssertionError(
                f"component {target} is a far-fill destination on {far} against a "
                f"near axis {near}; for the B family the two sets are COMPLEMENTARY "
                f"(iyee is 0 on the component's own axis and 1 on the other two, "
                f"fields.IYEE_SHIFTS), so far holds at most two axes and never the "
                f"near one")
        for axis in near:
            fill = near_fill_transcription(axis, expansion, contract)
            if not fill["identical"]:
                raise AssertionError(
                    "this family's ghost write is no longer the certified folded "
                    "complex NEAR fill's own line under the stated rename: "
                    f"{fill['rows']!r}")
        for axis in far:
            fill = far_fill_transcription(axis, expansion, contract)
            if not fill["identical"]:
                raise AssertionError(
                    "this family's ghost write is no longer the certified folded "
                    "complex FAR fill's own line under the stated rename: "
                    f"{fill['rows']!r}")
        _, _, displacement = statements["recurrence"][target]
        body.append("")
        # A cell any fill writes is OWNED BY ITS SOURCE THREAD. This thread stops
        # after fu there: the array path's fill overwrites the displacement it would
        # store, and forming v would read f at a word another thread writes.
        owned: List[str] = [f"{_COORDINATE[axis]} == 0" for axis in near]
        owned.extend(_LAST_FLAG[axis] for axis in far)
        if owned:
            body.extend([
                f"    // component {target}: the fills image "
                + ", ".join(
                    [f"stored cell 0 on {_COORDINATE[axis]} (near)" for axis in near]
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
            body.extend(_carry_blocks(target, near, far, zero_metal, indent))
            body.append("    }")

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": "\n".join(body),
    })


def compile_folded_complex_fused_magnetic_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, Bloch flags, walls, arm, mode)."""
    return compile_source(folded_complex_fused_magnetic_pair_source(
        codes, phased, zero_metal, expansion,
        contract)).folded_complex_fused_magnetic_pair_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _mirror_phase_reasons(grid: Any, codes: Optional[Sequence[int]]) -> List[str]:
    """Every folded axis must declare a readable +1/-1 parity.

    ``stepping._symmetry_phase`` RAISES on a ``None`` phase (stepping.py:2463-2469)
    rather than folding with the even-mirror default. The parity is a runtime word
    here rather than a source specialisation, which changes nothing about this clause:
    an undeclared parity would be passed as some default and the run would be wrong by
    twice the field wherever the parity mattered.
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
                f"the parity word this kernel multiplies the ghost by cannot be "
                f"defaulted")
    return reasons


def _far_carry_reasons(grid: Any, codes: Optional[Sequence[int]]) -> List[str]:
    """What a folded PERIODIC axis must satisfy for the far carry to be legal.

    THE CLAUSE THIS FAMILY USED TO REFUSE BY NAME. ``fill_folded_far_ghosts_B`` runs
    inside this seam (driver.py:3287) on a folded PERIODIC axis and images the top
    stored slot from ``stepping._far_reflect_rows``' row; the kernel now carries it,
    and these are the conditions the carry's OWNERSHIP MOVE rests on. The clause list
    is :func:`.folded_fused_magnetic_pair._far_carry_reasons`' — the two carries face
    the SAME geometric question, and the complex board's delta is the parity CHAIN,
    which is arithmetic rather than admission.

    * ``_stored_past_owned`` must AGREE with the code, both ways. The code decides
      whether the kernel emits a far carry at all AND whether the lifted curl head
      masks the top plane, so a disagreement is a plane the kernel steps and images
      inconsistently with the array path.
    * the reflect row must exist, must lie inside the stored extent below the plane it
      writes, and must not be stored 0 (the plane the near fill writes).
    * an axis storing exactly ``NEAR_SOURCE_INDEX + 1`` cells would make the far
      carry's write plane the near fill's read plane; refused rather than made
      order-dependent, exactly as ``symmetry.mirror_ghost_fill_coverage`` refuses it.

    MEASURED, over every folded PERIODIC extent ``Grid`` will build, none of these
    fires: the real fold's own sweep (58 extents, cells 3..60) found
    ``stored == ceil(n_full/2) + 2``, ``reflect == floor(n_full/2)`` and a minimum
    stored of 4, with zero violations
    (``results/metal_folded_far_carry_2026-08-20/grid_reflect_row_law.json``). They are
    asked anyway, because here a violated row is an OUT-OF-RANGE WRITE from a thread
    that is not the destination's, not a soft error on a whole-plane assignment.
    """
    from ..stepping import _far_reflect_rows, _stored_past_owned  # noqa: PLC0415

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
        row = int(row)
        extent = int(shape[axis])
        if not 0 <= row < extent - 1:
            reasons.append(
                f"axis {axis} reflect row {row} is outside [0, {extent - 1}); the far "
                f"carry would image the plane it writes, or read outside the "
                f"allocation from a thread that owns neither cell")
        if row == 0:
            reasons.append(
                f"axis {axis} reflect row is 0, the plane the NEAR fill writes: the "
                f"far carry would image a ghost rather than an owned cell")
        if extent - 1 == NEAR_SOURCE_INDEX:
            reasons.append(
                f"axis {axis} stores {extent} cells, so the far carry's write plane "
                f"IS the near fill's read plane (cell {NEAR_SOURCE_INDEX}); refused "
                f"rather than made order-dependent, exactly as "
                f"symmetry.mirror_ghost_fill_coverage refuses it")
    return reasons


def metal_folded_complex_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span folded complex ``step_B`` -> near fill -> wall -> ``update_H``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. That is the
    construction every shipped fused pair uses.

    ``probe`` is the expansion artifact both halves bind through. It is threaded, not
    re-read: a second read would be a second answer to one question, and a family that
    resolved its own arm could disagree with the halves it is made of.
    """
    reasons: List[str] = []

    curl = folded_complex_composition_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency, probe)
    if not curl.covered:
        reasons.extend(f"folded complex curl half: {reason}"
                       for reason in curl.reasons)
    magnetic = folded_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency, probe)
    if not magnetic.covered:
        reasons.extend(f"folded complex constitutive half: {reason}"
                       for reason in magnetic.reasons)

    # THE SOURCE SEAM. A MAGNETIC source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B.
    # IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. An ELECTRIC source is
    # injected in the D/E half and does NOT disqualify this pair.
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
    # degenerate object should still get its polarity named rather than only "no grid".
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    reasons.extend(_mirror_phase_reasons(grid, codes))
    reasons.extend(_far_carry_reasons(grid, codes))

    # THE WALL CLEAR IS CARRIED INLINE, so the grid must be able to answer which axes
    # are walled. A grid that cannot would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        # THE NEAR FILL IMAGES STORED CELL 2, and the source thread must exist. The
        # folded complex curl predicate already refuses a folded axis storing too few
        # cells; restated because THIS family writes the destination from a DIFFERENT
        # thread and a missing source plane is an out-of-range write, not a soft error.
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; the "
                    f"near fill images stored cell {NEAR_SOURCE_INDEX} and this "
                    f"kernel images it from that cell's own thread")

        # THE WALL CLEAR AND THE FILL MUST NOT MEET. On this family both act on
        # component m at stored cell 0 of axis m, and `_zero_metal` skips a folded axis
        # so they are disjoint by construction. CHECKED rather than inferred: reading
        # coverage off another module's guard is what this package's rule forbids, and
        # an overlap would be a plane of wrong values.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall there; "
                    f"stepping._zero_metal skips a folded axis, so the two passes "
                    f"have drifted and the carry's disjointness no longer holds")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedComplexFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs four driver passes, complex B never leaving a register.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no special
    case for it.

    ``runs`` is an alias of ``launches`` rather than a second counter, because with one
    dispatch per run the two ARE the same number and a second counter would be a second
    thing to get out of step.
    """

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values", "phases",
                 "parity_values", "far_parity_values", "reflect", "zero_metal",
                 "shape", "dtdx", "expansion", "params", "carried_axes",
                 "far_carried_axes")

    family = "folded complex fused B-curl/mirror-fill/stored-H pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phased", "phases", "zero_metal",
                   "carried_axes", "far_carried_axes", "reflect", "expansion")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 phases: Sequence[int],
                 parity_values: Sequence[Tuple[float, float]],
                 far_parity_values: Sequence[Tuple[float, float]],
                 reflect: Sequence[Optional[int]],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 expansion: str, params: Any, pointers: Sequence[Any],
                 functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.phases = tuple(int(value) for value in phases)
        self.parity_values = tuple((float(re), float(im))
                                   for re, im in parity_values)
        self.far_parity_values = tuple((float(re), float(im))
                                       for re, im in far_parity_values)
        self.reflect = tuple(None if value is None else int(value)
                             for value in reflect)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # float(): the array path multiplies a float32 volume by a Python float and
        # the packed struct holds numpy.float32 of the same value, so the two scalars
        # are the same bits.
        self.dtdx = float(dtdx)
        self.expansion = str(expansion)
        self.params = params
        self.carried_axes = tuple(
            near_fill_axes(self.codes, target) for target in range(3))
        self.far_carried_axes = tuple(
            far_fill_axes(self.codes, target) for target in range(3))
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def params_record_dtype() -> Any:
    """The host record's dtype — NINE ``float2`` members first, itemsize 104.

    ONE HOME FOR THE LAYOUT, because getting it wrong is a silent wrong answer rather
    than a crash: see :mod:`.complex_fused_magnetic_pair`'s own record for the
    measurement behind the ordering. The offsets are stated EXPLICITLY even though the
    complex-members-first order makes them the natural ones, so that the record cannot
    drift into agreeing with Metal only by accident; ``itemsize`` is stated because a
    record whose size the toolchain chose is a record nobody checked.
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["px", "py", "pz", "m0", "m1", "m2", "d0", "d1", "d2",
                  "nx", "ny", "nz", "n_elem", "dtdx", "rx", "ry", "rz"],
        "formats": [("<f4", 2), ("<f4", 2), ("<f4", 2),
                    ("<f4", 2), ("<f4", 2), ("<f4", 2),
                    ("<f4", 2), ("<f4", 2), ("<f4", 2),
                    "<u4", "<u4", "<u4", "<u4", "<f4", "<i4", "<i4", "<i4"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56, 64,
                    72, 76, 80, 84, 88, 92, 96, 100],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]],
                   parity_values: Sequence[Tuple[float, float]],
                   far_parity_values: Sequence[Tuple[float, float]],
                   reflect: Sequence[Optional[int]],
                   device: str) -> Any:
    """The scalars, Bloch phases, BOTH parity triples and the reflect rows, packed.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every other packed
    struct in this package gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME, because a wider element silently
    reinterprets. This record is neither — it is a packed struct of nine float2s, four
    uints, a float and three ints that nothing on the host reads back and nothing on
    the device writes — so it is built here, once, at plan time, and held by the plan.
    Nothing on the launch path allocates.

    ``-1`` stands for an axis with no reflect row (unfolded, or folded METALLIC, where
    the stored array stops at ``big_corner`` and no far ghost exists). It is a value
    the kernel never reads: the guard that would read it is emitted only for an axis
    :func:`far_fill_axes` returned, and that requires ``CODE_MIRROR_PERIODIC``. A
    sentinel rather than 0, so a source that read it anyway would index out of the
    volume and be caught, not silently image row 0.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    for name, (real, imag) in zip(("px", "py", "pz"), tuple(phase_values)):
        record[name] = (np.float32(real), np.float32(imag))
    for name, (real, imag) in zip(("m0", "m1", "m2"), tuple(parity_values)):
        record[name] = (np.float32(real), np.float32(imag))
    for name, (real, imag) in zip(("d0", "d1", "d2"), tuple(far_parity_values)):
        record[name] = (np.float32(real), np.float32(imag))
    rows = tuple(-1 if value is None else int(value) for value in reflect)
    if len(rows) != 3:
        raise ValueError(f"reflect must be a per-axis triple, got {reflect!r}")
    record["rx"], record["ry"], record["rz"] = rows
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def _parity_words(codes: Sequence[int], phases: Sequence[int], half: int,
                  admits: Tuple[int, ...]) -> Tuple[Tuple[float, float], ...]:
    """One half of the per-axis parity pair, host-rounded through ``numpy.complex64``.

    :func:`.folded_complex.mirror_parity_coefficients` is the ONE place a mirror parity
    becomes a complex word in this package, and it returns the near and far pair
    together — ``+phase`` for a shift-0 destination and ``-phase`` for a shift-1 one,
    both rounded on the host and PASSED, never synthesised in-kernel. An axis this half
    does not apply to takes ``(0.0, 0.0)``, a word no emitted line reads.
    """
    out: List[Tuple[float, float]] = []
    for axis, code in enumerate(codes):
        if int(code) in admits:
            word = mirror_parity_coefficients(int(phases[axis]))[half]
            out.append((float(word[0]), float(word[1])))
        else:
            out.append((0.0, 0.0))
    return tuple(out)


def parity_words(codes: Sequence[int], phases: Sequence[int]
                 ) -> Tuple[Tuple[float, float], ...]:
    """The per-axis NEAR parity word — ``+phase`` on any folded axis."""
    return _parity_words(codes, phases, 0, tuple(MIRROR_CODES))


def far_parity_words(codes: Sequence[int], phases: Sequence[int]
                     ) -> Tuple[Tuple[float, float], ...]:
    """The per-axis FAR parity word — ``-phase``, on a folded PERIODIC axis only.

    A SEPARATE WORD RATHER THAN A NEGATION of the near one, and the admitting set is
    narrower: ``fill_folded_far_ghosts_B`` runs only where ``_stored_past_owned`` is
    True, so a folded METALLIC axis carries the zero word here and the near word there.
    """
    return _parity_words(codes, phases, 1, (CODE_MIRROR_PERIODIC,))


def plan_metal_folded_complex_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        ) -> Optional[MetalFoldedComplexFusedMagneticPairPlan]:
    """Build the fused folded complex B/H plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason
    :func:`.folded_complex.plan_folded_complex_pml_curl` gives: a configuration this
    kernel does not carry must fall back to the array path, never raise into a caller
    that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING — every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not metal_folded_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    kinds = _boundary_kinds(grid, pml)
    walls = zero_metal_axes(grid)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    # backward=False: this is the B seam. The phase table is resolved through the
    # certified helper so the conjugation rule lives in one place.
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds), backward=BACKWARD)
    parities = parity_words(codes, phases)
    far_parities = far_parity_words(codes, phases)
    # `stepping._far_reflect_rows` is the ONE derivation of the row a folded PERIODIC
    # axis's far face images; asked here rather than re-derived, and passed as a
    # runtime int because nothing about an integer row can move a bit.
    from ..stepping import _far_reflect_rows  # noqa: PLC0415

    reflect = _far_reflect_rows(grid)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step, so
    # the groups are built in the order the kernel declares them. There is NO
    # inverse-mu volume: the H product is `source = B` with mu = 1 baked into the array
    # path too, which is what makes this signature 27 pointers rather than 30.
    flux = [bind_complex(name, getattr(fields, name))
            for name in curl_spec["targets"]]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in curl_spec["targets"]]
    electric = [bind_complex(name, getattr(fields, name))
                for name in curl_spec["sources"]]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in side_spec["targets"]]
    workspace = [bind_complex(name, getattr(fields, name))
                 for name in side_spec["aux"]]
    # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER one
    # (SUB_STEPS['step_B']['suffix'] == '_h'; CONSTITUTIVE_SIDES['H']['half_integer']
    # is False). That is the OPPOSITE pairing to the folded D/E pair, and the kernel
    # cannot tell — the gate carries a host mutation for each group.
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}{curl_spec['suffix']}",
                  getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}"))
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + electric + magnetic + workspace
                + curl_coefficients + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_complex_fused_magnetic_pair(
                codes, flags, walls, expansion, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor would
    # read "mps:0" where the mirrors were built with "mps" and put the struct on a
    # nominally different device from the volumes it describes.
    dtdx = grid.dt / grid.dx
    return MetalFoldedComplexFusedMagneticPairPlan(
        residency, volumes, codes, flags, values, phases, parities, far_parities,
        reflect, walls, grid.shape, dtdx, expansion,
        _params_tensor(grid.shape, dtdx, values, parities, far_parities, reflect,
                       residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"folded complex fused magnetic pair cannot fill {slot}",))
    return metal_folded_complex_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedComplexFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_complex_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY, and it is every other fused pair's.
    ``plan_step`` assigns at most one arm per slot and this product spans four, so
    there is no slot it could claim without a composition rule nothing has measured.
    Registering unwired keeps it ENUMERABLE for the disjointness sweep
    (``arms.registered``) while ``arms.arms_for`` skips it, so ``plan_step`` cannot
    select it and no existing arm's selection changes.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "folded complex fused B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded complex fused B/H pair: ",
                          noun="folded complex PML B-curl/mirror-fill/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
