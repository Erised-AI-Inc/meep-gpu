"""The FOLDED BETA COMPLEX fused Metal pair: a folded ``special_kz`` Bloch ``step_B``
welded into ``update_H``.

:mod:`.folded_complex_fused_magnetic_pair` WITH ONE EMITTER SWAPPED, and the swap is
the whole module. The certified folded complex Bloch curl body is replaced by the
certified FOLDED BETA complex curl body — :func:`.folded_beta.folded_beta_bloch_curl_source`
with ``backward=False`` and ``has_beta=True`` — and everything else is IMPORTED from
that shipped weld rather than re-spelled: the near mirror carry, the far ghost carry,
the parity chain, the wall clear, the ghost coefficient reload, the constitutive
transcription and the constitutive block emitter are all that module's own functions,
called here. The idiom is :mod:`.beta_fused_magnetic_pair`'s, which is the same swap
performed on the REAL unfolded seam.

This is the cell ``(folded beta complex, folded beta complex)`` at ``B_to_H``, and it
is the one B/H cell neither shipped folded weld can stand in for.
:mod:`.folded_complex_fused_magnetic_pair` refuses ``grid.beta`` through both halves'
own clause (``folded_complex._folded_complex_grid_reasons`` clause 8, which names "the
folded beta product" as the owner); :mod:`.beta_fused_magnetic_pair` refuses a fold by
name ("axis {a} is folded: stepping.fill_symmetry_bc_B and
stepping.fill_folded_far_ghosts_B both run inside this seam and neither is carried by
this family") and is real-storage besides. This module is the product they name.

===========================================================================
WHAT IT REACHES, AND WHAT IT CANNOT
===========================================================================

The cell is driven by THREE corpus rows and only TWO of them are reachable:

* ``tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2``
* ``tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7``
* ``tests:TestSpecialKz.test_eigsrc_kz_0_complex`` — **UNREACHABLE ON ANY BACKEND**.
  It declares a MAGNETIC source, which ``driver.step`` deposits BETWEEN ``step_B`` and
  ``update_H`` (driver.py:3292-3293). No launch can straddle a deposit, so the clause
  is a driver fact rather than this kernel's scope, and it is refused by name below.

**THE ROW COUNT IS NOT RE-DERIVED HERE AND THIS MODULE CLAIMS NO CELL MOVE.** The Metal
census that placed those rows finished 2026-08-19 02:15 and ``folded_beta.py`` was
edited at 05:16; the re-close round ``results/metal_coverage_special_kz_reclose_2026-08-19``
re-ran ``TestSpecialKz.test_special_kz`` only, not these two rows. The standing board
(``results/fusion_matrix_metal_2026-08-26_canonical``) could bound that staleness
because NO fused product covered any folded-beta cell; this product is the first that
would, so the +2 is a MATRIX RUN this module does not perform and does not assert. What
is asserted here is the predicate and the binding arithmetic, both of which are
measured on every build.

The two in-seam fills are LIVE on both reachable rows and are therefore CARRIED, not
refused: ``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B``. ``zero_metal_B`` is
NOT live on either, and is carried anyway — the emitter compiles it out on an unwalled
grid, and leaving it uncarried would make the walled specialisation a silent gap rather
than a compiled-out block.

===========================================================================
THE FILL CARRY IS THE RISK, AND IT IS TAKEN DELIBERATELY
===========================================================================

:mod:`.folded_beta` DECLINES to claim ``fill_B`` and ``fill_folded_far_ghosts_B``
(folded_beta.py:46-53) — "registering a third fill arm here would make both fill slots
AMBIGUOUS and drop them to the array path, which is a coverage LOSS dressed as
completeness". That decision is about an ARM TABLE ROW, not about the arithmetic: the
fills copy ``parity * field[plane]`` and touch no beta term, so
:mod:`.folded_complex`'s fill predicate already admits a folded beta run and its
composition matrix has pinned exactly that for ``fold_complex_2d_beta`` since before
that family existed.

So the arithmetic this weld carries inline for those two passes is
:func:`.folded_complex.folded_mirror_fill_complex_source`'s own — reached through
:func:`.folded_complex_fused_magnetic_pair._carry_blocks`, which is the shipped weld's
emitter and is CALLED rather than copied. This product takes over two driver passes its
own curl family declines to own an ARM for, and it does so through the emitter of the
family that does own them. Registering unwired (below) is what keeps that from
disturbing either arm table.

===========================================================================
THE SIGNATURE — 27 pointers plus one packed struct, unmoved
===========================================================================

    3 B  +  3 fu_B  +  3 E   (float2, the curl half)
  + 3 H  +  3 f_w_H          (float2, the constitutive half)
  + 6 curl coefficients  +  6 constitutive coefficients   (float32)
  = 27 pointers

:data:`PACKED_BINDINGS` is IMPORTED from
:mod:`.folded_complex_fused_magnetic_pair` rather than re-spelled, because the beta
term adds NO POINTER: its two coefficients are complex SCALARS and they ride inside the
same ``constant Params&`` the phases and parities already ride in
(``beta_fused_magnetic_pair.py:38-48`` states and its shipped 27-pointer signature
confirms the same thing on the real seam). There is no inverse-epsilon volume, for
:mod:`.folded_complex_fused_magnetic_pair`'s reason: ``update_H`` reads ``B`` and
``mu`` and never an inverse epsilon, so the certified complex constitutive's three
inverse-epsilon pointers are DROPPED and the count is 27 rather than 30. 28 of the 31
bindings the platform allows (device.py:72).

What the beta term DOES add is TWO ``float2`` members inside the packed struct, which
costs no binding at all: the record grows from 104 to :data:`PARAMS_ITEMSIZE` bytes and
the signature does not move.

:data:`SEPARATE_SCALAR_BINDINGS` is spelled HERE rather than imported, because the beta
parent binds its non-pointer arguments DIFFERENTLY from the plain complex one: six
``constant float&`` phase words (not three ``constant float2&``) plus four beta words.
27 + 4 uint + 1 ``dtdx`` + 6 phase + 4 beta = 42, twelve over the ceiling.
:func:`refuted_separate_scalar_source` builds that signature so a gate can require the
COMPILE FAILURE, which is what turns the number from an argument into a measurement.
The six parity words and three reflect rows are NOT counted there, for the reason
:mod:`.folded_complex_fused_magnetic_pair` gives for its own 35: the refutation measures
the PARENT'S binding style, and this family's own additions ride in the struct.

===========================================================================
WHAT IS LIFTED, AND WHERE THE SWAP SHOWS
===========================================================================

* **the curl** — :func:`.folded_beta.folded_beta_bloch_curl_source`, whose head (guard,
  decode, ghost gather, wrapped-lane predicates, the Bloch rotation, the curl grouping,
  THE BETA INSERT, the cell-0 ownership mask and the folded top-plane mask) is spliced
  verbatim by :func:`certified_curl_head`, cut at :data:`_RECURRENCE_MARK` — the same
  cut point :mod:`.folded_complex_fused_magnetic_pair` takes. Its split-field recurrence
  statements are pulled line by line by :func:`certified_curl_statements`;
* **the beta term itself** — ``special_kz._COMPLEX_BETA_INSERT``, which arrives inside
  that head. :func:`folded_beta_complex_fused_magnetic_pair_source` ASSERTS both of its
  statements are present, for :mod:`.beta_fused_magnetic_pair`'s reason: a lift that
  silently produced the ``has_beta=0`` arm would be
  :mod:`.folded_complex_fused_magnetic_pair` wearing this family's name, and every
  device leg would still pass because beta = 0 IS that family's arithmetic;
* **the wall clear, the two fills, the parity chain and the ghost constitutive** —
  :func:`.folded_complex_fused_magnetic_pair._carry_blocks` and
  :func:`~.folded_complex_fused_magnetic_pair.zero_metal_lines`, CALLED. Those functions
  in turn lift ``stepping._zero_metal``'s B diagonal, the certified folded complex
  fill's own ``c_mul(coefficient, plane)`` line and the driver's pass ORDER, and they
  re-measure every one of those lifts on every build;
* **the constitutive half** — :func:`.complex_fused_magnetic_pair.certified_constitutive_body`,
  reached through :func:`.folded_complex_fused_magnetic_pair.constitutive_transcription`
  and :func:`~.folded_complex_fused_magnetic_pair._constitutive_block`. It is
  ``complex_fields.bloch_constitutive_source("H")``'s own body. Nothing beta-dependent
  reaches it: ``_apply_constitutive_pml`` (stepping.py:2112) never sees ``grid.beta``,
  which enters the two CURLS only (stepping.py:384-391, :467-474) — the measurement
  :func:`.folded_beta.folded_beta_complex_constitutive_coverage` already ships on.

**THE ONE PLACE THE SWAP CHANGES THIS MODULE'S OWN CODE** is the split-field recurrence
lift. ``complex_fields._CURL_TEMPLATE`` spells the recurrence as THREE statements per
target (``p``, ``n``, ``v``, with the intermediates nested); ``special_kz._BETA_BLOCH_CURL_TEMPLATE``
spells the SAME arithmetic as FIVE (``p``, ``q``, ``n``, ``r``, ``v``, with the
intermediates named). :func:`certified_curl_statements` therefore anchors five lines per
target where the shipped weld anchors three, and the emitter places ``q`` beside ``p``
above the ``fu`` store and ``r`` beside ``v`` inside the ownership guard — the same
split, at the same two points, over two more statements. Nothing is retyped: every one
of the five is ``_line_starting``'s answer off the certified body.

===========================================================================
THE EXPANSION ARM NEEDS BOTH PROBES, AND THEY MUST AGREE
===========================================================================

One ``templates.complex_helpers(expansion)`` emission serves the whole kernel, and this
kernel carries complex products from BOTH parents' call sites:

* the beta insert's ``c_mul(float2(bpr, bpi), b)`` with a purely IMAGINARY coefficient
  on the left — ``special_kz.BETA_PROBE_PATTERN``;
* the mirror fill's ``c_mul(parity, v)`` with a ±1 real PARITY coefficient on the left —
  ``folded_complex.PARITY_PROBE_PATTERN``;
* the Bloch rotation, the field-left ``kms`` multiplies and the coefficient-left
  constitutive multiplies, which both pattern lists carry.

NEITHER PATTERN LIST IS A SUPERSET OF THE OTHER (``folded_complex`` carries
``c8_mul_c8_scalar_right`` and ``special_kz`` does not; ``special_kz`` carries the
imaginary-coefficient pattern and ``folded_complex`` does not), so
:func:`expansion_from_probes` requires BOTH artifacts, licenses each through its OWN
parent's rule, and refuses unless the two name the SAME arm. Reading one and inheriting
the other's verdict would be licensing an orientation nobody measured, which is the
guess the whole probe mechanism exists to prevent.

===========================================================================
WHAT THIS FAMILY REFUSES, EACH BY NAME
===========================================================================

* ``beta == 0`` — :mod:`.folded_complex_fused_magnetic_pair` owns that seam; the
  refusal arrives through ``folded_beta``'s clause 12 on both halves, and
  :func:`folded_beta_complex_fused_magnetic_pair_source` additionally refuses to emit a
  body whose beta insert is absent;
* an UNFOLDED complex beta run — :mod:`.special_kz`'s certified Bloch beta curl and
  ``beta_run_complex_constitutive_coverage`` own those slots separately, and no weld is
  built over them here. The refusal arrives through ``folded_beta``'s clause 5;
* a REAL folded beta run — :mod:`.beta_fused_magnetic_pair` is the real seam's weld and
  refuses a fold; the real FOLDED B/H weld is not built on this backend. Clause 2
  (storage) is the inversion;
* a MAGNETIC source anywhere in the seam — refused by name, and on this cell that
  clause is exactly one row of three (driver.py:3292-3293). Ignorance is never an empty
  set: ``Fields`` does not hold the source list, so an undeclared ``sources`` is a
  REFUSAL and not an assumed ``()``. An ELECTRIC source is injected in the D/E half and
  does NOT disqualify this pair;
* the D/E seam — ``step_D``'s negated strides, conjugated phase, ELECTRIC beta sign and
  off-diagonal constitutive are a different product entirely; :data:`BACKWARD` is False
  and never a parameter, and :data:`SLOT` is ``step_B``;
* ``has_beta=False`` — not a parameter. The identity arm belongs to
  :func:`.folded_beta.folded_beta_bloch_curl_source`'s own gate, and a ``has_beta=0``
  weld would be the shipped folded complex weld under a second name;
* a folded axis carrying a Bloch phase — refused by the certified emitter itself
  (``folded_beta.folded_beta_bloch_curl_source``), and not re-checked here: one raise,
  in the emitter that owns the rule;
* a susceptibility, a conductivity, a nonlinearity, BFAST, cylindrical coordinates, and
  an off-diagonal ``chi1inv`` row on the E side — all through the two halves' own
  predicates, which are conjoined here unweakened;
* a stored extent too shallow for the near fill's source plane, a reflect row outside
  the stored extent, and an axis reported both folded and walled — restated on this
  family because its carry writes destination cells from a DIFFERENT thread, where a
  violated row is an out-of-range write rather than a soft error.

===========================================================================
NOT WIRED
===========================================================================

``register_arms`` registers this product with ``wired=False``, as every other Metal
fused pair is. ``arms.arms_for`` skips an unwired arm, so ``plan_step`` cannot select it
and no existing arm's selection changes — in particular the wired ``folded beta
complex`` curl and constitutive arms, which admit exactly the configurations this weld
admits and would leave both slots UNSELECTED if this contended for them. Registering
unwired keeps it ENUMERABLE for the disjointness sweep (``arms.registered``). The seam
instances this family reaches are a PREDICATE VERDICT; the number actually stepped by a
fused kernel in production remains zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from ..triton_kernels.launch import SUB_STEPS
from ..triton_kernels.symmetry import (
    CODE_MIRROR_PERIODIC, CODE_PERIODIC, folded_axis_kinds,
)
from . import folded_beta, shaders, special_kz, templates
from .complex_fused_magnetic_pair import PACKED_BINDINGS
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .folded_complex import PARITY_PROBE_PATTERNS
from .folded_complex_fused_magnetic_pair import (
    NEAR_SOURCE_INDEX,
    _COORDINATE,
    _LAST_FLAG,
    _RECURRENCE_MARK,
    _body_of,
    _carry_blocks,
    _constitutive_block,
    _constitutive_coefficient_lines,
    _far_carry_reasons,
    _line_starting,
    _mirror_phase_reasons,
    constitutive_transcription,
    far_fill_axes,
    far_parity_words,
    near_fill_axes,
    near_fill_transcription,
    far_fill_transcription,
    parity_words,
    zero_metal_lines,
)
from .plans import KernelPlan
from .symmetry import MIRROR_CODES
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it does on the
#: two welds this one is made of. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
#:
#: TRUE SINCE 2026-08-30, in the same edit as the ``launch.FUSED_PAIR_ARMS`` row that
#: lets the seam loop bracket this family (the ``('folded beta complex', 'folded beta
#: complex')`` entry). Until that row existed this family had NO route into
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

FAMILY = "folded_beta_complex_fused_magnetic_pair"

#: The sub-step slot this arm holds a row on. It spans five; a refusal is NAMED on the
#: slot the fusion starts at rather than being invisible to the table.
SLOT = "step_B"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
#: ``BACKWARD`` is the direction ``SUB_STEPS[CURL_SUB_STEP]`` declares, restated as a
#: bool so the one legal binding is visible without importing :mod:`launch`.
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"
BACKWARD = False

#: Never a parameter. A ``has_beta=False`` weld would be
#: :mod:`.folded_complex_fused_magnetic_pair` under a second name, and every device leg
#: would still pass because beta = 0 IS that family's arithmetic.
HAS_BETA = True

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3288-3296) and in :data:`.coverage.RESIDENCY_ORDER`'s spelling. Read by
#: the whole-step gate's ``covered_passes``; declared, never inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_B", "fill_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: What the same 27 pointers need with the BETA PARENT's non-pointer arguments bound
#: separately, the way ``special_kz._BETA_BLOCH_CURL_TEMPLATE`` binds them:
#: 27 + 4 uint + 1 ``dtdx`` + 6 phase floats + 4 beta floats. Over the ceiling by
#: twelve; :func:`refuted_separate_scalar_source` builds it and a gate requires the
#: compile failure. Spelled here rather than imported because the beta parent's scalar
#: shape is NOT the plain complex parent's -- six ``constant float&`` phase words in
#: place of three ``constant float2&``, plus four words this family's twin has no
#: analogue for.
SEPARATE_SCALAR_BINDINGS = 42

#: The packed struct's size in bytes. ELEVEN ``float2`` members (three Bloch phases,
#: three NEAR mirror parities, three FAR ones, and the TWO beta coefficients), then four
#: uints, one float and three ints: 88 + 16 + 4 + 12 = 120, already a multiple of the
#: strictest member's 8-byte alignment so Metal adds no tail padding. Stated rather than
#: derived because a record that agreed with Metal only by accident is the failure mode
#: this constant exists for -- :func:`.complex_fused_magnetic_pair`'s own measurement is
#: the reason the ``float2`` members come FIRST, and a gate leg must LAUNCH the struct
#: and read every field back rather than trusting the arithmetic.
PARAMS_ITEMSIZE = 120

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the folded beta complex fused magnetic pair binds {PACKED_BINDINGS} buffers "
        f"and Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}; the signature "
        f"cannot be built at all")

if SEPARATE_SCALAR_BINDINGS <= MAX_BUFFER_BINDINGS:  # pragma: no cover - an invariant
    raise RuntimeError(
        f"the refuted separate-scalar signature binds {SEPARATE_SCALAR_BINDINGS} "
        f"buffers, which is NOT over Metal's {MAX_BUFFER_BINDINGS}; the packing this "
        f"family ships would then be a preference rather than a forced choice, and the "
        f"gate leg that requires the compile failure would be measuring nothing")

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY", "HAS_BETA",
    "NEAR_SOURCE_INDEX", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalFoldedBetaComplexFusedMagneticPairPlan",
    "beta_words", "certified_curl_head", "certified_curl_statements",
    "compile_folded_beta_complex_fused_magnetic_pair", "expansion_from_probes",
    "folded_beta_complex_fused_magnetic_pair_source",
    "metal_folded_beta_complex_fused_magnetic_pair_coverage", "params_record_dtype",
    "plan_metal_folded_beta_complex_fused_magnetic_pair",
    "refuted_separate_scalar_source", "signature_binding_census",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
#
# TWENTY-EIGHT BINDINGS: 15 float2 volumes + 12 float32 coefficient vectors + 1 packed
# Params&. The coefficients stay float32 under complex storage (stepping.py:41-50),
# which is what keeps this at twelve buffers rather than twenty-four and the whole
# product buildable at all.

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE ELEVEN float2 MEMBERS COME FIRST, for the reason `complex_fused_magnetic_pair`
// measured on this toolchain: Metal aligns float2 to 8 bytes, so with the scalars first
// the struct needs internal padding and the natural host record puts every phase one
// word early -- which reads as a plausible complex number rather than as garbage.
// Phases, parities and beta words first, no internal padding.
//
// `px`/`py`/`pz` are the Bloch phases, `m0`/`m1`/`m2` the NEAR mirror parity on axes
// x/y/z and `d0`/`d1`/`d2` the FAR one, all host-rounded and passed. `bp`/`bm` are the
// two beta coefficients at each sign -- `2*pi*beta*dt` multiplied by `+1j` (magnetic)
// through Python's own complex arithmetic and rounded ONCE by
// `special_kz.beta_curl_coefficients` (stepping.py:770-784), which is what puts the
// SIGNED zero in the real word and why that word is passed through rather than
// synthesised here.
//
// SIXTEEN SCALAR-SHAPED VALUES, ONE BINDING, AND THE BINDING COUNT IS UNCHANGED AT 28.
// The two beta words ride inside this record rather than as four more buffers, so the
// beta term costs nothing against the platform's 31-binding ceiling (device.py:72) --
// which matters here more than on the real seam, because complex storage already spent
// the headroom on float2 volumes.
struct Params {
    float2 px; float2 py; float2 pz;
    float2 m0; float2 m1; float2 m2;
    float2 d0; float2 d1; float2 d2;
    float2 bp; float2 bm;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
    int rx; int ry; int rz;
};

kernel void folded_beta_complex_fused_magnetic_pair_step(
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
    // THE PACKED ARGUMENTS ARE UNPACKED INTO THE LIFTED BODIES' OWN NAMES, once, before
    // any of the spliced text runs. Everything below this line is then
    // character-for-character what the folded BETA Bloch curl emitter and the certified
    // complex H constitutive emitter produce, plus the wall clear and the two fills.
    //
    // THE PHASE AND BETA WORDS ARE UNPACKED AS SEPARATE FLOATS, not as float2. That is
    // the beta parent's own spelling -- `special_kz._BETA_BLOCH_CURL_TEMPLATE` binds
    // `constant float& pxr/pxi/.../bpr/bpi/bmr/bmi` and its emitted lines read
    // `c_mul(b_x, float2(pxr, pxi))` and `c_mul(float2(bpr, bpi), b)` -- so the lift
    // gets the names it wrote. They are STORED as float2 members because the struct's
    // no-internal-padding property is what `complex_fused_magnetic_pair` measured, and
    // splitting them into eight loose floats would put `dtdx` back in front of a float2.
    //
    // The parity registers are named `mp0`/`mp1`/`mp2` (near) and `fp0`/`fp1`/`fp2`
    // (far) and NOT `c`: the lifted curl head already declares `float2 c = g2[ii];`,
    // and the certified fill's own spelling of the coefficient is `c`.
    // `folded_complex_fused_magnetic_pair`'s carry emitter is CALLED here, so these are
    // exactly the register names it writes -- a rename would be a silent miscompile.
    // `n0`/`n1`/`n2`, `p*`, `q*`, `r*` and `v*` are NOT available as parity names: the
    // lifted split-field recurrence owns all five.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float pxr = prm.px.x, pxi = prm.px.y;
    float pyr = prm.py.x, pyi = prm.py.y;
    float pzr = prm.pz.x, pzi = prm.pz.y;
    float bpr = prm.bp.x, bpi = prm.bp.y;
    float bmr = prm.bm.x, bmi = prm.bm.y;
    float2 mp0 = prm.m0, mp1 = prm.m1, mp2 = prm.m2;
    float2 fp0 = prm.d0, fp1 = prm.d1, fp2 = prm.d2;
    int reflect_x = prm.rx, reflect_y = prm.ry, reflect_z = prm.rz;
    (void)fp0; (void)fp1; (void)fp2;
    (void)reflect_x; (void)reflect_y; (void)reflect_z;

__BODY__
}
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 770-784->797-811

#: The two statements ``special_kz._COMPLEX_BETA_INSERT`` contributes to the lifted
#: head, as it spells them (special_kz.py:562-564). Checked present on every build: a
#: lift that silently produced the ``has_beta=0`` arm would be the shipped folded
#: complex weld under this family's name, and every device leg would still pass because
#: beta = 0 IS that weld's arithmetic. This is
#: :func:`.beta_fused_magnetic_pair.beta_fused_magnetic_pair_source`'s guard, restated
#: for the complex insert.
_BETA_STATEMENTS: Tuple[str, ...] = (
    "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);",
    "    curl1 = curl1 - c_mul(float2(bmr, bmi), a);",
)


def certified_curl_head(codes: Sequence[int], phased: Sequence[int], expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The folded BETA Bloch ``step_B`` curl body DOWN TO the split-field recurrence.

    Not a transcription: this is :func:`.folded_beta.folded_beta_bloch_curl_source`'s
    own output with the ``#include``/helpers/signature preamble removed and the cut
    taken at :data:`_RECURRENCE_MARK` -- the SAME cut point
    :mod:`.folded_complex_fused_magnetic_pair` takes, imported from it rather than
    re-spelled, because the beta template opens its split-field block with the same
    comment line. It carries the guard, the index decode, the ghost gather, the
    wrapped-lane predicates, the Bloch rotation, the curl grouping, THE BETA INSERT, the
    cell-0 ownership mask and the folded top-plane mask.

    ``backward`` is :data:`BACKWARD` and ``has_beta`` is :data:`HAS_BETA`; neither is a
    parameter. ``step_D``'s negated strides, conjugated phase and ELECTRIC beta sign are
    a different product with a different fill geometry entirely, and a ``has_beta=0``
    build is the shipped folded complex weld.
    """
    body = _body_of(
        folded_beta.folded_beta_bloch_curl_source(
            codes, BACKWARD, phased, expansion, HAS_BETA, contract),
        "folded beta complex curl")
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(_RECURRENCE_MARK):
            head = "".join(lines[:index])
            if not head.strip():
                raise AssertionError(
                    "the lifted folded beta complex curl head is empty")
            return head
    raise AssertionError(
        f"the certified folded beta complex curl body no longer opens its split-field "
        f"block with {_RECURRENCE_MARK!r}; this family cuts the lift there")


def certified_curl_statements(codes: Sequence[int], phased: Sequence[int],
                              expansion: str,
                              contract: str = shaders.CONTRACT_OFF,
                              ) -> Dict[str, Any]:
    """The split-field recurrence's own lines, pulled out of the certified beta body.

    Every arithmetic string this family emits below the cut comes from HERE rather than
    from a keyboard: the coefficient loads, the FIVE ``p``/``q``/``n``/``r``/``v``
    statements per target and the two store lines are the certified folded beta Bloch
    curl's, matched by an anchor that must identify exactly one line each.

    **FIVE, WHERE THE SHIPPED WELD LIFTS THREE, AND THAT IS THE WHOLE DELTA OF THIS
    FUNCTION.** ``complex_fields._CURL_TEMPLATE`` nests the intermediates
    (``n0 = c_mul_field_left(c_mul_field_left(p0, km_y) - curl0, si_y);``) so
    :func:`.folded_complex_fused_magnetic_pair.certified_curl_statements` anchors three
    lines; ``special_kz._BETA_BLOCH_CURL_TEMPLATE`` NAMES them (``q0`` and ``r0``) so the
    same arithmetic is five lines. Anchoring only the three the twin anchors would
    silently DROP ``q`` and ``r`` and emit a kernel that does not compile at best and is
    a different recurrence at worst, so the two extra anchors are required to match
    exactly one line each, like every other one.

    ``stores`` is the ``f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;`` triple SPLIT into its
    three statements, because this family emits them one per component inside that
    component's ownership block rather than as one line.
    """
    body = _body_of(
        folded_beta.folded_beta_bloch_curl_source(
            codes, BACKWARD, phased, expansion, HAS_BETA, contract),
        "folded beta complex curl")
    coefficients = [_line_starting(body, f"    float km_{axis} = ",
                                   f"the {axis} split-field coefficient load")
                    for axis in "xyz"]
    recurrence = []
    for target in range(3):
        recurrence.append((
            _line_starting(body, f"    float2 p{target} = ",
                           f"target {target}'s previous split field"),
            _line_starting(body, f"    float2 q{target} = ",
                           f"target {target}'s damped-minus-curl intermediate"),
            _line_starting(body, f"    float2 n{target} = ",
                           f"target {target}'s split-field recurrence"),
            _line_starting(body, f"    float2 r{target} = ",
                           f"target {target}'s pre-scaling displacement"),
            _line_starting(body, f"    float2 v{target} = ",
                           f"target {target}'s stepped displacement"),
        ))
    aux_store = _line_starting(body, "    u0[ii] = ", "the three fu stores")
    flux_store = _line_starting(body, "    f0[ii] = ", "the three flux stores")
    stores = tuple(f"{part.strip()};" for part in flux_store.strip().split(";")
                   if part.strip())
    if len(stores) != 3:
        raise AssertionError(
            f"the certified folded beta complex curl no longer stores three targets on "
            f"one line: {flux_store!r}")
    return {"coefficients": tuple(coefficients), "recurrence": tuple(recurrence),
            "aux_store": aux_store, "stores": stores}


def folded_beta_complex_fused_magnetic_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded quadruple, Bloch flags, walls, arm, mode).

    THE SPLICE IS :func:`.folded_complex_fused_magnetic_pair.folded_complex_fused_magnetic_pair_source`'S,
    step for step, with :func:`certified_curl_head` / :func:`certified_curl_statements`
    where its own stand and with the two extra recurrence statements placed at the two
    points the three-statement version already splits at. Every carry below the cut is
    that module's OWN emitter, called.

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC / MIRROR_PERIODIC
    quadruple :func:`.symmetry.folded_axis_kinds` resolves -- never a hand-built triple,
    because the MIRROR_METALLIC / MIRROR_PERIODIC split is this family's single point of
    failure and getting it backwards on one axis is a plane of wrong values rather than
    a crash.

    ``phased`` is the per-axis Bloch flag. THE PARITY IS NOT HERE: like the shipped
    folded complex weld and unlike the real fold, it is a runtime word, so it does not
    specialise the source and two grids differing only in a mirror parity compile to the
    SAME kernel. THE BETA COEFFICIENTS ARE NOT HERE EITHER, and that is
    :mod:`.folded_beta`'s own measurement (leg 6, 2026-08-16): a bound uniform and the
    same value baked into the source as a decimal literal produced IDENTICAL words --
    0/144 at four coefficients under both contraction modes -- so specialising them
    would buy a source variant per beta value and change nothing.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if not any(code in MIRROR_CODES for code in codes):
        raise ValueError(
            "no axis is folded: this product exists to carry the mirror fills inside "
            "the B seam of a BETA run, and an unfolded complex beta grid's B/H seam is "
            "special_kz's certified Bloch beta curl plus its complex constitutive "
            "companion, over which no weld is built")
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
    # (folded_beta.folded_beta_bloch_curl_source). It is not re-checked here: one raise,
    # in the emitter that owns the rule.

    transcription = constitutive_transcription(expansion, contract)
    if not transcription["identical"]:
        raise AssertionError(
            "this family's parameterised constitutive statements no longer reproduce "
            "complex_fields.bloch_constitutive_source('H'); the fused arithmetic would "
            "silently stop being the certified arithmetic. emitted="
            f"{transcription['emitted']!r} certified={transcription['certified']!r}")

    head = certified_curl_head(codes, phased, expansion, contract)
    # THE BETA TERM MUST BE PRESENT, and it must be present IN THE LIFTED HEAD -- the
    # insert sits between the dtdx curl and BOTH masks (folded_beta.py:22-30), which is
    # above the cut. `beta_fused_magnetic_pair`'s guard, restated for the complex
    # insert's own spelling.
    for statement in _BETA_STATEMENTS:
        if statement not in head:
            raise AssertionError(
                f"the lifted folded beta complex curl head does not carry "
                f"{statement.strip()!r}; this weld would be "
                f"folded_complex_fused_magnetic_pair under another name, and every "
                f"device leg would still pass because beta = 0 IS that family's "
                f"arithmetic")

    statements = certified_curl_statements(codes, phased, expansion, contract)
    body: List[str] = [head.rstrip("\n")]
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
        "    // FIVE statements per target here, not three: the beta parent NAMES the",
        "    // intermediates (q, r) that complex_fields nests. Same arithmetic, same",
        "    // order; every line below is _line_starting's answer off that body.",
    ])
    body.extend(statements["coefficients"])
    body.extend([
        "",
        "    // --- the constitutive coefficient, on the component's OWN axis -----",
        "    // Read HERE for the owned cell only. On this family the imaged ghost",
        "    // cell does NOT share it: the near fill images along the component's",
        "    // own axis, which is the axis this is indexed on.",
    ])
    body.extend(_constitutive_coefficient_lines(expansion, contract))
    body.append("")

    # The three recurrences and the three `fu` stores, in the certified body's own
    # order. `u` IS WRITTEN AT EVERY CELL, ghost cells included: `step_B` updates `fu`
    # everywhere and the fills touch B only (stepping.py:1451 writes `field`, never
    # `fu_field`).
    #
    # `q` RIDES WITH `p`, ABOVE THE STORE, and `r` RIDES WITH `v`, BELOW THE GUARD. The
    # split point is the twin's -- everything the `fu` store needs is emitted before it,
    # everything only the flux store needs is emitted inside the ownership guard -- and
    # the two named intermediates fall on the two sides by what they read: `q` feeds `n`
    # (which the `fu` store writes), `r` feeds `v` (which the flux store writes).
    for target in range(3):
        previous, damped, recurrence, _, _ = statements["recurrence"][target]
        body.extend([previous, damped, recurrence])
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
                f"component {target} is a far-fill destination on {far} against a near "
                f"axis {near}; for the B family the two sets are COMPLEMENTARY (iyee is "
                f"0 on the component's own axis and 1 on the other two, "
                f"fields.IYEE_SHIFTS), so far holds at most two axes and never the near "
                f"one")
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
        _, _, _, pre_scaling, displacement = statements["recurrence"][target]
        body.append("")
        # A cell any fill writes is OWNED BY ITS SOURCE THREAD. This thread stops after
        # fu there: the array path's fill overwrites the displacement it would store,
        # and forming v would read f at a word another thread writes.
        owned: List[str] = [f"{_COORDINATE[axis]} == 0" for axis in near]
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
        body.append(indent + pre_scaling.strip())
        body.append(indent + displacement.strip())
        # `zero_metal_B` on the cell this thread owns, then the store, then the
        # constitutive read of the SAME register -- the seam this family closes.
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


def compile_folded_beta_complex_fused_magnetic_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, Bloch flags, walls, arm, mode)."""
    return compile_source(folded_beta_complex_fused_magnetic_pair_source(
        codes, phased, zero_metal, expansion,
        contract)).folded_beta_complex_fused_magnetic_pair_step


# ---------------------------------------------------------------------------
# The binding arithmetic, measured off the EMITTED TEXT
# ---------------------------------------------------------------------------

def signature_binding_census(source: Optional[str] = None) -> Dict[str, int]:
    """Count the ``[[buffer(n)]]`` attributes in an emitted source and check them.

    THE POINT IS THAT THE NUMBER IS READ BACK OUT OF THE STRING. :data:`PACKED_BINDINGS`
    is imported from a sibling and :data:`PARAMS_ITEMSIZE` is spelled by hand; a
    signature that drifted from either would otherwise be discovered on a device, or
    not at all. This counts what the emitter actually wrote, requires the attribute
    indices to be the DENSE range ``0..n-1`` (a gap would mean a buffer nobody binds and
    a plan whose argument tuple is silently off by one), and requires exactly one of
    them to be the packed ``constant Params&``.

    ``source`` defaults to a representative specialisation -- a Bloch phase on the
    unfolded x axis with the fold on y, which is the shape both reachable corpus rows
    carry -- so the census can be taken without a caller choosing a configuration.
    """
    import re  # noqa: PLC0415

    if source is None:
        source = folded_beta_complex_fused_magnetic_pair_source(
            (CODE_PERIODIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC), (1, 0, 0),
            (False, False, False), "FMA_V1")
    signature = source.split("uint idx [[thread_position_in_grid]])", 1)[0]
    indices = sorted(int(value) for value in
                     re.findall(r"\[\[buffer\((\d+)\)\]\]", signature))
    if indices != list(range(len(indices))):
        raise AssertionError(
            f"the emitted signature's buffer attributes are not the dense range "
            f"0..{len(indices) - 1}: {indices!r}. A gap is a buffer nobody binds and a "
            f"plan whose argument tuple is off by one")
    pointers = signature.count("device float2*") + signature.count("device const float2*") \
        + signature.count("device const float*")
    packed = signature.count("constant Params&")
    if packed != 1:
        raise AssertionError(
            f"the emitted signature carries {packed} packed Params& bindings, not one; "
            f"the whole 42 -> 28 argument is that ALL the scalars ride in exactly one")
    if len(indices) != PACKED_BINDINGS:
        raise AssertionError(
            f"the emitted signature binds {len(indices)} buffers and PACKED_BINDINGS "
            f"says {PACKED_BINDINGS}")
    if pointers + packed != PACKED_BINDINGS:
        raise AssertionError(
            f"{pointers} pointers + {packed} packed struct != {PACKED_BINDINGS}")
    if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - checked at import
        raise AssertionError((PACKED_BINDINGS, MAX_BUFFER_BINDINGS))
    if SEPARATE_SCALAR_BINDINGS <= MAX_BUFFER_BINDINGS:  # pragma: no cover - at import
        raise AssertionError((SEPARATE_SCALAR_BINDINGS, MAX_BUFFER_BINDINGS))
    return {"bindings": len(indices), "pointers": pointers, "packed": packed,
            "ceiling": MAX_BUFFER_BINDINGS,
            "refuted_separate_scalar": SEPARATE_SCALAR_BINDINGS}


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 42-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so a gate can compile it and require the
    failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from an argument into
    a measurement. This is the signature ``special_kz._BETA_BLOCH_CURL_TEMPLATE`` uses,
    scaled to this pair's fifteen volumes -- so what it measures is precisely "the
    packing is FORCED by the six extra volumes the fusion brings AND by the beta
    parent's six-float phase spelling", not that separate scalars are bad style. Seven
    bindings wider than :func:`.complex_fused_magnetic_pair.refuted_separate_scalar_source`'s
    35, which is exactly the three float2 phases becoming six floats plus the four beta
    words.

    The body touches every buffer -- what is being measured is the SIGNATURE, and a body
    the compiler could drop would let dead-code elimination decide the answer.
    """
    written = ["f0", "f1", "f2", "u0", "u1", "u2", "h0", "h1", "h2", "w0", "w1", "w2"]
    read_complex = ["g0", "g1", "g2"]
    read_real = ["kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                 "kp0", "km0", "kp1", "km1", "kp2", "km2"]
    scalars = ("dtdx", "pxr", "pxi", "pyr", "pyi", "pzr", "pzi",
               "bpr", "bpi", "bmr", "bmi")
    lines: List[str] = []
    slot = 0
    for name in written:
        lines.append(f"    device float2*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in read_complex:
        lines.append(f"    device const float2* {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in read_real:
        lines.append(f"    device const float*  {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("nx", "ny", "nz", "n_elem"):
        lines.append(f"    constant uint&       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in scalars:
        lines.append(f"    constant float&      {name:<8}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - a construction invariant
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    body = "\n".join(f"    {name}[idx] = {name}[idx] * dtdx;" for name in written)
    touch_real = " + ".join(f"{name}[0]" for name in read_real)
    touch_complex = " + ".join(f"{name}[0].x" for name in read_complex)
    touch_scalar = " + ".join(scalars)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalar(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = ({touch_real}) + ({touch_complex}) + ({touch_scalar});",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz);",
        body,
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# The expansion arm — BOTH probes, and they must agree
# ---------------------------------------------------------------------------

def expansion_from_probes(parity_record: Any, beta_record: Any) -> Optional[str]:
    """The one expansion arm both parents' artifacts license, or ``None``.

    ONE ``templates.complex_helpers(expansion)`` EMISSION SERVES THE WHOLE KERNEL, and
    this kernel carries complex products from both parents' call sites. Neither pattern
    list is a superset of the other -- ``folded_complex.PARITY_PROBE_PATTERNS`` carries
    ``c8_mul_c8_scalar_right`` and the ±1 parity orientation,
    ``special_kz.BETA_PROBE_PATTERNS`` carries the purely-imaginary coefficient
    orientation -- so each record is licensed by ITS OWN parent's rule (which is where
    the ``NEITHER`` refusal and the all-ambiguous refusal live) and the two verdicts
    must then be the same word.

    ``None`` on any of: a missing record, a record its own parent refuses, or two
    records that disagree. A disagreement is a real answer and not a defect: it would
    mean the two orientations are reproduced by DIFFERENT transcriptions on this host,
    and a kernel that must pick one would be wrong on the other. Refusing is the only
    correct move, and inheriting one verdict for both would be exactly the guess the
    probe mechanism exists to prevent.
    """
    from .folded_complex import expansion_from_probe  # noqa: PLC0415

    parity = expansion_from_probe(parity_record)
    beta = special_kz.beta_expansion_from_probe(beta_record)
    if parity is None or beta is None or parity != beta:
        return None
    return parity


def _expansion_reasons(parity_record: Any, beta_record: Any) -> List[str]:
    """Why the arm cannot be bound, named rather than defaulted.

    EACH RECORD FALLS BACK TO ITS OWN LOADER WHEN NOT THREADED, which is what
    :func:`.folded_complex._expansion_reasons` does with its one artifact and what
    :func:`plan_metal_folded_beta_complex_fused_magnetic_pair` already does with both
    three lines after it calls this predicate. Without the fallback the two answers
    DISAGREE whenever this family is reached through the arm table rather than from
    its gate: ``ArmSpec.coverage`` threads no probe, so coverage refused "no probe
    licenses an EXPANSION arm" on a host where the plan builder would have loaded
    one from the environment and built the kernel. That disagreement is the reason
    ``launch._install_fused_pairs`` could never bracket this family's seam, measured
    on ``metal_composition_matrix``'s ``fold_complex_2d_beta`` row.

    NOTHING IS WIDENED BY THE FALLBACK. A missing or unclassifiable artifact still
    refuses BY NAME, and the artifact consulted here is the same one the builder
    consults, so the predicate and the plan can no longer answer differently.
    """
    from .folded_complex import expansion_from_probe  # noqa: PLC0415
    from .folded_complex import load_expansion_probe  # noqa: PLC0415

    reasons: List[str] = []
    parity_record = (parity_record if parity_record is not None
                     else load_expansion_probe())
    beta_record = (beta_record if beta_record is not None
                   else special_kz.load_expansion_probe())
    parity = expansion_from_probe(parity_record)
    beta = special_kz.beta_expansion_from_probe(beta_record)
    if parity is None:
        reasons.append(
            f"no folded complex parity probe licenses an EXPANSION arm (set "
            f"{PARITY_PROBE_PATTERNS!r}'s artifact through folded_complex's own "
            f"environment variable); this kernel emits the mirror fill's parity "
            f"product and the arm is a MEASURED platform fact, never a default")
    if beta is None:
        reasons.append(
            "no special_kz beta probe licenses an EXPANSION arm; this kernel emits the "
            "beta insert's imaginary-coefficient product and that orientation is not "
            "classified by the folded complex artifact")
    if parity is not None and beta is not None and parity != beta:
        reasons.append(
            f"the two probes disagree: the parity artifact licenses {parity!r} and the "
            f"beta artifact {beta!r}. One helpers emission serves the whole kernel, so "
            f"a disagreement means no single transcription reproduces both call sites "
            f"on this host")
    return reasons


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_beta_complex_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None, beta_probe: Any = None) -> Coverage:
    """May ONE dispatch span folded beta complex ``step_B`` -> fills -> wall -> ``update_H``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. That is the
    construction every shipped fused pair uses; the halves are
    :mod:`.folded_beta`'s ROUTING verdicts (a fold MANDATORY), which is what keeps this
    product disjoint from :mod:`.beta_fused_magnetic_pair` and from
    :mod:`.folded_complex_fused_magnetic_pair` in both directions.

    ``probe`` and ``beta_probe`` are the two expansion artifacts. They are threaded, not
    re-read inside the halves: a second read would be a second answer to one question.
    """
    reasons: List[str] = []

    curl = folded_beta.folded_beta_composition_bloch_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency, beta_probe)
    if not curl.covered:
        reasons.extend(f"folded beta complex curl half: {reason}"
                       for reason in curl.reasons)
    magnetic = folded_beta.folded_beta_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency, beta_probe)
    if not magnetic.covered:
        reasons.extend(f"folded beta complex constitutive half: {reason}"
                       for reason in magnetic.reasons)

    # THE PARITY ARM IS THIS PRODUCT'S OWN QUESTION and neither half asks it: the two
    # halves are folded_beta's, which never emits a mirror-parity product because that
    # family registers NO fill arm. The fill arithmetic this weld carries inline is
    # folded_complex's, so folded_complex's artifact is required too.
    reasons.extend(_expansion_reasons(probe, beta_probe))

    # THE SOURCE SEAM. A MAGNETIC source is injected BETWEEN the two halves
    # (driver.py:3292-3293), so a fused pair would consume a pre-injection B.
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
            f"source {index} ({type(source).__name__}) is magnetic: the driver injects "
            f"it BETWEEN step_B and update_H (driver.py:3292-3293), which is work "
            f"inside the seam this kernel closes"),
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
        # curl half's predicate already refuses a folded axis storing too few cells;
        # restated because THIS family writes the destination from a DIFFERENT thread
        # and a missing source plane is an out-of-range write, not a soft error.
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; the "
                    f"near fill images stored cell {NEAR_SOURCE_INDEX} and this kernel "
                    f"images it from that cell's own thread")

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
                    f"stepping._zero_metal skips a folded axis, so the two passes have "
                    f"drifted and the carry's disjointness no longer holds")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedBetaComplexFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs five driver passes on a folded complex beta run.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no special
    case for it.

    ``runs`` is an alias of ``launches`` rather than a second counter, because with one
    dispatch per run the two ARE the same number.
    """

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values", "phases",
                 "parity_values", "far_parity_values", "beta_values", "reflect",
                 "zero_metal", "shape", "dtdx", "expansion", "params", "carried_axes",
                 "far_carried_axes")

    family = "folded beta complex fused B-curl/mirror-fill/stored-H pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phased", "phases", "zero_metal", "beta_values",
                   "carried_axes", "far_carried_axes", "reflect", "expansion")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 phases: Sequence[int],
                 parity_values: Sequence[Tuple[float, float]],
                 far_parity_values: Sequence[Tuple[float, float]],
                 beta_values: Sequence[Tuple[float, float]],
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
        # float(): `beta_curl_coefficients` already rounded ONCE to complex64 and
        # float() of a numpy.float32 preserves the word, the sign of a zero included.
        # Rounding a second time here would be a second rounding, which is exactly what
        # that function's docstring measures 14,926 of 40,000 draws apart.
        self.beta_values = tuple((float(re), float(im)) for re, im in beta_values)
        self.reflect = tuple(None if value is None else int(value)
                             for value in reflect)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
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
    """The host record's dtype — ELEVEN ``float2`` members first, itemsize 120.

    ONE HOME FOR THE LAYOUT, because getting it wrong is a silent wrong answer rather
    than a crash: see :mod:`.complex_fused_magnetic_pair`'s own record for the
    measurement behind the ordering (a scalars-first struct puts every phase one word
    early and reads as a plausible complex number). The offsets are stated EXPLICITLY
    even though the complex-members-first order makes them the natural ones, so the
    record cannot drift into agreeing with Metal only by accident; ``itemsize`` is
    stated because a record whose size the toolchain chose is a record nobody checked.
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["px", "py", "pz", "m0", "m1", "m2", "d0", "d1", "d2", "bp", "bm",
                  "nx", "ny", "nz", "n_elem", "dtdx", "rx", "ry", "rz"],
        "formats": [("<f4", 2), ("<f4", 2), ("<f4", 2),
                    ("<f4", 2), ("<f4", 2), ("<f4", 2),
                    ("<f4", 2), ("<f4", 2), ("<f4", 2),
                    ("<f4", 2), ("<f4", 2),
                    "<u4", "<u4", "<u4", "<u4", "<f4", "<i4", "<i4", "<i4"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80,
                    88, 92, 96, 100, 104, 108, 112, 116],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]],
                   parity_values: Sequence[Tuple[float, float]],
                   far_parity_values: Sequence[Tuple[float, float]],
                   beta_values: Sequence[Tuple[float, float]],
                   reflect: Sequence[Optional[int]],
                   device: str) -> Any:
    """The scalars, phases, BOTH parity triples, the beta pair and the rows, packed.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every other packed
    struct in this package gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME, because a wider element silently
    reinterprets. This record is neither, so it is built here, once, at plan time, and
    held by the plan. Nothing on the launch path allocates.

    ``-1`` stands for an axis with no reflect row (unfolded, or folded METALLIC, where
    the stored array stops at ``big_corner`` and no far ghost exists). It is a value the
    kernel never reads: the guard that would read it is emitted only for an axis
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
    beta = tuple(beta_values)
    if len(beta) != 2:
        raise ValueError(
            f"beta_values must be the (plus, minus) pair special_kz."
            f"beta_curl_coefficients returns, got {beta_values!r}")
    for name, (real, imag) in zip(("bp", "bm"), beta):
        record[name] = (np.float32(real), np.float32(imag))
    rows = tuple(-1 if value is None else int(value) for value in reflect)
    if len(rows) != 3:
        raise ValueError(f"reflect must be a per-axis triple, got {reflect!r}")
    record["rx"], record["ry"], record["rz"] = rows
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def beta_words(grid: Any) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """``((bpr, bpi), (bmr, bmi))`` — the CURL ARM'S OWN coefficients, called.

    :func:`.special_kz.beta_curl_coefficients` is stepping.py:797-811 evaluated on the
    objects the array path evaluates it on, so the intermediate precision is the array
    path's too and the words this kernel receives are its bits, the sign of a zero
    included. ``magnetic=True`` because :data:`CURL_SUB_STEP` is ``step_B``;
    ``complex_storage=True`` because this is the complex arm. Neither is a parameter.
    """
    return special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(CURL_SUB_STEP == "step_B"),
        complex_storage=True)


def plan_metal_folded_beta_complex_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None, beta_probe: Any = None,
        ) -> Optional[MetalFoldedBetaComplexFusedMagneticPairPlan]:
    """Build the fused folded beta complex B/H plan, or ``None`` when refused.

    ``None`` is the only refusal, for the reason
    :func:`.folded_beta.plan_folded_beta_bloch_pml_curl` gives: a configuration this
    kernel does not carry must fall back to the array path, never raise into a caller
    that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. A gate compiles a deliberately broken copy of
    the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING -- every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    from .folded_complex import load_expansion_probe  # noqa: PLC0415

    parity_record = probe if probe is not None else load_expansion_probe()
    beta_record = (beta_probe if beta_probe is not None
                   else special_kz.load_expansion_probe())
    if not metal_folded_beta_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, residency, parity_record, beta_record).covered:
        return None
    expansion = expansion_from_probes(parity_record, beta_record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds, _far_reflect_rows  # noqa: PLC0415

    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    kinds = _boundary_kinds(grid, pml)
    walls = zero_metal_axes(grid)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    # THE PHASE FLAGS AND WORDS COME FROM `special_kz.bloch_phase_words`, not from
    # `complex_fields.phase_arguments`. That is the ONE line of this builder the swap
    # moves: the beta template spells its phases as SIX floats and takes the sub-step's
    # conjugation on the host, which is what this helper does and is where
    # `folded_beta`'s own builder gets them. `backward=BACKWARD` -- this is the B seam.
    flags, phase_flat = special_kz.bloch_phase_words(grid, kinds, BACKWARD)
    values = tuple((phase_flat[2 * axis], phase_flat[2 * axis + 1])
                   for axis in range(3))
    parities = parity_words(codes, phases)
    far_parities = far_parity_words(codes, phases)
    beta = beta_words(grid)
    # `stepping._far_reflect_rows` is the ONE derivation of the row a folded PERIODIC
    # axis's far face images; asked here rather than re-derived, and passed as a runtime
    # int because nothing about an integer row can move a bit.
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
    # (SUB_STEPS['step_B']['suffix'] == '_h'; CONSTITUTIVE_SIDES['H']['half_integer'] is
    # False). That is the OPPOSITE pairing to the folded D/E pair, and the kernel cannot
    # tell -- a gate must carry a host mutation for each group.
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
            selected[mode] = compile_folded_beta_complex_fused_magnetic_pair(
                codes, flags, walls, expansion, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor would read
    # "mps:0" where the mirrors were built with "mps" and put the struct on a nominally
    # different device from the volumes it describes.
    dtdx = grid.dt / grid.dx
    return MetalFoldedBetaComplexFusedMagneticPairPlan(
        residency, volumes, codes, flags, values, phases, parities, far_parities,
        beta, reflect, walls, grid.shape, dtdx, expansion,
        _params_tensor(grid.shape, dtdx, values, parities, far_parities, beta,
                       reflect, residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False,
            (f"folded beta complex fused magnetic pair cannot fill {slot}",))
    return metal_folded_beta_complex_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalFoldedBetaComplexFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_beta_complex_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY, and it is every other fused pair's.
    ``plan_step`` assigns at most one arm per slot and this product spans five, so there
    is no slot it could claim without a composition rule nothing has measured. It would
    additionally contend on ``step_B`` with :mod:`.folded_beta`'s wired ``folded beta
    complex`` curl arm, which admits exactly the same configurations, and
    ``_select_slot`` would then leave the slot UNSELECTED -- the certified curl would
    come off the device as well.

    Registering unwired keeps it ENUMERABLE for the disjointness sweep
    (``arms.registered``) while ``arms.arms_for`` skips it, so ``plan_step`` cannot
    select it and no existing arm's selection changes.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "folded beta complex fused B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded beta complex fused B/H pair: ",
                          noun=("folded beta complex PML B-curl/mirror-fill/"
                                "stored-H pair"),
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
