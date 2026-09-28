"""The FOLDED BETA COMPLEX fused D/E pair: a folded ``special_kz`` Bloch ``step_D``
welded into ``update_E``.

:mod:`.folded_complex_fused_pair` WITH ONE EMITTER SWAPPED, and the swap is the whole
module -- the D-side mirror of what :mod:`.folded_beta_complex_fused_magnetic_pair` is
to :mod:`.folded_complex_fused_magnetic_pair` on the other seam. The certified folded
complex Bloch curl body is replaced by the certified FOLDED BETA complex curl body
(:func:`.folded_beta.folded_beta_bloch_curl_source` with ``backward=True`` and
``has_beta=True``) and everything else is IMPORTED from that shipped weld rather than
re-spelled: the near and far carries, the parity chain, the wall clear, the ghost
coefficient reload, the constitutive transcription and the constitutive block emitter
are all :mod:`.folded_complex_fused_pair`'s own functions, called here.

This is the cell ``(folded beta complex, folded beta complex)`` at ``D_to_E``, and it
is the one D/E cell neither shipped folded weld can stand in for.
:mod:`.folded_complex_fused_pair` refuses ``grid.beta`` through both halves' own clause
(``folded_complex._folded_complex_grid_reasons`` clause 8, which names "the folded beta
product" as the owner); :mod:`.beta_fused_electric_pair` refuses a fold by name and is
real-storage besides. This module is the product they name.

===========================================================================
WHAT IT REACHES
===========================================================================

THREE corpus rows, and unlike its magnetic twin ALL THREE are reachable:

* ``tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2``
* ``tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7``
* ``tests:TestSpecialKz.test_eigsrc_kz_0_complex``

The magnetic twin's docstring records that third row as UNREACHABLE ON ANY BACKEND
because it deposits a MAGNETIC source inside the B/H seam. On THIS seam the deposit is
electric and is CARRIED (see :data:`CARRIES_DEPOSIT_REPAIR`), so the row is in scope
here. **THE ROW COUNT IS NOT RE-DERIVED IN THIS MODULE AND IT CLAIMS NO CELL MOVE** --
that is a matrix run, and this module asserts only its predicate and its binding
arithmetic, both measured on every build.

===========================================================================
THE SIGNATURE — 30 pointers plus one packed struct, EXACTLY ON THE CEILING
===========================================================================

:data:`PACKED_BINDINGS` is IMPORTED from :mod:`.folded_complex_fused_pair` rather than
re-spelled, because the beta term adds NO POINTER: its two coefficients are complex
SCALARS and they ride inside the same ``constant Params&`` the phases and parities
already ride in. The struct grows from fifteen scalar-shaped values to seventeen and
the binding count does not move -- 31, which is :data:`.device.MAX_BUFFER_BINDINGS`
exactly. The three inverse-epsilon volumes are what put this family ON the ceiling
where the magnetic twin sits three under it: ``update_H`` is ``H = B`` with mu = 1
baked in, ``update_E`` reads ``D * inverse_epsilon_for(component)``.

``bp``/``bm`` are ``2*pi*beta*dt`` at each sign, multiplied by ``-1j`` (ELECTRIC)
through Python's own complex arithmetic and rounded ONCE by
:func:`.special_kz.beta_curl_coefficients` with ``magnetic=False``. That flag is not a
default: the two seams take OPPOSITE signs, and a weld that reused the magnetic pair's
words would run a correctly-shaped, converged, wrong dispersion.

NOT WIRED AS AN ARM: ``plan_step`` assigns at most one arm per slot and this product
spans five. It registers ``wired=False`` and reaches its two slots through
``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, _call, zero_metal_axes
from ..triton_kernels.symmetry import folded_axis_kinds
from . import folded_beta, shaders, special_kz, templates
from .device import Residency, compile_source
from .folded_complex_fused_magnetic_pair import far_parity_words, parity_words
from .folded_complex_fused_pair import (
    NEAR_SOURCE_INDEX,
    PACKED_BINDINGS,
    _RECURRENCE_MARK,
    _body_of,
    _carry_blocks,
    _COORDINATE,
    _constitutive_block,
    _constitutive_coefficient_lines,
    _LAST_FLAG,
    _line_starting,
    constitutive_transcription,
    far_fill_transcription,
    near_fill_transcription,
    zero_metal_lines,
)
from .folded_fused_pair import (
    _far_carry_reasons,
    _mirror_phase_reasons,
    far_fill_axes,
    near_fill_axes,
)
from .plans import KernelPlan
from .symmetry import MIRROR_CODES
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit. Flipping it is a claim
#: about the PLAN this module builds -- that the leading slot saves and the trailing
#: slot restores -- and is only ever changed in the same edit as that wiring.
#:
#: TRUE FROM THIS FAMILY'S FIRST COMMIT, in the same edit as its
#: ``launch.FUSED_PAIR_ARMS`` row, and it is what makes the cell worth THREE rows
#: rather than none: every corpus row here carries an electric source inside the seam.
#: The machinery is ``deposit_repair.repair_cells``' image closure over a FOLDED seam
#: (deposit_repair.py:217-258) -- the inverse of the forward carry this kernel
#: implements -- and it is storage-agnostic, indexing and accumulating element-wise over
#: one ``complex64`` array per component (fields.py:573). ``repairable`` still refuses
#: BY NAME the folds whose fill map it cannot read: the cylindrical r = 0 axis, and a
#: folded axis storing no more than ``MIRROR_SOURCE_INDEX`` cells.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_beta_complex_fused_pair"

#: The sub-step slot this arm holds a row on. It spans five; a refusal is NAMED on the
#: slot the fusion starts at rather than being invisible to the table.
SLOT = "step_D"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: ``backward`` and ``has_beta`` for the lifted curl. Named rather than passed: a
#: ``has_beta=False`` build is the shipped folded complex D/E weld under this family's
#: name, and ``step_B``'s strides are the magnetic twin.
BACKWARD = True
HAS_BETA = True

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3292-3303).
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The host record's total size in bytes: eleven ``float2`` (88) + four ``uint`` (16) +
#: one ``float`` (4) + three ``int`` (12) = 120.
PARAMS_ITEMSIZE = 120

#: The two statements ``special_kz``'s complex beta insert contributes to the lifted
#: head, as it spells them. Checked present on every build: a lift that silently
#: produced the ``has_beta=0`` arm would be the shipped folded complex D/E weld under
#: this family's name, and every device leg would still pass because beta = 0 IS that
#: weld's arithmetic. The SIGN lives in the host-rounded ``bp``/``bm`` words, not in
#: these lines, which is why the two seams share them.
_BETA_STATEMENTS: Tuple[str, ...] = (
    "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);",
    "    curl1 = curl1 - c_mul(float2(bmr, bmi), a);",
)

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "FAMILY", "HAS_BETA", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES", "SLOT",
    "MetalFoldedBetaComplexFusedPairPlan", "beta_words", "certified_curl_head",
    "certified_curl_statements", "compile_folded_beta_complex_fused_pair",
    "folded_beta_complex_fused_pair_source",
    "metal_folded_beta_complex_fused_pair_coverage", "params_record_dtype",
    "plan_metal_folded_beta_complex_fused_pair", "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE ELEVEN float2 MEMBERS COME FIRST, for the reason `complex_fused_magnetic_pair`
// measured on this toolchain: Metal aligns float2 to 8 bytes, so with the scalars first
// the struct needs internal padding and the natural host record puts every phase one
// word early -- which reads as a plausible complex number rather than as garbage.
//
// `bp`/`bm` are the two beta coefficients at each sign -- `2*pi*beta*dt` multiplied by
// `-1j` (ELECTRIC) through Python's own complex arithmetic and rounded ONCE by
// `special_kz.beta_curl_coefficients` with magnetic=False, which is what puts the
// SIGNED zero in the real word and why that word is passed through rather than
// synthesised here.
//
// SEVENTEEN SCALAR-SHAPED VALUES, ONE BINDING, AND THE BINDING COUNT IS 31 EXACTLY.
// The two beta words ride inside this record rather than as four more buffers, so the
// beta term costs nothing against the platform's ceiling (device.py:72) -- which
// matters here more than on the magnetic seam, because the three inverse-epsilon
// volumes have already spent every spare slot.
struct Params {
    float2 px; float2 py; float2 pz;
    float2 m0; float2 m1; float2 m2;
    float2 d0; float2 d1; float2 d2;
    float2 bp; float2 bm;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
    int rx; int ry; int rz;
};

kernel void folded_beta_complex_fused_pair_step(
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
    device const float*  e0      [[buffer(15)]],
    device const float*  e1      [[buffer(16)]],
    device const float*  e2      [[buffer(17)]],
    device const float*  kmx     [[buffer(18)]],
    device const float*  sinvx   [[buffer(19)]],
    device const float*  kmy     [[buffer(20)]],
    device const float*  sinvy   [[buffer(21)]],
    device const float*  kmz     [[buffer(22)]],
    device const float*  sinvz   [[buffer(23)]],
    device const float*  kp0     [[buffer(24)]],
    device const float*  km0     [[buffer(25)]],
    device const float*  kp1     [[buffer(26)]],
    device const float*  km1     [[buffer(27)]],
    device const float*  kp2     [[buffer(28)]],
    device const float*  km2     [[buffer(29)]],
    constant Params&     prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE PHASE AND BETA WORDS ARE UNPACKED AS SEPARATE FLOATS, not as float2. That is
    // the beta parent's own spelling -- `special_kz._BETA_BLOCH_CURL_TEMPLATE` binds
    // `constant float& pxr/pxi/.../bpr/bpi/bmr/bmi` and its emitted lines read
    // `c_mul(b_x, float2(pxr, pxi))` and `c_mul(float2(bpr, bpi), b)` -- so the lift
    // gets the names it wrote. They are STORED as float2 members because the struct's
    // no-internal-padding property is what `complex_fused_magnetic_pair` measured.
    //
    // The parity registers are named `mp0`/`mp1`/`mp2` (near) and `fp0`/`fp1`/`fp2`
    // (far) and NOT `c`: the lifted curl head already declares `float2 c = g2[ii];`.
    // `folded_complex_fused_pair`'s carry emitter is CALLED here, so these are exactly
    // the register names it writes -- a rename would be a silent miscompile.
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
    (void)mp0; (void)mp1; (void)mp2;
    (void)fp0; (void)fp1; (void)fp2;
    (void)reflect_x; (void)reflect_y; (void)reflect_z;

__BODY__
}
"""


def certified_curl_head(codes: Sequence[int], phased: Sequence[int], expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The folded BETA Bloch ``step_D`` curl body DOWN TO the split-field recurrence.

    Not a transcription: this is :func:`.folded_beta.folded_beta_bloch_curl_source`'s
    own output with the ``#include``/helpers/signature preamble removed and the cut
    taken at :data:`.folded_complex_fused_pair._RECURRENCE_MARK` -- the SAME cut point
    the shipped folded complex D/E weld takes, IMPORTED from it rather than re-spelled,
    because the beta template opens its split-field block with the same comment line.

    ``backward`` is :data:`BACKWARD` and ``has_beta`` is :data:`HAS_BETA`; neither is a
    parameter.
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

    **FIVE STATEMENTS PER TARGET, WHERE THE SHIPPED WELD LIFTS THREE**, and that is the
    whole delta of this function -- the same delta the magnetic twin records.
    ``complex_fields._CURL_TEMPLATE`` NESTS the intermediates so the folded complex weld
    anchors three lines; ``special_kz._BETA_BLOCH_CURL_TEMPLATE`` NAMES them (``q`` and
    ``r``) so the same arithmetic is five. Anchoring only three would silently DROP
    ``q`` and ``r``, so the two extra anchors are required to match exactly one line
    each, like every other one.
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


def folded_beta_complex_fused_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded quadruple, Bloch flags, walls, arm, mode).

    THE SPLICE IS :func:`.folded_complex_fused_pair.folded_complex_fused_pair_source`'S,
    step for step, with :func:`certified_curl_head` / :func:`certified_curl_statements`
    where its own stand and with the two extra recurrence statements placed at the two
    points the three-statement version already splits at. Every carry below the cut is
    that module's OWN emitter, called.

    ``q`` RIDES WITH ``p``, ABOVE THE ``fu`` STORE, and ``r`` RIDES WITH ``v``, BELOW
    THE OWNERSHIP GUARD. The split point is the shipped weld's -- everything the ``fu``
    store needs is emitted before it, everything only the flux store needs inside the
    guard -- and the two named intermediates fall on the two sides by what they read.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if not any(code in MIRROR_CODES for code in codes):
        raise ValueError(
            "no axis is folded: this product exists to carry the mirror fills inside "
            "the D seam of a BETA run, and an unfolded complex beta grid's D/E seam is "
            "special_kz's certified Bloch beta curl plus its complex constitutive "
            "companion, over which no weld is built")
    phased = tuple(int(flag) for flag in phased)
    if len(phased) != 3:
        raise ValueError(f"phased must be a per-axis triple, got {phased!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    # THE DISJOINTNESS THE CARRY RESTS ON, asserted rather than trusted.
    for axis, code in enumerate(codes):
        if code in MIRROR_CODES and zero_metal[axis]:
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction and "
                f"this kernel's ghost carry relies on the two sets being disjoint")

    transcription = constitutive_transcription(expansion, contract)
    if not transcription["identical"]:
        raise AssertionError(
            "this family's parameterised constitutive statements no longer reproduce "
            "complex_fields.bloch_constitutive_source('E'); the fused arithmetic would "
            "silently stop being the certified arithmetic. emitted="
            f"{transcription['emitted']!r} certified={transcription['certified']!r}")

    head = certified_curl_head(codes, phased, expansion, contract)
    # THE BETA TERM MUST BE PRESENT, and IN THE LIFTED HEAD -- the insert sits between
    # the dtdx curl and BOTH masks, which is above the cut.
    for statement in _BETA_STATEMENTS:
        if statement not in head:
            raise AssertionError(
                f"the lifted folded beta complex curl head does not carry "
                f"{statement.strip()!r}; this weld would be folded_complex_fused_pair "
                f"under another name, and every device leg would still pass because "
                f"beta = 0 IS that family's arithmetic")

    statements = certified_curl_statements(codes, phased, expansion, contract)
    body: List[str] = [head.rstrip("\n")]
    body.extend([
        "",
        "    // The three top-plane flags the lifted head declares are read below by "
        "the far",
        "    // carry's ownership guard, and by folded_top_plane_mask's own lines "
        "above.",
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
        "    // Read HERE for the owned cell and reused at the NEAR ghost, which",
        "    // differs from its source only on axes that are NOT the component's",
        "    // own. The FAR ghost does NOT share it.",
    ])
    body.extend(_constitutive_coefficient_lines(expansion, contract))
    body.append("")

    for target in range(3):
        previous, damped, recurrence, _, _ = statements["recurrence"][target]
        body.extend([previous, damped, recurrence])
    body.append("")
    body.append(statements["aux_store"])

    for target in range(3):
        near = near_fill_axes(codes, target)
        if len(near) > 2:
            raise AssertionError(
                f"component {target} is a near-fill destination on {near}; for the D "
                f"family that set is at most two axes (fields.IYEE_SHIFTS)")
        far = far_fill_axes(codes, target)
        if len(far) > 1 or set(far) & set(near):
            raise AssertionError(
                f"component {target} is a far-fill destination on {far} against near "
                f"axes {near}; for the D family the two sets are COMPLEMENTARY")
        for axis in near:
            fill = near_fill_transcription(axis, expansion, contract)
            if not fill["identical"]:
                raise AssertionError(
                    "this family's ghost write is no longer the certified folded "
                    f"complex NEAR fill's own line: {fill['rows']!r}")
        for axis in far:
            fill = far_fill_transcription(axis, expansion, contract)
            if not fill["identical"]:
                raise AssertionError(
                    "this family's ghost write is no longer the certified folded "
                    f"complex FAR fill's own line: {fill['rows']!r}")
        _, _, _, pre_scaling, displacement = statements["recurrence"][target]
        body.append("")
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
        body.append(indent + pre_scaling.strip())
        body.append(indent + displacement.strip())
        cleared = zero_metal_lines(target, zero_metal, f"v{target}", indent)
        body.extend(cleared or [f"{indent}// no walled axis clears this target"])
        body.append(indent + statements["stores"][target])
        body.extend(_constitutive_block(
            target, "ii", f"c_mul_field_left(v{target}, e{target}[ii])",
            f"kp_{target}", f"km_{target}", f"o{target}_", indent))
        if owned:
            body.append("")
            body.extend(_carry_blocks(target, near, far, zero_metal, indent))
            body.append("    }")

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": "\n".join(body),
    })


def compile_folded_beta_complex_fused_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, Bloch flags, walls, arm, mode)."""
    return compile_source(folded_beta_complex_fused_pair_source(
        codes, phased, zero_metal, expansion,
        contract)).folded_beta_complex_fused_pair_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _expansion_reasons(parity_record: Any, beta_record: Any) -> List[str]:
    """Why the arm cannot be bound, named rather than defaulted.

    EACH RECORD FALLS BACK TO ITS OWN LOADER WHEN NOT THREADED, which is what
    :func:`.folded_complex._expansion_reasons` does with its one artifact and what the
    plan builder does with both. Without the fallback the predicate and the builder
    DISAGREE whenever this family is reached through the arm table, which is the defect
    the magnetic twin carried until 2026-08-30. Nothing is widened: a missing or
    unclassifiable artifact still refuses BY NAME.
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
            "no folded complex parity probe licenses an EXPANSION arm; this kernel "
            "emits the mirror fills' parity product and the arm is a MEASURED "
            "platform fact, never a default")
    if beta is None:
        reasons.append(
            "no special_kz beta probe licenses an EXPANSION arm; this kernel emits "
            "the beta insert's imaginary-coefficient product and that orientation is "
            "not classified by the folded complex artifact")
    if parity is not None and beta is not None and parity != beta:
        reasons.append(
            f"the two probes disagree: the parity artifact licenses {parity!r} and "
            f"the beta artifact {beta!r}. One helpers emission serves the whole "
            f"kernel, so a disagreement means no single transcription reproduces both "
            f"call sites on this host")
    return reasons


def metal_folded_beta_complex_fused_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None, beta_probe: Any = None) -> Coverage:
    """May ONE dispatch span folded beta complex ``step_D`` -> fills -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    The halves are :mod:`.folded_beta`'s ROUTING verdicts (a fold MANDATORY), which is
    what keeps this product disjoint from :mod:`.beta_fused_electric_pair` and from
    :mod:`.folded_complex_fused_pair` in both directions.
    """
    reasons: List[str] = []

    curl = folded_beta.folded_beta_composition_bloch_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency, beta_probe)
    if not curl.covered:
        reasons.extend(f"folded beta complex curl half: {reason}"
                       for reason in curl.reasons)
    electric = folded_beta.folded_beta_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency, beta_probe)
    if not electric.covered:
        reasons.extend(f"folded beta complex constitutive half: {reason}"
                       for reason in electric.reasons)

    # THE PARITY ARM IS THIS PRODUCT'S OWN QUESTION and neither half asks it: the two
    # halves are folded_beta's, which never emits a mirror-parity product because that
    # family registers NO fill arm. The fill arithmetic this weld carries inline is
    # folded_complex's, so folded_complex's artifact is required too.
    reasons.extend(_expansion_reasons(probe, beta_probe))

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D unless the
    # deposit repair brackets the launch -- which for this family it does.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver injects "
            f"it BETWEEN step_D and update_E (driver.py:3294-3299), which is work "
            f"inside the seam this kernel closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)
    reasons.extend(_far_carry_reasons(grid, codes))
    reasons.extend(_mirror_phase_reasons(grid, codes))

    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; the "
                    f"near fill images stored cell {NEAR_SOURCE_INDEX} and this "
                    f"kernel images it from that cell's own thread")

    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedBetaComplexFusedPairPlan(KernelPlan):
    """ONE dispatch that performs five driver passes on a folded complex beta run."""

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values",
                 "near_parities", "far_parities", "beta_values", "zero_metal",
                 "shape", "n_elem", "dtdx", "reflect", "expansion", "params")

    family = "folded beta complex PML D-curl/mirror-fill/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phased", "zero_metal", "reflect", "expansion")

    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 near_parities: Sequence[Tuple[float, float]],
                 far_parities: Sequence[Tuple[float, float]],
                 beta_values: Sequence[Tuple[float, float]],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 reflect: Sequence[Optional[int]], expansion: str, params: Any,
                 pointers: Sequence[Any], functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.near_parities = tuple((float(re), float(im))
                                   for re, im in near_parities)
        self.far_parities = tuple((float(re), float(im)) for re, im in far_parities)
        # float(): `beta_curl_coefficients` already rounded ONCE to complex64 and
        # float() of a numpy.float32 preserves the word, the sign of a zero included.
        self.beta_values = tuple((float(re), float(im)) for re, im in beta_values)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.reflect = tuple(None if row is None else int(row) for row in reflect)
        self.expansion = str(expansion)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def params_record_dtype() -> Any:
    """The host record's dtype — ELEVEN ``float2`` members first, itemsize 120."""
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


def beta_words(grid: Any) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """``((bpr, bpi), (bmr, bmi))`` — the CURL ARM'S OWN coefficients, called.

    :func:`.special_kz.beta_curl_coefficients` is stepping.py:797-811 evaluated on the
    objects the array path evaluates it on, so the words this kernel receives are its
    bits, the sign of a zero included. ``magnetic=False`` because :data:`CURL_SUB_STEP`
    is ``step_D`` -- NOT a default, because the two seams take opposite signs;
    ``complex_storage=True`` because this is the complex arm. Neither is a parameter.
    """
    return special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(CURL_SUB_STEP == "step_B"),
        complex_storage=True)


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]],
                   near_parities: Sequence[Tuple[float, float]],
                   far_parities: Sequence[Tuple[float, float]],
                   beta_values: Sequence[Tuple[float, float]],
                   reflect: Sequence[Optional[int]], device: str) -> Any:
    """The seventeen scalar-shaped values as one 120-byte device record."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    for name, (real, imag) in zip(("px", "py", "pz"), tuple(phase_values)):
        record[name] = (np.float32(real), np.float32(imag))
    for name, (real, imag) in zip(("m0", "m1", "m2"), tuple(near_parities)):
        record[name] = (np.float32(real), np.float32(imag))
    for name, (real, imag) in zip(("d0", "d1", "d2"), tuple(far_parities)):
        record[name] = (np.float32(real), np.float32(imag))
    beta = tuple(beta_values)
    if len(beta) != 2:
        raise ValueError(
            f"beta_values must be the ((bpr, bpi), (bmr, bmi)) pair "
            f"beta_curl_coefficients returns, got {beta_values!r}")
    for name, (real, imag) in zip(("bp", "bm"), beta):
        record[name] = (np.float32(real), np.float32(imag))
    rows = tuple(-1 if value is None else int(value) for value in reflect)
    if len(rows) != 3:
        raise ValueError(f"reflect must be a per-axis triple, got {reflect!r}")
    record["rx"], record["ry"], record["rz"] = rows
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_folded_beta_complex_fused_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None, beta_probe: Any = None,
        ) -> Optional[MetalFoldedBetaComplexFusedPairPlan]:
    """Build the folded beta complex fused D/E plan, or ``None`` when refused."""
    if not metal_folded_beta_complex_fused_pair_coverage(
            fields, pml, sources, residency, probe, beta_probe).covered:
        return None
    from .folded_complex import expansion_from_probe  # noqa: PLC0415
    from .folded_complex import load_expansion_probe  # noqa: PLC0415

    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds, _far_reflect_rows  # noqa: PLC0415
    from . import complex_fields  # noqa: PLC0415

    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    walls = zero_metal_axes(grid)
    kinds = _boundary_kinds(grid, pml)
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds), backward=BACKWARD)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    near_parities = parity_words(codes, phases)
    far_parities = far_parity_words(codes, phases)
    beta_values = beta_words(grid)
    reflect = _far_reflect_rows(grid)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    flux = [bind_complex(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Dx", "Dy", "Dz")]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in ("Hx", "Hy", "Hz")]
    electric = [bind_complex(name, getattr(fields, name))
                for name in ("Ex", "Ey", "Ez")]
    workspace = [bind_complex("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Ex", "Ey", "Ez")]
    inverse_epsilon = [
        bind_real("inv_eps_" + name, fields.inverse_epsilon_for(name))
        for name in ("Ex", "Ey", "Ez")]
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"))
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + magnetic + electric + workspace + inverse_epsilon
                + curl_coefficients + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_beta_complex_fused_pair(
                codes, flags, walls, expansion, mode)

    dtdx = grid.dt / grid.dx
    return MetalFoldedBetaComplexFusedPairPlan(
        residency, volumes, codes, flags, values, near_parities, far_parities,
        beta_values, walls, grid.shape, dtdx, reflect, expansion,
        _params_tensor(grid.shape, dtdx, values, near_parities, far_parities,
                       beta_values, reflect, residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"folded beta complex fused D/E pair cannot fill {slot}",))
    return metal_folded_beta_complex_fused_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedBetaComplexFusedPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_beta_complex_fused_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG IS EVERY OTHER FUSED PAIR'S. ``plan_step`` assigns at most one arm per
    slot and this product spans five. Registering unwired keeps it ENUMERABLE for the
    disjointness sweep while ``arms.arms_for`` skips it, so no existing arm's selection
    changes. It reaches its two slots through ``launch.FUSED_PAIR_ARMS``.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "folded beta complex fused D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded beta complex fused D/E pair: ",
                          noun=("folded beta complex PML D-curl/mirror-fill/"
                                "stored-E pair"),
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
