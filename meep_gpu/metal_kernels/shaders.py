"""Metal shader sources for the real-field PML step path, and their specialisation.

Two kernels live here, transcribed from ``stepping.py`` with the arithmetic held
character for character:

* :func:`curl_source` — ``stepping.step_B`` and ``stepping.step_D``: the curl
  (``_curl_from_operands``) and its two PML forms (``_apply_curl_regions``);
* :func:`constitutive_source` — ``stepping.update_H`` and ``stepping.update_E``:
  the ``dsigw`` accumulation and the plain ``f = src``
  (``_apply_constitutive_regions``).

ONE body serves both sub-steps in each pair, exactly as ``step_generic.cpp``'s
single ``step_curl`` does.

THE PML FORM IS CHOSEN PER STRUCTURE CHUNK, as MEEP chooses it (step_db.cpp:56-59,
update_eh.cpp:162-163) and as the array path steps it. Membership is runtime data,
never source text: each kernel binds the chunk record
(:func:`meep_gpu.chunk_map.chunk_bounds_record`) as ``constant ChunkBounds& cb``
and reads it through the anchored bounds block of :mod:`.chunk_bounds`, evaluated
at the cell's own stored index. The record carries the ghost-adjusted membership
line, so a ghost row takes its image row's form, as the array path's line does.

* Curl, per target: on a cell whose chunk carries the target's dsigu layer, the
  split-field recurrence ``n = ((fu*kms) - curl)*sinv; f = (((f*kms_u) + n) -
  fu)*sinv_u; fu = n``; everywhere else the dsig-only form ``f = ((f*kms) -
  curl)*sinv``, which is the recurrence's first stage with ``fu := f``, and ``fu``
  is neither read nor written. The lattice of the membership is the target's own
  Yee shift on its dsigu axis: half-integer for B, integer for D.
* Constitutive, per component: on a cell whose chunk carries the component's own
  (dsigw) layer, ``f = f + ((kps*src) - (kms*prev))`` with ``f_w = src``;
  everywhere else ``f = src`` and ``f_w`` is neither read nor written. Integer
  lattice for H, half-integer for E.
* The curl stencil is MEEP's left-to-right bracket ``dtdx * (((g1s - g1) + g2) -
  g2s)``; a partner whose derivative axis is invariant is absent (the record's
  presence word, ``fields.cpp:428-453``), giving ``dtdx * (g1s - g1)`` or
  ``(-dtdx) * (g2s - g2)``.

Both plain forms are OUTPUT SELECTS: the plain word is the select's other arm, not
the accumulation with unit or zero coefficients, which differs on a ``-0``.

WHY SOURCE SUBSTITUTION AND NOT RUNTIME BRANCHES. Triton specialises on
``tl.constexpr`` arguments: ``BACKWARD``, ``BCX``/``BCY``/``BCZ``, ``SCALE`` are
compile-time constants and the branches on them are gone before any float
expression is emitted. ``torch.mps.compile_shader`` takes a source string and
nothing else, so the same specialisation is performed HERE, by substitution, which
keeps the emitted arithmetic identical to the Triton body instead of handing the
compiler branch-hoisting opportunities around the float expressions. Each
specialisation is therefore a distinct source string and a distinct memo key.

THE TRANSCRIPTION RULES, each measured on this host rather than assumed:

1. **The contraction pragma.** Metal defaults to contraction ON; without the
   pragma the compiler contracts ``field*kms_u + fu`` into an fma and the result
   diverges (measured: 37,483-37,568 differing words of 262,144 per component on
   the composed PML expression). It is spelled ONCE, in
   :data:`_CONTRACTION_PRAGMA`, and ``test_metal_kernels`` fails if a second
   spelling — or any math-mode pragma — appears anywhere under this package.
   Unlike Triton's ``enable_fp_fusion``, this is a property of the SOURCE and not
   of the launch, which is why :func:`curl_source` takes a ``contract`` argument
   and the plan holds one compiled function per mode.
2. **Every paren is preserved.** ``dtdx * (((c_y - c) + b) - b_z)`` and
   ``f + ((kps*src) - (kms*prev))`` are the two groupings that decide
   bit-identity; any other association is a different float32 number (the
   package's previous pairwise curl grouping measured 27,479-27,544 differing
   words per component).
3. **The metallic ghost rule is a ternary, never a clamped load.** ``v ? p[o] :
   0.0f`` delivers the exact ``0.0`` past the wall that ``tl.load(..., other=0.0)``
   delivers, without dereferencing anything.
4. **min/max are spelled as selects** where they occur. They do not occur in
   either body below, and the rule is recorded because Metal's ``min``/``max``
   builtins return the SECOND operand when both are zeros and drop NaN, which
   disagrees with ``numpy.minimum``/``maximum``; the select agrees exactly
   (measured 0/8045). Neither body divides, calls ``sqrt``, or takes a min or a
   max, which is why this pair is the lowest-risk one to certify first.
5. **Negation, where it occurs, is spelled ``-x``.** On this platform that is a
   sign-bit operation (0/8045 mismatches, preserving -0 and subnormal bits), while
   ``0.0f - x`` and ``x * -1.0f`` are different operations (25 and 13 mismatches).
   Triton's measured fact — unary minus lowered as ``0.0 - x`` — does NOT reproduce
   here. The curl's absent-first-partner form is the one negation in this file,
   ``(-dtdx) * (g2s - g2)``: MEEP negates ``dtdx`` (step_generic.cpp:72-76), and the
   array path multiplies by the float32 word of ``-dtdx``, which is ``-x`` of the
   float32 word of ``dtdx``.

THE INDEX ARITHMETIC IS INTEGER AND EXACT, which is why the periodic wrap is
specialised to the one branch that can fire for a given direction (forward shifts
reach ``n`` but never ``-1``) rather than transcribing both of Triton's nested
``tl.where`` calls. No float rounds on that path, so the two spellings cannot
differ in bits — and the gate's synthetic leg runs both directions on every
configuration anyway.

Nothing here imports torch: a source string is a string, and this module must be
readable and testable on a host with no GPU and no optional dependency.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Sequence, Tuple

from . import chunk_bounds

#: Boundary codes matching ``stepping``'s string kinds, as
#: ``triton_kernels.kernels`` spells them.
PERIODIC = 0
METALLIC = 1

#: The two contraction modes a source may be built in. ``off`` is the shipped
#: one and the only one any plan launches by default; ``fast`` exists so the byte
#: gate can MEASURE the guard's effect (identical with, non-identical without)
#: rather than assert it from the pragma's presence.
CONTRACT_OFF = "off"
CONTRACT_FAST = "fast"
CONTRACT_MODES: Tuple[str, str] = (CONTRACT_OFF, CONTRACT_FAST)

#: THE ONE COMPILE OPTION THAT DECIDES BIT-IDENTITY on this platform, spelled
#: exactly once in this package. The Metal analogue of Triton's
#: ``ENABLE_FP_FUSION = False`` — but a file-scope property rather than a launch
#: keyword, so it is baked into the source string and a plan that wants the other
#: mode compiles a different one.
_CONTRACTION_PRAGMA = "#pragma clang fp contract({mode})"


def contraction_pragma(contract: str = CONTRACT_OFF) -> str:
    """The contraction directive for one mode, from the single spelling above."""
    if contract not in CONTRACT_MODES:
        raise ValueError(f"contract must be one of {CONTRACT_MODES}, got {contract!r}")
    return _CONTRACTION_PRAGMA.format(mode=contract)


# ---------------------------------------------------------------------------
# The curl: stepping.step_B / stepping.step_D
# ---------------------------------------------------------------------------

#: The chunk record's binding index on the curl: one past the last scalar.
CURL_RECORD_BUFFER = 20

#: The chunk record's binding index on the constitutive: one past the last scalar.
CONSTITUTIVE_RECORD_BUFFER = 22

#: Per curl target, MEEP's ``(g1, g1 axis, g2, g2 axis)`` as the kernel names them:
#: the source volume (``a``/``b``/``c`` = source 0/1/2) and the axis it is
#: differenced along. ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` with
#: ``first``/``second`` mapped onto the sub-step's three sources; the test file
#: checks the two tables agree.
CURL_OPERANDS: Tuple[Tuple[int, int, int, int], ...] = ((2, 1, 1, 2), (0, 2, 2, 0),
                                                       (1, 0, 0, 1))

#: Per curl target, ``(dsig axis, dsigu axis)``: vec.hpp's cycle_direction, the
#: same triple on both sides (``stepping.B_CURL_TERMS``).
CURL_CYCLE: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: The Yee sub-lattice of each curl side's membership along its dsigu axes: B
#: targets sit half a cell off on the two axes that are not their own, which
#: include every target's dsigu axis; D targets sit on the integer lattice there
#: (``fields.IYEE_SHIFTS``). The test file derives both from the Yee table.
CURL_HALF_INTEGER: Dict[bool, bool] = {False: True, True: False}

#: The Yee sub-lattice of each constitutive side's membership along each
#: component's own axis: integer for H, half-integer for E.
CONSTITUTIVE_HALF_INTEGER: Dict[str, bool] = {"H": False, "E": True}

_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__RECORD_STRUCT__

kernel void pml_curl_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device const float* kmx     [[buffer(9)]],
    device const float* sinvx   [[buffer(10)]],
    device const float* kmy     [[buffer(11)]],
    device const float* sinvy   [[buffer(12)]],
    device const float* kmz     [[buffer(13)]],
    device const float* sinvz   [[buffer(14)]],
    constant uint&      nx      [[buffer(15)]],
    constant uint&      ny      [[buffer(16)]],
    constant uint&      nz      [[buffer(17)]],
    constant uint&      n_elem  [[buffer(18)]],
    constant float&     dtdx    [[buffer(19)]],
    __RECORD_BINDING__,
    uint idx [[thread_position_in_grid]])
{
    // The dispatch is sized from the first tensor argument's element count, so a
    // volume wider than n_elem would step cells the array path does not own.
    if (idx >= n_elem) { return; }

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;

    // --- the chunk record: membership at this cell, partner presence ------------
__BOUNDS__

    // --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------------
    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    // METALLIC serves an exact 0.0 past the wall -- Triton's `other=0.0`.
    float a   = g0[ii];
    float b   = g1[ii];
    float c   = g2[ii];
    float a_y = vy ? g0[oy] : 0.0f;
    float a_z = vz ? g0[oz] : 0.0f;
    float b_x = vx ? g1[ox] : 0.0f;
    float b_z = vz ? g1[oz] : 0.0f;
    float c_x = vx ? g2[ox] : 0.0f;
    float c_y = vy ? g2[oy] : 0.0f;

    // --- the curl (stepping._curl_from_operands): MEEP's left-to-right bracket
    // ((g1s - g1) + g2) - g2s with both partners; (g1s - g1) without g2; and
    // (-dtdx) * (g2s - g2) without g1 (step_generic.cpp:72-76, :104-108). DO NOT
    // regroup: each operation rounds once.
    float curl0 = has_y ? (has_z ? (dtdx * (((c_y - c) + b) - b_z)) : (dtdx * (c_y - c)))
                        : (has_z ? ((-dtdx) * (b_z - b)) : 0.0f);
    float curl1 = has_z ? (has_x ? (dtdx * (((a_z - a) + c) - c_x)) : (dtdx * (a_z - a)))
                        : (has_x ? ((-dtdx) * (c_x - c)) : 0.0f);
    float curl2 = has_x ? (has_y ? (dtdx * (((b_x - b) + a) - a_y)) : (dtdx * (b_x - b)))
                        : (has_y ? ((-dtdx) * (a_y - a)) : 0.0f);

    // --- ownership mask (stepping._mask_non_owned_cells) ------------------------
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

    // --- the two PML forms, per structure chunk (stepping._apply_curl_regions) --
    // dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    // sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y). A target
    // whose dsigu membership holds steps the split-field recurrence; any other
    // steps the dsig-only form, which is the recurrence's first stage with fu := f,
    // and leaves fu unread and unwritten.
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];
    const bool du0 = __DU0__, du1 = __DU1__, du2 = __DU2__;

    float p0 = du0 ? u0[ii] : f0[ii];
    float n0 = ((p0 * km_y) - curl0) * si_y;
    float v0 = du0 ? ((((f0[ii] * km_z) + n0) - p0) * si_z) : n0;

    float p1 = du1 ? u1[ii] : f1[ii];
    float n1 = ((p1 * km_z) - curl1) * si_z;
    float v1 = du1 ? ((((f1[ii] * km_x) + n1) - p1) * si_x) : n1;

    float p2 = du2 ? u2[ii] : f2[ii];
    float n2 = ((p2 * km_x) - curl2) * si_x;
    float v2 = du2 ? ((((f2[ii] * km_y) + n2) - p2) * si_y) : n2;

    if (du0) { u0[ii] = n0; }
    if (du1) { u1[ii] = n1; }
    if (du2) { u2[ii] = n2; }
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
"""


def curl_bounds_entries(backward: bool) -> Tuple[Tuple[bool, int], ...]:
    """The record entries a curl reads: every axis, on the side's dsigu lattice."""
    half = CURL_HALF_INTEGER[bool(backward)]
    return tuple((half, axis) for axis in range(3))


def curl_bounds_block(backward: bool, struct: str = "cb") -> str:
    """The anchored bounds block of one curl side: membership and partner presence."""
    return chunk_bounds.prologue(curl_bounds_entries(backward), struct=struct,
                                 presence=(0, 1, 2))


def curl_membership_flags(backward: bool) -> Tuple[str, str, str]:
    """Per target, the membership local its form reads: its dsigu axis's line."""
    half = CURL_HALF_INTEGER[bool(backward)]
    return tuple(chunk_bounds.membership_name(half, dsigu)
                 for _dsig, dsigu in CURL_CYCLE)  # type: ignore[return-value]


def ghost(axis: str, code: int, backward: bool) -> str:
    """One axis's ghost rule, specialised the way the Triton constexpr specialises.

    METALLIC sets the validity flag the load's ternary reads; PERIODIC wraps the
    index. The wrap is written as the single branch a given direction can reach —
    a forward shift lands on ``n`` and never on ``-1`` — because the arithmetic is
    integer and exact and the unreachable branch cannot change a bit.
    """
    shifted, valid, extent = {"x": ("si", "vx", "nxi"),
                              "y": ("sj", "vy", "nyi"),
                              "z": ("sk", "vz", "nzi")}[axis]
    if code == METALLIC:
        return f"    {valid} = ({shifted} >= 0) && ({shifted} < {extent});"
    if code != PERIODIC:
        raise ValueError(f"boundary code {code!r} is neither PERIODIC nor METALLIC")
    if backward:
        return f"    {shifted} = ({shifted} < 0) ? ({extent} - 1) : {shifted};"
    return f"    {shifted} = ({shifted} == {extent}) ? 0 : {shifted};"


def ownership_mask(codes: Sequence[int], backward: bool, zero: str = "0.0f") -> str:
    """``stepping._mask_non_owned_cells``, unrolled per (target, metallic axis).

    Cell 0 of a metallic axis, for every target whose Yee shift there is 0. The B
    targets have a single zero shift each (Bx:x, By:y, Bz:z); the D targets have
    two (Dx:y,z  Dy:x,z  Dz:x,y). Masked, then the recurrence runs on zero.

    ``zero`` is the literal the mask writes. It defaults to the real families'
    ``0.0f`` — so the certified sources are emitted character for character, all
    THIRTY-SIX rows of ``fingerprints.json`` (eighteen labels in each of the two
    contraction modes), re-measured unchanged when this parameter landed — and a
    complex family passes ``float2(0.0f, 0.0f)``, which is the array path assigning
    a complex zero to BOTH planes (stepping.py:1943, :1949).
    """
    if backward:
        pairs = ((0, 1, "at_y"), (0, 2, "at_z"), (1, 0, "at_x"),
                 (1, 2, "at_z"), (2, 0, "at_x"), (2, 1, "at_y"))
    else:
        pairs = ((0, 0, "at_x"), (1, 1, "at_y"), (2, 2, "at_z"))
    lines = [f"    curl{target} = {flag} ? {zero} : curl{target};"
             for target, axis, flag in pairs if codes[axis] == METALLIC]
    return "\n".join(lines) or "    // no metallic axis: no ownership mask"


def curl_source(codes: Sequence[int], backward: bool,
                contract: str = CONTRACT_OFF) -> str:
    """The specialised ``pml_curl_step`` source for one (boundary triple, direction).

    ``codes`` is the per-axis 0/1 PERIODIC/METALLIC triple ``_boundary_kinds``
    resolves; ``backward`` selects ``step_D``'s negated strides over ``step_B``'s
    forward ones. Both are baked into the string, as Triton bakes its constexprs.
    The chunk division is NOT: it is the bound record, so one source serves every
    division and the specialisation set does not grow with the grid.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    flags = curl_membership_flags(backward)
    return substitute(_CURL_TEMPLATE, {
        "__CONTRACT__": contraction_pragma(contract),
        "__RECORD_STRUCT__": chunk_bounds.struct_source(),
        "__RECORD_BINDING__": chunk_bounds.binding(CURL_RECORD_BUFFER),
        "__BOUNDS__": curl_bounds_block(backward),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": ghost("x", codes[0], backward),
        "__GHOST_Y__": ghost("y", codes[1], backward),
        "__GHOST_Z__": ghost("z", codes[2], backward),
        "__MASK__": ownership_mask(codes, backward),
        "__DU0__": flags[0],
        "__DU1__": flags[1],
        "__DU2__": flags[2],
    })


# ---------------------------------------------------------------------------
# The constitutive accumulation: stepping.update_H / stepping.update_E
# ---------------------------------------------------------------------------

_CONSTITUTIVE_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__RECORD_STRUCT__

kernel void constitutive_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       w0      [[buffer(3)]],
    device float*       w1      [[buffer(4)]],
    device float*       w2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device const float* e0      [[buffer(9)]],
    device const float* e1      [[buffer(10)]],
    device const float* e2      [[buffer(11)]],
    device const float* kp0     [[buffer(12)]],
    device const float* km0     [[buffer(13)]],
    device const float* kp1     [[buffer(14)]],
    device const float* km1     [[buffer(15)]],
    device const float* kp2     [[buffer(16)]],
    device const float* km2     [[buffer(17)]],
    constant uint&      nx      [[buffer(18)]],
    constant uint&      ny      [[buffer(19)]],
    constant uint&      nz      [[buffer(20)]],
    constant uint&      n_elem  [[buffer(21)]],
    __RECORD_BINDING__,
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }

    int nyi = int(ny), nzi = int(nz);
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;

    // --- the chunk record: each component's own-axis membership at this cell -----
__BOUNDS__

    // The coefficient index is on the component's OWN axis: kps_x/kms_x for
    // target 0, _y for 1, _z for 2 (H_CONSTITUTIVE_TERMS / E_CONSTITUTIVE_TERMS).
    // That is MEEP's dsigw, NOT the dsig/dsigu cycle the curl recurrence uses.
    float kp_0 = kp0[i], km_0 = km0[i];
    float kp_1 = kp1[j], km_1 = km1[j];
    float kp_2 = kp2[k], km_2 = km2[k];
    const bool dw0 = __DW0__, dw1 = __DW1__, dw2 = __DW2__;

    // --- component 0 -----------------------------------------------------------
    // On a dsigw cell `prev` is read BEFORE `fw` is written (the array path copies
    // fw first) and the right-hand side is formed first: f + ((kps*src) -
    // (kms*prev)), MEEP's step_update_EDHB (step_generic.cpp:689, :697). Elsewhere
    // the word is src itself, an output select, and fw is not touched.
    float prev0 = dw0 ? w0[ii] : 0.0f;
    float src0 = __SRC0__;
    if (dw0) { w0[ii] = src0; }
    float a0 = src0;
    if (dw0) { a0 = f0[ii] + ((kp_0 * src0) - (km_0 * prev0)); }
    f0[ii] = a0;

    // --- component 1 -----------------------------------------------------------
    float prev1 = dw1 ? w1[ii] : 0.0f;
    float src1 = __SRC1__;
    if (dw1) { w1[ii] = src1; }
    float a1 = src1;
    if (dw1) { a1 = f1[ii] + ((kp_1 * src1) - (km_1 * prev1)); }
    f1[ii] = a1;

    // --- component 2 -----------------------------------------------------------
    float prev2 = dw2 ? w2[ii] : 0.0f;
    float src2 = __SRC2__;
    if (dw2) { w2[ii] = src2; }
    float a2 = src2;
    if (dw2) { a2 = f2[ii] + ((kp_2 * src2) - (km_2 * prev2)); }
    f2[ii] = a2;
}
"""


def constitutive_bounds_entries(side: str) -> Tuple[Tuple[bool, int], ...]:
    """The record entries a constitutive reads: every own axis, on the side's lattice."""
    half = CONSTITUTIVE_HALF_INTEGER[side]
    return tuple((half, axis) for axis in range(3))


def constitutive_bounds_block(side: str, struct: str = "cb") -> str:
    """The anchored bounds block of one constitutive side: own-axis membership."""
    return chunk_bounds.prologue(constitutive_bounds_entries(side), struct=struct)


def constitutive_membership_flags(side: str) -> Tuple[str, str, str]:
    """Per component, the membership local its form reads: its own axis's line."""
    half = CONSTITUTIVE_HALF_INTEGER[side]
    return tuple(chunk_bounds.membership_name(half, axis)
                 for axis in range(3))  # type: ignore[return-value]


#: The constitutive product per side. The H side reads B directly (mu = 1 is baked
#: into the array path too: ``stepping.update_H`` passes ``fields.Bx``); the E side
#: reads ``D * inv_eps`` with **D ON THE LEFT**, because the array path writes
#: ``source * fields.inverse_epsilon_for(component)`` (stepping.py:1011-1013).
#: Float multiplication is bitwise commutative, and a transcription is not a place
#: to rely on that.
_CONSTITUTIVE_PRODUCTS: Dict[str, Tuple[str, str, str]] = {
    "H": ("g0[ii]", "g1[ii]", "g2[ii]"),
    "E": ("g0[ii] * e0[ii]", "g1[ii] * e1[ii]", "g2[ii] * e2[ii]"),
}


def constitutive_source(side: str, contract: str = CONTRACT_OFF) -> str:
    """The specialised ``constitutive_step`` source for one side ('H' or 'E').

    The Yee sub-lattice is NOT chosen here. The host binds ``kps_a``/``kms_a`` for
    H and ``kps_a_h``/``kms_a_h`` for E (stepping.py:948 vs :1015), so the kernel
    never chooses and cannot choose wrong — but the host can, which is why the gate
    carries a swap mutation for it.

    The three inverse-epsilon buffers stay in the signature on both sides so the
    buffer layout is one layout. On the H side the plan binds the sources as
    placeholders, exactly as ``triton_kernels.launch.ConstitutivePlan`` does, and
    the substituted product never reads them. The chunk record binds last, at
    :data:`CONSTITUTIVE_RECORD_BUFFER`.
    """
    if side not in _CONSTITUTIVE_PRODUCTS:
        raise ValueError(f"side must be one of {tuple(_CONSTITUTIVE_PRODUCTS)}, "
                         f"got {side!r}")
    products = _CONSTITUTIVE_PRODUCTS[side]
    flags = constitutive_membership_flags(side)
    return substitute(_CONSTITUTIVE_TEMPLATE, {
        "__CONTRACT__": contraction_pragma(contract),
        "__RECORD_STRUCT__": chunk_bounds.struct_source(),
        "__RECORD_BINDING__": chunk_bounds.binding(CONSTITUTIVE_RECORD_BUFFER),
        "__BOUNDS__": constitutive_bounds_block(side),
        "__DW0__": flags[0],
        "__DW1__": flags[1],
        "__DW2__": flags[2],
        "__SRC0__": products[0],
        "__SRC1__": products[1],
        "__SRC2__": products[2],
    })


# ---------------------------------------------------------------------------
# Substitution and fingerprints
# ---------------------------------------------------------------------------

def substitute(template: str, replacements: Dict[str, str]) -> str:
    """Fill every placeholder, and refuse a template that still holds one.

    A leftover placeholder in a SHIPPED source is a compile error and would be
    caught immediately; a leftover in a MUTANT source is a silent no-op that would
    report the defect as uncaught. Both are refused here by name.
    """
    filled = template
    for token, value in replacements.items():
        if token not in filled:
            raise KeyError(f"placeholder {token!r} is not present in the template")
        filled = filled.replace(token, value)
    remaining = [line for line in filled.splitlines() if "__" in line and
                 line.strip().startswith("__")]
    if remaining:
        raise ValueError(f"unfilled placeholders remain: {remaining!r}")
    return filled


def source_sha256(source: str) -> str:
    """The fingerprint of one specialised source, as ``fingerprints.json`` stores it."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


#: Every specialisation the shipped plans can emit, as (label, source) pairs. The
#: fingerprint file records one sha256 per entry, so a compiler-visible edit to any
#: one of them is a fingerprint change rather than a silent recompile.
def enumerate_sources(contract: str = CONTRACT_OFF) -> Dict[str, str]:
    """Every shipped specialisation, keyed by a stable label."""
    out: Dict[str, str] = {}
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for cx in (PERIODIC, METALLIC):
            for cy in (PERIODIC, METALLIC):
                for cz in (PERIODIC, METALLIC):
                    label = f"pml_curl_step/{name}/{cx}{cy}{cz}"
                    out[label] = curl_source((cx, cy, cz), backward, contract)
    for side in ("H", "E"):
        out[f"constitutive_step/{side}"] = constitutive_source(side, contract)
    return out
