"""Complex-field (force_complex_fields / Bloch k_point) PML curl + constitutive, on Metal.

THE LARGEST CORPUS FAMILY ON EITHER TRACK: sixteen rows, and the sole admitter on
all four sub-steps for every one of them. Complex64 is stepped as ``float2``
volumes, each part by the array path's own float32 arithmetic.

NO ENTRY IN ``fingerprints.json``'s ``kernel_source_sha256``, deliberately: this
module's sources are a family of specialisations, and ``gate_metal_complex``
hashes every specialisation it launches into its own results directory. This
module's own bytes ARE hashed, under ``host_sha256``, because
``launch._host_modules()`` enumerates the package.

WIRED: the four arms register ``wired=True``; the separation from the real kernels is
clause 2 (complex storage required here, refused there) and from :mod:`.special_kz` is
clause 8 (``grid.beta`` refused here by name).

TWO KERNELS, one per sub-step family, each transcribing ``stepping.py``:

* :func:`bloch_curl_source` -- complex ``step_B`` / ``step_D``: the curl
  (``_curl_operands``, ``_curl_from_operands``) and its two PML forms
  (``_apply_curl_regions``);
* :func:`bloch_constitutive_source` -- complex ``update_H`` / ``update_E``: the
  ``dsigw`` accumulation and the plain ``f = src`` (``_apply_constitutive_regions``).

EVERY PRODUCT IS FORMED PART BY PART, as MEEP forms it (``meep_gpu.part_products``):
a realnum coefficient times a field is one float32 product per part
(:func:`.templates.complex_helpers`), and complex add and subtract are per part by
definition. So each part of every word is the real kernel's arithmetic
(:mod:`.shaders`), and the forms are the real kernel's:

* the PML form is chosen PER STRUCTURE CHUNK from the bound chunk record
  (:mod:`.chunk_bounds`, read through the same anchored blocks the real kernels
  emit): on a cell whose chunk carries the target's dsigu layer the split-field
  recurrence ``n = ((fu*kms) - curl)*sinv; f = (((f*kms_u) + n) - fu)*sinv_u; fu =
  n``, everywhere else the dsig-only form ``((f*kms) - curl)*sinv`` with ``fu``
  neither read nor written;
* the constitutive is ``f + ((kps*src) - (kms*prev))`` with ``f_w = src`` on a cell
  whose chunk carries the component's own layer and the OUTPUT SELECT ``f = src``
  everywhere else, ``f_w`` untouched;
* the stencil is MEEP's left-to-right bracket ``((g1s - g1) + g2) - g2s``, with the
  absent-partner forms from the record's presence word (step_generic.cpp:72-76,
  :104-108).

THE BLOCH CONNECTION IS ON THE UNSHIFTED OPERAND, AT PLANE 0. On a phased axis the
engine stores MEEP's owned window (``Grid.stored_window_low``): a shifted read one
stored cell over is always an owned word, so the ghost rule is the plain periodic
wrap, and the reads that leave the window are the SAME-INDEX reads across the half
cell on plane 0 (``stepping._bloch_ghost_operand``). Per curl term, the unshifted
partner differenced along a phased axis is replaced there by MEEP's connected word:
``conj(eikna)`` times stored cell 0 on the B side (the word below the window) and
``eikna`` times it on the D side (the word above). The connection is MEEP's three
spellings, decided on the host from ``thephase = 1.0 * factor`` in double
(boundaries.cpp:203-226, :359-366; ``part_products.connect_parts``): within 1e-13 of
+1 a copy, of -1 a unary minus per part, otherwise ``complex<realnum>(thephase) *
word`` with every product and sum rounded on its own (:func:`.templates.complex_helpers`
``c_mul``). Which spelling an axis takes is baked into the source
(:data:`CONNECT_NONE` / :data:`CONNECT_NEGATE` / :data:`CONNECT_PHASE`); the phase
words are bound.

WHY float2 IS FORCED AND NOT PREFERRED. Metal's buffer attribute indices run 0..30,
so a kernel binds at most 31 buffers; a 32nd is a COMPILE ERROR, measured on this
toolchain. The curl binds :data:`CURL_BINDINGS` = 24 (9 ``float2`` volumes, 6
coefficient vectors, 5 scalars, 3 phases, the chunk record). The same kernel with
re/im planes bound SEPARATELY needs 36 and cannot be built at all; the gate compiles
that signature and requires the failure.

WHAT IS TRANSCRIBED, and from where (``stepping``):

* term table            ``B_CURL_TERMS`` / ``D_CURL_TERMS``
* ghost rule            ``_shift_up`` / ``_shift_down``, PERIODIC and METALLIC
* Bloch connection      ``_bloch_ghost_operand`` and ``part_products.connect``
* curl                  ``_curl_from_operands``, presence from ``curl_presence``
* ownership mask        ``_mask_non_owned_cells`` -- a complex zero to BOTH planes
* PML forms             ``_apply_curl_regions`` (``_apply_pml_update`` /
                        ``_apply_dsig_only_update``)
* constitutive          ``_apply_constitutive_regions`` (``_apply_constitutive_pml``
                        / the plain select)
* coefficient pairing   ``_curl_coefficients`` / ``_constitutive_coefficients``;
                        host-bound, never chosen in-kernel
* phase legality        ``_bloch_phases``: a phased axis must resolve PERIODIC; PML
                        on a phased axis is admitted
* phase value           ``grid.bloch_phase``: exp(2*pi*i*k*L), the Brillouin edge
                        EXACTLY -1+0j

The run's first ``update_H`` starts H and ``f_w`` from B on the dsigw cells
(``stepping.separate_h_from_b``); the constitutive kernel steps the forms alone, so
:class:`ComplexConstitutivePlan` declares that it does not start H and the composer
keeps that slot on the array path where the start can change a word
(``launch.h_start_reasons``).

Import contract: importable WITHOUT torch. The predicates and the plan builders
(to ``None``) must answer on a host with no GPU, which is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
    COVERED_BOUNDARIES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _susceptibility_reasons,
)
from ..part_products import CONNECT_TOLERANCE
from ..readback import cmul
from . import chunk_bounds as _chunk_bounds
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import MAX_BUFFER_BINDINGS, compile_source
from .launch import SUB_STEPS
from .plans import KernelPlan

FAMILY = "complex_fields"

#: How many buffers the shipped curl binds: 9 float2 volumes + 6 coefficient vectors
#: + 5 scalars + 3 phases + the chunk record. CHECKED AGAINST THE PLATFORM CEILING AT
#: IMPORT, because
#: the whole ``float2`` design rests on this margin and the number was previously a
#: literal in three places (this module's docstring, the gate's ``bindings`` leg and
#: a test). One home, checked, so a binding added to the signature is refused here by
#: name rather than discovered as a compile error a reader has to attribute.
CURL_BINDINGS = 24
if CURL_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex curl binds {CURL_BINDINGS} buffers and Metal's ceiling on this "
        f"toolchain is {MAX_BUFFER_BINDINGS}; the signature cannot be built at all. "
        f"See `split_plane_curl_signature` for the measurement that established the "
        f"ceiling")

#: The chunk record's binding index on the curl: one past the last phase.
CURL_RECORD_BUFFER = 23

#: The chunk record's binding index on the constitutive: one past ``n_elem``.
CONSTITUTIVE_RECORD_BUFFER = 22

#: MEEP's three connection spellings on a phased axis (boundaries.cpp:359-366), as the
#: per-axis code a curl source is specialised on. NONE is an unphased axis or a phase
#: within 1e-13 of +1 (a copy: the word itself); NEGATE a phase within 1e-13 of -1 (a
#: unary minus per part, the Brillouin edge); PHASE every other phase, the unfused
#: ``complex<realnum>(thephase) * word``.
CONNECT_NONE = 0
CONNECT_NEGATE = 1
CONNECT_PHASE = 2
CONNECT_KINDS = (CONNECT_NONE, CONNECT_NEGATE, CONNECT_PHASE)


__all__ = [
    "CONNECT_KINDS",
    "CONNECT_NEGATE",
    "CONNECT_NONE",
    "CONNECT_PHASE",
    "CONSTITUTIVE_RECORD_BUFFER",
    "CURL_BINDINGS",
    "CURL_RECORD_BUFFER",
    "FAMILY",
    "ComplexConstitutivePlan",
    "ComplexPmlCurlPlan",
    "bloch_constitutive_source",
    "bloch_curl_source",
    "bloch_phase_table",
    "complex_constitutive_coverage",
    "complex_pml_curl_coverage",
    "enumerate_complex_sources",
    "phase_arguments",
    "plan_complex_constitutive",
    "plan_complex_constitutive_from_arrays",
    "plan_complex_pml_curl",
    "plan_complex_pml_curl_from_arrays",
    "register_arms",
    "split_plane_curl_signature",
]


# ---------------------------------------------------------------------------
# The curl kernel
# ---------------------------------------------------------------------------
#
# TWENTY-FOUR BINDINGS: 9 float2 volumes + 6 coefficient vectors + 5 scalars + 3
# phases + the chunk record. The re/im-split form of the same kernel needs 36 and is
# a compile error; see `split_plane_curl_signature`.

_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

__RECORD_STRUCT__

kernel void bloch_pml_curl_step(
    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device float2*       u0      [[buffer(3)]],
    device float2*       u1      [[buffer(4)]],
    device float2*       u2      [[buffer(5)]],
    device const float2* g0      [[buffer(6)]],
    device const float2* g1      [[buffer(7)]],
    device const float2* g2      [[buffer(8)]],
    device const float*  kmx     [[buffer(9)]],
    device const float*  sinvx   [[buffer(10)]],
    device const float*  kmy     [[buffer(11)]],
    device const float*  sinvy   [[buffer(12)]],
    device const float*  kmz     [[buffer(13)]],
    device const float*  sinvz   [[buffer(14)]],
    constant uint&       nx      [[buffer(15)]],
    constant uint&       ny      [[buffer(16)]],
    constant uint&       nz      [[buffer(17)]],
    constant uint&       n_elem  [[buffer(18)]],
    constant float&      dtdx    [[buffer(19)]],
    constant float2&     px      [[buffer(20)]],
    constant float2&     py      [[buffer(21)]],
    constant float2&     pz      [[buffer(22)]],
    __RECORD_BINDING__,
    uint idx [[thread_position_in_grid]])
{
__GUARD__

__DECODE__
    int nxi = int(nx);
    int nyz = nyi * nzi;

    // --- the chunk record: membership at this cell, partner presence ------------
__BOUNDS__

    // --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------------
    // A phased axis wraps plainly: the stored window makes every read one cell over
    // an owned word (stepping._bloch_ghost_operand).
    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    // METALLIC serves an exact complex 0 past the wall: a (+0.0, +0.0) word pair
    // IS the complex metallic ghost.
    float2 a   = g0[ii];
    float2 b   = g1[ii];
    float2 c   = g2[ii];
    float2 a_y = vy ? g0[oy] : float2(0.0f, 0.0f);
    float2 a_z = vz ? g0[oz] : float2(0.0f, 0.0f);
    float2 b_x = vx ? g1[ox] : float2(0.0f, 0.0f);
    float2 b_z = vz ? g1[oz] : float2(0.0f, 0.0f);
    float2 c_x = vx ? g2[ox] : float2(0.0f, 0.0f);
    float2 c_y = vy ? g2[oy] : float2(0.0f, 0.0f);

    // --- the Bloch connection (stepping._bloch_ghost_operand) ---------------------
    // Each term's UNSHIFTED partner, as read along the axis that term differences it
    // on: target 0 reads c along y and b along z, target 1 a along z and c along x,
    // target 2 b along x and a along y. On plane 0 of a phased axis that read is the
    // not-owned word across the half cell -- conj(eikna) times the stored word on
    // the B side, eikna times it on the D side, in MEEP's connection spelling.
    float2 c_dy = c, b_dz = b;
    float2 a_dz = a, c_dx = c;
    float2 b_dx = b, a_dy = a;
__PHASE_X__
__PHASE_Y__
__PHASE_Z__

    // --- the curl (stepping._curl_from_operands): MEEP's left-to-right bracket
    // ((g1s - g1) + g2) - g2s with both partners; (g1s - g1) without g2; and
    // (-dtdx) * (g2s - g2) without g1 (step_generic.cpp:72-76, :104-108). DO NOT
    // regroup: each operation rounds once, per part.
    float2 curl0 = has_y ? (has_z ? c_mul_coefficient_left(dtdx, ((c_y - c_dy) + b_dz) - b_z)
                                  : c_mul_coefficient_left(dtdx, c_y - c_dy))
                         : (has_z ? c_mul_coefficient_left(-dtdx, b_z - b_dz) : float2(0.0f, 0.0f));
    float2 curl1 = has_z ? (has_x ? c_mul_coefficient_left(dtdx, ((a_z - a_dz) + c_dx) - c_x)
                                  : c_mul_coefficient_left(dtdx, a_z - a_dz))
                         : (has_x ? c_mul_coefficient_left(-dtdx, c_x - c_dx) : float2(0.0f, 0.0f));
    float2 curl2 = has_x ? (has_y ? c_mul_coefficient_left(dtdx, ((b_x - b_dx) + a_dy) - a_y)
                                  : c_mul_coefficient_left(dtdx, b_x - b_dx))
                         : (has_y ? c_mul_coefficient_left(-dtdx, a_y - a_dy) : float2(0.0f, 0.0f));

    // --- ownership mask (stepping._mask_non_owned_cells) ------------------------
    // Writes +0.0 to BOTH planes: the array path assigns a complex zero.
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

    // --- the two PML forms, per structure chunk (stepping._apply_curl_regions) --
    // dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    // sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y). A target
    // whose dsigu membership holds steps the split-field recurrence; any other
    // steps the dsig-only form, which is the recurrence's first stage with fu := f,
    // and leaves fu unread and unwritten. Every product is one per part.
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];
    const bool du0 = __DU0__, du1 = __DU1__, du2 = __DU2__;

    float2 p0 = du0 ? u0[ii] : f0[ii];
    float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_y) - curl0, si_y);
    float2 v0 = du0 ? c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) - p0, si_z) : n0;

    float2 p1 = du1 ? u1[ii] : f1[ii];
    float2 n1 = c_mul_field_left(c_mul_field_left(p1, km_z) - curl1, si_z);
    float2 v1 = du1 ? c_mul_field_left((c_mul_field_left(f1[ii], km_x) + n1) - p1, si_x) : n1;

    float2 p2 = du2 ? u2[ii] : f2[ii];
    float2 n2 = c_mul_field_left(c_mul_field_left(p2, km_x) - curl2, si_x);
    float2 v2 = du2 ? c_mul_field_left((c_mul_field_left(f2[ii], km_y) + n2) - p2, si_y) : n2;

    if (du0) { u0[ii] = n0; }
    if (du1) { u1[ii] = n1; }
    if (du2) { u2[ii] = n2; }
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
"""

#: Per axis, the unshifted operands a term differences along it, as (connected
#: register, stored register): the B/D curl terms' (g1, g1 axis) and (g2, g2 axis)
#: pairs, ``shaders.CURL_OPERANDS`` regrouped by axis.
_PHASED_OPERANDS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "x": (("c_dx", "c"), ("b_dx", "b")),
    "y": (("c_dy", "c"), ("a_dy", "a")),
    "z": (("b_dz", "b"), ("a_dz", "a")),
}


def _phase_block(axis: str, kind: int) -> str:
    """One axis's Bloch connection of the unshifted operands, on plane 0.

    ``kind`` is :data:`CONNECT_NONE` (no line: the read is the stored word),
    :data:`CONNECT_NEGATE` (``-x``, a sign-bit operation per part) or
    :data:`CONNECT_PHASE` (``c_mul(word, phase)``, MEEP's unfused product). The same
    plane serves both sub-steps -- stored cell 0 -- and the host chose the factor
    (``conj(eikna)`` for B, ``eikna`` for D) in :func:`phase_arguments`.
    """
    index, phase = {"x": ("i", "px"), "y": ("j", "py"), "z": ("k", "pz")}[axis]
    kind = int(kind)
    if kind == CONNECT_NONE:
        return (f"    // axis {axis}: no connection (unphased, or a phase within "
                f"1e-13 of +1: a copy)")
    if kind not in CONNECT_KINDS:
        raise ValueError(f"connection kind must be one of {CONNECT_KINDS}, got {kind!r}")
    lines = [f"    if ({index} == 0) {{"]
    for connected, stored in _PHASED_OPERANDS[axis]:
        value = (f"-{stored}" if kind == CONNECT_NEGATE
                 else f"c_mul({stored}, {phase})")
        lines.append(f"        {connected} = {value};")
    lines.append("    }")
    return "\n".join(lines)


def bloch_curl_source(codes: Sequence[int], backward: bool,
                      phased: Sequence[int],
                      contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised complex ``bloch_pml_curl_step`` source.

    ``codes`` is the per-axis PERIODIC/METALLIC triple, ``backward`` selects
    ``step_D``'s negated strides over ``step_B``'s forward ones, ``phased`` is the
    per-axis connection kind (:data:`CONNECT_KINDS`, :func:`phase_arguments`). All
    three are baked into the string;
    the chunk division is NOT, it is the bound record, so the specialisation set does
    not grow with the grid.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3 or len(phased) != 3:
        raise ValueError(f"codes and phased must be per-axis triples, got "
                         f"{codes!r} / {phased!r}")
    for axis, (code, flag) in enumerate(zip(codes, phased)):
        if flag not in CONNECT_KINDS:
            raise ValueError(f"axis {axis} connection kind {flag!r} is not one of "
                             f"{CONNECT_KINDS}")
        if flag and code != templates.PERIODIC:
            raise ValueError(
                f"axis {axis} carries a Bloch phase but its ghost rule is "
                f"METALLIC; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same pairing)")
    flags = shaders.curl_membership_flags(backward)
    return templates.substitute(_CURL_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_BINDING__": _chunk_bounds.binding(CURL_RECORD_BUFFER),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__BOUNDS__": shaders.curl_bounds_block(backward),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__PHASE_X__": _phase_block("x", phased[0]),
        "__PHASE_Y__": _phase_block("y", phased[1]),
        "__PHASE_Z__": _phase_block("z", phased[2]),
        "__MASK__": templates.ownership_mask(codes, backward,
                                             zero=templates.COMPLEX_ZERO),
        "__DU0__": flags[0],
        "__DU1__": flags[1],
        "__DU2__": flags[2],
    })


# ---------------------------------------------------------------------------
# The constitutive kernel
# ---------------------------------------------------------------------------

_CONSTITUTIVE_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

__RECORD_STRUCT__

kernel void bloch_constitutive_step(
    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device float2*       w0      [[buffer(3)]],
    device float2*       w1      [[buffer(4)]],
    device float2*       w2      [[buffer(5)]],
    device const float2* g0      [[buffer(6)]],
    device const float2* g1      [[buffer(7)]],
    device const float2* g2      [[buffer(8)]],
    device const float*  e0      [[buffer(9)]],
    device const float*  e1      [[buffer(10)]],
    device const float*  e2      [[buffer(11)]],
    device const float*  kp0     [[buffer(12)]],
    device const float*  km0     [[buffer(13)]],
    device const float*  kp1     [[buffer(14)]],
    device const float*  km1     [[buffer(15)]],
    device const float*  kp2     [[buffer(16)]],
    device const float*  km2     [[buffer(17)]],
    constant uint&       nx      [[buffer(18)]],
    constant uint&       ny      [[buffer(19)]],
    constant uint&       nz      [[buffer(20)]],
    constant uint&       n_elem  [[buffer(21)]],
    __RECORD_BINDING__,
    uint idx [[thread_position_in_grid]])
{
__GUARD__

__DECODE__

    // --- the chunk record: each component's own-axis membership at this cell -----
__BOUNDS__

    // NO GHOST RULE, NO OWNERSHIP MASK AND NO PHASE: the constitutive sub-step reads
    // no neighbour and never sees a wrap. The coefficient index is on the
    // component's OWN axis -- kps_x/kms_x for target 0, _y for 1, _z for 2
    // (H_CONSTITUTIVE_TERMS / E_CONSTITUTIVE_TERMS). That is MEEP's dsigw, NOT the
    // dsig/dsigu cycle the curl recurrence uses.
    float kp_0 = kp0[i], km_0 = km0[i];
    float kp_1 = kp1[j], km_1 = km1[j];
    float kp_2 = kp2[k], km_2 = km2[k];
    const bool dw0 = __DW0__, dw1 = __DW1__, dw2 = __DW2__;

    // --- component 0 -----------------------------------------------------------
    // On a dsigw cell `prev` is read BEFORE `fw` is written (the array path copies
    // fw first) and the right-hand side is formed first: f + ((kps*src) -
    // (kms*prev)), MEEP's step_update_EDHB (step_generic.cpp:689, :697), each product
    // one per part. Elsewhere the word is src itself, an output select, and fw is not
    // touched. inv_eps is a float32 volume indexed by the COMPLEX CELL index.
    float2 prev0 = dw0 ? w0[ii] : float2(0.0f, 0.0f);
    float2 src0 = __SRC0__;
    if (dw0) { w0[ii] = src0; }
    float2 a0 = src0;
    if (dw0) { a0 = f0[ii] + (c_mul_coefficient_left(kp_0, src0) - c_mul_coefficient_left(km_0, prev0)); }
    f0[ii] = a0;

    // --- component 1 -----------------------------------------------------------
    float2 prev1 = dw1 ? w1[ii] : float2(0.0f, 0.0f);
    float2 src1 = __SRC1__;
    if (dw1) { w1[ii] = src1; }
    float2 a1 = src1;
    if (dw1) { a1 = f1[ii] + (c_mul_coefficient_left(kp_1, src1) - c_mul_coefficient_left(km_1, prev1)); }
    f1[ii] = a1;

    // --- component 2 -----------------------------------------------------------
    float2 prev2 = dw2 ? w2[ii] : float2(0.0f, 0.0f);
    float2 src2 = __SRC2__;
    if (dw2) { w2[ii] = src2; }
    float2 a2 = src2;
    if (dw2) { a2 = f2[ii] + (c_mul_coefficient_left(kp_2, src2) - c_mul_coefficient_left(km_2, prev2)); }
    f2[ii] = a2;
}
"""

#: The constitutive product per side. The H side reads B directly (mu = 1 is baked
#: into the array path too: ``stepping.update_H`` passes ``fields.Bx``); the E side
#: reads ``D * inv_eps``, the array path's ``part_products.scale(inverse_epsilon,
#: source)``: one float32 product per part. Multiplication is commutative, so the
#: helper's operand order moves no bit.
_CONSTITUTIVE_PRODUCTS: Dict[str, Tuple[str, str, str]] = {
    "H": ("g0[ii]", "g1[ii]", "g2[ii]"),
    "E": ("c_mul_field_left(g0[ii], e0[ii])",
          "c_mul_field_left(g1[ii], e1[ii])",
          "c_mul_field_left(g2[ii], e2[ii])"),
}


def bloch_constitutive_source(side: str,
                              contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised complex ``bloch_constitutive_step`` source for one side.

    The Yee sub-lattice is NOT chosen here. The host binds ``kps_a``/``kms_a`` for H
    and ``kps_a_h``/``kms_a_h`` for E (stepping.py:948 vs :1015), so the kernel never
    chooses and cannot choose wrong — but the host can, which is why the gate
    carries a swap mutation for it.

    The three inverse-epsilon buffers stay in the signature on both sides so the
    buffer layout is one layout. On the H side the plan binds float32 placeholders
    and the substituted product never reads them. The chunk record binds last, at
    :data:`CONSTITUTIVE_RECORD_BUFFER`.
    """
    if side not in _CONSTITUTIVE_PRODUCTS:
        raise ValueError(f"side must be one of {tuple(_CONSTITUTIVE_PRODUCTS)}, "
                         f"got {side!r}")
    products = _CONSTITUTIVE_PRODUCTS[side]
    flags = shaders.constitutive_membership_flags(side)
    return templates.substitute(_CONSTITUTIVE_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_BINDING__": _chunk_bounds.binding(CONSTITUTIVE_RECORD_BUFFER),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__BOUNDS__": shaders.constitutive_bounds_block(side),
        "__DW0__": flags[0],
        "__DW1__": flags[1],
        "__DW2__": flags[2],
        "__SRC0__": products[0],
        "__SRC1__": products[1],
        "__SRC2__": products[2],
    })


def split_plane_curl_signature() -> str:
    """The re/im-SPLIT curl signature, which must FAIL to compile. 36 bindings.

    This is not a strawman and it is not dead code: it is the evidence that
    ``float2`` volumes are FORCED by the 31-binding ceiling rather than chosen. The
    gate compiles this and requires the compile error, so a later edit that "tidies"
    the complex volumes into separate real and imaginary buffers is refused by a
    measurement rather than by a comment.
    """
    planes = [f"    device float* {name}{part} [[buffer({index})]]"
              for index, (name, part) in enumerate(
                  (f"{stem}{n}", part)
                  for stem in ("f", "u", "g") for n in range(3)
                  for part in ("re", "im"))]
    tail_start = len(planes)
    coefficients = [f"    device const float* c{n} [[buffer({tail_start + n})]]"
                    for n in range(6)]
    tail_start += 6
    scalars = [f"    constant uint& s{n} [[buffer({tail_start + n})]]" for n in range(4)]
    tail_start += 4
    scalars.append(f"    constant float& dtdx [[buffer({tail_start})]]")
    tail_start += 1
    phases = [f"    constant float& p{n} [[buffer({tail_start + n})]]" for n in range(6)]
    tail_start += 6
    record = [f"    constant int* cb [[buffer({tail_start})]]"]
    arguments = ",\n".join(planes + coefficients + scalars + phases + record)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void split_plane_curl(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    f0re[idx] = f0im[idx];\n}\n")


def split_plane_binding_count() -> int:
    """How many buffers the split form needs -- 36, above the measured ceiling."""
    return 9 * 2 + 6 + 5 + 6 + 1


def compile_bloch_curl(codes: Sequence[int], backward: bool, phased: Sequence[int],
                       contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(bloch_curl_source(codes, backward, phased,
                                            contract)).bloch_pml_curl_step


def compile_bloch_constitutive(side: str,
                               contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(bloch_constitutive_source(
        side, contract)).bloch_constitutive_step


def enumerate_complex_sources(contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit, keyed by a stable label.

    The connection triple is enumerated only over the axes a given boundary triple
    can legally phase -- a metallic axis cannot carry one -- so the count is the
    REACHABLE set rather than the Cartesian product, and a label that cannot be built
    is not fingerprinted as if it could.
    """
    out: Dict[str, str] = {}
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for cx in (templates.PERIODIC, templates.METALLIC):
            for cy in (templates.PERIODIC, templates.METALLIC):
                for cz in (templates.PERIODIC, templates.METALLIC):
                    codes = (cx, cy, cz)
                    for phased in _reachable_phase_flags(codes):
                        label = (f"bloch_pml_curl_step/{name}/{cx}{cy}{cz}/"
                                 f"ph{phased[0]}{phased[1]}{phased[2]}")
                        out[label] = bloch_curl_source(codes, backward, phased, contract)
    for side in ("H", "E"):
        out[f"bloch_constitutive_step/{side}"] = (
            bloch_constitutive_source(side, contract))
    return out


def _reachable_phase_flags(codes: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    options = [CONNECT_KINDS if code == templates.PERIODIC else (CONNECT_NONE,)
               for code in codes]
    return tuple((x, y, z) for x in options[0] for y in options[1] for z in options[2])


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors `coverage._grid_reasons` so the two can be diffed.
# Clause 2 is INVERTED (this product REQUIRES complex64 storage where every shipped
# Metal kernel requires real) and clause 7 (k = 0) is REPLACED by the per-axis
# phase-consistency clause: has_bloch is ADMITTED, and with no phase anywhere the
# kernel must reduce to the plain complex path, which the gate's unphased rows
# measure. `_layout_reasons`/`_volume_reasons` pin float32, so the complex twins
# below exist rather than a widened shared check.


def _complex_volume_reasons(label: str, array: Any, shape: Sequence[int]) -> List[str]:
    """Shape, dtype and contiguity of one COMPLEX volume, reported by name."""
    out: List[str] = []
    if tuple(getattr(array, "shape", ())) != tuple(shape):
        out.append(f"{label} shape {tuple(getattr(array, 'shape', ()))!r} != grid "
                   f"shape {tuple(shape)!r}")
    if str(getattr(array, "dtype", None)) != "complex64":
        out.append(f"{label} dtype {getattr(array, 'dtype', None)} is not complex64 "
                   f"(this product binds complex storage as float2 volumes)")
    flags = getattr(array, "flags", None)
    if not bool(getattr(flags, "c_contiguous", False)):
        out.append(f"{label} is not C-contiguous")
    return out


def _complex_layout_reasons(fields: Any, shape: Sequence[int],
                            names: Sequence[str]) -> List[str]:
    """Every named field volume must be complex64, C-contiguous and ``grid.shape``.

    The element bound is on COMPLEX CELLS, not words: the kernel indexes ``float2``
    and the dispatch is sized from a complex64 tensor's own element count, so the
    int32 flat index never doubles. That is the one bound the Triton twin has to
    halve and this one does not — a direct consequence of binding ``float2`` instead
    of a word view.
    """
    out: List[str] = []
    if len(shape) != 3:
        return [f"grid shape {tuple(shape)!r} is not three-dimensional"]
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if total >= 2 ** 31:
        out.append(f"{total} complex cells exceeds the kernel's int32 index range")
    for name in names:
        array = getattr(fields, name, None)
        if array is None:
            continue  # Already reported by the allocation clause.
        out.extend(_complex_volume_reasons(name, array, shape))
    return out


def _complex_grid_reasons(fields: Any, pml: Any, grid: Any, residency: Any) -> List[str]:
    """The clauses both complex predicates share."""
    reasons: List[str] = []

    # 1. The Metal backend, the host array module the mirrors copy from, and the
    #    resolved subnormal policy. IMPORTED, not re-spelled: this is the same
    #    backend question the shipped predicates ask, and asking it twice in two
    #    places is how the two answers drift.
    reasons.extend(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))

    # 2 (INVERTED). Complex64 storage REQUIRED. A real run belongs to the shipped
    #    real-field kernels; real storage bound as float2 is the wrong-stride wrong
    #    answer in the other direction. Either force_complex_fields or a nonzero
    #    k_point puts a run here (the array path itself raises on real storage with
    #    a phase, stepping.py:1905-1908); the dtype is verified by the layout clause.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32 (neither force_complex_fields nor a nonzero "
            "k_point): a real run belongs to the shipped real-field kernels")

    # 3. An absorber that actually absorbs: this is the split-field product only.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the complex "
                       "split-field path only)")

    # 4. Only the two ghost rules the curl kernel writes; a fold or the cylindrical
    #    axis also changes the stored extent for the constitutive side.
    kinds = _boundary_kinds(grid, pml if (pml is not None
                                          and getattr(pml, "is_active", False))
                            else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside "
                               f"{COVERED_BOUNDARIES}")
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 5 (REPLACES clause 7). Per-axis phase consistency. A phased axis must resolve
    #    PERIODIC (stepping._bloch_phases raises there, S:2346-2360; re-checked
    #    because a mis-baked phase flag is a PLANE OF WRONG VALUES, not a crash) and
    #    a metallic axis must carry a k component of exactly 0. has_bloch FALSE IS
    #    ADMITTED: with every phase flag 0 the kernel emits no rotation at all and
    #    must reduce to the plain complex path, and the gate's unphased rows are
    #    that reduction measured. AN ABSORBER ON A PHASED AXIS IS ADMITTED — settled
    #    on the array path by measurement rather than argument (S:2322-2337, 2.82e-07
    #    against CPU MEEP), and this predicate does not re-litigate it.
    #
    #    The phase is read DEFENSIVELY rather than through `_call`: an unreadable
    #    phase is not an unphased one, and admitting one would let
    #    `bloch_phase_table` raise inside the plan builder, violating its
    #    None-only refusal contract.
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if kinds is not None:
        phase_reader = getattr(grid, "bloch_phase", None)
        if not callable(phase_reader):
            reasons.append("grid.bloch_phase is missing or not callable; an "
                           "unreadable phase table is not an unphased one")
        for axis, kind in enumerate(kinds):
            if callable(phase_reader):
                try:
                    phase = phase_reader(axis)
                except Exception as exc:  # noqa: BLE001 - unreadable is refused
                    reasons.append(f"grid.bloch_phase({axis}) raised {exc!r}; an "
                                   f"unreadable phase is not an unphased one")
                else:
                    if phase is not None and kind != "periodic":
                        reasons.append(
                            f"axis {axis} carries Bloch phase {phase!r} but resolved "
                            f"to {kind!r}; only a periodic wrap can carry a phase")
            if kind == "metallic" and float(k_point[axis]) != 0.0:
                reasons.append(
                    f"axis {axis} is metallic with k component {k_point[axis]!r}; a "
                    f"PEC wall gives the axis no lattice vector for the phase "
                    f"(stepping._bloch_phases raises on the same pairing)")

    # 6. No conductivity on any curl target. Refused OUTRIGHT on a missing or
    #    non-callable reader — deliberately stricter than the shipped predicate's
    #    magnetic-flag fallback: inferring "no conductivity" from the ABSENCE of
    #    condfac_for is admission by attribute absence, and an electric conductivity
    #    has no fallback flag at all.
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    else:
        for target in CURL_TARGETS:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(f"a conductivity is installed on {target}; conductive "
                               f"complex stepping is a separate, unbuilt product")

    # 7. No dispersion, refused OUTRIGHT: complex ADE belongs to a later tranche.
    #    The shape clauses still run so an unreadable susceptibility is NAMED.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: complex-storage ADE is a "
                       "later tranche")
    reasons.extend(_susceptibility_reasons(fields))

    # 8. No instantaneous nonlinearity (the Pade factor REPLACES the constitutive
    #    product, S:970-971), no BFAST (:364-368), no special_kz beta (:727-784 —
    #    that is grid.beta and NOT a Bloch phase; refl-angular-kz2d.py and
    #    parallel-wvgs-force.py are beta runs and are refused here BY NAME so they
    #    land on the beta family rather than being silently admitted here).
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
                       f"(that is grid.beta, not a Bloch phase)")

    # 9. Stored E. Under an active PML always true (fields.py:678-700), but the
    #    clause is the REASON: an edit that makes stores_E optional under PML must
    #    be caught here.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return reasons


def complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                              residency: Any = None) -> Coverage:
    """May the complex split-field PML curl kernel step this (fields, pml, sub_step)?

    Off-diagonal epsilon is ADMITTED here — it is constitutive-only, its whole
    effect is inside ``update_E`` (stepping.py:1001-1008) — and refused by the E-side
    constitutive predicate. The same per-sub-step split ``coverage.py`` makes.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _complex_grid_reasons(fields, pml, grid, residency)

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        # BOTH sub-lattices, as pml_curl_coverage checks: the suffix the plan binds
        # is the sub-step's own, and a swap is the half-cell mutation the gate carries.
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                  residency: Any = None) -> Coverage:
    """May the complex constitutive kernel step ``update_H`` ('H') / ``update_E``?

    The E side additionally refuses everything that changes what ``source`` is: a
    registered polarization is already refused module-wide (clause 7), and an
    off-diagonal chi1inv row makes the sub-step non-element-wise (S:972-979), so it
    is refused HERE and admitted by the curl. With no polarization,
    ``displacement_minus_polarization`` returns the D array itself
    (fields.py:1096-1098), so binding D directly IS the array path's source.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _complex_grid_reasons(fields, pml, grid, residency)

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("an off-diagonal chi1inv row is installed (the row product "
                       "reads neighbours; this sub-step is element-wise)")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))

    if side == "E" and len(shape) == 3:
        # inv_eps stays float32 under complex storage (S:37-38, fields.py:1203-1204),
        # so the shipped float32 pin is exactly right and is reused unchanged.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))

    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The phase table and its argument encoding
# ---------------------------------------------------------------------------

def bloch_phase_table(grid: Any, kinds: Sequence[str]) -> Tuple[Optional[complex],
                                                               Optional[complex],
                                                               Optional[complex]]:
    """Per-axis Bloch factor ``eikna``, None on an unphased axis -- never ``1+0j``.

    Resolved through ``grid.bloch_phase`` exactly as ``stepping._bloch_phases``
    resolves it: the grid value is exp(2*pi*i*k*L) with the Brillouin edge EXACTLY
    -1+0j, and None means no connection. The non-periodic check mirrors the array
    path's raise and is unreachable behind an admitting predicate; it is kept so a
    caller who skips the predicate is refused loudly rather than handed a mis-baked
    connection.
    """
    if not getattr(grid, "has_bloch", False):
        return (None, None, None)
    phases: List[Optional[complex]] = []
    for axis in range(3):
        phase = grid.bloch_phase(axis)
        if phase is not None and kinds[axis] != "periodic":
            raise ValueError(
                f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                f"{kinds[axis]!r}; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same configuration)")
        phases.append(phase)
    return (phases[0], phases[1], phases[2])


def connection(phase: Optional[complex], backward: bool) -> Tuple[int, Tuple[float, float]]:
    """``(kind, (re, im))`` of one axis's connection on one curl side.

    ``stepping._bloch_ghost_operand``: the B side (forward differences) reads the word
    BELOW the stored window, ``thephase = 1.0 * conj(eikna)``; the D side (backward
    differences) the word ABOVE it, ``thephase = 1.0 * eikna``; both formed in double
    (``readback.cmul``, boundaries.cpp:203-226). MEEP's ``connect_phase`` then copies
    within 1e-13 of +1, negates within 1e-13 of -1, and otherwise multiplies by the
    phase rounded to the part dtype (boundaries.cpp:359-366, :495), which is the
    float32 pair returned for :data:`CONNECT_PHASE`. The other kinds return the
    placeholder ``(1.0, 0.0)``, which no emitted line reads.
    """
    if phase is None:
        return CONNECT_NONE, (1.0, 0.0)
    import numpy  # noqa: PLC0415

    eikna = complex(phase)
    factor = (eikna.real, eikna.imag) if backward else (eikna.real, -eikna.imag)
    pr, pi = cmul((1.0, 0.0), factor)
    if abs(pi) < CONNECT_TOLERANCE and abs(pr - 1.0) < CONNECT_TOLERANCE:
        return CONNECT_NONE, (1.0, 0.0)
    if abs(pi) < CONNECT_TOLERANCE and abs(pr + 1.0) < CONNECT_TOLERANCE:
        return CONNECT_NEGATE, (1.0, 0.0)
    return CONNECT_PHASE, (float(numpy.float32(pr)), float(numpy.float32(pi)))


def phase_arguments(phases: Sequence[Optional[complex]], backward: bool,
                    ) -> Tuple[Tuple[int, int, int], Tuple[Tuple[float, float], ...]]:
    """Encode the phase table as (per-axis connection kinds, three float2 pairs).

    One :func:`connection` per axis, for the curl side ``backward`` names. The kinds
    are what a curl source is specialised on (:func:`bloch_curl_source`); the pairs
    are what the plan binds as ``px``/``py``/``pz``.
    """
    kinds: List[int] = []
    values: List[Tuple[float, float]] = []
    for phase in phases:
        kind, value = connection(phase, backward)
        kinds.append(kind)
        values.append(value)
    return (kinds[0], kinds[1], kinds[2]), tuple(values)


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

def _complex_mirror(residency: Any, name: str, host: Any) -> Any:
    import numpy  # noqa: PLC0415

    return residency.mirror(name, host, dtype=numpy.complex64)


class ComplexPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free COMPLEX split-field PML curl sub-step.

    The complex volumes are bound as ``float2`` buffers and ``n_elem`` is the
    COMPLEX CELL count, which is also what the dispatch is sized from — the grid
    comes from the first tensor argument's element count, and a complex64 tensor's
    element count is its cell count. Binding a float32 word view instead would size
    the grid at twice the cells and step every cell twice.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc", "phased",
                 "phase_values", "residency", "volumes", "chunk_bounds")

    REPR_FIELDS = ("sub_step", "shape", "bc", "phased")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, phased, phase_values, residency: Any,
                 targets, auxiliaries, sources,
                 coefficients, functions: Dict[str, Any],
                 volumes: Sequence[str], *, chunk_bounds: Any) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # (the float32-rounded scalar on each plane) and Metal binds a Python float
        # into `constant float&` correctly rounded — measured, and the same clean
        # scalar ABI the real tranche certified.
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.residency = residency
        self.volumes = tuple(volumes)
        # THE CHUNK RECORD BINDS LAST (CURL_RECORD_BUFFER), checked against the plan's
        # shape and uploaded once as a plan-owned int32 vector, as the real curl plan
        # binds it (launch.PmlCurlPlan).
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, f"ComplexPmlCurlPlan({sub_step})")
        super().__init__(functions, (
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem, self.dtdx)
            + self.phase_values
            + (_chunk_bounds.tensor(self.chunk_bounds, _plan_device(residency)),)))


def _plan_device(residency: Any) -> str:
    """The device the plan's residency DECLARES, which the record tensor joins."""
    return str(getattr(residency, "device", "mps"))


def _curl_functions(codes, backward: bool, phased,
                    contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_bloch_curl(codes, backward, phased, mode)
            for mode in contract_variants}


def plan_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                          residency: Any = None,
                          contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,), *,
                          chunk_bounds: Any) -> Optional[ComplexPmlCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly. ``chunk_bounds`` is the record of the division the run steps
    with (the composer's ``StepContext.chunk_bounds``) and has no default.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not complex_pml_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    codes = [1 if kind == "metallic" else 0 for kind in kinds]
    flags, values = phase_arguments(bloch_phase_table(grid, kinds),
                                    backward=bool(spec["backward"]))
    targets = [_complex_mirror(residency, n, getattr(fields, n))
               for n in spec["targets"]]
    auxiliaries = [_complex_mirror(residency, "fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [_complex_mirror(residency, n, getattr(fields, n))
               for n in spec["sources"]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return ComplexPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, flags, values,
        residency, targets, auxiliaries, sources, coefficients,
        _curl_functions(codes, spec["backward"], flags, contract_variants),
        volumes, chunk_bounds=chunk_bounds)


def plan_complex_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                      flat: Dict[str, Any], codes,
                                      phases: Sequence[Optional[complex]],
                                      dtdx: float, residency: Any,
                                      functions: Optional[Dict[str, Any]] = None,
                                      contract_variants: Sequence[str] = (
                                          shaders.CONTRACT_OFF,),
                                      *, chunk_bounds: Any) -> ComplexPmlCurlPlan:
    """Build a plan from bare host arrays — the gate's and benchmark's route.

    ``phases`` is the per-axis ``Optional[complex]`` table of ``eikna``; the side's
    factor and connection kind are chosen HERE, per sub-step, exactly as the engine
    route chooses them (:func:`phase_arguments`). No predicate runs: the caller is a
    harness that constructed the configuration deliberately, including the
    deliberately wrong ones. ``chunk_bounds`` is REQUIRED: the record of the division
    the harness's reference steps with.

    ``functions`` is the MUTATION SEAM. The gate compiles a deliberately broken copy
    of the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING — every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    flags, values = phase_arguments(tuple(phases), backward=bool(spec["backward"]))
    targets = [_complex_mirror(residency, n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [_complex_mirror(residency, "fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [_complex_mirror(residency, n, arrays[n]) for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return ComplexPmlCurlPlan(
        sub_step, shape, dtdx, codes, flags, values, residency,
        targets, auxiliaries, sources, coefficients,
        functions if functions is not None
        else _curl_functions(codes, spec["backward"], flags,
                             contract_variants),
        volumes, chunk_bounds=chunk_bounds)


class ComplexConstitutivePlan(KernelPlan):
    """A launchable, allocation-free COMPLEX ``dsigw`` constitutive sub-step.

    The Yee sub-lattice is chosen HERE and nowhere else: ``kps_a``/``kms_a`` for the
    H side, ``kps_a_h``/``kms_a_h`` for the E side (stepping.py:948 vs :1015). The
    kernel takes six coefficient pointers and never asks which lattice they came
    from, so a swap on this line is a silent half-cell error in the absorber profile
    — converged, smooth, and wrong. The gate carries a mutation for exactly it.
    """

    __slots__ = ("side", "shape", "n_elem", "residency", "volumes",
                 "chunk_bounds")

    REPR_FIELDS = ("side", "shape")

    #: The kernel steps the forms alone and does not start H and ``f_w`` from B on a
    #: run's first ``update_H`` (``stepping.separate_h_from_b``); the composer reads
    #: this to keep that slot on the array path where the start can move a word.
    starts_h_from_b = False

    def __init__(self, side: str, shape, residency: Any, targets,
                 auxiliaries, sources, inverse_epsilon, coefficients,
                 functions: Dict[str, Any], volumes: Sequence[str], *,
                 chunk_bounds: Any) -> None:
        self.side = side
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.residency = residency
        self.volumes = tuple(volumes)
        # The H side has no inverse epsilon. The three buffers stay in the signature
        # so the layout is one layout, and REAL placeholders are bound: the
        # substituted product never reads them, but a `device const float*` argument
        # still has to bind, and a complex tensor there would be the wrong TYPE
        # rather than merely an unread one.
        placeholders = tuple(coefficients[:3] if inverse_epsilon is None
                             else inverse_epsilon)
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, f"ComplexConstitutivePlan({side})")
        super().__init__(functions, (
            tuple(targets) + tuple(auxiliaries) + tuple(sources) + placeholders
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem,
               _chunk_bounds.tensor(self.chunk_bounds, _plan_device(residency)))))


def _constitutive_functions(side: str,
                            contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_bloch_constitutive(side, mode)
            for mode in contract_variants}


def plan_complex_constitutive(fields: Any, pml: Any, side: str, residency: Any = None,
                              contract_variants: Sequence[str] = (
                                  shaders.CONTRACT_OFF,), *,
                              chunk_bounds: Any) -> Optional[ComplexConstitutivePlan]:
    """Build a complex constitutive plan from the engine's objects, or None.

    ``chunk_bounds`` as :func:`plan_complex_pml_curl` takes it.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not complex_constitutive_coverage(fields, pml, side, residency).covered:
        return None

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    targets = [_complex_mirror(residency, n, getattr(fields, n))
               for n in spec["targets"]]
    auxiliaries = [_complex_mirror(residency, n, getattr(fields, n))
                   for n in spec["aux"]]
    sources = [_complex_mirror(residency, n, getattr(fields, n))
               for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True)
         for n in spec["targets"]] if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ComplexConstitutivePlan(
        side, fields.grid.shape, residency, targets, auxiliaries,
        sources, inverse_epsilon, coefficients,
        _constitutive_functions(side, contract_variants), volumes,
        chunk_bounds=chunk_bounds)


def plan_complex_constitutive_from_arrays(
        side: str, arrays: Dict[str, Any], flat: Dict[str, Any],
        residency: Any, functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        suffix_key: str = "", *, chunk_bounds: Any) -> ComplexConstitutivePlan:
    """Build a complex constitutive plan from bare host arrays — the gate's route.

    ``suffix_key`` only namespaces the coefficient mirrors so a harness can bind the
    OTHER Yee sub-lattice under a different name in the same residency; it does not
    choose a lattice. Choosing is the caller's, which is what makes the half-cell
    swap a HOST mutation rather than a kernel one. ``chunk_bounds`` is required, as
    :func:`plan_complex_pml_curl_from_arrays` requires it.
    """
    spec = CONSTITUTIVE_SIDES[side]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    targets = [_complex_mirror(residency, n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [_complex_mirror(residency, n, arrays[n]) for n in spec["aux"]]
    sources = [_complex_mirror(residency, n, arrays[n]) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, arrays["inv_eps_" + n], constant=True)
         for n in spec["targets"]] if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}:{side}{suffix_key}",
                         flat[f"{stem}_{axis}"], constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ComplexConstitutivePlan(
        side, shape, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        functions if functions is not None
        else _constitutive_functions(side, contract_variants), volumes,
        chunk_bounds=chunk_bounds)


# ---------------------------------------------------------------------------
# Registration — WIRED, as of tranche 2
# ---------------------------------------------------------------------------

#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}


def _curl_arm_coverage(context: Any, slot: str) -> Coverage:
    return complex_pml_curl_coverage(context.fields, context.pml, slot,
                                     context.residency)


def _curl_arm_plan(context: Any, slot: str) -> Any:
    return plan_complex_pml_curl(context.fields, context.pml, slot,
                                 context.residency, context.contract_variants,
                                 chunk_bounds=context.chunk_bounds)


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return complex_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency)


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_complex_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants, chunk_bounds=context.chunk_bounds)


def register_arms() -> Tuple[Any, ...]:
    """Register this family's four arms, WIRED.

    THEY WERE ``wired=False`` FOR ONE TRANCHE, and the flip is what the composition
    sweep bought. The deferral was never about doubt in the kernel — the byte gate
    had certified it — but about whether wiring it could change the answer on a
    configuration some OTHER family owns. That is a question about predicates, not
    about arithmetic, and it was settled by constructing the configurations and
    calling every predicate on every slot rather than by reading clause lists.

    WHAT MAKES THIS THE ONLY ADMITTER, in both directions and by name: clause 2 is
    INVERTED here (complex64 storage REQUIRED, so the shipped real kernels' clause 2
    refuses everything this admits), and clause 8 refuses ``grid.beta`` by name, so a
    complex BETA run lands on :mod:`.special_kz`'s complex arm rather than being
    silently taken here. Neither is inherited: both are clauses this module states.
    """
    from . import arms  # noqa: PLC0415

    registered = [
        arms.register(family=FAMILY, slot=slot, label="complex/Bloch",
                      coverage=_curl_arm_coverage, plan=_curl_arm_plan,
                      prefix="complex/Bloch: ", noun="complex/Bloch PML curl",
                      wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="complex/Bloch",
                      coverage=_constitutive_arm_coverage,
                      plan=_constitutive_arm_plan,
                      prefix="complex/Bloch: ",
                      noun="complex/Bloch constitutive", wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    return tuple(registered)


#: Registered ON IMPORT, once. The registry refuses a duplicate (family, slot,
#: label) by design — a second registration is almost always a module reached under
#: two names, and silently keeping one would make WHICH KERNEL RUNS depend on import
#: order — so registration lives here rather than at every caller.
ARMS = register_arms()
