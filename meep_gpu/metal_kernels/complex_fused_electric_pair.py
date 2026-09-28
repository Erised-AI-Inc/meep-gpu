"""The COMPLEX/BLOCH fused Metal pair: complex ``step_D`` welded into ``update_E``.

THE D-SIDE TWIN OF :mod:`.complex_fused_magnetic_pair`, and the largest cell the
Metal fusion board had left unbuilt: 16 corpus instances, the biggest
``would_fit_not_built`` entry on either seam. Its B/H sibling shipped 16/16 on the
mirror cell; this is the same construction on the other half of the step.

ONE DISPATCH FOR ALL THREE COMPONENTS. The separate composition for this
configuration is TWO device dispatches (the certified complex curl, then the
certified complex constitutive) plus one host wall pass on a walled run; this is
ONE, and the complex displacement never leaves a register.

EVERYTHING HERE IS TRANSCRIBED, and the two halves are not re-spelled at all — they
are SPLICED FROM THE CERTIFIED COMPLEX EMITTERS' OWN OUTPUT:

* the D curl half — :func:`.complex_fields.bloch_curl_source` with ``backward=True``
  (complex_fields.py:248-343), whose body is lifted verbatim by
  :func:`certified_curl_body`. Its ghost gather is :func:`.templates.ghost`, its
  cell-0 mask :func:`.templates.ownership_mask` and its wrap rotation
  :func:`.complex_fields._phase_block`, all reached through that emitter rather than
  through a second copy;
* the wall clear — ``stepping._zero_metal`` (stepping.py:2206-2247) restricted to
  ``D_COMPONENTS`` (stepping.py:184), carried inline over the OFF-DIAGONAL rows
  :data:`.fused_dispersive_pair._ZERO_METAL_ROWS` — IMPORTED rather than re-derived,
  for the reason :func:`.fused_electric_pair.zero_metal_mask` gives — and writing
  :data:`.templates.COMPLEX_ZERO` rather than ``0.0f``;
* the constitutive half — :func:`.complex_fields.bloch_constitutive_source` on side
  ``"E"`` (complex_fields.py:545-570), whose body is lifted verbatim by
  :func:`certified_constitutive_body`, itself the transcription of
  ``stepping.update_E`` and ``stepping._apply_constitutive_pml``
  (stepping.py:2130-2143).

Because both halves are LIFTED rather than retyped, "the fused arithmetic is the
certified complex arithmetic" is a property of the construction that a reader can
check by running :func:`certified_curl_body`.

===========================================================================
THE SIGNATURE — 30 pointers plus one packed struct, EXACTLY ON THE CEILING
===========================================================================

    3 D  +  3 fu_D  +  3 H        (float2, the curl half)
  + 3 E  +  3 f_w_E               (float2, the constitutive half)
  + 3 inverse epsilon             (float32 — see below)
  + 6 curl coefficients  +  6 constitutive coefficients   (float32)
  = 30 pointers, + one ``constant Params&`` = 31 bindings

31 IS :data:`.device.MAX_BUFFER_BINDINGS` — THE COUNT, NOT THE HEADROOM. One more
pointer of any kind is a compile error, measured on this toolchain (torch 2.10.0,
MPS) rather than asserted: a 15-``float2``/15-``float``/``Params`` probe COMPILES
at 31 and the same probe with one more ``float`` FAILS with ``'buffer' attribute
parameter is out of bounds: must be between 0 and 30``. That is why the eight
non-pointer arguments ride PACKED and why :data:`SEPARATE_SCALAR_BINDINGS` is
spelled as data.

**THE THREE INVERSE-EPSILON POINTERS ARE WHY THIS IS 30 AND THE MAGNETIC TWIN IS
27**, and the difference is physics rather than layout, exactly as it is between
:mod:`.fused_electric_pair` and :mod:`.fused_magnetic_pair`: ``update_H`` is
``H = B`` with mu = 1 baked into the array path, so the magnetic pair drops the
three volumes the certified constitutive keeps; ``update_E`` cannot, because its
product is ``source * inverse_epsilon_for(component)`` (stepping.py:1011-1013) with D
on the LEFT, and all three volumes are read.

**THE INVERSE EPSILON STAYS float32 UNDER COMPLEX STORAGE.** ``c_mul_field_left``
takes a ``float2`` and a ``float`` — one real coefficient per cell, applied to both
planes — so these three bind ``device const float*`` and are indexed by the COMPLEX
CELL index, never word-doubled. Had they widened with the fields this signature
would need 33 pointers and could not be built at all.

**THE COEFFICIENTS STAY float32 TOO, AND THAT IS LOAD-BEARING** — the invariant
:mod:`.complex_fused_magnetic_pair` records at stepping.py:41-50. Twelve
coefficient buffers, not twenty-four.

===========================================================================
THE TWO YEE SUB-LATTICES, AND WHY THEY ARE CROSSED HERE
===========================================================================

The D curl reads the INTEGER split-field coefficients ``kms_a``/``sinv_a`` and the
E constitutive the HALF-INTEGER ``kps_a_h``/``kms_a_h``
(``complex_fields.SUB_STEPS['step_D']['suffix'] == ''``;
``CONSTITUTIVE_SIDES['E']['half_integer'] is True``). That is the OPPOSITE pairing
to :mod:`.complex_fused_magnetic_pair`, where the curl takes the half-integer
lattice and the constitutive the integer one. The kernel takes twelve coefficient
pointers and never asks which lattice they came from, so a swap on either group is
a silent half-cell error in the absorber profile.

===========================================================================
THE SEAM, AND WHAT SITS INSIDE IT — measured from the driver, not assumed
===========================================================================

The driver runs FIVE passes between ``step_D`` and ``update_E``
(driver.py:3292-3302): ``step_D`` -> ELECTRIC SOURCES (:3294-3299) ->
``fill_symmetry_bc_D`` (:3300) -> ``zero_metal_D`` (:3301) ->
``fill_folded_far_ghosts_D`` (:3302) -> ``update_E`` (:3303).

* **the electric source slot — CARRIED, through the deposit repair.** See
  :data:`CARRIES_DEPOSIT_REPAIR`. This is the clause that decides whether the cell
  is worth anything: the census gives this pair 16 rows admitting both halves, and
  every one of them carries an electric source. The repair is not an optimisation
  here; it is the product. A MAGNETIC source is injected in the B/H half and does
  NOT disqualify the pair.
* **``zero_metal_D`` — CARRIED INLINE**, on the register, through
  :func:`zero_metal_mask`, and on this side it writes a COMPLEX zero. The rows are
  the D OFF-DIAGONAL — Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z — the exact
  complement of the B side's diagonal table, and a family that reused the magnetic
  twin's rows would clear the wrong two components on every walled run.
  ``fu_D`` is deliberately NOT masked: the array-path pass touches ``D_COMPONENTS``
  only (stepping.py:2250).
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` — REFUSED** through both
  halves' own "a mirror plane is active" clause
  (:func:`.complex_fields._complex_grid_reasons` clause 4), which is what makes both
  fill slots dead for every configuration this family admits. A FOLDED complex D/E
  seam is a different product and is NOT built here.

THE STRUCT FIELD ORDER IS THE MAGNETIC TWIN'S, AND FOR THE MEASURED REASON IT
GIVES: Metal aligns ``float2`` to 8 bytes, so with the scalars first the natural
44-byte host record puts every phase one word early and reads back as a plausible
complex number rather than as garbage. The three ``float2`` members come FIRST and
:func:`params_record_dtype` is the one home for the layout.

NOT WIRED AS AN ARM, for the reason every fused pair records: ``plan_step`` assigns
at most one arm per slot and this product spans three. It registers ``wired=False``
and reaches its two slots through ``launch.FUSED_PAIR_ARMS`` instead.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import complex_fields, shaders, templates
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .fused_dispersive_pair import _ZERO_METAL_ROWS
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as the clause
#: read before the repair existed. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring.
#:
#: TRUE FROM THE FIRST COMMIT OF THIS FAMILY, in the same edit as the
#: ``launch.FUSED_PAIR_ARMS`` row that lets the seam loop bracket it. Three things
#: hold, and none of them is new here:
#:
#: * THE ARITHMETIC ON COMPLEX STORAGE was measured for the magnetic twin --
#:   ``deposit_repair`` indexes and accumulates element-wise over one ``complex64``
#:   array per component (deposit_repair.py:118-129, :146-168) and needs no change --
#:   and ``test_deposit_repair.py``'s ``COMPLEX_CASES`` is that measurement.
#: * THE ELECTRIC BRANCH is the one :mod:`.fused_electric_pair` already drives, with
#:   the same ``'D'`` pair label and the same two wrappers.
#: * THE FOLD BOUNDARY DOES NOT APPLY. What held the two folded complex families at
#:   ``False`` is that a folded seam runs ``fill_symmetry_bc_D`` /
#:   ``fill_folded_far_ghosts_D`` AFTER the injection. This family REFUSES every
#:   mirrored axis by name in its own predicate below, so no configuration it admits
#:   has a mirror image to miss.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "complex_fused_electric_pair"

#: The sub-step slot this arm is registered on. It spans three; it holds a row on the
#: first, so a refusal is NAMED on the slot the fusion starts at.
SLOT = "step_D"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3292-3303). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The shipped binding count: 30 pointers + one packed ``constant Params&``. This is
#: MAX_BUFFER_BINDINGS exactly.
PACKED_BINDINGS = 31

#: What the same 30 pointers need with the five scalars and three phases bound
#: SEPARATELY, the way the certified 23-binding complex curl binds them. Over the
#: ceiling; :func:`refuted_separate_scalar_source` builds it and the gate requires
#: the compile failure.
SEPARATE_SCALAR_BINDINGS = 38

#: What the same kernel needs with the complex volumes bound as SEPARATE re/im
#: planes: 30 field planes + 3 inverse epsilon + 12 coefficient + 5 scalars + 6
#: phase floats. Far over the ceiling, which is the measurement behind "float2 is
#: FORCED, not preferred".
SPLIT_PLANE_BINDINGS = 56

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex fused electric pair binds {PACKED_BINDINGS} buffers and "
        f"Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}; the signature "
        f"cannot be built at all")

#: The host record's total size in bytes — the magnetic twin's layout, unchanged.
PARAMS_ITEMSIZE = 48

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "PACKED_BINDINGS", "PARAMS_ITEMSIZE",
    "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT", "SPLIT_PLANE_BINDINGS",
    "MetalComplexFusedElectricPairPlan", "certified_constitutive_body",
    "certified_curl_body", "compile_complex_fused_electric_pair",
    "complex_fused_electric_pair_source", "enumerate_sources",
    "metal_complex_fused_electric_pair_coverage", "params_record_dtype",
    "plan_metal_complex_fused_electric_pair", "refuted_separate_scalar_source",
    "register_arms", "split_plane_pair_signature", "zero_metal_mask",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
#
# THIRTY-ONE BINDINGS: 15 float2 volumes + 3 inverse epsilon + 12 coefficient
# vectors + 1 packed Params&. See the module docstring for the two signatures this
# refutes.

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE THREE float2 MEMBERS COME FIRST. Metal aligns float2 to 8 bytes, so with the
// scalars first this struct needs internal padding and the natural host record puts
// every phase one word early — measured on this toolchain, and it reads as a
// plausible complex number rather than as garbage. Phases first, no internal padding.
struct Params {
    float2 px; float2 py; float2 pz;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
};

kernel void complex_fused_electric_pair_step(
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
    // THE EIGHT PACKED ARGUMENTS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES,
    // once, before any of the lifted text runs. Everything below this line is then
    // character-for-character what `complex_fields.bloch_curl_source` and
    // `complex_fields.bloch_constitutive_source` emit, plus the wall clear.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float2 px = prm.px, py = prm.py, pz = prm.pz;

__BODY__
}
"""

#: The marker that separates a certified shader's signature from its body. Both
#: complex templates end their parameter list with this exact line, so ONE anchor
#: lifts either body and a template that stopped carrying it raises here rather than
#: splicing a truncated kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The certified complex curl body's last statement — the D store. The wall clear
#: goes IMMEDIATELY BEFORE it, because the array path stores D and only then
#: overwrites the wall plane; masking the register first is the same field and one
#: fewer pass. The ``u`` store precedes it and is NOT masked: ``zero_metal_D`` passes
#: ``D_COMPONENTS`` (stepping.py:2250) and the split-field auxiliary is not in it.
_CURL_STORE = "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n"

#: The certified constitutive body's decode prologue, which the curl body has already
#: emitted. Splicing both would redeclare ``ii``/``i``/``j``/``k`` and the kernel
#: would not compile — the lift drops exactly this much and no more. Dropping through
#: it also drops the second ``n_elem`` guard, which is the same guard.
_CONSTITUTIVE_PROLOGUE_END = "    int i   = plane / nyi;\n"


def certified_curl_body(codes: Sequence[int], phased: Sequence[int], expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The complex ``step_D`` curl kernel's BODY, lifted from the certified emitter.

    Not a transcription: this is :func:`.complex_fields.bloch_curl_source`'s own
    output with the ``#include``/helpers/signature preamble and the closing brace
    removed. ``backward`` is True and never a parameter — this pair is the D seam,
    and ``step_B``'s forward strides and unconjugated phase are a different product
    (:mod:`.complex_fused_magnetic_pair`).
    """
    source = complex_fields.bloch_curl_source(codes, True, phased, expansion,
                                              contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified complex curl source no longer carries the body anchor; "
            "this family lifts that body and would otherwise splice a truncated "
            "kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified complex curl source does not end with '}'")
    return body[: -len("}\n")]


def certified_constitutive_body(expansion: str,
                                contract: str = shaders.CONTRACT_OFF) -> str:
    """The complex ``update_E`` constitutive kernel's BODY, lifted and renamed.

    Two edits, and both are POINTER SPELLING rather than arithmetic — the same kind
    of rename the magnetic twin applies, and for the same reason: fused, ``f0`` would
    mean D in the curl half and E in the constitutive half in one scope.

    * the decode prologue is dropped (the curl half already emitted it);
    * the three targets ``f0``/``f1``/``f2`` become ``h0``/``h1``/``h2``.

    THE INVERSE EPSILON IS **NOT** RENAMED, and that is the whole reason this lift is
    two edits where :func:`.fused_electric_pair.certified_constitutive_body` needs
    three with a placeholder. That family renames its E target onto ``e``, which
    collides with the inverse epsilon and forces the crossing rename; this one takes
    the magnetic twin's ``h`` for the target, so ``e0``/``e1``/``e2`` keep the
    meaning the certified body gave them and the fused signature declares them under
    that same name. One fewer rename is one fewer place a volume can be swapped.

    THE SOURCE LINES ARE NOT EDITED HERE. ``float2 src0 = c_mul_field_left(g0[ii],
    e0[ii]);`` is left standing and :func:`complex_fused_electric_pair_source` is
    what turns it into the register — so the seam is one visible substitution in one
    place. That also resolves the one genuine name collision in this splice: ``g0``
    means H in the curl half and D in the certified E body, and after the seam
    substitution the constitutive half references ``g`` nowhere at all.
    """
    source = complex_fields.bloch_constitutive_source("E", expansion, contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified complex constitutive source no longer carries the body "
            "anchor")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(
            "the certified complex constitutive source does not end with '}'")
    body = body[: -len("}\n")]
    if _CONSTITUTIVE_PROLOGUE_END not in body:
        raise AssertionError(
            "the certified complex constitutive body no longer carries its decode "
            "prologue; the lift would redeclare the curl half's indices")
    body = body.split(_CONSTITUTIVE_PROLOGUE_END, 1)[1]
    for target in range(3):
        old, new = f"f{target}[ii]", f"h{target}[ii]"
        if old not in body:
            raise AssertionError(
                f"the certified complex constitutive body has no {old}")
        body = body.replace(old, new)
    return body


def zero_metal_mask(zero_metal: Sequence[bool],
                    zero: str = templates.COMPLEX_ZERO) -> str:
    """``stepping.zero_metal_D`` for all three targets, carried inline on the register.

    The array path stores D and then overwrites the wall plane (stepping.py:2247);
    this masks the value before it is stored and before the E half consumes it, which
    is the same field and one fewer pass.

    The rows are :data:`.fused_dispersive_pair._ZERO_METAL_ROWS` — the D OFF-DIAGONAL
    (Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z), IMPORTED rather than re-derived,
    for the reason :func:`.fused_electric_pair.zero_metal_mask` gives. The magnetic
    twin's table is the complement and a family that reused it here would clear the
    wrong two components on every walled run.

    The zero is a COMPLEX zero — both planes — because the array path assigns one
    (stepping.py:1943, :1949). The select spelling is
    :func:`.templates.ownership_mask`'s — ``flag ? zero : v`` — because that is the
    form already measured to deliver an exact ``0.0`` on this platform.
    """
    lines = [f"    v{target} = {flag} ? {zero} : v{target};"
             for target, axis, flag in _ZERO_METAL_ROWS if bool(zero_metal[axis])]
    return "\n".join(lines) or "    // no walled axis clears a D component"


def complex_fused_electric_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundaries, phases, walls, arm, contraction).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string here,
    for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a source
    and nothing else, so specialisation by substitution is what keeps the emitted
    arithmetic identical to the bodies this lifts.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if len(phased) != 3:
        raise ValueError(f"phased must be a per-axis triple, got {phased!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    # A phased METALLIC axis is refused by the certified emitter itself
    # (complex_fields.bloch_curl_source raises, mirroring stepping._bloch_phases
    # S:2346-2360). It is not re-checked here: one raise, in the emitter that owns
    # the rule.

    curl = certified_curl_body(codes, phased, expansion, contract)
    if _CURL_STORE not in curl:
        raise AssertionError(
            "the certified complex curl body no longer stores the three targets on "
            "one line; the wall clear has no anchor to sit in front of")
    head, tail = curl.split(_CURL_STORE, 1)
    curl = "".join((
        head,
        "    // --- zero_metal_D, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The OFF-DIAGONAL for D: Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z.\n"
        "    // A COMPLEX zero: the array path assigns one to both planes\n"
        "    // (S:1896, :1902).\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    electric = certified_constitutive_body(expansion, contract)
    # THE SEAM, and it is exactly three lines. The certified complex E body opens each
    # component with `float2 srcN = c_mul_field_left(gN[ii], eN[ii]);` — a RELOAD of
    # the complex displacement the curl just stored, scaled by the inverse
    # permittivity. Fused, the reload becomes the live register and NOTHING ELSE
    # MOVES: the inverse-epsilon factor stays on the right and D stays on the LEFT,
    # which is the order `c_mul_field_left` means and the order the array path writes
    # (stepping.py:1011-1013) — not a commutation this transcription is entitled to
    # make.
    for target in range(3):
        old = (f"    float2 src{target} = "
               f"c_mul_field_left(g{target}[ii], e{target}[ii]);\n")
        if old not in electric:
            raise AssertionError(
                f"the certified complex E constitutive body no longer reads its "
                f"source as {old.strip()!r}; the seam has no anchor")
        electric = electric.replace(
            old,
            f"    // THE SEAM: the register step_D just wrote, not a reload of "
            f"D{'xyz'[target]}.\n"
            f"    float2 src{target} = c_mul_field_left(v{target}, e{target}[ii]);\n")
    # THE CURL'S `g` IS H AND THE CONSTITUTIVE'S `g` WAS D. After the seam there must
    # be no `g` left in the electric half at all -- one surviving read would take the
    # magnetic field as a displacement, which is a smooth, plausible, entirely wrong
    # answer rather than a crash.
    for target in range(3):
        if f"g{target}[" in electric:
            raise AssertionError(
                f"the lifted complex E constitutive half still reads g{target}, "
                f"which in the fused signature is H{'xyz'[target]} and not "
                f"D{'xyz'[target]}")

    body = "".join((
        curl,
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml:2083) --\n",
        electric,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": body,
    })


def compile_complex_fused_electric_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, phases, walls, arm, mode)."""
    return compile_source(complex_fused_electric_pair_source(
        codes, phased, zero_metal, expansion,
        contract)).complex_fused_electric_pair_step


def enumerate_sources(expansion: str,
                      contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit, keyed by a stable label.

    The phase triple is enumerated only over the axes a given boundary triple can
    legally phase — a metallic axis cannot carry one — so the count is the REACHABLE
    set rather than the Cartesian product, and a label that cannot be built is not
    fingerprinted as if it could. The wall triple is enumerated in full: a wall is a
    property of the grid, not of the boundary code the curl bakes.
    """
    out: Dict[str, str] = {}
    for cx in (templates.PERIODIC, templates.METALLIC):
        for cy in (templates.PERIODIC, templates.METALLIC):
            for cz in (templates.PERIODIC, templates.METALLIC):
                codes = (cx, cy, cz)
                legal = [tuple(1 if flag else 0 for flag in phased)
                         for phased in _phase_triples(codes)]
                for phased in legal:
                    for wx in (False, True):
                        for wy in (False, True):
                            for wz in (False, True):
                                walls = (wx, wy, wz)
                                label = (f"step_D:{''.join(str(c) for c in codes)}:"
                                         f"{''.join(str(p) for p in phased)}:"
                                         f"{''.join('1' if w else '0' for w in walls)}")
                                out[label] = complex_fused_electric_pair_source(
                                    codes, phased, walls, expansion, contract)
    return out


def _phase_triples(codes: Sequence[int]) -> Sequence[Tuple[bool, bool, bool]]:
    """Every phase triple legal against one boundary triple. A metallic axis cannot."""
    legal: List[Tuple[bool, bool, bool]] = []
    for px in (False, True):
        for py in (False, True):
            for pz in (False, True):
                triple = (px, py, pz)
                if any(triple[axis] and codes[axis] == templates.METALLIC
                       for axis in range(3)):
                    continue
                legal.append(triple)
    return tuple(legal)


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The same 30 pointers with the eight non-pointer arguments bound SEPARATELY.

    38 bindings — :data:`SEPARATE_SCALAR_BINDINGS` — and the gate REQUIRES the
    compile failure. This is not a strawman: it is the way
    :func:`.complex_fields.bloch_curl_source` binds its own arguments at its
    23-binding size, so "packed" is a measurement on this signature rather than a
    house style.
    """
    volumes = [f"    device float2* q{index} [[buffer({index})]]"
               for index in range(15)]
    reals = [f"    device const float* r{index} [[buffer({index})]]"
             for index in range(15, 30)]
    scalars = [f"    constant uint& s{index} [[buffer({index})]]"
               for index in range(30, 34)]
    scalars.append("    constant float& dtdx [[buffer(34)]]")
    phases = [f"    constant float2& p{index} [[buffer({index})]]"
              for index in range(35, 38)]
    arguments = ",\n".join(volumes + reals + scalars + phases)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            + shaders.contraction_pragma(contract) + "\n"
            "kernel void separate_scalar_pair(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    q0[idx] = float2(dtdx, r15[idx]);\n}\n")


def split_plane_pair_signature() -> str:
    """The re/im-SPLIT signature, which must FAIL to compile. 56 bindings.

    Evidence that ``float2`` volumes are FORCED by the 31-binding ceiling rather than
    chosen, exactly as :func:`.complex_fields.split_plane_curl_signature` is for the
    certified curl. The gate compiles this and requires the compile error.
    """
    planes = [f"    device float* v{index} [[buffer({index})]]"
              for index in range(30)]
    reals = [f"    device const float* r{index} [[buffer({index})]]"
             for index in range(30, 45)]
    scalars = [f"    constant uint& s{index} [[buffer({index})]]"
               for index in range(45, 49)]
    scalars.append("    constant float& dtdx [[buffer(49)]]")
    phases = [f"    constant float& p{index} [[buffer({index})]]"
              for index in range(50, 56)]
    arguments = ",\n".join(planes + reals + scalars + phases)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void split_plane_pair(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    v0[idx] = v1[idx];\n}\n")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_complex_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span complex ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. That is the
    construction the magnetic twin uses, and it is what makes the expansion-probe
    clause, the complex-storage clause, the off-diagonal clause and the fold clause
    INHERITED rather than restated.
    """
    reasons: List[str] = []

    curl = complex_fields.complex_pml_curl_coverage(
        fields, pml, "step_D", residency, probe)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = complex_fields.complex_constitutive_coverage(
        fields, pml, "E", residency, probe)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM, and for this family it is the clause that decides whether the
    # cell is worth anything at all: every one of the 16 census rows admitting both
    # halves carries an electric source. An ELECTRIC source is injected BETWEEN the
    # two halves (driver.py:3294-3299), so a fused pair would consume a pre-injection
    # D unless the deposit repair brackets the launch — which for this family it does.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. A MAGNETIC source is
    # injected in the B/H half and does NOT disqualify the pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294-3299)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS. Both halves already refuse a mirror plane
    # (complex_fields._complex_grid_reasons clause 4), so this restates by NAME what
    # that refusal buys on THIS seam — a reader should not have to chase a shared grid
    # clause to learn that two driver passes sit between the halves on a folded grid.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled. `zero_metal_axes` asks `has_metallic`/`is_metallic`/`is_mirrored`; a grid
    # that cannot answer would silently be treated as unwalled, which is a plane of
    # wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A polarization would make the constitutive source (D - sum P) rather than D.
    # `complex_constitutive_coverage("E")` already refuses it module-wide (clause 7);
    # restated because this family's kernel bakes the plain product and a reader
    # should not have to chase the other predicate to learn that — the same
    # restatement `fused_electric_pair` carries.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalComplexFusedElectricPairPlan(KernelPlan):
    """ONE dispatch performing three driver passes, complex D never leaving a register.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no special
    case for it.

    ``runs`` is an alias of ``launches`` rather than a second counter, because with
    one dispatch per run the two ARE the same number and a second counter would be a
    second thing to get out of step.
    """

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values",
                 "zero_metal", "shape", "n_elem", "dtdx", "expansion", "params")

    family = "complex fused PML D-curl/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phased", "zero_metal", "expansion")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 expansion: str, params: Any, pointers: Sequence[Any],
                 functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # The COMPLEX CELL count, which is also what the dispatch is sized from: the
        # grid comes from the first tensor argument's element count, and a complex64
        # tensor's element count is its cell count. Binding a float32 word view
        # instead would size the grid at twice the cells and step every cell twice.
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a complex64 volume by a Python float and
        # the packed struct holds numpy.float32 of the same value, so the two scalars
        # are the same bits.
        self.dtdx = float(dtdx)
        self.expansion = str(expansion)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def params_record_dtype() -> Any:
    """The host record's dtype — THE THREE ``float2`` MEMBERS FIRST, itemsize 48.

    ONE HOME FOR THE LAYOUT, because getting it wrong is a silent wrong answer rather
    than a crash: see :mod:`.complex_fused_magnetic_pair` for the measurement. The
    offsets are stated EXPLICITLY even though the phases-first order makes them the
    natural ones, so that the record cannot drift into agreeing with Metal only by
    accident; ``itemsize`` is stated because the natural record is 44 bytes and
    Metal's struct is 48.
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["px", "py", "pz", "nx", "ny", "nz", "n_elem", "dtdx"],
        "formats": [("<f4", 2), ("<f4", 2), ("<f4", 2),
                    "<u4", "<u4", "<u4", "<u4", "<f4"],
        "offsets": [0, 8, 16, 24, 28, 32, 36, 40],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]], device: str) -> Any:
    """The five scalars and three phases as one 48-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason the magnetic twin's
    ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME, because a wider element silently
    reinterprets. This record is neither, so it is built here, once, at plan time, and
    held by the plan. Nothing on the launch path allocates.
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
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_complex_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        ) -> Optional[MetalComplexFusedElectricPairPlan]:
    """Build the complex fused D/E plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING — every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not metal_complex_fused_electric_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    expansion = complex_fields.expansion_from_probe(
        probe if probe is not None else complex_fields.load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    kinds = _boundary_kinds(grid, pml)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    walls = zero_metal_axes(grid)
    # backward=True: this is the D seam. The phase table is resolved through the
    # certified helper so the conjugation rule lives in one place.
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds), backward=True)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step, so
    # the groups are built in the order the kernel declares them.
    flux = [bind_complex(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Dx", "Dy", "Dz")]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in ("Hx", "Hy", "Hz")]
    electric = [bind_complex(name, getattr(fields, name))
                for name in ("Ex", "Ey", "Ez")]
    workspace = [bind_complex("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Ex", "Ey", "Ez")]
    # THE INVERSE EPSILON IS float32 UNDER COMPLEX STORAGE — one real coefficient per
    # cell, applied to both planes by `c_mul_field_left`, indexed by the COMPLEX CELL
    # index. Three pointers, not six.
    inverse_epsilon = [
        bind_real("inv_eps_" + name, fields.inverse_epsilon_for(name))
        for name in ("Ex", "Ey", "Ez")]
    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER one
    # (complex_fields.SUB_STEPS['step_D']['suffix'] == '';
    # CONSTITUTIVE_SIDES['E']['half_integer'] is True). That is the OPPOSITE pairing
    # to the B/H pair, and the kernel cannot tell — the gate carries a host mutation
    # for each group. The coefficients are float32 in BOTH storage modes
    # (stepping.py:41-50), which is what keeps this at twelve buffers rather than
    # twenty-four.
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
            selected[mode] = compile_complex_fused_electric_pair(
                codes, flags, walls, expansion, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor would
    # read "mps:0" where the mirrors were built with "mps" and put the struct on a
    # nominally different device from the volumes it describes.
    dtdx = grid.dt / grid.dx
    return MetalComplexFusedElectricPairPlan(
        residency, volumes, codes, flags, values, walls, grid.shape, dtdx, expansion,
        _params_tensor(grid.shape, dtdx, values, residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"complex fused electric pair cannot fill {slot}",))
    return metal_complex_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalComplexFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_complex_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY, as it is on every other pair.
    ``plan_step`` assigns at most one arm per slot and this product spans three, so
    there is no slot it could claim through the arm table. Registering unwired keeps
    it ENUMERABLE for the disjointness sweep (``arms.registered``) while
    ``arms.arms_for`` skips it, so ``_select_slot`` cannot select it and no existing
    arm's selection changes — including the ``complex/Bloch`` curl and constitutive
    arms this pair's own predicate is built out of.

    THE OTHER HALF IS THE ABSORB TABLE. ``launch._install_fused_pairs`` reaches this
    row through ``arms.registered`` precisely BECAUSE it is registered — ``is_weld``
    with ``replaces`` covering ``update_E`` — and
    ``launch.FUSED_PAIR_ARMS["complex_fused_electric_pair"]`` declares which arm each
    of the two slots implements.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "complex fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="complex fused electric D/E pair: ",
                          noun="complex fused PML D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
