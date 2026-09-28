"""The COMPLEX/BLOCH fused Metal pair: complex ``step_B`` welded into ``update_H``.

The complex-storage twin of :mod:`.fused_magnetic_pair`, on the B/H seam the corpus
census says is the productive one. ONE DISPATCH FOR ALL THREE COMPONENTS: the plain
magnetic constitutive carries no pole buffers and no inverse-mu volume, so the whole
complex B half fits in a single launch — 27 pointers plus one packed
``constant Params&``, 28 of the 31 bindings the platform allows. The separate
composition for this configuration is TWO device dispatches (the certified complex
curl, then the certified complex constitutive) plus one host wall pass on a walled
run; this is ONE, and the complex flux density never leaves a register.

EVERYTHING HERE IS TRANSCRIBED, and the two halves are not re-spelled at all — they
are SPLICED FROM THE CERTIFIED COMPLEX EMITTERS' OWN OUTPUT:

* the B curl half — :func:`.complex_fields.bloch_curl_source` with ``backward=False``
  (complex_fields.py:248-343), whose body is lifted verbatim by
  :func:`certified_curl_body`. Its ghost gather is :func:`.templates.ghost`, its
  cell-0 mask :func:`.templates.ownership_mask` and its wrap rotation
  :func:`.complex_fields._phase_block`, all reached through that emitter rather than
  through a second copy;
* the wall clear — ``stepping._zero_metal`` (stepping.py:2206-2247) restricted to
  ``B_COMPONENTS`` (stepping.py:183), carried inline through :data:`_ZERO_METAL_ROWS`
  and writing :data:`.templates.COMPLEX_ZERO` rather than ``0.0f``;
* the constitutive half — :func:`.complex_fields.bloch_constitutive_source` on side
  ``"H"`` (complex_fields.py:436-512), whose body is lifted verbatim by
  :func:`certified_constitutive_body`, itself the transcription of
  ``stepping.update_H`` and ``stepping._apply_constitutive_pml``
  (stepping.py:2112-2143).

Because both halves are LIFTED rather than retyped, "the fused arithmetic is the
certified complex arithmetic" is a property of the construction that a reader can
check by running :func:`certified_curl_body` — and the gate's ``transcription`` leg
checks exactly that, because a construction is a hypothesis until something compares
the strings.

===========================================================================
THE SEAM, AND WHAT SITS INSIDE IT — measured from the driver, not assumed
===========================================================================

The driver runs FOUR passes between ``step_B`` and ``update_H``
(driver.py:3281-3289): ``step_B`` -> MAGNETIC SOURCES (:3283-3284) ->
``fill_symmetry_bc_B`` (:3285) -> ``zero_metal_B`` (:3286) ->
``fill_folded_far_ghosts_B`` (:3287) -> ``update_H`` (:3289).

**THE MAGNETIC SOURCE SLOT IS REAL, AND IT IS THE BINDING CLAUSE OF THIS FAMILY**,
exactly as it is for the real twin. ``driver.step`` injects magnetic currents at
driver.py:3283-3284, in the mirror of the electric slot at :3290-3299, and MEEP's own
order is what puts it there — ``step_db(B_stuff)``, ``step_source(B_stuff)``,
``step_boundaries(B_stuff)`` (step.cpp:64-72). A fused pair that spanned the slot
would consume a pre-injection B. So the clause is REFUSED BY NAME, with the polarity
flipped: an ELECTRIC source is injected in the D/E half and does NOT disqualify this
pair.

* **``zero_metal_B`` — CARRIED INLINE**, and on this side it writes a COMPLEX zero.
  ``_zero_metal`` writes zero into stored cell 0 of every component whose Yee shift
  on a walled axis is 0 (stepping.py:2244-2247), and for B that is the DIAGONAL —
  ``IYEE_SHIFTS`` (fields.py:214-219) gives Bx (0,1,1), By (1,0,1), Bz (1,1,0), so Bx
  clears on an x wall, By on a y wall, Bz on a z wall. That is the EXACT COMPLEMENT
  of the D side's off-diagonal table, and the difference is not cosmetic: a fused
  pair that reused the D table would clear the wrong two components on every walled
  run. The zero is ``float2(0.0f, 0.0f)`` because the array path assigns a complex
  zero to BOTH planes (stepping.py:1943, :1949); a real ``0.0f`` there would not
  compile, but the plane-wise question it stands for is live and the gate carries a
  mutation that clears only the real plane.
  ``fu_B`` is deliberately NOT masked: the array-path pass touches ``B_COMPONENTS``
  only (stepping.py:2250).
* **``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` — REFUSED** through both
  halves' own "a mirror plane is active" clause
  (:func:`.complex_fields._complex_grid_reasons` clause 4), which is what makes both
  fill slots dead for every configuration this family admits. A FOLDED complex B/H
  seam is a different product and is NOT built here.

===========================================================================
(1) WHAT float2 DOES TO THE POINTER COUNT — the question this family turns on
===========================================================================

**float2 DOES NOT DOUBLE THE POINTER COUNT; IT IS WHAT KEEPS IT FROM DOUBLING.** A
complex64 volume binds as ONE ``device float2*`` buffer, exactly as a float32 volume
binds as one ``device float*``. So the fifteen field volumes this pair needs stay
FIFTEEN pointers, and the count is the real twin's:

    3 B  +  3 fu_B  +  3 E   (float2, the curl half)
  + 3 H  +  3 f_w_H          (float2, the constitutive half)
  + 6 curl coefficients  +  6 constitutive coefficients   (float32; see below)
  = 27 pointers

There is NO inverse-mu volume: the H product is ``source = B`` with mu = 1 baked into
the array path too (``_CONSTITUTIVE_PRODUCTS["H"]`` is ``g0[ii]`` unmultiplied). The
certified complex constitutive kernel keeps three inverse-epsilon pointers on BOTH
sides so its buffer layout is one layout; this family serves the H side only, has no
second side to share a layout with, and DROPS them. That is why 27 rather than 30.

**THE COEFFICIENTS STAY float32 UNDER COMPLEX STORAGE, AND THAT IS LOAD-BEARING.**
Every recurrence on the complex step path has REAL coefficients — curl
(stepping.py:1648-1683), split-field PML (:1952-1982), constitutive dsigw
(:2112-2143) — and stepping.py:41-50 states the invariant directly: ``inv_eps`` and
every PML coefficient are float32 in BOTH storage modes. Twelve coefficient buffers,
not twenty-four. Had they widened with the fields, this signature would need 39
pointers and the fusion could not be built at all.

**BOTH THE float2 VOLUMES AND THE PACKED STRUCT ARE FORCED, and each is refused by a
separate measurement rather than by a comment:**

* bind the volumes as SEPARATE re/im planes and the pointer count alone is
  30 field + 12 coefficient = 42, before a single scalar —
  :func:`split_plane_pair_signature` builds that signature at
  :data:`SPLIT_PLANE_BINDINGS` = 53 and the gate requires the COMPILE FAILURE;
* keep float2 but bind the five scalars and three phases SEPARATELY, as the
  certified complex curl does at its own 23-binding size, and this pair needs
  27 + 5 + 3 = :data:`SEPARATE_SCALAR_BINDINGS` = 35 — over the ceiling.
  :func:`refuted_separate_scalar_source` builds it and the gate requires that
  failure too.

Packing all eight non-pointer arguments into one ``constant Params&`` buys back seven
slots and lands at :data:`PACKED_BINDINGS` = 28, three under the ceiling. The certified
complex curl could bind its phases separately because it binds only 9 volumes; this
pair binds 15, and that is the whole difference.

**THE STRUCT FIELD ORDER IS A MEASURED SAFETY PROPERTY, NOT A STYLE CHOICE.** Metal
aligns ``float2`` to 8 bytes. With the scalars first the struct needs INTERNAL padding
(``dtdx`` ends at byte 20, ``px`` must start at 24) and the natural NumPy record —
fields laid end to end, itemsize 44 — puts every phase one word early. Measured on
this toolchain 2026-08-19, that record reads ``px = (0.8660254, -1.0)`` where the host
wrote ``(-0.5, 0.8660254)``: each phase picks up its neighbour's component. THAT IS
NOT A CRASH AND IT IS NOT NOISE — it is a well-formed unit-modulus-ish complex number,
so a run would converge, look smooth, and be wrong. Declaring the three ``float2``
members FIRST removes the internal padding entirely: the natural offsets
(0, 8, 16, 24, 28, 32, 36, 40) then coincide with Metal's, and the same naive record
round-trips exactly. Both orders were measured; this module ships the order that
survives the mistake, and the gate's ``params_layout`` leg reads every field back off
the device and carries the 44-byte record as a must-catch mutation.

===========================================================================
(2) THE COMPLEX-MULTIPLY ARM — bound from an artifact, never chosen
===========================================================================

A complex product has two licensable transcription arms, ``FMA_V1`` and ``NAIVE``
(:data:`.templates.EXPANSIONS`), and which one the REFERENCE takes is a measured
platform fact. This family does not re-open that question and does not carry a probe
of its own: its arithmetic IS :mod:`.complex_fields`' arithmetic, spliced from that
module's own emitters, so it binds through that module's probe —
``$MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE``, :func:`.complex_fields.load_expansion_probe`
and :func:`.complex_fields.expansion_from_probe` — and inherits its refusals. A second
probe here would be a second answer to one question and a way for the two to drift.

**THE ARM THIS PACKAGE BINDS ON THIS HOST IS ``FMA_V1``, ON THIS EVIDENCE:**
``parity/meep_gpu/results/metal_complex_audit_2026-08-16/complex_expansion_probe.json``,
``backend: numpy``, ``measured: 2026-08-15``, ``numpy: 2.4.3`` — the backend the
ENGINE holds on this host, and the same numpy this machine runs today (2.4.3), so the
artifact is valid here rather than merely present. Its five patterns:

    c8_mul_c8                    FMA_V1     the generic complex product
    c8_mul_c8_scalar_right       FMA_V1     the phase rotation as S:1862 spells it
    c8_mul_f4_field_left         AMBIGUOUS_BOTH
    f4_mul_c8_coefficient_left   AMBIGUOUS_BOTH
    python_float_left            AMBIGUOUS_BOTH

Two patterns discriminate, both name ``FMA_V1``, and the three that cannot are
``AMBIGUOUS_BOTH`` by construction: their coefficient is real, so the fma's addend is
an exact ``+0.0`` and the two arms coincide bit for bit. A probe on which NOTHING
discriminates, or on which the discriminating patterns DISAGREE, yields ``None`` and
is a coverage refusal by name — never a default arm.

**THE ARM IS LOAD-BEARING FOR THIS PAIR ONLY THROUGH THE BLOCH ROTATION.** Every other
multiply either half performs takes a real coefficient (``dtdx``, ``kms``, ``sinv``,
``kps``), so its imaginary operand is an exact ``+0.0`` and the arms coincide. The
consequence is sharp and is stated here rather than discovered: **on an unphased row
this kernel's bytes do not depend on the arm at all**, so a gate that ran only
``k = 0`` rows would certify the arm binding vacuously. That is why
:data:`.CONFIGS`-style phased rows are required in the gate rather than sampled, and
why the arm-swap mutation is armed on a PHASED case.

===========================================================================
WHAT THE CORPUS SAYS THIS IS WORTH — measured, and it is not sixteen
===========================================================================

From the 186-row census
(``parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19``, the measured rows of
``examples.jsonl`` + ``tests.jsonl`` + ``tests_param_matched.jsonl``):

    admit the complex curl at step_B                                16 rows
    admit the complex constitutive at update_H                      16 rows
    admit BOTH halves                                               16 rows
    ... and declare NO MAGNETIC source                              12 rows

**SO THIS FAMILY REACHES 12 OF 186 ROWS, NOT THE 16 THE SLOT SIGNATURE SUGGESTS.**
The magnetic-source clause costs 4: three rows declare ``("D", "B")`` and one declares
``("B",)``. That gap is the same one the real twin reports (46 admitting both halves,
24 surviving the clause) and it is reported here beside the gate rather than left to
be discovered — the "16 rows" in the fusion matrix is the SEAM's row count, and a
fused product's row count is always the seam's minus the source clause.

NOT WIRED, for the reason the other fused pairs are not. ``STEP_ORDER`` assigns at most
one arm per slot and this product spans THREE of them (``step_B``, ``zero_metal_B``,
``update_H``); there is no slot it can claim without a composition rule nothing has
measured, so it registers ``wired=False``. The composer cannot select it, the
disjointness sweep can still enumerate it, and no existing arm's behaviour changes.

NO ENTRY IN ``fingerprints.json``'s ``kernel_source_sha256``, deliberately and for
:mod:`.complex_fields`' reason: this module's sources are a function of the
PROBE-BOUND arm, so a checked-in hash would record a choice rather than a measurement.
The gate hashes every specialisation of the arm it actually bound, into its own
results directory, beside the probe artifact that bound it.

Import contract: importable WITHOUT torch. The predicates and the plan builders (to
``None``) must answer on a host with no GPU, which is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, MAGNETIC_FIELD_TYPE
from . import complex_fields, shaders, templates
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
#:
#: TRUE SINCE 2026-08-28, in the same edit as the ``launch.FUSED_PAIR_ARMS`` row that
#: lets the seam loop bracket this family (``launch.py:660``, the
#: ``('complex/Bloch', 'complex/Bloch')`` entry). Two things had to hold and both were
#: measured rather than argued:
#:
#: * THE ARITHMETIC. ``deposit_repair.apply``'s magnetic branch had only ever been
#:   driven against the REAL ordinary ``update_H``. Complex storage is one
#:   ``complex64`` array per component (fields.py:573), not two real planes, and the
#:   repair indexes and accumulates element-wise (deposit_repair.py:118-129, :146-168),
#:   so it needs no change — but "needs no change" is not "measured". Driven through
#:   the driver's own consult order on ``force_complex_fields=True`` fields, four cases
#:   (point, 24-step, line, and a deposit inside the PML shell) are byte-identical to
#:   the array path over 24 arrays, each with the unrepaired null control diverging.
#:   ``test_deposit_repair.py``'s ``COMPLEX_CASES`` is that measurement.
#: * THE FOLD BOUNDARY DOES NOT APPLY. What holds the two folded families at ``False``
#:   is that a folded seam runs ``fill_symmetry_bc_B``/``fill_folded_far_ghosts_B``
#:   AFTER the injection and a POINT repair never visits the mirror image of a deposit
#:   index. This family REFUSES every mirrored axis by name in its own predicate
#:   below, so no configuration it admits has a mirror image to miss.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "complex_fused_magnetic_pair"

#: The sub-step slot this arm is registered on. It spans three; it holds a row on the
#: first, so a refusal is NAMED on the slot the fusion starts at rather than being
#: invisible to the table.
SLOT = "step_B"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3281-3289). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: The shipped binding count: 27 pointers + one packed ``constant Params&``.
PACKED_BINDINGS = 28

#: What the same 27 pointers need with the five scalars and three phases bound
#: SEPARATELY — the way the certified 23-binding complex curl binds them. Over the
#: ceiling; :func:`refuted_separate_scalar_source` builds it and the gate requires the
#: compile failure.
SEPARATE_SCALAR_BINDINGS = 35

#: What the same kernel needs with the complex volumes bound as SEPARATE re/im planes:
#: 30 field + 12 coefficient + 5 scalars + 6 phase floats. Far over the ceiling, which
#: is the measurement behind "float2 is FORCED, not preferred".
SPLIT_PLANE_BINDINGS = 53

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex fused magnetic pair binds {PACKED_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}; the signature cannot be "
        f"built at all")

#: The host record's total size in bytes. Metal aligns ``float2`` to 8, and with the
#: three phases FIRST there is no internal padding — only a trailing pad from 44 to
#: the struct's 8-byte alignment. Spelled as data so the plan and the gate size the
#: record from one number.
PARAMS_ITEMSIZE = 48

__all__ = [
    "FAMILY", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "SPLIT_PLANE_BINDINGS",
    "MetalComplexFusedMagneticPairPlan", "certified_constitutive_body",
    "certified_curl_body", "compile_complex_fused_magnetic_pair",
    "complex_fused_magnetic_pair_source", "enumerate_sources",
    "metal_complex_fused_magnetic_pair_coverage",
    "params_record_dtype", "plan_metal_complex_fused_magnetic_pair",
    "refuted_separate_scalar_source", "register_arms",
    "split_plane_pair_signature", "zero_metal_mask",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
#
# TWENTY-EIGHT BINDINGS: 15 float2 volumes + 12 coefficient vectors + 1 packed
# Params&. See the module docstring for the two signatures this refutes.

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

kernel void complex_fused_magnetic_pair_step(
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

#: The certified complex curl body's last statement — the B store. The wall clear goes
#: IMMEDIATELY BEFORE it, because the array path stores B and only then overwrites the
#: wall plane; masking the register first is the same field and one fewer pass. Note
#: the ``u`` store precedes it and is NOT masked: ``zero_metal_B`` passes
#: ``B_COMPONENTS`` (stepping.py:2250) and the split-field auxiliary is not in it.
_CURL_STORE = "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n"

#: The certified constitutive body's decode prologue, which the curl body has already
#: emitted. Splicing both would redeclare ``ii``/``i``/``j``/``k`` and the kernel would
#: not compile — the lift drops exactly this much and no more. Dropping through it also
#: drops the second ``n_elem`` guard, which is the same guard.
_CONSTITUTIVE_PROLOGUE_END = "    int i   = plane / nyi;\n"

#: ``_zero_metal`` for the B components, as (target, walled axis, flag) rows.
#:
#: A component is cleared on the wall of axis ``d`` exactly when its Yee shift there
#: is 0 (stepping.py:2256-2260, :2300). ``IYEE_SHIFTS`` (fields.py:216-217) gives
#: Bx (0,1,1), By (1,0,1), Bz (1,1,0), so these rows are THE DIAGONAL — the exact
#: complement of the D side's off-diagonal table. Spelled here rather than imported
#: from the real twin, for that reason.
_ZERO_METAL_ROWS: Tuple[Tuple[int, int, str], ...] = (
    (0, 0, "at_x"),
    (1, 1, "at_y"),
    (2, 2, "at_z"),
)


def certified_curl_body(codes: Sequence[int], phased: Sequence[int], expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The complex ``step_B`` curl kernel's BODY, lifted from the certified emitter.

    Not a transcription: this is :func:`.complex_fields.bloch_curl_source`'s own
    output with the ``#include``/helpers/signature preamble and the closing brace
    removed. ``backward`` is False and never a parameter — this pair is the B seam,
    and ``step_D``'s negated strides and conjugated phase are a different product.
    """
    source = complex_fields.bloch_curl_source(codes, False, phased, expansion,
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
    """The complex ``update_H`` constitutive kernel's BODY, lifted and renamed.

    Two edits, and both are POINTER SPELLING rather than arithmetic — the same kind
    of rename the real twin applies, and for the same reason: fused, ``f0`` would
    mean B in the curl half and H in the constitutive half in one scope.

    * the decode prologue is dropped (the curl half already emitted it);
    * the three targets ``f0``/``f1``/``f2`` become ``h0``/``h1``/``h2``.

    THE SOURCE LINES ARE NOT EDITED HERE. ``float2 src0 = g0[ii];`` is left standing
    and :func:`complex_fused_magnetic_pair_source` is what turns it into the register
    — so the seam is one visible substitution in one place, not a rename that quietly
    also moved it. That also resolves the one genuine name collision in this splice:
    ``g0`` means E in the curl half and B in the certified H body, and after the seam
    substitution the constitutive half references ``g`` nowhere at all.
    """
    source = complex_fields.bloch_constitutive_source("H", expansion, contract)
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
    """``stepping.zero_metal_B`` for all three targets, carried inline on the register.

    The array path stores B and then overwrites the wall plane (stepping.py:2247);
    this masks the value before it is stored and before the H half consumes it, which
    is the same field and one fewer pass.

    The zero is a COMPLEX zero — both planes — because the array path assigns one
    (stepping.py:1943, :1949). The select spelling is
    :func:`.templates.ownership_mask`'s — ``flag ? zero : v`` — because that is the
    form already measured to deliver an exact ``0.0`` on this platform.
    """
    lines = [f"    v{target} = {flag} ? {zero} : v{target};"
             for target, axis, flag in _ZERO_METAL_ROWS if bool(zero_metal[axis])]
    return "\n".join(lines) or "    // no walled axis clears a B component"


def complex_fused_magnetic_pair_source(
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
        "    // --- zero_metal_B, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The DIAGONAL for B: Bx on an x wall, By on y, Bz on z. A COMPLEX\n"
        "    // zero: the array path assigns one to both planes (S:1896, :1902).\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    magnetic = certified_constitutive_body(expansion, contract)
    # THE SEAM, and it is exactly three lines. The certified complex H body opens each
    # component with `float2 srcN = gN[ii];` — a RELOAD of the complex flux density
    # the curl just stored. Fused, that becomes the live register.
    for target in range(3):
        old = f"    float2 src{target} = g{target}[ii];\n"
        if old not in magnetic:
            raise AssertionError(
                f"the certified complex H constitutive body no longer reads its "
                f"source as {old.strip()!r}; the seam has no anchor")
        magnetic = magnetic.replace(
            old,
            f"    // THE SEAM: the register step_B just wrote, not a reload of "
            f"B{'xyz'[target]}.\n"
            f"    float2 src{target} = v{target};\n")

    body = "".join((
        curl,
        "\n    // --- update_H (stepping.update_H / _apply_constitutive_pml:2065) --\n",
        magnetic,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": body,
    })


def compile_complex_fused_magnetic_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, phases, walls, arm, mode)."""
    return compile_source(complex_fused_magnetic_pair_source(
        codes, phased, zero_metal, expansion,
        contract)).complex_fused_magnetic_pair_step


def enumerate_sources(expansion: str,
                      contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit, keyed by a stable label.

    The phase triple is enumerated only over the axes a given boundary triple can
    legally phase — a metallic axis cannot carry one — and the wall triple only over
    the axes that are METALLIC, because ``zero_metal_axes`` cannot report a wall on a
    periodic axis. So the count is the REACHABLE set rather than the Cartesian
    product, and a label that cannot be built is not fingerprinted as if it could.
    """
    out: Dict[str, str] = {}
    for cx in (templates.PERIODIC, templates.METALLIC):
        for cy in (templates.PERIODIC, templates.METALLIC):
            for cz in (templates.PERIODIC, templates.METALLIC):
                codes = (cx, cy, cz)
                phase_options = [(0, 1) if code == templates.PERIODIC else (0,)
                                 for code in codes]
                wall_options = [(False, True) if code == templates.METALLIC
                                else (False,) for code in codes]
                for px in phase_options[0]:
                    for py in phase_options[1]:
                        for pz in phase_options[2]:
                            for wx in wall_options[0]:
                                for wy in wall_options[1]:
                                    for wz in wall_options[2]:
                                        walls = (wx, wy, wz)
                                        label = (
                                            f"complex_fused_magnetic_pair_step/"
                                            f"{cx}{cy}{cz}/ph{px}{py}{pz}/"
                                            f"zm{int(wx)}{int(wy)}{int(wz)}/"
                                            f"{expansion}")
                                        out[label] = (
                                            complex_fused_magnetic_pair_source(
                                                codes, (px, py, pz), walls,
                                                expansion, contract))
    return out


# ---------------------------------------------------------------------------
# The two refuted signatures
# ---------------------------------------------------------------------------

def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 35-binding separate-scalar/phase signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and require
    the failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from an argument
    into a measurement. This is the signature the certified 23-binding complex curl
    uses, scaled to this pair's fifteen volumes — so what it measures is precisely
    "the packing is FORCED by the extra six volumes the fusion brings", not that
    separate scalars are bad style.

    The body is a trivial touch of every buffer — what is being measured is the
    SIGNATURE, and a body the compiler could drop would let dead-code elimination
    decide the answer.
    """
    written = ["f0", "f1", "f2", "u0", "u1", "u2", "h0", "h1", "h2", "w0", "w1", "w2"]
    read_complex = ["g0", "g1", "g2"]
    read_real = ["kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                 "kp0", "km0", "kp1", "km1", "kp2", "km2"]
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
    lines.append(f"    constant float&      dtdx    [[buffer({slot})]],")
    slot += 1
    for name in ("px", "py", "pz"):
        lines.append(f"    constant float2&     {name:<8}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - a construction invariant
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    body = "\n".join(f"    {name}[idx] = {name}[idx] * dtdx;" for name in written)
    touch_real = " + ".join(f"{name}[0]" for name in read_real)
    touch_complex = " + ".join(f"{name}[0].x" for name in read_complex)
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
        f"    float touch = ({touch_real}) + ({touch_complex});",
        "    touch = touch + px.x + py.y + pz.x;",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz);",
        body,
        "}",
        "",
    ))


def split_plane_pair_signature() -> str:
    """The re/im-SPLIT fused signature, which must FAIL to compile. 53 bindings.

    This is not a strawman and it is not dead code: it is the evidence that ``float2``
    volumes are FORCED for this family rather than chosen. Fifteen volumes as thirty
    planes exceeds the ceiling on the FIELD POINTERS ALONE — before a coefficient, a
    scalar or a phase — which is a stronger statement than the certified complex
    curl's 35 and is the reason it is spelled separately.

    The gate compiles this and requires the compile error, so a later edit that
    "tidies" the complex volumes into separate real and imaginary buffers is refused
    by a measurement rather than by a comment.
    """
    planes = [f"    device float* {stem}{n}{part} [[buffer({index})]]"
              for index, (stem, n, part) in enumerate(
                  (stem, n, part)
                  for stem in ("f", "u", "g", "h", "w") for n in range(3)
                  for part in ("re", "im"))]
    slot = len(planes)
    coefficients = [f"    device const float* c{n} [[buffer({slot + n})]]"
                    for n in range(12)]
    slot += 12
    scalars = [f"    constant uint& s{n} [[buffer({slot + n})]]" for n in range(4)]
    slot += 4
    scalars.append(f"    constant float& dtdx [[buffer({slot})]]")
    slot += 1
    phases = [f"    constant float& p{n} [[buffer({slot + n})]]" for n in range(6)]
    slot += 6
    if slot != SPLIT_PLANE_BINDINGS:  # pragma: no cover - a construction invariant
        raise AssertionError((slot, SPLIT_PLANE_BINDINGS))
    arguments = ",\n".join(planes + coefficients + scalars + phases)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void split_plane_pair(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    f0re[idx] = f0im[idx];\n}\n")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_complex_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span complex ``step_B`` -> wall -> ``update_H``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. That is the
    construction the real twin uses, and it is what makes the expansion-probe clause,
    the complex-storage clause and the fold clause inherited rather than restated.
    """
    reasons: List[str] = []

    curl = complex_fields.complex_pml_curl_coverage(
        fields, pml, "step_B", residency, probe)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    magnetic = complex_fields.complex_constitutive_coverage(
        fields, pml, "H", residency, probe)
    if not magnetic.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in magnetic.reasons)

    # THE SOURCE SEAM, and for this family it is the BINDING clause on the measured
    # corpus: 16 census rows admit both halves and 12 survive this. A MAGNETIC source
    # is injected BETWEEN the two halves (driver.py:3283-3284), so a fused pair would
    # consume a pre-injection B. IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not
    # hold the source list, so a predicate that inferred "no sources" from not being
    # told would be the over-covering refusal this clause exists to prevent. An
    # ELECTRIC source is injected in the D/E half and does NOT disqualify the pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283-3284)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS. Both halves already refuse a mirror plane
    # (complex_fields._complex_grid_reasons clause 4), so this restates by NAME what
    # that refusal buys on THIS seam — a reader should not have to chase a shared grid
    # clause to learn that two driver passes sit between the halves on a folded grid.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_B and "
                f"stepping.fill_folded_far_ghosts_B both run inside this seam "
                f"(driver.py:3285-3287) and neither is carried by this family")

    # zero_metal_B is CARRIED, so the grid must be able to answer which axes are
    # walled. `zero_metal_axes` asks `has_metallic`/`is_metallic`/`is_mirrored`; a grid
    # that cannot answer would silently be treated as unwalled, which is a plane of
    # wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalComplexFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs three driver passes, complex B never leaving a register.

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

    family = "complex fused PML B-curl/stored-H pair"

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
    than a crash: see the module docstring for the measurement. The offsets are stated
    EXPLICITLY even though the phases-first order makes them the natural ones, so that
    the record cannot drift into agreeing with Metal only by accident; ``itemsize`` is
    stated because the natural record is 44 bytes and Metal's struct is 48.
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

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason the real twin's
    ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME, because a wider element silently
    reinterprets. This record is neither — it is a packed struct of three float2s,
    four uints and one float that nothing on the host reads back and nothing on the
    device writes — so it is built here, once, at plan time, and held by the plan.
    Nothing on the launch path allocates.
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


def plan_metal_complex_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        ) -> Optional[MetalComplexFusedMagneticPairPlan]:
    """Build the complex fused B/H plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING — every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not metal_complex_fused_magnetic_pair_coverage(
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
    # backward=False: this is the B seam. The phase table is resolved through the
    # certified helper so the conjugation rule lives in one place.
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds), backward=False)

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
    flux = [bind_complex(name, getattr(fields, name)) for name in ("Bx", "By", "Bz")]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Bx", "By", "Bz")]
    electric = [bind_complex(name, getattr(fields, name))
                for name in ("Ex", "Ey", "Ez")]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in ("Hx", "Hy", "Hz")]
    workspace = [bind_complex("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Hx", "Hy", "Hz")]
    # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER one
    # (complex_fields.SUB_STEPS['step_B']['suffix'] == '_h';
    # CONSTITUTIVE_SIDES['H']['half_integer'] is False). That is the OPPOSITE pairing
    # to the D/E pair, and the kernel cannot tell — the gate carries a host mutation
    # for each group. The coefficients are float32 in BOTH storage modes
    # (stepping.py:41-50), which is what keeps this at twelve buffers rather than
    # twenty-four.
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"))
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
            selected[mode] = compile_complex_fused_magnetic_pair(
                codes, flags, walls, expansion, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor would
    # read "mps:0" where the mirrors were built with "mps" and put the struct on a
    # nominally different device from the volumes it describes.
    dtdx = grid.dt / grid.dx
    return MetalComplexFusedMagneticPairPlan(
        residency, volumes, codes, flags, values, walls, grid.shape, dtdx, expansion,
        _params_tensor(grid.shape, dtdx, values, residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"complex fused magnetic pair cannot fill {slot}",))
    return metal_complex_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalComplexFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_complex_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY. ``plan_step`` assigns at most one arm per
    slot and this product spans three, so there is no slot it could claim without a
    composition rule nothing has measured — the same open question every fused pair
    records. Registering unwired keeps it ENUMERABLE for the disjointness sweep
    (``arms.registered``) while ``arms.arms_for`` skips it, so ``plan_step`` cannot
    select it and no existing arm's selection changes.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "complex fused magnetic B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="complex fused magnetic B/H pair: ",
                          noun="complex fused PML B-curl/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
