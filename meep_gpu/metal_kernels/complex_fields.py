"""Complex-field (force_complex_fields / Bloch k_point) PML curl + constitutive, on Metal.

THE LARGEST CORPUS FAMILY ON EITHER TRACK: sixteen rows, and the sole admitter on
all four sub-steps for every one of them. Complex64 is stepped as ``float2``
volumes with the Bloch wrap phase applied as a complex rotation on the single
wrapped plane of each shifted operand.

NO ENTRY IN ``fingerprints.json``'s ``kernel_source_sha256``, deliberately. This
module's sources are a function of the PROBE-BOUND expansion arm, so a checked-in
hash would have to pick an arm and would then be a record of a choice rather than of
a measurement. ``gate_metal_complex.py`` hashes every specialisation of the arm it
actually bound, into its own results directory, beside the probe artifact that bound
it. This module's own bytes ARE hashed, under ``host_sha256``, because
``launch._host_modules()`` enumerates the package.

WIRED AS OF TRANCHE 2, AFTER THE SWEEP AND NOT BEFORE. The arms below register with
``wired=True``, so ``plan_step`` composes them; ``fastpath.plan_fast_path`` still
returns ``None`` on every branch and nothing here touches dispatch, which is a
different question. The order matters and is the point: the family was registered
``wired=False`` for a tranche precisely so the composition sweep could call its
predicates over the whole configuration matrix and MEASURE that no other family
co-admits anything it admits, and the flag flipped only once that measurement
existed. The separation is two clauses, each stated here rather than inherited:
``coverage._grid_reasons`` clause 2 refuses complex storage (so the shipped real
kernels refuse this module's whole domain) and this module's clause 8 refuses
``grid.beta`` by name (so a complex BETA run goes to :mod:`.special_kz` instead of
being taken here).

TWO KERNELS, one per sub-step family:

* :func:`bloch_curl_source` — complex ``stepping.step_B`` (:261) / ``step_D``
  (:379) under split-field PML, with the Bloch phase on the wrapped lane of each
  shifted operand;
* :func:`bloch_constitutive_source` — complex ``update_H`` (:907) / ``update_E``
  (:926), the ``dsigw`` accumulation. The constitutive sub-step never sees a wrap
  and therefore takes NO phase.

WHY float2 IS FORCED AND NOT PREFERRED. Metal's buffer attribute indices run 0..30,
so a kernel binds at most 31 buffers; a 32nd is a COMPILE ERROR, measured on this
toolchain. The curl binds 23 as ``float2`` volumes (9 field + 6 coefficient + 5
scalars + 3 phases). The same kernel with re/im planes bound SEPARATELY needs 35
and cannot be built at all. The gate compiles that signature and requires the
failure rather than leaving a future edit to rediscover the ceiling.

WHY THE PLANE TRANSCRIPTION IS LEGAL (from source): every recurrence on the complex
step path has REAL coefficients — curl (stepping.py:1648-1683), split-field PML
(:1952-1982), constitutive dsigw (:2112-2143) — and the imaginary plane enters ONLY
through the Bloch wrap multiply (:1767-1771 up, :1818-1822 down, both via
``_apply_bloch_phase`` :1846-1862) and through a complex source amplitude, which is
a ``sources``-side fact outside these kernels. stepping.py:41-50 states this and
adds the load-bearing invariant: ``inv_eps`` and every PML coefficient are float32
in BOTH storage modes.

WHY IT IS NOT BYTE-TRIVIAL, and this is where the Metal question INVERTS relative
to Triton's. On the Triton track the kernel could not force its own association and
the ``EXPANSION`` constexpr existed to match whatever CuPy did. Here the shader
spells the expansion EXPLICITLY under the file-scope contraction directive, so the
open question is the other one: **which arm does the HOST ORACLE take?** The engine
holds NumPy on this host, so the probe measures ``numpy``'s own complex64 multiply
per call-site orientation, and the answer is bound from that artifact — never
guessed. Measured 2026-08-15 (numpy 2.4.3, arm64):

    c8_mul_c8                    FMA_V1     naive misses 4095/16384 on random data
    c8_mul_c8_scalar_right       FMA_V1     the shape S:1862 actually uses
    c8_mul_f4_field_left         AMBIGUOUS_BOTH   (real operand: the arms coincide)
    f4_mul_c8_coefficient_left   AMBIGUOUS_BOTH
    python_float_left            AMBIGUOUS_BOTH

AMBIGUOUS_BOTH is ACCEPTED and is not a shrug: with ``c_im = +0.0`` the fma's
addend is an exact zero and both arms are exact, so the pattern cannot discriminate
and says so. A pattern matching NEITHER arm is a REFUSAL BY NAME with example
words, because one specialisation cannot represent a platform whose orientations
disagree.

THE ARM IS LOAD-BEARING FOR THE CURL ONLY, AND ONLY THROUGH THE BLOCH ROTATION —
measured 2026-08-15, not reasoned. Every other multiply either kernel performs has a
REAL coefficient (``dtdx``, ``kms``, ``sinv``, ``kps``, ``inv_eps``), so its
imaginary operand is an exact ``+0.0``, the fma's addend is exact, and the two arms
coincide bit for bit. ``gate_metal_complex`` leg ``contraction`` compiles BOTH arms
in BOTH contraction modes for all four sub-steps: the curl separates them (6 words on
``step_B``, 1 on ``step_D``, with two axes phased) and the CONSTITUTIVE KERNEL DOES
NOT — all four builds are bitwise equal there, which is why mutation ``m12`` has no
constitutive twin and why the constitutive rows are recorded as a predicted null with
that reason. The same fact explains why the contraction directive is inert on the
shipped arm: with the products spelled as explicit ``fma()`` calls there is no
implicit ``a*b + c`` left anywhere.

THREE SPELLINGS ARE LOAD-BEARING AND ALL THREE WERE MEASURED, NOT INHERITED:

1. **the zero cross terms** must be spelled literally. Folding ``z_im * 0.0f`` to a
   literal misses 12/128 words on the EXHAUSTIVE 64-pattern signed-zero table, in
   EITHER orientation — 12/128 with the field on the left and 12/128 with the
   coefficient on the left, re-measured 2026-08-15 through the gate's own
   ``folded_zero_cross_terms`` / ``folded_coefficient_left`` arms. (An earlier
   revision of this line said 24/128 for the coefficient-left orientation; that is
   the PLANE-WISE number, and no leg had ever measured a folded coefficient-left
   arm. The gate now carries one, so both figures come from the artifact.) Random
   data misses none of it (0/16,384) — which is why the gate's zero legs are
   exhaustive tables rather than sampled rows;
2. **the plane-wise fast path is wrong**: ``{re*c, im*c}`` misses 24/128 on the
   same table. This is the headline risk the folded-complex tranche names, and it
   is classified here explicitly rather than assumed away;
3. **negation is ``-x``.** The Triton track spells negated addends ``(a*b) * -1.0``
   because *Triton* lowers unary minus as ``0.0 - x`` and canonicalizes signed
   zeros. THAT REASON DOES NOT REPRODUCE ON METAL: measured on the exhaustive
   256-pattern table, ``-x`` and ``x * -1.0f`` are both exact (0/512) while
   ``0.0f - x`` misses 36/512. The workaround is NOT carried, and the refuted
   spelling is a must-catch gate mutation instead of a comment.

WHAT IS TRANSCRIBED, and from where (stepping.py unless noted):

* term table            ``B_CURL_TERMS`` / ``D_CURL_TERMS`` (:213-223)
* ghost rule            ``_shift_up`` (:1723) / ``_shift_down`` (:1787), PERIODIC
                        and METALLIC branches only
* Bloch wrap multiply   ``_apply_bloch_phase`` (:1846-1862): up-shift wrapped plane
                        ``*= phase`` (:1767-1771), down-shift plane 0
                        ``*= conj(phase)`` (:1818-1822); SKIPPED ENTIRELY when the
                        phase is None (:1768-1770 / :1819-1821), which is what keeps
                        k = 0 bit-identical to the plain complex engine
* curl grouping         ``_curl_from_operands`` (:1601-1636)
* ownership mask        ``_mask_non_owned_cells`` (:1865-1902) — writes a complex
                        zero to BOTH planes (:1896, :1902)
* PML recurrence        ``_apply_pml_update`` (:1905-1935)
* constitutive dsigw    ``_apply_constitutive_pml`` (:2065-2096)
* coefficient pairing   ``_curl_coefficients`` (:2418) / ``_constitutive_coefficients``
                        (:2428); host-bound, never chosen in-kernel
* phase legality        ``_bloch_phases`` (:2313-2367): a phased axis must resolve
                        PERIODIC; PML ON A PHASED AXIS IS ADMITTED (settled there by
                        measurement, 2.82e-07 against CPU MEEP)
* phase value           ``grid.bloch_phase`` (grid.py:1011-1017): exp(2*pi*i*k*L),
                        Brillouin edge EXACTLY -1+0j

Import contract: importable WITHOUT torch. The predicates and the plan builders
(to ``None``) must answer on a host with no GPU, which is the merge bar.
"""

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
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import MAX_BUFFER_BINDINGS, compile_source
from .launch import SUB_STEPS
from .plans import KernelPlan

FAMILY = "complex_fields"

#: How many buffers the shipped curl binds: 9 float2 volumes + 6 coefficient vectors
#: + 5 scalars + 3 phases. CHECKED AGAINST THE PLATFORM CEILING AT IMPORT, because
#: the whole ``float2`` design rests on this margin and the number was previously a
#: literal in three places (this module's docstring, the gate's ``bindings`` leg and
#: a test). One home, checked, so a binding added to the signature is refused here by
#: name rather than discovered as a compile error a reader has to attribute.
CURL_BINDINGS = 23
if CURL_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex curl binds {CURL_BINDINGS} buffers and Metal's ceiling on this "
        f"toolchain is {MAX_BUFFER_BINDINGS}; the signature cannot be built at all. "
        f"See `split_plane_curl_signature` for the measurement that established the "
        f"ceiling")

#: Environment variable naming the expansion probe artifact (see the module docstring).
PROBE_PATH_ENVIRONMENT = "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"

#: The backend the probe must have measured. The ENGINE holds NumPy on this host and
#: the kernel must reproduce the ENGINE's bytes, so the binding is to numpy's result
#: — the exact inversion of the Triton track, whose engine holds CuPy.
PROBE_BACKEND = "numpy"

#: The call-site patterns the probe must classify, one per operand orientation the
#: transcription carries. ``c8_mul_c8_scalar_right`` is separate from ``c8_mul_c8``
#: on purpose: stepping.py:1909 multiplies an ARRAY PLANE by a complex64 SCALAR, and
#: a backend that took a scalar fast path there would collapse the rotation to
#: plane-wise and evaporate the whole delta.
#:
#: THESE PATTERNS ARE ORIENTATIONS, NOT MEMORY SHAPES, AND THE DIFFERENCE WAS AN
#: ASSUMPTION UNTIL 2026-08-15. Every one of them is evaluated here on a contiguous,
#: out-of-place, equal-length 1-D array; every real call site is something else —
#: S:1862 is IN PLACE on a face view that is NON-CONTIGUOUS on y and z, S:1929 and
#: S:1635 are in place, S:1929/S:2086 broadcast a ``(n,1,1)`` coefficient. Contiguity,
#: broadcasting and output aliasing all steer numpy's loop selection, so "the probe's
#: arm is the engine's arm" needed measuring rather than asserting. Leg
#: ``call_site_shapes`` in ``gate_metal_complex.py`` now evaluates all eleven real
#: shapes and REFUSES a binding that does not survive them; measured on this host the
#: three face multiplies discriminate FMA_V1 (43 / 46 / 59 naive misses, 0 fused) and
#: the eight coefficient shapes are AMBIGUOUS_BOTH, which agrees with this table.
PROBE_PATTERNS: Tuple[str, ...] = (
    "c8_mul_c8",                    # generic complex product
    "c8_mul_c8_scalar_right",       # the phase rotation as S:1862 spells it
    "c8_mul_f4_field_left",         # fu *= kms (S:1929-1935); D * inv_eps (S:982-984)
    "f4_mul_c8_coefficient_left",   # kps*fw / kms*fw_previous (S:2086-2095)
    "python_float_left",            # xp.multiply(dtdx, total) (S:1635)
)

#: A pattern whose two arms are BOTH exact cannot discriminate and is accepted as
#: such. Every real-coefficient orientation lands here by construction: the fma's
#: addend is an exact zero, so the fused and separately-rounded forms coincide.
AMBIGUOUS_BOTH = "AMBIGUOUS_BOTH"

__all__ = [
    "CURL_BINDINGS",
    "FAMILY",
    "PROBE_BACKEND",
    "PROBE_PATH_ENVIRONMENT",
    "PROBE_PATTERNS",
    "AMBIGUOUS_BOTH",
    "ComplexConstitutivePlan",
    "ComplexPmlCurlPlan",
    "bloch_constitutive_source",
    "bloch_curl_source",
    "bloch_phase_table",
    "complex_constitutive_coverage",
    "complex_pml_curl_coverage",
    "enumerate_complex_sources",
    "expansion_from_probe",
    "load_expansion_probe",
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
# TWENTY-THREE BINDINGS: 9 float2 volumes + 6 coefficient vectors + 5 scalars + 3
# phases. The re/im-split form of the same kernel needs 35 and is a compile error;
# see `split_plane_curl_signature`.

_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

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
    uint idx [[thread_position_in_grid]])
{
__GUARD__

__DECODE__
    int nxi = int(nx);
    int nyz = nyi * nzi;

    // --- the ghost rule, per axis (stepping._shift_up:1723 / _shift_down:1787) --
    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    // METALLIC serves an exact complex 0 past the wall: a (+0.0, +0.0) word pair
    // IS the complex metallic ghost (S:1781-1783, S:1827-1829).
    float2 a   = g0[ii];
    float2 b   = g1[ii];
    float2 c   = g2[ii];
    float2 a_y = vy ? g0[oy] : float2(0.0f, 0.0f);
    float2 a_z = vz ? g0[oz] : float2(0.0f, 0.0f);
    float2 b_x = vx ? g1[ox] : float2(0.0f, 0.0f);
    float2 b_z = vz ? g1[oz] : float2(0.0f, 0.0f);
    float2 c_x = vx ? g2[ox] : float2(0.0f, 0.0f);
    float2 c_y = vy ? g2[oy] : float2(0.0f, 0.0f);

    // --- the Bloch phase, on the WRAPPED LANE ONLY, BEFORE the difference -------
    // Which operand crossed which face follows the stencil: b_x/c_x crossed x,
    // a_y/c_y crossed y, a_z/b_z crossed z. FIELD LEFT (S:1862); the host already
    // conjugated for the BACKWARD sub-step (S:1818-1822). An unphased axis emits
    // NO multiply at all — that skip, not a multiply by 1+0j, is what keeps k = 0
    // bit-identical to the plain complex engine (S:1768-1770 / S:1819-1821).
__PHASE_X__
__PHASE_Y__
__PHASE_Z__

    // --- the curl (stepping._curl_from_operands:1601): DO NOT flatten these parens
    // Complex add/sub is component-wise; the dtdx multiply is the zero-imaginary
    // complex product with the SCALAR ON THE LEFT (S:1635).
    float2 t0 = ((c_y - c) + (b - b_z));
    float2 t1 = ((a_z - a) + (c - c_x));
    float2 t2 = ((b_x - b) + (a - a_y));
    float2 curl0 = c_mul_coefficient_left(dtdx, t0);
    float2 curl1 = c_mul_coefficient_left(dtdx, t1);
    float2 curl2 = c_mul_coefficient_left(dtdx, t2);

    // --- ownership mask (stepping._mask_non_owned_cells:1865) -------------------
    // Writes +0.0 to BOTH planes: the array path assigns a complex zero (S:1896, :1902).
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

    // --- split-field recurrence (stepping._apply_pml_update:1905) ---------------
    // dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    // sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y). Every
    // multiply is the zero-imaginary complex product with the FIELD ON THE LEFT
    // (S:1929-1935); the loaded p registers ARE S:1928's fprev copy.
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];

    float2 p0 = u0[ii];
    float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_y) - curl0, si_y);
    float2 v0 = c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) - p0, si_z);

    float2 p1 = u1[ii];
    float2 n1 = c_mul_field_left(c_mul_field_left(p1, km_z) - curl1, si_z);
    float2 v1 = c_mul_field_left((c_mul_field_left(f1[ii], km_x) + n1) - p1, si_x);

    float2 p2 = u2[ii];
    float2 n2 = c_mul_field_left(c_mul_field_left(p2, km_x) - curl2, si_x);
    float2 v2 = c_mul_field_left((c_mul_field_left(f2[ii], km_y) + n2) - p2, si_y);

    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
"""

#: Which shifted operands crossed which face, and therefore carry that axis's phase.
_PHASED_OPERANDS: Dict[str, Tuple[str, str]] = {"x": ("b_x", "c_x"),
                                                "y": ("a_y", "c_y"),
                                                "z": ("a_z", "b_z")}


def _phase_block(axis: str, phased: bool, backward: bool) -> str:
    """One axis's wrap rotation, or the SKIP that keeps k = 0 bit-identical.

    The wrapped lane is where the unwrapped neighbour index left the array: an
    up-shift wraps at ``i == n-1`` (it read slot ``n``), a down-shift at ``i == 0``
    (it read ``-1``). One plane per phased axis, matching the single-plane multiply
    of ``_apply_bloch_phase`` (stepping.py:1909). On a collapsed (n = 1) axis every
    lane is the wrap lane, which is exactly ``xp.roll``'s behaviour there.
    """
    index, extent, phase = {"x": ("i", "nxi", "px"),
                            "y": ("j", "nyi", "py"),
                            "z": ("k", "nzi", "pz")}[axis]
    if not phased:
        return (f"    // axis {axis} carries no Bloch phase: the multiply is "
                f"SKIPPED, not done\n    // against 1+0j — that skip is the "
                f"bit-identity of k = 0 (S:1768-1770).")
    predicate = f"({index} == 0)" if backward else f"({index} == {extent} - 1)"
    lines = [f"    {{ bool w{axis} = {predicate};"]
    for operand in _PHASED_OPERANDS[axis]:
        lines.append(f"      {operand} = w{axis} ? c_mul({operand}, {phase}) : {operand};")
    lines.append("    }")
    return "\n".join(lines)


def bloch_curl_source(codes: Sequence[int], backward: bool,
                      phased: Sequence[int], expansion: str,
                      contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised complex ``bloch_pml_curl_step`` source.

    ``codes`` is the per-axis PERIODIC/METALLIC triple, ``backward`` selects
    ``step_D``'s negated strides over ``step_B``'s forward ones, ``phased`` is the
    per-axis 0/1 "this axis carries a Bloch factor" triple, and ``expansion`` is the
    PROBE-MEASURED arm name. All four are baked into the string, as Triton bakes its
    constexprs — a distinct source per specialisation, and therefore a distinct
    compile memo key and a distinct fingerprint.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3 or len(phased) != 3:
        raise ValueError(f"codes and phased must be per-axis triples, got "
                         f"{codes!r} / {phased!r}")
    for axis, (code, flag) in enumerate(zip(codes, phased)):
        if flag and code != templates.PERIODIC:
            raise ValueError(
                f"axis {axis} carries a Bloch phase but its ghost rule is "
                f"METALLIC; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same pairing, S:2346-2360)")
    return templates.substitute(_CURL_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__PHASE_X__": _phase_block("x", bool(phased[0]), backward),
        "__PHASE_Y__": _phase_block("y", bool(phased[1]), backward),
        "__PHASE_Z__": _phase_block("z", bool(phased[2]), backward),
        "__MASK__": templates.ownership_mask(codes, backward,
                                             zero=templates.COMPLEX_ZERO),
    })


# ---------------------------------------------------------------------------
# The constitutive kernel
# ---------------------------------------------------------------------------

_CONSTITUTIVE_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

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
    uint idx [[thread_position_in_grid]])
{
__GUARD__

__DECODE__

    // NO GHOST RULE, NO OWNERSHIP MASK AND NO PHASE: the constitutive sub-step
    // reads no neighbour and never sees a wrap (S:907-993). The coefficient index
    // is on the component's OWN axis — kps_x/kms_x for target 0, _y for 1, _z for
    // 2 (H_CONSTITUTIVE_TERMS / E_CONSTITUTIVE_TERMS, stepping.py:226-227). That
    // is MEEP's dsigw, NOT the dsig/dsigu cycle the curl recurrence uses.
    float kp_0 = kp0[i], km_0 = km0[i];
    float kp_1 = kp1[j], km_1 = km1[j];
    float kp_2 = kp2[k], km_2 = km2[k];

    // --- component 0 -----------------------------------------------------------
    // `prev` is read BEFORE `fw` is written (S:2083-2085). Reversed, this is wrong
    // only where kms != 0 -- INSIDE THE PML ONLY -- and looks like a slightly worse
    // absorber, not like a bug. inv_eps is a float32 volume indexed by the COMPLEX
    // CELL index (one real coefficient per cell, both planes), never word-doubled.
    float2 prev0 = w0[ii];
    float2 src0 = __SRC0__;
    w0[ii] = src0;
    float2 a0 = f0[ii];
    // Two SEPARATE accumulations, left to right: ((f + kps*src) - kms*prev). Each
    // product is the zero-imaginary complex form with the COEFFICIENT ON THE LEFT
    // (S:2086-2087, S:2093-2095).
    a0 = a0 + c_mul_coefficient_left(kp_0, src0);
    a0 = a0 - c_mul_coefficient_left(km_0, prev0);
    f0[ii] = a0;

    // --- component 1 -----------------------------------------------------------
    float2 prev1 = w1[ii];
    float2 src1 = __SRC1__;
    w1[ii] = src1;
    float2 a1 = f1[ii];
    a1 = a1 + c_mul_coefficient_left(kp_1, src1);
    a1 = a1 - c_mul_coefficient_left(km_1, prev1);
    f1[ii] = a1;

    // --- component 2 -----------------------------------------------------------
    float2 prev2 = w2[ii];
    float2 src2 = __SRC2__;
    w2[ii] = src2;
    float2 a2 = f2[ii];
    a2 = a2 + c_mul_coefficient_left(kp_2, src2);
    a2 = a2 - c_mul_coefficient_left(km_2, prev2);
    f2[ii] = a2;
}
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 226-227->227-228

#: The constitutive product per side. The H side reads B directly (mu = 1 is baked
#: into the array path too: ``stepping.update_H`` passes ``fields.Bx``); the E side
#: reads ``D * inv_eps`` with **D ON THE LEFT**, because the array path writes
#: ``source * fields.inverse_epsilon_for(component)`` (stepping.py:1011-1013).
#:
#: THE ORIENTATION IS KEPT BECAUSE IT IS THE ARRAY PATH'S, AND IT IS RECORDED HERE
#: THAT IT IS NOT LOAD-BEARING FOR A REAL COEFFICIENT. Measured on the exhaustive
#: 64-pattern signed-zero table (``gate_metal_complex`` leg ``zero_cross_terms``,
#: arm ``coefficient_left_orientation``): swapping to the coefficient-left spelling
#: moves 0/128 words. The real parts are commutative products with commutative
#: addends, and the imaginary parts are both a single rounding of the same exact
#: product plus a signed zero, whose addition is itself sign-commutative. The
#: corresponding mutation is therefore ``must_catch=False`` and is named as an
#: equivalence rather than left as an untested belief inherited from Triton.
#:
#: FOR A FULL COMPLEX PRODUCT THE ORIENTATION IS LOAD-BEARING and that IS measured:
#: the imaginary part fuses ``z_re*p_im`` in one spelling and ``p_re*z_im`` in the
#: other, which round differently. Mutation ``m13_phase_multiply_operands_swapped``
#: plants exactly that on the Bloch rotation and is caught.
_CONSTITUTIVE_PRODUCTS: Dict[str, Tuple[str, str, str]] = {
    "H": ("g0[ii]", "g1[ii]", "g2[ii]"),
    "E": ("c_mul_field_left(g0[ii], e0[ii])",
          "c_mul_field_left(g1[ii], e1[ii])",
          "c_mul_field_left(g2[ii], e2[ii])"),
}


def bloch_constitutive_source(side: str, expansion: str,
                              contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised complex ``bloch_constitutive_step`` source for one side.

    The Yee sub-lattice is NOT chosen here. The host binds ``kps_a``/``kms_a`` for H
    and ``kps_a_h``/``kms_a_h`` for E (stepping.py:948 vs :1015), so the kernel never
    chooses and cannot choose wrong — but the host can, which is why the gate
    carries a swap mutation for it.

    The three inverse-epsilon buffers stay in the signature on both sides so the
    buffer layout is one layout. On the H side the plan binds float32 placeholders
    and the substituted product never reads them.
    """
    if side not in _CONSTITUTIVE_PRODUCTS:
        raise ValueError(f"side must be one of {tuple(_CONSTITUTIVE_PRODUCTS)}, "
                         f"got {side!r}")
    products = _CONSTITUTIVE_PRODUCTS[side]
    return templates.substitute(_CONSTITUTIVE_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SRC0__": products[0],
        "__SRC1__": products[1],
        "__SRC2__": products[2],
    })


def split_plane_curl_signature() -> str:
    """The re/im-SPLIT curl signature, which must FAIL to compile. 35 bindings.

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
    arguments = ",\n".join(planes + coefficients + scalars + phases)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void split_plane_curl(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    f0re[idx] = f0im[idx];\n}\n")


def split_plane_binding_count() -> int:
    """How many buffers the split form needs — 35, above the measured ceiling."""
    return 9 * 2 + 6 + 5 + 6


def compile_bloch_curl(codes: Sequence[int], backward: bool, phased: Sequence[int],
                       expansion: str,
                       contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(bloch_curl_source(codes, backward, phased, expansion,
                                            contract)).bloch_pml_curl_step


def compile_bloch_constitutive(side: str, expansion: str,
                               contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(bloch_constitutive_source(
        side, expansion, contract)).bloch_constitutive_step


def enumerate_complex_sources(expansion: str,
                              contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit, keyed by a stable label.

    The phase triple is enumerated only over the axes a given boundary triple can
    legally phase — a metallic axis cannot carry one — so the count is the REACHABLE
    set rather than the Cartesian product, and a label that cannot be built is not
    fingerprinted as if it could.
    """
    out: Dict[str, str] = {}
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for cx in (templates.PERIODIC, templates.METALLIC):
            for cy in (templates.PERIODIC, templates.METALLIC):
                for cz in (templates.PERIODIC, templates.METALLIC):
                    codes = (cx, cy, cz)
                    for phased in _reachable_phase_flags(codes):
                        label = (f"bloch_pml_curl_step/{name}/{cx}{cy}{cz}/"
                                 f"ph{phased[0]}{phased[1]}{phased[2]}/{expansion}")
                        out[label] = bloch_curl_source(codes, backward, phased,
                                                       expansion, contract)
    for side in ("H", "E"):
        out[f"bloch_constitutive_step/{side}/{expansion}"] = (
            bloch_constitutive_source(side, expansion, contract))
    return out


def _reachable_phase_flags(codes: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    options = [(0, 1) if code == templates.PERIODIC else (0,) for code in codes]
    return tuple((x, y, z) for x in options[0] for y in options[1] for z in options[2])


# ---------------------------------------------------------------------------
# The expansion probe — how the arm is bound
# ---------------------------------------------------------------------------

def load_expansion_probe(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read the probe artifact JSON, or None when it is absent or unreadable.

    Unreadable is treated exactly like missing: both are refusals downstream, never
    a silent default arm.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    candidate = path if path is not None else os.environ.get(PROBE_PATH_ENVIRONMENT)
    if not candidate:
        return None
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except Exception:  # noqa: BLE001 - unreadable probe == missing probe
        return None
    return record if isinstance(record, dict) else None


def expansion_from_probe(record: Any) -> Optional[str]:
    """The single expansion arm a probe record licenses, or None.

    The record must name the backend the ENGINE holds (:data:`PROBE_BACKEND`),
    classify every pattern of :data:`PROBE_PATTERNS`, and leave exactly one arm
    standing once :data:`AMBIGUOUS_BOTH` patterns are removed. Three outcomes and
    all three are deliberate:

    * every pattern AMBIGUOUS_BOTH -> ``None``. A platform on which nothing
      discriminates has not been measured, and picking an arm from a probe that
      could not tell them apart is a guess wearing an artifact's clothes;
    * one arm among the discriminating patterns -> that arm;
    * two or more, or any pattern classified ``NEITHER`` -> ``None``. One
      specialisation cannot represent a platform whose orientations disagree, and a
      future disagreeing platform earns a new arm rather than a shrug.
    """
    if not isinstance(record, dict) or record.get("backend") != PROBE_BACKEND:
        return None
    patterns = record.get("patterns")
    if not isinstance(patterns, dict):
        return None
    decided: List[str] = []
    for name in PROBE_PATTERNS:
        value = patterns.get(name)
        if value == AMBIGUOUS_BOTH:
            continue
        if value not in templates.EXPANSIONS:
            return None
        decided.append(value)
    if len(set(decided)) != 1:
        return None
    return decided[0]


def _expansion_reasons(probe: Any = None) -> List[str]:
    """The clause that makes the arm binding a MEASUREMENT rather than a choice."""
    record = probe if probe is not None else load_expansion_probe()
    if record is None:
        return [f"no complex-multiply expansion probe artifact is available (set "
                f"{PROBE_PATH_ENVIRONMENT} or pass probe=); which arm the "
                f"{PROBE_BACKEND} reference takes is a measured platform fact and "
                f"may not be guessed"]
    if expansion_from_probe(record) is None:
        return [f"the expansion probe artifact is missing, ambiguous, disagreeing, "
                f"or not for the {PROBE_BACKEND} backend (it needs "
                f"backend={PROBE_BACKEND!r}, every pattern of {PROBE_PATTERNS} "
                f"classified, and exactly one arm standing among the "
                f"discriminating ones)"]
    return []


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


def _complex_grid_reasons(fields: Any, pml: Any, grid: Any, residency: Any,
                          probe: Any = None) -> List[str]:
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

    # 10. The expansion arm must come from a measured artifact.
    reasons.extend(_expansion_reasons(probe))

    return reasons


def complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                              residency: Any = None,
                              probe: Any = None) -> Coverage:
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

    reasons = _complex_grid_reasons(fields, pml, grid, residency, probe)

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
                                  residency: Any = None,
                                  probe: Any = None) -> Coverage:
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
    reasons = _complex_grid_reasons(fields, pml, grid, residency, probe)

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
    """Per-axis Bloch wrap factor, None on an unphased axis — never ``1+0j``.

    Resolved through ``grid.bloch_phase`` exactly as ``stepping._bloch_phases``
    resolves it (S:2343-2361): the grid value is exp(2*pi*i*k*L) with the Brillouin
    edge EXACTLY -1+0j (grid.py:1011-1017), and None means the multiply is SKIPPED —
    the skip IS the bit-identity of k = 0. The non-periodic check mirrors the array
    path's raise and is unreachable behind an admitting predicate; it is kept so a
    caller who skips the predicate is refused loudly rather than handed a mis-baked
    phase flag.
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


def phase_arguments(phases: Sequence[Optional[complex]], backward: bool,
                    ) -> Tuple[Tuple[int, int, int], Tuple[Tuple[float, float], ...]]:
    """Encode the phase table as (per-axis flags, three float2 pairs).

    The phase is rounded to complex64 BEFORE splitting — the array path's
    ``shifted.dtype.type(phase)`` (S:1862) — and for the BACKWARD sub-step the
    imaginary part is NEGATED, which is the conjugate ``_shift_down`` applies
    (S:1818-1822). Negation is exact, so the order of conjugation and rounding is
    immaterial. An unphased axis passes (1.0, 0.0) under flag 0, and the kernel
    emits no multiply for it at all.
    """
    import numpy  # noqa: PLC0415

    flags: List[int] = []
    values: List[Tuple[float, float]] = []
    for phase in phases:
        if phase is None:
            flags.append(0)
            values.append((1.0, 0.0))
            continue
        rounded = numpy.complex64(phase)
        real = float(numpy.float32(rounded.real))
        imag = float(numpy.float32(rounded.imag))
        if backward:
            imag = -imag
        flags.append(1)
        values.append((real, imag))
    return (flags[0], flags[1], flags[2]), tuple(values)


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
                 "phase_values", "expansion", "residency", "volumes")

    REPR_FIELDS = ("sub_step", "shape", "bc", "phased", "expansion")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, phased, phase_values,
                 expansion: str, residency: Any, targets, auxiliaries, sources,
                 coefficients, functions: Dict[str, Any],
                 volumes: Sequence[str]) -> None:
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
        self.expansion = str(expansion)
        self.residency = residency
        self.volumes = tuple(volumes)
        super().__init__(functions, (
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem, self.dtdx)
            + self.phase_values))


def _curl_functions(codes, backward: bool, phased, expansion: str,
                    contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_bloch_curl(codes, backward, phased, expansion, mode)
            for mode in contract_variants}


def plan_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                          residency: Any = None,
                          contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
                          probe: Any = None) -> Optional[ComplexPmlCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not complex_pml_curl_coverage(fields, pml, sub_step, residency, probe).covered:
        return None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
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
        sub_step, grid.shape, grid.dt / grid.dx, codes, flags, values, expansion,
        residency, targets, auxiliaries, sources, coefficients,
        _curl_functions(codes, spec["backward"], flags, expansion, contract_variants),
        volumes)


def plan_complex_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                      flat: Dict[str, Any], codes,
                                      phases: Sequence[Optional[complex]],
                                      dtdx: float, expansion: str, residency: Any,
                                      functions: Optional[Dict[str, Any]] = None,
                                      contract_variants: Sequence[str] = (
                                          shaders.CONTRACT_OFF,),
                                      ) -> ComplexPmlCurlPlan:
    """Build a plan from bare host arrays — the gate's and benchmark's route.

    ``phases`` is the per-axis ``Optional[complex]`` table; the conjugation for
    ``step_D`` is applied HERE, per sub-step, exactly as the engine route applies
    it. No predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.

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
        sub_step, shape, dtdx, codes, flags, values, expansion, residency,
        targets, auxiliaries, sources, coefficients,
        functions if functions is not None
        else _curl_functions(codes, spec["backward"], flags, expansion,
                             contract_variants),
        volumes)


class ComplexConstitutivePlan(KernelPlan):
    """A launchable, allocation-free COMPLEX ``dsigw`` constitutive sub-step.

    The Yee sub-lattice is chosen HERE and nowhere else: ``kps_a``/``kms_a`` for the
    H side, ``kps_a_h``/``kms_a_h`` for the E side (stepping.py:948 vs :1015). The
    kernel takes six coefficient pointers and never asks which lattice they came
    from, so a swap on this line is a silent half-cell error in the absorber profile
    — converged, smooth, and wrong. The gate carries a mutation for exactly it.
    """

    __slots__ = ("side", "shape", "n_elem", "expansion", "residency", "volumes")

    REPR_FIELDS = ("side", "shape", "expansion")

    def __init__(self, side: str, shape, expansion: str, residency: Any, targets,
                 auxiliaries, sources, inverse_epsilon, coefficients,
                 functions: Dict[str, Any], volumes: Sequence[str]) -> None:
        self.side = side
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.expansion = str(expansion)
        self.residency = residency
        self.volumes = tuple(volumes)
        # The H side has no inverse epsilon. The three buffers stay in the signature
        # so the layout is one layout, and REAL placeholders are bound: the
        # substituted product never reads them, but a `device const float*` argument
        # still has to bind, and a complex tensor there would be the wrong TYPE
        # rather than merely an unread one.
        placeholders = tuple(coefficients[:3] if inverse_epsilon is None
                             else inverse_epsilon)
        super().__init__(functions, (
            tuple(targets) + tuple(auxiliaries) + tuple(sources) + placeholders
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem)))


def _constitutive_functions(side: str, expansion: str,
                            contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_bloch_constitutive(side, expansion, mode)
            for mode in contract_variants}


def plan_complex_constitutive(fields: Any, pml: Any, side: str, residency: Any = None,
                              contract_variants: Sequence[str] = (
                                  shaders.CONTRACT_OFF,),
                              probe: Any = None) -> Optional[ComplexConstitutivePlan]:
    """Build a complex constitutive plan from the engine's objects, or None."""
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not complex_constitutive_coverage(fields, pml, side, residency, probe).covered:
        return None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
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
        side, fields.grid.shape, expansion, residency, targets, auxiliaries,
        sources, inverse_epsilon, coefficients,
        _constitutive_functions(side, expansion, contract_variants), volumes)


def plan_complex_constitutive_from_arrays(
        side: str, arrays: Dict[str, Any], flat: Dict[str, Any], expansion: str,
        residency: Any, functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        suffix_key: str = "") -> ComplexConstitutivePlan:
    """Build a complex constitutive plan from bare host arrays — the gate's route.

    ``suffix_key`` only namespaces the coefficient mirrors so a harness can bind the
    OTHER Yee sub-lattice under a different name in the same residency; it does not
    choose a lattice. Choosing is the caller's, which is what makes the half-cell
    swap a HOST mutation rather than a kernel one.
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
        side, shape, expansion, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        functions if functions is not None
        else _constitutive_functions(side, expansion, contract_variants), volumes)


# ---------------------------------------------------------------------------
# Registration — WIRED, as of tranche 2
# ---------------------------------------------------------------------------

#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}


def _curl_arm_coverage(context: Any, slot: str) -> Coverage:
    return complex_pml_curl_coverage(context.fields, context.pml, slot,
                                     context.residency,
                                     context.extra.get("complex_probe"))


def _curl_arm_plan(context: Any, slot: str) -> Any:
    return plan_complex_pml_curl(context.fields, context.pml, slot,
                                 context.residency, context.contract_variants,
                                 context.extra.get("complex_probe"))


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return complex_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.extra.get("complex_probe"))


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_complex_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants,
        context.extra.get("complex_probe"))


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
