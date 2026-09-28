"""Special-kz (``grid.beta``) PML curl on Metal — the real and Bloch arms.

WIRED AS OF TRANCHE 2 — six registered arms, and ``arms.register`` is called at the
bottom of this module. ``fastpath.plan_fast_path`` still returns ``None`` on every
branch and this module does not touch it; composition and dispatch are different
decisions and only the first has moved. Callers are ``plan_step``, the byte gate
(``parity/meep_gpu/gate_metal_special_kz.py``), the composition probes and the
laptop tests.

REGISTERING NOTHING WAS THE ONE SHAPE THAT COST SOMETHING, and this module used to
be the example. An arm absent from the registry cannot be swept by any other
family's composition probe — only by one that names this module — so the family
most likely to collide with :mod:`.complex_fields` was precisely the family a
registry-driven sweep could not see. The tranche-2 disjointness sweep reaches every
arm through :func:`.arms.registered`, which is why every family now registers.

THE SAFETY ARGUMENT IS UNCHANGED AND IS NOW MEASURED THROUGH THE TABLE: every
shipped real-field Metal predicate reaches ``coverage._grid_reasons``, whose clause
12 refuses a ``beta != 0`` run BY NAME, and ``complex_fields``' clause 8 refuses it
by name too. The predicates below invert that clause (they REQUIRE beta nonzero),
so no admitted overlap can exist in either direction. The sweep constructs the
configurations and calls the predicates rather than resting on this paragraph.

WHAT BETA IS (grid.py ``_resolve_beta``; stepping.py:754-811). A 2-D run carries
``exp(i*2*pi*beta*z)`` analytically, so ``d/dz`` on the invariant axis is the EXACT
factor ``i*2*pi*beta`` — an ``i*beta*zhat x`` cross product folded into the curl
(MEEP step_db.cpp:148-176). The family adds ONE term to the two curl sub-steps and
changes nothing else in the step loop:

* call sites: step_B (stepping.py:384-391) — Bx adds the Ey-center term at sign +1
  (:389), By the Ex-center term at sign -1 (:391), Bz nothing; step_D (:467-474) —
  Dx <- Hy at +1 (:443), Dy <- Hx at -1 (:445), Dz nothing;
* coefficient (:770-772): ``sign * 2*pi * grid.beta * grid.dt`` — **NO dtdx**, this
  is an analytic derivative and not a finite difference (:731-735); complex storage
  multiplies by ``+1j`` (B side) / ``-1j`` (D side); real storage is the same
  arithmetic under MEEP's implicit-i trick (step_db.cpp:148-160), a plain real add
  on both sides and not a second code path;
* rounding (:784): the coefficient is rounded ONCE on the host through
  ``partner_values.dtype.type(coefficient)``, then ``return -(c * partner)`` — the
  negation is of the PRODUCT, in curl sign convention, because the caller subtracts;
* position: added to ``curl`` AFTER ``_curl_from_operands`` (:342 / :429) and
  BEFORE ``_mask_non_owned_cells`` (:369 / :450) and ``_apply_curl`` (:374 / :455),
  so the increment rides the SAME split-field kms/sinv ladder as the finite
  difference curl, exactly and linearly;
* partners: the SAME-CELL component snapshots (:293 electric, :408 magnetic). Per
  ``B_CURL_TERMS``/``D_CURL_TERMS`` (:213-222) every beta partner is a CENTER
  operand the certified curl already loads — Bx: the second source Ey center; By:
  the first source Ex center; Dx: second Hy; Dy: first Hx — so the term costs no new
  pointer and no extra traffic, and the load reused must be the UNSHIFTED center
  one, never an ``ox``/``oy``/``oz`` shifted operand. In the complex arm the Bloch
  wrap rotation applies to SHIFTED operands only, so reusing the center registers
  keeps the beta partner UNROTATED by construction (a gate mutation plants the
  rotation and must be caught);
* no new state array, no ghost rule, no fold parity, no wrap. ``update_H`` /
  ``update_E`` (``_apply_constitutive_pml``, :2065) read nothing beta-dependent.

TWO KERNELS, one per storage family:

* ``beta_pml_curl_step`` — the certified ``shaders.pml_curl_step`` body plus the
  specialised beta term: one f32 multiply-and-subtract per affected target,
  reusing the loaded center operand;
* ``beta_bloch_pml_curl_step`` — the complex (``float2``) curl plus the same term
  through the general complex product with the COEFFICIENT ON THE LEFT (:784).

THE METAL FACTS THIS FAMILY APPLIES, each measured on this host rather than
inherited from the Triton port:

1. **Contraction is a SOURCE property here, not a launch keyword.** The Triton
   tranche had to certify this family with ``enable_fp_fusion=False`` because the
   multiply-subtract tail ``curl - (c*g)`` is exactly the shape a compiler
   contracts into an fma. On Metal that hazard is closed by the file-scope
   ``contract(off)`` directive baked into every source this module emits — and
   because it is a source property, the guard is a *different compiled kernel*
   rather than a different launch. The gate MEASURES the difference (the
   ``contract=fast`` variant must diverge) instead of asserting the pragma's
   presence. WHAT THAT MEANS HERE, stated: the beta tail cannot fuse with the
   ``dtdx`` curl that precedes it, so the term's POSITION between curl and mask is
   preserved by the same directive that preserves the curl's own grouping.
2. **The scalar ABI is clean.** A Python double bound to ``constant float&``
   arrives correctly rounded, so the host-rounded coefficient reaches the kernel as
   the array path's bits. That is why :func:`beta_curl_coefficients` rounds ONCE,
   on the host, exactly where stepping.py:811 rounds — and it is why Triton's dead
   ``tl.float64`` workaround has no analogue to port.
3. **Negation is a sign-bit operation.** ``-(x)`` preserves ``-0`` and subnormal
   bits here; Triton needed ``* -1.0`` because it lowers unary minus as ``0.0 - x``
   and canonicalizes a zero's sign. THAT WORKAROUND IS NOT INHERITED — it is
   measured away (``templates`` module docstring: ``0.0f - x`` misses 36/512 on the
   exhaustive signed-zero table, ``-x`` and ``* -1.0f`` both exact) — and the
   refuted spelling is a must-catch gate mutation.
4. **The expansion arm is a MEASURED property of the reference, not of the
   device.** The device spells whatever it is told to; the question is what
   ``numpy`` — the array path, the oracle — does. Measured 2026-08-15: numpy 2.4.3
   on this arm64 host takes FMA_V1 with the LEFT operand's real product fused
   (0/16,384 words; the four-rounded-product arm misses thousands). The complex arm
   therefore binds its ``EXPANSION`` from a probe artifact and REFUSES without one.

GROUPING CHOICES stepping.py does not force (the gate holds every one):

1. TWO host-rounded scalars per launch — one per sign — rather than one scalar
   negated in-kernel. The array path computes ``sign * 2*pi*beta*dt`` in f64 per
   call site and rounds each once (:770/:784); f64 negation and f32 rounding
   commute exactly, but binding both keeps the transcription literal per call site
   rather than resting on that identity.
2. ``curl - (c * g)`` carries the array path's ``curl + (-(c * g))``. IEEE-754
   defines subtraction AS addition of the negation, so the single subtract is the
   same bits on every input including signed zeros.
3. The complex product is the FULL multiply with the coefficient on the LEFT
   (:784 ``partner.dtype.type(coefficient) * partner_values``), through
   ``templates.c_mul`` whose first argument IS the left operand. ``c_re`` is a
   SIGNED ZERO produced by Python's own complex multiply and is passed through as
   an argument, never synthesized as ``+0.0``.
4. ``HAS_BETA`` is a source specialisation: the term is compiled in only when beta
   is nonzero. Engine-route plans always build 1 (the predicate requires beta
   nonzero, and the array path never enters the term at :356/:438); the 0 arm
   exists so the gate's identity leg can pin a beta = 0 build against the CERTIFIED
   plain curl on the same seeds.
5. Constitutive sub-steps on a beta run are NOT a new kernel, in EITHER storage:
   the restated predicates :func:`beta_run_constitutive_coverage` and
   :func:`beta_run_complex_constitutive_coverage` drop only the beta clause and
   delegate the arithmetic to the certified ``launch.ConstitutivePlan`` and
   ``complex_fields.ComplexConstitutivePlan`` respectively. The complex half was a
   named refusal until 2026-08-19, when the measurement behind "the constitutive
   sub-steps read nothing beta-dependent" was taken on complex storage too.
6. The fold is refused in BOTH storage families, by name, exactly as the Triton
   Phase A refuses it.
7. A conductivity on the curl targets is refused by name: these kernels transcribe
   the plain split-field recurrence only.
8. BOTH LAYOUT CLAUSES FAIL CLOSED, on both arms. The complex arm's own layout
   list once read an absent ``flags`` as "contiguous" and had no three-dimensional
   guard at all, and both were MEASURED admitting (see
   :func:`_complex_layout_reasons`) — a verdict of ``covered=True`` with an EMPTY
   reason tuple on a configuration whose plan builder would have raised. Admission
   by attribute absence is the thing this module refuses BY NAME in the
   conductivity, Bloch-phase and expansion-probe clauses; the layout clause now
   agrees with them and with :func:`_layout_reasons` on the real arm.

THE COMPLEX BODY'S RELATIONSHIP TO A FUTURE COMPLEX TRANCHE, stated rather than
left to be discovered: the complex arm below carries a full Bloch curl body
because no certified complex curl exists on this backend yet. When one lands, the
NAMED END STATE is to hoist the shared body into the fragment layer and have both
families substitute their own insert — deleting this copy outright, with no alias
and no re-export (the rename-outright rule). Until then this module's complex body is
pinned by its own gate against ``stepping.py``, which is the oracle either way.
"""

from __future__ import annotations

import math
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
    _layout_reasons,
    _susceptibility_reasons,
)
from . import arms, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import compile_source
from .launch import SUB_STEPS
from .plans import KernelPlan

__all__ = [
    "BETA_PROBE_PATTERN",
    "BETA_PROBE_PATTERNS",
    "BetaBlochPmlCurlPlan",
    "BetaPmlCurlPlan",
    "beta_bloch_curl_source",
    "beta_bloch_pml_curl_coverage",
    "beta_curl_coefficients",
    "beta_curl_source",
    "beta_expansion_from_probe",
    "beta_pml_curl_coverage",
    "beta_run_complex_constitutive_coverage",
    "beta_run_constitutive_coverage",
    "enumerate_sources",
    "load_expansion_probe",
    "plan_beta_bloch_pml_curl",
    "plan_beta_bloch_pml_curl_from_arrays",
    "plan_beta_pml_curl",
    "plan_beta_pml_curl_from_arrays",
    "plan_beta_run_complex_constitutive",
    "plan_beta_run_constitutive",
    "register_arms",
]

PERIODIC = templates.PERIODIC
METALLIC = templates.METALLIC

#: The probe artifact's environment override, and the pattern set this family
#: needs. The base four are the complex tranche's call-site orientations; the
#: FIFTH is this family's own and no earlier artifact carries it.
PROBE_PATH_ENVIRONMENT = "MEEP_GPU_METAL_EXPANSION_PROBE"

#: The NEW pattern: ``complex64 scalar (signed-zero real, +-imaginary) * complex64
#: array`` with the COEFFICIENT on the LEFT (stepping.py:811). The base patterns
#: cover only zero-imaginary coefficients and the field-left orientation; a general
#: scalar-left coefficient is a distinct dispatch case and is MEASURED, not assumed.
BETA_PROBE_PATTERN = "c8_mul_c8_imaginary_coefficient_left"

#: The base orientations, restated from the array path's own call sites:
#: ``xp.multiply(dtdx, total)`` (S:1635, python float LEFT), ``fu *= kms``
#: (S:1929-1935, field LEFT), ``kps * fw`` (S:2086-2095, coefficient LEFT) and the
#: Bloch phase ``shifted[plane] *= phase`` (S:1862, field LEFT).
BASE_PROBE_PATTERNS: Tuple[str, ...] = (
    "c8_mul_c8",
    "c8_mul_f4_field_left",
    "f4_mul_c8_coefficient_left",
    "python_float_left",
)

#: What a probe artifact must classify — and classify CONSISTENTLY — before the
#: complex beta curl may bind an ``EXPANSION``.
BETA_PROBE_PATTERNS: Tuple[str, ...] = BASE_PROBE_PATTERNS + (BETA_PROBE_PATTERN,)

#: The array module the probe must have measured. The Triton artifact names
#: ``cupy`` because that is ITS array path; here the oracle is the engine's own
#: NumPy, and an artifact cut on another backend licenses nothing.
PROBE_BACKEND = "numpy"


# ---------------------------------------------------------------------------
# The host coefficient — stepping.py:797-811, transcribed
# ---------------------------------------------------------------------------

def _numpy():
    """numpy, imported where it is used (this module is importable without it in
    the predicate-only role every Metal coverage module keeps)."""
    import numpy  # noqa: PLC0415

    return numpy


def beta_curl_coefficients(beta: float, dt: float, magnetic: bool,
                           complex_storage: bool):
    """The (plus-sign, minus-sign) beta coefficients, rounded EXACTLY as S:770-784.

    Per sign: ``coefficient = sign * 2*pi * beta * dt`` (:770); complex storage
    multiplies by ``+1j`` (magnetic) / ``-1j`` (electric) THROUGH Python's own
    complex arithmetic (:771-772) — which is what puts the SIGNED zero in the real
    word, and why that word is passed through rather than synthesized — then ONE
    rounding to the storage dtype (:784).

    Returns two f32 floats (real storage) or two ``(re, im)`` float pairs (complex
    storage). ``float()`` of a numpy.float32 preserves the zero's sign, and a
    Python float bound to ``constant float&`` arrives correctly rounded (the Metal
    scalar ABI is clean; measured), so the words the kernel receives are the array
    path's bits.

    ``beta`` AND ``dt`` ARE USED AS GIVEN — no ``float()`` normalisation — and that
    is the whole point rather than a detail. The expression is S:770's, evaluated on
    the objects the array path evaluates it on, so its intermediate PRECISION is
    whatever the caller's types make it. An earlier revision widened both through
    ``float()`` first, which collapses the chain to one f64 rounding; under NEP 50 a
    numpy-typed operand rounds the chain EARLIER, and the two spellings then differ.
    MEASURED 2026-08-15 on this host: with ``beta``/``dt`` as ``numpy.float32``,
    14,926 of 40,000 random draws produced a different float32 coefficient word. The
    path is live, not hypothetical — ``Grid._resolve_beta`` coerces ``beta`` to a
    Python float but NOTHING coerces ``dt``, so ``Grid(courant=numpy.float32(...))``
    hands this function a ``numpy.float32`` ``dt`` while ``stepping`` reads the same
    attribute. Pinned by
    ``test_metal_special_kz.test_the_coefficient_matches_stepping_on_numpy_typed_grid_scalars``.
    """
    import numpy  # noqa: PLC0415

    out: List[Any] = []
    for sign in (1.0, -1.0):
        coefficient: Any = sign * 2.0 * math.pi * beta * dt  # :770, verbatim
        if complex_storage:
            coefficient = coefficient * (1j if magnetic else -1j)  # :771-772
            rounded = numpy.complex64(coefficient)  # :784 — dtype.type, once
            out.append((float(numpy.float32(rounded.real)),
                        float(numpy.float32(rounded.imag))))
        else:
            out.append(float(numpy.float32(coefficient)))  # :784 — once
    return tuple(out)


# ---------------------------------------------------------------------------
# The REAL beta curl source
# ---------------------------------------------------------------------------

_BETA_CURL_TEMPLATE = r"""
__HEADER__

__CONTRACT__

kernel void beta_pml_curl_step(
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
    constant float&     beta_plus  [[buffer(20)]],
    constant float&     beta_minus [[buffer(21)]],
    uint idx [[thread_position_in_grid]])
{
__GUARD__

    int nxi = int(nx);
__DECODE__
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

    // --- the curl (stepping._curl_from_operands:1601): DO NOT flatten these parens
    float curl0 = dtdx * ((c_y - c) + (b - b_z));
    float curl1 = dtdx * ((a_z - a) + (c - c_x));
    float curl2 = dtdx * ((b_x - b) + (a - a_y));

    // --- the beta term (stepping._special_kz_beta_term:727-784) ------------------
    // AFTER the dtdx curl, BEFORE the ownership mask -- the array path's order
    // (S:356-363 / S:438-445 against S:369 / S:450). NO dtdx on the term: it is an
    // analytic derivative (S:731-735). The partners are the UNSHIFTED CENTER loads
    // the curl already made: target 0 takes the second source's `b` at sign +1,
    // target 1 the first source's `a` at sign -1; target 2 gets nothing
    // (step_db.cpp:148-176 runs `cc` over d_c in {X, Y} only). `curl - (c*g)` IS
    // the array path's `curl + (-(c*g))` by IEEE-754's definition of subtraction.
__BETA__

    // --- ownership mask (stepping._mask_non_owned_cells:1865) -------------------
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

    // --- split-field recurrence (stepping._apply_pml_update:1905) ---------------
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];

    float p0 = u0[ii];
    float n0 = ((p0 * km_y) - curl0) * si_y;
    float v0 = (((f0[ii] * km_z) + n0) - p0) * si_z;

    float p1 = u1[ii];
    float n1 = ((p1 * km_z) - curl1) * si_z;
    float v1 = (((f1[ii] * km_x) + n1) - p1) * si_x;

    float p2 = u2[ii];
    float n2 = ((p2 * km_x) - curl2) * si_x;
    float v2 = (((f2[ii] * km_y) + n2) - p2) * si_y;

    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
"""

#: The real beta insert, and the null one. The null is a COMMENT rather than an
#: omitted placeholder so a reader of a HAS_BETA=0 source can see that the slot
#: exists and is deliberately empty.
_REAL_BETA_INSERT = ("    curl0 = curl0 - (beta_plus * b);\n"
                     "    curl1 = curl1 - (beta_minus * a);")
_NO_BETA_INSERT = "    // HAS_BETA = 0: beta = 0 runs belong to the certified curl"


def beta_curl_source(codes: Sequence[int], backward: bool, has_beta: bool = True,
                     contract: str = templates.CONTRACT_OFF) -> str:
    """The specialised REAL ``beta_pml_curl_step`` source.

    ``codes`` is the per-axis PERIODIC/METALLIC triple ``_boundary_kinds``
    resolves; ``backward`` selects step_D's negated strides over step_B's forward
    ones; ``has_beta`` compiles the term in or out. All three are baked into the
    string, as the certified curl bakes its own — one specialisation, one source,
    one memo key.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    return templates.substitute(_BETA_CURL_TEMPLATE, {
        "__HEADER__": "#include <metal_stdlib>\nusing namespace metal;",
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__BETA__": _REAL_BETA_INSERT if has_beta else _NO_BETA_INSERT,
        "__MASK__": templates.ownership_mask(codes, backward),
    })


# ---------------------------------------------------------------------------
# The COMPLEX (Bloch) beta curl source
# ---------------------------------------------------------------------------
#
# THE BINDING BUDGET, measured: Metal's buffer attribute indices run 0..30
# (device.MAX_BUFFER_BINDINGS). This kernel binds 9 float2 volumes + 6 coefficient
# vectors + 4 extents + dtdx + 6 phase words + 4 beta words = 30. That is ONE
# below the ceiling and it is why the complex volumes MUST be float2: a re/im-split
# build needs 9 more buffers and is a compile error, not a slower kernel.

_BETA_BLOCH_CURL_TEMPLATE = r"""
__HEADER__

__CONTRACT__

__COMPLEX_HELPERS__

kernel void beta_bloch_pml_curl_step(
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
    constant float&      pxr     [[buffer(20)]],
    constant float&      pxi     [[buffer(21)]],
    constant float&      pyr     [[buffer(22)]],
    constant float&      pyi     [[buffer(23)]],
    constant float&      pzr     [[buffer(24)]],
    constant float&      pzi     [[buffer(25)]],
    constant float&      bpr     [[buffer(26)]],
    constant float&      bpi     [[buffer(27)]],
    constant float&      bmr     [[buffer(28)]],
    constant float&      bmi     [[buffer(29)]],
    uint idx [[thread_position_in_grid]])
{
__GUARD__

    int nxi = int(nx);
__DECODE__
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

    // --- wrapped-lane predicates, one plane per axis (stepping:1862 writes the
    //     phase onto the WRAPPED FACE only) ---------------------------------------
__WRAPPED__

    const float2 zero2 = float2(0.0f, 0.0f);
    float2 a   = g0[ii];
    float2 b   = g1[ii];
    float2 c   = g2[ii];
    float2 a_y = vy ? g0[oy] : zero2;
    float2 a_z = vz ? g0[oz] : zero2;
    float2 b_x = vx ? g1[ox] : zero2;
    float2 b_z = vz ? g1[oz] : zero2;
    float2 c_x = vx ? g2[ox] : zero2;
    float2 c_y = vy ? g2[oy] : zero2;

    // --- Bloch phase on the wrapped lane, BEFORE the difference -----------------
    // SHIFTED operands only. The beta partners (the a, b CENTERS) are never
    // rotated, which is what keeps the beta term's operand the array path's
    // unrotated snapshot (S:293/:408) by construction rather than by care.
__PHASE__

    // --- the curl (stepping._curl_from_operands:1601) ---------------------------
    // The parens are the array path's and are not flattened; the dtdx multiply is
    // `xp.multiply(dtdx, total)` (S:1635) -- COEFFICIENT ON THE LEFT.
    float2 t0 = ((c_y - c) + (b - b_z));
    float2 t1 = ((a_z - a) + (c - c_x));
    float2 t2 = ((b_x - b) + (a - a_y));
    float2 curl0 = c_mul_coefficient_left(dtdx, t0);
    float2 curl1 = c_mul_coefficient_left(dtdx, t1);
    float2 curl2 = c_mul_coefficient_left(dtdx, t2);

    // --- the beta term (stepping._special_kz_beta_term:727-784) ------------------
    // Coefficient LEFT (S:784), the FULL complex product, on the CENTER partners.
    // Complex subtraction is component-wise and IS the array path's complex
    // `curl + (-(c*g))`: complex negation negates both words.
__BETA__

    // --- ownership mask (stepping._mask_non_owned_cells:1865) -------------------
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

    // --- split-field recurrence (stepping._apply_pml_update:1905) ---------------
    // `fu *= kms` / `field *= kms_u` are FIELD-LEFT multiplies by a real scalar
    // (S:1929-1935); the zero cross terms are the array path's and are spelled.
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];

    float2 p0 = u0[ii];
    float2 q0 = c_mul_field_left(p0, km_y) - curl0;
    float2 n0 = c_mul_field_left(q0, si_y);
    float2 r0 = (c_mul_field_left(f0[ii], km_z) + n0) - p0;
    float2 v0 = c_mul_field_left(r0, si_z);

    float2 p1 = u1[ii];
    float2 q1 = c_mul_field_left(p1, km_z) - curl1;
    float2 n1 = c_mul_field_left(q1, si_z);
    float2 r1 = (c_mul_field_left(f1[ii], km_x) + n1) - p1;
    float2 v1 = c_mul_field_left(r1, si_x);

    float2 p2 = u2[ii];
    float2 q2 = c_mul_field_left(p2, km_x) - curl2;
    float2 n2 = c_mul_field_left(q2, si_x);
    float2 r2 = (c_mul_field_left(f2[ii], km_y) + n2) - p2;
    float2 v2 = c_mul_field_left(r2, si_y);

    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
"""

_COMPLEX_BETA_INSERT = (
    "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);\n"
    "    curl1 = curl1 - c_mul(float2(bmr, bmi), a);")

#: Which operands each axis's phase rotates, per the curl's own stencil. The x
#: wrap touches the two operands shifted along x (``b_x``, ``c_x``), and so on.
_PHASE_OPERANDS = {"x": ("b_x", "c_x"), "y": ("a_y", "c_y"), "z": ("a_z", "b_z")}


def _phase_lines(phased: Sequence[int]) -> str:
    """The wrapped-lane rotation, per phased axis, FIELD on the left (S:1862).

    NO ``backward`` ARGUMENT, deliberately: the sub-step's direction enters the
    phase only as a CONJUGATION, and that is done once on the host in
    :func:`bloch_phase_words` before the words are bound. Taking a direction here
    too would give the same fact two homes and let them disagree.
    """
    words = {"x": ("pxr", "pxi"), "y": ("pyr", "pyi"), "z": ("pzr", "pzi")}
    flags = {"x": "wx", "y": "wy", "z": "wz"}
    lines: List[str] = []
    for index, axis in enumerate("xyz"):
        if not phased[index]:
            continue
        re, im = words[axis]
        for operand in _PHASE_OPERANDS[axis]:
            lines.append(f"    {operand} = {flags[axis]} ? "
                         f"c_mul({operand}, float2({re}, {im})) : {operand};")
    return "\n".join(lines) or "    // no phased axis: no Bloch rotation"


def _wrapped_lines(backward: bool) -> str:
    """Which plane wraps: the LOW face on a backward shift, the HIGH face forward."""
    if backward:
        return "    bool wx = (i == 0), wy = (j == 0), wz = (k == 0);"
    return ("    bool wx = (i == nxi - 1), wy = (j == nyi - 1), "
            "wz = (k == nzi - 1);")


def beta_bloch_curl_source(codes: Sequence[int], backward: bool,
                           phased: Sequence[int], expansion: str,
                           has_beta: bool = True,
                           contract: str = templates.CONTRACT_OFF) -> str:
    """The specialised COMPLEX ``beta_bloch_pml_curl_step`` source.

    ``phased`` is the per-axis 0/1 flag for a nonzero Bloch phase and ``expansion``
    names the MEASURED complex-multiply arm (``NAIVE`` / ``FMA_V1``) — never a
    default, because which arm the reference takes is a platform fact.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3 or len(phased) != 3:
        raise ValueError(f"codes and phased must be per-axis triples, got "
                         f"{codes!r} / {phased!r}")
    return templates.substitute(_BETA_BLOCH_CURL_TEMPLATE, {
        "__HEADER__": "#include <metal_stdlib>\nusing namespace metal;",
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__COMPLEX_HELPERS__": templates.complex_helpers(expansion),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__WRAPPED__": _wrapped_lines(backward),
        "__PHASE__": _phase_lines(phased),
        "__BETA__": (_COMPLEX_BETA_INSERT if has_beta else
                     "    // HAS_BETA = 0: the identity arm, beta compiled out"),
        "__MASK__": templates.ownership_mask(codes, backward,
                                             zero=templates.COMPLEX_ZERO),
    })


def _emittable_bloch_specialisations() -> List[Tuple[Tuple[int, int, int],
                                                     Tuple[int, int, int]]]:
    """Every ``(codes, phased)`` pair the COMPLEX plan builder can produce.

    Derived from the two structural rules the shipped path enforces rather than
    from a hand-written list — which is what the list used to be, and it was WRONG:

    * a phase rides only a PERIODIC axis. ``bloch_phase_words`` reads
      ``grid.bloch_phase(axis)`` only where ``kinds[axis] == 'periodic'``, and
      :func:`_beta_complex_grid_reasons` clause 5 refuses a phase anywhere else;
    * the INVARIANT axis carries no phase. Clause 5b refuses ``k_z != 0`` (and the
      ``Grid`` refuses it at construction), so ``phased[2]`` is always 0.

    Boundary codes run over all eight triples, matching the real arm. ``cz =
    METALLIC`` is not constructible — ``Grid`` refuses a metallic invariant axis —
    so those rows are a deliberate SUPERSET, which is the safe direction for a weld.
    """
    out: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    for cx in (PERIODIC, METALLIC):
        for cy in (PERIODIC, METALLIC):
            for cz in (PERIODIC, METALLIC):
                for px in ((0, 1) if cx == PERIODIC else (0,)):
                    for py in ((0, 1) if cy == PERIODIC else (0,)):
                        out.append(((cx, cy, cz), (px, py, 0)))
    return out


def enumerate_sources(contract: str = templates.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation the shipped plans can emit, keyed by a stable label.

    The gate fingerprints these exactly as the certified pair is fingerprinted, so
    a compiler-visible edit to any one of them is a fingerprint event rather than a
    silent recompile. Both ``HAS_BETA`` arms are enumerated because the 0 arm is a
    SHIPPED kernel — the identity leg launches it.

    THE COMPLEX HALF USED TO BE A HAND-WRITTEN SUBSET AND THE WELD WAS HOLED WHERE
    THE GATE STOOD. It listed two boundary triples and three phase triples, so a
    metallic Y axis — the gate's OWN ``c3_brillouin_edge_metallic_y`` case, codes
    ``(0, 1, 0)`` — emitted a source whose sha256 appeared in no
    ``provenance.json`` row, on both sub-steps. Measured 2026-08-15: 32 of the
    reachable complex specialisations were absent, including the two the gate
    launched. The set is now DERIVED (:func:`_emittable_bloch_specialisations`), and
    ``test_metal_special_kz`` asserts that every source the family's own case matrix
    launches is in it — so the enumeration cannot drift behind the emitter again.
    """
    out: Dict[str, str] = {}
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for cx in (PERIODIC, METALLIC):
            for cy in (PERIODIC, METALLIC):
                for cz in (PERIODIC, METALLIC):
                    for has_beta in (True, False):
                        label = (f"beta_pml_curl_step/{name}/{cx}{cy}{cz}/"
                                 f"beta{int(has_beta)}")
                        out[label] = beta_curl_source((cx, cy, cz), backward,
                                                      has_beta, contract)
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for codes, phased in _emittable_bloch_specialisations():
            for expansion in sorted(templates.EXPANSIONS):
                for has_beta in (True, False):
                    label = (f"beta_bloch_pml_curl_step/{name}/"
                             f"{''.join(str(code) for code in codes)}/"
                             f"ph{''.join(str(flag) for flag in phased)}/"
                             f"{expansion}/beta{int(has_beta)}")
                    out[label] = beta_bloch_curl_source(
                        codes, backward, phased, expansion, has_beta, contract)
    return out


# ---------------------------------------------------------------------------
# The probe contract — an EXPANSION is measured or it is refused
# ---------------------------------------------------------------------------

def load_expansion_probe(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The measured probe artifact, or None when there is none.

    None is a REFUSAL upstream, never a default: which arm the reference takes is
    a property of the installed numpy on this machine and cannot be inferred from
    the source.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    candidate = path or os.environ.get(PROBE_PATH_ENVIRONMENT)
    if not candidate:
        return None
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


#: The verdict that constrains nothing. It is a real answer, not a failure: with a
#: coefficient whose imaginary (or real) word is exactly ±0 the fused arm's extra
#: product is EXACT, so the two transcriptions agree bit for bit and the pattern
#: cannot express a preference.
AMBIGUOUS = "AMBIGUOUS_BOTH"


def beta_expansion_from_probe(record: Any) -> Optional[str]:
    """The single ``EXPANSION`` arm the complex beta curl may bind, or None.

    THE RULE, and it is a MEASURED correction to the Triton track's rather than a
    copy of it. Triton accepted ``AMBIGUOUS_BOTH`` on the beta pattern ALONE and
    required the base four to agree on a definite arm, because on CuPy the base
    four discriminate. On this host they do not, and that is arithmetic rather than
    weakness: three of the four base call sites multiply by a REAL scalar, so one
    of the two cross products is exactly ±0, the fused arm's extra term is exact,
    and both transcriptions produce identical bytes. Measured 2026-08-15:

        c8_mul_c8                            FMA_V1           (discriminates)
        c8_mul_f4_field_left                 AMBIGUOUS_BOTH
        f4_mul_c8_coefficient_left           AMBIGUOUS_BOTH
        python_float_left                    AMBIGUOUS_BOTH
        c8_mul_c8_imaginary_coefficient_left AMBIGUOUS_BOTH

    Requiring all five to name one arm would refuse a host that HAS answered the
    question, so the rule is what the measurement supports:

    1. the record must name this backend and carry EVERY pattern of
       :data:`BETA_PROBE_PATTERNS` — a record cut before this tranche lacks the
       fifth and licenses nothing here, even though it would license base kernels;
    2. any verdict outside ``{NAIVE, FMA_V1, AMBIGUOUS_BOTH}`` — ``NEITHER``
       above all — is a REFUSAL BY NAME. A platform whose reference matches no
       transcription is exactly the case that must not be guessed at;
    3. AT LEAST ONE pattern must DISCRIMINATE. All-ambiguous is not a licence: it
       would mean nothing was measured and the arm was chosen by default, which is
       the guess this whole mechanism exists to prevent;
    4. every discriminating pattern must name the SAME arm.
    """
    if not isinstance(record, dict) or record.get("backend") != PROBE_BACKEND:
        return None
    patterns = record.get("patterns")
    if not isinstance(patterns, dict):
        return None
    definite: List[str] = []
    for name in BETA_PROBE_PATTERNS:
        value = patterns.get(name)
        if value == AMBIGUOUS:
            continue  # both arms reproduce the reference: no constraint
        if value not in templates.EXPANSIONS:
            return None  # missing, NEITHER, or junk -> refusal by name
        definite.append(value)
    if not definite or len(set(definite)) != 1:
        return None
    return definite[0]


def _beta_expansion_reasons(probe: Any = None) -> List[str]:
    """The complex arm's probe clause: refusal BY NAME when the artifact cannot license."""
    record = probe if probe is not None else load_expansion_probe()
    if record is None:
        return [f"no complex-multiply expansion probe artifact is available "
                f"(set {PROBE_PATH_ENVIRONMENT}); the EXPANSION arm is a MEASURED "
                f"property of this host's reference and may not be guessed — the "
                f"beta tranche additionally requires the {BETA_PROBE_PATTERN!r} "
                f"pattern, which no earlier artifact carries"]
    if beta_expansion_from_probe(record) is None:
        return [f"the expansion probe artifact is missing, ambiguous, not for the "
                f"{PROBE_BACKEND!r} backend, or lacks the {BETA_PROBE_PATTERN!r} "
                f"pattern this tranche adds (needs backend={PROBE_BACKEND!r} and "
                f"agreeing {BETA_PROBE_PATTERNS})"]
    return []


# ---------------------------------------------------------------------------
# Coverage — positive enumeration, named refusals
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors `coverage._grid_reasons` so the two can be diffed.
# Clause 12 (beta) is INVERTED in both families: this product REQUIRES beta != 0
# where every shipped Metal kernel requires 0, which is why no admitted-overlap
# ambiguity can arise — and the composition probe MEASURES that rather than
# resting on this comment.

def _beta_real_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every REAL-family beta predicate shares."""
    reasons: List[str] = []

    # 1. The Metal backend, the host array module and the subnormal policy. Shared
    #    with the certified predicate: one definition, imported.
    reasons.extend(_metal_backend_reasons(grid))

    # 2. Real storage REQUIRED: a complex-storage beta run belongs to the complex
    #    beta arm (kz_2d='complex' forces complex64).
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: a complex-storage beta run "
                       "belongs to the complex beta arm")

    # 3. An absorber that actually absorbs (split-field family only).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "split-field path only; no-PML beta is out of tranche)")

    # 4. Only the two ghost rules the curl writes.
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

    # 5. No mirror plane. The folded real beta curl is a separate family: the fold
    #    changes the ghost rule, adds a parity mask and changes the STORED EXTENT.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (this kernel does not carry the "
                       "fold; folded_beta's real arm is that product and CARRIES "
                       "it)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian, effective 2-D. The Grid refuses beta off 2-D Cartesian at
    #    construction (grid._resolve_beta; MEEP fields.cpp:546-547) — RESTATED here,
    #    never inferred from the constructor guard.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried (and the "
                       "grid itself refuses beta there)")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")
    if int(getattr(grid, "dimensions", 0)) != 2:
        reasons.append(f"grid dimensions={getattr(grid, 'dimensions', None)!r} is "
                       f"not the effective-2-D grid beta requires (MEEP "
                       f"fields.cpp:546-547)")

    # 7. k = 0. Real storage carries no Bloch phase; in-plane k with beta is the
    #    COMPLEX arm's domain.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r} (real "
                       f"storage carries no Bloch phase; in-plane k + beta is the "
                       f"complex arm's domain)")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 9a/9b. A registered susceptibility must be one this package understands.
    reasons.extend(_susceptibility_reasons(fields))

    # 9c. Stored E — the invariant behind admitting dispersion for the curl.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 10/11. No instantaneous nonlinearity, no BFAST.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is a "
                       "separate family)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is a "
                       "separate family)")

    # 12 (INVERTED). Beta must be NONZERO: a beta = 0 run belongs to the certified
    #     plain curl, and the array path never enters the term (S:356/:438).
    if float(getattr(grid, "beta", 0.0)) == 0.0:
        reasons.append("grid.beta is zero: this product exists only for special_kz "
                       "runs; a beta = 0 run belongs to the certified plain curl")

    # 12b. Real storage + off-diagonal epsilon + beta: stepping RAISES
    #      (stepping.py:800-810) and MEEP aborts (fields.cpp:548-549) — the
    #      implicit-i trick cancels only while TE and TM stay uncoupled. Refused for
    #      the WHOLE real family, not only update_E: the run raises inside the first
    #      beta curl.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("off-diagonal epsilon with beta in REAL storage: the "
                       "implicit-i trick no longer cancels (stepping.py:800-810 "
                       "raises; MEEP fields.cpp:548-549 aborts) — the run needs "
                       "force_complex_fields=True")

    return reasons


def _curl_conductivity_reasons(fields: Any, sub_step: Optional[str]) -> List[str]:
    """Clause 8, strict: a conductivity on this curl's targets is refused by name.

    A missing or non-callable ``condfac_for`` is refused OUTRIGHT rather than read
    as "no conductivity": inferring absence from an unreadable table is admission
    by attribute absence.
    """
    reasons: List[str] = []
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        return ["fields does not expose condfac_for; an unreadable conductivity "
                "table is not an absent one"]
    targets = (CURL_SUB_STEPS[sub_step] if sub_step is not None else CURL_TARGETS)
    for target in targets:
        try:
            conductive = reader(target) is not None
        except Exception as exc:  # noqa: BLE001 - unreadable means not covered
            reasons.append(f"condfac_for({target!r}) raised {exc!r}")
            continue
        if conductive:
            reasons.append(
                f"a conductivity is installed on {target}: this kernel transcribes "
                f"the plain split-field recurrence only, and the conductive family "
                f"refuses beta in its own predicate — the beta-under-condinv "
                f"composition is a named follow-up, not a silent overlap")
    return reasons


def beta_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                           residency: Any = None) -> Coverage:
    """May the REAL Metal beta curl step this (fields, pml, sub_step)?"""
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _beta_real_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_curl_conductivity_reasons(fields, sub_step))

    # 13/14. The NAMED SUB-STEP's nine arrays and their layout. Deliberately
    #        narrower than the certified predicate's 18-volume set, matching the
    #        per-sub-step scoping the Triton beta tranche records: every pointer
    #        the launched kernel touches is still checked, and the engine route
    #        allocates all 18 together so no engine-built configuration changes
    #        verdict.
    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def _complex_layout_reasons(fields: Any, shape: Sequence[int],
                            names: Sequence[str]) -> List[str]:
    """Complex64, C-contiguous, ``grid.shape``, three-dimensional, int32-indexable.

    Written here rather than imported from the Triton complex module because that
    module's clause list is about a CuPy array; the questions are the same and the
    dtype answer is the only one that differs from :func:`_layout_reasons`.

    IT FAILS CLOSED ON BOTH READS, and both were MEASURED admitting before they did:

    * a ``grid.shape`` that is not a triple. The kernel decodes ``i/j/k`` and
      :class:`BetaBlochPmlCurlPlan` multiplies ``shape[0..2]``, so a two-axis shape
      is an ``IndexError`` in the plan builder — but only if the predicate refuses
      it first. It did not: with a 2-D ``grid.shape`` AND 2-D volumes (they agree,
      so the per-volume clause is silent) and ``_coefficient_reasons`` skipped by
      its own ``len(shape) == 3`` guard, this returned ``covered=True`` with an
      EMPTY reason tuple. :func:`_layout_reasons` refuses the same shape by name on
      the real arm; the two now agree;
    * a volume whose ``flags`` cannot be read. The clause used to be
      ``flags is not None and not getattr(flags, "c_contiguous", True)`` — an
      absent ``flags``, or a ``flags`` with no ``c_contiguous``, was read as
      "contiguous". Measured: a stand-in exposing only ``dtype`` and ``shape``
      was ADMITTED with no reason at all. That is admission by attribute absence,
      which this module refuses BY NAME in three other clauses (the conductivity
      table, the Bloch phase table, the expansion probe) and now refuses here.
    """
    import numpy as np  # noqa: PLC0415

    expected = tuple(int(n) for n in shape)
    if len(expected) != 3:
        return [f"grid shape {expected!r} is not three-dimensional (the kernel "
                f"decodes i/j/k from a flat index and the plan sizes its dispatch "
                f"from three extents)"]
    reasons: List[str] = []
    for name in names:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if np.dtype(getattr(array, "dtype", None)) != np.complex64:
            reasons.append(f"{name} is {getattr(array, 'dtype', None)!r}, not "
                           f"complex64 (this arm mirrors complex volumes as "
                           f"float2; a float32 volume belongs to the real arm)")
        if tuple(getattr(array, "shape", ())) != expected:
            reasons.append(f"{name} has shape {getattr(array, 'shape', None)!r}, "
                           f"not {expected!r}")
        flags = getattr(array, "flags", None)
        if not bool(getattr(flags, "c_contiguous", False)):
            reasons.append(f"{name} is not C-contiguous (or does not report its "
                           f"layout; an unreadable layout is not a contiguous one)")
    count = 1
    for extent in expected:
        count *= int(extent)
    if count > 2 ** 31 - 1:
        reasons.append(f"{count} cells exceeds the int32 index bound the kernel uses")
    return reasons


def _beta_complex_grid_reasons(fields: Any, pml: Any, grid: Any,
                               probe: Any = None) -> List[str]:
    """The clauses every COMPLEX-family beta predicate shares."""
    reasons: List[str] = []

    reasons.extend(_metal_backend_reasons(grid))

    # 2 (INVERTED). Complex64 storage REQUIRED.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append("storage is real float32 (neither force_complex_fields nor "
                       "a nonzero k_point): a real beta run belongs to the real "
                       "beta arm")

    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the complex "
                       "split-field path only)")

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
        reasons.append("a mirror plane is active (this kernel does not carry the "
                       "fold; folded_beta's complex arm is that product and "
                       "CARRIES it, constitutive pair included)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried (and the "
                       "grid itself refuses beta there)")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")
    if int(getattr(grid, "dimensions", 0)) != 2:
        reasons.append(f"grid dimensions={getattr(grid, 'dimensions', None)!r} is "
                       f"not the effective-2-D grid beta requires")

    # 5. Per-axis phase consistency: a phased axis must resolve PERIODIC; a
    #    metallic axis must carry k = 0; an unreadable phase is REFUSED, never
    #    admitted as unphased.
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
                            f"axis {axis} carries Bloch phase {phase!r} but "
                            f"resolved to {kind!r}; only a periodic wrap can carry "
                            f"a phase")
            if kind == "metallic" and float(k_point[axis]) != 0.0:
                reasons.append(f"axis {axis} is metallic with k component "
                               f"{k_point[axis]!r}; a PEC wall gives the axis no "
                               f"lattice vector for the phase")

    # 5b. No k on the INVARIANT axis: beta IS that axis's analytic dependence, and
    #     the Grid refuses the pairing at construction. RESTATED, never inferred.
    if len(k_point) > 2 and float(k_point[2]) != 0.0:
        reasons.append(f"k_point component z = {k_point[2]!r} rides the invariant "
                       f"axis whose dependence beta already carries analytically")

    reasons.extend(_curl_conductivity_reasons(fields, None))

    # 7. No dispersion under complex storage on this backend.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: complex-storage ADE is a "
                       "future tranche")
    reasons.extend(_susceptibility_reasons(fields))

    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (a separate family)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (a separate family)")
    if float(getattr(grid, "beta", 0.0)) == 0.0:
        reasons.append("grid.beta is zero: this product exists only for special_kz "
                       "runs")
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 13. EXPANSION binding requires a MEASURED probe artifact carrying the
    #     extended pattern set.
    reasons.extend(_beta_expansion_reasons(probe))
    return reasons


def beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                 residency: Any = None,
                                 probe: Any = None) -> Coverage:
    """May the COMPLEX Metal beta curl step this (fields, pml, sub_step)?"""
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _beta_complex_grid_reasons(fields, pml, grid, probe)
    reasons.extend(_residency_declaration_reasons(residency))

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
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def beta_run_constitutive_coverage(fields: Any, pml: Any, side: str,
                                   residency: Any = None) -> Coverage:
    """May the CERTIFIED Metal constitutive kernel step a REAL beta run?

    The certified predicate's clause set with the beta clause INVERTED and nothing
    else changed: the constitutive sub-steps read nothing beta-dependent
    (``_apply_constitutive_pml``, stepping.py:2112), so admission delegates the
    ARITHMETIC to the certified ``launch.ConstitutivePlan`` unchanged — no new
    kernel and no new sub-step.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _beta_real_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))

    if side == "E" and (getattr(fields, "has_polarizations", False)
                        or (getattr(fields, "polarizations", ()) or ())):
        reasons.append("a susceptibility is registered: update_E's source is "
                       "(D - sum P), not D — that configuration belongs to the ADE "
                       "kernel, exactly as the certified predicate rules")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


def beta_run_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                           residency: Any = None,
                                           probe: Any = None) -> Coverage:
    """May the CERTIFIED COMPLEX constitutive kernel step an UNFOLDED beta run?

    THIS WAS AN UNCONDITIONAL REFUSAL UNTIL 2026-08-19, and the refusal was correct
    for as long as nobody had restated the predicate: ``complex_fields``' clause 8
    refuses ``grid.beta`` BY NAME so a complex beta run lands here rather than being
    taken there silently, and :mod:`.folded_beta` restated it only for a FOLDED beta
    run. The unfolded case had no product, so an unfolded complex beta run took its
    curls on the device and its constitutive pair on the array path.

    IT IS NOW THE SAME ONE-CLAUSE RESTATEMENT :func:`beta_run_constitutive_coverage`
    IS ON THE REAL ARM, and the reason it costs no kernel is MEASURED rather than
    argued. ``update_H``/``update_E`` read nothing beta-dependent —
    ``_apply_constitutive_pml`` (stepping.py:2112) never sees ``grid.beta``, which
    enters only the two curls (stepping.py:384-391, :467-474). Measured on this
    host, complex storage, the corpus row's own beta and in-plane k: the two
    sub-steps moved 7,319 float32 words and NOT ONE differed between
    ``beta=-0.3907`` and ``beta=0.0``. The certified complex constitutive kernel
    then reproduced ``stepping.py`` on that configuration over four cycles of both
    sides with zero differing words, with the predicate bypassed — so the ARITHMETIC
    was never the open question, only the ADMISSION.

    So this is ``complex_fields``' clause set with the beta clause INVERTED (clause
    12 of :func:`_beta_complex_grid_reasons`: beta must be NONZERO) and nothing else
    changed. The plan builder delegates to ``complex_fields``' own body, plan class
    and compiled functions, which is what keeps "the same kernel" a fact rather than
    a claim. The E side additionally refuses an off-diagonal ``chi1inv`` row for the
    reason the certified predicate does — the row product reads neighbours and this
    sub-step is element-wise — and the curl still admits it, since its whole effect
    is inside ``update_E``.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _beta_complex_grid_reasons(fields, pml, grid, probe)
    reasons.extend(_residency_declaration_reasons(residency))

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("an off-diagonal chi1inv row is installed (the row product "
                       "reads neighbours; this sub-step is element-wise). The "
                       "complex beta CURL admits the same configuration, because "
                       "an off-diagonal epsilon is constitutive-only")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        # inv_eps stays float32 under complex storage (stepping.py:41-50,
        # fields.py:1203-1204), so the shipped float32 pin is exactly right.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class BetaPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free REAL beta PML curl sub-step.

    ``launch.PmlCurlPlan``'s binding layout plus the two host-rounded beta
    scalars. ``run`` and the contraction-variant contract are the base's.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "beta_plus", "beta_minus", "has_beta", "residency", "volumes")

    family = "special_kz real beta curl"

    REPR_FIELDS = ("sub_step", "shape", "bc", "beta_plus", "beta_minus",
                   "has_beta")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, residency,
                 beta_plus: float, beta_minus: float, targets, auxiliaries,
                 sources, coefficients, functions: Dict[str, Any],
                 volumes: Sequence[str], has_beta: bool = True) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                             f"got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        # Already f32-rounded by the host (beta_curl_coefficients); float() keeps
        # the bits and the scalar ABI delivers them.
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.has_beta = bool(has_beta)
        self.residency = residency
        self.volumes = tuple(volumes)
        # Built ONCE, in the shader's exact binding order: 9 volumes, 6 coefficient
        # vectors, 4 extents, dtdx and the two host-rounded beta scalars = 22
        # arguments. The launch path unpacks this tuple and does nothing else.
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem,
               self.dtdx, self.beta_plus, self.beta_minus))


class BetaBlochPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free COMPLEX beta PML curl sub-step.

    The complex volumes are bound as ``complex64`` MIRRORS and reach the kernel as
    ``device float2*``; ``n_elem`` is the COMPLEX cell count, which is also the
    dispatch size the first tensor argument gives.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "phased", "phase_words", "beta_words", "has_beta", "expansion",
                 "residency", "volumes")

    family = "special_kz complex beta curl"

    REPR_FIELDS = ("sub_step", "shape", "bc", "phased", "expansion", "has_beta")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, phased,
                 phase_words, beta_words, expansion: str, residency, targets,
                 auxiliaries, sources, coefficients, functions: Dict[str, Any],
                 volumes: Sequence[str], has_beta: bool = True) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                             f"got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_words = tuple(float(word) for word in phase_words)
        self.beta_words = tuple(tuple(float(word) for word in pair)
                                for pair in beta_words)
        self.has_beta = bool(has_beta)
        self.expansion = str(expansion)
        self.residency = residency
        self.volumes = tuple(volumes)
        (bpr, bpi), (bmr, bmi) = self.beta_words
        # 9 float2 volumes, 6 coefficient vectors, 4 extents, dtdx, 6 phase words
        # and 4 beta words = 30 bindings, ONE below Metal's 31-buffer ceiling
        # (device.MAX_BUFFER_BINDINGS). That ceiling is why the complex volumes
        # must be float2: a re/im-split build needs nine more and cannot compile.
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem,
               self.dtdx) + self.phase_words + (bpr, bpi, bmr, bmi))


def _beta_curl_functions(codes, backward: bool, has_beta: bool,
                         contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_source(
        beta_curl_source(codes, backward, has_beta, mode)).beta_pml_curl_step
        for mode in contract_variants}


def _beta_bloch_functions(codes, backward: bool, phased, expansion: str,
                          has_beta: bool,
                          contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_source(
        beta_bloch_curl_source(codes, backward, phased, expansion, has_beta, mode)
    ).beta_bloch_pml_curl_step for mode in contract_variants}


def plan_beta_pml_curl(fields: Any, pml: Any, sub_step: str, residency: Any = None,
                       contract_variants: Sequence[str] = (templates.CONTRACT_OFF,),
                       ) -> Optional[BetaPmlCurlPlan]:
    """Build a REAL beta curl plan from the engine's objects, or None when refused."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not beta_pml_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    codes = [1 if kind == "metallic" else 0 for kind in resolve(grid, pml)]
    plus, minus = beta_curl_coefficients(grid.beta, grid.dt,
                                         magnetic=(sub_step == "step_B"),
                                         complex_storage=False)
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return BetaPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, residency, plus, minus,
        targets, auxiliaries, sources, coefficients,
        _beta_curl_functions(codes, spec["backward"], True, contract_variants),
        volumes)


def plan_beta_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                   flat: Dict[str, Any], codes, dtdx: float,
                                   beta_plus: float, beta_minus: float,
                                   residency: Any,
                                   functions: Optional[Dict[str, Any]] = None,
                                   contract_variants: Sequence[str] = (
                                       templates.CONTRACT_OFF,),
                                   has_beta: bool = True) -> BetaPmlCurlPlan:
    """Build a REAL beta curl plan from bare host arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones. ``functions`` is the
    MUTATION SEAM — dropping it is not a slowdown, it is a silent DISARMING, and
    every mutation leg would then launch the shipped kernel and report the defect
    as uncaught. ``has_beta=False`` is the identity leg's arm.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    targets = [residency.mirror(n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n]) for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return BetaPmlCurlPlan(
        sub_step, shape, dtdx, codes, residency, beta_plus, beta_minus,
        targets, auxiliaries, sources, coefficients,
        functions if functions is not None
        else _beta_curl_functions(codes, spec["backward"], has_beta,
                                  contract_variants),
        volumes, has_beta=has_beta)


def bloch_phase_words(grid: Any, kinds: Sequence[str], backward: bool
                      ) -> Tuple[Tuple[int, int, int], Tuple[float, ...]]:
    """The per-axis (phased flag, complex64-rounded phase words) for one sub-step.

    The D sub-step differences DOWNWARD, so its wrapped lane carries the CONJUGATE
    phase — the array path gets that from ``_bloch_phases`` per direction, and it
    is applied here per sub-step for the same reason.
    """
    import numpy as np  # noqa: PLC0415

    reader = getattr(grid, "bloch_phase", None)
    phased: List[int] = []
    words: List[float] = []
    for axis in range(3):
        phase = None
        if callable(reader) and kinds[axis] == "periodic":
            phase = reader(axis)
        if phase is None:
            phased.append(0)
            words.extend((1.0, 0.0))
            continue
        value = np.complex64(np.conj(phase) if backward else phase)
        phased.append(1)
        words.extend((float(np.float32(value.real)), float(np.float32(value.imag))))
    return tuple(phased), tuple(words)


def plan_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                             residency: Any = None,
                             contract_variants: Sequence[str] = (
                                 templates.CONTRACT_OFF,),
                             probe: Any = None
                             ) -> Optional[BetaBlochPmlCurlPlan]:
    """Build a COMPLEX beta curl plan from the engine's objects, or None."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not beta_bloch_pml_curl_coverage(fields, pml, sub_step, residency,
                                        probe).covered:
        return None
    record = probe if probe is not None else load_expansion_probe()
    expansion = beta_expansion_from_probe(record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    codes = [1 if kind == "metallic" else 0 for kind in kinds]
    phased, phase_words = bloch_phase_words(grid, kinds, bool(spec["backward"]))
    beta_words = beta_curl_coefficients(grid.beta, grid.dt,
                                        magnetic=(sub_step == "step_B"),
                                        complex_storage=True)
    # THE COMPLEX VOLUMES ARE MIRRORED AS complex64 AND THE DTYPE IS PASSED
    # EXPLICITLY. `Residency.mirror` defaults to float32 -- the real families'
    # storage -- and the default would CAST, discarding the imaginary half behind a
    # ComplexWarning and halving the dispatch grid. Measured before it was fixed:
    # 3840/3840 differing words on every complex case, a total divergence that
    # looks like a wrong kernel and is a wrong BINDING.
    complex64 = _numpy().complex64
    targets = [residency.mirror(n, getattr(fields, n), dtype=complex64)
               for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, getattr(fields, "fu_" + n),
                                    dtype=complex64) for n in spec["targets"]]
    sources = [residency.mirror(n, getattr(fields, n), dtype=complex64)
               for n in spec["sources"]]
    # The PML coefficient vectors stay REAL on both sides: the array path's
    # `fu *= kms` is a complex-by-real multiply (stepping.py:1976-1982), which the
    # kernel spells through `c_mul_field_left`.
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return BetaBlochPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, phased, phase_words,
        beta_words, expansion, residency, targets, auxiliaries, sources,
        coefficients,
        _beta_bloch_functions(codes, spec["backward"], phased, expansion, True,
                              contract_variants),
        volumes)


def plan_beta_bloch_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                         flat: Dict[str, Any], codes, phased,
                                         phase_words, dtdx: float,
                                         beta_words, expansion: str,
                                         residency: Any,
                                         functions: Optional[Dict[str, Any]] = None,
                                         contract_variants: Sequence[str] = (
                                             templates.CONTRACT_OFF,),
                                         has_beta: bool = True
                                         ) -> BetaBlochPmlCurlPlan:
    """Build a COMPLEX beta curl plan from bare host arrays — the gate's route.

    ``beta_words`` is ``((bpr, bpi), (bmr, bmi))`` — normally
    :func:`beta_curl_coefficients`'s output, or a deliberately wrong pair on a
    mutation leg. The step_D phase conjugation is the CALLER's here, exactly as
    the sub-step split makes it the plan builder's on the engine route.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    complex64 = _numpy().complex64
    targets = [residency.mirror(n, arrays[n], dtype=complex64)
               for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n], dtype=complex64)
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n], dtype=complex64)
               for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return BetaBlochPmlCurlPlan(
        sub_step, shape, dtdx, codes, phased, phase_words, beta_words, expansion,
        residency, targets, auxiliaries, sources, coefficients,
        functions if functions is not None
        else _beta_bloch_functions(codes, spec["backward"], phased, expansion,
                                   has_beta, contract_variants),
        volumes, has_beta=has_beta)


def plan_beta_run_constitutive(fields: Any, pml: Any, side: str,
                               residency: Any = None,
                               contract_variants: Sequence[str] = (
                                   templates.CONTRACT_OFF,)) -> Any:
    """A CERTIFIED constitutive plan for a REAL beta run, or None.

    The arithmetic, the source and the plan class are the certified ones; only the
    ADMISSION is this module's. Nothing about the constitutive sub-step is
    beta-dependent, and building the plan through the certified builder is what
    makes that a fact rather than a claim.
    """
    from .launch import (  # noqa: PLC0415 - avoids a circular import at module load
        ConstitutivePlan, _constitutive_functions,
    )

    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not beta_run_constitutive_coverage(fields, pml, side, residency).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror(n, getattr(fields, n)) for n in spec["aux"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True) for n in spec["targets"]]
        if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ConstitutivePlan(
        side, fields.grid.shape, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        _constitutive_functions(side, contract_variants), volumes)


def plan_beta_run_complex_constitutive(
        fields: Any, pml: Any, side: str, residency: Any = None,
        contract_variants: Sequence[str] = (templates.CONTRACT_OFF,),
        probe: Any = None) -> Any:
    """A CERTIFIED COMPLEX constitutive plan for an UNFOLDED beta run, or None.

    ``complex_fields``' own body, its own plan class and its own compiled functions;
    only the ADMISSION is this module's, exactly as
    :func:`plan_beta_run_constitutive` borrows the real one. Building the plan
    THROUGH the certified builder is what makes "the same kernel steps a beta run"
    a fact rather than a claim: nothing here re-spells any arithmetic.

    The EXPANSION arm comes from this module's own beta probe rather than
    ``complex_fields``' — the beta artifact classifies one more call-site
    orientation, so it licenses everything the complex artifact does and one pattern
    more. :mod:`.folded_beta` binds it the same way for the same reason.
    """
    from . import complex_fields  # noqa: PLC0415 - circular import at module load

    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not beta_run_complex_constitutive_coverage(
            fields, pml, side, residency, probe).covered:
        return None
    expansion = beta_expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    mirror = complex_fields._complex_mirror
    targets = [mirror(residency, n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [mirror(residency, n, getattr(fields, n)) for n in spec["aux"]]
    sources = [mirror(residency, n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True) for n in spec["targets"]]
        if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return complex_fields.ComplexConstitutivePlan(
        side, fields.grid.shape, expansion, residency, targets, auxiliaries,
        sources, inverse_epsilon, coefficients,
        complex_fields._constitutive_functions(side, expansion, contract_variants),
        volumes)


# ---------------------------------------------------------------------------
# WIRING — six registered arms, as of tranche 2
# ---------------------------------------------------------------------------
#
# THIS MODULE USED TO REGISTER NOTHING, exposing `ARMS` as a plain tuple instead so
# a probe that imported it BY NAME could sweep the rows without `plan_step` being
# able to select one. That cost was stated at the time and it was real:
# `arms.registered()` could not see this family at all, so a stranger's sweep —
# including the disjointness sweep this tranche runs — would have missed exactly the
# pair most likely to collide, this module's complex curl against
# `complex_fields`' complex curl. The rows are registered now, and the sweep that
# checks them for co-admission reaches them through the table like every other row.
#
# WHAT SEPARATES THEM, by name and in both directions:
#
#   * from the shipped real kernels — `coverage._grid_reasons` clause 12 refuses
#     `beta != 0`, and clause 12 here is INVERTED (beta must be nonzero);
#   * real from complex — clause 2 here requires REAL storage and
#     `_beta_bloch_grid_reasons` requires complex, the same inversion the shipped
#     and complex families use between themselves;
#   * complex beta from `complex_fields` — that module's clause 8 refuses
#     `grid.beta` BY NAME, which is what keeps refl-angular-kz2d.py and
#     parallel-wvgs-force.py here rather than silently there.
#
# THE ASYMMETRY IS GONE, AND IT WAS CLOSED THE WAY IT WAS NAMED. Both constitutive
# companions are now the CERTIFIED kernel under a restated predicate — the real arm
# borrows `launch.ConstitutivePlan` (`beta_run_constitutive_coverage` /
# `plan_beta_run_constitutive`), the complex arm borrows
# `complex_fields.ComplexConstitutivePlan`
# (`beta_run_complex_constitutive_coverage` / `plan_beta_run_complex_constitutive`)
# — so a beta run of EITHER storage takes all four sub-steps on the device.
#
# UNTIL 2026-08-19 the complex arm's constitutive pair was an unconditional refusal,
# registered anyway so `plan_step` could say WHY `update_H` was on the array path.
# What closed it is one clause, not a kernel: the constitutive sub-steps read
# nothing beta-dependent (`_apply_constitutive_pml`, S:2065 — beta enters only the
# curls at S:356-363 and :438-445), MEASURED on this host as 7,319 words moved by
# the pair and ZERO differing between the corpus row's beta and beta = 0, with the
# certified complex kernel then reproducing `stepping.py` over four cycles of both
# sides with the predicate bypassed. The last unadmitted slot pair in the Metal
# census (`TestSpecialKz.test_special_kz`, update_H and update_E) is this one.

#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}


def _real_curl_coverage(ctx: Any, slot: str) -> Coverage:
    return beta_pml_curl_coverage(ctx.fields, ctx.pml, slot, ctx.residency)


def _real_curl_plan(ctx: Any, slot: str) -> Any:
    return plan_beta_pml_curl(ctx.fields, ctx.pml, slot, ctx.residency,
                              ctx.contract_variants)


def _complex_curl_coverage(ctx: Any, slot: str) -> Coverage:
    return beta_bloch_pml_curl_coverage(ctx.fields, ctx.pml, slot, ctx.residency,
                                        ctx.extra.get("beta_probe"))


def _complex_curl_plan(ctx: Any, slot: str) -> Any:
    return plan_beta_bloch_pml_curl(ctx.fields, ctx.pml, slot, ctx.residency,
                                    ctx.contract_variants,
                                    ctx.extra.get("beta_probe"))


def _real_constitutive_coverage(ctx: Any, slot: str) -> Coverage:
    return beta_run_constitutive_coverage(ctx.fields, ctx.pml,
                                          _CONSTITUTIVE_SLOT_SIDES[slot],
                                          ctx.residency)


def _real_constitutive_plan(ctx: Any, slot: str) -> Any:
    return plan_beta_run_constitutive(ctx.fields, ctx.pml,
                                      _CONSTITUTIVE_SLOT_SIDES[slot],
                                      ctx.residency, ctx.contract_variants)


def _complex_constitutive_coverage(ctx: Any, slot: str) -> Coverage:
    return beta_run_complex_constitutive_coverage(
        ctx.fields, ctx.pml, _CONSTITUTIVE_SLOT_SIDES[slot], ctx.residency,
        ctx.extra.get("beta_probe"))


def _complex_constitutive_plan(ctx: Any, slot: str) -> Any:
    return plan_beta_run_complex_constitutive(
        ctx.fields, ctx.pml, _CONSTITUTIVE_SLOT_SIDES[slot], ctx.residency,
        ctx.contract_variants, ctx.extra.get("beta_probe"))


def register_arms() -> Tuple[arms.ArmSpec, ...]:
    """This family's six arms: two curl pairs and two constitutive pairs."""
    registered = [
        arms.register(family="special_kz_real", slot=slot,
                      label="special_kz real beta",
                      coverage=_real_curl_coverage, plan=_real_curl_plan,
                      prefix="special_kz real: ",
                      noun="special_kz real beta curl", wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family="special_kz_complex", slot=slot,
                      label="special_kz complex beta",
                      coverage=_complex_curl_coverage, plan=_complex_curl_plan,
                      prefix="special_kz complex: ",
                      noun="special_kz complex beta curl", wired=True)
        for slot in ("step_B", "step_D"))
    registered.extend(
        arms.register(family="special_kz_real", slot=slot,
                      label="special_kz real beta",
                      coverage=_real_constitutive_coverage,
                      plan=_real_constitutive_plan,
                      prefix="special_kz real: ",
                      noun="special_kz real beta constitutive", wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    registered.extend(
        arms.register(family="special_kz_complex", slot=slot,
                      label="special_kz complex beta",
                      coverage=_complex_constitutive_coverage,
                      plan=_complex_constitutive_plan,
                      prefix="special_kz complex: ",
                      noun="special_kz complex beta constitutive", wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    return tuple(registered)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[arms.ArmSpec, ...] = register_arms()
