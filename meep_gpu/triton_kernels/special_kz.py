"""Special-kz (``grid.beta``) PML curl tranche — Phase A: plain real + plain complex.

NOT WIRED. Production dispatch is untouched: ``fastpath.plan_fast_path`` still
returns ``None`` on every branch and this module does not touch it, ``plan_step``
composition is DEFERRED (``coverage._grid_reasons`` clause 12 and
``complex_fields._complex_grid_reasons`` clause 8 already refuse every beta run,
so these predicates cannot create an admitted-overlap ambiguity until a later
coordinated change adds launch-side forwarders), and ``fingerprints.json``
carries no entry for this file — the byte gate binds its own provenance record
inside its results directory instead. Callers are the gate
(``parity/meep_gpu/gate_triton_special_kz.py``), the composition probe
(``probe_triton_special_kz_composition.py``) and the laptop tests, nothing else.

WHAT THIS IS FOR. Six lifted corpus rows demand it and only six: the two
refl-angular-kz2d.py legs (beta = 0.332, real and complex storage),
test_special_kz.test_special_kz (in-plane kx = 0.9205 + beta = -0.3907, complex
storage), the two eigsrc_kz cases (beta = 0.2, Mirror — complex and folded-real)
and the two binary_grating special_kz cases (beta = -0.685 / -0.912, Mirror +
in-plane Bloch). Phase A (this file) carries the PLAIN real and PLAIN complex
beta curls, which fully unlocks refl-angular-kz2d.py and test_special_kz; the
folded rows stay refused by name — the folded REAL beta curl is Phase B
(restating ``symmetry.py``'s folded clauses; unlocks test_eigsrc_kz_1_real_imag)
and the folded COMPLEX rows are the complex tranche's own fold-under-complex-
storage composition gap, not a beta gap. ``parallel-wvgs-force.py`` (beta = 0.5)
is lifter-refused (no sources at capture) and is named here, not counted.

WHAT BETA IS (grid.py:668-697, stepping.py:754-811). A 2-D run carries
``exp(i*2*pi*beta*z)`` analytically, so ``d/dz`` on the invariant axis is the
EXACT factor ``i*2*pi*beta`` — an ``i*beta*zhat x`` cross product folded into
the curl (MEEP step_db.cpp:148-176). The family adds ONE term to the two curl
sub-steps and changes nothing else in the step loop:

* call sites: step_B (stepping.py:384-391) — Bx adds the Ey-center term at sign
  +1 (:389), By the Ex-center term at sign -1 (:391), Bz nothing; step_D
  (:438-445) — Dx <- Hy at +1 (:443), Dy <- Hx at -1 (:445), Dz nothing;
* coefficient (:770-772): ``sign * 2*pi * grid.beta * grid.dt`` — NO dtdx, this
  is an analytic derivative, not a finite difference (:733-735); complex storage
  multiplies by ``+1j`` (B side) / ``-1j`` (D side); real storage is the same
  arithmetic under MEEP's implicit-i trick (step_db.cpp:148-160), a plain real
  add on both sides, not a second code path;
* rounding (:784): the coefficient is rounded ONCE on the host via
  ``partner_values.dtype.type(coefficient)``, then ``return -(c * partner)`` —
  the negation is of the PRODUCT (curl sign convention; the caller subtracts);
* position: added to ``curl`` AFTER ``_curl_from_operands`` (:342/:429), BEFORE
  ``_mask_non_owned_cells`` (:369/:450) and ``_apply_curl`` (:374/:455), so the
  increment rides the SAME split-field kms/sinv ladder and the same condinv
  branch as the finite-difference curl, exactly and linearly (:762-767);
* partners: the SAME-CELL component snapshots (:293 electric, :408 magnetic).
  Per ``B_CURL_TERMS``/``D_CURL_TERMS`` (:213-223) every beta partner is a
  CENTER operand the certified curl kernels already load — Bx: the second
  source Ey center; By: the first source Ex center; Dx: second Hy; Dy: first Hx
  — so the term costs no new pointer and no extra memory traffic, and the load
  reused must be the UNSHIFTED center one, never an ox/oy/oz shifted operand.
  In the complex family the Bloch wrap rotation applies only to shifted
  operands, so reusing the center registers keeps the beta partner UNROTATED by
  construction (a gate mutation plants the rotation and must be caught);
* no new state array (contrast BFAST's per-component IIR state, :889-893), no
  ghost rule, no fold parity, no wrap; ``update_H``/``update_E``
  (``_apply_constitutive_pml``, :2065-2096) read nothing beta-dependent.

TWO KERNELS, one per storage family:

* :func:`beta_pml_curl_step` — ``kernels.pml_curl_step``'s body (certified) plus
  the constexpr-gated beta term: one f32 multiply-and-subtract per affected
  target, reusing the loaded center operand;
* :func:`beta_bloch_pml_curl_step` — ``complex_fields.bloch_pml_curl_step``'s
  body (certified) plus the ±i coefficient product on word pairs,
  through the module-local :func:`_mul_imag_coefficient_left` helper.

GROUPING CHOICES the gate must hold (stepping.py forces none of these):

1. TWO host-rounded scalars per launch — one per sign — rather than one scalar
   negated in-kernel. The array path computes ``sign * 2*pi*beta*dt`` in f64 per
   call site and rounds each once (:770/:784); f64 negation and f32 rounding
   commute exactly, but binding both keeps the transcription literal per call
   site rather than resting on that identity.
2. ``curl - (c * g)`` carries ``curl + (-(c * g))``. The array path negates the
   product and adds; IEEE-754 defines subtraction AS addition of the negation,
   so the single subtract is the same bits on every input including signed
   zeros — and it never leans on how Triton lowers unary minus.
3. The complex product is the FULL multiply with the coefficient on the LEFT
   (:784 ``c * partner_values``): ``re = c_re*z_re - c_im*z_im``,
   ``im = c_re*z_im + c_im*z_re`` with ``c_re`` a SIGNED zero produced by
   Python's complex multiply and passed through as an argument, never
   synthesized as ``+0.0``. Its ``EXPANSION`` arm binds to a measured probe
   artifact carrying the NEW pattern :data:`BETA_PROBE_PATTERN` — the base
   four patterns of the complex tranche do not cover a general-coefficient
   scalar-left product, so :func:`beta_expansion_license` requires the
   EXTENDED set (:data:`BETA_PROBE_PATTERNS`) present, and the patterns that
   CAN discriminate must agree; a missing artifact, or one no arm reproduces,
   is a refusal by name. The licence is
   :func:`complex_fields.expansion_license` over that superset — the same
   arbiter the base family uses, called with a longer list, so the
   non-discriminating clause reaches every pattern here rather than only the
   one this tranche adds.
4. ``HAS_BETA`` is a constexpr: the term is compiled only when beta != 0. The
   engine-route plans always bind 1 (the predicate requires beta nonzero; a
   beta = 0 run belongs to the certified plain kernels, and the array path
   never enters the term at :356/:438); the 0 arm exists so the gate's identity
   leg can pin the HAS_BETA=0 build byte-identical to the certified kernel.
5. Constitutive sub-steps on beta runs are NOT a new kernel: the restated
   predicates (:func:`beta_run_constitutive_coverage`,
   :func:`beta_run_complex_constitutive_coverage`) drop only the beta clause
   and delegate arithmetic to the certified ``kernels.constitutive_step`` /
   ``complex_fields.bloch_constitutive_step`` through the existing plan
   classes. The complex constitutive binds the BASE probe patterns (the kernel
   launched is the certified one, whose contract is the base set); only the
   beta CURL binds the extended set.
6. Phase A refuses the fold in BOTH storage families, by name. Phase B lifts
   the real-family fold through the folded family's own predicate; the complex
   fold is inherited from the complex tranche's composition gap and stays in
   that queue.
7. A conductivity on the curl targets is refused by name in BOTH directions:
   these kernels transcribe the plain split-field recurrence only, and
   ``conductivity.py``'s own predicates refuse beta (conductivity.py:519-520)
   — no silent overlap. The beta-under-condinv linearity claim (:762-767) is a
   reference-leg fact in the gate, not a kernel here.
8. Fused pairs are untouched: ``fused_curl_constitutive_*`` keeps refusing beta
   through its own predicate's grid clauses — beta lands in the UNFUSED curls
   only, and the source-seam rule is unchanged.

Import contract: this module is importable WITHOUT Triton — the predicates and
plan builders (to ``None``) must answer on the laptop that is the merge bar.
Triton is imported at module scope inside a guard, the kernels degrade to
:class:`complex_fields._UnavailableKernel`'s pattern, and ``ENABLE_FP_FUSION``
is imported from :mod:`kernels` only inside ``run()`` so the guard keeps its
single spelling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


class _UnavailableKernel:
    """Fail at the launch expression while leaving host predicates importable."""

    __slots__ = ("name", "error")

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self.error = error

    def __getitem__(self, grid):
        raise ImportError(
            f"the optional triton package is required to launch {self.name}; "
            f"host coverage remains available without it: {self.error}") from self.error


try:  # The host predicate and public refusal path must work without Triton.
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - exercised by the absence test
    _TRITON_IMPORT_ERROR = _exc

    class _MissingTriton:
        @staticmethod
        def jit(function):
            return _UnavailableKernel(function.__name__, _TRITON_IMPORT_ERROR)

    class _MissingLanguage:
        @staticmethod
        def constexpr(value):
            return value

    triton = _MissingTriton()  # type: ignore[assignment]
    tl = _MissingLanguage()  # type: ignore[assignment]

from .complex_fields import (  # noqa: E402 - this package's own module, restate-don't-edit base
    PROBE_PATTERNS,
    ComplexConstitutivePlan,
    _complex_layout_reasons,
    _expansion_reasons,
    _mul_coefficient_left,
    _mul_field_left,
    _phase_arguments,
    _rotate_field_left,
    _word_view,
    bloch_phase_table,
    expansion_from_probe,
    expansion_license,
    load_expansion_probe,
)
from .coverage import (  # READ-ONLY imports; nothing here mutates coverage.py
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
from .launch import SUB_STEPS, ConstitutivePlan, CupyPointer, _flat

# Same constexpr codes as ``kernels.PERIODIC``/``kernels.METALLIC`` and the
# complex tranche's expansion arms. Restated (not imported from kernels.py)
# because importing kernels.py would import Triton unconditionally and defeat
# this module's host-only coverage route; the laptop tests pin all four against
# their originators so the restatements cannot drift.
PERIODIC = tl.constexpr(0)
METALLIC = tl.constexpr(1)
NAIVE = tl.constexpr(0)
FMA_V1 = tl.constexpr(1)

#: Elements per program — real cells for the real kernel, complex CELLS for the
#: complex one. Restated from ``kernels.DEFAULT_BLOCK`` for the same import
#: reason; no autotune (the kernels write their own inputs in place,
#: kernels.py:56-59).
DEFAULT_BLOCK = 256

#: The NEW probe pattern this tranche adds: the beta coefficient product,
#: ``complex64 scalar (signed-zero real, +-imag) * complex64 array`` with the
#: COEFFICIENT on the left (stepping.py:811). The base four patterns cover only
#: zero-imaginary and field-left/general-c8 orientations; a general scalar-left
#: coefficient is a distinct compiled-dispatch case and is measured, not
#: assumed.
BETA_PROBE_PATTERN = "c8_mul_c8_imaginary_coefficient_left"

#: The pattern set a probe artifact must classify — and classify identically —
#: before the complex beta CURL may bind an ``EXPANSION`` constexpr.
BETA_PROBE_PATTERNS: Tuple[str, ...] = PROBE_PATTERNS + (BETA_PROBE_PATTERN,)

__all__ = [
    "BETA_PROBE_PATTERN",
    "BETA_PROBE_PATTERNS",
    "BetaBlochPmlCurlPlan",
    "BetaPmlCurlPlan",
    "beta_bloch_pml_curl_coverage",
    "beta_bloch_pml_curl_step",
    "beta_curl_coefficients",
    "beta_expansion_from_probe",
    "beta_expansion_license",
    "beta_pml_curl_coverage",
    "beta_pml_curl_step",
    "beta_run_complex_constitutive_coverage",
    "beta_run_constitutive_coverage",
    "plan_beta_bloch_pml_curl",
    "plan_beta_bloch_pml_curl_from_arrays",
    "plan_beta_pml_curl",
    "plan_beta_pml_curl_from_arrays",
    "plan_beta_run_complex_constitutive",
    "plan_beta_run_constitutive",
]


# ---------------------------------------------------------------------------
# The host coefficient — stepping.py:797-811, transcribed
# ---------------------------------------------------------------------------

def beta_curl_coefficients(beta: float, dt: float, magnetic: bool,
                           complex_storage: bool):
    """The (plus-sign, minus-sign) beta coefficients, host-computed and rounded
    EXACTLY as ``stepping._special_kz_beta_term`` computes and rounds them.

    Per sign: ``coefficient = sign * 2*pi * beta * dt`` in f64 (:770); complex
    storage multiplies by ``+1j`` (magnetic) / ``-1j`` (electric) THROUGH
    Python's own complex arithmetic (:771-772) — which is what puts the SIGNED
    zero in the real word, and why the real word is passed through rather than
    synthesized — then one rounding to the storage dtype (:784).

    Returns two f32 floats (real storage) or two ``(re, im)`` float pairs
    (complex storage). ``float()`` of a numpy.float32 preserves the zero's
    sign, and Triton types a Python float argument as fp32, so the words the
    kernel receives are the array path's bits.
    """
    import numpy  # noqa: PLC0415

    out: List[Any] = []
    for sign in (1.0, -1.0):
        coefficient: Any = sign * 2.0 * math.pi * float(beta) * float(dt)  # :770
        if complex_storage:
            coefficient = coefficient * (1j if magnetic else -1j)  # :771-772
            rounded = numpy.complex64(coefficient)  # :784 — dtype.type, once
            out.append((float(numpy.float32(rounded.real)),
                        float(numpy.float32(rounded.imag))))
        else:
            out.append(float(numpy.float32(coefficient)))  # :784 — once
    return tuple(out)


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------

@triton.jit
def _mul_imag_coefficient_left(c_re, c_im, z_re, z_im, EXPANSION: tl.constexpr):
    """(c_re + i*c_im) * (z) with the COEFFICIENT on the left — the S:784
    orientation ``partner.dtype.type(coefficient) * partner_values`` for the
    beta coefficient, whose real word is a SIGNED zero from Python's complex
    multiply (S:771-772) and whose imaginary word carries ±2*pi*beta*dt.

    The full multiply: ``re = c_re*z_re - c_im*z_im``, ``im = c_re*z_im +
    c_im*z_re``. FMA_V1 fuses the FIRST operand's product exactly as
    :func:`complex_fields._rotate_field_left` does for the field-left phase
    rotation — here the first operand is the coefficient, which is why this is
    a NEW probe pattern (:data:`BETA_PROBE_PATTERN`) and not a reuse of the
    field-left classification.

    NEGATION IS ``* -1.0``, NEVER unary ``-``: Triton lowers ``-x`` as
    ``0.0 - x`` (triton 3.1.0, language/semantic.py:386-391), and under
    round-to-nearest ``0.0 - (+0.0)`` is ``+0.0`` — the addend's zero SIGN is
    lost — while the licensed FMA_V1 transcription negates the cross product
    sign-exactly (S:784 is a NumPy/CuPy negation, a sign flip). ``* -1.0`` is
    the IEEE-exact negation (LLVM folds it to fneg). Measured: the gate's m9
    leg found the unary-minus spelling of this addend byte-identical to the
    FOLDED mutant on the engineered signed-zero state — the platform
    lowering, not the fold, was deciding the bytes.

    The ``c_re`` products are the signed-zero cross terms; folding them away
    is the gate's m9 mutation — caught at the PRODUCT layer (a wrapper kernel
    stores this helper's output words directly). Until 2026-08-12 the inlined
    complex_fields helpers' addends were spelled with unary ``-`` and
    canonicalized every ±0 to ``+0`` under the same lowering before any store
    (measured, the m9 leg's predicted-4-got-0), which made the fold byte-blind
    at the STORED layer. The complex tranche now carries the ``* -1.0``
    spelling too (fixed after the zero-init composition reachability analysis,
    probe_triton_complex_composition.py), so the fold's flip once again
    propagates sign-exactly downstream and the stored layer reverts to the
    original IEEE-negation prediction — the gate's stored leg pins that count
    and the device recut arbitrates.
    """
    if EXPANSION == FMA_V1:
        out_re = tl.math.fma(c_re, z_re, (c_im * z_im) * -1.0)
        out_im = tl.math.fma(c_re, z_im, c_im * z_re)
    else:
        out_re = (c_re * z_re) - (c_im * z_im)
        out_im = (c_re * z_im) + (c_im * z_re)
    return out_re, out_im


@triton.jit
def beta_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
    u0, u1, u2,                       # auxiliaries: fu_B*  or  fu_D*
    g0, g1, g2,                       # sources: Ex,Ey,Ez  or  Hx,Hy,Hz
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, one Yee sub-lattice
    nx, ny, nz, n_elem, dtdx,
    beta_plus, beta_minus,            # f32(sign*2*pi*beta*dt), host-rounded once (S:770/:784)
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    HAS_BETA: tl.constexpr,           # compiled-in only for beta != 0 runs
    BLOCK: tl.constexpr,
):
    """One REAL curl sub-step of all three components, PML recurrence included,
    with the special_kz beta term — ``kernels.pml_curl_step``'s certified body
    plus the constexpr-gated insert between the curl and the ownership mask,
    which is exactly where the array path adds it (S:356-363 after :342, before
    :369; S:438-445 after :429, before :450).

    The beta partners are the CENTER loads the curl already made: target 0
    takes the second source's center ``b`` at sign +1 (Bx <- Ey / Dx <- Hy)
    and target 1 the first source's center ``a`` at sign -1 (By <- Ex /
    Dy <- Hx); target 2 gets nothing (step_db.cpp:148-176 runs ``cc`` over d_c
    in {X, Y} only). ``curl - (c * g)`` is ``curl + (-(c * g))`` by IEEE-754's
    definition of subtraction — the array path's negate-then-add, without
    relying on unary-minus lowering. Real storage is MEEP's implicit-i trick:
    the SAME sign for both sub-steps (the ±i lives only in complex storage).
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vx, vy, vz = live, live, live
    if BCX == METALLIC:
        vx = live & (si >= 0) & (si < nx)
    else:
        si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
    if BCY == METALLIC:
        vy = live & (sj >= 0) & (sj < ny)
    else:
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    if BCZ == METALLIC:
        vz = live & (sk >= 0) & (sk < nz)
    else:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))

    ox = si * nyz + j * nz + k
    oy = i * nyz + sj * nz + k
    oz = i * nyz + j * nz + sk

    a = tl.load(g0 + idx, mask=live, other=0.0)
    b = tl.load(g1 + idx, mask=live, other=0.0)
    c = tl.load(g2 + idx, mask=live, other=0.0)
    a_y = tl.load(g0 + oy, mask=vy, other=0.0)
    a_z = tl.load(g0 + oz, mask=vz, other=0.0)
    b_x = tl.load(g1 + ox, mask=vx, other=0.0)
    b_z = tl.load(g1 + oz, mask=vz, other=0.0)
    c_x = tl.load(g2 + ox, mask=vx, other=0.0)
    c_y = tl.load(g2 + oy, mask=vy, other=0.0)

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
    curl0 = dtdx * ((c_y - c) + (b - b_z))
    curl1 = dtdx * ((a_z - a) + (c - c_x))
    curl2 = dtdx * ((b_x - b) + (a - a_y))

    # --- the beta term (stepping._special_kz_beta_term), CENTER partners only ---
    # AFTER the dtdx curl, BEFORE the ownership mask — the array path's order.
    # No dtdx on the term (analytic derivative, S:733-735); the subtraction IS
    # the array path's `curl + (-(c*g))` (module docstring, grouping choice 2).
    if HAS_BETA:
        curl0 = curl0 - (beta_plus * b)
        curl1 = curl1 - (beta_minus * a)

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    if BACKWARD:
        if BCY == METALLIC:
            curl0 = tl.where(at_y, 0.0, curl0)
        if BCZ == METALLIC:
            curl0 = tl.where(at_z, 0.0, curl0)
        if BCX == METALLIC:
            curl1 = tl.where(at_x, 0.0, curl1)
        if BCZ == METALLIC:
            curl1 = tl.where(at_z, 0.0, curl1)
        if BCX == METALLIC:
            curl2 = tl.where(at_x, 0.0, curl2)
        if BCY == METALLIC:
            curl2 = tl.where(at_y, 0.0, curl2)
    else:
        if BCX == METALLIC:
            curl0 = tl.where(at_x, 0.0, curl0)
        if BCY == METALLIC:
            curl1 = tl.where(at_y, 0.0, curl1)
        if BCZ == METALLIC:
            curl2 = tl.where(at_z, 0.0, curl2)

    # --- split-field recurrence (stepping._apply_pml_update) -------------------
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    p0 = tl.load(u0 + idx, mask=live, other=0.0)
    n0 = ((p0 * km_y) - curl0) * si_y
    v0 = (((tl.load(f0 + idx, mask=live, other=0.0) * km_z) + n0) - p0) * si_z

    p1 = tl.load(u1 + idx, mask=live, other=0.0)
    n1 = ((p1 * km_z) - curl1) * si_z
    v1 = (((tl.load(f1 + idx, mask=live, other=0.0) * km_x) + n1) - p1) * si_x

    p2 = tl.load(u2 + idx, mask=live, other=0.0)
    n2 = ((p2 * km_x) - curl2) * si_x
    v2 = (((tl.load(f2 + idx, mask=live, other=0.0) * km_y) + n2) - p2) * si_y

    tl.store(u0 + idx, n0, mask=live)
    tl.store(u1 + idx, n1, mask=live)
    tl.store(u2 + idx, n2, mask=live)
    tl.store(f0 + idx, v0, mask=live)
    tl.store(f1 + idx, v1, mask=live)
    tl.store(f2 + idx, v2, mask=live)


@triton.jit
def beta_bloch_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz or Dx,Dy,Dz (complex64 as words)
    u0, u1, u2,                       # auxiliaries: fu_B* or fu_D* (complex64 as words)
    g0, g1, g2,                       # sources: Ex,Ey,Ez or Hx,Hy,Hz (complex64 as words)
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, float32, one Yee sub-lattice
    nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    pxr, pxi, pyr, pyi, pzr, pzi,     # per-axis complex64-rounded phase (conj for BACKWARD)
    bp_re, bp_im, bm_re, bm_im,       # complex64-rounded ±sign beta coefficients (S:770-784)
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
    HAS_BETA: tl.constexpr,           # compiled-in only for beta != 0 runs
    EXPANSION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One COMPLEX curl sub-step with the special_kz beta term —
    ``complex_fields.bloch_pml_curl_step``'s certified body plus the
    ±i coefficient product on word pairs, inserted between the curl and the
    ownership mask exactly as the array path inserts it.

    The beta partners are the CENTER word pairs ``(b_re, b_im)`` (target 0,
    sign +1) and ``(a_re, a_im)`` (target 1, sign -1), loaded before the phase
    section and never touched by it — the Bloch wrap rotates SHIFTED operands
    only, so the partner is unrotated by construction (the gate's m5 mutation
    plants the rotation and must be caught). The ±i lives in the host-bound
    coefficient words: ``+1j`` for the B side, ``-1j`` for D (S:771-772),
    rounded once through numpy.complex64 with the signed-zero real word passed
    through. Plane-wise subtraction of the product IS complex ``curl +
    (-(c*g))``: complex negation negates both words and complex add is
    component-wise.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vx, vy, vz = live, live, live
    if BCX == METALLIC:
        vx = live & (si >= 0) & (si < nx)
    else:
        si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
    if BCY == METALLIC:
        vy = live & (sj >= 0) & (sj < ny)
    else:
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    if BCZ == METALLIC:
        vz = live & (sk >= 0) & (sk < nz)
    else:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))

    ox = si * nyz + j * nz + k
    oy = i * nyz + sj * nz + k
    oz = i * nyz + j * nz + sk

    # --- wrapped-lane predicates, one plane per axis ----------------------------
    if BACKWARD:
        wx, wy, wz = i == 0, j == 0, k == 0
    else:
        wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1

    # --- loads: two words per operand ------------------------------------------
    a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
    a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)
    a_z_re = tl.load(g0 + 2 * oz, mask=vz, other=0.0)
    a_z_im = tl.load(g0 + 2 * oz + 1, mask=vz, other=0.0)
    b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)
    b_x_im = tl.load(g1 + 2 * ox + 1, mask=vx, other=0.0)
    b_z_re = tl.load(g1 + 2 * oz, mask=vz, other=0.0)
    b_z_im = tl.load(g1 + 2 * oz + 1, mask=vz, other=0.0)
    c_x_re = tl.load(g2 + 2 * ox, mask=vx, other=0.0)
    c_x_im = tl.load(g2 + 2 * ox + 1, mask=vx, other=0.0)
    c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)
    c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)

    # --- Bloch phase on the wrapped lane, BEFORE the difference -----------------
    # SHIFTED operands only; the beta partners (a, b centers) are never rotated.
    if PHX:
        rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, pxi, EXPANSION)
        b_x_re = tl.where(wx, rot_re, b_x_re)
        b_x_im = tl.where(wx, rot_im, b_x_im)
        rot_re, rot_im = _rotate_field_left(c_x_re, c_x_im, pxr, pxi, EXPANSION)
        c_x_re = tl.where(wx, rot_re, c_x_re)
        c_x_im = tl.where(wx, rot_im, c_x_im)
    if PHY:
        rot_re, rot_im = _rotate_field_left(a_y_re, a_y_im, pyr, pyi, EXPANSION)
        a_y_re = tl.where(wy, rot_re, a_y_re)
        a_y_im = tl.where(wy, rot_im, a_y_im)
        rot_re, rot_im = _rotate_field_left(c_y_re, c_y_im, pyr, pyi, EXPANSION)
        c_y_re = tl.where(wy, rot_re, c_y_re)
        c_y_im = tl.where(wy, rot_im, c_y_im)
    if PHZ:
        rot_re, rot_im = _rotate_field_left(a_z_re, a_z_im, pzr, pzi, EXPANSION)
        a_z_re = tl.where(wz, rot_re, a_z_re)
        a_z_im = tl.where(wz, rot_im, a_z_im)
        rot_re, rot_im = _rotate_field_left(b_z_re, b_z_im, pzr, pzi, EXPANSION)
        b_z_re = tl.where(wz, rot_re, b_z_re)
        b_z_im = tl.where(wz, rot_im, b_z_im)

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
    t0_re = ((c_y_re - c_re) + (b_re - b_z_re))
    t0_im = ((c_y_im - c_im) + (b_im - b_z_im))
    t1_re = ((a_z_re - a_re) + (c_re - c_x_re))
    t1_im = ((a_z_im - a_im) + (c_im - c_x_im))
    t2_re = ((b_x_re - b_re) + (a_re - a_y_re))
    t2_im = ((b_x_im - b_im) + (a_im - a_y_im))
    curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
    curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
    curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

    # --- the beta term (stepping._special_kz_beta_term), CENTER partners only ---
    # AFTER the dtdx curl, BEFORE the ownership mask — the array path's order
    # (S:356-363 / S:438-445 against S:369/S:450). Coefficient LEFT (S:784).
    if HAS_BETA:
        t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)
        curl0_re = curl0_re - t_re
        curl0_im = curl0_im - t_im
        t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)
        curl1_re = curl1_re - t_re
        curl1_im = curl1_im - t_im

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    if BACKWARD:
        if BCY == METALLIC:
            curl0_re = tl.where(at_y, 0.0, curl0_re)
            curl0_im = tl.where(at_y, 0.0, curl0_im)
        if BCZ == METALLIC:
            curl0_re = tl.where(at_z, 0.0, curl0_re)
            curl0_im = tl.where(at_z, 0.0, curl0_im)
        if BCX == METALLIC:
            curl1_re = tl.where(at_x, 0.0, curl1_re)
            curl1_im = tl.where(at_x, 0.0, curl1_im)
        if BCZ == METALLIC:
            curl1_re = tl.where(at_z, 0.0, curl1_re)
            curl1_im = tl.where(at_z, 0.0, curl1_im)
        if BCX == METALLIC:
            curl2_re = tl.where(at_x, 0.0, curl2_re)
            curl2_im = tl.where(at_x, 0.0, curl2_im)
        if BCY == METALLIC:
            curl2_re = tl.where(at_y, 0.0, curl2_re)
            curl2_im = tl.where(at_y, 0.0, curl2_im)
    else:
        if BCX == METALLIC:
            curl0_re = tl.where(at_x, 0.0, curl0_re)
            curl0_im = tl.where(at_x, 0.0, curl0_im)
        if BCY == METALLIC:
            curl1_re = tl.where(at_y, 0.0, curl1_re)
            curl1_im = tl.where(at_y, 0.0, curl1_im)
        if BCZ == METALLIC:
            curl2_re = tl.where(at_z, 0.0, curl2_re)
            curl2_im = tl.where(at_z, 0.0, curl2_im)

    # --- split-field recurrence (stepping._apply_pml_update) -------------------
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    p0_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)
    p0_im = tl.load(u0 + 2 * idx + 1, mask=live, other=0.0)
    q_re, q_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)
    q_re = q_re - curl0_re
    q_im = q_im - curl0_im
    n0_re, n0_im = _mul_field_left(q_re, q_im, si_y, EXPANSION)
    e_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_z, EXPANSION)
    r_re = (r_re + n0_re) - p0_re
    r_im = (r_im + n0_im) - p0_im
    v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)

    p1_re = tl.load(u1 + 2 * idx, mask=live, other=0.0)
    p1_im = tl.load(u1 + 2 * idx + 1, mask=live, other=0.0)
    q_re, q_im = _mul_field_left(p1_re, p1_im, km_z, EXPANSION)
    q_re = q_re - curl1_re
    q_im = q_im - curl1_im
    n1_re, n1_im = _mul_field_left(q_re, q_im, si_z, EXPANSION)
    e_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_x, EXPANSION)
    r_re = (r_re + n1_re) - p1_re
    r_im = (r_im + n1_im) - p1_im
    v1_re, v1_im = _mul_field_left(r_re, r_im, si_x, EXPANSION)

    p2_re = tl.load(u2 + 2 * idx, mask=live, other=0.0)
    p2_im = tl.load(u2 + 2 * idx + 1, mask=live, other=0.0)
    q_re, q_im = _mul_field_left(p2_re, p2_im, km_x, EXPANSION)
    q_re = q_re - curl2_re
    q_im = q_im - curl2_im
    n2_re, n2_im = _mul_field_left(q_re, q_im, si_x, EXPANSION)
    e_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_y, EXPANSION)
    r_re = (r_re + n2_re) - p2_re
    r_im = (r_im + n2_im) - p2_im
    v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)

    # --- stores: u then f (kernels.py:193-198 order), both planes ---------------
    tl.store(u0 + 2 * idx, n0_re, mask=live)
    tl.store(u0 + 2 * idx + 1, n0_im, mask=live)
    tl.store(u1 + 2 * idx, n1_re, mask=live)
    tl.store(u1 + 2 * idx + 1, n1_im, mask=live)
    tl.store(u2 + 2 * idx, n2_re, mask=live)
    tl.store(u2 + 2 * idx + 1, n2_im, mask=live)
    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


# ---------------------------------------------------------------------------
# The extended probe contract
# ---------------------------------------------------------------------------

def beta_expansion_license(record: Any) -> Dict[str, Any]:
    """The full licensing verdict for the beta tranche — arm, basis, refusals.

    :func:`complex_fields.expansion_license` over the EXTENDED pattern set: the
    record must name the CuPy backend, classify every pattern of
    :data:`BETA_PROBE_PATTERNS` — the base four PLUS the imaginary-coefficient-
    left product — and the patterns that CAN discriminate must agree. A record
    cut before this tranche (no :data:`BETA_PROBE_PATTERN` key) refuses here even
    though it licenses the base complex kernels; §4(k) of the transcription spec.

    THE NON-DISCRIMINATING CLAUSE APPLIES TO EVERY PATTERN HERE, base four
    included, and that is the 2026-08-15 correction. This function used to carry
    its own loop that accepted ``AMBIGUOUS_BOTH`` for :data:`BETA_PROBE_PATTERN`
    and for nothing else, so a base pattern that could not tell the arms apart
    VETOED the licence. The justification for the special case was arithmetic —
    the engine's beta coefficients carry a SIGNED-ZERO real word, so with
    ``c_re`` exactly ±0 the fused arm's extra product is exact and the two arms
    are bit-identical (measured AMBIGUOUS_BOTH on the reference NumPy, all 16
    corpus coefficients, 2026-08-15) — but that argument is about WHY a pattern
    goes blind, not about WHICH pattern may. Under the flush policy the same
    thing happens to base patterns for a different reason, and vetoing there
    refuses a platform that has answered the question on the patterns that can
    answer it. The base family fixed this first; this is the same clause, not a
    second copy of it.

    WHAT DOES NOT SOFTEN. :data:`complex_fields.NEITHER` still refuses by name, a
    disagreement across coefficients still refuses by name, and a pattern that
    claims ambiguity while its own detail block measures the licensable arms some
    words APART is a self-contradicting record and refuses — excluded and
    disagreeing are opposite situations, and only the first is ever dropped from
    the agreement test.
    """
    return expansion_license(record, BETA_PROBE_PATTERNS)


def beta_expansion_from_probe(record: Any) -> Optional[int]:
    """The single ``EXPANSION`` constexpr the complex beta CURL may bind, or None.

    The narrow answer :func:`beta_expansion_license` computes; see there for the
    rule. Kept as the callers' entry point because a plan builder wants the
    constexpr or a refusal, not a verdict document.
    """
    return beta_expansion_license(record)["expansion"]


def _beta_expansion_reasons(probe: Any = None) -> List[str]:
    """Clause 13, extended: the beta curl's EXPANSION binding needs the probe
    artifact to carry the NEW pattern too — refusal by name when it does not.

    AND CUT UNDER THE POLICY THIS RUN USES, which this clause used not to check
    at all (2026-08-16). ``complex_fields._expansion_reasons`` ends in
    :func:`~complex_fields.expansion_policy_reasons`; this function and
    ``folded_complex._parity_expansion_reasons`` did not, so K3a/K3b licensed a
    flush-cut record that the base seam refused on the same input with the same
    policy readable. Extending the PATTERN SET never shrinks the question asked
    about the artifact.

    ``probe`` is three-valued — a record, ``None`` for "nothing offered", or a
    :class:`~meep_gpu.expansion_refusal.RefusedExpansionProbe` for "offered and
    refused", which never falls back to the environment.
    """
    # Imported in the BODY so ``_policy_in_force`` resolves through
    # ``complex_fields``' namespace at call time rather than being frozen here.
    from ..expansion_refusal import RefusedExpansionProbe  # noqa: PLC0415
    from .complex_fields import (  # noqa: PLC0415
        _policy_in_force, expansion_certification_reasons,
        expansion_policy_reasons)

    record = probe if probe is not None else load_expansion_probe()
    if isinstance(record, RefusedExpansionProbe):
        return [f"the expansion probe artifact offered through {record.key} was "
                f"REFUSED by this dispatch and may not license an arm at any "
                f"later rung: " + "; ".join(record.reasons)]
    if record is None:
        return [
            "no complex-multiply expansion probe artifact is available for this "
            "backend; the EXPANSION constexpr is a measured platform fact and "
            "may not be guessed (the beta tranche additionally requires the "
            f"{BETA_PROBE_PATTERN!r} pattern)"]
    verdict = beta_expansion_license(record)
    if verdict["expansion"] is None:
        # The verdict's OWN refusals, verbatim, not a summary of them: which
        # pattern failed and how is the whole value of a refusal by name, and a
        # one-line paraphrase ("missing, ambiguous, or lacks the pattern") sends
        # the reader back to the artifact to find out which of the three it was.
        return [
            f"the expansion probe artifact does not license an EXPANSION over "
            f"the EXTENDED pattern set {BETA_PROBE_PATTERNS} this tranche binds "
            f"(the base four plus {BETA_PROBE_PATTERN!r})"
        ] + list(verdict["refusals"])
    # BOTH policy questions — artifact-vs-run and kernel-vs-run. K3a/K3b's beta
    # curl was certified under the same policy as the base four, so the second
    # clause is the same constant asked at this tranche's seam.
    policy = _policy_in_force()
    return (list(expansion_policy_reasons(record, policy))
            + list(expansion_certification_reasons(policy)))


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors ``coverage._grid_reasons`` (real family) and
# ``complex_fields._complex_grid_reasons`` (complex family) so each pair can be
# diffed. In BOTH families clause 12 (beta) is INVERTED: this product REQUIRES
# beta != 0, where every shipped kernel requires 0 — the exact mirror of the
# complex tranche's clause-2 inversion, and the reason no admitted-overlap
# ambiguity can arise while dispatch stays disabled. Everything else is KEPT,
# restated rather than imported, because the shipped reason lists are built
# inside functions whose beta clause cannot be subtracted from outside.


def _beta_real_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every REAL-family beta predicate shares."""
    reasons: List[str] = []

    # 1. CuPy backend.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage REQUIRED: a complex-storage beta run belongs to the
    #    complex beta variant (kz_2d='complex' forces complex64,
    #    simulation.py:1554-1560; fields.py:571-573).
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: a complex-storage beta run "
                       "belongs to the complex beta variant")

    # 3. An absorber that actually absorbs (split-field family only, §4(h)).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "split-field path only; no-PML beta is out of tranche)")

    # 4. Only the two ghost rules the curl kernel writes.
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

    # 5. No mirror plane: Phase A. The folded REAL beta curl is Phase B and
    #    lifts this through the folded family's own predicate (§3/§4(g)).
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (the folded real beta curl is "
                       "Phase B; this Phase A kernel does not carry the fold)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian, effective 2-D. Grid refuses beta off 2-D Cartesian at
    #    construction (grid.py:668-697, MEEP fields.cpp:546-547) — RESTATED
    #    here, never inferred from the constructor guard (§4(d)/(e)).
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried (and the "
                       "grid itself refuses beta there, grid.py:668-697)")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")
    if int(getattr(grid, "dimensions", 0)) != 2:
        reasons.append(f"grid dimensions={getattr(grid, 'dimensions', None)!r} "
                       f"is not the effective-2-D grid beta requires "
                       f"(grid.py:668-697; MEEP fields.cpp:546-547)")

    # 7. k = 0. Real storage carries no Bloch phase (the array path raises on
    #    the pairing); an in-plane k with beta is the COMPLEX variant's domain.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r} "
                       f"(real storage carries no Bloch phase; in-plane k + "
                       f"beta is the complex variant's domain)")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 9a/9b. A registered susceptibility must be one this package understands
    #    (dispersion is ADMITTED for the curl, §4(c) pins the kinds).
    reasons.extend(_susceptibility_reasons(fields))

    # 10. No instantaneous nonlinearity (§4(b)).
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried; that family is a separate tranche)")

    # 11. No BFAST (§4(a) — a separate family with its own IIR state).
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is a "
                       "separate family)")

    # 12 (INVERTED). Beta must be NONZERO: a beta = 0 run belongs to the
    #    certified plain kernels, and the array path never enters the term
    #    (stepping.py:384/:467).
    if float(getattr(grid, "beta", 0.0)) == 0.0:
        reasons.append("grid.beta is zero: this product exists only for "
                       "special_kz runs; a beta = 0 run belongs to the "
                       "certified plain kernels")

    # 12b. Real storage + off-diagonal epsilon + beta: stepping raises
    #    (stepping.py:800-810) and MEEP aborts (fields.cpp:548-549) — the
    #    implicit-i trick cancels only while TE/TM stay uncoupled (§4(f)).
    #    Refused for the WHOLE real family, not only update_E: the run itself
    #    raises inside the first beta curl.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("off-diagonal epsilon with beta in REAL storage: the "
                       "implicit-i trick no longer cancels (stepping.py:800-810 "
                       "raises; MEEP fields.cpp:548-549 aborts) — the run needs "
                       "force_complex_fields=True")

    # 9c. Stored E — the invariant behind admitting dispersion for the curl.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return reasons


def _curl_conductivity_reasons(fields: Any, sub_step: str) -> List[str]:
    """Clause 8, strict: a conductivity on this curl's targets is refused by
    name (the conductive product is ``conductivity.py``'s, and ITS predicate
    refuses beta, conductivity.py:519-520 — no silent overlap in either
    direction). A missing or non-callable reader is refused OUTRIGHT, the
    complex tranche's stricter reading: inferring "no conductivity" from the
    absence of ``condfac_for`` is admission by attribute absence."""
    reasons: List[str] = []
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
        return reasons
    for target in CURL_SUB_STEPS[sub_step]:
        try:
            conductive = reader(target) is not None
        except Exception as exc:  # noqa: BLE001 - unreadable means not covered
            reasons.append(f"condfac_for({target!r}) raised {exc!r}")
            continue
        if conductive:
            reasons.append(
                f"a conductivity is installed on {target}: this kernel "
                f"transcribes the plain split-field recurrence only, and the "
                f"conductive family's own predicate refuses beta — the "
                f"beta-under-condinv composition is a named follow-up, not a "
                f"silent overlap")
    return reasons


def beta_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May the REAL special_kz curl kernel step this (fields, pml, sub_step)?

    Positive clauses only; a failing clause appends its reason and the scan
    continues. This is ``coverage.pml_curl_coverage``'s clause set with clause
    12 INVERTED (beta must be nonzero) plus the real-offdiag-beta refusal, and
    ONE DELIBERATE NARROWING recorded here rather than left silent: the
    shipped predicate's allocation+layout clause runs over ALL six fu_ volumes,
    both source triples and the 18-volume layout set as a SET
    (coverage.py:306-315 — "coverage is a set, not a la carte"), where clauses
    13/14 below check the NAMED SUB-STEP'S 9 arrays only. That is the
    per-sub-step scoping the certified complex tranche already uses
    (``complex_fields.complex_pml_curl_coverage``), every pointer the launched
    kernel touches is still checked, and the engine route allocates all 18
    together (``Fields.enable_pml_storage``) so no engine-built configuration
    changes verdict — but a hand-built run missing the OTHER sub-step's
    volumes is admitted here and refused by the shipped clause. Everything
    else is kept, restated.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _beta_real_grid_reasons(fields, pml, grid)
    reasons.extend(_curl_conductivity_reasons(fields, sub_step))

    # 13. The named sub-step's targets, auxiliaries and sources — NARROWER
    #     than the shipped clause (coverage.py:306-315 checks all 18 volumes
    #     as a set); the deliberate per-sub-step scoping the docstring records.
    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    # 14. Layout: float32, C-contiguous, grid.shape, int32 index bound.
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def _beta_complex_grid_reasons(fields: Any, pml: Any, grid: Any,
                               probe: Any = None,
                               curl_probe_contract: bool = True) -> List[str]:
    """The clauses every COMPLEX-family beta predicate shares.

    ``curl_probe_contract`` selects clause 13's pattern set: the beta CURL
    binds the EXTENDED set (:data:`BETA_PROBE_PATTERNS`, §4(k)); the certified
    constitutive kernel binds its own BASE set (grouping choice 5).
    """
    reasons: List[str] = []

    # 1. CuPy backend.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2 (INVERTED, as the complex tranche inverts it). Complex64 storage
    #    REQUIRED; a real beta run belongs to the real beta kernel.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32 (neither force_complex_fields nor a "
            "nonzero k_point): a real beta run belongs to the real beta kernel")

    # 3. An absorber that actually absorbs.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "complex split-field path only)")

    # 4. Ghost rules; fold refused ALWAYS in the complex family (§4(g) — the
    #    complex fold is the complex tranche's inherited composition gap, not a
    #    beta gap); cylindrical and non-2-D refused, restated.
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
        reasons.append("a mirror plane is active (complex-by-fold is the "
                       "complex tranche's composition gap, not a beta gap)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried (and the "
                       "grid itself refuses beta there, grid.py:668-697)")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")
    if int(getattr(grid, "dimensions", 0)) != 2:
        reasons.append(f"grid dimensions={getattr(grid, 'dimensions', None)!r} "
                       f"is not the effective-2-D grid beta requires "
                       f"(grid.py:668-697; MEEP fields.cpp:546-547)")

    # 5. Per-axis phase consistency (complex_fields clause 5, restated): a
    #    phased axis must resolve PERIODIC; a metallic axis must carry k = 0;
    #    an unreadable phase is refused, never admitted as unphased.
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
                            f"resolved to {kind!r}; only a periodic wrap can "
                            f"carry a phase")
            if kind == "metallic" and float(k_point[axis]) != 0.0:
                reasons.append(
                    f"axis {axis} is metallic with k component "
                    f"{k_point[axis]!r}; a PEC wall gives the axis no lattice "
                    f"vector for the phase")

    # 5b. No k component on the INVARIANT axis: beta IS the analytic z
    #     dependence (``exp(i*2*pi*beta*z)``, grid.py:668-697), so a planted
    #     k_point[2] would ride the axis beta already carries — the Grid
    #     refuses the pairing at construction ("translationally invariant"),
    #     and it is RESTATED here, never inferred from the constructor guard
    #     (the same discipline the dimensions/cylindrical clauses state).
    #     Byte-safe even when planted (both paths read the precomputed
    #     bloch_phases table, which stays None for the invariant axis) — this
    #     clause exists so the refusal is named, not implied.
    if len(k_point) > 2 and float(k_point[2]) != 0.0:
        reasons.append(
            f"k_point component z = {k_point[2]!r} rides the invariant axis "
            f"whose dependence beta already carries analytically "
            f"(grid.py:668-697 refuses the pairing at construction)")

    # 6. No conductivity anywhere on the curl targets (strict, complex-tranche
    #    reading; the conductive complex product does not exist).
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
                reasons.append(f"a conductivity is installed on {target}; "
                               f"conductive complex stepping is a separate, "
                               f"unbuilt product")

    # 7. No dispersion (complex ADE is a future tranche; no beta anchor
    #    carries a susceptibility under complex storage).
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: complex-storage ADE is "
                       "a future tranche")
    reasons.extend(_susceptibility_reasons(fields))

    # 8. No nonlinearity, no BFAST; beta INVERTED — must be nonzero.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is a "
                       "separate family)")
    if float(getattr(grid, "beta", 0.0)) == 0.0:
        reasons.append("grid.beta is zero: this product exists only for "
                       "special_kz runs; a beta = 0 complex run belongs to the "
                       "certified complex tranche")

    # 9. Stored E.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 13. EXPANSION binding requires a measured probe artifact — the EXTENDED
    #     set for the beta curl (§4(k)), the base set for the certified
    #     constitutive kernel (its own contract).
    if curl_probe_contract:
        reasons.extend(_beta_expansion_reasons(probe))
    else:
        reasons.extend(_expansion_reasons(probe))

    return reasons


def beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                 probe: Any = None) -> Coverage:
    """May the COMPLEX special_kz curl kernel step this (fields, pml, sub_step)?

    ``complex_fields.complex_pml_curl_coverage``'s clause set with clause 8's
    beta refusal INVERTED and the probe clause extended to
    :data:`BETA_PROBE_PATTERNS`. Off-diagonal epsilon is ADMITTED here exactly
    as the certified complex curl admits it (constitutive-only under complex
    storage; the real-storage hazard is the REAL family's 12b clause).
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _beta_complex_grid_reasons(fields, pml, grid, probe)

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


def beta_run_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the CERTIFIED real constitutive kernel step ``update_H``/``update_E``
    on a BETA run?

    ``coverage.constitutive_coverage``'s clause set with the beta clause
    INVERTED and nothing else changed: the constitutive sub-steps read nothing
    beta-dependent (``_apply_constitutive_pml``, stepping.py:2112-2143), so
    admission delegates the ARITHMETIC to the certified
    ``kernels.constitutive_step`` unchanged — no new kernel, no new sub-step.
    The E side keeps the shipped refusals (polarizations belong to the ADE
    kernel; off-diagonal rows are non-element-wise — under real storage with
    beta the whole RUN is refused by the curl predicate's 12b anyway).
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _beta_real_grid_reasons(fields, pml, grid)

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is "
                "(D - sum P), not D — that configuration belongs to the ADE "
                "kernel, exactly as the shipped predicate rules")
        # Off-diagonal epsilon is already refused family-wide by 12b (real
        # storage + beta raises in the curl, stepping.py:800-810); the shipped
        # E-side element-wise reason would be redundant here.

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
                                           probe: Any = None) -> Coverage:
    """May the CERTIFIED complex constitutive kernel step this side on a BETA run?

    ``complex_fields.complex_constitutive_coverage`` with the beta clause
    INVERTED. The kernel launched is the certified
    ``bloch_constitutive_step`` unchanged, so the probe clause here is the
    BASE pattern set (that kernel's own contract), not the extended one —
    grouping choice 5 in the module docstring.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _beta_complex_grid_reasons(fields, pml, grid, probe,
                                         curl_probe_contract=False)

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("an off-diagonal chi1inv row is installed (the row "
                       "product reads neighbours; this sub-step is element-wise)")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class BetaPmlCurlPlan:
    """A launchable, allocation-free REAL beta PML curl sub-step.

    ``launch.PmlCurlPlan`` plus the two host-rounded beta scalars and the
    ``HAS_BETA`` constexpr. Same two construction routes and one launch route:
    :func:`plan_beta_pml_curl` from the engine's objects through the predicate,
    :func:`plan_beta_pml_curl_from_arrays` from bare device arrays for the gate
    (including the deliberately wrong ones), both launched through :meth:`run`.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "beta_plus",
                 "beta_minus", "has_beta", "backward", "bc", "block",
                 "num_warps", "_targets", "_aux", "_sources", "_coefficients",
                 "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc,
                 beta_plus: float, beta_minus: float, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None, has_beta: int = 1) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        # Already f32-rounded by the host (beta_curl_coefficients); float()
        # keeps the bits, and Triton types a Python float argument as fp32.
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.has_beta = int(has_beta)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # The override exists for exactly one caller: the gate's mutation legs.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. ``guard`` is the gate's, not a caller's."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else beta_pml_curl_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.beta_plus, self.beta_minus,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"BetaPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, beta_plus={self.beta_plus!r}, "
                f"has_beta={self.has_beta}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_beta_pml_curl(fields: Any, pml: Any, sub_step: str,
                       block: Optional[int] = None,
                       num_warps: Optional[int] = None
                       ) -> Optional[BetaPmlCurlPlan]:
    """Build a REAL beta curl plan from the engine's objects, or None.

    None is the only refusal (Y-style). The coefficients come from
    :func:`beta_curl_coefficients` with ``magnetic`` from the sub-step —
    identical arithmetic to the array path's per-call-site computation.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not beta_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    plus, minus = beta_curl_coefficients(grid.beta, grid.dt,
                                         magnetic=(sub_step == "step_B"),
                                         complex_storage=False)
    return BetaPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        plus, minus,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
    )


def plan_beta_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                   flat: Dict[str, Any], codes, dtdx: float,
                                   beta_plus: float, beta_minus: float,
                                   block: Optional[int] = None,
                                   kernel: Any = None,
                                   num_warps: Optional[int] = None,
                                   has_beta: int = 1) -> BetaPmlCurlPlan:
    """Build a REAL beta curl plan from bare device arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones;
    ``kernel=`` carries the mutation override and ``has_beta=0`` the identity
    leg's certified-kernel arm.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return BetaPmlCurlPlan(
        sub_step, shape, dtdx, codes, beta_plus, beta_minus,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
    )


class BetaBlochPmlCurlPlan:
    """A launchable, allocation-free COMPLEX beta PML curl sub-step.

    ``complex_fields.ComplexPmlCurlPlan`` plus the four beta coefficient words
    and ``HAS_BETA``. The complex volumes are bound as float32 WORD VIEWS at
    plan time; ``n_elem`` stays the COMPLEX cell count.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "phased", "phase_values", "beta_words", "has_beta",
                 "expansion", "block", "num_warps", "_targets", "_aux",
                 "_sources", "_coefficients", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, phased,
                 phase_values, beta_words, expansion: int, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None, has_beta: int = 1) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple(float(value) for value in phase_values)
        # ((bp_re, bp_im), (bm_re, bm_im)) — already complex64-rounded words
        # with the signed-zero real parts passed through (float() keeps them).
        self.beta_words = tuple(tuple(float(word) for word in pair)
                                for pair in beta_words)
        self.has_beta = int(has_beta)
        self.expansion = int(expansion)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as the others."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = (self._kernel if self._kernel is not None
                  else beta_bloch_pml_curl_step)
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        (bp_re, bp_im), (bm_re, bm_im) = self.beta_words
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.phase_values,
            bp_re, bp_im, bm_re, bm_im,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            PHX=self.phased[0], PHY=self.phased[1], PHZ=self.phased[2],
            HAS_BETA=self.has_beta,
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"BetaBlochPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, phased={self.phased}, "
                f"beta_words={self.beta_words!r}, has_beta={self.has_beta}, "
                f"expansion={self.expansion}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             num_warps: Optional[int] = None,
                             probe: Any = None
                             ) -> Optional[BetaBlochPmlCurlPlan]:
    """Build a COMPLEX beta curl plan from the engine's objects, or None."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not beta_bloch_pml_curl_coverage(fields, pml, sub_step, probe=probe).covered:
        return None
    record = probe if probe is not None else load_expansion_probe()
    expansion = beta_expansion_from_probe(record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    phases = bloch_phase_table(grid, kinds)
    phased, values = _phase_arguments(phases, backward=bool(spec["backward"]))
    beta_words = beta_curl_coefficients(grid.beta, grid.dt,
                                        magnetic=(sub_step == "step_B"),
                                        complex_storage=True)
    return BetaBlochPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        phased, values, beta_words, expansion,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
    )


def plan_beta_bloch_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                         flat: Dict[str, Any], codes,
                                         phases: Sequence[Optional[complex]],
                                         dtdx: float, expansion: int,
                                         beta_words,
                                         block: Optional[int] = None,
                                         kernel: Any = None,
                                         num_warps: Optional[int] = None,
                                         has_beta: int = 1
                                         ) -> BetaBlochPmlCurlPlan:
    """Build a COMPLEX beta curl plan from bare device arrays — the gate's route.

    ``beta_words`` is ``((bp_re, bp_im), (bm_re, bm_im))`` — normally
    :func:`beta_curl_coefficients`'s output, or a deliberately wrong pair on a
    mutation leg. The step_D phase conjugation is applied HERE per sub-step,
    exactly as the complex tranche's from-arrays builder applies it.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    phased, values = _phase_arguments(tuple(phases), backward=bool(spec["backward"]))
    return BetaBlochPmlCurlPlan(
        sub_step, shape, dtdx, codes, phased, values, beta_words,
        int(expansion),
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
    )


def plan_beta_run_constitutive(fields: Any, pml: Any, side: str,
                               block: Optional[int] = None,
                               num_warps: Optional[int] = None
                               ) -> Optional[ConstitutivePlan]:
    """A CERTIFIED real constitutive plan for a beta run, or None.

    The arithmetic and the plan class are ``launch.ConstitutivePlan``'s,
    untouched; only the ADMISSION is this module's
    (:func:`beta_run_constitutive_coverage`, the shipped predicate minus the
    beta clause). The binding below restates ``launch.plan_constitutive``'s.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not beta_run_constitutive_coverage(fields, pml, side).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ConstitutivePlan(
        side, fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
    )


def plan_beta_run_complex_constitutive(fields: Any, pml: Any, side: str,
                                       block: Optional[int] = None,
                                       num_warps: Optional[int] = None,
                                       probe: Any = None
                                       ) -> Optional[ComplexConstitutivePlan]:
    """A CERTIFIED complex constitutive plan for a beta run, or None.

    The kernel and plan class are ``complex_fields``' (certified), untouched;
    the EXPANSION binds through the BASE pattern set because that is the
    certified kernel's own contract (module docstring, grouping choice 5).
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not beta_run_complex_constitutive_coverage(fields, pml, side,
                                                  probe=probe).covered:
        return None
    record = probe if probe is not None else load_expansion_probe()
    expansion = expansion_from_probe(record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ComplexConstitutivePlan(
        side, fields.grid.shape, expansion,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# WIRING — none, deliberately
# ---------------------------------------------------------------------------
#
# This module is NOT imported by the package ``__init__``, is not consulted by
# ``launch.plan_step``, and does not touch production dispatch, which keeps
# returning None on every branch. That is SAFE to defer, not merely convenient:
# ``coverage._grid_reasons`` clause 12 and the complex tranche's clause 8
# already refuse every beta run, so no shipped predicate can admit a
# configuration these predicates also admit — an admitted-overlap ambiguity
# cannot arise until a later coordinated change adds launch-side forwarders,
# and THAT change re-runs the byte gate. Tests and the gate import
# ``meep_gpu.triton_kernels.special_kz`` directly (the package ``__init__``
# eagerly imports only ``coverage``, so the direct import is safe on a
# Triton-less host).
