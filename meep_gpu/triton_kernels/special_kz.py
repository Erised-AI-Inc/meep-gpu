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
* position: MEEP runs the term as a pass of its own after every component's curl
  (step_db.cpp:161-175), and so does the array path: the curl forms first, then
  ``_special_kz_beta_term`` forms the increment, masked to an exact ``+0`` on every
  cell the curl's ownership mask drops, and ``_apply_folded_increment`` subtracts it
  through the pass's own additive ladder (:func:`.forms.fold_ladder`): ``f - (sinv *
  q)`` outside the component's dsigu chunk, ``fu - (sinv * q)`` and ``f - (sinv_u *
  (sinv * q))`` inside it;
* partners: the SAME-CELL component snapshots (:293 electric, :408 magnetic).
  Per ``B_CURL_TERMS``/``D_CURL_TERMS`` (:213-223) every beta partner is a
  CENTER operand the certified curl kernels already load — Bx: the second
  source Ey center; By: the first source Ex center; Dx: second Hy; Dy: first Hx
  — so the term costs no new pointer and no extra memory traffic, and the load
  reused must be the UNSHIFTED center one, never an ox/oy/oz shifted operand.
  In the complex family the Bloch connection applies only to shifted
  operands, so reusing the center registers keeps the beta partner UNROTATED by
  construction;
* no new state array (contrast BFAST's per-component IIR state, :889-893), no
  ghost rule, no fold parity, no wrap; ``update_H``/``update_E``
  (``_apply_constitutive_pml``, :2065-2096) read nothing beta-dependent.

TWO KERNELS, one per storage family:

* :func:`beta_pml_curl_step` — ``kernels.pml_curl_step``'s body plus the
  constexpr-gated beta pass after the forms: per affected target one f32 product
  of the host-rounded word and the loaded center operand, negated by ``* -1.0``,
  masked, and subtracted through the ladder;
* :func:`beta_bloch_pml_curl_step` — ``complex_fields.bloch_pml_curl_step``'s
  body plus the same pass on word pairs, the increment a quarter turn of the
  center partner's parts by one real word (``part_products.quarter_turn``):
  ``q_re = c * g_im``, ``q_im = (c * g_re) * -1.0``. No complex product is
  formed; :func:`_mul_imag_coefficient_left` stays defined for the sibling H->D
  weld alone.

GROUPING CHOICES the gate must hold (stepping.py forces none of these):

1. TWO host-rounded scalars per launch — one per sign — rather than one scalar
   negated in-kernel. The array path computes ``sign * 2*pi*beta*dt`` in f64 per
   call site and rounds each once (:770/:784); f64 negation and f32 rounding
   commute exactly, but binding both keeps the transcription literal per call
   site rather than resting on that identity.
2. The real increment is ``(c * g) * -1.0``, the array path's ``-(c * g)``, and
   the ladder SUBTRACTS it, the array path's spelling (``x - (-y)`` is ``x + y``
   in IEEE-754, signed zeros included). The negation is a multiply by ``-1.0``,
   exact on every input, so it never leans on how Triton lowers unary minus.
3. Complex storage forms NO complex product: the array path turns the
   partner's parts a quarter by one real word (``part_products.quarter_turn``,
   the word negated on the D side), so the pass is ``q_re = c * g_im`` and
   ``q_im = (c * g_re) * -1.0`` per target, one rounding per product, and nothing
   can contract it into a fused multiply-add.
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
   classes.
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
    ComplexConstitutivePlan,
    _complex_layout_reasons,
    _connect,
    _word_view,
    bloch_phase_table,
    connection_arguments,
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
from .launch import SUB_STEPS, ConstitutivePlan, CupyPointer, _flat, chunk_bounds_for

# Same constexpr codes as ``kernels.PERIODIC``/``kernels.METALLIC``. Restated (not
# imported from kernels.py) because importing kernels.py would import Triton
# unconditionally and defeat this module's host-only coverage route; the laptop
# tests pin both against their originators so the restatements cannot drift.
PERIODIC = tl.constexpr(0)
METALLIC = tl.constexpr(1)

# The forms, the fold ladder and the chunk record's word offsets, read by the kernels
# below. Imported only when Triton is (both modules import it at module scope), so
# the predicates and the plan builders still answer on a host without it.
if _TRITON_IMPORT_ERROR is None:
    from .forms import (axis_present, curl_form, curl_stencil,  # noqa: E402
                        fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

#: Elements per program — real cells for the real kernel, complex CELLS for the
#: complex one. Restated from ``kernels.DEFAULT_BLOCK`` for the same import
#: reason; no autotune (the kernels write their own inputs in place,
#: kernels.py:56-59).
DEFAULT_BLOCK = 256

__all__ = [
    "BetaBlochPmlCurlPlan",
    "BetaPmlCurlPlan",
    "beta_bloch_pml_curl_coverage",
    "beta_bloch_pml_curl_step",
    "beta_curl_coefficients",
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
# The host coefficient — stepping._special_kz_beta_term, transcribed
# ---------------------------------------------------------------------------

def beta_curl_coefficients(beta: float, dt: float, magnetic: bool,
                           complex_storage: bool) -> Tuple[float, float]:
    """The (plus-sign, minus-sign) beta words, host-computed and rounded EXACTLY as
    ``stepping._special_kz_beta_term`` computes and rounds them.

    Per sign: ``coefficient = sign * 2*pi * beta * dt`` in float64, in that order.
    Real storage rounds it once to float32 (``partner.dtype.type(coefficient)``), the
    same word on both sub-steps (MEEP's implicit-i trick). Complex storage forms the
    increment as a quarter turn of the partner's parts by ONE real word
    (``part_products.quarter_turn(coefficient if magnetic else -coefficient, ...)``):
    the B side's ``+i`` takes the coefficient, the D side's ``-i`` its negation, each
    rounded once to the parts' ``realnum``. No complex coefficient exists on either
    storage: the array path forms no complex product.

    Returns two floats. ``float()`` of a numpy.float32 keeps the bits, and Triton types a
    Python float argument as fp32, so the words the kernel receives are the array path's.
    """
    import numpy  # noqa: PLC0415

    out: List[float] = []
    for sign in (1.0, -1.0):
        coefficient = sign * 2.0 * math.pi * float(beta) * float(dt)
        if complex_storage and not magnetic:
            coefficient = -coefficient  # the D side's quarter turn is by -i
        out.append(float(numpy.float32(coefficient)))  # one rounding to realnum
    return out[0], out[1]


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------

@triton.jit
def _mul_imag_coefficient_left(c_re, c_im, z_re, z_im):
    """``(c_re + i*c_im) * (z_re + i*z_im)``, every product and sum rounded on its own.

    NONE OF THIS MODULE'S KERNELS CALLS IT. The beta pass forms no complex product: it
    is a quarter turn of the partner's parts by one real word
    (``part_products.quarter_turn``), which :func:`beta_bloch_pml_curl_step` spells
    inline. The helper keeps its name and signature for the sibling H->D weld
    (``beta_complex_fused_hd_pair``), which still transcribes the previous full product
    until its own lane re-lifts it.
    """
    out_re = (c_re * z_re) - (c_im * z_im)
    out_im = (c_re * z_im) + (c_im * z_re)
    return out_re, out_im


@triton.jit
def beta_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
    u0, u1, u2,                       # auxiliaries: fu_B*  or  fu_D*
    g0, g1, g2,                       # sources: Ex,Ey,Ez  or  Hx,Hy,Hz
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, one Yee sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,
    beta_plus, beta_minus,            # f32(sign*2*pi*beta*dt), host-rounded once
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    HAS_BETA: tl.constexpr,           # compiled-in only for beta != 0 runs
    BLOCK: tl.constexpr,
):
    """One REAL curl sub-step of all three components with MEEP's special-kz beta pass.

    ``kernels.pml_curl_step``'s body, character for character, plus ONE constexpr-gated
    insert after the three curl forms and before the stores: the beta pass
    (``stepping._special_kz_beta_term`` then ``stepping._apply_folded_increment``),
    which MEEP runs as a pass of its own after every component's curl
    (step_db.cpp:161-175) and which reaches only targets 0 and 1. Real storage is
    MEEP's implicit-i trick: the same signed word on both sub-steps.

    The pass follows the forms in register. Its increment is ``-(c * g)`` on the CENTER
    partner the curl already loaded, with ``c`` the host-rounded word of its sign, an
    exact ``+0`` on the cells the curl's ownership mask drops, and the beta ladder
    (:func:`.forms.fold_ladder`) subtracts it: ``f - (sinv * q)`` outside the
    component's dsigu chunk, ``fu - (sinv * q)`` and ``f - (sinv_u * (sinv * q))``
    inside it. ``HAS_BETA = 0`` builds ``kernels.pml_curl_step`` exactly (the gate's
    identity arm).
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall, which `tl.load`'s
    # `other=` delivers without dereferencing anything.
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
    # One call per target: MEEP's left-to-right association with both partners, the
    # one-partner form where the record says the other partner's derivative axis is
    # invariant, and an exact +0 with neither. The presence word is the record's
    # (stepping.curl_presence, MEEP's have_plus_deriv / have_minus_deriv).
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
    curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
    curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    # Cell 0 of a metallic axis, for every target whose Yee shift there is 0. The
    # B targets have a single zero shift each (Bx:x, By:y, Bz:z); the D targets
    # have two (Dx:y,z  Dy:x,z  Dz:x,y). Masked, then the recurrence runs on zero.
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
    # dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    # sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    # Which cells take the recurrence: the component's dsigu layer, on the component's
    # own sub-lattice. The dsigu axis is z for target 0, x for 1 and y for 2 (vec.hpp
    # cycle_direction); the sub-lattice is half-integer for every B target and integer
    # for every D one (the iyee table), so each target's record line is a compile-time
    # word index, chunk_map.record_index(half_integer, axis, 0).
    if BACKWARD:
        lattice = 0
    else:
        lattice = RECORD_HALF_INTEGER
    du0 = record_in(cb, lattice + 8, k)
    du1 = record_in(cb, lattice + 0, i)
    du2 = record_in(cb, lattice + 4, j)

    p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)
    v0, n0 = curl_form(tl.load(f0 + idx, mask=live, other=0.0), p0, curl0,
                       km_y, si_y, km_z, si_z, du0)

    p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)
    v1, n1 = curl_form(tl.load(f1 + idx, mask=live, other=0.0), p1, curl1,
                       km_z, si_z, km_x, si_x, du1)

    p2 = tl.load(u2 + idx, mask=live & du2, other=0.0)
    v2, n2 = curl_form(tl.load(f2 + idx, mask=live, other=0.0), p2, curl2,
                       km_x, si_x, km_y, si_y, du2)

    # --- the special-kz beta pass (stepping._special_kz_beta_term, then
    # stepping._apply_folded_increment) ----------------------------------------
    # MEEP's pass of its own after every component's curl (step_db.cpp:161-175),
    # applied to the registers the forms returned. The partners are the CENTER
    # loads: target 0 takes the second source at sign +1 (Bx <- Ey, Dx <- Hy),
    # target 1 the first at sign -1 (By <- Ex, Dy <- Hx); target 2 gets nothing.
    # The increment is the NEGATED product (no dtdx: an analytic derivative),
    # masked to +0 on the cells the curl's own mask drops, and the beta ladder
    # subtracts it (forms.fold_ladder).
    if HAS_BETA:
        q0 = (beta_plus * b) * -1.0
        q1 = (beta_minus * a) * -1.0
        if BACKWARD:
            if BCY == METALLIC:
                q0 = tl.where(at_y, 0.0, q0)
            if BCZ == METALLIC:
                q0 = tl.where(at_z, 0.0, q0)
            if BCX == METALLIC:
                q1 = tl.where(at_x, 0.0, q1)
            if BCZ == METALLIC:
                q1 = tl.where(at_z, 0.0, q1)
        else:
            if BCX == METALLIC:
                q0 = tl.where(at_x, 0.0, q0)
            if BCY == METALLIC:
                q1 = tl.where(at_y, 0.0, q1)
        v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)
        v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)

    # fu is written on the recurrence's cells ONLY: outside them the dsig-only form
    # reads and writes the field alone and fu keeps its word (the auxiliary contract).
    tl.store(u0 + idx, n0, mask=live & du0)
    tl.store(u1 + idx, n1, mask=live & du1)
    tl.store(u2 + idx, n2, mask=live & du2)
    tl.store(f0 + idx, v0, mask=live)
    tl.store(f1 + idx, v1, mask=live)
    tl.store(f2 + idx, v2, mask=live)


@triton.jit
def beta_bloch_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz or Dx,Dy,Dz (complex64 as words)
    u0, u1, u2,                       # auxiliaries: fu_B* or fu_D* (complex64 as words)
    g0, g1, g2,                       # sources: Ex,Ey,Ez or Hx,Hy,Hz (complex64 as words)
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, float32, one Yee sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
    beta_plus, beta_minus,            # f32 quarter-turn words per sign (B: +c, D: -c)
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kind per axis
    HAS_BETA: tl.constexpr,           # compiled-in only for beta != 0 runs
    BLOCK: tl.constexpr,
):
    """One COMPLEX curl sub-step with MEEP's special-kz beta pass, each part on its own.

    ``complex_fields.bloch_pml_curl_step``'s body, character for character, plus ONE
    constexpr-gated insert after the six part forms and before the stores. The beta
    increment on complex storage is a QUARTER TURN of the partner's parts by one real
    word (``part_products.quarter_turn``): ``i*c*(x + iy) = (-(c*y), c*x)``, so the
    negated increment is ``(c*y, -(c*x))``, with ``c = +2*pi*beta*dt*sign`` on the B
    side and its negation on the D side (``beta_curl_coefficients``). No complex
    product is formed. The partners are the CENTER words as loaded: the Bloch
    connection replaces plane 0 of the curl's connected copies only, never these.
    Each part takes the beta ladder (:func:`.forms.fold_ladder`), masked to ``(+0,
    +0)`` on the cells the curl's ownership mask drops. ``HAS_BETA = 0`` builds
    ``complex_fields.bloch_pml_curl_step`` exactly.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    # PERIODIC wraps (a plain roll, phased or not); METALLIC serves an exact 0.0 past
    # the wall -- a (+0.0, +0.0) word pair from `other=` IS the complex metallic ghost.
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

    # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
    # Target 0 reads c along y and b along z, target 1 a along z and c along x,
    # target 2 b along x and a along y (stepping's term table). `tl.where` is a
    # bitwise select, so every cell off plane 0 keeps its loaded words.
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
    b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
    a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
    c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
    b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
    a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
    if CNX:
        r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
        c1_re = tl.where(at_x, r_re, c_re)
        c1_im = tl.where(at_x, r_im, c_im)
        r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
        b2_re = tl.where(at_x, r_re, b_re)
        b2_im = tl.where(at_x, r_im, b_im)
    if CNY:
        r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
        c0_re = tl.where(at_y, r_re, c_re)
        c0_im = tl.where(at_y, r_im, c_im)
        r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
        a2_re = tl.where(at_y, r_re, a_re)
        a2_im = tl.where(at_y, r_im, a_im)
    if CNZ:
        r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
        b0_re = tl.where(at_z, r_re, b_re)
        b0_im = tl.where(at_z, r_im, b_im)
        r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
        a1_re = tl.where(at_z, r_re, a_re)
        a1_im = tl.where(at_z, r_im, a_im)

    # --- the curl (stepping._curl_from_operands), per part ---------------------
    # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
    # association with both partners, the one-partner form where the record says the
    # other partner's derivative axis is invariant, an exact +0 with neither. The
    # dtdx scale is the real scalar times each part (part_products.scale).
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
    curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
    curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
    curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
    curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
    curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
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

    # --- the PML forms (stepping._apply_curl_regions), per part ----------------
    # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). One record
    # select per target serves both parts; fu is read and written on the
    # recurrence's cells only (the auxiliary contract), both words.
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    if BACKWARD:
        lattice = 0
    else:
        lattice = RECORD_HALF_INTEGER
    du0 = record_in(cb, lattice + 8, k)
    du1 = record_in(cb, lattice + 0, i)
    du2 = record_in(cb, lattice + 4, j)

    p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)
    p_im = tl.load(u0 + 2 * idx + 1, mask=live & du0, other=0.0)
    v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=live, other=0.0), p_re, curl0_re,
                             km_y, si_y, km_z, si_z, du0)
    v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl0_im, km_y, si_y, km_z, si_z, du0)

    p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
    p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
    v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=live, other=0.0), p_re, curl1_re,
                             km_z, si_z, km_x, si_x, du1)
    v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl1_im, km_z, si_z, km_x, si_x, du1)

    p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
    p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
    v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=live, other=0.0), p_re, curl2_re,
                             km_x, si_x, km_y, si_y, du2)
    v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=live, other=0.0), p_im,
                             curl2_im, km_x, si_x, km_y, si_y, du2)

    # --- the special-kz beta pass (stepping._special_kz_beta_term, then
    # stepping._apply_folded_increment), per part ------------------------------
    # MEEP's pass of its own after every component's curl (step_db.cpp:161-175),
    # applied to the registers the forms returned. The partners are the CENTER
    # words as loaded, never connected: target 0 takes the second source at sign
    # +1, target 1 the first at sign -1; target 2 gets nothing. The increment is
    # the negated quarter turn (part_products.quarter_turn): i*c*(x + iy) is
    # (-(c*y), c*x), so its negation is (c*y, -(c*x)), the two negations of the
    # real part cancelling exactly. The words beta_plus/beta_minus already carry
    # the side's i or -i (special_kz.beta_curl_coefficients). Masked to (+0, +0)
    # on the cells the curl's own mask drops; each part takes the beta ladder.
    if HAS_BETA:
        q0_re = beta_plus * b_im
        q0_im = (beta_plus * b_re) * -1.0
        q1_re = beta_minus * a_im
        q1_im = (beta_minus * a_re) * -1.0
        if BACKWARD:
            if BCY == METALLIC:
                q0_re = tl.where(at_y, 0.0, q0_re)
                q0_im = tl.where(at_y, 0.0, q0_im)
            if BCZ == METALLIC:
                q0_re = tl.where(at_z, 0.0, q0_re)
                q0_im = tl.where(at_z, 0.0, q0_im)
            if BCX == METALLIC:
                q1_re = tl.where(at_x, 0.0, q1_re)
                q1_im = tl.where(at_x, 0.0, q1_im)
            if BCZ == METALLIC:
                q1_re = tl.where(at_z, 0.0, q1_re)
                q1_im = tl.where(at_z, 0.0, q1_im)
        else:
            if BCX == METALLIC:
                q0_re = tl.where(at_x, 0.0, q0_re)
                q0_im = tl.where(at_x, 0.0, q0_im)
            if BCY == METALLIC:
                q1_re = tl.where(at_y, 0.0, q1_re)
                q1_im = tl.where(at_y, 0.0, q1_im)
        v0_re, n0_re = fold_ladder(v0_re, n0_re, q0_re, si_y, si_z, du0)
        v0_im, n0_im = fold_ladder(v0_im, n0_im, q0_im, si_y, si_z, du0)
        v1_re, n1_re = fold_ladder(v1_re, n1_re, q1_re, si_z, si_x, du1)
        v1_im, n1_im = fold_ladder(v1_im, n1_im, q1_im, si_z, si_x, du1)

    # --- stores: fu (recurrence cells only), then the fields, both parts --------
    tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
    tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
    tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
    tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
    tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
    tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


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


def _beta_complex_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every COMPLEX-family beta predicate shares."""
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

    return reasons


def beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """May the COMPLEX special_kz curl kernel step this (fields, pml, sub_step)?

    ``complex_fields.complex_pml_curl_coverage``'s clause set with clause 8's
    beta refusal INVERTED. Off-diagonal epsilon is ADMITTED here exactly
    as the certified complex curl admits it (constitutive-only under complex
    storage; the real-storage hazard is the REAL family's 12b clause).
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _beta_complex_grid_reasons(fields, pml, grid)

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


def beta_run_complex_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the CERTIFIED complex constitutive kernel step this side on a BETA run?

    ``complex_fields.complex_constitutive_coverage`` with the beta clause
    INVERTED. The kernel launched is the certified ``bloch_constitutive_step``
    unchanged (grouping choice 5 in the module docstring).
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _beta_complex_grid_reasons(fields, pml, grid)

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

    ``launch.PmlCurlPlan`` plus the two host-rounded beta words and the ``HAS_BETA``
    constexpr. Same two construction routes and one launch route:
    :func:`plan_beta_pml_curl` from the engine's objects through the predicate,
    :func:`plan_beta_pml_curl_from_arrays` from bare device arrays for the gate
    (including the deliberately wrong ones), both launched through :meth:`run`.
    ``chunk_bounds`` is the chunk record, required and bound as ``launch.PmlCurlPlan``
    binds it (``launch.bind_chunk_bounds``).
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "beta_plus",
                 "beta_minus", "has_beta", "backward", "bc", "block",
                 "num_warps", "chunk_bounds", "_targets", "_aux", "_sources",
                 "_coefficients", "_cb", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc,
                 beta_plus: float, beta_minus: float, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None, has_beta: int = 1, *,
                 chunk_bounds: Any) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "BetaPmlCurlPlan")
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
            *self._targets, *self._aux, *self._sources, *self._coefficients, self._cb,
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
                f"num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_beta_pml_curl(fields: Any, pml: Any, sub_step: str,
                       block: Optional[int] = None,
                       num_warps: Optional[int] = None,
                       chunk_layout: Any = None
                       ) -> Optional[BetaPmlCurlPlan]:
    """Build a REAL beta curl plan from the engine's objects, or None.

    None is the only refusal (Y-style). The words come from
    :func:`beta_curl_coefficients` -- the array path's own per-call-site arithmetic.
    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default
    (``launch.chunk_bounds_for``).
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
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_beta_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                   flat: Dict[str, Any], codes, dtdx: float,
                                   beta_plus: float, beta_minus: float,
                                   block: Optional[int] = None,
                                   kernel: Any = None,
                                   num_warps: Optional[int] = None,
                                   has_beta: int = 1, *,
                                   chunk_bounds: Any) -> BetaPmlCurlPlan:
    """Build a REAL beta curl plan from bare device arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones; ``kernel=`` carries the
    mutation override and ``has_beta=0`` the identity leg's root-kernel arm.
    ``chunk_bounds`` has NO default: a harness plan built without a record is refused
    by name (``launch.bind_chunk_bounds``), because a mutation leg that fell back to
    some default division would stop measuring the forms it exists to measure.
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
        chunk_bounds=chunk_bounds,
    )


class BetaBlochPmlCurlPlan:
    """A launchable, allocation-free COMPLEX beta PML curl sub-step.

    ``complex_fields.ComplexPmlCurlPlan`` plus the two quarter-turn words and
    ``HAS_BETA``. The complex volumes are bound as float32 WORD VIEWS at plan time;
    ``n_elem`` stays the COMPLEX cell count. ``connections`` /
    ``connection_values`` are ``complex_fields.connection_arguments``' kinds and
    float32 parts for this sub-step's direction; ``chunk_bounds`` is the chunk record,
    required.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "connections", "connection_values", "beta_words", "has_beta",
                 "block", "num_warps", "chunk_bounds", "_targets",
                 "_aux", "_sources", "_coefficients", "_cb", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, connections,
                 connection_values, beta_words, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None, has_beta: int = 1, *,
                 chunk_bounds: Any) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        from .complex_fields import CONNECTION_KINDS  # noqa: PLC0415
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.connections = tuple(int(kind) for kind in connections)
        if len(self.connections) != 3 or any(
                kind not in CONNECTION_KINDS.values() for kind in self.connections):
            raise ValueError(f"connections must be three kinds in "
                             f"{sorted(CONNECTION_KINDS.values())}, got {connections!r}")
        for axis, (kind, code) in enumerate(zip(self.connections, self.bc)):
            # A connection exists only on a wrapping axis, as in ComplexPmlCurlPlan.
            if kind and code != getattr(PERIODIC, "value", PERIODIC):
                raise ValueError(f"axis {axis} carries a connection but its boundary "
                                 f"code {code} does not wrap")
        self.connection_values = tuple(float(value) for value in connection_values)
        # (plus, minus): one float32-rounded real word per sign, the side's i or -i
        # already applied (beta_curl_coefficients); float() keeps every bit.
        words = tuple(beta_words)
        if len(words) != 2 or any(isinstance(word, (tuple, list)) for word in words):
            raise ValueError(f"beta_words must be the two real quarter-turn words of "
                             f"beta_curl_coefficients(..., complex_storage=True), got "
                             f"{beta_words!r}")
        self.beta_words = tuple(float(word) for word in words)
        self.has_beta = int(has_beta)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "BetaBlochPmlCurlPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as the others."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = (self._kernel if self._kernel is not None
                  else beta_bloch_pml_curl_step)
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.connection_values,
            *self.beta_words,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            CNX=self.connections[0], CNY=self.connections[1], CNZ=self.connections[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"BetaBlochPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, connections={self.connections}, "
                f"beta_words={self.beta_words!r}, has_beta={self.has_beta}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             num_warps: Optional[int] = None,
                             chunk_layout: Any = None
                             ) -> Optional[BetaBlochPmlCurlPlan]:
    """Build a COMPLEX beta curl plan from the engine's objects, or None.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not beta_bloch_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    phases = bloch_phase_table(grid, kinds)
    connections, values = connection_arguments(phases, backward=bool(spec["backward"]))
    beta_words = beta_curl_coefficients(grid.beta, grid.dt,
                                        magnetic=(sub_step == "step_B"),
                                        complex_storage=True)
    return BetaBlochPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        connections, values, beta_words,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_beta_bloch_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                         flat: Dict[str, Any], codes,
                                         phases: Sequence[Optional[complex]],
                                         dtdx: float,
                                         beta_words,
                                         block: Optional[int] = None,
                                         kernel: Any = None,
                                         num_warps: Optional[int] = None,
                                         has_beta: int = 1, *,
                                         chunk_bounds: Any
                                         ) -> BetaBlochPmlCurlPlan:
    """Build a COMPLEX beta curl plan from bare device arrays — the gate's route.

    ``beta_words`` is ``(plus, minus)`` — normally :func:`beta_curl_coefficients`'s
    output, or a deliberately wrong pair on a mutation leg. The direction's connection
    factor is formed HERE, per sub-step, exactly as the engine route forms it
    (``complex_fields.connection_arguments``). ``chunk_bounds`` has no default (see
    :func:`plan_beta_pml_curl_from_arrays`).
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(spec["backward"]))
    return BetaBlochPmlCurlPlan(
        sub_step, shape, dtdx, codes, connections, values, beta_words,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
        chunk_bounds=chunk_bounds,
    )


def plan_beta_run_constitutive(fields: Any, pml: Any, side: str,
                               block: Optional[int] = None,
                               num_warps: Optional[int] = None,
                               chunk_layout: Any = None
                               ) -> Optional[ConstitutivePlan]:
    """A root real constitutive plan for a beta run, or None.

    The arithmetic and the plan class are ``launch.ConstitutivePlan``'s, untouched; only
    the ADMISSION is this module's (:func:`beta_run_constitutive_coverage`, the root
    predicate minus the beta clause). The binding restates ``launch.plan_constitutive``'s:
    the run's chunk record (``chunk_layout``, ``None`` for the one-rank default), and on
    the H side MEEP's start of H from ``fields``.
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
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
    )


def plan_beta_run_complex_constitutive(fields: Any, pml: Any, side: str,
                                       block: Optional[int] = None,
                                       num_warps: Optional[int] = None,
                                       chunk_layout: Any = None
                                       ) -> Optional[ComplexConstitutivePlan]:
    """A root complex constitutive plan for a beta run, or None.

    The kernel and plan class are ``complex_fields``', untouched (module docstring,
    grouping choice 5). The run's chunk record and, on the H side, MEEP's start of H
    from ``fields``, as ``complex_fields.plan_complex_constitutive`` binds them.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not beta_run_complex_constitutive_coverage(fields, pml, side).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ComplexConstitutivePlan(
        side, fields.grid.shape,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        h_state=fields,
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
