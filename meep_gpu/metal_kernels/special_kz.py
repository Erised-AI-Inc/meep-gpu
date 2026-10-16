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

WHAT BETA IS (grid.py ``_resolve_beta``; ``stepping._special_kz_beta_term``). A 2-D
run carries ``exp(i*2*pi*beta*z)`` analytically, so ``d/dz`` on the invariant axis is
the EXACT factor ``i*2*pi*beta`` — an ``i*beta*zhat x`` cross product (MEEP
step_db.cpp:148-176). MEEP adds it as A PASS OF ITS OWN after every component's curl
(step_db.cpp:161-175), through the additive ladder the pass ends in
(step_generic.cpp:255-333), and the array path does the same
(``stepping.step_B``/``step_D`` call ``_apply_folded_increment`` after the component
loop):

* targets: Bx adds the Ey-centre term at sign +1, By the Ex-centre term at sign -1,
  Bz nothing; Dx <- Hy at +1, Dy <- Hx at -1, Dz nothing;
* coefficient: ``sign * 2*pi * grid.beta * grid.dt`` — **NO dtdx**, this is an
  analytic derivative and not a finite difference — rounded ONCE to the storage's
  realnum on the host (:func:`beta_curl_coefficients`). Real storage adds
  ``c * g`` on both sides (MEEP's implicit-i trick, step_db.cpp:148-160); complex
  storage takes a QUARTER TURN of the partner's parts by ``c`` on the B side and
  ``-c`` on the D side (``part_products.quarter_turn``: two real products and an
  exact negation, no complex product);
* the ladder: the increment, negated (the curl sign convention) and masked to ``+0``
  on MEEP's unowned cells, is subtracted through ``f -= sinv*inc`` outside the
  component's dsigu chunks and ``fu -= (df = sinv*inc); f -= sinv_u*df`` inside
  them (:func:`.folded_pass.fold_ladder`), keyed on the curl's own region plan;
* partners: the SAME-CELL component snapshots. Every beta partner is a CENTRE
  operand the certified curl already loads — Bx: the second source Ey (``b``); By:
  the first source Ex (``a``); Dx: Hy; Dy: Hx — so the pass costs no new pointer and
  no extra traffic, and the load reused must be the UNSHIFTED stored centre, never a
  shifted or Bloch-connected copy;
* no new state array, no ghost rule, no fold parity, no wrap. ``update_H`` /
  ``update_E`` read nothing beta-dependent.

TWO KERNELS, one per storage family, each DERIVED from a certified curl's own
emitted text by the three checked edits of :func:`.folded_pass.with_folded_pass`:

* ``beta_pml_curl_step`` — :func:`.shaders.curl_source` plus the real pass;
* ``beta_bloch_pml_curl_step`` — :func:`.complex_fields.bloch_curl_source` (MEEP's
  Bloch connection, part-wise products) plus the complex pass.

The derivation means the chunk record and its per-chunk forms, MEEP's stencil and
its absent-partner forms, the ownership mask and the Bloch connection are the
certified emitters' text, and the only text this module can be blamed for is the
pass (:func:`beta_pass`).

THE METAL FACTS THIS FAMILY APPLIES, each measured on this host:

1. **Contraction is a SOURCE property here, not a launch keyword.** The pass's
   multiply-subtract shapes (``v - (sinv_u * df)``) are exactly what a compiler
   contracts into an fma; the file-scope contraction directive every emitted source
   carries keeps each operation rounded once, and the gate MEASURES the difference
   (the ``contract=fast`` variant must diverge) instead of asserting the directive.
2. **The scalar ABI is clean.** A Python double bound to ``constant float&`` arrives
   correctly rounded, so the host-rounded coefficient reaches the kernel as the array
   path's bits.
3. **Negation is a sign-bit operation.** ``-(x)`` preserves ``-0`` and subnormal bits
   here (``templates`` module docstring: ``0.0f - x`` misses 36/512 on the
   exhaustive signed-zero table, ``-x`` is exact), which the negated increment and the
   quarter turn rely on; the refuted spelling is a must-catch gate mutation.
4. **There is one complex helper body.** The complex arm emits the part-wise helpers
   (``templates.complex_helpers``), and the pass forms no complex product.

CHOICES the array path does not force (the gate holds every one):

1. TWO host-rounded words per launch — one per sign — rather than one negated
   in-kernel. ``sign * 2*pi*beta*dt`` is formed per call site and rounded once; f64
   negation and f32 rounding commute exactly, but binding both keeps the
   transcription literal per call site.
2. The increment is formed NEGATED and SUBTRACTED, as the array path forms and
   applies it: ``x - (+0)`` is ``x`` for every ``x``, so an unowned cell keeps its
   word, ``-0`` included, where adding a masked ``+0`` would turn a stored ``-0`` into
   ``+0``.
3. ``HAS_BETA`` is a source specialisation: the pass is compiled in only when beta is
   nonzero. Engine-route plans always build 1; the 0 arm exists so the gate's identity
   leg can pin a beta = 0 build against the CERTIFIED plain curl on the same seeds.
4. Constitutive sub-steps on a beta run are NOT a new kernel, in EITHER storage: the
   restated predicates :func:`beta_run_constitutive_coverage` and
   :func:`beta_run_complex_constitutive_coverage` drop only the beta clause and
   delegate the arithmetic to the certified ``launch.ConstitutivePlan`` and
   ``complex_fields.ComplexConstitutivePlan`` respectively.
5. The fold is refused in BOTH storage families, by name (:mod:`.folded_beta` is
   the product that composes them).
6. A conductivity on the curl targets is refused by name: these kernels transcribe
   the lossless forms only.
7. BOTH LAYOUT CLAUSES FAIL CLOSED, on both arms (:func:`_complex_layout_reasons`,
   and ``_layout_reasons`` on the real arm).
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
from . import arms, folded_pass, templates
from . import chunk_bounds as _chunk_bounds
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import compile_source
from .launch import SUB_STEPS
from .plans import KernelPlan

__all__ = [
    "BetaBlochPmlCurlPlan",
    "BetaPmlCurlPlan",
    "beta_bloch_curl_source",
    "beta_bloch_pml_curl_coverage",
    "beta_curl_coefficients",
    "beta_curl_source",
    "beta_parameters",
    "beta_pass",
    "beta_pml_curl_coverage",
    "beta_run_complex_constitutive_coverage",
    "beta_run_constitutive_coverage",
    "enumerate_sources",
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


# ---------------------------------------------------------------------------
# The host coefficient — stepping._special_kz_beta_term, transcribed
# ---------------------------------------------------------------------------

def _numpy():
    """numpy, imported where it is used (this module is importable without it in
    the predicate-only role every Metal coverage module keeps)."""
    import numpy  # noqa: PLC0415

    return numpy


def beta_curl_coefficients(beta: float, dt: float, magnetic: bool,
                           complex_storage: bool) -> Tuple[float, float]:
    """The (plus-sign, minus-sign) beta words, rounded EXACTLY as the array path rounds.

    Per sign, ``coefficient = sign * 2*pi * beta * dt`` in the caller's types
    (``stepping._special_kz_beta_term``). Real storage multiplies the partner by
    ``dtype.type(coefficient)`` on both sides — MEEP's implicit-i trick
    (step_db.cpp:148-160) makes the B and D sides the same real add. Complex storage
    forms the term as a QUARTER TURN of the parts by the realnum ``coefficient`` on
    the B side and ``-coefficient`` on the D side (``part_products.quarter_turn``,
    one product per part and an exact negation), so the word bound is that real
    coefficient rounded once — never a complex pair: no complex product is formed and
    no signed zero is synthesised.

    ``float()`` of a numpy.float32 preserves the bits, and a Python float bound to
    ``constant float&`` arrives correctly rounded (the Metal scalar ABI is clean;
    measured), so the words the kernel receives are the array path's.

    ``beta`` AND ``dt`` ARE USED AS GIVEN — no ``float()`` normalisation. The
    expression is the array path's, evaluated on the objects the array path evaluates
    it on, so its intermediate PRECISION is whatever the caller's types make it. Under
    NEP 50 a numpy-typed operand rounds the chain EARLIER than a Python float does.
    MEASURED 2026-08-15 on this host: with ``beta``/``dt`` as ``numpy.float32``,
    14,926 of 40,000 random draws produced a different float32 coefficient word. The
    path is live — ``Grid._resolve_beta`` coerces ``beta`` to a Python float but
    NOTHING coerces ``dt``. Pinned by
    ``test_metal_special_kz.test_the_coefficient_matches_stepping_on_numpy_typed_grid_scalars``.
    """
    import numpy  # noqa: PLC0415

    out: List[float] = []
    for sign in (1.0, -1.0):
        coefficient: Any = sign * 2.0 * math.pi * beta * dt  # the array path's, verbatim
        if complex_storage and not magnetic:
            coefficient = -coefficient  # quarter_turn(-coefficient, g) on the D side
        out.append(float(numpy.float32(coefficient)))  # dtype.type / _realnum — once
    return out[0], out[1]


# ---------------------------------------------------------------------------
# The beta curl sources, DERIVED from the certified curls
# ---------------------------------------------------------------------------
#
# Each source is a certified curl's own emitted text plus the beta pass, by the
# three checked edits of :func:`.folded_pass.with_folded_pass`: the entry point is
# renamed, the two beta words bind in front of the chunk record, and the pass runs
# after the per-chunk forms and before the stores. Nothing above the forms is this
# module's text: the record binding and its bounds block, the ghost gather, MEEP's
# left-to-right stencil with its absent-partner forms, the ownership mask and (on the
# complex arm) MEEP's Bloch connection are the certified emitters' own output.

#: Where the chunk record binds on each arm: the certified curl's own record slot
#: (``shaders.CURL_RECORD_BUFFER`` 20, ``complex_fields.CURL_RECORD_BUFFER`` 23) moves
#: two places up, and the two beta words take the slots it leaves.
REAL_RECORD_BUFFER = 22
COMPLEX_RECORD_BUFFER = 25


def beta_parameters(first: int, pad: str) -> str:
    """The two beta words' parameter lines, at ``first`` and ``first + 1``."""
    return (f"    constant float&{pad}beta_plus  [[buffer({first})]],\n"
            f"    constant float&{pad}beta_minus [[buffer({first + 1})]],\n")

#: Which targets the pass reaches, with the partner each reads: target 0 the second
#: source's UNSHIFTED centre ``b`` at sign +1, target 1 the first source's centre
#: ``a`` at sign -1, target 2 nothing (step_db.cpp:161-175 runs ``cc`` over d_c in
#: {X, Y} only). The centres are the stored words, never a Bloch-connected copy:
#: the array path reads its component snapshot (``_special_kz_beta_term``).
BETA_TARGETS: Tuple[Tuple[int, str, str], ...] = ((0, "beta_plus", "b"),
                                                   (1, "beta_minus", "a"))

#: The pass's text on a ``HAS_BETA = 0`` build: the slot exists and is empty.
NO_BETA_PASS = "    // HAS_BETA = 0: beta = 0 runs belong to the certified curl"


def beta_pass(rows: Sequence[Tuple[int, str, str]], complex_storage: bool,
              has_beta: bool = True) -> str:
    """The beta pass on the targets it reaches: the increment, its mask, the ladder.

    The increment is ``_special_kz_beta_term``'s, negated (the curl sign convention):
    real storage ``-(c * g)``; complex storage the quarter turn
    ``(-(c * g.im), c * g.re)`` negated per part. ``rows`` are the curl source's own
    ownership-mask rows (:func:`.folded_pass.ownership_rows`): MEEP's pass loops over
    owned cells only, so an unowned cell takes ``+0`` and the ladder's subtractions
    leave it as it was. The ladder is :func:`.folded_pass.fold_ladder`.
    """
    if not has_beta:
        return NO_BETA_PASS
    targets = tuple(target for target, _word, _partner in BETA_TARGETS)
    lines = []
    for target, word, partner in BETA_TARGETS:
        if complex_storage:
            lines += [f"    float2 bq{target} = float2(-({word} * {partner}.y), "
                      f"{word} * {partner}.x);  // part_products.quarter_turn",
                      f"    float2 bi{target} = -bq{target};"]
        else:
            lines.append(f"    float bi{target} = -({word} * {partner});")
    lines.append(folded_pass.masked_rows(rows, "bi", targets))
    lines.append(folded_pass.fold_ladder(targets, "bi", "bd", complex_storage))
    return folded_pass.folded_pass_block(
        "the special-kz beta pass (stepping._special_kz_beta_term, step_db.cpp:161-175):"
        " negated, masked to +0 where the curl is masked", "\n".join(lines))


def beta_curl_source(codes: Sequence[int], backward: bool, has_beta: bool = True,
                     contract: str = templates.CONTRACT_OFF) -> str:
    """The specialised REAL ``beta_pml_curl_step`` source.

    :func:`.shaders.curl_source`'s own output with the beta pass
    (:func:`beta_pass`) inserted after its per-chunk forms. ``codes`` is the per-axis
    PERIODIC/METALLIC triple ``_boundary_kinds`` resolves; ``backward`` selects
    step_D's negated strides over step_B's forward ones; ``has_beta`` compiles the
    pass in or out (the 0 arm is the certified curl under this entry point's name and
    binding layout, which the gate's identity leg pins).
    """
    from .shaders import CURL_RECORD_BUFFER, curl_source  # noqa: PLC0415

    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if REAL_RECORD_BUFFER != CURL_RECORD_BUFFER + 2:  # pragma: no cover - layout pin
        raise AssertionError("the real beta curl's record slot no longer follows "
                             "the certified curl's by the two beta words")
    source = curl_source(codes, backward, contract)
    return folded_pass.with_folded_pass(
        source, ("pml_curl_step", "beta_pml_curl_step"),
        beta_parameters(CURL_RECORD_BUFFER, "     "),
        (CURL_RECORD_BUFFER, REAL_RECORD_BUFFER),
        beta_pass(folded_pass.ownership_rows(source), False, has_beta),
        "special_kz.beta_curl_source")


def beta_bloch_curl_source(codes: Sequence[int], backward: bool,
                           phased: Sequence[int],
                           has_beta: bool = True,
                           contract: str = templates.CONTRACT_OFF) -> str:
    """The specialised COMPLEX ``beta_bloch_pml_curl_step`` source.

    :func:`.complex_fields.bloch_curl_source`'s own output — MEEP's Bloch connection
    of the unshifted operand on plane 0, part-wise products — with the beta pass
    inserted after its per-chunk forms. ``phased`` is the per-axis connection kind
    (``complex_fields.CONNECT_KINDS``, :func:`.complex_fields.phase_arguments`).
    """
    from .complex_fields import CURL_RECORD_BUFFER, bloch_curl_source  # noqa: PLC0415

    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3 or len(phased) != 3:
        raise ValueError(f"codes and phased must be per-axis triples, got "
                         f"{codes!r} / {phased!r}")
    if COMPLEX_RECORD_BUFFER != CURL_RECORD_BUFFER + 2:  # pragma: no cover - layout pin
        raise AssertionError("the complex beta curl's record slot no longer follows "
                             "the certified complex curl's by the two beta words")
    source = bloch_curl_source(codes, backward, phased, contract)
    return folded_pass.with_folded_pass(
        source, ("bloch_pml_curl_step", "beta_bloch_pml_curl_step"),
        beta_parameters(CURL_RECORD_BUFFER, "      "),
        (CURL_RECORD_BUFFER, COMPLEX_RECORD_BUFFER),
        beta_pass(folded_pass.ownership_rows(source), True, has_beta),
        "special_kz.beta_bloch_curl_source")


def _emittable_bloch_specialisations() -> List[Tuple[Tuple[int, int, int],
                                                     Tuple[int, int, int]]]:
    """Every ``(codes, connection kinds)`` pair the COMPLEX plan builder can produce.

    Derived from the structural rules the shipped path enforces rather than from a
    hand-written list:

    * a connection rides only a PERIODIC axis (``complex_fields.bloch_phase_table``
      raises otherwise, and :func:`_beta_complex_grid_reasons` refuses a phase
      anywhere else), and on a periodic axis it is any of the three kinds
      (``complex_fields.connection``: a copy, a negation at the Brillouin edge, or the
      unfused product);
    * the INVARIANT axis carries no phase. Clause 5b refuses ``k_z != 0`` (and the
      ``Grid`` refuses it at construction), so the z kind is always NONE.

    Boundary codes run over all eight triples, matching the real arm. ``cz =
    METALLIC`` is not constructible — ``Grid`` refuses a metallic invariant axis — so
    those rows are a deliberate SUPERSET, which is the safe direction for a weld.
    """
    from .complex_fields import CONNECT_KINDS, CONNECT_NONE  # noqa: PLC0415

    out: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    for cx in (PERIODIC, METALLIC):
        for cy in (PERIODIC, METALLIC):
            for cz in (PERIODIC, METALLIC):
                for px in (CONNECT_KINDS if cx == PERIODIC else (CONNECT_NONE,)):
                    for py in (CONNECT_KINDS if cy == PERIODIC else (CONNECT_NONE,)):
                        out.append(((cx, cy, cz), (px, py, CONNECT_NONE)))
    return out


def enumerate_sources(contract: str = templates.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation the shipped plans can emit, keyed by a stable label.

    The gate fingerprints these exactly as the certified pair is fingerprinted, so a
    compiler-visible edit to any one of them is a fingerprint event rather than a
    silent recompile. Both ``HAS_BETA`` arms are enumerated because the 0 arm is a
    SHIPPED kernel — the identity leg launches it. The complex set is DERIVED
    (:func:`_emittable_bloch_specialisations`), and ``test_metal_special_kz`` asserts
    that every source the family's own case matrix launches is in it.
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
            for has_beta in (True, False):
                label = (f"beta_bloch_pml_curl_step/{name}/"
                         f"{''.join(str(code) for code in codes)}/"
                         f"ph{''.join(str(flag) for flag in phased)}/"
                         f"beta{int(has_beta)}")
                out[label] = beta_bloch_curl_source(
                    codes, backward, phased, has_beta, contract)
    return out


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
      which this module refuses BY NAME in two other clauses (the conductivity
      table, the Bloch phase table) and now refuses here.
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


def _beta_complex_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
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
    return reasons


def beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                 residency: Any = None) -> Coverage:
    """May the COMPLEX Metal beta curl step this (fields, pml, sub_step)?"""
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _beta_complex_grid_reasons(fields, pml, grid)
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
                                           residency: Any = None) -> Coverage:
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
    reasons = _beta_complex_grid_reasons(fields, pml, grid)
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

    ``launch.PmlCurlPlan``'s binding layout plus the two host-rounded beta words,
    bound in front of the chunk record (:data:`REAL_RECORD_BUFFER`). The record is
    checked against the plan's shape (:func:`.chunk_bounds.plan_words`) and uploaded
    once as a plan-owned ``int32`` vector. ``run`` and the contraction-variant
    contract are the base's.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "beta_plus", "beta_minus", "has_beta", "residency", "volumes",
                 "chunk_bounds")

    family = "special_kz real beta curl"

    REPR_FIELDS = ("sub_step", "shape", "bc", "beta_plus", "beta_minus",
                   "has_beta")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, residency,
                 beta_plus: float, beta_minus: float, targets, auxiliaries,
                 sources, coefficients, functions: Dict[str, Any],
                 volumes: Sequence[str], has_beta: bool = True, *,
                 chunk_bounds: Any) -> None:
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
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, f"BetaPmlCurlPlan({sub_step})")
        # Built ONCE, in the shader's exact binding order: 9 volumes, 6 coefficient
        # vectors, 4 extents, dtdx, the two host-rounded beta words and the record =
        # 23 arguments. The launch path unpacks this tuple and does nothing else.
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem,
               self.dtdx, self.beta_plus, self.beta_minus,
               _chunk_bounds.tensor(self.chunk_bounds, _plan_device(residency))))


class BetaBlochPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free COMPLEX beta PML curl sub-step.

    ``complex_fields.ComplexPmlCurlPlan``'s binding layout — the complex volumes as
    ``complex64`` MIRRORS reaching the kernel as ``device float2*``, the three
    connection words as ``float2`` — plus the two host-rounded beta words in front of
    the chunk record (:data:`COMPLEX_RECORD_BUFFER`). ``n_elem`` is the COMPLEX cell
    count, which is also the dispatch size the first tensor argument gives.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "phased", "phase_values", "beta_plus", "beta_minus", "has_beta", "residency",
                 "volumes", "chunk_bounds")

    family = "special_kz complex beta curl"

    REPR_FIELDS = ("sub_step", "shape", "bc", "phased", "has_beta")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, phased,
                 phase_values, beta_plus: float, beta_minus: float,
                 residency, targets, auxiliaries, sources, coefficients,
                 functions: Dict[str, Any], volumes: Sequence[str],
                 has_beta: bool = True, *, chunk_bounds: Any) -> None:
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
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.has_beta = bool(has_beta)
        self.residency = residency
        self.volumes = tuple(volumes)
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, f"BetaBlochPmlCurlPlan({sub_step})")
        # 9 float2 volumes, 6 coefficient vectors, 4 extents, dtdx, 3 connection
        # words, 2 beta words and the record = 26 bindings, under Metal's 31-buffer
        # ceiling (device.MAX_BUFFER_BINDINGS).
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem,
               self.dtdx) + self.phase_values
            + (self.beta_plus, self.beta_minus,
               _chunk_bounds.tensor(self.chunk_bounds, _plan_device(residency))))


def _plan_device(residency: Any) -> str:
    """The device the plan's residency DECLARES, which the record tensor joins."""
    return str(getattr(residency, "device", "mps"))


def _beta_curl_functions(codes, backward: bool, has_beta: bool,
                         contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_source(
        beta_curl_source(codes, backward, has_beta, mode)).beta_pml_curl_step
        for mode in contract_variants}


def _beta_bloch_functions(codes, backward: bool, phased,
                          has_beta: bool,
                          contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_source(
        beta_bloch_curl_source(codes, backward, phased, has_beta, mode)
    ).beta_bloch_pml_curl_step for mode in contract_variants}


def plan_beta_pml_curl(fields: Any, pml: Any, sub_step: str, residency: Any = None,
                       contract_variants: Sequence[str] = (templates.CONTRACT_OFF,),
                       *, chunk_bounds: Any) -> Optional[BetaPmlCurlPlan]:
    """Build a REAL beta curl plan from the engine's objects, or None when refused.

    ``chunk_bounds`` is the record of the division the run steps with (the
    composer's ``StepContext.chunk_bounds``) and has no default, as
    :func:`.launch.plan_pml_curl` has none.
    """
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
        volumes, chunk_bounds=chunk_bounds)


def plan_beta_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                   flat: Dict[str, Any], codes, dtdx: float,
                                   beta_plus: float, beta_minus: float,
                                   residency: Any,
                                   functions: Optional[Dict[str, Any]] = None,
                                   contract_variants: Sequence[str] = (
                                       templates.CONTRACT_OFF,),
                                   has_beta: bool = True, *,
                                   chunk_bounds: Any) -> BetaPmlCurlPlan:
    """Build a REAL beta curl plan from bare host arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones. ``functions`` is the
    MUTATION SEAM — dropping it is not a slowdown, it is a silent DISARMING, and
    every mutation leg would then launch the shipped kernel and report the defect
    as uncaught. ``has_beta=False`` is the identity leg's arm. ``chunk_bounds`` is
    REQUIRED: the record of the division the harness's reference steps with.
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
        volumes, has_beta=has_beta, chunk_bounds=chunk_bounds)


def plan_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                             residency: Any = None,
                             contract_variants: Sequence[str] = (
                                 templates.CONTRACT_OFF,), *, chunk_bounds: Any,
                             ) -> Optional[BetaBlochPmlCurlPlan]:
    """Build a COMPLEX beta curl plan from the engine's objects, or None.

    The connection kinds and words are :func:`.complex_fields.phase_arguments` of
    :func:`.complex_fields.bloch_phase_table`, for the sub-step's side — the
    certified complex curl's own host decisions, so the lifted connection and the
    words it reads come from one place.
    """
    from .complex_fields import (  # noqa: PLC0415 - circular import at module load
        _complex_mirror, bloch_phase_table, phase_arguments,
    )

    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not beta_bloch_pml_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    codes = [1 if kind == "metallic" else 0 for kind in kinds]
    phased, phase_values = phase_arguments(bloch_phase_table(grid, kinds),
                                           backward=bool(spec["backward"]))
    plus, minus = beta_curl_coefficients(grid.beta, grid.dt,
                                         magnetic=(sub_step == "step_B"),
                                         complex_storage=True)
    targets = [_complex_mirror(residency, n, getattr(fields, n))
               for n in spec["targets"]]
    auxiliaries = [_complex_mirror(residency, "fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [_complex_mirror(residency, n, getattr(fields, n))
               for n in spec["sources"]]
    # The PML coefficient vectors stay REAL: the array path scales each part by the
    # realnum coefficient (part_products.scale), which the kernel spells per part.
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return BetaBlochPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, phased, phase_values,
        plus, minus, residency, targets, auxiliaries, sources,
        coefficients,
        _beta_bloch_functions(codes, spec["backward"], phased, True,
                              contract_variants),
        volumes, chunk_bounds=chunk_bounds)


def plan_beta_bloch_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                         flat: Dict[str, Any], codes,
                                         phases: Sequence[Optional[complex]],
                                         dtdx: float, beta_plus: float,
                                         beta_minus: float,
                                         residency: Any,
                                         functions: Optional[Dict[str, Any]] = None,
                                         contract_variants: Sequence[str] = (
                                             templates.CONTRACT_OFF,),
                                         has_beta: bool = True, *,
                                         chunk_bounds: Any,
                                         ) -> BetaBlochPmlCurlPlan:
    """Build a COMPLEX beta curl plan from bare host arrays — the gate's route.

    ``phases`` is the per-axis ``Optional[complex]`` table of ``eikna``; the side's
    connection kind and word are chosen HERE, per sub-step, exactly as the engine
    route chooses them (:func:`.complex_fields.phase_arguments`). ``beta_plus`` and
    ``beta_minus`` are normally :func:`beta_curl_coefficients`' words, or deliberately
    wrong ones on a mutation leg. ``chunk_bounds`` is REQUIRED.
    """
    from .complex_fields import _complex_mirror, phase_arguments  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    phased, phase_values = phase_arguments(tuple(phases),
                                           backward=bool(spec["backward"]))
    targets = [_complex_mirror(residency, n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [_complex_mirror(residency, "fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [_complex_mirror(residency, n, arrays[n]) for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return BetaBlochPmlCurlPlan(
        sub_step, shape, dtdx, codes, phased, phase_values, beta_plus, beta_minus, residency,
        targets, auxiliaries, sources, coefficients,
        functions if functions is not None
        else _beta_bloch_functions(codes, spec["backward"], phased,
                                   has_beta, contract_variants),
        volumes, has_beta=has_beta, chunk_bounds=chunk_bounds)


def plan_beta_run_constitutive(fields: Any, pml: Any, side: str,
                               residency: Any = None,
                               contract_variants: Sequence[str] = (
                                   templates.CONTRACT_OFF,),
                               *, chunk_bounds: Any) -> Any:
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
        _constitutive_functions(side, contract_variants), volumes,
        chunk_bounds=chunk_bounds)


def plan_beta_run_complex_constitutive(
        fields: Any, pml: Any, side: str, residency: Any = None,
        contract_variants: Sequence[str] = (templates.CONTRACT_OFF,), *, chunk_bounds: Any) -> Any:
    """A CERTIFIED COMPLEX constitutive plan for an UNFOLDED beta run, or None.

    ``complex_fields``' own body, its own plan class and its own compiled functions;
    only the ADMISSION is this module's, exactly as
    :func:`plan_beta_run_constitutive` borrows the real one. Building the plan
    THROUGH the certified builder is what makes "the same kernel steps a beta run"
    a fact rather than a claim: nothing here re-spells any arithmetic.
    """
    from . import complex_fields  # noqa: PLC0415 - circular import at module load

    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not beta_run_complex_constitutive_coverage(
            fields, pml, side, residency).covered:
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
        side, fields.grid.shape, residency, targets, auxiliaries,
        sources, inverse_epsilon, coefficients,
        complex_fields._constitutive_functions(side, contract_variants),
        volumes, chunk_bounds=chunk_bounds)


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
                              ctx.contract_variants, chunk_bounds=ctx.chunk_bounds)


def _complex_curl_coverage(ctx: Any, slot: str) -> Coverage:
    return beta_bloch_pml_curl_coverage(ctx.fields, ctx.pml, slot, ctx.residency)


def _complex_curl_plan(ctx: Any, slot: str) -> Any:
    return plan_beta_bloch_pml_curl(ctx.fields, ctx.pml, slot, ctx.residency,
                                    ctx.contract_variants,
                                    chunk_bounds=ctx.chunk_bounds)


def _real_constitutive_coverage(ctx: Any, slot: str) -> Coverage:
    return beta_run_constitutive_coverage(ctx.fields, ctx.pml,
                                          _CONSTITUTIVE_SLOT_SIDES[slot],
                                          ctx.residency)


def _real_constitutive_plan(ctx: Any, slot: str) -> Any:
    return plan_beta_run_constitutive(ctx.fields, ctx.pml,
                                      _CONSTITUTIVE_SLOT_SIDES[slot],
                                      ctx.residency, ctx.contract_variants,
                                      chunk_bounds=ctx.chunk_bounds)


def _complex_constitutive_coverage(ctx: Any, slot: str) -> Coverage:
    return beta_run_complex_constitutive_coverage(
        ctx.fields, ctx.pml, _CONSTITUTIVE_SLOT_SIDES[slot], ctx.residency)


def _complex_constitutive_plan(ctx: Any, slot: str) -> Any:
    return plan_beta_run_complex_constitutive(
        ctx.fields, ctx.pml, _CONSTITUTIVE_SLOT_SIDES[slot], ctx.residency,
        ctx.contract_variants,
        chunk_bounds=ctx.chunk_bounds)


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
