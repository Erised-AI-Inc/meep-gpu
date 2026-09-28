"""BFAST (broadband fixed-angle source technique) PML curl on Metal — real f32.

The Metal port of ``triton_kernels/bfast_curl.py``. Source of truth for the
ARITHMETIC is ``stepping.py``; source of truth for the STRUCTURE — the predicate
clauses, the plan shape, the refusals — is the Triton module, which is certified.

WHAT BFAST IS (stepping.py:826-931, grid.py:698-753). The time-sheared substitution
``t -> t - k.r/c``, under which a planewave at a FIXED incidence angle is
transversely uniform at EVERY frequency, so plain periodic boundaries are exact for
a whole band::

    dB/dt = -curl(E) + d/dt (k x E)
    dD/dt = +curl(H) - d/dt (k x H)

MEEP runs it as a second additive pass over the SAME shifted operands the curl
gathers (step_db.cpp:129-142). This engine folds it into the curl before
``_apply_curl``: step_B reads the flag at stepping.py:364 and folds at :392-396,
AFTER the beta terms (:384-391) and BEFORE ``_mask_non_owned_cells`` (:397) and
``_apply_curl`` (:374); step_D mirrors it (flag :422, fold :446-449, mask :450,
apply :455). The beta-then-bfast fold ORDER is byte-significant if both were
active, which is why clause 12 (beta refused) is KEPT below.

THE ARITHMETIC, per term (stepping.py:923-931), with ``operands = CurlOperands``
(``first`` = MEEP g1 = f_p, ``second`` = g2 = f_m; :1591-1604)::

    k1 = bfast[_bfast_axis(term.second)] if have_m else 0.0        (:884)
    k2 = bfast[_bfast_axis(term.first)]  if have_p else 0.0        (:885)
    if D side: k1, k2 = -k1, -k2   in host f64                     (:886-887)
    total   = k1_f32*(shifted_first + first)
              - k2_f32*(shifted_second + second)                   (:896-897)
    advance = total - dtype.type(2.0)*state                        (:901)
    _mask_non_owned_cells(advance, grid, term.iyee)                (:902)
    state  += advance   (IN PLACE)                                 (:903)
    return -advance     (curl sign convention)                     (:904)

It is the Tustin filter ``(1 - z^-1)/(1 + z^-1)``: NO dtdx and NO dt anywhere —
MEEP passes ``dtdx`` to ``step_bfast`` and the body never reads it (:836-837).
MARGINALLY STABLE on purpose: the homogeneous mode of ``F_n = -F_{n-1}`` is
``(-1)^n``, undamped forever (:841-847), so noise in ``f_bfast`` never decays.

THE k-INDEXING TRAP, and it is the reason this module IMPORTS its coefficients
rather than transcribing them a third time. ``_bfast_axis`` (:787-796) indexes
``bfast_scaled_k`` by the PARTNER component's OWN direction (MEEP's
``component_index``, vec.hpp:445), CROSS-assigned: k1 is indexed by the SECOND
partner but multiplies the FIRST's sum. stepping.py itself calls it "the single
easiest mistake in the whole pass", and it is silent — on a k along one axis it
merely moves the term to the wrong pair of components. ``bfast_curl_coefficients``,
``BFAST_TERMS`` and ``BFAST_STATE_NAMES`` are therefore imported from
``triton_kernels.bfast_curl``, exactly as ``metal_kernels.coverage`` imports every
leaf helper from ``triton_kernels.coverage``: one definition, already gate-certified
on the other track, not a second transcription that can drift. Those symbols are
pure Python over duck-typed objects and carry no Triton clause;
``test_metal_bfast.test_the_import_route_answers_what_steppings_own_call_site_answers``
rebuilds the expectation from ``stepping``'s OWN term tables and its own
``_bfast_axis`` — not from a hand-derived reading, which would pin the same reading
twice — over three k vectors, three invariance masks and both sides.

THE INVARIANT-AXIS GUARD, AND THE OVER-COVERAGE DEFECT IT FENCES. ``have_p``/
``have_m`` read ``Grid.is_invariant`` on the partner's DERIVATIVE axis (:882-883),
cross-gated. It is load-bearing because an invariant axis zeroes the curl's
DIFFERENCE automatically while the BFAST SUM of the same two samples is ``2*g``,
not zero (:862-871).

MEEP does TWO things when a flag is false and stepping.py does ONE. MEEP zeroes the
COEFFICIENT (step_db.cpp:130-133) *and* NULLS THE OPERAND ARRAY (step_db.cpp:62-63);
``step_bfast`` swaps a null g1 into the g2 slot carrying k1 := k2
(step_generic.cpp:342-346) and takes its single-operand branch, so EITHER false flag
makes MEEP's whole increment ``F_new = -F_prev`` with the other term dropped.
``stepping.py:911-912`` zeroes only the coefficient and keeps the other product
computed from the stored array (:923-924), so the two agree only while the surviving
k is itself zero. The Triton gate MEASURED the divergence by calling the predicate
with constructed objects — at ``bfast = (0.4, 0, 0.3)`` on a z-invariant 2-D grid the
guarded targets miss MEEP's ``-F_prev`` by max|diff| ~ 0.598 — and the divergent
class collapses to a single grid-level test: **a nonzero k component on a
DECLARED-invariant axis**. Clause 11a below REFUSES it by name, and
``test_metal_bfast`` re-measures the divergence on this host rather than inheriting
the number. The fix for ``stepping.py`` itself is upstream of this port and is NOT
made here; this kernel transcribes ``stepping.py``, which is what the byte gate
arbitrates.

WHAT IS METAL-SPECIFIC, and all that is:

* **The tail is a SOURCE SPECIALISATION, not a runtime branch.** Triton's
  ``HAS_BFAST`` is a ``tl.constexpr``; ``torch.mps.compile_shader`` takes a source
  string and nothing else, so the tail is substituted in or left out and each
  specialisation is a distinct string and a distinct memo key. The ``HAS_BFAST=0``
  build exists for one reason: the gate's identity leg pins it byte-identical to
  the certified ``shaders.pml_curl_step`` on the same seeds.
* **The contraction pragma is load-bearing for the tail too.** ``k1*(sum) -
  k2*(sum)`` is a multiply-subtract and would contract into an fma; the certified
  recurrence below it is the same shape. One spelling, imported from
  :mod:`.shaders` through :mod:`.templates`.
* **No unary minus appears anywhere in the tail.** The curl fold is spelled
  ``curl - adv``; IEEE-754 defines subtraction AS addition of the negation, so it
  carries the array path's ``curl + (-advance)`` on every input including signed
  zeros. Triton needed this because its unary minus lowers as ``0.0 - x``; on
  Metal ``-x`` was MEASURED to be a sign-bit operation (0/8045, preserving -0 and
  subnormal bits). The spelling is kept anyway because it is literal, not because
  the workaround transfers — that distinction is recorded rather than inherited.
* **Three IIR state buffers bind pointer-identically.** The driver's
  ``synchronize_magnetic_fields`` backs up and RESTORES ``f_bfast_B*`` around the
  magnetic half-step (driver.py:4126-4135); because the IIR is marginally stable a
  missed restore never decays. Under the residency layer "pointer-identical" means
  the mirror's ``host`` IS ``fields.f_bfast_*`` and a sync out writes it in place —
  which is a STRONGER requirement than on CuPy, and the gate holds it.

WIRED AS OF TRANCHE 2. This module registers two curl arms in :mod:`.arms` and
``plan_step`` composes them; ``fastpath.plan_fast_path`` still returns ``None`` on
every branch, which is the separate decision. No admitted-overlap ambiguity arises
on the two sub-steps this module plans — ``step_B`` and ``step_D`` — because every
other predicate registered on those slots refuses a BFAST run BY NAME
(``coverage._grid_reasons`` clause 11, ``complex_fields`` clause 8, ``special_kz``
clause 11) while clause 11 here is inverted. The composition sweep MEASURES that,
per registered arm, rather than assuming it.

THE EXCEPTION LIST IS NOT EMPTY, and an earlier revision of this paragraph said it
was. The Triton module records two shipped predicates that do NOT consult the shared
grid clauses and DO admit a BFAST run (``ade_update_p_coverage``,
``fused_ade_state_coverage``). Measured on this tree 2026-08-15, the Metal package has
its own: ``no_pml_constitutive.metal_null_constitutive_coverage`` is SHIPPED and
WIRED on ``update_H``/``update_E``, does not reach ``coverage._grid_reasons``, names
no BFAST clause, and ADMITS a BFAST run whenever the absorber is inactive. That is
not a defect and not an ambiguity: it is a DIFFERENT SLOT, and it is the correct
answer there — under an inactive absorber ``stepping.update_H`` returns at
:916-917 and ``update_E`` at :954-955 before reading an array, and the constitutive
sub-steps carry no BFAST state at all (the pass is on ``step_db`` only), which is the
same reasoning ``triton_kernels.bfast_curl.bfast_run_constitutive_coverage`` uses to
admit the constitutive pair on a BFAST run deliberately. It is recorded because the
claim "no shipped Metal predicate admits a BFAST run" was FALSE and unmeasured: the
probe's old check enumerated four predicates by hand and built an ACTIVE absorber in
every case, so it could not have seen this one from either direction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# ONE DEFINITION, IMPORTED. These are pure Python over duck-typed engine objects,
# carry no Triton clause, and are the certified transcription of the k-assignment
# trap described above. Re-deriving them here would be a third transcription of
# the same six cross-assigned indices.
#
# ``SUB_STEPS`` COMES FROM THE TRITON PACKAGE, NOT FROM ``.launch``, and the choice
# is deliberate rather than incidental: it is the SAME table
# ``triton_kernels.bfast_curl`` binds against, so the two ports cannot disagree
# about which targets, sources, direction or Yee suffix a sub-step has — and this
# module then has no import edge into the Metal composer at all, which is what
# lets an unwired family be built and gated without being able to perturb
# ``plan_step``.
#
# WHAT A TEST CAN AND CANNOT PIN HERE, stated because the first attempt got it
# wrong. Comparing this module's ``SUB_STEPS`` against the Triton table field by
# field is ``x == x`` — the import makes them ONE OBJECT, and no edit anywhere can
# make that compare fail. The identity is pinned as an identity, in one line. The
# compare that CAN fail is against ``metal_kernels.launch.SUB_STEPS``, a SEPARATE
# literal that the shipped Metal composer and the certified curl plan bind and that
# this family's identity leg compares across; a drifted ``suffix`` there is a
# half-cell-wrong absorber rather than a crash. ``test_metal_bfast`` holds both.
from ..triton_kernels.bfast_curl import (  # noqa: F401 - re-exported deliberately
    BFAST_STATE_NAMES,
    BFAST_TERMS,
    bfast_curl_coefficients,
)
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
from ..triton_kernels.launch import SUB_STEPS
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .plans import KernelPlan

#: The family name the arm table carries. One spelling, so a refusal message, a
#: registry row and an artifact column cannot drift apart.
FAMILY = "bfast_curl"

__all__ = [
    "BFAST_STATE_NAMES",
    "BFAST_TERMS",
    "FAMILY",
    "BfastPmlCurlPlan",
    "bfast_curl_coefficients",
    "bfast_curl_source",
    "bfast_pml_curl_coverage",
    "bfast_run_constitutive_coverage",
    "compile_bfast_curl",
    "enumerate_bfast_sources",
    "plan_bfast_pml_curl",
    "plan_bfast_pml_curl_from_arrays",
    "plan_bfast_run_constitutive",
    "register_arms",
]

PERIODIC = shaders.PERIODIC
METALLIC = shaders.METALLIC


# ---------------------------------------------------------------------------
# The shader
# ---------------------------------------------------------------------------

#: The certified curl body with THREE additions and no other change: the three
#: state pointers, the six host-rounded scalars, and the ``__BFAST_TAIL__`` slot
#: between the curl and the ownership mask. The recurrence below the tail is
#: character-for-character ``shaders._CURL_TEMPLATE``'s.
#:
#: 29 BINDINGS: 12 buffers + 3 states + 6 coefficient vectors is 21, plus 4 uints,
#: dtdx and 6 scalars. The Metal argument-table ceiling is 31, so this family fits
#: with two to spare — and ``test_metal_bfast`` asserts the count rather than
#: leaving a later edit to discover the ceiling at compile time.
_BFAST_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void bfast_pml_curl_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device float*       s0      [[buffer(9)]],
    device float*       s1      [[buffer(10)]],
    device float*       s2      [[buffer(11)]],
    device const float* kmx     [[buffer(12)]],
    device const float* sinvx   [[buffer(13)]],
    device const float* kmy     [[buffer(14)]],
    device const float* sinvy   [[buffer(15)]],
    device const float* kmz     [[buffer(16)]],
    device const float* sinvz   [[buffer(17)]],
    constant uint&      nx      [[buffer(18)]],
    constant uint&      ny      [[buffer(19)]],
    constant uint&      nz      [[buffer(20)]],
    constant uint&      n_elem  [[buffer(21)]],
    constant float&     dtdx    [[buffer(22)]],
    constant float&     k1_0    [[buffer(23)]],
    constant float&     k2_0    [[buffer(24)]],
    constant float&     k1_1    [[buffer(25)]],
    constant float&     k2_1    [[buffer(26)]],
    constant float&     k1_2    [[buffer(27)]],
    constant float&     k2_2    [[buffer(28)]],
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

    // --- the ghost rule, per axis (stepping._shift_up:1723 / _shift_down:1787) --
    // Unchanged from the certified curl: BFAST reads exactly the operands
    // _curl_operands already gathered (stepping.py:1560-1598), with the same
    // backward-stride selection for D -- MEEP negates the strides ONCE, before
    // BOTH calls (step_db.cpp:80-83). No new ghost, no new stencil.
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

    // --- ownership predicates, shared by the BFAST advance mask and the curl mask
    // (stepping._mask_non_owned_cells:1865 -- the SAME predicate, applied to the
    // advance at :902 and to the summed curl at :369/:450).
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);

__BFAST_TAIL__

    // --- ownership mask (stepping._mask_non_owned_cells:1865) -------------------
__MASK__

    // --- split-field recurrence (stepping._apply_pml_update:1905) ---------------
    // dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    // sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
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
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 1560-1598->1607-1645

#: The tail, spelled once. Substituted in for a ``HAS_BFAST=1`` build and replaced
#: by a comment for the identity build.
#:
#: THE SUMS PAIR EACH SHIFTED OPERAND WITH ITS CENTRE in the operand order of
#: stepping.py:923-924, and the mapping to this kernel's locals is the curl's own:
#: target 0's ``first`` is Ez shifted along y (``c_y``) with centre ``c``, its
#: ``second`` is Ey shifted along z (``b_z``) with centre ``b``; target 1 takes
#: (a_z, a) and (c_x, c); target 2 takes (b_x, b) and (a_y, a). Note the sign
#: pattern is NOT the curl's: the curl is ``+g1diff - g2diff`` built from
#: ``(shifted - at)`` then ``(at - shifted)``, while this is ``+k1*g1sum -
#: k2*g2sum`` with BOTH sums in the same orientation.
_BFAST_TAIL = r"""    // --- the BFAST tail (stepping._bfast_term:896-904), SHARED operands -------
    // AFTER the dtdx curl, BEFORE the ownership mask -- the array path's fold
    // order (:364-368 after :342, before :369). No dtdx anywhere (:836-837): this
    // is the Tustin filter, and MEEP's step_bfast never reads the dtdx it is
    // handed. All three targets always run the tail (grid.py:744-753); with
    // seeded state the zero-k advance -2*state is byte-visible, so no arm skips it.
    float st0 = s0[ii];
    float st1 = s1[ii];
    float st2 = s2[ii];
    float total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b));
    float total1 = (k1_1 * (a_z + a)) - (k2_1 * (c_x + c));
    float total2 = (k1_2 * (b_x + b)) - (k2_2 * (a_y + a));
    float adv0 = total0 - (2.0f * st0);
    float adv1 = total1 - (2.0f * st1);
    float adv2 = total2 - (2.0f * st2);
    // The advance carries the SAME ownership predicate the curl does, applied
    // BEFORE the state store (:902 precedes :903). MEEP writes F only inside its
    // owned-cell loop; F has no spatial stencil, so an unowned cell can never
    // contaminate an owned one, but leaving the advance in would make a
    // diagnostic read of f_bfast disagree with MEEP's.
__ADV_MASK__
    // state += advance, every cell written -- the array path's whole-array
    // in-place update (:903); the masked rows receive state + 0.0 exactly as the
    // array path computes them. The store lands BEFORE the summed-curl mask,
    // which is the array path's order.
    s0[ii] = st0 + adv0;
    s1[ii] = st1 + adv1;
    s2[ii] = st2 + adv2;
    // curl <- curl - adv: the caller-subtracts sign convention (:904 returns
    // -advance and :364-368 adds it). IEEE-754 defines subtraction AS addition of
    // the negation, so this carries the array path's result on every input
    // including signed zeros, and NO unary minus appears.
    curl0 = curl0 - adv0;
    curl1 = curl1 - adv1;
    curl2 = curl2 - adv2;
"""

_NO_BFAST_TAIL = ("    // HAS_BFAST=0: no second additive pass. This build exists\n"
                  "    // so the gate can pin it byte-identical to the certified\n"
                  "    // shaders.pml_curl_step on the same seeds.")


#: Per direction, the (target, axis, flag) triples ``stepping._mask_non_owned_cells``
#: zeroes — cell 0 of a metallic axis, for every target whose Yee shift there is 0
#: (stepping.py:1912-1949, reading ``fields.IYEE_SHIFTS``). RESTATED here because
#: ``shaders.ownership_mask`` generalises on the ZERO LITERAL (for the complex
#: family) and not on the masked VARIABLE, and this family needs the identical
#: predicate applied to ``adv`` at stepping.py:929 as well as to ``curl`` at :397.
#:
#: Duplicate-plus-pin, the doctrine ``metal_kernels.coverage`` already uses for its
#: clause list: ``test_metal_bfast.test_the_advance_mask_is_the_curl_mask`` asserts
#: this emitter's output equals ``shaders.ownership_mask``'s with the variable
#: renamed, over the full boundary/direction cross product, so the two cannot drift
#: into masking different cells.
_OWNED_CELL_PAIRS: Dict[bool, Tuple[Tuple[int, int, str], ...]] = {
    False: ((0, 0, "at_x"), (1, 1, "at_y"), (2, 2, "at_z")),
    True: ((0, 1, "at_y"), (0, 2, "at_z"), (1, 0, "at_x"),
           (1, 2, "at_z"), (2, 0, "at_x"), (2, 1, "at_y")),
}


def advance_mask(codes: Sequence[int], backward: bool) -> str:
    """The ownership mask applied to the BFAST advance (stepping.py:929).

    THE SAME PREDICATE the summed curl gets, not a similar one. Masking the advance
    masks the state and the increment together, which is what keeps an unowned
    cell's ``f_bfast`` at exactly the value MEEP's owned-cell loop never touches.
    """
    lines = [f"    adv{target} = {flag} ? 0.0f : adv{target};"
             for target, axis, flag in _OWNED_CELL_PAIRS[bool(backward)]
             if codes[axis] == METALLIC]
    return "\n".join(lines) or "    // no metallic axis: no ownership mask on adv"


def bfast_curl_source(codes: Sequence[int], backward: bool,
                      contract: str = shaders.CONTRACT_OFF,
                      has_bfast: bool = True) -> str:
    """The specialised ``bfast_pml_curl_step`` source for one configuration.

    ``codes`` is the per-axis 0/1 PERIODIC/METALLIC triple ``_boundary_kinds``
    resolves; ``backward`` selects ``step_D``'s negated strides over ``step_B``'s
    forward ones; ``has_bfast`` selects the tail. All three are baked into the
    string, as Triton bakes its constexprs, so no branch survives around a float
    expression.

    The ghost rule, the ownership mask and the substitution engine come from
    :mod:`.templates`, which RE-EXPORTS ``shaders``' certified emitters — measured
    ``templates.ghost is shaders.ghost`` and likewise for ``ownership_mask``,
    ``substitute`` and ``contraction_pragma``. They are the same objects, not copies,
    so there is nothing here for a test to pin: an equality assertion would be the
    ``x == x`` this module refuses elsewhere. What holds them is
    ``test_metal_kernels.test_the_checked_in_fingerprints_match_the_tree``, which
    re-hashes all thirty-six certified sources these emitters produce. (An earlier
    revision of this paragraph claimed ``test_metal_bfast`` pinned "copies"
    character-identical. There are no copies and there was no such test.)
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if has_bfast:
        tail = templates.substitute(_BFAST_TAIL, {
            "__ADV_MASK__": advance_mask(codes, backward),
        })
    else:
        tail = _NO_BFAST_TAIL
    return templates.substitute(_BFAST_CURL_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__BFAST_TAIL__": tail,
        "__MASK__": templates.ownership_mask(codes, backward),
    })


def compile_bfast_curl(codes: Sequence[int], backward: bool,
                       contract: str = shaders.CONTRACT_OFF,
                       has_bfast: bool = True) -> Any:
    """The specialised BFAST curl entry point for one configuration."""
    source = bfast_curl_source(codes, backward, contract, has_bfast)
    return compile_source(source).bfast_pml_curl_step


def enumerate_bfast_sources(contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every shipped BFAST specialisation, keyed by a stable label.

    The same shape ``shaders.enumerate_sources`` returns, so this family's gate can
    record one sha256 per emitted source in its own provenance block. The
    ``HAS_BFAST=0`` builds are enumerated too: they are shipped source, they are
    what the identity leg launches, and an un-fingerprinted mutant substrate is
    exactly the thing a later edit changes silently.
    """
    out: Dict[str, str] = {}
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for has_bfast in (True, False):
            for cx in (PERIODIC, METALLIC):
                for cy in (PERIODIC, METALLIC):
                    for cz in (PERIODIC, METALLIC):
                        label = (f"bfast_pml_curl_step/{name}/"
                                 f"{'on' if has_bfast else 'off'}/{cx}{cy}{cz}")
                        out[label] = bfast_curl_source((cx, cy, cz), backward,
                                                       contract, has_bfast)
    return out


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors ``coverage._grid_reasons`` so the two can be
# diffed. Clause 11 (BFAST) is INVERTED: this product REQUIRES bfast_active where
# every shipped kernel requires it off, which is why no admitted-overlap ambiguity
# can arise. Everything else is KEPT, restated rather than imported, because the
# shipped reason list is built inside a function whose BFAST clause cannot be
# subtracted from outside. ``test_metal_bfast`` runs this list against the Triton
# BFAST module's over a shared configuration matrix and asserts they agree once
# each side's own backend clause is removed, so the restatement cannot drift.


def _bfast_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every Metal BFAST predicate shares."""
    reasons: List[str] = []

    # 1. The Metal backend, and the host array module the mirrors copy from.
    #    IMPORTED from this package's own coverage module: one backend clause for
    #    the whole package, so a family cannot quietly admit a run on a host whose
    #    subnormal policy the MPS executor refuses.
    reasons.extend(_metal_backend_reasons(grid))

    # 2. Real storage REQUIRED (Phase A). Grid DELIBERATELY allows bfast + Bloch
    #    (grid.py:719-724 — two independent MEEP constructor slots), and a
    #    complex-storage BFAST run is the complex family's queued composition.
    if getattr(fields, "force_complex_fields", False):
        # The wording is the Triton module's VERBATIM, and deliberately so: the
        # clause-list equivalence test compares the two reason SETS exactly, which
        # is the pin that catches a clause added to one port and not the other.
        reasons.append("force_complex_fields=True: complex-storage BFAST is the "
                       "complex family's queued composition, not Phase A")

    # 3. An absorber that actually absorbs (split-field family only).
    #
    #    A DECLARED PORT-SPECIFIC DIVERGENCE, and the only one in this list. The
    #    Triton clause ends "— zero demand, and no_pml.py refuses bfast too", which
    #    is a statement about the TRITON package and is MEASURABLY FALSE of this one:
    #    `no_pml_constitutive.metal_null_constitutive_coverage` ADMITS a BFAST run
    #    under an inactive absorber (see the module docstring; it is a different slot
    #    and it is the correct answer there). Carrying the sentence over would have
    #    put a false claim in a refusal reason, so it is dropped and the divergence is
    #    DECLARED — `test_metal_bfast.PORT_SPECIFIC_CLAUSES` names both strings and
    #    asserts each is observed on its own side, so the rest of the clause list can
    #    still be compared as exact set equality.
    #
    #    The hole this closes: the equivalence pin's matrix built an ACTIVE PML in
    #    every row, so this clause was never compared and the divergence was silent.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "split-field path only; no-PML BFAST is refused by name)")

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

    # 5. No mirror plane anywhere.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not "
                       "carried by this family)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian only. Cylindrical + BFAST is UNREACHABLE — the Grid itself
    #    raises (Grid._resolve_bfast, grid.py:706-742, on MEEP's own
    #    step_generic.cpp:376 missing '- F[i]') — but the clause is KEPT: this
    #    package's doctrine forbids inferring a refusal from another module's guard.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried (and the "
                       "grid itself refuses bfast there, grid.py:706-742)")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0. A Bloch phase needs complex storage and multiplies one wrapped
    #    plane; the pairing is refused BY NAME rather than by omission.
    if getattr(grid, "has_bloch", False):
        # Wording verbatim from the Triton module, for the same reason as the
        # complex-storage clause above: the equivalence test compares the two
        # reason sets exactly.
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r} "
                       f"(Grid allows bfast+Bloch; this Phase A kernel "
                       f"refuses it by name — the complex family's queue)")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. No instantaneous nonlinearity.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried; that family is a separate tranche)")

    # 11 (INVERTED). BFAST must be ACTIVE: a k = 0 run belongs to the certified
    #    plain kernels, and the array path never enters the fold (:336/:422).
    if not getattr(grid, "bfast_active", False):
        reasons.append("grid.bfast_active is False: this product exists only for "
                       "BFAST runs; a k = 0 run belongs to the certified plain "
                       "kernels")

    # 11a (THIS FAMILY'S OWN — no counterpart in coverage._grid_reasons).
    #     A nonzero k component on a DECLARED-invariant axis is the one reachable
    #     class where stepping.py's BFAST pass is not MEEP's answer, so this
    #     product refuses to reproduce it faster. See the module docstring for the
    #     operand-nulling asymmetry and the measured max|diff| ~ 0.598.
    #
    #     A missing or unanswerable is_invariant is refused OUTRIGHT rather than
    #     read as "not invariant": inferring admission from an absent reader is
    #     the attribute-absence trap this package names elsewhere.
    bfast_k = getattr(grid, "bfast_scaled_k", None)
    reader = getattr(grid, "is_invariant", None)
    if bfast_k is None or not callable(reader):
        reasons.append("grid does not expose both bfast_scaled_k and a callable "
                       "is_invariant; the invariant-axis clause cannot be "
                       "answered, and an unanswerable question is not coverage")
    else:
        for axis in range(3):
            try:
                invariant = bool(reader(axis))
                component = float(bfast_k[axis])
            except Exception as exc:  # noqa: BLE001 - unreadable is not covered
                reasons.append(f"the invariant-axis clause could not be answered "
                               f"on axis {axis}: {exc!r}")
                continue
            if invariant and component != 0.0:
                reasons.append(
                    f"bfast_scaled_k[{axis}] = {component!r} on DECLARED-invariant "
                    f"axis {axis}: MEEP nulls the partner OPERAND as well as "
                    f"zeroing the coefficient (step_db.cpp:62-63 + "
                    f"step_generic.cpp:342-346), so its increment there is "
                    f"F_new = -F_prev; stepping.py:911-924 zeroes only the "
                    f"coefficient and keeps the other product — this kernel "
                    f"transcribes stepping.py, so the pairing is refused rather "
                    f"than accelerated")

    # 12. Beta KEPT refused: the beta-then-bfast fold ORDER (:356-363 before
    #     :364-368) is byte-significant, so bfast+beta is a coordinated follow-up.
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is "
                       f"nonzero: the beta-then-bfast fold order is "
                       f"byte-significant; bfast+beta is a coordinated follow-up, "
                       f"not this tranche")

    # 9c. Stored E — the invariant behind admitting dispersion for the curl.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return reasons


def _curl_conductivity_reasons(fields: Any) -> List[str]:
    """Conductivity refused on ALL six curl targets, BOTH directions.

    THIS PACKAGE SHIPS NO CONDUCTIVE PRODUCT AT ALL, so on Metal the refusal is
    unconditional rather than a hand-off: there is nothing for a conductive run to
    fall to but the array path. The cross-reference to ``triton_kernels/conductivity.py``
    :515-518 is kept because that module's predicate refuses bfast in return, which is
    what makes the pairing a NAMED follow-up on both tracks rather than a silent
    overlap — but the file named is the Triton package's, and this clause's wording
    therefore diverges from the Triton BFAST module's by that half-sentence. The
    divergence is DECLARED in ``test_metal_bfast.PORT_SPECIFIC_CLAUSES``, which
    asserts both strings are observed, so the rest of this list is still compared as
    exact set equality. (Before that pin existed this function was not compared to its
    Triton counterpart at all.)

    A missing or non-callable reader is refused OUTRIGHT: inferring "no conductivity"
    from the absence of ``condfac_for`` is admission by attribute absence.
    """
    reasons: List[str] = []
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
        return reasons
    for target in CURL_TARGETS:
        try:
            conductive = reader(target) is not None
        except Exception as exc:  # noqa: BLE001 - unreadable means not covered
            reasons.append(f"condfac_for({target!r}) raised {exc!r}")
            continue
        if conductive:
            reasons.append(
                f"a conductivity is installed on {target}: this kernel "
                f"transcribes the plain split-field recurrence only, this package "
                f"ships no conductive product, and the Triton conductive family's "
                f"own predicate refuses bfast in return "
                f"(triton_kernels/conductivity.py:515-518) — a named follow-up, "
                f"not a silent overlap")
    return reasons


def bfast_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                            residency: Any = None) -> Coverage:
    """May the Metal BFAST curl kernel step this (fields, pml, sub_step)?

    Allocation and layout are scoped to the NAMED sub-step's 12 arrays — 3 targets,
    3 ``fu_*``, 3 sources and 3 ``f_bfast_*`` states. A missing state RAISES in the
    array path (stepping.py:916-921); here it is a refusal by name, before any
    launch.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _bfast_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_curl_conductivity_reasons(fields))

    # 9a/9b. A registered susceptibility must be one this package understands.
    reasons.extend(_susceptibility_reasons(fields))

    # 13. The named sub-step's targets, auxiliaries, sources AND IIR states.
    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"])
             + BFAST_STATE_NAMES[sub_step])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    # 14. Layout: float32, C-contiguous, grid.shape — the states included, since
    #     they share the field storage layout (fields.py:632-651).
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class BfastPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free real-field BFAST PML curl sub-step.

    ``launch.PmlCurlPlan`` plus three state mirrors, six host-rounded scalars and
    the ``has_bfast`` specialisation. ``run`` and the contraction-variant contract
    are the base's, so this family cannot spell the guard its own way.

    THE STATE MIRRORS WRAP ``fields.f_bfast_*`` THEMSELVES. On CuPy that is
    literally pointer identity; here it is the mirror's ``host`` reference plus an
    in-place ``sync_out``, which is the same requirement one level up. The driver's
    flux backup/restore around the magnetic half-step (driver.py:4126-4135) depends
    on it, and because the IIR is marginally stable a missed restore never decays.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "ks", "has_bfast",
                 "backward", "bc", "residency", "volumes")

    family = "real-field BFAST PML curl"

    REPR_FIELDS = ("sub_step", "shape", "bc", "ks", "has_bfast")

    def __init__(self, sub_step: str, shape, dtdx: float, bc,
                 ks: Sequence[float], residency: Residency,
                 targets, auxiliaries, sources, states, coefficients,
                 functions: Dict[str, Any], volumes: Sequence[str],
                 has_bfast: bool = True) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(
                f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        # Already f32-rounded by the host (bfast_curl_coefficients); float() keeps
        # the bits, and a Python double bound to `constant float&` arrives
        # correctly rounded — MEASURED on this host (5/5 needles), unlike Triton's
        # dead tl.float64 path. So the words the kernel receives are the array
        # path's.
        self.ks = tuple(float(value) for value in ks)
        if len(self.ks) != 6:
            raise ValueError(f"ks must be the six (k1,k2) scalars, got {ks!r}")
        self.has_bfast = bool(has_bfast)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.residency = residency
        self.volumes = tuple(volumes)
        # Built ONCE, in the shader's exact binding order: 12 volumes, 3 states,
        # 6 coefficient vectors, 4 extents, dtdx, 6 scalars = 29 arguments. The
        # launch path unpacks this tuple and does nothing else.
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources) + tuple(states)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem,
               self.dtdx) + self.ks)


def _bfast_functions(codes, backward: bool, contract_variants: Sequence[str],
                     has_bfast: bool) -> Dict[str, Any]:
    return {mode: compile_bfast_curl(codes, backward, mode, has_bfast)
            for mode in contract_variants}


def plan_bfast_pml_curl(fields: Any, pml: Any, sub_step: str,
                        residency: Optional[Residency] = None,
                        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
                        ) -> Optional[BfastPmlCurlPlan]:
    """Build a BFAST curl plan from the engine's own objects, or None.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    The six scalars come from :func:`bfast_curl_coefficients` with the grid's OWN
    declared-dimensionality invariance flags (grid.py:1165-1183 — never a shape
    test) and the sub-step's side, which is identical arithmetic to the array
    path's per-call-site computation.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not bfast_pml_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    codes = [1 if kind == "metallic" else 0 for kind in resolve(grid, pml)]
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    ks = bfast_curl_coefficients(grid.bfast_scaled_k, invariant,
                                 magnetic=(sub_step == "step_B"))
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    states = [residency.mirror(n, getattr(fields, n))
              for n in BFAST_STATE_NAMES[sub_step]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]) + BFAST_STATE_NAMES[sub_step])
    return BfastPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, ks, residency,
        targets, auxiliaries, sources, states, coefficients,
        _bfast_functions(codes, spec["backward"], contract_variants, True),
        volumes, has_bfast=True)


def plan_bfast_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                    flat: Dict[str, Any], codes, dtdx: float,
                                    ks: Sequence[float], residency: Residency,
                                    functions: Optional[Dict[str, Any]] = None,
                                    contract_variants: Sequence[str] = (
                                        shaders.CONTRACT_OFF,),
                                    has_bfast: bool = True,
                                    ) -> BfastPmlCurlPlan:
    """Build a BFAST curl plan from bare host arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones. ``functions`` is the
    mutation seam — dropping it is not a silent slowdown but a silent DISARMING,
    since every mutation leg would then launch the shipped kernel and report the
    defect as uncaught. ``ks`` carries the (possibly deliberately wrong) six
    scalars and ``has_bfast=False`` the identity leg's arm.

    ``arrays`` must carry the sub-step's three ``f_bfast_*`` states beside its
    targets, auxiliaries and sources.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    targets = [residency.mirror(n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n]) for n in spec["sources"]]
    states = [residency.mirror(n, arrays[n]) for n in BFAST_STATE_NAMES[sub_step]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]) + BFAST_STATE_NAMES[sub_step])
    return BfastPmlCurlPlan(
        sub_step, shape, dtdx, codes, ks, residency,
        targets, auxiliaries, sources, states, coefficients,
        functions if functions is not None
        else _bfast_functions(codes, spec["backward"], contract_variants,
                              has_bfast),
        volumes, has_bfast=has_bfast)


# ---------------------------------------------------------------------------
# The constitutive companion — no new kernel, a restated predicate
# ---------------------------------------------------------------------------
#
# WHY THIS IS SOUND, and it is a READ of stepping.py rather than an analogy. BFAST's
# whole effect is the second additive curl term and its IIR state, and both live in
# `step_db`: `_bfast_advance` (S:874-905) is called from the CURL path only.
# `update_H` (S:907-925) and `update_E` (S:927-994) contain no BFAST branch, read no
# `f_bfast_*` volume and consult no `bfast_scaled_k` — `update_H` is
# `_apply_constitutive_pml` over the three magnetic terms and `update_E` is the
# `gs*us` product plus the same auxiliary. So the ARITHMETIC of the constitutive
# pair on a BFAST run is byte-for-byte the arithmetic the certified constitutive
# kernel already reproduces, and only the ADMISSION has to change.
#
# THIS IS THE SAME MOVE `special_kz.beta_run_constitutive_coverage` MAKES, for the
# same reason, and it is the cheapest coverage on either track: no kernel, no new
# specialisation, no new byte claim beyond re-running the certified one on a
# configuration the shipped predicate refused for a clause that does not apply to
# this sub-step. What it buys is whole-step composition — without it a BFAST run
# takes its curls on the device and its constitutive pair on the array path, which
# forces a sync per sub-step and makes the residency verdict refuse.

def bfast_run_constitutive_coverage(fields: Any, pml: Any, side: str,
                                    residency: Any = None) -> Coverage:
    """May the CERTIFIED Metal constitutive kernel step a BFAST run?

    :func:`_bfast_grid_reasons`' clause set with nothing added and nothing removed
    — the BFAST clause is already inverted there — plus the constitutive side's own
    clauses, exactly as the certified predicate spells them. The conductivity clause
    is deliberately NOT applied: a conductivity changes the CURL recurrence
    (``stepping._apply_curl`` reads ``condfac_for`` at S:479 and nothing else does),
    so refusing it here would refuse a configuration this sub-step steps correctly.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _bfast_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_susceptibility_reasons(fields))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is (D - sum P), "
                "not D (S:967) — that configuration belongs to the ADE kernel")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product reads "
                "neighbours; this sub-step is element-wise)")

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


def plan_bfast_run_constitutive(fields: Any, pml: Any, side: str,
                                residency: Any = None,
                                contract_variants: Sequence[str] = (
                                    templates.CONTRACT_OFF,)) -> Any:
    """A CERTIFIED constitutive plan for a BFAST run, or None.

    The arithmetic, the source and the plan class are the certified ones; only the
    ADMISSION is this module's. Building the plan through the certified builder —
    rather than re-deriving one here — is what makes "same kernel" a fact instead
    of a claim: a divergence would have to come from the predicate, which is the
    only thing this family contributes.
    """
    from .launch import (  # noqa: PLC0415 - avoids a circular import at module load
        ConstitutivePlan, _constitutive_functions,
    )

    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not bfast_run_constitutive_coverage(fields, pml, side, residency).covered:
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


# ---------------------------------------------------------------------------
# WIRING — two curl arms and the constitutive pair, as of tranche 2
# ---------------------------------------------------------------------------
#
# THIS MODULE USED TO REGISTER NOTHING and called that omission the safety argument
# for building the family before dispatch was decided. The argument was sound and
# the omission was the wrong instrument: an unregistered arm is invisible to every
# sweep that does not name this module, so "nothing co-admits it" could only be
# checked by a probe written for this family. The arms are registered now, and the
# disjointness sweep reaches them the same way it reaches every other row.
#
# WHAT SEPARATES THEM: `coverage._grid_reasons` clause 11 refuses a BFAST run BY
# NAME, `complex_fields` clause 8 refuses it by name, `special_kz` clause 11
# refuses it by name, and clause 11 here is INVERTED — `grid.bfast_active` must be
# True. So on the curl slots the separation is total and stated four times.
#
# THE ONE ADMISSION THAT IS NOT AN AMBIGUITY, and it was measured rather than
# assumed: `no_pml_constitutive.metal_null_constitutive_coverage` ADMITS a BFAST
# run whenever the absorber is inactive. That is a DIFFERENT SLOT (update_H /
# update_E) and it is the correct answer there — `stepping.update_H` returns at
# :916-917 and `update_E` at :954-955 before reading an array, and the constitutive
# sub-steps carry no BFAST state at all, the pass being on `step_db` only. A BFAST
# run under an inactive absorber therefore composes as: array path on the curls
# (this family requires an ACTIVE absorber, clause 3), null plans on the
# constitutive pair. The sweep records that composition rather than flagging it.
#
# `update_P` still carries no Metal arm at all, so the Triton module's two
# exceptions (`ade_update_p_coverage`, `fused_ade_state_coverage`) have no
# counterpart here. That is a fact about today's tree, not a law, and the sweep
# re-measures it rather than inheriting this sentence.


def _curl_arm_coverage(context: Any, slot: str) -> Coverage:
    return bfast_pml_curl_coverage(context.fields, context.pml, slot,
                                   context.residency)


def _curl_arm_plan(context: Any, slot: str) -> Optional[BfastPmlCurlPlan]:
    return plan_bfast_pml_curl(context.fields, context.pml, slot,
                               context.residency, context.contract_variants)


#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return bfast_run_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency)


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_bfast_run_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """This family's four arms: two curls and the certified constitutive pair."""
    from . import arms  # noqa: PLC0415 - deferred: `arms` imports nothing of ours

    registered = [
        arms.register(family=FAMILY, slot=slot, label="BFAST",
                      coverage=_curl_arm_coverage, plan=_curl_arm_plan,
                      prefix="BFAST: ", noun="BFAST PML curl", wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="BFAST",
                      coverage=_constitutive_arm_coverage,
                      plan=_constitutive_arm_plan,
                      prefix="BFAST: ", noun="BFAST constitutive", wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    return tuple(registered)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[Any, ...] = register_arms()
