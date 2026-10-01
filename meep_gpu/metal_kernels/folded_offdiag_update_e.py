"""The FOLDED off-diagonal (tensor) electric constitutive sub-step, on Metal.

``update_E`` with off-diagonal chi1inv rows installed on a MIRROR-FOLDED grid — the
intersection of two families that are ALREADY CERTIFIED on this backend
(:mod:`.offdiag_update_e` and :mod:`.symmetry`), which is exactly why it is cheap
AND exactly why it is dangerous: each half is byte-identical alone, and whether they
stay so together is a question, not a corollary.

BUILT AS A COMPOSITION WHERE THE SOURCE ALLOWS, AND THE BOUNDARY IS STATED RATHER
THAN BLURRED. Three of the four pieces are the certified emitters' own output,
reached by CALLING them:

* the kernel TEMPLATE is ``offdiag_update_e._TEMPLATE`` itself — not a copy, the
  module attribute — so the fold adds no binding, no argument and no line of
  scaffolding (28 bindings, the certified count);
* the ghost block of a PERIODIC or METALLIC axis is
  ``offdiag_update_e._two_way_ghost``'s text, character for character;
* the whole component block of a component whose LIVE row slots take no negated
  mirror partner is ``offdiag_update_e._component_source``'s text, character for
  character — including the wall mask and the row sum.

TWO PIECES ARE THIS FAMILY'S OWN BODY, and they are the two the fold changes:

* :func:`_folded_two_way_ghost` for a MIRROR axis — the DOWN index is REDIRECTED to
  stored row :data:`MIRROR_SOURCE_INDEX` on the face-0 lane instead of masked, and
  the UP face is masked exactly as METALLIC;
* :func:`_folded_term` for a term whose partner axis carries a NEGATED mirror ghost
  — the certified term with the ghost lane's sign flipped.

That split is CHECKABLE BY CHARACTER and ``test_metal_folded_offdiag`` checks it in
both directions: an unfolded code triple must make :func:`folded_offdiag_source`
return a string EQUAL to ``offdiag_update_e.offdiag_source``'s, and a folded one
must differ in exactly the emitted blocks named above. On the Triton track the
analogous "an arm reduces to the certified body" claim can only be settled with a
PTX read — ``triton_kernels/offdiag_update_e.py:114`` and ``:69`` both rest on a
"PTX-verified-different binary", and ``triton_kernels/folded_offdiag_update_e.py:
102-106`` restates the first. Here it is string equality, which is strictly better
evidence for THIS claim and is the one place this port is ahead of that one.

WHAT IT REPLACES. ``stepping.update_E`` (stepping.py:954) with an active PML,
``fields.has_offdiagonal_epsilon`` True, NOT ``fields.has_nonlinearity``, and at
least one axis resolving to MIRROR through ``stepping._boundary_kinds``
(stepping.py:2191-2196) — the ``elif offdiagonal:`` branch (stepping.py:1001-1008).
Every line of the arithmetic is :mod:`.offdiag_update_e`'s, unchanged, because the
array path's is; the whole delta is what ``_shift_down`` and ``_shift_up`` do on a
MIRROR axis.

===========================================================================
THE ELEMENT-WISE ARGUMENT DOES NOT TRANSFER — why this family exists
===========================================================================

:func:`symmetry.folded_constitutive_coverage` re-admits the CERTIFIED constitutive
kernel onto a folded grid on exactly one argument: that sub-step READS NO
NEIGHBOUR, so the fold's ghost rules cannot reach it, and the only thing the fold
changes — the stored extent — is already carried by the PML coefficient vectors.
The OFF-DIAGONAL row product reads neighbours (stepping.py:1243-1249), so that
argument is unavailable. It reaches the sub-step in ONE role and not the other, and
both halves matter:

1. **THE PARTNER-AXIS DOWN SHIFT IS LIVE ON A FOLD.** ``_shift_down`` is called WITH
   a component name and the plane's parity (stepping.py:1243-1245), so a MIRROR
   partner axis takes stepping.py:1870-1873 — ``parity * field[MIRROR_SOURCE_INDEX]``,
   a LIVE parity-weighted interior plane, neither the periodic wrap nor the metallic
   zero. And it is NOT masked away afterwards:
   ``_mask_metallic_wall_coupling`` asks ``is_metallic and NOT is_mirrored``
   (stepping.py:1282) and therefore ABSTAINS on the fold plane, with its own
   measurement behind the abstention (stepping.py:1266-1277: zeroing the fold plane
   the metallic way costs 2.0e-02 even / 3.5e-03 odd). "The mask fires" and "the
   ghost is live" are the same switch, thrown the other way by the fold.
2. **THE OWN-AXIS UP SHIFT IS *NOT* PARITY-WEIGHTED, ON EITHER TERMINATION.**
   ``_offdiagonal_terms`` calls ``_shift_up`` with FOUR arguments (stepping.py:
   1248-1249) — no ``component``, no ``mirror_phase``, no ``reflect_row`` — so the
   folded-PERIODIC reflect branch (stepping.py:1819-1827) cannot fire and both mirror
   terminations fall to stepping.py:1828-1830, an exact ``0.0``. That is
   byte-identical to the METALLIC arm this kernel already carries, and it is
   transcription FIDELITY rather than a physics claim: see ARRAY-PATH FINDING below.

CONSEQUENCE FOR THE BINDINGS, and it is the reason this family adds no argument:
because (2) needs no reflect row and because the parity of (1) is a COMPILE-TIME
specialisation on this backend (see below), the folded kernel's signature is the
certified kernel's — 28 bindings, same order, same meaning. Contrast the Triton
twin, which passes three runtime ``gw*`` scalars.

BOTH MIRROR CODES BEHAVE IDENTICALLY IN THIS KERNEL, and that is a PREDICTED NULL
with a derivation rather than a collapse: ``update_E`` has no ownership mask
(``_mask_non_owned_cells`` is a curl-only call) and, by (2), no reflect row, so the
MIRROR_METALLIC / MIRROR_PERIODIC split cannot reach it. The codes are still CARRIED
and still CLASSIFIED through :func:`symmetry.folded_axis_kinds`, because that
classifier cross-checks ``_stored_past_owned`` against ``grid.is_metallic`` and
refuses a disagreement — collapsing the codes here would discard the cross-check —
and because a plan built here and one built in :mod:`.symmetry` must index the same
table. The equality is a sibling-test null control, not an assumption.

===========================================================================
WHAT IS METAL-SPECIFIC — measured on this host, not inherited
===========================================================================

THE GHOST WEIGHT IS A COMPILE-TIME SPECIALISATION HERE, NOT A RUNTIME SCALAR. This
is the exact inversion of the Triton family's central platform rule
(``triton_kernels/folded_offdiag_update_e.py:93-107`` holds the weight RUNTIME
because Triton lowers ``-x`` as ``0.0 - x`` and canonicalizes signed zeros, and
rests on "1.0 * x is exact in f32 for every input"). MEASURED 2026-08-16, torch
2.10.0 MPS / numpy 2.4.3 / macOS 26.2 arm64, under ``shaders.contraction_pragma``'s
``off`` mode, against NumPy float32, over a 30-word set carrying 4 signed zeros, 10
subnormals across the band and 16 normals including +/-FLT_MAX — IN THIS FAMILY'S
OWN TERM SHAPE rather than in the abstract:

    THE GHOST VALUE ALONE, w * x
        at ? -x : x        (select the VALUE)     0/30  EXACT
        x * -1.0f          (compile-time)         0/30  EXACT
        x * 1.0f / plain copy   (the even arm)    0/30  EXACT
        (at ? -1.0f : 1.0f) * x  (LITERAL select) 0/30  EXACT
        0.0f - x                                 12/30  WRONG  [zeros 2/4, subn 10/10]
        RUNTIME w * x, w = -1.0f                 10/30  WRONG  [zeros 0/4, subn 10/10]
        RUNTIME w * x, w = +1.0f                 10/30  WRONG
        (at ? w : 1.0f) * x, RUNTIME w            10/30  WRONG  [subn 10/10]

    THE TERM'S OWN SUM, a + w * x, with a NORMAL (1e-30)
        every spelling above                      0/30  EXACT

Three things follow and each changes what this file does:

* **it is the RUNTIME-ness of the weight that flushes, not the select.**
  ``(at ? -1.0f : 1.0f) * x`` — a select between two LITERALS — is exact while
  ``(at ? w : 1.0f) * x`` with the same values arriving in a buffer misses 10/30.
  That refines the record :mod:`.symmetry` carries (symmetry.py's measured table
  lists only the runtime form) and it is why the weight is baked into the source
  here rather than bound. ``-x`` is used because it is exact and is the spelling
  :mod:`.symmetry` already ships;
* **the subnormal half of the divergence is bounded by the precondition and the
  SIGNED-ZERO half is not.** ``0.0f - x`` is wrong on 2 of 4 signed zeros with no
  subnormal in sight — a D volume that is exactly zero on the ghost plane is
  ordinary, not exotic — so the refuted spelling is a must-catch mutation rather
  than a comment. The runtime-weight spellings diverge ONLY on subnormals, which
  makes them refused-under-the-precondition rather than wrong;
* **the divergence needs the value to survive to the output.** Added to a NORMAL
  partner every spelling agrees, because the addition renormalises. So a leg that
  drove only normal data through the whole term would measure nothing about the
  spelling, which is why the sibling tests carry the ghost plane's own value class.

NOTHING IN THIS FAMILY DIVIDES, TAKES A SQUARE ROOT, OR TAKES A MIN OR MAX, so
``fast::divide`` (2471 mismatches on this host) and the signed-zero min/max hazard
(shaders.py:41-46) do not arise. The only negation is the ghost lane's ``-x`` and
the only other sign work is integer index arithmetic, which is exact.

THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND. ``torch.mps.compile_shader``
exposes no disassembly, so this family cannot refuse a compile whose emitted code
violates the policy and cannot establish that the contraction guard was obeyed. The
byte legs and the mutation legs are the only arbiters and they are BEHAVIOURAL —
they catch a wrong answer, not a wrong instruction. That is this certification's one
weakness against the Triton one. It is PARTLY offset here and only partly: the
"reduces to the certified body" claim, which Triton could settle only with a PTX
read, is settled by string equality (see the top of this docstring). What remains
unauditable is whether the compiler honoured the contraction directive that
``shaders.contraction_pragma`` emits — the directive is deliberately spelled in
``shaders.py`` and nowhere else, and a test fails the build on a second spelling
anywhere in the package, including in a docstring like this one.

===========================================================================
THE INVERTED CLAUSES — three inversions, because there are three neighbours
===========================================================================

``registry.py``'s contract is that every family's grid-reason list carries the same
numbered questions and answers exactly one of them THE OTHER WAY. This family sits
between three others and must invert against each; a missed inversion leaves the
slot UNSELECTED, which is a silent coverage loss rather than an error.

**CLAUSE 5 (the fold) — INVERTED against every UNFOLDED family.**
``coverage._grid_reasons`` clause 5 (coverage.py:168-174) refuses a fold by name and
covers ``pml_curl`` / ``constitutive``; ``offdiag_update_e`` calls that helper
UNMODIFIED (offdiag_update_e.py's grid clauses), so the certified off-diagonal arm
refuses every fold; ``bfast_curl``, ``complex_fields`` and ``special_kz`` refuse it
again, each by name. HERE IT IS REQUIRED —
:func:`folded_offdiag_composition_coverage` appends "no mirror plane is active" when
the grid carries none, so this arm and the certified off-diagonal arm can never both
admit ``update_E``.

**CLAUSE b (the live row slot) — INVERTED against the FOLDED-DIAGONAL family.**
:func:`symmetry.folded_constitutive_coverage` (side ``E``) refuses when
``fields.has_offdiagonal_epsilon``; this predicate REQUIRES at least one surviving
row SLOT. THE TWO ARE DISJOINT THROUGH AN ENGINE COUPLING RATHER THAN THROUGH AN
INVERTED CLAUSE, exactly as ``constitutive`` and ``offdiag_update_e`` already are:
the flag is a read-only property over the very dict the row accessor reads, so the
installer's zero-row drop (fields.py:1302-1303) is what keeps them in step. That
coupling is an ENGINE fact, not a composition fact, so it is not asserted — the
composition matrix carries a PLANTED configuration (flag False, one slot live, on a
FOLDED grid) whose expected outcome is that ``update_E`` is left UNSELECTED with
both claimants named. Requiring the flag here instead would look tidier and would be
worse: the builder needs a live SLOT, and a predicate that admitted on the flag
would hand the builder a configuration it raises on.

**CLAUSE 5 AGAIN, THE OTHER WAY — against the folded CURL arms.** Those hold
``step_B``/``step_D`` and this one holds ``update_E``; different slots, so no
inversion is needed and none is claimed. The slot is the separation and it is stated
so a reader does not go looking for a clause.

**THE ABSORBER — a second, independent separation against ``no_pml_constitutive``.**
That family requires an INACTIVE absorber and this one an ACTIVE one, on the same
two constitutive slots. It carries no fold clause at all (correctly: ``update_E``
returns at stepping.py:983-984 before reading an array), which is why a folded no-PML
run selects the null pair and not this family.

**REFUSED BY NAME, each naming the family that owns it:** complex64 storage (the
folded complex family's), a nonzero ``beta`` (the folded special_kz family's), BFAST,
a registered susceptibility (the folded dispersive family's — its source is
``D - sum P``, not ``D``), chi2/chi3, a nonzero ``k_point`` on any axis, and
cylindrical coordinates folded or not.

**FOLD x CYLINDRICAL IS A THREE-WAY HAZARD** and is refused TWICE on purpose.
``_boundary_kinds`` puts ``is_axis`` AHEAD of ``is_mirrored`` (stepping.py:2191-2196),
so a folded r axis would report CYL_AXIS and the fold would vanish silently; and
``_mirror_phases`` (stepping.py:2346-2366) puts ``(-1)**grid.m`` into the SAME SLOT
the mirror phase occupies, so a predicate that reads a phase without first refusing
``is_axis`` reads a cylindrical m-factor as a mirror parity. The grid flag alone is
not the inversion; the per-axis clause is there too.

**THE STANDALONE PREDICATE IS NOT THE ROUTING PREDICATE.**
:func:`folded_offdiag_constitutive_coverage` deliberately ADMITS an unfolded grid so
a gate can prove the reduction to the certified kernel; registering THAT verdict
would make every unfolded off-diagonal row ambiguous. The narrower
:func:`folded_offdiag_composition_coverage` is what :func:`register_arms` registers
— the same split :func:`symmetry.folded_composition_curl_coverage` makes, for the
same reason: selecting between two valid products by branch order would make the
numerical method depend on composer order.

**REGISTERED-BUT-CANNOT-WIN IS NOT USED HERE**, and the absence is stated rather
than left to be noticed. ``registry.py`` keeps ``registered`` and ``wins`` separate
so a missing capability gets NAMED at composition time instead of falling silently
to the array path; this family has no such gap — its one slot is fully carried on
both fold terminations, at both parities, in 2-D and 3-D. The gaps it DOES have are
refusals by name (a registered susceptibility, complex storage, beta), which is the
same mechanism one level up.

===========================================================================
ARRAY-PATH FINDING — reported, not fixed here
===========================================================================

On a folded PERIODIC axis, ``_offdiagonal_terms``' own-axis ``_shift_up`` serves an
exact ZERO past the stored top (point 2 above), where MEEP's ghost is the
parity-weighted image of ``_far_reflect_rows``' row. The guard that would refuse it,
``FdtdDriver._require_folded_far_face_is_quiet`` (driver.py:2986-3038), returns
immediately unless ``fields.has_nonlinearity``, so an off-diagonal folded run with a
live far face is NOT refused; nor is it pinned
(``test_tensor_epsilon.py::test_tensor_fold_equivalence_is_exact`` trims the far row
with ``[:-1]``, test_tensor_epsilon.py:554). THIS KERNEL REPRODUCES THE ARRAY PATH,
byte for byte, either way. The finding belongs to ``stepping.py`` and
``driver.py``, and the Triton twin records it at
``triton_kernels/folded_offdiag_update_e.py:276-299``.

It is UNREACHABLE ON EVERY MEASURED CORPUS ROW — a fact INHERITED from the
predicate-coverage census (``parity/meep_gpu/results/predicate_coverage_2026-08-14_
allnine/``, re-cut against this family's arm this round only insofar as the
composition sweep re-ran; the per-row BC triples are that census's numbers, not this
round's): all 19 folded off-diagonal rows are folded METALLIC, over the two triples
``(METALLIC, MIRROR_METALLIC, PERIODIC)`` x10 and
``(MIRROR_METALLIC, MIRROR_METALLIC, PERIODIC)`` x9, where the zero ghost is the
array path's own exact value. Not one folded PERIODIC off-diagonal row exists there,
so this is a synthetic concern only — which is precisely why the sibling tests carry
a folded PERIODIC case rather than only the corpus's shape. THE R01..R22 ROW-MASK
PATTERN THOSE 19 ROWS DRIVE IS NOT MEASURED and is not guessed here: the coverage
battery records ``has_offdiagonal_epsilon`` and not the surviving slot set, so the
specialisation count is BOUNDED (2 triples x some subset of 63 masks) rather than
known. The sibling suite therefore compiles every row mask over four triples rather
than the corpus's presumed few.

STALE DOCSTRING, REPORTED AND NOT FIXED: stepping.py:1219-1220 still claims "a
folded axis never reaches this function: ``Fields.set_epsilon_volumes`` refuses the
combination at install". It does not — fields.py:1210-1246 installs the rows,
``_validated_offdiagonal_rows`` (fields.py:1263-1310) has no fold clause, and this
whole family exists because it does. :mod:`.offdiag_update_e` already logs the same
finding.

WIRED. One arm, on ``update_E``, gated on the fold AND on the off-diagonal
disjunction so an unfolded or row-free run gets the OTHER family's named refusal
rather than two. ``fastpath.plan_fast_path`` still returns ``None`` on every branch,
which is the separate decision this port does not make.

CERTIFIED BY ``parity/meep_gpu/gate_metal_folded_offdiag.py``, artifact
``parity/meep_gpu/results/metal_folded_offdiag_2026-08-16/``: **362,814 uint32 word
comparisons, 0 differing, exit 0** over seven legs — ``byte`` (14 cases, oracle
moving 864-8,424 words each), ``value_class``, ``whole_step`` (6 cases x 6 complete
steps through the driver's five passes per half, first divergent step reported, every
slot's launch counter asserted), ``reduction`` (504 unfolded (row mask, code triple)
pairs emitting the certified source character for character), ``precondition`` and
``mutations`` — **13 must-catch mutations all caught over 415 mutant launches, 3
declared nulls**, including all five defects a mirror is specifically prone to. The
subnormal precondition is reported as a WINDOW and is DEMONSTRATED TO FIRE: clean at
1.0 and 1e-30, ``[0, 5]`` at 1e-38 and 1e-40. One case is REFUSED BY NAME rather
than certified — a subnormal ghost plane, which diverges in 32 words — and that
refusal is the coverage boundary this backend's native flush imposes, not a defect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import arms, offdiag_update_e as _offdiag, shaders, symmetry as _symmetry
from . import templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import compile_source
from .plans import KernelPlan
from ..triton_kernels.coverage import (
    Coverage,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
)

#: The registry family name. One spelling, so a refusal message, a registry row and
#: an artifact column cannot drift apart.
FAMILY = "folded_offdiag_constitutive"

#: The slot this family claims. One, and only one: the fold's curl arms live in
#: :mod:`.symmetry` and the separation there is the SLOT, not a clause.
SLOT = "update_E"

#: The arm label the composer reports.
LABEL = "folded offdiag"

# ---------------------------------------------------------------------------
# Facts imported by VALUE from the two parent families
# ---------------------------------------------------------------------------
#
# Every one of these already has a home. Re-spelling any of them here would be a
# second place to get the same fact wrong, and the two places would then have to be
# kept in step by a reader rather than by the interpreter. A sibling test asserts
# each identity so an import that silently became a copy is caught.

#: Targets, sources and each component's OWN axis (offdiag_update_e.py:157-158).
E_TERMS: Tuple[Tuple[str, str, int], ...] = _offdiag.E_TERMS

#: The coupling partners in MEEP's ``cycle_direction`` order (offdiag_update_e.py:165).
TRANSVERSE_PARTNERS: Tuple[Tuple[int, int], ...] = _offdiag.TRANSVERSE_PARTNERS

#: The six coefficient slots in plan/kernel argument order (offdiag_update_e.py:170).
ROW_SLOTS: Tuple[Tuple[str, str], ...] = _offdiag.ROW_SLOTS

#: Per component, the axes whose metallic wall plane the coupling mask zeroes.
WALL_MASK_AXES: Tuple[Tuple[int, int], ...] = _offdiag.WALL_MASK_AXES

#: The Yee sub-lattice this side reads: half-integer (stepping.py:1015).
HALF_INTEGER: bool = _offdiag.HALF_INTEGER

#: 24 buffers + 4 scalars. THE FOLD ADDS NONE — no reflect row (the own-axis up
#: shift is not parity-weighted) and no ghost weight (it is baked into the source).
BINDING_COUNT: int = _offdiag.BINDING_COUNT

#: The four ghost-rule codes, :mod:`.symmetry`'s own values, so a plan built here
#: and one built there index the SAME table.
CODE_PERIODIC = _symmetry.CODE_PERIODIC
CODE_METALLIC = _symmetry.CODE_METALLIC
CODE_MIRROR_METALLIC = _symmetry.CODE_MIRROR_METALLIC
CODE_MIRROR_PERIODIC = _symmetry.CODE_MIRROR_PERIODIC
MIRROR_CODES: Tuple[int, int] = _symmetry.MIRROR_CODES

#: The stored row a mirror ghost images — MEEP's halved origin ``io = -2`` maps the
#: ghost at -1 onto stored cell 2 (stepping.py:156-160, ``_mirror_source`` :1582-1588,
#: which RAISES below three stored cells).
MIRROR_SOURCE_INDEX: int = _symmetry.MIRROR_SOURCE_INDEX

__all__ = [
    "ARM",
    "BINDING_COUNT",
    "CODE_METALLIC",
    "CODE_MIRROR_METALLIC",
    "CODE_MIRROR_PERIODIC",
    "CODE_PERIODIC",
    "E_TERMS",
    "FAMILY",
    "LABEL",
    "MIRROR_CODES",
    "MIRROR_SOURCE_INDEX",
    "MetalFoldedOffdiagConstitutivePlan",
    "ROW_SLOTS",
    "SLOT",
    "compile_folded_offdiag",
    "corpus_digest",
    "folded_offdiag_composition_coverage",
    "folded_offdiag_constitutive_coverage",
    "folded_offdiag_source",
    "mirror_arm_is_reachable",
    "mirror_ghost_weights",
    "negated_axes",
    "plan_folded_offdiag_constitutive",
    "plan_folded_offdiag_constitutive_from_arrays",
    "register_arms",
    "specialisations",
]


# ---------------------------------------------------------------------------
# The parity, and the two host questions it answers
# ---------------------------------------------------------------------------

def mirror_ghost_weights(grid: Any) -> Tuple[float, float, float]:
    """The mirror ghost weight per axis: exactly ``-phase``, or 1.0 where unfolded.

    ``stepping._offdiagonal_terms`` (:1214-1216) passes
    ``component = "D" + AXIS_NAMES[partner_axis]`` on ``axis = partner_axis`` and the
    plane's declared phase, and ``_shift_down``'s MIRROR branch (:1823-1826) weights
    the ghost by ``fields.mirror_parity(...)``. Every D component has Yee shift 1 on
    its OWN axis (fields.py:214-219), and
    ``mirror_parity(c, axis, phase) == phase * (1 - 2*iyee[c][axis])`` therefore
    collapses to ``-phase`` for EVERY partner and EVERY axis. One signed scalar per
    folded axis is the whole parity input to this sub-step.

    DERIVED THROUGH ``fields.mirror_parity`` ITSELF rather than by restating the
    collapse, so the two cannot drift; a sibling test asserts the collapse over all
    three axes and both phases.

    NOTE THE POLARITY, because it inverts the reading a reader arrives with: an EVEN
    plane (``phase = +1``) gives weight ``-1`` and therefore the NEGATED source arm,
    while an ODD plane (``phase = -1``) gives weight ``+1`` and reduces to the
    CERTIFIED term text verbatim. The odd mirror is the cheap one here.

    ``float('nan')`` marks an unreadable plane. It is a REFUSAL, not a default: the
    predicate names it and the plan builder refuses before any source is emitted.
    """
    try:  # pragma: no cover - the engine is always importable in practice
        from ..fields import mirror_parity  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - unreadable means the predicate refuses
        mirror_parity = None  # type: ignore[assignment]
    out: List[float] = []
    for axis in range(3):
        if not bool(_call(grid, "is_mirrored", axis, default=False)):
            out.append(1.0)
            continue
        phase = _call(grid, "mirror_phase", axis, default=None)
        if phase not in (1, -1) or mirror_parity is None:
            out.append(float("nan"))
            continue
        out.append(float(mirror_parity("D" + "xyz"[axis], axis, int(phase))))
    return (out[0], out[1], out[2])


def negated_axes(boundary_codes: Sequence[int],
                 ghost_weights: Sequence[float]) -> Tuple[int, int, int]:
    """Which axes emit the NEGATED ghost lane — the source specialisation triple.

    THE ONE DERIVATION, so the index redirect and the sign cannot disagree. The
    redirect keys off the boundary CODE and the sign off the WEIGHT, and a
    disagreement would read stored row 2 without the parity, or apply the parity to
    an ordinary neighbour: a plane of wrong values, not a crash. The plan validates
    the pair against the codes; a sibling test pins the derivation.

    An axis is negated iff it carries a mirror code AND its weight is exactly
    ``-1.0``. An unfolded axis is never negated whatever its weight says — the ghost
    lane does not exist there and the weight is never read.
    """
    codes = tuple(int(code) for code in boundary_codes)
    weights = tuple(float(weight) for weight in ghost_weights)
    if len(codes) != 3 or len(weights) != 3:
        raise ValueError(f"codes and weights are per-axis triples, got {codes!r} "
                         f"and {weights!r}")
    return tuple(  # type: ignore[return-value]
        int(code in MIRROR_CODES and weight == -1.0)
        for code, weight in zip(codes, weights))


def mirror_arm_is_reachable(grid: Any, fields: Any) -> Tuple[bool, Tuple[str, ...]]:
    """Can the mirror ghost arm change a byte for THIS (grid, rows) pair?

    THE TEST IS AT SLOT LEVEL, NOT ROW LEVEL, and the difference is a measured
    counterexample rather than a nicety. The mirror rule on axis ``a`` enters this
    sub-step ONLY through the DOWN shift of the PARTNER axis (``_shift_down`` is
    called with ``axis = partner_axis``, stepping.py:1243-1245), so a fold on ``a``
    is byte-visible iff some SURVIVING row slot takes ``E_a`` AS ITS PARTNER. A row
    can be live through its OTHER slot and never touch the fold: the Triton track
    measured fold X with the single live slot ``Ey <- Ez`` giving MIRROR bytes ==
    METALLIC bytes, while a row-level test ("some live row is not ``Ex``") answers
    reachable because the live row IS ``Ey``.

    Returns ``(reachable, notes)``. NOT a coverage clause and deliberately not a
    routing rule — routing by predicate order would make the numerical method a
    function of composer order. It is what a gate's identity leg asserts against,
    and what a later round may use to prefer the certified kernel where the two are
    measured equal.
    """
    notes: List[str] = []
    rows = _offdiag.row_volumes_for(fields)
    live_slots = tuple((row, partner) for (row, partner), value
                       in zip(ROW_SLOTS, rows) if value is not None)
    reachable = False
    for axis in range(3):
        if not bool(_call(grid, "is_mirrored", axis, default=False)):
            continue
        own = "E" + "xyz"[axis]
        crossing = sorted(f"{row}<-{partner}" for row, partner in live_slots
                          if partner == own)
        if crossing:
            reachable = True
            notes.append(
                f"axis {axis} is folded and is the PARTNER axis of live slot(s) "
                f"{crossing}: the mirror ghost is byte-visible there")
        else:
            live = sorted(f"{row}<-{partner}" for row, partner in live_slots)
            notes.append(
                f"axis {axis} is folded but no live row slot takes {own} as its "
                f"partner (live slots {live}): the mirror arm is unreachable and "
                f"this axis is byte-identical to the certified kernel's METALLIC "
                f"code")
    return reachable, tuple(notes)


# ---------------------------------------------------------------------------
# The shader — the certified template, two specialised blocks
# ---------------------------------------------------------------------------

def _folded_two_way_ghost(axis: str, code: int) -> str:
    """One axis's ghost rule, with the two MIRROR terminations added.

    A PERIODIC or METALLIC axis returns ``offdiag_update_e._two_way_ghost``'s text
    UNCHANGED — the certified emitter is CALLED, so an edit to the certified ghost
    rule reaches this family without a second edit and the reduction claim is a
    property of the code rather than of a comment.

    A MIRROR axis (either termination) differs from METALLIC in exactly two ways,
    and both are transcribed:

    * DOWN is REDIRECTED, NOT MASKED. ``stepping._shift_down``'s MIRROR branch
      (:1818-1826) serves ``parity * field[MIRROR_SOURCE_INDEX]`` at face 0 — a live
      interior plane — so the validity flag stays true and the index moves. The
      PARITY is not here: it belongs to the term, because only the partner-axis role
      of an axis carries it (:func:`_folded_term`).
    * UP IS THE METALLIC ARM, EXACTLY. ``_offdiagonal_terms`` calls ``_shift_up``
      without ``component``/``reflect_row`` (stepping.py:1248-1249), so both mirror
      terminations take stepping.py:1828-1830, an exact ``0.0``. The emitted text is
      the METALLIC one's up half, character for character.

    The face-0 predicate is spelled ``(i == 0)`` inline rather than as ``at_x``
    because the certified TEMPLATE declares ``at_x``/``at_y``/``at_z`` AFTER the
    ghost blocks and this family reuses that template unedited.
    """
    code = int(code)
    if code not in MIRROR_CODES:
        return _offdiag._two_way_ghost(axis, code)
    down, up, _dvalid, uvalid, extent = _offdiag._TWO_WAY_NAMES[axis]
    home = _offdiag._HOME["xyz".index(axis)]
    return (f"    {down} = ({home} == 0) ? {MIRROR_SOURCE_INDEX} : {down};\n"
            f"    {uvalid} = ({up} < {extent});")


def _folded_term(component: int, partner_axis: int, coefficient: str,
                 tag: str) -> str:
    """One partner's OFFDIAG term with a NEGATED mirror ghost on the partner axis.

    ``offdiag_update_e._term``'s expression with the ghost lane's sign flipped::

        0.25*((g[i] + w*g[i-sx])*u[i] + (g[i+s] + w*g[(i+s)-sx])*u[i+s])

    where ``w`` is ``-1`` on the partner axis's face-0 lane and ``1`` everywhere
    else. BOTH ghosted loads take the SAME lane predicate: the own-axis shift does
    not move the partner coordinate, so ``(i+s)`` is at the partner face exactly when
    ``i`` is.

    THE SIGN IS A SELECT ON THE VALUE, ``at ? -x : x``, and not a multiply by a
    weight. Measured on this host (module docstring): the select-on-value and the
    compile-time ``x * -1.0f`` are exact over signed zeros, the whole subnormal band
    and normals; ``0.0f - x`` misses 12/30 INCLUDING 2 of 4 signed zeros, and any
    RUNTIME weight misses every subnormal (10/30), ``(at ? w : 1.0f) * x`` included —
    every element of that probe is on the ghost lane, so what it shows here is that
    the SELECT does not save a runtime weight; that this form additionally flushes
    the NON-ghost lanes is :mod:`.symmetry`'s measurement, not this one's. The
    refuted spellings are sibling-test mutations, not comments.

    The guarded loads are hoisted into named temporaries so the sign applies to the
    LOADED VALUE. Everything else — the coefficient multiply sitting BETWEEN the two
    shifts, the two shifts going in OPPOSITE directions, ``0.25`` scaling the sum
    last, the coefficient never being ghosted (stepping.py:1243-1249 shifts the FIELD
    before the multiply and shifts the own axis AWAY from the fold) — is the
    certified term's, unchanged, because the array path's is unchanged.
    """
    own_axis = E_TERMS[component][2]
    down, _up, dvalid, _uvalid, _extent = _offdiag._TWO_WAY_NAMES["xyz"[partner_axis]]
    _odown, oup, _odvalid, ouvalid, _oextent = _offdiag._TWO_WAY_NAMES["xyz"[own_axis]]
    partner_volume = f"g{partner_axis}"
    at_partner = ("at_x", "at_y", "at_z")[partner_axis]

    home = "ii"
    down_index = _offdiag._index({partner_axis: down})
    up_index = _offdiag._index({own_axis: oup})
    corner_index = _offdiag._index({own_axis: oup, partner_axis: down})
    return "\n".join([
        f"    float dn_{tag} = {dvalid} ? {partner_volume}[{down_index}] : 0.0f;",
        f"    float cn_{tag} = ({ouvalid} && {dvalid})"
        f" ? {partner_volume}[{corner_index}] : 0.0f;",
        f"    float near_{tag} = {partner_volume}[{home}]",
        f"        + ({at_partner} ? -dn_{tag} : dn_{tag});",
        f"    float far_{tag} = ({ouvalid} ? {partner_volume}[{up_index}] : 0.0f)",
        f"        + ({at_partner} ? -cn_{tag} : cn_{tag});",
        f"    float unear_{tag} = {coefficient}[{home}];",
        f"    float ufar_{tag} = {ouvalid} ? {coefficient}[{up_index}] : 0.0f;",
        f"    float term_{tag} = 0.25f * ((near_{tag} * unear_{tag})"
        f" + (far_{tag} * ufar_{tag}));",
    ])


def _component_source(component: int, row_mask: Sequence[int],
                      walls: Sequence[int], negate: Sequence[int]) -> str:
    """The whole ``src{c}`` block for one component, on the arm its rows select.

    REDUCTION IS AT SLOT LEVEL AND IS STRUCTURAL. A component none of whose LIVE row
    slots takes a NEGATED mirror partner returns ``offdiag_update_e._component_source``'s
    text — the certified emitter called, not copied — so the certified arm is
    character-identical rather than similar, and the property is checkable by a test
    instead of by a PTX read.

    That covers three cases at once, and the second and third are the ones a
    row-level reduction would miss:

    * no axis is folded at all;
    * an axis is folded but the component's live slots do not take it as a PARTNER
      (the measured counterexample: fold X with the single live slot ``Ey <- Ez``);
    * an axis is folded and IS a partner, but its plane is ODD, so the weight is
      ``-phase == +1`` and the certified text is already exact.

    Only when a live slot's partner axis is a negated mirror does the folded body
    appear, and then only for that slot's term.
    """
    partners = TRANSVERSE_PARTNERS[component]
    if not any(row_mask[2 * component + offset] and negate[partners[offset]]
               for offset in (0, 1)):
        return _offdiag._component_source(component, row_mask, walls)

    gs = f"    float gs{component} = g{component}[ii];"
    us = f"    float us{component} = e{component}[ii];"
    lines = [gs, us]
    accumulated: List[str] = []
    for offset in (0, 1):
        if not row_mask[2 * component + offset]:
            continue
        partner_axis = partners[offset]
        tag = f"{component}{offset}"
        emit = _folded_term if negate[partner_axis] else _offdiag._term
        lines.append(emit(component, partner_axis,
                          _offdiag._ROW_BUFFERS[2 * component + offset], tag))
        accumulated.append(tag)

    # `total` accumulates offset 1 then offset 2 (stepping.py:1250) and the wall mask
    # runs BEFORE the row sum (:1252 precedes the add at :1007-1008). Both are the
    # certified emitter's own text, reached by calling it.
    lines.append(f"    float total{component} = term_{accumulated[0]};")
    for tag in accumulated[1:]:
        lines.append(f"    total{component} = total{component} + term_{tag};")
    lines.append(_offdiag._wall_mask(component, walls))
    lines.append(f"    float src{component} = (gs{component} * us{component})"
                 f" + total{component};")
    return "\n".join(lines)


def folded_offdiag_source(row_mask: Sequence[int], codes: Sequence[int],
                          walls: Sequence[int], negate: Sequence[int],
                          contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised ``offdiag_constitutive_step`` source for a folded grid.

    THE TEMPLATE IS ``offdiag_update_e._TEMPLATE`` ITSELF, the module attribute — so
    the kernel name, the 28 bindings, the index decode, the ``at_*`` predicates, the
    coefficient reads and the three PML tails are the certified kernel's by
    construction and cannot drift. Only the four substituted blocks can differ.

    ``row_mask`` is the six-flag :data:`ROW_SLOTS` liveness triple-pair, ``codes``
    the per-axis four-value ghost triple :func:`symmetry.folded_axis_kinds` resolves,
    ``walls`` the per-axis metallic-wall DECLARATION triple (a DIFFERENT question
    from ``codes``, carried separately for the reason :mod:`.offdiag_update_e`
    gives), and ``negate`` the per-axis ghost-sign triple :func:`negated_axes`
    derives. All four are baked into the string, as Triton bakes its constexprs — and
    ``negate`` is baked where Triton leaves it a runtime scalar, which is this
    backend's measured requirement rather than a preference.
    """
    row_mask = tuple(int(flag) for flag in row_mask)
    codes = tuple(int(code) for code in codes)
    walls = tuple(int(flag) for flag in walls)
    negate = tuple(int(flag) for flag in negate)
    if len(row_mask) != len(ROW_SLOTS):
        raise ValueError(f"row_mask must fill the {len(ROW_SLOTS)} slots of "
                         f"ROW_SLOTS, got {row_mask!r}")
    if not any(row_mask):
        raise ValueError(
            "no row slot survives: that configuration is symmetry's folded "
            "constitutive product, and emitting this source for it would overlap "
            "the two families")
    if len(codes) != 3 or len(walls) != 3 or len(negate) != 3:
        raise ValueError(f"codes, walls and negate are per-axis triples, got "
                         f"{codes!r}, {walls!r} and {negate!r}")
    for axis, code in enumerate(codes):
        if code not in (CODE_PERIODIC, CODE_METALLIC) + MIRROR_CODES:
            raise ValueError(
                f"axis {axis} boundary code {code!r} is none of PERIODIC/METALLIC/"
                f"MIRROR_METALLIC/MIRROR_PERIODIC; symmetry.folded_axis_kinds is "
                f"the only supported source of these codes")
        if negate[axis] and code not in MIRROR_CODES:
            raise ValueError(
                f"axis {axis} is marked negated with boundary code {code!r}: the "
                f"ghost lane exists only on a mirror axis, so a sign there would "
                f"be applied to an ordinary neighbour")
        if walls[axis] and code in MIRROR_CODES:
            raise ValueError(
                f"axis {axis} is folded AND wall-masked; "
                f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                f"(stepping.py:1282) and zeroing the fold plane costs 2.0e-02 "
                f"(stepping.py:1266-1277)")
    return templates.substitute(_offdiag._TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__GUARD__": templates.GUARD,
        "__GHOST_X__": _folded_two_way_ghost("x", codes[0]),
        "__GHOST_Y__": _folded_two_way_ghost("y", codes[1]),
        "__GHOST_Z__": _folded_two_way_ghost("z", codes[2]),
        "__SRC0__": _component_source(0, row_mask, walls, negate),
        "__SRC1__": _component_source(1, row_mask, walls, negate),
        "__SRC2__": _component_source(2, row_mask, walls, negate),
    })


def compile_folded_offdiag(row_mask: Sequence[int], codes: Sequence[int],
                           walls: Sequence[int], negate: Sequence[int],
                           contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (rows, codes, walls, negate, mode).

    The entry point NAME is the certified kernel's, because the source is the
    certified template: one kernel name for the whole off-diagonal surface.
    """
    return compile_source(
        folded_offdiag_source(row_mask, codes, walls, negate, contract)
    ).offdiag_constitutive_step


# ---------------------------------------------------------------------------
# The specialisation corpus and its fingerprint
# ---------------------------------------------------------------------------

#: Every per-axis (code, wall, negate) state this emitter accepts. A wall is
#: possible only on a METALLIC axis — ``wall_mask_axes`` asks
#: ``is_metallic and not is_mirrored`` and ``_boundary_kinds`` resolves a declared
#: metallic axis to METALLIC — and a negated ghost only on a mirror axis, so the
#: inconsistent combinations are not enumerated rather than being enumerated and
#: rejected. SEVEN states per axis.
_AXIS_STATES: Tuple[Tuple[int, int, int], ...] = (
    (CODE_PERIODIC, 0, 0),
    (CODE_METALLIC, 0, 0),
    (CODE_METALLIC, 1, 0),
    (CODE_MIRROR_METALLIC, 0, 0),
    (CODE_MIRROR_METALLIC, 0, 1),
    (CODE_MIRROR_PERIODIC, 0, 0),
    (CODE_MIRROR_PERIODIC, 0, 1),
)


def specialisations() -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...],
                                     Tuple[int, ...], Tuple[int, ...]], ...]:
    """Every (row_mask, codes, walls, negate) quadruple this family can emit.

    Canonically ordered, and DELIBERATELY RESTRICTED TO TRIPLES CARRYING AT LEAST ONE
    MIRROR AXIS: an unfolded triple emits ``offdiag_update_e``'s source exactly, and
    enumerating it here would hash the certified family's surface a second time under
    this family's name. 63 row masks x (7^3 - 3^3) axis triples = 19,908 sources.

    THE ENUMERATION IS WIDER THAN THE REACHABLE SET and the artifact should not be
    read as claiming otherwise. On the measured corpus every folded off-diagonal row
    is 2-D and folded METALLIC over just two BC triples; what the byte legs certified
    is the subset they launched, recorded individually beside this digest.
    """
    out = []
    for mask in range(1, 1 << len(ROW_SLOTS)):
        row_mask = tuple((mask >> bit) & 1 for bit in range(len(ROW_SLOTS)))
        for sx in _AXIS_STATES:
            for sy in _AXIS_STATES:
                for sz in _AXIS_STATES:
                    codes = (sx[0], sy[0], sz[0])
                    if not any(code in MIRROR_CODES for code in codes):
                        continue
                    out.append((row_mask, codes, (sx[1], sy[1], sz[1]),
                                (sx[2], sy[2], sz[2])))
    return tuple(out)


def corpus_digest(contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """One sha256 over every specialisation, in canonical order, plus the count.

    A digest rather than a per-source list, for the reason
    :func:`offdiag_update_e.corpus_digest` gives: a single changed character anywhere
    in the emitter moves it. It is a FINGERPRINT, not a certification.
    """
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    count = 0
    for row_mask, codes, walls, negate in specialisations():
        digest.update(
            folded_offdiag_source(row_mask, codes, walls, negate, contract).encode())
        count += 1
    return {"count": count, "sha256": digest.hexdigest(), "contract": contract}


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------

def _folded_offdiag_grid_reasons(fields: Any, pml: Any, grid: Any,
                                 ) -> Tuple[Optional[Tuple[int, int, int]],
                                            List[str]]:
    """Everything ``coverage._grid_reasons`` refuses EXCEPT the fold, plus the fold's
    own classification.

    RE-STATED, not imported-and-subtracted. Subtracting a reason string from another
    predicate's output would make this file's coverage a function of that file's
    phrasing, and a clause renamed there would silently WIDEN coverage here. Clause
    numbering follows ``coverage._grid_reasons`` so the two read side by side, and a
    sibling test runs both lists over a shared configuration matrix and asserts they
    agree once each side's fold clauses are removed.

    Deliberately NOT :func:`symmetry._folded_curl_grid_reasons` either: that helper
    is the folded CURL's contract and refuses conductivity, which changes the curl
    recurrence and nothing in ``update_E`` — ``update_E`` reads no ``condfac_for``
    at all. Inheriting a curl-only refusal would silently narrow an independent
    sub-step.
    """
    reasons: List[str] = []

    # 1. The Metal backend, the host array module the mirrors copy from, and the
    #    resolved subnormal policy. IMPORTED: one backend clause for the package.
    reasons.extend(_metal_backend_reasons(grid))

    # 2. Real storage. Under complex64 the fold's parity stops being a sign flip and
    #    becomes a full complex multiply by (+/-1, +0) with its zero cross terms, and
    #    the row product's operand order becomes byte-visible. Both belong to the
    #    folded complex family, composed with the off-diagonal one, which is a later
    #    leg on either track.
    if getattr(fields, "force_complex_fields", False):
        reasons.append(
            "force_complex_fields=True: under complex storage the fold's parity is "
            "a full complex multiply with zero cross terms, not a sign flip; folded "
            "complex + off-diagonal is a later composed leg")

    # 3. An absorber that actually absorbs. Without one `update_E` takes the plain
    #    `field[...] = constitutive` branch (stepping.py:1022), a different sub-step,
    #    and `no_pml_constitutive` is the family that owns it. THIS IS THE SECOND,
    #    INDEPENDENT SEPARATION from the null pair on this slot.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this kernel implements the split-field "
                       "dsigw path only)")

    # 4/5. The ghost rules INCLUDING the two folded ones. `COVERED_BOUNDARIES` is NOT
    #      consulted and NOT widened — widening that shared tuple would admit a fold
    #      into every plain arm at once. `folded_axis_kinds` REPLACES the clause: it
    #      refuses an unresolvable grid, a folded axis with no owned_cells(), a mirror
    #      phase outside +/-1, a folded axis with too few stored cells for the ghost
    #      to image row MIRROR_SOURCE_INDEX, and a folded axis whose two independent
    #      routes to "periodic or metallic" disagree. Nothing is decided here.
    codes, fold_reasons = _symmetry.folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(fold_reasons)

    # 6. Cartesian only, refused TWICE. `_boundary_kinds` puts `is_axis` AHEAD of
    #    `is_mirrored` (stepping.py:2191-2196), so a folded r axis reports CYL_AXIS
    #    and the fold vanishes silently; and `_mirror_phases` (stepping.py:2346-2366)
    #    puts (-1)**grid.m into the SAME SLOT the mirror phase occupies, so reading a
    #    phase without first refusing `is_axis` reads an m-factor as a parity.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0 on EVERY axis. A folded axis refuses a nonzero k outright
    #    (`stepping._bloch_phases` raises; `driver._require_bloch_is_representable`
    #    refuses the zone edge too), so a kernel cannot lift what the array path will
    #    not run; a Bloch phase on another axis needs complex storage.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. No instantaneous nonlinearity. The Pade factor REPLACES the row product
    #     (stepping.py:999-1000) and, on a fold, the transverse sums shift component
    #     PRODUCTS carrying no single parity — which the driver itself refuses once
    #     the far face is live (`_require_folded_far_face_is_quiet`,
    #     driver.py:2986-3038). Refused here regardless of the far face.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append(
            "chi2/chi3 is installed: the Pade factor replaces the row product and a "
            "nonlinear fold has no single parity")

    # 11/12. BFAST adds a second additive curl term; beta adds out-of-plane
    #        couplings — and the ENGINE additionally raises for real + offdiag +
    #        beta (stepping.py:800-810, mirroring MEEP fields.cpp:548-549). Both are
    #        silent additions, not errors, so both are refused by name.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    if codes is None and not reasons:  # pragma: no cover - belt and braces
        reasons.append("the per-axis boundary codes could not be resolved")
    return codes, reasons


def folded_offdiag_constitutive_coverage(fields: Any, pml: Any,
                                         residency: Any = None) -> Coverage:
    """May the Metal folded off-diagonal kernel step ``update_E``? THE WIDE VERDICT.

    POSITIVE CLAUSES ONLY; a failing clause appends its reason and the scan
    continues, so a refusal reports everything that disqualified the run rather than
    the first thing.

    ZERO FOLDED AXES IS ADMITTED HERE, and that is not an oversight: with every axis
    resolving to PERIODIC/METALLIC, :func:`folded_offdiag_source` returns
    ``offdiag_update_e.offdiag_source``'s string EXACTLY, which is a property a
    sibling test measures rather than a claim. THIS VERDICT MUST NOT BE REGISTERED ON
    A SLOT — it would make every unfolded off-diagonal row ambiguous.
    :func:`folded_offdiag_composition_coverage` is the routing verdict.

    E-side clauses, each a silent wrong answer if missing — the certified
    off-diagonal predicate's list, restated:

    a. **no registered polarization.** With poles the source is ``D - sum P`` formed
       in per-component scratch buffers (fields.py:1107-1138) and the coupling would
       read THOSE, not D. A later fused leg.
    b. **at least one surviving off-diagonal row SLOT (INVERTED against
       :func:`symmetry.folded_constitutive_coverage`)**, counted over the six
       ROW_SLOTS — deliberately not the bare ``has_offdiagonal_epsilon`` flag, which
       a row planted past the installer under a diagonal key sets while every slot is
       dead, and the builder would then raise where this predicate had admitted.
    c. **stored E** — forced True by any surviving row (fields.py:1254-1255); belt
       and braces with a reason.
    d. every surviving row volume real, f32, C-contiguous, grid-shape, and aliasing
       neither an E/f_w OUTPUT nor a D SOURCE — ``offdiag_update_e._row_reasons``,
       IMPORTED rather than restated because it is the same clause about the same
       installer, and because the SOURCE-alias hazard it refuses is a measured,
       composition-only defect (19,620 differing words over two complete steps) that
       no single sub-step can see.
    e. volume inverse epsilon per component; this kernel carries NO scalar arm.
    f. layout and the half-integer coefficient tables AT THE FOLDED STORED EXTENT —
       the clause that stops "the coefficient vectors are built at the stored length"
       from being an assumption, since ``_coefficient_reasons`` compares each
       vector's length against ``grid.shape[axis]``, which on a folded axis IS the
       stored count.
    g. **a readable mirror ghost weight of exactly +/-1 on every folded axis.**
       ``folded_axis_kinds`` already refuses a phase outside +/-1; this clause names
       the value that decides the SOURCE, because an unreadable plane must not select
       a specialisation by accident.
    h. the RESIDENCY declaration (the Metal-only clause: two sub-steps that mirror
       one volume separately each hold a private copy, and the second launch reads
       the first one's stale bytes).

    Conductivity is NOT a clause, deliberately: it changes the CURL sub-steps only.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    codes, reasons = _folded_offdiag_grid_reasons(fields, pml, grid)
    reasons.extend(_susceptibility_reasons(fields))
    reasons.extend(_residency_declaration_reasons(residency))

    # (a) No poles: the coupling must read the aliased D primaries.
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if states or getattr(fields, "has_polarizations", False):
        reasons.append(
            "a susceptibility is registered: the offdiag source becomes "
            "D - sum P in per-component scratch buffers (fields.py:1107-1138) — "
            "the fused dispersive + offdiag + fold kernel is a later leg")

    # (b) INVERTED against symmetry's folded constitutive: a live row SLOT.
    if not any(value is not None for value in _offdiag.row_volumes_for(fields)):
        reasons.append(
            "no off-diagonal chi1inv row survived installation: that configuration "
            "is symmetry.folded_constitutive_coverage(side='E')'s and this "
            "predicate must not overlap it")

    # (c) Stored E.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # (d) The surviving rows, readable and well-formed — the certified clause.
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_offdiag._row_reasons(fields, shape))

    # (e)/(f) The volumes this sub-step reads and writes, and their layout.
    names = tuple(term[0] for term in E_TERMS)
    names += tuple("f_w_" + term[0] for term in E_TERMS)
    names += tuple(term[1] for term in E_TERMS)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_layout_reasons(fields, shape, names))
    if len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
        if pml is not None and getattr(pml, "is_active", False):
            reasons.extend(_coefficient_reasons(
                pml, shape, ("kps", "kms"), ("_h",) if HALF_INTEGER else ("",)))

    # (g) The ghost weights that DECIDE THE SOURCE. On this backend the sign is a
    #     compile-time specialisation, so an unreadable plane cannot be allowed to
    #     pick one: it is refused by name here and the builder never runs.
    for axis, weight in enumerate(mirror_ghost_weights(grid)):
        if weight != weight or weight not in (1.0, -1.0):
            reasons.append(
                f"axis {axis} mirror ghost weight is {weight!r}, not +1.0 or -1.0; "
                f"the weight is baked into the SOURCE on this backend, so an "
                f"unreadable plane would select a specialisation rather than raise")

    # A folded axis must be able to hold the plane its ghost images.
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= MIRROR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; "
                    f"the mirror ghost images stored row {MIRROR_SOURCE_INDEX}")

    return Coverage(not reasons, tuple(reasons))


def folded_offdiag_composition_coverage(fields: Any, pml: Any,
                                        residency: Any = None) -> Coverage:
    """THE ROUTING VERDICT — the wide one with a fold MANDATORY.

    The clause that separates this family from ``offdiag_update_e`` on ``update_E``,
    spelled as its own function so a reader can see the inversion in one place.
    Registering the wide verdict instead would make every unfolded off-diagonal row
    ambiguous and would leave the numerical method depending on composer order — the
    reason :func:`symmetry.folded_composition_curl_coverage` exists.

    ASKED TWICE and deliberately, exactly as :func:`symmetry._requires_a_fold` asks
    it: once of the GRID (``has_symmetry`` and some ``is_mirrored`` axis) and once of
    the RESOLVED CODES. The two can disagree — a grid can declare a mirror the
    boundary resolver reports as something else, which is precisely the cylindrical
    hazard — and a family that asked only the first would admit a run whose kernel
    carries no fold at all.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    base = folded_offdiag_constitutive_coverage(fields, pml, residency)
    reasons = list(base.reasons)
    codes, _ = _symmetry.folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if not _symmetry._has_real_fold(grid):
        reasons.append(
            "no mirror plane is active: the folded off-diagonal product is an "
            "equivalence product here, not a composition candidate — the certified "
            "offdiag_update_e family owns the unfolded run")
    elif codes is not None and not any(int(code) in MIRROR_CODES for code in codes):
        reasons.append(
            "no axis resolves to a mirror boundary: the grid declares a fold that "
            "stepping._boundary_kinds does not report as one")
    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedOffdiagConstitutivePlan(KernelPlan):
    """A launchable, allocation-free folded off-diagonal ``update_E``.

    A SIBLING of :class:`.offdiag_update_e.MetalOffdiagConstitutivePlan`,
    deliberately not a subclass. The two hold the SAME bindings in the SAME order —
    the fold adds no argument on this backend — but the certified plan's
    ``__init__`` validates its boundary codes against ``{PERIODIC, METALLIC}``, which
    is a correct clause for that family and would have to be widened to inherit. A
    subclass that widened it would make the certified plan accept a mirror code and
    then emit the certified source for it, which is a plane of wrong values rather
    than a refusal. Sibling, therefore, with the ALIAS REFUSAL reached by CALLING the
    certified staticmethod so there is one implementation of that rule.

    The Yee sub-lattice is chosen at the BUILDERS and nowhere else:
    ``kps_a_h``/``kms_a_h``, the half-integer tables (stepping.py:1015). The six ROW
    SLOTS are bound in :data:`ROW_SLOTS` order; a dead slot's pointer is bound to the
    component's own D mirror (never read — the specialisation is what stops the read,
    the buffer still has to bind).

    THE WALL AXES ARE CARRIED SEPARATELY FROM THE BOUNDARY CODES, as in the certified
    plan, and here the separation stops being future-proofing and becomes load
    bearing: within the certified family's admitted space ``walls[a] ==
    (codes[a] == METALLIC)`` was MEASURED to be locked, and a fold is exactly the
    configuration that breaks the lock — a folded METALLIC axis is ``is_metallic``
    True and ``is_mirrored`` True, so the mask abstains while the declaration stands.
    Conflating the two would zero a plane MEEP steps, at a measured cost of 2.0e-02
    (stepping.py:1266-1277). The plan REFUSES the pairing by name.
    """

    __slots__ = ("shape", "n_elem", "row_mask", "boundary_codes", "wall_axes",
                 "ghost_weights", "negate", "residency", "volumes")

    REPR_FIELDS = ("shape", "row_mask", "boundary_codes", "wall_axes", "negate")

    family = "folded off-diagonal constitutive"

    def __init__(self, shape, residency, targets, auxiliaries, sources,
                 inverse_epsilon, rows, coefficients, boundary_codes, wall_axes,
                 ghost_weights, functions: Dict[str, Any], volumes: Sequence[str],
                 host_rows: Sequence[Any] = (),
                 host_arrays: Sequence[Any] = ()) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]

        self.boundary_codes = tuple(int(code) for code in boundary_codes)
        if len(self.boundary_codes) != 3 or any(
                code not in (CODE_PERIODIC, CODE_METALLIC) + MIRROR_CODES
                for code in self.boundary_codes):
            raise ValueError(
                f"boundary codes must be three of {{{CODE_PERIODIC}, "
                f"{CODE_METALLIC}, {CODE_MIRROR_METALLIC}, {CODE_MIRROR_PERIODIC}}}, "
                f"got {boundary_codes!r}")
        self.wall_axes = tuple(int(flag) for flag in wall_axes)
        if len(self.wall_axes) != 3 or any(flag not in (0, 1)
                                           for flag in self.wall_axes):
            raise ValueError(f"wall axes must be three of {{0, 1}}, got "
                             f"{wall_axes!r}")
        for axis, (code, flag) in enumerate(zip(self.boundary_codes,
                                                self.wall_axes)):
            if code in MIRROR_CODES and flag:
                raise ValueError(
                    f"axis {axis} is folded AND wall-masked; "
                    f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                    f"(stepping.py:1282) and zeroing the fold plane costs 2.0e-02 "
                    f"(stepping.py:1266-1277)")

        self.ghost_weights = tuple(float(weight) for weight in ghost_weights)
        if len(self.ghost_weights) != 3:
            raise ValueError(f"ghost weights must be three floats, got "
                             f"{ghost_weights!r}")
        for axis, (code, weight) in enumerate(zip(self.boundary_codes,
                                                  self.ghost_weights)):
            if code in MIRROR_CODES and weight not in (1.0, -1.0):
                raise ValueError(
                    f"axis {axis} is folded with ghost weight {weight!r}; the "
                    f"mirror ghost carries mirror_parity('D'+axis, axis, phase) == "
                    f"-phase, exactly +1.0 or -1.0 (fields.py:117-182, "
                    f"stepping.py:1870-1873)")
        # ONE derivation of the sign, and it is checked against the codes here for
        # the reason the Triton twin checks its MG constexprs: the index redirect
        # keys off the CODE and the sign off the WEIGHT, and a disagreement reads
        # stored row 2 without the parity, or applies the parity to an ordinary
        # neighbour.
        self.negate = negated_axes(self.boundary_codes, self.ghost_weights)

        # The mirror ghost images stored row MIRROR_SOURCE_INDEX, so a folded axis
        # needs more than that many cells — `stepping._mirror_source` (:1535-1541)
        # raises, and a kernel would silently read a neighbour's plane.
        for axis, code in enumerate(self.boundary_codes):
            if code in MIRROR_CODES and self.shape[axis] <= MIRROR_SOURCE_INDEX:
                raise ValueError(
                    f"axis {axis} is folded with {self.shape[axis]} stored cells; "
                    f"the mirror ghost images stored row {MIRROR_SOURCE_INDEX}")

        rows = tuple(rows)
        if len(rows) != len(ROW_SLOTS):
            raise ValueError(f"rows must fill the {len(ROW_SLOTS)} slots of "
                             f"ROW_SLOTS (None where dead), got {len(rows)}")
        self.row_mask = tuple(int(value is not None) for value in rows)
        if not any(self.row_mask):
            raise ValueError(
                "no row slot survives: that configuration belongs to "
                "symmetry.plan_folded_constitutive, and building this plan for it "
                "would overlap the two")
        self.residency = residency
        self.volumes = tuple(volumes)

        # ONE implementation of the alias rule, reached by calling the certified
        # one: outputs must be distinct, no input may alias an output, and no
        # read-only input may alias a written SOURCE (the measured composition-only
        # defect — identical for one sub-step, 19,620 differing words over two).
        _offdiag.MetalOffdiagConstitutivePlan._require_no_aliasing(
            host_arrays, host_rows)

        component_of_slot = tuple(
            next(index for index, term in enumerate(E_TERMS) if term[0] == row)
            for row, _partner in ROW_SLOTS)
        bound_rows = tuple(
            value if value is not None else tuple(sources)[component_of_slot[index]]
            for index, value in enumerate(rows))

        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(inverse_epsilon) + bound_rows + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem))


def _functions(row_mask, codes, walls, negate,
               contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_folded_offdiag(row_mask, codes, walls, negate, mode)
            for mode in contract_variants}


def plan_folded_offdiag_constitutive(
        fields: Any, pml: Any, residency: Any = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        ) -> Optional[MetalFoldedOffdiagConstitutivePlan]:
    """Build the folded off-diagonal ``update_E`` plan from the engine's own objects.

    None means REFUSED, and the reasons come from
    :func:`folded_offdiag_constitutive_coverage`. The codes come from
    :func:`symmetry.folded_axis_kinds` and from nowhere else — building them from
    ``_boundary_kinds`` alone would lose the MIRROR_METALLIC / MIRROR_PERIODIC split,
    and while that split cannot reach THIS kernel (the docstring's predicted null) it
    carries the cross-check that refuses a grid whose two routes to the termination
    disagree.

    The sources are taken through ``displacement_minus_polarization_volumes`` — the
    accessor the array path itself reads (stepping.py:996, :1004) — which with no
    poles admitted hands back the aliased D primaries (fields.py:1107-1138).
    """
    if not folded_offdiag_constitutive_coverage(fields, pml, residency).covered:
        return None
    codes, _reasons = _symmetry.folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    weights = mirror_ghost_weights(fields.grid)
    walls = _offdiag.wall_mask_axes(fields.grid)
    negate = negated_axes(codes, weights)

    targets = tuple(term[0] for term in E_TERMS)
    volumes = fields.displacement_minus_polarization_volumes()
    rows = _offdiag.row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)

    host_targets = [getattr(fields, n) for n in targets]
    host_aux = [getattr(fields, "f_w_" + n) for n in targets]
    host_sources = [volumes[n] for n in targets]
    host_inv = [fields.inverse_epsilon_for(n) for n in targets]
    host_coefficients = [getattr(pml, f"{stem}_{axis}_h")
                         for axis in "xyz" for stem in ("kps", "kms")]

    mirrored_targets = [residency.mirror(n, a)
                        for n, a in zip(targets, host_targets)]
    mirrored_aux = [residency.mirror("f_w_" + n, a)
                    for n, a in zip(targets, host_aux)]
    mirrored_sources = [residency.mirror(term[1], a)
                        for term, a in zip(E_TERMS, host_sources)]
    mirrored_inv = [residency.mirror("inv_eps_" + n, a, constant=True)
                    for n, a in zip(targets, host_inv)]
    mirrored_rows = [
        None if value is None
        else residency.mirror(f"chi1inv_offdiag:{row}:{partner}", value,
                              constant=True)
        for (row, partner), value in zip(ROW_SLOTS, rows)]
    mirrored_coefficients = [
        residency.mirror(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"),
                         constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    volume_names = (tuple(targets) + tuple("f_w_" + n for n in targets)
                    + tuple(term[1] for term in E_TERMS))
    return MetalFoldedOffdiagConstitutivePlan(
        fields.grid.shape, residency, mirrored_targets, mirrored_aux,
        mirrored_sources, mirrored_inv, mirrored_rows, mirrored_coefficients,
        codes, walls, weights,
        _functions(row_mask, codes, walls, negate, contract_variants),
        volume_names,
        host_rows=rows,
        host_arrays=(host_targets, host_aux, host_sources, host_inv,
                     host_coefficients))


def plan_folded_offdiag_constitutive_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any],
        rows: Dict[str, Dict[str, Any]], boundary_codes: Sequence[int],
        wall_axes: Sequence[int], ghost_weights: Sequence[float], residency: Any,
        functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        ) -> MetalFoldedOffdiagConstitutivePlan:
    """Build it from bare host arrays — the gate's and the sibling tests' route.

    ``arrays`` is keyed by component name (``Ex``, ``f_w_Ex``, ``Dx``,
    ``inv_eps_Ex``), ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE CALLER
    already selected, and ``rows`` maps row component -> partner component ->
    coefficient volume, with absent entries marking dead slots (the caller performs
    the install-time zero-row drop itself). No coverage predicate runs here: the
    caller is a harness that constructed the configuration on purpose, including the
    deliberately wrong ones.

    ``functions=`` IS LOAD-BEARING: the mutation legs route a mutated kernel through
    it, and a builder that drops it launches the SHIPPED kernel and reports a pass
    for a defect it never introduced.
    """
    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    row_values = tuple((rows.get(row) or {}).get(partner)
                       for row, partner in ROW_SLOTS)
    row_mask = tuple(int(value is not None) for value in row_values)
    negate = negated_axes(boundary_codes, ghost_weights)

    host_targets = [arrays[n] for n in targets]
    host_aux = [arrays["f_w_" + n] for n in targets]
    host_sources = [arrays[term[1]] for term in E_TERMS]
    host_inv = [arrays["inv_eps_" + n] for n in targets]
    host_coefficients = [flat[f"{stem}_{axis}"]
                         for axis in "xyz" for stem in ("kps", "kms")]

    mirrored_targets = [residency.mirror(n, a)
                        for n, a in zip(targets, host_targets)]
    mirrored_aux = [residency.mirror("f_w_" + n, a)
                    for n, a in zip(targets, host_aux)]
    mirrored_sources = [residency.mirror(term[1], a)
                        for term, a in zip(E_TERMS, host_sources)]
    mirrored_inv = [residency.mirror("inv_eps_" + n, a, constant=True)
                    for n, a in zip(targets, host_inv)]
    mirrored_rows = [
        None if value is None
        else residency.mirror(f"chi1inv_offdiag:{row}:{partner}", value,
                              constant=True)
        for (row, partner), value in zip(ROW_SLOTS, row_values)]
    mirrored_coefficients = [
        residency.mirror(f"pml:{stem}_{axis}:folded_offdiag", flat[f"{stem}_{axis}"],
                         constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    volume_names = (tuple(targets) + tuple("f_w_" + n for n in targets)
                    + tuple(term[1] for term in E_TERMS))
    return MetalFoldedOffdiagConstitutivePlan(
        shape, residency, mirrored_targets, mirrored_aux, mirrored_sources,
        mirrored_inv, mirrored_rows, mirrored_coefficients, boundary_codes,
        wall_axes, ghost_weights,
        functions if functions is not None
        else _functions(row_mask, boundary_codes, wall_axes, negate,
                        contract_variants),
        volume_names,
        host_rows=row_values,
        host_arrays=(host_targets, host_aux, host_sources, host_inv,
                     host_coefficients))


# ---------------------------------------------------------------------------
# WIRING — one arm, on one slot
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    return folded_offdiag_composition_coverage(context.fields, context.pml,
                                               context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedOffdiagConstitutivePlan]:
    return plan_folded_offdiag_constitutive(context.fields, context.pml,
                                            context.residency,
                                            context.contract_variants)


def _has_fold_and_rows(context: Any) -> bool:
    """The cheap gate that decides whether this family is CONSULTED at all.

    BOTH halves, because this family is the INTERSECTION of two others and a
    consulted arm contributes all of its reasons by name. On an unfolded run with
    rows the certified off-diagonal arm speaks; on a folded run without rows the
    folded constitutive arm speaks; in neither case does a reader want this family's
    full refusal list beside theirs. Gated out, it contributes nothing.

    Deliberately the cheapest possible read and never the predicate itself, and
    deliberately NOT a routing decision: the gate decides who is asked, the inverted
    clauses decide who admits. Note that the gate passing is NOT the same question as
    the predicate admitting — the planted flag-false/slot-live configuration passes
    this gate and so does the folded constitutive's, which is how that pair is
    measured to fail closed rather than assumed to be disjoint.
    """
    # THE WHOLE BODY IS GUARDED, not only the row read. `plan_step` evaluates a
    # gate EAGERLY inside `arms.arms_for`, which sits outside the per-arm
    # try/except, so a raising `fields.grid` or `fields.chi1inv_offdiagonal_for`
    # would escape a composer whose whole contract is that it never raises. An
    # unreadable state is NOT a live one: the arm goes silent and the slot falls to
    # the array path, which is always correct.
    try:
        fields = getattr(context, "fields", None)
        grid = getattr(fields, "grid", None)
        if grid is None or not _symmetry._has_real_fold(grid):
            return False
        return any(value is not None
                   for value in _offdiag.row_volumes_for(fields))
    except Exception:  # noqa: BLE001 - unreadable state is not live state
        return False


def register_arms() -> Tuple[Any, ...]:
    """This family's ONE arm: the folded off-diagonal ``update_E``.

    WHAT SEPARATES IT FROM EVERY OTHER ARM ON ``update_E``, each an inverted clause
    rather than an omission. The list is by FAMILY rather than by count, because the
    count changes as families land and a number here would go stale silently:

      ``constitutive``          ``coverage._grid_reasons`` clause 5 refuses a fold by
                                name (two strings: the grid-level one and one per
                                folded axis), and it refuses an off-diagonal row on
                                top of that;
      ``offdiag_constitutive``  the SAME clause 5, reached through the same helper —
                                it refuses every fold. Clause 5 HERE is inverted by
                                :func:`folded_offdiag_composition_coverage`;
      ``folded``                requires NO surviving off-diagonal row (it gates the
                                ELEMENT-WISE body); this family requires one. That
                                pair is disjoint through the installer's zero-row
                                drop rather than through an inverted clause, exactly
                                as ``constitutive``/``offdiag_constitutive`` are, and
                                the composition matrix carries the planted
                                disagreement to prove it fails CLOSED;
      ``no_pml_constitutive``   requires an INACTIVE absorber and this family an
                                ACTIVE one — a second, independent separation;
      ``folded_complex``        requires complex64 STORAGE, which this family refuses
                                by name; both require a fold, so storage is the whole
                                inversion between them;
      ``complex_fields`` /      refuse a fold by name, and this family refuses
      ``special_kz`` /          complex storage, beta and BFAST by name.
      ``bfast_curl``

    REGISTERED-BUT-CANNOT-WIN IS NOT USED, and the absence is deliberate: every
    configuration this family's predicate admits, its plan builds. Its gaps are
    refusals by name, which is the same mechanism one level up.
    """
    return (arms.register(
        family=FAMILY, slot=SLOT, label=LABEL,
        coverage=_arm_coverage, plan=_arm_plan,
        prefix=f"{LABEL}: ", noun="folded off-diagonal constitutive",
        gate=_has_fold_and_rows, wired=True),)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[Any, ...] = register_arms()

#: The single arm, for callers that want it by name rather than by index.
ARM = ARMS[0]
