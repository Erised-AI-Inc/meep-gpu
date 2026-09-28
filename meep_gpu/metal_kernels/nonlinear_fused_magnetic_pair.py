"""The fused magnetic B/H pair on a NONLINEAR run — admission only, no new kernel.

WHAT THIS MODULE ADDS, AND WHAT IT DELIBERATELY DOES NOT. It adds one predicate and
one plan builder. It adds NO arithmetic, NO template and NO emitter: the kernel a
covered configuration compiles here is
:func:`.fused_magnetic_pair.fused_magnetic_pair_source`, character for character,
because on a nonlinear run the B/H seam IS the ordinary one.

THE MEASUREMENT THAT LICENSES THAT, and it is already in the tree rather than being
made here. ``stepping.step_B`` and ``stepping.update_H`` do not read chi2/chi3 — the
Pade factor enters ``update_E`` alone — so :mod:`.nonlinear_update_e` already ships
two SPINE ARMS whose whole content is that fact:

* ``nonlinear_run_pml_curl_coverage`` (nonlinear_update_e.py:1437-1548) delegates to
  ``coverage.pml_curl_coverage`` through :class:`.nonlinear_update_e._LinearScopeView`
  and its plan builder calls ``launch.plan_pml_curl`` — the CERTIFIED ordinary curl;
* ``nonlinear_run_constitutive_coverage`` does the same for ``coverage.constitutive_coverage``
  on side ``"H"``, and its builder calls ``launch.plan_constitutive``.

So the two halves the planner selects at this cell on a nonlinear row are, already
and by construction, the two halves :mod:`.fused_magnetic_pair` welds. This module
is the same one-clause inversion applied ONE LEVEL UP — to the WELD instead of to
each half — and the view it inverts through is IMPORTED from the spine rather than
copied, so a change to that view moves all three arms together.

WHY THAT MAKES THE CONJUNCTION EXACT, not merely plausible.
:func:`.fused_magnetic_pair.metal_fused_magnetic_pair_coverage` is
``pml_curl_coverage(fields, pml, "step_B") AND constitutive_coverage(fields, pml,
"H")`` plus three seam clauses that read the GRID and the SOURCE LIST — neither of
which the view touches. Handing it the view therefore evaluates precisely the
conjunction of the two shipped spine predicates plus those seam clauses. There is no
third thing to certify. :func:`nonlinear_fused_magnetic_pair_equivalence` states that
as a checkable identity and the test module and the device gate both run it.

WHAT THE CORPUS SAYS THIS IS WORTH. Two seam-instances, both reachable, measured at
``parity/meep_gpu/results/below_the_cut_census_2026-08-20/census.json``:
``examples:3rd-harm-1d.py`` and ``tests:Test3rdHarm1d.test_3rd_harm_1d``. Both carry
an ELECTRIC source only, so the magnetic source seam (driver.py:3283-3284) is clear
on both; both are all-periodic in x and y with a METALLIC z boundary, so the inline
``zero_metal_B`` carry this family inherits is LIVE on those rows rather than
compiled out. The D/E partner cell is worth ZERO — both rows inject electrically
between ``step_D`` and ``update_E`` — and is not built.

REGISTERED UNWIRED, for :mod:`.fused_magnetic_pair`'s reason: the product spans three
driver passes and ``plan_step`` assigns at most one arm per slot, so there is no slot
it could claim without a composition rule nothing has measured.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import fused_magnetic_pair, shaders
from .device import Residency
# THE SAME VIEW THE TWO SHIPPED SPINE ARMS INVERT THROUGH, imported rather than
# re-spelled: a second copy could drift from the one the halves are certified under
# and the conjunction below would stop being exact.
from .nonlinear_update_e import _LinearScopeView, _nonlinear_spine_reasons
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? NO, and it may
#: not -- which now has to be SAID here rather than left to the delegation.
#: :func:`.fused_magnetic_pair.metal_fused_magnetic_pair_coverage` declares True, and
#: while it declared False this module could take the seam verdict from it wholesale.
#: It no longer can: what that predicate returns for a magnetic source is an
#: ADMISSION, and this arm may not inherit it, because ``launch.FUSED_PAIR_ARMS``
#: carries a row for ``fused_magnetic_pair`` and for nothing else. This family is
#: refused at the absorb table before its predicate is asked, so
#: ``launch._install_fused_pair`` -- the only builder of ``LeadingRepairPlan`` /
#: ``TrailingRepairPlan`` on this track -- never sees it, and a launch of it would
#: compute the constitutive half against a pre-injection B and report success. So the
#: seam clause is asked AGAIN in the predicate below, against the real fields and with
#: this constant.
#:
#: WHAT IS MISSING IS THE WIRING AND ONLY THE WIRING. The kernel is byte-identical to
#: the welded family's and its constitutive half is just as pointwise -- chi2/chi3
#: enters ``update_E`` alone (``stepping.py:952-990``) and cannot reach the B/H seam.
#: Establishing an absorb declaration for this family is the edit that would license
#: flipping this, and it is a COMPOSITION measurement rather than a repair one.
#:
#: BOTH MEASURED CORPUS ROWS ARE UNAFFECTED: ``examples:3rd-harm-1d.py`` and
#: ``tests:Test3rdHarm1d.test_3rd_harm_1d`` carry an ELECTRIC source only, so the
#: magnetic seam is clear on both and this refusal costs nothing that was ever served.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "nonlinear_fused_magnetic_pair"

#: The slot this arm holds a row on, and the passes one launch performs. Both are
#: the welded family's own, because the welded family's own kernel is what runs.
SLOT = fused_magnetic_pair.SLOT
REPLACES: Tuple[str, ...] = fused_magnetic_pair.REPLACES

#: The binding shape is INHERITED, not restated. This family compiles
#: ``fused_magnetic_pair_source`` unchanged, so a divergence between these numbers
#: and that module's would mean the two had stopped being the same kernel.
PACKED_BINDINGS = fused_magnetic_pair.PACKED_BINDINGS
SEPARATE_SCALAR_BINDINGS = fused_magnetic_pair.SEPARATE_SCALAR_BINDINGS

__all__ = [
    "CARRIES_DEPOSIT_REPAIR",
    "FAMILY", "PACKED_BINDINGS", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "nonlinear_fused_magnetic_pair_equivalence",
    "nonlinear_fused_magnetic_pair_source",
    "metal_nonlinear_fused_magnetic_pair_coverage",
    "plan_metal_nonlinear_fused_magnetic_pair",
]


# ---------------------------------------------------------------------------
# The source — the welded family's, unchanged
# ---------------------------------------------------------------------------

def nonlinear_fused_magnetic_pair_source(
        codes: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """The fused B/H source for a nonlinear run: the ORDINARY one, byte for byte.

    A one-line forward, and it is a one-line forward ON PURPOSE. Spelling a second
    template here — even one produced by copying the first — would put two strings
    in the tree that must stay equal, and nothing would be measuring that they do.
    The device gate's ``transcription`` leg asserts this returns the shipped
    family's own bytes for the same specialisation.
    """
    return fused_magnetic_pair.fused_magnetic_pair_source(codes, zero_metal, contract)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_nonlinear_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_B`` -> wall -> ``update_H`` on a nonlinear run?

    Requires an INSTALLED chi2/chi3 — the shared clause 10 inverted, which is what
    makes this arm disjoint from :mod:`.fused_magnetic_pair`'s in both directions —
    and then delegates EVERY other question to that family's own predicate through
    the spine's view.

    ``_nonlinear_spine_reasons`` is the same inversion the two shipped spine arms
    use, so a run this admits is exactly a run on which both of those arms already
    hold their slots.
    """
    reasons: List[str] = _nonlinear_spine_reasons(fields)
    if reasons:
        return Coverage(False, tuple(reasons))
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    weld = fused_magnetic_pair.metal_fused_magnetic_pair_coverage(
        _LinearScopeView(fields), pml, sources, residency)
    reasons = list(weld.reasons)

    # THE SOURCE SEAM, ASKED HERE RATHER THAN INHERITED. It used to arrive with
    # everything else from the delegation, which was right while no product on this
    # track could carry an in-seam deposit. `fused_magnetic_pair` now declares
    # CARRIES_DEPOSIT_REPAIR, so what that predicate returns for a magnetic source is
    # an ADMISSION -- and it is an admission this weld may not inherit: nothing
    # brackets THIS product's launch. `launch._install_fused_pair` is the only builder
    # of LeadingRepairPlan / TrailingRepairPlan on this track, and it builds one only
    # for a family that declares CARRIES_DEPOSIT_REPAIR; this one declares False.
    #
    # THAT SENTENCE USED TO REST ON THE ABSORB TABLE INSTEAD, and it no longer can.
    # Until 2026-09-17 it read that `launch.FUSED_PAIR_ARMS` "carries a row for
    # `fused_magnetic_pair` and for nothing else, so this family is refused at the
    # absorb table before its predicate is ever asked". That was true and is now
    # false: this family HAS a row, and this predicate IS asked on every row the
    # composer walks. Nothing about the refusal below changes -- it was never the
    # absorb table that made the seam safe, it was this clause, asked against the REAL
    # fields with `carries_repair=CARRIES_DEPOSIT_REPAIR`. The wiring is what makes
    # that distinction load-bearing rather than incidental.
    #
    # Asked against the REAL fields rather than the view, and with this module's own
    # declaration, which is False. The same correction
    # `triton_kernels/nonlinear_fused_magnetic_pair.py` made when the Triton pairs
    # flipped.
    reasons.extend(_deposit_repair.seam_source_reasons(
        # Spelled as the literal every other product site spells, not as a name: the
        # seam-versus-prose witness test reads this argument as a constant, and a name
        # would silently exempt this site from it.
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283-3284)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))
    # DEDUPLICATED, because the delegation already reported the UNDECLARED case and
    # this clause reports it again in the same words. Two copies of one refusal would
    # read as two independent facts.
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def nonlinear_fused_magnetic_pair_equivalence(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Dict[str, Any]:
    """THE IDENTITY THIS MODULE RESTS ON, as a measurement rather than a claim.

    The weld's predicate is the conjunction of its two halves' predicates plus the
    seam clauses, and on a nonlinear run those two halves are the two SHIPPED spine
    arms. So on any configuration:

        this predicate  ==  spine curl arm  AND  spine H arm  AND  the seam clauses

    and the first two conjuncts can be evaluated directly. This returns all of them
    so a caller — the test module and the gate's ``equivalence`` leg both do — can
    assert the identity instead of reading the docstring. A configuration where the
    weld admits and either spine arm refuses would mean the weld had WEAKENED a half
    it claims to inherit, which is the one failure this construction can have.
    """
    from .nonlinear_update_e import (  # noqa: PLC0415 - one import, at call time
        nonlinear_run_constitutive_coverage, nonlinear_run_pml_curl_coverage,
    )

    weld = metal_nonlinear_fused_magnetic_pair_coverage(fields, pml, sources,
                                                        residency)
    curl = nonlinear_run_pml_curl_coverage(fields, pml, "step_B", residency)
    magnetic = nonlinear_run_constitutive_coverage(fields, pml, "H", residency)
    return {
        "weld_covered": bool(weld.covered),
        "weld_reasons": list(weld.reasons),
        "spine_curl_covered": bool(curl.covered),
        "spine_curl_reasons": list(curl.reasons),
        "spine_constitutive_covered": bool(magnetic.covered),
        "spine_constitutive_reasons": list(magnetic.reasons),
        # The weld may only be NARROWER than the two halves, never wider: it adds
        # the seam clauses and removes nothing.
        "weld_never_wider_than_the_halves": bool(
            (not weld.covered) or (curl.covered and magnetic.covered)),
    }


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def plan_metal_nonlinear_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[fused_magnetic_pair.MetalFusedMagneticPairPlan]:
    """Build the fused B/H plan for a nonlinear run, or ``None`` when refused.

    THE PLAN CLASS IS THE WELDED FAMILY'S, not a subclass. The two are the same
    dispatch of the same compiled function over the same buffers; a distinct class
    would give the whole-step arbiter and the composition probe a second type to
    special-case for no measured difference.

    ``functions`` is the mutation seam and is FORWARDED. Dropping it would silently
    disarm every mutation leg the gate runs against this arm — each would launch
    the shipped kernel and score its defect uncaught.
    """
    if not metal_nonlinear_fused_magnetic_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    return fused_magnetic_pair.plan_metal_fused_magnetic_pair(
        _LinearScopeView(fields), pml, sources, residency, contract_variants,
        functions)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False,
                        (f"nonlinear fused magnetic pair cannot fill {slot}",))
    return metal_nonlinear_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[fused_magnetic_pair.MetalFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_nonlinear_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    Unwired for :func:`.fused_magnetic_pair.register_arms`'s reason and for one
    more that is specific to this arm: were it wired it would contend for
    ``step_B`` with the nonlinear SPINE curl arm, which is wired and admits exactly
    the same configurations. ``_select_slot`` would see two admitters and leave the
    slot UNSELECTED — the fusion would not merely fail to be chosen, it would take
    the certified curl off the device as well.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "nonlinear fused magnetic B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="nonlinear fused magnetic B/H pair: ",
                          noun="fused nonlinear-run PML B-curl/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
