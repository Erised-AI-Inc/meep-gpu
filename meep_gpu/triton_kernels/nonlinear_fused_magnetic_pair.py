"""The fused magnetic B/H pair on a NONLINEAR run — admission only, no new kernel.

WHAT THIS MODULE ADDS, AND WHAT IT DELIBERATELY DOES NOT. It adds one predicate,
one plan builder and one checkable identity. It adds NO arithmetic, NO kernel and
NO plan class: the kernel a covered configuration launches here is
:func:`.kernels.fused_curl_constitutive_B`, character for character, through
:class:`.launch.FusedPairPlan` built by :func:`.launch.plan_fused_pair`, because
on a chi2/chi3 run the B/H seam IS the ordinary one.

THE MEASUREMENT THAT LICENSES THAT IS ALREADY IN THE TREE and is not made here.
``stepping.step_B`` and ``stepping.update_H`` never read chi2 or chi3 — the Pade
factor enters ``update_E`` alone (stepping.py:999-1000) — so :mod:`.nonlinear_update_e`
already ships two SPINE ARMS whose whole content is that fact, both wired into
``launch.plan_step``:

* ``nonlinear_run_pml_curl_coverage`` (nonlinear_update_e.py:1119) admits the
  CERTIFIED ``kernels.pml_curl_step`` on a nonlinear run's ``step_B``, and its
  builder returns a ``launch.PmlCurlPlan``;
* ``nonlinear_run_constitutive_coverage`` (:1222) does the same for
  ``kernels.constitutive_step`` on side ``'H'``, and its builder returns a
  ``launch.ConstitutivePlan``.

So the two halves ``plan_step`` already selects at this cell on a nonlinear row
are, by construction, the two halves :func:`.kernels.fused_curl_constitutive_B`
welds. This module applies the same one-clause inversion ONE LEVEL UP — to the
WELD instead of to each half.

===========================================================================
THE CONJUNCTION, AND WHY IT IS EXACT RATHER THAN MERELY PLAUSIBLE
===========================================================================

:func:`.coverage.fused_pair_coverage` is ``pml_curl_coverage(step_B)`` AND
``constitutive_coverage('H')`` plus two seam clauses that read the SOURCE LIST and
the GRID — neither of which the chi2/chi3 clause touches. The single place either
half reads the nonlinearity is ``coverage._grid_reasons`` clause 10
(coverage.py:195-196), and :class:`_LinearScopeView` inverts exactly that read
while forwarding every array, coefficient, boundary, conductivity, dispersion and
layout fact unchanged — the same idiom
:class:`.complex_no_pml_conductive._CurlScopeView` uses for its own two masks.

THE PREDICATE HERE IS A CONJUNCTION OF **THREE** THINGS, NOT TWO, and the third is
not decoration:

    the two SHIPPED SPINE PREDICATES  AND  ``fused_pair_coverage`` through the view

The shipped spine predicates are NARROWER than the shared clauses they restate in
one measured place — ``nonlinear_run_pml_curl_coverage`` refuses an unallocated
CURL TARGET (``Bx``..``Dz``), which ``coverage.pml_curl_coverage`` admits through a
hole in its allocation clause (documented at nonlinear_update_e.py:1146-1160, where
it is also reported upward). A weld built only on the view would therefore be WIDER
than the two arms it claims to inherit, on exactly that hole. Conjoining the spine
arms as well makes "the weld is never wider than either half" true by construction
rather than by inspection, and :func:`nonlinear_fused_magnetic_pair_equivalence`
states it as a checkable identity that the test module and the device gate both run.

===========================================================================
WHAT THE CORPUS SAYS THIS IS WORTH — measured, not argued
===========================================================================

TWO seam-instances, both reachable, from the 186-row Triton census
``parity/meep_gpu/results/predicate_coverage_2026-08-16_wired_convention`` as
scored by ``results/fusion_matrix_triton_2026-08-20_closed/`` at the B->H cell
(``nonlinear run PML``, ``nonlinear run``):

    examples:3rd-harm-1d.py                shape (1, 1, 2500)
    tests:Test3rdHarm1d.test_3rd_harm_1d   shape (1, 1, 2000)

Both carry an ELECTRIC source ONLY (``source_field_types == ['D']``), so the
magnetic source seam (driver.py:3283-3284) is clear on both; both are periodic in
x and y with a METALLIC z boundary and ``has_metallic``, so the inline
``zero_metal_B`` carry this weld inherits from the shipped kernel is LIVE on those
rows rather than compiled out; neither is folded, so the two symmetry passes in
the seam are the no-ops the predicate re-checks below. The matrix's
``live_in_seam_passes`` column reads ``['zero_metal_B']`` on both.

THE D/E PARTNER CELL IS WORTH ZERO and is not built: both rows inject
ELECTRICALLY between ``step_D`` and ``update_E``, which no fused product on any
backend can straddle.

THE OTHER BACKEND BUILT THIS FIRST, and its numbers are the ones this follows:
``metal_kernels/nonlinear_fused_magnetic_pair.py`` (236 lines, the same
admission-only shape) is certified by
``parity/meep_gpu/gate_metal_below_the_cut_fused_pairs.py``.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; there is no slot
it can claim without a composition rule nothing has measured. There is a second
reason specific to this arm: wired, it would contend for ``step_B`` with the
nonlinear SPINE curl arm, which IS wired and admits exactly the same
configurations, and ``_select_slot`` (launch.py:1983-1996) leaves a slot with two
admitters UNSELECTED — the fusion would not merely fail to be chosen, it would take
the certified curl off the device with it. Nothing in ``launch.py`` names this
module and ``fastpath.plan_fast_path`` is unchanged.

DEVICE STATUS: see the gate ``parity/meep_gpu/probe_triton_nonlinear_fused_magnetic_pair.py``
and its artifact. A weld licenses a claim, not a dispatch: nothing in a default run
reaches this plan.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import Coverage, MAGNETIC_FIELD_TYPE, _call, fused_pair_coverage
from .nonlinear_update_e import (
    nonlinear_run_constitutive_coverage,
    nonlinear_run_pml_curl_coverage,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? NO, and it may
#: not: ``launch._install_fused_pair`` -- the only builder of ``LeadingRepairPlan`` /
#: ``TrailingRepairPlan`` -- installs the ordinary and dispersive pairs and never this
#: one, and ``deposit_repair.repairable`` refuses an instantaneous chi2/chi3 by name in
#: any case, because the nonlinear constitutive step is not the linear accumulation the
#: repair inverts. So this module refuses every in-seam deposit, which is what it did
#: before ``coverage.py`` was wired and is what its gate measured.
CARRIES_DEPOSIT_REPAIR = False

#: The fused pair key this weld is: ``coverage.FUSED_PAIRS['B']``.
PAIR = "B"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3281-3288). Declared, never inferred from the slot name. Only THREE
#: of them do work on an admitted configuration — the two symmetry fills return at
#: their first line without a mirror plane, which the predicate re-checks — and the
#: inert two are listed anyway: what a launch REPLACES is what the driver would
#: otherwise have called.
REPLACES: Tuple[str, ...] = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: The kernel this family launches, by name. It is the SHIPPED ordinary one; a
#: divergence between this name and what :class:`.launch.FusedPairPlan` dispatches
#: for ``pair='B'`` would mean the two had stopped being the same kernel.
KERNEL = "kernels.fused_curl_constitutive_B"

__all__ = [
    "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "KERNEL", "PAIR", "REPLACES",
    "nonlinear_fused_magnetic_pair_coverage",
    "nonlinear_fused_magnetic_pair_equivalence",
    "plan_nonlinear_fused_magnetic_pair",
]


class _LinearScopeView:
    """Expose a nonlinear run to the UNCHANGED ordinary fused-pair predicate.

    ``coverage._grid_reasons`` clause 10 (coverage.py:195-196) is the ONLY read of
    ``has_nonlinearity`` in :mod:`.coverage`, and it refuses the chi2/chi3 run on
    every sub-step because the shared clause is written for the whole step. The two
    kernels this weld concatenates never read chi2 or chi3 at all. This view
    inverts that one read and forwards everything else — arrays, PML coefficients,
    boundary declarations, conductivity, dispersion, layout — untouched, so the
    remaining clauses are evaluated in force on the REAL configuration.

    The same idiom as :class:`.complex_no_pml_conductive._CurlScopeView`, and used
    for BOTH the predicate and the reused builder so a covered verdict can never
    reach a builder that re-applies the broader clause.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        if name == "has_nonlinearity":
            return False
        return getattr(object.__getattribute__(self, "_fields"), name)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_LinearScopeView({object.__getattribute__(self, '_fields')!r})"


def _nonlinear_spine_reasons(fields: Any) -> List[str]:
    """The inverted chi2/chi3 clause: this arm requires an INSTALLED nonlinearity.

    What makes this family disjoint from :func:`.coverage.fused_pair_coverage` in
    BOTH directions — that predicate refuses every nonlinear run through clause 10,
    and this one refuses every linear run here. Two predicates admitting one
    configuration is a dispatcher picking by ordering, which is the failure the
    arm table exists to prevent.
    """
    try:
        active = bool(getattr(fields, "has_nonlinearity", False))
    except Exception as exc:  # noqa: BLE001 - unreadable state is a refusal
        return [f"fields.has_nonlinearity raised {exc!r}"]
    if not active:
        return ["no chi2/chi3 is installed: that configuration is the ordinary "
                "fused pair's (coverage.fused_pair_coverage) and this admission "
                "must not overlap it"]
    return []


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def nonlinear_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                           sources: Any = None) -> Coverage:
    """May ONE launch span ``step_B`` -> wall -> ``update_H`` on a nonlinear run?

    A conjunction of THREE predicates, none of them weakened:

    1. the two SHIPPED spine arms, :func:`.nonlinear_update_e.nonlinear_run_pml_curl_coverage`
       on ``step_B`` and :func:`.nonlinear_update_e.nonlinear_run_constitutive_coverage`
       on ``'H'`` — the two halves ``plan_step`` selects at this cell today;
    2. :func:`.coverage.fused_pair_coverage` on ``pair='B'`` through
       :class:`_LinearScopeView`, which is what carries the SEAM clauses (the
       magnetic-source refusal and the wall-readability clause) and re-evaluates
       both halves' shared clauses in force;
    3. the inverted chi2/chi3 clause, which is what makes this arm disjoint from
       the ordinary fused pair.

    ``sources`` MUST BE DECLARED, for :func:`.coverage.fused_pair_coverage`'s own
    reason: ``Fields`` does not hold the source list, so ``None`` is a REFUSAL and
    not an assumed empty set. That clause is what caps this family — it is also
    what a nonlinear corpus row clears, because both such rows source
    electrically.

    THE TWO SYMMETRY PASSES in this seam (driver.py:3285, :3287) return at their
    first line without ``grid.has_symmetry()`` (stepping.py:1481-1482, :1565-1566),
    and every clause set above already refuses a mirror plane. RE-CHECKED HERE
    ANYWAY rather than read off another module's guard: if the nonlinear scope ever
    admits a fold, this weld would silently swallow two passes that had started
    doing work.
    """
    reasons: List[str] = _nonlinear_spine_reasons(fields)
    if reasons:
        return Coverage(False, tuple(reasons))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    # 1. The two shipped spine arms, by name, with their own reasons prefixed so a
    #    reader can tell which side said it — the construction
    #    complex_fused_magnetic_pair and folded_fused_magnetic_pair both use.
    curl = nonlinear_run_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"nonlinear spine curl half: {reason}"
                       for reason in curl.reasons)
    constitutive = nonlinear_run_constitutive_coverage(fields, pml,
                                                       CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"nonlinear spine constitutive half: {reason}"
                       for reason in constitutive.reasons)

    # 2. The WELD's own predicate, unchanged, through the one-clause view. This is
    #    where the wall-readability clause is asked.
    weld = fused_pair_coverage(_LinearScopeView(fields), pml, PAIR, sources)
    if not weld.covered:
        reasons.extend(f"fused pair: {reason}" for reason in weld.reasons)

    # 2b. THE SOURCE SEAM, ASKED HERE RATHER THAN INHERITED. It used to arrive with
    #     everything else from clause 2, which was right while no product could carry
    #     an in-seam deposit. ``coverage.py`` now declares CARRIES_DEPOSIT_REPAIR, so
    #     what clause 2 returns for a magnetic source is an ADMISSION -- and it is an
    #     admission this weld may not inherit, for two independent reasons:
    #
    #     * nothing brackets THIS product's launch. ``launch._install_fused_pair`` is
    #       the only builder of LeadingRepairPlan / TrailingRepairPlan and it installs
    #       the ordinary and dispersive pairs, never this one;
    #     * the repair could not carry it anyway. ``deposit_repair.repairable``
    #       refuses an instantaneous chi2/chi3 by name -- the constitutive step is not
    #       the linear accumulation the repair inverts -- and clause 2 cannot see that,
    #       because ``_LinearScopeView`` is precisely the view that hides the
    #       nonlinearity from it.
    #
    #     So the clause is asked again here, against the REAL fields and with this
    #     module's own declaration, which is False.
    reasons.extend(f"fused pair: {reason}" for reason in
                   _deposit_repair.seam_source_reasons(
                       # Spelled as the literal every other product site spells, not as
                       # PAIR: the seam-versus-prose witness test reads this argument as
                       # a constant, and a name would silently exempt this site from it.
                       fields, sources, 'B',
                       undeclared=(
                           "the source set was not declared: this predicate cannot "
                           "infer an empty magnetic source seam from Fields"),
                       refusal=lambda index, source: (
                           f"source {index} ({type(source).__name__}) is magnetic: the "
                           f"driver injects it BETWEEN step_B and update_H "
                           f"(driver.py:3283), which is work inside the seam this "
                           f"kernel closes"),
                       carries_repair=CARRIES_DEPOSIT_REPAIR))

    # 3. The two symmetry passes, re-checked. Every clause set above refuses a
    #    fold already; this is the clause that keeps that true if one ever stops.
    if _call(grid, "has_symmetry", default=False):
        reasons.append(
            "a mirror plane is active: stepping.fill_symmetry_bc_B (driver.py:3285) "
            "and fill_folded_far_ghosts_B (:3287) run inside this seam and this "
            "weld carries neither")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the two symmetry passes "
                f"in this seam are no longer no-ops and this weld carries neither")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def nonlinear_fused_magnetic_pair_equivalence(fields: Any, pml: Any,
                                              sources: Any = None,
                                              ) -> Dict[str, Any]:
    """THE IDENTITY THIS MODULE RESTS ON, as a measurement rather than a claim.

    The weld's predicate is the conjunction of its two halves' predicates plus the
    seam clauses, and on a nonlinear run those two halves are the two SHIPPED spine
    arms. So on ANY configuration::

        this predicate  ==  spine curl arm  AND  spine H arm  AND  the seam clauses

    and the first two conjuncts can be evaluated directly. Every one of them is
    returned so a caller — the test module and the gate's ``equivalence`` leg both
    do — can assert the identity instead of reading this docstring. A configuration
    where the weld admits and either spine arm refuses would mean the weld had
    WIDENED a half it claims to inherit, which is the one failure this construction
    can have; ``weld_never_wider_than_the_halves`` is that assertion.
    """
    weld = nonlinear_fused_magnetic_pair_coverage(fields, pml, sources)
    curl = nonlinear_run_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    magnetic = nonlinear_run_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    scoped = fused_pair_coverage(_LinearScopeView(fields), pml, PAIR, sources)
    return {
        "weld_covered": bool(weld.covered),
        "weld_reasons": list(weld.reasons),
        "spine_curl_covered": bool(curl.covered),
        "spine_curl_reasons": list(curl.reasons),
        "spine_constitutive_covered": bool(magnetic.covered),
        "spine_constitutive_reasons": list(magnetic.reasons),
        "scoped_fused_pair_covered": bool(scoped.covered),
        "scoped_fused_pair_reasons": list(scoped.reasons),
        # The weld may only be NARROWER than what it conjoins, never wider.
        "weld_never_wider_than_the_halves": bool(
            (not weld.covered)
            or (curl.covered and magnetic.covered and scoped.covered)),
    }


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def plan_nonlinear_fused_magnetic_pair(fields: Any, pml: Any, sources: Any = None,
                                       block: Optional[int] = None,
                                       num_warps: Any = "default",
                                       kernel: Any = None) -> Any:
    """Build the fused B/H plan for a nonlinear run, or ``None`` when refused.

    THE PLAN CLASS IS THE WELDED FAMILY'S, not a subclass, and THE BINDING LIST IS
    NOT TRANSCRIBED. :func:`.launch.plan_fused_pair` is called through the same
    :class:`_LinearScopeView`, so every volume, every coefficient table and — the
    hazard this avoids — the CROSSED SUB-LATTICE PAIRING (``step_B`` reads the
    HALF-INTEGER ``kms_a_h``/``sinv_a_h`` while ``update_H`` reads the INTEGER
    ``kps_a``/``kms_a``) is chosen by the shipped builder and nowhere else. A
    second spelling of that list here would be a half-cell error in the absorber
    profile waiting to happen: converged, smooth and wrong.

    ``None`` is the only refusal: a configuration this weld does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.

    ``kernel`` is THE MUTATION SEAM and it is honoured. ``plan_fused_pair`` takes
    no such argument — only ``plan_fused_pair_from_arrays`` does — so it is
    installed onto the built plan's own ``_kernel`` slot, which
    :meth:`.launch.FusedPairPlan.run` reads and which
    ``plan_fused_pair_from_arrays`` fills the same way. Dropping it would not be a
    slowdown, it would be a DISARMING: every mutation leg of the gate would launch
    the shipped kernel and score its planted defect uncaught.
    """
    if not nonlinear_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    from .launch import FUSED_DEFAULT_NUM_WARPS, plan_fused_pair  # noqa: PLC0415

    warps = (FUSED_DEFAULT_NUM_WARPS if num_warps == "default" else num_warps)
    plan = plan_fused_pair(_LinearScopeView(fields), pml, PAIR, sources,
                           block=block, num_warps=warps)
    if plan is None:  # pragma: no cover - the predicate above already admitted
        return None
    if kernel is not None:
        plan._kernel = kernel  # noqa: SLF001 - the documented mutation seam
    return plan
