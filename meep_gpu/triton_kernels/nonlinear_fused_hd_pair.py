"""The H->D weld on a NONLINEAR run — admission only, no new kernel.

WHAT THIS MODULE ADDS, AND WHAT IT DELIBERATELY DOES NOT. It adds one predicate, one
plan builder and one checkable identity. It adds NO arithmetic, NO kernel and NO plan
class: the kernel a covered configuration launches here is
:func:`.fused_hd_pair.fused_constitutive_curl_H_to_D`, character for character,
through :class:`.fused_hd_pair.FusedHdPairPlan` built by
:func:`.fused_hd_pair.plan_fused_hd_pair`, because on a chi2/chi3 run the H->D seam IS
the ordinary one.

**SO THE HONEST HEADLINE IS THAT THIS FAMILY'S GATE CERTIFIES NO NEW KERNEL.** What it
certifies is that the shipped weld, driven on a chi2/chi3 configuration, is
bit-identical to the array path over complete driver steps — a measurement the
released ``fused_hd_pair`` artifact does not contain, because its own predicate
refuses every nonlinear run.

THE MEASUREMENT THAT LICENSES THE ADMISSION IS ALREADY IN THE TREE and is not made
here. ``stepping.update_H`` and ``stepping.step_D`` never read chi2 or chi3 — the Pade
factor enters ``update_E`` alone (stepping.py:999-1000) — so :mod:`.nonlinear_update_e`
already ships two SPINE ARMS whose whole content is that fact, both wired into
``launch.plan_step``:

* ``nonlinear_run_pml_curl_coverage`` (nonlinear_update_e.py:1119) admits the CERTIFIED
  ``kernels.pml_curl_step`` on a nonlinear run's ``step_B`` **and ``step_D``** — the
  sub-step is a mandatory argument and both are in ``coverage.CURL_SUB_STEPS`` — and
  its builder returns a ``launch.PmlCurlPlan``;
* ``nonlinear_run_constitutive_coverage`` (:1222) does the same for
  ``kernels.constitutive_step`` on side ``'H'``, and its builder returns a
  ``launch.ConstitutivePlan``.

Those are exactly the two halves :func:`.fused_hd_pair.fused_constitutive_curl_H_to_D`
welds. :mod:`.nonlinear_fused_magnetic_pair` applies this one-clause inversion to the
B->H weld; this module applies it to the H->D weld, and the two are siblings rather
than one being derived from the other.

===========================================================================
THE CONJUNCTION, AND WHY IT IS THREE THINGS RATHER THAN TWO
===========================================================================

:func:`.fused_hd_pair.fused_hd_pair_coverage` is ``constitutive_coverage('H')`` AND
``pml_curl_coverage('step_D')`` plus the seam clauses — the electric integrated-source
withdraw, the fold, and the rotation's allocation invariant. The single place either
half reads the nonlinearity is ``coverage._grid_reasons`` clause 10
(coverage.py:195-196), and :class:`_LinearScopeView` inverts exactly that read while
forwarding every array, coefficient, boundary, conductivity, dispersion and layout
fact unchanged.

THE PREDICATE HERE IS A CONJUNCTION OF **THREE** THINGS, NOT TWO, and the third is not
decoration:

    the two SHIPPED SPINE PREDICATES  AND  ``fused_hd_pair_coverage`` through the view

The shipped spine predicates are NARROWER than the shared clauses they restate in one
measured place — ``nonlinear_run_pml_curl_coverage`` refuses an unallocated CURL TARGET
(``Bx``..``Dz``), which ``coverage.pml_curl_coverage`` admits through a hole in its
allocation clause (documented at nonlinear_update_e.py:1146-1160, where it is also
reported upward). A weld built only on the view would therefore be WIDER than the two
arms it claims to inherit, on exactly that hole. Conjoining the spine arms as well
makes "the weld is never wider than either half" true by construction rather than by
inspection, and :func:`nonlinear_fused_hd_pair_equivalence` states it as a checkable
identity that the test module and the device gate both run.

A FOURTH CLAUSE INVERTS THE NONLINEARITY IN THE OTHER DIRECTION: this admission
REQUIRES an installed chi2/chi3, so it is disjoint from
:func:`.fused_hd_pair.fused_hd_pair_coverage` in BOTH directions. Two predicates
admitting one configuration is a dispatcher picking by ordering, which is the failure
the arm table exists to prevent.

===========================================================================
WHAT THE CORPUS SAYS THIS IS WORTH — measured, not argued
===========================================================================

TWO seam-instances, both reachable, from
``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597) at the
H->D cell (``nonlinear run``, ``nonlinear run PML``)::

    examples:3rd-harm-1d.py                shape (1, 1, 2500)
    tests:Test3rdHarm1d.test_3rd_harm_1d   shape (1, 1, 2000)

Both carry ``bucket: buildable_not_built``, ``withdraw_in_seam: false``,
``integrated_electric_sources: 0`` and ``served_by: null``. Both source ELECTRICALLY
and NOT with an integrated source, which is what clears this seam's one clause: the
electric withdraw between the two consults (driver.py:3313-3314) is the no-op on every
electric source on those rows.

**THE COVERAGE WAS NOT ALREADY OPEN ON THIS BACKEND, and that was checked rather than
assumed.** The Metal board's ``launch.FUSED_PAIR_EXTRA_ARMS`` has carried
``"fused_hd_pair": (("nonlinear PML magnetic", "nonlinear PML curl"),)`` since
2026-09-05, so on that backend the H->D nonlinear cell was predicate-admitted before
its gate ran and the 2026-09-07 Metal lane supplied an OWED MEASUREMENT rather than
opening the cell. On THIS backend
``launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` carries two rows and neither is
``fused_hd_pair``'s; ``launch.CERTIFIED_FUSED_PAIR_ARMS`` carries no ``fused_hd_pair``
row at all (that product is itself unwired), and
``fused_hd_pair.fused_hd_pair_coverage`` refuses every chi2/chi3 run through
``coverage._grid_reasons`` clause 10. So this admission is new here.

===========================================================================
THE SEAM, AND WHAT IS NOT INSTALLED
===========================================================================

Exactly one statement stands between the two consults — the electric integrated-source
withdraw (driver.py:3313-3314) — so :data:`CARRIES_DEPOSIT_REPAIR` is False as a FACT
about the driver rather than a choice, and :mod:`..withdraw_hoist` is the module that
owns the seam. That clause is asked by :func:`.fused_hd_pair.fused_hd_pair_coverage`
through the view, against the REAL source list, and it is NOT re-asked here: unlike
the B->H seam there is no injection in this one, so there is no admission this module
would be inheriting that the view could hide. The chi2/chi3 nonlinearity is invisible
to a WITHDRAW in any case — the withdraw subtracts a standing dipole from ``D``, and
``D`` is what this seam produces.

:data:`INSTALLABLE` is False, and it is the SHIPPED product's flag that decides
whether anything is installed: this module builds
:class:`.fused_hd_pair.FusedHdPairPlan` and that class's family declares
``INSTALLABLE = False``, so ``launch._declared_uninstallable`` refuses it by name on
every configuration whether or not this admission exists. The flag is restated here so
a reader of this module does not have to follow the import to learn it, and a laptop
test pins the two equal.

SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own definition. A
released product credits its admitted seam-instances while executing NOWHERE, and this
module changes nothing about that: it is in no ``fastpath.RELEASED_FUSED_ARMS``
envelope and ``fastpath`` never plans it. **No timing exists for this shape and none is
licensed here.**

Import contract: importable WITHOUT Triton — the predicate, the identity and the plan
builder (to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import fused_hd_pair as _weld
from .coverage import Coverage, _call
from .nonlinear_update_e import (
    nonlinear_run_constitutive_coverage,
    nonlinear_run_pml_curl_coverage,
)

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "nonlinear_fused_hd_pair"

#: The sub-step slot this product STARTS at, in the driver's own order.
SLOT = "update_H"

#: The driver call sites ONE launch performs, in driver order (driver.py:3311, :3315).
#: READ from the welded family rather than respelled -- there is one span here and it
#: is that family's.
REPLACES: Tuple[str, ...] = _weld.REPLACES

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _weld.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE: str = _weld.CONSTITUTIVE_SIDE
CURL_SUB_STEP: str = _weld.CURL_SUB_STEP

#: The arm pair this admission implements, read off the two SPINE predicates it
#: conjoins -- the labels ``launch._select_slot`` writes on a nonlinear run.
ARMS: Tuple[str, str] = ("nonlinear run", "nonlinear run PML")

#: The module whose kernel and plan class a covered configuration launches. NAMED as
#: data so the record, the test and the gate all read the same answer to "which
#: device string ran", and so that "this gate certifies no new kernel" is a
#: declaration rather than a sentence in a docstring.
CERTIFIES_NO_NEW_KERNEL = True
WELDED_FAMILY: str = _weld.FAMILY
WELDED_KERNEL = "fused_constitutive_curl_H_to_D"

#: The volumes one launch rotates and steps in place, and the constitutive sources --
#: the welded family's, imported so a rename there cannot leave a stale copy here.
ROTATED: Tuple[str, ...] = _weld.ROTATED
IN_PLACE: Tuple[str, ...] = _weld.IN_PLACE
CONSTITUTIVE_SOURCES: Tuple[str, ...] = _weld.CONSTITUTIVE_SOURCES

#: Does this product bracket its launch with the DEPOSIT repair? NO -- a fact about
#: the driver: nothing is injected between the two consults.
CARRIES_DEPOSIT_REPAIR: bool = _weld.CARRIES_DEPOSIT_REPAIR

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = _weld.REPAIR_PATHS

#: Does this product perform the seam's electric withdraw before its launch? NO --
#: the welded family's declaration, and the reason is its INSTALLABLE flag.
HOISTS_THE_WITHDRAW: bool = _weld.HOISTS_THE_WITHDRAW

#: May the composer install this product? NO. RESTATED from the welded family rather
#: than declared independently: this module builds that family's plan, so that
#: family's flag is what ``launch._declared_uninstallable`` reads. A laptop test pins
#: the two equal.
INSTALLABLE: bool = _weld.INSTALLABLE

INSTALLABLE_REASON = (
    "THIS ADMISSION INSTALLS NOTHING OF ITS OWN: it builds "
    "fused_hd_pair.FusedHdPairPlan, whose family declares INSTALLABLE = False, so "
    "launch._declared_uninstallable refuses it by name on every configuration "
    "whether or not this admission exists. The welded family's own reason is "
    "fused_hd_pair.INSTALLABLE_REASON and is not restated here, because restating it "
    "would create a second place for it to drift. WHAT IS SPECIFIC TO THIS FAMILY is "
    "the arbitration on ITS OWN TWO ROWS, and it is MEASURED rather than asserted: "
    "over the driver's step_B - update_H - step_D - update_E slot path launches are "
    "4 - (installed pairs) and a TWO-SLOT H->D product takes one slot from EACH "
    "neighbour. Both nonlinear rows have a released B->H neighbour "
    "(nonlinear_fused_magnetic_pair) and NO D->E neighbour -- that module's own "
    "docstring records the D/E partner cell as worth ZERO because both rows inject "
    "ELECTRICALLY between step_D and update_E, which no fused product on any backend "
    "can straddle -- so the PREDICATE join says TIE on both rows rather than LOSS. "
    "The gate's arbitration leg re-derives that through the shipped composer on real "
    "fixtures and its lift leg re-derives it per driven corpus row under "
    "`arbitration_over_the_driven_rows`; those measurements are the record rather "
    "than this sentence. WHAT THEY MEASURED: both driven corpus rows are a TIE -- 2 "
    "of 2, no LOSS, no GAIN -- re-derived from the composer slot tables recorded in "
    "results/triton_nonlinear_fused_hd_pair_2026-09-07/keep/lift/, each showing one "
    "installed pair (`fused pair B (nonlinear)` over step_B and update_H) and no "
    "D->E pair. Installing this product would leave the per-step launch count "
    "UNCHANGED on both rows; it lowers it on neither, so nothing about the launch "
    "count argues for installing it. THE ARTIFACTS THEMSELVES SAY LOSS ON BOTH and "
    "are wrong: the kit scored the verdict off the count of SLOTS carrying a fused "
    "label rather than the count of PAIRS, and one pair holds two slots -- this "
    "family's own prediction above is what caught it (fixed 2026-09-08 in "
    "parity/meep_gpu/triton_hd_tail_gate_kit.py; the records are not edited and a "
    "re-run is what re-earns them)")

__all__ = [
    "ARMS", "CARRIES_DEPOSIT_REPAIR", "CERTIFIES_NO_NEW_KERNEL",
    "CONSTITUTIVE_SIDE", "CONSTITUTIVE_SOURCES", "CURL_SUB_STEP", "FAMILY",
    "HOISTS_THE_WITHDRAW", "INSTALLABLE", "INSTALLABLE_REASON", "IN_PLACE",
    "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM", "SLOT", "WELDED_FAMILY",
    "WELDED_KERNEL",
    "explain_nonlinear_fused_hd_pair", "nonlinear_fused_hd_pair_coverage",
    "nonlinear_fused_hd_pair_equivalence", "plan_nonlinear_fused_hd_pair",
]


class _LinearScopeView:
    """Expose a nonlinear run to the UNCHANGED ordinary H->D predicate.

    ``coverage._grid_reasons`` clause 10 (coverage.py:195-196) is the ONLY read of
    ``has_nonlinearity`` in :mod:`.coverage`, and it refuses the chi2/chi3 run on every
    sub-step because the shared clause is written for the whole step. The two kernels
    this weld concatenates never read chi2 or chi3 at all. This view inverts that one
    read and forwards everything else -- arrays, PML coefficients, boundary
    declarations, conductivity, dispersion, layout -- untouched, so the remaining
    clauses are evaluated in force on the REAL configuration.

    The same idiom as :class:`.nonlinear_fused_magnetic_pair._LinearScopeView` and
    :class:`.complex_no_pml_conductive._CurlScopeView`, and used for BOTH the predicate
    and the reused builder so a covered verdict can never reach a builder that
    re-applies the broader clause.

    IT IS DELIBERATELY NOT IMPORTED FROM THE B->H SIBLING, and it is NOT THE SAME
    CLASS either. That module's view is read-only, and it can be: ``FusedPairPlan``
    does not rotate anything. **THIS SEAM'S PLAN DOES.**
    :class:`.offdiag_scratch_weld.ScratchWeldPairPlan` holds the object it was built
    with and, after every launch, ``setattr``-s the six ``H``/``f_w_H`` names onto it
    to move the engine's references onto the freshly written scratch. A view with
    ``__slots__`` and no ``__setattr__`` raises ``AttributeError: '_LinearScopeView'
    object has no attribute 'Hx'`` at the first rotation -- measured on this lane's
    first device run of this admission, 2026-09-08.

    So the write side is forwarded too, and that is the CORRECT semantics rather than
    a workaround: this class exists to invert ONE READ, and every other access --
    read or write -- must reach the engine's own ``Fields``. A view that swallowed the
    rotation would leave the engine stepping from last step's magnetic field with no
    error at all.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        if name == "has_nonlinearity":
            return False
        return getattr(object.__getattribute__(self, "_fields"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Every write reaches the engine. THE ROTATION IS A WRITE."""
        setattr(object.__getattribute__(self, "_fields"), name, value)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_LinearScopeView({object.__getattribute__(self, '_fields')!r})"


def _nonlinear_spine_reasons(fields: Any) -> List[str]:
    """The inverted chi2/chi3 clause: this arm requires an INSTALLED nonlinearity.

    What makes this family disjoint from
    :func:`.fused_hd_pair.fused_hd_pair_coverage` in BOTH directions -- that predicate
    refuses every nonlinear run through clause 10, and this one refuses every linear
    run here. Two predicates admitting one configuration is a dispatcher picking by
    ordering, which is the failure the arm table exists to prevent.
    """
    try:
        active = bool(getattr(fields, "has_nonlinearity", False))
    except Exception as exc:  # noqa: BLE001 - unreadable state is a refusal
        return [f"fields.has_nonlinearity raised {exc!r}"]
    if not active:
        return ["no chi2/chi3 is installed: that configuration is the ordinary H->D "
                "weld's (fused_hd_pair.fused_hd_pair_coverage) and this admission "
                "must not overlap it"]
    return []


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def nonlinear_fused_hd_pair_coverage(fields: Any, pml: Any,
                                     sources: Any = None) -> Coverage:
    """May ONE launch span ``update_H`` -> the withdraw -> ``step_D``, nonlinearly?

    A conjunction of FOUR things, none of them weakened:

    1. the inverted chi2/chi3 clause, which is what makes this arm disjoint from the
       ordinary H->D weld;
    2. the two SHIPPED spine arms,
       :func:`.nonlinear_update_e.nonlinear_run_constitutive_coverage` on ``'H'`` and
       :func:`.nonlinear_update_e.nonlinear_run_pml_curl_coverage` on ``step_D`` --
       the two halves ``plan_step`` selects at this cell today, asked in the DRIVER'S
       order so the first refusal a reader sees names the half the driver reaches
       first;
    3. :func:`.fused_hd_pair.fused_hd_pair_coverage` through
       :class:`_LinearScopeView`, which is what carries the SEAM clauses (the electric
       integrated-source withdraw, the fold, the rotation's allocation invariant) and
       re-evaluates both halves' shared clauses in force;
    4. the fold, re-checked. Every clause set above refuses a mirror plane already;
       this is the clause that keeps that true if one ever stops -- and unlike the
       B->H seam the reason here is not a swallowed pass (this seam is fill-free) but
       that a folded run selects the ``folded`` arms on both slots.

    ``sources`` MUST BE DECLARED, for :mod:`..withdraw_hoist`'s own reason: ``Fields``
    does not hold the source list, so ``None`` is a REFUSAL and not an assumed empty
    set. That clause is what caps this family -- and it is also what both nonlinear
    corpus rows clear, because neither carries an INTEGRATED electric source.
    """
    reasons: List[str] = _nonlinear_spine_reasons(fields)
    if reasons:
        return Coverage(False, tuple(reasons))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    # 2. The two shipped spine arms, by name, IN THE DRIVER'S ORDER, with their own
    #    reasons prefixed so a reader can tell which side said it.
    constitutive = nonlinear_run_constitutive_coverage(fields, pml,
                                                       CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"nonlinear spine constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = nonlinear_run_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"nonlinear spine curl half: {reason}"
                       for reason in curl.reasons)

    # 3. The WELD's own predicate, unchanged, through the one-clause view. This is
    #    where the electric-withdraw clause and the rotation invariant are asked.
    #
    #    IT IS NOT RE-ASKED AFTERWARDS, and that is a difference from the B->H
    #    sibling worth naming: on that seam the driver INJECTS a magnetic source
    #    between the halves, so the admission clause 3 returns for a source is one
    #    the view could hide and the sibling asks again against the real fields.
    #    Nothing is injected in THIS seam (the magnetic injection is one seam earlier
    #    at driver.py:3283-3284 and the electric one is one seam later at :3317-3322),
    #    so `deposit_repair` is not consulted by either module and there is no
    #    inherited admission to re-check. What IS in the seam is the electric
    #    WITHDRAW, and `_LinearScopeView` does not hide a source: it forwards the
    #    `sources` argument untouched, because that argument is passed to the
    #    predicate directly rather than read off `Fields`.
    weld = _weld.fused_hd_pair_coverage(_LinearScopeView(fields), pml, sources)
    if not weld.covered:
        reasons.extend(f"H->D weld: {reason}" for reason in weld.reasons)

    # 4. The fold, re-checked rather than read off another module's guard.
    if _call(grid, "has_symmetry", default=False):
        reasons.append(
            "a mirror plane is active: a folded run selects the `folded` arms on "
            "both slots of this seam, which is folded_fused_hd_pair's cell")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the `folded` arms are "
                f"selected on both slots and this admission implements the "
                f"`nonlinear run` / `nonlinear run PML` pair")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_nonlinear_fused_hd_pair(fields: Any, pml: Any,
                                    sources: Any = None) -> Coverage:
    """The predicate under the name a report reads. One home for the verdict."""
    return nonlinear_fused_hd_pair_coverage(fields, pml, sources)


def nonlinear_fused_hd_pair_equivalence(fields: Any, pml: Any,
                                        sources: Any = None) -> Dict[str, Any]:
    """THE IDENTITY THIS MODULE RESTS ON, as a measurement rather than a claim.

    The weld's predicate is the conjunction of its two halves' predicates plus the
    seam clauses, and on a nonlinear run those two halves are the two SHIPPED spine
    arms. So on ANY configuration::

        this predicate  ==  spine H arm  AND  spine curl arm  AND  the seam clauses

    and the first two conjuncts can be evaluated directly. Every one of them is
    returned so a caller -- the test module and the gate's ``equivalence`` leg both do
    -- can assert the identity instead of reading this docstring. A configuration
    where this predicate admits and either spine arm refuses would mean it had
    WIDENED a half it claims to inherit, which is the one failure this construction
    can have; ``weld_never_wider_than_the_halves`` is that assertion.
    """
    weld = nonlinear_fused_hd_pair_coverage(fields, pml, sources)
    constitutive = nonlinear_run_constitutive_coverage(fields, pml,
                                                       CONSTITUTIVE_SIDE)
    curl = nonlinear_run_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    scoped = _weld.fused_hd_pair_coverage(_LinearScopeView(fields), pml, sources)
    return {
        "weld_covered": bool(weld.covered),
        "weld_reasons": list(weld.reasons),
        "spine_constitutive_covered": bool(constitutive.covered),
        "spine_constitutive_reasons": list(constitutive.reasons),
        "spine_curl_covered": bool(curl.covered),
        "spine_curl_reasons": list(curl.reasons),
        "scoped_hd_weld_covered": bool(scoped.covered),
        "scoped_hd_weld_reasons": list(scoped.reasons),
        # The admission may only be NARROWER than what it conjoins, never wider.
        "weld_never_wider_than_the_halves": bool(
            (not weld.covered)
            or (constitutive.covered and curl.covered and scoped.covered)),
    }


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def plan_nonlinear_fused_hd_pair(fields: Any, pml: Any, sources: Any = None,
                                 block: Optional[int] = None,
                                 num_warps: Optional[int] = 1,
                                 kernel: Any = None) -> Any:
    """Build the fused H->D plan for a nonlinear run, or ``None`` when refused.

    THE PLAN CLASS IS THE WELDED FAMILY'S, not a subclass, and THE BINDING LIST IS NOT
    TRANSCRIBED. :func:`.fused_hd_pair.plan_fused_hd_pair` is called through the same
    :class:`_LinearScopeView`, so every volume, every coefficient table and -- the
    hazard this avoids -- the SUB-LATTICE PAIRING (the assertion that ``step_D``'s
    ``kms`` group and ``update_H``'s are the same INTEGER volumes) is chosen and
    checked by the shipped builder and nowhere else. A second spelling of that list
    here would be a half-cell error in the absorber profile waiting to happen:
    converged, smooth and wrong.

    ``None`` is the only refusal: a configuration this weld does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    ``kernel`` is THE MUTATION SEAM and it is FORWARDED rather than installed
    afterwards: :func:`.fused_hd_pair.plan_fused_hd_pair` takes the argument itself.
    Dropping it would not be a slowdown, it would be a DISARMING -- every mutation leg
    of the gate would launch the shipped kernel and score its planted defect uncaught.
    """
    if not nonlinear_fused_hd_pair_coverage(fields, pml, sources).covered:
        return None
    return _weld.plan_fused_hd_pair(
        _LinearScopeView(fields), pml, sources, block=block, num_warps=num_warps,
        kernel=kernel)
