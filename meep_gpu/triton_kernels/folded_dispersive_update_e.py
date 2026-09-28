"""``update_E`` when a mirror fold AND a susceptibility are live at once.

WHY THIS IS ITS OWN MODULE
--------------------------
Two shipped predicates each refuse one half of this configuration, and each
refusal is correct about its OWN kernel:

* :func:`symmetry.folded_constitutive_coverage` (``side='E'``) refuses a
  registered susceptibility (symmetry.py:811-813) — "folded dispersive update_E
  is not yet composed; its source is (D - sum P), not D". True: the kernel it
  plans is ``launch.ConstitutivePlan``, whose source IS ``D``;
* :func:`dispersive_update_e.dispersive_constitutive_coverage` refuses the fold,
  inherited from ``coverage._grid_reasons`` clause 5.

Neither refusal is about arithmetic that changes, and the composition of the two
belongs to neither file — the same reason ``folded_offdiag_update_e.py`` is its
own module rather than a section of ``symmetry.py`` or ``offdiag_update_e.py``.
Both of those files are also hashed into ``fingerprints.json``'s ``host_sha256``,
which no laptop may rewrite; a new module carries its own provenance instead.

WHY NO KERNEL IS NEEDED
-----------------------
The dispersive ``update_E`` body is ELEMENT-WISE: one cell's ``(D - sum P)``
times one cell's inverse epsilon, then MEEP's ``dsigw`` tail
(``stepping._apply_constitutive_pml``, S:2065-2096). It reads no neighbour at
all. A fold contributes exactly one thing to an element-wise sub-step — the
STORED EXTENT — and the field arrays, the pole arrays and the PML coefficient
vectors already carry it. The layout and coefficient clauses below check that by
name rather than assuming it.

MEASURED, NOT ARGUED. The certified body against ``stepping.update_E``, EIGHT
complete cycles, uint32 over the whole stored inventory (fields, PML auxiliaries,
``f_w``, every pole's ``P`` and ``P_prev``), with a +-0 lattice held LIVE in ``D``
through every cycle so the constitutive leg could not pass by being a no-op —
zero-init is a fixed point of this sub-step, and a leg whose census reaches zero
is reported VACUOUS rather than IDENTICAL::

    C_folded_dispersive_update_E               IDENTICAL   0 / 69120  (>=13824 moved)
    C_folded_dispersive_update_E_ROWSHAPE_2D   IDENTICAL   0 /  9360   (>=1872 moved)

ON TWO SHAPES, and the second one is the point. The first fixture is a synthetic
3-D box, ``[9,16,16]`` with one mirrored axis — NOT "the corpus configuration",
which this file used to claim it was. The four rows this admission is for are
``TestLoadDump`` 2-D at ``[250,126,1]`` with boundary kinds
``('metallic','mirror','periodic')``, so the ROWSHAPE leg puts the same admission
on a 2-D grid with those kinds and a fold, and measures it there too.

``moved`` is the MINIMUM over cycles of what the sub-step alone wrote, bracketed
around that one call on both sides. (results/residual_closure_2026-08-15/device/
bodies/bodies.json; minus0 >= 1728 and >= 234 respectively.)

AND THROUGH THIS MODULE'S OWN BUILDER. Both legs above reach the certified body
by patching the incumbent predicate. The closure round's second leg builds through
:func:`plan_folded_dispersive_constitutive` with
:func:`folded_dispersive_constitutive_coverage` UNPATCHED — a plan of ``None``
would be a failed leg — and measures the same 0 / 69120 and 0 / 9360
(results/residual_closure_2026-08-15/device/newpred/new_predicates.json).

Four slots on four rows, each of which reaches FULL whole-step coverage now that
the arm exists — but all four are ``TestLoadDump`` 2-D cases, which exercise
load/dump machinery rather than distinct physics, so this closure ranks last of
the round's four on physics value even though it is the cheapest to state.

WHAT IS NOT HERE
----------------
No kernel and no arithmetic. The kernel is
``dispersive_update_e.constitutive_step_dispersive``, the plan class is
``DispersiveConstitutivePlan`` and the pole binding is ``LivePoleBinding``, all
three untouched and already certified; only the ADMISSION is this module's — the
shape ``folded_complex.plan_folded_beta_run_constitutive`` established. Nothing
here imports Triton or CuPy at module scope, so the predicate is importable and
answerable on the merge-bar laptop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
    _volume_reasons,
)
from .symmetry import _folded_constitutive_grid_reasons, _has_real_fold

#: The shared clause builders this predicate composes from, named as data so the
#: laptop test can assert every one still exists in the files it borrows from. A
#: rename there then fails at the merge bar instead of silently dropping a clause.
SHARED_CLAUSES = (
    "_call", "_coefficient_reasons", "_inverse_epsilon_reasons",
    "_layout_reasons", "_susceptibility_reasons", "_volume_reasons",
)

#: results/residual_closure_2026-08-15/device/bodies/bodies.json — kept as data so
#: a test can assert the module still states what licensed it.
MEASURED: Dict[str, Any] = {
    "artifact": "results/residual_closure_2026-08-15/device/bodies/bodies.json",
    "case": "C_folded_dispersive_update_E",
    "verdict": "IDENTICAL",
    "differing": 0,
    "compared": 69120,
    #: The per-sub-step MINIMUM over cycles, not the whole loop's footprint.
    "moved": 13824,
    #: The fixture is a synthetic 3-D box, NOT a corpus row. The rows are 2-D, so
    #: the same admission is measured on a row-SHAPED grid as well.
    "fixture_shape": [9, 16, 16],
    "rowshape": {"case": "C_folded_dispersive_update_E_ROWSHAPE_2D",
                 "verdict": "IDENTICAL", "differing": 0, "compared": 9360,
                 "moved": 1872, "fixture_shape": [24, 13, 1],
                 "boundary_kinds": ["metallic", "mirror", "periodic"]},
    #: The same identity through THIS module's builder, predicate unpatched.
    "own_builder": {
        "artifact": ("results/residual_closure_2026-08-15/device/newpred/"
                     "new_predicates.json"),
        "cases": {"NEW_folded_dispersive_update_E": {"differing": 0,
                                                     "compared": 69120},
                  "NEW_folded_dispersive_update_E_ROWSHAPE_2D": {"differing": 0,
                                                                 "compared": 9360}},
    },
    "cycles": 8,
    #: The neighbouring configuration that DIVERGED, and why this predicate
    #: refuses an off-diagonal row rather than admitting one.
    "offdiagonal_control": {"case": "G_folded_offdiag_dispersive_update_E",
                            "verdict": "DIVERGENT",
                            "differing": 13824, "compared": 69120},
}


def folded_dispersive_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """May the CERTIFIED dispersive ``update_E`` kernel step a FOLDED run?

    A conjunction of two certified predicates' clause sets, each restated with
    the one clause it got wrong about the other INVERTED: a real fold is
    REQUIRED (so this cannot compete with ``dispersive_constitutive_coverage``)
    and at least one pole is REQUIRED (so it cannot compete with
    ``symmetry.folded_constitutive_coverage``). Disjoint from both by
    construction, and tests pin both directions.

    STILL REFUSED HERE, BY NAME, and none of it is implied by the two inversions:

    * an off-diagonal ``chi1inv`` row. Fold + off-diagonal + dispersion at
      ``update_E`` is a real corpus configuration (``absorbed_power_density.py``)
      and it DIVERGED — 13824 of 69120 words, first at ``Ex`` word 0 — because
      the row product reads partner volumes through the shift helpers
      (``stepping._offdiagonal_terms``, S:1190-1225). Composing
      ``folded_offdiag_update_e``'s body with the pole sum is what that would
      take, and it is not built;
    * chi2/chi3, BFAST, ``beta``, cylindrical coordinates, Bloch, complex
      storage, an inactive absorber, a magnetic or non-lorentzian/drude
      susceptibility, more than ``MAX_POLES`` poles on one component, a
      malformed ``P``/``P_prev`` volume, and ``stores_E`` false.

    Conductivity is NOT a clause, deliberately and for the reason
    :func:`symmetry._folded_constitutive_grid_reasons` already records: it
    changes the CURL sub-steps, never this one.
    """
    from .dispersive_update_e import MAX_POLES  # noqa: PLC0415

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES["E"]
    reasons: List[str] = list(_folded_constitutive_grid_reasons(fields, pml, grid))
    reasons.extend(_susceptibility_reasons(fields))

    states = tuple(getattr(fields, "polarizations", ()) or ())

    # INVERTED (against symmetry.py:811-813). A pole must be REGISTERED: with none
    # the source is D and the configuration is folded_constitutive_coverage's. Two
    # predicates admitting one configuration means the dispatcher picks a body by
    # ordering, which is how a wrong answer gets chosen at random.
    if not states:
        reasons.append(
            "no susceptibility is registered: update_E's source is D, not "
            "(D - sum P), and that configuration is "
            "folded_constitutive_coverage(side='E')'s")

    # INVERTED (against coverage._grid_reasons clause 5, which
    # dispersive_constitutive_coverage inherits). A real fold is REQUIRED.
    if not _has_real_fold(grid):
        reasons.append(
            "no mirror plane is active: an unfolded dispersive update_E is "
            "dispersive_constitutive_coverage's and this admission must not "
            "overlap it")

    # Element-wise only. The fold does not soften this clause; it sharpens it —
    # the measured 13824/69120 divergence on the folded off-diagonal dispersive
    # configuration is exactly this clause firing on the case that would
    # otherwise slip through.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed: the row product reads "
            "partner volumes through the shift helpers (stepping.py:1219-1254) "
            "and this sub-step is element-wise (measured 13824/69120 words "
            "differing on the folded dispersive off-diagonal configuration)")

    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # Per-component pole count and every pole volume's layout —
    # dispersive_constitutive_coverage's clauses (e) and (f), restated because
    # that predicate refuses this grid outright and cannot be called for its
    # sub-clauses.
    shape = tuple(getattr(grid, "shape", ()))
    per_component: Dict[str, int] = {name: 0 for name in spec["targets"]}
    for index, state in enumerate(states):
        driven = _call(state, "driven", default=None)
        if driven is None:
            reasons.append(
                f"polarization {index} ({type(state).__name__}) does not report "
                f"driven(); an unreadable susceptibility is not a covered one")
            continue
        for name in tuple(driven):
            if name in per_component:
                per_component[name] += 1
            for slot in ("P", "P_prev"):
                array = (getattr(state, slot, {}) or {}).get(name)
                if array is None:
                    reasons.append(
                        f"polarization {index} {slot}[{name}] is not allocated")
                    continue
                if len(shape) != 3:
                    continue  # the shape clause below reports it once
                # coverage's own shape/dtype/contiguity check, REUSED rather than
                # re-stated: a re-statement dropped the contiguity clause once
                # before (no_pml._derive_reasons records the incident), and a
                # reversed view has the right shape and the right dtype and is
                # read in the wrong order by a flat-indexed kernel.
                reasons.extend(_volume_reasons(
                    f"polarization {index} {slot}[{name}]", array, shape))
    for name, count in per_component.items():
        if count > MAX_POLES:
            reasons.append(
                f"{name} is driven by {count} poles, more than the kernel's "
                f"MAX_POLES={MAX_POLES} compiled slots")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_layout_reasons(fields, shape, names))
    if len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
        if pml is not None and getattr(pml, "is_active", False):
            suffix = ("_h",) if spec["half_integer"] else ("",)
            reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


def plan_folded_dispersive_constitutive(fields: Any, pml: Any,
                                        block: Optional[int] = None) -> Any:
    """A CERTIFIED dispersive ``update_E`` plan for a folded run, or None.

    The kernel, the plan class and the LIVE pole binding are
    ``dispersive_update_e``'s, untouched; only the admission is this module's.
    The pole pointers must stay live — ``PolarizationState.update`` rotates
    ``P``/``P_prev``/``_scratch`` every step, so a snapshotted plan is stale
    after the first component of the first step and stale in a way that still
    computes — which is why this builds a ``LivePoleBinding`` rather than
    caching device views the way ``launch.PmlCurlPlan`` may.

    None is the only refusal: a configuration this admission does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not folded_dispersive_constitutive_coverage(fields, pml).covered:
        return None
    from .dispersive_update_e import (  # noqa: PLC0415
        E_TERMS,
        DispersiveConstitutivePlan,
        LivePoleBinding,
        poles_per_component,
    )
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    order = poles_per_component(fields)
    targets = tuple(term[0] for term in E_TERMS)
    return DispersiveConstitutivePlan(
        fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        [getattr(fields, "f_w_" + name) for name in targets],
        [getattr(fields, term[1]) for term in E_TERMS],
        [fields.inverse_epsilon_for(name) for name in targets],
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        LivePoleBinding(fields, order),
    )


def explain_folded_dispersive(fields: Any, pml: Any) -> Coverage:
    """The verdict with its reasons, for reports. Needs no Triton."""
    return folded_dispersive_constitutive_coverage(fields, pml)


# ---------------------------------------------------------------------------
# WIRING — the planner seam is IN; dispatch is held shut by its enable
# ---------------------------------------------------------------------------
#
# Added 2026-08-14 by the residual-group closure round, which did not own
# ``launch.py`` and could add no arm. AS IT NOW STANDS: ``launch.plan_step``'s
# ``update_E`` table carries a ``folded dispersive`` arm gated on
# ``folded_grid_active and dispersion_active``, the package ``__init__`` exports
# the predicate and the builder behind the lazy seam, and
# ``launch.FAMILY_MODULES`` names this module.
#
# THE ARM IS ADDITIVE AND REORDERS NOTHING, which is what the two inverted clauses
# buy: the ``dispersive`` arm refuses a fold (coverage._grid_reasons clause 5) and
# the ``folded`` arm refuses a registered susceptibility (symmetry.py:811-813), so
# this arm can only ever win a slot on which both of them already refused. The
# planner's disjointness sweep drives a real folded dispersive triple through the
# shipped composer and asserts the whole ``selected`` dict, which is what turns
# that from an argument into a measurement.
#
# NEITHER INCUMBENT CLAUSE WAS NARROWED, deliberately. Narrowing ``symmetry.py``'s
# susceptibility clause instead of adding this arm would make TWO predicates admit
# one slot, and ``_select_slot`` fails closed on that: the slot would be left
# UNSELECTED and fall to the array path — a silent coverage LOSS, not an error.
#
# Four slots on four rows (the ``TestLoadDump`` 2-D cases), each reaching full
# whole-step coverage from this arm: their curls and ``update_H`` are
# already admitted by ``symmetry.folded_pml_curl_coverage`` and
# ``folded_constitutive_coverage(side='H')``, and their ``update_P`` by
# ``coverage.ade_update_p_coverage``.
#
# Dispatch is a separate question and is unchanged by this: ``fastpath`` reaches
# the composer from the driver, and what holds it shut is
# ``fastpath.DISPATCH_BY_DEFAULT`` being False.
#
# NO ENTRY IN ``fingerprints.json`` IS CLAIMED, deliberately and for the reason
# that record already states about the families wired on 2026-08-13: provenance
# for a specialized module lives with its own gate trio, and ``host_sha256`` names
# bytes that ran on a GPU. This module plans a CERTIFIED kernel, so what its gate
# has to add is the admission leg, not a new kernel fingerprint.
