"""The ADE ``update_P`` recurrence when the split-field absorber is INERT.

WHY THIS MODULE EXISTS AND WHY IT IS NOT IN ``coverage.py``
----------------------------------------------------------
:func:`coverage.ade_update_p_coverage` covers exactly this sub-step and refuses
this configuration for ONE clause (coverage.py:584-587)::

    drive = getattr(fields, "f_w_" + component, None)
    if drive is None:
        reasons.append(f"{drive_name} is not allocated (the drive field under PML)")

That clause pins the EXISTENCE of ``f_w_*``. What the kernel actually needs is
the LAYOUT of whatever ``Fields.drive_field`` hands it — and ``drive_field``
(fields.py:1158-1162) returns ``f_w_*`` under an active layer and the STORED E
without one::

    if self._pml_active:
        return getattr(self, 'f_w_' + component)
    return getattr(self, component)

Under ``mp.Absorber`` — a conductivity, no split-field PML — there is no ``f_w``
at all, so the shipped predicate refuses a configuration whose kernel body is
IDENTICAL. Only which pointer is bound changes, and the binding is
``AdeUpdatePPlan.run``'s (launch.py:1443-1458), which takes ``drive`` as a
callable and never reaches for a field itself; ``PolarizationState.update``
(dispersion.py:658) takes it the same way.

``coverage.py`` is another session's file this round, so the clause is RESTATED
here with that one clause inverted rather than weakened there — the house
pattern ``folded_constitutive_coverage`` and ``nonlinear_update_e``'s
``_nonlinear_grid_reasons`` established. The incumbent's refusals all stay true.

MEASURED, NOT ARGUED
--------------------
Certified body against ``dispersion.PolarizationState.update``, EIGHT complete
cycles, uint32 over the whole stored inventory
(results/residual_closure_2026-08-15/device/bodies/bodies.json). ``moved`` is the
MINIMUM over cycles of what the sub-step alone wrote — bracketed around that one
call, on both sides — so a frozen update cannot borrow the perturbation's
footprint::

    B_ade_update_p_no_pml               IDENTICAL      0 / 7680   (>=3072 moved)
    H_ade_update_p_conductive_no_pml    IDENTICAL      0 / 7680   (>=3072 moved)
    B_ade_wrong_drive_CONTROL           DIVERGENT   3072 / 15360

AND THROUGH THIS MODULE'S OWN BUILDER. Those three reach the certified body by
patching the incumbent predicate, which licenses a claim about the kernel, not
about the admission below. The closure round's second leg builds through
:func:`plan_no_pml_ade_update_p` with :func:`no_pml_ade_update_p_coverage`
UNPATCHED — a plan of ``None`` would be a failed leg — and measures the same
(results/residual_closure_2026-08-15/device/newpred/new_predicates.json)::

    NEW_no_pml_ade_update_P              IDENTICAL     0 / 7680   (>=3072 moved)
    NEW_no_pml_ade_update_P_conductive   IDENTICAL     0 / 7680   (>=3072 moved)

The CONTROL is the load-bearing half of this module. It bound the STORED E as the
drive under an ACTIVE layer — "the single most likely silent wrong answer in
dispersion" (fields.py:1149-1156), because the two agree exactly outside the
absorber — and it diverged. So the inverted clause must keep REQUIRING ``f_w``
wherever the layer is active, and it does: an active layer is refused outright
below, by name, and belongs to the incumbent.

The conductive case is measured rather than inferred because that is the only
piece of residual groups (F) and (H) that closes at all: those rows still need a
conductive no-PML curl and a STORE arm for ``update_E``, neither of which is
built, and both of which DIVERGED (1536/4608 per curl sub-step; 1536/7680 on the
store arm).

WHAT IS NOT HERE
----------------
No kernel and no arithmetic. The kernel is ``kernels.ade_update_p``, the plan is
``launch.AdeUpdatePPlan``, both untouched and both already certified; only the
ADMISSION is this module's. Nothing here imports Triton or CuPy, so the predicate
is importable and answerable on the merge-bar laptop.
"""

from __future__ import annotations

from typing import Any, List, Optional

from . import coverage as _coverage

#: The shared clause builders this predicate composes from, named as data so the
#: laptop test can assert every one still exists in the other session's file. A
#: rename in ``coverage.py`` then fails at the merge bar instead of silently
#: dropping a clause here.
SHARED_CLAUSES = (
    "_call", "_volume_reasons", "ELECTRIC_COMPONENTS",
    "COVERED_SUSCEPTIBILITY_KINDS",
)

#: The measurement that licenses this admission, and the CONTROL that makes it
#: non-vacuous. Kept as data so the test can assert the module still states it.
MEASURED: dict = {
    "artifact": "results/residual_closure_2026-08-15/device/bodies/bodies.json",
    #: ``moved`` is the per-sub-step MINIMUM over cycles, on both sides, not the
    #: whole loop's footprint.
    "cases": {
        "B_ade_update_p_no_pml": {"verdict": "IDENTICAL",
                                  "differing": 0, "compared": 7680, "moved": 3072},
        "H_ade_update_p_conductive_no_pml": {"verdict": "IDENTICAL",
                                             "differing": 0, "compared": 7680,
                                             "moved": 3072},
        "B_ade_wrong_drive_CONTROL": {"verdict": "DIVERGENT",
                                      "differing": 3072, "compared": 15360,
                                      "moved": 3072},
    },
    #: The same identity, re-measured through THIS module's builder with THIS
    #: module's predicate unpatched. This is what licenses the admission; the
    #: block above licenses only the kernel body.
    "own_builder": {
        "artifact": ("results/residual_closure_2026-08-15/device/newpred/"
                     "new_predicates.json"),
        "cases": {
            "NEW_no_pml_ade_update_P": {"verdict": "IDENTICAL", "differing": 0,
                                        "compared": 7680, "moved": 3072},
            "NEW_no_pml_ade_update_P_conductive": {"verdict": "IDENTICAL",
                                                   "differing": 0, "compared": 7680,
                                                   "moved": 3072},
        },
    },
    "cycles": 8,
}


def _inert_layer_reasons(fields: Any, pml: Any, component: str) -> List[str]:
    """The INVERTED clause, in three parts, none of which implies the others.

    ``coverage.ade_update_p_coverage`` gets all three for free from one
    ``f_w is not None`` test. Split apart, because the three failure modes are
    different and only one of them is the one being inverted:

    a. the LAYER must be inert. An active split-field absorber is the incumbent's
       configuration, and admitting it here would put two predicates on one
       sub-step — which is how a dispatcher picks a body by ordering. Disjoint by
       construction, and the CONTROL leg says what the wrong pick costs: 3072 of
       15360 words;
    b. ``Fields`` must not be in PML STORAGE MODE either. ``enable_pml_storage``
       is a one-way switch (fields.py:678-721) that sets ``_pml_active`` — the
       flag ``drive_field`` branches on. A ``Fields`` switched on while the layer
       handed to the stepper is inert would drive P from ``f_w``, which
       ``update_E`` never writes on that path (it takes the
       ``field[...] = constitutive`` branch, stepping.py:1019-1022): a frozen drive,
       smooth and wrong. ``no_pml._no_pml_grid_reasons`` clause 3b refuses the
       mirror image of this hazard for the curl;
    c. the drive field must BE the stored E array, by IDENTITY. This is the clause
       that replaces the incumbent's existence test, and it is asked of
       ``Fields.drive_field`` itself rather than derived — the launcher calls that
       method, so this predicate checks what the launcher will read rather than
       what this module believes it will read.
    """
    reasons: List[str] = []

    if pml is not None and getattr(pml, "is_active", False):
        reasons.append(
            "an active PML layer is installed: under one the drive field is "
            "f_w_* and that configuration is coverage.ade_update_p_coverage's "
            "(binding the stored E there diverged 3072/15360 words)")

    if bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "Fields is in PML storage mode: drive_field would return f_w_"
            f"{component}, which update_E does not write on the no-absorber "
            "path (stepping.py:1019-1022) — a frozen drive field")

    drive_name = "f_w_" + component
    if getattr(fields, drive_name, None) is not None:
        reasons.append(
            f"{drive_name} is allocated: this admission covers the run that has "
            f"no split-field auxiliary, and the one that has it is the "
            f"incumbent predicate's")

    reader = getattr(fields, "drive_field", None)
    if not callable(reader):
        reasons.append("fields does not expose drive_field(); an unreadable "
                       "drive is not a covered one")
        return reasons
    try:
        drive = reader(component)
    except Exception as exc:  # noqa: BLE001 - unreadable means not covered
        reasons.append(f"drive_field({component!r}) raised {exc!r}")
        return reasons
    if drive is None:
        reasons.append(f"drive_field({component!r}) is None")
        return reasons
    stored = getattr(fields, component, None)
    if stored is None or drive is not stored:
        reasons.append(
            f"drive_field({component!r}) is not the stored {component} array; "
            f"without an absorber MEEP's w IS the constitutive product update_E "
            f"wrote (fields.py:1158-1162), and anything else is a different "
            f"drive")
    return reasons


def no_pml_ade_update_p_coverage(fields: Any, pml: Any, state: Any,
                                 component: str) -> "_coverage.Coverage":
    """May the CERTIFIED ADE kernel advance this (state, component) with no absorber?

    :func:`coverage.ade_update_p_coverage`'s clause set, RESTATED, with the
    drive-field clause inverted from "``f_w_*`` exists" to "the layer is inert,
    ``Fields`` is not in PML storage mode, and ``drive_field`` hands back the
    stored E with the layout the kernel indexes".

    Every other clause is kept verbatim in force, and this predicate REFUSES, by
    name:

    * complex storage. Group (I)'s leg did not diverge, it RAISED —
      ``KeyError: 'complex64'`` — because the certified body is float32 while
      ``P``/``P_prev``/``_scratch`` are complex64 there. That is an unbuilt
      kernel, not a rounding gap, and calling it a divergence would misreport it;
    * a component outside ``ELECTRIC_COMPONENTS``, a state that does not drive
      it, a susceptibility kind outside ``{lorentzian, drude}``;
    * a missing or non-finite ``(c_now, c_prev, c_drive)`` triple, and
      coefficients built for a different ``dt`` than the grid carries;
    * a non-3-D grid, a cell count past the kernel's int32 index range, and any
      of ``P``/``P_prev``/``_scratch`` that is not float32, C-contiguous and
      exactly ``grid.shape``;
    * a ``sigma`` that is neither a scalar nor a volume.

    CONDUCTIVITY IS ADMITTED, deliberately and by measurement. ``mp.Absorber``
    installs one and it enters the CURL sub-steps
    (``stepping._apply_conductive_update``, S:1938-1951), never this one; the
    conductive fixture measured 0 of 7680 words differing over four cycles with a
    conductivity on both D and B. It is named here rather than left silent
    because "no clause fired" and "this feature was considered" are different
    statements, and only the second one is coverage.

    THE FOLD, cylindrical coordinates, Bloch and BFAST are NOT clauses here and
    that is not an oversight: this sub-step is element-wise with no stencil, no
    neighbour read and no boundary pass — everything a boundary does to P arrives
    through the drive field W, which already carries it (the incumbent predicate
    carries no such clause either, for the same reason). What a fold does change
    is the stored extent, and the extent is checked: every buffer must be exactly
    ``grid.shape``, which on a folded grid IS the folded extent.
    """
    reasons: List[str] = []
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")
    if getattr(fields, "force_complex_fields", False):
        reasons.append(
            "force_complex_fields=True: the certified body is float32 while P, "
            "P_prev and _scratch are complex64 there, and the launch raises "
            "KeyError: 'complex64' — an unbuilt kernel, not a rounding gap")

    if component not in _coverage.ELECTRIC_COMPONENTS:
        reasons.append(f"component {component!r} is outside "
                       f"{_coverage.ELECTRIC_COMPONENTS}")
        return _coverage.Coverage(False, tuple(reasons))

    if not _coverage._call(state, "drives", component, default=False):
        reasons.append(f"this susceptibility does not drive {component}")

    kind = getattr(getattr(state, "susceptibility", None), "kind", None)
    if kind not in _coverage.COVERED_SUSCEPTIBILITY_KINDS:
        reasons.append(f"kind {kind!r} is outside "
                       f"{_coverage.COVERED_SUSCEPTIBILITY_KINDS}")

    coefficients = getattr(state, "_coefficients", None)
    if coefficients is None or len(tuple(coefficients)) != 3:
        reasons.append("the (c_now, c_prev, c_drive) coefficient triple is missing")
    else:
        for name, value in zip(("c_now", "c_prev", "c_drive"), tuple(coefficients)):
            try:
                number = float(value)
            except Exception:  # noqa: BLE001 - a non-numeric coefficient is not coverage
                reasons.append(f"{name}={value!r} is not a float")
                continue
            if number != number or number in (float("inf"), float("-inf")):
                reasons.append(f"{name}={number!r} is not finite")

    dt = getattr(grid, "dt", None)
    if dt is None:
        reasons.append("grid carries no dt")
    elif getattr(getattr(state, "grid", None), "dt", dt) != dt:
        reasons.append("the state's coefficients were built for a different dt")

    # THE INVERTED CLAUSE.
    reasons.extend(_inert_layer_reasons(fields, pml, component))

    # Without an absorber ``update_E`` writes the stored E only when ``stores_E``
    # is set; otherwise it returns at stepping.py:984 and the array the drive
    # points at is never refreshed. The incumbent needs no such clause — an active
    # PML forces storage — so this one is genuinely new here rather than restated.
    if not getattr(fields, "stores_E", False):
        reasons.append(
            "E is recomputed from D rather than stored: update_E returns at "
            "stepping.py:984 and the drive field would never be written")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return _coverage.Coverage(False, tuple(reasons))
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if total >= 2 ** 31:
        reasons.append(f"{total} cells exceeds the kernel's int32 index range")

    buffers = {
        "P": (getattr(state, "P", {}) or {}).get(component),
        "P_prev": (getattr(state, "P_prev", {}) or {}).get(component),
        "_scratch": getattr(state, "_scratch", None),
    }
    for name, array in buffers.items():
        if array is None:
            reasons.append(f"{name}[{component}] is not allocated")
            continue
        reasons.extend(_coverage._volume_reasons(f"{name}[{component}]", array, shape))

    drive = None
    reader = getattr(fields, "drive_field", None)
    if callable(reader):
        try:
            drive = reader(component)
        except Exception:  # noqa: BLE001 - already reported by the clause above
            drive = None
    if drive is not None:
        reasons.extend(_coverage._volume_reasons(
            f"drive_field({component!r})", drive, shape))

    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    if sigma is None:
        reasons.append(f"sigma[{component}] is missing")
    elif getattr(sigma, "shape", ()):
        reasons.extend(_coverage._volume_reasons(f"sigma[{component}]", sigma, shape))
    else:
        try:
            float(sigma)
        except Exception:  # noqa: BLE001 - neither a scalar nor a volume
            reasons.append(
                f"sigma[{component}]={sigma!r} is neither a scalar nor a volume")

    return _coverage.Coverage(not reasons, tuple(reasons))


def plan_no_pml_ade_update_p(fields: Any, pml: Any, state: Any,
                             block: Optional[int] = None) -> Any:
    """A CERTIFIED ADE plan for an absorber-free run, or None.

    ALL OR NOTHING PER STATE, for ``launch.plan_ade_update_p``'s reason and not a
    new one: ``PolarizationState.update`` rotates a SHARED scratch buffer from
    component to component inside one call, so covering two of three components
    and leaving the third to the array path would interleave two rotations over
    one buffer set.

    The plan class is ``launch.AdeUpdatePPlan``, untouched — including its
    per-launch pointer resolution, which is what keeps it correct across the
    three-buffer rotation. The caller drives it as ``plan.run(fields.drive_field)``,
    exactly as it would under a PML; the difference this module admits is entirely
    in what that bound method returns.
    """
    driven = getattr(state, "driven", None)
    components = tuple(driven()) if callable(driven) else ()
    if not components:
        return None
    for component in components:
        if not no_pml_ade_update_p_coverage(fields, pml, state, component).covered:
            return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import AdeUpdatePPlan  # noqa: PLC0415

    return AdeUpdatePPlan(state, components, fields.grid.shape,
                          DEFAULT_BLOCK if block is None else block)


def explain_no_pml_ade(fields: Any, pml: Any, state: Any,
                       component: str) -> "_coverage.Coverage":
    """The verdict with its reasons, for reports. Needs no Triton."""
    return no_pml_ade_update_p_coverage(fields, pml, state, component)


# ---------------------------------------------------------------------------
# WIRING — the planner seam is IN; dispatch is held shut by its enable
# ---------------------------------------------------------------------------
#
# Added 2026-08-14 by the residual-group closure round, which did not own
# ``launch.py`` and could add no arm. AS IT NOW STANDS: ``launch.plan_step``'s
# ``update_P`` block decides its ADE family ONCE for the whole sub-step, from the
# two predicates rather than from builder order, and this one owns the slot where
# the incumbent refuses for the drive-field clause. The package ``__init__``
# exports the predicate and the builder behind the same lazy seam, and
# ``launch.FAMILY_MODULES`` names this module.
#
# WHY THE PREDICATES ARE CONSULTED THERE AT ALL, when the block is builder-first
# everywhere else: a builder-first FALLBACK would silently prefer the incumbent if
# both families ever admitted, which is ``plan_step``'s clause (b) failure mode
# spelled "pick by order". Both verdicts are taken before anything is built, and a
# double admission leaves the slot UNSELECTED — the array path, which is correct.
#
# The gate on the arm is ``launch.absorber_inactive``, and it is NECESSARY in the
# gate rule's sense for this family too: :func:`_inert_layer_reasons` refuses an
# active layer BY NAME, so a closed gate can only ever remove a refusal.
#
# ``coverage.py`` was deliberately NOT narrowed. Its clause :584-587 could be
# rewritten to say the drive field must be ``f_w`` only WHEN THE LAYER IS ACTIVE,
# and that would make this module's restatement redundant — but it would also make
# the two predicates admit the SAME configuration, which the composer fails closed
# on. The scope change is made by adding an arm, never by widening one.
#
# Three slots on three rows — ``absorber-1d.py``, ``TestAbsorber.test_absorber``
# and ``material-dispersion.py``. None of the three reaches full whole-step
# coverage from this alone: ``absorber-1d.py`` and ``TestAbsorber.test_absorber``
# also need the conductive no-PML curl (measured 1536/4608 differing per curl
# sub-step) and ``material-dispersion.py`` the STORE arm of ``update_E``
# (1536/7680). Both are genuinely unbuilt kernels and neither is built here.
#
# Dispatch is a separate question and is unchanged by this: ``fastpath`` reaches
# the composer from the driver, and what holds it shut is
# ``fastpath.DISPATCH_BY_DEFAULT`` being False.
