"""Which configurations the Triton kernels may step.

The predicates are written POSITIVELY: every requirement is named and checked, and
coverage is never inferred from the absence of a known blocker. That direction
matters because almost everything these kernels do not carry produces a *silent
wrong answer* rather than a crash — a folded run stepped by the curl kernel still
returns a smooth, plausible field, it is just not the one MEEP computes.

Three predicates live here, one per kernel:

* :func:`pml_curl_coverage` — the real-field split-field PML curl (``step_B`` and
  ``step_D``);
* :func:`constitutive_coverage` — the ``dsigw`` constitutive accumulation
  (``update_H`` and ``update_E``);
* :func:`ade_update_p_coverage` — one Lorentz/Drude pole's ADE recurrence
  (``update_P``), per (state, component).

COVERAGE IS A SET PER SUB-STEP, NOT PER RUN. A feature that changes one sub-step's
semantics disqualifies THAT sub-step; the others may still dispatch, because the
array path and the kernels compute the same bits and the sub-steps are separated
by array-path boundary work in the driver's loop. Dispersion is the case that
forced the distinction: it changes ``update_E`` (source becomes ``D - sum P``) and
adds ``update_P``, and it changes the curl not at all — the curl differences the
STORED E and H arrays, which exist under PML whether or not a pole is registered.

DISPERSION AND OFF-DIAGONAL EPSILON ARE ADMITTED FOR THE CURL, DELIBERATELY.
Both are *constitutive-only* features: their entire effect is inside ``update_E``
(``stepping.py:996-1022``, ``fields.py:1096-1105``), and both force stored E
(``fields.py:676``/``:1255``), which is the invariant the curl actually depends on.
Refusing one while silently admitting the other — which is what this file did
until the composability pass — was an inconsistency, not a safety margin.

Nothing here imports Triton, CuPy or the kernel module: the predicates are the part
that has to be readable, testable and importable on any machine, including one
with no GPU and no optional dependency installed.

The mutations this file's tests apply — widen the boundary set by one axis, drop
one covered-configuration check, widen the narrowed dispersion clause one step too
far — are what make a predicate more than a function that returns True.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Sequence, Tuple
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? YES, and the
#: wiring this claims is ``launch._install_fused_pair`` (``launch.py:1639``): when
#: ``deposit_repair.in_seam_sources`` is non-empty it puts a ``LeadingRepairPlan`` in
#: this pair's FIRST slot (``step_B`` / ``step_D``), which saves the field and ``f_w``
#: at every deposit point immediately before the launch, and a ``TrailingRepairPlan``
#: in the SECOND (``update_H`` / ``update_E``), which recomputes them after the driver
#: has injected, filled symmetry and cleared walls. Both pairs this predicate answers
#: for are installed through that one call (``launch.py:2967``), so there is no route
#: by which a fused pair from this module reaches the device unbracketed.
#:
#: THE PRECONDITION THIS PRODUCT MEETS is that its constitutive half is POINTWISE, on
#: both sides of the step: ``update_H`` is ``H = B/mu`` cell by cell
#: (``stepping.py:947-951``) and the diagonal ``update_E`` is ``(D - sum P) * inv_eps``
#: cell by cell (``stepping.py:1012-1015``), while the kernel's own half reads six values
#: at ``+ idx`` and its coefficients at ``+ i``/``+ j``/``+ k``
#: (``kernels.py:478-483``). No neighbour read on either side, so a whole-grid launch
#: against an uninjected field is wrong ONLY at the deposit points and nowhere else.
#: The two configurations that are NOT pointwise -- an off-diagonal chi1inv row, whose
#: ``update_E`` is a stencil over the partner volumes, and an instantaneous chi2/chi3 --
#: are refused per run by ``deposit_repair.repairable``, which the clause below
#: consults rather than assumes.
#:
#: IT IS NOT A HINT. A product that passes True without those wrappers computes the
#: constitutive half against a pre-injection field and reports success, which is the
#: exact failure ``deposit_repair`` exists to prevent. Flag and wiring move together.
CARRIES_DEPOSIT_REPAIR = True

# The ghost rules the curl kernel implements. ``stepping._boundary_kinds`` can also
# return "mirror" (a folded axis) and "cylindrical-axis"; both are refused.
COVERED_BOUNDARIES: Tuple[str, ...] = ("periodic", "metallic")

# The six curl targets, and the six auxiliaries their split-field recurrence needs.
CURL_TARGETS: Tuple[str, ...] = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
CURL_SUB_STEPS = {
    "step_B": ("Bx", "By", "Bz"),
    "step_D": ("Dx", "Dy", "Dz"),
}

B_SOURCES: Tuple[str, ...] = ("Ex", "Ey", "Ez")
D_SOURCES: Tuple[str, ...] = ("Hx", "Hy", "Hz")

#: The susceptibility kinds the ADE kernel's three-term recurrence covers. Kept as
#: a literal rather than imported so this module stays engine-import-free; a test
#: pins it against ``dispersion.SUSCEPTIBILITY_KINDS`` so the two cannot drift.
COVERED_SUSCEPTIBILITY_KINDS: Tuple[str, ...] = ("lorentzian", "drude")

#: A polarization may drive these and only these. ``PolarizationState`` allocates
#: over E components only (dispersion.py:640-647) and ``from_meep`` refuses
#: ``H_susceptibilities`` (from_meep.py:1566-1572) — so the clause below is vacuous
#: TODAY, and is written anyway: this module's own rule forbids inferring coverage
#: from another module's guard.
ELECTRIC_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")
MAGNETIC_COMPONENTS: Tuple[str, ...] = ("Bx", "By", "Bz", "Hx", "Hy", "Hz")

#: The constitutive sub-steps, keyed as ``constitutive_coverage``'s ``side``.
#: ``targets`` are written, ``aux`` is ``f_w_*`` (this sub-step's ``fu``), ``sources``
#: is what the constitutive product reads. ``half_integer`` selects the Yee
#: sub-lattice the coefficient pair comes from — integer for H, half-integer for E
#: (``stepping.py:948`` vs ``:986``); swapped, it is a half-cell error in the
#: absorber profile, converged and smooth and wrong.
CONSTITUTIVE_SIDES = {
    "H": {
        "targets": ("Hx", "Hy", "Hz"),
        "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
        "sources": ("Bx", "By", "Bz"),
        "half_integer": False,
    },
    "E": {
        "targets": ("Ex", "Ey", "Ez"),
        "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
        "sources": ("Dx", "Dy", "Dz"),
        "half_integer": True,
    },
}


class Coverage(NamedTuple):
    """Verdict plus the reasons, so a refusal can be reported rather than guessed at."""

    covered: bool
    reasons: Tuple[str, ...]

    def __bool__(self) -> bool:  # `if coverage:` reads as the verdict.
        return self.covered


def _boundary_kinds(grid: Any, pml: Any) -> Optional[Tuple[str, str, str]]:
    """Resolve the per-axis ghost rule through ``stepping``'s own function.

    Imported lazily and defensively: this is the ONE place the predicates consult
    the array path, and they do so because re-deriving the fold/declaration
    precedence here would be a second transcription that could drift from the
    engine it is gating. A raised exception is itself a refusal — a configuration
    the array path will not resolve is not one a kernel may step.
    """
    try:
        from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no stepping, no coverage
        return None
    try:
        return resolve(grid, pml)
    except Exception:  # noqa: BLE001 - an unresolvable grid is refused, not crashed on
        return None


def _grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every kernel in this package shares.

    Backend, storage width, the absorber, the ghost rules, the fold, the coordinate
    system, Bloch, the instantaneous nonlinearity, BFAST and beta. Conductivity is
    deliberately per curl sub-step and is checked by :func:`pml_curl_coverage`;
    it does not alter either constitutive sub-step.
    Shared because a configuration whose ghost rule the curl refuses must not be
    half-fused: an individual sub-step may fall out of a plan, but the GRID the plan
    is built for is one grid and either every kernel here understands it or none
    does.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernels launch against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. The recurrences are the same shape in complex64, the STORAGE
    #    is not, and a complex run stepped as float32 reads the wrong stride.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")

    # 3. An absorber that actually absorbs. Without one the plain path is the
    #    bit-identical one (stepping._pml_is_active).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (these kernels implement the split-field path only)")

    # 4. Only the two ghost rules the curl kernel writes. The constitutive sub-steps
    #    read no neighbour and would not need this — it is kept because coverage is a
    #    set, not a la carte, and a grid whose ghost rule one kernel refuses must not
    #    be half-fused by another.
    kinds = _boundary_kinds(grid, pml if (pml is not None and getattr(pml, "is_active", False))
                            else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside {COVERED_BOUNDARIES}")

    # 5. No mirror plane anywhere: a fold changes the ghost rule, adds a parity mask
    #    at cell 0, adds two driver passes to the loop, and CHANGES THE STORED EXTENT.
    #    That last one is why the constitutive kernels refuse it too even though their
    #    sub-step reads no neighbour: a folded axis changes n_a and therefore every
    #    cell's coefficient index — a smooth, plausible, entirely wrong field.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian only. Same reason as 5 on the constitutive side: an axial extent
    #    moves the coefficient index.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0. A Bloch phase needs complex storage and multiplies one wrapped plane.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. No instantaneous nonlinearity. The Pade factor REPLACES the constitutive
    #     product (stepping.py:999-1000) and composes with dispersion, so it is refused
    #     on every sub-step rather than only on update_E.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not carried)")

    # 11/12. BFAST adds a second additive term to every curl; beta adds out-of-plane
    #        couplings. Both are silent additions, not errors.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    return reasons


def _susceptibility_reasons(fields: Any) -> List[str]:
    """The shape a registered susceptibility must have for ANY kernel here.

    NOT a refusal of dispersion — see the module docstring. What is refused is a
    susceptibility whose *kind* or whose *target set* this package has not
    transcribed. Both clauses are vacuous against the engine as it stands and both
    are written anyway, because "another module already guards it" is exactly the
    reasoning this file exists to refuse.
    """
    reasons: List[str] = []
    states = getattr(fields, "polarizations", ()) or ()
    for index, state in enumerate(states):
        # 9a. Electric targets only. A magnetic susceptibility would drive B through
        #     a sub-step that does not exist here, and the curl would difference an
        #     H array something else was writing.
        driven = _call(state, "driven", default=None)
        if driven is None:
            reasons.append(
                f"polarization {index} ({type(state).__name__}) does not report driven(); "
                "an unreadable susceptibility is not a covered one")
        else:
            for name in tuple(driven):
                if name not in ELECTRIC_COMPONENTS:
                    reasons.append(
                        f"polarization {index} drives {name!r}, outside {ELECTRIC_COMPONENTS}")
        for name in MAGNETIC_COMPONENTS:
            if _call(state, "drives", name, default=False):
                reasons.append(f"polarization {index} drives magnetic component {name}")
        # 9b. Only the two recurrence kinds the ADE kernel transcribes.
        kind = getattr(getattr(state, "susceptibility", None), "kind", None)
        if kind not in COVERED_SUSCEPTIBILITY_KINDS:
            reasons.append(
                f"polarization {index} kind {kind!r} is outside "
                f"{COVERED_SUSCEPTIBILITY_KINDS}")
    return reasons


def pml_curl_coverage(fields: Any, pml: Any,
                      sub_step: Optional[str] = None) -> Coverage:
    """Is this (fields, pml) pair one the Triton real-field PML curl may step?

    Every clause corresponds to a line of the target spec's covered-configuration
    predicate; a failing clause appends its own reason and the scan continues, so a
    refusal reports everything that disqualified the run rather than the first thing.

    Dispersion is ADMITTED. The curl differences the STORED E and H arrays and never
    reads a polarization, an inverse epsilon or ``displacement_minus_polarization``:
    ``stepping.step_B``/``step_D``/``_apply_curl``/``_curl_operands``/
    ``_curl_from_operands``/``_mask_non_owned_cells``/``_apply_pml_update`` mention P
    nowhere. What dispersion changes is the VALUES ``update_E`` writes into the array
    the curl reads, which is not an operation this kernel performs.
    """
    if sub_step is not None and sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _grid_reasons(fields, pml, grid)

    # 8. Conductivity changes only the curl whose primary targets carry it. The
    # opposite curl and both constitutive sub-steps remain ordinary products. When
    # no sub-step is supplied this retains the conservative aggregate verdict used
    # by reports; builders and composers always ask for one named sub-step.
    conductive_targets = (CURL_SUB_STEPS[sub_step]
                           if sub_step is not None else CURL_TARGETS)
    reader = getattr(fields, "condfac_for", None)
    if callable(reader):
        for target in conductive_targets:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}; this curl belongs "
                    f"to the conductive PML product")
    if (sub_step in (None, "step_B")
            and getattr(fields, "has_magnetic_conductivity", False)
            and not callable(reader)):
        reasons.append(
            "a magnetic (B) conductivity is installed but condfac_for is unavailable")

    # 9a/9b. A registered susceptibility must be one this package understands.
    reasons.extend(_susceptibility_reasons(fields))

    # 9c. STORED E — the load-bearing invariant behind admitting dispersion.
    #     The B curl may difference E while a pole is live ONLY because E is a stored
    #     array that update_E wrote, never a `D * inv_eps` product recomputed inside
    #     the curl sub-step. `Fields.enable_pml_storage` (fields.py:678) calls
    #     `enable_field_storage` (:700 -> `_stored_E = True` at :676), so an active PML
    #     already forces it — but the clause is the REASON, not the redundancy: an
    #     edit that ever makes stores_E optional under PML must be caught HERE.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 13. The six PML auxiliaries and the six curl sources.
    for name in tuple("fu_" + target for target in CURL_TARGETS) + B_SOURCES + D_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    # 14. Layout. Shape and contiguity are what the flat index arithmetic assumes.
    shape = tuple(getattr(grid, "shape", ()))
    volumes = (tuple(CURL_TARGETS) + tuple("fu_" + t for t in CURL_TARGETS)
               + B_SOURCES + D_SOURCES)
    reasons.extend(_layout_reasons(fields, shape, volumes))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the Triton constitutive kernel step ``update_H`` (side='H') / ``update_E`` ('E')?

    The sub-step is ``stepping._apply_constitutive_pml`` (stepping.py:2112), MEEP's
    ``step_update_EDHB`` with ``dsigw`` active::

        fwprev = fw[i]; fw[i] = g[i] * u[i];
        f[i] += (kap+sig)*fw[i] - (kap-sig)*fwprev

    It reads no neighbour, so it needs no ghost rule of its own — but it is refused
    on a folded or cylindrical grid all the same, because those change the stored
    extent and therefore the coefficient index on the component's own axis.

    The E side additionally refuses everything that changes what ``source`` is:

    * a registered polarization — ``source`` becomes ``D - sum P``
      (``fields.py:1096-1105``, ``stepping.py:1010``) and ``update_P`` closes the step.
      That configuration belongs to the ADE kernel and the two must not silently
      overlap;
    * an instantaneous chi2/chi3 — the Pade factor replaces the product
      (``stepping.py:999-1000``); refused in :func:`_grid_reasons` for every sub-step;
    * an off-diagonal chi1inv row — the row product reads the OTHER components'
      volumes at neighbouring cells and the sub-step stops being element-wise
      (``stepping.py:1001-1008``). Admitted by the CURL predicate, deliberately, and
      refused here, deliberately: that per-sub-step split is the whole point;
    * ``stores_E`` false — with PML off and no susceptibility ``update_E`` writes
      ``field[...] = constitutive`` instead (``stepping.py:1022``), a different
      sub-step. (PML is already required, so this is belt and braces with a reason.)
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _grid_reasons(fields, pml, grid)
    reasons.extend(_susceptibility_reasons(fields))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is (D - sum P), not D "
                "(fields.py:1096-1105) — that configuration belongs to the ADE kernel")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product reads "
                "neighbours; this sub-step is element-wise)")
        if not getattr(fields, "stores_E", False):
            reasons.append("E is recomputed from D rather than stored")

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


#: The magnetic field type ``sources.VolumeSource.field_type`` reports. Kept as a
#: literal so this module stays engine-import-free; a test pins it against
#: ``sources.FIELD_TYPE_B`` so the two cannot drift.
MAGNETIC_FIELD_TYPE = "B"

#: Which cross-sub-step pairs :func:`fused_pair_coverage` knows about. They are
#: enumerated independently because the D/E pair carries inverse epsilon, a
#: two-components-per-wall metallic wipe and the opposite source seam.
FUSED_PAIRS = {
    "B": {"curl": "step_B", "constitutive": "H",
          "source_family": "magnetic", "source_field_type": "B"},
    "D": {"curl": "step_D", "constitutive": "E",
          "source_family": "electric", "source_field_type": "D"},
}


def zero_metal_axes(grid: Any) -> Tuple[bool, bool, bool]:
    """The ``ZM_X``/``ZM_Y``/``ZM_Z`` constexprs, decided the way ``_zero_metal`` decides.

    ``stepping._zero_metal`` (stepping.py:2253) returns early unless
    ``grid.has_metallic``, then clears stored cell 0 on every axis that is
    ``is_metallic(axis) and not is_mirrored(axis)``. One function so the predicate
    and the compile-time choice cannot disagree — the same discipline
    :func:`sigma_is_volume` exists for, and for the same reason: getting them out
    of step is a plane of wrong values, not a crash.

    It is deliberately NOT ``_boundary_kinds``. That function resolves the GHOST
    rule, which a fold outranks and an invariant axis softens to periodic; this
    one asks the grid's own declaration, which is the question ``_zero_metal``
    asks.
    """
    if not bool(_call(grid, "has_metallic", default=False)):
        return (False, False, False)
    return tuple(  # type: ignore[return-value]
        bool(_call(grid, "is_metallic", axis, default=False))
        and not bool(_call(grid, "is_mirrored", axis, default=False))
        for axis in range(3))


def fused_pair_coverage(fields: Any, pml: Any, pair: str = "B",
                        sources: Any = None) -> Coverage:
    """May one curl + metallic wipe + constitutive pair step this run?

    Positive clauses only, and every one of them is a conjunction of predicates
    that already exist:

    1. :func:`pml_curl_coverage` — the curl half is the curl kernel, unchanged;
    2. :func:`constitutive_coverage` on the pair's H or E side — likewise;
    3. **a source from this pair's field family is CARRIED, not refused.** The
       driver injects it between the curl and constitutive calls, so it is real
       work inside the seam this kernel closes, and fusing over it naively would
       drop the deposit. This product does not fuse over it naively: it declares
       :data:`CARRIES_DEPOSIT_REPAIR`, so ``launch._install_fused_pair`` brackets
       the launch with ``deposit_repair.LeadingRepairPlan`` /
       ``TrailingRepairPlan`` and the deposit points are recomputed from the
       injected field afterwards. What is still refused here is what the repair
       cannot reconstruct exactly — an off-diagonal or nonlinear constitutive
       half, or a source that does not publish the index it writes — and
       ``deposit_repair.seam_source_reasons`` returns those reasons per run;
    4. **the walls are carried inline.** ``zero_metal_B`` or ``zero_metal_D``
       also runs in the seam. It is transcribed into the kernel from
       :func:`zero_metal_axes`, so a walled run stays covered — but the grid must
       be able to answer the question, and a folded metallic axis (which
       ``_zero_metal`` skips for a reason worth 1.28e+00 relative L2) is refused
       by clause 1 already.

    ``sources`` MUST BE DECLARED. ``Fields`` does not hold the source list — the
    driver does — so this predicate cannot look it up, and a predicate that infers
    "no source in this seam" from its own ignorance is the over-covering refusal
    §10.4 names as the dangerous failure. ``None`` is therefore a REFUSAL, not an
    empty set; the caller passes ``driver._sources`` (possibly ``()``).
    """
    if pair not in FUSED_PAIRS:
        raise ValueError(f"pair must be one of {tuple(FUSED_PAIRS)}, got {pair!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = FUSED_PAIRS[pair]
    reasons: List[str] = []
    curl = pml_curl_coverage(fields, pml, spec["curl"])
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    constitutive = constitutive_coverage(fields, pml, spec["constitutive"])
    if not constitutive.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in constitutive.reasons)

    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, pair,
        undeclared=(
            "the source set was not declared: this predicate cannot read it off "
            "Fields, and inferring 'no magnetic source' from not knowing is how a "
            "predicate over-covers"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is {spec['source_family']}: the "
            f"driver injects it BETWEEN {spec['curl']} and "
            f"update_{spec['constitutive']}, "
            f"which is work inside the seam this kernel closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # Clause 4. The fold is already refused above; this asks whether the grid can
    # answer the wall question at all, since an unanswerable one compiles to
    # ZM=False and silently skips a plane the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_{pair} cannot be carried inline")

    return Coverage(not reasons, tuple(reasons))


def ade_update_p_coverage(fields: Any, state: Any, component: str) -> Coverage:
    """May the Triton ADE kernel advance this (state, component) pair's polarization?

    The sub-step is ``dispersion.PolarizationState.update`` (dispersion.py:679-691),
    element-wise in space with no stencil, no neighbour read and no boundary pass —
    everything a boundary does to P arrives through the drive field W, which already
    carries it.

    Two things here are silent wrong answers rather than crashes and are therefore
    checked by name:

    * ``SIGMA_IS_VOLUME`` is a compile-time choice, and choosing it wrongly reads a
      scalar as a pointer or a pointer as a scalar. :func:`sigma_is_volume` is the
      single place that decides it, so the clause and the constexpr cannot disagree;
    * the drive field. ``Fields.drive_field`` returns ``f_w_*`` under PML and the
      stored E without (``fields.py:1140-1163``) — "the single most likely silent
      wrong answer in dispersion", because the two agree exactly outside the
      absorber. The launcher must go through ``drive_field``; this predicate pins
      that ``f_w_*`` exists, so that read cannot silently fall back.
    """
    reasons: List[str] = []
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")

    if component not in ELECTRIC_COMPONENTS:
        reasons.append(f"component {component!r} is outside {ELECTRIC_COMPONENTS}")
        return Coverage(False, tuple(reasons))

    if not _call(state, "drives", component, default=False):
        reasons.append(f"this susceptibility does not drive {component}")

    kind = getattr(getattr(state, "susceptibility", None), "kind", None)
    if kind not in COVERED_SUSCEPTIBILITY_KINDS:
        reasons.append(f"kind {kind!r} is outside {COVERED_SUSCEPTIBILITY_KINDS}")

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

    # dt: PolarizationState.update raises when the two disagree (dispersion.py:681-687),
    # and the kernel bakes the coefficients that dt produced.
    dt = getattr(grid, "dt", None)
    if dt is None:
        reasons.append("grid carries no dt")
    elif getattr(getattr(state, "grid", None), "dt", dt) != dt:
        reasons.append("the state's coefficients were built for a different dt")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return Coverage(False, tuple(reasons))
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
        reasons.extend(_volume_reasons(f"{name}[{component}]", array, shape))

    # The drive field. Under PML this must be f_w_*, never the stored E.
    drive_name = "f_w_" + component
    drive = getattr(fields, drive_name, None)
    if drive is None:
        reasons.append(f"{drive_name} is not allocated (the drive field under PML)")
    else:
        reasons.extend(_volume_reasons(drive_name, drive, shape))

    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    if sigma is None:
        reasons.append(f"sigma[{component}] is missing")
    elif getattr(sigma, "shape", ()):
        reasons.extend(_volume_reasons(f"sigma[{component}]", sigma, shape))
    else:
        try:
            float(sigma)
        except Exception:  # noqa: BLE001 - neither a scalar nor a volume
            reasons.append(f"sigma[{component}]={sigma!r} is neither a scalar nor a volume")

    return Coverage(not reasons, tuple(reasons))


def sigma_is_volume(state: Any, component: str) -> bool:
    """The ``SIGMA_IS_VOLUME`` constexpr, decided the same way the predicate checked it.

    One function so the check and the compile-time choice cannot disagree; getting
    them out of step reads a scalar as a pointer, which is a wrong answer and not a
    crash. The gate's mutation m13 is exactly this.
    """
    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    return bool(getattr(sigma, "shape", ()))


def _layout_reasons(fields: Any, shape: Sequence[int],
                    names: Sequence[str]) -> List[str]:
    """Every named field volume must be float32, C-contiguous and exactly ``grid.shape``."""
    out: List[str] = []
    if len(shape) != 3:
        return [f"grid shape {tuple(shape)!r} is not three-dimensional"]
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if total >= 2 ** 31:
        # The kernels index with int32 offsets; 2**31 elements is where that stops
        # being an arithmetic identity and starts being a wrong answer.
        out.append(f"{total} cells exceeds the kernel's int32 index range")
    for name in names:
        array = getattr(fields, name, None)
        if array is None:
            continue  # Already reported by the allocation clause.
        out.extend(_volume_reasons(name, array, shape))
    return out


def _volume_reasons(label: str, array: Any, shape: Sequence[int]) -> List[str]:
    """Shape, dtype and contiguity of one volume, reported by name."""
    out: List[str] = []
    if tuple(getattr(array, "shape", ())) != tuple(shape):
        out.append(f"{label} shape {tuple(getattr(array, 'shape', ()))!r} != grid shape "
                   f"{tuple(shape)!r}")
    if str(getattr(array, "dtype", None)) != "float32":
        out.append(f"{label} dtype {getattr(array, 'dtype', None)} is not float32")
    flags = getattr(array, "flags", None)
    if not bool(getattr(flags, "c_contiguous", False)):
        out.append(f"{label} is not C-contiguous")
    return out


def _inverse_epsilon_reasons(fields: Any, shape: Sequence[int]) -> List[str]:
    """The three per-component inverse-epsilon volumes ``update_E`` multiplies by.

    ``Fields.inverse_epsilon_for`` (fields.py:1337) may hand back three distinct
    volumes or three aliases of one array (``set_isotropic_epsilon_volume``); both
    are covered and neither changes the kernel, which binds three pointers either
    way. What is NOT covered is a scalar, which would need a different constexpr
    arm — so a component that does not answer with a volume is refused by name.
    """
    out: List[str] = []
    reader = getattr(fields, "inverse_epsilon_for", None)
    if not callable(reader):
        return ["fields does not expose inverse_epsilon_for"]
    for component in ELECTRIC_COMPONENTS:
        try:
            volume = reader(component)
        except Exception as exc:  # noqa: BLE001 - an unanswerable component is refused
            out.append(f"inverse_epsilon_for({component!r}) raised {exc!r}")
            continue
        if volume is None:
            out.append(f"inverse_epsilon_for({component!r}) is None")
            continue
        if not getattr(volume, "shape", ()):
            out.append(f"inverse_epsilon_for({component!r}) is a scalar, not a volume")
            continue
        out.extend(_volume_reasons(f"inv_eps[{component}]", volume, shape))
    return out


def _coefficient_reasons(pml: Any, shape: Sequence[int], stems: Sequence[str],
                         suffixes: Sequence[str]) -> List[str]:
    """The named coefficient vectors on all three axes, on the named sub-lattices.

    The curl asks for ``kms``/``sinv`` on BOTH sub-lattices (it plans either
    sub-step); a constitutive side asks for ``kps``/``kms`` on ITS OWN sub-lattice
    only — integer for H, half-integer for E. That pairing is checked here and bound
    on the host; a swap is a half-cell error, not a crash, which is why the gate
    carries a mutation for it.
    """
    out: List[str] = []
    for axis_index, axis in enumerate("xyz"):
        for suffix in suffixes:
            for stem in stems:
                name = f"{stem}_{axis}{suffix}"
                array = getattr(pml, name, None)
                if array is None:
                    out.append(f"pml.{name} is missing")
                    continue
                if str(array.dtype) != "float32":
                    out.append(f"pml.{name} dtype {array.dtype} is not float32")
                if int(array.size) != int(shape[axis_index]):
                    out.append(
                        f"pml.{name} has {int(array.size)} entries, axis {axis} has "
                        f"{int(shape[axis_index])} cells")
    return out


def _call(obj: Any, name: str, *args: Any, default: Any = None) -> Any:
    """Call ``obj.name(*args)`` when it exists and is callable, else ``default``."""
    attr = getattr(obj, name, None)
    if attr is None:
        return default
    if callable(attr):
        try:
            return attr(*args)
        except Exception:  # noqa: BLE001 - an unanswerable question is not coverage
            return default
    return attr


def dispersive_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """Re-export the specialized dispersive ``update_E`` predicate lazily.

    The implementation lives beside its kernel because its pole-count and live-P
    requirements are specific to that product.  Keeping this forwarding function
    here gives composition one coverage namespace without copying any numerical
    rule or importing the optional Triton module during ordinary engine startup.
    """
    from .dispersive_update_e import (  # noqa: PLC0415
        dispersive_constitutive_coverage as specialized_coverage,
    )

    return specialized_coverage(fields, pml)
