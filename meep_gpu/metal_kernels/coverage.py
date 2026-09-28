"""Which configurations the Metal kernels may step.

The predicates are written POSITIVELY, for the same reason the Triton ones are:
almost everything these kernels do not carry produces a *silent wrong answer*
rather than a crash. A refusal reports EVERY disqualifying clause, not the first,
and coverage is A SET PER SUB-STEP, not per run.

NOTHING NUMERICAL IS RE-DERIVED HERE. Every leaf helper and every constant table
is IMPORTED from :mod:`meep_gpu.triton_kernels.coverage` — one definition, not a
copy: :func:`_boundary_kinds` (the one place the predicates consult the array
path, deliberately, so the fold/declaration precedence is not transcribed twice),
:func:`_layout_reasons`, :func:`_volume_reasons`, :func:`_inverse_epsilon_reasons`,
:func:`_coefficient_reasons`, :func:`_susceptibility_reasons`,
:func:`zero_metal_axes`, and the tables ``COVERED_BOUNDARIES``, ``CURL_TARGETS``,
``CURL_SUB_STEPS``, ``B_SOURCES``, ``D_SOURCES``, ``CONSTITUTIVE_SIDES``.

WHAT IS RE-IMPLEMENTED, AND ONLY THAT: :func:`_grid_reasons`. Its twelve-clause
list is backend-free in eleven of them; clause 1 is ``xp.__name__ != "cupy"``,
which is the wrong question on a host whose engine holds NumPy and whose device
memory is a torch MPS tensor. So the clause list is written out again with clause
1 replaced by :func:`_metal_backend_reasons`, and clauses 2-12 transcribed
verbatim.

THE DUPLICATION IS A STATED DEBT WITH A DISCHARGE PLAN. It exists because
``triton_kernels/`` is read-only this round, and the honest interim is
duplicate-plus-pin rather than a string filter over the Triton list — a filter is
an inference from a literal, and it would SILENTLY MISS a clause later added on
the Triton side, which is the exact over-covering failure this file exists to
prevent. ``test_metal_kernels.test_grid_reasons_match_the_triton_clause_list``
runs BOTH functions over a shared configuration matrix and asserts the reason sets
are equal once each side's own backend clause is removed, so drift fails a test in
either direction. NAMED END STATE, to be done when the fence lifts: hoist the
shared clause body into a backend-neutral ``meep_gpu/kernel_coverage.py`` that both
packages import, and DELETE the duplicate outright — no alias, no re-export, no
deprecated wrapper (the rename-outright rule).

THE RESIDENCY CLAUSE IS GENUINELY NEW (:func:`residency_reasons`). The CuPy path
is zero copy, so a Triton plan's arrays ARE the engine's arrays and no such
invariant exists. Here a plan owns persistent device mirrors, and a mirror is
valid only while nothing else writes the host array. Any array-path sub-step that
writes a mirrored volume between launches silently invalidates it — a smooth,
plausible, entirely wrong field, which is the failure class these predicates exist
to refuse. So the Metal backend clause is not merely "the arrays are MPS tensors":
the mirror set, the sub-steps that are live for this run, and the sub-steps the
caller brackets with an explicit sync must all be DECLARED, and an undeclared one
is a refusal rather than an assumption.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# One definition, imported rather than copied. Everything in this list is pure
# Python over duck-typed engine objects and carries no backend clause of its own.
from ..triton_kernels.coverage import (  # noqa: F401 - re-exported deliberately
    B_SOURCES,
    CONSTITUTIVE_SIDES,
    COVERED_BOUNDARIES,
    COVERED_SUSCEPTIBILITY_KINDS,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    D_SOURCES,
    ELECTRIC_COMPONENTS,
    MAGNETIC_COMPONENTS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
    _volume_reasons,
    zero_metal_axes,
)

#: The array module the residency layer mirrors FROM. Design (b): the engine keeps
#: its NumPy arrays and stays authoritative for boundary work, and the plan owns
#: device copies of the volumes it steps. Design (a) — giving the engine a torch
#: array module so ``Fields``/``Grid`` allocate MPS tensors directly — is the true
#: analogue of the CuPy path and is NOT this pass: it touches
#: ``backends.resolve_backend`` and every array-path operation.
MIRRORED_HOST_MODULE = "numpy"


def _metal_backend_reasons(grid: Any) -> List[str]:
    """Clause 1, the ONE clause that is not the Triton list's.

    Positive enumeration, three requirements, each named:

    1. torch imports at all — this package's optional dependency;
    2. this build has MPS and the device is available;
    3. the engine's array module is the one the residency layer mirrors from;
    4. the resolved float32 subnormal policy is one this executor can deliver.
       Clause 4 is a PROCESS property rather than a grid property and is here
       anyway, because it is a property of the BACKEND: Metal flushes denormals
       natively and exposes no lever, so a run resolved to ``keep`` — which is
       this arm64 host's default, MEEP's own behaviour being measured — is
       refused rather than silently run under a policy the device was never in.
       See :mod:`meep_gpu.metal_kernels.subnormal`.

    Written as its own function so the equivalence test can subtract exactly these
    strings from the Metal side rather than pattern-match them, and so a reader can
    see the whole backend question in one place.
    """
    reasons: List[str] = []
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - no torch, no Metal coverage
        return [f"torch is not importable ({exc!r}); the Metal kernels launch "
                f"through torch.mps.compile_shader"]
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is None:
        reasons.append("this torch build has no torch.backends.mps")
    else:
        if not bool(_call(backend, "is_built", default=False)):
            reasons.append("this torch build was not built with MPS")
        if not bool(_call(backend, "is_available", default=False)):
            reasons.append("no MPS device is available on this host")
    if not hasattr(getattr(torch, "mps", None), "compile_shader"):
        reasons.append("torch.mps.compile_shader is missing; hand-written Metal "
                       "sources cannot be compiled on this build")
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != MIRRORED_HOST_MODULE:
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not "
                       f"{MIRRORED_HOST_MODULE} (the residency layer mirrors host "
                       f"arrays; a device-to-device path is not built)")
    from .subnormal import mps_policy_reasons  # noqa: PLC0415 - kept off import

    reasons.extend(mps_policy_reasons())
    return reasons


def _grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every kernel in this package shares.

    Clause 1 is :func:`_metal_backend_reasons`. Clauses 2-12 are
    ``triton_kernels.coverage._grid_reasons``'s, transcribed, and the test named in
    the module docstring is what keeps them in step.
    """
    reasons: List[str] = []

    # 1. The Metal backend, and the host array module the mirrors copy from.
    reasons.extend(_metal_backend_reasons(grid))

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


# ---------------------------------------------------------------------------
# The residency invariant — no Triton counterpart
# ---------------------------------------------------------------------------

#: Which field volumes each sub-step of one complete driver step READS and WRITES.
#:
#: The FOUR entries that are NOT ``STEP_ORDER`` slots are the seam work the driver
#: does between them, and they are here because they are exactly what makes a
#: mirror go stale: ``zero_metal_B``/``zero_metal_D`` clear stored cell 0 on a
#: walled run (stepping.py:2206-2247, called at driver.py:3286 and :3301), and
#: ``fill_folded_far_ghosts_B``/``_D`` image the top plane of every folded PERIODIC
#: axis (stepping.py:1484-1532, called at driver.py:3287 and :3302). A residency
#: model that enumerated only the seven ``STEP_ORDER`` slots would hold a mirror
#: across a wall clear it never saw.
#:
#: THE TWO FAR-GHOST ENTRIES WERE MISSING AND THAT WAS BLOCKING, not cosmetic. One
#: complete magnetic half is FIVE passes, not three — ``step_B`` -> inject ->
#: ``fill_symmetry_bc_B`` -> ``zero_metal_B`` -> ``fill_folded_far_ghosts_B`` ->
#: ``update_H`` (driver.py:3282-3287; the electric half at :3293-3302) — and the
#: fifth WRITES B on every folded PERIODIC run. Without an entry a device mirror
#: held across a step would have been stale in exactly the plane the fold owns and
#: this predicate would not have said so, which is the stale-mirror class it exists
#: to refuse. The fold is the configuration that reaches it.
#:
#: ``fill_B``/``fill_D`` are the NEAR seam — source injection and
#: ``fill_symmetry_bc_*`` together, both of which write the primary volumes between
#: the curl and the constitutive sub-step. They are one entry because they write the
#: same three volumes and nothing sits between them.
SUB_STEP_VOLUMES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "step_B": {"writes": ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"),
               "reads": ("Ex", "Ey", "Ez")},
    "fill_B": {"writes": ("Bx", "By", "Bz"), "reads": ()},
    "zero_metal_B": {"writes": ("Bx", "By", "Bz"), "reads": ()},
    # Reads AND writes the same three volumes: the far ghost images a stored row of
    # the array it writes into (stepping.py:1529-1532), which is why it cannot be
    # modelled as a pure write and why fusing it across the wall clear above is a
    # different answer.
    "fill_folded_far_ghosts_B": {"writes": ("Bx", "By", "Bz"),
                                 "reads": ("Bx", "By", "Bz")},
    "update_H": {"writes": ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz"),
                 "reads": ("Bx", "By", "Bz")},
    "step_D": {"writes": ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"),
               "reads": ("Hx", "Hy", "Hz")},
    "fill_D": {"writes": ("Dx", "Dy", "Dz"), "reads": ()},
    "zero_metal_D": {"writes": ("Dx", "Dy", "Dz"), "reads": ()},
    "fill_folded_far_ghosts_D": {"writes": ("Dx", "Dy", "Dz"),
                                 "reads": ("Dx", "Dy", "Dz")},
    "update_E": {"writes": ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"),
                 "reads": ("Dx", "Dy", "Dz")},
    # update_P reads the DRIVE FIELD, which under PML is f_w_* and not the stored E
    # (fields.py:1140-1163). That is the one dispersive read that touches a volume
    # the constitutive mirror writes, and it is why a dispersive run cannot hold a
    # mirror across an array-path update_P without a declared sync.
    "update_P": {"writes": (), "reads": ("f_w_Ex", "f_w_Ey", "f_w_Ez")},
}

#: The order one complete driver step visits them in, seam work included.
RESIDENCY_ORDER: Tuple[str, ...] = (
    "step_B", "fill_B", "zero_metal_B", "fill_folded_far_ghosts_B", "update_H",
    "step_D", "fill_D", "zero_metal_D", "fill_folded_far_ghosts_D", "update_E",
    "update_P",
)


def folded_periodic_axes(grid: Any) -> Tuple[bool, bool, bool]:
    """Which axes run the driver's FIFTH pass — ``fill_folded_far_ghosts_*``.

    Asked of ``stepping._stored_past_owned`` (stepping.py:1503-1518) and of nothing
    else, exactly as :func:`zero_metal_axes` asks the grid's own metallic
    declaration: that function is what ``_fill_folded_far_ghosts`` itself selects
    axes with (stepping.py:1567), so the residency model and the array path cannot
    disagree about which passes run. Re-deriving "folded and not declared metallic"
    here would be a second implementation of the fold's single point of failure.

    ``_stored_past_owned`` is True exactly on a folded PERIODIC axis, at either
    full-count parity: there and only there does the stored array carry MEEP's
    not-owned allocation slot half a cell past ``big_corner``. An unreadable
    ``stepping`` answers all-False, which makes the two far-ghost slots NOT LIVE —
    and that is the one place this function is not fail-closed, so it is named:
    ``residency_reasons`` is a refusal engine, and a caller that cannot import the
    engine has no mirrors to hold either.
    """
    try:
        from ..stepping import _stored_past_owned  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no stepping, no folded passes to model
        return (False, False, False)
    out = []
    for axis in range(3):
        try:
            out.append(bool(_stored_past_owned(grid, axis)))
        except Exception:  # noqa: BLE001 - an unanswerable axis is not a folded one
            out.append(False)
    return (out[0], out[1], out[2])


def sub_step_volumes(sub_step: str) -> Tuple[str, ...]:
    """Every volume one sub-step touches, read or written — ONCE EACH.

    Deduplicated, in writes-then-reads order. Every sub-step but one has disjoint
    read and write sets, so this is a no-op for them; the far ghost passes IMAGE A
    STORED ROW OF THE ARRAY THEY WRITE (stepping.py:1529-1532), so their two sets
    are the same three volumes and a naive concatenation would report each name
    twice in a refusal message. The de-duplication is here rather than in
    :func:`residency_reasons` so every caller sees one list.
    """
    spec = SUB_STEP_VOLUMES.get(sub_step)
    if spec is None:
        raise ValueError(f"{sub_step!r} is not one of {RESIDENCY_ORDER}")
    out: List[str] = []
    for name in tuple(spec["writes"]) + tuple(spec["reads"]):
        if name not in out:
            out.append(name)
    return tuple(out)


def residency_reasons(mirrored: Optional[Sequence[str]],
                      planned: Optional[Sequence[str]],
                      live: Optional[Sequence[str]],
                      synced: Optional[Sequence[str]] = (),
                      null: Optional[Sequence[str]] = (),
                      planned_volumes: Optional[
                          Mapping[str, Sequence[str]]] = None) -> List[str]:
    """Why this mirror set is not safe to hold across one complete step.

    THE INVARIANT: a mirror is valid only while nothing else writes the host array.
    So for every sub-step that is LIVE for this run and touches a mirrored volume,
    one of three things must be true — the sub-step is IN THE PLAN (it runs on the
    device, against the mirror), the caller has DECLARED that it brackets that
    sub-step with an explicit sync out and back, or the sub-step is NULL (it
    performs no operation at all, so it writes nothing to go stale). Anything else
    is a mirror the array path silently invalidated.

    ``mirrored``, ``planned`` and ``live`` are all DECLARED and ``None`` is a
    REFUSAL for each, never an empty set. The same reasoning
    ``fused_pair_coverage`` refuses an undeclared ``sources`` applies with more
    force here: this predicate cannot read the driver's source list, the grid's
    walls or the pole register off a plan, and inferring "no seam work" from its
    own ignorance is how a predicate over-covers.

    ``synced`` defaults to the empty tuple rather than to ``None`` because an
    empty sync set is a real and common answer (nothing is bracketed), whereas an
    empty LIVE set would mean the step does nothing.

    ``planned_volumes`` is the selected plan's own binding declaration per slot.
    It matters whenever a family replaces a driver's sub-step with a narrower
    recurrence: the no-PML curl binds B and D directly and allocates none of the
    ``fu_*`` or stored E/H arrays in :data:`SUB_STEP_VOLUMES`.  With no declaration
    the historical table remains the fail-closed fallback; with a declaration,
    every planned slot must be present and only its declared mirrors are required.

    ``null`` IS THE THIRD CASE, AND IT WAS ADDED BECAUSE THE MODEL WITHOUT IT WAS
    MEASURABLY WRONG. ``planned`` means "runs on the device, against the mirror".
    The no-PML null family's plans run on NEITHER: ``stepping.update_H`` returns at
    stepping.py:944-945 and ``update_E`` at :983-984 without reading an array, so
    the plan binds no buffer and writes no host volume. Declared as ``planned`` —
    which is the only set tranche 1's composer had for them — the verdict refused
    the family's own configuration::

        update_H is planned but ('Hx','Hy','Hz','f_w_Hx','f_w_Hy','f_w_Hz',
                                 'Bx','By','Bz') carry no mirror

    SIX of those nine are ``None`` on such a run, not four — measured through
    :func:`sub_step_volumes` on the gate's own fixture (2026-08-15): ``Hx Hy Hz``
    because H is never stored without PML (fields.py:664-666), and
    ``f_w_Hx f_w_Hy f_w_Hz`` because ``f_w_*`` is allocated only by
    ``enable_pml_storage``. The E side is the same shape: ``Ex Ey Ez`` (served on
    demand as ``D * inv_eps``) plus ``f_w_Ex f_w_Ey f_w_Ez``. Only ``Bx/By/Bz`` and
    ``Dx/Dy/Dz`` exist, so the refusal demanded mirrors of arrays that do not exist
    for two thirds of the names it listed. A ``null`` member therefore requires NO
    mirror (it binds nothing) and STALES no mirror (it writes nothing) — the two
    halves of the invariant, both satisfied vacuously and for the same reason.

    A sub-step may not be declared BOTH planned and null: those are contradictory
    claims about the same call, and picking one silently is how a composer's own
    bookkeeping error becomes a coverage verdict.
    """
    reasons: List[str] = []
    if mirrored is None:
        reasons.append("the mirror set was not declared: a residency verdict "
                       "cannot be inferred from a plan's own bindings")
    if planned is None:
        reasons.append("the planned sub-step set was not declared")
    if live is None:
        reasons.append("the live sub-step set was not declared: which seam work "
                       "runs depends on the sources, the walls and the poles, and "
                       "none of those is readable from here")
    if reasons:
        return reasons

    mirror_set = set(mirrored or ())
    planned_set = set(planned or ())
    live_set = set(live or ())
    synced_set = set(synced or ())
    null_set = set(null or ())

    declared_volumes: Optional[Dict[str, Tuple[str, ...]]] = None
    if planned_volumes is not None:
        try:
            items = tuple(planned_volumes.items())
        except Exception as exc:  # noqa: BLE001 - malformed is a refusal
            return [f"planned_volumes is not a readable mapping ({exc!r})"]
        declared_volumes = {}
        for name, values in items:
            try:
                resolved = tuple(values)
            except Exception as exc:  # noqa: BLE001 - malformed is a refusal
                reasons.append(
                    f"planned_volumes[{name!r}] is not iterable ({exc!r})")
                continue
            if any(not isinstance(value, str) or not value for value in resolved):
                reasons.append(
                    f"planned_volumes[{name!r}] contains a non-name entry: "
                    f"{resolved!r}")
                continue
            declared_volumes[str(name)] = tuple(dict.fromkeys(resolved))
        for name in sorted(set(declared_volumes) - planned_set):
            reasons.append(
                f"planned_volumes declares {name!r}, but that sub-step is not planned")
        for name in sorted(planned_set - set(declared_volumes)):
            reasons.append(
                f"{name} is planned but has no plan-specific volume declaration")
        if reasons:
            return reasons

    for name in sorted(planned_set | live_set | synced_set | null_set):
        if name not in SUB_STEP_VOLUMES:
            reasons.append(f"{name!r} is not one of {RESIDENCY_ORDER}")
    if reasons:
        return reasons

    for name in sorted(planned_set & null_set):
        reasons.append(
            f"{name} is declared both planned and null: a sub-step either runs on "
            f"the device against the mirror or performs no operation, and a "
            f"composer that claims both has lost track of which plan filled the "
            f"slot")
    if reasons:
        return reasons

    for name in RESIDENCY_ORDER:
        if name not in live_set:
            continue
        touched = tuple(v for v in sub_step_volumes(name) if v in mirror_set)
        if not touched:
            continue
        if name in planned_set or name in synced_set or name in null_set:
            continue
        reasons.append(
            f"{name} runs on the array path and touches mirrored {touched!r}: a "
            f"mirror held across it is stale, and a stale mirror is a smooth, "
            f"plausible, wrong field rather than an error")

    # The other direction: a planned sub-step whose volumes are not all mirrored
    # would launch against a buffer nothing keeps in step with the host array. A
    # NULL sub-step is exempt by construction — it binds no buffer to launch
    # against — and is deliberately absent from this loop rather than filtered out
    # of it, so the exemption is visible where it is granted.
    for name in sorted(planned_set):
        required = (sub_step_volumes(name) if declared_volumes is None
                    else declared_volumes[name])
        missing = tuple(v for v in required if v not in mirror_set)
        if missing:
            reasons.append(f"{name} is planned but {missing!r} carry no mirror")

    for name in sorted(synced_set - live_set):
        reasons.append(f"{name} is declared synced but is not live for this run; "
                       f"a sync declaration for a sub-step that never runs is a "
                       f"claim about work that does not happen")

    for name in sorted(null_set - live_set):
        reasons.append(f"{name} is declared null but is not live for this run; a "
                       f"null declaration for a sub-step that never runs is a "
                       f"claim about work that does not happen")
    return reasons


def residency_coverage(mirrored: Optional[Sequence[str]],
                       planned: Optional[Sequence[str]],
                       live: Optional[Sequence[str]],
                       synced: Optional[Sequence[str]] = (),
                       null: Optional[Sequence[str]] = (),
                       planned_volumes: Optional[
                           Mapping[str, Sequence[str]]] = None) -> Coverage:
    """:func:`residency_reasons` as a verdict."""
    reasons = residency_reasons(
        mirrored, planned, live, synced, null, planned_volumes)
    return Coverage(not reasons, tuple(reasons))


def _residency_declaration_reasons(residency: Any) -> List[str]:
    """A sub-step predicate's own share of the residency question.

    The per-sub-step predicates cannot answer the whole invariant — that needs the
    LIVE set, which only the composer knows — but they can refuse a plan built
    with no residency at all, which is the case where two sub-steps would silently
    stop sharing a volume's mirror.
    """
    if residency is None:
        return ["the residency was not declared: two sub-steps that mirror the "
                "same volume separately would each hold a private copy, and the "
                "second launch would read the first one's stale bytes"]
    if not hasattr(residency, "mirror"):
        return [f"residency {type(residency).__name__} does not expose mirror(); "
                "a plan cannot register the volumes it steps"]
    return []


# ---------------------------------------------------------------------------
# The two sub-step predicates
# ---------------------------------------------------------------------------

def pml_curl_coverage(fields: Any, pml: Any, sub_step: Optional[str] = None,
                      residency: Any = None) -> Coverage:
    """Is this (fields, pml) pair one the Metal real-field PML curl may step?

    The clause list is ``triton_kernels.coverage.pml_curl_coverage``'s, with the
    shared block replaced by this module's :func:`_grid_reasons` and one clause
    added for the residency declaration. Dispersion and off-diagonal epsilon are
    ADMITTED for the same reason they are there: both are constitutive-only
    features, and the curl differences the STORED E and H arrays.
    """
    if sub_step is not None and sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))

    # 8. Conductivity changes only the curl whose primary targets carry it.
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


def constitutive_coverage(fields: Any, pml: Any, side: str,
                          residency: Any = None) -> Coverage:
    """May the Metal constitutive kernel step ``update_H`` (side='H') / ``update_E``?

    ``triton_kernels.coverage.constitutive_coverage``'s clause list, over this
    module's :func:`_grid_reasons`, plus the residency declaration. The E side
    refuses everything that changes what ``source`` is — a registered polarization
    (``D - sum P``), an off-diagonal chi1inv row (the row product reads neighbours),
    and ``stores_E`` false.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
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
