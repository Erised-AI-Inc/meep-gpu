"""Which CYLINDRICAL (Dcyl) configurations the hand-CUDA curl pair may step, at m = 0.

NOTHING HERE IMPORTS CUPY, NUMPY OR THE KERNEL MODULE, for the reason
``coverage.py``'s header gives at length: the predicate is the part of this track
whose failure mode is a *silent wrong answer* rather than a crash, so it has to be
importable, exercisable and mutable on a machine with no GPU. ``cylindrical_kernels``
holds ``cp.RawKernel`` objects and imports ``cupy`` at module scope; this does not.

WHY IT IS A SEPARATE MODULE FROM ``coverage.py`` RATHER THAN A WIDENING OF IT.
``covers_real_pml_curl`` refuses Dcyl BY NAME, twice — ``coverage.py:499-500``
("cylindrical (Dcyl): prefix-sum radial derivative and axis-row rules") and
:501-503 (``axis N is the cylindrical r = 0 axis``). Those two clauses are RIGHT for
the kernels they answer for and must stay: ``step_B_pml_real`` has no prefix
pointer, no axis tail and no ``4*Courant`` scalar, so admitting a Dcyl grid there is
a wrong answer on Bz, Dz, Bx and Dy. The Dcyl kernels are a SECOND PAIR with a
different signature, so they get a second predicate, and the shared refusal is left
exactly where it is. The Metal port states the same separation as "clause 6
INVERTED" (``metal_kernels/cylindrical_real.py``) and the Triton one as "they do not
call ``coverage._grid_reasons``".

WHAT IS SHARED AND WHAT IS RESTATED. The GRID-SHAPED helpers are IMPORTED from
``coverage`` — ``_grid_facts``, ``_backend``, ``_array_problem``,
``_coefficient_vector_problem`` — because they read a grid and answer a question
about layout, and a second spelling of "is this a float32 C-contiguous volume of
this shape" is a second place to get it wrong. What is NOT imported is any predicate
or any clause LIST: this module enumerates its own refusals positively, in the same
order and with the same numbering as ``covers_real_pml_curl``, so the two can be
diffed by eye. A test pins that every imported helper still exists, so a rename in
the shared file fails at the merge bar instead of silently dropping a clause.

WHAT THIS PAIR CARRIES, ENUMERATED POSITIVELY:

* ``grid.cylindrical`` TRUE, ``grid.m == 0`` EXACTLY, r the leading axis;
* boundary kinds ``('axis', 'periodic', <'metallic'|'periodic'>)`` — the r axis is
  the cylindrical axis, phi is the one-cell invariant axis, and z terminates either
  way. Anything else is refused;
* real float32 storage, an active split-field PML, stored E, no fold, no Bloch, no
  BFAST, no beta, no conductivity on this sub-step's targets, no chi2/chi3.

WHAT IT REFUSES BY NAME AND WHY EACH IS A WRONG ANSWER RATHER THAN A CRASH:

* ``m != 0``. |m| >= 1 forces complex64 storage, the ``i*m/r`` coupling
  (``stepping._cylindrical_imr_term``), the |m| = 1 axis increments FOLDED INTO the
  curl on both sides (``stepping._cylindrical_axis_increment_B`` / ``_D``) and the
  |m| >= 2 six-component near-axis zeroing (``_cylindrical_axis_rows``). A kernel
  that ignored those would return a smooth, plausible, wrong field. That is a second
  kernel, refused BY NAME rather than by the absence of an ``i*m/r`` term — the
  Triton module's rule, kept.
* ``force_complex_fields``. A complex-storage m = 0 run is perfectly constructible
  (it is an independent switch) and this kernel would read the wrong stride on it.
  Kept UN-INVERTED, exactly as the Metal family keeps it.
* a phi extent other than 1 cell. Dcyl is 2.5-D; the ``exp(i*m*phi)`` dependence is
  analytic and the kernel's phi ghost is the invariant-axis self-difference.
* a mirror fold. ``Grid`` already refuses a mirror on a Dcyl cell, and this clause is
  written anyway, because "another module already guards it" is exactly the reasoning
  that produces a silent admission when that other module moves.
"""

from __future__ import annotations

from typing import Any, Tuple

from .coverage import (
    BC_CODES,
    CURL_SUB_STEPS,
    CYL_AXIS,
    METALLIC,
    PERIODIC,
    _array_problem,
    _backend,
    _boundary_kinds_from,
    _coefficient_vector_problem,
    _complex_volume_problem,
    _grid_facts,
    complex_expansion_refusal,
)

#: The helpers this module borrows from the shared predicate file, named as data so
#: a test can assert every one of them still exists. A rename there would otherwise
#: turn into an ImportError at the first Dcyl run rather than at the merge bar.
SHARED_HELPERS: Tuple[str, ...] = (
    "_grid_facts", "_backend", "_array_problem", "_coefficient_vector_problem",
    "_boundary_kinds_from", "BC_CODES", "CURL_SUB_STEPS",
    # The COMPLEX pair's two extra borrowings. ``_complex_volume_problem`` is the
    # complex64 twin of ``_array_problem`` and ``complex_expansion_refusal`` is the
    # ONE arbiter of an expansion licence in this package -- a second spelling of
    # either would be a second place to get the layout or the arm wrong.
    "_complex_volume_problem", "complex_expansion_refusal")

#: The one azimuthal order this pair carries. See the module docstring.
COVERED_M: int = 0

#: ``grid.is_axis`` on the three axes, required EXACTLY. r must be the leading axis:
#: the prefix scan runs along axis 0 (``stepping.cylindrical_rderiv_prefix``, whose
#: docstring pins "axis order here is the engine's: ``f_p`` is (r, phi, z) with r the
#: leading axis"), and the kernel's ``i`` is that axis.
CYLINDRICAL_AXIS_FLAGS: Tuple[bool, bool, bool] = (True, False, False)

#: The r and phi ghost rules, required EXACTLY; z is free and is checked against
#: :data:`BC_CODES` like any Cartesian axis. r = 0 is the cylindrical axis and phi is
#: the one-cell invariant axis, whose PERIODIC wrap returns the SAME element and
#: therefore differences to an exact ``+0.0``.
REQUIRED_R_KIND: str = CYL_AXIS
REQUIRED_PHI_KIND: str = PERIODIC

#: Every volume the two kernels dereference: six targets, six auxiliaries, six
#: sources. The D side additionally reads the stored Hp for the on-axis Dz add, which
#: is ``fields.Hy`` under an active PML — the same array already in this list.
DEREFERENCED: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def cylindrical_boundary_codes(kinds: Tuple[str, ...]) -> Tuple[int, int, int]:
    """Map a Dcyl grid's resolved kinds onto the kernels' two integer codes.

    THE r AXIS COMPILES AS METALLIC, and that is a transcription rather than a
    convenience: ``stepping._shift_up``'s ``CYL_AXIS`` branch SHARES the ``METALLIC``
    one character for character (stepping.py:1828-1830) — a hard zero at the far r
    face, which is the PEC wall a Dcyl cell carries at ``r_max``.
    ``stepping._shift_down``'s ``CYL_AXIS`` branch does NOT (:1830-1846): it images
    stored row 0 with a direction sign and the ``(-1)^m`` phase. That ghost is
    UNOBSERVABLE at m = 0 — the only terms taking a shift-down along r are ``Dy``
    (partner Hz, second operand) and ``Dz`` (partner Hp, first operand), both with
    r-Yee shift 0, so ``_mask_non_owned_cells`` zeroes their curl at exactly the row
    the near ghost writes — and BOTH sibling tracks measured it rather than argued
    it: Triton's ``axis_ghost_sign`` mutation came back 16/16 identical (uncaught)
    and Metal's array-path probe moved 0 of 33,012 words with 32 ghost substitutions
    fired. The CUDA gate carries the same mutation, and it is armed as a NULL.

    A PERIODIC r axis is refused BY NAME: its far ghost would wrap the outer radius
    onto the axis row.
    """
    codes = []
    for axis, kind in enumerate(kinds):
        if axis == 0:
            if kind != CYL_AXIS:
                raise ValueError(
                    f"the r axis resolved to boundary {kind!r}; the cylindrical curl "
                    f"pair serves {CYL_AXIS!r} only (a PERIODIC r axis would wrap the "
                    f"far radius onto the axis row)")
            codes.append(BC_CODES[METALLIC])
            continue
        if kind not in BC_CODES:
            raise ValueError(
                f"axis {axis} resolved to boundary {kind!r}, which the cylindrical "
                f"curl pair does not serve (only {sorted(BC_CODES)}).")
        codes.append(BC_CODES[kind])
    return tuple(codes)


def covers_real_pml_cylindrical_curl(fields: Any, pml: Any, grid: Any,
                                     sub_step: str) -> tuple:
    """Whether ``cyl_step_B_pml_real`` / ``_D_`` may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``; returns ``(covered, reason)`` with
    ``reason`` naming the FIRST refusal, exactly as ``covers_real_pml_curl`` does, so
    a configuration that unexpectedly stays on the array path says why.

    THE CLAUSE ORDER MIRRORS ``covers_real_pml_curl`` so the two can be diffed by
    eye. The backend clause fires first for the same reason it does there — a grid
    that cannot answer must not pre-empt it.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {sorted(CURL_SUB_STEPS)}, got {sub_step!r}")

    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    # 2. REAL STORAGE, kept un-inverted: a complex-storage m = 0 run is
    #    constructible and this kernel would read the wrong stride on it.
    if getattr(fields, "force_complex_fields", False):
        return False, "complex64 storage: the recurrence is the same but the storage is not"
    # 3. An absorber that actually absorbs. Without one the plain path is the
    #    bit-identical one and this recurrence is the wrong sub-step entirely.
    if not (pml is not None and getattr(pml, "is_active", False)):
        return False, "no active PML layer"
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable

    # 6. INVERTED. This pair REQUIRES what every Cartesian predicate refuses.
    if not facts["cylindrical"]:
        return False, "not a cylindrical (Dcyl) grid: this pair steps Dcyl only"
    if facts["axis"] != CYLINDRICAL_AXIS_FLAGS:
        return False, (f"grid.is_axis {facts['axis']!r} is not "
                       f"{CYLINDRICAL_AXIS_FLAGS!r} (r must be the leading axis)")

    # 14. THE INTERNAL SPLIT against a future |m| >= 1 pair, refused BY NAME.
    m = getattr(grid, "m", None)
    if m is None:
        return False, "grid does not report m"
    try:
        m = int(m)
    except Exception as exc:  # noqa: BLE001 - an unreadable m is a refusal
        return False, f"grid.m could not be read as an integer: {type(exc).__name__}: {exc}"
    if m != COVERED_M:
        return False, (f"grid.m = {m}; this pair carries m = {COVERED_M} ONLY (|m| >= 1 "
                       f"forces complex storage, the i*m/r coupling and the per-|m| "
                       f"axis rules — a separate kernel)")

    # 5. NO FOLD. Grid already refuses a mirror on a Dcyl cell; written anyway.
    if facts["has_symmetry"] or any(facts["mirrored"]):
        return False, "a mirror plane is active: symmetry folding is not carried"

    # 4. THE GHOST RULES, per axis and spelled positively. r must be the axis, phi
    #    must be the one-cell invariant axis, z is free among the two kernel codes.
    kinds = _boundary_kinds_from(facts)
    if kinds[0] != REQUIRED_R_KIND:
        return False, (f"axis 0 resolves to boundary {kinds[0]!r}, not "
                       f"{REQUIRED_R_KIND!r}: the r axis must be the cylindrical axis")
    if kinds[1] != REQUIRED_PHI_KIND:
        return False, (f"axis 1 resolves to boundary {kinds[1]!r}, not "
                       f"{REQUIRED_PHI_KIND!r}: phi is the invariant axis")
    if kinds[2] not in BC_CODES:
        return False, f"axis 2 resolves to boundary {kinds[2]!r}, which has no kernel"

    if facts["has_bloch"]:
        return False, "nonzero Bloch k: the wrapped plane carries a phase real storage cannot hold"
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"

    # 8. THIS sub-step's targets only — ``_apply_curl`` reads the conductivity per
    #    TERM (stepping.py:508), so a D sigma disqualifies step_D and leaves step_B.
    try:
        conductive = [component for component in CURL_SUB_STEPS[sub_step]
                      if fields.condfac_for(component) is not None]
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"fields could not be asked for a conductivity: "
                       f"{type(exc).__name__}: {exc}")
    if conductive:
        return False, (f"{conductive[0]} carries a conductivity: routes to the "
                       f"three-history conductive-PML recurrence")

    if not getattr(fields, "stores_E", False):
        return False, "E is recomputed from D rather than stored"
    if getattr(fields, "_chi2_components", None) or getattr(fields, "_chi3_components", None):
        return False, "instantaneous chi2/chi3: a Pade factor on the constitutive product"

    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    if int(shape[1]) != 1:
        return False, (f"phi extent is {int(shape[1])} cells, not 1: Dcyl is 2.5-D and "
                       f"the exp(i*m*phi) dependence is analytic")

    # THE INT32 INDEX RANGE. Both kernels declare ``int idx`` and add ``ny*nz`` to it
    # for the radial neighbour; the B side adds it AGAIN to reach the prefix's next
    # row. Past 2**31 elements that stops being an identity and starts being a wrong
    # answer at a wrapped index — a silently corrupted volume, not a launch failure.
    # The bound is taken on the PREFIX's element count, which is the larger of the
    # two on the B side (nr + 1 rows).
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    prefix_cells = (int(shape[0]) + 1) * int(shape[1]) * int(shape[2])
    if max(cells, prefix_cells) >= 2 ** 31:
        return False, (f"{max(cells, prefix_cells)} cells exceeds the kernel's int32 "
                       f"index range")

    for name in DEREFERENCED:
        problem = _array_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem

    # The pair plans EITHER sub-step, so BOTH Yee sub-lattices must be present and
    # well formed: B reads the half-integer positions and D the integer ones, and
    # swapping them is a half-cell error in the absorber profile, not a crash.
    for axis, name in enumerate(("x", "y", "z")):
        for half in (True, False):
            suffix = "_h" if half else ""
            for label in ("kms", "sinv"):
                attribute = f"{label}_{name}{suffix}"
                problem = _coefficient_vector_problem(
                    attribute, getattr(pml, attribute, None), xp, axis, shape)
                if problem is not None:
                    return False, problem

    # ``accurate_fields_near_cylorigin`` is IRRELEVANT AT m = 0 — it selects how many
    # near-axis rows the |m| >= 2 stability hack holds at zero
    # (``stepping._cylindrical_axis_rows``) and at m = 0 there is no such hack. It is
    # required READABLE anyway, so that a later widening to |m| >= 2 cannot inherit
    # this predicate's silence about it. The Triton module makes the same demand for
    # the same reason.
    if getattr(grid, "accurate_fields_near_cylorigin", None) is None:
        return False, ("grid does not report accurate_fields_near_cylorigin; the flag "
                       "is irrelevant at m = 0 but must be readable before any "
                       "|m| >= 2 admission inherits this predicate's silence about it")

    return True, "covered"


# =============================================================================
# THE COMPLEX Dcyl CURL PAIR, |m| >= 1 -- a THIRD pair with a THIRD predicate
# =============================================================================
#
# WHY A THIRD PREDICATE RATHER THAN A WIDENING OF EITHER EXISTING ONE. The two it
# sits beside are both RIGHT and both stay exactly as they are:
#
# * :func:`covers_real_pml_cylindrical_curl` above refuses ``force_complex_fields``
#   by name -- correct, because ``cyl_step_B_pml_real`` reads a float32 stride,
#   has no i*m/r pointer, no axis-increment scalars and the WRONG axis rules (the
#   m = 0 pair);
# * ``coverage.covers_real_pml_complex_curl`` refuses Dcyl by name -- correct,
#   because ``step_B_pml_complex_bloch`` has no prefix pointer, no i*m/r row,
#   no axis increment and no near-axis zeroing. The sibling Triton tranche MEASURED
#   that refusal (1,281,955 differing uint32 words, 56 of 80 rows) rather than
#   arguing it, and this family's gate carries the same control.
#
# So each of the two shared clauses is now load-bearing in two directions at once
# and neither may be widened; ``cylindrical_complex_kernels`` is a third pair with a
# third signature, and this is its predicate.
#
# WHAT IT CARRIES, ENUMERATED POSITIVELY:
#
# * ``grid.cylindrical`` TRUE, ANY integer ``grid.m`` (|m| >= 1 since 2026-08-27,
#   m = 0 since 2026-09-04), r the leading axis, at least TWO radial rows;
# * complex64 storage with ``force_complex_fields`` SET (the clause the real pair
#   INVERTS), an active split-field PML, stored E;
# * boundary kinds ``('axis', 'periodic', <'metallic'|'periodic'>)``;
# * a bound expansion arm with a policy-matched licence.
#
# THE INTERNAL SPLIT AGAINST THE REAL PAIR IS ON STORAGE ALONE since 2026-09-04. Until
# then this predicate ALSO refused ``m == 0`` by name -- the kernel compiled no such
# arm, and m = 0 was the real pair's arithmetic. But a complex-storage m = 0 run is
# constructible (``force_complex_fields`` is an independent switch) and the corpus
# carries one (``examples:dipole_in_vacuum_cyl_off_axis.py``), so the two refusals
# left it with NO product: the real pair refused the storage, this one the order.
# ``cylindrical_complex_kernels`` now compiles an ``m_class == 0`` arm (the real
# pair's axis rules in cf arithmetic; its docstring point 6), and the two
# cylindrical products partition on ``force_complex_fields`` exactly as the real
# pair's clause 2 always said they should: "a complex-storage m = 0 run is perfectly
# constructible and this kernel would read the wrong stride on it". No triple can be
# admitted by both, because one requires the declaration the other refuses.
#
# WHAT IT REFUSES BY NAME, each because it would be a smooth wrong field:
#
# * a Bloch phase. Dcyl Bloch is z-only in MEEP, no corpus row carries one, and the
#   kernel passes every phase flag 0. Admitting one later is a predicate widening
#   plus a gate row, not a silent extension.
# * fewer than :data:`MINIMUM_RADIAL_ROWS` radial rows. The |m| = 1 axis increment
#   reads the FIRST OFF-AXIS row -- ``stepping.py:671``'s ``xp.take(Ez, 1, axis=0)``,
#   which NumPy raises ``IndexError`` on at nr = 1 (measured on a real
#   ``Grid(cell_size=(1.0, 0, 20), m=1, complex)``) -- and the kernel's matching load
#   ``sx + j*sy + k`` is true on every lane of such a grid, so it would read a full
#   plane PAST THE END of the source volume. The clause is UNCONDITIONAL rather than
#   |m| = 1 only: at |m| >= 2 an nr = 1 grid does not fault but every row is inside
#   ``zero_rows``, so the volume is held at zero and the configuration is degenerate
#   in both products -- and a clause that re-derives the m class is a clause that can
#   disagree with ``cylindrical_complex_kernels.m_class``.

#: The smallest radial extent this pair may step. See the clause above.
MINIMUM_RADIAL_ROWS: int = 2

#: Every volume the two complex kernels dereference. Same list as the real pair's --
#: the kernels read the same components -- but each is checked as a COMPLEX64
#: C-contiguous volume, which is the clause the real pair's ``_array_problem`` would
#: get backwards.
COMPLEX_DEREFERENCED: Tuple[str, ...] = DEREFERENCED


def covers_pml_cylindrical_complex_curl(fields: Any, pml: Any, grid: Any,
                                        sub_step: str, license: Any = None,
                                        subnormal_policy: Any = None) -> tuple:
    """Whether ``cyl_step_B_pml_complex`` / ``..._D_...`` may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``; returns ``(covered, reason)`` with
    ``reason`` naming the FIRST refusal. ``license`` is the verdict
    ``triton_kernels.complex_fields.expansion_license`` returns and
    ``subnormal_policy`` the policy this run installs; both are REQUIRED in practice,
    because the arm is compiled into the binary at this seam and there is no later
    rung to check it at. A WRONG ARM IS A WRONG ANSWER, not a crash.

    THE CLAUSE ORDER MIRRORS :func:`covers_real_pml_cylindrical_curl` so the two can
    be diffed by eye, with the storage clause INVERTED and three clauses added (the
    radial minimum, the Bloch refusal, the licence). The ``m`` clause is READ and
    admitted at every integer since 2026-09-04; the real pair's clause 14 keeps its
    ``m != 0`` refusal, so the two partition on storage.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {sorted(CURL_SUB_STEPS)}, got {sub_step!r}")

    # 0. THE ARM, FIRST. It is compiled into the binary and nothing downstream can
    #    check it; a licence cut under the other float32 subnormal policy is refused
    #    by the same arbiter the Cartesian complex family uses.
    refusal = complex_expansion_refusal(license, subnormal_policy)
    if refusal is not None:
        return False, refusal

    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    # 2. INVERTED against the m = 0 pair: complex64 storage is REQUIRED. At |m| >= 1
    #    the array path REFUSES real storage outright (stepping.py:730-736 raises and
    #    MEEP's change_m aborts on the same combination), so a real Dcyl run is m = 0
    #    and belongs to the other pair. This clause checks the DECLARATION; the
    #    storage dtype itself is checked per volume below.
    if not getattr(fields, "force_complex_fields", False):
        return False, ("real float32 storage: |m| >= 1 has no real-storage form "
                       "(stepping.py:730-736 raises, MEEP's change_m aborts), so a "
                       "real Dcyl run is m = 0 and belongs to the real pair")
    # 3. An absorber that actually absorbs: this is the split-field product only.
    if not (pml is not None and getattr(pml, "is_active", False)):
        return False, "no active PML layer"
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable

    # 6. INVERTED. This pair REQUIRES what every Cartesian predicate refuses.
    if not facts["cylindrical"]:
        return False, "not a cylindrical (Dcyl) grid: this pair steps Dcyl only"
    if facts["axis"] != CYLINDRICAL_AXIS_FLAGS:
        return False, (f"grid.is_axis {facts['axis']!r} is not "
                       f"{CYLINDRICAL_AXIS_FLAGS!r} (r must be the leading axis)")

    # 14. THE ORDER IS READ, NOT SPLIT ON. Every integer m is an arm of this family
    #     since 2026-09-04 (``cylindrical_complex_kernels.m_class``: M_ZERO / M_ONE /
    #     M_MANY); the split against the real pair is clause 2 above, on storage. An
    #     unreadable m is still a refusal: the m class is compiled into the launch.
    m = getattr(grid, "m", None)
    if m is None:
        return False, "grid does not report m"
    try:
        m = int(m)
    except Exception as exc:  # noqa: BLE001 - an unreadable m is a refusal
        return False, f"grid.m could not be read as an integer: {type(exc).__name__}: {exc}"

    # 5. NO FOLD. Grid already refuses a mirror on a Dcyl cell; written anyway.
    if facts["has_symmetry"] or any(facts["mirrored"]):
        return False, "a mirror plane is active: symmetry folding is not carried"

    # 4. THE GHOST RULES, per axis and spelled positively.
    kinds = _boundary_kinds_from(facts)
    if kinds[0] != REQUIRED_R_KIND:
        return False, (f"axis 0 resolves to boundary {kinds[0]!r}, not "
                       f"{REQUIRED_R_KIND!r}: the r axis must be the cylindrical axis")
    if kinds[1] != REQUIRED_PHI_KIND:
        return False, (f"axis 1 resolves to boundary {kinds[1]!r}, not "
                       f"{REQUIRED_PHI_KIND!r}: phi is the invariant axis")
    if kinds[2] not in BC_CODES:
        return False, f"axis 2 resolves to boundary {kinds[2]!r}, which has no kernel"

    # NO BLOCH PHASE AT ALL -- refused by name rather than by the kernel's silence.
    if facts["has_bloch"]:
        return False, ("a Bloch phase on a cylindrical grid: this pair compiles no "
                       "rotation and no Dcyl corpus row carries one (a z-only Dcyl "
                       "phase is a future widening, not a silent admission)")
    try:
        k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        return False, f"grid.k_point could not be read: {type(exc).__name__}: {exc}"
    if any(float(component) != 0.0 for component in k_point):
        return False, f"k_point {k_point!r} is not exactly zero"

    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"

    # 8. THIS sub-step's targets only -- ``_apply_curl`` reads the conductivity per
    #    TERM (stepping.py:508), so a D sigma disqualifies step_D and leaves step_B.
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        return False, ("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    for component in CURL_SUB_STEPS[sub_step]:
        try:
            conductive = reader(component) is not None
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
            return False, (f"condfac_for({component!r}) raised "
                           f"{type(exc).__name__}: {exc}")
        if conductive:
            return False, (f"{component} carries a conductivity: routes to the "
                           f"three-history conductive-PML recurrence")
    if getattr(fields, "has_magnetic_conductivity", False):
        return False, "a magnetic (B) conductivity is installed"

    if not getattr(fields, "stores_E", False):
        return False, "E is recomputed from D rather than stored"
    if getattr(fields, "_chi2_components", None) or getattr(fields, "_chi3_components", None):
        return False, "instantaneous chi2/chi3: a Pade factor on the constitutive product"
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        return False, ("a susceptibility is registered: complex cylindrical ADE is a "
                       "future tranche and no Dcyl corpus row combines the two")

    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    if int(shape[1]) != 1:
        return False, (f"phi extent is {int(shape[1])} cells, not 1: Dcyl is 2.5-D and "
                       f"the exp(i*m*phi) dependence is analytic")
    if int(shape[0]) < MINIMUM_RADIAL_ROWS:
        return False, (f"radial extent is {int(shape[0])} cell(s), fewer than "
                       f"{MINIMUM_RADIAL_ROWS}: the |m| = 1 axis increment reads the "
                       f"FIRST OFF-AXIS row (stepping.py:671) and the kernel's "
                       f"matching load would run past the end of the source volume")

    # ``accurate_fields_near_cylorigin`` is READ, never assumed: it selects the
    # |m| >= 2 near-axis row count (``stepping._cylindrical_axis_rows``) and a grid
    # that cannot answer would compile the wrong constant -- a plane of wrong values,
    # not a crash.
    if getattr(grid, "accurate_fields_near_cylorigin", None) is None:
        return False, ("grid does not report accurate_fields_near_cylorigin; it "
                       "selects the |m| >= 2 near-axis row count and may not be assumed")

    # THE INT32 INDEX RANGE, HALVED against the real pair's. Both kernels declare
    # ``int idx`` and address WORDS at ``2*idx``, and the B side additionally reads
    # the prefix at ``idx + sx`` on a volume one row taller. Past 2**31 words that
    # stops being an identity and starts being a wrong answer at a wrapped index.
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    prefix_cells = (int(shape[0]) + 1) * int(shape[1]) * int(shape[2])
    if 2 * max(cells, prefix_cells) >= 2 ** 31:
        return False, (f"{2 * max(cells, prefix_cells)} float32 words exceeds the "
                       f"kernel's int32 index range")

    for name in COMPLEX_DEREFERENCED:
        problem = _complex_volume_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem

    # The pair plans EITHER sub-step, so BOTH Yee sub-lattices must be present and
    # well formed: B reads the half-integer positions and D the integer ones, and
    # swapping them is a half-cell error in the absorber profile, not a crash. The
    # COEFFICIENTS STAY FLOAT32 under complex storage, so the real pair's vector
    # check is exactly the right one and is reused rather than re-spelled.
    for axis, name in enumerate(("x", "y", "z")):
        for half in (True, False):
            suffix = "_h" if half else ""
            for label in ("kms", "sinv"):
                attribute = f"{label}_{name}{suffix}"
                problem = _coefficient_vector_problem(
                    attribute, getattr(pml, attribute, None), xp, axis, shape)
                if problem is not None:
                    return False, problem

    return True, "covered"
