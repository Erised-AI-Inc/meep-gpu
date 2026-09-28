"""Which configurations the CUDA in-seam passes may serve — with no CuPy in sight.

THE THREE PASSES, AND WHY THEY NEEDED A MODULE OF THEIR OWN. Between a curl and
the constitutive update that closes it the driver runs three host-side passes
(driver.py:3284-3286 on the B side, :3300-3302 on the D side)::

    step_B  ->  magnetic sources  ->  fill_symmetry_bc_B
                                  ->  zero_metal_B
                                  ->  fill_folded_far_ghosts_B  ->  update_H

On the Triton and Metal tracks those passes have device code; on the hand-CUDA
track, until this file, they had none. Counted 2026-08-21 over each track's
NON-TEST modules: ``zero_metal`` 19 Triton / 21 Metal, ``fill_symmetry`` 12 / 17,
``fill_folded_far_ghosts`` 11 / 17. The CUDA column reads 1 / 2 / 0 modules that
NAME the pass and 0 / 0 / 0 that carry device code for it -- the hits are five
copies of one comment in ``step_curl_kernels.py``, a sentence in its own module
comment, and one refusal string in ``coverage.py``, read rather than counted.
Every CUDA sub-step therefore returned to the host between the curl and the
constitutive, by construction.

WHAT THIS DOES NOT DO. It does not fuse anything, and nothing here brings a fused
CUDA product one launch closer on its own. It supplies the PARTS such a product
would have to absorb, each measured against ``stepping`` on its own.

NOTHING HERE IMPORTS CUPY, NUMPY OR THE KERNEL MODULE, for the reason the sibling
``coverage.py`` states in its own header: the predicate is the part whose failure
mode is a SILENT WRONG ANSWER rather than a crash, so it has to be importable,
exercisable and mutable on a laptop.

A SEPARATE FILE FROM ``coverage.py``, DELIBERATELY. That file's bytes are pinned
by digest in thirty-seven places in ``certification.json`` — every gate that ever
ran recorded the coverage source it imported — and a new predicate landing there
would move a digest no device run of this tranche backs. Nothing in this file
changes a byte of that one.

=============================================================================
THE TRANSCRIPTIONS, WITH THE LINE THEY CAME FROM
=============================================================================

* :func:`zero_metal_axes`      ``stepping._zero_metal`` (stepping.py:2206-2247)
* :func:`mirror_fill_phases`   ``stepping._fill_symmetry_ghost_cells`` (:1426-1450)
                               through ``stepping._mirror_phases`` (:2291)
* :func:`folded_far_rows`      ``stepping._fill_folded_far_ghosts`` (:1489-1541),
                               ``stepping._stored_past_owned`` (:1454-1469) and
                               ``stepping._far_reflect_rows`` (:1661-1692)

THE PARITY IS ONE SIGNED NUMBER PER PLANE, AND THAT IS MEASURED, NOT ASSERTED.
``fields.mirror_parity(c, axis, phase) == phase * (1 - 2 * IYEE_SHIFTS[c][axis])``
over all 12 components x 3 axes x 2 phases — 72 combinations, 0 mismatches,
re-measured on this laptop before this file was written and pinned by
``test_in_seam_passes.py``. The NEAR fill only ever touches components whose Yee
shift on the folded axis is 0, so their parity is ``+phase``; the FAR fill only
ever touches shift-1 components, so theirs is ``-phase``. One signed float per
launch is the whole parity input.

WHICH COMPONENTS EACH PASS TOUCHES, AND THE INVERSION BETWEEN THE FAMILIES.
``IYEE_SHIFTS`` (fields.py:214-219) makes the B/D families exact opposites::

    Bx (0,1,1)   By (1,0,1)   Bz (1,1,0)   -> shift 0 on its OWN axis only
    Dx (1,0,0)   Dy (0,1,0)   Dz (0,0,1)   -> shift 0 on the OTHER TWO axes

so on one folded axis ``a``:

===================  =========================  =========================
pass (shift tested)  B family                   D family
===================  =========================  =========================
zero_metal (== 0)    component ``a`` (one)      components != ``a`` (two)
near fill  (== 0)    component ``a`` (one)      components != ``a`` (two)
far fill   (== 1)    components != ``a`` (two)  component ``a`` (one)
===================  =========================  =========================

THE D SIDE'S FAR FILL IS NOT THE B SIDE'S MIRRORED. The near fill reaches two
components on D and one on B, and the far fill flips it — which is exactly the
inversion the fused track measured this week and the reason the D column above is
transcribed from ``IYEE_SHIFTS`` afresh rather than derived from the B one.

=============================================================================
WHAT IS REFUSED, BY NAME
=============================================================================

Enumerated positively. A feature that lands after this is written is refused
because it was never admitted, not admitted because nobody remembered to refuse
it.

* COMPLEX STORAGE. A Bloch run stores complex64 and these kernels index float32.
  This is the largest single gap and it is stated rather than discovered.
* THE CYLINDRICAL RADIAL AXIS. ``stepping._mirror_phases`` (:2291-2311) puts
  ``(-1)**grid.m`` in the same slot as a mirror plane's declared phase, so the
  near fill runs on ``r`` with a phase that is not +/-1 at odd ``m`` and whose
  per-component sign rule is ``r_to_minus_r_symmetry``'s, not
  ``mirror_parity``'s. Different arithmetic, refused by name.
* A MIRRORED AXIS WITH NO DECLARED PHASE. ``stepping._symmetry_phase`` (:2402)
  raises there rather than folding with the even default; so does this.
* A FOLDED AXIS WHOSE TWO TERMINATION ROUTES DISAGREE. ``is_metallic`` and
  ``stored_cells > owned_cells`` are two independent readings of one fact
  (``stepping._stored_past_owned``); a disagreement means one has drifted and
  neither may be trusted, and the consequence is a ghost plane written where MEEP
  steps the plane. Same cross-check, and the same reasoning, as
  ``coverage._fold_termination_problem``.
* ANYTHING BUT A C-CONTIGUOUS 3-D FLOAT32 ARRAY on the family's three components.
  The kernels do their own index arithmetic from ``(nx, ny, nz)``.

NOTHING DISPATCHES THESE KERNELS. No module in ``meep_gpu/`` imports
``cuda_kernels`` at all (``test_package_boundary.py`` pins the absence in both
directions), so a predicate here is a statement about what the kernels must not
be asked to serve, not about any run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

#: The two families the driver runs each pass on, and the components of each in
#: x, y, z order. ``stepping.B_COMPONENTS`` / ``D_COMPONENTS`` (stepping.py:183-184)
#: for the metallic wipe; ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` targets
#: (:213-222) for the two mirror fills. The two spellings name the same six
#: arrays, which ``test_in_seam_passes.py`` pins against ``stepping`` rather than
#: leaving as a coincidence.
FAMILIES: Tuple[str, ...] = ("B", "D")

FAMILY_COMPONENTS: Dict[str, Tuple[str, str, str]] = {
    "B": ("Bx", "By", "Bz"),
    "D": ("Dx", "Dy", "Dz"),
}

#: ``fields.IYEE_SHIFTS`` (fields.py:214-219) for the six arrays these passes
#: touch, restated rather than imported so this module stays engine-import-free.
#: Pinned equal to the engine's table by ``test_in_seam_passes.py``.
IYEE_SHIFTS: Dict[str, Tuple[int, int, int]] = {
    "Bx": (0, 1, 1), "By": (1, 0, 1), "Bz": (1, 1, 0),
    "Dx": (1, 0, 0), "Dy": (0, 1, 0), "Dz": (0, 0, 1),
}

#: ``stepping.MIRROR_SOURCE_INDEX`` (stepping.py:160) — MEEP's ``io = -2`` halved
#: origin, so the near ghost at stored cell 0 images stored cell 2.
MIRROR_SOURCE_INDEX = 2

#: The three passes, named as the driver names them.
PASSES: Tuple[str, ...] = ("zero_metal", "fill_symmetry", "fill_folded_far")


# ---------------------------------------------------------------------------
# The per-axis derivations
# ---------------------------------------------------------------------------

def shift_zero_components(family: str, axis: int) -> Tuple[int, ...]:
    """Indices of the family's components whose Yee shift on ``axis`` is 0.

    The set ``_zero_metal`` clears and the set ``_fill_symmetry_ghost_cells``
    near-fills — the same test (``shifts[axis] == 0``) in both, so it is one
    function here.
    """
    names = FAMILY_COMPONENTS[family]
    return tuple(c for c in range(3) if IYEE_SHIFTS[names[c]][axis] == 0)


def shift_one_components(family: str, axis: int) -> Tuple[int, ...]:
    """Indices of the family's components whose Yee shift on ``axis`` is 1.

    The set ``_fill_folded_far_ghosts`` fills (``shifts[axis] != 1: continue``,
    stepping.py:1583). READ OFF ``IYEE_SHIFTS`` and not off the complement of
    :func:`shift_zero_components`, because that is the same table read twice and
    the D side's inversion is exactly where an inherited complement would be
    plausible and wrong.
    """
    names = FAMILY_COMPONENTS[family]
    return tuple(c for c in range(3) if IYEE_SHIFTS[names[c]][axis] == 1)


def component_parity(family: str, component: int, axis: int, phase: int) -> int:
    """``fields.mirror_parity(c, axis, phase)`` for one of the six arrays.

    ``phase * (1 - 2 * iyee[c][axis])``, which is the closed form of MEEP's
    ``symmetry::phase_shift`` for a single mirror (fields.py:117-157): one
    direction flipped, none permuted, so the pseudovector loop toggles once and
    the handedness term never fires. Measured equal to ``fields.mirror_parity``
    on all 72 component/axis/phase combinations; ``test_in_seam_passes.py`` pins
    it against the engine's own function rather than against this comment.
    """
    shift = IYEE_SHIFTS[FAMILY_COMPONENTS[family][component]][axis]
    return int(phase) * (1 - 2 * shift)


def zero_metal_axes(grid: Any) -> Tuple[bool, bool, bool]:
    """The ``wall_x``/``wall_y``/``wall_z`` flags, decided as ``_zero_metal`` decides.

    ``stepping._zero_metal`` (stepping.py:2206-2247) returns early unless
    ``grid.has_metallic``, then clears stored cell 0 on every axis that is
    ``is_metallic(axis) and not is_mirrored(axis)``.

    A FOLDED METALLIC AXIS IS EXCLUDED AND THAT IS THE WHOLE POINT OF THE
    ``not is_mirrored`` TERM. ``grid_volume::halve`` moves such an axis's origin
    to ``io = -2``, so its stored cell 0 sits a full cell BELOW the mirror plane
    and holds the parity-weighted ghost the near fill writes there — not a wall.
    Clearing it destroys the fold: measured in the engine's own docstring at
    1.28e+00 complex relative L2 against CPU MEEP with the ghost zeroed, against
    2.0e-07 with the skip in place.

    One function, so the launcher and the predicate cannot disagree about which
    axes carry a wall.
    """
    if not bool(getattr(grid, "has_metallic", False)):
        return (False, False, False)
    return tuple(  # type: ignore[return-value]
        bool(grid.is_metallic(axis)) and not bool(grid.is_mirrored(axis))
        for axis in range(3))


def mirror_fill_phases(grid: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Each axis's declared mirror phase, None where the near fill does not run.

    ``stepping._fill_symmetry_ghost_cells`` (:1440-1450) skips an axis whose
    ``_mirror_phases`` entry is None, i.e. an unfolded one, and the whole pass
    returns early unless ``grid.has_symmetry()``.

    THE CYLINDRICAL SLOT IS NOT READ HERE. ``_mirror_phases`` puts ``(-1)**m``
    in the same tuple position for the radial axis; :func:`refusals` rejects such
    a grid before this is called, so a float never reaches a caller expecting
    +/-1.
    """
    if not bool(grid.has_symmetry()):
        return (None, None, None)
    return tuple(  # type: ignore[return-value]
        grid.mirror_phase(axis) if bool(grid.is_mirrored(axis)) else None
        for axis in range(3))


def stored_past_owned(grid: Any) -> Tuple[Optional[bool], ...]:
    """``stepping._stored_past_owned`` per axis — None where unanswerable.

    Transcribed from stepping.py:1503-1518: ``is_mirrored(axis) and
    stored_cells(axis) > owned_cells(axis)``, true on a folded PERIODIC axis at
    either full-count parity and nowhere else.

    ONE DELIBERATE DIVERGENCE, the same one ``coverage._stored_past_owned_from``
    makes: a grid that cannot answer comes back None rather than False. The
    engine's False there is a convenience for the stub grids in
    ``test_stepping.py``; here it would classify a folded PERIODIC axis as having
    no far ghost and silently drop the pass on the one axis that needs it.
    """
    stored = getattr(grid, "stored_cells", None)
    owned = getattr(grid, "owned_cells", None)
    if not (callable(stored) and callable(owned)):
        return (None, None, None)
    out = []
    for axis in range(3):
        try:
            out.append(bool(grid.is_mirrored(axis))
                       and int(stored(axis)) > int(owned(axis)))
        except Exception:  # noqa: BLE001 - an unanswerable axis is not a False one
            out.append(None)
    return tuple(out)


def folded_far_rows(grid: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """The stored row each folded-periodic axis's far ghost images, None elsewhere.

    ``stepping._far_reflect_rows`` (:1661-1692): ``n_full - stored_cells + 2`` on
    an axis that is mirrored and NOT metallic, None on every other.

    THE ROW IS NOT ALWAYS ``n - 2``, and that is the whole reason this is derived
    rather than baked. ``big_corner`` IS the second mirror at an even full count
    (image at slot ``n - 2``) and one half-cell above it at an odd one (slot
    ``n - 3``). Writing the fixed ``n - 2`` reflects about the window top instead
    of about the mirror — a whole cell wrong on every odd-count run, which is
    what ``reflect_row_n_minus_two`` plants on the gate.

    ONLY THE AXES ``_fill_folded_far_ghosts`` ACTUALLY VISITS get a row: that
    pass iterates ``_stored_past_owned`` axes (:1518), not every mirrored one, so
    a folded METALLIC axis is None here twice over — once by
    ``_far_reflect_rows``' own metallic clause and once by never being visited.
    """
    past = stored_past_owned(grid)
    rows: list = []
    for axis in range(3):
        if not past[axis]:
            rows.append(None)
            continue
        rows.append(int(grid.shape_full[axis]) - int(grid.stored_cells(axis)) + 2)
    return tuple(rows)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def _backend_name(grid: Any) -> Optional[str]:
    try:
        xp = getattr(grid, "xp", None)
        return None if xp is None else xp.__name__
    except Exception:  # noqa: BLE001
        return None


def _array_reasons(fields: Any, grid: Any, family: str) -> list:
    """Every reason the family's three arrays are not what the kernel indexes."""
    reasons: list = []
    try:
        shape = tuple(int(n) for n in grid.shape)
    except Exception as exc:  # noqa: BLE001
        return [f"the grid could not state its shape: {type(exc).__name__}: {exc}"]
    if len(shape) != 3:
        reasons.append(f"grid shape {shape} is not 3-D; the kernels decompose a "
                       f"linear index into (i, j, k) from (nx, ny, nz)")
    for name in FAMILY_COMPONENTS[family]:
        array = getattr(fields, name, None)
        if array is None:
            reasons.append(f"{name} is not allocated; an optional store is served "
                           f"on demand by the array path and the kernel has no "
                           f"pointer to hand it")
            continue
        dtype = getattr(getattr(array, "dtype", None), "name", None)
        if dtype != "float32":
            reasons.append(f"{name} is {dtype}; these kernels index float32 "
                           f"(a Bloch run stores complex64 and needs its own pair)")
        if tuple(getattr(array, "shape", ())) != shape:
            reasons.append(f"{name} has shape {tuple(getattr(array, 'shape', ()))}, "
                           f"not the grid's {shape}")
        flags = getattr(array, "flags", None)
        if flags is not None and not bool(getattr(flags, "c_contiguous", False)):
            reasons.append(f"{name} is not C-contiguous; the kernel's stride "
                           f"arithmetic assumes the (nx, ny, nz) row-major layout")
    return reasons


def _shared_reasons(fields: Any, grid: Any, family: str) -> list:
    """The clauses every one of the three passes carries."""
    reasons: list = []
    if family not in FAMILY_COMPONENTS:
        return [f"family {family!r} is not one of {FAMILIES}"]
    name = _backend_name(grid)
    if name != "cupy":
        reasons.append(f"backend is {name!r}, not cupy; these are CUDA kernels")
    try:
        if any(bool(grid.is_axis(axis)) for axis in range(3)):
            reasons.append(
                "cylindrical radial axis: stepping._mirror_phases puts (-1)**m in "
                "the mirror-phase slot for r, whose per-component sign rule is "
                "r_to_minus_r_symmetry's rather than mirror_parity's")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"the grid could not say whether an axis is the cylindrical "
                       f"radial one: {type(exc).__name__}: {exc}")
    reasons.extend(_array_reasons(fields, grid, family))
    return reasons


def covers_zero_metal(fields: Any, grid: Any, family: str) -> Tuple[bool, str]:
    """May ``zero_metal_B``/``_D`` serve this run? ``(covered, reason)``.

    The pass itself has no arithmetic — it stores ``+0.0f`` into one plane — so
    the clauses are all about the LAYOUT and about WHICH AXES carry a wall. The
    one substantive question is the folded-metallic exclusion, and
    :func:`zero_metal_axes` is the single place it is decided.
    """
    reasons = _shared_reasons(fields, grid, family)
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"the grid could not state its metallic/mirrored axes: "
                       f"{type(exc).__name__}: {exc}")
    if reasons:
        return False, "; ".join(reasons)
    return True, "covered: real float32 storage, no cylindrical axis, walls readable"


def covers_fill_symmetry(fields: Any, grid: Any, family: str) -> Tuple[bool, str]:
    """May ``fill_symmetry_B``/``_D`` serve this run? ``(covered, reason)``."""
    reasons = _shared_reasons(fields, grid, family)
    try:
        phases = mirror_fill_phases(grid)
    except Exception as exc:  # noqa: BLE001
        return False, "; ".join(reasons + [
            f"the grid could not state its mirror phases: {type(exc).__name__}: {exc}"])
    for axis, phase in enumerate(phases):
        if phase is None:
            continue
        if int(phase) not in (1, -1):
            reasons.append(
                f"axis {axis} is folded with mirror phase {phase!r}; "
                f"stepping._symmetry_phase carries +1 or -1 and raises on None "
                f"rather than folding with the even default")
    if reasons:
        return False, "; ".join(reasons)
    return True, "covered: every folded axis carries a declared +/-1 mirror phase"


def covers_fill_folded_far(fields: Any, grid: Any, family: str) -> Tuple[bool, str]:
    """May ``fill_folded_far_B``/``_D`` serve this run? ``(covered, reason)``.

    Carries the near fill's clauses plus the termination cross-check: a folded
    axis whose ``is_metallic`` reading and whose ``stored_cells > owned_cells``
    reading disagree is refused outright, because the two are independent routes
    to one fact and the consequence of trusting the wrong one is a ghost plane
    written where MEEP steps the plane.
    """
    covered, reason = covers_fill_symmetry(fields, grid, family)
    reasons = [] if covered else [reason]
    try:
        past = stored_past_owned(grid)
        rows = folded_far_rows(grid)
        stored = tuple(int(grid.stored_cells(a)) for a in range(3))
        mirrored = tuple(bool(grid.is_mirrored(a)) for a in range(3))
        metallic = tuple(bool(grid.is_metallic(a)) for a in range(3))
    except Exception as exc:  # noqa: BLE001
        return False, "; ".join(reasons + [
            f"the grid could not state its fold extents: {type(exc).__name__}: {exc}"])
    for axis in range(3):
        if not mirrored[axis]:
            continue
        if past[axis] is None:
            reasons.append(
                f"axis {axis} is folded and this grid cannot say whether it stores "
                f"the slot past MEEP's owned window; the far ghost cannot be "
                f"decided from is_metallic alone")
            continue
        if bool(past[axis]) == metallic[axis]:
            reasons.append(
                f"axis {axis} is folded with stored_cells > owned_cells "
                f"{bool(past[axis])} and is_metallic {metallic[axis]}; the two "
                f"routes to the fold's termination disagree and neither can be "
                f"trusted")
            continue
        row = rows[axis]
        if row is None:
            continue
        if not (0 <= row <= stored[axis] - 2):
            reasons.append(
                f"axis {axis}'s far ghost would image stored row {row} of "
                f"{stored[axis]}; the image row must be inside the array and "
                f"below the slot being written")
    if reasons:
        return False, "; ".join(reasons)
    return True, ("covered: every folded axis's termination agrees on both routes "
                  "and its image row is in bounds")


COVERAGE_BY_PASS = {
    "zero_metal": covers_zero_metal,
    "fill_symmetry": covers_fill_symmetry,
    "fill_folded_far": covers_fill_folded_far,
}


def covers(pass_name: str, fields: Any, grid: Any, family: str) -> Tuple[bool, str]:
    """One pass's predicate by name, for callers that hold the pass as data."""
    if pass_name not in COVERAGE_BY_PASS:
        raise ValueError(f"pass must be one of {PASSES}, got {pass_name!r}")
    return COVERAGE_BY_PASS[pass_name](fields, grid, family)


def refusals(fields: Any, grid: Any) -> Dict[str, Dict[str, Any]]:
    """Every (pass, family) verdict for one run, for a plan log or a record."""
    out: Dict[str, Dict[str, Any]] = {}
    for pass_name in PASSES:
        for family in FAMILIES:
            covered, reason = covers(pass_name, fields, grid, family)
            out[f"{pass_name}:{family}"] = {"covered": covered, "reason": reason}
    return out


def plan(pass_name: str, grid: Any) -> Tuple[Dict[str, Any], ...]:
    """The launches one pass needs on this grid, IN THE ORDER THE ARRAY PATH APPLIES THEM.

    ONE LAUNCH PER FOLDED AXIS FOR THE TWO FILLS, IN X, Y, Z ORDER, and that
    order is load-bearing rather than cosmetic. ``_fill_symmetry_ghost_cells``
    walks ``for term ... for axis in range(3)``, so a component unowned on two
    planes has its corner written twice and ends up carrying the PRODUCT of both
    parities — the doubly mirrored value MEEP's chunk connection produces. Two
    axes in one launch cannot promise that order, so the plan is a list and the
    launcher walks it.

    (Whether the order changes the bytes is a separate question from whether it
    is transcribed. The sibling Triton gate measured the reversed order at 14/14
    identical, which is what one expects from multiplies by exactly +/-1 — but
    matching the array path is what makes that a measurement about a
    configuration rather than a property leaned on.)

    ZERO_METAL IS ONE LAUNCH FOR ALL THREE AXES, and that is not the same choice
    made lazily. Its write is the constant ``+0.0f`` and it reads nothing, so two
    axes' planes commute exactly — a cell on both is written zero twice. There is
    no corner to order.
    """
    if pass_name == "zero_metal":
        walls = zero_metal_axes(grid)
        if not any(walls):
            return ()
        return ({"axes": tuple(int(bool(w)) for w in walls)},)
    phases = mirror_fill_phases(grid)
    if pass_name == "fill_symmetry":
        return tuple({"axis": axis, "phase": int(phases[axis])}
                     for axis in range(3) if phases[axis] is not None)
    if pass_name == "fill_folded_far":
        rows = folded_far_rows(grid)
        return tuple({"axis": axis, "phase": int(phases[axis]),
                      "reflect_row": int(rows[axis])}
                     for axis in range(3)
                     if rows[axis] is not None and phases[axis] is not None)
    raise ValueError(f"pass must be one of {PASSES}, got {pass_name!r}")


def plane_extent(shape: Sequence[int], axis: int) -> int:
    """How many cells one face of ``axis`` holds — the launch's element count."""
    nx, ny, nz = (int(n) for n in shape)
    return (ny * nz, nx * nz, nx * ny)[axis]
