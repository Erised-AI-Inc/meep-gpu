"""Carry an in-seam source deposit across a fused curl/constitutive launch.

THE PROBLEM THIS SOLVES. The driver injects a source BETWEEN the curl and the
constitutive half -- magnetic at ``driver.py:3293-3294``, electric at ``:3318-3322`` -- so
a product that fuses those two halves consumes a pre-injection field. Every fusion
predicate therefore refuses any row carrying a source in its seam, which excludes 215 of
the 387 seam-instances: 58 magnetic and 157 electric.

That exclusion is a property of the PRODUCT'S SHAPE, not of the seam. Three facts, each
read from the code rather than assumed:

* the deposit is a SPARSE scatter -- ``D[ix, iy, iz] -= ...`` at ``sources.py:1305``;
* the fused kernel already stores the whole curl output and its PML auxiliary before the
  constitutive half runs (``triton_kernels/kernels.py:478-483``);
* that half has NO neighbour reads -- six loads at ``+ idx`` and six per-axis coefficient
  loads at ``+ i``/``+ j``/``+ k``, and the array path's is element-wise too.

So the fused launch may run over the whole grid against an uninjected field, and the
deposit points may be REPAIRED afterwards. Measured: a median 0.016% of a real grid and at
most 3.45% on any grid large enough for throughput to matter, so the repair is three to
four orders of magnitude cheaper than the round trip the fusion saves.

WHY A REPAIR AND NOT A SECOND INJECTION. The constitutive half is STATEFUL --
``_apply_constitutive_pml`` (``stepping.py:2130-2134``) reads ``fw`` before overwriting it
and then ACCUMULATES into the field:

    fw_previous = fw.copy() ; fw[...] = source
    field += kps * fw ; field -= kms * fw_previous

Correcting that afterwards by adding a delta is not float-exact. :func:`save` therefore
captures both arrays at the deposit points BEFORE the launch, and :func:`apply` recomputes
them from the injected field in the same operand order.

WHERE IT BELONGS IN THE STEP, and it needs no driver change. ``step_D`` and ``update_E``
are separate consults (``driver.py:3315``, ``:3331``) and ``FastPath.dispatch`` contracts
that True means the WHOLE sub-step ran. A fused arm owning both slots runs the kernel at
the first and spends the second on the repair -- with the driver's inject, symmetry fill
and wall clear untouched in between. ``zero_metal_D`` at ``driver.py:3327`` runs
unconditionally and is NOT behind a consult, so by the second consult the field is already
injected, filled and cleared: the repair needs no wall handling of its own.

EVIDENCE. ``parity/meep_gpu/probe_deposit_repair.py`` byte-compares this shape against the
driver's own order over 21 arrays: 7/7 identical on the host emulation and 6/6 with the
real ``fused_curl_constitutive_D`` on an RTX A6000, with every null control diverging.

A FOLDED SEAM NEEDS MORE THAN THE DEPOSIT POINTS, and that is the second half of this
module. ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*`` run AFTER the injection,
so a deposit landing on a row a fill READS FROM changes cells the fill then rebuilds;
:func:`repair_cells` is what extends the saved and restored set to those images. The
comment above it carries the measurement.

WHICH COMPONENTS ARE REPAIRED: THE ONE EACH SOURCE WRITES. A source deposits into ONE
primary array, and a diagonal constitutive target reads only its own component at the
cell, so at a deposit cell the other two targets' inputs are exactly what the fused
launch consumed and its results there stand. Per source, :func:`save` captures and
:func:`apply` restores only the target whose input that source's injection -- or the
driver's conductive replay of it -- writes; when that cannot be established they keep
every target of the seam and count the fallback. The comment above
:func:`_written_targets` carries the citations and the measurement.

WHAT THIS MODULE REFUSES. Only the DIAGONAL constitutive is repairable here. The
off-diagonal arm is a stencil over the partner volumes the curl wrote in place
(``stepping.py:1228-1251``) and the nonlinear arm is not a linear accumulation, so both
are refused by name rather than repaired approximately. On a folded grid it also
refuses, by name, any fold whose fill map it cannot read: the cylindrical r = 0 axis,
whose below-axis ghost is a different image rule entirely, and a folded axis too short
to hold the near fill's source row.

AND IT REFUSES THE ABSORBER IT CANNOT INVERT. Everything above is about the seam; this
is about the recurrence. ``field + kps*fresh - kms*fw_prev`` is the SPLIT-FIELD
constitutive update and nothing else, so the layer whose coefficients it reads has to be
the layer that actually ran. ``stepping._pml_is_active`` sends a layer absorbing on none
of its six faces down the PLAIN path instead -- ``update_H`` returns without touching H
(``stepping.py:944-945``) and ``update_E`` writes ``E = constitutive`` with no ``f_w`` at
all (``:990-993``) -- and a repair that inverted the split-field recurrence there would
write an answer the step never computed. Measured on a 14x14 grid with a zero-thickness
layer: 23 words across ``Ex``/``Ey``/``Ez`` and ``f_w_E*`` differ from the driver's own
order, while the same protocol at thickness 2 is byte-identical. :func:`repairable`
therefore takes the layer, refuses ``None`` and an inactive layer BY NAME, and refuses by
name any coefficient it cannot recognise as the per-axis broadcast layout
:func:`.pml.PML._reshape_for_broadcast` writes -- a missing attribute, a wrong rank, or a
non-unit extent on an axis that coefficient does not grade. An unrecognised layout is not
a diagnostic curiosity: :func:`apply` broadcasts each coefficient to the field's own
shape, so a coefficient of the wrong layout broadcasts SILENTLY and repairs every cell
with the wrong absorption.

The layer is OPTIONAL in the signature, and that is deliberate rather than lax. ``None``
IS an answer -- it names the plain path and is refused -- so a caller that was never
handed a layer has to be distinguishable from one that was handed nothing; the clauses
that need no layer still run for it. Every path that EXECUTES a repair supplies the
layer: :func:`apply` has always taken it, :class:`LeadingRepairPlan` holds it and hands
it to :func:`save`, so a configuration this cannot invert is refused before the fused
launch runs rather than after it has written.

===========================================================================
THE SECOND REPAIR, AND WHY THE FIRST ONE'S REFUSAL STANDS UNCHANGED BESIDE IT
===========================================================================

Everything above describes ONE repair: the inverse of the split-field recurrence. The
paragraph before this one refuses the plain branch by name because that repair does not
invert it. That refusal was and remains CORRECT -- and it is not the same statement as
"the plain branch cannot be repaired". :data:`PLAIN_PATH` is the second repair, and it
inverts the plain branch instead.

WHAT THE PLAIN BRANCH ACTUALLY DOES, read off ``stepping.update_E``
(``stepping.py:982-984``, ``:958-959``, ``:969``, ``:981-984``, ``:990-993``)::

    pml_active = _pml_is_active(pml)                                    # False here
    if not pml_active and not fields.stores_E: return                   # :954-955
    _require_field_storage(fields, E_STORAGE_ARRAYS)                     # :958-959
    for component, _, _axis in E_CONSTITUTIVE_TERMS:                     # :969
        source       = fields.displacement_minus_polarization(component) # :981
        constitutive = source * fields.inverse_epsilon_for(component)    # :982-984
        getattr(fields, component)[...] = constitutive                   # :993

THREE FACTS MAKE THAT EXACTLY INVERTIBLE, and each is a property of those lines rather
than an approximation of them:

1. **IT IS A PURE OVERWRITE, so there is NO STATE TO SAVE.** ``field[...] = constitutive``
   neither reads ``field`` nor accumulates into it. The split-field branch reads ``field``
   AND ``fw`` before writing and is stateful, which is the whole reason :func:`save`
   exists; this branch reads neither. What :func:`save` captures on this path is the CELL
   SET and nothing else -- the same closure, saved for the same reason (the two halves may
   not compute it separately), with an empty state beside it.
2. **THERE IS NO ``f_w``.** It is not allocated at all on this path, and
   ``Fields.drive_field`` returns the STORED E rather than ``f_w`` when the layer is
   inactive (``fields.py:1160-1162``). So ``update_P`` -- which runs after ``update_E``
   at ``driver.py:3333``, downstream of the repair's own slot -- reads exactly the array
   this repair rewrote. Nothing else consumes the constitutive product.
3. **EVERY OPERAND IS AT THE CELL'S OWN INDEX.** ``displacement_minus_polarization``
   reads ``D_c`` and each ``P_n[c]`` (``fields.py:1079-1105``) and ``inverse_epsilon_for``
   is a per-cell volume, so no neighbour is read and the product at a cell depends on
   that cell's two operands and nothing else.

So the plain repair is one line per component, GATHERED THEN MULTIPLIED::

    E_c[cells] = displacement_minus_polarization(c)[cells] * inverse_epsilon_for(c)[cells]

recomputed after the driver's inject / symmetry fill / wall clear. Per cell that is the
same IEEE multiplication of the same two operands, in the same order, as
``stepping.update_E``'s whole-volume ``source * inverse_epsilon_for(c)`` (``:982-984``)
-- one correctly rounded product with nothing to reassociate -- so forming it at the
cells alone is exact rather than close. Two spellings of ``cells`` carry it:
``_constitutive_at`` indexes by the 3-tuple, and keeps the whole-volume product only
when ``inv_eps`` does not share the volume's shape and therefore broadcasts; the linear
route gathers both operands through one C-order index into their flat views
(``_apply_plain``). Measured 2026-09-19 against the whole-volume spelling this replaced
(this file at sha256 ``c1d557040c6d``, git ``d8c2d9e``), as uint32 words: 0 differing
on both spellings -- the linear one on the two plain-path corpus lifts
(``absorber_1d``, ``no_pml_dispersive_2d``, 6 perturbed trials each, through a plan and
through the module-level calls), the 3-tuple one in
``test_deposit_repair_linear_route.py``'s ``plain_fortran_target`` case.

MEASURED BEFORE IT WAS WRITTEN, on real ``Grid``/``Fields`` objects and the shipped
``stepping`` functions: 13 of 13 unfolded configurations -- 0, 2 and 5 poles, with and
without a material conductivity, 1-D/3-D, periodic and metallic walls, real and complex
storage, point and sheet deposits -- came back bit-identical over 6 complete D->E steps
against the driver's own order, comparing every stored volume AND every polarization
buffer as uint32, with a null control that withheld the repair diverging on every one.
``parity/meep_gpu/probe_plain_deposit_repair.py`` is that measurement, kept.

AND THE MAGNETIC SEAM IS REFUSED HERE, BY NAME, RATHER THAN SERVED VACUOUSLY. On the
plain branch ``update_H`` returns without touching H (``stepping.py:944-945``); H is
served on demand from B and no array is written. A "repair" there would have nothing to
put back and would report success for a sub-step that stored nothing -- which is why
:data:`PLAIN_PATH` admits seam ``D`` only.

WHICH REPAIR RUNS IS DECIDED BY THE LAYER, NOT BY THE CALLER. :func:`repair_path_for`
asks ``stepping._pml_is_active`` -- the same function ``update_E`` branches on -- so the
path this module inverts is by construction the recurrence that ran. What the caller
declares in ``paths`` is which repairs IT carries, and a product handed a configuration
whose recurrence it does not carry is refused by name. THE DEFAULT IS
``(SPLIT_FIELD_PATH,)``, so every predicate, plan and probe written before this paragraph
keeps exactly the answer it had -- including both refusal texts above, which still fire,
unedited, for every caller that has not declared the second path.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import host_writes

__all__ = ["SEAMS", "MAGNETIC_FIELD_TYPE", "repairable", "fill_image_rules",
           "repair_cells", "save", "apply", "in_seam_sources", "seam_source_reasons",
           "LeadingRepairPlan", "TrailingRepairPlan", "DepositNotRepairable",
           "SPLIT_FIELD_PATH", "PLAIN_PATH", "REPAIR_PATHS", "PLAIN_PATH_SEAMS",
           "repair_path_for"]

#: ``sources.FIELD_TYPE_B``. Spelled here rather than imported so this module stays
#: importable by predicates on a host with no engine objects built yet.
MAGNETIC_FIELD_TYPE = "B"

#: THE ORIGINAL REPAIR: the inverse of ``_apply_constitutive_pml``'s split-field
#: recurrence ``(field + kps*fresh) - kms*fw_prev`` (``stepping.py:2130-2134``), which
#: is stateful and therefore needs ``field`` and ``fw`` captured before the launch.
#: Runs when ``stepping._pml_is_active`` is True, and only then.
SPLIT_FIELD_PATH = "split_field"

#: THE SECOND REPAIR: the inverse of ``update_E``'s plain branch
#: ``field[...] = (D - sum P) * inv_eps`` (``stepping.py:1019-1022``), which is a pure
#: overwrite and therefore needs no state at all -- only the cell set. Runs when
#: ``stepping._pml_is_active`` is False. See the module docstring's second half for the
#: arithmetic and for what was measured before this was written.
PLAIN_PATH = "plain"

#: Every recurrence this module can invert. A ``paths`` entry outside this tuple is a
#: refusal rather than a silent no-op: a caller naming a repair that does not exist has
#: declared something about its own wiring that is not true.
REPAIR_PATHS: Tuple[str, ...] = (SPLIT_FIELD_PATH, PLAIN_PATH)

#: The seams :data:`PLAIN_PATH` serves. ``D`` ONLY, and the absence of ``B`` is a
#: measurement rather than an omission: with an inactive layer ``update_H`` returns at
#: ``stepping.py:944-945`` without touching H, so the magnetic constitutive stores
#: nothing on this branch and there is no value for a repair to put back. Admitting it
#: would report a repaired sub-step that never wrote.
PLAIN_PATH_SEAMS: Tuple[str, ...] = ("D",)

#: Per seam: the constitutive targets, where each reads its source, and which Yee
#: sub-lattice its coefficients come from. Mirrors ``stepping.H_CONSTITUTIVE_TERMS`` and
#: the electric branch of ``stepping.update_E``.
SEAMS: Dict[str, Dict[str, Any]] = {
    "B": {"pair": "B", "targets": (("Hx", "Bx", "x"), ("Hy", "By", "y"), ("Hz", "Bz", "z")),
          "half_integer": False, "injected": ("Bx", "By", "Bz")},
    "D": {"pair": "D", "targets": (("Ex", "Dx", "x"), ("Ey", "Dy", "y"), ("Ez", "Dz", "z")),
          "half_integer": True, "injected": ("Dx", "Dy", "Dz")},
}


#: Axis name to array axis, for the broadcast layout ``PML._reshape_for_broadcast``
#: writes: ``(n,1,1)`` for ``'x'``, ``(1,n,1)`` for ``'y'``, ``(1,1,n)`` for ``'z'``.
_COEFFICIENT_AXIS = {"x": 0, "y": 1, "z": 2}


class _PmlUnsupplied:
    """The caller named no PML layer, which is NOT the same as naming ``None``.

    ``None`` is a PML answer -- it names the plain constitutive path, which this repair
    refuses -- so a caller that simply was not handed the layer has to be a distinct
    third state, or the clauses that need no layer could never run for it.
    """

    def __repr__(self) -> str:  # pragma: no cover - a diagnostic spelling
        return "<no PML layer supplied>"


_PML_UNSUPPLIED = _PmlUnsupplied()


class DepositNotRepairable(RuntimeError):
    """The seam carries a deposit this module will not reconstruct exactly."""


def _deposit_index(source: Any) -> Optional[Tuple[Any, Any, Any]]:
    """The (ix, iy, iz) the injection writes, on the grid's own array module, or None."""
    ix = getattr(source, "_point_ix", None)
    if ix is None:
        return None
    return ix, getattr(source, "_point_iy"), getattr(source, "_point_iz")


# ---------------------------------------------------------------------------
# The folded seam: the cells a post-injection fill images a deposit into
# ---------------------------------------------------------------------------
#
# A POINT REPAIR IS NOT ENOUGH ON A FOLDED GRID, and that is a measurement rather than
# an argument. ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*`` run AFTER the
# injection (``driver.py:3305-3306`` and ``:3309-3310`` magnetic, ``:3325-3326`` and
# ``:3329-3330`` electric; each fill sits behind its own consult, the wall clear between
# them behind none) and each writes one plane from another, so a deposit landing on a plane a
# fill READS FROM changes a cell the repair never visits: the fused launch computed
# that cell's constitutive result from the pre-injection field, and a repair that only
# writes the deposit index leaves it there.
#
# Measured on ``TestLoadDump.test_load_dump_fields_2d`` lifted from the corpus (5x5 um,
# resolution 50, one Y mirror), Ez deposit moved one cell off the plane to stored
# [125, 2, 0]: at the image [125, 0, 0] the point repair's ``Ez`` and ``f_w_Ez`` came
# back BYTE-EQUAL to the value a run with no source at all produced, on a zero
# background and on a seeded-noise one alike -- its error there is exactly the
# unrepaired leg's error. It is never a double application; the image is simply left
# uninjected. The same on a folded PERIODIC far ghost, on a two-fold corner (3 images),
# on a mixed near+far component (3 images), and on the magnetic seam, where a missed B
# image contaminates the ELECTRIC half in the same timestep because ``update_H``'s
# ghost feeds ``step_D``'s curl.
#
# WHAT THE SET MUST BE, per deposit point and computed with the DEPOSITED COMPONENT's
# own Yee shifts:
#
#   near  ``stepping._fill_symmetry_ghost_cells`` (stepping.py:1426-1452) writes stored
#         0 from stored ``MIRROR_SOURCE_INDEX`` on every axis carrying a mirror phase
#         where that component's Yee shift is 0;
#   far   ``stepping._fill_folded_far_ghosts`` (:1489-1533) writes the last stored slot
#         from ``stepping._far_reflect_rows(grid)[axis]`` on every folded PERIODIC axis
#         where that shift is 1.
#
# so the cells a point reaches are the point with any non-empty subset of its ACTIVE
# axes moved to their destinations -- at most seven extra cells, and empty unless an
# index equals a fill's source row, which is an integer comparison and not a search.
# This is the same closed form the folded fused products already implement in the
# FORWARD direction (``metal_kernels/folded_fused_magnetic_pair.carried_destinations``,
# :516-551, measured at ``results/metal_folded_far_carry_2026-08-20/ownership.json``);
# the repair needs its inverse, restricted to one point.
#
# NO PHASES ARE NEEDED. :func:`apply` reads the constitutive input out of the
# already-filled D/B array, which carries whatever parity the fill wrote, so the set is
# cells and nothing else.
#
# THE TWO CONSTANTS ARE READ, NEVER SPELLED. ``MIRROR_SOURCE_INDEX`` (stepping.py:160)
# and ``_far_reflect_rows`` (:1708, ``n_full - n_stored + 2``, which is ``n - 3`` at an
# odd full count and NOT ``n - 2``) both move with the engine, and a carry that
# hard-coded either would silently under-cover.
#
# OVER-COVERING IS SAFE, UNDER-COVERING IS THE SILENT WRONG ANSWER. :func:`apply` at a
# cell whose D did not change recomputes ``field_old + kp*fresh - km*fw_old`` from the
# same pre-launch state in the same operand order, which is bit-identical -- which is
# why a deposit ON a fill's destination costs nothing, and why the CELL SET is still
# over-covered wherever the closure cannot tell.
#
# THE COMPONENT SET NO LONGER IS (2026-09-20). Until then the repair also touched all
# three components at every deposit index, on the same argument. That was safe and it
# was two thirds of the bracket's work: a source writes ONE array, a diagonal
# constitutive target reads only its own component at the cell, so the other two
# targets were recomputed from inputs nothing had changed. :func:`_written_targets`
# now names the one target a source's injection feeds and the repair visits that one;
# the closure above was always computed per component, with the DEPOSITED component's
# own Yee shifts, so it is unchanged. See the comment above :func:`_written_targets`
# for what is cited and for the fail-closed fallback to all three.


def _yee_shifts(target: str) -> Tuple[int, int, int]:
    """The Yee sub-lattice of one primary array, from ``stepping``'s own curl tables.

    Which fills image a deposit is decided by the DEPOSITED component's shifts -- the
    near fill takes shift 0 and the far ghost shift 1 -- so this is read from the same
    table the fills iterate rather than restated.
    """
    from . import stepping  # noqa: PLC0415 - avoids a cycle at import time

    for term in stepping.B_CURL_TERMS + stepping.D_CURL_TERMS:
        if term.target == target:
            return tuple(int(shift) for shift in term.iyee)  # type: ignore[return-value]
    raise DepositNotRepairable(
        f"no curl term declares {target!r}, so its Yee sub-lattice -- which is what "
        f"decides which post-injection fills image a deposit in it -- is unknown")


def _fill_image_rules(grid: Any, shifts: Sequence[int], extents: Sequence[int]
                      ) -> Tuple[Tuple[int, int, int], ...]:
    """``(axis, source row, destination row)`` for every fill that images this component.

    MIRRORS THE TWO FILLS' OWN CONDITIONS rather than restating them: the same
    ``has_symmetry`` guard, the same ``_mirror_phases`` slot (which is where a
    cylindrical grid's ``(-1)^m`` also rides), the same shift test, and the same
    ``_stored_past_owned`` / ``_far_reflect_rows`` pair. A rule that drifted from the
    fill it inverts would under-cover exactly where the fill is live.
    """
    from . import stepping  # noqa: PLC0415 - avoids a cycle at import time

    if not grid.has_symmetry():
        return ()
    parities = stepping._mirror_phases(grid)
    reflect = stepping._far_reflect_rows(grid)
    rules: List[Tuple[int, int, int]] = []
    for axis in range(3):
        if parities[axis] is None:
            continue
        if shifts[axis] == 0:
            rules.append((axis, int(stepping.MIRROR_SOURCE_INDEX), 0))
        elif stepping._stored_past_owned(grid, axis) and reflect[axis] is not None:
            rules.append((axis, int(reflect[axis]), int(extents[axis]) - 1))
    return tuple(rules)


def fill_image_rules(fields: Any, target: str) -> Tuple[Tuple[int, int, int], ...]:
    """Per axis, ``(axis, the row a post-injection fill READS FROM, the row it WRITES)``.

    The rule :func:`repair_cells` inverts, exposed so a caller can PLACE a deposit on an
    imaged row rather than search for one -- which is what a gate needs to build a case
    that is about the images rather than about whichever cell a source happened to land
    on. Empty on an unfolded grid and on a component no fill touches.
    """
    return _fill_image_rules(fields.grid, _yee_shifts(target),
                             getattr(fields, target).shape)


def repair_cells(fields: Any, target: str, index: Tuple[Any, Any, Any]
                 ) -> Tuple[Any, Any, Any]:
    """``index`` together with every cell the post-injection fills image it into.

    Returns three flat index arrays on the grid's own array module, the deposit points
    first. Duplicates are possible -- a sheet that spans both a fill's source row and
    its destination produces one -- and are harmless: every entry recomputes the same
    value from the same saved state, so a repeated write is the write again.
    """
    grid = fields.grid
    xp = grid.xp
    columns = [xp.asarray(part).reshape(-1) for part in index]
    rules = _fill_image_rules(grid, _yee_shifts(target),
                              getattr(fields, target).shape)
    if not rules:
        return columns[0], columns[1], columns[2]
    masks = {axis: (columns[axis] == source) for axis, source, _destination in rules}
    extra: List[List[Any]] = [[], [], []]
    for size in range(1, len(rules) + 1):
        for combination in itertools.combinations(rules, size):
            selection = masks[combination[0][0]]
            for axis, _source, _destination in combination[1:]:
                selection = selection & masks[axis]
            moved = list(columns)
            for axis, _source, destination in combination:
                moved[axis] = xp.full_like(columns[axis], destination)
            for axis in range(3):
                extra[axis].append(moved[axis][selection])
    return tuple(xp.concatenate([columns[axis]] + extra[axis])  # type: ignore[return-value]
                 for axis in range(3))


def _folded_seam_reasons(grid: Any) -> Tuple[str, ...]:
    """Why THIS grid's post-injection fills cannot be inverted onto a deposit index.

    FAIL CLOSED, like the rest of this module: a fill whose source row cannot be named
    is a fill whose image cannot be repaired, and admitting on ignorance is the
    over-covering failure the fusion clauses exist to prevent. Empty on an unfolded
    grid, where neither fill runs at all.
    """
    from . import stepping  # noqa: PLC0415 - avoids a cycle at import time

    try:
        if not grid.has_symmetry():
            return ()
    except Exception as error:  # pragma: no cover - a grid that cannot answer
        return (f"grid.has_symmetry raised {error!r}, so whether this seam runs a "
                f"post-injection fill at all cannot be established",)
    reasons: List[str] = []
    for axis in range(3):
        try:
            cylindrical_axis = bool(grid.is_axis(axis))
        except Exception:  # pragma: no cover - a grid that cannot answer
            cylindrical_axis = False
        if cylindrical_axis:
            reasons.append(
                f"axis {axis} is the cylindrical r = 0 axis of a folded grid: its "
                f"below-axis ghost is the r_to_minus_r image the shift helpers apply "
                f"with per-component direction signs (stepping.py:171-177), and its "
                f"slot in _mirror_phases carries (-1)^m rather than a mirror phase "
                f"(stepping.py:2363), so the fill closure this repair inverts has "
                f"never been measured on it")
    try:
        parities = stepping._mirror_phases(grid)
        stepping._far_reflect_rows(grid)
        extents = [int(grid.stored_cells(axis)) for axis in range(3)]
        for axis in range(3):
            stepping._stored_past_owned(grid, axis)
    except Exception as error:  # pragma: no cover - a grid that cannot answer
        return tuple(reasons) + (
            f"the fold's fill map could not be read from this grid ({error!r}); "
            f"stepping._mirror_phases, ._far_reflect_rows, ._stored_past_owned and "
            f"grid.stored_cells are what name the rows a fill images",)
    for axis in range(3):
        if parities[axis] is None:
            continue
        if extents[axis] <= stepping.MIRROR_SOURCE_INDEX:
            reasons.append(
                f"folded axis {axis} stores {extents[axis]} cells, so the near fill's "
                f"source row {stepping.MIRROR_SOURCE_INDEX} does not exist "
                f"(stepping._mirror_source raises on it); a closure that compared "
                f"against a row the grid does not have would find no image and repair "
                f"nothing where the fill does the most")
    return tuple(reasons)


def _coefficient_layout_reasons(name: str, coefficient: Any, component: str,
                                axis: str, field_shape: Sequence[int]) -> Tuple[str, ...]:
    """Why ``apply`` cannot broadcast THIS coefficient the way the sub-step did.

    RECOGNISED IS DEFINED BY WHAT :func:`apply` DOES WITH IT, not by a remembered
    constant: it calls ``xp.broadcast_to(kps, field.shape)`` and then indexes the result
    at the repaired cells, so a coefficient is recognisable exactly when it is the rank-3
    per-axis layout ``PML._reshape_for_broadcast`` writes (``pml.py:659-669``) -- unit on
    the two axes it does not grade, and either the field's own extent or 1 on the axis it
    does. Anything else still BROADCASTS, silently, which is why an unrecognised layout
    has to be a refusal rather than a warning: a profile carrying x-grading in a y-shaped
    array would repair every cell with the wrong absorption and report success.
    """
    shape = getattr(coefficient, "shape", None)
    if shape is None:
        return (f"{name} is a {type(coefficient).__name__} carrying no shape, so the "
                f"per-axis absorption apply broadcasts to {component}'s "
                f"{tuple(int(n) for n in field_shape)} cannot be recognised",)
    try:
        shape = tuple(int(extent) for extent in shape)
    except Exception as error:  # pragma: no cover - a shape that is not a size tuple
        return (f"{name} carries a shape apply cannot read ({error!r}), so it cannot be "
                f"recognised as a coefficient layout for {component}",)
    index = _COEFFICIENT_AXIS[axis]
    if len(field_shape) != 3:
        return (f"{component} is {len(field_shape)}-dimensional ({tuple(field_shape)}), "
                f"not the rank-3 volume {name} is graded against, so no layout can be "
                f"recognised as the one the constitutive sub-step read",)
    graded = tuple(int(field_shape[index]) if position == index else 1
                   for position in range(3))
    if len(shape) != 3 or any(shape[position] != 1
                              for position in range(len(shape)) if position != index) \
            or shape[index] not in (1, int(field_shape[index])):
        return (f"{name} has shape {shape}, which is not a coefficient layout this "
                f"repair recognises for {component} on axis {axis!r}: apply broadcasts it "
                f"to {tuple(int(n) for n in field_shape)}, and only {graded} (the axis "
                f"graded) or {(1, 1, 1)} (no grading on this axis) index the same "
                f"absorption the constitutive sub-step read",)
    return ()


def _checked_targets(pair: str, targets: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
    """The seam targets the per-target clauses answer for: ``targets``, or ALL of them.

    ``None`` -- and an empty selection, which names nothing to narrow TO -- is every
    target of the seam, which is what every predicate asks and what each clause asked
    before the repair was restricted. :func:`save` and :func:`apply` pass the targets
    they are about to touch: a clause about a component this call neither reads nor
    writes cannot make what it does write wrong, and refusing on one is two thirds of
    the per-step host cost of these checks.
    """
    return tuple(targets) if targets else SEAMS[pair]["targets"]


def _absorber_reasons(fields: Any, pml: Any, pair: str, *,
                      targets: Optional[Sequence[Any]] = None) -> Tuple[str, ...]:
    """Why THIS layer's constitutive recurrence is not the one :func:`apply` inverts.

    Never raises: a layer that cannot answer a question is refused with the question it
    could not answer, which is the same fail-closed shape as :func:`_folded_seam_reasons`.

    ``targets`` narrows the PER-TARGET clauses (allocation, coefficient layout) to the
    targets a restricted repair touches; see :func:`_checked_targets`. The layer-wide
    clauses above them never narrow.
    """
    from . import stepping  # noqa: PLC0415 - avoids a cycle at import time

    if pml is None:
        return ("no PML layer: stepping._pml_is_active takes the plain path, where "
                "update_H returns without touching H (stepping.py:944-945) and update_E "
                "writes E = constitutive with no f_w accumulation at all (:1019-1022), so "
                "the kps/kms recurrence this repair inverts is not the one that ran",)
    try:
        active = bool(pml.is_active)
    except Exception as error:  # pragma: no cover - a layer that cannot answer
        return (f"the PML layer ({type(pml).__name__}) could not answer is_active "
                f"({error!r}), so whether the constitutive halves take the split-field "
                f"path or the plain one cannot be established",)
    if not active:
        return ("the PML layer absorbs on none of its six faces (is_active False), so "
                "stepping._pml_is_active sends both constitutive halves down the PLAIN "
                "path -- update_H does nothing and update_E writes E = constitutive -- "
                "while this repair recomputes field + kps*fresh - kms*fw_prev; measured "
                "on a 14x14 grid at thickness 0, that leaves 23 words of Ex/Ey/Ez and "
                "f_w_E* differing from the driver's own order",)

    spec = SEAMS[pair]
    reasons: List[str] = []
    for component, _source, axis in _checked_targets(pair, targets):
        field = getattr(fields, component, None)
        if field is None or getattr(field, "shape", None) is None:
            reasons.append(
                f"{component} is not allocated, so the shape its coefficients must "
                f"broadcast to -- and therefore whether they are the ones the "
                f"constitutive sub-step read -- is unknown")
            continue
        if axis not in _COEFFICIENT_AXIS:
            reasons.append(
                f"seam {pair!r} declares {component} on axis {axis!r}, which is not one "
                f"of {tuple(_COEFFICIENT_AXIS)}, so its coefficients cannot be named")
            continue
        suffix = f"_{axis}_h" if spec["half_integer"] else f"_{axis}"
        try:
            coefficients = stepping._constitutive_coefficients(
                pml, axis, half_integer=spec["half_integer"])
        except Exception as error:
            reasons.append(
                f"the constitutive coefficients for {component} (kps{suffix} / "
                f"kms{suffix}) could not be read from this layer ({error!r})")
            continue
        for name, coefficient in zip(("kps" + suffix, "kms" + suffix), coefficients):
            reasons.extend(_coefficient_layout_reasons(
                name, coefficient, component, axis, field.shape))
    return tuple(reasons)


def repair_path_for(pml: Any) -> str:
    """Which constitutive recurrence a sub-step reading ``pml`` actually RUNS.

    Asks ``stepping._pml_is_active`` -- the same call ``update_E`` and ``update_H``
    branch on (``stepping.py:942``, ``:953``) -- so the repair this module selects is by
    construction the inverse of the arithmetic that executed, rather than of the
    arithmetic a caller believed executed. A layer that cannot answer is the SPLIT-FIELD
    answer, because that is the path whose clauses then refuse it by name; treating an
    unreadable layer as "plain" would route it to a repair whose own clauses have no
    reason to look at a layer at all.
    """
    from . import stepping  # noqa: PLC0415 - avoids a cycle at import time

    try:
        return SPLIT_FIELD_PATH if stepping._pml_is_active(pml) else PLAIN_PATH
    except Exception:  # noqa: BLE001 - a layer that cannot answer is the refused one
        return SPLIT_FIELD_PATH


def _path_declaration_reasons(paths: Sequence[str]) -> Tuple[str, ...]:
    """Why this ``paths`` argument is not a set of repairs this module can perform."""
    if not paths:
        return ("the caller declared no repair path, so no recurrence has been claimed "
                f"as invertible; name at least one of {REPAIR_PATHS}",)
    unknown = tuple(str(name) for name in paths if name not in REPAIR_PATHS)
    if unknown:
        return (f"unknown repair path(s) {unknown}: this module inverts "
                f"{REPAIR_PATHS} and nothing else, so a caller naming another has "
                f"declared wiring that does not exist",)
    return ()


def _selected_path(pml: Any, paths: Sequence[str]) -> str:
    """WHICH path's clauses answer for this ``(layer, declaration)`` pair.

    The layer names the recurrence that RAN (:func:`repair_path_for`). When the caller
    carries that one, it is the one asked -- its clauses are the ones that describe what
    executed. When the caller does NOT carry it, the caller's own path is asked instead,
    so the refusal comes out of the repair that was declared and names the fork from that
    side; asking the path nobody carries would produce a reason about wiring that is not
    there. With no layer supplied there is nothing to ask, so the split-field path answers
    whenever it is declared -- the conservative half, and the one every caller written
    before the second repair passes.
    """
    fallback = SPLIT_FIELD_PATH if SPLIT_FIELD_PATH in paths else PLAIN_PATH
    if pml is _PML_UNSUPPLIED:
        return fallback
    ran = repair_path_for(pml)
    return ran if ran in paths else fallback


def _plain_path_reasons(fields: Any, pml: Any, pair: str, *,
                        targets: Optional[Sequence[Any]] = None) -> Tuple[str, ...]:
    """Why THIS configuration's plain constitutive is not the one :func:`apply` inverts.

    THE MIRROR OF :func:`_absorber_reasons`, and deliberately not a relaxation of it.
    That function refuses everything whose recurrence is NOT the split-field one; this
    one refuses everything whose recurrence is not the plain one. Neither text is
    reachable from the other's path, so both keep saying exactly what they always said.

    Never raises, for the same fail-closed reason: a configuration that cannot answer a
    question is refused with the question it could not answer.

    ``targets`` narrows the allocation clause exactly as in :func:`_absorber_reasons`.
    """
    if repair_path_for(pml) is not PLAIN_PATH:
        return ("the layer is ACTIVE, so both constitutive halves take the SPLIT-FIELD "
                "path (stepping.py:2130-2134) and write E = (E + kps*fresh) - kms*fw_prev; "
                "the plain repair writes E = constitutive and would drop the absorption "
                "the sub-step applied",)
    if pair not in PLAIN_PATH_SEAMS:
        return (f"seam {pair!r} is not served by the plain repair: with an inactive layer "
                f"update_H returns without touching H (stepping.py:944-945), so the "
                f"magnetic constitutive stores NOTHING on this branch and a repair here "
                f"would report a sub-step repaired that never wrote",)
    reasons: List[str] = []
    if not bool(getattr(fields, "stores_E", False)):
        reasons.append(
            "stores_E is False, so update_E returns at stepping.py:983-984 without "
            "writing anything: E is served on demand from D * inv_eps (Fields.get_E) "
            "and there is no stored array for a repair to put back")
    for component, _source, _axis in _checked_targets(pair, targets):
        field = getattr(fields, component, None)
        if field is None or getattr(field, "shape", None) is None:
            reasons.append(
                f"{component} is not allocated, so the array the plain branch writes "
                f"straight to storage (stepping.py:1019-1022) does not exist")
    for reader in ("displacement_minus_polarization", "inverse_epsilon_for"):
        if not callable(getattr(fields, reader, None)):
            reasons.append(
                f"fields does not expose {reader}; the plain repair recomputes "
                f"stepping.update_E:981-984 through it, and a constitutive product this "
                f"cannot form is not one it can restore")
    return tuple(reasons)


def repairable(fields: Any, pair: str,
               pml: Any = _PML_UNSUPPLIED, *,
               paths: Sequence[str] = (SPLIT_FIELD_PATH,),
               targets: Optional[Sequence[Any]] = None
               ) -> Tuple[bool, Tuple[str, ...]]:
    """May a deposit in ``pair``'s seam be carried? ``(ok, reasons)``, never an exception.

    FAIL CLOSED: anything this cannot establish is a refusal, because a predicate that
    admits on ignorance is the over-covering failure the fusion clauses exist to prevent.

    ``pml`` is the layer the constitutive half will read. Supplying it adds the clauses
    of whichever recurrence :func:`repair_path_for` says RAN. For the default ``paths``
    that is :func:`_absorber_reasons`, which refuses BY NAME a layer whose recurrence
    this repair does not invert -- ``None``, an inactive layer, or any coefficient whose
    layout is not the one :func:`apply` broadcasts. Omitting it is a third state, not
    ``None``: the clauses that need no layer still run, and every path that actually
    executes a repair supplies it (:func:`apply` takes it, :class:`LeadingRepairPlan`
    hands it to :func:`save`).

    ``paths`` IS THE CALLER'S DECLARATION OF WHAT IT CARRIES, not a request. The layer
    decides which recurrence ran; ``paths`` decides whether this caller can invert that
    one. The default is ``(SPLIT_FIELD_PATH,)``, which is the only repair that existed
    before :data:`PLAIN_PATH` was measured, so every predicate, plan and probe that does
    not name it gets exactly the answer it got before -- both absorber refusals included,
    unedited, since the split-field clauses still run whenever that path is declared.
    A caller declaring :data:`PLAIN_PATH` alone MUST supply the layer: with none supplied
    there is nothing to ask which recurrence ran, and admitting on that ignorance is the
    over-covering failure the fusion clauses exist to prevent.

    ``targets`` is for :func:`save` alone: the seam targets it is about to capture, so
    the per-target clauses (coefficient layout, allocation, ``f_w``) answer for those and
    not for components the restricted repair never touches. EVERY PREDICATE LEAVES IT
    ``None`` and is answered for the whole seam, as before -- a predicate names no
    source, so it has nothing to narrow to. The seam-wide clauses (the fold, the path,
    off-diagonal, nonlinear) never narrow.
    """
    reasons = []
    if pair not in SEAMS:
        return False, (f"unknown seam {pair!r}",)
    declaration = _path_declaration_reasons(paths)
    if declaration:
        return False, declaration
    grid = getattr(fields, "grid", None)
    if grid is None:
        reasons.append(
            "fields carries no grid, so whether this seam runs a post-injection "
            "symmetry or far-ghost fill -- and which rows it images -- is unknown")
    else:
        reasons.extend(_folded_seam_reasons(grid))
    # WHICH CLAUSES RUN IS DECIDED BY THE LAYER, through _selected_path. A caller that
    # carries the split-field path keeps being answered by its clauses whatever the layer
    # turns out to be, so `None` and an inactive layer still produce the two named
    # refusals they always did; a caller that carries only the plain path is refused by
    # the mirror clause instead, which names the same fork from the other side.
    selected = _selected_path(pml, paths)
    if pml is not _PML_UNSUPPLIED:
        reasons.extend(_absorber_reasons(fields, pml, pair, targets=targets)
                       if selected is SPLIT_FIELD_PATH
                       else _plain_path_reasons(fields, pml, pair, targets=targets))
    elif selected is PLAIN_PATH:
        reasons.append(
            "no layer was supplied, so which constitutive recurrence ran cannot be "
            "established; the plain repair is the inverse of the branch "
            "stepping._pml_is_active takes when it is False, and a caller that carries "
            "only that branch has to be handed the layer to be sure it is the one")
    if pair == "D":
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row: update_E reads the PARTNER components' "
                "volumes at shifted indices (stepping.py:1228-1251), so the constitutive "
                "half is a stencil over what the curl wrote in place and no repair over "
                "the deposit points can reconstruct it")
        if getattr(fields, "has_nonlinearity", False):
            reasons.append(
                "an instantaneous chi2/chi3: the constitutive step is not the linear "
                "accumulation this repair inverts")
    # THE SPLIT-FIELD PATH'S STATE CLAUSE, and it belongs to that path alone. The plain
    # branch allocates no `f_w_*` at all (fields.py:1160-1162 returns the stored E
    # instead), so requiring one there would refuse every configuration the second
    # repair exists for -- while dropping it for the split-field path would let a
    # stateful recurrence be "restored" from state that was never captured.
    if selected is SPLIT_FIELD_PATH:
        for component, _source, _axis in _checked_targets(pair, targets):
            if getattr(fields, "f_w_" + component, None) is None:
                reasons.append(
                    f"f_w_{component} is not allocated; there is no state to save")
                break
    return (not reasons), tuple(reasons)


#: The reserved key :func:`save` stamps its own repair path into, so :func:`apply` reads
#: which recurrence the state beside it was captured for instead of being told a second
#: time. A tuple, so it cannot collide with a ``(component, index, field)`` key.
_PATH_KEY = ("__repair_path__",)

#: The reserved key :func:`save` stamps its :class:`_PreparedCells` into -- a plan's, or
#: the one-shot one it builds for a caller holding no plan -- for the same reason as
#: :data:`_PATH_KEY`: :func:`apply` reads the linear spelling of the cell set from the
#: state rather than being handed it as a sixth argument. Always present in a dict
#: :func:`save` returns; a dict built by hand without it takes the 3-tuple route in
#: :func:`apply`, which writes the same cells. That is not only
#: discipline -- ``apply`` is monkeypatched at a fixed five-argument arity by
#: ``parity/meep_gpu/probe_dispatch_corpus_byteparity.py:345``, so a keyword there would
#: break the armed control that proves a skipped repair is visible.
_PREPARED_KEY = ("__prepared_cells__",)

#: The reserved key :func:`save` stamps ``{source index: the seam targets it captured}``
#: into, for the same reason as the two above: :func:`apply` repairs exactly what was
#: SAVED, read from the state, rather than deciding a second time which component each
#: source writes -- a decision made twice is one that can disagree once, and the side
#: that disagreed would restore a component from state nobody captured. A dict WITHOUT
#: the stamp -- built by hand, or by a ``save`` older than the restriction -- named no
#: targets, and what it meant was every target of the seam; :func:`apply` reads it so.
_TARGETS_KEY = ("__source_targets__",)


# ---------------------------------------------------------------------------
# The per-step cost of the bracket: one linear index per component, built once
# ---------------------------------------------------------------------------
#
# THE BRACKET WAS THE DOMINANT COST OF EVERY FUSED PLAN THAT LOSES TO ITS SINGLES,
# measured 2026-09-17/19: ~2 ms/step of host-issued work on the two NVIDIA kernel
# tables (Triton and hand-CUDA, both CuPy-backed), 31 of 31 timed rows holding a
# bracketed seam losing and all 6 clean-seam one-pair rows winning. NO METAL FUSED
# TIMING EXISTS: no ``fused_timing*`` / ``fused_ablation*`` result carries a Metal leg,
# and the Metal bracket is this same code on host NumPy, where it costs tens of
# microseconds rather than milliseconds (measured below), so nothing here is a claim
# about that table. Counted on the pml_2d lift, per step: save 12 array ops (6 three-array fancy
# gathers, each followed by a redundant ``xp.array(copy=True)``), apply 30 (9 gathers, 6
# scatters, THREE WHOLE-VOLUME ``volume * inv_eps`` products, 12 small elementwise), and
# on a folded grid 16 more in ``repair_cells`` -- six of them boolean-mask gathers, each
# a device-to-host sync on CuPy. A three-array fancy index is several kernels and
# allocations on a device, not one.
#
# WHAT CHANGED, and why each piece is exact rather than close:
#
# * THE CELL SET IS BUILT ONCE PER PLAN, not once per step. :class:`LeadingRepairPlan`
#   owns a :class:`_PreparedCells`; each ``(component, source)`` entry holds the
#   ``repair_cells`` 3-tuple AND its C-order linear index. The entry is keyed on what
#   the set is a function of -- the deposit index arrays (identity; the sources rebind
#   them, ``sources.py:1667``/``:1829``/``:1943``/``:2069``, and nothing writes into
#   them), the fill rules :func:`_fill_image_rules` derives this step (equality), and
#   the ``repair_cells`` the module holds now (identity, so a gate's point-only leg that
#   swaps it is honoured on the step it is swapped). A changed key rebuilds the entry.
# * A CALLER HOLDING NO PLAN TAKES THE SAME ROUTE, from a ONE-SHOT cache (2026-09-19,
#   reviewer finding F3). The gates and probes that bracket a launch by hand call the
#   module-level :func:`save`/:func:`apply` -- e.g.
#   ``parity/meep_gpu/gate_cuda_conductive_fused_electric_pair.py:697``/``:711``,
#   ``probe_cuda_three_slot_weld.py:459``, ``probe_cuda_electric_fold_closure.py:254`` --
#   while every production installer builds the plans (``triton_kernels/launch.py:1750``,
#   ``metal_kernels/launch.py:1371``, ``cuda_kernels/fused_pairs.py:4359``/``:4402``). With
#   ``prepared=None`` :func:`save` used to take the 3-tuple route, so those legs certified
#   a route the product no longer ran. It now builds a fresh :class:`_PreparedCells` for
#   the one call and stamps it like a plan's, so both halves run the product's route and
#   fall back exactly where a plan's entry would. Fresh per call, never module-global: a
#   point-only control that swaps ``repair_cells`` between calls gets its own set on the
#   next call, with no key to go stale. The price is that such a caller builds the entry
#   -- ``repair_cells`` plus :func:`_linear_index`'s six host reads -- on every call
#   rather than once, which is a gate's cost and not the product's.
# * EVERY ARRAY IS RE-READ FROM ``fields`` EVERY STEP. Only the index is cached, and it
#   is used on an array only when that array's shape is the one the index was built for
#   and it is C-contiguous (:func:`_flat`), so ``reshape(-1)`` is a VIEW that a scatter
#   writes through. The index was range-checked per axis when it was built, so an equal
#   shape keeps it in range with no per-step reduction. Any array that fails either
#   test sends that ``(component, source)`` down the 3-tuple route instead -- never a
#   refusal, because the two routes read and write the same cells.
# * THE CONSTITUTIVE PRODUCT IS GATHERED, THEN MULTIPLIED. ``(volume * inv_eps)[cells]``
#   and ``volume[cells] * inv_eps[cells]`` are the same float product of the same two
#   operands per cell -- IEEE multiplication is correctly rounded and no neighbour is
#   read -- so forming it over the whole volume was pure cost. Taken only when the two
#   arrays have the same shape; a broadcastable ``inv_eps`` keeps the whole-volume form.
# * NO SECOND COPY. An integer-array (advanced) index returns a NEW array -- NumPy
#   documents that advanced indexing always returns a copy, and CuPy follows NumPy's
#   indexing semantics (read, not measured: this host has no CuPy) -- so the
#   ``xp.array(..., copy=True)`` after it copied a copy. It is kept for the one index
#   shape that is not advanced -- scalar parts, which CuPy answers with a 0-d VIEW -- in
#   :func:`_gathered`.
# * THE COEFFICIENTS AT THE CELLS ARE GATHERED ONCE, by the shipped spelling
#   ``xp.broadcast_to(kps, shape)[cells]``, and reused while the layer hands back the
#   SAME ``kps``/``kms`` objects (checked every step). ``PML._compute_coefficients`` is
#   called once, from ``PML.__init__`` (``pml.py:385``), binds each coefficient
#   (``pml.py:698-717``) and nothing in the engine writes into one afterwards.
#
# THE RECURRENCE IS UNTOUCHED: ``restored = f_old + kp_c * fresh`` then
# ``restored - km_c * fw_old`` -- the operand order of ``_apply_constitutive_pml``
# (``stepping.py:2130-2134``) exactly as the 3-tuple route writes it.
#
# MEASURED 2026-09-18 on NumPy lifts through the two plans, steady state (the first
# step builds the cache): pml_2d save 12 -> 6 ops and apply 30 -> 27, every one a
# one-array gather/scatter or an elementwise op over the deposit cells, with no
# whole-volume product; folded_2d save 28 -> 6 (the closure is not rebuilt); the plain
# path 9 -> 12 ops but 3 -> 0 whole-volume products. Host time per step 90.7 -> 57.8 us
# (pml_2d), 111.8 -> 69.0 us (folded_2d), 30.5 -> 24.8 us (absorber_1d, plain). Against
# a copy of the arithmetic this replaced, 25 seams -- the corpus lifts plus every folded
# closure case of ``parity/meep_gpu/probe_triton_folded_deposit_closure.py``, real and
# complex, both seams, both paths -- 0 of 84,449,844 words differ as uint32 (a plan leg
# and a module-level leg, 42,224,922 each), with the linear route taken on every entry
# of the PLAN leg; the module-level leg then took the 3-tuple route, which is F3 below.
# ``test_deposit_repair_linear_route.py`` pins that.
#
# RE-MEASURED 2026-09-19 after the one-shot (F3), same 25 seams and 6 trials each, now
# counting BOTH live routes against that copy: 0 of 84,449,844 words differ (42,224,922
# per route), and the plan and module-level routes differ from each other by 0. Route
# counters, identical on the two: 414 linear saves and 450 linear repairs -- 18 of 18
# per split-field seam (23 seams), 0/18 on the 2 plain seams, which save no state -- and
# 0 fallbacks on either. Before the one-shot, the module-level leg took the 3-tuple
# route on every one of those entries by construction: it had no cache to spell it.
#
# STILL WHOLE-VOLUME, and out of this change: on a dispersive row
# ``displacement_minus_polarization`` forms D - sum P over the whole grid, once per
# REPAIRED component per step (``fields.py:1079-1105``), exactly as ``update_E`` does.
# Gathering it would restate the polarization sum here rather than call the one that
# ran. Since 2026-09-20 that is once per component some source WRITES rather than three
# times: ``1 + poles`` whole-volume passes per written component, not ``3 * (1 + poles)``.
#
# RESTRICTED TO THE WRITTEN COMPONENT, 2026-09-20 (the comment above
# :func:`_written_targets` carries the argument and the citations). Per single-component
# source per step the bracket is ONE entry where it was three, so every route counter
# above reads 1 per source per step on a row timed after this change and 3 on a row
# timed before it -- ``linear_repairs`` / step is what tells the two apart in a timing
# record. Host time of one bracket, NumPy, both legs interleaved in ONE process against
# a copy of this file at sha256 ``e4f88c1e4fc8`` (a no-op inner, steady state, median of
# 6 windows of 400 steps): pml_2d 69.4 -> 32.7 us, magnetic_seam_2d 65.0 -> 30.9 us,
# folded_2d 82.2 -> 40.4 us, dispersive_2d 86.5 -> 38.5 us, absorber_1d (plain)
# 29.9 -> 18.4 us. The per-target clauses of the two checks below narrow with it
# (:func:`_checked_targets`), which is part of that.
#
# NOT CHANGED, and deliberately: :func:`repairable` still runs every step inside
# :func:`save`, and :func:`apply` still re-asserts :func:`_absorber_reasons` before it
# writes. Both are host-only and issue no array operation, and they are now the largest
# single piece of the restricted bracket's host time. CACHING THEIR VERDICT PER PLAN
# WAS EXAMINED WITH THE RESTRICTION AND NOT BUILT, because the proof it needs fails. A
# cached verdict is sound only if every attribute the checks read is covered by an
# event that DROPS the plan. The one such event is ``FdtdDriver.invalidate_fast_path``
# (``driver.py:3100-3113``), which replaces the plan and is called by the DRIVER's
# mutators (``:1183``, ``:1254``, ``:1487``, ``:1553``, ``:1697``, ``:1765``, ``:1868``,
# ``:2021``, ``:2256``, ``:2472``, ``:2508``, ``:4015``, ``:4440``, ``:4464``). What the
# checks read is reachable around it:
#
# * ``fields.has_nonlinearity`` -- ``Fields._chi2_components``, written by the public
#   ``Fields.set_nonlinear_volumes`` (``fields.py:838``). ``FdtdDriver.fields`` is a
#   public attribute, and a caller that reaches the mutator through it calls no
#   invalidation. THIS ONE ATTRIBUTE IS ENOUGH: today the next :func:`save` refuses by
#   name before the launch; a cached "repairable" would restore a linear accumulation
#   over a constitutive that is no longer linear, and report success.
# * ``fields.has_offdiagonal_epsilon`` -- ``Fields._chi1inv_offdiagonal``, written by
#   the public ``Fields.set_epsilon_volumes`` (``fields.py:1210``), the same way.
# * ``fields.f_w_*``, each target's allocation and shape, ``fields.stores_E`` -- plain
#   attributes of a public object.
# * ``pml.is_active`` and the six ``kps*`` / ``kms*`` tables' identity and shape -- the
#   plan holds the layer object, and nothing stops one being rebound on it
#   (:func:`_coefficients_at` re-checks identity every step for exactly that reason).
# * the grid's fold map (``has_symmetry``, ``is_axis``, ``_mirror_phases``,
#   ``_far_reflect_rows``, ``stored_cells``).
# * AND THE CALLERS THAT HOLD NO PLAN AT ALL: every gate and probe that brackets a
#   launch by hand calls the module-level :func:`save` / :func:`apply`, where there is no
#   plan to key a verdict on and no event that could drop it.
#
# A stale "repairable" is a silent wrong answer, and what the cache would buy is tens of
# microseconds of host time against a ~1 ms device-side bracket.
# ``test_deposit_repair_component_restriction.py`` mutates each attribute above under a
# live plan and requires the next step to refuse before the launch, so a cache added
# later has to key on all of them or fail there.


class _CellEntry:
    """One ``(component, source)`` cell set, in the two spellings the repair reads it by.

    ``cells`` is exactly what :func:`repair_cells` returned; ``linear`` is its C-order
    index into an array of ``shape``, or None when the set cannot be spelled that way
    (a column that is not a 1-D integer array, columns of different lengths, or an index
    outside ``shape`` on its own axis), in which case every step takes the 3-tuple route.
    """

    __slots__ = ("builder", "index", "shifts", "rules", "cells", "count", "shape",
                 "linear", "coefficients")


class _PreparedCells:
    """One plan's deposit cell sets, built on the first step and reused on every other --
    or, built by :func:`save` for a caller holding no plan, one call's, discarded with the
    saved dict it rides in.

    The four ROUTE counters record which route each ``(component, source)`` took, per
    call: ``linear_*`` the cached linear index, ``cells_*`` the 3-tuple fallback. They
    are what a test reads to know the fast route was TAKEN -- a bit-identity check
    against a route that silently fell back would pass for the wrong reason.

    The two RESTRICTION counters say the same about the component restriction, per
    source per :func:`save`: ``restricted_sources`` a source narrowed to the target(s)
    its injection writes, ``restriction_fallbacks`` a source for which that could not
    be established and which therefore kept every target of the seam. A fallback is a
    sound repair -- it is the full bracket this module shipped until 2026-09-20 -- and
    it is three entries where one was possible, so a route gate or a timing row that
    reads a non-zero ``restriction_fallbacks`` is reading a bracket that did not get
    cheaper. The route counters carry the same fact without it: they advance ONE per
    single-component source per step when restricted and THREE when not, which is also
    what tells a row timed before the restriction from one timed after.
    """

    __slots__ = ("entries", "linear_saves", "cells_saves", "linear_repairs",
                 "cells_repairs", "restricted_sources", "restriction_fallbacks")

    def __init__(self) -> None:
        self.entries: Dict[Tuple[str, int], _CellEntry] = {}
        self.linear_saves = 0
        self.cells_saves = 0
        self.linear_repairs = 0
        self.cells_repairs = 0
        self.restricted_sources = 0
        self.restriction_fallbacks = 0


def _linear_index(xp: Any, cells: Tuple[Any, Any, Any], shape: Any) -> Optional[Any]:
    """``cells`` as ONE C-order index into an array of ``shape``, or None.

    Range-checked PER AXIS, once: a linear index is in range for the array whenever each
    column is in range for its axis, but not conversely -- ``iz == nz`` would alias the
    next row silently where the 3-tuple index raises. The check reads six scalars back
    to the host, which is why a plan does it once, when the entry is built, and never
    per step; a caller holding no plan builds its one-shot entry, and so pays the six
    reads, on every :func:`save` (see the comment above :class:`_CellEntry`).
    """
    if shape is None or len(shape) != 3 or len(cells) != 3:
        return None
    extents = tuple(int(extent) for extent in shape)
    columns = []
    for column in cells:
        kind = getattr(getattr(column, "dtype", None), "kind", None)
        if getattr(column, "ndim", None) != 1 or kind not in ("i", "u"):
            return None
        columns.append(column.astype(xp.int64))
    if len({int(column.size) for column in columns}) != 1:
        return None
    if int(columns[0].size):
        for axis, column in enumerate(columns):
            if int(column.min()) < 0 or int(column.max()) >= extents[axis]:
                return None
    _nx, ny, nz = extents
    return (columns[0] * ny + columns[1]) * nz + columns[2]


def _prepared_entry(prepared: _PreparedCells, fields: Any, component: str, target: str,
                    index: int, idx: Tuple[Any, Any, Any]) -> _CellEntry:
    """The cached cell set for one ``(component, source)``, rebuilt if its key moved."""
    grid = fields.grid
    target_shape = getattr(fields, target).shape
    entry = prepared.entries.get((component, index))
    if (entry is not None and entry.builder is repair_cells
            and entry.index[0] is idx[0] and entry.index[1] is idx[1]
            and entry.index[2] is idx[2]
            and entry.rules == _fill_image_rules(grid, entry.shifts, target_shape)):
        return entry
    entry = _CellEntry()
    entry.builder = repair_cells
    entry.index = idx
    entry.shifts = _yee_shifts(target)
    entry.rules = _fill_image_rules(grid, entry.shifts, target_shape)
    # THE MODULE'S repair_cells, looked up now: the gates' point-only control leg swaps
    # it on the module, and the set it returns is the set this entry must carry.
    entry.cells = repair_cells(fields, target, idx)
    entry.count = int(getattr(entry.cells[0], "size", len(entry.cells[0])))
    shape = getattr(getattr(fields, component), "shape", None)
    entry.linear = _linear_index(grid.xp, entry.cells, shape)
    entry.shape = (tuple(int(extent) for extent in shape)
                   if entry.linear is not None else None)
    entry.coefficients = None
    prepared.entries[(component, index)] = entry
    return entry


def _linear_entry(prepared: Optional[_PreparedCells], component: str, index: int,
                  cells: Any) -> Optional[_CellEntry]:
    """The prepared entry whose linear index spells THESE cells, or None.

    Identity with the saved 3-tuple is the tie: the linear route may only write the set
    :func:`save` captured state for, so an entry that no longer holds that exact tuple
    is not used.
    """
    if prepared is None:
        return None
    entry = prepared.entries.get((component, index))
    if entry is None or entry.linear is None or entry.cells is not cells:
        return None
    return entry


def _flat(array: Any, shape: Tuple[int, ...]) -> Optional[Any]:
    """``array`` as its own 1-D C-order VIEW, or None when the linear index cannot use it.

    Shape equality keeps the per-axis range check made at build time valid; C-contiguity
    makes ``reshape(-1)`` a view, so a scatter through it lands in ``array``.
    """
    if getattr(array, "shape", None) != shape:
        return None
    flags = getattr(array, "flags", None)
    if flags is None or not flags.c_contiguous:
        return None
    return array.reshape(-1)


def _flat_views(shape: Tuple[int, ...], arrays: Sequence[Any]) -> Optional[List[Any]]:
    """:func:`_flat` of every array, or None if any one of them fails it."""
    views = []
    for array in arrays:
        view = _flat(array, shape)
        if view is None:
            return None
        views.append(view)
    return views


def _fresh_at(fields: Any, component: str, linear: Any) -> Any:
    """``displacement_minus_polarization(component)`` at flat cells ``linear`` only.

    THE WHOLE-VOLUME FORM IS THE PROBLEM THIS AVOIDS. ``Fields.displacement_minus_polarization``
    forms ``D - sum P`` over the entire grid into a shared scratch every step, and
    under a held residency each of those reads is a whole-volume sync. ``apply``
    consumes that volume at the deposit cells only, so this gathers D and each
    contributing P at those cells and forms the same differences, in the same
    contributor order, in the same dtype -- elementwise subtraction rounds per
    element, so the subset is bit-for-bit the subset of the whole. When no
    susceptibility drives the component it is D at the cells, exactly as the
    whole-volume form aliases D.
    """
    displacement = host_writes.raw(fields, "D" + component[1])
    fresh = host_writes.gather(displacement, linear)
    contributors = [state for state in (getattr(fields, "polarizations", ()) or ())
                    if state.drives(component)]
    if not contributors:
        return fresh
    fresh = fresh.copy()
    for state in contributors:
        fresh -= host_writes.gather(host_writes.raw_item(state.P, component), linear)
    return fresh


def _sparse_linear(xp: Any, array: Any, cells: Tuple[Any, ...]) -> Optional[Any]:
    """Flat C-order indices for ``array[cells]``, or None when the array cannot take them.

    THE SAME GUARD ``_flat`` MAKES, and it is not optional: a linear index is only a
    C-order flat index when the array is C-contiguous. On a Fortran-ordered or
    strided target ``reshape(-1)`` is a COPY, so a scatter through it lands nowhere
    and a gather through it reads the wrong cells -- measured by the linear-route
    tests the moment the sparse door skipped this check. Those arrays keep the
    3-tuple fancy index, which is exact on any layout.
    """
    flags = getattr(array, "flags", None)
    if flags is None or not flags.c_contiguous:
        return None
    return _linear_index(xp, cells, getattr(array, "shape", None))


def _gathered(xp: Any, array: Any, cells: Tuple[Any, ...]) -> Any:
    """``array[cells]`` as a NEW array: the pre-launch state :func:`save` keeps.

    An index whose every part is an array of rank >= 1 is an advanced index, which
    returns a new array on NumPy and CuPy alike, so no second copy is made. Any scalar
    part makes it a basic index -- a 0-d VIEW on CuPy that the launch would overwrite
    -- and that one shape keeps the explicit copy it always had.
    """
    linear = _sparse_linear(xp, array, cells)
    if linear is not None:
        return host_writes.gather(array, linear)
    values = array[cells]
    if all(getattr(part, "ndim", 0) >= 1 for part in cells):
        return values
    return xp.array(values, copy=True)


def _scatter_cells(xp: Any, array: Any, cells: Tuple[Any, ...], values: Any) -> None:
    """``array[cells] = values`` through the sparse door when the cells are linearisable.

    The 3-tuple route is the fallback the linear route refuses (a non-3-D shape, an
    out-of-range column). When it can still be expressed as flat indices it is
    scattered; when it cannot, the write goes through a whole-volume ``acquire`` --
    correct, and paid for only on the shape that needs it.
    """
    linear = _sparse_linear(xp, array, cells)
    if linear is not None:
        host_writes.scatter(array, linear, values)
        return
    host_writes.acquire(array)[cells] = values


def _constitutive_at(volume: Any, inverse: Any, cells: Tuple[Any, ...]) -> Any:
    """The constitutive product at ``cells`` on the 3-tuple route.

    ``volume[cells] * inverse[cells]`` is per cell the same float product of the same two
    operands as ``(volume * inverse)[cells]``, without forming the other N - n cells.
    The whole-volume form is kept only when the two do not share a shape, where the
    product broadcasts and indexing each operand separately would not.
    """
    if inverse is None:
        return volume[cells]
    if getattr(volume, "shape", None) == getattr(inverse, "shape", None):
        return volume[cells] * inverse[cells]
    return (volume * inverse)[cells]


def _coefficients_at(xp: Any, entry: _CellEntry, kps: Any, kms: Any) -> Tuple[Any, Any]:
    """``kps``/``kms`` at the entry's cells, gathered once per coefficient object.

    Gathered by the 3-tuple route's own spelling, so the cached values are by
    construction the ones it reads; reused while the layer returns the same two objects.
    """
    cached = entry.coefficients
    if cached is not None and cached[0] is kps and cached[1] is kms:
        return cached[2], cached[3]
    kp_cells = xp.broadcast_to(xp.asarray(kps), entry.shape)[entry.cells]
    km_cells = xp.broadcast_to(xp.asarray(kms), entry.shape)[entry.cells]
    entry.coefficients = (kps, kms, kp_cells, km_cells)
    return kp_cells, km_cells


# ---------------------------------------------------------------------------
# The component restriction: repair the target a source WRITES, and only that one
# ---------------------------------------------------------------------------
#
# A SOURCE WRITES ONE ARRAY, AND A DIAGONAL CONSTITUTIVE TARGET READS ONLY ITS OWN.
# Both halves are read from the code, because an argument that narrows what is
# repaired is the one kind here that fails SILENTLY when it is wrong:
#
# * WHAT AN INJECTION WRITES. All four engine source classes resolve their array from
#   ``self.component`` through ``sources._ARRAY_FOR_COMPONENT`` (``sources.py:120-133``;
#   ``Ez`` and ``Dz`` both name ``Dz``, ``Hz`` and ``Bz`` both name ``Bz``) and write
#   that array alone:
#     ``ContinuousSource.inject``      ``_d_array_for`` at ``sources.py:1733``, the
#                                      write at ``:1742``;
#     ``ExtendedSource.inject``        ``_d_array_for`` at ``:1850``, ``_inject_points``
#                                      at ``:1855-1856`` (the write is ``:1305``);
#     ``GaussianPulsedSource.inject``  ``_d_array_for`` at ``:1962``, ``_inject_points``
#                                      at ``:1969-1970``;
#     ``VolumeSource.inject``          ``_array_for`` at ``:2139``, ``_inject_points`` at
#                                      ``:2151-2152`` -- integrated or not -- and its
#                                      ``withdraw`` (``:2118-2120``) writes the same
#                                      array BEFORE the leading consult, so before
#                                      :func:`save`.
#   ``_d_array_for`` / ``_array_for`` (``sources.py:232-237``) are ``getattr(fields,
#   <that map's value>)``. Each class also mirrors its deposit into ``fu_<array>``
#   (``:1745-1749``, ``_inject_fu_mirror`` ``:1418-1426``): that is the CURL's split
#   auxiliary, which no constitutive half reads, so it bears on nothing repaired here.
# * WHAT THE CONDUCTIVE REPLAY WRITES. ``driver._inject_electric_through_conductivity``
#   groups the scaled sources by ``"D" + source.component[1]`` (``driver.py:3406``),
#   snapshots and rewrites that array at the sources' own cells (``:3431``, ``:3445``)
#   around the same ``inject`` calls (``:3433``), and injects the integrated ones
#   unscaled (``:3401``). For every electric spelling that name IS the map's value;
#   :func:`_written_targets` checks the two agree rather than assuming it.
# * WHAT A TARGET READS. ``_apply_constitutive_pml`` and the plain branch of
#   ``update_E`` read, for component c at a cell, ``D_c`` / ``B_c``, the poles'
#   ``P_n[c]``, ``inv_eps(c)``, c's own coefficients and c's own ``field`` / ``f_w`` AT
#   THAT CELL. The two couplings that would break this are both refused by name in
#   :func:`repairable`: an off-diagonal ``chi1inv`` row and an instantaneous
#   ``chi2``/``chi3``. ``stepping.H_CONSTITUTIVE_TERMS`` is diagonal with mu = 1.
# * WHAT RUNS BETWEEN THE TWO CONSULTS. The injection, ``fill_symmetry_bc_*``,
#   ``zero_metal_*`` and ``fill_folded_far_ghosts_*`` (``driver.py:3293-3310``,
#   ``:3318-3330``). Each fill and the wall clear write a component's plane from the
#   SAME component's plane, so a deposit into ``D_c`` reaches ``D_c`` cells only --
#   the image closure of :func:`repair_cells`, computed per component.
#
# So at every cell of a component the source did NOT write, the constitutive input is
# what the fused launch already consumed, and the launch's result stands. Recomputing
# it was bit-identical (the over-coverage note above :func:`_yee_shifts`) and was two
# of every three entries in the bracket.
#
# MEASURED before it was written, with the restriction applied in-process: 24 of 24
# host configurations bit-identical with a diverging wrong-component control, and on
# the device (one RTX A6000, both NVIDIA tables, 2026-09-20) 84 of 84 legs bit-identical
# with the fused step 1.47-1.66x faster at <= 1.0M cells
# (``the design notes (deposit-repair-kernel-scope)`` sections 1 and 9).
# ``test_deposit_repair_component_restriction.py`` pins the shipped form against the
# frozen full-bracket arithmetic, the wrong-component control included.
#
# FAIL CLOSED, AND COUNT IT. Anything that stops the written array being established
# -- a source class whose ``inject`` is not one of the four read above (a subclass
# included: it may override ``inject``), an ``inject`` REBOUND on the class or on the
# instance so that what runs is not the function that was read, a ``component`` that
# is not a string or that the map does not hold, an array the seam does not inject,
# the two electric spellings disagreeing, or a lookup answering anything but a
# repeat-free selection of this seam's own targets -- keeps ALL the seam's targets for
# that source, which is the bracket this module shipped until 2026-09-20, and bumps
# ``_PreparedCells.restriction_fallbacks``. WHAT WOULD BREAK THE ARGUMENT, and is
# therefore what to re-read before widening the list below: a source that writes a
# second array, a fill that images one component into another, an admitted
# off-diagonal permittivity or permeability, or a nonlinearity.
#
# THE TYPE IS NOT THE INJECT, so :func:`_runs_the_read_inject` looks at the function.
# An exact-class source whose ``inject`` had been rebound to write ``Dx`` beside ``Dz``
# was narrowed to ``Ez`` by its type alone, counted as RESTRICTED, and left ``Ex`` and
# ``f_w_Ex`` differing from the driver's order at the deposit cell; the full bracket
# read 0 words on the same case, by over-covering
# (``test_deposit_repair_component_restriction.py`` keeps that narrowing as the null
# control of its rebound-inject test). Two in-tree instruments rebind an ``inject``
# (``parity/meep_gpu/probe_dispatch_seam_sweep.py:191`` on the instance,
# ``gate_metal_complex_conductive_fused_pair.py:970`` on the class); both delegate and
# write nothing of their own, so under them the bracket is the full one, exact and
# COUNTED, for as long as the wrapper is in place. WHAT THIS STILL DOES NOT SEE: a
# rebinding of what a read ``inject`` CALLS -- ``sources._inject_points``,
# ``_d_array_for``, ``_array_for`` -- or of the driver's conductive replay. Nothing in
# the tree does either.

#: The engine source classes whose ``inject`` was READ for the citations above. Matched
#: by exact type against ``sources``' own class of that name, never by ``isinstance``:
#: a subclass may override ``inject``, and an unread ``inject`` is a fallback.
_READ_INJECTS: Tuple[str, ...] = ("ContinuousSource", "ExtendedSource",
                                  "GaussianPulsedSource", "VolumeSource")


def _runs_the_read_inject(source: Any, kind: Any, engine_sources: Any) -> bool:
    """Whether calling ``source.inject`` runs the function that was read for ``kind``.

    ON THE CLASS: the function in the class's OWN dict, defined in the sources module
    under that class's name. ``__globals__`` is the test because it cannot be dressed
    up: ``functools.wraps`` copies ``__module__`` and ``__qualname__`` onto a wrapper
    and cannot copy this, and it is an identity, so it holds wherever the tree was
    staged to and whatever file name the bytecode carries.

    ON THE INSTANCE: nothing named ``inject`` in the instance's dict -- or exactly the
    bound method of that same function on this same source, which is what an
    instrument leaves behind when it restores by ``setattr`` rather than ``del``
    (``probe_dispatch_seam_sweep.py:195``). An instance whose dict cannot be read is
    not established, like everything else here.
    """
    read = kind.__dict__.get("inject")
    if (getattr(read, "__globals__", None) is not vars(engine_sources)
            or getattr(read, "__qualname__", None) != kind.__name__ + ".inject"):
        return False
    try:
        own = vars(source).get("inject")
    except TypeError:
        return False
    return own is None or (getattr(own, "__func__", None) is read
                           and getattr(own, "__self__", None) is source)


def _written_targets(source: Any, pair: str) -> Optional[Tuple[Any, ...]]:
    """The targets of ``pair`` whose constitutive INPUT this source's injection writes.

    ``None`` whenever that cannot be established, which the caller reads as "every
    target". Never raises. Looked up through the module on every call, so a control
    that plants a wrong answer here is honoured on the step it is planted.
    """
    spec = SEAMS.get(pair)
    if spec is None:
        return None
    try:
        from . import sources as engine_sources  # noqa: PLC0415 - engine objects, late
    except Exception:  # noqa: BLE001 - no engine to ask is a fallback, not an error
        return None
    kind = type(source)
    if (kind.__name__ not in _READ_INJECTS
            or getattr(engine_sources, kind.__name__, None) is not kind
            or not _runs_the_read_inject(source, kind, engine_sources)):
        return None
    component = getattr(source, "component", None)
    if not isinstance(component, str):
        return None
    written = getattr(engine_sources, "_ARRAY_FOR_COMPONENT", {}).get(component)
    if written is None or written not in spec["injected"]:
        return None
    if pair == "D" and written != "D" + component[1:2]:
        return None  # the conductive replay's own spelling (driver.py:3406) disagrees
    matches = tuple(target for target in spec["targets"] if target[1] == written)
    return matches if len(matches) == 1 else None


def _source_targets(source: Any, pair: str) -> Tuple[Tuple[Any, ...], bool]:
    """``(the seam targets to repair for this source, whether that is a restriction)``.

    :func:`_written_targets` is a seam a control may patch, so its answer is CHECKED
    before it narrows anything: only a non-empty, repeat-free selection of this seam's
    own targets is taken, in the seam's order; anything else is the full set.
    """
    full = SEAMS[pair]["targets"]
    try:
        answer = _written_targets(source, pair)
        if type(answer) is tuple and len(answer) == 1 and answer[0] in full:
            return answer, True  # every engine source class: one written target
        if (isinstance(answer, tuple) and answer and len(set(answer)) == len(answer)
                and all(target in full for target in answer)):
            return tuple(target for target in full if target in answer), True
    except Exception:  # noqa: BLE001 - an answer that cannot be checked is a fallback
        pass
    return full, False


def _union_in_seam_order(pair: str, selections: Sequence[Sequence[Any]]
                         ) -> Optional[Tuple[Any, ...]]:
    """Every target some source selected, in ``SEAMS`` order; ``None`` for none at all."""
    if len(selections) == 1:
        return tuple(selections[0]) or None  # already in seam order, see _source_targets
    chosen = tuple(target for target in SEAMS[pair]["targets"]
                   if any(target in selection for selection in selections))
    return chosen or None


def _repair_work(sources: Sequence[Any], pair: str, saved: Dict[Any, Any],
                 state: Sequence[str]) -> Tuple[List[Tuple[Any, List[int]]], List[str]]:
    """What :func:`apply` is to repair, read off the SAVED state: ``(work, missing)``.

    ``work`` is ``(target, [source index, ...])`` in ``SEAMS`` order, holding only
    targets some source was saved for -- so on a dispersive row ``D - sum P`` is formed
    once per WRITTEN component, each consumed before the next is formed, which is the
    invariant the shared scratch (``fields.py:1096-1105``) needs. ``missing`` names every
    piece of state a repair would need and the dict does not hold; the caller refuses on
    it BEFORE the first write, because a partial repair is worse than none.
    """
    full = SEAMS[pair]["targets"]
    stamped = saved.get(_TARGETS_KEY)
    indices: Dict[Any, List[int]] = {target: [] for target in full}
    missing: List[str] = []
    for index, source in enumerate(sources):
        if _deposit_index(source) is None:
            continue
        targets = full if stamped is None else stamped.get(index)
        if targets is None:
            missing.append(
                f"no saved state for source {index}; save and apply were handed "
                f"different source lists")
            continue
        for target in targets:
            if target not in indices:
                missing.append(
                    f"source {index} was saved for {target!r}, which seam {pair!r} does "
                    f"not declare; save and apply read different seam tables")
                continue
            absent = [name for name in state
                      if (target[0], index, name) not in saved]
            if absent:
                # The save ran over a different source list, so there is no state for
                # this deposit. Raising beats writing part of the seam: `dispatch`
                # already reported that the whole sub-step ran.
                missing.append(
                    f"no saved state for {target[0]} of source {index} ({absent}); save "
                    f"and apply were handed different source lists")
                continue
            indices[target].append(index)
    return [(target, found) for target, found in indices.items() if found], missing


def save(fields: Any, sources: Sequence[Any], pair: str,
         pml: Any = _PML_UNSUPPLIED, *,
         paths: Sequence[str] = (SPLIT_FIELD_PATH,),
         prepared: Optional[_PreparedCells] = None) -> Dict[Any, Any]:
    """Capture the state the fused launch will overwrite, before it runs.

    THE CELL SET IS SAVED WITH THE STATE, not recomputed in :func:`apply`. The two
    halves must agree exactly -- ``apply`` needs ``field_old`` and ``fw_old`` at every
    image it restores -- and a set computed twice is a set that can disagree once. That
    holds on BOTH paths, and on :data:`PLAIN_PATH` the cell set is the whole of what is
    saved: the plain branch is a pure overwrite (``stepping.py:1019-1022``), so there is no
    previous value to put back and an empty state beside the closure is the honest
    record of that rather than an omission.

    ``pml`` is forwarded to :func:`repairable`, so a layer whose recurrence the repair
    cannot invert is refused HERE -- before the fused launch runs -- rather than at
    :func:`apply`, after the launch has already written a field nothing can put back.

    THE PATH IS STAMPED INTO THE RESULT, not passed again. :func:`apply` reads it from
    the saved dict, so the two halves cannot be told different things about which
    recurrence the state describes -- the same reason the cell set travels with it.

    ``prepared`` is a plan's :class:`_PreparedCells` (:class:`LeadingRepairPlan` owns
    one): the cell sets are built on its first call and reused. A caller that passes
    none -- the gates and probes that bracket a launch by hand -- gets a ONE-SHOT one,
    built for this call, so it takes the SAME route the product runs rather than a
    route of its own: every ``(component, source)`` is gathered through the linear index
    unless that entry's guard fails (an index :func:`_linear_index` will not spell, or
    an array :func:`_flat` will not view), and then through the 3-tuple, exactly as on
    a plan. Either way the object is stamped under :data:`_PREPARED_KEY` so
    :func:`apply` reads the same entries, and its counters record which route each entry
    took. The saved dict is therefore the ``(component, index, "cells" | "field" |
    "fw")`` entries it always held, :data:`_PATH_KEY`, :data:`_PREPARED_KEY` and
    :data:`_TARGETS_KEY`. ``field``/``fw`` hold the same cells in the same order
    whichever spelling of the index gathered them, and each is a new array, never a view
    the launch could overwrite: an integer-array index on the linear route,
    :func:`_gathered` on the 3-tuple one.

    ONLY THE TARGETS A SOURCE WRITES ARE CAPTURED (2026-09-20): per source,
    :func:`_source_targets` names them -- one for every engine source class -- or, when
    it cannot, every target of the seam, counted as a fallback. The selection is stamped
    under :data:`_TARGETS_KEY` so :func:`apply` restores exactly what was captured here.
    See the comment above :func:`_written_targets`.
    """
    deposits = []
    if pair in SEAMS:
        for index, source in enumerate(sources):
            idx = _deposit_index(source)
            if idx is not None:
                deposits.append((index, idx) + _source_targets(source, pair))
    # The per-target clauses answer for the targets about to be captured -- all of them
    # when nothing deposits, which keeps an empty seam refused exactly where it was.
    ok, reasons = repairable(
        fields, pair, pml, paths=paths,
        targets=(_union_in_seam_order(pair, [targets for _i, _x, targets, _r in deposits])
                 if deposits else None))
    if not ok:
        raise DepositNotRepairable("; ".join(reasons))
    path = _selected_path(pml, paths)
    xp = fields.grid.xp
    if prepared is None:
        # ONE-SHOT, never module-global: see the comment above `_CellEntry` (F3).
        prepared = _PreparedCells()
    stamped: Dict[int, Tuple[Any, ...]] = {}
    saved: Dict[Any, Any] = {_PATH_KEY: path, _PREPARED_KEY: prepared,
                             _TARGETS_KEY: stamped}
    for index, idx, targets, restricted in deposits:
        if restricted:
            prepared.restricted_sources += 1
        else:
            prepared.restriction_fallbacks += 1
        stamped[index] = targets
        for component, target, _axis in targets:
            entry = _prepared_entry(prepared, fields, component, target, index, idx)
            cells = entry.cells
            saved[(component, index, "cells")] = cells
            if path is PLAIN_PATH:
                continue  # a pure overwrite reads no previous value; see the docstring
            # RAW fetches, then GATHERS: this reads a handful of cells, and under a
            # held residency a barriered fetch would sync the whole volume first.
            field = host_writes.raw(fields, component)
            fw = host_writes.raw(fields, "f_w_" + component)
            views = (_flat_views(entry.shape, (field, fw))
                     if entry.linear is not None else None)
            if views is not None:
                saved[(component, index, "field")] = host_writes.gather(field, entry.linear)
                saved[(component, index, "fw")] = host_writes.gather(fw, entry.linear)
                prepared.linear_saves += 1
                continue
            saved[(component, index, "field")] = _gathered(xp, field, cells)
            saved[(component, index, "fw")] = _gathered(xp, fw, cells)
            prepared.cells_saves += 1
    return saved


def _apply_plain(fields: Any, pml: Any, sources: Sequence[Any], pair: str,
                 saved: Dict[Any, Any]) -> int:
    """The plain branch's repair: recompute the constitutive product and store it.

    ``stepping.update_E``'s inactive-layer arm, restricted to the saved cells: the
    constitutive product is GATHERED at the cells and then multiplied
    (``dmp(c)[cells] * inv_eps(c)[cells]``), the same correctly rounded product of the
    same two operands the whole-volume ``stepping.py:1010-1013`` then ``:993`` forms at
    each cell, so the stored words are identical (measured, 0 of 84,449,844). No
    ``f_w``, no ``kps``/``kms``, and no previous value -- the sub-step read none of them.

    THE BACKSTOP RUNS HERE TOO. Like :func:`apply`, this is the only place the
    arithmetic lands, so the plain clauses are asserted whatever the caller handed
    :func:`save`; refusing before the first write is the whole point.
    """
    work, missing = _repair_work(sources, pair, saved, ("cells",))
    reasons = _plain_path_reasons(fields, pml, pair,
                                  targets=[target for target, _found in work])
    if reasons:
        raise DepositNotRepairable("; ".join(reasons))
    if missing:
        raise DepositNotRepairable("; ".join(missing))
    prepared = saved.get(_PREPARED_KEY)
    repaired = 0
    for (component, _source_name, _axis), found in work:
        # THE SAME OPERANDS AS update_E's product, GATHERED FIRST. Every operand is at
        # the cell's own index (fields.py:1079-1105, a per-cell inv_eps volume), so the
        # product at a cell is the same float product whether the other cells were
        # formed or not; forming all of them cost a whole-volume multiply per component
        # per step (see the comment above `_CellEntry`). `displacement_minus_polarization`
        # is still called once per REPAIRED component, as update_E calls it, because on
        # a dispersive row it is what forms D - sum P in the shared scratch.
        inverse = fields.inverse_epsilon_for(component)
        field = host_writes.raw(fields, component)
        volume = None
        xp = fields.grid.xp
        for index in found:
            cells = saved[(component, index, "cells")]
            entry = _linear_entry(prepared, component, index, cells)
            if entry is not None and _flat(field, entry.shape) is not None:
                linear = entry.linear
                host_writes.scatter(
                    field, linear,
                    _fresh_at(fields, component, linear) * inverse.reshape(-1)[linear])
                prepared.linear_repairs += 1
                repaired += entry.count
                continue
            if volume is None:
                volume = fields.displacement_minus_polarization(component)
            _scatter_cells(xp, field, cells, _constitutive_at(volume, inverse, cells))
            if prepared is not None:
                prepared.cells_repairs += 1
            repaired += int(getattr(cells[0], "size", len(cells[0])))
    return repaired


def apply(fields: Any, pml: Any, sources: Sequence[Any], pair: str,
          saved: Dict[Any, Any]) -> int:
    """Recompute the constitutive result at the deposit points. Returns points repaired.

    Runs AFTER the driver's inject / symmetry fill / wall clear, so the injected field is
    already final; this only redoes the accumulation the fused launch performed against
    the pre-injection value.

    THAT SLOT IS REQUIRED, NOT MERELY CONVENIENT, and on a folded seam it was measured
    to be: the same extended repair run one line EARLIER -- after the injection but
    before the three fills -- reproduces the point repair's failure exactly, because
    the fills write D and B only, while E and f_w at a ghost are written by nothing but
    ``update_E`` and this call. A repair placed before the fill reads a D at the image
    that still holds the launch's pre-injection value and recomputes the same wrong
    answer. At this slot D at every image is already final, so no wall handling, no
    fill and no ordering among the images is needed.

    WHAT IS REPAIRED IS WHAT :func:`save` CAPTURED: per source, the targets stamped under
    :data:`_TARGETS_KEY` -- the one that source writes, or all of them on a fallback. A
    saved dict carrying no stamp is read as every target for every source.
    """
    from . import stepping  # noqa: PLC0415 - avoids a cycle at import time

    # WHICH REPAIR THE SAVED STATE DESCRIBES, read from the state itself. A dict without
    # the stamp came from a `save` predating the second repair, or from a caller that
    # built one by hand; either way the split-field default is what it meant, and that
    # is the path whose backstop then refuses anything it cannot invert.
    if saved.get(_PATH_KEY, SPLIT_FIELD_PATH) is PLAIN_PATH:
        return _apply_plain(fields, pml, sources, pair, saved)

    # WHAT IS REPAIRED IS WHAT WAS SAVED, target by target and source by source, read
    # from the state (`_TARGETS_KEY`) and never decided a second time here.
    work, missing = _repair_work(sources, pair, saved, ("cells", "field", "fw"))

    # THE BACKSTOP THAT NEEDS NO CALLER CHANGE. This function has always been handed the
    # layer, and it is the only place the repair's arithmetic actually lands, so the
    # absorber clauses are asserted here whatever the caller passed to `save` -- for
    # the targets about to be written, and for all of them when there are none.
    # Refusing before the first write is the whole point: a partial repair is worse
    # than none, which is also why missing state is refused here and not mid-loop.
    absorber = _absorber_reasons(fields, pml, pair,
                                 targets=[target for target, _found in work])
    if absorber:
        raise DepositNotRepairable("; ".join(absorber))
    if missing:
        raise DepositNotRepairable("; ".join(missing))

    spec = SEAMS[pair]
    xp = fields.grid.xp
    prepared = saved.get(_PREPARED_KEY)
    repaired = 0
    for (component, source_name, axis), found in work:
        # RAW, because every touch below is a gather or a scatter of the deposit
        # cells. The D seam's ``volume`` is formed AT THE CELLS by ``_fresh_at``
        # rather than over the whole grid, for the reason its docstring gives; the
        # whole-volume form is built only if a cells-route entry needs it.
        field = host_writes.raw(fields, component)
        fw = host_writes.raw(fields, "f_w_" + component)
        kps, kms = stepping._constitutive_coefficients(
            pml, axis, half_integer=spec["half_integer"])
        if pair == "D":
            volume = None
            inverse = fields.inverse_epsilon_for(component)
        else:
            volume = host_writes.raw(fields, source_name)
            inverse = None
        kp_full = km_full = None
        for index in found:
            cells = saved[(component, index, "cells")]
            field_old = saved[(component, index, "field")]
            fw_old = saved[(component, index, "fw")]
            entry = _linear_entry(prepared, component, index, cells)
            # THE LINEAR ROUTE IS NOW A GATHER, HOST ARITHMETIC, AND A SCATTER. The
            # arithmetic is byte-unchanged from the flat-view form it replaces --
            # same operands, same order, same dtype -- and only the deposit cells
            # cross the bus. The whole-volume acquire that stood here moved ~14
            # volumes a step on the 2026-09-22 ladder and is why the Metal route
            # lost to stock MEEP at every size.
            shape_ok = (entry is not None
                        and _flat(field, entry.shape) is not None
                        and _flat(fw, entry.shape) is not None)
            if shape_ok:
                linear = entry.linear
                if inverse is None:
                    fresh = host_writes.gather(volume, linear)
                else:
                    fresh = _fresh_at(fields, component, linear) * inverse.reshape(-1)[linear]
                kp_cells, km_cells = _coefficients_at(xp, entry, kps, kms)
                # SAME OPERAND ORDER as _apply_constitutive_pml: (f + kp*fw) - km*fw_prev.
                restored = field_old + kp_cells * fresh
                restored = restored - km_cells * fw_old
                host_writes.scatter(field, linear, restored)
                host_writes.scatter(fw, linear, fresh)
                prepared.linear_repairs += 1
                repaired += entry.count
                continue
            if volume is None:
                volume = fields.displacement_minus_polarization(component)
            if kp_full is None:
                kp_full = xp.broadcast_to(xp.asarray(kps), field.shape)
                km_full = xp.broadcast_to(xp.asarray(kms), field.shape)
            fresh = _constitutive_at(volume, inverse, cells)
            # SAME OPERAND ORDER as _apply_constitutive_pml: (f + kp*fw) - km*fw_prev.
            restored = field_old + kp_full[cells] * fresh
            restored = restored - km_full[cells] * fw_old
            _scatter_cells(xp, field, cells, restored)
            _scatter_cells(xp, fw, cells, fresh)
            if prepared is not None:
                prepared.cells_repairs += 1
            repaired += int(getattr(cells[0], "size", len(cells[0])))
    return repaired


# ---------------------------------------------------------------------------
# The shared source-presence clause, and the two-consult plans that earn it
# ---------------------------------------------------------------------------
#
# TWENTY-ONE PREDICATES ASKED THIS QUESTION IN TWENTY-ONE HAND-COPIED COPIES -- 11 Triton
# and 10 Metal -- each refusing any in-seam source outright. That was correct while no
# product could carry a deposit. Now that one can, the answer depends on something the
# copies had no way to express: whether THIS product brackets its launch with the repair.
# So the clause moves here, takes ``carries_repair`` from the caller, and the copies
# delegate. A product that has not been wired passes the default and keeps refusing.


def in_seam_sources(sources: Any, pair: str) -> Tuple[Any, ...]:
    """The subset of ``sources`` the driver injects inside ``pair``'s seam.

    ``driver.py:3293-3294`` injects the magnetic list between ``step_B`` and ``update_H``;
    ``:3318-3322`` injects the electric list between ``step_D`` and ``update_E``. A source
    on the other side of the step is not this seam's problem.
    """
    return tuple(source for _index, source in _in_seam_indexed(sources, pair))


def _in_seam_indexed(sources: Any, pair: str) -> Tuple[Tuple[int, Any], ...]:
    """In-seam sources paired with their index in the CALLER'S list, not in the subset.

    The hand-written clauses this replaced enumerated the whole list and reported that
    index, so "source 1" named the second source the run declared. Renumbering against the
    filtered subset would keep every existing test passing -- they mostly declare one
    source -- while making the diagnostic point at the wrong source on any mixed run.
    """
    selected = []
    for index, source in enumerate(tuple(sources or ())):
        magnetic = str(getattr(source, "field_type", "")) == MAGNETIC_FIELD_TYPE
        if (pair == "B") == magnetic:
            selected.append((index, source))
    return tuple(selected)


def seam_source_reasons(fields: Any, sources: Any, pair: str, *,
                        undeclared: str, refusal: Any,
                        carries_repair: bool = False,
                        pml: Any = _PML_UNSUPPLIED,
                        repair_paths: Sequence[str] = (SPLIT_FIELD_PATH,)
                        ) -> Tuple[str, ...]:
    """Why this configuration's sources bar the fusion, or ``()`` if they do not.

    THE DECISION LIVES HERE; THE PROSE STAYS AT THE SITE. ``undeclared`` is the site's own
    not-declared text and ``refusal(index, source)`` returns its own in-seam text, because
    those strings are pinned by that product's gate and its tests -- centralising the
    wording would silently reword twenty-one certified refusals. What is shared is the
    part that was duplicated and is now allowed to change: WHICH sources count as in-seam,
    and whether a deposit is carryable.

    ``carries_repair`` is the product's own declaration that it wraps its launch in
    :class:`LeadingRepairPlan` / :class:`TrailingRepairPlan`. IT IS NOT A HINT: a product
    that passes True without those wrappers computes the constitutive half against a
    pre-injection field and reports success, which is the exact failure this module exists
    to make impossible. The wiring and the flag change together or not at all.

    ``pml`` is forwarded to :func:`repairable`, which adds the absorber clauses. NO
    SHIPPED PREDICATE FORWARDS IT YET -- each has the layer in hand and passes it to its
    own coverage halves, but not to here -- so today those clauses are asserted at
    :func:`save` and :func:`apply` instead. That is sound rather than merely tolerable:
    every family declaring ``CARRIES_DEPOSIT_REPAIR`` already refuses an absent or
    inactive layer in its curl and constitutive halves (``metal_kernels/coverage.py:152``),
    so no row this admits can reach a plain-path absorber. Forwarding it here would state
    that once instead of relying on it twice, and belongs in the same change as the
    predicate sites.

    ``repair_paths`` is the product's declaration of WHICH repair its two slots install,
    and it travels with ``carries_repair`` for the same reason: both describe the wiring
    rather than request it. A product declaring :data:`PLAIN_PATH` must forward ``pml``
    as well -- :func:`repairable` refuses a plain-only declaration with no layer by name,
    because which recurrence ran cannot be established without one.
    """
    if sources is None:
        return (undeclared,)
    seam = _in_seam_indexed(sources, pair)
    if not seam:
        return ()
    if not carries_repair:
        return tuple(refusal(index, source) for index, source in seam)

    reasons = []
    ok, why = repairable(fields, pair, pml, paths=repair_paths)
    if not ok:
        reasons.extend(f"the deposit repair cannot carry this seam: {reason}"
                       for reason in why)
    for index, source in seam:
        if _deposit_index(source) is None:
            reasons.append(
                f"source {index} ({type(source).__name__}) does not publish the index it "
                f"writes, so its deposit cannot be saved and restored across the launch")
    return tuple(reasons)


class LeadingRepairPlan:
    """The fused launch, with the deposit captured immediately before it.

    Occupies the FIRST of the two slots a fused pair owns (``step_B`` / ``step_D``). The
    save has to happen here rather than at plan-build time: it reads the field state as of
    this step, and a plan is built once per configuration freeze.
    """

    __slots__ = ("inner", "pair", "absorbed_by", "_fields", "_pml", "_sources", "saved",
                 "repairs", "repair_paths", "prepared")

    def __init__(self, inner: Any, fields: Any, pml: Any, sources: Sequence[Any],
                 pair: str,
                 repair_paths: Sequence[str] = (SPLIT_FIELD_PATH,)) -> None:
        self.inner = inner
        self.pair = pair
        self.absorbed_by = inner
        self._fields = fields
        self._pml = pml
        self._sources = tuple(sources)
        self.saved: Optional[Dict[Any, Any]] = None
        self.repairs = 0
        #: The repairs the INSTALLING product carries. Defaults to the split-field one,
        #: so every caller written before the second repair keeps its exact behaviour;
        #: the layer still decides which of the declared paths actually runs.
        self.repair_paths = tuple(repair_paths)
        #: This plan's cell sets and their linear indices, built on the first save --
        #: after :func:`repairable` has admitted the configuration -- and reused every
        #: step. See the comment above :class:`_CellEntry`.
        self.prepared = _PreparedCells()

    def run(self, *args: Any, **kwargs: Any) -> None:
        # The layer goes with the save: this is the last moment a configuration the
        # repair cannot invert can be refused BEFORE the fused launch overwrites the
        # field it would have had to put back.
        self.saved = save(self._fields, self._sources, self.pair, self._pml,
                          paths=self.repair_paths, prepared=self.prepared)
        self.inner.run(*args, **kwargs)


class TrailingRepairPlan:
    """The repair, in the second of the two slots (``update_H`` / ``update_E``).

    Replaces ``NoopPlan`` for a seam that carries a deposit. The driver has injected,
    filled symmetry and cleared walls by the time this runs, so the field it reads is
    final and the repair needs no wall handling of its own.
    """

    __slots__ = ("slot", "absorbed_by", "_leading", "_fields", "_pml")

    def __init__(self, slot: str, leading: LeadingRepairPlan, fields: Any,
                 pml: Any) -> None:
        self.slot = slot
        self.absorbed_by = leading.inner
        self._leading = leading
        self._fields = fields
        self._pml = pml

    def run(self, *_args: Any, **_kwargs: Any) -> None:
        saved = self._leading.saved
        if saved is None:
            # The first consult did not run, so the fused launch did not either and
            # there is nothing to repair. Raising is right: a silent no-op here leaves
            # the constitutive half UNDONE, and the driver's contract says True meant
            # the whole sub-step ran.
            raise DepositNotRepairable(
                f"{self.slot}: the paired launch did not run, so no deposit was saved")
        self._leading.repairs = apply(self._fields, self._pml, self._leading._sources,
                                      self._leading.pair, saved)
        self._leading.saved = None
