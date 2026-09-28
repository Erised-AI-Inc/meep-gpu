"""
Leapfrog stepping for the MEEP-compatible FDTD engine — the four MEEP sub-steps
that advance the Yee lattice by one timestep, plus the mirror-plane repairs that
MEEP's owned-cell loop leaves to the boundary pass.

``step_B`` and ``step_D`` apply the discrete curl to the primary fields: forward
differences for B, and MEEP's negated strides (backward differences) for D, so
that a single ``field -= curl`` expression produces dB/dt = -curl(E) and
dD/dt = +curl(H). Where a PML layer is present the same pass runs the
split-field auxiliary recurrence. ``update_H`` and ``update_E`` close the
constitutive relations: without PML they are no-ops, because H = B and
E = D*inv_eps are served on demand by ``Fields``; with PML they accumulate the
stored fields through the third PML direction. ``fill_symmetry_bc_D`` is
exported for the driver to call after source injection, matching MEEP's
step_db -> step_source -> step_boundaries order; the B-side counterpart runs at
the end of ``step_B``.

Inputs are a ``Fields`` container (which owns the arrays, the ``Grid``, and the
on-demand E/H views) and an optional ``PML`` coefficient table; every entry
point mutates field state in place and returns nothing. The driver's
per-timestep order is step_B, update_H, step_D, inject sources,
fill_symmetry_bc_D, update_E.

An instantaneous nonlinearity — ``mp.Medium(chi2=..., chi3=...)`` — lives entirely
inside ``update_E``, and is deliberately NOT built like the dispersion it sits
beside: no auxiliary field, no ODE, no history, nothing per-step to advance. It is
a pointwise Pade factor on the constitutive product (:func:`calc_nonlinear_u`,
transcribed from MEEP's ``calc_nonlinear_u``), applied where E is recovered from
D — which is the same function that threads the PML auxiliary, so it composes with
PML, dispersion and diagonal epsilon without any of the three knowing about it.
Its one non-local ingredient is MEEP's ``Dsqr``, which averages the two TRANSVERSE
D components onto the updated component's Yee position and therefore reads
neighbouring cells under the same boundary rules the curl obeys.

Array work goes through ``grid.xp``, so this module runs unchanged on NumPy (the
reference and test path) and CuPy (the accelerated path); it imports neither.
Fields are float32 or complex64, while ``inv_eps`` and every PML coefficient are
float32. Units are MEEP natural units (c = 1, frequency = 1/wavelength), so
dtdx = dt/dx is the Courant number and the curl carries no other scale factor.

Every kernel here is dtype-agnostic on purpose, and that is a statement about the
physics rather than about NumPy's casting rules: the curl, the constitutive
relations, the split-field PML recurrence and the polarization ADE all have real
coefficients, so nothing in the loop ever needs the imaginary plane. A real-field
run therefore does not approximate the complex one, it reproduces its real part
bit for bit — the two differ only where an imaginary input enters, which is a
Bloch phase (refused in :func:`_bloch_phases`) or a complex source amplitude
(refused in ``sources``). Keeping the coefficient arrays float32 in BOTH modes is
what makes that true: a complex64 coefficient would multiply the real part by the
imaginary part of the field and the equivalence would be gone.

Boundary conditions. There are exactly three, they are chosen per axis, and a PML
decides none of them:
- mirror, on any folded axis — X, Y or Z alike: a zero ghost at the far face, a
  parity-weighted mirror ghost ``parity * f[2]`` at the near face, and ownership
  masking of cell 0 for every component whose Yee shift is 0 on that axis. The
  parity is ``fields.mirror_parity``, which carries the plane's declared phase, so
  an odd mirror (``phase = -1``) reaches these kernels as the same rule with every
  sign inverted rather than as a second code path;
- metallic, on an unfolded axis declared so (``Grid.is_metallic``): a perfect
  electric conductor on both faces. A zero ghost on EITHER side, because there is no
  field beyond a perfect conductor, the same cell-0 masking a fold uses, and
  ``zero_metal_B`` / ``zero_metal_D`` — which the driver runs after the magnetic and
  the electric currents respectively — holding the wall plane itself at zero against
  anything a source writes there;
- periodic wrap, on EVERY other axis, absorber or not — plain at k = 0, and
  Bloch-periodic when ``grid.k_point`` gives that axis a phase, in which case the
  wrapped value is multiplied by ``grid.bloch_phase(axis)`` going up and by its
  conjugate going down.

A REDUCED-DIMENSION run adds no fourth rule, and that is the point rather than an
omission. MEEP 2-D and 1-D are translational invariance along the missing axes: MEEP
leaves those directions out of ``LOOP_OVER_DIRECTIONS``, which sets ``stride(d) = 0``
(vec.cpp ``num_changed``), so ``g1[i + s] - g1[i]`` reads the same sample twice and
the difference is an exact zero rather than a small one. This engine gives such an
axis ONE cell and the ordinary periodic wrap, where ``roll`` returns the same plane
and the subtraction is the same exact zero — no branch, no special case, and nothing
added to the per-timestep loop. That equivalence is MEEP's own: a one-pixel periodic
direction is what ``fields::nosize_direction`` calls a lower-dimensional emulation,
and MEEP's genuine 2-D run and the same problem as a 3-D cell one pixel deep in z
step to **0.0e+00** of each other.

What the invariant axis must NOT be is metallic, and ``Grid`` refuses it there rather
than here. A PEC wall zeroes exactly the samples whose Yee shift on the walled axis is
0, which on the single stored cell of an invariant z is Ex, Ey and Hz — the entire TE
polarization — so the run returns a smooth exact zero for one polarization while the
other stays exact.

That is MEEP's own arrangement, including which one is reached when. MEEP's
``fields::fields`` starts every face at ``Metallic`` (fields.cpp) and
``fields::use_bloch`` (boundaries.cpp) overwrites the whole axis with ``Periodic``,
which ``Simulation`` calls for every direction as soon as a ``k_point`` is given —
including ``k_point=mp.Vector3()``. Neither consults an absorber: a PML is a
material, and ``structure_chunk::use_pml`` grades a conductivity and leaves the
chunk connection as it found it. So a wave leaving an absorbing PERIODIC face
re-enters at the opposite one in MEEP too; the absorber, not the boundary, is what
stops it. A metallic face reflects it instead, which is why the two cannot be
substituted for one another: measured 1.28e+00 complex relative L2 apart on the same
cell.

A metallic wall is implemented as MEEP implements it, which is not as a ghost rule
alone. ``on_metal_boundary`` (boundaries.cpp) picks out the samples whose doubled
coordinate lands exactly on the little or big corner of the cell — precisely the
components whose Yee shift is 0 on that axis, i.e. tangential E/D and normal H/B —
``find_metals`` collects pointers to them, and ``step_boundaries`` zeroes them before
every sub-step's communication. Everything else on the axis sits half a cell off the
wall and is free. This engine stores one wall plane per metallic axis (cell 0, the
low wall) and supplies the other as the zero ghost of ``_shift_up``, so the two walls
are complete without MEEP's extra ``num + 1`` slot.

Terminating an absorbing X or Y metallically instead — which this pass used to do
whenever both faces of the axis carried a layer — puts a perfect mirror where MEEP
has a wrap. It is invisible while the layer is thick enough to swallow the field
before it reaches the wall, and costs orders of magnitude when it is not: measured
against CPU MEEP on the uniform-PML sheet case, complex Ez relative L2 6.31e-04
with the metallic termination against 4.00e-07 with the wrap, on the same run at
the same resolution.

A Bloch phase only means anything on an axis that actually repeats, so a nonzero k
is refused on a mirrored axis (whose plane forces the field even or odd, i.e.
k = 0) AND on any axis carrying an absorber, even though that axis still wraps: the
layer terminates the wave before it reaches the face, so the phase would describe a
periodicity the run does not have. Bloch also needs complex storage — a real field
cannot carry the phase — and that too is refused loudly rather than silently
dropping the imaginary part.

MEEP source references: step_db.cpp (fields_chunk::step_db and its stride
negation), step_generic.cpp (step_curl, step_update_EDHB), step.cpp (update_eh),
boundaries.cpp (step_boundaries, use_bloch, locate_point_in_user_volume),
symmetry.cpp (mirror transforms).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Sequence, Tuple

from . import host_writes
from .fields import IYEE_SHIFTS, mirror_parity

BlochPhases = Tuple[Optional[complex], Optional[complex], Optional[complex]]
MirrorPhases = Tuple[Optional[int], Optional[int], Optional[int]]
FarReflectRows = Tuple[Optional[int], Optional[int], Optional[int]]

if TYPE_CHECKING:  # Annotations only; the runtime coupling is duck-typed on the arrays.
    from .fields import Fields
    from .grid import Grid
    from .pml import PML

AXIS_X, AXIS_Y, AXIS_Z = 0, 1, 2
AXIS_NAMES = ("x", "y", "z")

# MEEP symmetry.cpp: the transform runs on doubled coordinates, so with the
# halved grid's origin at io = -2 the ghost at index -1 maps onto stored cell 2
# for the iyee=1 components that the backward differences read.
MIRROR_SOURCE_INDEX = 2

# Ghost-cell rules for one axis. PERIODIC wraps; MIRROR is the symmetry plane —
# zero at the far face (forward differences) and phase * f[2] at the near face;
# METALLIC is a perfect electric conductor on both faces — zero on either side,
# because there is no field beyond a perfect conductor.
# There is deliberately no absorbing rule: a PML is a material, not a boundary
# condition, so an axis carrying one still wraps (see the module docstring).
PERIODIC = "periodic"
MIRROR = "mirror"
METALLIC = "metallic"
# The cylindrical radial axis (grid.is_axis): the far face is a metallic wall
# (zero ghost), the near face the r_to_minus_r image — the field's own value at
# +dr/2 with R- and P-direction components sign-flipped and an overall (-1)^m
# (MEEP installs r_to_minus_r_symmetry(m) for exactly these ghosts). Every array
# the shift helpers move has Yee shift 1 on the shifted axis, so the below-axis
# ghost at -dr/2 always reflects stored row 0.
CYL_AXIS = "axis"

# The wall samples a metallic axis holds at exactly zero: the components whose Yee
# shift on that axis is 0 sit ON the boundary plane, and MEEP's step_boundaries
# zeroes them at every sub-step (boundaries.cpp find_metals / zero_metal). Everything
# else on the axis is half a cell off the plane and free.
B_COMPONENTS = ("Bx", "By", "Bz")
D_COMPONENTS = ("Dx", "Dy", "Dz")
H_COMPONENTS = ("Hx", "Hy", "Hz")
E_COMPONENTS = ("Ex", "Ey", "Ez")


class CurlTerm(NamedTuple):
    """One component's curl stencil plus the metadata its boundary and PML rules need."""

    target: str  # Primary field component this curl updates.
    first: str  # Component differenced with the leading stride (MEEP's g1).
    first_axis: int
    second: str  # Component differenced with the trailing stride (MEEP's g2).
    second_axis: int
    dsig: str  # PML cycle direction driving the auxiliary update.
    dsigu: str  # PML cycle direction driving the field update.

    @property
    def iyee(self) -> Tuple[int, int, int]:
        """Target's Yee shifts (vec.hpp iyee_shift); 0 on an axis => cell 0 unowned there.

        Read from ``fields.IYEE_SHIFTS`` rather than restated on each term: the two
        used to be separate transcriptions of the same six triples, and a mirror on
        a third axis would have needed the second one extended in lockstep.
        """
        return IYEE_SHIFTS[self.target]


# MEEP step_generic.cpp: f -= dtdx * (g1[i+s1] - g1[i] + g2[i] - g2[i+s2]).
# The PML direction cycle is vec.hpp cycle_direction, X->Y->Z: the component's
# own axis picks dsig (next) and dsigu (next again).
B_CURL_TERMS: Tuple[CurlTerm, ...] = (
    CurlTerm("Bx", "Ez", AXIS_Y, "Ey", AXIS_Z, "y", "z"),  # curl_x = dEz/dy - dEy/dz
    CurlTerm("By", "Ex", AXIS_Z, "Ez", AXIS_X, "z", "x"),  # curl_y = dEx/dz - dEz/dx
    CurlTerm("Bz", "Ey", AXIS_X, "Ex", AXIS_Y, "x", "y"),  # curl_z = dEy/dx - dEx/dy
)
D_CURL_TERMS: Tuple[CurlTerm, ...] = (
    CurlTerm("Dx", "Hz", AXIS_Y, "Hy", AXIS_Z, "y", "z"),  # curl_x = dHz/dy - dHy/dz
    CurlTerm("Dy", "Hx", AXIS_Z, "Hz", AXIS_X, "z", "x"),  # curl_y = dHx/dz - dHz/dx
    CurlTerm("Dz", "Hy", AXIS_X, "Hx", AXIS_Y, "x", "y"),  # curl_z = dHy/dx - dHx/dy
)

# update_eh (step_generic.cpp step_update_EDHB): dsigw is the component's own
# axis. H reads the integer-position coefficients, E the half-integer ones.
H_CONSTITUTIVE_TERMS = (("Hx", "Bx", "x"), ("Hy", "By", "y"), ("Hz", "Bz", "z"))
E_CONSTITUTIVE_TERMS = (("Ex", "Dx", "x"), ("Ey", "Dy", "y"), ("Ez", "Dz", "z"))

# Storage each PML sub-step reads or accumulates into; the driver allocates it
# with Fields.enable_pml_storage() before the first step.
B_PML_ARRAYS = ("Ex", "Ey", "Ez", "fu_Bx", "fu_By", "fu_Bz")
D_PML_ARRAYS = ("Hx", "Hy", "Hz", "fu_Dx", "fu_Dy", "fu_Dz")
H_PML_ARRAYS = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
E_PML_ARRAYS = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")

# Stored E is needed by dispersion as well as by PML; the constitutive sub-step
# asserts it rather than allocating, because allocation also switches get_E's mode.
E_STORAGE_ARRAYS = ("Ex", "Ey", "Ez")

__all__ = [
    "step_B", "step_D", "update_H", "update_E", "update_P",
    "fill_symmetry_bc_B", "fill_symmetry_bc_D",
    "zero_metal_B", "zero_metal_D", "fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D",
    "calc_nonlinear_u", "nonlinear_margin", "NonlinearMargin",
]


class NonlinearMargin(NamedTuple):
    """How far a nonlinear run is from the point where MEEP's Pade factor stops meaning anything.

    ``denominator`` is the smallest ``1 + 2*c2 + 3*c3`` anywhere on the grid and
    ``expansion`` the largest ``|c2| + |c3|``, both over every component carrying a
    chi2 or chi3; ``component`` names the one that produced ``expansion``.
    """

    component: str
    denominator: float
    expansion: float


def _acquire_written(fields: "Fields", names: Sequence[str]) -> None:
    """Declare the WHOLE-volume host writes an array-path sub-step is about to make.

    A no-op unless an execution backend is holding arrays. When one is, this is the
    path a held run takes only under a route gate's CONTROL -- the "raised" control
    forces the dispatch consult off so the array-path sub-step runs on the host, and
    the in-seam mutation replaces a wall wipe with a host write -- or under a genuine
    mid-run fallback. Those passes write whole volumes, so the whole-volume
    ``acquire`` is exactly right for them: it syncs the device's words down, unseals,
    and marks the mirror host-owned so the next launch syncs the result back up. The
    2026-09-23 preflight raised at ``fw[...] = source`` in ``_apply_constitutive_pml``
    on every held case's controls until this existed. Names that are not allocated
    on this run, or are not mirrored, cost nothing.
    """
    if not host_writes.holding():
        return
    for name in names:
        array = host_writes.raw(fields, name, None)
        if array is not None and hasattr(array, "shape"):
            host_writes.acquire(array)


def _aux_names(components: Sequence[str], prefixes: Sequence[str]) -> Tuple[str, ...]:
    return tuple(f"{prefix}{c}" for c in components for prefix in prefixes)


def step_B(fields: "Fields", pml: Optional["PML"] = None) -> None:  # Curl sub-step for B (forward differences).
    """Advance B by one timestep: B -= dt * curl(E).

    MEEP source: step_db.cpp (fields_chunk::step_db, B_stuff), step_generic.cpp
    (step_curl). Forward differences throughout — MEEP negates the strides only
    for D_stuff — so each component reads its two E neighbours one cell up:

        Bx: curl_x = dEz/dy - dEy/dz
        By: curl_y = dEx/dz - dEz/dx
        Bz: curl_z = dEy/dx - dEx/dy

    With PML the curl feeds the split-field recurrence at half-integer positions
    instead of being subtracted directly, and with a B conductivity — which in
    practice means an ``mp.Absorber``, the only thing that sets one — through the
    same three-stage ``f_cond`` recurrence the D side uses. Both go through
    :func:`_apply_curl`. This does NOT repair the mirror-plane
    cells: both symmetry fills, and both metallic-wall passes, are deferred to the
    driver, because MEEP runs ``step_boundaries`` after ``step_source`` on each
    side alike — ``step_db(B_stuff)`` then ``step_source(B_stuff)`` then
    ``step_boundaries(B_stuff)`` (step.cpp:67-72), the D half repeating it verbatim
    at :95-103 — and a current injected into a cell the boundary pass images or
    wipes has to land before that pass, not after.
    """
    _acquire_written(fields, B_COMPONENTS + _aux_names(B_COMPONENTS, ("fu_", "f_cond_", "f_bfast_")))
    grid = fields.grid
    scratch = fields.scratch
    dtdx = grid.dt / grid.dx  # MEEP: Courant = dt * gv.a = dt / dx.
    pml_active = _pml_is_active(pml)
    if pml_active:
        _require_pml_storage(fields, B_PML_ARRAYS)
    boundaries = _boundary_kinds(grid, pml if pml_active else None)
    parities = _mirror_phases(grid)
    reflect_rows = _far_reflect_rows(grid)
    electric = _component_snapshot(fields, ("Ex", "Ey", "Ez"))
    phases = _bloch_phases(grid, boundaries, electric["Ex"], pml if pml_active else None)
    cylindrical = getattr(grid, "cylindrical", False)
    axis_increment = None
    if cylindrical:
        _require_cylindrical_steppable(grid)
        # MEEP step_db.cpp:92-119, the B-side Z case: Bz's radial difference is of
        # the prefix sum of (1/r)d(r*Ep)/dr. Ep sits at the node (r-shift 0), so
        # ir0 = 0.0 where the D side's Hp gave 0.5.
        #
        # THE PREFIX IS A CUMSUM, NOT A FIELD, so the generic shift machinery's
        # metallic far ghost is wrong for it: a zero ghost turns the last row's
        # forward difference into MINUS THE WHOLE ACCUMULATED SUM. MEEP stores
        # the wall row (nr + 1 rows) and its Bz reads prefix[wall] - prefix[last],
        # with the wall's Ep held at zero by zero_metal — a LOCAL value. The same
        # thing here: the prefix runs over Ep extended by its zero wall row, and
        # Bz's radial difference is taken of consecutive extended rows directly.
        # Measured before this: the m = 1 run's first divergence was born at the
        # LAST TWO ROWS at step 3 (rows 0-17 exactly zero error) and grew inward,
        # with a spurious Hz where MEEP's cancels to 5e-11. The D side never
        # needs the wall row: its backward difference reads rows i and i-1.
        xp = grid.xp
        # Ep extended by its zero wall row. A pooled buffer rather than
        # zeros_like + concatenate, which allocated the whole volume again every
        # step; the two writes below move exactly the bytes concatenate moved.
        #
        # The ``scratch is None`` branch exists so the POOL CAN BE TURNED OFF here as
        # it can everywhere else on the step path: that switch is what
        # test_step_scratch_is_bit_identical differences against, and without it the
        # cylindrical path — which holds `prefix_ext`, the one pooled buffer carried
        # live across a whole term loop — could not be pinned by that test at all.
        # Only the allocation differs between the branches; the two writes below are
        # shared, so the comparison cannot drift as either side is edited.
        rows = electric["Ey"].shape[0]
        extended_shape = (rows + 1,) + electric["Ey"].shape[1:]
        extended = (xp.empty(extended_shape, dtype=electric["Ey"].dtype)
                    if scratch is None
                    else scratch.take("cyl_extended", extended_shape, electric["Ey"].dtype))
        extended[_span(0, 0, rows)] = electric["Ey"]
        extended[_face(0, rows)] = 0
        prefix_ext = cylindrical_rderiv_prefix(xp, extended, 0.0, scratch=scratch)
        axis_increment = _cylindrical_axis_increment_B(fields, electric, boundaries,
                                                       phases, dtdx)
    bfast = grid.bfast_active
    for term in B_CURL_TERMS:
        term_sources = electric
        operands = _curl_operands(grid.xp, term_sources, term, boundaries, phases, parities,
                                  backward=False, reflect_rows=reflect_rows,
                                  scratch=scratch)
        curl = _curl_from_operands(operands, dtdx, scratch=scratch)
        if cylindrical and term.target == "Bz":
            # Bz's whole curl in cylindrical: the phi-derivative partner is the
            # invariant-axis zero (the i*m/r part is added below), so the curl IS
            # the radial forward difference of the extended prefix.
            curl = dtdx * (prefix_ext[1:] - prefix_ext[:-1])
        if cylindrical and grid.m != 0:
            # step_db.cpp:177-294 — the B-side i*m/r couplings: Br takes its
            # dropped plus partner Ez, Bz its dropped minus partner Er; the_m's
            # (1-2*(ft==B_stuff)) flips both signs relative to the D side.
            if term.target == "Bx":
                curl = curl + _cylindrical_imr_term(fields, "Bx", electric["Ez"], +1.0)
            elif term.target == "Bz":
                curl = curl + _cylindrical_imr_term(fields, "Bz", electric["Ex"], -1.0)
        if grid.beta != 0.0:
            # step_db.cpp:148-176 — the B-side out-of-plane couplings. `cc` runs over
            # d_c in {X, Y} only and its partner is `direction_component(Ex, other)`:
            # Bx <- Ey at sign +1, By <- Ex at sign -1. Bz gets nothing.
            if term.target == "Bx":
                curl = curl + _special_kz_beta_term(fields, electric["Ey"], +1.0, magnetic=True)
            elif term.target == "By":
                curl = curl + _special_kz_beta_term(fields, electric["Ex"], -1.0, magnetic=True)
        if bfast:
            # step_db.cpp:129-142 — the second additive pass, on the SAME shifted
            # operands, the SAME loop bounds and the SAME PML/conductivity slices as
            # the curl above, which is why it folds in here rather than running again.
            curl = curl + _bfast_term(fields, term, operands, magnetic=True)
        _mask_non_owned_cells(curl, grid, term.iyee)
        if axis_increment is not None and axis_increment[0] == term.target:
            # The axis-row dfcnd, in curl sign convention (the step subtracts).
            curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)
        # B sits at half-integer positions in the transverse directions.
        _apply_curl(fields, term, curl, pml, pml_active, half_integer=True, scratch=scratch)
    if cylindrical:
        _cylindrical_axis_zero_B(fields)


def step_D(fields: "Fields", pml: Optional["PML"] = None) -> None:  # Curl sub-step for D (negated strides).
    """Advance D by one timestep: D += dt * curl(H).

    MEEP source: step_db.cpp, step_generic.cpp (step_curl).

    MEEP negates the strides for D_stuff::

        if (ft == D_stuff) { stride_p = -stride_p; stride_m = -stride_m; }

    so this sub-step differences the H neighbours one cell *down* while keeping
    the shared ``field -= curl`` form. The double negative is what turns it into
    dD/dt = +curl(H); the sign lives in the strides, not in the update.

        Dx: curl_x = dHz/dy - dHy/dz
        Dy: curl_y = dHx/dz - dHz/dx
        Dz: curl_z = dHy/dx - dHx/dy

    Unlike step_B this does not repair the mirror-plane cells: MEEP's order is
    step_db -> step_source -> step_boundaries, so the driver must call
    ``fill_symmetry_bc_D`` after injecting sources and before update_E.
    """
    _acquire_written(fields, D_COMPONENTS + _aux_names(D_COMPONENTS, ("fu_", "f_cond_", "f_bfast_")))
    grid = fields.grid
    scratch = fields.scratch
    dtdx = grid.dt / grid.dx
    pml_active = _pml_is_active(pml)
    if pml_active:
        _require_pml_storage(fields, D_PML_ARRAYS)
    boundaries = _boundary_kinds(grid, pml if pml_active else None)
    parities = _mirror_phases(grid)
    magnetic = _component_snapshot(fields, ("Hx", "Hy", "Hz"))
    phases = _bloch_phases(grid, boundaries, magnetic["Hx"], pml if pml_active else None)
    cylindrical = getattr(grid, "cylindrical", False)
    axis_increment = None
    if cylindrical:
        _require_cylindrical_steppable(grid)
        # MEEP step_db.cpp:92-119 (Z case): Dz's radial difference is taken of the
        # prefix sum of (1/r)d(r*Hp)/dr, not of Hp itself; the unmodified curl then
        # produces the cylindrical derivative. ir0 = 0.5 = half of Hp's r-shift.
        prefixed = dict(magnetic)
        prefixed["Hy"] = cylindrical_rderiv_prefix(grid.xp, magnetic["Hy"], 0.5,
                                                   scratch=scratch)
        axis_increment = _cylindrical_axis_increment_D(fields, magnetic, boundaries,
                                                       phases, dtdx)
    bfast = grid.bfast_active
    for term in D_CURL_TERMS:
        term_sources = magnetic
        if cylindrical and term.target == "Dz":
            term_sources = prefixed
        operands = _curl_operands(grid.xp, term_sources, term, boundaries, phases, parities,
                                  backward=True, scratch=scratch)
        curl = _curl_from_operands(operands, dtdx, scratch=scratch)
        if cylindrical and grid.m != 0:
            # step_db.cpp:177-294 — the D-side i*m/r couplings: Dr takes +im/r*Hz
            # (its dropped plus partner), Dz takes -im/r*Hr (its dropped minus
            # partner); signs from the_m's (1-2*(d_c==R)) with ft = D.
            if term.target == "Dx":
                curl = curl + _cylindrical_imr_term(fields, "Dx", magnetic["Hz"], -1.0)
            elif term.target == "Dz":
                curl = curl + _cylindrical_imr_term(fields, "Dz", magnetic["Hx"], +1.0)
        if grid.beta != 0.0:
            # step_db.cpp:148-176 — the D-side out-of-plane couplings. `cc` runs over
            # d_c in {X, Y} only and its partner is `direction_component(Hx, other)`:
            # Dx <- Hy at sign +1, Dy <- Hx at sign -1. Dz gets nothing.
            if term.target == "Dx":
                curl = curl + _special_kz_beta_term(fields, magnetic["Hy"], +1.0, magnetic=False)
            elif term.target == "Dy":
                curl = curl + _special_kz_beta_term(fields, magnetic["Hx"], -1.0, magnetic=False)
        if bfast:
            # step_db.cpp:129-142, D side: the same pass with k1 and k2 both negated,
            # which is what turns +d/dt(k x E) on the B side into -d/dt(k x H) here.
            curl = curl + _bfast_term(fields, term, operands, magnetic=False)
        _mask_non_owned_cells(curl, grid, term.iyee)
        if axis_increment is not None and axis_increment[0] == term.target:
            # The axis-row dfcnd, in curl sign convention (the step subtracts).
            curl[_face(0, 0)] = -axis_increment[1]
        # D sits at integer positions in the transverse directions.
        _apply_curl(fields, term, curl, pml, pml_active, half_integer=False, scratch=scratch)
    if cylindrical:
        _cylindrical_axis_zero_D(fields)


def _apply_curl(fields: "Fields", term: Any, curl: Any, pml: Optional["PML"],
                pml_active: bool, half_integer: bool,
                scratch: Any = None) -> None:  # The shared D/B curl application.
    """Subtract one curl from its target through whichever of MEEP's four paths applies.

    ONE function for D and B because ``step_generic.cpp``'s ``step_curl`` is one
    function for both: the field type reaches it only through the caller's negated
    strides and through which sigma slice the PML coefficients come from, never
    through the arithmetic. The conductivity is likewise a single expression on both
    sides — same dt, same ``condinv = 1/(1 + sigma*dt/2)`` (structure.cpp:697-699),
    no half-step offset — which is why the B side needed no new recurrence, only the
    branches.

    ``half_integer`` selects the PML sigma sub-lattice: B sits at half-integer
    transverse positions and D at integer ones. Conductivity is read PER COMPONENT
    (``fields.condfac_for``), so a component with no sigma takes the plain path even
    while its neighbours are lossy — which is MEEP's own allocation granularity.
    """
    target = getattr(fields, term.target)
    condfac = fields.condfac_for(term.target)
    if pml_active:
        kms, sinv = _curl_coefficients(pml, term.dsig, half_integer=half_integer)
        kms_u, sinv_u = _curl_coefficients(pml, term.dsigu, half_integer=half_integer)
        fu = getattr(fields, "fu_" + term.target)
        if condfac is not None:
            f_cond = getattr(fields, "f_cond_" + term.target)
            if f_cond is None:
                raise RuntimeError(
                    "Conductivity + PML stepping requires Fields to allocate "
                    f"f_cond_{term.target} before the first step."
                )
            _apply_conductive_pml_update(
                fields.grid.xp,
                target,
                curl,
                condfac,
                fields.condinv_for(term.target),
                kms,
                sinv,
                kms_u,
                sinv_u,
                fu,
                f_cond,
                scratch=scratch,
            )
        else:
            _apply_pml_update(target, curl, kms, sinv, kms_u, sinv_u, fu, scratch=scratch)
    elif condfac is not None:
        _apply_conductive_update(target, curl, condfac, fields.condinv_for(term.target))
    else:
        target -= curl


def _require_cylindrical_steppable(grid: "Grid") -> None:  # One home for future cyl gates.
    """Refuse cylindrical configurations the stepper does not carry.

    Currently none: m = 0, |m| = 1 (their axis rules) and |m| > 1 (the near-axis
    zeroing) are all stepped. The function remains as the single place a future
    gate belongs, so a new unstepped configuration is refused by name here and
    not discovered as a smooth wrong field.
    """
    del grid


def _cylindrical_axis_increment_D(fields: "Fields", magnetic: Dict[str, Any],
                                  boundaries: Tuple[str, str, str],
                                  phases: Any, dtdx: float):
    """The r = 0 axis-row increment for the D sub-step — (target, dfcnd row) or None.

    step_db.cpp:299-341 (m = 0) and :347-397 (|m| = 1). m = 0: the on-axis Dz
    takes the analytic limit of (1/r)d(r*Hp)/dr — 4*Courant*Hp at the first
    off-axis Hp site. |m| = 1: Dp on the axis takes d(Hr)/dz - 2*Hz (the factor
    2 is the doubled-ivec compensation — Hz sits half a cell off the axis).

    The increment is FOLDED INTO THE CURL ARRAY at row 0 by the caller rather
    than added to the field afterwards, for the same reason the i*m/r pass is:
    MEEP's axis loop is the single kap/sig-laddered recurrence over that row
    (its main loop never visits it), and folding the dfcnd into the curl makes
    our one `_apply_pml_update` pass exactly that recurrence with the right
    coefficient positions. The plain `+=` this used to be was exact only while
    every live ladder axis carried zero sigma on the axis row — true for m = 0
    (dsig = R, sigma(r=0) = 0) and FALSE for |m| = 1, whose Dp ladders run along
    Z and R: measured with r+z PML, Ep 4.6e-01 and Hr 2.39 against an exact m=0.
    """
    grid = fields.grid
    if grid.m == 0:
        # NOT folded into the curl: measured, the fold breaks m=0 under PML
        # (Er 2.7e-01 / Hp 4.5e-01 where the post-add is 1.7e-3.6e-07). The
        # split-field routing (curl -> fu -> field) is not MEEP's axis ladder
        # for this rule; the plain post-add is exact because Dz's dsig is R,
        # whose sigma is zero on the axis row. Applied in _cylindrical_axis_zero_D.
        return None
    if abs(grid.m) == 1:
        hr = magnetic["Hx"]
        hr_below = _shift_down(grid.xp, hr, 2, boundaries[2], "Hx", phases[2], None)
        return "Dy", dtdx * (hr[_face(0, 0)] - hr_below[_face(0, 0)]
                             - 2.0 * magnetic["Hz"][_face(0, 0)])
    return None


def _cylindrical_axis_zero_D(fields: "Fields") -> None:
    """The D-side axis-row zero constraints, applied after the field update.

    m = 0: Dp is identically zero on the axis (an azimuthal vector at r = 0 has
    no direction to point). |m| = 1: Dz is zeroed there instead. |m| > 1 — MEEP's
    default zero_fields_near_cylorigin behaviour (step_db.cpp:398-434): ALL
    THREE D components held at zero on every row within |m| pixels of the axis,
    together with their fu/f_cond auxiliaries where allocated. MEEP's own
    comment calls it a stability hack — without it the run needs
    Courant <~ 1/(|m| + 0.5); with it, <~ 0.62 for all m — and notes it
    "probably spoils 2nd-order accuracy" near the axis. Reproduced because the
    reference behaviour is the contract, not because it is pretty.

    The set of components is the POST-#3164 one (upstream 593a4b42, "Fix bug in
    zero_fields_near_cylorigin for |m| >= 2", first released in 1.33.0): the
    1.29-era code zeroed only Dp and Dz here, and the two references genuinely
    differ — measured MEEP-vs-MEEP on the m=2 oracle config, pristine 1.33 vs
    1.29 is 3.0e-02..1.7e-01. This engine follows the fix; the m=2 oracle test
    is version-aware about which reference it is stepping against.
    """
    grid = fields.grid
    if grid.m == 0:
        # ``get_H("Hy")`` is the stored H under PML and the B array otherwise --
        # its own branch, reproduced here on RAW fetches so the face read does not
        # sync a whole volume. The correction is a face read-modify-write with a
        # float scale: ``k * Hy`` then ``+=``, two roundings on both sides.
        hy = host_writes.raw(fields, "Hy" if fields._pml_active else "By")
        host_writes.add_scaled_face(host_writes.raw(fields, "Dz"), 0, 0, hy, 0,
                                    4.0 * (grid.dt / grid.dx))
        host_writes.zero(host_writes.raw(fields, "Dy"), 0, 0)
    elif abs(grid.m) == 1:
        host_writes.zero(host_writes.raw(fields, "Dz"), 0, 0)
    else:
        rows = _cylindrical_axis_rows(grid)
        for name in ("Dx", "Dy", "Dz",
                     "fu_Dx", "fu_Dy", "fu_Dz", "f_cond_Dx", "f_cond_Dy", "f_cond_Dz"):
            aux = host_writes.raw(fields, name)
            if aux is not None:
                host_writes.zero_rows(aux, 0, rows)


def _cylindrical_axis_rows(grid) -> slice:
    """Which r rows the |m| >= 2 near-axis constraint holds at zero.

    MEEP's two branches, and the flag that chooses between them
    (``fields::zero_fields_near_cylorigin``, the inverse of the user-facing
    ``accurate_fields_near_cylorigin``, python/simulation.py:2483):

    * DEFAULT — every row within ``rmax = |m| - int(origin_r*a + 0.5)`` pixels of the
      axis. This grid's r origin is the axis, so ``rmax`` is ``|m|``. MEEP calls this
      David's hack and its own comment says it "probably spoils 2nd-order accuracy";
      it buys a stable run at Courant < ~0.62 for every m.
    * ACCURATE — the r = 0 row only, i.e. an ordinary boundary condition at the axis
      and nothing else. Stable only for Courant <~ 1/(|m| + 0.5), which ``Grid``
      refuses above rather than warns about.

    The COMPONENT SET is the same in both branches and is post-#3164 (all three D
    components and all three B), which is measured rather than transcribed: the
    read-only checkout available here is v1.31.0-31, PRE-#3164, and its default branch
    zeroes only Dp/Dz and Br. See ``test_cylindrical``'s accurate-branch case for the
    parity number that establishes it against the 1.33.0 oracle.
    """
    return slice(0, 1) if grid.accurate_fields_near_cylorigin else slice(0, abs(grid.m))


def _cylindrical_axis_increment_B(fields: "Fields", electric: Dict[str, Any],
                                  boundaries: Tuple[str, str, str],
                                  phases: Any, dtdx: float):
    """The r = 0 axis-row increment for the B sub-step — (target, dfcnd row) or None.

    |m| = 1 only: Br on the axis takes -Courant*(Ep[i] - Ep[i + 1 along z])
    minus i*m*Courant*Ez at the FIRST OFF-AXIS Ez row (MEEP's
    ``f[Ez][1-cmp] + (nz+1)`` — one r row up, and the other complex part with
    the (1-2*cmp)*m factor, which is the -i*m multiplication written in complex
    form). Folded into the curl at row 0 by the caller so the PML recurrence
    applies MEEP's ladders — see `_cylindrical_axis_increment_D`.
    """
    grid = fields.grid
    if abs(grid.m) != 1:
        return None
    ep = electric["Ey"]
    ep_above = _shift_up(grid.xp, ep, 2, boundaries[2], phases[2])
    ez_off_axis = grid.xp.take(electric["Ez"], 1, axis=0)
    increment = (-dtdx) * (ep[_face(0, 0)] - ep_above[_face(0, 0)]) \
        - 1j * (grid.m * dtdx) * ez_off_axis
    return "Bx", increment


def _cylindrical_axis_zero_B(fields: "Fields") -> None:
    """The B-side axis-row zero constraints, applied after the field update.

    m = 0: Br on the axis is identically zero (a radial vector at r = 0 points
    nowhere). |m| > 1: ALL THREE B components held at zero within |m| pixels of
    the axis, with their fu AND f_cond auxiliaries (step_db.cpp, the same
    stability hack as the D side; the f_cond names are listed for symmetry with
    the D side and are None on every run this engine currently lifts, because
    the only B conductivity it builds is an ``mp.Absorber``, which the converter
    refuses on a cylindrical cell). The component set is post-#3164, like the
    D side — the 1.29-era code zeroed only Br.
    """
    grid = fields.grid
    if grid.m == 0:
        host_writes.zero(host_writes.raw(fields, "Bx"), 0, 0)
    elif abs(grid.m) > 1:
        rows = _cylindrical_axis_rows(grid)
        for name in ("Bx", "By", "Bz",
                     "fu_Bx", "fu_By", "fu_Bz", "f_cond_Bx", "f_cond_By", "f_cond_Bz"):
            aux = host_writes.raw(fields, name)
            if aux is not None:
                host_writes.zero_rows(aux, 0, rows)


def _cylindrical_imr_term(fields: "Fields", target: str, partner_values: Any,
                          sign: float) -> Any:
    """The i*m/r coupling — MEEP step_db.cpp:177-294, in native-complex form.

    MEEP adds, per real/imaginary part ``cmp``, ``the_m / r * g[i]`` with ``g``
    the OTHER part of the partner (``f[c_g][1 - cmp]``) and::

        the_m = 2*m * (1-2*cmp) * (1-2*(ft==B_stuff)) * (1-2*(d_c==R)) * Courant

    The ``(1-2*cmp)`` swap of parts IS multiplication by -i: writing the two rows
    as one complex update, delta_f = -i * s * 2m * Courant / r2 * g, where
    ``s = (1-2*(ft==B)) * (1-2*(d_c==R))`` arrives as ``sign`` and ``r2`` is the
    doubled ILOC r-coordinate at the target's own sites, ``2*ir + iyee_r`` (the
    factor 2 in the_m is the doubled-units compensation, kept together with it
    here). Complex storage is REQUIRED — the coupling has no real-storage form,
    which is MEEP's own rule (``change_m`` aborts on real fields for m != 0).

    Returned in CURL SIGN CONVENTION (the caller subtracts the curl), so the
    increment is negated once here. Folding it into the curl before the PML /
    conductivity application reproduces MEEP's separate laddered pass exactly:
    that pass is increment-only (no second decay of ``the_f``), and every ladder
    is linear in its increment.

    The axis row of a shift-0 target has r2 = 0; MEEP's OWNED0 loop never visits
    it (the per-m axis rules own that row) and the caller's mask zeroes it, so
    the divisor is clamped rather than special-cased.
    """
    grid = fields.grid
    xp = grid.xp
    if partner_values.dtype.kind != "c":
        raise ValueError(
            f"cylindrical m={grid.m} needs complex fields: the i*m/r coupling puts the "
            f"partner's field a quarter turn out of phase, which float32 storage cannot "
            f"carry (MEEP's change_m aborts on the same combination). Build the run with "
            f"force_complex_fields=True."
        )
    # The row vector is a GRID INVARIANT — Yee shift, radial extent, m, Courant — so
    # it is built once per (target, sign) rather than rebuilt from an arange, a
    # maximum, a reshape and a cast on every term of every step.
    def _build():
        iyee_r = IYEE_SHIFTS[target][0]
        r_doubled = 2 * xp.arange(partner_values.shape[0], dtype=xp.float64) + iyee_r
        divisor = xp.maximum(r_doubled, 1.0).reshape(-1, 1, 1)  # Row 0 of a shift-0 target is masked.
        return ((-1j) * (sign * 2.0 * grid.m * (grid.dt / grid.dx))
                / divisor).astype(partner_values.dtype)

    key = ("cyl_imr", target, float(sign), int(partner_values.shape[0]),
           partner_values.dtype, float(grid.m), float(grid.dt / grid.dx))
    factor = (_build() if fields.scratch is None
              else fields.scratch.constant(key, _build))
    return -(factor * partner_values)


def _special_kz_beta_term(fields: "Fields", partner_values: Any, sign: float,
                          magnetic: bool) -> Any:
    """The out-of-plane ``i*2*pi*beta`` coupling — MEEP step_db.cpp:148-176.

    The 2-D run carries ``exp(i*2*pi*beta*z)`` analytically, so ``d/dz`` on the
    invariant axis is the EXACT factor ``i*2*pi*beta``. Folded into the curl it is an
    ``i*beta*zhat x`` cross product, which couples the TE and TM polarizations. There
    is no 1/dx and no finite difference: this is an analytic derivative, so the term
    must not be scaled by ``dtdx``.

    MEEP adds, per real/imaginary part ``cmp``, ``betadt * g[i]`` with::

        g      = f[c_g][1 - cmp] ? f[c_g][1 - cmp] : f[c_g][cmp]
        betadt = 2*pi*beta*dt * (d_c == X ? +1 : -1)
                 * (f[c_g][1-cmp] ? (ft == D_stuff ? -1 : +1) * (2*cmp - 1) : 1)

    over ``d_c`` in {X, Y} only, so exactly Dx, Dy, Bx and By get a term and the z
    components get none. ``cc`` and its partner ``c_g`` are the two the cross product
    pairs: Dx<-Hy, Dy<-Hx, Bx<-Ey, By<-Ex, with ``sign = (d_c == X ? +1 : -1)``.

    COMPLEX STORAGE. The ``(2*cmp - 1)`` swap of parts IS multiplication by -i, so
    reading the two ``cmp`` rows as one complex update collapses MEEP's four cases to

        D: delta_f = sign * (-1j) * 2*pi*beta*dt * g
        B: delta_f = sign * (+1j) * 2*pi*beta*dt * g

    REAL STORAGE. ``f[c_g][1 - cmp]`` is NULL, the extra factor collapses to 1, and
    the same code is a plain real add of ``sign * 2*pi*beta*dt * g`` on BOTH sides.
    That is MEEP's "implicitly store i*(TM fields)" trick (its comment at
    step_db.cpp:148-160): the i's cancel because every observable is a conjugated
    bilinear. It is an interpretation of the same arithmetic, not a second code path,
    so reproducing the arithmetic reproduces the run — the one place the implicit i
    has to be paid explicitly is an eigenmode source, where MEEP applies
    ``special_kz_phasefix`` (mpb.cpp:295-304) before injecting.

    Returned in CURL SIGN CONVENTION (the caller subtracts the curl), so the
    increment is negated once here — the same fold :func:`_cylindrical_imr_term` uses,
    exact for the same reason: MEEP's beta pass is increment-only, it picks its PML
    and conductivity slices with the IDENTICAL ``cycle_direction(gv.dim, d_c, 1)`` and
    ``(..., 2)`` the main curl uses (step_db.cpp:56-59 against :166-169), and every
    ladder in :func:`_apply_curl` is linear in its increment.
    """
    grid = fields.grid
    coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt
    if partner_values.dtype.kind == "c":
        coefficient = coefficient * (1j if magnetic else -1j)
    elif fields.has_offdiagonal_epsilon:
        # MEEP's own pairing refusal, fields.cpp:548-549: an off-diagonal epsilon
        # ALREADY couples the two polarizations, and then the implicit i on the TM
        # half no longer cancels out of the constitutive product.
        raise ValueError(
            f"beta={grid.beta!r} with an off-diagonal epsilon needs complex fields: the "
            f"implicit i this engine carries on the TM half in real storage (MEEP's own "
            f"trick, step_db.cpp:148-160) cancels only while mu and epsilon leave TE and TM "
            f"uncoupled. MEEP aborts on the same combination (fields.cpp:548-549). Build the "
            f"run with force_complex_fields=True."
        )
    return -(partner_values.dtype.type(coefficient) * partner_values)


def _bfast_axis(component: str) -> int:  # MEEP's component_index — the component's OWN direction.
    """x -> 0, y -> 1, z -> 2 from a component name's last letter.

    ``component_index`` (vec.hpp:445) indexes ``bfast_scaled_k`` by the component's own
    direction, NOT by the direction its derivative is taken along. Getting that wrong is
    the single easiest mistake in the whole pass — it is why MEEP wrote the "puts k1 in
    direction of g2" comments at step_db.cpp:131/:133 — and it is silent, because on a
    k along one axis it merely moves the term to the wrong pair of components.
    """
    return AXIS_NAMES.index(component[-1].lower())


def _bfast_term(fields: "Fields", term: CurlTerm, operands: CurlOperands,
                magnetic: bool) -> Any:
    """The BFAST increment for one component — MEEP step_db.cpp:129-142 + step_generic.cpp:335-471.

    BFAST (broadband fixed angle source technique) is the time-sheared substitution
    ``t -> t - k.r/c``, under which a planewave at a FIXED incidence angle is
    transversely uniform at EVERY frequency, so plain periodic boundaries are exact
    for a whole band. In the stepper it is exactly::

        dB/dt = -curl(E) + d/dt (k x E)
        dD/dt = +curl(H) - d/dt (k x H)

    THE ARITHMETIC, transcribed. MEEP's eight branches all compute, per cell,

        S      = k1*(g1[i+s1] + g1[i]) - k2*(g2[i+s2] + g2[i])
        F_prev = F[i];  F[i] = S - F_prev;  raw = F[i] - F_prev

    and then push ``raw`` through a PURELY ADDITIVE ladder — no ``(kap - sig)*f`` and no
    ``(1 - dt/2*cnd)*f`` decay factor anywhere, because ``step_curl`` already applied
    those to the old field on this same timestep. So the whole feature reduces to
    ``raw``, and folding ``-raw`` into the curl before :func:`_apply_curl` reproduces
    all four of MEEP's paths exactly, for the reason :func:`_special_kz_beta_term`
    gives for its own fold: every ladder is AFFINE in the curl and the decay factors
    multiply only the old field, which the fold leaves alone. Checked branch by branch:

    * plain (step_generic.cpp:369-370): ``target -= curl`` gives ``f += raw``;
    * conductive (:353-354): ``f`` changes by ``-delta*condinv`` -> ``f += raw*cndinv``;
    * PML (:412-414 for the fu stage, :461-462 for the sig stage):
      ``fu += raw*siginv``, ``f += siginvu * that``;
    * conductive + PML (:479-489, MEEP's own "MOST GENERAL CASE"):
      ``fcnd += raw*cndinv``, then the two PML stages.

    WHY THE SUM AND NOT THE DIFFERENCE. ``S`` is twice the two-point average of the
    cross product interpolated onto the target's own Yee position, and ``F_n = S_n -
    F_{n-1}`` with output ``F_n - F_{n-1}`` is the z-domain filter
    ``(1 - z^-1)/(1 + z^-1)`` — the TUSTIN (bilinear) derivative — so ``raw`` is
    ``dt * d/dt`` of that average. That is where the ``d/dt`` of the equations above
    comes from, and it is why no ``dtdx`` and no explicit ``dt`` appear: MEEP passes
    ``dtdx`` to ``step_bfast`` and the body never reads it. Measured against
    ``dt * dA/dt`` on a smooth turn-on (``test_stepping``): 3.5e-04 relative, which is
    the finite-difference reference's own second-order floor, not the identity's.

    MARGINALLY STABLE, ON PURPOSE. The homogeneous mode of ``F_n = -F_{n-1}`` is
    ``(-1)^n``, undamped forever: a unit perturbation with no drive gives -1, +1, -1, …
    indefinitely (pinned in ``test_stepping``). MEEP has exactly this property. It
    means single-precision Nyquist noise in ``f_bfast`` never decays, unlike every other
    auxiliary in this stepper — and it is why the array MUST be restored around the
    magnetic synchronization half-step (``FdtdDriver._SYNC_AUXILIARY``) rather than
    left to decay away.

    THE k ASSIGNMENT (step_db.cpp:129-136), the subtle part::

        k1 = have_m ? bfast_scaled_k[component_index(c_m)] : 0   // multiplies g1 = f_p
        k2 = have_p ? bfast_scaled_k[component_index(c_p)] : 0   // multiplies g2 = f_m
        if (ft == D_stuff) { k1 = -k1; k2 = -k2; }

    so each k is indexed by the OTHER partner's own direction. For Bx (``first`` = Ez,
    ``second`` = Ey) that is ``k1 = k_y`` on Ez and ``k2 = k_z`` on Ey, i.e.
    ``(k x E)_x = k_y E_z - k_z E_y`` — the cross product, component for component.
    Note the sign pattern is NOT the curl's: the curl is ``+g1diff - g2diff`` built
    from ``(shifted - at)`` then ``(at - shifted)``, while this is ``+k1*g1sum -
    k2*g2sum`` with both sums in the same orientation.

    ``have_p``/``have_m`` are MEEP's ``figure_out_step_plan`` flags (fields.cpp:428-455),
    which are false exactly when the derivative direction is not one the grid resolves
    (``has_direction(gv.dim, cross(dc1, dc2))``) — so this reads
    ``grid.is_invariant`` on the partner's derivative axis. It is load-bearing in a
    reduced-dimension run and it is NOT the same test the curl passes silently: an
    invariant axis makes the curl's DIFFERENCE an exact zero all by itself, but the
    BFAST SUM of the same two samples is ``2*g``, not zero. MEEP's other reason for a
    false flag — ``gv.has_field``, which drops all but Ex/Hy/Dx/By in D1 — needs no
    separate test here: on the two components MEEP does step in D1 the two rules agree
    exactly, and the components it omits are identically zero on this grid.

    Returned in CURL SIGN CONVENTION (the caller subtracts the curl), so ``raw`` is
    negated once here. The state advance is masked with the same predicate the caller
    applies to the curl, because MEEP writes ``F`` only inside its owned-cell loop
    (``sub_gv.little_owned_corner0(cc)``): ``F`` has NO spatial stencil, so an unowned
    cell can never contaminate an owned one, but leaving the advance in would make a
    diagnostic read of ``f_bfast`` disagree with MEEP's.
    """
    grid = fields.grid
    bfast = grid.bfast_scaled_k
    have_p = not grid.is_invariant(term.first_axis)
    have_m = not grid.is_invariant(term.second_axis)
    k1 = bfast[_bfast_axis(term.second)] if have_m else 0.0
    k2 = bfast[_bfast_axis(term.first)] if have_p else 0.0
    if not magnetic:  # MEEP's `if (ft == D_stuff) { k1 = -k1; k2 = -k2; }`.
        k1, k2 = -k1, -k2

    state = getattr(fields, "f_bfast_" + term.target)
    if state is None:
        raise RuntimeError(
            f"BFAST stepping requires Fields to allocate f_bfast_{term.target} before the "
            f"first step (Fields._ensure_bfast_storage)."
        )
    dtype = state.dtype
    total = (dtype.type(k1) * (operands.shifted_first + operands.first)
             - dtype.type(k2) * (operands.shifted_second + operands.second))
    # F_new = S - F_prev, so the advance F_new - F_prev is S - 2*F_prev; masking the
    # advance masks the state and the increment together, which is what keeps an
    # unowned cell's F at exactly the value MEEP's loop never touches.
    advance = total - dtype.type(2.0) * state
    _mask_non_owned_cells(advance, grid, term.iyee)
    state += advance
    return -advance


def update_H(fields: "Fields", pml: Optional["PML"] = None) -> None:  # Magnetic constitutive sub-step.
    """Apply the magnetic constitutive relation H = B/mu with mu = 1.

    MEEP source: step.cpp (update_eh, H_stuff), step_generic.cpp
    (step_update_EDHB). Without PML this is a no-op — H is numerically identical
    to B and ``Fields.get_H`` serves it on demand, so nothing is stored. With
    PML the stored H accumulates absorption along the component's own axis
    (MEEP's dsigw) from integer-position coefficients.
    """
    _acquire_written(fields, H_COMPONENTS + _aux_names(H_COMPONENTS, ("f_w_",)))
    if not _pml_is_active(pml):
        return  # H is served on demand from B; there is no stored state to update.
    _require_pml_storage(fields, H_PML_ARRAYS)
    for component, source, axis in H_CONSTITUTIVE_TERMS:
        kps, kms = _constitutive_coefficients(pml, axis, half_integer=False)
        _apply_constitutive_pml(getattr(fields, component), getattr(fields, source),
                                kps, kms, getattr(fields, "f_w_" + component),
                                scratch=fields.scratch)


def update_E(fields: "Fields", pml: Optional["PML"] = None) -> None:  # Electric constitutive sub-step.
    """Apply the electric constitutive relation E = (D - sum P) * inv_eps.

    MEEP source: step.cpp (update_eh, E_stuff), update_eh.cpp (f_minus_p),
    step_generic.cpp (step_update_EDHB).

    With neither PML nor a susceptibility this is a no-op — ``Fields.get_E``
    computes D*inv_eps on demand and nothing is stored. Otherwise the stored E is
    written here, from D minus every polarization that drives the component
    (MEEP's ``dmp``, update_eh.cpp:143-146). With PML the stored E additionally
    accumulates absorption along the component's own axis from
    half-integer-position coefficients, because an E component is half-integer in
    its own direction; ``f_w`` keeps the un-absorbed constitutive product, which is
    what :meth:`~.fields.Fields.drive_field` hands the polarizations.

    P is consumed here and advanced afterwards, so this reads P^n and writes E^n.
    Running :func:`update_P` first is MEASURABLY equivalent rather than wrong — both
    indices shift together and cancel — but the equivalence holds only while nothing
    reads P between the two calls, so MEEP's order is what is kept. See
    :meth:`~.driver.FdtdDriver.step` for the derivation and the measurement.

    An instantaneous chi2/chi3 enters HERE and only here, as a scalar factor on the
    constitutive product (:func:`calc_nonlinear_u`). MEEP applies it inside the very
    function that also handles the PML auxiliary, so it composes with PML, with
    dispersion and with diagonal epsilon by construction rather than by arrangement:
    the same ``constitutive`` value is simply scaled before ``f_w`` takes it.
    """
    _acquire_written(fields, E_COMPONENTS + _aux_names(E_COMPONENTS, ("f_w_",)))
    pml_active = _pml_is_active(pml)
    if not pml_active and not fields.stores_E:
        return  # E is served on demand from D * inv_eps; there is no stored state to update.
    if pml_active:
        _require_pml_storage(fields, E_PML_ARRAYS)
    else:
        _require_field_storage(fields, E_STORAGE_ARRAYS)
    nonlinear = fields.has_nonlinearity
    offdiagonal = fields.has_offdiagonal_epsilon
    # All three D - sum P volumes have to be alive at once for the transverse Yee
    # average — the nonlinearity's |D|^2 and the off-diagonal row product both read
    # the OTHER components' volumes while this one is being written — so either
    # feature reads them together. A run with neither keeps the exact call it always
    # made, which is what holds the degenerate case to byte equality, not round-off.
    displacement = (_nonlinear_displacement(fields, pml if pml_active else None)
                    if (nonlinear or offdiagonal) else None)
    for component, _, axis_name in E_CONSTITUTIVE_TERMS:
        if nonlinear:
            constitutive = _nonlinear_constitutive(fields, component, displacement)
        elif offdiagonal:
            # MEEP step_update_EDHB (u1 && u2, no chi3): the full row product
            # fw[i] = gs*us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2).
            source = displacement["volumes"][component]
            constitutive = source * fields.inverse_epsilon_for(component)
            coupling = _offdiagonal_terms(fields, component, displacement)
            if coupling is not None:
                constitutive = constitutive + coupling
        else:
            source = fields.displacement_minus_polarization(component)
            constitutive = source * fields.inverse_epsilon_for(
                component
            )  # MEEP step_update_EDHB: fw[i] = gs*us.
        if pml_active:
            kps, kms = _constitutive_coefficients(pml, axis_name, half_integer=True)
            _apply_constitutive_pml(getattr(fields, component), constitutive,
                                    kps, kms, getattr(fields, "f_w_" + component),
                                    scratch=fields.scratch)
        else:
            # MEEP update_eh.cpp with dsigw == NO_DIRECTION: E = chi1inv * (D - P),
            # written into storage rather than returned, because P advances next.
            getattr(fields, component)[...] = constitutive


def calc_nonlinear_u(dsqr: Any, di: Any, chi1inv: Any, chi2: Any,
                     chi3: Any) -> Any:  # MEEP's Pade factor for E = chi1inv * u * D.
    """MEEP's ``calc_nonlinear_u`` (step_generic.cpp:542-548), transcribed term for term.

    Given ``Dsqr = |D|^2`` at this component's position and ``Di`` the component
    itself, returns the factor ``u`` such that ``Ei = chi1inv * u * Di``::

        c2 = Di   * chi2 * (chi1inv * chi1inv)
        c3 = Dsqr * chi3 * (chi1inv * chi1inv * chi1inv)
        u  = (1 + c2 + 2*c3) / (1 + 2*c2 + 3*c3)

    THE APPROXIMATION, IN MEEP'S OWN WORDS (step_generic.cpp:537-541): inverting
    ``D = eps*E + chi2*E^2 + chi3*|E|^2 E`` for E is a cubic, and this Pade
    approximant stands in for solving it. "This is inaccurate if the nonlinear index
    change is large, of course, but in that case the chi2/chi3 power-series expansion
    isn't accurate anyway, so the cubic isn't physical there either." Expanding to
    first order gives ``u = 1 - c2 - c3 + O(2)``, which is exactly the first
    perturbative correction, so the approximant is right to the order the material
    model itself is defined at.

    It is NOT unconditionally valid, which MEEP leaves to the user and this engine
    does not: ``1 + 2*c2 + 3*c3`` is a pole, and past it ``u`` changes sign and the
    recovered E is large and backwards. :func:`nonlinear_margin` measures the distance
    to it and :class:`~.driver.FdtdDriver` refuses a run that crosses it.

    Nothing here is stateful and nothing is per-step: no auxiliary field, no ODE, no
    history. This is deliberately unrelated to the dispersion ADE, which it is easy
    to assume it must resemble.
    """
    c2 = di * chi2 * (chi1inv * chi1inv)
    c3 = dsqr * chi3 * (chi1inv * chi1inv * chi1inv)
    return (1 + c2 + 2 * c3) / (1 + 2 * c2 + 3 * c3)


def _nonlinear_displacement(fields: "Fields",
                            pml: Optional["PML"]) -> Dict[str, Any]:  # D - sum P plus its ghost rules.
    """The three ``D - sum P`` volumes plus the ghost rules their Yee average needs.

    Bundled because the transverse average reads NEIGHBOURING cells of the other two
    components, so it needs exactly what the curl stencils need — the per-axis
    boundary kind, the Bloch wrap factor, and the mirror parity — resolved once for
    the whole sub-step rather than per component.
    """
    grid = fields.grid
    volumes = fields.displacement_minus_polarization_volumes()
    boundaries = _boundary_kinds(grid, pml)
    return {
        "volumes": volumes,
        "boundaries": boundaries,
        "parities": _mirror_phases(grid),
        "phases": _bloch_phases(grid, boundaries, volumes["Ex"], pml),
        "sums": {},  # Per-component transverse sums, formed once and reused by both parts.
    }


def _nonlinear_constitutive(fields: "Fields", component: str,
                            displacement: Dict[str, Any]) -> Any:  # One component's scaled gs*us.
    """``(gs * us) * calc_nonlinear_u(...)`` for one component — MEEP step_generic.cpp:646-655.

    REAL AND IMAGINARY PARTS ARE INDEPENDENT, and that is MEEP's doing, not a
    convenience. ``update_eh``'s ``DOCMP`` loop runs the whole nonlinear update once
    per Cartesian part of the complex field (update_eh.cpp:149), passing
    ``dmp[dc][cmp]`` — so Re E is recovered from Re D with a ``u`` built only from
    real parts, and likewise for the imaginary half. Forming ``Dsqr`` from the complex
    D instead would make it complex, hand ``u`` an imaginary part MEEP never has, and
    cost the complex-storage runs their parity floor while every magnitude still
    looked sensible.
    """
    gs = displacement["volumes"][component]
    us = fields.inverse_epsilon_for(component)
    # MEEP's MOST GENERAL CASE (step_generic.cpp:590-601) scales the WHOLE row
    # product by the Pade factor: fw = (gs*us + OFFDIAG + OFFDIAG) * u. The
    # coefficient is real, so taking a part of the complex coupling term equals
    # coupling the same part of the partner volumes — the DOCMP split commutes.
    coupling = _offdiagonal_terms(fields, component, displacement)
    if not fields.is_nonlinear(component):
        # A linear component in a partly nonlinear run: MEEP's `else if (u)` branch.
        return gs * us if coupling is None else gs * us + coupling
    chi2 = fields.chi2_for(component)
    chi3 = fields.chi3_for(component)
    transverse = _nonlinear_transverse_sums(fields, component, displacement)
    if gs.dtype.kind != "c":
        row = gs * us if coupling is None else gs * us + coupling
        return row * calc_nonlinear_u(_nonlinear_dsqr(gs, transverse, _WHOLE),
                                      gs, us, chi2, chi3)
    row_real = gs.real * us if coupling is None else gs.real * us + coupling.real
    row_imag = gs.imag * us if coupling is None else gs.imag * us + coupling.imag
    real = row_real * calc_nonlinear_u(_nonlinear_dsqr(gs, transverse, _REAL),
                                       gs.real, us, chi2, chi3)
    imaginary = row_imag * calc_nonlinear_u(_nonlinear_dsqr(gs, transverse, _IMAGINARY),
                                            gs.imag, us, chi2, chi3)
    return (real + 1j * imaginary).astype(gs.dtype, copy=False)


def _WHOLE(values: Any) -> Any:  # A real-storage field has no parts to separate.
    return values


def _REAL(values: Any) -> Any:  # MEEP's cmp == 0 array.
    return values.real


def _IMAGINARY(values: Any) -> Any:  # MEEP's cmp == 1 array.
    return values.imag


def _nonlinear_dsqr(gs: Any, transverse: Tuple[Any, Any], part: Any) -> Any:  # MEEP's Dsqr argument.
    """``gs*gs + 0.0625*(g1s*g1s + g2s*g2s)`` — MEEP step_generic.cpp:646-654.

    ``0.0625 = (1/4)^2`` is what turns each UNNORMALIZED four-point sum
    (:func:`_nonlinear_transverse_sums`) back into a mean before it is squared, so
    ``Dsqr`` really is ``|D|^2`` interpolated to this component's Yee position.

    ``part`` selects the real or imaginary half AFTER the sums are formed, never
    before: a Bloch wrap multiplies the whole complex value by ``exp(i k L)`` and so
    mixes the two, exactly as MEEP's ``step_boundaries`` does when it fills a phased
    chunk connection into the separate ``cmp`` arrays.
    """
    first, second = transverse
    scaled = part(gs)
    first_part = part(first)
    second_part = part(second)
    return scaled * scaled + 0.0625 * (first_part * first_part + second_part * second_part)


def _nonlinear_transverse_sums(fields: "Fields", component: str,
                               displacement: Dict[str, Any]) -> Tuple[Any, Any]:  # MEEP's g1s and g2s.
    """MEEP's ``g1s`` and ``g2s``: the two transverse D components summed onto this Yee point.

    step_generic.cpp:646-648::

        g1s = g1[i] + g1[i+s] + g1[i-s1] + g1[i+(s-s1)]

    where ``s`` is the stride along the component's OWN axis and ``s1`` the stride
    along the transverse one. The geometry is forced and the two shifts genuinely go
    in OPPOSITE directions: for Ez at ``(x_i, y_j, z_k+1/2)`` the partner Dx sits at
    ``(x_i+1/2, y_j, z_k)``, so reaching Ez costs half a cell DOWN in x and half a
    cell UP in z. Those are the four corners of the sum. Taking both shifts the same
    way round is a half-cell registration error that survives every scalar test and
    shows up only in the mixed-polarization terms.

    The partners follow ``cycle_direction`` (vec.hpp:586), X -> Y -> Z, so Ez takes
    Dx then Dy. Neighbour reads go through the same ``_shift_up`` / ``_shift_down``
    helpers the curl uses, so a periodic wrap, a Bloch phase and a mirror plane
    behave here exactly as they do there; both shifts are linear in the field, which
    is what lets the doubly-shifted corner be formed by shifting the already-shifted
    pair rather than by a third roll.
    """
    volumes = displacement["volumes"]
    boundaries = displacement["boundaries"]
    phases = displacement["phases"]
    parities = displacement["parities"]
    cache = displacement["sums"]
    if component in cache:
        return cache[component]
    xp = fields.grid.xp
    own_axis = AXIS_NAMES.index(component[1])
    sums = []
    for offset in (1, 2):  # MEEP's cycle_direction(dim, d_ec, 1) then (..., 2).
        partner_axis = (own_axis + offset) % 3
        values = volumes["E" + AXIS_NAMES[partner_axis]]
        # Half a cell DOWN the partner's own axis: g[i] + g[i-s1].
        pair = values + _shift_down(xp, values, partner_axis, boundaries[partner_axis],
                                    "D" + AXIS_NAMES[partner_axis], phases[partner_axis],
                                    parities[partner_axis])
        # Then the same pair half a cell UP this component's axis: + g[i+s] + g[i+(s-s1)].
        sums.append(pair + _shift_up(xp, pair, own_axis, boundaries[own_axis], phases[own_axis]))
    cache[component] = (sums[0], sums[1])
    return cache[component]


def _offdiagonal_terms(fields: "Fields", component: str,
                       displacement: Dict[str, Any]) -> Optional[Any]:  # MEEP's OFFDIAG sum.
    """The off-diagonal row contribution — MEEP step_generic.cpp:582-583, transcribed::

        OFFDIAG(u, g, sx) = 0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])

    ``s`` is the stride along this component's OWN axis, ``sx`` the partner's.
    The pairing is MEEP's "stable averaging", and it is exact rather than merely
    stable: ``structure_chunk::set_chi1inv`` registers the off-diagonal entries at
    the component's Yee site minus half a cell along its own axis — the integer
    node (anisotropic_averaging.cpp:248-257, ``here - shift1``) — so ``u[i]``
    multiplies the two-point partner average AT ITS OWN NODE and ``u[i+s]`` the
    average at the next node up. A plain four-point average times ``u[i]`` is the
    same arithmetic only where the coefficient is uniform; on the material
    interfaces the coefficient exists for, the two forms part company.

    The partner pair is formed exactly as :func:`_nonlinear_transverse_sums`
    forms its half — a half-cell DOWN the partner's axis, then the product
    shifted half a cell UP this component's axis — through the same
    ``_shift_up`` / ``_shift_down`` helpers the curl uses, so a periodic wrap, a
    Bloch phase and a metallic wall behave here exactly as they do there. The
    coefficient is periodic and carries no phase of its own; the wrap factor on
    the shifted PRODUCT is the field's, which is the same scalar-per-plane
    multiplication either way. (A folded axis never reaches this function:
    ``Fields.set_epsilon_volumes`` refuses the combination at install.)

    Returns ``None`` for a component with no surviving rows, so the caller's
    degenerate arithmetic is the diagonal engine's, not a ``+ 0`` copy of it.
    """
    rows = fields.chi1inv_offdiagonal_for(component)
    if not rows:
        return None
    volumes = displacement["volumes"]
    boundaries = displacement["boundaries"]
    phases = displacement["phases"]
    parities = displacement["parities"]
    xp = fields.grid.xp
    own_axis = AXIS_NAMES.index(component[1])
    total = None
    for offset in (1, 2):  # MEEP's cycle_direction(dim, d_ec, 1) then (..., 2).
        partner_axis = (own_axis + offset) % 3
        partner = "E" + AXIS_NAMES[partner_axis]
        coefficient = rows.get(partner)
        if coefficient is None:
            continue
        values = volumes[partner]
        # g[i] + g[i-sx]: half a cell DOWN the partner's own axis.
        pair = values + _shift_down(xp, values, partner_axis, boundaries[partner_axis],
                                    "D" + AXIS_NAMES[partner_axis], phases[partner_axis],
                                    parities[partner_axis])
        product = pair * coefficient
        # (pair*u)[i] + (pair*u)[i+s]: the same product half a cell UP this axis.
        term = 0.25 * (product + _shift_up(xp, product, own_axis,
                                           boundaries[own_axis], phases[own_axis]))
        total = term if total is None else total + term
    if total is not None:
        _mask_metallic_wall_coupling(total, fields.grid, IYEE_SHIFTS[component])
    return total


def _mask_metallic_wall_coupling(term: Any, grid: "Grid", iyee: Tuple[int, int, int]) -> None:
    """Zero the coupling term at METALLIC wall planes — and ONLY there.

    A metallic wall's tangential E is never stepped in MEEP — the owned loop for
    a shift-0 component starts one cell in, so the wall plane stays at its
    initialized zero. The diagonal product is zero there on its own
    (zero_metal_D holds the wall D at zero), but the coupling term reads the
    PARTNER volumes, which are alive beside the wall; measured unmasked on the
    uniform-tensor metallic oracle: 2.6e-02, against the 2.0e-07 floor masked.

    A MIRROR plane is the opposite case, which is why this is not
    :func:`_mask_non_owned_cells`: the fold-plane rows genuinely carry coupling,
    computed from the mirror ghosts the shift helpers already supply with the
    field's parity — and that is correct for BOTH coefficient parities. Measured
    by driven fold-equivalence against the full domain (12 steps, all primaries
    plus stored E): an even coefficient (chi_yz under an X fold) at 3.1e-12, an
    odd one (chi_xy, vanishing on the plane as any symmetric structure's must)
    at 8.3e-13, both together 4.7e-12 — while zeroing the fold plane the way the
    metallic rule does costs 2.0e-02 even and 3.5e-03 odd. The coefficient is
    never ghosted anywhere in the stencil (the partner-axis shift touches the
    field before the multiply; the own-axis shift goes UP, away from the fold),
    so no coefficient-parity bookkeeping is missing.
    """
    for axis in range(3):
        if iyee[axis] != 0:
            continue
        if grid.is_metallic(axis) and not grid.is_mirrored(axis):
            term[_face(axis, 0)] = 0


def cylindrical_rderiv_prefix(xp: Any, f_p: Any, ir0: float, scratch: Any = None) -> Any:
    """MEEP's ``f_rderiv_int``: the prefix sum that turns (1/r)·d(r·Fp)/dr into a plain difference.

    step_db.cpp:99-119, transcribed. The Dz update in cylindrical coordinates
    needs ``(1/r)·d(r·Fp)/dr`` where the ordinary curl computes ``dFp/dr``.
    Rather than a second curl kernel, MEEP builds the running sum::

        F[0,  :] = 0
        F[ir, :] = F[ir-1, :] + (1/((ir+ir0)-0.5)) * (f_p[ir]*(ir+ir0) - f_p[ir-1]*((ir-1)+ir0))

    whose plain radial difference IS the wanted derivative, and feeds ``F`` to the
    unmodified curl. ``ir0 = origin_r*a + 0.5*iyee_shift(Fp).in_direction(R)`` —
    for a cell touching the axis (origin_r = 0) and Fp = Hp (r-shift 1), ir0 = 0.5,
    so the weights are ``r/dr`` at Hp's own half-integer sites and the divisor is
    the integer site the difference lands on. Axis order here is the engine's:
    ``f_p`` is (r, phi, z) with r the leading axis; the sum runs along r.

    Vectorized as a cumulative sum — bit-equal to MEEP's sequential loop in exact
    arithmetic, and pinned against an explicitly transcribed loop in
    test_cylindrical.py rather than trusted (float summation order differs from a
    fused loop only in rounding; the test measures it).
    """
    real_dtype = f_p.real.dtype
    if scratch is None:
        counts = xp.arange(f_p.shape[0], dtype=xp.float64) + ir0  # (ir + ir0) per row.
        weights = counts.reshape(-1, 1, 1).astype(real_dtype)
        divisor = (counts[1:] - 0.5).reshape(-1, 1, 1).astype(real_dtype)
        weighted = f_p * weights
        increment = xp.zeros_like(f_p)
        increment[1:] = (weighted[1:] - weighted[:-1]) / divisor
        return xp.cumsum(increment, axis=0)

    # The two row vectors are GRID INVARIANTS — they depend on the radial extent, on
    # ir0 and on the storage dtype and on nothing that steps — but this runs twice a
    # timestep, so rebuilding them (an arange, two reshapes, two casts) was pure
    # per-step waste. The three full-volume temporaries are pooled for the reason
    # StepScratch exists.
    key = ("cyl_rderiv", int(f_p.shape[0]), float(ir0), real_dtype)
    weights, divisor = scratch.constant(key, lambda: _cylindrical_rderiv_weights(
        xp, f_p.shape[0], ir0, real_dtype))
    weighted = scratch.take("cyl_weighted", f_p.shape, f_p.dtype)
    xp.multiply(f_p, weights, out=weighted)
    increment = scratch.take("cyl_increment", f_p.shape, f_p.dtype)
    increment[_face(0, 0)] = 0  # The sum starts at zero; xp.zeros_like used to say so.
    xp.subtract(weighted[1:], weighted[:-1], out=increment[1:])
    increment[1:] /= divisor
    return xp.cumsum(increment, axis=0,
                     out=scratch.take("cyl_prefix", f_p.shape, f_p.dtype))


def _cylindrical_rderiv_weights(xp: Any, rows: int, ir0: float,
                                real_dtype: Any) -> Tuple[Any, Any]:  # (r/dr per row, its divisor).
    """The two invariant row vectors of :func:`cylindrical_rderiv_prefix`."""
    counts = xp.arange(rows, dtype=xp.float64) + ir0  # (ir + ir0) per row.
    return (counts.reshape(-1, 1, 1).astype(real_dtype),
            (counts[1:] - 0.5).reshape(-1, 1, 1).astype(real_dtype))


def nonlinear_margin(fields: "Fields",
                     pml: Optional["PML"] = None) -> Optional[NonlinearMargin]:  # Distance to the pole.
    """Measure how close this run's D field is to the Pade pole in :func:`calc_nonlinear_u`.

    Returns None when the run carries no nonlinearity. Otherwise it recomputes ``c2``
    and ``c3`` from the CURRENT D — the same stencil ``update_E`` uses, so the numbers
    describe the arithmetic actually performed — and reduces them to the worst
    ``1 + 2*c2 + 3*c3`` (the denominator, whose zero is the pole) and the worst
    ``|c2| + |c3|`` (the expansion parameter the approximant is a series in).

    The two are one inequality apart: ``1 + 2*c2 + 3*c3 >= 1 - 3*(|c2| + |c3|)``, so an
    expansion below 1/3 places the denominator strictly above zero and the run
    provably on the linear side of the pole. That is the bound
    :class:`~.driver.FdtdDriver` enforces, and it is derived rather than tuned.

    Both parts of a complex field are measured, since MEEP nonlinearizes them
    independently and either one can reach the pole alone.
    """
    if not fields.has_nonlinearity:
        return None
    xp = fields.grid.xp
    displacement = _nonlinear_displacement(fields, pml if _pml_is_active(pml) else None)
    volumes = displacement["volumes"]
    worst = NonlinearMargin(fields.nonlinear_components[0], float("inf"), 0.0)
    for component in fields.nonlinear_components:
        us = fields.inverse_epsilon_for(component)
        chi2 = fields.chi2_for(component)
        chi3 = fields.chi3_for(component)
        transverse = _nonlinear_transverse_sums(fields, component, displacement)
        stored = volumes[component]
        parts = (_WHOLE,) if stored.dtype.kind != "c" else (_REAL, _IMAGINARY)
        for part in parts:
            gs = part(stored)
            c2 = gs * chi2 * (us * us)
            c3 = _nonlinear_dsqr(stored, transverse, part) * chi3 * (us * us * us)
            denominator = float(xp.min(1 + 2 * c2 + 3 * c3))
            expansion = float(xp.max(abs(c2) + abs(c3)))
            worst = NonlinearMargin(
                component if expansion > worst.expansion else worst.component,
                min(denominator, worst.denominator),
                max(expansion, worst.expansion),
            )
    return worst


def update_P(fields: "Fields", pml: Optional["PML"] = None) -> None:  # Advance every polarization one step.
    """Advance every susceptibility's P by one timestep — MEEP update_pols.cpp:40-62.

    Runs AFTER :func:`update_E`, which is what makes the drive field
    W = (D - sum P) * inv_eps the value at the step just closed. The engine's step
    order mirrors MEEP's ``update_eh(E_stuff)`` then ``update_pols(E_stuff)``
    (step.cpp:105-114).

    Needs no boundary pass of any kind. The isotropic update is purely element-wise
    — MEEP's ``OFFDIAG`` neighbour reads live only in the anisotropic branches — so
    there is nothing for a periodic wrap, a Bloch phase or a mirror ghost to act on.
    Under mirror symmetry it therefore runs over the WHOLE stored array with no
    ownership mask: cell 0 of a folded axis gets the right P by induction, because
    its drive field is already right (``fill_symmetry_bc_D`` repairs D before
    ``update_E``). ``pml`` is accepted and unused for the same reason: the absorber
    reaches P only through W.

    The H-side slot MEEP fills after ``update_eh(H_stuff)`` is deliberately empty —
    magnetic susceptibilities are refused at the driver.
    """
    if host_writes.holding():
        # The ADE pool -- P, P_prev and the scratch the update writes into -- is held
        # under a Metal residency and rotated every launch; the array-path update
        # writes all three, so all three are acquired whole. Same rule as the other
        # sub-steps, reached only by a control or a fallback on a held run.
        for state in (getattr(fields, "polarizations", ()) or ()):
            for mapping_name in ("P", "P_prev"):
                mapping = getattr(state, mapping_name, None)
                if isinstance(mapping, dict):
                    for value in list(dict.values(mapping)):
                        if hasattr(value, "shape"):
                            host_writes.acquire(value)
            scratch = host_writes.raw(state, "_scratch", None)
            if scratch is not None and hasattr(scratch, "shape"):
                host_writes.acquire(scratch)
    del pml  # The absorber reaches the polarization through W and nowhere else.
    if not fields.has_polarizations:
        return
    for state in fields.polarizations:
        state.update(fields.drive_field, fields.grid.dt)


def fill_symmetry_bc_D(fields: "Fields") -> None:  # D-side mirror repair; the driver calls it post-injection.
    """Repair the D cells the owned-cell loop skips on a mirror plane.

    MEEP source: boundaries.cpp (connect_the_chunks, step_boundaries).

    Call this AFTER source injection and before update_E: MEEP's order is
    step_db -> step_source -> step_boundaries, so injecting into a cell that the
    boundary pass is about to overwrite is intentional and order-dependent.
    """
    _fill_symmetry_ghost_cells(fields, D_CURL_TERMS)


def fill_symmetry_bc_B(fields: "Fields") -> None:  # B-side mirror repair; the driver calls it post-injection.
    """Repair the B cells the owned-cell loop skips on a mirror plane.

    MEEP source: boundaries.cpp (connect_the_chunks, step_boundaries for B_stuff).

    Call this AFTER magnetic source injection and before update_H, exactly as
    :func:`fill_symmetry_bc_D` is called after the electric one: MEEP's B half is
    ``step_db(B_stuff)`` / ``step_source(B_stuff)`` / ``step_boundaries(B_stuff)``
    (step.cpp:67-72), the same three in the same order as the D half at :95-103.

    Running it a sub-step early — inside ``step_B``, where it used to live on the
    claim that MEEP's B boundary pass has "no source injection in between" — leaves
    every mirror ghost one injection stale, because the pass images the owned cell
    BEFORE that cell receives its current. It is invisible to any run with no
    magnetic current, and on a Y fold it can only reach ``By``: ownership is
    ``little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)``, so of the three
    B components only the one with Yee shift 0 on the folded axis has a ghost row at
    all. That is exactly the sheet an ``mp.EigenModeSource`` drives, which is why the
    defect presented as an eigenmode one on a single component: mode-decomposition.py
    at resolution 17 read Hy 2.72e-03 against CPU MEEP with Ez at 1.32e-06 and Hx at
    2.51e-06, and 1.62e-06 on Hy with the fill in this slot (whole volume 2.309e-03 ->
    1.802e-06, Ez and Hx bit-unchanged). On a straight-waveguide reduction of the same
    launch, folded Hy 5.20e-03 -> 7.43e-07 against the unfolded twin's 7.37e-07.
    """
    _fill_symmetry_ghost_cells(fields, B_CURL_TERMS)


def _fill_symmetry_ghost_cells(fields: "Fields", terms: Sequence[CurlTerm]) -> None:  # Mirror-fill one family.
    """Write ``cell 0 = parity * cell 2`` on every mirrored axis a component does not own.

    Ownership follows MEEP's little_owned_corner0(c) = little_corner + 2 -
    iyee_shift(c). On a halved grid (io = -2 in doubled coordinates) that is
    -iyee_shift, so cell 0 is unowned exactly when the component's Yee shift on
    that axis is 0. The axes are filled in X, Y, Z order, which leaves a corner
    unowned on two planes carrying the product of both parities — the doubly
    mirrored value MEEP's chunk connection produces — and a corner unowned on all
    three carrying the product of three.
    """
    grid = fields.grid
    if not grid.has_symmetry():
        return
    parities = _mirror_phases(grid)
    for term in terms:
        # RAW, then a face-to-face copy through the residency door: under a hold
        # both planes are on the device and nothing crosses the bus. A barriered
        # fetch here synced the whole volume, per component, per fold axis, per
        # step -- which is why mirror-folded grids refused the hold until now.
        array = host_writes.raw(fields, term.target)
        shifts = term.iyee
        for axis in range(3):
            if parities[axis] is None or shifts[axis] != 0:
                continue
            _write_mirror_ghost(array, axis, _symmetry_phase(term.target, axis, parities[axis]))


def _write_mirror_ghost(field: Any, axis: int, phase: int) -> None:  # cell 0 <- phase * cell 2 on one axis.
    _mirror_source(field, axis)  # the range check it has always made
    host_writes.copy_face(field, axis, 0, field, MIRROR_SOURCE_INDEX, phase)


def _stored_past_owned(grid: "Grid", axis: int) -> bool:  # Does this axis store MEEP's far ghost slot?
    """True exactly on a folded PERIODIC axis, at either full-count parity.

    There — and only there — ``Grid.stored_cells`` exceeds ``Grid.owned_cells``
    by one: the stored array carries MEEP's ``big_corner`` plane (a shift-0
    sample, owned and stepped) AND the shift-1 slot half a cell past it, which
    is MEEP's not-owned allocation slot (``update_ntot``'s num + 1,
    vec.cpp:293-296) that ``connect_the_chunks`` fills with the parity-weighted
    image about the second mirror at doubled ``n_full``. The ``getattr`` guard
    serves the stub grids in test_stepping.py, which model the registration
    conventions only and never build the extra slot.
    """
    reader = getattr(grid, "owned_cells", None)
    if not callable(reader):
        return False
    return grid.is_mirrored(axis) and grid.stored_cells(axis) > reader(axis)


def fill_folded_far_ghosts_D(fields: "Fields") -> None:  # D-side far-plane ghost; driver calls post-injection.
    """Fill the far ghost slot of every folded-periodic even axis, D family.

    The mirror image of :func:`zero_metal_D`'s slot in the step order: MEEP's
    ``step_boundaries`` runs after ``step_source`` on each side (step.cpp:64-105),
    so the image is taken AFTER the electric currents land — a current deposited
    at doubled ``n_full - 1`` must appear in the ghost at ``n_full + 1`` the same
    sub-step, exactly as ``connect_the_chunks``' copy would carry it.
    """
    _fill_folded_far_ghosts(fields, D_CURL_TERMS)


def fill_folded_far_ghosts_B(fields: "Fields") -> None:  # B-side far-plane ghost; driver calls post-injection.
    """The magnetic half of :func:`fill_folded_far_ghosts_D`, in the same relative slot."""
    _fill_folded_far_ghosts(fields, B_CURL_TERMS)


def _fill_folded_far_ghosts(fields: "Fields", terms: Sequence[CurlTerm]) -> None:  # Far-plane fill, one family.
    """Image the top shift-1 slot of a folded periodic axis about the second mirror.

    The far twin of :func:`_fill_symmetry_ghost_cells`. On such an axis the last
    stored slot of a shift-1 component sits one half-cell past MEEP's
    ``big_corner``, above the second mirror at doubled ``n_full``; its image
    under translation + reflection is the stored row
    :func:`_far_reflect_rows` derives, ``n_full - stored + 2``, carrying
    ``mirror_parity(component, axis, plane phase)`` and no Bloch factor (a folded
    axis refuses a nonzero k outright). MEEP's ``connect_the_chunks`` builds
    exactly this copy (boundaries.cpp:282 ``locate_component_point``:
    ``locate_point_in_user_volume`` then ``S.transform`` / ``S.phase_shift``);
    the D sub-step's backward differences at the top row are what consume it.

    THE IMAGE ROW IS NOT ALWAYS ``n - 2``. Both parities put the ghost one
    half-cell above ``big_corner``, but ``big_corner`` is the second mirror
    itself at an EVEN count (ghost at doubled ``n_full + 1``, image ``n_full -
    1`` — the row below, slot ``n - 2``) and one half-cell ABOVE it at an ODD
    one (ghost at doubled ``n_full + 2``, image ``n_full - 2`` — slot ``n - 3``).
    Writing the fixed ``n - 2`` reflects about the window top instead of about
    the mirror, which is a whole cell wrong on every odd-count run.

    Runs whole-plane per component like the near-face fill; the shift-0
    components need nothing here (their top slot IS ``big_corner``, owned and
    stepped), and a folded METALLIC axis keeps its unstored zero ghost.
    """
    grid = fields.grid
    if not grid.has_symmetry():
        return
    axes = [axis for axis in range(3) if _stored_past_owned(grid, axis)]
    if not axes:
        return
    parities = _mirror_phases(grid)
    reflect_rows = _far_reflect_rows(grid)
    for term in terms:
        array = host_writes.raw(fields, term.target)
        shifts = term.iyee
        for axis in axes:
            if shifts[axis] != 1:
                continue
            host_writes.copy_face(array, axis, -1, array, reflect_rows[axis],
                                  _symmetry_phase(term.target, axis, parities[axis]))


def _mirror_source(field: Any, axis: int) -> Any:  # The stored cell a mirrored ghost reflects from.
    if field.shape[axis] <= MIRROR_SOURCE_INDEX:
        raise ValueError(
            f"a mirrored axis needs more than {MIRROR_SOURCE_INDEX} stored cells to reflect "
            f"from, axis {AXIS_NAMES[axis]} has {field.shape[axis]}"
        )
    return field[_face(axis, MIRROR_SOURCE_INDEX)]


class CurlOperands(NamedTuple):
    """The four arrays one curl stencil reads: each partner and its shifted neighbour.

    Gathered once so the curl's DIFFERENCES and BFAST's SUMS share a single pass over
    memory. MEEP cannot do this — its ``step_curl`` and ``step_bfast`` are separate
    loops over the whole grid (step_db.cpp:124-142), so a BFAST run there pays a second
    full read of every component. Here the extra term is a handful of FLOPs on operands
    already in registers.
    """

    first: Any           # MEEP's g1 = f_p, at the cell.
    shifted_first: Any   # g1[i + s1] — one cell up (B) or down (D).
    second: Any          # MEEP's g2 = f_m, at the cell.
    shifted_second: Any  # g2[i + s2].


def _curl_operands(xp: Any, snapshot: Dict[str, Any], term: CurlTerm,
                   boundaries: Tuple[str, str, str], phases: BlochPhases,
                   parities: MirrorPhases, backward: bool,
                   reflect_rows: FarReflectRows = (None, None, None),
                   scratch: Any = None) -> CurlOperands:
    """Gather one component's two curl partners and their shifted neighbours.

    ``backward`` selects MEEP's negated strides (s = -1, the D sub-step) over the
    forward ones (s = +1, the B sub-step) — the same choice for both the curl and the
    BFAST pass, because MEEP hands ``step_bfast`` the SAME already-negated
    ``stride_p``/``stride_m`` it handed ``step_curl`` (step_db.cpp:80-83 negates them
    once, before both calls).
    """
    first = snapshot[term.first]
    second = snapshot[term.second]
    # The two shifted operands are alive at the same time — the curl differences them
    # and BFAST sums them — so they take SEPARATE pool tags. One tag for both would
    # hand the same buffer back twice and silently turn the stencil into g1 - g1.
    if backward:
        shifted_first = _shift_down(xp, first, term.first_axis,
                                    boundaries[term.first_axis], term.first,
                                    phases[term.first_axis], parities[term.first_axis],
                                    scratch=scratch, scratch_tag="stencil_first")
        shifted_second = _shift_down(xp, second, term.second_axis,
                                     boundaries[term.second_axis], term.second,
                                     phases[term.second_axis], parities[term.second_axis],
                                     scratch=scratch, scratch_tag="stencil_second")
    else:
        shifted_first = _shift_up(xp, first, term.first_axis, boundaries[term.first_axis],
                                  phases[term.first_axis], term.first,
                                  parities[term.first_axis],
                                  reflect_rows[term.first_axis],
                                  scratch=scratch, scratch_tag="stencil_first")
        shifted_second = _shift_up(xp, second, term.second_axis, boundaries[term.second_axis],
                                   phases[term.second_axis], term.second,
                                   parities[term.second_axis],
                                   reflect_rows[term.second_axis],
                                   scratch=scratch, scratch_tag="stencil_second")
    return CurlOperands(first, shifted_first, second, shifted_second)


def _curl_from_operands(operands: CurlOperands, dtdx: float,
                        scratch: Any = None) -> Any:  # dtdx * the discrete curl.
    """Return dtdx * (g1[i+s] - g1[i] + g2[i] - g2[i+s]) from an already-gathered stencil.

    The expression form allocates FOUR full-volume temporaries per call and runs six
    times a timestep; with a :class:`~.fields.StepScratch` the same four passes write
    into two pooled buffers instead. Bit-identical by construction — the ufuncs, their
    operand order and the promoted dtype are the same, only the destination differs —
    and pinned that way by the before/after byte comparison in
    ``test_step_scratch_is_bit_identical``.

    The returned buffer is the pool's ``curl`` slot and stays valid until the next
    curl is formed, which is exactly how far the caller carries it: the term loop
    masks it, may add a coupling term (allocating a fresh array, which releases the
    slot) and hands it to :func:`_apply_curl` before the next iteration.
    """
    if scratch is None:
        return dtdx * ((operands.shifted_first - operands.first)
                       + (operands.second - operands.shifted_second))
    xp = scratch.xp
    first = operands.first
    # The four operands share the field storage dtype in every configuration the
    # engine steps; the promotion call is kept for the case that stops being true, but
    # it is not paid six times a step to learn that they already agree.
    dtype = first.dtype
    if not (operands.shifted_first.dtype == dtype and operands.second.dtype == dtype
            and operands.shifted_second.dtype == dtype):
        dtype = xp.result_type(operands.shifted_first, first,
                               operands.second, operands.shifted_second)
    total = scratch.take("curl", first.shape, dtype)
    partial = scratch.take("curl_partial", first.shape, dtype)
    xp.subtract(operands.shifted_first, operands.first, out=total)
    xp.subtract(operands.second, operands.shifted_second, out=partial)
    xp.add(total, partial, out=total)
    xp.multiply(dtdx, total, out=total)  # Operand order kept as the expression had it.
    return total


def _curl(xp: Any, snapshot: Dict[str, Any], term: CurlTerm, dtdx: float,
          boundaries: Tuple[str, str, str], phases: BlochPhases,
          parities: MirrorPhases, backward: bool,
          reflect_rows: FarReflectRows = (None, None, None),
          scratch: Any = None) -> Any:  # One component's dtdx-scaled curl.
    """Return dtdx * (g1[i+s] - g1[i] + g2[i] - g2[i+s]) for one component.

    ``backward`` selects MEEP's negated strides (s = -1, the D sub-step) over
    the forward ones (s = +1, the B sub-step). ``phases`` carries the per-axis
    Bloch wrap factor, None on an axis that wraps unphased; ``parities`` carries
    the per-axis mirror phase, None on an axis with no plane; ``reflect_rows``
    the stored row a folded PERIODIC axis's far face reflects from, None on
    every other axis (:func:`_far_reflect_rows`).
    """
    return _curl_from_operands(
        _curl_operands(xp, snapshot, term, boundaries, phases, parities, backward,
                       reflect_rows, scratch=scratch),
        dtdx,
        scratch=scratch,
    )


def _far_reflect_rows(grid: "Grid") -> FarReflectRows:  # Far-face image row per axis, None where absent.
    """The stored row the first sample PAST a folded PERIODIC axis's window top images.

    Under periodic boundaries a mirror at doubled 0 implies a second mirror at
    doubled ``n_full`` (MEEP reaches past the folded half by a lattice translation
    followed by the symmetry transform — boundaries.cpp ``connect_the_chunks`` ->
    ``locate_component_point``), so doubled coordinate ``p`` above the plane
    images ``2*n_full - p``. Every array :func:`_shift_up` shifts has Yee shift 0
    on the shifted axis, so the sample one past the stored top sits at doubled
    ``origin + 2*n_q = 2*n_q - 2`` and its image is stored row
    ``n_full - n_q + 2``: two rows below the stored top at an even full count,
    three at an odd one, where ``big_corner`` — and so the stored top — sits one
    half-cell above the second mirror rather than on it. The same row serves the
    shift-1 far ghost (:func:`_fill_folded_far_ghosts`): the image of stored row
    ``j`` at Yee shift ``s`` is ``n_full - j - s + 2``, and the ghost's ``(j, s) =
    (n_q - 1, 1)`` lands on the same row as this pass's ``(n_q, 0)``.
    ``None`` on unfolded axes and on folded METALLIC axes, where the
    zero ghost IS the boundary condition (MEEP holds that plane at zero,
    find_metals / zero_metal). A folded axis refuses a nonzero Bloch phase
    outright (`FdtdDriver._require_bloch_is_representable`), so the reflect
    carries the mirror parity and no wrap factor.
    """
    rows: list[Optional[int]] = []
    for axis in range(3):
        if not grid.is_mirrored(axis) or grid.is_metallic(axis):
            rows.append(None)
            continue
        n_full = grid.shape_full[axis]
        rows.append(n_full - grid.stored_cells(axis) + 2)
    return tuple(rows)


def _rolled(xp: Any, field: Any, shift: int, axis: int,
            scratch: Any, tag: str) -> Any:  # xp.roll(field, shift, axis), pooled.
    """``xp.roll(field, shift, axis)`` for ``shift`` of -1 or +1, into a pooled buffer.

    ``xp.roll`` materializes a FULL COPY of the volume to express a one-cell shift, and
    the stencil needs twelve of them per timestep — measured at 126 us each on a
    640x640 float32 array against 53 us for the same bytes written into an existing
    buffer, because the allocation is most of the cost above a megabyte (see
    :class:`~.fields.StepScratch`). This writes the same two slice copies ``roll``
    performs into a reused buffer instead, so the movement is identical byte for byte
    and only the destination is not fresh.

    ``scratch`` of None keeps the allocating call, which is what every direct caller
    outside the curl stencil takes.
    """
    if scratch is None:
        return xp.roll(field, shift, axis=axis)
    out = scratch.like(tag, field)
    count = field.shape[axis]
    if shift == -1:
        out[_span(axis, 0, count - 1)] = field[_span(axis, 1, count)]
        out[_face(axis, -1)] = field[_face(axis, 0)]
    elif shift == 1:
        out[_span(axis, 1, count)] = field[_span(axis, 0, count - 1)]
        out[_face(axis, 0)] = field[_face(axis, -1)]
    else:
        raise ValueError(f"_rolled carries the one-cell stencil shifts only, got {shift!r}.")
    return out


def _shift_up(xp: Any, field: Any, axis: int, boundary: str,
              phase: Optional[complex] = None, component: Optional[str] = None,
              mirror_phase: Optional[int] = None,
              reflect_row: Optional[int] = None,
              scratch: Any = None,
              scratch_tag: str = "shift_up") -> Any:  # Neighbour one cell up.
    """Return field[i+1] along one axis, applying that axis's far-face ghost rule.

    The far side of a symmetry plane over METALLIC boundaries is metallic: MEEP's
    forward differences read a zero ghost past the last owned cell of such a folded
    axis. A declared METALLIC axis takes the same rule, and for the same reason —
    there is no field beyond a perfect electric conductor.

    The far side of a symmetry plane over PERIODIC boundaries is the SECOND
    mirror at +L/2 (a mirror at 0 under a lattice translation implies one at half
    the lattice vector), and the ghost there is the parity-weighted image of a
    stored interior plane, not zero: ``reflect_row`` names that plane
    (:func:`_far_reflect_rows`) and ``component``/``mirror_phase`` carry the
    parity. Callers that shift arrays with no single component parity — the
    nonlinear transverse sums — leave ``reflect_row`` unset and keep the zero
    face; the driver refuses to combine a nonlinearity with a live-faced periodic
    fold rather than serve that zero as an answer.

    WHICH SAMPLE THE ZERO STANDS FOR, on a metallic axis. Every array this helper
    shifts has Yee shift 0 on the shifted axis: the two curl partners of a component
    always sit on the opposite side of the half-cell from it, and the forward
    difference is always the shift-1 component reading its shift-0 neighbours (Bx
    reads Ez and Ey; Dx's backward difference reads Hz and Hy, which is
    :func:`_shift_down`'s case). A shift-0 sample lives at ``origin + i*dx``, so cell
    ``i`` here is MEEP's array slot ``i``, the plane one past the last stored cell is
    MEEP's slot ``N`` — the HIGH wall — and MEEP holds that slot at exactly zero
    (boundaries.cpp ``find_metals`` collects it, ``zero_metal`` clears it every
    sub-step). Stored cell 0 is the LOW wall and is held at zero here for the same
    reason, so the periodic roll would in fact deliver the right value by itself; the
    zero is written anyway so the stencil states the boundary rather than inheriting
    it from an invariant maintained three functions away.

    On a periodic axis the wrapped far face is the field one lattice vector up,
    ``f(x + L)``, which under Bloch boundaries is ``bloch_phase * f(x)`` — so the
    wrapped plane, and only that plane, is multiplied by ``phase``. ``phase`` is
    None at k = 0 and the multiply is then skipped entirely rather than done
    against ``1 + 0j``, which is what keeps k = 0 bit-identical to the plain
    periodic engine.
    """
    shifted = _rolled(xp, field, -1, axis, scratch, scratch_tag)  # roll(f, -1)[i] = f[i+1].
    if boundary == PERIODIC:
        if phase is not None:
            _apply_bloch_phase(shifted, axis, -1, phase)
        return shifted
    if boundary == MIRROR and reflect_row is not None and component is not None:
        # Folded PERIODIC axis: the sample one past the stored top images stored
        # row `reflect_row` about the second mirror at doubled n_full, with the
        # component's own parity about that plane (same plane phase as the
        # declared mirror — the transform is translation-then-reflection, and at
        # k = 0 the translation carries no factor).
        shifted[_face(axis, -1)] = (_symmetry_phase(component, axis, mirror_phase)
                                    * field[_face(axis, reflect_row)])
        return shifted
    if boundary in (MIRROR, METALLIC, CYL_AXIS):
        shifted[_face(axis, -1)] = 0
        return shifted
    raise ValueError(f"unknown boundary condition {boundary!r} on axis {AXIS_NAMES[axis]}")


def _shift_down(xp: Any, field: Any, axis: int, boundary: str, component: str,
                phase: Optional[complex] = None,
                mirror_phase: Optional[int] = None,
                scratch: Any = None,
                scratch_tag: str = "shift_down") -> Any:  # Neighbour below.
    """Return field[i-1] along one axis, applying that axis's near-face ghost rule.

    A mirrored axis reflects with the component's parity, which carries the
    plane's declared phase (``mirror_phase``: +1 even, -1 odd).
    Ownership masking then zeroes the curl at cell 0 of every mirrored axis whose
    Yee shift is 0 — which, for the D sub-step, is exactly the cell that consumes
    the mirror ghost. The ghost is still built so the stencil stays correct on its
    own terms rather than depending on the masking rule for its validity.

    A METALLIC axis reflects nothing: past a perfect electric conductor the field is
    zero, so the near ghost is zero. Every array this helper shifts has Yee shift 1 on
    the shifted axis (see :func:`_shift_up` for why the two are always opposite), so
    the ghost sits half a cell OUTSIDE the low wall — a sample MEEP does not allocate
    at all, because its owned loop for the shift-0 consumer starts one cell in. Its
    consumer is stored cell 0, which is the wall itself and whose curl
    :func:`_mask_non_owned_cells` drops; the zero is written so the stencil is right
    on its own terms rather than depending on that mask for its validity, exactly as
    the mirror ghost is.

    The periodic near face wraps *down* by one lattice vector, ``f(x - L)``, so it
    carries the CONJUGATE of the up-going Bloch factor — MEEP's
    ``locate_point_in_user_volume`` multiplies by ``conj(eikna[d])`` on exactly
    this translation (boundaries.cpp). Taking the same factor in both directions
    is the classic sign error: it leaves every magnitude plausible and moves only
    the phase, which is what a band structure is made of.
    """
    shifted = _rolled(xp, field, 1, axis, scratch, scratch_tag)  # roll(f, +1)[i] = f[i-1].
    if boundary == PERIODIC:
        if phase is not None:
            _apply_bloch_phase(shifted, axis, 0, phase.conjugate())
        return shifted
    if boundary == MIRROR:
        shifted[_face(axis, 0)] = (_symmetry_phase(component, axis, mirror_phase)
                                   * _mirror_source(field, axis))
        return shifted
    if boundary == METALLIC:
        shifted[_face(axis, 0)] = 0
        return shifted
    if boundary == CYL_AXIS:
        # MEEP r_to_minus_r_symmetry(m): the ghost at -dr/2 is the field's own
        # value at +dr/2 (stored row 0 — every shifted array has Yee shift 1 on
        # this axis, so no other row can be the image) with R- and P-DIRECTION
        # components sign-flipped and an overall (-1)^m. Under the x->r, y->phi
        # map that flips the x- and y-direction families and keeps z. The
        # (-1)^m factor rides in the mirror_phase slot, threaded by
        # _mirror_phases from grid.m.
        sign = -1.0 if component[1] in ("x", "y") else 1.0
        if mirror_phase is not None:
            sign *= mirror_phase
        shifted[_face(axis, 0)] = sign * field[_face(axis, 0)]
        return shifted
    raise ValueError(f"unknown boundary condition {boundary!r} on axis {AXIS_NAMES[axis]}")


def _apply_bloch_phase(shifted: Any, axis: int, index: int, phase: complex) -> None:  # Phase one wrapped plane.
    """Multiply the single wrapped plane of a rolled array by its Bloch factor, in place.

    ``shifted`` is always a shift buffer — a fresh ``xp.roll`` or the pooled
    equivalent (:func:`_rolled`) — so this never touches stored field state, and the
    buffer is rewritten in full before its next use either way. Real storage cannot
    carry the phase, and NumPy would
    signal that as a casting error from the in-place multiply; it is checked here
    instead so the message names the actual configuration problem.
    """
    if shifted.dtype.kind != "c":
        raise ValueError(
            f"Bloch boundaries need complex fields: axis {AXIS_NAMES[axis]} carries phase "
            f"{phase!r} but the field storage is {shifted.dtype}. Build the run with "
            f"force_complex_fields=True, or use k_point = 0 on this axis."
        )
    shifted[_face(axis, index)] *= shifted.dtype.type(phase)


def _mask_non_owned_cells(curl: Any, grid: "Grid", iyee: Tuple[int, int, int]) -> None:  # Drop unowned cells.
    """Zero the curl in the cells MEEP's owned-cell loop never visits, or forces to zero.

    MEEP step_db.cpp starts each loop at little_owned_corner0(c) = little_corner
    + 2 - iyee_shift(c), which on a halved grid excludes cell 0 for every
    component whose Yee shift on the mirrored axis is 0. The rule is per-axis and
    reads ``Grid.is_mirrored``, so a folded Z masks exactly as a folded X does.

    A METALLIC axis masks the same cell for a different reason, and the arithmetic
    lands in the same place. Stored cell 0 of a shift-0 component sits AT the low
    wall (``origin + 0*dx``), where a perfect electric conductor forces the field to
    zero: MEEP allocates that plane as an unconnected ghost (``connect_the_chunks``
    skips it through ``on_metal_boundary``) and separately zeroes the high wall it
    does own, so neither wall carries a stepped value. Masking here is what makes the
    engine's single stored wall plane inert; :func:`zero_metal_D` then holds it at
    zero against anything injected into it.

    Masking the curl rather than skipping the field write is what keeps the PML
    auxiliaries honest: an unowned cell then integrates a zero curl and decays,
    instead of accumulating a curl assembled from a ghost it does not own.
    """
    for axis in range(3):
        if iyee[axis] != 0:
            # A component with Yee shift 1 does own cell 0 — but on a folded
            # PERIODIC axis (either count parity) its LAST stored slot is the
            # far ghost half a cell past MEEP's big_corner, which `owns` puts
            # outside the owned window (vec.cpp:445-462: 0 < p - io <= 2*num).
            # Mask its curl so the fill pass (`fill_folded_far_ghosts_*`) is
            # what writes it, exactly as cell 0's mask pairs with
            # `fill_symmetry_bc_D`.
            if _stored_past_owned(grid, axis):
                curl[_face(axis, -1)] = 0
            continue
        if grid.is_mirrored(axis) or grid.is_metallic(axis) or grid.is_axis(axis):
            # The cylindrical axis masks for MEEP's own reason: little_owned_corner0
            # deliberately excludes r = 0 ("which is updated separately",
            # vec.hpp:1100-1104) — the axis row belongs to the per-m rules.
            curl[_face(axis, 0)] = 0


def _apply_pml_update(field: Any, curl: Any, kms: Any, sinv: Any,
                      kms_u: Any, sinv_u: Any, fu: Any,
                      scratch: Any = None) -> None:  # Split-field PML curl recurrence.
    """Run the split-field PML curl recurrence in place.

    MEEP step_generic.cpp step_curl, both dsig and dsigu active, no conductivity::

        realnum fprev = fu[i];
        fu[i] = ((kap[k] - sig[k]) * fu[i]
                 - dtdx * (g1[i+s1] - g1[i] + g2[i] - g2[i+s2])) * siginv[k];
        f[i] = siginvu[ku] * ((kapu[ku] - sigu[ku]) * f[i] + fu[i] - fprev);

    ``curl`` already carries the dtdx factor. Both arrays are updated in place,
    so references held by the driver and the monitors stay valid, and the
    broadcast-shaped float32 coefficients are applied without materializing a
    full-volume copy of each.

    ``fprev`` is the only allocation left, and this ran six times a timestep at the top
    of the profile (35% of a 2-D PML step, measured); with a
    :class:`~.fields.StepScratch` the copy goes into a pooled buffer instead. The
    ``previous`` tag is shared with :func:`_apply_constitutive_pml`, whose sub-step
    never overlaps this one.
    """
    fu_previous = fu.copy() if scratch is None else scratch.copy_of("previous", fu)
    fu *= kms
    fu -= curl
    fu *= sinv
    field *= kms_u
    field += fu
    field -= fu_previous
    field *= sinv_u


def _apply_conductive_update(field: Any, curl: Any, condfac: Any, condinv: Any) -> None:  # Lossy D, no PML.
    """Run the D-conductivity curl update in place, MEEP step_generic.cpp:87-97::

        f[i] = ((1 - dt/2*cnd[i]) * f[i] - dtdx * curl) * cndinv[i]

    ``curl`` already carries the dtdx factor, and ``condfac``/``condinv`` are the
    two float32 volumes :meth:`~.fields.Fields.set_d_conductivity` derives. The loss
    is applied to D, not to E: MEEP's sigma_D enters the equations as ``sigma_D * D``
    rather than the textbook ``sigma * E``, so it differs from the usual electric
    conductivity by a factor of epsilon (doc/docs/Materials.md).
    """
    field *= condfac
    field -= curl
    field *= condinv


def _apply_conductive_pml_update(
    xp: Any,
    field: Any,
    curl: Any,
    condfac: Any,
    condinv: Any,
    kms: Any,
    sinv: Any,
    kms_u: Any,
    sinv_u: Any,
    fu: Any,
    f_cond: Any,
    scratch: Any = None,
) -> None:
    """Run MEEP's most-general conductivity + two-axis PML curl recurrence.

    ``step_generic.cpp::step_curl`` updates three histories in order::

        f_cond <- (condfac*f_cond - curl) * condinv
        f_u    <- (kms*f_u + f_cond - f_cond_previous) * sinv
        f      <- (kms_u*f + f_u - f_u_previous) * sinv_u

    ``condinv`` contains the material conductivity only, exactly as the
    ``step_db.cpp`` call site passes ``s->condinv[component][direction]``.
    MEEP partitions a chunk according to whether each of the two transverse
    PML conductivities is active. The four spatial cases below preserve those
    semantics; simply substituting identity coefficients into the general
    recurrence is not equivalent after a source has been injected into D.

    THIS IS THE ENGINE'S MOST EXPENSIVE CURL and it is reached by any run combining a
    material conductivity with a PML — an ``mp.Absorber`` beside a layer, a lossy
    medium anywhere in a PML cell. Measured on a 640x640 cell against the same cell
    with a lossless medium: 19.2-21.4 Mcell-steps/s against 56.7, i.e. ~2.7x the plain
    PML recurrence, because it evaluates all four of MEEP's subchunk cases everywhere
    and selects. Two things paid for that beyond the arithmetic, and both are removed
    here without changing a single selected value:

      * FIVE full-volume ``.copy()`` calls per curl term per step. They now come from
        the :class:`~.fields.StepScratch` pool, under tags of their own — ``previous``
        and the curl/stencil tags are live on this path, so they cannot be shared.
      * THREE ``xp.where`` results, each a fresh full volume read twice and written
        once, to overwrite a selection of cells in an array that already holds the
        other branch. ``copyto(dst, src, where=mask)`` states the same selection and
        allocates nothing. The masks are the same broadcast booleans as before, so
        every cell takes the value it took, bit for bit.
    """
    def _copy(tag: str, array: Any) -> Any:
        return array.copy() if scratch is None else scratch.copy_of(tag, array)

    # MEEP partitions a chunk by which transverse PML directions are active and
    # selects one of four special cases. They are not interchangeable after source
    # injection: the source is written into D, never into f_cond or f_u. The four
    # masks are broadcast-shaped (one entry per plane of the PML axis), not volumes,
    # so they are rebuilt per call rather than cached against a PML that may be
    # replaced under a live Fields.
    dsig_active = (kms != 1.0) | (sinv != 1.0)
    dsigu_active = (kms_u != 1.0) | (sinv_u != 1.0)
    dsig_inactive = xp.logical_not(dsig_active)
    dsigu_inactive = xp.logical_not(dsigu_active)
    field_previous = _copy("cond_field_previous", field)
    f_cond_previous = _copy("cond_f_cond_previous", f_cond)
    fu_previous = _copy("cond_fu_previous", fu)

    # Conductive curl history for cases where the first PML direction is active.
    f_cond *= condfac
    f_cond -= curl
    f_cond *= condinv

    # If only the second PML direction is active, MEEP applies conductivity to f_u
    # directly. If both are active, f_u absorbs the increment of f_cond.
    fu_conductive = _copy("cond_fu_conductive", fu_previous)
    fu_conductive *= condfac
    fu_conductive -= curl
    fu_conductive *= condinv
    fu *= kms
    fu += f_cond
    fu -= f_cond_previous
    fu *= sinv
    xp.copyto(fu, fu_conductive, where=dsig_inactive)

    # Cases with the second PML direction active consume the selected f_u update.
    field *= kms_u
    field += fu
    field -= fu_previous
    field *= sinv_u

    # With only the first direction active, D absorbs f_cond directly.
    first_only = _copy("cond_first_only", field_previous)
    first_only *= kms
    first_only += f_cond
    first_only -= f_cond_previous
    first_only *= sinv

    # With neither direction active, D itself takes the ordinary conductive step.
    direct = field_previous
    direct *= condfac
    direct -= curl
    direct *= condinv

    # The two-way selection the nested `where` expressed: keep the split-recurrence
    # result wherever dsigu is active, and otherwise take `first_only` or `direct`.
    # Both sources are read-only here, so writing into `field` in two passes cannot
    # see its own output.
    xp.copyto(field, first_only, where=dsigu_inactive & dsig_active)
    xp.copyto(field, direct, where=dsigu_inactive & dsig_inactive)
    # Histories do not exist in MEEP's inactive subchunks. Preserve their prior
    # values there so a diagnostic read cannot mistake unused arithmetic for state.
    xp.copyto(f_cond, f_cond_previous, where=dsig_inactive)
    xp.copyto(fu, fu_previous, where=dsigu_inactive)


def _apply_constitutive_pml(field: Any, source: Any, kps: Any, kms: Any, fw: Any,
                            scratch: Any = None) -> None:  # dsigw accumulation.
    """Run the dsigw constitutive PML accumulation in place.

    MEEP step_generic.cpp step_update_EDHB with dsigw active::

        realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
        fw[i] = g[i] * u[i];                   // B for H, D*inv_eps for E
        f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;

    The expression form allocates THREE full-volume temporaries per component per
    step — the ``fwprev`` copy and the two coefficient products — and ran second in
    the profile at 22% of a 2-D PML step. With a :class:`~.fields.StepScratch` all
    three come from the pool: ``previous`` for the copy (shared with
    :func:`_apply_pml_update`, whose sub-step never overlaps this one) and
    ``constitutive`` for the product, which is formed, consumed and re-formed within
    this call.
    """
    if scratch is None:
        fw_previous = fw.copy()
        fw[...] = source
        field += kps * fw
        field -= kms * fw_previous
        return
    xp = scratch.xp
    fw_previous = scratch.copy_of("previous", fw)
    fw[...] = source
    product = scratch.take("constitutive", fw.shape, xp.result_type(kps, fw))
    xp.multiply(kps, fw, out=product)
    field += product
    xp.multiply(kms, fw_previous, out=product)
    field -= product


def _boundary_kinds(grid: "Grid", pml: Optional["PML"]) -> Tuple[str, str, str]:  # Ghost rule per axis.
    """Resolve the ghost-cell rule for each axis: mirror where folded, then the declaration.

    The absorber does not DECIDE any of it — the rule is the symmetry and nothing
    else — but it has to AGREE, so the ``pml`` argument is checked against the fold
    (:func:`_require_consistent_pml`) rather than ignored. MEEP's boundary condition
    comes from ``fields::use_bloch``
    (boundaries.cpp), which every ``mp.Simulation`` carrying a ``k_point`` — zero or
    not — applies to all six faces, and which knows nothing about PML;
    ``structure_chunk::use_pml`` only grades a conductivity on top of it. An axis
    that absorbs therefore still wraps, and the wave leaving one face re-enters at
    the other, where the layer absorbs it.

    This pass used to make an unmirrored X or Y metallic when both of its faces
    carried a layer — the uniform ``PML(thickness)`` case. That was a perfect mirror
    where MEEP has a wrap, and it was the whole residual behind the uniform layer's
    CPU-MEEP floor: 6.31e-04 complex Ez against 4.00e-07 for the identical run with
    the wrap restored, on a case whose z-only sibling already reached 4.5e-07.

    A mirrored axis is the one real exception: its lower face is the symmetry plane,
    which the PML deliberately skips (an absorber there would eat the mirrored half)
    and which reflects rather than repeating. Any of the three axes may carry one.

    THE FOLD OUTRANKS THE DECLARATION, and that is not a tie-break, it is the same
    answer twice. MEEP keeps ``symmetry`` and ``fields::boundaries`` as separate
    mechanisms, so a folded axis still declares an outer condition for the full cell's
    two faces; but the stored quadrant's near face is the plane (mirror ghost) and its
    far face is the cell wall, which this kernel already terminates with the zero
    ghost a PEC wall has. A folded METALLIC axis is therefore exactly what MIRROR
    already implements — and it is the case the zero ghost is EXACT for, which is why
    ``FdtdDriver._require_folded_far_face_is_quiet`` stops refusing it. A folded
    PERIODIC axis stores its second-mirror plane and reflects past it — the
    reflect rule of ``_shift_up`` plus the far-ghost fill
    (``fill_folded_far_ghosts_*``), measured at the CPU-MEEP fold floor with the
    face fully live; only the nonlinear transverse sums still terminate it with
    the zero, which is what narrows ``_require_folded_far_face_is_quiet`` to
    nonlinear runs.

    An INVARIANT axis resolves to PERIODIC and gets no rule of its own: with one cell
    the wrap returns the same plane and the difference is exactly zero, which is what
    MEEP's ``stride(d) = 0`` computes for a direction it does not have. ``Grid``
    refuses the other two conditions there, so this only asserts what it guaranteed —
    cheaply, and where a future caller who reached past ``Grid`` would land.
    """
    _require_consistent_pml(grid, pml)
    kinds = tuple(
        CYL_AXIS if grid.is_axis(axis)
        else MIRROR if grid.is_mirrored(axis)
        else (METALLIC if grid.is_metallic(axis) else PERIODIC)
        for axis in range(3)
    )
    if getattr(grid, "has_invariant", False):
        for axis in range(3):
            if grid.is_invariant(axis) and kinds[axis] != PERIODIC:
                raise ValueError(
                    f"axis {AXIS_NAMES[axis]} is translationally invariant and resolved to "
                    f"{kinds[axis]!r}. An invariant axis carries the periodic wrap and nothing "
                    f"else — that is what makes its finite difference an exact zero rather than "
                    f"a boundary. A {METALLIC!r} one would zero every component whose Yee shift "
                    f"there is 0 and return a smooth exact zero for that whole polarization; a "
                    f"{MIRROR!r} one would reflect a half that does not exist."
                )
    return kinds


def zero_metal_D(fields: "Fields") -> None:  # D-side PEC wall; the driver calls it post-injection.
    """Hold the D samples that lie ON a metallic wall at exactly zero.

    MEEP source: boundaries.cpp (``find_metals``, ``fields_chunk::zero_metal``),
    step.cpp (``step_boundaries`` runs "the metals first" for every field type).

    Call this AFTER the electric sources and before update_E, alongside
    :func:`fill_symmetry_bc_D`: MEEP's order is step_db -> step_source ->
    step_boundaries, so a current deposited onto the wall is injected first and wiped
    second. Getting that order backwards leaves a source sitting on a perfect
    conductor, which radiates.

    :func:`zero_metal_B` is the magnetic half and follows the magnetic sources for the
    same reason — MEEP runs the pair of passes on each side, not once.

    Only the primary fields need it. E and H are reached from D and B by relations
    that map zero to zero — ``E = (D - sum P) * inv_eps`` element-wise, and under PML
    an accumulation of ``fw = (D - sum P) * inv_eps`` values that are themselves zero
    on the wall — and every polarization is driven by that same W, so a wall cell
    starts at zero and is never handed anything else. The curl mask
    (:func:`_mask_non_owned_cells`) is what keeps that true from one step to the next;
    this call is what keeps it true across a source, a seeded initial condition, or
    anything else that writes into D directly.
    """
    _zero_metal(fields, D_COMPONENTS)


def zero_metal_B(fields: "Fields") -> None:  # B-side PEC wall; the driver calls it post-injection.
    """Hold the B samples that lie ON a metallic wall at exactly zero — MEEP's step_boundaries(B_stuff).

    The magnetic half of :func:`zero_metal_D`, and it runs in the same relative slot:
    AFTER the magnetic currents, because MEEP's ``step_db(B_stuff) ->
    step_source(B_stuff) -> step_boundaries(B_stuff)`` puts the wipe after the
    injection (step.cpp:64-72). Running it at the end of ``step_B`` instead — one
    statement earlier, before the driver injects — is invisible in every run whose
    magnetic sources are off the wall, which is every run in this suite bar one: a
    mutation that deleted the call entirely survived the whole CPU-MEEP metallic
    cross-validation, and only an Hx current placed ON an x wall separates the two.
    """
    _zero_metal(fields, B_COMPONENTS)


def _zero_metal(fields: "Fields", components: Sequence[str]) -> None:  # Zero the wall plane of each component.
    """Write zero into stored cell 0 of every component that sits ON a metallic wall.

    A component lies on the wall of axis ``d`` exactly when its Yee shift there is 0
    (MEEP ``on_metal_boundary``: the sample's doubled coordinate equals the little or
    big corner, which only a shift-0 component can reach). For an x-normal wall that
    is Ey, Ez, Dy, Dz, Hx, Bx — tangential E and normal B, the two families a perfect
    electric conductor short out.

    Stored cell 0 is the LOW wall. The HIGH wall is MEEP's slot ``N``, which this
    engine does not store: its value is supplied as the zero ghost by
    :func:`_shift_up`, so the pair of walls is complete without the extra plane.

    A FOLDED METALLIC AXIS HAS NO STORED WALL AT ALL, and writing one is not a
    harmless extra zero. ``grid_volume::halve`` moves that axis's origin to
    ``io = -2`` (vec.cpp), so its stored cell 0 sits one full cell BELOW the mirror
    plane and holds the parity-weighted ghost ``fill_symmetry_bc_D`` writes there —
    not a wall. Both of the axis's real walls are elsewhere: the near one is the
    mirror plane, which is not a wall, and the far one is the plane at ``L/2`` that a
    folded axis does not store and :func:`_shift_up` supplies as zero. Clearing cell 0
    there destroys the fold instead of imposing a boundary: measured on a 3x1x2 cell
    at resolution 10 with an X mirror and metallic walls, complex relative L2 against
    CPU MEEP 1.28e+00 with the fold's ghost zeroed against 2.0e-07 with this skip in
    place — the same magnitude as running the wrong boundary condition entirely, and
    just as smooth.
    """
    grid = fields.grid
    if not getattr(grid, "has_metallic", False):
        return  # No wall anywhere: not one plane is touched, and periodic runs stay byte-identical.
    walled = [
        axis for axis in range(3) if grid.is_metallic(axis) and not grid.is_mirrored(axis)
    ]
    if not walled:
        return
    # RAW, then ONE batched clear for the whole pass. This pass writes one wall FACE
    # per component and reads nothing: under a held residency a barriered fetch once
    # synced six volumes a step down and back up for slabs of zeros, and then a door
    # a face paid a device launch each (nine a ``pml_3d`` step). ``zero_faces`` is,
    # on the array path, exactly ``array[_face(axis, 0)] = 0`` per request in this
    # order; under a residency, one device launch per owner, and nothing moves.
    requests = []
    for name in components:
        array = host_writes.raw(fields, name)
        if array is None:
            continue  # An optional store (H without PML, E without dispersion) is served on demand.
        shifts = IYEE_SHIFTS[name]
        for axis in walled:
            if shifts[axis] == 0:
                requests.append((array, axis, 0))
    host_writes.zero_faces(requests)


def _require_consistent_pml(grid: "Grid", pml: Optional["PML"]) -> None:
    """Refuse an absorber that disagrees with the grid about which axes are folded.

    A PML table decides two things from the fold: it skips the face the mirror plane
    occupies, and it grades cell 0 from the near wall rather than treating it as the
    periodic image of the far one (``pml._graded_sigma``'s ``wraps`` flag). Both are
    silent when wrong — a layer written across the mirror plane absorbs the half of
    the domain that is supposed to be reconstructed, and a folded cell 0 graded as a
    wrapped one takes the FULL conductivity of the opposite wall right on the plane.
    Neither raises, neither diverges, and both return a smooth field.

    The two halves can only disagree while something reads the fold from a narrower
    set of axes than ``Grid`` does; this is the tripwire for exactly that, and it
    costs nothing on a layer that already agrees.

    Only an axis that actually absorbs is checked. A folded axis with no layer on
    it has sigma = 0 on every cell whatever the table believes about its wrap
    (``_graded_sigma`` never enters the grading loop), so refusing there would
    reject a run that is exactly right.
    """
    if pml is None:
        return
    for axis in range(3):
        if not grid.is_mirrored(axis) or not pml.axis_has_pml(axis):
            continue
        low, _high = pml.axis_faces(axis)
        if low:
            raise ValueError(
                f"the PML puts {low} cells on the low {AXIS_NAMES[axis]} face, which is the "
                f"mirror plane of this folded axis: an absorber there eats the half of the "
                f"domain the fold reconstructs. Ask for the high face only."
            )
        if pml.axis_wraps(axis):
            raise ValueError(
                f"the PML treats the {AXIS_NAMES[axis]} axis as wrapping while the grid folds "
                f"it with a mirror plane. A folded axis does not wrap — its cell 0 straddles "
                f"the plane instead of imaging the far face — so the two disagree about which "
                f"wall grades cell 0. Teach the PML this axis can fold before folding it."
            )


def _mirror_phases(grid: "Grid") -> MirrorPhases:  # Declared phase of each axis's plane, None where absent.
    """Per-axis mirror phase, straight from the grid — the parities' one input.

    ``+1`` (even) or ``-1`` (odd) on a folded axis and ``None`` on an unfolded one.
    Resolved once per sub-step and threaded through, so a component's parity is
    always ``mirror_parity(component, axis, declared phase)`` and never the
    even-mirror default standing in for a plane that declared otherwise — which
    would run an even fold for a run asked to be odd, complete, and be wrong by
    twice the field wherever the parity mattered.

    The cylindrical radial axis rides in the same slot with ``(-1)^m`` — the
    overall factor of MEEP's ``r_to_minus_r_symmetry(m)`` — which
    ``_shift_down``'s AXIS branch multiplies onto the per-component direction
    sign. One slot, because the two are the same kind of fact: the phase the
    below-face image carries.
    """
    return tuple(
        (-1.0) ** grid.m if grid.is_axis(axis) else grid.mirror_phase(axis)
        for axis in range(3)
    )


def _bloch_phases(grid: "Grid", boundaries: Tuple[str, str, str], sample: Any,
                  pml: Optional["PML"] = None) -> BlochPhases:
    """Resolve the Bloch wrap factor for each axis, refusing the boundaries that cannot carry one.

    A Bloch phase is a property of a *wrap*: it says what the field is one lattice
    vector away. A mirrored axis does not wrap — its plane reflects, which forces the
    field to be even or odd about it and therefore k = 0 — so a nonzero k there has
    nowhere to apply. Silently dropping it would run a plain symmetric boundary while
    the caller believed they had set a Bloch condition, and the result would look
    entirely reasonable, so it raises instead.

    An axis that carries an absorber is NOT refused, and used to be. The stance was
    that the layer terminates the wave, so the field one lattice vector up is whatever
    the absorber left rather than ``exp(i*2*pi*k*L)`` times the field here. MEEP takes
    the opposite view and it is the right one: ``use_bloch`` makes every direction
    Periodic as soon as any ``k_point`` is given, and a PML is a graded material
    underneath that wrap, not a replacement for it. So the axis does repeat, the wrap
    does carry the phase, and the absorber simply means little survives to use it.

    Settled by measurement rather than by argument, because the two readings are
    equally sayable: stepping the refused configuration — ``k_point=(0.3, 0, 0)`` with
    a PML on x AND z — reproduces CPU MEEP's complex Ez at **2.82e-07**, against
    2.63e-07 for the same run with the x absorber removed. The refusal was costing
    seven scripts of MEEP's own example corpus, ``pw-source.py`` among them, for a
    divergence that does not exist.

    The complex-storage requirement is checked once here against ``sample`` (any of
    the sub-step's source components) rather than per wrapped plane, so a real-field
    run is refused before the first curl rather than at the first phased face.
    """
    if not grid.has_bloch:  # The common case: no phase objects built, no per-axis work.
        return (None, None, None)
    phases: list[Optional[complex]] = []
    for axis in range(3):
        phase = grid.bloch_phase(axis)
        if phase is not None and boundaries[axis] != PERIODIC:
            reflects = (
                "a mirror plane reflects rather than repeating, which forces the field even or "
                "odd about it and so k = 0"
                if boundaries[axis] == MIRROR
                else "a perfect-electric-conductor wall terminates the axis instead of repeating "
                     "it, so there is no lattice vector for the phase to describe"
            )
            raise ValueError(
                f"k_point component {AXIS_NAMES[axis]} = {grid.k_point[axis]!r} needs a wrapping "
                f"boundary, but axis {AXIS_NAMES[axis]} resolved to {boundaries[axis]!r} — "
                f"{reflects}. Drop that boundary on this axis or set its k component to 0."
            )
        phases.append(phase)
    if any(phase is not None for phase in phases) and sample.dtype.kind != "c":
        raise ValueError(
            f"Bloch boundaries need complex fields, but the field storage is {sample.dtype}; "
            f"build the Fields with force_complex_fields=True or use k_point = (0, 0, 0)."
        )
    return (phases[0], phases[1], phases[2])


def _component_snapshot(fields: "Fields", names: Sequence[str]) -> Dict[str, Any]:  # Read the curl inputs once.
    """Read each source component once per sub-step.

    Without PML ``get_E`` allocates a fresh D*inv_eps array on every call, so
    reading per curl term would trade three allocations for six — and the three it
    does make are pooled here (:class:`~.fields.StepScratch`), because the sub-step
    reads them and drops them. That pooling stays INSIDE the stepper rather than
    moving into ``get_E``, whose documented contract is that a caller may mutate the
    array it returns; the curl never does.
    """
    return {name: _read_component(fields, name) for name in names}


def _read_component(fields: "Fields", name: str) -> Any:  # Fetch E or H from storage or on demand.
    if name in ("Ex", "Ey", "Ez"):
        if fields.stores_E or fields.scratch is None:
            return fields.get_E(name)
        # The derived branch of get_E, into this sub-step's own buffer. Read-only
        # for as long as the term loop holds it: the curl differences it, BFAST
        # sums it, and nothing in either writes back through it.
        scratch = fields.scratch
        displacement = getattr(fields, "D" + name[1])
        inverse = fields.inverse_epsilon_for(name)
        target = scratch.take("snapshot_" + name, displacement.shape,
                              scratch.xp.result_type(displacement, inverse))
        scratch.xp.multiply(displacement, inverse, out=target)
        return target
    if name in ("Hx", "Hy", "Hz"):
        return fields.get_H(name)
    raise ValueError(f"the curl sub-steps difference E and H components only, got {name!r}")


def _symmetry_phase(component: str, axis: int, mirror_phase: Optional[int]) -> int:  # Parity about one plane.
    """One component's parity about the plane on ``axis`` — ``fields.mirror_parity``.

    ``mirror_phase`` is the plane's declared phase and must not be None here: this
    is only ever reached on a MIRROR boundary, and an axis resolved to MIRROR has a
    plane by construction (``_boundary_kinds``). None means the two disagreed, which
    would otherwise silently fold with the even-mirror default.
    """
    if mirror_phase is None:
        raise ValueError(
            f"axis {AXIS_NAMES[axis]} resolved to a mirror boundary but carries no mirror "
            f"plane; the boundary kinds and Grid.mirror_phase are out of step."
        )
    return mirror_parity(component, axis, mirror_phase)


def _curl_coefficients(pml: "PML", axis: str, half_integer: bool) -> Tuple[Any, Any]:  # Curl PML pair.
    """Return (kap-sig, 1/(kap+sig)) for a curl sub-step on one axis.

    The B curl reads half-integer positions and the D curl integer ones; getting
    that pairing backwards is a silent half-cell error, not a crash.
    """
    suffix = _coefficient_suffix(axis, half_integer)
    return getattr(pml, "kms" + suffix), getattr(pml, "sinv" + suffix)


def _constitutive_coefficients(pml: "PML", axis: str, half_integer: bool) -> Tuple[Any, Any]:  # update_eh pair.
    """Return (kap+sig, kap-sig) for a constitutive update on one axis.

    update_eh reads integer positions for H and half-integer ones for E.
    """
    suffix = _coefficient_suffix(axis, half_integer)
    return getattr(pml, "kps" + suffix), getattr(pml, "kms" + suffix)


def _coefficient_suffix(axis: str, half_integer: bool) -> str:  # PML attribute suffix, e.g. "_y" or "_y_h".
    if axis not in AXIS_NAMES:
        raise ValueError(f"PML axis must be one of {AXIS_NAMES}, got {axis!r}")
    return f"_{axis}_h" if half_integer else f"_{axis}"


def _pml_is_active(pml: Optional["PML"]) -> bool:  # A layer with no absorbing face is the same as no layer.
    """True when this layer absorbs on at least one of its six faces.

    A table whose every face is zero carries sigma = 0 everywhere, so the split-field
    recurrence would reduce to the plain update — but only to within the round-off of
    multiplying by 1.0 and adding a difference of two auxiliaries. Taking the no-PML
    path instead makes it bit-identical to ``pml=None``.
    """
    return pml is not None and pml.is_active


def _require_pml_storage(fields: "Fields", names: Sequence[str]) -> None:  # Precondition, not a mode switch.
    """Fail loudly when the PML sub-steps are asked to run without their storage.

    The driver owns the one-way switch (``Fields.enable_pml_storage``) because it
    also changes what get_E/get_H return; stepping asserts the precondition
    rather than silently flipping the mode mid-run.
    """
    unallocated = [name for name in names if getattr(fields, name, None) is None]
    if unallocated:
        raise RuntimeError(
            "PML stepping requires Fields.enable_pml_storage() before the first step; "
            f"unallocated: {', '.join(unallocated)}"
        )


def _require_field_storage(fields: "Fields", names: Sequence[str]) -> None:  # Stored E precondition.
    """Fail loudly when the constitutive sub-step is asked to store E without its arrays.

    The driver owns the one-way switch (``Fields.enable_field_storage``), which
    ``add_susceptibility`` calls, because it also changes what get_E returns;
    stepping asserts the precondition rather than flipping the mode mid-run.
    """
    unallocated = [name for name in names if getattr(fields, name, None) is None]
    if unallocated:
        raise RuntimeError(
            "Dispersive stepping requires Fields.enable_field_storage() before the first step; "
            f"unallocated: {', '.join(unallocated)}"
        )


def _face(axis: int, index: int) -> Tuple[Any, ...]:  # Index tuple selecting one slab of a 3-D array.
    return (slice(None),) * axis + (index,)


def _span(axis: int, start: int, stop: int) -> Tuple[Any, ...]:  # Index tuple selecting a run of slabs.
    return (slice(None),) * axis + (slice(start, stop),)
