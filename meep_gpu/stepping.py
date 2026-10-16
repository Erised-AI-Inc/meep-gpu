"""
Leapfrog stepping for the MEEP-compatible FDTD engine — the four MEEP sub-steps
that advance the Yee lattice by one timestep, plus the mirror-plane repairs that
MEEP's owned-cell loop leaves to the boundary pass.

``step_B`` and ``step_D`` apply the discrete curl to the primary fields: forward
differences for B, and MEEP's negated strides (backward differences) for D, so
that a single ``field -= curl`` expression produces dB/dt = -curl(E) and
dD/dt = +curl(H). Where a PML layer is present the same pass takes MEEP's form
PER STRUCTURE CHUNK (step_db.cpp:56-59): the split-field auxiliary recurrence on
the cells whose chunk carries the component's dsigu layer, the dsig-only form on
the field alone everywhere else, read from the run's
:class:`~.chunk_map.RegionPlan`. ``update_H`` and ``update_E`` close the
constitutive relations: without PML they are no-ops, because H = B and
E = D*inv_eps are served on demand by ``Fields``; with PML they accumulate the
stored fields through the third PML direction on the cells whose chunk carries
that layer (update_eh.cpp:162-163), and store H = B, E = D*inv_eps elsewhere. An
auxiliary (``fu_*``, ``f_w_*``, ``f_cond_*``) is read and written only on its own
form's cells. ``fill_symmetry_bc_D`` is
exported for the driver to call after source injection, matching MEEP's
step_db -> step_source -> step_boundaries order; the B-side counterpart runs at
the end of ``step_B``.

The two FOLDED INCREMENTS are MEEP's own passes after the curl, not terms inside it.
``step_db`` (step_db.cpp:47-175) runs, per component, ``STEP_CURL`` and then
``STEP_BFAST`` on the same component (:124-142), and only after every component has
been through both, the special-kz beta pass over Bx and By (or Dx and Dy) (:161-175).
Each pass adds its increment through its own additive ladder (step_generic.cpp:255-333
beta, :335-533 BFAST), and two additions into one word do not commute in rounding, so
the order on every cell is curl form, then BFAST ladder, then beta ladder
(:func:`_apply_folded_increment`). In a 2-D run, the only one that carries beta, the two
passes add nonzero increments to disjoint components — BFAST's to the z components, since
a one-partner component's BFAST k is zero (:func:`_bfast_term`), and beta's to x and y —
so there the order can reach a signed zero and nothing else. The cylindrical i*m/r
coupling is a third such pass (step_db.cpp:177-294), after every component's curl and
through the same ladder (:func:`_cylindrical_imr_increment`), and the cylindrical radial
derivative is MEEP's prefix sum restarted in every structure chunk
(:func:`cylindrical_rderiv_difference`).

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
Bloch phase (refused on real storage in :func:`_bloch_phases`), a quarter-turn
coupling (special-kz beta, cylindrical i*m/r) or a complex source amplitude
(refused in ``sources``). Keeping the coefficient arrays float32 in BOTH modes is
what makes that true: a complex64 coefficient would multiply the real part by the
imaginary part of the field and the equivalence would be gone.

COMPLEX STORAGE IS STEPPED PART BY PART, as MEEP steps its two ``realnum`` arrays
(the ``DOCMP`` loops): every product on a field goes through :mod:`~.part_products`
-- a real coefficient times each part, a quarter turn as a swap of parts, a Bloch
connection as MEEP's separately rounded ``complex<float>`` product. NumPy's and
CuPy's complex multiply contracts to a fused multiply-add on hosts that have one and
flips zero signs through its ``(k + 0j)`` cross terms, so the array path forms none
(``parity/meep_gpu/complex_product_interceptor.py`` checks it at run time).

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
  Bloch-periodic when ``grid.k_point`` gives that axis a phase. Then the engine
  stores MEEP's owned window: stored cell 0 of a component with Yee shift 0 on the
  axis is the owned top point ``lc + 2N`` (``Grid.stored_window_low``), the shifts
  stay plain rolls, and the reads that leave the window -- the same-index read half
  a cell below it (``conj(eikna)``) or above it (``eikna``) -- take MEEP's not-owned
  word (:func:`_bloch_ghost_operand`).

A REDUCED-DIMENSION run adds no fourth boundary rule. MEEP 2-D and 1-D are
translational invariance along the missing axes, and this engine gives such an axis
ONE cell and the ordinary periodic wrap, where ``roll`` returns the same plane. A
one-pixel periodic direction is what ``fields::nosize_direction`` calls a
lower-dimensional emulation, and MEEP's genuine 2-D run and the same problem as a 3-D
cell one pixel deep in z step to **0.0e+00** of each other.

The CURL does take a different form there, because MEEP's does. ``figure_out_step_plan``
(fields.cpp:428-453) records a curl partner only when its derivative direction is a
direction of the grid, and ``step_curl`` (step_generic.cpp:72-76, :104, :108) then
evaluates ``dtdx * (g1[i + s1] - g1[i])`` with ``g2`` absent and
``(-dtdx) * (g2[i + s2] - g2[i])`` with ``g1`` absent. The full bracket in MEEP's left
to right order, ``((g1s - g1) + g2) - g2s``, is NOT those forms when ``g2s == g2``:
``(a + g2) - g2`` rounds. So :func:`_curl_from_operands` takes the partner presence
(:func:`curl_presence`, ``grid.is_invariant`` of the partner's derivative axis) and
evaluates the form MEEP evaluates.

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
k = 0) and on a metallic one; an axis carrying an absorber still wraps and carries
the phase, as MEEP's does (:func:`_bloch_phases`). Bloch also needs complex storage
— a real field cannot carry the phase — and that too is refused loudly rather than
silently dropping the imaginary part.

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

from . import host_writes, part_products
from .fields import IYEE_SHIFTS, mirror_parity
from .readback import cmul

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

# MEEP step_db.cpp:177-294, the cylindrical i*m/r pass: per (target term, partner, sign)
# with ``c_g = d_c == R ? plus_component : minus_component`` and the sign
# ``(1-2*(ft==B_stuff)) * (1-2*(d_c==R))`` of ``the_m`` (package x is R, z is Z).
_CYLINDRICAL_IMR_B = ((B_CURL_TERMS[0], "Ez", +1.0), (B_CURL_TERMS[2], "Ex", -1.0))
_CYLINDRICAL_IMR_D = ((D_CURL_TERMS[0], "Hz", -1.0), (D_CURL_TERMS[2], "Hx", +1.0))

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
        # the prefix sum of (1/r)d(r*Ep)/dr, summed per structure chunk. Ep sits at the
        # node (r-shift 0), so ir0 = 0.0 where the D side's Hp gave 0.5.
        #
        # THE PREFIX IS A CUMSUM, NOT A FIELD, so the generic shift machinery's
        # metallic far ghost is wrong for it: a zero ghost turns the last row's
        # forward difference into MINUS THE WHOLE ACCUMULATED SUM. MEEP stores
        # the wall row (nr + 1 rows) and its Bz reads prefix[wall] - prefix[last],
        # with the wall's Ep held at zero by zero_metal — a LOCAL value. The same
        # thing here: the prefix runs over Ep extended by its zero wall row, and
        # Bz's radial difference is taken of consecutive extended rows directly.
        # Measured before the wall row: the m = 1 run's first divergence was born at the
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
        # cylindrical path — which holds `radial`, the one pooled buffer carried
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
        radial = cylindrical_rderiv_difference(
            xp, extended, 0.0, _cylindrical_scan_starts(fields, pml if pml_active else None, "Bz"),
            rows, backward=False, scratch=scratch)
        axis_increment = _cylindrical_axis_increment_B(fields, electric, boundaries,
                                                       phases, dtdx)
    bfast = grid.bfast_active
    for term in B_CURL_TERMS:
        presence = curl_presence(grid, term)
        if cylindrical and term.target == "Bz":
            # Bz's whole curl in cylindrical: the minus partner is the phi derivative
            # MEEP drops (step_db.cpp:86-92; its i*m/r part is a pass of its own below),
            # so the curl IS dtdx times the radial forward difference of the per-chunk
            # prefix. A cylindrical grid carries no BFAST, so no operands are gathered.
            operands = None
            curl = part_products.scale(dtdx, radial, out=radial)
        else:
            operands = _curl_operands(grid.xp, electric, term, boundaries, phases, parities,
                                      backward=False, reflect_rows=reflect_rows,
                                      scratch=scratch, gather=presence)
            curl = _curl_from_operands(operands, dtdx, presence, scratch=scratch)
        _mask_non_owned_cells(curl, grid, term.iyee)
        if axis_increment is not None and axis_increment[0] == term.target:
            # The axis-row dfcnd, in curl sign convention (the step subtracts).
            curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)
        # B sits at half-integer positions in the transverse directions.
        _apply_curl(fields, term, curl, pml, pml_active, half_integer=True, scratch=scratch)
        if bfast:
            # step_db.cpp:129-142 — STEP_BFAST on this component right after its
            # STEP_CURL, on the SAME (already gathered) partners and the same PML and
            # conductivity slices, through its own ladder.
            _apply_folded_increment(
                fields, term,
                _bfast_term(fields, term, operands, magnetic=True,
                            plain=_bfast_plain_cells(fields, term, presence, pml, pml_active)),
                pml, pml_active, half_integer=True)
    if grid.beta != 0.0:
        # step_db.cpp:161-175 — the B-side out-of-plane couplings, a pass of their own
        # after every component's curl. `cc` runs over d_c in {X, Y} only and its
        # partner is `direction_component(Ex, other)`: Bx <- Ey at sign +1, By <- Ex at
        # sign -1. Bz gets nothing. The pass loops over MEEP's owned cells only.
        for term, partner, sign in ((B_CURL_TERMS[0], "Ey", +1.0), (B_CURL_TERMS[1], "Ex", -1.0)):
            increment = _special_kz_beta_term(fields, electric[partner], sign, magnetic=True)
            _mask_non_owned_cells(increment, grid, term.iyee)
            _apply_folded_increment(fields, term, increment, pml, pml_active, half_integer=True)
    if cylindrical and grid.m != 0:
        # step_db.cpp:177-294 — the B-side i*m/r couplings, a pass of their own after every
        # component's curl: Br takes its dropped plus partner Ez, Bz its dropped minus
        # partner Er; the_m's (1-2*(ft==B_stuff)) flips both signs relative to the D side.
        # The pass loops over MEEP's owned cells only (LOOP_OVER_VOL_OWNED0).
        for term, partner, sign in _CYLINDRICAL_IMR_B:
            increment = _cylindrical_imr_increment(fields, term.target, electric[partner], sign)
            _mask_non_owned_cells(increment, grid, term.iyee)
            _apply_folded_increment(fields, term, increment, pml, pml_active, half_integer=True)
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
        # prefix sum of (1/r)d(r*Hp)/dr, not of Hp itself, summed per structure chunk;
        # the unmodified curl of that prefix is the cylindrical derivative. ir0 = 0.5 =
        # half of Hp's r-shift. The D side never needs a wall row: its backward
        # difference reads rows g and g - 1.
        radial = cylindrical_rderiv_difference(
            grid.xp, magnetic["Hy"], 0.5,
            _cylindrical_scan_starts(fields, pml if pml_active else None, "Dz"),
            magnetic["Hy"].shape[0], backward=True, scratch=scratch)
        axis_increment = _cylindrical_axis_increment_D(fields, magnetic, boundaries,
                                                       phases, dtdx)
    bfast = grid.bfast_active
    for term in D_CURL_TERMS:
        presence = curl_presence(grid, term)
        if cylindrical and term.target == "Dz":
            # Dz's whole curl in cylindrical: the minus partner is the phi derivative
            # MEEP drops (step_db.cpp:86-92; its i*m/r part is a pass of its own below),
            # so the curl IS dtdx times the radial backward difference of the
            # per-chunk prefix, ``dtdx * (F[g-1] - F[g])`` under D's negated stride.
            operands = None
            curl = part_products.scale(dtdx, radial, out=radial)
        else:
            operands = _curl_operands(grid.xp, magnetic, term, boundaries, phases, parities,
                                      backward=True, scratch=scratch, gather=presence)
            curl = _curl_from_operands(operands, dtdx, presence, scratch=scratch)
        _mask_non_owned_cells(curl, grid, term.iyee)
        if axis_increment is not None and axis_increment[0] == term.target:
            # The axis-row dfcnd, in curl sign convention (the step subtracts).
            curl[_face(0, 0)] = -axis_increment[1]
        # D sits at integer positions in the transverse directions.
        _apply_curl(fields, term, curl, pml, pml_active, half_integer=False, scratch=scratch)
        if bfast:
            # step_db.cpp:129-142, D side: the same pass with k1 and k2 both negated,
            # which is what turns +d/dt(k x E) on the B side into -d/dt(k x H) here.
            _apply_folded_increment(
                fields, term,
                _bfast_term(fields, term, operands, magnetic=False,
                            plain=_bfast_plain_cells(fields, term, presence, pml, pml_active)),
                pml, pml_active, half_integer=False)
    if grid.beta != 0.0:
        # step_db.cpp:161-175 — the D-side out-of-plane couplings, after every
        # component's curl. `cc` runs over d_c in {X, Y} only and its partner is
        # `direction_component(Hx, other)`: Dx <- Hy at sign +1, Dy <- Hx at sign -1.
        # Dz gets nothing.
        for term, partner, sign in ((D_CURL_TERMS[0], "Hy", +1.0), (D_CURL_TERMS[1], "Hx", -1.0)):
            increment = _special_kz_beta_term(fields, magnetic[partner], sign, magnetic=False)
            _mask_non_owned_cells(increment, grid, term.iyee)
            _apply_folded_increment(fields, term, increment, pml, pml_active, half_integer=False)
    if cylindrical and grid.m != 0:
        # step_db.cpp:177-294 — the D-side i*m/r couplings, after every component's curl:
        # Dr takes +im/r*Hz (its dropped plus partner), Dz takes -im/r*Hr (its dropped
        # minus partner); signs from the_m's (1-2*(d_c==R)) with ft = D.
        for term, partner, sign in _CYLINDRICAL_IMR_D:
            increment = _cylindrical_imr_increment(fields, term.target, magnetic[partner], sign)
            _mask_non_owned_cells(increment, grid, term.iyee)
            _apply_folded_increment(fields, term, increment, pml, pml_active, half_integer=False)
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

    UNDER A PML THE FORM IS CHOSEN PER STRUCTURE CHUNK, as MEEP chooses it
    (step_db.cpp:56-59): the run's :class:`~.chunk_map.RegionPlan` names, per
    component, the cells whose owning chunk carries the dsigu layer. There the
    split-field recurrence runs (:func:`_apply_pml_update`); everywhere else the
    dsig-only form ``f = ((kms*f) - curl)*sinv`` runs on the field alone
    (:func:`_apply_dsig_only_update`), which IS MEEP's plain ``f - curl`` wherever
    the dsig coefficients are exactly 1 — the plan's precondition, checked when it
    is built. ``fu`` is read and written only on the recurrence's cells
    (:func:`_apply_curl_regions`). With a conductivity all four of MEEP's conductive
    forms are selected on the dsig and dsigu memberships, in the chunks that keep the
    component's conductivity, and the two forms above everywhere else
    (:func:`_apply_conductive_pml_update`).
    """
    target = getattr(fields, term.target)
    condfac = fields.condfac_for(term.target)
    if pml_active:
        kms, sinv = _curl_coefficients(pml, term.dsig, half_integer=half_integer)
        kms_u, sinv_u = _curl_coefficients(pml, term.dsigu, half_integer=half_integer)
        fu = getattr(fields, "fu_" + term.target)
        plan = _region_plan(fields, pml)
        if condfac is not None:
            f_cond = getattr(fields, "f_cond_" + term.target)
            if f_cond is None:
                raise RuntimeError(
                    "Conductivity + PML stepping requires Fields to allocate "
                    f"f_cond_{term.target} before the first step."
                )
            xp = fields.grid.xp
            sigma = fields.conductivity_for(term.target)
            every_chunk = all(plan.conductive_chunks(term.target, sigma))
            _apply_conductive_pml_update(
                xp,
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
                dsig_active=plan.axis_mask(AXIS_NAMES.index(term.dsig), term.target, xp),
                dsigu_active=plan.mask("curl", term.target, xp),
                conductive=None if every_chunk else plan.conductive_mask(term.target, sigma, xp),
                scratch=scratch,
            )
        else:
            _apply_curl_regions(plan, term.target, target, curl, kms, sinv, kms_u, sinv_u, fu,
                                fields.grid.xp, scratch)
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

    MEEP forms ``dfcnd = (sd*Courant) * (f_p[i] - f_p[i - sd] - f_m_mult*f_m[i])``
    (:377): the bracket in ``realnum`` left to right, then a DOUBLE product with the
    double ``Courant``, rounded once (:func:`_realnum_double_product`). The curl's
    ``dtdx``, a ``realnum``, is not that factor: it is the same word only where
    ``Courant`` is a float32 word (0.5 is; 0.3 is not), so it is not read here.
    """
    del dtdx  # the axis rule multiplies by the double Courant (grid.courant)
    grid = fields.grid
    if grid.m == 0:
        # NOT folded into the curl: measured, the fold breaks m=0 under PML
        # (Er 2.7e-01 / Hp 4.5e-01 where the post-add is 1.7e-3.6e-07). The
        # split-field routing (curl -> fu -> field) is not MEEP's axis ladder
        # for this rule; the plain post-add is exact because Dz's dsig is R,
        # whose sigma is zero on the axis row. Applied in _cylindrical_axis_zero_D.
        return None
    if abs(grid.m) == 1:
        # sd = +1 on the D side: f_p = Hr, f_m = Hz, f_m_mult = 2.
        hr = _bloch_ghost_operand(grid.xp, magnetic["Hx"], 2, boundaries[2], phases[2], below=False)
        hr_below = _shift_down(grid.xp, magnetic["Hx"], 2, boundaries[2], "Hx", None)
        bracket = (hr[_face(0, 0)] - hr_below[_face(0, 0)]) \
            - part_products.scale(2.0, magnetic["Hz"][_face(0, 0)])
        return "Dy", _realnum_double_product(+1.0 * float(grid.courant), bracket)
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
        # sync a whole volume. The correction is a face read-modify-write: MEEP's
        # ``dfcnd = g[i] * (Courant * 4)`` is a DOUBLE product rounded once (:324),
        # then ``the_f + dfcnd`` in realnum (its kap - sig and siginv are exactly 1
        # on the axis row, Dz's dsig being R and its dsigu P).
        hy = host_writes.raw(fields, "Hy" if fields._pml_active else "By")
        dz = host_writes.raw(fields, "Dz")
        source = host_writes.face_index(hy.shape, 0, 0)
        target = host_writes.face_index(dz.shape, 0, 0)
        dfcnd = _realnum_double_product(4.0 * float(grid.courant),
                                        host_writes.gather(hy, source))
        host_writes.scatter(dz, target, host_writes.gather(dz, target) + dfcnd)
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

    MEEP's arithmetic, per part (step_db.cpp:377 with sd = -1): the bracket
    ``(Ep[i] - Ep[i+1]) - ((1-2*cmp)*m) * Ez_other[row 1]`` in ``realnum``, left to
    right, then the DOUBLE product with ``-Courant``, rounded once
    (:func:`_realnum_double_product`). ``((1-2*cmp)*m) * Ez_other`` per part is
    ``(m*Ez.im, (-m)*Ez.re)``, the negated quarter turn by ``m``. The curl's
    ``dtdx`` is not read, as on the D side.
    """
    del dtdx  # the axis rule multiplies by the double Courant (grid.courant)
    grid = fields.grid
    if abs(grid.m) != 1:
        return None
    ep = _bloch_ghost_operand(grid.xp, electric["Ey"], 2, boundaries[2], phases[2], below=True)
    ep_above = _shift_up(grid.xp, electric["Ey"], 2, boundaries[2])
    ez_off_axis = grid.xp.take(electric["Ez"], 1, axis=0)
    partner = -part_products.quarter_turn(float(grid.m), ez_off_axis)  # f_m_mult * f_m, per part
    bracket = (ep[_face(0, 0)] - ep_above[_face(0, 0)]) - partner
    return "Bx", _realnum_double_product(-1.0 * float(grid.courant), bracket)


def _realnum_double_product(factor: float, values: Any) -> Any:
    """``realnum(double(factor) * double(v))`` per stored part: a double product rounded once.

    The form of MEEP's axis rules (step_db.cpp:324, :377), where a ``realnum`` operand
    meets the double ``Courant``: C promotes the float to double, multiplies in double and
    rounds the result to ``realnum`` once. A float32 ``factor`` times the operand rounds
    the factor first; the two are the same word only where ``factor`` is a float32 word.
    """
    xp = part_products.array_module(values)
    if values.dtype.kind != "c":
        return (values.astype(xp.float64) * factor).astype(values.dtype)
    out = xp.empty(values.shape, dtype=values.dtype)
    out.real[...] = (values.real.astype(xp.float64) * factor).astype(values.real.dtype)
    out.imag[...] = (values.imag.astype(xp.float64) * factor).astype(values.imag.dtype)
    return out


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


def _cylindrical_imr_increment(fields: "Fields", target: str, partner_values: Any,
                               sign: float) -> Any:
    """The i*m/r pass's increment — MEEP step_db.cpp:177-294 — negated, per part.

    MEEP adds, per real/imaginary part ``cmp``, over its owned cells
    (``LOOP_OVER_VOL_OWNED0``) and after every component's curl::

        the_m = 2*m * (1-2*cmp) * (1-2*(ft==B_stuff)) * (1-2*(d_c==R)) * Courant
        rinv  = the_m / here.r()
        raw   = rinv * g[i]          with g = f[c_g][1 - cmp], the partner's OTHER part

    ``the_m`` is an integer product times the double ``Courant``, rounded once to
    ``realnum``; ``here.r()`` is the target's doubled ILOC r coordinate, an int, so
    ``rinv`` is a ``realnum`` QUOTIENT ``the_m / float(2*ir + iyee_r)`` (the factor 2 in
    ``the_m`` is the doubled-units compensation), not a double ``2m*Courant/r`` rounded
    once. ``sign`` is ``(1-2*(ft==B)) * (1-2*(d_c==R))``. The ``(1-2*cmp)`` makes the
    imaginary part's ``rinv`` the exact negation of the real part's, so the two rows are
    ``re += rinv*g.im`` and ``im += (-rinv)*g.re``: a quarter turn of the parts by the
    real-part ``rinv`` (:func:`~.part_products.quarter_turn`), never a complex product.
    Complex storage is REQUIRED — the coupling has no real-storage form, which is MEEP's
    own rule (``change_m`` aborts on real fields for m != 0).

    Returned in CURL SIGN CONVENTION, ``-raw``, for :func:`_apply_folded_increment`, which
    SUBTRACTS it through MEEP's additive ladder: the pass is increment-only (no decay of
    ``the_f``), keyed on the chunk's ``dsig`` and ``dsigu`` layers exactly as the curl of
    the same component keys. MEEP's eight loops are that ladder's forms: an R target's
    ``dsig`` is P, which never carries a layer, so it takes ``f += raw`` or, in a dsigu (Z)
    chunk, ``fu += raw; f += siginvu*raw``; a Z target's ``dsigu`` is P, so it takes
    ``f += raw`` or, in a dsig (R) chunk, ``f += raw*siginv``; a conductivity scales
    ``raw`` by ``cndinv`` and, on the dsig cells, adds it to ``f_cond``.

    The axis row of a shift-0 target has ``r = 0``; MEEP's owned loop never visits it
    (the per-m axis rules own that row) and the caller masks it to ``+0``, so the divisor
    is clamped rather than special-cased.
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
    real_dtype = partner_values.real.dtype
    # The row vector is a GRID INVARIANT — Yee shift, radial extent, m, Courant — so it is
    # built once per (target, sign) rather than on every pass of every step.
    def _build():
        iyee_r = IYEE_SHIFTS[target][0]
        the_m = real_dtype.type(sign * 2.0 * grid.m * float(grid.courant))
        r_doubled = 2 * xp.arange(partner_values.shape[0], dtype=xp.int64) + iyee_r
        here_r = xp.maximum(r_doubled, 1).astype(real_dtype).reshape(-1, 1, 1)
        return the_m / here_r

    key = ("cyl_imr_rinv", target, float(sign), int(partner_values.shape[0]),
           real_dtype, int(grid.m), float(grid.courant))
    rinv = (_build() if fields.scratch is None
            else fields.scratch.constant(key, _build))
    return part_products.quarter_turn(rinv, partner_values)


def _special_kz_beta_term(fields: "Fields", partner_values: Any, sign: float,
                          magnetic: bool) -> Any:
    """The out-of-plane ``i*2*pi*beta`` increment — MEEP step_db.cpp:148-176.

    The 2-D run carries ``exp(i*2*pi*beta*z)`` analytically, so ``d/dz`` on the
    invariant axis is the EXACT factor ``i*2*pi*beta``. In the curl equations it is an
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

    formed as a quarter turn of the parts by the real ``-+sign * 2*pi*beta*dt``
    (:func:`~.part_products.quarter_turn`), never as a complex product.

    REAL STORAGE. ``f[c_g][1 - cmp]`` is NULL, the extra factor collapses to 1, and
    the same code is a plain real add of ``sign * 2*pi*beta*dt * g`` on BOTH sides.
    That is MEEP's "implicitly store i*(TM fields)" trick (its comment at
    step_db.cpp:148-160): the i's cancel because every observable is a conjugated
    bilinear. It is an interpretation of the same arithmetic, not a second code path,
    so reproducing the arithmetic reproduces the run — the one place the implicit i
    has to be paid explicitly is an eigenmode source, where MEEP applies
    ``special_kz_phasefix`` (mpb.cpp:295-304) before injecting.

    ``betadt`` is formed in double and rounded once to ``realnum``, so the increment is
    ``float32(coefficient) * g`` per part, one rounding.

    Returned NEGATED, in curl sign convention: the caller (``step_B`` / ``step_D``)
    masks MEEP's unowned cells to ``+0`` and hands it to :func:`_apply_folded_increment`,
    which SUBTRACTS it through MEEP's beta ladder (step_generic.cpp:255-333) as a pass
    of its own after every component's curl (step_db.cpp:161-175). The pass picks its
    PML and conductivity slices with the same ``cycle_direction(gv.dim, d_c, 1)`` and
    ``(..., 2)`` the curl uses (step_db.cpp:56-59 against :166-169), so it keys on the
    curl's own region plan.
    """
    grid = fields.grid
    coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt
    if partner_values.dtype.kind == "c":
        return -part_products.quarter_turn(coefficient if magnetic else -coefficient,
                                           partner_values)
    if fields.has_offdiagonal_epsilon:
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
                magnetic: bool, plain: Any = True) -> Any:
    """The BFAST increment for one component — MEEP step_db.cpp:129-142 + step_generic.cpp:335-533.

    BFAST (broadband fixed angle source technique) is the time-sheared substitution
    ``t -> t - k.r/c``, under which a planewave at a FIXED incidence angle is
    transversely uniform at EVERY frequency, so plain periodic boundaries are exact
    for a whole band. In the stepper it is exactly::

        dB/dt = -curl(E) + d/dt (k x E)
        dD/dt = +curl(H) - d/dt (k x H)

    THE ARITHMETIC, transcribed. With both partners MEEP computes, per cell, in this
    order and with each operation rounded once::

        F_prev = F[i]
        F[i]   = (k1*(g1[i+s1] + g1[i]) - k2*(g2[i+s2] + g2[i])) - F[i]
        raw    = F[i] - F_prev

    so the state is stored (one rounding) and the increment is a SECOND rounding of
    its difference from the previous state. ``raw`` then goes through the pass's own
    additive ladder (:func:`_apply_folded_increment`), after the curl of the same
    component has been applied: MEEP's ``STEP_BFAST`` is a pass of its own after
    ``STEP_CURL`` (step_db.cpp:124-142), not a term inside the curl.

    ONE PARTNER. MEEP hands ``step_bfast`` NULL for a partner its step plan lacks and,
    when that is ``g1``, swaps ``g1`` with ``g2`` and ``k1`` with ``k2``
    (step_generic.cpp:342-346); the branch then reads ``k1*(g1[i+s1] + g1[i]) - F[i]``.
    In MEEP's PLAIN branch — no PML layer on either axis in the chunk and no
    conductivity — the line reads ``F[i] = k1*(g1[i+s1] + g1[i])`` with no ``- F[i]``
    (step_generic.cpp:373-379), unlike the seven other branches. That is MEEP's
    arithmetic, so it is reproduced: ``plain`` names the cells (``True``, ``False``, or a
    mask broadcastable to the field; :func:`_bfast_plain_cells`). The surviving k of a
    one-partner component is always zero — it is indexed by the ABSENT partner's
    presence — so the two spellings differ only in signed zeros; the product by the
    zero k (``-0.0`` on the D side) is still formed, because its sign reaches the store.

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
    ``grid.is_invariant`` on the partner's derivative axis (:func:`curl_presence`). An
    absent partner is not read at all, and its k is zero, as MEEP's NULL pointer makes
    it. MEEP's other reason for a false flag — ``gv.has_field``, which drops all but
    Ex/Hy/Dx/By in D1 — needs no separate test here: on the components MEEP does step
    in D1 the two rules agree, and a component with neither partner (one MEEP does not
    step) takes a zero increment and keeps its state.

    Returned NEGATED, in curl sign convention, with MEEP's unowned cells at ``+0``:
    :func:`_apply_folded_increment` subtracts it, so an unowned cell is left exactly as
    it was. The state is written on owned cells only, because MEEP writes ``F`` only
    inside its owned-cell loop (``sub_gv.little_owned_corner0(cc)``): ``F`` has NO
    spatial stencil, so an unowned cell can never contaminate an owned one, but writing
    it would make a diagnostic read of ``f_bfast`` disagree with MEEP's.
    """
    grid = fields.grid
    xp = grid.xp
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
    realnum = state.real.dtype.type  # k1, k2 are realnum; a part-wise scale on complex storage.
    if have_p and have_m:
        total = (part_products.scale(realnum(k1), operands.shifted_first + operands.first)
                 - part_products.scale(realnum(k2), operands.shifted_second + operands.second))
        updated = total - state
    elif have_p or have_m:
        # step_generic.cpp:342-346: an absent g1 is replaced by g2, k1 by k2.
        single = (part_products.scale(realnum(k1), operands.shifted_first + operands.first)
                  if have_p else
                  part_products.scale(realnum(k2), operands.shifted_second + operands.second))
        if plain is True:
            updated = single  # step_generic.cpp:376, MEEP's plain one-partner line.
        elif plain is False:
            updated = single - state
        else:
            updated = xp.where(plain, single, single - state)
    else:
        return xp.zeros_like(state)
    increment = -(updated - state)
    _mask_non_owned_cells(increment, grid, term.iyee)
    owned = xp.ones(state.shape, dtype=bool)
    _mask_non_owned_cells(owned, grid, term.iyee)
    xp.copyto(state, updated, where=owned)
    return increment


def _bfast_plain_cells(fields: "Fields", term: CurlTerm, presence: Tuple[bool, bool],
                       pml: Optional["PML"], pml_active: bool) -> Any:
    """Where MEEP's BFAST takes its PLAIN branch for one component: ``True``, ``False`` or a mask.

    The plain branch (step_generic.cpp:365-380) is the chunk that carries neither the
    component's dsig layer nor its dsigu layer (step_db.cpp:56-59) and no conductivity.
    It matters only to a one-partner component (:func:`_bfast_term`); a component with
    both partners, or none, reads ``True`` and is never consulted. The conductivity is
    read per component (``fields.condfac_for``), as the curl reads it.
    """
    if presence[0] == presence[1]:
        return True
    if fields.condfac_for(term.target) is not None:
        return False
    if not pml_active:
        return True
    xp = fields.grid.xp
    plan = _region_plan(fields, pml)
    dsig = plan.axis_mask(AXIS_NAMES.index(term.dsig), term.target, xp)
    dsigu = plan.mask("curl", term.target, xp)
    return xp.logical_not(dsig) & xp.logical_not(dsigu)


def _apply_folded_increment(fields: "Fields", term: CurlTerm, increment: Any,
                            pml: Optional["PML"], pml_active: bool,
                            half_integer: bool) -> None:
    """Add one folded pass's increment to its target through MEEP's additive ladder.

    The special-kz beta pass (step_generic.cpp:255-333) and the BFAST pass (:335-533)
    end in the same ladder on ``raw``, keyed on the chunk's dsig and dsigu layers and
    the component's conductivity, exactly as the curl of the same component keys
    (step_db.cpp:56-59, :166-169)::

        neither layer       f += raw                (conductive: f += raw*cndinv)
        dsig only           f += raw*siginv         (conductive: dfcnd = raw*cndinv;
                                                     fcnd += dfcnd; f += dfcnd*siginv)
        dsigu (only, both)  fu += (df = raw*siginv); f += siginvu*df
                            (conductive: raw*cndinv in place of raw, and fcnd += it on
                             the dsig cells)

    There is no decay factor anywhere: the curl already applied those to the old field.
    The ``siginv`` of an axis is exactly 1 outside the chunks that carry its layer (the
    region plan's precondition, :func:`~.chunk_map.check_unit_coefficients`), so
    ``raw*siginv`` IS ``raw`` there and two forms cover the four: ``f += raw*siginv``
    outside the component's dsigu chunks and the ``fu`` stage inside them, dispatched as
    :func:`_apply_curl_regions` dispatches the curl. ``fu`` is read and written on its
    own cells only, and ``f_cond`` on the dsig cells of the chunks that keep the
    component's conductivity only (the auxiliary contract, as the curl writes it).

    ``increment`` is ``-raw`` (curl sign convention), with MEEP's unowned cells already
    at ``+0``: every stage SUBTRACTS it, which is the same word as adding ``raw``
    (``x - (-y)`` is ``x + y`` in IEEE-754, signed zeros included, and ``(-a)*b`` is
    ``-(a*b)`` exactly), while ``x - (+0)`` is ``x`` for every ``x`` — so an unowned
    cell, which MEEP's loop never visits, keeps its word, ``-0`` included. Adding a
    masked ``+0`` would turn a stored ``-0`` into ``+0``.
    """
    target = getattr(fields, term.target)
    xp = fields.grid.xp
    condfac = fields.condfac_for(term.target)
    if condfac is not None:
        increment = part_products.scale(fields.condinv_for(term.target), increment)
    if not pml_active:
        target -= increment
        return
    plan = _region_plan(fields, pml)
    _, sinv = _curl_coefficients(pml, term.dsig, half_integer=half_integer)
    _, sinv_u = _curl_coefficients(pml, term.dsigu, half_integer=half_integer)
    if condfac is not None:
        f_cond = getattr(fields, "f_cond_" + term.target)
        if f_cond is None:
            raise RuntimeError(
                "Conductivity + PML stepping requires Fields to allocate "
                f"f_cond_{term.target} before the first step."
            )
        dsig = plan.axis_mask(AXIS_NAMES.index(term.dsig), term.target, xp)
        # f_cond exists only in the chunks that keep the component's conductivity, as the
        # curl writes it (_apply_conductive_pml_update). In a chunk that keeps none, every
        # sample is zero, so condinv is exactly 1 there and the field's word is the
        # lossless ladder's either way.
        sigma = fields.conductivity_for(term.target)
        if not all(plan.conductive_chunks(term.target, sigma)):
            dsig = dsig & plan.conductive_mask(term.target, sigma, xp)
        xp.copyto(f_cond, f_cond - increment, where=dsig)
    df = part_products.scale(sinv, increment)
    fu = getattr(fields, "fu_" + term.target)
    spec = plan.spec("curl", term.target)
    mode = spec.mode
    if mode == "outside_only":
        target -= df
        return
    if mode == "inside_only":
        fu -= df
        target -= part_products.scale(sinv_u, df)
        return
    axis = spec.axis
    if mode == "split":
        for start, stop in spec.inside:
            index = _span(axis, start, stop)
            part = _cut(df, index, axis)
            fu_view = _cut(fu, index, axis)
            fu_view -= part
            field_view = _cut(target, index, axis)
            field_view -= part_products.scale(_cut(sinv_u, index, axis), part)
        for start, stop in spec.outside:
            index = _span(axis, start, stop)
            field_view = _cut(target, index, axis)
            field_view -= _cut(df, index, axis)
        return
    # select / mask: the outside form on the whole grid, the inside form selected.
    inside = plan.mask("curl", term.target, xp)
    xp.copyto(fu, fu - df, where=inside)
    inside_form = target - part_products.scale(sinv_u, df)
    target -= df
    xp.copyto(target, inside_form, where=inside)


def update_H(fields: "Fields", pml: Optional["PML"] = None) -> None:  # Magnetic constitutive sub-step.
    """Apply the magnetic constitutive relation H = B/mu with mu = 1.

    MEEP source: step.cpp (update_eh, H_stuff), step_generic.cpp
    (step_update_EDHB). Without PML this is a no-op — H is numerically identical
    to B and ``Fields.get_H`` serves it on demand, so nothing is stored. With
    PML the stored H accumulates absorption along the component's own axis
    (MEEP's dsigw) from integer-position coefficients, on the cells whose chunk
    carries that layer; everywhere else H is B's word, as MEEP's H there is B's
    storage (:func:`_apply_constitutive_regions`). The run's first call starts H and
    ``f_w`` as MEEP does (:func:`separate_h_from_b`).
    """
    _acquire_written(fields, H_COMPONENTS + _aux_names(H_COMPONENTS, ("f_w_",)))
    if not _pml_is_active(pml):
        return  # H is served on demand from B; there is no stored state to update.
    _require_pml_storage(fields, H_PML_ARRAYS)
    separate_h_from_b(fields, pml)
    plan = _region_plan(fields, pml)
    xp = fields.grid.xp
    for component, source, axis in H_CONSTITUTIVE_TERMS:
        kps, kms = _constitutive_coefficients(pml, axis, half_integer=False)
        _apply_constitutive_regions(plan, component, getattr(fields, component),
                                    getattr(fields, source), kps, kms,
                                    getattr(fields, "f_w_" + component), xp,
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
    its own direction, on the cells whose chunk carries that layer; there ``f_w``
    keeps the un-absorbed constitutive product, and everywhere else E IS that
    product and ``f_w`` is not touched (:func:`_apply_constitutive_regions`). That
    split is what :meth:`~.fields.Fields.drive_field` hands the polarizations.

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
    plan = None
    if pml_active:
        _require_pml_storage(fields, E_PML_ARRAYS)
        plan = _region_plan(fields, pml)
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
    absent = fields.absent_components
    for component, _, axis_name in E_CONSTITUTIVE_TERMS:
        if component in absent:
            # MEEP allocated no array for it and update_eh steps only allocated ones
            # (``if (f[ec][cmp])``, update_eh.cpp:150): its word stays +0, and so do its
            # f_w words (:meth:`~.fields.Fields.set_absent_components`).
            continue
        if nonlinear:
            constitutive = _nonlinear_constitutive(fields, component, displacement)
        elif offdiagonal:
            # MEEP step_update_EDHB (u1 and/or u2, no chi3): the full row product
            # fw[i] = gs*us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2), formed in
            # double and rounded once by the store (step_generic.cpp:611, :719).
            row = _offdiagonal_terms(fields, component, displacement)
            if row is None:
                constitutive = part_products.scale(fields.inverse_epsilon_for(component),
                                                   displacement["volumes"][component])
            else:
                constitutive = _rounded_row(fields.grid.xp, row)
        else:
            source = fields.displacement_minus_polarization(component)
            # MEEP step_update_EDHB: fw[i] = gs*us, per part.
            constitutive = part_products.scale(fields.inverse_epsilon_for(component), source)
        if pml_active:
            kps, kms = _constitutive_coefficients(pml, axis_name, half_integer=True)
            _apply_constitutive_regions(plan, component, getattr(fields, component),
                                        constitutive, kps, kms,
                                        getattr(fields, "f_w_" + component), fields.grid.xp,
                                        scratch=fields.scratch)
        else:
            # MEEP update_eh.cpp with dsigw == NO_DIRECTION: E = chi1inv * (D - P),
            # written into storage rather than returned, because P advances next.
            getattr(fields, component)[...] = constitutive
    if nonlinear and fields.grid.has_symmetry():
        _image_nonlinear_mirror_ghosts(fields, plan)


def _image_nonlinear_mirror_ghosts(fields: "Fields", plan: Any) -> None:  # E and f_w ghosts <- parity image.
    """Write E, and its ``f_w`` on the accumulation's cells, at every mirror ghost as its image.

    MEEP never evaluates the constitutive update at a mirror ghost: ``update_eh``
    loops over owned points only (``little_owned_corner0``, update_eh.cpp:191), and
    ``step_boundaries(WE_stuff)`` and ``step_boundaries(E_stuff)`` (step.cpp:105-121)
    then write each ghost as the parity image of its owned cell, W and E alike. This
    engine evaluates the update on the whole stored volume, ghosts included, which is
    the image while the update is pointwise. The nonlinear ``Dsqr`` reads the
    neighbouring cells of the transverse partners, so at a ghost it is not: there the
    ghost is written from its image here, after the update, as
    :func:`_fill_symmetry_ghost_cells` writes D and B. MEASURED (bit_parity_0_10
    phase-0 leg P0-h, the folded nonlinear case against MEEP 1.33.0 built without
    contraction): without it, ghost E words depart from step 68 and owned words
    from step 90; with it, 0 owned or wall words differ to step 1000. A linear run
    keeps the pointwise ghost, which reaches no owned or wall word (the same leg's
    linear control).

    ``f_w`` is written only where the accumulation runs (the auxiliary contract), and
    a ghost row's membership is its image row's (``chunk_map``'s ghost-adjusted
    line), so the copy writes exactly the cells the update wrote. Axes are taken in
    X, Y, Z order, so a corner unowned on two planes carries both parities, as for D.
    """
    grid = fields.grid
    xp = grid.xp
    parities = _mirror_phases(grid)
    for component in E_COMPONENTS:
        shifts = IYEE_SHIFTS[component]
        axes = [axis for axis in range(3) if parities[axis] is not None and shifts[axis] == 0]
        if not axes or component in fields.absent_components:
            continue  # An absent component is not written at all (update_E).
        field = getattr(fields, component)
        for axis in axes:
            _write_mirror_ghost(field, axis, _symmetry_phase(component, axis, parities[axis]))
        mode = "outside_only" if plan is None else plan.spec("constitutive", component).mode
        if mode == "outside_only":
            continue
        fw = getattr(fields, "f_w_" + component)
        inside = (None if mode == "inside_only"
                  else xp.broadcast_to(plan.mask("constitutive", component, xp), fw.shape))
        for axis in axes:
            phase = _symmetry_phase(component, axis, parities[axis])
            if inside is None:
                _write_mirror_ghost(fw, axis, phase)
                continue
            _mirror_source(fw, axis)  # the range check the whole-face copy makes
            ghost = _face(axis, 0)
            xp.copyto(fw[ghost], part_products.scale(phase, fw[_face(axis, MIRROR_SOURCE_INDEX)]),
                      where=inside[ghost])


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
        "pml": pml,  # The active layer or None: the off-diagonal presence's chunk division.
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
    xp = fields.grid.xp
    # MEEP's MOST GENERAL CASE (step_generic.cpp:590-601) scales the WHOLE row
    # product by the Pade factor: fw = (gs*us + OFFDIAG + OFFDIAG) * u. The row is a
    # double (each OFFDIAG is, :580-581), so the product is formed in double and the
    # store rounds it once. The coefficient is real, so the row is formed per part
    # of a complex field — the DOCMP split (update_eh.cpp:149).
    row = _offdiagonal_terms(fields, component, displacement)
    if not fields.is_nonlinear(component):
        # A linear component in a partly nonlinear run: MEEP's `else if (u)` branch.
        return part_products.scale(us, gs) if row is None else _rounded_row(xp, row)
    chi2 = fields.chi2_for(component)
    chi3 = fields.chi3_for(component)
    transverse = _nonlinear_transverse_sums(fields, component, displacement)
    if gs.dtype.kind != "c":
        factor = calc_nonlinear_u(_nonlinear_dsqr(gs, transverse, _WHOLE), gs, us, chi2, chi3)
        if row is None:
            return (gs * us) * factor
        return _rounded_row(xp, (row[0] * factor.astype(xp.float64),))
    factor_real = calc_nonlinear_u(_nonlinear_dsqr(gs, transverse, _REAL), gs.real, us, chi2, chi3)
    factor_imag = calc_nonlinear_u(_nonlinear_dsqr(gs, transverse, _IMAGINARY), gs.imag, us, chi2, chi3)
    if row is not None:
        return _rounded_row(xp, (row[0] * factor_real.astype(xp.float64),
                                 row[1] * factor_imag.astype(xp.float64)))
    real = (gs.real * us) * factor_real
    imaginary = (gs.imag * us) * factor_imag
    return part_products.from_parts(real, imaginary, gs.dtype)


def _WHOLE(values: Any) -> Any:  # A real-storage field has no parts to separate.
    return values


def _REAL(values: Any) -> Any:  # MEEP's cmp == 0 array.
    return values.real


def _IMAGINARY(values: Any) -> Any:  # MEEP's cmp == 1 array.
    return values.imag


def _nonlinear_dsqr(gs: Any, transverse: Tuple[Any, Any], part: Any) -> Any:  # MEEP's Dsqr argument.
    """``gs*gs + 0.0625*(g1s*g1s + g2s*g2s)`` — MEEP step_generic.cpp:646-654, as its C evaluates it.

    ``0.0625 = (1/4)^2`` is what turns each UNNORMALIZED four-point sum
    (:func:`_nonlinear_transverse_sums`) back into a mean before it is squared, so
    ``Dsqr`` really is ``|D|^2`` interpolated to this component's Yee position.

    THE PRECISION IS MEEP's. ``gs * gs`` and ``g1s * g1s + g2s * g2s`` are ``realnum``
    (float32) operations, but ``0.0625`` is a double literal, so the scaled sum and the
    addition that follows it are double operations, and the ``const realnum Dsqr``
    parameter of ``calc_nonlinear_u`` (:542) rounds the result once. Here: the two
    float32 terms are formed in the field's own precision, widened, scaled and added
    as float64 ufuncs, and the sum is rounded once back to the field's precision.
    Against the float32 form this moves Dsqr only where the exact ``0.0625 * t`` lies
    below float32's normal range (MEASURED, bit_parity_0_10 phase-0 leg P0-h:
    12,568,559 of 396,821,530 operand pairs differ, 0 outside that band), which on
    the measured cases reaches no stored E word.

    ``part`` selects the real or imaginary half AFTER the sums are formed, never
    before: a Bloch wrap multiplies the whole complex value by ``exp(i k L)`` and so
    mixes the two, exactly as MEEP's ``step_boundaries`` does when it fills a phased
    chunk connection into the separate ``cmp`` arrays.
    """
    first, second = transverse
    scaled = part(gs)
    first_part = part(first)
    second_part = part(second)
    transverse_squared = first_part * first_part + second_part * second_part
    widened = (scaled * scaled).astype("float64") + 0.0625 * transverse_squared.astype("float64")
    return widened.astype(scaled.dtype)


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

    THE ASSOCIATION IS MEEP's, LEFT TO RIGHT: ``((g[i] + g[i+s]) + g[i-s1]) +
    g[i+(s-s1)]``, each addition one rounding in the field's precision. Pairing the
    corners instead, ``(g[i] + g[i-s1]) + (g[i+s] + g[i+(s-s1)])``, is the same sum in
    exact arithmetic and a different float32 word. MEASURED against MEEP 1.33.0 single
    precision built without contraction (bit_parity_0_10 phase-0 leg P0-h, the four
    unfolded nonlinear cases): the pair form moved 79,850 to 2,056,001 stored words on
    three of them, this order none on any.

    The partners follow ``cycle_direction`` (vec.hpp:586), X -> Y -> Z, so Ez takes
    Dx then Dy. Neighbour reads go through the same ``_shift_up`` / ``_shift_down``
    helpers the curl uses, and the same-index reads through
    :func:`_bloch_ghost_operand`, so a periodic wrap, a Bloch phase and a mirror plane
    behave here exactly as they do there. The corner ``g[i+(s-s1)]`` is the
    down-shifted partner shifted up, one stored word, as MEEP reads it. On a
    reduced-dimension grid's invariant axis MEEP's stride is 0, so ``g[i+s]`` or
    ``g[i-s1]`` is ``g[i]`` itself; the one-cell periodic roll the shift helpers make
    there returns the same word. The Bloch phase lands on the operand that is NOT
    shifted along the phased axis, as it does in the curl: ``g[i]`` and ``g[i-s1]``
    take it on the own axis, ``g[i]`` and ``g[i+s]`` on the partner's. (On a corner
    phased on BOTH axes the two connections apply one after the other, where MEEP
    forms the corner's not-owned word with one product of the accumulated phase; that
    is the complex-nonlinear family's identity item, reached by neither family's
    cases, and not a value change.)
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
        # The operands not shifted along the partner's axis carry its phase on plane 0.
        same = _bloch_ghost_operand(xp, values, partner_axis, boundaries[partner_axis],
                                    phases[partner_axis], below=False)
        # g[i-s1]: half a cell DOWN the partner's own axis (a plain roll on a phased axis).
        down = _shift_down(xp, values, partner_axis, boundaries[partner_axis],
                           "D" + AXIS_NAMES[partner_axis], parities[partner_axis])
        # g[i] and g[i-s1] are not shifted along this component's axis: its phase lands here.
        first = _bloch_ghost_operand(xp, same, own_axis, boundaries[own_axis],
                                     phases[own_axis], below=True)
        third = _bloch_ghost_operand(xp, down, own_axis, boundaries[own_axis],
                                     phases[own_axis], below=True)
        # g[i+s] and g[i+(s-s1)]: half a cell UP this component's axis.
        up = _shift_up(xp, same, own_axis, boundaries[own_axis])
        corner = _shift_up(xp, down, own_axis, boundaries[own_axis])
        sums.append(((first + up) + third) + corner)
    cache[component] = (sums[0], sums[1])
    return cache[component]


def _offdiagonal_terms(fields: "Fields", component: str,
                       displacement: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:  # MEEP's row, in double.
    """The constitutive row with its off-diagonal terms, as MEEP's C evaluates it, unrounded.

    MEEP step_generic.cpp:580-581, :611 (PML) and :719 (no PML), transcribed::

        OFFDIAG(u, g, sx) = 0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])
        fw[i] = gs*us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)

    The pair sums, the products and their sum are ``realnum`` (float32) operations,
    but ``0.25`` is a double literal, so each OFFDIAG is a double (the scale is exact:
    a power of two). ``gs*us`` is one float32 product, widened, and the row is summed
    left to right in double; the store rounds it once. This returns that double row,
    one float64 volume per stored part — ``(row,)`` for real storage, ``(real,
    imaginary)`` for complex64, whose parts MEEP steps separately with the same real
    coefficients (``DOCMP``, update_eh.cpp:149) — and the caller rounds it
    (:func:`_rounded_row`) or, for a nonlinear component, scales it by the Pade factor
    in double first. ``None`` for a component with no surviving rows, so the caller's
    degenerate arithmetic is the diagonal engine's, not a ``+ 0`` copy of it.

    ``s`` is the stride along this component's OWN axis, ``sx`` the partner's. The
    pairing is MEEP's "stable averaging", and it is exact rather than merely stable:
    ``structure_chunk::set_chi1inv`` registers the off-diagonal entries at the
    component's Yee site minus half a cell along its own axis — the integer node
    (anisotropic_averaging.cpp:248-257, ``here - shift1``) — so ``u[i]`` multiplies
    the two-point partner average AT ITS OWN NODE and ``u[i+s]`` the average at the
    next node up. The partner pair is formed exactly as
    :func:`_nonlinear_transverse_sums` forms its half — a half-cell DOWN the
    partner's axis, then the product shifted half a cell UP this component's axis —
    through the same ``_shift_up`` / ``_shift_down`` helpers the curl uses, and the
    same-index reads through :func:`_bloch_ghost_operand`, so a periodic wrap, a Bloch
    phase and a metallic wall behave here exactly as they do there. The coefficient
    is periodic and carries no phase of its own; the connection on the same-index
    PRODUCT is the field's, applied after the coefficient (where MEEP connects each
    word before the product -- the off-diagonal family's identity item, not a value
    change). (A folded axis never reaches this function: ``Fields.set_epsilon_volumes``
    refuses the combination at install.) Every product is a real coefficient times one
    stored part (:mod:`.part_products`). The wall mask
    (:func:`_mask_metallic_wall_coupling`) zeroes each term's float32 sum before the
    scale.

    PRESENCE. A term is added only on the cells whose chunk keeps the partner's
    ``chi1inv`` array, MEEP's NULL pointer elsewhere (:meth:`~.fields.Fields.offdiag_presence`,
    read from the run's chunk division, :func:`_chunk_layout`): MEEP drops the term
    there rather than adding a zero, which differs from adding one only when the row
    so far is ``-0``. The terms are added in MEEP's ``cycle_direction`` order (1, then
    2); its SWAP (step_generic.cpp:572-576) only moves a lone term into the first slot.
    """
    rows = fields.chi1inv_offdiagonal_for(component)
    if not rows:
        return None
    volumes = displacement["volumes"]
    boundaries = displacement["boundaries"]
    phases = displacement["phases"]
    parities = displacement["parities"]
    grid = fields.grid
    xp = grid.xp
    iyee = IYEE_SHIFTS[component]
    layout = _chunk_layout(fields, displacement.get("pml"))
    own_axis = AXIS_NAMES.index(component[1])
    inverse = fields.inverse_epsilon_for(component)
    # gs*us: one float32 product per part (realnum * realnum), then widened.
    row = [(part * inverse).astype(xp.float64)
           for part in part_products.parts(volumes[component])]
    for offset in (1, 2):  # MEEP's cycle_direction(dim, d_ec, 1) then (..., 2).
        partner_axis = (own_axis + offset) % 3
        partner = "E" + AXIS_NAMES[partner_axis]
        coefficient = rows.get(partner)
        if coefficient is None:
            continue
        values = volumes[partner]
        # g[i] + g[i-sx]: half a cell DOWN the partner's own axis.
        pair = (_bloch_ghost_operand(xp, values, partner_axis, boundaries[partner_axis],
                                     phases[partner_axis], below=False)
                + _shift_down(xp, values, partner_axis, boundaries[partner_axis],
                              "D" + AXIS_NAMES[partner_axis], parities[partner_axis]))
        product = part_products.scale(coefficient, pair)
        # (g[i] + g[i-sx])*u[i] per part, at this cell's own node (read across a phased
        # window edge as MEEP's not-owned word below it), and the same product half a
        # cell UP this axis: (g[i+s] + g[(i+s)-sx])*u[i+s].
        here = _bloch_ghost_operand(xp, product, own_axis, boundaries[own_axis],
                                    phases[own_axis], below=True)
        above = _shift_up(xp, product, own_axis, boundaries[own_axis])
        present = fields.offdiag_presence(component, partner, layout)
        for index, (low, high) in enumerate(zip(part_products.parts(here),
                                                part_products.parts(above))):
            summed = low + high
            _mask_metallic_wall_coupling(summed, grid, iyee)
            term = 0.25 * summed.astype(xp.float64)  # The double literal: exact.
            if present is None:
                row[index] += term
            else:
                row[index] = xp.where(present, row[index] + term, row[index])
    return tuple(row)


def _rounded_row(xp: Any, row: Sequence[Any]) -> Any:  # MEEP's store of a double row.
    """Round a double row from :func:`_offdiagonal_terms` once to the stored type.

    One part is a float32 volume, two are a complex64 one; each part is rounded on
    its own (the assignment to ``realnum``) and the two are assembled by
    :func:`~.part_products.from_parts`, never through a complex conversion.
    """
    parts = [part.astype(xp.float32) for part in row]
    return parts[0] if len(parts) == 1 else part_products.from_parts(parts[0], parts[1],
                                                                     xp.complex64)


def _chunk_layout(fields: "Fields", pml: Optional["PML"]) -> Any:
    """The chunk division ``update_E``'s off-diagonal presence is read from, or None for one chunk.

    The lifted simulation's own when the driver installed one
    (:attr:`~.fields.Fields.chunk_layout`); otherwise the one-rank division: the
    region plan's under an active layer (the one the PML forms step with), and None
    for an unlayered grid, whose one-rank division is a single chunk.
    """
    layout = fields.chunk_layout
    if layout is not None:
        return layout
    if pml is not None and _pml_is_active(pml):
        return _region_plan(fields, pml).layout
    return None


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


def cylindrical_rderiv_difference(xp: Any, f_p: Any, ir0: float, starts: Any, n_target: int,
                                  backward: bool, scratch: Any = None) -> Any:
    """MEEP's radial difference of ``f_rderiv_int``, the per-chunk prefix sum of (1/r)·d(r·Fp)/dr.

    step_db.cpp:99-119, transcribed. The Z update in cylindrical coordinates needs
    ``(1/r)·d(r·Fp)/dr`` where the ordinary curl computes ``dFp/dr``, so MEEP builds, in
    EVERY structure chunk over that chunk's own rows ``ir = 0 .. nr``, the running sum::

        F[0,  :] = 0
        rinv     = 1.0 / ((ir + ir0) - 0.5)           (double, stored as realnum)
        F[ir, :] = F[ir-1, :] + rinv * (f_p[ir]*(ir+ir0) - f_p[ir-1]*((ir-1)+ir0))

    in ``realnum`` arithmetic, left to right, and feeds ``F`` to the unmodified curl, whose
    radial difference is the wanted derivative. ``ir0 = origin_r*a + 0.5*iyee_shift(Fp)``
    with the CHUNK's origin, so ``ir + ir0`` is the global row plus half Fp's r-shift and
    the increment of a row is the same in every chunk; what is per chunk is the SUM, which
    restarts at ``+0`` on each chunk's first row. A chunk whose first row is ``s`` sums
    ``F_s[s] = +0``, ``F_s[g] = F_s[g-1] + inc[g]``, and a target cell reads the sum of the
    chunk that owns it:

    * forward (B side, Bz, half-integer rows ``j``): ``F_s[j+1] - F_s[j]``;
    * backward (D side, Dz, integer rows ``g``, D's negated stride): ``F_s[g-1] - F_s[g]``;

    the stencil's ``g1[i+s1] - g1[i]`` either way, returned unscaled (the caller multiplies
    by ``dtdx``). So the cut row between two chunks holds two values, the lower chunk's
    full sum and the upper chunk's zero, and no single prefix array represents the
    division.

    ``starts`` is, per target cell, the first stored row of its owning chunk: an integer
    array broadcastable against the target volume (``(n_target, 1, 1)`` on a division with
    global r cuts, the full volume on one without), or ``None`` for one chunk starting at
    the axis (:func:`_cylindrical_scan_starts`). ``f_p`` is ``(r, phi, z)`` with r the
    leading axis; the B side passes Ep extended by its zero wall row (``n_target = rows -
    1``), the D side Hp itself (``n_target = rows``). Row 0 of a backward target, the axis,
    reads ``+0`` (the caller masks it; the per-m axis rules own it).

    ``rinv`` is MULTIPLIED, as MEEP stores and multiplies it; dividing by ``(ir + ir0) -
    0.5`` is a different float32 word. The sum is sequential along r within a chunk:
    NumPy's ``cumsum`` along axis 0 accumulates row after row (pinned against the
    explicit loop in ``test_cylindrical``), and any other array module takes the row loop
    (CuPy's ``cumsum`` is a parallel scan, another association). Complex storage runs per
    part (:mod:`~.part_products`; a complex sum or difference is already per part).
    """
    import numpy  # noqa: PLC0415 - host-side chunk arithmetic on a few integers

    real_dtype = f_p.real.dtype
    rows = f_p.shape[0]
    rest = f_p.shape[1:]
    if scratch is None:
        weights, rinv = _cylindrical_rderiv_rows(xp, rows, ir0, real_dtype)
        weighted = part_products.scale(weights, f_p)
        increment = xp.empty_like(f_p)
        out = xp.empty((n_target,) + rest, dtype=f_p.dtype)
    else:
        # The two row vectors are GRID INVARIANTS (radial extent, ir0, storage dtype); the
        # full-volume temporaries are pooled for the reason StepScratch exists. The
        # returned buffer is the pool's ``cyl_radial`` slot, valid until the next call.
        key = ("cyl_rderiv_rows", int(rows), float(ir0), real_dtype)
        weights, rinv = scratch.constant(key, lambda: _cylindrical_rderiv_rows(
            xp, rows, ir0, real_dtype))
        weighted = scratch.take("cyl_weighted", f_p.shape, f_p.dtype)
        part_products.scale(weights, f_p, out=weighted)
        increment = scratch.take("cyl_increment", f_p.shape, f_p.dtype)
        out = scratch.take("cyl_radial", (n_target,) + rest, f_p.dtype)
    increment[_face(0, 0)] = 0  # Never read: every chunk's first row sums from +0.
    xp.subtract(weighted[1:], weighted[:-1], out=increment[1:])
    part_products.scale(rinv, increment[1:], out=increment[1:])
    out[...] = 0
    if starts is None:
        segments = (0,)
    else:
        segments = sorted({int(v) for v in numpy.unique(_host_ints(starts))})
    for s in segments:
        prefix = _cylindrical_segment_scan(xp, increment, s, scratch)
        if backward:  # F_s[g-1] - F_s[g] for g in (s, n_target)
            lo, hi = max(s + 1, 1), n_target
            shifted, unshifted = prefix[lo - 1 - s:hi - 1 - s], prefix[lo - s:hi - s]
        else:  # F_s[j+1] - F_s[j] for j in [s, rows - 1)
            lo, hi = s, min(n_target, rows - 1)
            shifted, unshifted = prefix[lo + 1 - s:hi + 1 - s], prefix[lo - s:hi - s]
        if hi <= lo:
            continue
        if starts is None:
            xp.subtract(shifted, unshifted, out=out[lo:hi])
        else:
            xp.copyto(out[lo:hi], shifted - unshifted, where=xp.asarray(starts[lo:hi] == s))
    return out


def _cylindrical_segment_scan(xp: Any, increment: Any, s: int, scratch: Any = None) -> Any:
    """One chunk's ``f_rderiv_int`` from its first row ``s`` on: ``F[0] = +0``, then one add per row.

    Rows ``s .. rows-1`` of the radial increment, the first replaced by ``+0`` (MEEP's
    ``f_rderiv_int[iz] = 0``; ``+0 + x`` is MEEP's first row, which a sum that starts from
    ``x`` itself is not when ``x`` is ``-0``). A chunk ends where the next one begins; rows
    past its end are summed too and never read.
    """
    import numpy  # noqa: PLC0415 - which array module owns the increment

    tail = increment[s:]
    prefix = (xp.empty_like(tail) if scratch is None
              else scratch.take(f"cyl_prefix_{s}", tail.shape, tail.dtype))
    if xp is numpy:
        seed = increment[s].copy()
        tail[0] = 0
        numpy.cumsum(tail, axis=0, out=prefix)
        tail[0] = seed
        return prefix
    prefix[0] = 0
    for g in range(1, tail.shape[0]):
        xp.add(prefix[g - 1], tail[g], out=prefix[g])
    return prefix


def _host_ints(values: Any) -> Any:  # A host NumPy view of a small integer array.
    import numpy  # noqa: PLC0415 - host-side

    get = getattr(values, "get", None)
    return numpy.asarray(get() if callable(get) else values)


def _cylindrical_rderiv_rows(xp: Any, rows: int, ir0: float,
                             real_dtype: Any) -> Tuple[Any, Any]:  # (ir + ir0 per row, its rinv).
    """The two invariant row vectors of :func:`cylindrical_rderiv_difference`.

    ``weights[g] = realnum(g + ir0)`` and ``rinv[g-1] = realnum(1.0 / ((g + ir0) - 0.5))``
    for rows ``g >= 1``, the quotient in double as MEEP writes it (``1.0`` is a double
    literal) and rounded once.
    """
    counts = xp.arange(rows, dtype=xp.float64) + ir0  # (ir + ir0) per row.
    return (counts.reshape(-1, 1, 1).astype(real_dtype),
            (1.0 / (counts[1:] - 0.5)).reshape(-1, 1, 1).astype(real_dtype))


def _cylindrical_scan_starts(fields: "Fields", pml: Optional["PML"], target: str) -> Any:
    """Per cell of ``target`` (Bz or Dz), the first stored r row of the chunk that owns it.

    MEEP sums ``f_rderiv_int`` over each structure chunk's own rows (step_db.cpp:104-118),
    so the radial difference a cell reads depends on where its owning chunk begins. The
    division is the run's (:func:`_chunk_layout`: the lifted simulation's own, else the
    one-rank division of the active layer); ``None`` when there is one chunk, which starts
    at the axis. A chunk box ``[lo, hi]`` in doubled r owns ``lo < p <= hi`` (the axis
    point ``p = 0`` belongs to the chunk starting there, vec.cpp:448-450) and its rows
    start at stored row ``lo // 2`` for either Yee shift (the r origin is the axis).
    Global r cuts (:meth:`~.chunk_map.ChunkLayout.boundaries`) give one start per row,
    shaped ``(n, 1, 1)``; a division whose r cuts differ along z gives one per cell, read
    from the owner volume.
    """
    import numpy  # noqa: PLC0415 - host-side chunk arithmetic on a few integers

    layout = _chunk_layout(fields, pml)
    if layout is None or len(layout.boxes) == 1:
        return None
    iyee = IYEE_SHIFTS[target]
    rows = fields.grid.stored_cells(0)
    cuts = layout.boundaries(0)
    if cuts is not None:
        lows = numpy.asarray(cuts[:-1], dtype=numpy.int64)
        doubled = 2 * numpy.arange(rows, dtype=numpy.int64) + iyee[0]
        owner = numpy.searchsorted(lows, doubled, side="left") - 1
        starts = lows[numpy.maximum(owner, 0)] // 2
        line = starts.reshape(-1, 1, 1)
        if numpy.all(line == 0):
            return None
        return fields.grid.xp.asarray(line)
    lows = numpy.asarray([lo[0] for lo, _ in layout.boxes], dtype=numpy.int64)
    return fields.grid.xp.asarray(lows[layout.owner_volume(iyee)] // 2)


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

    Gathered once and read twice: by the curl's DIFFERENCES and, on a BFAST run, by the
    BFAST pass's SUMS (:func:`_bfast_term`), which MEEP runs as a separate loop over the
    same partners right after the curl of the same component (step_db.cpp:124-142).

    A shifted operand is None when its partner is absent from MEEP's step plan
    (:func:`curl_presence`): neither the curl's form nor the BFAST pass reads an absent
    partner (MEEP hands both a NULL pointer), so the shifted copy is not made.
    """

    first: Any           # MEEP's g1 = f_p, at the cell.
    shifted_first: Any   # g1[i + s1] — one cell up (B) or down (D); None when not gathered.
    second: Any          # MEEP's g2 = f_m, at the cell.
    shifted_second: Any  # g2[i + s2]; None when not gathered.


def curl_presence(grid: "Grid", term: CurlTerm) -> Tuple[bool, bool]:
    """``(g1 present, g2 present)`` for one curl term, as MEEP's step plan records them.

    ``fields_chunk::figure_out_step_plan`` (fields.cpp:428-453) sets ``have_plus_deriv``
    / ``have_minus_deriv`` only when the partner's derivative direction ``cross(dc1,
    dc2)`` is a direction of the grid, and ``step_db`` hands ``step_curl`` a NULL partner
    otherwise. The partner's derivative axis here is ``term.first_axis`` /
    ``term.second_axis``, so the test is ``grid.is_invariant`` on it: z at 2-D, x and y
    at 1-D, phi (y) under cylindrical coordinates. The last agrees with step_db.cpp's
    explicit Dcyl rule, which drops ``f_p`` for an R component and ``f_m`` for a Z
    component (their phi-derivative partners) and keeps both for a P component. At 1-D,
    MEEP's other reason for a missing partner (``gv.has_field`` keeps only Ex, Hy, Dx
    and By) agrees with this one on the four components MEEP steps.
    """
    return (not grid.is_invariant(term.first_axis), not grid.is_invariant(term.second_axis))


def _curl_operands(xp: Any, snapshot: Dict[str, Any], term: CurlTerm,
                   boundaries: Tuple[str, str, str], phases: BlochPhases,
                   parities: MirrorPhases, backward: bool,
                   reflect_rows: FarReflectRows = (None, None, None),
                   scratch: Any = None,
                   gather: Tuple[bool, bool] = (True, True)) -> CurlOperands:
    """Gather one component's two curl partners and their shifted neighbours.

    ``backward`` selects MEEP's negated strides (s = -1, the D sub-step) over the
    forward ones (s = +1, the B sub-step) — the same choice for both the curl and the
    BFAST pass, because MEEP hands ``step_bfast`` the SAME already-negated
    ``stride_p``/``stride_m`` it handed ``step_curl`` (step_db.cpp:80-83 negates them
    once, before both calls).

    ``gather`` names which partners' shifted neighbours are formed (the step passes
    :func:`curl_presence`); one left out is None in the result.

    On a Bloch-phased axis the shifted neighbours are plain rolls of the STORED
    arrays and the unshifted operand carries the phase on its plane 0
    (:func:`_bloch_ghost_operand`): the engine stores MEEP's owned window there. The
    rolls are taken first, from the stored arrays, and the unshifted operands are
    replaced after them; rolling an already-connected copy would move MEEP's
    not-owned word into an owned cell.
    """
    first = snapshot[term.first]
    second = snapshot[term.second]
    gather_first, gather_second = gather
    shifted_first = shifted_second = None
    # The two shifted operands are alive at the same time — the curl differences them
    # and BFAST sums them — so they take SEPARATE pool tags. One tag for both would
    # hand the same buffer back twice and silently turn the stencil into g1 - g1.
    if backward:
        if gather_first:
            shifted_first = _shift_down(xp, first, term.first_axis,
                                        boundaries[term.first_axis], term.first,
                                        parities[term.first_axis],
                                        scratch=scratch, scratch_tag="stencil_first")
        if gather_second:
            shifted_second = _shift_down(xp, second, term.second_axis,
                                         boundaries[term.second_axis], term.second,
                                         parities[term.second_axis],
                                         scratch=scratch, scratch_tag="stencil_second")
    else:
        if gather_first:
            shifted_first = _shift_up(xp, first, term.first_axis, boundaries[term.first_axis],
                                      term.first, parities[term.first_axis],
                                      reflect_rows[term.first_axis],
                                      scratch=scratch, scratch_tag="stencil_first")
        if gather_second:
            shifted_second = _shift_up(xp, second, term.second_axis,
                                       boundaries[term.second_axis], term.second,
                                       parities[term.second_axis],
                                       reflect_rows[term.second_axis],
                                       scratch=scratch, scratch_tag="stencil_second")
    # B's forward differences read the unshifted partner one half cell BELOW the
    # target (MEEP's not-owned word under the window); D's backward differences read
    # it one half cell ABOVE (the not-owned word over the window).
    first = _bloch_ghost_operand(xp, first, term.first_axis, boundaries[term.first_axis],
                                 phases[term.first_axis], below=not backward,
                                 scratch=scratch, scratch_tag="bloch_first")
    second = _bloch_ghost_operand(xp, second, term.second_axis, boundaries[term.second_axis],
                                  phases[term.second_axis], below=not backward,
                                  scratch=scratch, scratch_tag="bloch_second")
    return CurlOperands(first, shifted_first, second, shifted_second)


def _curl_from_operands(operands: CurlOperands, dtdx: float, presence: Tuple[bool, bool],
                        scratch: Any = None) -> Any:  # dtdx * the discrete curl.
    """Return MEEP's ``dtdx``-scaled curl from an already-gathered stencil.

    ``step_curl`` (step_generic.cpp) evaluates, with ``presence`` from
    :func:`curl_presence`:

    * both partners: ``dtdx * (g1[i+s1] - g1[i] + g2[i] - g2[i+s2])`` (:104), which C
      associates left to right as ``((g1s - g1) + g2) - g2s``;
    * ``g2`` absent: ``dtdx * (g1s - g1)`` (:108);
    * ``g1`` absent: MEEP swaps the partners and negates ``dtdx`` (:72-76), so
      ``(-dtdx) * (g2s - g2)``;
    * neither: MEEP steps no such component (``gv.has_field`` keeps Ex, Hy, Dx and By
      at 1-D, the only grid with one), and the curl here is an exact zero, so the
      target is left as it is.

    Each operation rounds once, as the C does. The pairwise grouping
    ``(g1s - g1) + (g2 - g2s)`` is a different float32 number, and so is the full
    bracket over an absent partner: ``(a + g2) - g2`` rounds where the absent form
    returns ``a``.

    The expression form allocates full-volume temporaries per call and runs six times
    a timestep; with a :class:`~.fields.StepScratch` the same passes write into the
    pool's ``curl`` buffer instead. Bit-identical by construction — the ufuncs, their
    operand order and the promoted dtype are the same, only the destination differs —
    and pinned that way by the before/after byte comparison in
    ``test_step_scratch_is_bit_identical``.

    The returned buffer is the pool's ``curl`` slot and stays valid until the next
    curl is formed, which is exactly how far the caller carries it: the term loop
    masks it, may add a coupling term (allocating a fresh array, which releases the
    slot) and hands it to :func:`_apply_curl` before the next iteration.
    """
    first_present, second_present = presence
    first, second = operands.first, operands.second
    if scratch is None:
        if first_present and second_present:
            return part_products.scale(dtdx, ((operands.shifted_first - first) + second)
                                       - operands.shifted_second)
        if first_present:
            return part_products.scale(dtdx, operands.shifted_first - first)
        if second_present:
            return part_products.scale(-dtdx, operands.shifted_second - second)
        zero = first.copy()
        zero[...] = 0
        return zero
    xp = scratch.xp
    used = ((first, operands.shifted_first) if first_present else ()) \
        + ((second, operands.shifted_second) if second_present else ())
    # The operands share the field storage dtype in every configuration the engine
    # steps; the promotion call is kept for the case that stops being true, but it is
    # not paid six times a step to learn that they already agree.
    dtype = first.dtype
    if not all(array.dtype == dtype for array in used):
        dtype = xp.result_type(*used)
    total = scratch.take("curl", first.shape, dtype)
    if first_present and second_present:
        xp.subtract(operands.shifted_first, first, out=total)
        xp.add(total, second, out=total)
        xp.subtract(total, operands.shifted_second, out=total)
        part_products.scale(dtdx, total, out=total)
    elif first_present:
        xp.subtract(operands.shifted_first, first, out=total)
        part_products.scale(dtdx, total, out=total)
    elif second_present:
        xp.subtract(operands.shifted_second, second, out=total)
        part_products.scale(-dtdx, total, out=total)
    else:
        total[...] = 0
    return total


def _curl(xp: Any, snapshot: Dict[str, Any], term: CurlTerm, dtdx: float,
          boundaries: Tuple[str, str, str], phases: BlochPhases,
          parities: MirrorPhases, backward: bool, presence: Tuple[bool, bool],
          reflect_rows: FarReflectRows = (None, None, None),
          scratch: Any = None) -> Any:  # One component's dtdx-scaled curl.
    """Return MEEP's dtdx-scaled curl for one component (:func:`_curl_from_operands`).

    ``backward`` selects MEEP's negated strides (s = -1, the D sub-step) over
    the forward ones (s = +1, the B sub-step). ``phases`` carries the per-axis
    Bloch wrap factor, None on an axis that wraps unphased; ``parities`` carries
    the per-axis mirror phase, None on an axis with no plane; ``presence`` the
    partner presence (:func:`curl_presence`); ``reflect_rows`` the stored row a
    folded PERIODIC axis's far face reflects from, None on every other axis
    (:func:`_far_reflect_rows`).
    """
    return _curl_from_operands(
        _curl_operands(xp, snapshot, term, boundaries, phases, parities, backward,
                       reflect_rows, scratch=scratch, gather=presence),
        dtdx,
        presence,
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
              component: Optional[str] = None,
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

    On a periodic axis the wrapped far face is a plain roll, Bloch phase or not.
    The shifted array has Yee shift 0 on the axis, so its stored cell 0 is MEEP's
    OWNED top point ``lc + 2N`` wherever the axis carries a phase (the stored
    window, :func:`_bloch_ghost_operand`), and the word one cell up from the top
    stored cell ``lc + 2N - 2`` is exactly that owned word. The phase lands on the
    UNSHIFTED operand's plane 0 instead.
    """
    shifted = _rolled(xp, field, -1, axis, scratch, scratch_tag)  # roll(f, -1)[i] = f[i+1].
    if boundary == PERIODIC:
        return shifted
    if boundary == MIRROR and reflect_row is not None and component is not None:
        # Folded PERIODIC axis: the sample one past the stored top images stored
        # row `reflect_row` about the second mirror at doubled n_full, with the
        # component's own parity about that plane (same plane phase as the
        # declared mirror — the transform is translation-then-reflection, and at
        # k = 0 the translation carries no factor).
        shifted[_face(axis, -1)] = part_products.scale(
            _symmetry_phase(component, axis, mirror_phase), field[_face(axis, reflect_row)])
        return shifted
    if boundary in (MIRROR, METALLIC, CYL_AXIS):
        shifted[_face(axis, -1)] = 0
        return shifted
    raise ValueError(f"unknown boundary condition {boundary!r} on axis {AXIS_NAMES[axis]}")


def _shift_down(xp: Any, field: Any, axis: int, boundary: str, component: str,
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

    The periodic near face is a plain roll, Bloch phase or not. Its consumer is a
    component with Yee shift 0 on the axis, whose stored cell 0 is MEEP's owned top
    point ``lc + 2N`` wherever the axis carries a phase (the stored window,
    :func:`_bloch_ghost_operand`): one half cell down from there is the shifted
    array's top stored cell ``lc + 2N - 1``, an owned word. The phase lands on the
    UNSHIFTED operand's plane 0 instead, as ``eikna`` (the word above the window),
    while the B side's reads below the window take ``conj(eikna)``; taking the same
    factor in both directions is the classic sign error, which leaves every
    magnitude plausible and moves only the phase.
    """
    shifted = _rolled(xp, field, 1, axis, scratch, scratch_tag)  # roll(f, +1)[i] = f[i-1].
    if boundary == PERIODIC:
        return shifted
    if boundary == MIRROR:
        shifted[_face(axis, 0)] = part_products.scale(
            _symmetry_phase(component, axis, mirror_phase), _mirror_source(field, axis))
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
        shifted[_face(axis, 0)] = part_products.scale(sign, field[_face(axis, 0)])
        return shifted
    raise ValueError(f"unknown boundary condition {boundary!r} on axis {AXIS_NAMES[axis]}")


def _bloch_ghost_operand(xp: Any, field: Any, axis: int, boundary: str,
                         phase: Optional[complex], below: bool, scratch: Any = None,
                         scratch_tag: str = "bloch_ghost") -> Any:  # Same-index read across the window edge.
    """A stencil operand read at the SAME stored index, with plane 0 as MEEP's not-owned word.

    THE STORED WINDOW. On an axis MEEP makes Bloch-periodic it OWNS, for a component
    with Yee shift 0 there, the points ``lc + 2 .. lc + 2N``, and holds ``lc`` as a
    not-owned word that ``step_boundaries`` fills from the owned ``lc + 2N``
    (boundaries.cpp:203-230, step.cpp:171-221). This engine stores the same window:
    stored cell 0 of a shift-0 component on a phased axis IS the owned top point
    ``lc + 2N`` (:meth:`~.grid.Grid.wrap_count` states the rule for every positional
    reader), and the shift-1 cells ``lc + 1 + 2i`` are owned by both. A read one
    stored cell over is then always an owned word (:func:`_shift_up`,
    :func:`_shift_down`: plain rolls); the reads that leave the window are the ones at
    the SAME stored index across the half cell, on plane 0:

    * ``below=True``: a shift-1 target reads its shift-0 neighbour half a cell down
      (B's forward curl at cell 0 reads E at ``lc``): the not-owned word below the
      window, ``conj(eikna)`` times stored cell 0;
    * ``below=False``: a shift-0 target reads its shift-1 neighbour half a cell up
      (D's backward curl at cell 0, which is ``lc + 2N``, reads H at ``lc + 2N + 1``):
      the not-owned word above the window, ``eikna`` times stored cell 0.

    The word is MEEP's: ``thephase = 1.0 * conj(eikna)`` (or ``* eikna``) in
    ``complex<double>`` (``locate_point_in_user_volume``, boundaries.cpp:203-226),
    connected by ``connect_phase`` -- a copy within 1e-13 of +1, a negation of -1,
    else ``complex<realnum>(thephase) * word`` with every operation rounded on its own
    (:func:`~.part_products.connect`). NumPy's complex multiply contracts on this
    host and on x86-64, so it is not used.

    Returns ``field`` itself when the axis wraps unphased or does not wrap; otherwise
    a copy (pooled under ``scratch_tag``) -- the stored field is never written. Real
    storage cannot carry a phase, which :func:`_bloch_phases` refuses before the
    first curl.
    """
    if phase is None or boundary != PERIODIC:
        return field
    eikna = complex(phase)
    factor = (eikna.real, -eikna.imag) if below else (eikna.real, eikna.imag)
    thephase = cmul((1.0, 0.0), factor)
    out = field.copy() if scratch is None else scratch.copy_of(scratch_tag, field)
    out[_face(axis, 0)] = part_products.connect(field[_face(axis, 0)], thephase)
    return out


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
    part_products.scale(kms, fu, out=fu)
    fu -= curl
    part_products.scale(sinv, fu, out=fu)
    part_products.scale(kms_u, field, out=field)
    field += fu
    field -= fu_previous
    part_products.scale(sinv_u, field, out=field)


def _apply_dsig_only_update(field: Any, curl: Any, kms: Any, sinv: Any) -> None:  # Q: no auxiliary.
    """Run MEEP's dsig-only curl form in place: ``f = ((kms * f) - curl) * sinv``.

    step_generic.cpp:183, the branch of a chunk that carries the dsig layer and not
    the dsigu one. Where the chunk carries neither, MEEP runs plain ``f - curl``
    (:104); with ``kms = sinv = 1.0f`` exactly — every coefficient outside its chunk,
    the region plan's precondition — this IS that word: ``1*x`` and ``x*1`` are exact
    in IEEE-754, signed zero and subnormals included. So one form covers every cell
    outside the dsigu chunks, and it reads and writes the field alone. Multiplication
    operand order is free (IEEE multiplication is commutative), so ``f *= kms`` is
    MEEP's ``kms * f``; on complex storage each product runs per part
    (:func:`~.part_products.scale`).
    """
    part_products.scale(kms, field, out=field)
    field -= curl
    part_products.scale(sinv, field, out=field)


def _region_plan(fields: "Fields", pml: "PML") -> Any:
    """The :class:`~.chunk_map.RegionPlan` this run steps its PML forms with.

    The one the driver installed (:meth:`~.fields.Fields.install_region_plan`, built
    from its chunk division: the lifted simulation's own, or the one-rank default)
    while it was built for this ``pml`` and grid; otherwise the one-rank division of
    this grid and layer, built once and installed — the plan a caller stepping
    ``Fields`` directly, with no driver, gets.
    """
    plan = fields.region_plan
    if plan is not None and plan.pml is pml and plan.grid is fields.grid:
        return plan
    from .chunk_map import ChunkLayout, RegionPlan, region_backend  # noqa: PLC0415 - cycle

    grid = fields.grid
    plan = RegionPlan(ChunkLayout.one_rank(grid, pml), grid, pml, backend=region_backend(grid.xp))
    fields.install_region_plan(plan)
    return plan


def _cut(array: Any, index: Tuple[Any, ...], axis: int) -> Any:
    """``array[index]`` along ``axis`` — unless the array broadcasts there (a size-1 axis)."""
    return array if array.shape[axis] == 1 else array[index]


def _apply_curl_regions(plan: Any, component: str, field: Any, curl: Any, kms: Any, sinv: Any,
                        kms_u: Any, sinv_u: Any, fu: Any, xp: Any, scratch: Any = None) -> None:
    """One curl component's two forms, each on its own cells (the auxiliary contract).

    The recurrence (:func:`_apply_pml_update`) on the cells whose chunk carries the
    component's dsigu layer, the dsig-only form (:func:`_apply_dsig_only_update`)
    everywhere else; ``fu`` keeps its word on every cell outside the recurrence's.
    The plan's route decides how: ``split`` runs each form on views of its runs along
    the dsigu axis; ``select`` and ``mask`` run both whole-grid, select the field on
    the line or the full volume, and restore ``fu`` outside. All three store the same
    word in every array (``test_region_dispatch.py``).
    """
    spec = plan.spec("curl", component)
    mode = spec.mode
    if mode == "outside_only":
        _apply_dsig_only_update(field, curl, kms, sinv)
        return
    if mode == "inside_only":
        _apply_pml_update(field, curl, kms, sinv, kms_u, sinv_u, fu, scratch=scratch)
        return
    axis = spec.axis
    if mode == "split":
        for start, stop in spec.inside:
            index = _span(axis, start, stop)
            _apply_pml_update(_cut(field, index, axis), _cut(curl, index, axis),
                              _cut(kms, index, axis), _cut(sinv, index, axis),
                              _cut(kms_u, index, axis), _cut(sinv_u, index, axis),
                              _cut(fu, index, axis), scratch=scratch)
        for start, stop in spec.outside:
            index = _span(axis, start, stop)
            _apply_dsig_only_update(_cut(field, index, axis), _cut(curl, index, axis),
                                    _cut(kms, index, axis), _cut(sinv, index, axis))
        return
    # select / mask: both forms on the whole grid, then the outside cells take the
    # dsig-only word and fu its previous one.
    outside = plan.mask("curl", component, xp, outside=True)
    dsig_only = field.copy() if scratch is None else scratch.copy_of("region_form", field)
    _apply_dsig_only_update(dsig_only, curl, kms, sinv)
    kept = fu.copy() if scratch is None else scratch.copy_of("region_aux", fu)
    _apply_pml_update(field, curl, kms, sinv, kms_u, sinv_u, fu, scratch=scratch)
    xp.copyto(field, dsig_only, where=outside)
    xp.copyto(fu, kept, where=outside)


def _apply_constitutive_regions(plan: Any, component: str, field: Any, source: Any, kps: Any,
                                kms: Any, fw: Any, xp: Any, scratch: Any = None) -> None:
    """One constitutive component's two forms, each on its own cells (the auxiliary contract).

    The accumulation (:func:`_apply_constitutive_pml`) on the cells whose chunk
    carries the component's own (dsigw) layer; ``f = src`` everywhere else —
    ``H = B``, ``E = (D - P) * chi1inv`` — MEEP's plain constitutive
    (update_eh.cpp:162-170; in a chunk without dsigw MEEP's H IS B's storage,
    fields.cpp:485-494). ``f_w`` keeps its word outside. Routes as
    :func:`_apply_curl_regions`.
    """
    spec = plan.spec("constitutive", component)
    mode = spec.mode
    if mode == "outside_only":
        field[...] = source
        return
    if mode == "inside_only":
        _apply_constitutive_pml(field, source, kps, kms, fw, scratch=scratch)
        return
    axis = spec.axis
    if mode == "split":
        for start, stop in spec.inside:
            index = _span(axis, start, stop)
            _apply_constitutive_pml(_cut(field, index, axis), _cut(source, index, axis),
                                    _cut(kps, index, axis), _cut(kms, index, axis),
                                    _cut(fw, index, axis), scratch=scratch)
        for start, stop in spec.outside:
            index = _span(axis, start, stop)
            field[index] = _cut(source, index, axis)
        return
    # select / mask. The accumulation leaves f_w == src on every cell, so the outside
    # cells take their field word from f_w BEFORE f_w is restored: ``source`` itself
    # may be a pooled buffer the accumulation's own product reused.
    outside = plan.mask("constitutive", component, xp, outside=True)
    kept = fw.copy() if scratch is None else scratch.copy_of("region_aux", fw)
    _apply_constitutive_pml(field, source, kps, kms, fw, scratch=scratch)
    xp.copyto(field, fw, where=outside)
    xp.copyto(fw, kept, where=outside)


def separate_h_from_b(fields: "Fields", pml: "PML") -> None:
    """MEEP's start of H and its ``f_w``: copies of B, on the dsigw cells, at the first ``update_H``.

    MEEP allocates no H array at first: ``alloc_f`` points H at B's storage
    (fields.cpp:485-494). The first ``update_eh(H_stuff)`` in a chunk that carries
    H's own layer then allocates H as a COPY of B and ``f_w`` as a copy of that H
    (update_eh.cpp:165-177) — after ``step_db(B)`` and ``step_source(B)``, so a
    magnetic current deposited inside the layer before that first update is in both.
    This engine allocates H and ``f_w`` at zero; the two starts agree wherever B is
    zero at that moment, and differ where a magnetic source has already written B in
    a dsigw cell whose ``kms`` is not 1 (MEASURED by phase 0's P0-f, the in-layer
    magnetic cases: the copy start reproduces MEEP's step-1 H word 4 of 4 where the
    zero start reproduces 0 of 4).

    Runs once per run: :attr:`~.fields.Fields.h_separated` records that it has. A
    state seeded mid-run (a capsule, a test's random volumes) carries ``True``.
    Elsewhere H is B's word on every step anyway (:func:`_apply_constitutive_regions`).
    """
    if fields.h_separated:
        return
    plan = _region_plan(fields, pml)
    xp = fields.grid.xp
    for component, source, _axis in H_CONSTITUTIVE_TERMS:
        mode = plan.spec("constitutive", component).mode
        if mode == "outside_only":
            continue
        field = getattr(fields, component)
        fw = getattr(fields, "f_w_" + component)
        magnetic = getattr(fields, source)
        if mode == "inside_only":
            field[...] = magnetic
            fw[...] = magnetic
            continue
        inside = plan.mask("constitutive", component, xp)
        xp.copyto(field, magnetic, where=inside)
        xp.copyto(fw, magnetic, where=inside)
    fields.h_separated = True


def _apply_conductive_update(field: Any, curl: Any, condfac: Any, condinv: Any) -> None:  # Lossy D, no PML.
    """Run the D-conductivity curl update in place, MEEP step_generic.cpp:87-97::

        f[i] = ((1 - dt/2*cnd[i]) * f[i] - dtdx * curl) * cndinv[i]

    ``curl`` already carries the dtdx factor, and ``condfac``/``condinv`` are the
    two float32 volumes :meth:`~.fields.Fields.set_d_conductivity` derives. The loss
    is applied to D, not to E: MEEP's sigma_D enters the equations as ``sigma_D * D``
    rather than the textbook ``sigma * E``, so it differs from the usual electric
    conductivity by a factor of epsilon (doc/docs/Materials.md).
    """
    part_products.scale(condfac, field, out=field)
    field -= curl
    part_products.scale(condinv, field, out=field)


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
    *,
    dsig_active: Any,
    dsigu_active: Any,
    conductive: Any,
    scratch: Any = None,
) -> None:
    """Run MEEP's ``step_curl`` for a component that carries a conductivity, under a PML.

    ``step_generic.cpp::step_curl`` picks one of eight forms per structure chunk, from
    three per-chunk facts: whether the chunk carries the dsig layer and the dsigu layer
    (``sigsize[d] > 1``, step_db.cpp:56-59) and whether it keeps a conductivity array
    for the component (``s->conductivity[c][d]`` is NULL in a chunk whose sampled
    conductivity is all zero, structure.cpp:883-900). ``X`` is the curl as handed in
    (``dtdx`` applied), ``cf = condfac``, ``ci = condinv``; every operation rounds to
    float32 in this order:

    * conductive, dsig and dsigu (:200-210)::

          fcnd' = ((cf * fcnd) - X) * ci
          fu'   = ((kms * fu) + (fcnd' - fcnd)) * sinv
          f'    = (((kms_u * f) + fu') - fu) * sinv_u

    * conductive, dsig only (:160-168): ``fcnd'`` as above, then
      ``f' = ((kms * f) + (fcnd' - fcnd)) * sinv``;
    * conductive, dsigu only (:112-119): ``fu' = ((cf * fu) - X) * ci``, then ``f'`` as in
      the first form;
    * conductive, neither (:84-90): ``f' = ((cf * f) - X) * ci``;
    * a chunk with no conductivity: the non-conductive forms, :func:`_apply_pml_update`
      where the chunk carries dsigu and :func:`_apply_dsig_only_update` elsewhere
      (:104, :138-141, :182-183, :231-234) -- the one transcription of each the
      lossless curl uses.

    The increment ``(fcnd' - fcnd)`` is formed FIRST and then added: MEEP's grouping
    (:167, :209). ``((kms * x) + fcnd') - fcnd`` is a different float32 number.

    Where a chunk without conductivity does not carry dsig, its forms and the
    conductive ones store the same word: every sample of such a chunk is zero, so
    ``cf = ci = 1.0`` exactly and ``((1 * x) - X) * 1`` is ``x - X``. Where it carries
    dsig they differ, because the conductive form runs the ``f_cond`` stage --
    ``(fcnd - X) - fcnd`` is not ``-X`` in float32. So the chunk's conductivity decides
    the word only on dsig cells, and the selection below follows MEEP's chunk
    structure on every cell anyway.

    THE MEMBERSHIPS ARE DATA. ``dsig_active`` / ``dsigu_active`` are the chunk
    memberships of the two layers on this component's lattice (the run's
    :class:`~.chunk_map.RegionPlan`; broadcastable bool lines or full volumes), and
    ``conductive`` is the plan's per-cell volume of the chunks that keep the
    component's conductivity, or ``None`` when every chunk keeps it. A chunk that
    carries a layer owns cells whose own sigma is exactly zero, and a chunk that keeps
    a conductivity owns cells whose own conductivity is zero.

    THE AUXILIARY CONTRACT: ``fu`` is written only on the dsigu cells and ``f_cond``
    only on the dsig cells of the chunks that keep a conductivity -- the cells whose
    form reads them -- and both keep their words everywhere else.

    THIS IS THE ENGINE'S MOST EXPENSIVE CURL, reached by any run combining a material
    conductivity with a PML. It evaluates every form on the whole volume and selects
    with ``copyto(..., where=mask)``, which allocates nothing; the full-volume
    temporaries come from the :class:`~.fields.StepScratch` pool under tags of their
    own (``previous`` and the curl/stencil tags are live on this path). Where some
    chunk keeps no conductivity the two non-conductive forms run as well, on copies.
    """
    def _copy(tag: str, array: Any) -> Any:
        return array.copy() if scratch is None else scratch.copy_of(tag, array)

    dsig_inactive = xp.logical_not(dsig_active)
    dsigu_inactive = xp.logical_not(dsigu_active)

    # The chunks with no conductivity: the non-conductive forms, on copies, before the
    # conductive ones overwrite anything they read.
    plain_field = plain_fu = None
    if conductive is not None:
        plain_field = _copy("cond_plain_field", field)
        plain_fu = _copy("cond_plain_fu", fu)
        _apply_pml_update(plain_field, curl, kms, sinv, kms_u, sinv_u, plain_fu, scratch=scratch)
        dsig_only = _copy("cond_plain_dsig_only", field)
        _apply_dsig_only_update(dsig_only, curl, kms, sinv)
        xp.copyto(plain_field, dsig_only, where=dsigu_inactive)

    field_previous = _copy("cond_field_previous", field)
    fu_previous = _copy("cond_fu_previous", fu)

    scale = part_products.scale  # Every product below is a realnum times a field, per part.

    # The conductive stage: fcnd' = ((cf * fcnd) - X) * ci, and its increment
    # (fcnd' - fcnd_prev), formed before it is added anywhere.
    f_cond_next = _copy("cond_f_cond_next", f_cond)
    scale(condfac, f_cond_next, out=f_cond_next)
    f_cond_next -= curl
    scale(condinv, f_cond_next, out=f_cond_next)
    increment = _copy("cond_increment", f_cond_next)
    increment -= f_cond

    # fu: ((kms * fu) + increment) * sinv where dsig is carried, ((cf * fu) - X) * ci
    # where only dsigu is.
    fu_dsigu_only = _copy("cond_fu_dsigu_only", fu_previous)
    scale(condfac, fu_dsigu_only, out=fu_dsigu_only)
    fu_dsigu_only -= curl
    scale(condinv, fu_dsigu_only, out=fu_dsigu_only)
    scale(kms, fu, out=fu)
    fu += increment
    scale(sinv, fu, out=fu)
    xp.copyto(fu, fu_dsigu_only, where=dsig_inactive)

    # f where dsigu is carried: (((kms_u * f) + fu') - fu_prev) * sinv_u.
    scale(kms_u, field, out=field)
    field += fu
    field -= fu_previous
    scale(sinv_u, field, out=field)

    # f where only dsig is carried: ((kms * f) + increment) * sinv.
    dsig_form = _copy("cond_dsig_form", field_previous)
    scale(kms, dsig_form, out=dsig_form)
    dsig_form += increment
    scale(sinv, dsig_form, out=dsig_form)

    # f where neither is: ((cf * f) - X) * ci.
    direct = field_previous
    scale(condfac, direct, out=direct)
    direct -= curl
    scale(condinv, direct, out=direct)

    xp.copyto(field, dsig_form, where=dsigu_inactive & dsig_active)
    xp.copyto(field, direct, where=dsigu_inactive & dsig_inactive)
    if conductive is not None:
        not_conductive = xp.logical_not(conductive)
        xp.copyto(field, plain_field, where=not_conductive)
        xp.copyto(fu, plain_fu, where=not_conductive)
        xp.copyto(f_cond, f_cond_next, where=conductive & dsig_active)
    else:
        xp.copyto(f_cond, f_cond_next, where=dsig_active)
    # fu keeps its word outside the dsigu cells, on either kind of chunk.
    xp.copyto(fu, fu_previous, where=dsigu_inactive)


def _apply_constitutive_pml(field: Any, source: Any, kps: Any, kms: Any, fw: Any,
                            scratch: Any = None) -> None:  # dsigw accumulation.
    """Run the dsigw constitutive PML accumulation in place.

    MEEP step_generic.cpp step_update_EDHB with dsigw active::

        realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
        fw[i] = g[i] * u[i];                   // B for H, D*inv_eps for E
        f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;

    (step_generic.cpp:689, :697). The right-hand side is formed first, so the update is
    ``f + ((kps * fw) - (kms * fwprev))``: the two products, their difference, then one
    add into ``f``. ``(f + kps * fw) - kms * fwprev`` is a different float32 number.

    The expression form allocates THREE full-volume temporaries per component per
    step — the ``fwprev`` copy and the two coefficient products — and ran second in
    the profile at 22% of a 2-D PML step. With a :class:`~.fields.StepScratch` all
    three come from the pool: ``previous`` for the copy (shared with
    :func:`_apply_pml_update`, whose sub-step never overlaps this one), which then
    takes ``kms * fwprev`` in place, and ``constitutive`` for ``kps * fw`` and the
    difference.
    """
    if scratch is None:
        fw_previous = fw.copy()
        fw[...] = source
        field += part_products.scale(kps, fw) - part_products.scale(kms, fw_previous)
        return
    xp = scratch.xp
    fw_previous = scratch.copy_of("previous", fw)
    fw[...] = source
    product = scratch.take("constitutive", fw.shape, xp.result_type(kps, fw))
    part_products.scale(kps, fw, out=product)
    part_products.scale(kms, fw_previous, out=fw_previous)
    xp.subtract(product, fw_previous, out=product)
    field += product


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
        part_products.scale(inverse, displacement, out=target)
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
