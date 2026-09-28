"""
Grid is the computational-domain descriptor for the MEEP-compatible FDTD engine — the analog
of MEEP's ``meep::grid_volume``. It turns a requested cell size and resolution
into the integer Yee-grid dimensions every other module allocates against, and
owns the five conventions the rest of the engine depends on: cell counts
(``int(L*resolution + 0.5)``), the timestep (``courant/resolution``), mirror
symmetry (any of X, Y, Z halved to MEEP's ``n_full - n_full//2 + 1`` owned cells
— a folded PERIODIC axis stores one more at either parity, the ``big_corner``
plane and its ghost slot — MEEP's
``halve()``, both count parities — with cell 0 straddling the mirror plane at
-0.5*dx, each plane carrying MEEP's declared
``phase``), the outer boundary condition of each axis (``boundaries``: periodic
or metallic, per axis and per side), the Bloch phase a field picks up
crossing a periodic boundary (``k_point``), and the dimensionality of the run
(``dimensions``: 3, or 2/1 with the missing axes carried as translational
invariance).

Inputs: resolution (cells per unit length), cell_size (Lx, Ly, Lz), courant
factor, a ``symmetry`` sequence of :class:`Mirror` planes (a bare axis name is
shorthand for an even one), Bloch wavevector, the per-axis ``boundaries``
declaration, ``dimensions``, and the array module ``xp`` (NumPy by default,
CuPy on a CUDA host). Outputs: ``dx``/``dt``, the stored and
full-domain shapes, float32 cell-center coordinate arrays, ``bloch_phase`` — the
per-axis wrap factor the stepping kernels and the cell-centred readback apply —
``mirror_phase``, the per-axis mirror phase the fold and the ghost cells apply,
``boundary_condition``/``is_metallic``, the declared outer condition of a face,
``axis_wraps``, which says whether an axis has a lattice vector at all, and
``position_to_index``, the strictly-in-range position-to-Yee-index mapping used
to name a single stored cell. Placement that must survive a request past the end
of a wrapping axis goes through ``sources._build_source_points`` instead, which
folds onto the lattice and carries the Bloch phase.

Units are MEEP natural units (c = 1, frequency = 1/wavelength); host
applications perform any external unit conversion.

MEEP source references:
- meep/vec.hpp: grid_volume, yee_grid_point, iyee_shift, interpolate_linear,
  ndim/start_at_direction/stop_at_direction/LOOP_OVER_DIRECTIONS
- meep/vec.cpp: grid_volume::halve(), mirror(), symmetry::operator*, vol1d/vol2d/vol3d
- meep/step.cpp: dt = Courant / a
- meep/fields.cpp: fields::fields (boundaries[b][d] = Metallic by default)
- meep/boundaries.cpp: fields::use_bloch (eikna), locate_point_in_user_volume,
  on_metal_boundary, find_metals

Yee staggering (meep/vec.hpp): Ex at (i+0.5, j, k)*dx, Ey at (i, j+0.5, k)*dx,
Ez at (i, j, k+0.5)*dx, Hx at (i, j+0.5, k+0.5)*dx, Hy at (i+0.5, j, k+0.5)*dx,
Hz at (i+0.5, j+0.5, k)*dx.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

import numpy

MIRROR_AXES = ("X", "Y", "Z")  # Axes a mirror plane may be normal to; all three are foldable.
AXIS_NAMES = ("x", "y", "z")  # Axis order of k_point, cell_size, and the field-array axes.

# Which axes MEEP drops for each ``dimensions`` value, in this engine's x, y, z order.
# The assignment is MEEP's, not a convention chosen here: ``start_at_direction`` and
# ``stop_at_direction`` (vec.hpp) make ``LOOP_OVER_DIRECTIONS`` run X, Y for D2 and Z
# alone for D1, and ``Simulation._create_grid_volume`` builds them as
# ``vol2d(cell_size.x, cell_size.y, a)`` and ``vol1d(cell_size.z, a)``. So MEEP 2-D is
# the X-Y PLANE with z invariant, and MEEP 1-D is the Z AXIS with x and y invariant —
# a 1-D run is NOT "the x axis", and guessing it that way would rotate every source
# component and every polarization by 90 degrees while still stepping smoothly.
INVARIANT_AXES_BY_DIMENSIONS = {3: (), 2: (2,), 1: (0, 1)}
SUPPORTED_DIMENSIONS = tuple(sorted(INVARIANT_AXES_BY_DIMENSIONS))

# The two outer-boundary conditions this engine steps, named as MEEP names them
# (meep.hpp ``boundary_condition``). MEEP has five; the other three are Magnetic
# (a perfect magnetic conductor), None (an interior chunk face) and the unused
# Symmetric — see :func:`_normalize_boundaries` for why they are refused rather
# than approximated by one of these.
PERIODIC = "periodic"  # MEEP Periodic: the face IS a lattice vector; a field wrapping it may carry a Bloch phase.
METALLIC = "metallic"  # MEEP Metallic: a perfect electric conductor wall — tangential E and normal B are zero on it.
# The cylindrical radial axis: NOT a caller-facing condition (only Grid's own
# cylindrical mode installs it, always as the LOW side of r with a metallic high
# wall). MEEP's Dcyl cell has no low-R boundary at all when it touches the axis
# (grid_volume::has_boundary, vec.cpp:465-473); stencil reads below r = 0 are
# served by ``r_to_minus_r_symmetry(m)`` ghosts — the field's own image with R-
# and P-direction components sign-flipped and an overall ``(-1)^m`` — and the
# axis row itself is written by the per-m rules in step_db.cpp:299-460 rather
# than by a wall condition.
AXIS = "axis"
BOUNDARY_CONDITIONS = (PERIODIC, METALLIC)
SIDES = ("low", "high")  # Face of an axis, in ascending-coordinate order.
# MEEP's own boundary_side enum is High = 0, Low = 1 (meep.hpp) — the reverse of the
# obvious reading, and already a measured trap in from_meep's PML face table. Sides are
# named here rather than numbered so the order cannot be got wrong silently.


@dataclass(frozen=True)
class Mirror:
    """One mirror plane: the axis it is normal to, and the phase its transform carries.

    The translation of ``mp.Mirror(direction, phase)``. MEEP builds it as
    ``mirror(axis, gv) * phase`` (python/simulation.py ``_create_symmetries``),
    where ``meep::mirror`` (vec.cpp) flips only ``axis`` and
    ``symmetry::operator*`` stores the factor in ``symmetry::ph``. That factor
    multiplies EVERY component's parity about the plane
    (``symmetry::phase_shift``), so ``phase=-1`` — an ODD mirror — is not a
    different fold, it is the same fold with every sign inverted. The parity a
    given component ends up with is derived in ``fields.mirror_parity``; this
    class carries only what the caller declared.

    ``phase`` is restricted to +1 and -1. A mirror is an involution, so the
    represented phase must square to 1; MEEP's ``symmetry::ph`` is a general
    complex only because the same slot serves ``r_to_minus_r_symmetry``, whose
    ``exp(i*m*pi)`` belongs to a rotation about the cylindrical axis rather than
    to a Cartesian mirror. Anything else is refused rather than being folded with
    a phase that does not compose back to the identity.

    Attributes:
        axis: 'X', 'Y', or 'Z' — the direction NORMAL to the plane, as in MEEP.
        phase: +1 for an even mirror (MEEP's default), -1 for an odd one.
    """

    axis: str
    phase: int = 1

    def __post_init__(self):  # Normalize the axis name and refuse a phase that is not an involution.
        axis = str(self.axis).upper()
        if axis not in MIRROR_AXES:
            raise ValueError(
                f"mirror axis must be one of {list(MIRROR_AXES)}, got {self.axis!r}"
            )
        object.__setattr__(self, "axis", axis)
        if self.phase not in (1, -1):
            raise ValueError(
                f"mirror phase must be +1 (even) or -1 (odd), got {self.phase!r}: a mirror is "
                f"its own inverse, so its phase has to square to 1. MEEP's complex "
                f"symmetry::ph exists for r_to_minus_r_symmetry (cylindrical m), not for a "
                f"Cartesian mirror."
            )
        object.__setattr__(self, "phase", int(self.phase))

    @property
    def axis_index(self) -> int:  # 0 for X, 1 for Y, 2 for Z.
        return MIRROR_AXES.index(self.axis)

    def __repr__(self) -> str:  # Signed phase, so an odd plane cannot be mistaken for an even one.
        return f"Mirror({self.axis!r}, phase={self.phase:+d})"


MirrorSpec = Union[Mirror, str]  # A plane, or the bare axis name that is shorthand for an even one.


def _normalize_mirrors(symmetry: Iterable[MirrorSpec]) -> tuple[Mirror, ...]:
    """Turn a ``symmetry`` request into mirror planes, one per axis at most.

    A bare axis name is the shorthand for an even mirror, which is MEEP's own
    default phase and every mirror this engine had before phases existed. Two
    planes normal to the same axis are refused: a second one is either a
    duplicate or a contradiction (an even and an odd mirror about the same plane
    force the field to be both symmetric and antisymmetric, i.e. zero), and
    neither is something to resolve silently by taking the last entry.
    """
    mirrors: list[Mirror] = []
    for entry in tuple(symmetry):
        plane = entry if isinstance(entry, Mirror) else Mirror(entry)
        clash = next((existing for existing in mirrors if existing.axis == plane.axis), None)
        if clash is not None:
            raise ValueError(
                f"two mirror planes normal to {plane.axis}: {clash!r} and {plane!r}. One axis "
                f"carries one plane; an even and an odd mirror about the same plane would force "
                f"the field to be both even and odd about it, which only the zero field is."
            )
        mirrors.append(plane)
    return tuple(mirrors)


BoundarySpec = Any  # None, one condition, a 3-sequence, or a per-axis mapping; see below.


def _normalize_boundaries(boundaries: BoundarySpec) -> tuple[tuple[str, str], ...]:
    """Turn an outer-boundary request into one ``(low, high)`` pair per axis.

    THE DEFAULT IS PERIODIC HERE AND METALLIC IN MEEP, and the mismatch is
    deliberate rather than an oversight. MEEP's ``fields::fields`` sets
    ``boundaries[b][d] = Metallic`` on every face and only ``use_bloch`` overwrites
    it with ``Periodic`` (fields.cpp, boundaries.cpp); this engine keeps periodic as
    the default because every run it has ever cross-validated is one, and flipping
    the default would change the answer of every existing caller silently. The MEEP
    default is reached by the converter instead: ``from_meep`` reads ``sim.k_point``
    and asks for metallic walls when it is unset, which is the one place that knows
    what MEEP would have built.

    Accepted spellings, all resolving to the same normalized tuple:

    * ``None`` or ``()`` — periodic on every axis, the historical behaviour, and the
      one spelling that must stay BYTE-IDENTICAL to the engine before this argument
      existed (nothing downstream builds a metallic object for it);
    * a single condition, ``"metallic"`` — that condition on all six faces;
    * a mapping keyed by axis name, ``{"x": "metallic"}`` — the named axes take their
      condition and every unnamed axis stays periodic, matching how ``setup_pml``
      spells a per-axis request;
    * a 3-sequence, one entry per axis in x, y, z order.

    Each per-axis entry is either one condition (both faces) or a ``(low, high)``
    pair. The pair is representable because MEEP's own table is per side, but the two
    sides of one axis are then required to AGREE, and that refusal is the point of
    this function. MEEP decides whether a face translates by reading
    ``boundaries[High][d]`` alone (boundaries.cpp ``locate_point_in_user_volume``),
    so ``("metallic", "periodic")`` is a run whose low-side declaration MEEP ignores
    entirely: it steps as fully periodic and only the array slice differs. Accepting
    it here would let a caller write a wall MEEP does not build.
    """
    if boundaries is None:
        return ((PERIODIC, PERIODIC),) * 3
    if isinstance(boundaries, str):
        entries: list[Any] = [boundaries] * 3
    elif isinstance(boundaries, dict):
        unknown = [key for key in boundaries if key not in AXIS_NAMES]
        if unknown:
            raise ValueError(
                f"boundaries names {unknown}, which are not axes; keys must be from "
                f"{list(AXIS_NAMES)} (an axis left out stays {PERIODIC!r})"
            )
        entries = [boundaries.get(name, PERIODIC) for name in AXIS_NAMES]
    else:
        entries = list(boundaries)
        if not entries:
            return ((PERIODIC, PERIODIC),) * 3
        if len(entries) != 3:
            raise ValueError(
                f"boundaries must give one entry per axis (x, y, z), got {len(entries)}: "
                f"{boundaries!r}"
            )
    resolved: list[tuple[str, str]] = []
    for axis, entry in enumerate(entries):
        if isinstance(entry, str):
            sides = (entry, entry)
        else:
            try:
                low, high = tuple(entry)
            except (TypeError, ValueError):
                raise ValueError(
                    f"boundaries entry for the {AXIS_NAMES[axis]} axis must be a condition or a "
                    f"(low, high) pair of them, got {entry!r}"
                ) from None
            sides = (low, high)
        for side, condition in zip(SIDES, sides):
            if condition not in BOUNDARY_CONDITIONS:
                raise ValueError(
                    f"boundary condition on the {side} {AXIS_NAMES[axis]} face is {condition!r}; "
                    f"this engine steps {list(BOUNDARY_CONDITIONS)}. MEEP's Magnetic (a perfect "
                    f"MAGNETIC conductor) and None (an interior chunk face) are not implemented, "
                    f"and neither is approximated by the other two: a magnetic wall reverses which "
                    f"components the wall zeroes, so running it as metallic returns a smooth field "
                    f"for the complementary problem."
                )
        if sides[0] != sides[1]:
            raise ValueError(
                f"the {AXIS_NAMES[axis]} axis declares {sides[0]!r} on its low face and "
                f"{sides[1]!r} on its high face. One axis carries one condition: MEEP decides "
                f"whether a face translates from boundaries[High][d] alone "
                f"(boundaries.cpp locate_point_in_user_volume), so a mixed axis is a run whose "
                f"low-side declaration MEEP silently ignores — it would step as "
                f"{sides[1]!r} on both faces. Declare the condition you mean on both."
            )
        resolved.append((str(sides[0]), str(sides[1])))
    return tuple(resolved)


# Float slack, in cells, when deciding whether a request is inside an axis: an absolute
# floor for float64 arithmetic plus a term that scales with the axis. The coordinate
# arrays this module publishes are float32 (`_create_coordinates`), so a caller that
# reads `grid.x[i]` and hands it straight back can land a few float32 ULP OUTSIDE its own
# axis — refusing that would turn representation noise into an error. The scaling term is
# the worst rounding of a coordinate anywhere on the axis, which spans at most 2*n cells
# from the origin; even for a 10000-cell axis it stays six orders of magnitude below the
# half cell at which a request would start naming a different sample.
INDEX_TOLERANCE_CELLS = 1e-9
FLOAT32_EPS = float(numpy.finfo(numpy.float32).eps)


def index_tolerance(n: int) -> float:  # Cells of float slack allowed outside an n-cell axis.
    return INDEX_TOLERANCE_CELLS + 2.0 * FLOAT32_EPS * n


def meep_cell_count(length: float, resolution: float) -> int:  # MEEP vec.hpp: int(size*a + 0.5).
    return int(length * resolution + 0.5)


@dataclass
class Grid:
    """MEEP-compatible computational grid (translation of meep::grid_volume).

    Attributes:
        resolution: Grid points per unit length (MEEP: gv.a).
        cell_size: (Lx, Ly, Lz) requested cell dimensions, snapped to n*dx.
        courant: Courant factor (MEEP default 0.5).
        symmetry: Mirror planes, e.g. ``(Mirror('X', -1), 'Y')``. A bare axis
            name is the shorthand for an even mirror; any of X, Y and Z may be
            folded, and __post_init__ replaces the request with the normalized
            :class:`Mirror` tuple.
        k_point: Bloch wavevector (kx, ky, kz) in MEEP's units of 2*pi/distance,
            so a field wrapping up past the far face of axis d is multiplied by
            ``exp(i * 2*pi * k_d * L_d)``. All-zero (the default) is plain
            periodicity and leaves every wrap untouched.
        boundaries: Outer boundary condition per axis — ``"periodic"`` (the
            default, and MEEP's ``use_bloch``) or ``"metallic"`` (MEEP's default
            perfect-electric-conductor wall). Spellable as one condition for all
            three axes, a ``{"x": ...}`` mapping, or a 3-sequence; each entry may be
            a ``(low, high)`` pair, which must agree. See
            :func:`_normalize_boundaries`.
        dimensions: 3 (the default), 2 or 1 — MEEP's ``dimensions``. A reduced
            run carries its missing axes as TRANSLATIONAL INVARIANCE: 2 makes z
            invariant (MEEP's X-Y plane) and 1 makes x and y invariant (MEEP's z
            axis). See :meth:`is_invariant` for what that means to every other
            module, and the class docstring section below for why it is not a
            thin slab.
        beta: MEEP's out-of-plane 2-D wavevector (``fields::beta``, the
            ``special_kz`` mode ``mp.Simulation(kz_2d=...)`` selects). A 2-D run with
            ``beta != 0`` carries an analytic ``exp(i*2*pi*beta*z)`` dependence on the
            invariant axis, so ``d/dz`` is the exact factor ``i*2*pi*beta`` rather
            than a difference; it couples TE to TM in the curl and it is NOT a Bloch
            phase (``k_point`` stays zero on z). Legal only at ``dimensions=2``.
        bfast_scaled_k: MEEP's ``bfast_scaled_k`` — the BROADBAND FIXED-ANGLE SOURCE
            TECHNIQUE, i.e. the scaled wavevector ``n * sin(theta)`` of the incidence
            direction. It shears time by ``t -> t - k.r/c``, which adds
            ``+d/dt (k x E)`` to dB/dt and ``-d/dt (k x H)`` to dD/dt and makes a
            planewave at a FIXED angle transversely uniform at every frequency, so
            ordinary periodic boundaries serve a whole BAND at one angle instead of
            one frequency. Any nonzero component turns the pass on for every
            component. Refused on a cylindrical grid; see :meth:`_resolve_bfast`.
        xp: Array module used for the coordinate arrays and every downstream
            allocation (numpy by default, cupy when a CUDA device is in use).

    Reduced dimensions, in one paragraph. MEEP 2-D is an infinite structure with
    every field constant along z and every d/dz identically zero — not a thin
    slab, which radiates out of plane. MEEP implements it by leaving z out of
    ``LOOP_OVER_DIRECTIONS`` entirely, which sets ``stride(Z) = 0`` (vec.cpp
    ``num_changed``) and makes ``g[i + s_z] - g[i]`` read the same sample twice.
    This engine reproduces it as ONE CELL on the invariant axis with a periodic
    wrap, whose difference is the same sample twice and therefore the same exact
    zero. That is not an approximation of MEEP's arrangement, it IS MEEP's: a
    one-pixel periodic direction is what MEEP itself calls a lower-dimensional
    emulation (fields.cpp ``nosize_direction``), and the two agree to 0.0e+00 —
    bit for bit — on the same cell, source and step count.

    The invariant axis is therefore forced to be periodic, and a metallic wall,
    a mirror plane, a Bloch phase or a nonzero extent on it are all refused. A
    metallic invariant axis is the dangerous one: a PEC wall zeroes exactly the
    components whose Yee shift on that axis is 0, which for an invariant z is
    Ex, Ey and Hz — the entire TE polarization — so a 2-D TE run terminated that
    way returns a perfectly smooth zero, while the TM run beside it is exact.
    """

    resolution: float
    cell_size: tuple[float, float, float]
    courant: float = 0.5  # MEEP default from step.cpp
    symmetry: tuple[MirrorSpec, ...] = ()  # Mirror planes; normalized to Mirror objects.
    k_point: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Bloch wavevector, MEEP units
    boundaries: BoundarySpec = None  # Per-axis outer condition; None is periodic everywhere.
    dimensions: int = 3  # MEEP's dimensions: 3, or 2/1 with the missing axes invariant.
    # MEEP's Dcyl, under the mapping x -> r, y -> phi, z -> z (the Cartesian Yee
    # table lands exactly on the Dcyl shifts; the design notes (fdtd-cylindrical-plan)
    # section 1). phi is the one-cell invariant axis; ``m`` is the azimuthal number
    # of the analytic exp(i*m*phi) dependence.
    cylindrical: bool = False
    m: int = 0
    # MEEP's ``accurate_fields_near_cylorigin`` (the INVERSE of its internal
    # ``fields::zero_fields_near_cylorigin``, set at python/simulation.py:2483). It
    # selects the near-axis treatment for |m| >= 2 only — step_db.cpp has entirely
    # separate branches for m = 0 and |m| = 1, which this flag does not reach — and it
    # is Courant-reducing: MEEP's own comment puts the stable bound at ~1/(|m| + 0.5)
    # without the default's zero rows, against ~0.62 with them.
    accurate_fields_near_cylorigin: bool = False
    # MEEP's ``fields::beta`` — the out-of-plane wavevector a 2-D run carries as an
    # analytic exp(i*2*pi*beta*z) dependence rather than as grid points, which is what
    # ``mp.Simulation(kz_2d=...)`` turns ``k_point.z`` into on a zero-thickness cell
    # (python/simulation.py:1555-1570, fields.cpp:2478-2487). It is NOT a Bloch phase:
    # nothing wraps, and ``k_point`` keeps a hard zero on z.
    beta: float = 0.0
    # MEEP's ``fields::bfast_scaled_k`` — the BROADBAND FIXED-ANGLE SOURCE TECHNIQUE
    # (BFAST). It is a knob on ``fields``, not on ``structure``
    # (python/simulation.py:1239/1537/2486), so it changes NOTHING about epsilon, the
    # PML, the sources or the constitutive relations; grep for "bfast" over MEEP's
    # src/ finds it only in the curl sub-step, in the flux backup/restore pair, in
    # the dump/load state list and in the CW solver's state vector. What it adds is a
    # second additive pass inside ``step_db`` (step_db.cpp:129-142, STEP_BFAST):
    #
    #     dB/dt = -curl(E) + d/dt (k x E)
    #     dD/dt = +curl(H) - d/dt (k x H)
    #
    # which is the time-sheared substitution t -> t - k.r/c. Under it a planewave at
    # a FIXED incidence angle is transversely uniform at EVERY frequency, so plain
    # periodic boundaries (k_point = 0) are exact for a whole band instead of for a
    # single frequency — the reason MEEP's own test asks for one angle across eleven
    # frequencies rather than eleven angles. Components are MEEP's scaled k, i.e.
    # ``n * sin(theta)`` in the incidence plane; ANY nonzero component turns the pass
    # on for every field component (step_db.cpp:65).
    bfast_scaled_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    xp: Any = field(default=numpy, repr=False, compare=False)

    def __post_init__(self):  # Derive dx/dt, integer dimensions, symmetry halving, and coordinates.
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.courant <= 0:
            raise ValueError(f"courant must be positive, got {self.courant}")
        if len(self.cell_size) != 3:
            raise ValueError(f"cell_size must be (Lx, Ly, Lz), got {self.cell_size!r}")

        if self.cylindrical:
            self._resolve_cylindrical()
        else:
            if int(self.m) != 0:
                raise ValueError(
                    f"m={self.m!r} names an azimuthal number, which only the cylindrical "
                    f"mode has; a Cartesian grid has no exp(i*m*phi) to carry."
                )
            self._resolve_dimensions()

        self._resolve_beta()
        self._resolve_bfast()

        # MEEP vec.hpp: inva = 1.0 / a where a = resolution
        self.dx = 1.0 / self.resolution
        # MEEP step.cpp: dt = Courant * inva = Courant / resolution
        self.dt = self.courant / self.resolution

        self.Lx, self.Ly, self.Lz = self.cell_size

        # MEEP vec.hpp: num[d] = int(size[d] * a + 0.5) — truncation of x+0.5, not
        # Python's banker's rounding (they disagree at exact halves). An INVARIANT
        # axis is not measured at all: MEEP's vol2d/vol1d never sees its extent, and
        # the one cell this engine gives it stands for the whole infinite direction.
        counts = [
            1 if self.is_invariant(axis)
            else meep_cell_count(self.cell_size[axis], self.resolution)
            for axis in range(3)
        ]
        self.nx_full, self.ny_full, self.nz_full = counts
        if min(counts) < 1:
            raise ValueError(
                f"cell_size {self.cell_size!r} at resolution {self.resolution} yields an empty "
                f"grid ({self.nx_full}, {self.ny_full}, {self.nz_full})"
            )

        self.symmetry = _normalize_mirrors(self.symmetry)
        self._require_symmetry_is_on_a_real_axis()
        self.mirrors_by_axis = tuple(
            next((plane for plane in self.symmetry if plane.axis_index == axis), None)
            for axis in range(3)
        )
        self.sym_x, self.sym_y, self.sym_z = (
            plane is not None for plane in self.mirrors_by_axis
        )

        # An ODD full count folds too, and it is MEEP's own fold, not an
        # approximation of it. The symmetry point is `icenter() = io +
        # ivec(nx, ny, nz).round_down_to_even()` (vec.cpp:1089-1101, with the
        # comment that icenter - io must be EVEN so transforms preserve the Yee
        # lattice), and the centred origin is `io = -(n - n % 2)`, so the fold
        # plane lands on a grid POINT — doubled coordinate 0 — for BOTH parities,
        # never through a cell centre. What an odd count changes is the WINDOW:
        # the cell spans [-(N-1)dx/2, (N+1)dx/2], half a cell higher than the
        # symmetric even window, with (N-1)/2 cells below the plane and (N+1)/2
        # above (measured: get_array_metadata x in [-0.95, +1.05] for N=21).
        #
        # Measured, MEEP's own folded against unfolded runs of the same script
        # (MEEP 1.33.0 single precision, identical dt and step counts, signal
        # reached, complex relative L2 — i.e. what MEEP's odd fold itself costs):
        #
        #     PERIODIC folded axis, k=0:  3.70e-07 (N=15, res 50) and 4.71e-07
        #         (N=21, res 60) against an even control of 3.16e-07 — the odd
        #         fold is EXACT on a periodic axis; the window shift is pure
        #         bookkeeping under the wrap.
        #     PML-terminated folded axis: interior (outside the PML shell)
        #         1.04e-04 / 1.31e-05 / 3.95e-06 at dpml 1/2/3 — the mirrored
        #         sigma profile is offset one cell in the shell, and the gap
        #         drains away with absorber depth (even control 7.7e-07).
        #
        # What an odd fold is NOT is a symmetric answer to the script as written:
        # on a BARE metallic axis (no absorber) MEEP's own folded run differs from
        # its unfolded run by 2.71e-02 (even control 2.22e-08), because the fold
        # answers the shifted window (implied lower wall at -(N+1)dx/2) rather
        # than the declared one. That is MEEP's semantics for the script, the
        # parity oracle here is the FOLDED run, and three of MEEP's own examples
        # are written this way (binary_grating_phasemap.py ny_full=21,
        # cavity-farfield.py nx_full=1541/ny_full=245, metasurface_lens.py
        # ny_full=15); this engine used to refuse the odd count outright, which
        # kept all three from lifting at all.
        counts_full = (self.nx_full, self.ny_full, self.nz_full)

        # The outer-boundary table resolves BEFORE the stored counts: a mirror
        # fold's storage depends on what terminates the axis (below).
        self.boundaries = _normalize_boundaries(self.boundaries)
        if self.cylindrical:
            # The r axis is the mode's own: the low side is the AXIS (no boundary
            # in MEEP — ghosts come from r_to_minus_r, the row from the per-m
            # rules) and the high side a metallic wall exactly as a Cartesian
            # metallic axis stores it (N cells, wall as the unstored zero ghost).
            # A caller-declared r condition would be a different physical problem,
            # so anything but the default is refused rather than overridden.
            if self.boundaries[0] != (PERIODIC, PERIODIC):
                raise ValueError(
                    f"the cylindrical r axis carries its own boundary pair (the axis at "
                    f"r=0, a metallic wall at r_max); leave the x/r entry of `boundaries` "
                    f"unset, got {self.boundaries[0]!r}"
                )
            self.boundaries = ((AXIS, METALLIC),) + tuple(self.boundaries[1:])
        self.metallic_axes = tuple(pair[1] == METALLIC and pair[0] != AXIS
                                   for pair in self.boundaries)
        self.has_metallic = any(self.metallic_axes)

        # MEEP vec.cpp:1069-1074 halve(): set_num_direction(d, 1 + (big_corner -
        # icenter) / 2) with origin icenter - 2. In doubled coordinates relative to
        # io, big_corner = 2N and icenter = N - N%2, so big_corner - icenter =
        # N + N%2 and MEEP's halved cell count (`owned_cells`) is
        # N - N//2 + 1: N/2 + 1 at even N (the familiar halving) and (N+1)/2 + 1 at
        # odd N — the extra cell is the one whose centre sits at exactly +L/2, the
        # top of the shifted window. Every axis folds the same way — the halving
        # knows nothing about which direction a run happens to propagate along.
        #
        # A folded PERIODIC axis stores ONE CELL MORE than the halved count, at
        # BOTH count parities. `update_ntot` allocates num + 1 slots per direction
        # (vec.cpp:293-296) and `owns` reports every doubled point p with
        # `0 < p - io <= 2*num` as owned and stepped (vec.cpp:445-462), so the
        # halved chunk's top allocation slot carries MEEP's shift-0 sample AT
        # `big_corner` — owned, stepped, real storage — and its shift-1 partner
        # one half-cell above, which is the not-owned ghost `connect_the_chunks`
        # fills and `stepping.fill_folded_far_ghosts_*` reproduces.
        #
        # WHERE THE TWO PARITIES DIFFER is only which plane `big_corner` IS.
        # A mirror at doubled 0 under periodic boundaries implies a SECOND mirror
        # at doubled N (translation + reflection). At an EVEN count `big_corner`
        # is that plane exactly — a shift-0 Yee plane and an independent degree
        # of freedom for every component the second mirror makes even (measured:
        # |Ez| on y = +L/2 is 8.2e-01 of interior under an even mirror, 8.1e-08
        # under an odd one). At an ODD count the second mirror at doubled N is a
        # shift-1 sample the halved count already holds, and `big_corner` sits
        # one half-cell ABOVE it, at doubled N + 1 — the top of the odd count's
        # shifted window.
        #
        # THAT TOP SLOT IS NOT REDUNDANT AT AN ODD COUNT, which is why this
        # engine stored the halved count there and read the sample past it
        # through `stepping._shift_up`'s reflect rule instead. Reflection about
        # the second mirror is exact only where the MEDIUM is symmetric about it,
        # and at an odd count nothing makes it so: MEEP anchors the absorber at
        # the window top (`use_pml(..., user_volume.boundary_location(side, d))`,
        # structure.cpp:226, and `boundary_location` is `loc(Ez, ntot() - 1)`,
        # vec.cpp:692-700), i.e. at doubled N + 1, so `pml_x` (structure.cpp:625-
        # 628) grades sigma about a wall half a cell above the plane the
        # reflection assumes. Measured, whole-volume complex L2 against CPU MEEP's
        # own folded run (6.0 x 4.1 cell at resolution 10, n_full = 41, Mirror(Y),
        # k_point = 0, 250 steps):
        #
        #     absorber on the folded axis: 3.55e-02 / 1.32e-02 / 8.62e-03 at
        #         dpml 0.5 / 1.0 / 1.5, against 2.51e-07 with no absorber there.
        #     a DIELECTRIC edge between the second mirror and the window top,
        #         no absorber at all: 9.74e-01 — the same defect without a PML,
        #         and the even control with the same slab reads 2.62e-07.
        #
        # Both collapse to the fold floor once the slot is stored and stepped.
        #
        # A folded METALLIC axis keeps the halved count: MEEP holds its
        # window-top plane at zero (find_metals / zero_metal, boundaries.cpp:
        # 306-343), which is what the zero ghost already says, and the odd
        # metallic corner is measured clean at 3.13e-07 with the same absorber.
        self.nx, self.ny, self.nz = (
            counts_full[axis] - counts_full[axis] // 2 + 1
            + (1 if (self.mirrors_by_axis[axis]
                     and not self.metallic_axes[axis]) else 0)
            if self.mirrors_by_axis[axis] else counts_full[axis]
            for axis in range(3)
        )

        # Snap the requested cell size back to the integer grid (full dimensions).
        self.Lx = self.nx_full * self.dx
        self.Ly = self.ny_full * self.dx
        self.Lz = self.nz_full * self.dx

        self.shape = (self.nx, self.ny, self.nz)
        self.shape_full = (self.nx_full, self.ny_full, self.nz_full)
        self.total_cells = self.nx * self.ny * self.nz
        self.total_cells_full = self.nx_full * self.ny_full * self.nz_full

        self._require_boundaries_leave_invariant_axes_alone()
        self._require_one_cell_axes_are_periodic()

        self._resolve_bloch()
        self._create_coordinates()

    def _resolve_cylindrical(self):  # MEEP Dcyl: x -> r, y -> phi (invariant), z -> z.
        """Resolve the cylindrical mode's axis table and validate its constraints.

        The mapping is the plan's (the design notes (fdtd-cylindrical-plan) §1): the
        Cartesian Yee-shift table lands exactly on MEEP's Dcyl shifts under
        x -> r, y -> phi, z -> z, with phi the one-cell invariant axis — so the
        invariant table here is ``(False, True, False)``, which no ``dimensions``
        value spells (2-D makes Z invariant). ``dimensions`` is reported as 2 to
        match MEEP's own normalization (``CYLINDRICAL`` becomes ``dimensions=2`` +
        ``is_cylindrical``, python/simulation.py).

        Constraints, each MEEP's own:
        * ``m`` must be an integer — the field's exp(i*m*phi) must be single-valued.
        * The phi extent must be written as 0 (an extent would suggest a phi grid;
          there is none — m carries the dependence analytically).
        * Mirrors are refused: MEEP's cylindrical symmetry is
          ``r_to_minus_r_symmetry``, installed by the mode itself, not a Cartesian
          fold (and ``exp(i*m*pi)`` is not an involution phase).
        * Bloch is Z-only in MEEP's Dcyl; a k on r or phi is refused.
        """
        if int(self.m) != self.m:
            raise ValueError(
                f"m must be an integer (the azimuthal dependence exp(i*m*phi) must be "
                f"single-valued around the axis), got {self.m!r}"
            )
        self.m = int(self.m)
        if self.dimensions not in (2, 3):
            raise ValueError(
                f"cylindrical mode sets its own axis table; leave dimensions at its "
                f"default (got dimensions={self.dimensions!r})"
            )
        self.dimensions = 2  # MEEP's own normalization of CYLINDRICAL.
        if self.accurate_fields_near_cylorigin and abs(self.m) >= 2:
            # REFUSED rather than warned. Without the default's zero rows the near-axis
            # update is unstable unless the Courant factor is at or below ~1/(|m| + 0.5)
            # — MEEP's own comment in the branch this selects — and a cylindrical
            # instability is a smooth growing mode, not a crash: the run completes and
            # returns a field-shaped array. MEEP's own test sets Courant = 1/(|m| + 0.6)
            # by hand for exactly this reason.
            limit = 1.0 / (abs(self.m) + 0.5)
            if self.courant > limit:
                raise ValueError(
                    f"accurate_fields_near_cylorigin with |m|={abs(self.m)} needs "
                    f"courant <= 1/(|m| + 0.5) = {limit:g}, got {self.courant:g}. Without "
                    f"the default treatment's zero rows near the axis the near-axis update "
                    f"is unstable above that bound, and the instability is a smooth growing "
                    f"mode rather than a crash."
                )
        if float(self.cell_size[1]) != 0.0:
            raise ValueError(
                f"a cylindrical cell has no phi extent — the exp(i*m*phi) dependence is "
                f"analytic, not gridded; write cell_size=(r_size, 0, z_size), got "
                f"{self.cell_size!r}"
            )
        if self.symmetry:
            raise ValueError(
                "mirror symmetry is not available in cylindrical mode: the mode's own "
                "r_to_minus_r ghost rule stands in for the below-axis half, and a "
                "Cartesian fold of r or z is not implemented."
            )
        if any(float(self.k_point[axis]) != 0.0 for axis in (0, 1)):
            raise ValueError(
                f"cylindrical Bloch is Z-only in MEEP (boundaries.cpp); k_point="
                f"{self.k_point!r} carries a component on r or phi."
            )
        self.invariant_axes = (False, True, False)
        self.has_invariant = True
        for axis in (0, 2):
            if float(self.cell_size[axis]) <= 0.0:
                raise ValueError(
                    f"cylindrical cell_size must have positive r and z extents, got "
                    f"{self.cell_size!r}"
                )

    def _resolve_beta(self):  # Validate MEEP's out-of-plane 2-D wavevector.
        """Normalize ``beta`` and refuse it where MEEP's own ``fields`` refuses it.

        ``beta`` is the analytic ``exp(i*2*pi*beta*z)`` dependence of a 2-D run —
        MEEP's ``special_kz`` mode, reached from ``mp.Simulation(kz_2d=...)`` on a
        cell with ``cell_size.z == 0 and k_point.z != 0``. The third axis is carried
        as a PHASE, not as grid points: ``d/dz`` becomes the exact factor
        ``i*2*pi*beta``, which enters the curl as an ``i*beta*zhat x`` cross product
        coupling the TE and TM polarizations (``step_db.cpp:148-176``). Nothing wraps,
        so this is not Bloch and it does not belong in ``k_point``; MEEP passes it in
        its own ``fields`` constructor slot and hands ``use_bloch`` only
        ``Vector3(kx, ky)`` (``python/simulation.py:2478-2504``).

        The one restriction is MEEP's, verbatim: ``fields::is_aniso2d`` aborts with
        "Nonzero beta unsupported in dimensions other than 2" for any grid that is not
        ``D2`` (``src/fields.cpp:546-547``). A cylindrical grid is one of those.
        """
        beta = float(self.beta)
        self.beta = beta
        if beta == 0.0:
            return
        if self.cylindrical or self.dimensions != 2:
            raise ValueError(
                f"beta={beta!r} is MEEP's out-of-plane 2-D wavevector and is legal only on a "
                f"2-D Cartesian grid; this grid is "
                f"{'cylindrical' if self.cylindrical else f'dimensions={self.dimensions}'}. "
                f"MEEP aborts on the same combination (fields.cpp:546-547, "
                f"\"Nonzero beta unsupported in dimensions other than 2\")."
            )

    def _resolve_bfast(self):  # Normalize MEEP's broadband fixed-angle wavevector.
        """Normalize ``bfast_scaled_k`` and refuse the one geometry MEEP cannot express.

        BFAST adds ``+d/dt (k x E)`` to dB/dt and ``-d/dt (k x H)`` to dD/dt. MEEP
        writes it as a second additive pass over the SAME shifted operands the curl
        already gathers (step_db.cpp:129-142 -> step_generic.cpp:335-471), so it costs
        no new boundary rule, no new PML slice and no source change.

        CYLINDRICAL IS REFUSED, and not for want of effort. MEEP's Dcyl branch nulls
        ``f_p`` for the R component and ``f_m`` for Z by hand (step_db.cpp:87/:96)
        while leaving ``have_p``/``have_m`` TRUE, so the single-operand branch of
        ``step_bfast`` becomes reachable with a nonzero k — and that branch
        (step_generic.cpp:376) is written ``F[i] = k1 * (g1[i+s1] + g1[i])`` with the
        ``- F[i]`` its seven single-operand siblings (:360, :400, :422, :449, :469,
        :499, :525) all carry MISSING. On a Cartesian grid the
        omission is dead: ``g2`` is NULL only when ``have_m`` is false, and the guard
        at step_db.cpp:131 then forces ``k1 = 0``, so ``F`` is identically zero there.
        Under Dcyl it would be live, and it would turn the Tustin derivative into a
        plain two-point sum. Refusing is the honest answer to an upstream
        inconsistency nobody has ever run, rather than transcribing it.

        NOT refused, deliberately: a nonzero ``k_point`` alongside BFAST (MEEP takes
        both — they are independent constructor slots, ``bfast_scaled_k`` to
        ``fields`` and ``k_point`` to ``use_bloch``), and any Courant factor. MEEP
        neither derives nor clamps the Courant for BFAST: its own test computes
        ``(1 - k_x)/sqrt(3)`` in USER code (test_refl_angular.py:52) and its notebook
        hard-codes 0.1. Inventing a gate MEEP does not have would refuse valid runs.
        """
        bfast = tuple(float(value) for value in self.bfast_scaled_k)
        if len(bfast) != 3:
            raise ValueError(
                f"bfast_scaled_k must be (kx, ky, kz), got {self.bfast_scaled_k!r}"
            )
        self.bfast_scaled_k = bfast
        if not any(bfast):
            return
        if self.cylindrical:
            raise ValueError(
                f"bfast_scaled_k={bfast} is not representable on a cylindrical grid: MEEP's "
                f"Dcyl branch nulls f_p (R) and f_m (Z) by hand at step_db.cpp:87/:96 while "
                f"have_p/have_m stay true, which makes step_generic.cpp:376 reachable with a "
                f"nonzero k — and that branch omits the '- F[i]' its seven siblings carry, so "
                f"the IIR derivative degenerates into a bare two-point sum. Run the "
                f"cylindrical case at a single frequency with a k_point instead."
            )

    @property
    def bfast_active(self) -> bool:  # MEEP's `use_bfast`, step_db.cpp:65.
        """Whether the BFAST pass runs — ANY nonzero component, for EVERY component.

        MEEP's own test is the disjunction ``bfast_scaled_k[0] || [1] || [2]``
        (step_db.cpp:65), evaluated once per chunk and applied to all six field
        components; a k along x alone still allocates ``f_bfast`` for all of them and
        still runs the pass, most of whose per-component factors are then zero.
        """
        return any(self.bfast_scaled_k)

    def _resolve_dimensions(self):  # Validate `dimensions` and the extents it makes meaningless.
        """Turn ``dimensions`` into the per-axis invariance table, and check the cell against it.

        MEEP's own two spellings are kept apart on purpose. ``dimensions`` is what
        ``Simulation._create_grid_volume`` switches on, and the axes it drops are
        fixed by ``start_at_direction``/``stop_at_direction`` (see
        :data:`INVARIANT_AXES_BY_DIMENSIONS`). MEEP additionally INFERS 2-D from a
        zero ``cell_size.z`` (``Simulation._infer_dimensions``), which is the idiom
        most 2-D scripts are written in — but that inference belongs to the
        converter, which is the only place that has an ``mp.Simulation`` to read it
        from. Here ``dimensions`` is the whole statement, and a zero extent is
        REQUIRED on each axis it makes invariant rather than being a second way to
        say the same thing.

        Requiring exactly zero is the loud half of the contract. MEEP silently
        ignores ``cell_size.z`` for ``dimensions=2`` — a run declared 2-D on a
        ``(2, 3, 4)`` cell steps identically to the same run on ``(2, 3, 0)``,
        measured 0.0e+00 apart in MEEP itself — so accepting the extent here would
        let a caller write a 4-unit-thick cell and be handed a single cell of
        infinite extent with nothing saying so. The converter reproduces MEEP's
        leniency by passing 0 explicitly; this class refuses it.
        """
        try:
            dimensions = int(self.dimensions)
        except (TypeError, ValueError):
            raise ValueError(
                f"dimensions must be one of {list(SUPPORTED_DIMENSIONS)}, got "
                f"{self.dimensions!r}"
            ) from None
        if dimensions != self.dimensions or dimensions not in INVARIANT_AXES_BY_DIMENSIONS:
            raise ValueError(
                f"dimensions must be one of {list(SUPPORTED_DIMENSIONS)}, got "
                f"{self.dimensions!r}. MEEP's cylindrical mode (mp.CYLINDRICAL) is a "
                f"different discretization — r-phi-z curls, a 1/r metric and an "
                f"m-dependent axis boundary — not a fourth value of this argument."
            )
        self.dimensions = dimensions
        invariant = INVARIANT_AXES_BY_DIMENSIONS[dimensions]
        self.invariant_axes = tuple(axis in invariant for axis in range(3))
        self.has_invariant = bool(invariant)
        for axis in range(3):
            length = self.cell_size[axis]
            name = AXIS_NAMES[axis]
            if self.is_invariant(axis):
                if float(length) != 0.0:
                    raise ValueError(
                        f"dimensions={dimensions} makes the {name} axis translationally "
                        f"invariant, but cell_size gives it an extent of {length!r}. An "
                        f"invariant axis is INFINITE and uniform, so a length there describes "
                        f"nothing this run can represent: MEEP never reads it either "
                        f"(vol2d takes only x and y, vol1d only z). Pass 0 on the {name} "
                        f"axis, or run the cell you wrote at dimensions=3."
                    )
            elif not (length > 0):
                raise ValueError(
                    f"cell_size components must be positive on every axis this run resolves, "
                    f"got {self.cell_size!r} at dimensions={dimensions}"
                    + (
                        f" (only {', '.join(AXIS_NAMES[a] for a in invariant)} may be zero)"
                        if invariant else ""
                    )
                )

    def _require_symmetry_is_on_a_real_axis(self):  # A mirror plane needs an axis with extent.
        """Refuse a mirror plane normal to an invariant axis.

        The even-cell-count rule below would already reject it — an invariant axis
        holds exactly one cell — but on the wrong grounds, and the message would send
        a reader off to change a resolution that is not the problem. A fold halves an
        axis by reflecting one half onto the other; an axis that is already infinite
        and uniform has no two halves to exchange, and MEEP itself steps such a run
        while printing ``WARNING vol mismatch`` and doubling its own loop volume.
        """
        for plane in self.symmetry:
            axis = plane.axis_index
            if not self.is_invariant(axis):
                continue
            raise ValueError(
                f"{plane!r} folds the {AXIS_NAMES[axis]} axis, which dimensions="
                f"{self.dimensions} makes translationally invariant. A mirror plane "
                f"reflects one half of an axis onto the other, and an invariant axis is "
                f"uniform and infinite — it has no halves. Fold one of the axes this run "
                f"actually resolves "
                f"({', '.join(AXIS_NAMES[a].upper() for a in range(3) if not self.is_invariant(a))}), "
                f"or drop the plane."
            )

    def _require_boundaries_leave_invariant_axes_alone(self):  # No PEC wall on an infinite axis.
        """Refuse a metallic wall on an invariant axis, and say what it would silently do.

        An invariant axis has no outer face to put a condition on: MEEP does not loop
        over it, so it has no entry in ``fields::boundaries`` at all. This engine gives
        it one cell and a periodic wrap, which is MEEP's own lower-dimensional
        emulation (fields.cpp ``nosize_direction``) and the reason the difference along
        it is exactly zero.

        Declaring it metallic instead is not a boundary choice, it is a polarization
        filter. ``zero_metal`` clears exactly the samples whose Yee shift on the walled
        axis is 0 (boundaries.cpp ``on_metal_boundary``), which on the single stored
        cell of an invariant z is Ex, Ey, Hz — the whole TE polarization. A 2-D TE run
        walled that way returns exactly zero everywhere while the TM run beside it
        stays exact, so nothing about the result reads as a boundary mistake.
        """
        if not self.has_invariant:
            return
        for axis in range(3):
            if not self.is_invariant(axis) or not self.metallic_axes[axis]:
                continue
            raise ValueError(
                f"the {AXIS_NAMES[axis]} axis is declared {METALLIC!r} and dimensions="
                f"{self.dimensions} makes it translationally invariant. An invariant axis "
                f"has no outer face — MEEP does not loop over it, so it carries no "
                f"boundary condition at all — and a perfect electric conductor there is "
                f"not a wall but a polarization filter: it zeroes every component whose "
                f"Yee shift on this axis is 0"
                + (
                    " (Ex, Ey and Hz — the entire TE polarization of a 2-D run)"
                    if axis == 2 else ""
                )
                + f" and returns a smooth exact zero for them. Leave the {AXIS_NAMES[axis]} "
                f"axis {PERIODIC!r}, which is what a one-pixel invariant direction is in "
                f"MEEP too (fields.cpp nosize_direction)."
            )

    def _require_one_cell_axes_are_periodic(self):  # MEEP: "unit directions are periodic by default".
        """Refuse a metallic wall on an axis that holds exactly one cell.

        MEEP does not have this configuration. ``fields::fields`` overrides it, with its
        own comment (fields.cpp:80-85)::

            // unit directions are periodic by default:
            FOR_DIRECTIONS(d) {
              if (gv.has_boundary(High, d) && gv.has_boundary(Low, d) && d != R &&
                  s->user_volume.num_direction(d) == 1)
                use_bloch(d, 0.0);
            }

        — so a one-cell direction comes out Periodic whatever ``k_point`` says, which is
        also what makes ``nosize_direction`` fire on it. Verified against CPU MEEP on a
        0.1 x 0.1 x 4 cell at resolution 10 with an Ex point source and no ``k_point``:
        the engine reproduces MEEP at **1.805e-07** with those axes declared periodic
        and at **1.000e+00** with them declared metallic, because a PEC wall zeroes
        every component whose Yee shift on the walled axis is 0 and a one-cell axis has
        nothing else — the run returns an exact, smooth, complete zero.

        Refused rather than silently rewritten, even though MEEP rewrites it, because a
        caller who WROTE ``boundaries="metallic"`` on such a cell has a wrong mental
        model of what they asked for and should be told. The converter never produces
        it: :func:`~.from_meep._lift_boundaries` transcribes the same rule.
        """
        counts = (self.nx_full, self.ny_full, self.nz_full)
        for axis in range(3):
            if not self.metallic_axes[axis] or counts[axis] != 1 or self.is_invariant(axis):
                continue
            raise ValueError(
                f"the {AXIS_NAMES[axis]} axis is declared {METALLIC!r} and holds exactly one "
                f"cell. MEEP has no such run: fields::fields overrides a unit direction to "
                f"Periodic whatever the k_point says (fields.cpp, \"unit directions are "
                f"periodic by default\"). A perfect electric conductor on both faces of a "
                f"single cell zeroes every component whose Yee shift on this axis is 0 and "
                f"leaves nothing between them, so the run returns an exact zero for that whole "
                f"family — measured 1.000e+00 complex relative L2 against CPU MEEP, against "
                f"1.805e-07 for the same cell with this axis {PERIODIC!r}. Declare it "
                f"{PERIODIC!r}, or give it more than one cell."
            )

    def _resolve_bloch(self):  # Validate k_point and precompute the per-axis wrap phases.
        """Turn ``k_point`` into the per-axis wrap factor, MEEP's ``eikna``.

        MEEP boundaries.cpp ``fields::use_bloch``::

            eikna[d] = exp(I * kk * ((2 * pi / a) * gv.num_direction(d)))

        with ``a`` the resolution and ``num_direction(d)`` the full cell count, so
        the exponent is ``i * 2*pi * k_d * L_d``: MEEP's k is in units of
        2*pi/distance, and dropping the 2*pi would detune the boundary by a factor
        of 6.28 while still producing a smooth, plausible-looking field.

        MEEP pins the Brillouin-zone edge exactly rather than through ``exp``::

            if (real(kk) * gv.num_direction(d) == 0.5 * a) eikna[d] = -exp(-imag(kk) * ...)

        which for a real k is exactly -1. The same comparison is reproduced here on
        the same two quantities, so k*L = 1/2 lands on -1 + 0j on both sides
        instead of on MEEP's -1 and this port's -1 + 1.2e-16j. Only +1/2 is special-
        cased, matching MEEP: k*L = -1/2 goes through ``exp`` on both sides.

        A zero component yields ``None``, not ``1 + 0j``. The stepping and readback
        paths skip the multiply entirely on a ``None`` axis, which is what makes
        k = 0 bit-identical to the pre-Bloch engine rather than merely equal to
        within a rounding of ``f * (1 + 0j)``.

        A mirror plane forces the field to be even or odd about it, which is only
        consistent with k_d = 0; that combination is refused here rather than
        silently producing a band structure for a system nobody asked for.
        """
        values = tuple(self.k_point)
        if len(values) != 3:
            raise ValueError(f"k_point must be (kx, ky, kz), got {self.k_point!r}")
        components = []
        for axis_name, entry in zip(AXIS_NAMES, values):
            if isinstance(entry, complex):
                # MEEP's use_bloch takes a complex kk (an evanescent Bloch state, whose
                # imaginary part grows or decays the field across the cell). Nothing
                # here implements the |eikna| != 1 case, so it is named rather than
                # dropped through float(), which would only say "can't convert complex".
                raise ValueError(
                    f"k_point component {axis_name} must be real, got {entry!r}: complex Bloch "
                    f"wavevectors (evanescent states, MEEP's complex use_bloch) are not implemented"
                )
            number = float(entry)
            if not math.isfinite(number):
                raise ValueError(
                    f"k_point component {axis_name} must be finite, got {entry!r}"
                )
            components.append(number)
        self.k_point = (components[0], components[1], components[2])

        for axis, (axis_name, k_value) in enumerate(zip(AXIS_NAMES, self.k_point)):
            if k_value != 0.0 and self.is_mirrored(axis):
                raise ValueError(
                    f"mirror symmetry on the {axis_name.upper()} axis requires k_point "
                    f"component {axis_name} = 0, got {k_value!r}: a mirror plane forces the "
                    f"field to be even or odd about it, which only a zero Bloch phase allows"
                )
            if k_value != 0.0 and self.is_invariant(axis):
                raise ValueError(
                    f"k_point component {axis_name} = {k_value!r} on an axis that dimensions="
                    f"{self.dimensions} makes translationally invariant. A Bloch phase says "
                    f"how the field changes one lattice vector along an axis; an invariant "
                    f"axis is one along which it does not change at all, by construction. "
                    f"MEEP refuses the same pairing by construction: a nonzero k_point.z with "
                    f"a zero cell_size.z makes Simulation._infer_dimensions build a 3-D run "
                    f"instead of a 2-D one, so the phase and the reduction are never both in "
                    f"effect. Drop the {axis_name} component, or run at dimensions=3 with a "
                    f"real extent on {axis_name}."
                )
            if k_value != 0.0 and self.is_metallic(axis):
                raise ValueError(
                    f"k_point component {axis_name} = {k_value!r} on an axis declared "
                    f"{METALLIC!r}: a Bloch phase says what the field is one LATTICE VECTOR "
                    f"away, and a perfect-electric-conductor wall gives the axis no lattice "
                    f"vector at all. MEEP cannot express this either — fields::use_bloch sets "
                    f"boundaries[side][d] = Periodic on the axis it phases (boundaries.cpp), so "
                    f"a k_point and a metallic wall on the same axis are mutually exclusive "
                    f"there too. Drop the k component, or declare this axis {PERIODIC!r}."
                )

        counts = (self.nx_full, self.ny_full, self.nz_full)
        lengths = (self.Lx, self.Ly, self.Lz)
        self.bloch_phases = tuple(
            self._axis_bloch_phase(self.k_point[axis], counts[axis], lengths[axis])
            for axis in range(3)
        )
        self.has_bloch = any(phase is not None for phase in self.bloch_phases)

    def _axis_bloch_phase(self, k_value: float, n_full: int, length: float) -> Optional[complex]:
        """Wrap factor for one axis: ``exp(i*2*pi*k*L)``, or None when k is zero."""
        if k_value == 0.0:
            return None
        if k_value * n_full == 0.5 * self.resolution:  # MEEP's exact Brillouin-zone edge test.
            return complex(-1.0, 0.0)
        return cmath.exp(2j * math.pi * k_value * length)

    def bloch_phase(self, axis: int) -> Optional[complex]:
        """Factor a field picks up crossing this axis's UPPER boundary, or None at k = 0.

        The one definition of the sign convention in this engine::

            f(x + L_d) = bloch_phase(d) * f(x)

        so a stencil reading one cell up past the far face multiplies the wrapped
        value by ``bloch_phase(d)``, and one reading one cell down past the near
        face multiplies by its conjugate. That is MEEP's
        ``locate_point_in_user_volume``, which translates a point *up* into the cell
        by a lattice vector and multiplies by ``conj(eikna[d])`` to recover the
        field at the original, lower point (boundaries.cpp).

        Args:
            axis: 0 for X, 1 for Y, 2 for Z.

        Returns:
            The complex wrap factor, or None when this axis has no Bloch phase.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return self.bloch_phases[axis]

    def origin_doubled(self, axis: int) -> int:
        """MEEP's ``little_corner`` for one axis, in doubled integer coordinates.

        The one definition of where an axis starts. Cell ``i`` of a component with
        Yee shift ``s`` sits at doubled coordinate ``origin + 2*i + s``, i.e. at
        ``origin * dx/2 + (i + s/2) * dx``.

        MEEP fixes it in ``Simulation._create_grid_volume`` -> ``grid_volume::
        center_origin()`` -> ``shift_origin(-icenter())`` with (vec.cpp)::

            icenter() = io + ivec(nx, ny, nz).round_down_to_even()

        so a centred axis ends up at ``io = -(n - n % 2)``. **The parity matters.**
        An axis with an EVEN cell count starts at ``-n``, which is ``-L/2``; an axis
        with an ODD one starts at ``-(n - 1)``, half a cell higher, because MEEP
        requires ``icenter - io`` to be even in every component so that the symmetry
        transforms preserve the Yee lattice. Taking ``-n`` for both — which this
        engine did — puts every odd-count axis half a cell away from MEEP's, and
        nothing about the result says so: the field is smooth, the run completes, and
        only a measurement that resolves the axis disagrees. Measured on a
        2.0 x 1.5 x 2.0 cell at resolution 10 (ny = 15, odd; nx = nz = 20, even) with
        an off-axis point source and no PML, against CPU MEEP: transmitted flux
        4.8e-02 wrong through a small plane and 3.4e-02 through the full cross
        section, where the same run on a 2.0 x 1.6 x 2.0 cell (ny = 16) is exact at
        7.8e-08 / 1.3e-07.

        A mirror-folded axis is ``-2``, one full cell below the plane, from MEEP's
        ``grid_volume::halve()`` (``set_origin(d, icenter - 2)``); its cell 0
        straddles the mirror plane. That is true of a folded Z exactly as it is of
        a folded X: ``halve()`` takes a direction, and nothing in it distinguishes
        the axis a run happens to propagate along.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        if self.is_mirrored(axis):
            return -2
        if self.cylindrical and axis == 0:
            # MEEP Dcyl: r starts at exactly 0 — volcyl's io = (0, 0) and
            # center_origin shifts z only (icenter's r-component is 0,
            # vec.cpp:722-730); an r < 0 origin aborts in MEEP outright.
            return 0
        n_full = (self.nx_full, self.ny_full, self.nz_full)[axis]
        return -(n_full - n_full % 2)  # MEEP icenter(): round_down_to_even.

    def axis_origin(self, axis: int) -> float:  # Physical coordinate the stored axis starts from.
        """Lower edge of cell 0 on one axis: :meth:`origin_doubled` in length units."""
        return self.origin_doubled(axis) * self.dx / 2

    def is_mirrored(self, axis: int) -> bool:  # Does a mirror plane fold this axis?
        """Whether this axis is halved by a mirror plane — the one definition of "folded"."""
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return self.mirrors_by_axis[axis] is not None

    def is_axis(self, axis: int) -> bool:  # Is this the cylindrical radial axis at r = 0?
        """Whether this axis's low side is the cylindrical AXIS — the one definition.

        True only for the r axis of a cylindrical grid. Consumers: the shift
        helpers (below-axis ghosts are the ``r_to_minus_r`` image, the high face a
        metallic zero), the ownership masks (the axis row of a shift-0 component
        is written by the per-m rules, not the curl — MEEP's
        ``little_owned_corner0`` deliberately excludes r = 0, vec.hpp:1100-1104),
        and ``zero_metal`` (which must NOT hold the axis row at zero the way it
        holds a wall).
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return bool(self.cylindrical) and axis == 0

    def mirror_phase(self, axis: int) -> Optional[int]:
        """Declared phase of this axis's mirror plane, or None when it carries none.

        MEEP's ``symmetry::ph`` for the plane, straight from
        ``mp.Mirror(direction, phase)``. +1 is an even mirror and -1 an odd one;
        every component's own parity is that factor times the parity the
        component's Yee direction gives it (``fields.mirror_parity``).

        ``None`` rather than ``1`` on an unmirrored axis, following
        :meth:`bloch_phase`: "this axis has no plane" and "this axis has an even
        plane" are different statements, and a caller that treats the first as the
        second silently folds an axis nobody asked to fold.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        plane = self.mirrors_by_axis[axis]
        return None if plane is None else plane.phase

    def boundary_condition(self, axis: int, side: str) -> str:
        """The declared condition on one face — the one place the per-side table is read.

        ``side`` is ``"low"`` or ``"high"``, in ascending-coordinate order, spelled
        rather than numbered because MEEP's own ``boundary_side`` enum is High = 0 and
        Low = 1 (meep.hpp) and reading that the obvious way is already a measured trap
        elsewhere in this package.

        The mirror plane is NOT reported here. A fold is not an outer boundary
        condition in MEEP either — ``symmetry`` and ``fields::boundaries`` are separate
        mechanisms, and a folded run still declares a condition for the full cell's two
        faces. Ask :meth:`is_mirrored` about the fold.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        if side not in SIDES:
            raise ValueError(f"side must be one of {list(SIDES)}, got {side!r}")
        return self.boundaries[axis][SIDES.index(side)]

    def is_metallic(self, axis: int) -> bool:
        """Whether this axis is terminated by perfect-electric-conductor walls.

        The one definition of "PEC axis". Both faces carry the same condition by
        construction (:func:`_normalize_boundaries` refuses a mixed axis), so this
        single flag is the whole statement.

        Independent of :meth:`is_mirrored`: MEEP's default cell with a mirror plane is
        BOTH folded and metallic, and that combination is the one this engine has
        always been able to step exactly — a folded axis terminates its far face with a
        zero ghost, which is what a PEC wall is (see ``stepping._shift_up``).
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return self.metallic_axes[axis]

    def is_invariant(self, axis: int) -> bool:
        """Whether this axis is translationally invariant — the one definition of "reduced".

        True for z at ``dimensions=2`` and for x and y at ``dimensions=1``, which is
        MEEP's own assignment (:data:`INVARIANT_AXES_BY_DIMENSIONS`). Such an axis
        holds exactly ONE cell standing for an infinite uniform direction, wraps
        periodically onto itself, and therefore contributes a difference of exactly
        zero to every stencil that crosses it — which is what makes a 2-D run cheap
        and its polarizations decouple.

        Read this rather than testing ``stored_cells(axis) == 1``: a 3-D run may
        legitimately have a one-cell axis (MEEP builds one for ``cell_size.x = 0`` at
        ``dimensions=3``), and that axis DOES take a boundary condition, may be
        metallic, and may carry a Bloch phase. The two look identical in the shape
        tuple and are different simulations.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return self.invariant_axes[axis]

    def nosize_direction(self, axis: int) -> bool:
        """MEEP's ``fields::nosize_direction`` — a one-pixel periodic axis stands for the whole direction.

        Transcribed from fields.cpp::

            return (gv.has_boundary(Low, d) && gv.has_boundary(High, d) &&
                    boundaries[Low][d] == Periodic && boundaries[High][d] == Periodic &&
                    gv.num_direction(d) == 1);

        with MEEP's own comment: "One-pixel periodic dimensions are used almost
        exclusively to emulate lower-dimensional computations, so if the user passes an
        empty size in that direction, they probably really intended to specify that
        whole dimension."

        It is what makes a zero-extent source request on such an axis a WHOLE-AXIS
        request rather than a delta function: ``add_volume_source`` multiplies the
        amplitude by ``gv.a`` once per zero-size direction and skips exactly the
        directions this reports (sources.cpp:482). Getting that wrong scales a 2-D
        point source by the resolution — a factor of 10 at ``resolution=10``, with a
        perfectly smooth field to show for it.

        Deliberately not the same predicate as :meth:`is_invariant`. This one is true
        of any one-cell periodic axis, declared invariant or not, because that is the
        condition MEEP tests; a genuinely 3-D run with a one-cell periodic axis gets
        MEEP's behaviour here for free.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return self.axis_wraps(axis) and (self.nx_full, self.ny_full, self.nz_full)[axis] == 1

    def axis_wraps(self, axis: int) -> bool:
        """Whether this axis is periodic — the one definition of "has a lattice vector".

        Exactly the axes that are neither mirror-folded nor metallic. A PML does not
        change it: MEEP's ``use_bloch`` makes every face Periodic and grades the
        absorber underneath as a material, so ``-L/2`` and ``+L/2`` stay the same
        lattice point even on an absorbing axis (``stepping.periodic_axes`` reports the
        same thing). A folded axis is the opposite: the fold is a boundary condition,
        not a translation, so the stored quadrant has no image outside itself. A
        metallic axis is the same story for a different reason — MEEP never translates
        across a face it did not mark Periodic (boundaries.cpp
        ``locate_point_in_user_volume`` tests ``boundaries[High][d] == Periodic``), and
        the field beyond a PEC wall is not an image of anything, it is zero.

        This is what separates the two out-of-range answers in
        :meth:`position_to_index` — a request past a wrapping axis is a lattice request
        that source placement folds onto the cell, and a request past a folded or
        walled axis cannot be represented at all.

        The cylindrical r axis never wraps: below r = 0 lies the field's own
        ``r_to_minus_r`` image (a boundary rule, not a translation) and past
        r_max a metallic wall.
        """
        return (not self.is_mirrored(axis) and not self.is_metallic(axis)
                and not self.is_axis(axis))

    def _create_coordinates(self):  # Build float32 cell-center coordinates for each axis.
        """Cell-center coordinates in MEEP conventions.

        Cell i is centered at ``axis_origin + (i + 0.5)*dx``. Without symmetry and
        with an even cell count that is the familiar ``-L/2 + (i + 0.5)*dx``; with an
        odd count MEEP's origin sits half a cell higher (see :meth:`origin_doubled`,
        which carries the measured cost of getting that parity wrong), and with
        symmetry (meep/vec.cpp halve()) it is ``-dx`` for both parities, so cell 0
        straddles the mirror plane and the last of the ``owned_cells``
        cells lands at ``L/2 - 0.5*dx`` for an even count and at exactly ``+L/2``
        for an odd one (the top of the odd count's half-cell-higher window).
        """
        xp = self.xp
        half_dx = self.dx / 2
        counts = (self.nx, self.ny, self.nz)
        axes = []
        for axis in range(3):
            start = self.axis_origin(axis) + half_dx
            axes.append(xp.linspace(
                start, start + (counts[axis] - 1) * self.dx, counts[axis], dtype=xp.float32
            ))
        self.x, self.y, self.z = axes

    def axis_index(self, axis: str) -> int:  # 'x'/'y'/'z' -> 0/1/2, refusing anything else.
        """Resolve an axis NAME to its index — the one place the spelling is checked.

        Replaces an ``axis_extent`` that also returned the axis length and a folded
        flag, neither of which anything read. A tuple element nobody consumes cannot
        be wrong in any way a test would notice, so it drifts: that flag was the
        one place in this class still able to disagree with :meth:`is_mirrored`.
        """
        if axis not in AXIS_NAMES:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        return AXIS_NAMES.index(axis)

    def stored_cells(self, axis: int) -> int:  # Cells actually allocated on one axis.
        """Stored count for one axis — halved where a mirror plane folds it.

        On a folded PERIODIC axis this is ONE MORE than :meth:`owned_cells`, at
        both count parities: the extra cell's shift-0 sample is MEEP's
        ``big_corner`` (owned and stepped, MEEP's own ``num + 1`` allocation,
        vec.cpp:293-296 with vec.cpp:445-462 ``owns``), and its shift-1 slot is
        the far ghost the fill pass supplies. ``big_corner`` is the second mirror
        at doubled ``n_full`` at an even count and one half-cell above it — the
        odd window's top — at an odd one.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return (self.nx, self.ny, self.nz)[axis]

    def owned_cells(self, axis: int) -> int:  # MEEP's halved cell count for one axis.
        """MEEP's own cell count for this axis — ``halve()``'s num where folded.

        ``n_full - n_full//2 + 1`` on a mirrored axis (vec.cpp:1069-1074: ``num =
        1 + (big_corner - icenter)/2``, origin ``icenter - 2``), the full count
        everywhere else. This is the count MEEP's ownership arithmetic is built
        on — ``big_corner = origin + 2*num`` (:meth:`big_corner_doubled`), the
        PML wall position, the source-deposition owned corners — and it differs
        from :meth:`stored_cells` in exactly one case: a folded PERIODIC axis
        stores one cell beyond it (the ``big_corner`` plane plus its ghost slot;
        see ``__init__``), at both count parities. Consumers that transcribe MEEP's
        corner arithmetic must read THIS count; reading the stored count instead
        puts every corner one cell too high on such an axis, which is invisible
        until a source, a monitor or an absorber lands on the window top.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        n_full = (self.nx_full, self.ny_full, self.nz_full)[axis]
        if self.is_mirrored(axis):
            return n_full - n_full // 2 + 1
        return (self.nx, self.ny, self.nz)[axis]

    def big_corner_doubled(self, axis: int) -> int:  # MEEP's big_corner for one axis.
        """MEEP's ``big_corner`` in doubled coordinates — the window top of this axis.

        ``origin_doubled + 2 * owned_cells`` (vec.hpp ``big_corner()``: io + 2*num),
        which is doubled ``n_full + n_full % 2`` for a centred or folded axis and
        ``2 * n_full`` for the cylindrical r axis. On a folded periodic axis the
        stored array runs one cell PAST this corner (this corner's own shift-0
        plane and the shift-1 ghost slot above it), so ``origin + 2 *
        stored_cells`` is NOT this number there — that is the whole reason this
        method exists.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return self.origin_doubled(axis) + 2 * self.owned_cells(axis)

    def position_to_index(self, pos: float, axis: str, iyee_shift: int = 1) -> tuple[int, float]:
        """Convert a position to a lower grid index plus its linear-interpolation weight.

        MEEP references: vec.hpp interpolate_linear() / IVEC_LOOP_WEIGHT,
        sources.cpp src_vol_chunkloop.

        An axis's stored cells start at the physical origin ``io * dx/2``, where
        ``io`` is MEEP's little-corner in doubled coordinates: ``-n_full`` for a
        centered axis (origin -L/2) and ``-2`` after halve() for a mirror axis
        (origin -dx). Cell i of a component with Yee shift s then sits at
        ``origin + (i + s/2)*dx``, which inverts to ``i = (pos - origin)/dx - s/2``:

            centered, iyee_shift=1: component at -L/2 + (i+0.5)*dx
            centered, iyee_shift=0: component at -L/2 + i*dx
            mirror,   iyee_shift=1: component at (i-0.5)*dx — cell 0 straddles the plane
            mirror,   iyee_shift=0: component at (i-1)*dx

        THE CONTRACT — this method interpolates BETWEEN stored samples and does
        nothing else. Let ``f = (pos - origin)/dx - iyee_shift/2`` be the fractional
        index of ``pos`` on this axis's component lattice:

        *In range* — ``0 <= f <= n-1``, i.e. ``pos`` lies between the first and last
        stored samples. Returns ``(idx_low, weight_high)`` with ``idx_low`` in
        ``[0, n-2]`` and ``weight_high`` in ``[0, 1]``, and ``idx_low + weight_high``
        reproduces ``f`` (the top sample comes back as ``(n-2, 1.0)``, not ``(n-1,
        0.0)``, so the pair always names a real neighbour). A single-cell axis returns
        ``(0, 0.0)``.

        *Wraps* — nothing here. A position past the ends of a WRAPPING axis
        (:meth:`axis_wraps`) is a lattice request, and folding it needs the Bloch
        factor of :meth:`bloch_phase`, which an ``(int, float)`` pair cannot carry.
        Source placement does the fold, in MEEP's own machinery
        (``sources._build_source_points``, a translation of ``loop_in_chunks``): it
        walks the request's ivec ladder, folds each rung onto the cell modulo the
        lattice, and multiplies by ``conj(bloch_phase ** shift)``.

        *Raises* — everything else. Both out-of-range cases raise here rather than
        being served approximately, and the message says which one it is.

        WHAT THIS REPLACED, and why it is worth a paragraph. The previous contract
        tolerated one cell of overhang and returned an extrapolating weight outside
        [0, 1] against a clamped index. Nothing downstream wanted extrapolation: the
        CW point source dropped the negative half of the stencil and deposited ~1.5x
        the requested current onto one cell, and every position in the outer half-cell
        of a wrapping axis — a source asked for at a cell face, an entirely ordinary
        request — was placed off the end of the lattice instead of onto its image.
        Measured against CPU MEEP on a 1 x 1 x 2 cell at resolution 10, complex
        relative L2 over the whole Ez volume: 5.4e-1 at either z face (half-integer
        placement, both cell parities), and 1.2e0 at either x face with 6.2e-1 in the
        middle of the last cell (integer placement, where the whole top cell needs the
        wrap). Every one of those runs completed and returned a smooth, plausible
        field. See ``test_sources.py`` for the swept cross-validation, worst now 2.2e-7.

        Args:
            pos: Position in MEEP coordinates.
            axis: 'x', 'y', or 'z'.
            iyee_shift: 1 for half-integer (cell-center) placement, 0 for integer.

        Returns:
            (lower_index, weight_for_upper)
        """
        axis_index = self.axis_index(axis)
        n = self.stored_cells(axis_index)
        if iyee_shift not in (0, 1):
            raise ValueError(f"iyee_shift must be 0 or 1, got {iyee_shift!r}")
        if not math.isfinite(pos):
            raise ValueError(f"position must be finite, got {pos!r} on the {axis} axis")
        if self.is_invariant(axis_index):
            # Every position on an invariant axis names the same sample, because the
            # field does not vary along it — so there is no range to be outside of and
            # nothing to interpolate between. MEEP says the same thing by dropping the
            # coordinate entirely: py_v3_to_vec builds a D2 vec from x and y alone, and
            # a 2-D source asked for at z = 7.3 lands where z = 0 does, measured
            # 0.0e+00 apart in MEEP itself.
            return 0, 0.0
        offset = 0.5 * iyee_shift

        # Physical origin of the stored axis — one definition, in `axis_origin`:
        # MEEP halve() sets io = -2 (one full cell below the mirror plane), and a
        # centred axis takes io = -(n_full - n_full % 2), which is -L/2 only when the
        # cell count is even.
        origin = self.axis_origin(axis_index)
        idx_float = (pos - origin) / self.dx - offset

        top = float(n - 1)
        tolerance = index_tolerance(n)
        if idx_float < -tolerance or idx_float > top + tolerance:
            raise ValueError(
                f"position {pos} is outside the {axis} axis: fractional index {idx_float:.6f} is "
                f"not in [0, {n - 1}] for {n} stored cells with iyee_shift={iyee_shift}. "
                + (
                    "This axis wraps, so a request past its ends names a lattice image rather "
                    "than a point off the end. Carrying it needs the Bloch factor, which this "
                    "method's (index, weight) pair cannot hold, so it refuses instead of "
                    "extrapolating: place a source through sources._build_source_points, which "
                    "folds the request onto the cell and applies conj(bloch_phase ** shift)."
                    if self.axis_wraps(axis_index)
                    else
                    "This axis is mirror-folded, so it carries no lattice vector at all: only "
                    "the stored quadrant exists, and a position outside it has no image the "
                    "engine can represent. Run the full domain."
                    if self.is_mirrored(axis_index)
                    else
                    "This axis is terminated by metallic walls, so it carries no lattice vector "
                    "at all: past a perfect electric conductor there is no field, and no image "
                    "of one either. Move the request inside the cell, or declare this axis "
                    "periodic if it is meant to repeat."
                )
            )
        idx_float = min(max(idx_float, 0.0), top)  # Absorb the tolerance; never widen the range.
        idx_low = max(0, min(int(math.floor(idx_float)), n - 2))
        weight_high = idx_float - idx_low  # Against the chosen cell, so the pair stays consistent.
        return idx_low, float(weight_high)

    def meshgrid(self):  # 3D cell-center coordinate meshgrid in (x, y, z) index order.
        return self.xp.meshgrid(self.x, self.y, self.z, indexing="ij")

    def has_symmetry(self) -> bool:  # True when any mirror plane is active.
        return bool(self.symmetry)

    @property
    def symmetry_axes(self) -> tuple[str, ...]:  # Folded axis names, in X, Y, Z order.
        return tuple(
            plane.axis for plane in self.mirrors_by_axis if plane is not None
        )

    def symmetry_reduction_factor(self) -> int:  # Memory reduction from symmetry (1, 2, 4, or 8).
        return 2 ** len(self.symmetry)

    @property
    def metallic_axis_names(self) -> tuple[str, ...]:  # Names of the PEC-terminated axes, x, y, z order.
        return tuple(AXIS_NAMES[axis] for axis in range(3) if self.metallic_axes[axis])

    @property
    def invariant_axis_names(self) -> tuple[str, ...]:  # Names of the invariant axes, x, y, z order.
        return tuple(AXIS_NAMES[axis] for axis in range(3) if self.invariant_axes[axis])

    @property
    def active_axis_names(self) -> tuple[str, ...]:  # Names of the axes this run resolves.
        return tuple(AXIS_NAMES[axis] for axis in range(3) if not self.invariant_axes[axis])

    def __repr__(self) -> str:
        planes = ", ".join(f"{plane.axis}{plane.phase:+d}" for plane in self.symmetry)
        sym_str = f", symmetry=({planes})" if planes else ""
        bloch_str = f", k_point={self.k_point}" if self.has_bloch else ""
        walls = "".join(self.metallic_axis_names)
        pec_str = f", metallic={walls}" if walls else ""
        dim_str = f", dimensions={self.dimensions}" if self.dimensions != 3 else ""
        return (
            f"Grid(resolution={self.resolution}, shape={self.shape}, "
            f"dx={self.dx:.4f}, dt={self.dt:.6f}{dim_str}{sym_str}{bloch_str}{pec_str})"
        )
