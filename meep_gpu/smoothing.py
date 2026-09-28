"""
Subpixel epsilon smoothing implements MEEP's ``eps_averaging`` for this engine's material
array.

A staircased material boundary drops FDTD from second- to first-order accurate, so
a sharp grating or photonic crystal costs percent-level error at every resolution
a user would actually run. MEEP fixes that by smoothing the INVERSE permittivity
tensor over one voxel before time-stepping: the mean of ``1/eps`` for the field
component normal to the interface and the inverse of the mean ``eps`` for the
components parallel to it. This module computes that same tensor and hands the
engine the piece its single scalar ``inv_eps`` array can carry.

The scheme (doc/docs/Subpixel_Smoothing.md, src/anisotropic_averaging.cpp
``eff_chi1inv_row``, src/meepgeom.cpp ``eff_chi1inv_matrix`` /
``fallback_chi1inv_row``), for isotropic materials::

    chi1inv = P * <1/eps> + (I - P) * (1/<eps>),      P_ij = n_i n_j

``<.>`` averages over the voxel — a cube of side ``dx`` centred on the Yee position
of the component being smoothed (``grid_volume::dV(here, smoothing_diameter=1)``) —
and ``n`` is the unit interface normal.

WHICH HALF OF THAT ACTUALLY RUNS. The anisotropy is real in this module and is checked
against MEEP value for value, but it is unreachable through
:meth:`FdtdDriver.set_epsilon_smoothed`: that method requires epsilon to be invariant
along the smoothed component's own axis ``d``, and BOTH ways this module finds a normal
are symmetric in ``d`` for such a structure. The sphere rule below is octahedrally
symmetric, so it pairs every point ``u`` with a point of equal weight at ``-u_d`` and
identical other coordinates, and the gradient along ``d`` vanishes EXACTLY (measured
2.5e-17). The plane fit takes ``n_d`` from a central difference of two bisections at
``+d`` and ``-d``, which for a ``d``-invariant epsilon return bit-identical heights, so
that difference is exactly 0.0. Either way ``n_d = 0``, and the installed diagonal is
``1/<eps>`` — the plain arithmetic mean — in every configuration the driver accepts,
with the normal having no influence on it at all. That is the right physics for that
configuration
rather than a shortcut: a structure invariant along ``d`` and driven in ``E_d`` puts E
tangential to every interface it has, where ``<eps>`` is the correct effective medium
and is what MEEP computes too. The harmonic branch is what the OTHER polarization
needs, and reaching it needs the per-component tensor this engine does not store. So
the second-order convergence this module buys is bought for the tangential
polarization; the normal-direction row entry is verified here and used by nothing yet.

THE NORMAL, AND WHICH VOXELS ARE AVERAGED AT ALL. MEEP takes ``n`` from the gradient of
epsilon integrated against a 50-point degree-11 quadrature rule on a sphere of radius
``dx`` about that same point (``material_function::normal_vector`` with
``sphere-quad.h``), and this module reproduces that rule, that radius, and MEEP's "the
first ``2**ndims`` sphere points all agree, so do not average" shortcut. Where the
gradient vanishes MEEP point-samples instead, and so does this: a homogeneous medium is
returned bit-identical to the unsmoothed path rather than merely close. That sphere
gradient still decides WHICH voxels are averaged; what it no longer decides, under the
default fill rule, is the answer.

THE FILL FRACTION: TWO RULES, AND WHY THE ANALYTIC ONE IS THE DEFAULT. MEEP has two
paths of its own. For a geometric object ``meepgeom.cpp eff_chi1inv_matrix`` takes the
EXACT overlap of the voxel with the object and the EXACT object normal; only for a user
material function does it fall back to summing samples. This module is only ever given a
material function, so it used to copy the fallback — and inherited its error, which is
first order in the sample count and was the dominant error of the whole feature: on a
sharp grating against CPU MEEP, 3.4e-2 at the old default of 16 samples per axis and
still 2.1e-3 at 256.

``fill_rule="planar"`` (the default for a callable epsilon) recovers the geometric path
instead of the fallback. For an interface that is locally a PLANE inside the voxel — a
grating, a slab, a photonic crystal, every flat face of anything — both the fill and the
normal are closed forms once the plane is known, and the plane can be MEASURED out of the
material function rather than sampled: five columns through the voxel are bisected to
find where epsilon jumps, four central differences give the slopes and the normal, and
:func:`cube_plane_fill` integrates the cube exactly. Bisection converges geometrically,
so the interface is located to 5e-16 of a cell instead of ``1/(2s)``. Measured on the
same grating: 1.5e-6 at every supersample from 8 to 256 — three orders below the old
default, flat, and at the engine's own reproduction floor for that case (its agreement
with MEEP's UNSMOOTHED run is 1.9e-6). Per voxel against MEEP's own analytic
``chi1inv``: 2.98e-8, which is 2**-25, the float32 MEEP stores that array in.

Nothing is believed without a check, because a wrong fill is a smooth, plausible,
entirely finite wrong material. A voxel keeps the sampled answer unless all five columns
bracket the same two materials, the fine lattice agrees those are the only two present,
the fifth column confirms the surface is planar to
:data:`PLANE_RESIDUAL_TOLERANCE`, and the analytic fill lands within the midpoint sums'
own resolution of the sampled fill. ``SmoothedEpsilon.planar_fraction`` reports how much
of the interface passed; a bar's corner is two planes and does not.

Two limits of the sampled fallback remain, both measured in ``test_smoothing.py``, and
the first of them is now confined to it. The sphere-quadrature normal of a DISCONTINUOUS
epsilon carries a rule artifact that does not shrink with resolution: a plane tilted 2:1
reads back 0.0395 off in direction (worst voxel, bit-identical at resolutions
10 / 20 / 40 / 80), and a sphere up to 0.22 off. The fitted plane reads the same tilt to
4e-15. Second, and NOT fixed here, the 8-point prefix shortcut can declare a voxel
uniform while it straddles a CORNER — all eight samples land in the surrounding
material — and then a voxel holding a real fraction of the other material comes back
unsmoothed. Measured: one voxel of a rectangular bar, 17 % filled, 53 % away from MEEP's
analytic value. That case is silent in MEEP; here it is counted as
``SmoothedEpsilon.unresolved_fraction``.

WHAT THE ENGINE CAN CONSUME, and the residual that leaves. MEEP stores ``chi1inv``
per E component as a full symmetric tensor; this engine stores ONE scalar volume
shared by all three D components (``FdtdDriver.set_epsilon``). :func:`smooth_inverse_epsilon`
therefore takes the component whose row is wanted and returns that row's DIAGONAL
entry, ``n_d**2 * <1/eps> + (1 - n_d**2) / <eps>``, at that component's own Yee
positions; the off-diagonals and the other two rows come back on the result object
(:class:`SmoothedEpsilon`) for a future per-component consumer but are not installed.
That reduction is EXACT — not approximate — when epsilon is invariant along the
chosen component's own axis and the run carries only the polarization containing
that component, because then the other two D components are identically zero and
the off-diagonal entries of this row vanish with them (``n_d = 0`` is not required;
``n`` simply has no component along the invariant axis). That covers the case this
feature exists for: a grating or 1-D/2-D photonic crystal in the polarization whose
E field lies along the invariant axis. :meth:`FdtdDriver.set_epsilon_smoothed`
checks the invariance and refuses anything else rather than shipping a smooth,
plausible, few-percent-wrong field.

Input comes either as a vectorized callable ``epsilon(x, y, z)`` evaluated wherever
the scheme needs it — which is what MEEP does with a ``material_function`` whose
``do_averaging`` is set — or as an array already supersampled onto the fine lattice
:func:`fine_coordinates` describes, for callers that build epsilon some other way.
Only the callable can carry the analytic fill: an array is piecewise constant on the
fine lattice, so bisecting it would land on a fine-cell edge and hand back the very
quantization the analytic fill removes. ``fill_rule="auto"`` therefore resolves to
``"planar"`` for a callable and ``"sampled"`` for an array, every result says which
one ran, and asking for ``"planar"`` with an array raises rather than downgrading
quietly.

Every coordinate handed to the callable is first wrapped into the periodic cell
``[axis_origin, axis_origin + L)``: this engine's boundaries are always periodic (a
PML is a material inside the cell), the voxel of a boundary cell straddles the
wrap, the sphere the normal is read from reaches one full cell past it, and the
bisection columns reach up to 2.3 cells past it — MEEP does the same through
``ensure_periodicity``. ``supersample`` still sets how finely the fine lattice sees
the structure: it decides whether a voxel meets an interface at all (a feature
thinner than ``dx/s`` is invisible), whether the voxel holds two materials or more,
and the accuracy of any voxel the plane fit declines. It accepts a per-axis triple
so an axis the structure does not vary along costs one sample rather than ``s``.

Not smoothed, deliberately, because MEEP does not smooth them either: susceptibility
sigma (doc/docs/Materials.md:60 — ``anisotropic_averaging.cpp:333-345`` point-samples
sigma while :252-257 averages chi1inv), the D conductivity, and mu. A dispersive
material is smoothed in its INSTANTANEOUS response eps_infinity alone, which is the
limitation MEEP states first in doc/docs/Subpixel_Smoothing.md.

Array backend: every array operation goes through ``grid.xp``. This module never
imports cupy or meep.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy

E_COMPONENTS = ("Ex", "Ey", "Ez")  # Components whose chi1inv row this module can produce.
AXIS_NAMES = ("x", "y", "z")
# Yee shift of each E component, in half cells, matching dispersion.component_coordinates.
E_YEE_SHIFTS = {"Ex": (1, 0, 0), "Ey": (0, 1, 0), "Ez": (0, 0, 1)}

SMOOTHING_DIAMETER = 1.0  # structure.cpp:220 — the voxel is one cell across, per axis.
# meep/src/anisotropic_averaging.cpp:101 — below this the gradient carries no direction
# and MEEP point-samples instead. Absolute, and compared against a gradient that still
# carries MEEP's factor of R = the voxel diameter, so the reproduction is scale-exact.
GRADIENT_FLOOR = 1e-8
NQUAD3 = 50  # sphere-quad.cpp NQUAD3: the 50-point degree-11 octahedral rule used in 3-D.
# Midpoint samples per axis per voxel. RETUNED from 16 to 32 when the analytic planar
# fill landed, and for a different reason than the old note here gave: with the fill
# analytic, `s` no longer sets ACCURACY (a grating reads 1.5e-6 against CPU MEEP at
# s = 8 and at s = 256 alike). What it still sets is DETECTION — a voxel whose s**3
# samples are all equal is declared homogeneous and returned untouched, so a feature
# thinner than dx/s is invisible — how binary a voxel is judged to be, how tight the
# analytic fill's cross-check against the sums is, and the accuracy of the voxels the
# plane fit declines.
#
# The cost is a ONE-TIME pre-pass, so the honest unit is time steps of the run it
# precedes. Measured on the 13.5k-voxel grating grid (resolution 15): the pre-pass runs
# at 8.4 ns per fine sample and one time step costs 101 ns per voxel, so a scalar `s`
# costs about s**3 / 12 time steps and a (s, 1, s) triple about s**2 / 12. That ratio is
# resolution-independent — both sides scale with the voxel count — which is what makes
# it a defensible default rather than a number tuned to one grid:
#
#     s        scalar pre-pass   as time steps    (s,1,s) triple   as time steps
#     16       0.465 s              341            0.031 s             23
#     32       2.93  s             2731            0.101 s             74
#     64       23    s (est.)     21845            0.327 s            240
#     128      ---                174763           1.232 s            940
#     256      ---               1398101           5.44  s           4000
#
# 32 is the largest that stays well under a typical 1k-20k-step run while doubling the
# detection reach and halving the declined-voxel error against 16. 64 — which is where
# the pre-analytic plan was heading, because accuracy demanded it — would cost more than
# the whole run and now buys nothing measurable. The planar fit itself is not what costs:
# it runs over the interface only, O(N**(2/3)) of the grid, and adds 0.006 s to the 2.93 s
# above.
DEFAULT_SUPERSAMPLE = 32
FILL_RULES = ("auto", "planar", "sampled")  # How the voxel fill fraction is obtained.
DEFAULT_FILL_RULE = "auto"  # planar for a callable epsilon, sampled for a fine array.
# Fine samples evaluated in one pass. Bounds peak memory rather than total work: the
# pixel loop is chunked until it fits. 8M float64 is 64 MB per temporary.
_CHUNK_BUDGET = 8_000_000

# --- constants of the analytic planar path ---------------------------------------
# Lateral offset of the four slope probes from the voxel centre, in cells. Small enough
# that all five probes stay inside the voxel (so the plane measured is the one that
# actually cuts it), large enough that the slope difference is not a cancellation.
PLANE_PROBE = 0.25
# The fifth probe pays for a curvature correction as well as the planarity check. For a
# surface w = h0 + (h_uu u**2 + h_vv v**2)/2 the four side probes read
# h0 + (h_uu + h_vv) a**2 / 4 while the centre reads h0, and the voxel's fill wants the
# MEAN height over the lateral square, h0 + (h_uu + h_vv) dx**2 / 24. The ratio of those
# two offsets is 1 / (6 PLANE_PROBE**2) and is exactly this gain, so the leading O(dx)
# term of the linearization's fill error cancels. Measured on a cylinder of radius
# 0.4217, worst averaged voxel: without it 1.10e-2 / 7.94e-3 / 4.92e-3 at resolutions
# 20 / 40 / 80, with it 2.58e-3 / 2.71e-3 / 1.06e-3 (RMS 1.54e-3 / 1.03e-3 / 3.11e-4).
# The worst voxel plateaus rather than converging because the residue is the voxels the
# surface leaves through a LATERAL face, where the mean-height argument does not hold —
# so this is a large constant improvement, not a change of order. A plane makes the four
# probes and the centre agree to the last bit, so the correction is identically zero
# there and planar exactness is untouched.
PLANE_CURVATURE_GAIN = 1.0 / (6.0 * PLANE_PROBE * PLANE_PROBE)
# Bisection halvings per probe column. The bracket is at most ~2.3 cells wide, so 52
# halvings drive it to 5e-16 cells — below the float64 resolution of the absolute
# coordinate the callable is evaluated at, i.e. as exact as the interface can be located.
PLANE_BISECTION_STEPS = 52
# Below this the SMALLER of the two fitted slopes is taken as exactly zero. The
# four-term closed form divides by both slopes, so a tiny one is a cancellation that
# costs ~5e-16/(p q); dropping it instead falls back to the two-term form, which is the
# exact b = 0 limit, and costs O(small**2 / big). 1e-4 balances the two at about 1e-8 in
# the fill — see test_analytic_fill_matches_a_brute_force_cube.
PLANE_SLOPE_FLOOR = 1e-4
# And below this the LARGER slope is dropped too, leaving the plain clamp. This one can
# be far smaller because the two-term form divides only by the larger slope, so its
# round-off is 5e-16/big rather than 5e-16/(big*small); dropping a slope this small
# costs at most big/4 in the fill, and only where the plane grazes a cube face.
_PLANE_SLOPE_FLOOR_BIG = 1e-8
# Largest |h(centre) - mean of the four probe heights|, in cells, that still counts as
# "locally planar". A plane gives ~1e-15 here and a corner gives O(1), so this only has
# to separate curvature from kinks. For a smooth surface the residual is
# (kappa_u + kappa_v) dx**2 / 64 and the fill error it implies is 64/24 times it, so
# 4e-3 caps the curvature-induced fill error at about 1.1e-2 — beyond that the voxel
# falls back to the midpoint sums, whose own error is 1/(2s).
PLANE_RESIDUAL_TOLERANCE = 4e-3
# Slopes above this mean the sphere gradient chose the wrong dominant axis and the
# height-function description of the interface is ill-conditioned; fall back instead.
PLANE_SLOPE_CEILING = 4.0
# Safety net on the closed form: the analytic fill must agree with the midpoint-sum fill
# to within the sum's own resolution, times this. The midpoint sums locate a plane to
# 0.5 * sum_j |n_j| / s_j of a cell, so anything past 3x that is not a coarse sum and a
# fine analytic answer disagreeing, it is one of them being wrong.
PLANE_CROSSCHECK_SLACK = 3.0


def _shift3(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """sphere-quad.cpp ``SHIFT3(x, y, z)``: rotate the coordinate triple one place."""
    x, y, z = point
    return (z, x, y)


def _spherical_quadrature_points() -> tuple[list[tuple[float, float, float]], list[float]]:
    """The 50-point, 11th-degree unit-sphere rule of sphere-quad.cpp, unsorted.

    A transcription of ``spherical_quadrature_points(..., 50)`` — McLaren's formula
    U3:11-1, chosen there because it preserves octahedral (simple-cubic) symmetry, so
    an axis-aligned interface produces an exactly axis-aligned normal. The four
    orbits and their integer weights over 725760 are MEEP's; the loop bodies mutate
    the seed point in place exactly as the C does, because the sign flips and the
    coordinate rotations compose across iterations.
    """
    points: list[tuple[float, float, float]] = []
    weights: list[float] = []

    def emit(point, weight):
        points.append(point)
        weights.append(weight)

    x0, y0, z0 = 1.0, 0.0, 0.0
    weight = 9216 / 725760.0
    for _ in range(2):
        x0 = -x0
        for _ in range(3):
            x0, y0, z0 = _shift3((x0, y0, z0))
            emit((x0, y0, z0), weight)

    x0 = y0 = math.sqrt(0.5)
    z0 = 0.0
    weight = 16384 / 725760.0
    for _ in range(2):
        x0 = -x0
        for _ in range(2):
            y0 = -y0
            for _ in range(3):
                x0, y0, z0 = _shift3((x0, y0, z0))
                emit((x0, y0, z0), weight)

    x0 = y0 = z0 = math.sqrt(1.0 / 3.0)
    weight = 15309 / 725760.0
    for _ in range(2):
        x0 = -x0
        for _ in range(2):
            y0 = -y0
            for _ in range(2):
                z0 = -z0
                emit((x0, y0, z0), weight)

    x0 = y0 = math.sqrt(1.0 / 11.0)
    z0 = 3 * x0
    weight = 14641 / 725760.0
    for _ in range(2):
        x0 = -x0
        for _ in range(2):
            y0 = -y0
            for _ in range(2):
                z0 = -z0
                for _ in range(3):
                    x0, y0, z0 = _shift3((x0, y0, z0))
                    emit((x0, y0, z0), weight)

    if len(points) != NQUAD3:
        raise ValueError(
            f"the 3-D sphere quadrature must hold {NQUAD3} points, built {len(points)}."
        )
    return points, weights


def _sort_by_distance(points: list, weights: list) -> tuple[list, list]:
    """sphere-quad.cpp ``sort_by_distance``: greedily maximize each point's spacing.

    MEEP's generator writes the table in this order and
    ``material_function::normal_vector`` depends on it: the "epsilon is uniform here"
    shortcut looks only at the FIRST ``2**ndims`` entries, so which points those are
    decides whether a given voxel is averaged at all. The float32 cast of the squared
    distance is MEEP's (``double d2 = float(dist2(...))``) and is kept because it can
    break ties differently from a float64 comparison.
    """
    points = list(points)
    weights = list(weights)
    count = len(points)
    for i in range(1, count):
        best_min = 0.0
        best_sum = 0.0
        best_index = i
        for j in range(i, count):
            minimum = 1e20
            total = 0.0
            for k in range(i):
                separation = float(
                    numpy.float32(
                        sum((points[k][axis] - points[j][axis]) ** 2 for axis in range(3))
                    )
                )
                minimum = min(minimum, separation)
                total += separation
            if minimum > best_min or (minimum == best_min and total > best_sum):
                best_min = minimum
                best_sum = total
                best_index = j
        points[i], points[best_index] = points[best_index], points[i]
        weights[i], weights[best_index] = weights[best_index], weights[i]
    return points, weights


def sphere_quadrature() -> tuple[numpy.ndarray, numpy.ndarray]:
    """MEEP's 3-D unit-sphere quadrature: ``(points (50, 3), weights (50,))``.

    ``sphere_quad[2]`` of the generated ``sphere-quad.h``, in the same order, so index
    ``i`` here is index ``i`` there. Weights sum to 1; the rule integrates polynomials
    through degree 11 exactly, which is what makes the normal it produces
    second-order accurate for a locally planar interface.
    """
    points, weights = _spherical_quadrature_points()
    points, weights = _sort_by_distance(points, weights)
    return (
        numpy.asarray(points, dtype=numpy.float64),
        numpy.asarray(weights, dtype=numpy.float64),
    )


_SPHERE_POINTS, _SPHERE_WEIGHTS = sphere_quadrature()
# anisotropic_averaging.cpp:66 — min_iters = 1 << number_of_directions(dim); this engine
# is always 3-D. The uniformity shortcut compares consecutive values over this prefix.
_UNIFORM_PREFIX = 1 << 3


def normalize_supersample(supersample: Any) -> tuple[int, int, int]:
    """Resolve ``supersample`` into a per-axis triple of midpoint counts.

    A scalar applies to all three axes. A triple lets an axis the structure does not
    vary along cost one sample instead of ``s`` — a y-invariant grating at
    ``(64, 1, 64)`` is 64x cheaper than ``64`` and numerically identical.
    """
    if isinstance(supersample, (int, numpy.integer)) and not isinstance(supersample, bool):
        values = (int(supersample),) * 3
    elif isinstance(supersample, Sequence) and not isinstance(supersample, (str, bytes)):
        if len(supersample) != 3:
            raise ValueError(
                f"supersample takes a scalar or a per-axis triple, got {len(supersample)} entries: "
                f"{supersample!r}."
            )
        values = tuple(int(value) for value in supersample)
    else:
        raise ValueError(
            f"supersample must be an int or a 3-tuple of ints, got {supersample!r}."
        )
    for axis, value in enumerate(values):
        if value < 1:
            raise ValueError(
                f"supersample must be >= 1 on every axis; axis {AXIS_NAMES[axis]} got {value}. "
                f"1 means one midpoint sample, i.e. no averaging along that axis, which is "
                f"correct only where the structure does not vary."
            )
    return values  # type: ignore[return-value]


def _require_component(component: Any) -> str:  # Validate the E component being smoothed.
    if component not in E_COMPONENTS:
        raise ValueError(
            f"component selects which chi1inv row to smooth and must be one of {E_COMPONENTS}, "
            f"got {component!r}. Magnetic (mu) smoothing is not implemented on this engine."
        )
    return component


def voxel_centers(grid: Any, component: str) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Per-axis coordinates of the voxel centres one E component is smoothed on.

    Axis ``d`` index ``i`` sits at ``grid.axis_origin(d) + (i + shift_d/2) * dx``, the
    same rule ``dispersion.component_coordinates`` states and the position MEEP centres
    ``gv.dV(here, 1)`` on. Measuring from ``-L/2`` instead of ``axis_origin`` is the
    half-cell error that cost 6.6e-2 on a grating and only shows up on an odd-count
    axis, so it is taken from the grid and never re-derived.
    """
    _require_component(component)
    shifts = E_YEE_SHIFTS[component]
    counts = (grid.nx, grid.ny, grid.nz)
    return tuple(  # type: ignore[return-value]
        grid.axis_origin(axis)
        + (numpy.arange(counts[axis], dtype=numpy.float64) + 0.5 * shifts[axis]) * grid.dx
        for axis in range(3)
    )


def _axis_length(grid: Any, axis: int) -> float:  # Physical length of one axis, from the grid.
    return (grid.nx, grid.ny, grid.nz)[axis] * grid.dx


def _wrap_axis(grid: Any, axis: int, coordinates: Any) -> Any:
    """Fold coordinates into the periodic cell ``[axis_origin, axis_origin + L)``.

    Every epsilon evaluation goes through this: the fine lattice of a boundary
    voxel starts half a cell below the cell floor on an unshifted axis, and the
    sphere the interface normal is read from reaches a full cell past either end.
    This engine's boundaries are always periodic — a PML is a material inside the
    cell, not a wall — so the material there is the material at the wrapped point,
    which is what MEEP evaluates too (``ensure_periodicity``, on by default).
    Without the wrap a structure touching the cell edge is silently smoothed
    against whatever the caller's epsilon happens to return outside the cell.
    """
    origin = grid.axis_origin(axis)
    return origin + (coordinates - origin) % _axis_length(grid, axis)


def fine_coordinates(
    grid: Any, component: str, supersample: Any = DEFAULT_SUPERSAMPLE
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Per-axis coordinates of the supersampled lattice the voxel means are summed over.

    Axis ``d`` holds ``n_d * s_d`` samples; sample ``J = i * s_d + m`` is the midpoint
    of subcell ``m`` of voxel ``i``::

        axis_origin(d) + (i + shift_d/2 - 1/2) * dx + (m + 1/2) * dx / s_d

    wrapped into the periodic cell (:func:`_wrap_axis`) — one uniform lattice of
    spacing ``dx / s_d``, one period long, whose leading samples on an unshifted axis
    are the top of the cell because cell 0's voxel straddles the boundary. A caller
    building the epsilon array form of :func:`smooth_inverse_epsilon` must sample on
    THESE coordinates; anything else reintroduces the half-cell registration error
    the array path exists to avoid.
    """
    _require_component(component)
    shifts = E_YEE_SHIFTS[component]
    rates = normalize_supersample(supersample)
    counts = (grid.nx, grid.ny, grid.nz)
    axes = []
    for axis in range(3):
        rate = rates[axis]
        start = grid.axis_origin(axis) + (0.5 * shifts[axis] - 0.5) * grid.dx
        step = grid.dx / rate
        raw = start + (numpy.arange(counts[axis] * rate, dtype=numpy.float64) + 0.5) * step
        axes.append(_wrap_axis(grid, axis, raw))
    return tuple(axes)  # type: ignore[return-value]


@dataclass
class SmoothedEpsilon:
    """One component's smoothed material, plus what the reduction to a scalar cost.

    Attributes:
        component: The E component whose chi1inv row this is.
        inverse_epsilon: float32 ``chi1inv[component][own direction]`` at the component's
            own Yee positions — the array ``FdtdDriver`` installs.
        epsilon: float32 effective permittivity for readback and the CFL check: the
            float32 reciprocal of ``inverse_epsilon`` in the smoothed voxels, and
            exactly the point-sampled input value in the untouched ones (the same
            pair, computed by the same operations, that ``set_epsilon`` stores).
        row: All three entries of this component's chi1inv row in float64, every one of
            them evaluated on the component's OWN voxel. The diagonal entry is MEEP's;
            the two off-diagonals are NOT — ``structure_chunk::set_chi1inv``
            (anisotropic_averaging.cpp:255-257) takes them from a second
            ``eff_chi1inv_row`` call on ``gv.dV(here - shift1)``, a voxel half a cell
            LOWER along the component's own axis, because ``chi1inv[c][d]`` multiplies
            ``D_d``, which does not live where ``D_c`` does. A future per-component
            tensor consumer must re-evaluate them on that shifted voxel rather than
            take these; they are exported only because they are what the projection
            formula produced here, and they are identically zero in every configuration
            this engine accepts anyway (see ``offdiagonal_magnitude``).
        normal: Per-axis components of the unit interface normal, zero where the voxel
            was not averaged.
        mean_epsilon / mean_inverse_epsilon: ``<eps>`` and ``<1/eps>`` over each voxel.
        smoothed: Boolean volume — True where the anisotropic average was applied,
            False where MEEP's shortcut point-samples instead.
        interface_fraction: Fraction of voxels that were smoothed. 0.0 means the
            structure never met a voxel boundary and the result is the unsmoothed array.
        unresolved: Boolean volume — True where the fine lattice DID resolve two
            materials in the voxel and the scheme still declined to average it, so the
            voxel carries the point-sampled value. The only way MEEP's shortcut is
            wrong rather than merely approximate, and the one number here that can
            actually warn; see ``unresolved_fraction``.
        unresolved_fraction: ``unresolved`` as a fraction of the voxels the fine lattice
            found inhomogeneous — 0.0 when every resolved interface was averaged.
            Measured 3.6 % (one corner voxel) on a rectangular bar, where that voxel
            held 17 % of the other material and came back 53 % away from MEEP's
            analytic value. Nonzero means some resolved geometry was silently ignored.
        offdiagonal_magnitude: Largest ``|chi1inv[component][d]|`` over the two
            directions the shared scalar cannot carry, relative to the diagonal — the
            part of MEEP's tensor this engine drops, and 0 exactly when the normal has
            no component along the smoothed axis. It CANNOT warn about anything a
            caller reaching this through :meth:`FdtdDriver.set_epsilon_smoothed` can
            hit: that method requires epsilon to be invariant along the component's own
            axis, which forces the gradient along that axis to vanish exactly, so this
            is float round-off (measured 1.1e-17 on a grating) in every accepted run.
            Watch ``unresolved_fraction`` instead.
        interface_resolution: ``dx / (2 * min(supersample))`` — how precisely a material
            boundary is located by the midpoint sums, in length units. A worst-axis
            bound: an axis given ``supersample=1`` counts, because 1 is only exact
            where the structure does not vary along it. It bounds the SAMPLED voxels
            only; a voxel in ``planar`` beat it by ten orders of magnitude.
        fill_rule: Which fill rule actually ran, ``"planar"`` or ``"sampled"`` — never
            ``"auto"``, which is resolved before anything is computed.
        planar: Boolean volume — True where the fill fraction and the normal came from
            the analytic interface plane rather than from the midpoint sums.
        planar_fraction: ``planar`` as a fraction of the SMOOTHED voxels. 1.0 means
            every averaged voxel was analytic and the result does not depend on
            ``supersample`` at all; 0.0 under ``fill_rule="planar"`` means nothing
            passed the plane checks and the answer is the sampled one.
        planarity_residual: Worst ``|h(centre) - mean of the four probe heights|`` over
            the accepted voxels, in cells. Bit-level zero for a genuine plane; for a
            curved surface it is ``(kappa_u + kappa_v) dx**2 / 64``, and the fill error
            it implies is 64/24 times it. This is the analytic path's own error bar.
    """

    component: str
    inverse_epsilon: Any
    epsilon: Any
    row: Mapping[str, Any]
    normal: Mapping[str, Any]
    mean_epsilon: Any
    mean_inverse_epsilon: Any
    smoothed: Any
    interface_fraction: float
    unresolved: Any
    unresolved_fraction: float
    offdiagonal_magnitude: float
    interface_resolution: float
    fill_rule: str
    planar: Any
    planar_fraction: float
    planarity_residual: float

    def __repr__(self) -> str:  # Compact, and leads with what the caller must judge.
        return (
            f"SmoothedEpsilon({self.component}, smoothed {self.interface_fraction:.1%} of voxels, "
            f"{self.planar_fraction:.1%} of them analytic ({self.fill_rule}), "
            f"{self.unresolved_fraction:.1%} of resolved interfaces left unsmoothed, "
            f"dropped off-diagonal {self.offdiagonal_magnitude:.2e}, "
            f"sampled interfaces located to {self.interface_resolution:.3g})"
        )


class _EpsilonSource:
    """Adapter over the two accepted epsilon spellings: a callable, or a fine array."""

    def __init__(self, epsilon: Any, grid: Any, component: str, rates: tuple[int, int, int]):
        self.grid = grid
        self.rates = rates
        self.xp = grid.xp
        self.callable = None
        self.array = None
        if callable(epsilon):
            self.callable = epsilon
            return
        values = self.xp.asarray(epsilon, dtype=self.xp.float64)
        if values.ndim != 3:
            raise ValueError(
                f"A supersampled epsilon array must be 3-D, got {values.ndim} dimensions. Pass a "
                f"callable epsilon(x, y, z) instead if you have no array."
            )
        expected = tuple(
            count * rate
            for count, rate in zip((grid.nx, grid.ny, grid.nz), rates)
        )
        shape = tuple(int(value) for value in values.shape)
        if shape != expected:
            raise ValueError(
                f"A supersampled epsilon array must have shape {expected} — the grid shape "
                f"{(grid.nx, grid.ny, grid.nz)} times supersample {rates} — got {shape}. Build it "
                f"on smoothing.fine_coordinates(grid, {component!r}, {rates}); sampling anywhere "
                f"else reintroduces the half-cell registration error."
            )
        self.array = values
        self.fine_origin = tuple(
            grid.axis_origin(axis) + (0.5 * E_YEE_SHIFTS[component][axis] - 0.5) * grid.dx
            for axis in range(3)
        )
        self.fine_step = tuple(grid.dx / rate for rate in rates)
        self.fine_count = expected

    def fine_block(self, axis_coordinates, index_ranges):
        """Epsilon on the fine lattice of one chunk, as ``(nx*sx, ny*sy, nz*sz)``."""
        if self.callable is not None:
            return self._evaluate(*axis_coordinates)
        slices = tuple(
            slice(start * rate, stop * rate)
            for (start, stop), rate in zip(index_ranges, self.rates)
        )
        return self.array[slices]

    def at(self, x, y, z):
        """Epsilon at arbitrary points — the sphere quadrature, and the plane bisection.

        The array form has nothing between its fine samples, so a sphere node is
        answered with the fine cell that CONTAINS it: the normal an array epsilon
        produces is quantized to the fine lattice, where a callable's is exact. The
        voxel means are unaffected (both paths sum the same lattice) and so is the
        smooth/point-sample decision, so nothing :meth:`FdtdDriver.set_epsilon_smoothed`
        installs can move — ``n_d = 0`` there and the diagonal does not use the normal.
        It does move the off-diagonals and the harmonic branch: measured on a
        rectangular bar's corner voxel, an array epsilon read the normal as
        (-0.707, 0, 0.707) where the callable read (-0.848, 0, 0.530), a 6.9 %
        difference in that voxel's ``Ez`` diagonal. Pass the callable when the tensor
        row matters.

        The planar fill bisects through this method too, and that is why it refuses an
        array outright (:func:`resolve_fill_rule`) rather than running on a quantized
        answer: a bisection on a piecewise-constant array converges to a fine-cell edge,
        which is the ``dx/(2s)`` quantization the analytic fill exists to remove.
        """
        if self.callable is not None:
            return self._evaluate(x, y, z, broadcast=False)
        xp = self.xp
        indices = []
        for axis, coordinate in enumerate((x, y, z)):
            raw = (coordinate - self.fine_origin[axis]) / self.fine_step[axis]
            index = xp.floor(raw).astype(xp.int64) % self.fine_count[axis]
            indices.append(index)
        return self.array[indices[0], indices[1], indices[2]]

    def center_values(self, center_coordinates):
        """Epsilon exactly at the voxel centres, or None when only samples exist.

        MEEP's trivial branch point-samples ``chi1p1(ft, v.center())`` — the exact
        centre, not the nearest quadrature sample — so the callable form does the
        same. The array form has nothing between its fine samples and returns None;
        the caller falls back to the fine sample containing the centre, which is the
        centre itself for odd ``supersample`` and ``dx/(2s)`` above it for even.
        """
        if self.callable is None:
            return None
        return self._evaluate(*center_coordinates)

    def _evaluate(self, x, y, z, broadcast: bool = True):  # Call the user's epsilon, checked.
        xp = self.xp
        x = _wrap_axis(self.grid, 0, x)
        y = _wrap_axis(self.grid, 1, y)
        z = _wrap_axis(self.grid, 2, z)
        if broadcast:
            arguments = (x[:, None, None], y[None, :, None], z[None, None, :])
            expected = (x.size, y.size, z.size)
        else:
            arguments = (x, y, z)
            expected = tuple(int(n) for n in xp.broadcast_shapes(x.shape, y.shape, z.shape))
        values = xp.asarray(self.callable(*arguments), dtype=xp.float64)
        try:
            values = xp.broadcast_to(values, expected)
        except ValueError:
            raise ValueError(
                f"epsilon(x, y, z) must broadcast with its coordinate arguments: called with "
                f"shapes {tuple(tuple(int(n) for n in a.shape) for a in arguments)} it returned "
                f"shape {tuple(int(n) for n in values.shape)}, which does not broadcast to "
                f"{expected}. Write the callable in vectorized NumPy operations (no float(), no "
                f"scalar if/else on the coordinates)."
            ) from None
        if not bool(xp.all(xp.isfinite(values))) or bool(xp.any(values <= 0.0)):
            raise ValueError(
                "epsilon(x, y, z) returned a non-finite or non-positive value. Subpixel "
                "smoothing averages 1/eps, so a zero or negative permittivity is not a hard "
                "material here but a divide by zero; MEEP refuses to average it too "
                "(meepgeom.cpp eps_ever_negative)."
            )
        return values


def _chunk_ranges(counts: tuple[int, int, int], rates: tuple[int, int, int]) -> list:
    """Split the voxel grid so one chunk's fine lattice stays under the memory budget.

    Chunks the two leading axes only; the trailing axis is never split, so the fine
    block handed to the reducer is always contiguous in the fastest-varying direction.
    """
    per_voxel = rates[0] * rates[1] * rates[2]
    row_samples = counts[2] * per_voxel
    y_chunk = max(1, min(counts[1], int(_CHUNK_BUDGET // max(1, row_samples))))
    plane_samples = y_chunk * row_samples
    x_chunk = max(1, min(counts[0], int(_CHUNK_BUDGET // max(1, plane_samples))))
    ranges = []
    for x_start in range(0, counts[0], x_chunk):
        x_stop = min(counts[0], x_start + x_chunk)
        for y_start in range(0, counts[1], y_chunk):
            y_stop = min(counts[1], y_start + y_chunk)
            ranges.append(((x_start, x_stop), (y_start, y_stop), (0, counts[2])))
    return ranges


def _positive_power(x, exponent, xp):  # max(x, 0) ** exponent, the pieces of the cube spline.
    positive = xp.where(x > 0.0, x, 0.0)
    return positive ** exponent


def cube_plane_fill(level, slope_u, slope_v, xp=numpy):
    """Exact fraction of a unit cube lying below the plane ``w = level + p u + q v``.

    The cube is ``[-1/2, 1/2]**3`` and ``(u, v, w)`` are its own axes, so ``level`` is
    the interface height at the cube centre in cells and ``p``/``q`` are the height's
    lateral slopes. This is the closed form MEEP reaches through
    ``box_overlap_with_object`` for a geometric object (``meepgeom.cpp``), written for
    a plane rather than for each primitive: it replaces the fill fraction's midpoint
    quantization with an identity, so the answer no longer depends on ``supersample``.

    Derivation. ``F = int int clamp(g + p u + q v, 0, 1) du dv`` over the centred unit
    square, with ``g = level + 1/2``, and ``clamp(x, 0, 1) = ramp(x) - ramp(x - 1)``, so
    ``F = J(g) - J(g - 1)`` with ``J(g) = int int max(g + p u + q v, 0)``. Differentiating
    ``J`` twice in ``g`` gives the density of ``-(a U + b V)`` for ``U, V`` uniform on
    ``[0, 1]`` and ``a = |p|``, ``b = |q|`` — a trapezoid — and integrating that twice
    back, with the linear part fixed by ``J -> G + (a + b)/2`` as ``G -> +inf``::

        J = G + (a+b)/2 + [ psi(-G) - psi(-G-a) - psi(-G-b) + psi(-G-a-b) ] / (a b)

    with ``psi(x) = max(x, 0)**3 / 6`` and ``G = g - (a + b)/2``. Verified term by term
    against a brute-force cube in ``test_analytic_fill_matches_a_brute_force_cube``.

    The two degenerate limits are taken explicitly rather than by dividing by a slope
    that is nearly zero — the axis-aligned interface a grating is made of is exactly
    that case, and it is where the four-term numerator cancels completely. As ``b -> 0``
    the bracket over ``a b`` tends to ``[psi'(-G) - psi'(-G-a)] / a``, and as ``a -> 0``
    as well to ``max(-G, 0)``; ``_PLANE_SLOPE_FLOOR`` and ``_PLANE_SLOPE_FLOOR_BIG`` say
    where each limit is taken. Both are exact when the slope really is zero.

    Args:
        level: Interface height at the cube centre, in cells (the cube is one cell).
        slope_u / slope_v: ``dh/du`` and ``dh/dv``, dimensionless. Either sign.
        xp: The array module; every operation goes through it.

    Returns:
        The fill fraction, clipped into ``[0, 1]`` — the clip only removes a few ULP of
        round-off at the two ends, where the plane misses the cube.
    """
    a = xp.abs(xp.asarray(slope_u, dtype=xp.float64))
    b = xp.abs(xp.asarray(slope_v, dtype=xp.float64))
    big = xp.maximum(a, b)
    small = xp.minimum(a, b)
    # Two floors, because they trade off against different errors: the four-term form
    # divides by big*small and the two-term form only by big.
    big = xp.where(big > _PLANE_SLOPE_FLOOR_BIG, big, 0.0)
    small = xp.where((small > PLANE_SLOPE_FLOOR) & (big > 0.0), small, 0.0)
    safe_big = xp.where(big > 0.0, big, 1.0)
    safe_small = xp.where(small > 0.0, small, 1.0)

    def integral(shifted):  # J above, with its two degenerate limits.
        four = (
            _positive_power(-shifted, 3, xp)
            - _positive_power(-shifted - big, 3, xp)
            - _positive_power(-shifted - small, 3, xp)
            + _positive_power(-shifted - big - small, 3, xp)
        ) / (6.0 * safe_big * safe_small)
        two = (
            _positive_power(-shifted, 2, xp) - _positive_power(-shifted - big, 2, xp)
        ) / (2.0 * safe_big)
        none = _positive_power(-shifted, 1, xp)
        correction = xp.where(big <= 0.0, none, xp.where(small <= 0.0, two, four))
        return shifted + 0.5 * (big + small) + correction

    offset = xp.asarray(level, dtype=xp.float64) + 0.5 - 0.5 * (big + small)
    return xp.clip(integral(offset) - integral(offset - 1.0), 0.0, 1.0)


@dataclass
class _VoxelMoments:
    """What one pass over a chunk's fine lattice yields, per voxel.

    Attributes:
        mean_epsilon / mean_inverse: the midpoint sums ``<eps>`` and ``<1/eps>``.
        homogeneous: every fine sample in the voxel bit-equal.
        center: the fine sample containing the voxel centre.
        low / high: the smallest and largest fine sample in the voxel.
        binary: every fine sample equal to ``low`` or to ``high`` — the precondition of
            the analytic planar path, which describes a voxel by ONE interface between
            TWO materials and would otherwise average a third one away silently.
    """

    mean_epsilon: Any
    mean_inverse: Any
    homogeneous: Any
    center: Any
    low: Any
    high: Any
    binary: Any


def _voxel_moments(fine, rates, xp) -> _VoxelMoments:
    """The midpoint sums over each voxel's fine lattice, plus what the planar path needs.

    The means are MIDPOINT sums over the fine lattice. MEEP has two rules here and this
    is neither of them exactly: for a geometric object ``meepgeom.cpp
    eff_chi1inv_matrix`` uses the EXACT analytic overlap of the voxel with the object
    (``box_overlap_with_object``), and only for a user material function does it fall
    back to ``eff_chi1inv_row``'s ``ms``-cubed sum — which is a LEFT-EDGE rule
    (``v.get_min_corner() + vec(i*d.x()/ms, ...)``, ``i`` from 0 to ``ms-1``, so the
    max corner is never sampled), started at ``ms = 10`` and doubled until the means
    settle to ``tol``. The midpoint rule here is deliberately the better of the two
    sums: it locates a planar interface to ``dx/(2s)`` rather than ``dx/s``, and it is
    unbiased where the left-edge rule is not.

    What that still costs against MEEP's ANALYTIC fill is first order in ``1/s``, and it
    used to be the dominant error of this whole module. Measured per voxel on a
    rectangular bar at resolution 20, worst face voxel: ``s`` = 8 -> 7.4e-2,
    16 -> 3.5e-2, 32 -> 1.6e-2, 64 -> 8.7e-3, 128 -> 4.5e-3, 256 -> 2.1e-3. End to end on
    the grating transmission spectrum of ``test_smoothing.py``, worst frequency against
    CPU MEEP: unsmoothed 3.9e-1, then 2.3e-1 / 3.4e-2 / 5.8e-2 / 8.4e-3 / 1.5e-2 / 2.1e-3
    for the same ``s`` ladder — first order in ``1/s`` with a sawtooth on top, because
    which side of a subcell the interface lands on is not monotone in ``s``.

    That ladder is why ``fill_rule="planar"`` exists and is the default: it replaces
    these sums with :func:`cube_plane_fill` wherever the interface is locally a plane,
    which is every voxel of a grating except its corners, and the ``1/s`` term stops
    being the answer's accuracy. These sums remain the FALLBACK — for a corner, a
    non-binary voxel, or an epsilon supplied as an array — and they remain what decides
    whether a voxel meets an interface at all, so ``s`` still sets how thin a feature is
    seen.

    ``homogeneous`` is every fine sample in the voxel being bit-equal: that voxel meets
    no interface, so it is returned untouched rather than round-tripped through a sum —
    which is what makes a uniform medium, and an interface lying exactly on a voxel
    boundary, come back bit-identical to the unsmoothed array instead of one ULP away
    from it.
    """
    shape = fine.shape
    blocks = (
        shape[0] // rates[0], rates[0],
        shape[1] // rates[1], rates[1],
        shape[2] // rates[2], rates[2],
    )
    grouped = fine.reshape(blocks)
    axes = (1, 3, 5)
    count = float(rates[0] * rates[1] * rates[2])
    mean_epsilon = grouped.sum(axis=axes) / count
    mean_inverse = (1.0 / grouped).sum(axis=axes) / count
    first = grouped[:, :1, :, :1, :, :1]
    homogeneous = (grouped == first).all(axis=axes)
    low = grouped.min(axis=axes)
    high = grouped.max(axis=axes)
    binary = (
        (grouped == low[:, None, :, None, :, None])
        | (grouped == high[:, None, :, None, :, None])
    ).all(axis=axes)
    # The fine sample containing the voxel centre: subcell s//2 on each axis. For odd
    # s that IS the centre; for even s it is dx/(2s) above it. The callable form
    # replaces this with an exact-centre evaluation; the array form has nothing
    # between its samples, and this deterministic pick avoids the floor() coin-toss
    # an even s puts exactly on a fine-cell boundary.
    center = grouped[
        :, rates[0] // 2, :, rates[1] // 2, :, rates[2] // 2
    ]
    return _VoxelMoments(
        mean_epsilon=mean_epsilon,
        mean_inverse=mean_inverse,
        homogeneous=homogeneous,
        center=xp.ascontiguousarray(center),
        low=low,
        high=high,
        binary=binary,
    )


def _interface_normal(source, centers, radius, xp):
    """MEEP's ``material_function::normal_vector``, vectorized over a chunk of voxels.

    ``gradient = R * sum_i w_i * eps(p + R * u_i) * u_i`` over the 50-point sphere rule,
    with ``R`` the voxel diameter — MEEP keeps that factor, and the 1e-8 floor it is
    compared against is absolute, so the factor is kept here too. Returns the gradient
    and MEEP's uniformity shortcut: consecutive samples over the first 8 sphere points
    all bit-equal means "do not average this voxel", whatever the other 42 say.
    """
    x, y, z = centers
    shape = (x.size, y.size, z.size)
    gradient = [xp.zeros(shape, dtype=xp.float64) for _ in range(3)]
    prefix = []
    for index in range(NQUAD3):
        unit = _SPHERE_POINTS[index]
        weight = float(_SPHERE_WEIGHTS[index])
        values = source.at(
            (x + radius * unit[0])[:, None, None],
            (y + radius * unit[1])[None, :, None],
            (z + radius * unit[2])[None, None, :],
        )
        values = xp.broadcast_to(xp.asarray(values, dtype=xp.float64), shape)
        for axis in range(3):
            if unit[axis] != 0.0:
                gradient[axis] = gradient[axis] + values * (weight * unit[axis])
        if index < _UNIFORM_PREFIX:
            prefix.append(values)
    for axis in range(3):
        gradient[axis] = gradient[axis] * radius
    uniform = xp.ones(shape, dtype=bool)
    for index in range(1, _UNIFORM_PREFIX):
        uniform = uniform & (prefix[index] == prefix[index - 1])
    return gradient, uniform


@dataclass
class _PlaneFit:
    """The local interface plane of a batch of voxels, and whether it may be believed."""

    accepted: Any  # Boolean, per voxel of the batch.
    fill: Any  # Fraction of the voxel on the `below` side, exact for a plane.
    below: Any  # Epsilon on the low side of the plane along the dominant axis.
    above: Any  # Epsilon on the high side.
    normal: Any  # (3, M) unit interface normal, oriented along grad(eps) like MEEP's.
    residual: Any  # |h(centre) - mean of the four probes| in cells; 0 for a plane.


def _axis_selector(axis_index, xp):  # (3, M) one-hot rows: e_axis for each voxel.
    return xp.stack(
        [xp.where(axis_index == axis, 1.0, 0.0) for axis in range(3)]
    )


def _bisect_interface(source, base, direction, reach, xp):
    """Locate a material discontinuity along ``direction`` by bisection, batched.

    ``base`` is ``(3, K)`` start points and ``direction`` ``(3, K)`` unit axis vectors;
    the search runs over signed offsets in ``[-reach, +reach]``. Returns the offset of
    the discontinuity, the material on each end, and whether the ends differed at all —
    an unbracketed column means the interface does not cross there (or crosses twice),
    which is the caller's cue that this voxel is not one plane.

    Bisection is what replaces the fill fraction's midpoint quantization: it converges
    geometrically, so :data:`PLANE_BISECTION_STEPS` halvings put the interface within
    5e-16 of a cell rather than within ``1/(2s)`` of one. The branch test is a bit-exact
    comparison against the low-end material rather than a threshold, so a third material
    met on the way sends the bracket upward and the caller's binary check rejects it.
    """
    def sample(offset):  # Epsilon at base + offset * direction, wrapped into the cell.
        return source.at(*[base[axis] + offset * direction[axis] for axis in range(3)])

    low = -xp.asarray(reach, dtype=xp.float64)
    high = -low
    value_low = xp.asarray(sample(low), dtype=xp.float64)
    value_high = xp.asarray(sample(high), dtype=xp.float64)
    bracketed = value_low != value_high
    for _ in range(PLANE_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        value = xp.asarray(sample(middle), dtype=xp.float64)
        take_low = value == value_low
        low = xp.where(take_low, middle, low)
        high = xp.where(take_low, high, middle)
    return 0.5 * (low + high), value_low, value_high, bracketed


def _fit_interface_plane(source, centers, unit, dx, xp) -> _PlaneFit:
    """Measure the interface plane of each voxel in a batch, then take its EXACT fill.

    This is the analytic half of MEEP's geometric path (``meepgeom.cpp``
    ``eff_chi1inv_matrix``) reconstructed from the material function instead of from a
    geometric object, which is the only thing this engine is given. MEEP knows the
    object, so it can ask for the exact overlap and the exact surface normal; here the
    same two numbers are RECOVERED, and they are exact for the same reason MEEP's are —
    for a locally planar interface both are closed forms once the plane is known.

    The plane is measured, not sampled. The sphere gradient (already computed, and only
    accurate to a few percent in direction for a discontinuous epsilon) is used for one
    thing only: to pick the axis ``d`` the interface is most nearly normal to, so the
    interface can be written as a height ``w = h(u, v)`` over the other two. Five
    columns parallel to ``d`` — the voxel centre and four probes ``a = PLANE_PROBE``
    cells out along ``u`` and ``v`` — are then bisected to find where the material
    changes. Four give the slopes by central differences and the fifth pays for the
    curvature, which is both the planarity check and a correction to the level::

        p = (h(+u) - h(-u)) / (2 a)      q = (h(+v) - h(-v)) / (2 a)
        c = (h(+u) + h(-u) + h(+v) + h(-v)) / 4 - h(0)
        level = h(0) + PLANE_CURVATURE_GAIN * c       residual = |c| / dx

    For a plane ``c`` is bit-level zero, so the level is ``h(0)`` exactly and the fit is
    exact. For a smoothly curved surface ``residual`` is ``(h_uu + h_vv) dx / 64`` in
    cells and the correction cancels the leading term of the linearization's fill error.
    For a corner it is O(1), or a column fails to bracket entirely. The normal follows
    from the same slopes as ``(-p, -q, 1)`` normalized, which is EXACT for a plane where
    the sphere quadrature is 4e-2 off in direction.

    Everything is then checked before it is believed, because a wrong fill here is a
    smooth, plausible, entirely finite wrong material:

    * all five columns must bracket a discontinuity,
    * all five must see the SAME two materials, in the same order,
    * the slopes must stay under :data:`PLANE_SLOPE_CEILING` (past that the sphere
      gradient chose the wrong dominant axis and the height description is degenerate),
    * the residual must stay under :data:`PLANE_RESIDUAL_TOLERANCE`.

    The caller adds the two checks that need the fine lattice: the voxel must hold no
    more than two distinct materials, and the analytic fill must agree with the
    midpoint-sum fill to within the sum's own resolution.

    Args:
        source: the :class:`_EpsilonSource`; must be the callable form, because an array
            epsilon has nothing between its samples and would quantize the bisection
            straight back to the fine lattice.
        centers: ``(3, M)`` voxel-centre coordinates of the batch.
        unit: ``(3, M)`` sphere-quadrature unit normal, used only to choose ``d``.
        dx: the cell size.

    Returns:
        A :class:`_PlaneFit`. ``accepted`` is False wherever any check above failed and
        the caller must keep the midpoint sums for that voxel.
    """
    count = centers.shape[1]
    dominant = xp.argmax(xp.abs(unit), axis=0)
    along = _axis_selector(dominant, xp)
    first = _axis_selector((dominant + 1) % 3, xp)
    second = _axis_selector((dominant + 2) % 3, xp)

    # (-p, -q, 1) is parallel to the normal expressed in (u, v, d), so the sphere
    # normal gives a first estimate of the slopes — used ONLY to size the search.
    normal_along = xp.sum(unit * along, axis=0)
    safe_along = xp.where(xp.abs(normal_along) > 0.0, normal_along, 1.0)
    slope_guess = (
        xp.abs(xp.sum(unit * first, axis=0)) + xp.abs(xp.sum(unit * second, axis=0))
    ) / xp.abs(safe_along)
    # A plane that cuts this voxel has |h(centre)| <= (1 + |p| + |q|)/2 cells, and a
    # probe a cells out adds a(|p| + |q|); 0.55 and 0.85 carry both plus room for the
    # sphere normal's own few-percent error in the slope estimate.
    reach = dx * (0.55 + 0.85 * xp.clip(slope_guess, 0.0, PLANE_SLOPE_CEILING))

    offsets = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    base = xp.concatenate(
        [
            centers + PLANE_PROBE * dx * (step_u * first + step_v * second)
            for step_u, step_v in offsets
        ],
        axis=1,
    )
    direction = xp.concatenate([along] * len(offsets), axis=1)
    heights, value_low, value_high, bracketed = _bisect_interface(
        source, base, direction, xp.concatenate([reach] * len(offsets)), xp
    )
    heights = heights.reshape(len(offsets), count)
    value_low = value_low.reshape(len(offsets), count)
    value_high = value_high.reshape(len(offsets), count)
    bracketed = bracketed.reshape(len(offsets), count)

    below = value_low[0]
    above = value_high[0]
    consistent = bracketed.all(axis=0)
    consistent = consistent & (value_low == below).all(axis=0)
    consistent = consistent & (value_high == above).all(axis=0)

    span = 2.0 * PLANE_PROBE * dx
    slope_u = (heights[1] - heights[2]) / span
    slope_v = (heights[3] - heights[4]) / span
    curvature = 0.25 * (heights[1] + heights[2] + heights[3] + heights[4]) - heights[0]
    residual = xp.abs(curvature) / dx
    accepted = (
        consistent
        & (xp.abs(slope_u) <= PLANE_SLOPE_CEILING)
        & (xp.abs(slope_v) <= PLANE_SLOPE_CEILING)
        & (residual <= PLANE_RESIDUAL_TOLERANCE)
    )

    level = (heights[0] + PLANE_CURVATURE_GAIN * curvature) / dx
    fill = cube_plane_fill(level, slope_u, slope_v, xp)
    # The height is measured along +d, so `below` fills the part of the voxel under the
    # plane. The normal is (-p, -q, 1) in (u, v, d) — exact for a plane — turned to face
    # the way epsilon increases, which is the orientation MEEP's gradient normal carries.
    raw = along - slope_u * first - slope_v * second
    length = xp.sqrt(xp.sum(raw * raw, axis=0))
    orientation = xp.where(above >= below, 1.0, -1.0)
    normal = raw * (orientation / length)
    return _PlaneFit(
        accepted=accepted,
        fill=fill,
        below=below,
        above=above,
        normal=normal,
        residual=residual,
    )


def _apply_planar_fill(source, block_centers, moments, unit, candidate, dx, rates, xp):
    """Replace the midpoint sums by the analytic planar fill wherever it is verifiable.

    Only voxels that would be SMOOTHED and hold at most two materials are candidates —
    a trivial-branch voxel is point-sampled by MEEP and stays that way, and a voxel with
    three materials has no single interface to fit. Candidates are gathered into a flat
    batch so the bisection runs over the interface alone rather than over the volume:
    that is O(N**(2/3)) of the grid, which is why the analytic path costs a fraction of
    the sampling it replaces.

    The last check lives here rather than in :func:`_fit_interface_plane` because it
    needs the fine lattice: the analytic fill must land within the midpoint sums' OWN
    resolution of the sampled fill. The sums locate a plane to ``0.5 * sum_j |n_j| / s_j``
    of a cell, so a disagreement past :data:`PLANE_CROSSCHECK_SLACK` times that is not a
    coarse answer next to a fine one — it is one of the two being wrong, and the sampled
    one is kept. The check goes vacuous, deliberately and visibly, on an axis given
    ``supersample = 1`` that the structure does vary along: there the sums have no
    resolution to compare against, and that configuration is exactly the one
    :meth:`FdtdDriver.set_epsilon_smoothed` already refuses.

    Returns ``(mean_epsilon, mean_inverse, unit, planar, worst_residual)`` — the first
    three with the accepted voxels replaced, ``planar`` the boolean volume of which
    voxels those were, and the worst planarity residual seen among them.
    """
    shape = moments.mean_epsilon.shape
    selected = candidate & moments.binary & ~moments.homogeneous
    flat = xp.nonzero(selected.reshape(-1))[0]
    if int(flat.size) == 0:
        return (moments.mean_epsilon, moments.mean_inverse, unit,
                xp.zeros(shape, dtype=bool), 0.0)

    plane_x = block_centers[0][flat // (shape[1] * shape[2])]
    plane_y = block_centers[1][(flat // shape[2]) % shape[1]]
    plane_z = block_centers[2][flat % shape[2]]
    centers = xp.stack([plane_x, plane_y, plane_z])
    rough = xp.stack([unit[a].reshape(-1)[flat] for a in range(3)])
    fit = _fit_interface_plane(source, centers, rough, dx, xp)

    low = moments.low.reshape(-1)[flat]
    high = moments.high.reshape(-1)[flat]
    pair_low = xp.minimum(fit.below, fit.above)
    pair_high = xp.maximum(fit.below, fit.above)
    # The plane's two materials must be the two the fine lattice found, or the bisection
    # walked out of this voxel and measured a different interface.
    accepted = fit.accepted & (pair_low == low) & (pair_high == high)

    mean = fit.fill * fit.below + (1.0 - fit.fill) * fit.above
    inverse = fit.fill / fit.below + (1.0 - fit.fill) / fit.above
    sampled_mean = moments.mean_epsilon.reshape(-1)[flat]
    contrast = fit.below - fit.above
    safe_contrast = xp.where(contrast != 0.0, contrast, 1.0)
    sampled_fill = (sampled_mean - fit.above) / safe_contrast
    tolerance = PLANE_CROSSCHECK_SLACK * sum(
        xp.abs(fit.normal[a]) / rates[a] for a in range(3)
    ) + 1e-9
    accepted = accepted & (xp.abs(fit.fill - sampled_fill) <= tolerance)

    def scatter(base, values):  # Write the accepted subset back into a block-shaped array.
        updated = xp.array(base, copy=True).reshape(-1)
        updated[flat] = xp.where(accepted, values, updated[flat])
        return updated.reshape(shape)

    planar = xp.zeros(shape, dtype=bool).reshape(-1)
    planar[flat] = accepted
    residual = fit.residual[accepted]
    return (
        scatter(moments.mean_epsilon, mean),
        scatter(moments.mean_inverse, inverse),
        [scatter(unit[a], fit.normal[a]) for a in range(3)],
        planar.reshape(shape),
        float(xp.max(residual)) if int(residual.size) else 0.0,
    )


def resolve_fill_rule(fill_rule: Any, epsilon: Any) -> str:
    """Turn the requested fill rule into the one that will run, or refuse.

    ``"auto"`` is not a silent default: it is the statement that the analytic path needs
    something BETWEEN the fine samples and only a callable epsilon has it. An array is
    piecewise constant on the fine lattice, so bisecting it would return a fine-cell
    edge and hand the quantization straight back; ``"auto"`` therefore runs ``"sampled"``
    for an array and ``"planar"`` for a callable, and every result says which
    (:attr:`SmoothedEpsilon.fill_rule`). Asking for ``"planar"`` with an array raises
    instead of quietly downgrading.
    """
    if fill_rule not in FILL_RULES:
        raise ValueError(
            f"fill_rule must be one of {FILL_RULES}, got {fill_rule!r}. 'planar' takes the "
            f"exact fill of the local interface plane, 'sampled' takes the midpoint sums over "
            f"the supersampled lattice, and 'auto' picks 'planar' for a callable epsilon and "
            f"'sampled' for an array."
        )
    if callable(epsilon):
        return "planar" if fill_rule == "auto" else fill_rule
    if fill_rule == "planar":
        raise ValueError(
            "fill_rule='planar' needs a callable epsilon(x, y, z). A supersampled ARRAY is "
            "piecewise constant on the fine lattice, so bisecting it would locate every "
            "interface on a fine-cell edge — the dx/(2s) quantization the analytic fill exists "
            "to remove. Pass the callable, or ask for fill_rule='sampled' explicitly."
        )
    return "sampled"


def smooth_inverse_epsilon(
    grid: Any,
    epsilon: Any,
    component: str,
    supersample: Any = DEFAULT_SUPERSAMPLE,
    fill_rule: str = DEFAULT_FILL_RULE,
) -> SmoothedEpsilon:
    """Smooth one E component's inverse permittivity the way MEEP's ``eps_averaging`` does.

    Args:
        grid: The run's :class:`~.grid.Grid`; supplies ``dx``, the cell counts, the
            axis origins, and the array module.
        epsilon: Either a vectorized callable ``epsilon(x, y, z)`` returning the
            permittivity at broadcast coordinate arrays (MEEP's ``material_function``
            with ``do_averaging = True``), or an array already sampled on
            :func:`fine_coordinates` for this component and ``supersample``. Only the
            callable form can carry the analytic fill (:func:`resolve_fill_rule`).
        component: Which E component's chi1inv row to produce, one of ``Ex``/``Ey``/``Ez``.
            The returned scalar is that row's diagonal entry.
        supersample: Midpoint samples per axis inside each voxel — a scalar or a
            per-axis triple. Under ``fill_rule="planar"`` this no longer sets the
            accuracy of a planar interface, which is analytic; it sets how FINELY a
            feature is seen at all (a voxel whose ``s**3`` samples are all equal is
            declared homogeneous and returned untouched), and it is the accuracy of the
            voxels the planar fit declines — corners, non-binary voxels, anything too
            curved. Under ``fill_rule="sampled"`` it is still the dominant error term,
            first order in ``1/s``: see :func:`_voxel_moments` for the measured ladder.
            1 on an axis the structure does not vary along is exact and free, so spell
            it as a triple — a y-invariant grating at ``(256, 1, 256)`` costs 4.8 s
            where the scalar ``64`` costs 16.7 s on the same 13.5k-voxel grid.
        fill_rule: ``"planar"`` takes the fill fraction and the normal from the local
            interface PLANE, measured by bisection and integrated in closed form
            (:func:`cube_plane_fill`), which removes the ``1/s`` quantization entirely
            wherever the interface is a plane — a grating, a slab, a photonic crystal.
            ``"sampled"`` is the midpoint-sum path alone. ``"auto"`` (the default) is
            ``"planar"`` for a callable epsilon and ``"sampled"`` for an array; see
            :func:`resolve_fill_rule`. The planar path never replaces a voxel it cannot
            verify, so every voxel it declines keeps the sampled answer.

    Returns:
        A :class:`SmoothedEpsilon` carrying the installable scalar, the full tensor row,
        and the diagnostics that say what the scalar reduction dropped.

    Raises:
        ValueError: for an unknown component or fill rule, a mirror-folded grid (the
            stored quadrant is not a period, so the fine lattice would not wrap), a
            supersample below 1, a mis-shaped array, ``fill_rule="planar"`` with an array
            epsilon, or an epsilon that is not everywhere finite and positive.
    """
    _require_component(component)
    if getattr(grid, "has_symmetry", lambda: False)():
        raise ValueError(
            f"Subpixel smoothing is not implemented on a mirror-folded grid (symmetry="
            f"{tuple(getattr(grid, 'symmetry', ()))}): the stored quadrant is half a period, so "
            f"the voxel means and the sphere the normal is taken from would wrap onto the wrong "
            f"material across the mirror plane. Run the full domain, or smooth with symmetry=()."
        )
    resolved_rule = resolve_fill_rule(fill_rule, epsilon)
    rates = normalize_supersample(supersample)
    xp = grid.xp
    source = _EpsilonSource(epsilon, grid, component, rates)
    centers = voxel_centers(grid, component)
    counts = (grid.nx, grid.ny, grid.nz)
    axis = E_COMPONENTS.index(component)

    row = [xp.zeros(counts, dtype=xp.float64) for _ in range(3)]
    normal_out = [xp.zeros(counts, dtype=xp.float64) for _ in range(3)]
    mean_epsilon = xp.zeros(counts, dtype=xp.float64)
    mean_inverse = xp.zeros(counts, dtype=xp.float64)
    smoothed_mask = xp.zeros(counts, dtype=bool)
    unresolved_mask = xp.zeros(counts, dtype=bool)  # Resolved structure, declined anyway.
    inhomogeneous_mask = xp.zeros(counts, dtype=bool)  # Fine lattice saw two materials.
    planar_mask = xp.zeros(counts, dtype=bool)  # Fill came from the analytic plane.
    point_sample = xp.zeros(counts, dtype=xp.float64)  # MEEP's trivial-branch eps.
    worst_residual = 0.0

    radius = grid.dx * SMOOTHING_DIAMETER  # volume::diameter() of the one-cell voxel.
    for index_ranges in _chunk_ranges(counts, rates):
        block_centers = tuple(
            xp.asarray(centers[a][index_ranges[a][0]:index_ranges[a][1]], dtype=xp.float64)
            for a in range(3)
        )
        fine_axes = tuple(
            xp.asarray(
                _fine_axis(grid, component, rates, a, index_ranges[a]), dtype=xp.float64
            )
            for a in range(3)
        )
        fine = source.fine_block(fine_axes, index_ranges)
        moments = _voxel_moments(fine, rates, xp)
        block_mean, block_inverse = moments.mean_epsilon, moments.mean_inverse
        homogeneous, center_value = moments.homogeneous, moments.center
        exact_center = source.center_values(block_centers)
        if exact_center is not None:
            # MEEP's trivial branch reads chi1p1 at v.center() itself. A constant
            # callable returns the same float there as on the fine lattice, so the
            # homogeneous bit-pin survives the substitution.
            center_value = exact_center
        gradient, uniform_prefix = _interface_normal(source, block_centers, radius, xp)

        magnitude = xp.sqrt(gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2)
        # anisotropic_averaging.cpp: the prefix shortcut and the 1e-8 gradient floor both
        # send the voxel to the point-sampled branch. Homogeneity is this engine's
        # addition and is what pins the bit-identical cases; it can only fire where the
        # average would have returned the same value anyway.
        trivial = uniform_prefix | (magnitude < GRADIENT_FLOOR) | homogeneous
        safe = xp.where(trivial, 1.0, magnitude)
        unit = [gradient[a] / safe for a in range(3)]
        unit = [xp.where(trivial, 0.0, unit[a]) for a in range(3)]

        planar = xp.zeros(block_mean.shape, dtype=bool)
        if resolved_rule == "planar":
            block_mean, block_inverse, unit, planar, residual = _apply_planar_fill(
                source, block_centers, moments, unit, ~trivial, grid.dx, rates, xp
            )
            worst_residual = max(worst_residual, residual)

        inverse_mean = 1.0 / xp.where(trivial, 1.0, block_mean)
        delta = xp.where(trivial, 0.0, block_inverse - inverse_mean)
        block_row = [unit[axis] * unit[a] * delta for a in range(3)]
        block_row[axis] = block_row[axis] + xp.where(trivial, 1.0 / center_value, inverse_mean)

        block_slice = tuple(slice(start, stop) for start, stop in index_ranges)
        for a in range(3):
            row[a][block_slice] = block_row[a]
            normal_out[a][block_slice] = unit[a]
        mean_epsilon[block_slice] = xp.where(trivial, center_value, block_mean)
        mean_inverse[block_slice] = xp.where(trivial, 1.0 / center_value, block_inverse)
        smoothed_mask[block_slice] = ~trivial
        planar_mask[block_slice] = planar
        # The one failure this scheme can hide. `homogeneous` sends a voxel to the
        # point-sampled branch only where the average would have returned the same
        # value, but the other two conditions do not: MEEP's 8-point prefix can find
        # every one of its samples equal while the voxel still straddles a corner (all
        # eight land in the surrounding material), and then a voxel holding a real
        # fraction of the other material comes back unsmoothed with nothing to say so.
        inhomogeneous_mask[block_slice] = ~homogeneous
        unresolved_mask[block_slice] = trivial & ~homogeneous
        point_sample[block_slice] = center_value

    # The float32 pair the driver installs. In an untouched voxel it must be exactly
    # what set_epsilon would have stored — float32 point sample and its float32
    # reciprocal, in that order — or "smoothing a uniform medium changes nothing" is
    # only true to a rounding. In a smoothed voxel the INVERSE is the physics
    # (stepping multiplies D by it), so it is the direct float32 cast of the float64
    # row entry, and epsilon is ITS float32 reciprocal for the energy guard and
    # readback.
    untouched = point_sample.astype(xp.float32)
    inverse = xp.where(
        smoothed_mask, row[axis].astype(xp.float32), 1.0 / untouched
    ).astype(xp.float32)
    effective = xp.where(smoothed_mask, 1.0 / inverse, untouched).astype(xp.float32)
    smoothed_count = int(xp.count_nonzero(smoothed_mask))
    total = counts[0] * counts[1] * counts[2]
    inhomogeneous_count = int(xp.count_nonzero(inhomogeneous_mask))
    unresolved_count = int(xp.count_nonzero(unresolved_mask))
    planar_count = int(xp.count_nonzero(planar_mask))
    diagonal_scale = float(xp.max(xp.abs(row[axis]))) or 1.0
    offdiagonal = max(
        float(xp.max(xp.abs(row[a]))) for a in range(3) if a != axis
    ) / diagonal_scale
    return SmoothedEpsilon(
        component=component,
        inverse_epsilon=xp.ascontiguousarray(inverse),
        epsilon=xp.ascontiguousarray(effective),
        row={AXIS_NAMES[a]: row[a] for a in range(3)},
        normal={AXIS_NAMES[a]: normal_out[a] for a in range(3)},
        mean_epsilon=mean_epsilon,
        mean_inverse_epsilon=mean_inverse,
        smoothed=smoothed_mask,
        interface_fraction=smoothed_count / total,
        unresolved=unresolved_mask,
        unresolved_fraction=(unresolved_count / inhomogeneous_count) if inhomogeneous_count else 0.0,
        offdiagonal_magnitude=offdiagonal,
        interface_resolution=grid.dx / (2.0 * min(rates)),
        fill_rule=resolved_rule,
        planar=planar_mask,
        planar_fraction=(planar_count / smoothed_count) if smoothed_count else 0.0,
        planarity_residual=worst_residual,
    )


def _fine_axis(grid, component, rates, axis, index_range):  # Fine coordinates of one chunk.
    start, stop = index_range
    rate = rates[axis]
    origin = grid.axis_origin(axis) + (0.5 * E_YEE_SHIFTS[component][axis] - 0.5) * grid.dx
    step = grid.dx / rate
    indices = numpy.arange(start * rate, stop * rate, dtype=numpy.float64)
    return origin + (indices + 0.5) * step


def epsilon_variation_along(
    grid: Any, epsilon: Any, component: str, supersample: Any = DEFAULT_SUPERSAMPLE
) -> float:
    """How much the sampled epsilon varies along ``component``'s own axis, relative.

    The single shared ``inv_eps`` array can carry one component's smoothed row exactly
    only where the structure is invariant along that component's own axis — that is the
    axis whose Yee position is half a cell from the array's integer registration, and
    the axis along which the other two D components would have to be zero for the
    dropped off-diagonals not to matter. 0.0 means the reduction is exact; anything
    else is the fraction of the material the shared scalar cannot represent, and
    :meth:`FdtdDriver.set_epsilon_smoothed` refuses on it rather than approximating.
    """
    _require_component(component)
    rates = normalize_supersample(supersample)
    xp = grid.xp
    source = _EpsilonSource(epsilon, grid, component, rates)
    axis = E_COMPONENTS.index(component)
    counts = (grid.nx, grid.ny, grid.nz)
    worst = 0.0
    scale = 0.0
    for index_ranges in _chunk_ranges(counts, rates):
        fine_axes = tuple(
            xp.asarray(_fine_axis(grid, component, rates, a, index_ranges[a]), dtype=xp.float64)
            for a in range(3)
        )
        fine = source.fine_block(fine_axes, index_ranges)
        reference = xp.take(fine, xp.asarray([0]), axis=axis)
        worst = max(worst, float(xp.max(xp.abs(fine - reference))))
        scale = max(scale, float(xp.max(xp.abs(fine))))
    return worst / scale if scale else 0.0
