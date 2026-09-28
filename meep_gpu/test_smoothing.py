"""
Value-level tests for :mod:`smoothing`, MEEP's subpixel epsilon averaging.

The module was wired into ``FdtdDriver.set_epsilon_smoothed`` before any test existed,
so this file has two jobs: check its arithmetic against hand-computed closed forms and
against CPU MEEP's own ``chi1inv``, and pin the four things about it that are true but
surprising, each of which would otherwise read as a success.

WHAT IT ACTUALLY IMPLEMENTS. MEEP's scheme, value for value:
``chi1inv_row[i] = n_d n_i (<1/eps> - 1/<eps>)`` plus ``1/<eps>`` on the diagonal
(``anisotropic_averaging.cpp:200-207``), the 50-point degree-11 octahedral sphere rule
for the normal at radius ``dx`` (``sphere-quad.cpp``, ``material_function::
normal_vector``), and the "first ``2**ndims`` samples agree, do not average" shortcut.
The harmonic mean of eps is what the field component normal to the interface sees and
the arithmetic mean is what the tangential ones see, and both branches are checked here
against hand values.

TWO FILL RULES, AND EVERY VALUE TEST RUNS ON BOTH. Where the two means come from is a
separate question from what is done with them. ``fill_rule="sampled"`` is MEEP's
material-function fallback — midpoint sums over an ``s``-per-axis lattice, first order
in ``1/s``. ``fill_rule="planar"`` (the default for a callable epsilon) is MEEP's
GEOMETRIC path reconstructed: the interface plane is measured by bisection and the cube
integral is closed form, so the fill stops depending on ``s`` at all. Tests that pin a
hand value run under both, exactly for the sums and to an ULP for the plane; tests that
pin a SAMPLING artifact say ``fill_rule="sampled"`` and have an analytic counterpart
next to them showing the artifact gone.

THE SURPRISES, all pinned below:

1. Through the driver only the ARITHMETIC branch ever runs. ``set_epsilon_smoothed``
   requires epsilon to be invariant along the smoothed component's own axis; the sphere
   rule is octahedrally symmetric and the plane fit's ``n_d`` is a central difference of
   two bit-identical bisections, so either way the normal along that axis is exactly
   zero, and the installed diagonal is ``1/<eps>`` with the normal contributing nothing.
   Correct physics for that configuration, but it means the anisotropy is verified and
   unused.
2. Under ``fill_rule="sampled"`` the supersample, not the scheme, sets the accuracy —
   first order in ``1/s``, and 16 leaves about as much error as staircasing did
   (``test_supersample_error_is_first_order_in_one_over_s``). Under the default it does
   not: the same ladder is flat at 1.4e-15
   (``test_planar_fill_is_independent_of_supersample``).
3. The sphere-quadrature normal of a discontinuous epsilon is off by a fixed amount that
   does NOT shrink with resolution (``test_oblique_plane_normal_artifact_is_dx_independent``);
   the fitted plane's normal is exact on the same structure
   (``test_planar_normal_removes_the_quadrature_artifact_entirely``).
4. The 8-point shortcut can leave a corner voxel unsmoothed while it holds 17 % of the
   other material (``test_corner_voxel_is_declined_and_reported``). That one is NOT
   fixed by the analytic fill — a corner is two planes, and the fit declines it too
   (``test_planar_path_declines_a_corner_and_keeps_the_sampled_answer``).
5. The analytic fill is exact for a plane and only second-order for a curved surface, so
   a cylinder is where it is approximate rather than exact
   (``test_planar_fill_of_a_cylinder_beats_the_sums_and_keeps_improving``).

MUTATIONS CHECKED. Each was applied to ``smoothing.py`` and the fast (non-MEEP)
selection re-run; the count is how many of the 99 failed. 26 of 27 are caught.

  49 swap the two means (harmonic <-> arithmetic)
  30 left-edge instead of midpoint fine samples, internal lattice
  29 put the Yee half-cell shift on the wrong axis
  26 always smooth row 0 instead of the component's own row
  24 skip the homogeneity bit-pin
  23 normalize the voxel means by the wrong count
  22 use a fixed ``n = x_hat`` instead of the gradient
  16 measure voxel centres from ``-L/2`` instead of ``grid.axis_origin``
  15 average eps instead of 1/eps for the normal direction
  13 drop the projection and install ``1/<eps>`` unconditionally
  11 fill the voxel with the ABOVE material instead of the below one (sign of the fill)
  10 drop the curvature correction from the plane's level
   9 sphere radius ``2 dx`` instead of the voxel diameter
   9 take the fitted normal as ``(p, q, 1)`` instead of ``(-p, -q, 1)``
   8 orient the fitted normal down-gradient instead of up-gradient
   7 halve PLANE_BISECTION_STEPS to 6, so the interface is located to 3e-2 of a cell
   6 use ``h(centre)`` in place of the four-probe mean in the planarity residual
   5 drop the periodic wrap in ``_wrap_axis``
   5 skip the two-material consistency check across the five columns
   4 accept every plane fit regardless of the residual
   3 drop the 1e-8 gradient floor
   3 left-edge instead of midpoint in ``fine_coordinates``
   2 drop the binary check, so a three-material voxel is fitted as two
   2 drop MEEP's 8-point uniform-prefix shortcut
   2 skip ``sort_by_distance``, so the wrong 8 nodes gate that shortcut
   1 forget MEEP's factor of R on the gradient
   0 SURVIVES: build the float32 inverse as ``1 / float32(1 / row)`` instead of
     ``float32(row)``. Not a gap — that reordering differs in the last bit for 29 % of
     arbitrary float64 values but for none this module produces, because the diagonal
     is always ``1/<eps>`` with ``<eps>`` a short-denominator rational. There is no
     input on which the two spellings disagree, so no test can separate them.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from .dispersion import LORENTZIAN, Susceptibility, component_coordinates
from .driver import FdtdDriver
from .grid import Grid
from . import smoothing as sm

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)

E1, E2 = 1.0, 4.0  # The pair every hand value below is computed from.
# Both fill rules, and how exactly each reproduces a hand-computed mean. The midpoint
# sums are a sum of the same rationals the hand value is built from, so they land on it
# bit for bit; the analytic plane reaches it through a bisection and a cubic, so it
# lands 1-2 ULP away. Anything looser than 1e-14 would stop separating the two.
FILL_RULES = [("sampled", 0.0), ("planar", 1e-14)]


# --- helpers ----------------------------------------------------------------------


def _equals(value, expected, tolerance):
    """Exact equality when ``tolerance`` is 0, otherwise a relative comparison."""
    if tolerance == 0.0:
        assert value == expected, f"{value!r} != {expected!r}"
    else:
        assert value == pytest.approx(expected, rel=tolerance)


def _step(axis: int, position: float, low: float = E1, high: float = E2):
    """A single planar interface normal to ``axis``: ``low`` below ``position``."""

    def epsilon(x, y, z):
        coordinate = (x, y, z)[axis]
        others = sum(value for index, value in enumerate((x, y, z)) if index != axis)
        return np.where(coordinate < position, low, high) + 0.0 * others

    return epsilon


def _bar(x_span, z_span, inside=E2, outside=E1):
    """A y-invariant rectangular bar — the grating cross-section, corners and all."""

    def epsilon(x, y, z):
        within = (
            (x >= x_span[0]) & (x <= x_span[1]) & (z >= z_span[0]) & (z <= z_span[1])
        )
        return np.where(within, inside, outside) + 0.0 * y

    return epsilon


def _voxel_span(grid, component, axis, index):  # Physical [min, max] of one voxel edge to edge.
    shift = sm.E_YEE_SHIFTS[component][axis]
    centre = grid.axis_origin(axis) + (index + 0.5 * shift) * grid.dx
    return centre - 0.5 * grid.dx, centre + 0.5 * grid.dx


def _mixture(fill_low):  # (<eps>, <1/eps>) for a voxel `fill_low` filled with E1.
    return (
        fill_low * E1 + (1.0 - fill_low) * E2,
        fill_low / E1 + (1.0 - fill_low) / E2,
    )


def _brute_force_means(epsilon, grid, component, index, rates):
    """``(<eps>, <1/eps>)`` for one voxel, re-derived with an obvious triple loop.

    Deliberately shares no code with :func:`smoothing._voxel_moments` — no reshape, no
    chunking, no vectorized wrap — so it catches a defect in that machinery rather than
    reproducing it. Coordinates are folded by hand the way ``_wrap_axis`` folds them.
    """
    lengths = [(grid.nx, grid.ny, grid.nz)[a] * grid.dx for a in range(3)]
    lows = [_voxel_span(grid, component, a, index[a])[0] for a in range(3)]
    total_eps = 0.0
    total_inverse = 0.0
    for i in range(rates[0]):
        for j in range(rates[1]):
            for k in range(rates[2]):
                point = []
                for axis, step in enumerate((i, j, k)):
                    raw = lows[axis] + (step + 0.5) * grid.dx / rates[axis]
                    origin = grid.axis_origin(axis)
                    point.append(origin + (raw - origin) % lengths[axis])
                value = float(
                    epsilon(
                        np.array([point[0]]), np.array([point[1]]), np.array([point[2]])
                    )[0]
                )
                total_eps += value
                total_inverse += 1.0 / value
    count = rates[0] * rates[1] * rates[2]
    return total_eps / count, total_inverse / count


def _staircased(grid, epsilon):
    """The array ``set_epsilon`` would be handed: point samples at INTEGER Yee positions."""
    axes = [grid.axis_origin(a) + np.arange((grid.nx, grid.ny, grid.nz)[a]) * grid.dx
            for a in range(3)]
    values = epsilon(axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :])
    return np.broadcast_to(values, grid.shape).astype(np.float32)


# --- 1. The quadrature rule ------------------------------------------------------


def test_sphere_quadrature_is_meeps_50_point_degree_11_rule():
    """``sphere_quadrature`` must be ``sphere_quad[2]`` of MEEP's generated header.

    Degree 11 is the property the whole scheme rests on: it makes the gradient of a
    locally planar interface second-order accurate, and octahedral symmetry makes an
    axis-aligned interface produce an exactly axis-aligned normal.
    """
    points, weights = sm.sphere_quadrature()
    assert points.shape == (sm.NQUAD3, 3)
    assert weights.shape == (sm.NQUAD3,)
    # Every node is on the unit sphere to the last bit, and the weights are a partition.
    assert np.max(np.abs(np.linalg.norm(points, axis=1) - 1.0)) == 0.0
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.all(weights > 0.0)
    # MEEP's four orbits and their integer weights over 725760 (sphere-quad.cpp).
    expected = {9216 / 725760.0, 16384 / 725760.0, 15309 / 725760.0, 14641 / 725760.0}
    assert {float(w) for w in np.unique(weights)} == expected
    assert sorted(np.bincount(np.unique(weights, return_inverse=True)[1]).tolist()) == [6, 8, 12, 24]
    # Exact through degree 11 against the closed forms for the unit sphere surface.
    def integrate(function):
        return float(np.sum(weights * function(points[:, 0], points[:, 1], points[:, 2])))

    assert integrate(lambda a, b, c: np.ones_like(a)) == pytest.approx(1.0, abs=1e-15)
    assert integrate(lambda a, b, c: a) == pytest.approx(0.0, abs=1e-15)
    assert integrate(lambda a, b, c: a * a) == pytest.approx(1 / 3, abs=1e-15)
    assert integrate(lambda a, b, c: a ** 4) == pytest.approx(1 / 5, abs=1e-15)
    assert integrate(lambda a, b, c: a * a * b * b) == pytest.approx(1 / 15, abs=1e-15)
    assert integrate(lambda a, b, c: a ** 6) == pytest.approx(1 / 7, abs=1e-15)
    assert integrate(lambda a, b, c: a ** 4 * b * b) == pytest.approx(1 / 35, abs=1e-15)
    assert integrate(lambda a, b, c: a * a * b * b * c * c) == pytest.approx(1 / 105, abs=1e-15)
    # Degree 12 is NOT exact — the rule really is 11th order and not something better.
    assert integrate(lambda a, b, c: a ** 12) != pytest.approx(1 / 13, abs=1e-6)


def test_sphere_quadrature_prefix_is_the_shortcut_meep_reads():
    """The first ``2**3`` nodes decide whether a voxel is averaged at all.

    ``material_function::normal_vector`` compares consecutive epsilon values over this
    prefix only, so which nodes land there is physics, not bookkeeping — MEEP's
    ``sort_by_distance`` puts the six axis directions first and then a body diagonal
    pair. A rule sorted any other way changes which voxels get smoothed.
    """
    points, _ = sm.sphere_quadrature()
    prefix = points[: sm._UNIFORM_PREFIX]
    assert sm._UNIFORM_PREFIX == 8
    axis_nodes = {tuple(row) for row in prefix[:6]}
    assert axis_nodes == {
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    }
    diagonal = 1.0 / math.sqrt(3.0)
    assert {tuple(np.round(row, 12)) for row in prefix[6:8]} == {
        tuple(np.round([diagonal] * 3, 12)), tuple(np.round([-diagonal] * 3, 12))
    }


def test_sphere_quadrature_is_octahedrally_symmetric():
    """Closed under sign flips and coordinate rotation, with equal weights.

    This is precisely why an interface normal to x comes back as exactly (1, 0, 0):
    the y and z sums cancel node against node rather than nearly cancelling. It is also
    why the gradient along an axis epsilon does not vary along vanishes EXACTLY, which
    is the property ``set_epsilon_smoothed``'s whole contract rests on.
    """
    points, weights = sm.sphere_quadrature()
    table = {tuple(np.round(row, 12)): float(w) for row, w in zip(points, weights)}
    assert len(table) == sm.NQUAD3
    for row, weight in zip(points, weights):
        for signs in ((1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, -1)):
            flipped = tuple(np.round(row * np.array(signs), 12))
            assert table[flipped] == float(weight), f"{row} is not sign-symmetric"
        rotated = tuple(np.round(np.roll(row, 1), 12))
        assert table[rotated] == float(weight), f"{row} is not rotation-symmetric"


# --- 2a. Analytic known values: the two means, per axis ---------------------------


@pytest.mark.parametrize("fill_rule,tolerance", FILL_RULES)
@pytest.mark.parametrize(
    "component,axis",
    [("Ex", 0), ("Ey", 1), ("Ez", 2)],
)
def test_interface_normal_to_the_components_own_axis_gives_the_harmonic_mean(
    component, axis, fill_rule, tolerance
):
    """``n`` along the smoothed component's own axis: the diagonal must be ``<1/eps>``.

    This is the branch that carries the physics of a field component NORMAL to an
    interface — D is continuous there, E jumps, and the effective permittivity is the
    harmonic mean, whose inverse is the arithmetic mean of ``1/eps``. Hand value with
    the interface 13/16 of the way across a voxel of E1 = 1 into E2 = 4::

        <1/eps> = 13/16 / 1 + 3/16 / 4 = 0.859375

    The interface is placed on a fine-cell boundary so the MIDPOINT fill is exactly
    13/16 too: both rules must land on the same hand value, the sampled one bit for bit
    and the analytic one within an ULP or two of the bisection and the cubic.
    """
    supersample = 16
    grid = Grid(resolution=20, cell_size=(0.6, 0.6, 0.6))
    index = [4, 4, 4]
    low, _ = _voxel_span(grid, component, axis, index[axis])
    position = low + 13 * grid.dx / supersample
    rates = tuple(supersample if a == axis else 1 for a in range(3))

    result = sm.smooth_inverse_epsilon(
        grid, _step(axis, position), component, rates, fill_rule=fill_rule
    )
    cell = tuple(index)
    mean_epsilon, mean_inverse = _mixture(13 / 16)

    assert bool(np.asarray(result.smoothed)[cell]), "the interface voxel was not averaged"
    assert result.fill_rule == fill_rule
    assert bool(np.asarray(result.planar)[cell]) == (fill_rule == "planar")
    _equals(float(np.asarray(result.mean_epsilon)[cell]), mean_epsilon, tolerance)
    _equals(float(np.asarray(result.mean_inverse_epsilon)[cell]), mean_inverse, tolerance)
    # The normal points along the axis, exactly, and the diagonal is the harmonic mean.
    normal = [float(np.asarray(result.normal[name])[cell]) for name in "xyz"]
    assert abs(abs(normal[axis]) - 1.0) < 1e-15
    assert max(abs(normal[a]) for a in range(3) if a != axis) < 1e-15
    diagonal = float(np.asarray(result.row["xyz"[axis]])[cell])
    assert diagonal == pytest.approx(mean_inverse, rel=1e-14)
    assert diagonal == pytest.approx(0.859375, rel=1e-14)
    # And it is NOT the arithmetic mean — 0.859375 against 0.64, so a swap shows.
    assert 1.0 / mean_epsilon == pytest.approx(0.64, rel=1e-14)
    assert abs(diagonal - 1.0 / mean_epsilon) / diagonal > 0.2


@pytest.mark.parametrize("fill_rule,tolerance", FILL_RULES)
@pytest.mark.parametrize(
    "component,normal_axis",
    [("Ex", 1), ("Ex", 2), ("Ey", 0), ("Ey", 2), ("Ez", 0), ("Ez", 1)],
)
def test_interface_tangential_to_the_component_gives_the_arithmetic_mean(
    component, normal_axis, fill_rule, tolerance
):
    """``n`` perpendicular to the component: the diagonal must be ``1/<eps>``.

    E tangential to an interface is continuous, so the effective permittivity is the
    plain volume average and chi1inv is its reciprocal. Hand value with the interface
    5/16 across::

        1/<eps> = 1 / (5/16 * 1 + 11/16 * 4) = 1/3.0625 = 0.32653061224489793

    Every configuration ``FdtdDriver.set_epsilon_smoothed`` accepts lands in this
    branch, so this is the value the engine actually installs.
    """
    supersample = 16
    grid = Grid(resolution=20, cell_size=(0.6, 0.6, 0.6))
    index = [4, 4, 4]
    low, _ = _voxel_span(grid, component, normal_axis, index[normal_axis])
    position = low + 5 * grid.dx / supersample
    rates = tuple(supersample if a == normal_axis else 1 for a in range(3))

    result = sm.smooth_inverse_epsilon(
        grid, _step(normal_axis, position), component, rates, fill_rule=fill_rule
    )
    cell = tuple(index)
    mean_epsilon, mean_inverse = _mixture(5 / 16)
    own = sm.E_COMPONENTS.index(component)

    assert bool(np.asarray(result.smoothed)[cell])
    _equals(float(np.asarray(result.mean_epsilon)[cell]), mean_epsilon, tolerance)
    diagonal = float(np.asarray(result.row["xyz"[own]])[cell])
    assert diagonal == pytest.approx(1.0 / mean_epsilon, rel=1e-14)
    assert diagonal == pytest.approx(0.32653061224489793, rel=1e-14)
    # The normal has no component along the smoothed axis, so the row is diagonal.
    assert abs(float(np.asarray(result.normal["xyz"[own]])[cell])) < 1e-15
    for a in range(3):
        if a != own:
            assert abs(float(np.asarray(result.row["xyz"[a]])[cell])) < 1e-15
    # And it is NOT the harmonic mean.
    assert abs(diagonal - mean_inverse) / diagonal > 0.3


@pytest.mark.parametrize("fill_rule", ["sampled", "planar"])
def test_oblique_interface_matches_the_projection_formula_and_a_brute_force_mixture(fill_rule):
    """An oblique plane: MEEP's full projection, against an independent re-derivation.

    The structure is a lamellar stack tilted 2:1 in x-z whose phase advances by an
    integer over each cell edge, so it is genuinely cell-periodic and the wrap adds no
    second interface. For every averaged voxel the row must equal MEEP's
    ``n_d n_i (<1/eps> - 1/<eps>) + delta_id / <eps>`` built from the returned normal
    and means, and the means themselves must equal an obvious triple-loop sum that
    shares no code with the module.

    Both fill rules face that sum, and they agree with it to 6e-15 — which is a
    property of THIS structure, not a general one: the tilt is chosen so the interface
    lands on fine-cell boundaries at ``s`` = 24, where the midpoint sum happens to be
    exact. ``test_planar_fill_beats_the_sampled_fill_off_the_fine_lattice`` is the case
    where they part company.
    """
    supersample = 24
    grid = Grid(resolution=20, cell_size=(1.0, 0.1, 1.0))
    exact_normal = np.array([2.0, 0.0, 1.0]) / math.sqrt(5.0)

    def epsilon(x, y, z):
        return np.where((2.0 * x + z) % 1.0 < 0.5, E1, E2) + 0.0 * y

    rates = (supersample, 1, supersample)
    result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", rates, fill_rule=fill_rule)
    mask = np.asarray(result.smoothed)
    assert mask.sum() > 100, "the tilted stack must cross many voxels"

    means = np.asarray(result.mean_epsilon)
    inverses = np.asarray(result.mean_inverse_epsilon)
    normals = [np.asarray(result.normal[name]) for name in "xyz"]
    rows = [np.asarray(result.row[name]) for name in "xyz"]
    delta = inverses - 1.0 / means
    for a in range(3):
        expected = normals[1] * normals[a] * delta + (1.0 / means if a == 1 else 0.0)
        np.testing.assert_allclose(rows[a][mask], expected[mask], rtol=1e-13, atol=0.0)

    # The two means, re-derived by hand on a sample of the averaged voxels.
    for index in np.argwhere(mask)[:: max(1, mask.sum() // 8)]:
        reference = _brute_force_means(epsilon, grid, "Ey", tuple(index), rates)
        assert means[tuple(index)] == pytest.approx(reference[0], rel=1e-13)
        assert inverses[tuple(index)] == pytest.approx(reference[1], rel=1e-13)

    # The tilt is recovered, the normal stays in the plane of variation exactly, and
    # the diagonal is the arithmetic mean because n_y is zero. The sphere quadrature
    # reads the tilt to a few percent; the fitted plane reads it to the last bits.
    bound = 0.04 if fill_rule == "sampled" else 1e-13
    for index in np.argwhere(mask):
        cell = tuple(index)
        found = np.array([normals[a][cell] for a in range(3)])
        sign = 1.0 if found @ exact_normal > 0 else -1.0
        assert np.linalg.norm(sign * found - exact_normal) < bound
        assert abs(found[1]) < 1e-15
    np.testing.assert_allclose(rows[1][mask], (1.0 / means)[mask], rtol=1e-14, atol=0.0)


def test_installed_scalar_is_the_row_diagonal_at_the_components_yee_positions():
    """``inverse_epsilon`` is the float32 diagonal, and ``epsilon`` its reciprocal.

    The stepping kernels multiply D by ``inv_eps``, so the INVERSE is the physics and
    ``epsilon`` exists for readback and the energy guard. In an averaged voxel the
    inverse must be the DIRECT float32 cast of the float64 row entry, and in an
    untouched one it must be exactly what ``set_epsilon`` would have stored. An
    off-centre cylinder is included alongside the bar so that the equalities are
    checked over 64 distinct fill fractions rather than a handful.

    Not pinned, because it is not observable: reordering this into
    ``float32(1 / float32(1 / row))`` differs in the last bit for 29 % of arbitrary
    float64 values, but never for the values this module produces — the diagonal is
    always ``1 / <eps>`` with ``<eps>`` a short-denominator rational, whose float32
    reciprocal round-trips exactly. That reordering is the one mutation of this module
    the suite cannot catch, and it is a no-op rather than a gap.
    """
    grid = Grid(resolution=20, cell_size=(1.0, 0.2, 1.0))

    def cylinder(x, y, z):
        # A curved, off-centre surface: no two interface voxels share a fill fraction,
        # where a cylinder centred on the grid would collapse to 7 by symmetry.
        return np.where((x - 0.0713) ** 2 + (z + 0.0431) ** 2 < 0.4217 ** 2, E2, E1) + 0.0 * y

    for epsilon in (_bar((-0.155, 0.315), (-0.3, 0.3)), cylinder):
        result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (32, 1, 32))
        mask = np.asarray(result.smoothed)
        inverse = np.asarray(result.inverse_epsilon)
        effective = np.asarray(result.epsilon)

        assert inverse.dtype == np.float32 and effective.dtype == np.float32
        assert inverse.shape == grid.shape
        diagonal = np.asarray(result.row["y"]).astype(np.float32)
        np.testing.assert_array_equal(inverse[mask], diagonal[mask])
        np.testing.assert_array_equal(effective[mask], (np.float32(1.0) / inverse[mask]))
        # Untouched voxels carry the point sample and ITS reciprocal, bit for bit.
        point = np.asarray(result.mean_epsilon).astype(np.float32)
        np.testing.assert_array_equal(effective[~mask], point[~mask])
        np.testing.assert_array_equal(inverse[~mask], (np.float32(1.0) / point[~mask]))
    # The cylinder really does span many distinct values, or the equalities are luck.
    assert len(np.unique(np.asarray(result.row["y"])[mask])) > 40


# --- 2b. The interface normal ----------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_normal_interface_gives_an_exactly_axis_aligned_normal(axis):
    """Octahedral symmetry, made a value: the off-axis components are below 1e-15."""
    grid = Grid(resolution=20, cell_size=(0.5, 0.5, 0.5))
    low, _ = _voxel_span(grid, "Ez", axis, 3)
    result = sm.smooth_inverse_epsilon(
        grid, _step(axis, low + 0.37 * grid.dx), "Ez",
        tuple(16 if a == axis else 1 for a in range(3)),
    )
    mask = np.asarray(result.smoothed)
    assert mask.any()
    for other in range(3):
        if other == axis:
            continue
        values = np.asarray(result.normal["xyz"[other]])[mask]
        assert np.max(np.abs(values)) < 1e-15, f"normal leaked into axis {other}"
    along = np.abs(np.asarray(result.normal["xyz"[axis]])[mask])
    np.testing.assert_allclose(along, 1.0, rtol=1e-14)


def test_gradient_vanishes_exactly_along_an_axis_epsilon_does_not_vary_along():
    """The property the driver's whole contract rests on, checked as an equality.

    ``set_epsilon_smoothed`` reduces MEEP's tensor to one scalar and is only exact
    because the dropped off-diagonals are identically zero. That holds because the
    sphere rule pairs every node with its reflection in the invariant axis, so the
    gradient there cancels node against node. If it were merely small the reduction
    would be an approximation and the driver's refusal would be theatre.
    """
    for component, structures in (
        ("Ey", [("xz", lambda x, y, z: np.where((2 * x + z) % 1.0 < 0.5, E1, E2) + 0.0 * y)]),
        ("Ez", [("xy", lambda x, y, z: np.where((x + 3 * y) % 1.0 < 0.4, E1, E2) + 0.0 * z)]),
        ("Ex", [("yz", lambda x, y, z: np.where((y - 2 * z) % 1.0 < 0.6, E1, E2) + 0.0 * x)]),
    ):
        own = sm.E_COMPONENTS.index(component)
        grid = Grid(resolution=20, cell_size=(1.0, 1.0, 1.0))
        rates = tuple(1 if a == own else 16 for a in range(3))
        for label, epsilon in structures:
            result = sm.smooth_inverse_epsilon(grid, epsilon, component, rates)
            mask = np.asarray(result.smoothed)
            assert mask.any(), f"{component}/{label} produced no averaged voxel"
            leak = np.max(np.abs(np.asarray(result.normal["xyz"[own]])[mask]))
            assert leak < 1e-15, f"{component}/{label}: n_{'xyz'[own]} = {leak:.3e}"
            assert result.offdiagonal_magnitude < 1e-15
            # And so the installed value is the arithmetic mean, not a mixture.
            means = np.asarray(result.mean_epsilon)
            np.testing.assert_allclose(
                np.asarray(result.row["xyz"[own]])[mask], (1.0 / means)[mask],
                rtol=1e-14, atol=0.0,
            )


def test_oblique_plane_normal_artifact_is_dx_independent():
    """The sphere-quadrature normal of a STEP function has a fixed error, not a small one.

    A 50-point polynomial rule cannot integrate a discontinuity exactly, so the
    recovered direction of a tilted plane is off by an amount set by the tilt alone.
    Measured on a 2:1 lamellar stack: worst voxel 0.0395, mean 0.0276 — and identical
    at four resolutions, which is what proves it is the rule and not the discretization.
    MEEP carries the same artifact for a user material function and escapes it for a
    geometric object by taking the normal analytically; ``fill_rule="planar"`` escapes
    it the same way (next test), so this pins what the SAMPLED path is left with.
    """
    exact = np.array([2.0, 0.0, 1.0]) / math.sqrt(5.0)

    def epsilon(x, y, z):
        return np.where((2.0 * x + z) % 1.0 < 0.5, E1, E2) + 0.0 * y

    signature = []
    for resolution in (10, 20, 40, 80):
        grid = Grid(resolution=resolution, cell_size=(1.0, 0.1, 1.0))
        result = sm.smooth_inverse_epsilon(
            grid, epsilon, "Ey", (24, 1, 24), fill_rule="sampled"
        )
        mask = np.asarray(result.smoothed)
        found = np.stack([np.asarray(result.normal[name])[mask] for name in "xyz"], axis=1)
        signs = np.where(found @ exact > 0, 1.0, -1.0)[:, None]
        errors = np.linalg.norm(signs * found - exact, axis=1)
        signature.append((float(errors.max()), float(errors.mean())))
        # The interface length in voxels grows with the resolution, so the population
        # grows; the ERROR is what must not.
        assert mask.sum() == 6 * resolution * grid.ny, "unexpected interface voxel count"

    worst, average = zip(*signature)
    assert max(worst) == pytest.approx(0.03947, abs=2e-5)
    assert max(average) == pytest.approx(0.02757, abs=2e-5)
    # Identical at every resolution: a quadrature artifact, not a convergence error.
    assert max(worst) - min(worst) < 1e-12
    assert max(average) - min(average) < 1e-12


def test_planar_normal_removes_the_quadrature_artifact_entirely():
    """The fitted plane's normal is EXACT where the sphere rule is 4e-2 out.

    MEEP's geometric path takes the normal from the object; the planar path takes it
    from the two central differences of the bisected interface height, which is the same
    closed form for anything locally planar. The 2:1 stack of the previous test then
    reads back at the last bits instead of 0.0395 out — four orders, on the same
    structure, at the same resolutions, from the same call with one argument changed.

    This does not move what ``FdtdDriver.set_epsilon_smoothed`` installs (its invariance
    forces ``n_d = 0``, and the diagonal is then ``1/<eps>`` whatever the normal is). It
    moves the off-diagonals and the harmonic branch, which is what a per-component
    tensor consumer would need.
    """
    exact = np.array([2.0, 0.0, 1.0]) / math.sqrt(5.0)

    def epsilon(x, y, z):
        return np.where((2.0 * x + z) % 1.0 < 0.5, E1, E2) + 0.0 * y

    for resolution in (10, 20, 40, 80):
        grid = Grid(resolution=resolution, cell_size=(1.0, 0.1, 1.0))
        result = sm.smooth_inverse_epsilon(
            grid, epsilon, "Ey", (24, 1, 24), fill_rule="planar"
        )
        mask = np.asarray(result.smoothed)
        assert result.planar_fraction == 1.0, "a plain tilted plane must all be analytic"
        found = np.stack([np.asarray(result.normal[name])[mask] for name in "xyz"], axis=1)
        signs = np.where(found @ exact > 0, 1.0, -1.0)[:, None]
        errors = np.linalg.norm(signs * found - exact, axis=1)
        assert float(errors.max()) < 1e-13, (
            f"resolution {resolution}: worst normal error {errors.max():.3e}"
        )
        # A plane has no curvature, so the planarity diagnostic must read as zero too.
        assert result.planarity_residual < 1e-13


def test_gradient_carries_meeps_factor_of_the_voxel_diameter():
    """``gradient = R * sum_i w_i eps(p + R u_i) u_i`` with ``R = dx``, re-derived by hand.

    MEEP keeps the factor of ``R`` (``normal_vector`` accumulates ``(pt - p) * ...``,
    and ``pt - p`` is ``R u_i``) and compares the result against an ABSOLUTE 1e-8
    floor, so the factor is not cosmetic: drop it and the floor fires at a gradient
    ``1/dx`` times smaller. It cannot be seen in the unit normal — any positive scale
    normalizes away — so it is pinned here, on the private function, against a plain
    sum over the quadrature table. ``R`` is ``volume::diameter()`` of
    ``gv.dV(here, 1.0)``, a cube of side ``dx``, hence ``dx`` and not ``sqrt(3) dx``.
    """
    grid = Grid(resolution=20, cell_size=(0.5, 0.5, 0.5))
    rates = (1, 1, 16)
    low = _voxel_span(grid, "Ey", 2, 4)[0]
    epsilon = _step(2, low + 0.4 * grid.dx)
    source = sm._EpsilonSource(epsilon, grid, "Ey", rates)
    centres = sm.voxel_centers(grid, "Ey")
    radius = grid.dx * sm.SMOOTHING_DIAMETER
    assert radius == grid.dx

    gradient, _ = sm._interface_normal(source, centres, radius, np)
    points, weights = sm.sphere_quadrature()
    cell = (2, 3, 4)
    centre = [centres[a][cell[a]] for a in range(3)]
    expected = np.zeros(3)
    for node, weight in zip(points, weights):
        probe = [centre[a] + radius * node[a] for a in range(3)]
        value = float(epsilon(np.array([probe[0]]), np.array([probe[1]]),
                              np.array([probe[2]]))[0])
        expected += weight * value * np.array(node)
    expected *= radius
    for axis in range(3):
        assert float(np.asarray(gradient[axis])[cell]) == pytest.approx(
            expected[axis], rel=1e-13, abs=1e-18)
    # The z gradient must be O(R) in size, which is what the floor comparison scales
    # against: dropping the factor would make it 20x larger at this resolution.
    magnitude = float(np.linalg.norm(expected))
    assert magnitude == pytest.approx(radius * abs(expected[2]) / radius, rel=1.0)
    assert 0.5 * radius < magnitude < 5.0 * radius


def test_gradient_floor_catches_a_symmetric_feature_the_prefix_shortcut_misses():
    """The 1e-8 floor is reachable, and without it the normal is 0/0.

    A slab thinner than the sampling sphere, centred exactly on a voxel, defeats both
    of the other trivial tests: the voxel is NOT internally uniform, and the first
    eight sphere nodes do NOT all agree (the four in-plane ones sit in the slab, the
    two axial ones outside it). But the gradient cancels EXACTLY by the reflection
    symmetry of the octahedral rule, so only the floor stands between the scheme and
    ``0 / 0``. MEEP has the same guard at ``anisotropic_averaging.cpp:101``.
    """
    grid = Grid(resolution=20, cell_size=(0.5, 0.5, 0.5))
    component, axis, index = "Ey", 2, 5
    centre = sm.voxel_centers(grid, component)[axis][index]
    thickness = 0.5 * grid.dx  # Thin enough that the +-dx sphere nodes fall outside.

    def epsilon(x, y, z):
        return np.where(np.abs(z - centre) < thickness / 2.0, E2, E1) + 0.0 * (x + y)

    result = sm.smooth_inverse_epsilon(grid, epsilon, component, (1, 1, 32))
    cell = (0, 0, index)
    # Every value must be finite — the failure mode this guards is NaN, not an error.
    assert np.all(np.isfinite(np.asarray(result.inverse_epsilon)))
    assert np.all(np.isfinite(np.asarray(result.epsilon)))
    for name in "xyz":
        assert np.all(np.isfinite(np.asarray(result.row[name])))

    # The voxel took the trivial branch, and neither of the other two conditions is why.
    assert not bool(np.asarray(result.smoothed)[cell])
    source = sm._EpsilonSource(epsilon, grid, component, (1, 1, 32))
    gradient, uniform = sm._interface_normal(
        source, sm.voxel_centers(grid, component), grid.dx, np)
    assert not bool(np.asarray(uniform)[cell]), "the 8-node prefix must NOT be uniform here"
    magnitude = float(np.sqrt(sum(float(np.asarray(gradient[a])[cell]) ** 2 for a in range(3))))
    assert magnitude < sm.GRADIENT_FLOOR, f"gradient {magnitude:.3e} does not reach the floor"
    # And the voxel is genuinely inhomogeneous, so the floor is doing real work: this is
    # reported as unresolved rather than passing for a uniform cell.
    assert bool(np.asarray(result.unresolved)[cell])
    assert result.unresolved_fraction > 0.0


def test_sphere_interface_normal_points_radially():
    """A curved interface: the normal follows the radius, at MEEP's fallback accuracy.

    Epsilon is HIGH inside, so the gradient points inward and the unit normal is
    ``-r_hat``. The error is bounded by the same rule artifact as the plane and does
    not converge — measured max 0.22, mean 0.08 — so this pins direction recovery, not
    an order.
    """
    radius = 0.3

    def epsilon(x, y, z):
        return np.where(x * x + y * y + z * z < radius * radius, E2, E1)

    for resolution, bound in ((10, 0.13), (20, 0.18), (40, 0.22)):
        grid = Grid(resolution=resolution, cell_size=(1.0, 1.0, 1.0))
        result = sm.smooth_inverse_epsilon(grid, epsilon, "Ez", 6)
        centres = sm.voxel_centers(grid, "Ez")
        mask = np.asarray(result.smoothed)
        assert mask.sum() > 100
        indices = np.argwhere(mask)
        positions = np.stack(
            [centres[a][indices[:, a]] for a in range(3)], axis=1
        )
        exact = -positions / np.linalg.norm(positions, axis=1, keepdims=True)
        found = np.stack(
            [np.asarray(result.normal[name])[mask] for name in "xyz"], axis=1
        )
        errors = np.linalg.norm(found - exact, axis=1)
        assert errors.max() < bound, f"res {resolution}: worst normal error {errors.max():.4f}"
        assert errors.mean() < 0.1
        # A radial normal, not a fixed one: the direction must actually vary.
        assert np.ptp(found[:, 0]) > 1.5


# --- 2c. Degenerate pins ---------------------------------------------------------


@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez"])
@pytest.mark.parametrize("cell_size", [(0.6, 0.5, 0.7), (0.35, 0.35, 0.45)])
def test_homogeneous_medium_smooths_to_exactly_itself(component, cell_size):
    """A uniform medium must come back BYTE-IDENTICAL, not merely close.

    Smoothing something with no interface in it has to be a no-op, and "close" is not
    good enough: a run that silently shifts every cell of epsilon by an ULP is a
    different run. The second cell size gives odd counts on every axis, where the
    engine's registration has historically been half a cell out.
    """
    grid = Grid(resolution=12, cell_size=cell_size)
    value = 2.25
    result = sm.smooth_inverse_epsilon(
        grid, lambda x, y, z: value + 0.0 * (x + y + z), component, 8
    )
    plain = np.full(grid.shape, value, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(result.epsilon), plain)
    np.testing.assert_array_equal(
        np.asarray(result.inverse_epsilon), (np.float32(1.0) / plain)
    )
    assert result.interface_fraction == 0.0
    assert result.unresolved_fraction == 0.0
    assert not np.asarray(result.smoothed).any()
    # Nothing was averaged, so the normal is identically zero everywhere.
    for name in "xyz":
        assert not np.asarray(result.normal[name]).any()


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez"])
def test_interface_exactly_on_a_voxel_boundary_is_not_perturbed(component, axis):
    """An interface that lands on voxel edges leaves every voxel uniform — bit for bit.

    This is the case a smoothing bug is most likely to survive: the correct answer is
    the staircased array, so an implementation that averages anyway still looks
    plausible. The structure is a SLAB, both of whose faces sit on a voxel edge of the
    component being smoothed, with vacuum around it so the periodic wrap adds no third
    interface. Every voxel is then internally uniform, the result must equal the
    point-sampled array exactly, and no voxel may be flagged as averaged. Voxel edges
    are at integer positions on the component's own axis and half-integer positions on
    the others, so getting the Yee shift wrong fails this on two axes out of three.
    """
    grid = Grid(resolution=20, cell_size=(0.5, 0.5, 0.5))
    low = _voxel_span(grid, component, axis, 3)[0]
    high = _voxel_span(grid, component, axis, 6)[1]

    def epsilon(x, y, z):
        coordinate = (x, y, z)[axis]
        others = sum(value for index, value in enumerate((x, y, z)) if index != axis)
        return np.where((coordinate >= low) & (coordinate < high), E2, E1) + 0.0 * others

    result = sm.smooth_inverse_epsilon(grid, epsilon, component, 16)

    assert not np.asarray(result.smoothed).any(), "a boundary-aligned interface was averaged"
    assert result.interface_fraction == 0.0
    assert result.unresolved_fraction == 0.0
    # The point-sampled comparison must be made at the component's own Yee positions,
    # which is where an untouched voxel takes its value from.
    axes = sm.voxel_centers(grid, component)
    expected = np.broadcast_to(
        epsilon(axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]),
        grid.shape,
    ).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(result.epsilon), expected)
    np.testing.assert_array_equal(
        np.asarray(result.inverse_epsilon), (np.float32(1.0) / expected)
    )
    # Both materials really are present, or the pin is vacuous.
    assert len(np.unique(expected)) == 2


def test_driver_smoothed_equals_set_epsilon_byte_for_byte_without_an_interface():
    """Through the driver: smoothing a uniform medium changes no bit of the material.

    The engine has no "smoothing off" switch — ``set_epsilon`` IS the off path — so
    this is the disabled-vs-enabled identity, and it must hold on the two arrays the
    stepping kernels read rather than on a norm of them.
    """
    for cell in ((0.6, 0.5, 0.7), (0.35, 0.4, 0.45)):
        plain = FdtdDriver(cell_size=cell, resolution=12, force_complex_fields=True)
        smoothed = FdtdDriver(cell_size=cell, resolution=12, force_complex_fields=True)
        try:
            plain.set_epsilon(np.full(plain.grid.shape, 6.25, dtype=np.float32))
            report = smoothed.set_epsilon_smoothed(
                lambda x, y, z: 6.25 + 0.0 * (x + y + z), "Ey", supersample=8
            )
            np.testing.assert_array_equal(
                np.asarray(smoothed.fields.eps), np.asarray(plain.fields.eps)
            )
            np.testing.assert_array_equal(
                np.asarray(smoothed.fields.inv_eps), np.asarray(plain.fields.inv_eps)
            )
            assert report.interface_fraction == 0.0
        finally:
            plain.close()
            smoothed.close()


def test_untouched_voxels_of_a_mixed_structure_keep_the_point_sampled_value():
    """Only interface voxels move; the interior is the staircased array bit for bit."""
    grid = Grid(resolution=20, cell_size=(1.0, 0.2, 1.0))
    epsilon = _bar((-0.155, 0.315), (-0.3, 0.3))
    result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (32, 1, 32))
    mask = np.asarray(result.smoothed)
    assert 0.0 < result.interface_fraction < 0.25, "the mask must be a boundary, not the volume"

    axes = sm.voxel_centers(grid, "Ey")
    point = np.broadcast_to(
        epsilon(axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]),
        grid.shape,
    ).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(result.epsilon)[~mask], point[~mask])
    # And the averaged voxels really did move, or the mask means nothing.
    assert not np.array_equal(np.asarray(result.epsilon)[mask], point[mask])


# --- 3. Registration: the historical half-cell defects ---------------------------


@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez"])
@pytest.mark.parametrize("cell_size", [(0.8, 0.8, 0.8), (0.75, 0.55, 0.65)])
def test_voxel_centers_agree_with_the_engines_one_registration_rule(component, cell_size):
    """The voxel centres must be ``dispersion.component_coordinates``, odd counts included.

    Two modules deriving the same Yee positions independently is how this engine's
    4.8e-2 odd-count defect happened. The second cell size has an odd count on every
    axis, where ``axis_origin`` and ``-L/2`` differ by half a cell.
    """
    grid = Grid(resolution=20, cell_size=cell_size)
    expected = component_coordinates(grid, component)
    found = sm.voxel_centers(grid, component)
    for axis in range(3):
        np.testing.assert_allclose(found[axis], np.asarray(expected[axis]), rtol=0.0, atol=1e-15)
    # The shift is on the component's OWN axis and nowhere else.
    own = sm.E_COMPONENTS.index(component)
    for axis in range(3):
        offset = float(found[axis][0] - grid.axis_origin(axis)) / grid.dx
        assert offset == pytest.approx(0.5 if axis == own else 0.0, abs=1e-12)


def test_fine_lattice_tiles_the_voxel_and_wraps_into_the_cell():
    """``fine_coordinates`` is one uniform lattice, one period long, folded into the cell.

    A caller building the array form must sample here; anything else reintroduces the
    half-cell registration error. Two properties make that checkable: the unwrapped
    lattice has constant spacing ``dx/s``, and the wrap sends the leading samples of
    an unshifted axis to the TOP of the cell because voxel 0 straddles the boundary.
    """
    supersample = 8
    grid = Grid(resolution=10, cell_size=(0.7, 0.5, 0.5))  # nx = 7, odd.
    for component in sm.E_COMPONENTS:
        axes = sm.fine_coordinates(grid, component, supersample)
        own = sm.E_COMPONENTS.index(component)
        for axis in range(3):
            count = (grid.nx, grid.ny, grid.nz)[axis]
            values = np.asarray(axes[axis])
            assert values.size == count * supersample
            origin = grid.axis_origin(axis)
            length = count * grid.dx
            assert np.all(values >= origin - 1e-12)
            assert np.all(values < origin + length + 1e-12)
            # Unfold and check the spacing is exactly dx/s with the right start.
            start = origin + (0.5 * sm.E_YEE_SHIFTS[component][axis] - 0.5) * grid.dx
            unwrapped = start + (np.arange(count * supersample) + 0.5) * grid.dx / supersample
            np.testing.assert_allclose(
                values, origin + (unwrapped - origin) % length, rtol=0.0, atol=1e-14
            )
            # On the component's own axis the voxel is centred on the Yee point, so the
            # lattice starts inside the cell; on the others voxel 0 straddles the floor.
            if axis == own:
                assert values[0] > origin
            else:
                assert values[0] > origin + 0.5 * length, "the wrap did not fold voxel 0"


@pytest.mark.parametrize("fill_rule,tolerance", FILL_RULES)
def test_structure_touching_the_cell_edge_is_averaged_across_the_periodic_wrap(
    fill_rule, tolerance
):
    """The boundary voxel must see the material on the far side of the cell.

    Every axis of this engine is periodic (a PML is a material inside the cell), and
    the voxel of cell 0 on an unshifted axis straddles the wrap. MEEP evaluates the
    wrapped point too (``ensure_periodicity``). Without the fold, a structure touching
    the cell edge is smoothed against whatever the caller's epsilon returns outside —
    silently, and only at the boundary. Both fill rules go through the same fold, and
    the analytic one bisects across it, so both are checked here.
    """
    grid = Grid(resolution=20, cell_size=(1.0, 0.2, 1.0))
    origin = grid.axis_origin(0)
    half = origin + 0.5 * grid.nx * grid.dx
    # E2 in the lower half of the cell, E1 in the upper: as a PERIODIC structure that is
    # two interfaces, one at mid-cell and one exactly on the wrap. Voxel 0 of an
    # x-unshifted component straddles the wrap, so it must come out half and half —
    # which it can only do by evaluating epsilon at the folded point near the cell top.
    epsilon = _step(0, half, low=E2, high=E1)
    result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (16, 1, 1), fill_rule=fill_rule)
    mask = np.asarray(result.smoothed)
    assert mask[0, 0, 0], "the voxel straddling the wrap was not averaged"
    wrapped_plane = grid.ny * grid.nz
    assert mask.sum() == 2 * wrapped_plane, "exactly the two interface planes may average"
    assert mask[grid.nx // 2, 0, 0], "the mid-cell interface must average too"

    mean_epsilon, mean_inverse = _mixture(0.5)
    for index in (0, grid.nx // 2):
        _equals(float(np.asarray(result.mean_epsilon)[index, 0, 0]), mean_epsilon, tolerance)
        _equals(
            float(np.asarray(result.mean_inverse_epsilon)[index, 0, 0]),
            mean_inverse, tolerance,
        )
        # Ey with an x-normal interface is tangential: the arithmetic mean, 1/2.5.
        assert float(np.asarray(result.row["y"])[index, 0, 0]) == pytest.approx(0.4, rel=1e-14)
    # The wrapped voxel is the one the fold buys: without it, cell 0 would read whatever
    # the callable returns below the cell floor, which here is E2 on both halves.
    assert float(np.asarray(result.epsilon)[0, 0, 0]) == np.float32(1.0 / 0.4)


def test_array_form_reproduces_the_callable_form_on_the_published_lattice():
    """The two accepted spellings of epsilon must agree where it matters, and they do.

    ``fine_coordinates`` is the contract between them; a caller who samples anywhere
    else gets the half-cell error the array path exists to avoid, so the lattice and
    the reducer have to be consistent. The voxel means, the smooth/point-sample
    decision and the unresolved report must be BIT-identical between the two.

    What is not identical, and is not free: an array epsilon has nothing between its
    samples, so the sphere quadrature reads it piecewise-constant and the interface
    NORMAL comes back quantized to the fine lattice. Measured at a bar's corner voxel,
    the array form reads (-0.707, 0, 0.707) where the callable reads
    (-0.848, 0, 0.530), which moves that voxel's Ez diagonal by 6.9 %. It cannot move
    anything the driver installs, because the driver's invariance forces ``n_d = 0``
    and the diagonal then ignores the normal — which is what the Ey assertion here
    pins.

    The comparison is made under ``fill_rule="sampled"`` on BOTH sides, because that is
    the only rule the array form has: bisecting a piecewise-constant array would return
    a fine-cell edge and hand back exactly the quantization the analytic fill removes,
    so ``smooth_inverse_epsilon`` refuses ``"planar"`` for an array rather than
    downgrading it silently (checked below).
    """
    grid = Grid(resolution=20, cell_size=(0.75, 0.2, 1.0))  # nx = 15, odd.
    epsilon = _bar((-0.155, 0.215), (-0.3, 0.3))  # y-invariant, so Ey is the driver case.
    rates = (16, 1, 16)
    normal_differs = {}
    for component in sm.E_COMPONENTS:
        axes = sm.fine_coordinates(grid, component, rates)
        values = np.broadcast_to(
            epsilon(axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]),
            tuple(n * r for n, r in zip(grid.shape, rates)),
        )
        from_callable = sm.smooth_inverse_epsilon(
            grid, epsilon, component, rates, fill_rule="sampled"
        )
        from_array = sm.smooth_inverse_epsilon(grid, values, component, rates)
        assert from_array.fill_rule == "sampled", "an array epsilon must resolve to sampled"
        assert from_array.planar_fraction == 0.0
        # The geometry the means see is the same lattice, so these are exact equalities.
        np.testing.assert_array_equal(
            np.asarray(from_array.mean_epsilon), np.asarray(from_callable.mean_epsilon))
        np.testing.assert_array_equal(
            np.asarray(from_array.mean_inverse_epsilon),
            np.asarray(from_callable.mean_inverse_epsilon))
        np.testing.assert_array_equal(
            np.asarray(from_array.smoothed), np.asarray(from_callable.smoothed))
        np.testing.assert_array_equal(
            np.asarray(from_array.unresolved), np.asarray(from_callable.unresolved))
        assert from_array.interface_fraction == from_callable.interface_fraction
        normal_differs[component] = max(
            float(np.max(np.abs(np.asarray(from_array.normal[name])
                                - np.asarray(from_callable.normal[name]))))
            for name in "xyz"
        )
        if component == "Ey":
            # The driver's component: the bar is y-invariant, so n_y is exactly zero on
            # BOTH paths and the diagonal cannot feel the quantized normal. The array
            # the engine would install is therefore bit-identical.
            own = "xyz"[sm.E_COMPONENTS.index(component)]
            assert np.max(np.abs(np.asarray(from_array.normal[own]))) < 1e-15
            assert np.max(np.abs(np.asarray(from_callable.normal[own]))) < 1e-15
            np.testing.assert_array_equal(
                np.asarray(from_array.row[own]), np.asarray(from_callable.row[own]))
            np.testing.assert_array_equal(
                np.asarray(from_array.inverse_epsilon),
                np.asarray(from_callable.inverse_epsilon))
        else:
            # Here the normal reaches the diagonal, and the quantization shows in it.
            own = "xyz"[sm.E_COMPONENTS.index(component)]
            diagonal_gap = float(np.max(np.abs(
                np.asarray(from_array.row[own]) - np.asarray(from_callable.row[own]))))
            assert diagonal_gap > 0.0 if component == "Ez" else diagonal_gap >= 0.0
    # The quantized normal is real — the sphere reads the array piecewise-constant, so
    # two of the three components see a different direction — and it reaches the
    # installed value only where n_d != 0, which the Ey branch above pins.
    assert max(normal_differs.values()) > 0.05, "the fine-lattice normal quantization vanished"
    assert normal_differs["Ez"] > 0.1


def test_chunking_does_not_change_the_result():
    """Splitting the voxel loop is a memory strategy, not a numerical one."""
    grid = Grid(resolution=20, cell_size=(1.0, 0.2, 1.0))
    epsilon = _bar((-0.155, 0.315), (-0.3, 0.3))
    reference = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (32, 1, 32))
    original = sm._CHUNK_BUDGET
    try:
        sm._CHUNK_BUDGET = 5000  # Force many chunks on both leading axes.
        chunked = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (32, 1, 32))
    finally:
        sm._CHUNK_BUDGET = original
    assert len(sm._chunk_ranges(grid.shape, (32, 1, 32))) == 1
    np.testing.assert_array_equal(
        np.asarray(chunked.inverse_epsilon), np.asarray(reference.inverse_epsilon)
    )
    np.testing.assert_array_equal(
        np.asarray(chunked.smoothed), np.asarray(reference.smoothed)
    )


# --- 4. The scalar-reduction contract, and what the driver refuses ----------------


@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez"])
def test_driver_installs_the_arithmetic_mean_in_every_configuration_it_accepts(component):
    """The reachable half of the scheme, stated as a test.

    ``set_epsilon_smoothed`` requires invariance along the component's own axis, which
    forces ``n_d = 0``, which collapses the diagonal to ``1/<eps>``. So the array the
    engine installs is always the reciprocal of the plain volume average of epsilon,
    and the interface normal — the expensive part — never touches it. Anyone reading
    the docstrings and expecting the harmonic mean to be doing work should read this.
    """
    own = sm.E_COMPONENTS.index(component)
    grid_cell = (1.0, 1.0, 1.0)
    axes = [a for a in range(3) if a != own]

    def epsilon(x, y, z):
        first, second = (x, y, z)[axes[0]], (x, y, z)[axes[1]]
        phase = (2.0 * first + second) % 1.0
        return np.where(phase < 0.45, E1, E2) + 0.0 * (x, y, z)[own]

    driver = FdtdDriver(cell_size=grid_cell, resolution=20, force_complex_fields=True)
    try:
        rates = tuple(1 if a == own else 32 for a in range(3))
        report = driver.set_epsilon_smoothed(epsilon, component, supersample=rates)
        mask = np.asarray(report.smoothed)
        assert mask.any()
        assert report.offdiagonal_magnitude < 1e-15
        means = np.asarray(report.mean_epsilon)
        expected = np.where(mask, 1.0 / means, 1.0 / means).astype(np.float32)
        np.testing.assert_array_equal(np.asarray(driver.fields.inv_eps), expected)
        # The normal is nonzero and varies — it is computed, and it is then ignored.
        along = np.asarray(report.normal["xyz"[axes[0]]])[mask]
        assert np.max(np.abs(along)) > 0.1
    finally:
        driver.close()


def test_driver_refuses_variation_along_the_smoothed_components_own_axis():
    """The reduction is exact or refused — never quietly approximate.

    A structure varying along the component's own axis would need the off-diagonal
    entries the single shared ``inv_eps`` cannot store. The result would be a smooth,
    complete, few-percent-wrong field, which is the failure mode this engine has
    shipped most often, so the driver raises instead.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=20, force_complex_fields=True)
    try:
        varies_in_y = lambda x, y, z: np.where(y < 0.1, E1, E2) + 0.0 * (x + z)
        with pytest.raises(ValueError, match="invariant along the y axis"):
            driver.set_epsilon_smoothed(varies_in_y, "Ey", supersample=8)
        # The same structure is fine for a component that lies along an invariant axis.
        report = driver.set_epsilon_smoothed(varies_in_y, "Ex", supersample=(1, 32, 1))
        assert report.interface_fraction > 0.0
        # A tolerance admits it, and the refusal message carries the measured variation.
        with pytest.raises(ValueError, match=r"varies by [0-9.e+-]+"):
            driver.set_epsilon_smoothed(varies_in_y, "Ey", supersample=8,
                                        variation_tolerance=0.1)
        driver.set_epsilon_smoothed(varies_in_y, "Ey", supersample=8, variation_tolerance=1.0)
    finally:
        driver.close()


def test_variation_measure_is_zero_for_an_invariant_structure_and_positive_otherwise():
    """The gate on the scalar reduction: exactly 0.0 for an invariant epsilon.

    ``supersample = 1`` on the tested axis narrows the check to variation BETWEEN
    voxels — it still samples one point per voxel along that axis, so a structure with
    a period longer than a cell is still caught, and only sub-voxel variation slips
    through. A thin feature that fits inside one cell along the component's own axis is
    therefore invisible to the gate, which is the one hole in it.
    """
    grid = Grid(resolution=20, cell_size=(1.0, 1.0, 1.0))
    invariant = _bar((-0.155, 0.315), (-0.3, 0.3))  # y-invariant.
    assert sm.epsilon_variation_along(grid, invariant, "Ey", (16, 16, 16)) == 0.0
    assert sm.epsilon_variation_along(grid, invariant, "Ey", (16, 1, 16)) == 0.0
    assert sm.epsilon_variation_along(grid, invariant, "Ex", (16, 16, 16)) > 0.5
    assert sm.epsilon_variation_along(grid, invariant, "Ez", (16, 16, 16)) > 0.5
    # Rate 1 still sees cell-to-cell variation along the tested axis.
    assert sm.epsilon_variation_along(grid, invariant, "Ex", (1, 16, 16)) > 0.5
    # But a feature thinner than one cell along that axis is not seen at rate 1.
    subcell = _bar((0.0, 0.3 * grid.dx), (-0.3, 0.3))
    assert sm.epsilon_variation_along(grid, subcell, "Ex", (16, 16, 16)) > 0.5
    assert sm.epsilon_variation_along(grid, subcell, "Ex", (1, 16, 16)) == 0.0


@pytest.mark.parametrize(
    "supersample,message",
    [
        (0, "must be >= 1 on every axis"),
        ((8, 0, 8), "must be >= 1 on every axis"),
        ((8, 8), "per-axis triple"),
        (2.5, "must be an int or a 3-tuple"),
        (True, "must be an int or a 3-tuple"),
    ],
)
def test_refuses_a_bad_supersample(supersample, message):
    grid = Grid(resolution=10, cell_size=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match=message):
        sm.smooth_inverse_epsilon(grid, lambda x, y, z: 1.0 + 0.0 * (x + y + z), "Ey", supersample)


def test_refuses_an_unknown_component_and_magnetic_smoothing():
    grid = Grid(resolution=10, cell_size=(0.5, 0.5, 0.5))
    for bad in ("Hx", "ex", "Er", None, 1):
        with pytest.raises(ValueError, match="chi1inv row"):
            sm.smooth_inverse_epsilon(grid, lambda x, y, z: 1.0 + 0.0 * (x + y + z), bad, 4)
    with pytest.raises(ValueError, match="chi1inv row"):
        sm.voxel_centers(grid, "Hz")


def test_refuses_non_positive_or_non_finite_epsilon():
    """Averaging ``1/eps`` makes a zero permittivity a divide, not a hard material."""
    grid = Grid(resolution=10, cell_size=(0.5, 0.5, 0.5))
    for bad in (lambda x, y, z: 0.0 * (x + y + z),
                lambda x, y, z: np.where(x < 0, -2.0, 1.0) + 0.0 * (y + z),
                lambda x, y, z: np.where(x < 0, np.inf, 1.0) + 0.0 * (y + z),
                lambda x, y, z: np.where(x < 0, np.nan, 1.0) + 0.0 * (y + z)):
        with pytest.raises(ValueError, match="non-finite or non-positive"):
            sm.smooth_inverse_epsilon(grid, bad, "Ey", 4)


def test_refuses_a_callable_that_does_not_broadcast():
    grid = Grid(resolution=10, cell_size=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="must broadcast"):
        sm.smooth_inverse_epsilon(grid, lambda x, y, z: np.array([1.0, 2.0, 3.0]), "Ey", 4)


def test_refuses_a_mis_shaped_or_wrongly_sampled_array():
    grid = Grid(resolution=10, cell_size=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="must be 3-D"):
        sm.smooth_inverse_epsilon(grid, np.ones(10), "Ey", 4)
    with pytest.raises(ValueError, match="fine_coordinates"):
        sm.smooth_inverse_epsilon(grid, np.ones(grid.shape), "Ey", 4)


def test_refuses_a_mirror_folded_grid():
    """A folded quadrant is not a period, so the voxel means would wrap onto a mirror."""
    grid = Grid(resolution=10, cell_size=(1.0, 1.0, 0.5), symmetry=("X",))
    assert grid.has_symmetry()
    with pytest.raises(ValueError, match="mirror-folded grid"):
        sm.smooth_inverse_epsilon(grid, lambda x, y, z: 1.0 + 0.0 * (x + y + z), "Ey", 4)


def test_driver_refuses_a_bad_variation_tolerance():
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=10, force_complex_fields=True)
    try:
        for bad in (-1e-9, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="variation_tolerance"):
                driver.set_epsilon_smoothed(
                    lambda x, y, z: 1.0 + 0.0 * (x + y + z), "Ey",
                    supersample=4, variation_tolerance=bad,
                )
    finally:
        driver.close()


# --- 5. The two silent failure modes ---------------------------------------------


def test_corner_voxel_is_declined_and_reported():
    """MEEP's 8-point shortcut can skip a corner voxel — and that must be VISIBLE.

    At the corner of a rectangular bar all six axis directions and both body diagonals
    of the sampling sphere can land in the surrounding material, so
    ``normal_vector`` returns zero and ``eff_chi1inv_row`` point-samples a voxel that
    is 17 % filled with the other material. MEEP does exactly this for a user material
    function (it escapes only via the analytic geometric path), and the value is 53 %
    away from MEEP's analytic answer. It used to be indistinguishable from a genuinely
    uniform voxel; ``unresolved_fraction`` now counts it.
    """
    grid = Grid(resolution=20, cell_size=(1.0, 0.2, 1.0))
    epsilon = _bar((-0.125, 0.245), (-0.2969, 0.3443))
    result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (64, 1, 64))

    unresolved = np.asarray(result.unresolved)
    assert unresolved.any(), "the corner voxel must be reported, not merely skipped"
    assert result.unresolved_fraction == pytest.approx(4 / 112, abs=1e-9)
    # It is a corner: near a bar face on BOTH axes, and not averaged.
    centres = sm.voxel_centers(grid, "Ey")
    smoothed = np.asarray(result.smoothed)
    for index in np.argwhere(unresolved):
        assert not smoothed[tuple(index)]
        near_x = min(abs(centres[0][index[0]] - edge) for edge in (-0.125, 0.245))
        near_z = min(abs(centres[2][index[2]] - edge) for edge in (-0.2969, 0.3443))
        assert near_x <= 0.5 * grid.dx and near_z <= 0.5 * grid.dx
        # The voxel carries the unsmoothed point sample even though it is not uniform.
        assert float(np.asarray(result.epsilon)[tuple(index)]) in (E1, E2)
    # Everything else that the fine lattice resolved WAS averaged.
    assert result.unresolved_fraction < 0.05


def test_a_uniform_medium_reports_no_unresolved_voxels():
    """The diagnostic must not fire where there is nothing to miss."""
    grid = Grid(resolution=20, cell_size=(0.5, 0.5, 0.5))
    for epsilon in (lambda x, y, z: 3.0 + 0.0 * (x + y + z),
                    _step(0, _voxel_span(Grid(resolution=20, cell_size=(0.5, 0.5, 0.5)),
                                         "Ey", 0, 4)[0])):
        result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", 16)
        assert result.unresolved_fraction == 0.0
        assert not np.asarray(result.unresolved).any()


def test_susceptibility_sigma_is_not_smoothed_with_epsilon():
    """MEEP smooths eps_infinity only, and so does this engine — deliberately.

    ``anisotropic_averaging.cpp`` averages chi1inv at :252-257 and POINT-SAMPLES sigma
    at :333-345, which is the first limitation MEEP's own Subpixel_Smoothing.md states.
    A run that also smoothed sigma would be a different material model from MEEP's, so
    the pairing is checked: the smoothed eps survives adding a Lorentz term, and the
    stored sigma is exactly the point-sampled volume that went in.
    """
    driver = FdtdDriver(cell_size=(1.0, 0.4, 1.0), resolution=20, force_complex_fields=True)
    try:
        epsilon = _bar((-0.155, 0.315), (-0.3, 0.3), inside=4.0, outside=1.0)
        report = driver.set_epsilon_smoothed(epsilon, "Ey", supersample=(32, 1, 32))
        installed = np.array(np.asarray(driver.fields.inv_eps), copy=True)
        assert report.interface_fraction > 0.0

        sigma_volume = np.zeros(driver.grid.shape, dtype=np.float32)
        sigma_volume[:] = 0.5
        driver.add_susceptibility(
            Susceptibility(kind=LORENTZIAN, frequency=1.0, gamma=0.0), {"Ey": sigma_volume}
        )
        # eps_infinity is untouched by adding the term...
        np.testing.assert_array_equal(np.asarray(driver.fields.inv_eps), installed)
        # ...and sigma is the array handed in, not an average of it.
        stored = driver.fields.polarizations[0].sigma["Ey"]
        np.testing.assert_array_equal(np.asarray(stored), sigma_volume)
    finally:
        driver.close()


# --- 6. Supersample: the term that actually limits the result --------------------


def _bar_exact_diagonal(grid, x_span, z_span):
    """``1/<eps>`` per voxel from the ANALYTIC overlap of each voxel with the bar.

    Separable, so the exact fill is a product of two 1-D clipped overlaps — which is
    what MEEP's ``box_overlap_with_object`` computes for a Block and what the module's
    planar path has to reproduce. Shares no code with :mod:`smoothing`.
    """
    overlaps = {}
    for axis, (low, high) in ((0, x_span), (2, z_span)):
        count = (grid.nx, grid.ny, grid.nz)[axis]
        fractions = np.empty(count)
        for index in range(count):
            v_low, v_high = _voxel_span(grid, "Ey", axis, index)
            fractions[index] = max(0.0, min(v_high, high) - max(v_low, low)) / grid.dx
        overlaps[axis] = fractions
    fill = overlaps[0][:, None] * overlaps[2][None, :]
    mean = fill * E2 + (1.0 - fill) * E1
    result = np.empty(grid.shape)
    result[:] = (1.0 / mean)[:, None, :]
    return result


def test_supersample_error_is_first_order_in_one_over_s():
    """Halving the subcell halves the SAMPLED error — the rule the planar path replaces.

    The voxel means are midpoint sums, so a planar interface is located to ``dx/(2s)``
    and the fill fraction carries an ``O(1/s)`` error. Against the EXACT analytic fill
    (which is what MEEP uses for a geometric object) the worst interface voxel of a
    rectangular bar reads 7.4e-2 / 3.5e-2 / 1.6e-2 / 8.7e-3 / 4.5e-3 / 2.1e-3 for
    ``s`` = 8 .. 256. That ladder is why ``fill_rule="planar"`` exists and is the
    default; ``test_planar_fill_is_independent_of_supersample`` is the same measurement
    with the analytic fill, where the ladder is flat at 1.4e-15.
    """
    grid = Grid(resolution=20, cell_size=(1.0, 0.2, 1.0))
    x_span, z_span = (-0.125, 0.245), (-0.2969, 0.3443)
    epsilon = _bar(x_span, z_span)
    reference = _bar_exact_diagonal(grid, x_span, z_span)
    errors = []
    for supersample in (8, 16, 32, 64, 128, 256):
        result = sm.smooth_inverse_epsilon(
            grid, epsilon, "Ey", (supersample, 1, supersample), fill_rule="sampled"
        )
        mask = np.asarray(result.smoothed) & ~np.asarray(result.unresolved)
        relative = np.abs(np.asarray(result.row["y"]) - reference) / reference
        errors.append(float(relative[mask].max()))

    assert errors[1] == pytest.approx(3.5e-2, rel=0.15), f"s=16 measured {errors[1]:.3e}"
    assert errors[-1] == pytest.approx(2.1e-3, rel=0.15), f"s=256 measured {errors[-1]:.3e}"
    # First order: each doubling of s must roughly halve the error, and never make it
    # worse. A scheme whose error did not track 1/s would not be a midpoint sum.
    for coarse, fine in zip(errors, errors[1:]):
        assert 1.7 < coarse / fine < 2.5, f"ratio {coarse / fine:.2f} is not first order"
    # And the sampled rule is NOT small at any affordable s: this is why it is not the
    # default any more.
    assert errors[1] > 1e-2, "if the sampled rule became accurate, retune the docstrings too"


def test_planar_fill_is_independent_of_supersample():
    """The claim the analytic fill is FOR: the same ``s`` ladder, flat at machine zero.

    Every voxel of the bar that the planar path accepts must reproduce the analytic
    overlap to the last bits, at ``s`` = 8 and at ``s`` = 256 alike, because the fill
    fraction no longer comes from the fine lattice at all — it comes from a bisected
    plane and a closed-form cube integral. Side by side with
    ``test_supersample_error_is_first_order_in_one_over_s``:

        supersample        8       16       32       64      128      256
        sampled        7.4e-2  3.5e-2  1.6e-2  8.7e-3  4.5e-3  2.1e-3
        planar         1.4e-15 1.4e-15 1.4e-15 1.4e-15 1.4e-15 1.4e-15

    The bar's CORNER voxel is excluded, and that exclusion is the honest limit of this:
    a corner is two planes, the fit declines it (``planar`` is False there), and it
    keeps the sampled answer. ``planar_fraction`` says how much of the interface that
    is — 96 % analytic here, the remaining 4 % being the single corner.
    """
    grid = Grid(resolution=20, cell_size=(1.0, 0.2, 1.0))
    x_span, z_span = (-0.125, 0.245), (-0.2969, 0.3443)
    epsilon = _bar(x_span, z_span)
    reference = _bar_exact_diagonal(grid, x_span, z_span)

    analytic, declined = [], []
    for supersample in (8, 16, 32, 64, 128, 256):
        result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (supersample, 1, supersample))
        assert result.fill_rule == "planar", "a callable epsilon must resolve to planar"
        usable = np.asarray(result.smoothed) & ~np.asarray(result.unresolved)
        planar = np.asarray(result.planar)
        relative = np.abs(np.asarray(result.row["y"]) - reference) / reference
        assert planar.sum() > 90, "the bar's faces must all be analytic"
        analytic.append(float(relative[usable & planar].max()))
        declined.append(float(relative[usable & ~planar].max()))
        assert result.planar_fraction == pytest.approx(104 / 108, abs=1e-9)
        assert result.planarity_residual == 0.0, "a flat face has no curvature"

    assert max(analytic) < 1e-14, f"planar ladder is {analytic}"
    # Flat, not merely small: the whole point is that s stopped mattering.
    assert max(analytic) / max(min(analytic), 1e-300) < 10.0
    # And the corner really is the residue, still paying the 1/s rule.
    assert declined[0] > 100 * max(analytic)
    assert declined[-1] < declined[0]


def test_default_supersample_and_fill_rule_are_the_documented_values():
    """A silent change of either default would move every measured number in this file.

    The supersample default moved 16 -> 32 when the planar fill landed, and it moved for
    a different reason than the old plan had: with the fill analytic, ``s`` no longer
    sets ACCURACY (see ``test_planar_fill_is_independent_of_supersample``), it sets
    DETECTION — the thinnest feature a voxel can notice is ``dx/s``. Measured on the
    13.5k-voxel grating grid, the one-time pre-pass costs about ``s**3 / 12`` time steps
    of the run it precedes (0.465 s = 341 steps at 16, 2.93 s = 2731 steps at 32, an
    estimated 21845 steps at 64), a ratio that is resolution-independent because both
    sides scale with the voxel count. 32 is the largest that stays under a typical run;
    64 would cost more than the whole run and buy nothing measurable. Full table in
    ``smoothing.DEFAULT_SUPERSAMPLE``'s comment block.
    """
    assert sm.DEFAULT_SUPERSAMPLE == 32
    assert sm.DEFAULT_FILL_RULE == "auto"
    assert sm.FILL_RULES == ("auto", "planar", "sampled")
    assert sm.SMOOTHING_DIAMETER == 1.0
    assert sm.GRADIENT_FLOOR == 1e-8
    assert sm.NQUAD3 == 50
    # The analytic path's own constants; every measured number below assumes them.
    assert sm.PLANE_PROBE == 0.25
    assert sm.PLANE_BISECTION_STEPS == 52
    assert sm.PLANE_SLOPE_FLOOR == 1e-4
    assert sm.PLANE_RESIDUAL_TOLERANCE == 4e-3
    assert sm.PLANE_SLOPE_CEILING == 4.0
    assert sm.PLANE_CROSSCHECK_SLACK == 3.0


# --- 6b. The analytic planar fill ------------------------------------------------


def _clamped_wedge_integral(level, slope_u, slope_v, samples=200001):
    """``int int clamp(level + p u + q v + 1/2, 0, 1)`` — exact in u, midpoint in v.

    An independent derivation of what :func:`smoothing.cube_plane_fill` computes: the
    inner integral is done with the antiderivative of ``clamp``, so the only error is
    the outer midpoint rule over a function with two kinks, ``O(1/samples**2)`` away
    from them and measured below 2e-12 here. It shares no algebra with the shipped
    closed form — no cubic spline, no inclusion-exclusion, no degenerate branches.
    """
    v = (np.arange(samples) + 0.5) / samples - 0.5
    offset = level + 0.5 + slope_v * v
    if abs(slope_u) < 1e-13:
        return float(np.mean(np.clip(offset, 0.0, 1.0)))
    width = abs(slope_u)
    low, high = offset - width / 2.0, offset + width / 2.0

    def antiderivative(value):
        return np.where(value < 0.0, 0.0,
                        np.where(value <= 1.0, value * value / 2.0, value - 0.5))

    return float(np.mean((antiderivative(high) - antiderivative(low)) / width))


def test_analytic_fill_matches_an_independent_integral_of_the_same_cube():
    """``cube_plane_fill`` against a differently-derived integral, over the whole range.

    The closed form is a cubic spline reached by inclusion-exclusion; the reference
    integrates the same clamped plane analytically in one direction and by midpoints in
    the other. They agree over slopes from 0 to 1 and levels that put the plane anywhere
    from below the cube to above it: worst measured 2.1e-12, which is the REFERENCE's
    own midpoint error, not the closed form's.
    """
    worst = 0.0
    for slope_u, slope_v in ((0.0, 0.0), (0.3, 0.0), (0.0, 0.7), (1.0, 1.0),
                             (0.62, -0.41), (-0.9, 0.15), (1.0, -1.0)):
        for level in np.linspace(-1.05, 1.05, 61):
            found = float(sm.cube_plane_fill(level, slope_u, slope_v))
            expected = _clamped_wedge_integral(level, slope_u, slope_v)
            worst = max(worst, abs(found - expected))
    assert worst < 1e-11, f"worst gap against the independent integral {worst:.3e}"

    # A crude 3-D count too, because the reference above shares the reduction to a
    # height function with the closed form and a flipped inequality would survive it.
    n = 200
    axis = (np.arange(n) + 0.5) / n - 0.5
    u, v, w = np.meshgrid(axis, axis, axis, indexing="ij")
    for level, slope_u, slope_v in ((0.2, 0.5, -0.25), (-0.3, 0.0, 0.0), (0.05, 1.0, 1.0)):
        counted = float(np.mean(w <= level + slope_u * u + slope_v * v))
        assert counted == pytest.approx(
            float(sm.cube_plane_fill(level, slope_u, slope_v)), abs=3.0 / n
        )

    # Bounds, monotonicity in the level, and the reflection identity F(h) + F(-h) = 1.
    # The sweep runs past +-(1 + |p| + |q|)/2, where the plane clears the cube entirely.
    levels = np.linspace(-1.6, 1.6, 2001)
    for slope_u, slope_v in ((0.0, 0.0), (1.0, 1.0), (0.7, -0.3), (1e-6, 0.9)):
        values = np.asarray(sm.cube_plane_fill(levels, slope_u, slope_v))
        assert values.min() == 0.0 and values.max() == 1.0
        # Monotone in the level to round-off: raising the plane can only add material.
        assert np.all(np.diff(values) >= -1e-14)
        mirrored = np.asarray(sm.cube_plane_fill(-levels, slope_u, slope_v))
        assert float(np.max(np.abs(values + mirrored - 1.0))) < 1e-9


def test_analytic_fill_is_exact_where_the_plane_is_axis_aligned():
    """The grating case: zero slopes must give ``level + 1/2`` to the last bit.

    An axis-aligned interface is where the four-term closed form cancels completely
    (both slopes divide it), so it is taken by an explicit limit instead. That limit has
    to be EXACT, not merely close: it is what every grating, slab and photonic crystal
    in this engine lands on, and one ULP of drift there would move a bit-identical pin.
    """
    for level in (-0.5, -0.3, -0.1875, 0.0, 0.3125, 0.4999, 0.5, 0.9):
        expected = min(max(level + 0.5, 0.0), 1.0)
        assert float(sm.cube_plane_fill(level, 0.0, 0.0)) == expected
    # And a slope under the floor is dropped rather than divided by, continuously: the
    # jump across PLANE_SLOPE_FLOOR is 1.4e-6 at worst and the branch below it is exact.
    levels = np.linspace(-1.2, 1.2, 4001)
    floored = np.asarray(sm.cube_plane_fill(levels, 0.3, sm.PLANE_SLOPE_FLOOR))
    kept = np.asarray(sm.cube_plane_fill(levels, 0.3, sm.PLANE_SLOPE_FLOOR * 1.0000001))
    assert float(np.max(np.abs(floored - kept))) < 2e-6
    assert float(np.max(np.abs(
        np.asarray(sm.cube_plane_fill(levels, 0.3, 0.0)) - floored))) == 0.0


def test_planar_fill_beats_the_sampled_fill_off_the_fine_lattice():
    """One voxel, one hand value, both rules: the quantization, and its removal.

    The interface sits 0.6137 of the way across the voxel — deliberately NOT on a
    fine-cell boundary, which is where every hand value elsewhere in this file is placed
    so the two rules can be compared at all. The midpoint sum at ``s`` = 16 can only
    answer in sixteenths and picks 10/16 = 0.625; the analytic fill answers 0.6137::

        sampled  <eps> = 0.6250 * 1 + 0.3750 * 4 = 2.1250
        planar   <eps> = 0.6137 * 1 + 0.3863 * 4 = 2.1589

    1.6 % apart on this voxel, and the planar one is the exact answer MEEP's geometric
    path would give. This is the whole feature in a single number.
    """
    supersample, position_fraction = 16, 0.6137
    grid = Grid(resolution=20, cell_size=(0.6, 0.6, 0.6))
    index = (4, 4, 4)
    low, _ = _voxel_span(grid, "Ey", 2, index[2])
    epsilon = _step(2, low + position_fraction * grid.dx)
    rates = (1, 1, supersample)

    sampled = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", rates, fill_rule="sampled")
    planar = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", rates, fill_rule="planar")

    quantized_fill = math.floor(position_fraction * supersample + 0.5) / supersample
    assert quantized_fill == 0.625
    assert float(np.asarray(sampled.mean_epsilon)[index]) == pytest.approx(
        _mixture(quantized_fill)[0], rel=1e-15)
    assert float(np.asarray(planar.mean_epsilon)[index]) == pytest.approx(
        _mixture(position_fraction)[0], rel=1e-14)
    assert float(np.asarray(planar.mean_inverse_epsilon)[index]) == pytest.approx(
        _mixture(position_fraction)[1], rel=1e-14)
    gap = abs(float(np.asarray(sampled.mean_epsilon)[index])
              - float(np.asarray(planar.mean_epsilon)[index]))
    assert gap / _mixture(position_fraction)[0] == pytest.approx(0.0157, rel=0.05)
    assert bool(np.asarray(planar.planar)[index])
    assert not bool(np.asarray(sampled.planar)[index])


@pytest.mark.parametrize("cell_size", [(0.6, 0.6, 0.6), (0.55, 0.45, 0.65)])
@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez"])
def test_planar_fill_is_exact_at_every_placement_and_both_cell_parities(component, cell_size):
    """Sweep ORDINARY inputs: the interface slid across a cell, odd counts and even.

    Every silent wrong answer this engine has shipped was found by sweeping ordinary
    inputs rather than by adding an exotic one — an odd cell count that registered half
    a cell off, a flux plane wrong at round coordinates. So the analytic fill is swept
    the same way: 17 sub-cell placements of a planar interface, on both an even-count
    and an odd-count grid, for all three components, and every averaged voxel must
    reproduce the exact overlap of that plane with that voxel.

    ``cell_size`` (0.55, 0.45, 0.65) at resolution 20 gives 11 x 9 x 13 — odd on every
    axis, where ``axis_origin`` is half a cell from ``-L/2``. The structure is a SLAB
    three cells thick sitting in the middle of the cell rather than a half space, so the
    periodic wrap does not add a third face and the exact overlap is a plain clip.
    """
    grid = Grid(resolution=20, cell_size=cell_size)
    own = sm.E_COMPONENTS.index(component)
    normal_axis = (own + 2) % 3  # An axis the component does not lie along.
    count = (grid.nx, grid.ny, grid.nz)[normal_axis]
    assert count >= 7, "the slab needs room to sit clear of both cell edges"
    rates = tuple(24 if a == normal_axis else 1 for a in range(3))
    base = _voxel_span(grid, component, normal_axis, 2)[0]
    thickness = 3.0 * grid.dx

    for step in range(17):
        offset = (step + 0.5) / 17.0
        low_face = base + offset * grid.dx
        high_face = low_face + thickness

        def epsilon(x, y, z, low_face=low_face, high_face=high_face):
            coordinate = (x, y, z)[normal_axis]
            others = sum(v for i, v in enumerate((x, y, z)) if i != normal_axis)
            inside = (coordinate >= low_face) & (coordinate < high_face)
            return np.where(inside, E2, E1) + 0.0 * others

        result = sm.smooth_inverse_epsilon(grid, epsilon, component, rates)
        planar = np.asarray(result.planar)
        means = np.asarray(result.mean_epsilon)
        assert planar.sum() >= 2, f"offset {offset:.3f}: both faces must be fitted"
        # The exact overlap, per voxel, of [low_face, high_face] with the voxel —
        # computed here from the voxel edges and nothing else.
        for index in np.argwhere(planar):
            low, high = _voxel_span(grid, component, normal_axis, int(index[normal_axis]))
            fill = max(0.0, min(high, high_face) - max(low, low_face)) / grid.dx
            assert means[tuple(index)] == pytest.approx(
                fill * E2 + (1.0 - fill) * E1, rel=2e-14
            ), f"offset {offset:.3f} voxel {tuple(index)}: fill {fill}"
        assert result.planarity_residual < 1e-13


def test_planar_path_declines_a_corner_and_keeps_the_sampled_answer():
    """Two planes in one voxel is not one plane, and the fit must say so, not average.

    At a bar corner the five bisection columns disagree — some bracket the x face and
    some the z face, or one of them finds no discontinuity at all within reach — so the
    voxel is refused and keeps the midpoint sums. That is the honest outcome: the fill
    of a corner voxel is a product of two overlaps, not a plane's, and MEEP reaches it
    only because it knows the object. The refusal has to be VISIBLE, which is what
    ``planar_fraction`` is for.
    """
    grid = Grid(resolution=20, cell_size=(1.0, 0.2, 1.0))
    x_span, z_span = (-0.125, 0.245), (-0.2969, 0.3443)
    epsilon = _bar(x_span, z_span)
    rates = (32, 1, 32)
    planar_result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", rates)
    sampled_result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", rates, fill_rule="sampled")

    smoothed = np.asarray(planar_result.smoothed)
    planar = np.asarray(planar_result.planar)
    declined = smoothed & ~planar
    assert declined.any(), "a bar has corners; some voxel must be declined"
    assert 0.9 < planar_result.planar_fraction < 1.0

    # A declined voxel is a corner: within half a cell of a bar face on BOTH axes.
    centres = sm.voxel_centers(grid, "Ey")
    for index in np.argwhere(declined):
        near_x = min(abs(centres[0][index[0]] - edge) for edge in x_span)
        near_z = min(abs(centres[2][index[2]] - edge) for edge in z_span)
        assert near_x <= 0.5 * grid.dx and near_z <= 0.5 * grid.dx

    # And it carries EXACTLY the sampled answer — the planar pass may not perturb a
    # voxel it declined, or the fallback would be a third, undocumented rule.
    np.testing.assert_array_equal(
        np.asarray(planar_result.mean_epsilon)[declined],
        np.asarray(sampled_result.mean_epsilon)[declined],
    )
    np.testing.assert_array_equal(
        np.asarray(planar_result.row["y"])[declined],
        np.asarray(sampled_result.row["y"])[declined],
    )


def test_planar_path_declines_a_voxel_holding_three_materials():
    """A third material has no single interface, so the two-material fill cannot apply.

    Two parallel faces a third of a cell apart put three distinct epsilon values inside
    one voxel. The bisection would find only the first of them and the closed form would
    then average two of the three, silently. The fine lattice's binary check is what
    stops that, so it is checked on a structure built to defeat it.
    """
    grid = Grid(resolution=20, cell_size=(0.5, 0.2, 0.5))
    centre = sm.voxel_centers(grid, "Ey")[2][5]

    def epsilon(x, y, z):
        first = np.where(z < centre - 0.15 * grid.dx, 1.0, 2.0)
        return np.where(z > centre + 0.15 * grid.dx, 5.0, first) + 0.0 * (x + y)

    result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (1, 1, 64))
    sampled = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (1, 1, 64), fill_rule="sampled")
    cell = (0, 0, 5)
    assert bool(np.asarray(result.smoothed)[cell]), "the three-material voxel must average"
    assert not bool(np.asarray(result.planar)[cell]), "three materials are not one plane"
    assert float(np.asarray(result.mean_epsilon)[cell]) == float(
        np.asarray(sampled.mean_epsilon)[cell]
    )
    # Every value in it is one of the three, so the midpoint sum is the right fallback.
    assert 1.0 < float(np.asarray(result.mean_epsilon)[cell]) < 5.0


_CYLINDER_RADIUS = 0.4217  # Off-grid at every resolution tested, so no face lands on a line.


def _cylinder(x, y, z):
    return np.where(x * x + z * z < _CYLINDER_RADIUS ** 2, E2, E1) + 0.0 * y


def _cylinder_exact_fill(grid, ix, iz, samples=200001):
    """Exact area fraction of one (x, z) voxel square inside the circle.

    Integrates the circle's height ``sqrt(R**2 - x**2)`` clipped to the voxel over x with
    a fine midpoint rule — a 1-D quadrature of a smooth integrand away from |x| = R, so
    it is good to ~1e-10 and is an independent derivation of the fill.
    """
    low_x, high_x = _voxel_span(grid, "Ey", 0, ix)
    low_z, high_z = _voxel_span(grid, "Ey", 2, iz)
    x = low_x + (np.arange(samples) + 0.5) * (high_x - low_x) / samples
    half = np.sqrt(np.clip(_CYLINDER_RADIUS ** 2 - x * x, 0.0, None))
    height = np.clip(np.minimum(high_z, half) - np.maximum(low_z, -half), 0.0, None)
    return float(height.mean() / (high_z - low_z))


def test_planar_residual_measures_curvature_and_gates_on_it():
    """A curved surface is not a plane, and the diagnostic has to say how far off it is.

    The fit linearizes the surface at the voxel centre, so a cylinder leaves
    ``(h_uu + h_vv) dx**2 / 64`` in ``planarity_residual``, first order in dx once
    expressed in cells. The prefactor is NOT ``1/R``: the height function's second
    derivative is ``1 / (R cos**3 theta)`` where theta is the tilt away from the
    dominant axis, so the worst voxel of a cylinder — the one at 45 degrees, where the
    dominant axis is about to switch — reads 3.9x the flat-face estimate. Measured
    3.55e-3 / 3.58e-3 / 1.79e-3 at resolutions 20 / 40 / 80.

    Resolution 20 is the case that shows the gate doing its job: the true residual there
    would be twice the resolution-40 value, above ``PLANE_RESIDUAL_TOLERANCE``, so the
    worst voxels are DECLINED rather than fitted and ``planar_fraction`` drops to 0.81.
    That is the mechanism by which too much curvature falls back to the sums instead of
    being linearized anyway.
    """
    residuals, shares = [], []
    for resolution in (20, 40, 80):
        grid = Grid(resolution=resolution, cell_size=(1.2, 0.2, 1.2))
        result = sm.smooth_inverse_epsilon(grid, _cylinder, "Ey", (16, 1, 16))
        residuals.append(result.planarity_residual)
        shares.append(result.planar_fraction)
        assert result.planarity_residual <= sm.PLANE_RESIDUAL_TOLERANCE
        # Between the flat-face estimate and the 45-degree worst case, both derived above.
        flat_face = (1.0 / _CYLINDER_RADIUS) * grid.dx / 64.0
        assert flat_face <= result.planarity_residual <= 5.0 * flat_face

    assert shares[0] == pytest.approx(0.81, abs=0.03), f"shares {shares}"
    assert shares[1] == 1.0 and shares[2] == 1.0
    # Once nothing is being clipped by the gate, halving dx halves the residual.
    assert residuals[1] / residuals[2] == pytest.approx(2.0, rel=0.1)


def test_planar_fill_of_a_cylinder_beats_the_sums_and_keeps_improving():
    """The curved case: the plane fit converges with dx, the midpoint sums do not.

    A cylinder is where the analytic fill is APPROXIMATE — the interface is a plane only
    to second order inside a voxel — so this is the honest measurement of what it costs
    there. Against the exact circular overlap, worst averaged voxel at ``s`` = 16:

        resolution        20        40        80
        sampled       9.6e-03   1.4e-02   2.7e-02
        planar        2.6e-03   2.7e-03   1.1e-03
        RMS, planar   1.5e-03   1.0e-03   3.1e-04

    Two readings. The sampled error does not converge at all — it is set by ``1/s``, and
    it gets WORSE with resolution because the same fill error meets a larger contrast
    per voxel. The planar error falls, and would fall faster but for the voxels the
    circle leaves through a LATERAL face, where the mean-height correction
    (:data:`smoothing.PLANE_CURVATURE_GAIN`) does not apply; those hold the worst-voxel
    figure up while the RMS keeps dropping.
    """
    errors = {}
    for resolution in (20, 40, 80):
        grid = Grid(resolution=resolution, cell_size=(1.2, 0.2, 1.2))
        fitted = np.asarray(sm.smooth_inverse_epsilon(grid, _cylinder, "Ey", (16, 1, 16)).planar)
        assert fitted.sum() > 40
        for rule in ("sampled", "planar"):
            result = sm.smooth_inverse_epsilon(
                grid, _cylinder, "Ey", (16, 1, 16), fill_rule=rule)
            means = np.asarray(result.mean_epsilon)
            relative = []
            for index in np.argwhere(fitted)[::5]:
                fill = _cylinder_exact_fill(grid, int(index[0]), int(index[2]))
                expected = fill * E2 + (1.0 - fill) * E1
                relative.append(abs(means[tuple(index)] - expected) / expected)
            errors[(rule, resolution)] = (float(np.max(relative)),
                                          float(np.sqrt(np.mean(np.square(relative)))))

    for resolution in (20, 40, 80):
        planar_worst = errors[("planar", resolution)][0]
        sampled_worst = errors[("sampled", resolution)][0]
        assert planar_worst < sampled_worst, (
            f"resolution {resolution}: planar {planar_worst:.2e} sampled {sampled_worst:.2e}"
        )
    # The advantage grows: 3.7x at resolution 20, 25x at 80.
    ratio = {r: errors[("sampled", r)][0] / errors[("planar", r)][0] for r in (20, 40, 80)}
    assert ratio[20] > 3.0 and ratio[80] > 15.0, f"ratios {ratio}"
    # And the planar RMS converges where the sampled one does not.
    planar_rms = [errors[("planar", r)][1] for r in (20, 40, 80)]
    sampled_rms = [errors[("sampled", r)][1] for r in (20, 40, 80)]
    assert planar_rms[2] < 0.25 * planar_rms[0], f"planar RMS {planar_rms}"
    assert sampled_rms[2] > 0.5 * sampled_rms[0], f"sampled RMS {sampled_rms}"


def test_refuses_a_bad_fill_rule_and_planar_with_an_array():
    """Loud on both: an unknown rule, and asking for the analytic fill without a callable."""
    grid = Grid(resolution=10, cell_size=(0.5, 0.5, 0.5))
    uniform = lambda x, y, z: 2.0 + 0.0 * (x + y + z)  # noqa: E731
    for bad in ("Planar", "exact", "", None, 1, "midpoint"):
        with pytest.raises(ValueError, match="fill_rule must be one of"):
            sm.smooth_inverse_epsilon(grid, uniform, "Ey", 4, fill_rule=bad)
    rates = (4, 4, 4)
    axes = sm.fine_coordinates(grid, "Ey", rates)
    values = np.broadcast_to(
        uniform(axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]),
        tuple(n * r for n, r in zip(grid.shape, rates)),
    )
    with pytest.raises(ValueError, match="needs a callable"):
        sm.smooth_inverse_epsilon(grid, values, "Ey", rates, fill_rule="planar")
    # ...but 'auto' resolves it, visibly, rather than raising or pretending.
    assert sm.resolve_fill_rule("auto", values) == "sampled"
    assert sm.resolve_fill_rule("auto", uniform) == "planar"
    assert sm.smooth_inverse_epsilon(grid, values, "Ey", rates).fill_rule == "sampled"


def test_driver_passes_the_fill_rule_through_and_installs_the_analytic_material():
    """Through ``set_epsilon_smoothed``: the default is analytic, and it reaches inv_eps.

    The driver is where the choice actually matters, so the two rules are run through it
    on the same y-invariant bar and the installed float32 arrays are compared. They must
    differ — otherwise the argument does nothing — and the analytic one must be the
    reciprocal of the exact overlap on the voxels it accepted.
    """
    grid_cell = (1.0, 0.2, 1.0)
    x_span, z_span = (-0.125, 0.245), (-0.2969, 0.3443)
    epsilon = _bar(x_span, z_span)
    reference = _bar_exact_diagonal(Grid(resolution=20, cell_size=grid_cell), x_span, z_span)
    installed = {}
    for rule in ("sampled", "planar", "auto"):
        driver = FdtdDriver(cell_size=grid_cell, resolution=20, force_complex_fields=True)
        try:
            report = driver.set_epsilon_smoothed(
                epsilon, "Ey", supersample=(16, 1, 16), fill_rule=rule)
            installed[rule] = np.array(np.asarray(driver.fields.inv_eps), copy=True)
            assert report.fill_rule == ("sampled" if rule == "sampled" else "planar")
            if rule != "sampled":
                accepted = np.asarray(report.planar)
                np.testing.assert_allclose(
                    installed[rule][accepted],
                    reference[accepted].astype(np.float32), rtol=2e-7, atol=0.0)
        finally:
            driver.close()
    np.testing.assert_array_equal(installed["auto"], installed["planar"])
    assert not np.array_equal(installed["sampled"], installed["planar"])


def test_interface_resolution_reports_the_worst_axis():
    grid = Grid(resolution=20, cell_size=(0.5, 0.5, 0.5))
    epsilon = _bar((-0.125, 0.245), (-0.2969, 0.3443))
    for rates, expected in (((16, 16, 16), grid.dx / 32), ((64, 1, 64), grid.dx / 2),
                            ((64, 64, 64), grid.dx / 128)):
        result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", rates)
        assert result.interface_resolution == pytest.approx(expected, rel=1e-12)


# --- 7. CPU-MEEP oracles ---------------------------------------------------------

_CHI1INV_ORACLE = '''"""MEEP's own smoothed chi1inv row for axis-aligned Blocks at off-grid faces."""
import json
import sys

import numpy as np
import meep as mp

case = json.loads(sys.argv[1])
output_path = sys.argv[2]
resolution = case["resolution"]
Lx, Ly, Lz = case["cell"]
geometry = [mp.Block(size=mp.Vector3(case["bar_x"][1] - case["bar_x"][0], mp.inf,
                                    case["bar_z"][1] - case["bar_z"][0]),
                     center=mp.Vector3(0.5 * sum(case["bar_x"]), 0, 0.5 * sum(case["bar_z"])),
                     material=mp.Medium(epsilon=case["eps"]))]
# A Block is MEEP's ANALYTIC path (meepgeom.cpp eff_chi1inv_matrix): the exact overlap
# of the pixel with the object and the exact object normal, i.e. the best answer MEEP
# has, not its sphere-quadrature fallback.
simulation = mp.Simulation(cell_size=mp.Vector3(Lx, Ly, Lz), resolution=resolution,
                           geometry=geometry, eps_averaging=case["averaging"],
                           k_point=mp.Vector3(0, 0, 0), force_complex_fields=True)
simulation.init_sim()
nx, ny, nz = (int(length * resolution + 0.5) for length in (Lx, Ly, Lz))
# MEEP's own axis origin in doubled coordinates: -(n - n % 2), vec.cpp icenter().
iox, ioy, ioz = (-(n - n % 2) for n in (nx, ny, nz))
rows = np.zeros((3, nx, ny, nz))
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            iloc = mp.ivec(iox + 2 * i, ioy + 2 * j + 1, ioz + 2 * k)  # Ey Yee point.
            for axis, direction in enumerate((mp.X, mp.Y, mp.Z)):
                rows[axis, i, j, k] = simulation.fields.get_chi1inv(mp.Ey, direction, iloc).real
np.save(output_path, rows)
'''


def _chi1inv_oracle(tmp_path, case):  # MEEP's smoothed Ey chi1inv row, in its own process.
    script_path = tmp_path / "meep_chi1inv_oracle.py"
    script_path.write_text(_CHI1INV_ORACLE, encoding="utf-8")
    output_path = tmp_path / f"chi1inv_{case['averaging']}.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(case), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP chi1inv oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


_CHI1INV_CASE = {
    "resolution": 20, "cell": [1.0, 0.2, 1.0], "eps": 4.0,
    "bar_x": [-0.125, 0.245], "bar_z": [-0.2969, 0.3443], "averaging": True,
}


@requires_meep
@skip_without_meep
def test_smoothed_chi1inv_matches_cpu_meep(tmp_path):
    """The strongest available pin: the array itself, against MEEP's ``get_chi1inv``.

    MEEP's Block path is analytic in both the fill and the normal, so this compares the
    module's answer with the best number MEEP can produce rather than with its own
    fallback. Measured on the bar's face voxels (the corners are a separate story
    below):

        fill_rule="sampled", s = 64      8.7e-03
        fill_rule="sampled", s = 256     2.1e-03
        fill_rule="planar",  any s       2.98e-08

    The planar row is the point, and 2.98e-08 is not this module's error: it is 2**-25,
    the largest relative rounding of a float32, and MEEP stores ``chi1inv`` in
    ``realnum`` = float32. The two sides are computing the SAME closed form — MEEP from
    the Block it was given, this module from the plane it measured out of the material
    function — and they agree to the last bit MEEP is able to report. (Against the
    separately derived exact overlap, in ``test_planar_fill_is_independent_of_supersample``,
    the same voxels read 1.4e-15.) The off-diagonals must be identically zero on both
    sides for this y-invariant bar.
    """
    reference = _chi1inv_oracle(tmp_path, _CHI1INV_CASE)
    grid = Grid(resolution=_CHI1INV_CASE["resolution"], cell_size=tuple(_CHI1INV_CASE["cell"]))
    epsilon = _bar(tuple(_CHI1INV_CASE["bar_x"]), tuple(_CHI1INV_CASE["bar_z"]),
                   inside=_CHI1INV_CASE["eps"], outside=1.0)
    assert reference.shape == (3,) + grid.shape
    # MEEP really did smooth: a staircased structure has two distinct values, no more.
    assert len(np.unique(np.round(reference[1], 12))) > 3
    # Off-diagonals: zero in MEEP's tensor for this y-invariant bar.
    assert np.max(np.abs(reference[0])) == 0.0 and np.max(np.abs(reference[2])) == 0.0

    previous = None
    for supersample, bound in ((64, 9.0e-3), (256, 2.3e-3)):
        result = sm.smooth_inverse_epsilon(
            grid, epsilon, "Ey", (supersample, 1, supersample), fill_rule="sampled")
        mine = np.asarray(result.row["y"])
        assert np.max(np.abs(np.asarray(result.row["x"]))) < 1e-15
        assert np.max(np.abs(np.asarray(result.row["z"]))) < 1e-15

        relative = np.abs(mine - reference[1]) / np.abs(reference[1])
        comparable = ~np.asarray(result.unresolved)
        worst = float(relative[comparable].max())
        assert worst < bound, f"s={supersample}: worst voxel {worst:.3e} exceeds {bound:.1e}"
        assert float(np.sqrt(np.mean(relative[comparable] ** 2))) < 0.2 * bound
        if previous is not None:
            assert worst < 0.5 * previous, "quadrupling s must cut the error at least fourfold"
        previous = worst

        # The one voxel that is not merely coarse: MEEP smooths the bar corner
        # analytically, the sphere shortcut declines it, and the gap is 50 %+.
        unresolved = np.asarray(result.unresolved)
        assert unresolved.any()
        assert float(relative[unresolved].max()) > 0.4
        assert result.unresolved_fraction < 0.05

    # And the analytic path, where the two implementations of the same closed form meet.
    for supersample in (16, 64):
        result = sm.smooth_inverse_epsilon(grid, epsilon, "Ey", (supersample, 1, supersample))
        relative = np.abs(np.asarray(result.row["y"]) - reference[1]) / np.abs(reference[1])
        fitted = np.asarray(result.planar)
        assert fitted.sum() > 100, "the bar's faces must all be fitted"
        # float32 is MEEP's storage, so 2**-25 is the floor this comparison can reach.
        assert float(relative[fitted].max()) <= np.float32(2.0 ** -24), (
            f"s={supersample}: worst fitted voxel {relative[fitted].max():.3e} against MEEP"
        )
        assert np.max(np.abs(np.asarray(result.row["x"]))) < 1e-15
        assert np.max(np.abs(np.asarray(result.row["z"]))) < 1e-15
        # The voxels NOT fitted are the bar's corners, and they are the only ones left
        # paying the sampling — which is what planar_fraction reports.
        declined = ~fitted & np.asarray(result.smoothed)
        assert float(relative[declined].max()) > 100 * float(relative[fitted].max())


_GRATING_CASE = {
    "period": 1.0, "cell_z": 4.0, "resolution": 15,
    "eps": 12.0, "bar_x": [-0.155, 0.315], "bar_z": [-0.3, 0.3],
    "kx": 0.4, "pml_cells": 10,
    "fcen": 1.0, "df": 0.4, "nfreq": 5, "source_fwidth": 1.0,
    "source_z": -1.0, "upstream_z": -1.15, "downstream_z": 1.0, "until": 22.0,
}

_GRATING_ORACLE = '''"""CPU-MEEP transmission through a SHARP bar grating, smoothing on and off."""
import json
import sys

import numpy as np
import meep as mp

case = json.loads(sys.argv[1])
output_path = sys.argv[2]
period, Lz, resolution = case["period"], case["cell_z"], case["resolution"]
spec = (case["fcen"], case["df"], case["nfreq"])
plane = mp.Vector3(period, period, 0)
results = {}


def bar():
    return [mp.Block(size=mp.Vector3(case["bar_x"][1] - case["bar_x"][0], mp.inf,
                                     case["bar_z"][1] - case["bar_z"][0]),
                     center=mp.Vector3(0.5 * sum(case["bar_x"]), 0, 0.5 * sum(case["bar_z"])),
                     material=mp.Medium(epsilon=case["eps"]))]


for averaging in (True, False):
    tag = "avg" if averaging else "raw"
    for name, with_bar in (("vacuum", False), ("grating", True)):
        arguments = dict(
            cell_size=mp.Vector3(period, period, Lz), resolution=resolution,
            boundary_layers=[mp.PML(case["pml_cells"] / resolution, direction=mp.Z)],
            sources=[mp.Source(mp.GaussianSource(case["fcen"], fwidth=case["source_fwidth"]),
                               component=mp.Ey, center=mp.Vector3(0, 0, case["source_z"]),
                               size=plane)],
            k_point=mp.Vector3(case["kx"], 0, 0), force_complex_fields=True,
            eps_averaging=averaging)
        if with_bar:
            arguments["geometry"] = bar()
        simulation = mp.Simulation(**arguments)
        downstream = simulation.add_flux(
            *spec, mp.FluxRegion(center=mp.Vector3(0, 0, case["downstream_z"]), size=plane,
                                 direction=mp.Z), decimation_factor=1)
        upstream = simulation.add_flux(
            *spec, mp.FluxRegion(center=mp.Vector3(0, 0, case["upstream_z"]), size=plane,
                                 direction=mp.Z), decimation_factor=1)
        simulation.run(until=case["until"])
        results[f"{tag}_{name}_down"] = np.asarray(mp.get_fluxes(downstream))
        results[f"{tag}_{name}_up"] = np.asarray(mp.get_fluxes(upstream))
        results["freqs"] = np.asarray(mp.get_flux_freqs(downstream))

# The staircased array the engine's own set_epsilon path is fed: point samples at the
# INTEGER Yee positions inv_eps multiplies, measured from MEEP's axis origin.
nx, ny, nz = (int(length * resolution + 0.5) for length in (period, period, Lz))
dx = 1.0 / resolution
xs = -(nx - nx % 2) * dx / 2 + np.arange(nx) * dx
zs = -(nz - nz % 2) * dx / 2 + np.arange(nz) * dx
inside = (((xs >= case["bar_x"][0]) & (xs <= case["bar_x"][1]))[:, None]
          & ((zs >= case["bar_z"][0]) & (zs <= case["bar_z"][1]))[None, :])
results["eps_staircase"] = np.repeat(
    np.where(inside, case["eps"], 1.0)[:, None, :], ny, axis=1)
np.savez(output_path, **results)
'''


@pytest.fixture(scope="module")
def grating_oracle(tmp_path_factory):  # One CPU-MEEP process for the sharp-grating case.
    tmp_path = tmp_path_factory.mktemp("sharp_grating_oracle")
    script_path = tmp_path / "meep_grating_oracle.py"
    script_path.write_text(_GRATING_ORACLE, encoding="utf-8")
    output_path = tmp_path / "sharp_grating.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(_GRATING_CASE), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=1800,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP grating oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _grating_transmission(epsilon=None, supersample=None, fill_rule="auto"):
    """One engine run of the sharp-grating case; returns (T, report)."""
    case = _GRATING_CASE
    driver = FdtdDriver(
        cell_size=(case["period"], case["period"], case["cell_z"]),
        resolution=case["resolution"], force_complex_fields=True,
        k_point=(case["kx"], 0.0, 0.0),
    )
    try:
        driver.setup_pml({"z": case["pml_cells"]})
        report = None
        if supersample is not None:
            report = driver.set_epsilon_smoothed(
                _bar(tuple(case["bar_x"]), tuple(case["bar_z"]),
                     inside=case["eps"], outside=1.0),
                "Ey", supersample=supersample, fill_rule=fill_rule,
            )
        elif epsilon is not None:
            driver.set_epsilon(epsilon)
        driver.add_source({
            "component": "Ey", "source_type": "gaussian", "frequency": case["fcen"],
            "fwidth": case["source_fwidth"], "center": (0.0, 0.0, case["source_z"]),
            "size": (case["period"], case["period"], 0.0),
        })
        plane = dict(fcen=case["fcen"], df=case["df"], nfreq=case["nfreq"],
                     size=(case["period"], case["period"], 0.0), direction=2)
        downstream = driver.add_flux_monitor(
            center=(0.0, 0.0, case["downstream_z"]), **plane)
        driver.run(until=case["until"])
        return np.asarray(downstream.get_flux_spectrum(), dtype=float), report
    finally:
        driver.close()


def _worst_relative(candidate, reference):
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert np.all(np.abs(reference) > 0.0)
    return float(np.max(np.abs(candidate - reference) / np.abs(reference)))


@requires_meep
@skip_without_meep
def test_grating_gap_against_cpu_meep_closes_with_smoothing(grating_oracle):
    """The measurement this feature exists for, before and after, on both fill rules.

    A sharp eps = 12 bar grating, Bloch at kx = 0.4, PML in z, Ey drive — the
    configuration the scalar reduction is exact for. Transmission normalised by each
    engine's own empty-cell run, so the comparison is of the material treatment and
    nothing else. Worst of five frequencies against CPU MEEP's default
    (``eps_averaging=True``):

        engine point-sampled (no smoothing)          3.90e-01

        supersample                8      16      32      64     128     256
        fill_rule="sampled"     2.3e-1  3.4e-2  5.8e-2  8.4e-3  1.5e-2  2.1e-3
        fill_rule="planar"      1.5e-6  1.5e-6  1.5e-6  1.5e-6  1.5e-6  1.5e-6

    The sampled row is the old behaviour and the reason this file used to pin "the
    default does not deliver": first order in 1/s with a sawtooth, so 16 left about as
    much error as the staircasing it replaced and even 256 stopped at 2.1e-3. The planar
    row is flat, because the fill fraction no longer comes from the lattice at all — and
    1.5e-6 is at the engine's own floor for this case: its agreement with MEEP's
    UNSMOOTHED run, the control below, is 1.9e-06. The smoothing has stopped being the
    limiting error of this configuration.

    The planar pre-pass is also CHEAPER than the sampled one it replaces at the same s
    (0.023 s against 0.038 s at s = 8 on this grid): the plane fit runs over the
    interface only, which is O(N**(2/3)) of the grid.
    """
    reference = grating_oracle
    smoothed_meep = reference["avg_grating_down"] / reference["avg_vacuum_down"]
    staircased_meep = reference["raw_grating_down"] / reference["raw_vacuum_down"]
    # The grating has to do something, or every ratio below compares 1.0 with 1.0.
    assert np.max(np.abs(smoothed_meep - 1.0)) > 0.2, "this grating barely diffracts"
    assert _worst_relative(smoothed_meep, staircased_meep) > 0.3, (
        "MEEP's own smoothed and staircased answers must differ, or there is no gap to close"
    )

    vacuum, _ = _grating_transmission()
    # Control: against MEEP's UNSMOOTHED run the engine is already exact, so anything
    # this test measures afterwards is the material treatment alone.
    staircased, _ = _grating_transmission(epsilon=reference["eps_staircase"])
    control = _worst_relative(staircased / vacuum, staircased_meep)
    assert control < 1e-4, f"the engine core does not reproduce MEEP point-sampled: {control:.2e}"

    before = _worst_relative(staircased / vacuum, smoothed_meep)
    assert before > 0.2, f"the staircased gap must be large to be worth closing: {before:.2e}"

    sampled, planar = {}, {}
    for supersample in (16, 64, 256):
        rates = (supersample, 1, supersample)
        transmission, report = _grating_transmission(supersample=rates, fill_rule="sampled")
        sampled[supersample] = _worst_relative(transmission / vacuum, smoothed_meep)
        assert report.fill_rule == "sampled" and report.planar_fraction == 0.0
        transmission, report = _grating_transmission(supersample=rates)
        planar[supersample] = _worst_relative(transmission / vacuum, smoothed_meep)
        assert report.fill_rule == "planar"
        # Every averaged voxel of this grating is a flat face; the corners are declined
        # by MEEP's 8-point shortcut before the plane fit ever sees them.
        assert report.planar_fraction == 1.0
        assert report.planarity_residual == 0.0
        assert report.offdiagonal_magnitude < 1e-15
        assert report.unresolved_fraction < 0.10

    # The sampled ladder, unchanged: still first order, still short of MEEP at any s a
    # caller would pay for. Kept because it is what the planar rule has to beat.
    assert sampled[16] == pytest.approx(3.4e-2, rel=0.2), f"measured {sampled[16]:.3e}"
    assert sampled[256] == pytest.approx(2.1e-3, rel=0.2), f"measured {sampled[256]:.3e}"
    assert sampled[16] > 1e-2

    # And the planar rule: three orders below the sampled default, at the engine's own
    # floor, and INDEPENDENT of the supersample — which is the property being bought.
    for supersample in (16, 64, 256):
        assert planar[supersample] < 5e-6, (
            f"planar at s={supersample} reached {planar[supersample]:.2e}"
        )
        assert planar[supersample] < sampled[supersample] / 100.0
    spread = max(planar.values()) / min(planar.values())
    assert spread < 1.05, f"the planar answer must not depend on s, spread {spread:.3f}"
    assert max(planar.values()) < before / 10000.0
    # It is at the core's own reproduction floor, not merely small.
    assert max(planar.values()) < 5.0 * control


@requires_meep
@skip_without_meep
def test_smoothing_survives_pml_bloch_and_odd_cell_counts(grating_oracle):
    """Smoothing combined with the features it has to coexist with.

    The grating case already carries a z-only PML, a Bloch phase on an unabsorbed
    axis, and transverse axes of 15 cells — odd, where the axis origin is half a cell
    from -L/2 and every historical registration defect in this engine has lived. This
    checks that the smoothed material is registered on MEEP's origin and not on -L/2:
    the same structure smoothed about the wrong origin is two orders worse, so the
    measurement can tell them apart.

    Run at the SHIPPED settings — default fill rule, default supersample — because the
    analytic fill introduces a second thing that could be registered wrongly: the
    bisection columns start from the voxel centres and reach past the periodic wrap, so
    an origin error would move them too.
    """
    case = _GRATING_CASE
    assert int(case["period"] * case["resolution"] + 0.5) % 2 == 1, "keep an odd count here"
    smoothed_meep = grating_oracle["avg_grating_down"] / grating_oracle["avg_vacuum_down"]
    vacuum, _ = _grating_transmission()

    correct, report = _grating_transmission(supersample=sm.DEFAULT_SUPERSAMPLE)
    right_origin = _worst_relative(correct / vacuum, smoothed_meep)
    assert right_origin < 5e-6, f"odd counts + PML + Bloch measured {right_origin:.2e}"
    assert report.planar_fraction == 1.0

    # The same smoothing, deliberately registered from -L/2: the defect class that cost
    # 6.6e-02 on this grating before Grid.origin_doubled rounded the count down to even.
    grid = Grid(resolution=case["resolution"],
                cell_size=(case["period"], case["period"], case["cell_z"]))
    shift = grid.axis_origin(0) + case["period"] / 2.0
    assert abs(shift) == pytest.approx(0.5 * grid.dx, rel=1e-9), (
        "an odd-count axis must sit half a cell above -L/2, or this control is vacuous"
    )
    offset_bar = _bar((case["bar_x"][0] + shift, case["bar_x"][1] + shift),
                      (case["bar_z"][0], case["bar_z"][1]),
                      inside=case["eps"], outside=1.0)
    driver = FdtdDriver(
        cell_size=(case["period"], case["period"], case["cell_z"]),
        resolution=case["resolution"], force_complex_fields=True,
        k_point=(case["kx"], 0.0, 0.0))
    try:
        driver.setup_pml({"z": case["pml_cells"]})
        driver.set_epsilon_smoothed(offset_bar, "Ey", supersample=sm.DEFAULT_SUPERSAMPLE)
        driver.add_source({
            "component": "Ey", "source_type": "gaussian", "frequency": case["fcen"],
            "fwidth": case["source_fwidth"], "center": (0.0, 0.0, case["source_z"]),
            "size": (case["period"], case["period"], 0.0)})
        monitor = driver.add_flux_monitor(
            center=(0.0, 0.0, case["downstream_z"]), fcen=case["fcen"], df=case["df"],
            nfreq=case["nfreq"], size=(case["period"], case["period"], 0.0), direction=2)
        driver.run(until=case["until"])
        offset = np.asarray(monitor.get_flux_spectrum(), dtype=float)
    finally:
        driver.close()
    wrong_origin = _worst_relative(offset / vacuum, smoothed_meep)
    assert wrong_origin > 20.0 * right_origin, (
        f"the comparison cannot see a half-cell registration error: correct "
        f"{right_origin:.2e}, shifted {wrong_origin:.2e}"
    )


# --- 8. Convergence order: the decisive test ------------------------------------

_SLAB_INDEX = 2.0
# 2 pi f n d = 2.26 pi at f = 1: the steepest flank of the Fabry-Perot fringe, where the
# transmittance is most sensitive to the slab THICKNESS and so to where the staircase
# puts its faces. The exact value matters for a second reason: d / dx must not be near a
# whole number of cells at any resolution tested, because a slab that happens to be an
# integer number of cells thick is staircased EXACTLY RIGHT and its error collapses.
# frac(d/dx) here is 0.30 / 0.82 / 0.60 / 0.64 at resolutions 20 / 28 / 40 / 56.
_SLAB_THICKNESS = 0.565


def _slab_transmittance(frequencies, thickness):
    """Analytic Airy transmittance of a lossless slab — the independent truth."""
    contrast = (_SLAB_INDEX ** 2 - 1.0) ** 2 / (4.0 * _SLAB_INDEX ** 2)
    phase = 2.0 * np.pi * np.asarray(frequencies) * _SLAB_INDEX * thickness
    return 1.0 / (1.0 + contrast * np.sin(phase) ** 2)


def _slab_run(resolution, mode, low=None, thickness=None, supersample=None,
              fill_rule="auto"):
    """Transmittance of a slab with OFF-GRID faces, or of the empty cell.

    The transverse extent is four cells at every resolution: the run is uniform in x
    and y (k = 0, a sheet source spanning the full periodic cross section), so this is
    a 1-D problem and paying for a wide cell would only buy cost.
    """
    transverse, cell_z, until = 4.0 / resolution, 5.0, 30.0

    def slab(x, y, z):
        return np.where((z >= low) & (z <= low + thickness), _SLAB_INDEX ** 2, 1.0) + 0.0 * (x + y)

    driver = FdtdDriver(cell_size=(transverse, transverse, cell_z), resolution=resolution,
                        force_complex_fields=True)
    try:
        driver.setup_pml({"z": int(round(resolution))})  # 1.0 length unit, held fixed.
        if mode == "smooth":
            driver.set_epsilon_smoothed(slab, "Ey", supersample=(1, 1, supersample),
                                        fill_rule=fill_rule)
        elif mode == "stair":
            driver.set_epsilon(_staircased(driver.grid, slab))
        driver.add_source({
            "component": "Ey", "source_type": "gaussian", "frequency": 1.0, "fwidth": 0.6,
            "center": (0.0, 0.0, -1.2), "size": (transverse, transverse, 0.0)})
        monitor = driver.add_flux_monitor(
            center=(0.0, 0.0, 1.2), fcen=1.0, df=0.0, nfreq=1,
            size=(transverse, transverse, 0.0), direction=2)
        driver.run(until=until)
        return (np.asarray(monitor.get_flux_spectrum(), dtype=float),
                np.asarray(monitor.frequencies, dtype=float))
    finally:
        driver.close()


def test_smoothing_restores_second_order_convergence():
    """The claim that justifies the whole module, measured against an analytic answer.

    A staircased interface moves by up to dx/2, an O(dx) perturbation of the GEOMETRY,
    so any observable that depends on where the interface is converges only first
    order. Subpixel smoothing recovers the true position to O(dx**2) and the
    observable follows. The observable is the transmittance of a slab of index 2 whose
    faces never land on a grid line, on the steepest flank of its Fabry-Perot fringe,
    against the closed-form Airy formula — an independent truth, not another
    discretization, so this measures the engine's order and not agreement with MEEP.

    The error is an RMS over four sub-cell placements of the slab, because the
    staircased error is a SAWTOOTH in the face position — a single placement reads any
    order you like.

    Measured (RMS relative error in transmittance, four placements):

        resolution                20        28        40        56      | 80
        staircased              9.8e-02   9.4e-02   6.9e-02   4.7e-02   | 9.0e-03
        planar,   s = 32        3.9e-02   2.0e-02   1.0e-02   4.8e-03
        sampled,  s = 256       3.8e-02   2.0e-02   9.9e-03   5.1e-03   | 2.5e-03
        sampled,  s = 16        4.1e-02   1.9e-02   1.3e-02   3.6e-03

        fitted order over 20..56:  staircased 0.74
                                   planar s=32 2.01  (pairwise 1.95, 1.90, 2.24)
                                   sampled s=256 1.96 (pairwise 1.83, 2.02, 1.99)
                                   sampled s=16 2.22 (pairwise 2.31, 1.05, 3.79)

    So the geometry is what limited it, and once it is right the residual is the
    scheme's own O(dx**2) numerical dispersion. The run below is the SHIPPED
    configuration — the default fill rule at the default supersample — and it reaches
    the order that used to need ``s`` = 256 of sampling. The sampled s = 16 row is the
    warning against reading a single fit: its 2.22 is the 1/s sawtooth passing through,
    not convergence, which its pairwise 2.31 / 1.05 / 3.79 gives away.

    Resolution 80 is measured and shown but excluded from the FIT, honestly rather than
    quietly: the staircased error collapses there (9.0e-03, which alone would read as
    order 4.6) because the O(dx) geometric error and the O(dx**2) dispersion error
    happen to cancel at that combination, and including it drags the staircased fit to
    1.59. The smoothed fit does not care — 1.96 over 20..56, 1.96 over all five.

    A last measured caution about ``supersample`` in the planar rule: it is still what
    DETECTS an interface, and at ``s`` = 8 this same measurement degrades (pairwise
    1.09, 2.58, 2.24) because a slab face landing in the outer 1/16 of a voxel leaves
    every fine sample equal and the voxel is never averaged. That, not accuracy, is what
    the default of 32 is sized for.
    """
    resolutions = (20, 28, 40, 56)
    offsets = (0.0, 0.25, 0.5, 0.75)
    steps = 1.0 / np.array(resolutions, dtype=float)
    errors = {"stair": [], "smooth": []}
    for resolution in resolutions:
        vacuum, frequencies = _slab_run(resolution, "vacuum", low=0.0,
                                        thickness=_SLAB_THICKNESS)
        assert float(vacuum[0]) > 0.0, "an empty-cell run with no flux makes every ratio absurd"
        exact = _slab_transmittance(frequencies, _SLAB_THICKNESS)
        # On the steep flank, or the observable barely feels the interface position.
        assert 0.6 < float(exact[0]) < 0.9, f"T_exact = {float(exact[0]):.4f} is off the flank"
        for mode, supersample in (("stair", None), ("smooth", sm.DEFAULT_SUPERSAMPLE)):
            batch = []
            for offset in offsets:
                # A low face that is off-grid at every resolution, slid across one cell.
                low = -0.2903 + offset / resolution
                flux, _ = _slab_run(resolution, mode, low=low, thickness=_SLAB_THICKNESS,
                                    supersample=supersample)
                batch.append(np.abs(flux / vacuum - exact) / exact)
            errors[mode].append(float(np.sqrt(np.mean(np.square(np.concatenate(batch))))))

    staircased = np.array(errors["stair"])
    smoothed = np.array(errors["smooth"])
    order = {name: float(np.polyfit(np.log(steps), np.log(values), 1)[0])
             for name, values in (("stair", staircased), ("smooth", smoothed))}

    # The whole claim, in two numbers. The bands are wide enough for the residual
    # sawtooth and the dispersion term but they cannot overlap.
    assert 0.5 < order["stair"] < 1.4, (
        f"staircased order {order['stair']:.2f} (expected ~1), errors {staircased}")
    assert order["smooth"] > 1.75, (
        f"smoothed order {order['smooth']:.2f} — smoothing must restore second order, "
        f"errors {smoothed}")
    assert order["smooth"] > order["stair"] + 0.6
    # And better in absolute terms at every resolution, not merely steeper.
    assert np.all(smoothed < staircased)
    assert smoothed[-1] < staircased[-1] / 5.0
