"""Unit tests for ``mp.Absorber``'s graded conductivity (meep_gpu.absorber).

The numbers pinned here are MEEP's own, in two forms: constants transcribed from
``meepgeom.cpp`` (the profile table) and quantities MEASURED against pristine MEEP
1.33.0 single precision while building this module (the two ``sigma_max`` values,
which came out of the reconstruction probe that reproduces ``mp.Absorber``
bit-identically through a ``material_function``).

Everything runs on NumPy — no MEEP, no CuPy — so a machine with neither still
verifies the arithmetic. Field parity against MEEP itself lives in
``test_driver_vs_meep.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from .absorber import (
    CONDUCTIVITY_COMPONENTS,
    DEFAULT_R_ASYMPTOTIC,
    QUADRATIC_PROFILE_INTEGRAL,
    AbsorberLayer,
    absorber_conductivity,
    absorber_profile,
    _owned_indices,
)
from .fields import IYEE_SHIFTS
from .grid import Grid, Mirror


def _grid(resolution: float, cell, symmetry=()) -> Grid:
    dimensions = 2 if cell[2] == 0.0 else 3
    return Grid(resolution=resolution, cell_size=cell, symmetry=symmetry,
                dimensions=dimensions)


# --------------------------------------------------------------------------------
# The profile table: meepgeom.cpp:744-763.
# --------------------------------------------------------------------------------

def test_profile_table_matches_meeps_own_constants():
    """N and sigma_max for the two configurations measured bit-identical against MEEP.

    L = 1 at resolution 20 gives N = 40 and sigma_max = 25.904082296183013; L = 0.7 at
    resolution 13 — 9.1 cells, deliberately not a whole number — gives N = 18 and
    37.005831851690026. Both numbers were read off the MEEP-reproducing probe, so a
    change to the formula that still "looks quadratic" is caught here.
    """
    count, profile = absorber_profile(1.0, 20.0)
    assert count == 40
    assert profile[-1] == pytest.approx(25.904082296183013, rel=1e-14)
    count, profile = absorber_profile(0.7, 13.0)
    assert count == 18
    assert profile[-1] == pytest.approx(37.005831851690026, rel=1e-14)


def test_profile_samples_at_half_pixels_not_pixels():
    """``dx = gv.inva * 0.5``: the table is twice as fine as the grid.

    Reading it as a whole pixel halves N, which coarsens every interpolated value
    without changing sigma_max — a defect no end-point check would see.
    """
    count, _ = absorber_profile(2.0, 10.0)
    assert count == 40  # 2 * 2.0 * 10, not 2.0 * 10.


def test_profile_is_quadratic_from_zero_with_meeps_prefactor():
    count, profile = absorber_profile(1.5, 16.0, R_asymptotic=1e-12)
    prefac = -math.log(1e-12) / (4.0 * 1.5 * QUADRATIC_PROFILE_INTEGRAL)
    assert profile[0] == 0.0
    assert profile[-1] == pytest.approx(prefac, rel=1e-14)
    index = np.arange(count + 1)
    np.testing.assert_allclose(profile, prefac * (index / count) ** 2, rtol=1e-14)


def test_profile_thickness_is_not_rounded_to_a_cell():
    """Two thicknesses inside one half-pixel bin are DIFFERENT absorbers.

    ``prefac`` and the lookup's ``ui = N*(x - edge)/L`` divide by the RAW thickness,
    so 0.70 and 0.71 at resolution 13 share N = 18 and still differ. Rounding L is the
    plausible-looking correction that collapses them onto one layer.
    """
    count_a, profile_a = absorber_profile(0.70, 13.0)
    count_b, profile_b = absorber_profile(0.71, 13.0)
    assert count_a == count_b == 18
    assert profile_a[-1] != profile_b[-1]


def test_profile_refuses_a_layer_thinner_than_one_half_pixel():
    with pytest.raises(ValueError, match="N = 0"):
        absorber_profile(0.01, 10.0)


# --------------------------------------------------------------------------------
# The lookup: meepgeom.cpp:1596-1625.
# --------------------------------------------------------------------------------

def _reference_sigma(coordinate, thickness, half_length, resolution,
                     side, R=DEFAULT_R_ASYMPTOTIC):
    """meepgeom.cpp:1596-1625 rewritten as a scalar loop — an independent second reading."""
    count, profile = absorber_profile(thickness, resolution, R)
    if side == "high":
        edge = half_length - thickness
        if coordinate < edge:
            return 0.0
        argument = count * (coordinate - edge) / thickness
    else:
        edge = thickness - half_length
        if coordinate > edge:
            return 0.0
        argument = count * (edge - coordinate) / thickness
    index = int(argument)
    if index >= count:
        return profile[count]
    fraction = argument - index
    return profile[index] * (1.0 - fraction) + profile[index + 1] * fraction


def test_lookup_matches_a_scalar_transcription_cell_by_cell():
    grid = _grid(13.0, (8.0, 8.0, 0.0))
    layers = [AbsorberLayer(0.7, axis=0, side="high"), AbsorberLayer(0.7, axis=0, side="low")]
    volumes = absorber_conductivity(grid, layers)
    sigma = np.asarray(volumes["Dz"])  # Dz has Yee shift 0 along x.
    half_length = grid.Lx / 2.0
    origin = grid.axis_origin(0)
    owned = _owned_indices(grid, 0, 0)
    for index in range(grid.nx):
        position = origin + int(owned[index]) * grid.dx  # shift 0 on x for Dz.
        expected = (_reference_sigma(position, 0.7, half_length, 13.0, "high")
                    + _reference_sigma(position, 0.7, half_length, 13.0, "low"))
        assert sigma[index, 0, 0] == pytest.approx(expected, rel=1e-13, abs=1e-15)


def test_the_six_components_are_sampled_at_their_own_yee_positions():
    """Dx and Dz differ by half a cell in x, and their sigma must differ by the same.

    One shared volume is the defect this guards: measured, snapping the sample position
    costs 1.51e-02 (node) to 1.07e-01 (centre) relative Linf on a 2-D absorber run.
    """
    grid = _grid(13.0, (8.0, 8.0, 0.0))
    volumes = absorber_conductivity(grid, [AbsorberLayer(0.7, axis=0, side="high")])
    assert set(volumes) == set(CONDUCTIVITY_COMPONENTS)
    half_length = grid.Lx / 2.0
    origin = grid.axis_origin(0)
    for component in CONDUCTIVITY_COMPONENTS:
        shift = IYEE_SHIFTS[component][0]
        owned = _owned_indices(grid, 0, shift)
        sigma = np.asarray(volumes[component])
        for index in range(grid.nx):
            position = origin + (int(owned[index]) + 0.5 * shift) * grid.dx
            expected = _reference_sigma(position, 0.7, half_length, 13.0, "high")
            assert sigma[index, 0, 0] == pytest.approx(expected, rel=1e-13, abs=1e-15)
    # And the two sub-lattices really are different arrays, not the same one twice.
    assert not np.allclose(np.asarray(volumes["Dx"]), np.asarray(volumes["Dz"]))


def test_electric_and_magnetic_volumes_are_both_built():
    """MEEP's FOR_D_AND_B loop, structure.cpp:377-379 — six volumes, not three.

    Dropping the magnetic half measured 4.08e-03 (1-D) and 6.46e-01 (2-D) relative
    Linf against MEEP, three to six orders above the parity band.
    """
    grid = _grid(13.0, (8.0, 8.0, 0.0))
    volumes = absorber_conductivity(grid, [AbsorberLayer(0.7, axis=0, side="high")])
    for component in ("Bx", "By", "Bz"):
        assert float(np.max(np.asarray(volumes[component]))) > 0.0
    # Bz shares Dx's x sub-lattice (both shift 1), so their x profiles coincide exactly.
    np.testing.assert_array_equal(np.asarray(volumes["Bz"]), np.asarray(volumes["Dx"]))


def test_faces_and_layers_accumulate_rather_than_replace():
    """meepgeom.cpp's ``cond_val +=``: a corner carries both axes' ramps."""
    grid = _grid(10.0, (4.0, 4.0, 0.0))
    x_only = absorber_conductivity(grid, [AbsorberLayer(1.0, axis=0, side="high")])
    y_only = absorber_conductivity(grid, [AbsorberLayer(1.0, axis=1, side="high")])
    both = absorber_conductivity(grid, [AbsorberLayer(1.0, axis=0, side="high"),
                                        AbsorberLayer(1.0, axis=1, side="high")])
    np.testing.assert_allclose(np.asarray(both["Dz"]),
                               np.asarray(x_only["Dz"]) + np.asarray(y_only["Dz"]),
                               rtol=1e-13, atol=1e-15)


def test_interior_is_exactly_zero():
    grid = _grid(10.0, (4.0, 4.0, 0.0))
    volumes = absorber_conductivity(grid, [AbsorberLayer(1.0, axis=0, side="high"),
                                           AbsorberLayer(1.0, axis=0, side="low")])
    sigma = np.asarray(volumes["Dz"])
    interior = sigma[grid.nx // 2, :, :]
    assert float(np.max(np.abs(interior))) == 0.0


def test_meeps_side_enum_puts_each_layer_on_the_face_it_names():
    """A swapped ``boundary_side`` enum (MEEP's is High = 0, Low = 1) is visible here.

    On the INTEGER sub-lattice the two faces are not even index-mirrors of each other:
    the low face's first stored row sits ON the wall (sigma = sigma_max) while the high
    face's last stored row is half a cell short of it, so a 40-cell axis ends up with
    10 nonzero cells low against 9 high. Dz is used for that reason — Dx sits at
    half-integers, where the two DO mirror and a swap would be invisible in the shape.
    """
    # A METALLIC axis, so the seam rule below does not enter: on a wrapping axis
    # index 0 is the far face's ghost and both spellings light it up.
    grid = Grid(resolution=10.0, cell_size=(4.0, 4.0, 0.0), dimensions=2,
                boundaries={"x": "metallic"})
    low = np.asarray(absorber_conductivity(
        grid, [AbsorberLayer(1.0, axis=0, side="low")])["Dz"])[:, 0, 0]
    high = np.asarray(absorber_conductivity(
        grid, [AbsorberLayer(1.0, axis=0, side="high")])["Dz"])[:, 0, 0]
    assert float(np.max(low[:5])) > 0.0
    assert float(np.max(low[-5:])) == 0.0
    assert float(np.max(high[-5:])) > 0.0
    assert float(np.max(high[:5])) == 0.0
    assert int(np.count_nonzero(low)) == 10 and int(np.count_nonzero(high)) == 9
    assert low[0] == pytest.approx(absorber_profile(1.0, 10.0)[1][-1], rel=1e-13)
    assert not np.allclose(low, high[::-1])


def test_the_seam_cell_of_a_wrapping_axis_takes_the_far_faces_sigma():
    """MEEP's ``little_owned_corner0`` puts the integer sub-lattice's seam at index N.

    ``little_corner + 2 - iyee_shift(c)`` (vec.hpp:1102-1104): ``step_curl`` runs from
    ``i >= 1 - shift``, so an INTEGER component's index 0 is never stepped —
    ``step_boundaries`` fills it from the owned copy one lattice vector up. The
    coefficient that governs the seam is therefore the one at ``+L/2``.

    Invisible for anything symmetric, which is what makes it dangerous. Measured on a
    low-only ``mp.Absorber(1)`` over a periodic 6-unit cell against CPU MEEP: taking
    index 0's own sigma drifts 2.9e-06 (t = 2) -> 1.7e-04 (t = 4) -> 1.5e-01 (t = 8);
    taking index N's holds 2.4e-07 / 3.4e-07 / 4.0e-07. The high-only layer goes
    2.8e-01 -> 3.2e-07. Two-sided layers are bit-identical either way.
    """
    wrapping = _grid(10.0, (4.0, 4.0, 0.0))
    walled = Grid(resolution=10.0, cell_size=(4.0, 4.0, 0.0), dimensions=2,
                  boundaries={"x": "metallic"})
    peak = absorber_profile(1.0, 10.0)[1][-1]
    for grid, expected_seam in ((wrapping, peak), (walled, 0.0)):
        # A HIGH-only layer: nothing physically graded near x = -L/2 at all.
        sigma = np.asarray(absorber_conductivity(
            grid, [AbsorberLayer(1.0, axis=0, side="high")])["Dz"])[:, 0, 0]
        assert sigma[0] == pytest.approx(expected_seam, rel=1e-13, abs=1e-15)
        assert float(np.max(sigma[1:6])) == 0.0  # Only the seam plane, not a shifted layer.
    # Half-integer components are stepped from index 0 and take their own coordinate.
    half_integer = np.asarray(absorber_conductivity(
        wrapping, [AbsorberLayer(1.0, axis=0, side="high")])["Dx"])[:, 0, 0]
    assert float(half_integer[0]) == 0.0
    # A two-sided layer is bit-identical on both grids: the two ends agree.
    both = [AbsorberLayer(1.0, axis=0, side="high"), AbsorberLayer(1.0, axis=0, side="low")]
    np.testing.assert_array_equal(
        np.asarray(absorber_conductivity(wrapping, both)["Dz"]),
        np.asarray(absorber_conductivity(walled, both)["Dz"]))


def test_a_folded_axis_grades_only_its_stored_half():
    """A mirror-folded axis's low face is the IMAGE of its high face, not a second layer.

    The stored cells start one cell below the mirror plane (``origin_doubled`` = -2), so
    MEEP's low-side test ``x <= L - geometry_edge`` is never satisfied there. Nothing has
    to be special-cased: ``geometry_edge`` stays the FULL cell's half-length.
    """
    grid = _grid(10.0, (4.0, 4.0, 0.0), symmetry=(Mirror("y"),))
    volumes = absorber_conductivity(grid, [AbsorberLayer(1.0, axis=1, side="high")])
    sigma = np.asarray(volumes["Dz"])
    assert grid.ny < grid.ny_full
    assert float(np.max(sigma[:, 0, :])) == 0.0        # The mirror plane absorbs nothing.
    assert float(np.max(sigma[:, -1, :])) > 0.0        # The far face does.


def test_geometry_edge_is_the_full_cell_not_the_stored_half():
    """A folded run grades against the FULL cell's half-length, row by row.

    ``geometry_edge`` comes off ``s->user_volume`` (meepgeom.cpp:1955-1975), which is
    the cell the user declared — MEEP halves the CHUNK, not the lattice. Taking the
    stored half instead would move the graded region a full quarter-cell inwards, which
    still looks like a perfectly reasonable absorber.
    """
    folded = _grid(10.0, (4.0, 4.0, 0.0), symmetry=(Mirror("y"),))
    sigma = np.asarray(absorber_conductivity(
        folded, [AbsorberLayer(1.0, axis=1, side="high")])["Dz"])[0, :, 0]
    half_length = 4.0 / 2.0  # The FULL cell, not folded.ny * folded.dx / 2.
    origin = folded.axis_origin(1)
    for index in range(folded.ny):
        position = origin + index * folded.dx  # Dz has y-shift 0.
        expected = _reference_sigma(position, 1.0, half_length, 10.0, "high")
        assert sigma[index] == pytest.approx(expected, rel=1e-13, abs=1e-15)
    assert float(np.max(sigma)) > 0.0  # The far face really is graded.


def test_returned_volumes_are_float64_so_a_material_sum_rounds_once():
    grid = _grid(10.0, (4.0, 4.0, 0.0))
    volumes = absorber_conductivity(grid, [AbsorberLayer(1.0, axis=0, side="high")])
    for component in CONDUCTIVITY_COMPONENTS:
        assert np.asarray(volumes[component]).dtype == np.float64


def test_no_layers_returns_nothing_at_all():
    grid = _grid(10.0, (4.0, 4.0, 0.0))
    assert absorber_conductivity(grid, []) == {}


@pytest.mark.parametrize("kwargs, match", [
    (dict(thickness=1.0, axis=3, side="high"), "axis must be"),
    (dict(thickness=1.0, axis=0, side="middle"), "side must be"),
    (dict(thickness=0.0, axis=0, side="high"), "finite and > 0"),
    (dict(thickness=1.0, axis=0, side="high", R_asymptotic=1.0), r"\(0, 1\)"),
])
def test_layer_refuses_nonsense(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AbsorberLayer(**kwargs)
