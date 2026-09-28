"""
Value tests for the FDTD grid conventions and the UPML coefficient table.

They pin the conventions the rest of the engine indexes against: MEEP's cell
count ``int(L*resolution + 0.5)``, ``dx = 1/resolution`` and
``dt = courant/resolution``, mirror-symmetry halving with cell 0 straddling the
plane at -0.5*dx — on any of the three axes, each plane carrying MEEP's declared
phase — the position-to-index consistency contract, and the closed-form PML
grading (including the lower-face skip on symmetric axes). NumPy path only —
no CuPy, no CUDA.
"""

from __future__ import annotations

import math

import numpy
import pytest

from .grid import Grid, Mirror, meep_cell_count
from .pml import PML, absorbing_cells, half_cell_extent


def make_grid(**overrides) -> Grid:  # Small default grid; overrides keep the call sites terse.
    params = dict(resolution=10.0, cell_size=(2.0, 2.0, 4.0))
    params.update(overrides)
    return Grid(**params)


def ravel(coefficients) -> numpy.ndarray:  # Flatten a broadcast-shaped coefficient array.
    return numpy.asarray(coefficients).ravel()


def test_cell_counts_use_meep_truncation_not_python_rounding():
    # L*resolution lands exactly on .5 for all three axes: MEEP rounds up, Python
    # round() rounds to even, so x and z would be one cell short with round().
    grid = Grid(resolution=5.0, cell_size=(0.5, 1.5, 2.5))
    assert (grid.nx_full, grid.ny_full, grid.nz_full) == (3, 8, 13)
    assert (round(0.5 * 5.0), round(1.5 * 5.0), round(2.5 * 5.0)) == (2, 8, 12)
    for length in (0.5, 1.5, 2.5, 3.0, 4.2):
        assert meep_cell_count(length, 5.0) == int(length * 5.0 + 0.5)


def test_dx_dt_and_snapped_cell_size():
    grid = make_grid(resolution=10.0, courant=0.5)
    assert grid.dx == pytest.approx(1.0 / 10.0)
    assert grid.dt == pytest.approx(0.5 / 10.0)
    assert Grid(resolution=20.0, cell_size=(1.0, 1.0, 1.0), courant=0.4).dt == pytest.approx(0.02)
    # Requested size is snapped back onto the integer grid.
    snapped = Grid(resolution=10.0, cell_size=(2.04, 2.0, 4.0))
    assert snapped.nx_full == 20
    assert snapped.Lx == pytest.approx(20 * snapped.dx)


def test_shapes_and_coordinate_dtypes():
    grid = make_grid()
    assert grid.shape == (20, 20, 40)
    assert grid.shape_full == (20, 20, 40)
    assert grid.total_cells == 20 * 20 * 40
    assert grid.xp is numpy
    assert grid.x.dtype == numpy.float32
    assert (grid.x.size, grid.y.size, grid.z.size) == grid.shape
    mesh_x, mesh_y, mesh_z = grid.meshgrid()
    assert mesh_x.shape == mesh_y.shape == mesh_z.shape == grid.shape
    assert not grid.has_symmetry()
    assert grid.symmetry_reduction_factor() == 1


def test_symmetry_halving_and_reduction_factor():
    grid = make_grid(symmetry=("X", "Y"))
    # A folded PERIODIC axis at an even count stores MEEP's halved window PLUS
    # the second-mirror plane and its ghost slot (owned_cells + 1); the halved
    # window itself is owned_cells = N//2 + 1.
    assert grid.owned_cells(0) == grid.nx_full // 2 + 1 == 11
    assert grid.nx == grid.ny == 12
    assert grid.nz == grid.nz_full == 40  # Unfolded axes keep their full count.
    assert grid.shape == (12, 12, 40)
    assert grid.shape_full == (20, 20, 40)
    assert grid.symmetry_reduction_factor() == 4
    assert make_grid(symmetry=("X",)).symmetry_reduction_factor() == 2


@pytest.mark.parametrize("axis,name", [(0, "x"), (1, "y"), (2, "z")])
def test_every_axis_halves_the_same_way_including_z(axis, name):
    """The halving is a property of a direction, not of "the propagation axis".

    Z used to be excluded by construction (``nz = nz_full``), which is the kind of
    special case that hides for as long as nobody folds it: a Z mirror would have
    stored the full axis, folded nothing, saved nothing, and stepped a domain half
    of which was never meant to exist.
    """
    grid = make_grid(symmetry=(("X", "Y", "Z")[axis],))
    counts_full = (grid.nx_full, grid.ny_full, grid.nz_full)
    # Periodic fold at an even count: MEEP's halved window (owned_cells) plus
    # the second-mirror plane and its ghost slot.
    assert grid.owned_cells(axis) == counts_full[axis] // 2 + 1
    assert grid.shape[axis] == counts_full[axis] // 2 + 2
    assert grid.big_corner_doubled(axis) == counts_full[axis]  # Window top, doubled.
    assert grid.shape_full == counts_full
    for other in range(3):
        if other != axis:
            assert grid.shape[other] == counts_full[other]
    assert grid.symmetry_reduction_factor() == 2
    assert grid.is_mirrored(axis) and not grid.axis_wraps(axis)
    assert grid.origin_doubled(axis) == -2  # MEEP halve(): io = icenter - 2.
    assert grid.symmetry_axes == (("X", "Y", "Z")[axis],)
    # Cell 0 straddles the plane, on the negative side, on whichever axis is folded.
    coordinate = (grid.x, grid.y, grid.z)[axis]
    assert float(coordinate[0]) == pytest.approx(-0.5 * grid.dx, abs=1e-7)
    # The last stored cell is the second-mirror plane cell, half a cell PAST the
    # window top; the last cell of MEEP's halved window sits one below it.
    assert float(coordinate[-1]) == pytest.approx(
        counts_full[axis] * grid.dx / 2 + 0.5 * grid.dx, abs=1e-6)
    assert float(coordinate[-2]) == pytest.approx(
        counts_full[axis] * grid.dx / 2 - 0.5 * grid.dx, abs=1e-6)
    assert grid.position_to_index(0.0, name, iyee_shift=1) == (0, 0.5)


def test_three_mirrors_fold_the_whole_cell_eightfold():
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), symmetry=("X", "Y", "Z"))
    assert grid.shape == (12, 12, 12)  # owned_cells 11 + the far plane and ghost slot.
    assert grid.shape_full == (20, 20, 20)
    assert grid.symmetry_reduction_factor() == 8  # MEEP's maximum in 3D.
    assert all(grid.is_mirrored(axis) for axis in range(3))
    assert not any(grid.axis_wraps(axis) for axis in range(3))


def test_odd_full_count_on_symmetry_axis_stores_meeps_halved_window():
    """An ODD folded count is MEEP's own fold, not a refusal: only the window shifts.

    The plane sits on a grid POINT for both parities — ``icenter()`` rounds the
    count down to even so ``icenter - io`` stays even (vec.cpp:1089-1101), and the
    centred origin ``io = -(N - N%2)`` puts doubled 0 on a lattice point either
    way. What an odd N changes is the WINDOW: the cell spans
    ``[-(N-1)dx/2, (N+1)dx/2]``, i.e. (N-1)/2 cells below the plane and (N+1)/2
    above. ``halve()`` (vec.cpp:1069-1074) stores ``1 + (big - icenter)/2 =
    N - N//2 + 1`` OWNED cells from origin -2 — for N=21 that is 12 cells whose
    last centre lands at exactly +L/2, half a cell above where an even axis ends
    — and a folded PERIODIC axis stores one more at either parity, the
    ``big_corner`` cell MEEP's ``num + 1`` allocation carries (vec.cpp:293-296)
    and ``owns`` includes (vec.cpp:445-462).
    """
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 4.0), symmetry=("Y",))
    assert grid.ny_full == 21 and grid.owned_cells(1) == 12  # N - N//2 + 1, per halve().
    assert grid.ny == 13  # Plus the big_corner cell; periodic, so it is not a wall.
    assert grid.origin_doubled(1) == -2  # Same as even: cell 0 straddles the plane.
    assert float(grid.y[0]) == pytest.approx(-0.05, abs=1e-6)
    assert float(grid.y[-2]) == pytest.approx(1.05, abs=1e-6)  # Exactly +Ly/2.
    assert float(grid.y[-1]) == pytest.approx(1.15, abs=1e-6)  # The big_corner cell.
    # A folded METALLIC axis keeps the owned count: MEEP holds its window-top
    # plane at zero, which is what the unstored zero ghost already says.
    walled = Grid(resolution=10.0, cell_size=(2.0, 2.1, 4.0), symmetry=("Y",),
                  boundaries="metallic")
    assert walled.owned_cells(1) == 12 and walled.ny == 12
    # Even control: halve() gives owned_cells = N//2 + 1; a periodic fold stores
    # one more (the second-mirror plane and its ghost slot), so the last stored
    # centre sits half a cell PAST the window top.
    even = Grid(resolution=10.0, cell_size=(2.0, 2.0, 4.0), symmetry=("Y",))
    assert even.owned_cells(1) == 11 and even.ny == 12 and even.origin_doubled(1) == -2
    assert float(even.y[-2]) == pytest.approx(0.95, abs=1e-6)  # +Ly/2 - dx/2.
    assert float(even.y[-1]) == pytest.approx(1.05, abs=1e-6)  # The plane cell.
    # Every axis folds the same way at odd counts, X and Z included.
    assert Grid(resolution=10.0, cell_size=(0.9, 2.0, 4.0), symmetry=("X",)).nx == 7
    assert Grid(resolution=10.0, cell_size=(2.0, 2.0, 0.9), symmetry=("Z",)).nz == 7
    # Odd on an UNFOLDED axis keeps MEEP's half-cell-higher origin, as before.
    odd = Grid(resolution=10.0, cell_size=(2.0, 0.9, 4.0), symmetry=("X",))
    assert odd.ny_full == 9 and odd.origin_doubled(1) == -8


def test_a_metallic_wall_on_an_invariant_axis_is_refused_by_name():
    """A PEC on an invariant axis is a polarization filter, and it fails SILENTLY.

    This test exists because the mutation battery found the guard unprotected:
    disabling `_require_boundaries_leave_invariant_axes_alone` outright broke
    nothing, so the refusal could have been deleted and the suite would have stayed
    green. What it prevents is not an exception but a smooth exact zero — MEEP's
    `on_metal_boundary` clears every component whose Yee shift on the walled axis is
    0, which on the single stored cell of an invariant z is Ex, Ey and Hz: the whole
    TE polarization of a 2-D run. Measured 1.000e+00 against CPU MEEP that way,
    against 1.805e-07 with the periodic wrap MEEP itself uses there
    (`fields.cpp nosize_direction`).

    MEEP's DEFAULT boundary is Metallic, so this is the configuration a 2-D lift
    reaches by doing nothing at all, not an exotic one.
    """
    # 2-D: z is invariant, so a wall there is refused; x and y are free to carry one.
    with pytest.raises(ValueError, match="z axis is declared 'metallic'"):
        Grid(resolution=10.0, cell_size=(2.0, 2.0, 0.0), dimensions=2,
             boundaries={"z": "metallic"})
    for axis in ("x", "y"):
        walled = Grid(resolution=10.0, cell_size=(2.0, 2.0, 0.0), dimensions=2,
                      boundaries={axis: "metallic"})
        assert walled.is_metallic("xyz".index(axis))
        assert not walled.is_metallic(2), "the invariant axis must stay periodic"

    # 1-D: x and y are the invariant pair, z is the resolved axis.
    for axis in ("x", "y"):
        with pytest.raises(ValueError, match=f"{axis} axis is declared 'metallic'"):
            Grid(resolution=10.0, cell_size=(0.0, 0.0, 4.0), dimensions=1,
                 boundaries={axis: "metallic"})
    one_d = Grid(resolution=10.0, cell_size=(0.0, 0.0, 4.0), dimensions=1,
                 boundaries={"z": "metallic"})
    assert one_d.is_metallic(2) and not any(one_d.is_metallic(a) for a in (0, 1))

    # The message has to say what to do instead, not just what is wrong.
    with pytest.raises(ValueError, match="periodic"):
        Grid(resolution=10.0, cell_size=(2.0, 2.0, 0.0), dimensions=2,
             boundaries="metallic")


def test_unknown_symmetry_token_raises():
    for bad in (("Q",), ("XY",), ("X", "W")):
        with pytest.raises(ValueError, match="mirror axis must be"):
            make_grid(symmetry=bad)
    # A lowercase axis is the same plane, not a different one.
    assert make_grid(symmetry=("x",)).symmetry == make_grid(symmetry=("X",)).symmetry


def test_two_planes_on_one_axis_are_refused_rather_than_resolved():
    # A duplicate is at best redundant; an even and an odd mirror about the same
    # plane force the field to be both symmetric and antisymmetric about it, which
    # only the zero field is. Taking the last entry would fold a run nobody asked for.
    with pytest.raises(ValueError, match="two mirror planes normal to X"):
        make_grid(symmetry=("X", "X"))
    with pytest.raises(ValueError, match="two mirror planes normal to X"):
        make_grid(symmetry=(Mirror("X", +1), Mirror("X", -1)))


def test_mirror_phase_is_declared_carried_and_kept_out_of_the_unmirrored_axes():
    grid = make_grid(symmetry=(Mirror("X", -1), "Y"))
    assert grid.mirror_phase(0) == -1  # Declared odd.
    assert grid.mirror_phase(1) == +1  # Bare axis name: MEEP's default even mirror.
    assert grid.mirror_phase(2) is None  # No plane at all — not "an even plane".
    # The phase changes the parities, not the storage: an odd fold halves the same way.
    assert grid.shape == make_grid(symmetry=("X", "Y")).shape
    # ... and it is not lost in the dataclass identity, which would make an odd run
    # compare equal to the even one it disagrees with everywhere.
    assert make_grid(symmetry=(Mirror("X", -1),)) != make_grid(symmetry=(Mirror("X", +1),))
    assert "X-1" in repr(make_grid(symmetry=(Mirror("X", -1),)))
    for bad in (0, 2, -2, 0.5, "even"):
        with pytest.raises(ValueError, match="mirror phase must be"):
            Mirror("X", bad)
    with pytest.raises(ValueError, match="axis must be 0"):
        grid.mirror_phase(3)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_bloch_phase_is_refused_on_every_folded_axis(axis):
    """A mirror plane forces the field even or odd about it, which is k = 0.

    Refused per axis and on all three of them: dropping the phase instead would run
    a plain symmetric boundary while the caller believed they had set a Bloch
    condition, and the band structure that came back would look entirely
    reasonable.
    """
    k_point = [0.0, 0.0, 0.0]
    k_point[axis] = 0.25
    with pytest.raises(ValueError, match="requires k_point"):
        make_grid(symmetry=(("X", "Y", "Z")[axis],), k_point=tuple(k_point))
    # The same phase on an axis this run does NOT fold is ordinary Bloch periodicity.
    other = (axis + 1) % 3
    fine = make_grid(symmetry=(("X", "Y", "Z")[axis],), k_point=tuple(
        0.25 if index == other else 0.0 for index in range(3)))
    assert fine.bloch_phase(other) is not None
    assert fine.bloch_phase(axis) is None


def test_axis_names_resolve_once_and_reject_anything_else():
    grid = make_grid(symmetry=("Z",))
    assert [grid.axis_index(name) for name in ("x", "y", "z")] == [0, 1, 2]
    assert [grid.stored_cells(axis) for axis in range(3)] == list(grid.shape)
    for bad in ("X", "w", 0):
        with pytest.raises(ValueError, match="axis must be 'x'"):
            grid.axis_index(bad)
    with pytest.raises(ValueError, match="axis must be 0"):
        grid.stored_cells(3)


def test_symmetric_axis_places_cell_zero_at_minus_half_dx():
    grid = make_grid(symmetry=("X", "Y"))
    assert float(grid.x[0]) == pytest.approx(-0.5 * grid.dx, abs=1e-7)
    assert float(grid.y[0]) == pytest.approx(-0.5 * grid.dx, abs=1e-7)
    assert float(grid.x[-2]) == pytest.approx(grid.Lx / 2 - 0.5 * grid.dx, abs=1e-6)
    assert float(grid.x[-1]) == pytest.approx(grid.Lx / 2 + 0.5 * grid.dx, abs=1e-6)
    # The unreduced axes keep the centered convention.
    assert float(grid.z[0]) == pytest.approx(-grid.Lz / 2 + 0.5 * grid.dx, abs=1e-6)
    plain = make_grid()
    assert float(plain.x[0]) == pytest.approx(-plain.Lx / 2 + 0.5 * plain.dx, abs=1e-6)


def test_position_to_index_is_consistent_with_the_coordinate_arrays():
    grid = make_grid()
    for pos in (-0.9, -0.15, 0.0, 0.37, 0.94):
        idx_low, weight = grid.position_to_index(pos, "z", iyee_shift=1)
        assert 0 <= idx_low <= grid.nz - 2
        assert 0.0 <= weight <= 1.0
        interpolated = float(grid.z[idx_low]) + weight * grid.dx
        assert interpolated == pytest.approx(pos, abs=1e-6)
    # Integer-position components sit half a cell lower than the cell centers.
    assert grid.position_to_index(0.0, "z", iyee_shift=0)[0] == 20


def test_position_to_index_on_a_symmetric_axis():
    # Cell 0 straddles the mirror plane at -0.5*dx, so the plane itself lands
    # halfway between stored cells 0 and 1.
    grid = make_grid(symmetry=("X",))
    idx_low, weight = grid.position_to_index(0.0, "x", iyee_shift=1)
    assert (idx_low, weight) == (0, 0.5)
    assert float(grid.x[idx_low]) + weight * grid.dx == pytest.approx(0.0, abs=1e-7)
    # Every stored cell center maps back onto its own fractional index.
    for cell in range(grid.nx):
        idx_low, weight = grid.position_to_index(float(grid.x[cell]), "x", iyee_shift=1)
        assert idx_low + weight == pytest.approx(cell, abs=1e-4)
    # Integer-position components sit a further half cell below.
    assert grid.position_to_index(0.0, "x", iyee_shift=0) == (1, 0.0)


def test_position_to_index_spans_exactly_the_stored_samples():
    # The in-range half of the contract, at both ends. The first and last stored
    # samples are IN, and the top one comes back naming a real neighbour — (n-2, 1.0),
    # not (n-1, 0.0) — so `idx_low + 1` is always a cell that exists.
    grid = make_grid()
    low_idx, low_weight = grid.position_to_index(-grid.Lz / 2 + 0.5 * grid.dx, "z", iyee_shift=1)
    assert (low_idx, low_weight) == (0, pytest.approx(0.0, abs=1e-12))
    top_idx, top_weight = grid.position_to_index(grid.Lz / 2 - 0.5 * grid.dx, "z", iyee_shift=1)
    assert top_idx == grid.nz - 2
    assert top_weight == pytest.approx(1.0)
    assert top_idx + 1 == grid.nz - 1
    # Integer-placed components reach a further half cell down and stop a half cell
    # short at the top: the last sample is at +L/2 - dx, not at +L/2 - dx/2.
    assert grid.position_to_index(-grid.Lz / 2, "z", iyee_shift=0) == (0, pytest.approx(0.0, abs=1e-12))
    assert grid.position_to_index(grid.Lz / 2 - grid.dx, "z", iyee_shift=0) == (
        grid.nz - 2, pytest.approx(1.0))
    # A single-cell axis has no neighbour to interpolate towards, so the one sample it
    # holds is the only answer and `idx_low + 1` is NOT a cell — the exception to the
    # rule above, called out because the clamp is what makes it come back as (0, 0).
    thin = Grid(resolution=10.0, cell_size=(0.1, 2.0, 2.0))
    assert thin.nx == 1
    assert thin.position_to_index(float(thin.x[0]), "x", iyee_shift=1) == (0, pytest.approx(0.0))
    with pytest.raises(ValueError, match="outside the x axis"):
        thin.position_to_index(float(thin.x[0]) + 0.5 * thin.dx, "x", iyee_shift=1)


def test_position_to_index_refuses_to_extrapolate_off_a_wrapping_axis():
    # THE REGRESSION. This used to tolerate one cell of overhang and return an
    # extrapolating weight against a clamped index — (0, -0.5) at the low z face — and
    # the CW point source built on it deposited ~1.5x the requested current onto one
    # cell instead of splitting it across the periodic face. Measured at 5.4e-1
    # complex relative L2 against CPU MEEP (see test_sources.py); now 1.8e-7.
    grid = make_grid()
    half_cell = 0.5 * grid.dx
    for pos in (-grid.Lz / 2, -grid.Lz / 2 - 2 * grid.dx, grid.Lz / 2, grid.Lz / 2 + 2 * grid.dx):
        with pytest.raises(ValueError, match="outside the z axis") as raised:
            grid.position_to_index(pos, "z", iyee_shift=1)
        # The message must send the caller to the machinery that CAN represent it.
        assert "wraps" in str(raised.value) and "_build_source_points" in str(raised.value)
    # Half a cell in from either face is genuinely in range and is not refused.
    grid.position_to_index(-grid.Lz / 2 + half_cell, "z", iyee_shift=1)
    grid.position_to_index(grid.Lz / 2 - half_cell, "z", iyee_shift=1)


def test_position_to_index_refuses_a_folded_axis_for_a_different_reason():
    # A wrapping axis has a lattice image of the request; a folded one has nothing at
    # all, and the message has to say which, because the fixes are different.
    grid = make_grid(symmetry=("X",))
    assert grid.axis_wraps(0) is False
    assert grid.axis_wraps(1) is True and grid.axis_wraps(2) is True
    with pytest.raises(ValueError, match="outside the x axis") as raised:
        grid.position_to_index(-grid.dx, "x", iyee_shift=1)  # Below the straddling cell 0.
    assert "mirror-folded" in str(raised.value)
    assert "metallic" not in str(raised.value), (
        "A folded axis and a walled axis both refuse, but for different reasons and with "
        "different fixes; the message must name the one that actually applies."
    )
    assert "_build_source_points" not in str(raised.value), (
        "A folded axis has no lattice vector, so pointing at the wrapping placement path "
        "would send the caller somewhere that cannot help."
    )
    # +Lx/2 itself is now interpolable — the stored second-mirror plane cell sits
    # half a cell past it — so the refusal starts past the plane cell's centre.
    low, weight = grid.position_to_index(grid.Lx / 2, "x", iyee_shift=1)
    assert (low, weight) == (grid.nx - 2, pytest.approx(0.5, abs=1e-4))
    with pytest.raises(ValueError, match="outside the x axis"):
        grid.position_to_index(grid.Lx / 2 + grid.dx, "x", iyee_shift=1)  # Past the plane cell.


def test_position_to_index_tolerates_float32_coordinate_round_trips():
    # The coordinate arrays are float32, and cell 0 of a folded axis rounds to a hair
    # BELOW its exact position — a hard [0, n-1] test rejects the grid's own coordinate.
    # A tolerance too tight here is not a wrong answer, it is a spurious refusal.
    odd_counts = Grid(resolution=10.0, cell_size=(1.9, 1.9, 2.1))  # 19, 19, 21 cells.
    for grid in (make_grid(), make_grid(symmetry=("X", "Y")), odd_counts):
        for axis, coordinates, count in (("x", grid.x, grid.nx),
                                         ("y", grid.y, grid.ny),
                                         ("z", grid.z, grid.nz)):
            for cell in range(count):
                idx_low, weight = grid.position_to_index(float(coordinates[cell]), axis, iyee_shift=1)
                assert idx_low + weight == pytest.approx(cell, abs=1e-4)
            # The slack is float-noise wide, not half-a-cell wide: a tenth of a cell out
            # is a real request and must still be refused.
            with pytest.raises(ValueError, match=f"outside the {axis} axis"):
                grid.position_to_index(float(coordinates[0]) - 0.1 * grid.dx, axis, iyee_shift=1)


def test_position_to_index_rejects_unknown_axis_shift_and_nonfinite_positions():
    grid = make_grid()
    with pytest.raises(ValueError, match="axis"):
        grid.position_to_index(0.0, "w")
    with pytest.raises(ValueError, match="iyee_shift"):
        grid.position_to_index(0.0, "x", iyee_shift=2)
    # NaN and inf used to slip past the range comparisons (both are False against
    # everything) and die inside math.floor with a message about integer conversion.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            grid.position_to_index(bad, "z", iyee_shift=1)


def test_grid_rejects_degenerate_geometry():
    with pytest.raises(ValueError, match="resolution"):
        Grid(resolution=0.0, cell_size=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="courant"):
        Grid(resolution=10.0, cell_size=(1.0, 1.0, 1.0), courant=0.0)
    with pytest.raises(ValueError, match="cell_size"):
        Grid(resolution=10.0, cell_size=(1.0, -1.0, 1.0))


def expected_prefac(pml: PML) -> float:  # Closed form: (order+1) * (-ln R) / (4 * thickness * dx).
    return (pml.order + 1) * (-math.log(pml.R_asymptotic)) / (
        4.0 * pml.thickness * pml.grid.dx)


def test_pml_sigma_closed_form_at_the_outer_boundary():
    grid = make_grid()
    pml = PML(grid=grid, thickness=5)
    prefac = 3.0 * (-math.log(1e-15)) / (4.0 * pml.thickness * grid.dx)
    assert pml.prefac() == pytest.approx(prefac, rel=1e-12)
    assert prefac == pytest.approx(expected_prefac(pml), rel=1e-12)

    # An integer-position sample sits exactly ON the lower wall (u = 1), but the
    # matching sample on the upper wall is the periodic image stored at index 0, so
    # the last stored cell only reaches u = (thickness - 1) / thickness. The two
    # faces are therefore NOT index mirrors of each other; asserting sigma[-1] ==
    # prefac is what let a one-cell upper-face misgrading ship.
    sigma = pml.get_sigma_profile("x")
    assert sigma[0] == pytest.approx(prefac, rel=1e-6)
    assert sigma[-1] == pytest.approx(
        prefac * ((pml.thickness - 1) / pml.thickness) ** pml.order, rel=1e-6)
    assert sigma[pml.thickness] == 0.0  # First interior cell above the lower layer.
    assert sigma[grid.nx - pml.thickness] == 0.0  # Inner edge of the upper layer: u = 0.
    # Stored coefficients carry the dt/2 factor and kappa = 1.
    stored_sigma = 0.5 * grid.dt * prefac
    assert ravel(pml.kms_x)[0] == pytest.approx(1.0 - stored_sigma, rel=1e-6)
    assert ravel(pml.kps_x)[0] == pytest.approx(1.0 + stored_sigma, rel=1e-6)
    assert ravel(pml.sinv_x)[0] == pytest.approx(1.0 / (1.0 + stored_sigma), rel=1e-6)
    # Half-integer positions are half a cell inside the boundary.
    half_sigma = 0.5 * grid.dt * prefac * ((pml.thickness - 0.5) / pml.thickness) ** 2
    assert ravel(pml.kms_x_h)[0] == pytest.approx(1.0 - half_sigma, rel=1e-6)
    assert ravel(pml.kms_x_h)[0] != pytest.approx(ravel(pml.kms_x)[0], rel=1e-6)


def test_pml_profile_is_symmetric_about_the_walls_not_the_index_range():
    # Reflecting a Yee lattice about the cell centre maps an integer-position sample
    # j to N - j and a half-integer one to N - 1 - j. So the half-integer sets are
    # palindromic in raw index order, while the integer sets are palindromic only
    # after dropping index 0 — whose partner, the sample on the upper wall, is its
    # own periodic image and is not stored separately. Demanding plain index-mirror
    # symmetry of the integer sets asserts a property MEEP does not have, and is
    # satisfied exactly by grading the upper face one full cell too deep.
    pml = PML(grid=make_grid(), thickness=4)
    for axis in ("x", "y", "z"):
        for name in (f"kms_{axis}_h", f"sinv_{axis}_h", f"kps_{axis}_h"):
            values = ravel(getattr(pml, name))
            numpy.testing.assert_allclose(values, values[::-1], rtol=1e-6, atol=0.0,
                                          err_msg=f"{name} is not mirror symmetric")
        for name in (f"kms_{axis}", f"sinv_{axis}", f"kps_{axis}"):
            values = ravel(getattr(pml, name))
            numpy.testing.assert_allclose(values[1:], values[1:][::-1], rtol=1e-6, atol=0.0,
                                          err_msg=f"{name} is not symmetric about the walls")
        profile = pml.get_sigma_profile(axis)
        numpy.testing.assert_allclose(profile[1:], profile[1:][::-1], rtol=1e-6)
        # ...and the integer profile must NOT be symmetric across the whole index
        # range, so a regression back to the mirrored upper face fails here.
        assert not numpy.allclose(profile, profile[::-1], rtol=1e-6)


def meep_sigma(grid: Grid, pml: PML, axis: str, offset: float) -> numpy.ndarray:
    """Independent re-derivation of MEEP's sigma for one axis and Yee offset.

    Two MEEP facts, transcribed separately from the engine's:

    1. structure.cpp ``pml_x`` (v1.29.0:625-628, byte-identical in v1.33.0) grades by
       distance from the wall on the half-cell index, quantizing BOTH the layer extent
       and that distance to whole half-cells::

           here = i * 0.5 / a
           x    = 0.5/a * ((int)(dx*(2a) + 0.5) - (int)(|bloc - here|*(2a) + 0.5))

       applied where ``x > 0`` (structure.cpp:682) at profile argument ``u = x / dx``
       (structure.cpp:683) — the RAW requested thickness in the denominator, never the
       snapped extent — with each wall carrying its own ``dx`` and therefore its own
       ``prefac = -ln(R)/(4*dx*integral)`` (structure.cpp:635). Written out here with
       MEEP's own ``a``, ``dx`` and ``bloc`` rather than this engine's cell counts, so
       the transcription is of the C expression and not of the port.

       ``prefac`` IS TRANSCRIBED, NOT CALLED. Reading it back from ``PML.prefac`` left
       the one fractional quantity with no independent oracle at all: substituting the
       snapped extent for the raw thickness in that divisor passed all 142 tests in this
       file with bit-identical numbers, because ``expected`` was computed with the very
       divisor under test. It is a real error away from MEEP — measured as a peak-sigma
       shift of +5.0000% at 2.1 cells, +4.0000% at 5.2 and -3.6364% at 5.3 (the
       ``|0.5*N - c| <= 0.25`` bound, evaluated at the thicknesses this file
       parametrizes) — and every whole-cell prefac written out elsewhere in this file is
       blind to it, because ``0.5 * N == c`` exactly there.
    2. meep/vec.hpp ``little_owned_corner0`` + step_db.cpp's loop bound make the OWNED
       half-cell indices 1..2N, so array cell ``j`` at Yee shift ``s`` sits at
       ``i = 2*j + s`` except that ``i = 0`` is the non-owned image of ``i = 2N``.
       A mirror-folded axis does not wrap and keeps the plain ``2*j + s``.

    A whole number of cells is the special case ``int(dx*(2a) + 0.5) == 2*n_pml``, where
    ``u`` collapses to ``1 - |i - i_wall| / (2*n_pml)`` and ``x > 0`` to ``u > 0``. It is
    only away from whole cells that the two forms separate — and they separate in THREE
    independent places: the extent (quantized), the normalization (raw) and ``prefac``
    (raw). All three are written out here rather than folded together or borrowed from
    the engine, because a shared quantity is not an oracle for itself.
    """
    index = ("x", "y", "z").index(axis)
    n_cells = (grid.nx, grid.ny, grid.nz)[index]
    low, high = pml.thickness_by_face[index]
    wraps = not (grid.sym_x, grid.sym_y, False)[index]
    a = float(grid.resolution)
    # The upper wall is MEEP's window top: 2 * halve()'s num (owned_cells). On a
    # folded PERIODIC axis at an even count the stored array runs one cell past
    # it (the second-mirror plane and its ghost slot), so 2 * n_cells is NOT the
    # wall there; everywhere else the two coincide.
    wall_high = 2 * grid.owned_cells(index)
    expected = numpy.zeros(n_cells)
    for j in range(n_cells):
        half_index = 2 * j + int(2 * offset)
        if wraps and half_index == 0:
            half_index = wall_high
        for wall, n_pml in ((0, low), (wall_high, high)):
            if n_pml <= 0:
                continue
            dx = n_pml / a  # MEEP's layer thickness, in length units.
            here = half_index * 0.5 / a
            bloc = wall * 0.5 / a
            x = 0.5 / a * (int(dx * (2 * a) + 0.5) - int(abs(bloc - here) * (2 * a) + 0.5))
            if x > 0:
                u = x / dx
                # structure.cpp:635, transcribed rather than read back from the engine.
                prefac = -math.log(pml.R_asymptotic) / (4 * dx * (1.0 / (pml.order + 1)))
                expected[j] = 0.5 * grid.dt * prefac * u**pml.order
    return expected


@pytest.mark.parametrize("cell_size, thickness", [
    # Uniform, over even and odd cell counts and even and odd thicknesses.
    ((2.0, 2.0, 4.0), 4), ((2.0, 2.0, 4.0), 5), ((3.0, 3.1, 6.0), 10),
    ((4.5, 3.2, 6.4), 15), ((2.0, 3.0, 4.0), 3),
    # ...and the per-face spellings, where the owned-index rule is observable: the
    # wrapped point at cell 0 belongs to the high face, so a low-only layer leaves it
    # alone, a high-only layer absorbs hardest there, and an unequal pair grades it
    # with the HIGH face's prefac.
    ((2.0, 2.0, 4.0), {"z": (5, 0)}), ((2.0, 2.0, 4.0), {"z": (0, 5)}),
    ((2.0, 2.0, 4.0), {"z": (8, 4)}), ((2.0, 2.0, 4.0), {"z": (4, 8)}),
    ((3.0, 3.1, 6.0), {"x": (7, 3), "z": (0, 9)}),
    ((2.0, 3.0, 4.1), {"x": 3, "y": (0, 4), "z": (6, 2)}),
    # ...and FRACTIONAL thicknesses, where the extent snaps to a half cell while the
    # profile keeps normalising by the raw thickness. The half-integer cases (10.5,
    # 4.5) land on an exact half-cell extent; the rest do not, so 2*c rounds down
    # (5.2 -> 10 half-cells) in some and up (5.3 -> 11) in others, which is the pair
    # of directions a floor() or a ceil() would each get half right.
    ((2.0, 2.0, 4.0), 4.5), ((2.0, 2.0, 4.0), 5.2), ((2.0, 2.0, 4.0), 5.3),
    ((3.0, 3.1, 6.0), 10.5), ((4.5, 3.2, 6.4), 12.7), ((2.0, 3.0, 4.1), 3.05),
    ((2.0, 2.0, 4.0), {"z": (5.5, 0)}), ((2.0, 2.0, 4.0), {"z": (0, 5.5)}),
    ((2.0, 2.0, 4.0), {"z": (8.4, 4.6)}), ((2.0, 2.0, 4.0), {"z": (4.6, 8.4)}),
    ((3.0, 3.1, 6.0), {"x": (7.15, 3.85), "z": (0, 9.9)}),
    ((2.0, 3.0, 4.1), {"x": 3.5, "y": (0, 4.2), "z": (6.8, 2.1)}),
])
def test_pml_grading_matches_meeps_distance_to_wall_rule(cell_size, thickness):
    # Checked through the public coefficient arrays (kps = 1 + 0.5*dt*prefac*u**order),
    # which is what the update kernels actually consume, over a matrix of cell counts
    # (even and odd), thicknesses (even and odd), face layouts and both Yee offsets.
    # This pins the rule rather than any one measured number, so no constant can be
    # tuned to pass it.
    grid = Grid(resolution=10.0, cell_size=cell_size)
    pml = PML(grid=grid, thickness=thickness)
    for axis in ("x", "y", "z"):
        for offset, suffix in ((0.0, ""), (0.5, "_h")):
            expected = meep_sigma(grid, pml, axis, offset)
            actual = ravel(getattr(pml, f"kps_{axis}{suffix}")) - 1.0
            numpy.testing.assert_allclose(
                actual, expected, rtol=1e-5, atol=1e-6 * max(float(expected.max()), 1e-30),
                err_msg=f"kps_{axis}{suffix} deviates from MEEP's pml_x grading "
                        f"on {cell_size} at thickness {thickness}")


@pytest.mark.parametrize("symmetry", [("X",), ("Y",), ("X", "Y")])
def test_a_folded_axis_keeps_the_plain_index_because_it_does_not_wrap(symmetry):
    # The owned-index remap is scoped to wrapping axes. On a folded one MEEP's halved
    # grid_volume already puts the owned integer cells at 2, 4, ... 2N of the stored
    # array (which holds n_full//2 + 1 cells, cell 0 straddling the mirror plane and
    # unowned), so the plain 2*j + s index is already MEEP's there. Applying the remap
    # anyway would put the outer wall's peak sigma on the mirror-plane cell.
    grid = make_grid(symmetry=symmetry)
    for thickness in (6, {"x": {"high": 6}, "y": {"high": 6}, "z": (6, 3)}):
        pml = PML(grid=grid, thickness=thickness)
        for axis in ("x", "y", "z"):
            for offset, suffix in ((0.0, ""), (0.5, "_h")):
                numpy.testing.assert_allclose(
                    ravel(getattr(pml, f"kps_{axis}{suffix}")) - 1.0,
                    meep_sigma(grid, pml, axis, offset), rtol=1e-5, atol=1e-12,
                    err_msg=f"kps_{axis}{suffix} on a {symmetry} grid at {thickness}")


def test_the_folded_pml_table_is_the_full_domain_table_under_the_half_cell_shift():
    """The ``skip_lower`` grading path, pinned on the coefficients rather than on a run.

    A folded axis stores one PML face instead of two and its cells are shifted by half
    a cell (cell 0 straddles the mirror plane), so its outer-face grading must be the
    full-domain grading of the same physical cells. Quadrant cell q is full-domain cell
    ``cx - 1 + q`` with ``cx = n_full // 2`` — the same identification the folded-run
    equivalence tests use.

    This is the exact, boundary-free form of what
    ``test_driver_vs_meep.py::test_symmetry_run_with_pml_matches_full_domain_run``
    used to assert end to end. That run-level comparison can no longer be exact: the
    full-domain run wraps its outer x face (MEEP's boundary) while the folded run still
    terminates it, so the two are the same system only while that face is quiet.
    """
    for thickness in (6, {"x": {"high": 6}}, {"x": {"high": 4}, "z": 8}):
        full = PML(grid=make_grid(), thickness=thickness)
        folded = PML(grid=make_grid(symmetry=("X",)), thickness=thickness)
        centre = full.grid.nx // 2
        owned = folded.grid.owned_cells(0)
        for suffix in ("", "_h"):
            for name in ("kms", "sinv", "kps"):
                numpy.testing.assert_allclose(
                    ravel(getattr(folded, f"{name}_x{suffix}"))[:owned],
                    ravel(getattr(full, f"{name}_x{suffix}"))[centre - 1:],
                    rtol=0.0, atol=1e-7,
                    err_msg=f"{name}_x{suffix} differs between the folded and full tables")
            # The stored second-mirror plane cell's INTEGER sample sits exactly ON
            # the upper wall — the same lattice point the full domain stores as its
            # wrapped cell 0 (graded from the upper wall with the high face's own
            # prefac) — so the two coefficient rows must agree. The half-integer
            # slot past it is the far ghost, whose curl is masked and whose value
            # the fill pass overwrites, so its coefficient carries no run.
            numpy.testing.assert_allclose(
                ravel(getattr(folded, f"kms_x"))[-1:],
                ravel(getattr(full, f"kms_x"))[0:1],
                rtol=0.0, atol=1e-7,
                err_msg="the plane cell's wall coefficients differ from the full "
                        "domain's wrapped wall plane")
        # Not vacuous: the compared span really does contain the absorber.
        assert ravel(folded.kms_x).min() < 0.5


def test_pml_interior_is_untouched_and_broadcast_shapes_are_right():
    grid = make_grid()
    pml = PML(grid=grid, thickness=4)
    assert pml.kms_x.shape == (grid.nx, 1, 1)
    assert pml.kms_y.shape == (1, grid.ny, 1)
    assert pml.kms_z.shape == (1, 1, grid.nz)
    assert pml.kms_x.dtype == numpy.float32
    assert pml.sinv_z_h.dtype == numpy.float32
    interior = slice(pml.thickness, grid.nx - pml.thickness)
    numpy.testing.assert_allclose(ravel(pml.kms_x)[interior], 1.0, rtol=0.0, atol=0.0)
    numpy.testing.assert_allclose(ravel(pml.sinv_x)[interior], 1.0, rtol=0.0, atol=0.0)
    numpy.testing.assert_allclose(ravel(pml.kps_x_h)[interior], 1.0, rtol=0.0, atol=0.0)


def test_pml_skips_the_lower_face_on_a_symmetry_axis():
    grid = make_grid(symmetry=("X",))
    pml = PML(grid=grid, thickness=4)
    lower_x = ravel(pml.kms_x)[: pml.thickness]
    numpy.testing.assert_allclose(lower_x, 1.0, rtol=0.0, atol=0.0)
    numpy.testing.assert_allclose(ravel(pml.sinv_x)[: pml.thickness], 1.0, rtol=0.0, atol=0.0)
    numpy.testing.assert_allclose(pml.get_sigma_profile("x")[: pml.thickness], 0.0)
    # The outer face is still absorbing, and the untouched Y axis keeps both faces.
    assert ravel(pml.kms_x)[-1] < 1.0
    assert ravel(pml.kms_y)[0] < 1.0
    assert ravel(pml.kms_y)[-1] < 1.0


def test_interior_slice_keeps_the_symmetry_plane():
    grid = make_grid(symmetry=("X",))
    pml = PML(grid=grid, thickness=4)
    x0, x1, y0, y1, z0, z1 = pml.interior_slice()
    # No lower PML to trim; the high trim is measured from the wall (owned_cells),
    # so the second-mirror plane and ghost slot stored past it stay excluded.
    assert (x0, x1) == (0, grid.owned_cells(0) - pml.thickness)
    assert x1 == grid.nx - 1 - pml.thickness  # The folded axis stores one row past the wall.
    assert (y0, y1) == (pml.thickness, grid.ny - pml.thickness)
    assert (z0, z1) == (pml.thickness, grid.nz - pml.thickness)
    plain = PML(grid=make_grid(), thickness=4)
    assert plain.interior_slice() == (4, 16, 4, 16, 4, 36)


def test_zero_thickness_pml_is_the_identity():
    pml = PML(grid=make_grid(), thickness=0)
    for name in ("kms_x", "sinv_y", "kps_z", "kms_x_h", "sinv_y_h", "kps_z_h"):
        numpy.testing.assert_allclose(ravel(getattr(pml, name)), 1.0, rtol=0.0, atol=0.0)
    numpy.testing.assert_allclose(pml.get_sigma_profile("z"), 0.0)


def test_pml_rejects_degenerate_configurations():
    grid = make_grid()
    with pytest.raises(ValueError, match="thickness"):
        PML(grid=grid, thickness=-1)
    with pytest.raises(ValueError, match="does not fit"):
        PML(grid=grid, thickness=11)  # 2 * 11 > nx = 20
    with pytest.raises(ValueError, match="order"):
        PML(grid=grid, thickness=4, order=0)
    with pytest.raises(ValueError, match="R_asymptotic"):
        PML(grid=grid, thickness=4, R_asymptotic=0.0)
    with pytest.raises(ValueError, match="axis"):
        PML(grid=grid, thickness=4).get_sigma_profile("w")


# --- Per-axis / per-side thickness -----------------------------------------------

COEFFICIENT_NAMES = tuple(
    f"{name}_{axis}{suffix}"
    for name in ("kms", "sinv", "kps")
    for axis in ("x", "y", "z")
    for suffix in ("", "_h")
)


def uniform_coefficients_before_per_face(grid: Grid, thickness: int, order: int = 2,
                                         R_asymptotic: float = 1e-15) -> dict:
    """Transcription of the pre-per-face coefficient table, for a bit-for-bit pin.

    One scalar thickness, one ``prefac`` taken from it, the lower X/Y face dropped on
    a mirrored axis, and the same float expression order the old code used
    (``0.5 * dt * prefac`` folded before the profile). Reproduced independently here
    so that "the scalar path is unchanged" is a statement about the arithmetic rather
    than about the new code agreeing with itself.
    """
    dx_pml = thickness * grid.dx
    prefac = 0.0 if dx_pml <= 0 else float(
        -numpy.log(R_asymptotic) / (4.0 * dx_pml * (1.0 / (order + 1))))
    scale = 0.5 * grid.dt * prefac
    table = {}
    for axis, n_cells, skip_lower in (("x", grid.nx, grid.sym_x),
                                      ("y", grid.ny, grid.sym_y),
                                      ("z", grid.nz, False)):
        # The upper wall is the window top, 2 * halve()'s num (owned_cells) —
        # NOT 2 * stored: a folded periodic axis at an even count stores the
        # second-mirror plane and its ghost slot past MEEP's corner, and the
        # wall does not move with them.
        wall_high = 2 * grid.owned_cells(("x", "y", "z").index(axis))
        for offset, suffix in ((0.0, ""), (0.5, "_h")):
            sig = numpy.zeros(n_cells, dtype=numpy.float32)
            if thickness > 0:
                half_index = 2 * numpy.arange(n_cells) + int(2 * offset)
                walls = [] if skip_lower else [0]
                walls.append(wall_high)
                for wall in walls:
                    u = 1.0 - numpy.abs(half_index - wall) / (2.0 * thickness)
                    inside = u > 0
                    sig[inside] = scale * u[inside] ** order
            kap = numpy.ones(n_cells, dtype=numpy.float32)
            table[f"kms_{axis}{suffix}"] = kap - sig
            table[f"sinv_{axis}{suffix}"] = 1.0 / (kap + sig)
            table[f"kps_{axis}{suffix}"] = kap + sig
    return table


@pytest.mark.parametrize("overrides", [{}, {"symmetry": ("X",)}, {"symmetry": ("X", "Y")}])
@pytest.mark.parametrize("thickness", [0, 1, 4, 7])
def test_scalar_thickness_reproduces_the_pre_per_face_table_bit_for_bit(overrides, thickness):
    """The scalar spelling must not cost a single ULP as the table grows per-face support.

    It is what every recorded PML number was measured with (the 5.84e-4 / 6.31e-4
    CPU-MEEP floor, the transparency scaling, the symmetry runs). Bytes, not allclose:
    a reordered float expression would pass a tolerance and quietly move the floor.

    Positive control, measured rather than assumed. The comparison is on stored
    float32, so it can only see a change above that type's 1.192e-07 resolution, and
    prefac carries R's logarithm, so a relative nudge of R shrinks by 1/ln(1e-15) =
    1/34.5 on the way in: 1e-05 in R is 2.895e-07 in prefac, 2.4 float32 eps. Bisected
    per combination, the smallest nudge that moves any byte is 2.540e-06 at 1 cell
    (7.355e-08 in prefac, 0.62 float32 eps), 1.217e-06 at 4 cells and 5.761e-08 at 7 —
    thicker layers grade more cells and so detect more finely, and the three symmetry
    spellings share a threshold because halving an axis drops cells without changing
    the sigma values. The control must clear the coarsest of them, so 1 cell sets the
    bar and 1e-05 clears it by 3.9x and no more. Below each combination's own
    threshold, 400 log-spaced probes per combination move nothing, so these are floors
    rather than bisection artifacts. At 1e-05 the nudge moves 579 of the 5112 stored
    float32 entries the 12 combinations compare.

    Three of the 12 (thickness=0) are inert: sigma is identically zero and the table is
    all ones, so no nudge of R can move them. Their control is the other direction —
    an accidental grading is visible, the thickness=1 table differing from theirs in
    18, 15 and 12 of the 18 arrays for the three symmetry spellings.
    """
    grid = make_grid(**overrides)
    pml = PML(grid=grid, thickness=thickness)
    expected = uniform_coefficients_before_per_face(grid, thickness)
    for name in COEFFICIENT_NAMES:
        actual = numpy.asarray(getattr(pml, name), dtype=numpy.float32).ravel()
        assert actual.tobytes() == expected[name].astype(numpy.float32).tobytes(), (
            f"{name} changed for the scalar thickness={thickness} spelling on grid {overrides}"
        )
    assert pml.thickness == thickness
    assert pml.uniform_thickness == thickness
    assert pml.is_active == (thickness > 0)
    if pml.is_active:
        nudged = uniform_coefficients_before_per_face(
            grid, thickness, R_asymptotic=1e-15 * 1.00001)
        assert any(nudged[name].astype(numpy.float32).tobytes() != expected[name].tobytes()
                   for name in COEFFICIENT_NAMES)
    else:
        graded = uniform_coefficients_before_per_face(grid, 1)
        assert any(graded[name].astype(numpy.float32).tobytes() != expected[name].tobytes()
                   for name in COEFFICIENT_NAMES)


def test_every_spelling_of_the_same_layer_builds_the_same_coefficients():
    # A caller typing {"z": 4} and one typing ((0,0),(0,0),(4,4)) must not get two
    # different absorbers; equality is checked on bytes so no spelling is even a
    # rounding away from another.
    grid = make_grid()
    equivalents = (
        {"z": 4},
        {"z": (4, 4)},
        {"z": {"low": 4, "high": 4}},
        {"Z": 4.0},
        {2: 4},
        (0, 0, 4),
        ((0, 0), (0, 0), (4, 4)),
    )
    reference = PML(grid=grid, thickness=equivalents[0])
    assert reference.thickness_by_face == ((0, 0), (0, 0), (4, 4))
    for spelling in equivalents[1:]:
        candidate = PML(grid=grid, thickness=spelling)
        assert candidate.thickness_by_face == reference.thickness_by_face, spelling
        for name in COEFFICIENT_NAMES:
            assert (numpy.asarray(getattr(candidate, name)).tobytes()
                    == numpy.asarray(getattr(reference, name)).tobytes()), (spelling, name)


def test_an_axis_left_out_of_the_request_carries_no_absorber_at_all():
    # The crux of the feature: absorbing in z must leave x and y exactly as they are
    # without a PML, so those axes can stay (Bloch-)periodic. Exactly 1.0, not
    # 1.0 within a tolerance — the update kernels multiply by these every step.
    grid = make_grid()
    pml = PML(grid=grid, thickness={"z": 6})
    assert pml.is_active and pml.axis_has_pml(2)
    assert not pml.axis_has_pml(0) and not pml.axis_has_pml(1)
    for axis in ("x", "y"):
        for suffix in ("", "_h"):
            numpy.testing.assert_array_equal(ravel(getattr(pml, f"kms_{axis}{suffix}")), 1.0)
            numpy.testing.assert_array_equal(ravel(getattr(pml, f"sinv_{axis}{suffix}")), 1.0)
            numpy.testing.assert_array_equal(ravel(getattr(pml, f"kps_{axis}{suffix}")), 1.0)
        numpy.testing.assert_array_equal(pml.get_sigma_profile(axis), 0.0)
    # ...while z is graded exactly as the uniform layer of the same thickness grades it.
    uniform = PML(grid=grid, thickness=6)
    for suffix in ("", "_h"):
        assert (numpy.asarray(pml.__dict__[f"kms_z{suffix}"]).tobytes()
                == numpy.asarray(uniform.__dict__[f"kms_z{suffix}"]).tobytes())
    assert pml.interior_slice() == (0, grid.nx, 0, grid.ny, 6, grid.nz - 6)


def test_each_face_is_graded_from_its_own_thickness():
    # MEEP calls structure_chunk::use_pml once per (direction, side) with that side's
    # own dx, so prefac is a property of the FACE. A shallower face is graded more
    # steeply — halving the thickness doubles prefac — and taking one prefac for the
    # whole axis would leave the thin face under-absorbing at the same peak sigma.
    grid = make_grid()
    pml = PML(grid=grid, thickness={"z": (8, 4)})
    assert pml.thickness_by_face == ((0, 0), (0, 0), (8, 4))
    assert pml.thickness == 8  # The deepest face, the back-compatible scalar.
    profile = pml.get_sigma_profile("z")
    assert pml.prefac(4) == pytest.approx(2.0 * pml.prefac(8), rel=1e-12)
    # Cell 0 holds the lattice point at half-cell index 2N — the one MEEP owns as its
    # big corner — so it belongs to the HIGH wall and is graded with the HIGH face's
    # thickness and prefac, at u = 1. Reading it as the low wall's u = 1 sample (the
    # prefac(8) this used to assert) is the defect that cost a one-sided layer 1.2e-01
    # against CPU MEEP; see PML._graded_sigma.
    assert profile[0] == pytest.approx(pml.prefac(4), rel=1e-6)
    assert profile[0] != pytest.approx(pml.prefac(8), rel=1e-3)  # The two are 2x apart.
    assert profile[-1] == pytest.approx(pml.prefac(4) * (3.0 / 4.0) ** 2, rel=1e-6)
    assert profile[8] == 0.0 and profile[grid.nz - 4] == 0.0  # Inner edges of each layer.
    numpy.testing.assert_array_equal(profile[8:grid.nz - 4], 0.0)
    # The low face therefore grades seven integer samples, not eight: its eighth would
    # be the wall itself, which is the same lattice point as the high wall.
    assert numpy.count_nonzero(profile[1:8]) == 7
    # Independent per-face re-derivation of MEEP's pml_x over both walls at once,
    # including its owned half-cell index (1..2N, so cell 0 of the integer set is 2N).
    for offset, suffix in ((0.0, ""), (0.5, "_h")):
        expected = numpy.zeros(grid.nz)
        for j in range(grid.nz):
            half_index = 2 * j + int(2 * offset) or 2 * grid.nz
            for wall, n_pml in ((0, 8), (2 * grid.nz, 4)):
                u = 1.0 - abs(half_index - wall) / (2.0 * n_pml)
                if u > 0:
                    expected[j] = 0.5 * grid.dt * pml.prefac(n_pml) * u ** pml.order
        numpy.testing.assert_allclose(
            ravel(getattr(pml, f"kps_z{suffix}")) - 1.0, expected, rtol=1e-5, atol=1e-12)


def test_one_sided_layers_are_not_index_mirrors_of_each_other():
    # A low-only and a high-only layer of the same thickness are the same absorber at
    # opposite ends, but on a Yee lattice the integer sets are not index reversals:
    # the wrapped lattice point stored at index 0 belongs to the HIGH wall (MEEP owns
    # the half-cell indices 1..2N, so the integer set is 2, 4, ... 2N), so only a
    # high-only layer reaches u = 1 there. Reversing one to build the other, or reading
    # index 0 as the low wall, is the exact defect PML._graded_sigma records.
    grid = make_grid()
    low = PML(grid=grid, thickness={"z": (5, 0)})
    high = PML(grid=grid, thickness={"z": (0, 5)})
    # Half-integer samples sit half a cell inside both walls, so those sets ARE exact
    # index mirrors...
    numpy.testing.assert_array_equal(ravel(low.kms_z_h), ravel(high.kms_z_h)[::-1])
    # ...while the integer sets are the same grading offset by one whole cell.
    assert not numpy.allclose(ravel(low.kms_z), ravel(high.kms_z)[::-1])
    numpy.testing.assert_array_equal(ravel(low.kms_z)[1:], ravel(high.kms_z)[::-1][:-1])
    # Cell 0 is the wrapped point: the high-only layer absorbs hardest there and the
    # low-only layer does not absorb there at all. This assertion used to read the
    # other way round, which is what a one-sided run then measured as 1.2e-01 against
    # CPU MEEP with a live wave on the un-absorbed face.
    assert ravel(high.kms_z)[0] == ravel(high.kms_z).min()
    assert ravel(low.kms_z)[0] == 1.0
    assert ravel(low.kms_z).min() > ravel(high.kms_z).min()
    # A 5-cell low face therefore grades 4 integer samples and 5 half-integer ones,
    # while a 5-cell high face grades 5 of each — MEEP's own asymmetry, not a bug.
    assert numpy.count_nonzero(low.get_sigma_profile("z")) == 4
    assert numpy.count_nonzero(high.get_sigma_profile("z")) == 5
    # Each is graded only on its own half, apart from that one wrapped point.
    numpy.testing.assert_array_equal(low.get_sigma_profile("z")[5:], 0.0)
    numpy.testing.assert_array_equal(high.get_sigma_profile("z")[1: grid.nz - 5], 0.0)
    assert low.interior_slice() == (0, grid.nx, 0, grid.ny, 5, grid.nz)
    assert high.interior_slice() == (0, grid.nx, 0, grid.ny, 0, grid.nz - 5)


def test_cell_zero_is_graded_from_the_wall_meep_owns_it_at():
    """The wrapped lattice point at index 0 is MEEP's half-cell index 2N, not 0.

    MEEP's ``LOOP_OVER_VOL`` runs from ``little_owned_corner0(c) = little_corner + 2 -
    iyee_shift(c)`` to ``big_corner()`` (meep/vec.hpp:1102, step_db.cpp:124), so the
    owned half-cell indices are 1, 3, ... 2N-1 for the half-integer set and 2, 4, ...
    2N for the integer one. Index 0 is never owned: it is the non-owned periodic image
    of 2N. This array stores that point once, at cell 0, and must grade it from the
    UPPER wall.

    Invisible on a uniform layer — u = 1 from either wall — which is why every pinned
    coefficient number survived it. Worth an entire ``1/(kappa+sigma)`` at that plane
    otherwise, and that plane is where a one-sided layer's wrap re-enters.
    """
    grid = make_grid()
    for faces, absorbs_at_zero in ((( 6, 0), False), ((0, 6), True), ((6, 6), True), ((6, 3), True)):
        pml = PML(grid=grid, thickness={"z": faces})
        sigma = pml.get_sigma_profile("z")
        assert bool(sigma[0] > 0.0) is absorbs_at_zero, faces
        if absorbs_at_zero:  # Graded from the HIGH face's own thickness, at u = 1.
            assert sigma[0] == pytest.approx(pml.prefac(faces[1]), rel=1e-6), faces
    # A mirror-folded axis does not wrap: its cell 0 straddles the mirror plane rather
    # than imaging the far face, MEEP's halved grid_volume already puts the owned
    # integer cells at 2, 4, ... 2N of the stored array, and cell 0 is simply unowned.
    # So the remap must NOT reach it, and a folded axis carrying only a high face has
    # no absorber at cell 0.
    folded = PML(grid=make_grid(symmetry=("X",)), thickness={"x": {"high": 6}})
    assert not folded.axis_wraps(0) and folded.axis_wraps(1) and folded.axis_wraps(2)
    assert folded.get_sigma_profile("x")[0] == 0.0
    with pytest.raises(ValueError, match="axis must be"):
        folded.axis_wraps(3)


def test_a_zero_request_is_no_absorber_rather_than_a_silent_one():
    grid = make_grid()
    for spelling in (0, {"z": 0}, {"z": (0, 0)}, (0, 0, 0)):
        pml = PML(grid=grid, thickness=spelling)
        assert not pml.is_active, spelling
        assert pml.thickness == 0
        for name in COEFFICIENT_NAMES:
            numpy.testing.assert_array_equal(ravel(getattr(pml, name)), 1.0)
        assert pml.interior_slice() == (0, grid.nx, 0, grid.ny, 0, grid.nz)
    # Positive control: the same assertions must fail for a layer that does absorb.
    assert PML(grid=grid, thickness={"z": 1}).is_active


def test_a_mirrored_axis_refuses_a_named_lower_face_but_still_takes_the_shorthand():
    grid = make_grid(symmetry=("X",))
    # The scalar shorthand means "every face that can carry one" and drops it, as
    # it always has...
    assert PML(grid=grid, thickness=4).thickness_by_face == ((0, 4), (4, 4), (4, 4))
    # ...but naming that face is a request that cannot be honoured anywhere else.
    with pytest.raises(ValueError, match="mirror plane"):
        PML(grid=grid, thickness={"x": 4})
    with pytest.raises(ValueError, match="mirror plane"):
        PML(grid=grid, thickness={"x": (4, 0)})
    high_only = PML(grid=grid, thickness={"x": {"high": 4}})
    assert high_only.thickness_by_face == ((0, 4), (0, 0), (0, 0))
    uniform = PML(grid=grid, thickness=4)
    assert (numpy.asarray(high_only.kms_x).tobytes() == numpy.asarray(uniform.kms_x).tobytes())


def test_thickness_specifications_that_cannot_mean_anything_raise():
    grid = make_grid()  # 20 x 20 x 40 cells.
    for spelling, message in (
        ({"w": 4}, "Unknown PML axis key"),
        ({"z": {"middle": 4}}, "Unknown PML side"),
        ((4, 4), "one entry per axis"),
        ((4, 4, 4, 4), "one entry per axis"),
        ({"z": (1, 2, 3)}, "scalar or a .low, high. pair"),
        ({"z": True}, "number of grid cells"),
        (float("inf"), "finite"),
        (float("nan"), "finite"),
        ("4", "number of grid cells"),
        (-1, "non-negative"),
        ({"z": -2}, "non-negative"),
        (-0.5, "non-negative"),
        ({"x": 11}, "does not fit"),          # 11 + 11 > nx = 20
        ({"x": (14, 7)}, "does not fit"),     # 21 > 20, even though neither face alone is
        ({"z": 21}, "does not fit"),          # 42 > nz = 40
        ({"x": 10.25}, "does not fit"),       # 20.5 > nx = 20: the fit rule is not integer
    ):
        with pytest.raises(ValueError, match=message):
            PML(grid=grid, thickness=spelling)
    # An axis key given twice is a contradiction, not a last-one-wins merge.
    with pytest.raises(ValueError, match="specified twice"):
        PML(grid=grid, thickness={"z": 4, "Z": 5})
    # ...and the boundary of the fit rule is legal: the two faces may exactly meet.
    assert PML(grid=grid, thickness={"x": 10}).thickness_by_face[0] == (10, 10)
    # A fractional cell count is NOT one of the things that cannot mean anything: MEEP
    # takes mp.PML(0.5) at resolution 71 as 35.5 cells and builds it (structure.cpp
    # pml_x snaps the EXTENT to a half cell, nothing else). It is accepted here and
    # kept as the float it was asked for, while a whole request stays an int so the
    # pre-existing table and repr are untouched. Not the bytes — those are the same
    # whichever type carries the count (measured: 84 of 84 coefficient arrays identical
    # when the same whole request is carried as a float), so this split is about the
    # types downstream consumers see, not about the coefficients.
    fractional = PML(grid=grid, thickness={"x": 4.5})
    assert fractional.thickness_by_face[0] == (4.5, 4.5)
    assert isinstance(fractional.thickness_by_face[0][0], float)
    assert isinstance(PML(grid=grid, thickness={"x": 4.0}).thickness_by_face[0][0], int)


# --- Fractional-cell thickness: MEEP's half-cell extent ---------------------------


@pytest.mark.parametrize("cells, extent", [
    # MEEP: N = (int)(dx * (2*a) + 0.5) half cells (structure.cpp:627). The C cast
    # truncates, so this is ROUND-HALF-UP on the half-cell count, not the round-half-
    # to-even that numpy.round and Python's round() perform.
    (0.0, 0), (0.1, 0), (0.2, 0), (0.24, 0),   # Under a quarter cell: MEEP builds none.
    (0.25, 1), (0.5, 1), (0.75, 2), (1.0, 2),
    (4.0, 8), (4.5, 9), (5.2, 10), (5.3, 11),
    (35.2, 70), (35.245, 70), (35.25, 71),     # 70.4, 70.49, 70.50 half cells.
    (35.74995, 71), (35.75, 72),               # 71.4999, 71.50 half cells.
    (35.5, 71),                                # mp.PML(0.5) at resolution 71.
])
def test_the_layer_extent_snaps_to_half_cells_and_rounds_half_up(cells, extent):
    """MEEP quantizes the layer EXTENT to whole half-cells, and only the extent.

    ``pml_x`` (structure.cpp:625-628) is one expression, and the rounding primitive in
    it is a C cast of ``v + 0.5``. Reaching for ``numpy.round`` / ``round`` instead is
    banker's rounding — ``round(70.5) == 70`` where MEEP gives 71 — and the two
    disagree only at an exact even half-cell count, which is a silent, rare and
    thoroughly untestable-by-accident divergence. ``math.ceil`` gets the sub-quarter-
    cell case wrong in the other direction: below 0.25 cells MEEP's ``found_pml``
    stays false (structure.cpp:648-654) and it allocates no sigma array at all.
    """
    assert half_cell_extent(cells) == extent
    # ...and the ceil()/round() alternatives really are different functions here, so
    # this table is not satisfiable by any of them.
    assert half_cell_extent(35.25) != int(round(2.0 * 35.25))
    assert half_cell_extent(0.2) != math.ceil(2.0 * 0.2)


@pytest.mark.parametrize("cells, absorbing", [
    (0.0, 0), (0.2, 0), (0.25, 1), (1.0, 1), (4.0, 4), (4.5, 5),
    (5.2, 5), (5.3, 6), (5.5, 6), (35.5, 36),
])
def test_the_absorbing_cell_count_is_the_integer_samples_that_are_graded(cells, absorbing):
    """How many whole array cells one face touches — ``ceil(N/2)``, not ``int(cells)``.

    The integer-position sample of cell ``j`` sits at half-cell distance ``2*j`` from
    the wall and is graded while ``2*j < N``, so a face reaches ``ceil(N/2)`` cells.
    This is the number the interior-trimming consumers need (``PML.interior_slice``,
    ``DFTMonitor.set_region_from_pml``); truncating with ``int(5.5) == 5`` instead
    leaves the cell carrying the second-largest sigma inside a "PML interior" region,
    and the spectrum that comes back is smooth, plausible and low.
    """
    assert absorbing_cells(cells) == absorbing
    grid = make_grid()  # 20 x 20 x 40 cells.
    if 0 < cells <= 10:
        # Counted on the HIGH face, whose integer samples are the plain ones: on a
        # wrapping axis cell 0 holds the lattice point MEEP owns at 2N, so it belongs
        # to the high wall. A low face therefore grades one integer sample FEWER — its
        # own wall sample is that same shared point — which is MEEP's asymmetry and not
        # a different count of absorbing cells.
        high = PML(grid=grid, thickness={"z": (0, cells)})
        assert numpy.count_nonzero(high.get_sigma_profile("z")) == absorbing
        assert high.interior_slice()[4:] == (0, grid.nz - absorbing)
        low = PML(grid=grid, thickness={"z": (cells, 0)})
        assert numpy.count_nonzero(low.get_sigma_profile("z")) == max(absorbing - 1, 0)
        assert low.interior_slice()[4:] == (absorbing, grid.nz)


def test_a_fractional_layer_is_neither_of_its_whole_cell_neighbours():
    """35.5 cells is a third absorber, not 35 rounded or 36 rounded.

    Two independent quantities separate them, and a plausible implementation gets each
    one wrong on its own: the extent snaps to 71 half cells (between 70 and 72), and
    the profile normalizes by the RAW 35.5 (between 35 and 36), so sigma differs from
    both neighbours everywhere it is nonzero — not only in the one cell the extent adds.
    Refusing this configuration and rounding to a neighbour is what the converter used
    to do; stepped against CPU MEEP at this resolution the substitution lands 8.75e-04
    (to 35) and 9.27e-04 (to 36) away from a run that reproduces MEEP at 1.16e-06
    (``test_from_meep.py``, the ``pml_fractional_cells_res71`` case).
    """
    grid = Grid(resolution=71.0, cell_size=(1.0, 1.0, 2.0))
    exact = PML(grid=grid, thickness={"z": 35.5}).get_sigma_profile("z")
    for neighbour in (35, 36):
        rounded = PML(grid=grid, thickness={"z": neighbour}).get_sigma_profile("z")
        difference = numpy.linalg.norm(exact - rounded) / numpy.linalg.norm(exact)
        assert difference > 1e-3, (
            f"a {neighbour}-cell layer is indistinguishable from the 35.5-cell one "
            f"(relative sigma difference {difference:.3e}); the test would pass under "
            f"the rounding this feature exists to remove"
        )
    # The extent alone accounts for one cell of the difference: 71 half cells reaches
    # 36 integer samples, 70 reaches 35 and 72 reaches 36.
    assert half_cell_extent(35.5) == 71
    assert (absorbing_cells(35.5), absorbing_cells(35), absorbing_cells(36)) == (36, 35, 36)
    # ...and the normalization accounts for the rest: two thicknesses sharing an extent
    # are still different absorbers, because prefac and u both carry the raw thickness.
    same_extent = PML(grid=grid, thickness={"z": 35.4}).get_sigma_profile("z")
    assert half_cell_extent(35.4) == half_cell_extent(35.5)
    assert numpy.count_nonzero(same_extent) == numpy.count_nonzero(exact)
    assert not numpy.allclose(same_extent, exact, rtol=1e-4)


def test_the_profile_argument_uses_the_raw_thickness_so_u_can_exceed_one():
    """``u = x/dx`` normalizes by the requested thickness, not by the snapped extent.

    structure.cpp:683 divides by ``dx`` — the number the caller asked for — while the
    numerator is quantized to ``N`` half cells, so ``u`` at the wall is ``N/(2*c)`` and
    is exactly 1 only when ``2*c`` is a whole number. Renormalising to ``(N - n)/N`` so
    that ``u`` tops out at 1 is the change a reviewer asks for (``u > 1`` looks like a
    bug, and the documentation says u runs 0 to 1); it also collapses every thickness
    inside one half-cell bin onto a single absorber, which
    ``test_a_fractional_layer_is_neither_of_its_whole_cell_neighbours`` shows MEEP does
    not do.
    """
    grid = Grid(resolution=71.0, cell_size=(1.0, 1.0, 2.0))
    for cells, u_wall in ((0.37 * 71, 1.008755), (0.123 * 71, 0.973320), (35.5, 1.0)):
        pml = PML(grid=grid, thickness={"z": (0, cells)})
        # Cell 0 holds the wrapped lattice point at half-cell index 2N, i.e. ON the
        # upper wall, so its sigma is prefac * u_wall**order with no distance term.
        peak = pml.get_sigma_profile("z")[0] / pml.prefac(cells)
        assert peak == pytest.approx(u_wall**2, rel=1e-5), (
            f"{cells} cells: u at the wall is {math.sqrt(peak):.6f}, expected {u_wall:.6f} "
            f"= N/(2*c) with N = {half_cell_extent(cells)}"
        )


def test_a_sub_quarter_cell_layer_grades_nothing_because_meep_builds_none():
    """Below a quarter cell MEEP's ``found_pml`` never trips and no sigma array exists.

    structure.cpp:648-654 scans for a point with ``x > 0`` and returns before allocating
    when it finds none; ``N = 0`` makes that certain. Ceiling such a request to one half
    cell would absorb where MEEP does not, and raising would refuse a configuration MEEP
    accepts — so the layer is built and grades nothing, which is what MEEP steps.
    """
    grid = make_grid()
    for cells in (0.1, 0.2, 0.24):
        pml = PML(grid=grid, thickness={"z": cells})
        numpy.testing.assert_array_equal(pml.get_sigma_profile("z"), 0.0)
        for name in COEFFICIENT_NAMES:
            numpy.testing.assert_array_equal(ravel(getattr(pml, name)), 1.0)
        assert pml.interior_slice() == (0, grid.nx, 0, grid.ny, 0, grid.nz)
    # Positive control: a quarter cell is the first thickness that grades anything.
    assert numpy.count_nonzero(PML(grid=grid, thickness={"z": 0.25}).get_sigma_profile("z")) > 0


def per_face_coefficients_before_fractional(grid: Grid, table, order: int = 2,
                                            R_asymptotic: float = 1e-15) -> dict:
    """Transcription of the per-face coefficient table as it stood before fractional support.

    ``u = 1 - |i - i_wall| / (2 * n_pml)`` with the extent test ``u > 0``, each face
    carrying its own ``prefac``, and the owned-index remap on a wrapping axis — the
    whole-cell special case of MEEP's ``pml_x``, written in the float association order
    the code used before the general form was threaded through it. Reproduced
    independently so that "a whole number of cells costs nothing" is a statement about
    the arithmetic rather than about the new code agreeing with itself.
    """
    out = {}
    for axis, n_cells, wraps in (("x", grid.nx, grid.axis_wraps(0)),
                                 ("y", grid.ny, grid.axis_wraps(1)),
                                 ("z", grid.nz, grid.axis_wraps(2))):
        faces = table[("x", "y", "z").index(axis)]
        # The upper wall is the window top, 2 * halve()'s num (owned_cells) —
        # one cell short of the stored end on a folded periodic even axis.
        wall_high = 2 * grid.owned_cells(("x", "y", "z").index(axis))
        for offset, suffix in ((0.0, ""), (0.5, "_h")):
            sig = numpy.zeros(n_cells, dtype=numpy.float32)
            half_index = 2 * numpy.arange(n_cells) + int(2 * offset)
            if wraps:
                half_index = (half_index - 1) % (2 * n_cells) + 1
            for wall, n_pml in ((0, faces[0]), (wall_high, faces[1])):
                if n_pml <= 0:
                    continue
                prefac = float(-numpy.log(R_asymptotic)
                               / (4.0 * (n_pml * grid.dx) * (1.0 / (order + 1))))
                scale = 0.5 * grid.dt * prefac
                u = 1.0 - numpy.abs(half_index - wall) / (2.0 * n_pml)
                inside = u > 0
                sig[inside] = scale * u[inside] ** order
            kap = numpy.ones(n_cells, dtype=numpy.float32)
            out[f"kms_{axis}{suffix}"] = kap - sig
            out[f"sinv_{axis}{suffix}"] = 1.0 / (kap + sig)
            out[f"kps_{axis}{suffix}"] = kap + sig
    return out


@pytest.mark.parametrize("overrides", [{}, {"symmetry": ("X",)}, {"symmetry": ("X", "Y")}])
@pytest.mark.parametrize("thickness", [
    1, 4, 7, {"z": (5, 0)}, {"z": (0, 5)}, {"z": (8, 4)},
    {"x": {"high": 3}, "y": {"high": 6}, "z": (9, 2)},
])
def test_a_whole_cell_layer_is_byte_identical_under_fractional_support(overrides, thickness):
    """Every whole-cell number this engine has measured must survive the fractional form.

    The general form ``u = (N - n) / (2*c)`` is mathematically identical to the old
    ``u = 1 - n / (2*c)`` when ``N == 2*c``, and numerically different at DOUBLE: it
    moves 247 of the first 256 integer thicknesses. The engine therefore writes
    ``u = 1 + ((N - 2*c) - n) / (2*c)``, whose ``N - 2*c`` term is exactly 0.0 for a
    whole-cell face and whose remaining arithmetic is the old expression operation for
    operation. Bytes, not allclose: a tolerance would pass a reordered expression and
    quietly move the CPU-MEEP floors recorded in ``FdtdDriver.setup_pml`` (2.9e-07
    uniform, 5.5e-07 one-sided, 4.9e-07 asymmetric).

    WHAT THIS PIN DOES NOT COVER, measured rather than assumed: the double-precision
    reassociation above does not reach these bytes. Sigma is stored float32, and over
    4236 whole-cell sigma arrays built both ways — three resolutions, three cell sizes,
    every (low, high) pair up to 8, both Yee offsets — 2952 differ at double and 0 of
    388,600 stored float32 entries differ. Substituting the plain form into
    ``pml._graded_sigma`` passes all 142 tests in this file, these 21 combinations
    included. The defensive spelling is kept because it makes the whole-cell path
    provably untouched by the fractional arithmetic, not because this pin can see it.
    """
    grid = make_grid(**overrides)
    pml = PML(grid=grid, thickness=thickness)
    expected = per_face_coefficients_before_fractional(grid, pml.thickness_by_face)
    for name in COEFFICIENT_NAMES:
        actual = numpy.asarray(getattr(pml, name), dtype=numpy.float32).ravel()
        assert actual.tobytes() == expected[name].astype(numpy.float32).tobytes(), (
            f"{name} moved for the whole-cell thickness={thickness} on grid {overrides}"
        )
    # Positive control: the comparison is on stored float32, so it can only see a
    # change above that type's 1.192e-07 resolution. prefac carries R's logarithm, so a
    # relative nudge of R shrinks by 1/ln(1e-15) = 1/34.5 on the way in: 1e-05 in R is
    # 2.895e-07 in prefac, 2.4 float32 eps. Bisected over these 21 combinations, the
    # smallest nudge that moves any byte at all is 2.540e-06 (7.355e-08 in prefac, 0.62
    # float32 eps), so 1e-05 clears the threshold by 3.9x and no more — it is a control
    # on the pin's real resolution rather than 394x above it, which is where the 1e-03
    # this used to nudge by sat.
    if pml.is_active:
        nudged = per_face_coefficients_before_fractional(
            grid, pml.thickness_by_face, R_asymptotic=1e-15 * 1.00001)
        assert any(nudged[name].astype(numpy.float32).tobytes() != expected[name].tobytes()
                   for name in COEFFICIENT_NAMES)


# --- Outer boundary conditions: periodic (the default) vs metallic (MEEP's) --------------


def test_the_default_boundary_is_periodic_on_every_axis_and_costs_nothing_to_spell():
    # The default must stay periodic: every floor this engine records was measured in
    # it, and MEEP's opposite default is reached through from_meep rather than here.
    implicit = make_grid()
    assert implicit.boundaries == (("periodic", "periodic"),) * 3
    assert implicit.has_metallic is False
    assert implicit.metallic_axes == (False, False, False)
    for axis in range(3):
        assert implicit.axis_wraps(axis) is True
        assert implicit.is_metallic(axis) is False
        assert implicit.boundary_condition(axis, "low") == "periodic"
        assert implicit.boundary_condition(axis, "high") == "periodic"
    # Saying it out loud resolves to the identical table, in every spelling.
    for spelling in ("periodic", ("periodic",) * 3, {"x": "periodic"}, (),
                     [("periodic", "periodic")] * 3, None):
        assert make_grid(boundaries=spelling).boundaries == implicit.boundaries


def test_every_metallic_spelling_resolves_to_the_same_per_axis_table():
    everywhere = make_grid(boundaries="metallic").boundaries
    assert everywhere == (("metallic", "metallic"),) * 3
    for spelling in (("metallic",) * 3, [("metallic", "metallic")] * 3,
                     {"x": "metallic", "y": "metallic", "z": "metallic"}):
        assert make_grid(boundaries=spelling).boundaries == everywhere
    # A per-axis request leaves the axes it does not name periodic — the same rule
    # setup_pml's mapping follows, so the two arguments read alike.
    one_axis = make_grid(boundaries={"y": "metallic"})
    assert one_axis.boundaries == (("periodic", "periodic"), ("metallic", "metallic"),
                                   ("periodic", "periodic"))
    assert one_axis.metallic_axes == (False, True, False)
    assert one_axis.metallic_axis_names == ("y",)
    assert [one_axis.axis_wraps(axis) for axis in range(3)] == [True, False, True]
    assert make_grid(boundaries=("periodic", "metallic", "periodic")).boundaries == one_axis.boundaries
    assert "metallic=y" in repr(one_axis)


def test_a_boundary_request_that_cannot_mean_anything_raises():
    for spelling, message in (
        ("magnetic", "this engine steps"),                    # MEEP's PMC: not the same wall.
        (("metallic", "none", "periodic"), "this engine steps"),
        ({"w": "metallic"}, "not axes"),
        (("metallic", "metallic"), "one entry per axis"),
        (("metallic",) * 4, "one entry per axis"),
        ((("metallic", "metallic", "metallic"), "periodic", "periodic"), "must be a condition"),
        ((7, "periodic", "periodic"), "must be a condition"),
    ):
        with pytest.raises(ValueError, match=message):
            make_grid(boundaries=spelling)


def test_an_axis_may_not_declare_one_condition_per_side():
    # Representable because MEEP's own table is per side; refused because MEEP reads
    # boundaries[High][d] alone when it decides whether a face translates, so the low
    # declaration would be silently ignored rather than honoured.
    for pair in (("metallic", "periodic"), ("periodic", "metallic")):
        with pytest.raises(ValueError, match="low face and") as raised:
            make_grid(boundaries=(pair, "periodic", "periodic"))
        assert "boundaries[High][d]" in str(raised.value)
    # The agreeing pair spelling is accepted, and is the same as the bare condition.
    assert (make_grid(boundaries=(("metallic", "metallic"), "periodic", "periodic")).boundaries
            == make_grid(boundaries={"x": "metallic"}).boundaries)


def test_a_bloch_phase_and_a_metallic_wall_cannot_share_an_axis():
    # A phase says what the field is one LATTICE VECTOR away, and a wall gives the
    # axis none. MEEP cannot express the pair either: use_bloch sets the axis Periodic.
    with pytest.raises(ValueError, match="on an axis declared 'metallic'") as raised:
        make_grid(boundaries={"x": "metallic"}, k_point=(0.3, 0.0, 0.0))
    assert "use_bloch" in str(raised.value)
    # The same k on a PERIODIC axis of the same run is fine — the refusal is per axis.
    mixed = make_grid(boundaries={"x": "metallic"}, k_point=(0.0, 0.0, 0.25))
    assert mixed.bloch_phase(0) is None and mixed.bloch_phase(2) is not None
    assert mixed.has_bloch is True


def test_a_folded_axis_may_also_be_metallic_which_is_meeps_own_default_cell():
    # mp.Simulation(symmetries=[mp.Mirror(mp.X)]) with no k_point is BOTH, and the two
    # are independent mechanisms in MEEP as well (symmetry vs fields::boundaries).
    grid = make_grid(symmetry=("X",), boundaries="metallic")
    assert grid.is_mirrored(0) is True and grid.is_metallic(0) is True
    assert grid.boundary_condition(0, "high") == "metallic"
    assert grid.axis_wraps(0) is False  # Neither mechanism leaves a lattice vector.
    # The wall changes no origin, but it DOES change the storage: a metallic fold
    # keeps MEEP's halved window (its window-top plane is zero_metal-held, so the
    # zero ghost is its value), while a periodic fold stores the second-mirror
    # plane and its ghost slot on top of it.
    periodic_fold = make_grid(symmetry=("X",))
    assert grid.nx == grid.owned_cells(0) == periodic_fold.nx - 1
    assert [grid.origin_doubled(a) for a in range(3)] == [
        periodic_fold.origin_doubled(a) for a in range(3)]


@pytest.mark.parametrize("cell_size", [(2.0, 2.0, 4.0), (1.9, 1.1, 2.1)])  # Even and odd counts.
def test_a_metallic_wall_changes_no_registration(cell_size):
    # The wall is a boundary CONDITION: MEEP's center_origin runs before any of it, so
    # the cell counts, the origins and the coordinate arrays must be untouched. Getting
    # this wrong is half a cell everywhere and still a smooth field (Grid.origin_doubled).
    periodic = Grid(resolution=10.0, cell_size=cell_size)
    metallic = Grid(resolution=10.0, cell_size=cell_size, boundaries="metallic")
    assert metallic.shape == periodic.shape and metallic.shape_full == periodic.shape_full
    for axis in range(3):
        assert metallic.origin_doubled(axis) == periodic.origin_doubled(axis)
        assert metallic.axis_origin(axis) == periodic.axis_origin(axis)
    for name in ("x", "y", "z"):
        assert numpy.array_equal(getattr(metallic, name), getattr(periodic, name))
    # And position_to_index still names every stored cell of every axis.
    for axis_name, coordinates in (("x", metallic.x), ("y", metallic.y), ("z", metallic.z)):
        for cell, position in enumerate(coordinates):
            low, weight = metallic.position_to_index(float(position), axis_name, iyee_shift=1)
            assert low + weight == pytest.approx(cell, abs=1e-4)


def test_position_to_index_past_a_metallic_axis_names_the_wall_not_the_fold():
    # Three different "outside the axis" stories with three different fixes; the message
    # must not offer the lattice-fold path to an axis that has no lattice.
    grid = make_grid(boundaries={"x": "metallic"})
    with pytest.raises(ValueError, match="outside the x axis") as raised:
        grid.position_to_index(grid.Lx, "x", iyee_shift=1)
    text = str(raised.value)
    assert "terminated by metallic walls" in text
    assert "mirror plane folds it" not in text
    assert "_build_source_points" not in text, (
        "A metallic axis has no lattice vector, so the wrapping placement path cannot help."
    )
    # Its periodic sibling still gets the lattice answer, in the same grid.
    with pytest.raises(ValueError, match="_build_source_points"):
        grid.position_to_index(grid.Ly, "y", iyee_shift=1)


def test_boundary_condition_refuses_an_axis_or_a_side_it_cannot_resolve():
    grid = make_grid(boundaries="metallic")
    for axis in (-1, 3, "x", None):
        with pytest.raises(ValueError, match="axis must be 0"):
            grid.boundary_condition(axis, "low")
        with pytest.raises(ValueError, match="axis must be 0"):
            grid.is_metallic(axis)
    for side in ("High", "LOW", 0, 1, None):
        with pytest.raises(ValueError, match="side must be one of"):
            grid.boundary_condition(0, side)


# --- where the layer bites, and which cells MEEP's chunking hands to it ------------


def test_sigma_bites_matches_the_stored_grading_at_both_yee_offsets():
    """`sigma_bites` is the coefficient tables' own support, not a re-derivation.

    The integer set must agree with `get_sigma_profile` (the same grading pass), the
    first cell AT the snapped extent must read False with no tolerance (MEEP's u = 0
    sample carries sigma exactly 0 — why "touching the inner edge" is exact), and the
    half-integer set is the B-side support, whose deepest graded low-face sample is
    cell extent/2 - 1 (distance 2j+1 < extent) — half a cell shy of the integer edge.
    """
    grid = Grid(resolution=20.0, cell_size=(12.0, 12.0, 0.0), dimensions=2)
    layer = PML(grid, thickness={"x": 40, "y": 40})
    integer = layer.sigma_bites("x")
    assert integer.shape == (grid.nx,)
    assert numpy.array_equal(integer, layer.get_sigma_profile("x") != 0.0)
    # 40-cell faces on a wrapping axis: cell 0 is the image of the high wall (bites),
    # cells 1..39 grade the low face, cell 40 sits AT the extent with sigma exactly 0.
    assert bool(integer[0]) is True and bool(integer[39]) is True
    assert bool(integer[40]) is False and bool(integer[120]) is False
    assert bool(integer[grid.nx - 40]) is False and bool(integer[grid.nx - 39]) is True
    half = layer.sigma_bites("x", half_integer=True)
    assert bool(half[0]) is True and bool(half[39]) is True  # half-index 79 < 80.
    assert bool(half[40]) is False  # half-index 81 is already outside the extent.
    # An axis with no layer bites nowhere at either offset.
    assert not layer.sigma_bites("z").any()
    assert not layer.sigma_bites("z", half_integer=True).any()


def test_in_pml_chunk_owns_one_integer_plane_past_the_inner_edge():
    """The chunk rule is structure.cpp:509-524 ownership, NOT the sigma support.

    MEEP's per-side effort volume is `int(cells + 1.5)` array cells from the wall and
    chunk selection reads sigsize per CHUNK (step_db.cpp:56-59), so the boundary
    cells whose own sigma is exactly zero still step the both-active branch.
    Measured discriminators on the 12x12 evidence-packet cell (CW Ez at x = -5,
    against CPU MEEP): deposits at y = -4.0 (cell 40) and y = -3.975 (cells 40+41)
    are both-active — mirroring either reads 2.3e-01 … 4.6e-01 where leaving them
    split reads ~7e-07 — while y = -3.9 (cell 42) runs unsplit (4.9583e-01
    unmirrored, 5.6135e-07 mirrored). Integer samples: a chunk spanning doubled
    [c0, c1] owns c0+2..c1, so the low chunk owns cells 1..span and the boundary
    plane belongs to it, not to the interior neighbour; half-integer samples are
    c0+1..c1-1, cells 0..span-1.
    """
    grid = Grid(resolution=20.0, cell_size=(12.0, 12.0, 0.0), dimensions=2)
    layer = PML(grid, thickness={"x": 40, "y": 40})
    integer = layer.in_pml_chunk("y")
    assert bool(integer[40]) is True and bool(integer[41]) is True  # the measured pair
    assert bool(integer[42]) is False  # the measured unsplit cell
    assert bool(integer[grid.ny - 41]) is False and bool(integer[grid.ny - 40]) is True
    assert bool(integer[0]) is True, "cell 0 is the high wall's own plane on a wrapping axis"
    half = layer.in_pml_chunk("y", half_integer=True)
    assert bool(half[0]) is True and bool(half[40]) is True
    assert bool(half[41]) is False
    assert bool(half[grid.ny - 41]) is True and bool(half[grid.ny - 42]) is False
    assert not layer.in_pml_chunk("z").any()


def test_beta_is_carried_as_a_scalar_and_refused_off_a_2d_cartesian_grid():
    """MEEP's out-of-plane 2-D wavevector: a Grid field, legal only on a D2 grid.

    ``beta`` is what ``mp.Simulation(kz_2d=...)`` turns ``k_point.z`` into on a
    zero-thickness cell (python/simulation.py:1555-1570). It is NOT a Bloch phase —
    MEEP hands ``use_bloch`` only ``Vector3(kx, ky)`` (:2498-2504) and passes this in
    the fields constructor's own slot (:2478-2487) — so it lives beside ``k_point``
    rather than in it, and it stays zero unless asked for.

    The refusal is MEEP's, verbatim: ``fields::is_aniso2d`` aborts with "Nonzero beta
    unsupported in dimensions other than 2" for any grid that is not D2
    (src/fields.cpp:546-547). A cylindrical grid is one of those, and a nonzero beta
    there would otherwise be carried silently into a stepper that never reads it.
    """
    grid = Grid(resolution=20.0, cell_size=(2.0, 2.0, 0.0), dimensions=2, beta=0.35)
    assert grid.beta == 0.35
    assert grid.k_point == (0.0, 0.0, 0.0), "beta is not a Bloch component and must not become one."
    assert Grid(resolution=20.0, cell_size=(2.0, 2.0, 0.0), dimensions=2).beta == 0.0

    for spelling in (
        dict(cell_size=(2.0, 2.0, 2.0), dimensions=3),
        dict(cell_size=(0.0, 0.0, 2.0), dimensions=1),
        dict(cell_size=(2.0, 0.0, 2.0), cylindrical=True),
    ):
        with pytest.raises(ValueError, match="only on a 2-D Cartesian grid"):
            Grid(resolution=20.0, beta=0.35, **spelling)
