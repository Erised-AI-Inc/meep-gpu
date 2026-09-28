"""
Unit tests for the FDTD field storage (fields.Fields).

These cover the conventions the rest of the engine is written against: the Yee
offset table, the mirror parities derived from MEEP's ``symmetry::phase_shift``
(any axis, either declared phase), cell-centering (periodic on full axes,
metallic at the far edge of a symmetry-halved one), quadrant-to-full-domain
reconstruction with those parities, MEEP get_array() output geometry, the lazy
PML storage switch, and the loud failures on unknown component names.

Everything here runs on the NumPy path — no CuPy, no CUDA, no MEEP — so the
numerics stay verifiable on any developer machine. Grids are kept tiny
(<= 40 cells per axis) so the whole module runs in well under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

from .dispersion import LORENTZIAN, PolarizationState, Susceptibility
from .fields import (
    IYEE_SHIFTS,
    SYMMETRY_PHASES,
    Fields,
    get_symmetry_phase,
    mirror_parity,
    require_even_xy_planes_only,
)
from .grid import Grid, Mirror

ALL_COMPONENTS = ('Ex', 'Ey', 'Ez', 'Dx', 'Dy', 'Dz', 'Hx', 'Hy', 'Hz', 'Bx', 'By', 'Bz')


def _make_grid(nx: int, ny: int, nz: int, symmetry: tuple = (),
               boundaries=None) -> Grid:  # Unit-dx grid with the requested full dimensions.
    return Grid(resolution=1.0, cell_size=(float(nx), float(ny), float(nz)), symmetry=symmetry,
                boundaries=boundaries)


def _ramp(shape, axis: int) -> np.ndarray:  # Distinct float32 values varying along one axis only.
    values = np.arange(1, shape[axis] + 1, dtype=np.float32)
    broadcast_shape = [1, 1, 1]
    broadcast_shape[axis] = shape[axis]
    return np.broadcast_to(values.reshape(broadcast_shape), shape).astype(np.float32)


def _distinct(shape) -> np.ndarray:  # A field with a different value in every cell.
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) + 1.0


def test_iyee_shift_table_spot_checks():
    assert IYEE_SHIFTS['Ex'] == (1, 0, 0)
    assert IYEE_SHIFTS['Ey'] == (0, 1, 0)
    assert IYEE_SHIFTS['Ez'] == (0, 0, 1)
    assert IYEE_SHIFTS['Hx'] == (0, 1, 1)
    assert IYEE_SHIFTS['Hy'] == (1, 0, 1)
    assert IYEE_SHIFTS['Hz'] == (1, 1, 0)
    assert set(IYEE_SHIFTS) == set(ALL_COMPONENTS)


def test_iyee_shift_table_structure():
    for electric, displacement in (('Ex', 'Dx'), ('Ey', 'Dy'), ('Ez', 'Dz')):
        assert IYEE_SHIFTS[electric] == IYEE_SHIFTS[displacement]
        assert sum(IYEE_SHIFTS[electric]) == 1  # E/D: half-integer in its own direction only.
    for magnetic, induction in (('Hx', 'Bx'), ('Hy', 'By'), ('Hz', 'Bz')):
        assert IYEE_SHIFTS[magnetic] == IYEE_SHIFTS[induction]
        assert sum(IYEE_SHIFTS[magnetic]) == 2  # H/B: half-integer in both transverse directions.
    for component, shifts in IYEE_SHIFTS.items():
        # At least one zero per component is what guarantees to_cell_center always
        # allocates instead of returning the stored array aliased.
        assert 0 in shifts, component


def test_symmetry_phase_lookup():
    assert get_symmetry_phase('Ex', 'x') == -1
    assert get_symmetry_phase('Ex', 'y') == +1
    assert get_symmetry_phase('Hz', 'x') == -1
    assert get_symmetry_phase('Hz', 'y') == -1
    for electric, displacement in (('Ex', 'Dx'), ('Ey', 'Dy'), ('Ez', 'Dz')):
        assert SYMMETRY_PHASES[electric] == SYMMETRY_PHASES[displacement]
    for magnetic, induction in (('Hx', 'Bx'), ('Hy', 'By'), ('Hz', 'Bz')):
        assert SYMMETRY_PHASES[magnetic] == SYMMETRY_PHASES[induction]


# The table the engine hardcoded before the parity was derived, transcribed here
# from the module docstring it used to carry. It is the regression guard on the
# derivation: `mirror_parity(c, axis, +1)` must reproduce every entry, so a
# rewritten rule cannot quietly re-sign an even X/Y mirror while the folded runs
# still "look" symmetric.
LEGACY_EVEN_XY_PHASES = {
    'Ex': (-1, +1), 'Ey': (+1, -1), 'Ez': (+1, +1),
    'Dx': (-1, +1), 'Dy': (+1, -1), 'Dz': (+1, +1),
    'Hx': (+1, -1), 'Hy': (-1, +1), 'Hz': (-1, -1),
    'Bx': (+1, -1), 'By': (-1, +1), 'Bz': (-1, -1),
}


def test_the_derived_parities_reproduce_the_hardcoded_even_xy_table():
    assert SYMMETRY_PHASES == LEGACY_EVEN_XY_PHASES
    for component, (phase_x, phase_y) in LEGACY_EVEN_XY_PHASES.items():
        assert mirror_parity(component, 0) == phase_x, component
        assert mirror_parity(component, 1) == phase_y, component
        assert get_symmetry_phase(component, 'x') == phase_x
        assert get_symmetry_phase(component, 'y') == phase_y


def test_the_even_xy_table_refuses_the_planes_it_cannot_express():
    """A consumer of ``SYMMETRY_PHASES`` must refuse an odd or Z plane, not fold it wrongly.

    The table is ``mirror_parity`` frozen at phase = +1 on X and Y. Read through it a
    Z plane is invisible and an odd plane comes back with every sign inverted, and
    both produce a full, plausible field — the exact failure this package treats as a
    defect. Nothing on the live path reads the table any more, but the deferred fused
    CUDA kernels still do, so the rule lives here, beside the table, where it can be
    tested without a CUDA device.

    Both halves matter: the four folds the table CAN express must still pass, or the
    guard is just a ban on symmetry.
    """
    for symmetry in ((), ("X",), ("Y",), ("X", "Y")):
        grid = Grid(resolution=10, cell_size=(2.0, 2.0, 2.0), symmetry=symmetry, xp=np)
        require_even_xy_planes_only(grid)  # Must not raise: the table has every entry.

    for symmetry in ((Mirror("X", -1),), (Mirror("Y", -1),), (Mirror("Z", +1),),
                     (Mirror("Z", -1),), (Mirror("X", +1), Mirror("Z", -1))):
        grid = Grid(resolution=10, cell_size=(2.0, 2.0, 2.0), symmetry=symmetry, xp=np)
        with pytest.raises(NotImplementedError, match="SYMMETRY_PHASES"):
            require_even_xy_planes_only(grid)


def test_mirror_parity_matches_meeps_phase_shift_rule_on_every_axis_and_phase():
    """A true vector is odd about its own axis; a pseudovector is even about its own.

    MEEP vec.cpp ``symmetry::phase_shift``: a single mirror flips exactly one
    direction and permutes none, so ``flip`` starts as ``dir(c) == axis``, the
    magnetic branch toggles it once, and the handedness term never fires. The
    declared phase multiplies the result. Restated here independently of the
    implementation so a rewrite of one is checked against the other.
    """
    for phase in (+1, -1):
        for axis in range(3):
            for component in ('Ex', 'Ey', 'Ez', 'Dx', 'Dy', 'Dz'):
                own = 'xyz'.index(component[1])
                assert mirror_parity(component, axis, phase) == phase * (-1 if own == axis else +1)
            for component in ('Hx', 'Hy', 'Hz', 'Bx', 'By', 'Bz'):
                own = 'xyz'.index(component[1])
                assert mirror_parity(component, axis, phase) == phase * (+1 if own == axis else -1)


def test_an_odd_mirror_inverts_every_parity_and_a_z_mirror_is_the_x_rule_rotated():
    # phase = -1 is the SAME fold with every sign flipped — the property that makes
    # an Ex dipole on x = 0 (odd under an even mirror) foldable at all.
    for component in ALL_COMPONENTS:
        for axis in range(3):
            assert mirror_parity(component, axis, -1) == -mirror_parity(component, axis, +1)
    # Z is not a special axis: Ez is odd about z the way Ex is odd about x, and Hz
    # is even about z the way Hx is even about x.
    assert mirror_parity('Ez', 2) == -1
    assert mirror_parity('Ex', 2) == +1
    assert mirror_parity('Hz', 2) == +1
    assert mirror_parity('Hx', 2) == -1
    # A Jz dipole on z = 0 needs the odd mirror, exactly as a Jx dipole on x = 0 does.
    assert mirror_parity('Ez', 2, -1) == +1


def test_symmetry_phase_rejects_unknown_inputs():
    with pytest.raises(ValueError, match="unknown field component"):
        get_symmetry_phase('Qx', 'x')
    with pytest.raises(ValueError, match="direction must be"):
        get_symmetry_phase('Ex', 'w')
    with pytest.raises(ValueError, match="mirror axis must be"):
        mirror_parity('Ex', 3)
    with pytest.raises(ValueError, match="mirror phase must be"):
        mirror_parity('Ex', 0, 0)
    with pytest.raises(ValueError, match="mirror phase must be"):
        get_symmetry_phase('Ex', 'x', 2)


def test_allocation_shapes_and_dtypes():
    grid = _make_grid(8, 6, 10)
    fields = Fields(grid=grid)
    for name in ('Dx', 'Dy', 'Dz', 'Bx', 'By', 'Bz'):
        array = getattr(fields, name)
        assert array.shape == grid.shape
        assert array.dtype == np.float32
        assert not array.any()
    for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'fu_Dx', 'f_w_Ex'):
        assert getattr(fields, name) is None  # Not allocated until PML is enabled.
    assert fields.eps.dtype == np.float32
    assert fields.inv_eps.dtype == np.float32

    complex_fields = Fields(grid=grid, force_complex_fields=True)
    assert complex_fields.Dx.dtype == np.complex64
    assert complex_fields.eps.dtype == np.float32  # Material stays float32 in complex runs.


def test_get_component_rejects_unknown_name():
    fields = Fields(grid=_make_grid(4, 4, 4))
    for bad_name in ('Ee', 'ex', 'E', '', 'Jz'):
        with pytest.raises(ValueError, match="Invalid component name"):
            fields.get_component(bad_name)


def test_get_e_computes_from_d_without_pml():
    fields = Fields(grid=_make_grid(4, 4, 6))
    fields.set_background_eps(4.0)
    fields.Dx[:] = _distinct(fields.Dx.shape)

    electric = fields.get_E('Ex')
    np.testing.assert_allclose(electric, fields.Dx * 0.25, rtol=0, atol=0)
    assert electric is not fields.Dx  # Freshly allocated, safe to mutate.
    assert fields.Ex is None
    np.testing.assert_array_equal(fields.get_component('Ex'), electric)


def test_get_h_aliases_b_without_pml():
    fields = Fields(grid=_make_grid(4, 4, 6))
    fields.By[:] = 3.0
    # Documented (and load-bearing) aliasing: mu = 1 means H is the B array itself.
    assert fields.get_H('Hy') is fields.By
    assert fields.get_component('Hy') is fields.By


def test_enable_pml_storage_allocates_and_switches_modes():
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=True)
    assert fields._pml_active is False

    fields.enable_pml_storage()
    assert fields._pml_active is True

    pml_names = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz',
                 'fu_Bx', 'fu_By', 'fu_Bz', 'fu_Dx', 'fu_Dy', 'fu_Dz',
                 'f_w_Ex', 'f_w_Ey', 'f_w_Ez', 'f_w_Hx', 'f_w_Hy', 'f_w_Hz']
    assert len(pml_names) == 18
    for name in pml_names:
        array = getattr(fields, name)
        assert array is not None, name
        assert array.shape == grid.shape
        assert array.dtype == np.complex64

    # Mode switch: E/H now come from storage, not from D/inv_eps or B.
    fields.Dx[:] = 5.0
    fields.Bx[:] = 7.0
    assert fields.get_E('Ex') is fields.Ex
    assert fields.get_H('Hx') is fields.Hx
    assert not np.asarray(fields.get_E('Ex')).any()


def test_enable_pml_storage_is_idempotent():
    fields = Fields(grid=_make_grid(4, 4, 6))
    fields.enable_pml_storage()
    stored_e = fields.Ex
    stored_aux = fields.fu_Dz
    fields.Ex[:] = 2.5
    fields.fu_Dz[:] = -1.5

    fields.enable_pml_storage()  # Second call must not discard accumulated state.
    assert fields.Ex is stored_e
    assert fields.fu_Dz is stored_aux
    assert float(fields.Ex[0, 0, 0]) == 2.5
    assert float(fields.fu_Dz[0, 0, 0]) == -1.5


def test_reset_zeros_primary_and_pml_arrays():
    fields = Fields(grid=_make_grid(4, 4, 6))
    fields.Dz[:] = 1.0
    fields.reset()
    assert not fields.Dz.any()

    fields.enable_pml_storage()
    fields.Ez[:] = 1.0
    fields.f_w_Hz[:] = 1.0
    fields.Bx[:] = 1.0
    fields.reset()
    assert not fields.Ez.any()
    assert not fields.f_w_Hz.any()
    assert not fields.Bx.any()


def test_material_setters():
    grid = _make_grid(8, 8, 8)
    fields = Fields(grid=grid)
    np.testing.assert_allclose(fields.eps, 1.0)
    np.testing.assert_allclose(fields.inv_eps, 1.0)

    fields.set_background_eps(2.0)
    np.testing.assert_allclose(fields.eps, 2.0)
    np.testing.assert_allclose(fields.inv_eps, 0.5)
    assert fields.eps.dtype == np.float32

    fields.add_sphere(center=(0.0, 0.0, 0.0), radius=2.0, eps=9.0)
    inside = fields.eps[grid.nx // 2, grid.ny // 2, grid.nz // 2]
    assert float(inside) == 9.0
    assert float(fields.inv_eps[grid.nx // 2, grid.ny // 2, grid.nz // 2]) == pytest.approx(1.0 / 9.0)
    assert float(fields.eps[0, 0, 0]) == 2.0  # Corner is outside the sphere.
    np.testing.assert_allclose(fields.inv_eps, 1.0 / fields.eps, rtol=1e-6)


def test_component_epsilon_selects_the_matching_constitutive_array():
    grid = _make_grid(4, 4, 4)
    fields = Fields(grid=grid)
    baseline_bytes = fields.bytes_per_cell()
    epsilon = {
        "Ex": np.full(grid.shape, 2.0, dtype=np.float32),
        "Ey": np.full(grid.shape, 3.0, dtype=np.float32),
        "Ez": np.full(grid.shape, 4.0, dtype=np.float32),
    }
    inverse = {name: 1.0 / values for name, values in epsilon.items()}
    fields.set_epsilon_volumes(epsilon, inverse)
    fields.Dx.fill(2.0)
    fields.Dy.fill(3.0)
    fields.Dz.fill(4.0)

    assert fields.has_component_epsilon
    np.testing.assert_array_equal(fields.get_E("Ex"), 1.0)
    np.testing.assert_array_equal(fields.get_E("Ey"), 1.0)
    np.testing.assert_array_equal(fields.get_E("Ez"), 1.0)
    assert fields.epsilon_for("Ex")[0, 0, 0] == 2.0
    assert fields.epsilon_for("Ey")[0, 0, 0] == 3.0
    assert fields.epsilon_for("Ez")[0, 0, 0] == 4.0
    # Isotropic material owns two arrays; component material owns six.
    assert fields.bytes_per_cell() == baseline_bytes + 4 * 4


def test_to_cell_center_periodic_single_axis():
    grid = _make_grid(4, 4, 5)
    fields = Fields(grid=grid)
    # Bz has iyee_shift (1,1,0): only Z is interpolated, and Z is always periodic.
    fields.Bz[:] = _ramp(grid.shape, axis=2)
    centered = fields.to_cell_center('Bz')

    values = np.arange(1, grid.nz + 1, dtype=np.float32)
    expected_line = 0.5 * (values + np.roll(values, -1))
    assert expected_line[-1] == pytest.approx(0.5 * (grid.nz + 1))  # Wraps to cell 0.
    np.testing.assert_allclose(centered[2, 3, :], expected_line, rtol=0, atol=1e-6)
    assert centered.shape == grid.shape


def test_to_cell_center_periodic_two_axes():
    grid = _make_grid(4, 5, 3)
    fields = Fields(grid=grid)
    # Ez has iyee_shift (0,0,1): interpolate X then Y, both periodic on a full grid.
    fields.Dz[:] = _distinct(grid.shape)
    raw = fields.Dz.copy()
    centered = fields.to_cell_center('Dz')

    expected = 0.5 * (raw + np.roll(raw, -1, axis=0))
    expected = 0.5 * (expected + np.roll(expected, -1, axis=1))
    np.testing.assert_allclose(centered, expected, rtol=0, atol=1e-6)
    # A hand-checked corner: average of the four X-Y neighbours (with wrap).
    hand = 0.25 * (raw[3, 4, 1] + raw[0, 4, 1] + raw[3, 0, 1] + raw[0, 0, 1])
    assert centered[3, 4, 1] == pytest.approx(float(hand), rel=1e-6)


def test_to_cell_center_symmetry_axis_uses_metallic_far_edge():
    grid = _make_grid(8, 4, 4, symmetry=('X',), boundaries='metallic')
    assert grid.nx == 5  # owned_cells: a metallic fold keeps MEEP's halved window.
    fields = Fields(grid=grid)
    # Bx has iyee_shift (0,1,1): X is the only interpolated axis, and it is halved.
    fields.Bx[:] = _ramp(grid.shape, axis=0)
    centered = fields.to_cell_center('Bx')

    values = np.arange(1, grid.nx + 1, dtype=np.float32)
    expected_line = np.empty_like(values)
    expected_line[:-1] = 0.5 * (values[:-1] + values[1:])
    expected_line[-1] = 0.5 * values[-1]  # Metallic BC: the cell beyond the edge is zero.
    np.testing.assert_allclose(centered[:, 1, 2], expected_line, rtol=0, atol=1e-6)
    assert centered[-1, 1, 2] == pytest.approx(0.5 * grid.nx)
    # The same field on a full grid would wrap instead of terminating.
    full_grid_fields = Fields(grid=_make_grid(5, 4, 4))
    full_grid_fields.Bx[:] = _ramp(full_grid_fields.Bx.shape, axis=0)
    assert full_grid_fields.to_cell_center('Bx')[-1, 1, 2] == pytest.approx(0.5 * (5 + 1))
    # A PERIODIC fold stores the second-mirror plane cell and REFLECTS past it:
    # the far average reads parity * the image row, not a metallic zero
    # (stepping._far_reflect_rows' n_full - n_q + 2, here row 4).
    periodic = Fields(grid=_make_grid(8, 4, 4, symmetry=('X',)))
    assert periodic.grid.nx == 6  # owned_cells + the plane and its ghost slot.
    periodic.Bx[:] = _ramp(periodic.grid.shape, axis=0)
    line = periodic.to_cell_center('Bx')[:, 1, 2]
    np.testing.assert_allclose(line[:-1], 0.5 * (np.arange(1, 6) + np.arange(2, 7)),
                               rtol=0, atol=1e-6)
    assert line[-1] == pytest.approx(0.5 * (6 + 5))  # parity +1, image row 4 holds 5.


@pytest.mark.parametrize('axis,component', [(0, 'Bx'), (1, 'By'), (2, 'Bz')])
def test_to_cell_center_terminates_every_folded_axis_metallically(axis, component):
    """The metallic far edge is a property of a FOLDED axis, not of X and Y.

    Z used to take the periodic branch unconditionally ("Z never carries
    symmetry"), which on a folded Z would average the last stored plane against
    cell 0 — a sample half a cell on the far side of the mirror plane, several
    cells away. The result is smooth and plausible and wrong only at the edge.
    """
    cells = [4, 4, 4]
    cells[axis] = 8
    grid = _make_grid(*cells, symmetry=(('X', 'Y', 'Z')[axis],), boundaries='metallic')
    n_stored = grid.shape[axis]
    assert n_stored == cells[axis] // 2 + 1
    fields = Fields(grid=grid)
    # Bx/By/Bz are integer-positioned on x/y/z respectively, so the folded axis is
    # the one being interpolated and the other two are already at a cell centre.
    getattr(fields, component)[:] = _ramp(grid.shape, axis=axis)
    centered = fields.to_cell_center(component)

    values = np.arange(1, n_stored + 1, dtype=np.float32)
    expected = np.empty_like(values)
    expected[:-1] = 0.5 * (values[:-1] + values[1:])
    expected[-1] = 0.5 * values[-1]  # Metallic BC: the cell beyond the edge is zero.
    line = centered[_line(axis, centered.shape)]
    np.testing.assert_allclose(line, expected, rtol=0, atol=1e-6)
    assert line[-1] == pytest.approx(0.5 * n_stored)
    assert line[-1] != pytest.approx(0.5 * (n_stored + 1)), "a periodic wrap would give this"


def _line(axis, shape):  # Index tuple reading one line along `axis` through a fixed cell.
    index = [min(1, shape[other] - 1) for other in range(3)]
    index[axis] = slice(None)
    return tuple(index)


def test_to_cell_center_skips_axes_already_at_cell_center():
    grid = _make_grid(8, 4, 4, symmetry=('X',))
    fields = Fields(grid=grid)
    # Ex has iyee_shift (1,0,0): the halved X axis is untouched; Y and Z average.
    fields.Dx[:] = _ramp(grid.shape, axis=0)
    centered = fields.to_cell_center('Dx')
    np.testing.assert_allclose(centered[:, 0, 0], np.arange(1, grid.nx + 1), rtol=0, atol=1e-6)


@pytest.mark.parametrize('component', ALL_COMPONENTS)
def test_to_cell_center_never_aliases_or_mutates_storage(component):
    """Every component interpolates along at least one axis, so the result is
    always a fresh array — the reason the input copy is unnecessary."""
    grid = _make_grid(8, 6, 4, symmetry=('X', 'Y'))
    fields = Fields(grid=grid)
    for name in ('Dx', 'Dy', 'Dz', 'Bx', 'By', 'Bz'):
        getattr(fields, name)[:] = _distinct(grid.shape)
    before = {name: getattr(fields, name).copy() for name in ('Dx', 'Dy', 'Dz', 'Bx', 'By', 'Bz')}

    centered = fields.to_cell_center(component)
    for name, snapshot in before.items():
        assert centered is not getattr(fields, name)
        np.testing.assert_array_equal(getattr(fields, name), snapshot)


def test_to_cell_center_rejects_unknown_component():
    fields = Fields(grid=_make_grid(4, 4, 4))
    with pytest.raises(ValueError, match="Invalid component name"):
        fields.to_cell_center('Qz')


def _build_mirror_symmetric_full_field(shape_full, component, planes):
    """Full-domain array obeying the mirror parities of one component.

    ``planes`` maps axis -> declared mirror phase. Values on the stored side of
    each plane are arbitrary; the negative interior is generated from them with
    ``mirror_parity``, which is exactly the relation _reconstruct_full_domain has
    to reproduce.
    """
    rng = np.random.default_rng(20260728)
    full = rng.standard_normal(shape_full).astype(np.float32)
    for axis, mirror_phase in planes.items():
        phase = mirror_parity(component, axis, mirror_phase)
        centre = shape_full[axis] // 2
        for step in range(1, centre):
            full[_slab(axis, centre - 1 - step)] = phase * full[_slab(axis, centre + step)]
    return full


def _slab(axis, index):  # Index tuple selecting along one axis only.
    return (slice(None),) * axis + (index,)


# Every axis, both declared phases, and the combinations. The odd cases are what
# say the reconstruction reads the plane's phase instead of the even-mirror table.
RECONSTRUCTION_PLANES = {
    'x_even': {0: +1},
    'x_odd': {0: -1},
    'y_odd': {1: -1},
    'z_even': {2: +1},
    'z_odd': {2: -1},
    'xy_even': {0: +1, 1: +1},
    'xz_mixed': {0: -1, 2: +1},
    'xyz_odd': {0: -1, 1: -1, 2: -1},
}


@pytest.mark.parametrize('shape_full', [(8, 6, 4), (9, 7, 5)])  # Even and ODD folded counts.
@pytest.mark.parametrize('component', ['Ex', 'Ey', 'Ez', 'Hx', 'Hz', 'Bz'])
@pytest.mark.parametrize('case', sorted(RECONSTRUCTION_PLANES))
def test_quadrant_reconstruction_round_trips_every_plane_set(case, component, shape_full):
    """Fold-then-unfold is the identity for BOTH cell-count parities.

    The stored quadrant is ``full[centre - 1:]`` on each folded axis with
    ``centre = n_full // 2``, plus MEEP's ``big_corner`` cell (stored at both
    parities). The odd axis's top full-domain cell — centre at exactly +L/2 —
    has no lower image inside MEEP's shifted window, so the mirrored interior
    must start from ``quadrant[centre]``, not ``quadrant[n_q - 1]``. Starting
    from the quadrant count instead is silent at even counts and writes every
    odd-count mirror cell one cell off.
    """
    planes = RECONSTRUCTION_PLANES[case]
    grid = _make_grid(*shape_full, symmetry=tuple(
        Mirror('XYZ'[axis], phase) for axis, phase in sorted(planes.items())))
    fields = Fields(grid=grid)
    full = _build_mirror_symmetric_full_field(grid.shape_full, component, planes)

    quadrant = full
    for axis in planes:  # Stored half: cell 0 sits at -0.5*dx on each folded axis.
        sliced = quadrant[_slab(axis, slice(grid.shape_full[axis] // 2 - 1, None))]
        # A periodic fold also stores MEEP's big_corner cell, at BOTH parities —
        # on the full periodic domain that is cell 0's own sample one lattice
        # vector up (no phase at k = 0). The unfold drops it, so the round trip
        # is unchanged by its value; it is built faithfully here.
        sliced = np.concatenate((sliced, quadrant[_slab(axis, slice(0, 1))]),
                                axis=axis)
        quadrant = sliced
    assert quadrant.shape == grid.shape

    reconstructed = fields._reconstruct_full_domain(quadrant, component)
    assert reconstructed.shape == grid.shape_full
    np.testing.assert_array_equal(reconstructed, full)


def test_quadrant_reconstruction_applies_the_component_phase():
    grid = _make_grid(8, 6, 4, symmetry=('Y',))
    fields = Fields(grid=grid)
    quadrant = _distinct(grid.shape)
    cy = grid.ny_full // 2

    even = fields._reconstruct_full_domain(quadrant, 'Ez')  # phase_y = +1
    odd = fields._reconstruct_full_domain(quadrant, 'Ey')   # phase_y = -1
    np.testing.assert_array_equal(even[:, cy - 1, :], quadrant[:, 0, :])  # Boundary cell: no phase.
    np.testing.assert_array_equal(odd[:, cy - 1, :], quadrant[:, 0, :])
    np.testing.assert_array_equal(even[:, cy - 2, :], quadrant[:, 2, :])
    np.testing.assert_array_equal(odd[:, cy - 2, :], -quadrant[:, 2, :])


def test_quadrant_reconstruction_reads_the_declared_phase_not_the_even_table():
    """The same component, the same axis, opposite folds — the sign must follow.

    Reconstructing from ``SYMMETRY_PHASES`` (the even X/Y table) instead of the
    plane's declared phase leaves every even run correct and silently mirrors an
    odd run with the wrong sign, which no shape or round-trip check would notice.
    """
    quadrant = _distinct((5, 6, 4))
    reconstructions = {}
    for phase in (+1, -1):
        grid = _make_grid(8, 6, 4, symmetry=(Mirror('X', phase),))
        reconstructions[phase] = Fields(grid=grid)._reconstruct_full_domain(quadrant, 'Ez')
    centre = 4
    # Ez is even about x under an even mirror and odd about it under an odd one.
    np.testing.assert_array_equal(reconstructions[+1][centre - 2], quadrant[2])
    np.testing.assert_array_equal(reconstructions[-1][centre - 2], -quadrant[2])
    # The boundary cell already sits on the negative side and takes no phase either way.
    np.testing.assert_array_equal(reconstructions[+1][centre - 1], quadrant[0])
    np.testing.assert_array_equal(reconstructions[-1][centre - 1], quadrant[0])


def test_to_meep_array_shape_without_symmetry():
    grid = _make_grid(6, 5, 4)
    fields = Fields(grid=grid)
    fields.Dz[:] = _distinct(grid.shape)
    centered = fields.to_cell_center('Dz')
    out = fields.to_meep_array('Dz')

    assert out.shape == (grid.nx + 1, grid.ny + 1, grid.nz + 1)
    np.testing.assert_array_equal(out[1:, 1:, 1:], centered)
    np.testing.assert_array_equal(out[1:, 1:, 0], centered[:, :, -1])  # Periodic boundary cells.
    np.testing.assert_array_equal(out[0, 1:, 1:], centered[-1, :, :])
    np.testing.assert_array_equal(out[1:, 0, 1:], centered[:, -1, :])
    assert out[0, 0, 0] == centered[-1, -1, -1]


def test_to_meep_array_shape_with_symmetry():
    grid_xy = _make_grid(8, 6, 4, symmetry=('X', 'Y'))
    fields_xy = Fields(grid=grid_xy)
    fields_xy.Dz[:] = _distinct(grid_xy.shape)
    out_xy = fields_xy.to_meep_array('Dz')
    assert out_xy.shape == (grid_xy.nx_full, grid_xy.ny_full, grid_xy.nz + 1)

    grid_x = _make_grid(8, 6, 4, symmetry=('X',))
    fields_x = Fields(grid=grid_x)
    fields_x.Dz[:] = _distinct(grid_x.shape)
    out_x = fields_x.to_meep_array('Dz')
    assert out_x.shape == (grid_x.nx_full, grid_x.ny + 1, grid_x.nz + 1)

    # A folded Z is the same statement on the axis that used to be special-cased:
    # it comes back at its full count with no periodic duplicate prepended.
    grid_z = _make_grid(8, 6, 4, symmetry=(Mirror('Z', -1),))
    fields_z = Fields(grid=grid_z)
    fields_z.Dz[:] = _distinct(grid_z.shape)
    assert fields_z.to_meep_array('Dz').shape == (grid_z.nx + 1, grid_z.ny + 1, grid_z.nz_full)

    grid_all = _make_grid(8, 6, 4, symmetry=('X', 'Y', 'Z'))
    fields_all = Fields(grid=grid_all)
    fields_all.Dz[:] = _distinct(grid_all.shape)
    assert fields_all.to_meep_array('Dz').shape == grid_all.shape_full

    # An ODD folded count: MEEP's convention is measured per (boundary, parity).
    # A folded METALLIC axis comes back at its full count (the window shifts, the
    # count does not), while a folded PERIODIC odd axis gains one plane: the odd
    # window's bottom sits half a cell above the requested -L/2, and MEEP serves
    # the site at -L/2 - 0*dx as the wrap image of the TOP stored plane — measured
    # an exact duplicate (0.0 difference, field peak 2.2) where an even folded
    # periodic axis gains nothing.
    grid_odd = _make_grid(9, 6, 4, symmetry=('X',))  # Default boundaries: periodic.
    fields_odd = Fields(grid=grid_odd)
    fields_odd.Dz[:] = _distinct(grid_odd.shape)
    # 9 - 9//2 + 1 owned, plus the big_corner cell a folded PERIODIC axis stores
    # at either parity.
    assert grid_odd.nx == 7 and grid_odd.owned_cells(0) == 6
    out_odd = fields_odd.to_meep_array('Dz')
    assert out_odd.shape == (10, grid_odd.ny + 1, grid_odd.nz + 1)
    np.testing.assert_array_equal(out_odd[0], out_odd[-1])  # The wrap duplicate.

    grid_odd_pec = Grid(resolution=1.0, cell_size=(9.0, 6.0, 4.0), symmetry=('X',),
                        boundaries='metallic')
    fields_pec = Fields(grid=grid_odd_pec)
    fields_pec.Dz[:] = _distinct(grid_odd_pec.shape)
    assert fields_pec.to_meep_array('Dz').shape == (9, grid_odd_pec.ny + 1, grid_odd_pec.nz + 1)


def test_a_collapsed_axis_is_averaged_with_its_lattice_image_not_selected():
    """MEEP COLLAPSES a zero-extent direction; it does not pick a plane out of it.

    ``sim.get_array(component=c)`` slices ``fields::total_volume()`` = ``gv.interior()``
    (fields.cpp:717-724, vec.cpp:289-291), which has ZERO EXTENT on a one-cell axis.
    ``array_slice`` loops that slice on the CENTERED grid (array_slice.cpp:675), so its
    bounds straddle the requested plane — ``is = -1``, ``ie = +1`` doubled
    (loop_in_chunks.cpp:352-356) — the zero-extent branch of
    ``compute_boundary_weights`` gives both samples weight ``0.5``
    (loop_in_chunks.cpp:275-287), the chunkloop keeps exactly those weights
    (array_slice.cpp:353-368) and ``collapse_array`` sums them
    (array_slice.cpp:582-587).

    The lower sample is one lattice vector down, so it is ``conj(bloch_phase)`` times
    the stored one and the collapse costs ``(1 + conj(bloch_phase)) / 2``. At k = 0
    that is exactly 1 — which is why taking one plane was right on every non-Bloch run
    — and under a Bloch phase it is short by ``tan(pi*k*dx)``, FIRST ORDER in dx, the
    error that halves with resolution instead of converging away. The two resolutions
    below pin that scaling with no MEEP in the loop; the end-to-end number against CPU
    MEEP is ``test_from_meep``'s ``unit_axes_three_d_bloch`` parity case.
    """
    for resolution in (10.0, 20.0):
        k = 0.616
        # One cell on x and y — what `from_meep` builds for MEEP's `cell_size.x = 0`
        # at dimensions=3, and what `vol3d` builds too ((xsize == 0) ? 1 : ...,
        # vec.cpp:923-931).
        cell = 1.0 / resolution
        grid = Grid(resolution=resolution, cell_size=(cell, cell, 2.0), dimensions=3,
                    k_point=(k, 0.0, 0.0))
        assert grid.shape_full[0] == 1 and grid.bloch_phase(1) is None
        fields = Fields(grid=grid, force_complex_fields=True)
        fields.Dz[:] = _distinct(grid.shape).astype(np.complex64) * (1.0 + 0.5j)

        collapsed = fields.to_meep_array('Dz')
        centred = fields.to_cell_center('Dz')
        assert collapsed.shape == (grid.nz + 1,), "the one-cell x and y axes must be gone"

        phase = grid.bloch_phase(0)
        one_plane = np.asarray(centred)[0, 0, :]  # The pre-fix rule: keep the stored plane.
        expected = (one_plane * np.complex64(0.5 * (1.0 + np.conjugate(phase)))).astype(
            collapsed.dtype)
        np.testing.assert_allclose(collapsed[1:], expected, rtol=2e-6, atol=0)
        assert collapsed.dtype == np.complex64, "the collapse promoted the stored dtype"

        # The size of what taking one plane would have cost, and its dx scaling.
        shortfall = float(np.linalg.norm(one_plane - expected) / np.linalg.norm(expected))
        np.testing.assert_allclose(shortfall, np.tan(np.pi * k / resolution), rtol=1e-5)


def test_a_collapsed_axis_without_a_bloch_phase_is_untouched():
    """k = 0 must stay bit-identical: the collapse factor is exactly 1, not ``1 + 0j``."""
    grid = Grid(resolution=10.0, cell_size=(0.1, 0.1, 2.0), dimensions=3)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.Dz[:] = _distinct(grid.shape)
    centred = np.asarray(fields.to_cell_center('Dz'))
    collapsed = fields.to_meep_array('Dz')
    assert collapsed.dtype == np.float32, "a k = 0 collapse promoted a real array"
    np.testing.assert_array_equal(collapsed[1:], centred[0, 0, :])


def test_to_meep_array_preserves_dtype_for_complex_fields():
    grid = _make_grid(6, 6, 4)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.Bz[:] = (1.0 + 2.0j)
    out = fields.to_meep_array('Bz')
    assert out.dtype == np.complex64
    np.testing.assert_allclose(out, 1.0 + 2.0j, rtol=1e-6)


def test_repr_memory_counts_material_arrays_at_four_bytes():
    grid = _make_grid(8, 8, 8)
    fields = Fields(grid=grid, force_complex_fields=True)
    cells = grid.total_cells

    expected_mb = (cells * 6 * 8 + cells * 2 * 4) / 1e6  # 6 complex64 D/B + 2 float32 material.
    assert f"memory={expected_mb:.1f}MB" in repr(fields)
    assert "pml_storage" not in repr(fields)

    fields.enable_pml_storage()
    expected_pml_mb = (cells * 24 * 8 + cells * 2 * 4) / 1e6  # 24 complex64 field arrays.
    assert f"memory={expected_pml_mb:.1f}MB" in repr(fields)
    assert "pml_storage=True" in repr(fields)

    real_fields = Fields(grid=grid)
    expected_real_mb = (cells * 6 * 4 + cells * 2 * 4) / 1e6
    assert f"memory={expected_real_mb:.1f}MB" in repr(real_fields)


def test_to_cell_center_phases_the_wrapped_plane_under_bloch_boundaries():
    """The wrapped neighbour of an interpolated axis is one lattice vector away.

    ``to_cell_center`` averages ``0.5*(f[i] + f[i+1])`` along every axis whose Yee
    shift is 0, and at ``i = n-1`` that neighbour is stored cell 0 — one lattice
    vector up, which under Bloch boundaries carries ``grid.bloch_phase(axis)``.
    MEEP applies the same factor filling an array from a shifted chunk
    (loop_in_chunks.cpp ``ph *= pow(eikna[d], ishift)``).

    This lives in ``Fields`` rather than in the driver so that every reader gets
    it — ``to_meep_array`` and any direct caller included. While it lived only in
    ``FdtdDriver.get_field``, a ``Fields`` object read directly under a nonzero k
    returned a wrong far plane on each interpolated axis, silently.
    """
    k_z = 0.3
    grid = Grid(resolution=1.0, cell_size=(3.0, 3.0, 4.0), k_point=(0.0, 0.0, k_z))
    phase = grid.bloch_phase(2)
    assert phase is not None
    fields = Fields(grid=grid, force_complex_fields=True)
    # Bz has iyee_shift (1,1,0): z is the only interpolated axis, so the wrap phase
    # appears exactly once and can be read off directly.
    line = np.array([1.0, 2.0, 5.0, 11.0], dtype=np.complex64)
    fields.Bz[:] = np.broadcast_to(line.reshape(1, 1, grid.nz), grid.shape)
    centered = fields.to_cell_center('Bz')

    expected = 0.5 * (line + phase * np.roll(line, -1))
    expected[:-1] = 0.5 * (line[:-1] + line[1:])  # Interior planes never wrapped.
    np.testing.assert_allclose(centered[1, 2, :], expected, rtol=0, atol=1e-6)
    assert centered[1, 2, -1] == pytest.approx(0.5 * (line[-1] + phase * line[0]), rel=1e-6)

    # Controls: the two ways to get this wrong are both measurably different here,
    # because k = 0.3 is not at the zone edge where the phase is its own conjugate.
    for label, wrong in (("unphased", 0.5 * (line[-1] + line[0])),
                         ("conjugated", 0.5 * (line[-1] + phase.conjugate() * line[0]))):
        assert abs(centered[1, 2, -1] - wrong) > 0.1 * abs(centered[1, 2, -1]), (
            f"The {label} far plane is within 10% of the measured value; this cannot see it."
        )

    # k = 0 keeps the plain periodic average, bit-for-bit: the factor is None there
    # and the multiply is skipped rather than done against 1 + 0j.
    plain = Fields(grid=Grid(resolution=1.0, cell_size=(3.0, 3.0, 4.0)), force_complex_fields=True)
    plain.Bz[:] = np.broadcast_to(line.reshape(1, 1, 4), plain.Bz.shape)
    np.testing.assert_array_equal(
        plain.to_cell_center('Bz')[1, 2, :], 0.5 * (line + np.roll(line, -1))
    )


def test_to_cell_center_refuses_real_storage_under_a_bloch_phase():
    """Real fields cannot carry the boundary phase; dropping its imaginary part is not an option."""
    grid = Grid(resolution=1.0, cell_size=(3.0, 3.0, 4.0), k_point=(0.0, 0.0, 0.3))
    real_fields = Fields(grid=grid, force_complex_fields=False)
    with pytest.raises(ValueError, match="need complex fields"):
        real_fields.to_cell_center('Bz')
    # The positive control: complex storage on the same grid is served.
    Fields(grid=grid, force_complex_fields=True).to_cell_center('Bz')


# ---------------------------------------------------------------------------
# Dispersion and conductivity storage
# ---------------------------------------------------------------------------


def _dispersive_fields(sigma=0.6, cells=(4, 4, 6)):
    grid = _make_grid(*cells)
    fields = Fields(grid=grid, force_complex_fields=True)
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), sigma, grid, np.complex64)
    if state.driven():
        fields.enable_field_storage()
    fields.polarizations.append(state)
    return fields, state


def test_enable_field_storage_allocates_E_and_deliberately_not_H():
    """A susceptibility needs a stored E; it must NOT get a stored H.

    ``update_H`` writes nothing without PML — H is numerically identical to B — so a
    stored H would sit at zero for the whole run while ``get_H`` returned it. That
    failure is total, silent, and produces no warning of any kind.
    """
    fields, _ = _dispersive_fields()
    assert fields.stores_E and not fields._pml_active
    for name in ('Ex', 'Ey', 'Ez'):
        assert getattr(fields, name) is not None
    for name in ('Hx', 'Hy', 'Hz', 'f_w_Ex', 'fu_Dx'):
        assert getattr(fields, name) is None
    fields.Bx[1, 1, 1] = 3.0 + 1.0j
    assert fields.get_H('Hx') is fields.Bx, "H must still be served from B without PML"


def test_enable_pml_storage_keeps_an_E_a_susceptibility_already_allocated():
    # Re-allocating E here would discard a live polarization's drive history — and the
    # run would carry on producing smooth, plausible, wrong fields.
    fields, _ = _dispersive_fields()
    fields.Ez[2, 2, 2] = 7.0 + 0.0j
    stored = fields.Ez
    fields.enable_pml_storage()
    assert fields.Ez is stored and fields.Ez[2, 2, 2] == 7.0
    assert fields._pml_active and fields.stores_E
    assert fields.get_H('Hx') is fields.Hx


def test_get_E_returns_storage_once_a_susceptibility_is_present():
    fields, _ = _dispersive_fields()
    fields.Dz[1, 1, 1] = 4.0 + 0.0j
    fields.Ez[1, 1, 1] = 9.0 + 0.0j
    assert fields.get_E('Ez') is fields.Ez
    assert complex(fields.get_E('Ez')[1, 1, 1]) == 9.0 + 0.0j, (
        "with P live, E must come from storage: D*inv_eps is one polarization out of date"
    )
    plain = Fields(grid=_make_grid(4, 4, 6), force_complex_fields=True)
    plain.Dz[1, 1, 1] = 4.0 + 0.0j
    assert complex(plain.get_E('Ez')[1, 1, 1]) == 4.0 + 0.0j


def test_displacement_minus_polarization_aliases_D_when_nothing_drives_the_component():
    """MEEP feeds ``f[dc]`` straight through when f_minus_p was never allocated.

    Aliasing rather than forming ``D - 0.0`` is what makes a zero-strength
    susceptibility byte-identical: not doing the subtraction is exact by
    construction, where doing it is only exact for finite values.
    """
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=True)
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), {"Ez": 1.0}, grid, np.complex64)
    fields.enable_field_storage()
    fields.polarizations.append(state)
    assert fields.displacement_minus_polarization('Ex') is fields.Dx
    fields.Dz[:] = 5.0
    state.P['Ez'][:] = 2.0
    result = fields.displacement_minus_polarization('Ez')
    assert result is not fields.Dz
    np.testing.assert_array_equal(result, np.full(grid.shape, 3.0, dtype=np.complex64))
    # Every driven term is subtracted, not just the first.
    second = PolarizationState(Susceptibility(1.4, 0.1, LORENTZIAN), 1.0, grid, np.complex64)
    second.P['Ez'][:] = 1.0
    fields.polarizations.append(second)
    np.testing.assert_array_equal(
        fields.displacement_minus_polarization('Ez'), np.full(grid.shape, 2.0, dtype=np.complex64)
    )


def test_drive_field_is_the_constitutive_product_not_the_absorbed_E():
    """W = (D - sum P)*inv_eps, which is ``f_w`` under PML and the stored E without it.

    They agree EXACTLY outside an absorber, so driving the polarization from the
    stored E passes every no-PML test and is wrong only inside the layer — smoothly,
    plausibly, and in the region a transmission spectrum is normalised against. The
    distinction lives in one accessor so it cannot be re-derived differently.
    """
    fields, _ = _dispersive_fields()
    fields.Ez[:] = 1.0
    assert fields.drive_field('Ez') is fields.Ez
    fields.enable_pml_storage()
    fields.f_w_Ez[:] = 2.0
    assert fields.drive_field('Ez') is fields.f_w_Ez
    assert complex(fields.drive_field('Ez')[0, 0, 0]) == 2.0 + 0.0j


def test_reset_clears_stored_E_and_every_polarization_history():
    fields, state = _dispersive_fields()
    fields.Dz[:] = 3.0
    fields.Ez[:] = 3.0
    state.P['Ez'][:] = 1.0
    state.P_prev['Ez'][:] = 2.0
    fields.reset()
    assert not np.any(fields.Dz) and not np.any(fields.Ez)
    assert not np.any(state.P['Ez']) and not np.any(state.P_prev['Ez'])


def test_conductivity_coefficients_follow_meeps_definitions():
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=True)
    assert not fields.has_conductivity
    sigma_d = np.full(grid.shape, 0.4, dtype=np.float32)
    fields.set_d_conductivity(sigma_d)
    half_dt = grid.dt / 2.0
    assert fields.has_conductivity
    assert fields.conductive_components == ("Dx", "Dy", "Dz")
    for component in ("Dx", "Dy", "Dz"):
        condfac, condinv = fields.condfac_for(component), fields.condinv_for(component)
        np.testing.assert_allclose(condfac, 1.0 - 0.4 * half_dt, rtol=1e-6)
        np.testing.assert_allclose(condinv, 1.0 / (1.0 + 0.4 * half_dt), rtol=1e-6)
        assert condfac.dtype == np.float32 and condinv.dtype == np.float32
    # One volume installed under three keys: uniform sigma is registration-free, so
    # MEEP's three per-component arrays coincide and there is nothing to gain by
    # allocating them separately.
    assert fields.conductivity_for("Dx") is fields.conductivity_for("Dz")
    # The B side is independent and untouched by the D setter (structure.cpp:377-379).
    assert not fields.has_magnetic_conductivity
    fields.set_b_conductivity(np.full(grid.shape, 0.9, dtype=np.float32))
    assert fields.has_magnetic_conductivity and fields.has_conductivity
    np.testing.assert_allclose(fields.condinv_for("By"), 1.0 / (1.0 + 0.9 * half_dt), rtol=1e-6)
    fields.set_d_conductivity(None)
    assert not fields.has_conductivity and fields.condinv_for("Dx") is None
    assert fields.has_magnetic_conductivity  # Clearing one side leaves the other alone.
    fields.set_b_conductivity(None)
    assert fields.conductive_components == ()
    with pytest.raises(ValueError, match="does not match"):
        fields.set_d_conductivity(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="not\\s+D-side components"):
        fields.set_d_conductivity({"Bx": np.zeros(grid.shape, dtype=np.float32)})


@pytest.mark.parametrize("conductivity_first", [False, True])
def test_conductivity_plus_pml_allocates_and_resets_the_three_f_cond_histories(
    conductivity_first,
):
    """MEEP allocates one f_cond per D component, from either setup order."""
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=True)
    if conductivity_first:
        fields.set_d_conductivity(np.full(grid.shape, 0.4, dtype=np.float32))
        fields.enable_pml_storage()
    else:
        fields.enable_pml_storage()
        fields.set_d_conductivity(np.full(grid.shape, 0.4, dtype=np.float32))

    histories = (fields.f_cond_Dx, fields.f_cond_Dy, fields.f_cond_Dz)
    assert all(array is not None for array in histories)
    assert all(array.dtype == np.complex64 for array in histories)
    assert fields.field_bytes_per_cell() == 27 * 8
    fields.f_cond_Dx.fill(2.0 + 3.0j)
    fields.reset()
    assert np.max(np.abs(fields.f_cond_Dx)) == 0.0

    fields.set_d_conductivity(None)
    assert fields.f_cond_Dx is fields.f_cond_Dy is fields.f_cond_Dz is None
    assert fields.field_bytes_per_cell() == 24 * 8


def test_memory_accounting_covers_the_susceptibilities():
    """A six-term metal fit is larger than the whole field storage; the budget must say so."""
    grid = _make_grid(4, 4, 6)
    plain = Fields(grid=grid, force_complex_fields=True)
    assert plain.bytes_per_cell() == 6 * 8 + 2 * 4  # 6 complex64 D/B + eps + inv_eps.
    plain.enable_pml_storage()
    assert plain.bytes_per_cell() == 24 * 8 + 2 * 4  # 24 arrays with PML.
    dispersive, _ = _dispersive_fields()
    # 6 D/B + 3 stored E, plus P + P_prev on three components and one shared scratch.
    assert dispersive.bytes_per_cell() == 9 * 8 + 2 * 4 + 7 * 8
    assert "susceptibilities=1" in repr(dispersive)


# ---------------------------------------------------------------------------
# Real-valued (float32) field mode — MEEP's default
# ---------------------------------------------------------------------------


def _all_field_arrays(fields: Fields):  # Every dtype-carrying array the container owns, by name.
    """Enumerated here rather than read off ``Fields``, so the memory model has an
    independent witness: a new array that the model forgot to count, or a counted
    array that is never allocated, shows up as a disagreement instead of as two
    copies of the same mistake."""
    named = {}
    for name in ('Dx', 'Dy', 'Dz', 'Bx', 'By', 'Bz',
                 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz',
                 'fu_Bx', 'fu_By', 'fu_Bz', 'fu_Dx', 'fu_Dy', 'fu_Dz',
                 'f_cond_Dx', 'f_cond_Dy', 'f_cond_Dz',
                 'f_w_Ex', 'f_w_Ey', 'f_w_Ez', 'f_w_Hx', 'f_w_Hy', 'f_w_Hz'):
        array = getattr(fields, name)
        if array is not None:
            named[name] = array
    if fields._fmp_scratch is not None:
        named['_fmp_scratch'] = fields._fmp_scratch
    for index, state in enumerate(fields.polarizations):
        for component, array in state.P.items():
            named[f'P{index}_{component}'] = array
        for component, array in state.P_prev.items():
            named[f'Pprev{index}_{component}'] = array
        if state._scratch is not None:
            named[f'scratch{index}'] = state._scratch
    return named


@pytest.mark.parametrize("cells", [(4, 4, 6), (5, 5, 7)])  # Even and odd counts alike.
def test_real_mode_allocates_float32_through_every_storage_stage(cells):
    """D/B, the stored E, the PML H and twelve auxiliaries, and P all follow the flag.

    Any one of them left complex64 would still run — NumPy promotes silently on the
    way in — and would cost exactly the memory the mode exists to save, while every
    number stayed correct. Only the dtypes say so.
    """
    grid = _make_grid(*cells)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.polarizations.append(
        PolarizationState(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6, grid, fields._field_dtype())
    )
    fields.enable_field_storage()
    fields.enable_pml_storage()
    fields.displacement_minus_polarization('Ez')  # Materialise the D - sum P scratch.

    arrays = _all_field_arrays(fields)
    # 24 field arrays + the D - sum P scratch, plus P/P_prev on three driven
    # components and the susceptibility's own shared scratch.
    assert len(arrays) == 25 + 7, "the enumeration missed an allocated array"
    wrong = {name: array.dtype for name, array in arrays.items() if array.dtype != np.float32}
    assert not wrong, f"real mode allocated non-float32 storage: {wrong}"
    # The material arrays are float32 in BOTH modes; they are not part of the halving.
    assert fields.eps.dtype == np.float32 and fields.inv_eps.dtype == np.float32
    assert fields._field_dtype() == np.float32


def test_field_storage_is_exactly_halved_by_real_mode():
    """The efficiency claim, checked against the arrays rather than against the model.

    ``field_bytes_per_cell`` is asserted twice over: against the summed ``nbytes`` of
    the arrays actually allocated (so the model cannot drift from reality) and against
    the complex run's own number (so the halving is exact rather than approximate).
    The TOTAL is deliberately NOT half — eps and inv_eps are float32 either way — and
    that is pinned too, because promising 2x and delivering 1.75x is the kind of
    over-claim that sizes a GPU job wrong.
    """
    grid = _make_grid(6, 6, 8)
    cells = grid.total_cells
    measurements = {}
    for complex_fields in (False, True):
        fields = Fields(grid=grid, force_complex_fields=complex_fields)
        fields.polarizations.append(
            PolarizationState(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6, grid,
                              fields._field_dtype())
        )
        fields.enable_pml_storage()
        fields.displacement_minus_polarization('Ez')
        actual = sum(array.nbytes for array in _all_field_arrays(fields).values())
        assert fields.field_bytes_per_cell() * cells == actual, (
            "field_bytes_per_cell disagrees with the bytes actually allocated"
        )
        measurements[complex_fields] = (fields.field_bytes_per_cell(), fields.bytes_per_cell())

    real_field_bytes, real_total = measurements[False]
    complex_field_bytes, complex_total = measurements[True]
    assert 2 * real_field_bytes == complex_field_bytes, (
        f"real-mode field storage {real_field_bytes} B/cell is not half of {complex_field_bytes}"
    )
    assert real_total == real_field_bytes + 2 * 4  # eps + inv_eps, float32 in both modes.
    assert 2 * real_total > complex_total, "the total cannot be exactly halved; do not claim it is"


def test_repr_reports_the_real_dtype_and_its_smaller_footprint():
    grid = _make_grid(8, 8, 8)
    real_fields = Fields(grid=grid, force_complex_fields=False)
    expected_mb = (grid.total_cells * 6 * 4 + grid.total_cells * 2 * 4) / 1e6
    assert "dtype=float32" in repr(real_fields)
    assert f"memory={expected_mb:.1f}MB" in repr(real_fields)


def test_real_mode_conductivity_and_material_arrays_stay_float32():
    """A float32 coefficient volume is what makes real mode the exact real part.

    A complex64 ``condfac`` would multiply the field's real part by its imaginary
    part; over real storage NumPy would then refuse the in-place write, and over
    complex storage it would quietly mix the two planes. Both modes take the same
    real coefficients, so neither can happen.
    """
    grid = _make_grid(4, 4, 6)
    for complex_fields in (False, True):
        fields = Fields(grid=grid, force_complex_fields=complex_fields)
        fields.set_d_conductivity(np.full(grid.shape, 0.4, dtype=np.float32))
        assert fields.conductivity_for("Dx").dtype == np.float32
        assert fields.condfac_for("Dx").dtype == np.float32
        assert fields.condinv_for("Dx").dtype == np.float32
        assert fields.eps.dtype == np.float32 and fields.inv_eps.dtype == np.float32


def test_real_mode_readbacks_stay_real_and_keep_their_values():
    """to_cell_center / to_meep_array / get_E must not promote on the way out."""
    grid = _make_grid(6, 6, 4)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.Bz[:] = 2.5
    fields.Dz[:] = 4.0
    assert fields.to_cell_center('Bz').dtype == np.float32
    assert fields.to_meep_array('Bz').dtype == np.float32
    np.testing.assert_array_equal(fields.to_cell_center('Bz'), np.full(grid.shape, 2.5, np.float32))
    assert fields.get_E('Ez').dtype == np.float32
    np.testing.assert_array_equal(fields.get_E('Ez'), np.full(grid.shape, 4.0, np.float32))

    folded = _make_grid(8, 6, 4, symmetry=('X',))
    folded_fields = Fields(grid=folded, force_complex_fields=False)
    folded_fields.Dz[:] = _distinct(folded.shape)
    assert folded_fields.to_meep_array('Dz').dtype == np.float32


# --------------------------------------------------------------------------------------
# Instantaneous chi2 / chi3 storage (MEEP structure.cpp set_chi2 / set_chi3).
# --------------------------------------------------------------------------------------


def test_an_identically_zero_chi2_chi3_pair_installs_no_nonlinearity_at_all():
    """MEEP deletes the trivial pair (structure.cpp:822-826) and so does this.

    Storing zeros instead would be a defensible-looking choice and would cost the
    engine its byte-identical linear path: the Pade factor evaluates to exactly 1.0,
    but multiplying by it reassociates the constitutive product in float32 and every
    recorded CPU-MEEP floor was measured without that multiply.
    """
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=False)
    zero = {name: 0.0 for name in ('Ex', 'Ey', 'Ez')}
    fields.set_nonlinear_volumes(zero, zero)
    assert not fields.has_nonlinearity
    assert fields.nonlinear_components == ()
    assert fields.chi2_for('Ez') == 0.0 and fields.chi3_for('Ez') == 0.0

    # An all-zero VOLUME is trivial too, not merely a scalar zero: MEEP's `trivial`
    # flag is computed cell by cell over the array it just filled.
    fields.set_nonlinear_volumes({'Ez': np.zeros(grid.shape, np.float32)},
                                 {'Ez': np.zeros(grid.shape, np.float32)})
    assert not fields.has_nonlinearity

    # One nonzero cell is enough to make it live, which is the boundary of that rule.
    volume = np.zeros(grid.shape, np.float32)
    volume[2, 2, 3] = 1e-4
    fields.set_nonlinear_volumes({'Ez': 0.0}, {'Ez': volume})
    assert fields.nonlinear_components == ('Ez',)


def test_chi2_and_chi3_are_installed_as_a_pair_per_component():
    """MEEP requires both present if either is (structure.cpp:815-826, 851-862).

    ``step_update_EDHB`` branches on chi3 and then dereferences chi2[i], so a lone
    chi2 would be a null read in MEEP and a missing term here. Setting either one
    must leave the partner as an explicit zero on the SAME components.
    """
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.set_nonlinear_volumes({'Ez': 0.0}, {'Ez': 0.02})
    assert fields.nonlinear_components == ('Ez',)
    assert fields.chi2_for('Ez') == 0.0 and fields.chi3_for('Ez') == 0.02
    assert set(fields._chi2_components) == set(fields._chi3_components)

    fields.set_nonlinear_volumes({'Ex': 0.05}, {})
    assert fields.nonlinear_components == ('Ex',)
    assert fields.chi3_for('Ex') == 0.0 and fields.chi2_for('Ex') == 0.05
    assert set(fields._chi2_components) == set(fields._chi3_components)
    # A component that was never named is linear and reports zero for both.
    assert fields.chi2_for('Ey') == 0.0 and fields.chi3_for('Ey') == 0.0
    assert not fields.is_nonlinear('Ey')


def test_set_nonlinear_volumes_refuses_what_it_cannot_store():
    """Loud on every malformed input; a silent default here is a silently wrong material."""
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=False)
    zero = {name: 0.0 for name in ('Ex', 'Ey', 'Ez')}

    with pytest.raises(ValueError, match="electric components"):
        fields.set_nonlinear_volumes({'Hx': 1.0}, zero)
    with pytest.raises(ValueError, match="electric components"):
        fields.set_nonlinear_volumes(zero, {'Ez2': 1.0})
    with pytest.raises(ValueError, match="does not match the grid shape"):
        fields.set_nonlinear_volumes(zero, {'Ez': np.ones((3, 3, 3), np.float32)})
    with pytest.raises(ValueError, match="finite"):
        fields.set_nonlinear_volumes(zero, {'Ez': float('nan')})
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros(grid.shape, np.float32)
        bad[0, 0, 0] = np.inf
        fields.set_nonlinear_volumes({'Ez': bad}, zero)
    assert not fields.has_nonlinearity, "no refused call may leave a partial material behind"

    # A NEGATIVE coefficient is an ordinary material and must NOT be refused: chi3 < 0
    # is self-defocusing and chi2 < 0 a reversed Pockels coefficient. This is where
    # chi2/chi3 part company with sigma and the conductivity, which are gain when
    # negative and are rejected.
    fields.set_nonlinear_volumes({'Ez': -0.02}, {'Ez': -0.05})
    assert fields.nonlinear_components == ('Ez',)


def test_the_three_displacement_minus_polarization_volumes_are_live_at_once():
    """The nonlinearity's transverse average needs all three; the shared scratch cannot serve it.

    ``displacement_minus_polarization`` reuses ONE buffer across components on the
    invariant that each is consumed before the next is formed. MEEP's ``Dsqr`` reads
    ``dmp[dc]``, ``dmp[dc_1]`` and ``dmp[dc_2]`` together, so this checks the three
    are distinct objects with distinct, correct contents — a shared buffer would make
    all three the same array and give every component the last one's D - P.
    """
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=False)
    susceptibility = Susceptibility(1.1, 0.05, LORENTZIAN)
    fields.polarizations.append(
        PolarizationState(susceptibility, 0.6, grid, fields._field_dtype())
    )
    fields.enable_field_storage()
    for index, name in enumerate(('Dx', 'Dy', 'Dz')):
        getattr(fields, name)[...] = index + 1.0
    state = fields.polarizations[0]
    for index, name in enumerate(('Ex', 'Ey', 'Ez')):
        state.P[name][...] = 0.1 * (index + 1)

    volumes = fields.displacement_minus_polarization_volumes()
    assert len({id(volumes[name]) for name in ('Ex', 'Ey', 'Ez')}) == 3
    for index, name in enumerate(('Ex', 'Ey', 'Ez')):
        np.testing.assert_allclose(np.asarray(volumes[name]),
                                   (index + 1.0) - 0.1 * (index + 1), rtol=1e-6)
    # Still correct on a second call, i.e. the buffers are reused rather than stale.
    again = fields.displacement_minus_polarization_volumes()
    for name in ('Ex', 'Ey', 'Ez'):
        np.testing.assert_array_equal(np.asarray(again[name]), np.asarray(volumes[name]))

    # With NO polarization the volumes alias the D arrays outright, as MEEP passes
    # f[dc] through when f_minus_p was never allocated: a nonlinear non-dispersive
    # run must allocate nothing here.
    plain = Fields(grid=grid, force_complex_fields=False)
    aliased = plain.displacement_minus_polarization_volumes()
    assert aliased['Ez'] is plain.Dz


def test_chi2_and_chi3_volumes_are_counted_in_the_memory_budget():
    """A dispatch predicate that under-reports storage OOMs after the setup is paid for."""
    grid = _make_grid(4, 4, 6)
    fields = Fields(grid=grid, force_complex_fields=False)
    baseline = fields.bytes_per_cell()

    fields.set_nonlinear_volumes({name: 0.01 for name in ('Ex', 'Ey', 'Ez')},
                                 {name: 0.02 for name in ('Ex', 'Ey', 'Ez')})
    assert fields.bytes_per_cell() == baseline, "a uniform chi2/chi3 allocates no volume"

    volume = np.full(grid.shape, 0.02, np.float32)
    fields.set_nonlinear_volumes({name: 0.01 for name in ('Ex', 'Ey', 'Ez')},
                                 {name: volume for name in ('Ex', 'Ey', 'Ez')})
    # One array, aliased into all three components, so it is counted once.
    assert fields.bytes_per_cell() == baseline + 4
    assert "nonlinear=('Ex', 'Ey', 'Ez')" in repr(fields)
    assert "nonlinear" not in repr(Fields(grid=grid, force_complex_fields=False))
