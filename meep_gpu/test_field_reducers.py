"""Value-level tests for the field-function reducers (`dft.py` + `driver.py`).

These pin MEEP's ``fields::integrate`` / ``fields::max_abs`` (src/integrate.cpp) and the
five ``*_in_box`` measures built on them (src/energy_and_flux.cpp). Everything here runs
on the NumPy backend without MEEP installed: the oracles are the CONSTANTS a MEEP run
produced, recorded in each test with the cell that produced them, so the file stays a
regression net on a machine that has neither MEEP nor CUDA. The end-to-end contrast
against a live MEEP is the assertions stage of ``parity/meep_gpu/survey_meep_tests.py``,
which drives MEEP's own ``test_field_functions``, ``test_simulation`` and
``test_wvg_src`` through this code.

What each block exists to catch, in the order a wrong implementation usually fails:

* **The loop lattice.** ``cgrid`` is Centered only when the requested components do not
  all share a Yee shift (integrate.cpp:135-143). Getting that wrong is not a small error:
  ``electric_energy_in_box`` rebuilt from the centred array is 64%-99% wrong. The
  constant-integrand block pins the three per-parity measures a 10x10 cell produces, which
  differ in the third digit and are therefore invisible to any "is it about right" check.
* **The flat axis.** A zero-extent direction contributes TWO sample rows whose weights sum
  to 1 — an interpolation onto the requested plane, not a collapse to the nearest row.
  ``max_abs`` takes its maximum over both, which is what MEEP's own 0.27593732 constant
  depends on.
* **The invariant axis.** A direction the grid does not have is not a flat axis: MEEP has
  no loop for it, so it contributes ONE row at weight 1 and coordinate 0.
* **The material integrands.** ``Dielectric`` is not a field read. It is
  ``(4*ninveps)/tr`` over the four-point sum of ``chi1inv`` (integrate.cpp:85-96), and on
  a vacuum cell every wrong version of it also returns 1.
* **The magnetic half-step.** ``synchronize_magnetic_fields`` must be exactly undone by
  ``restore_magnetic_fields``, PML auxiliaries included, and must be refcounted so a
  nested call is a no-op. Leaving it out is 2.5e-02 on test_wvg_src's near-field flux.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from meep_gpu import dft
from meep_gpu.driver import FdtdDriver


def _walls(dimensions: int):
    """PEC on the run's REAL axes only.

    An invariant axis has no outer face — MEEP never loops it — and this driver refuses a
    wall there, so the boundary table is spelled per axis rather than as one word.
    """
    live = {1: ("z",), 2: ("x", "y"), 3: ("x", "y", "z")}[dimensions]
    return {axis: "metallic" for axis in live}


# --- the loop lattice --------------------------------------------------------------

def _constant_measure(driver, components):
    """``integrate(components, 1)`` over the whole declared cell: the measure MEEP assigns."""
    return driver.integrate_field_function(
        components, lambda _location, *_values: 1.0,
        center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 0.0),
    ).real


@pytest.mark.parametrize("symmetry", [None, ("X", "Y")])
def test_constant_integrand_measures_the_cell_once_per_cgrid_parity(symmetry):
    """The three per-parity measures of a 10x10 cell at resolution 20, from MEEP.

    Measured with MEEP 1.33.0 on ``test_field_functions``'s own cell, integrating the
    constant 1 over the whole cell:

        cgrid = Centered  (parity 1, 1)  ->  99.75015625
        cgrid = Hx        (parity 1, 0)  ->  99.62531250
        cgrid = Ez        (parity 0, 0)  ->  99.50062500

    and they factor exactly as the per-axis weight sums 9.9875 (parity 1) and 9.975
    (parity 0, whose 201-point ladder loses one end to the owned-corner clip):
    9.9875**2, 9.9875*9.975 and 9.975**2. Three numbers within 0.25% of each other and of
    the naive answer 100 — which is why a reducer that ignores ``cgrid`` altogether looks
    right until it is asked for an energy.

    Run with AND without the mirror symmetry, because the measured ratio between the
    Ez and Centered readings was identical either way: the parity rule is the cgrid's
    Yee shift, not anything about the fold.
    """
    driver = FdtdDriver(cell_size=(10, 10, 0), resolution=20, dimensions=2,
                        boundaries=_walls(2), symmetry=symmetry or ())
    try:
        # [Ex, Hz] do not share a Yee shift, so this is the Centered branch.
        assert _constant_measure(driver, ("Ex", "Hz")) == pytest.approx(99.75015625, rel=1e-12)
        assert _constant_measure(driver, ("Hx",)) == pytest.approx(99.6253125, rel=1e-12)
        assert _constant_measure(driver, ("Ez",)) == pytest.approx(99.500625, rel=1e-12)
        # The material lattice is the centred one (iyee_shift(Dielectric) is all ones).
        assert _constant_measure(driver, ("Dielectric",)) == pytest.approx(99.75015625, rel=1e-12)
    finally:
        driver.close()


def test_cgrid_is_centered_only_when_the_shifts_differ():
    """integrate.cpp:133-143 — the same_grid test, and what ``Dielectric`` counts as."""
    assert dft.reducer_cgrid(("Ex", "Dx")) == "Ex"  # E with its own D: one lattice.
    assert dft.reducer_cgrid(("Ez", "Hx")) is None  # None stands for Centered.
    assert dft.reducer_cgrid(("Ex", "Hz", "Dielectric")) is None
    assert dft.reducer_cgrid(("Dielectric", "Permeability")) == "Dielectric"
    assert dft.reducer_component_shift("Dielectric") == (1, 1, 1)
    assert dft.reducer_component_shift("Hz") == (1, 1, 0)


# --- flat and invariant axes -------------------------------------------------------

def test_a_flat_axis_contributes_two_rows_and_an_invariant_axis_one():
    """A zero-extent DIRECTION interpolates over two rows; a missing one is not a direction.

    MEEP's ``compute_boundary_weights`` gives a zero-extent axis ``s0 = w0``, ``e0 = w1``
    over the two cells bracketing the coordinate (loop_in_chunks.cpp:275-287) and no
    factor of ``inva``; a direction the grid does not have is never looped at all. On a
    2-D 10x10 cell that makes a ``size=(1, 0, 0)`` request 22 x 2 x 1 points (the ladder
    brackets [-0.5, 0.5] outward onto cell centres, so it reaches 0.025 past each end),
    not 22 x 1 x 1 and not 22 x 2 x 2.
    """
    driver = FdtdDriver(cell_size=(10, 10, 0), resolution=20, dimensions=2,
                        boundaries=_walls(2))
    try:
        grid = driver.grid
        bounds = dft._reducer_volume_bounds(grid, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        axes = [dft._reducer_axis_plan(grid, axis, *bounds[axis], 1) for axis in range(3)]
        assert [int(axis.indices.size) for axis in axes] == [22, 2, 1]
        assert float(axes[1].weights.sum()) == pytest.approx(1.0, rel=1e-12)
        assert axes[2].invariant is True
        assert float(axes[2].weights[0]) == 1.0
        assert float(axes[2].coords[0]) == 0.0  # vec2py reports 0 for a missing direction.
        # The flat axis's two rows sit half a cell either side of the requested plane.
        assert float(axes[1].coords[1] - axes[1].coords[0]) == pytest.approx(grid.dx, rel=1e-12)
        # The measure over a 1 x 0 line is its length: no factor of dx for the flat axis.
        assert _line_measure(driver) == pytest.approx(1.0, rel=1e-12)
    finally:
        driver.close()


def _line_measure(driver):
    return driver.integrate_field_function(
        ("Ex", "Hz"), lambda _location, *_values: 1.0,
        center=(0.0, 0.0, 0.0), size=(1.0, 0.0, 0.0),
    ).real


def test_max_abs_sees_both_rows_of_a_flat_axis():
    """The maximum is over the SAME point set the sum runs over, unweighted.

    Collapsing the flat axis to one row would silently drop half the candidate points.
    The integrand here is +1 on one side of the plane and -3 on the other, so a collapse
    to the nearer row returns 1 and the correct answer is 3 — and the INTEGRAL is -1
    either way once the two half weights are applied, which is exactly how a collapsed
    max survives a sum-only check.
    """
    driver = FdtdDriver(cell_size=(10, 10, 0), resolution=20, dimensions=2,
                        boundaries=_walls(2))
    try:
        def integrand(location, *_values):
            return 1.0 if location[1] < 0.0 else -3.0

        assert driver.max_abs_field_function(
            ("Ex", "Hz"), integrand, center=(0.0, 0.0, 0.0), size=(1.0, 0.0, 0.0)
        ) == pytest.approx(3.0, rel=1e-12)
        assert driver.integrate_field_function(
            ("Ex", "Hz"), integrand, center=(0.0, 0.0, 0.0), size=(1.0, 0.0, 0.0)
        ).real == pytest.approx(-1.0, rel=1e-12)
    finally:
        driver.close()


def test_max_abs_is_taken_before_the_integration_weight():
    """integrate.cpp:123 — ``maxabs`` is updated from the raw integrand, then the weight.

    A partially covered end cell carries weight 0.125 here; taking the maximum after the
    multiply would report 0.125 * the peak. The peak is placed in that end cell on
    purpose, by making the integrand the x coordinate.
    """
    driver = FdtdDriver(cell_size=(10, 10, 0), resolution=20, dimensions=2,
                        boundaries=_walls(2))
    try:
        peak = driver.max_abs_field_function(
            ("Ex", "Hz"), lambda location, *_values: location[0],
            center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 0.0),
        )
        # The outermost centred sample MEEP OWNS on a 200-cell metallic axis. The ladder
        # itself reaches 5.025, one cell further out; `little_owned_corner` /
        # `big_owned_corner` clamp it back, and the peak lands on the survivor.
        assert peak == pytest.approx(4.975, rel=1e-12)
    finally:
        driver.close()


# --- the callback contract ----------------------------------------------------------

def test_the_callback_sees_the_location_then_one_complex_per_component():
    """python/meep.i:158-188 ``py_field_func_wrap``: loc first, then Python complexes."""
    driver = FdtdDriver(cell_size=(1, 1, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2))
    try:
        seen = []

        def integrand(location, ex, hz, eps):
            seen.append((location, ex, hz, eps))
            return 0.0

        driver.integrate_field_function(("Ex", "Hz", "Dielectric"), integrand,
                                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
        assert seen, "the reducer visited no points at all"
        location, ex, hz, eps = seen[0]
        assert len(location) == 3 and all(isinstance(value, float) for value in location)
        assert isinstance(ex, complex) and isinstance(hz, complex) and isinstance(eps, complex)
        assert eps == pytest.approx(1.0)  # Vacuum cell: the trace is 1.
    finally:
        driver.close()


# --- the material integrands --------------------------------------------------------

def test_dielectric_is_the_chi1inv_trace_and_not_one_component():
    """``fvals = (4*ninveps)/tr`` — integrate.cpp:85-96, not a field read.

    Built so the three components CANNOT be confused: each E component is given its own
    permittivity, so the correct answer is the harmonic-style trace
    ``3 / (1/2 + 1/4 + 1/8) = 3.428571...`` and reading any single component would give
    2, 4 or 8. A vacuum cell — which is all MEEP's own ``test_field_functions`` has —
    returns 1 for every one of those.
    """
    driver = FdtdDriver(cell_size=(1, 1, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2))
    try:
        xp = driver.xp
        shape = driver.grid.shape
        driver.fields.set_epsilon_volumes(
            {"Ex": xp.full(shape, 2.0, dtype=xp.float32),
             "Ey": xp.full(shape, 4.0, dtype=xp.float32),
             "Ez": xp.full(shape, 8.0, dtype=xp.float32)},
            {"Ex": xp.full(shape, 0.5, dtype=xp.float32),
             "Ey": xp.full(shape, 0.25, dtype=xp.float32),
             "Ez": xp.full(shape, 0.125, dtype=xp.float32)},
        )
        value = driver.integrate_field_function(
            ("Dielectric",), lambda _location, eps: eps,
            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
        )
        assert value.real == pytest.approx(3.0 / (0.5 + 0.25 + 0.125), rel=1e-6)
        # Permeability is MEEP's NULL-invmu branch (`tr += 4`): exactly 1, everywhere.
        assert driver.integrate_field_function(
            ("Permeability",), lambda _location, mu: mu,
            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
        ).real == pytest.approx(1.0, rel=1e-12)
    finally:
        driver.close()


def test_the_material_average_runs_on_each_component_s_own_yee_offsets():
    """``yee2cent_offsets(iecs[k])`` — the SAME offsets the field is averaged with.

    integrate.cpp:73-74 builds the material offsets outside the ``if (cgrid == Centered)``
    branch at :70, so ``chi1inv`` is ALWAYS four-point averaged, on the component's own
    lattice, whatever ``cgrid`` is. Reading raw per-component ``chi1inv`` instead is
    invisible on any uniform cell and was measured 1.7e-02 to 3.6e-01 wrong against MEEP
    on the eps=12 waveguide cell of ``test_wvg_src``.

    The discriminator is an ALTERNATING inverse permittivity, 1, 3, 1, 3, ... along one
    axis, on Ex alone (Yee shift (1, 0, 0) — averaged along y, not along x):

        along Y  -> every two-point average is exactly 2, so the trace is 2 + 1 + 1 and
                    the Dielectric integrand is a CONSTANT 3/4 everywhere;
        along X  -> no average on that axis, so the integrand alternates between
                    3/(1+2) = 1 and 3/(3+2) = 0.6.

    A reader that skips the average returns the alternating pattern in both cases, and a
    reader that averages the wrong axis swaps them. Both stay smooth, finite and near 1.
    """
    driver = FdtdDriver(cell_size=(2, 2, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2))
    try:
        xp = driver.xp
        shape = driver.grid.shape
        # An INTERIOR box: the outermost row of a shift-0 component sits on the PEC wall,
        # where the material read falls back to the nearest stored row (see
        # `dft._material_trace`) and the alternation this test relies on is broken.
        centre, extent = (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)
        measure = driver.integrate_field_function(
            ("Dielectric",), lambda _location, _eps: 1.0, center=centre, size=extent).real
        for axis, expected in ((1, 0.75), (0, None)):
            pattern = np.ones(shape, dtype=np.float32)
            index = [None, None, None]
            index[axis] = slice(1, None, 2)
            pattern[tuple(slice(None) if part is None else part for part in index)] = 3.0
            driver.fields.set_epsilon_volumes(
                {"Ex": xp.asarray(1.0 / pattern), "Ey": xp.ones(shape, dtype=xp.float32),
                 "Ez": xp.ones(shape, dtype=xp.float32)},
                {"Ex": xp.asarray(pattern), "Ey": xp.ones(shape, dtype=xp.float32),
                 "Ez": xp.ones(shape, dtype=xp.float32)},
            )
            total = driver.integrate_field_function(
                ("Dielectric",), lambda _location, eps: eps, center=centre, size=extent).real
            peak = driver.max_abs_field_function(
                ("Dielectric",), lambda _location, eps: eps, center=centre, size=extent)
            if expected is not None:
                # Averaged away: a constant, so the integral is the measure times it and
                # the maximum is it.
                assert total / measure == pytest.approx(expected, rel=1e-5)
                assert peak == pytest.approx(expected, rel=1e-5)
            else:
                # Not averaged on this axis: the pattern survives, so the maximum is the
                # HIGH value of the alternation and not the mean.
                assert peak == pytest.approx(1.0, rel=1e-5)
                assert total / measure == pytest.approx(0.8, rel=5e-2)
    finally:
        driver.close()


# --- the energy measures ------------------------------------------------------------

def _seed_fields(driver, seed=7):
    """Fill every primary array with a reproducible, direction-asymmetric pattern."""
    rng = np.random.default_rng(seed)
    for component in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        driver.set_field(component, rng.standard_normal(driver.grid.shape).astype(np.float32))


def test_electric_energy_in_box_runs_on_each_pair_s_own_lattice():
    """energy_and_flux.cpp:67-89 — (E_d, D_d) share a Yee shift, so cgrid is the component.

    Contrasted against the same reduction forced onto the CENTERED lattice, which is what
    rebuilding the energy from ``get_array`` would give. On a random field they are far
    apart (the four-point average annihilates the near-Nyquist residue), and only the raw
    form is MEEP's. The contrast is the assertion: both are finite, positive and
    plausible, so a value check alone cannot tell them apart.
    """
    driver = FdtdDriver(cell_size=(2, 2, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2))
    try:
        _seed_fields(driver)
        centre, extent = driver.total_volume()
        raw = driver.electric_energy_in_box(centre, extent)
        centred = 0.0
        for component in ("Ex", "Ey", "Ez"):
            # Force the Centered branch by asking for a pair that does NOT share a shift,
            # then picking the two the energy actually wants out of it.
            integral, _ = dft.integrate_field_function(
                driver.fields, driver.grid,
                (component, "D" + component[1], "Hz" if component != "Ez" else "Ex"),
                lambda _location, e, d, _extra: (e.conjugate() * d).real,
                centre, extent,
            )
            centred += 0.5 * integral.real
        assert raw > 0.0
        assert not math.isclose(raw, centred, rel_tol=0.05), (
            f"the raw-Yee energy {raw!r} and the centred one {centred!r} agree, so this "
            f"test can no longer tell the two lattices apart"
        )
    finally:
        driver.close()


def test_electric_energy_in_box_survives_a_mirror_fold():
    """A folded run reaches the discarded half through the reflected gather, not a clamp.

    The raw-Yee path is the new code, and a fold is where it can silently halve: a
    reducer that clamped its ladder at the mirror plane instead of reflecting through it
    would return the positive half's energy alone. The unfolded run of the same cell is
    the oracle, and the two must agree to the storage's own resolution.
    """
    seed = 11
    unfolded = FdtdDriver(cell_size=(2, 2, 0), resolution=10, dimensions=2,
                          boundaries=_walls(2))
    folded = FdtdDriver(cell_size=(2, 2, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2), symmetry=("X", "Y"))
    try:
        rng = np.random.default_rng(seed)
        # A field that IS mirror-symmetric under both planes, so the folded run can hold
        # it: build it on the full grid from an even function of both coordinates.
        nx, ny, nz = unfolded.grid.shape
        base = rng.standard_normal((nx // 2 + 1, ny // 2 + 1, nz)).astype(np.float32)
        for component in ("Dz",):
            full = np.zeros((nx, ny, nz), dtype=np.float32)
            for i in range(nx):
                for j in range(ny):
                    full[i, j] = base[min(i, nx - 1 - i), min(j, ny - 1 - j)]
            unfolded.set_field(component, full)
            folded.set_field(component, full[nx // 2 - 1:, ny // 2 - 1:][:folded.grid.shape[0],
                                                                        :folded.grid.shape[1]])
        centre, extent = unfolded.total_volume()
        whole = unfolded.electric_energy_in_box(centre, extent)
        centre_f, extent_f = folded.total_volume()
        half = folded.electric_energy_in_box(centre_f, extent_f)
        assert whole > 0.0
        # The two total volumes differ (a fold makes MEEP's own total volume symmetric),
        # so this pins the ORDER OF MAGNITUDE, not equality: a reducer that integrated
        # only the stored quadrant would come back near a quarter of this.
        assert 0.6 < half / whole < 1.6, (
            f"folded energy {half!r} against unfolded {whole!r}: the reflected gather is "
            f"not reaching the discarded half"
        )
    finally:
        unfolded.close()
        folded.close()


def test_modal_volume_is_the_energy_over_its_own_maximum():
    """energy_and_flux.cpp:288-290, and the nan an unstepped run must be allowed to give.

    MEEP answers ``nan`` on a field-free simulation (0/0) and
    ``assertAlmostEqual(x, nan)`` fails, which is a real outcome a test can assert on.
    Guarding it into a finite number would hand back a modal volume for a simulation with
    no field in it.
    """
    driver = FdtdDriver(cell_size=(2, 2, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2))
    try:
        centre, extent = driver.total_volume()
        assert math.isnan(driver.modal_volume_in_box(centre, extent))
        _seed_fields(driver)
        energy = driver.electric_energy_in_box(centre, extent)
        peak = driver.electric_energy_max_in_box(centre, extent)
        assert peak > 0.0
        assert driver.modal_volume_in_box(centre, extent) == pytest.approx(energy / peak,
                                                                          rel=1e-12)
    finally:
        driver.close()


# --- the magnetic half-step ---------------------------------------------------------

_MAGNETIC_ARRAYS = ("Bx", "By", "Bz", "Hx", "Hy", "Hz")


def test_synchronize_then_restore_is_bit_identical():
    """energy_and_flux.cpp:146-186 — every B and H array comes back exactly as it was.

    Bit-identical rather than close: ``restore_component`` is a ``memcpy``, and a restore
    that reconstructed the original by undoing the average would leave a drift that only
    shows after many calls. A PML is switched on so the ``f_u`` / ``f_cond`` auxiliaries
    the B ladder advances are in play — MEEP backs those up too, and restores them.
    """
    driver = FdtdDriver(cell_size=(2, 2, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2))
    try:
        driver.setup_pml({"x": 4, "y": 4})
        _seed_fields(driver)
        driver.step()
        before = {name: np.array(driver.fields.get_component(name), copy=True)
                  for name in _MAGNETIC_ARRAYS}
        driver.synchronize_magnetic_fields()
        during = {name: np.array(driver.fields.get_component(name), copy=True)
                  for name in _MAGNETIC_ARRAYS}
        driver.restore_magnetic_fields()
        for name in _MAGNETIC_ARRAYS:
            after = np.asarray(driver.fields.get_component(name))
            assert np.array_equal(after, before[name]), f"{name} was not restored exactly"
        # Something actually moved, or the round trip proves nothing.
        assert any(not np.array_equal(during[name], before[name]) for name in _MAGNETIC_ARRAYS)
    finally:
        driver.close()


def test_synchronize_is_refcounted_so_nesting_is_a_no_op():
    """``synchronized_magnetic_fields++`` / ``--`` — the second call must not half-step twice.

    ``field_energy_in_box`` synchronizes and then calls ``magnetic_energy_in_box``, which
    is itself a public entry point; without the counter the inner call would advance B a
    second time and the outer restore would put back the once-advanced arrays.
    """
    driver = FdtdDriver(cell_size=(2, 2, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2))
    try:
        _seed_fields(driver)
        driver.step()
        driver.synchronize_magnetic_fields()
        once = np.array(driver.fields.get_component("By"), copy=True)
        driver.synchronize_magnetic_fields()
        assert np.array_equal(np.asarray(driver.fields.get_component("By")), once)
        driver.restore_magnetic_fields()  # Inner: restores nothing.
        assert np.array_equal(np.asarray(driver.fields.get_component("By")), once)
        driver.restore_magnetic_fields()  # Outer: restores.
        assert not np.array_equal(np.asarray(driver.fields.get_component("By")), once)
        driver.restore_magnetic_fields()  # Nothing outstanding: a no-op, not an error.
    finally:
        driver.close()


def test_flux_in_box_synchronizes_and_leaves_the_fields_where_it_found_them():
    """``flux_in_box`` = synchronize -> ``flux_in_box_wrongH`` -> restore (:217-222).

    The two answers must DIFFER — B and H are half a step behind E and D, so a Poynting
    product of the stored arrays is a product at two different times — and the fields must
    be untouched afterwards. Measured on test_wvg_src's own cell the gap is 2.5e-02 on the
    near-field flux, and MEEP's published constant is the synchronized one.
    """
    driver = FdtdDriver(cell_size=(2, 2, 0), resolution=10, dimensions=2,
                        boundaries=_walls(2))
    try:
        _seed_fields(driver)
        driver.step()
        before = np.array(driver.fields.get_component("Hy"), copy=True)
        centre, extent = driver.total_volume()
        synchronized = driver.flux_in_box(0, centre, extent)
        unsynchronized = driver._flux_in_box_wrong_h(0, centre, extent)
        assert np.array_equal(np.asarray(driver.fields.get_component("Hy")), before)
        assert synchronized != unsynchronized
    finally:
        driver.close()


# --- MEEP's own total volume --------------------------------------------------------

@pytest.mark.parametrize(
    "cell, resolution, dimensions, symmetry, expected",
    [
        # Measured against sim.fields.total_volume() on MEEP 1.33.0, per axis.
        ((10, 10, 0), 20, 2, ("X", "Y"), ((0.0, 0.0, 0.0), (9.9, 9.9, 0.0))),
        ((10, 10, 0), 20, 2, None, ((-0.025, -0.025, 0.0), (9.95, 9.95, 0.0))),
        ((16, 8, 0), 10, 2, None, ((-0.05, -0.05, 0.0), (15.9, 7.9, 0.0))),
        ((2, 1.5, 0), 10, 2, None, ((-0.05, 0.0, 0.0), (1.9, 1.4, 0.0))),
        ((0, 0, 4), 20, 1, None, ((0.0, 0.0, -0.025), (0.0, 0.0, 3.95))),
    ],
)
def test_total_volume_is_meep_s_gv_interior_unioned_with_its_symmetry_images(
        cell, resolution, dimensions, symmetry, expected):
    """fields.cpp:716-724 — ``gv.interior()``, not the declared cell and not ``surroundings()``.

    One cell short at the top on every axis, and short at BOTH ends of a folded one (the
    union with the reflected image is symmetric). The 2 x 1.5 row is there because its y
    count is ODD, where MEEP's window sits half a cell above ``-L/2``; the 1-D row pins
    that the invariant axes come back at zero extent rather than at the declared cell.
    """
    driver = FdtdDriver(cell_size=cell, resolution=resolution, dimensions=dimensions,
                        boundaries=_walls(dimensions), symmetry=symmetry or ())
    try:
        centre, extent = driver.total_volume()
        assert centre == pytest.approx(expected[0], abs=1e-12)
        assert extent == pytest.approx(expected[1], abs=1e-12)
    finally:
        driver.close()


def test_cylindrical_is_refused_rather_than_answered():
    """A Dcyl run needs loop_in_chunks' ring measure and MEEP's (r, z) locations."""
    driver = FdtdDriver(cell_size=(2, 0, 2), resolution=10, cylindrical=True)
    try:
        with pytest.raises(NotImplementedError, match="Cartesian"):
            driver.electric_energy_in_box()
    finally:
        driver.close()


_BFAST_STATE_ARRAYS = ("f_bfast_Bx", "f_bfast_By", "f_bfast_Bz")


def test_synchronize_then_restore_returns_the_bfast_iir_state_exactly():
    """energy_and_flux.cpp:113/:130 — ``f_bfast`` is BACKUP'd and RESTORE'd with the rest.

    It has to be, and not merely for tidiness. The BFAST auxiliary is the state of
    ``F_n = S_n - F_{n-1}``, whose homogeneous mode is ``(-1)^n`` — MARGINALLY STABLE,
    undamped forever. Every other auxiliary this half-step touches decays: leave ``f_u``
    or ``f_cond`` one step ahead and the error dies out with the PML's own absorption.
    Leave ``f_bfast`` one step ahead and it never does, so ONE flux call would put a
    permanent alternating error into every subsequent timestep of the run.

    MEASURED with the ``f_bfast`` entries deleted from ``_SYNC_AUXILIARY``: one flux
    call, then 40 further steps, moves the six primary arrays 7.86e-01 relative to the
    same run without the flux call — against 0.0e+00 with them restored, on a run whose
    parity floor against CPU MEEP is 4.4e-06.

    A k with all three components nonzero, because a k along ONE axis leaves four of
    the six states identically zero: each component's two k's are indexed by its two
    partners' own directions, so ``k = (k_x, 0, 0)`` reaches only Bz/Dz (through the
    ``k_x`` slot) and By/Dy (through nothing at all in 2-D).
    """
    driver = FdtdDriver(cell_size=(2, 2, 2), resolution=10, dimensions=3,
                        bfast_scaled_k=(0.2, 0.3, 0.15), boundaries=_walls(3))
    try:
        assert driver.grid.bfast_active
        _seed_fields(driver)
        driver.step()
        for name in _BFAST_STATE_ARRAYS:
            assert float(np.abs(np.asarray(getattr(driver.fields, name))).max()) > 0.0, name
        before = {name: np.array(getattr(driver.fields, name), copy=True)
                  for name in _BFAST_STATE_ARRAYS}

        driver.synchronize_magnetic_fields()
        during = {name: np.array(getattr(driver.fields, name), copy=True)
                  for name in _BFAST_STATE_ARRAYS}
        driver.restore_magnetic_fields()

        for name in _BFAST_STATE_ARRAYS:
            after = np.asarray(getattr(driver.fields, name))
            assert np.array_equal(after, before[name]), f"{name} was not restored exactly"
        # The half-step DOES advance it, so the round trip is not vacuously true.
        assert any(not np.array_equal(during[name], before[name])
                   for name in _BFAST_STATE_ARRAYS)
    finally:
        driver.close()


def test_a_flux_call_mid_run_leaves_the_bfast_run_on_the_track_it_was_on():
    """The property the backup exists for: measuring must not change the run.

    A synchronize/restore pair is what ``flux_in_box`` does, so this is the shape of
    every mid-run flux measurement. With ``f_bfast`` restored the two tracks agree
    BIT for BIT; the failure mode without it is a slow, never-decaying drift rather
    than a blow-up, which is why it needs a bit-identity assertion and not a tolerance.
    """
    def _run(measure: bool):
        driver = FdtdDriver(cell_size=(2, 2, 2), resolution=10, dimensions=3,
                            bfast_scaled_k=(0.2, 0.3, 0.15), boundaries=_walls(3))
        try:
            _seed_fields(driver)
            for _ in range(8):
                driver.step()
            assert all(float(np.abs(np.asarray(getattr(driver.fields, name))).max()) > 0.0
                       for name in _BFAST_STATE_ARRAYS)
            if measure:
                centre, extent = driver.total_volume()
                driver.flux_in_box(2, centre, extent)
            for _ in range(40):
                driver.step()
            return {name: np.array(driver.fields.get_component(name), copy=True)
                    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz")}
        finally:
            driver.close()

    undisturbed = _run(measure=False)
    measured = _run(measure=True)
    for name, reference in undisturbed.items():
        assert np.array_equal(measured[name], reference), (
            f"a mid-run flux call moved {name} on a BFAST run — the f_bfast IIR state "
            f"was not put back, and its error mode never decays."
        )
    assert any(float(np.abs(value).max()) > 0.0 for value in undisturbed.values())
