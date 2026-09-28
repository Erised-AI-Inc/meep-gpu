"""
Value-level tests for the FDTD frequency-domain monitors (`dft.py`).

Most of this runs on the NumPy backend with stub grids and fields, so the
numerics stay verifiable on machines without CUDA; the two cross-validation
cases at the end drive the real engine against CPU MEEP. The suite pins the
quantities that a "nonzero output" smoke test would miss: steady-state amplitude
recovery, the half-step E/H sampling stagger, the fractional-cell surface
weighting of the flux integral (including for an off-centre plane), the
physical-volume-to-index mapping with and without mirror symmetry, and the
symmetry unfolding phases.

Registration has its own block at the end. Both monitor kinds derive their cells
from one function, so the tests there assert the DFT region and the flux plane
agree with each other *and* with an independent transcription of MEEP's
floor/ceil bracket — a shape that depends on the parity of a cell index, a
"zero-thickness" axis three cells thick, a region clamped at a periodic face, and
a region that resolves to no cells at all are all failures those pin, and each
had shipped. The end-to-end sweep against CPU MEEP lives in
test_driver_integration.py; these are its fast, oracle-free counterparts.

The multi-frequency cases target that feature's own failure modes rather than
its happy path. A spectrum accumulator can be wrong in ways that still look
like a spectrum: every bin carrying the first frequency's phase (a flat, fully
correlated spectrum), bins offset by one against `frequencies`, or the whole
stack quietly zeroed. So the frequency axis is pinned three ways — against an
independent double-precision evaluation of the same sum, against one dedicated
single-frequency monitor per bin, and against CPU MEEP's own `get_fluxes` /
`get_dft_array` at every frequency — and each contrast assertion is paired with
a value assertion, so nothing that produced no signal can pass by being small.

`_StubGrid` and `_StubFields` are minimal stand-ins for `grid.Grid` and
`fields.Fields`; they reproduce the registration conventions the monitors rely
on (cell centres at `-L/2 + (i+0.5)*dx`, or at `-0.5*dx + i*dx` on a
symmetry-halved axis) without coupling these tests to those modules.

The CPU-MEEP oracle runs in a subprocess and hands back a `.npz`: meep and torch
each ship their own OpenMP runtime and abort when loaded into one process, and
the pytest process reaches torch through neighbouring engine modules.
"""

from __future__ import annotations

import cmath
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from .grid import Grid, Mirror
from .driver import FdtdDriver
from .dft import (
    DFTMonitor,
    EnergyMonitor,
    FluxDftData,
    FluxMonitor,
    ForceMonitor,
    Near2FarMonitor,
    Near2FarRegion,
    _boundary_weight_ladder,
    _load_flux_planes,
    _box_regions,
    _fractional_cell_weights,
    _is_closed_box,
    _sliced_component,
    meep_frequency_span,
    normalize_frequencies,
    yee_shifts,
)

SQRT_TWO_PI = math.sqrt(2 * math.pi)


class _StubGrid:  # Minimal Grid stand-in with MEEP cell registration.
    def __init__(self, resolution, cell_size, courant=0.5, symmetry=(), xp=np):
        self.xp = xp
        self.resolution = float(resolution)
        self.courant = float(courant)
        self.symmetry = tuple(symmetry)
        self.dx = 1.0 / self.resolution
        self.dt = self.courant / self.resolution
        Lx, Ly, Lz = cell_size
        self.nx_full = int(round(Lx * self.resolution))
        self.ny_full = int(round(Ly * self.resolution))
        self.nz_full = int(round(Lz * self.resolution))
        # All three axes fold, and each one halves the same way. Modelling X and Y
        # only — which this stub used to do, with `nz` never halved, `z` never given
        # the mirror registration and `has_symmetry` reading two flags — made a Z fold
        # untestable HERE rather than merely untested: a unit test asking for one got
        # a grid that reported no symmetry at all and passed for the wrong reason.
        self.sym_x = "X" in self.symmetry
        self.sym_y = "Y" in self.symmetry
        self.sym_z = "Z" in self.symmetry
        # MEEP halve(): N - N//2 + 1 stored cells (== N//2 + 1 at even N), origin -2
        # for both parities; an unfolded axis starts at -(N - N%2), which is -L/2
        # only at even N. The stub models the registration Grid owns, so it carries
        # the same parity rules.
        self.nx = self.nx_full - self.nx_full // 2 + 1 if self.sym_x else self.nx_full
        self.ny = self.ny_full - self.ny_full // 2 + 1 if self.sym_y else self.ny_full
        self.nz = self.nz_full - self.nz_full // 2 + 1 if self.sym_z else self.nz_full
        self.Lx = self.nx_full * self.dx
        self.Ly = self.ny_full * self.dx
        self.Lz = self.nz_full * self.dx
        self.shape = (self.nx, self.ny, self.nz)
        self.x = self._axis(self.nx_full, self.nx, self.sym_x)
        self.y = self._axis(self.ny_full, self.ny, self.sym_y)
        self.z = self._axis(self.nz_full, self.nz, self.sym_z)

    def _axis(self, n_full, count, halved):  # Cell-centre coordinates for one axis.
        origin = -self.dx if halved else -(n_full - n_full % 2) * self.dx / 2
        start = origin + self.dx / 2
        return np.linspace(start, start + (count - 1) * self.dx, count, dtype=np.float32)

    def has_symmetry(self):  # True when any mirror plane is active.
        return self.sym_x or self.sym_y or self.sym_z


class _StubFields:  # Minimal Fields stand-in returning preset component arrays.
    def __init__(self, grid, components, dtype=np.complex64):
        self.grid = grid
        self._arrays = {
            name: np.zeros(grid.shape, dtype=dtype) for name in components
        }

    def set_uniform(self, name, value):  # Fill a component with one value; float32 storage refuses phase.
        self._arrays[name][...] = self._arrays[name].dtype.type(value)

    def set_array(self, name, values):  # Fill a component with a whole-grid array.
        self._arrays[name][...] = values.astype(self._arrays[name].dtype)

    def get_component(self, name):
        if name not in self._arrays:
            raise ValueError(f"Component '{name}' not allocated in the stub")
        return self._arrays[name]

    def component_factors(self, name):  # Fields.component_factors: the stub always stores.
        return (self.get_component(name),)

    def sliced_component(self, name, index):  # Fields.sliced_component: the stub always stores.
        return self.get_component(name)[index]


def _drive_monochromatic(monitor, fields, grid, num_steps, amplitude, omega, stagger_h=True):
    """Step a monitor through a spatially uniform monochromatic field.

    E components carry `amplitude * exp(-i*omega*t)`; H components carry the same
    waveform evaluated half a step earlier when `stagger_h` is True, which is how
    a leapfrog solver actually holds them. Returns the elapsed runtime.
    """
    for step in range(1, num_steps + 1):
        time = step * grid.dt
        for name in fields._arrays:
            if name[0] in ("E", "D"):
                fields.set_uniform(name, amplitude * np.exp(-1j * omega * time))
            else:
                sample_time = time - 0.5 * grid.dt if stagger_h else time
                fields.set_uniform(name, amplitude * np.exp(-1j * omega * sample_time))
        monitor.update(fields, time)
    return num_steps * grid.dt


def test_yee_shift_table_and_loud_unknown_component():  # Yee table values and loud failure.
    assert yee_shifts("Ex") == (1, 0, 0)
    assert yee_shifts("Ez") == (0, 0, 1)
    assert yee_shifts("Hy") == (1, 0, 1)
    assert yee_shifts("Bz") == (1, 1, 0)
    with pytest.raises(ValueError, match="Unknown field component"):
        yee_shifts("Eq")
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="Unknown field component"):
        DFTMonitor(grid=grid, frequency=1.0, components=("Ex", "Ew"))
    with pytest.raises(ValueError, match="at least one field component"):
        DFTMonitor(grid=grid, frequency=1.0, components=())


def test_steady_state_amplitude_recovery():  # A = DFT*sqrt(2*pi)/runtime within 1%.
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex", "Ez"))
    monitor.set_region_from_pml(2)
    assert monitor.region == (2, 8, 2, 8, 2, 8)
    fields = _StubFields(grid, ("Ex", "Ez"))
    amplitude = 2.0 - 0.5j
    runtime = _drive_monochromatic(
        monitor, fields, grid, num_steps=200, amplitude=amplitude, omega=2 * np.pi * 1.0
    )
    for component in ("Ex", "Ez"):
        recovered = monitor.get_dft(component) * SQRT_TWO_PI / runtime
        assert recovered.shape == (6, 6, 6)
        assert np.abs(recovered - amplitude).max() < 0.01 * abs(amplitude)


def test_zero_field_recovers_zero_amplitude():  # Degenerate case must not score as signal.
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex",))
    fields = _StubFields(grid, ("Ex",))
    for step in range(1, 51):
        monitor.update(fields, step * grid.dt)
    assert float(np.abs(monitor.get_dft("Ex")).max()) == 0.0
    assert float(monitor.get_intensity().max()) == 0.0


@pytest.mark.parametrize("factor", [2, 3, 5])
def test_explicit_decimation_preserves_monochromatic_amplitude(factor):
    """MEEP's n*dt weight exactly compensates for sampling every nth step."""
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    frequency = 0.8
    monitor = DFTMonitor(
        grid=grid,
        frequency=frequency,
        components=("Ex",),
        decimation_factor=factor,
        # The default whole-grid region reaches every far face; wrap the axes so
        # the uniform field averages with its own image rather than a wall zero
        # (a non-wrapping axis now means a PEC wall — see _sliced_component).
        periodic=(True, True, True),
    )
    fields = _StubFields(grid, ("Ex",))
    amplitude = 1.25 - 0.4j
    num_steps = factor * 80
    runtime = _drive_monochromatic(
        monitor,
        fields,
        grid,
        num_steps=num_steps,
        amplitude=amplitude,
        omega=2 * np.pi * frequency,
    )
    recovered = monitor.get_dft("Ex") * SQRT_TWO_PI / runtime
    assert np.max(np.abs(recovered - amplitude)) < 2e-5


def test_decimation_uses_the_global_step_index_and_reset_restarts_it():
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    monitor = DFTMonitor(
        grid=grid,
        frequency=1.0,
        components=("Ex",),
        decimation_factor=3,
    )
    fields = _StubFields(grid, ("Ex",))
    fields.set_uniform("Ex", 1.0)

    monitor.update(fields, 4 * grid.dt, current_step=4)
    monitor.update(fields, 5 * grid.dt, current_step=5)
    assert np.max(np.abs(monitor.get_dft("Ex"))) == 0.0
    monitor.update(fields, 6 * grid.dt, current_step=6)
    expected = (
        np.exp(1j * 2 * np.pi * 6 * grid.dt)
        * grid.dt
        * 3
        / SQRT_TWO_PI
    )
    assert monitor.get_dft("Ex")[3, 3, 3] == pytest.approx(expected, rel=2e-6)

    monitor.reset()
    monitor.update(fields, grid.dt)
    monitor.update(fields, 2 * grid.dt)
    assert np.max(np.abs(monitor.get_dft("Ex"))) == 0.0
    monitor.update(fields, 3 * grid.dt)
    assert np.max(np.abs(monitor.get_dft("Ex"))) > 0.0


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_dft_and_flux_monitors_reject_invalid_decimation(bad):
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="decimation_factor"):
        DFTMonitor(
            grid=grid,
            frequency=1.0,
            components=("Ex",),
            decimation_factor=bad,
        )
    with pytest.raises(ValueError, match="decimation_factor"):
        FluxMonitor(
            grid=grid,
            frequency=1.0,
            center=(0.0, 0.0, 0.0),
            size=(0.4, 0.4, 0.0),
            decimation_factor=bad,
        )


def test_driver_resolves_automatic_decimation_after_monitor_before_source():
    """The high-level API matches MEEP's source/monitor Nyquist rule when requested."""
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10)
    dft = driver.add_dft_monitor(
        frequencies=(0.8, 1.2),
        components=("Ez",),
        decimation_factor=0,
    )
    assert dft.decimation_factor == 1  # No bandwidth-limited source exists yet.

    driver.add_source(
        {
            "component": "Ez",
            "frequency": 1.0,
            "source_type": "gaussian",
            "fwidth": 0.4,
        }
    )
    # dt=.05, monitor edge=1.2, source edge=1+.4/2=1.2:
    # floor(1 / (.05 * (1.2 + 1.2))) = 8.
    assert dft.decimation_factor == 8

    flux = driver.add_flux_monitor(
        frequencies=(0.8, 1.2),
        center=(0.0, 0.0, 0.4),
        size=(0.4, 0.4, 0.0),
        decimation_factor=0,
    )
    explicit = driver.add_dft_monitor(
        frequencies=(0.8, 1.2),
        components=("Ez",),
        decimation_factor=1,
    )
    assert flux.decimation_factor == 8
    assert explicit.decimation_factor == 1


def test_driver_automatic_decimation_handles_general_and_unbandlimited_sources():
    """VolumeSource Gaussian envelopes count; CW/custom-only runs stay at one."""
    gaussian = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10)
    gaussian.add_source(
        {
            "component": "Hx",
            "frequency": 1.0,
            "source_type": "gaussian",
            "fwidth": 0.4,
            "is_integrated": True,
        }
    )
    assert gaussian.add_dft_monitor(
        1.2, components=("Ez",), decimation_factor=0
    ).decimation_factor == 8

    continuous = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10)
    continuous.add_source({"component": "Ez", "frequency": 1.0})
    assert continuous.add_dft_monitor(
        1.2, components=("Ez",), decimation_factor=0
    ).decimation_factor == 1

    custom = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10)
    custom.add_source(
        {
            "component": "Ez",
            "source_type": "custom",
            "src_func": lambda time: np.exp(-time * time),
        }
    )
    assert custom.add_dft_monitor(
        1.2, components=("Ez",), decimation_factor=0
    ).decimation_factor == 1


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_driver_rejects_invalid_automatic_decimation_requests(bad):
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10)
    with pytest.raises(ValueError, match="decimation_factor"):
        driver.add_dft_monitor(1.0, components=("Ez",), decimation_factor=bad)
    with pytest.raises(ValueError, match="decimation_factor"):
        driver.add_flux_monitor(
            1.0,
            center=(0.0, 0.0, 0.0),
            size=(0.4, 0.4, 0.0),
            decimation_factor=bad,
        )


def test_driver_automatic_decimation_is_identical_to_the_resolved_explicit_factor():
    """Automatic selection changes scheduling only; factor eight is reproducible explicitly."""
    def run(decimation_factor):
        driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10)
        driver.add_source(
            {
                "component": "Ez",
                "frequency": 1.0,
                "source_type": "gaussian",
                "fwidth": 0.4,
            }
        )
        monitor = driver.add_dft_monitor(
            frequencies=(0.8, 1.2),
            components=("Ez",),
            center=(0.0, 0.0, 0.4),
            size=(0.4, 0.4, 0.2),
            decimation_factor=decimation_factor,
        )
        driver.run(num_steps=80)
        return monitor.decimation_factor, np.asarray(monitor.get_dft_spectrum("Ez"))

    automatic_factor, automatic = run(0)
    explicit_factor, explicit = run(8)
    assert automatic_factor == explicit_factor == 8
    np.testing.assert_array_equal(automatic, explicit)


def _nonlinear_guard_driver():
    """The nonlinear composition probe's synthetic configuration, source included.

    dt = courant/resolution = 0.35/15, Gaussian edge = 0.8 + 0.4/2 = 1.0 and the
    monitor edge below is 1.0, so the LINEAR automatic factor is
    floor(1 / (dt * 2.0)) = 21 — the value the engine measurably resolved on a
    chi2/chi3 run before it mirrored MEEP's guard (recorded in
    parity/meep_gpu/probe_triton_nonlinear_composition.py, dft_flux_decimation).
    """
    driver = FdtdDriver(
        cell_size=(1.6, 1.6, 0.8),
        resolution=15.0,
        dimensions=3,
        boundaries="metallic",
        courant=0.35,
    )
    driver.add_source(
        {"component": "Ez", "frequency": 0.8, "source_type": "gaussian", "fwidth": 0.4}
    )
    return driver


def _probe_flux_monitor(driver, decimation_factor):
    return driver.add_flux_monitor(
        fcen=0.8,
        df=0.4,
        nfreq=5,
        center=(0.0, 0.0, 0.2),
        size=(0.8, 0.8, 0.0),
        decimation_factor=decimation_factor,
    )


def test_nonlinear_run_resolves_automatic_decimation_to_one():
    """MEEP's has_nonlinearities guard (dft.cpp:207-210): chi2/chi3 radiate
    harmonics outside the source band, so the source-bandwidth Nyquist bound is
    void and a nonlinear run samples every step."""
    linear = _nonlinear_guard_driver()
    assert _probe_flux_monitor(linear, 0).decimation_factor == 21  # Pinned: a
    # regression back to bandwidth-only resolution shows up as 21, not 1.

    nonlinear = _nonlinear_guard_driver()
    nonlinear.set_chi2(0.045)
    nonlinear.set_chi3(0.08)
    assert _probe_flux_monitor(nonlinear, 0).decimation_factor == 1
    assert nonlinear.add_dft_monitor(
        fcen=0.8, df=0.4, nfreq=5, components=("Ez",), decimation_factor=0
    ).decimation_factor == 1


def test_chi_installed_after_an_automatic_monitor_re_resolves_it():
    """MEEP never faces this ordering (its dft objects are created after the
    structure's chi2/chi3 exist, so add_dft always sees the final nonlinearity);
    here late installation re-resolves the factor, and removal — both members of
    the pair identically zero deletes it, structure.cpp:822-826 — restores the
    bandwidth-derived one."""
    driver = _nonlinear_guard_driver()
    monitor = _probe_flux_monitor(driver, 0)
    assert monitor.decimation_factor == 21

    driver.set_chi3(0.08)
    assert monitor.decimation_factor == 1

    driver.set_chi3(0.0)  # chi2 was never set: the trivial pair is removed.
    assert monitor.decimation_factor == 21


def test_explicit_decimation_is_not_overridden_by_the_nonlinear_guard():
    """MEEP's guard lives INSIDE the ``decimation_factor == 0`` block
    (dft.cpp:195-216): an explicit request is honored verbatim on a nonlinear
    run, whether chi arrives before or after the monitor."""
    chi_first = _nonlinear_guard_driver()
    chi_first.set_chi3(0.08)
    assert _probe_flux_monitor(chi_first, 4).decimation_factor == 4

    chi_last = _nonlinear_guard_driver()
    explicit = _probe_flux_monitor(chi_last, 4)
    chi_last.set_chi3(0.08)
    assert explicit.decimation_factor == 4


def test_e_and_h_sampling_is_staggered_by_half_a_step():  # Pins E at t, H at t - dt/2.
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    frequency = 1.0
    omega = 2 * np.pi * frequency
    amplitude = 1.0 + 0j
    num_steps = 160

    staggered = DFTMonitor(grid=grid, frequency=frequency, components=("Ex", "Hy"))
    fields = _StubFields(grid, ("Ex", "Hy"))
    runtime = _drive_monochromatic(
        staggered, fields, grid, num_steps, amplitude, omega, stagger_h=True
    )
    recovered_e = (staggered.get_dft("Ex") * SQRT_TWO_PI / runtime)[3, 3, 3]
    recovered_h = (staggered.get_dft("Hy") * SQRT_TWO_PI / runtime)[3, 3, 3]
    assert abs(recovered_e - amplitude) < 0.01
    assert abs(recovered_h - amplitude) < 0.01

    unstaggered = DFTMonitor(grid=grid, frequency=frequency, components=("Ex", "Hy"))
    fields = _StubFields(grid, ("Ex", "Hy"))
    runtime = _drive_monochromatic(
        unstaggered, fields, grid, num_steps, amplitude, omega, stagger_h=False
    )
    # Feeding H the E-time samples leaves exactly the half-step phase rotation
    # the monitor removes: exp(-i*omega*dt/2).
    expected = amplitude * np.exp(-1j * omega * 0.5 * grid.dt)
    recovered_h = (unstaggered.get_dft("Hy") * SQRT_TWO_PI / runtime)[3, 3, 3]
    assert abs(recovered_h - expected) < 0.01
    assert abs(recovered_h - amplitude) > 0.1


def test_region_mapping_known_values_without_symmetry():  # size*a + 2 cells for faces between centres.
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    monitor.set_region_from_volume(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 0.0))
    assert monitor.region == (9, 31, 9, 31, 19, 21)
    x, y, z = monitor.get_coordinates()
    assert x.shape == (22,)
    assert np.isclose(x[0], -1.05, atol=1e-5)
    assert np.isclose(x[-1], 1.05, atol=1e-5)
    assert np.isclose(z[0], -0.05, atol=1e-5)


def test_region_mapping_known_values_with_symmetry():  # Ladder on the halved axis, served through the fold.
    """The ladder's cells on a mirror-halved axis, whose origin is -dx and not -L/2.

    `_axis_origin_doubled` carries MEEP's `halve()` registration, so the same ladder
    that places a full axis places a folded one. The part of it below cell 0 is the
    mirror image, and it is REGISTERED, not clipped: the reflected gather
    (`folded_axis_sites` — MEEP's use_symmetry=true loop, dft.cpp:231) keeps the
    full requested span, so the region here matches the 22-cell full-domain ladder
    and the shape `mp.get_dft_array` reports for the same request. Requested cell
    ``-j`` samples stored cell ``1 + j`` (its image about the plane between cells
    -1 and... at doubled 0), with the component's parity applied at sample time.

    A monitor that already spans both sides has no quadrant to unfold, and
    `get_dft_full` says so instead of reflecting it again.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0), symmetry=("X", "Y"))
    assert (grid.nx, grid.ny, grid.nz) == (21, 21, 40)
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    monitor.set_region_from_volume(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 0.0))
    assert monitor.region == (-10, 12, -10, 12, 19, 21)
    x, _, _ = monitor.get_coordinates()
    assert np.isclose(x[0], -1.05, atol=1e-5)  # The full requested span, below-plane included.
    gather_x = monitor._gather[0]
    assert list(np.asarray(gather_x.base)[:3]) == [11, 10, 9]  # Cell -10 <- stored 11.
    assert list(np.asarray(gather_x.base)[-3:]) == [9, 10, 11]
    with pytest.raises(ValueError, match="below the X mirror plane"):
        monitor.get_dft_full("Ez")


def test_region_mapping_in_the_stored_half_is_untouched_by_the_fold_gather():
    """A region wholly at or above the plane registers exactly as before the gather."""
    grid = _StubGrid(10, (4.0, 4.0, 4.0), symmetry=("X",))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    monitor.set_region_from_volume(center=(0.55, 0.0, 0.0), size=(0.9, 2.0, 0.0))
    assert monitor.region[0] >= 0 and monitor._gather is None


def test_flux_normal_axis_brackets_the_requested_coordinate():
    """The plane's normal axis samples the cells that bracket it, weighted by proximity.

    MEEP's zero-thickness rule (loop_in_chunks.cpp case 4, "interpolation, not
    integration"): one cell at weight 1 when the request lands on a row of cell
    centres, otherwise the two rows either side at `w0` and `w1`. The expectation
    here is built from `grid.z` and a linear fraction — not from the ladder the
    monitor uses — so agreement means two independent derivations, not one
    formula compared with itself.

    The engine used to round this axis to a single index, which is why the
    off-centre coordinates below are the interesting ones: a request 0.001 above a
    row and one 0.049 above it used to return the same plane, and z = 0.5 or 1.0
    (a round multiple of dx, exactly between two rows) resolved half a cell away.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    centres = np.asarray(grid.z, dtype=np.float64)
    for plane_z in (0.45, 0.55, 0.451, 0.47, 0.5, 0.549, 0.0, 1.0, -0.32):
        monitor = FluxMonitor(
            grid=grid, frequency=1.0, center=(0.0, 0.0, plane_z), size=(2.0, 2.0, 0.0)
        )
        assert monitor.direction == 2
        assert monitor._region[:4] == (9, 31, 9, 31), "the in-plane registration must not move"
        z0, z1 = monitor._region[4:6]
        weights = np.asarray(monitor._weights)
        normal_weights = weights[weights.shape[0] // 2, weights.shape[1] // 2, :]
        on_a_row = np.isclose(centres, plane_z, atol=1e-6)
        if on_a_row.any():
            expected_cells, expected_weights = [int(np.argmax(on_a_row))], [1.0]
        else:
            above = int(np.searchsorted(centres, plane_z))
            expected_cells = [above - 1, above]
            fraction = (plane_z - centres[above - 1]) / grid.dx
            expected_weights = [1.0 - fraction, fraction]
        assert list(range(z0, z1)) == expected_cells, (
            f"z = {plane_z} samples cells {list(range(z0, z1))} "
            f"(centres {centres[z0:z1]}) rather than {expected_cells}."
        )
        np.testing.assert_allclose(
            normal_weights, expected_weights, atol=1e-6,
            err_msg=f"z = {plane_z}: normal weights {normal_weights} != {expected_weights}",
        )
        # The weights must interpolate, not integrate: they sum to 1 and their
        # weighted mean is the requested coordinate itself.
        assert float(normal_weights.sum()) == pytest.approx(1.0, abs=1e-6)
        assert float(normal_weights @ centres[z0:z1]) == pytest.approx(plane_z, abs=1e-6)


def test_flux_plane_off_the_grid_is_refused_rather_than_reporting_zero_power():
    """A monitor that misses the cell must raise; an empty accumulator reads as a real zero.

    The positive control is the same plane one step inside, which resolves to a
    populated region — without it this would pass against a constructor that
    refused every plane.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    inside = FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, 1.95), size=(2.0, 2.0, 0.0))
    assert inside._region[5] - inside._region[4] == 1
    for outside in (9.0, -9.0):
        with pytest.raises(ValueError, match="does not intersect"):
            FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, outside), size=(2.0, 2.0, 0.0))
    # A negative extent would invert the ladder's bounds and weight the plane with
    # a nonsense taper rather than failing.
    with pytest.raises(ValueError, match="non-negative"):
        FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, 0.45), size=(2.0, -2.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, float("nan")), size=(2.0, 2.0, 0.0))


def test_flux_direction_must_be_a_real_axis():  # Loud on an unknown normal direction.
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    with pytest.raises(ValueError, match="Flux direction"):
        FluxMonitor(
            grid=grid, frequency=1.0, center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 0.0),
            direction=3,
        )


def test_boundary_weight_ladder_integrates_to_the_requested_extent():  # Fractional-cell measure.
    resolution = 10.0
    for where_min, where_max in [(-1.0, 1.0), (-0.63, 1.37), (0.02, 0.42), (-0.25, 0.25)]:
        is_doubled, ie_doubled, _ = _boundary_weight_ladder(where_min, where_max, resolution)
        count = (ie_doubled - is_doubled) // 2 + 1
        grid = _StubGrid(resolution, (4.0, 4.0, 4.0))
        start = int(round((is_doubled + grid.nx_full - 1) / 2.0))
        weights = _fractional_cell_weights(
            grid, 0, where_min, where_max, start, count
        )
        expected_cells = (where_max - where_min) * resolution
        assert float(weights.sum()) == pytest.approx(expected_cells, abs=1e-4)


def _accumulate_uniform_plane_wave(monitor, grid, amplitude, frequency, num_steps):
    """Drive a flux monitor with a spatially uniform +z-travelling wave.

    Ex and Hy carry the same amplitude (vacuum impedance 1 in MEEP units) and the
    field is uniform across the plane, so the Yee-to-centre averaging is exact and
    the flux reduces to `|DFT|^2 * area` — isolating the surface weighting.
    """
    omega = 2 * np.pi * frequency
    fields = _StubFields(grid, ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    for step in range(1, num_steps + 1):
        time = step * grid.dt
        fields.set_uniform("Ex", amplitude * np.exp(-1j * omega * time))
        fields.set_uniform("Hy", amplitude * np.exp(-1j * omega * (time - 0.5 * grid.dt)))
        monitor.update(fields, time)
    return num_steps * grid.dt


def test_flux_of_an_analytic_plane_wave_matches_the_expected_poynting_value():
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    amplitude, frequency, num_steps = 1.5, 1.0, 200
    plane_size = (2.0, 2.0, 0.0)
    monitor = FluxMonitor(
        grid=grid, frequency=frequency, center=(0.0, 0.0, 0.45), size=plane_size
    )
    runtime = _accumulate_uniform_plane_wave(monitor, grid, amplitude, frequency, num_steps)
    dft_magnitude = amplitude * runtime / SQRT_TWO_PI
    expected = dft_magnitude**2 * plane_size[0] * plane_size[1]
    assert monitor.get_flux() == pytest.approx(expected, rel=0.01)


# The second run's complex amplitude, chosen so that the correct convention and
# every plausible corruption of it give THREE DIFFERENT numbers. With a = 1 and
# b = exp(2*pi*i/3), in units of one run's flux:
#
#   |b - a|**2 = 3     subtract the fields               <- MEEP's convention
#   |b + a|**2 = 1     load without the negation
#   |b|**2 - |a|**2 = 0    subtract the powers
#   Re[(b-a) conj(b+a)] = 0    negate the E side only
#
# A phase of i would have been useless here: |i - 1|**2 and |i + 1|**2 are both 2,
# so a dropped negation would have read as a pass.
_SECOND_RUN_AMPLITUDE = cmath.exp(2j * math.pi / 3)


def _normalization_pair(amplitude_a=1.0, frequency=1.0, num_steps=200):
    """One accumulation of a plane wave, plus an empty monitor on the same plane.

    ``|b| == |a|``, so the two runs carry the same POWER: a normalization that
    subtracted powers returns exactly zero, which is finite, smooth and entirely
    believable as a reflectance — the failure shape this package has been bitten by
    before. Subtracting the FIELDS returns ``|b - a|**2`` times one run's flux.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    plane = dict(center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0))
    first = FluxMonitor(grid=grid, frequency=frequency, **plane)
    _accumulate_uniform_plane_wave(first, grid, amplitude_a, frequency, num_steps)
    second = FluxMonitor(grid=grid, frequency=frequency, **plane)
    return grid, plane, frequency, num_steps, first, second


def test_flux_dft_data_round_trip_replaces_the_accumulator_exactly():
    """``load_dft_data`` is MEEP's REPLACE (meep.i:497-511), not an add."""
    grid, plane, frequency, num_steps, first, second = _normalization_pair()
    saved = first.get_dft_data()

    # Accumulate something else entirely into `second`, then load: the loaded run's
    # spectrum must survive intact rather than being summed with what was there.
    _accumulate_uniform_plane_wave(second, grid, 3.0, frequency, num_steps)
    assert second.get_flux() != pytest.approx(first.get_flux(), rel=1e-6)
    second.load_dft_data(saved)
    assert second.get_flux() == pytest.approx(first.get_flux(), rel=0, abs=0), (
        "load_dft_data must replace the accumulator bit for bit"
    )

    # And the saved copy is detached: stepping the monitor it came from further
    # must not reach back into it. That is what makes the first run disposable.
    _accumulate_uniform_plane_wave(first, grid, 5.0, frequency, num_steps)
    third = FluxMonitor(grid=grid, frequency=frequency, **plane)
    third.load_dft_data(saved)
    assert third.get_flux() == pytest.approx(second.get_flux(), rel=0, abs=0)


def test_load_minus_flux_data_subtracts_THE_FIELDS_not_the_power():
    """The minus convention: ``E2 - E1`` and ``H2 - H1``, then the Poynting product.

    MEEP's ``load_minus_flux_data`` loads the saved transform and negates it
    (``python/simulation.py:3642-3649`` -> ``dft_flux::scale_dfts``, dft.cpp:588-591
    -> ``dft_chunk::scale_dft``, dft.cpp:401-405), so the second run's accumulation
    lands on top of ``-E1``/``-H1``. The reported flux is
    ``Re[(E2-E1) x conj(H2-H1)]``, which is NOT ``flux2 - flux1``.

    Every plausible corruption reads a different number on this pair — 1x for a
    dropped negation, 0 for a difference of powers, 0 for a one-sided negation,
    against 3x for MEEP's convention (see ``_SECOND_RUN_AMPLITUDE``). Pinned again
    at the MEEP boundary by
    ``test_from_meep.py::test_two_run_normalization_idiom_matches_meeps_own``.
    """
    grid, plane, frequency, num_steps, first, second = _normalization_pair()
    unit_flux = first.get_flux()
    assert unit_flux > 0.0

    second.load_minus_dft_data(first.get_dft_data())
    _accumulate_uniform_plane_wave(second, grid, _SECOND_RUN_AMPLITUDE, frequency, num_steps)
    subtracted = second.get_flux()

    expected = abs(_SECOND_RUN_AMPLITUDE - 1.0) ** 2 * unit_flux
    assert subtracted == pytest.approx(expected, rel=1e-4), (
        f"subtracting the fields must give |b-a|**2 = 3 times one run's flux "
        f"({expected:.6e}); got {subtracted / unit_flux:.6f}x — 1x means the negation "
        f"was dropped, 0 means the powers were subtracted or only E was negated"
    )
    standalone = FluxMonitor(grid=grid, frequency=frequency, **plane)
    _accumulate_uniform_plane_wave(standalone, grid, _SECOND_RUN_AMPLITUDE, frequency, num_steps)
    difference_of_powers = standalone.get_flux() - unit_flux
    assert abs(difference_of_powers) < 1e-4 * unit_flux, (
        "the two runs must carry equal power, so a wrong convention reads zero here"
    )


def test_scale_dfts_scales_the_H_side_as_well_as_the_E_side():
    """``dft_flux::scale_dfts`` (dft.cpp:588-591) scales BOTH chunk lists.

    Scaling only E turns a negation into a sign flip of the flux instead of a
    subtraction of the field, which is why the whole-monitor scale is pinned
    rather than inferred from the flux (which is quadratic and cannot see it).
    """
    grid, _plane, frequency, num_steps, first, _second = _normalization_pair()
    before_E = {name: np.array(block, copy=True) for name, block in first._dft_E.items()}
    before_H = {name: np.array(block, copy=True) for name, block in first._dft_H.items()}
    flux_before = first.get_flux()

    first.scale_dfts(-1.0)
    for name, block in before_E.items():
        assert np.array_equal(np.asarray(first._dft_E[name]), -block)
    for name, block in before_H.items():
        assert np.array_equal(np.asarray(first._dft_H[name]), -block)
    assert first.get_flux() == pytest.approx(flux_before, rel=0, abs=0), (
        "negating both sides leaves the flux unchanged — (-E) x conj(-H) = E x conj(H). "
        "A one-sided negation would flip its sign, which is what this pins."
    )

    first.scale_dfts(2.0)
    assert first.get_flux() == pytest.approx(4.0 * flux_before, rel=1e-6)


def test_multi_plane_flux_data_pairs_each_plane_with_its_own_record():
    """The flux-BOX case: several planes, one container, and the pairing must hold.

    ``from_meep.FluxMigration`` — MEEP's ``add_flux(fcen, df, nfreq, *FluxRegions)``
    rebuilt as one driver monitor per region — saves and loads through this same
    helper with its ``parts`` list. The failure it must not have is a SWAP: two
    planes of the same shape and the same frequencies pass every structural check,
    so only the values can tell them apart. Each plane is therefore given a
    different amplitude and each is required to come back with its own.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    frequency, num_steps = 1.0, 200
    planes = [FluxMonitor(grid=grid, frequency=frequency,
                          center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0))
              for _ in range(2)]
    for monitor, amplitude in zip(planes, (1.0, 3.0)):
        _accumulate_uniform_plane_wave(monitor, grid, amplitude, frequency, num_steps)
    fluxes = [monitor.get_flux() for monitor in planes]
    assert fluxes[1] == pytest.approx(9.0 * fluxes[0], rel=1e-4), "9x, so a swap is visible"

    saved = FluxDftData(planes=tuple(
        record for monitor in planes for record in monitor.get_dft_data().planes))
    assert len(saved.planes) == 2

    fresh = [FluxMonitor(grid=grid, frequency=frequency,
                         center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0))
             for _ in range(2)]
    _load_flux_planes(saved, fresh, "test")
    assert [monitor.get_flux() for monitor in fresh] == pytest.approx(fluxes, rel=0, abs=0)

    # And loading the SAME container into the planes reversed must not go unnoticed:
    # nothing structural distinguishes them, so this is the only check there is.
    reversed_planes = [FluxMonitor(grid=grid, frequency=frequency,
                                   center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0))
                       for _ in range(2)]
    _load_flux_planes(FluxDftData(planes=tuple(reversed(saved.planes))),
                      reversed_planes, "test")
    assert [m.get_flux() for m in reversed_planes] == pytest.approx(list(reversed(fluxes)),
                                                                    rel=0, abs=0)


def test_flux_dft_data_refuses_data_that_does_not_belong_to_this_monitor():
    """Stricter than MEEP's total-size check, because a mismatch here BROADCASTS.

    MEEP compares one element count per side (``meep.i:499-505``) and aborts. This
    engine stores a named ``(nf, nx, ny, nz)`` block per component, so a block of
    the wrong shape or a frequency list that does not line up would broadcast into
    a plausible normalized spectrum instead of failing. Each is refused by name.
    """
    grid, plane, frequency, num_steps, first, second = _normalization_pair()
    saved = first.get_dft_data()

    wrong_shape = FluxMonitor(grid=grid, frequency=frequency,
                              center=plane["center"], size=(1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="shape"):
        wrong_shape.load_dft_data(saved)

    wrong_frequencies = FluxMonitor(grid=grid, frequencies=(frequency, 2.0 * frequency),
                                    **plane)
    with pytest.raises(ValueError, match="frequencies"):
        wrong_frequencies.load_dft_data(saved)

    with pytest.raises(TypeError, match="FluxDftData"):
        second.load_dft_data({"E": None, "H": None})

    with pytest.raises(ValueError, match="flux plane"):
        second.load_dft_data(FluxDftData(planes=saved.planes * 2))

    # A refusal must leave the monitor untouched rather than half-loaded.
    assert second.get_flux() == 0.0


def _accumulate_z_graded_plane_wave(monitor, grid, amplitude, frequency, num_steps, slope):
    """Drive a flux monitor with a +z wave whose Poynting integrand is linear in z.

    `Hy` is uniform and `Ex` carries `1 + slope*z` sampled half a cell low, which
    is where `Ex` actually sits on the Yee grid: the monitor's Yee-to-centre
    average, `0.5*(Ex[k] + Ex[k+1])`, then puts exactly `1 + slope*z_k` at cell
    centre `k`. The flux integrand `Re(Ex conj(Hy))` is therefore linear in z with
    no discretisation error, so the flux a correctly placed plane reports has a
    closed form at *any* requested coordinate — which a spatially uniform field
    (the test above) cannot distinguish.
    """
    omega = 2 * np.pi * frequency
    fields = _StubFields(grid, ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    z = np.asarray(grid.z, dtype=np.float64)
    profile = np.broadcast_to(
        (1.0 + slope * (z - 0.5 * grid.dx))[None, None, :], grid.shape
    )
    for step in range(1, num_steps + 1):
        time = step * grid.dt
        fields.set_array("Ex", amplitude * profile * np.exp(-1j * omega * time))
        fields.set_uniform("Hy", amplitude * np.exp(-1j * omega * (time - 0.5 * grid.dt)))
        monitor.update(fields, time)
    return num_steps * grid.dt


def test_flux_plane_interpolates_a_graded_field_onto_its_requested_coordinate():
    """Known value at eight normal coordinates, on and off the grid's rows of centres.

    With the integrand linear in z, `flux = |DFT|^2 * area * (1 + slope*z)` at the
    requested z exactly — the interpolation weights sum to 1 and their weighted
    mean is the request. This is the analytic form of the defect that used to
    make a plane at z = 1.0 read 7% low against CPU MEEP: the value depends on
    where the plane was asked for, and a placement that snapped to the nearest row
    of cell centres returns its neighbour's number instead.

    The tolerance is 2e-06 (measured worst 3.6e-07, float32 accumulators over 200
    steps) and the positions are 0.01 apart, which the slope turns into a 1.0%
    step in the expected flux — 5000x the tolerance, so a plane that ignored a
    tenth of a cell of its request cannot pass. That margin is asserted rather
    than described.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    amplitude, frequency, num_steps, slope = 1.5, 1.0, 200, 2.0
    plane_size = (2.0, 2.0, 0.0)
    positions = (0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.53, 0.55)
    measured, predicted = [], []
    for plane_z in positions:
        monitor = FluxMonitor(
            grid=grid, frequency=frequency, center=(0.0, 0.0, plane_z), size=plane_size
        )
        runtime = _accumulate_z_graded_plane_wave(
            monitor, grid, amplitude, frequency, num_steps, slope
        )
        dft_magnitude = amplitude * runtime / SQRT_TWO_PI
        expected = dft_magnitude**2 * plane_size[0] * plane_size[1] * (1.0 + slope * plane_z)
        measured.append(monitor.get_flux())
        predicted.append(expected)
        assert monitor.get_flux() == pytest.approx(expected, rel=2e-6), (
            f"z = {plane_z}: flux {monitor.get_flux():.6e} against the analytic "
            f"{expected:.6e} for a plane placed at its requested coordinate."
        )
    # The requested coordinate has to matter by much more than the tolerance,
    # otherwise the bound above would also be met by a plane that ignored it.
    for index in range(1, len(positions)):
        separation = abs(predicted[index] - predicted[index - 1]) / predicted[index]
        assert separation > 1000 * 2e-6, (
            f"z = {positions[index - 1]} and z = {positions[index]} differ by only "
            f"{separation:.3e}; the tolerance above would not distinguish them."
        )
    # Measured, not merely assumed: every position gives its own number.
    assert len(set(measured)) == len(positions), f"positions collapsed onto one flux: {measured}"
    # And a degenerate control: a zero field must give exactly zero, not the
    # `1 + slope*z` offset the weights would produce if they leaked a constant.
    quiet = FluxMonitor(grid=grid, frequency=frequency, center=(0.0, 0.0, 0.48), size=plane_size)
    _accumulate_z_graded_plane_wave(quiet, grid, 0.0, frequency, num_steps, slope)
    assert quiet.get_flux() == 0.0


def test_off_centre_flux_plane_carries_the_same_weighting():  # Pins the fractional-cell fix.
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    amplitude, frequency, num_steps = 1.5, 1.0, 200
    plane_size = (2.0, 2.0, 0.0)
    monitor = FluxMonitor(
        grid=grid, frequency=frequency, center=(0.37, -0.23, 0.45), size=plane_size
    )
    runtime = _accumulate_uniform_plane_wave(monitor, grid, amplitude, frequency, num_steps)
    dft_magnitude = amplitude * runtime / SQRT_TWO_PI
    expected = dft_magnitude**2 * plane_size[0] * plane_size[1]
    # The weight ladder is exact for any plane offset, so the bound is the same
    # 1% as for a centred plane. The superseded empirical edge weighting scores
    # +15.6% here when centred and +13.6% at this offset; unweighted summation
    # scores +21.0%. Both fail this bound, which is the point of the test.
    assert monitor.get_flux() == pytest.approx(expected, rel=0.01)
    weights = np.asarray(monitor._weights)
    assert float(weights.sum()) == pytest.approx(
        plane_size[0] * grid.resolution * plane_size[1] * grid.resolution, abs=1e-3
    )


def test_flux_weights_register_against_a_symmetry_halved_axis():
    """A plane spanning a folded axis keeps its full requested span and weights.

    On a symmetry-halved axis the doubled-coordinate origin is -2 rather than
    `-n_full`, so the ladder has to be offset differently. The requested 2.0-wide
    plane spans 22 ladder cells; the ten below the plane are served through the
    reflected gather (`folded_axis_sites`), so the weights integrate to the FULL
    requested extent — which is what makes the integral the power through the
    surface that was asked for, MEEP's own answer under symmetry
    (``loop_in_chunks`` visits each symmetry image, so ``mp.get_fluxes`` returns
    the full-domain number folded or not).
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0), symmetry=("X",))
    monitor = FluxMonitor(
        grid=grid, frequency=1.0, center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0)
    )
    assert monitor._region[:2] == (-10, 12)
    weights = np.asarray(monitor._weights)
    weights_x = weights[:, weights.shape[1] // 2, 0]
    assert float(weights_x.sum()) == pytest.approx(20.0, abs=1e-4)
    assert weights_x[0] == pytest.approx(0.125)  # Outer taper, both ends alike.
    assert weights_x[-1] == pytest.approx(0.125)
    weights_y = weights[weights.shape[0] // 2, :, 0]
    assert float(weights_y.sum()) == pytest.approx(20.0, abs=1e-4)
    monitor.get_flux()  # Served, not refused; the value is pinned against MEEP elsewhere.


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_flux_plane_spanning_the_fold_reads_the_reflected_gather(axis):
    """A surface reaching below a mirror plane samples its image cells, signed.

    ``_register_volume`` used to clamp the request at cell 0 of a folded axis and
    ``get_flux`` refused the misnamed half-surface integral. The reflected gather
    now serves the below-plane cells from their stored images — MEEP's own
    use_symmetry=true loop — so the registration spans the request and the read
    answers. Checked on all three axes, because a Z fold is exactly the case that
    used to skip its symmetry-aware paths in silence. The VALUE against
    ``mp.get_fluxes`` (sign included, both mirror parities, even and odd stored
    counts) is pinned by test_driver_vs_meep.py's fold-gather cases; this pins
    the registration shape and the per-cell image map.
    """
    from meep_gpu.dft import MIRROR_SITE_DIRECT, MIRROR_SITE_REFLECTED, _MirrorGather

    symmetry = (("X", "Y", "Z")[axis],)
    grid = _StubGrid(10, (4.0, 4.0, 4.0), symmetry=symmetry)
    size = [2.0, 2.0, 2.0]
    size[(axis + 1) % 3] = 0.0  # A plane normal to another axis, spanning the folded one.
    spanning = FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, 0.0), size=tuple(size))
    assert spanning._region[2 * axis] == -10  # The full request, below-plane included.
    assert spanning._region[2 * axis + 1] == 12
    entry = spanning._gather[axis]
    assert isinstance(entry, _MirrorGather)
    base = list(np.asarray(entry.base))
    assert base[:3] == [11, 10, 9]  # Requested cell -10 samples stored cell 11.
    assert base[-3:] == [9, 10, 11]
    spanning.get_flux()
    spanning.get_flux_spectrum()

    # A plane wholly inside the stored half registers exactly as it always did:
    # a contiguous slice, no gather.
    centre = [0.0, 0.0, 0.0]
    centre[axis] = 1.0
    inside = FluxMonitor(grid=grid, frequency=1.0, center=tuple(centre), size=tuple(size))
    assert inside._gather is None
    assert inside.get_flux() == 0.0  # Nothing accumulated, but it answers.


def test_flux_sign_follows_the_propagation_direction():  # Reversed H reverses the flux sign.
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    forward = FluxMonitor(
        grid=grid, frequency=1.0, center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0)
    )
    _accumulate_uniform_plane_wave(forward, grid, 1.0, 1.0, 120)
    backward = FluxMonitor(
        grid=grid, frequency=1.0, center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0)
    )
    _accumulate_uniform_plane_wave(backward, grid, 1.0, 1.0, 120)
    for component in ("Hx", "Hy", "Hz"):
        backward._dft_H[component] *= -1
    assert forward.get_flux() > 0.0
    assert backward.get_flux() == pytest.approx(-forward.get_flux(), rel=1e-5)


def test_flux_reset_clears_the_accumulators():  # Degenerate after reset, not stale.
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    monitor = FluxMonitor(
        grid=grid, frequency=1.0, center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0)
    )
    _accumulate_uniform_plane_wave(monitor, grid, 1.0, 1.0, 40)
    assert monitor.get_flux() > 0.0
    monitor.reset()
    assert monitor.get_flux() == 0.0


def test_symmetry_reconstruction_applies_the_mirror_phase():  # Quadrant unfolding, X mirror only.
    grid = _StubGrid(10, (2.0, 2.0, 0.4), symmetry=("X",))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex",))
    quadrant = monitor.get_dft("Ex")
    assert quadrant.shape == (grid.nx, grid.ny, grid.nz)
    nx_quadrant = grid.nx
    ramp = np.arange(1, nx_quadrant + 1, dtype=np.complex64)
    quadrant[...] = ramp[:, None, None]

    for phase_x in (+1, -1):
        full = monitor._reconstruct_from_quadrant(quadrant, (phase_x, None, None))
        nx_full = 2 * (nx_quadrant - 1)
        assert full.shape == (nx_full, grid.ny, grid.nz)
        cx = nx_full // 2
        for offset in range(nx_quadrant - 1):
            assert full[cx + offset, 0, 0] == pytest.approx(ramp[offset + 1])
            assert full[cx - 1 - offset, 0, 0] == pytest.approx(phase_x * ramp[offset + 1])


def test_symmetry_reconstruction_in_both_axes_multiplies_the_phases():  # Diagonal quadrant phase.
    grid = _StubGrid(10, (2.0, 2.0, 0.2), symmetry=("X", "Y"))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex",))
    quadrant = monitor.get_dft("Ex")
    rng = np.random.default_rng(7)
    quadrant[...] = rng.normal(size=quadrant.shape).astype(np.complex64)
    phase_x, phase_y = -1, +1
    full = monitor._reconstruct_from_quadrant(quadrant, (phase_x, phase_y, None))
    nx_full, ny_full = 2 * (grid.nx - 1), 2 * (grid.ny - 1)
    assert full.shape == (nx_full, ny_full, grid.nz)
    cx, cy = nx_full // 2, ny_full // 2
    assert full[cx + 1, cy + 1, 0] == pytest.approx(quadrant[2, 2, 0])
    assert full[cx - 2, cy + 1, 0] == pytest.approx(phase_x * quadrant[2, 2, 0])
    assert full[cx + 1, cy - 2, 0] == pytest.approx(phase_y * quadrant[2, 2, 0])
    assert full[cx - 2, cy - 2, 0] == pytest.approx(phase_x * phase_y * quadrant[2, 2, 0])


def test_symmetry_reconstruction_on_an_odd_folded_axis_clips_to_meeps_window():
    """Whole-axis unfold at an ODD full count: N cells out, top stored cell unpaired.

    The quadrant shape alone cannot say whether ``m`` stored cells came from a
    full count of ``2m - 2`` or ``2m - 3`` — parity is genuinely lost at
    ``halve()`` — so the full count is taken from the grid, never derived as
    ``2 * (m - 1)``. For N=21 the stored 12 cells unfold to 21: ten mirrored
    cells (the eleventh image would sit half a cell below MEEP's shifted window
    and is clipped), the straddling cell, and the ten remaining positive cells.
    """
    grid = _StubGrid(10, (2.1, 2.0, 0.4), symmetry=("X",))
    assert (grid.nx_full, grid.nx) == (21, 12)
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex",))
    quadrant = monitor.get_dft("Ex")
    ramp = np.arange(1, grid.nx + 1, dtype=np.complex64)
    quadrant[...] = ramp[:, None, None]

    for phase_x in (+1, -1):
        full = monitor._reconstruct_from_quadrant(quadrant, (phase_x, None, None))
        assert full.shape == (grid.nx_full, grid.ny, grid.nz)
        cx = grid.nx_full // 2  # 10: first cell on the positive side of the plane.
        # Positive block: quadrant cell q is full cell cx - 1 + q, filling to the top.
        for q in range(1, grid.nx):
            assert full[cx - 1 + q, 0, 0] == pytest.approx(ramp[q])
        # Mirrored block (written last, straddling cell included, as at even counts):
        # full cell i is the image of stored cell cx - i.
        for i in range(cx):
            assert full[i, 0, 0] == pytest.approx(phase_x * ramp[cx - i])

    # The coordinates pair one-to-one with the reconstruction on the odd axis too.
    x, y, z = monitor.get_coordinates_full()
    assert (x.size, y.size, z.size) == (grid.nx_full, grid.ny, grid.nz)
    assert np.allclose(np.diff(x), grid.dx, atol=1e-5)
    assert np.isclose(x[0], -(grid.nx_full // 2 - 0.5) * grid.dx, atol=1e-5)
    assert np.isclose(x[-1], (grid.nx_full // 2 + 0.5) * grid.dx, atol=1e-5)

    # And the no-electric-components fallback of get_intensity_full agrees.
    empty = DFTMonitor(grid=grid, frequency=1.0, components=("Hz",))
    assert empty.get_intensity_full().shape == (grid.nx_full, grid.ny, grid.nz)


def test_full_coordinates_pair_one_to_one_with_the_reconstruction():
    """Unfolded axes must be the same length as the unfolded data, and uniform.

    Naively mirroring the whole quadrant yields two extra coordinates per
    mirrored axis, silently mispairing every value in the reconstruction.
    """
    grid = _StubGrid(10, (2.0, 2.0, 0.4), symmetry=("X", "Y"))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex",))
    x, y, z = monitor.get_coordinates_full()
    full_shape = monitor.get_dft_full("Ex").shape
    assert (x.size, y.size, z.size) == full_shape
    assert np.allclose(np.diff(x), grid.dx, atol=1e-5)
    assert np.isclose(x[0], -(grid.nx - 1.5) * grid.dx, atol=1e-5)
    assert np.isclose(x[full_shape[0] // 2 - 1], -0.5 * grid.dx, atol=1e-5)


def test_no_symmetry_reconstruction_is_a_passthrough():  # get_dft_full == get_dft without mirrors.
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex",))
    monitor.get_dft("Ex")[...] = 3.0 + 1j
    assert np.array_equal(monitor.get_dft_full("Ex"), monitor.get_dft("Ex"))


def test_intensity_full_empty_fallback_matches_the_populated_shape():  # Fix: shape from the quadrant.
    """A monitor with no electric components returns the shape a populated one would.

    The cell used to be 0.4 thick in z, which at resolution 10 is four cells — and
    a 2-cell PML leaves *no* interior there. Both monitors then carried a
    `(7, 7, 0)` region, and this test compared one empty shape with another: it
    passed because both sides had been reduced to nothing, which is the failure
    mode the comparison exists to catch. The cell is now thick enough to have an
    interior, the populated shape is asserted to be populated, and the region that
    collapses to nothing is checked to raise (`_require_populated_region`) instead
    of being allocated and silently reduced to zeros.
    """
    grid = _StubGrid(10, (2.0, 2.0, 1.0), symmetry=("X", "Y"))
    populated = DFTMonitor(grid=grid, frequency=1.0, components=("Ex",))
    populated.set_region_from_pml(2)
    magnetic_only = DFTMonitor(grid=grid, frequency=1.0, components=("Hz",))
    magnetic_only.set_region_from_pml(2)
    # 20 full-domain cells per mirrored axis less the 2-cell PML on its one
    # absorbing face; z keeps its interior between two absorbing faces.
    assert populated.get_intensity_full().shape == (16, 16, 6), "the control must carry cells"
    assert magnetic_only.get_intensity_full().shape == populated.get_intensity_full().shape

    too_thin = _StubGrid(10, (2.0, 2.0, 0.4), symmetry=("X", "Y"))
    monitor = DFTMonitor(grid=too_thin, frequency=1.0, components=("Ex",))
    with pytest.raises(ValueError, match="does not intersect"):
        monitor.set_region_from_pml(2)  # 4 cells of z, 2 absorbed on each face.


def test_get_dft_rejects_an_unmonitored_component():  # Loud, not KeyError or silent zeros.
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex",))
    with pytest.raises(ValueError, match="not monitored"):
        monitor.get_dft("Ey")


def test_accumulator_dtypes_are_complex64():  # dtype contract shared with the sibling modules.
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex", "Hy"))
    assert monitor.get_dft("Ex").dtype == np.complex64
    flux = FluxMonitor(
        grid=grid, frequency=1.0, center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0)
    )
    assert flux._dft_E["Ex"].dtype == np.complex64
    assert flux._weights.dtype == np.float32
    fields = _StubFields(grid, ("Ex", "Hy"))
    fields.set_uniform("Ex", 1.0)
    monitor.update(fields, grid.dt)
    assert monitor.get_dft("Ex").dtype == np.complex64


# --- Multi-frequency monitors -------------------------------------------------


def _drive_random_fields(monitors, fields, grid, num_steps, seed, extra=None):
    """Step monitors through an uncorrelated field sequence, returning the samples.

    Random fields are deliberate: a monochromatic drive is invariant under most
    of the ways a frequency axis can be mis-wired (one bin's phase copied to all,
    bins transposed against `frequencies`), while an uncorrelated sequence makes
    every bin a different number. `extra(time)` is called after each update so a
    caller can accumulate its own reference from the same field state.
    """
    rng = np.random.default_rng(seed)
    for step in range(1, num_steps + 1):
        time = step * grid.dt
        for name in fields._arrays:
            fields.set_array(
                name, rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
            )
        for monitor in monitors:
            monitor.update(fields, time)
        if extra is not None:
            extra(time)
    return num_steps * grid.dt


def _accumulate_tones(monitor, grid, tones, num_steps):
    """Drive a flux monitor with a uniform +z plane wave carrying several tones.

    Ex and Hy carry the same amplitudes (vacuum impedance 1 in MEEP units), so
    the flux at each tone reduces to `|A*T/sqrt(2*pi)|^2 * area`. H is sampled
    half a step early, as a leapfrog solver holds it.
    """
    fields = _StubFields(grid, ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    for step in range(1, num_steps + 1):
        time = step * grid.dt
        electric = sum(amp * np.exp(-2j * np.pi * freq * time) for amp, freq in tones)
        magnetic = sum(
            amp * np.exp(-2j * np.pi * freq * (time - 0.5 * grid.dt)) for amp, freq in tones
        )
        fields.set_uniform("Ex", electric)
        fields.set_uniform("Hy", magnetic)
        monitor.update(fields, time)
    return num_steps * grid.dt


def test_frequency_normalization_accepts_both_spellings_and_rejects_the_rest():
    assert normalize_frequencies(1.25) == (1.25,)
    assert normalize_frequencies(frequencies=1.25) == (1.25,)
    assert normalize_frequencies([0.8, 1.0]) == (0.8, 1.0)
    assert normalize_frequencies(frequencies=np.linspace(0.8, 1.2, 3)) == (0.8, 1.0, 1.2)
    # Both spellings at once is accepted only when they agree: spelling one
    # monitor's frequencies twice is redundant, two different sets are a bug.
    assert normalize_frequencies(1.0, 1.0) == (1.0,)
    assert normalize_frequencies([0.8, 1.0], (0.8, 1.0)) == (0.8, 1.0)
    with pytest.raises(ValueError, match="Conflicting monitor frequencies"):
        normalize_frequencies(1.0, [1.0, 1.2])
    with pytest.raises(ValueError, match="at least one frequency"):
        normalize_frequencies()
    with pytest.raises(ValueError, match="at least one frequency"):
        normalize_frequencies(frequencies=[])
    with pytest.raises(ValueError, match="distinct"):
        normalize_frequencies([1.0, 1.2, 1.0])
    # A NEGATIVE frequency is a valid monitor frequency: MEEP stores omega raw
    # (dft.cpp:219-221), accumulates polar(1, omega*t) with that sign
    # (dft.cpp:268-270), and abs()-es it only for the decimation bound
    # (dft.cpp:201-206) — the minus-frequency near2far of
    # dipole_in_vacuum_cyl_off_axis.py rides on it.
    assert normalize_frequencies([1.0, -1.0]) == (1.0, -1.0)
    # ZERO is valid too, and for the same reason read one line earlier: `update_dft`
    # has NO branch on omega (dft.cpp:266-269), so at omega = 0 the phase factor is
    # exactly 1 and the bin is a running time integral of the field. Measured against
    # MEEP's own accumulator in test_from_meep
    # (test_a_dc_dft_monitor_accumulates_what_meeps_own_accumulator_does): 1.35e-06
    # absolute on a DC-only monitor and 2.31e-06 mixed with a 0.8 bin whose own
    # residual is 2.61e-06 — one accumulation floor, shared.
    assert normalize_frequencies([1.0, 0.0]) == (1.0, 0.0)
    with pytest.raises(ValueError, match="must be finite"):
        normalize_frequencies([1.0, float("nan")])
    with pytest.raises(ValueError, match="must be finite"):
        normalize_frequencies([1.0, float("inf")])
    # "1e15" parses as a float, so a text frequency has to be refused explicitly.
    with pytest.raises(ValueError, match="must be numbers"):
        normalize_frequencies("1.0e15")
    with pytest.raises(ValueError, match="must be numbers"):
        normalize_frequencies([1.0, "1.2"])
    with pytest.raises(ValueError, match="must be numbers"):
        normalize_frequencies(object())


def test_negative_frequency_accumulates_the_conjugate_of_the_positive_bin():
    """A -f monitor bin is MEEP's conjugate accumulator, and it is exact here.

    dft.cpp stores ``omega[i] = 2*pi*freq[i]`` raw (:219-221) and rotates
    ``polar(1.0, omega*time)`` with that sign (:268-270), so for a REAL field
    sequence the -f bin must equal the conjugate of the +f bin — an identity of
    the arithmetic, not a tolerance: ``exp(+i|w|t)`` and ``conj(exp(-i|w|t))``
    round identically and the real field factor distributes over the conjugate.
    Anything short of equality means the two bins were not accumulated with
    sign-symmetric phases.
    """
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    monitor = DFTMonitor(grid=grid, frequencies=[0.8, -0.8], components=("Ex",))
    fields = _StubFields(grid, ("Ex",))
    rng = np.random.default_rng(7)
    for step in range(1, 41):
        fields.set_uniform("Ex", float(rng.normal()))
        monitor.update(fields, step * grid.dt)
    plus = np.asarray(monitor.get_dft("Ex", 0))
    minus = np.asarray(monitor.get_dft("Ex", 1))
    assert float(np.abs(plus).max()) > 0.0, "the drive accumulated nothing"
    assert np.array_equal(minus, np.conj(plus)), (
        "the -f bin is not the conjugate of the +f bin: worst "
        f"{np.abs(minus - np.conj(plus)).max():.3e}"
    )


def test_near2far_monitor_refuses_negative_frequencies_by_name():
    """The raw accumulators take -f; the far-field EVALUATOR does not — refused loudly.

    The mp.Simulation migration stores a negative-frequency near2far as raw Yee
    accumulators and MEEP evaluates the far field (load_near2far); this class's
    own green3d/greencyl are measured at positive frequencies only, so handing
    them a negative one must raise with the workaround named, not radiate an
    unpinned answer.
    """
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    patch = Near2FarRegion(center=(0.05, -0.15, 0.25), size=(0.0, 0.0, 0.0),
                           weight=1.0, direction=2)
    with pytest.raises(ValueError, match="positive frequencies only"):
        Near2FarMonitor(grid, frequencies=[1.0, -1.0], regions=[patch], closed=False)


def test_meep_frequency_span_reproduces_meeps_expansion():
    """MEEP's (fcen, df, nfreq) rule, including the single-point special case.

    `meep.simulation.fix_dft_args` returns `[fcen]` for nfreq == 1 rather than
    `linspace(..., 1)`, which would sit at `fcen - df/2`. Half a bandwidth is a
    plausible-looking wrong frequency, so the case is pinned explicitly.
    """
    assert meep_frequency_span(1.0, 0.4, 5) == pytest.approx((0.8, 0.9, 1.0, 1.1, 1.2))
    assert meep_frequency_span(1.0, 0.4, 1) == (1.0,)
    assert meep_frequency_span(1.0, 0.0, 1) == (1.0,)
    with pytest.raises(ValueError, match="nfreq"):
        meep_frequency_span(1.0, 0.4, 0)
    with pytest.raises(ValueError, match="df"):
        meep_frequency_span(1.0, -0.4, 3)


def test_accumulators_carry_a_leading_frequency_axis():  # Shapes and accessors of the contract.
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    frequencies = (0.8, 1.0, 1.2)
    monitor = DFTMonitor(grid=grid, frequencies=frequencies, components=("Ex", "Hy"))
    assert monitor.frequencies == frequencies
    assert monitor.num_frequencies == 3
    assert monitor.frequency == 0.8  # The frequency the default freq_index=0 accessors return.
    assert monitor.get_dft_spectrum("Ex").shape == (3, 10, 10, 10)
    assert monitor.get_dft("Ex", 2).shape == (10, 10, 10)
    assert monitor.get_dft("Ex").shape == (10, 10, 10)
    assert monitor.get_intensity(1).shape == (10, 10, 10)
    # get_dft hands back a view, as it did before the frequency axis existed.
    monitor.get_dft("Ex", 1)[...] = 2.0 + 1j
    assert np.all(monitor.get_dft_spectrum("Ex")[1] == np.complex64(2.0 + 1j))
    assert np.all(monitor.get_dft_spectrum("Ex")[0] == 0)
    flux = FluxMonitor(
        grid=grid, frequencies=frequencies, center=(0.0, 0.0, 0.05), size=(0.4, 0.4, 0.0)
    )
    assert flux.frequencies == frequencies and flux.num_frequencies == 3
    assert flux._dft_E["Ex"].shape == (3, 6, 6, 1)
    assert flux._weights.shape == (6, 6, 1)  # Surface measure is frequency-independent.
    assert flux.get_flux_spectrum().shape == (3,)
    # The single-frequency call shape still works, positionally and by keyword.
    assert DFTMonitor(grid, 1.0, components=("Ez",)).frequencies == (1.0,)
    assert DFTMonitor(grid=grid, frequency=1.0, components=("Ez",)).num_frequencies == 1


def test_frequency_index_is_validated_loudly():  # No wraparound, no silent clamp.
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    monitor = DFTMonitor(grid=grid, frequencies=(0.8, 1.0), components=("Ex",))
    flux = FluxMonitor(
        grid=grid, frequencies=(0.8, 1.0), center=(0.0, 0.0, 0.05), size=(0.4, 0.4, 0.0)
    )
    for bad in (-1, 2, 7):
        with pytest.raises(ValueError, match="out of range"):
            monitor.get_dft("Ex", bad)
        with pytest.raises(ValueError, match="out of range"):
            monitor.get_intensity(bad)
        with pytest.raises(ValueError, match="out of range"):
            flux.get_flux(bad)
    with pytest.raises(ValueError, match="whole number"):
        monitor.get_dft("Ex", 0.5)
    with pytest.raises(ValueError, match="whole number"):
        flux.get_flux("first")


def test_accumulator_budget_refuses_an_implausible_allocation():
    """The nf multiplier must be refused before the allocator sees it.

    A working single-frequency monitor turned into a 200-point spectrum is 200x
    the memory; on GPU that aborts the job mid-run and on host it swaps. The
    positive control is the same frequency set on a region that does fit, so the
    guard cannot pass by refusing everything.
    """
    large = _StubGrid(10, (10.0, 10.0, 10.0))  # 100^3 cells: 24 MB per frequency, 3 components.
    frequencies = np.linspace(0.5, 1.5, 200)
    with pytest.raises(ValueError, match="MAX_ACCUMULATOR_BYTES") as raised:
        DFTMonitor(grid=large, frequencies=frequencies, components=("Ex", "Ey", "Ez"))
    assert "4 GiB per-monitor limit" in str(raised.value)
    assert "200 frequencies" in str(raised.value)
    small = _StubGrid(10, (1.0, 1.0, 1.0))
    fits = DFTMonitor(grid=small, frequencies=frequencies, components=("Ex", "Ey", "Ez"))
    assert fits.get_dft_spectrum("Ex").shape == (200, 10, 10, 10)
    fine = _StubGrid(100, (10.0, 10.0, 10.0))  # 1000x1000 flux plane: 48 MB per frequency.
    with pytest.raises(ValueError, match="MAX_ACCUMULATOR_BYTES"):
        FluxMonitor(
            grid=fine,
            frequencies=np.linspace(0.5, 1.5, 100),
            center=(0.0, 0.0, 0.005),
            size=(10.0, 10.0, 0.0),
        )


def test_single_frequency_path_is_unchanged_bit_for_bit():
    """A one-frequency monitor must reproduce the pre-spectrum numbers exactly.

    The reference here is the code the frequency axis replaced: a complex64
    scalar phase multiplied into the cell-centred slice and accumulated into an
    (nx, ny, nz) buffer, then reduced by the weighted Poynting sum. Equality is
    bitwise, not approximate — a spectrum feature that moved the last bit of
    every existing single-frequency result would quietly re-baseline the parity
    numbers the rest of this engine is measured against.
    """
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    frequency = 0.87
    omega = 2 * np.pi * frequency
    scale = grid.dt / SQRT_TWO_PI
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    monitor = DFTMonitor(grid=grid, frequency=frequency, components=components)
    flux = FluxMonitor(
        grid=grid, frequency=frequency, center=(0.0, 0.0, 0.05), size=(0.4, 0.4, 0.0)
    )
    fields = _StubFields(grid, components)
    legacy = {name: np.zeros(grid.shape, dtype=np.complex64) for name in components}

    def accumulate_legacy(time):  # The superseded scalar-phase accumulation, verbatim.
        for name in components:
            sample_time = time if name[0] in ("E", "D") else time - 0.5 * grid.dt
            phase = np.complex64(np.exp(1j * omega * sample_time) * scale)
            legacy[name] += phase * _sliced_component(fields, grid, name, monitor.region)

    _drive_random_fields([monitor, flux], fields, grid, 40, seed=5, extra=accumulate_legacy)
    for name in components:
        assert np.array_equal(monitor.get_dft(name), legacy[name]), f"{name} DFT drifted"

    weights = flux._weights
    Ex = flux._dft_E["Ex"][0] * weights
    Ey = flux._dft_E["Ey"][0] * weights
    Ez = flux._dft_E["Ez"][0] * weights
    Hx, Hy = flux._dft_H["Hx"][0], flux._dft_H["Hy"][0]
    # dA is one product, applied once: reducing in float32 and then multiplying by
    # dx twice separately lands a whole ULP away, which is the resolution this
    # comparison works at.
    dA = grid.dx * grid.dx
    legacy_flux = float(np.sum(np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))) * dA * flux.weight)
    assert abs(legacy_flux) > 0.0, "The reference flux is zero; the comparison would be vacuous."
    assert flux.get_flux() == legacy_flux
    assert flux.get_flux_spectrum()[0] == legacy_flux
    assert Ez.shape == (6, 6, 1)  # The Z-normal reduction ignores Ez; shape is still pinned.


def test_one_multifrequency_monitor_equals_one_monitor_per_frequency():
    """Bin i of an nf-point monitor must equal a dedicated monitor at frequency i.

    This is the structural assertion the whole feature rests on, and it is exact:
    the per-timestep accumulation is the same arithmetic whether the phase comes
    from a length-1 or a length-nf array. It fails outright for the failure modes
    a smoke test cannot see — one phase broadcast to every bin, bins offset
    against `frequencies`, or a stale phase reused across the stack.
    """
    grid = _StubGrid(10, (1.0, 1.0, 1.0))
    frequencies = (0.63, 0.91, 1.0, 1.37)
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    plane = {"center": (0.0, 0.0, 0.05), "size": (0.4, 0.4, 0.0)}
    combined = DFTMonitor(grid=grid, frequencies=frequencies, components=components)
    combined_flux = FluxMonitor(grid=grid, frequencies=frequencies, **plane)
    singles = [DFTMonitor(grid=grid, frequency=f, components=components) for f in frequencies]
    single_fluxes = [FluxMonitor(grid=grid, frequency=f, **plane) for f in frequencies]
    fields = _StubFields(grid, components)
    _drive_random_fields(
        [combined, combined_flux, *singles, *single_fluxes], fields, grid, 40, seed=11
    )
    spectrum = combined_flux.get_flux_spectrum()
    for index, frequency in enumerate(frequencies):
        for name in components:
            assert np.array_equal(combined.get_dft(name, index), singles[index].get_dft(name)), (
                f"bin {index} (f={frequency}) of the {name} spectrum differs from its own monitor"
            )
        assert spectrum[index] == single_fluxes[index].get_flux()
    # ...and the bins are genuinely different numbers, so the equality above is
    # not being satisfied by a stack of identical planes.
    assert len({float(np.abs(combined.get_dft("Ex", i)).sum()) for i in range(4)}) == 4
    assert len(set(spectrum)) == 4


def test_cw_amplitude_is_recovered_at_resonance_and_suppressed_off_it():
    """The analytic case: one CW tone, several monitored frequencies.

    Every bin is checked against the same sum evaluated in double precision, so
    the oracle covers phase, scale and bin alignment rather than magnitude alone.
    Four of the monitored frequencies are commensurate with the run window
    (their offsets from the source are whole multiples of 1/runtime) and must
    cancel to nothing; the fifth sits half a bin away, where the rectangular
    window's Dirichlet kernel has a known, non-zero sidelobe. That two-sided
    bound is what keeps the suppression assertion honest: a stack that was
    zeroed fails the sidelobe's lower bound and the on-resonance recovery, while
    one that copied the source bin's phase into every bin fails its upper bound.
    """
    grid = _StubGrid(10, (1.0, 1.0, 1.0))  # dt = 0.05, so 400 steps is a runtime of 20.
    source_frequency, num_steps, amplitude = 1.0, 400, 1.7 - 0.6j
    frequencies = (0.6, 0.8, 1.0, 1.2, 1.425)
    monitor = DFTMonitor(grid=grid, frequencies=frequencies, components=("Ex",))
    fields = _StubFields(grid, ("Ex",))
    runtime = _drive_monochromatic(
        monitor, fields, grid, num_steps, amplitude, 2 * np.pi * source_frequency
    )
    spectrum = np.asarray(monitor.get_dft_spectrum("Ex"))
    assert spectrum.shape == (5, 10, 10, 10)

    sample_times = np.arange(1, num_steps + 1) * grid.dt
    peak = abs(amplitude) * runtime / SQRT_TWO_PI
    for index, frequency in enumerate(frequencies):
        expected = complex(
            np.sum(
                amplitude
                * np.exp(2j * np.pi * (frequency - source_frequency) * sample_times)
                * (grid.dt / SQRT_TWO_PI)
            )
        )
        measured = complex(spectrum[index, 3, 3, 3])
        # float32 accumulation over 400 steps; measured worst case 4.5e-05 of the peak.
        assert abs(measured - expected) < 1e-3 * peak, (
            f"bin {index} (f={frequency}) is {measured}, double-precision reference {expected}"
        )

    recovered = complex(spectrum[2, 3, 3, 3]) * SQRT_TWO_PI / runtime
    assert abs(recovered - amplitude) < 0.01 * abs(amplitude)
    for index in (0, 1, 3):
        assert abs(complex(spectrum[index, 3, 3, 3])) < 1e-3 * peak

    theta = 2 * np.pi * (frequencies[4] - source_frequency) * grid.dt
    sidelobe = abs(np.sin(num_steps * theta / 2) / np.sin(theta / 2)) / num_steps
    measured_sidelobe = abs(complex(spectrum[4, 3, 3, 3])) / peak
    assert 0.02 < sidelobe < 0.06, "the analytic sidelobe itself must be small but non-zero"
    assert measured_sidelobe == pytest.approx(sidelobe, rel=0.01)


def test_flux_spectrum_resolves_two_tones_and_is_not_flat():
    """Two tones of known amplitude ratio must land in their own bins.

    A flat spectrum is the shape a mis-broadcast frequency axis produces, so the
    assertions are on the structure: the 4:1 power ratio between the tones, and
    three empty bins between and around them. Both tone offsets are whole
    multiples of 1/runtime, so the tones are exactly orthogonal over the window
    and each bin's expected power is analytic.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    plane_size = (2.0, 2.0, 0.0)
    tones = ((1.5, 1.0), (0.75, 1.2))
    frequencies = (0.6, 0.8, 1.0, 1.2, 1.4)
    monitor = FluxMonitor(
        grid=grid, frequencies=frequencies, center=(0.0, 0.0, 0.45), size=plane_size
    )
    runtime = _accumulate_tones(monitor, grid, tones, num_steps=400)
    spectrum = monitor.get_flux_spectrum()
    assert spectrum.shape == (5,)
    area = plane_size[0] * plane_size[1]
    for index, (amplitude, frequency) in ((2, tones[0]), (3, tones[1])):
        expected = (amplitude * runtime / SQRT_TWO_PI) ** 2 * area
        assert spectrum[index] == pytest.approx(expected, rel=0.01), f"tone at {frequency}"
    assert spectrum[3] / spectrum[2] == pytest.approx(0.25, rel=0.02)
    for index in (0, 1, 4):
        assert abs(spectrum[index]) < 1e-3 * spectrum[2]
    assert monitor.get_flux(3) == spectrum[3]  # get_flux is the spectrum's own entry.


def test_zero_field_gives_an_exactly_zero_spectrum_with_a_positive_control():
    """The degenerate case: no field must read as no power at every frequency.

    The positive control is the same monitor geometry on a driven field, so a
    monitor that accumulated nothing at all cannot pass this by returning zeros.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    frequencies = (0.8, 1.0, 1.2)
    plane = {"center": (0.0, 0.0, 0.45), "size": (2.0, 2.0, 0.0)}
    quiet_flux = FluxMonitor(grid=grid, frequencies=frequencies, **plane)
    quiet_dft = DFTMonitor(grid=grid, frequencies=frequencies, components=("Ex", "Ey", "Ez"))
    fields = _StubFields(grid, ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    for step in range(1, 41):
        quiet_flux.update(fields, step * grid.dt)
        quiet_dft.update(fields, step * grid.dt)
    assert np.array_equal(quiet_flux.get_flux_spectrum(), np.zeros(3))
    assert float(np.abs(quiet_dft.get_dft_spectrum("Ex")).max()) == 0.0
    for index in range(3):
        assert float(quiet_dft.get_intensity(index).max()) == 0.0

    driven = FluxMonitor(grid=grid, frequencies=frequencies, **plane)
    _accumulate_tones(driven, grid, ((1.0, 1.0),), num_steps=40)
    assert driven.get_flux_spectrum()[1] > 0.0


def test_reset_clears_every_frequency():  # Not just bin 0 — a stale tail reads as signal.
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    frequencies = (0.8, 1.0, 1.2)
    monitor = FluxMonitor(
        grid=grid, frequencies=frequencies, center=(0.0, 0.0, 0.45), size=(2.0, 2.0, 0.0)
    )
    field_monitor = DFTMonitor(grid=grid, frequencies=frequencies, components=("Ex",))
    fields = _StubFields(grid, ("Ex",))
    _drive_monochromatic(field_monitor, fields, grid, 40, 1.0 + 0j, 2 * np.pi)
    _accumulate_tones(monitor, grid, ((1.0, 0.8), (1.0, 1.0), (1.0, 1.2)), num_steps=40)
    assert all(abs(value) > 0.0 for value in monitor.get_flux_spectrum())
    assert float(np.abs(field_monitor.get_dft_spectrum("Ex")).min()) > 0.0
    monitor.reset()
    field_monitor.reset()
    assert np.array_equal(monitor.get_flux_spectrum(), np.zeros(3))
    assert float(np.abs(field_monitor.get_dft_spectrum("Ex")).max()) == 0.0


# --- Real-valued (float32) field mode -------------------------------------------
#
# MEEP's default storage. The monitor contract over real fields is MEEP's own
# (dft.cpp:294-305): the ACCUMULATOR stays complex64 — the DFT of a real signal is
# complex — while the field arrays stay float32 and are never promoted on the way
# through. The absolute anchor is `test_real_mode_spectrum_matches_cpu_meep` below,
# against MEEP run in its real default.


def _drive_real_cosine(monitors, fields, grid, num_steps, amplitude, omega):
    """Step monitors through spatially uniform real cosines, E at t and H at t - dt/2.

    Each component carries its own weight so the Poynting cross terms
    (Ex*Hy - Ey*Hx and friends) cannot cancel by symmetry — a uniform drive
    zeroes every flux exactly and turns any equality built on it vacuous.
    """
    weights = {"Ex": 1.0, "Ey": 0.3, "Ez": 0.7, "Hx": 0.1, "Hy": 0.9, "Hz": 0.5}
    for step in range(1, num_steps + 1):
        time = step * grid.dt
        for name in fields._arrays:
            sample_time = time if name[0] in ("E", "D") else time - 0.5 * grid.dt
            fields.set_uniform(name, amplitude * weights[name] * math.cos(omega * sample_time))
        for monitor in monitors:
            monitor.update(fields, time)


def test_dft_accumulator_stays_complex_over_real_fields_and_promotes_nothing():
    """Accumulating float32 fields must produce the same complex64 DFT, not a real one.

    Two failure shapes are pinned. An accumulator that followed the field dtype
    would store |a real DFT| and quietly discard every phase a flux or intensity
    later needs; a monitor that promoted the FIELD arrays instead would keep
    perfect numbers while costing exactly the memory real mode exists to save.
    The value check is an exact byte comparison against the same drive through
    complex64 storage carrying identical real values: `phase * field` does the
    identical float32 arithmetic on the real plane either way, so the two
    accumulators cannot differ by even a bit.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    frequencies = (0.8, 1.0, 1.2)
    plane = {"center": (0.0, 0.0, 0.45), "size": (2.0, 2.0, 0.0)}
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

    accumulated = {}
    for dtype in (np.float32, np.complex64):
        flux = FluxMonitor(grid=grid, frequencies=frequencies, **plane)
        volume = DFTMonitor(grid=grid, frequencies=frequencies, components=("Ex", "Ez"))
        fields = _StubFields(grid, components, dtype=dtype)
        _drive_real_cosine((flux, volume), fields, grid, num_steps=40,
                           amplitude=1.5, omega=2 * np.pi * 1.0)
        for name in components:
            assert fields._arrays[name].dtype == dtype, (
                f"a monitor promoted the {name} storage from {dtype}"
            )
        assert volume.get_dft_spectrum("Ex").dtype == np.complex64
        assert volume.get_dft_spectrum("Ez").dtype == np.complex64
        for stack in (flux._dft_E, flux._dft_H):
            for name, accumulator in stack.items():
                assert accumulator.dtype == np.complex64, f"flux {name} accumulator left complex64"
        accumulated[dtype] = (
            volume.get_dft_spectrum("Ex").copy(),
            np.asarray(flux.get_flux_spectrum()),
        )

    real_dft, real_flux = accumulated[np.float32]
    complex_dft, complex_flux = accumulated[np.complex64]
    # The DFT of a real cosine is genuinely complex — both planes carry signal — so
    # the byte comparison below is not satisfiable by a pair of empty arrays.
    assert float(np.abs(real_dft.real).max()) > 0.0 and float(np.abs(real_dft.imag).max()) > 0.0
    assert real_dft.tobytes() == complex_dft.tobytes(), (
        "the DFT over float32 fields differs from the same drive over complex64 storage"
    )
    np.testing.assert_array_equal(real_flux, complex_flux)
    assert abs(real_flux[1]) > 0.0, "the on-tone flux bin is empty; the equality above is vacuous"


# --- CPU-MEEP cross-validation ------------------------------------------------

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)

# One broadband run drives both spectrum comparisons. The geometry is shared with
# the engine side through the JSON argument, so the two can never drift apart.
# The DFT volume's faces fall between cell centres (0.3/0.7 in z, +/-0.2 in x and
# y, on a 0.1 grid whose centres are at odd multiples of 0.05), which is the
# ordinary case: both codes round outward to the same cells and the arrays are
# comparable element for element. Faces landing *on* cell centres, and every other
# way a region can meet a boundary, are swept in
# test_driver_integration.py::test_dft_region_registration_matches_cpu_meep_everywhere.
_SPECTRUM_CASE = {
    "resolution": 10,
    "cell": [3.0, 3.0, 5.0],
    "pml_cells": 8,
    "fcen": 1.0,
    "df": 0.8,
    "nfreq": 7,
    "source_center": [0.0, 0.0, -0.05],
    "source_size": [1.0, 1.0, 0.0],
    "plane_size": [1.4, 1.4, 0.0],
    "downstream_z": 1.05,
    "upstream_z": -1.15,
    "dft_center": [0.0, 0.0, 0.5],
    "dft_size": [0.4, 0.4, 0.4],
    "until": 20.0,
}

_SPECTRUM_ORACLE_SCRIPT = '''"""CPU-MEEP reference spectra for the multi-frequency monitor tests."""
import json
import sys

import numpy as np
import meep as mp

case = json.loads(sys.argv[1])
output_path = sys.argv[2]

# "complex": False selects MEEP's DEFAULT real-field storage — the reference for the
# engine's own real mode. k_point stays the zero vector: it makes every boundary
# periodic without forcing complex fields (fields.cpp:146 aborts only for k != 0).
simulation = mp.Simulation(
    cell_size=mp.Vector3(*case["cell"]),
    resolution=case["resolution"],
    boundary_layers=[mp.PML(case["pml_cells"] / case["resolution"])],
    sources=[
        mp.Source(
            mp.GaussianSource(frequency=case["fcen"], fwidth=case["df"]),
            component=mp.Ez,
            center=mp.Vector3(*case["source_center"]),
            size=mp.Vector3(*case["source_size"]),
        )
    ],
    force_complex_fields=bool(case.get("complex", True)),
    k_point=mp.Vector3(0, 0, 0),
)
# decimation_factor=1 turns off MEEP's automatic DFT decimation: the default
# subsamples the accumulation from the source bandwidth, which is a different
# (aliasing-prone) quantity from the every-step transform under test here.
spec = (case["fcen"], case["df"], case["nfreq"])
downstream = simulation.add_flux(
    *spec,
    mp.FluxRegion(
        center=mp.Vector3(0, 0, case["downstream_z"]),
        size=mp.Vector3(*case["plane_size"]),
        direction=mp.Z,
    ),
    decimation_factor=case.get("decimation_factor", 1),
)
upstream = simulation.add_flux(
    *spec,
    mp.FluxRegion(
        center=mp.Vector3(0, 0, case["upstream_z"]),
        size=mp.Vector3(*case["plane_size"]),
        direction=mp.Z,
    ),
    decimation_factor=case.get("decimation_factor", 1),
)
dft_fields = simulation.add_dft_fields(
    [mp.Ez],
    *spec,
    center=mp.Vector3(*case["dft_center"]),
    size=mp.Vector3(*case["dft_size"]),
    decimation_factor=case.get("decimation_factor", 1),
)

simulation.run(until=case["until"])
assert simulation.fields.is_real != bool(case.get("complex", True)), (
    "the oracle is not in the storage mode it claims"
)

np.savez(
    output_path,
    freqs=np.asarray(mp.get_flux_freqs(downstream)),
    flux_downstream=np.asarray(mp.get_fluxes(downstream)),
    flux_upstream=np.asarray(mp.get_fluxes(upstream)),
    dft_ez=np.stack(
        [
            np.asarray(simulation.get_dft_array(dft_fields, mp.Ez, index))
            for index in range(case["nfreq"])
        ]
    ),
)
'''


def _run_spectrum_oracle(tmp_path_factory, case, label):  # One CPU-MEEP subprocess run.
    import json

    tmp_path = tmp_path_factory.mktemp(label)
    script_path = tmp_path / "meep_spectrum_oracle.py"
    script_path.write_text(_SPECTRUM_ORACLE_SCRIPT, encoding="utf-8")
    output_path = tmp_path / "spectra.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(case), str(output_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP spectrum oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


@pytest.fixture(scope="module")
def spectrum_oracle(tmp_path_factory):  # One CPU-MEEP run, shared by both cross-validations.
    return _run_spectrum_oracle(tmp_path_factory, _SPECTRUM_CASE, "dft_spectrum_oracle")


@pytest.fixture(scope="module")
def engine_spectrum():  # The same physics through this engine's multi-frequency monitors.
    return _run_engine_spectrum(complex_fields=True)


def _run_engine_spectrum(
    complex_fields,
    *,
    decimation_factor=1,
):  # One engine run of the spectrum case, either storage mode.
    from .driver import FdtdDriver

    case = _SPECTRUM_CASE
    frequencies = meep_frequency_span(case["fcen"], case["df"], case["nfreq"])
    driver = FdtdDriver(
        cell_size=tuple(case["cell"]), resolution=case["resolution"],
        force_complex_fields=complex_fields,
    )
    driver.setup_pml(case["pml_cells"])
    driver.add_source(
        {
            "component": "Ez",
            "frequency": case["fcen"],
            "center": tuple(case["source_center"]),
            "size": tuple(case["source_size"]),
            "source_type": "gaussian",
            "fwidth": case["df"],
        }
    )
    plane = {"grid": driver.grid, "frequencies": frequencies, "size": tuple(case["plane_size"])}
    downstream = FluxMonitor(
        center=(0.0, 0.0, case["downstream_z"]),
        decimation_factor=decimation_factor,
        **plane,
    )
    upstream = FluxMonitor(
        center=(0.0, 0.0, case["upstream_z"]),
        decimation_factor=decimation_factor,
        **plane,
    )
    fields_monitor = DFTMonitor(
        grid=driver.grid,
        frequencies=frequencies,
        components=("Ez",),
        decimation_factor=decimation_factor,
    )
    fields_monitor.set_region_from_volume(
        center=tuple(case["dft_center"]), size=tuple(case["dft_size"])
    )
    monitors = (downstream, upstream, fields_monitor)
    # The monitors are stepped here rather than registered with the driver so this
    # case pins the monitor contract itself; the timing is the driver's own —
    # update() at the post-increment simulation time.
    steps = int(round(case["until"] / driver.dt))
    assert abs(steps * driver.dt - case["until"]) < 1e-9, "the run must end on a whole timestep"
    for _ in range(steps):
        driver.step()
        for monitor in monitors:
            monitor.update(driver.fields, driver.time)
    return {
        "frequencies": np.asarray(frequencies),
        "flux_downstream": downstream.get_flux_spectrum(),
        "flux_upstream": upstream.get_flux_spectrum(),
        "dft_ez": np.asarray(fields_monitor.get_dft_spectrum("Ez")),
        # What the run's storage actually was, so the real-mode case can assert the
        # monitors were accumulating from float32 arrays rather than from a silent
        # promotion (which would produce perfect numbers at double the memory).
        "field_dtype": driver.fields.Dz.dtype,
        "stored_e_dtype": driver.fields.Ez.dtype,
    }


def _run_driver_spectrum(decimation_factor):
    """Run the broadband case through the high-level monitor registration API."""
    case = _SPECTRUM_CASE
    driver = FdtdDriver(
        cell_size=tuple(case["cell"]),
        resolution=case["resolution"],
        force_complex_fields=True,
    )
    driver.setup_pml(case["pml_cells"])
    driver.add_source(
        {
            "component": "Ez",
            "frequency": case["fcen"],
            "center": tuple(case["source_center"]),
            "size": tuple(case["source_size"]),
            "source_type": "gaussian",
            "fwidth": case["df"],
        }
    )
    spectrum = {
        "fcen": case["fcen"],
        "df": case["df"],
        "nfreq": case["nfreq"],
        "decimation_factor": decimation_factor,
    }
    downstream = driver.add_flux_monitor(
        center=(0.0, 0.0, case["downstream_z"]),
        size=tuple(case["plane_size"]),
        **spectrum,
    )
    upstream = driver.add_flux_monitor(
        center=(0.0, 0.0, case["upstream_z"]),
        size=tuple(case["plane_size"]),
        **spectrum,
    )
    dft_fields = driver.add_dft_monitor(
        components=("Ez",),
        center=tuple(case["dft_center"]),
        size=tuple(case["dft_size"]),
        **spectrum,
    )
    driver.run(until=case["until"])
    return {
        "frequencies": np.asarray(dft_fields.frequencies),
        "flux_downstream": downstream.get_flux_spectrum(),
        "flux_upstream": upstream.get_flux_spectrum(),
        "dft_ez": np.asarray(dft_fields.get_dft_spectrum("Ez")),
        "factors": (
            downstream.decimation_factor,
            upstream.decimation_factor,
            dft_fields.decimation_factor,
        ),
    }


@pytest.fixture(scope="module")
def spectrum_oracle_decimated(tmp_path_factory):
    return _run_spectrum_oracle(
        tmp_path_factory,
        dict(_SPECTRUM_CASE, decimation_factor=4),
        "dft_spectrum_oracle_decimated",
    )


@pytest.fixture(scope="module")
def engine_spectrum_decimated():
    return _run_engine_spectrum(complex_fields=True, decimation_factor=4)


@pytest.fixture(scope="module")
def spectrum_oracle_automatic(tmp_path_factory):
    return _run_spectrum_oracle(
        tmp_path_factory,
        dict(_SPECTRUM_CASE, decimation_factor=0),
        "dft_spectrum_oracle_automatic",
    )


@pytest.fixture(scope="module")
def engine_spectrum_automatic():
    return _run_driver_spectrum(decimation_factor=0)


@requires_meep
@skip_without_meep
def test_decimated_dft_and_flux_spectra_match_cpu_meep(
    spectrum_oracle_decimated,
    engine_spectrum_decimated,
):
    """Explicit factor-four sampling matches MEEP for fields and integrated flux."""
    reference = spectrum_oracle_decimated
    candidate = engine_spectrum_decimated
    for key in ("flux_downstream", "flux_upstream", "dft_ez"):
        assert candidate[key].shape == reference[key].shape
        denominator = np.linalg.norm(reference[key].ravel())
        assert denominator > 0.0, f"{key} CPU-MEEP oracle is degenerate"
        relative_l2 = float(
            np.linalg.norm((candidate[key] - reference[key]).ravel()) / denominator
        )
        assert relative_l2 < 5e-5, f"{key} relative L2 {relative_l2:.3e}"


@requires_meep
@skip_without_meep
def test_automatic_dft_and_flux_decimation_matches_cpu_meep(
    spectrum_oracle_automatic,
    engine_spectrum_automatic,
):
    """MEEP's default source/monitor bandwidth rule matches in scheduling and values."""
    reference = spectrum_oracle_automatic
    candidate = engine_spectrum_automatic
    # dt=.05; both the Gaussian source and monitor band end at 1.4.
    assert candidate["factors"] == (7, 7, 7)
    np.testing.assert_allclose(candidate["frequencies"], reference["freqs"], rtol=1e-12)
    for key in ("flux_downstream", "flux_upstream", "dft_ez"):
        assert candidate[key].shape == reference[key].shape
        denominator = np.linalg.norm(reference[key].ravel())
        assert denominator > 0.0, f"{key} CPU-MEEP oracle is degenerate"
        relative_l2 = float(
            np.linalg.norm((candidate[key] - reference[key]).ravel()) / denominator
        )
        assert relative_l2 < 5e-5, f"{key} relative L2 {relative_l2:.3e}"


@requires_meep
@skip_without_meep
def test_flux_spectrum_matches_cpu_meep(spectrum_oracle, engine_spectrum):
    """A broadband transmission spectrum, frequency by frequency, against MEEP.

    A Gaussian sheet source in vacuum radiates into a PML-terminated cell, with
    one flux plane each side of it at equal distance. Both codes accumulate seven
    frequencies spanning 0.6 to 1.4 in one pass, and MEEP's own `get_fluxes` is
    the oracle. Measured against the 5% cross-validation bar: 1.6e-07 relative L2
    downstream, 3.0e-07 upstream; worst single frequency 1.1e-06 downstream (at
    1.4) and 1.8e-06 upstream (at 0.6), the two band edges, where the pulse
    carries ~1e-5 of its peak power. (Those digits were the MEEP-1.29 oracle's;
    re-baselined 2026-08-06 on pristine 1.33.0 single the same comparison reads
    1.2e-07 / 5.4e-08 — oracle-side, since the engine at HEAD and at the working
    tree measure identically, and the decimated twin of this case is
    digit-stable.) The band itself spans five orders of
    magnitude, so a monitor whose bins all carried one frequency's phase could
    not sit inside any of these numbers.
    """
    assert engine_spectrum["frequencies"] == pytest.approx(spectrum_oracle["freqs"]), (
        "meep_frequency_span disagrees with MEEP's own (fcen, df, nfreq) expansion"
    )
    for side in ("downstream", "upstream"):
        reference = spectrum_oracle[f"flux_{side}"]
        candidate = engine_spectrum[f"flux_{side}"]
        assert candidate.shape == reference.shape == (_SPECTRUM_CASE["nfreq"],)
        assert np.max(np.abs(reference)) > 0.0, "the MEEP oracle carries no power"
        assert np.max(np.abs(candidate)) > 0.0, "the engine carries no power"
        for index, frequency in enumerate(spectrum_oracle["freqs"]):
            relative = abs(candidate[index] - reference[index]) / abs(reference[index])
            assert relative < 0.05, (
                f"{side} flux at f={frequency:.4f}: engine {candidate[index]:.6e} vs MEEP "
                f"{reference[index]:.6e} ({relative:.2e} relative)"
            )
        relative_l2 = float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
        assert relative_l2 < 0.05, f"{side} flux spectrum relative L2 {relative_l2:.3e}"
    # The spectrum has real structure — the source's own Gaussian envelope — and
    # the two planes see equal and opposite power, since they sit the same
    # distance either side of the sheet.
    downstream = engine_spectrum["flux_downstream"]
    upstream = engine_spectrum["flux_upstream"]
    assert downstream.max() / abs(downstream).min() > 1e4, "the band is flat; nothing was resolved"
    assert np.all(upstream < 0.0), "the upstream plane must report power flowing towards -z"
    assert np.max(np.abs(upstream + downstream) / downstream.max()) < 0.01


@requires_meep
@skip_without_meep
def test_dft_field_spectrum_matches_cpu_meep(spectrum_oracle, engine_spectrum):
    """The DFT field volume at every frequency, against MEEP's `get_dft_array`.

    Complex relative L2 per frequency, so a bin with the right magnitude and the
    wrong phase fails. Measured 2.2e-07 over the whole stack; worst single
    frequency 6.5e-06, at the 0.6 edge of the band where the field is 1/180 of
    its peak.
    """
    reference = spectrum_oracle["dft_ez"]
    candidate = engine_spectrum["dft_ez"]
    assert candidate.shape == reference.shape, (
        f"engine DFT volume {candidate.shape} does not match MEEP's {reference.shape}; the "
        f"monitor volume's faces must fall between cell centres for the two to register"
    )
    assert candidate.shape[0] == _SPECTRUM_CASE["nfreq"]
    magnitudes = np.array([float(np.linalg.norm(plane)) for plane in reference])
    assert magnitudes.min() > 0.0, "a MEEP reference frequency is empty; the comparison is vacuous"
    for index, frequency in enumerate(spectrum_oracle["freqs"]):
        relative = float(
            np.linalg.norm(candidate[index] - reference[index]) / magnitudes[index]
        )
        assert relative < 0.05, (
            f"Ez DFT at f={frequency:.4f}: complex relative L2 {relative:.2e} exceeds the 5% bar"
        )
    total = float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
    assert total < 0.05, f"Ez DFT spectrum relative L2 {total:.3e}"
    # Each frequency must be its own field, not a copy of the strongest one.
    assert magnitudes.max() / magnitudes.min() > 100.0


@pytest.fixture(scope="module")
def spectrum_oracle_real(tmp_path_factory):  # The same case with MEEP in its default real storage.
    return _run_spectrum_oracle(
        tmp_path_factory, dict(_SPECTRUM_CASE, complex=False), "dft_spectrum_oracle_real"
    )


@pytest.fixture(scope="module")
def engine_spectrum_real():  # The same physics through this engine's float32 mode.
    return _run_engine_spectrum(complex_fields=False)


@requires_meep
@skip_without_meep
def test_real_mode_spectrum_matches_cpu_meep(spectrum_oracle_real, engine_spectrum_real,
                                             spectrum_oracle):
    """The monitor chain over float32 fields, against MEEP run in its real default.

    Same broadband PML case as the complex validation — flux planes both sides of a
    Gaussian sheet plus a DFT volume, seven frequencies in one pass — with BOTH
    codes in real storage. This is the anchor the real-vs-complex byte equalities
    cannot provide: a real run's DFT is a different physical measurement from the
    complex run's. The real field is the analytic signal's real part, so its
    positive-frequency DFT is HALF the complex one (plus the negative-frequency
    leakage that half discards), and a flux is a QUARTER — measured 0.4999999 /
    0.7500000 between MEEP's own two modes on this very case. Agreement here
    therefore says the engine's real mode reproduces MEEP's real mode, not merely
    its own complex run wearing a cast. Measured floors: flux relative L2 5.9e-08
    downstream / 3.6e-08 upstream, DFT stack 2.7e-07; held at ~5x.
    """
    assert engine_spectrum_real["field_dtype"] == np.float32, (
        "the real-mode engine run promoted its primary storage"
    )
    assert engine_spectrum_real["stored_e_dtype"] == np.float32, (
        "the real-mode engine run promoted its stored E"
    )
    assert engine_spectrum_real["frequencies"] == pytest.approx(spectrum_oracle_real["freqs"])
    for side in ("downstream", "upstream"):
        reference = spectrum_oracle_real[f"flux_{side}"]
        candidate = engine_spectrum_real[f"flux_{side}"]
        assert np.max(np.abs(reference)) > 0.0, "the real MEEP oracle carries no power"
        relative_l2 = float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
        assert relative_l2 < 0.05, f"real {side} flux relative L2 {relative_l2:.3e} exceeds 5%"
        assert relative_l2 < 3e-7, (
            f"real {side} flux relative L2 {relative_l2:.3e} regressed from the 3e-07 held bound"
        )
    reference = spectrum_oracle_real["dft_ez"]
    candidate = engine_spectrum_real["dft_ez"]
    assert candidate.shape == reference.shape
    stack_l2 = float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
    assert stack_l2 < 1.5e-6, f"real Ez DFT stack relative L2 {stack_l2:.3e} regressed from 1.5e-06"
    # The DFT of a real run is genuinely complex — the accumulator, not the storage.
    assert candidate.dtype == np.complex64
    assert float(np.abs(candidate.imag).max()) > 0.0

    # Control: the real oracle must actually BE the other storage mode. The factor-4
    # power convention above puts MEEP's two references 0.75 apart on this case, so
    # if the two oracles agree to the held bounds the "real" one was silently built
    # complex (a dropped case flag, say) and everything above validated nothing
    # real-specific.
    upstream_gap = float(
        np.linalg.norm(spectrum_oracle_real["flux_upstream"] - spectrum_oracle["flux_upstream"])
        / np.linalg.norm(spectrum_oracle["flux_upstream"])
    )
    assert upstream_gap > 0.5, (
        f"MEEP's real and complex upstream fluxes differ by only {upstream_gap:.3e}, not the "
        f"factor-4 power convention; the 'real' oracle did not run real storage"
    )


def _accumulate_z_profile_plane_wave(monitor, grid, profile, frequency, num_steps):
    """Drive a flux monitor with a +z wave whose Ex carries an arbitrary per-cell profile.

    `Hy` is uniform, so the flux integrand is `Re(Ex conj(Hy))` and the monitor's
    answer is the DFT magnitude squared times the Yee-averaged `Ex` at the plane.
    Unlike `_accumulate_z_graded_plane_wave` the profile is not linear, so the
    value the average produces at the last row of cell centres cannot be guessed
    from the interior — it depends on what the wrapped neighbour actually holds.
    """
    omega = 2 * np.pi * frequency
    fields = _StubFields(grid, ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    volume = np.broadcast_to(np.asarray(profile, dtype=np.complex128)[None, None, :], grid.shape)
    for step in range(1, num_steps + 1):
        time = step * grid.dt
        fields.set_array("Ex", volume * np.exp(-1j * omega * time))
        fields.set_uniform("Hy", np.exp(-1j * omega * (time - 0.5 * grid.dt)))
        monitor.update(fields, time)
    return num_steps * grid.dt


def test_flux_plane_on_the_last_row_of_centres_averages_across_the_periodic_face():
    """`Ex` at the last row of cell centres must average with the wrapped cell, not itself.

    `Ex` sits at integer positions in z, so reaching a cell centre costs a
    `0.5*(Ex[k] + Ex[k+1])` average (MEEP fields.cpp yee2cent). At the last stored
    row `k+1` is one lattice vector away — stored cell 0 — and MEEP's chunk reads
    it through the periodic connection. This engine used to skip the average
    entirely there and report the raw Yee value, half a cell off, in silence:
    9.1e-02 against CPU MEEP for exactly this plane (see
    test_driver_integration.py::test_flux_plane_reaching_past_a_periodic_face_matches_cpu_meep).

    The expectation is built here from the profile array itself, so it is an
    independent derivation rather than the monitor's own formula re-run. The
    profile is deliberately discontinuous across the wrap — 3.0 at cell 0 against
    1.0 at cell 39 — which makes the two candidate answers differ by a factor of
    two, far outside any tolerance.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    frequency, num_steps = 1.0, 200
    plane_size = (2.0, 2.0, 0.0)
    profile = np.ones(grid.nz, dtype=np.float64)
    profile[0] = 3.0  # The wrapped neighbour of the last row, and nothing else.

    monitor = FluxMonitor(grid=grid, frequency=frequency, center=(0.0, 0.0, 1.95), size=plane_size)
    assert monitor._wrapped == (False, False, True), "The last row of centres must wrap in z."
    runtime = _accumulate_z_profile_plane_wave(monitor, grid, profile, frequency, num_steps)
    dft_magnitude = runtime / SQRT_TWO_PI
    area = plane_size[0] * plane_size[1]
    wrapped_average = 0.5 * (profile[grid.nz - 1] + profile[0])
    assert monitor.get_flux() == pytest.approx(dft_magnitude**2 * area * wrapped_average, rel=2e-6)

    # The control: a NON-wrapping axis terminates on a PEC wall, which MEEP's
    # step_boundaries holds at exactly zero, so the last row's average is
    # 0.5*(Ex[nz-1] + 0) — a different number by exactly the factor the profile's
    # discontinuity sets. (Returning the RAW un-averaged value there instead is
    # what this engine used to do, and it is the measured 2.02x flux defect the
    # wall-zero plane repaired — see _sliced_component and
    # the design notes (absorber-placement-evidence) §2.3 row 29.)
    clipped = FluxMonitor(grid=grid, frequency=frequency, center=(0.0, 0.0, 1.95),
                          size=plane_size, periodic=(False, False, False))
    assert clipped._wrapped == (False, False, False)
    _accumulate_z_profile_plane_wave(clipped, grid, profile, frequency, num_steps)
    wall_average = 0.5 * (profile[grid.nz - 1] + 0.0)
    assert clipped.get_flux() == pytest.approx(dft_magnitude**2 * area * wall_average, rel=2e-6)
    assert abs(clipped.get_flux() - monitor.get_flux()) > 0.4 * abs(monitor.get_flux()), (
        "The wrapped and walled answers must be far apart, or this test proves nothing."
    )

    # An interior plane is untouched by any of this: same profile, same run, and the
    # contiguous path, so the wrap support cannot have moved the numbers it was not
    # meant to reach.
    interior = FluxMonitor(grid=grid, frequency=frequency, center=(0.0, 0.0, 0.45), size=plane_size)
    assert interior._wrapped == (False, False, False)
    _accumulate_z_profile_plane_wave(interior, grid, profile, frequency, num_steps)
    assert interior.get_flux() == pytest.approx(dft_magnitude**2 * area * 1.0, rel=2e-6)


def test_flux_wrap_carries_the_bloch_phase_of_the_lattice_vector_it_crossed():
    """A cell sampled one lattice vector up must arrive multiplied by `bloch_phase`.

    MEEP loop_in_chunks.cpp multiplies each lattice-shifted copy by
    `pow(eikna[d], ishift)`; the same factor appears inside the Yee average that
    straddles the face. Flux is `Re(E conj(H))`, so a phase applied to a whole
    image cancels — it is exactly this average, where only one of the two terms
    carries the factor, that a dropped phase changes.

    Run on the real `Grid` rather than the stub, because the phase is `Grid`'s
    definition of the sign convention and the point is that the monitor uses that
    one and not a second copy of it.
    """
    k_z = 0.3
    grid = Grid(resolution=10.0, cell_size=(1.0, 1.0, 2.0), k_point=(0.0, 0.0, k_z))
    phase = grid.bloch_phase(2)
    assert phase is not None and phase == pytest.approx(cmath.exp(2j * np.pi * k_z * grid.Lz))

    frequency, num_steps = 1.0, 200
    plane_size = (0.6, 0.6, 0.0)
    # The wrapped cell's value must be COMPLEX. Flux takes Re(Ex conj(Hy)) and Hy is
    # real here, so with a real profile `phase * f[0]` and `conj(phase) * f[0]` have
    # the same real part and a conjugated wrap phase is invisible — a blind spot the
    # mutation run found, where only the CPU-MEEP comparison caught it.
    profile = np.ones(grid.nz, dtype=np.complex128)
    profile[0] = 3.0 + 2.0j
    last_centre = float(np.asarray(grid.z)[-1])

    monitor = FluxMonitor(grid=grid, frequency=frequency, center=(0.0, 0.0, last_centre),
                          size=plane_size)
    assert monitor._wrapped[2]
    runtime = _accumulate_z_profile_plane_wave(monitor, grid, profile, frequency, num_steps)
    dft_magnitude = runtime / SQRT_TWO_PI
    area = plane_size[0] * plane_size[1]
    # Hy is real and uniform, so the flux picks up the REAL part of the averaged Ex.
    expected_average = 0.5 * (profile[grid.nz - 1] + phase * profile[0])
    assert monitor.get_flux() == pytest.approx(
        dft_magnitude**2 * area * expected_average.real, rel=1e-5
    )

    # Controls: dropping the phase, or taking its conjugate, are both measurably
    # different answers here — 0.3 is not at the zone edge, where the phase is its
    # own conjugate and neither mistake would show.
    unphased = dft_magnitude**2 * area * 0.5 * (profile[grid.nz - 1] + profile[0])
    conjugated = dft_magnitude**2 * area * 0.5 * (profile[grid.nz - 1] + phase.conjugate() * profile[0])
    for label, wrong in (("unphased", unphased), ("conjugated", conjugated)):
        assert abs(monitor.get_flux() - wrong) > 0.05 * abs(monitor.get_flux()), (
            f"The {label} answer is within 5% of the measured flux; this test cannot see it."
        )


def test_flux_plane_beyond_the_adjacent_lattice_images_is_refused():
    """One image on each side is served; anything further out raises.

    The positive controls sit either side of the line: a plane on the far face
    samples the first image and is answered, and a plane several cells outside the
    cell is refused rather than reported as the power of an image nobody asked
    about — the same silent-plausible-answer failure as reporting zero.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    on_the_face = FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, 2.0), size=(2.0, 2.0, 0.0))
    assert on_the_face._wrapped[2] and on_the_face._region[4:6] == (39, 41)
    one_image_up = FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, 2.05), size=(2.0, 2.0, 0.0))
    assert one_image_up._region[4:6] == (40, 41)
    for outside in (9.0, -9.0, 6.05):
        with pytest.raises(ValueError, match="does not intersect"):
            FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, outside), size=(2.0, 2.0, 0.0))
    # On a non-wrapping axis the old clipping refusal still applies, and says so.
    with pytest.raises(ValueError, match="does not intersect"):
        FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, 9.0), size=(2.0, 2.0, 0.0),
                    periodic=(False, False, False))


def test_flux_wrap_is_refused_on_a_mirror_axis_and_kept_off_by_default_there():
    """A mirrored axis never wraps: MEEP reaches its far half by symmetry, not by lattice shift."""
    grid = _StubGrid(10, (4.0, 4.0, 4.0), symmetry=("X",))
    monitor = FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, 1.95), size=(2.0, 2.0, 0.0))
    assert monitor._wrapped[0] is False, "A mirror-folded axis must not be continued periodically."
    assert monitor._wrapped[2] is True, "...while the unmirrored z axis still wraps."
    # The below-plane half of the x request rides the reflected gather, not a wrap.
    assert monitor._region[:2] == (-10, 12)


# --- DFT-region registration: the axis ladder, shared with the flux plane -----


def _independent_ladder(where_min, where_max, resolution, n_full):
    """MEEP's floor/ceil bracket of a requested extent, transcribed from the source.

    loop_in_chunks.cpp `vec2diel_floor` / `vec2diel_ceil`: `1 + 2*floor(x*a - 0.5)`
    and `1 + 2*ceil(x*a - 0.5)` in MEEP's doubled coordinates, where cell centres
    are the odd integers and cell `j` of a centred axis sits at `-n_full + 2j + 1`.
    Written out here rather than imported so the expectations below are a second
    derivation and not `dft._axis_ladder` compared with itself.
    """
    low = 1 + 2 * math.floor(where_min * resolution - 0.5)
    high = 1 + 2 * math.ceil(where_max * resolution - 0.5)
    return (low + n_full - 1) // 2, (high - low) // 2 + 1


def test_dft_and_flux_register_the_same_volume_on_the_same_cells():
    """One volume, two monitor kinds, one set of cells — the unification itself.

    `DFTMonitor` used to map a volume to indices with its own `round()`-based
    formula while `FluxMonitor` used MEEP's ladder, so the two could disagree about
    the same request. They now share `_register_volume`, and this sweeps volumes
    whose faces fall on cell centres, between them, and past the cell, at both
    parities, asserting the two agree with each other *and* with the ladder
    transcribed above.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    volumes = [
        ((0.0, 0.0, 0.0), (2.0, 2.0, 0.0)),
        ((0.0, 0.0, 0.0), (2.1, 1.9, 0.0)),   # Both faces of each axis on cell centres.
        ((0.0, 0.0, 0.25), (1.9, 2.1, 0.0)),
        ((0.15, -0.35, 0.45), (0.6, 0.6, 0.0)),
        ((0.0, 0.0, 1.95), (2.0, 2.0, 0.0)),  # Last row of centres.
        ((0.0, 0.0, 0.0), (4.0, 4.0, 0.0)),   # The full periodic cross-section.
    ]
    for centre, size in volumes:
        monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",),
                             periodic=(True, True, True))
        monitor.set_region_from_volume(center=centre, size=size)
        plane = FluxMonitor(grid=grid, frequency=1.0, center=centre, size=size)
        assert monitor.region == plane._region, (
            f"volume {centre}/{size}: DFT registered {monitor.region}, flux {plane._region}"
        )
        for axis in range(3):
            first, count = _independent_ladder(
                centre[axis] - size[axis] / 2, centre[axis] + size[axis] / 2, grid.resolution,
                (grid.nx_full, grid.ny_full, grid.nz_full)[axis],
            )
            if not monitor._wrapped[axis]:  # A clipped axis keeps only its stored part.
                first, count = max(0, first), min(count, (grid.nx, grid.ny, grid.nz)[axis])
            assert monitor.region[2 * axis:2 * axis + 2] == (first, first + count), (
                f"volume {centre}/{size}, axis {axis}: {monitor.region} vs ladder "
                f"{(first, first + count)}"
            )


def test_dft_region_width_does_not_depend_on_the_parity_of_a_centre_aligned_face():
    """Two faces on rows of cell centres must cost the same cell count either way.

    MEEP's outward rounding reaches a face sitting exactly on a row of cell centres
    and stops there, so a region of size `s` centred on the origin spans `s*a + 1`
    cells when both faces are centre-aligned. The old half-cell-margin-plus-`round()`
    mapping instead resolved that tie by Python's banker's rounding, which depends
    on whether the cell index is even or odd: at resolution 10 in a 4-unit cell,
    size 2.1 came out 23 cells wide against MEEP's 22, while size 1.9 came out 20 —
    MEEP's own answer — from the identical geometry one row over. A width that
    tracks index parity is not one a caller can reason about.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    for size, expected_cells in ((2.1, 22), (1.9, 20), (2.5, 26), (0.9, 10)):
        monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
        monitor.set_region_from_volume(center=(0.0, 0.0, 0.0), size=(size, size, 0.0))
        for axis in (0, 1):
            width = monitor.region[2 * axis + 1] - monitor.region[2 * axis]
            assert width == expected_cells, (
                f"size {size} on axis {axis}: {width} cells, MEEP's ladder gives {expected_cells}"
            )
        # The faces really are on rows of cell centres, or this pins nothing.
        assert abs((size / 2) * grid.resolution % 1.0 - 0.5) < 1e-12


def test_zero_thickness_dft_axis_spans_the_one_or_two_cells_that_bracket_it():
    """A flat axis is 1 cell on a row of centres and 2 between rows — never 3.

    MEEP's zero-thickness rule (loop_in_chunks.cpp case 4): `vec2diel_floor` and
    `vec2diel_ceil` of the same coordinate, which is one cell when the coordinate is
    a row of cell centres and the two rows either side otherwise. The old mapping
    padded the request by half a cell on *each* side before rounding, so a flat axis
    asked for at a row of centres registered **three** cells — a "zero-thickness"
    monitor two cells thick, the same defect class as the flux plane's ten-times-too-
    large normal extent — and one asked for at the cell face was clamped to a single
    row half a cell away.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    for coordinate, expected in ((0.05, 1), (-1.95, 1), (0.0, 2), (0.03, 2), (0.5, 2)):
        monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
        monitor.set_region_from_volume(center=(0.0, 0.0, coordinate), size=(1.0, 1.0, 0.0))
        cells = monitor.region[5] - monitor.region[4]
        assert cells == expected, f"z = {coordinate}: {cells} cells, expected {expected}"
        z = monitor.get_coordinates()[2]
        assert len(z) == expected
        # The cells actually bracket the request rather than sitting to one side.
        assert z[0] - 1e-6 <= coordinate <= z[-1] + 1e-6, f"z = {coordinate} outside {z}"

    # On the cell face the bracketing pair is the last stored row and the first row
    # of the next lattice image; clamping to the stored one alone reported a plane
    # half a cell away and did it in silence (4.9e-01 against CPU MEEP).
    on_face = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",), periodic=(True, True, True))
    on_face.set_region_from_volume(center=(0.0, 0.0, 2.0), size=(1.0, 1.0, 0.0))
    assert on_face.region[4:6] == (39, 41) and on_face._wrapped[2]


def test_dft_region_continued_past_a_periodic_face_carries_the_bloch_phase():
    """Cells past the face are the wrapped ones, multiplied by `grid.bloch_phase`.

    A profile that is discontinuous across the wrap (2.0 at stored cell 0 against
    1.0 everywhere else) makes the continued cells distinguishable from any clamped
    or repeated alternative, and the expectation is built from the profile and
    `Grid.bloch_phase` rather than from the monitor's own gather.

    `Ez` has Yee shift 1 in z, so its cell-centre value needs no average there and
    each region cell is exactly one stored cell — which is what makes the wrapped
    entries readable one at a time.
    """
    k_z = 0.35
    grid = Grid(resolution=10.0, cell_size=(1.0, 1.0, 2.0), k_point=(0.0, 0.0, k_z))
    phase = grid.bloch_phase(2)
    assert phase is not None and abs(phase - 1.0) > 0.5, "the wrap factor must be visible"

    profile = np.ones(grid.nz, dtype=np.complex128)
    profile[0] = 2.0
    profile[1] = 3.0
    fields = _StubFields(grid, ("Ez",))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",), periodic=(True, True, True))
    # Upper face at z = 1.02, a fifth of a cell past the +z cell face at z = 1.0.
    monitor.set_region_from_volume(center=(0.0, 0.0, 0.87), size=(0.4, 0.4, 0.3))
    first, count = _independent_ladder(0.72, 1.02, grid.resolution, grid.nz)
    assert monitor.region[4:6] == (first, first + count) and first + count > grid.nz

    steps = 100
    for step in range(1, steps + 1):
        time = step * grid.dt
        fields.set_array(
            "Ez", np.broadcast_to(profile[None, None, :], grid.shape) * np.exp(-2j * np.pi * time)
        )
        monitor.update(fields, time)
    recovered = np.asarray(monitor.get_dft("Ez")) * SQRT_TWO_PI / (steps * grid.dt)

    expected = np.array(
        [profile[cell % grid.nz] * phase ** (cell // grid.nz)
         for cell in range(first, first + count)],
        dtype=np.complex128,
    )
    assert np.allclose(recovered[0, 0, :], expected, rtol=2e-2, atol=2e-3), (
        f"{recovered[0, 0, :]} vs {expected}"
    )
    # The wrapped tail must be the *phased* image and not a plain copy: with
    # profile[0] = 2 and profile[1] = 3, dropping the factor is a large, visible
    # change, and repeating the last stored cell a different one again.
    assert abs(recovered[0, 0, -1] - profile[1]) > 0.5, "the wrap dropped its Bloch factor"
    assert abs(recovered[0, 0, -1] - profile[grid.nz - 1]) > 0.5, "the far cell was repeated"


def test_dft_region_coordinates_follow_the_cells_across_a_face():
    """`get_coordinates` must describe the cells the region actually holds.

    A region continued into the next lattice image holds indices outside `[0, n)`,
    where slicing `grid.z` silently returns fewer coordinates than there are cells —
    or, for a negative start, the coordinates of the *far* end of the axis. Both are
    a caller plotting one thing against the label of another.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",), periodic=(True, True, True))
    monitor.set_region_from_volume(center=(0.0, 0.0, 1.95), size=(1.0, 1.0, 0.3))
    x, y, z = monitor.get_coordinates()
    assert z.shape[0] == monitor.region[5] - monitor.region[4]
    assert monitor.region[5] > grid.nz, "this region must reach past the face to prove anything"
    assert np.all(np.diff(z) > 0), f"coordinates fold back at the face: {z}"
    assert z[-1] > grid.z[-1], "the continued cells must sit beyond the last stored row"
    # Inside the axis the built coordinates and the plain slice agree.
    interior = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    interior.set_region_from_volume(center=(0.0, 0.0, 0.45), size=(1.0, 1.0, 0.3))
    x0, x1, y0, y1, z0, z1 = interior.region
    for built, sliced in zip(interior.get_coordinates(), (grid.x[x0:x1], grid.y[y0:y1],
                                                          grid.z[z0:z1])):
        assert np.allclose(built, sliced, atol=1e-6)


def test_dft_region_that_resolves_to_nothing_is_refused():
    """The degenerate case: an empty accumulator reads as a real measurement of zero.

    Every accessor on a monitor with no cells returns zeros — `get_dft`,
    `get_intensity`, the whole spectrum — and nothing in that answer says the region
    missed the cell. The positive control is a request one step inside, which must
    still register.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    inside = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    inside.set_region_from_volume(center=(0.0, 0.0, 1.95), size=(1.0, 1.0, 0.0))
    assert inside.region[5] - inside.region[4] >= 1

    for outside in (9.0, -9.0):
        monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
        with pytest.raises(ValueError, match="does not intersect"):
            monitor.set_region_from_volume(center=(0.0, 0.0, outside), size=(1.0, 1.0, 0.4))
    # ...and past the served lattice images on a wrapping axis, rather than
    # answering with the power of an image nobody asked about.
    for outside in (9.0, -9.0, 6.05):
        monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",),
                             periodic=(True, True, True))
        with pytest.raises(ValueError, match="does not intersect"):
            monitor.set_region_from_volume(center=(0.0, 0.0, outside), size=(1.0, 1.0, 0.0))
    # An index region given directly is checked too — it never sees the ladder.
    with pytest.raises(ValueError, match="spans"):
        DFTMonitor(grid=grid, frequency=1.0, components=("Ez",), region=(5, 5, 0, 10, 0, 10))
    with pytest.raises(ValueError, match="non-negative|finite"):
        DFTMonitor(grid=grid, frequency=1.0, components=("Ez",)).set_region_from_volume(
            center=(0.0, 0.0, 0.0), size=(1.0, -1.0, 0.0)
        )


def test_dft_registration_takes_a_fresh_periodic_answer():
    """`set_region_from_volume` accepts the boundaries in force when the region resolves.

    A driver adds monitors before it knows the run's final boundaries and resolves
    their regions when stepping starts, so the flags stored at construction can be
    a step behind — a PML installed in between turns a wrapping face into an
    absorbing one, and a region continued across it would sample cells the absorber
    now terminates.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",), periodic=(True, True, True))
    monitor.set_region_from_volume(center=(0.0, 0.0, 1.95), size=(1.0, 1.0, 0.3))
    assert monitor._wrapped[2] and monitor.region[5] > grid.nz

    monitor.set_region_from_volume(center=(0.0, 0.0, 1.95), size=(1.0, 1.0, 0.3),
                                   periodic=(False, False, False))
    assert monitor._wrapped == (False, False, False)
    assert monitor.region[5] == grid.nz, "an absorbing face must clip, not continue"
    assert monitor.periodic == (False, False, False)


def test_pml_interior_default_keeps_the_mirror_plane_it_unfolds_about():
    """A mirrored axis has no absorber on its lower face, and no cells to give up there.

    The default region of a DFT monitor in a run with a PML is the absorber's
    interior. On a symmetry-halved axis the lower face is the mirror plane — MEEP
    puts no PML there, and `FdtdDriver.setup_pml` validates its thickness against
    exactly that rule — but the region was trimmed by the PML thickness at *both*
    ends regardless.

    The cost was not the dropped cells. `get_dft_full` unfolds the stored quadrant
    about its own cell 0, which is the cell that straddles the mirror plane; start
    the region four cells in and the reconstruction mirrors about a plane four cells
    off the symmetry plane and returns a full-domain field that looks like one. On a
    2x2x3 cell at resolution 10 with X and Y mirrors and a 4-cell PML the default
    monitor registered x cells 4..7 of the 11 stored, and `get_dft_full` returned a
    4-cell "full domain" spanning x = -0.55..0.55 of a 2.0-wide cell — where the
    absorber's interior is 12 cells spanning -0.55..0.55 in *steps of dx*.
    """
    grid = _StubGrid(10, (2.0, 2.0, 3.0), symmetry=("X", "Y"))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    monitor.set_region_from_pml(4)
    assert monitor.region == (0, grid.nx - 4, 0, grid.ny - 4, 4, grid.nz - 4)
    # Unfolding now recovers the absorber's interior exactly: nx_full - 2*thickness.
    assert monitor.get_dft_full("Ez").shape == (grid.nx_full - 8, grid.ny_full - 8, grid.nz - 8)
    x, y, z = monitor.get_coordinates_full()
    assert len(x) == grid.nx_full - 8 and len(y) == grid.ny_full - 8
    assert np.allclose(np.diff(x), grid.dx, atol=1e-5), "the unfolded axis must be uniformly spaced"
    assert x[0] == pytest.approx(-x[-1], abs=1e-5), "and centred on the mirror plane"

    # The unmirrored control: z is trimmed at both faces, as it always was.
    assert monitor.region[4:6] == (4, grid.nz - 4)
    plain = _StubGrid(10, (2.0, 2.0, 3.0))
    unmirrored = DFTMonitor(grid=plain, frequency=1.0, components=("Ez",))
    unmirrored.set_region_from_pml(4)
    assert unmirrored.region == (4, plain.nx - 4, 4, plain.ny - 4, 4, plain.nz - 4)


def test_unfolding_a_region_that_misses_the_mirror_plane_is_refused():
    """`get_dft_full` mirrors about the region's own cell 0; that must *be* the plane.

    A region sitting entirely in the positive half unfolded about its own near face
    instead, and the result was not merely displaced. On a 2x2x3 cell at resolution
    10 with X and Y mirrors, a region on x cells 5..9 came back as six cells whose
    coordinates ran [-0.75, -0.65, 0.45, 0.55, 0.65, 0.75] — two of the four
    mirrored cells missing, a gap in the middle of a supposedly uniform axis, and
    the cell where the two blocks overlap holding the positive value under a
    mirrored label. Every accessor that unfolds is checked, because any one of them
    handing back a shape is enough for the wrong array to reach a caller.

    The positive control is a region that does reach the plane, which must still
    unfold — otherwise this would pass against an accessor that refused everything.
    A region that CROSSES the plane is a third case: the reflected gather has
    already registered it in full-domain coordinates, so there is no quadrant to
    unfold and get_dft_full says so.
    """
    grid = _StubGrid(10, (2.0, 2.0, 3.0), symmetry=("X", "Y"))
    reaching = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    reaching.set_region_from_volume(center=(0.25, 0.25, 0.0), size=(0.6, 0.6, 0.2))
    assert reaching.region[0] == 0 and reaching.region[2] == 0
    assert reaching.get_dft_full("Ez").shape[0] == 2 * (reaching.region[1] - 1)
    assert len(reaching.get_coordinates_full()[0]) == reaching.get_dft_full("Ez").shape[0]

    crossing = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    crossing.set_region_from_volume(center=(0.0, 0.0, 0.0), size=(0.6, 0.6, 0.2))
    assert crossing.region[0] < 0, "the gather registers the full requested span"
    with pytest.raises(ValueError, match="below the X mirror plane"):
        crossing.get_dft_full("Ez")

    offset = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    offset.set_region_from_volume(center=(0.6, 0.6, 0.0), size=(0.2, 0.2, 0.2))
    assert offset.region[0] > 0, "this region must miss the mirror plane to prove anything"
    for call in (lambda: offset.get_dft_full("Ez"),
                 lambda: offset.get_intensity_full(),
                 lambda: offset.get_coordinates_full()):
        with pytest.raises(ValueError, match="mirror plane"):
            call()
    # The quadrant accessors are untouched: the data is fine, only its full-domain
    # image is undefined.
    assert offset.get_dft("Ez").shape == (4, 4, 4)
    assert offset.get_intensity().shape == (4, 4, 4)
    # The no-electric-components fallback must not be the way round the check.
    magnetic = DFTMonitor(grid=grid, frequency=1.0, components=("Hz",))
    magnetic.set_region_from_volume(center=(0.6, 0.6, 0.0), size=(0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="mirror plane"):
        magnetic.get_intensity_full()


def test_a_mirrored_axis_is_never_continued_periodically():
    """A mirror-folded axis is reached by the symmetry transform, not a lattice shift.

    Continuing one periodically would sample the folded half as though it were the
    neighbouring cell — the right kind of number from the wrong half of the domain.
    Neither the driver nor `FluxMonitor`'s default ever asks for it; asking
    explicitly is refused rather than served.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0), symmetry=("X",))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",), periodic=(True, True, True))
    with pytest.raises(ValueError, match="mirror-folded"):
        monitor.set_region_from_volume(center=(0.0, 0.0, 1.95), size=(1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="mirror-folded"):
        FluxMonitor(grid=grid, frequency=1.0, center=(0.0, 0.0, 1.95), size=(1.0, 1.0, 0.0),
                    periodic=(True, False, True))
    # The unmirrored z axis still continues, and the mirrored x axis still clips.
    allowed = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",), periodic=(False, True, True))
    allowed.set_region_from_volume(center=(0.0, 0.0, 1.95), size=(1.0, 1.0, 0.0))
    assert allowed._wrapped == (False, False, True)


# ---------------------------------------------------------------------------
# Near-to-far-field transformation
# ---------------------------------------------------------------------------
#
# The driver-level cross-validation — CPU MEEP, the analytic Hertzian dipole, and
# the sphere-integral-equals-box-flux energy check — lives in
# test_driver_integration.py. What is pinned here is everything that can be pinned
# without stepping Maxwell: the Green's function itself against a transcription
# written from the textbook in the test, the surface measure against the geometric
# area of the patch, the outward-normal bookkeeping, and the refusals.


def _n2f_stub_field(grid, values):  # Uniform tangential field over the whole stub grid.
    fields = _StubFields(grid, tuple(values))
    for name, value in values.items():
        fields.set_uniform(name, value)
    return fields


def _textbook_farfield(point, source, currents, magnetic_currents, frequency, eps=1.0, mu=1.0):
    """Free-space radiation of one point pair (J, M), written out from the textbook.

    Deliberately independent of `dft.py`: the dyadic Green's function
    `Ghat = g[(1 + (ikr-1)/(kr)^2) I + (3 - 3ikr - (kr)^2)/(kr)^2 rhat rhat]` with
    `g = exp(ikr)/(4 pi r)`, and

        E = i w mu Ghat.J  -  dg/dr (rhat x M)
        H = i w eps Ghat.M  +  dg/dr (rhat x J)

    for the `exp(-i w t)` convention MEEP's DFT implies. If this and `dft.py` ever
    agree by sharing a mistake, they will have to have made it twice.
    """
    offset = np.asarray(point, dtype=np.complex128) - np.asarray(source, dtype=np.float64)
    distance = float(np.linalg.norm(np.real(offset)))
    unit = np.real(offset) / distance
    omega = 2 * np.pi * frequency
    wavenumber = omega * math.sqrt(eps * mu)
    kr = wavenumber * distance
    green = cmath.exp(1j * kr) / (4 * np.pi * distance)
    isotropic = 1.0 + (1j * kr - 1.0) / kr ** 2
    radial = (3.0 - 3j * kr - kr ** 2) / kr ** 2
    gradient = green * (1j * wavenumber - 1.0 / distance)
    J = np.asarray(currents, dtype=np.complex128)
    M = np.asarray(magnetic_currents, dtype=np.complex128)
    dyad_j = isotropic * J + radial * unit * np.dot(unit, J)
    dyad_m = isotropic * M + radial * unit * np.dot(unit, M)
    electric = 1j * omega * mu * green * dyad_j - gradient * np.cross(unit, M)
    magnetic = 1j * omega * eps * green * dyad_m + gradient * np.cross(unit, J)
    return np.concatenate([electric, magnetic])


def test_near2far_green_function_matches_an_independent_transcription():
    """One point patch radiating one known current pair, against the textbook formula.

    A patch flat on all three axes and centred exactly on a cell centre resolves to a
    SINGLE cell of weight 1, and its area element is `dx**0 = 1` (MEEP's `dV0` picks up
    one factor of 1/a per direction the monitor extends along, and this one extends
    along none). So its equivalent currents are exactly `J = weight*(n x H)` and
    `M = -weight*(n x E)` evaluated on the accumulated DFTs, with no surface integral
    in the way — which isolates the Green's function, the dyadic coefficients, the
    `i*omega*mu` prefactor and the two cross products from the registration machinery
    that the rest of this block covers.
    """
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    frequency = 1.0
    # Cell centres sit at -1 + (i+0.5)*0.1, i.e. at ...,-0.15, -0.05, 0.05,...
    patch = Near2FarRegion(center=(0.05, -0.15, 0.25), size=(0.0, 0.0, 0.0),
                           weight=1.0, direction=2)
    monitor = Near2FarMonitor(grid, frequencies=[frequency], regions=[patch], closed=False)
    assert monitor.face_monitors[0].region == (10, 11, 8, 9, 12, 13), \
        "a point patch on a cell centre must resolve to exactly one cell"
    assert monitor.face_monitors[0].components == ("Ex", "Ey", "Hx", "Hy"), \
        "only the components tangential to the patch carry an equivalent current"

    amplitudes = {"Ex": 0.7 - 0.2j, "Ey": -0.35 + 0.9j, "Hx": 0.11 + 0.4j, "Hy": -0.6 - 0.15j}
    fields = _n2f_stub_field(grid, amplitudes)
    for step in range(1, 41):
        monitor.update(fields, step * grid.dt, step)

    accumulated = {
        name: complex(monitor.face_monitors[0].get_dft(name)[0, 0, 0])
        for name in amplitudes
    }
    # n = +z, so n x H = (-Hy, Hx, 0) and -n x E = (Ey, -Ex, 0).
    J = np.array([-accumulated["Hy"], accumulated["Hx"], 0.0])
    M = np.array([accumulated["Ey"], -accumulated["Ex"], 0.0])
    source = np.array([0.05, -0.15, 0.25])
    # The DFT accumulators are complex64, so the two evaluations can only agree to
    # float32 round-off; measured 5.7e-08, against a 1e-06 bar. A formulation
    # difference -- a missing dyadic term, a swapped cross-product sign, a factor of
    # omega -- is a fractional error, orders of magnitude above this.
    for point in ([2.0, 3.0, -4.0], [0.06, -0.14, 0.9], [-30.0, 12.0, 55.0]):
        expected = _textbook_farfield(point, source, J, M, frequency)
        produced = monitor.farfield(point)
        assert np.linalg.norm(produced - expected) <= 1e-6 * np.linalg.norm(expected), (
            f"far field at {point} disagrees with the textbook Green's function: "
            f"{produced} vs {expected}"
        )
        assert np.linalg.norm(expected) > 0.0, "the transcription must produce signal"


def test_near2far_green_function_reaches_its_two_analytic_limits():
    """The dyadic must become the transverse projector far out and the static one close in.

    `Ghat -> g*(I - rhat rhat)` as `k*r -> infinity` and `Ghat -> g/(k r)^2 *
    (3 rhat rhat - I)` as `k*r -> 0`. Both are checked on the same one-cell patch as
    above, which makes them statements about the coefficients rather than about a
    fitted pattern: a far field built from the far-field limit alone would pass the
    first and fail the second by orders of magnitude.
    """
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    # Both limits are reached in `k*r`, and `r` cannot be shrunk below the half-cell
    # the transformation refuses to evaluate inside, so the SECOND limit is reached by
    # lowering the frequency instead. One monitor carries both.
    radiating, quasistatic = 1.0, 1.0e-4
    patch = Near2FarRegion(center=(0.05, 0.05, 0.05), size=(0.0, 0.0, 0.0),
                           weight=1.0, direction=2)
    monitor = Near2FarMonitor(grid, frequencies=[radiating, quasistatic],
                              regions=[patch], closed=False)
    # An Hx-only near field gives a purely y-directed J and no M at all, so the
    # electric far field is exactly `i w mu Ghat . yhat * J`.
    fields = _n2f_stub_field(grid, {"Ex": 0.0, "Ey": 0.0, "Hx": 1.0, "Hy": 0.0})
    for step in range(1, 41):
        monitor.update(fields, step * grid.dt, step)
    origin = np.array([0.05, 0.05, 0.05])

    omega = 2 * np.pi * radiating
    strength = complex(monitor.face_monitors[0].get_dft("Hx", 0)[0, 0, 0])
    far = origin + np.array([0.0, 900.0, 0.0])  # rhat parallel to J: the projector kills it.
    transverse = origin + np.array([900.0, 0.0, 0.0])
    parallel_field = monitor.farfields(far, freq_index=0)[0:3]
    transverse_field = monitor.farfields(transverse, freq_index=0)[0:3]
    assert np.linalg.norm(parallel_field) < 5e-3 * np.linalg.norm(transverse_field), (
        "at k*r >> 1 the dyadic must project out the component along rhat"
    )
    green = cmath.exp(1j * omega * 900.0) / (4 * np.pi * 900.0)
    expected = 1j * omega * green * strength
    assert abs(transverse_field[1] - expected) < 1e-3 * abs(expected), (
        "the transverse far field must be i*omega*mu*g*J with no extra factor"
    )

    # Static limit: k*r = 2*pi*1e-4*0.5 << 1, where Ghat -> g/(kr)^2 (3 rhat rhat - I).
    omega_static = 2 * np.pi * quasistatic
    static_strength = complex(monitor.face_monitors[0].get_dft("Hx", 1)[0, 0, 0])
    radius = 0.5
    near = origin + np.array([radius, 0.0, 0.0])
    static = monitor.farfields(near, freq_index=1)[0:3]
    kr = omega_static * radius
    assert kr < 1e-3, "this limit is only the static one if k*r is small"
    scale = (cmath.exp(1j * kr) / (4 * np.pi * radius)) / kr ** 2
    # J is along +y and rhat along +x, so (3 rhat rhat - I).J = -J.
    predicted = 1j * omega_static * scale * (-static_strength)
    assert abs(static[1] - predicted) < 1e-3 * abs(predicted), (
        f"the k*r -> 0 dyadic must be the static one: {static[1]} vs {predicted}"
    )
    # And the static term is what dominates there: the far-field limit alone would be
    # smaller by 1/(k*r)^2, seven orders of magnitude.
    assert abs(predicted) > 1e5 * abs(1j * omega_static * strength
                                      * cmath.exp(1j * kr) / (4 * np.pi * radius))


def test_near2far_patch_weights_integrate_to_the_requested_area():
    """The surface measure must be the patch's geometric area, wherever it falls.

    MEEP builds the near2far chunks with `include_dV_and_interp_weights`, so each
    cell contributes its `loop_in_chunks` fractional overlap times the area element.
    The sum of those weights times `dx**2` is therefore the requested area exactly —
    for a face on a row of cell centres, one between two rows, one whose extent is not
    a whole number of cells, and on odd as well as even grids. This is the assertion a
    patch snapped onto the nearest sample plane, or weighted with the interior weight
    everywhere, cannot satisfy.
    """
    for cell, resolution in (((2.0, 2.0, 2.0), 10), ((2.1, 2.3, 1.9), 10), ((1.0, 1.0, 1.0), 7)):
        grid = _StubGrid(resolution, cell)
        for normal_offset in (0.0, 0.05, 0.037, -0.062):
            for extent in (0.4, 0.55, 0.31):
                patch = Near2FarRegion(
                    center=(0.0, 0.0, normal_offset), size=(extent, extent, 0.0),
                    weight=1.0, direction=2,
                )
                monitor = Near2FarMonitor(grid, frequencies=[1.0], regions=[patch], closed=False)
                weights = monitor._face_weights(monitor._faces[0])
                area = float(np.sum(np.asarray(weights))) * grid.dx ** 2
                assert area == pytest.approx(extent * extent, rel=1e-6), (
                    f"patch of extent {extent} at z={normal_offset} on a {grid.shape} grid "
                    f"integrated to {area}, not {extent * extent}"
                )


def test_near2far_box_carries_the_outward_normal_on_every_face():
    """Six patches, opposite signs on opposite faces, and the closure test that says so.

    One flipped weight turns a face's contribution inward and changes nothing else a
    caller can see, so both the construction and the predicate that accepts it are
    pinned, together with the mutations each must reject.
    """
    regions = _box_regions((0.1, -0.2, 0.3), (1.0, 2.0, 0.5))
    assert len(regions) == 6
    for axis, size, centre in ((0, 1.0, 0.1), (1, 2.0, -0.2), (2, 0.5, 0.3)):
        faces = [region for region in regions if region.direction == axis]
        assert len(faces) == 2
        low = next(face for face in faces if face.weight < 0)
        high = next(face for face in faces if face.weight > 0)
        assert low.center[axis] == pytest.approx(centre - size / 2)
        assert high.center[axis] == pytest.approx(centre + size / 2)
        assert low.size[axis] == 0.0 and high.size[axis] == 0.0
    assert _is_closed_box(regions)

    flipped = tuple(
        Near2FarRegion(region.center, region.size, -region.weight, region.direction)
        if index == 3 else region
        for index, region in enumerate(regions)
    )
    assert not _is_closed_box(flipped), "two faces of one axis with the same sign is not a box"
    assert not _is_closed_box(regions[:5]), "five faces do not close a box"
    shrunk = tuple(
        Near2FarRegion(region.center, (region.size[0], region.size[1] * 0.9, region.size[2]),
                       region.weight, region.direction)
        if index == 0 else region
        for index, region in enumerate(regions)
    )
    assert not _is_closed_box(shrunk), "a face short of its neighbours leaves a hole"


def test_near2far_refuses_an_open_surface_that_was_not_asked_for():
    """An open surface produces a smooth plausible pattern; it has to be opted into.

    The equivalence theorem applies to a surface that encloses the sources. A single
    aperture plane is a legitimate measurement of a half-space and a silent disaster
    when it was meant to be a box, so the constructor refuses anything that is not a
    closed box unless `closed=False` says the caller means it.
    """
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    single = [Near2FarRegion(center=(0.0, 0.0, 0.35), size=(0.6, 0.6, 0.0), weight=1.0)]
    with pytest.raises(ValueError, match="do not form a closed box"):
        Near2FarMonitor(grid, frequencies=[1.0], regions=single)
    acknowledged = Near2FarMonitor(grid, frequencies=[1.0], regions=single, closed=False)
    assert acknowledged.closed is False
    assert Near2FarMonitor.box(grid, frequencies=[1.0], center=(0, 0, 0),
                               size=(0.6, 0.6, 0.6)).closed is True
    with pytest.raises(ValueError, match="at least one near-field patch"):
        Near2FarMonitor(grid, frequencies=[1.0], regions=[])
    with pytest.raises(ValueError, match="positive extent on every axis"):
        Near2FarMonitor.box(grid, frequencies=[1.0], center=(0, 0, 0), size=(0.6, 0.0, 0.6))


def test_near2far_patch_normal_is_resolved_or_refused():
    """A patch whose normal was guessed radiates a plausible pattern the wrong way."""
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    assert Near2FarRegion((0, 0, 0.3), (0.4, 0.4, 0.0)).normal_axis(grid.dx) == 2
    assert Near2FarRegion((0.3, 0, 0), (0.0, 0.4, 0.4)).normal_axis(grid.dx) == 0
    with pytest.raises(ValueError, match="axes thinner than one cell"):
        Near2FarRegion((0, 0, 0), (0.4, 0.4, 0.4)).normal_axis(grid.dx)  # No flat axis.
    with pytest.raises(ValueError, match="axes thinner than one cell"):
        Near2FarRegion((0, 0, 0), (0.4, 0.0, 0.0)).normal_axis(grid.dx)  # Two flat axes.
    with pytest.raises(ValueError, match="must be the flat axis"):
        Near2FarRegion((0, 0, 0), (0.4, 0.4, 0.0), direction=0).normal_axis(grid.dx)
    with pytest.raises(ValueError, match="0 \\(X\\), 1 \\(Y\\) or 2 \\(Z\\)"):
        Near2FarRegion((0, 0, 0), (0.4, 0.4, 0.0), direction=3).normal_axis(grid.dx)
    with pytest.raises(ValueError, match="finite and non-negative"):
        Near2FarRegion((0, 0, 0), (0.4, 0.4, -0.1)).normal_axis(grid.dx)


def test_near2far_refuses_the_runs_whose_near_field_is_not_one_surface():
    """A Bloch phase and a mirror fold both leave a surface that is not the surface.

    At `k_point != 0` the far field is the phased sum over every lattice image and one
    period alone is the aperture pattern of a unit cell. On a folded grid the stored
    quadrant holds only part of every patch, so the box is not closed at all. Both come
    back as smooth, correctly-shaped radiation patterns, so both are refusals.
    """
    folded = _StubGrid(10, (2.0, 2.0, 2.0), symmetry=("X",))
    with pytest.raises(ValueError, match="mirror-folded grid"):
        Near2FarMonitor.box(folded, frequencies=[1.0], center=(0, 0, 0), size=(0.6, 0.6, 0.6))

    grid = Grid(resolution=10, cell_size=(2.0, 2.0, 2.0), k_point=(0.15, 0.0, 0.0))
    with pytest.raises(ValueError, match="k_point"):
        Near2FarMonitor.box(grid, frequencies=[1.0], center=(0, 0, 0), size=(0.6, 0.6, 0.6))
    # Positive control: k = 0 is an ordinary run and must still be accepted.
    plain = Grid(resolution=10, cell_size=(2.0, 2.0, 2.0), k_point=(0.0, 0.0, 0.0))
    assert Near2FarMonitor.box(plain, frequencies=[1.0], center=(0, 0, 0),
                               size=(0.6, 0.6, 0.6)).closed is True


def test_near2far_with_no_accumulated_field_is_exactly_zero():
    """A surface that saw nothing must return zeros, and a control must not.

    The degenerate answer here is the dangerous one: every reduction over an empty
    accumulator is a perfectly smooth pattern of zeros, and a run that recorded nothing
    is indistinguishable from one that radiated nothing. Zero is the honest answer, so
    it is pinned EXACTLY -- and paired with a control that makes the same call on the
    same monitor produce signal, so a transformation that always returns zero cannot
    pass this.
    """
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    monitor = Near2FarMonitor.box(grid, frequencies=[1.0], center=(0, 0, 0), size=(0.6, 0.6, 0.6))
    silent = monitor.farfields(np.array([[3.0, 1.0, -2.0], [0.0, 0.0, 40.0]]))
    assert np.count_nonzero(silent) == 0, "an unaccumulated surface must radiate exactly nothing"
    assert monitor.radiation_pattern(np.array([0.4, 1.2]), 0.0, 20.0).tolist() == [0.0, 0.0]

    fields = _n2f_stub_field(grid, {name: 0.3 - 0.2j for name in
                                    ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")})
    for step in range(1, 21):
        monitor.update(fields, step * grid.dt, step)
    loud = monitor.farfields(np.array([[3.0, 1.0, -2.0], [0.0, 0.0, 40.0]]))
    assert np.min(np.abs(loud)) > 0.0, "the positive control must produce signal everywhere"

    monitor.reset()
    assert np.count_nonzero(monitor.farfields(np.array([[3.0, 1.0, -2.0]]))) == 0


def test_near2far_flipping_a_face_weight_negates_exactly_that_face():
    """The outward normal is carried by the weight, and nothing else depends on it."""
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    patch = Near2FarRegion(center=(0.0, 0.0, 0.35), size=(0.4, 0.4, 0.0), weight=1.0)
    inverted = Near2FarRegion(center=(0.0, 0.0, 0.35), size=(0.4, 0.4, 0.0), weight=-1.0)
    values = {"Ex": 0.7 - 0.2j, "Ey": -0.35 + 0.9j, "Hx": 0.11 + 0.4j, "Hy": -0.6 - 0.15j}
    produced = []
    for region in (patch, inverted):
        monitor = Near2FarMonitor(grid, frequencies=[1.0], regions=[region], closed=False)
        fields = _n2f_stub_field(grid, values)
        for step in range(1, 21):
            monitor.update(fields, step * grid.dt, step)
        produced.append(monitor.farfield([2.0, -1.0, 5.0]))
    assert np.max(np.abs(produced[0])) > 0.0
    np.testing.assert_allclose(produced[1], -produced[0], rtol=1e-12, atol=0.0)


def test_near2far_refuses_a_far_point_on_the_surface_and_a_stale_read():
    """Two calls that would otherwise return a finite, meaningless number."""
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    monitor = Near2FarMonitor.box(grid, frequencies=[1.0], center=(0, 0, 0), size=(0.6, 0.6, 0.6))
    with pytest.raises(ValueError, match="lies on the near-field surface"):
        monitor.farfield([0.05, 0.05, 0.35])  # A cell centre of the +z face.
    with pytest.raises(ValueError, match="finite"):
        monitor.farfield([0.0, 0.0, np.inf])
    with pytest.raises(ValueError, match="one \\(x, y, z\\) triple"):
        monitor.farfields(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="finite positive radius"):
        monitor.radiation_pattern(0.5, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite positive eps and mu"):
        Near2FarMonitor.box(grid, frequencies=[1.0], center=(0, 0, 0), size=(0.6,) * 3, eps=0.0)


def test_near2far_places_each_cell_where_it_is_not_where_the_patch_was_asked_for():
    """A patch between two rows of cell centres radiates from BOTH rows, at their own z.

    MEEP's `farfield_lowlevel` takes each surface cell's own location (`IVEC_LOOP_LOC`)
    into the Green's function; the requested plane coordinate appears only through the
    interpolation weights. A patch asked for at z = 0.30 on a resolution-10 grid is
    bracketed by the rows at 0.25 and 0.35 with weight 1/2 each, so its far field is the
    half-and-half sum of two sources 0.1 apart — not one source at 0.30, which is what
    collapsing the flat axis onto the requested coordinate would give.

    The distinction is a phase one and therefore invisible in a magnitude plot: the two
    rows are 0.63 radians apart at this frequency, so the correct answer is ~5% below
    the collapsed one in magnitude and rotated in phase. Both candidates are computed
    here from the textbook formula and the wrong one is required to be measurably wrong,
    so the assertion cannot pass by the two being indistinguishable.
    """
    grid = _StubGrid(10, (2.0, 2.0, 2.0))
    frequency = 1.0
    patch = Near2FarRegion(center=(0.05, -0.15, 0.30), size=(0.0, 0.0, 0.0),
                           weight=1.0, direction=2)
    monitor = Near2FarMonitor(grid, frequencies=[frequency], regions=[patch], closed=False)
    face = monitor.face_monitors[0]
    assert face.region == (10, 11, 8, 9, 12, 13 + 1), "the flat axis must bracket both rows"
    weights = np.asarray(monitor._face_weights(monitor._faces[0])).reshape(-1)
    np.testing.assert_allclose(weights, [0.5, 0.5], rtol=1e-6)

    values = {"Ex": 0.7 - 0.2j, "Ey": -0.35 + 0.9j, "Hx": 0.11 + 0.4j, "Hy": -0.6 - 0.15j}
    fields = _n2f_stub_field(grid, values)
    for step in range(1, 41):
        monitor.update(fields, step * grid.dt, step)
    accumulated = {name: complex(face.get_dft(name)[0, 0, 0]) for name in values}
    J = np.array([-accumulated["Hy"], accumulated["Hx"], 0.0])
    M = np.array([accumulated["Ey"], -accumulated["Ex"], 0.0])

    point = [0.05, -0.15, 5.3]  # Straight out along the patch normal: maximum phase spread.
    bracketing = 0.5 * (
        _textbook_farfield(point, (0.05, -0.15, 0.25), J, M, frequency)
        + _textbook_farfield(point, (0.05, -0.15, 0.35), J, M, frequency)
    )
    collapsed = _textbook_farfield(point, (0.05, -0.15, 0.30), J, M, frequency)
    produced = monitor.farfield(point)
    assert np.linalg.norm(produced - bracketing) <= 1e-6 * np.linalg.norm(bracketing), (
        "each surface cell must radiate from its own position"
    )
    separation = np.linalg.norm(bracketing - collapsed) / np.linalg.norm(bracketing)
    assert separation > 0.02, (
        f"the two candidates differ by only {separation:.3e}; this test would pass either way"
    )


# ---------------------------------------------------------------------------
# Monitors on a fold: every axis, either phase, and the cell-centre average.
# ---------------------------------------------------------------------------

_SAMPLE_SOURCE = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                  "size": (0.0, 0.0, 0.0), "source_type": "gaussian", "fwidth": 0.4}


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_folded_axis_gets_the_metallic_plane_its_cell_centre_average_needs(axis):
    """A monitor reaching a folded far face must still be CELL-CENTRED on that axis.

    THE DEFECT THIS PINS. `_sliced_component` clips a non-wrapping axis at the last
    stored cell, so a region that reaches the folded far face got an extended slice of
    exactly the region's own length; `_apply_yee_interpolation` then found nothing to
    consume and skipped the average FOR THE WHOLE AXIS. The quadrant came back holding
    raw Yee samples on the folded axis while every other axis was cell-centred, with
    nothing in the shape, dtype or magnitude to say so — and the default whole-grid
    region of every folded run reached that face, so it was the ordinary case rather
    than an edge one.

    Measured with the DEFAULT monitor, against the same cells of the equivalent
    unfolded run (worst over the three E components):

        folded X   plane withheld 6.55e-01     plane supplied 0.0
        folded Y   plane withheld 6.55e-01     plane supplied 0.0
        folded Z   plane withheld 6.55e-01     plane supplied 0.0

    exactly 0.0 on every axis, because with the plane supplied the folded quadrant is
    the same arithmetic on the same values as the corresponding block of the unfolded
    run. Both numbers come from THIS process: the pre-fix path is reached by
    withholding the plane, so the cost is measured rather than recalled.

    The far plane of a folded axis is the metallic zero the fold terminates on, which
    is what `fields.to_cell_center` has always used there, so the two readbacks now
    agree instead of disagreeing silently.

    The control is the component whose Yee shift on the folded axis is 1: it needs no
    average on that axis and was correct throughout, so a "fix" that perturbed
    everything would fail here. The folded axis is made long enough that its far face
    stays dark for the run, since a live face is a real difference between the two
    runs (a zero ghost against a periodic wrap) and belongs to
    `FdtdDriver._require_folded_far_face_is_quiet` rather than to the average.
    """
    from . import dft as dft_module
    from .fields import IYEE_SHIFTS, mirror_parity

    cell = [3.0, 3.0, 3.0]
    cell[axis] = 4.0  # Long on the folded axis, so its far face stays dark for the run.
    cell = tuple(cell)
    name = "XYZ"[axis]
    # A current this plane leaves even, so the fold accepts it (sources.py's parity rule).
    driving = next(c for c in ("Ex", "Ey", "Ez") if mirror_parity(c, axis, +1) > 0)

    def run(symmetry):
        driver = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True,
                            symmetry=symmetry)
        driver.add_source({**_SAMPLE_SOURCE, "component": driving})
        monitor = driver.add_dft_monitor(1.0, components=("Ex", "Ey", "Ez"))
        driver.run(num_steps=14)
        return monitor

    def worst_against(reference, folded):
        worst, worst_component = 0.0, None
        for component in ("Ex", "Ey", "Ez"):
            quadrant = np.asarray(folded.get_dft(component))
            whole = np.asarray(reference.get_dft(component))
            centre = whole.shape[axis] // 2
            block = whole[(slice(None),) * axis
                          + (slice(centre - 1, centre - 1 + quadrant.shape[axis]),)]
            if block.shape[axis] < quadrant.shape[axis]:
                # The folded quadrant's last cell is the stored second-mirror
                # plane cell of a periodic fold — on the full periodic domain
                # that is cell 0 one lattice vector up (no phase at k = 0).
                block = np.concatenate(
                    (block, whole[(slice(None),) * axis + (slice(0, 1),)]), axis=axis)
            scale = float(np.abs(block).max())
            assert scale > 0.0, f"{component}: the monitor accumulated nothing"
            error = float(np.abs(quadrant - block).max()) / scale
            if error > worst:
                worst, worst_component = error, component
        return worst, worst_component

    reference = run(())
    folded = run((name,))
    assert folded.region[2 * axis + 1] == folded.grid.stored_cells(axis), (
        "this case must reach the folded far face, or it does not exercise the clip"
    )
    unaffected = [c for c in ("Ex", "Ey", "Ez") if IYEE_SHIFTS[c][axis] == 1]
    assert unaffected, "every fold must leave one component needing no average on its axis"

    supplied, component = worst_against(reference, folded)
    assert supplied == 0.0, (
        f"{component} on a folded {name} axis differs from the unfolded run by {supplied:.3e}; "
        f"with the metallic plane supplied the two are the same arithmetic and must agree "
        f"exactly, so any residual means the folded axis is not cell-centred"
    )

    # Withhold the far plane — the pre-fix path — and the same comparison blows up.
    # A periodic fold's far plane comes from _append_reflected_planes (the second
    # mirror's image row); the metallic-zero append serves metallic terminations.
    original = dft_module._append_reflected_planes
    try:
        dft_module._append_reflected_planes = (
            lambda grid, sampled, axes, component, lows: sampled)
        skipped, skipped_component = worst_against(reference, run((name,)))
    finally:
        dft_module._append_reflected_planes = original
    assert skipped > 0.1, (
        f"withholding the far plane changes the folded {name} quadrant by only "
        f"{skipped:.3e} (worst {skipped_component}); this case cannot see the defect it pins"
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("phase", [+1, -1])
def test_the_unfolding_shape_and_coordinates_generalize_to_every_plane(axis, phase):
    """`get_dft_full` must unfold the axis that is folded — including Z — and only it.

    The unfolding used to name X and Y explicitly (`_full_shape`,
    `_reconstruct_from_quadrant`, `get_coordinates_full`), so a Z fold fell through
    every branch and `get_dft_full` returned the STORED QUADRANT under a full-domain
    name: half the cells, the wrong coordinates, and no error.
    """
    from .fields import mirror_parity

    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0),
                symmetry=(Mirror("XYZ"[axis], phase),))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ex", "Ey", "Ez"))
    rng = np.random.default_rng(11)
    for component in ("Ex", "Ey", "Ez"):
        monitor.get_dft(component)[...] = (
            rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
        ).astype(np.complex64)

    for component in ("Ex", "Ey", "Ez"):
        quadrant = np.asarray(monitor.get_dft(component))
        unfolded = np.asarray(monitor.get_dft_full(component))
        # A whole-axis quadrant unfolds to exactly the FULL count: the stored
        # second-mirror plane cell (the extra row of a folded periodic even
        # axis) has no full-domain cell centre and is dropped, as in
        # fields._reconstruct_full_domain.
        expected = list(grid.shape)
        expected[axis] = grid.shape_full[axis]
        assert unfolded.shape == tuple(expected), (
            f"Mirror({'XYZ'[axis]!r}, {phase:+d}): {component} unfolded to {unfolded.shape}, "
            f"expected {tuple(expected)}"
        )
        parity = mirror_parity(component, axis, phase)
        centre = unfolded.shape[axis] // 2
        kept = unfolded.shape[axis] - centre + 1  # Quadrant rows with a home, cell 0 included.
        mirrored = parity * np.flip(
            quadrant[(slice(None),) * axis + (slice(1, centre + 1),)], axis=axis)
        np.testing.assert_array_equal(
            unfolded[(slice(None),) * axis + (slice(0, centre),)], mirrored)
        # The positive half is the quadrant, unphased, from the straddling cell on.
        np.testing.assert_array_equal(
            unfolded[(slice(None),) * axis + (slice(centre, None),)],
            quadrant[(slice(None),) * axis + (slice(1, kept),)])

    coordinates = monitor.get_coordinates_full()
    unfolded_shape = np.asarray(monitor.get_dft_full("Ex")).shape
    assert tuple(len(values) for values in coordinates) == unfolded_shape
    assert coordinates[axis][0] < 0.0 < coordinates[axis][-1]
    assert np.allclose(np.diff(coordinates[axis]), grid.dx, atol=1e-5)


def test_unfolding_three_planes_at_once_carries_the_product_of_their_parities():
    """A cell mirrored in all three planes must carry parity_x * parity_y * parity_z.

    The axes are unfolded one at a time, which is what lets one rule serve one, two or
    three planes; this is the statement that the composition is right, on the fold
    that has no X/Y-only spelling at all.
    """
    from .fields import mirror_parity

    phases = (+1, -1, -1)
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0),
                symmetry=tuple(Mirror("XYZ"[axis], phases[axis]) for axis in range(3)))
    monitor = DFTMonitor(grid=grid, frequency=1.0, components=("Ez",))
    rng = np.random.default_rng(3)
    quadrant = (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)).astype(np.complex64)
    monitor.get_dft("Ez")[...] = quadrant

    unfolded = np.asarray(monitor.get_dft_full("Ez"))
    centres = [unfolded.shape[axis] // 2 for axis in range(3)]
    assert unfolded.shape == grid.shape_full  # Plane cells dropped, as in fields.
    product = 1
    for axis in range(3):
        product *= mirror_parity("Ez", axis, phases[axis])
    # Quadrant cell (2, 2, 2) maps to the positive block at centre+1 on each axis, and
    # its triply mirrored image sits two cells below the plane on each.
    assert unfolded[centres[0] + 1, centres[1] + 1, centres[2] + 1] == pytest.approx(quadrant[2, 2, 2])
    assert unfolded[centres[0] - 2, centres[1] - 2, centres[2] - 2] == pytest.approx(
        product * quadrant[2, 2, 2])
    assert product == -1, "this phase set must actually invert the corner, or the check is blind"


def test_the_fold_helpers_defer_to_the_grid_and_cover_every_axis_in_the_fallback():
    """`_axis_is_mirrored` / `_axis_mirror_phase` must ASK the grid, and know about Z.

    Both exist because the stub grids in this file model the registration conventions
    and carry no methods, so the helpers fall back to the per-axis flags. Two things
    can go wrong quietly and neither shows up in a monitor's output:

      * the helper stops deferring and answers from the flags — harmless while the two
        agree, wrong the moment a grid's own rule differs from a flag someone set;
      * the fallback enumerates X and Y, as every one of these tables used to, so a
        folded Z reads as an ordinary periodic axis.

    The deferral is proved with an object whose method DISAGREES with its flags, which
    is the only way to tell which one was read.
    """
    from .dft import _axis_is_mirrored, _axis_mirror_phase

    class _Disagrees:  # Method says Z is folded and odd; the flags say nothing is folded.
        sym_x = sym_y = sym_z = False

        def is_mirrored(self, axis):
            return axis == 2

        def mirror_phase(self, axis):
            return -1 if axis == 2 else None

    grid = _Disagrees()
    assert [_axis_is_mirrored(grid, axis) for axis in range(3)] == [False, False, True]
    assert [_axis_mirror_phase(grid, axis) for axis in range(3)] == [None, None, -1]

    class _FlagsOnly:  # No methods at all: the stub-grid path.
        sym_x = False
        sym_y = False
        sym_z = True

    flags = _FlagsOnly()
    assert [_axis_is_mirrored(flags, axis) for axis in range(3)] == [False, False, True], (
        "the fallback must read sym_z; enumerating X and Y is how a folded Z became invisible"
    )
    assert [_axis_mirror_phase(flags, axis) for axis in range(3)] == [None, None, 1]
    # And the real Grid agrees with both readings, which is what lets the fallback exist.
    real = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), symmetry=(Mirror("Z", -1),))
    assert [_axis_is_mirrored(real, axis) for axis in range(3)] == [False, False, True]
    assert [_axis_mirror_phase(real, axis) for axis in range(3)] == [None, None, -1]


def _disc_extraction_cylindrical_grid():
    """The Dcyl cell `disc_extraction_efficiency.py` builds, at its own resolution.

    MEEP's own numbers for this cell, read off the live `fields.gv` after `init_sim`:
    ``nr = 250``, ``nz = 115``, ``little_corner = (r 0, z -114)``, ``a = 50``. The z
    count is ODD because MEEP rounded a 2.2916667 cell to 115 pixels, which is what puts
    the cell's low z face half a cell above ``-size_z/2`` and makes the script's own
    near-field surface reach past it.
    """
    return Grid(resolution=50.0, cell_size=(5.0, 0.0, 2.3), cylindrical=True, m=0)


def _disc_extraction_regions():
    """The two Near2FarRegions of `disc_extraction_efficiency.py`, computed as it does.

    Recomputed from the script's own arithmetic rather than transcribed as decimals:
    the whole point of the z case below is a bound that lands a fraction of a cell
    outside the grid, and rounding it here would move it.
    """
    disc_thickness = 0.7 * 1.0 / 2.4
    padding, pml = 1.0, 1.0
    size_z = disc_thickness + padding + pml
    r_um = 4.0
    cap_z = 0.5 * size_z - pml
    wall_z_center = 0.5 * size_z - pml - 0.5 * (padding + disc_thickness)
    wall_z_size = padding + disc_thickness
    cap = ((0.0, r_um), (0.0, 0.0), (cap_z, cap_z))
    wall = ((r_um, r_um), (0.0, 0.0),
            (wall_z_center - 0.5 * wall_z_size, wall_z_center + 0.5 * wall_z_size))
    return cap, wall


@pytest.mark.parametrize(
    "component, first_doubled, first_weight",
    # MEEP's own chunk list for this run's z-cap region, `swigobj.F` walked after
    # init_sim: the r ladder's first stored site and the weight MEEP reports for it.
    #   Er  is=(1, 14)  s0.r=0.8750      Hp  is=(399, 13) ...  (r wall, not this region)
    #   Hr  is=(0, 13)  s0.r=0.5000  dV0=0
    #   Ep  is=(0, 14)  s0.r=0.5000  dV0=0
    [
        ("Ex", 1, 0.875),   # Er: half-integer in r, so its ladder site at doubled -1 is
                            # clipped and the survivor keeps the ladder's s1, not s0.
        ("Hy", 1, 0.875),   # Hp: same parity, same clip.
        ("Hx", 0, 0.5),     # Hr: integer in r — MEEP's Dcyl exception owns the axis row,
        ("Ey", 0, 0.5),     # Ep: and it survives at s0, weighted by 2*pi*0 = 0.
    ],
)
def test_cylindrical_near2far_ladder_is_clipped_at_the_axis_and_ringed_at_the_yee_radius(
    component, first_doubled, first_weight
):
    """The radial ladder of a cap spanning r = 0, against MEEP's own chunk.

    Two separate facts, both measured against the chunk list `disc_extraction_efficiency.py`
    makes MEEP build (resolution 50, cell 5 x 2.3, 4 chunks):

    * **The clip.** A cap from r = 0 outward puts the half-integer components (Er, Hp)
      one site below the axis. MEEP does not refuse that and does not renormalize it: it
      loops from the chunk's owned corner (loop_in_chunks.cpp:439-442) and applies the
      start taper only where the chunk start still coincides with the ladder start, so
      one clipped site leaves the survivor holding the ladder's ``s1``. MEEP reports
      ``is.r = 1`` and ``s0.r = 0.8750`` for Er, which is ``1 - (1 - 0.5)**2 / 2``. The
      integer components (Hr, Ep) are not clipped at all, because
      ``grid_volume::little_owned_corner`` (vec.cpp:432-436) hands the r = 0 row back to
      them; MEEP reports ``is.r = 0`` and ``s0.r = 0.5000``.
    * **The ring radius.** The weight carries ``2*pi*r`` at the COMPONENT's own Yee
      radius, ``doubled/2 * dx``. MEEP reports ``dV0 = 0`` for Hr and Ep — the axis row
      is weighted to nothing — and ``dV0 = 0.00125664`` for Er, which is
      ``(1/50) * 2*pi * 0.01``, i.e. 2*pi*(dx/2) times the one-dimensional dV. A
      cell-centre radius ``(index + 0.5)*dx`` would put Hr and Ep at ``2*pi*dx/2``
      instead of zero, which is the half-cell slip this pins.
    """
    from .dft import YeeRegionDFT

    grid = _disc_extraction_cylindrical_grid()
    assert (grid.nx, grid.nz) == (250, 115), (
        "this test's numbers are MEEP's for a 250 x 115 Dcyl grid; the cell no longer "
        "rasterizes to it"
    )
    assert grid.origin_doubled(0) == 0 and grid.origin_doubled(2) == -114

    cap, _wall = _disc_extraction_regions()
    accumulator = YeeRegionDFT(grid=grid, component=component, bounds=cap, frequencies=(1.0,))

    assert accumulator.site_doubled(0, accumulator.first_index(0)) == first_doubled

    # Read the r profile at one z index and normalize by an interior row, which divides
    # out dV0 and the flat z axis's own interpolation weight and leaves the two things
    # under test: the radius each site is ringed at, and the taper the clip left behind.
    weights = np.asarray(accumulator._weights)[:, 0, 0]
    reference_row = 40
    reference_doubled = accumulator.site_doubled(0, accumulator.first_index(0) + reference_row)
    profile = weights / weights[reference_row]
    for row in (5, 40, 120):  # Interior: ladder weight 1, so the profile IS r/r_ref.
        doubled = accumulator.site_doubled(0, accumulator.first_index(0) + row)
        assert profile[row] == pytest.approx(doubled / reference_doubled, rel=1e-12), (
            f"{component}: interior row {row} is not 2*pi*r at its own Yee radius "
            f"(doubled {doubled}); a cell-centre radius would read "
            f"{(accumulator.first_index(0) + row + 0.5) * 2 / reference_doubled:.6f}"
        )
    assert profile[0] == pytest.approx(
        first_weight * first_doubled / reference_doubled, abs=1e-12
    ), (
        f"{component}: the first surviving site must carry MEEP's ladder weight "
        f"{first_weight} on the ring at doubled r = {first_doubled}"
    )
    if first_doubled == 0:
        assert weights[0] == 0.0, (
            f"{component} is an integer-r component, so its first site IS the axis and "
            f"MEEP weights it 2*pi*0 = 0 (dV0 = 0 on the live chunk). A cell-centre "
            f"radius would give it 2*pi*dx/2 and it would radiate."
        )


@pytest.mark.parametrize(
    "component, first_doubled, first_weight",
    # MEEP's own low-z chunk for this run's r-wall region:
    #   Hz  is=(399, -112)  s0.z=1.0000      Ez  is=(400, -113)  s0.z=0.9783
    #   Hp  is=(399, -113)  s0.z=0.9783      Ep  is=(400, -112)  s0.z=1.0000
    [("Hz", -112, 1.0), ("Ez", -113, 0.9782986111111112),
     ("Hy", -113, 0.9782986111111112), ("Ey", -112, 1.0)],
)
def test_near2far_ladder_reaching_below_the_cell_is_clipped_to_meeps_owned_corner(
    component, first_doubled, first_weight
):
    """A surface that pokes out of the cell loses those sites, not the whole lift.

    `disc_extraction_efficiency.py` places its r-wall region from the NOMINAL cell size
    while MEEP rasterized the cell to an odd 115 pixels, so the region's low z bound sits
    0.29 of a cell below the stored grid. MEEP clips it at the chunk's owned corner —
    ``little_corner + 2 - iyee_shift(c)``, i.e. stored index ``1 - parity``
    (vec.hpp:1102-1104) — and leaves the survivors unrenormalized, exactly as on the
    cylindrical axis. This engine used to raise instead, which is why all three
    cylindrical corpus scripts failed at construction.

    The four numbers below are MEEP's own, off the live chunk list: the z-shift-0
    components (Hz, Ep) lose TWO sites and their survivor holds weight 1.0, while the
    z-shift-1 components (Ez, Hp) lose ONE and hold the ladder's ``s1``, 0.9783 =
    ``1 - (1 - 0.7917)**2 / 2``.
    """
    from .dft import YeeRegionDFT

    grid = _disc_extraction_cylindrical_grid()
    _cap, wall = _disc_extraction_regions()
    accumulator = YeeRegionDFT(grid=grid, component=component, bounds=wall, frequencies=(1.0,))

    assert accumulator.first_index(2) >= 0
    assert accumulator.site_doubled(2, accumulator.first_index(2)) == first_doubled

    # The z weights alone: divide out the r ladder and the ring, which the r-normal
    # wall applies on its single (or paired) r site.
    weights = np.asarray(accumulator._weights)
    axial = weights[0, 0, :] / weights[0, 0, len(weights[0, 0]) // 2]
    assert axial[0] == pytest.approx(first_weight, rel=1e-9), (
        f"{component}: MEEP reports s0.z = {first_weight} for this chunk after the clip"
    )


def test_first_owned_index_is_meeps_little_owned_corner():
    """The clip bound itself, both parities and the Dcyl axis exception.

    ``little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)`` is stored index
    ``1 - parity`` on every axis of every grid; the ONE exception is MEEP's explicit
    ``if (dim == Dcyl && origin.r() == 0.0 && iloc.r() == 2)`` (vec.cpp:432-436), which
    hands the r = 0 row back to a shift-0 component. Getting that exception wrong does
    not raise: it silently drops the axis row of Ep, Hr and Ez from every cylindrical
    near-field surface.
    """
    from .dft import _clip_to_owned, _first_owned_index

    cartesian = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0))
    cylindrical = Grid(resolution=10.0, cell_size=(2.0, 0.0, 2.0), cylindrical=True, m=0)
    for axis in range(3):
        assert _first_owned_index(cartesian, axis, 0) == 1
        assert _first_owned_index(cartesian, axis, 1) == 0
    assert _first_owned_index(cylindrical, 0, 0) == 0, (
        "the cylindrical AXIS row is owned by a shift-0 component — MEEP's own special case"
    )
    assert _first_owned_index(cylindrical, 0, 1) == 0
    assert _first_owned_index(cylindrical, 2, 0) == 1, (
        "the exception is the r axis only; z takes the generic corner"
    )

    ladder = np.array([0.125, 0.875, 1.0, 1.0, 0.9, 0.4])
    # No clip: the ladder is returned untouched.
    assert _clip_to_owned(0, 3, 6, ladder, "probe") == (3, 6, ladder)
    # Two sites clipped: the survivors keep the weights they had — MEEP applies the
    # start taper only where the chunk start still coincides with the ladder start.
    start, count, kept = _clip_to_owned(1, -1, 6, ladder, "probe")
    assert (start, count) == (1, 4)
    assert list(kept) == [1.0, 1.0, 0.9, 0.4], (
        "renormalizing the survivors would add back the 0.125 + 0.875 MEEP dropped"
    )
    with pytest.raises(ValueError, match="entirely outside the sites MEEP owns"):
        _clip_to_owned(4, -6, 6, ladder, "probe")


@pytest.mark.parametrize(
    "component, is_x, weights",
    # MEEP's own dft chunks for binary_grating_n2f.py's point Near2FarRegion
    # (resolution 25, cell (8.5, 0, 0) -> 213 x 1 pixels, little_corner x = -212,
    # region at x = 1.75 = doubled 87.5), read off the live list:
    #   ez  N=2  is=(86, 2)  ie=(88, 2)  s0=(0.25, ...)  s1=(0.75, ...)  dV0=1.0
    #   hy  N=2  is=(87, 2)  ie=(89, 2)  s0=(0.75, ...)  s1=(0.25, ...)  dV0=1.0
    # and TWO chunks per component — one per lattice image of the one-pixel y axis,
    # with single-site y weights 0.0 and 1.0, whose sum is this accumulator's 1.0.
    [("Ez", 86, (0.25, 0.75)), ("Hy", 87, (0.75, 0.25))],
)
def test_unit_axis_region_takes_meeps_single_plane_at_the_summed_image_weight(
    component, is_x, weights
):
    """A zero-extent request on a one-pixel periodic axis is the WHOLE direction.

    That is MEEP's own reading — ``nosize_direction`` (fields.cpp:730-737), the
    lower-dimensional emulation every ``mp.Vector3(sx, 0, 0)`` script relies on —
    and the ownership ladder is the wrong tool for it: ``first_owned = 1 - parity``
    clips stored index 0, which is the axis's ONLY site, so the constructor used to
    raise "lies entirely outside the sites MEEP owns" for a region MEEP accepts
    (the measured failure of binary_grating_n2f.py's point region). The single
    stored plane carries weight exactly 1 — the sum of MEEP's two lattice-image
    chunks (0.0 + 1.0) — and NO 1/resolution factor: MEEP's own chunks report
    ``dV0 = 1.0`` for the point region, a region with no extent integrates nothing.

    The x ladder beside it must stay untouched: the zero-extent split pair at the
    component's own parity, weights (w0, w1) matching MEEP's chunk s0/s1 exactly.
    """
    from .dft import YeeRegionDFT

    grid = Grid(resolution=25.0, cell_size=(8.52, 0.04, 0.0), dimensions=2)
    assert (grid.nx, grid.ny, grid.nz) == (213, 1, 1)
    assert grid.origin_doubled(0) == -212, "MEEP's odd-count origin, -(n - 1)"
    assert grid.nosize_direction(1) and not grid.is_invariant(1), (
        "y must be the one-pixel periodic emulation, not an invariant axis — the "
        "two are different simulations and only the former is under test"
    )

    accumulator = YeeRegionDFT(
        grid=grid, component=component,
        bounds=((1.75, 1.75), (0.0, 0.0), (0.0, 0.0)), frequencies=(1.0,),
    )
    assert accumulator.site_doubled(0, accumulator.first_index(0)) == is_x, (
        f"{component}: the x split pair starts at MEEP's is.x = {is_x}"
    )
    assert accumulator._slices[0] == slice((is_x + 212 - (is_x % 2)) // 2,
                                           (is_x + 212 - (is_x % 2)) // 2 + 2)
    assert accumulator._slices[1] == slice(0, 1), (
        "the one-pixel y axis contributes its single stored plane; the ownership "
        "ladder would have clipped it away entirely"
    )
    full = np.asarray(accumulator._weights)
    assert full.shape == (2, 1, 1)
    assert full[:, 0, 0] == pytest.approx(list(weights), abs=1e-12), (
        f"{component}: x weights must be MEEP's chunk s0/s1 = {weights} with the y "
        f"plane at exactly 1.0 (the summed image weights) and no dV factor — a "
        f"1/resolution factor here would scale the whole far field by 25"
    )


# --- force and energy monitors ----------------------------------------------------


def test_a_force_region_whose_declared_direction_is_not_its_normal_is_refused_by_name():
    """The off-diagonal stress branch is named, not approximated.

    ``add_dft_force`` resolves ``fd`` from the region's declared direction and ``nd``
    from its geometry, and ``fd != nd`` selects a completely different registration —
    two centred accumulators per family walked in parallel (stress.cpp:168-177) rather
    than one raw-Yee accumulator squared. No case in MEEP's own tests or examples
    reaches it, so there is no oracle for a transcription; silently running the
    diagonal reduction on an off-diagonal region would return a finite, plausible
    force nobody could check.

    The positive control is the same region declared diagonally, which must build.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    with pytest.raises(ValueError, match="OFF-DIAGONAL"):
        ForceMonitor(grid=grid, frequencies=(1.0,), center=(1.2, 0.0, 0.0),
                     size=(0.0, 2.0, 2.0), force_direction=1, normal=0)
    diagonal = ForceMonitor(grid=grid, frequencies=(1.0,), center=(1.2, 0.0, 0.0),
                            size=(0.0, 2.0, 2.0), force_direction=0, normal=0)
    assert len(diagonal._accumulators) == 6, (
        "the diagonal branch registers E and H on all three field directions "
        "(LOOP_OVER_FIELD_DIRECTIONS, vec.hpp:147-149)")


def test_the_diagonal_stress_weights_add_the_force_axis_and_subtract_the_other_two():
    """``weight1 = where->weight * (d == fd ? +0.5 : -0.5)``, stress.cpp:180.

    The constant and the SIGN RULE are separate facts and a transcription can get the
    constant right while losing the rule; the result is then a force of the wrong sign
    and magnitude with nothing else about the run disturbed. Read off the monitor
    rather than recomputed from the same expression, and checked for both a plain and
    a negated region weight, because a flux-box face is spelled ``weight=-1``.

    A COMPLEX region weight is kept rather than refused, and only its real part
    survives: ``stress_sum`` takes ``real(extra_weight * F * conj(F))``
    (stress.cpp:95-97) and ``|F|^2`` is real, so the imaginary part cannot reach the
    answer. That is MEEP's own algebra, not a simplification here.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    for weight, expected in ((1.0, (-0.5, 0.5, -0.5)),
                             (-1.0, (0.5, -0.5, 0.5)),
                             (2.0 + 7.0j, (-1.0, 1.0, -1.0))):
        monitor = ForceMonitor(grid=grid, frequencies=(1.0,), center=(0.0, 1.2, 0.0),
                               size=(2.0, 0.0, 2.0), force_direction=1, normal=1,
                               weight=weight)
        assert monitor._axis_weights == expected, (
            f"region weight {weight}: the diagonal weights are {monitor._axis_weights}, "
            f"expected {expected} — +0.5 on the force axis, -0.5 on the other two.")


def test_the_diagonal_force_reduction_applies_the_cell_weight_once_not_twice():
    """``sqrt_dV_and_interp_weights`` means the weight lands on ``|F|^2`` exactly once.

    MEEP folds ``sqrt(IVEC_LOOP_WEIGHT)`` into the diagonal chunks per step
    (dft.cpp:296-300), so ``real(extra_weight * F * conj(F))`` carries ``w``, not
    ``w^2``. This accumulator stores the raw transform and applies the full weight at
    read time, which is the same thing — but the obvious spelling, reusing
    ``YeeRegionDFT.packed`` as every other consumer does, multiplies the TRANSFORM by
    ``w`` and then squares it.

    Written against an independently computed expectation: a known transform is
    planted in each accumulator and the reduction compared against
    ``sum_d weight1_d * sum_sites w * |F|^2`` built from ``_weights`` directly. The
    control is ``packed``'s own quadratic form, which must DISAGREE — otherwise the
    two routes are indistinguishable here and the test proves nothing.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    monitor = ForceMonitor(grid=grid, frequencies=(1.0,), center=(0.0, 1.25, 0.0),
                           size=(1.34, 0.0, 1.34), force_direction=1, normal=1)
    expected = 0.0
    control = 0.0
    for index, (axis, accumulator) in enumerate(monitor._accumulators):
        planted = np.arange(accumulator._dft.size, dtype=np.float64).reshape(
            accumulator._dft.shape) * (0.017 + 0.031 * index) + 0.11j * (index + 1)
        accumulator._dft = grid.xp.asarray(planted.astype(np.complex64))
        held = np.asarray(accumulator._dft)[0]
        weights = np.abs(np.asarray(accumulator._weights))
        expected += monitor._axis_weights[axis] * float(
            np.sum(weights * np.abs(held.astype(np.complex128)) ** 2))
        control += monitor._axis_weights[axis] * float(
            np.sum(np.abs((held * weights).astype(np.complex128)) ** 2))
    measured = monitor.get_force(0)
    assert abs(measured - expected) <= 1e-9 * abs(expected), (
        f"the reduction gave {measured} where applying the cell weight once gives "
        f"{expected}.")
    assert abs(control - expected) > 0.1 * abs(expected), (
        f"the once-weighted and twice-weighted reductions agree to "
        f"{abs(control - expected) / abs(expected):.2e} on this region, so the test "
        f"cannot tell them apart — pick a region whose cells are partially covered.")


def test_the_energy_reduction_weights_only_the_side_meep_gives_the_measure_to():
    """E and H carry ``include_dV_and_interp_weights``; D and B do not (dft.cpp:722-729).

    So each pair picks up the volume measure exactly once, the same trick
    :class:`FluxMonitor` plays with E and H. Weighting both members is the
    symmetric-looking spelling and squares the measure, which on a region whose ends
    are partially covered is neither MEEP's answer nor a constant factor from it.

    Planted transforms again, and the control is the both-sides form, which must
    differ. ``total()`` is checked to be exactly ``electric() + magnetic()``
    (dft.cpp:703-714) rather than an independently reduced third quantity.
    """
    grid = _StubGrid(10, (4.0, 4.0, 4.0))
    monitor = EnergyMonitor(grid=grid, frequencies=(1.0,), center=(0.0, 0.0, 0.63),
                            size=(0.0, 1.34, 0.0))
    weights = np.asarray(monitor._weights, dtype=np.float64)
    planted = {}
    for index, component in enumerate(sorted(monitor._dft)):
        block = np.arange(monitor._dft[component].size, dtype=np.float64).reshape(
            monitor._dft[component].shape) * (0.013 + 0.007 * index)
        block = block + 0.05j * (index + 1)
        planted[component] = block.astype(np.complex64)
        monitor._dft[component] = grid.xp.asarray(planted[component])

    def reduce_pairs(pairs, both_sides):
        total = 0.0
        for weighted, plain in pairs:
            first = planted[weighted][0].astype(np.complex128) * weights
            second = planted[plain][0].astype(np.complex128)
            if both_sides:
                second = second * weights
            total += float(np.sum(0.5 * np.real(np.conj(first) * second)))
        return total * monitor._measure

    electric_pairs = (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz"))
    magnetic_pairs = (("Hx", "Bx"), ("Hy", "By"), ("Hz", "Bz"))
    expected_e = reduce_pairs(electric_pairs, both_sides=False)
    expected_m = reduce_pairs(magnetic_pairs, both_sides=False)
    # 1e-6, not 1e-9: this reduction runs in the accumulators' own complex64 — MEEP's
    # `dft_energy::electric` likewise multiplies in `complex<realnum>` (dft.cpp:686,
    # unlike `stress_sum`, which promotes to complex<double> explicitly) — while the
    # expectation above is built in float64. Measured 6.1e-09 relative on this region;
    # the both-sides control below is four orders of magnitude further away, so the
    # looser bound costs the test nothing.
    assert abs(monitor.electric(0) - expected_e) <= 1e-6 * abs(expected_e), (
        f"electric() gave {monitor.electric(0)}, expected {expected_e}")
    assert abs(monitor.magnetic(0) - expected_m) <= 1e-6 * abs(expected_m), (
        f"magnetic() gave {monitor.magnetic(0)}, expected {expected_m}")
    assert monitor.total(0) == monitor.electric(0) + monitor.magnetic(0)
    both = reduce_pairs(electric_pairs, both_sides=True)
    assert abs(both - expected_e) > 0.1 * abs(expected_e), (
        f"weighting both sides of each pair differs by only "
        f"{abs(both - expected_e) / abs(expected_e):.2e}, so this region cannot tell "
        f"the two spellings apart.")


# --- monitors on a run that DERIVES E ----------------------------------------------

@pytest.mark.parametrize(
    "label,kwargs,center,size,gathers",
    [
        # Whole-grid default region on a periodic run: the contiguous slice, and the
        # wrapping slice on every axis whose Yee shift is 0.
        ("default_region", dict(), None, None, False),
        ("default_region_bloch", dict(k_point=(0.31, 0.19, 0.0),
                                      force_complex_fields=True), None, None, False),
        # A region crossing the periodic face takes `_gathered_component` instead.
        ("crosses_the_far_face", dict(), (2.6, 0.0, 0.0), (2.0, 3.0, 0.0), True),
        ("crosses_the_far_face_bloch", dict(k_point=(0.31, 0.19, 0.0),
                                            force_complex_fields=True),
         (2.6, 0.4, 0.0), (2.0, 3.0, 0.0), True),
        # A folded run whose region reaches BELOW the plane gathers through the
        # mirror tables, whose per-site parity is the other deferred factor.
        ("reaches_below_an_even_plane", dict(symmetry=("Y",)),
         (0.0, -1.1, 0.0), (3.0, 3.0, 0.0), True),
        ("reaches_below_an_odd_plane", dict(symmetry=(Mirror("Y", -1),)),
         (0.0, -1.1, 0.0), (3.0, 3.0, 0.0), True),
    ],
)
def test_monitor_on_a_derived_E_run_matches_the_materialized_product(
        label, kwargs, center, size, gathers):
    """A monitor must read the same bytes whether E is stored or derived.

    With no PML and no susceptibility nothing stores E, so `Fields.component_factors`
    hands the monitors `D` and `inv_eps` and they gather each and multiply the blocks
    — never forming the product over the whole grid to keep a plane of it (measured
    1.298 -> 0.847 ms per monitor per step on a 640x640 cell). The transformation is
    only legitimate because slicing, `take` and the deferred phase/parity factors are
    all pointwise or index-permuting, so this pins the claim the direct way: the same
    fields, sampled once with E derived and once with the identical product written
    into storage, must agree BYTE FOR BYTE. A tolerance here would pass on a run that
    quietly applied a Bloch factor to epsilon or reordered the phase multiplications.
    """
    del label
    driver = FdtdDriver(resolution=20.0, cell_size=(3.0, 3.0, 0.0), dimensions=2, **kwargs)
    generator = np.random.default_rng(4)
    for name in ("Dx", "Dy", "Dz"):
        array = getattr(driver.fields, name)
        values = generator.standard_normal(array.shape)
        if np.iscomplexobj(array):
            values = values + 1j * generator.standard_normal(array.shape)
        array[...] = values
    # A non-uniform epsilon, so a factor applied to the wrong array cannot cancel.
    for component in ("Ex", "Ey", "Ez"):
        inverse = driver.fields.inverse_epsilon_for(component)
        inverse[...] = (0.5 + 0.25 * generator.random(inverse.shape)).astype(inverse.dtype)

    monitor = driver.add_dft_monitor(components=["Ex", "Ey", "Ez"], fcen=1.0, df=0.2,
                                     nfreq=2, center=center, size=size)
    driver.step()  # Resolves the region and the gather tables.
    assert not driver.fields.stores_E
    # Which of the two sampling paths this case exercises — asserted so a change in
    # the region defaults cannot quietly make half these cases test the same branch.
    assert (monitor._gather is not None) is gathers
    samples = {component: np.asarray(monitor._sample(driver.fields, component))
               for component in ("Ex", "Ey", "Ez")}
    assert all(float(np.abs(block).max()) > 0.0 for block in samples.values())
    derived = {component: block.tobytes() for component, block in samples.items()}

    # Now materialize exactly the same product into storage and sample again.
    driver.fields.enable_field_storage()
    for component in ("Ex", "Ey", "Ez"):
        displacement = getattr(driver.fields, "D" + component[1])
        getattr(driver.fields, component)[...] = (
            displacement * driver.fields.inverse_epsilon_for(component))
    assert driver.fields.stores_E
    stored = {component: np.asarray(monitor._sample(driver.fields, component)).tobytes()
              for component in ("Ex", "Ey", "Ez")}
    assert derived == stored
