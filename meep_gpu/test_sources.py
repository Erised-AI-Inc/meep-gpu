"""
Value tests for the FDTD current sources (`sources.py`).

Covers the conventions that other modules and the CPU-MEEP parity work depend
on: the `1/(-2*pi*i*f)` amplitude factor, delta-function scaling per zero-size
dimension, MEEP's Simpson boundary weights (including the load-bearing `elif`
ordering), the Gaussian and continuous-wave temporal waveforms, value-level
source injection, and the loud-failure behavior for bad components, dtypes and
off-plane sources under symmetry.

The magnetic (`Hx`/`Hy`/`Hz`), `is_integrated` and custom-envelope paths add
three things MEEP alone can settle: which array the current lands in, which
half-step it is evaluated at, and which of the dipole/current pair the waveform
represents. Each is wrong in a way that still produces a plausible field, so
each has a CPU-MEEP cross-validation at the bottom of this file that asserts on
the *complex* field — a half-step error is pure phase and survives every
magnitude comparison — and each of those carries the wrong convention as a
control, so a bound that a mistake could also meet cannot pass silently.

Most waveform reference values are MEEP's own, produced by evaluating
`meep.continuous_src_time.dipole` / `meep.gaussian_src_time.dipole` from meep
1.29.0 and hard-coded so the bulk of the suite stays a pure-NumPy, meep-free
unit test. Only the cross-validation cases need meep, and they run it in a
subprocess (meep and torch each ship an OpenMP runtime and abort when loaded
into one process). Everything runs on the NumPy backend; no CUDA or CuPy is
involved.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from . import sources as src
from .fields import Fields
from .grid import Grid, Mirror
from .sources import (
    ContinuousEnvelope,
    ContinuousSource,
    CustomEnvelope,
    ExtendedSource,
    GaussianEnvelope,
    GaussianPulsedSource,
    SourceEnvelope,
    VolumeSource,
    create_gaussian_beam_source,
)
from .stepping import fill_symmetry_bc_D, step_B, step_D, update_E, update_H

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(MEEP_MISSING, reason="CPU MEEP is not installed in this environment")


class _StubGrid:
    """Stand-in for `grid.Grid` exposing only what sources.py reads.

    Mirrors the documented Grid contract (cell counts `int(round(L*resolution))`,
    `dx = 1/resolution`, `dt = courant/resolution`, symmetry halving to
    `n_full//2 + 1` with cell 0 straddling the mirror plane at -0.5*dx). Using a
    stub keeps these tests independent of the sibling grid module.

    `origin_doubled` is deliberately absent: `_build_source_points` reads it
    defensively and falls back to the same `-(n - n % 2)` rule, and leaving the
    stub without it keeps that fallback exercised. `axis_wraps` and `bloch_phase`
    ARE provided, because the lattice fold is the behaviour these tests measure
    and a stub that silently reported "no phase" would let an unphased wrap pass.
    """

    def __init__(self, resolution, cell_size, symmetry=(), courant=0.5, xp=np, k_point=(0.0, 0.0, 0.0)):
        self.xp = xp
        self.resolution = float(resolution)
        self.courant = float(courant)
        self.dx = 1.0 / self.resolution
        self.dt = self.courant / self.resolution
        self.nx_full = int(round(cell_size[0] * resolution))
        self.ny_full = int(round(cell_size[1] * resolution))
        self.nz_full = int(round(cell_size[2] * resolution))
        self.sym_x = 'X' in symmetry
        self.sym_y = 'Y' in symmetry
        self.nx = self.nx_full // 2 + 1 if self.sym_x else self.nx_full
        self.ny = self.ny_full // 2 + 1 if self.sym_y else self.ny_full
        self.nz = self.nz_full
        self.Lx = self.nx_full * self.dx
        self.Ly = self.ny_full * self.dx
        self.Lz = self.nz_full * self.dx
        self.shape = (self.nx, self.ny, self.nz)
        self.k_point = tuple(float(value) for value in k_point)

    def axis_wraps(self, axis):  # Periodic unless mirror-folded (Grid.axis_wraps).
        return not (self.sym_x, self.sym_y, False)[axis]

    def bloch_phase(self, axis):  # exp(i*2*pi*k*L) crossing the upper face, or None at k = 0.
        k_value = self.k_point[axis]
        if k_value == 0.0:
            return None
        return complex(np.exp(2j * np.pi * k_value * (self.Lx, self.Ly, self.Lz)[axis]))


class _StubFields:
    """Stand-in for `fields.Fields` holding only the primary D and B arrays."""

    def __init__(self, grid, dtype=np.complex64):
        xp = grid.xp
        self.Dx = xp.zeros(grid.shape, dtype=dtype)
        self.Dy = xp.zeros(grid.shape, dtype=dtype)
        self.Dz = xp.zeros(grid.shape, dtype=dtype)
        self.Bx = xp.zeros(grid.shape, dtype=dtype)
        self.By = xp.zeros(grid.shape, dtype=dtype)
        self.Bz = xp.zeros(grid.shape, dtype=dtype)


def _unit_grid(resolution=10.0, cell=(2.0, 2.0, 2.0), symmetry=()):  # 2x2x2 natural-unit cell.
    return _StubGrid(resolution, cell, symmetry=symmetry)


def test_amp_factor_is_reciprocal_of_minus_two_pi_i_f():
    """MEEP sources.cpp:104 — amp = 1/(-2*pi*i*f), i.e. +i/(2*pi*f)."""
    frequency = 0.8
    expected = 1.0 / (-2j * np.pi * frequency)
    assert expected == pytest.approx(1j / (2 * np.pi * frequency))

    grid = _unit_grid()
    cw = ContinuousSource(grid=grid, frequency=frequency, component='Ez')
    ext = ExtendedSource(grid=grid, frequency=frequency, component='Ez',
                         center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    pulse = GaussianPulsedSource(grid=grid, frequency=frequency, fwidth=0.2, component='Ez',
                                 center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    for source in (cw, ext, pulse):
        assert source._amp_factor == expected
        assert source._omega == pytest.approx(2 * np.pi * frequency)

    with pytest.raises(ValueError):
        ContinuousSource(grid=grid, frequency=0.0, component='Ez')


def test_zero_frequency_and_fwidth_rejected():
    grid = _unit_grid()
    with pytest.raises(ValueError):
        GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.0, component='Ez',
                             center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))


def test_delta_scaling_multiplies_resolution_per_zero_size_dimension():
    """A point source picks up one factor of the resolution per zero-size axis."""
    grid = _unit_grid(resolution=10.0)
    cw = ContinuousSource(grid=grid, frequency=0.8, component='Ez', size=(0.0, 0.0, 0.0))
    # The scaling lives in the per-point amplitudes, where MEEP puts it, so it is
    # measured as the total deposited weight rather than as a separate multiplier.
    assert sum(cw._point_weights) == pytest.approx(10.0 ** 3)

    # Same rule inside the extended-source machinery (MEEP sources.cpp:481-484):
    # a fully zero-size volume source carries amplitude * a^3 in total.
    point = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                           center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    assert sum(point._amplitudes) == pytest.approx(10.0 ** 3)

    # A z-normal sheet has one zero-size axis, so its total is a^1 times the
    # transverse weight sum, and the Simpson weights along an extended axis sum
    # to the extent in cells (0.4 * 10 = 4 per axis).
    sheet = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                           center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    assert sum(sheet._amplitudes) == pytest.approx(10.0 * 4.0 * 4.0)

    volume = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                            center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4))
    assert sum(volume._amplitudes) == pytest.approx(4.0 * 4.0 * 4.0)


def test_point_extended_source_lands_on_hand_computed_cells():
    """Hand-worked loop_in_chunks trace for an Ez point source at the origin.

    Grid: resolution 10, 2x2x2 cell, no symmetry -> 20 cells/axis, dx = 0.1.
    iyee_shift(Ez) = (0,0,1), so iyee_c = (1,1,0) and yee_c = (0.05, 0.05, 0).
    X/Y: vec2diel_floor(0.05, 10) = 1, minus iyee_c -> is = ie = 0 (one point).
    Z:   vec2diel_floor(0, 10) = -1, vec2diel_ceil(0, 10) = 1 -> two points,
         zero-extent weights w0 = w1 = 0.5 each.
    Indices: (is - io)/2 with io = -20 -> (10, 10, 9) and (10, 10, 10).
    Amplitude: 0.5 * (1 * 10^3) = 500 at each point.
    """
    grid = _unit_grid(resolution=10.0)
    point = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                           center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    assert point._indices == [(10, 10, 9), (10, 10, 10)]
    assert point._amplitudes == [pytest.approx(500.0), pytest.approx(500.0)]
    assert point._n_source_points == 2


def test_simpson_boundary_weights_hand_values():
    """MEEP loop_in_chunks.cpp:257-297, one case per branch of the ladder."""
    a = 10.0
    # 3+ cells: is=0, ie=6, w0 = 0.8, w1 = 1.2
    assert src._compute_boundary_weights(0.02, 0.32, 0, 6, a) == pytest.approx(
        (0.32, 0.98, 0.72, 0.98))
    # 2 cells: is=0, ie=4 -> s1 = e1 = 1 - 0.02 - 0.02
    assert src._compute_boundary_weights(0.02, 0.22, 0, 4, a) == pytest.approx(
        (0.32, 0.96, 0.72, 0.96))
    # 1 cell: is=0, ie=2 -> s0 = 0.32 - 0.02, e0 = 0.72 - 0.02, s1 = e0, e1 = s0
    assert src._compute_boundary_weights(0.02, 0.12, 0, 2, a) == pytest.approx(
        (0.30, 0.70, 0.70, 0.30))
    # Zero extent straddling a cell boundary: is=-1, ie=1 -> w0 = w1 = 0.5
    assert src._compute_boundary_weights(0.0, 0.0, -1, 1, a) == pytest.approx(
        (0.5, 0.5, 0.5, 0.5))
    # Zero extent landing exactly on a grid position: is = ie = 0
    assert src._compute_boundary_weights(0.0, 0.0, 0, 0, a) == pytest.approx(
        (1.0, 1.0, 1.0, 1.0))
    # Odd span falls through to the unweighted default
    assert src._compute_boundary_weights(0.0, 0.05, 0, 1, a) == pytest.approx(
        (1.0, 1.0, 1.0, 1.0))


def test_zero_extent_branch_precedes_the_one_cell_branch():
    """The `elif` ordering is load-bearing: is=-1, ie=1 is both zero-extent and 1-cell.

    Zero-extent gives s0 = w0 = 0.5; the 1-cell formula would give
    w0^2/2 - (1-w1)^2/2 = 0.125 - 0.125 = 0, i.e. a point source that injects
    nothing. Reordering the ladder silently produces the second answer.
    """
    s0, s1, e0, e1 = src._compute_boundary_weights(0.0, 0.0, -1, 1, 10.0)
    assert s0 == pytest.approx(0.5)
    assert (s0, s1, e0, e1) != pytest.approx((0.0, 0.5, 0.5, 0.0))


def test_ivec_loop_weight_follows_meep_test_order():
    """MEEP vec.hpp:372-378 tests is, then is+2, then ie, then ie-2."""
    s0, s1, e0, e1 = 0.1, 0.2, 0.3, 0.4
    assert src._ivec_loop_weight_1d(0, 1, s0, s1, e0, e1) == s0          # single point -> s0
    assert src._ivec_loop_weight_1d(1, 2, s0, s1, e0, e1) == s1          # is+2 wins over ie
    assert src._ivec_loop_weight_1d(4, 5, s0, s1, e0, e1) == e0
    assert src._ivec_loop_weight_1d(3, 5, s0, s1, e0, e1) == e1
    assert src._ivec_loop_weight_1d(3, 8, s0, s1, e0, e1) == 1.0         # interior


def test_gaussian_width_peak_and_cutoff_conventions():
    """MEEP conventions: width = 1/fwidth (not 1/(2*pi*fwidth)), peak = start + cutoff*width."""
    grid = _unit_grid()
    pulse = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Ez',
                                 center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                 start_time=0.0, cutoff=5.0)
    assert pulse.width == pytest.approx(5.0)
    assert pulse.width != pytest.approx(1.0 / (2 * np.pi * 0.2))
    assert pulse.peak_time == pytest.approx(25.0)

    # Carrier referenced to (t - peak_time): at the peak the dipole is exactly amp_factor.
    assert pulse.dipole(pulse.peak_time) == pytest.approx(1.0 / (-2j * np.pi * 0.8))
    # Envelope is exp(-tt^2/(2*width^2)) about the peak.
    assert abs(pulse.dipole(pulse.peak_time - 5.0)) == pytest.approx(
        abs(pulse.dipole(pulse.peak_time)) * math.exp(-0.5))
    # Hard cut beyond cutoff*width past the peak.
    assert pulse.dipole(pulse.peak_time + 25.5) == 0.0
    # Zero has no carrier to correct (1/(-2*pi*i*f) divides by it); negative is
    # MEEP's conjugate carrier and is covered by its own test below.
    with pytest.raises(ValueError):
        GaussianPulsedSource(grid=grid, frequency=0.0, fwidth=0.2, component='Ez',
                             center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))


def test_gaussian_dipole_matches_meep_reference_values():
    """Reference values from meep 1.29.0 gaussian_src_time.dipole (f=0.8, start=0, cutoff=5)."""
    grid = _unit_grid()
    pulse = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Ez',
                                 center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    reference = {
        0.0: 3.631783606241514e-21 + 7.413940919067652e-07j,
        1.0: -1.878728565205145e-06 + 6.104359147105494e-07j,
        12.5: 2.1409214632255034e-17 + 0.00874097521308224j,
        20.0: 1.182180366625591e-16 + 0.1206654407875674j,
        25.0: 0.1989436788648692j,
        26.3: 0.04783096208182026 + 0.18628929781258766j,
        40.0: -6.495716608939711e-18 + 0.0022100646398150207j,
    }
    for time, expected in reference.items():
        assert pulse.dipole(time) == pytest.approx(expected, rel=1e-12, abs=1e-24)

    # Narrower pulse: the same cut, now at |t - 10| > 10.
    narrow = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.5, component='Ez',
                                  center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    assert narrow.dipole(5.0) == pytest.approx(8.563685852902014e-18 + 0.00874097521308224j,
                                               rel=1e-12, abs=1e-24)
    assert narrow.dipole(25.0) == 0.0


def test_negative_frequency_is_meeps_conjugate_carrier():
    """dipole(-f, t) == conj(dipole(+f, t)), bit-exact, for every carrier class.

    MEEP stores the carrier frequency raw — ``gaussian_src_time`` keeps
    ``freq = f`` with no sign restriction (sources.cpp:72-96) — and both the
    carrier ``exp(-i*2*pi*f*(t-t0))`` (sources.cpp:106) and the amplitude
    correction ``1/(-2*pi*i*f)`` (sources.cpp:104) take the sign with it, so a
    negative frequency IS the conjugate waveform: envelope and time window are
    even in f, carrier and amp factor conjugate. Pinned as exact equality
    (measured 0.0e+00 worst over the pulse), not a tolerance: ``exp(+i|w|t)``
    and ``conj(exp(-i|w|t))`` round identically, and anything less than
    equality means a sign leaked somewhere else than the carrier.

    The pair is what MEEP's plus/minus-omega superposition trick drives
    (dipole_in_vacuum_cyl_off_axis.py): GaussianSource(+f) + GaussianSource(-f)
    beat into a REAL current, checked here as vanishing imaginary part.
    """
    grid = _unit_grid()
    times = np.linspace(0.0, 25.0, 251)

    plus_env = GaussianEnvelope(frequency=0.8, fwidth=0.4)
    minus_env = GaussianEnvelope(frequency=-0.8, fwidth=0.4)
    for time in times:
        assert minus_env.dipole(time) == np.conj(plus_env.dipole(time))
    # The superposition is a real dipole: g(+f) + g(-f) = env * sin(w*(t-t0))/(pi*f),
    # so Im == 0 identically — including at the peak, where the SINE vanishes too
    # (t - t0 = 0), which is why the nonzero-real probes sit off the carrier's zeros.
    peak = plus_env.peak_time
    assert (plus_env.dipole(peak) + minus_env.dipole(peak)) == 0.0
    for time in (peak - 1.3, peak + 0.7):
        pair = plus_env.dipole(time) + minus_env.dipole(time)
        assert pair.imag == 0.0
        assert abs(pair.real) > 0.0

    cw_plus = ContinuousEnvelope(frequency=0.8)
    cw_minus = ContinuousEnvelope(frequency=-0.8)
    for time in times[:40]:
        assert cw_minus.dipole(time) == np.conj(cw_plus.dipole(time))

    # The full source classes carry the identical waveform machinery, and a
    # negative-frequency pulse deposits the same spatial table as its twin.
    pulse_plus = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.4, component='Ez',
                                      center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    pulse_minus = GaussianPulsedSource(grid=grid, frequency=-0.8, fwidth=0.4, component='Ez',
                                       center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    assert pulse_minus._indices == pulse_plus._indices
    assert pulse_minus._amplitudes == pulse_plus._amplitudes
    for time in (peak - 0.9, peak, peak + 2.1):
        assert pulse_minus.dipole(time) == np.conj(pulse_plus.dipole(time))
        assert pulse_minus.current(time, grid.dt) == np.conj(pulse_plus.current(time, grid.dt))

    cw_source = ContinuousSource(grid=grid, frequency=-0.8, component='Ez')
    assert cw_source._amp_factor == np.conj(1.0 / (-2j * np.pi * 0.8))

    # Zero and non-finite stay refused for every class that carries a carrier.
    for bad in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonzero"):
            GaussianEnvelope(frequency=bad, fwidth=0.4)
        with pytest.raises(ValueError, match="finite and nonzero"):
            ContinuousEnvelope(frequency=bad)


def test_continuous_source_turn_on_ramp():
    """MEEP continuous_src_time: (1+tanh(ts))*(1+tanh(te))/4 with slowness 3.

    Reference values from meep 1.29.0 (f=0.8, width=2, start=0, end=100).
    """
    grid = _unit_grid()
    ramped = ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                              width=2.0, slowness=3.0, start_time=0.0, end_time=100.0)
    reference = {
        0.0: 0.0004919127472273864j,
        0.5: 0.0004759462770956021 - 0.00065508385090995j,
        3.0: 0.0055457995164916035 - 0.00763313818905681j,
        7.0: -0.08548718326626531 - 0.11766301347969377j,
        20.0: -7.796337182160481e-16 + 0.1989435134376243j,
    }
    for time, expected in reference.items():
        assert ramped.dipole(time) == pytest.approx(expected, rel=1e-12, abs=1e-24)

    steady = 1.0 / (2 * np.pi * 0.8)
    assert abs(ramped.dipole(0.0)) < 0.01 * steady        # starts near zero
    assert abs(ramped.dipole(50.0)) == pytest.approx(steady, rel=1e-9)  # reaches full amplitude
    assert ramped.dipole(-1.0) == 0.0                     # before start_time
    assert ramped.dipole(101.0) == 0.0                    # after end_time
    assert abs(ramped.dipole(99.5)) < 0.01 * steady       # turn-off ramp is symmetric

    # MEEP's default (width = 0) is the hard switch-on: full amplitude at t = 0.
    hard = ContinuousSource(grid=grid, frequency=0.8, component='Ez')
    assert hard.dipole(0.0) == pytest.approx(0.1989436788648692j, rel=1e-12)


def test_current_is_the_forward_difference_of_dipole():
    grid = _unit_grid()
    for source in (
        ContinuousSource(grid=grid, frequency=0.8, component='Ez', width=2.0, end_time=100.0),
        ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                       center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0)),
        GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Ez',
                             center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0)),
    ):
        dt = grid.dt
        expected = (source.dipole(3.0 + dt) - source.dipole(3.0)) / dt
        assert source.current(3.0, dt) == pytest.approx(expected)


def test_extended_source_injects_expected_values_at_expected_cells():
    """Value-level injection: D -= dt * current * stored amplitude, nowhere else."""
    grid = _unit_grid(resolution=10.0)
    fields = _StubFields(grid)
    source = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    time = 1.25
    current = source.current(time, grid.dt)
    source.inject(fields, time)

    expected = -grid.dt * current * 500.0
    assert fields.Dz[10, 10, 9] == pytest.approx(expected, rel=1e-6)
    assert fields.Dz[10, 10, 10] == pytest.approx(expected, rel=1e-6)
    touched = np.zeros(grid.shape, dtype=bool)
    touched[10, 10, 9] = touched[10, 10, 10] = True
    assert np.all(fields.Dz[~touched] == 0)
    assert np.all(fields.Dx == 0) and np.all(fields.Dy == 0)

    # A second call accumulates rather than overwriting.
    source.inject(fields, time)
    assert fields.Dz[10, 10, 9] == pytest.approx(2 * expected, rel=1e-6)


def test_continuous_point_source_matches_extended_point_source():
    """The CW point class and the extended class agree cell for cell and value for value.

    Both now place through `_build_source_points`, so the interesting half of this is
    the amplitude bookkeeping either side of it: `ExtendedSource` folds the user
    amplitude into the per-point table and `ContinuousSource` keeps it in `dipole()`,
    and only one of the two may also carry the `a` per zero-size axis. A double count
    or a dropped factor shows up here as a clean factor of 10 (or 1000).
    """
    grid = _unit_grid(resolution=10.0)
    time = 0.75
    # Both an on-sample point and one at a cell face, where the classes used to differ:
    # the extended path folded onto the lattice and the CW path extrapolated off it.
    for center in ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.35)):
        cw_fields = _StubFields(grid)
        ext_fields = _StubFields(grid)
        cw = ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                              center=center, size=(0.0, 0.0, 0.0), amplitude=0.5 - 0.25j)
        ext = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                             center=center, size=(0.0, 0.0, 0.0), amplitude=0.5 - 0.25j)
        assert sorted(cw._point_indices) == sorted(ext._indices)
        cw.inject(cw_fields, time)
        ext.inject(ext_fields, time)
        assert np.count_nonzero(ext_fields.Dz) > 0, "Neither source deposited anything to compare."
        np.testing.assert_allclose(cw_fields.Dz, ext_fields.Dz, rtol=1e-6, atol=1e-12)


# --- point-source placement across a cell, including its faces -------------------
#
# These use the real `Grid` rather than `_StubGrid`: the whole subject is the
# registration of a request against the lattice, and the half-cell origin parity of an
# odd-count axis is exactly the thing a stub would paper over.


def _deposition(source):  # {cell index: complex amplitude} for one source's point table.
    indices = source._point_indices if hasattr(source, "_point_indices") else source._indices
    weights = source._point_weights if hasattr(source, "_point_weights") else source._amplitudes
    assert len(set(indices)) == len(indices), "A cell appears twice; injection would drop one."
    return dict(zip(indices, weights))


def _cell_positions(grid, axis, cell_index, count=9):  # Positions spanning one full cell, faces included.
    low = grid.axis_origin(axis) + cell_index * grid.dx
    return [low + step * grid.dx / (count - 1) for step in range(count)]


def test_point_source_at_a_cell_face_wraps_onto_the_lattice():
    """A request at a face is a lattice request, not a request off the end of the array.

    THE REGRESSION. The CW point source used to place this through
    `Grid.position_to_index`, which tolerated a cell of overhang and returned an
    extrapolating weight; the stencil then dropped the negative half and left 1.5x the
    requested current on one cell (2x on an integer-placed axis), with nothing said.
    """
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0))  # 20 cells/axis, dx = 0.1.
    whole = grid.resolution ** 3  # MEEP's a per zero-size axis: a point source is a^3.

    # Ez is HALF-INTEGER in z, so the faces at +-1.0 fall between two z samples and the
    # partner across the face is the sample at the other end of the axis.
    for face, expected_cells in ((-1.0, {0, grid.nz - 1}), (1.0, {grid.nz - 1, 0})):
        deposit = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                                               center=(0.0, 0.0, face)))
        assert {cell[2] for cell in deposit} == expected_cells
        assert {cell[:2] for cell in deposit} == {(10, 10)}  # x = y = 0 land on samples.
        for amplitude in deposit.values():
            assert amplitude == pytest.approx(0.5 * whole)

    # Ez is INTEGER-placed in x, so the faces at +-0.5*L ARE samples — the same sample,
    # one lattice vector apart — and the deposit is a single cell of full weight.
    for face, expected_cell in ((-1.0, 0), (1.0, 0)):
        deposit = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                                               center=(face, 0.0, 0.0)))
        assert set(deposit) == {(expected_cell, 10, 9), (expected_cell, 10, 10)}
        assert sum(deposit.values()) == pytest.approx(whole)


@pytest.mark.parametrize("cell_size, shape", [((2.0, 2.0, 2.0), (20, 20, 20)),
                                              ((1.9, 2.0, 1.9), (19, 20, 19))])
def test_a_point_source_anywhere_in_a_cell_deposits_exactly_one_point_source(cell_size, shape):
    """Sweep a full cell at both ends of both Yee placements; the total never moves.

    Conservation is the invariant the extrapolating stencil broke, and it breaks
    loudly: 1.5x at a half-integer face, 2x at an integer one. Run at an ODD cell count
    as well as an even one, because MEEP rounds an axis's little corner down to even
    and an odd axis therefore starts half a cell higher than -L/2 — a registration an
    even-only sweep cannot see.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size)
    assert grid.shape == shape, "The parametrisation is meant to pin one odd and one even count."
    whole = grid.resolution ** 3
    for axis in (0, 2):  # Ez: integer-placed in x, half-integer in z.
        for cell_index in (0, grid.shape[axis] - 1):  # The two cells that touch a face.
            for position in _cell_positions(grid, axis, cell_index):
                center = [0.0, 0.0, 0.0]
                center[axis] = position
                deposit = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                                                       center=tuple(center)))
                weights = list(deposit.values())
                where = f"axis {axis}, cell {cell_index}, position {position:.4f}"
                assert all(weight.imag == 0.0 and weight.real > 0.0 for weight in weights), (
                    f"A k = 0 deposition must be positive and real ({where}); a negative weight "
                    f"is the extrapolation signature."
                )
                assert sum(weights).real == pytest.approx(whole, rel=1e-9), (
                    f"Deposited {sum(weights).real:.6g} instead of {whole:.6g} at {where}."
                )
                assert max(weight.real for weight in weights) <= whole * (1 + 1e-9), (
                    f"One cell took more than the whole point source at {where}."
                )


def test_a_single_cell_axis_folds_both_ends_of_the_stencil_onto_its_one_cell():
    """The degenerate wrap: an axis one cell deep, where both neighbours are the cell.

    This must not score as perfect by accident. A clipping placement also lands
    everything on cell 0 and would look identical if the only assertion were "one cell,
    index 0" — so the total is asserted too, and the same request is checked at a face,
    at the sample, and off it. A stencil that folded without summing would leave half.

    THE TOTAL IS ``a**2``, NOT ``a**3``, and that is MEEP's own arithmetic rather than a
    weakened bound. ``add_volume_source`` multiplies the amplitude by ``gv.a`` once per
    zero-size direction *except* a direction ``fields::nosize_direction`` reports
    (sources.cpp:482), and a one-pixel PERIODIC axis is exactly what that function
    picks out — MEEP's comment says such an axis is "used almost exclusively to emulate
    lower-dimensional computations", so a zero extent there means the whole direction
    rather than a delta function. This test pinned ``a**3`` while the clause was
    missing, which put the resolution as a factor on every source in such a run:
    measured **9.000e+00** complex relative L2 against CPU MEEP on this very cell
    (0.1 x 2 x 2 at resolution 10, CW Ez point source, k_point set), against
    **2.106e-07** with the clause restored.
    """
    grid = Grid(resolution=10.0, cell_size=(0.1, 2.0, 2.0))  # nx = 1: x is one cell deep.
    assert grid.nx == 1 and grid.axis_wraps(0)
    assert grid.nosize_direction(0) and not grid.nosize_direction(1)
    whole = grid.resolution ** 2  # a**3 for the two delta directions, /a for the emulated one.
    for x in (grid.axis_origin(0), grid.axis_origin(0) + 0.5 * grid.dx,
              grid.axis_origin(0) + grid.dx, grid.axis_origin(0) + 0.25 * grid.dx):
        deposit = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                                               center=(x, 0.0, 0.0)))
        assert {cell[0] for cell in deposit} == {0}
        assert sum(deposit.values()).real == pytest.approx(whole, rel=1e-9), (
            f"A one-cell axis lost part of the source at x = {x:.4f}: the two ends of the "
            f"stencil fold onto the same cell and have to be summed, not overwritten."
        )


def test_the_delta_scaling_skips_only_a_one_pixel_periodic_axis():
    """MEEP's ``nosize_direction`` clause, on the three axes that are not it.

    ``amp *= gv.a`` per zero-size direction is what makes a point source a current
    DENSITY; skipping it where it should not be skipped divides the source by the
    resolution just as surely as applying it where it should not multiplies it. The
    clause fires on a one-pixel PERIODIC axis and on nothing else, so this walks the
    three ways an axis can hold one cell and still not qualify — walled, folded, and
    simply wider than a cell — against the one that does.
    """
    resolution = 10.0
    whole = resolution ** 3

    def total(grid):
        # Ex is half-integer in x, so this centre is exactly its own sample plane on
        # every grid below: the whole source lands on one cell and the total measures the
        # delta scaling alone, with no clipping or interpolation mixed into it.
        centre = (grid.axis_origin(0) + 0.5 * grid.dx, 0.0, 0.0)
        return sum(_deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ex',
                                                center=centre)).values()).real

    # Two cells, not one: MEEP forces a UNIT direction Periodic whatever k_point says
    # (fields.cpp, "unit directions are periodic by default"), so a one-cell metallic
    # axis is a run neither code has and Grid refuses it.
    walled = Grid(resolution=resolution, cell_size=(0.2, 2.0, 2.0), boundaries={"x": "metallic"})
    assert walled.nx == 2 and not walled.nosize_direction(0)
    assert total(walled) == pytest.approx(whole, rel=1e-9), (
        "A METALLIC axis is not a lower-dimensional emulation — MEEP's nosize_direction "
        "requires Periodic on both faces — so its delta scaling stands."
    )

    # halve() keeps N - N//2 + 1 = 2 owned cells and the even periodic fold stores the
    # +L/2 plane cell besides (Grid.stored_cells), so the axis is 3 cells — and its low
    # face is the mirror, not Periodic, so the clause cannot fire from the count or the
    # boundary reading.
    folded = Grid(resolution=resolution, cell_size=(0.2, 2.0, 2.0), symmetry=("X",))
    assert folded.nx == 3 and folded.owned_cells(0) == 2 and not folded.nosize_direction(0)

    wide = Grid(resolution=resolution, cell_size=(2.0, 2.0, 2.0))
    assert not any(wide.nosize_direction(axis) for axis in range(3))
    assert total(wide) == pytest.approx(whole, rel=1e-9), (
        "An ordinary 3-D cell must reach a**3 exactly as it always did; the clause is a "
        "no-op on every axis with more than one cell."
    )

    reduced = Grid(resolution=resolution, cell_size=(2.0, 2.0, 0.0), dimensions=2)
    assert reduced.nosize_direction(2) and not reduced.nosize_direction(0)
    assert total(reduced) == pytest.approx(resolution ** 2, rel=1e-9), (
        "A 2-D point source is a**2, which is what MEEP's LOOP_OVER_DIRECTIONS reaches "
        "when z is not one of its directions at all."
    )

    thin = Grid(resolution=resolution, cell_size=(2.0, 2.0, 0.1))
    assert thin.nz_full == 1 and thin.nosize_direction(2) and not thin.has_invariant
    assert total(thin) == pytest.approx(resolution ** 2, rel=1e-9), (
        "MEEP's clause keys on the CELL COUNT and the boundary condition, not on "
        "dimensions: a 3-D run with a one-pixel periodic z axis gets the same treatment, "
        "which is why this engine reproduces such a run at 2.106e-07 rather than 9.0e+00."
    )


def test_the_lattice_wrap_carries_the_bloch_phase():
    """The wrapped cell is phased, and phased the right way round.

    A wrap that dropped the factor, or applied its conjugate, changes nothing about the
    magnitudes — this is the failure mode that survives every |field| comparison.
    """
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), k_point=(0.0, 0.0, 0.3))
    phase = grid.bloch_phase(2)
    assert phase == pytest.approx(complex(np.exp(2j * np.pi * 0.3 * 2.0)))
    half = 0.5 * grid.resolution ** 3

    # z = -1.0 splits between cell 0 (its own cell) and cell nz-1, which is one lattice
    # vector BELOW the request. Representing a source specified at the lower position
    # inside the cell divides by the wrap factor — MEEP's `amp * conj(shift_phase)` with
    # shift = -1, i.e. conj(phase**-1) = phase.
    deposit = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                                           center=(0.0, 0.0, -1.0)))
    assert deposit[(10, 10, 0)] == pytest.approx(half)
    assert deposit[(10, 10, grid.nz - 1)] == pytest.approx(half * phase)
    assert deposit[(10, 10, grid.nz - 1)] != pytest.approx(half), "The wrap dropped the Bloch factor."
    assert deposit[(10, 10, grid.nz - 1)] != pytest.approx(half * np.conj(phase)), (
        "The wrap applied the conjugate of the Bloch factor; magnitudes cannot tell."
    )

    # The face one lattice vector the other way takes the conjugate, and the two faces
    # are consistent: they are the same physical point, so the two depositions differ by
    # exactly one factor of the wrap.
    upper = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                                         center=(0.0, 0.0, 1.0)))
    assert upper[(10, 10, 0)] == pytest.approx(half * np.conj(phase))
    assert upper[(10, 10, grid.nz - 1)] == pytest.approx(half)
    for cell in ((10, 10, 0), (10, 10, grid.nz - 1)):
        assert upper[cell] == pytest.approx(deposit[cell] * np.conj(phase))


def test_a_wrapping_axis_and_a_folded_axis_behave_differently_in_one_source():
    """A face placement wraps on z and clips on a folded x, in the same source.

    The two boundary kinds meet inside one deposition, which is where a rule applied
    per-source rather than per-axis would go wrong.
    """
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), symmetry=('X',))
    assert (grid.axis_wraps(0), grid.axis_wraps(1), grid.axis_wraps(2)) == (False, True, True)
    deposit = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                                           center=(0.0, 0.0, -1.0)))
    # Folded x: integer-placed components sit at (i-1)*dx, so the mirror plane is cell 1
    # and the source stays on it — the fold is a boundary condition, not a lattice image.
    assert {cell[0] for cell in deposit} == {1}
    # Wrapping z: the face still splits across the two ends of the axis.
    assert {cell[2] for cell in deposit} == {0, grid.nz - 1}
    assert sum(deposit.values()) == pytest.approx(grid.resolution ** 3)


def test_continuous_point_source_in_the_discarded_half_is_refused():
    """A CW point used to skip the fold check every other source class runs.

    THE RULE MOVED, and this test with it. It used to require the source ON the
    plane, which was stricter than MEEP by a whole class of run: a folded grid
    stores one quadrant and reconstructs the rest, so a source anywhere in the
    STORED quadrant is two sources in the unfolded domain — which is what
    ``mp.Mirror`` asserts, and what MEEP steps (``use_symmetry=false``,
    sources.cpp:487, images implicit). What is still refused is the DISCARDED
    quadrant, where MEEP deposits nothing at all and returns a peak of exactly 0.
    """
    grid = _unit_grid(resolution=10.0, symmetry=('X', 'Y'))
    ContinuousSource(grid=grid, frequency=0.8, component='Ez', center=(0.0, 0.0, 0.3))
    # In the stored half, off the plane: legal, and reproduced at the fold floor.
    ContinuousSource(grid=grid, frequency=0.8, component='Ez', center=(0.3, 0.0, 0.0))
    ContinuousSource(grid=grid, frequency=0.8, component='Ez', center=(0.0, 0.3, 0.0))
    with pytest.raises(ValueError, match="X symmetry the stored half"):
        ContinuousSource(grid=grid, frequency=0.8, component='Ez', center=(-0.3, 0.0, 0.0))
    with pytest.raises(ValueError, match="Y symmetry the stored half"):
        ContinuousSource(grid=grid, frequency=0.8, component='Ez', center=(0.0, -0.3, 0.0))


def test_a_source_wider_than_the_cell_is_refused_and_a_sub_pixel_overhang_is_snapped():
    """MEEP sources.cpp:467-476, both branches.

    The ladder folds every rung onto the cell modulo the lattice, so a request spanning
    more than one period lands two lattice images on the same cells and sums them —
    a silently over-driven source. MEEP aborts past one pixel of overhang and snaps
    what is inside it; both are reproduced, and the snapped run is cross-validated
    against CPU MEEP at 1.5e-7 (see the placement sweep's sibling measurement).
    """
    grid = _unit_grid(resolution=10.0)  # 2 x 2 x 2, dx = 0.1, so the snap band is (2.0, 2.1].

    def sheet(width, offset=0.0, amp_func=None):  # x-extended, z-normal sheet.
        return ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                              center=(offset, 0.0, 0.0), size=(width, 0.0, 0.0),
                              amp_func=amp_func)

    def ramp(x, y, z):  # Varies along the over-wide axis, so WHERE the window lands shows.
        del y, z
        return complex(1.0 + 3.0 * x)

    exact = sum(abs(amplitude) for amplitude in sheet(2.0)._amplitudes)
    for width in (2.05, 2.1):  # Within one pixel: snapped back to exactly one period.
        assert sum(abs(a) for a in sheet(width)._amplitudes) == pytest.approx(exact, rel=1e-9), (
            f"A {width} wide request on a 2.0 cell deposited a different total than the "
            f"cell-wide one; it wrapped onto itself instead of being snapped."
        )
        # WHERE the snapped window lands, which a UNIFORM cell-spanning sheet cannot
        # show: it deposits 1 per cell however the window is offset. With an amplitude
        # that varies along the axis it does show, and MEEP's arithmetic is asymmetric —
        # the minimum moves DOWN by half the excess and the maximum is then placed one
        # cell length above the NEW minimum, so the whole window slides down by the FULL
        # excess instead of staying centred (sources.cpp:471-474). CPU MEEP confirms it:
        # widths 2.0 / 2.05 / 2.1 with this ramp all match at 1.3e-7, and MEEP's own
        # answers for 2.0 and 2.05 differ by 1.4e-1, so the case really discriminates.
        excess = width - 2.0
        assert _deposition(sheet(width, amp_func=ramp)) == pytest.approx(
            _deposition(sheet(2.0, offset=-excess, amp_func=ramp))
        ), f"The {width} wide request snapped to a different window than MEEP's."
        assert _deposition(sheet(width, amp_func=ramp)) != pytest.approx(
            _deposition(sheet(2.0, amp_func=ramp))
        ), "The snap left the window centred on the request; MEEP's slides it down."
    for width in (2.10001, 2.2, 4.0):  # Past one pixel: MEEP aborts, and so does this.
        with pytest.raises(ValueError, match="exceeds the cell length"):
            sheet(width)
    # The guard is per axis and reads the FULL cell length, so a folded axis is measured
    # against the whole cell rather than the stored quadrant.
    folded = _unit_grid(resolution=10.0, symmetry=('X',))
    ExtendedSource(grid=folded, frequency=0.8, component='Ez', center=(0.0, 0.0, 0.0),
                   size=(2.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="exceeds the cell length"):
        ExtendedSource(grid=folded, frequency=0.8, component='Ez', center=(0.0, 0.0, 0.0),
                       size=(2.2, 0.0, 0.0))


def test_gaussian_source_injects_only_inside_the_pulse():
    grid = _unit_grid(resolution=10.0)
    fields = _StubFields(grid)
    pulse = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Ex',
                                 center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    pulse.inject(fields, 1000.0)  # far past the cutoff
    assert np.all(fields.Dx == 0)
    pulse.inject(fields, pulse.peak_time)
    assert np.count_nonzero(fields.Dx) == pulse._n_source_points
    assert np.all(fields.Dy == 0) and np.all(fields.Dz == 0)


def test_injection_refuses_a_storage_dtype_that_is_neither_complex64_nor_float32():
    """float32 and complex64 are the two supported widths; anything else is a caller bug.

    complex128 in particular would run — NumPy narrows on the in-place subtract — and
    silently cost twice the memory of the mode the caller thought they had asked for.
    """
    grid = _unit_grid(resolution=10.0)
    ext = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                         center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    cw = ContinuousSource(grid=grid, frequency=0.8, component='Ez')
    pulse = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Ez',
                                 center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    wide_fields = _StubFields(grid, dtype=np.complex128)
    for source in (ext, cw, pulse):
        with pytest.raises(ValueError, match="complex64 or float32"):
            source.inject(wide_fields, 1.0)


def test_injection_into_real_fields_takes_the_real_part_of_the_current():
    """MEEP step.cpp:298 — ``f[c][0][i] -= real(A)``, with the imaginary plane absent.

    Asserted against the complex run's own real part with zero tolerance, per source
    class, because the two paths form the same product and only the last step differs.
    A source that instead injected ``|A|`` or ``Re(amp)*Re(current)`` would still look
    like a working source: same support, same order of magnitude, wrong phase.
    """
    grid = _unit_grid(resolution=10.0)
    sources = {
        'extended': lambda: ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                                           center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0)),
        'continuous': lambda: ContinuousSource(grid=grid, frequency=0.8, component='Ez'),
        'gaussian': lambda: GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2,
                                                 component='Ez', center=(0.0, 0.0, 0.0),
                                                 size=(0.4, 0.4, 0.0)),
        'volume': lambda: VolumeSource(grid=grid, component='Hx', center=(0.0, 0.0, 0.0),
                                       size=(0.0, 0.0, 0.0),
                                       envelope=ContinuousEnvelope(frequency=0.8)),
    }
    for label, build in sources.items():
        real_fields = _StubFields(grid, dtype=np.float32)
        complex_fields = _StubFields(grid, dtype=np.complex64)
        array = 'Bx' if label == 'volume' else 'Dz'
        for time in (0.4, 0.9, 1.7):
            build().inject(real_fields, time)
            build().inject(complex_fields, time)
        real_values = getattr(real_fields, array)
        complex_values = getattr(complex_fields, array)
        assert real_values.dtype == np.float32, f"{label} promoted the storage"
        assert np.count_nonzero(real_values) > 0, f"{label} injected nothing at all"
        np.testing.assert_array_equal(
            real_values, complex_values.real,
            err_msg=f"{label}: real injection is not the real part of the complex one",
        )
        # The imaginary part is a real quantity that was dropped, not an empty one:
        # if it were zero the equality above would be trivially satisfiable.
        assert np.count_nonzero(complex_values.imag) > 0, f"{label}: nothing was dropped"


def test_real_injection_of_a_phased_spatial_amplitude_is_the_complex_run_s_real_part():
    """A phased spatial amplitude is PROJECTED, not refused — MEEP's step.cpp:307.

    ``Re(a*s) = Re(a)Re(s) - Im(a)Im(s)`` re-reads the caller's spatial phase as a
    per-point time shift, and this module used to refuse the pairing on exactly that
    ground. MEEP takes ``real(A)`` with no condition attached, and a Gaussian beam's
    amplitude is complex BY CONSTRUCTION (wavefront curvature and the Gouy phase), so
    the refusal declined a whole class of runs MEEP completes — its own
    ``gaussian-beam.py`` among them — rather than one exotic input.

    What the projection is worth is measured end to end against CPU MEEP in
    ``test_from_meep.py`` (``beam_real_fields`` 5.4e-07, its pulsed twin 2.2e-07,
    ``eigenmode_oblique_waveguide_real_fields`` 1.2e-06, each on the bound its complex
    twin holds). What is pinned HERE is the arithmetic those numbers rest on: for every
    source spelling that can carry phase, the real deposit is the complex deposit's
    real part EXACTLY, so the real run is that run's real part rather than a different
    source that resembles it.
    """
    grid = _unit_grid(resolution=10.0)
    volume = dict(center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    # Every route by which phase reaches the deposition table, and the array each
    # drives. The beam is the one that made this a policy question rather than an
    # exotic-input question.
    phased = {
        'complex amplitude': ('Dz', lambda: ExtendedSource(
            grid=grid, frequency=0.8, component='Ez', amplitude=1j, **volume)),
        'complex amp_func': ('Dz', lambda: ExtendedSource(
            grid=grid, frequency=0.8, component='Ez',
            amp_func=lambda x, y, z: 1.0 + 0.5j, **volume)),
        'pulsed': ('Dz', lambda: GaussianPulsedSource(
            grid=grid, frequency=0.8, fwidth=0.2, component='Ez',
            amplitude=0.5 + 0.5j, **volume)),
        'continuous point': ('Dz', lambda: ContinuousSource(
            grid=grid, frequency=0.8, component='Ez', amplitude=2j)),
        'magnetic volume': ('Bx', lambda: VolumeSource(
            grid=grid, component='Hx', amplitude=1j,
            envelope=ContinuousEnvelope(frequency=0.8), **volume)),
        'gaussian beam': ('Dz', lambda: src.create_gaussian_beam_source(
            grid=grid, frequency=0.8, waist_radius=0.3, source_z=-0.5,
            focus_position=(0.0, 0.0, 0.5), source_size=(0.6, 0.6), component='Ez')),
    }
    for label, (array, build) in phased.items():
        real_fields = _StubFields(grid, dtype=np.float32)
        complex_fields = _StubFields(grid, dtype=np.complex64)
        for time in (0.4, 0.9, 1.7):
            build().inject(real_fields, time)
            build().inject(complex_fields, time)
        real_values = getattr(real_fields, array)
        complex_values = getattr(complex_fields, array)
        assert real_values.dtype == np.float32, f"{label}: the storage was promoted"
        assert np.count_nonzero(real_values) > 0, f"{label}: nothing was injected at all"
        np.testing.assert_array_equal(
            real_values, complex_values.real,
            err_msg=f"{label}: the real deposit is not the complex deposit's real part",
        )
        # The dropped plane is a real quantity, not an empty one — otherwise the
        # equality above would hold for a source carrying no phase in the first place.
        assert np.count_nonzero(complex_values.imag) > 0, f"{label}: nothing was dropped"

    # THE CONTROL, and the mistake actually worth separating: realifying the AMPLITUDE
    # and then taking the real part of the product — `Re(a)*s` where MEEP deposits
    # `Re(a*s)`. It is the one-line "real storage cannot carry a complex amplitude"
    # fix, it deposits into the same cells with the same envelope, and it is a
    # different current. Spelled on the amp_func route because that is the one where
    # Re(a) is not degenerate.
    correct = _StubFields(grid, dtype=np.float32)
    realified = _StubFields(grid, dtype=np.float32)
    ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                   amp_func=lambda x, y, z: 1.0 + 0.5j, **volume).inject(correct, 0.9)
    ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                   amp_func=lambda x, y, z: 1.0 + 0.0j, **volume).inject(realified, 0.9)
    deposited = float(np.abs(correct.Dz).max())
    difference = float(np.abs(correct.Dz - realified.Dz).max())
    assert difference > 0.1 * deposited, (
        f"Re(a)*s and Re(a*s) differ by only {difference:.3e} of a {deposited:.3e} "
        f"deposit, so this comparison cannot see which of the two was injected"
    )


def test_source_points_have_unique_linear_indices():
    """The vectorized injection is only correct because the points are unique."""
    grid = _unit_grid(resolution=10.0)
    source = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                            center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.2))
    linear = [(x * grid.ny + y) * grid.nz + z for (x, y, z) in source._indices]
    assert len(set(linear)) == len(linear)

    previous = src.DEBUG_VALIDATE_SOURCE_INDICES
    src.DEBUG_VALIDATE_SOURCE_INDICES = True
    try:
        checked = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                                 center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.2))
        assert checked._n_source_points == source._n_source_points
    finally:
        src.DEBUG_VALIDATE_SOURCE_INDICES = previous


def test_amp_func_receives_all_three_relative_offsets():
    """amp_func(x_rel, y_rel, z_rel) — the z offset is passed, not dropped."""
    grid = _unit_grid(resolution=10.0)
    seen = []

    def amp_func(x_rel, y_rel, z_rel):  # Record the offsets and weight by z.
        seen.append((x_rel, y_rel, z_rel))
        return 1.0 + z_rel

    geometry = dict(grid=grid, frequency=0.8, component='Ez',
                    center=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2))
    source = ExtendedSource(amp_func=amp_func, **geometry)
    plain = ExtendedSource(**geometry)
    assert seen, "amp_func was never evaluated"
    z_offsets = {round(z, 6) for (_, _, z) in seen}
    assert len(z_offsets) > 1, "z offsets must vary across a volume source"
    assert max(abs(z) for z in z_offsets) > 0.0

    # Point for point, the amplitude carries the z-dependent factor the function
    # returned — proof the z offset reaches amp_func rather than being dropped.
    assert source._indices == plain._indices
    plain_amplitude = dict(zip(plain._indices, plain._amplitudes))
    ratios_by_iz = {}
    for (index, weighted) in zip(source._indices, source._amplitudes):
        ratio = round((weighted / plain_amplitude[index]).real, 9)
        ratios_by_iz.setdefault(index[2], set()).add(ratio)
    assert all(len(ratios) == 1 for ratios in ratios_by_iz.values())  # ratio depends on z only
    applied = {next(iter(ratios)) for ratios in ratios_by_iz.values()}
    assert len(applied) == len(ratios_by_iz) > 1  # a distinct factor per z plane
    assert applied <= {round(1.0 + offsets[2], 9) for offsets in seen}

    beam = create_gaussian_beam_source(
        grid=grid, frequency=0.8, waist_radius=0.3, source_z=-0.5,
        focus_position=(0.0, 0.0, 0.5), source_size=(0.4, 0.4), component='Ex')
    assert beam._n_source_points > 0  # the bundled factory takes three arguments too


def test_gaussian_pulse_reuses_the_extended_source_geometry():
    grid = _unit_grid(resolution=10.0)
    geometry = dict(grid=grid, frequency=0.8, component='Ey',
                    center=(0.0, 0.0, 0.1), size=(0.4, 0.0, 0.2))
    cw = ExtendedSource(**geometry)
    pulse = GaussianPulsedSource(fwidth=0.2, **geometry)
    assert pulse._indices == cw._indices
    assert pulse._amplitudes == cw._amplitudes


def test_unknown_components_raise():
    """No silent 'Ez' fallback anywhere (degenerate-returns discipline)."""
    grid = _unit_grid()
    fields = _StubFields(grid)
    with pytest.raises(ValueError, match="component"):
        ContinuousSource(grid=grid, frequency=0.8, component='Bogus')
    with pytest.raises(ValueError, match="component"):
        ExtendedSource(grid=grid, frequency=0.8, component='Hx',
                       center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    with pytest.raises(ValueError, match="component"):
        GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Centered',
                             center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    with pytest.raises(ValueError, match="component"):
        src._d_array_for(fields, 'Hz')
    with pytest.raises(ValueError, match="axis"):
        src._iyee_shift('Ez', 3)

    # A component switched to an unsupported value after setup still raises at injection.
    good = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                          center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    good.component = 'Hy'
    with pytest.raises(ValueError, match="component"):
        good.inject(fields, 1.0)


def test_symmetry_requires_sources_on_the_mirror_plane():
    grid = _unit_grid(resolution=10.0, symmetry=('X', 'Y'))
    assert grid.nx == grid.nx_full // 2 + 1

    on_plane = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                              center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    assert on_plane._n_source_points > 0
    # The stored quadrant only holds the +x/+y half, so the source is clipped to it.
    assert max(ix for (ix, _, _) in on_plane._indices) < grid.nx
    assert max(iy for (_, iy, _) in on_plane._indices) < grid.ny

    # Off the plane but inside the STORED quadrant: legal, as MEEP has it.
    ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                   center=(0.3, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    with pytest.raises(ValueError, match="X symmetry the stored half"):
        ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                       center=(-0.3, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    with pytest.raises(ValueError, match="Y symmetry the stored half"):
        ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                       center=(0.0, -0.3, 0.0), size=(0.4, 0.4, 0.0))
    with pytest.raises(ValueError, match="X symmetry the stored half"):
        GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Ez',
                             center=(-0.3, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    # Within dx/10 of the plane is accepted.
    ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                   center=(0.005, 0.0, 0.0), size=(0.4, 0.4, 0.0))


def test_source_outside_the_stored_grid_is_empty_and_injects_nothing():
    grid = _unit_grid(resolution=10.0, symmetry=('X',))
    fields = _StubFields(grid)
    # Far along -x: entirely in the mirrored half that is not stored.
    source = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    source._indices, source._amplitudes = [], []
    source._setup_index_arrays()
    assert source._n_source_points == 0
    source.inject(fields, 1.0)
    assert np.all(fields.Dz == 0)


def test_source_arrays_are_complex64_on_the_numpy_backend():
    grid = _unit_grid(resolution=10.0)
    source = ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                            center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    assert source._point_amps.dtype == np.complex64
    assert source._point_ix.dtype == np.int64
    assert source._point_ix.shape == (source._n_source_points,)


# --- Magnetic (H/B) components -------------------------------------------------

def test_iyee_shift_of_magnetic_components_is_the_complement_of_the_electric_one():
    """MEEP vec.hpp:1132-1140 — magnetic components carry the shift on the OTHER two axes."""
    assert [src._iyee_shift('Hx', d) for d in range(3)] == [0, 1, 1]
    assert [src._iyee_shift('Hy', d) for d in range(3)] == [1, 0, 1]
    assert [src._iyee_shift('Hz', d) for d in range(3)] == [1, 1, 0]
    # The B spellings name the same staggering as their H names.
    for magnetic, primary in (('Hx', 'Bx'), ('Hy', 'By'), ('Hz', 'Bz')):
        assert [src._iyee_shift(magnetic, d) for d in range(3)] == [
            src._iyee_shift(primary, d) for d in range(3)]
    # An electric component keeps its own axis, so the two families never agree.
    assert [src._iyee_shift('Ex', d) for d in range(3)] == [1, 0, 0]


def test_field_type_places_each_component_in_the_right_half_step_slot():
    """MEEP step.cpp:64-100 — B sources are driven at time(), D sources at time()+dt/2."""
    grid = _unit_grid(resolution=10.0)
    point = dict(center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    magnetic = VolumeSource(grid=grid, component='Hy',
                            envelope=ContinuousEnvelope(frequency=0.8), **point)
    electric = VolumeSource(grid=grid, component='Ez',
                            envelope=ContinuousEnvelope(frequency=0.8), **point)
    assert magnetic.field_type == src.FIELD_TYPE_B == 'B'
    assert electric.field_type == src.FIELD_TYPE_D == 'D'
    # The classes that predate magnetic support all belong to the D slot.
    assert ContinuousSource(grid=grid, frequency=0.8, component='Ez').field_type == 'D'
    assert ExtendedSource(grid=grid, frequency=0.8, component='Ez',
                          center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0)).field_type == 'D'
    assert GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Ez',
                                center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0)).field_type == 'D'
    for component in ('Hx', 'Hy', 'Hz', 'Bx', 'By', 'Bz'):
        assert src._field_type_for(component) == 'B'
    with pytest.raises(ValueError, match="component"):
        src._field_type_for('Centered')


def test_magnetic_point_source_lands_on_hand_computed_cells():
    """Hand-worked loop_in_chunks trace for an Hx point source at the origin.

    Grid: resolution 10, 2x2x2 cell, no symmetry -> 20 cells/axis, dx = 0.1.
    iyee_shift(Hx) = (0,1,1), so iyee_c = (1,0,0) and yee_c = (0.05, 0, 0).
    X: vec2diel_floor(0.05, 10) = 1, minus iyee_c -> is = ie = 0 (one point, weight 1).
    Y/Z: vec2diel_floor(0, 10) = -1, ceil = 1 -> two points each, weights 0.5.
    Indices: (is + 2i - io)/2 with io = -20 -> x = 10, y and z in {9, 10}.
    Amplitude: 1 * 0.5 * 0.5 * 10^3 = 250 at each of the four points.

    This is the mirror image of the Ez trace above: the Ez source splits along z
    and is sharp in x/y, the Hx source splits along y and z and is sharp in x. A
    magnetic component that reused the electric Yee shift would land on the wrong
    cells here while still producing a plausible radiating field.
    """
    grid = _unit_grid(resolution=10.0)
    source = VolumeSource(grid=grid, component='Hx', center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0), envelope=ContinuousEnvelope(frequency=0.8))
    assert source._indices == [(10, 9, 9), (10, 9, 10), (10, 10, 9), (10, 10, 10)]
    assert source._amplitudes == [pytest.approx(250.0)] * 4
    assert sum(source._amplitudes) == pytest.approx(10.0 ** 3)


def test_magnetic_source_injects_into_b_and_leaves_d_untouched():
    """Value-level: B -= dt * current * stored amplitude, and no D array is written."""
    grid = _unit_grid(resolution=10.0)
    fields = _StubFields(grid)
    envelope = ContinuousEnvelope(frequency=0.8)
    source = VolumeSource(grid=grid, component='Hy', center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0), envelope=envelope)
    time = 1.25
    source.inject(fields, time)

    expected = -grid.dt * envelope.current(time, grid.dt) * 250.0
    written = np.array([fields.By[index] for index in source._indices])
    np.testing.assert_allclose(written, np.full(4, expected), rtol=1e-6)
    touched = np.zeros(grid.shape, dtype=bool)
    for index in source._indices:
        touched[index] = True
    assert np.all(fields.By[~touched] == 0)
    assert np.all(fields.Bx == 0) and np.all(fields.Bz == 0)
    assert np.all(fields.Dx == 0) and np.all(fields.Dy == 0) and np.all(fields.Dz == 0)

    # 'By' names the same current as 'Hy' (MEEP src_vol folds B onto H).
    same = VolumeSource(grid=grid, component='By', center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0), envelope=ContinuousEnvelope(frequency=0.8))
    assert same._indices == source._indices
    assert same.field_type == 'B'


def test_magnetic_sheet_source_reuses_the_extended_source_geometry():
    """The loop_in_chunks path is component-driven, so a sheet works for either family."""
    grid = _unit_grid(resolution=10.0)
    sheet = VolumeSource(grid=grid, component='Hz', center=(0.0, 0.0, 0.0),
                         size=(0.4, 0.4, 0.0), envelope=ContinuousEnvelope(frequency=0.8))
    # One zero-size axis -> a^1, and the Simpson weights along an extended axis sum
    # to its extent in cells (0.4 * 10 = 4), exactly as for the electric sheet.
    assert sum(sheet._amplitudes) == pytest.approx(10.0 * 4.0 * 4.0)
    assert sheet._n_source_points > 1


# --- Temporal envelopes --------------------------------------------------------

def test_envelopes_reproduce_the_pinned_meep_waveforms():
    """The envelope objects and the CW/Gaussian classes must be the same waveform."""
    grid = _unit_grid()
    continuous = ContinuousEnvelope(frequency=0.8, width=2.0, slowness=3.0,
                                    start_time=0.0, end_time=100.0)
    ramped = ContinuousSource(grid=grid, frequency=0.8, component='Ez',
                              width=2.0, slowness=3.0, start_time=0.0, end_time=100.0)
    for time in (0.0, 0.5, 3.0, 7.0, 20.0, 99.5, 101.0):
        assert continuous.dipole(time) == ramped.dipole(time)
    assert continuous.dipole(0.0) == pytest.approx(0.0004919127472273864j, rel=1e-12)

    gaussian = GaussianEnvelope(frequency=0.8, fwidth=0.2)
    pulse = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component='Ez',
                                 center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    assert gaussian.peak_time == pulse.peak_time == pytest.approx(25.0)
    assert gaussian.width == pulse.width == pytest.approx(5.0)
    for time in (0.0, 1.0, 12.5, 25.0, 26.3, 40.0, 51.0):
        assert gaussian.dipole(time) == pulse.dipole(time)
    assert gaussian.dipole(26.3) == pytest.approx(
        0.04783096208182026 + 0.18628929781258766j, rel=1e-12)

    with pytest.raises(ValueError):
        GaussianEnvelope(frequency=0.8, fwidth=0.0)
    with pytest.raises(ValueError):
        ContinuousEnvelope(frequency=0.0)
    with pytest.raises(NotImplementedError):
        SourceEnvelope().dipole(1.0)


def test_custom_envelope_current_semantics_follow_meep():
    """MEEP meep.hpp:1066-1071 — func is the current, unless the source is integrated."""
    constant = CustomEnvelope(func=lambda t: 3.0 + 4.0j)
    dt = 0.05
    assert constant.dipole(1.0) == 3.0 + 4.0j
    # Not integrated: the function IS the current, so a constant drives a constant
    # current. Differencing it (the src_time default) would give exactly zero — an
    # unmeasurable run that a "produces some field" test would still let through.
    assert constant.current(1.0, dt) == 3.0 + 4.0j

    integrated = CustomEnvelope(func=lambda t: 3.0 + 4.0j, is_integrated=True)
    assert integrated.current(1.0, dt) == 0.0

    ramp = CustomEnvelope(func=lambda t: 2.0 * t, is_integrated=True)
    assert ramp.current(1.0, dt) == pytest.approx(2.0)      # d/dt (2t) = 2
    assert ramp.dipole(3.0) == pytest.approx(6.0)

    # Hand-computed against the analytic waveform, real and imaginary parts both.
    wave = CustomEnvelope(func=lambda t: np.exp(-((t - 1.5) ** 2)) * np.exp(-2j * np.pi * t))
    for time in (0.0, 0.75, 1.5, 2.4):
        expected = math.exp(-((time - 1.5) ** 2)) * complex(np.exp(-2j * np.pi * time))
        assert wave.dipole(time) == pytest.approx(expected, rel=1e-12, abs=1e-18)

    with pytest.raises(ValueError, match="callable"):
        CustomEnvelope(func=1.0)


def test_custom_envelope_time_window_gates_the_waveform():
    """MEEP meep.hpp:1072-1078 — outside [start_time, end_time] the waveform is exactly zero."""
    window = CustomEnvelope(func=lambda t: 1.0 + 0j, start_time=1.0, end_time=2.0)
    assert window.dipole(0.999) == 0.0
    assert window.dipole(1.0) == 1.0      # inclusive at both ends, as in MEEP
    assert window.dipole(2.0) == 1.0
    assert window.dipole(2.001) == 0.0
    assert window.current(5.0, 0.05) == 0.0
    # The MEEP default window is effectively unbounded.
    assert CustomEnvelope(func=lambda t: 1.0 + 0j).dipole(-1e6) == 1.0


# --- is_integrated -------------------------------------------------------------

def _integrated_offset(source, fields, component_array, times):  # Inject at each time, return the array.
    for time in times:
        source.inject(fields, time)
    return getattr(fields, component_array)


def test_integrated_source_tracks_the_dipole_instead_of_accumulating_current():
    """MEEP update_eh.cpp:126-140 — the array must hold D - dipole(t), not a running sum.

    The distinction is the whole feature: a non-integrated source accumulates
    `dt * current` step after step, so its total depends on every step taken; an
    integrated source's offset is a function of the current time alone and comes
    back to zero when the waveform does.
    """
    grid = _unit_grid(resolution=10.0)
    fields = _StubFields(grid)
    envelope = GaussianEnvelope(frequency=0.8, fwidth=0.5, is_integrated=True)
    source = VolumeSource(grid=grid, component='Ez', center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0), envelope=envelope, amplitude=1.0)
    assert source.is_integrated is True
    dt = grid.dt
    probe = source._indices[0]
    amplitude = source._amplitudes[0]

    peak_excursion = 0.0
    time = 0.0
    for _ in range(400):
        source.inject(fields, time)
        # MEEP evaluates the integrated E-side dipole at time() + dt, which is half a
        # step past this slot's time; the array must equal -amp * dipole there exactly.
        expected = -amplitude * envelope.dipole(time + 0.5 * dt)
        assert fields.Dz[probe] == pytest.approx(expected, rel=2e-5, abs=2e-4)
        peak_excursion = max(peak_excursion, abs(complex(fields.Dz[probe])))
        time += dt

    # Past the pulse the dipole is zero, so the source has given the field back
    # everything it lent it — the property MEEP's is_integrated exists to provide.
    assert envelope.dipole(time + 0.5 * dt) == 0.0
    assert peak_excursion > 1.0, "the pulse never drove the field; the check would be vacuous"
    assert abs(complex(fields.Dz[probe])) < 1e-5 * peak_excursion

    # A non-integrated source of the same waveform does not have that property step
    # by step: at the peak it holds the accumulated current, not the dipole.
    plain_fields = _StubFields(grid)
    plain = VolumeSource(grid=grid, component='Ez', center=(0.0, 0.0, 0.0),
                         size=(0.0, 0.0, 0.0),
                         envelope=GaussianEnvelope(frequency=0.8, fwidth=0.5))
    time = 0.0
    for _ in range(40):
        plain.inject(plain_fields, time)
        time += dt
    integrated_here = -amplitude * envelope.dipole(time + 0.5 * dt)
    assert abs(complex(plain_fields.Dz[probe]) - integrated_here) > 0.1 * abs(integrated_here)


def test_integrated_magnetic_source_uses_the_same_half_step_offset():
    """MEEP step.cpp:74 — the integrated H-side dipole is read at time() + dt/2 too."""
    grid = _unit_grid(resolution=10.0)
    fields = _StubFields(grid)
    envelope = ContinuousEnvelope(frequency=0.8, width=2.0, is_integrated=True)
    source = VolumeSource(grid=grid, component='Hz', center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0), envelope=envelope)
    dt = grid.dt
    probe = source._indices[0]
    amplitude = source._amplitudes[0]
    time = 0.0
    for _ in range(60):
        source.inject(fields, time)
        expected = -amplitude * envelope.dipole(time + 0.5 * dt)
        assert fields.Bz[probe] == pytest.approx(expected, rel=2e-5, abs=2e-4)
        time += dt
    assert np.all(fields.Dz == 0)


def test_integrated_source_reset_forgets_the_applied_offset():
    grid = _unit_grid(resolution=10.0)
    fields = _StubFields(grid)
    source = VolumeSource(grid=grid, component='Ez', center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=ContinuousEnvelope(frequency=0.8, is_integrated=True))
    source.inject(fields, 1.0)
    assert source._applied_dipole != 0
    source.reset()
    assert source._applied_dipole == 0


def test_integrated_source_accepts_a_symmetric_grid():
    """An is_integrated source builds under mirror symmetry — the refusal is retired.

    The fear the old refusal encoded — the mirror repair pass copies cells while
    they hold ``f - dipole`` — is not a defect: the ghost the repair writes is
    ``parity * (f - dipole)`` at the source's image cells, which IS MEEP's
    ``f_minus_p`` there, and ``withdraw`` restores the owned points before the
    next curl. MEEP itself folds the combination exactly (folded vs unfolded MEEP
    measured 1.7e-07..1.0e-06 for an integrated sheet through a Mirror(Y) with a
    PML), and the engine reproduces MEEP's folded run at its ordinary fold floor
    — pinned against CPU MEEP by test_from_meep.py's
    ``test_integrated_source_through_the_fold_matches_meep``. The refusal cost
    finite_grating.py, absorbed_power_density.py and mie_scattering.py their
    lifts.
    """
    grid = _unit_grid(resolution=10.0, symmetry=('X',))
    integrated = VolumeSource(grid=grid, component='Ez', center=(0.0, 0.0, 0.0),
                              size=(0.0, 0.0, 0.0),
                              envelope=ContinuousEnvelope(frequency=0.8, is_integrated=True))
    assert integrated.is_integrated
    # The parity rules are untouched: a component the plane makes odd still refuses.
    with pytest.raises(ValueError, match="parity -1"):
        VolumeSource(grid=grid, component='Ex', center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                     envelope=ContinuousEnvelope(frequency=0.8, is_integrated=True))


# --- Degenerate and loud-failure behavior -------------------------------------

def test_zero_custom_envelope_injects_exactly_nothing():
    """A silent envelope must leave the fields at exactly zero, not merely small."""
    grid = _unit_grid(resolution=10.0)
    for is_integrated in (False, True):
        fields = _StubFields(grid)
        silent = VolumeSource(
            grid=grid, component='Ez', center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0),
            envelope=CustomEnvelope(func=lambda t: 0.0, is_integrated=is_integrated))
        for step in range(50):
            silent.inject(fields, step * grid.dt + 0.5 * grid.dt)
        assert np.count_nonzero(fields.Dz) == 0
        assert np.all(fields.Dz == 0)

    # Positive control on the identical setup: an envelope that is not silent does
    # write, so the assertion above is measuring the envelope and not a dead path.
    fields = _StubFields(grid)
    loud = VolumeSource(grid=grid, component='Ez', center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0),
                        envelope=CustomEnvelope(func=lambda t: 1.0 + 0j))
    loud.inject(fields, 0.5 * grid.dt)
    assert np.count_nonzero(fields.Dz) == loud._n_source_points

    # Same for the magnetic path: a zero envelope leaves B exactly zero.
    fields = _StubFields(grid)
    quiet_magnet = VolumeSource(grid=grid, component='Hx', center=(0.0, 0.0, 0.0),
                                size=(0.0, 0.0, 0.0),
                                envelope=CustomEnvelope(func=lambda t: 0.0))
    for step in range(20):
        quiet_magnet.inject(fields, step * grid.dt)
    assert np.all(fields.Bx == 0)


def test_volume_source_rejects_unknown_components_envelopes_and_dtypes():
    grid = _unit_grid(resolution=10.0)
    point = dict(center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="component"):
        VolumeSource(grid=grid, component='Centered',
                     envelope=ContinuousEnvelope(frequency=0.8), **point)
    with pytest.raises(ValueError, match="component"):
        VolumeSource(grid=grid, component='Bogus',
                     envelope=ContinuousEnvelope(frequency=0.8), **point)
    with pytest.raises(ValueError, match="SourceEnvelope"):
        VolumeSource(grid=grid, component='Ez', envelope='gaussian', **point)
    with pytest.raises(ValueError, match="component"):
        src._array_for(_StubFields(grid), 'Centered')

    # A component switched to an unsupported value after setup still raises.
    good = VolumeSource(grid=grid, component='Hx',
                        envelope=ContinuousEnvelope(frequency=0.8), **point)
    good.component = 'Centered'
    with pytest.raises(ValueError, match="component"):
        good.inject(_StubFields(grid), 1.0)

    # Real-valued float32 storage is a SUPPORTED width — MEEP's default — served by
    # taking real(current * dt) (step.cpp:307), and a complex amplitude does not
    # change that: the projection is MEEP's own and is pinned by
    # test_real_injection_of_a_phased_spatial_amplitude_is_the_complex_run_s_real_part.
    # The refusal below is about a width that is neither of the two.
    magnetic = VolumeSource(grid=grid, component='Hx',
                            envelope=ContinuousEnvelope(frequency=0.8), **point)
    real_fields = _StubFields(grid, dtype=np.float32)
    magnetic.inject(real_fields, 1.0)
    assert np.count_nonzero(real_fields.Bx) > 0, "real storage must be injected into, not refused"
    with pytest.raises(ValueError, match="complex64 or float32"):
        magnetic.inject(_StubFields(grid, dtype=np.float64), 1.0)
    phased = VolumeSource(grid=grid, component='Hx', amplitude=1j,
                          envelope=ContinuousEnvelope(frequency=0.8), **point)
    phased_fields = _StubFields(grid, dtype=np.float32)
    phased.inject(phased_fields, 1.0)
    assert np.count_nonzero(phased_fields.Bx) > 0, (
        "a complex amplitude on real storage is projected, not refused"
    )

    # A source in the half an active mirror plane DISCARDS is refused for either
    # family; one merely off the plane, in the stored half, is not (MEEP steps it).
    symmetric = _unit_grid(resolution=10.0, symmetry=('Y',))
    VolumeSource(grid=symmetric, component='Hx', center=(0.0, 0.3, 0.0),
                 size=(0.0, 0.0, 0.0), envelope=ContinuousEnvelope(frequency=0.8))
    with pytest.raises(ValueError, match="Y symmetry the stored half"):
        VolumeSource(grid=symmetric, component='Hx', center=(0.0, -0.3, 0.0),
                     size=(0.0, 0.0, 0.0), envelope=ContinuousEnvelope(frequency=0.8))


def test_volume_source_accepts_a_bare_callable_as_its_envelope():
    grid = _unit_grid(resolution=10.0)
    source = VolumeSource(grid=grid, component='Ez', center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0), envelope=lambda t: 2.0 + 0j)
    assert isinstance(source.envelope, CustomEnvelope)
    assert source.current(1.0, grid.dt) == 2.0 + 0j
    assert source.is_integrated is False


def test_volume_source_reproduces_the_extended_source_on_their_shared_case():
    """Same component, geometry and waveform must give the same injection, cell for cell."""
    grid = _unit_grid(resolution=10.0)
    geometry = dict(center=(0.0, 0.0, 0.1), size=(0.4, 0.0, 0.2))
    legacy = ExtendedSource(grid=grid, frequency=0.8, component='Ey',
                            width=2.0, end_time=50.0, **geometry)
    general = VolumeSource(grid=grid, component='Ey', envelope=ContinuousEnvelope(
        frequency=0.8, width=2.0, end_time=50.0), **geometry)
    assert general._indices == legacy._indices
    assert general._amplitudes == legacy._amplitudes

    legacy_fields, general_fields = _StubFields(grid), _StubFields(grid)
    legacy.inject(legacy_fields, 1.25)
    general.inject(general_fields, 1.25)
    np.testing.assert_array_equal(general_fields.Dy, legacy_fields.Dy)


# --- CPU-MEEP cross-validation -------------------------------------------------

_ORACLE_SCRIPT = '''"""Generate one CPU-MEEP reference field set for the source parity tests."""
import sys

import numpy as np
import meep as mp

case = sys.argv[1]
output_path = sys.argv[2]

# The *_real cases run MEEP's DEFAULT storage — force_complex_fields=False — which is
# the mode the engine's own real-field runs are cross-validated against. k_point is
# still the zero vector: that makes every boundary periodic without forcing complex
# fields (fields.cpp:146 aborts only when is_real AND k != 0).
complex_fields = not case.endswith("_real")

if case == "folded_cw_real":
    # Full-domain reference for the mirror-folded real run: a 3x3x4 cell whose Ez CW
    # point source sits on the X mirror plane, stopped at t = 1.0 while the folded
    # engine's far x face is still dark (the fold is exact only until the wave
    # reaches it; see test_driver_vs_meep's folded_cell case for the measured decay
    # of that agreement).
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 4),
        resolution=10,
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, 0.05))],
        k_point=mp.Vector3(0, 0, 0),
    )
    simulation.run(until=1.0)
    assert simulation.fields.is_real, "the folded reference must run MEEP's default real storage"
    np.savez(output_path, Ez=np.asarray(simulation.get_array(component=mp.Ez)))
    raise SystemExit(0)

if case in ("magnetic_cw", "magnetic_cw_real"):
    source_time = mp.ContinuousSource(frequency=1.0)
    component = mp.Hx
elif case == "electric_cw_real":
    source_time = mp.ContinuousSource(frequency=1.0)
    component = mp.Ez
elif case == "integrated_cw":
    source_time = mp.ContinuousSource(frequency=1.0, width=1.0, is_integrated=True)
    component = mp.Ez
elif case == "custom_envelope":
    source_time = mp.CustomSource(
        src_func=lambda t: np.exp(-((t - 1.5) ** 2)) * np.exp(-2j * np.pi * t)
    )
    component = mp.Ez
else:
    raise SystemExit("unknown oracle case: " + case)

simulation = mp.Simulation(
    cell_size=mp.Vector3(2, 2, 4),
    resolution=10,
    sources=[mp.Source(source_time, component=component, center=mp.Vector3(0, 0, -1.05))],
    force_complex_fields=complex_fields,
    k_point=mp.Vector3(0, 0, 0),
)
simulation.run(until=3)
assert simulation.fields.is_real != complex_fields, "the oracle is not in the storage mode it claims"
np.savez(
    output_path,
    **{name: np.asarray(simulation.get_array(component=getattr(mp, name)))
       for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
)
'''


def _meep_source_oracle(tmp_path, case):  # Run CPU MEEP in its own process and load the reference fields.
    script_path = tmp_path / "meep_source_oracle.py"
    script_path.write_text(_ORACLE_SCRIPT, encoding="utf-8")
    output_path = tmp_path / f"{case}.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), case, str(output_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP oracle '{case}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


# MEEP's fields::step() order, with both source slots exposed so a test can drive a
# source at the wrong half-step on purpose (step.cpp:64-100). The parity cases run
# it on a 2x2x4 cell at resolution 10 with no PML, which is 60 steps to t = 3.
# `complex_fields=False` steps the same sequence over float32 storage; the D-side
# mirror repair is the driver's own placement (after injection, before update_E) and
# is a no-op on an unfolded grid, so the historical cases are unchanged by it.
def _run_source(source, num_steps=60, b_offset=0.0, d_offset=0.5,
                complex_fields=True):  # Step a hand-built field set.
    grid = source.grid
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    magnetic = source.field_type == "B"
    dt = grid.dt
    time = 0.0
    for _ in range(num_steps):
        step_B(fields, None)
        if magnetic:
            source.inject(fields, time + b_offset * dt)
        update_H(fields, None)
        step_D(fields, None)
        if not magnetic:
            source.inject(fields, time + d_offset * dt)
        fill_symmetry_bc_D(fields)
        update_E(fields, None)
        time += dt
    return fields


def _parity_grid():  # The cell the oracle script simulates.
    return Grid(resolution=10, cell_size=(2.0, 2.0, 4.0))


def _complex_relative_l2(candidate, reference):  # Relative L2 on the complex field: catches phase error too.
    engine_field = np.asarray(candidate, dtype=np.complex128).ravel()
    meep_field = np.asarray(reference, dtype=np.complex128).ravel()
    reference_norm = float(np.linalg.norm(meep_field))
    candidate_norm = float(np.linalg.norm(engine_field))
    assert reference_norm > 0.0, "CPU-MEEP produced an all-zero field; the comparison is meaningless."
    assert candidate_norm > 0.0, "The engine produced an all-zero field; that must not score as perfect."
    return float(np.linalg.norm(engine_field - meep_field) / reference_norm)


def _compare(fields, reference, names):  # Relative L2 per component against MEEP's get_array output.
    # MEEP's get_array returns (N+1) points per axis, the extra plane being the
    # periodic duplicate at index 0; index i + 1 is this grid's cell i.
    return {name: _complex_relative_l2(fields.to_cell_center(name), reference[name][1:, 1:, 1:])
            for name in names}


@requires_meep
@skip_without_meep
def test_magnetic_source_matches_cpu_meep(tmp_path):  # Feature 1: Hx current, and the half-step it lives on.
    reference = _meep_source_oracle(tmp_path, "magnetic_cw")
    grid = _parity_grid()
    # z = -1.05 is exactly an Hx Yee point on both grids, so neither side splits the
    # source along z and the comparison is not measuring interpolation.
    def make():
        return VolumeSource(grid=grid, component='Hx', center=(0.0, 0.0, -1.05),
                            size=(0.0, 0.0, 0.0), envelope=ContinuousEnvelope(frequency=1.0))

    names = ("Hx", "Ey", "Ez", "Hy", "Hz")
    errors = _compare(_run_source(make()), reference, names)
    # Measured 2.1e-07 (Ez) to 6.0e-07 (Hy) — the same order as the engine's core
    # algorithm parity, ~10^5 inside the 5% cross-validation bar.
    for name, error in errors.items():
        assert error < 0.05, f"Complex {name} relative L2 {error:.3e} exceeds the 5% bar."
        assert error < 1e-5, f"Complex {name} relative L2 {error:.3e} regressed from the 1e-5 held bound."

    # The convention this proves: MEEP drives B sources at time(), half a step before
    # D sources (step.cpp:64 vs :95). Injecting at the D slot instead is a pure phase
    # error — every magnitude is still plausible — so the wrong convention is run here
    # and required to be far outside the bar. Measured 1.6e-01 on every component.
    wrong = _compare(_run_source(make(), b_offset=0.5), reference, names)
    for name, error in wrong.items():
        assert error > 0.05, (
            f"Injecting the magnetic current half a step late still matched MEEP in {name} "
            f"({error:.3e}); this test cannot tell the two conventions apart."
        )


# --- real-field (float32) mode: CPU-MEEP cross-validation ------------------------
#
# MEEP's DEFAULT storage. The oracles below run WITHOUT force_complex_fields, so the
# engine's real path is measured against MEEP's real path — not against the real part
# of a complex reference, which would leave a shared analytic-signal convention error
# invisible. Real-vs-complex agreement inside THIS engine is pinned bit-for-bit
# elsewhere (test_stepping's sweep, and `test_injection_into_real_fields_takes_the_
# real_part_of_the_current` above); these are the absolute anchors.


def _real_relative_l2(candidate, reference):  # Relative L2 on real float arrays, in float64.
    engine_field = np.asarray(candidate, dtype=np.float64).ravel()
    meep_field = np.asarray(reference, dtype=np.float64).ravel()
    reference_norm = float(np.linalg.norm(meep_field))
    candidate_norm = float(np.linalg.norm(engine_field))
    assert reference_norm > 0.0, "CPU-MEEP produced an all-zero field; the comparison is meaningless."
    assert candidate_norm > 0.0, "The engine produced an all-zero field; that must not score as perfect."
    return float(np.linalg.norm(engine_field - meep_field) / reference_norm)


@requires_meep
@skip_without_meep
@pytest.mark.parametrize(
    "case, component, names, held_bound",
    [
        # Measured 2.3e-07 (Ez) to 3.0e-07 (Hy) electric, 1.8e-07 (Ez) to 7.6e-07
        # (Hy) magnetic — the same order as the complex-mode floors (2.41e-07 core),
        # because a real run IS the complex run's real plane. Held at ~3x measured.
        ("electric_cw_real", "Ez", ("Ez", "Hx", "Hy"), 2.5e-6),
        ("magnetic_cw_real", "Hx", ("Hx", "Ey", "Ez", "Hy", "Hz"), 2.5e-6),
    ],
)
def test_real_field_run_matches_cpu_meeps_default_mode(tmp_path, case, component, names,
                                                       held_bound):
    """Both injection families, stepped over float32 storage, against MEEP's default mode.

    This is the absolute cross-validation real mode rests on: the same CW point
    source, cell and run as the complex parity cases, with BOTH codes in real
    storage. The electric case drives the D slot and the magnetic case the B slot
    half a step earlier, so a real-mode error in either injection time or in
    ``real(current * dt)`` itself (|A|, or Re(amp)*Re(current), say) has nowhere to
    hide — the five compared components include everything the curl couples the
    source into.
    """
    reference = _meep_source_oracle(tmp_path, case)
    grid = _parity_grid()
    source = VolumeSource(grid=grid, component=component, center=(0.0, 0.0, -1.05),
                          size=(0.0, 0.0, 0.0), envelope=ContinuousEnvelope(frequency=1.0))
    fields = _run_source(source, complex_fields=False)
    for name in names:
        candidate = fields.to_cell_center(name)
        assert candidate.dtype == np.float32, f"{name} left real mode during the run"
        error = _real_relative_l2(candidate, reference[name][1:, 1:, 1:])
        assert error < 0.05, f"Real {name} relative L2 {error:.3e} exceeds the 5% bar."
        assert error < held_bound, (
            f"Real {name} relative L2 {error:.3e} regressed from the {held_bound:.1e} held bound."
        )


@requires_meep
@skip_without_meep
def test_folded_real_run_matches_cpu_meeps_full_domain(tmp_path):
    """Mirror symmetry over float32 storage, against MEEP's real full-domain run.

    The folded quadrant must reproduce MEEP cell for cell under the established
    correspondence (quadrant cell q = MEEP index centre + q), while the far face is
    still dark — the same t = 1.0 window the complex folded case is held in.
    Real mode measured 2.7e-06 at the 2026-08-06 re-baselining (5.8e-06 before the
    folded-far-face rework); held at 2.5e-05. The unfolded real run of the same
    case is the control that pins the residual on the fold rather than on real
    mode: measured 1.7e-07, held at 1e-06.
    """
    reference = _meep_source_oracle(tmp_path, "folded_cw_real")["Ez"]
    specification = dict(component='Ez', center=(0.0, 0.0, 0.05), size=(0.0, 0.0, 0.0))

    folded_grid = Grid(resolution=10, cell_size=(3.0, 3.0, 4.0), symmetry=("X",))
    folded = _run_source(
        VolumeSource(grid=folded_grid, envelope=ContinuousEnvelope(frequency=1.0),
                     **specification),
        num_steps=20, complex_fields=False,  # 20 steps of dt = 0.05 is t = 1.0.
    )
    full_grid = Grid(resolution=10, cell_size=(3.0, 3.0, 4.0))
    full = _run_source(
        VolumeSource(grid=full_grid, envelope=ContinuousEnvelope(frequency=1.0),
                     **specification),
        num_steps=20, complex_fields=False,
    )

    centre = full_grid.nx // 2  # Quadrant cell q is full cell centre-1+q, i.e. MEEP index centre+q.
    quadrant = folded.to_cell_center("Ez")
    assert quadrant.dtype == np.float32, "the folded run left real mode"
    # The stored layout carries one row past MEEP's owned window (the second-mirror
    # plane cell of an even periodic fold); rows 0..n-2 are the owned quadrant this
    # comparison has always pinned, and the plane row is covered by the complex
    # live-far-face cases and the bit-for-bit fold equivalences in test_driver_vs_meep.
    folded_error = _real_relative_l2(quadrant[:-1], reference[centre:, 1:, 1:])
    full_error = _real_relative_l2(full.to_cell_center("Ez"), reference[1:, 1:, 1:])
    assert full_error < 1e-6, f"the unfolded real control is not at its floor: {full_error:.3e}"
    assert folded_error < 0.05, f"Folded real Ez relative L2 {folded_error:.3e} exceeds the 5% bar."
    assert folded_error < 2.5e-5, (
        f"Folded real Ez relative L2 {folded_error:.3e} regressed from the 2.5e-05 held bound."
    )


@requires_meep
@skip_without_meep
def test_integrated_source_matches_cpu_meep(tmp_path):  # Feature 2: is_integrated.
    reference = _meep_source_oracle(tmp_path, "integrated_cw")
    grid = _parity_grid()

    def make(is_integrated):
        return VolumeSource(
            grid=grid, component='Ez', center=(0.0, 0.0, -1.05), size=(0.0, 0.0, 0.0),
            envelope=ContinuousEnvelope(frequency=1.0, width=1.0, is_integrated=is_integrated))

    names = ("Ez", "Hx", "Hy")
    errors = _compare(_run_source(make(True)), reference, names)
    # Measured 1.1e-07 (Ez) to 2.1e-07 (Hx).
    for name, error in errors.items():
        assert error < 0.05, f"Complex {name} relative L2 {error:.3e} exceeds the 5% bar."
        assert error < 1e-5, f"Complex {name} relative L2 {error:.3e} regressed from the 1e-5 held bound."

    # Control: the same waveform injected as a plain current is a different run.
    # Measured 1.6e-01 — without this, a build that ignored the flag would pass.
    plain = _compare(_run_source(make(False)), reference, names)
    for name, error in plain.items():
        assert error > 0.05, (
            f"The non-integrated source also matched MEEP's integrated run in {name} "
            f"({error:.3e}); the is_integrated flag is not being honoured."
        )


@requires_meep
@skip_without_meep
def test_custom_envelope_matches_cpu_meep(tmp_path):  # Feature 3: caller-supplied waveform.
    reference = _meep_source_oracle(tmp_path, "custom_envelope")
    grid = _parity_grid()

    def waveform(t):  # The same chirp-free Gaussian-windowed carrier the oracle uses.
        return np.exp(-((t - 1.5) ** 2)) * np.exp(-2j * np.pi * t)

    names = ("Ez", "Hx", "Hy")
    source = VolumeSource(grid=grid, component='Ez', center=(0.0, 0.0, -1.05),
                          size=(0.0, 0.0, 0.0), envelope=CustomEnvelope(func=waveform))
    errors = _compare(_run_source(source), reference, names)
    # Measured 1.8e-07 (Ez) to 3.1e-07 (Hx).
    for name, error in errors.items():
        assert error < 0.05, f"Complex {name} relative L2 {error:.3e} exceeds the 5% bar."
        assert error < 1e-5, f"Complex {name} relative L2 {error:.3e} regressed from the 1e-5 held bound."

    # Control: reading the same function as a dipole (MEEP's is_integrated branch)
    # differences it instead of injecting it, which is a different physical source.
    # Measured 1.0e+00 — a full-scale disagreement.
    integrated = VolumeSource(grid=grid, component='Ez', center=(0.0, 0.0, -1.05),
                              size=(0.0, 0.0, 0.0),
                              envelope=CustomEnvelope(func=waveform, is_integrated=True))
    wrong = _compare(_run_source(integrated), reference, names)
    for name, error in wrong.items():
        assert error > 0.05, (
            f"Treating the custom waveform as a dipole still matched MEEP in {name} "
            f"({error:.3e}); the current/dipole distinction is not being made."
        )


# --- point-source placement: swept CPU-MEEP cross-validation ---------------------

_PLACEMENT_ORACLE_SCRIPT = '''"""CPU-MEEP references for the point-source placement sweep."""
import json
import sys

import numpy as np
import meep as mp

with open(sys.argv[1]) as handle:
    cases = json.load(handle)
output_path = sys.argv[2]

arrays = {}
for index, (cell, k_point, center) in enumerate(cases):
    simulation = mp.Simulation(
        cell_size=mp.Vector3(*cell),
        resolution=10,
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(*center))],
        force_complex_fields=True,
        k_point=mp.Vector3(*k_point),
    )
    simulation.run(until=3)
    arrays[str(index)] = np.asarray(simulation.get_array(component=mp.Ez))
np.savez(output_path, **arrays)
'''


def _meep_placement_oracle(tmp_path, cases):  # Every placement in one CPU-MEEP process.
    # One subprocess for the whole table: each run is ~20 ms and importing meep is ~2 s,
    # so a process per case would cost a minute of import for a second of physics.
    script_path = tmp_path / "meep_placement_oracle.py"
    script_path.write_text(_PLACEMENT_ORACLE_SCRIPT, encoding="utf-8")
    cases_path = tmp_path / "placement_cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    output_path = tmp_path / "placements.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), str(cases_path), str(output_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP placement oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


# label -> (cell size, k_point, source centre). Resolution 10 throughout, so dx = 0.1.
#
# 1 x 1 x 2 is nx = ny = 10 and nz = 20 — EVEN. 0.9 x 1 x 1.9 is nx = 9 and nz = 19 —
# ODD, and MEEP rounds an axis's little corner DOWN TO EVEN, so an odd axis starts half
# a cell higher than -L/2: x runs [-0.4, 0.5] and z runs [-0.9, 1.0]. Every even-count
# placement is repeated at odd counts because that half-cell registration is invisible
# to an even-only sweep — this engine has shipped exactly that bug before.
#
# Ez is INTEGER-placed in x (samples at origin + i*dx, so the cell faces ARE samples and
# the whole last cell of the axis needs the wrap) and HALF-INTEGER in z (samples at
# origin + (i+0.5)*dx, so the faces fall BETWEEN samples and split across them). The two
# placements fail differently, so both are swept across a full cell face to face.
_PLACEMENT_CASES = (
    ("even z, lower face",         (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.000)),
    ("even z, quarter cell in",    (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, -0.975)),
    ("even z, cell centre",        (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, -0.950)),
    ("even z, three quarters in",  (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, -0.925)),
    ("even z, next face",          (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, -0.900)),
    ("even z, upper face",         (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0,  1.000)),
    ("even x, lower face",         (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (-0.500, 0.0, 0.0)),
    ("even x, last sample",        (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.400, 0.0, 0.0)),
    ("even x, last cell centre",   (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.450, 0.0, 0.0)),
    ("even x, upper face",         (1.0, 1.0, 2.0), (0.0, 0.0, 0.0), (0.500, 0.0, 0.0)),
    ("odd z, lower face",          (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.0, 0.0, -0.900)),
    ("odd z, quarter cell in",     (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.0, 0.0, -0.875)),
    ("odd z, cell centre",         (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.0, 0.0, -0.850)),
    ("odd z, three quarters in",   (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.0, 0.0, -0.825)),
    ("odd z, next face",           (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.0, 0.0, -0.800)),
    ("odd z, upper face",          (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.0, 0.0,  1.000)),
    ("odd x, lower face",          (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (-0.400, 0.0, 0.0)),
    ("odd x, last sample",         (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.400, 0.0, 0.0)),
    ("odd x, last cell centre",    (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.450, 0.0, 0.0)),
    ("odd x, upper face",          (0.9, 1.0, 1.9), (0.0, 0.0, 0.0), (0.500, 0.0, 0.0)),
    # The wrap has to carry exp(i*2*pi*k*L), which is pure phase and survives every
    # magnitude comparison. kx = 0.37 is deliberately not a simple fraction of the cell.
    ("bloch kz, upper z face",     (1.0, 1.0, 2.0), (0.0, 0.0, 0.3), (0.0, 0.0, 1.000)),
    ("bloch kz, lower z face",     (1.0, 1.0, 2.0), (0.0, 0.0, 0.3), (0.0, 0.0, -1.000)),
    ("bloch kx, upper x face",     (1.0, 1.0, 2.0), (0.4, 0.0, 0.0), (0.500, 0.0, 0.0)),
    ("bloch kx, last cell centre", (1.0, 1.0, 2.0), (0.37, 0.0, 0.0), (0.450, 0.0, 0.0)),
)

_PLACEMENT_INDEX = {label: index for index, (label, *_rest) in enumerate(_PLACEMENT_CASES)}


def _placement_field(label):  # Run one placement through the engine and cell-centre Ez.
    _label, cell, k_point, center = _PLACEMENT_CASES[_PLACEMENT_INDEX[label]]
    grid = Grid(resolution=10, cell_size=cell, k_point=k_point)
    source = ContinuousSource(grid=grid, frequency=1.0, component='Ez', center=center)
    return _run_source(source, num_steps=60).to_cell_center('Ez')  # 60 steps at dt = 0.05 is t = 3.


@requires_meep
@skip_without_meep
def test_point_source_placement_matches_cpu_meep_across_a_cell(tmp_path):
    """Sweep a point source face to face across a cell and cross-validate every stop.

    THE REGRESSION THIS PINS. A request in the outer half-cell of a wrapping axis used
    to be placed by extrapolating off the end of the array instead of by folding onto
    the lattice, and the negative half of the extrapolation was then dropped by a
    positive-weight filter. Every run completed and returned a smooth, plausible field.
    Complex relative L2 over the whole Ez volume, with the old placement restored by
    monkeypatch and re-measured against these very references — before -> after:

        even z, lower face          5.405e-01 -> 1.575e-07
        even z, quarter cell in     2.696e-01 -> 1.656e-07
        even z, cell centre         1.748e-07 -> 1.748e-07   (already right, unchanged)
        even z, three quarters in   1.790e-07 -> 1.790e-07   (already right, unchanged)
        even z, next face           1.575e-07 -> 1.575e-07   (already right, unchanged)
        even z, upper face          5.405e-01 -> 1.575e-07
        even x, last cell centre    6.199e-01 -> 1.449e-07
        even x, upper face          1.204e+00 -> 1.575e-07
        odd z, lower face           5.397e-01 -> 2.008e-07
        odd z, upper face           5.397e-01 -> 2.008e-07
        odd x, last cell centre     6.143e-01 -> 1.439e-07
        odd x, upper face           1.195e+00 -> 2.008e-07
        bloch kz, upper z face      5.339e-01 -> 1.627e-07
        bloch kz, lower z face      5.339e-01 -> 1.531e-07
        bloch kx, upper x face      1.151e+00 -> 2.132e-07
        bloch kx, last cell centre  5.819e-01 -> 1.807e-07

    12 of the 24 placements were wrong, worst 1.204e+00; the worst now is 2.234e-07.
    The 12 that were already at the engine's floor are swept anyway AND their numbers
    are bit-identical before and after, which is what rules out the other way this
    engine has been wrong: a fix that moved the whole registration by half a cell to
    repair the faces would have disturbed every one of them.
    """
    references = _meep_placement_oracle(tmp_path, [case[1:] for case in _PLACEMENT_CASES])
    errors = {}
    for index, (label, _cell, _k_point, _center) in enumerate(_PLACEMENT_CASES):
        # MEEP's get_array returns (N+1) points per axis, the extra plane being the
        # periodic duplicate at index 0; index i + 1 is this grid's cell i.
        errors[label] = _complex_relative_l2(_placement_field(label),
                                             references[str(index)][1:, 1:, 1:])
    for label, error in errors.items():
        assert error < 0.05, f"'{label}': complex Ez relative L2 {error:.3e} exceeds the 5% bar."
        assert error < 1e-5, (
            f"'{label}': complex Ez relative L2 {error:.3e} regressed from the 1e-5 held bound "
            f"(measured worst 2.3e-7 across the whole table)."
        )


@requires_meep
@skip_without_meep
def test_the_placement_sweep_can_tell_placements_apart(tmp_path):
    """Controls. Without these the sweep above proves only that something ran.

    Two ways a placement comparison can be vacuous: the metric might not resolve a
    one-cell move (so landing on the wrong cell would still 'pass'), and it might not
    resolve the Bloch phase (so an unphased wrap would still 'pass'). Both are measured
    here against the same references, and both must land far outside the bar.
    """
    references = _meep_placement_oracle(tmp_path, [case[1:] for case in _PLACEMENT_CASES])

    def reference(label):
        return references[str(_PLACEMENT_INDEX[label])][1:, 1:, 1:]

    # One cell of misplacement: the upper x face against the reference for the sample
    # one cell below it. Measured 1.2e+00 — the size of the original defect, which is
    # what makes the 1e-7 agreement above meaningful.
    off_by_one = _complex_relative_l2(_placement_field("even x, upper face"),
                                      reference("even x, last sample"))
    assert off_by_one > 0.05, (
        f"A one-cell placement error still scores {off_by_one:.3e}; the sweep cannot tell "
        f"placements apart and proves nothing."
    )

    # An unphased wrap: the Bloch run against the k = 0 reference for the same position.
    # The magnitudes are nearly identical, so only a complex metric separates them.
    unphased = _complex_relative_l2(_placement_field("bloch kx, upper x face"),
                                    reference("even x, upper face"))
    assert unphased > 0.05, (
        f"The kx = 0.4 run still matches the k = 0 reference at {unphased:.3e}; the wrap "
        f"phase is not being measured."
    )

    # And the degenerate run that must never score as perfect: a source of zero
    # amplitude produces an all-zero field, which differs from the reference everywhere
    # and must be reported as such, not waved through as "nothing to disagree about".
    grid = Grid(resolution=10, cell_size=(1.0, 1.0, 2.0))
    silent = ContinuousSource(grid=grid, frequency=1.0, component='Ez',
                              center=(0.0, 0.0, -1.0), amplitude=0.0)
    quiet_field = _run_source(silent, num_steps=60).to_cell_center('Ez')
    assert not np.any(quiet_field), "The zero-amplitude control is not actually silent."
    with pytest.raises(AssertionError, match="must not score as perfect"):
        _complex_relative_l2(quiet_field, reference("even z, lower face"))


# ---------------------------------------------------------------------------
# The source rules a mirror plane imposes, at the plane's own declared phase.
# ---------------------------------------------------------------------------

_DRIVABLE = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _accepts(grid, component):  # Does the parity rule let this component drive this fold?
    try:
        src._validate_symmetry_parity(grid, component)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("phase", [+1, -1])
def test_the_source_parity_rule_reads_the_plane_not_the_even_xy_table(axis, phase):
    """The refused set must be exactly the components the DECLARED plane makes odd.

    The rule used to be read off ``fields.SYMMETRY_PHASES`` — ``mirror_parity``
    frozen at MEEP's default ``phase=+1`` on X and Y alone. That is right for an even
    X or Y mirror and wrong everywhere else: it says nothing about Z (so a Z fold ran
    the check against an axis it had no entry for and accepted everything), and on an
    odd plane it refuses precisely the components that plane makes EVEN.

    Asserted against ``mirror_parity`` at the plane's own phase rather than a
    hand-written list, and asserted to be non-trivial in both directions: each fold
    must accept some components and refuse some, or a rule that refused everything
    (or nothing) would pass.
    """
    from .fields import mirror_parity

    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0),
                symmetry=(Mirror("XYZ"[axis], phase),))
    accepted = {c for c in _DRIVABLE if _accepts(grid, c)}
    expected = {c for c in _DRIVABLE if mirror_parity(c, axis, phase) > 0}
    assert accepted == expected, (
        f"Mirror({'XYZ'[axis]!r}, {phase:+d}) accepts {sorted(accepted)}; the plane's own "
        f"parity says {sorted(expected)}"
    )
    assert 0 < len(accepted) < len(_DRIVABLE), (
        f"Mirror({'XYZ'[axis]!r}, {phase:+d}) accepts {len(accepted)} of {len(_DRIVABLE)} "
        f"components; a rule that refuses everything or nothing is not a rule"
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_two_phases_of_one_plane_accept_complementary_currents(axis):
    """An odd plane is the same fold with every sign inverted, so the sets swap.

    This is the property that makes ``mp.Mirror(direction, phase=-1)`` worth exposing:
    a current along the plane's OWN normal — an ``Ex`` dipole on ``x = 0`` — is odd
    under an even mirror and even under an odd one, so declaring the odd plane is the
    only way to fold it. Under the old rule that dipole was refused outright on every
    axis and no phase argument existed to change it.
    """
    even = {c for c in _DRIVABLE
            if _accepts(Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0),
                             symmetry=(Mirror("XYZ"[axis], +1),)), c)}
    odd = {c for c in _DRIVABLE
           if _accepts(Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0),
                            symmetry=(Mirror("XYZ"[axis], -1),)), c)}
    assert even.isdisjoint(odd), f"axis {'XYZ'[axis]}: {sorted(even & odd)} accepted by both phases"
    assert even | odd == set(_DRIVABLE), (
        f"axis {'XYZ'[axis]}: {sorted(set(_DRIVABLE) - (even | odd))} accepted by neither phase"
    )
    # The named case: the electric current ALONG the plane's normal.
    along_the_normal = f"E{'xyz'[axis]}"
    assert along_the_normal in odd and along_the_normal not in even


def _refusal(grid, component, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0), amp_func=None):
    """The parity rule's verdict on a full declaration: None if accepted, else the message."""
    try:
        src._validate_symmetry_parity(grid, component, center, size, amp_func)
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("phase", [+1, -1])
def test_the_parity_rule_is_the_profile_not_the_centre(axis, phase):
    """A sheet whose PROFILE is odd about the plane is foldable; a uniform one is not.

    THE PREDICATE WAS THE CENTRE, and that is measurably the wrong question. It judged
    an extended source by where its middle sat, so an ``mp.EigenModeSource`` spanning
    the whole folded axis was refused because its centre happened to lie on the plane
    — costing five of MEEP's own ``python/tests`` cases (``test_mode_coeffs`` x3,
    ``test_special_kz`` x2) and 38 MEEP-authored assertions whose folded answers are
    right.

    MEEP does not cancel anything: ``add_volume_source`` deposits with
    ``use_symmetry=false`` (sources.cpp:487), so the half of the source in the
    discarded half is never deposited and is supplied at readback as
    ``parity * (owned value)``. The folded run therefore reproduces the declared run
    exactly when the declared current ALREADY has that parity. Measured in stock MEEP
    1.33.0, folded vs unfolded at the pulse peak over a shared sub-volume, 10x6 cell
    at resolution 10, every source centred on x = 0 and driving a component the plane
    makes odd:

        zero-extent point,      Mirror(X, -1)   rel L2  1.006
        uniform sheet |x|<=2,   Mirror(X, -1)   rel L2  2.829
        odd sheet sin(pi x/4),  Mirror(X, -1)   rel L2  3.71e-07
        uniform sheet |x|<=2,   Mirror(X, +1)   rel L2  0.569
        odd sheet sin(pi x/4),  Mirror(X, +1)   rel L2  4.05e-07

    so the rule has to separate the third and fifth rows from the rest. Extent alone
    does not: both uniform sheets span the folded axis and are 57% and 283% wrong.
    This pins all three verdicts on every axis at both phases, and pins them for a
    component the plane makes odd only — the even components are the other test's job.
    """
    from .fields import mirror_parity

    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0),
                symmetry=(Mirror("XYZ"[axis], phase),))
    component = next(c for c in _DRIVABLE if mirror_parity(c, axis, phase) < 0)
    on_the_plane = (0.0, 0.0, 0.0)
    spanning = [0.0, 0.0, 0.0]
    spanning[axis] = 1.0

    # 1. Zero extent on the plane: the parity condition constrains the source against
    #    itself and admits only zero current. Stock MEEP runs it and is 1.006 wrong.
    assert _refusal(grid, component, on_the_plane) is not None, (
        f"a zero-extent {component} source on the {'XYZ'[axis]} plane must still be refused"
    )
    # 2. Extent across the plane but a UNIFORM profile: a constant is even about every
    #    plane, so the demanded odd parity is impossible by inspection.
    uniform = _refusal(grid, component, on_the_plane, tuple(spanning))
    assert uniform is not None, (
        f"a uniform {component} sheet spanning {'xyz'[axis]} is even about the plane and "
        f"cannot carry parity -1; stock MEEP answers it 57%-283% wrong"
    )
    assert "uniform" in uniform, f"the refusal must name the profile as the cause: {uniform}"
    # 3. Extent across the plane WITH a profile: accepted. This is the case the five
    #    refused corpus rows are, and the case MEEP reproduces at 3.7e-07.
    assert _refusal(grid, component, on_the_plane, tuple(spanning),
                    lambda dx, dy, dz: 1.0 + 0j) is None, (
        f"an extended {component} sheet with an amp_func must be accepted — this is the "
        f"declaration test_mode_coeffs and test_special_kz make, and MEEP's folded answer "
        f"for it matches its unfolded one to 4.3e-06"
    )
    # 4. And the extent has to be on the FOLDED axis. A sheet extended only across the
    #    other two still meets the plane as a zero-extent line and is refused, which is
    #    what says the rule reads size[axis] and not "size is nonzero somewhere".
    sideways = [0.5, 0.5, 0.5]
    sideways[axis] = 0.0
    assert _refusal(grid, component, on_the_plane, tuple(sideways),
                    lambda dx, dy, dz: 1.0 + 0j) is not None, (
        f"a {component} sheet with no extent along {'xyz'[axis]} lies IN the plane; extent "
        f"on the other axes cannot give it the parity the fold demands"
    )


def test_every_offending_mirror_plane_is_named_not_only_the_first():
    """One declaration, two folds, one message. Naming the first sent a search wrong.

    An ``mp.EigenModeSource`` expands into four component sheets and a cell can carry
    more than one mirror plane, so the refusal a user actually sees was whichever
    (component, plane) pair happened to be constructed first. On the five corpus rows
    that was ``Ey`` while ``Hz`` was refused too — recorded in-process — and the
    single-component message read as a single-component defect.

    ``Ex`` is odd about BOTH ``Mirror('X', +1)`` and ``Mirror('Y', -1)``: the first
    because a true vector is odd about the plane it is normal to, the second because
    the odd phase inverts every sign. Both must appear, and each must name its own
    plane's whole odd set so a sibling sheet is not a second surprise.
    """
    from .fields import mirror_parity

    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0),
                symmetry=(Mirror("X", +1), Mirror("Y", -1)))
    assert mirror_parity("Ex", 0, +1) < 0 and mirror_parity("Ex", 1, -1) < 0
    message = _refusal(grid, "Ex")
    assert message is not None
    assert "Mirror('X', phase=+1)" in message, message
    assert "Mirror('Y', phase=-1)" in message, message
    # The X plane's odd set is {Ex, Hy, Hz}; the odd Y plane's is {Ex, Ez, Hy} — every
    # sheet of a four-component declaration that would raise next, named up front.
    for name in ("Ex", "Hy", "Hz"):
        assert name in message, f"the X plane's odd set must be named in full: {message}"
    for name in ("Ex", "Ez", "Hy"):
        assert name in message, f"the odd Y plane's odd set must be named in full: {message}"


def test_the_parity_refusal_states_meeps_mechanism_and_not_a_cancellation():
    """The message is evidence a user acts on, so it must describe what MEEP does.

    It used to say the fold "cancels it against its own mirror image" and that "the
    implied full-domain current is J - J = 0". No line of MEEP's source supports that:
    ``loop_in_chunks`` with ``use_symmetry=false`` visits only chunks that
    untransformed intersect the source volume (loop_in_chunks.cpp:377), so the
    discarded half is never deposited and no J ever meets a -J. Nor is MEEP's answer a
    null — for the point Ex dipole on x = 0 that the old wording described, stock MEEP
    returns max |Ex| 5.081 against the unfolded run's 0.761.

    Pinned because a wrong mechanism in a refusal is worse than no refusal: it tells
    the reader to look for a zero.
    """
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), symmetry=(Mirror("X", +1),))
    message = _refusal(grid, "Ex")
    assert message is not None
    assert "use_symmetry=false" in message and "sources.cpp:487" in message, message
    assert "cancel" not in message.lower(), (
        f"the refusal still claims a cancellation MEEP does not perform: {message}"
    )
    assert "J - J" not in message, message


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_source_must_sit_in_the_half_every_active_mirror_plane_stores(axis):
    """The centre rule is per axis. Enumerating X and Y let a Z fold place anywhere.

    THE RULE IS THE STORED HALF, not the plane. It was "on the plane" until the two
    were measured apart: MEEP adds sources with ``use_symmetry=false``
    (sources.cpp:487) and images them implicitly, so a source anywhere in the stored
    half is a legal declaration — MEEP's folded run with one source at +1.1 equals
    its unfolded run with that source and its parity-signed image to 1.6e-07..3.1e-07
    across five (component, phase) combinations, and this engine reproduces MEEP's
    folded answer at 3.8e-07..9.3e-07. Three of MEEP's own examples were refused a
    lift by the stricter reading.

    The DISCARDED half still raises, and is worth more than the old rule was: MEEP
    deposits nothing there and returns a field whose peak is exactly 0 — a complete,
    smooth, entirely empty run.
    """
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), symmetry=("XYZ"[axis],))
    on_the_plane = [0.0, 0.0, 0.0]
    src._validate_symmetry(grid, tuple(on_the_plane))  # Must not raise.
    stored_half = list(on_the_plane)
    stored_half[axis] = 0.5
    src._validate_symmetry(grid, tuple(stored_half))  # Legal: MEEP images it.
    discarded_half = list(on_the_plane)
    discarded_half[axis] = -0.5
    with pytest.raises(ValueError, match=f"With {'XYZ'[axis]} symmetry the stored half"):
        src._validate_symmetry(grid, tuple(discarded_half))
    # Half a cell out on an axis with no plane constrains nothing, either side.
    for sign in (+1.0, -1.0):
        elsewhere = list(on_the_plane)
        elsewhere[(axis + 1) % 3] = sign * 0.5
        src._validate_symmetry(grid, tuple(elsewhere))


def test_a_deposit_landing_on_a_metallic_wall_plane_is_dropped_not_clamped():
    """The wall plane is MEEP's inert zero_metal point; folding it inward was the defect.

    MEEP stores the plane ON a metallic high wall and deposits its Simpson edge
    weight there (`owns` includes the big corner, vec.cpp:445-462), then zeroes
    it before anything reads it (find_metals + zero_metal, boundaries.cpp:187-201
    and :311-340, run FIRST in the same step_boundaries half-step as step_source
    — step.cpp:100-103, :245). This engine keeps the wall as the unstored zero
    ghost, so the faithful image is to DROP the wall ring's weight. Clamping it
    onto the last interior ring instead SUMMED it there, which measured (3x6 Dcyl
    cell, res 10, CW, until 20, no absorber, whole-volume complex rel L2 against
    CPU MEEP 1.33.0): full-radius Ep plane 7.1e-02, Ez 1.28e-01, Hr 1.6e-02, a
    POINT Ep in the last half-open cell 1.00e+00 — its w0 + w1 = 1 collapsed
    onto ring 29, exactly double MEEP's effective w0 — and 3.2e-02 for a
    full-width Ey line in a Cartesian PEC box; with the drop every case sits at
    its interior control's floor (6.9e-07 … 9.6e-07, controls bit-unchanged).
    This test pins that arithmetic at the deposition table, MEEP-free.
    """
    grid = Grid(resolution=10.0, cell_size=(3.0, 0.0, 6.0), cylindrical=True, m=1,
                boundaries={"z": "metallic"})

    # POINT Ep in the last half-open cell: rings 2.9 and 3.0 split its weight
    # w0 = w1 = 1/2; the wall ring is dropped, so ONLY ring 29 receives w0 —
    # half of what the same point deposits one cell in. The old clamp summed
    # both onto ring 29, which is the measured 2x current (rel L2 1.00e+00).
    wall_point = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ey',
                                              center=(2.95, 0.0, 0.3)))
    interior_point = _deposition(ContinuousSource(grid=grid, frequency=0.8, component='Ey',
                                                  center=(2.85, 0.0, 0.3)))
    assert {cell[0] for cell in wall_point} == {29}
    assert {cell[0] for cell in interior_point} == {28, 29}
    total = sum(wall_point.values())
    interior_total = sum(interior_point.values())
    assert total == pytest.approx(interior_total / 2, rel=1e-6), (
        "The wall ring's w1 must be dropped, not folded onto ring 29"
    )

    # Full-radius Ep plane (zone_plate.py's spelling): rings 0..29 with the wall
    # ring dropped. Its extent ends exactly ON the wall, so w1 = 1 and the last
    # interior ring's own weight e1 equals the interior weight — any excess at
    # ring 29 is the wall's e0 = 1/2 folded inward, the old defect.
    plane = _deposition(ExtendedSource(grid=grid, frequency=0.8, component='Ey',
                                       center=(1.5, 0.0, 0.3), size=(3.0, 0.0, 0.0)))
    r_weights = {}
    for (r_index, _, _), value in plane.items():
        r_weights[r_index] = r_weights.get(r_index, 0j) + value
    assert set(r_weights) == set(range(30)), "rings 0..29 live, the wall ring dropped"
    assert r_weights[29] == pytest.approx(r_weights[15], rel=1e-6), (
        "ring 29 must keep its own e1 weight, not e1 + the wall's e0"
    )

    # Half-integer-r stencil (Er): its outermost sample sits at R_max - dx/2, a
    # cell short of the wall — the drop branch is unreachable and full width is
    # measured exact with no change (9.3e-07 in the same sweep).
    er_plane = _deposition(ExtendedSource(grid=grid, frequency=0.8, component='Ex',
                                          center=(1.5, 0.0, 0.3), size=(3.0, 0.0, 0.0)))
    assert {cell[0] for cell in er_plane} == set(range(30))

    # Cartesian PEC box, full-width integer-x (Ey) line: the same rule at both
    # walls — the LOW wall plane is not owned (excluded and re-weighted, MEEP's
    # little_owned_corner clip), the HIGH wall plane is owned-then-zeroed
    # (dropped here). Both edges land exactly on walls, so every surviving cell
    # carries the interior weight.
    box = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), boundaries="metallic")
    line = _deposition(ExtendedSource(grid=box, frequency=0.8, component='Ey',
                                      center=(0.0, -0.35, 0.15), size=(2.0, 0.0, 0.0)))
    x_weights = {}
    for (x_index, _, _), value in line.items():
        x_weights[x_index] = x_weights.get(x_index, 0j) + value
    assert set(x_weights) == set(range(1, 20)), (
        "cells 1..19: the low wall plane is unowned, the high wall plane is inert"
    )
    assert x_weights[1] == pytest.approx(x_weights[10], rel=1e-6)
    assert x_weights[19] == pytest.approx(x_weights[10], rel=1e-6)


def test_a_full_width_source_on_a_folded_live_face_deposits_uniformly():
    """The lattice images of a folded PERIODIC axis land the far end's edge weight.

    MEEP's `loop_in_chunks` keys its lattice-shift loop on the boundary being
    Periodic (loop_in_chunks.cpp:393), not on the symmetry, so a full-width
    source on a mirror-folded periodic axis deposits through the ishift = -1
    image as well as the direct pass: the top of the stored window receives the
    request's OTHER end's boundary weight (0.5 with the volume endpoint ON the
    sample, 1/8 with it midway), and the total per row is uniform — measured
    uniform 1.0 in CPU MEEP 1.33.0 on every parity/stencil combination, against
    this engine's clip-only profile [1, ..., 1, 0.5] (even, shift-0), [1, ...,
    7/8] (even, shift-1) and [1, ..., 1, 0.5] (odd, shift-1) before the image
    pass. Whole-volume cost of the missing weight: 9.5e-03 … 6.0e-02 on the
    corpus family (binary_grating.py, chirped_pulse.py, diffracted_planewave.py,
    binary_grating_phasemap.py), floors of 8e-07 … 1.1e-06 after. This test pins
    the deposition arithmetic MEEP-free, one case per row geometry:

    - EVEN count, shift-0 (Ez): a sample ON the second mirror at +L/2 — 0.5
      direct + 0.5 imaged; stored rows 1..9 uniform, row 0 unowned.
    - EVEN count, shift-1 (Ey): the last live row at doubled n_full - 1 — 7/8
      direct + 1/8 imaged; rows 0..8 uniform, and the far GHOST slot (row 9,
      doubled n_full + 1) receives nothing — `fill_folded_far_ghosts` owns it.
    - ODD count, shift-1 (Ey): a sample ON the second mirror at doubled n_full —
      0.5 + 0.5; rows 0..8 uniform, row 9 the far ghost the fill pass owns.
    - ODD count, shift-0 (Ez): the top rung lands at doubled n_full + 1, MEEP's
      ``big_corner``, which ``owns`` includes (vec.cpp:445-462) and
      ``Grid.stored_cells`` now allocates at BOTH parities — so it is DEPOSITED
      into row 9, not dropped and not clamped onto row 8. The row sets are
      therefore the same as the even count's; only which plane row 9 IS differs.
    """
    def y_profile(grid, component, sy):
        deposit = _deposition(ExtendedSource(grid=grid, frequency=0.8, component=component,
                                             center=(-1.8, 0.0, 0.0), size=(0.0, sy, 0.0)))
        rows: dict = {}
        for (_, y_index, _), value in deposit.items():
            rows[y_index] = rows.get(y_index, 0j) + value
        return rows

    # EVEN full count: 0.32 * 50 = 16 -> 10 stored rows (plane row 9 + ghost slot).
    # Ez rides the even plane; Ey is odd about it (its own axis flips), so the Ey
    # cases declare phase = -1 — the plane that makes Ey even — exactly as the
    # corpus TE scripts do (mp.Mirror(mp.Y, phase=-1) with an Hz drive).
    even = Grid(resolution=50.0, cell_size=(6.0, 0.32, 0.0), dimensions=2,
                symmetry=('Y',))
    even_odd_plane = Grid(resolution=50.0, cell_size=(6.0, 0.32, 0.0), dimensions=2,
                          symmetry=(Mirror('Y', -1),))
    assert even.stored_cells(1) == 10 and even.owned_cells(1) == 9

    ez_even = y_profile(even, 'Ez', 0.32)
    assert set(ez_even) == set(range(1, 10)), (
        "shift-0 rows 1..9: the plane row IS deposited, the below-plane row 0 is unowned"
    )
    for row, total in ez_even.items():
        assert total == pytest.approx(ez_even[5], rel=1e-6), (
            f"row {row} must carry the uniform current; the plane row at half weight "
            f"was the folded-live-face divergence family"
        )

    ey_even = y_profile(even_odd_plane, 'Ey', 0.32)
    assert set(ey_even) == set(range(9)), (
        "shift-1 rows 0..8: the far ghost slot (row 9) belongs to the fill pass, not the source"
    )
    for row, total in ey_even.items():
        assert total == pytest.approx(ey_even[4], rel=1e-6), (
            f"row {row} must carry the uniform current (7/8 direct + 1/8 imaged at the top)"
        )

    # ODD full count: 0.30 * 50 = 15 -> 10 stored rows (big_corner row 9 at doubled
    # n_full + 1, plus the shift-1 ghost slot above it). The second mirror at
    # doubled n_full is a shift-1 sample, row 8's own half-integer slot.
    odd = Grid(resolution=50.0, cell_size=(6.0, 0.30, 0.0), dimensions=2,
               symmetry=('Y',))
    odd_odd_plane = Grid(resolution=50.0, cell_size=(6.0, 0.30, 0.0), dimensions=2,
                         symmetry=(Mirror('Y', -1),))
    assert odd.stored_cells(1) == 10 and odd.owned_cells(1) == 9

    ez_odd = y_profile(odd, 'Ez', 0.30)
    assert set(ez_odd) == set(range(1, 10)), (
        "shift-0 rows 1..9: the top rung IS MEEP's big_corner sample (owned, stepped, "
        "and stored at both parities), so it is deposited — dropping it, or clamping "
        "it onto row 8, both lose the far end's boundary weight"
    )
    for row, total in ez_odd.items():
        assert total == pytest.approx(ez_odd[4], rel=1e-6), (
            f"row {row} must carry the uniform current (7/8 direct + 1/8 imaged at the top)"
        )

    ey_odd = y_profile(odd_odd_plane, 'Ey', 0.30)
    assert set(ey_odd) == set(range(9)), (
        "shift-1 rows 0..8, the second-mirror row 8 included; row 9 is the far ghost "
        "the fill pass owns"
    )
    for row, total in ey_odd.items():
        assert total == pytest.approx(ey_odd[4], rel=1e-6), (
            f"row {row} must carry the uniform current (0.5 direct + 0.5 imaged on the plane)"
        )

    # A WINDOWED source (clear of both planes) has no image pass to receive:
    # every deposited row carries the interior weight, exactly as before the fix.
    windowed = y_profile(even, 'Ez', 0.16)
    interior_rows = sorted(windowed)
    assert 9 not in windowed and 0 not in windowed
    for row in interior_rows[1:-1]:
        assert windowed[row] == pytest.approx(windowed[interior_rows[len(interior_rows) // 2]],
                                              rel=1e-6)


def test_the_fold_helpers_defer_to_the_grid_and_cover_every_axis_in_the_fallback():
    """`_axis_is_mirrored` / `_axis_mirror_phase` must ASK the grid, and know about Z.

    The counterpart of the same check in test_dft.py, for this module's copies. They
    exist for the stub grids here, which carry the per-axis flags and no methods; the
    two failure modes are answering from the flags when the grid has its own rule, and
    a fallback that enumerates X and Y so a folded Z reads as unfolded — which is
    exactly how a Z fold used to skip its source-parity check.
    """
    from .sources import _axis_is_mirrored, _axis_mirror_phase

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
    real = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), symmetry=(Mirror("Z", -1),))
    assert [_axis_is_mirrored(real, axis) for axis in range(3)] == [False, False, True]
    assert [_axis_mirror_phase(real, axis) for axis in range(3)] == [None, None, -1]


def test_unsplit_deposit_flags_follow_the_component_ladder_axes_and_the_chunk_rule():
    """Which deposits mirror into f_u: sigma bites on dsig AND outside every dsigu chunk.

    The truth table this encodes is the evidence packet's (§1.5-1.7), measured against
    CPU MEEP on the 12x12 cell: an Ez deposit deep in the x layer (dsig) but outside
    the y layer's CHUNK mirrors (4.9581e-01 -> 5.6e-07); the corner, the y-only cell
    and the interior do not; and the chunk's own boundary cells — sigma exactly zero,
    ownership per structure.cpp:509-524 — do not either (mirroring one read 4.59e-01).
    The axes come from stepping's curl tables per component (Dz: dsig = x, dsigu = y),
    an integrated source never mirrors (MEEP's step_source skips it, step.cpp:300),
    and a CYLINDRICAL grid takes the same path: under x = R, y = P, z = Z the same
    tables equal MEEP's Dcyl cycle_direction start = 2 binding (vec.hpp:583-589),
    the phi axis never carries a layer, so an Ez deposit under the r layer mirrors
    everywhere sigma bites — corners INCLUDED, the row where Dcyl differs from the
    Cartesian table (dsigu = P never rescues it; measured 7.05e+01 -> 2.04e-05 on
    the sweep's corner case with every control bit-unchanged).
    """
    from .pml import PML
    from .sources import _unsplit_deposit_flags

    grid = Grid(resolution=20.0, cell_size=(12.0, 12.0, 0.0), dimensions=2)
    layer = PML(grid, thickness={"x": 40, "y": 40})
    points = [
        (20, 120, 0),   # deep x layer, y interior      -> mirrors
        (20, 40, 0),    # deep x layer, y chunk boundary (cell 40, sigma_y = 0) -> no
        (20, 41, 0),    # ... and the owned plane past it (cell 41)             -> no
        (20, 42, 0),    # first cell MEEP's interior chunk owns                 -> mirrors
        (20, 30, 0),    # the corner: both layers       -> no (both-active everywhere)
        (120, 30, 0),   # y layer only (sigma_dsig = 0) -> no
        (120, 120, 0),  # interior                      -> no
        (40, 120, 0),   # AT the x extent, sigma_x exactly 0 -> no (nothing to damp)
    ]
    flags = _unsplit_deposit_flags(layer, grid, "Ez", points)
    assert flags == [True, False, False, True, False, False, False, False]
    # Hz drives Bz: dsig = x, dsigu = y at HALF-integer offsets — cell 40 is already
    # outside the half-integer sigma support, and the y chunk keeps cells 0..40.
    flags_hz = _unsplit_deposit_flags(layer, grid, "Hz", points)
    assert flags_hz == [True, False, True, True, False, False, False, False]
    # No mirroring anywhere it cannot be right: every flag False collapses to None.
    assert _unsplit_deposit_flags(layer, grid, "Ez", [(120, 120, 0)]) is None
    # A cylindrical grid runs the same rule on the same tables. Ez (dsig = R,
    # dsigu = P): a deposit under the r-high layer mirrors — in the z layer's
    # CORNER too, because no phi layer exists to put it in a both-active chunk —
    # while the interior, the z-only cell and the sigma-zero inner edge do not.
    cyl_grid = Grid(resolution=10.0, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=0,
                    boundaries={"z": "metallic"})
    cyl_layer = PML(cyl_grid, thickness={"x": (0, 5), "z": (5, 5)})
    cyl_points = [
        (18, 0, 20),   # deep r layer, z interior      -> mirrors
        (18, 0, 2),    # deep r layer, deep z layer    -> mirrors (the Dcyl corner row)
        (5, 0, 2),     # z layer only (sigma_dsig = 0) -> no
        (5, 0, 20),    # interior                      -> no
        (15, 0, 20),   # AT the r extent, sigma_r exactly 0 -> no
    ]
    assert _unsplit_deposit_flags(cyl_layer, cyl_grid, "Ez", cyl_points) == [
        True, True, False, False, False]
    # Ep drives Dy: dsig = z, dsigu = x — the z layer mirrors it only OUTSIDE the
    # r layer's chunk (which owns one integer plane past its sigma support).
    assert _unsplit_deposit_flags(cyl_layer, cyl_grid, "Ey", cyl_points) == [
        False, False, True, False, False]
