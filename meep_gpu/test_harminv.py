"""
Validation for :mod:`meep_gpu.harminv` — harmonic inversion by filter diagonalization.

The backbone is signals whose answer is known in closed form: sums of decaying
complex exponentials built from chosen frequencies, decay rates, amplitudes and
phases, so every reported number has an exact target rather than a reference
array. Those tests need nothing installed. On top of them sits a cross-validation
against CPU MEEP's own ``mp.py_do_harminv`` (the external harminv C library) on
identical signals, run in a subprocess because MEEP must not share a process with
anything else here.

WHAT THE NUMBERS COME OUT AT, measured (relative, worst case over the sweeps below):

* single mode, clean, swept over odd and even record lengths, three band
  placements and an offset start: freq 4.6e-15, decay 3.6e-12, amp 4.8e-15
* two modes 1/100 of a Fourier bin apart: freq 1e-10, both recovered — an FFT of
  the same record shows ONE peak. This is the case that separates filter
  diagonalization from a peak-find and it is pinned as such. Splitting still
  works at 1/300 of a bin, and in exploration down to 1/1000.
* heavily damped, Q from 200 down to 0.5:  Q 1.4e-12
* vs CPU MEEP on the same clean signals: freq 1.1e-11, decay 2.7e-8, Q 2.7e-8,
  amp 8.0e-9. On the noisy record the two codes fit the noise differently and
  land 8.9e-7 apart in frequency and 1.3e-4 in Q, both around the true value.
* noise: the reported Q degrades roughly linearly in the noise-to-signal ratio,
  ~1.7x it — 1e-3 noise on a unit mode gives Q to 0.2%, 1e-1 gives 17%. The mode
  is still found at a noise level of 3x the signal, with ``err`` risen from 1e-15
  to 1e-3, which is the point of ``err``.

THE FAILURE MODE THIS FILE IS BUILT AGAINST is a confident wrong answer. Harmonic
inversion will happily fit modes to noise, and the danger is not that it fails but
that it returns a plausible frequency with a plausible Q. So: pure noise, a
constant signal, an all-zero signal, a repeated pole and a record shorter than the
requested basis each get a test asserting refusal or a large error, and the
pure-noise case is compared seed by seed against what MEEP itself reports rather
than against a wish. Sample counts are swept ODD and EVEN and the mode is placed
off band centre and near both band edges, because a half-sample offset that only
shows up at odd counts is a documented failure in this package's history.

Its mirror image — a real result silently screened away — has its own tests, since
an empty mode list reads as "this cavity has no resonances". Two ways to reach one
are pinned: an error estimate of exactly zero meeting an infinite
``rel_err_thresh`` (inf * 0 is NaN, which compares false against everything), and a
NaN threshold.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from .harminv import (
    DEFAULT_RANK_TOL,
    Harminv,
    Mode,
    _basis_size,
    _correlation_length,
    _generate_U,
    _resolve_basis_size,
    _solve_modes,
    do_harminv,
    harminv,
)

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)

SAMPLE_INTERVAL = 0.05  # MEEP time units between samples; the dt of every signal below.


def synthesize(num_samples, dt, terms, start_time=0.0):
    """Sum of decaying complex exponentials, MEEP's convention.

    ``terms`` is ``[(complex amplitude, freq, decay)]`` and the signal is
    ``sum a * exp(-2i*pi*(freq + i*decay)*t)`` sampled at ``start_time + k*dt``.
    A DECAYING mode has ``decay < 0``, matching ``Mode.decay``; the sign is written
    out here because getting it backwards turns loss into gain silently.
    """
    time = start_time + np.arange(num_samples) * dt
    signal = np.zeros(num_samples, dtype=np.complex128)
    for amplitude, freq, decay in terms:
        signal = signal + amplitude * np.exp(-1j * 2 * np.pi * (freq + 1j * decay) * time)
    return signal


def nearest_mode(modes, target_freq):
    return min(modes, key=lambda mode: abs(mode.freq - target_freq)) if modes else None


def expected_quality(freq, decay):
    return abs(freq) / (-2 * decay)


# --------------------------------------------------------------------------------------
# Exact recovery from signals with a known closed-form answer
# --------------------------------------------------------------------------------------


def test_single_mode_recovers_frequency_decay_amplitude_and_phase():
    """One decaying exponential: every reported number has an exact target."""
    freq, decay = 0.30, -0.001
    amplitude = 1.3 * np.exp(1j * 0.4)
    signal = synthesize(400, SAMPLE_INTERVAL, [(amplitude, freq, decay)])

    modes = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10)

    assert len(modes) == 1
    mode = modes[0]
    assert abs(mode.freq - freq) / freq < 1e-12
    assert abs(mode.decay - decay) / abs(decay) < 1e-9
    assert abs(mode.Q - expected_quality(freq, decay)) / expected_quality(freq, decay) < 1e-9
    assert abs(mode.amp - amplitude) / abs(amplitude) < 1e-12
    assert abs(np.angle(mode.amp) - 0.4) < 1e-12
    assert mode.err < 1e-12


def test_two_well_separated_modes_are_both_recovered():
    signal = synthesize(
        1500,
        SAMPLE_INTERVAL,
        [(1.0, 0.26, -3e-4), (0.4 * np.exp(-1j * 2.0), 0.34, -8e-4)],
    )

    modes = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10)

    assert len(modes) == 2
    assert [round(mode.freq, 9) for mode in modes] == [0.26, 0.34]  # sorted by |freq|
    assert abs(modes[0].amp - 1.0) < 1e-10
    assert abs(modes[1].amp - 0.4 * np.exp(-1j * 2.0)) < 1e-10
    assert abs(modes[0].decay + 3e-4) < 1e-12
    assert abs(modes[1].decay + 8e-4) < 1e-12


def test_two_closely_spaced_modes_beat_the_fourier_limit():
    """The test that proves this is filter diagonalization and not a peak-find.

    The two lines sit 1/100 of a Fourier bin apart. The periodogram of the same
    record has a single peak — asserted here, not assumed — and harminv still
    returns two modes with ten correct digits each.
    """
    num_samples = 2000
    record_length = num_samples * SAMPLE_INTERVAL
    fourier_bin = 1.0 / record_length
    separation = 0.01 * fourier_bin
    first, second = 0.30, 0.30 + separation
    signal = synthesize(
        num_samples,
        SAMPLE_INTERVAL,
        [(1.0, first, -2e-4), (0.7 * np.exp(1j * 1.1), second, -3e-4)],
    )

    # An FFT cannot see two lines here: both fall inside one bin, and the
    # periodogram has exactly one local maximum across the search band. MEEP's
    # exp(-i*omega*t) convention puts a mode of frequency f in NumPy's -f bin, so
    # the frequency axis is negated before the band is cut out.
    spectrum = np.abs(np.fft.fft(signal))
    frequencies = -np.fft.fftfreq(num_samples, SAMPLE_INTERVAL)
    assert int(first / fourier_bin) == int(second / fourier_bin)
    in_band = (frequencies > 0.25) & (frequencies < 0.35)
    ordering = np.argsort(frequencies[in_band])
    band_spectrum = spectrum[in_band][ordering]
    local_maxima = np.flatnonzero(
        (band_spectrum[1:-1] > band_spectrum[:-2]) & (band_spectrum[1:-1] > band_spectrum[2:])
    )
    assert local_maxima.size == 1, "the FFT premise of this test no longer holds"

    modes = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.35, 10)

    assert len(modes) == 2, f"failed to split lines {separation:.2e} apart"
    assert abs(modes[0].freq - first) / first < 1e-10
    assert abs(modes[1].freq - second) / second < 1e-10
    assert abs(modes[0].decay + 2e-4) / 2e-4 < 1e-6
    assert abs(modes[1].decay + 3e-4) / 3e-4 < 1e-6
    assert abs(modes[0].amp - 1.0) < 1e-6
    assert abs(modes[1].amp - 0.7 * np.exp(1j * 1.1)) < 1e-6
    assert max(mode.err for mode in modes) < 1e-9


@pytest.mark.parametrize("separation_in_bins", [1.0, 0.3, 0.1, 0.03, 0.01, 0.003])
def test_close_mode_resolution_degrades_gracefully(separation_in_bins):
    """Splitting survives to 1/300 of a Fourier bin; the error grows, it does not lie."""
    num_samples = 2000
    separation = separation_in_bins / (num_samples * SAMPLE_INTERVAL)
    first, second = 0.30, 0.30 + separation
    signal = synthesize(
        num_samples, SAMPLE_INTERVAL, [(1.0, first, -2e-4), (0.7, second, -3e-4)]
    )

    modes = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.35, 10)

    assert len(modes) == 2
    assert abs(modes[0].freq - first) / first < 1e-9
    assert abs(modes[1].freq - second) / second < 1e-9


@pytest.mark.parametrize("quality", [200.0, 50.0, 20.0, 5.0, 2.0, 1.0, 0.5])
def test_heavily_damped_modes(quality):
    """Q from a good cavity down to a mode that dies inside one period."""
    freq = 0.3
    decay = -freq / (2 * quality)
    lifetime_samples = int(6 / abs(2 * np.pi * decay) / SAMPLE_INTERVAL)
    num_samples = min(max(200, lifetime_samples), 4000)
    signal = synthesize(num_samples, SAMPLE_INTERVAL, [(2.0, freq, decay)])

    # Q_thresh=0 because MEEP's default screening deliberately discards Q < 50.
    modes = harminv(signal, SAMPLE_INTERVAL, 0.05, 0.55, 10, Q_thresh=0.0)

    mode = nearest_mode(modes, freq)
    assert mode is not None, f"lost the Q={quality} mode entirely"
    assert abs(mode.freq - freq) / freq < 1e-11
    assert abs(mode.Q - quality) / quality < 1e-9
    assert abs(mode.amp - 2.0) / 2.0 < 1e-11


def test_growing_mode_is_reported_as_gain_not_loss():
    """A mode with positive decay must come back with positive decay and negative Q.

    A sign flip anywhere in the omega = i*ln(u) chain turns gain into loss while
    leaving |freq| and |Q| right, which no magnitude assertion would catch.
    """
    freq, decay = 0.3, +0.0008
    signal = synthesize(600, SAMPLE_INTERVAL, [(1.0, freq, decay)])

    modes = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10, Q_thresh=0.0)

    assert len(modes) == 1
    assert modes[0].decay > 0
    assert abs(modes[0].decay - decay) / decay < 1e-9
    assert modes[0].Q < 0
    assert abs(modes[0].Q - freq / (-2 * decay)) / abs(freq / (2 * decay)) < 1e-9


@pytest.mark.parametrize("num_samples", [57, 58, 401, 402, 403, 999, 1000, 1001])
@pytest.mark.parametrize("freq", [0.2103, 0.30, 0.3897])
@pytest.mark.parametrize("start_time", [0.0, 0.017])
def test_odd_and_even_counts_off_centre_bands_and_offset_start(num_samples, freq, start_time):
    """Sweep the shapes that have hidden half-sample bugs in this package before.

    Odd and even record lengths, the mode at band centre and hard against both
    edges, and a record that does not start at t = 0.
    """
    decay = -0.001
    amplitude = 1.3 * np.exp(1j * 0.4)
    signal = synthesize(num_samples, SAMPLE_INTERVAL, [(amplitude, freq, decay)], start_time)
    # The amplitude is referenced to the FIRST SAMPLE, not to t = 0.
    expected_amplitude = amplitude * np.exp(-1j * 2 * np.pi * (freq + 1j * decay) * start_time)

    modes = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10)

    assert len(modes) == 1
    mode = modes[0]
    assert abs(mode.freq - freq) / freq < 1e-12
    assert abs(mode.decay - decay) / abs(decay) < 1e-8
    assert abs(mode.amp - expected_amplitude) / abs(expected_amplitude) < 1e-11


def test_amplitude_scales_linearly_and_frequency_does_not():
    """Scaling the record scales amp and leaves freq, decay, Q and err alone."""
    signal = synthesize(800, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001), (0.5, 0.32, -0.002)])

    baseline = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.37, 10)
    scaled = harminv(1e-7 * signal, SAMPLE_INTERVAL, 0.25, 0.37, 10)

    assert len(baseline) == len(scaled) == 2
    for original, other in zip(baseline, scaled):
        assert abs(original.freq - other.freq) < 1e-12
        assert abs(original.decay - other.decay) < 1e-14
        assert abs(other.amp - 1e-7 * original.amp) < 1e-16
        assert other.err < 1e-9


def test_real_valued_signal_collapses_the_conjugate_pair():
    """A real record answers with +f and -f; MEEP reports one, and so do we."""
    complex_signal = synthesize(
        2000,
        SAMPLE_INTERVAL,
        [(1.0, 0.300, -2e-4), (0.7 * np.exp(1j * 1.1), 0.302, -3e-4)],
    )

    modes = harminv(np.real(complex_signal), SAMPLE_INTERVAL, 0.25, 0.35, 10)

    assert len(modes) == 2, "the negative-frequency images were not collapsed"
    assert all(mode.freq > 0 for mode in modes)
    assert abs(modes[0].freq - 0.300) < 1e-11
    assert abs(modes[1].freq - 0.302) < 1e-11
    # Re(a e^-iwt) splits the amplitude evenly between the +w and -w lines.
    # Which member of the pair survives decides whether the reported amplitude is
    # a or its conjugate, because MEEP reports every mode at |freq| (bands.cpp:189)
    # and keeps whichever of the pair has the smaller error (bands.cpp:160-166).
    # Here the survivor is always the +f one, and not by luck: the trial basis is
    # placed across the POSITIVE band, so the -f image is always the worse
    # conditioned of the two and always loses the tie-break. MEEP's own error
    # measure carries no such bias and its choice does flip with record length,
    # which is why the parity test below compares |amp| on real-valued input.
    assert abs(modes[0].amp - 0.5) < 1e-9
    assert abs(modes[1].amp - 0.5 * 0.7 * np.exp(1j * 1.1)) < 1e-9


@pytest.mark.parametrize("num_samples", [1500, 1777, 2000, 2001, 2500, 3000])
def test_real_signal_amplitude_phase_is_not_conjugated_at_any_record_length(num_samples):
    """The +f member wins the tie-break at every length, so the phase sign is stable."""
    signal = np.real(
        synthesize(num_samples, SAMPLE_INTERVAL, [(0.7 * np.exp(1j * 1.1), 0.302, -3e-4)])
    )

    modes = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.35, 10)

    mode = nearest_mode(modes, 0.302)
    assert mode is not None
    assert abs(mode.amp - 0.5 * 0.7 * np.exp(1j * 1.1)) < 1e-6
    assert np.angle(mode.amp) > 0, "amplitude came back conjugated"


def test_accepts_a_device_array_by_copying_to_the_host_once():
    """Post-processing is host-side by design; a CuPy-like array must still work."""

    class FakeDeviceArray:
        def __init__(self, host):
            self._host = host

        def get(self):  # The only interface backends.to_numpy uses.
            return self._host

    signal = synthesize(400, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])

    modes = harminv(FakeDeviceArray(signal), SAMPLE_INTERVAL, 0.2, 0.4, 10)

    assert len(modes) == 1
    assert abs(modes[0].freq - 0.3) < 1e-12


# --------------------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------------------


def test_noise_degrades_recovery_gradually_and_the_error_estimate_tracks_it():
    """Quantify where recovery breaks down. See the module docstring for the summary.

    Measured on a unit-amplitude f = 0.3, Q = 150 mode in a 1000-sample record with
    complex white noise added. The relative Q error stays under about twice the
    noise-to-signal ratio, and ``err`` rises from 1e-15 to 1e-3 across the sweep,
    which is the only signal a user has that the fit is being pushed.
    """
    num_samples = 1000
    freq, decay = 0.3, -0.001
    quality = expected_quality(freq, decay)
    clean = synthesize(num_samples, SAMPLE_INTERVAL, [(1.0, freq, decay)])
    generator = np.random.default_rng(7)

    previous_error_estimate = 0.0
    for noise_level, quality_bound in (
        (1e-6, 1e-5),
        (1e-4, 1e-3),
        (1e-3, 1e-2),
        (1e-2, 5e-2),
        (1e-1, 4e-1),
    ):
        worst_quality_error = 0.0
        worst_error_estimate = 0.0
        for _ in range(4):
            noise = generator.normal(size=num_samples) + 1j * generator.normal(size=num_samples)
            modes = harminv(clean + noise_level * noise, SAMPLE_INTERVAL, 0.2, 0.4, 10)
            mode = nearest_mode(modes, freq)
            assert mode is not None, f"lost the mode entirely at noise {noise_level}"
            assert abs(mode.freq - freq) / freq < 1e-2
            worst_quality_error = max(worst_quality_error, abs(mode.Q - quality) / quality)
            worst_error_estimate = max(worst_error_estimate, mode.err)
        assert worst_quality_error < quality_bound, (
            f"noise {noise_level}: Q error {worst_quality_error:.2e} exceeds {quality_bound}"
        )
        assert worst_error_estimate > previous_error_estimate, (
            "err must grow with noise or it is not reporting confidence"
        )
        previous_error_estimate = worst_error_estimate

    assert previous_error_estimate > 1e-6, (
        "err stayed tiny at 10% noise, so it is not measuring anything"
    )


def test_a_mode_buried_in_noise_is_still_found_but_flagged():
    """At a noise level three times the mode, the fit survives and ``err`` says so."""
    num_samples = 2000
    clean = synthesize(num_samples, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])
    generator = np.random.default_rng(21)
    noise = generator.normal(size=num_samples) + 1j * generator.normal(size=num_samples)

    modes = harminv(clean + 3.0 * noise, SAMPLE_INTERVAL, 0.28, 0.32, 10)

    mode = nearest_mode(modes, 0.3)
    assert mode is not None
    assert abs(mode.freq - 0.3) / 0.3 < 5e-2
    assert mode.err > 1e-6, "a mode fitted through 3x noise must not report full confidence"


# --------------------------------------------------------------------------------------
# Degenerate inputs: refuse or flag, never invent a resonance
# --------------------------------------------------------------------------------------


def test_all_zero_record_returns_no_modes():
    assert harminv(np.zeros(400), SAMPLE_INTERVAL, 0.2, 0.4) == []
    assert do_harminv(np.zeros(400), SAMPLE_INTERVAL, 0.2, 0.4) == []


def test_constant_record_returns_no_modes():
    """A DC record is a zero-frequency, zero-Q mode: outside the band and below Q_thresh."""
    assert harminv(np.ones(400), SAMPLE_INTERVAL, 0.2, 0.4) == []
    assert harminv(np.full(400, 3.5 + 1j), SAMPLE_INTERVAL, 0.2, 0.4) == []
    # It is not that we find nothing — with the screening off, the DC line is there.
    unscreened = harminv(
        np.ones(400), SAMPLE_INTERVAL, 0.0001, 0.4, Q_thresh=-1.0, err_thresh=math.inf
    )
    assert unscreened == [] or all(abs(mode.freq) < 1e-6 for mode in unscreened)


@pytest.mark.parametrize("seed", range(12))
def test_pure_noise_returns_no_confident_mode(seed):
    """Noise has no resonance. Anything reported must carry a large error estimate.

    Filter diagonalization always finds *something*; the contract is that what it
    finds is flagged. A genuine clean mode scores err ~1e-15 here, so the floor of
    1e-6 asserted below is five orders of margin, and the amplitudes are a small
    fraction of the record's own RMS.
    """
    generator = np.random.default_rng(seed)
    noise = generator.normal(size=800) + 1j * generator.normal(size=800)
    noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2)))

    modes = harminv(noise, SAMPLE_INTERVAL, 0.2, 0.4, 10)

    assert len(modes) <= 3, f"invented {len(modes)} resonances in white noise"
    for mode in modes:
        assert mode.err > 1e-6, (
            f"noise mode at f={mode.freq:.4f} reported err={mode.err:.2e}, "
            "which is the confidence of a real resonance"
        )
        assert abs(mode.amp) < 0.2 * noise_rms


def test_record_shorter_than_the_requested_basis_refuses():
    signal = synthesize(400, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])
    correlation_length = _correlation_length(400)

    with pytest.raises(ValueError, match="exceeds what a 400-sample record supports"):
        harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, nf=correlation_length + 1)

    # One below the limit is accepted, so the boundary is where it claims to be.
    assert harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, nf=correlation_length) is not None


def test_a_derived_basis_is_capped_by_the_record_rather_than_overrunning_it():
    """mxbands=100 on a short record must not ask for a basis bigger than the data."""
    signal = synthesize(60, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])

    modes = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4)  # mxbands defaults to 100, K is 29

    assert len(modes) == 1
    assert abs(modes[0].freq - 0.3) / 0.3 < 1e-9


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(dt=0.0), "positive finite sample interval"),
        (dict(dt=-0.05), "positive finite sample interval"),
        (dict(dt=math.nan), "positive finite sample interval"),
        (dict(fmin=0.4, fmax=0.2), "empty or inverted"),
        (dict(fmin=0.3, fmax=0.3), "empty or inverted"),
        (dict(fmin=math.inf), "band must be finite"),
        (dict(mxbands=0), "mxbands must be at least 1"),
        (dict(spectral_density=0.0), "spectral_density must be positive"),
        (dict(rank_tol=1.0), "rank_tol must lie"),
        (dict(rank_tol=-1e-9), "rank_tol must lie"),
        (dict(nf=1), "nf must be at least 2"),
    ],
)
def test_invalid_arguments_raise_rather_than_defaulting(kwargs, message):
    defaults = dict(dt=SAMPLE_INTERVAL, fmin=0.2, fmax=0.4)
    defaults.update(kwargs)
    signal = synthesize(400, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])

    with pytest.raises(ValueError, match=message):
        harminv(signal, **defaults)


def test_non_finite_samples_raise():
    """A diverged run must not be harmonically inverted into a plausible mode."""
    signal = synthesize(400, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])
    diverged = signal.copy()
    diverged[137] = np.inf

    with pytest.raises(ValueError, match="non-finite sample"):
        harminv(diverged, SAMPLE_INTERVAL, 0.2, 0.4)

    nan_signal = signal.copy()
    nan_signal[0] = np.nan
    with pytest.raises(ValueError, match="non-finite sample"):
        harminv(nan_signal, SAMPLE_INTERVAL, 0.2, 0.4)


def test_multidimensional_input_raises():
    with pytest.raises(ValueError, match="1-D time series"):
        harminv(np.zeros((8, 8)), SAMPLE_INTERVAL, 0.2, 0.4)


def test_a_record_too_short_for_any_basis_raises():
    with pytest.raises(ValueError, match="at least 6 samples"):
        harminv(np.ones(5), SAMPLE_INTERVAL, 0.2, 0.4)


@pytest.mark.parametrize("num_samples", [284, 285])
def test_a_mode_fitted_to_the_last_bit_is_not_screened_out_by_inf_times_zero(num_samples):
    """A perfect fit gives err at machine zero, and MEEP's default rel_err_thresh is inf.

    inf * 0 is NaN and every comparison against NaN is false, so the relative-error
    screen would drop EVERY mode and return an empty list for a flawless signal —
    an unmeasurable result wearing the clothes of "this cavity has no resonances".
    Upstream hit the same pathology: MEEP #2959 changed bands.cpp's screens from
    ``err <`` to ``err <=`` so a perfectly fitted mode stops screening ITSELF out
    against ``rel_err_thresh * min_err``; this engine transcribed the post-#2959
    convention and additionally spells inf-times-zero as "no limit" outright.

    The err the eigensolve returns for these two record lengths is EXACTLY 0.0 on
    the environment this test was written on, and 1.1e-16 on another (a different
    numpy/LAPACK — measured under the MEEP-1.33 oracle env). Asserting ``== 0.0``
    therefore tested the linear-algebra library, not this engine; the machine-zero
    BOUND below keeps the precondition honest on both, and the exact-zero
    arithmetic itself is pinned platform-independently by the ``_screen_modes``
    unit test that follows.
    """
    time = np.arange(num_samples) * SAMPLE_INTERVAL
    signal = np.exp(-1j * 2 * np.pi * 0.5 * time)

    modes = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.55, 4, nf=4, Q_thresh=0.0)

    assert len(modes) == 1, "a perfectly fitted mode was screened out"
    assert abs(modes[0].freq - 0.5) < 1e-12
    assert modes[0].err < 1e-14, (
        f"err = {modes[0].err!r}: this record no longer fits to machine zero, so the "
        f"inf-times-min_err path is not the one being exercised")
    # And the relative screen still works when it is given a finite ratio.
    assert harminv(
        signal, SAMPLE_INTERVAL, 0.2, 0.55, 4, nf=4, Q_thresh=0.0, rel_err_thresh=2.0
    ) == modes


def test_screen_modes_keeps_an_exact_zero_error_mode_under_infinite_rel_thresh():
    """The inf * 0 arithmetic itself, pinned with a literal 0.0 — no LAPACK in the loop.

    The end-to-end test above depends on the eigensolve actually reaching err == 0,
    which the linear-algebra backend decides; this one hands ``_screen_modes`` an
    errors array CONTAINING exact zero, so the NaN-screen regression cannot hide
    behind a backend that returns 1e-16 instead. Both modes here are perfect fits;
    both must survive the default infinite rel_err_thresh, and the finite-ratio
    screen must still drop the worse one when min_err is nonzero.
    """
    from .harminv import _screen_modes

    frequencies = np.array([0.5 - 0.0j, 0.30 - 0.001j], dtype=complex)
    amplitudes = np.array([1.0 + 0.0j, 0.5 + 0.0j], dtype=complex)
    errors = np.array([0.0, 0.0], dtype=float)
    kept = _screen_modes(frequencies, amplitudes, errors, 64, SAMPLE_INTERVAL,
                         0.2, 0.55, 4, Q_thresh=0.0, rel_err_thresh=math.inf,
                         err_thresh=0.01, rel_amp_thresh=-1.0, amp_thresh=0.0)
    assert sorted(kept) == [0, 1], (
        f"kept {kept}: exact-zero-error modes were screened by inf * 0 (the NaN "
        f"comparison pathology)")
    # min_err = 0 with a FINITE ratio: 2.0 * 0.0 = 0.0, and the post-#2959 <=
    # keeps every other exact-zero mode rather than dropping all of them.
    errors_mixed = np.array([0.0, 5e-3], dtype=float)
    kept_mixed = _screen_modes(frequencies, amplitudes, errors_mixed, 64, SAMPLE_INTERVAL,
                               0.2, 0.55, 4, Q_thresh=0.0, rel_err_thresh=2.0,
                               err_thresh=0.01, rel_amp_thresh=-1.0, amp_thresh=0.0)
    assert kept_mixed == [0], (
        f"kept {kept_mixed}: the finite relative screen should keep exactly the "
        f"zero-error mode (2.0 * min_err = 0.0, err <= 0.0 true only for it)")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(Q_thresh=math.nan), "Q_thresh is NaN"),
        (dict(err_thresh=math.nan), "err_thresh is NaN"),
        (dict(rel_err_thresh=math.nan), "rel_err_thresh is NaN"),
        (dict(amp_thresh=math.nan), "amp_thresh is NaN"),
        (dict(rel_amp_thresh=math.nan), "rel_amp_thresh is NaN"),
        (dict(amp_thresh=math.inf), "must be finite"),
        (dict(rel_amp_thresh=-math.inf), "must be finite"),
    ],
)
def test_screening_thresholds_that_would_silently_empty_the_result_raise(kwargs, message):
    """A NaN threshold compares false against everything and returns "no modes"."""
    signal = synthesize(400, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])

    with pytest.raises(ValueError, match=message):
        harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10, **kwargs)


# --------------------------------------------------------------------------------------
# Internal numerics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_samples", [50, 51, 399, 400, 401])
@pytest.mark.parametrize("basis_size", [3, 5, 20])
@pytest.mark.parametrize("power", [0, 1, 2])
def test_closed_form_correlation_matrix_matches_the_literal_double_sum(
    num_samples, basis_size, power
):
    """U(p) is summed in closed form; check it against the definition it stands for.

    The definition is O(K^2) per element, so this runs on short records only — but
    at odd and even lengths, and at a trial spacing tight enough to exercise the
    cancellation in the 1/(y - x) denominator.
    """
    signal = synthesize(
        num_samples, SAMPLE_INTERVAL, [(1.0, 0.30, -0.001), (0.3, 0.31, 0.0)]
    )
    correlation_length = _correlation_length(num_samples)
    theta = 2 * np.pi * np.linspace(0.29, 0.32, basis_size) * SAMPLE_INTERVAL

    closed_form = _generate_U(signal, theta, correlation_length, power)

    inverse_z = np.exp(1j * theta)
    vandermonde = inverse_z[:, None] ** np.arange(correlation_length)[None, :]
    hankel = np.empty((correlation_length, correlation_length), dtype=np.complex128)
    for row in range(correlation_length):
        hankel[row] = signal[row + power:row + power + correlation_length]
    literal = vandermonde @ hankel @ vandermonde.T

    assert np.allclose(closed_form, literal, rtol=1e-11, atol=1e-11 * np.abs(literal).max())
    assert np.allclose(closed_form, closed_form.T, rtol=1e-12), "U(p) must be symmetric"


def test_coincident_trial_frequencies_use_the_exact_diagonal_form():
    """The 1/(y - x) closed form is singular on the diagonal; the branch must be exact."""
    num_samples = 201
    signal = synthesize(num_samples, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])
    correlation_length = _correlation_length(num_samples)
    single_theta = 2 * np.pi * 0.3 * SAMPLE_INTERVAL
    # Two identical trial frequencies: every entry takes the coincident branch.
    theta = np.array([single_theta, single_theta])

    matrix = _generate_U(signal, theta, correlation_length, 1)

    weight_index = np.arange(2 * correlation_length - 1)
    weights = np.minimum(weight_index + 1, 2 * correlation_length - 1 - weight_index)
    expected = np.sum(
        weights
        * np.exp(1j * single_theta * weight_index)
        * signal[1:1 + 2 * correlation_length - 1]
    )
    assert np.allclose(matrix, expected, rtol=1e-12)


def test_correlation_length_leaves_room_for_the_two_step_matrix():
    """K must satisfy 2K + 2 <= n or U(2) reads past the end of the record."""
    for num_samples in range(6, 40):
        correlation_length = _correlation_length(num_samples)
        assert 2 * correlation_length + 2 <= num_samples
        assert 2 * (correlation_length + 1) + 2 > num_samples  # and it is the largest such K


def test_basis_size_follows_meeps_formula():
    """MEEP src/bands.cpp:74-79, including the clamp order that lets mxbands win."""
    # int(|fmax-fmin| * dt * n * density), here 0.2 * 0.05 * 400 * 1.1 = 4.4 -> 4
    assert _basis_size(400, 0.05, 0.2, 0.4, mxbands=1, spectral_density=1.1) == 4
    assert _basis_size(400, 0.05, 0.2, 0.4, mxbands=10, spectral_density=1.1) == 10
    # The 150 ceiling applies before mxbands raises it again.
    assert _basis_size(100000, 0.05, 0.2, 0.4, mxbands=1, spectral_density=1.1) == 150
    assert _basis_size(100000, 0.05, 0.2, 0.4, mxbands=300, spectral_density=1.1) == 300
    # The floor of 2 for a band too narrow to derive anything from.
    assert _basis_size(400, 0.05, 0.2, 0.2001, mxbands=1, spectral_density=1.1) == 2


def test_rank_truncation_recovers_a_mode_far_weaker_than_the_dominant_one():
    """The SVD floor sets how weak a second mode can be; pin where it actually is."""
    signal = synthesize(
        4000, SAMPLE_INTERVAL, [(1.0, 0.300, -2e-4), (1e-4, 0.310, -2e-4)]
    )

    modes = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.35, 20)

    weak = nearest_mode(modes, 0.310)
    assert weak is not None and abs(weak.freq - 0.310) < 1e-8
    assert abs(abs(weak.amp) - 1e-4) / 1e-4 < 1e-3

    # A tighter floor is allowed to lose it, and must lose it quietly rather than
    # returning something near 0.310 that is not the mode.
    coarse = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.35, 20, rank_tol=1e-4)
    near_weak = [mode for mode in coarse if abs(mode.freq - 0.310) < 1e-3]
    assert not near_weak or abs(near_weak[0].freq - 0.310) < 1e-6
    assert nearest_mode(coarse, 0.300) is not None  # the strong mode is never lost


def test_rank_tol_default_is_the_documented_value():
    assert DEFAULT_RANK_TOL == 1e-9


def test_truncation_turned_off_still_reports_only_the_real_mode():
    """rank_tol=0 keeps U(0)'s null space; those vectors must be flagged, not fitted.

    Without the null-space guard every one of them divides by a round-off-level
    number and comes back as a mode with an infinite or NaN amplitude.
    """
    signal = synthesize(400, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)])

    frequencies, amplitudes, errors = _solve_modes(
        signal, 2 * np.pi * np.linspace(0.2, 0.4, 10) * SAMPLE_INTERVAL,
        _correlation_length(400), SAMPLE_INTERVAL, rank_tol=0.0,
    )
    flagged = ~np.isfinite(errors)
    assert flagged.sum() == 9, "the null space was not flagged"
    assert np.all(amplitudes[flagged] == 0.0)
    assert np.all(np.isfinite(amplitudes)), "an amplitude blew up instead of being flagged"

    modes = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10, rank_tol=0.0)
    assert len(modes) == 1
    assert abs(modes[0].freq - 0.3) < 1e-11
    assert abs(modes[0].amp - 1.0) < 1e-11


def test_a_repeated_pole_is_rejected_rather_than_fitted_with_a_giant_amplitude():
    """t*exp(-i*w*t) is not a sum of simple exponentials; the pair of poles coalesces.

    The two-pole fit reproduces the record well, so the error estimate stays small
    and the frequency and Q come out right — but the individual amplitudes are the
    difference of two cancelling numbers a million times the size of the signal.
    Unguarded this reports a confident resonance with a meaningless amplitude. MEEP's
    harminv returns nothing for this record and so must we.
    """
    num_samples = 400
    time = np.arange(num_samples) * SAMPLE_INTERVAL
    repeated_pole = time * np.exp(-1j * 2 * np.pi * (0.3 - 0.001j) * time)
    assert np.abs(repeated_pole).max() < 25.0  # nothing in the record is large

    assert harminv(repeated_pole, SAMPLE_INTERVAL, 0.25, 0.35, 10) == []
    # And not merely screened out by the default thresholds: it is flagged at source.
    unscreened = harminv(
        repeated_pole, SAMPLE_INTERVAL, 0.25, 0.35, 10, Q_thresh=0.0, err_thresh=math.inf
    )
    assert unscreened == []

    # A genuine pair of distinct poles at the same amplitudes is NOT rejected.
    genuine = synthesize(
        num_samples, SAMPLE_INTERVAL, [(1.0, 0.298, -0.001), (1.0, 0.302, -0.001)]
    )
    assert len(harminv(genuine, SAMPLE_INTERVAL, 0.25, 0.35, 10)) == 2


def test_distinct_positive_and_negative_frequency_modes_are_kept_apart():
    """A complex record can hold independent +f and -f' lines; neither may eat the other.

    The conjugate-pair collapse only applies inside a window of 2/n in harminv's
    dt = 1 units (bands.cpp:166). These two are outside it and are not conjugates,
    so both must survive — and the -0.34 line must be reported at +0.34, since MEEP
    reports every mode at |freq| (bands.cpp:189).
    """
    num_samples = 2000
    time = np.arange(num_samples) * SAMPLE_INTERVAL
    signal = np.exp(-1j * 2 * np.pi * (0.30 - 3e-4j) * time) + 0.6 * np.exp(
        -1j * 2 * np.pi * (-0.34 - 4e-4j) * time
    )

    modes = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.40, 10)

    assert len(modes) == 2, "an independent negative-frequency mode was collapsed away"
    assert all(mode.freq > 0 for mode in modes), "a mode was reported at negative frequency"
    assert abs(modes[0].freq - 0.30) < 1e-11 and abs(modes[0].amp - 1.0) < 1e-9
    assert abs(modes[1].freq - 0.34) < 1e-9 and abs(modes[1].amp - 0.6) < 1e-9


def test_error_estimate_is_the_documented_consistency_ratio():
    """Pin err's definition, normalization included, against an independent solve.

    At rank 1 the eigenvector is fixed by the truncation alone, so the whole
    quantity can be recomputed from the U(p) matrices without an eigensolver:
    u = B'U(1)B / B'U(0)B and err = |B'U(2)B / B'U(0)B - u^2| / |u|^2. The mode is
    coarsely sampled and strongly damped on purpose — |u| = exp(2*pi*decay*dt) is
    then about 0.46, far enough from 1 that dropping the |u|^2 normalization changes
    the answer by a factor of five.
    """
    num_samples = 300
    sample_interval = 0.5
    freq, decay = 0.3, -0.25
    generator = np.random.default_rng(3)
    signal = synthesize(num_samples, sample_interval, [(1.0, freq, decay)])
    signal = signal + 1e-5 * (
        generator.normal(size=num_samples) + 1j * generator.normal(size=num_samples)
    )
    theta = 2 * np.pi * np.linspace(0.1, 0.5, 4) * sample_interval
    correlation_length = _correlation_length(num_samples)

    frequencies, amplitudes, errors = _solve_modes(
        signal, theta, correlation_length, sample_interval, rank_tol=1e-2
    )
    assert frequencies.size == 1, "this check needs the rank-1 case"

    matrices = [_generate_U(signal, theta, correlation_length, p) for p in (0, 1, 2)]
    _, _, right_hermitian = np.linalg.svd(matrices[0])
    vector = right_hermitian[0].conj()
    quadratic = [vector @ matrix @ vector for matrix in matrices]
    eigenvalue = quadratic[1] / quadratic[0]
    expected_error = abs(quadratic[2] / quadratic[0] - eigenvalue ** 2) / abs(eigenvalue) ** 2
    expected_amplitude = (
        np.sum(np.exp(1j * np.outer(theta, np.arange(correlation_length)))
               @ signal[:correlation_length] * vector) ** 2 / quadratic[0]
    )

    assert abs(eigenvalue) < 0.6, "the damping premise of this check no longer holds"
    assert expected_error > 1e-9, "err is at round-off, so this compares nothing"
    assert abs(errors[0] - expected_error) / expected_error < 1e-6
    assert abs(amplitudes[0] - expected_amplitude) / abs(expected_amplitude) < 1e-6


def test_derived_basis_is_capped_by_the_record_and_explicit_nf_is_vetted():
    """MEEP's mxbands=100 default must not ask 20 samples for a 100-function basis."""
    # Derived: capped at the correlation length, silently, because it is a capacity ask.
    assert _resolve_basis_size(60, 29, SAMPLE_INTERVAL, 0.2, 0.4, 100, 1.1, None) == 29
    assert _resolve_basis_size(20, 9, SAMPLE_INTERVAL, 0.2, 0.4, 100, 1.1, None) == 9
    # Not capped when the record can carry it.
    assert _resolve_basis_size(4000, 1999, SAMPLE_INTERVAL, 0.2, 0.4, 100, 1.1, None) == 100
    # Explicit: an assertion about the data, so it raises rather than shrinking.
    assert _resolve_basis_size(4000, 1999, SAMPLE_INTERVAL, 0.2, 0.4, 100, 1.1, 30) == 30
    with pytest.raises(ValueError, match="exceeds what a 60-sample record supports"):
        _resolve_basis_size(60, 29, SAMPLE_INTERVAL, 0.2, 0.4, 100, 1.1, 30)
    with pytest.raises(ValueError, match="nf must be at least 2"):
        _resolve_basis_size(4000, 1999, SAMPLE_INTERVAL, 0.2, 0.4, 100, 1.1, 1)


# --------------------------------------------------------------------------------------
# MEEP API parity
# --------------------------------------------------------------------------------------


def test_mode_exposes_meeps_field_names_in_meeps_order():
    assert Mode._fields == ("freq", "decay", "Q", "amp", "err")
    mode = Mode(0.3, -0.001, 150.0, 1 + 2j, 1e-12)
    assert (mode.freq, mode.decay, mode.Q, mode.amp, mode.err) == mode
    # err is a float, so the comparisons users write do not raise.
    assert mode.err < 1e-6 and mode.err.real == mode.err


def test_do_harminv_returns_meeps_triples():
    signal = synthesize(400, SAMPLE_INTERVAL, [(1.3 + 0.2j, 0.3, -0.001)])

    bands = do_harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10)

    assert len(bands) == 1
    freq, amplitude, error = bands[0]
    assert isinstance(freq, complex) and isinstance(amplitude, complex)
    assert isinstance(error, float)
    assert abs(freq.real - 0.3) < 1e-12 and abs(freq.imag + 0.001) < 1e-12
    assert abs(amplitude - (1.3 + 0.2j)) < 1e-11


def test_quality_threshold_screens_low_q_modes():
    signal = synthesize(
        2000, SAMPLE_INTERVAL, [(1.0, 0.28, -0.03), (1.0, 0.32, -0.0005)]
    )  # Q = 4.7 and Q = 320

    assert [round(mode.freq, 6) for mode in harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10)] == [0.32]
    relaxed = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10, Q_thresh=1.0)
    assert [round(mode.freq, 6) for mode in relaxed] == [0.28, 0.32]


def test_band_screening_drops_modes_outside_the_requested_range():
    signal = synthesize(2000, SAMPLE_INTERVAL, [(1.0, 0.30, -3e-4), (1.0, 0.42, -3e-4)])

    inside = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.35, 10)
    assert [round(mode.freq, 6) for mode in inside] == [0.30]

    widened = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.45, 10)
    assert [round(mode.freq, 6) for mode in widened] == [0.30, 0.42]


def test_amplitude_thresholds_screen_weak_modes():
    signal = synthesize(2000, SAMPLE_INTERVAL, [(1.0, 0.30, -3e-4), (0.01, 0.34, -3e-4)])

    assert len(harminv(signal, SAMPLE_INTERVAL, 0.25, 0.4, 10)) == 2
    assert [round(mode.freq, 6) for mode in
            harminv(signal, SAMPLE_INTERVAL, 0.25, 0.4, 10, amp_thresh=0.1)] == [0.30]
    assert [round(mode.freq, 6) for mode in
            harminv(signal, SAMPLE_INTERVAL, 0.25, 0.4, 10, rel_amp_thresh=0.5)] == [0.30]


def test_error_thresholds_screen_poorly_determined_modes():
    generator = np.random.default_rng(5)
    signal = synthesize(1000, SAMPLE_INTERVAL, [(1.0, 0.30, -0.001)])
    signal = signal + 0.05 * (generator.normal(size=1000) + 1j * generator.normal(size=1000))

    loose = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 20, Q_thresh=0.0, err_thresh=1.0)
    assert len(loose) > 1
    best_error = min(mode.err for mode in loose)

    tight = harminv(
        signal, SAMPLE_INTERVAL, 0.2, 0.4, 20, Q_thresh=0.0, err_thresh=best_error * 1.5
    )
    assert len(tight) < len(loose)

    relative = harminv(
        signal, SAMPLE_INTERVAL, 0.2, 0.4, 20, Q_thresh=0.0, err_thresh=1.0,
        rel_err_thresh=2.0,
    )
    assert len(relative) < len(loose)
    assert all(mode.err <= 2.0 * best_error for mode in relative)


def test_mxbands_truncates_after_screening_keeping_the_lowest_error_modes():
    signal = synthesize(
        3000,
        SAMPLE_INTERVAL,
        [(1.0, 0.26, -3e-4), (1.0, 0.30, -3e-4), (1.0, 0.34, -3e-4)],
    )

    assert len(harminv(signal, SAMPLE_INTERVAL, 0.22, 0.38, 10)) == 3
    assert len(harminv(signal, SAMPLE_INTERVAL, 0.22, 0.38, 2)) == 2
    assert len(harminv(signal, SAMPLE_INTERVAL, 0.22, 0.38, 1)) == 1


def test_harminv_class_collects_and_analyzes_as_a_step_function():
    """MEEP's protocol: __call__(sim, 'step') collects, __call__(sim, 'finish') analyses."""

    class StubSimulation:
        def __init__(self):
            self.step_index = 0
            self.requested = []

        def meep_time(self):
            return self.step_index * SAMPLE_INTERVAL

        def get_field_point(self, component, point):
            self.requested.append((component, point))
            value = np.exp(
                -1j * 2 * np.pi * (0.3 - 0.001j) * self.step_index * SAMPLE_INTERVAL
            )
            self.step_index += 1
            return value

    instance = Harminv(c="Ez", pt=(0.1, 0.2, 0.3), fcen=0.3, df=0.2)
    simulation = StubSimulation()
    for _ in range(600):
        instance(simulation, "step")
    assert instance.modes == []  # nothing reported until the run finishes
    instance(simulation, "finish")

    assert len(instance.data) == 600
    assert abs(instance.data_dt - SAMPLE_INTERVAL) < 1e-12
    assert simulation.requested[0] == ("Ez", (0.1, 0.2, 0.3))
    assert len(instance.modes) == 1
    assert abs(instance.modes[0].freq - 0.3) / 0.3 < 1e-9
    assert abs(instance.modes[0].Q - 150.0) / 150.0 < 1e-6


def test_harminv_class_carries_meeps_default_thresholds():
    instance = Harminv(fcen=0.3, df=0.2)
    assert instance.spectral_density == 1.1
    assert instance.Q_thresh == 50.0
    assert instance.rel_err_thresh == math.inf
    assert instance.err_thresh == 0.01
    assert instance.rel_amp_thresh == -1.0
    assert instance.amp_thresh == -1.0
    assert instance.mxbands is None
    assert instance.modes == [] and instance.data == []


def test_harminv_class_thresholds_reach_the_solver():
    instance = Harminv(fcen=0.3, df=0.2, mxbands=10)
    instance.data = list(
        synthesize(2000, SAMPLE_INTERVAL, [(1.0, 0.28, -0.03), (1.0, 0.32, -0.0005)])
    )
    instance.data_dt = SAMPLE_INTERVAL

    assert [round(mode.freq, 6) for mode in instance.analyze()] == [0.32]
    instance.Q_thresh = 1.0
    assert [round(mode.freq, 6) for mode in instance.analyze()] == [0.28, 0.32]


def test_harminv_class_refuses_incomplete_setup():
    signal = list(synthesize(400, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)]))

    without_band = Harminv()
    without_band.data, without_band.data_dt = signal, SAMPLE_INTERVAL
    with pytest.raises(ValueError, match="fcen and df"):
        without_band.analyze()

    without_dt = Harminv(fcen=0.3, df=0.2)
    without_dt.data = signal
    with pytest.raises(ValueError, match="no sample interval"):
        without_dt.analyze()

    without_hooks = Harminv(fcen=0.3, df=0.2)
    with pytest.raises(ValueError, match="meep_time"):
        without_hooks(object(), "step")


def test_harminv_class_treats_mxbands_zero_as_meeps_default():
    """simulation.py:1207 — 0 or None both mean 100."""
    instance = Harminv(fcen=0.3, df=0.2, mxbands=0)
    instance.data = list(synthesize(400, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)]))
    instance.data_dt = SAMPLE_INTERVAL

    assert len(instance.analyze()) == 1


# --------------------------------------------------------------------------------------
# Cross-validation against CPU MEEP's harminv
# --------------------------------------------------------------------------------------

_MEEP_ORACLE_SCRIPT = '''"""CPU-MEEP harminv reference for meep_gpu.harminv parity tests."""
import sys

import numpy as np
import meep as mp

bundle = np.load(sys.argv[1], allow_pickle=False)
output_path = sys.argv[2]

results = {}
for name in bundle.files:
    if name.endswith("__params"):
        continue
    signal = bundle[name]
    dt, fmin, fmax, mxbands, Q_thresh, err_thresh = bundle[name + "__params"]
    bands = mp.py_do_harminv(
        list(signal.astype(complex)), float(dt), float(fmin), float(fmax),
        int(mxbands), 1.1, float(Q_thresh), mp.inf, float(err_thresh), -1.0, -1.0,
    )
    rows = [[f.real, f.imag, a.real, a.imag, e.real] for f, a, e in bands]
    results[name] = np.array(rows, dtype=float).reshape(len(rows), 5)

np.savez(output_path, **results)
'''


def _meep_harminv_oracle(tmp_path, cases):
    """Run CPU MEEP's harminv on each named signal in its own process."""
    bundle_path = tmp_path / "harminv_signals.npz"
    payload = {}
    for name, (signal, dt, fmin, fmax, mxbands, q_thresh, err_thresh) in cases.items():
        payload[name] = np.asarray(signal, dtype=np.complex128)
        payload[name + "__params"] = np.array(
            [dt, fmin, fmax, mxbands, q_thresh, err_thresh], dtype=float
        )
    np.savez(bundle_path, **payload)

    script_path = tmp_path / "harminv_oracle.py"
    script_path.write_text(_MEEP_ORACLE_SCRIPT, encoding="utf-8")
    output_path = tmp_path / "harminv_reference.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), str(bundle_path), str(output_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP harminv oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as reference:
        return {name: reference[name] for name in reference.files}


def _parity_cases():
    """Signals covering the shapes that matter, at odd and even record lengths."""
    generator = np.random.default_rng(31)
    noisy = synthesize(1000, SAMPLE_INTERVAL, [(1.0, 0.3, -0.001)]) + 1e-3 * (
        generator.normal(size=1000) + 1j * generator.normal(size=1000)
    )
    return {
        "single_even": (
            synthesize(400, SAMPLE_INTERVAL, [(1.3 * np.exp(1j * 0.4), 0.30, -0.001)]),
            SAMPLE_INTERVAL, 0.2, 0.4, 10, 50.0, 0.01,
        ),
        "single_odd": (
            synthesize(401, SAMPLE_INTERVAL, [(1.3 * np.exp(1j * 0.4), 0.30, -0.001)]),
            SAMPLE_INTERVAL, 0.2, 0.4, 10, 50.0, 0.01,
        ),
        "off_centre_low_edge": (
            synthesize(555, SAMPLE_INTERVAL, [(0.8, 0.2103, -0.0007)]),
            SAMPLE_INTERVAL, 0.2, 0.4, 10, 50.0, 0.01,
        ),
        "two_separated": (
            synthesize(
                1500, SAMPLE_INTERVAL,
                [(1.0, 0.26, -3e-4), (0.4 * np.exp(-1j * 2.0), 0.34, -8e-4)],
            ),
            SAMPLE_INTERVAL, 0.2, 0.4, 10, 50.0, 0.01,
        ),
        "two_close": (
            synthesize(
                2000, SAMPLE_INTERVAL,
                [(1.0, 0.300, -2e-4), (0.7 * np.exp(1j * 1.1), 0.302, -3e-4)],
            ),
            SAMPLE_INTERVAL, 0.25, 0.35, 10, 50.0, 0.01,
        ),
        "real_valued": (
            np.real(
                synthesize(
                    2000, SAMPLE_INTERVAL,
                    [(1.0, 0.300, -2e-4), (0.7 * np.exp(1j * 1.1), 0.302, -3e-4)],
                )
            ),
            SAMPLE_INTERVAL, 0.25, 0.35, 10, 50.0, 0.01,
        ),
        "damped": (
            synthesize(700, SAMPLE_INTERVAL, [(2.0, 0.3, -0.0075)]),
            SAMPLE_INTERVAL, 0.05, 0.55, 10, 0.0, 0.01,
        ),
        "noisy": (noisy, SAMPLE_INTERVAL, 0.2, 0.4, 10, 50.0, 0.01),
    }


def _match_against_meep(ours, reference_rows):
    """Pair each MEEP mode with our nearest-in-frequency mode; return pairs and extras."""
    pairs = []
    matched_indices = set()
    for row in reference_rows:
        meep_freq = float(row[0])
        index = min(
            range(len(ours)), key=lambda i: abs(ours[i].freq - meep_freq), default=None
        )
        assert index is not None, f"MEEP found a mode at {meep_freq:.6f} and we found none"
        matched_indices.add(index)
        pairs.append((ours[index], row))
    extras = [mode for i, mode in enumerate(ours) if i not in matched_indices]
    return pairs, extras


@requires_meep
@skip_without_meep
def test_matches_cpu_meep_harminv_on_identical_signals(tmp_path):
    """Same signals through MEEP's harminv C library and through this module.

    The contract has two halves. First, every mode MEEP reports we also report, to
    the per-quantity bounds asserted at the end — that is the parity a user
    switching backends depends on. Second, we may report modes MEEP's internal
    culling discarded (it prunes its basis in a way this module does not, module
    header deviation 1), but only if they are unmistakably flagged: a hundred times
    the error of the matched modes and under a hundredth of their amplitude. A
    plausible extra resonance would be a failure; a flagged one is a knob away
    (``rel_err_thresh``, ``rel_amp_thresh``) from being screened out.
    """
    cases = _parity_cases()
    reference = _meep_harminv_oracle(tmp_path, cases)

    worst = dict(freq=0.0, decay=0.0, quality=0.0, amplitude=0.0)
    worst_noisy = dict(freq=0.0, quality=0.0)
    extra_count = 0
    for name, (signal, dt, fmin, fmax, mxbands, q_thresh, err_thresh) in cases.items():
        ours = harminv(
            signal, dt, fmin, fmax, mxbands, Q_thresh=q_thresh, err_thresh=err_thresh
        )
        theirs = reference[name]
        assert len(ours) >= len(theirs), (
            f"{name}: found {len(ours)} modes, MEEP found {len(theirs)} "
            f"(ours {[round(m.freq, 6) for m in ours]}, "
            f"MEEP {[round(float(row[0]), 6) for row in theirs]})"
        )
        pairs, extras = _match_against_meep(ours, theirs)
        for mode, row in pairs:
            meep_freq, meep_decay, amp_real, amp_imag, _ = row
            meep_amp = complex(amp_real, amp_imag)
            meep_quality = meep_freq / (-2 * meep_decay)
            frequency_error = abs(mode.freq - meep_freq) / abs(meep_freq)
            quality_error = abs(mode.Q - meep_quality) / abs(meep_quality)
            if name == "noisy":
                # Both codes fit noise here, so they disagree by the noise and not
                # by the algorithm; the true frequency is 0.3 and both land ~1e-6
                # off it in different directions. Bounded separately, loosely, and
                # against the TRUTH rather than against each other.
                worst_noisy["freq"] = max(worst_noisy["freq"], frequency_error)
                worst_noisy["quality"] = max(worst_noisy["quality"], quality_error)
                assert abs(mode.freq - 0.3) / 0.3 < 1e-5
                continue
            worst["freq"] = max(worst["freq"], frequency_error)
            worst["decay"] = max(worst["decay"], abs(mode.decay - meep_decay) / abs(meep_decay))
            worst["quality"] = max(worst["quality"], quality_error)
            if name == "real_valued":
                # MEEP may report the -f member of the collapsed conjugate pair,
                # whose amplitude is conj(a) against the same |freq|; see
                # test_real_valued_signal_collapses_the_conjugate_pair.
                worst["amplitude"] = max(
                    worst["amplitude"],
                    min(
                        abs(mode.amp - meep_amp) / abs(meep_amp),
                        abs(mode.amp - np.conj(meep_amp)) / abs(meep_amp),
                    ),
                )
            else:
                worst["amplitude"] = max(
                    worst["amplitude"], abs(mode.amp - meep_amp) / abs(meep_amp)
                )

        extra_count += len(extras)
        if extras:
            matched_error = max(mode.err for mode, _ in pairs)
            matched_amplitude = max(abs(mode.amp) for mode, _ in pairs)
            for mode in extras:
                assert mode.err > 100 * matched_error, (
                    f"{name}: extra mode at f={mode.freq:.6f} carries err={mode.err:.2e}, "
                    f"comparable to the matched modes' {matched_error:.2e}"
                )
                assert abs(mode.amp) < 0.01 * matched_amplitude, (
                    f"{name}: extra mode at f={mode.freq:.6f} has amplitude "
                    f"{abs(mode.amp):.2e} against the matched {matched_amplitude:.2e}"
                )

    assert worst["freq"] < 1e-9, worst
    assert worst["decay"] < 1e-6, worst
    assert worst["quality"] < 1e-6, worst
    assert worst["amplitude"] < 1e-6, worst
    assert worst_noisy["freq"] < 1e-5, worst_noisy
    assert worst_noisy["quality"] < 1e-3, worst_noisy
    assert extra_count <= 2, f"{extra_count} modes beyond MEEP's across the case set"


@requires_meep
@skip_without_meep
def test_error_estimate_tracks_meeps(tmp_path):
    """Our err is a different formula from the C library's; pin how close it runs.

    MEEP's harminv reports a "crude estimate of the relative error in the complex
    frequency" by a route this module does not reproduce (module header, deviation
    2). Three things are asserted, over every parity case:

    1. Neither code ever reports a genuine mode with an error near a threshold —
       every matched mode is orders below MEEP's default err_thresh of 0.01.
    2. We are never much MORE pessimistic than MEEP, which is the direction that
       would lose a mode to screening.
    3. Where the error is a real number rather than round-off, the two agree
       within a factor of two.

    The measured spread the other way is recorded rather than asserted: on the
    ill-conditioned close-pair cases ours runs ~1000x smaller (1e-14 against
    1e-11), which is the SVD truncation, not a disagreement about the mode.
    """
    cases = _parity_cases()
    reference = _meep_harminv_oracle(tmp_path, cases)

    compared_meaningful = 0
    for name, (signal, dt, fmin, fmax, mxbands, q_thresh, err_thresh) in cases.items():
        ours = harminv(
            signal, dt, fmin, fmax, mxbands, Q_thresh=q_thresh, err_thresh=err_thresh
        )
        pairs, _ = _match_against_meep(ours, reference[name])
        for mode, row in pairs:
            meep_error = float(row[4])
            assert mode.err < 1e-6 and meep_error < 1e-6, (
                f"{name} f={mode.freq:.6f}: a genuine mode is not confidently reported"
            )
            assert mode.err < max(50 * meep_error, 1e-13), (
                f"{name} f={mode.freq:.6f}: err {mode.err:.2e} is far above "
                f"MEEP's {meep_error:.2e}; screening would diverge"
            )
            if meep_error > 1e-9:  # only the noisy record gets here
                compared_meaningful += 1
                assert 0.02 < mode.err / meep_error < 50.0, (
                    f"{name} f={mode.freq:.6f}: err {mode.err:.2e} vs MEEP {meep_error:.2e}"
                )
    assert compared_meaningful >= 1, "no case exercised a non-round-off error estimate"


@requires_meep
@skip_without_meep
def test_pure_noise_screening_matches_meep_seed_by_seed(tmp_path):
    """The degenerate case, judged against MEEP rather than against a hope.

    Both codes fit modes to white noise on some seeds and neither does on others.
    The contract asserted here is that we are not MORE credulous than MEEP: on
    every seed our mode count is within one of theirs, and where MEEP reports
    nothing we report at most one flagged mode.
    """
    cases = {}
    for seed in range(12):
        generator = np.random.default_rng(seed)
        cases[f"noise_{seed}"] = (
            generator.normal(size=800) + 1j * generator.normal(size=800),
            SAMPLE_INTERVAL, 0.2, 0.4, 10, 50.0, 0.01,
        )
    reference = _meep_harminv_oracle(tmp_path, cases)

    for name, (signal, dt, fmin, fmax, mxbands, q_thresh, err_thresh) in cases.items():
        ours = harminv(
            signal, dt, fmin, fmax, mxbands, Q_thresh=q_thresh, err_thresh=err_thresh
        )
        theirs = reference[name]
        assert len(ours) <= len(theirs) + 1, (
            f"{name}: {len(ours)} spurious modes against MEEP's {len(theirs)}"
        )
        for mode in ours:
            assert mode.err > 1e-6, f"{name}: noise mode reported as confident"
        if len(theirs):
            meep_errors = [float(row[4]) for row in theirs]
            assert min(mode.err for mode in ours) > 1e-6
            assert min(meep_errors) > 1e-6  # MEEP flags them too, at the same scale


# --------------------------------------------------------------------------------------
# Practical guidance, executable so it cannot go stale
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("quality", [30, 300, 3000])
def test_record_length_needed_for_a_given_q(quality):
    """How long must a run be to measure Q? Shorter than the lifetime, it turns out.

    With a 1e-3 noise floor on a unit-amplitude mode:

    * ~100 oscillation periods of record gives Q to better than 0.1% at Q = 30,
      300 and 3000 alike — even at Q = 3000, where 100 periods is only a tenth of
      the mode's 1/e amplitude lifetime. Filter diagonalization fits the decay, it
      does not need to watch it finish.
    * ~20 periods still gives a few 0.1%; ~5 periods gives a few % at high Q and
      is where the estimate starts to depend on the noise realization.

    The band matters less than the length: the band only has to CONTAIN the mode
    and exclude modes you do not want fitted. Narrow it to separate a crowded
    spectrum, not to improve a single mode.
    """
    freq = 0.3
    decay = -freq / (2 * quality)
    generator = np.random.default_rng(11)
    num_samples = int(100 / freq / SAMPLE_INTERVAL)  # 100 periods
    signal = synthesize(num_samples, SAMPLE_INTERVAL, [(1.0, freq, decay)])
    signal = signal + 1e-3 * (
        generator.normal(size=num_samples) + 1j * generator.normal(size=num_samples)
    )

    modes = harminv(signal, SAMPLE_INTERVAL, 0.2, 0.4, 10, Q_thresh=0.0)

    mode = nearest_mode(modes, freq)
    assert mode is not None
    assert abs(mode.freq - freq) / freq < 1e-5
    assert abs(mode.Q - quality) / quality < 1e-3


def test_band_width_does_not_have_to_be_narrow():
    """Documented guidance: a wide band costs nothing for a well-separated mode."""
    signal = synthesize(
        2000, SAMPLE_INTERVAL, [(1.0, 0.30, -2e-4), (0.7, 0.3005, -3e-4)]
    )

    for half_width in (0.005, 0.05, 0.2):
        modes = harminv(signal, SAMPLE_INTERVAL, 0.30 - half_width, 0.30 + half_width, 20)
        assert len(modes) == 2, f"half-width {half_width} lost a mode"
        assert abs(modes[0].freq - 0.30) < 1e-9
        assert abs(modes[1].freq - 0.3005) < 1e-9


def test_rel_amp_thresh_measures_against_screened_modes_not_the_raw_eigen_set():
    """``rel_amp_thresh`` must not be defeated by a term the other screens already threw out.

    bands.cpp:120-126 takes the reference amplitude over every mode the solver produced,
    before any screening. That is harmless with the C library, whose raw set is already
    culled, and a trap here: the transient a source leaves when it switches off is fitted
    by a handful of very heavily damped terms whose amplitudes run far above the physical
    mode's. Measured on a conductivity-loaded cavity record — genuine mode at amplitude
    0.188, five artefacts at up to 33, all of them thrown out by ``Q_thresh`` — the raw
    reference makes the perfectly ordinary request ``rel_amp_thresh = 0.01`` return an
    EMPTY list for a record with a clean resonance in it.

    The signal here reproduces that shape directly: a long-lived line at 0.30 under a
    fifty-times-stronger term at Q ~ 1 that no threshold but Q would catch.
    """
    signal = synthesize(2000, SAMPLE_INTERVAL, [(1.0, 0.30, -2e-4), (50.0, 0.31, -0.15)])

    # Q_thresh alone throws the loud term out, so the genuine mode is what is left.
    modes = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.35, 10)
    assert [round(mode.freq, 5) for mode in modes] == [0.30]
    assert abs(modes[0].amp) == pytest.approx(1.0, rel=1e-3)

    # And asking for "only strong modes" must not then delete it. Every one of these is
    # far below the ratio 1.0 / 50 the raw reference would impose.
    for threshold in (0.01, 0.1, 0.5, 0.9):
        screened = harminv(signal, SAMPLE_INTERVAL, 0.25, 0.35, 10, rel_amp_thresh=threshold)
        assert [round(mode.freq, 5) for mode in screened] == [0.30], (
            f"rel_amp_thresh={threshold} deleted the only surviving mode; the reference is "
            f"being taken over modes the other screens rejected"
        )

    # The threshold still has to WORK, or the fix above is just a way of ignoring it: a
    # weak second line inside the band goes when asked to.
    two = synthesize(2000, SAMPLE_INTERVAL, [(1.0, 0.30, -2e-4), (0.01, 0.33, -2e-4)])
    assert len(harminv(two, SAMPLE_INTERVAL, 0.25, 0.35, 10)) == 2
    assert [round(mode.freq, 5) for mode in
            harminv(two, SAMPLE_INTERVAL, 0.25, 0.35, 10, rel_amp_thresh=0.5)] == [0.30]
