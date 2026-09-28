"""
Harmonic inversion (filter diagonalization) for the MEEP-compatible FDTD engine —
the pure post-processing step that turns a field-versus-time series into resonant
modes: complex frequency, amplitude, Q and a per-mode error estimate.

This is how a cavity, ring resonator or photonic-crystal defect mode gets its Q
and its resonant frequency out of an FDTD run. Given the time series f(t) of one
field component at one point, it fits

    f(t) = sum_n a_n exp(-i * 2*pi * (freq_n + i*decay_n) * t)

over a chosen frequency band, and reports each term. It touches no field array
and no stepper state, so it is identical on CPU and GPU by construction.

Public surface, named exactly as MEEP names it (``python/simulation.py`` class
``Harminv``, ``Mode``), because a MEEP script must read the same here:

* ``Mode`` — ``freq`` (Re omega / 2*pi), ``decay`` (Im omega / 2*pi), ``Q``,
  ``amp`` (complex), ``err`` (mode-error figure of merit);
* ``harminv(signal, dt, fmin, fmax, ...)`` — the direct call, returning
  ``list[Mode]`` sorted by increasing ``|freq|``;
* ``do_harminv(...)`` — the ``(complex freq, complex amp, float err)`` triples,
  mirroring MEEP's ``mp.py_do_harminv`` return shape for scripts that use it;
* ``Harminv`` — the step-function object: ``h = Harminv(c, pt, fcen, df)``,
  handed to a run loop, collecting the point field each step and analysing at the
  end into ``h.modes``. Sources must be off while it collects, exactly as in MEEP.

WHY NOT AN FFT. A discrete Fourier transform resolves two lines only if they are
farther apart than 1/T for a record of length T. Filter diagonalization fits the
*model* instead of the spectrum, so two modes 1/50th of a Fourier bin apart come
out to twelve digits from the same record (``test_harminv.py`` pins that case).
That is the whole reason this exists rather than a peak-find.

METHOD. Mandelshtam and Taylor, "Harmonic inversion of time signals", J. Chem.
Phys. 107, 6756 (1997); erratum ibid. 109, 4128 (1998) — the method MEEP's
external harminv library implements. With c_m the samples and z_j = exp(-2*pi*i*
f_j*dt) a set of ``nf`` trial values spanning the band, build the two symmetric
matrices

    U(p)_{j j'} = sum_{n,n'=0}^{K-1} z_j^-n z_j'^-n' c_{n+n'+p}       p = 0, 1

and solve the generalized eigenproblem U(1) B = u U(0) B. Each eigenvalue is a
mode: u_k = exp(-i*omega_k*dt), so omega_k = i*ln(u_k)/dt, and the eigenvector
gives the amplitude. K is the correlation length, half the record.

The double sum is never formed. Summing the geometric series in n and n' gives,
with x = z_j^-1 and y = z_j'^-1 and C_q(w) = sum_{m=0}^{K-1} w^m c_{m+q},

    U(p)_{j j'} = [ y*C_p(y) - x*C_p(x) - y*x^K*C_{p+K}(y) + x*y^K*C_{p+K}(x) ]
                  / (y - x)

which costs O(nf*K) for the moments plus O(nf^2) for the matrix, and is exact:
``test_harminv.py`` checks it against the literal double sum to 1e-13 relative.
The x = y case is the removable singularity and is evaluated from its own exact
form, sum_m min(m+1, 2K-1-m) * x^m * c_{m+p}, not from a limit.

Cost is O(nf*K) plus O(nf^3): a 400 000-sample record with the default 100-function
basis takes about six seconds and 350 MB, which is nothing beside the FDTD run that
produced it. The trial-frequency Vandermonde is built in blocks so its footprint
does not grow with the record.

DELIBERATE DIFFERENCES FROM THE harminv C LIBRARY. Two, both documented because
this is a reimplementation from the published method and not a transcription of
that library's source:

1. Rank reduction is by truncated SVD of U(0) — singular values below
   ``rank_tol`` times the largest are dropped and the eigenproblem is solved in
   the surviving subspace. The generalized problem is singular whenever ``nf``
   exceeds the number of modes actually present (the normal case), so some
   regularization is mandatory; this one is stable and has one honest knob.
   Measured effect: on a two-mode signal with a 1e-4 amplitude ratio this
   recovers the weak mode, where the C library reports only the strong one.

2. ``err`` is the standard filter-diagonalization consistency check — how far the
   fitted mode fails to also satisfy the two-step problem,

       err_k = |B_k^T U(2) B_k / B_k^T U(0) B_k - u_k^2| / |u_k|^2

   which is zero for a mode that truly is a decaying exponential in the record
   and large for a fitting artefact. It needs U(2), hence one more sample of the
   record; K = (n-2)//2 rather than (n-1)//2. Calibrated against the C library
   through MEEP on identical signals (``test_harminv.py::test_error_estimate_
   tracks_meeps``): where the error is a meaningful number rather than round-off
   the two agree within a factor of two (0.53x on a noisy record), and across
   well-conditioned clean signals the ratio spans 0.16x to 25x. On badly
   conditioned ones — two lines a fifth of a Fourier bin apart, where U(0) is
   rank 2 out of a 10-function basis — ours runs about a thousand times SMALLER
   (1e-14 against 1e-11); both are far below any threshold, and the difference is
   the truncated-SVD conditioning of deviation 1, not a disagreement about the
   mode. What matters, and what is tested, is the ordering: the genuine mode of a
   noisy record scores 1e-7 while its neighbouring artefacts score 1e-4 and white
   noise scores 1e-3. Treat it as MEEP documents it: a figure of merit, smaller
   is better, not an error bar. It says nothing about resolution, cell-size or
   discretization error.

3. ``rel_amp_thresh`` measures against the strongest mode that passed the OTHER
   screens, where bands.cpp:126 measures against the whole raw eigen-set. On a
   real FDTD record the two are different sets: fitting the transient left by a
   source switching off costs a few very heavily damped terms, measured at up to
   175x the record's own peak amplitude on a cavity record whose genuine mode
   sits at 0.188. ``Q_thresh`` throws every one of them out, but taking the
   reference before that screen sets it by an artefact and makes
   ``rel_amp_thresh = 0.01`` return NOTHING for a record with a perfectly good
   resonance in it. The C library culls those terms before MEEP's loop ever sees
   them — handed the identical record it returns exactly one mode — so its
   reference is over physical modes, and this reproduces that behaviour rather
   than the letter of the loop. See ``_screen_modes``.

Everything else is MEEP's, verbatim from ``src/bands.cpp do_harminv()``: the
default basis size ``int(|fmax-fmin| * dt * n * spectral_density)`` clamped to
[2, 150] then raised to ``mxbands``; the six-way mode screening on band, Q,
absolute and relative error and absolute and relative amplitude; the
positive/negative frequency pairing that collapses the conjugate pair a real
signal produces; the truncation to ``mxbands``; and the final sort by |freq|.
An all-zero record returns no modes rather than dividing by zero, as it does
there.

WHAT A REAL FDTD RECORD RETURNS. With MEEP's default thresholds this returns
MORE modes than the C library does — six against one on the cavity record above.
The extra five are the SVD-truncation's near-null-space vectors: they carry
amplitudes seven orders below the genuine mode and errors seven orders above it,
so they are trivially separable, but they are there, and a script that reads
``modes[0]`` without looking at ``err`` or ``amp`` will not get what it expects.
Sort by ``abs(mode.amp)``, or set ``rel_amp_thresh``, or screen on ``err``. The
same rank reduction is what recovers a second mode at 1e-4 of the dominant
amplitude where the C library reports only the strong one; this is its cost.

WHAT REFUSES RATHER THAN GUESSES. A record shorter than the requested basis, an
inverted or empty band, a non-finite sample (a diverged run), a non-1-D input or
a non-positive dt all raise ValueError. Two degenerate eigenvector cases are
reported with ``err = inf`` and zero amplitude so the screening drops them,
never with a fabricated amplitude: a vector lying in U(0)'s numerical null space
(reachable by turning the truncation down), and a pair of COALESCED vectors — a
repeated pole, the signal ``t*exp(-i*w*t)`` rather than a sum of simple
exponentials — whose individual amplitudes are the difference of two enormous
cancelling numbers. The second matters: left alone that case reports the right
frequency and the right Q with an amplitude a million times the signal and a
small error estimate, which is the exact shape of a plausible wrong answer. CPU
MEEP returns nothing for that record and neither does this.

HOST-SIDE BY CHOICE. The record is one scalar per step, so this module is plain
NumPy: it copies a device array to the host once, through ``backends.to_numpy``,
and does the linear algebra there. Nothing is gained by keeping an nf x nf
eigenproblem on the GPU, and LAPACK on the host is the reference path that makes
the result identical whichever backend produced the field.

UNITS. MEEP's throughout. ``dt`` is the interval between samples in MEEP time
units, and ``fmin``/``fmax``/``freq``/``decay`` are frequencies (1/time), not
angular frequencies — omega = 2*pi*freq. ``Q = |freq| / (-2*decay)``, so a
decaying mode has ``decay < 0`` and ``Q > 0``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import math
from collections import namedtuple
from collections.abc import Sequence
from typing import Any

import numpy as np

from .backends import to_numpy

# MEEP python/simulation.py:851 — the reported per-mode record, same field names
# and same order, so `for m in h.modes: m.freq, m.Q` is source-compatible.
# One deliberate type change: `err` is a float here, where MEEP's SWIG layer
# hands back a complex with zero imaginary part. A float supports everything the
# complex does (`.real`, `abs()`) plus the ordering comparisons users actually
# write (`m.err < 1e-6`), which raise TypeError on a complex.
Mode = namedtuple("Mode", ["freq", "decay", "Q", "amp", "err"])

# MEEP src/bands.cpp:74-76 — the basis-size clamp. The ceiling keeps the nf x nf
# eigenproblem small; the floor of 2 is what the trial-frequency spacing needs.
MAX_BASIS_SIZE = 150
MIN_BASIS_SIZE = 2

# Singular values of U(0) below this fraction of the largest are treated as null
# space. U(0)'s singular values scale as the SQUARE of a mode's amplitude, so
# 1e-9 keeps modes down to ~3e-5 of the dominant amplitude.
DEFAULT_RANK_TOL = 1e-9

# |x - y| below this (x, y on the unit circle) takes the removable-singularity
# branch of the U(p) closed form. Only exact coincidence needs it; the cancellation
# at finite separation is benign because the numerator vanishes at the same rate.
_COINCIDENT_TRIAL_TOL = 1e-13

# A mode whose eigenvector has |B^T U(0) B| below this times ||U(0)|| lies in U(0)'s
# numerical null space: its amplitude is 0/0. Only reachable with the truncation
# turned down (rank_tol near 0), which is a supported thing to do.
_NORMALIZATION_FLOOR = 1e-14

# Two eigenvectors that have COALESCED (a repeated pole, e.g. a record containing
# t*exp(-i*w*t) rather than a sum of simple exponentials) are the defective case of a
# complex-symmetric pencil, and the signature is the BILINEAR self-overlap B^T U(0) B
# collapsing relative to the sesquilinear one B^H U(0) B. The pair still fits the
# record, so the error estimate stays small, but the individual amplitudes are the
# difference of two enormous cancelling numbers and mean nothing — left alone this
# reports a resonance at the right frequency with an amplitude a million times the
# signal. Measured separation: a true repeated pole scores 4e-7, while the hardest
# legitimate case tested (two modes 1/1000 of a Fourier bin apart) scores 3e-2 and a
# weak mode at 1e-4 of the dominant amplitude scores 12. The floor sits four orders
# below anything real. MEEP's harminv rejects the repeated-pole record outright.
_DEGENERACY_FLOOR = 1e-6


def _as_host_signal(signal: Any) -> np.ndarray:  # One host copy; device arrays welcome, see header.
    samples = np.asarray(to_numpy(signal))
    if samples.ndim != 1:
        raise ValueError(
            f"harminv needs a 1-D time series, got shape {samples.shape}; "
            "pass one scalar per time step"
        )
    samples = samples.astype(np.complex128, copy=False)
    if not np.all(np.isfinite(samples)):
        bad = int(np.count_nonzero(~np.isfinite(samples)))
        raise ValueError(
            f"harminv signal has {bad} non-finite sample(s); a diverged or "
            "truncated run cannot be harmonically inverted"
        )
    return samples


def _correlation_length(num_samples: int) -> int:
    """K in the U(p) definition — the record's half length.

    U(2) reads c[2K+1], so the record must hold 2K+2 samples: K = (n-2)//2. MEEP's
    library, which forms only U(0) and U(1), can afford (n-1)//2; the difference is
    at most one sample of resolution and buys the error estimate.
    """
    return (num_samples - 2) // 2


def _basis_size(
    num_samples: int, dt: float, fmin: float, fmax: float, mxbands: int, spectral_density: float
) -> int:
    """MEEP src/bands.cpp:74-79 — trial-basis size from the band and record length."""
    count = int(abs(fmax - fmin) * dt * num_samples * spectral_density)
    count = min(count, MAX_BASIS_SIZE)
    count = max(count, MIN_BASIS_SIZE)
    if mxbands > count:  # bands.cpp:79 — an explicit mxbands raises the basis to match.
        count = mxbands
    return count


def _resolve_basis_size(
    num_samples: int, correlation_length: int, dt: float, fmin: float, fmax: float,
    mxbands: int, spectral_density: float, nf: int | None,
) -> int:
    """Settle the final trial-basis size, capping a derived one and vetting an explicit one.

    A DERIVED size is a request for capacity, not an assertion, so it is capped at
    what the record can support — MEEP's ``mxbands`` default of 100 asks for a
    100-function basis regardless of how short the run was. An EXPLICIT ``nf`` is an
    assertion about the data and raises instead of being quietly shrunk.
    """
    if nf is None:
        derived = _basis_size(num_samples, dt, fmin, fmax, mxbands, spectral_density)
        return min(derived, correlation_length)
    if nf < MIN_BASIS_SIZE:
        raise ValueError(f"harminv nf must be at least {MIN_BASIS_SIZE}, got {nf}")
    if nf > correlation_length:
        raise ValueError(
            f"harminv nf={nf} exceeds what a {num_samples}-sample record supports "
            f"(K={correlation_length}); the basis would be larger than the data. "
            f"Record at least {2 * nf + 2} samples or lower nf."
        )
    return int(nf)


def _signal_moments(samples: np.ndarray, theta: np.ndarray, correlation_length: int,
                    offsets: Sequence[int]) -> list[np.ndarray]:
    """C_q(x_j) = sum_{m=0}^{K-1} x_j^m c_{m+q} for each requested q, x_j = exp(i*theta_j).

    Chunked over trial frequencies so the Vandermonde block never exceeds a few MB
    for a long record.
    """
    num_trials = theta.size
    powers_of_m = np.arange(correlation_length)
    moments = [np.empty(num_trials, dtype=np.complex128) for _ in offsets]
    slices = [samples[q:q + correlation_length] for q in offsets]
    chunk = max(1, int(2 ** 21 // max(correlation_length, 1)))
    for start in range(0, num_trials, chunk):
        stop = min(start + chunk, num_trials)
        vandermonde = np.exp(1j * np.outer(theta[start:stop], powers_of_m))
        for moment, sliced in zip(moments, slices):
            moment[start:stop] = vandermonde @ sliced
    return moments


def _generate_U(samples: np.ndarray, theta: np.ndarray, correlation_length: int,
                power: int) -> np.ndarray:
    """The nf x nf symmetric matrix U(``power``), by the closed form in the header.

    ``theta_j = 2*pi*f_j*dt``, so the trial value is z_j = exp(-i*theta_j) and the
    quantity the sums are in is x_j = z_j^-1 = exp(+i*theta_j).
    """
    K = correlation_length
    x = np.exp(1j * theta)
    x_to_K = np.exp(1j * theta * K)
    moment_p, moment_p_plus_K = _signal_moments(samples, theta, K, (power, power + K))

    scaled_p = x * moment_p                        # x*C_p(x), the two leading terms
    scaled_p_plus_K = x * moment_p_plus_K          # x*C_{p+K}(x), the two trailing terms
    numerator = (
        scaled_p[None, :]
        - scaled_p[:, None]
        - x_to_K[:, None] * scaled_p_plus_K[None, :]
        + x_to_K[None, :] * scaled_p_plus_K[:, None]
    )
    denominator = x[None, :] - x[:, None]

    coincident = np.abs(denominator) < _COINCIDENT_TRIAL_TOL
    safe_denominator = np.where(coincident, 1.0, denominator)
    matrix = numerator / safe_denominator

    if coincident.any():  # Removable singularity: the exact x = y sum, not a limit.
        weight_index = np.arange(2 * K - 1)
        weights = np.minimum(weight_index + 1, 2 * K - 1 - weight_index)
        weighted = weights * samples[power:power + 2 * K - 1]
        rows, cols = np.nonzero(coincident)
        for row, col in zip(rows, cols):
            mean_theta = 0.5 * (theta[row] + theta[col])
            matrix[row, col] = np.exp(1j * mean_theta * weight_index) @ weighted
    return matrix


def _solve_modes(samples: np.ndarray, theta: np.ndarray, correlation_length: int, dt: float,
                 rank_tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter-diagonalization core: return (complex frequency, amplitude, error) arrays.

    Frequencies are MEEP's — omega/(2*pi) with omega = i*ln(u)/dt — so ``.real`` is
    the resonance and ``.imag`` the decay rate. Nothing here is screened; that is
    :func:`_screen_modes`.
    """
    K = correlation_length
    U0 = _generate_U(samples, theta, K, 0)
    U1 = _generate_U(samples, theta, K, 1)
    U2 = _generate_U(samples, theta, K, 2)

    left, singular_values, right_hermitian = np.linalg.svd(U0)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return (np.empty(0, dtype=np.complex128),) * 2 + (np.empty(0),)
    rank = int(np.count_nonzero(singular_values > rank_tol * singular_values[0]))
    if rank == 0:
        return (np.empty(0, dtype=np.complex128),) * 2 + (np.empty(0),)

    # Generalized problem U1 B = u U0 B reduced onto the surviving subspace:
    # with U0 = W S Y^H and B = Y_r S_r^-1/2 b, projecting on W_r^H gives an
    # ordinary eigenproblem for b, and B carries the eigenvector back.
    left_r = left[:, :rank]
    right_r = right_hermitian[:rank].conj().T
    inverse_sqrt = 1.0 / np.sqrt(singular_values[:rank])
    reduced = (inverse_sqrt[:, None] * (left_r.conj().T @ U1 @ right_r)) * inverse_sqrt[None, :]
    eigenvalues, reduced_vectors = np.linalg.eig(reduced)
    vectors = right_r @ (inverse_sqrt[:, None] * reduced_vectors)

    # Amplitude and error are RATIOS of bilinear forms in the eigenvector, so the
    # eigenvector scale (and the sqrt branch a B^T U0 B = 1 normalization would
    # need) cancels. Columns are unit-normalized only to make the floor test scale-free.
    column_norms = np.linalg.norm(vectors, axis=0)
    column_norms[column_norms == 0.0] = 1.0
    vectors = vectors / column_norms
    self_overlap = np.einsum("jk,jl,lk->k", vectors, U0, vectors)
    energy_overlap = np.einsum("jk,jl,lk->k", vectors.conj(), U0, vectors)
    two_step = np.einsum("jk,jl,lk->k", vectors, U2, vectors)
    initial_overlap = _signal_moments(samples, theta, K, (0,))[0] @ vectors

    defective = np.abs(self_overlap) < _NORMALIZATION_FLOOR * singular_values[0]
    defective |= np.abs(self_overlap) < _DEGENERACY_FLOOR * np.abs(energy_overlap)
    defective |= eigenvalues == 0.0
    safe_overlap = np.where(defective, 1.0, self_overlap)
    safe_eigenvalues = np.where(defective, 1.0, eigenvalues)

    amplitudes = np.where(defective, 0.0, initial_overlap ** 2 / safe_overlap)
    errors = np.where(
        defective,
        np.inf,
        np.abs(two_step / safe_overlap - safe_eigenvalues ** 2) / np.abs(safe_eigenvalues) ** 2,
    )
    frequencies = np.where(
        defective, np.nan, 1j * np.log(safe_eigenvalues) / (2 * np.pi * dt)
    )
    return frequencies, amplitudes, np.real(errors)


def _screen_modes(frequencies: np.ndarray, amplitudes: np.ndarray, errors: np.ndarray,
                  num_samples: int, dt: float, fmin: float, fmax: float, mxbands: int,
                  Q_thresh: float, rel_err_thresh: float, err_thresh: float,
                  rel_amp_thresh: float, amp_thresh: float) -> list[int]:
    """MEEP src/bands.cpp:110-186 — the screening, pairing, truncation and sort, in order."""
    order = sorted(range(len(frequencies)), key=lambda i: errors[i])  # bands.cpp:110-117
    if not order:
        return []
    min_err = errors[order[0]]                                        # bands.cpp:120

    # bands.cpp:139 screens on err <= rel_err_thresh * min_err. The default
    # rel_err_thresh is infinite and min_err is EXACTLY zero whenever some mode fits
    # the record to the last bit — reachable, not hypothetical: a lossless line at
    # f = 0.5 with a 284-sample record does it. inf * 0 is NaN, and every comparison
    # against NaN is false, so that product would screen out EVERY mode and return an
    # empty list for a perfect signal. "No limit" is spelled as no limit here.
    if math.isinf(rel_err_thresh) or math.isinf(min_err):
        relative_error_bound = math.inf
    else:
        relative_error_bound = rel_err_thresh * min_err

    kept: list[int] = []
    for index in order:                                               # bands.cpp:129-143
        frequency = frequencies[index]
        if not np.isfinite(frequency):
            continue
        magnitude = abs(frequency.real)
        quality = abs(frequency.real / (-2 * frequency.imag)) if frequency.imag else math.inf
        amplitude = abs(amplitudes[index])
        if (
            fmin <= magnitude <= fmax
            and quality > Q_thresh
            and errors[index] <= err_thresh
            and errors[index] <= relative_error_bound
            and amplitude > amp_thresh
        ):
            kept.append(index)

    # bands.cpp:126 takes the reference amplitude for `rel_amp_thresh` over the WHOLE
    # raw eigen-set. Here it is taken over the modes that passed the screens above, and
    # this is a deliberate divergence with a measurement behind it.
    #
    # The raw set of a real FDTD record is not the C library's raw set. Fitting the
    # transient that follows a source switching off costs a handful of very heavily
    # damped terms — measured on a conductivity-loaded cavity record (3001 samples, a
    # genuine mode at f = 0.496, Q = 99, amplitude 0.188): five artefacts at Q ~ 1.9
    # carrying amplitudes up to 33, which is 175x the record's own peak. Every one is
    # thrown out by `Q_thresh`, but taking the reference before that screen puts it at
    # 33 and `rel_amp_thresh = 0.01` — an ordinary request, the obvious way to ask for
    # "only strong modes" — then screens out the real mode too and returns NOTHING. The
    # C library culls those terms internally before MEEP ever sees them (fed the
    # identical record it returns exactly one mode), so bands.cpp's reference is over
    # physical modes there and this reproduces that rather than the letter of the loop.
    if rel_amp_thresh > 0.0 and kept:
        finite = [abs(amplitudes[index]) for index in kept if math.isfinite(abs(amplitudes[index]))]
        reference = max(finite) if finite else 0.0
        if reference > 0.0:
            kept = [
                index
                for index in kept
                if abs(amplitudes[index]) > rel_amp_thresh * reference
            ]

    # bands.cpp:144-172 — a real-valued record answers with a conjugate pair; drop
    # the mirror image, keeping whichever of the two carries the smaller error.
    # Comparisons are in harminv's dt = 1 units, where the pairing window is 2/n.
    eliminated: set[int] = set()
    for position, index in enumerate(kept):
        if index in eliminated:
            continue
        scaled = frequencies[index].real * dt
        if scaled >= 0.0:
            continue
        closest_diff = -2 * scaled
        closest_position = position
        for other_position, other in enumerate(kept):
            if other in eliminated:
                continue
            diff = abs(frequencies[other].real * dt + scaled)
            if diff < closest_diff:
                closest_position = other_position
                closest_diff = diff
        if closest_position != position and closest_diff < 2.0 / num_samples:
            partner = kept[closest_position]
            eliminated.add(index if errors[index] >= errors[partner] else partner)
    kept = [index for index in kept if index not in eliminated]

    kept = kept[:mxbands]                                             # bands.cpp:174
    kept.sort(key=lambda i: abs(frequencies[i].real))                 # bands.cpp:177-183
    return kept


def do_harminv(
    signal: Any,
    dt: float,
    fmin: float,
    fmax: float,
    mxbands: int = 100,
    spectral_density: float = 1.1,
    Q_thresh: float = 50.0,
    rel_err_thresh: float = math.inf,
    err_thresh: float = 0.01,
    rel_amp_thresh: float = -1.0,
    amp_thresh: float = -1.0,
    nf: int | None = None,
    rank_tol: float = DEFAULT_RANK_TOL,
) -> list[tuple[complex, complex, float]]:
    """Harmonically invert ``signal``, returning ``(freq, amp, err)`` per mode.

    Mirrors MEEP's ``mp.py_do_harminv`` (``python/meep.i:288``) argument for
    argument and default for default, including the screening thresholds. ``freq``
    is complex: ``.real`` the resonance (MEEP frequency units, 1/time) and
    ``.imag`` the decay rate, negative for a decaying mode. ``.real`` is reported
    as a magnitude, as MEEP reports it.

    ``signal`` is one scalar per time step, real or complex, host or device array.
    ``dt`` is the interval BETWEEN SAMPLES, which is the step size only if every
    step was recorded.

    Two arguments are ours, not MEEP's: ``nf`` overrides the derived trial-basis
    size, and ``rank_tol`` sets the SVD truncation (see the module header).
    """
    samples = _as_host_signal(signal)
    num_samples = samples.size

    if not (dt > 0.0) or not math.isfinite(dt):
        raise ValueError(f"harminv needs a positive finite sample interval dt, got {dt!r}")
    if not math.isfinite(fmin) or not math.isfinite(fmax):
        raise ValueError(f"harminv band must be finite, got fmin={fmin!r} fmax={fmax!r}")
    if fmax <= fmin:
        raise ValueError(
            f"harminv band is empty or inverted: fmin={fmin!r} fmax={fmax!r}. "
            "The screening keeps only modes inside [fmin, fmax], so this would "
            "silently return nothing."
        )
    if mxbands < 1:
        raise ValueError(f"harminv mxbands must be at least 1, got {mxbands!r}")
    if not (spectral_density > 0.0):
        raise ValueError(f"harminv spectral_density must be positive, got {spectral_density!r}")
    if not (rank_tol >= 0.0) or rank_tol >= 1.0:
        raise ValueError(f"harminv rank_tol must lie in [0, 1), got {rank_tol!r}")
    # A NaN threshold compares false against every mode, so it screens the whole
    # result set out and returns an empty list that looks like "no resonances".
    for name, value in (
        ("Q_thresh", Q_thresh), ("rel_err_thresh", rel_err_thresh),
        ("err_thresh", err_thresh), ("rel_amp_thresh", rel_amp_thresh),
        ("amp_thresh", amp_thresh),
    ):
        if math.isnan(value):
            raise ValueError(f"harminv {name} is NaN; it would screen out every mode")
    if math.isinf(rel_amp_thresh) or math.isinf(amp_thresh):
        raise ValueError(
            "harminv amp_thresh and rel_amp_thresh must be finite "
            f"(got {amp_thresh!r}, {rel_amp_thresh!r}); use -1 for MEEP's 'no limit'"
        )

    correlation_length = _correlation_length(num_samples)
    if correlation_length < MIN_BASIS_SIZE:
        raise ValueError(
            f"harminv needs at least {2 * MIN_BASIS_SIZE + 2} samples to form its "
            f"correlation matrices, got {num_samples}"
        )

    basis_size = _resolve_basis_size(
        num_samples, correlation_length, dt, fmin, fmax, mxbands, spectral_density, nf
    )

    if not samples.any():  # bands.cpp:82-88 — an all-zero record has no modes, not zero-frequency ones.
        return []

    trial_frequencies = np.linspace(fmin, fmax, basis_size)
    theta = 2 * np.pi * trial_frequencies * dt
    frequencies, amplitudes, errors = _solve_modes(
        samples, theta, correlation_length, dt, rank_tol
    )
    kept = _screen_modes(
        frequencies, amplitudes, errors, num_samples, dt, fmin, fmax, mxbands,
        Q_thresh, rel_err_thresh, err_thresh, rel_amp_thresh, amp_thresh,
    )
    return [
        (complex(abs(frequencies[i].real), frequencies[i].imag),  # bands.cpp:189 reports |Re f|
         complex(amplitudes[i]),
         float(errors[i]))
        for i in kept
    ]


def harminv(signal: Any, dt: float, fmin: float, fmax: float, mxbands: int = 100,
            **kwargs: Any) -> list[Mode]:
    """Harmonically invert ``signal``, returning MEEP ``Mode`` records.

    The same computation as :func:`do_harminv` with MEEP's ``Q`` derivation applied
    (``python/simulation.py:1198``). Modes come back sorted by increasing |freq|.
    Keyword arguments are :func:`do_harminv`'s.
    """
    return [
        Mode(freq.real, freq.imag,
             freq.real / (-2 * freq.imag) if freq.imag != 0 else math.inf,
             amp, err)
        for freq, amp, err in do_harminv(signal, dt, fmin, fmax, mxbands, **kwargs)
    ]


class Harminv:
    """MEEP's ``mp.Harminv`` step function: collect a point field, analyse at the end.

    ``h = Harminv(c=Ez, pt=..., fcen=..., df=...)`` then hand ``h`` to the run loop.
    It appends ``sim.get_field_point(c, pt)`` on every step and, when the run
    finishes, fills ``h.modes`` with :class:`Mode` records. Keep a reference to the
    instance — the results live on it, exactly as in MEEP.

    Only analyse data taken AFTER THE SOURCES ARE OFF. A driven signal is not a sum
    of free decaying modes and the fit will report the source, not the cavity.

    The six screening thresholds are instance attributes with MEEP's defaults, so
    ``h.Q_thresh = 200`` before the run behaves as it does there. ``rank_tol`` is
    ours (module header).

    The collection half needs a run loop that offers MEEP's step-function protocol
    — ``sim.meep_time()`` and ``sim.get_field_point(c, pt)``. Without one, feed the
    series in directly: set ``h.data`` and ``h.data_dt`` and call
    :meth:`analyze`, or skip the class and call :func:`harminv`.
    """

    def __init__(self, c: Any = None, pt: Any = None, fcen: float | None = None,
                 df: float | None = None, mxbands: int | None = None) -> None:
        self.c = c                       # Field component to record, e.g. the driver's Ez.
        self.pt = pt                     # Point to record it at.
        self.fcen = fcen                 # Band centre; the band is fcen -+ df/2.
        self.df = df                     # Band width.
        self.mxbands = mxbands           # Cap on reported modes; None means MEEP's 100.
        self.data: list[complex] = []    # The collected time series.
        self.data_dt = 0                 # Interval between collected samples.
        self.modes: list[Mode] = []      # Filled by analyze(); MEEP's result attribute.
        self.spectral_density = 1.1      # simulation.py:1128-1134, MEEP's defaults verbatim.
        self.Q_thresh = 50.0
        self.rel_err_thresh = math.inf
        self.err_thresh = 0.01
        self.rel_amp_thresh = -1.0
        self.amp_thresh = -1.0
        self.rank_tol = DEFAULT_RANK_TOL  # Ours; see the module header.
        self._last_time = 0.0

    def __call__(self, sim: Any, todo: str) -> None:
        """MEEP's two-argument step-function protocol: collect on 'step', analyse on 'finish'."""
        if todo == "step":
            self._collect(sim)
        elif todo == "finish":
            self.analyze()

    def _collect(self, sim: Any) -> None:
        for attribute in ("meep_time", "get_field_point"):
            if not hasattr(sim, attribute):
                raise ValueError(
                    f"Harminv needs a run loop offering {attribute}(); this simulation "
                    "object has none. Collect the series yourself and set "
                    "harminv_instance.data / .data_dt, or call harminv() directly."
                )
        now = sim.meep_time()
        self.data_dt = now - self._last_time
        self._last_time = now
        self.data.append(sim.get_field_point(self.c, self.pt))

    def analyze(self, dt: float | None = None) -> list[Mode]:
        """Invert the collected series into :attr:`modes` and return them."""
        if self.fcen is None or self.df is None:
            raise ValueError("Harminv needs both fcen and df to define its band")
        interval = self.data_dt if dt is None else dt
        if not interval:
            raise ValueError(
                "Harminv has no sample interval: set data_dt to the time between "
                "collected samples, or pass dt="
            )
        self.modes = harminv(
            self.data,
            interval,
            self.fcen - self.df / 2,
            self.fcen + self.df / 2,
            100 if not self.mxbands else self.mxbands,  # simulation.py:1207, 0 means the default
            spectral_density=self.spectral_density,
            Q_thresh=self.Q_thresh,
            rel_err_thresh=self.rel_err_thresh,
            err_thresh=self.err_thresh,
            rel_amp_thresh=self.rel_amp_thresh,
            amp_thresh=self.amp_thresh,
            rank_tol=self.rank_tol,
        )
        return self.modes


__all__ = ["Harminv", "Mode", "do_harminv", "harminv"]
