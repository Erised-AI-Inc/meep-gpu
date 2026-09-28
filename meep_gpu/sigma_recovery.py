"""Recover per-point susceptibility sigma from MEEP's public ``get_chi1inv`` API.

THE COMPATIBILITY PATH. ``fields.get_susceptibility_sigma`` — the reader added by
``parity/meep_gpu/meep-sigma-reader.patch`` — returns each susceptibility's sigma
at a Yee point exactly, but it exists only on a patched MEEP. This module
recovers the same quantity from a STOCK MEEP of any version, so a structured
dispersive cell lifts against whatever MEEP the caller already has installed.

MEEP evaluates ``eps(w) = eps_inf + SUM_n sigma_n * chi_n(w)`` at every point
(``structure_chunk::get_chi1inv_at_pt``'s frequency branch, monitor.cpp), with
``chi_n`` fixed by the pole and ``sigma_n`` the per-point unknown. Sampling
``get_chi1inv`` at several frequencies and solving the resulting linear system
therefore recovers sigma without any C++ change. Three things make the
difference between a recovery that works and one that is confidently wrong, and
all three are measured rather than assumed.

**1. Solve for the PRODUCT, never for sigma directly.** MEEP's Drude idiom is a
near-zero ``frequency`` carrying an enormous ``sigma`` — ``meep.materials.Au``
ships ``f = 1e-10`` with ``sigma = 4.03e+21``, and what enters ``chi`` is the
product ``sigma * w0**2 ~ 40``. A basis written the obvious way, with
``sigma_n`` as the unknown, spans forty orders of magnitude and is structurally
singular: measured on that exact material, condition number **8.6e+16**,
recovered errors up to **3000 %**, the Drude term wrong by 33 orders of
magnitude, and one sigma returned as **-59.2 where the truth is +2.01** — a
negative sigma is gain, and ``lstsq`` raises nothing on the way out. This module
solves for ``c_n = sigma_n * w0_n**2`` and divides afterwards, which preserves
relative error and drops the condition number to **1.0e+03**.

**2. Sample geometrically across the poles.** Measured on the same material,
worst relative error over all six poles:

    linear 0.6-3.0, n = N+1          cond 3.64e+05   worst 2.8e-05
    linear 0.6-3.0, oversampled      cond 1.47e+05   worst 6.6e-05
    geometric 0.1-12, oversampled    cond 3.12e+03   worst 1.9e-05
    geometric spanning the poles     cond 1.02e+03   worst 3.2e-06
    the same at 4N samples           cond 1.02e+03   worst 1.1e-06

Conditioning is set by the BAND, not the sample count; oversampling then buys
accuracy at linear cost.

**3. Cross-validate at a held-out frequency, and refuse rather than return.**
Conditioning alone does not catch a wrong pole list — and the pole list is an
assumption this module cannot verify from inside. After solving, the recovered
model is evaluated at a frequency that was NOT used in the fit and compared
against MEEP's own value there. A pole list that disagrees with the structure
MEEP actually built fails that check; without it, the failure mode is a
plausible number and no error. :class:`SigmaRecoveryRefused` is raised rather
than returning a number nobody can audit.

The matrix depends only on the poles, so it is factored ONCE and applied to
every point as a matrix product — the per-point cost is the ``get_chi1inv``
calls, not a solve.

Accuracy against the exact reader, measured point by point: see
``test_sigma_recovery.py``. This path is 4-6 significant digits where the reader
is exact, which is why the reader stays preferred when it is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

# Condition number above which the solve is refused outright. MEEP built with
# --enable-single stores sigma as float32, so the sampled eps carries ~7 decimal
# digits; a solve of condition number C spends log10(C) of them. 1e8 leaves
# nothing at all and is the hard stop. The realistic materials measured here sit
# at 1e3 (Au, geometric band), four orders below it — this bound refuses the
# structurally singular formulations, not the workable ones.
CONDITION_LIMIT = 1.0e8

# Relative disagreement at the held-out frequency above which the recovery is
# refused. The measured residual for a correct pole list on Au is ~1e-6; 1e-3
# is three orders looser, so it accepts real recoveries and rejects a pole list
# that does not describe the structure.
HOLDOUT_TOLERANCE = 1.0e-3


class SigmaRecoveryRefused(RuntimeError):
    """The recovery cannot be trusted, so no number is returned.

    Raised for an ill-conditioned basis, a degenerate pole set, or a
    cross-validation failure at the held-out frequency. Every message names the
    measured quantity that failed and what it was measured against.
    """


@dataclass(frozen=True)
class SusceptibilityPole:
    """One Lorentz or Drude pole, in MEEP's own units (f = omega/2pi).

    ``drude`` selects MEEP's ``no_omega_0_denominator`` branch: a Lorentz term
    is ``sigma*w0^2 / (w0^2 - w^2 - i*g*w)`` and a Drude term drops the ``w0^2``
    from the denominator only (susceptibility.cpp). ``sigma`` is deliberately
    absent — it is the per-point unknown this module solves for.
    """

    frequency: float
    gamma: float
    drude: bool = False

    def basis(self, w: np.ndarray) -> np.ndarray:
        """The pole's contribution per unit of ``c = sigma * w0**2``."""
        denominator = (0.0 if self.drude else self.frequency ** 2) - w * w - 1j * self.gamma * w
        return 1.0 / denominator

    @property
    def scale(self) -> float:
        """``w0**2`` — the factor between the solved ``c`` and ``sigma`` itself."""
        return float(self.frequency) ** 2


def choose_frequencies(poles: Sequence[SusceptibilityPole], oversample: int = 2) -> np.ndarray:
    """Geometric sampling band spanning the poles — the measured best strategy.

    Half the lowest finite pole to 1.5x the highest, geometrically spaced, with
    ``oversample * (N + 1)`` points. The band is what sets the conditioning (the
    table in the module docstring); the count then buys accuracy linearly.

    A pole set with no finite frequency at all is pure Drude, where every basis
    element is ``1/(-w^2 - i*g*w)`` and the natural scale is the damping rather
    than a resonance; the band is taken from the gammas instead.
    """
    finite = [p.frequency for p in poles if p.frequency > 1e-6]
    if finite:
        low, high = 0.5 * min(finite), 1.5 * max(finite)
    else:
        gammas = [p.gamma for p in poles if p.gamma > 0.0]
        if not gammas:
            raise SigmaRecoveryRefused(
                "every susceptibility has frequency and gamma of zero, so no pole has a scale "
                "to sample around; nothing can be recovered from chi1inv."
            )
        low, high = 0.5 * min(gammas), 1.5 * max(gammas)
    count = max(oversample * (len(poles) + 1), len(poles) + 1)
    return np.geomspace(low, high, count)


@dataclass
class SigmaRecoveryPlan:
    """A factored recovery for one pole set — reusable across every grid point.

    The matrix depends only on the poles and the sampling frequencies, so it is
    built and pseudo-inverted once here; :meth:`recover` is then a matrix
    product over however many points the caller has.
    """

    poles: tuple
    frequencies: np.ndarray
    holdout_frequency: float
    condition_number: float
    _pseudo_inverse: np.ndarray = field(repr=False)
    _holdout_row: np.ndarray = field(repr=False)

    @property
    def sample_count(self) -> int:
        return int(self.frequencies.size)

    def recover(self, eps_samples: Any, holdout_eps: Any) -> np.ndarray:
        """Solve for sigma at every point, then cross-validate at the held-out point.

        Args:
            eps_samples: ``(nfreq, ...)`` complex permittivity sampled at
                :attr:`frequencies` — MEEP's ``1 / get_chi1inv(c, d, iloc, w)``.
            holdout_eps: ``(...)`` the same quantity at
                :attr:`holdout_frequency`, which took no part in the fit.

        Returns:
            ``(npoles, ...)`` real sigma per pole per point.

        Raises:
            SigmaRecoveryRefused: if the recovered model disagrees with MEEP at
                the held-out frequency by more than :data:`HOLDOUT_TOLERANCE`.
        """
        samples = np.asarray(eps_samples)
        if samples.shape[0] != self.sample_count:
            raise ValueError(
                f"expected {self.sample_count} sampled frequencies, got {samples.shape[0]}."
            )
        flat = samples.reshape(self.sample_count, -1)
        # The unknowns (eps_inf and every c_n) are REAL, so the real and imaginary
        # parts are two real equations per frequency rather than one complex one.
        # Stacking them enforces that physics instead of hoping the solve honours
        # it, and it doubles the equation count for free.
        stacked = np.concatenate([flat.real, flat.imag], axis=0)
        solution = self._pseudo_inverse @ stacked           # (1 + npoles, npoints)

        predicted = self._holdout_row @ solution            # (npoints,)
        actual = np.asarray(holdout_eps).reshape(-1)
        scale = np.maximum(np.abs(actual), 1e-30)
        residual = float(np.max(np.abs(predicted - actual.real) / scale))
        if not np.isfinite(residual) or residual > HOLDOUT_TOLERANCE:
            raise SigmaRecoveryRefused(
                f"the recovered susceptibility model disagrees with MEEP at the held-out "
                f"frequency {self.holdout_frequency:g} by {residual:.3e} (limit "
                f"{HOLDOUT_TOLERANCE:g}). The pole list this plan was built from does not "
                f"describe the structure MEEP actually rasterized — an extra_materials entry "
                f"that was not enumerated, or two declarations that susceptibility_equiv "
                f"merged into one chain entry. Recovered sigma would be a confident wrong "
                f"answer, so none is returned."
            )

        coefficients = solution[1:]                          # c_n = sigma_n * w0_n**2
        sigma = np.empty_like(coefficients)
        for index, pole in enumerate(self.poles):
            sigma[index] = coefficients[index] / pole.scale
        return sigma.reshape((len(self.poles),) + samples.shape[1:])


def plan_recovery(
    poles: Sequence[SusceptibilityPole],
    frequencies: Optional[Sequence[float]] = None,
    oversample: int = 2,
) -> SigmaRecoveryPlan:
    """Build and factor the recovery for one pole set, refusing a hopeless one.

    The returned plan carries the frequencies the caller must sample, plus one
    held-out frequency for the cross-check :meth:`SigmaRecoveryPlan.recover`
    performs.
    """
    poles = tuple(poles)
    if not poles:
        raise SigmaRecoveryRefused("no susceptibilities to recover.")
    for index, pole in enumerate(poles):
        if pole.scale <= 0.0:
            raise SigmaRecoveryRefused(
                f"susceptibility {index} has frequency {pole.frequency!r}; sigma is recovered "
                f"as c/w0**2 and a zero w0 makes that division undefined. MEEP's own Drude "
                f"idiom uses a tiny but nonzero frequency (1e-10 in meep.materials.Au)."
            )
    grid = (np.asarray(frequencies, dtype=float) if frequencies is not None
            else choose_frequencies(poles, oversample))
    # The held-out frequency sits inside the band but off the sampling grid: the
    # geometric mean of the two middle samples, which is never one of them.
    middle = grid.size // 2
    holdout = float(np.sqrt(grid[middle - 1] * grid[middle])) if grid.size > 1 else float(grid[0] * 1.1)

    def rows(w):
        return np.concatenate([[1.0], [p.basis(w) for p in poles]])

    complex_matrix = np.array([rows(w) for w in grid])
    matrix = np.concatenate([complex_matrix.real, complex_matrix.imag], axis=0)
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > CONDITION_LIMIT:
        raise SigmaRecoveryRefused(
            f"the susceptibility basis is singular to working precision (condition number "
            f"{condition:.3e}, limit {CONDITION_LIMIT:.0e}) over the sampled band "
            f"[{grid[0]:g}, {grid[-1]:g}]. Two poles are indistinguishable there, or the band "
            f"does not resolve them. Recovering sigma from chi1inv is not possible for this "
            f"material; use a MEEP carrying fields.get_susceptibility_sigma instead."
        )
    return SigmaRecoveryPlan(
        poles=poles,
        frequencies=grid,
        holdout_frequency=holdout,
        condition_number=condition,
        _pseudo_inverse=np.linalg.pinv(matrix),
        _holdout_row=rows(holdout).real,
    )
