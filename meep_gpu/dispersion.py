"""
Dispersive-material model for the MEEP-compatible FDTD engine: MEEP's Lorentz and
Drude susceptibilities, the auxiliary-differential-equation state they carry, and
the closed forms that make both testable without stepping Maxwell at all.

``Susceptibility`` is one term of MEEP's sum, in MEEP's own user units
(``frequency`` = f_n = omega_n/2*pi, ``gamma`` = gamma_n/2*pi), so a term written
here transcribes ``mp.LorentzianSusceptibility`` / ``mp.DrudeSusceptibility``
argument for argument. It exposes four pure functions and no arrays:

* ``coefficients(dt)`` — the three recurrence constants MEEP computes in
  ``susceptibility.cpp:192-195``;
* ``chi1(f)`` — the CONTINUUM susceptibility MEEP reports through
  ``mp.Medium.epsilon(f)`` (``susceptibility.cpp:297-307``);
* ``chi1_discrete(f, dt)`` — the EXACT susceptibility of the discrete recurrence,
  which has no discretisation error and is therefore the target a correctness test
  can hold at float32 round-off rather than at a physics tolerance;
* ``poles(dt)`` / ``stability(dt)`` — the roots of the recurrence's characteristic
  polynomial and MEEP's own (disabled since issue #12) instability predicate.

``PolarizationState`` owns the P / P_prev volumes for one susceptibility on one
grid and advances them element-wise. It has no spatial stencil at all, which is
why the driver needs no boundary pass for P: a Bloch phase, a mirror ghost and a
PML all reach the polarization through its drive field and nowhere else.

Units. ``chi1`` is invariant to whether its frequencies carry the 2*pi, because
the factor cancels between numerator and denominator; ``coefficients`` is NOT,
because it carries a bare dt. Every frequency inside the recurrence is therefore
angular, and every frequency crossing this module's boundary is MEEP's. A 2*pi
dropped from one and not the other moves a resonance by 6.28x while leaving every
field smooth and plausible, so the two conventions never share a variable name
here.

Two structural facts drive the rest of the engine:

* with any susceptibility present, E can no longer be recomputed from D on
  demand — P advances at the end of the step, so ``D * inv_eps`` stops being the
  E of the step just finished. MEEP allocates a stored E for exactly this reason
  (``update_eh.cpp:166-171``), and so does :class:`~.fields.Fields`;
* the polarization is driven by W = (D - sum P) * inv_eps, not by the stored E.
  Those differ only INSIDE a PML, where the stored E has already had the
  absorbing accumulation applied — so driving P from E is exactly right
  everywhere except in the absorber, which is the shape of a silent wrong answer.

``inv_eps`` becomes the INSTANTANEOUS response 1/eps_infinity once a
susceptibility is present, not 1/eps(omega). Nothing in the array changes for a
non-dispersive run.

Debye relaxation is deliberately not a native kind. MEEP has no Debye
susceptibility, and the parity model requires that CPU MEEP can run every job, so
a native kind would be a GPU-only capability with no oracle. It ships instead as
the overdamped-Lorentz mapping in :func:`debye_to_lorentzian`, whose residual is
computed and returned rather than swallowed.

Array work goes through ``grid.xp``; this module imports neither cupy nor meep,
and its scalar mathematics is float64 throughout, cast to the field dtype only
where it meets an array.

P follows the field dtype: float32 in a real run, complex64 in a complex one
(``PolarizationState`` takes it from ``Fields._field_dtype``). The recurrence
coefficients are REAL — ``chi1_discrete`` and ``poles`` are complex, but
``coefficients`` returns three real numbers — so the update never needs the
imaginary plane, and a real run's P is the real part of the complex run's P to
the bit. Allocating P complex64 over real fields would silently double a
six-term metal fit's memory, which is the largest single allocation such a run
makes.

MEEP source references: src/susceptibility.cpp (lorentzian_susceptibility::
update_P, subtract_P, chi1, the disabled lorentzian_unstable), src/update_eh.cpp
(f_minus_p, the E allocation rule), src/update_pols.cpp (the W selection),
python/geom.py (LorentzianSusceptibility / DrudeSusceptibility / _get_epsmu),
doc/docs/Materials.md (the model, the stability rule of thumb, conductivity).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence, Tuple

if TYPE_CHECKING:  # Annotations only; the runtime coupling is duck-typed on grid.xp.
    from .grid import Grid

LORENTZIAN = "lorentzian"  # mp.LorentzianSusceptibility.
DRUDE = "drude"  # mp.DrudeSusceptibility (susceptibility.cpp: no_omega_0_denominator).
SUSCEPTIBILITY_KINDS: Tuple[str, ...] = (LORENTZIAN, DRUDE)

# The three electric components a susceptibility can drive, and their Yee shifts
# along (x, y, z) in units of dx/2 (meep/vec.hpp iyee_shift): 1 = half-integer on
# that axis, 0 = integer. MEEP point-samples sigma at each component's OWN
# position and never subpixel-averages it (anisotropic_averaging.cpp:333-345
# against :252-257 for chi1inv), so the three differ by half a cell at any
# material interface.
E_COMPONENTS: Tuple[str, str, str] = ("Ex", "Ey", "Ez")
E_YEE_SHIFTS: Dict[str, Tuple[int, int, int]] = {
    "Ex": (1, 0, 0),
    "Ey": (0, 1, 0),
    "Ez": (0, 0, 1),
}

# |z| above 1 by more than this is refused; between 1 and this it is reported as
# marginal. float32 coefficients cannot resolve a pole magnitude more finely, and
# the physically marginal cases (Drude's z = 1 DC mode, a lossless Lorentzian)
# sit exactly at 1 and must not be refused.
POLE_TOLERANCE = 1e-6


class DispersionInstability(ValueError):
    """A susceptibility whose discrete recurrence has a pole outside the unit circle.

    Raised at setup, never mid-run: a Lorentzian too fast for the timestep does not
    crash, it returns a large, smooth, entirely plausible field, so the refusal has
    to happen before any array is allocated.
    """


@dataclass(frozen=True)
class StabilityReport:
    """The discrete-pole verdict for one susceptibility at one timestep.

    ``pole_magnitude`` is max |z| over the roots of the recurrence's characteristic
    polynomial. Exactly 1.0 is *marginal*, not unstable, and it is the physically
    correct answer for two shipped cases: a Drude term (whose z = 1 root is the
    free-electron DC mode) and a lossless Lorentzian (gamma = 0, an undamped
    oscillator). Neither grows; both simply never forget their initial transient.
    """

    frequency: float  # MEEP units, f_0 = omega_0/2*pi.
    gamma: float  # MEEP units, gamma/2*pi.
    kind: str
    dt: float
    pole_magnitude: float
    stable: bool
    marginal: bool

    @property
    def growth_per_1000_steps(self) -> float:  # What an unstable pole multiplies the field by.
        """|z|**1000, saturated at inf — the number that says whether a run is salvageable."""
        try:
            return float(self.pole_magnitude**1000)
        except OverflowError:
            return math.inf

    def describe(self) -> str:  # One line naming the term and its verdict, for an exception message.
        return (
            f"{self.kind} susceptibility f0={self.frequency:g}, gamma={self.gamma:g} at "
            f"dt={self.dt:g}: max|z|={self.pole_magnitude:.9f} "
            f"({'stable' if self.stable else 'UNSTABLE'}"
            f"{', marginal' if self.marginal else ''}), "
            f"growth over 1000 steps {self.growth_per_1000_steps:.3e}"
        )


@dataclass(frozen=True)
class Susceptibility:
    """One term of MEEP's sum_n, in MEEP units — the transcription of mp.*Susceptibility.

    ``frequency`` is f_n = omega_n/2*pi and ``gamma`` is gamma_n/2*pi, exactly the
    numbers ``mp.LorentzianSusceptibility(frequency=..., gamma=...)`` takes. For a
    Drude term ``frequency`` is a dimensional scale factor multiplying sigma, NOT a
    resonance (python/geom.py:786-813) — the difference is the whole content of
    MEEP's ``no_omega_0_denominator`` flag.

    The model (doc/docs/Materials.md, MEEP's exp(-i*omega*t) convention)::

        D = eps_inf E + sum_n P_n
        d2P/dt2 + gamma dP/dt + omega_0^2 P = sigma omega_0^2 E     (Lorentz)
        d2P/dt2 + gamma dP/dt                = sigma omega_0^2 E     (Drude)

    ``sigma`` is deliberately NOT a field of this class: it is a per-cell,
    per-component volume owned by :class:`PolarizationState`, whereas
    ``frequency``/``gamma``/``kind`` are scalars that fix the recurrence
    coefficients for the whole grid. Every method here is evaluated at sigma = 1.

    Gain (gamma < 0) is refused: doc/docs/Materials.md records that a
    frequency-independent negative imaginary part of epsilon is unconditionally
    unstable, and Im chi1(f) >= 0 for f > 0 is equivalent to gamma >= 0 here.
    """

    frequency: float
    gamma: float = 0.0
    kind: str = LORENTZIAN

    def __post_init__(self):  # Validate the model parameters; passivity is one of them.
        if self.kind not in SUSCEPTIBILITY_KINDS:
            raise ValueError(
                f"Susceptibility kind must be one of {SUSCEPTIBILITY_KINDS}, got {self.kind!r}. "
                f"Noisy, gyrotropic and multilevel-atom kinds are not implemented on this engine; "
                f"those jobs belong on CPU MEEP."
            )
        frequency = float(self.frequency)
        gamma = float(self.gamma)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError(
                f"Susceptibility frequency is MEEP's f_n = omega_n/2*pi and must be finite and "
                f"strictly positive, got {self.frequency!r}. A zero frequency drives nothing at "
                f"all (the whole term's numerator is sigma*omega_n**2), which would score as a "
                f"perfectly transparent material rather than as an error."
            )
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError(
                f"Susceptibility gamma is MEEP's gamma_n/2*pi and must be finite and >= 0, got "
                f"{self.gamma!r}. A negative gamma is gain: it makes Im eps(f) < 0 for f > 0, "
                f"which doc/docs/Materials.md records as unconditionally unstable in the time "
                f"domain because every run contains both signs of omega."
            )
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "gamma", gamma)

    @property
    def is_drude(self) -> bool:  # MEEP's no_omega_0_denominator: the restoring force is dropped.
        return self.kind == DRUDE

    def coefficients(self, dt: float) -> Tuple[float, float, float]:
        """Return (c_now, c_prev, c_drive) for ``P_new = c_now*P + c_prev*P_prev + c_drive*sigma*W``.

        Verbatim from MEEP susceptibility.cpp:192-195 and the isotropic loop at
        :251-258::

            omega2pi = 2*pi*omega_0;  g2pi = gamma*2*pi
            omega0dtsqr = omega2pi**2 * dt**2
            gamma1inv = 1/(1 + g2pi*dt/2);  gamma1 = 1 - g2pi*dt/2
            omega0dtsqr_denom = 0 if Drude else omega0dtsqr
            p = gamma1inv*(p*(2 - omega0dtsqr_denom) - gamma1*pp + omega0dtsqr*(s*w))

        The Drude flag zeroes the restoring force in ``c_now`` and NOTHING else —
        ``c_drive`` keeps the full omega0dtsqr. Applying it to the drive instead
        gives P == 0, i.e. a metal that transmits like vacuum, which scores as a
        pristine transmission spectrum rather than as a failure.

        Computed in float64 and returned as Python floats; the caller casts where
        the constants meet the field arrays. MEEP rounds each intermediate to
        float32 (``realnum``), which this deliberately does not reproduce — the
        closed form is the same and evaluating it more accurately can only move the
        engine towards the continuum answer, not away from it.
        """
        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}.")
        omega_rad = 2.0 * math.pi * self.frequency  # Angular: the recurrence carries a bare dt.
        gamma_rad = 2.0 * math.pi * self.gamma
        omega0dtsqr = (omega_rad * step) ** 2
        half_gamma_dt = gamma_rad * step / 2.0
        gamma1inv = 1.0 / (1.0 + half_gamma_dt)
        gamma1 = 1.0 - half_gamma_dt
        restoring = 0.0 if self.is_drude else omega0dtsqr  # MEEP's omega0dtsqr_denom.
        return (
            gamma1inv * (2.0 - restoring),
            -gamma1inv * gamma1,
            gamma1inv * omega0dtsqr,
        )

    def chi1(self, frequency: float) -> complex:
        """Continuum susceptibility at sigma = 1 — MEEP susceptibility.cpp:297-307.

        ``chi1`` is what ``mp.Medium.epsilon(f)`` reports (python/geom.py
        ``eval_susceptibility``), so it is the material model's parity target and it
        is what a caller asking "what is n at my frequency?" means. It is NOT what
        the time-stepped engine reproduces: see :meth:`chi1_discrete`, which differs
        from it by 5.3e-3 ON RESONANCE at resolution 40.

        The 2*pi cancels between numerator and denominator, so this may be evaluated
        in MEEP units or angular ones as long as it is consistent; MEEP units are
        used here so the arguments match the dataclass fields.
        """
        probe = float(frequency)
        if not math.isfinite(probe) or probe < 0.0:
            raise ValueError(f"chi1 needs a finite, non-negative frequency, got {frequency!r}.")
        f0 = self.frequency
        if self.is_drude:
            if probe == 0.0:
                raise ValueError(
                    f"A Drude susceptibility (f0={f0:g}, gamma={self.gamma:g}) diverges at zero "
                    f"frequency — free carriers have no static polarizability — so chi1(0) has no "
                    f"finite value. Probe at a frequency inside the simulated band."
                )
            return _ratio_or_raise(f0 * f0, complex(-probe * probe, -self.gamma * probe))
        return _ratio_or_raise(f0 * f0, complex(f0 * f0 - probe * probe, -self.gamma * probe))

    def chi1_discrete(self, frequency: float, dt: float) -> complex:
        """EXACT susceptibility of the discrete recurrence at sigma = 1 — no discretisation error.

        Substituting P^m = P z^m, W^m = W z^m with z = exp(-i*omega*dt) into the
        recurrence of :meth:`coefficients` and solving for P/W::

            chi_tilde(omega) = w2 / [ w2d - 4 sin^2(omega dt/2) - i (gamma_rad dt) sin(omega dt) ]

        with w2 = (omega_0 dt)**2 and w2d = 0 for Drude, w2 for Lorentz (all angular).
        Dividing through by dt**2 and taking dt -> 0 recovers :meth:`chi1` exactly,
        second order in dt.

        This is the analytic target that makes the kernel testable at round-off:
        a measurement of the engine against :meth:`chi1` is measuring the engine's
        discretisation, but a measurement against this is measuring only its
        correctness.
        """
        probe = float(frequency)
        step = float(dt)
        if not math.isfinite(probe) or probe < 0.0:
            raise ValueError(f"chi1_discrete needs a finite, non-negative frequency, got {frequency!r}.")
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}.")
        omega_rad = 2.0 * math.pi * probe
        omega0_rad = 2.0 * math.pi * self.frequency
        gamma_rad = 2.0 * math.pi * self.gamma
        w2 = (omega0_rad * step) ** 2
        restoring = 0.0 if self.is_drude else w2
        denominator = complex(
            restoring - 4.0 * math.sin(omega_rad * step / 2.0) ** 2,
            -(gamma_rad * step) * math.sin(omega_rad * step),
        )
        return _ratio_or_raise(w2, denominator)

    def poles(self, dt: float) -> Tuple[complex, complex]:
        """Roots of ``(1+g2) z**2 - (2-w2d) z + (1-g2) = 0`` — the recurrence's z-plane poles.

        For a Drude term these are EXACTLY z = 1 and z = (1-g2)/(1+g2): the first is
        the free-electron DC mode, which never decays, so an abruptly started drive
        leaves a permanent constant in P. That is physics, not an instability — no
        pole test can flag |z| = 1 — and it is why a Drude medium must be driven by a
        pulsed, zero-DC source.
        """
        step = float(dt)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}.")
        half_gamma_dt = 2.0 * math.pi * self.gamma * step / 2.0
        w2 = (2.0 * math.pi * self.frequency * step) ** 2
        restoring = 0.0 if self.is_drude else w2
        a = 1.0 + half_gamma_dt
        b = -(2.0 - restoring)
        c = 1.0 - half_gamma_dt
        discriminant = cmath.sqrt(complex(b * b - 4.0 * a * c, 0.0))
        return ((-b + discriminant) / (2.0 * a), (-b - discriminant) / (2.0 * a))

    def stability(self, dt: float) -> StabilityReport:
        """MEEP's own discrete-pole instability predicate, susceptibility.cpp:160-175.

        MEEP ships that function under ``#if 0`` with "FIXME: this test seems to be
        too conservative (issue #12)", so the algebra is MEEP's and the policy is
        not: |z| > 1 + :data:`POLE_TOLERANCE` is refused, and the band between 1 and
        that is reported as marginal rather than rejected. The marginal band is not
        slack — a Drude term and a lossless Lorentzian both sit at |z| = 1 exactly.

        Checked against MEEP's documented rule of thumb (Materials.md:75-77,
        "unstable for omega_n dt/2 > 1 ... f_n should be less than 1/(pi dt)"): at
        dt = 0.5/40, 1/(pi dt) = 25.4648, and this reports max|z| = 1.000000 at
        f0 = 25.210 and 1.326584 at f0 = 25.719.
        """
        first, second = self.poles(dt)
        magnitude = max(abs(first), abs(second))
        return StabilityReport(
            frequency=self.frequency,
            gamma=self.gamma,
            kind=self.kind,
            dt=float(dt),
            pole_magnitude=magnitude,
            stable=magnitude <= 1.0 + POLE_TOLERANCE,
            marginal=1.0 - POLE_TOLERANCE <= magnitude <= 1.0 + POLE_TOLERANCE,
        )

    def require_stable(self, dt: float) -> StabilityReport:  # Refuse a term the leapfrog cannot carry.
        """Raise :class:`DispersionInstability` when the discrete poles leave the unit circle.

        The three remedies are MEEP's (Materials.md:75) and are named in the message,
        because a caller who has just been refused needs to know that raising the
        resolution, lowering the Courant factor, or refitting the model are the
        options — not that the material is impossible.
        """
        report = self.stability(dt)
        if report.stable:
            return report
        limit = 1.0 / (math.pi * float(dt))
        raise DispersionInstability(
            f"{report.describe()}. The discretised polarization equation has a pole outside the "
            f"unit circle, so this run would grow without bound while staying finite and "
            f"plausible for thousands of steps. MEEP's rule of thumb is f_n < 1/(pi*dt) = "
            f"{limit:g} for this timestep (doc/docs/Materials.md:75-77). Remedies, in MEEP's "
            f"order: raise the resolution (dt and dx both shrink), lower the Courant factor (dt "
            f"alone shrinks), or refit the material with a lower-frequency model."
        )


def _ratio_or_raise(numerator: float, denominator: complex) -> complex:  # Divide, or say why not.
    """Divide, refusing a zero denominator instead of returning inf or nan.

    A susceptibility evaluated exactly at a pole of its own closed form (a lossless
    Lorentzian probed at its resonance) has no finite value. Returning inf would
    propagate into an epsilon volume and then into a field, where it becomes nan and
    is blamed on the stepping.
    """
    if denominator == 0:
        raise ValueError(
            "Susceptibility denominator is exactly zero — the closed form has a pole here "
            "(a lossless resonance probed exactly on resonance). Probe slightly off it, or "
            "give the term a nonzero gamma."
        )
    return numerator / denominator


def debye_chi1(delta_epsilon: float, tau: float, frequency: float) -> complex:
    """Continuum susceptibility of a Debye relaxation, ``delta_eps / (1 - i*omega*tau)``.

    MEEP's exp(-i*omega*t) convention, ``tau`` in MEEP time units and ``frequency``
    in MEEP frequency units, so omega = 2*pi*f.
    """
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"Debye relaxation time tau must be finite and positive, got {tau!r}.")
    return delta_epsilon / complex(1.0, -2.0 * math.pi * float(frequency) * tau)


def debye_to_lorentzian(
    delta_epsilon: float, tau: float, scale_frequency: float
) -> Tuple[Susceptibility, float]:
    """Map a Debye relaxation onto the overdamped Lorentzian that stands in for it.

    MEEP has NO Debye susceptibility — python/geom.py ships Lorentzian, Drude, their
    noisy variants, gyrotropic and multilevel-atom kinds and nothing else — and the
    engine's parity model requires that CPU MEEP can run every job this engine
    accepts. A native Debye kind would therefore be a GPU-only capability with no
    oracle, so Debye is expressed in terms MEEP already has::

        chi_lorentz(omega) = delta_eps / (1 - (omega/omega_0)**2 - i*omega*gamma/omega_0**2)
        chi_debye(omega)   = delta_eps / (1 - i*omega*tau)

    which agree when ``gamma_rad = tau * omega_0**2``, i.e. ``gamma_meep = 2*pi *
    scale_frequency**2 * tau``, with sigma = delta_eps. The mapping is exact only in
    the limit omega << omega_0; the residual is the retained ``-(omega/omega_0)**2``
    term, so choose ``scale_frequency`` well above the simulated band and report
    :func:`debye_mapping_error` at its edge rather than assuming it is small.

    Returns ``(susceptibility, sigma)``. The caller still has to pass ``sigma``
    through ``add_susceptibility``; it is returned rather than folded in because
    sigma is a per-cell volume everywhere else in this module.
    """
    if not math.isfinite(delta_epsilon) or delta_epsilon < 0.0:
        raise ValueError(
            f"Debye strength delta_epsilon must be finite and >= 0, got {delta_epsilon!r}; a "
            f"negative strength is gain."
        )
    scale = float(scale_frequency)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Debye scale_frequency must be finite and positive, got {scale_frequency!r}.")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"Debye relaxation time tau must be finite and positive, got {tau!r}.")
    gamma_meep = 2.0 * math.pi * scale * scale * float(tau)
    return Susceptibility(frequency=scale, gamma=gamma_meep, kind=LORENTZIAN), float(delta_epsilon)


def debye_mapping_error(
    delta_epsilon: float, tau: float, scale_frequency: float, frequency: float
) -> float:
    """Relative error of :func:`debye_to_lorentzian` at one frequency, ``|chi_L - chi_D|/|chi_D|``.

    Reported, never swallowed: the mapping is O((omega/omega_0)**2) and a caller who
    picks ``scale_frequency`` too close to the band gets a material that is smooth,
    causal and wrong.
    """
    susceptibility, sigma = debye_to_lorentzian(delta_epsilon, tau, scale_frequency)
    mapped = sigma * susceptibility.chi1(frequency)
    exact = debye_chi1(delta_epsilon, tau, frequency)
    if exact == 0:
        raise ValueError("The Debye susceptibility is zero here; a relative error is undefined.")
    return abs(mapped - exact) / abs(exact)


def component_coordinates(grid: "Grid", component: str) -> Tuple[Any, Any, Any]:
    """Per-axis coordinate vectors at ONE E component's own Yee positions.

    The registration rule, and the one place it is written down: axis d's stored
    index i sits at ``grid.axis_origin(d) + (i + shift_d/2) * dx``, where ``shift_d``
    is the component's ``iyee_shift`` (1 on its own axis, 0 on the others). That is
    the same integer-position convention ``FdtdDriver.set_epsilon`` documents, plus
    the half-cell offset on the component's own axis that MEEP applies when it
    point-samples sigma (anisotropic_averaging.cpp:333-345).

    Measuring from ``-L/2`` instead of ``grid.axis_origin(d)`` is the same half-cell
    error that cost 6.6e-2 on a grating's epsilon; it agrees only when that axis's
    cell count is even, which is why every validated case sweeps both parities.
    """
    if component not in E_YEE_SHIFTS:
        raise ValueError(
            f"sigma is sampled at an electric component's Yee position; expected one of "
            f"{E_COMPONENTS}, got {component!r}."
        )
    xp = grid.xp
    shifts = E_YEE_SHIFTS[component]
    counts = (grid.nx, grid.ny, grid.nz)
    return tuple(
        grid.axis_origin(axis) + (xp.arange(counts[axis], dtype=xp.float64) + 0.5 * shifts[axis]) * grid.dx
        for axis in range(3)
    )


def sample_region(grid: "Grid", predicate: Callable[[Any, Any, Any], Any], component: str) -> Any:
    """Sample a region indicator at one E component's Yee positions, as float32.

    ``predicate(x, y, z)`` receives three broadcast coordinate arrays and returns a
    boolean or float volume. Every caller that builds a sigma volume should go
    through this rather than re-deriving the offsets: the half-cell registration
    error it removes is the single defect class this engine has hit most often.
    """
    xp = grid.xp
    x, y, z = component_coordinates(grid, component)
    values = predicate(x[:, None, None], y[None, :, None], z[None, None, :])
    volume = xp.broadcast_to(xp.asarray(values), (grid.nx, grid.ny, grid.nz))
    return xp.ascontiguousarray(volume.astype(xp.float32))


def normalize_sigma(sigma: Any, grid: "Grid") -> Dict[str, Any]:
    """Resolve the three accepted sigma spellings into one dict keyed by E component.

    * a scalar — uniform, stored as a Python float. No array is allocated and the
      kernel multiplies by a scalar, so this form is exactly registration-free and
      is what every analytic test uses.
    * one array — replicated (aliased, not copied) to all three components at the
      SAME integer registration ``inv_eps`` uses, so a two-material geometry has one
      boundary description. It carries the known half-cell error at interfaces for
      the components whose Yee shift is 1 along the interface normal.
    * a mapping ``{'Ex': ..., 'Ey': ..., 'Ez': ...}`` — per component, each sampled
      at its own Yee position (:func:`sample_region`). This is the MEEP-exact form;
      missing keys are 0.0.

    Arrays accept this grid's shape or MEEP's ``(N+1)`` padded shape, trimmed the
    same way ``set_epsilon`` trims. Every value is checked finite and >= 0: a
    negative sigma is gain, and it would pass every magnitude comparison while
    growing.
    """
    xp = grid.xp
    if isinstance(sigma, Mapping):
        unknown = sorted(set(sigma) - set(E_COMPONENTS))
        if unknown:
            raise ValueError(
                f"sigma keys must be electric components {E_COMPONENTS}, got unexpected {unknown}. "
                f"Magnetic (H-side) susceptibilities are not implemented on this engine."
            )
        return {name: _coerce_sigma(sigma.get(name, 0.0), grid, name) for name in E_COMPONENTS}
    if _is_scalar(sigma):
        value = _coerce_sigma_scalar(sigma)
        return {name: value for name in E_COMPONENTS}
    shared = _coerce_sigma_array(sigma, grid, "sigma")
    del xp  # The shared array is aliased into all three slots, never copied per component.
    return {name: shared for name in E_COMPONENTS}


def _is_scalar(value: Any) -> bool:  # A Python/NumPy scalar rather than a volume.
    if isinstance(value, (bool,)):
        return False
    if isinstance(value, (int, float)):
        return True
    shape = getattr(value, "shape", None)
    return shape == ()


def _coerce_sigma(value: Any, grid: "Grid", component: str) -> Any:  # One component's sigma.
    if _is_scalar(value):
        return _coerce_sigma_scalar(value)
    return _coerce_sigma_array(value, grid, f"sigma[{component!r}]")


def _coerce_sigma_scalar(value: Any) -> float:  # Validate a uniform sigma.
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"sigma must be finite and >= 0, got {value!r}. A negative sigma is gain: it flips the "
            f"sign of Im eps and grows the field while every magnitude still looks reasonable."
        )
    return number


def _coerce_sigma_array(value: Any, grid: "Grid", label: str) -> Any:  # Validate and trim a sigma volume.
    xp = grid.xp
    values = xp.array(value, dtype=xp.float32, order="C", copy=True)
    if values.ndim != 3:
        raise ValueError(f"{label} must be a 3-D array or a scalar, got {values.ndim} dimensions.")
    grid_shape = (grid.nx, grid.ny, grid.nz)
    meep_shape = tuple(n + 1 for n in grid_shape)
    shape = tuple(int(n) for n in values.shape)
    if shape == grid_shape:
        trimmed = values
    elif shape == meep_shape:
        trimmed = values[:-1, :-1, :-1]  # Same trim set_epsilon applies to MEEP's padded arrays.
    else:
        raise ValueError(
            f"{label} shape {shape} matches neither the grid shape {grid_shape} nor MEEP's "
            f"boundary-padded shape {meep_shape}."
        )
    if not bool(xp.all(xp.isfinite(trimmed))) or bool(xp.any(trimmed < 0.0)):
        raise ValueError(
            f"{label} must be finite and >= 0 in every cell. A negative sigma is gain and would "
            f"grow the field while every magnitude still looked reasonable."
        )
    return xp.ascontiguousarray(trimmed)


def sigma_is_trivial(value: Any, xp: Any) -> bool:  # MEEP's trivial_sigma: identically zero.
    """True when this sigma drives nothing anywhere.

    MEEP allocates no P array for a component whose sigma is identically zero
    (susceptibility.cpp ``trivial_sigma`` feeding ``needs_P``), and skipping it is
    EXACT rather than an approximation: the recurrence is homogeneous and starts at
    zero, so such a P is provably 0.0 in every cell for every step. That exactness
    is what makes the zero-strength degenerate case reduce byte-for-byte to the
    non-dispersive path.
    """
    if _is_scalar(value):
        return float(value) == 0.0
    return not bool(xp.any(value != 0.0))


class PolarizationState:
    """Storage and timestepping for ONE susceptibility on ONE grid.

    Translation of ``meep::lorentzian_data`` (susceptibility.cpp:98-104) plus the
    per-component sigma volumes ``meep::susceptibility`` owns (meep.hpp:79-100).

    The update is element-wise in space — no stencil, no neighbour read — which is
    why P needs no boundary pass of any kind: no Bloch phase, no mirror ghost, no
    ownership mask. Everything a boundary does to the polarization arrives through
    the drive field W, which already carries it. (MEEP's ``OFFDIAG`` neighbour reads
    and its ``if (s[i] != 0)`` guard live only in the anisotropic branches, which
    this engine does not implement; copying either into an isotropic kernel would
    freeze P_prev in sigma = 0 cells.)
    """

    def __init__(self, susceptibility: Susceptibility, sigma: Any, grid: "Grid", dtype: Any):
        """Allocate P/P_prev for the components this term actually drives.

        The susceptibility's stability at this grid's dt is NOT checked here — the
        driver checks it before constructing, so the refusal names the run's
        configuration rather than an array shape.
        """
        self.susceptibility = susceptibility
        self.grid = grid
        self.sigma: Dict[str, Any] = normalize_sigma(sigma, grid)
        xp = grid.xp
        self._driven: Tuple[str, ...] = tuple(
            name for name in E_COMPONENTS if not sigma_is_trivial(self.sigma[name], xp)
        )
        self._coefficients = susceptibility.coefficients(grid.dt)
        shape = grid.shape
        self.P: Dict[str, Any] = {name: xp.zeros(shape, dtype=dtype) for name in self._driven}
        self.P_prev: Dict[str, Any] = {name: xp.zeros(shape, dtype=dtype) for name in self._driven}
        # One shared scratch for the whole term: the three-way rotation below moves
        # it from component to component, so a second buffer is never live at once.
        self._scratch = xp.zeros(shape, dtype=dtype) if self._driven else None

    def driven(self) -> Tuple[str, ...]:  # MEEP needs_P + trivial_sigma: the components carrying a P.
        return self._driven

    def drives(self, component: str) -> bool:  # Does this term contribute to that component's D - P?
        return component in self._driven

    def update(self, drive: Callable[[str], Any], dt: float) -> None:
        """Advance every driven component one step — MEEP susceptibility.cpp:251-258.

        ``drive(component)`` must return W = (D - sum P) * inv_eps for that
        component, NOT the stored E: the two differ inside a PML, where the stored E
        has already had the absorbing accumulation applied. MEEP takes the same
        distinction at update_pols.cpp:44 (``w = f_w ? f_w : f``).

        The buffer rotation costs zero copies: this step's result is written into the
        scratch buffer, the scratch slot then takes over the array that held P_prev,
        and P/P_prev shift down. Every buffer is owned by exactly one slot at a time,
        so no two names ever alias.
        """
        if not self._driven:
            return
        if dt != self.grid.dt:
            raise ValueError(
                f"PolarizationState coefficients were built for dt={self.grid.dt!r} and cannot be "
                f"stepped at dt={dt!r}; a susceptibility's recurrence constants are fixed by the "
                f"timestep."
            )
        xp = self.grid.xp
        c_now, c_prev, c_drive = self._coefficients
        for component in self._driven:
            w = drive(component)
            p = self.P[component]
            p_prev = self.P_prev[component]
            scratch = self._scratch
            xp.multiply(p, c_now, out=scratch)  # c_now * P^n
            scratch += c_prev * p_prev  # - (1-g2)/(1+g2) * P^(n-1)
            scratch += c_drive * (self.sigma[component] * w)  # + w2/(1+g2) * sigma * W^n
            self.P[component] = scratch
            self.P_prev[component] = p
            self._scratch = p_prev  # The retired history buffer becomes next component's scratch.

    def subtract_into(self, component: str, target: Any) -> None:  # MEEP subtract_P: D -> D - P.
        if component in self.P:
            target -= self.P[component]

    def reset(self) -> None:  # Forget the polarization history, for a driver restarting its clock.
        """Zero P and P_prev.

        P_prev matters as much as P: the recurrence is second order, so a re-run that
        kept a stale P_prev would start with a history it never lived through and
        produce a wrong but entirely plausible field — the same defect class the
        integrated-source reset documents.
        """
        for arrays in (self.P, self.P_prev):
            for array in arrays.values():
                array.fill(0)

    def bytes_per_cell(self, element_bytes: int) -> int:  # Memory this term costs, for the budget.
        """P + P_prev for each driven component, plus its share of the shared scratch.

        sigma is counted only when it is an array, and a shared array is counted once
        even though three components reference it — which is what it costs.
        """
        total = 2 * len(self._driven) * element_bytes
        if self._driven:
            total += element_bytes  # The one scratch buffer.
        counted: list[int] = []
        for name in E_COMPONENTS:
            value = self.sigma[name]
            if _is_scalar(value):
                continue
            if any(value is seen for seen in counted):
                continue
            counted.append(value)
            total += 4  # float32 sigma volume.
        return total

    def __repr__(self) -> str:
        uniform = [name for name in E_COMPONENTS if _is_scalar(self.sigma[name])]
        form = "uniform" if len(uniform) == 3 else "volume"
        return (
            f"PolarizationState({self.susceptibility.kind}, f0={self.susceptibility.frequency:g}, "
            f"gamma={self.susceptibility.gamma:g}, sigma={form}, driven={self._driven})"
        )


def summed_chi1(
    susceptibilities: Sequence[Tuple[Susceptibility, float]], frequency: float
) -> complex:  # sum_n sigma_n * chi1_n(f).
    """Total continuum susceptibility of a list of (term, uniform sigma) pairs.

    The one place the sum is written, so ``get_epsilon(frequency=...)`` and any
    analytic target computed in a test cannot drift apart.
    """
    return sum(
        (sigma * term.chi1(frequency) for term, sigma in susceptibilities), start=complex(0.0, 0.0)
    )


def conductivity_factor(d_conductivity: float, frequency: float) -> complex:
    """MEEP's ``1 + i*sigma_D/omega`` multiplier on the WHOLE permittivity.

    python/geom.py ``_get_epsmu``: ``epsmu = (1 + 1j/(2*pi*f) * conductivity) *
    epsmu`` — it multiplies eps_infinity AND every susceptibility, it is not an
    additive imaginary part. doc/docs/Materials.md states the same as
    ``Im eps = eps_inf * sigma_D / omega``, and the worked example there
    (eps = 3.4 + 0.101i at f = 0.42 needing sigma_D = 2*pi*0.42*0.101/3.4) is the
    numeric check on this convention.
    """
    probe = float(frequency)
    if not math.isfinite(probe) or probe <= 0.0:
        raise ValueError(
            f"A conductivity's contribution to eps diverges as 1/omega, so it has no value at "
            f"f = 0; got {frequency!r}."
        )
    if d_conductivity == 0.0:
        return complex(1.0, 0.0)  # Exactly 1: a zero conductivity must not tint eps complex.
    return complex(1.0, d_conductivity / (2.0 * math.pi * probe))
