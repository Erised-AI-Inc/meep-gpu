"""Check dispersion closed forms against MEEP's susceptibility recurrence."""

from __future__ import annotations

import cmath
import math


def meep_coeffs(
    f0: float,
    gamma_meep: float,
    dt: float,
    *,
    drude: bool = False,
) -> tuple[float, float, float]:
    """Return the scalar coefficients from MEEP's susceptibility update."""
    omega2pi = 2 * math.pi * f0
    gamma2pi = 2 * math.pi * gamma_meep
    omega0_dt_squared = omega2pi * omega2pi * dt * dt
    gamma_inverse = 1 / (1 + gamma2pi * dt / 2)
    gamma_factor = 1 - gamma2pi * dt / 2
    denominator = 0.0 if drude else omega0_dt_squared
    current = gamma_inverse * (2 - denominator)
    previous = -gamma_inverse * gamma_factor
    drive = gamma_inverse * omega0_dt_squared
    return current, previous, drive


def chi_discrete(
    f: float,
    f0: float,
    gamma_meep: float,
    dt: float,
    *,
    sigma: float = 1.0,
    drude: bool = False,
) -> complex:
    """Return the exact transfer function of the discrete recurrence."""
    omega = 2 * math.pi * f
    omega0 = 2 * math.pi * f0
    gamma = 2 * math.pi * gamma_meep
    omega0_dt_squared = (omega0 * dt) ** 2
    restoring_term = 0.0 if drude else omega0_dt_squared
    denominator = (
        restoring_term
        - 4 * math.sin(omega * dt / 2) ** 2
        - 1j * (gamma * dt) * math.sin(omega * dt)
    )
    return sigma * omega0_dt_squared / denominator


def chi_continuum(
    f: float,
    f0: float,
    gamma_meep: float,
    *,
    sigma: float = 1.0,
    drude: bool = False,
) -> complex:
    """Return MEEP's continuum Lorentz or Drude susceptibility."""
    if drude:
        return sigma * f0 * f0 / complex(-f * f, -gamma_meep * f)
    return sigma * f0 * f0 / complex(
        f0 * f0 - f * f,
        -gamma_meep * f,
    )


def simulate(
    f: float,
    f0: float,
    gamma_meep: float,
    dt: float,
    *,
    sigma: float = 1.0,
    drude: bool = False,
    nsteps: int = 60_000,
    ramp_steps: int = 0,
) -> complex:
    """Simulate the recurrence and return its steady-state response ratio."""
    current, previous, drive = meep_coeffs(
        f0,
        gamma_meep,
        dt,
        drude=drude,
    )
    polarization = 0j
    prior_polarization = 0j
    omega = 2 * math.pi * f

    for step in range(nsteps):
        envelope = 1.0
        if ramp_steps:
            ramp_fraction = min(step, ramp_steps) / ramp_steps
            envelope = 0.5 * (1 - math.cos(math.pi * ramp_fraction))
        field = envelope * cmath.exp(-1j * omega * step * dt)
        polarization, prior_polarization = (
            current * polarization
            + previous * prior_polarization
            + drive * sigma * field,
            polarization,
        )

    field = cmath.exp(-1j * omega * nsteps * dt)
    return polarization / field


def poles(
    f0: float,
    gamma_meep: float,
    dt: float,
    *,
    drude: bool = False,
) -> tuple[complex, complex]:
    """Return the two roots of the homogeneous update polynomial."""
    gamma_dt = 2 * math.pi * gamma_meep * dt / 2
    omega0_dt_squared = (2 * math.pi * f0 * dt) ** 2
    restoring_term = 0.0 if drude else omega0_dt_squared
    a = 1 + gamma_dt
    b = -(2 - restoring_term)
    c = 1 - gamma_dt
    discriminant = cmath.sqrt(b * b - 4 * a * c)
    return (
        (-b + discriminant) / (2 * a),
        (-b - discriminant) / (2 * a),
    )


def meep_unstable(f0: float, gamma_meep: float, dt: float) -> bool:
    """Evaluate MEEP's Lorentz stability predicate."""
    omega = 2 * math.pi * f0
    gamma = 2 * math.pi * gamma_meep
    gamma_dt = gamma * dt / 2
    omega0_dt_squared = (omega * dt) ** 2
    b = (1 - omega0_dt_squared / 2) / (1 + gamma_dt)
    c = (1 - gamma_dt) / (1 + gamma_dt)
    return (
        b * b > c
        and 2 * b * b - c + 2 * abs(b) * math.sqrt(b * b - c) > 1
    )


def verify_transfer_functions(dt: float) -> None:
    """Verify simulated, discrete, and continuum susceptibilities."""
    print("=== discrete chi: simulated recurrence vs closed form ===")
    for drude in (False, True):
        recurrence_errors: list[float] = []
        for f in (0.3, 0.9, 1.0, 1.1, 2.0):
            gamma_meep = 0.05 if drude else 0.1
            simulated = simulate(
                f,
                1.0,
                gamma_meep,
                dt,
                drude=drude,
                ramp_steps=20_000 if drude else 0,
            )
            discrete = chi_discrete(
                f,
                1.0,
                gamma_meep,
                dt,
                drude=drude,
            )
            continuum = chi_continuum(
                f,
                1.0,
                gamma_meep,
                drude=drude,
            )
            recurrence_error = abs(simulated - discrete) / abs(discrete)
            discretization_error = abs(discrete - continuum) / abs(continuum)
            recurrence_errors.append(recurrence_error)
            print(
                f"  drude={drude} f={f:<5} |chi|={abs(discrete):8.4f}  "
                f"recurrence_error={recurrence_error:.2e}  "
                f"discretization_error={discretization_error:.2e}"
            )

        tolerance = 5e-4 if drude else 5e-12
        assert max(recurrence_errors) < tolerance


def verify_second_order_convergence() -> None:
    """Verify that the discrete susceptibility approaches the continuum at O(dt^2)."""
    print("\n=== convergence of discrete -> continuum (should be O(dt^2)) ===")
    errors: list[float] = []
    for resolution in (20, 40, 80, 160):
        dt = 0.5 / resolution
        discrete = chi_discrete(0.7, 1.0, 0.1, dt)
        continuum = chi_continuum(0.7, 1.0, 0.1)
        relative_error = abs(discrete - continuum) / abs(continuum)
        errors.append(relative_error)
        print(f"  res={resolution:<4} rel diff={relative_error:.3e}")

    refinement_ratios = [
        coarse / fine
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    ]
    assert all(3.9 < ratio < 4.1 for ratio in refinement_ratios)


def verify_poles_and_stability(dt: float) -> None:
    """Verify the Drude poles and MEEP stability boundary."""
    print("\n=== poles / stability ===")
    drude_poles = poles(1.0, 0.05, dt, drude=True)
    gamma_dt = 2 * math.pi * 0.05 * dt / 2
    predicted_second_pole = (1 - gamma_dt) / (1 + gamma_dt)
    print(
        "  Drude poles (expect exactly 1 and (1-g2)/(1+g2)):",
        [f"{abs(pole):.12f}" for pole in drude_poles],
    )
    print("  predicted second Drude pole:", f"{predicted_second_pole:.12f}")
    assert min(abs(pole - 1) for pole in drude_poles) < 1e-12
    assert min(abs(pole - predicted_second_pole) for pole in drude_poles) < 1e-12

    boundary = 1 / (math.pi * dt)
    below = boundary * 0.99
    above = boundary * 1.01
    print("\n  lossless Lorentz stability boundary f0 = 1/(pi*dt) =", boundary)
    for f0 in (below, above):
        roots = poles(f0, 0.0, dt)
        unstable = meep_unstable(f0, 0.0, dt)
        print(
            f"    f0={f0:8.3f}  |z|max={max(abs(root) for root in roots):.6f}  "
            f"meep_unstable={unstable}"
        )
    assert not meep_unstable(below, 0.0, dt)
    assert meep_unstable(above, 0.0, dt)


def main() -> None:
    """Run the standalone closed-form verification suite."""
    dt = 0.5 / 40
    verify_transfer_functions(dt)
    verify_second_order_convergence()
    verify_poles_and_stability(dt)


if __name__ == "__main__":
    main()
