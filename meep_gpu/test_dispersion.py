"""
Value tests for the dispersive-material model, ADE kernel, and the
physics the two of them together are supposed to produce.

The file is layered deliberately, tightest first, because the loose layers cannot
catch what the tight ones can:

1. **Pure functions, no arrays.** ``chi1`` against MEEP's own expression,
   ``chi1_discrete`` against the recurrence it claims to solve, the poles, and
   MEEP's documented stability limit. Every units and 2*pi error dies here, before
   a single field exists.
2. **The kernel.** ``PolarizationState`` against the scalar recurrence, its buffer
   rotation, its component skipping, and its lack of any spatial stencil.
3. **The physics.** A measured phase velocity against the EXACT discrete
   dispersion relation (which carries no discretisation error, so the bound is
   1e-4 rather than a physics tolerance), then the same measurement against the
   CONTINUUM n(omega) as a CONVERGENCE test — the error must fall four-fold per
   halving of dx, which is the only honest way to compare a second-order scheme
   against a closed form. A Drude half-space against the Fresnel reflectance and
   the analytic skin depth, both asserted, because checking only one lets a factor
   of two through while still producing a beautiful exponential.
4. **The degenerate and refused cases.** A zero-strength susceptibility must
   reduce BYTE-IDENTICALLY to the non-dispersive engine; an unstable Lorentzian
   must be refused at setup; a run that diverges anyway must raise rather than
   return the large, smooth, finite, entirely wrong field that is this feature's
   real failure mode.

Every comparison against CPU MEEP lives in ``test_driver_vs_meep.py``; nothing
here imports meep.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from .dispersion import (
    DRUDE,
    E_COMPONENTS,
    LORENTZIAN,
    DispersionInstability,
    PolarizationState,
    StabilityReport,
    Susceptibility,
    component_coordinates,
    conductivity_factor,
    debye_chi1,
    debye_mapping_error,
    debye_to_lorentzian,
    normalize_sigma,
    sample_region,
    sigma_is_trivial,
    summed_chi1,
)
from .driver import FdtdDivergence, FdtdDriver
from .grid import Grid


def _scalar_recurrence(susceptibility, sigma, frequency, dt, steps):
    """Drive MEEP's recurrence with W^n = exp(-i*2*pi*f*n*dt), in float64.

    Returns ``(P^steps, W^steps)``. This is the reference the array kernel is held
    against, and the thing ``chi1_discrete`` claims to be the transfer function of.
    """
    c_now, c_prev, c_drive = susceptibility.coefficients(dt)
    omega = 2.0 * math.pi * frequency
    polarization, previous = 0j, 0j
    for step in range(steps):
        drive = cmath.exp(-1j * omega * step * dt)
        polarization, previous = (
            c_now * polarization + c_prev * previous + c_drive * (sigma * drive),
            polarization,
        )
    return polarization, cmath.exp(-1j * omega * steps * dt)


def _vacuum_wavenumber(frequency, dt, dx):  # Numerical k of the empty Yee lattice.
    return 2.0 / dx * math.asin(dx * math.sin(2.0 * math.pi * frequency * dt / 2.0) / dt)


def _discrete_wavenumber(epsilon_tilde, frequency, dt, dx):
    """k(omega) of the 1-D numerical dispersion relation in a homogeneous medium.

        sin(k dx/2)/dx = sqrt(eps_tilde) sin(omega dt/2)/dt

    with ``eps_tilde = eps_inf + sum sigma*chi1_discrete``. Exact for the scheme —
    no discretisation error — which is what lets a measurement against it be held
    at 1e-4 instead of at a physics tolerance. The branch with ``Im k >= 0`` is the
    decaying one.
    """
    wavenumber = 2.0 / dx * cmath.asin(
        dx * cmath.sqrt(epsilon_tilde) * math.sin(2.0 * math.pi * frequency * dt / 2.0) / dt
    )
    return wavenumber if wavenumber.imag >= 0 else -wavenumber


def _continuum_index(epsilon):  # sqrt(eps) on the passive branch, Im n >= 0.
    index = cmath.sqrt(epsilon)
    return index if index.imag >= 0 else -index


def _uniform_driver(cell, resolution, epsilon, terms, courant=0.5, pml=None, **kwargs):
    driver = FdtdDriver(cell_size=cell, resolution=resolution, courant=courant,
                        force_complex_fields=True, **kwargs)
    driver.set_epsilon(np.full(driver.shape, epsilon, dtype=np.float32))
    for term, sigma in terms:
        driver.add_susceptibility(term, sigma)
    if pml is not None:
        driver.setup_pml(pml)
    return driver


def _propagation_probe(terms, epsilon, frequencies, resolution, monitor_z, gap_cells=3,
                       until=60.0, fcen=1.1, fwidth=0.9):
    """Measure exp(i k dz) between two point monitors in a homogeneous dispersive medium.

    The measured quantity is the COMPLEX RATIO of the two DFT samples, which is
    ``exp(i k dz)`` for a forward-propagating wave and needs no phase unwrapping at
    all — the branch ambiguity that would otherwise make this test's answer depend
    on the model it is testing never arises. ``n`` is recovered afterwards, for
    reporting, by taking the branch nearest the analytic one.
    """
    driver = _uniform_driver((0.2, 0.2, 8.0), resolution, epsilon, terms, pml={"z": resolution})
    driver.add_source({"component": "Ex", "source_type": "gaussian", "frequency": fcen,
                       "fwidth": fwidth, "center": (0.0, 0.0, -2.5), "size": (0.2, 0.2, 0.0)})
    separation = gap_cells * driver.dx
    near = driver.add_dft_monitor(frequencies=frequencies, components=("Ex",),
                                  center=(0.0, 0.0, monitor_z), size=(0.0, 0.0, 0.0))
    far = driver.add_dft_monitor(frequencies=frequencies, components=("Ex",),
                                 center=(0.0, 0.0, monitor_z + separation), size=(0.0, 0.0, 0.0))
    driver.run(until=until)
    measured = []
    for index, frequency in enumerate(frequencies):
        first = complex(np.asarray(near.get_dft("Ex", index)).ravel()[0])
        second = complex(np.asarray(far.get_dft("Ex", index)).ravel()[0])
        measured.append((frequency, second / first, abs(first)))
    dt, dx = driver.dt, driver.dx
    driver.close()
    return measured, separation, dt, dx


# ---------------------------------------------------------------------------
# Layer 1 — the closed forms, with no arrays anywhere
# ---------------------------------------------------------------------------


def test_chi1_reproduces_meeps_own_expression():  # susceptibility.cpp:297-307, transcribed independently.
    lorentz = Susceptibility(1.1, 0.05, LORENTZIAN)
    drude = Susceptibility(1.0, 0.05, DRUDE)
    for frequency in (0.3, 0.9, 1.0, 1.4, 2.0):
        f0, gamma = lorentz.frequency, lorentz.gamma
        expected = f0 * f0 / complex(f0 * f0 - frequency * frequency, -gamma * frequency)
        assert lorentz.chi1(frequency) == pytest.approx(expected, rel=1e-12)
        # MEEP's Drude form drops the omega_0^2 term from the DENOMINATOR only; the
        # numerator keeps the full sigma*omega_0^2 (python/geom.py DrudeSusceptibility).
        f0 = drude.frequency
        expected = f0 * f0 / complex(-frequency * frequency, -drude.gamma * frequency)
        assert drude.chi1(frequency) == pytest.approx(expected, rel=1e-12)


def test_a_lorentzian_is_passive_at_every_positive_frequency():  # Im eps >= 0: causality, and gamma's sign.
    for kind in (LORENTZIAN, DRUDE):
        term = Susceptibility(1.0, 0.1, kind)
        for frequency in np.linspace(0.05, 5.0, 60):
            assert term.chi1(float(frequency)).imag >= 0.0, (
                f"{kind} chi1 has a negative imaginary part at f={frequency}, which is gain."
            )


def test_chi1_discrete_converges_to_chi1_at_second_order():
    # The convergence RATE is the assertion, not the value: a chi1_discrete silently
    # implemented as chi1 would pass any fixed bound at high resolution and hide the
    # whole point of having a separate discrete form.
    term = Susceptibility(1.0, 0.1, LORENTZIAN)
    errors = []
    for resolution in (20, 40, 80, 160):
        dt = 0.5 / resolution
        exact = term.chi1(0.7)
        errors.append(abs(term.chi1_discrete(0.7, dt) - exact) / abs(exact))
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.02), (
            f"chi1_discrete -> chi1 convergence is {coarse / fine:.3f}x per halving of dt, not 4x; "
            f"errors {errors}"
        )
    assert errors[0] == pytest.approx(9.961e-4, rel=1e-3)


def test_chi1_discrete_differs_from_chi1_most_on_resonance():
    # 5.3e-3 at resolution 40 BEFORE any Maxwell discretisation is added. This is why
    # the primary propagation test targets the discrete relation and the continuum
    # comparison is run as a convergence test instead of at a fixed bound.
    term = Susceptibility(1.0, 0.1, LORENTZIAN)
    dt = 0.5 / 40
    on_resonance = abs(term.chi1_discrete(1.0, dt) - term.chi1(1.0)) / abs(term.chi1(1.0))
    far_below = abs(term.chi1_discrete(0.3, dt) - term.chi1(0.3)) / abs(term.chi1(0.3))
    assert on_resonance == pytest.approx(5.25e-3, rel=0.02)
    assert far_below < 1e-5
    assert on_resonance > 100 * far_below


@pytest.mark.parametrize("frequency", [0.3, 0.9, 1.0, 1.1, 2.0])
def test_the_recurrence_transfer_function_is_chi1_discrete(frequency):
    # A damped Lorentzian's transient decays, so P/W converges and the ratio form is
    # available. 60 000 steps at gamma = 0.1 leaves a 5e-13 residual in float64.
    term = Susceptibility(1.0, 0.1, LORENTZIAN)
    dt = 0.5 / 40
    polarization, drive = _scalar_recurrence(term, 1.0, frequency, dt, 60000)
    expected = term.chi1_discrete(frequency, dt)
    assert abs(polarization / drive - expected) / abs(expected) < 1e-9


@pytest.mark.parametrize("frequency", [0.3, 0.9, 1.0, 2.0])
def test_a_drude_terms_residual_against_chi1_discrete_is_a_constant(frequency):
    """The z = 1 pole is physics, and this is the assertion shape it forces.

    A Drude recurrence has roots exactly z = 1 and z = (1-g2)/(1+g2). The first is
    the free-electron DC mode: an abruptly started drive excites it and it never
    decays, so ``P/W`` never converges and the ratio form of the previous test FAILS
    ON CORRECT CODE. What is true is that ``P^n - chi_tilde*W^n`` is a CONSTANT, and
    asserting that pins both the transfer function and the pole structure at once —
    it also fails if ``c_now`` carries the Lorentz restoring term instead of zero.
    """
    term = Susceptibility(1.0, 0.05, DRUDE)
    dt = 0.5 / 40
    chi = term.chi1_discrete(frequency, dt)
    residuals = []
    for steps in (20000, 60000):
        polarization, drive = _scalar_recurrence(term, 1.0, frequency, dt, steps)
        residuals.append(polarization - chi * drive)
    drift = abs(residuals[1] - residuals[0]) / abs(residuals[0])
    assert drift < 1e-7, f"Drude residual drifted by {drift:.2e}; it should be constant."
    assert residuals[0].real == pytest.approx(math.pi / 4, rel=1e-6)


def test_a_lossless_lorentzians_transient_neither_decays_nor_grows():
    """gamma = 0 puts BOTH poles on the unit circle, so the ratio form fails here too.

    The Sellmeier case MEEP's docs recommend (Materials.md:20-30) is marginally
    stable, not unstable: its transient is an undamped oscillation at the resonance,
    so ``P - chi_tilde*W`` neither settles to zero (as a damped Lorentzian's does)
    nor to a constant (as a Drude term's DC mode does) — it circles at fixed
    amplitude. That amplitude is what can be asserted, and asserting it catches both
    a decaying pole (which would mean gamma leaked in) and a growing one.
    """
    term = Susceptibility(0.5, 0.0, LORENTZIAN)
    dt = 0.5 / 40
    chi = term.chi1_discrete(0.3, dt)
    magnitudes = []
    for steps in (20000, 40000, 60000):
        polarization, drive = _scalar_recurrence(term, 1.0, 0.3, dt, steps)
        magnitudes.append(abs(polarization - chi * drive))
    for magnitude in magnitudes:
        assert magnitude == pytest.approx(magnitudes[0], rel=1e-2), (
            f"a lossless transient must hold its amplitude; measured {magnitudes}"
        )
    assert magnitudes[-1] <= magnitudes[0], "a lossless transient must not grow"
    report = term.stability(dt)
    assert report.marginal and report.stable
    assert report.pole_magnitude == pytest.approx(1.0, abs=1e-12)


def test_a_drude_terms_poles_are_exactly_one_and_the_damped_root():
    term = Susceptibility(1.0, 0.05, DRUDE)
    dt = 0.5 / 40
    poles = sorted(term.poles(dt), key=abs, reverse=True)
    half_gamma_dt = 2.0 * math.pi * term.gamma * dt / 2.0
    assert poles[0] == pytest.approx(1.0, abs=1e-12)
    assert poles[1] == pytest.approx((1.0 - half_gamma_dt) / (1.0 + half_gamma_dt), rel=1e-12)
    report = term.stability(dt)
    assert report.stable and report.marginal
    assert report.pole_magnitude == pytest.approx(1.0, abs=1e-9)


def test_stability_matches_meeps_documented_frequency_limit():
    """doc/docs/Materials.md:75-77 — a lossless term is unstable above f0 = 1/(pi*dt).

    Reproducing the documented rule from the transcribed algebra is what says the
    transcription is right; the two were derived independently (MEEP's predicate is
    the disabled ``lorentzian_unstable``, the doc's rule is ``omega dt/2 > 1``).
    """
    dt = 0.5 / 40
    limit = 1.0 / (math.pi * dt)
    assert limit == pytest.approx(25.4648, rel=1e-4)
    assert Susceptibility(limit * 0.99, 0.0, LORENTZIAN).stability(dt).stable
    unstable = Susceptibility(limit * 1.01, 0.0, LORENTZIAN).stability(dt)
    assert not unstable.stable
    assert unstable.pole_magnitude == pytest.approx(1.3263, rel=1e-3)
    assert unstable.growth_per_1000_steps > 1e100


def test_require_stable_raises_and_names_the_remedies():
    dt = 0.5 / 40
    term = Susceptibility(1.0 / (math.pi * dt) * 1.2, 0.0, LORENTZIAN)
    with pytest.raises(DispersionInstability) as raised:
        term.require_stable(dt)
    message = str(raised.value)
    assert "max|z|" in message and "resolution" in message and "Courant" in message
    assert f"{1.0 / (math.pi * dt):g}" in message


def test_gain_and_unsupported_kinds_are_refused():
    with pytest.raises(ValueError, match="gain"):
        Susceptibility(1.0, -0.01, LORENTZIAN)
    with pytest.raises(ValueError, match="kind"):
        Susceptibility(1.0, 0.1, "noisy_lorentzian")
    with pytest.raises(ValueError, match="strictly positive"):
        Susceptibility(0.0, 0.1, LORENTZIAN)


def test_the_recurrence_coefficients_carry_the_angular_frequency():
    """A dropped 2*pi is invisible to chi1 and fatal to the timestepping.

    ``chi1`` is a ratio of two omega^2 quantities, so the factor cancels and the
    function is right in either unit system; ``coefficients`` carries a bare dt and
    is not. The three constants are therefore pinned against the angular closed form
    directly, which is the only place the convention is observable in isolation.
    """
    term = Susceptibility(1.1, 0.05, LORENTZIAN)
    dt = 0.5 / 40
    omega_dt_squared = (2.0 * math.pi * 1.1 * dt) ** 2
    half_gamma_dt = 2.0 * math.pi * 0.05 * dt / 2.0
    c_now, c_prev, c_drive = term.coefficients(dt)
    assert c_now == pytest.approx((2.0 - omega_dt_squared) / (1.0 + half_gamma_dt), rel=1e-14)
    assert c_prev == pytest.approx(-(1.0 - half_gamma_dt) / (1.0 + half_gamma_dt), rel=1e-14)
    assert c_drive == pytest.approx(omega_dt_squared / (1.0 + half_gamma_dt), rel=1e-14)
    # Dropping the 2*pi from omega alone would move the resonance by 6.28x.
    assert c_drive / (1.1 * dt) ** 2 == pytest.approx((2 * math.pi) ** 2 / (1 + half_gamma_dt), rel=1e-12)


def test_drude_zeroes_the_restoring_force_and_not_the_drive():
    """MEEP's omega0dtsqr_denom changes c_now ONLY.

    Applying the flag to ``c_drive`` instead gives P == 0 everywhere — a metal that
    transmits like vacuum, which scores as a pristine transmission spectrum rather
    than as a failure. The two coefficients are therefore compared term by term
    against the Lorentzian with the same parameters.
    """
    dt = 0.5 / 40
    lorentz = Susceptibility(1.0, 0.05, LORENTZIAN).coefficients(dt)
    drude = Susceptibility(1.0, 0.05, DRUDE).coefficients(dt)
    assert drude[2] == pytest.approx(lorentz[2], rel=1e-14), "Drude must keep the full drive term."
    assert drude[1] == pytest.approx(lorentz[1], rel=1e-14), "Drude must keep c_prev unchanged."
    assert drude[0] != pytest.approx(lorentz[0], rel=1e-9), "Drude must drop the restoring force."
    half_gamma_dt = 2.0 * math.pi * 0.05 * dt / 2.0
    assert drude[0] == pytest.approx(2.0 / (1.0 + half_gamma_dt), rel=1e-14)


def test_conductivity_factor_matches_meeps_worked_example():
    # doc/docs/Materials.md: eps = 3.4 + 0.101i at f = 0.42 needs
    # D_conductivity = 2*pi*0.42*0.101/3.4, and eps(f) = eps_inf * (1 + i sigma_D/omega).
    sigma_d = 2.0 * math.pi * 0.42 * 0.101 / 3.4
    epsilon = 3.4 * conductivity_factor(sigma_d, 0.42)
    assert epsilon.real == pytest.approx(3.4, rel=1e-12)
    assert epsilon.imag == pytest.approx(0.101, rel=1e-12)
    assert conductivity_factor(0.0, 0.42) == 1.0 + 0.0j


def test_summed_chi1_adds_every_term():
    terms = [(Susceptibility(1.0, 0.05, DRUDE), 0.8), (Susceptibility(1.4, 0.1, LORENTZIAN), 0.5)]
    total = summed_chi1(terms, 0.9)
    assert total == pytest.approx(sum(sigma * term.chi1(0.9) for term, sigma in terms), rel=1e-12)
    assert total != pytest.approx(terms[0][1] * terms[0][0].chi1(0.9), rel=1e-3)


def test_debye_maps_onto_an_overdamped_lorentzian_with_a_reported_error():
    """The Debye stand-in is exact only for omega << omega_0, and says by how much.

    MEEP has no Debye susceptibility, so a native kind would be a GPU-only capability
    with no CPU oracle. The mapping's residual is the retained -(omega/omega_0)^2
    term, so the error must fall as 1/scale^2; asserting that rate is what stops the
    converter quietly shipping a material that is smooth, causal and wrong.
    """
    delta_epsilon, tau, band_edge = 2.0, 0.3, 0.5
    errors = [debye_mapping_error(delta_epsilon, tau, scale, band_edge) for scale in (5.0, 10.0, 20.0)]
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.1)
    assert errors[-1] < 2e-3
    term, sigma = debye_to_lorentzian(delta_epsilon, tau, 20.0)
    assert sigma == delta_epsilon
    assert term.gamma == pytest.approx(2.0 * math.pi * 20.0**2 * tau, rel=1e-12)
    assert term.kind == LORENTZIAN
    # And the Debye closed form itself, in MEEP's exp(-i omega t) convention.
    assert debye_chi1(delta_epsilon, tau, band_edge) == pytest.approx(
        delta_epsilon / complex(1.0, -2 * math.pi * band_edge * tau), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Layer 2 — sigma registration and the polarization kernel
# ---------------------------------------------------------------------------


def _grid(cells=(4, 5, 6), resolution=10.0):
    return Grid(resolution=resolution, cell_size=tuple(n / resolution for n in cells),
                courant=0.5, symmetry=(), k_point=(0.0, 0.0, 0.0), xp=np)


def test_normalize_sigma_accepts_the_three_spellings():
    grid = _grid()
    uniform = normalize_sigma(0.7, grid)
    assert set(uniform) == set(E_COMPONENTS) and all(value == 0.7 for value in uniform.values())
    volume = np.full(grid.shape, 0.25, dtype=np.float32)
    shared = normalize_sigma(volume, grid)
    assert all(shared[name] is shared["Ex"] for name in E_COMPONENTS), (
        "a single sigma volume must be aliased into all three components, not copied three times"
    )
    per_component = normalize_sigma({"Ex": volume, "Ez": 0.5}, grid)
    assert per_component["Ey"] == 0.0 and per_component["Ez"] == 0.5
    assert per_component["Ex"].shape == grid.shape


def test_normalize_sigma_refuses_gain_and_mismatched_shapes():
    grid = _grid()
    with pytest.raises(ValueError, match="gain"):
        normalize_sigma(-0.1, grid)
    with pytest.raises(ValueError, match="gain"):
        normalize_sigma(np.full(grid.shape, -1.0, dtype=np.float32), grid)
    with pytest.raises(ValueError, match="matches neither"):
        normalize_sigma(np.zeros((3, 3, 3), dtype=np.float32), grid)
    with pytest.raises(ValueError, match="electric components"):
        normalize_sigma({"Hx": 1.0}, grid)
    # MEEP's (N+1) padded shape is trimmed exactly as set_epsilon trims it.
    padded = np.zeros(tuple(n + 1 for n in grid.shape), dtype=np.float32)
    padded[:-1, :-1, :-1] = 3.0
    assert float(normalize_sigma(padded, grid)["Ex"].min()) == 3.0


def test_sigma_is_trivial_detects_both_zero_spellings():
    grid = _grid()
    assert sigma_is_trivial(0.0, np)
    assert not sigma_is_trivial(1e-30, np)
    assert sigma_is_trivial(np.zeros(grid.shape, dtype=np.float32), np)
    volume = np.zeros(grid.shape, dtype=np.float32)
    volume[1, 1, 1] = 1e-20
    assert not sigma_is_trivial(volume, np)


@pytest.mark.parametrize("cells", [(4, 5, 6), (5, 5, 7), (8, 8, 9)])
@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_sample_region_registers_at_each_components_own_yee_position(cells, offset):
    """The half-cell registration hunt, swept over odd AND even cell counts.

    MEEP point-samples sigma at each component's OWN Yee position and never subpixel
    averages it, so Ex, Ey and Ez disagree by half a cell at any interface. Two
    errors have cost this engine 6.6e-2 and 4.8e-2 before: measuring from -L/2
    instead of ``Grid.axis_origin`` (which agree only for EVEN cell counts, hence
    the odd cases here), and sampling at the cell centre. The assertion is that the
    boundary lands on the analytically correct index for each component, at an
    interface both on a lattice point and half a cell off it.
    """
    grid = _grid(cells)
    boundary = grid.axis_origin(2) + (cells[2] // 2 + offset) * grid.dx
    for component in E_COMPONENTS:
        volume = sample_region(grid, lambda x, y, z: np.where(z >= boundary, 1.0, 0.0), component)
        coordinates = component_coordinates(grid, component)
        expected_first = int(np.argmax(np.asarray(coordinates[2]) >= boundary))
        first_hot = int(np.argmax(volume[0, 0, :] > 0.0))
        assert first_hot == expected_first, (
            f"{component} sigma boundary landed at z index {first_hot}, not {expected_first}, for "
            f"cells={cells} offset={offset}"
        )
        assert volume.dtype == np.float32
        # The z-shifted component must differ from the others exactly when the
        # interface does not sit on its own half-cell.
        assert volume.shape == grid.shape
    z_positions = np.asarray(component_coordinates(grid, "Ez")[2])
    x_positions = np.asarray(component_coordinates(grid, "Ex")[2])
    assert np.allclose(z_positions - x_positions, 0.5 * grid.dx)


def test_sample_region_refuses_a_magnetic_component():
    with pytest.raises(ValueError, match="electric component"):
        sample_region(_grid(), lambda x, y, z: np.ones_like(z), "Hx")


def test_polarization_state_reproduces_the_scalar_recurrence():
    """The array kernel against the float64 model, including the buffer rotation."""
    grid = _grid((3, 3, 3))
    term = Susceptibility(1.1, 0.05, LORENTZIAN)
    state = PolarizationState(term, 0.6, grid, np.complex64)
    omega = 2.0 * math.pi * 0.9
    reference, previous = 0j, 0j
    c_now, c_prev, c_drive = term.coefficients(grid.dt)
    for step in range(400):
        drive_value = cmath.exp(-1j * omega * step * grid.dt)
        buffer = np.full(grid.shape, drive_value, dtype=np.complex64)
        state.update(lambda component: buffer, grid.dt)
        reference, previous = (
            c_now * reference + c_prev * previous + c_drive * (0.6 * complex(np.complex64(drive_value))),
            reference,
        )
    for component in E_COMPONENTS:
        measured = complex(state.P[component][1, 1, 1])
        assert abs(measured - reference) <= 2e-5 * abs(reference), (
            f"{component}: array kernel {measured} against scalar model {reference}"
        )


def test_polarization_state_buffers_never_alias():
    """Every P / P_prev / scratch slot must hold a distinct array after every step.

    The three-way rotation costs zero copies precisely because the retired history
    buffer becomes the next component's scratch. If it ever aliased two live slots,
    one component's polarization would silently overwrite another's — and the fields
    would stay finite and smooth.
    """
    grid = _grid((3, 3, 3))
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), 1.0, grid, np.complex64)
    ones = np.ones(grid.shape, dtype=np.complex64)
    for _ in range(5):
        state.update(lambda component: ones, grid.dt)
        live = [state.P[name] for name in E_COMPONENTS]
        live += [state.P_prev[name] for name in E_COMPONENTS]
        live.append(state._scratch)
        identities = {id(array) for array in live}
        assert len(identities) == len(live), "a polarization buffer is aliased into two slots"


def test_polarization_state_skips_components_whose_sigma_is_zero():
    grid = _grid((3, 3, 3))
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN),
                              {"Ez": 1.0}, grid, np.complex64)
    assert state.driven() == ("Ez",)
    assert set(state.P) == {"Ez"} and set(state.P_prev) == {"Ez"}
    assert not state.drives("Ex")
    # Skipping is EXACT, not an approximation: the recurrence is homogeneous and
    # starts at zero, so a sigma = 0 component's P is provably 0.0 for ever.
    target = np.ones(grid.shape, dtype=np.complex64)
    state.subtract_into("Ex", target)
    assert np.array_equal(target, np.ones(grid.shape, dtype=np.complex64))
    empty = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), 0.0, grid, np.complex64)
    assert empty.driven() == () and empty._scratch is None


def test_polarization_update_has_no_spatial_stencil():
    # The isotropic update is element-wise; MEEP's OFFDIAG neighbour reads live only
    # in the anisotropic branches. That is the whole reason P needs no boundary pass.
    grid = _grid((5, 5, 5))
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), 1.0, grid, np.complex64)
    drive = np.zeros(grid.shape, dtype=np.complex64)
    drive[2, 2, 2] = 1.0
    for _ in range(3):
        state.update(lambda component: drive, grid.dt)
    polarization = state.P["Ez"]
    assert polarization[2, 2, 2] != 0.0
    off_cell = np.array(polarization, copy=True)
    off_cell[2, 2, 2] = 0.0
    assert not np.any(off_cell), "the polarization update leaked into a neighbouring cell"


def test_polarization_state_reset_clears_the_history_too():
    grid = _grid((3, 3, 3))
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), 1.0, grid, np.complex64)
    ones = np.ones(grid.shape, dtype=np.complex64)
    for _ in range(4):
        state.update(lambda component: ones, grid.dt)
    assert np.any(state.P["Ez"]) and np.any(state.P_prev["Ez"])
    state.reset()
    for component in E_COMPONENTS:
        assert not np.any(state.P[component])
        assert not np.any(state.P_prev[component]), (
            "P_prev must be cleared too: the ADE is second order, so a stale history "
            "would restart the run from a past it never lived through"
        )


def test_polarization_state_refuses_a_foreign_timestep():
    grid = _grid((3, 3, 3))
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), 1.0, grid, np.complex64)
    with pytest.raises(ValueError, match="dt"):
        state.update(lambda component: np.zeros(grid.shape, dtype=np.complex64), grid.dt * 2)


def test_bytes_per_cell_counts_a_susceptibility():
    grid = _grid((4, 4, 4))
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), 1.0, grid, np.complex64)
    # Three driven components: P + P_prev each (6 arrays) plus one shared scratch.
    assert state.bytes_per_cell(8) == 7 * 8
    single = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), {"Ez": 1.0}, grid, np.complex64)
    assert single.bytes_per_cell(8) == 3 * 8
    volume = np.ones(grid.shape, dtype=np.float32)
    shared = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), volume, grid, np.complex64)
    assert shared.bytes_per_cell(8) == 7 * 8 + 4  # One aliased sigma volume, counted once.


# ---------------------------------------------------------------------------
# Layer 3 — the physics
# ---------------------------------------------------------------------------


LORENTZ_BAND = (0.7, 0.9, 1.1, 1.3, 1.5)
LORENTZ_TERM = (Susceptibility(1.1, 0.2, LORENTZIAN), 0.6)
LORENTZ_EPS_INF = 2.25


def _measure_index(resolution):
    """Measured (complex ratio, n) at each band frequency, plus the analytic targets."""
    measured, separation, dt, dx = _propagation_probe(
        [LORENTZ_TERM], LORENTZ_EPS_INF, LORENTZ_BAND, resolution, monitor_z=-2.2
    )
    rows = []
    for frequency, ratio, amplitude in measured:
        term, sigma = LORENTZ_TERM
        epsilon_tilde = LORENTZ_EPS_INF + sigma * term.chi1_discrete(frequency, dt)
        exact = cmath.exp(1j * _discrete_wavenumber(epsilon_tilde, frequency, dt, dx) * separation)
        phase = -1j * cmath.log(ratio)
        branch = round(((_discrete_wavenumber(epsilon_tilde, frequency, dt, dx) * separation) - phase).real
                       / (2 * math.pi))
        index_measured = (phase + 2 * math.pi * branch) / separation / (2 * math.pi * frequency)
        index_continuum = _continuum_index(LORENTZ_EPS_INF + sigma * term.chi1(frequency))
        rows.append({
            "frequency": frequency,
            "ratio_error": abs(ratio - exact) / abs(exact),
            "n_measured": index_measured,
            "n_continuum": index_continuum,
            "n_error": abs(index_measured - index_continuum) / abs(index_continuum),
            "amplitude": amplitude,
        })
    return rows


def test_a_lorentz_mediums_phase_velocity_matches_the_exact_discrete_relation():
    """The primary correctness test: measured exp(i k dz) against the closed form.

    Everything is exercised at once — the ADE, the D = eps_inf E + P coupling, the
    P/E time indexing and the sign convention — and the target carries NO
    discretisation error, so the bound is round-off-scale rather than a physics
    tolerance. Measured on the COMPLEX ratio, which is exp(i k dz) for a forward
    wave and therefore needs no phase unwrapping: the branch ambiguity that would
    make the answer depend on the model being tested never arises.

    A vacuum calibration in the same geometry fixes the achievable floor
    empirically, and a blind control with the susceptibility removed must miss the
    dispersive target by orders of magnitude, so the test cannot pass on vacuum.
    """
    rows = _measure_index(resolution=20)
    for row in rows:
        assert row["amplitude"] > 1e-6, (
            f"f={row['frequency']}: the near monitor measured nothing ({row['amplitude']:.2e}); an "
            f"unmeasurable run must not score as perfect."
        )
        assert row["ratio_error"] < 1e-4, (
            f"f={row['frequency']}: measured exp(i k dz) is {row['ratio_error']:.2e} from the exact "
            f"discrete relation."
        )

    # Calibration: the same geometry with no susceptibility, against the vacuum
    # numerical relation. This is the floor the bound above is set five-fold above.
    calibration, separation, dt, dx = _propagation_probe(
        [], LORENTZ_EPS_INF, LORENTZ_BAND, 20, monitor_z=-2.2
    )
    for frequency, ratio, _ in calibration:
        expected = cmath.exp(1j * _discrete_wavenumber(LORENTZ_EPS_INF, frequency, dt, dx) * separation)
        assert abs(ratio - expected) / abs(expected) < 2e-4

    # Blind control: vacuum must be nowhere near the dispersive target.
    term, sigma = LORENTZ_TERM
    blind = []
    for frequency, ratio, _ in calibration:
        epsilon_tilde = LORENTZ_EPS_INF + sigma * term.chi1_discrete(frequency, dt)
        expected = cmath.exp(1j * _discrete_wavenumber(epsilon_tilde, frequency, dt, dx) * separation)
        blind.append(abs(ratio - expected) / abs(expected))
    assert min(blind) > 5e-2, (
        f"a run with no susceptibility came within {min(blind):.2e} of the dispersive target; the "
        f"test would pass on vacuum"
    )


def test_the_measured_refractive_index_converges_to_the_continuum_closed_form():
    """n(omega) against ``sqrt(eps_inf + sigma*chi1(f))``, spanning the resonance.

    This target is the CONTINUUM closed form, so the comparison necessarily carries
    the scheme's discretisation error — 5.3e-3 on resonance at resolution 40 for
    chi1 alone, before any Maxwell discretisation. Asserting a fixed tight bound
    would therefore either fail on correct code or bake that error in as if it were
    right, so the assertion is the CONVERGENCE RATE: halving dx must quarter the
    error. A fixed loose bound is asserted alongside it, and the measured and
    analytic n are reported at every frequency.
    """
    coarse = _measure_index(resolution=20)
    fine = _measure_index(resolution=40)
    report = []
    for low, high in zip(coarse, fine):
        report.append(
            f"f={low['frequency']:.2f}  n_measured={low['n_measured']:.4f}  "
            f"n_analytic={low['n_continuum']:.4f}  rel={low['n_error']:.2e} "
            f"(resolution 40: {high['n_error']:.2e})"
        )
        assert low["n_error"] < 3e-2, "\n".join(report)
        ratio = low["n_error"] / high["n_error"]
        assert ratio > 2.8, (
            f"n(omega) error fell only {ratio:.2f}x when dx halved, not ~4x — the agreement is not "
            f"second-order convergence to the continuum answer.\n" + "\n".join(report)
        )
    # Anomalous dispersion: Re n rises towards the resonance and drops beyond it,
    # and the loss peaks on resonance. A material that merely had the right average
    # index would not do this.
    real_parts = [row["n_measured"].real for row in coarse]
    losses = [row["n_measured"].imag for row in coarse]
    assert real_parts[1] > real_parts[0] and real_parts[3] < real_parts[2]
    assert losses[2] == max(losses)
    print("\n".join(report))


def test_a_drude_half_space_matches_the_fresnel_reflectance_and_the_analytic_skin_depth():
    """Two independent numbers from one run, both asserted.

    ``R = |(1-n)/(1+n)|^2`` from a two-point decomposition of the standing wave in
    front of the metal (which needs no separate normalisation run: two samples fix
    the forward and backward amplitudes exactly), and the field 1/e depth
    ``delta_E = 1/(2 pi f Im n)`` from the decay inside it. The INTENSITY depth is
    asserted separately against ``delta_E/2``, because checking only one of them
    lets a factor of two through while still producing a beautiful exponential.

    The metal is a sigma volume built with ``sample_region``, so this is also the
    heterogeneous-registration case, and it runs THROUGH the +z absorber — a
    dispersive material overlapping a PML, which doc/docs/Materials.md:79 warns may
    be unstable and which this measurement shows is reproduced here.
    """
    term, sigma = Susceptibility(1.0, 0.05, DRUDE), 1.0
    frequencies = (0.5, 0.6, 0.7, 0.8)
    resolution = 40
    driver = FdtdDriver(cell_size=(0.2, 0.2, 8.0), resolution=resolution, courant=0.5,
                        force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    metal = {component: sample_region(driver.grid,
                                      lambda x, y, z: np.where(z >= 0.0, sigma, 0.0), component)
             for component in E_COMPONENTS}
    driver.add_susceptibility(term, metal)
    driver.setup_pml({"z": resolution})
    driver.add_source({"component": "Ex", "source_type": "gaussian", "frequency": 0.65,
                       "fwidth": 0.4, "center": (0.0, 0.0, -2.0), "size": (0.2, 0.2, 0.0)})
    separation = 3 * driver.dx
    positions = (-1.0, -1.0 + separation, 0.25, 0.25 + separation)
    monitors = [driver.add_dft_monitor(frequencies=frequencies, components=("Ex",),
                                       center=(0.0, 0.0, z), size=(0.0, 0.0, 0.0))
                for z in positions]
    driver.run(until=80.0)
    report = []
    for index, frequency in enumerate(frequencies):
        samples = [complex(np.asarray(m.get_dft("Ex", index)).ravel()[0]) for m in monitors]
        assert all(abs(value) > 1e-8 for value in samples), f"f={frequency}: nothing was measured"
        k_vacuum = _vacuum_wavenumber(frequency, driver.dt, driver.dx)
        basis = np.array([
            [cmath.exp(1j * k_vacuum * positions[0]), cmath.exp(-1j * k_vacuum * positions[0])],
            [cmath.exp(1j * k_vacuum * positions[1]), cmath.exp(-1j * k_vacuum * positions[1])],
        ])
        forward, backward = np.linalg.solve(basis, np.array(samples[:2]))
        reflectance = abs(backward / forward) ** 2

        n = _continuum_index(1.0 + sigma * term.chi1(frequency))
        fresnel = abs((1.0 - n) / (1.0 + n)) ** 2
        field_depth_analytic = 1.0 / (2.0 * math.pi * frequency * n.imag)
        field_depth = 1.0 / (-1j * cmath.log(samples[3] / samples[2]) / separation).imag
        intensity_depth = separation / math.log(abs(samples[2]) ** 2 / abs(samples[3]) ** 2)

        report.append(
            f"f={frequency:.2f}  R_measured={reflectance:.5f} R_fresnel={fresnel:.5f} "
            f"(rel {abs(reflectance - fresnel) / fresnel:.2e})  delta_E={field_depth:.5f} vs "
            f"{field_depth_analytic:.5f}  delta_I={intensity_depth:.5f} vs "
            f"{field_depth_analytic / 2:.5f}"
        )
        assert abs(reflectance - fresnel) / fresnel < 5e-3, "\n".join(report)
        assert abs(field_depth - field_depth_analytic) / field_depth_analytic < 5e-3, "\n".join(report)
        assert abs(intensity_depth - field_depth_analytic / 2) / (field_depth_analytic / 2) < 5e-3, (
            "\n".join(report)
        )
        assert reflectance < 1.0, "a passive metal cannot reflect more than it receives"
    driver.close()
    print("\n".join(report))


# ---------------------------------------------------------------------------
# Layer 4 — degenerate cases and refusals
# ---------------------------------------------------------------------------


def _short_run(configure, complex_fields=True):  # Short driven run, either storage mode.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, courant=0.5,
                        force_complex_fields=complex_fields)
    driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
    configure(driver)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -0.35),
                       "size": (0.0, 0.0, 0.0)})
    driver.run(num_steps=200)
    fields = {name: driver.get_field(name, cell_centered=False)
              for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    driver.close()
    return fields


@pytest.mark.parametrize("spelling", ["scalar", "array"])
def test_a_zero_strength_susceptibility_is_bit_identical_to_the_non_dispersive_engine(spelling):
    """The single most important regression guard, pinned on raw bytes.

    A sigma of exactly zero must reduce EXACTLY — not to within a tolerance — to the
    engine as it was before dispersion existed. It holds structurally: no P array is
    allocated for a trivial sigma, ``displacement_minus_polarization`` then returns
    the D array itself rather than forming ``D - 0.0``, and the stored-E branch
    writes precisely the product the on-demand branch computes. Anything less than
    byte equality means the dispersive path is doing arithmetic to a run that asked
    for none.

    Note what this pin is BLIND to: it would still pass if ``get_E`` recomputed
    ``D*inv_eps`` while a live polarization existed, because with sigma = 0 there is
    no polarization. The propagation tests above are what cover that.
    """
    baseline = _short_run(lambda driver: None)

    def add_zero(driver):
        sigma = 0.0 if spelling == "scalar" else np.zeros(driver.shape, dtype=np.float32)
        driver.add_susceptibility(Susceptibility(1.0, 0.1, LORENTZIAN), sigma)

    dispersive = _short_run(add_zero)
    for name, reference in baseline.items():
        assert np.array_equal(dispersive[name], reference), (
            f"{name} differs after adding a zero-strength susceptibility; the degenerate case must "
            f"be byte-identical, not merely close."
        )
    # Positive control: the same comparison on a NONZERO sigma must fail, so the pin
    # cannot be passing because the run computed nothing.
    live = _short_run(
        lambda driver: driver.add_susceptibility(Susceptibility(1.0, 0.1, LORENTZIAN), 0.6)
    )
    assert not np.array_equal(live["Ez"], baseline["Ez"])
    assert np.linalg.norm(live["Ez"] - baseline["Ez"]) / np.linalg.norm(baseline["Ez"]) > 0.05


def test_every_susceptibility_in_the_list_acts():
    """A second term must change the answer — the bug where only polarizations[0] is applied.

    Every single-term test passes with that defect, which is why the two-term case is
    not optional.
    """
    first = Susceptibility(1.0, 0.05, DRUDE)
    second = Susceptibility(1.4, 0.1, LORENTZIAN)
    one = _short_run(lambda driver: driver.add_susceptibility(first, 0.8))
    both = _short_run(lambda driver: (driver.add_susceptibility(first, 0.8),
                                      driver.add_susceptibility(second, 0.5)))
    swapped = _short_run(lambda driver: (driver.add_susceptibility(second, 0.5),
                                         driver.add_susceptibility(first, 0.8)))
    difference = np.linalg.norm(both["Ez"] - one["Ez"]) / np.linalg.norm(one["Ez"])
    assert difference > 0.05, f"adding a second susceptibility changed the field by only {difference:.2e}"
    # Order of addition must not matter beyond float32 non-associativity: D - P1 - P2
    # and D - P2 - P1 differ only in the summation order of the same three terms.
    assert np.linalg.norm(swapped["Ez"] - both["Ez"]) <= 1e-5 * np.linalg.norm(both["Ez"])


def test_an_unstable_susceptibility_is_refused_before_any_array_is_allocated():
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, courant=0.5,
                        force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    too_fast = 1.0 / (math.pi * driver.dt) * 1.5
    with pytest.raises(DispersionInstability, match="unit circle"):
        driver.add_susceptibility(Susceptibility(too_fast, 0.0, LORENTZIAN), 1.0)
    assert driver.fields.polarizations == []
    assert not driver.fields.has_polarizations
    driver.close()


def test_a_courant_factor_too_large_for_eps_infinity_is_refused():
    # doc/docs/Materials.md:73 — S < n_min/sqrt(3), and the fastest numerical modes
    # see the INSTANTANEOUS response, so n_min = sqrt(min eps_inf), not sqrt(eps(0)).
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, courant=0.55,
                        force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 0.8, dtype=np.float32))
    with pytest.raises(ValueError, match="eps_inf"):
        driver.add_susceptibility(Susceptibility(1.0, 0.1, LORENTZIAN), 1.0)
    driver.close()
    # The same term on eps_inf = 1 at the same Courant factor is legal.
    fine = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, courant=0.55,
                      force_complex_fields=True)
    fine.set_epsilon(np.full(fine.shape, 1.0, dtype=np.float32))
    fine.add_susceptibility(Susceptibility(1.0, 0.1, LORENTZIAN), 1.0)
    fine.close()


def test_a_susceptibility_added_after_stepping_is_refused():
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    driver.step()
    with pytest.raises(RuntimeError, match="before the first step"):
        driver.add_susceptibility(Susceptibility(1.0, 0.1, LORENTZIAN), 1.0)
    driver.close()


def test_the_divergence_guard_fires_on_a_run_that_slipped_past_the_pole_test():
    """A guard that has never fired is not a guard.

    The setup gate is bypassed deliberately — the coefficients of a knowingly
    unstable term are installed on a legal one — because the point is to test what
    happens when a run diverges ANYWAY: from a PML interaction, a conductivity, or a
    boundary the closed-form pole test knows nothing about. The failure this
    prevents is not a NaN; it is a large, smooth, finite, entirely plausible field,
    so the instability chosen here is a MILD one (|z| ~ 1.02, a percent per step)
    that stays finite throughout and is caught by the energy criterion rather than
    by an overflow.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, courant=0.5,
                        force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    # A WEAK coupling (sigma = 0.002) keeps the growth entirely in the polarization's
    # own pole rather than in the P/E feedback, so the run stays finite for hundreds
    # of steps and the energy criterion is what fires — the branch that matters.
    driver.add_susceptibility(Susceptibility(1.0, 0.05, LORENTZIAN), 0.002)
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 1.0, "center": (0.0, 0.0, 0.05), "size": (0.0, 0.0, 0.0)})
    # omega_0*dt just past 2 — MEEP's rule of thumb — which is |z| slightly above 1.
    unstable = Susceptibility(2.0001 / (2 * math.pi * driver.dt), 0.0, LORENTZIAN)
    report = unstable.stability(driver.dt)
    assert 1.0 < report.pole_magnitude < 1.05, f"the mutation must be MILD: {report.describe()}"
    driver.fields.polarizations[0]._coefficients = unstable.coefficients(driver.dt)
    driver._susceptibilities[0] = (unstable, 0.002)
    with pytest.raises(FdtdDivergence) as raised, np.errstate(all="ignore"):
        driver.run(until=60.0)
    message = str(raised.value)
    assert "max|z|" in message, "the guard must name the susceptibility responsible"
    assert f"{unstable.frequency:g}" in message
    assert "grew" in message, "the mild case must be caught by the energy criterion, not by a NaN"
    assert np.isfinite(driver.get_field("Ez")).all(), (
        "the field was still entirely finite when the guard fired — which is the whole point"
    )
    driver.close()


def test_a_stable_dispersive_run_is_not_accused_by_the_guard():
    # The control for the test above: the guard must not fire on a legal run, or it
    # would simply be a run-length limit.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, courant=0.5,
                        force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
    driver.add_susceptibility(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.0, 0.0, -0.35), "size": (0.0, 0.0, 0.0)})
    driver.run(until=40.0)
    assert np.isfinite(driver.get_field("Ez")).all()
    driver.close()


def test_reset_restarts_a_dispersive_run_identically():
    """A re-run must reproduce the first run exactly, P_prev included.

    The ADE is second order, so a reset that cleared P but not P_prev would restart
    from a history the run never lived through — the same defect class the
    integrated-source reset documents, and equally invisible in the output.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
    driver.add_susceptibility(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -0.35),
                       "size": (0.0, 0.0, 0.0)})
    driver.run(num_steps=120)
    first = driver.get_field("Ez", cell_centered=False)
    assert np.linalg.norm(first) > 0.0
    driver.reset()
    assert not np.any(driver.fields.polarizations[0].P["Ez"])
    assert not np.any(driver.fields.polarizations[0].P_prev["Ez"])
    driver.run(num_steps=120)
    assert np.array_equal(driver.get_field("Ez", cell_centered=False), first)
    driver.close()


def test_get_epsilon_reports_the_instantaneous_and_the_dispersive_permittivity():
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
    term, sigma = Susceptibility(1.1, 0.05, LORENTZIAN), 0.6
    driver.add_susceptibility(term, sigma)
    instantaneous = driver.get_epsilon()
    assert instantaneous.dtype == np.float32 and np.allclose(instantaneous, 2.25)
    at_frequency = driver.get_epsilon(frequency=0.9)
    assert np.iscomplexobj(at_frequency)
    assert at_frequency.flat[0] == pytest.approx(2.25 + sigma * term.chi1(0.9), rel=1e-12)
    with pytest.raises(ValueError):
        driver.get_epsilon(frequency=0.0)
    assert driver.susceptibility_report()[0].startswith("lorentzian")
    driver.close()


def test_the_two_orders_of_update_E_and_update_P_agree():
    """Swapping the two sub-steps is a relabelling, not a detuning — measured, then pinned.

    The design this feature was built from expected the swap to detune the resonance
    by roughly omega_0*dt. It does not, and the algebra says why: advancing P first
    drives it from the previous step's W AND makes the stored E subtract the
    already-advanced P, so both indices shift by one and cancel. Writing
    ``S^m = P^(m+1)`` turns the swapped scheme into the original one term for term.

    This is pinned rather than merely noted because the equivalence holds only while
    nothing reads P BETWEEN the two calls. Add a monitor, a diagnostic or an
    absorbed-power tally there and it stops being true — and this test is what would
    say so, rather than a parity floor drifting a few percent months later.
    """
    from . import stepping as stepping_module

    def run(swapped):
        driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, courant=0.5,
                            force_complex_fields=True)
        driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
        driver.add_susceptibility(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)
        driver.setup_pml({"z": 6})
        driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                           "fwidth": 0.6, "center": (0.0, 0.0, -0.25), "size": (0.0, 0.0, 0.0)})
        if not swapped:
            driver.run(num_steps=150)
        else:
            for _ in range(150):
                driver._require_open()
                if driver._pending_monitor_regions:
                    driver._resolve_monitor_regions()
                dt = driver.grid.dt
                stepping_module.step_B(driver.fields, driver.pml)
                stepping_module.update_H(driver.fields, driver.pml)
                stepping_module.step_D(driver.fields, driver.pml)
                for source in driver._sources:
                    source.inject(driver.fields, driver.time + 0.5 * dt)
                stepping_module.fill_symmetry_bc_D(driver.fields)
                stepping_module.update_P(driver.fields, driver.pml)   # Swapped.
                stepping_module.update_E(driver.fields, driver.pml)
                driver.step_count += 1
        result = driver.get_field("Ez", cell_centered=False)
        driver.close()
        return result

    ordered = run(swapped=False)
    swapped = run(swapped=True)
    assert np.linalg.norm(ordered) > 0.0, "the run produced nothing; the comparison would be vacuous"
    np.testing.assert_array_equal(swapped, ordered)


def test_a_zero_strength_susceptibility_allocates_nothing_at_all():
    """The degenerate case must cost no memory either, not merely produce the same field.

    Once ``update_E`` keys on the storage flag rather than on the polarization list,
    switching storage on for a sigma = 0 term still produces a byte-identical run —
    the stored E is written with exactly the product the on-demand branch computes —
    so the bit-identity pin alone cannot see the waste. Three complex64 volumes on a
    256^3 grid is 400 MB, which is the difference between a job that fits on the card
    and one that does not, so it is pinned here instead.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    baseline = driver.bytes_per_cell()
    driver.add_susceptibility(Susceptibility(1.0, 0.1, LORENTZIAN), 0.0)
    assert driver.bytes_per_cell() == baseline, "a zero-strength term must allocate nothing"
    assert driver.fields.Ex is None and not driver.fields.stores_E
    assert driver.fields.polarizations[0].driven() == ()
    driver.add_susceptibility(Susceptibility(1.0, 0.1, LORENTZIAN), 0.6)
    assert driver.bytes_per_cell() == baseline + 3 * 8 + 7 * 8
    assert driver.fields.stores_E
    driver.close()
