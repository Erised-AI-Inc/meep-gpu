"""
Tests for the STOCK-MEEP susceptibility-sigma recovery, and its lift into the converter.

Two layers, in the order a defect would reach them:

1. **The numerics alone**, with no MEEP at all — the basis against MEEP's own
   ``chi1`` closed form retyped from susceptibility.cpp:297-307, an exact recovery
   from a synthetic permittivity, the sampling band, and each of the three
   refusals :mod:`~.sigma_recovery` exists to make (a singular basis, a
   zero-frequency pole, a pole list the held-out frequency disagrees with), and
   the converter's clamp and refusal of a recovered gain.
2. **The lift end to end on stock MEEP**, against CPU MEEP stepping the same
   cell — with the declared-geometry lookup forced to decline, so that the
   INVERSION is the route under test. The lookup is the default route
   (``sigma_lookup.py``), so without that the lift would exercise it instead; the
   route each lift took is asserted rather than assumed. Measured on 2026-08-04
   against MEEP's stored sigma words, point by point, the inversion agrees to
   5.7e-08 on the two-block Lorentz+Drude cell (resolution 10, cond 5.71e+00, 6
   samples) and to 1.4e-06 on ``meep.materials.Au`` (6 poles, resolution 20, cond
   8.57e+02, 14 samples).

THE NEGATIVE CONTROL, and why it is not optional. The whole recovery turns on
solving for ``c_n = sigma_n * w0_n**2`` rather than for ``sigma_n``. That choice
looks like a detail and is the difference between a working recovery and a
confidently wrong one: on ``meep.materials.Au`` the naive basis is singular to
working precision and returns a NEGATIVE sigma where the truth is positive — gain,
from a passive metal, with nothing raised on the way out. The numbers are measured
in ``test_the_naive_sigma_basis_is_singular_where_the_product_basis_is_not`` so
that undoing the reparameterization can never be silent.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from .sigma_recovery import (
    CONDITION_LIMIT,
    SigmaRecoveryRefused,
    SusceptibilityPole,
    choose_frequencies,
    plan_recovery,
)

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)


def meep_chi1(frequency: float, gamma: float, drude: bool, sigma: float, f: float) -> complex:
    """MEEP's ``lorentzian_susceptibility::chi1``, retyped from susceptibility.cpp:297-307.

    Written out here rather than imported so that the oracle for the basis is an
    independent transcription of the C++ this whole module is built on. The Drude
    branch (``no_omega_0_denominator``) drops ``w0**2`` from the DENOMINATOR only —
    it stays in the numerator, which is why a Drude term's ``frequency`` is a scale
    factor rather than a resonance.
    """
    if drude:
        return sigma * frequency * frequency / complex(-f * f, -gamma * f)
    return sigma * frequency * frequency / complex(frequency * frequency - f * f, -gamma * f)


def synthetic_epsilon(poles, sigma, eps_inf, frequencies) -> np.ndarray:
    """``eps(w) = eps_inf + SUM_n sigma_n*chi_n(w)`` at each frequency, per point.

    Built from :func:`meep_chi1`, i.e. from the C++ closed form rather than from the
    module under test, so a recovery that agrees with it agrees with MEEP.
    """
    sigma = np.asarray(sigma, dtype=float)          # (npoles, npoints)
    eps_inf = np.asarray(eps_inf, dtype=float)      # (npoints,)
    out = np.empty((len(frequencies), sigma.shape[1]), dtype=complex)
    for index, f in enumerate(frequencies):
        column = eps_inf.astype(complex)
        for n, pole in enumerate(poles):
            column = column + np.array([
                meep_chi1(pole.frequency, pole.gamma, pole.drude, value, float(f))
                for value in sigma[n]
            ])
        out[index] = column
    return out


# --- the numerics alone (no MEEP) --------------------------------------------------


def test_the_basis_is_meeps_chi1_divided_by_the_product_it_solves_for():
    """``pole.basis(w) * (sigma*w0**2)`` must BE MEEP's chi1, for both kinds.

    The reparameterization in one assertion: the module's unknown is ``c = sigma*w0**2``
    and its basis is ``chi1 / c``, so multiplying one by the other has to return the
    C++ function exactly. A basis written per unit SIGMA instead — the naive
    parameterization the negative control below measures — fails this by the factor
    ``w0**2``, which is 1e-20 for MEEP's Drude idiom.
    """
    cases = [
        SusceptibilityPole(frequency=1.0, gamma=0.1),
        SusceptibilityPole(frequency=2.0, gamma=0.05, drude=True),
        SusceptibilityPole(frequency=1e-10, gamma=0.0427474, drude=True),  # Au's Drude term.
        SusceptibilityPole(frequency=10.7433, gamma=1.78571),
    ]
    for pole in cases:
        for sigma in (0.4, 7.0, 4.0314e21):
            for f in (0.2, 1.0, 3.7):
                mine = pole.basis(np.array([f]))[0] * sigma * pole.scale
                theirs = meep_chi1(pole.frequency, pole.gamma, pole.drude, sigma, f)
                error = abs(mine - theirs) / max(abs(theirs), 1e-300)
                assert error < 1e-14, (
                    f"basis * sigma*w0**2 for f0={pole.frequency:g} drude={pole.drude} "
                    f"sigma={sigma:g} at f={f:g} is {mine!r} against MEEP's {theirs!r} "
                    f"(relative {error:.3e})"
                )


def test_sigma_is_recovered_exactly_from_a_synthetic_permittivity():
    """Three poles, five points, one of them vacuum: recovered to the solve's own noise.

    The end-to-end statement of what the module claims, with MEEP replaced by the
    closed form it implements. The vacuum point (every sigma zero) is included
    deliberately — it is the case that recovers as a least-squares residue rather
    than as an exact zero, and the case the converter's gain clamp exists for.
    """
    poles = [
        SusceptibilityPole(frequency=1.0, gamma=0.1),
        SusceptibilityPole(frequency=2.0, gamma=0.05, drude=True),
        SusceptibilityPole(frequency=0.4, gamma=0.02),
    ]
    truth = np.array([
        [0.4, 0.0, 1.7, 0.0, 0.25],
        [0.0, 0.7, 0.0, 0.0, 3.10],
        [0.9, 0.0, 0.0, 0.0, 0.05],
    ])
    eps_inf = np.array([4.0, 2.0, 1.0, 1.0, 12.0])
    plan = plan_recovery(poles)
    samples = synthetic_epsilon(poles, truth, eps_inf, plan.frequencies)
    holdout = synthetic_epsilon(poles, truth, eps_inf, [plan.holdout_frequency])[0]

    recovered = plan.recover(samples, holdout)
    scale = float(np.abs(truth).max())
    worst = float(np.abs(recovered - truth).max()) / scale
    print(f"[sigma_recovery] synthetic cond={plan.condition_number:.3e} "
          f"n={plan.sample_count} worst={worst:.3e}", flush=True)
    assert worst < 1e-12, (
        f"recovered sigma differs from the truth by {worst:.3e} of the largest sigma; "
        f"the closed form and the basis have diverged"
    )


def test_choose_frequencies_spans_the_poles_geometrically():
    """The band is what sets the conditioning, so the band is what is pinned.

    Half the lowest finite pole to 1.5x the highest, geometrically spaced,
    ``oversample*(N+1)`` points — the strategy the module docstring's measured table
    selected. A linear band on the same poles is two orders worse conditioned.
    """
    poles = [
        SusceptibilityPole(frequency=0.4, gamma=0.02),
        SusceptibilityPole(frequency=10.0, gamma=1.0),
        SusceptibilityPole(frequency=1e-10, gamma=0.04, drude=True),  # Excluded: not finite-ish.
    ]
    grid = choose_frequencies(poles, oversample=2)
    assert grid.size == 2 * (len(poles) + 1)
    assert grid[0] == pytest.approx(0.5 * 0.4)
    assert grid[-1] == pytest.approx(1.5 * 10.0)
    ratios = grid[1:] / grid[:-1]
    assert np.allclose(ratios, ratios[0]), f"the band is not geometric: {ratios}"
    # A pure-Drude set has no resonance to sample around and falls back to the gammas.
    only_drude = [SusceptibilityPole(frequency=1e-10, gamma=0.04, drude=True)]
    fallback = choose_frequencies(only_drude)
    assert fallback[0] == pytest.approx(0.5 * 0.04)
    assert fallback[-1] == pytest.approx(1.5 * 0.04)


def test_two_indistinguishable_poles_are_refused_rather_than_split():
    """A duplicated pole gives two identical columns; the sigma between them is arbitrary.

    This is the safety net under the enumeration in the converter: deduplicating the
    declared susceptibilities by anything FINER than MEEP's ``susceptibility_equiv``
    (by including sigma, say) produces exactly this basis, and it is refused rather
    than answered.
    """
    poles = [
        SusceptibilityPole(frequency=1.0, gamma=0.1),
        SusceptibilityPole(frequency=1.0, gamma=0.1),
    ]
    with pytest.raises(SigmaRecoveryRefused) as refusal:
        plan_recovery(poles)
    assert "singular" in str(refusal.value)
    assert f"{CONDITION_LIMIT:.0e}" in str(refusal.value)
    assert str(refusal.value).endswith("run this simulation on CPU MEEP."), str(refusal.value)


def test_a_zero_frequency_pole_is_refused_because_sigma_is_c_over_w0_squared():
    """The reparameterization's one precondition, refused by name rather than dividing by zero."""
    with pytest.raises(SigmaRecoveryRefused) as refusal:
        plan_recovery([SusceptibilityPole(frequency=0.0, gamma=0.1)])
    assert "1e-10" in str(refusal.value), "the message should name MEEP's own Drude idiom"


def test_a_wrong_lorentz_pole_is_caught_at_the_held_out_frequency():
    """Conditioning cannot see a wrong pole list; the cross-validation can.

    The plan is built from poles close enough to the truth to give a perfectly
    well-conditioned basis (measured 5.03) and a plausible-looking sigma — the
    failure mode that returns a number nobody can audit. The held-out frequency,
    which took no part in the fit, is what separates them: a resonance 15 % off
    scores **1.09e-01** against the 1e-03 limit, and 5 % off still scores 1.49e-02.
    """
    truth_poles = [
        SusceptibilityPole(frequency=1.0, gamma=0.1),
        SusceptibilityPole(frequency=2.0, gamma=0.05, drude=True),
    ]
    wrong_poles = [
        SusceptibilityPole(frequency=1.15, gamma=0.1),
        SusceptibilityPole(frequency=2.0, gamma=0.05, drude=True),
    ]
    truth = np.array([[0.4, 0.0], [0.0, 0.7]])
    eps_inf = np.array([4.0, 2.0])
    plan = plan_recovery(wrong_poles)
    assert plan.condition_number < 1e2, (
        f"the wrong-pole basis is meant to be WELL conditioned (measured 5.03 when this was "
        f"written, now {plan.condition_number:.3e}); otherwise this test measures the "
        f"conditioning refusal instead of the cross-check"
    )
    samples = synthetic_epsilon(truth_poles, truth, eps_inf, plan.frequencies)
    holdout = synthetic_epsilon(truth_poles, truth, eps_inf, [plan.holdout_frequency])[0]
    with pytest.raises(SigmaRecoveryRefused) as refusal:
        plan.recover(samples, holdout)
    assert "held-out" in str(refusal.value)


def test_a_drude_frequency_is_not_observable_from_chi1inv_and_need_not_be():
    """The one pole parameter chi1inv cannot see — and why that costs nothing.

    A Drude term's chi1 is ``sigma*w0**2 / (-f**2 - i*gamma*f)`` (susceptibility.cpp,
    ``no_omega_0_denominator``): ``w0`` appears ONLY in the numerator, so the shape of
    the basis does not depend on it at all and ``(w0, sigma)`` and
    ``(w0', sigma*w0**2/w0'**2)`` are the same material to any permittivity
    measurement. The held-out cross-check therefore does not fire here, and must not
    be strengthened until it does — a refusal would be refusing a correct recovery.

    It is harmless because the DECLARED ``w0`` is the one MEEP built the structure
    with and the one :class:`~.dispersion.Susceptibility` will multiply back in: the
    recovery solves for the product, divides by ``w0**2``, and the engine squares it
    again. Measured here on a 20 % frequency error: sigma comes back 0.4861 instead of
    0.7 — 31 % "wrong" — while the product that is the entire physical content is
    recovered to 3.9e-16.
    """
    truth_poles = [
        SusceptibilityPole(frequency=1.0, gamma=0.1),
        SusceptibilityPole(frequency=2.0, gamma=0.05, drude=True),
    ]
    wrong_poles = [
        SusceptibilityPole(frequency=1.0, gamma=0.1),
        SusceptibilityPole(frequency=2.4, gamma=0.05, drude=True),
    ]
    truth = np.array([[0.4, 0.0], [0.0, 0.7]])
    eps_inf = np.array([4.0, 2.0])
    plan = plan_recovery(wrong_poles)
    samples = synthetic_epsilon(truth_poles, truth, eps_inf, plan.frequencies)
    holdout = synthetic_epsilon(truth_poles, truth, eps_inf, [plan.holdout_frequency])[0]

    recovered = plan.recover(samples, holdout)  # Must NOT raise.
    product = float(recovered[1, 1]) * wrong_poles[1].scale
    truth_product = 0.7 * truth_poles[1].scale
    error = abs(product - truth_product) / truth_product
    print(f"[sigma_recovery] drude w0 unobservable: sigma={float(recovered[1, 1]):.6g} "
          f"product error={error:.3e}", flush=True)
    assert error < 1e-12, (
        f"sigma*w0**2 — the only thing a Drude term's chi1 depends on — came back {error:.3e} "
        f"wrong, so the frequency mismatch is NOT merely a reparameterization"
    )


def test_a_recovered_gain_is_clamped_inside_the_noise_and_refused_beyond_it():
    """A negative recovered sigma is clamped to MEEP's zero, or refused with what to do instead.

    ``from_meep._settle_recovered_sigma`` runs after every inversion. A least-squares
    residue of either sign at a point where MEEP holds sigma = 0 is clamped to exactly
    zero; a negative sigma worth more permittivity than ``_RECOVERED_GAIN_TOLERANCE``
    is gain, and the lift refuses it with the number and an instruction the user can
    follow on any MEEP. Driven with no MEEP: one Lorentz pole, the plan the recovery
    would have used, and a volume holding the value under test. The two values carry
    7.6e-12 and 7.6e-01 of permittivity (scale 1, largest basis magnitude 7.56 over the
    four sampled frequencies 0.5 .. 1.5), five orders below and six above the 1e-06
    limit, so no change of the limit by less than five orders flips either assertion.
    """
    from .dispersion import LORENTZIAN, Susceptibility
    from .from_meep import _settle_recovered_sigma

    plan = plan_recovery([SusceptibilityPole(frequency=1.0, gamma=0.1, drude=False)])
    term = Susceptibility(frequency=1.0, gamma=0.1, kind=LORENTZIAN)
    volumes = {name: np.full((1, 2, 2, 1), 0.4) for name in ("Ex", "Ey", "Ez")}
    volumes["Ez"][0, 0, 0, 0] = -1e-12
    _settle_recovered_sigma(volumes, (term,), plan)
    assert volumes["Ez"][0, 0, 0, 0] == 0.0, volumes["Ez"]
    assert float(volumes["Ex"].min()) == 0.4, volumes["Ex"]

    volumes["Ez"][0, 1, 1, 0] = -0.1
    with pytest.raises(SigmaRecoveryRefused) as refusal:
        _settle_recovered_sigma(volumes, (term,), plan)
    message = str(refusal.value)
    assert "A negative sigma is gain" in message, message
    assert message.endswith("Run this simulation on CPU MEEP."), message


@requires_meep
@skip_without_meep
def test_the_naive_sigma_basis_is_singular_where_the_product_basis_is_not():
    """THE NEGATIVE CONTROL: solving for sigma directly fails on a real metal.

    Same material, same sampled permittivity, same solver — only the unknown
    changes. Measured on ``meep.materials.Au`` (6 poles, MEEP's Drude idiom
    ``f = 1e-10`` carrying ``sigma = 4.03e+21``), with the permittivity synthesized
    from MEEP's own closed form so nothing else can differ:

        unknown = sigma_n            cond 6e+16 .. 2e+17   worst relative error 9.1e+01
        unknown = sigma_n * w0_n**2  cond 8.57e+02         worst relative error 3.9e-14

    (The naive condition number is reported differently by the two environments'
    LAPACK builds — 2.478e+17 on MEEP 1.29's, 5.914e+16 on 1.33's — which is itself
    what "singular to working precision" means. Both are nine orders past the
    module's 1e+08 limit; the assertion is against the limit, not against a digit.)

    The naive solve does not raise, does not produce a nan, and returns a NEGATIVE
    sigma for a passive metal. This test exists so that undoing the
    reparameterization — in :meth:`~.sigma_recovery.SusceptibilityPole.basis` or in
    the division at the end of :meth:`~.sigma_recovery.SigmaRecoveryPlan.recover` —
    cannot be silent.
    """
    import meep as mp  # noqa: F401 - meep.materials imports it, and the marker gates this test.
    from meep.materials import Au

    poles = [
        SusceptibilityPole(frequency=float(term.frequency), gamma=float(term.gamma),
                           drude=type(term).__name__ == "DrudeSusceptibility")
        for term in Au.E_susceptibilities
    ]
    truth = np.array([[float(term.sigma_diag.x)] for term in Au.E_susceptibilities])
    eps_inf = np.array([float(Au.epsilon_diag.x)])

    plan = plan_recovery(poles)
    samples = synthetic_epsilon(poles, truth, eps_inf, plan.frequencies)
    holdout = synthetic_epsilon(poles, truth, eps_inf, [plan.holdout_frequency])[0]
    recovered = plan.recover(samples, holdout)
    product_error = float(np.max(np.abs(recovered - truth) / np.abs(truth)))

    # The same solve, with sigma_n as the unknown: every column multiplied by w0**2.
    naive_columns = np.array([
        np.concatenate([[1.0], [pole.basis(np.array([f]))[0] * pole.scale for pole in poles]])
        for f in plan.frequencies
    ])
    naive = np.concatenate([naive_columns.real, naive_columns.imag], axis=0)
    naive_condition = float(np.linalg.cond(naive))
    stacked = np.concatenate([samples.real, samples.imag], axis=0)
    naive_sigma = np.linalg.lstsq(naive, stacked, rcond=None)[0][1:]
    naive_error = float(np.max(np.abs(naive_sigma - truth) / np.abs(truth)))

    print(f"[sigma_recovery] Au naive   cond={naive_condition:.3e} worst={naive_error:.3e}\n"
          f"[sigma_recovery] Au product cond={plan.condition_number:.3e} "
          f"worst={product_error:.3e}", flush=True)

    assert naive_condition > CONDITION_LIMIT, (
        f"the naive (per-sigma) basis on Au is conditioned at {naive_condition:.3e}, inside the "
        f"module's own {CONDITION_LIMIT:.0e} limit — it would no longer be refused, and this "
        f"control has stopped controlling anything"
    )
    assert naive_error > 1.0, (
        f"the naive parameterization recovered Au's sigma to {naive_error:.3e} relative — better "
        f"than the 9.1e+01 measured when this was written, which means the reparameterization is "
        f"no longer load-bearing and this control is dead"
    )
    assert float(naive_sigma.min()) < 0.0, (
        f"the naive solve returned no negative sigma (min {float(naive_sigma.min()):.3e}); the "
        f"documented failure mode is gain from a passive metal, silently"
    )
    assert product_error < 1e-9, (
        f"the product parameterization recovered Au's sigma to only {product_error:.3e}"
    )


# --- cells -------------------------------------------------------------------------


def _two_block_cell(mp, resolution: int = 10):
    """A Lorentz block and a Drude block in vacuum: three media, dispersion structured."""
    geometry = [
        mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
        mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=2.0, E_susceptibilities=[
                     mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)])),
    ]
    return mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=resolution,
                         geometry=geometry, force_complex_fields=True,
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez,
                                            center=mp.Vector3(0, -1.2, 0))])


def _shared_pole_cell(mp, resolution: int = 10):
    """Two blocks whose susceptibilities differ ONLY in sigma — ONE chain entry in MEEP.

    ``susceptibility_equiv`` (meepgeom.cpp:1633-1647) ignores sigma, so these two
    declarations are merged into a single entry carrying one sigma VOLUME. The
    recovery has to enumerate one unknown, not two: two would be two identical basis
    columns and the plan would refuse as singular.
    """
    geometry = [
        mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
        mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=2.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.9)])),
    ]
    return mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=resolution,
                         geometry=geometry, force_complex_fields=True,
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez,
                                            center=mp.Vector3(0, -1.2, 0))])


# --- the lift, end to end ----------------------------------------------------------


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["two_block", "shared_pole"])
def test_a_structured_dispersive_cell_lifts_and_steps_through_the_inversion(monkeypatch, case):
    """A cell whose media differ in dispersion runs on a STOCK MEEP, through the INVERSION.

    ``_why_the_lookup_cannot_run`` is the seam that is forced: the declared-geometry
    lookup is the DEFAULT route, so without this the test would exercise it and
    quietly stop covering the inversion at all. The route taken is asserted, not
    assumed. Measured
    against CPU MEEP stepping the same cell (MEEP 1.29.0, the stock ``reference``
    environment):

        two_block    Ez/Hx/Hy  **1.00e-07 .. 1.50e-07**   dropped-terms control 1.7e-01 .. 2.0e-01
        shared_pole  Ez/Hx/Hy  **1.27e-07 .. 1.68e-07**   dropped-terms control 1.2e-01 .. 1.4e-01

    ``shared_pole`` is the enumeration case: its two blocks declare the same pole
    with different sigma, which MEEP merges into ONE chain entry (susceptibility_equiv
    ignores sigma). Deduplicating on anything finer would give two identical basis
    columns and a refused lift; deduplicating on the medium signature — which
    INCLUDES sigma — is exactly that mistake.

    The control is the same reference against a run with the susceptibilities
    dropped, six orders away, because a lift that silently installed no dispersion
    at all would still return a smooth complete field.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import (
        SIGMA_ROUTE_INVERSION,
        _materials_of,
        _unique_susceptibilities,
        gpu_compatibility,
        run_on_gpu,
    )

    monkeypatch.setattr(
        from_meep, "_why_the_lookup_cannot_run",
        lambda mp_module, simulation, media: "forced off so this test covers the inversion")

    build = _two_block_cell if case == "two_block" else _shared_pole_cell
    expected_terms = 2 if case == "two_block" else 1
    assert len(_unique_susceptibilities(_materials_of(mp, build(mp)))) == expected_terms, (
        f"{case}: the recovery would solve for the wrong number of unknowns"
    )

    verdict = gpu_compatibility(build(mp))
    assert verdict.supported, f"{case}: the gate refused a recoverable cell: {verdict.reasons}"

    result = run_on_gpu(build(mp), until=6.0, prefer_gpu=False)
    try:
        assert result.sigma_lift.route == SIGMA_ROUTE_INVERSION, (
            f"{case}: this test means to cover the chi1inv inversion and the lift took "
            f"{result.sigma_lift}"
        )
        ours = {name: np.asarray(result.get_array(name)) for name in ("Ez", "Hx", "Hy")}
    finally:
        result.close()

    reference = build(mp)
    reference.run(until=6.0)
    control = build(mp)
    for obj in control.geometry:
        obj.material = mp.Medium(epsilon=obj.material.epsilon_diag.x)
    control.run(until=6.0)

    for name, mine in ours.items():
        theirs = np.asarray(reference.get_array(component=getattr(mp, name), cmplx=True))
        blind = np.asarray(control.get_array(component=getattr(mp, name), cmplx=True))
        error = float(np.linalg.norm(mine - theirs) / np.linalg.norm(theirs))
        dropped = float(np.linalg.norm(blind - theirs) / np.linalg.norm(theirs))
        print(f"[sigma_recovery] stock lift {case:12} {name} parity={error:.3e} "
              f"control={dropped:.3e}", flush=True)
        assert error < 5e-6, f"{case} {name}: recovered-dispersion lift {error:.3e} from CPU MEEP"
        assert dropped > 1e-2, (
            f"{case} {name}: dropping the susceptibilities entirely moves the field only "
            f"{dropped:.3e}, so this cell cannot tell a working dispersion lift from a missing one"
        )


# --- the gate ----------------------------------------------------------------------


@requires_meep
@skip_without_meep
def test_the_gate_refuses_a_zero_frequency_susceptibility_in_any_medium():
    """The recovery's own precondition, verified rather than trusted.

    ``sigma`` comes back as ``c/w0**2``, so a zero-frequency pole has no recovery at
    all. :func:`~.from_meep._check_sigma_is_recoverable` does not re-check it because
    :func:`~.from_meep._check_susceptibility` already refuses a non-positive frequency
    for EVERY medium in the cell — including, since the dispersion difference became
    liftable, the media that are not ``materials[0]``. That claim is what this test
    measures: the bad term is put in the SECOND block, which is only reached by the
    loop inside the accepted branch.
    """
    import meep as mp

    from .from_meep import gpu_compatibility

    geometry = [
        mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
        mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=2.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=0.0, gamma=0.1, sigma=0.7)])),
    ]
    sim = mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=10, geometry=geometry,
                        force_complex_fields=True,
                        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                                           center=mp.Vector3(0, -1.2, 0))])
    verdict = gpu_compatibility(sim)
    joined = "\n".join(verdict.reasons)
    assert not verdict.supported, "a zero-frequency susceptibility was accepted"
    assert "media[2].E_susceptibilities[0]" in joined, joined
    assert "must be finite " in joined and "positive" in joined, joined


@requires_meep
@skip_without_meep
def test_a_conductivity_shared_by_media_that_differ_in_dispersion_is_refused():
    """``D_conductivity`` scales the sampled permittivity by a factor the basis has no column for.

    monitor.cpp:340-344 multiplies eps by ``1 + i*sigma_D/f`` AFTER summing the
    susceptibilities, so every recovery sample is off by that factor. The held-out
    cross-check would refuse it, but for the wrong stated reason, so the gate names it
    instead, and the refusal ends with an instruction the user can follow on any MEEP.
    """
    import meep as mp

    from .from_meep import gpu_compatibility

    def build():
        geometry = [
            mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                     material=mp.Medium(epsilon=4.0, D_conductivity=0.2, E_susceptibilities=[
                         mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
            mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                     material=mp.Medium(epsilon=2.0, D_conductivity=0.2, E_susceptibilities=[
                         mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)])),
        ]
        return mp.Simulation(
            cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=10, geometry=geometry,
            default_material=mp.Medium(D_conductivity=0.2), force_complex_fields=True,
            sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                               center=mp.Vector3(0, -1.2, 0))])

    verdict = gpu_compatibility(build())
    joined = "\n".join(verdict.reasons)
    assert not verdict.supported, "a conductivity was accepted on the recovery path"
    assert "D_conductivity_diag" in joined and "monitor.cpp" in joined, joined
    assert "Drop the conductivity, or run this simulation on CPU MEEP." in joined, joined
