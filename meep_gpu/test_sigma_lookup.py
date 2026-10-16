"""Tests for the DECLARED-GEOMETRY sigma lookup and the per-point check under it.

Three layers, in the order a defect would reach them:

1. **The verification algebra alone**, with no MEEP: the columns against MEEP's
   own ``chi1`` closed form retyped from susceptibility.cpp:297-307, a correct
   sigma passing, a wrong one failing, and the degeneracy the check cannot see
   through being refused rather than checked blind.
2. **The check FAILING when it should**, and the three cells that need the three
   images MEEP searches — all on stock MEEP. The defect the
   check is aimed at is not hypothetical: this engine's own Yee coordinate and the
   one MEEP snapped the point to differ by one ulp at a block face that no binary
   float represents, and that flips which medium 81 of 10800 points are in. Both
   halves are measured here — that the two coordinates really do disagree, and
   that substituting one for the other moves the check by seven orders. The
   refusal stock MEEP leaves is pinned here too, on both routes: a smoothed curved
   interface anywhere in the cell, with the escape it names shown to work, and a
   declared ``epsilon_offdiag``, which that escape does not clear.
3. **The lift end to end on a stock MEEP**, against CPU MEEP stepping the same
   cell, plus the route record that says which of the two paths ran.

THE HONEST COMPARISON. The lookup is NOT cheaper than the inversion at ANY object
count, including zero — measured per call on this machine, ``mp.is_point_in_object``
costs 4.38 us against ``get_chi1inv``'s 0.64 us at a nonzero frequency, and end to
end the lookup runs 1.4-2.0x the inversion at one or two objects and 14.6x at
thirty-two. What it is, is exact: it installs the declared number rather than a
solved one. Measured on 2026-08-04 against MEEP's stored sigma words, point by
point: two-block Lorentz+Drude at resolution 10, lookup 1.70e-08 and inversion
5.67e-08; ``meep.materials.Au`` (6 poles), lookup 3.28e-08 and inversion 1.35e-06.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from .sigma_lookup import (
    VERIFICATION_TOLERANCE,
    SigmaLookupRefused,
    plan_verification,
    sigma_of_medium,
)
from .sigma_recovery import CONDITION_LIMIT, SusceptibilityPole
from .test_sigma_recovery import meep_chi1, synthetic_epsilon

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)


# --- the verification algebra alone (no MEEP) --------------------------------------


def test_the_verification_columns_are_meeps_chi1_per_unit_sigma():
    """``columns[w, n] * sigma`` must BE MEEP's ``chi1``, for both pole kinds.

    The whole check is ``eps_inf + columns @ sigma`` compared against MEEP, so if
    the columns are not MEEP's own closed form the check measures the wrong
    function and can pass a wrong lookup. Written against the C++ retyped in
    ``test_sigma_recovery.meep_chi1`` rather than against the module under test.

    Note the parameterization difference from :mod:`~.sigma_recovery`, which is
    deliberate: the inversion SOLVES, so it carries ``c = sigma*w0**2`` to keep its
    basis conditionable; the lookup is handed sigma itself and only evaluates, so
    its columns are per unit sigma and no reparameterization is needed.
    """
    poles = [
        SusceptibilityPole(frequency=1.0, gamma=0.1),
        SusceptibilityPole(frequency=2.0, gamma=0.05, drude=True),
        SusceptibilityPole(frequency=10.7433, gamma=1.78571),
    ]
    check = plan_verification(poles)
    for index, pole in enumerate(poles):
        for sigma in (0.4, 7.0, 4.0314e21):
            for f_index, f in enumerate(check.frequencies):
                mine = check._columns[f_index, index] * sigma
                theirs = meep_chi1(pole.frequency, pole.gamma, pole.drude, sigma, float(f))
                error = abs(mine - theirs) / max(abs(theirs), 1e-300)
                assert error < 1e-14, (
                    f"column {index} at f={f:g} times sigma={sigma:g} is {mine!r} against "
                    f"MEEP's {theirs!r} (relative {error:.3e})"
                )


def test_a_correct_sigma_passes_and_a_wrong_one_is_caught_at_its_own_sensitivity():
    """The check's two jobs, and the number that converts one into the other.

    A sigma equal to the truth reproduces the synthesized permittivity to the
    solve's own noise. A sigma wrong by ``delta`` on ONE pole moves the residual by
    ``delta * sensitivity[n]`` — that is what
    :attr:`~.sigma_lookup.SigmaVerification.sensitivity` means — so the tolerance
    converts directly into the largest error that can pass. Measured here on a
    three-pole set: an error of 1e-6 in a sigma of 0.4 scores **6.5e-07** against
    the 1e-04 limit — accepted, and correctly so, since at that size it changes no
    field — and the reported sensitivity predicts 5.1e-07 for it.

    The third point is vacuum on purpose. It is where a lookup that resolved the
    wrong medium puts a whole sigma where MEEP holds zero, which is the defect this
    check exists for, and it scores **1.1e+00**.
    """
    poles = [
        SusceptibilityPole(frequency=1.0, gamma=0.1),
        SusceptibilityPole(frequency=2.0, gamma=0.05, drude=True),
        SusceptibilityPole(frequency=0.4, gamma=0.02),
    ]
    truth = np.array([
        [0.4, 0.0, 1.7],
        [0.0, 0.7, 0.0],
        [0.9, 0.0, 0.0],
    ])
    eps_inf = np.array([4.0, 2.0, 1.0])
    check = plan_verification(poles)
    measured = synthetic_epsilon(poles, truth, eps_inf, check.frequencies)

    exact = float(np.max(check.residuals(eps_inf, truth, measured)))
    assert exact < 1e-14, f"an exact sigma scored {exact:.3e} against its own synthesis"
    check.require(eps_inf, truth, measured, where="exact")  # Must not raise.

    small = truth.copy()
    small[0, 0] += 1e-6
    tiny = float(np.max(check.residuals(eps_inf, small, measured)))
    predicted = 1e-6 * float(check.sensitivity[0]) / 4.0  # eps_inf = 4 sets the scale.
    print(f"[sigma_lookup] 1e-6 sigma error scores {tiny:.3e}, sensitivity predicts "
          f"~{predicted:.3e}", flush=True)
    assert tiny < VERIFICATION_TOLERANCE, "a 1e-6 sigma error is refused; the check is too tight"
    assert tiny > 0.2 * predicted, (
        f"a 1e-6 error in sigma scored {tiny:.3e} where the reported sensitivity "
        f"{float(check.sensitivity[0]):.3e} predicts ~{predicted:.3e}; the sensitivity no longer "
        f"describes what the check can see"
    )

    wrong_medium = truth.copy()
    wrong_medium[:, 2] = truth[:, 0]  # The third point resolved to the first point's medium.
    worst = float(np.max(check.residuals(eps_inf, wrong_medium, measured)))
    print(f"[sigma_lookup] a wrong medium at one point scores {worst:.3e} "
          f"(limit {VERIFICATION_TOLERANCE:g})", flush=True)
    with pytest.raises(SigmaLookupRefused) as refusal:
        check.require(eps_inf, wrong_medium, measured, where="Ez",
                      locate=lambda index: f"point {index}")
    assert "point 2" in str(refusal.value), str(refusal.value)
    assert worst > 1e3 * VERIFICATION_TOLERANCE, (
        f"one point in the wrong medium moved the check by only {worst:.3e}; the two "
        f"populations are no longer separated"
    )


def test_two_indistinguishable_poles_are_refused_rather_than_checked_blind():
    """Where an error in one sigma is cancellable by an error in the other, refuse.

    The check evaluates rather than solves, so it is normally immune to
    conditioning — but not to outright degeneracy: two poles with the same shape
    over the band admit a whole line of sigma vectors reproducing the same
    permittivity, and the residual stays at zero along it. That is the one thing
    this check cannot see, so the pole set is refused instead of being checked.
    """
    with pytest.raises(SigmaLookupRefused) as refusal:
        plan_verification([
            SusceptibilityPole(frequency=1.0, gamma=0.1),
            SusceptibilityPole(frequency=1.0, gamma=0.1),
        ])
    assert "indistinguishable" in str(refusal.value)
    assert f"{CONDITION_LIMIT:.0e}" in str(refusal.value)


def test_the_degeneracy_test_reads_pole_SHAPES_not_pole_MAGNITUDES():
    """``meep.materials.Au`` must pass, and it only does if the columns are normalized.

    MEEP's Drude idiom pairs ``f = 1e-10`` with ``sigma = 4.03e+21`` beside Lorentz
    sigmas of order 1, so the raw per-sigma columns span forty orders and their
    condition number reports that spread rather than any confusability. Normalizing
    each column asks the question that matters — are these poles distinguishable
    over the band, and lands at **5.2e+00** — eight orders inside the limit,
    where the raw columns would report a condition number past 1e+40.
    """
    meep_module = pytest.importorskip("meep") if not MEEP_MISSING else None
    del meep_module
    poles = [
        SusceptibilityPole(frequency=1e-10, gamma=0.0427474, drude=True),
        SusceptibilityPole(frequency=0.33472, gamma=0.19438),
        SusceptibilityPole(frequency=0.66944, gamma=0.27826),
        SusceptibilityPole(frequency=2.39310, gamma=0.7017),
        SusceptibilityPole(frequency=3.42200, gamma=2.0115),
        SusceptibilityPole(frequency=10.7433, gamma=1.78571),
    ]
    check = plan_verification(poles)
    print(f"[sigma_lookup] Au verification cond={check.condition_number:.3e} "
          f"n={check.sample_count}", flush=True)
    assert check.condition_number < CONDITION_LIMIT, (
        f"the six-pole gold basis is reported at condition {check.condition_number:.3e}, past the "
        f"{CONDITION_LIMIT:.0e} limit — the column normalization has been dropped and every real "
        f"metal is now refused"
    )


def test_sigma_of_medium_reproduces_meepgeoms_row_selection():
    """``geom_epsilon::sigma_row``: the matching entry's diagonal, or zero.

    Three things at once, all of them meepgeom.cpp:1680-1705 — the row selected is
    ``component_index(c)``, the entry is found by the equivalence key rather than by
    position, and a medium that declares no equivalent entry contributes an exact
    zero rather than being skipped.
    """
    def key_of(term):
        return (term.kind, term.frequency)

    def term(kind, frequency, sigma):
        return SimpleNamespace(
            kind=kind, frequency=frequency,
            sigma_diag=SimpleNamespace(x=sigma[0], y=sigma[1], z=sigma[2]),
        )

    keys = [("lorentz", 1.0), ("drude", 2.0), ("lorentz", 5.0)]
    medium = SimpleNamespace(E_susceptibilities=[
        term("drude", 2.0, (0.7, 0.8, 0.9)),      # Declared SECOND in the chain order.
        term("lorentz", 1.0, (0.1, 0.2, 0.3)),
    ])
    assert list(sigma_of_medium(medium, keys, 0, key_of)) == [0.1, 0.7, 0.0]
    assert list(sigma_of_medium(medium, keys, 1, key_of)) == [0.2, 0.8, 0.0]
    assert list(sigma_of_medium(medium, keys, 2, key_of)) == [0.3, 0.9, 0.0]
    vacuum = SimpleNamespace(E_susceptibilities=[])
    assert list(sigma_of_medium(vacuum, keys, 2, key_of)) == [0.0, 0.0, 0.0]


# --- shared cells ------------------------------------------------------------------


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


def _gold_cell(mp, resolution: int = 20):
    """``meep.materials.Au`` flush with the LOW x face — six poles and the ownership trap.

    The block runs from x = -0.5 to 0.0 in a cell that ends at x = -0.5, so its low
    face lands on the one lattice plane per axis that ``_lattice_indices`` enumerates
    and MEEP does not own. MEEP answers vacuum-with-no-susceptibility there whatever
    the geometry says; without the ``gv.owns`` gate the lookup takes the declared
    sigma instead, and 21 of 1281 points disagree with MEEP's stored sigma.
    """
    from meep.materials import Au
    return mp.Simulation(cell_size=mp.Vector3(1.0, 1.0, 0.0), resolution=resolution,
                         geometry=[mp.Block(center=mp.Vector3(-0.25, 0, 0),
                                            size=mp.Vector3(0.5, 0.5, mp.inf), material=Au)],
                         force_complex_fields=True,
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez, center=mp.Vector3(0.3, 0.3, 0))])


def _unrepresentable_face_cell(mp, resolution: int = 30):
    """A dispersive block whose faces sit at +-1/3, which no binary float represents.

    dx = 1/30, so the faces land ON Yee points — and "on" is decided by
    ``fabs(proj) <= 0.5*size`` at the last bit. This is the cell where MEEP's
    snapped coordinate and this engine's own differ by one ulp at 81 of 10800
    points, which is the whole reason :func:`~.from_meep._yee_point_location` reads
    the location off MEEP instead of recomputing it.
    """
    geometry = [
        mp.Block(center=mp.Vector3(-1.0 / 3.0, 0, 0),
                 size=mp.Vector3(2.0 / 3.0, 2.0 / 3.0, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
    ]
    return mp.Simulation(cell_size=mp.Vector3(2.0, 2.0, 0.0), resolution=resolution,
                         geometry=geometry, force_complex_fields=True,
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez,
                                            center=mp.Vector3(0.5, 0.5, 0))])


def _wrapped_object_cell(mp, resolution: int = 10):
    """A dispersive block CROSSING the high x face of a ``k_point`` cell.

    Two periodic mechanisms at once, and the lookup needs both:

    * libctl replicates each object by the lattice vectors when
      ``ensure_periodicity and k_point`` (geom.c ``LOOP_PERIODIC``), so the OWNED
      points just inside the low face are covered only by the object's wrapped
      copy, which the declared list does not contain.
    * MEEP wraps a lattice point below the low corner to the high face before it
      looks for an owning chunk (``locate_point_in_user_volume``), so the low plane
      — which ``_lattice_indices`` enumerates and no chunk owns — reads the
      material at the high face rather than vacuum.

    Gating on ownership alone scores 1.2e-01 on this cell.
    """
    geometry = [
        mp.Block(center=mp.Vector3(1.9, 0, 0), size=mp.Vector3(1.0, 2.0, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
    ]
    return mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=resolution,
                         geometry=geometry, force_complex_fields=True,
                         k_point=mp.Vector3(0.1, 0, 0),
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez,
                                            center=mp.Vector3(0, -1.2, 0))])


def _mirrored_cell(mp, resolution: int = 10):
    """The two-block cell folded by ``mp.Mirror(mp.Y)`` — the symmetry-image case.

    MEEP stores only the folded half, and ``_lattice_indices`` enumerates only that
    half — including its low plane, which the folded grid_volume does not own.
    MEEP reads that plane through the symmetry transform (a negation of the lattice
    coordinate, since ``geometry_center`` is the origin), not as vacuum; treating
    it as vacuum scores 1.2e+00.
    """
    geometry = [
        mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
        mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=2.0, E_susceptibilities=[
                     mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)])),
    ]
    return mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=resolution,
                         geometry=geometry, symmetries=[mp.Mirror(mp.Y)],
                         force_complex_fields=True,
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez, center=mp.Vector3(0, 0, 0))])


def _rounded_periodic_cell(mp, resolution: int = 10):
    """A ``k_point`` cell that is not a whole number of pixels, a dispersive slab on its low face.

    stochastic_emitter_reciprocity.py's arrangement, small. 4.56 at resolution 10 is
    45.6 pixels, which MEEP rounds to 46, and libctl's periodic copies are shifted by
    the rounded grid's size (``make_geom_epsilon``, meepgeom.cpp:1956-1985: 4.6 here),
    not by the declared 4.56. The slab runs from the declared low face y = -2.28 to
    -1.78, so its copy shifted by 4.56 would cover [2.28, 2.78] and claim the rounded
    grid's top face y = 2.3, which MEEP's copy at [2.32, 2.82] leaves as vacuum.
    """
    slab = mp.Medium(epsilon=2.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4),
        mp.DrudeSusceptibility(frequency=0.5, gamma=0.05, sigma=3.0)])
    return mp.Simulation(cell_size=mp.Vector3(1.0, 4.56, 0.0), resolution=resolution,
                         geometry=[mp.Block(center=mp.Vector3(0, -2.03, 0),
                                            size=mp.Vector3(mp.inf, 0.5, mp.inf), material=slab)],
                         k_point=mp.Vector3(),
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez, center=mp.Vector3(0, 0.5, 0))])


def _face_crossing_cell(mp, resolution: int = 15):
    """test_dump_load.py's 3-D cell in 2-D: a dispersive block across the periodic x face.

    2.3 x 2.1 at resolution 15 grids 35 x 32, so MEEP's lattice vector along x is
    35/15 = 2.3333333333333335 and the block's copy ends at 1.5 - 2.3333333333333335 =
    -0.8333333333333335, one ulp below the Yee point x = -25/30 = -0.8333333333333334.
    libctl's bounding-box gate keeps that point out of the copy; the object test, run on
    the point less the shift, rounds onto the block's face 1.5 and lets it in.
    """
    block = mp.Medium(epsilon=3.1, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.5, gamma=0.2, sigma=0.9)])
    return mp.Simulation(cell_size=mp.Vector3(2.3, 2.1, 0.0), resolution=resolution,
                         geometry=[mp.Block(center=mp.Vector3(0, 0, 0), size=mp.Vector3(1, 1, mp.inf),
                                            material=mp.Medium(index=2.3)),
                                   mp.Block(center=mp.Vector3(1, 0, 0), size=mp.Vector3(1, 1, mp.inf),
                                            material=block)],
                         default_material=mp.Medium(index=1.4), k_point=mp.Vector3(0.4, -1.3, 0),
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Hz,
                                            center=mp.Vector3(-0.2, 0.3, 0))])


def _one_d_wrapped_cell(mp, resolution: int = 10):
    """A 1-D ``k_point`` cell with a dispersive slab across its periodic z face.

    The slab runs from z = 1.65 to 2.15 in a cell of 4 (40 pixels), so its part past
    the high face z = 2 belongs, in libctl's tree, to the copy shifted by -4, which
    covers the owned points from the low face z = -2 up to -1.85. MEEP's D1 lattice is
    ``(0, 0, 4)`` and libctl's ``LOOP_PERIODIC`` runs its three-dimensional case there
    (``geom_initialize`` leaves ``dimensions = 3``), so those copies lie along z.
    """
    slab = mp.Medium(epsilon=2.2, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.1, gamma=0.1, sigma=0.7)])
    return mp.Simulation(cell_size=mp.Vector3(0, 0, 4.0), resolution=resolution, dimensions=1,
                         geometry=[mp.Block(center=mp.Vector3(0, 0, 1.9),
                                            size=mp.Vector3(mp.inf, mp.inf, 0.5), material=slab)],
                         k_point=mp.Vector3(),
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ex, center=mp.Vector3(0, 0, 0.3))])


def _curved_interface_cell(mp, curve_is_dispersive: bool, eps_averaging: bool,
                           resolution: int = 10):
    """Two dispersive media and a cylinder whose curved face subpixel averaging smooths.

    With ``curve_is_dispersive`` the cylinder is the Lorentz medium. Without it the
    dispersion sits on two axis-aligned blocks and the cylinder is a plain
    dielectric, so the only curved interface in the cell bounds no susceptibility.
    """
    lorentz = mp.Medium(epsilon=4.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])
    drude = mp.Medium(epsilon=2.0, E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)])
    if curve_is_dispersive:
        geometry = [
            mp.Cylinder(radius=0.73, height=mp.inf, center=mp.Vector3(-1.0, 0, 0),
                        material=lorentz),
            mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                     material=drude),
        ]
    else:
        geometry = [
            mp.Block(center=mp.Vector3(-1.0, 0.8, 0), size=mp.Vector3(1.4, 1.2, mp.inf),
                     material=lorentz),
            mp.Block(center=mp.Vector3(1.0, 0.8, 0), size=mp.Vector3(1.4, 1.2, mp.inf),
                     material=drude),
            mp.Cylinder(radius=0.5, height=mp.inf, center=mp.Vector3(0, -1.0, 0),
                        material=mp.Medium(epsilon=3.0)),
        ]
    return mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=resolution,
                         geometry=geometry, force_complex_fields=True,
                         eps_averaging=eps_averaging,
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez,
                                            center=mp.Vector3(0, 1.8, 0))])


def _anisotropic_unsmoothed_cell(mp, resolution: int = 10):
    """Two dispersive axis-aligned blocks, ``eps_averaging=False``, off-diagonal rows anyway.

    A third, non-dispersive block declares ``epsilon_offdiag``, so MEEP's tensor carries
    off-diagonal entries there with subpixel averaging off and every shape an
    axis-aligned block.
    """
    lorentz = mp.Medium(epsilon=4.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])
    drude = mp.Medium(epsilon=2.0, E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)])
    anisotropic = mp.Medium(epsilon_diag=mp.Vector3(3.0, 3.0, 3.0),
                            epsilon_offdiag=mp.Vector3(0.5, 0, 0))
    return mp.Simulation(
        cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=resolution,
        force_complex_fields=True, eps_averaging=False,
        geometry=[mp.Block(center=mp.Vector3(-1.0, 0.8, 0), size=mp.Vector3(1.4, 1.2, mp.inf),
                           material=lorentz),
                  mp.Block(center=mp.Vector3(1.0, 0.8, 0), size=mp.Vector3(1.4, 1.2, mp.inf),
                           material=drude),
                  mp.Block(center=mp.Vector3(0, -1.0, 0), size=mp.Vector3(1.0, 1.0, mp.inf),
                           material=anisotropic)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                           center=mp.Vector3(0, 1.8, 0))])


def _overlapping_cell(mp, resolution: int = 10):
    """Two dispersive blocks that overlap, so object PRECEDENCE decides the middle.

    libctl stores the tree in reverse declaration order (geom.c:1579), which is what
    MEEP's manual means by "later objects in the list take precedence". Searching
    forwards instead puts the first block's sigma in the overlap, which is a
    different structure everywhere the two meet.
    """
    geometry = [
        mp.Block(center=mp.Vector3(-0.5, 0, 0), size=mp.Vector3(2.0, 2.0, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
        mp.Block(center=mp.Vector3(0.5, 0, 0), size=mp.Vector3(2.0, 2.0, mp.inf),
                 material=mp.Medium(epsilon=2.0, E_susceptibilities=[
                     mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)])),
    ]
    return mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=resolution,
                         geometry=geometry, force_complex_fields=True,
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez,
                                            center=mp.Vector3(0, -1.2, 0))])


# --- the check firing when it should (any MEEP) ------------------------------------


@requires_meep
@skip_without_meep
def test_this_engines_own_yee_coordinate_differs_from_meeps_snap_at_an_unrepresentable_face():
    """THE TRAP, measured before it is injected: the two coordinates are not equal.

    ``_lattice_indices`` proves this engine's Yee coordinates sit on MEEP's integer
    lattice to within 1e-7 of a half cell, and it would be easy to read that as "so
    either coordinate will do". It will not. Containment is
    ``fabs(proj) <= 0.5*size`` at the last bit (libctl geom.c:305), so on a block
    whose faces are at +-1/3 with dx = 1/30 the two coordinates put **81 of 10800**
    Yee points in different media — measured here, not asserted from a docstring.

    This test measures the DISAGREEMENT. The one below measures what the check does
    about it.
    """
    import meep as mp

    from .backends import to_numpy
    from .dispersion import component_coordinates
    from .from_meep import _lattice_indices, _yee_point_location, lift_simulation

    sim = _unrepresentable_face_cell(mp)
    driver = lift_simulation(sim, prefer_gpu=False)
    try:
        gv = sim.fields.gv
        grid = driver.grid
        obj = sim.geometry[0]
        lattice_per_unit = 2.0 * float(sim.resolution)
        flips = 0
        total = 0
        for name, component in (("Ex", mp.Ex), ("Ey", mp.Ey), ("Ez", mp.Ez)):
            indices = _lattice_indices(gv, grid, name, component, lattice_per_unit)
            nominal = [np.asarray(to_numpy(axis), dtype=np.float64)
                       for axis in component_coordinates(grid, name)]
            for i, ix in enumerate(indices[0]):
                for j, jy in enumerate(indices[1]):
                    location = mp.ivec(int(ix), int(jy), int(indices[2][0]))
                    snapped = _yee_point_location(mp, gv, component, location)
                    ours = (float(nominal[0][i]), float(nominal[1][j]), snapped[2])
                    total += 1
                    if bool(mp.is_point_in_object(mp.Vector3(*snapped), obj)) != bool(
                            mp.is_point_in_object(mp.Vector3(*ours), obj)):
                        flips += 1
        print(f"[sigma_lookup] snapped vs nominal coordinate: {flips}/{total} containment flips",
              flush=True)
        assert flips > 0, (
            f"MEEP's snapped Yee coordinate and this engine's own now agree on containment at "
            f"all {total} points of a cell built specifically to separate them. Either the grid "
            f"arithmetic changed or the block faces landed on representable coordinates; find a "
            f"cell where they still differ, because the lookup's choice of coordinate is only "
            f"load-bearing while they do"
        )
    finally:
        driver.close()


@requires_meep
@skip_without_meep
def test_the_check_catches_a_lookup_at_the_nominal_coordinate_and_the_lift_falls_back(monkeypatch):
    """THE CHECK DOING ITS JOB, on the defect the test above measures.

    ``_yee_point_location`` is nudged by exactly one ulp — the size of the
    disagreement measured above — and the lookup's per-point check must refuse
    rather than install a structure that differs from MEEP's at the interface
    cells. Measured: the honest lookup scores **1.88e-08** on this cell and the
    nudged one **1.76e-01**, seven orders apart across a 1e-04 limit.

    Then the whole lift is run with the same nudge in place, and it must still
    produce a driver — through the chi1inv inversion, which shares none of the
    lookup's dependencies — with the lookup's measured refusal recorded on
    ``driver.sigma_lift.fallback_reason``. That is the difference between a
    fallback and a silent degrade: the number that caused it survives.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import (
        SIGMA_ROUTE_INVERSION,
        _lookup_sigma_volumes,
        _materials_of,
        _yee_point_location,
        lift_simulation,
    )

    sim = _unrepresentable_face_cell(mp)
    driver = lift_simulation(sim, prefer_gpu=False)
    try:
        assert driver.sigma_lift.route == from_meep.SIGMA_ROUTE_LOOKUP, driver.sigma_lift
        honest = driver.sigma_lift.worst_residual
        materials = _materials_of(mp, sim)

        def nudged(mp_module, gv, component, location):
            return tuple(float(np.nextafter(value, -np.inf))
                         for value in _yee_point_location(mp_module, gv, component, location))

        monkeypatch.setattr(from_meep, "_yee_point_location", nudged)
        with pytest.raises(SigmaLookupRefused) as refusal:
            _lookup_sigma_volumes(mp, sim, driver, materials)
        message = str(refusal.value)
        print(f"[sigma_lookup] honest lookup {honest:.3e}, one-ulp nudge refused: "
              f"{message.splitlines()[0]}", flush=True)
        assert "worst relative disagreement" in message and "lattice point" in message
        assert honest < VERIFICATION_TOLERANCE
    finally:
        driver.close()

    fallen_back = lift_simulation(_unrepresentable_face_cell(mp), prefer_gpu=False)
    try:
        record = fallen_back.sigma_lift
        assert record.route == SIGMA_ROUTE_INVERSION, (
            f"with the lookup refusing, the lift produced {record}; it should have fallen back to "
            f"the inversion, which uses no geometry at all"
        )
        assert "worst relative disagreement" in record.fallback_reason, (
            f"the fallback recorded no measured reason: {record.fallback_reason!r}"
        )
    finally:
        fallen_back.close()


@requires_meep
@skip_without_meep
def test_a_structure_flush_with_the_cell_face_needs_meeps_own_ownership_gate():
    """THE UNOWNED PLANE: a structure flush with the plane MEEP does not own.

    ``_lattice_indices`` enumerates the full closed lattice, and one plane per axis
    of it lies outside the region MEEP owns — ``little_owned_corner`` is
    ``little_corner + 2 - iyee_shift``, so Ez's lowest x plane is not owned. There
    MEEP answers vacuum-with-no-susceptibility at every frequency
    (monitor.cpp:182-184), whatever the geometry says. The gold block is placed
    flush with that plane precisely so the two answers differ: with the ``gv.owns``
    gate the lookup agreed with MEEP's stored sigma at 0 of 1281 points wrong,
    without it at 21 (measured 2026-08-04).

    Dropping the gate makes the lookup's own per-point check fail — it would be
    predicting Au's permittivity where MEEP reports vacuum — so the lift falls back
    to the inversion and the route stops being the lookup.
    """
    import meep as mp

    from .from_meep import SIGMA_ROUTE_LOOKUP, lift_simulation

    driver = lift_simulation(_gold_cell(mp), prefer_gpu=False)
    try:
        record = driver.sigma_lift
        print(f"[sigma_lookup] gold flush with the low face -> {record}", flush=True)
        assert record.route == SIGMA_ROUTE_LOOKUP, (
            f"the flush-face gold cell did not lift through the lookup: {record}. The ownership "
            f"gate is what makes it possible — without it the lookup claims Au's sigma on the "
            f"plane MEEP does not own and reports vacuum for"
        )
        assert record.worst_residual < VERIFICATION_TOLERANCE
    finally:
        driver.close()


@requires_meep
@skip_without_meep
def test_a_periodic_cell_wraps_both_the_objects_and_the_lattice_point():
    """The two periodic mechanisms, on a stock MEEP, on one cell that needs both.

    ``_wrapped_object_cell`` puts a dispersive block across the high x face of a
    ``k_point`` run. libctl replicates the object by the lattice vector, so the
    owned points just inside the LOW face are covered only by that copy; and MEEP
    wraps the low PLANE (which no chunk owns) to the high face before reading, so
    it is the block's material rather than vacuum. Gating on ``gv.owns`` alone —
    which is right on every metallic cell and was this route's first spelling —
    scores **1.2e-01** here and sends the cell to the inversion. With both wraps it
    lifts through the lookup at **1.88e-08**.
    """
    import meep as mp

    from .from_meep import SIGMA_ROUTE_LOOKUP, lift_simulation

    driver = lift_simulation(_wrapped_object_cell(mp), prefer_gpu=False)
    try:
        record = driver.sigma_lift
        print(f"[sigma_lookup] object across the periodic face -> {record}", flush=True)
        assert record.route == SIGMA_ROUTE_LOOKUP, (
            f"a periodic cell with an object across the cell face did not lift through the "
            f"lookup: {record}"
        )
        assert record.worst_residual < VERIFICATION_TOLERANCE
    finally:
        driver.close()


@requires_meep
@skip_without_meep
def test_the_periodic_copies_are_shifted_by_meeps_rounded_lattice_not_the_declared_cell(monkeypatch):
    """libctl's periodic copies sit one ROUNDED cell apart, and the lookup follows them.

    ``_rounded_periodic_cell`` declares 4.56 and MEEP grids 4.6. Shifted by the
    declared size, the low slab's copy claims the rounded grid's top face, which
    MEEP holds as vacuum: the per-point check refused the whole cell there, and the
    chi1inv inversion installed sigma volumes that are not MEEP's float32 words
    (measured on stochastic_emitter_reciprocity.py: five of six Ag terms off by 1 to
    0x12f ulps, which a subnormal-range polarization then showed as a 1-ulp E word at
    step 1387). Shifted by MEEP's lattice it lifts through the lookup, and every
    installed sigma word is the declared sigma rounded once, as MEEP stores it. The
    control re-installs the declared size and must fall back.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import SIGMA_ROUTE_INVERSION, SIGMA_ROUTE_LOOKUP, lift_simulation

    mp.verbosity(0)
    sim = _rounded_periodic_cell(mp)
    driver = lift_simulation(sim, prefer_gpu=False)
    try:
        record = driver.sigma_lift
        print(f"[sigma_lookup] rounded periodic cell -> {record}", flush=True)
        assert record.route == SIGMA_ROUTE_LOOKUP, record
        assert record.worst_residual < VERIFICATION_TOLERANCE
        declared = sorted(np.float32(term.sigma_diag.z).view(np.uint32)
                          for term in sim.geometry[0].material.E_susceptibilities)
        installed = sorted({int(value) for state in driver.fields.polarizations
                            for value in np.unique(np.asarray(state.sigma["Ez"]).view(np.uint32))
                            if value})
        assert installed == [int(word) for word in declared], (installed, declared)
    finally:
        driver.close()

    declared_size = from_meep._vector3(sim.cell_size)
    monkeypatch.setattr(from_meep, "_geometry_lattice_size", lambda simulation: declared_size)
    control = lift_simulation(_rounded_periodic_cell(mp), prefer_gpu=False)
    try:
        assert control.sigma_lift.route == SIGMA_ROUTE_INVERSION, (
            f"with the copies shifted by the declared cell the lookup must decline here, or "
            f"this test cannot see the defect it pins: {control.sigma_lift}")
    finally:
        control.close()


@requires_meep
@skip_without_meep
def test_a_1d_periodic_cell_takes_libctls_copies_along_z():
    """libctl's periodic copies run its three-dimensional loop in every MEEP dimension.

    ``make_geom_epsilon`` calls ``geom_initialize``, which sets libctl's ``dimensions``
    to 3, and MEEP's ``set_dimensions`` then sets MEEP's own ``dim``, not libctl's: so a
    1-D cell's copies are the three along z of its ``(0, 0, nz/resolution)`` lattice.
    ``_one_d_wrapped_cell`` holds the slab at z = -1.95 only through the copy shifted
    by -4. MEEP's own permittivity there is the slab's; the search finds the slab; the
    control, libctl's one-dimensional case (three copies along x, all at the origin in
    a 1-D lattice), finds the default medium. Asked of the search directly: in a 1-D cell
    the lift's lookup also checks Ey and Ez, which MEEP holds no permittivity for, so the
    lift itself takes the inversion there whatever the copies are.
    """
    import meep as mp

    from . import from_meep

    mp.verbosity(0)
    sim = _one_d_wrapped_cell(mp)
    sim.init_sim()
    point = (0.0, 0.0, -1.95)
    meep_epsilon = complex(sim.get_epsilon_point(mp.Vector3(*point), frequency=0))
    print(f"[sigma_lookup] 1-D cell: MEEP's permittivity at z = -1.95 -> {meep_epsilon}", flush=True)
    assert meep_epsilon.real > 1.5, f"MEEP holds the slab's copy at z = -1.95: {meep_epsilon}"
    objects = tuple(sim.geometry)
    shifts = from_meep._lattice_shifts(mp, sim)
    assert from_meep._declared_material_at(mp, sim, point, objects, shifts) is sim.geometry[0].material
    size = from_meep._geometry_lattice_size(sim)
    one_dimensional_case = tuple((i * size[0], 0.0, 0.0) for i in (-1, 0, 1))
    assert from_meep._declared_material_at(mp, sim, point, objects, one_dimensional_case) is (
        sim.default_material), ("with libctl's one-dimensional case the search must miss the copy, "
                                "or this test cannot see the defect it pins")


@requires_meep
@skip_without_meep
def test_the_lattice_shifts_are_libctls_three_dimensional_loop_in_every_dimension():
    """The copies, in order, for a 1-D, a 2-D and a cylindrical ``k_point`` cell.

    ``LOOP_PERIODIC``'s three-dimensional case (geom.c:502-518): x outermost, then y,
    then z, each over -1, 0, 1, an axis of zero lattice length stepped once at -1
    (a shift of -0.0). Read from MEEP's own structure, so the lattice is the rounded
    grid's.
    """
    import meep as mp

    from .from_meep import _lattice_shifts

    mp.verbosity(0)
    source = [mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez, center=mp.Vector3())]
    one_d = mp.Simulation(cell_size=mp.Vector3(0, 0, 4.0), resolution=10, dimensions=1,
                          k_point=mp.Vector3(),
                          sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ex,
                                             center=mp.Vector3())])
    two_d = mp.Simulation(cell_size=mp.Vector3(2.0, 3.0, 0), resolution=10, k_point=mp.Vector3(),
                          sources=source)
    cylinder = mp.Simulation(cell_size=mp.Vector3(2.0, 0, 3.0), resolution=10,
                             dimensions=mp.CYLINDRICAL, m=1, k_point=mp.Vector3(),
                             sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                                                center=mp.Vector3(0.6, 0, 0.2))])
    expected = {
        "1-D": [(-0.0, -0.0, -4.0), (-0.0, -0.0, 0.0), (-0.0, -0.0, 4.0)],
        "2-D": [(i * 2.0, j * 3.0, -0.0) for i in (-1, 0, 1) for j in (-1, 0, 1)],
        "cylindrical": [(i * 2.0, -0.0, k * 3.0) for i in (-1, 0, 1) for k in (-1, 0, 1)],
    }
    for name, sim in (("1-D", one_d), ("2-D", two_d), ("cylindrical", cylinder)):
        sim.init_sim()
        shifts = [tuple(float(v) for v in shift) for shift in _lattice_shifts(mp, sim)]
        print(f"[sigma_lookup] {name} lattice shifts -> {shifts}", flush=True)
        assert shifts == expected[name], (name, shifts)
        signs = [tuple(np.signbit(shift)) for shift in shifts]
        assert signs == [tuple(np.signbit(shift)) for shift in expected[name]], (name, signs)


@requires_meep
@skip_without_meep
def test_the_bounding_box_gate_decides_a_point_on_a_periodic_copys_face(monkeypatch):
    """libctl's tree search gates each copy on its bounding box before the object test.

    ``_face_crossing_cell`` puts a Yee point one ulp outside a periodic copy's box while
    the object test, evaluated on the point less the shift, rounds onto the block's face
    and calls it inside. MEEP point-samples the default medium there; without the gate
    the lookup takes the block's sigma, the per-point check refuses the cell (measured
    on test_dump_load.py's 3-D cell: a relative 1.2 at lattice point (-25, -14, 14)), and
    the chi1inv inversion installs sigma words that are not MEEP's. The control drops
    the gate and must fall back.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import SIGMA_ROUTE_INVERSION, SIGMA_ROUTE_LOOKUP, lift_simulation

    mp.verbosity(0)
    driver = lift_simulation(_face_crossing_cell(mp), prefer_gpu=False)
    try:
        record = driver.sigma_lift
        print(f"[sigma_lookup] block across the periodic face -> {record}", flush=True)
        assert record.route == SIGMA_ROUTE_LOOKUP, record
        assert record.worst_residual < VERIFICATION_TOLERANCE
    finally:
        driver.close()

    monkeypatch.setattr(from_meep, "_libctl_bounding_box", lambda module, obj: None)
    control = lift_simulation(_face_crossing_cell(mp), prefer_gpu=False)
    try:
        assert control.sigma_lift.route == SIGMA_ROUTE_INVERSION, (
            f"without the bounding-box gate the lookup must decline here, or this test "
            f"cannot see the defect it pins: {control.sigma_lift}")
    finally:
        control.close()


@requires_meep
@skip_without_meep
def test_the_looser_cylinder_box_holds_every_point_the_object_test_admits():
    """A cylinder's box only spares object tests, so it must never exclude a point inside.

    Tilted cylinders, a cone and a wedge, sampled on a lattice through and around each
    (with points on the end faces and the rims): every point ``is_point_in_object``
    admits lies inside the box :func:`~.from_meep._libctl_bounding_box` gives.
    """
    import itertools

    import meep as mp

    from .from_meep import _libctl_bounding_box

    axis = mp.Vector3(1, 2, 3).unit()
    objects = [
        mp.Cylinder(radius=0.4, height=1.2, axis=axis, center=mp.Vector3(0.1, -0.2, 0.3)),
        mp.Cylinder(radius=0.5, height=mp.inf, center=mp.Vector3(0, 0, 0)),
        mp.Cone(radius=0.1, radius2=0.6, height=1.0, axis=mp.Vector3(1, 0, 1).unit()),
        mp.Wedge(radius=0.7, height=0.4, wedge_angle=2.0, axis=mp.Vector3(0, 0, 1)),
    ]
    admitted = 0
    for obj in objects:
        (lx, ly, lz), (hx, hy, hz) = _libctl_bounding_box(mp, obj)
        c = obj.center
        span = [v * 0.05 for v in range(-24, 25)]
        for dx, dy, dz in itertools.product(span, span, (-0.6, -0.5, -0.2, 0.0, 0.2, 0.5, 0.6)):
            point = mp.Vector3(c.x + dx, c.y + dy, c.z + dz)
            if mp.is_point_in_object(point, obj):
                admitted += 1
                assert lx <= point.x <= hx and ly <= point.y <= hy and lz <= point.z <= hz, (
                    type(obj).__name__, point)
    assert admitted > 1000, admitted


@requires_meep
@skip_without_meep
def test_a_mirror_folded_cell_reads_its_unowned_plane_through_the_symmetry_image():
    """The third image MEEP searches, on stock MEEP: the mirror.

    ``fields::get_chi1inv`` searches ``chunks[i]->gv.owns(S.transform(iloc, sn))``
    over every symmetry image, so on a folded cell the low plane of the FOLDED
    grid_volume — which ``_lattice_indices`` enumerates and no chunk owns — is read
    at its reflection rather than as vacuum. The reflection is a negation of the
    lattice coordinate because ``geometry_center`` is required to be the origin.
    Without it the lookup calls that plane vacuum and the per-point check scores
    **1.2e+00**; with it the cell lifts at **1.76e-07**.
    """
    import meep as mp

    from .from_meep import SIGMA_ROUTE_LOOKUP, lift_simulation

    driver = lift_simulation(_mirrored_cell(mp), prefer_gpu=False)
    try:
        record = driver.sigma_lift
        print(f"[sigma_lookup] mirror-folded cell -> {record}", flush=True)
        assert record.route == SIGMA_ROUTE_LOOKUP, (
            f"a mirror-folded cell did not lift through the lookup: {record}"
        )
        assert record.worst_residual < VERIFICATION_TOLERANCE
    finally:
        driver.close()


@requires_meep
@skip_without_meep
def test_a_crowded_cell_goes_to_the_inversion_before_the_lookup_pays_for_it():
    """The one dispatch rule that is about cost rather than correctness.

    The lookup asks ``is_point_in_object`` once per object per Yee point until one
    answers, so its cost grows with the object count where the inversion's does not
    — measured 1.4-2.0x the inversion at one or two objects, 5.2x at eight, 14.6x
    at thirty-two. :data:`~.from_meep._LOOKUP_MAX_OBJECTS` bounds that. Both routes
    are correct here, so the cell still lifts — through the inversion, with the
    reason recorded.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import SIGMA_ROUTE_INVERSION, lift_simulation

    count = from_meep._LOOKUP_MAX_OBJECTS + 1
    dispersive = mp.Medium(epsilon=4.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])
    geometry = [
        mp.Block(center=mp.Vector3(-1.8 + 0.2 * index, 0, 0),
                 size=mp.Vector3(0.14, 1.4, mp.inf), material=dispersive)
        for index in range(count)
    ]
    sim = mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=10, geometry=geometry,
                        force_complex_fields=True,
                        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                                           center=mp.Vector3(0, -1.2, 0))])
    driver = lift_simulation(sim, prefer_gpu=False)
    try:
        record = driver.sigma_lift
        print(f"[sigma_lookup] {count} objects -> {record}", flush=True)
        assert record.route == SIGMA_ROUTE_INVERSION, record
        assert str(count) in record.fallback_reason, record.fallback_reason
    finally:
        driver.close()


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("route", ["lookup", "inversion"])
@pytest.mark.parametrize("curve", ["dispersive", "plain"])
def test_a_smoothed_curved_interface_anywhere_in_the_cell_is_refused_by_name(
        monkeypatch, curve, route):
    """THE REFUSAL STOCK MEEP LEAVES, and the escape it names, on both routes.

    Both routes read eps as ``1/chi1inv[c][c]`` and check that the tensor is
    diagonal at every lattice point of the cell, so subpixel averaging at ANY curved
    face refuses the structured dispersive lift: at the dispersive cylinder, and
    equally at a plain dielectric cylinder beside axis-aligned dispersive blocks. The
    refusal names no MEEP build the user would have to make, and the first escape it
    names works: the same cell with ``eps_averaging=False`` lifts through the same
    route. The inversion is reached as the end-to-end inversion test reaches it,
    with the lookup forced to decline.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import (
        SIGMA_ROUTE_INVERSION,
        SIGMA_ROUTE_LOOKUP,
        MeepSimulationNotLiftable,
        lift_simulation,
    )

    if route == "inversion":
        monkeypatch.setattr(
            from_meep, "_why_the_lookup_cannot_run",
            lambda mp_module, simulation, media: "forced off so this test covers the inversion")
    dispersive = curve == "dispersive"
    with pytest.raises(MeepSimulationNotLiftable) as refusal:
        lift_simulation(_curved_interface_cell(mp, dispersive, True), prefer_gpu=False)
    message = str(refusal.value)
    assert "chi1inv tensor is not diagonal" in message, message
    assert ("has to be checked against" if route == "lookup"
            else "has to be recovered from chi1inv") in message, message
    assert ("Where the rows come from subpixel averaging, set eps_averaging=False or make "
            "every shape in the cell an axis-aligned Block; otherwise run this simulation on "
            "CPU MEEP.") in message, message

    driver = lift_simulation(_curved_interface_cell(mp, dispersive, False), prefer_gpu=False)
    try:
        expected = SIGMA_ROUTE_LOOKUP if route == "lookup" else SIGMA_ROUTE_INVERSION
        assert driver.sigma_lift.route == expected, driver.sigma_lift
    finally:
        driver.close()


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("route", ["lookup", "inversion"])
def test_a_declared_off_diagonal_permittivity_is_refused_without_the_smoothing_escape(
        monkeypatch, route):
    """Off-diagonal rows that ``eps_averaging=False`` does not clear, on both routes.

    A declared ``epsilon_offdiag`` puts off-diagonal entries in MEEP's tensor with
    subpixel averaging off and every shape an axis-aligned block, so the escape the
    refusal offers for smoothing cannot apply. The refusal names that source, so it does
    not send the user to a setting the cell already has, and it ends with the
    instruction that does apply.

    A cylindrical cell is refused by the same check on the builds measured here, but is
    not pinned: at a nonzero frequency MEEP 1.33.0 returns
    ``chi1_inv_tensor[component_index(c) + 3 * d]`` (monitor.cpp:367) from a nine-entry
    tensor (monitor.cpp:311), and ``d`` is R or P (3 or 4, vec.hpp:79) in a cylindrical
    structure, so the entry the check reads there lies past the end of the array
    rather than being a value MEEP defines.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import MeepSimulationNotLiftable, lift_simulation

    if route == "inversion":
        monkeypatch.setattr(
            from_meep, "_why_the_lookup_cannot_run",
            lambda mp_module, simulation, media: "forced off so this test covers the inversion")
    with pytest.raises(MeepSimulationNotLiftable) as refusal:
        lift_simulation(_anisotropic_unsmoothed_cell(mp), prefer_gpu=False)
    message = str(refusal.value)
    assert "chi1inv tensor is not diagonal" in message, message
    assert ("has to be checked against" if route == "lookup"
            else "has to be recovered from chi1inv") in message, message
    assert ("so do a medium with a nonzero epsilon_offdiag and a cylindrical cell"
            in message), message
    assert "otherwise run this simulation on CPU MEEP." in message, message


# --- the lift, end to end, and the route record ------------------------------------


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["two_block", "overlapping"])
def test_a_structured_dispersive_cell_lifts_through_the_lookup_and_steps(case):
    """The whole point: a cell whose media differ in dispersion runs on a STOCK MEEP.

    Measured against CPU MEEP stepping the same cell:

        two_block    Ez/Hx/Hy  **1.06e-07 .. 1.49e-07**   dropped-terms control 1.7e-01 .. 2.0e-01
        overlapping  Ez/Hx/Hy  **1.23e-07 .. 1.38e-07**   dropped-terms control 3.7e-01 .. 5.1e-01

    ``overlapping`` is the precedence case: its two blocks overlap, so the middle
    of the cell belongs to whichever object libctl's tree searches first. Searching
    the declaration list forwards instead of backwards puts the wrong sigma there —
    and the per-point check catches it before anything steps.

    The control is the same reference against a run with the susceptibilities
    dropped, six orders away, because a lift that silently installed no dispersion
    at all would still return a smooth complete field.
    """
    import meep as mp

    from .from_meep import SIGMA_ROUTE_LOOKUP, run_on_gpu

    build = _two_block_cell if case == "two_block" else _overlapping_cell

    result = run_on_gpu(build(mp), until=6.0, prefer_gpu=False)
    try:
        record = result.sigma_lift
        assert record is not None and record.route == SIGMA_ROUTE_LOOKUP, (
            f"{case}: the lift did not take the lookup route: {record}"
        )
        assert record.verified_points > 0 and record.worst_residual < VERIFICATION_TOLERANCE
        print(f"[sigma_lookup] stock lift {case:12} {record}", flush=True)
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
        print(f"[sigma_lookup] stock lift {case:12} {name} parity={error:.3e} "
              f"control={dropped:.3e}", flush=True)
        assert error < 5e-6, f"{case} {name}: looked-up-dispersion lift {error:.3e} from CPU MEEP"
        assert dropped > 1e-2, (
            f"{case} {name}: dropping the susceptibilities entirely moves the field only "
            f"{dropped:.3e}, so this cell cannot tell a working dispersion lift from a missing one"
        )


@requires_meep
@skip_without_meep
def test_the_route_actually_taken_is_reported_on_the_driver_and_the_result():
    """Every lift says which of the two sigma routes ran, or that none did.

    Without this a test that means to exercise one route has no way to know it did
    — the whole dispatch is invisible from outside, and both produce a driver that
    steps.
    """
    import meep as mp

    from .from_meep import SIGMA_ROUTE_LOOKUP, lift_simulation

    driver = lift_simulation(_two_block_cell(mp), prefer_gpu=False)
    try:
        record = driver.sigma_lift
        assert record.route == SIGMA_ROUTE_LOOKUP, record
        assert record.term_count == 2
        assert len(record.verification_frequencies) == 3, record.verification_frequencies
        assert record.tolerance == VERIFICATION_TOLERANCE
        assert "verified at" in str(record) and "worst disagreement" in str(record)
    finally:
        driver.close()

    # A cell with ONE medium has no per-point sigma volume at all, and says so.
    uniform = mp.Simulation(
        cell_size=mp.Vector3(2.0, 2.0, 0.0), resolution=10, force_complex_fields=True,
        default_material=mp.Medium(epsilon=4.0, E_susceptibilities=[
            mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)]),
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                           center=mp.Vector3(0, 0, 0))])
    driver = lift_simulation(uniform, prefer_gpu=False)
    try:
        assert getattr(driver, "sigma_lift", None) is None
    finally:
        driver.close()
