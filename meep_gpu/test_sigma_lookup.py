"""Tests for the DECLARED-GEOMETRY sigma lookup and the per-point check under it.

Four layers, in the order a defect would reach them:

1. **The verification algebra alone**, with no MEEP: the columns against MEEP's
   own ``chi1`` closed form retyped from susceptibility.cpp:297-307, a correct
   sigma passing, a wrong one failing, and the degeneracy the check cannot see
   through being refused rather than checked blind.
2. **The lookup against the EXACT reader**, point by point. This is what the
   patched environment exists for, and it is also where the retirement question
   gets decided: the same cells are looked up AND inverted against the same
   reader, and the two disagreements are printed side by side. Measured:

       two-block Lorentz+Drude, res 10   lookup 1.70e-08   inversion 5.67e-08
       meep.materials.Au, 6 poles        lookup 3.28e-08   inversion 1.35e-06

   SKIPS on a stock MEEP, where there is no exact reader to compare against.
3. **The check FAILING when it should**, and the three cells that need the three
   images MEEP searches — all on any MEEP, no reader required. The defect the
   check is aimed at is not hypothetical: this engine's own Yee coordinate and the
   one MEEP snapped the point to differ by one ulp at a block face that no binary
   float represents, and that flips which medium 81 of 10800 points are in. Both
   halves are measured here — that the two coordinates really do disagree, and
   that substituting one for the other moves the check by seven orders.
4. **The lift end to end on a stock MEEP**, against CPU MEEP stepping the same
   cell, plus the route record that says which of the three paths ran.

THE HONEST COMPARISON, and why it is a test rather than a claim in a docstring.
The lookup is NOT cheaper than the inversion at ANY object count, including zero —
measured per call on this machine, ``mp.is_point_in_object`` costs 4.38 us against
``get_chi1inv``'s 0.64 us at a nonzero frequency, and end to end the lookup runs
1.4-2.0x the inversion at one or two objects and 14.6x at thirty-two. What it is,
is exact: it installs the declared number rather than a solved one.
``test_the_lookup_is_closer_to_the_reader_than_the_inversion_is`` is the
measurement that decides whether that is still true, and it asserts the ordering
rather than a digit.
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
    sigma instead, and 21 of 1281 points disagree with the exact reader.
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


def _match_term(term, candidates) -> int:
    """Index of ``term`` in ``candidates``, matched on physics rather than on identity."""
    best = None
    for index, other in enumerate(candidates):
        if other.kind != term.kind:
            continue
        distance = (abs(other.frequency - term.frequency) / max(abs(term.frequency), 1e-30)
                    + abs(other.gamma - term.gamma) / max(abs(term.gamma), 1e-30))
        if best is None or distance < best[1]:
            best = (index, distance)
    assert best is not None and best[1] < 1e-5, (
        f"no chain entry matches {term} among {candidates} (closest {best})"
    )
    return best[0]


def _worst_against(mine, theirs, terms, reference_terms) -> float:
    worst = 0.0
    for index, term in enumerate(terms):
        matched = _match_term(term, reference_terms)
        for name in ("Ex", "Ey", "Ez"):
            scale = max(float(np.abs(theirs[name][matched]).max()), 1e-300)
            worst = max(worst, float(np.max(np.abs(mine[name][index] - theirs[name][matched])))
                        / scale)
    return worst


# --- against the exact reader (patched MEEP only) ----------------------------------


@requires_meep
@skip_without_meep
@pytest.mark.parametrize(
    "case",
    ["two_block", "gold", "unrepresentable_face", "overlapping", "wrapped_object", "mirrored"])
def test_the_looked_up_sigma_matches_the_exact_reader_point_by_point(case):
    """The lookup against ``fields.get_susceptibility_sigma``, over every Yee point.

    Both routes run over the SAME initialized simulation and the same driver, so
    nothing but the route differs. Measured worst disagreement, relative to each
    term's own largest sigma:

        two_block             2 poles   **1.70e-08**
        gold                  6 poles   **3.28e-08**
        unrepresentable_face  1 pole    **1.49e-08**
        overlapping           2 poles   **1.70e-08**
        wrapped_object        1 pole    **1.49e-08**
        mirrored              2 poles   **1.70e-08**

    Every one of those is MEEP's own float32 storage rounding — 0.4 comes back as
    0.400000005960464 — and nothing else, because the lookup installs the declared
    number rather than a solved one. The bound asserted is 1e-6, which is two
    orders above the float32 floor and two below the smallest structural defect any
    of these cells can produce.

    SKIPPED on a stock MEEP: there is no exact reader to compare against, which is
    a statement about the oracle, not a silent pass. The stock environment's own
    bars are the check-fires test and the end-to-end lift below.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import _lookup_sigma_volumes, _materials_of, _sigma_volumes_through_the_reader

    if not from_meep._has_sigma_reader(mp):
        pytest.skip("this MEEP lacks the sigma reader; build one with MEEP_SIGMA_PATCH=1 "
                    "via parity/meep_gpu/build_meep_133_macos.sh")

    build = {
        "two_block": _two_block_cell, "gold": _gold_cell,
        "unrepresentable_face": _unrepresentable_face_cell, "overlapping": _overlapping_cell,
        "wrapped_object": _wrapped_object_cell, "mirrored": _mirrored_cell,
    }[case]
    sim = build(mp)
    driver = from_meep.lift_simulation(sim, prefer_gpu=False)
    try:
        materials = _materials_of(mp, sim)
        exact_terms, exact = _sigma_volumes_through_the_reader(mp, sim, driver)
        terms, mine, record = _lookup_sigma_volumes(mp, sim, driver, materials)
        assert len(terms) == len(exact_terms), (
            f"{case}: the lookup enumerated {len(terms)} susceptibilities where MEEP's chain "
            f"holds {len(exact_terms)}; the dedup no longer agrees with susceptibility_equiv"
        )
        worst = _worst_against(mine, exact, terms, exact_terms)
        print(f"[sigma_lookup] reader-vs-lookup {case:22} terms={len(terms)} worst={worst:.3e} "
              f"verified={record.verified_points} pts residual={record.worst_residual:.3e}",
              flush=True)
        assert worst < 1e-6, (
            f"{case}: the looked-up sigma differs from the exact reader by {worst:.3e} of a "
            f"term's own largest value, well past MEEP's float32 storage rounding"
        )
    finally:
        driver.close()


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["two_block", "gold"])
def test_the_lookup_is_closer_to_the_reader_than_the_inversion_is(case):
    """THE RETIREMENT QUESTION, measured: which compatibility route is more accurate?

    Both are run over the same cell and the same driver and both are subtracted
    from the exact reader. Measured worst disagreement, relative to each term's own
    largest sigma:

        two_block   lookup **1.70e-08**   inversion **5.67e-08**   (3.3x closer)
        gold        lookup **3.28e-08**   inversion **1.35e-06**   (41x closer)

    The gap widens with the pole count, which is the expected shape: the lookup
    installs a declared constant and the inversion solves a system whose
    conditioning grows with N. If this test ever fails, the lookup has stopped
    being the better default and the dispatch in
    ``_lift_susceptibilities_structured`` should be reconsidered — which is why it
    asserts the ordering rather than a digit.

    Cost is the other half of the comparison and it goes the OTHER way at every
    object count: measured, the lookup runs 1.4-2.0x the inversion's wall time at
    one or two objects and 14.6x at thirty-two, because ``is_point_in_object`` is
    4.38 us against ``get_chi1inv``'s 0.64 us and the scan is linear in the object
    count. The lookup is not chosen for speed, and
    ``from_meep._LOOKUP_MAX_OBJECTS`` is what bounds the price.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import (
        _lookup_sigma_volumes,
        _materials_of,
        _recover_sigma_volumes,
        _sigma_volumes_through_the_reader,
    )

    if not from_meep._has_sigma_reader(mp):
        pytest.skip("this MEEP lacks the sigma reader; build one with MEEP_SIGMA_PATCH=1 "
                    "via parity/meep_gpu/build_meep_133_macos.sh")

    sim = (_two_block_cell if case == "two_block" else _gold_cell)(mp)
    driver = from_meep.lift_simulation(sim, prefer_gpu=False)
    try:
        materials = _materials_of(mp, sim)
        exact_terms, exact = _sigma_volumes_through_the_reader(mp, sim, driver)
        lookup_terms, looked_up, _ = _lookup_sigma_volumes(mp, sim, driver, materials)
        inverted_terms, inverted = _recover_sigma_volumes(mp, sim, driver, materials)
        lookup_error = _worst_against(looked_up, exact, lookup_terms, exact_terms)
        inversion_error = _worst_against(inverted, exact, inverted_terms, exact_terms)
        print(f"[sigma_lookup] {case:10} vs reader: lookup={lookup_error:.3e} "
              f"inversion={inversion_error:.3e}", flush=True)
        assert lookup_error <= inversion_error, (
            f"{case}: the declared-geometry lookup is {lookup_error:.3e} from the exact reader "
            f"where the chi1inv inversion is {inversion_error:.3e}. The lookup is the DEFAULT "
            f"compatibility route on the strength of being the more accurate one; it no longer is"
        )
    finally:
        driver.close()


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

    monkeypatch.setattr(from_meep, "_has_sigma_reader", lambda module: False)

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
def test_a_structure_flush_with_the_cell_face_needs_meeps_own_ownership_gate(monkeypatch):
    """THE UNOWNED PLANE, on a stock MEEP where there is no reader to compare against.

    ``_lattice_indices`` enumerates the full closed lattice, and one plane per axis
    of it lies outside the region MEEP owns — ``little_owned_corner`` is
    ``little_corner + 2 - iyee_shift``, so Ez's lowest x plane is not owned. There
    MEEP answers vacuum-with-no-susceptibility at every frequency
    (monitor.cpp:182-184), whatever the geometry says. The gold block is placed
    flush with that plane precisely so the two answers differ: with the ``gv.owns``
    gate the lookup agrees with the exact reader at 0 of 1281 points wrong, without
    it at 21.

    On the patched environment the reader test above measures that directly. This
    one is the stock-MEEP half, and it does not need the reader: dropping the gate
    makes the lookup's own per-point check fail — it would be predicting Au's
    permittivity where MEEP reports vacuum — so the lift falls back to the
    inversion and the route stops being the lookup.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import SIGMA_ROUTE_LOOKUP, lift_simulation

    monkeypatch.setattr(from_meep, "_has_sigma_reader", lambda module: False)
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
def test_a_periodic_cell_wraps_both_the_objects_and_the_lattice_point(monkeypatch):
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

    from . import from_meep
    from .from_meep import SIGMA_ROUTE_LOOKUP, lift_simulation

    monkeypatch.setattr(from_meep, "_has_sigma_reader", lambda module: False)
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
def test_a_mirror_folded_cell_reads_its_unowned_plane_through_the_symmetry_image(monkeypatch):
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

    from . import from_meep
    from .from_meep import SIGMA_ROUTE_LOOKUP, lift_simulation

    monkeypatch.setattr(from_meep, "_has_sigma_reader", lambda module: False)
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
def test_a_crowded_cell_goes_to_the_inversion_before_the_lookup_pays_for_it(monkeypatch):
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

    monkeypatch.setattr(from_meep, "_has_sigma_reader", lambda module: False)

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


# --- the lift, end to end, and the route record ------------------------------------


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["two_block", "overlapping"])
def test_a_structured_dispersive_cell_lifts_through_the_lookup_and_steps(monkeypatch, case):
    """The whole point: a cell whose media differ in dispersion runs on a STOCK MEEP.

    ``_has_sigma_reader`` is the single seam both the gate and the lift consult, so
    forcing it False is precisely "pretend this MEEP has no reader" — and on the
    stock environment it is already False, which makes the patch a no-op there and
    the test the real thing. Measured against CPU MEEP stepping the same cell:

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

    from . import from_meep
    from .from_meep import SIGMA_ROUTE_LOOKUP, run_on_gpu

    monkeypatch.setattr(from_meep, "_has_sigma_reader", lambda module: False)
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
def test_the_route_actually_taken_is_reported_on_the_driver_and_the_result(monkeypatch):
    """Every lift says which of the three sigma routes ran, or that none did.

    Without this a test that means to exercise one route has no way to know it did
    — the whole dispatch is invisible from outside, and all three produce a driver
    that steps. The reader route is asserted only where the reader exists; the
    lookup is asserted with it forced away, which is the state a stock MEEP is
    already in.
    """
    import meep as mp

    from . import from_meep
    from .from_meep import SIGMA_ROUTE_LOOKUP, SIGMA_ROUTE_READER, lift_simulation

    if from_meep._has_sigma_reader(mp):
        driver = lift_simulation(_two_block_cell(mp), prefer_gpu=False)
        try:
            assert driver.sigma_lift.route == SIGMA_ROUTE_READER, driver.sigma_lift
            assert driver.sigma_lift.term_count == 2
        finally:
            driver.close()

    monkeypatch.setattr(from_meep, "_has_sigma_reader", lambda module: False)
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
