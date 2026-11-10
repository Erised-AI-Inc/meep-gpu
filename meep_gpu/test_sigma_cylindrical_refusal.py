"""A cylindrical cell whose media differ in dispersion is refused by name at the gate.

Both per-point sigma routes (the declared-geometry lookup and the chi1inv inversion)
read MEEP's ``get_chi1inv`` at a nonzero frequency. In a cylindrical structure MEEP
answers that query from a tensor built from the Cartesian rows, which such a
structure does not fill, and returns an entry indexed past the end of that nine-entry
array for the r and phi directions (MEEP 1.35.0 src/monitor.cpp:336, 1.33.0 :367). The
lift used to refuse these cells only because the value read there happened to be
nonzero on the builds measured, so the refusal rested on memory MEEP does not define.
``gpu_compatibility`` now refuses them by name, before ``init_sim``, and these tests pin
that, the cells it must leave alone, and that the clause is what refuses.
"""

from __future__ import annotations

import importlib.util

import pytest

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)

REASON = "cylindrical (Dcyl) cell's media differ in their E_susceptibilities"


def _lorentz(mp):
    return [mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)]


def _drude(mp):
    return [mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)]


def _dcyl(mp, near, far, *, default=None):
    """Two blocks in a cylindrical (r, z) cell, ``near`` the axis and ``far`` from it."""
    geometry = [
        mp.Block(center=mp.Vector3(0.5, 0, -0.6), size=mp.Vector3(0.6, mp.inf, 0.8),
                 material=near),
        mp.Block(center=mp.Vector3(1.3, 0, 0.6), size=mp.Vector3(0.6, mp.inf, 0.8),
                 material=far),
    ]
    return mp.Simulation(cell_size=mp.Vector3(2.0, 0, 3.0), resolution=10,
                         dimensions=mp.CYLINDRICAL, m=1, geometry=geometry,
                         default_material=default or mp.Medium(),
                         sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                            component=mp.Ez,
                                            center=mp.Vector3(0.9, 0, 0.0))])


def _reasons(sim):
    from .from_meep import gpu_compatibility  # noqa: PLC0415

    return list(gpu_compatibility(sim).reasons)


@requires_meep
@skip_without_meep
def test_a_cylindrical_cell_whose_media_differ_in_dispersion_is_refused_by_name():
    import meep as mp

    from .from_meep import MeepSimulationNotLiftable, lift_simulation

    mp.verbosity(0)
    sim = _dcyl(mp, mp.Medium(epsilon=4.0, E_susceptibilities=_lorentz(mp)),
                mp.Medium(epsilon=2.0, E_susceptibilities=_drude(mp)))
    reasons = _reasons(sim)
    hits = [reason for reason in reasons if REASON in reason]
    assert len(hits) == 1, reasons
    assert "src/monitor.cpp:336 in MEEP 1.35.0" in hits[0], hits[0]
    assert hits[0].endswith("otherwise run this simulation on CPU MEEP."), hits[0]
    with pytest.raises(MeepSimulationNotLiftable) as refusal:
        lift_simulation(sim, prefer_gpu=False)
    assert REASON in str(refusal.value)
    assert sim.structure is None, "the refusal must come before init_sim"


@requires_meep
@skip_without_meep
def test_the_clause_is_what_refuses_the_cell(monkeypatch):
    """Without the clause the gate answers supported: nothing else refuses this cell."""
    import meep as mp

    from . import from_meep

    mp.verbosity(0)
    monkeypatch.setattr(from_meep, "_check_structured_sigma_cylindrical",
                        lambda mp_module, sim, materials, reasons: None)
    sim = _dcyl(mp, mp.Medium(epsilon=4.0, E_susceptibilities=_lorentz(mp)),
                mp.Medium(epsilon=2.0, E_susceptibilities=_drude(mp)))
    assert _reasons(sim) == []


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("cell", ["one_dispersive_medium", "permittivity_only",
                                  "dispersive_default_and_block"])
def test_cylindrical_cells_the_routes_never_read_are_not_refused_by_it(cell):
    import meep as mp

    mp.verbosity(0)
    if cell == "one_dispersive_medium":
        medium = mp.Medium(epsilon=3.0, E_susceptibilities=_lorentz(mp))
        sim = _dcyl(mp, medium, medium, default=medium)
    elif cell == "permittivity_only":
        sim = _dcyl(mp, mp.Medium(epsilon=4.0), mp.Medium(epsilon=2.0))
    else:
        medium = mp.Medium(epsilon=3.0, E_susceptibilities=_lorentz(mp))
        sim = _dcyl(mp, mp.Medium(epsilon=3.0, E_susceptibilities=_drude(mp)), medium,
                    default=medium)
    reasons = _reasons(sim)
    if cell == "dispersive_default_and_block":
        assert any(REASON in reason for reason in reasons), reasons
    else:
        assert not any(REASON in reason for reason in reasons), reasons
        assert reasons == [], reasons


@requires_meep
@skip_without_meep
def test_the_same_media_in_a_cartesian_cell_are_not_refused_by_it():
    import meep as mp

    mp.verbosity(0)
    geometry = [
        mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=_lorentz(mp))),
        mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=2.0, E_susceptibilities=_drude(mp))),
    ]
    sim = mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=10,
                        geometry=geometry, force_complex_fields=True,
                        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                           component=mp.Ez, center=mp.Vector3(0, -1.2, 0))])
    assert _reasons(sim) == []
