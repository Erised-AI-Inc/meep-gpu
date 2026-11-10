"""``gpu_compatibility`` and ``lift_simulation`` agree on cells with off-diagonal ``chi1inv`` rows.

A cell whose media differ in their E_susceptibilities is lifted by reading per-point
sigma volumes, and both routes for that read the permittivity as ``1/chi1inv[c][c]``,
which holds only where MEEP's tensor is diagonal; the lift checks that at every lattice
point after ``init_sim`` and refuses where it fails. ``gpu_compatibility`` never builds
anything, and it used to answer ``supported`` for every such cell, so a script that asked
first was told yes and then refused. A medium that declares a nonzero ``epsilon_offdiag``
decides the lift's answer from the declaration alone, so the gate now makes that refusal
itself and the two agree.

Subpixel averaging at a curved or rotated interface is the one source of those rows the
declaration does not decide, and the cells below measure why: on the same media a
smoothed disc of a different permittivity is refused, while a smoothed disc of the
background's permittivity, a sphere in a 1-D cell and a cylinder lying in the 2-D plane
all lift. No rule the gate can read off the declaration separates them, so for those
cells the gate answers ``supported`` and the lift's refusal names the class.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)


def _cell(mp, name):
    lorentz = [mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)]
    drude = [mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)]
    drude_block = mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                           material=mp.Medium(epsilon=2.0, E_susceptibilities=drude))
    plane = dict(cell_size=mp.Vector3(4, 4, 0), resolution=10, force_complex_fields=True,
                 sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                                    center=mp.Vector3(0, -1.2, 0))])
    disc = mp.Sphere(radius=0.73, center=mp.Vector3(-1.0, 0, 0),
                     material=mp.Medium(epsilon=4.0, E_susceptibilities=lorentz))
    offdiag = mp.Medium(epsilon_diag=mp.Vector3(4, 3, 2), epsilon_offdiag=mp.Vector3(0.4, 0, 0),
                        E_susceptibilities=lorentz)
    offdiag_block = mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                             material=offdiag)
    if name == "two_blocks":
        return mp.Simulation(geometry=[mp.Block(center=mp.Vector3(-1.0, 0, 0),
                                                size=mp.Vector3(1.46, 1.46, mp.inf),
                                                material=disc.material), drude_block], **plane)
    if name == "disc_unsmoothed":
        return mp.Simulation(geometry=[disc, drude_block], eps_averaging=False, **plane)
    if name == "disc_smoothed":
        return mp.Simulation(geometry=[disc, drude_block], eps_averaging=True, **plane)
    if name == "disc_of_the_background_permittivity_smoothed":
        metal = mp.Sphere(radius=0.73, center=mp.Vector3(-1.0, 0, 0),
                          material=mp.Medium(epsilon=1.0, E_susceptibilities=drude))
        return mp.Simulation(geometry=[metal], eps_averaging=True, **plane)
    if name == "sphere_in_a_1d_cell_smoothed":
        return mp.Simulation(cell_size=mp.Vector3(0, 0, 4), resolution=10, dimensions=1,
                             eps_averaging=True,
                             geometry=[mp.Sphere(radius=0.73, center=mp.Vector3(0, 0, 0.3),
                                                 material=disc.material)],
                             sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                                component=mp.Ex,
                                                center=mp.Vector3(0, 0, -1.2))])
    if name == "cylinder_in_the_plane_smoothed":
        return mp.Simulation(geometry=[mp.Cylinder(radius=0.5, height=mp.inf,
                                                   axis=mp.Vector3(1, 0, 0),
                                                   center=mp.Vector3(0, 0.4, 0),
                                                   material=disc.material)],
                             eps_averaging=True, **plane)
    if name == "rotated_block_smoothed":
        angle = math.radians(30.0)
        return mp.Simulation(
            geometry=[mp.Block(center=mp.Vector3(-0.9, 0.1, 0), size=mp.Vector3(1.3, 0.8, mp.inf),
                               e1=mp.Vector3(math.cos(angle), math.sin(angle), 0),
                               e2=mp.Vector3(-math.sin(angle), math.cos(angle), 0),
                               material=disc.material), drude_block],
            eps_averaging=True, **plane)
    if name == "offdiag_block_unsmoothed":
        return mp.Simulation(geometry=[offdiag_block, drude_block], eps_averaging=False, **plane)
    if name == "offdiag_block_smoothed":
        return mp.Simulation(geometry=[offdiag_block, drude_block], eps_averaging=True, **plane)
    if name == "offdiag_default":
        return mp.Simulation(
            geometry=[mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                               material=disc.material)],
            default_material=mp.Medium(epsilon_diag=mp.Vector3(2, 2, 2),
                                       epsilon_offdiag=mp.Vector3(0.3, 0, 0)),
            eps_averaging=False, **plane)
    if name == "offdiag_with_one_dispersion":
        # The media agree on their dispersion: the epsilon lift ingests the rows.
        return mp.Simulation(
            geometry=[mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                               material=mp.Medium(epsilon_diag=mp.Vector3(4, 3, 2),
                                                  epsilon_offdiag=mp.Vector3(0.4, 0, 0)))],
            eps_averaging=False, **plane)
    raise ValueError(name)


#: The cells and what both sides answer: True lifts, False refuses.
AGREE = {
    "two_blocks": True,
    "disc_unsmoothed": True,
    "disc_of_the_background_permittivity_smoothed": True,
    "sphere_in_a_1d_cell_smoothed": True,
    "cylinder_in_the_plane_smoothed": True,
    "offdiag_with_one_dispersion": True,
    "offdiag_block_unsmoothed": False,
    "offdiag_block_smoothed": False,
    "offdiag_default": False,
}

#: The class only the rasterized structure decides: the gate answers supported and
#: the lift refuses, naming it.
RASTERIZED = ("disc_smoothed", "rotated_block_smoothed")


def _lift(mp, name):
    from .from_meep import MeepSimulationNotLiftable, lift_simulation  # noqa: PLC0415

    try:
        driver = lift_simulation(_cell(mp, name), prefer_gpu=False)
    except MeepSimulationNotLiftable as refusal:
        return False, str(refusal)
    driver.close()
    return True, ""


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("name", sorted(AGREE))
def test_the_gate_and_the_lift_give_the_same_answer(name):
    import meep as mp

    from .from_meep import gpu_compatibility

    mp.verbosity(0)
    gate = gpu_compatibility(_cell(mp, name))
    lifted, why = _lift(mp, name)
    assert gate.supported is AGREE[name], gate.reasons
    assert lifted is AGREE[name], why
    if not AGREE[name]:
        assert any("declares epsilon_offdiag" in reason for reason in gate.reasons), gate.reasons


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("name", RASTERIZED)
def test_smoothing_is_refused_by_the_lift_and_named_as_the_class_the_gate_cannot_see(name):
    import meep as mp

    from .from_meep import gpu_compatibility

    mp.verbosity(0)
    assert gpu_compatibility(_cell(mp, name)).supported
    lifted, why = _lift(mp, name)
    assert not lifted
    assert "chi1inv tensor is not diagonal" in why, why
    assert "the one source of them gpu_compatibility cannot see before init_sim" in why, why
    assert "set eps_averaging=False or make every shape in the cell an axis-aligned Block" in why


@requires_meep
@skip_without_meep
def test_without_the_clause_the_gate_says_supported_and_the_lift_refuses(monkeypatch):
    """What the clause closes: the gate's old answer on a declared off-diagonal cell."""
    import meep as mp

    from . import from_meep

    mp.verbosity(0)
    monkeypatch.setattr(from_meep, "_check_sigma_reads_a_diagonal_tensor",
                        lambda materials, reasons: None)
    assert from_meep.gpu_compatibility(_cell(mp, "offdiag_block_unsmoothed")).supported
    lifted, why = _lift(mp, "offdiag_block_unsmoothed")
    assert not lifted and "chi1inv tensor is not diagonal" in why, why
