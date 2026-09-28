"""A Lorentzian slab under absorbing boundaries, in two dimensions.

A dispersive material: the slab carries one Lorentzian susceptibility, so every
step also advances its polarization. An 8 x 6 cell, a PML 1 thick, a Gaussian
source and a flux plane on either side of the slab. At the default resolution of
20 the grid is 160 x 120 (19,200 cells) and the run takes 2,200 steps.

This example shows the calls, not a speed-up: two-dimensional problems of this
size are faster on MEEP itself (README, "Will it help?").

    python examples/dispersive_slab_2d.py                   # this host's GPU
    python examples/dispersive_slab_2d.py --leg array       # the array path, dispatch off
    python examples/dispersive_slab_2d.py --leg reference   # the NumPy reference
    python examples/dispersive_slab_2d.py --leg meep        # MEEP itself
"""

import meep as mp

from common import run_leg


def build(resolution=None):
    lorentz = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.4, gamma=0.02, sigma=1.1)])
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 6, 0),
        resolution=resolution or 20,
        boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(mp.Vector3(2.0, mp.inf, mp.inf), center=mp.Vector3(),
                           material=lorentz)],
        sources=[mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.3),
                           component=mp.Ez, center=mp.Vector3(-2.8, 0))],
    )
    before = sim.add_flux(0.4, 0.3, 5, mp.FluxRegion(center=mp.Vector3(-2.0, 0),
                                                     size=mp.Vector3(0, 4)))
    after = sim.add_flux(0.4, 0.3, 5, mp.FluxRegion(center=mp.Vector3(2.5, 0),
                                                    size=mp.Vector3(0, 4)))
    return sim, [before, after], 55.0


if __name__ == "__main__":
    run_leg("dispersive_slab_2d", build, field_name="Ez")
