"""A dielectric slab waveguide with two flux planes, in two dimensions.

A 10 x 6 cell, a slab of permittivity 12 and width 1, a PML 1 thick, a Gaussian
source in the slab and a flux plane on either side of the middle. At the default
resolution of 20 the grid is 200 x 120 (24,000 cells) and the run takes 2,400
steps.

This example shows the calls, not a speed-up: two-dimensional problems of this
size are faster on MEEP itself (README, "Will it help?").

    python examples/slab_flux_2d.py                   # this host's GPU
    python examples/slab_flux_2d.py --leg array       # the array path, dispatch off
    python examples/slab_flux_2d.py --leg reference   # the NumPy reference
    python examples/slab_flux_2d.py --leg meep        # MEEP itself
"""

import meep as mp

from common import run_leg


def build(resolution=None):
    sim = mp.Simulation(
        cell_size=mp.Vector3(10, 6, 0),
        resolution=resolution or 20,
        boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.GaussianSource(frequency=0.25, fwidth=0.3),
                           component=mp.Ez, center=mp.Vector3(-3.5, 0))],
    )
    near = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(center=mp.Vector3(-2.5, 0),
                                                    size=mp.Vector3(0, 4)))
    far = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(center=mp.Vector3(3.0, 0),
                                                   size=mp.Vector3(0, 4)))
    return sim, [near, far], 60.0


if __name__ == "__main__":
    run_leg("slab_flux_2d", build, field_name="Ez")
