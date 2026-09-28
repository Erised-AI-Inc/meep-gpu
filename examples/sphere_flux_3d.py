"""A dielectric sphere under absorbing boundaries, in three dimensions.

The problem class the package is aimed at, at its smallest measured size: a
4 x 4 x 4 cell, a sphere of permittivity 9 and radius 0.8, a PML 0.8 thick, a
Gaussian source and one flux monitor. At the default resolution of 12 the grid
is 48 cells per side (110,592 cells) and the run takes 840 steps.

    python examples/sphere_flux_3d.py                   # this host's GPU
    python examples/sphere_flux_3d.py --leg array       # the array path, dispatch off
    python examples/sphere_flux_3d.py --leg reference   # the NumPy reference
    python examples/sphere_flux_3d.py --leg meep        # MEEP itself
    python examples/sphere_flux_3d.py --resolution 32   # 2,097,152 cells
"""

import meep as mp

from common import run_leg


def build(resolution=None):
    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 4, 4),
        resolution=resolution or 12,
        boundary_layers=[mp.PML(0.8)],
        geometry=[mp.Sphere(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=9))],
        sources=[mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.4),
                           component=mp.Ez, center=mp.Vector3(-1.2, 0, 0))],
    )
    flux = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(center=mp.Vector3(0.9, 0, 0),
                                                   size=mp.Vector3(0, 2, 2)))
    return sim, [flux], 35.0


if __name__ == "__main__":
    run_leg("sphere_flux_3d", build, field_name="Ez")
