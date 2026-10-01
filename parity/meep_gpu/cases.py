"""Representative simulation builders, one per shape that matters.

Each builder returns a fresh, unconstructed-fields mp.Simulation. Kept in one
module so the timing harness and the profiler drive IDENTICAL problems.
"""
from __future__ import annotations

import meep as mp


def case_2d_plain(res=40, n=16):
    # Plain 2-D, periodic (no PML), no monitors: the bare curl+constitutive core.
    cell = mp.Vector3(n, n, 0)
    src = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                     center=mp.Vector3(0, 0))]
    return mp.Simulation(cell_size=cell, resolution=res, sources=src,
                         boundary_layers=[], geometry=[])


def case_2d_pml(res=40, n=16, dpml=1.0):
    cell = mp.Vector3(n, n, 0)
    src = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                     center=mp.Vector3(0, 0))]
    return mp.Simulation(cell_size=cell, resolution=res, sources=src,
                         boundary_layers=[mp.PML(dpml)], geometry=[])


def case_2d_pml_geom(res=40, n=16, dpml=1.0):
    # PML + a dielectric block: the epsilon array is non-uniform.
    cell = mp.Vector3(n, n, 0)
    geom = [mp.Block(size=mp.Vector3(4, 4, mp.inf), center=mp.Vector3(2, 2),
                     material=mp.Medium(epsilon=12))]
    src = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                     center=mp.Vector3(-4, -4))]
    return mp.Simulation(cell_size=cell, resolution=res, sources=src,
                         boundary_layers=[mp.PML(dpml)], geometry=geom)


def case_2d_symmetry(res=40, n=16, dpml=1.0):
    cell = mp.Vector3(n, n, 0)
    src = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                     center=mp.Vector3(0, 0))]
    return mp.Simulation(cell_size=cell, resolution=res, sources=src,
                         boundary_layers=[mp.PML(dpml)], geometry=[],
                         symmetries=[mp.Mirror(mp.Y)])


def case_2d_dispersive(res=40, n=16, dpml=1.0):
    # Lorentz susceptibility everywhere -> update_P runs every step.
    med = mp.Medium(epsilon=2.25,
                    E_susceptibilities=[mp.LorentzianSusceptibility(
                        frequency=1.1, gamma=1e-5, sigma=0.5)])
    cell = mp.Vector3(n, n, 0)
    src = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                     center=mp.Vector3(0, 0))]
    return mp.Simulation(cell_size=cell, resolution=res, sources=src,
                         boundary_layers=[mp.PML(dpml)], geometry=[],
                         default_material=med)


def case_3d_pml(res=15, n=6, dpml=1.0):
    cell = mp.Vector3(n, n, n)
    src = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                     center=mp.Vector3(0, 0, 0))]
    return mp.Simulation(cell_size=cell, resolution=res, sources=src,
                         boundary_layers=[mp.PML(dpml)], geometry=[])


def case_cylindrical(res=40, r=8.0, z=8.0, dpml=1.0):
    cell = mp.Vector3(r, 0, z)
    src = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Er,
                     center=mp.Vector3(1.0, 0, 0))]
    return mp.Simulation(cell_size=cell, resolution=res, sources=src,
                         dimensions=mp.CYLINDRICAL, m=0,
                         boundary_layers=[mp.PML(dpml)], geometry=[])


def case_2d_dft(res=40, n=16, dpml=1.0):
    # Same as case_2d_pml; monitors are attached by the harness on each side.
    return case_2d_pml(res=res, n=n, dpml=dpml)


BUILDERS = {
    "2d_plain": case_2d_plain,
    "2d_pml": case_2d_pml,
    "2d_pml_geom": case_2d_pml_geom,
    "2d_symmetry": case_2d_symmetry,
    "2d_dispersive": case_2d_dispersive,
    "3d_pml": case_3d_pml,
    "cylindrical": case_cylindrical,
    "2d_dft": case_2d_dft,
}


def case_2d_cond_pml(res=40, n=16, dpml=1.0):
    # PML + a D-conductivity everywhere: MEEP's most general curl recurrence.
    med = mp.Medium(epsilon=2.0, D_conductivity=0.4)
    cell = mp.Vector3(n, n, 0)
    src = [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                     center=mp.Vector3(0, 0))]
    return mp.Simulation(cell_size=cell, resolution=res, sources=src,
                         boundary_layers=[mp.PML(dpml)], geometry=[],
                         default_material=med)


BUILDERS["2d_cond_pml"] = case_2d_cond_pml
