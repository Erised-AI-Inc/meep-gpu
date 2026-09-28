"""First run: a simulation built by MEEP, stepped on this host's GPU.

This is the README's first-run block, as a file. The case is a dielectric sphere
in a 4 x 4 x 4 cell with absorbing boundaries and one flux monitor, at 48 cells
per side (110,592 cells).

    python examples/quickstart.py

On a host with no GPU the default request raises before MEEP initializes
anything, and the message names prefer_gpu=False, the NumPy reference.
"""

import meep as mp
from meep_gpu import gpu_compatibility, run_on_gpu

sim = mp.Simulation(                              # your existing script
    cell_size=mp.Vector3(4, 4, 4),
    resolution=12,
    boundary_layers=[mp.PML(0.8)],
    geometry=[mp.Sphere(radius=0.8, material=mp.Medium(epsilon=9))],
    sources=[mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.4),
                       component=mp.Ez, center=mp.Vector3(-1.2, 0, 0))],
)
flux = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(center=mp.Vector3(0.9, 0, 0),
                                               size=mp.Vector3(0, 2, 2)))

verdict = gpu_compatibility(sim)                  # supported, or the reasons why not
print("supported:", verdict.supported, list(verdict.reasons))

result = run_on_gpu(sim, until=35)                # instead of sim.run(until=35)
try:
    spectrum = result.get_flux_spectrum(flux)
    ez = result.get_array("Ez")
    print("steps:", result.steps, " MEEP time:", result.meep_time)
    print("step path:", result.driver.active_step_path)
    print("flux:", " ".join(f"{value:.6e}" for value in spectrum))
    print("max |Ez|:", f"{abs(ez).max():.6e}")
finally:
    result.close()                                # releases host and device arrays
