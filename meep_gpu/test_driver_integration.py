"""
Cross-feature integration tests for the standalone FDTD driver.

The magnetic-current sources, the ``is_integrated`` injection path and the
custom temporal envelopes live in `sources.py`; the multi-frequency DFT and flux
accumulators live in `dft.py`; the step order, the source dispatch and the
monitor plumbing live in `driver.py`. Each was built and validated against CPU
MEEP on its own. This file exercises them *through the driver's public API and
together*, which is where independently-built features meet for the first time:
a magnetic source has to be injected in the B slot at `t` rather than the D slot
at `t + dt/2`, an integrated source has to carry its dipole offset across steps
and be cleared by `reset()`, and both have to drive a broadband flux monitor
behind a PML and produce MEEP's spectrum.

The controls matter as much as the measurements. Every cross-validation here is
paired with a deliberately wrong variant — the same source injected in the other
half-step slot, the same waveform with `is_integrated` dropped — asserted to
*exceed* the 5% bar, so a bound that a mistake could also satisfy cannot pass.
The timing errors these catch are pure phase errors, invisible to a
magnitude-only comparison, so every field assertion is on the complex field.

Flux-plane placement is checked here too, positionally rather than at one lucky
coordinate: `test_flux_plane_lands_where_it_was_asked_for_across_a_whole_cell`
sweeps the plane's normal coordinate through a cell and a half in steps of a
tenth of a cell and cross-validates every position against its own CPU-MEEP
flux. It replaces a pin that recorded the ~7% error a plane used to pick up when
its coordinate fell between two rows of cell centres, which is now interpolated
(dft.py `FluxMonitor._setup_region`).

The real-field (float32) section at the end of the file needs no oracle at all, and
says three things the CPU-MEEP comparisons cannot. First a negative: a complex-mode
run's raw D and B bytes are exactly what the code produced BEFORE real mode existed,
checked by reverting the three decision points real mode introduced in-process and
demanding identical bytes across fourteen configurations — every complex floor this
package records was measured before that work landed, so nothing else keeps them
valid. Second, that a real run is bit-for-bit the complex run's real part on all six
primary arrays, with the imaginary plane required to carry real weight so the equality
discriminates one injection convention from another. Third, how the two ingredients a
single real plane cannot carry are handled, and they are handled differently: a Bloch
phase RAISES (MEEP refuses it too — boundaries.cpp ``use_bloch``), with the fields
checked to be still exactly zero afterwards so no partial deposit was made before the
refusal, while a complex spatial amplitude is PROJECTED to ``real(A)``, which is
MEEP's own unconditional step.cpp:307 and is asserted to equal the complex run's real
part on every source family.

Everything runs on plain NumPy; no CUDA device and no cupy are involved. The CPU
MEEP references come from one subprocess, since meep and this package must not
share an interpreter.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from . import fields as fields_module  # Patched wholesale by the complex-mode byte-identity test.
from . import sources as sources_module
from .backends import to_numpy
from .dft import Near2FarMonitor, Near2FarRegion
from .dispersion import DRUDE, LORENTZIAN, Susceptibility
from .driver import FdtdDriver, FdtdNonlinearityOutOfRange
from .sources import FIELD_TYPE_B, FIELD_TYPE_D, VolumeSource

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(MEEP_MISSING, reason="CPU MEEP is not installed in this environment")

CROSS_VALIDATION_BAR = 0.05  # the parity matrix §7.

# Every z coordinate below is a half-cell offset (x.x5), i.e. a row of cell
# centres at resolution 10, so the feature under test is the only thing being
# measured and no interpolation weight enters the comparison. Coordinates off the
# sample planes are covered on their own by the positional sweep below.
_SOURCE_Z = -1.05
_PLANE_Z = 1.05

_ORACLE_SCRIPT = '''"""CPU-MEEP references for the FDTD driver cross-feature integration tests."""
import sys

import numpy as np
import meep as mp

output_path = sys.argv[1]
results = {}


def volume(cell, sources, until, components, pml=None):  # Run one case, return stacked field volumes.
    simulation = mp.Simulation(
        cell_size=mp.Vector3(*cell),
        resolution=10,
        sources=sources,
        boundary_layers=[mp.PML(pml)] if pml else [],
        force_complex_fields=True,
        k_point=mp.Vector3(0, 0, 0),
    )
    simulation.run(until=until)
    return np.stack([np.asarray(simulation.get_array(component=c)) for c in components])


# --- magnetic (Hx) point source, no PML -------------------------------------
results["magnetic"] = volume(
    (2, 2, 4),
    [mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Hx, center=mp.Vector3(0, 0, -1.05))],
    3,
    (mp.Hx, mp.Ey, mp.Ez),
)

# --- is_integrated continuous electric source -------------------------------
results["integrated"] = volume(
    (2, 2, 4),
    [mp.Source(mp.ContinuousSource(frequency=1.0, width=1.0, is_integrated=True),
               component=mp.Ez, center=mp.Vector3(0, 0, -1.05))],
    3,
    (mp.Ez, mp.Hx, mp.Hy),
)


# --- custom temporal envelope -----------------------------------------------
def wave(t):  # Gaussian-windowed carrier; the engine side uses this same expression.
    return np.exp(-((t - 1.5) ** 2)) * np.exp(-2j * np.pi * t)


results["custom"] = volume(
    (2, 2, 4),
    [mp.Source(mp.CustomSource(src_func=wave), component=mp.Ez, center=mp.Vector3(0, 0, -1.05))],
    3,
    (mp.Ez, mp.Hx, mp.Hy),
)

# --- everything at once: magnetic + integrated + gaussian -> broadband flux --
combined = mp.Simulation(
    cell_size=mp.Vector3(3, 3, 5),
    resolution=10,
    sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.6, is_integrated=True),
                       component=mp.Hx, center=mp.Vector3(0, 0, -1.05), size=mp.Vector3(1, 1, 0))],
    boundary_layers=[mp.PML(0.8)],
    force_complex_fields=True,
    k_point=mp.Vector3(0, 0, 0),
)
combined_flux = combined.add_flux(
    1.0, 0.6, 5,
    mp.FluxRegion(center=mp.Vector3(0, 0, 1.05), size=mp.Vector3(1.4, 1.4, 0)),
    decimation_factor=1,
)
# Flux is a time-averaged power and is blind to a global phase, so the combined
# case also records the complex DFT field, which is not. The volume's faces fall
# between cell centres (0.35/0.75 in z, +/-0.2 in x and y on a 0.1 grid) so both
# codes round outward to the same cells.
combined_fields = combined.add_dft_fields(
    [mp.Ey], 1.0, 0.6, 5,
    center=mp.Vector3(0, 0, 0.55), size=mp.Vector3(0.4, 0.4, 0.4),
    decimation_factor=1,
)
combined.run(until=25)
results["combined_freqs"] = np.asarray(mp.get_flux_freqs(combined_flux))
results["combined_flux"] = np.asarray(mp.get_fluxes(combined_flux))
results["combined_dft_ey"] = np.stack(
    [np.asarray(combined.get_dft_array(combined_fields, mp.Ey, index)) for index in range(5)]
)

# --- one run, the flux plane swept across a cell and a half in z ------------
sweep_z = [round(0.95 + 0.01 * step, 6) for step in range(16)]
sweep = mp.Simulation(
    cell_size=mp.Vector3(3, 3, 5),
    resolution=10,
    sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.8), component=mp.Ez,
                       center=mp.Vector3(0, 0, -1.05), size=mp.Vector3(1, 1, 0))],
    boundary_layers=[mp.PML(0.8)],
    force_complex_fields=True,
    k_point=mp.Vector3(0, 0, 0),
)
sweep_planes = [
    sweep.add_flux(1.0, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, z), size=mp.Vector3(1.4, 1.4, 0),
                                            direction=mp.Z), decimation_factor=1)
    for z in sweep_z
]
# Same run: monitors given a thickness along their own normal, which MEEP turns
# from a surface integral into a volume one (loop_in_chunks dV0).
thicknesses = [0.05, 0.1, 0.3]
thick_planes = [
    sweep.add_flux(1.0, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, 1.05),
                                            size=mp.Vector3(1.4, 1.4, thickness), direction=mp.Z),
                   decimation_factor=1)
    for thickness in thicknesses
]
sweep.run(until=25)
results["sweep_z"] = np.asarray(sweep_z)
results["sweep_flux"] = np.asarray([mp.get_fluxes(plane)[0] for plane in sweep_planes])
results["thicknesses"] = np.asarray(thicknesses)
results["thickness_flux"] = np.asarray([mp.get_fluxes(plane)[0] for plane in thick_planes])

# --- flux planes that reach past a PERIODIC face, and Bloch-periodic gratings -
# No PML anywhere here: the boundaries wrap, so the field at the far face is not
# negligible and a monitor whose sampling reaches past it has something real to
# lose. Two of these planes are ordinary requests (the last row of cell centres,
# the full periodic cross-section), and MEEP answers all of them by continuing the
# region into the neighbouring lattice image rather than clipping at the face.
WRAP_CELL = (2.0, 2.0, 3.0)
WRAP_SOURCE = [mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.8), component=mp.Ez,
                         center=mp.Vector3(0, 0, -0.55), size=mp.Vector3(1, 1, 0))]
WRAP_PLANES = [
    ("interior_1.23", (0.0, 0.0, 1.23), (1.4, 1.4, 0.0)),
    ("last_centre_1.45", (0.0, 0.0, 1.45), (1.4, 1.4, 0.0)),
    ("past_hi_face_1.47", (0.0, 0.0, 1.47), (1.4, 1.4, 0.0)),
    ("past_lo_face_-1.47", (0.0, 0.0, -1.47), (1.4, 1.4, 0.0)),
    ("on_hi_face_1.50", (0.0, 0.0, 1.50), (1.4, 1.4, 0.0)),
    ("full_cross_section", (0.0, 0.0, 0.55), (2.0, 2.0, 0.0)),
    ("wider_than_the_cell", (0.0, 0.0, 0.55), (3.0, 3.0, 0.0)),
]
wrap_sim = mp.Simulation(cell_size=mp.Vector3(*WRAP_CELL), resolution=10, sources=WRAP_SOURCE,
                         force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
wrap_monitors = [
    wrap_sim.add_flux(1.0, 0, 1, mp.FluxRegion(center=mp.Vector3(*centre), size=mp.Vector3(*size),
                                               direction=mp.Z), decimation_factor=1)
    for _, centre, size in WRAP_PLANES
]
wrap_sim.run(until=12)
results["wrap_names"] = np.array([name for name, _, _ in WRAP_PLANES])
results["wrap_flux"] = np.asarray([mp.get_fluxes(monitor)[0] for monitor in wrap_monitors])

# Bloch-periodic cell measured by a flux plane: the grating / band-structure
# transmission, and the one place the boundary phase and the monitor's wrap meet.
GRATING_K = [(0.3, 0.0), (0.15, -0.25)]
grating = []
for kx, ky in GRATING_K:
    grating_sim = mp.Simulation(cell_size=mp.Vector3(*WRAP_CELL), resolution=10, sources=WRAP_SOURCE,
                                force_complex_fields=True, k_point=mp.Vector3(kx, ky, 0))
    inside = grating_sim.add_flux(1.0, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, 0.55),
                                                           size=mp.Vector3(1.4, 1.4, 0),
                                                           direction=mp.Z), decimation_factor=1)
    spanning = grating_sim.add_flux(1.0, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, 0.55),
                                                             size=mp.Vector3(2, 2, 0),
                                                             direction=mp.Z), decimation_factor=1)
    grating_sim.run(until=12)
    grating.append([mp.get_fluxes(inside)[0], mp.get_fluxes(spanning)[0]])
results["grating_k"] = np.asarray(GRATING_K)
results["grating_flux"] = np.asarray(grating)

# --- DFT-field region whose upper face is the cell face ----------------------
# Ex has Yee shift 0 in y and z, so reaching a cell centre costs an average with
# the next plane; for the "touching" slab that plane is one lattice vector away.
dft_sim = mp.Simulation(cell_size=mp.Vector3(*WRAP_CELL), resolution=10, sources=WRAP_SOURCE,
                        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
dft_touching = dft_sim.add_dft_fields([mp.Ex], 1.0, 0, 1, center=mp.Vector3(0, 0, 1.15),
                                      size=mp.Vector3(0.4, 0.4, 0.5), decimation_factor=1)
dft_interior = dft_sim.add_dft_fields([mp.Ex], 1.0, 0, 1, center=mp.Vector3(0, 0, 0.25),
                                      size=mp.Vector3(0.4, 0.4, 0.5), decimation_factor=1)
dft_sim.run(until=12)
results["dft_touching"] = np.asarray(dft_sim.get_dft_array(dft_touching, mp.Ex, 0))
results["dft_interior"] = np.asarray(dft_sim.get_dft_array(dft_interior, mp.Ex, 0))

# The same slab reaching PAST the +z face, in a run whose z faces absorb. MEEP's
# boundaries are Periodic here too — use_bloch set them, the layer is a material
# graded underneath — so loop_in_chunks takes the lattice shift and the returned
# array covers the whole requested volume, absorber or no absorber.
for name, layers in (("pmlz", [mp.PML(0.6, direction=mp.Z)]), ("pmlall", [mp.PML(0.4)])):
    past_sim = mp.Simulation(cell_size=mp.Vector3(*WRAP_CELL), resolution=10, sources=WRAP_SOURCE,
                             boundary_layers=layers, force_complex_fields=True,
                             k_point=mp.Vector3(0, 0, 0))
    past_face = past_sim.add_dft_fields([mp.Ex], 1.0, 0, 1, center=mp.Vector3(0, 0, 1.3),
                                        size=mp.Vector3(0.4, 0.4, 0.6), decimation_factor=1)
    past_interior = past_sim.add_dft_fields([mp.Ex], 1.0, 0, 1, center=mp.Vector3(0, 0, 0.25),
                                            size=mp.Vector3(0.4, 0.4, 0.5), decimation_factor=1)
    past_sim.run(until=12)
    results["dft_past_face_" + name] = np.asarray(past_sim.get_dft_array(past_face, mp.Ex, 0))
    results["dft_past_interior_" + name] = np.asarray(past_sim.get_dft_array(past_interior, mp.Ex, 0))

# The same slab under a Bloch phase on the axis it touches: the wrapped plane now
# arrives multiplied, and this is the only case that can see a dropped factor there.
DFT_BLOCH_KZ = 0.35
bloch_dft_sim = mp.Simulation(cell_size=mp.Vector3(*WRAP_CELL), resolution=10, sources=WRAP_SOURCE,
                              force_complex_fields=True, k_point=mp.Vector3(0, 0, DFT_BLOCH_KZ))
bloch_touching = bloch_dft_sim.add_dft_fields([mp.Ex], 1.0, 0, 1, center=mp.Vector3(0, 0, 1.15),
                                              size=mp.Vector3(0.4, 0.4, 0.5), decimation_factor=1)
bloch_interior = bloch_dft_sim.add_dft_fields([mp.Ex], 1.0, 0, 1, center=mp.Vector3(0, 0, 0.25),
                                              size=mp.Vector3(0.4, 0.4, 0.5), decimation_factor=1)
bloch_dft_sim.run(until=12)
results["dft_bloch_kz"] = np.asarray(DFT_BLOCH_KZ)
results["dft_bloch_touching"] = np.asarray(bloch_dft_sim.get_dft_array(bloch_touching, mp.Ex, 0))
results["dft_bloch_interior"] = np.asarray(bloch_dft_sim.get_dft_array(bloch_interior, mp.Ex, 0))

# --- DFT-region registration sweep ------------------------------------------
# Every way a region can meet a boundary: faces on and between rows of cell
# centres, a region on the last row, past a face, centred on a face, spanning the
# whole periodic cross-section, spanning the whole cell, wider than the cell, and
# flat in one, two or three axes. Each is measured at k = 0 and under a Bloch
# wavevector with a component on all three axes, so a phase dropped on any wrapped
# axis shows up.
#
# The oracle is two things per case. ATLAS is one MEEP monitor covering the cell
# and a full lattice image on each side; every case's cells are a block of it, so
# comparing against that block checks *which* cells the engine registered as well
# as what is in them, uniformly, whether or not the case wraps. Alongside it each
# case's own get_dft_array pins the shape MEEP gives that request.
REG_K = {"k0": (0.0, 0.0, 0.0), "bloch": (0.15, -0.25, 0.35)}
REG_ATLAS_SIZE = (4.0, 4.0, 4.0)  # WRAP_CELL plus one image per face, on every axis.
REG_CASES = [
    ("interior",           (0.0, 0.0, 0.25),  (0.4, 0.4, 0.5)),
    ("centre_face_even",   (0.0, 0.0, 0.25),  (2.1, 0.4, 0.5)),
    ("centre_face_odd",    (0.0, 0.0, 0.25),  (1.9, 0.4, 0.5)),
    ("centre_face_z",      (0.0, 0.0, 0.25),  (0.4, 0.4, 0.9)),
    ("touch_far_face_z",   (0.0, 0.0, 1.15),  (0.4, 0.4, 0.5)),
    ("last_centre_z",      (0.0, 0.0, 1.25),  (0.4, 0.4, 0.4)),
    ("past_hi_face_z",     (0.0, 0.0, 1.35),  (0.4, 0.4, 0.5)),
    ("past_lo_face_z",     (0.0, 0.0, -1.35), (0.4, 0.4, 0.5)),
    ("on_hi_face_z",       (0.0, 0.0, 1.50),  (0.4, 0.4, 0.4)),
    ("span_full_xy",       (0.0, 0.0, 0.25),  (2.0, 2.0, 0.4)),
    ("whole_cell",         (0.0, 0.0, 0.0),   (2.0, 2.0, 3.0)),
    ("wider_than_cell_x",  (0.0, 0.0, 0.25),  (3.0, 0.4, 0.4)),
    ("flat_z_on_centre",   (0.0, 0.0, 0.25),  (0.4, 0.4, 0.0)),
    ("flat_z_between",     (0.0, 0.0, 0.30),  (0.4, 0.4, 0.0)),
    ("flat_z_at_face",     (0.0, 0.0, 1.50),  (0.4, 0.4, 0.0)),
    ("flat_yz",            (0.0, 0.0, 0.25),  (0.4, 0.0, 0.0)),
    ("flat_xy_between",    (0.0, 0.0, 0.30),  (0.0, 0.0, 0.4)),
    ("point_on_centre",    (0.05, 0.05, 0.25), (0.0, 0.0, 0.0)),
    ("point_between",      (0.0, 0.0, 0.30),  (0.0, 0.0, 0.0)),
]
results["reg_names"] = np.array([name for name, _, _ in REG_CASES])
results["reg_volumes"] = np.array([[*centre, *size] for _, centre, size in REG_CASES])
results["reg_atlas_size"] = np.asarray(REG_ATLAS_SIZE)
for tag, k in REG_K.items():
    reg_sim = mp.Simulation(cell_size=mp.Vector3(*WRAP_CELL), resolution=10, sources=WRAP_SOURCE,
                            force_complex_fields=True, k_point=mp.Vector3(*k))
    reg_atlas = reg_sim.add_dft_fields([mp.Ex, mp.Ez], 1.0, 0, 1, center=mp.Vector3(0, 0, 0),
                                       size=mp.Vector3(*REG_ATLAS_SIZE), decimation_factor=1)
    reg_monitors = {
        name: reg_sim.add_dft_fields([mp.Ex, mp.Ez], 1.0, 0, 1, center=mp.Vector3(*centre),
                                     size=mp.Vector3(*size), decimation_factor=1)
        for name, centre, size in REG_CASES
    }
    reg_sim.run(until=12)
    results[f"reg_{tag}_k"] = np.asarray(k)
    for component, label in ((mp.Ex, "Ex"), (mp.Ez, "Ez")):
        results[f"reg_{tag}_atlas_{label}"] = np.asarray(
            reg_sim.get_dft_array(reg_atlas, component, 0)
        )
        for name, _, _ in REG_CASES:
            results[f"reg_{tag}_{name}_{label}"] = np.asarray(
                reg_sim.get_dft_array(reg_monitors[name], component, 0)
            )

np.savez(output_path, **results)
'''


@pytest.fixture(scope="module")
def meep_reference(tmp_path_factory):  # One CPU-MEEP subprocess for every case in this file.
    directory = tmp_path_factory.mktemp("integration_oracle")
    script_path = directory / "oracle.py"
    script_path.write_text(_ORACLE_SCRIPT, encoding="utf-8")
    output_path = directory / "reference.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), str(output_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP integration oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {key: archive[key] for key in archive.files}


def _complex_relative_l2(candidate, reference):  # Relative L2 on the complex field: catches phase, not just amplitude.
    driver_field = np.asarray(candidate, dtype=np.complex128).ravel()
    meep_field = np.asarray(reference, dtype=np.complex128).ravel()
    reference_norm = float(np.linalg.norm(meep_field))
    candidate_norm = float(np.linalg.norm(driver_field))
    assert reference_norm > 0.0, "CPU-MEEP produced an all-zero field; the comparison would be meaningless."
    assert candidate_norm > 0.0, "The driver produced an all-zero field; an unmeasurable run must not score as perfect."
    return float(np.linalg.norm(driver_field - meep_field) / reference_norm)


def _custom_wave(time):  # The same waveform the oracle script drives MEEP's CustomSource with.
    return np.exp(-((time - 1.5) ** 2)) * np.exp(-2j * np.pi * time)


def _run_point_case(source, components, cell=(2, 2, 4), until=3.0):  # One driver run, host copies of the components.
    driver = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True)
    driver.add_source(source)
    driver.run(until=until)
    # MEEP's get_array returns N+1 points per axis, index 0 being the periodic
    # duplicate; index i + 1 is this grid's cell i.
    return [driver.get_field(name) for name in components]


def _compare(driver_fields, reference, components, bar=CROSS_VALIDATION_BAR):  # Per-component complex L2 against MEEP.
    errors = {}
    for index, name in enumerate(components):
        errors[name] = _complex_relative_l2(driver_fields[index], reference[index][1:, 1:, 1:])
    worst = max(errors.values())
    assert worst < bar, f"Complex relative L2 {errors} exceeds the {bar:.0%} cross-validation bar."
    return errors


@requires_meep
@skip_without_meep
def test_magnetic_source_through_the_driver_matches_cpu_meep(meep_reference):
    """A magnetic current added through ``add_source`` must land in the B half-step slot.

    Measured 2.9e-07 / 2.1e-07 / 2.1e-07 (Hx / Ey / Ez), the same order as the
    engine's core-algorithm parity. The control injects the identical source in
    the electric slot at ``t + dt/2`` instead of the magnetic slot at ``t``;
    that is a pure half-step phase error, so it is asserted to *exceed* the bar
    rather than merely differ, and it is measured on the complex field because a
    magnitude comparison cannot see it at all.
    """
    components = ("Hx", "Ey", "Ez")
    source = {"component": "Hx", "frequency": 1.0, "center": (0.0, 0.0, _SOURCE_Z), "size": (0.0, 0.0, 0.0)}
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    built = driver.add_source(source)
    assert isinstance(built, VolumeSource) and built.field_type == FIELD_TYPE_B
    driver.run(until=3.0)
    errors = _compare([driver.get_field(name) for name in components], meep_reference["magnetic"], components)
    assert max(errors.values()) < 1e-5, f"Magnetic-source parity drifted from the recorded ~3e-07: {errors}"

    # Control: same source, wrong slot. Injected after step_D at t + dt/2.
    wrong = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    wrong.add_source(source)
    for injected in wrong._sources:
        injected._field_type = FIELD_TYPE_D  # Force the electric slot.
    wrong.run(until=3.0)
    control = {
        name: _complex_relative_l2(wrong.get_field(name), meep_reference["magnetic"][index][1:, 1:, 1:])
        for index, name in enumerate(components)
    }
    assert min(control.values()) > CROSS_VALIDATION_BAR, (
        f"Injecting a magnetic source in the electric slot should break parity, but it stayed within "
        f"the bar: {control}. The bound this test asserts would then be satisfiable by the wrong code."
    )


@requires_meep
@skip_without_meep
def test_integrated_source_through_the_driver_matches_cpu_meep(meep_reference):
    """``is_integrated`` must reach the source from the driver's source dict.

    MEEP excludes an integrated source from ``step_source`` and subtracts its
    dipole inside ``update_eh`` instead, so the field sees the time integral of
    the current. Measured 1.8e-07 / 2.2e-07 / 2.1e-07 (Ez / Hx / Hy) at the
    2026-08-06 re-baselining on MEEP 1.33.0 single; the Ez case read 1.1e-07
    before the source-in-absorber wave's withdraw-before-every-curl-ladder
    rework, and the pre-wave engine still measures 1.13e-07 on the same 1.33
    oracle, so the movement is that wave's and not the oracle's. The
    control is the identical waveform with the flag dropped, asserted to exceed
    the bar — a driver that accepted ``is_integrated`` and quietly ignored it
    would otherwise pass.
    """
    components = ("Ez", "Hx", "Hy")
    base = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, _SOURCE_Z),
            "size": (0.0, 0.0, 0.0), "width": 1.0}
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    built = driver.add_source({**base, "is_integrated": True})
    assert isinstance(built, VolumeSource) and built.is_integrated
    driver.run(until=3.0)
    errors = _compare([driver.get_field(name) for name in components], meep_reference["integrated"], components)
    assert max(errors.values()) < 1e-5, f"is_integrated parity drifted from the recorded ~2e-07: {errors}"

    plain = _run_point_case(base, components)
    control = {
        name: _complex_relative_l2(plain[index], meep_reference["integrated"][index][1:, 1:, 1:])
        for index, name in enumerate(components)
    }
    assert min(control.values()) > CROSS_VALIDATION_BAR, (
        f"Dropping is_integrated should break parity against an integrated MEEP source, but it stayed "
        f"within the bar: {control}."
    )


@requires_meep
@skip_without_meep
def test_custom_envelope_through_the_driver_matches_cpu_meep(meep_reference):
    """``source_type='custom'`` with a ``src_func`` must reproduce MEEP's CustomSource.

    MEEP reads a non-integrated custom function as the current itself rather
    than differencing it (meep.hpp:1066-1071). Measured 1.8e-07 / 3.1e-07 /
    2.9e-07 (Ez / Hx / Hy).
    """
    components = ("Ez", "Hx", "Hy")
    fields = _run_point_case(
        {"component": "Ez", "center": (0.0, 0.0, _SOURCE_Z), "size": (0.0, 0.0, 0.0),
         "source_type": "custom", "src_func": _custom_wave},
        components,
    )
    errors = _compare(fields, meep_reference["custom"], components)
    assert max(errors.values()) < 1e-5, f"Custom-envelope parity drifted from the recorded ~3e-07: {errors}"


@requires_meep
@skip_without_meep
def test_magnetic_integrated_gaussian_source_drives_a_broadband_flux_spectrum(meep_reference):
    """Every new feature at once, through the driver, against MEEP's ``get_fluxes``.

    A magnetic source (B slot, ``t``) carrying a gaussian envelope with
    ``is_integrated`` (dipole offset across steps) radiates through a PML into a
    five-frequency flux monitor (leading frequency axis, one accumulation pass).
    Measured relative L2 3.8e-07 against MEEP over a band spanning 7.2e+04 in
    power, so a flat or mis-broadcast spectrum cannot sit inside the bound.

    This is the combination none of the three features was built or validated
    with: the source dispatch, the injection slot, the integrated bookkeeping and
    the spectrum axis all have to be right simultaneously.

    The flux spectrum alone is *not* enough to pin the injection slot: flux is a
    time-averaged power, so a source injected half a step late gives 3.850e-07
    against the correct 3.843e-07 — measured, by mutating the step loop. The
    complex DFT field spectrum is checked alongside it for exactly that reason;
    it moves to 1.6e-01 under the same mutation.
    """
    frequencies = meep_reference["combined_freqs"]
    reference = meep_reference["combined_flux"]
    driver = FdtdDriver(cell_size=(3, 3, 5), resolution=10, force_complex_fields=True)
    driver.setup_pml(8)
    source = driver.add_source({
        "component": "Hx", "frequency": 1.0, "center": (0.0, 0.0, _SOURCE_Z), "size": (1.0, 1.0, 0.0),
        "source_type": "gaussian", "fwidth": 0.6, "is_integrated": True,
    })
    assert source.field_type == FIELD_TYPE_B and source.is_integrated
    monitor = driver.add_flux_monitor(fcen=1.0, df=0.6, nfreq=5,
                                      center=(0.0, 0.0, _PLANE_Z), size=(1.4, 1.4, 0.0))
    field_monitor = driver.add_dft_monitor(fcen=1.0, df=0.6, nfreq=5, components=("Ey",),
                                           center=(0.0, 0.0, 0.55), size=(0.4, 0.4, 0.4))
    driver.run(until=25.0)
    spectrum = monitor.get_flux_spectrum()

    np.testing.assert_allclose(monitor.frequencies, frequencies, rtol=1e-12)
    assert spectrum.shape == reference.shape == (5,)
    # The band must actually be a band: a spectrum that came out flat would make
    # the relative-L2 bound below far easier to meet than it should be.
    assert float(reference.max() / reference.min()) > 1e3
    relative_l2 = float(np.linalg.norm(spectrum - reference) / np.linalg.norm(reference))
    assert relative_l2 < CROSS_VALIDATION_BAR, (
        f"Combined magnetic + integrated + broadband flux spectrum relative L2 {relative_l2:.3e} "
        f"exceeds the {CROSS_VALIDATION_BAR:.0%} bar.\nengine: {spectrum}\nmeep:   {reference}"
    )
    assert relative_l2 < 1e-4, f"Combined-feature parity drifted from the recorded 3.8e-07: {relative_l2:.3e}"
    # Per bin as well as in aggregate, so one dominant bin cannot carry the norm.
    for frequency, got, want in zip(frequencies, spectrum, reference):
        assert abs(got - want) / abs(want) < 1e-4, f"f={frequency:.4f}: engine {got:.6e} vs meep {want:.6e}"

    # The phase-sensitive arm: the complex DFT field spectrum over a volume.
    dft_reference = meep_reference["combined_dft_ey"]
    dft_spectrum = np.asarray(field_monitor.get_dft_spectrum("Ey"))
    assert dft_spectrum.shape == dft_reference.shape, (
        f"DFT field spectrum shape {dft_spectrum.shape} does not match MEEP's {dft_reference.shape}."
    )
    dft_l2 = _complex_relative_l2(dft_spectrum, dft_reference)
    assert dft_l2 < CROSS_VALIDATION_BAR, (
        f"Combined-feature complex DFT field spectrum relative L2 {dft_l2:.3e} exceeds the "
        f"{CROSS_VALIDATION_BAR:.0%} bar; unlike the flux above, this arm sees the injection slot."
    )
    assert dft_l2 < 1e-4, f"Combined-feature DFT field parity drifted from the recorded 2.5e-07: {dft_l2:.3e}"


def test_reset_clears_the_integrated_source_offset():
    """``reset()`` must clear a VolumeSource's applied-dipole offset along with the clock.

    An integrated source subtracts the *increment* ``dipole(t) - dipole(t_prev)``
    each step, so its state is the dipole it has already applied. Without
    resetting it, a second run starts by subtracting the increment from the old
    final dipole to the new initial one — a wrong but entirely plausible field,
    with no error anywhere. The check is bit-exact equality of the two runs,
    which is the only bound that distinguishes "reset" from "nearly reset".
    """
    def run_once():
        driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -0.35),
                           "size": (0.0, 0.0, 0.0), "width": 1.0, "is_integrated": True})
        return driver

    fresh = run_once()
    fresh.run(num_steps=40)
    first = fresh.get_field("Ez")

    reused = run_once()
    reused.run(num_steps=40)
    assert float(np.max(np.abs(reused._sources[0]._applied_dipole))) > 0.0, (
        "The integrated source never applied a dipole; this test would not be measuring a reset."
    )
    reused.reset()
    assert reused._sources[0]._applied_dipole == 0j
    reused.run(num_steps=40)
    second = reused.get_field("Ez")

    assert float(np.max(np.abs(first))) > 0.0, "The run carries no field; the comparison would be vacuous."
    np.testing.assert_array_equal(
        second, first,
        err_msg="A re-run after reset() diverges from a fresh run: the integrated dipole offset survived.",
    )


def test_decay_stop_accepts_a_magnetic_gaussian_source_and_refuses_an_endless_one():
    """The decay stop has to know when the new source types stop emitting.

    ``_last_source_time`` reads a ``VolumeSource`` through its envelope. A
    gaussian one ends at ``peak_time + cutoff*width`` and the run may proceed; a
    continuous one runs to MEEP's ~1e20 default ``end_time`` and the run is
    refused up front rather than stepping to the ceiling first. A source type it
    could not classify would raise instead of being assumed finished, which is
    what keeps the criterion from firing while a source is still driving.

    The probe is ``Ey``, not ``Ez``. An Hx dipole's E field circulates about the
    x axis, so on the z axis above it ``Ez`` is a radiation null: measured peak
    3.7e-07 against ``Ey``'s 4.9e-01, i.e. the null probe sees nothing but
    float32 noise, which never decays. That case is exercised too — it must fail
    loudly rather than report a decayed run, which is the whole point of
    refusing the criterion while the running peak is unmeasurable.
    """
    def build(source):  # The geometry the electric decay tests use, driven magnetically.
        driver = FdtdDriver(cell_size=(2.0, 2.0, 4.0), resolution=10, force_complex_fields=True)
        driver.setup_pml(8)
        driver.add_source(source)
        return driver

    decay_arguments = {"until_after_sources": 2.0, "decay_by": 1e-4, "decay_component": "Ey",
                       "decay_point": (0.0, 0.0, 0.55), "max_time": 200.0}
    magnetic_pulse = {"component": "Hx", "frequency": 1.0, "center": (0.0, 0.0, -0.5),
                      "size": (0.0, 0.0, 0.0), "source_type": "gaussian", "fwidth": 1.0}

    pulsed = build(magnetic_pulse)
    assert pulsed._last_source_time() == pytest.approx(10.0, rel=1e-9)  # peak 5.0 + cutoff 5 * width 1.
    pulsed.run(**decay_arguments)
    assert pulsed.time < 200.0, "The magnetic pulse never decayed inside the ceiling."
    assert pulsed.time > 10.0, "The run stopped before the source had finished emitting."
    assert float(np.max(np.abs(pulsed.get_field("Ey")))) > 0.0, "The decayed run never carried a field."

    # The same run probed at its radiation null must fail loudly, not report a
    # decayed field: there the peak is float32 noise and the ratio never falls.
    blind = build(magnetic_pulse)
    with pytest.raises(RuntimeError, match="had not decayed"):
        blind.run(**{**decay_arguments, "decay_component": "Ez"})

    endless = build({"component": "Hx", "frequency": 1.0, "center": (0.0, 0.0, -0.5), "size": (0.0, 0.0, 0.0)})
    assert endless._last_source_time() > 1e19
    with pytest.raises(ValueError, match="still emitting"):
        endless.run(**decay_arguments)

    # A custom waveform is opaque, so its default end_time is MEEP's ~1e20 and a
    # decay stop is refused rather than guessing when the function goes quiet.
    custom = build({"component": "Ez", "center": (0.0, 0.0, -0.5), "size": (0.0, 0.0, 0.0),
                    "source_type": "custom", "src_func": _custom_wave})
    with pytest.raises(ValueError, match="still emitting"):
        custom.run(**decay_arguments)
    bounded = build({"component": "Ez", "center": (0.0, 0.0, -0.5), "size": (0.0, 0.0, 0.0),
                     "source_type": "custom", "src_func": _custom_wave, "end_time": 5.0})
    assert bounded._last_source_time() == pytest.approx(5.0)
    bounded.run(**decay_arguments)
    assert bounded.time > 5.0, "The run stopped before the custom source's end_time."


def test_zero_amplitude_magnetic_source_leaves_b_exactly_zero():
    """A silent magnetic source must leave B exactly zero, with a positive control.

    Exactly zero, not small: the injection is skipped when the scale is 0, so
    round-off cannot accumulate. The control is the identical geometry at unit
    amplitude, asserted to write a nonzero B — without it this would pass just as
    well against a source that was never wired to the B slot at all.
    """
    def run(amplitude):
        driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
        driver.add_source({"component": "Hy", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                           "size": (0.0, 0.0, 0.0), "amplitude": amplitude})
        driver.run(num_steps=30)
        return driver

    quiet = run(0.0)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        values = quiet.get_field(name)
        assert np.count_nonzero(values) == 0, f"A zero-amplitude magnetic source left {name} nonzero."

    loud = run(1.0)
    assert np.count_nonzero(loud.get_field("By")) > 0, (
        "The unit-amplitude control wrote nothing to By; the zero-amplitude assertion above is vacuous."
    )
    assert np.count_nonzero(loud.get_field("Dx")) > 0, "The magnetic source never coupled into the electric field."


def test_new_source_dispatch_rejects_every_malformed_spelling():
    """The widened dispatch must stay loud: no key silently ignored, no kind silently defaulted."""
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10)
    point = {"center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)}

    # A component that names no field at all.
    with pytest.raises(ValueError, match="Unknown source component"):
        driver.add_source({"component": "Jz", "frequency": 1.0, **point})
    # A custom source without its waveform: the one key that makes it a source.
    with pytest.raises(ValueError, match="src_func"):
        driver.add_source({"component": "Ez", "source_type": "custom", **point})
    with pytest.raises(ValueError, match="callable"):
        driver.add_source({"component": "Ez", "source_type": "custom", "src_func": 3.0, **point})
    # A custom waveform carries no carrier, so 'frequency' must be refused rather
    # than accepted and ignored.
    with pytest.raises(ValueError, match="Unknown source keys"):
        driver.add_source({"component": "Ez", "source_type": "custom", "src_func": _custom_wave,
                           "frequency": 1.0, **point})
    # 'fwidth' belongs to the gaussian kind only, on the magnetic path too.
    with pytest.raises(ValueError, match="Unknown source keys"):
        driver.add_source({"component": "Hz", "frequency": 1.0, "fwidth": 0.2, "is_integrated": True, **point})
    # A frequency-carrying kind still needs its frequency.
    with pytest.raises(ValueError, match="needs 'frequency'"):
        driver.add_source({"component": "Hz", **point})
    # An integrated source RUNS folded now (the fold parity is pinned against CPU
    # MEEP in test_from_meep.py); the dispatch must accept the spelling.
    folded = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, symmetry=("X",))
    folded.add_source({"component": "Ez", "frequency": 1.0, "is_integrated": True, **point})


def test_electric_and_magnetic_sources_keep_their_own_injection_slots():
    """Two sources of different families in one run must each land in their own slot.

    The step loop walks ``_sources`` twice and filters on ``field_type``. A loop
    that injected everything once — in either slot — would put one family half a
    step out, so this asserts each family reproduces, cell for cell, what it
    produces alone in the same geometry. Superposition holds exactly here because
    the run is linear and the two sources drive different primary arrays.
    """
    def build(sources):
        driver = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, force_complex_fields=True)
        for source in sources:
            driver.add_source(source)
        driver.run(num_steps=25)
        return driver

    electric = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -0.55), "size": (0.0, 0.0, 0.0)}
    magnetic = {"component": "Hx", "frequency": 1.2, "center": (0.0, 0.0, 0.55), "size": (0.0, 0.0, 0.0)}

    together = build([electric, magnetic])
    # Reversed creation order must not matter: the slots are chosen per source.
    reversed_order = build([magnetic, electric])
    alone_electric = build([electric])
    alone_magnetic = build([magnetic])

    for name in ("Ez", "Ey", "Hx", "Hy"):
        combined = together.get_field(name)
        superposed = alone_electric.get_field(name) + alone_magnetic.get_field(name)
        scale = float(np.max(np.abs(combined)))
        assert scale > 0.0, f"{name} is identically zero; the superposition check would be vacuous."
        deviation = float(np.max(np.abs(combined - superposed))) / scale
        assert deviation < 1e-5, (
            f"{name}: a two-family run does not equal the sum of the single-family runs ({deviation:.3e}); "
            f"one family is being injected in the wrong slot."
        )
        np.testing.assert_array_equal(
            reversed_order.get_field(name), combined,
            err_msg=f"{name} depends on the order the sources were added.",
        )

    # And the two families really did both contribute, so the equality above is not
    # comparing a field against itself plus zero.
    assert float(np.max(np.abs(alone_electric.get_field("Ez")))) > 0.0
    assert float(np.max(np.abs(alone_magnetic.get_field("Hx")))) > 0.0


@pytest.mark.parametrize("axis,phase", [(axis, phase)
                                        for axis in ("X", "Y", "Z")
                                        for phase in (+1, -1)])
def test_odd_parity_sources_are_refused_on_a_mirror_plane(axis, phase):
    """A source antisymmetric about an active mirror must raise, not run wrong.

    A source centre sits on every active plane, and the fold reconstructs the far
    half as ``mirror_parity`` times this one. A component the plane makes ODD is
    therefore cancelled by its own mirror image, and before this guard the run
    completed and disagreed with the equivalent full-domain run by 1.48e+00 (Ex
    under an even X) and 9.06e-01 (Hy/Hz under an even X) — a full, plausible
    field that is simply wrong.

    THE RULE IS NOW DERIVED FROM THE PLANE, NOT FROM THE EVEN X/Y TABLE. The
    refusal used to read ``fields.SYMMETRY_PHASES``, which is ``mirror_parity``
    frozen at MEEP's default ``phase=+1`` on X and Y alone. That made the rule
    correct only for even X/Y mirrors, refused an ``Ex`` dipole outright — the
    one case ``mp.Mirror(direction, phase=-1)`` exists for — and said nothing at
    all about Z. It is asserted here against ``mirror_parity`` evaluated at the
    plane's OWN declared phase, over every axis and both phases, so the accepted
    and refused sets are exactly complementary between ``+1`` and ``-1``:

        Mirror('X', +1)   refuses Ex, Hy, Hz     Mirror('X', -1)   refuses Ey, Ez, Hx

    The accepted half is asserted to REPRODUCE the full-domain run, so this
    cannot pass by refusing everything — and the odd-phase half could not pass at
    all before the descriptor was threaded, because ``FdtdDriver`` had no way to
    ask for it.
    """
    from .fields import mirror_parity
    from .grid import Mirror

    index = "XYZ".index(axis)
    primaries = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")
    cell = [2.0, 2.0, 2.0]
    cell[2 if index != 2 else 0] = 4.0  # Long on an unfolded axis, so nothing reaches a wall.
    cell = tuple(cell)
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        source = {"component": component, "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                  "size": (0.0, 0.0, 0.0), "source_type": "gaussian", "fwidth": 1.0}
        folded = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True,
                            symmetry=(Mirror(axis, phase),))
        if mirror_parity(component, index, phase) < 0:
            with pytest.raises(ValueError, match="cannot carry the parity the fold demands"):
                folded.add_source(source)
            continue
        # Even about the plane: accepted, and it must fold exactly.
        full = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True)
        folded.add_source(source)
        full.add_source(source)
        folded.run(num_steps=12)
        full.run(num_steps=12)
        scale = max(float(np.max(np.abs(full.get_field(name, cell_centered=False)))) for name in primaries)
        assert scale > 0.0, f"{component} under {axis}{phase:+d}: the run carries no field."
        worst = max(
            float(np.max(np.abs(
                folded.get_field(name, cell_centered=False)
                - _quadrant_of(full.get_field(name, cell_centered=False),
                               ((index, phase),), full.shape))))
            for name in primaries
        )
        assert worst / scale < 1e-5, (
            f"{component} is even about {axis}{phase:+d} but the folded run diverges "
            f"by {worst / scale:.3e}."
        )


@requires_meep
@skip_without_meep
def test_flux_plane_lands_where_it_was_asked_for_across_a_whole_cell(meep_reference):
    """A flux plane must read correctly at *every* normal coordinate, not just the sample planes.

    Sixteen planes, one run, z stepping 0.01 from 0.95 to 1.10 at resolution 10 —
    a cell and a half. Two of those requests (0.95, 1.05) land on a row of cell
    centres; two (1.00, 1.10) land on a round multiple of ``dx``, exactly halfway
    between two rows; the rest land at arbitrary fractions of a cell. Each is
    cross-validated against its own CPU-MEEP flux from the same geometry.

    This replaces a pin that recorded the opposite. The monitor's normal axis used
    to collapse to a single index through ``round(c*a + L*a/2 - 0.5)``, which at a
    round coordinate hits a ``.5`` tie: Python's round-half-to-even picked a side
    half a cell from where MEEP puts the plane, and the flux came back ~7% wrong
    — through the 5% bar every other cross-validation here is held to — at
    precisely the coordinate a user is most likely to type. Worse, the whole cell
    of requests between two sample planes returned one bit-identical number.

    The fix is MEEP's own rule, not a snap: a zero-thickness axis is case 4 of
    loop_in_chunks.cpp ("interpolation, not integration"), and the flux integrand
    is linearly interpolated between the two rows that bracket the request
    (``dft.FluxMonitor._setup_region``). Worst error over these sixteen positions
    9.3e-07, against 7.2e-02 before, with five of the sixteen previously outside
    the 5% bar and none now — so the placement sits with the engine's other flux
    numbers instead.

    Three independent things are asserted, because the value comparison alone
    could be met by luck on a coarse sweep:

    * every position matches MEEP (the bar, and the recorded drift bound);
    * the engine's own sixteen values are all distinct — the collapse this
      replaces returned one number for nine consecutive requests, which no
      value-versus-MEEP bound at a single position can see;
    * the sampled cells bracket the request and the normal weights sum to 1, so
      the plane is *interpolated* onto its coordinate rather than widened into a
      two-cell slab that double-counts.
    """
    sweep_z = [float(z) for z in meep_reference["sweep_z"]]
    reference = np.asarray(meep_reference["sweep_flux"], dtype=np.float64)
    assert len(sweep_z) == 16 and reference.shape == (16,)

    driver = FdtdDriver(cell_size=(3, 3, 5), resolution=10, force_complex_fields=True)
    driver.setup_pml(8)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, _SOURCE_Z),
                       "size": (1.0, 1.0, 0.0), "source_type": "gaussian", "fwidth": 0.8})
    monitors = [
        driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, plane_z), size=(1.4, 1.4, 0.0))
        for plane_z in sweep_z
    ]
    driver.run(until=25.0)
    measured = np.array([monitor.get_flux() for monitor in monitors], dtype=np.float64)

    # MEEP itself must resolve the sixteen requests to sixteen different fluxes:
    # against a reference that did not move, matching it would prove nothing.
    spread = (reference.max() - reference.min()) / reference.max()
    assert spread > 0.1, (
        f"CPU MEEP's flux varies by only {spread:.1%} across the swept cell; this test could not "
        f"tell a correctly placed plane from a snapped one."
    )
    assert len(set(reference.tolist())) == 16, "CPU MEEP returned a repeated flux across the sweep."

    errors = np.abs(measured - reference) / np.abs(reference)
    report = "\n".join(
        f"  z={z:.2f}: engine {got:.6e} meep {want:.6e} rel {err:.3e} cells {monitor._region[4:6]}"
        for z, got, want, err, monitor in zip(sweep_z, measured, reference, errors, monitors)
    )
    assert errors.max() < CROSS_VALIDATION_BAR, (
        f"A flux plane misses MEEP by {errors.max():.3e} somewhere in the swept cell, past the "
        f"{CROSS_VALIDATION_BAR:.0%} bar:\n{report}"
    )
    assert errors.max() < 5e-6, f"Flux-plane placement drifted from the recorded ~9e-07:\n{report}"
    # Called out by name: these are the two coordinates the old snapping put half a
    # cell out, at 7.2e-02 and 7.0e-02.
    for round_coordinate in (1.00, 1.10):
        index = sweep_z.index(round_coordinate)
        assert errors[index] < 5e-6, (
            f"z = {round_coordinate} is a round multiple of dx, the tie the old registration lost: "
            f"{errors[index]:.3e}."
        )

    # No two requests may resolve to the same measurement. The defect this replaces
    # returned one bit-identical number for every request in a cell, which a
    # per-position comparison against MEEP catches only where the collapse lands
    # far from the truth.
    assert len(set(measured.tolist())) == 16, (
        f"The engine returned the same flux for different plane positions; a cell of requested "
        f"coordinates is collapsing onto one placement:\n{report}"
    )
    for index in range(1, 16):
        step = abs(measured[index] - measured[index - 1]) / abs(measured[index])
        assert step > 1e-3, (
            f"z = {sweep_z[index - 1]:.2f} and z = {sweep_z[index]:.2f} differ by only {step:.3e}; "
            f"0.01 of a 0.1 cell should move the flux by ~1.5e-03."
        )

    # Placement, independent of any value: the cells the monitor samples must
    # bracket the requested coordinate, and the weights along the normal must sum
    # to 1 — an interpolation. A region widened to two cells but summed with
    # weight 1.0 each would pass every bound above only by doubling the power.
    grid_z = np.asarray(to_numpy(driver.grid.z), dtype=np.float64)
    for plane_z, monitor in zip(sweep_z, monitors):
        z0, z1 = monitor._region[4:6]
        assert z1 - z0 in (1, 2), f"z = {plane_z} sampled {z1 - z0} cells along the normal."
        lower, upper = grid_z[z0], grid_z[z1 - 1]  # float32 axis: 1e-5 is a few ULP, not a real slack.
        assert lower - 1e-5 <= plane_z <= upper + 1e-5, (
            f"z = {plane_z} is not bracketed by the cells it samples ({lower:.3f}..{upper:.3f})."
        )
        if z1 - z0 == 1:  # One cell only when the request is that cell's own centre.
            assert abs(lower - plane_z) < 1e-5, (
                f"z = {plane_z} collapsed onto the single plane {lower:.3f} half a cell away."
            )
        weights = np.asarray(to_numpy(monitor._weights), dtype=np.float64)
        total = float(weights.sum())
        assert total == pytest.approx(1.4 * 10 * 1.4 * 10, rel=1e-5), (
            f"z = {plane_z}: the surface weights integrate to {total:.4f} rather than the 196 cells "
            f"of a 1.4 x 1.4 plane; the normal axis is being summed, not interpolated."
        )
        # The weight column through an interior in-plane cell is the normal-axis
        # profile alone, and it must sum to 1: two cells at 0.5, or one at 1.0.
        column = weights[weights.shape[0] // 2, weights.shape[1] // 2, :]
        assert float(column.sum()) == pytest.approx(1.0, rel=1e-6), (
            f"z = {plane_z}: the normal-axis weights {column} do not sum to 1."
        )


@requires_meep
@skip_without_meep
def test_flux_monitor_with_thickness_carries_meeps_volume_measure(meep_reference):
    """A monitor given a depth along its own normal must reproduce MEEP, not a plane's number.

    MEEP picks up one factor of ``1/a`` per direction the monitor actually
    extends along (loop_in_chunks.cpp, ``dV0``), so a region with three extents is
    integrated over a volume and returns a power times a length rather than a
    power. This engine hard-coded ``dx^2`` regardless, which made every such
    monitor read exactly ``1/dx`` — a factor of ten at resolution 10 — too large,
    with a perfectly plausible number and no error.

    Every thickness reaches the class through the driver's public path. A whole
    cell or more on a DECLARED normal used to be refused there and the monitor
    had to be built by hand; it is not refused any more, because MEEP allows the
    same shape and its own ``test_visualization.py`` writes one
    (``FluxRegion(size=(4,4,4), direction=mp.X)``) — the refusal was an
    accepted-at-the-gate, raises-on-lift hole, and closing it is what
    ``test_from_meep.py::test_flux_region_may_extend_along_its_own_normal``
    measures against ``mp.get_fluxes``. An INFERRED normal with extent still
    raises, since there the axis would be a guess.

    Three thicknesses are checked against CPU MEEP, and the flat plane in the
    same run is the control that the ordinary case did not move: measured
    3.9e-07 / 1.3e-07 / 2.4e-07, against the hard-coded surface measure's factor
    of ten.
    """
    thicknesses = [float(value) for value in meep_reference["thicknesses"]]
    reference = np.asarray(meep_reference["thickness_flux"], dtype=np.float64)
    driver = FdtdDriver(cell_size=(3, 3, 5), resolution=10, force_complex_fields=True)
    driver.setup_pml(8)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, _SOURCE_Z),
                       "size": (1.0, 1.0, 0.0), "source_type": "gaussian", "fwidth": 0.8})
    monitors = []
    for thickness in thicknesses:
        plane = {"frequencies": 1.0, "center": (0.0, 0.0, _PLANE_Z),
                 "size": (1.4, 1.4, thickness), "direction": 2}
        monitors.append(driver.add_flux_monitor(**plane))
    # The normal must still be DECLARED: a size with no flat axis and no direction
    # is an unlabelled volume, and guessing its normal is what this refusal stops.
    with pytest.raises(ValueError, match="no flat axis"):
        driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, _PLANE_Z),
                                size=(1.4, 1.4, 0.4))
    flat = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, _PLANE_Z), size=(1.4, 1.4, 0.0))
    driver.run(until=25.0)
    for thickness, monitor, want in zip(thicknesses, monitors, reference):
        got = monitor.get_flux()
        error = abs(got - want) / abs(want)
        assert error < CROSS_VALIDATION_BAR, (
            f"thickness {thickness}: engine {got:.6e} vs meep {want:.6e} ({error:.3e}), past the "
            f"{CROSS_VALIDATION_BAR:.0%} bar."
        )
        assert error < 5e-6, f"thickness {thickness} drifted from the recorded ~3e-07: {error:.3e}."
        assert monitor._measure == pytest.approx(driver.grid.dx**3), (
            f"thickness {thickness} kept a surface measure; MEEP integrates it over a volume."
        )
    # The control: a thick monitor is a different measurement from a flat one, so
    # the bound above cannot be met by ignoring the thickness. The flat plane in
    # the same run keeps the surface measure and its own MEEP-matching value.
    assert flat._measure == pytest.approx(driver.grid.dx**2)
    flat_reference = float(meep_reference["sweep_flux"][10])  # sweep index 10 is z = 1.05.
    assert abs(flat.get_flux() - flat_reference) / abs(flat_reference) < 5e-6
    assert abs(reference[0]) < 0.1 * abs(flat_reference), (
        "The thin monitor's volume integral is not distinguishable from the flat plane's power."
    )


def test_flux_planes_bracketing_a_source_conserve_power_at_any_offset():
    """Two planes equidistant from a symmetric source must carry equal and opposite power.

    An analytic property of the configuration rather than a recorded number: an
    ``Ez`` sheet at the centre of a vacuum cell radiates symmetrically, so with
    no absorber between the two planes whatever leaves through one must arrive
    through the other. Nothing here needs CPU MEEP.

    The offsets are the point. ``0.45`` puts both planes on a row of cell centres
    — the only case the engine handled before — while ``0.53``, ``0.70`` and
    ``1.00`` put them between rows, where each plane is now interpolated from the
    two rows that bracket it. The bound is bit-exact equality, not a tolerance:
    the interpolation weights on the two sides are mirror images, so the
    cancellation is exact in float32 and any asymmetry in the placement shows up
    immediately. Measured 0.0 at every offset.

    The geometry is symmetric on purpose. Move the source half a cell up, so the
    two PML boundaries sit at different distances, and the same check measures
    4.4e-05 here against CPU MEEP's own 4.3e-05 in that configuration: a real
    asymmetry of the geometry, which would mask the placement error this test
    exists to catch.
    """
    driver = FdtdDriver(cell_size=(3.0, 3.0, 5.0), resolution=10, force_complex_fields=True)
    driver.setup_pml(8)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                       "size": (1.0, 1.0, 0.0), "source_type": "gaussian", "fwidth": 0.8})
    pairs = {
        offset: (
            driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, -offset), size=(1.4, 1.4, 0.0)),
            driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, +offset), size=(1.4, 1.4, 0.0)),
        )
        for offset in (0.45, 0.53, 0.70, 1.00)
    }
    driver.run(until=25.0)

    for offset, (below, above) in pairs.items():
        upstream, downstream = below.get_flux(), above.get_flux()
        assert downstream > 0.0, f"offset {offset}: power must flow away from the source on the +z side."
        assert upstream < 0.0, f"offset {offset}: power must flow away from the source on the -z side."
        assert upstream == -downstream, (
            f"offset {offset}: planes at -{offset} and +{offset} of a symmetric source carry "
            f"{upstream:.9e} and {downstream:.9e}, an asymmetry of "
            f"{abs(upstream + downstream) / abs(downstream):.3e}."
        )

    # The offsets really do sample different placements, so the equality above is
    # not four repetitions of one measurement: an interpolated plane spans two
    # cells, a plane on a row of centres spans one, and the power falls off with
    # distance from the source.
    widths = {offset: above._region[5] - above._region[4] for offset, (_, above) in pairs.items()}
    assert widths[0.45] == 1 and widths[0.53] == 2 and widths[0.70] == 2 and widths[1.00] == 2, widths
    magnitudes = [pairs[offset][1].get_flux() for offset in (0.45, 0.53, 0.70, 1.00)]
    assert all(later < earlier for earlier, later in zip(magnitudes, magnitudes[1:])), (
        f"Flux does not fall with distance from the source: {magnitudes}."
    )


_WRAP_CELL = (2.0, 2.0, 3.0)
_WRAP_SOURCE = {"component": "Ez", "frequency": 1.0, "fwidth": 0.8, "source_type": "gaussian",
                "center": (0.0, 0.0, -0.55), "size": (1.0, 1.0, 0.0)}
_WRAP_PLANES = [
    ("interior_1.23", (0.0, 0.0, 1.23), (1.4, 1.4, 0.0)),
    ("last_centre_1.45", (0.0, 0.0, 1.45), (1.4, 1.4, 0.0)),
    ("past_hi_face_1.47", (0.0, 0.0, 1.47), (1.4, 1.4, 0.0)),
    ("past_lo_face_-1.47", (0.0, 0.0, -1.47), (1.4, 1.4, 0.0)),
    ("on_hi_face_1.50", (0.0, 0.0, 1.50), (1.4, 1.4, 0.0)),
    ("full_cross_section", (0.0, 0.0, 0.55), (2.0, 2.0, 0.0)),
    ("wider_than_the_cell", (0.0, 0.0, 0.55), (3.0, 3.0, 0.0)),
]


@requires_meep
@skip_without_meep
def test_flux_plane_reaching_past_a_periodic_face_matches_cpu_meep(meep_reference):
    """A flux plane is continued into the next lattice image, not clipped at the face.

    MEEP does not stop a monitor at a periodic boundary: ``loop_in_chunks``
    (lines 390-415) enumerates the lattice shifts whose translated cell still
    intersects the request and weights each copy by ``pow(eikna[d], ishift)``.
    Clipping instead dropped whatever the far face carried, and did it in silence.
    Measured here against CPU MEEP in a cell with no absorber, before the fix and
    after:

        last row of cell centres  z = 1.45   9.1e-02 -> 1.7e-07
        a fifth of a cell past it z = 1.47   3.1e-01 -> 8.1e-08
        past the low face         z = -1.47  1.6e-01 -> 1.6e-07
        on the face itself        z = 1.50   6.0e-01 -> 1.8e-07
        the full cross-section    2 x 2      5.1e-02 -> 1.7e-07
        wider than the cell       3 x 3      1.5e+00 -> 8.0e-07

    None of those is an exotic request: the first is the last row of samples the
    grid has, and the fifth is the standard transmission plane through a periodic
    cell. The interior plane at z = 1.23 is the control — it never reached a face,
    it took the contiguous path before and after, and it must not move.

    ``wider_than_the_cell`` is the multi-image case: the ladder covers 32 cells of
    a 20-cell axis, so cells are sampled twice with different lattice shifts, which
    is MEEP's tiling reproduced rather than approximated.
    """
    names = [str(name) for name in meep_reference["wrap_names"]]
    reference = meep_reference["wrap_flux"]
    driver = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    driver.add_source(_WRAP_SOURCE)
    monitors = [driver.add_flux_monitor(frequencies=1.0, center=centre, size=size)
                for _, centre, size in _WRAP_PLANES]
    driver.run(until=12.0)

    report, worst = [], 0.0
    for (name, _, _), monitor in zip(_WRAP_PLANES, monitors):
        want = float(reference[names.index(name)])
        got = monitor.get_flux()
        error = abs(got - want) / abs(want)
        worst = max(worst, error)
        report.append(f"  {name}: engine {got:.6e} meep {want:.6e} rel {error:.3e}")
        assert error < CROSS_VALIDATION_BAR, (
            f"{name}: engine {got:.6e} vs meep {want:.6e} ({error:.3e}), past the "
            f"{CROSS_VALIDATION_BAR:.0%} bar.\n" + "\n".join(report)
        )
    # Far tighter than the 5% bar: with no absorber this is the same core algorithm
    # as the 1.7e-07 no-PML floor, so anything above round-off is a real defect.
    assert worst < 5e-6, f"Worst position drifted from the recorded ~2e-07:\n" + "\n".join(report)

    # The planes must be distinguishable measurements, or the bound above could be
    # met by returning one number for all of them.
    values = [monitor.get_flux() for monitor in monitors]
    assert len(set(values)) == len(values), f"Planes returned duplicate values: {values}"
    # And the two that wrap really do sample cells outside the stored axis, rather
    # than passing because the clipped answer happened to be close.
    assert any(monitor._wrapped[2] for monitor in monitors), "No plane exercised the z wrap."
    assert monitors[5]._wrapped[:2] == (True, True), "The full cross-section must wrap in x and y."
    assert monitors[0]._wrapped == (False, False, False), (
        "The interior control must still take the contiguous path."
    )


@requires_meep
@skip_without_meep
def test_bloch_periodic_cell_flux_matches_cpu_meep(meep_reference):
    """Bloch boundaries and a flux plane in one run — the grating transmission measurement.

    The two features were built independently and collide exactly here: a plane
    spanning the full periodic cross-section samples the cells one lattice vector
    outside the cell, and those cells carry the boundary phase. Flux is
    ``Re(E conj(H))``, so a phase applied to a whole image cancels — what does not
    cancel is the phase inside the Yee average that straddles the face, which is
    where a dropped factor shows up.

    Measured against CPU MEEP with the wrap in place: the interior plane 6.2e-07
    and 3.9e-08, the spanning plane 1.7e-07 and 2.8e-08 at k = (0.3, 0) and
    (0.15, -0.25). Before the wrap the spanning plane read 4.9e-03 and 1.2e-02
    while the interior plane was already exact, which is what a cross-feature
    defect looks like: each feature correct alone, wrong together.
    """
    wavevectors = [tuple(float(value) for value in row) for row in meep_reference["grating_k"]]
    reference = meep_reference["grating_flux"]
    report = []
    for index, (kx, ky) in enumerate(wavevectors):
        driver = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True,
                            k_point=(kx, ky, 0.0))
        driver.add_source(_WRAP_SOURCE)
        inside = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 0.55), size=(1.4, 1.4, 0.0))
        spanning = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 0.55), size=(2.0, 2.0, 0.0))
        driver.run(until=12.0)
        assert inside._wrapped == (False, False, False)
        assert spanning._wrapped[:2] == (True, True), "The spanning plane must cross both faces."

        for label, monitor, want in (("interior", inside, float(reference[index][0])),
                                     ("spanning", spanning, float(reference[index][1]))):
            got = monitor.get_flux()
            error = abs(got - want) / abs(want)
            report.append(f"  k=({kx},{ky}) {label}: engine {got:.6e} meep {want:.6e} rel {error:.3e}")
            assert error < 5e-6, (
                f"k=({kx},{ky}) {label}: engine {got:.6e} vs meep {want:.6e} ({error:.3e}).\n"
                + "\n".join(report)
            )
        # The spanning plane is a different measurement from the interior one, so
        # the pair cannot be passing on one number, and a run that ignored k_point
        # would have to reproduce a different reference per k.
        assert abs(spanning.get_flux() - inside.get_flux()) > 1e-3 * abs(inside.get_flux())

    assert abs(reference[0][1] - reference[1][1]) > 1e-3 * abs(reference[0][1]), (
        "The two wavevectors must give different spanning-plane power, or one reference "
        "would satisfy both and the k dependence would be untested."
    )


@requires_meep
@skip_without_meep
def test_ignoring_the_boundary_phase_fails_the_bloch_flux_comparison(meep_reference):
    """The negative control for the test above: k = 0 must not reproduce a Bloch reference.

    Without this, a driver that accepted ``k_point`` and quietly dropped it — or a
    monitor that wrapped without the phase — would still pass, because flux is a
    real power whose overall scale is set mostly by the source.
    """
    reference = meep_reference["grating_flux"]
    driver = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    driver.add_source(_WRAP_SOURCE)
    spanning = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 0.55), size=(2.0, 2.0, 0.0))
    driver.run(until=12.0)
    for index in range(reference.shape[0]):
        want = float(reference[index][1])
        blind = abs(spanning.get_flux() - want) / abs(want)
        assert blind > 0.05, (
            f"A k = 0 run reproduces the Bloch reference at index {index} to {blind:.3e}, so the "
            f"comparison cannot tell whether the boundary phase reached the monitor."
        )


def test_an_absorbing_run_still_wraps_its_flux_planes_where_meep_does():
    """A PML is a material, so it removes no monitor's wrap.

    MEEP takes a lattice shift where ``boundaries[High][d] == Periodic``
    (loop_in_chunks.cpp) — and ``fields::use_bloch`` makes that every direction of any
    run carrying a ``k_point``, absorber or not. Clipping an absorbing axis instead put
    the monitors on a different boundary condition from the stepping that fills them:
    measured worst per-frequency flux against CPU MEEP through a plane spanning the full
    cross section of a cell whose transverse axes absorb, 2.3e-02 clipped against
    1.6e-07 wrapped (``test_driver_vs_meep.py::
    test_a_flux_plane_spanning_an_absorbing_axis_matches_cpu_meep``).
    """
    absorbing = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, force_complex_fields=True)
    absorbing.setup_pml(4)
    assert absorbing._monitor_periodic_axes() == (True, True, True)
    wrapped = absorbing.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 1.45), size=(1.0, 1.0, 0.0))
    assert wrapped._wrapped == (False, False, True)

    # The control: the same plane in the same cell without the absorber, which must give
    # the identical answer — the layer is what used to change it, and no longer does.
    open_cell = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, force_complex_fields=True)
    assert open_cell._monitor_periodic_axes() == (True, True, True)
    assert open_cell.add_flux_monitor(
        frequencies=1.0, center=(0.0, 0.0, 1.45), size=(1.0, 1.0, 0.0)
    )._wrapped == wrapped._wrapped

    # And a mirror-folded axis never wraps, whatever the absorber does: MEEP reaches its
    # far half through the symmetry transform, which this engine's flux does not unfold.
    folded = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, symmetry=("X",),
                        force_complex_fields=True)
    assert folded._monitor_periodic_axes() == (False, True, True)


def test_adding_a_pml_under_a_wrapping_flux_monitor_is_allowed_and_order_free():
    """``setup_pml`` after ``add_flux_monitor`` is a legal ordering, and changes nothing.

    It used to raise, on the reasoning that the new absorber invalidated a region built
    on periodic cells. It does not: a monitor's wrap flags come from the symmetry alone,
    so the two creation orders build the same region and accumulate the same spectrum.
    Pinned both ways, because "the order does not matter" is exactly the kind of claim
    that stops being true silently.
    """
    def build(pml_first):
        driver = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, force_complex_fields=True)
        if pml_first:
            driver.setup_pml(4)
            monitor = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 1.45),
                                              size=(1.0, 1.0, 0.0))
        else:
            monitor = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 1.45),
                                              size=(1.0, 1.0, 0.0))
            driver.setup_pml(4)
        driver.add_source(dict(component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                               source_type="gaussian", frequency=1.0, fwidth=0.5))
        driver.run(until=3.0)
        return monitor

    pml_first, monitor_first = build(True), build(False)
    assert pml_first._region == monitor_first._region
    assert pml_first._wrapped == monitor_first._wrapped == (False, False, True)
    first, second = (np.asarray(m.get_flux_spectrum(), dtype=float) for m in (pml_first, monitor_first))
    assert float(np.max(np.abs(first))) > 0.0, "an empty spectrum would make the comparison vacuous"
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


@requires_meep
@skip_without_meep
def test_dft_region_touching_a_periodic_face_keeps_its_yee_average(meep_reference):
    """A DFT region ending at the last stored cell must still be interpolated to centres.

    The Yee-to-centre average consumes one plane past the region in every
    direction whose Yee shift is 0. Where the region ended at the last stored
    cell, `_sliced_component` used to clip that plane away — and the cost was not
    the far plane alone. With nothing left to consume, `_apply_yee_interpolation`
    skipped the average for the **whole region** on that axis, so a slab whose
    upper z face is the cell face came back **2.7e-01** wrong overall and
    **4.7e-01** wrong on its *first* plane, six cells from the boundary. The
    default region of a DFT monitor in a run with no PML is the whole grid, so
    every interpolated axis of the default monitor carried it.

    Now the plane is fetched from the neighbouring lattice image with its Bloch
    factor: 1.1e-07, the same as the interior control in the same run, which is
    the point — the interior slab was always right, so a comparison against it
    alone could not have found this.

    Registration is *not* what was wrong here: the engine and MEEP already agreed
    on the cells for this particular slab (7 cells of z either way), which is why
    this case could isolate the Yee average. Registration was wrong elsewhere, and
    separately — see
    `test_dft_region_registration_matches_cpu_meep_everywhere`, which now derives
    both monitors' cells from the same ladder.
    """
    driver = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    driver.add_source(_WRAP_SOURCE)
    touching = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                      center=(0.0, 0.0, 1.15), size=(0.4, 0.4, 0.5))
    interior = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                      center=(0.0, 0.0, 0.25), size=(0.4, 0.4, 0.5))
    driver.run(until=12.0)

    assert touching.region[5] == driver.grid.nz, "The slab must actually reach the last stored row."
    assert interior.region[5] < driver.grid.nz, "The control must not."
    for label, monitor, want in (("touching", touching, meep_reference["dft_touching"]),
                                 ("interior", interior, meep_reference["dft_interior"])):
        got = np.asarray(to_numpy(monitor.get_dft("Ex")))
        assert got.shape == want.shape, f"{label}: engine {got.shape} vs meep {want.shape}"
        error = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert error < 5e-6, f"{label} region: relative L2 {error:.4e} vs CPU MEEP's get_dft_array."
        # Plane by plane, so a large interior cannot average away a wrong face. The
        # denominator is the region's own largest plane, not each plane's norm: one
        # of these z planes is a node of the standing pattern (MEEP norm 6.0e-09),
        # and dividing it by itself measures round-off rather than placement.
        scale = max(float(np.linalg.norm(want[..., index])) for index in range(want.shape[2]))
        assert scale > 0.0, f"{label}: the CPU-MEEP region is empty; the comparison is vacuous."
        for index in range(got.shape[2]):
            plane = float(np.linalg.norm(got[..., index] - want[..., index]) / scale)
            assert plane < 5e-6, f"{label} region, z-plane {index}: error {plane:.4e} of the region."

    # An absorbing run takes the same wrap, because MEEP's absorbing faces are Periodic
    # too. Pinned here so the scoping cannot drift back into clipping them; the number
    # that says clipping is wrong is
    # `test_a_dft_region_reaching_past_an_absorbing_face_matches_cpu_meep`.
    absorbing = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    absorbing.setup_pml(4)
    assert absorbing.add_dft_monitor(
        frequencies=1.0, components=("Ex",), center=(0.0, 0.0, 1.15), size=(0.4, 0.4, 0.5)
    ).periodic == (True, True, True)


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("label, thickness", [("pmlz", {"z": 6}), ("pmlall", 4)])
def test_a_dft_region_reaching_past_an_absorbing_face_matches_cpu_meep(
        meep_reference, label, thickness):
    """A DFT region continued past an ABSORBING face must be the region MEEP builds.

    This is the measurement behind ``_periodic_axes``. The engine used to clip such a
    region at the face, on the reasoning that MEEP takes no lattice shift where the
    boundary is not ``Periodic`` — true, but a PML run's boundaries ARE Periodic
    (``fields::use_bloch``, called for every direction of any run with a ``k_point``;
    ``structure_chunk::use_pml`` only grades a conductivity underneath). Clipped, the
    engine returned SIX z planes where MEEP returns EIGHT for the same request, and the
    two extra planes are not empty — they carry the wrapped field from the far end of
    the cell.

    Measured with the region continued: 1.6e-07 (z-only layer) and 1.8e-07 (uniform
    layer), the same band as the interior control in the same run and as the no-PML
    version of the identical slab.
    """
    driver = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    driver.setup_pml(thickness)
    driver.add_source(_WRAP_SOURCE)
    past = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                  center=(0.0, 0.0, 1.3), size=(0.4, 0.4, 0.6))
    interior = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                      center=(0.0, 0.0, 0.25), size=(0.4, 0.4, 0.5))
    driver.run(until=12.0)

    assert past.region[5] > driver.grid.nz, "the slab must actually reach past the last stored row"
    assert interior.region[5] < driver.grid.nz, "the control must not"
    for name, monitor, want in (("past_face", past, meep_reference["dft_past_face_" + label]),
                                ("interior", interior, meep_reference["dft_past_interior_" + label])):
        got = np.asarray(to_numpy(monitor.get_dft("Ex")))
        assert got.shape == want.shape, (
            f"{name}: engine {got.shape} vs meep {want.shape} — a clipped region returns a "
            f"different volume from the one that was asked for")
        error = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert error < 5e-6, f"{name}: relative L2 {error:.4e} vs CPU MEEP's get_dft_array."

    # The two continued planes must carry something, or "matches MEEP" would be a
    # statement about two empty slices.
    tail = np.asarray(to_numpy(past.get_dft("Ex")))[..., -2:]
    whole = np.asarray(to_numpy(past.get_dft("Ex")))
    assert float(np.max(np.abs(tail))) > 0.01 * float(np.max(np.abs(whole))), (
        "the wrapped planes are empty; this case is not exercising the continuation")


@requires_meep
@skip_without_meep
def test_bloch_dft_region_touching_the_wrapped_face_carries_the_phase(meep_reference):
    """The plane a DFT region borrows across a Bloch face arrives multiplied by the phase.

    The k = 0 version of this test cannot see a dropped Bloch factor — the factor
    is 1 there — and the mutation run proved it: deleting the phase from the DFT
    gather survived the whole suite until this case existed. The wrap is on z and
    so is the wavevector, because `Ex` has Yee shift 0 in z and that is the average
    that straddles the face.

    Measured: the touching slab 2.7e-07 and the interior slab in the same run
    1.8e-07 against `get_dft_array`. The negative control is the same geometry at
    k = 0, which must be far outside the bound — otherwise the reference could not
    distinguish a run that applied the boundary phase from one that ignored it.
    """
    k_z = float(meep_reference["dft_bloch_kz"])
    driver = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True,
                        k_point=(0.0, 0.0, k_z))
    driver.add_source(_WRAP_SOURCE)
    touching = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                      center=(0.0, 0.0, 1.15), size=(0.4, 0.4, 0.5))
    interior = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                      center=(0.0, 0.0, 0.25), size=(0.4, 0.4, 0.5))
    driver.run(until=12.0)
    assert touching.region[5] == driver.grid.nz and interior.region[5] < driver.grid.nz

    measured = {}
    for label, monitor, want in (("touching", touching, meep_reference["dft_bloch_touching"]),
                                 ("interior", interior, meep_reference["dft_bloch_interior"])):
        got = np.asarray(to_numpy(monitor.get_dft("Ex")))
        assert got.shape == want.shape, f"{label}: engine {got.shape} vs meep {want.shape}"
        measured[label] = got
        error = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert error < 5e-6, f"{label} region under k_z={k_z}: relative L2 {error:.4e}."

    # The far plane is where the factor lands, so check it on its own rather than
    # letting six correct planes average it away.
    want_face = meep_reference["dft_bloch_touching"][..., -1]
    face = float(np.linalg.norm(measured["touching"][..., -1] - want_face) / np.linalg.norm(want_face))
    assert face < 5e-6, f"The wrapped far plane alone is {face:.4e} from CPU MEEP's."

    # Negative control: k = 0 must not reproduce the k_z reference.
    plain = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    plain.add_source(_WRAP_SOURCE)
    blind_monitor = plain.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                          center=(0.0, 0.0, 1.15), size=(0.4, 0.4, 0.5))
    plain.run(until=12.0)
    blind = np.asarray(to_numpy(blind_monitor.get_dft("Ex")))
    want = meep_reference["dft_bloch_touching"]
    assert float(np.linalg.norm(blind - want) / np.linalg.norm(want)) > 0.05, (
        "A k = 0 run reproduces the Bloch DFT reference, so this comparison cannot tell "
        "whether the boundary phase reached the monitor."
    )


# --- DFT-region registration sweep ------------------------------------------
# The engine side of the oracle block above. The case table itself is read back
# out of the oracle's own output, so there is one source of truth for the volumes
# and the two sides cannot drift.

_REG_BAR = 5e-6  # Same bound as the other DFT-field cross-validations in this file.


def _meep_axis_ladder(where_min, where_max, resolution, n_full):
    """Cells MEEP brackets a requested extent with, on one centred axis.

    An independent transcription of loop_in_chunks.cpp `vec2diel_floor` /
    `vec2diel_ceil` — `1 + 2*floor(x*a - 0.5)` and `1 + 2*ceil(x*a - 0.5)` in
    MEEP's doubled coordinates, whose cell centres are the odd integers — written
    here so the expected region is derived from the reference, not from the same
    `dft._axis_ladder` the engine registers with. A formula compared with itself
    proves nothing.

    Returns:
        (first_cell, cell_count) on the stored grid, unclamped; a request reaching
        past a face keeps the cells of the neighbouring lattice image.
    """
    low_doubled = 1 + 2 * math.floor(where_min * resolution - 0.5)
    high_doubled = 1 + 2 * math.ceil(where_max * resolution - 0.5)
    first_cell = (low_doubled + n_full - 1) // 2  # Cell j sits at -n_full + 2j + 1.
    return first_cell, (high_doubled - low_doubled) // 2 + 1


def _expected_region(cell, resolution, centre, size):  # Ladder region for one monitor volume.
    bounds = [(centre[axis] - size[axis] / 2, centre[axis] + size[axis] / 2) for axis in range(3)]
    region = []
    for axis in range(3):
        first, count = _meep_axis_ladder(
            *bounds[axis], resolution, int(round(cell[axis] * resolution))
        )
        region += [first, first + count]
    return tuple(region)


def _atlas_block(atlas, atlas_region, region):
    """The part of the whole-cell reference array a monitor region covers.

    The atlas is one MEEP DFT monitor spanning the cell and a full lattice image on
    each face, so any region this sweep can produce — including one continued past a
    boundary — is a contiguous block of it, addressed by the difference of the two
    ladder starts. Indexing it is therefore a statement about *which cells* the
    engine registered as well as about their values.
    """
    slices = []
    for axis in range(3):
        start = region[2 * axis] - atlas_region[2 * axis]
        stop = start + (region[2 * axis + 1] - region[2 * axis])
        assert 0 <= start < stop <= atlas.shape[axis], (
            f"axis {axis}: region {region} is outside the atlas {atlas_region}; widen the atlas."
        )
        slices.append(slice(start, stop))
    return atlas[tuple(slices)]


def _meep_collapsed(region_array, cell, resolution, centre, size):
    """Reduce an engine region the way MEEP's `get_dft_array` reduces its own.

    MEEP (dft.cpp `process_dft_component`, array_slice.cpp `collapse_array`) does
    two things this engine does not, both at read time: an axis whose *requested*
    extent is zero keeps the ladder's `w0`/`w1` interpolation weights and is then
    summed away, and an axis that came out one cell wide is dropped from the shape.
    Neither is registration — the engine hands back the region it accumulated — so
    they are applied here, to compare like with like. The weights are plain linear
    interpolation onto the requested coordinate, derived from the cell centres
    rather than from the engine's own weight helper.
    """
    dx = 1.0 / resolution
    reduced = np.asarray(region_array, dtype=np.complex128)
    for axis in (2, 1, 0):
        first = _expected_region(cell, resolution, centre, size)[2 * axis]
        count = reduced.shape[axis]
        if size[axis] == 0.0:
            origin = -cell[axis] / 2 + (first + 0.5) * dx  # Centre of the region's first cell.
            fraction = (centre[axis] - origin) / dx
            weights = np.array([1.0]) if count == 1 else np.array([1.0 - fraction, fraction])
            assert count <= 2, f"a zero-thickness axis resolved to {count} cells"
            reduced = np.tensordot(reduced, weights.astype(np.complex128), axes=([axis], [0]))
        elif count == 1:
            reduced = np.take(reduced, 0, axis=axis)
    return reduced


def _run_registration_sweep(reference, tag):  # One driver run; every case's monitor, by name.
    names = [str(name) for name in reference["reg_names"]]
    volumes = np.asarray(reference["reg_volumes"], dtype=float)
    k_point = tuple(float(value) for value in reference[f"reg_{tag}_k"])
    driver = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True,
                        k_point=k_point)
    driver.add_source(_WRAP_SOURCE)
    atlas_size = tuple(float(value) for value in reference["reg_atlas_size"])
    monitors = {
        "": driver.add_dft_monitor(frequencies=1.0, components=("Ex", "Ez"),
                                   center=(0.0, 0.0, 0.0), size=atlas_size)
    }
    for name, volume in zip(names, volumes):
        monitors[name] = driver.add_dft_monitor(
            frequencies=1.0, components=("Ex", "Ez"),
            center=tuple(volume[:3]), size=tuple(volume[3:]),
        )
    driver.run(until=12.0)
    return driver, names, volumes, monitors


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("tag", ["k0", "bloch"])
def test_dft_region_registration_matches_cpu_meep_everywhere(meep_reference, tag):
    """Every way a DFT region can meet a boundary, against CPU MEEP, cells and values.

    Two things were wrong with `DFTMonitor`'s registration and both were silent.

    It derived its own indices from a fixed half-cell margin and `round()` instead
    of MEEP's outward-rounding ladder, so a face landing exactly on a row of cell
    centres cost it a cell or not *depending on the parity of that row*: at
    resolution 10 a region of size 2.1 registered 20 cells where MEEP registers 22,
    while the same region at size 1.9 agreed exactly. And it clamped a region to the
    stored grid where MEEP continues it into the neighbouring lattice image, so a
    region reaching past a periodic face simply came back smaller — holding whatever
    cells happened to be in range, with nothing in the result to say so.

    Measured against `get_dft_array` in a 2x2x3 periodic cell at resolution 10,
    before and after — complex relative L2 on Ex, or the engine/MEEP shapes where
    the two could not be compared at all:

        case                 before                     after (k=0 / Bloch)
        interior (control)   1.136e-07                  1.136e-07 / 1.030e-07
        centre_face_even     (20,6,7) vs (22,6,7)       9.870e-08 / 1.125e-07
        centre_face_odd      9.822e-08                  9.822e-08 / 1.116e-07
        last_centre_z        (6,6,6) vs (6,6,5)         1.063e-07 / 1.265e-07
        past_hi_face_z       (6,6,5) vs (6,6,7)         1.118e-07 / 1.299e-07
        past_lo_face_z       (6,6,5) vs (6,6,7)         1.053e-07 / 1.275e-07
        on_hi_face_z         (6,6,3) vs (6,6,6)         1.074e-07 / 1.235e-07
        span_full_xy         (20,20,7) vs (22,22,6)     1.090e-07 / 1.183e-07
        whole_cell           (20,20,30) vs (22,22,32)   1.120e-07 / 1.189e-07
        wider_than_cell_x    (20,6,7) vs (32,6,6)       1.008e-07 / 1.124e-07
        flat_z_on_centre     3 cells for a flat axis    1.650e-07 / 1.222e-07
        flat_z_at_face       4.905e-01 (values)         4.321e-07 / 2.286e-07
        point_on_centre      3 cells for a flat axis    5.243e-07 / 0.000e+00

    `flat_z_at_face` is the one that returned a plausible number rather than a wrong
    shape: a zero-thickness monitor asked for at z = 1.5 — the cell face, a round
    coordinate — was clamped onto the last row of cell centres at z = 1.45 and
    reported that row's field as the face's, 49% wrong at k = 0 and 21% under a
    Bloch phase, in silence. `flat_z_on_centre` is the same defect class one step
    earlier: a "zero-thickness" region three cells thick.

    The bound below is 5e-6, the same as the other DFT-field cross-validations here;
    the worst case over the whole sweep is 5.2e-07 at k = 0 (point_on_centre/Ex,
    where the reference is 1/90 of the run's peak) and 1.3e-07 under the Bloch
    phase.

    The comparison is against a whole-cell atlas monitor block, so it fails both
    when the values are wrong and when the region covers the wrong cells, and the
    expected region comes from `_meep_axis_ladder`, an independent transcription of
    MEEP's own floor/ceil bracket.
    """
    reference = meep_reference
    driver, names, volumes, monitors = _run_registration_sweep(reference, tag)
    resolution = 10
    atlas_size = tuple(float(value) for value in reference["reg_atlas_size"])
    atlas_region = _expected_region(_WRAP_CELL, resolution, (0.0, 0.0, 0.0), atlas_size)
    assert monitors[""].region == atlas_region, (
        f"the atlas monitor itself is misregistered: {monitors[''].region} vs {atlas_region}"
    )

    worst = (0.0, "")
    for name, volume in zip(names, volumes):
        centre, size = tuple(volume[:3]), tuple(volume[3:])
        monitor = monitors[name]
        expected = _expected_region(_WRAP_CELL, resolution, centre, size)
        assert monitor.region == expected, (
            f"{name}: engine registered {monitor.region}, MEEP's ladder gives {expected}"
        )
        for component in ("Ex", "Ez"):
            atlas = np.asarray(reference[f"reg_{tag}_atlas_{component}"])
            got = np.asarray(to_numpy(monitor.get_dft(component)))
            want = _atlas_block(atlas, atlas_region, monitor.region)
            assert got.shape == want.shape
            scale = float(np.linalg.norm(want))
            assert scale > 0.0, f"{name}/{component}: the CPU-MEEP block is empty"
            error = float(np.linalg.norm(got - want) / scale)
            assert error < _REG_BAR, (
                f"{name}/{component} at {tag}: relative L2 {error:.3e} against CPU MEEP's "
                f"cells {monitor.region}"
            )
            worst = max(worst, (error, f"{name}/{component}"))

            # ...and against MEEP's own get_dft_array for that request, reduced the
            # way MEEP reduces it. A POINT request (zero thickness on all three axes)
            # is excluded: get_dft_array cannot serve one on any platform. Between
            # cell centres its `collapse_array` returns one corner scaled by 1/8
            # (measured 7.0e+00 against a manual interpolation of MEEP's own atlas);
            # on a single grid site it returns 0, because a rank-0 region has no
            # chunk in `process_dft_component` (MEEP 1.33.0 dft.cpp:1180-1183;
            # measured 0.0 at (0.25, 0.25, 0.25) on x86-64 and on arm64, against an
            # atlas cell of 0.27). A point exactly on a site only appeared to be
            # served on arm64, whose MEEP build fuses the multiply-add in its grid
            # rounding and so adds a zero-weight neighbour there that x86-64 does
            # not. Points are compared against the atlas above; every case with one
            # or two zero-thickness axes reduces correctly and is compared here too.
            flat_axes = [axis for axis in range(3) if size[axis] == 0.0]
            if len(flat_axes) == 3:
                continue
            direct = np.asarray(reference[f"reg_{tag}_{name}_{component}"])
            collapsed = _meep_collapsed(got, _WRAP_CELL, resolution, centre, size)
            assert collapsed.shape == direct.shape, (
                f"{name}/{component} at {tag}: engine reduces to {collapsed.shape}, MEEP's "
                f"get_dft_array gives {direct.shape}"
            )
            # Scaled by the region's own magnitude, not by the collapsed one: a flat
            # monitor on the x = y = 0 line averages Ex to a node (9.3e-09 out of
            # cells carrying 2.0e-01), and dividing that by itself measures the
            # float32 round-off of the cancellation rather than the placement.
            direct_error = float(np.linalg.norm(collapsed - direct) / scale)
            assert direct_error < _REG_BAR, (
                f"{name}/{component} at {tag}: {direct_error:.3e} against get_dft_array"
            )
    assert worst[0] > 0.0, "every case matched bit for bit; the comparison is probably vacuous"
    print(f"registration sweep worst case at {tag}: {worst[1]} {worst[0]:.2e}")


@requires_meep
@skip_without_meep
def test_dft_region_wrap_is_the_bloch_phase_and_not_a_plain_copy(meep_reference):
    """The cells a DFT region borrows across a face arrive multiplied by the boundary phase.

    The sweep above runs at k = 0 as well, where every wrap factor is 1 and a
    dropped phase is invisible. This is the control that makes the Bloch half of it
    mean something: the same wrapping regions run at k = 0 must be *far* outside the
    bound against the Bloch reference. Measured: `whole_cell` 1.2e+00 and
    `past_hi_face_z` 1.4e+00 at k = 0 against the Bloch oracle, versus 1.2e-07 and
    1.3e-07 for the run that actually carries the phase.
    """
    reference = meep_reference
    _, names, volumes, monitors = _run_registration_sweep(reference, "k0")
    atlas_region = _expected_region(
        _WRAP_CELL, 10, (0.0, 0.0, 0.0), tuple(float(v) for v in reference["reg_atlas_size"])
    )
    wrapping = ("whole_cell", "past_hi_face_z", "on_hi_face_z", "span_full_xy")
    for name in wrapping:
        monitor = monitors[name]
        assert any(monitor._wrapped), f"{name} does not actually wrap; it cannot be the control"
        blind = np.asarray(to_numpy(monitor.get_dft("Ex")))
        want = _atlas_block(np.asarray(reference["reg_bloch_atlas_Ex"]), atlas_region,
                            monitor.region)
        error = float(np.linalg.norm(blind - want) / np.linalg.norm(want))
        assert error > 0.05, (
            f"{name}: a k = 0 run reproduces the Bloch reference to {error:.2e}, so the Bloch "
            f"half of the sweep cannot tell whether the boundary phase reached the monitor."
        )


def test_dft_registration_refuses_what_it_cannot_place():
    """A region that resolves to nothing, or to a distant image, raises rather than reports zeros.

    The degenerate case this whole area needs: an accumulator with no cells reduces
    to zeros on every accessor — zero field, zero intensity — and nothing in that
    number says the monitor missed the cell. The positive controls are the same
    requests one step inside, which must still register.
    """
    driver = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    driver.add_source(_WRAP_SOURCE)
    inside = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                    center=(0.0, 0.0, 1.45), size=(0.4, 0.4, 0.0))
    far = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                 center=(0.0, 0.0, 9.0), size=(0.4, 0.4, 0.4))
    with pytest.raises(ValueError, match="does not intersect"):
        driver.run(until=0.5)
    assert inside.region[5] - inside.region[4] >= 1, "the control plane must still register"
    assert far is not None

    # On a non-periodic (absorbing) run the same request is clipped, and clipped to
    # nothing must raise too rather than allocate an empty accumulator.
    absorbing = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    absorbing.setup_pml(4)
    absorbing.add_source(_WRAP_SOURCE)
    absorbing.add_dft_monitor(frequencies=1.0, components=("Ex",),
                              center=(0.0, 0.0, 5.0), size=(0.4, 0.4, 0.4))
    with pytest.raises(ValueError, match="does not intersect"):
        absorbing.run(until=0.5)

    ok = FdtdDriver(cell_size=_WRAP_CELL, resolution=10, force_complex_fields=True)
    ok.setup_pml(4)
    ok.add_source(_WRAP_SOURCE)
    interior = ok.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                  center=(0.0, 0.0, 0.25), size=(0.4, 0.4, 0.4))
    ok.run(until=0.5)
    assert np.prod([interior.region[2 * axis + 1] - interior.region[2 * axis]
                    for axis in range(3)]) > 0


# --- Cross-feature: per-axis PML + Bloch + both monitor kinds, in one run ----------
#
# Per-axis PML, Bloch phases, and monitor registration were each built separately, and
# this is where the three collide: a Bloch-periodic grating absorbing in z only, with a
# source and flux planes spanning the full periodic cross section and DFT volumes that
# both wrap that cross section and reach into the absorber. Nothing about the geometry
# matches the cases any of those features was developed against — the modulation and the
# Bloch phase are on Y (not X), the period is 0.8 at resolution 20, the wavevector is
# negative, and the polarisation is Ex.
_INTEGRATION_GRATING = {
    "period": 0.8,            # y: the modulated axis AND the Bloch-phased one
    "cell_x": 0.3,            # the other periodic axis, unmodulated, k = 0
    "cell_z": 3.0,
    "resolution": 20,         # nx=6, ny=16, nz=60 — every count even, so the axis-origin
                              # parity is not what this case is measuring
    "ky": -0.35,
    "pml_cells": 12,          # z only; x and y stay (Bloch-)periodic
    "fcen": 1.0, "df": 0.5, "nfreq": 4,
    "source_fwidth": 0.9,
    "source_z": -0.75,
    "upstream_z": -0.85, "downstream_z": 0.70,
    "cross_z": 0.45, "cross_thickness": 0.3,      # DFT over the full periodic cross section
    "straddle_z": 0.85, "straddle_thickness": 0.4,  # DFT half inside the z absorber
    "eps_peak": 3.0, "eps_width": 0.22,
    "until": 20.0,
}

_INTEGRATION_ORACLE_SCRIPT = '''"""CPU-MEEP reference for the per-axis-PML / Bloch / monitor integration case."""
import json
import math
import sys

import numpy as np
import meep as mp

case = json.loads(sys.argv[1])
output_path = sys.argv[2]
results = {}

PERIOD, CELL_X, CELL_Z, RES = case["period"], case["cell_x"], case["cell_z"], case["resolution"]
FCEN, DF, NFREQ = case["fcen"], case["df"], case["nfreq"]


def epsilon_at(y, z):  # Smooth super-Gaussian slab, sinusoidally modulated across the period.
    envelope = math.exp(-((z / case["eps_width"]) ** 4))
    ridge = 0.5 + 0.5 * math.cos(2.0 * math.pi * y / PERIOD)
    return 1.0 + (case["eps_peak"] - 1.0) * envelope * ridge


plane = mp.Vector3(CELL_X, PERIOD, 0)
spec = (FCEN, DF, NFREQ)
for name, with_grating in (("vacuum", False), ("grating", True)):
    arguments = dict(
        cell_size=mp.Vector3(CELL_X, PERIOD, CELL_Z), resolution=RES,
        boundary_layers=[mp.PML(case["pml_cells"] / RES, direction=mp.Z)],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=case["source_fwidth"]),
                           component=mp.Ex, center=mp.Vector3(0, 0, case["source_z"]),
                           size=plane)],
        k_point=mp.Vector3(0, case["ky"], 0), force_complex_fields=True, eps_averaging=False)
    if with_grating:
        arguments["material_function"] = lambda p: mp.Medium(epsilon=epsilon_at(p.y, p.z))
    sim = mp.Simulation(**arguments)
    down = sim.add_flux(*spec, mp.FluxRegion(center=mp.Vector3(0, 0, case["downstream_z"]),
                                             size=plane, direction=mp.Z), decimation_factor=1)
    up = sim.add_flux(*spec, mp.FluxRegion(center=mp.Vector3(0, 0, case["upstream_z"]),
                                           size=plane, direction=mp.Z), decimation_factor=1)
    cross = sim.add_dft_fields([mp.Ex, mp.Hy], FCEN, 0, 1,
                               center=mp.Vector3(0, 0, case["cross_z"]),
                               size=mp.Vector3(CELL_X, PERIOD, case["cross_thickness"]),
                               decimation_factor=1)
    straddle = sim.add_dft_fields([mp.Ex], FCEN, 0, 1,
                                  center=mp.Vector3(0, 0, case["straddle_z"]),
                                  size=mp.Vector3(CELL_X, PERIOD, case["straddle_thickness"]),
                                  decimation_factor=1)
    sim.run(until=case["until"])
    results[name + "_downstream"] = np.asarray(mp.get_fluxes(down))
    results[name + "_upstream"] = np.asarray(mp.get_fluxes(up))
    results[name + "_cross_Ex"] = np.asarray(sim.get_dft_array(cross, mp.Ex, 0))
    results[name + "_cross_Hy"] = np.asarray(sim.get_dft_array(cross, mp.Hy, 0))
    results[name + "_straddle_Ex"] = np.asarray(sim.get_dft_array(straddle, mp.Ex, 0))
    results["freqs"] = np.asarray(mp.get_flux_freqs(down))

# The analytic grating sampled where the engine's inv_eps is registered — the integer
# Yee position measured from MEEP's own axis origin, -(n - n % 2) in doubled coordinates
# — and half a cell across it, which the comparison must be able to tell apart.
nx = int(CELL_X * RES + 0.5)
ny, nz = int(PERIOD * RES + 0.5), int(CELL_Z * RES + 0.5)
dx = 1.0 / RES
origin_y, origin_z = -(ny - ny % 2) * dx / 2, -(nz - nz % 2) * dx / 2
for label, offset in (("integer", 0.0), ("center", 0.5)):
    ys = origin_y + (np.arange(ny) + offset) * dx
    zs = origin_z + (np.arange(nz) + offset) * dx
    sampled = np.array([[epsilon_at(y, z) for z in zs] for y in ys])
    results["eps_" + label] = np.repeat(sampled[None, :, :], nx, axis=0)

np.savez(output_path, **results)
'''


@pytest.fixture(scope="module")
def integration_grating_oracle(tmp_path_factory):  # One CPU-MEEP process for the grating case.
    import json

    directory = tmp_path_factory.mktemp("integration_grating_oracle")
    script_path = directory / "grating_oracle.py"
    script_path.write_text(_INTEGRATION_ORACLE_SCRIPT, encoding="utf-8")
    output_path = directory / "grating.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(_INTEGRATION_GRATING), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP grating integration oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _integration_grating_run(case, epsilon=None, ky=None):  # One engine run of the case.
    """Build and run the grating, returning both flux spectra and both DFT monitors."""
    driver = FdtdDriver(
        cell_size=(case["cell_x"], case["period"], case["cell_z"]),
        resolution=case["resolution"], force_complex_fields=True,
        k_point=(0.0, case["ky"] if ky is None else ky, 0.0),
    )
    driver.setup_pml({"z": case["pml_cells"]})
    # The three features meeting: an absorber on z alone, a Bloch phase on the axes it
    # leaves alone, and monitors spanning the full periodic cross section. Every
    # unmirrored axis wraps for the monitors exactly as it does for the stepping.
    assert driver._monitor_periodic_axes() == (True, True, True)
    if epsilon is not None:
        driver.set_epsilon(epsilon)
    plane = (case["cell_x"], case["period"], 0.0)
    driver.add_source(dict(component="Ex", center=(0.0, 0.0, case["source_z"]), size=plane,
                           source_type="gaussian", frequency=case["fcen"],
                           fwidth=case["source_fwidth"]))
    downstream = driver.add_flux_monitor(
        fcen=case["fcen"], df=case["df"], nfreq=case["nfreq"],
        center=(0.0, 0.0, case["downstream_z"]), size=plane, direction=2)
    upstream = driver.add_flux_monitor(
        fcen=case["fcen"], df=case["df"], nfreq=case["nfreq"],
        center=(0.0, 0.0, case["upstream_z"]), size=plane, direction=2)
    cross = driver.add_dft_monitor(
        frequencies=case["fcen"], components=("Ex", "Hy"),
        center=(0.0, 0.0, case["cross_z"]),
        size=(case["cell_x"], case["period"], case["cross_thickness"]))
    straddle = driver.add_dft_monitor(
        frequencies=case["fcen"], components=("Ex",),
        center=(0.0, 0.0, case["straddle_z"]),
        size=(case["cell_x"], case["period"], case["straddle_thickness"]))
    driver.run(until=case["until"])
    return (np.asarray(downstream.get_flux_spectrum()), np.asarray(upstream.get_flux_spectrum()),
            cross, straddle)


def _worst_relative_spectrum(candidate, reference):  # Worst per-entry relative error.
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert np.all(np.abs(reference) > 0.0), "a zero reference entry would make the ratio meaningless"
    return float(np.max(np.abs(candidate - reference) / np.abs(reference)))


@requires_meep
@skip_without_meep
def test_bloch_grating_with_z_pml_and_both_monitor_kinds_matches_cpu_meep(integration_grating_oracle):
    """Per-axis PML, Bloch phases and monitor registration, measured together.

    One run carries all three: an absorber on z alone, a Bloch phase on the periodic y
    axis it leaves alone, a source and two flux planes spanning the full periodic cross
    section, a DFT volume spanning that same cross section (so it wraps in BOTH x and y
    and its y wrap carries the Bloch factor), and a second DFT volume half inside the z
    absorber. Each is compared against CPU MEEP for the same request.

    Measured (relative L2 on the complex DFT, worst per-frequency relative on flux):

        vacuum flux            3.8e-07 downstream, 3.3e-07 upstream
        grating flux           8.2e-07 downstream, 2.4e-07 upstream
        transmission T(f)      1.1e-06 per frequency
        cross-section DFT      2.5e-07 (Ex), 3.4e-07 (Hy)
        DFT inside the PML     2.5e-07

    The grating diffracts hard — T runs from 0.15 to 0.94 across the band — so the
    normalised spectrum is not a comparison of 1.0 with 1.0.
    """
    case = _INTEGRATION_GRATING
    reference = integration_grating_oracle
    bar = 5e-6

    vacuum_down, vacuum_up, vacuum_cross, vacuum_straddle = _integration_grating_run(case)
    grating_down, grating_up, grating_cross, grating_straddle = _integration_grating_run(
        case, epsilon=reference["eps_integer"])

    assert _worst_relative_spectrum(vacuum_down, reference["vacuum_downstream"]) < bar
    assert _worst_relative_spectrum(vacuum_up, reference["vacuum_upstream"]) < bar
    assert _worst_relative_spectrum(grating_down, reference["grating_downstream"]) < bar
    assert _worst_relative_spectrum(grating_up, reference["grating_upstream"]) < bar

    # The DFT volume spanning the full periodic cross section must WRAP on both
    # transverse axes; clipped there it would be a different measurement that still
    # returned a full array of plausible numbers.
    assert vacuum_cross._wrapped == (True, True, False)
    for label, monitor, straddle in (("vacuum", vacuum_cross, vacuum_straddle),
                                     ("grating", grating_cross, grating_straddle)):
        for component in ("Ex", "Hy"):
            error = _complex_relative_l2(monitor.get_dft(component),
                                         reference[f"{label}_cross_{component}"])
            assert error < bar, f"{label} cross-section {component}: {error:.3e} vs CPU MEEP"
        error = _complex_relative_l2(straddle.get_dft("Ex"), reference[f"{label}_straddle_Ex"])
        assert error < bar, f"{label} DFT straddling the z absorber: {error:.3e} vs CPU MEEP"

    meep_transmission = reference["grating_downstream"] / reference["vacuum_downstream"]
    transmission = grating_down / vacuum_down
    worst = _worst_relative_spectrum(transmission, meep_transmission)
    assert worst < bar, (
        f"Transmission spectrum deviates from CPU MEEP by {worst:.3e} per frequency "
        f"(engine {np.array2string(transmission, precision=5)}, "
        f"meep {np.array2string(meep_transmission, precision=5)})."
    )
    assert np.max(np.abs(meep_transmission - 1.0)) > 0.5, "this grating barely diffracts"

    # Control 1 — the Bloch phase is live. A run that accepted k_point and dropped it
    # once the PML existed would still return a complete, smooth spectrum.
    unphased_vacuum, _, _, _ = _integration_grating_run(case, ky=0.0)
    unphased_grating, _, _, _ = _integration_grating_run(case, epsilon=reference["eps_integer"],
                                                         ky=0.0)
    unphased = _worst_relative_spectrum(unphased_grating / unphased_vacuum, meep_transmission)
    assert unphased > 1000.0 * worst, (
        f"A run at k = 0 must not reproduce the k_y = {case['ky']} spectrum, but it came "
        f"within {unphased:.3e} against {worst:.3e}."
    )

    # Control 2 — the measurement resolves a half-cell material shift, so it is not
    # passing on slack.
    offset_down, _, _, _ = _integration_grating_run(case, epsilon=reference["eps_center"])
    offset = _worst_relative_spectrum(offset_down / vacuum_down, meep_transmission)
    assert offset > 1000.0 * worst, (
        f"The comparison cannot resolve a half-cell epsilon shift: correct sampling gives "
        f"{worst:.3e} and half-cell-offset sampling {offset:.3e}."
    )


# --- The two silent wrong answers the grating case above used to hide -------------
#
# Both are pre-existing registration defects that only a structured, cross-section-
# spanning run exposes, which is why they survived until the per-axis PML made that run
# legal. They are pinned here on their own, with an oracle that measures a FLUX — a
# scalar with no field-readback convention in it — so neither can be argued away as a
# comparison artifact.
_REGISTRATION_ORACLE_SCRIPT = '''"""CPU-MEEP references for the source-wrap and axis-origin registration defects."""
import sys

import numpy as np
import meep as mp

output_path = sys.argv[1]
results = {}

# --- A. A source spanning the full periodic cross section, at k = 0 and at k != 0.
#        MEEP continues the source volume into the neighbouring lattice image rather
#        than clipping it at the face (loop_in_chunks lattice-shift loop).
for label, kx in (("srcwrap_k0", 0.0), ("srcwrap_k04", 0.4)):
    sim = mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.0, 4.0), resolution=15,
        boundary_layers=[mp.PML(10 / 15, direction=mp.Z)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=1.0), component=mp.Ey,
                           center=mp.Vector3(0, 0, -1.0), size=mp.Vector3(1.0, 1.0, 0))],
        k_point=mp.Vector3(kx, 0, 0), force_complex_fields=True, eps_averaging=False)
    flux = sim.add_flux(1.0, 0.4, 5, mp.FluxRegion(center=mp.Vector3(0, 0, 1.0),
                                                   size=mp.Vector3(1.0, 1.0, 0),
                                                   direction=mp.Z), decimation_factor=1)
    sim.run(until=22.0)
    results[label] = np.asarray(mp.get_fluxes(flux))

# --- B. An ODD cell count on one axis, with an off-axis point source so the field is
#        structured along it, and the EVEN control that must stay exact either way.
for label, cell in (("origin_odd", (2.0, 1.5, 2.0)), ("origin_even", (2.0, 1.6, 2.0))):
    sim = mp.Simulation(
        cell_size=mp.Vector3(*cell), resolution=10,
        boundary_layers=[mp.PML(0.4, direction=mp.Z)],
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0.05, 0.15, -0.35))],
        k_point=mp.Vector3(0, 0, 0), force_complex_fields=True)
    small = sim.add_flux(1.0, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, 0.35),
                                                  size=mp.Vector3(0.6, 0.6, 0),
                                                  direction=mp.Z), decimation_factor=1)
    whole = sim.add_flux(1.0, 0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, 0.35),
                                                  size=mp.Vector3(cell[0], cell[1], 0),
                                                  direction=mp.Z), decimation_factor=1)
    sim.run(until=4.0)
    results[label + "_small"] = np.asarray(mp.get_fluxes(small))
    results[label + "_whole"] = np.asarray(mp.get_fluxes(whole))

np.savez(output_path, **results)
'''


@pytest.fixture(scope="module")
def registration_oracle(tmp_path_factory):  # One CPU-MEEP process for both registration cases.
    directory = tmp_path_factory.mktemp("registration_oracle")
    script_path = directory / "registration_oracle.py"
    script_path.write_text(_REGISTRATION_ORACLE_SCRIPT, encoding="utf-8")
    output_path = directory / "registration.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP registration oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("label, kx", [("srcwrap_k0", 0.0), ("srcwrap_k04", 0.4)])
def test_source_spanning_the_periodic_cross_section_matches_cpu_meep(registration_oracle, label, kx):
    """A sheet source filling the cell must carry its wrapped plane, with its Bloch factor.

    The standard grating source: an Ey sheet spanning the full periodic cross section of
    a 1 x 1 x 4 cell at resolution 15, absorbing in z only. MEEP continues the source
    volume into the neighbouring lattice image and scales the copy by
    ``conj(eikna ** ishift)``; clipping it at the face instead dropped a whole plane of
    the injected current and clamped the overhanging one back onto the last cell, where
    the fancy-index subtraction kept only one of the two.

    Measured on the transmitted flux at five frequencies, before -> after:
    1.29e-01 -> 2.3e-07 at k = 0, and 9.5e-03 -> 3.9e-07 at kx = 0.4. The k = 0 case is
    the honest size of the defect; at kx = 0.4 the wrapped plane's own phase happened to
    cancel most of what was dropped, which is why the flagship grating measurement sat
    at the percent level rather than the percent-times-ten level and read as converged.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 4.0), resolution=15, force_complex_fields=True,
                        k_point=(kx, 0.0, 0.0))
    driver.setup_pml({"z": 10})
    driver.add_source(dict(component="Ey", center=(0.0, 0.0, -1.0), size=(1.0, 1.0, 0.0),
                           source_type="gaussian", frequency=1.0, fwidth=1.0))
    flux = driver.add_flux_monitor(fcen=1.0, df=0.4, nfreq=5, center=(0.0, 0.0, 1.0),
                                   size=(1.0, 1.0, 0.0), direction=2)
    driver.run(until=22.0)
    spectrum = np.asarray(flux.get_flux_spectrum())
    assert np.min(np.abs(spectrum)) > 0.0, "an all-zero spectrum must not score as agreement"
    worst = _worst_relative_spectrum(spectrum, registration_oracle[label])
    assert worst < 5e-6, f"{label}: transmitted flux {worst:.3e} from CPU MEEP"


def test_a_source_filling_the_cell_injects_the_whole_cross_section_exactly():
    """The oracle-free half: the folded source deposits the full current, once per cell.

    A source spanning the full cell integrates to exactly one unit of current per
    transverse cell — the wrapped plane completes the tapered one it wraps onto — and
    every cell appears exactly once, which is the uniqueness invariant
    ``_index_arrays``'s fancy-index subtraction depends on. Both parities of the cell
    count and all three electric components are checked, because the loss was a
    different size for each: the previous behaviour ran from 0.93 of the correct total
    (Ey on a 15-cell axis) down to 0.56 (Ez on a 6-cell one).
    """
    from .grid import Grid
    from .sources import _build_source_points

    for component in ("Ex", "Ey", "Ez"):
        for cells in (6, 7, 15, 16):
            length = cells / 10
            grid = Grid(cell_size=(length, length, 3.0), resolution=10)
            indices, amplitudes = _build_source_points(
                grid, component, (0.0, 0.0, -0.75), (length, length, 0.0), 1.0 + 0j, None)
            assert len(set(indices)) == len(indices), (
                f"{component} on {cells} cells: duplicate source indices, which "
                f"D[ix, iy, iz] -= amps silently drops all but one of")
            total = sum(abs(value) for value in amplitudes)
            expected = grid.nx * grid.ny * grid.resolution  # one unit per cell, times the
            assert abs(total / expected - 1.0) < 1e-6, (   # 1/dx of the flat z axis
                f"{component} on {cells} cells injects {total / expected:.5f} of the "
                f"cross section's current")

    # Degenerate control: a source that does NOT reach a face must be untouched by the
    # fold, so a change that always wrapped could not pass this pair.
    grid = Grid(cell_size=(1.6, 1.6, 3.0), resolution=10)
    indices, amplitudes = _build_source_points(
        grid, "Ey", (0.0, 0.0, -0.75), (0.4, 0.4, 0.0), 1.0 + 0j, None)
    assert len(set(indices)) == len(indices)
    assert all(0 <= index[axis] < (grid.nx, grid.ny, grid.nz)[axis]
               for index in indices for axis in range(3))
    assert 0.0 < sum(abs(value) for value in amplitudes) < grid.nx * grid.ny * grid.resolution


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("label, cell", [("origin_odd", (2.0, 1.5, 2.0)),
                                         ("origin_even", (2.0, 1.6, 2.0))])
def test_axis_origin_parity_matches_cpu_meep_on_odd_and_even_cell_counts(registration_oracle,
                                                                         label, cell):
    """An axis with an ODD cell count starts half a cell above -L/2, as MEEP's does.

    ``Simulation._create_grid_volume`` calls ``grid_volume::center_origin()``, which
    shifts by ``-icenter()`` where ``icenter = io + round_down_to_even(n)``. The engine
    took ``-n`` on every axis, which is right only for an even count; on an odd one it
    placed the whole axis half a cell from MEEP's.

    Nothing in the output says so. This case is a plain periodic vacuum run with a point
    source — no Bloch phase, no per-axis PML, no wrapping monitor — and it reports a
    complete, smooth field and a plausible power. Measured on the transmitted flux,
    before -> after: 4.8e-02 -> 2.8e-08 through a small plane and 3.4e-02 -> 1.4e-07
    through the whole cross section, on a 1.5-length axis (15 cells) at resolution 10.
    The 1.6-length control (16 cells) is exact both ways at 7.8e-08, which is what makes
    this a parity defect rather than a grid-wide one — and what let it survive a suite
    whose cases are all even.
    """
    counts = tuple(int(round(extent * 10)) for extent in cell)
    assert counts == (20, 15, 20) if label == "origin_odd" else counts == (20, 16, 20)

    driver = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True)
    driver.setup_pml({"z": 4})
    driver.add_source(dict(component="Ez", center=(0.05, 0.15, -0.35), size=(0.0, 0.0, 0.0),
                           source_type="continuous", frequency=1.0))
    small = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 0.35),
                                    size=(0.6, 0.6, 0.0), direction=2)
    whole = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 0.35),
                                    size=(cell[0], cell[1], 0.0), direction=2)
    driver.run(until=4.0)
    for monitor, key in ((small, label + "_small"), (whole, label + "_whole")):
        measured = float(monitor.get_flux(0))
        assert measured != 0.0, "a zero flux must not score as agreement"
        expected = float(registration_oracle[key][0])
        error = abs(measured - expected) / abs(expected)
        assert error < 5e-6, f"{key}: flux {measured:.8g} vs CPU MEEP {expected:.8g} ({error:.3e})"


def test_the_axis_origin_is_meeps_round_down_to_even_and_only_moves_odd_axes():
    """Pin the origin rule itself, so the parity cannot drift back under a passing suite.

    An even axis keeps ``-n`` — every pinned number in this package was measured on one
    — and an odd axis takes ``-(n - 1)``, half a cell higher. A mirror-folded axis is
    -2 whatever its count, from MEEP's ``halve()``.
    """
    from .grid import Grid

    even = Grid(cell_size=(2.0, 1.6, 2.0), resolution=10)
    assert (even.origin_doubled(0), even.origin_doubled(1), even.origin_doubled(2)) == (-20, -16, -20)
    assert even.axis_origin(0) == pytest.approx(-1.0)  # -L/2 exactly, as before
    assert float(to_numpy(even.y)[0]) == pytest.approx(-0.8 + 0.05)

    odd = Grid(cell_size=(2.0, 1.5, 2.0), resolution=10)
    assert odd.origin_doubled(1) == -14, "an odd axis rounds the cell count down to even"
    assert odd.axis_origin(1) == pytest.approx(-0.7)  # -L/2 + dx/2, half a cell up
    assert float(to_numpy(odd.y)[0]) == pytest.approx(-0.65)
    assert float(to_numpy(odd.y)[-1]) == pytest.approx(0.75)

    folded = Grid(cell_size=(2.0, 1.5, 2.0), resolution=10, symmetry=("X",))
    assert folded.origin_doubled(0) == -2, "a mirror-folded axis keeps halve()'s origin"
    assert folded.origin_doubled(1) == -14

    with pytest.raises(ValueError, match="axis must be"):
        even.origin_doubled(3)


def test_a_dft_monitor_resolves_against_the_boundaries_in_force_when_stepping_starts():
    """Adding the monitor before or after setup_pml must give the same region.

    ``add_dft_monitor`` stamps the run's wrap flags when the monitor is ADDED, but the
    region resolves when stepping STARTS — the whole point of deferring it. The two
    orders must agree, whatever the rule says, and the region they agree on is the one
    MEEP builds for the same request: a 2 x 2 x 3 cell at resolution 10 with ``{"z": 6}``
    and a monitor at z = 1.3 of thickness 0.6 reaches z cells 24..32, two of them
    continued into the neighbouring lattice image, and that array matches
    ``get_dft_array`` to 1.6e-07 (``test_a_dft_region_reaching_past_an_absorbing_face_
    matches_cpu_meep``). It read 24..30 while the absorber was allowed to clip the wrap,
    which returned six z planes where MEEP returns eight.
    """
    def build(pml_first):
        driver = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, force_complex_fields=True)
        if pml_first:
            driver.setup_pml({"z": 6})
            monitor = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                             center=(0.0, 0.0, 1.3), size=(0.4, 0.4, 0.6))
        else:
            monitor = driver.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                             center=(0.0, 0.0, 1.3), size=(0.4, 0.4, 0.6))
            driver.setup_pml({"z": 6})
        driver.add_source(dict(component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                               source_type="gaussian", frequency=1.0, fwidth=0.5))
        driver.run(until=1.0)
        return monitor

    pml_first, monitor_first = build(True), build(False)
    assert pml_first.region == monitor_first.region == (7, 13, 7, 13, 24, 32)
    assert pml_first._wrapped == monitor_first._wrapped == (False, False, True)
    assert pml_first.periodic == monitor_first.periodic == (True, True, True)
    assert np.allclose(to_numpy(pml_first.get_dft("Ex")), to_numpy(monitor_first.get_dft("Ex")))
    assert float(np.max(np.abs(to_numpy(pml_first.get_dft("Ex"))))) > 0.0

    # Control: the same request with no absorber at all must give the same region and the
    # same wrap, which is the statement — the layer is not part of this answer.
    wrapping = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, force_complex_fields=True)
    monitor = wrapping.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                       center=(0.0, 0.0, 1.3), size=(0.4, 0.4, 0.6))
    wrapping.add_source(dict(component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                             source_type="gaussian", frequency=1.0, fwidth=0.5))
    wrapping.run(until=1.0)
    assert monitor._wrapped == (False, False, True)
    assert monitor.region == (7, 13, 7, 13, 24, 32)

    # And a folded axis still does not wrap, so the rule is the symmetry and not "always".
    folded = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, symmetry=("X",),
                        force_complex_fields=True)
    folded.setup_pml({"z": 6})
    folded_monitor = folded.add_dft_monitor(frequencies=1.0, components=("Ex",),
                                            center=(0.0, 0.0, 1.3), size=(0.4, 0.4, 0.6))
    folded.add_source(dict(component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                           source_type="gaussian", frequency=1.0, fwidth=0.5))
    folded.run(until=1.0)
    assert folded_monitor.periodic == (False, True, True)


def test_the_per_face_pml_interior_default_trims_each_face_by_its_own_thickness():
    """A per-face layer's default monitor region goes through the public per-face entry.

    ``set_region_from_pml`` takes ONE thickness and would trim a periodic x axis by the
    z layer's, discarding live interior — on a cell a grating period wide it would leave
    no region at all. ``set_region_from_faces`` trims each face by its own, so an axis
    carrying no absorber keeps every cell.
    """
    driver = FdtdDriver(cell_size=(0.6, 0.6, 3.0), resolution=10, force_complex_fields=True)
    driver.setup_pml({"z": (8, 3)})
    monitor = driver.add_dft_monitor(frequencies=1.0, components=("Ex",))
    driver.add_source(dict(component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                           source_type="gaussian", frequency=1.0, fwidth=0.5))
    driver.run(until=0.5)
    # x and y whole (no absorber), z trimmed 8 at the low face and 3 at the high one.
    assert monitor.region == (0, 6, 0, 6, 8, 27)

    # The scalar layer still goes through the scalar entry point, unchanged.
    scalar = FdtdDriver(cell_size=(2.0, 2.0, 3.0), resolution=10, force_complex_fields=True)
    scalar.setup_pml(4)
    scalar_monitor = scalar.add_dft_monitor(frequencies=1.0, components=("Ex",))
    scalar.add_source(dict(component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                           source_type="gaussian", frequency=1.0, fwidth=0.5))
    scalar.run(until=0.5)
    assert scalar_monitor.region == (4, 16, 4, 16, 4, 26)


def test_set_region_from_faces_refuses_a_mirror_plane_and_an_empty_interior():
    """The per-face entry point refuses what would return a plausible wrong array."""
    from .dft import DFTMonitor
    from .grid import Grid

    folded = Grid(cell_size=(2.0, 2.0, 3.0), resolution=10, symmetry=("X",))
    monitor = DFTMonitor(folded, frequencies=1.0, components=("Ex",))
    with pytest.raises(ValueError, match="mirror-folded"):
        monitor.set_region_from_faces(((4, 4), (0, 0), (0, 0)))
    # Naming only the face a mirrored axis CAN carry is fine, and the region keeps cell 0.
    monitor.set_region_from_faces(((0, 4), (0, 0), (6, 6)))
    assert monitor.region == (0, folded.nx - 4, 0, folded.ny, 6, folded.nz - 6)

    plain = Grid(cell_size=(2.0, 2.0, 3.0), resolution=10)
    monitor = DFTMonitor(plain, frequencies=1.0, components=("Ex",))
    with pytest.raises(ValueError, match="does not intersect"):
        monitor.set_region_from_faces(((10, 10), (0, 0), (0, 0)))
    with pytest.raises(ValueError, match="non-negative"):
        monitor.set_region_from_faces(((-1, 0), (0, 0), (0, 0)))
    with pytest.raises(ValueError, match="per-face PML table"):
        monitor.set_region_from_faces(((0, 0), (0, 0)))


# ---------------------------------------------------------------------------
# Real-valued (float32) field mode: complex-mode integrity, equivalence, refusals
# ---------------------------------------------------------------------------
#
# Real storage is MEEP's default and this engine's newest mode, and it was added by
# threading a dtype through code every existing complex-mode floor was measured on.
# Three separate things therefore need saying, and each gets its own test below:
# complex runs did not move; a real run is the same physics; and the two ingredients a
# real run cannot carry are refused rather than quietly reinterpreted.
#
# The CPU-MEEP anchors live next door — `test_driver_vs_meep.py` holds the real-mode
# field parity (plain / PML / dispersive / folded / odd-cell) and `test_dft.py` holds
# the real-mode spectrum. Nothing here needs an oracle.

_PRIMARY_COMPONENTS = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")

_REAL_MODE_CW = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -0.35),
                 "size": (0.0, 0.0, 0.0)}


def _compact_blob(shape, dx):  # A few-cell Gaussian D blob, for the source-free configurations.
    def profile(count):
        positions = (np.arange(count) - count / 2) * dx
        return np.where(np.abs(positions) <= 5.0 * dx, np.exp(-((positions / (2.0 * dx)) ** 2)), 0.0)
    nx, ny, nz = shape
    return (profile(nx)[:, None, None] * profile(ny)[None, :, None] * profile(nz)[None, None, :])


def _storage_mode_case(name, complex_fields):
    """One configuration in either storage mode, as (driver, num_steps).

    The set spans every path the real-fields work touched — plain stepping, the PML
    recurrence, the Bloch wrap, the mirror fold, the polarization ADE, the D
    conductivity — at BOTH parities of cell count, because this engine has twice
    shipped a registration error that every even-count grid hid. `bloch` is
    complex-only by construction and is skipped by the real leg of the callers.
    """
    if name == "plain_even":
        driver = FdtdDriver(cell_size=(1.2, 1.2, 1.6), resolution=10,
                            force_complex_fields=complex_fields)
        driver.add_source(dict(_REAL_MODE_CW))
        return driver, 40
    if name == "plain_odd":
        driver = FdtdDriver(cell_size=(1.1, 1.3, 1.7), resolution=10,
                            force_complex_fields=complex_fields)
        driver.add_source({**_REAL_MODE_CW, "center": (0.15, -0.25, -0.35)})
        return driver, 40
    if name == "pml_even":
        driver = FdtdDriver(cell_size=(2.0, 2.0, 2.4), resolution=10,
                            force_complex_fields=complex_fields)
        driver.setup_pml(5)
        driver.add_source({**_REAL_MODE_CW, "size": (0.4, 0.4, 0.0)})
        return driver, 30
    if name == "pml_odd":
        driver = FdtdDriver(cell_size=(2.1, 2.3, 2.5), resolution=10,
                            force_complex_fields=complex_fields)
        driver.setup_pml({"x": 5, "z": 4})
        driver.add_source({**_REAL_MODE_CW, "component": "Ey", "source_type": "gaussian",
                           "fwidth": 0.6, "center": (0.25, -0.35, -0.35)})
        return driver, 30
    if name == "bloch":
        driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10,
                            force_complex_fields=complex_fields, k_point=(0.3, -0.2, 0.15))
        driver.add_source({**_REAL_MODE_CW, "center": (0.0, 0.0, 0.05)})
        return driver, 30
    if name == "symmetry":
        driver = FdtdDriver(cell_size=(4.0, 2.0, 2.0), resolution=10,
                            force_complex_fields=complex_fields, symmetry=("X",))
        blob = _compact_blob((40, 20, 20), driver.dx)
        # Stored layout of a periodic even fold: rows 19.. plus the wrapped row 0
        # (the second-mirror plane cell; see _quadrant_of).
        driver.set_field("Dz", np.concatenate((blob[19:], blob[0:1]), axis=0))
        return driver, 12
    if name == "dispersive":
        driver = FdtdDriver(cell_size=(1.2, 1.2, 1.6), resolution=10, courant=0.4,
                            force_complex_fields=complex_fields)
        driver.set_epsilon(np.full(driver.shape, 1.5, dtype=np.float32))
        driver.add_susceptibility(Susceptibility(1.0, 0.05, DRUDE), 0.8)
        driver.add_susceptibility(Susceptibility(1.4, 0.1, LORENTZIAN), 0.5)
        driver.add_source({**_REAL_MODE_CW, "source_type": "gaussian", "fwidth": 0.6})
        return driver, 30
    if name == "dispersive_pml_odd":
        driver = FdtdDriver(cell_size=(2.1, 2.1, 2.5), resolution=10,
                            force_complex_fields=complex_fields)
        driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
        driver.add_susceptibility(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)
        driver.setup_pml(5)
        driver.add_source({**_REAL_MODE_CW, "size": (0.4, 0.4, 0.0)})
        return driver, 30
    if name == "conductivity":
        driver = FdtdDriver(cell_size=(1.2, 1.2, 1.6), resolution=10,
                            force_complex_fields=complex_fields)
        driver.set_epsilon(np.full(driver.shape, 3.4, dtype=np.float32))
        driver.set_conductivity(2 * math.pi * 0.42 * 0.101 / 3.4)
        driver.add_source({**_REAL_MODE_CW, "frequency": 0.42})
        return driver, 40
    if name == "magnetic":
        driver = FdtdDriver(cell_size=(1.1, 1.3, 1.7), resolution=10,
                            force_complex_fields=complex_fields)
        driver.add_source({**_REAL_MODE_CW, "component": "Hx", "center": (0.15, -0.25, -0.35)})
        return driver, 25
    if name == "integrated":
        driver = FdtdDriver(cell_size=(1.1, 1.3, 1.7), resolution=10,
                            force_complex_fields=complex_fields)
        driver.add_source({**_REAL_MODE_CW, "center": (0.15, -0.25, -0.35), "is_integrated": True})
        return driver, 25
    if name == "custom_envelope":
        driver = FdtdDriver(cell_size=(1.1, 1.3, 1.7), resolution=10,
                            force_complex_fields=complex_fields)
        driver.add_source({"component": "Ez", "source_type": "custom", "size": (0.0, 0.0, 0.0),
                           "center": (0.15, -0.25, -0.35),
                           "src_func": lambda t: np.exp(-1j * 2 * np.pi * t)})
        return driver, 25
    if name == "real_amp_func":
        driver = FdtdDriver(cell_size=(1.1, 1.3, 1.7), resolution=10,
                            force_complex_fields=complex_fields)
        driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                           "fwidth": 0.6, "center": (0.15, -0.25, -0.35), "size": (0.4, 0.4, 0.0),
                           "amp_func": lambda x, y, z: 1.0 + 0.5 * x})
        return driver, 25
    if name == "seeded_only":
        driver = FdtdDriver(cell_size=(1.3, 1.1, 1.7), resolution=10,
                            force_complex_fields=complex_fields)
        blob = _compact_blob(driver.shape, driver.dx)
        driver.set_field("Dz", blob)
        driver.set_field("Bx", 0.5 * blob)
        return driver, 25
    raise AssertionError(f"unknown storage-mode configuration {name!r}")


_STORAGE_MODE_CASES = (
    "plain_even", "plain_odd", "pml_even", "pml_odd", "bloch", "symmetry", "dispersive",
    "dispersive_pml_odd", "conductivity", "magnetic", "integrated", "custom_envelope",
    "real_amp_func", "seeded_only",
)

_REAL_CAPABLE_CASES = tuple(name for name in _STORAGE_MODE_CASES if name != "bloch")

# Seeded from a real array with no source at all: these carry no imaginary plane even in
# complex storage, which changes what a real-vs-complex comparison can claim about them.
_SOURCE_FREE_CASES = ("symmetry", "seeded_only")


def _primary_field_bytes(driver):  # The six primary arrays, concatenated raw.
    blob = b"".join(
        driver.get_field(name, cell_centered=False).tobytes() for name in _PRIMARY_COMPONENTS
    )
    peak = max(
        float(np.abs(driver.get_field(name, cell_centered=False)).max())
        for name in _PRIMARY_COMPONENTS
    )
    assert peak > 0.0, "the configuration produced no field at all; a byte comparison would be vacuous"
    return blob


def _run_storage_case(name, complex_fields):  # Run one configuration and return its raw primary bytes.
    driver, num_steps = _storage_mode_case(name, complex_fields)
    driver.run(num_steps=num_steps)
    blob = _primary_field_bytes(driver)
    dtype = driver.get_field("Dz", cell_centered=False).dtype
    driver.close()
    return blob, dtype


def _install_pre_real_complex_paths(monkeypatch):
    """Restore the complex-only code the real-fields work replaced, verbatim.

    Three decision points were introduced, and each is reverted here to the expression
    it replaced: :meth:`Fields._field_dtype` returned ``complex64`` unconditionally,
    ``_step_source_values`` did not exist and ``_inject_points`` subtracted the whole
    complex product, and ``ContinuousSource.inject`` subtracted ``complex64(dt*w*J)``
    with no dtype resolution and no ``.real``.

    A complex-mode run that is not byte-identical with these installed has had its
    arithmetic restructured by the real-fields work, which puts every recorded
    complex-mode floor in doubt — the floors are what the parity suite next door
    measures, and they were all taken before this mode existed. Reverting the code
    rather than pinning a digest keeps the comparison portable: it is two runs in one
    process, not a number recorded on one machine.
    """
    monkeypatch.setattr(fields_module.Fields, "_field_dtype",
                        lambda self: self.grid.xp.complex64, raising=True)
    monkeypatch.setattr(
        sources_module, "_step_source_values",
        lambda array, amps, scale, component: amps * np.complex64(scale),
        raising=True,
    )

    def pre_real_inject(self, fields, time):  # sources.py ContinuousSource.inject, before real mode.
        dt = self.grid.dt
        current = self.current(time, dt)
        array = sources_module._d_array_for(fields, self.component)
        for (ix, iy, iz), weight in zip(self._point_indices, self._point_weights):
            array[ix, iy, iz] -= np.complex64(dt * weight * current)

    monkeypatch.setattr(sources_module.ContinuousSource, "inject", pre_real_inject, raising=True)


@pytest.mark.parametrize("case", _STORAGE_MODE_CASES)
def test_complex_mode_bytes_are_untouched_by_the_real_field_code_path(monkeypatch, case):
    """Every complex-mode floor in this package was measured before real mode existed.

    So the first thing real mode has to prove is a negative: that a complex run's raw
    D and B bytes are exactly what the pre-real-mode code produced. Fourteen
    configurations, covering plain stepping, the PML recurrence, the Bloch wrap, the
    mirror fold, a two-term polarization ADE, the D conductivity, all four source
    families and a source-free seeded run — at both parities of cell count.

    Byte equality rather than a tolerance is the whole point: a restructured expression
    that moves the last mantissa bit is invisible at any bound anyone would write down,
    and it silently invalidates a 2.41e-07 floor recorded to three digits.
    """
    reference, reference_dtype = _run_storage_case(case, complex_fields=True)
    assert reference_dtype == np.complex64, "the complex leg did not allocate complex storage"
    _install_pre_real_complex_paths(monkeypatch)
    candidate, candidate_dtype = _run_storage_case(case, complex_fields=True)
    assert candidate_dtype == np.complex64
    assert candidate == reference, (
        f"{case}: complex-mode D/B bytes differ from the pre-real-mode code path, so the "
        f"real-fields work moved a complex run and every recorded complex floor is suspect"
    )


def test_the_pre_real_mode_comparison_can_actually_fail():
    """The positive control for the test above: prove the patch points are live.

    A reversion test whose patches are never consulted passes unconditionally. Nudging
    the injected current by one part in a million — through the two injection seams
    :func:`_install_pre_real_complex_paths` replaces — must change the bytes, once on
    the vectorized ``_step_source_values`` path (``real_amp_func``) and once on the
    ``ContinuousSource`` scalar loop (``plain_even``). The third seam,
    ``Fields._field_dtype``, is inert on a complex run by construction — returning
    complex64 unconditionally is what it already does there — and that inertness is
    exactly the property the reversion test exists to confirm, so it needs no control.
    """
    for case in ("plain_even", "real_amp_func"):
        reference, _ = _run_storage_case(case, complex_fields=True)
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(
                sources_module, "_step_source_values",
                lambda array, amps, scale, component:
                    amps * np.complex64(scale * 1.000001),
                raising=True,
            )

            def nudged_inject(self, fields, time):  # The scalar path, off by the same factor.
                dt = self.grid.dt
                current = self.current(time, dt) * 1.000001
                array = sources_module._d_array_for(fields, self.component)
                for (ix, iy, iz), weight in zip(self._point_indices, self._point_weights):
                    array[ix, iy, iz] -= np.complex64(dt * weight * current)

            patcher.setattr(sources_module.ContinuousSource, "inject", nudged_inject, raising=True)
            nudged, _ = _run_storage_case(case, complex_fields=True)
        assert nudged != reference, (
            f"{case}: a 1e-06 change to the injected current left the bytes identical, so the "
            f"reversion test's patch points are not on the path it claims to check"
        )


@pytest.mark.parametrize("case", _REAL_CAPABLE_CASES)
def test_real_mode_is_bit_for_bit_the_real_part_of_the_complex_run(case):
    """The equivalence claim, on the raw arrays rather than on a readback.

    ``test_driver_vs_meep.py`` makes this comparison on the cell-centred Ez of six
    CPU-MEEP cases; this one makes it on all six primary arrays of thirteen
    configurations, including the three source families the oracle cases do not use (a
    magnetic current, an ``is_integrated`` current, a caller-supplied waveform) and a
    real ``amp_func``. Exact equality is the correct bar because every coefficient in
    the loop is real and injection takes ``real(A)``: the real run performs the same
    float32 operations on the same values, so anything but identity means a coefficient
    or an intermediate went complex somewhere.
    """
    real_bytes, real_dtype = _run_storage_case(case, complex_fields=False)
    complex_bytes, complex_dtype = _run_storage_case(case, complex_fields=True)
    assert real_dtype == np.float32, f"{case}: real mode allocated {real_dtype}, not float32"
    assert complex_dtype == np.complex64
    # The six arrays are the same shape and dtype, so the concatenation is one valid
    # complex64 buffer and its real plane is the concatenation of the six real planes.
    complex_values = np.frombuffer(complex_bytes, dtype=np.complex64)
    expected = np.ascontiguousarray(complex_values.real).tobytes()
    assert real_bytes == expected, (
        f"{case}: real-mode D/B bytes are not the complex run's real part; the mode is either "
        f"dropping something the complex run keeps or carrying something it does not"
    )
    weight = float(np.linalg.norm(complex_values.imag) / np.linalg.norm(complex_values.real))
    if case in _SOURCE_FREE_CASES:
        # A run seeded from a real array and driven by nothing has no way to acquire an
        # imaginary plane, so the statement here is the complementary one: the complex
        # run must not have INVENTED phase from real inputs. A nonzero imaginary plane
        # would mean a coefficient somewhere is complex, which is what would make the
        # real mode an approximation rather than the same arithmetic.
        assert weight == 0.0, (
            f"{case}: a real-seeded, source-free complex run acquired an imaginary plane "
            f"({weight:.3e} of its real plane); some coefficient in the loop is complex"
        )
    else:
        # Not vacuous: the imaginary plane the real run discards has to be worth discarding,
        # or every injection convention would coincide and this equality would prove nothing.
        assert weight > 0.1, (
            f"{case}: the complex run's imaginary plane is only {weight:.3e} of its real plane, "
            f"so this equality does not discriminate one injection convention from another"
        )


def test_real_mode_takes_meeps_real_of_the_current_and_no_other_convention():
    """MEEP step.cpp:298 is ``f[c][0][i] -= real(A)`` with ``A = amp * current * dt``.

    Held against the arithmetic rather than against another run. After one step from
    rest there is no curl to add — H is still zero — so D is exactly what the source
    deposited, and the deposit can be written out longhand from the source object's own
    ``current()``. Every other convention a port could plausibly land on is then
    measured: ``imag(A)`` is the imaginary plane MEEP drops, ``abs(A)`` loses the sign
    and the phase, ``2*real(A)`` is the factor a reader who remembered
    ``2*Re(z) = z + conj(z)`` would introduce, and ``real(A)`` at the wrong half-step is
    the pure phase error the electric slot exists to prevent.

    The frequency is 2.5, not 1.0, for one reason: it puts the first electric slot at 45
    degrees of the carrier, where ``real``, ``imag``, ``abs`` and ``2*real`` are all far
    apart. At f = 1 the current is 95 % real at that instant and ``abs(A)`` lands within
    5 % of ``real(A)``, so the test would pass on the wrong convention. Four cells per
    wavelength is badly under-resolved for propagation and irrelevant here: nothing
    propagates in one step of pure injection.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=False)
    source = driver.add_source({"component": "Ez", "frequency": 2.5, "center": (0.0, 0.0, 0.0),
                               "size": (0.0, 0.0, 0.0)})
    dt = driver.grid.dt
    # step.cpp:64-100 — an electric current is calculated at t + dt/2, and the run starts at 0.
    current = source.current(0.5 * dt, dt)
    assert abs(current.imag) > 0.5 * abs(current.real), (
        "the current is nearly real at this instant, so the wrong conventions below would "
        "coincide with the right one; pick another frequency"
    )
    driver.step()
    measured = driver.get_field("Dz", cell_centered=False)
    assert measured.dtype == np.float32

    def deposit(transform):  # What one convention would leave in Dz after the first step.
        expected = np.zeros(driver.shape, dtype=np.float32)
        for (ix, iy, iz), weight in zip(source._point_indices, source._point_weights):
            expected[ix, iy, iz] -= transform(np.complex64(dt * weight * current))
        return expected

    np.testing.assert_array_equal(
        measured, deposit(lambda value: value.real),
        err_msg="real-mode injection is not MEEP's real(amplitude * current * dt)",
    )
    scale = float(np.abs(measured).max())
    assert scale > 0.0, "nothing was injected; the comparison is vacuous"
    for label, transform in (
        ("imag(A)", lambda value: value.imag),
        ("abs(A)", lambda value: np.float32(abs(value))),
        ("2*real(A)", lambda value: np.float32(2.0 * value.real)),
    ):
        gap = float(np.abs(measured - deposit(transform)).max()) / scale
        assert gap > 0.1, f"the {label} convention lands within {gap:.3e} of real(A); this test is blind"
    # And the half-step matters: the same convention evaluated at t rather than t + dt/2.
    at_start = np.zeros(driver.shape, dtype=np.float32)
    early = source.current(0.0, dt)
    for (ix, iy, iz), weight in zip(source._point_indices, source._point_weights):
        at_start[ix, iy, iz] -= np.complex64(dt * weight * early).real
    assert float(np.abs(measured - at_start).max()) / scale > 0.01, (
        "injecting at t rather than t + dt/2 makes no difference here; the slot is untested"
    )
    driver.close()


@pytest.mark.parametrize("axis, k_point", [(0, (0.3, 0.0, 0.0)), (1, (0.0, -0.25, 0.0)),
                                           (2, (0.0, 0.0, 0.5))])
def test_real_mode_refuses_a_bloch_phase_on_any_axis(axis, k_point):
    """A boundary phase has nowhere to live in one real plane, on every axis alike.

    MEEP aborts on the same pairing (boundaries.cpp:103), and the alternative — promoting
    the run to complex to make it work — would silently double the memory of a run the
    caller sized deliberately, which is the one thing this mode exists to control.
    """
    with pytest.raises(ValueError, match="complex fields") as refusal:
        FdtdDriver(cell_size=(2, 2, 2), resolution=10, force_complex_fields=False,
                   k_point=k_point)
    # The offending axis is named, so the caller knows which component to zero.
    assert "XYZ"[axis] in str(refusal.value)
    # The same wavevector is served once the storage can carry the phase, so the refusal
    # is about the pairing and not about the k_point itself.
    FdtdDriver(cell_size=(2, 2, 2), resolution=10, force_complex_fields=True,
               k_point=k_point).close()


@pytest.mark.parametrize(
    "label, source_data",
    [
        ("point amplitude", {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                             "size": (0.0, 0.0, 0.0), "amplitude": 1j}),
        ("sheet amplitude", {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                             "size": (0.4, 0.4, 0.0), "amplitude": 0.5 + 0.5j}),
        ("gaussian amplitude", {"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                                "fwidth": 0.6, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                                "amplitude": -1j}),
        ("magnetic amplitude", {"component": "Hx", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                                "size": (0.0, 0.0, 0.0), "amplitude": 1j}),
        ("integrated amplitude", {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                                  "size": (0.0, 0.0, 0.0), "amplitude": 1j, "is_integrated": True}),
    ],
)
def test_real_mode_projects_a_complex_spatial_amplitude_on_every_source_family(label, source_data):
    """``Re(a*s)`` is MEEP's own step.cpp:307, taken on every source family alike.

    With ``a`` real, ``Re(a * s(t))`` is the physical current the caller wrote down.
    With ``a`` complex it is ``Re(a)Re(s) - Im(a)Im(s)``, which re-reads their spatial
    phase as a per-point time shift — and this engine used to REFUSE that pairing on
    that ground. MEEP takes ``real(A)`` unconditionally, so refusing did not make the
    engine stricter, it made it decline runs MEEP completes; a Gaussian beam's
    amplitude is complex by construction, so that was most real-field beam scripts.

    Pinned per family because each reaches the injection layer by a different route —
    the scalar point loop, the fancy-indexed sheet, the pulsed early-out, the magnetic
    B array, and the integrated path that deposits a dipole from ``update_eh`` instead
    of a current from ``step_source``. The assertion is the same for all five: real
    storage stays float32, something is actually deposited, and what is deposited is
    the complex run's real part exactly.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=False)
    driver.add_source(dict(source_data))
    driver.run(num_steps=3)
    complex_driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10,
                                force_complex_fields=True)
    complex_driver.add_source(dict(source_data))
    complex_driver.run(num_steps=3)

    deposited = 0.0
    for component in _PRIMARY_COMPONENTS:
        values = driver.get_field(component, cell_centered=False)
        reference = complex_driver.get_field(component, cell_centered=False)
        assert values.dtype == np.float32, f"{label}: {component} was promoted to {values.dtype}"
        np.testing.assert_array_equal(
            values, np.real(reference),
            err_msg=f"{label}: real {component} is not the complex run's real part",
        )
        deposited = max(deposited, float(np.abs(values).max()))
    assert deposited > 0.0, f"{label}: the projected source deposited nothing at all"
    driver.close()
    complex_driver.close()


def test_real_mode_projects_a_complex_amp_func_and_still_serves_a_real_one():
    """An ``amp_func`` is the practical way phase reaches the deposition table.

    A converging beam's spherical phase is written as an ``amp_func``, not as a scalar
    ``amplitude``, so the projection is checked on the STORED per-point weights rather
    than on the caller's argument. The second half is the control that keeps the first
    honest: an engine that had simply stopped applying ``amp_func`` at all on real
    storage would satisfy "no longer refuses" and fail here.
    """
    phased = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=False)
    phased.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.6, "center": (0.0, 0.0, 0.0), "size": (0.4, 0.4, 0.0),
                       "amp_func": lambda x, y, z: np.exp(1j * 2.0 * x)})
    phased.run(num_steps=3)
    complex_phased = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10,
                                force_complex_fields=True)
    complex_phased.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                               "fwidth": 0.6, "center": (0.0, 0.0, 0.0), "size": (0.4, 0.4, 0.0),
                               "amp_func": lambda x, y, z: np.exp(1j * 2.0 * x)})
    complex_phased.run(num_steps=3)
    ours = phased.get_field("Dz", cell_centered=False)
    theirs = complex_phased.get_field("Dz", cell_centered=False)
    assert float(np.abs(ours).max()) > 0.0, "the phased profile deposited nothing"
    np.testing.assert_array_equal(ours, np.real(theirs))
    # The phase is not decoration: a run that ignored it would deposit |A| or Re(a)*s
    # instead, and the imaginary plane it drops has to be a real quantity for the
    # equality above to mean anything.
    assert float(np.abs(np.imag(theirs)).max()) > 0.0, "nothing was dropped"
    phased.close()
    complex_phased.close()

    real_profile = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=False)
    real_profile.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                             "fwidth": 0.6, "center": (0.0, 0.0, 0.0), "size": (0.4, 0.4, 0.0),
                             "amp_func": lambda x, y, z: 1.0 + 0.5 * x})
    real_profile.run(num_steps=5)
    assert float(np.abs(real_profile.get_field("Dz", cell_centered=False)).max()) > 0.0
    real_profile.close()


def test_a_complex_time_envelope_is_not_refused_because_it_is_meeps_own_convention():
    """The refusal must be about the SPATIAL amplitude, not about any complex number.

    Every MEEP source carries a complex temporal waveform — ``ContinuousSource`` is
    ``exp(-i*2*pi*f*t)`` — and ``real(A)`` of it is precisely what a real run is. An
    over-broad guard that refused a complex ``src_func`` would make real mode unusable
    for the case it is the default for, so the permissive half is pinned too.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=False)
    driver.add_source({"component": "Ez", "source_type": "custom", "center": (0.0, 0.0, 0.0),
                       "size": (0.0, 0.0, 0.0),
                       "src_func": lambda t: np.exp(-1j * 2 * np.pi * t)})
    driver.run(num_steps=20)
    values = driver.get_field("Dz", cell_centered=False)
    assert values.dtype == np.float32 and float(np.abs(values).max()) > 0.0
    driver.close()


def test_real_mode_halves_the_field_allocation_of_a_live_run():
    """The memory claim, measured on a running driver rather than on the model alone.

    ``Fields.field_bytes_per_cell`` is unit-tested against the arrays it counts in
    ``test_fields.py``; what this adds is the end-to-end statement a caller sizing a GPU
    job actually needs — that a real run of a real configuration, PML and polarizations
    and all, holds half the field bytes a complex one does, and that the arrays on the
    driver agree with the number it reports.

    The total is deliberately NOT halved: ``eps`` and ``inv_eps`` are float32 in both
    modes, and over-promising 2x is how a job gets sized wrong in the other direction.
    """
    measurements = {}
    for complex_fields in (False, True):
        driver = FdtdDriver(cell_size=(2.1, 2.1, 2.5), resolution=10, courant=0.4,
                            force_complex_fields=complex_fields)
        driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
        driver.add_susceptibility(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)
        driver.setup_pml(5)
        driver.add_source({**_REAL_MODE_CW, "size": (0.4, 0.4, 0.0)})
        driver.run(num_steps=6)
        cells = driver.grid.total_cells
        expected_element = 4 if not complex_fields else 8
        for name in _PRIMARY_COMPONENTS + ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                                           "fu_Dz", "f_w_Ez"):
            array = getattr(driver.fields, name)
            assert array is not None, f"{name} was never allocated; the accounting is wrong"
            assert array.itemsize == expected_element, (
                f"{name} is {array.dtype} in a "
                f"{'complex' if complex_fields else 'real'} run"
            )
        for state in driver.fields.polarizations:
            for array in list(state.P.values()) + list(state.P_prev.values()):
                assert array.itemsize == expected_element, "a polarization array followed the wrong dtype"
        measurements[complex_fields] = (
            driver.fields.field_bytes_per_cell(), driver.fields.bytes_per_cell(), cells
        )
        driver.close()

    (real_field, real_total, cells) = measurements[False]
    (complex_field, complex_total, complex_cells) = measurements[True]
    assert cells == complex_cells
    assert 2 * real_field == complex_field, (
        f"a live real run holds {real_field} B/cell of field storage, not half of {complex_field}"
    )
    # The saving is a real number of bytes on a real grid, not a per-cell abstraction.
    assert (complex_field - real_field) * cells > 1_000_000
    assert real_total == real_field + 2 * 4, "eps and inv_eps are the only float32 remainder"
    assert 2 * real_total > complex_total, "the TOTAL is not halved; do not claim that it is"


# ---------------------------------------------------------------------------
# Complex input into a real-valued setter must raise, never drop Im silently
# ---------------------------------------------------------------------------


def _vacuum_driver(**kwargs):  # Smallest driver that can accept material and seed arrays.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, **kwargs)
    return driver


def test_a_complex_epsilon_is_refused_rather_than_silently_made_lossless():
    # NumPy would drop Im(eps) with only a ComplexWarning, leaving the run
    # indistinguishable from one handed the lossless medium -- the failure mode a
    # user porting `set_epsilon(n**2)` from MEEP with a complex n walks straight into.
    driver = _vacuum_driver()
    lossy = np.full(driver.shape, 2.25 + 0.1j, dtype=np.complex128)
    with pytest.raises(ValueError, match="must be real-valued"):
        driver.set_epsilon(lossy)
    # The refusal names the two ways to express the imaginary part.
    try:
        driver.set_epsilon(lossy)
    except ValueError as error:
        assert "set_conductivity" in str(error) and "add_susceptibility" in str(error)
    # Positive control: the real part alone is accepted, so the guard is not blanket.
    driver.set_epsilon(np.real(lossy).astype(np.float32))
    assert float(to_numpy(driver.fields.eps).max()) == pytest.approx(2.25)
    driver.close()


def test_a_complex_conductivity_is_refused_rather_than_silently_halved():
    driver = _vacuum_driver()
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    with pytest.raises(ValueError, match="must be real-valued"):
        driver.set_conductivity(np.full(driver.shape, 0.3 + 0.2j, dtype=np.complex128))
    with pytest.raises(ValueError, match="must be real-valued"):
        driver.set_conductivity(0.3 + 0.2j)  # Scalar spelling takes the same path.
    driver.set_conductivity(0.3)  # Positive control.
    assert driver.fields.has_conductivity
    driver.close()


def test_a_complex_seed_into_a_real_field_run_is_refused_but_allowed_in_complex_mode():
    # The same array is legitimate in complex storage and impossible in real storage;
    # the guard must key off the target dtype, not reject complex seeds outright.
    seed_shape = _vacuum_driver().shape
    seed = np.zeros(seed_shape, dtype=np.complex128)
    seed[1, 1, 1] = 1.0 + 2.0j

    real_run = _vacuum_driver(force_complex_fields=False)
    real_run.set_epsilon(np.full(real_run.shape, 1.0, dtype=np.float32))
    with pytest.raises(ValueError, match="real-field run"):
        real_run.set_field("Dz", seed)
    assert float(np.abs(to_numpy(real_run.fields.Dz)).max()) == 0.0, "no partial write before the raise"
    real_run.set_field("Dz", np.real(seed).astype(np.float32))  # Positive control.
    assert float(to_numpy(real_run.fields.Dz)[1, 1, 1]) == pytest.approx(1.0)
    real_run.close()

    complex_run = _vacuum_driver(force_complex_fields=True)
    complex_run.set_epsilon(np.full(complex_run.shape, 1.0, dtype=np.float32))
    complex_run.set_field("Dz", seed)  # Legitimate here: the imaginary part is storable.
    assert complex(to_numpy(complex_run.fields.Dz)[1, 1, 1]) == 1.0 + 2.0j
    complex_run.close()


# ---------------------------------------------------------------------------
# Near-to-far-field transformation
# ---------------------------------------------------------------------------
#
# `dft.py`'s own tests pin the Green's function against a textbook transcription and
# the surface measure against the geometric area of a patch, with no Maxwell solve in
# the way. What is pinned here is the whole chain end to end, three ways that do not
# share a failure mode:
#
#   1. Against the CLOSED-FORM field of a Hertzian dipole -- shape, absolute
#      amplitude, polarization and the near-field longitudinal term -- with the error
#      required to FALL when the grid is refined, which no tuned constant can fake at
#      two resolutions at once.
#   2. Against CPU MEEP's own `mp.Near2FarRegion` / `sim.get_farfield` on identical
#      runs, on odd and even cell counts, for an electric and a magnetic dipole, and
#      with the box and the source both off centre.
#   3. Against ENERGY CONSERVATION: the far-field Poynting integrated over a sphere
#      must equal the power the same run's flux monitors measure through the
#      near-field box itself. That is the one check with no free scale anywhere in it,
#      so a prefactor error -- a missing 4*pi, a stray factor of omega, the wrong area
#      element -- fails it and nothing else in the transformation can absorb it.
#
# The degenerate cases are here too, and they are the point of the block: a box that
# accumulated nothing must radiate EXACTLY zero rather than a small smooth pattern, a
# surface that does not enclose the source must be refused rather than served, a
# surface inside the absorber must be refused rather than quietly returning the right
# shape at the wrong power, and a monitor the driver already steps must refuse to be
# stepped again rather than double every accumulation.

N2F_FCEN, N2F_FWIDTH = 1.0, 1.0  # width = 1/fwidth = 1, which fixes the dipole moment at 1.
N2F_RES, N2F_PML_CELLS, N2F_UNTIL = 15, 9, 14.0
N2F_CELL, N2F_BOX = 2.4, 1.0  # PML interior +-0.6; the box faces sit at +-0.5.
N2F_POINTS = np.array([
    radius * np.array([math.sin(theta) * math.cos(phi),
                       math.sin(theta) * math.sin(phi),
                       math.cos(theta)])
    for radius in (4.0, 30.0)
    for theta in (0.3, 1.0, math.pi / 2, 2.2, math.pi - 0.35)
    for phi in (0.0, 1.4, 3.1, 4.9)
])

_NEAR2FAR_ORACLE_SCRIPT = '''"""CPU-MEEP near-to-far references for the driver's Near2FarMonitor."""
import sys

import numpy as np
import meep as mp

output_path = sys.argv[1]
points = np.load(output_path + ".points.npy")
RES, PML, FCEN, FWIDTH, UNTIL, BOX = 15, 9, 1.0, 1.0, 14.0, 1.0

# Identical physics to the driver cases below. `odd` differs from `even` only in the
# cell count -- 37 cells against 36 -- which is the sweep that found this engine
# registering a monitor half a cell off when the count was odd (port reference: every
# test used even counts, so a 4.8e-02 error was invisible).
CASES = {
    "even": dict(cell=2.4, box_center=(0, 0, 0), component="Ez", source=(0, 0, 0)),
    "odd": dict(cell=37 / 15, box_center=(0, 0, 0), component="Ez", source=(0, 0, 0)),
    "magnetic": dict(cell=2.4, box_center=(0, 0, 0), component="Hz", source=(0, 0, 0)),
    "offset": dict(cell=2.4, box_center=(0.11, -0.07, 0.13), component="Ex",
                   source=(0.093, -0.137, 0.061)),
}

results = {}
for name, case in CASES.items():
    regions = []
    for axis in range(3):
        for sign in (-1, +1):
            center = list(case["box_center"])
            center[axis] += sign * BOX / 2
            size = [BOX, BOX, BOX]
            size[axis] = 0.0
            regions.append(mp.Near2FarRegion(
                center=mp.Vector3(*center), size=mp.Vector3(*size),
                direction=[mp.X, mp.Y, mp.Z][axis], weight=float(sign)))
    simulation = mp.Simulation(
        cell_size=mp.Vector3(*(case["cell"],) * 3), resolution=RES,
        boundary_layers=[mp.PML(PML / RES)],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=FWIDTH),
                           component=getattr(mp, case["component"]),
                           center=mp.Vector3(*case["source"]))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
    near2far = simulation.add_near2far(FCEN, 0, 1, *regions, decimation_factor=1)
    simulation.run(until=UNTIL)
    results[name] = np.array(
        [simulation.get_farfield(near2far, mp.Vector3(*point)) for point in points],
        dtype=np.complex128)
np.savez(output_path, **results)
'''


@pytest.fixture(scope="module")
def near2far_reference(tmp_path_factory):  # One CPU-MEEP subprocess for every near2far case.
    directory = tmp_path_factory.mktemp("near2far_oracle")
    script_path = directory / "near2far_oracle.py"
    script_path.write_text(_NEAR2FAR_ORACLE_SCRIPT, encoding="utf-8")
    output_path = directory / "near2far_reference.npz"
    np.save(str(output_path) + ".points.npy", N2F_POINTS)
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=1800,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP near2far oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {key: archive[key] for key in archive.files}


def _near2far_run(cell=N2F_CELL, resolution=N2F_RES, pml_cells=N2F_PML_CELLS,
                  component="Ez", source_center=(0.0, 0.0, 0.0),
                  box_center=(0.0, 0.0, 0.0), box=N2F_BOX, until=N2F_UNTIL,
                  frequencies=(N2F_FCEN,), with_box_flux=False):
    """One driver run carrying a near-field box, and optionally its six flux planes."""
    driver = FdtdDriver(cell_size=(cell,) * 3, resolution=resolution, force_complex_fields=True)
    driver.setup_pml(pml_cells)
    driver.add_source({"component": component, "source_type": "gaussian",
                       "frequency": N2F_FCEN, "fwidth": N2F_FWIDTH,
                       "center": tuple(source_center)})
    monitor = Near2FarMonitor.on_driver(driver, frequencies=list(frequencies),
                                        center=tuple(box_center), size=(box,) * 3)
    planes = []
    if with_box_flux:
        for axis in range(3):
            for sign in (-1, +1):
                planes.append((sign, driver.add_flux_monitor(
                    frequencies=list(frequencies),
                    center=tuple(box_center[a] + (sign * box / 2 if a == axis else 0.0)
                                 for a in range(3)),
                    size=tuple(0.0 if a == axis else box for a in range(3)),
                    direction=axis)))
    driver.run(until=until)
    return driver, monitor, planes


def _hertzian_dipole(points, frequency=N2F_FCEN, eps=1.0, mu=1.0):
    """Closed-form fields of a unit z-directed point current at the origin.

    The textbook Hertzian dipole under MEEP's ``exp(-i*omega*t)`` convention, written
    here rather than imported: `E = i*omega*mu*Ghat.zhat`, `H = dg/dr * (rhat x zhat)`
    with the FULL dyadic, so the comparison covers the `1/r**2` and `1/r**3` terms as
    well as the radiating one. The nearest observation radius below is 4 wavelengths,
    where the longitudinal term is still 4% of the transverse one -- large enough that
    a far-field-only transformation fails this and a correct one does not.
    """
    positions = np.asarray(points, dtype=np.float64)
    radius = np.linalg.norm(positions, axis=1)
    unit = positions / radius[:, None]
    omega = 2 * np.pi * frequency
    wavenumber = omega * math.sqrt(eps * mu)
    green = np.exp(1j * wavenumber * radius) / (4 * np.pi * radius)
    kr = wavenumber * radius
    isotropic = 1.0 + (1j * kr - 1.0) / kr ** 2
    radial = (3.0 - 3j * kr - kr ** 2) / kr ** 2
    axis = np.array([0.0, 0.0, 1.0])
    electric = 1j * omega * mu * green[:, None] * (
        isotropic[:, None] * axis + radial[:, None] * unit * unit[:, 2:3])
    magnetic = (green * (1j * wavenumber - 1.0 / radius))[:, None] * np.cross(
        unit, np.broadcast_to(axis, unit.shape))
    return np.concatenate([electric, magnetic], axis=1)


def _dipole_residual(monitor, points=None):
    """Best-fit dipole moment and the residual of the far field around it."""
    observation = N2F_POINTS if points is None else points
    produced = monitor.farfields(observation, freq_index=0)
    exact = _hertzian_dipole(observation)
    moment = np.vdot(exact.ravel(), produced.ravel()) / np.vdot(exact.ravel(), exact.ravel())
    residual = np.linalg.norm(produced - moment * exact) / np.linalg.norm(moment * exact)
    return complex(moment), float(residual)


def _sphere_power(monitor, radius, freq_index=0, order=32):
    """Far-field radiated power: Gauss-Legendre in cos(theta), uniform in phi."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    theta = np.arccos(nodes)
    phi = np.linspace(0.0, 2 * np.pi, 2 * order, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    quadrature = np.outer(weights, np.full(phi.size, 2 * np.pi / phi.size))
    return float(np.sum(monitor.radiation_pattern(theta_grid, phi_grid, radius,
                                                  freq_index=freq_index) * quadrature))


def test_near2far_reproduces_the_closed_form_dipole_field():
    """The far field of an enclosed point dipole, against the analytic Hertzian dipole.

    Four separate statements, because a transformation can get the SHAPE right and the
    scale wrong, or the scale right and the polarization wrong:

    * The full six-component field, at 40 points spanning two radii and the whole
      sphere, matches the closed form to 1.0e-02 relative (measured 1.0161e-02 at
      resolution 15) after fitting ONE complex amplitude.
    * That amplitude is not free. MEEP's ``GaussianSource`` has ``width = 1/fwidth``
      and ``peak_time = 5*width``, so its transform at the carrier is exactly
      ``width * exp(i*omega*peak_time)`` = 1 for ``fwidth = 1`` and ``fcen = 1``;
      the fit returns |p| = 1.00025. The absolute far-field amplitude is therefore
      pinned with no free parameter at all. (Confirmed against the derivation by
      sweeping fwidth: 0.5 -> |p| = 2.0005, 1.0 -> 1.00025, 2.0 -> 0.50013.)
    * The field is transverse and axially symmetric to nine decimal places, which the
      pattern comparison alone would not show.
    * The RATIO of the longitudinal to the transverse component matches the Hertzian
      near-field term ``2*cot(theta)/(k*r)``. That number is only produced by the full
      dyadic Green's function; a transformation using its far-field limit would return
      zero there and pass every other assertion in this test.
    """
    driver, monitor, _ = _near2far_run()
    moment, residual = _dipole_residual(monitor)
    assert residual < 0.02, f"far field departs from the Hertzian dipole by {residual:.4e}"
    assert abs(moment) == pytest.approx(1.0 / N2F_FWIDTH, rel=0.01), (
        f"the absolute far-field amplitude must be the source's own transform "
        f"({1.0 / N2F_FWIDTH}), got |p| = {abs(moment):.6f}"
    )
    assert abs(np.angle(moment)) < 0.2, (
        f"the fitted phase is the numerical-dispersion delay over half a cell width and "
        f"must be small; got {np.angle(moment):.4f} rad"
    )

    # Polarization and symmetry at one representative direction.
    theta, radius = 0.7, 25.0
    point = radius * np.array([math.sin(theta), 0.0, math.cos(theta)])
    field = monitor.farfield(point)
    radial_unit = point / radius
    polar_unit = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    azimuth_unit = np.array([0.0, 1.0, 0.0])
    electric = field[0:3]
    assert abs(np.dot(electric, azimuth_unit)) < 1e-7 * np.linalg.norm(electric), (
        "a z dipole radiates no azimuthal E"
    )
    assert np.linalg.norm(electric) / np.linalg.norm(field[3:6]) == pytest.approx(1.0, rel=1e-3), (
        "|E|/|H| must be the free-space impedance, which is 1 in MEEP units"
    )
    wavenumber = 2 * np.pi * N2F_FCEN
    longitudinal = np.dot(electric, radial_unit) / np.dot(electric, polar_unit)
    expected = (2.0 / math.tan(theta)) * (1.0 / (wavenumber * radius) ** 2
                                          - 1j / (wavenumber * radius))
    expected /= -(1.0 + (1j * wavenumber * radius - 1.0) / (wavenumber * radius) ** 2)
    assert abs(longitudinal - expected) < 0.05 * abs(expected), (
        f"the longitudinal near-field term must be the Hertzian one: "
        f"{longitudinal:.6f} vs {expected:.6f}"
    )
    assert abs(longitudinal) > 0.01, "this assertion is vacuous unless the term is measurable"
    driver.close()


def test_near2far_dipole_error_falls_when_the_grid_is_refined():
    """The analytic error must be discretization, not a fixed offset a constant could hide.

    Measured against the closed-form dipole: 2.4913e-02 at resolution 10 and
    6.3156e-03 at resolution 20, a factor of 3.9 for a factor of two in the grid. A
    tuned prefactor can satisfy one resolution; only a correct transformation of a
    correct near field converges. The radiation pattern's departure from ``sin^2`` is
    required to fall with it, so the shape converges as well as the norm.
    """
    errors, spreads = {}, {}
    angles = np.linspace(0.15, np.pi - 0.15, 13)
    for resolution in (10, 20):
        driver, monitor, _ = _near2far_run(resolution=resolution,
                                           pml_cells=int(round(0.6 * resolution)))
        _, errors[resolution] = _dipole_residual(monitor)
        pattern = monitor.radiation_pattern(angles, 0.0, 20.0) / np.sin(angles) ** 2
        spreads[resolution] = float((pattern.max() - pattern.min()) / pattern.mean())
        driver.close()
    assert errors[20] < 0.5 * errors[10], (
        f"the analytic error must fall with resolution: {errors[10]:.4e} -> {errors[20]:.4e}"
    )
    assert errors[20] < 0.01 and errors[10] < 0.04
    assert spreads[20] < 0.7 * spreads[10], (
        f"the pattern's departure from sin^2 must fall too: "
        f"{spreads[10]:.4e} -> {spreads[20]:.4e}"
    )


def test_near2far_radiated_power_equals_the_near_field_box_flux():
    """Energy conservation, the one check with no free scale in it.

    The far-field Poynting integrated over a sphere is the power crossing the
    near-field box, which the same run's six flux planes measure directly. Measured
    2.0933 against 2.0865 at resolution 15 -- 0.33% -- and the sphere integral is
    independent of its own radius to six digits, which is the 1/r**2 falloff asserted
    rather than assumed. Any error in the absolute prefactor (the 1/(4*pi), the
    i*omega*mu, the dx**2 area element, the fractional-cell weights) shows up here and
    nowhere else in this file, because every other comparison either fits an amplitude
    or compares against a code that would have to share the mistake.
    """
    driver, monitor, planes = _near2far_run(with_box_flux=True)
    box_power = sum(sign * plane.get_flux(0) for sign, plane in planes)
    assert box_power > 0.0, "the box must carry outgoing power for this comparison to mean anything"
    near = _sphere_power(monitor, 20.0)
    far = _sphere_power(monitor, 200.0)
    assert near == pytest.approx(box_power, rel=0.02), (
        f"far-field sphere power {near:.6g} does not match the box flux {box_power:.6g}"
    )
    assert far == pytest.approx(near, rel=1e-4), (
        f"the radiated power must not depend on the sphere it is measured on: "
        f"{near:.8g} at r=20 against {far:.8g} at r=200"
    )
    driver.close()


@requires_meep
@skip_without_meep
def test_near2far_matches_cpu_meep(near2far_reference):
    """Against `mp.Near2FarRegion` / `sim.get_farfield` on identical runs.

    Four cases, all within the 5% bar: an electric dipole in a 36-cell cell
    (3.4281e-02), the same in a 37-cell one (3.4282e-02), a magnetic dipole
    (1.6484e-02), and an off-centre Ex dipole in an off-centre box (2.2428e-02).

    The residual is not a normalization difference -- the best-fit complex scale
    between the two is 0.970 to 1.001 and converges to 1 as the grid is refined
    (0.98191 at resolution 15, 0.99911 at resolution 30, with the L2 falling 2.81e-02
    -> 9.18e-03). It is where the equivalent currents are EVALUATED: this engine
    samples every component at cell centres, so `J` and `M` describe the same point,
    while MEEP evaluates each at its own Yee position half a cell apart. Both are
    consistent discretizations of the same integral; measured against the closed-form
    dipole, this engine's is the closer of the two (departure from `sin^2` at
    resolution 30: 1.77e-02 here against 3.01e-02 for MEEP on the same run).

    ODD AGAINST EVEN is the assertion this case exists for. The two cell counts land
    within 3e-05 of each other in relative L2, so the registration carries no cell-parity
    dependence -- the defect class that once put a monitor half a cell off and stayed
    invisible because every test used an even count.
    """
    measured = {}
    for name, case in (
        ("even", dict()),
        ("odd", dict(cell=37 / 15)),
        ("magnetic", dict(component="Hz")),
        ("offset", dict(component="Ex", source_center=(0.093, -0.137, 0.061),
                        box_center=(0.11, -0.07, 0.13))),
    ):
        driver, monitor, _ = _near2far_run(**case)
        produced = monitor.farfields(N2F_POINTS, freq_index=0)
        measured[name] = _complex_relative_l2(produced, near2far_reference[name])
        assert measured[name] < CROSS_VALIDATION_BAR, (
            f"near2far case '{name}' is {measured[name]:.4e} from CPU MEEP"
        )
        driver.close()
    assert abs(measured["odd"] - measured["even"]) < 0.01 * measured["even"], (
        f"an odd cell count must register exactly as an even one does: "
        f"{measured['odd']:.6e} against {measured['even']:.6e}"
    )


def test_near2far_with_no_source_inside_is_exactly_zero_not_merely_small():
    """An empty box must radiate exactly nothing, with a control that is not zero.

    This is the degenerate case that would otherwise score as a clean result: an
    accumulator that never saw a field reduces to a perfectly smooth pattern of zeros
    at every angle and every radius, and nothing about it says the run recorded
    nothing. Zero is the honest answer, so it is asserted EXACTLY -- and the same
    geometry with a source in it is asserted to produce signal, so a transformation
    that always returns zero cannot pass.
    """
    empty = FdtdDriver(cell_size=(N2F_CELL,) * 3, resolution=N2F_RES, force_complex_fields=True)
    empty.setup_pml(N2F_PML_CELLS)
    monitor = Near2FarMonitor.on_driver(empty, frequencies=[N2F_FCEN],
                                        center=(0.0, 0.0, 0.0), size=(N2F_BOX,) * 3)
    empty.run(until=4.0)
    silent = monitor.farfields(N2F_POINTS, freq_index=0)
    assert np.count_nonzero(silent) == 0, (
        f"a run with no source must radiate exactly zero, got max |field| "
        f"{np.max(np.abs(silent)):.3e}"
    )
    assert _sphere_power(monitor, 20.0) == 0.0
    empty.close()

    driver, loud, _ = _near2far_run()
    signal = loud.farfields(N2F_POINTS, freq_index=0)
    assert np.min(np.abs(signal)) > 0.0, "the positive control must radiate at every point"
    driver.close()


def test_near2far_is_extinguished_inside_the_closed_surface():
    """Love's equivalence: the currents radiate the true field outside and none inside.

    A sign error on either equivalent current -- `J = n x H`, `M = -n x E` -- leaves a
    far field that is still smooth, still falls off as 1/r and still looks like a
    radiation pattern, but destroys the interior cancellation completely. Measured at
    resolution 15, a tenth of a wavelength either side of a face: 2.0e-02 inside
    against 4.2e-01 outside, a factor of 20. The ratio falls with resolution
    (8.2e-02 at res 10, 4.9e-02 at res 15, 2.5e-02 at res 20), which is what makes it a
    cancellation rather than a coincidence.
    """
    ratios = {}
    for resolution in (10, 20):
        driver, monitor, _ = _near2far_run(resolution=resolution,
                                           pml_cells=int(round(0.6 * resolution)))
        gap = 0.12  # Comfortably outside the half-cell the transformation refuses inside.
        inside = np.max(np.abs(monitor.farfield([0.0, 0.0, N2F_BOX / 2 - gap])))
        outside = np.max(np.abs(monitor.farfield([0.0, 0.0, N2F_BOX / 2 + gap])))
        assert outside > 0.0
        ratios[resolution] = inside / outside
        driver.close()
    assert ratios[10] < 0.15, f"the interior field must be extinguished, got {ratios[10]:.3e}"
    assert ratios[20] < 0.5 * ratios[10], (
        f"the extinction must improve with resolution: {ratios[10]:.3e} -> {ratios[20]:.3e}"
    )


def test_near2far_refuses_a_surface_that_does_not_enclose_the_source():
    """A source outside the box is radiated as though it were inside, and looks fine.

    The equivalence theorem replaces the sources the surface ENCLOSES. One outside it
    is part of the incident field on the surface, and transforming anyway produces a
    pattern with no defect to see in it. So it is a refusal, with the source that
    caused it named.
    """
    driver = FdtdDriver(cell_size=(N2F_CELL,) * 3, resolution=N2F_RES, force_complex_fields=True)
    driver.setup_pml(N2F_PML_CELLS)
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": N2F_FCEN,
                       "fwidth": N2F_FWIDTH, "center": (0.0, 0.0, 0.55)})
    with pytest.raises(ValueError, match="not inside the near-field box"):
        Near2FarMonitor.on_driver(driver, frequencies=[N2F_FCEN],
                                  center=(0.0, 0.0, 0.0), size=(N2F_BOX,) * 3)
    # An extended source only PARTLY outside is refused on the same terms.
    driver.close()

    straddling = FdtdDriver(cell_size=(N2F_CELL,) * 3, resolution=N2F_RES,
                            force_complex_fields=True)
    straddling.setup_pml(N2F_PML_CELLS)
    straddling.add_source({"component": "Ez", "source_type": "gaussian",
                           "frequency": N2F_FCEN, "fwidth": N2F_FWIDTH,
                           "center": (0.0, 0.0, 0.3), "size": (0.2, 0.2, 0.6)})
    with pytest.raises(ValueError, match="not inside the near-field box"):
        Near2FarMonitor.on_driver(straddling, frequencies=[N2F_FCEN],
                                  center=(0.0, 0.0, 0.0), size=(N2F_BOX,) * 3)
    # Positive control: the same extended source fully inside a box that contains it.
    inside = Near2FarMonitor.on_driver(straddling, frequencies=[N2F_FCEN],
                                       center=(0.0, 0.0, 0.15), size=(0.9, 0.9, 0.9))
    assert inside.closed is True
    # And an OPEN surface is exempt: the caller has already said the premise does not hold.
    aperture = Near2FarMonitor.on_driver(
        straddling, frequencies=[N2F_FCEN], closed=False,
        regions=[Near2FarRegion(center=(0.0, 0.0, 0.5), size=(1.0, 1.0, 0.0), weight=1.0)])
    assert aperture.closed is False
    straddling.close()


def test_near2far_refuses_a_surface_inside_the_absorbing_layer():
    """A patch in the PML returns the right pattern at the wrong power.

    Sphere-integrated radiated power for the same dipole at resolution 15 with a
    9-cell PML (interior +-0.6): box 1.0 -> 2.0933, box 1.2 (faces exactly on the
    interior edge) -> 2.1129, box 1.4 -> 1.9401, box 1.6 -> 1.1643. A 7.3% and then a
    44% loss, on a pattern that stays the same clean dipole lobe throughout, which is
    why this is a refusal and not a warning. The boundary of the refusal is the
    interior edge plus the same half-cell allowance the source check uses.
    """
    driver = FdtdDriver(cell_size=(N2F_CELL,) * 3, resolution=N2F_RES, force_complex_fields=True)
    driver.setup_pml(N2F_PML_CELLS)
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": N2F_FCEN,
                       "fwidth": N2F_FWIDTH, "center": (0.0, 0.0, 0.0)})
    for box in (1.4, 1.6):
        with pytest.raises(ValueError, match="outside the PML interior"):
            Near2FarMonitor.on_driver(driver, frequencies=[N2F_FCEN],
                                      center=(0.0, 0.0, 0.0), size=(box,) * 3)
    # Touching the interior edge is legal and is the standard configuration.
    assert Near2FarMonitor.on_driver(driver, frequencies=[N2F_FCEN], center=(0.0, 0.0, 0.0),
                                     size=(1.2,) * 3).closed is True
    # An off-centre box is checked face by face, not by its size.
    with pytest.raises(ValueError, match="outside the PML interior"):
        Near2FarMonitor.on_driver(driver, frequencies=[N2F_FCEN],
                                  center=(0.0, 0.0, 0.25), size=(1.0,) * 3)
    driver.close()

    # A run with no absorber constrains nothing: the guard must key off the layer.
    free = FdtdDriver(cell_size=(N2F_CELL,) * 3, resolution=N2F_RES, force_complex_fields=True)
    free.add_source({"component": "Ez", "source_type": "gaussian", "frequency": N2F_FCEN,
                     "fwidth": N2F_FWIDTH, "center": (0.0, 0.0, 0.0)})
    assert Near2FarMonitor.on_driver(free, frequencies=[N2F_FCEN], center=(0.0, 0.0, 0.0),
                                     size=(2.0,) * 3).closed is True
    free.close()


def test_near2far_faces_registered_on_a_driver_are_stepped_exactly_once():
    """The driver-owned and the hand-stepped monitor must produce the same far field.

    A near-field surface is six DFT monitors, and the two ways of driving them -- the
    driver's own monitor list, or `update()` called from a stepping loop -- must not
    both run. Doubling every accumulation scales the far field by exactly two, which
    reads as a calibration question rather than as a bug, so `update()` on a
    driver-registered monitor raises. The equality below is what makes that refusal
    safe to rely on: the same run, transformed both ways, agrees to round-off.
    """
    driver = FdtdDriver(cell_size=(N2F_CELL,) * 3, resolution=N2F_RES, force_complex_fields=True)
    driver.setup_pml(N2F_PML_CELLS)
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": N2F_FCEN,
                       "fwidth": N2F_FWIDTH, "center": (0.0, 0.0, 0.0)})
    registered = Near2FarMonitor.on_driver(driver, frequencies=[N2F_FCEN],
                                           center=(0.0, 0.0, 0.0), size=(N2F_BOX,) * 3)
    standalone = Near2FarMonitor.box(driver.grid, frequencies=[N2F_FCEN],
                                     center=(0.0, 0.0, 0.0), size=(N2F_BOX,) * 3)
    while driver.time < N2F_UNTIL:
        driver.step()
        standalone.update(driver.fields, driver.time, driver.step_count)
    by_driver = registered.farfields(N2F_POINTS, freq_index=0)
    by_hand = standalone.farfields(N2F_POINTS, freq_index=0)
    assert np.max(np.abs(by_driver)) > 0.0
    assert _complex_relative_l2(by_hand, by_driver) < 1e-6, (
        "a hand-stepped near-field surface must reproduce the driver-stepped one"
    )
    with pytest.raises(RuntimeError, match="accumulate every timestep twice"):
        registered.update(driver.fields, driver.time, driver.step_count)
    # And the doubling the refusal prevents is real: stepping the standalone monitor a
    # second time over the same run scales its far field by exactly two.
    doubled = Near2FarMonitor.box(driver.grid, frequencies=[N2F_FCEN],
                                  center=(0.0, 0.0, 0.0), size=(N2F_BOX,) * 3)
    for _ in range(2):
        doubled.update(driver.fields, driver.time, driver.step_count)
    once = Near2FarMonitor.box(driver.grid, frequencies=[N2F_FCEN],
                               center=(0.0, 0.0, 0.0), size=(N2F_BOX,) * 3)
    once.update(driver.fields, driver.time, driver.step_count)
    np.testing.assert_allclose(
        doubled.farfields(N2F_POINTS[:4], freq_index=0),
        2.0 * once.farfields(N2F_POINTS[:4], freq_index=0), rtol=1e-6, atol=0.0)
    driver.close()


def test_near2far_far_field_before_stepping_is_refused_not_answered():
    """The driver resolves a monitor's region when stepping starts, not when it is added.

    Read before then, every patch still covers the whole cell and the surface integral
    would be a volume integral over the entire domain -- a large, smooth, entirely
    wrong number that no shape check would catch.
    """
    driver = FdtdDriver(cell_size=(N2F_CELL,) * 3, resolution=N2F_RES, force_complex_fields=True)
    driver.setup_pml(N2F_PML_CELLS)
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": N2F_FCEN,
                       "fwidth": N2F_FWIDTH, "center": (0.0, 0.0, 0.0)})
    monitor = Near2FarMonitor.on_driver(driver, frequencies=[N2F_FCEN],
                                        center=(0.0, 0.0, 0.0), size=(N2F_BOX,) * 3)
    with pytest.raises(ValueError, match="has not been registered on the grid yet"):
        monitor.farfield([0.0, 0.0, 8.0])
    driver.run(num_steps=1)
    monitor.farfield([0.0, 0.0, 8.0])  # Registered now; the call is legal.
    driver.close()


def test_near2far_multi_frequency_bins_are_independent():
    """A spectrum of far fields must not be one frequency's answer repeated.

    Each bin is compared against a dedicated single-frequency monitor on its own run,
    which is the contrast a broadband monitor carrying the first bin's phase
    everywhere would fail while still looking like a spectrum.
    """
    frequencies = (0.85, 1.0, 1.2)
    driver, broadband, _ = _near2far_run(frequencies=frequencies)
    spectrum = broadband.farfield_spectrum(N2F_POINTS)
    assert spectrum.shape == (len(N2F_POINTS), len(frequencies), 6)
    driver.close()
    for index, frequency in enumerate(frequencies):
        solo_driver, solo, _ = _near2far_run(frequencies=(frequency,))
        alone = solo.farfields(N2F_POINTS, freq_index=0)
        assert _complex_relative_l2(spectrum[:, index, :], alone) < 1e-6, (
            f"bin {index} of the broadband monitor must equal its own single-frequency run"
        )
        solo_driver.close()
    for first in range(len(frequencies)):
        for second in range(first + 1, len(frequencies)):
            separation = _complex_relative_l2(spectrum[:, first, :], spectrum[:, second, :])
            assert separation > 0.05, (
                f"bins {first} and {second} are {separation:.2e} apart; a spectrum whose bins "
                f"all carry one frequency's answer would look exactly like this"
            )


def test_near2far_in_a_dielectric_medium_uses_its_index_and_impedance():
    """The ambient medium enters as `k = omega*sqrt(eps*mu)` and `Z = sqrt(mu/eps)`.

    Every other case in this block runs in vacuum, where `eps` and `mu` are both one
    and a transformation that confused them would be indistinguishable from a correct
    one — the gap a mutation run found. In a uniform `eps = 2.25` fill the two separate:
    the far field must be the Hertzian dipole of a medium with index 1.5 (residual
    2.0050e-02 against the closed form, the same size as the vacuum case's
    1.1458e-02), and its wave impedance must be `sqrt(mu/eps) = 0.666667`, measured
    0.666694. Swapping `eps` for `mu` in the magnetic equation moves that ratio by a
    factor of 2.25 and leaves everything else looking correct.
    """
    permittivity = 2.25
    driver = FdtdDriver(cell_size=(N2F_CELL,) * 3, resolution=N2F_RES, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, permittivity, dtype=np.float32))
    driver.setup_pml(N2F_PML_CELLS)
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": N2F_FCEN,
                       "fwidth": N2F_FWIDTH, "center": (0.0, 0.0, 0.0)})
    monitor = Near2FarMonitor.on_driver(driver, frequencies=[N2F_FCEN], center=(0.0, 0.0, 0.0),
                                        size=(N2F_BOX,) * 3, eps=permittivity, mu=1.0)
    driver.run(until=N2F_UNTIL)

    produced = monitor.farfields(N2F_POINTS, freq_index=0)
    exact = _hertzian_dipole(N2F_POINTS, eps=permittivity, mu=1.0)
    moment = np.vdot(exact.ravel(), produced.ravel()) / np.vdot(exact.ravel(), exact.ravel())
    residual = np.linalg.norm(produced - moment * exact) / np.linalg.norm(moment * exact)
    assert residual < 0.04, (
        f"the far field in an eps={permittivity} medium departs from that medium's "
        f"Hertzian dipole by {residual:.4e}"
    )
    theta, radius = 0.7, 25.0
    field = monitor.farfield(radius * np.array([math.sin(theta), 0.0, math.cos(theta)]))
    impedance = np.linalg.norm(field[0:3]) / np.linalg.norm(field[3:6])
    assert impedance == pytest.approx(math.sqrt(1.0 / permittivity), rel=1e-3), (
        f"the far-field wave impedance must be sqrt(mu/eps) = "
        f"{math.sqrt(1.0 / permittivity):.6f}, got {impedance:.6f}"
    )
    # The vacuum answer is a different one: a transformation ignoring `eps` would
    # return that instead, and the wavelength difference is 50%.
    vacuum = _hertzian_dipole(N2F_POINTS, eps=1.0, mu=1.0)
    vacuum_moment = np.vdot(vacuum.ravel(), produced.ravel()) / np.vdot(vacuum.ravel(), vacuum.ravel())
    vacuum_residual = np.linalg.norm(produced - vacuum_moment * vacuum) / np.linalg.norm(
        vacuum_moment * vacuum)
    assert vacuum_residual > 10 * residual, (
        f"this test is only meaningful if the medium changes the answer: eps={permittivity} "
        f"residual {residual:.3e} against vacuum residual {vacuum_residual:.3e}"
    )
    driver.close()


# --------------------------------------------------------------------------------
# Point sampling and the step-function protocol — the two hooks Harminv runs on.
# --------------------------------------------------------------------------------

_ALL_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Dx", "Dy", "Dz", "Bx", "By", "Bz")


def _independent_point_sample(driver, component, point):
    """Trilinear read of one component from the raw Yee array, written from scratch.

    Deliberately shares nothing with ``FdtdDriver._build_point_field_reader`` — it pulls
    the whole volume to the host, recomputes the fractional index from
    ``axis_origin`` and the component's Yee shift, and sums the eight corners in plain
    NumPy. A stencil that shifted, or a weight that went to the wrong corner, would
    have to be wrong the same way in both to escape.
    """
    from .dft import yee_shifts

    shifts = yee_shifts(component)
    array = np.asarray(driver.get_field(component, cell_centered=False))
    lows, weights = [], []
    for axis in range(3):
        fractional = (
            (point[axis] - driver.grid.axis_origin(axis)) / driver.grid.dx - 0.5 * shifts[axis]
        )
        low = int(np.floor(fractional))
        assert 0 <= low and low + 1 < array.shape[axis], "the reference stencil left the grid"
        lows.append(low)
        weights.append((1.0 - (fractional - low), fractional - low))
    total = 0.0 + 0.0j
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                total += (
                    weights[0][i] * weights[1][j] * weights[2][k]
                    * array[lows[0] + i, lows[1] + j, lows[2] + k]
                )
    return total


def _pulsed_driver(cell_size, resolution, **kwargs):
    driver = FdtdDriver(cell_size=cell_size, resolution=resolution, **kwargs)
    driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
    driver.add_source({
        "source_type": "gaussian", "component": "Ez", "frequency": 0.35, "fwidth": 0.4,
        "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": 1.0,
    })
    return driver


@pytest.mark.parametrize(
    "cell_size, resolution",
    [((1.0, 1.0, 1.0), 12), ((1.1, 0.9, 1.3), 10), ((1.0, 1.1, 0.9), 11)],
    ids=["even", "odd_mixed", "odd_even_odd"],
)
def test_get_field_point_matches_an_independent_trilinear_read(cell_size, resolution):
    """MEEP's ``get_field_point`` is trilinear interpolation, and this is the arithmetic.

    Every one of the twelve components, at an off-centre point that lands on no grid
    line, on grids with EVEN and ODD cell counts on each axis — the parity matters
    because ``axis_origin`` is half a cell higher on an odd axis and a stencil built
    from ``-L/2`` instead would be off by exactly that on odd axes only.

    Measured worst deviation from the independent read: 1.2e-07, which is float32
    field storage and not the interpolation.
    """
    driver = _pulsed_driver(cell_size, resolution)
    driver.run(num_steps=40)
    point = (0.113, -0.071, 0.037)
    worst = 0.0
    for component in _ALL_COMPONENTS:
        expected = _independent_point_sample(driver, component, point)
        produced = driver.get_field_point(component, point)
        worst = max(worst, abs(produced - expected) / max(abs(expected), 1e-30))
    assert worst < 1e-5, f"point sampling deviates from an independent trilinear read by {worst:.2e}"
    # It has to be reading a field at all.
    assert abs(driver.get_field_point("Ez", point)) > 1e-8
    driver.close()


def test_get_field_point_reduces_to_the_stored_sample_at_a_yee_position():
    """No interpolation to hide behind: at a component's own Yee point the answer is the cell.

    Ez sits at ``(x_i, y_j, z_k + dz/2)``, so asking for exactly that point must return
    ``Ez[i, j, k]`` and not a blend with its neighbour. This is the check that catches a
    half-cell registration error, which every smooth-field test passes.
    """
    driver = _pulsed_driver((1.0, 1.0, 1.0), 10)
    driver.run(num_steps=25)
    stored = np.asarray(driver.get_field("Ez", cell_centered=False))
    dx = driver.grid.dx
    for index in ((3, 4, 5), (6, 2, 7)):
        point = tuple(
            driver.grid.axis_origin(axis) + (index[axis] + 0.5 * shift) * dx
            for axis, shift in enumerate((0, 0, 1))    # IYEE_SHIFTS["Ez"]
        )
        produced = driver.get_field_point("Ez", point)
        assert produced == pytest.approx(complex(stored[index]), rel=1e-6, abs=1e-30)
        # Half a cell off in z is a DIFFERENT number, so the test can tell them apart.
        shifted = driver.get_field_point("Ez", (point[0], point[1], point[2] + 0.5 * dx))
        assert abs(shifted - produced) > 1e-3 * abs(produced)
    driver.close()


def test_get_field_point_carries_the_bloch_factor_across_the_lattice():
    """A periodic cell has a field outside itself, and it is the phased image.

    ``Grid.bloch_phase``: ``f(x + L) = exp(i*2*pi*k*L) * f(x)``. A point one lattice
    vector up must come back with exactly that factor — including for a point in the
    outer HALF CELL, where the two interpolation legs land on opposite faces and only a
    per-leg fold can express it.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, k_point=(0.3, 0.0, 0.0))
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    driver.add_source({
        "source_type": "gaussian", "component": "Ez", "frequency": 0.35, "fwidth": 0.4,
        "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": 1.0,
    })
    driver.run(num_steps=30)
    length = driver.grid.Lx
    factor = np.exp(2j * np.pi * 0.3 * length)
    for point in ((0.13, 0.05, -0.02), (-0.44, 0.11, 0.07)):
        inside = driver.get_field_point("Ez", point)
        assert abs(inside) > 1e-10
        for images in (1, -1, 3):
            imaged = driver.get_field_point("Ez", (point[0] + images * length,) + point[1:])
            assert imaged == pytest.approx(inside * factor ** images, rel=1e-6)
    # The outer half cell of a half-shifted component: its lower leg is off the end and
    # has to be fetched from the far face with the phase on it.
    edge = driver.get_field_point("Ez", (-0.5, 0.05, -0.02))
    assert np.isfinite(edge) and abs(edge) > 0.0
    driver.close()


@pytest.mark.parametrize("axis", ["X", "Y"])
def test_get_field_point_unfolds_a_mirrored_axis(axis):
    """``pt`` is in USER coordinates, so a folded run must answer for the half it dropped.

    The engine stores only ``x >= 0`` on a folded axis. MEEP reflects the point and
    multiplies by the component's parity (``S.phase_shift``), and so does this: the
    check is that a folded run and the identical unfolded run agree at the SAME
    physical point on both sides of the plane, sign included.

    Measured against a full-domain run at (-0.23, -0.11, -0.07) and its mirror: 8.3e-06
    or better on every component that carries a field, with Hx and Hy coming back with
    the OPPOSITE sign either side of an X plane — which is the whole point, and what a
    fold that forgot the parity would get exactly wrong while still looking smooth.
    """
    def build(symmetry):
        driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, symmetry=symmetry)
        driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
        driver.add_source({
            "source_type": "gaussian", "component": "Ez", "frequency": 0.35, "fwidth": 0.4,
            "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": 1.0,
        })
        driver.run(num_steps=25)                # Short enough that the far face stays dark.
        return driver

    full, folded = build(()), build((axis,))
    reference = abs(full.get_field_point("Ez", (0.23, 0.11, 0.07)))
    assert reference > 1e-8
    flipped = 0
    for component in ("Ez", "Ex", "Ey", "Hx", "Hy", "Hz"):
        for point in ((-0.23, -0.11, -0.07), (0.23, 0.11, 0.07)):
            expected = full.get_field_point(component, point)
            if abs(expected) < 1e-6 * reference:
                continue                        # Identically zero by symmetry; nothing to compare.
            produced = folded.get_field_point(component, point)
            assert produced == pytest.approx(expected, rel=1e-4), (
                f"{component} at {point} under a {axis} mirror: folded {produced} against "
                f"full-domain {expected}"
            )
        mirrored_axis = "XYZ".index(axis)
        low = tuple(-v if i == mirrored_axis else v for i, v in enumerate((0.23, 0.11, 0.07)))
        across = full.get_field_point(component, low)
        here = full.get_field_point(component, (0.23, 0.11, 0.07))
        if abs(here) > 1e-6 * reference and abs(across + here) < 1e-4 * abs(here):
            flipped += 1
    assert flipped >= 1, (
        f"no component came back ODD about the {axis} plane, so this test never exercised "
        f"the parity factor at all"
    )
    full.close()
    folded.close()


def test_a_point_reader_survives_reset_and_a_storage_switch():
    """The cached stencil must never outlive the arrays it reads.

    ``get_field_point`` caches the (indices, weights, phase) for a (component, point)
    pair. If it cached the ARRAY instead, a ``reset()`` — which reallocates — would keep
    answering from the old run, and the numbers would stay smooth and plausible.
    """
    driver = _pulsed_driver((1.0, 1.0, 1.0), 10)
    driver.run(num_steps=20)
    point = (0.13, 0.07, 0.02)
    before = driver.get_field_point("Ez", point)
    assert abs(before) > 1e-10

    driver.reset()
    assert driver.get_field_point("Ez", point) == 0.0, "a reset driver still reports a field"
    driver.run(num_steps=20)
    assert driver.get_field_point("Ez", point) == pytest.approx(before, rel=1e-12)

    # E is computed on demand without PML and stored with it; a reader built in one mode
    # must not keep reading the other.
    driver.reset()
    assert not driver.fields.stores_E
    driver.setup_pml({"z": 4})
    assert driver.fields.stores_E
    driver.run(num_steps=20)
    assert driver.get_field_point("Ez", point) == pytest.approx(
        complex(_independent_point_sample(driver, "Ez", point)), rel=1e-5
    )
    driver.close()


def test_get_field_point_refuses_what_it_cannot_answer():
    driver = _pulsed_driver((1.0, 1.0, 1.0), 10)
    driver.run(num_steps=5)
    with pytest.raises(ValueError, match="Unknown field component"):
        driver.get_field_point("Ea", (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="must be finite"):
        driver.get_field_point("Ez", (0.0, float("nan"), 0.0))
    with pytest.raises(ValueError, match="must be finite"):
        driver.get_field_point("Ez", (float("inf"), 0.0, 0.0))
    # A point outside a PERIODIC cell is a lattice image and is answered, not refused.
    assert np.isfinite(driver.get_field_point("Ez", (5.3, 0.0, 0.0)))
    driver.close()

    # A point outside a MIRROR-FOLDED axis has no image at all, and says so.
    folded = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, symmetry=("X",))
    folded.set_epsilon(np.full(folded.shape, 1.0, dtype=np.float32))
    with pytest.raises(ValueError, match="mirror-folded"):
        folded.get_field_point("Ez", (1.4, 0.0, 0.0))
    folded.close()


def test_meep_time_is_the_simulation_clock():
    driver = _pulsed_driver((1.0, 1.0, 1.0), 10)
    assert driver.meep_time() == 0.0
    driver.run(num_steps=7)
    assert driver.meep_time() == pytest.approx(7 * driver.dt)
    assert driver.meep_time() == driver.time
    driver.close()


def test_step_functions_follow_meeps_protocol():
    """MEEP's ``_eval_step_func``: two-argument gets every todo, one-argument only 'step'.

    Ordering is MEEP's too — a step function sees time t, then the step to t + dt runs —
    so N steps produce N+1 samples, the first at the time ``run`` was entered.
    """
    driver = _pulsed_driver((1.0, 1.0, 1.0), 10)
    driver.run(num_steps=4)
    start = driver.meep_time()

    two_arg = []
    driver.run(num_steps=3, step_functions=[lambda sim, todo: two_arg.append((todo, sim.meep_time()))])
    assert [todo for todo, _ in two_arg] == ["step"] * 4 + ["finish"]
    assert [round(t, 9) for _, t in two_arg] == [
        round(start + index * driver.dt, 9) for index in (0, 1, 2, 3, 3)
    ]

    one_arg = []
    driver.run(num_steps=3, step_functions=[lambda sim: one_arg.append(sim.meep_time())])
    assert len(one_arg) == 4, "a one-argument step function must be called for 'step' only"

    # Several at once, each called once per step, in the order given.
    order = []
    driver.run(num_steps=2, step_functions=[
        lambda sim, todo: order.append(("a", todo)),
        lambda sim: order.append(("b", "step")),
    ])
    assert order[:4] == [("a", "step"), ("b", "step"), ("a", "step"), ("b", "step")]
    assert order[-1] == ("a", "finish")
    driver.close()


def test_step_functions_refuse_what_meep_refuses_and_skip_finish_on_a_raise():
    driver = _pulsed_driver((1.0, 1.0, 1.0), 10)
    with pytest.raises(ValueError, match="must be callable"):
        driver.run(num_steps=1, step_functions=[42])
    with pytest.raises(ValueError, match="f\\(sim\\) or f\\(sim, todo\\)"):
        driver.run(num_steps=1, step_functions=[lambda: None])
    with pytest.raises(ValueError, match="f\\(sim\\) or f\\(sim, todo\\)"):
        driver.run(num_steps=1, step_functions=[lambda a, b, c: None])

    # A run that raises has no 'finish': an analysis step function must keep its previous
    # result rather than publish one built from a truncated record.
    seen = []

    def explode(sim, todo):
        seen.append(todo)
        if len(seen) > 2:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        driver.run(num_steps=10, step_functions=[explode])
    assert "finish" not in seen
    driver.close()


def test_harminv_as_a_step_function_needs_no_driver_support_beyond_the_two_hooks():
    """``Harminv`` is duck-typed on ``meep_time`` and ``get_field_point``, and says which is missing."""
    from .harminv import Harminv

    driver = _pulsed_driver((1.0, 1.0, 1.0), 10)
    probe = Harminv(c="Ez", pt=(0.13, 0.07, 0.02), fcen=0.35, df=0.3)
    driver.run(num_steps=60, step_functions=[probe])
    assert len(probe.data) == 61
    assert probe.data_dt == pytest.approx(driver.dt, rel=1e-9)
    assert all(isinstance(value, complex) for value in probe.data)

    class Bare:
        def meep_time(self):
            return 0.0

    with pytest.raises(ValueError, match="get_field_point"):
        Harminv(c="Ez", pt=(0, 0, 0), fcen=0.35, df=0.3)(Bare(), "step")
    driver.close()


# --------------------------------------------------------------------------------
# The instantaneous nonlinearity, checked against the published formula rather than
# against the implementation that transcribed it.
# --------------------------------------------------------------------------------


def _pade_factor(dsqr, di, chi1inv, chi2, chi3):
    """MEEP ``calc_nonlinear_u`` (step_generic.cpp:542-548), retyped from the C.

    Deliberately NOT imported from :mod:`.stepping`. A transcription error that landed
    in the engine would be invisible to a test that reuses the engine's own copy.
    """
    c2 = di * chi2 * (chi1inv * chi1inv)
    c3 = dsqr * chi3 * (chi1inv * chi1inv * chi1inv)
    return (1 + c2 + 2 * c3) / (1 + 2 * c2 + 3 * c3)


@pytest.mark.parametrize("permittivity", [1.0, 4.0, 11.7])
@pytest.mark.parametrize("chi2, chi3", [(0.0, 0.05), (0.0, -0.05), (0.03, 0.0), (0.03, 0.05)])
def test_the_constitutive_law_is_the_published_pade_factor_to_the_last_bit(permittivity, chi2, chi3):
    """D in, E out, through the whole driver, against the formula written out by hand.

    A uniform D is installed in all three components, so every one of MEEP's four-point
    transverse sums evaluates to ``4*D0`` and ``Dsqr = D0^2 + 0.0625*(16*D0^2 + 16*D0^2)``
    in closed form — which means the 0.0625 weight, the four-point stencil AND the Pade
    expression are all pinned by one number, with no interpolation left to argue about.
    A single step then recovers E from that D (the curl of a uniform field is zero, so
    D does not move) and the readback is compared against the hand-typed factor.

    Measured worst relative deviation over 48 combinations of permittivity, sign,
    coefficient and amplitude: 0.0e+00 — bit-exact, not merely close.
    """
    worst = 0.0
    for displacement in (0.05, 0.3, 1.0, 2.0):
        driver = FdtdDriver(cell_size=(0.6, 0.6, 0.6), resolution=10, force_complex_fields=False)
        driver.set_epsilon(np.full(driver.shape, permittivity, dtype=np.float32))
        if chi2:
            driver.set_chi2(chi2)
        if chi3:
            driver.set_chi3(chi3)
        for name in ("Dx", "Dy", "Dz"):
            driver.set_field(name, np.full(driver.shape, displacement, dtype=np.float32))
        driver.run(num_steps=1)
        produced = float(np.asarray(driver.get_field("Ez", cell_centered=False))[3, 3, 3])
        inverse = 1.0 / np.float32(permittivity)
        dsqr = displacement ** 2 + 0.0625 * (2 * (4 * displacement) ** 2)
        expected = displacement * inverse * _pade_factor(dsqr, displacement, inverse, chi2, chi3)
        worst = max(worst, abs(produced - expected) / abs(expected))
        driver.close()
    assert worst == 0.0, (
        f"the driver's constitutive law deviates from the published Pade factor by {worst:.3e} "
        f"at eps={permittivity}, chi2={chi2}, chi3={chi3}"
    )


def _hard_nonlinear_digest(install_zero_nonlinearity, chi3=0.0):
    """PML, Lorentz dispersion, D conductivity and a diagonal epsilon, all at once."""
    driver = FdtdDriver(cell_size=(1.1, 0.9, 1.7), resolution=10, force_complex_fields=True)
    driver.set_epsilon_components({
        "Ex": np.full(driver.shape, 2.0, dtype=np.float32),
        "Ey": np.full(driver.shape, 3.5, dtype=np.float32),
        "Ez": np.full(driver.shape, 5.0, dtype=np.float32),
    })
    driver.setup_pml({"z": 5})
    driver.set_conductivity(0.02)
    driver.add_susceptibility(Susceptibility(kind=LORENTZIAN, frequency=0.6, gamma=0.05), 0.3)
    if install_zero_nonlinearity:
        driver.set_chi2(np.zeros(driver.shape, dtype=np.float32))
        driver.set_chi3(0.0)
    if chi3:
        driver.set_chi3(chi3)
    driver.add_source({
        "source_type": "gaussian", "component": "Ey", "frequency": 0.6, "fwidth": 0.4,
        "center": (0.05, -0.05, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": 1.0,
    })
    driver.run(num_steps=60)
    blob = b"".join(
        np.ascontiguousarray(driver.get_field(name, cell_centered=False)).tobytes()
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Dx", "Dy", "Dz", "Bx", "By", "Bz")
    )
    peak = max(float(np.abs(driver.get_field(name)).max()) for name in ("Ex", "Ey", "Ez"))
    driver.close()
    return hashlib.sha256(blob).hexdigest(), peak


def test_a_zero_nonlinearity_is_byte_identical_with_pml_dispersion_and_diagonal_epsilon():
    """The byte pin on the HARDEST combination, because that is where the floors live.

    Every recorded accuracy floor in this package was measured with no chi2/chi3
    installed. If installing an identically zero pair moved a single bit of a run that
    also carries a PML, a Lorentz pole, a D conductivity and a per-component epsilon,
    each of those floors would silently be describing a different arithmetic path than
    the one users get. The digest is a sha256 over the raw bytes of all twelve field
    volumes, not a comparison to a tolerance.
    """
    bare, peak = _hard_nonlinear_digest(False)
    zeroed, _ = _hard_nonlinear_digest(True)
    live, _ = _hard_nonlinear_digest(False, chi3=2e-3)
    assert peak > 0.0, "the control run carries no field, so the digest compares nothing"
    assert bare == zeroed, (
        "installing an identically zero chi2/chi3 changed the bytes of a PML + dispersive + "
        "conductive + diagonal-epsilon run"
    )
    assert bare != live, (
        "a live chi3 = 2e-3 produced the same bytes as no nonlinearity, so this digest cannot "
        "tell the two apart and the identity above is vacuous"
    )


_FRESH_PROCESS_DIGEST = (
    "from meep_gpu.test_driver_integration import _hard_nonlinear_digest as digest;"
    "value, peak = digest(False);"
    "print(value, peak)"
)


def _digest_in_a_fresh_process(hash_seed: str) -> tuple[str, float]:
    """``_hard_nonlinear_digest(False)`` again, in a brand-new interpreter."""
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environment["PYTHONHASHSEED"] = hash_seed
    completed = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_DIGEST],
        capture_output=True, text=True, env=environment, timeout=600,
    )
    assert completed.returncode == 0, (
        f"the fresh-process digest child failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    value, peak = completed.stdout.strip().split()[-2:]
    return value, float(peak)


def test_the_stepper_is_bit_reproducible_across_fresh_processes():
    """The same input must give the same BYTES in a different interpreter, twice over.

    The in-process byte pin above cannot see a run whose arithmetic depends on
    process state — heap layout, a warmed thread pool, a randomized hash seed — and
    that is a real failure mode measured elsewhere in this stack: MEEP's own
    ``fields::get_eigenmode`` returns a different eigenvector on repeated identical
    calls, which makes ``coupler.py``'s corpus parity a spread rather than a digit
    (``the design notes (meep-corpus-inventory)``). That defect is upstream, and this
    test is the guard that says so — if this engine ever acquires the same property,
    every floor recorded in this package would be a number about one particular run.

    Three digests over the hardest configuration (PML + Lorentz pole + D
    conductivity + diagonal epsilon + a pulsed source): this process, and two fresh
    interpreters started under DIFFERENT ``PYTHONHASHSEED`` values so that any
    dependence on set/dict iteration order shows up as a mismatch rather than
    hiding behind a seed that happened to agree.
    """
    in_process, peak = _hard_nonlinear_digest(False)
    assert peak > 0.0, "the run carries no field, so the digests compare nothing"
    first, first_peak = _digest_in_a_fresh_process("0")
    second, second_peak = _digest_in_a_fresh_process("987654321")
    assert (first_peak, second_peak) == (peak, peak), (
        f"the fresh processes did not even reproduce the peak field: {peak} vs {first_peak} "
        f"and {second_peak}"
    )
    assert in_process == first == second, (
        "the driver's twelve field volumes are not bit-reproducible across processes: "
        f"this process {in_process}, PYTHONHASHSEED=0 {first}, PYTHONHASHSEED=987654321 "
        f"{second}. Identical input must give identical bytes; anything else makes every "
        "accuracy floor in this package a statement about one lucky run."
    )


@pytest.mark.parametrize(
    "chi3, amplitude, refused",
    [(2e-3, 1.0, False), (5.0, 1.0, False), (5e3, 1.0, False), (5e5, 1.0, True),
     (5e3, 100.0, True), (5e5, 100.0, True)],
)
def test_the_pade_guard_tracks_the_field_and_not_just_the_coefficient(chi3, amplitude, refused):
    """The expansion parameter is quadratic in the drive and linear in chi3, and the guard follows.

    The point of the guard is that the SAME chi3 is fine at one source amplitude and
    past the pole at a hundred times it, so a check on the material alone could not
    work. Measured expansion ``max(|c2| + |c3|)`` on this cell: 4.3e-08 at chi3 = 2e-3,
    1.1e-04 at 5, 1.0e-01 at 5e3 — a hundredfold in chi3 for a hundredfold in the
    parameter — then 0.74 at 5e5 (refused), and at a 100x amplitude the same 5e3 that
    passed reaches 1.98 and is refused.
    """
    driver = FdtdDriver(cell_size=(1.1, 0.9, 1.7), resolution=10, force_complex_fields=True)
    driver.set_epsilon_components({
        "Ex": np.full(driver.shape, 2.0, dtype=np.float32),
        "Ey": np.full(driver.shape, 3.5, dtype=np.float32),
        "Ez": np.full(driver.shape, 5.0, dtype=np.float32),
    })
    driver.setup_pml({"z": 5})
    driver.set_chi3(chi3)
    driver.add_source({
        "source_type": "gaussian", "component": "Ey", "frequency": 0.6, "fwidth": 0.4,
        "center": (0.05, -0.05, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": amplitude,
    })
    if refused:
        with pytest.raises(FdtdNonlinearityOutOfRange, match="Pade approximant"):
            driver.run(num_steps=60)
    else:
        driver.run(num_steps=60)
        margin = driver.nonlinear_margin()
        assert margin.expansion < 1.0 / 3.0
        assert margin.denominator > 0.0, "the approximant's denominator crossed its pole unnoticed"
    driver.close()


# ---------------------------------------------------------------------------
# Mirror planes reachable from the driver: any axis, either declared phase.
# ---------------------------------------------------------------------------



def _mirror_plane_sets():
    """Every non-empty set of mirror planes, each axis at each declared phase.

    Three axes at two phases each, over all one-, two- and three-plane
    combinations: 6 + 12 + 8 = 26 folds. The engine had 3 of them reachable
    (X, Y, XY, all at phase +1).
    """
    import itertools

    sets = []
    for size in (1, 2, 3):
        for axes in itertools.combinations(range(3), size):
            for phases in itertools.product((+1, -1), repeat=size):
                sets.append(tuple(zip(axes, phases)))
    return sets


def _plane_label(planes):  # "X+1,Z-1" — readable in a parametrised test id.
    return ",".join(f"{'XYZ'[axis]}{phase:+d}" for axis, phase in planes)


def _compact_profile_along(positions, dx):  # Narrow Gaussian truncated to a few cells.
    values = np.exp(-((positions / (2.0 * dx)) ** 2))
    return np.where(np.abs(positions) <= 5.0 * dx, values, 0.0)


def _parity_respecting_seed(component, shape_full, dx, planes):
    """A compact blob carrying the parity each declared plane forces on ``component``.

    The seed has to obey the symmetry or the comparison is vacuous — a folded run
    can only reproduce a full-domain run whose state is in the folded subspace.
    The parity comes from ``fields.mirror_parity`` at the plane's own phase, so an
    odd plane is seeded with an ODD profile (and exactly zero on the plane, which is
    where an odd field has to vanish), and the same builder serves all 26 folds.

    Sample positions per axis follow the component's Yee shift: an integer-shifted
    axis samples ``(i - n/2)*dx``, so index i and index n-i are mirror partners and
    index n/2 lands ON the plane; a half-shifted axis samples ``(i + 0.5 - n/2)*dx``,
    where i and n-1-i are partners and nothing lands on the plane.
    """
    from .fields import IYEE_SHIFTS, mirror_parity

    shifts = IYEE_SHIFTS[component]
    phases = dict(planes)
    axes = []
    for axis in range(3):
        count = shape_full[axis]
        positions = (np.arange(count) + 0.5 * shifts[axis] - count / 2) * dx
        values = _compact_profile_along(positions, dx)
        phase = phases.get(axis)
        if phase is not None and mirror_parity(component, axis, phase) < 0:
            values = values * np.sign(positions)  # Odd about the plane; zero on it.
        axes.append(values)
    return (axes[0][:, None, None] * axes[1][None, :, None]
            * axes[2][None, None, :]).astype(np.complex64)


def _quadrant_of(volume, planes, shape_full):  # The half/quarter/eighth a fold stores.
    """Slice the stored layout of a PERIODIC fold out of a full-domain array.

    Quadrant row q is full cell ``q + n_full//2 - 1``; at an even full count the
    stored array also carries the second-mirror plane cell, which on the full
    periodic domain is cell 0 one lattice vector up (no phase at k = 0) — the
    same append ``test_stepping._fold`` makes.
    """
    for axis, _phase in planes:
        sliced = volume[(slice(None),) * axis + (slice(shape_full[axis] // 2 - 1, None),)]
        if shape_full[axis] % 2 == 0:
            sliced = np.concatenate(
                (sliced, volume[(slice(None),) * axis + (slice(0, 1),)]), axis=axis)
        volume = sliced
    return volume


@pytest.mark.parametrize("planes", _mirror_plane_sets(), ids=_plane_label)
@pytest.mark.parametrize("counts", ["even", "odd-unfolded"])
def test_a_fold_on_any_axis_at_either_phase_reproduces_the_full_domain_exactly(planes, counts):
    """THE PLUMBING TEST: a folded run must BE the full-domain run, to the last bit.

    The fold kernel was already pinned bit-for-bit for every axis and both phases
    (``test_fields.py`` reconstruction cases, ``test_stepping.py`` ghost cells), but
    nothing above ``Grid`` could ask for an odd plane or a Z one:
    ``FdtdDriver(symmetry=...)`` took bare axis names, installed phase +1, and raised
    "Z mirrors are not implemented". This drives all 26 folds through the DRIVER, which
    is the layer the descriptor had to be threaded through, and demands the same
    exactly-0.0 standard the kernel meets — not a tolerance.

    Exactly 0.0 is reachable because the two runs are then the same arithmetic in the
    same order on the same values: the seed is compact enough that neither run's outer
    faces are ever reached (the folded far face is a zero ghost where the full run
    wraps, so a live face would make them different systems — see
    ``FdtdDriver._require_folded_far_face_is_quiet``), and the state is seeded rather
    than driven, because a source accumulates into the two grids in a different order
    and lands at ~2e-6 instead.

    ``counts`` sweeps an ODD cell count on the unfolded axes. A folded axis cannot
    itself be odd — ``halve()`` needs cell 0 to straddle the plane at -0.5*dx, and
    ``Grid`` refuses an odd count there — so the odd-count case puts 41 and 43 cells
    on whichever axes are NOT folded, which is where an off-by-half-a-cell
    registration error would surface.
    """
    from .grid import Mirror

    lengths = []
    for axis in range(3):
        folded = any(plane_axis == axis for plane_axis, _ in planes)
        lengths.append(4.0 if folded or counts == "even" else (4.1 if axis == 0 else 4.3))
    cell = tuple(lengths)
    symmetry = tuple(Mirror("XYZ"[axis], phase) for axis, phase in planes)

    full = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True)
    folded = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True,
                        symmetry=symmetry)
    seed = _parity_respecting_seed("Dz", full.shape, full.dx, planes)
    quadrant = _quadrant_of(seed, planes, full.shape)
    assert quadrant.shape == folded.shape, (
        f"{_plane_label(planes)}: the fold stores {folded.shape} but the quadrant of the "
        f"seed is {quadrant.shape}; the halving and the slice disagree"
    )
    full.set_field("Dz", seed)
    folded.set_field("Dz", quadrant)
    for _ in range(12):
        full.step()
        folded.step()

    scale = max(float(np.max(np.abs(full.get_field(name, cell_centered=False))))
                for name in _PRIMARY_COMPONENTS)
    assert scale > 0.0, f"{_plane_label(planes)}: the run carries no field; the test is vacuous"
    for name in _PRIMARY_COMPONENTS:
        reference = _quadrant_of(full.get_field(name, cell_centered=False), planes, full.shape)
        difference = float(np.max(np.abs(folded.get_field(name, cell_centered=False) - reference)))
        assert difference == 0.0, (
            f"{_plane_label(planes)} ({counts} counts): folded {name} differs from the "
            f"full-domain run by {difference:.3e} ({difference / scale:.3e} of peak); the "
            f"fold kernel meets exactly 0.0, so any residual is the plumbing above it"
        )


@pytest.mark.parametrize("axis,phase", [(2, +1), (2, -1), (0, -1)])
def test_a_fold_survives_pml_dispersion_and_an_off_centre_source(axis, phase):
    """The folds that were unreachable, exercised with everything else switched on.

    A Z plane and an odd plane are new to the driver, and the features they have to
    coexist with are exactly the ones whose per-axis tables used to enumerate X and Y:
    the absorber (``pml._resolve_mirror_faces`` skipped the low face on X and Y only,
    so a folded Z was graded from BOTH walls and absorbed across its own mirror plane),
    the monitor wrap flags, and the source registration. Each leg here is the folded
    run against the equivalent full-domain run.

    The absorber is asked for with the SCALAR shorthand, which means "every face that
    can carry a layer": on the folded run that is five faces, because the mirror plane
    cannot hold one, and on the full-domain run it is six. That is the pairing the two
    runs need — the folded axis absorbs on its stored far face and its mirrored half
    absorbs by reflection — and it is also the request that exercises
    ``_resolve_mirror_faces`` skipping the right face on a Z fold.

    The source is deliberately OFF CENTRE on an unfolded axis, at no round multiple of
    dx. It must stay on every active mirror plane — that is the one thing a fold
    requires of it — but nothing ties it to the origin of the axes that are NOT folded,
    and a registration that quietly assumed a centred source would pass a centred test.
    """
    from .fields import mirror_parity
    from .grid import Mirror

    name = "XYZ"[axis]
    cell = [3.0, 3.0, 3.0]
    cell[axis] = 4.0
    cell = tuple(cell)
    unfolded_axis = 0 if axis != 0 else 1
    centre = [0.0, 0.0, 0.0]
    centre[unfolded_axis] = 0.35  # Off centre, and at no round multiple of dx.
    component = next(c for c in ("Ex", "Ey", "Ez") if mirror_parity(c, axis, phase) > 0)
    source = {"component": component, "frequency": 1.0, "center": tuple(centre),
              "size": (0.0, 0.0, 0.0), "source_type": "gaussian", "fwidth": 0.5}

    def build(symmetry, leg):
        driver = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True,
                            symmetry=symmetry)
        if leg == "pml":
            driver.setup_pml(4)  # Every face that can carry one; the plane cannot.
        if leg == "dispersion":
            driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
            driver.add_susceptibility(
                Susceptibility(frequency=1.1, gamma=0.05, kind=LORENTZIAN),
                np.full(driver.shape, 0.3, dtype=np.float32))
        driver.add_source(source)
        driver.run(num_steps=14)
        return driver

    for leg in ("plain", "pml", "dispersion"):
        folded = build((Mirror(name, phase),), leg)
        full = build((), leg)
        scale = max(float(np.max(np.abs(full.get_field(field, cell_centered=False))))
                    for field in _PRIMARY_COMPONENTS)
        assert scale > 0.0, f"{name}{phase:+d}/{leg}: the run carries no field"
        worst = 0.0
        for field in _PRIMARY_COMPONENTS:
            reference = _quadrant_of(full.get_field(field, cell_centered=False),
                                     ((axis, phase),), full.shape)
            worst = max(worst, float(np.max(np.abs(
                folded.get_field(field, cell_centered=False) - reference))))
        # A driven run accumulates the injected current into the two grids in a
        # different order, so this is float32 round-off rather than the seeded
        # case's exact zero; a fold that is not wired through reads ~1e0.
        assert worst / scale < 1e-5, (
            f"Mirror({name!r}, {phase:+d}) with {leg} diverges from the full-domain run by "
            f"{worst / scale:.3e}"
        )


@pytest.mark.parametrize("axis,phase", [(0, +1), (0, -1), (1, -1), (2, +1), (2, -1)])
def test_a_fold_carries_a_bloch_phase_on_another_axis_and_refuses_one_on_its_own(axis, phase):
    """The two halves of the mirror + Bloch question, on every axis and both phases.

    A nonzero k on the plane's OWN axis stays refused: inside the Brillouin zone a
    mirror forces the field to repeat with no phase, which only k = 0 satisfies. (The
    zone EDGE is compatible — MEEP #3155 — and is refused here too, for the reason
    ``_require_bloch_is_representable`` gives; that is a narrower gap than this test
    covers.)

    A mirror on one axis with a Bloch phase on ANOTHER is SERVED, and this measures
    it. It used to raise "is not implemented" on the claim that the folded-axis
    readback averaged across the mirror plane with no Bloch factor. That claim is
    false of this tree and was false when it was written: the readback branches PER
    AXIS (``Fields.to_cell_center``), a mirrored axis takes the fold branch and needs
    no Bloch factor precisely because its own k is pinned to zero by the refusal
    above, and every unfolded axis takes ``Fields._wrapped_neighbour``, which applies
    ``grid.bloch_phase``. MEEP composes the two the same way and with no cross term —
    the lattice phase accumulates over the shift loop (loop_in_chunks.cpp:415) and
    meets ``symmetry::phase_shift`` (vec.cpp:1347) only as a product at :202.

    The measurement is the folded quadrant against the equivalent full-domain run at
    the same 1e-5 relative bar the neighbouring fold cases use, with a k = 0 folded
    run as the negative control — that run is what an engine which accepted the
    ``k_point`` and dropped it would produce, and it must land far outside the bar or
    the comparison cannot tell whether the phase was carried at all.
    """
    from .fields import mirror_parity
    from .grid import Mirror

    plane = Mirror("XYZ"[axis], phase)
    own_axis = [0.0, 0.0, 0.0]
    own_axis[axis] = 0.25
    with pytest.raises(ValueError, match="mirror-symmetric axis"):
        FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, symmetry=(plane,),
                   k_point=tuple(own_axis))

    bloch_axis = (axis + 1) % 3
    wavevector = [0.0, 0.0, 0.0]
    # Not a simple fraction of the cell, so no missing 2*pi and no sign slip can
    # cancel by luck, and far from the zone edge, whose phase is its own conjugate.
    wavevector[bloch_axis] = 0.1234567
    cell = [3.0, 3.0, 3.0]
    cell[axis] = 4.0
    cell = tuple(cell)
    component = next(c for c in ("Ex", "Ey", "Ez") if mirror_parity(c, axis, phase) > 0)
    source = {"component": component, "frequency": 1.0, "center": (0.0, 0.0, 0.0),
              "size": (0.0, 0.0, 0.0), "source_type": "gaussian", "fwidth": 0.5}

    def build(symmetry, k_point):
        driver = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True,
                            symmetry=symmetry, k_point=k_point)
        driver.add_source(source)
        # Long enough for the front to reach the wrap plane and come back round. This
        # is not padding: at 14 steps the boundary is never touched, the k = 0 control
        # below reads exactly 0.0, and the comparison measures nothing about the phase
        # at all. Measured control at t = 3 / 6 / 10: 6.7e-04 / 2.6e-03 / 2.1e-02 on
        # the even planes and 2.4e-03 / 7.1e-03 / 3.8e-02 on the odd ones.
        driver.run(until=6.0)
        return driver

    full = build((), tuple(wavevector))
    folded = build((plane,), tuple(wavevector))
    blind = build((plane,), (0.0, 0.0, 0.0))

    scale = max(float(np.max(np.abs(full.get_field(field, cell_centered=False))))
                for field in _PRIMARY_COMPONENTS)
    assert scale > 0.0, f"Mirror({'XYZ'[axis]!r}, {phase:+d}): the run carries no field"
    worst = 0.0
    control = 0.0
    for field in _PRIMARY_COMPONENTS:
        reference = _quadrant_of(full.get_field(field, cell_centered=False),
                                 ((axis, phase),), full.shape)
        worst = max(worst, float(np.max(np.abs(
            folded.get_field(field, cell_centered=False) - reference))))
        control = max(control, float(np.max(np.abs(
            blind.get_field(field, cell_centered=False) - reference))))
    # The bar is the neighbouring fold cases' 1e-5, which allows for a driven run
    # accumulating the injected current into the two grids in a different order. All
    # five pairs in fact measure exactly 0.0 here.
    assert worst / scale < 1e-5, (
        f"Mirror({'XYZ'[axis]!r}, {phase:+d}) with k on {'XYZ'[bloch_axis]} diverges from the "
        f"full-domain run by {worst / scale:.3e}"
    )
    assert control / scale > 1e-3, (
        f"Mirror({'XYZ'[axis]!r}, {phase:+d}): a folded run at k = 0 reproduces the k != 0 "
        f"reference to {control / scale:.3e}, so this comparison cannot see whether the Bloch "
        f"phase was carried through the fold at all."
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("phase", [+1, -1])
def test_an_odd_profile_sheet_folds_exactly_and_an_even_one_silently_does_not(axis, phase):
    """The behaviour behind the narrowed source-parity rule, and the hole it leaves.

    A sheet spanning the folded axis and driving a component the plane makes ODD is
    now accepted, because MEEP's deposition is a CLIP (``use_symmetry=false``,
    sources.cpp:487) and the folded run reproduces the declared run exactly when the
    declared current already carries the plane's parity. That condition is a property
    of the PROFILE, and this measures both sides of it through the driver:

    * ``sin(pi u / half)`` is odd about the plane — the folded run must BE the
      full-domain run. In stock MEEP the same comparison reads 3.7e-07 / 4.05e-07.
    * ``cos(pi u / (2 half))`` is even — the fold keeps the owned half and images it
      with the opposite sign, so the current it really drives is the odd extension of
      half the sheet. That run does not fail, does not warn, and returns a smooth
      plausible field. Stock MEEP answers the uniform version of it 57% (even phase)
      to 283% (odd phase) away from the unfolded run.

    THE SECOND CASE IS THE NEGATIVE CONTROL AND IT IS ALSO A KNOWN GAP: the rule
    refuses a uniform profile (provably even, no ``amp_func`` at all) and a zero-extent
    source (the parity condition then admits only zero current), but it does not
    evaluate a supplied ``amp_func``'s parity. It cannot decide that per source — on
    the five corpus rows this unlocked, the wrong-parity sheets are numerical dust
    from an ``eig_parity``-unset eigenmode expansion (profile residual 1.93/1.45 at
    9.3e-09 and 5.2e-09 of the declaration's current) while every sheet carrying real
    current is odd to 1.6e-15, so a residual test with no cross-sheet weight refuses
    all five. This test exists so the gap is a measured number in the suite rather
    than a sentence in a docstring.
    """
    from .fields import mirror_parity
    from .grid import Mirror

    component = next(c for c in ("Ex", "Ey", "Ez") if mirror_parity(c, axis, phase) < 0)
    cell = [2.0, 2.0, 2.0]
    cell[axis] = 4.0
    half = 1.5  # inside the 4.0 axis, clear of the wrap plane
    size = [0.0, 0.0, 0.0]
    size[axis] = 2 * half

    def odd_profile(dx, dy, dz):
        return complex(np.sin(np.pi * (dx, dy, dz)[axis] / half))

    def even_profile(dx, dy, dz):
        return complex(np.cos(np.pi * (dx, dy, dz)[axis] / (2 * half)))

    def build(symmetry, amp_func):
        driver = FdtdDriver(cell_size=tuple(cell), resolution=10, force_complex_fields=True,
                            symmetry=symmetry)
        driver.add_source({"component": component, "frequency": 1.0,
                           "center": (0.0, 0.0, 0.0), "size": tuple(size),
                           "source_type": "continuous", "amp_func": amp_func})
        driver.run(num_steps=40)  # CW, hard switch-on: O(1) field, long enough for the
        #                          even-profile control to separate (it grows with t).
        return driver

    plane = Mirror("XYZ"[axis], phase)
    for label, amp_func, expectation in (("odd", odd_profile, "folds"),
                                         ("even", even_profile, "diverges")):
        full = build((), amp_func)
        folded = build((plane,), amp_func)
        scale = max(float(np.max(np.abs(full.get_field(f, cell_centered=False))))
                    for f in _PRIMARY_COMPONENTS)
        assert scale > 0.0, f"{label}: the full-domain run carries no field"
        worst = 0.0
        for name in _PRIMARY_COMPONENTS:
            reference = _quadrant_of(full.get_field(name, cell_centered=False),
                                     ((axis, phase),), full.shape)
            worst = max(worst, float(np.max(np.abs(
                folded.get_field(name, cell_centered=False) - reference))))
        if expectation == "folds":
            assert worst / scale < 1e-5, (
                f"Mirror({'XYZ'[axis]!r}, {phase:+d}): a {component} sheet with an ODD profile "
                f"must fold exactly, measured {worst / scale:.3e}"
            )
        else:
            assert worst / scale > 1e-2, (
                f"Mirror({'XYZ'[axis]!r}, {phase:+d}): a {component} sheet with an EVEN profile "
                f"reproduces the full-domain run to {worst / scale:.3e}, so this case cannot "
                f"show what the rule does not check and the odd-profile pass above proves nothing"
            )


def test_a_z_fold_no_longer_skips_its_source_parity_check_and_its_dft_unfold():
    """THE SILENT SKIP. A Z fold used to bypass both, and neither would have said so.

    ``sources._validate_symmetry`` / ``_validate_symmetry_parity`` and every monitor
    table in ``dft.py`` read the fold from a hard-coded ``(sym_x, sym_y, False)``
    triple. On a Z fold that ``False`` is not a refusal — it is a claim that the axis
    is not folded, and it made three checks pass by not running:

      1. the source-centre rule ("a source must sit on the mirror plane") accepted a
         source anywhere along z, and folded it about a plane it was not on;
      2. the source-PARITY rule accepted a current the plane makes odd, which the
         fold then reproduces WRONG — not by cancelling it (MEEP deposits with
         use_symmetry=false, sources.cpp:487, so no J ever meets a -J) but by
         keeping only the owned half and imaging the rest with the opposite sign,
         which is the declared run only if the declared current already has that
         parity. For the Ez point below stock MEEP returns a smooth, finished
         field 57%-283% from its own unfolded answer;
      3. ``get_dft_full`` returned the STORED QUADRANT and called it the full domain —
         right dtype, plausible values, half the cells and the wrong coordinates.

    Each is asserted here to happen now, and the pre-fix answer is reconstructed
    in-process so the cost of the skip is a measured number rather than a claim.
    """
    from .grid import Mirror

    cell = (2.0, 2.0, 4.0)
    on_the_plane = {"component": "Ex", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                    "size": (0.0, 0.0, 0.0), "source_type": "gaussian", "fwidth": 0.5}

    # 1. A source in the half the Z fold DISCARDS is refused. The old triple said z
    #    was not folded, so this ran and folded the source about z = 0 regardless.
    #    (The rule is the stored half, not the plane: MEEP images an off-plane source
    #    implicitly and this engine reproduces it at the fold floor — see
    #    `sources._validate_symmetry`. What MEEP does NOT do is deposit anything in
    #    the discarded half, where it returns a peak of exactly 0.)
    driver = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True,
                        symmetry=("Z",))
    driver.add_source({**on_the_plane, "component": "Ex", "center": (0.0, 0.0, 0.5)})
    with pytest.raises(ValueError, match="With Z symmetry the stored half"):
        driver.add_source({**on_the_plane, "center": (0.0, 0.0, -0.5)})

    # 2. A current the Z plane makes odd is refused. BOTH HALVES OF THE PHASE TABLE are
    #    exercised, because the two are refused by different mechanisms and the wording
    #    that stood here ("its own mirror image") is true of only one of them.
    #
    #    Which components a plane makes odd is exactly which ones it half-shifts, at
    #    phase +1, and exactly the complement at phase -1 (measured off `fields.
    #    IYEE_SHIFTS` against `mirror_parity`, and asserted below rather than asserted
    #    of). So:
    #
    #      phase=+1, Ez: Ez IS half-shifted on z, so it has NO sample site on z = 0 and
    #        cannot be its own image. What goes wrong is the clip: half the deposit
    #        lands in the discarded half and is dropped, and the fold images the rest
    #        with the opposite sign. Stock MEEP runs it and returns a smooth field
    #        57%-283% from its own unfolded answer.
    #      phase=-1, Ex: Ex is NOT half-shifted on z, so its site sits ON the plane, is
    #        genuinely its own image, and parity -1 forces J = -J, i.e. J = 0.
    #
    #    Only the second is the self-cancellation the old docstring described, and the
    #    pinned case was the first — which is why it is spelled out here.
    from .fields import IYEE_SHIFTS, mirror_parity

    assert mirror_parity("Ez", 2, +1) < 0 and IYEE_SHIFTS["Ez"][2] != 0, (
        "the even Z plane must make Ez odd AND half-shift it on z — no site on the plane"
    )
    assert mirror_parity("Ex", 2, -1) < 0 and IYEE_SHIFTS["Ex"][2] == 0, (
        "the odd Z plane must make Ex odd while leaving it unshifted on z — a real self-image"
    )
    with pytest.raises(ValueError, match=r"Ez source centred on the mirror plane cannot carry"):
        driver.add_source({**on_the_plane, "component": "Ez"})
    odd_plane = FdtdDriver(cell_size=cell, resolution=10, force_complex_fields=True,
                           symmetry=(Mirror("Z", -1),))
    with pytest.raises(ValueError, match=r"Ex source centred on the mirror plane cannot carry"):
        odd_plane.add_source({**on_the_plane, "component": "Ex"})
    # The complementary plane accepts exactly the current the other refuses, which is
    # what says the rule is the parity and not a blanket ban on the axis.
    odd_plane.add_source({**on_the_plane, "component": "Ez"})
    # ...and the even plane accepts the Ex the odd one just refused — step 3 below adds
    # exactly that source to `driver` and runs it.

    # 3. get_dft_full really unfolds the z axis now. Build the run, then measure what
    #    the skipped path would have returned. The region starts AT the plane (a
    #    region crossing it registers the full span through the reflected gather
    #    and has no quadrant to unfold; get_dft_full says so).
    driver.add_source(on_the_plane)
    monitor = driver.add_dft_monitor(1.0, components=("Ex", "Ey", "Ez"),
                                     center=(0.0, 0.0, 0.9), size=(1.2, 1.2, 1.8))
    driver.run(num_steps=14)

    quadrant = np.asarray(monitor.get_dft("Ex"))
    unfolded = np.asarray(monitor.get_dft_full("Ex"))
    assert unfolded.shape[2] == 2 * (quadrant.shape[2] - 1), (
        f"a Z fold must unfold its z axis: quadrant {quadrant.shape} -> {unfolded.shape}"
    )
    assert unfolded.shape[:2] == quadrant.shape[:2], "the unfolded axes must be untouched"
    coordinates = monitor.get_coordinates_full()
    assert len(coordinates[2]) == unfolded.shape[2], (
        "the z coordinates must pair one-to-one with the unfolded data"
    )
    assert coordinates[2][0] < 0.0 < coordinates[2][-1], (
        f"an unfolded z axis must span the plane, got {coordinates[2][0]:.3f}..{coordinates[2][-1]:.3f}"
    )

    # What the skip returned: the quadrant itself, under a full-domain name. It is not
    # merely short — the half it omits is the half that carries the mirrored field.
    from .fields import mirror_parity

    parity = mirror_parity("Ex", 2, 1)
    expected_far_half = parity * np.flip(quadrant[:, :, 1:], axis=2)
    centre = unfolded.shape[2] // 2
    np.testing.assert_array_equal(unfolded[:, :, :centre], expected_far_half)
    scale = float(np.abs(unfolded).max())
    assert scale > 0.0, "the monitor accumulated nothing; the unfolding check is vacuous"
    missing = float(np.abs(unfolded[:, :, :centre]).max()) / scale
    assert missing > 0.1, (
        f"the half a skipped unfold would have dropped carries only {missing:.3e} of the "
        f"peak; this run cannot demonstrate the cost of the skip"
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("phase", [+1, -1])
def test_dft_unfolding_takes_the_planes_own_phase_on_every_axis(axis, phase):
    """The unfolded DFT must carry ``mirror_parity`` at the plane's declared phase.

    Reading the even X/Y table instead leaves every even run correct, says nothing
    about Z, and mirrors an odd run with the WRONG SIGN on half its cells — a full,
    smooth, correctly shaped full-domain array that is the negative of the truth
    wherever the plane reflected it. Nothing about the array's shape, dtype or
    magnitude reports that, which is why the sign is asserted cell by cell against
    the one derivation rather than against a recorded volume.
    """
    from .fields import mirror_parity
    from .grid import Mirror

    cell = [2.0, 2.0, 2.0]
    cell[axis] = 4.0
    component = next(c for c in ("Ex", "Ey", "Ez") if mirror_parity(c, axis, phase) > 0)
    driver = FdtdDriver(cell_size=tuple(cell), resolution=10, force_complex_fields=True,
                        symmetry=(Mirror("XYZ"[axis], phase),))
    driver.add_source({"component": component, "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                       "size": (0.0, 0.0, 0.0), "source_type": "gaussian", "fwidth": 0.5})
    size = [1.2, 1.2, 1.2]
    size[axis] = 1.8  # Interior: the far-plane cell of a folded periodic even axis has no image.
    centre_at_plane = [0.0, 0.0, 0.0]
    centre_at_plane[axis] = 0.9  # Low face exactly at the plane; get_dft_full unfolds.
    monitor = driver.add_dft_monitor(1.0, components=("Ex", "Ey", "Ez"),
                                     center=tuple(centre_at_plane), size=tuple(size))
    # The same request CROSSING the plane rides the reflected gather instead: its
    # below-plane cells must carry the SAME per-component parity the unfold does —
    # requested cell -j samples parity * stored cell 1+j, asserted cell for cell.
    spanning = driver.add_dft_monitor(1.0, components=("Ex", "Ey", "Ez"),
                                      center=(0.0, 0.0, 0.0), size=tuple(size))
    driver.run(num_steps=14)

    for name in ("Ex", "Ey", "Ez"):
        quadrant = np.asarray(monitor.get_dft(name))
        unfolded = np.asarray(monitor.get_dft_full(name))
        parity = mirror_parity(name, axis, phase)
        centre = unfolded.shape[axis] // 2
        mirrored = parity * np.flip(
            quadrant[(slice(None),) * axis + (slice(1, None),)], axis=axis)
        np.testing.assert_array_equal(
            unfolded[(slice(None),) * axis + (slice(0, centre),)], mirrored,
            err_msg=f"{name} under Mirror({'XYZ'[axis]!r}, {phase:+d}) unfolded with the wrong sign")
        # And the unfolding is not a no-op on this component.
        if float(np.abs(quadrant).max()) > 0.0:
            assert float(np.abs(mirrored).max()) > 0.0

        start = spanning.region[2 * axis]
        assert start < 0, f"the spanning monitor must register below the plane, got {spanning.region}"
        gathered = np.asarray(spanning.get_dft(name))
        below = -start
        for offset in range(below):
            row = gathered[(slice(None),) * axis + (offset,)]  # requested cell start+offset < 0
            image_cell = 1 - (start + offset)  # its stored image about the plane
            image = gathered[(slice(None),) * axis + (image_cell - start,)]
            np.testing.assert_array_equal(
                row, parity * image,
                err_msg=f"{name} under Mirror({'XYZ'[axis]!r}, {phase:+d}): gathered cell "
                        f"{start + offset} is not parity x its image cell {image_cell}")


@pytest.mark.parametrize("axis,phase", [(0, +1), (2, +1), (2, -1)])
def test_a_live_far_face_is_refused_on_every_folded_axis(axis, phase):
    """A live periodic far face is SERVED linearly and refused only under a nonlinearity.

    A folded periodic axis now stores its second-mirror plane and reflects past
    it (``Grid.stored_cells``, ``stepping._far_reflect_rows``,
    ``fill_folded_far_ghosts_*``), so a linear folded run with the face fully
    live must BE the full-domain engine run's stored layout, bit for bit —
    asserted here on every axis and both phases, at a face amplitude the old
    zero-ghost treatment measured 8.8e-01 against. The refusal that remains is
    the nonlinear one: ``_nonlinear_transverse_sums`` shifts component PRODUCTS,
    which carry no single mirror parity and still read a zero far ghost, so a
    chi3 fold is refused once that face lights up — and is NOT refused while it
    stays dark, so the guard is not simply rejecting every folded nonlinear run.
    """
    from .fields import mirror_parity
    from .grid import Mirror

    name = "XYZ"[axis]
    cell = [3.0, 3.0, 3.0]
    cell[axis] = 2.0  # Short on the folded axis, so the wave reaches its far face.
    component = next(c for c in ("Ex", "Ey", "Ez") if mirror_parity(c, axis, phase) > 0)
    source = {"component": component, "frequency": 1.0, "center": (0.0, 0.0, 0.0),
              "size": (0.0, 0.0, 0.0)}

    def build(symmetry, chi3=0.0):
        driver = FdtdDriver(cell_size=tuple(cell), resolution=10, force_complex_fields=True,
                            symmetry=symmetry)
        if chi3:
            driver.set_chi3(chi3)
        driver.add_source(source)
        return driver

    live = build((Mirror(name, phase),))
    live.run(until=3.0)  # Far face fully live: served, no refusal.
    face_axis, ratio = live._folded_far_face_ratio()
    assert face_axis == axis and ratio > 1e-2, (
        f"the far face of the folded {name} axis carries only {ratio:.3e}; this case is "
        f"not exercising a live face and proves nothing"
    )
    full = build(())
    full.run(until=3.0)
    for primary in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        reference = _quadrant_of(np.asarray(full.get_field(primary, cell_centered=False)),
                                 ((axis, phase),), full.shape)
        got = np.asarray(live.get_field(primary, cell_centered=False))
        np.testing.assert_array_equal(got, reference, err_msg=(
            f"{primary}: the folded {name}{phase:+d} run with a live far face differs "
            f"from the full-domain run's stored layout"))

    # The nonlinear guard: dark face allowed, live face refused.
    quiet = build((Mirror(name, phase),), chi3=1e-6)
    quiet.run(until=0.4)
    assert quiet._folded_far_face_ratio()[1] < 1e-2, "this case starts live; it proves nothing"
    hot = build((Mirror(name, phase),), chi3=1e-6)
    with pytest.raises(RuntimeError, match="mirror-folded NONLINEAR run carries"):
        hot.run(until=3.0)
    assert hot._folded_far_face_ratio()[1] > 1e-2


@pytest.mark.parametrize("planes", [(("X", +1),), (("Z", -1),), (("X", +1), ("Y", +1))])
def test_a_folded_flux_plane_reports_the_full_domain_number(planes):
    """A flux plane spanning a folded axis integrates the full surface it names.

    ``_register_volume`` used to clamp such a plane at the mirror plane and
    ``get_flux`` refused the misnamed half-surface integral — measured then: 89 %
    of the full-domain power for one fold, 79 % for two, no fixed factor
    recovering it. The reflected gather serves the below-plane cells from their
    stored images now (MEEP's own ``use_symmetry=true`` loop), so the folded run
    must reproduce ITS OWN full-domain twin's number: same plane, same source,
    same run length, folded against unfolded, the driver-only self-consistency
    twin of the MEEP-oracle fold-gather cases in test_from_meep.py.

    A plane wholly inside the stored half (the Z-fold parametrisation) keeps its
    exact-equality assertion, so this pins the gather against the untouched path
    and not merely against itself.
    """
    from .grid import Mirror

    def run(symmetry, centre, size):
        driver = FdtdDriver(cell_size=(3.0, 3.0, 3.0), resolution=10,
                            force_complex_fields=True, symmetry=tuple(symmetry))
        driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
        driver.setup_pml(4)
        driver.add_source({"component": "Ez", "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                           "frequency": 1.0})
        monitor = driver.add_flux_monitor(frequencies=1.0, center=centre, size=size)
        driver.run(until=12 * driver.dt)
        return driver, monitor

    symmetry = tuple(Mirror(axis, phase) for axis, phase in planes)
    folded_axes = tuple("XYZ".index(axis) for axis, _ in planes)
    # A plane normal to z, spanning x and y, so every X or Y plane reaches below it.
    spanning = ((0.0, 0.0, 0.6), (2.0, 2.0, 0.0))
    full_driver, full_monitor = run((), *spanning)
    folded_driver, folded_monitor = run(symmetry, *spanning)
    gathered_axes = tuple(axis for axis in folded_axes if axis != 2)

    reference = full_monitor.get_flux()
    folded = folded_monitor.get_flux()
    assert reference > 0.0, "the unfolded run carries no power; the comparison is vacuous"
    if gathered_axes:
        for axis in gathered_axes:
            assert folded_monitor._region[2 * axis] < 0, (
                f"{_plane_label(planes)}: the plane must register its below-plane cells "
                f"on axis {axis}, got region {folded_monitor._region}"
            )
        assert folded == pytest.approx(reference, rel=1e-05), (
            f"{_plane_label(planes)}: the folded plane reported {folded:.6e} against the "
            f"full-domain {reference:.6e} ({folded / reference:.4f}x) — the clamped "
            f"integral read 0.89x / 0.79x here before the gather"
        )
    else:
        # A Z fold does not reach a plane normal to z that sits above the mirror plane,
        # and the number must then be the full-domain number exactly.
        assert folded_monitor._gather is None
        assert folded == full_monitor.get_flux(), (
            f"{_plane_label(planes)}: an untouched folded plane must reproduce the full-domain "
            f"flux exactly, got {folded:.6e} vs {full_monitor.get_flux():.6e}"
        )
    folded_driver.close()
    full_driver.close()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_fold_is_exact_in_a_graded_medium_registered_at_the_component_positions(axis):
    """The fold is exact; a shared-scalar epsilon is what is half a cell off about the plane.

    A folded run and its unfolded twin are the same system only if they were given the
    same medium. ``set_epsilon`` takes ONE array indexed at the integer Yee position, so
    the component whose Yee shift on the folded axis is 1 (``Ex`` under an X plane) sits
    half a cell above the sample it is multiplied by: the two cells straddling the plane
    take eps(-dx) and eps(0), which are not mirror images. The medium is then not
    symmetric about the plane even though the geometry is, and the FULL-domain run is the
    one that drifts — the folded run enforces the symmetry exactly through its ghost cell.

    This pins both halves of that statement, which is the only way the measurement means
    anything: sampled at each component's own coordinates the fold reproduces the full
    domain EXACTLY, and sampled once at the integer position it does not — while grading
    along an axis that is NOT folded is exact either way. Without the second assertion a
    fold that had quietly stopped folding would pass.
    """
    from .dispersion import component_coordinates
    from .grid import Mirror

    def profile(position):  # Even about 0, so the geometry is symmetric about any plane.
        return 2.0 + 0.8 * np.exp(-(position ** 2) / 0.25)

    def build(symmetry, per_component, graded_axis):
        driver = FdtdDriver(cell_size=(3.0, 3.0, 3.0), resolution=10,
                            force_complex_fields=True, symmetry=tuple(symmetry))
        shape = driver.shape
        if per_component:
            volumes = {}
            for component in ("Ex", "Ey", "Ez"):
                coordinates = component_coordinates(driver.grid, component)[graded_axis]
                spread = [1, 1, 1]
                spread[graded_axis] = -1
                volumes[component] = (
                    profile(np.asarray(coordinates, dtype=np.float64).reshape(spread))
                    * np.ones(shape)
                ).astype(np.float32)
            driver.set_epsilon_components(volumes)
        else:
            spread = [1, 1, 1]
            spread[graded_axis] = -1
            positions = (driver.grid.axis_origin(graded_axis)
                         + np.arange(shape[graded_axis], dtype=np.float64) * driver.dx)
            driver.set_epsilon(
                (profile(positions.reshape(spread)) * np.ones(shape)).astype(np.float32)
            )
        component = next(name for name in ("Ex", "Ey", "Ez")
                         if fields_module.mirror_parity(name, axis, +1) > 0)
        driver.add_source({"component": component, "center": (0.0, 0.0, 0.0),
                           "size": (0.0, 0.0, 0.0), "frequency": 1.0})
        driver.run(until=12 * driver.dt)
        return driver

    def fold_gap(per_component, graded_axis):
        folded = build((Mirror("XYZ"[axis], +1),), per_component, graded_axis)
        full = build((), per_component, graded_axis)
        worst = 0.0
        for name in ("Dx", "Dy", "Dz"):
            reference = _quadrant_of(
                np.asarray(full.get_field(name, cell_centered=False)),
                ((axis, +1),), full.shape)
            got = np.asarray(folded.get_field(name, cell_centered=False))
            scale = float(np.max(np.abs(reference))) or 1.0
            worst = max(worst, float(np.max(np.abs(got - reference))) / scale)
        folded.close()
        full.close()
        return worst

    unfolded_axis = (axis + 1) % 3
    assert fold_gap(True, axis) == 0.0, (
        f"a {'XYZ'[axis]} fold in a medium graded across the plane and sampled at each "
        f"component's own Yee position must reproduce the full domain exactly, got "
        f"{fold_gap(True, axis):.3e}"
    )
    assert fold_gap(False, unfolded_axis) == 0.0, (
        f"a {'XYZ'[axis]} fold in a medium graded along the unfolded {'XYZ'[unfolded_axis]} "
        f"axis must be exact even with a shared-scalar epsilon"
    )
    shared = fold_gap(False, axis)
    assert shared > 1e-4, (
        f"the shared-scalar epsilon is registered half a cell off about the {'XYZ'[axis]} "
        f"plane, so the two runs hold different media and must NOT agree; measured "
        f"{shared:.3e}. If this now passes, the registration changed and set_epsilon's "
        f"docstring is stale"
    )
