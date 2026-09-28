"""
Cross-validation and contract tests for :class:`FdtdDriver`.

The numerical cases use CPU MEEP as the oracle: the same physics is set up twice
— once through ``meep.Simulation`` in a subprocess, once through the driver on
the NumPy backend — and the resulting Ez volumes are compared with quantified
bounds (relative L2 plus Pearson correlation of the magnitudes), never with a
nonzero-only assertion. The oracle runs in a subprocess for a platform reason:
meep and torch each ship their own OpenMP runtime and abort when loaded into one
process, and the pytest process reaches torch through neighbouring engine
modules. Oracle output travels as a ``.npy`` file in the test's tmp_path.

The Bloch-boundary cases are cross-validated the same way, at four wavevectors
including the Brillouin-zone edge, a negative component, one that is not a simple
fraction of the cell, and one phasing all three axes at once. They are compared on
the COMPLEX field: a Bloch boundary contributes nothing but phase, so a sign error
or a missing 2*pi leaves every magnitude looking reasonable. Each case also
measures the same reference against a k = 0 run — the run an engine that accepted
``k_point`` and ignored it would produce — which must land far outside the bar.

The remaining cases need no oracle: a symmetry run is compared cell for cell
against the equivalent full-domain run (the equivalence that proves the
post-injection D-side mirror fill of port-reference §5.2 is wired in), and a
zero-amplitude source is checked to leave the fields exactly zero — with a
positive control, so a run that computes nothing cannot pass as a perfect one.
A k = 0 run is pinned bit-for-bit against a driver built with no ``k_point`` at
all, so the Bloch machinery cannot cost the pre-existing parity floor a single ULP.

Two of the oracle-free cases guard the PML specifically, because a coefficient
table can be self-consistently wrong in a way an oracle-only suite catches late:
an even source must stay even under an absorbing boundary, and the interior of a
PML-terminated run must reproduce a run whose boundaries are never reached, by a
margin that improves as the grid is refined. The second is the scaling assertion —
it separates a correctly graded layer from one whose error is a fixed fraction of
a cell, which no tuned constant can fake at two resolutions at once.

Three more oracles need no CPU MEEP either. Monitors are cross-checked against
themselves: a run carrying several monitors must reproduce, region for region,
what each monitor accumulates when it is the only one in the run — a bug that
aimed them all at one region or updated only the first would otherwise pass
unnoticed, since every monitor would still be full of plausible numbers. Two flux
planes placed symmetrically either side of a symmetric sheet source must carry
equal and opposite power, which is an analytic property of the configuration
rather than a recorded number. And the field-decay stop is checked against a
transcription of MEEP's own ``stop_when_fields_decayed`` closure, evaluated
offline on a probe history recorded through ``get_field``, so the driver's
device-side windowing is compared against the criterion it claims to implement
rather than against itself.

The last section swaps BOTH codes into real (float32) storage — MEEP's own default,
reached by omitting ``force_complex_fields`` — and repeats the plain, absorbing,
dispersive and mirror-folded comparisons there, plus two cases with odd cell counts on
all three axes and a source at no round coordinate. Its oracle asserts
``simulation.fields.is_real`` before saving, because a case that silently ran complex
would satisfy every bound while validating nothing real-specific. A companion oracle
runs one case both ways inside MEEP to establish that MEEP's real mode IS the real part
of its complex mode, byte for byte, which is what makes the engine's own byte equality
the right target rather than an over-claim.

Everything here runs on plain NumPy; no CUDA device and no cupy are involved.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from .dispersion import DRUDE, E_COMPONENTS, LORENTZIAN, Susceptibility, sample_region
from .driver import (
    FdtdCancelled,
    FdtdDriver,
    FdtdNonlinearityOutOfRange,
    _normalize_frequencies,
)
from .grid import Grid, Mirror
from .harminv import Harminv
from .stepping import PERIODIC, _boundary_kinds

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(MEEP_MISSING, reason="CPU MEEP is not installed in this environment")

_ORACLE_SCRIPT = '''"""Generate one CPU-MEEP reference Ez volume for the FDTD driver parity tests."""
import sys

import numpy as np
import meep as mp

case = sys.argv[1]
output_path = sys.argv[2]

if case == "point_no_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 4),
        resolution=10,
        sources=[
            mp.Source(
                mp.ContinuousSource(frequency=1.0),
                component=mp.Ez,
                center=mp.Vector3(0, 0, -1.05),
            )
        ],
        force_complex_fields=True,
        k_point=mp.Vector3(0, 0, 0),
    )
elif case == "sheet_with_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 6),
        resolution=10,
        sources=[
            mp.Source(
                mp.ContinuousSource(frequency=1.0),
                component=mp.Ez,
                center=mp.Vector3(0, 0, -1.5),
                size=mp.Vector3(1, 1, 0),
            )
        ],
        boundary_layers=[mp.PML(1.0)],
        force_complex_fields=True,
        k_point=mp.Vector3(0, 0, 0),
    )
elif case.startswith("bloch:"):
    # "bloch:kx,ky,kz" — same cell, source and run for every k, so the only thing
    # that varies between the references is the boundary phase itself.
    kx, ky, kz = (float(value) for value in case.split(":", 1)[1].split(","))
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 2),
        resolution=10,
        sources=[
            mp.Source(
                mp.ContinuousSource(frequency=1.0),
                component=mp.Ez,
                center=mp.Vector3(0, 0, 0.05),
            )
        ],
        force_complex_fields=True,
        k_point=mp.Vector3(kx, ky, kz),
    )
elif case == "odd_folded_metallic_pml":
    # cavity-farfield.py's shape: metallic outer walls (MEEP's default — no k_point),
    # PML on every face, X and Y mirrors, and an ODD cell count on both folded axes
    # (21 x 21 at resolution 10). The oracle is MEEP's own FOLDED run: at an odd
    # count MEEP folds about a grid point and shifts the window half a cell up, and
    # that folded system — not the unfolded script — is what the engine reproduces.
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2.1, 2.1, 2.0),
        resolution=10,
        sources=[
            mp.Source(
                mp.ContinuousSource(frequency=1.0),
                component=mp.Ez,
                center=mp.Vector3(0, 0, 0),
            )
        ],
        boundary_layers=[mp.PML(0.5)],
        symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y)],
        force_complex_fields=True,
    )
    simulation.run(until=3)
    np.save(output_path, np.asarray(simulation.get_array(component=mp.Ez)))
    raise SystemExit(0)
elif case.startswith("fold_bloch"):
    # A mirror on X with a Bloch phase on Y — the pairing this engine refused as
    # "not implemented" until it was measured. "fold_bloch_full:ky" is the plain
    # unfolded run; "fold_bloch_folded:ky" is MEEP's OWN folded run of the same
    # system, which exists to establish that stock MEEP serves the combination
    # rather than merely tolerating it. Same cell, source and stopping time as
    # `folded_cell`, so the far x face is live at t = 3.
    kind, ky = case.split(":", 1)
    symmetries = [mp.Mirror(mp.X)] if kind.endswith("_folded") else []
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 4),
        resolution=10,
        sources=[
            mp.Source(
                mp.ContinuousSource(frequency=1.0),
                component=mp.Ez,
                center=mp.Vector3(0, 0, 0.05),
            )
        ],
        symmetries=symmetries,
        force_complex_fields=True,
        k_point=mp.Vector3(0, float(ky), 0),
    )
    simulation.run(until=3)
    np.save(output_path, np.asarray(simulation.get_array(component=mp.Ez)))
    raise SystemExit(0)
elif case.startswith("folded_cell:"):
    # A cell whose x faces the wave DOES reach, so the folded run's far x face — where
    # this engine has a zero ghost and MEEP has the parity-weighted mirror image — is
    # the only thing that distinguishes the two. "t" selects how far the front has got.
    until = float(case.split(":", 1)[1])
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 4),
        resolution=10,
        sources=[
            mp.Source(
                mp.ContinuousSource(frequency=1.0),
                component=mp.Ez,
                center=mp.Vector3(0, 0, 0.05),
            )
        ],
        force_complex_fields=True,
        k_point=mp.Vector3(0, 0, 0),
    )
    simulation.run(until=until)
    np.save(output_path, np.asarray(simulation.get_array(component=mp.Ez)))
    raise SystemExit(0)
elif case.startswith("bfast:"):
    # test_refl_angular's own configuration, at a resolution a unit test can afford:
    # a z-stratified cell one pixel wide in x and y (MEEP builds it in 3-D — only
    # cell_size.z collapses a run), a Gaussian Ex source, PML in z, and MEEP's
    # BFAST shear along x. "bfast:theta,use" — `use` 0 turns bfast_scaled_k off
    # while keeping the Courant, which is the NEGATIVE CONTROL's reference.
    theta_deg, use_bfast = case.split(":", 1)[1].split(",")
    theta_rad = float(theta_deg) * np.pi / 180.0
    scaled_k = 1.4 * np.sin(theta_rad)
    simulation = mp.Simulation(
        resolution=20,
        cell_size=mp.Vector3(z=9.0),
        dimensions=3,
        default_material=mp.Medium(index=1.4),
        sources=[
            mp.Source(
                mp.GaussianSource(2.0, fwidth=1.0),
                component=mp.Ex,
                center=mp.Vector3(z=-3.5),
            )
        ],
        boundary_layers=[mp.PML(1.0)],
        k_point=mp.Vector3(),
        bfast_scaled_k=(scaled_k, 0, 0) if int(use_bfast) else (0, 0, 0),
        Courant=(1 - scaled_k) / 3 ** 0.5,
        geometry=[
            mp.Block(size=mp.Vector3(mp.inf, mp.inf, 4.5), center=mp.Vector3(z=2.25),
                     material=mp.Medium(index=3.5))
        ],
    )
    simulation.run(until=8.0)
    # The three components P polarization in the XZ plane populates. Ey, Hx and Hz
    # stay identically zero in both codes, which the test asserts separately.
    np.save(output_path, np.stack([
        np.asarray(simulation.get_array(component=mp.Ex)).reshape(-1),
        np.asarray(simulation.get_array(component=mp.Ez)).reshape(-1),
        np.asarray(simulation.get_array(component=mp.Hy)).reshape(-1),
    ]))
    raise SystemExit(0)
else:
    raise SystemExit("unknown oracle case: " + case)

simulation.run(until=3)
if case.startswith("bloch:"):
    # Ez interpolates across x and y on the way to cell centres and Ex across y and
    # z, so the pair between them exercises the wrap phase on every axis.
    np.save(output_path, np.stack([
        np.asarray(simulation.get_array(component=mp.Ez)),
        np.asarray(simulation.get_array(component=mp.Ex)),
    ]))
else:
    np.save(output_path, np.asarray(simulation.get_array(component=mp.Ez)))
'''


def _meep_oracle(tmp_path, case):  # Run CPU MEEP in its own process and load the reference Ez volume.
    script_path = tmp_path / "meep_oracle.py"
    script_path.write_text(_ORACLE_SCRIPT, encoding="utf-8")
    slug = "".join(character if character.isalnum() else "_" for character in case)
    output_path = tmp_path / f"{slug}_ez.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), case, str(output_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP oracle '{case}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


_FLUX_ORACLE_SCRIPT = '''"""CPU-MEEP flux spectrum through a plane spanning an absorbing axis."""
import sys

import numpy as np
import meep as mp

case = sys.argv[1]
output_path = sys.argv[2]

if case == "x_and_y_pml":
    layers = [mp.PML(0.4, direction=mp.X), mp.PML(0.4, direction=mp.Y)]
else:
    raise SystemExit("unknown flux oracle case: " + case)

simulation = mp.Simulation(
    cell_size=mp.Vector3(2, 2, 4), resolution=10, boundary_layers=layers,
    sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), component=mp.Ez,
                       center=mp.Vector3(0, 0, -0.5))],
    force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
# The plane spans the full x-y cross section, both of whose axes absorb. MEEP still
# takes the lattice shift there: use_bloch made every face Periodic, and the layer is a
# material graded underneath it.
region = simulation.add_flux(
    1.0, 0.4, 3, mp.FluxRegion(center=mp.Vector3(0, 0, 0.8), size=mp.Vector3(2, 2, 0),
                               direction=mp.Z), decimation_factor=1)
simulation.run(until=12)
np.save(output_path, np.asarray(mp.get_fluxes(region)))
'''


def _flux_oracle(tmp_path, case):  # One CPU-MEEP flux spectrum, in its own process.
    script_path = tmp_path / "meep_flux_oracle.py"
    script_path.write_text(_FLUX_ORACLE_SCRIPT, encoding="utf-8")
    output_path = tmp_path / f"{case}_flux.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), case, str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP flux oracle '{case}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


def _magnitude_metrics(candidate, reference):  # Relative L2 error and Pearson correlation of |field|.
    driver_magnitude = np.abs(np.asarray(candidate, dtype=np.complex128)).ravel()
    meep_magnitude = np.abs(np.asarray(reference, dtype=np.complex128)).ravel()
    reference_norm = float(np.linalg.norm(meep_magnitude))
    candidate_norm = float(np.linalg.norm(driver_magnitude))
    assert reference_norm > 0.0, "CPU-MEEP oracle produced an all-zero field; the comparison is meaningless."
    assert candidate_norm > 0.0, "Driver produced an all-zero field; an unmeasurable run must not score as perfect."
    relative_l2 = float(np.linalg.norm(driver_magnitude - meep_magnitude) / reference_norm)
    correlation = float(np.corrcoef(driver_magnitude, meep_magnitude)[0, 1])
    return relative_l2, correlation


def _complex_relative_l2(candidate, reference):  # Relative L2 on the complex field: catches phase error too.
    driver_field = np.asarray(candidate, dtype=np.complex128).ravel()
    meep_field = np.asarray(reference, dtype=np.complex128).ravel()
    return float(np.linalg.norm(driver_field - meep_field) / np.linalg.norm(meep_field))


def _assert_folded_matches_full(full, folded, components, tolerance=1e-5):  # Quadrant must reproduce the full run.
    # Quadrant cell q sits at the same physical position as full-domain cell
    # cx - 1 + q for every component: quadrant cell 0 straddles the mirror plane at
    # -dx/2 (half-integer components) or -dx (integer ones), which is cell cx - 1 of
    # the full grid either way. One tolerance scale is taken across all compared
    # components so a component that is still near zero cannot be judged against its
    # own round-off, and a run that produced nothing fails instead of passing.
    center_index = full.shape[0] // 2
    scale = max(float(np.max(np.abs(full.get_field(name, cell_centered=False)))) for name in components)
    assert scale > 0.0, "The full-domain run carries no field; the equivalence check would be vacuous."
    for name in components:
        full_values = full.get_field(name, cell_centered=False)
        folded_values = folded.get_field(name, cell_centered=False)
        reference = full_values[center_index - 1:]
        if reference.shape[0] < folded_values.shape[0]:
            # The stored second-mirror plane cell of a periodic even fold: full
            # cell 0's own samples one lattice vector up (no phase at k = 0).
            reference = np.concatenate((reference, full_values[0:1]), axis=0)
        np.testing.assert_allclose(
            folded_values,
            reference,
            rtol=0.0,
            atol=tolerance * scale,
            err_msg=f"Folded run diverges from the full-domain run in {name}.",
        )


def _compact_profile(positions, dx):  # Narrow Gaussian, truncated to a few cells of support.
    width = 2.0 * dx
    support = 5.0 * dx
    values = np.exp(-((positions / width) ** 2))
    return np.where(np.abs(positions) <= support, values, 0.0)


def _symmetric_seed(nx, ny, nz, dx):  # Compact D-field blob, exactly even about the x = 0 mirror plane.
    # Integer-positioned components sample x_i = (i - nx/2) * dx, so index i and index
    # nx - i are mirror partners and a function of |x| is exactly even in x. The blob is
    # truncated to a few cells so the x boundaries stay identically zero for the whole
    # run: the folded grid terminates its far x face metallically while the full grid
    # wraps periodically, and only an untouched boundary makes the two runs comparable
    # to round-off. Wrapping in y and z is harmless — both runs are periodic there.
    profile_x = _compact_profile((np.arange(nx) - nx / 2) * dx, dx)
    profile_y = _compact_profile((np.arange(ny) - ny / 2) * dx, dx)
    profile_z = _compact_profile((np.arange(nz) - nz / 2) * dx, dx)
    return (profile_x[:, None, None] * profile_y[None, :, None] * profile_z[None, None, :]).astype(np.complex64)


def _transverse_seed(nx, ny, nz, dx):  # Compact in x and y, uniform in z: a seed that can decay away.
    # dDz/dz = 0, so the seeded D carries no bound charge. A Dz blob that also varies in
    # z does, and its static field is permanent — no absorber removes it, and a probe
    # anywhere near it never decays no matter how long the run. That is physics, not a
    # defect, but it makes the varying-in-z blob useless as a decay case.
    profile_x = _compact_profile((np.arange(nx) - nx / 2) * dx, dx)
    profile_y = _compact_profile((np.arange(ny) - ny / 2) * dx, dx)
    return (profile_x[:, None, None] * profile_y[None, :, None] * np.ones(nz)[None, None, :]).astype(np.complex64)


@requires_meep
@skip_without_meep
def test_point_source_matches_cpu_meep(tmp_path):  # Core algorithm parity: CW point source, no PML.
    meep_ez = _meep_oracle(tmp_path, "point_no_pml")
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    # z = -1.05 lands exactly on an Ez Yee point on both grids, so neither side splits
    # the source between cells and the comparison is not measuring interpolation.
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.05), "size": (0.0, 0.0, 0.0)}
    )
    driver.run(until=3.0)
    driver_ez = driver.get_field("Ez")
    # MEEP's get_array returns (N+1) points per axis, the extra plane being the
    # periodic duplicate at index 0; index i + 1 is this grid's cell i.
    reference = meep_ez[1:, 1:, 1:]
    assert driver_ez.shape == reference.shape
    relative_l2, correlation = _magnitude_metrics(driver_ez, reference)
    assert relative_l2 < 0.05, f"|Ez| relative L2 error {relative_l2:.4f} exceeds the 5% cross-validation bar."
    assert correlation > 0.999, f"|Ez| correlation {correlation:.6f} below 0.999."
    # Magnitude agreement is blind to a global phase (the weakness of the bundle's
    # correlation-only suite), so the complex field is held to the same bound.
    complex_l2 = _complex_relative_l2(driver_ez, reference)
    assert complex_l2 < 0.05, f"Complex Ez relative L2 error {complex_l2:.4f} exceeds the 5% bar."


@requires_meep
@skip_without_meep
def test_sheet_source_with_pml_matches_cpu_meep(tmp_path):  # PML path parity: extended sheet source.
    meep_ez = _meep_oracle(tmp_path, "sheet_with_pml")
    driver = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=True)
    driver.setup_pml(10)  # 1.0 length unit at resolution 10.
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.5), "size": (1.0, 1.0, 0.0)}
    )
    driver.run(until=3.0)
    driver_ez = driver.get_field("Ez")
    reference = meep_ez[1:, 1:, 1:]
    assert driver_ez.shape == reference.shape
    relative_l2, correlation = _magnitude_metrics(driver_ez, reference)
    assert correlation > 0.999, f"|Ez| correlation {correlation:.6f} below 0.999 on the PML path."
    # Held far inside the <5% bar of the parity matrix §7: measured
    # 2.9e-07 magnitude / 4.0e-07 complex, the same band as the no-PML core. History of
    # this one number, which is the engine's most common configuration: 8.2e-02 / 1.1e-01
    # while the upper-face sigma was graded one cell too deep, then 5.8e-04 / 6.3e-04
    # while X and Y under a two-sided layer were terminated metallically instead of
    # wrapping as MEEP does (stepping._boundary_kinds), then here.
    assert relative_l2 < 1e-5, f"|Ez| relative L2 error {relative_l2:.3e} exceeds the 1e-5 bar."
    complex_l2 = _complex_relative_l2(driver_ez, reference)
    assert complex_l2 < 1e-5, f"Complex Ez relative L2 error {complex_l2:.3e} exceeds the 1e-5 bar."


# --- Bloch-periodic boundaries ---------------------------------------------------

# Four wavevectors covering the ways the phase can be got wrong: the Brillouin-zone
# edge (k*L = 1/2, which MEEP pins to exactly -1 rather than computing exp(i*pi)), a
# negative component (which flips which face carries the conjugate), a value that is
# not a simple fraction of the cell (so a 2*pi factor cannot cancel by luck), and one
# with all three axes phased at once (so the per-axis factors cannot be crossed).
_BLOCH_K_POINTS = (
    (0.25, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.1234567, 0.0, 0.0),
    (0.15, -0.35, 0.4),
)
_BLOCH_CELL = (2.0, 2.0, 2.0)
_BLOCH_RESOLUTION = 10
# x = y = 0 is an Ez integer Yee point and z = 0.05 a half-integer one on both grids,
# so neither side splits the source between cells (as in the k = 0 point-source case).
_BLOCH_SOURCE_CENTER = (0.0, 0.0, 0.05)


def _bloch_case(k_point):  # Oracle case string for one wavevector.
    return "bloch:" + ",".join(repr(float(component)) for component in k_point)


def _bloch_driver(k_point):  # The driver half of the Bloch comparison, identical but for k.
    driver = FdtdDriver(
        cell_size=_BLOCH_CELL,
        resolution=_BLOCH_RESOLUTION,
        force_complex_fields=True,
        k_point=k_point,
    )
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": _BLOCH_SOURCE_CENTER, "size": (0.0, 0.0, 0.0)}
    )
    driver.run(until=3.0)
    return driver


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("k_point", _BLOCH_K_POINTS)
def test_bloch_boundary_matches_cpu_meep(tmp_path, k_point):  # Bloch parity: complex field, per k.
    """The Bloch wrap phase, cross-validated against ``mp.Simulation(k_point=...)``.

    The comparison is on the COMPLEX field, not on magnitudes. A Bloch boundary
    contributes nothing but phase, so |Ez| barely moves when the sign or the 2*pi is
    wrong — the seeded-eigenmode controls in test_stepping.py measure the same two
    defects at 1.6e-01 and 3.6e-01 on magnitudes against 1.98 and 4.3e-01 on the
    complex field. Ez and Ex are both compared because their Yee shifts make them
    interpolate across different axes on the way to cell centres, so between them
    every axis's wrap phase appears in the readback.

    The k = 0 run is included as the negative control: an engine that accepted
    ``k_point`` and ignored it would be exactly that run, and it must be far outside
    the bar against every one of these references.

    Keep all four wavevectors. The zone-edge one is doubly degenerate and cannot
    stand in for the rest: its phase is exactly -1, which is its own conjugate, so it
    cannot see a sign error; and anti-periodicity plus a source centred on x = 0
    forces Ez to vanish on the wrap plane itself, so it cannot see a readback that
    drops the phase either. Both defects were confirmed to survive this case alone
    and to be caught by the other three.
    """
    reference = _meep_oracle(tmp_path, _bloch_case(k_point))
    assert reference.shape[0] == 2, "The Bloch oracle carries an (Ez, Ex) pair."
    driver = _bloch_driver(k_point)
    ignored_k = _bloch_driver((0.0, 0.0, 0.0))

    for index, component in enumerate(("Ez", "Ex")):
        # MEEP's get_array returns (N+1) points per axis, the extra plane being the
        # boundary image at index 0; index i + 1 is this grid's cell i.
        meep_field = reference[index][1:, 1:, 1:]
        assert float(np.linalg.norm(meep_field)) > 0.0, f"CPU MEEP produced an all-zero {component}."
        driver_field = driver.get_field(component)
        assert driver_field.shape == meep_field.shape
        complex_l2 = _complex_relative_l2(driver_field, meep_field)
        # Bound at 5e-06 rather than the 5% bar the PML cases carry: with no absorber
        # this is the same core algorithm as the k = 0 point-source case, whose
        # recorded floor is 1.7e-07, so anything above round-off is a real defect.
        # Measured across the four k values: Ez 2.3e-07 to 4.0e-07, Ex 4.4e-07 to
        # 7.0e-07 (Ex is a weaker, more interpolated component of a z dipole).
        assert complex_l2 < 5e-6, (
            f"k_point={k_point}: complex {component} relative L2 {complex_l2:.4e} exceeds the "
            f"5e-06 no-PML cross-validation floor."
        )
        magnitude_l2, correlation = _magnitude_metrics(driver_field, meep_field)
        assert magnitude_l2 < 5e-6 and correlation > 0.9999

        blind = _complex_relative_l2(ignored_k.get_field(component), meep_field)
        assert blind > 0.05, (  # Measured 8.7e-01 to 1.35.
            f"k_point={k_point}: a run at k = 0 reproduces this reference to {blind:.4e}, so the "
            f"comparison cannot tell whether the boundary phase was applied at all."
        )


@requires_meep
@skip_without_meep
def test_bloch_at_k_zero_reproduces_the_plain_periodic_parity_floor(tmp_path):
    """k = 0 through the Bloch machinery must land on the recorded no-PML floor.

    The established core-algorithm floor is 1.7e-07 relative L2 against CPU MEEP
    with no PML. Adding a boundary phase must not move it, and the way that is
    guaranteed is by skipping the multiply entirely when k = 0 rather than
    multiplying by ``1 + 0j`` — which test_stepping pins bit-for-bit. This is the
    same statement re-measured against the oracle: 1.6e-07 on the magnitude and
    2.4e-07 on the complex field, identical to the last digit whether ``k_point``
    is passed as zero or left off.
    """
    meep_ez = _meep_oracle(tmp_path, "point_no_pml")
    reference = meep_ez[1:, 1:, 1:]
    measured = []
    for k_point in (None, (0.0, 0.0, 0.0)):
        arguments = {} if k_point is None else {"k_point": k_point}
        driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True, **arguments)
        driver.add_source(
            {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.05), "size": (0.0, 0.0, 0.0)}
        )
        driver.run(until=3.0)
        magnitude_l2, _ = _magnitude_metrics(driver.get_field("Ez"), reference)
        complex_l2 = _complex_relative_l2(driver.get_field("Ez"), reference)
        assert magnitude_l2 < 1e-6 and complex_l2 < 1e-6, (
            f"k_point={k_point}: Ez relative L2 {magnitude_l2:.3e} (magnitude) / {complex_l2:.3e} "
            f"(complex) has moved off the recorded 1.7e-07 no-PML parity floor."
        )
        measured.append(complex_l2)
    assert measured[0] == measured[1], "Passing an explicit zero k_point changed the parity floor."


def test_zero_k_point_runs_bit_identically_to_a_driver_built_without_one():
    # The degenerate case, checked on the raw field bytes rather than a tolerance: an
    # explicit k = 0 must not merely agree with the pre-Bloch path, it must BE it.
    fields = {}
    for label, arguments in (("implicit", {}), ("explicit", {"k_point": (0.0, 0.0, 0.0)})):
        driver = FdtdDriver(cell_size=(1.2, 1.2, 1.6), resolution=10, force_complex_fields=True, **arguments)
        driver.add_source(
            {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -0.35), "size": (0.4, 0.4, 0.0)}
        )
        driver.run(num_steps=40)
        fields[label] = {name: driver.get_field(name, cell_centered=False) for name in ("Dz", "Bx", "By")}
        fields[label]["Ez"] = driver.get_field("Ez")

    for name, values in fields["implicit"].items():
        assert values.tobytes() == fields["explicit"][name].tobytes(), (
            f"{name} is not bit-identical between an implicit and an explicit k_point of zero."
        )
        assert float(np.abs(values).max()) > 0.0, "An all-zero run would make the comparison vacuous."


def test_bloch_refuses_real_fields_and_symmetry_but_not_an_absorber():
    # Each of these would otherwise produce a plausible field for a system the caller
    # did not ask for, so each raises with a message naming the actual conflict.
    with pytest.raises(ValueError, match="complex fields"):
        FdtdDriver(cell_size=(2, 2, 2), resolution=10, force_complex_fields=False, k_point=(0.3, 0.0, 0.0))
    # k = 0 with real fields stays legal: the refusal is about the phase, not the dtype.
    FdtdDriver(cell_size=(2, 2, 2), resolution=10, force_complex_fields=False, k_point=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="mirror-symmetric axis"):
        FdtdDriver(cell_size=(2, 2, 2), resolution=10, symmetry=("X",), k_point=(0.3, 0.0, 0.0))
    # A phase on an axis that is NOT folded used to raise here. It does not any more:
    # the fold and the lattice phase compose per axis, exactly as MEEP composes them,
    # and the pairing reproduces CPU MEEP at 2.8e-06 with the folded far face live
    # (test_a_fold_and_a_bloch_phase_on_another_axis_match_cpu_meep, k = 0 control
    # 8.5e-01). Both spellings are legal.
    FdtdDriver(cell_size=(2, 2, 2), resolution=10, symmetry=("X",), k_point=(0.0, 0.0, 0.3))
    FdtdDriver(cell_size=(2, 2, 2), resolution=10, symmetry=("X",), k_point=(0.0, 0.3, 0.0))

    # A layer ON the phased axis used to raise here. It does not any more: MEEP's
    # use_bloch keeps that axis Periodic and grades the absorber underneath the wrap,
    # and stepping the pairing reproduces CPU MEEP at 2.82e-07 (parity case
    # `bloch_on_pml_axis`, control 1.4e-01). Both spellings are now legal.
    FdtdDriver(cell_size=(2, 2, 2), resolution=10, k_point=(0.3, 0.0, 0.0)).setup_pml(4)
    FdtdDriver(cell_size=(2, 2, 2), resolution=10).setup_pml(4)
    # The layer off the phased axis is the grating, exercised end to end in
    # test_grating_transmission_*.
    FdtdDriver(cell_size=(2, 2, 2), resolution=10, k_point=(0.3, 0.0, 0.0)).setup_pml({"z": 4})

    with pytest.raises(ValueError, match="k_point"):
        FdtdDriver(cell_size=(2, 2, 2), resolution=10, k_point=(0.1, 0.2))
    with pytest.raises(ValueError, match="finite"):
        FdtdDriver(cell_size=(2, 2, 2), resolution=10, k_point=(float("inf"), 0.0, 0.0))


def test_pml_run_preserves_the_mirror_symmetry_of_the_source():  # PML tables, no oracle needed.
    # The sheet case is exactly even in x and in y, so Ez must be even in both. The
    # engine's own no-PML path holds this bit-exactly (measured 0.0), which makes any
    # asymmetry here a property of the PML coefficient tables alone: a lower/upper
    # face pair graded differently cannot produce an even field. Measured 1.8e-04;
    # with the upper face graded one cell too deep it was 8.3e-02 in x, 1.2e-01 in y.
    driver = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=True)
    driver.setup_pml(10)
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.5), "size": (1.0, 1.0, 0.0)}
    )
    driver.run(until=3.0)
    driver_ez = driver.get_field("Ez")
    scale = float(np.linalg.norm(driver_ez))
    assert scale > 0.0, "The run carries no field; the symmetry check would be vacuous."
    for axis, flipped in (("x", driver_ez[::-1, :, :]), ("y", driver_ez[:, ::-1, :])):
        asymmetry = float(np.linalg.norm(driver_ez - flipped)) / scale
        assert asymmetry < 1e-3, f"PML broke the source's mirror symmetry in {axis}: {asymmetry:.3e}"


def _pml_transparency(pml_cells, resolution, until=2.0):  # Interior error against a boundary-free run.
    # A correct absorber is invisible: the interior of a PML-terminated run must
    # reproduce the same physics computed in a domain large enough that no boundary is
    # reached. The wide run needs no PML — by `until` a wave from the origin has
    # travelled 2.0 of the 2.5 units to its nearest face — so it is the free-space truth.
    source = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)}
    wide = FdtdDriver(cell_size=(5.0, 5.0, 5.0), resolution=resolution, force_complex_fields=True)
    wide.add_source(source)
    wide.run(until=until)
    narrow = FdtdDriver(cell_size=(3.0, 3.0, 3.0), resolution=resolution, force_complex_fields=True)
    narrow.setup_pml(pml_cells)
    narrow.add_source(source)
    narrow.run(until=until)
    reference, candidate = wide.get_field("Ez"), narrow.get_field("Ez")
    # Both cells are centred on the origin, so narrow cell j is wide cell j + offset.
    offset = (reference.shape[0] - candidate.shape[0]) // 2
    inner = slice(pml_cells, candidate.shape[0] - pml_cells)
    outer = slice(offset + pml_cells, offset + candidate.shape[0] - pml_cells)
    interior = candidate[inner, inner, inner]
    truth = reference[outer, outer, outer]
    truth_norm = float(np.linalg.norm(truth))
    assert truth_norm > 0.0, "The boundary-free reference run carries no field."
    return float(np.linalg.norm(interior - truth)) / truth_norm


def test_pml_is_transparent_and_improves_as_the_grid_is_refined():  # Scaling, no oracle needed.
    # The anti-tuning assertion. A one-cell misplacement of the grading is a fixed
    # fraction 1/n_pml of the layer, so it behaves as O(dx) and barely improves under
    # refinement; a correct grading converges properly. Holding the layer at 0.6 units
    # and refining, the measured transparency error goes 5.7e-03 -> 8.8e-04 (6.4x),
    # where the one-cell-deep grading went 7.6e-02 -> 5.2e-02 (only 1.5x). Any constant
    # tuned to pass one resolution fails the ratio at the other.
    coarse = _pml_transparency(6, resolution=10)
    fine = _pml_transparency(9, resolution=15)
    assert coarse < 1.2e-2, f"PML is not transparent at resolution 10: {coarse:.3e}"
    assert fine < 3.0e-3, f"PML is not transparent at resolution 15: {fine:.3e}"
    assert coarse / fine > 3.0, (
        f"Refining the grid improved PML transparency only {coarse / fine:.2f}x "
        f"({coarse:.3e} -> {fine:.3e}); a correctly graded layer converges, an "
        f"O(dx) coefficient error does not."
    )
    # A thicker layer must also absorb better at fixed resolution. This leg pins the
    # expected physics but does not by itself discriminate the one-cell bug, which
    # improved with thickness too — the resolution ratio above is what catches it.
    assert _pml_transparency(10, resolution=10) < coarse


def test_symmetry_run_matches_full_domain_run():  # Mirror symmetry must reproduce the full-domain physics.
    # This is the equivalence test port-reference §5.2 asks for. Under an X mirror the
    # D cells at index 0 are non-owned: the curl step masks them, and only the
    # post-injection fill repairs them from their mirror partner at index 2. With the
    # fill missing those cells stay stale, feed the next curl, and the quadrant drifts
    # away from the full-domain answer within a few steps.
    cell_size = (4.0, 2.0, 2.0)  # Wide in x so the seeded blob never reaches an x boundary.
    resolution = 10
    num_steps = 12
    full = FdtdDriver(cell_size=cell_size, resolution=resolution, force_complex_fields=True)
    folded = FdtdDriver(cell_size=cell_size, resolution=resolution, force_complex_fields=True, symmetry=("X",))
    nx_full, ny_full, nz_full = full.shape
    seed = _symmetric_seed(nx_full, ny_full, nz_full, full.dx)
    center_index = nx_full // 2
    full.set_field("Dz", seed)
    # The stored second-mirror plane cell of the even periodic fold takes full cell 0's
    # samples one lattice vector up (no phase at k = 0), as in _assert_folded_matches_full.
    folded_seed = np.concatenate((seed[center_index - 1 :], seed[0:1]), axis=0)
    folded.set_field("Dz", folded_seed)
    assert folded.shape[0] == folded_seed.shape[0]
    for _ in range(num_steps):
        full.step()
        folded.step()
    _assert_folded_matches_full(full, folded, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"))


def test_symmetry_run_with_on_plane_source_matches_full_domain_run():  # On-plane source must fold identically.
    cell_size = (2.0, 2.0, 4.0)
    resolution = 10
    num_steps = 12  # Short enough that nothing reaches the outer boundaries.
    source = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)}
    full = FdtdDriver(cell_size=cell_size, resolution=resolution, force_complex_fields=True)
    folded = FdtdDriver(cell_size=cell_size, resolution=resolution, force_complex_fields=True, symmetry=("X",))
    full.add_source(source)
    folded.add_source(source)
    full.run(num_steps=num_steps)
    folded.run(num_steps=num_steps)
    # Looser than the seeded case (which agrees bit for bit) because the two grids
    # accumulate the injected current in float32 in a different order; the measured
    # deviation is ~2e-6 of the field, while a missing mirror fill shows up at ~30%.
    _assert_folded_matches_full(full, folded, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), tolerance=1e-4)


def _folded_pml_deviation(until, thickness=6, cell_size=(3.0, 3.0, 4.0)):
    """Folded-quadrant vs full-domain deviation for a PML run, plus the far-x-face level."""
    source = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)}
    runs = {}
    for label, symmetry in (("full", ()), ("folded", ("X",))):
        driver = FdtdDriver(cell_size=cell_size, resolution=10, force_complex_fields=True,
                            symmetry=symmetry)
        driver.setup_pml(thickness)
        driver.add_source(source)
        driver.run(until=until)
        runs[label] = driver.get_field("Ez", cell_centered=False)
    # Quadrant cell q is full-domain cell cx - 1 + q, as in _assert_folded_matches_full;
    # the stored second-mirror plane cell is full cell 0's image one lattice vector up.
    reference = runs["full"][runs["full"].shape[0] // 2 - 1:]
    if reference.shape[0] < runs["folded"].shape[0]:
        reference = np.concatenate((reference, runs["full"][0:1]), axis=0)
    assert runs["folded"].shape == reference.shape
    scale = float(np.linalg.norm(reference))
    assert scale > 0.0, "The full-domain PML run carries no field; the equivalence would be vacuous."
    peak = float(np.abs(runs["full"]).max())
    return (float(np.linalg.norm(runs["folded"] - reference)) / scale,
            float(np.abs(runs["full"][-1]).max()) / peak)


def test_symmetry_run_with_pml_matches_full_domain_run():  # The skip_lower grading path.
    """A folded quadrant must reproduce the full-domain run wherever the two are the same system.

    A folded axis stores one PML face instead of two, and its cells are shifted by half
    a cell (cell 0 straddles the mirror plane). Both the samples and the outer wall move
    by the same amount, so the distance-to-wall grading is unchanged and the quadrant
    must reproduce the full-domain run. With the upper face graded one cell too deep the
    two disagreed by 1.1e-02, because the folded axis kept only the face that was
    mis-graded.

    The two runs share the outer x boundary again: the full-domain run WRAPS it
    (MEEP's condition, and the fix that took the uniform-PML floor from 6.31e-04 to
    4.00e-07), and a folded periodic axis now stores its second-mirror plane and
    reflects past it — which IS that wrap, seen from the quadrant. The equivalence
    is therefore exact whether the far face is quiet or live (measured 0.0 bit for
    bit at until=1.0 with the face at 2.6e-06 of peak, and at until=2.0 with it at
    1.2e-03 — where the zero-ghost treatment this replaces was bounded only by the
    face's own amplitude). The grading itself is pinned exactly and boundary-free by
    ``test_grid_pml.py::test_the_folded_pml_table_is_the_full_domain_table_under_the_half_cell_shift``.
    """
    # While the far x face is quiet the absorber is already fully engaged (the wave
    # entered it at t ~ 0.45), so this exercises the grading, not an empty layer.
    quiet_deviation, quiet_face = _folded_pml_deviation(until=1.0)
    assert quiet_face < 1e-5, (
        f"the far x face must still be quiet in the first case, carries {quiet_face:.2e}")
    assert quiet_deviation < 1e-7, (
        f"Folded PML run diverges from the full-domain run while the wrapped face is "
        f"dead: {quiet_deviation:.3e}")

    # Once the wave reaches the wrapped face, the stored plane serves it and the
    # equivalence stays at the same floor — the fold is the full system, not an
    # approximation that holds while the boundary is dark.
    live_deviation, live_face = _folded_pml_deviation(until=2.0)
    assert live_face > 100.0 * quiet_face, "the far face must actually be live in this case"
    assert live_deviation < 1e-7, (
        f"Folded PML run diverges from the full-domain run by {live_deviation:.3e} with the "
        f"wrapped face live at {live_face:.3e} — the stored second-mirror plane is not "
        f"reproducing the full domain's wrap.")


@requires_meep
@skip_without_meep
def test_a_folded_run_matches_cpu_meep_with_its_far_face_live(tmp_path):
    """The symmetry path, against CPU MEEP rather than against the engine's own full run.

    Every other folded case here is engine-vs-engine, which cannot see an error the two
    spellings share. MEEP's ``Mirror`` sits UNDER periodic boundaries, so a cell with a
    mirror at x = 0 has a second mirror at x = +/-L/2, and MEEP's ghost past the folded
    half is the parity-weighted image of an interior plane (boundaries.cpp
    ``connect_the_chunks`` -> ``locate_component_point``). This engine used to put a
    zero ghost there and refuse the run once that face lit up — the measured cost of
    serving it anyway was 8.8e-01 at a face amplitude of 9.1e-02 of peak. It now
    stores the second-mirror plane (one cell past MEEP's halved window,
    ``Grid.stored_cells``), fills the ghost slot past it with the parity-weighted
    image (``stepping.fill_folded_far_ghosts_*``), and reflects the beyond-window
    reads (``_shift_up``), so the fold must be at the CPU-MEEP floor with the face
    fully live:

      t = 1.0   far face 3.4e-05 of peak   folded 4.6e-06   unfolded 2.7e-07  (dark control)
      t = 3.0   far face 9.1e-02           folded at floor  unfolded 2.1e-07  (was 8.8e-01)

    The stored plane cell is compared against MEEP's own cell 0 one lattice vector
    down — the same wrap identification the bit-for-bit engine equivalences use.
    """
    def folded_and_full(until):
        reference = _meep_oracle(tmp_path, f"folded_cell:{until}")
        full = FdtdDriver(cell_size=(3, 3, 4), resolution=10, force_complex_fields=True)
        full.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                         "size": (0.0, 0.0, 0.0)})
        full.run(until=until)
        folded = FdtdDriver(cell_size=(3, 3, 4), resolution=10, force_complex_fields=True,
                            symmetry=("X",))
        folded.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                           "size": (0.0, 0.0, 0.0)})
        return reference, full, folded

    def folded_reference(reference, centre):
        # Stored rows 0..n-2 are MEEP indices centre..N; the plane cell is MEEP's
        # cell 0 (get_array row 1 — row 0 is the periodic duplicate plane).
        return np.concatenate((reference[centre:, 1:, 1:], reference[1:2, 1:, 1:]), axis=0)

    # 1. Dark far face: the fold reproduces MEEP, and the unfolded run of the same case
    #    is the control that says the residual belongs to the fold.
    reference, full, folded = folded_and_full(1.0)
    folded.run(until=1.0)
    centre = full.grid.nx // 2  # Quadrant cell q is full cell centre-1+q, i.e. MEEP index centre+q.
    full_error = _complex_relative_l2(full.get_field("Ez"), reference[1:, 1:, 1:])
    folded_error = _complex_relative_l2(folded.get_field("Ez"), folded_reference(reference, centre))
    assert full_error < 1e-6, f"the unfolded control is not at its floor: {full_error:.3e}"
    assert folded_error < 1e-5, (
        f"Folded run's complex Ez relative L2 against CPU MEEP is {folded_error:.3e} while its far "
        f"x face carries only {folded._folded_far_face_ratio()[1]:.3e} of peak; measured 4.6e-06."
    )

    # 2. Live far face: SERVED, at the same floor — and the face really is live, so
    #    this is the regime the old zero ghost measured 8.8e-01 in.
    reference, full, folded = folded_and_full(3.0)
    assert _complex_relative_l2(full.get_field("Ez"), reference[1:, 1:, 1:]) < 1e-6
    folded.run(until=3.0)
    ratio = folded._folded_far_face_ratio()[1]
    assert ratio > 1e-2, f"the far face carries {ratio:.3e}; this case is not exercising a live face"
    live_error = _complex_relative_l2(folded.get_field("Ez"), folded_reference(reference, centre))
    assert live_error < 1e-5, (
        f"Folded run's complex Ez relative L2 against CPU MEEP is {live_error:.3e} with the far "
        f"face at {ratio:.3e} of peak — the live-face fold must sit at the floor the zero ghost "
        f"missed by 8.8e-01."
    )


_FOLD_BLOCH_KY = 0.1234567  # Not a simple fraction of the 3.0 cell, and far from the zone edge.


@requires_meep
@skip_without_meep
def test_a_fold_and_a_bloch_phase_on_another_axis_match_cpu_meep(tmp_path, monkeypatch):
    """Mirror on X, Bloch on Y: the pairing that was refused, measured against MEEP.

    The refusal (deleted 2026-08-08) said the folded-axis cell-centred readback
    averaged across the mirror plane with no Bloch factor. The readback in fact
    branches PER AXIS: ``Fields.to_cell_center`` sends a mirrored axis down the fold
    branch, which needs no Bloch factor because a mirror pins that axis's k to zero
    (still refused, in ``_require_bloch_is_representable``), and every unfolded axis
    down ``Fields._wrapped_neighbour``, which applies ``grid.bloch_phase``. MEEP
    composes the two the same way and with no cross term (loop_in_chunks.cpp:415 for
    the lattice phase, vec.cpp:1347 ``symmetry::phase_shift``, multiplied together
    only at loop_in_chunks.cpp:202).

    Ez is the right component to compare: its Yee shift makes it interpolate across x
    and y on the way to cell centres, so the FOLDED axis and the BLOCH axis both
    appear in one readback. A Bloch phase on z would never reach ``_wrapped_neighbour``
    for Ez at all and the case would look green while testing nothing.

    Four controls, each answering a different way this could pass for the wrong reason
    (all numbers measured on this tree at resolution 10, t = 3, far x face live at
    9.4e-02 of peak):

    * MEEP's OWN folded run of the same system reproduces MEEP's unfolded run to
      2.6e-07 — stock MEEP serves mirror + Bloch, so the comparison below is against
      a supported configuration and not against an accident.
    * The engine's UNFOLDED run sits at 2.5e-07, which is where the residual would
      live if it belonged to the Bloch phase rather than to the fold.
    * A folded run at k = 0 — what an engine that accepted ``k_point`` and dropped it
      would produce — reads 8.5e-01 against the same reference.
    * THE MUTATION, and it is the one that matters: dropping the Bloch factor from
      ``_wrapped_neighbour`` moves BOTH legs to ~9.2e-02, and the folded leg lands on
      exactly the value it takes when the factor is dropped only on folded runs
      (9.321311259e-02 either way). That equality is the claim: the fold routes its
      Bloch axis through the same code the unfolded run uses and adds no separate
      hazard. See ``mutate_from_meep.py::bloch_factor_dropped_on_a_folded_run``.
    """
    from .fields import Fields

    reference = _meep_oracle(tmp_path, f"fold_bloch_full:{_FOLD_BLOCH_KY}")
    meep_folded = _meep_oracle(tmp_path, f"fold_bloch_folded:{_FOLD_BLOCH_KY}")
    # MEEP's folded array has no prepended periodic plane on the folded axis, so its
    # x rows are the unfolded array's rows 1..N.
    meep_self = _complex_relative_l2(meep_folded, reference[1:])
    assert meep_self < 1e-6, (
        f"CPU MEEP's own Mirror(X) + k_y run disagrees with its unfolded run by {meep_self:.3e}; "
        f"the configuration this test compares against is not one MEEP serves."
    )

    def build(symmetry, ky):
        driver = FdtdDriver(cell_size=(3, 3, 4), resolution=10, force_complex_fields=True,
                            symmetry=symmetry, k_point=(0.0, ky, 0.0))
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                           "size": (0.0, 0.0, 0.0)})
        driver.run(until=3.0)
        return driver

    def folded_reference(centre):
        # Stored rows 0..n-2 are MEEP indices centre..N; the last stored row is the
        # second-mirror plane cell, which is MEEP's cell 0 (get_array row 1).
        return np.concatenate((reference[centre:, 1:, 1:], reference[1:2, 1:, 1:]), axis=0)

    def errors():
        full = build((), _FOLD_BLOCH_KY)
        folded = build(("X",), _FOLD_BLOCH_KY)
        centre = full.grid.nx // 2
        return (_complex_relative_l2(full.get_field("Ez"), reference[1:, 1:, 1:]),
                _complex_relative_l2(folded.get_field("Ez"), folded_reference(centre)),
                float(folded._folded_far_face_ratio()[1]),
                centre)

    full_error, folded_error, ratio, centre = errors()
    assert ratio > 1e-2, f"the far x face carries {ratio:.3e}; the fold is not being exercised"
    assert full_error < 1e-6, f"the unfolded control is not at its floor: {full_error:.3e}"
    assert folded_error < 1e-5, (
        f"Mirror(X) with k_y={_FOLD_BLOCH_KY}: complex Ez relative L2 against CPU MEEP is "
        f"{folded_error:.3e} with the far x face at {ratio:.3e} of peak; measured 2.8e-06."
    )

    blind = build(("X",), 0.0)
    blind_error = _complex_relative_l2(blind.get_field("Ez"), folded_reference(centre))
    assert blind_error > 0.05, (  # Measured 8.5e-01.
        f"a folded run at k = 0 reproduces this reference to {blind_error:.3e}, so the comparison "
        f"cannot tell whether the Bloch phase was carried through the fold at all."
    )

    # The negative control that converges to a WRONG answer rather than erroring: the
    # engine still runs, still returns a smooth complete field, and is off by 9e-02.
    original = Fields._wrapped_neighbour
    monkeypatch.setattr(
        Fields, "_wrapped_neighbour",
        lambda self, array, axis: self.grid.xp.roll(array, -1, axis=axis))
    mutated_full, mutated_folded, _, _ = errors()
    monkeypatch.setattr(Fields, "_wrapped_neighbour", original)
    assert mutated_folded > 1e-2 and mutated_full > 1e-2, (
        f"dropping the Bloch factor from _wrapped_neighbour left the folded run at "
        f"{mutated_folded:.3e} and the unfolded run at {mutated_full:.3e}; this comparison does "
        f"not constrain the phase. Measured 9.32e-02 and 9.18e-02."
    )
    assert abs(mutated_folded - mutated_full) / mutated_full < 0.05, (
        f"the mutation moved the folded leg to {mutated_folded:.3e} and the unfolded leg to "
        f"{mutated_full:.3e}. They must move together — that is what says the fold carries no "
        f"Bloch hazard of its own, only the one the unfolded path already has."
    )


@requires_meep
@skip_without_meep
def test_odd_count_folded_run_matches_cpu_meeps_own_fold(tmp_path):
    """cavity-farfield.py's regime: metallic + PML, X and Y mirrors, ODD counts both.

    At an odd count MEEP still folds about a grid point — ``icenter()`` rounds the
    count down to even (vec.cpp:1089-1101) — and only the WINDOW shifts half a cell
    up: 21 cells span [-1.0, +1.1], the stored quadrant is 12 cells (halve() gives
    N - N//2 + 1), and the top stored centre sits at exactly +L/2. The oracle is
    MEEP's own FOLDED run of the same script, which is the system the engine
    reproduces; the engine used to refuse this outright, which is what kept
    cavity-farfield.py (nx_full=1541, ny_full=245, both folded, both odd,
    PML-backed metallic) from lifting at all.

    Both folded axes are METALLIC here, so the zero far ghost is the shifted
    window's own top wall and the far-face gate does not apply — unlike the
    folded-PERIODIC odd case, which lifts now but stops at the same live-far-face
    refusal its even-count siblings do (pinned below).
    """
    reference = _meep_oracle(tmp_path, "odd_folded_metallic_pml")
    assert reference.shape == (21, 21, 20)  # Folded axes at FULL count, metallic z at N.
    driver = FdtdDriver(cell_size=(2.1, 2.1, 2.0), resolution=10, force_complex_fields=True,
                        symmetry=("X", "Y"), boundaries="metallic")
    driver.setup_pml(5)  # mp.PML(0.5) at resolution 10.
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
                       "size": (0.0, 0.0, 0.0)})
    driver.run(until=3.0)
    assert driver.shape == (12, 12, 20)
    centre = 21 // 2  # Quadrant cell q is full-window cell centre - 1 + q; no leading
    # duplicate to skip on any axis: metallic MEEP arrays carry no periodic image.
    error = _complex_relative_l2(driver.get_field("Ez"), reference[centre - 1:, centre - 1:, :])
    assert error < 1e-5, (
        f"Odd-count folded run's complex Ez relative L2 against CPU MEEP's own fold is "
        f"{error:.3e}; the even-count folded floor is ~1e-06 and this case must sit at it."
    )


def test_odd_count_folded_periodic_axis_lifts_and_serves_its_live_far_face():
    """An odd folded PERIODIC axis constructs AND serves a live far face, bit for bit.

    binary_grating_phasemap.py (ny_full=21) and metasurface_lens.py (ny_full=15)
    fold a periodic transverse axis at an odd count. This case used to pin two
    successive family boundaries: first the odd-count lift refusal (fixed by the
    odd ``halve()`` window), then the far-face runtime gate (fixed by storing
    MEEP's ``big_corner`` cell, imaged by
    ``stepping.fill_folded_far_ghosts_*``). Both gone, what remains to pin is the
    positive statement: the folded run IS the full-domain run's stored rows
    ``cy - 1 .. cy - 2 + owned`` even with the far face carrying 1.3e-01 of peak,
    measured 0.0 bit for bit at this configuration.

    THE TOP STORED ROW HAS NO FULL-DOMAIN ROW OF ITS OWN. ``big_corner`` sits at
    doubled ``n_full + 1``, one cell above the full run's last row, so its
    reference is that run's row 0 one lattice vector down (k = 0, no phase). That
    row is what the odd count used to serve by reflecting about the second
    mirror, which is exact only where the medium is symmetric about it — the
    defect this pins the repair of. The refusal that survives is nonlinear-only
    (``test_driver_integration.py::
    test_a_live_far_face_is_refused_on_every_folded_axis``).
    """
    source = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
              "size": (0.0, 0.0, 0.0)}
    folded = FdtdDriver(cell_size=(2.0, 2.1, 2.0), resolution=10, force_complex_fields=True,
                        symmetry=("Y",))  # Periodic boundaries: the driver default.
    assert folded.grid.ny_full == 21 and folded.grid.ny == 13  # The lift does not refuse.
    assert folded.grid.owned_cells(1) == 12  # halve()'s num; the 13th is big_corner.
    folded.add_source(source)
    folded.run(until=3.0)  # And neither does the run, with its far face fully live:
    axis, ratio = folded._folded_far_face_ratio()
    assert axis == 1 and ratio > 1e-2, (
        f"the far Y face carries only {ratio:.3e}; this case is not exercising a live face")
    full = FdtdDriver(cell_size=(2.0, 2.1, 2.0), resolution=10, force_complex_fields=True)
    full.add_source(source)
    full.run(until=3.0)
    cy = full.shape[1] // 2
    owned = folded.grid.owned_cells(1)
    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        theirs = full.get_field(name, cell_centered=False)
        mine = np.asarray(folded.get_field(name, cell_centered=False))
        np.testing.assert_array_equal(
            mine[:, :owned, :], theirs[:, cy - 1: cy - 1 + owned, :],
            err_msg=f"{name}: the odd-count folded run with a live far face differs from "
                    f"the full-domain run's stored rows")
        np.testing.assert_array_equal(  # Measured 0.0 bit for bit, all six components.
            mine[:, owned, :], theirs[:, 0, :],
            err_msg=f"{name}: the big_corner row is not the full run's row 0 one lattice "
                    f"vector down")


def test_zero_amplitude_source_leaves_fields_exactly_zero():  # Degenerate case, with a positive control.
    quiet = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
    quiet.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": 0.0}
    )
    quiet.run(num_steps=10)
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        values = quiet.get_field(component, cell_centered=False)
        assert np.all(values == 0), f"{component} is nonzero after stepping a zero-amplitude source."
    driven = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
    driven.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": 1.0}
    )
    driven.run(num_steps=10)
    assert float(np.max(np.abs(driven.get_field("Ez", cell_centered=False)))) > 0.0


def test_run_requires_one_finite_stopping_condition():  # §5.12 guard: no open-ended runs, no silent defaults.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10)
    with pytest.raises(ValueError):
        driver.run()
    with pytest.raises(ValueError):
        driver.run(num_steps=10, until=1.0)
    with pytest.raises(ValueError):
        driver.run(until=float("inf"))
    with pytest.raises(ValueError):
        driver.run(until=0.0)


def test_cancel_check_stops_the_run():  # Cancellation is raised, not swallowed, and leaves the state intact.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    progress = []
    with pytest.raises(FdtdCancelled):
        driver.run(
            num_steps=40,
            progress_cb=lambda completed, total: progress.append((completed, total)),
            cancel_check=lambda: driver.step_count >= 10,
            progress_interval=5,
        )
    assert driver.step_count == 10
    assert progress == [(5, 40)]


def test_epsilon_shape_validation_checks_every_axis():  # §5.8: the half-matching array must raise, not be trimmed.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10)
    nx, ny, nz = driver.shape
    driver.set_epsilon(np.full((nx, ny, nz), 2.25, dtype=np.float32))
    assert np.allclose(driver.get_epsilon(), 2.25)
    driver.set_epsilon(np.full((nx + 1, ny + 1, nz + 1), 4.0, dtype=np.float32))
    assert driver.get_epsilon().shape == (nx, ny, nz)
    assert np.allclose(driver.get_epsilon(), 4.0)
    with pytest.raises(ValueError):
        driver.set_epsilon(np.full((nx + 1, ny, nz), 4.0, dtype=np.float32))
    with pytest.raises(ValueError):
        driver.set_epsilon(np.zeros((nx, ny, nz), dtype=np.float32))


def test_component_epsilon_mapping_is_loud_and_component_specific():
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.2), resolution=10)
    shape = driver.shape
    epsilon = {
        "Ex": np.full(shape, 2.0, dtype=np.float32),
        "Ey": np.full(shape, 3.0, dtype=np.float32),
        "Ez": np.full(shape, 4.0, dtype=np.float32),
    }
    driver.set_epsilon_components(epsilon)
    driver.fields.Dx.fill(2.0)
    driver.fields.Dy.fill(3.0)
    driver.fields.Dz.fill(4.0)

    for component in ("Ex", "Ey", "Ez"):
        np.testing.assert_array_equal(driver.fields.get_E(component), 1.0)
        np.testing.assert_array_equal(
            driver.get_epsilon(component=component),
            epsilon[component],
        )
    with pytest.raises(ValueError, match="missing"):
        driver.set_epsilon_components({"Ex": epsilon["Ex"], "Ey": epsilon["Ey"]})
    with pytest.raises(ValueError, match="unexpected"):
        driver.set_epsilon_components({**epsilon, "Hx": epsilon["Ex"]})
    with pytest.raises(ValueError, match=r"Epsilon\['Ey'\].*shape"):
        driver.set_epsilon_components(
            {**epsilon, "Ey": np.ones((shape[0] + 2, shape[1], shape[2]))}
        )


def test_unknown_names_raise_rather_than_defaulting():  # Loud errors instead of the bundle's silent fallbacks.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10)
    # 'Z' used to belong here, back when Z mirrors were refused. All three axes fold
    # now (see the Z fold-equivalence and CPU-MEEP cases), so the loud-error case
    # moved to an axis name that names no direction at all; the positive half — a Z
    # fold reproducing the full domain exactly — is asserted there.
    with pytest.raises(ValueError, match="Unknown symmetry axis"):
        FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, symmetry=("W",))
    assert FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10,
                      symmetry=("Z",)).grid.is_mirrored(2)
    # 'Hz' used to belong on this list, back when current sources drove D only. Magnetic
    # sources are implemented and cross-validated now (see the Hx oracle in
    # test_driver_integration.py), so the loud-error case moved to a component name that
    # names no field at all; the positive half is asserted there.
    with pytest.raises(ValueError):
        driver.add_source({"component": "Jz", "frequency": 1.0})
    with pytest.raises(ValueError):
        driver.add_source({"component": "Ez", "frequency": 1.0, "source_type": "sinusoid"})
    with pytest.raises(ValueError):
        driver.add_source({"component": "Ez", "frequency": 1.0, "fwidth": 0.2})
    with pytest.raises(ValueError):
        driver.get_field("Sz")
    with pytest.raises(ValueError):
        driver.add_flux_monitor(1.0, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))


def test_gpu_request_never_degrades_silently():  # prefer_gpu resolves this host's GPU or raises.
    from .backends import available_gpu

    gpu = available_gpu()
    if gpu is None:
        with pytest.raises(RuntimeError):
            FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, prefer_gpu=True)
    else:
        driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10, prefer_gpu=True)
        assert driver.gpu == gpu
        assert driver.xp.__name__ == ("cupy" if gpu == "cuda" else "numpy")
    assert FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=10).gpu is None


def test_gaussian_source_monitors_and_lifecycle():  # The remaining driver paths, on one dielectric run.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
    driver.setup_pml(3)
    monitor = driver.add_dft_monitor(1.0, components=("Ez", "Hx"), center=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.0))
    downstream_flux = driver.add_flux_monitor(1.0, center=(0.0, 0.0, 0.3), size=(0.4, 0.4, 0.0))
    source = driver.add_source(
        {
            "component": "Ez",
            "frequency": 1.0,
            "center": (0.0, 0.0, -0.3),
            "size": (0.4, 0.4, 0.0),
            "source_type": "gaussian",
            "fwidth": 1.0,
        }
    )
    # Timing is read off the source itself, so the check pins the envelope's shape
    # rather than a particular width convention: quiet before the pulse, energetic after.
    driver.run(until=source.peak_time - 4.0 * source.width)
    quiet_field = float(np.max(np.abs(driver.get_field("Ez"))))
    driver.run(until=source.peak_time + source.width)
    driven_field = float(np.max(np.abs(driver.get_field("Ez"))))
    assert driven_field > 100.0 * quiet_field, "Gaussian pulse did not turn on around its peak time."
    x0, x1, y0, y1, z0, z1 = monitor.region
    assert monitor.get_dft("Ez").shape == (x1 - x0, y1 - y0, z1 - z0)
    assert np.all(np.isfinite(monitor.get_dft("Ez")))
    flux = downstream_flux.get_flux()
    assert np.isfinite(flux) and flux > 0.0, "Flux through a plane downstream of the source should be positive."
    driver.reset()
    assert driver.step_count == 0 and driver.time == 0.0
    assert float(np.max(np.abs(driver.get_field("Ez")))) == 0.0
    assert float(np.max(np.abs(monitor.get_dft("Ez")))) == 0.0
    driver.close()
    with pytest.raises(RuntimeError):
        driver.step()


def test_monitor_region_default_is_independent_of_creation_order():  # §5.14: PML-aware defaults resolve at run start.
    before = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, force_complex_fields=True)
    monitor_before = before.add_dft_monitor(1.0, components=("Ez",))
    before.setup_pml(4)
    after = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, force_complex_fields=True)
    after.setup_pml(4)
    monitor_after = after.add_dft_monitor(1.0, components=("Ez",))
    for driver in (before, after):
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
        driver.run(num_steps=5)
    assert monitor_before.region == monitor_after.region
    assert monitor_before.get_dft("Ez").shape == monitor_after.get_dft("Ez").shape
    interior = (4, before.shape[0] - 4, 4, before.shape[1] - 4, 4, before.shape[2] - 4)
    assert tuple(monitor_before.region) == interior


# --- Several monitors in one run -------------------------------------------------

_SHEET_SOURCE = {  # Symmetric Ez sheet in the z = 0 plane; radiates equally into +z and -z.
    "component": "Ez",
    "frequency": 1.0,
    "center": (0.0, 0.0, 0.0),
    "size": (0.6, 0.6, 0.0),
}
_DFT_MONITOR_SPECS = (  # (frequency, centre, size) — three distinct frequencies at three depths.
    (0.75, (0.0, 0.0, -0.6), (0.4, 0.4, 0.0)),
    (1.00, (0.0, 0.0, 0.0), (0.4, 0.4, 0.0)),
    (1.25, (0.0, 0.0, 0.6), (0.4, 0.4, 0.0)),
)


def _monitor_run():  # A PML-terminated box wide enough to hold the sheet source clear of the absorber.
    driver = FdtdDriver(cell_size=(3.0, 3.0, 4.0), resolution=10, force_complex_fields=True)
    driver.setup_pml(8)  # Interior |x|, |y| <= 0.7, so the 0.6-wide sheet stays clear of the layer.
    return driver


def _monitor_frequencies(monitor):  # Frequency tuple of a monitor, on either dft.py generation.
    frequencies = getattr(monitor, "frequencies", None)
    return tuple(frequencies) if frequencies is not None else (monitor.frequency,)


def test_three_dft_monitors_accumulate_their_own_region_and_frequency():  # Monitors must not share state.
    # Each monitor is compared against a run in which it is the only monitor: same
    # physics, same accumulation, so agreement must be exact (measured 0.0 relative on
    # every component). A driver that pointed every monitor at one region, or updated
    # only the first, would still fill all three with plausible numbers — this is what
    # separates that from a correct run. The three regions and the three DFTs are also
    # asserted to differ, so the comparison cannot be satisfied by three copies of one
    # answer.
    together = _monitor_run()
    monitors = [
        together.add_dft_monitor(frequencies=frequency, components=("Ez", "Hx"), center=center, size=size)
        for frequency, center, size in _DFT_MONITOR_SPECS
    ]
    together.add_source(_SHEET_SOURCE)
    together.run(until=6.0)
    assert len({tuple(monitor.region) for monitor in monitors}) == 3, "The three monitors share a region."
    reference_scale = float(np.linalg.norm(monitors[1].get_dft("Ez")))
    assert reference_scale > 0.0, "The monitored run carries no field; the comparison would be vacuous."
    for index, (frequency, center, size) in enumerate(_DFT_MONITOR_SPECS):
        alone = _monitor_run()
        solo = alone.add_dft_monitor(frequencies=frequency, components=("Ez", "Hx"), center=center, size=size)
        alone.add_source(_SHEET_SOURCE)
        alone.run(until=6.0)
        assert _monitor_frequencies(monitors[index]) == (frequency,)
        assert tuple(monitors[index].region) == tuple(solo.region)
        for component in ("Ez", "Hx"):
            shared = np.asarray(monitors[index].get_dft(component))
            reference = np.asarray(solo.get_dft(component))
            deviation = float(np.linalg.norm(shared - reference))
            assert deviation == 0.0, (
                f"Monitor {index} ({component}) differs from the same monitor run alone by "
                f"{deviation:.3e}; monitors are interfering."
            )
    for first, second in ((0, 1), (1, 2), (0, 2)):
        difference = float(np.linalg.norm(
            np.asarray(monitors[first].get_dft("Ez")) - np.asarray(monitors[second].get_dft("Ez"))
        ))
        assert difference > 1e-3 * reference_scale, (
            f"Monitors {first} and {second} accumulated the same data ({difference:.3e}); they were "
            f"asked for different frequencies over different regions."
        )


def test_source_side_and_transmission_flux_monitors_coexist():  # The practical two-plane measurement.
    # Two flux planes equidistant from a symmetric sheet source must carry equal and
    # opposite power: an analytic property of the configuration, not a recorded number.
    # The planes sit on cell centres that are mirror partners about the source plane
    # (a plane requested on a cell boundary resolves to the cell above it, which is
    # consistent but not mirror-symmetric, and would blur the comparison). Measured
    # asymmetry 0.0 — bit-exact. A driver that aimed both monitors at one plane would
    # produce two equal fluxes of the same sign and fail here.
    driver = _monitor_run()
    upstream = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, -0.45), size=(1.0, 1.0, 0.0))
    downstream = driver.add_flux_monitor(frequencies=1.0, center=(0.0, 0.0, 0.45), size=(1.0, 1.0, 0.0))
    detuned = driver.add_flux_monitor(frequencies=1.6, center=(0.0, 0.0, 0.45), size=(1.0, 1.0, 0.0))
    driver.add_source(_SHEET_SOURCE)
    driver.run(until=8.0)
    upstream_flux, downstream_flux = upstream.get_flux(), downstream.get_flux()
    assert downstream_flux > 0.0, "Power must flow away from the source on the +z side."
    assert upstream_flux < 0.0, "Power must flow away from the source on the -z side."
    asymmetry = abs(upstream_flux + downstream_flux) / abs(downstream_flux)
    assert asymmetry < 1e-6, (
        f"Equidistant planes either side of a symmetric sheet source carry {upstream_flux:.6e} and "
        f"{downstream_flux:.6e}, an asymmetry of {asymmetry:.3e}."
    )
    # The off-frequency plane shares its position with the downstream one and must still
    # hold its own number: same region, different frequency, different flux.
    off_frequency_flux = detuned.get_flux()
    assert np.isfinite(off_frequency_flux)
    assert abs(off_frequency_flux - downstream_flux) > 1e-3 * abs(downstream_flux), (
        "The 1.6-frequency plane reproduced the 1.0-frequency flux at the same position; the "
        "monitors are sharing one accumulator."
    )
    # And each must equal what it measures when it is the only monitor in the run.
    for label, frequency, center, expected in (
        ("upstream", 1.0, (0.0, 0.0, -0.45), upstream_flux),
        ("downstream", 1.0, (0.0, 0.0, 0.45), downstream_flux),
        ("detuned", 1.6, (0.0, 0.0, 0.45), off_frequency_flux),
    ):
        alone = _monitor_run()
        solo = alone.add_flux_monitor(frequencies=frequency, center=center, size=(1.0, 1.0, 0.0))
        alone.add_source(_SHEET_SOURCE)
        alone.run(until=8.0)
        assert solo.get_flux() == expected, f"The {label} plane changed when other monitors were present."


def test_one_broadband_monitor_equals_a_monitor_per_frequency():  # The spectrum axis carries what it says.
    # A monitor asked for three frequencies must accumulate exactly what three monitors
    # asked for one frequency each accumulate, index for index. This is the driver-side
    # pin on the shared spectrum contract: a forwarded list that was truncated,
    # reordered, or collapsed to its first entry would still produce a full, plausible
    # spectrum. Measured 0.0 difference on every index for both monitor kinds.
    band = (0.75, 1.0, 1.25)
    region = {"center": (0.0, 0.0, 0.45), "size": (0.4, 0.4, 0.0)}
    plane = {"center": (0.0, 0.0, 0.45), "size": (1.0, 1.0, 0.0)}
    driver = _monitor_run()
    spectrum = driver.add_dft_monitor(frequencies=band, components=("Ez",), **region)
    singles = [driver.add_dft_monitor(frequencies=f, components=("Ez",), **region) for f in band]
    flux_spectrum = driver.add_flux_monitor(frequencies=band, **plane)
    flux_singles = [driver.add_flux_monitor(frequencies=f, **plane) for f in band]
    driver.add_source(_SHEET_SOURCE)
    driver.run(until=6.0)
    assert _monitor_frequencies(spectrum) == band
    assert spectrum.get_dft_spectrum("Ez").shape[0] == len(band)
    assert float(np.linalg.norm(spectrum.get_dft("Ez", freq_index=1))) > 0.0
    for index, frequency in enumerate(band):
        difference = float(np.linalg.norm(
            np.asarray(spectrum.get_dft("Ez", freq_index=index)) - np.asarray(singles[index].get_dft("Ez"))
        ))
        assert difference == 0.0, (
            f"Spectrum index {index} (f = {frequency}) differs from its own single-frequency monitor "
            f"by {difference:.3e}."
        )
        assert flux_spectrum.get_flux(freq_index=index) == flux_singles[index].get_flux(), (
            f"Flux spectrum index {index} (f = {frequency}) differs from its single-frequency monitor."
        )
    np.testing.assert_allclose(
        flux_spectrum.get_flux_spectrum(),
        [monitor.get_flux() for monitor in flux_singles],
        rtol=0.0,
        atol=0.0,
    )


def test_monitor_frequency_specifications_expand_the_way_meep_does():  # Scalar, sequence, and fcen/df/nfreq.
    assert _normalize_frequencies(1.25) == (1.25,)
    assert _normalize_frequencies([0.8, 1.0, 1.2]) == (0.8, 1.0, 1.2)
    assert _normalize_frequencies(np.array([0.8, 1.2])) == (0.8, 1.2)
    assert _normalize_frequencies(frequency=2.5) == (2.5,)
    spread = _normalize_frequencies(fcen=1.0, df=0.4, nfreq=5)
    np.testing.assert_allclose(spread, [0.8, 0.9, 1.0, 1.1, 1.2], rtol=0.0, atol=1e-12)
    # MEEP's own expansion returns the centre for a one-point spectrum, so (fcen, df, 1)
    # lands on fcen. numpy.linspace(fcen - df/2, fcen + df/2, 1) would return 0.8 here
    # and detune the monitor by half the bandwidth in silence.
    assert _normalize_frequencies(fcen=1.0, df=0.4, nfreq=1) == (1.0,)
    # Negative monitor frequencies are stored raw, as MEEP stores them
    # (dft.cpp:219-221 keeps omega signed; the -f bin is the conjugate of +f).
    assert _normalize_frequencies([1.0, -1.0]) == (1.0, -1.0)
    # ZERO is a monitor frequency too: `update_dft` has no branch on omega
    # (dft.cpp:266-269), so the DC bin's phase factor is exactly 1 and the
    # accumulator is a running time integral of the field. Measured against MEEP's
    # own accumulator at 1.35e-06 ABSOLUTE, the same floor its 0.8-bin neighbour
    # sits on (2.61e-06) — see test_from_meep's DC monitor case.
    assert _normalize_frequencies([1.0, 0.0]) == (1.0, 0.0)
    for bad in (
        {},  # No specification at all.
        {"frequencies": 1.0, "fcen": 1.0, "df": 0.2, "nfreq": 3},  # Two specifications.
        {"frequencies": 1.0, "frequency": 1.0},
        {"fcen": 1.0, "df": 0.2},  # Incomplete triple.
        {"fcen": 1.0, "df": 0.2, "nfreq": 0},
        {"fcen": 1.0, "df": -0.2, "nfreq": 3},
        {"fcen": 1.0, "df": 0.0, "nfreq": 3},  # Three points of zero bandwidth are one frequency.
        {"frequencies": []},
        {"frequencies": [1.0, float("inf")]},
        {"frequencies": "1.0"},
    ):
        with pytest.raises(ValueError):
            _normalize_frequencies(**bad)


def test_driver_accepts_every_monitor_frequency_spelling():  # The same spellings through the driver.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
    scalar = driver.add_dft_monitor(1.0, components=("Ez",))
    assert _monitor_frequencies(scalar) == (1.0,)
    flux = driver.add_flux_monitor(fcen=1.0, df=0.4, nfreq=1, center=(0.0, 0.0, 0.2), size=(0.4, 0.4, 0.0))
    assert _monitor_frequencies(flux) == (1.0,)
    # A multi-frequency monitor must carry the whole list through to the accumulators:
    # keeping one frequency of several would hand back a plausible single-frequency
    # answer under a multi-frequency request.
    broadband = driver.add_dft_monitor(frequencies=(0.8, 1.0, 1.2), components=("Ez",))
    assert _monitor_frequencies(broadband) == (0.8, 1.0, 1.2)
    assert broadband.get_dft_spectrum("Ez").shape[0] == 3
    assert broadband.get_dft("Ez", freq_index=2).shape == broadband.get_dft("Ez", freq_index=0).shape
    spectrum = driver.add_flux_monitor(
        fcen=1.0, df=0.4, nfreq=3, center=(0.0, 0.0, 0.2), size=(0.4, 0.4, 0.0)
    )
    np.testing.assert_allclose(_monitor_frequencies(spectrum), [0.8, 1.0, 1.2], rtol=0.0, atol=1e-12)
    assert spectrum.get_flux_spectrum().shape == (3,)


# --- Field-decay stopping condition ----------------------------------------------

_DECAY_PROBE = (0.0, 0.0, 0.55)  # An exact Ez sample: integer in x/y, a cell centre in z.
_DECAY_WINDOW = 2.0
_DECAY_BY = 1e-4
# MEEP gaussian_src_time::last_time = peak_time + cutoff, with peak_time = start_time +
# cutoff*width and width = 1/fwidth: 5.0 + 5.0*1.0 = 10.0 for the pulse below.
_DECAY_SOURCE_END = 10.0
_DECAY_SPECTRUM = (0.75, 1.0, 1.25)


def _decay_case():  # A pulsed run in a PML box, with a three-frequency spectrum monitor set.
    driver = FdtdDriver(cell_size=(2.0, 2.0, 4.0), resolution=10, force_complex_fields=True)
    driver.setup_pml(8)
    monitors = [
        driver.add_dft_monitor(frequency, components=("Ez",), center=(0.0, 0.0, 0.5), size=(0.2, 0.2, 0.0))
        for frequency in _DECAY_SPECTRUM
    ]
    driver.add_source(
        {
            "component": "Ez",
            "frequency": 1.0,
            "center": (0.0, 0.0, -0.5),
            "size": (0.0, 0.0, 0.0),
            "source_type": "gaussian",
            "fwidth": 1.0,
        }
    )
    return driver, monitors


def _ez_probe_index(driver, point):  # Yee index of the Ez sample nearest a point.
    # grid.py's staggering puts Ez at (i, j, k+0.5)*dx from the corner -L/2, so its x and
    # y samples are the cell edges -L/2 + i*dx while its z samples are the cell centres.
    # Derived from that registration rather than from the driver's own index helper, so
    # the probe is not being checked against itself.
    ix = int(round((point[0] + driver.grid.Lx / 2) / driver.dx))
    iy = int(round((point[1] + driver.grid.Ly / 2) / driver.dx))
    iz = int(np.argmin(np.abs(np.asarray(driver.grid.z) - point[2])))
    return ix, iy, iz


def _meep_decay_stop(times, magnitudes, window, decay_by, source_end_time):
    """First time MEEP's ``stop_when_fields_decayed`` fires on a recorded probe history.

    Transcription of the closure in meep/simulation.py — the peak of ``|f|**2`` over a
    trailing window of ``window``, compared against the largest completed window peak —
    plus the ``round_time() >= last_source_time()`` gate that ``_run_sources_until``
    wraps around it. Evaluated offline on a history recorded through ``get_field``, so
    it shares no machinery with the driver's own probe.

    Returns:
        (stop_time, ratio) at the first firing, or (None, None) if it never fires.
    """
    window_start, window_peak, running_peak = 0.0, 0.0, 0.0
    for time, magnitude in zip(times, magnitudes):
        window_peak = max(window_peak, magnitude ** 2)
        if time <= window_start + window:
            continue
        completed, window_peak, window_start = window_peak, 0.0, time
        running_peak = max(running_peak, completed)
        if running_peak > 0.0 and time >= source_end_time and completed <= running_peak * decay_by:
            return time, completed / running_peak
    return None, None


def test_field_decay_stop_matches_the_meep_criterion_and_the_converged_spectrum():
    # The decay run is checked three ways: it stops exactly where a transcription of
    # MEEP's criterion says it should, the field there really has decayed (ratio
    # measured from the independently recorded probe history), and the spectrum it
    # accumulated matches a run more than three times longer. Measured: stop at
    # t = 12.30 (246 steps) against a fixed run to t = 40 (800 steps, 3.25x), a window
    # ratio of 1.9e-08 against the 1e-04 asked for, and at most 6.8e-07 relative L2
    # across the three monitored frequencies.
    decayed, decay_monitors = _decay_case()
    decayed.run(
        until_after_sources=_DECAY_WINDOW,
        decay_by=_DECAY_BY,
        decay_component="Ez",
        decay_point=_DECAY_PROBE,
        max_time=200.0,
    )
    long_run, long_monitors = _decay_case()
    probe_index = _ez_probe_index(long_run, _DECAY_PROBE)
    times, magnitudes = [], []
    while long_run.time < 20.0:  # Record the probe history one step at a time, then finish the run.
        long_run.run(num_steps=1)
        times.append(long_run.time)
        magnitudes.append(abs(long_run.get_field("Ez", cell_centered=False)[probe_index]))
    long_run.run(until=40.0)

    expected_stop, ratio = _meep_decay_stop(
        times, magnitudes, _DECAY_WINDOW, _DECAY_BY, _DECAY_SOURCE_END
    )
    assert expected_stop is not None, "The recorded history never satisfies the criterion; the case is broken."
    assert decayed.time == pytest.approx(expected_stop, abs=0.5 * decayed.dt), (
        f"The driver stopped at t = {decayed.time:.3f}, MEEP's criterion at t = {expected_stop:.3f}."
    )
    assert ratio <= _DECAY_BY, f"Stopped at a window ratio of {ratio:.3e}, above the {_DECAY_BY:.0e} asked for."
    assert max(magnitudes) > 0.0 and ratio < 1.0, "The probe never measured a field to decay from."
    assert decayed.time > _DECAY_SOURCE_END, (
        f"Stopped at t = {decayed.time:.3f}, before the source finished at t = {_DECAY_SOURCE_END}."
    )
    assert decayed.step_count < 0.5 * long_run.step_count, (
        f"The decay stop saved nothing: {decayed.step_count} steps against {long_run.step_count}."
    )
    for index, frequency in enumerate(_DECAY_SPECTRUM):
        short = np.asarray(decay_monitors[index].get_dft("Ez"))
        full = np.asarray(long_monitors[index].get_dft("Ez"))
        scale = float(np.linalg.norm(full))
        assert scale > 0.0, f"The reference run accumulated nothing at f = {frequency}."
        deviation = float(np.linalg.norm(short - full)) / scale
        assert deviation < 1e-4, (
            f"Stopping on decay changed the f = {frequency} DFT by {deviation:.3e} relative to the "
            f"full-length run; the spectrum was not converged when the run stopped."
        )


def _quiet_decay_case(amplitude):  # A small pulsed run whose amplitude can be turned off.
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
    driver.setup_pml(3)
    driver.add_source(
        {
            "component": "Ez",
            "frequency": 1.0,
            "center": (0.0, 0.0, 0.0),
            "size": (0.0, 0.0, 0.0),
            "source_type": "gaussian",
            "fwidth": 2.0,  # width 0.5, peak at 2.5, last emission at 5.0.
            "amplitude": amplitude,
        }
    )
    return driver


def test_field_decay_stop_never_passes_off_an_unmeasured_run_as_decayed():  # The degenerate case.
    # MEEP's criterion is `window_peak <= running_peak * decay_by`, which is satisfied
    # by 0 <= 0 the moment the first window closes on a probe that never saw anything:
    # a run with no field would stop immediately and report a perfectly decayed one.
    # Here it must run to the ceiling and raise, naming what it measured. The positive
    # control is the same run with the source switched on, which must stop on its own.
    quiet = _quiet_decay_case(amplitude=0.0)
    with pytest.raises(RuntimeError) as raised:
        quiet.run(
            until_after_sources=1.0,
            decay_by=1e-3,
            decay_component="Ez",
            decay_point=(0.0, 0.0, 0.3),
            max_time=8.0,
        )
    assert "decay" in str(raised.value).lower()
    assert quiet.time == pytest.approx(8.0, abs=quiet.dt), "The ceiling must be reached before raising."
    assert float(np.max(np.abs(quiet.get_field("Ez")))) == 0.0

    driven = _quiet_decay_case(amplitude=1.0)
    driven.run(
        until_after_sources=1.0,
        decay_by=1e-3,
        decay_component="Ez",
        decay_point=(0.0, 0.0, 0.3),
        max_time=8.0,
    )
    assert driven.time < 8.0, "The driven control never decayed; the degenerate case proves nothing."
    assert driven.time >= 5.0, "Stopped before the pulse had finished emitting."


def _yee_index(driver, shifts, point):  # Index of the sample a component with these Yee shifts puts at `point`.
    # grid.py: a component with Yee shift s on an axis sits at -L/2 + (i + s/2)*dx, so
    # i = (pos + L/2)/dx - s/2. The probe points below land exactly on a sample, so the
    # rounding is not deciding anything.
    lengths = (driver.grid.Lx, driver.grid.Ly, driver.grid.Lz)
    return tuple(
        int(round((point[axis] + lengths[axis] / 2) / driver.dx - 0.5 * shifts[axis])) for axis in range(3)
    )


def test_decay_probe_reads_the_same_value_get_field_reports():  # The per-step probe, against the public accessor.
    # The decay probe cannot go through get_field: that interpolates and, without PML,
    # allocates the whole volume every step. It reads one cell instead, deriving E = D/eps
    # itself when no PML is active — so the value it reads is checked here against what
    # get_field reports for the same cell, on a run whose epsilon is not 1 (a probe that
    # returned |D| instead of |E| would be four times too large and invisible to any
    # ratio-based test).
    driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 4.0, dtype=np.float32))
    nx, ny, nz = driver.shape
    driver.set_field("Dz", _symmetric_seed(nx, ny, nz, driver.dx))
    driver.run(num_steps=6)
    for component, shifts, point in (
        ("Ez", (0, 0, 1), (0.1, 0.1, 0.15)),  # E without PML: D/eps, evaluated at one cell.
        ("Dz", (0, 0, 1), (0.1, 0.1, 0.15)),  # A primary field: read straight out of storage.
        ("Hx", (0, 1, 1), (0.1, 0.15, 0.15)),  # H without PML is numerically B.
    ):
        index = _yee_index(driver, shifts, point)
        expected = abs(driver.get_field(component, cell_centered=False)[index])
        assert expected > 0.0, f"The seeded run leaves {component} zero at the probe; nothing is being compared."
        measured = float(driver._point_magnitude_reader(component, point)())
        assert measured == pytest.approx(expected, rel=1e-6), (
            f"The decay probe reads |{component}| = {measured:.6e} where get_field reports {expected:.6e}."
        )
    with_pml = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, force_complex_fields=True)
    with_pml.setup_pml(4)
    with_pml.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    with_pml.run(num_steps=12)
    point = (0.1, 0.1, 0.15)
    index = _yee_index(with_pml, (0, 0, 1), point)
    expected = abs(with_pml.get_field("Ez", cell_centered=False)[index])
    assert expected > 0.0
    assert float(with_pml._point_magnitude_reader("Ez", point)()) == pytest.approx(expected, rel=1e-6)


def test_decay_stop_works_on_a_seeded_run_with_no_sources():  # No source at all, just a seeded pulse.
    # There need not be a source: with none, "after the sources have finished" is
    # satisfied from the first step, and the criterion measures a seeded pulse passing
    # the probe and being absorbed. Measured stop at t = 5.90 of a 12.0 ceiling, with
    # |Ez| at the probe down to 0.37% of what it peaked at as the pulse went by.
    driver = FdtdDriver(cell_size=(3.0, 3.0, 3.0), resolution=10, force_complex_fields=True)
    driver.setup_pml(6)
    nx, ny, nz = driver.shape
    driver.set_field("Dz", _transverse_seed(nx, ny, nz, driver.dx))
    probe = (0.7, 0.0, 0.05)  # Off to the side of the seed, still inside the PML interior.
    index = _yee_index(driver, (0, 0, 1), probe)
    passing = 0.0
    for _ in range(20):  # Let the pulse reach the probe, recording the peak it makes there.
        driver.run(num_steps=1)
        passing = max(passing, abs(driver.get_field("Ez", cell_centered=False)[index]))
    assert passing > 0.0, "The seeded pulse never reached the probe; the decay stop would be vacuous."
    driver.run(
        until_after_sources=0.5,
        decay_by=1e-4,
        decay_component="Ez",
        decay_point=probe,
        max_time=12.0,
    )
    assert driver.time < 12.0, f"Never decayed inside the ceiling (stopped at t = {driver.time})."
    residual = abs(driver.get_field("Ez", cell_centered=False)[index])
    assert residual < 0.05 * passing, (
        f"Stopped with |Ez| = {residual:.3e} at the probe against {passing:.3e} as it passed; not decayed."
    )


def test_field_decay_stop_rejects_runs_that_could_never_terminate():  # Loud, and before the long run.
    # A continuous source ends at ~1e20, so "after the sources have finished" can never
    # arrive: the run must be refused up front rather than stepping the whole ceiling
    # first and only then admitting it.
    continuous = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
    continuous.setup_pml(3)
    continuous.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)}
    )
    with pytest.raises(ValueError) as raised:
        continuous.run(
            until_after_sources=1.0,
            decay_by=1e-3,
            decay_component="Ez",
            decay_point=(0.0, 0.0, 0.3),
            max_time=20.0,
        )
    assert "end_time" in str(raised.value)
    assert continuous.step_count == 0, "The run was refused, so nothing should have been stepped."


def test_decay_stop_arguments_are_validated_together():  # No half-specified decay stops.
    driver = _quiet_decay_case(amplitude=1.0)
    point = (0.0, 0.0, 0.3)
    for arguments in (
        {"until_after_sources": 1.0},  # No decay_by / component / point / ceiling.
        {"until_after_sources": 1.0, "decay_by": 1e-3, "decay_component": "Ez", "decay_point": point},
        {"until_after_sources": 1.0, "decay_by": 1e-3, "decay_point": point, "max_time": 8.0},
        {"until_after_sources": 0.0, "decay_by": 1e-3, "decay_component": "Ez",
         "decay_point": point, "max_time": 8.0},
        {"until_after_sources": 1.0, "decay_by": 0.0, "decay_component": "Ez",
         "decay_point": point, "max_time": 8.0},
        {"until_after_sources": 1.0, "decay_by": 1.5, "decay_component": "Ez",
         "decay_point": point, "max_time": 8.0},
        {"until_after_sources": 1.0, "decay_by": 1e-3, "decay_component": "Sz",
         "decay_point": point, "max_time": 8.0},
        {"until_after_sources": 1.0, "decay_by": 1e-3, "decay_component": "Ez",
         "decay_point": (0.0, 0.0, 99.0), "max_time": 8.0},
        {"until_after_sources": 1.0, "decay_by": 1e-3, "decay_component": "Ez",
         "decay_point": point, "max_time": float("inf")},
        {"until": 1.0, "until_after_sources": 1.0, "decay_by": 1e-3, "decay_component": "Ez",
         "decay_point": point, "max_time": 8.0},
        {"num_steps": 5, "decay_by": 1e-3},  # Decay arguments without the decay stop.
        {"until": 1.0, "max_time": 8.0},
    ):
        with pytest.raises(ValueError):
            driver.run(**arguments)
    assert driver.step_count == 0, "A rejected run must not have stepped."


def test_decay_run_still_honours_cancellation_and_progress():  # Cancellation wins over the decay stop.
    driver = _quiet_decay_case(amplitude=1.0)
    progress = []
    with pytest.raises(FdtdCancelled):
        driver.run(
            until_after_sources=1.0,
            decay_by=1e-3,
            decay_component="Ez",
            decay_point=(0.0, 0.0, 0.3),
            max_time=8.0,
            progress_cb=lambda completed, total: progress.append((completed, total)),
            cancel_check=lambda: driver.step_count >= 20,
            progress_interval=10,
        )
    assert driver.step_count == 20
    ceiling_steps = int(round(8.0 / driver.dt))
    assert progress == [(10, ceiling_steps)], (
        f"Progress under a decay stop should count against the ceiling's {ceiling_steps} steps, got {progress}."
    )


# --- PML thickness and source placement ------------------------------------------


def test_pml_thickness_is_validated_per_axis_with_the_faces_that_axis_carries():
    # A mirror-folded axis stores one absorbing face, not two: its lower boundary is the
    # mirror plane, which PML.__post_init__ already counts correctly. The driver used to
    # validate with min(nx, ny, nz) and two faces everywhere and so refused legal
    # symmetric configurations. Same cell, same thickness, and only the symmetry differs.
    folded = FdtdDriver(cell_size=(1.6, 2.0, 2.0), resolution=10, force_complex_fields=True, symmetry=("X",))
    # 16 full x cells fold to 16//2 + 1 owned cells, plus the second-mirror plane
    # and its ghost slot (periodic fold at an even count).
    assert folded.shape == (10, 20, 20) and folded.grid.owned_cells(0) == 9
    folded.setup_pml(8)
    assert folded.pml.thickness == 8
    assert folded.pml.interior_slice()[:2] == (0, 1), "A folded axis keeps its cells at the mirror plane."
    full = FdtdDriver(cell_size=(1.6, 2.0, 2.0), resolution=10, force_complex_fields=True)
    assert full.shape == (16, 20, 20)
    with pytest.raises(ValueError, match="X axis"):
        full.setup_pml(8)  # Two faces of 8 cells cover all 16.
    with pytest.raises(ValueError, match="X axis"):
        # One face still has to leave an interior: 9 cells of PML on 9 stored cells.
        FdtdDriver(cell_size=(1.6, 2.0, 2.0), resolution=10, symmetry=("X",)).setup_pml(9)
    # The accepted folded layer has to work, not merely be accepted.
    folded.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    folded.run(num_steps=20)
    field = folded.get_field("Ez")
    assert np.all(np.isfinite(field)) and float(np.max(np.abs(field))) > 0.0


def test_source_in_the_pml_is_admitted_on_cartesian_and_dcyl_either_order():
    # The source/absorber guard is RETIRED on both grids: a deep source is accepted
    # from either creation order and the run steps, because the deposit is mirrored
    # into f_u wherever MEEP's chunking runs the unsplit recurrence and the
    # integrated dipole is withdrawn before every ladder. Measured on the evidence
    # packet's cell (12x12 um, res 20, CW Ez, until 20, whole-volume complex L2 vs
    # CPU MEEP): 4.9581e-01 -> 5.5710e-07 for the x-PML non-integrated case, with
    # the corner / y-PML / interior controls bit-unchanged; the full parity cases
    # live in test_from_meep (source_spans_the_pml, source_on_the_wrapped_pml_seam).
    deep = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -2.3), "size": (0.4, 0.4, 0.0)}
    pml_first = FdtdDriver(cell_size=(3.0, 3.0, 6.0), resolution=10, force_complex_fields=True)
    pml_first.setup_pml(10)  # Interior |z| <= 2.0, so z = -2.3 is three cells inside the layer.
    pml_first.add_source(deep)
    pml_first.run(num_steps=5)
    field = pml_first.get_field("Ez")
    assert np.all(np.isfinite(field)) and float(np.max(np.abs(field))) > 0.0
    source_first = FdtdDriver(cell_size=(3.0, 3.0, 6.0), resolution=10, force_complex_fields=True)
    source_first.add_source(deep)
    source_first.setup_pml(10)
    assert source_first.pml is not None
    source_first.run(num_steps=5)
    # The Dcyl arm retired an hour behind the Cartesian one, when the sweep measured
    # the same f_u mirror transferring unchanged (sources.py deposit-mirror comment:
    # Ez mid-r-layer 4.08e+00 -> 3.12e-06, corner 7.05e+01 -> 2.04e-05, m in
    # {0, 1, 2}, controls bit-unchanged); the lifted parity case is pinned in
    # test_from_meep (cylindrical_source_in_pml, measured 1.1016e-06). Either
    # creation order admits, binds the layer, and steps finite.
    deep_r = {"component": "Ez", "frequency": 1.0, "center": (1.8, 0.0, 0.0),
              "size": (0.0, 0.0, 0.0)}
    for pml_before_source in (True, False):
        cyl = FdtdDriver(cell_size=(2.0, 0.0, 4.0), resolution=10, cylindrical=True,
                         m=0, force_complex_fields=True, boundaries={"z": "metallic"})
        if pml_before_source:
            cyl.setup_pml({"x": (0, 5), "z": (5, 5)})
            cyl.add_source(deep_r)
        else:
            cyl.add_source(deep_r)
            cyl.setup_pml({"x": (0, 5), "z": (5, 5)})
        assert cyl.pml is not None
        cyl.run(num_steps=5)
        field = cyl.get_field("Ez")
        assert np.all(np.isfinite(field)) and float(np.max(np.abs(field))) > 0.0, (
            f"the admitted Dcyl source must step finite with pml_before_source={pml_before_source}"
        )


def _interior_versus_boundary_free(sheet_width):  # Interior error of a PML run against a boundary-free one.
    sheet = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -0.5),
             "size": (sheet_width, sheet_width, 0.0)}
    narrow = FdtdDriver(cell_size=(3.0, 3.0, 4.0), resolution=10, force_complex_fields=True)
    narrow.setup_pml(10)  # Interior |x|, |y| <= 0.5.
    narrow.add_source(sheet)
    narrow.run(until=1.5)
    wide = FdtdDriver(cell_size=(5.0, 5.0, 6.0), resolution=10, force_complex_fields=True)
    wide.add_source(sheet)
    wide.run(until=1.5)  # No face is reached in 1.5 units, so this is free space.
    candidate, reference = narrow.get_field("Ez"), wide.get_field("Ez")
    offset_xy = (reference.shape[0] - candidate.shape[0]) // 2
    offset_z = (reference.shape[2] - candidate.shape[2]) // 2
    inner_xy = slice(10, candidate.shape[0] - 10)
    inner_z = slice(10, candidate.shape[2] - 10)
    outer_xy = slice(offset_xy + 10, offset_xy + candidate.shape[0] - 10)
    outer_z = slice(offset_z + 10, offset_z + candidate.shape[2] - 10)
    truth = reference[outer_xy, outer_xy, outer_z]
    truth_norm = float(np.linalg.norm(truth))
    assert truth_norm > 0.0, "The boundary-free reference run carries no field."
    return float(np.linalg.norm(candidate[inner_xy, inner_xy, inner_z] - truth)) / truth_norm


def test_a_source_on_the_wrapped_boundary_plane_of_a_high_only_layer_is_admitted():
    """The seam plane the per-face reading misses is admitted now — and still real.

    ``{"z": (0, n)}`` reads as "the low face carries nothing", but on a wrapping axis
    ``-L/2`` IS ``+L/2``: one lattice point, stored once at cell 0, owned by the high
    face's grading (MEEP's owned half-cell index 2N) at the absorber's PEAK sigma.
    The sigma-profile pins below keep that fact honest. A source there used to be
    refused by name ("SAME lattice plane"); with the f_u deposit mirror and the
    integrated-dipole withdraw landed it is admitted from either creation order and
    measured at the floor — ``test_from_meep``'s ``source_on_the_wrapped_pml_seam``
    parity case is this exact configuration at 9.6206e-07 whole-volume complex Ez
    against CPU MEEP (a z-only layer never enters Ez's dsig/dsigu ladder, and the
    deposit-cell arithmetic grades cell 0 from the upper wall).
    """
    driver = FdtdDriver(cell_size=(2, 2, 8), resolution=10, force_complex_fields=True)
    driver.setup_pml({"z": (0, 10)})
    # The physical fact the old refusal was built on, still pinned: the layer's
    # maximum sigma sits on cell 0, and cell 1 — half a cell further in at the
    # integer positions — carries nothing.
    profile = driver.pml.get_sigma_profile("z")
    assert profile[0] == float(profile.max()) > 0.0
    assert profile[1] == 0.0

    # The seam itself, the Yee-snapping half cell beside it, and a sheet whose lower
    # edge reaches it are all accepted, and the run steps to a finite field.
    for center, size in (((0.0, 0.0, -4.0), (0.0, 0.0, 0.0)),
                         ((0.0, 0.0, -4.05), (0.0, 0.0, 0.0)),
                         ((0.0, 0.0, -3.0), (0.0, 0.0, 2.0))):
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": center,
                           "size": size})
    driver.run(num_steps=5)
    field = driver.get_field("Ez")
    assert np.all(np.isfinite(field)) and float(np.max(np.abs(field))) > 0.0

    # ... and from the other creation order too.
    late = FdtdDriver(cell_size=(2, 2, 8), resolution=10, force_complex_fields=True)
    late.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -4.0),
                     "size": (0.0, 0.0, 0.0)})
    late.setup_pml({"z": (0, 10)})
    assert late.pml is not None
    late.run(num_steps=5)


def test_a_source_touching_the_pml_edge_is_allowed_because_it_stays_accurate():
    # Where the guard's line sits has to be earned. A sheet whose edge lands exactly on
    # the inner edge of the layer is the standard configuration (the CPU-MEEP sheet case
    # is one), and it must still reproduce a boundary-free run: measured 5.6e-04,
    # against 2.1e-04 for the same sheet two cells clear of the layer. Touching
    # therefore costs a factor of 2.7 and stays ~90x inside the 5% cross-validation bar,
    # while the 0.3-unit penetration measured during the PML investigation cost 1.0e-01
    # — through that bar. Half a cell is where the guard draws the line.
    touching = _interior_versus_boundary_free(1.0)  # Sheet edge at |x| = 0.5 = the inner edge.
    clear = _interior_versus_boundary_free(0.6)  # Sheet edge at |x| = 0.3, two cells clear.
    assert clear < 1.5e-3, f"The control sheet, well clear of the layer, already errs by {clear:.3e}."
    assert touching < 5e-3, f"A sheet touching the PML edge errs by {touching:.3e}; the guard lets it through."
    assert touching < 10.0 * clear, (
        f"Touching the layer costs {touching / clear:.1f}x the error of a sheet clear of it "
        f"({touching:.3e} against {clear:.3e}); the allowance is no longer a small penalty."
    )


# --- Per-axis / per-side PML, and the Bloch-periodic grating it unlocks -----------

# One case, read by both halves of the comparison so the two cannot drift. A period
# of 1.0 across, 4.0 along the propagation axis, absorbing in z only, Bloch-periodic
# in x at an oblique kx, and a broadband Ey sheet spanning the full periodic cross
# section: the standard grating measurement, and the configuration this whole
# feature exists to make legal.
_GRATING_CASE = {
    "period": 1.0,
    "cell_z": 4.0,
    "resolution": 15,
    "kx": 0.4,               # sin(theta) = kx / f: 35 deg at f = 0.7, 18 deg at f = 1.3.
    "pml_cells": 10,         # 0.667 length units on each z face; none on x or y.
    "fcen": 1.0, "df": 0.4, "nfreq": 5,   # Monitor span, inside the source bandwidth.
    "source_fwidth": 1.0,    # Every monitored bin carries at least a fifth of the peak power.
    "source_z": -1.0,
    "upstream_z": -1.15,     # Between the source and the layer: the reflected power.
    "downstream_z": 1.0,
    "eps_peak": 4.0,         # Smooth cos^2 ridge in x inside a Gaussian slab in z.
    "eps_width": 0.3,
    "until": 22.0,
}

_PER_AXIS_ORACLE_SCRIPT = '''"""CPU-MEEP references for the per-axis / per-side PML cases."""
import json
import math
import sys

import numpy as np
import meep as mp

case = json.loads(sys.argv[1])
output_path = sys.argv[2]
period, Lz, resolution = case["period"], case["cell_z"], case["resolution"]
results = {}

# --- 1. The grating: PML in z only, Bloch in x, run once empty and once with the
# structure so each engine normalises its own transmission spectrum.
def epsilon_at(x, z):
    envelope = math.exp(-((z / case["eps_width"]) ** 2))
    ridge = 0.5 + 0.5 * math.cos(2.0 * math.pi * x / period)
    return 1.0 + (case["eps_peak"] - 1.0) * envelope * ridge

spec = (case["fcen"], case["df"], case["nfreq"])
plane = mp.Vector3(period, period, 0)
for name, with_grating in (("vacuum", False), ("grating", True)):
    arguments = dict(
        cell_size=mp.Vector3(period, period, Lz),
        resolution=resolution,
        boundary_layers=[mp.PML(case["pml_cells"] / resolution, direction=mp.Z)],
        sources=[mp.Source(mp.GaussianSource(case["fcen"], fwidth=case["source_fwidth"]),
                           component=mp.Ey,
                           center=mp.Vector3(0, 0, case["source_z"]),
                           size=mp.Vector3(period, period, 0))],
        k_point=mp.Vector3(case["kx"], 0, 0),
        force_complex_fields=True,
        eps_averaging=False,  # Point-sample the material, so both engines see one function.
    )
    if with_grating:
        arguments["material_function"] = lambda p: mp.Medium(epsilon=epsilon_at(p.x, p.z))
    simulation = mp.Simulation(**arguments)
    downstream = simulation.add_flux(
        *spec, mp.FluxRegion(center=mp.Vector3(0, 0, case["downstream_z"]), size=plane,
                             direction=mp.Z), decimation_factor=1)
    upstream = simulation.add_flux(
        *spec, mp.FluxRegion(center=mp.Vector3(0, 0, case["upstream_z"]), size=plane,
                             direction=mp.Z), decimation_factor=1)
    simulation.run(until=case["until"])
    results[name + "_downstream"] = np.asarray(mp.get_fluxes(downstream))
    results[name + "_upstream"] = np.asarray(mp.get_fluxes(upstream))
    results["freqs"] = np.asarray(mp.get_flux_freqs(downstream))

# The same epsilon sampled where the engine's inv_eps array is registered. `integer`
# is the Yee position of the components inv_eps multiplies; `center` is half a cell
# off, and the test requires the comparison to be able to tell them apart.
#
# Both are measured from MEEP's OWN axis origin, which is -(n - n % 2) in doubled
# coordinates (vec.cpp icenter(), applied by Simulation._create_grid_volume's
# center_origin) — that is -L/2 only when the cell count is EVEN. This grating's
# transverse axes hold 15 cells, so -L/2 is half a cell below where MEEP registers
# and sampling there costs 6.6e-02 on the transmitted flux against 3.0e-07 here.
nx, ny, nz = (int(period * resolution + 0.5), int(period * resolution + 0.5),
              int(Lz * resolution + 0.5))
dx = 1.0 / resolution
origin_x = -(nx - nx % 2) * dx / 2
origin_z = -(nz - nz % 2) * dx / 2
for label, offset in (("integer", 0.0), ("center", 0.5)):
    xs = origin_x + (np.arange(nx) + offset) * dx
    zs = origin_z + (np.arange(nz) + offset) * dx
    sampled = np.array([[epsilon_at(x, z) for z in zs] for x in xs])
    results["eps_" + label] = np.repeat(sampled[:, None, :], ny, axis=1)
# The registration this engine used before the origin parity was corrected: the same
# analytic grating sampled from -L/2 on every axis. On an odd-count axis that is half
# a cell off, and the headline comparison has to be able to say so.
for label, offset in (("halfcell_off", 0.0),):
    xs = -period / 2 + (np.arange(nx) + offset) * dx
    zs = -Lz / 2 + (np.arange(nz) + offset) * dx
    sampled = np.array([[epsilon_at(x, z) for z in zs] for x in xs])
    results["eps_" + label] = np.repeat(sampled[:, None, :], ny, axis=1)

# --- 2. A z-only layer with periodic (unphased) x and y, on the point-source cell
# the no-PML parity case already uses.
simulation = mp.Simulation(
    cell_size=mp.Vector3(2, 2, 4), resolution=10,
    boundary_layers=[mp.PML(1.0, direction=mp.Z)],
    sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                       center=mp.Vector3(0, 0, 0.55))],
    force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
simulation.run(until=3)
results["z_only_ez"] = np.asarray(simulation.get_array(component=mp.Ez))

# --- 3. Per-side asymmetry: one absorbing face, the opposite one left periodic. The
# cell is tall enough that the far face stays quiet for the whole run even at the
# grid's own front speed dx/dt = 2c, so the comparison measures the layer rather than
# the pathological wrap of a live wave into the far absorber's outer edge.
for label, side, center in (("one_sided_low", mp.Low, -2.0), ("one_sided_high", mp.High, 2.0)):
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 8), resolution=10,
        boundary_layers=[mp.PML(1.0, direction=mp.Z, side=side)],
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, center))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
    simulation.run(until=2.5)
    results[label + "_ez"] = np.asarray(simulation.get_array(component=mp.Ez))

# --- 4. Both z faces absorbing at DIFFERENT thicknesses, the only configuration in
# which MEEP's per-side prefac (-ln R / (4 * that side's dx * integral)) is visible:
# the thin face has to be graded 2.5x more steeply to reach the same reflection.
simulation = mp.Simulation(
    cell_size=mp.Vector3(2, 2, 6), resolution=10,
    boundary_layers=[mp.PML(1.0, direction=mp.Z, side=mp.Low),
                     mp.PML(0.4, direction=mp.Z, side=mp.High)],
    sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                       center=mp.Vector3(0, 0, 0.05))],
    force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
simulation.run(until=4)
results["asymmetric_ez"] = np.asarray(simulation.get_array(component=mp.Ez))

# --- 5. The same one-sided cells run long enough that the wave HAS crossed the
# un-absorbed face and re-entered at the absorber's outer edge. MEEP is happy to do
# this — the face wraps and the layer is a material — so it is a parity case, not an
# unsupported configuration.
for label, side, center in (("live_low", mp.Low, -2.0), ("live_high", mp.High, 2.0)):
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 8), resolution=10,
        boundary_layers=[mp.PML(1.0, direction=mp.Z, side=side)],
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, center))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
    simulation.run(until=12)
    results[label + "_ez"] = np.asarray(simulation.get_array(component=mp.Ez))

# --- 6. A two-sided layer on X alone, y and z left periodic: the transverse
# termination on its own, with no z absorber to hide it.
simulation = mp.Simulation(
    cell_size=mp.Vector3(3, 2, 4), resolution=10,
    boundary_layers=[mp.PML(1.0, direction=mp.X)],
    sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                       center=mp.Vector3(0, 0, 0.05))],
    force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
simulation.run(until=3)
results["x_only_ez"] = np.asarray(simulation.get_array(component=mp.Ez))

# --- 7. The discriminating control for the boundary rule: the SAME uniform-PML run
# with no k_point at all, which leaves MEEP's default `Metallic` on all six faces
# (fields.cpp:73, only use_bloch turns them Periodic). That is the run this engine's
# old metallic termination was actually modelling, and the engine must now be far
# closer to the periodic reference than to this one. Its get_array has N samples per
# axis rather than N+1: there is no periodic duplicate to return.
simulation = mp.Simulation(
    cell_size=mp.Vector3(3, 3, 6), resolution=10,
    sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                       center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))],
    boundary_layers=[mp.PML(1.0)],
    force_complex_fields=True)
simulation.run(until=3)
results["uniform_metallic_ez"] = np.asarray(simulation.get_array(component=mp.Ez))

# --- 8. A layer that is NOT a whole number of cells, plus the two whole-cell
# absorbers a rounding converter would substitute for it. mp.PML snaps only the
# layer's EXTENT, to the nearest half cell (structure.cpp pml_x), and normalises the
# profile by the raw thickness, so 5.5 cells is a third absorber rather than 5 or 6.
# One-sided and run long enough for the wave to re-enter through the un-absorbed
# face, so the fraction is measured together with the owned-index rule rather than
# on a quiet cell where either would do.
for label, thickness in (("fractional", 0.55), ("fractional_down", 0.5),
                         ("fractional_up", 0.6)):
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 8), resolution=10,
        boundary_layers=[mp.PML(thickness, direction=mp.Z, side=mp.Low)],
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, -2.0))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
    simulation.run(until=12)
    results[label + "_ez"] = np.asarray(simulation.get_array(component=mp.Ez))

np.savez(output_path, **results)
'''


@pytest.fixture(scope="module")
def per_axis_oracle(tmp_path_factory):  # One CPU-MEEP process for every per-axis PML case.
    import json

    tmp_path = tmp_path_factory.mktemp("per_axis_pml_oracle")
    script_path = tmp_path / "meep_per_axis_oracle.py"
    script_path.write_text(_PER_AXIS_ORACLE_SCRIPT, encoding="utf-8")
    output_path = tmp_path / "per_axis.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(_GRATING_CASE), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP per-axis PML oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _grating_run(case, epsilon=None, kx=None):  # One engine run of the grating case.
    period, cell_z = case["period"], case["cell_z"]
    driver = FdtdDriver(
        cell_size=(period, period, cell_z), resolution=case["resolution"],
        force_complex_fields=True,
        k_point=(case["kx"] if kx is None else kx, 0.0, 0.0),
    )
    # The whole point: absorb along z while x and y stay Bloch-periodic. Before this
    # existed the next line raised, whatever the k_point was.
    driver.setup_pml({"z": case["pml_cells"]})
    assert driver._monitor_periodic_axes() == (True, True, True), (
        "every unmirrored axis wraps, or a monitor spanning the full periodic cross "
        "section silently loses its far face."
    )
    if epsilon is not None:
        driver.set_epsilon(epsilon)
    driver.add_source({
        "component": "Ey", "source_type": "gaussian",
        "frequency": case["fcen"], "fwidth": case["source_fwidth"],
        "center": (0.0, 0.0, case["source_z"]), "size": (period, period, 0.0),
    })
    plane = dict(fcen=case["fcen"], df=case["df"], nfreq=case["nfreq"],
                 size=(period, period, 0.0), direction=2)
    downstream = driver.add_flux_monitor(center=(0.0, 0.0, case["downstream_z"]), **plane)
    upstream = driver.add_flux_monitor(center=(0.0, 0.0, case["upstream_z"]), **plane)
    driver.run(until=case["until"])
    return (np.asarray(downstream.get_flux_spectrum(), dtype=float),
            np.asarray(upstream.get_flux_spectrum(), dtype=float))


def _worst_relative(candidate, reference):  # Largest per-entry relative deviation.
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert np.all(np.abs(reference) > 0.0), "a zero reference entry would make the ratio meaningless"
    return float(np.max(np.abs(candidate - reference) / np.abs(reference)))


@requires_meep
@skip_without_meep
def test_grating_transmission_spectrum_matches_cpu_meep(per_axis_oracle):
    """The headline measurement: T(f) through a Bloch-periodic grating, PML in z only.

    Per-axis PML, a Bloch phase on an unabsorbed axis, a source and two flux planes
    spanning the full periodic cross section, and five frequencies in one pass — all
    at once, against CPU MEEP. Each engine normalises against its own empty-cell run,
    so the comparison is of the physics (boundaries, absorber, flux) and not of how
    the two place a sheet source that fills the cell.

    Measured at resolution 15, per frequency: transmission 3.8e-07, empty-cell flux
    3.1e-07 downstream and 1.9e-07 upstream, grating flux 3.0e-07 downstream and
    1.7e-07 upstream. Two defects used to hold this at the percent level and are
    pinned by the controls below — the source lost its wrapped plane at the periodic
    face (sources.py `_build_source_points`), and every odd-count axis was registered
    half a cell from MEEP's origin (grid.py `origin_doubled`).
    """
    case = _GRATING_CASE
    reference = per_axis_oracle
    meep_transmission = reference["grating_downstream"] / reference["vacuum_downstream"]

    vacuum_downstream, vacuum_upstream = _grating_run(case)
    grating_downstream, grating_upstream = _grating_run(case, epsilon=reference["eps_integer"])
    transmission = grating_downstream / vacuum_downstream

    assert _worst_relative(vacuum_downstream, reference["vacuum_downstream"]) < 5e-6
    assert _worst_relative(vacuum_upstream, reference["vacuum_upstream"]) < 5e-6
    assert _worst_relative(grating_downstream, reference["grating_downstream"]) < 5e-6
    assert _worst_relative(grating_upstream, reference["grating_upstream"]) < 5e-6
    worst = _worst_relative(transmission, meep_transmission)
    assert worst < 5e-6, (
        f"Transmission spectrum deviates from CPU MEEP by {worst:.3f} per frequency "
        f"(engine {np.array2string(transmission, precision=4)}, "
        f"meep {np.array2string(meep_transmission, precision=4)})."
    )
    # The grating has to do something, or the ratio is a comparison of 1.0 with 1.0.
    assert np.max(np.abs(meep_transmission - 1.0)) > 0.2, "this grating barely diffracts"

    # Control 1 — the Bloch phase. An engine that accepted the k_point and dropped it
    # once a PML existed would still return a complete, smooth spectrum; at this
    # incidence it lands 100x outside the bar.
    unphased_vacuum, _ = _grating_run(case, kx=0.0)
    unphased_grating, _ = _grating_run(case, epsilon=reference["eps_integer"], kx=0.0)
    unphased = _worst_relative(unphased_grating / unphased_vacuum, meep_transmission)
    assert unphased > 0.2, (
        f"A run at k = 0 must not reproduce the k = {case['kx']} spectrum, but it came "
        f"within {unphased:.3f}."
    )

    # Control 2 — the epsilon registration. inv_eps[i,j,k] multiplies the D component
    # stored at the INTEGER Yee position `axis_origin + i*dx`, not at the cell centre,
    # and the measurement is sharp enough to say so: sampling the same analytic
    # grating half a cell across lands at 6.7e-02, five orders out. (This is the
    # non-uniform-epsilon oracle the set_epsilon docstring asks for.)
    offset_downstream, _ = _grating_run(case, epsilon=reference["eps_center"])
    offset = _worst_relative(offset_downstream / vacuum_downstream, meep_transmission)
    assert offset > 1000.0 * worst, (
        f"The comparison cannot distinguish the epsilon registration: integer sampling "
        f"gives {worst:.2e} and half-cell-offset sampling {offset:.2e}."
    )

    # Control 3 — the axis ORIGIN parity, which is a different half-cell than control 2
    # and only exists on an odd-count axis. This grating's transverse axes hold 15
    # cells; sampling the same analytic epsilon from -L/2, as the engine's own
    # registration implied before `Grid.origin_doubled` rounded the count down to even
    # the way MEEP's icenter() does, lands at 6.6e-02. Nothing about that run says so:
    # it is a complete, smooth, plausible transmission spectrum.
    stale_downstream, _ = _grating_run(case, epsilon=reference["eps_halfcell_off"])
    stale = _worst_relative(stale_downstream / vacuum_downstream, meep_transmission)
    assert stale > 1000.0 * worst, (
        f"The comparison cannot distinguish MEEP's odd-count axis origin from -L/2: "
        f"the correct origin gives {worst:.2e} and -L/2 gives {stale:.2e}."
    )


@requires_meep
@skip_without_meep
def test_z_only_pml_with_periodic_transverse_axes_matches_cpu_meep(per_axis_oracle):
    # A layer on z alone leaves x and y wrapping, which is what MEEP does under a PML
    # too, so this configuration reproduces MEEP far more closely than the uniform
    # layer's 5.8e-4 — there is no metallic approximation left anywhere in it.
    reference = per_axis_oracle["z_only_ez"][1:, 1:, 1:]
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    driver.setup_pml({"z": 10})
    assert driver.pml.thickness_by_face == ((0, 0), (0, 0), (10, 10))
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.55), "size": (0.0, 0.0, 0.0)}
    )
    driver.run(until=3.0)
    field = driver.get_field("Ez")
    complex_l2 = _complex_relative_l2(field, reference)
    assert complex_l2 < 1e-3, (
        f"Complex Ez relative L2 {complex_l2:.3e} exceeds the 1e-3 PML-path bar; measured 4.5e-07."
    )
    relative_l2, correlation = _magnitude_metrics(field, reference)
    assert relative_l2 < 1e-3 and correlation > 0.9999

    # Control: the transverse boundaries are load-bearing in this comparison. Adding a
    # layer on x and y — which turns those axes metallic — moves the answer far off.
    terminated = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    terminated.setup_pml({"x": 5, "y": 5, "z": 10})
    terminated.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.55), "size": (0.0, 0.0, 0.0)}
    )
    terminated.run(until=3.0)
    assert _complex_relative_l2(terminated.get_field("Ez"), reference) > 0.1


@requires_meep
@skip_without_meep
@pytest.mark.parametrize(
    "label, thickness, source_z",
    [("one_sided_low", {"z": (10, 0)}, -2.0), ("one_sided_high", {"z": (0, 10)}, 2.0)],
)
def test_one_sided_pml_matches_cpu_meep(per_axis_oracle, label, thickness, source_z):
    """A layer on one z face, the opposite face left periodic — MEEP's mp.PML(side=...).

    The cell is 8 long and the run stops at t = 2.5, so the un-absorbed face is not
    reached even at the grid's own front speed dx/dt = 2c: what is measured is the
    per-side grading, which is graded from that face's own thickness and its own
    prefac. Measured 2.13e-05 for both sides — identical to the digit, which is the
    z-mirror consistency of the two configurations.
    """
    reference = per_axis_oracle[label + "_ez"][1:, 1:, 1:]
    driver = FdtdDriver(cell_size=(2, 2, 8), resolution=10, force_complex_fields=True)
    driver.setup_pml(thickness)
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, source_z), "size": (0.0, 0.0, 0.0)}
    )
    driver.run(until=2.5)
    complex_l2 = _complex_relative_l2(driver.get_field("Ez"), reference)
    # Measured 2.9e-07 / 3.1e-07 — the same band as the two-sided layers. It read
    # 2.13e-05 while cell 0 was graded from the low wall instead of the wall MEEP owns
    # it at (pml.PML._graded_sigma): even with the far face quiet, the wrapped point
    # carried the wrong sigma and a little of the wave had already reached it.
    assert complex_l2 < 1e-6, (
        f"{label}: complex Ez relative L2 {complex_l2:.3e} exceeds the 1e-6 bar; measured 2.9e-07."
    )

    # Control: the layer must be on the face that was asked for. Putting it on the
    # other face leaves the wave unabsorbed and lands three orders of magnitude out,
    # so a table that symmetrised the two faces — or swapped them — cannot pass.
    swapped = FdtdDriver(cell_size=(2, 2, 8), resolution=10, force_complex_fields=True)
    swapped.setup_pml({"z": (thickness["z"][1], thickness["z"][0])})
    swapped.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, source_z), "size": (0.0, 0.0, 0.0)}
    )
    swapped.run(until=2.5)
    assert _complex_relative_l2(swapped.get_field("Ez"), reference) > 0.01


@requires_meep
@skip_without_meep
@pytest.mark.parametrize(
    "label, thickness, source_z",
    [("live_low", {"z": (10, 0)}, -2.0), ("live_high", {"z": (0, 10)}, 2.0)],
)
def test_one_sided_pml_with_a_live_wave_on_the_unabsorbed_face_matches_cpu_meep(
        per_axis_oracle, label, thickness, source_z):
    """The same one-sided cells run until the wave HAS wrapped into the absorber backwards.

    This is the configuration the engine used to get 40 % wrong while only warning
    about it in a docstring, and the explanation on file was that entering a graded
    layer from its high-sigma side reflects strongly in both codes and amplifies a
    round-off difference. That was wrong. Stepped side by side against CPU MEEP the two
    runs are bit-comparable until the exact step the wave first crosses the seam, and
    then differ at the first absorber cell by a factor of 0.6069 — which is
    ``1/(kappa+sigma)`` at that cell to four digits. The engine was applying an
    absorption there that MEEP was not: cell 0 stores the WRAPPED lattice point, MEEP
    owns it at half-cell index 2N (the upper wall) rather than 0, and a one-sided low
    layer therefore does not absorb there at all. See pml.PML._graded_sigma.

    Measured 5.5e-07 / 6.0e-07, against 1.164e-01 / 1.165e-01 before. The run is long
    enough (t = 12 on a cell 8 long) that the wave crosses the un-absorbed face many
    times, so the comparison is dominated by what happens at the seam.
    """
    reference = per_axis_oracle[label + "_ez"][1:, 1:, 1:]
    driver = FdtdDriver(cell_size=(2, 2, 8), resolution=10, force_complex_fields=True)
    driver.setup_pml(thickness)
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, source_z), "size": (0.0, 0.0, 0.0)}
    )
    driver.run(until=12.0)
    field = driver.get_field("Ez")
    complex_l2 = _complex_relative_l2(field, reference)
    assert complex_l2 < 1e-5, (
        f"{label}: complex Ez relative L2 {complex_l2:.3e} exceeds the 1e-5 bar; measured 5.5e-07."
    )

    # The un-absorbed face must genuinely be carrying a wave, or this is just the quiet
    # case again under a longer name.
    face = float(np.abs(np.asarray(reference)[:, :, -1 if source_z < 0 else 0]).mean())
    assert face > 0.1 * float(np.abs(np.asarray(reference)).mean()), (
        f"{label}: the un-absorbed face carries {face:.3e}; the wrap is not being exercised.")

    # Control: the layer must be on the face that was asked for. With it on the other
    # face the wave is absorbed where it should pass and vice versa.
    swapped = FdtdDriver(cell_size=(2, 2, 8), resolution=10, force_complex_fields=True)
    swapped.setup_pml({"z": (thickness["z"][1], thickness["z"][0])})
    swapped.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, source_z), "size": (0.0, 0.0, 0.0)}
    )
    swapped.run(until=12.0)
    assert _complex_relative_l2(swapped.get_field("Ez"), reference) > 1000.0 * complex_l2


@requires_meep
@skip_without_meep
def test_a_fractional_cell_pml_matches_cpu_meep_and_neither_whole_cell_neighbour(per_axis_oracle):
    """5.5 cells is an absorber MEEP builds, and it is not 5 cells or 6 cells.

    ``mp.PML(0.55)`` at resolution 10 asks for half a cell more than five. MEEP does
    not round it: ``pml_x`` (structure.cpp:625-628) snaps only the layer's EXTENT, and
    only to the nearest HALF cell, while ``prefac`` and the profile argument both
    divide by the raw 0.55 (structure.cpp:635,683). The engine refused this
    configuration outright until the coefficient table could carry a fraction, and the
    converter would otherwise have rounded it — which is why the two whole-cell
    neighbours are run as controls rather than named in a comment.

    Measured: 5.63e-07 against CPU MEEP, with the 5-cell absorber 3.26e-02 away from
    that same reference and the 6-cell one 2.89e-02 — nearly five orders of magnitude
    of separation, on three runs that are each a perfectly ordinary decaying field.
    Both neighbours reproduce their OWN oracle at 6.01e-07 and 5.89e-07, so the gap is
    the thickness and not a defect the fractional path happens to avoid. The layer is
    one-sided, and the run is long enough (t = 12 on a cell 8 long) for the wave to
    come back through the un-absorbed face into the absorber's outer edge, so the
    fraction is measured on top of the owned-index rule rather than on a plane where
    either grading would do.
    """
    reference = per_axis_oracle["fractional_ez"][1:, 1:, 1:]

    def run(thickness):  # One driver run of the same cell at one low-face thickness.
        driver = FdtdDriver(cell_size=(2, 2, 8), resolution=10, force_complex_fields=True)
        driver.setup_pml({"z": (thickness, 0)})
        driver.add_source(
            {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -2.0), "size": (0.0, 0.0, 0.0)}
        )
        driver.run(until=12.0)
        return driver.get_field("Ez")

    complex_l2 = _complex_relative_l2(run(5.5), reference)
    print(f"[pml] 5.5 cells vs CPU MEEP: {complex_l2:.3e}", flush=True)
    assert complex_l2 < 5e-6, (
        f"a 5.5-cell layer differs from CPU MEEP by {complex_l2:.3e}, above the 5e-6 bar; "
        f"measured 5.63e-07, the same band as every whole-cell one-sided layer."
    )
    # The controls are the whole-cell absorbers a round(), a floor() or a ceil() would
    # have built. Each must be far outside the bar, or the test above would pass under
    # the rounding this feature exists to remove.
    for neighbour in (5, 6):
        rounded = _complex_relative_l2(run(neighbour), reference)
        print(f"[pml] {neighbour} cells vs the 5.5-cell reference: {rounded:.3e}", flush=True)
        assert rounded > 1e-3, (
            f"a {neighbour}-cell layer sits {rounded:.3e} from the 5.5-cell reference; the "
            f"fractional thickness is not being distinguished from its neighbours"
        )
    # ...and each neighbour must be right against its OWN oracle, so the separation
    # above is the thickness and not some defect the fractional path happens to avoid.
    for neighbour, label in ((5, "fractional_down"), (6, "fractional_up")):
        own = _complex_relative_l2(run(neighbour), per_axis_oracle[label + "_ez"][1:, 1:, 1:])
        print(f"[pml] {neighbour} cells vs its own CPU-MEEP run: {own:.3e}", flush=True)
        assert own < 5e-6, (
            f"the {neighbour}-cell control is {own:.3e} from its own CPU-MEEP run; measured "
            f"6.0e-07 and 5.9e-07, so a control that misses here is not a control"
        )


@requires_meep
@skip_without_meep
def test_two_faces_of_different_thickness_match_cpu_meep(per_axis_oracle):
    """A 10-cell layer on z-low and a 4-cell one on z-high, both absorbing the same wave.

    This is the only configuration that can see MEEP's per-side ``prefac``: it is
    ``-ln(R) / (4 * dx_pml * integral)`` with THAT side's ``dx_pml``, so the thin face
    is graded 2.5x more steeply than the thick one to reach the same asymptotic
    reflection. A layer that took one prefac for the whole axis would put the thick
    face's grading on the thin one and under-absorb there, which is a plausible field,
    not a crash.

    Measured 9.9e-05. Every way of collapsing the two faces into one thickness —
    10 and 10, 4 and 4, or the two swapped — lands at 6.5e-02 to 9.9e-02, three
    orders of magnitude out, so the comparison is measuring the per-face table and
    not merely the presence of an absorber.
    """
    reference = per_axis_oracle["asymmetric_ez"][1:, 1:, 1:]

    def run(thickness):
        driver = FdtdDriver(cell_size=(2, 2, 6), resolution=10, force_complex_fields=True)
        driver.setup_pml(thickness)
        driver.add_source({"component": "Ez", "frequency": 1.0,
                           "center": (0.0, 0.0, 0.05), "size": (0.0, 0.0, 0.0)})
        driver.run(until=4.0)
        return driver.get_field("Ez")

    asymmetric = _complex_relative_l2(run({"z": (10, 4)}), reference)
    # Measured 4.9e-07. It read 9.9e-05 while cell 0 — the wrapped lattice point, which
    # MEEP owns at the UPPER wall — was graded with the low face's prefac instead of the
    # high face's; on a (10, 4) layer those differ by 2.5x, so that one plane carried
    # 2.5x too little absorption (pml.PML._graded_sigma).
    assert asymmetric < 1e-5, (
        f"Complex Ez relative L2 {asymmetric:.3e} exceeds the 1e-5 bar; measured 4.9e-07."
    )
    for collapsed in ({"z": 10}, {"z": 4}, {"z": (4, 10)}):
        assert _complex_relative_l2(run(collapsed), reference) > 1000.0 * asymmetric, (
            f"{collapsed} must not reproduce the (10, 4) reference."
        )


@requires_meep
@skip_without_meep
def test_x_wraps_under_a_two_sided_layer_the_way_meep_does(per_axis_oracle):
    """A layer on X alone, with y and z periodic: the transverse termination on its own.

    An axis absorbing on BOTH faces used to be terminated metallically here, on the
    reasoning that a PML "terminates" its axis. MEEP does not work that way: the
    boundary condition comes from ``fields::use_bloch`` (boundaries.cpp), which every
    ``mp.Simulation`` carrying a ``k_point`` — including ``mp.Vector3()`` — applies to
    all six faces and which never consults the absorber, while
    ``structure_chunk::use_pml`` only grades a conductivity underneath it.

    This case isolates that decision: there is no z layer to swallow the field before
    the x walls, so the metallic approximation is not hidden. Measured 4.8e-07,
    against 9.5e-04 with the metallic termination.
    """
    reference = per_axis_oracle["x_only_ez"][1:, 1:, 1:]
    driver = FdtdDriver(cell_size=(3, 2, 4), resolution=10, force_complex_fields=True)
    driver.setup_pml({"x": 10})
    assert driver.pml.thickness_by_face == ((10, 10), (0, 0), (0, 0))
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05), "size": (0.0, 0.0, 0.0)}
    )
    driver.run(until=3.0)
    complex_l2 = _complex_relative_l2(driver.get_field("Ez"), reference)
    assert complex_l2 < 1e-5, (
        f"Complex Ez relative L2 {complex_l2:.3e} exceeds the 1e-5 bar; measured 4.8e-07."
    )


@requires_meep
@skip_without_meep
def test_a_uniform_layer_reproduces_meeps_periodic_run_and_not_its_metallic_one(per_axis_oracle, tmp_path):
    """The boundary rule, pinned against BOTH of MEEP's two boundary conditions at once.

    ``mp.Simulation`` leaves every face ``Metallic`` (fields.cpp:73) unless a
    ``k_point`` is given, in which case ``use_bloch`` turns them all ``Periodic``. Those
    are two different physical runs of the same cell, absorber and source, and this
    engine implements the periodic one. Asserting only "close to the periodic
    reference" would not say so while the absorber is thick enough to make the two MEEP
    runs resemble each other, so the metallic reference is measured too and has to come
    out far worse.

    Measured: 4.0e-07 against the periodic reference, 1.3e-03 against the metallic one —
    a factor of 3000. Before the rule changed those read 6.3e-04 and 6.5e-04, i.e. the
    engine was equidistant from the two and reproduced neither.
    """
    reference = _meep_oracle(tmp_path, "sheet_with_pml")[1:, 1:, 1:]
    metallic = per_axis_oracle["uniform_metallic_ez"]
    driver = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=True)
    driver.setup_pml(10)
    driver.add_source(
        {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.5), "size": (1.0, 1.0, 0.0)}
    )
    driver.run(until=3.0)
    field = np.asarray(driver.get_field("Ez"))
    # MEEP's metallic get_array has N samples per axis, not N+1: with no periodic wrap
    # there is no duplicate plane to return, so its sample j is this grid's cell j + 1.
    assert metallic.shape == field.shape, (metallic.shape, field.shape)
    periodic_error = _complex_relative_l2(field, reference)
    metallic_error = _complex_relative_l2(field[1:, 1:, 1:], metallic[1:, 1:, 1:])
    assert periodic_error < 1e-5, (
        f"Complex Ez relative L2 {periodic_error:.3e} against MEEP's periodic run exceeds the "
        f"1e-5 bar; measured 4.0e-07 (6.31e-04 with the metallic termination)."
    )
    assert metallic_error > 100.0 * periodic_error, (
        f"The engine is {metallic_error:.3e} from MEEP's METALLIC run and {periodic_error:.3e} "
        f"from its periodic one; it must be unambiguously reproducing the periodic one."
    )


def test_the_scalar_layer_runs_bit_identically_to_the_per_face_spelling_of_itself():
    # End-to-end companion to the coefficient-table pin in test_grid_pml.py: the whole
    # run, not just the table. Everything measured under a uniform PML — the 5.84e-4 /
    # 6.31e-4 CPU-MEEP floor, the transparency scaling, the symmetry equivalence — was
    # measured through setup_pml(t), so per-face support must not move a single bit of
    # it. Bytes, not allclose: a reordered float expression passes a tolerance.
    def run(thickness, **arguments):
        driver = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=True,
                            **arguments)
        driver.setup_pml(thickness)
        driver.add_source({"component": "Ez", "frequency": 1.0,
                           "center": (0.0, 0.0, -1.5), "size": (1.0, 1.0, 0.0)})
        driver.run(until=1.5)
        return {name: np.asarray(driver.get_field(name, cell_centered=False))
                for name in ("Ez", "Ex", "Hy", "Dz", "Bx")}

    scalar = run(10)
    assert max(float(np.abs(values).max()) for values in scalar.values()) > 0.0
    for spelling in ({"x": 10, "y": 10, "z": 10}, (10, 10, 10),
                     ((10, 10), (10, 10), (10, 10)), {"x": (10, 10), "y": 10, "z": {"low": 10, "high": 10}}):
        per_face = run(spelling)
        for name, values in scalar.items():
            assert per_face[name].tobytes() == values.tobytes(), f"{name} moved for {spelling!r}"
    # And on a folded axis, where the scalar shorthand drops the mirror-plane face.
    folded_scalar = run(8, symmetry=("X",))
    folded_named = run({"x": {"high": 8}, "y": 8, "z": 8}, symmetry=("X",))
    for name, values in folded_scalar.items():
        assert folded_named[name].tobytes() == values.tobytes(), f"{name} moved on the folded axis"


def test_bloch_and_pml_coexist_on_the_same_axis_and_step():
    """A layer on the phased axis is installed and stepped, not refused.

    This asserted the refusal until it was measured. MEEP's `use_bloch` makes every
    direction Periodic as soon as any k_point is given and a PML is graded material
    underneath that wrap, so the axis repeats and the wrap carries its phase; the
    absorber only means little survives to use it. Stepping the formerly-refused
    pairing reproduces CPU MEEP at 2.82e-07, against 2.63e-07 with the absorber taken
    off the phased axis (parity case `bloch_on_pml_axis`, whose no-absorber control
    sits at 1.4e-01). Seven scripts of MEEP's own corpus needed it.
    """
    # The grating: absorb along z, x and y keep their Bloch wrap.
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, k_point=(0.3, 0.2, 0.0))
    driver.setup_pml({"z": 8})
    assert driver.pml.thickness_by_face == ((0, 0), (0, 0), (8, 8))
    driver.run(num_steps=3)

    # And every pairing that used to raise: installed, and stepping.
    for k_point, thickness, expected_faces in (
        ((0.3, 0.0, 0.0), 4, ((4, 4), (4, 4), (4, 4))),
        ((0.3, 0.0, 0.0), {"x": 4}, ((4, 4), (0, 0), (0, 0))),
        ((0.3, 0.0, 0.0), {"x": (4, 0)}, ((4, 0), (0, 0), (0, 0))),
        ((0.0, 0.2, 0.0), {"y": 4, "z": 4}, ((0, 0), (4, 4), (4, 4))),
        ((0.0, 0.0, 0.3), {"z": 4}, ((0, 0), (0, 0), (4, 4))),
    ):
        allowed = FdtdDriver(cell_size=(2, 2, 4), resolution=10, k_point=k_point)
        allowed.setup_pml(thickness)
        assert allowed.pml is not None, f"k={k_point} PML {thickness} was not installed"
        assert allowed.pml.thickness_by_face == expected_faces
        allowed.run(num_steps=3)


def test_setup_pml_refuses_a_layer_that_absorbs_nowhere_or_does_not_fit():
    # Degenerate and impossible requests, each with the reason named. A zero request is
    # a fully periodic run wearing an absorber's name, which is the silent kind.
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10)
    for thickness in (0, {"z": 0}, {"x": (0, 0), "z": 0}, (0, 0, 0)):
        with pytest.raises(ValueError, match="absorbs on no face"):
            driver.setup_pml(thickness)
    for thickness, axis in ((10, "X"), ({"x": 10}, "X"), ({"z": 20}, "Z"),
                            ({"z": (39, 1)}, "Z"), ({"x": (0, 20)}, "X")):
        with pytest.raises(ValueError, match="leaves no interior") as raised:
            driver.setup_pml(thickness)
        assert f"{axis} axis" in str(raised.value)
    with pytest.raises(ValueError, match="Unknown PML axis key"):
        driver.setup_pml({"w": 4})
    assert driver.pml is None
    # A layer that exactly fills the axis but for one cell is legal and steps.
    driver.setup_pml({"x": (9, 10)})
    assert driver.pml.thickness_by_face == ((9, 10), (0, 0), (0, 0))
    driver.run(num_steps=2)


def test_monitors_wrap_wherever_the_stepping_wraps_and_nowhere_else():
    # dft.py takes a lattice shift where the driver says the face wraps, so this tuple is
    # the whole contract between the boundaries and every monitor — and it must be the
    # SAME tuple the stepping uses, or a monitor measures a boundary the run does not
    # have. The absorber does not appear in it: MEEP's boundaries come from use_bloch,
    # which is Periodic on every direction of a run carrying a k_point, and
    # loop_in_chunks takes its lattice shift on exactly that flag.
    def periodic_axes(thickness, **driver_arguments):
        driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, **driver_arguments)
        if thickness is not None:
            driver.setup_pml(thickness)
        return driver._monitor_periodic_axes()

    for thickness in (None, 4, {"z": 4}, {"z": (4, 0)}, {"x": 4}):
        assert periodic_axes(thickness) == (True, True, True), (
            f"{thickness!r}: an absorber is a material, not a boundary condition")
    assert periodic_axes(None, symmetry=("X",)) == (False, True, True)
    assert periodic_axes({"z": 4}, symmetry=("X",)) == (False, True, True)

    # The stepping's own answer, on the same runs: the two rules are one statement.
    for thickness, symmetry in ((None, ()), (4, ()), ({"z": 4}, ()), ({"x": 4}, ()),
                                (None, ("X",)), ({"z": 4}, ("X",))):
        driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, symmetry=symmetry)
        if thickness is not None:
            driver.setup_pml(thickness)
        kinds = _boundary_kinds(driver.grid, driver.pml)
        assert driver._monitor_periodic_axes() == tuple(kind == PERIODIC for kind in kinds), (
            f"{thickness!r} / {symmetry!r}: monitors and stepping disagree about which faces wrap")

    # And the monitors actually receive it: a flux plane spanning the full cross section
    # registers its wrap on x and y whether or not the layer covers them.
    for thickness in ({"z": 4}, 4):
        driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10)
        driver.setup_pml(thickness)
        monitor = driver.add_flux_monitor(
            frequency=1.0, center=(0.0, 0.0, 1.0), size=(2.0, 2.0, 0.0), direction=2)
        assert monitor.periodic == (True, True, True)
        assert monitor._wrapped[:2] == (True, True)


@requires_meep
@skip_without_meep
def test_a_flux_plane_spanning_an_absorbing_axis_matches_cpu_meep(tmp_path):
    """The measurement that says the absorber must not clip a monitor's wrap.

    A plane spanning the full cross section of a cell whose TRANSVERSE axes carry a
    layer is the case the old rule got wrong: it dropped the wrap on those axes, on the
    reasoning that a wrapped plane inside an absorber has nothing left in it. Attenuated
    is not absent. Measured worst per-frequency relative flux against CPU MEEP:

        x absorbing        1.45e-02 clipped   ->  1.37e-07 wrapped
        x and y absorbing  2.33e-02 clipped   ->  1.59e-07 wrapped

    against 2.03e-07 for the identical plane with no transverse absorber at all, which
    is the band a correct wrap belongs in. The clipped answer is not a rounding of the
    right one — it is a different power through the same plane.
    """
    reference = _flux_oracle(tmp_path, "x_and_y_pml")
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    driver.setup_pml({"x": 4, "y": 4})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.0, 0.0, -0.5), "size": (0.0, 0.0, 0.0)})
    monitor = driver.add_flux_monitor(fcen=1.0, df=0.4, nfreq=3,
                                      center=(0.0, 0.0, 0.8), size=(2.0, 2.0, 0.0), direction=2)
    assert monitor.periodic == (True, True, True), "the layer must not have taken the wrap away"
    driver.run(until=12.0)
    spectrum = np.asarray(monitor.get_flux_spectrum(), dtype=float)
    worst = float(np.max(np.abs(spectrum - reference) / np.abs(reference)))
    assert worst < 1e-5, (
        f"Worst per-frequency flux error {worst:.3e} against CPU MEEP; measured 1.6e-07. "
        f"Clipping the wrap on the absorbing axes reads 2.3e-02 here."
    )

    # The control that says the bar is discriminating: the same run with the wrap dropped
    # on the absorbing axes, which is what the engine used to do.
    clipped_driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    clipped_driver.setup_pml({"x": 4, "y": 4})
    clipped_driver._monitor_periodic_axes = lambda: (False, False, True)
    clipped_driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                               "fwidth": 0.5, "center": (0.0, 0.0, -0.5), "size": (0.0, 0.0, 0.0)})
    clipped_monitor = clipped_driver.add_flux_monitor(
        fcen=1.0, df=0.4, nfreq=3, center=(0.0, 0.0, 0.8), size=(2.0, 2.0, 0.0), direction=2)
    clipped_driver.run(until=12.0)
    clipped = np.asarray(clipped_monitor.get_flux_spectrum(), dtype=float)
    clipped_worst = float(np.max(np.abs(clipped - reference) / np.abs(reference)))
    assert clipped_worst > 1000.0 * worst, (
        f"the clipped plane reads {clipped_worst:.3e}, not far enough from {worst:.3e} for this "
        f"comparison to be measuring the wrap"
    )


def test_a_source_may_span_any_face_including_the_one_that_absorbs():
    # The per-face clearance refusals are retired on Cartesian grids: a grating
    # source fills the periodic cross section, and one reaching into the absorbing
    # face is admitted too (the deposit-mirror/withdraw fixes; parity in
    # test_from_meep source_spans_the_pml at 6.3e-07). The folded case — the stored
    # quadrant's implied lower layer — is likewise admitted and must simply build
    # and step; its parity twin is the folded corpus family.
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    driver.setup_pml({"z": 8})
    driver.add_source({"component": "Ey", "frequency": 1.0,
                       "center": (0.0, 0.0, 0.0), "size": (2.0, 2.0, 0.0)})
    driver.add_source({"component": "Ey", "frequency": 1.0,
                       "center": (0.0, 0.0, -1.4), "size": (2.0, 2.0, 0.0)})
    driver.run(num_steps=5)
    assert np.all(np.isfinite(driver.get_field("Ey")))
    folded = FdtdDriver(cell_size=(2, 2, 4), resolution=10, symmetry=("X",))
    folded.setup_pml({"x": {"high": 4}})
    assert folded.pml.thickness_by_face == ((0, 4), (0, 0), (0, 0))
    # The BUILD-time refusal is gone, and so is the linear runtime guard: a
    # folded periodic axis now stores its second-mirror plane and reflects past
    # it, so a sheet depositing beside the far x face is simply served (the old
    # zero-ghost treatment measured 8.8e-01 there; the live-face fold is now at
    # the CPU-MEEP floor — test_a_folded_run_matches_cpu_meep_with_its_far_face_live).
    folded.add_source({"component": "Ez", "frequency": 1.0,
                       "center": (0.0, 0.0, 0.0), "size": (1.6, 0.0, 0.0)})
    folded.run(num_steps=5)
    sheet_field = folded.get_field("Ez")
    assert np.all(np.isfinite(sheet_field)) and float(np.max(np.abs(sheet_field))) > 0.0


# ---------------------------------------------------------------------------
# Dispersive materials and conductivity, against CPU MEEP
# ---------------------------------------------------------------------------

_DISPERSION_ORACLE_SCRIPT = '''"""CPU-MEEP references for the dispersive-material and conductivity cases.

Every case sets eps_averaging=False. MEEP subpixel-averages the nondispersive part
of epsilon but never sigma (doc/docs/Materials.md:60, anisotropic_averaging.cpp:333-345
point-samples sigma while :252-257 averages chi1inv), so leaving it on would make
eps_inf and sigma disagree about where an interface is and the comparison would be
measuring smoothing rather than dispersion. Courant is stated explicitly on both sides.
"""
import sys

import numpy as np
import meep as mp

case = sys.argv[1]
output_path = sys.argv[2]

LORENTZ = mp.Medium(epsilon=2.25, E_susceptibilities=[
    mp.LorentzianSusceptibility(frequency=1.1, gamma=0.05, sigma=0.6)])
CONDUCTIVE = mp.Medium(epsilon=3.4, D_conductivity=2 * np.pi * 0.42 * 0.101 / 3.4)

if case == "material_model":
    # Stepping-free: the material model alone, at 25 frequencies spanning the band.
    frequencies = np.linspace(0.4, 2.0, 25)
    two_term = mp.Medium(epsilon=1.5, E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=1.0, gamma=0.05, sigma=0.8),
        mp.LorentzianSusceptibility(frequency=1.4, gamma=0.1, sigma=0.5)])
    rows = []
    for medium in (LORENTZ, two_term, CONDUCTIVE):
        rows.append([np.asarray(medium.epsilon(float(f)))[0, 0] for f in frequencies])
    np.save(output_path, np.array(rows, dtype=np.complex128))
    raise SystemExit(0)

if case == "lorentz_no_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 4), resolution=10, default_material=LORENTZ,
        eps_averaging=False, Courant=0.5,
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.05))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
elif case == "drude_no_pml":
    medium = mp.Medium(epsilon=1.0, E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=1.0, gamma=0.05, sigma=1.0)])
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 4), resolution=10, default_material=medium,
        eps_averaging=False, Courant=0.4,
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.6), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.05))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
elif case == "lossless_no_pml":
    medium = mp.Medium(epsilon=1.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=2.0, gamma=0.0, sigma=1.0)])
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 4), resolution=10, default_material=medium,
        eps_averaging=False, Courant=0.5,
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.6), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.05))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
elif case == "two_term_no_pml":
    medium = mp.Medium(epsilon=1.5, E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=1.0, gamma=0.05, sigma=0.8),
        mp.LorentzianSusceptibility(frequency=1.4, gamma=0.1, sigma=0.5)])
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 4), resolution=10, default_material=medium,
        eps_averaging=False, Courant=0.4,
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.6), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.05))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
elif case == "conductivity_no_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 4), resolution=10, default_material=CONDUCTIVE,
        eps_averaging=False, Courant=0.5,
        sources=[mp.Source(mp.ContinuousSource(frequency=0.42), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.05))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
elif case == "conductivity_uniform_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 6), resolution=10, default_material=CONDUCTIVE,
        eps_averaging=False, Courant=0.5, boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(mp.ContinuousSource(frequency=0.42), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(0.6, 0.6, 0))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
elif case == "conductivity_z_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 6), resolution=10, default_material=CONDUCTIVE,
        eps_averaging=False, Courant=0.5,
        boundary_layers=[mp.PML(1.0, direction=mp.Z)],
        sources=[mp.Source(mp.ContinuousSource(frequency=0.42), component=mp.Ex,
                           center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
    simulation.run(until=3)
    np.save(output_path, np.asarray(simulation.get_array(component=mp.Ex)))
    raise SystemExit(0)
elif case == "lorentz_uniform_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 6), resolution=10, default_material=LORENTZ,
        eps_averaging=False, Courant=0.5, boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
elif case.startswith("lorentz_bloch:"):
    kx, ky, kz = (float(value) for value in case.split(":", 1)[1].split(","))
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 2), resolution=10, default_material=LORENTZ,
        eps_averaging=False, Courant=0.5,
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, 0.05))],
        force_complex_fields=True, k_point=mp.Vector3(kx, ky, kz))
elif case == "dispersive_slab":
    # Vacuum + a dispersive slab normal to z, on an ODD z cell count, with eps_inf = 1
    # everywhere so the ONLY material step is sigma. Geometry rather than a
    # material_function: with a material_function MEEP enumerates susceptibilities from
    # geometry / extra_materials / default_material only (meepgeom.cpp:1816-1830) and
    # would silently allocate no polarizations at all.
    slab = mp.Medium(epsilon=1.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.1, gamma=0.05, sigma=0.6)])
    simulation = mp.Simulation(
        cell_size=mp.Vector3(0.4, 0.4, 6.05), resolution=20, eps_averaging=False, Courant=0.5,
        geometry=[mp.Block(size=mp.Vector3(mp.inf, mp.inf, 2.0), center=mp.Vector3(0, 0, 0.5),
                           material=slab)],
        boundary_layers=[mp.PML(1.0, direction=mp.Z)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.6), component=mp.Ex,
                           center=mp.Vector3(0, 0, -1.75), size=mp.Vector3(0.4, 0.4, 0))],
        force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
    simulation.run(until=12)
    np.save(output_path, np.asarray(simulation.get_array(component=mp.Ex)))
    raise SystemExit(0)
else:
    raise SystemExit("unknown dispersion oracle case: " + case)

simulation.run(until=3)
np.save(output_path, np.asarray(simulation.get_array(component=mp.Ez)))
'''


def _dispersion_oracle(tmp_path, case):  # One CPU-MEEP dispersive reference, in its own process.
    script_path = tmp_path / "meep_dispersion_oracle.py"
    script_path.write_text(_DISPERSION_ORACLE_SCRIPT, encoding="utf-8")
    slug = "".join(character if character.isalnum() else "_" for character in case)
    output_path = tmp_path / f"{slug}.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), case, str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP dispersion oracle '{case}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


def _uniform_dispersive_driver(cell, resolution, epsilon, terms, courant=0.5, k_point=(0, 0, 0),
                               conductivity=None, symmetry=()):
    driver = FdtdDriver(cell_size=cell, resolution=resolution, courant=courant,
                        force_complex_fields=True, k_point=k_point, symmetry=symmetry)
    driver.set_epsilon(np.full(driver.shape, epsilon, dtype=np.float32))
    if conductivity is not None:
        driver.set_conductivity(conductivity)
    for term, sigma in terms:
        driver.add_susceptibility(term, sigma)
    return driver


@requires_meep
@skip_without_meep
def test_material_model_matches_mp_medium_epsilon(tmp_path):
    """The stepping-free oracle: eps(f) against ``mp.Medium.epsilon(f)``, 25 frequencies.

    This is the cheapest possible place to catch a units error, a dropped 2*pi or a
    flipped Drude sign — it costs no time steps at all, and every one of those
    mistakes leaves the subsequent fields perfectly smooth. The conductivity case
    additionally pins MEEP's ``(1 + i sigma_D/omega)`` multiplier, which scales the
    WHOLE permittivity rather than adding an imaginary part.
    """
    reference = _dispersion_oracle(tmp_path, "material_model")
    frequencies = np.linspace(0.4, 2.0, 25)
    cases = [
        (2.25, [(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)], None),
        (1.5, [(Susceptibility(1.0, 0.05, DRUDE), 0.8),
               (Susceptibility(1.4, 0.1, LORENTZIAN), 0.5)], None),
        (3.4, [], 2 * np.pi * 0.42 * 0.101 / 3.4),
    ]
    for row, (epsilon, terms, conductivity) in zip(reference, cases):
        driver = _uniform_dispersive_driver((1.0, 1.0, 1.0), 10, epsilon, terms,
                                            conductivity=conductivity)
        measured = np.array([driver.get_epsilon(frequency=float(f)).flat[0] for f in frequencies])
        driver.close()
        error = np.max(np.abs(measured - row) / np.abs(row))
        assert error < 1e-6, f"eps(f) differs from mp.Medium.epsilon by {error:.2e}"
    # Blind control: the instantaneous permittivity is NOT the dispersive one, so a
    # get_epsilon that ignored the susceptibilities would fail this by a wide margin.
    driver = _uniform_dispersive_driver((1.0, 1.0, 1.0), 10, 2.25,
                                        [(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)])
    blind = np.max(np.abs(driver.get_epsilon().flat[0] - reference[0]) / np.abs(reference[0]))
    driver.close()
    assert blind > 0.05


@requires_meep
@skip_without_meep
@pytest.mark.parametrize(
    "case, epsilon, terms, courant, source, bound",
    [
        ("lorentz_no_pml", 2.25, [(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)], 0.5,
         {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.05), "size": (0.0, 0.0, 0.0)},
         1e-5),
        ("drude_no_pml", 1.0, [(Susceptibility(1.0, 0.05, DRUDE), 1.0)], 0.4,
         {"component": "Ez", "source_type": "gaussian", "frequency": 1.0, "fwidth": 0.6,
          "center": (0.0, 0.0, -1.05), "size": (0.0, 0.0, 0.0)}, 5e-6),
        ("lossless_no_pml", 1.0, [(Susceptibility(2.0, 0.0, LORENTZIAN), 1.0)], 0.5,
         {"component": "Ez", "source_type": "gaussian", "frequency": 1.0, "fwidth": 0.6,
          "center": (0.0, 0.0, -1.05), "size": (0.0, 0.0, 0.0)}, 5e-6),
        ("two_term_no_pml", 1.5, [(Susceptibility(1.0, 0.05, DRUDE), 0.8),
                                  (Susceptibility(1.4, 0.1, LORENTZIAN), 0.5)], 0.4,
         {"component": "Ez", "source_type": "gaussian", "frequency": 1.0, "fwidth": 0.6,
          "center": (0.0, 0.0, -1.05), "size": (0.0, 0.0, 0.0)}, 5e-6),
    ],
)
def test_homogeneous_dispersive_media_match_cpu_meep(tmp_path, case, epsilon, terms, courant,
                                                     source, bound):
    """Homogeneous Lorentz, Drude, lossless Sellmeier and a two-term medium, no PML.

    Compared on the COMPLEX field: a susceptibility is a phase shift as much as an
    amplitude, and every magnitude stays plausible under a detuned resonance. The
    two-term case is not optional — a kernel that applied only ``polarizations[0]``
    passes every single-term case here.

    Each case carries a blind control: the identical run with the susceptibilities
    removed must miss the reference by more than 5e-2, so no case can pass because
    dispersion had no effect.
    """
    reference = _dispersion_oracle(tmp_path, case)[1:, 1:, 1:]
    driver = _uniform_dispersive_driver((2, 2, 4), 10, epsilon, terms, courant=courant)
    driver.add_source(dict(source))
    driver.run(until=3.0)
    measured = driver.get_field("Ez")
    driver.close()
    assert measured.shape == reference.shape
    complex_l2 = _complex_relative_l2(measured, reference)
    assert complex_l2 < bound, f"{case}: complex Ez relative L2 {complex_l2:.3e} exceeds {bound:g}"

    blind_driver = _uniform_dispersive_driver((2, 2, 4), 10, epsilon, [], courant=courant)
    blind_driver.add_source(dict(source))
    blind_driver.run(until=3.0)
    blind = _complex_relative_l2(blind_driver.get_field("Ez"), reference)
    blind_driver.close()
    assert blind > 5e-2, f"{case}: a non-dispersive run came within {blind:.2e} of the reference"


@requires_meep
@skip_without_meep
def test_a_conductive_medium_matches_cpu_meep(tmp_path):
    """MEEP's own worked example: eps = 3.4 + 0.101i at f = 0.42 via D_conductivity.

    doc/docs/Materials.md gives the medium and the number, so the case is the
    documentation's rather than one chosen to pass. The blind control is the same run
    with the conductivity dropped, which must be far off — a conductivity's whole
    effect is a slow amplitude decay that a correlation check would never notice.
    """
    reference = _dispersion_oracle(tmp_path, "conductivity_no_pml")[1:, 1:, 1:]
    conductivity = 2 * np.pi * 0.42 * 0.101 / 3.4
    source = {"component": "Ez", "frequency": 0.42, "center": (0.0, 0.0, -1.05),
              "size": (0.0, 0.0, 0.0)}
    driver = _uniform_dispersive_driver((2, 2, 4), 10, 3.4, [], conductivity=conductivity)
    driver.add_source(dict(source))
    driver.run(until=3.0)
    complex_l2 = _complex_relative_l2(driver.get_field("Ez"), reference)
    driver.close()
    assert complex_l2 < 5e-6, f"Conductive Ez relative L2 {complex_l2:.3e}"

    lossless = _uniform_dispersive_driver((2, 2, 4), 10, 3.4, [])
    lossless.add_source(dict(source))
    lossless.run(until=3.0)
    blind = _complex_relative_l2(lossless.get_field("Ez"), reference)
    lossless.close()
    assert blind > 5e-2, f"a lossless run came within {blind:.2e} of the conductive reference"


@requires_meep
@skip_without_meep
def test_a_dispersive_medium_inside_a_uniform_pml_matches_cpu_meep(tmp_path):
    """Dispersion composed with the absorber — the case that sees W against E.

    Driving the polarization from the STORED E instead of from ``(D - sum P)*inv_eps``
    is exactly zero-error outside a PML, so every other dispersive case here passes
    with that defect. Only this one can see it, which is why it is not optional.
    """
    reference = _dispersion_oracle(tmp_path, "lorentz_uniform_pml")[1:, 1:, 1:]
    driver = _uniform_dispersive_driver((3, 3, 6), 10, 2.25,
                                        [(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)])
    driver.setup_pml(10)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.5),
                       "size": (1.0, 1.0, 0.0)})
    driver.run(until=3.0)
    complex_l2 = _complex_relative_l2(driver.get_field("Ez"), reference)
    driver.close()
    assert complex_l2 < 2e-5, f"Dispersive PML Ez relative L2 {complex_l2:.3e}"

    blind = _uniform_dispersive_driver((3, 3, 6), 10, 2.25, [])
    blind.setup_pml(10)
    blind.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.5),
                      "size": (1.0, 1.0, 0.0)})
    blind.run(until=3.0)
    blind_l2 = _complex_relative_l2(blind.get_field("Ez"), reference)
    blind.close()
    assert blind_l2 > 5e-2


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("k_point", [(0.25, 0.0, 0.0), (0.3, -0.2, 0.15)])
def test_a_dispersive_medium_under_bloch_boundaries_matches_cpu_meep(tmp_path, k_point):
    """Dispersion composed with a Bloch phase, on the complex field.

    A boundary phase contributes nothing but phase, and so does a detuned resonance,
    so the two could cancel in a magnitude comparison. Both are held to the same
    complex bound, and the k = 0 control must be far away.
    """
    case = "lorentz_bloch:{},{},{}".format(*k_point)
    reference = _dispersion_oracle(tmp_path, case)[1:, 1:, 1:]
    driver = _uniform_dispersive_driver((2, 2, 2), 10, 2.25,
                                        [(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)],
                                        k_point=k_point)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                       "size": (0.0, 0.0, 0.0)})
    driver.run(until=3.0)
    complex_l2 = _complex_relative_l2(driver.get_field("Ez"), reference)
    driver.close()
    assert complex_l2 < 1e-5, f"Bloch+dispersion complex Ez relative L2 {complex_l2:.3e}"

    unphased = _uniform_dispersive_driver((2, 2, 2), 10, 2.25,
                                          [(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)])
    unphased.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                         "size": (0.0, 0.0, 0.0)})
    unphased.run(until=3.0)
    blind = _complex_relative_l2(unphased.get_field("Ez"), reference)
    unphased.close()
    assert blind > 5e-2, f"a k = 0 run came within {blind:.2e} of the k = {k_point} reference"


@requires_meep
@skip_without_meep
def test_a_dispersive_slab_matches_cpu_meep_and_discriminates_its_registration(tmp_path):
    """Heterogeneous sigma, on an ODD z cell count, with eps_inf uniform.

    Making eps_inf 1 everywhere leaves sigma as the ONLY material step, so this
    measures the sigma registration and nothing else. The z cell count is odd (121 at
    resolution 20 on a 6.05 cell) because ``Grid.origin_doubled`` puts MEEP's origin
    half a cell above ``-L/2`` exactly then — the half-cell error that cost this
    engine 4.8e-2 was invisible for years because every case used even counts.

    The two wrong registrations are run side by side and must be measurably worse,
    so the test proves it can DISCRIMINATE rather than merely agreeing.
    """
    reference = _dispersion_oracle(tmp_path, "dispersive_slab")[1:, 1:, 1:]
    term = Susceptibility(1.1, 0.05, LORENTZIAN)
    def inside(z):
        return np.logical_and(z >= -0.5, z <= 1.5)

    def build(sampler):
        driver = FdtdDriver(cell_size=(0.4, 0.4, 6.05), resolution=20, courant=0.5,
                            force_complex_fields=True)
        assert driver.shape[2] % 2 == 1, "this case exists to exercise an odd cell count"
        driver.set_epsilon(np.ones(driver.shape, dtype=np.float32))
        driver.add_susceptibility(term, sampler(driver))
        driver.setup_pml({"z": 20})
        driver.add_source({"component": "Ex", "source_type": "gaussian", "frequency": 1.0,
                           "fwidth": 0.6, "center": (0.0, 0.0, -1.75), "size": (0.4, 0.4, 0.0)})
        driver.run(until=12.0)
        result = driver.get_field("Ex")
        driver.close()
        return result

    def correct(driver):  # Each component at its OWN Yee position — MEEP's rule.
        return {name: sample_region(driver.grid, lambda x, y, z: np.where(inside(z), 0.6, 0.0), name)
                for name in E_COMPONENTS}

    def cell_centred(driver):  # Half a cell out on every axis.
        positions = driver.grid.axis_origin(2) + (np.arange(driver.shape[2]) + 0.5) * driver.dx
        volume = np.where(inside(positions), 0.6, 0.0).astype(np.float32)
        return np.broadcast_to(volume, driver.shape).copy()

    def from_half_length(driver):  # Measured from -L/2, which is the origin only for even counts.
        positions = -driver.grid.Lz / 2.0 + np.arange(driver.shape[2]) * driver.dx
        volume = np.where(inside(positions), 0.6, 0.0).astype(np.float32)
        return np.broadcast_to(volume, driver.shape).copy()

    measured = _complex_relative_l2(build(correct), reference)
    assert measured < 5e-4, f"dispersive slab complex Ex relative L2 {measured:.3e}"
    for label, sampler in (("cell-centred", cell_centred), ("from -L/2", from_half_length)):
        wrong = _complex_relative_l2(build(sampler), reference)
        assert wrong > 20 * max(measured, 1e-6), (
            f"{label} sigma registration scored {wrong:.2e} against {measured:.2e} for the correct "
            f"one; this comparison cannot tell the registrations apart and so proves nothing"
        )


def test_a_dispersive_symmetry_run_reproduces_the_full_domain_exactly():
    """Folded against full with a live polarization — the exactness criterion, unchanged.

    ``update_P`` runs over the whole stored array with no ownership mask and no
    boundary pass, on the argument that cell 0 of a mirrored axis gets the right P by
    induction from a drive field ``fill_symmetry_bc_D`` has already repaired. That is
    an assumption, and this is what discharges it: the non-dispersive engine matches
    its folded run to round-off, so a half-repaired mirror cell in P would show up
    immediately.
    """
    cell_size = (4.0, 2.0, 2.0)
    resolution = 10
    term, sigma = Susceptibility(1.1, 0.05, LORENTZIAN), 0.6

    def build(symmetry):
        driver = FdtdDriver(cell_size=cell_size, resolution=resolution,
                            force_complex_fields=True, symmetry=symmetry)
        driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
        driver.add_susceptibility(term, sigma)
        return driver

    full = build(())
    folded = build(("X",))
    nx_full, ny_full, nz_full = full.shape
    seed = _symmetric_seed(nx_full, ny_full, nz_full, full.dx)
    center_index = nx_full // 2
    full.set_field("Dz", seed)
    folded.set_field("Dz", np.concatenate((seed[center_index - 1:], seed[0:1]), axis=0))
    for _ in range(24):
        full.step()
        folded.step()
    assert np.max(np.abs(full.fields.polarizations[0].P["Ez"])) > 0.0, (
        "the polarization never grew; the equivalence check would be vacuous"
    )
    _assert_folded_matches_full(full, folded, ("Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ez"))
    # The polarization itself must fold too, not merely the fields derived from it.
    full_P = np.asarray(full.fields.polarizations[0].P["Ez"])
    np.testing.assert_allclose(
        folded.fields.polarizations[0].P["Ez"],
        np.concatenate((full_P[center_index - 1:], full_P[0:1]), axis=0),
        rtol=0.0, atol=1e-5 * float(np.max(np.abs(full_P))),
    )
    full.close()
    folded.close()


@requires_meep
@skip_without_meep
def test_a_conductivity_inside_a_uniform_pml_matches_cpu_meep_from_either_setup_order(
    tmp_path,
):
    """The three-stage f_cond/f_u/D recurrence closes the former 5.2% gap."""
    reference = _dispersion_oracle(tmp_path, "conductivity_uniform_pml")[1:, 1:, 1:]
    conductivity = 2 * np.pi * 0.42 * 0.101 / 3.4
    measured = []
    for conductivity_first in (False, True):
        driver = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=True)
        driver.set_epsilon(np.full(driver.shape, 3.4, dtype=np.float32))
        if conductivity_first:
            driver.set_conductivity(conductivity)
            driver.setup_pml(10)
        else:
            driver.setup_pml(10)
            driver.set_conductivity(conductivity)
        driver.add_source(
            {
                "component": "Ez",
                "frequency": 0.42,
                "center": (0.0, 0.0, -1.5),
                "size": (0.6, 0.6, 0.0),
            }
        )
        driver.run(until=3.0)
        field = driver.get_field("Ez")
        assert all(
            getattr(driver.fields, f"f_cond_D{axis}") is not None
            for axis in ("x", "y", "z")
        )
        measured.append(field)
        driver.close()

    for field in measured:
        complex_l2 = _complex_relative_l2(field, reference)
        assert complex_l2 < 2e-5, f"Conductivity + PML Ez relative L2 {complex_l2:.3e}"
    np.testing.assert_array_equal(measured[0], measured[1])

    # The source stencil must stay two cells inside the layer: at one cell the
    # recurrence is 1.4e-2 from MEEP and at the edge it is 3.7e-2.
    touching = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=True)
    touching.set_epsilon(np.full(touching.shape, 3.4, dtype=np.float32))
    touching.setup_pml(10)
    touching.set_conductivity(conductivity)
    with pytest.raises(ValueError, match="at least 2 grid cells"):
        touching.add_source(
            {
                "component": "Ez",
                "frequency": 0.42,
                "center": (0.0, 0.0, -1.5),
                "size": (1.0, 1.0, 0.0),
            }
        )
    touching.close()

    late_pml = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=True)
    late_pml.set_epsilon(np.full(late_pml.shape, 3.4, dtype=np.float32))
    late_pml.set_conductivity(conductivity)
    late_pml.add_source(
        {
            "component": "Ez",
            "frequency": 0.42,
            "center": (0.0, 0.0, -1.5),
            "size": (1.0, 1.0, 0.0),
        }
    )
    with pytest.raises(ValueError, match="at least 2 grid cells"):
        late_pml.setup_pml(10)
    late_pml.close()


@requires_meep
@skip_without_meep
def test_a_conductivity_inside_a_z_only_pml_matches_cpu_meep(tmp_path):
    """The one-axis special case composes transverse conductivity with z absorption."""
    reference = _dispersion_oracle(tmp_path, "conductivity_z_pml")[1:, 1:, 1:]
    conductivity = 2 * np.pi * 0.42 * 0.101 / 3.4
    driver = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 3.4, dtype=np.float32))
    driver.set_conductivity(conductivity)
    driver.setup_pml({"z": 10})
    driver.add_source(
        {
            "component": "Ex",
            "frequency": 0.42,
            "center": (0.0, 0.0, -1.5),
            "size": (1.0, 1.0, 0.0),
        }
    )
    driver.run(until=3.0)
    measured = driver.get_field("Ex")
    driver.close()
    complex_l2 = _complex_relative_l2(measured, reference)
    assert complex_l2 < 2e-5, f"Conductivity + z-PML Ex relative L2 {complex_l2:.3e}"


def test_an_integrated_source_under_conductivity_is_undamped_and_unscaled():
    # This pairing used to be REFUSED: MEEP applies an integrated source's dipole in
    # update_eh, downstream of the conductivity, so it is never damped — and the old
    # telescoped increment left the dipole standing in D where the conductive
    # recurrence decayed it. The withdraw slot (VolumeSource.withdraw, called before
    # step_D) removes the standing offset before every ladder, and the driver
    # injects the integrated dipole WITHOUT the condinv scaling step_source applies
    # to currents (MEEP skips integrated sources there entirely, step.cpp:300).
    # Measured against CPU MEEP (2-D 8x8 cell, res 10, eps 2.25, uniform
    # sigma_D = 0.4, CW Ez is_integrated point source, until 15, complex Ez relative
    # L2 over the volume): 2.0417e-07, where the telescoped path reads 6.3802e-02.
    # This non-oracle residue of the old refusal test pins that the pairing now
    # steps: a finite, nonzero field with the conductivity genuinely active.
    driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 3.4, dtype=np.float32))
    driver.set_conductivity(0.5)
    driver.add_source({"component": "Ez", "frequency": 0.42, "center": (0.0, 0.0, -1.05),
                       "size": (0.0, 0.0, 0.0), "is_integrated": True})
    driver.run(num_steps=300)
    damped = driver.get_field("Ez")
    assert np.all(np.isfinite(damped)) and float(np.max(np.abs(damped))) > 0.0
    driver.close()
    twin = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=True)
    twin.set_epsilon(np.full(twin.shape, 3.4, dtype=np.float32))
    twin.add_source({"component": "Ez", "frequency": 0.42, "center": (0.0, 0.0, -1.05),
                     "size": (0.0, 0.0, 0.0), "is_integrated": True})
    twin.run(num_steps=300)
    lossless = twin.get_field("Ez")
    twin.close()
    # The peak sits AT the source cell, where the standing dipole dominates either
    # way; the volume norm is what the loss visibly eats (measured ratio 0.543).
    assert float(np.linalg.norm(damped)) < 0.75 * float(np.linalg.norm(lossless)), (
        "The conductivity did not damp the run: either it was dropped, or the "
        "integrated dipole is being injected around it in the wrong slot."
    )


# ---------------------------------------------------------------------------
# Real-valued (float32) field mode — MEEP's DEFAULT — against CPU MEEP's default
# ---------------------------------------------------------------------------
#
# Every floor above was measured with BOTH codes in complex storage. A real run is a
# different arrangement of the same physics, so it gets its own oracle: MEEP built
# WITHOUT force_complex_fields, which allocates one plane per component and injects
# `f[c][0][i] -= real(A)` (step.cpp:295-312, the `if (!is_real)` line skipped). The
# cases below are the four the mode has to survive — plain, absorbing, dispersive and
# mirror-folded — plus odd cell counts on all three axes and a source at no round
# coordinate, because this engine's two historical silent-wrong-answer classes were
# both invisible to even counts and centred sources.

_REAL_ORACLE_SCRIPT = '''"""CPU-MEEP references with MEEP in its DEFAULT real-field storage."""
import sys

import numpy as np
import meep as mp

case = sys.argv[1]
output_path = sys.argv[2]

LORENTZ = mp.Medium(epsilon=2.25, E_susceptibilities=[
    mp.LorentzianSusceptibility(frequency=1.1, gamma=0.05, sigma=0.6)])
# force_complex_fields is deliberately never set below: its absence IS the case.
# k_point=(0,0,0) is still passed, because that is what makes MEEP's faces Periodic
# rather than metallic, and use_bloch refuses real fields only for k != 0
# (boundaries.cpp:103 `if (is_real && kk != 0.0)`).
K0 = mp.Vector3(0, 0, 0)

if case == "point_no_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 4), resolution=10,
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.05))],
        k_point=K0)
    component, until = mp.Ez, 3.0
elif case == "sheet_with_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 6), resolution=10, boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))],
        k_point=K0)
    component, until = mp.Ez, 3.0
elif case == "lorentz_no_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 4), resolution=10, default_material=LORENTZ,
        eps_averaging=False, Courant=0.5,
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, -1.05))],
        k_point=K0)
    component, until = mp.Ez, 3.0
elif case == "odd_cells_off_centre":
    # 2.1 / 2.3 / 3.1 at resolution 10 -> 21 / 23 / 31 cells, all odd, and a source
    # centre that is neither a mirror plane nor a round coordinate.
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2.1, 2.3, 3.1), resolution=10,
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0.25, -0.35, -0.65))],
        k_point=K0)
    component, until = mp.Ez, 2.5
elif case == "odd_cells_off_centre_pml":
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2.1, 2.3, 3.1), resolution=10, boundary_layers=[mp.PML(0.5)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.6), component=mp.Ey,
                           center=mp.Vector3(0.25, -0.35, -0.65))],
        k_point=K0)
    component, until = mp.Ey, 2.5
elif case == "folded_cell":
    # MEEP runs the FULL domain with no Mirror: the engine's folded quadrant is compared
    # against the half of MEEP's cell it stores, exactly as the complex folded case does.
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3, 3, 4), resolution=10,
        sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                           center=mp.Vector3(0, 0, 0.05))],
        k_point=K0)
    component, until = mp.Ez, 1.0
elif case == "both_modes_point":
    # One case run BOTH ways inside MEEP, so the engine's real-vs-complex equality can
    # be held against the same statement about CPU MEEP rather than against nothing.
    volumes = []
    for complex_fields in (False, True):
        arguments = dict(
            cell_size=mp.Vector3(2, 2, 4), resolution=10,
            sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                               center=mp.Vector3(0, 0, -1.05))],
            k_point=K0)
        if complex_fields:
            arguments["force_complex_fields"] = True
        one = mp.Simulation(**arguments)
        one.run(until=3.0)
        assert one.fields.is_real == (not complex_fields), "MEEP did not honour the storage mode"
        volumes.append(np.asarray(one.get_array(component=mp.Ez)).astype(np.complex128))
    np.save(output_path, np.stack(volumes))
    raise SystemExit(0)
else:
    raise SystemExit("unknown real oracle case: " + case)

simulation.run(until=until)
# The assertion is the point of the file: a case that silently ran complex would
# validate nothing real-specific, and every bound below would still pass.
assert simulation.fields.is_real, "MEEP allocated complex storage for a real-mode case"
array = np.asarray(simulation.get_array(component=component))
assert array.dtype == np.float32, f"MEEP returned {array.dtype} for a real-mode case"
np.save(output_path, array)
'''


def _real_oracle(tmp_path, case):  # One CPU-MEEP reference with MEEP in real (default) storage.
    script_path = tmp_path / "meep_real_oracle.py"
    script_path.write_text(_REAL_ORACLE_SCRIPT, encoding="utf-8")
    output_path = tmp_path / f"real_{case}.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), case, str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP real-mode oracle '{case}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


def _real_mode_driver(case, complex_fields=False):  # The engine side of one real-mode oracle case.
    """The driver configuration for each ``_REAL_ORACLE_SCRIPT`` case, in either storage.

    Returns (driver, component, until). One body serves both storage modes so the real
    and complex legs of :func:`test_real_and_complex_field_modes_are_the_same_physics`
    are textually the same run — which is the property that makes a byte comparison
    between them mean something — and so neither can drift from the configuration the
    oracle comparison used.
    """
    if case == "point_no_pml":
        driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, force_complex_fields=complex_fields)
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.05),
                           "size": (0.0, 0.0, 0.0)})
        return driver, "Ez", 3.0
    if case == "sheet_with_pml":
        driver = FdtdDriver(cell_size=(3, 3, 6), resolution=10, force_complex_fields=complex_fields)
        driver.setup_pml(10)
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.5),
                           "size": (1.0, 1.0, 0.0)})
        return driver, "Ez", 3.0
    if case == "lorentz_no_pml":
        driver = FdtdDriver(cell_size=(2, 2, 4), resolution=10, courant=0.5,
                            force_complex_fields=complex_fields)
        driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
        driver.add_susceptibility(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.05),
                           "size": (0.0, 0.0, 0.0)})
        return driver, "Ez", 3.0
    if case == "odd_cells_off_centre":
        driver = FdtdDriver(cell_size=(2.1, 2.3, 3.1), resolution=10,
                            force_complex_fields=complex_fields)
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.25, -0.35, -0.65),
                           "size": (0.0, 0.0, 0.0)})
        return driver, "Ez", 2.5
    if case == "odd_cells_off_centre_pml":
        driver = FdtdDriver(cell_size=(2.1, 2.3, 3.1), resolution=10,
                            force_complex_fields=complex_fields)
        driver.setup_pml(5)
        driver.add_source({"component": "Ey", "source_type": "gaussian", "frequency": 1.0,
                           "fwidth": 0.6, "center": (0.25, -0.35, -0.65), "size": (0.0, 0.0, 0.0)})
        return driver, "Ey", 2.5
    if case == "folded_cell":
        driver = FdtdDriver(cell_size=(3, 3, 4), resolution=10,
                            force_complex_fields=complex_fields, symmetry=("X",))
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                           "size": (0.0, 0.0, 0.0)})
        return driver, "Ez", 1.0
    raise AssertionError(f"no driver configuration for real-mode case {case!r}")


def _run_real_mode_case(case, complex_fields):  # Run one case in one storage mode, return its field.
    driver, component, until = _real_mode_driver(case, complex_fields=complex_fields)
    driver.run(until=until)
    field = driver.get_field(component)
    driver.close()
    return field, component, until


@requires_meep
@skip_without_meep
@pytest.mark.parametrize(
    "case, bound",
    [
        ("point_no_pml", 1.0e-6),            # measured 2.51e-07 (complex floor 2.41e-07)
        ("sheet_with_pml", 1.5e-6),          # measured 3.62e-07 (complex floor 4.00e-07)
        ("lorentz_no_pml", 1.0e-5),          # measured 2.84e-06 (complex floor 3.2e-06)
        ("odd_cells_off_centre", 1.0e-6),    # measured 1.34e-07, all three axes odd
        ("odd_cells_off_centre_pml", 1.0e-6),  # measured 7.72e-08, odd + absorbing + Ey
    ],
)
def test_real_field_mode_matches_cpu_meep_in_its_own_default_mode(tmp_path, case, bound):
    """Real float32 storage against MEEP's real float32 storage, four physics paths.

    The bounds are the complex-mode floors carried over, at ~4x headroom, and they are
    not free: the two codes agree here only if the real path takes MEEP's
    ``real(amp * current * dt)`` at injection AND keeps every stepping coefficient real.
    A source convention error (``imag(A)``, ``abs(A)``, a factor of two) survives no
    leg of this, because it is an amplitude and phase error on the one field the whole
    run is driven by.

    The last two cases carry odd cell counts on all three axes and a source at
    (0.25, -0.35, -0.65) — no round coordinate, no symmetry plane. That pairing is what
    caught this engine's half-cell registration bug, which every even-count centred
    case passed.
    """
    reference = _real_oracle(tmp_path, case)
    assert reference.dtype == np.float32, "the real-mode oracle did not return real storage"
    driver, component, until = _real_mode_driver(case)
    driver.run(until=until)
    measured = driver.get_field(component)
    assert measured.dtype == np.float32, (
        f"the engine promoted its real run to {measured.dtype}; the mode's whole point is the "
        f"narrower storage, and a promoted run would pass every number below"
    )
    driver.close()
    # MEEP's get_array returns N+1 points per axis, index 0 being the periodic duplicate.
    trimmed = reference[1:, 1:, 1:]
    assert measured.shape == trimmed.shape
    relative_l2 = _complex_relative_l2(measured, trimmed)
    assert relative_l2 < bound, (
        f"{case}: real-mode {component} relative L2 against real-mode CPU MEEP is "
        f"{relative_l2:.3e}, over the {bound:g} bound"
    )


@requires_meep
@skip_without_meep
def test_a_mirror_folded_real_run_matches_cpu_meep(tmp_path):
    """The fold and real storage composed, against MEEP's full-domain real run.

    Both halves of the complex folded case apply unchanged: the quadrant is exact only
    while its far x face is dark, so this runs to t = 1.0 and keeps the unfolded run of
    the same case as the control that says the residual belongs to the fold and not to
    the storage mode. Measured folded 1.58e-07, unfolded 1.56e-07 at the 2026-08-06
    re-baselining — the folded residual collapsed onto the unfolded control with the
    folded-far-face rework (it read 5.76e-06 before it, against the complex mode's
    then-4.6e-06 / 2.7e-07 pair, differing only because the denominator is
    ``|Re(E)|`` rather than ``|E|``).
    """
    reference = _real_oracle(tmp_path, "folded_cell")
    folded, component, until = _real_mode_driver("folded_cell")
    folded.run(until=until)
    measured = folded.get_field(component)
    assert measured.dtype == np.float32
    folded.close()

    unfolded = FdtdDriver(cell_size=(3, 3, 4), resolution=10, force_complex_fields=False)
    unfolded.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                         "size": (0.0, 0.0, 0.0)})
    unfolded.run(until=until)
    centre = unfolded.grid.nx // 2  # Quadrant cell q is MEEP index centre + q.
    unfolded_error = _complex_relative_l2(unfolded.get_field(component), reference[1:, 1:, 1:])
    unfolded.close()

    assert unfolded_error < 1e-6, f"the unfolded real control is not at its floor: {unfolded_error:.3e}"
    # The stored layout carries one row past MEEP's owned window (the
    # second-mirror plane cell of a periodic even fold); rows 0..n-2 are the
    # owned quadrant this comparison has always pinned, and the plane cell is
    # covered by the live-far-face cases and the bit-for-bit fold equivalences.
    folded_error = _complex_relative_l2(measured[:-1], reference[centre:, 1:, 1:])
    assert folded_error < 2e-5, (
        f"the folded real run's relative L2 against real CPU MEEP is {folded_error:.3e}; "
        f"measured 5.76e-06"
    )


@pytest.mark.parametrize(
    "case",
    ["point_no_pml", "sheet_with_pml", "lorentz_no_pml", "odd_cells_off_centre",
     "odd_cells_off_centre_pml", "folded_cell"],
)
def test_real_and_complex_field_modes_are_the_same_physics(case):
    """A run with no complex ingredient must give the same answer either way — exactly.

    Every coefficient in the curl kernels, the constitutive relations, the split-field
    PML recurrence and the polarization ADE is real, and injection takes ``real(A)``, so
    the real plane of a complex run never reads its imaginary plane. The consequence is
    not "close": the float32 arithmetic performed is identical operation for operation,
    so the real run is the complex run's real part to the BYTE. CPU MEEP's own two modes
    have the same relationship, which the companion test measures directly.

    The discrimination leg matters more than the equality. If the complex field happened
    to be (near-)real, ``Re`` would not be a distinguished reduction of it — ``Im``,
    ``|.|`` and ``2*Re`` would all land in the same place — and the equality above would
    hold for a run that had lost or gained a whole plane. So the imaginary part is
    required to carry real weight and each of those three reductions is required to miss
    by a wide margin. (What the *injection* convention itself is, rather than which
    reduction of the answer matches, is pinned in ``test_driver_integration.py``'s
    ``test_real_mode_takes_meeps_real_of_the_current_and_no_other_convention``, which
    writes MEEP's ``A = amp * current * dt`` out longhand.)
    """
    real_field, component, _ = _run_real_mode_case(case, complex_fields=False)
    complex_field, _, _ = _run_real_mode_case(case, complex_fields=True)

    assert real_field.dtype == np.float32 and complex_field.dtype == np.complex64
    expected = np.ascontiguousarray(complex_field.real)
    assert real_field.tobytes() == expected.tobytes(), (
        f"{case}: real mode is not bit-for-bit the real part of the complex run "
        f"(relative L2 {_complex_relative_l2(real_field, expected):.3e})"
    )

    imaginary_weight = float(np.linalg.norm(complex_field.imag) / np.linalg.norm(complex_field.real))
    assert imaginary_weight > 0.1, (
        f"{case}: the complex run's imaginary plane carries only {imaginary_weight:.3e} of its "
        f"real plane, so 'real == Re(complex)' does not discriminate a convention here"
    )
    for label, reduction in (
        ("Im", complex_field.imag),
        ("|.|", np.abs(complex_field)),
        ("2*Re", 2.0 * complex_field.real),
    ):
        gap = _complex_relative_l2(reduction, expected)
        assert gap > 0.1, f"{case}: the {label} reduction comes within {gap:.3e} of Re; Re is not distinguished here"


@requires_meep
@skip_without_meep
def test_cpu_meeps_own_two_storage_modes_have_the_same_relationship(tmp_path):
    """The oracle for the equality above: MEEP real == Re(MEEP complex), byte for byte.

    Without this, "the engine's real mode is exactly its complex run's real part" could
    be a property of this engine alone rather than of the algorithm — and a reader would
    have no way to tell an exact reproduction from a rounded one. Measured relative L2
    0.0, with identical bytes, on MEEP 1.29.0.
    """
    volumes = _real_oracle(tmp_path, "both_modes_point")
    meep_real, meep_complex = volumes[0], volumes[1]
    assert float(np.abs(meep_real.imag).max()) == 0.0, "the real-mode volume is not real"
    difference = _complex_relative_l2(meep_real.real, meep_complex.real)
    assert difference == 0.0, (
        f"CPU MEEP's real mode differs from the real part of its complex mode by {difference:.3e}; "
        f"the engine's own byte equality is then the wrong target"
    )
    imaginary_weight = float(
        np.linalg.norm(meep_complex.imag) / np.linalg.norm(meep_complex.real)
    )
    assert imaginary_weight > 0.1, "MEEP's complex volume is nearly real; the comparison is weak"


# ======================================================================================
# Instantaneous chi2 / chi3 — mp.Medium(chi2=..., chi3=...).
#
# MEEP implements these natively, so there is a real oracle and the bar is the engine's
# own reproduction floor rather than a percentage. Every case carries a LINEAR CONTROL:
# the same run with the nonlinearity removed, measured against the same MEEP reference.
# Without it a comparison that landed on the floor would prove only that the engine and
# MEEP agree about the linear part, which they already did.
# ======================================================================================

_NONLINEAR_ORACLE_SCRIPT = '''"""CPU-MEEP reference for an instantaneous chi2/chi3 medium."""
import sys

import numpy as np
import meep as mp

case = sys.argv[1]
output_path = sys.argv[2]

geometry, arguments = case.split(":", 1)
chi2, chi3 = (float(value) for value in arguments.split(","))
epsilon = 4.0 if geometry.endswith("_eps4") else 1.0
geometry = geometry[: -len("_eps4")] if geometry.endswith("_eps4") else geometry
medium = mp.Medium(epsilon=epsilon, chi2=chi2, chi3=chi3)
layers = []

if geometry == "point":
    cell = mp.Vector3(2, 2, 4)
    source = mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ez,
                       center=mp.Vector3(0, 0, -1.05))
elif geometry == "sheet_pml":
    cell = mp.Vector3(1, 1, 6)
    layers = [mp.PML(1.0, direction=mp.Z)]
    source = mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ex,
                       center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))
elif geometry == "odd":
    # Odd cell counts on all three axes (11 x 9 x 37) and a source at no round
    # coordinate: the two defects this engine has actually shipped.
    cell = mp.Vector3(1.1, 0.9, 3.7)
    source = mp.Source(mp.ContinuousSource(frequency=1.0), component=mp.Ey,
                       center=mp.Vector3(0.05, -0.05, -0.85))
else:
    raise SystemExit("unknown nonlinear oracle geometry: " + geometry)

simulation = mp.Simulation(
    cell_size=cell,
    resolution=10,
    default_material=medium,
    sources=[source],
    boundary_layers=layers,
    force_complex_fields=True,
    k_point=mp.Vector3(0, 0, 0),
)
simulation.run(until=3)
np.save(output_path, np.stack([
    np.asarray(simulation.get_array(component=mp.Ex)),
    np.asarray(simulation.get_array(component=mp.Ey)),
    np.asarray(simulation.get_array(component=mp.Ez)),
]))
'''

# Driver-side twin of the oracle script's geometries. One table, so the two codes cannot
# describe different systems: the case string selects the same row on both sides.
_NONLINEAR_GEOMETRY = {
    "point": {"cell": (2.0, 2.0, 4.0), "component": "Ez", "center": (0.0, 0.0, -1.05),
              "size": (0.0, 0.0, 0.0), "pml": 0},
    "sheet_pml": {"cell": (1.0, 1.0, 6.0), "component": "Ex", "center": (0.0, 0.0, -1.5),
                  "size": (1.0, 1.0, 0.0), "pml": 10},
    "odd": {"cell": (1.1, 0.9, 3.7), "component": "Ey", "center": (0.05, -0.05, -0.85),
            "size": (0.0, 0.0, 0.0), "pml": 0},
}


def _nonlinear_oracle(tmp_path, case):  # One CPU-MEEP nonlinear reference, in its own process.
    script_path = tmp_path / "meep_nonlinear_oracle.py"
    script_path.write_text(_NONLINEAR_ORACLE_SCRIPT, encoding="utf-8")
    slug = "".join(character if character.isalnum() else "_" for character in case)
    output_path = tmp_path / f"{slug}.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), case, str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP nonlinear oracle '{case}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    # MEEP's get_array returns (N+1) points per axis, the extra plane being the
    # periodic duplicate at index 0; index i + 1 is this grid's cell i.
    return np.load(output_path)[:, 1:, 1:, 1:]


def _nonlinear_driver(geometry, chi2, chi3):  # The same system through this engine.
    epsilon = 4.0 if geometry.endswith("_eps4") else 1.0
    spec = _NONLINEAR_GEOMETRY[geometry[: -len("_eps4")] if geometry.endswith("_eps4") else geometry]
    driver = FdtdDriver(cell_size=spec["cell"], resolution=10, force_complex_fields=True)
    if epsilon != 1.0:
        driver.set_epsilon(np.full(driver.shape, epsilon))
    if spec["pml"]:
        driver.setup_pml({"z": spec["pml"]})
    if chi2:
        driver.set_chi2(chi2)
    if chi3:
        driver.set_chi3(chi3)
    driver.add_source({"component": spec["component"], "frequency": 1.0,
                       "center": spec["center"], "size": spec["size"]})
    driver.run(until=3.0)
    return np.stack([driver.get_field(name) for name in ("Ex", "Ey", "Ez")]), driver


# Coefficients chosen so the nonlinearity is worth several percent of the field while the
# Pade expansion parameter stays an order below its 1/3 validity bound — the regime a
# nonlinear-optics run actually lives in. The recorded floors are the measurements.
_NONLINEAR_CASES = {
    # case: (geometry, chi2, chi3, floor, the linear engine's error on the same reference)
    "kerr": ("point", 0.0, 6e-4, 1.0e-06, 1.7e-01),
    "pockels": ("point", 6e-4, 0.0, 1.0e-06, 3.9e-02),
    "both_in_a_dielectric": ("point_eps4", 4e-3, 4e-4, 1.0e-06, 2.1e-02),
    "kerr_with_pml": ("sheet_pml", 0.0, 0.05, 1.0e-06, 5.9e-02),
    "odd_counts": ("odd", 4e-4, 4e-4, 1.0e-06, 8.9e-02),
}


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", sorted(_NONLINEAR_CASES))
def test_an_instantaneous_nonlinearity_matches_cpu_meep(tmp_path, case):
    """chi2/chi3 against MEEP's own implementation, with the linear engine as the control.

    Measured, complex Ex/Ey/Ez relative L2 (nonlinear engine, then the same run with the
    nonlinearity removed, against the same MEEP reference)::

        kerr (chi3=6e-4, point)            4.04e-07   vs   1.78e-01
        pockels (chi2=6e-4, point)         3.45e-07   vs   4.00e-02
        both, eps=4 (chi2=4e-3,chi3=4e-4)  1.92e-07   vs   2.07e-02
        kerr + PML (chi3=0.05, sheet)      3.62e-07   vs   6.01e-02
        chi2+chi3, 11x9x37 odd cell        2.73e-07   vs   9.04e-02

    Every one lands on the engine's EXISTING linear parity floor (1.5e-07 .. 4.0e-07),
    so the nonlinearity costs nothing and adds nothing; the control is four to five
    orders worse, so none of them can be passing by accident.

    ``both_in_a_dielectric`` is the only case that pins the powers of ``chi1inv`` in
    MEEP's ``calc_nonlinear_u`` (``chi1inv^2`` for c2, ``chi1inv^3`` for c3): at
    eps = 1 those are indistinguishable and swapping them changes nothing, while at
    eps = 4 it costs 4.4e-02. ``kerr`` is the only one that pins the transverse Yee
    average, because a plane wave has nothing in the other two components — dropping
    it costs 2.5e-02 there and nothing at all on the sheet.
    """
    geometry, chi2, chi3, floor, control_scale = _NONLINEAR_CASES[case]
    reference = _nonlinear_oracle(tmp_path, f"{geometry}:{chi2},{chi3}")
    produced, driver = _nonlinear_driver(geometry, chi2, chi3)
    assert produced.shape == reference.shape

    relative_l2 = _complex_relative_l2(produced, reference)
    assert relative_l2 < floor, (
        f"{case}: complex E relative L2 {relative_l2:.3e} against CPU MEEP exceeds the "
        f"recorded floor {floor:.1e}"
    )

    linear, _ = _nonlinear_driver(geometry, 0.0, 0.0)
    control = _complex_relative_l2(linear, reference)
    assert control > 0.2 * control_scale, (
        f"{case}: a LINEAR engine scored {control:.3e} against the nonlinear reference, far "
        f"below the {control_scale:.1e} this case is supposed to separate. The nonlinearity is "
        f"too weak here to prove anything and the comparison above is vacuous."
    )
    assert control > 1e3 * relative_l2, (
        f"{case}: the nonlinear result ({relative_l2:.3e}) is not decisively better than the "
        f"linear one ({control:.3e})"
    )
    # The run stayed well inside the Pade approximant's validity, so the agreement is
    # not being propped up by both codes saturating.
    margin = driver.nonlinear_margin()
    assert margin.expansion < 0.1, f"{case}: expansion parameter {margin.expansion:.3g} is not small"


def test_a_zero_nonlinearity_leaves_the_driver_bit_for_bit_unchanged():
    """set_chi2(0) / set_chi3(0) must not cost a single ULP of the recorded floors.

    Held at the DRIVER level as well as the kernel level, because the driver is where
    the storage mode switches: a zero pair that switched E into storage would change
    what ``get_E`` returns and change the answer without changing the physics.
    """
    digests = []
    for install in (False, True):
        driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=False)
        driver.setup_pml({"z": 6})
        if install:
            driver.set_chi2(0.0)
            driver.set_chi3(np.zeros(driver.shape, dtype=np.float32))
        assert not driver.fields.has_nonlinearity
        assert driver.fields.stores_E == driver.fields._pml_active
        driver.add_source({"component": "Ex", "frequency": 1.0, "center": (0.0, 0.0, -0.3),
                           "size": (1.0, 1.0, 0.0)})
        driver.run(num_steps=40)
        digests.append(np.stack([driver.get_field(name, cell_centered=False)
                                 for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]))
    np.testing.assert_array_equal(
        digests[0], digests[1],
        err_msg="installing an identically zero chi2/chi3 moved the field; the linear path is "
                "supposed to be untouched, and every recorded floor was measured without it",
    )
    assert np.abs(digests[0]).max() > 0.0, "the control run carries no field"


def test_third_harmonic_generation_appears_only_with_chi3():
    """The physics a linear engine cannot fake: drive at f, measure 3f.

    A CW plane wave in a chi3 medium, DFT'd at a downstream plane over a whole number
    of periods once the transient has settled. Over an exact integer number of periods
    a pure sinusoid at f contributes NOTHING at 3f, so the linear control measures the
    engine's own round-off and any peak above it is generated.

    Measured |E(3f)| / |E(f)| at the monitor plane: 4.8e-07 with chi3 = 0 (round-off)
    against 1.6e-01 with chi3 = 0.2 — five orders of separation. The same run also
    scales as the cube of the drive, which is the signature of a chi3 term rather than
    of any nonlinearity at all.
    """
    linear = _harmonic_ratio(chi2=0.0, chi3=0.0, harmonic=3)
    nonlinear = _harmonic_ratio(chi2=0.0, chi3=0.2, harmonic=3)
    assert linear < 1e-5, (
        f"a LINEAR run already shows |E(3f)|/|E(f)| = {linear:.2e}; the window is not an integer "
        f"number of periods and the measurement has no floor to stand on"
    )
    assert nonlinear > 1e-2, f"chi3 produced no third harmonic ({nonlinear:.2e})"
    assert nonlinear > 1e3 * linear

    # Cubic in the drive amplitude: E(3f) ~ A^3 while E(f) ~ A, so the RATIO ~ A^2 and
    # doubling the amplitude must quadruple it. A quadratic (chi2-like) term would give
    # a factor of two and a numerical artefact would give one.
    weak = _harmonic_ratio(chi2=0.0, chi3=0.02, harmonic=3, amplitude=0.5)
    strong = _harmonic_ratio(chi2=0.0, chi3=0.02, harmonic=3, amplitude=1.0)
    assert strong / weak == pytest.approx(4.0, rel=0.15), (
        f"the third harmonic scales as A^{np.log2(strong / weak):.2f} rather than the A^2 a "
        f"chi3 ratio must follow"
    )


def test_second_harmonic_generation_appears_only_with_chi2():
    """The chi2 counterpart: drive at f, measure 2f, with the same linear floor.

    Measured |E(2f)| / |E(f)|: 7.7e-07 with chi2 = 0 against 4.1e-01 with chi2 = 0.2.
    A chi3 medium of the same strength gives 4.0e-07 — BELOW the linear floor, i.e.
    nothing at all, because a Kerr medium is centrosymmetric and radiates only odd
    harmonics. That separates the two nonlinearities from each other rather than
    merely from zero, which no amount of "some nonlinearity is happening" can fake.
    """
    linear = _harmonic_ratio(chi2=0.0, chi3=0.0, harmonic=2)
    nonlinear = _harmonic_ratio(chi2=0.2, chi3=0.0, harmonic=2)
    assert linear < 1e-5, f"a LINEAR run already shows |E(2f)|/|E(f)| = {linear:.2e}"
    assert nonlinear > 1e-2, f"chi2 produced no second harmonic ({nonlinear:.2e})"
    assert nonlinear > 1e3 * linear

    # A Kerr medium is centrosymmetric and radiates only odd harmonics.
    kerr = _harmonic_ratio(chi2=0.0, chi3=0.2, harmonic=2)
    assert kerr < 1e-3 * nonlinear, (
        f"chi3 produced a second harmonic {kerr:.2e} comparable to chi2's {nonlinear:.2e}; the "
        f"two terms are not being applied at their own orders"
    )
    # Quadratic in the drive: the ratio is linear in amplitude.
    weak = _harmonic_ratio(chi2=0.02, chi3=0.0, harmonic=2, amplitude=0.5)
    strong = _harmonic_ratio(chi2=0.02, chi3=0.0, harmonic=2, amplitude=1.0)
    assert strong / weak == pytest.approx(2.0, rel=0.15), (
        f"the second harmonic scales as A^{np.log2(strong / weak):.2f} rather than the A^1 a "
        f"chi2 ratio must follow"
    )


_HARMONIC_FREQUENCY = 1.0 / 3.0
_HARMONIC_PERIOD = 3.0
_HARMONIC_RESOLUTION = 15


def _harmonic_plane_wave(chi2, chi3, amplitude, frequencies, length=10.0, transverse=0.2,
                         settle=30.0, windows=6):
    """A settled CW plane wave in a nonlinear medium, DFT'd over whole periods.

    PML on z only, periodic across a transverse cell one source wide, so the run is a
    plane wave and the transverse Yee average contributes nothing — the harmonics come
    from the driven component alone and cannot be an artefact of the interpolation.

    The monitor is RESET after the transient, so its window is an exact whole number of
    periods of f and therefore of every harmonic of f. Over such a window the DFT of a
    pure sinusoid at f is exactly zero at 3f: that is what gives the linear control a
    floor at round-off instead of at the leakage of an arbitrary window.
    """
    driver = FdtdDriver(cell_size=(transverse, transverse, length),
                        resolution=_HARMONIC_RESOLUTION, force_complex_fields=False)
    driver.setup_pml({"z": int(round(1.5 * _HARMONIC_RESOLUTION))})
    if chi2:
        driver.set_chi2(chi2)
    if chi3:
        driver.set_chi3(chi3)
    driver.add_source({"component": "Ex", "frequency": _HARMONIC_FREQUENCY,
                       "center": (0.0, 0.0, -0.25 * length),
                       "size": (transverse, transverse, 0.0),
                       "amplitude": amplitude, "width": 3.0})
    monitor = driver.add_dft_monitor(frequencies=frequencies, components=("Ex",),
                                     center=(0.0, 0.0, 0.25 * length),
                                     size=(transverse, transverse, 0.0))
    driver.run(until=settle)
    monitor.reset()
    driver.run(num_steps=int(round(windows * _HARMONIC_PERIOD / driver.dt)))
    return driver, monitor


def _harmonic_ratio(chi2, chi3, harmonic, amplitude=1.0):  # |E(n*f)| / |E(f)| at the monitor.
    _driver, monitor = _harmonic_plane_wave(
        chi2, chi3, amplitude,
        frequencies=[_HARMONIC_FREQUENCY, harmonic * _HARMONIC_FREQUENCY],
    )
    fundamental = float(np.abs(monitor.get_dft("Ex", 0)).mean())
    assert fundamental > 0.0, "the fundamental never reached the monitor plane"
    return float(np.abs(monitor.get_dft("Ex", 1)).mean()) / fundamental


def test_the_kerr_index_shift_matches_its_closed_form():
    """dn = 3*chi3*E0^2/(8*n0), the AC Kerr coefficient MEEP documents.

    doc/docs/Units_and_Nonlinearity.md gives ``n2 = 3*chi3/(4*n0^2)`` with
    ``dn = n2*I`` and ``I = n0*E0^2/2`` for a plane wave, i.e.
    ``dn = 3*chi3*E0^2/(8*n0)``. That is the leading term of the exact statement:
    E = E0*cos(wt) makes E^3 carry ``(3/4)*E0^2`` at the fundamental, so
    ``eps_eff = eps + (3/4)*chi3*E0^2`` and ``n_eff = n0 + 3*chi3*E0^2/(8*n0)``.

    Measured here as the EXTRA phase a plane wave picks up over four length units of
    Kerr medium, against the same run with chi3 = 0 — the phase of the ratio of ratios,
    which is immune to the several full wraps in the raw phase difference. E0 comes
    from the time domain over the same window, so nothing depends on the DFT's scale.

    Measured (resolution 15): chi3 = 0.05 -> +4.72e-03 against +4.61e-03 predicted
    (+2.3%); chi3 = 0.1 -> +9.13e-03 against +9.04e-03 (+1.0%); chi3 = 0.2 ->
    +1.68e-02 against +1.74e-02 (-3.6%). The deviation changes SIGN across the range,
    which is the signature of two competing errors — grid dispersion at small shifts,
    the next order of the power series at large ones — rather than of a scale factor.
    """
    planes = (1.0, 5.0)
    reference, _ = _kerr_phase_and_amplitude(0.0, planes)
    for chi3, tolerance in ((0.05, 0.05), (0.1, 0.05), (0.2, 0.06)):
        ratio, amplitude = _kerr_phase_and_amplitude(chi3, planes)
        shift = float(np.angle(ratio / reference))
        measured = shift / (2.0 * np.pi * _HARMONIC_FREQUENCY * (planes[1] - planes[0]))
        predicted = 3.0 * chi3 * amplitude ** 2 / 8.0
        assert measured > 0.0, f"chi3 = {chi3} must RAISE the index, measured {measured:.3e}"
        assert measured == pytest.approx(predicted, rel=tolerance), (
            f"chi3 = {chi3}: measured index shift {measured:.4e} against the closed form "
            f"{predicted:.4e} (E0 = {amplitude:.4f})"
        )


def _kerr_phase_and_amplitude(chi3, planes, length=16.0, transverse=0.2, settle=40.0, windows=6):
    """Steady-state phase across two planes, plus the peak |Ex| between them.

    Returns the COMPLEX ratio of the two planes' DFT accumulators rather than a phase
    difference: at f = 1/3 over four length units the raw difference wraps several
    times, and the extra shift the nonlinearity contributes is a hundredth of a
    radian. The ratio of one run's ratio to the linear run's isolates it unambiguously.
    """
    driver = FdtdDriver(cell_size=(transverse, transverse, length),
                        resolution=_HARMONIC_RESOLUTION, force_complex_fields=False)
    driver.setup_pml({"z": int(round(2.0 * _HARMONIC_RESOLUTION))})
    if chi3:
        driver.set_chi3(chi3)
    driver.add_source({"component": "Ex", "frequency": _HARMONIC_FREQUENCY,
                       "center": (0.0, 0.0, -5.0), "size": (transverse, transverse, 0.0),
                       "amplitude": 1.0, "width": 3.0})
    monitors = [driver.add_dft_monitor(frequencies=[_HARMONIC_FREQUENCY], components=("Ex",),
                                       center=(0.0, 0.0, position),
                                       size=(transverse, transverse, 0.0))
                for position in planes]
    driver.run(until=settle)
    for monitor in monitors:
        monitor.reset()
    low, high = (int(round((position + length / 2.0) / driver.dx)) for position in planes)
    peak = 0.0
    for _ in range(int(round(windows * _HARMONIC_PERIOD / driver.dt))):
        driver.step()
        peak = max(peak, float(np.abs(driver.fields.get_E("Ex")[:, :, low:high + 1]).max()))
    accumulated = [complex(monitor.get_dft("Ex", 0).mean()) for monitor in monitors]
    assert abs(accumulated[0]) > 0.0 and peak > 0.0, "the probe never measured a field"
    return accumulated[1] / accumulated[0], peak


def test_a_run_that_leaves_the_pade_validity_range_is_refused_not_returned():
    """Past the approximant's pole the recovered E is finite, smooth and backwards.

    MEEP performs no check and hands the field back; this engine measures the distance
    to the pole every stability interval and raises. The bound is derived, not tuned:
    ``1 + 2*c2 + 3*c3 >= 1 - 3*(|c2| + |c3|)``, so an expansion parameter below 1/3
    keeps the denominator strictly positive.

    The pairing that makes this necessary is ordinary: chi3 = 0.5 with a unit-amplitude
    point source, neither of which looks extreme on its own.
    """
    driver = FdtdDriver(cell_size=(2.0, 2.0, 4.0), resolution=10, force_complex_fields=True)
    driver.set_chi3(0.5)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.05),
                       "size": (0.0, 0.0, 0.0)})
    with pytest.raises(FdtdNonlinearityOutOfRange, match="Pade"):
        driver.run(until=3.0)

    # The identical run at a coefficient the approximant does describe completes, so
    # the guard is not simply refusing every nonlinear run.
    gentle = FdtdDriver(cell_size=(2.0, 2.0, 4.0), resolution=10, force_complex_fields=True)
    gentle.set_chi3(6e-4)
    gentle.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, -1.05),
                       "size": (0.0, 0.0, 0.0)})
    gentle.run(until=3.0)
    margin = gentle.nonlinear_margin()
    assert 0.0 < margin.expansion < 1.0 / 3.0
    assert margin.denominator > 1.0 - 3.0 * margin.expansion - 1e-6
    assert gentle.nonlinear_margin().component in ("Ex", "Ey", "Ez")


def test_the_driver_refuses_a_nonlinearity_it_cannot_represent():
    """Every rejected spelling, with the storage left untouched behind it."""
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=True)
    with pytest.raises(ValueError, match="electric components"):
        driver.set_chi3({"Hz": 0.1})
    with pytest.raises(ValueError, match="matches neither the grid shape"):
        driver.set_chi2(np.ones((3, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="must be finite"):
        driver.set_chi3(float("inf"))
    with pytest.raises(ValueError, match="real, instantaneous"):
        driver.set_chi3(0.1 + 0.2j)
    with pytest.raises(ValueError, match="3-D array"):
        driver.set_chi3(np.ones(8, dtype=np.float32))
    assert not driver.fields.has_nonlinearity and not driver.fields.stores_E

    # MEEP's (N+1) boundary-padded shape is accepted and trimmed, as every other
    # material volume in this driver is.
    padded = np.full(tuple(n + 1 for n in driver.shape), 0.01, dtype=np.float32)
    driver.set_chi3(padded)
    assert driver.fields.has_nonlinearity
    assert tuple(driver.fields.chi3_for("Ez").shape) == driver.shape
    # A per-component mapping reaches each component's own volume, and a component
    # left out of the mapping is zero rather than inheriting a neighbour's.
    driver.set_chi3({"Ez": 0.02})
    assert driver.fields.nonlinear_components == ("Ez",)


def test_a_mirror_symmetry_is_refused_for_chi2_and_allowed_for_chi3():
    """The fold is a claim about the MATERIAL as well as the geometry.

    A chi2 medium is not centrosymmetric, so a mirror plane that makes the driven
    component odd is not a symmetry of it and the folded half would be reconstructed
    from a symmetry the run does not have. Measured: with chi3 alone a folded run
    reproduces the full-domain run bit for bit; adding chi2 breaks it by 4.1e-02 in a
    single step, with nothing about the field looking wrong.
    """
    driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10,
                        force_complex_fields=True, symmetry=("X",))
    with pytest.raises(ValueError, match="centrosymmetric"):
        driver.set_chi2(0.01)
    assert not driver.fields.has_nonlinearity

    driver.set_chi3(0.01)
    assert driver.fields.nonlinear_components == ("Ex", "Ey", "Ez")
    # Ey and Ez stay EVEN about an even X plane, so a chi2 on those is accepted.
    driver.set_chi2({"Ey": 0.01, "Ez": 0.01})
    assert driver.fields.nonlinear_components == ("Ex", "Ey", "Ez")


def test_a_nonlinearity_added_after_stepping_is_refused():
    """The storage switch cannot happen mid-run without a window of empty readbacks.

    Installing chi2/chi3 moves E from computed-on-demand to stored, and the array it
    allocates is zeros; between that and the next ``update_E`` every ``get_field``
    would report an empty volume for a run carrying a field. Same rule
    ``add_susceptibility`` and ``set_conductivity`` follow, for the same class of
    reason.
    """
    driver = FdtdDriver(cell_size=(1.0, 1.0, 2.0), resolution=10, force_complex_fields=False)
    driver.add_source({"component": "Ex", "frequency": 1.0, "center": (0.0, 0.0, -0.3),
                       "size": (1.0, 1.0, 0.0)})
    driver.run(num_steps=5)
    assert np.abs(driver.get_field("Ex")).max() > 0.0
    with pytest.raises(RuntimeError, match="before the first step"):
        driver.set_chi3(0.01)
    assert not driver.fields.has_nonlinearity

    # An identically zero pair installs nothing, so it stays permitted mid-run: it is
    # not a mode switch, and refusing it would make a no-op raise.
    driver.set_chi3(0.0)
    assert not driver.fields.stores_E
    # And reset() clears the run, so the material can be set up again.
    driver.reset()
    driver.set_chi3(0.01)
    assert driver.fields.has_nonlinearity and driver.fields.stores_E


# --------------------------------------------------------------------------------
# End-to-end: a resonant cavity, its Q pulled out by Harminv, against CPU MEEP.
# --------------------------------------------------------------------------------

_CAVITY_ORACLE_SCRIPT = '''"""CPU-MEEP reference: cavity modes from mp.Harminv, for the Harminv parity test."""
import sys

import numpy as np
import meep as mp

sigma = float(sys.argv[1])
output_path = sys.argv[2]

simulation = mp.Simulation(
    cell_size=mp.Vector3(1, 1, 1),
    resolution=14,
    default_material=mp.Medium(epsilon=4.0, D_conductivity=sigma),
    k_point=mp.Vector3(0, 0, 0),          # Bloch-periodic, not MEEP's default metal walls.
    force_complex_fields=True,
    sources=[
        mp.Source(
            mp.GaussianSource(0.5, fwidth=0.4),
            component=mp.Ez,
            center=mp.Vector3(0.13, -0.07, 0.19),
        )
    ],
)
probe = mp.Harminv(mp.Ez, mp.Vector3(-0.21, 0.11, 0.05), 0.5, 0.25)
simulation.run(until=8.0)                 # Sources off, and the record starts at t = 8.
simulation.run(probe, until=8.0 + 3000 * simulation.fields.dt)
np.save(
    output_path,
    np.array(
        [[mode.freq, mode.decay, mode.Q, mode.amp.real, mode.amp.imag, abs(mode.err)]
         for mode in probe.modes],
        dtype=np.float64,
    ).reshape(-1, 6),
)
'''


@pytest.fixture(scope="module")
def cavity_oracle(tmp_path_factory):  # One CPU-MEEP process per conductivity, reused by the tests.
    directory = tmp_path_factory.mktemp("cavity_oracle")
    script_path = directory / "cavity_oracle.py"
    script_path.write_text(_CAVITY_ORACLE_SCRIPT, encoding="utf-8")
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    def load(sigma):
        output_path = directory / f"cavity_{sigma!r}.npy"
        if not output_path.exists():
            completed = subprocess.run(
                [sys.executable, str(script_path), repr(sigma), str(output_path)],
                capture_output=True, text=True, env=environment, timeout=900,
            )
            assert completed.returncode == 0 and output_path.exists(), (
                f"CPU-MEEP cavity oracle (sigma={sigma}) failed (exit {completed.returncode}).\n"
                f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
            )
        return np.load(output_path)

    return load


def _cavity_modes(sigma, collect_steps=3000, probe_offset=(0.0, 0.0, 0.0)):
    """Drive the cavity, then hand the run loop a Harminv probe, exactly as MEEP is used."""
    driver = FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=14, force_complex_fields=True)
    driver.set_epsilon(np.full(driver.shape, 4.0, dtype=np.float32))
    if sigma:
        driver.set_conductivity(sigma)
    driver.add_source({
        "source_type": "gaussian", "component": "Ez", "frequency": 0.5, "fwidth": 0.4,
        "center": (0.13, -0.07, 0.19), "size": (0.0, 0.0, 0.0), "amplitude": 1.0,
    })
    driver.run(until=8.0)                 # Sources off before the record opens, as MEEP requires.
    point = tuple(base + shift for base, shift in zip((-0.21, 0.11, 0.05), probe_offset))
    probe = Harminv(c="Ez", pt=point, fcen=0.5, df=0.25)
    probe.rel_amp_thresh = 0.01
    driver.run(num_steps=collect_steps, step_functions=[probe])
    modes = list(probe.modes)
    samples = len(probe.data)
    interval = probe.data_dt
    driver.close()
    return modes, samples, interval


@requires_meep
@skip_without_meep
@pytest.mark.parametrize(
    "sigma, expected_Q, freq_floor, quality_floor",
    [
        (0.0314159, 99.2, 1e-6, 2e-4),      # Q ~ 99, the record decays by ~e^-1.3
        (0.00785398, 396.9, 1e-6, 1e-3),    # Q ~ 397, only ~34 % decay over the record
    ],
    ids=["Q100", "Q400"],
)
def test_cavity_Q_matches_cpu_meep_and_its_closed_form(
    cavity_oracle, sigma, expected_Q, freq_floor, quality_floor
):
    """THE END-TO-END PATH: step a cavity, sample a point every step, invert, get Q.

    This is the only test that runs the stepper, ``get_field_point``, the
    step-function protocol and ``harminv`` in one line of causation, and it is the
    path a MEEP user actually writes::

        probe = Harminv(c="Ez", pt=..., fcen=..., df=...)
        driver.run(num_steps=..., step_functions=[probe])
        probe.modes[0].Q

    THE CAVITY is a Bloch-periodic box of uniform eps = 4 carrying MEEP's
    ``D_conductivity``. It is a genuine resonator — the box supports discrete standing
    waves at ``f = |n| / (L*sqrt(eps))`` — and its Q is known in CLOSED FORM, which is
    what makes this more than an agreement between two codes. With
    ``eps(w) = eps(1 + i*sigma_D/w)`` the dispersion relation
    ``eps*w^2 + i*eps*sigma_D*w - c^2 k^2 = 0`` gives ``Im w = -sigma_D / 2`` whatever
    the mode, so

        Q = |Re w| / (-2 Im w) = 2*pi*f / sigma_D

    with ``f`` the resonance the run actually lands on. Taking the MEASURED f rather
    than the ideal 0.5 makes the prediction immune to grid dispersion, which at 14
    cells per wavelength moves the line to 0.49606.

    THE MEDIUM IS UNIFORM ON PURPOSE. A structured cavity would fold the epsilon
    REGISTRATION into the comparison — this engine stores one scalar ``inv_eps``
    shared by the three D components, and a z-stratified slab needs different sampling
    for Dz than for Dx and Dy. Measured: a hand-built slab sampled at cell centres
    lands 4.5e-02 away from MEEP in resonance frequency, which says nothing about
    Harminv. Epsilon registration has its own oracle
    (``test_grating_transmission_spectrum_matches_cpu_meep``, 3.8e-07 with three
    half-cell controls); this test is about the resonance and its extraction.

    MEASURED, engine against CPU MEEP running ``mp.Harminv`` on the same cavity from
    the same starting time:

        sigma_D      freq            Q             amp        Q vs the closed form
        0.0314159    2.2e-07         3.0e-05       5.4e-05     ours 1.6e-05, MEEP 1.4e-05
        0.00785398   3.2e-08         2.0e-04       5.0e-06     ours 1.4e-04, MEEP 5.9e-05

    The AMPLITUDE agreement is the part that pins ``get_field_point`` in the plane the
    mode varies in: both records are one scalar per step at the same point from the
    same time, so an x or y registration error moves it and moves it hard — measured
    9.1e-01 for a QUARTER cell, 1.8e+00 for a half, against the 5.4e-05 the aligned
    probe scores. The control below asserts exactly that, so the comparison cannot pass
    by both codes reading nothing.

    WHAT THE AMPLITUDE CANNOT SEE, said plainly: the z position. An Ez mode in this box
    must have ``k`` in the xy plane (E is transverse), so every mode this probe can see
    is uniform in z, and moving the probe a whole cell along z changes the fitted
    amplitude by 3e-08. Ez's half-integer axis IS z, so this test does not constrain
    the half-cell Yee shift at all — mutating ``0.5 * iyee_shift`` to zero leaves it
    green. That pin lives in the unit tests
    (``test_get_field_point_reduces_to_the_stored_sample_at_a_yee_position`` and
    ``test_get_field_point_matches_an_independent_trilinear_read``, both of which the
    same mutation fails). Recorded here because a reader would otherwise assume an
    end-to-end MEEP comparison covers it.
    """
    reference = cavity_oracle(sigma)
    assert reference.shape[0] == 1, f"CPU MEEP found {reference.shape[0]} modes, expected one"
    meep_freq, meep_decay, meep_Q, amp_real, amp_imag, _ = reference[0]
    meep_amp = complex(amp_real, amp_imag)

    modes, samples, interval = _cavity_modes(sigma)
    assert samples == 3001, f"expected one sample per step plus the opening one, got {samples}"
    assert interval == pytest.approx(1.0 / 28.0, rel=1e-9)
    assert len(modes) == 1, (
        f"expected one cavity mode, got {[(m.freq, m.Q, abs(m.amp)) for m in modes]}"
    )
    mode = modes[0]

    assert mode.freq == pytest.approx(meep_freq, rel=freq_floor)
    assert mode.Q == pytest.approx(meep_Q, rel=quality_floor)
    assert mode.decay == pytest.approx(meep_decay, rel=quality_floor)
    assert abs(mode.amp - meep_amp) / abs(meep_amp) < 3e-4, (
        f"the point-sampled record differs from MEEP's: amplitude {mode.amp} against "
        f"{meep_amp}. Frequency and Q are properties of the cavity and survive a "
        f"mis-sampled probe; the amplitude does not."
    )
    # Control: the amplitude comparison has to be able to FAIL. A probe a quarter of a
    # cell along x lands 9.1e-01 away, four orders past the bar above.
    displaced, _, _ = _cavity_modes(sigma, probe_offset=(0.25 / 14.0, 0.0, 0.0))
    assert abs(displaced[0].amp - meep_amp) / abs(meep_amp) > 0.1, (
        "a quarter-cell probe displacement must not reproduce MEEP's amplitude, or the "
        "comparison says nothing about where the point was sampled"
    )

    # The closed form, evaluated at the frequency the discretization actually produced.
    analytic_Q = 2 * math.pi * mode.freq / sigma
    assert mode.Q == pytest.approx(analytic_Q, rel=quality_floor), (
        f"Q = {mode.Q:.6f} against the analytic 2*pi*f/sigma_D = {analytic_Q:.6f}"
    )
    assert mode.Q == pytest.approx(expected_Q, rel=0.01)
    assert mode.decay < 0.0, "a passive cavity must DECAY; a positive decay is gain"
    assert mode.err < 1e-9, f"the fit is not clean (err {mode.err:.2e})"


@requires_meep
@skip_without_meep
def test_a_lossless_cavity_reports_no_decay_in_either_code(cavity_oracle):
    """The control the two lossy cases need: remove sigma_D and the decay must vanish.

    Without it, "Q tracks 2*pi*f/sigma_D" could be satisfied by an engine that reported
    some fixed decay unrelated to the material. Measured with sigma_D = 0: CPU MEEP
    reports decay 3.7e-09 and this engine 6.1e-10, both round-off against the 2.5e-03
    and 6.2e-04 of the lossy runs, and the two agree on the frequency to 5.0e-09.
    """
    reference = cavity_oracle(0.0)
    assert reference.shape[0] == 1
    meep_freq, meep_decay = reference[0][0], reference[0][1]

    modes, _, _ = _cavity_modes(0.0)
    assert len(modes) == 1
    mode = modes[0]
    assert mode.freq == pytest.approx(meep_freq, rel=1e-6)
    assert abs(mode.decay) < 1e-7, (
        f"a lossless cavity decays at {mode.decay:.3e}; the lossy runs at this frequency "
        f"decay at 6.2e-04 and 2.5e-03, so anything near those would be the engine's own"
    )
    assert abs(meep_decay) < 1e-7
    # And it is the same resonance the lossy runs found, not a different line.
    assert mode.freq == pytest.approx(0.49607, abs=1e-4)


# ---------------------------------------------------------------------------
# Mirror planes MEEP has natively: odd phase, and the Z axis.
# ---------------------------------------------------------------------------

_SYMMETRY_ORACLE_SCRIPT = '''"""CPU-MEEP reference for a declared mirror plane: mp.Mirror(direction, phase)."""
import sys

import numpy as np
import meep as mp

case = sys.argv[1]
output_path = sys.argv[2]

# "axis:phase:component" — the plane MEEP folds with, and the current that drives it.
axis_name, phase_text, component_name = case.split(":")
direction = {"X": mp.X, "Y": mp.Y, "Z": mp.Z}[axis_name]
phase = int(phase_text)
component = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez}[component_name]

# Long on the folded axis so the wave does not reach its far face in the run below:
# that face is the one place this engine's storage and MEEP's differ (a zero ghost
# here, the parity-weighted mirror image there), and it is guarded separately.
lengths = [2.0, 2.0, 2.0]
lengths["XYZ".index(axis_name)] = 4.0

simulation = mp.Simulation(
    cell_size=mp.Vector3(*lengths),
    resolution=10,
    sources=[mp.Source(mp.ContinuousSource(frequency=1.0), component=component,
                       center=mp.Vector3(0, 0, 0))],
    symmetries=[mp.Mirror(direction, phase=phase)],
    force_complex_fields=True,
    k_point=mp.Vector3(0, 0, 0),
)
simulation.run(until=1.0)
# get_array() unfolds the symmetry, so these are full-domain volumes: the phase
# convention is in the reference, not just the magnitude.
np.save(output_path, np.stack([
    np.asarray(simulation.get_array(component=mp.Ex)),
    np.asarray(simulation.get_array(component=mp.Ey)),
    np.asarray(simulation.get_array(component=mp.Ez)),
]))
'''


def _symmetry_oracle(tmp_path, case):  # One CPU-MEEP run under mp.Mirror, unfolded by MEEP itself.
    script_path = tmp_path / "meep_symmetry_oracle.py"
    script_path.write_text(_SYMMETRY_ORACLE_SCRIPT, encoding="utf-8")
    slug = "".join(character if character.isalnum() else "_" for character in case)
    output_path = tmp_path / f"{slug}_sym.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), case, str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP symmetry oracle '{case}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


# (axis, phase, driving component). The component is one the plane leaves EVEN, which
# is what the engine's source-parity rule requires and what makes the case physical:
# an Ex dipole is odd under mp.Mirror(mp.X) and even under mp.Mirror(mp.X, phase=-1),
# which is the whole reason MEEP exposes the argument.
_MIRROR_ORACLE_CASES = [
    ("X", -1, "Ex"),  # Odd X: the dipole along the plane's own normal.
    ("Y", -1, "Ey"),
    ("Z", +1, "Ex"),  # Even Z: reachable at all only now.
    ("Z", -1, "Ez"),  # Odd Z: the dipole along the Z plane's normal.
]


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("axis,phase,component", _MIRROR_ORACLE_CASES)
def test_declared_mirror_planes_match_cpu_meep(tmp_path, axis, phase, component):
    """A fold this engine performs must be the fold MEEP performs, phase included.

    MEEP takes ``mp.Mirror(direction, phase)`` natively on all three axes, so this is
    a real oracle rather than an internal consistency check, and ``get_array()``
    unfolds MEEP's own quadrant — the reference is a FULL-DOMAIN volume carrying
    MEEP's phase convention, compared against ``Fields.to_meep_array`` which unfolds
    this engine's. A parity taken from the even X/Y table instead of the plane's
    declared phase reproduces every magnitude and inverts the sign of half the
    domain, which is why the comparison is the complex relative L2 of the unfolded
    volume and not a magnitude or a quadrant.

    Measured (complex relative L2 against CPU MEEP, worst of the three E components,
    with the equivalent UNFOLDED run of the same case as the control):

        Mirror('X', -1) driven by Ex    folded 2.78e-07    control 2.78e-07
        Mirror('Y', -1) driven by Ey    folded 2.78e-07    control 2.78e-07
        Mirror('Z', +1) driven by Ex    folded 3.23e-07    control 3.23e-07
        Mirror('Z', -1) driven by Ez    folded 3.15e-07    control 3.15e-07

    The folded and unfolded figures agree to every digit printed, so the fold costs
    nothing at all here, and both sit on the engine's established no-PML core floor
    of 2.41e-07. The control is asserted alongside precisely so that a fold which
    happened to be as wrong as the physics could not pass.
    """
    from .grid import Mirror

    reference = _symmetry_oracle(tmp_path, f"{axis}:{phase}:{component}")
    index = "XYZ".index(axis)
    cell = [2.0, 2.0, 2.0]
    cell[index] = 4.0
    source = {"component": component, "frequency": 1.0, "center": (0.0, 0.0, 0.0),
              "size": (0.0, 0.0, 0.0)}

    folded = FdtdDriver(cell_size=tuple(cell), resolution=10, force_complex_fields=True,
                        symmetry=(Mirror(axis, phase),))
    folded.add_source(source)
    folded.run(until=1.0)
    full = FdtdDriver(cell_size=tuple(cell), resolution=10, force_complex_fields=True)
    full.add_source(source)
    full.run(until=1.0)

    # The fold must be dark at its far face, or the two codes are not the same system
    # there (guarded by _require_folded_far_face_is_quiet, exercised elsewhere).
    face_axis, face_ratio = folded._folded_far_face_ratio()
    assert face_ratio < 1e-3, f"the far {'xyz'[face_axis]} face carries {face_ratio:.3e} of peak"

    for offset, name in enumerate(("Ex", "Ey", "Ez")):
        meep_volume = reference[offset]
        scale = float(np.abs(reference).max())
        if float(np.abs(meep_volume).max()) < 1e-3 * scale:
            continue  # A component this configuration does not drive; nothing to compare.
        # MEEP prepends the periodic duplicate on every UNFOLDED axis and returns a
        # folded axis at its full count — exactly Fields.to_meep_array's convention.
        folded_error = _complex_relative_l2(folded.fields.to_meep_array(name), meep_volume)
        # The unfolded control has no folded axis, so it still carries the duplicate
        # there; drop it to line the two up cell for cell.
        control = full.fields.to_meep_array(name)[
            (slice(None),) * index + (slice(1, None),)
        ]
        full_error = _complex_relative_l2(control, meep_volume)
        assert full_error < 1e-6, (
            f"the unfolded control for {name} under Mirror({axis!r}, {phase:+d}) is not at its "
            f"floor: {full_error:.3e}; the fold comparison below would be meaningless"
        )
        assert folded_error < 1e-6, (
            f"{name} under Mirror({axis!r}, {phase:+d}) differs from CPU MEEP's own folded run "
            f"by {folded_error:.3e} (unfolded control {full_error:.3e})"
        )


@requires_meep
@skip_without_meep
def test_the_declared_phase_is_not_cosmetic_against_cpu_meep(tmp_path):
    """Folding at the wrong phase must be a LOUD disagreement with MEEP, not a nuance.

    The bound in the case above only means something if the opposite phase fails it.
    Both folds are legal grids and both run to completion; what separates them is the
    sign the reconstruction carries into the far half, and this measures that
    separation against MEEP rather than against the engine's own other answer.
    """
    from .grid import Mirror

    reference = _symmetry_oracle(tmp_path, "Z:-1:Ez")
    source = {"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.0),
              "size": (0.0, 0.0, 0.0)}
    folded = FdtdDriver(cell_size=(2.0, 2.0, 4.0), resolution=10, force_complex_fields=True,
                        symmetry=(Mirror("Z", -1),))
    folded.add_source(source)
    folded.run(until=1.0)
    right = _complex_relative_l2(folded.fields.to_meep_array("Ez"), reference[2])
    assert right < 1e-6, f"the correct phase is not at its floor: {right:.3e}"

    # The even Z plane refuses this source outright — an Ez current on z = 0 is odd
    # about it — so the wrong phase cannot even be RUN, which is the strongest form
    # of "not cosmetic". The sign it would have carried is measured directly instead.
    even = FdtdDriver(cell_size=(2.0, 2.0, 4.0), resolution=10, force_complex_fields=True,
                      symmetry=(Mirror("Z", +1),))
    with pytest.raises(ValueError, match="cannot carry the parity the fold demands"):
        even.add_source(source)
    mirrored_at_the_wrong_phase = folded.fields.to_meep_array("Ez").copy()
    centre = mirrored_at_the_wrong_phase.shape[2] // 2
    mirrored_at_the_wrong_phase[:, :, :centre] *= -1  # What the even table would have written.
    wrong = _complex_relative_l2(mirrored_at_the_wrong_phase, reference[2])
    assert wrong > 0.5, (
        f"inverting the reconstructed half changes the answer by only {wrong:.3e}; this case "
        f"cannot tell the two phases apart and is not a test of the phase"
    )


# --- Metallic (PEC) outer boundaries, against CPU MEEP ------------------------------------

_METALLIC_ORACLE_SCRIPT = '''"""CPU-MEEP references for metallic-wall runs, as a .npz of named component arrays."""
import json
import sys

import numpy as np
import meep as mp

spec = json.loads(sys.argv[1])
output_path = sys.argv[2]

source_kind = spec.get("source", "continuous")
envelope = (mp.ContinuousSource(frequency=1.0) if source_kind == "continuous"
            else mp.GaussianSource(1.0, fwidth=0.5))
medium = {"epsilon": spec.get("epsilon", 1.0)}
if spec.get("lorentz"):
    frequency, gamma, sigma = spec["lorentz"]
    medium["E_susceptibilities"] = [mp.LorentzianSusceptibility(
        frequency=frequency, gamma=gamma, sigma=sigma)]
if spec.get("conductivity"):
    medium["D_conductivity"] = spec["conductivity"]
if spec.get("chi3"):
    medium["chi3"] = spec["chi3"]

arguments = dict(
    cell_size=mp.Vector3(*spec["cell"]),
    resolution=spec["resolution"],
    sources=[mp.Source(envelope, component=mp.Ez,
                       center=mp.Vector3(*spec["src"]),
                       size=mp.Vector3(*spec.get("src_size", (0, 0, 0))))],
    default_material=mp.Medium(**medium),
    force_complex_fields=True,
)
# k_point ABSENT is the whole point of these cases: MEEP's fields::fields leaves every
# face Metallic and only use_bloch (which Simulation calls under `if self.k_point:`)
# turns an axis periodic.
if spec.get("k_point") is not None:
    arguments["k_point"] = mp.Vector3(*spec["k_point"])
if spec.get("pml"):
    arguments["boundary_layers"] = [
        mp.PML(thickness, direction=getattr(mp, axis.upper()))
        for axis, thickness in spec["pml"].items()
    ]
if spec.get("symmetry"):
    arguments["symmetries"] = [mp.Mirror(getattr(mp, axis), phase=phase)
                               for axis, phase in spec["symmetry"]]

simulation = mp.Simulation(**arguments)
simulation.init_sim()
# A per-DIRECTION use_bloch is the only route to a mixed cell MEEP has, and it is
# reachable only after init_sim — which is exactly why from_meep refuses an
# initialized simulation rather than guessing what its six faces ended up as.
for axis, k in (spec.get("bloch_axes") or {}).items():
    simulation.fields.use_bloch(getattr(mp, axis.upper()), float(k))
simulation.run(until=spec["until"])
np.savez(output_path, **{
    name: np.asarray(simulation.get_array(component=getattr(mp, name)))
    for name in ("Ez", "Ex", "Hy", "Hx")
})
'''


def _metallic_oracle(tmp_path, name, spec):  # One CPU-MEEP metallic reference, in its own process.
    script_path = tmp_path / "meep_metallic_oracle.py"
    script_path.write_text(_METALLIC_ORACLE_SCRIPT, encoding="utf-8")
    output_path = tmp_path / f"{name}.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(spec), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP metallic oracle '{name}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {key: archive[key] for key in archive.files}


def _metallic_driver(spec):  # The same run on this engine, boundaries taken from the spec.
    boundaries = spec.get("boundaries")
    if boundaries is None:
        boundaries = "periodic" if spec.get("k_point") is not None else "metallic"
    arguments = dict(cell_size=tuple(spec["cell"]), resolution=spec["resolution"],
                     force_complex_fields=True, boundaries=boundaries)
    if spec.get("k_point") is not None:
        arguments["k_point"] = tuple(spec["k_point"])
    if spec.get("bloch_axes"):
        arguments["k_point"] = tuple(
            float(spec["bloch_axes"].get(axis, 0.0)) for axis in ("x", "y", "z"))
    if spec.get("symmetry"):
        arguments["symmetry"] = tuple(
            Mirror(axis, phase) for axis, phase in spec["symmetry"])
    driver = FdtdDriver(**arguments)
    if spec.get("epsilon"):
        driver.set_epsilon(np.full(driver.shape, spec["epsilon"], dtype=np.float32))
    if spec.get("lorentz"):
        frequency, gamma, sigma = spec["lorentz"]
        driver.add_susceptibility(
            Susceptibility(frequency=frequency, gamma=gamma, kind=LORENTZIAN), sigma)
    if spec.get("conductivity"):
        driver.set_conductivity(spec["conductivity"])
    if spec.get("chi3"):
        driver.set_chi3(spec["chi3"])
    if spec.get("pml"):
        driver.setup_pml({axis: int(round(thickness * spec["resolution"]))
                          for axis, thickness in spec["pml"].items()})
    source = {"component": "Ez", "frequency": 1.0, "center": tuple(spec["src"]),
              "size": tuple(spec.get("src_size", (0.0, 0.0, 0.0)))}
    if spec.get("source") == "gaussian":
        source.update(source_type="gaussian", fwidth=0.5)
    driver.add_source(source)
    driver.run(until=spec["until"])
    return driver


def _meep_layout(driver, component):
    """This run's field in ``sim.get_array``'s own layout, ready to compare with no slicing.

    ``Fields.to_meep_array`` prepends MEEP's periodic duplicate plane on every axis it
    does not fold; a metallic axis has no lattice image to duplicate and MEEP returns
    ``N`` there rather than ``N + 1``, so that plane is dropped again — the same
    correction :meth:`GpuRunResult.get_array` applies, reproduced here so the driver
    cases and the converter cases are compared on one convention.
    """
    values = np.array(np.asarray(driver.fields.to_meep_array(component)), copy=True)
    for axis in range(3):
        if driver.grid.is_metallic(axis) and not driver.grid.is_mirrored(axis):
            first = (slice(None),) * axis + (0,)
            last = (slice(None),) * axis + (-1,)
            assert np.array_equal(values[first], values[last]), (
                "to_meep_array no longer prepends a plain duplicate; the slice below would "
                "then drop a real plane of field.")
            values = values[(slice(None),) * axis + (slice(1, None),)]
        elif driver.grid.bloch_phase(axis) is not None and not driver.grid.is_mirrored(axis):
            first = (slice(None),) * axis + (0,)
            values[first] = values[first] * np.conjugate(driver.grid.bloch_phase(axis))
    return values


# Every case below leaves ``k_point`` unset, which is MEEP's DEFAULT and was this
# converter's single most common refusal (37 of the 57 surveyed examples). The
# recorded floors are the same band as the periodic ones: no-PML core 2.41e-07,
# uniform PML 4.00e-07, Lorentz 3.2e-06, conductivity 2.7e-07.
_METALLIC_CASES = {
    # Even cell counts, off-centre source, and the same run at odd counts: the
    # half-cell origin shift of an odd axis is invisible in a field that still looks
    # like a field (Grid.origin_doubled), so both parities are swept.
    "even_counts": dict(cell=[1.0, 1.0, 2.0], resolution=10, src=[0.03, -0.05, -0.35],
                        until=2.0, bar=1e-6),
    "odd_counts": dict(cell=[1.1, 0.9, 1.5], resolution=10, src=[0.03, -0.05, -0.25],
                       until=2.0, bar=1e-6),
    # Long enough that the wave has hit every wall many times: a boundary condition
    # that is only nearly right diverges here and not at t = 2.
    "odd_counts_long": dict(cell=[1.1, 0.9, 1.5], resolution=10, src=[0.03, -0.05, -0.25],
                            until=6.0, bar=1e-6),
    "pml_on_z_only": dict(cell=[1.0, 1.0, 3.0], resolution=10, src=[0.05, 0.05, -0.45],
                          until=3.0, pml={"z": 0.6}, bar=1e-6),
    "uniform_pml": dict(cell=[1.4, 1.4, 2.4], resolution=10, src=[0.05, -0.05, -0.35],
                        until=3.0, pml={"x": 0.4, "y": 0.4, "z": 0.4}, bar=2e-6),
    "structured_epsilon_4": dict(cell=[1.0, 1.1, 2.0], resolution=10, src=[0.03, -0.05, -0.35],
                                 until=2.0, epsilon=4.0, bar=1e-6),
    "sheet_source": dict(cell=[1.2, 1.0, 2.0], resolution=10, src=[0.0, 0.0, -0.35],
                         src_size=[0.6, 0.6, 0.0], until=2.5, bar=1e-6),
    "lorentz": dict(cell=[1.0, 1.0, 2.0], resolution=10, src=[0.05, 0.05, -0.35],
                    until=3.0, epsilon=2.0, lorentz=[1.1, 0.05, 0.4], source="gaussian",
                    bar=1e-5),
    "d_conductivity": dict(cell=[1.0, 1.0, 2.0], resolution=10, src=[0.05, 0.05, -0.35],
                           until=3.0, epsilon=2.25, conductivity=0.3, source="gaussian",
                           bar=1e-5),
    "chi3": dict(cell=[1.0, 1.0, 2.0], resolution=10, src=[0.0, 0.0, -0.35],
                 src_size=[0.5, 0.5, 0.0], until=3.0, epsilon=2.0, chi3=1.0, bar=1e-5),
    # Everything the feature list says must compose, composed: a walled cell that is
    # also folded on x, absorbing on z, and dispersive — at odd counts on y, so the
    # half-cell origin shift is in play too.
    "pml_symmetry_and_dispersion": dict(
        cell=[3.0, 1.1, 2.4], resolution=10, src=[0.0, -0.05, -0.35], until=4.0,
        symmetry=[["X", 1]], pml={"z": 0.4}, epsilon=2.0, lorentz=[1.1, 0.05, 0.4],
        source="gaussian", bar=1e-5),
}


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("name", sorted(_METALLIC_CASES))
def test_a_metallic_run_reproduces_cpu_meep_at_the_engines_own_floors(tmp_path, name):
    """PEC walls — MEEP's default, and this converter's largest refusal until now.

    Measured, complex relative L2 over the whole volume against
    ``sim.get_array``, worst of Ez / Ex / Hy / Hx:

        even_counts          1.8e-07 .. 3.1e-07
        odd_counts           1.9e-07 .. 4.0e-07
        odd_counts_long      3.4e-07 .. 5.6e-07   (t = 6, many wall reflections)
        pml_on_z_only        4.2e-07 .. 6.7e-07
        uniform_pml          5.4e-07 .. 9.9e-07
        structured_epsilon_4 1.2e-07 .. 1.9e-07
        sheet_source         1.1e-07 .. 2.2e-07
        lorentz              2.5e-07 .. 4.3e-07
        d_conductivity       2.4e-07 .. 2.7e-07
        chi3                 2.2e-07 .. 2.4e-07
        pml_symmetry_and_dispersion
                             4.3e-07 .. 6.6e-07   (walled + folded + absorbing +
                                                   dispersive, odd y count)

    The SHAPE is asserted before the values and is a check in its own right: MEEP
    returns ``N`` points per axis for a metallic axis and ``N + 1`` for a periodic
    one, so a run that stepped the wrong boundary condition cannot even line up.
    """
    spec = _METALLIC_CASES[name]
    reference = _metallic_oracle(tmp_path, name, spec)
    driver = _metallic_driver(spec)
    try:
        assert driver.boundaries == (("metallic", "metallic"),) * 3
        worst = 0.0
        for component in ("Ez", "Ex", "Hy", "Hx"):
            candidate = _meep_layout(driver, component)
            expected = reference[component]
            assert candidate.shape == expected.shape, (
                f"{name}/{component}: this engine returns {candidate.shape} where CPU MEEP "
                f"returns {expected.shape}. MEEP gives N points per METALLIC axis and N+1 per "
                f"periodic one, so a shape mismatch is a boundary-condition mismatch."
            )
            error = _complex_relative_l2(candidate, expected)
            print(f"metallic {name}/{component}: complex relative L2 {error:.3e}", flush=True)
            worst = max(worst, error)
        assert worst < spec["bar"], (
            f"Metallic case '{name}' is {worst:.3e} against CPU MEEP, above its {spec['bar']:g} bar."
        )
    finally:
        driver.close()


_MIXED_BOUNDARY_CASES = {
    # MEEP's declared arguments cannot express a mixed cell — k_point is all-or-nothing
    # — so the oracle reaches for fields.use_bloch(direction, k) after init_sim, which
    # is per DIRECTION. That is the same route from_meep refuses to guess at.
    "pec_xy_periodic_z": dict(bloch_axes={"z": 0.0}),
    "pec_xz_periodic_y": dict(bloch_axes={"y": 0.0}),
    "pec_x_periodic_yz": dict(bloch_axes={"y": 0.0, "z": 0.0}),
    "pec_xy_bloch_z": dict(bloch_axes={"z": 0.3}),
    "pec_y_bloch_x_periodic_z": dict(bloch_axes={"x": 0.2, "z": 0.0}),
}


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("name", sorted(_MIXED_BOUNDARY_CASES))
def test_metallic_periodic_and_bloch_axes_coexist_in_one_run(tmp_path, name):
    """Different boundary conditions on different axes of the SAME cell, against CPU MEEP.

    This is what makes the feature per-axis rather than a global mode, and it is the
    configuration a blanket "restore the old metallic rule" would get wrong: the
    metallic axes must wall while the periodic ones keep wrapping and the Bloch ones
    keep their phase. Measured, worst of Ez / Ex / Hy / Hx:

        pec_xy_periodic_z         1.2e-07 .. 3.0e-07   shape (10, 11, 21)
        pec_xz_periodic_y         1.4e-07 .. 3.5e-07   shape (10, 12, 20)
        pec_x_periodic_yz         1.2e-07 .. 3.6e-07   shape (10, 12, 21)
        pec_xy_bloch_z  k = 0.3   1.2e-07 .. 3.1e-07   shape (10, 11, 21)
        pec_y_bloch_x   k = 0.2   1.4e-07 .. 3.1e-07   shape (11, 11, 21)

    The per-axis shape is the first assertion and carries real information: N on a
    metallic axis, N + 1 on a periodic or Bloch one, in the same array.
    """
    axes = _MIXED_BOUNDARY_CASES[name]["bloch_axes"]
    spec = dict(cell=[1.0, 1.1, 2.0], resolution=10, src=[0.03, -0.05, -0.35], until=2.5,
                bloch_axes=axes,
                boundaries={axis: ("periodic" if axis in axes else "metallic")
                            for axis in ("x", "y", "z")})
    reference = _metallic_oracle(tmp_path, name, spec)
    driver = _metallic_driver(spec)
    try:
        for axis, axis_name in enumerate(("x", "y", "z")):
            assert driver.grid.is_metallic(axis) is (axis_name not in axes)
        worst = 0.0
        for component in ("Ez", "Ex", "Hy", "Hx"):
            candidate = _meep_layout(driver, component)
            expected = reference[component]
            assert candidate.shape == expected.shape, (
                f"{name}/{component}: {candidate.shape} vs CPU MEEP {expected.shape}; a "
                f"metallic axis contributes N points and a periodic one N+1."
            )
            error = _complex_relative_l2(candidate, expected)
            print(f"mixed {name}/{component}: complex relative L2 {error:.3e}", flush=True)
            worst = max(worst, error)
        assert worst < 1e-6, f"Mixed-boundary case '{name}' is {worst:.3e} against CPU MEEP."
    finally:
        driver.close()


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("symmetry,cell", [
    ([["X", 1]], [3.0, 1.0, 2.0]),
    ([["X", 1], ["Y", 1]], [2.0, 2.0, 2.0]),
])
def test_a_folded_metallic_run_is_exact_with_the_wave_hitting_the_wall(tmp_path, symmetry, cell):
    """The metallic fold's zero far ghost IS the boundary condition, and this measures it.

    A folded METALLIC axis keeps MEEP's halved ``n_full // 2 + 1`` cells ending at
    ``L/2 - dx`` and terminates the plane MEEP owns at ``L/2`` with a zero ghost.
    Under PERIODIC boundaries that plane is a real, live, parity-weighted image —
    now stored and stepped as one cell more (the second mirror; the live-face
    fold cases pin it against CPU MEEP) — but under METALLIC walls it is one MEEP
    holds at exactly zero
    (``find_metals`` / ``zero_metal``), so the zero ghost is not an approximation of
    anything — it IS the boundary condition, and no extra storage exists. The runs
    below go to t = 6 and t = 5 with a source at the origin of a 3 x 1 x 2 and a
    2 x 2 x 2 cell, i.e. with the wave crossing the far wall repeatedly, and measure
    2.1e-07 .. 4.6e-07 and 2.9e-07 .. 6.6e-07 — the engine's ordinary floor. The
    far-face guard is asserted to be inactive here, because a passing number with the
    guard silently refusing nothing would be the same result for the wrong reason.
    """
    spec = dict(cell=cell, resolution=10, src=[0.0, 0.0, -0.35], until=6.0 if len(symmetry) == 1 else 5.0,
                symmetry=symmetry)
    reference = _metallic_oracle(tmp_path, "folded_metallic", spec)
    driver = _metallic_driver(spec)
    try:
        # The guard reports "no folded axis to measure" precisely because every folded
        # axis of this run is metallic.
        assert driver._folded_far_face_ratio() == (-1, 0.0)
        # ... and the far face really is live, so this is not a dark-face run in
        # disguise: the periodic twin of the same case measures its own far face
        # well above the old guard's threshold (it is now served — the periodic
        # fold stores its second-mirror plane — and stores one cell more).
        periodic_twin = FdtdDriver(cell_size=tuple(cell), resolution=10,
                                   force_complex_fields=True,
                                   symmetry=tuple(Mirror(a, p) for a, p in symmetry))
        periodic_twin.add_source({"component": "Ez", "frequency": 1.0,
                                  "center": tuple(spec["src"]), "size": (0.0, 0.0, 0.0)})
        periodic_twin.run(until=spec["until"])
        live_ratio = periodic_twin._folded_far_face_ratio()[1]
        for axis in range(3):  # Periodic folds store the plane and ghost slot; metallic keep MEEP's window.
            expected = driver.shape[axis] + (1 if periodic_twin.grid.is_mirrored(axis) else 0)
            assert periodic_twin.shape[axis] == expected
        periodic_twin.close()
        assert live_ratio > 1e-2, (
            f"the far face carries only {live_ratio:.3e}; this case does not exercise the "
            f"live-face regime, so it proves nothing about the metallic one.")

        worst = 0.0
        for component in ("Ez", "Ex", "Hy", "Hx"):
            candidate = _meep_layout(driver, component)
            expected = reference[component]
            assert candidate.shape == expected.shape
            error = _complex_relative_l2(candidate, expected)
            print(f"folded metallic {symmetry}/{component}: complex relative L2 {error:.3e}",
                  flush=True)
            worst = max(worst, error)
        assert worst < 1e-6, (
            f"Folded metallic run is {worst:.3e} against CPU MEEP with its far face live; the "
            f"identical folded run on PERIODIC boundaries is refused at 8.8e-01."
        )
    finally:
        driver.close()


@requires_meep
@skip_without_meep
def test_the_metallic_and_periodic_answers_are_different_simulations_not_different_accuracies(
        tmp_path):
    """The positive control for every case above: the two boundaries must DISAGREE.

    If they did not, every metallic number in this file could be produced by an engine
    that quietly ran the periodic problem — which is exactly what this package did
    before, and why the converter refused a k_point-less simulation outright. Measured
    on this 1 x 1 x 2 cell: 1.22e+00 complex relative L2 between the two (the 1.28e+00
    the refusal used to quote is the same measurement on a 2 x 2 x 4 cell), on fields
    whose magnitudes are both entirely plausible.
    """
    spec = dict(cell=[1.0, 1.0, 2.0], resolution=10, src=[0.03, -0.05, -0.35], until=2.0)
    metallic = _metallic_oracle(tmp_path, "control_metallic", spec)["Ez"]
    periodic = _metallic_oracle(
        tmp_path, "control_periodic", dict(spec, k_point=[0.0, 0.0, 0.0]))["Ez"]
    # MEEP itself returns different shapes, which is the first half of the statement.
    assert metallic.shape == (10, 10, 20) and periodic.shape == (11, 11, 21)
    gap = _complex_relative_l2(metallic, periodic[1:, 1:, 1:])
    print(f"metallic vs periodic on the same cell: complex relative L2 {gap:.3e}", flush=True)
    assert gap > 1.0, (
        f"CPU MEEP's metallic and periodic answers differ by only {gap:.3e} on this cell, so it "
        f"cannot tell the two boundary conditions apart and every metallic case here is "
        f"unfalsifiable. Pick a cell the wave crosses."
    )


_FLUX_METALLIC_ORACLE = '''"""CPU-MEEP flux spectrum through a plane in a metallic (no k_point) box."""
import json
import sys

import numpy as np
import meep as mp

spec = json.loads(sys.argv[1])
simulation = mp.Simulation(
    cell_size=mp.Vector3(*spec["cell"]), resolution=spec["resolution"],
    sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), component=mp.Ez,
                       center=mp.Vector3(*spec["src"]))],
    force_complex_fields=True)  # k_point deliberately absent: MEEP's metallic default.
region = simulation.add_flux(
    1.0, 0.4, 3, mp.FluxRegion(center=mp.Vector3(*spec["plane"]),
                               size=mp.Vector3(*spec["plane_size"]), direction=mp.Z),
    decimation_factor=1)
simulation.run(until=spec["until"])
np.save(sys.argv[2], np.asarray(mp.get_fluxes(region)))
'''


def _metallic_flux_oracle(tmp_path, name, spec):  # One CPU-MEEP metallic flux spectrum.
    script_path = tmp_path / "meep_metallic_flux.py"
    script_path.write_text(_FLUX_METALLIC_ORACLE, encoding="utf-8")
    output_path = tmp_path / f"{name}_flux.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(spec), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP metallic flux oracle '{name}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


@requires_meep
@skip_without_meep
def test_a_flux_plane_is_exact_at_every_extent_up_to_and_including_the_metallic_walls(tmp_path):
    """The flux integral is exact right out to the wall — near-wall planes included.

    ``FluxMonitor`` integrates CELL-CENTRED samples, and cell centring is an average
    with the next Yee plane up. On a wrapping axis that average is a circular
    convolution, which commutes with the weighted sum; on a metallic axis it
    terminates on the PEC wall, which MEEP's ``step_boundaries`` holds at exactly
    zero — so ``_sliced_component`` appends that zero plane past the far face of
    every non-wrapping axis, and the average telescopes on both kinds of axis.

    Dropping the average instead (which is what the engine did while only
    mirror-folded axes got the zero plane) was worth 9.7e-03 … 1.4e-02 on every
    full-width transverse extent and EXACTLY 2.02x MEEP on a full-cross-section
    plane one cell inside the high wall — a silently accepted plane, since the old
    refusal exempted the normal axis. Re-measured with the zero plane landed
    (same 2 x 2 x 4 cell, res 10, Gaussian Ez at (0.05, -0.05, -0.5), worst
    per-frequency relative difference against ``mp.get_fluxes``):

        extent 1.60 / 1.70              1.1e-07 / 1.0e-07  (unmoved controls)
        extent 1.72 / 1.80 / 2.00       4.3e-08 / 7.7e-08 / 4.7e-08  (were refused)
        normal z = 1.90 / 1.925 / 1.95  5.5e-08 / 1.3e-07 / 1.1e-07  (was the 2.02x)
        periodic twin, full width       1.1e-07  (unmoved)

    (the design notes (absorber-placement-evidence) §2.3, §3.3.)
    """
    spec = dict(cell=[2.0, 2.0, 4.0], resolution=10, src=[0.05, -0.05, -0.5],
                plane=[0.0, 0.0, 0.8], until=12.0)

    def engine_flux(extent, boundaries, plane=None):
        driver = FdtdDriver(cell_size=tuple(spec["cell"]), resolution=spec["resolution"],
                            force_complex_fields=True, **boundaries)
        driver.add_source({"component": "Ez", "frequency": 1.0, "fwidth": 0.5,
                           "source_type": "gaussian", "center": tuple(spec["src"]),
                           "size": (0.0, 0.0, 0.0)})
        monitor = driver.add_flux_monitor(center=tuple(plane or spec["plane"]),
                                          size=(extent, extent, 0.0), direction=2,
                                          fcen=1.0, df=0.4, nfreq=3, decimation_factor=1)
        driver.run(until=spec["until"])
        values = np.asarray(monitor.get_flux_spectrum())
        driver.close()
        return values

    # 1. The unmoved interior control, then the once-refused wider extents, all at
    #    the floor against their own CPU-MEEP oracles.
    for extent, label in ((1.7, "inside"), (1.72, "touching"), (2.0, "full")):
        reference = _metallic_flux_oracle(
            tmp_path, label, dict(spec, plane_size=[extent, extent, 0.0]))
        measured = engine_flux(extent, {"boundaries": "metallic"})
        worst = float(np.max(np.abs(measured - reference) / np.abs(reference)))
        print(f"metallic flux, extent {extent}: worst relative {worst:.3e}", flush=True)
        assert worst < 1e-6, f"extent {extent} metallic flux is {worst:.3e} from CPU MEEP"

    # 2. The normal axis: a full-width plane one cell inside the high wall was the
    #    silently-accepted 2.02x case; it must sit at the floor now.
    reference = _metallic_flux_oracle(
        tmp_path, "high_wall",
        dict(spec, plane=[0.0, 0.0, 1.95], plane_size=[2.0, 2.0, 0.0]))
    measured = engine_flux(2.0, {"boundaries": "metallic"}, plane=[0.0, 0.0, 1.95])
    worst = float(np.max(np.abs(measured - reference) / np.abs(reference)))
    print(f"metallic flux, plane at z=1.95: worst relative {worst:.3e}", flush=True)
    assert worst < 1e-6, f"the near-wall normal-axis plane is {worst:.3e} from CPU MEEP"

    # 3. The SAME full-width plane on a periodic run stays exact — the zero plane is
    #    appended only where an axis does not wrap.
    periodic_reference = _flux_oracle_periodic_full(tmp_path, spec)
    periodic = engine_flux(2.0, {"k_point": (0.0, 0.0, 0.0)})
    periodic_worst = float(
        np.max(np.abs(periodic - periodic_reference) / np.abs(periodic_reference)))
    print(f"periodic flux, full cross section: worst relative {periodic_worst:.3e}", flush=True)
    assert periodic_worst < 1e-6, (
        f"the periodic control is {periodic_worst:.3e}; the wall fix must not have touched "
        f"the wrapping path."
    )
    # 4. And a plane on the PERIODIC axes of a mixed run is unaffected.
    mixed = engine_flux(2.0, {"boundaries": {"z": "metallic"}})
    assert np.all(np.isfinite(mixed))


_PERIODIC_FULL_FLUX_ORACLE = '''"""CPU-MEEP flux through the full cross section of a PERIODIC cell."""
import json
import sys

import numpy as np
import meep as mp

spec = json.loads(sys.argv[1])
simulation = mp.Simulation(
    cell_size=mp.Vector3(*spec["cell"]), resolution=spec["resolution"],
    sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), component=mp.Ez,
                       center=mp.Vector3(*spec["src"]))],
    force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
region = simulation.add_flux(
    1.0, 0.4, 3, mp.FluxRegion(center=mp.Vector3(*spec["plane"]),
                               size=mp.Vector3(spec["cell"][0], spec["cell"][1], 0),
                               direction=mp.Z), decimation_factor=1)
simulation.run(until=spec["until"])
np.save(sys.argv[2], np.asarray(mp.get_fluxes(region)))
'''


def _flux_oracle_periodic_full(tmp_path, spec):  # The periodic control for the case above.
    script_path = tmp_path / "meep_periodic_full_flux.py"
    script_path.write_text(_PERIODIC_FULL_FLUX_ORACLE, encoding="utf-8")
    output_path = tmp_path / "periodic_full_flux.npy"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(spec), str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP periodic full-cross-section flux oracle failed "
        f"(exit {completed.returncode}).\nstderr tail:\n{completed.stderr[-2000:]}"
    )
    return np.load(output_path)


_DFT_METALLIC_ORACLE = '''"""CPU-MEEP DFT field over a region of a metallic (no k_point) box."""
import json
import sys

import numpy as np
import meep as mp

spec = json.loads(sys.argv[1])
simulation = mp.Simulation(
    cell_size=mp.Vector3(*spec["cell"]), resolution=spec["resolution"],
    sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), component=mp.Ez,
                       center=mp.Vector3(*spec["src"]))],
    force_complex_fields=True)
cell = simulation.add_dft_fields(
    [mp.Ez], [1.0],
    where=mp.Volume(center=mp.Vector3(*spec["centre"]), size=mp.Vector3(*spec["size"])),
    decimation_factor=1)
simulation.run(until=spec["until"])
np.save(sys.argv[2], np.asarray(simulation.get_dft_array(cell, mp.Ez, 0)))
'''


@requires_meep
@skip_without_meep
def test_a_dft_region_is_exact_up_to_and_including_the_metallic_walls(tmp_path):
    """A DFT region's Yee-to-centre average terminates on the wall zero, everywhere.

    ``dft._sliced_component`` builds the cell-centring average from one plane past
    the region's far face: it fetches that plane from the periodic wrap where the
    axis wraps and appends an explicit ZERO past the far face of every non-wrapping
    axis — a mirror-folded axis terminates on the fold's far wall, a metallic axis
    on the PEC wall itself, both of which MEEP holds at exactly zero. While only
    the folded axes got the zero plane, a whole-grid region in a PEC box lost the
    average for the whole axis and came back as raw Yee samples: measured
    5.20e-01 complex relative L2 against ``sim.get_dft_array`` on the 1 x 1.1 x 2
    cell below, where the same region one cell narrower read 5.61e-08 and the
    periodic twin 7.92e-08. The refusal that stood in front of that wrong answer
    (``_require_dft_regions_clear_of_metal``) is deleted with the fix, and this
    test now pins the repaired answer at the floor instead.
    """
    spec = dict(cell=[1.0, 1.1, 2.0], resolution=10, src=[0.05, -0.05, -0.35], until=8.0)

    def engine_dft(size, centre, boundaries, component="Ez"):
        driver = FdtdDriver(cell_size=tuple(spec["cell"]), resolution=spec["resolution"],
                            force_complex_fields=True, **boundaries)
        driver.add_source({"component": "Ez", "frequency": 1.0, "fwidth": 0.5,
                           "source_type": "gaussian", "center": tuple(spec["src"]),
                           "size": (0.0, 0.0, 0.0)})
        monitor = driver.add_dft_monitor(frequencies=1.0, center=centre, size=size,
                                         components=(component,), decimation_factor=1)
        driver.run(until=spec["until"])
        values = np.squeeze(np.asarray(monitor.get_dft(component)))
        driver.close()
        return values

    def oracle(name, size, centre):
        script_path = tmp_path / "meep_metallic_dft.py"
        script_path.write_text(_DFT_METALLIC_ORACLE, encoding="utf-8")
        output_path = tmp_path / f"{name}_dft.npy"
        environment = dict(os.environ)
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        completed = subprocess.run(
            [sys.executable, str(script_path),
             json.dumps(dict(spec, size=list(size), centre=list(centre))), str(output_path)],
            capture_output=True, text=True, env=environment, timeout=900,
        )
        assert completed.returncode == 0 and output_path.exists(), (
            f"CPU-MEEP metallic DFT oracle '{name}' failed (exit {completed.returncode}).\n"
            f"stderr tail:\n{completed.stderr[-2000:]}"
        )
        return np.load(output_path)

    # 1. A region stopping short of every wall has the plane its average needs, and is
    #    at the engine's floor (the unmoved control).
    inside_size, inside_centre = (0.6, 0.6, 1.0), (0.0, 0.0, 0.0)
    reference = oracle("inside", inside_size, inside_centre)
    measured = engine_dft(inside_size, inside_centre, {"boundaries": "metallic"})
    assert measured.shape == reference.shape
    error = _complex_relative_l2(measured, reference)
    print(f"metallic DFT region inside the walls: complex relative L2 {error:.3e}", flush=True)
    assert error < 1e-6, f"the interior metallic DFT region is {error:.3e} from CPU MEEP"

    # 2. The whole-grid DEFAULT-shaped region reaches every wall and now matches CPU
    #    MEEP at the floor — the configuration the deleted refusal used to block.
    whole_size, whole_centre = (1.0, 1.1, 2.0), (0.0, 0.0, 0.0)
    reference = oracle("whole", whole_size, whole_centre)
    measured = engine_dft(whole_size, whole_centre, {"boundaries": "metallic"})
    assert measured.shape == reference.shape
    error = _complex_relative_l2(measured, reference)
    print(f"metallic DFT region to the walls: complex relative L2 {error:.3e}", flush=True)
    assert error < 1e-6, f"the whole-grid metallic DFT region is {error:.3e} from CPU MEEP"

    # 3. The identical whole-grid region on a PERIODIC run stays exact and finite: the
    #    zero plane is appended only where an axis does not wrap.
    driver = FdtdDriver(cell_size=tuple(spec["cell"]), resolution=spec["resolution"],
                        force_complex_fields=True, k_point=(0.0, 0.0, 0.0))
    driver.add_source({"component": "Ez", "frequency": 1.0, "fwidth": 0.5,
                       "source_type": "gaussian", "center": tuple(spec["src"]),
                       "size": (0.0, 0.0, 0.0)})
    monitor = driver.add_dft_monitor(frequencies=1.0, center=(0.0, 0.0, 0.0),
                                     size=(1.0, 1.1, 2.0), components=("Ez",),
                                     decimation_factor=1)
    driver.run(until=spec["until"])
    assert np.all(np.isfinite(np.asarray(monitor.get_dft("Ez"))))
    driver.close()

    # 4. A component that needs no average on the walled axes never depended on the
    #    appended plane: Hz has Yee shift 1 on x and y, so both spellings must agree.
    from .fields import IYEE_SHIFTS

    assert IYEE_SHIFTS["Hz"][0] == 1 and IYEE_SHIFTS["Hz"][1] == 1
    values = engine_dft((1.0, 1.1, 1.0), (0.0, 0.0, 0.0),
                        {"boundaries": {"x": "metallic", "y": "metallic"}}, component="Hz")
    assert np.all(np.isfinite(values))


def test_the_monitor_wrap_flags_and_the_stepping_boundary_kinds_agree_axis_for_axis():
    """One definition of "this axis has a lattice vector", read by both halves of the run.

    A monitor measures the field the stepping produced, so ``_periodic_axes`` (the
    monitor side) and ``stepping._boundary_kinds`` (the kernel side) have to answer the
    same question the same way. They have drifted before: the Z slot of
    ``_periodic_axes`` was once a hard-coded ``False``, which handed a folded Z run the
    periodic wrap of an axis with no lattice vector.

    Pinned as a declaration AND now also observable through results: a monitor whose
    sampling reaches a metallic face is no longer refused — ``_sliced_component``
    terminates the Yee average on the wall zero for every non-wrapping axis — so a
    wrong flag on a walled axis flips that axis between "wrap" and "wall zero" and
    moves an accepted flux/DFT answer (the near-wall cases in
    ``test_a_flux_plane_is_exact_at_every_extent_up_to_and_including_the_metallic_walls``
    and ``test_a_dft_region_is_exact_up_to_and_including_the_metallic_walls`` measure
    exactly that plane against CPU MEEP).
    """
    from .driver import _periodic_axes
    from .stepping import METALLIC, MIRROR

    for boundaries, symmetry, expected in (
        (None, (), (True, True, True)),
        ("metallic", (), (False, False, False)),
        ({"y": "metallic"}, (), (True, False, True)),
        ({"x": "metallic"}, ("Z",), (False, True, False)),
        (None, ("X", "Y"), (False, False, True)),
        ({"z": "metallic"}, ("X",), (False, True, False)),
    ):
        grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 2.0), boundaries=boundaries,
                    symmetry=symmetry)
        wraps = _periodic_axes(grid)
        assert wraps == expected, (boundaries, symmetry, wraps)
        kinds = _boundary_kinds(grid, None)
        for axis in range(3):
            assert wraps[axis] is (kinds[axis] == PERIODIC), (
                f"axis {'xyz'[axis]} of boundaries={boundaries!r} symmetry={symmetry!r}: the "
                f"monitors call it {'wrapping' if wraps[axis] else 'terminated'} while the "
                f"stepping kernels use the {kinds[axis]!r} ghost rule."
            )
            assert grid.axis_wraps(axis) is wraps[axis]
            # ... and the two ways an axis can lose its lattice vector stay distinct.
            assert (kinds[axis] == MIRROR) is grid.is_mirrored(axis)
            assert (kinds[axis] == METALLIC) is (
                grid.is_metallic(axis) and not grid.is_mirrored(axis))
