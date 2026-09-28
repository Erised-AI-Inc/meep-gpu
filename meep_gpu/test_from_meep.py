"""
Cross-validation and refusal tests for the ``mp.Simulation`` converter.

Every numerical case here is the strongest form of the comparison available: ONE
``mp.Simulation`` object is built, lifted with :func:`~.from_meep.run_on_gpu`,
and then run by CPU MEEP itself, so the two codes cannot be given different
inputs by a transcription slip in the test. The measured quantity is the complex
relative L2 of ``result.get_array(c)`` against ``sim.get_array(component=c)`` —
whole volumes, MEEP's own layout, no slicing — and the bound asserted is this
engine's established floor for that feature class, not a loose "it ran" bar.

Several cases carry a CONTROL alongside the parity number: the same reference
compared against a run with the feature disabled or inverted. A Bloch lift that
ignored ``k_point``, a one-sided PML placed on the wrong face, a ``chi3`` that
never reached the constitutive update and an ``amp_func`` that was dropped all
produce smooth, plausible, complete fields, so the only thing separating a
working converter from a broken one is that the control must be far outside the
bar the parity number is inside. Where a control exists the test asserts both.

Everything that needs a live ``mp.Simulation`` runs in a SUBPROCESS. That is not
about the oracle — the converter needs MEEP in-process by construction — but
about the host: meep and torch each ship their own OpenMP runtime and abort when
loaded together, and a pytest session that collects this package alongside the
host application's FDTD suite reaches torch. Keeping ``import meep`` out of this
module also keeps it importable, and collectable, on a machine that has never had
MEEP installed, which is the same rule every other suite in this package follows.
The child writes JSON; the parent asserts on the numbers and prints one flushed
line per case so a long run says where it is.

The refusal half is the more important half. A converter that silently
approximates is worse than no converter: an FDTD run set up wrong returns a
smooth, finite, entirely plausible field, and nothing downstream says so. Each
refusal test therefore checks that ``gpu_compatibility`` reports the feature by
name AND that ``lift_simulation`` declines to build anything.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .driver import FdtdDriver
from .from_meep import GpuCompatibility, GpuRunResult, MigratedMonitors

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)
# The child imports `meep_gpu`, so the directory holding the package has to be on
# its path however pytest was invoked.
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


_CHILD_SCRIPT = '''"""Build one mp.Simulation, lift it, and compare against CPU MEEP itself."""
import json
import math
import sys

import numpy as np
import meep as mp

from meep_gpu import (
    MeepSimulationNotLiftable,
    gpu_compatibility,
    lift_simulation,
    run_on_gpu,
)
# Imported as MODULES, not names: the lift-side controls below replace attributes on
# them, and a `from ... import` would bind the originals here and never see the swap.
from meep_gpu import from_meep, gaussian_beam

CELL = mp.Vector3(2, 2, 4)
RESOLUTION = 10
# z = -1.05 is an Ez Yee point on both grids, so neither code splits the source
# between cells and the comparison is not measuring interpolation.
POINT = mp.Vector3(0, 0, -1.05)
# The cases name components the way the driver does; MEEP's get_array wants its
# own integer constant, and mp.Ez is 4 rather than 2 (the enum interleaves the
# cylindrical directions), so the map is read from the module, never written out.
MEEP_COMPONENTS = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez,
                   "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz}


# test_refl_angular.py's own BFAST numbers: the scaled wavevector n1*sin(theta) at
# theta = 35.7 deg in n1 = 1.4, and the Courant the test computes from it IN USER CODE
# (test_refl_angular.py:52). MEEP neither derives nor clamps a BFAST Courant, and this
# engine does not either — inventing a gate MEEP does not have would refuse valid runs.
BFAST_K = 1.4 * math.sin(math.radians(35.7))
BFAST_COURANT = (1.0 - BFAST_K) / 3.0 ** 0.5


def cw(component=mp.Ez, center=POINT, **kwargs):
    return [mp.Source(mp.ContinuousSource(frequency=1.0), component=component,
                      center=center, **kwargs)]


def simulation(**overrides):
    settings = dict(cell_size=CELL, resolution=RESOLUTION, sources=cw(),
                    force_complex_fields=True, k_point=mp.Vector3(0, 0, 0))
    settings.update(overrides)
    return mp.Simulation(**settings)


def meep_build():
    """The MEEP release and precision this child compared against.

    The parent never imports MEEP, so a bound that depends on the MEEP build, or a
    check that depends on its release, reads them from the report.
    """
    return {"meep_version": mp.__version__,
            "meep_precision": "single" if mp.is_single_precision() else "double"}


def gaussian_beam_profile(v):
    """A transverse taper that is OFF-CENTRE in both x and y.

    Deliberately not centred on the source: MEEP passes ``amp_func`` the offset
    RELATIVE to the source centre and so does this engine, and a converter that
    forwarded absolute coordinates instead would displace the profile. A taper
    centred at the origin cannot see that — the displacement is zero wherever the
    source happens to sit on an axis — so the peak is moved off both axes and the
    source itself is placed off-axis to match.
    """
    return float(np.exp(-((v.x - 0.25) ** 2 + (v.y + 0.15) ** 2) / 0.18))


def _rotated_b_conductivity_medium():
    """A medium with a rotated MAGNETIC conductivity and NO electric one.

    On MEEP 1.33.0 it cannot be built through the constructor. ``mp.Medium.__init__``
    assigns ``self.B_conductivity_offdiag = Vector3(*D_conductivity_offdiag)``
    (python/geom.py:450 on MEEP 1.33.0) — the D keyword, not the B one — so a
    ``B_conductivity_offdiag=`` argument is silently dropped and a
    ``D_conductivity_offdiag=`` argument sets BOTH sides. MEEP 1.34.0 assigns each
    keyword to its own attribute. The attribute assignment below reaches the B side
    alone on both releases, and it is a spelling a user can write.
    """
    medium = mp.Medium(epsilon=2.0)
    medium.B_conductivity_offdiag = mp.Vector3(0.1, 0.0, 0.0)
    return medium


# --- per-point epsilon routes: the same slab, spelled four ways --------------------
#
# One structure — a z-slab of epsilon 6 in vacuum — declared through each of MEEP's
# per-point material routes, so a route that lifted the WRONG structure (or none)
# cannot hide behind an unusual geometry. `SLAB_SHIFT` is half a cell at RESOLUTION
# 10, which is the registration control every structured case carries.
SLAB_HALF_THICKNESS = 0.4
SLAB_SHIFT = 0.05


def slab_material(v):  # material_function: returns an mp.Medium per point.
    return mp.Medium(epsilon=6.0 if abs(v.z) < SLAB_HALF_THICKNESS else 1.0)


def slab_material_shifted(v):  # The registration control, half a cell up.
    return mp.Medium(epsilon=6.0 if abs(v.z - SLAB_SHIFT) < SLAB_HALF_THICKNESS else 1.0)


def slab_epsilon(v):  # epsilon_func: returns a NUMBER per point.
    return 6.0 if abs(v.z) < SLAB_HALF_THICKNESS else 1.0


def dispersive_material(v):  # A callable whose poles MEEP registers only if a MEDIUM also declares them.
    return mp.Medium(epsilon=2.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])


# The grid MEEP puts a per-point conductivity into. `damping = pi*0.8` is
# test_adjoint_solver.py:329's shape (pi*fcen) with fcen = 0.8, and the design is a
# fixed random field so the loss volume is genuinely graded rather than a step —
# a shared-volume or a mis-registered read has somewhere to show.
_MATERIAL_GRID_DAMPING = math.pi * 0.8
_MATERIAL_GRID_DESIGN = np.random.default_rng(3).uniform(0.0, 1.0, (30, 30))


def _damped_material_grid():
    grid = mp.MaterialGrid(mp.Vector3(30, 30), mp.Medium(index=1.0), mp.Medium(index=3.5),
                           damping=_MATERIAL_GRID_DAMPING, do_averaging=False,
                           grid_type="U_MEAN")
    grid.update_weights(_MATERIAL_GRID_DESIGN.flatten())
    return grid


def _slab_array(shift=0.0):
    """The same slab as a permittivity VOLUME, MEEP's ndarray material route.

    Sampled 10x finer than the grid along z so half a cell (0.05) is a whole number
    of array entries — otherwise the shifted control would differ from the unshifted
    one by the array's own quantization rather than by the registration under test.
    """
    nz = 400
    z = -0.5 * CELL.z + (np.arange(nz) + 0.5) * (CELL.z / nz)
    column = np.where(np.abs(z - shift) < SLAB_HALF_THICKNESS, 6.0, 1.0)
    return np.broadcast_to(column, (4, 4, nz)).copy()


def amp_sheet_data(nx=24, ny=24, size=(1.6, 1.6)):
    """``gaussian_beam_profile`` sampled onto an array, for the amp_data route.

    Sampled at the array's own cell CENTRES, which is where MEEP's interpolator
    reads them from (``0.5 + p/size`` scaled by n, minus the half-cell in
    ``map_coordinates``) — so the array and the callable describe the same profile
    and the two routes are comparable.
    """
    x = -0.5 * size[0] + (np.arange(nx) + 0.5) * (size[0] / nx)
    y = -0.5 * size[1] + (np.arange(ny) + 0.5) * (size[1] / ny)
    data = np.zeros((nx, ny, 1), dtype=np.complex128)
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            data[i, j, 0] = gaussian_beam_profile(mp.Vector3(xv, yv, 0.0))
    return data


def ramp(t):  # A custom waveform with no carrier of its own.
    return complex(np.sin(2 * np.pi * 1.0 * t) * min(1.0, t / 1.5))


def _ramped_grid(**kwargs):
    """A mp.MaterialGrid whose weights RAMP, so every intermediate material is exercised.

    Deliberately not a binary (0/1) design: a two-valued grid is indistinguishable
    from a pair of blocks, and would pass a lift that ignored the weights entirely
    wherever the two endpoint media happened to line up with the block edges. A
    monotone ramp in z puts a DIFFERENT interpolated permittivity in every layer,
    so the whole u -> epsilon map is under test rather than its two ends.

    The ramp runs along z because that is the axis the source sits on and the axis
    the half-cell-shifted controls displace; a gradient there also splits the
    anisotropic average cleanly (n = z-hat), which is what puts the parallel rows
    on meps and the perpendicular row on 1/minveps.
    """
    weights = np.zeros((4, 4, 8))
    for k in range(8):
        weights[:, :, k] = k / 7.0
    settings = dict(grid_size=mp.Vector3(4, 4, 8), medium1=mp.Medium(epsilon=1.0),
                    medium2=mp.Medium(epsilon=6.0), weights=weights.ravel())
    settings.update(kwargs)
    return mp.MaterialGrid(**settings)


# case -> (overrides, run kwargs, component compared, controls). A control is a
# second simulation whose reference the LIFTED field must NOT match.
CASES = {
    "plain": (dict(), dict(until=3.0), "Ez", {}),
    "pml_uniform": (
        dict(cell_size=mp.Vector3(3, 3, 6), boundary_layers=[mp.PML(1.0)],
             sources=cw(center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))),
        dict(until=3.0), "Ez", {},
    ),
    "pml_z_low_only": (
        dict(cell_size=mp.Vector3(2, 2, 6),
             boundary_layers=[mp.PML(1.0, direction=mp.Z, side=mp.Low)],
             sources=cw(center=mp.Vector3(0, 0, -1.05))),
        dict(until=4.0), "Ez",
        # MEEP's boundary_side enum is High = 0, Low = 1. Reading it the intuitive
        # way absorbs the opposite face and still runs perfectly, so the run with
        # the layer on the HIGH face is the control the lift must NOT reproduce.
        {"high_face": dict(cell_size=mp.Vector3(2, 2, 6),
                           boundary_layers=[mp.PML(1.0, direction=mp.Z, side=mp.High)],
                           sources=cw(center=mp.Vector3(0, 0, -1.05)))},
    ),
    # mp.Absorber. Not a matched layer: a scalar electric AND magnetic conductivity
    # graded into the cell (python/simulation.py:316). The controls are the two ways
    # to get it plausibly wrong — substitute a PML of the same thickness, or build
    # only the electric half — and both are references the lift must NOT reproduce.
    "absorber_uniform": (
        dict(cell_size=mp.Vector3(3, 3, 6), boundary_layers=[mp.Absorber(1.0)],
             sources=cw(center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))),
        dict(until=3.0), "Ez",
        {"as_pml": dict(cell_size=mp.Vector3(3, 3, 6), boundary_layers=[mp.PML(1.0)],
                        sources=cw(center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))),
         "electric_only": "absorber_electric_only",
         "shared_sigma": "absorber_shared_sigma"},
    ),
    # ONE FACE ONLY, and run long. Two things are under test that a two-sided layer
    # cannot see. MEEP's High = 0 / Low = 1 enum, resolved independently of the PML
    # path's copy; and the SEAM RULE — on a wrapping axis an integer component's index
    # 0 is not stepped at all but filled from the owned copy at index N, so the sigma
    # governing the seam is the FAR face's (vec.hpp:1102-1104, and
    # meep_gpu.absorber._owned_indices). Taking index 0's own sigma instead drifts
    # 2.9e-06 at t = 2, 1.7e-04 at t = 4 and 1.5e-01 at t = 8 — hence until = 8, where
    # the defect is four orders clear of the bar rather than one.
    "absorber_z_low_only": (
        dict(cell_size=mp.Vector3(2, 2, 6),
             boundary_layers=[mp.Absorber(1.0, direction=mp.Z, side=mp.Low)],
             sources=cw(center=mp.Vector3(0, 0, -1.05))),
        dict(until=8.0), "Ez",
        {"high_face": dict(cell_size=mp.Vector3(2, 2, 6),
                           boundary_layers=[mp.Absorber(1.0, direction=mp.Z, side=mp.High)],
                           sources=cw(center=mp.Vector3(0, 0, -1.05)))},
    ),
    # MEEP allows PML on some faces and an absorber on others. The two mechanisms are
    # independent per face and must compose: the absorber installs NO boundary region
    # (python/simulation.py:4956), so its face keeps its own wall while the PML face
    # gets sigma. The control puts a PML on the absorber's face as well.
    "absorber_and_pml_on_different_faces": (
        dict(cell_size=mp.Vector3(3, 3, 6),
             boundary_layers=[mp.Absorber(1.0, direction=mp.X), mp.PML(1.0, direction=mp.Z)],
             sources=cw(center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0))),
        dict(until=3.0), "Ez",
        {"pml_everywhere": dict(
            cell_size=mp.Vector3(3, 3, 6),
            boundary_layers=[mp.PML(1.0, direction=mp.X), mp.PML(1.0, direction=mp.Z)],
            sources=cw(center=mp.Vector3(0, 0, -1.5), size=mp.Vector3(1, 1, 0)))},
    ),
    # The two configurations the source/absorber guard used to refuse on ANY grid,
    # now admitted on Cartesian grids: the non-integrated deposit is mirrored into
    # f_u wherever MEEP's chunking runs the unsplit recurrence, and the integrated
    # dipole is withdrawn from D/B before every curl ladder (evidence packet
    # §1.5-1.7; the guard keeps only its cylindrical arm). Both entries are the
    # refusal table's own former cases, unchanged, so the admission is measured on
    # exactly what used to be blocked. Standalone floors on this cell at until = 6:
    # 6.3253e-07 (spanning sheet) and 9.6206e-07 (seam point).
    "source_spans_the_pml": (
        dict(boundary_layers=[mp.PML(0.5)],
             sources=cw(center=mp.Vector3(0, 0, -1.05), size=mp.Vector3(2, 2, 0))),
        dict(until=6.0), "Ez", {},
    ),
    # The wrapped seam of a one-sided layer: -L/2 and +L/2 are one lattice point and
    # the HIGH face's grading owns it at peak sigma. For Ez the z layer enters
    # neither ladder axis (dsig = x, dsigu = y), and the deposit-cell arithmetic —
    # sigma_bites / in_pml_chunk with the wrapped cell 0 owned by the high face —
    # is exactly what this case exercises.
    "source_on_the_wrapped_pml_seam": (
        dict(boundary_layers=[mp.PML(0.5, direction=mp.Z, side=mp.High)],
             sources=cw(center=mp.Vector3(0, 0, -2.0))),
        dict(until=6.0), "Ez", {},
    ),
    # The INTEGRATED arm of the same admission: an is_integrated Ez in the y layer is
    # exactly the cell where the dipole's standing offset used to be decayed by the
    # split recurrence's dsigu damping (Defect A). The withdraw slot removes the
    # offset before every curl ladder; with it stubbed out this identical case reads
    # 9.5837e-01, so the bound is pinning the fix and not the geometry. Measured
    # 4.9019e-07 standalone (whole-volume complex Ez, until 6).
    "integrated_source_in_the_pml": (
        dict(boundary_layers=[mp.PML(0.5)],
             sources=[mp.Source(mp.ContinuousSource(frequency=1.0, is_integrated=True),
                                component=mp.Ez, center=mp.Vector3(0, -0.8, 0))]),
        dict(until=6.0), "Ez", {},
    ),
    # The CYLINDRICAL arm of the same admission, on the refusal table's own former
    # case verbatim: a CW Ez point 0.3 into the r-high layer of a Dcyl cell. Ez's
    # Dcyl ladder binding is dsig = R, dsigu = P (cycle_direction start = 2,
    # vec.hpp:583-589, equal to the Cartesian tables under x = R, y = P, z = Z),
    # and P never carries sigma, so this deposit always lands where MEEP runs the
    # UNSPLIT recurrence — the case is Defect B with no Cartesian analogue of a
    # rescuing corner, and it is what the Dcyl sweep measured at 4.08e+00 before
    # the f_u mirror ran on cylindrical grids. Standalone floor on this cell at
    # until = 6: 1.1016e-06.
    "cylindrical_source_in_pml": (
        dict(dimensions=mp.CYLINDRICAL, cell_size=mp.Vector3(2, 0, 4), m=0,
             boundary_layers=[mp.PML(0.5)], k_point=False,  # MEEP's own Dcyl walls.
             sources=cw(center=mp.Vector3(1.8, 0, 0))),
        dict(until=6.0), "Ez", {},
    ),
    # The refusal table's own former `cylindrical_source_on_r_wall`, verbatim —
    # zone_plate.py's spelling: a full-radius Ep plane (integer-r stencil) whose
    # outermost ring lands ON the r-max metallic wall. MEEP deposits that ring
    # into the stored wall plane and zeroes it before anything reads it
    # (find_metals/zero_metal, step.cpp:245); the engine's clamp used to fold it
    # onto the last interior ring instead — measured 7.1e-02 (bare-wall Ep
    # plane), 1.00e+00 (point Ep in the last half-open cell) before the
    # `metal_high` drop in `sources._build_source_points`, floors after.
    "cylindrical_source_on_r_wall": (
        dict(dimensions=mp.CYLINDRICAL, cell_size=mp.Vector3(2, 0, 4), m=1,
             boundary_layers=[mp.PML(0.5)], k_point=False,  # MEEP's own Dcyl walls.
             sources=cw(component=mp.Ep, center=mp.Vector3(1.0, 0, 0),
                        size=mp.Vector3(2, 0, 0))),
        # Ez, not Ep: at m = 1 the Ep drive couples all six components, and Ez is
        # the name MEEP's get_array serves through the same map as the other Dcyl
        # cases (mp.Ey is a different enum member than mp.Ep and reads back zero).
        dict(until=6.0), "Ez", {},
    ),
    # A PML that is NOT a whole number of cells: mp.PML(0.55) at resolution 10 is 5.5
    # cells, which the converter used to refuse and would otherwise have rounded. MEEP
    # snaps only the layer's EXTENT, to the nearest half cell, and normalizes the
    # profile by the raw thickness (structure.cpp pml_x / use_pml:635,683).
    #
    # The two controls are the whole-cell neighbours — the absorbers a round(), a
    # floor() or a ceil() would have substituted. Both must sit far outside the parity
    # bound, or this case would pass under the rounding it exists to remove.
    "pml_fractional_cells": (
        dict(cell_size=mp.Vector3(2, 2, 6), boundary_layers=[mp.PML(0.55)],
             sources=cw(center=mp.Vector3(0, 0, -1.05))),
        dict(until=4.0), "Ez",
        {"rounded_down": dict(cell_size=mp.Vector3(2, 2, 6),
                              boundary_layers=[mp.PML(0.5)],
                              sources=cw(center=mp.Vector3(0, 0, -1.05))),
         "rounded_up": dict(cell_size=mp.Vector3(2, 2, 6),
                            boundary_layers=[mp.PML(0.6)],
                            sources=cw(center=mp.Vector3(0, 0, -1.05)))},
    ),
    # 5.2 cells: NOT an exact half-cell count, which is the only place the two
    # quantized-vs-raw quantities can be told apart from each other. At 5.5 and at 35.5
    # the snapped extent obeys 0.5*N == c exactly, so every one of those runs is blind
    # to which of the two `prefac` divides by; here N = 10 and 0.5*N = 5.0, a 4.0000%
    # shift in peak sigma (measured) if the extent is substituted for the thickness.
    # The controls are the whole-cell neighbours as above — 5.0 doubling as the layer
    # the snapped extent would name.
    "pml_fractional_cells_uneven": (
        dict(cell_size=mp.Vector3(2, 2, 6), boundary_layers=[mp.PML(0.52)],
             sources=cw(center=mp.Vector3(0, 0, -1.05))),
        dict(until=4.0), "Ez",
        {"rounded_down": dict(cell_size=mp.Vector3(2, 2, 6),
                              boundary_layers=[mp.PML(0.5)],
                              sources=cw(center=mp.Vector3(0, 0, -1.05))),
         "rounded_up": dict(cell_size=mp.Vector3(2, 2, 6),
                            boundary_layers=[mp.PML(0.6)],
                            sources=cw(center=mp.Vector3(0, 0, -1.05)))},
    ),
    # The corpus spelling this feature exists for: planar_cavity_ldos.py's
    # mp.PML(0.5) at resolution 71 — 35.5 cells, an exact half-cell extent, on a cell
    # small enough to step here. Its controls are the same two neighbours at that
    # resolution, where one cell is 1/71 of a length unit rather than 1/10.
    "pml_fractional_cells_res71": (
        dict(cell_size=mp.Vector3(0.4, 0.4, 1.6), resolution=71,
             boundary_layers=[mp.PML(0.5, direction=mp.Z)],
             sources=cw(center=mp.Vector3(0, 0, -0.2))),
        dict(until=1.2), "Ez",
        {"rounded_down": dict(cell_size=mp.Vector3(0.4, 0.4, 1.6), resolution=71,
                              boundary_layers=[mp.PML(35.0 / 71.0, direction=mp.Z)],
                              sources=cw(center=mp.Vector3(0, 0, -0.2))),
         "rounded_up": dict(cell_size=mp.Vector3(0.4, 0.4, 1.6), resolution=71,
                            boundary_layers=[mp.PML(36.0 / 71.0, direction=mp.Z)],
                            sources=cw(center=mp.Vector3(0, 0, -0.2)))},
    ),
    "bloch": (
        # force_complex_fields is left at MEEP's False on purpose: MEEP promotes
        # this run to complex storage itself because k is nonzero, so the lift has
        # to read fields.is_real rather than the constructor argument. Reading the
        # argument builds a real driver, which refuses the Bloch phase outright.
        dict(cell_size=mp.Vector3(2, 2, 2), k_point=mp.Vector3(0.1234567, 0, 0),
             force_complex_fields=False, sources=cw(center=mp.Vector3(0, 0, 0.05))),
        dict(until=3.0), "Ez",
        # A converter that accepted k_point and dropped it produces exactly this.
        {"k_zero": dict(cell_size=mp.Vector3(2, 2, 2), k_point=mp.Vector3(0, 0, 0),
                        force_complex_fields=True,
                        sources=cw(center=mp.Vector3(0, 0, 0.05)))},
    ),
    # A Bloch phase ON AN ABSORBING AXIS, which this engine used to refuse in three
    # separate places. MEEP steps it: use_bloch makes every direction Periodic as soon
    # as any k_point is given, and a PML is graded material underneath that wrap rather
    # than a replacement for it, so the axis still repeats and the wrap still carries
    # exp(i*2*pi*k*L) — the absorber only means little survives to use it. Seven
    # scripts of MEEP's own corpus need this, pw-source.py among them.
    #
    # The negative control is the same run with the x absorber removed: if the engine
    # were quietly dropping the phase on an absorbing axis, THAT is what it would be
    # reproducing, and the two differ far above the parity floor.
    "bloch_on_pml_axis": (
        dict(cell_size=mp.Vector3(3, 1, 3), k_point=mp.Vector3(0.3, 0, 0),
             force_complex_fields=False,
             boundary_layers=[mp.PML(0.8, direction=mp.X),
                              mp.PML(0.8, direction=mp.Z)],
             sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.3),
                                component=mp.Ez, center=mp.Vector3(0, 0, -0.6))]),
        dict(until=6.0), "Ez",
        {"no_x_absorber": dict(cell_size=mp.Vector3(3, 1, 3), k_point=mp.Vector3(0.3, 0, 0),
                               force_complex_fields=False,
                               boundary_layers=[mp.PML(0.8, direction=mp.Z)],
                               sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.3),
                                                  component=mp.Ez,
                                                  center=mp.Vector3(0, 0, -0.6))])},
    ),
    # k_point=False is MEEP's DEFAULT, and it means a perfect-electric-conductor box:
    # fields::fields leaves every face Metallic and nothing calls use_bloch
    # (fields.cpp; python/simulation.py `if self.k_point:`). This was the converter's
    # single most common refusal — 37 of the 57 surveyed MEEP examples — and the shape
    # of MEEP's own array is the first thing that has to line up: N per axis here
    # against N + 1 for the periodic run of the same cell.
    "metallic": (dict(k_point=False), dict(until=2.0), "Ez", {}),
    # Long enough that the wave has crossed every wall repeatedly. A nearly-right
    # boundary is exact at t = 1 and diverges here.
    "metallic_long": (dict(k_point=False, cell_size=mp.Vector3(1.0, 1.1, 1.5),
                           sources=cw(center=mp.Vector3(0.05, -0.05, -0.25))),
                      dict(until=6.0), "Ez", {}),
    # Odd cell counts on all three axes: MEEP's centre_origin rounds DOWN TO EVEN, so
    # an odd axis starts half a cell higher, and a wall registered half a cell out is
    # still a smooth field (Grid.origin_doubled).
    "metallic_odd_counts": (dict(k_point=False, cell_size=mp.Vector3(1.1, 0.9, 1.5),
                                 sources=cw(center=mp.Vector3(0.05, -0.05, -0.25))),
                            dict(until=2.0), "Ex", {}),
    "metallic_pml": (dict(k_point=False, boundary_layers=[mp.PML(0.4)]),
                     dict(until=3.0), "Ez", {}),
    # MEEP's own default arrangement: folded AND metallic. Under PERIODIC boundaries
    # the same fold is refused once its far face goes live; under a wall the zero ghost
    # IS the boundary condition, so this runs long past that point.
    "metallic_mirror_x": (dict(k_point=False, cell_size=mp.Vector3(4, 2, 2),
                               symmetries=[mp.Mirror(mp.X)],
                               sources=cw(component=mp.Ez, center=mp.Vector3(0, 0, 0))),
                          dict(until=4.0), "Ez", {}),
    # MEEP does not promote a k_point-less run to complex, so this is also the
    # real-storage path through the whole metallic stack.
    "metallic_real_fields": (dict(k_point=False, force_complex_fields=False),
                             dict(until=2.0), "Ez", {}),
    "metallic_lorentz": (
        dict(k_point=False, default_material=mp.Medium(epsilon=2.0, E_susceptibilities=[
            mp.LorentzianSusceptibility(frequency=1.1, gamma=0.05, sigma=0.6)])),
        dict(until=3.0), "Ez",
        {"non_dispersive": dict(k_point=False, default_material=mp.Medium(epsilon=2.0))},
    ),
    # --- reduced dimensions ---------------------------------------------------
    # MEEP 2-D is TRANSLATIONAL INVARIANCE along z, not a thin slab, and its
    # components decouple into TM (Ez, Hx, Hy) and TE (Hz, Ex, Ey). Both are lifted,
    # and both are here: a boundary or an amplitude that is right for one and wrong
    # for the other is the failure this pair exists to separate. MEEP's array is
    # 2-D for these, so the shape assertion in the parent is also the rank pin.
    "two_d_tm": (
        dict(cell_size=mp.Vector3(2, 3, 0), sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ez",
        # A delta-function amplitude scaling applied to the INVARIANT axis multiplies
        # every 2-D source by the resolution — MEEP skips it there
        # (fields::nosize_direction), and skipping the skip is a smooth, stable field
        # exactly 10x too large at resolution 10.
        {"amplitude_x_resolution": dict(
            cell_size=mp.Vector3(2, 3, 0),
            sources=cw(center=mp.Vector3(0.05, -0.15, 0), amplitude=RESOLUTION))},
    ),
    "two_d_te": (
        dict(cell_size=mp.Vector3(2, 3, 0),
             sources=cw(component=mp.Hz, center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Hz", {},
    ),
    # k_point unset: MEEP's PEC box on x and y, with z still invariant. The TE case is
    # the one that separates a correct invariant axis from a metallic one — a wall on
    # z zeroes exactly Ex, Ey and Hz, so TE returns a smooth exact zero while TM stays
    # exact (Grid._require_boundaries_leave_invariant_axes_alone).
    "two_d_metallic_tm": (
        dict(k_point=False, cell_size=mp.Vector3(2, 3, 0),
             sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ez", {},
    ),
    "two_d_metallic_te": (
        dict(k_point=False, cell_size=mp.Vector3(2, 3, 0),
             sources=cw(component=mp.Hz, center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Hz", {},
    ),
    "two_d_pml": (
        dict(k_point=False, cell_size=mp.Vector3(4, 4, 0), boundary_layers=[mp.PML(0.5)],
             sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ez", {},
    ),
    # MEEP builds no boundary region for a direction its grid_volume does not have, so
    # this layer reaches nothing: measured 0.0e+00 between the two MEEP runs. The lift
    # drops it for the same reason, and the case pins that the drop is the no-op MEEP
    # makes it rather than a silently different absorber.
    "two_d_pml_on_the_invariant_axis": (
        dict(k_point=False, cell_size=mp.Vector3(4, 4, 0),
             boundary_layers=[mp.PML(0.5), mp.PML(0.5, direction=mp.Z)],
             sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ez", {},
    ),
    "two_d_lorentz": (
        dict(cell_size=mp.Vector3(2, 3, 0), sources=cw(center=mp.Vector3(0.05, -0.15, 0)),
             default_material=mp.Medium(epsilon=2.0, E_susceptibilities=[
                 mp.LorentzianSusceptibility(frequency=1.5, gamma=0.1, sigma=0.4)])),
        dict(until=4.0), "Ez",
        {"non_dispersive": dict(default_material=mp.Medium(epsilon=2.0))},
    ),
    "two_d_mirror_x": (
        dict(k_point=False, cell_size=mp.Vector3(2, 3, 0), symmetries=[mp.Mirror(mp.X)],
             sources=cw(center=mp.Vector3(0, -0.15, 0))),
        dict(until=4.0), "Ez", {},
    ),
    "two_d_bloch": (
        dict(cell_size=mp.Vector3(2, 3, 0), k_point=mp.Vector3(0.1234567, -0.3, 0),
             force_complex_fields=False, sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ez",
        {"k_zero": dict(cell_size=mp.Vector3(2, 3, 0), k_point=mp.Vector3(0, 0, 0),
                        force_complex_fields=True,
                        sources=cw(center=mp.Vector3(0.05, -0.15, 0)))},
    ),
    # --- special_kz: the out-of-plane wavevector a 2-D run carries as a PHASE -------
    # `kz_2d` on a zero-thickness cell with a nonzero k_point.z puts that component in
    # MEEP's `fields::beta` slot instead of use_bloch (python/simulation.py:1555-1570,
    # :2478-2504). The third axis stays analytic — exp(i*2*pi*beta*z), so d/dz is the
    # EXACT factor i*2*pi*beta — and the curl gains an i*beta*zhat x cross product that
    # COUPLES TM to TE (step_db.cpp:148-176). k_point.x = k_point.y = 0 here on purpose:
    # that is the only way to reach MEEP's real-storage arm, where the whole coupling
    # collapses to a plain real add under the implicit-i-on-TM convention. The pair
    # therefore differs ONLY in storage mode, and `real_imag` is the sole case in this
    # file that exercises the real beta path at all.
    #
    # The compared component is Ex, NOT the driven Ez: a TM point source in a 2-D run
    # leaves Ex identically zero unless the beta term runs, so Ex measures the coupling
    # itself rather than the run around it (Ez here has norm 1.9e+01 against Ex's
    # 5.1e+00, and an engine with no beta term still reproduces the shape of Ez). The
    # control is the SAME simulation at -beta, whose Ex is this one's negated: it is the
    # reference a wrong relative sign between the D and B couplings — or between the
    # X and Y ones — lands on, and it is smooth, complete and the right magnitude.
    "special_kz_complex": (
        dict(cell_size=mp.Vector3(2, 3, 0), k_point=mp.Vector3(0, 0, 0.35), kz_2d="complex",
             sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ex",
        {"beta_sign_flipped": dict(cell_size=mp.Vector3(2, 3, 0),
                                   k_point=mp.Vector3(0, 0, -0.35), kz_2d="complex",
                                   sources=cw(center=mp.Vector3(0.05, -0.15, 0)))},
    ),
    "special_kz_real_imag": (
        dict(cell_size=mp.Vector3(2, 3, 0), k_point=mp.Vector3(0, 0, 0.35), kz_2d="real/imag",
             force_complex_fields=False, sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ex",
        {"beta_sign_flipped": dict(cell_size=mp.Vector3(2, 3, 0),
                                   k_point=mp.Vector3(0, 0, -0.35), kz_2d="real/imag",
                                   force_complex_fields=False,
                                   sources=cw(center=mp.Vector3(0.05, -0.15, 0)))},
    ),
    # The DRIVEN component of the same two runs, with the beta term's ABSENCE as the
    # control: k_point = 0 is the run an engine that accepted kz_2d and ignored it
    # produces, and Ez is exactly where that engine looks most nearly right.
    "special_kz_complex_driven": (
        dict(cell_size=mp.Vector3(2, 3, 0), k_point=mp.Vector3(0, 0, 0.35), kz_2d="complex",
             sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ez",
        {"no_beta": dict(cell_size=mp.Vector3(2, 3, 0), k_point=mp.Vector3(0, 0, 0),
                         sources=cw(center=mp.Vector3(0.05, -0.15, 0)))},
    ),
    "special_kz_real_imag_driven": (
        dict(cell_size=mp.Vector3(2, 3, 0), k_point=mp.Vector3(0, 0, 0.35), kz_2d="real/imag",
             force_complex_fields=False, sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ez",
        {"no_beta": dict(cell_size=mp.Vector3(2, 3, 0), k_point=mp.Vector3(0, 0, 0),
                         force_complex_fields=False,
                         sources=cw(center=mp.Vector3(0.05, -0.15, 0)))},
    ),
    # --- BFAST: the broadband fixed-angle source technique --------------------------
    # test_refl_angular.py's own configuration at a resolution a unit case can afford:
    # MEEP's `bfast_scaled_k` shears time by t -> t - k.r/c, adding +d/dt(k x E) to
    # dB/dt and -d/dt(k x H) to dD/dt (step_db.cpp:129-142), so a planewave at a FIXED
    # angle is transversely uniform at every frequency and plain periodic boundaries
    # serve a whole band. The compared component is the driven Ex; Ez and Hy are the
    # other two P-polarization components and Ey/Hx/Hz stay identically zero.
    #
    # TWO CONTROLS, and they are the same physical mistake reached from both sides.
    # `no_bfast` is MEEP's OWN run of the same cell at the same Courant with the
    # shear removed — a NORMAL-INCIDENCE run, smooth, complete and converged, whose
    # analytic Fresnel reflectance is ((1.4-3.5)/(1.4+3.5))^2 = 0.18367 against
    # BFAST's 0.124120 at 35.7 deg, i.e. 48 % away. `bfast_dropped_on_lift` reaches
    # the same answer from the engine side by zeroing the k after the lift, which is
    # exactly the blind spot the gate had until 2026-08-04.
    "bfast_fixed_angle": (
        dict(dimensions=3, resolution=20, cell_size=mp.Vector3(z=9.0),
             default_material=mp.Medium(index=1.4), force_complex_fields=False,
             sources=[mp.Source(mp.GaussianSource(2.0, fwidth=1.0), component=mp.Ex,
                                center=mp.Vector3(z=-3.5))],
             boundary_layers=[mp.PML(1.0)], k_point=mp.Vector3(),
             bfast_scaled_k=(BFAST_K, 0, 0), Courant=BFAST_COURANT,
             geometry=[mp.Block(size=mp.Vector3(mp.inf, mp.inf, 4.5),
                                center=mp.Vector3(z=2.25), material=mp.Medium(index=3.5))]),
        dict(until=8.0), "Ex",
        {"no_bfast": dict(bfast_scaled_k=(0, 0, 0)),
         "bfast_dropped_on_lift": "bfast_dropped"},
    ),
    # BFAST ALONGSIDE A NONZERO k_point. MEEP allows both and hands them to two
    # DIFFERENT constructor slots — `bfast_scaled_k` to `mp.fields` (simulation.py:2486)
    # and `k_point` to `use_bloch` — so they are independent knobs that happen never to
    # meet in the corpus: test_refl_angular's BFAST branch sets `k = mp.Vector3()`
    # precisely because the shear is what fixes the angle, and its non-BFAST branch sets
    # the k instead. Refusing the pair would have refused a valid MEEP run on the
    # strength of a sample; measuring it is the alternative. A real x extent, so the
    # Bloch phase is a wrap across many cells rather than the one-pixel `nosize_direction`
    # case, with the shear on the SAME axis the phase runs along — the arrangement in
    # which a term folded into the wrong operand could hide behind the other.
    "bfast_with_bloch_k": (
        dict(dimensions=3, resolution=20, cell_size=mp.Vector3(1.0, 0.0, 6.0),
             default_material=mp.Medium(index=1.4), force_complex_fields=True,
             sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.6), component=mp.Ex,
                                center=mp.Vector3(z=-2.0))],
             boundary_layers=[mp.PML(1.0, direction=mp.Z)],
             k_point=mp.Vector3(0.23, 0, 0), bfast_scaled_k=(0.35, 0, 0),
             Courant=(1 - 0.35) / 3 ** 0.5,
             geometry=[mp.Block(size=mp.Vector3(mp.inf, mp.inf, 2.0),
                                center=mp.Vector3(z=1.0), material=mp.Medium(index=2.5))]),
        dict(until=6.0), "Ex",
        # The control is the same Bloch run with the shear dropped: an engine that
        # took the k_point and ignored the bfast_scaled_k returns it, and it is a
        # complete, smooth, converged field at a different incidence angle.
        {"no_bfast": dict(bfast_scaled_k=(0, 0, 0))},
    ),
    "two_d_structured": (
        dict(k_point=False, cell_size=mp.Vector3(4, 4, 0), boundary_layers=[mp.PML(0.5)],
             sources=cw(center=mp.Vector3(0.05, -0.15, 0)),
             geometry=[mp.Block(size=mp.Vector3(0.6, mp.inf, mp.inf),
                                center=mp.Vector3(0.3, 0, 0), material=mp.Medium(epsilon=4.0))]),
        dict(until=4.0), "Ez",
        {"no_geometry": dict(geometry=[])},
    ),
    # ring_gds.py's class: a structured cell whose BACKGROUND is not vacuum (Si in
    # SiO2 here as there), metallic walls behind the PML. The lattice rows this
    # grid stores on the low walls are owned by no MEEP chunk (grid_volume::owns
    # needs o > 0, vec.cpp:445-463), so fields::get_chi1inv answers its
    # outside-the-cell default — literal vacuum, monitor.cpp:180-183 — below the
    # declared span. The lift must EXCLUDE exactly those measured planes from the
    # declared-media spread check (and nothing else: the forced gate
    # `wall_exclusion_disabled_spread_still_refuses` shows the check still bites)
    # and then hold the structured floor, wall rows included in the comparison.
    "structured_nonvacuum_background": (
        dict(k_point=False, cell_size=mp.Vector3(4, 4, 0), boundary_layers=[mp.PML(0.5)],
             default_material=mp.Medium(index=1.4),
             sources=cw(center=mp.Vector3(0.05, -0.15, 0)),
             geometry=[mp.Block(size=mp.Vector3(0.6, mp.inf, mp.inf),
                                center=mp.Vector3(0.3, 0, 0), material=mp.Medium(index=3.4))]),
        dict(until=4.0), "Ez",
        {"no_geometry": dict(geometry=[])},
    ),
    "two_d_odd_counts": (
        dict(cell_size=mp.Vector3(2.1, 3.3, 0), sources=cw(center=mp.Vector3(0.07, -0.13, 0))),
        dict(until=4.0), "Ez", {},
    ),
    # dimensions=2 with a nonzero cell_size.z: MEEP never reads that extent (vol2d takes
    # x and y), so this is the same run as the (2, 3, 0) cell — 0.0e+00 apart in MEEP
    # itself. The lift zeroes it explicitly rather than passing it to a Grid that
    # requires zero there.
    "two_d_declared_with_z_extent": (
        dict(dimensions=2, cell_size=mp.Vector3(2, 3, 4),
             sources=cw(center=mp.Vector3(0.05, -0.15, 0))),
        dict(until=4.0), "Ez", {},
    ),
    # An EigenModeSource, lifted by re-running MEEP's own synthesis: the MPB mode
    # solve via sim.get_eigenmode, then the four equivalent-current deposits through
    # this engine's own add_volume_source transcription (_lift_eigenmode_source).
    # The control is the degradation this entry used to refuse against — the same
    # source as a plain Ez sheet, which launches BOTH ways with the wrong profile
    # and is exactly what a converter that ignored the eigenmode data produces.
    "eigenmode_source_waveguide": (
        dict(cell_size=mp.Vector3(8, 6, 0), boundary_layers=[mp.PML(1.0)],
             geometry=[mp.Block(mp.Vector3(mp.inf, 0.5, mp.inf),
                                material=mp.Medium(epsilon=12))],
             sources=[mp.EigenModeSource(
                 mp.GaussianSource(frequency=1.0, fwidth=0.2),
                 center=mp.Vector3(-2, 0), size=mp.Vector3(0, 3), eig_band=1)]),
        dict(until=12.0), "Ez",
        {"plain_ez_sheet": dict(sources=[mp.Source(
            mp.GaussianSource(frequency=1.0, fwidth=0.2), component=mp.Ez,
            center=mp.Vector3(-2, 0), size=mp.Vector3(0, 3))])},
    ),
    # The TE polarization of the same launch (eig_parity=EVEN_Z -> Hz/Ex/Ey). Not
    # redundant with the TM case: each polarization drives a DIFFERENT pair of the
    # four equivalent-current sheets (TM: Ez+Hy, TE: Ey+Hz), and the mutation battery
    # proved it — a sign flip on the Hz sheet SURVIVED the TM case because the TM
    # mode's Ey is ~zero, so that sheet carries no current there to flip.
    "eigenmode_source_waveguide_te": (
        dict(cell_size=mp.Vector3(8, 6, 0), boundary_layers=[mp.PML(1.0)],
             geometry=[mp.Block(mp.Vector3(mp.inf, 0.5, mp.inf),
                                material=mp.Medium(epsilon=12))],
             sources=[mp.EigenModeSource(
                 mp.GaussianSource(frequency=1.0, fwidth=0.2),
                 center=mp.Vector3(-2, 0), size=mp.Vector3(0, 3), eig_band=1,
                 eig_parity=mp.EVEN_Z)]),
        dict(until=12.0), "Hz", {},
    ),
    # THE SAME LAUNCH AT A NONZERO beta — MEEP's special_kz — in both storage modes.
    # An eigenmode source is the one source kind that reaches beta twice over: the mode
    # SOLVE carries kz (mpb.cpp:394 kcart[2] = beta, held fixed through the Newton
    # frequency match at :489/:509/:611), and the injected sheets then need MEEP's own
    # source-side phasefix in real storage. Both controls are lift-side, because both
    # defects leave the MEEP input untouched:
    #   no_kz_phasefix      — the -i on Ez/Hx/Hy that add_eigenmode_source applies and
    #                         fields::get_eigenmode does not (mpb.cpp:847). Real
    #                         storage only; measured 1.45 without it.
    #   full_3d_sample_point — the mode sampled at the component's real z Yee offset
    #                         instead of in MEEP's reduced coordinates, which costs
    #                         exactly cos(pi*beta*dx) of the whole injected mode.
    #                         Measured 4.9e-04 at kz = 0.2, 2.0e-03 at kz = 0.4, and
    #                         INVISIBLE at beta = 0 — no other eigenmode case can see
    #                         it. kz = 0.4 is used here to put it clear of the bar.
    "eigenmode_source_special_kz": (
        dict(cell_size=mp.Vector3(8, 6, 0), boundary_layers=[mp.PML(1.0)],
             k_point=mp.Vector3(0, 0, 0.4), kz_2d="complex",
             geometry=[mp.Block(mp.Vector3(mp.inf, 0.5, mp.inf),
                                material=mp.Medium(epsilon=12))],
             sources=[mp.EigenModeSource(
                 mp.GaussianSource(frequency=1.0, fwidth=0.2),
                 center=mp.Vector3(-2, 0), size=mp.Vector3(0, 3), eig_band=1)]),
        dict(until=12.0), "Ez",
        {"full_3d_sample_point": "full_3d_sample_point"},
    ),
    "eigenmode_source_special_kz_real_imag": (
        dict(cell_size=mp.Vector3(8, 6, 0), boundary_layers=[mp.PML(1.0)],
             k_point=mp.Vector3(0, 0, 0.4), kz_2d="real/imag", force_complex_fields=False,
             geometry=[mp.Block(mp.Vector3(mp.inf, 0.5, mp.inf),
                                material=mp.Medium(epsilon=12))],
             sources=[mp.EigenModeSource(
                 mp.GaussianSource(frequency=1.0, fwidth=0.2),
                 center=mp.Vector3(-2, 0), size=mp.Vector3(0, 3), eig_band=1)]),
        dict(until=12.0), "Ez",
        {"no_kz_phasefix": "no_kz_phasefix", "full_3d_sample_point": "full_3d_sample_point"},
    ),
    # THE SAME TM LAUNCH UNDER A FOLD, compared on Hy. mode-decomposition.py's own
    # configuration in miniature: a windowed EigenModeSource straddling mp.Mirror(mp.Y)
    # with eig_parity=ODD_Z+EVEN_Y. It is the only case in this table that puts a
    # MAGNETIC current on a folded axis, and it is compared on Hy rather than Ez
    # because Hy is the only B component with Yee shift 0 on Y — the only one whose
    # cell 0 is a mirror ghost rather than an owned cell (little_owned_corner0(c) =
    # little_corner + 2 - iyee_shift(c)). Every other folded case in this file drives
    # an electric dipole, which is exactly why the B-side fill running a sub-step
    # early survived all of them: mode-decomposition.py at resolution 17 read Hy
    # 2.72e-03 while Ez and Hx sat at 1.32e-06 / 2.51e-06, and 1.62e-06 on Hy once the
    # fill moved to the driver's post-injection slot.
    "eigenmode_source_waveguide_folded": (
        dict(cell_size=mp.Vector3(8, 6, 0), boundary_layers=[mp.PML(1.0)],
             symmetries=[mp.Mirror(mp.Y)],
             geometry=[mp.Block(mp.Vector3(mp.inf, 0.5, mp.inf),
                                material=mp.Medium(epsilon=12))],
             sources=[mp.EigenModeSource(
                 mp.GaussianSource(frequency=1.0, fwidth=0.2),
                 center=mp.Vector3(-2, 0), size=mp.Vector3(0, 3), eig_band=1,
                 eig_parity=mp.ODD_Z + mp.EVEN_Y)]),
        dict(until=12.0), "Hy",
        {"b_ghosts_filled_before_the_source": "b_ghosts_filled_before_the_source"},
    ),
    # OBLIQUE launches — direction=NO_DIRECTION, where the propagation axis comes from
    # eig_kpoint and not from the source normal. MEEP's synthesis is unchanged (the same
    # four Love-equivalence sheets, mpb.cpp:872-900); what changes is the MPB SOLVE, so
    # the lift passes NO_DIRECTION through to sim.get_eigenmode while taking the sheet
    # normal from the source volume. oblique-planewave.py's own cell, at resolution 20:
    # a 40-deg planewave in n=1.5 with a matching Bloch k_point, PML on x only, and an
    # eig_vol one pixel tall inside a source line 6 um long — MEEP skips the
    # eig_vol-contains-where requirement for NO_DIRECTION (mpb.cpp:371) and this case
    # depends on it, so no containment pre-flight may be added.
    "eigenmode_oblique_planewave": (
        dict(cell_size=mp.Vector3(10, 6, 0), resolution=20,
             boundary_layers=[mp.PML(thickness=2.0, direction=mp.X)],
             default_material=mp.Medium(index=1.5),
             k_point=mp.Vector3(1.5, 0, 0).rotate(mp.Vector3(0, 0, 1), np.radians(40.0)),
             sources=[mp.EigenModeSource(
                 mp.ContinuousSource(frequency=1.0),
                 center=mp.Vector3(), size=mp.Vector3(0, 6, 0),
                 direction=mp.NO_DIRECTION, eig_band=1, eig_parity=mp.ODD_Z,
                 eig_kpoint=mp.Vector3(1.5, 0, 0).rotate(
                     mp.Vector3(0, 0, 1), np.radians(40.0)),
                 eig_vol=mp.Volume(center=mp.Vector3(),
                                   size=mp.Vector3(0, 1 / 20, 0)))]),
        dict(until=12.0), "Ez",
        {"axis_solve": "axis_solve", "no_phase": "no_phase",
         "plain_ez_sheet": dict(sources=[mp.Source(
             mp.ContinuousSource(frequency=1.0), component=mp.Ez,
             center=mp.Vector3(), size=mp.Vector3(0, 6, 0))])},
    ),
    # oblique-source.py's cell: a 20-deg rotated eps=12 waveguide with eig_vol=None, so
    # eig_vol DOES contain where and the axis_solve degradation converges instead of
    # aborting — the silent case. force_complex_fields is set because the oblique mode
    # profile is a complex in-plane ramp, which this engine refuses to project onto real
    # storage (see the eigenmode_no_direction_real_fields refusal).
    "eigenmode_oblique_waveguide": (
        dict(cell_size=mp.Vector3(10, 10, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(thickness=2.0)],
             geometry=[mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, 1.0, mp.inf),
                                e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), np.radians(20)),
                                e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), np.radians(20)),
                                material=mp.Medium(epsilon=12))],
             sources=[mp.EigenModeSource(
                 mp.GaussianSource(0.15, fwidth=0.03),
                 center=mp.Vector3(), size=mp.Vector3(y=3.0),
                 direction=mp.NO_DIRECTION, eig_band=1, eig_parity=mp.ODD_Z,
                 eig_kpoint=mp.Vector3(x=1).rotate(mp.Vector3(z=1), np.radians(20)))]),
        dict(until=40.0), "Ez",
        {"axis_solve": "axis_solve", "no_phase": "no_phase",
         "plain_ez_sheet": dict(sources=[mp.Source(
             mp.GaussianSource(0.15, fwidth=0.03), component=mp.Ez,
             center=mp.Vector3(), size=mp.Vector3(y=3.0))])},
    ),
    # GAUSSIAN BEAMS, lifted the same way: MEEP's analytic beam evaluator
    # (gaussianbeam::get_fields, transcribed in meep_gpu.gaussian_beam and validated
    # against MEEP's own compiled routine in test_gaussian_beam.py) plus the same four
    # equivalent-current sheets, deposited through this engine's add_volume_source.
    #
    # The 2-D TM case carries z components on beam_x0 and beam_kdir, which
    # py_v3_to_vec keeps on a D2 vec (simulation.py:130) and get_fields reads: the
    # `reduce_beam_vectors` control is the dimensional reduction that is CORRECT for a
    # source's centre and size and wrong for the beam vectors, and it lands at 8.4e+00.
    "beam_two_d_tm": (
        dict(cell_size=mp.Vector3(10, 10, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(2.0)],
             sources=[mp.GaussianBeamSource(
                 mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, -2),
                 size=mp.Vector3(6, 0), beam_x0=mp.Vector3(0, 2, 0.7),
                 beam_kdir=mp.Vector3(0, 1, 0.5), beam_w0=0.8,
                 beam_E0=mp.Vector3(0, 0, 1))]),
        dict(until=6.0), "Ez",
        {"flip_sheet_signs": "flip_sheet_signs",
         "reduce_beam_vectors": "reduce_beam_vectors",
         "plain_ez_sheet": dict(sources=[mp.Source(
             mp.ContinuousSource(frequency=1.0), component=mp.Ez,
             center=mp.Vector3(0, -2), size=mp.Vector3(6, 0))])},
    ),
    # amplitude=5.0 is deliberate and is the whole control for one MEEP behaviour:
    # MEEP IGNORES it on a beam (measured deposited-current ratio exactly 1.000000,
    # test_gaussian_beam.py), because GaussianBeam3DSource.add_source never passes it
    # (source.py:751-773). So the CPU-MEEP reference is the amplitude-1 run, and a lift
    # that honoured it — which is exactly what an EigenModeSource correctly does — would
    # be 5x too large. The `beam_honours_the_source_amplitude` mutation anchor pins it.
    # The OTHER polarization, and not redundant: MEEP's D2 parity filter
    # (sources.cpp:501-504) keeps a different PAIR of the four sheets for each, so
    # `flip_sheet_signs` is invisible here and `flip_other_sheet_signs` is invisible in
    # the TM case. Measured, both ways round.
    "beam_two_d_te": (
        dict(cell_size=mp.Vector3(10, 10, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(2.0)],
             sources=[mp.GaussianBeamSource(
                 mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, -2),
                 size=mp.Vector3(6, 0), beam_x0=mp.Vector3(0, 2),
                 beam_kdir=mp.Vector3(0, 1), beam_w0=0.8,
                 beam_E0=mp.Vector3(1, 0, 0), amplitude=5.0)]),
        dict(until=6.0), "Ey",
        {"flip_other_sheet_signs": "flip_other_sheet_signs",
         "plain_ey_sheet": dict(sources=[mp.Source(
             mp.ContinuousSource(frequency=1.0), component=mp.Ey,
             center=mp.Vector3(0, -2), size=mp.Vector3(6, 0))])},
    ),
    # The 3-D x-polarized beam is the ONLY configuration in which MEEP's uninitialized
    # gb_Hx read reaches a sheet with unit weight, so it is the only one where the
    # `hx_is_ey` control — the "correct" physics — separates from parity at all.
    "beam_three_d": (
        dict(cell_size=mp.Vector3(4, 4, 4), resolution=10, k_point=False,
             boundary_layers=[mp.PML(0.5)],
             sources=[mp.GaussianBeamSource(
                 mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, 0, -1),
                 size=mp.Vector3(3, 3, 0), beam_x0=mp.Vector3(0, 0, 1),
                 beam_kdir=mp.Vector3(0, 0, 1), beam_w0=0.8,
                 beam_E0=mp.Vector3(1, 0, 0))]),
        dict(until=4.0), "Ex",
        {"hx_is_ey": "hx_is_ey", "flip_sheet_signs": "flip_sheet_signs",
         "plain_ex_sheet": dict(sources=[mp.Source(
             mp.ContinuousSource(frequency=1.0), component=mp.Ex,
             center=mp.Vector3(0, 0, -1), size=mp.Vector3(3, 3, 0))])},
    ),
    # THE SAME BEAM ON REAL float32 STORAGE — MEEP's default, and what MEEP's own
    # gaussian-beam.py declares by declaring nothing. This is the case the engine used
    # to report `supported=True` for and then raise inside driver.run(): a beam's
    # spatial amplitude is complex by construction (wavefront curvature and the Gouy
    # phase), and the injection layer refused to project it. It now transcribes MEEP's
    # unconditional `real(A)` (step.cpp:307).
    #
    # The cell is gaussian-beam.py's own — 14 um, mp.PML(2), beam_x0 3 um downstream,
    # w0 = 0.8 — at resolution 20 rather than 50, with the source sheet pulled to
    # size 10 so it stops at the absorber's inner edge instead of spanning it,
    # keeping this case about REAL STORAGE alone (a full-width sheet is admitted
    # now — the `source_spans_the_pml` parity case — but overlaps the layer and
    # would fold the deposit-mirror machinery into a beam-storage measurement).
    "beam_real_fields": (
        dict(force_complex_fields=False, k_point=False,
             cell_size=mp.Vector3(14, 14, 0), resolution=20,
             boundary_layers=[mp.PML(2.0)],
             sources=[mp.GaussianBeamSource(
                 src=mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, -4.0),
                 size=mp.Vector3(10.0), beam_x0=mp.Vector3(0, 3.0),
                 beam_kdir=mp.Vector3(0, 1, 0), beam_w0=0.8,
                 beam_E0=mp.Vector3(0, 0, 1))]),
        dict(until=8.0), "Ez",
        {"drop_beam_phase": "drop_beam_phase",
         "plain_ez_sheet": dict(sources=[mp.Source(
             mp.ContinuousSource(frequency=1.0), component=mp.Ez,
             center=mp.Vector3(0, -4.0), size=mp.Vector3(10.0))])},
    ),
    # A PULSE on the same cell. Not redundant with the CW case above and it is the half
    # that decides the policy: the argument for projecting rather than refusing is that
    # `Re(A e^{-iwt})` with complex A is a real sinusoid carrying the right amplitude
    # AND the right phase, so the spatial phase becomes a per-point time shift that is
    # physically correct at monochromatic steady state. A GaussianSource is never
    # monochromatic and never at steady state, so if the projection held only where
    # that reading does, this is the case that would show it.
    "beam_real_fields_pulse": (
        dict(force_complex_fields=False, k_point=False,
             cell_size=mp.Vector3(14, 14, 0), resolution=20,
             boundary_layers=[mp.PML(2.0)],
             sources=[mp.GaussianBeamSource(
                 src=mp.GaussianSource(frequency=1.0, fwidth=0.4),
                 center=mp.Vector3(0, -4.0), size=mp.Vector3(10.0),
                 beam_x0=mp.Vector3(0, 3.0), beam_kdir=mp.Vector3(0, 1, 0),
                 beam_w0=0.8, beam_E0=mp.Vector3(0, 0, 1))]),
        dict(until=8.0), "Ez",
        {"drop_beam_phase": "drop_beam_phase",
         "plain_ez_sheet": dict(sources=[mp.Source(
             mp.GaussianSource(frequency=1.0, fwidth=0.4), component=mp.Ez,
             center=mp.Vector3(0, -4.0), size=mp.Vector3(10.0))])},
    ),
    # --- mp.GaussianBeam2DSource: MEEP's own synthesis, CALLED rather than transcribed --
    # The exact 2-D beam, which is a different mechanism from every case above: a
    # Hankel-function Green's function evaluated ENTIRELY IN PYTHON before add_source
    # is reached (get_fields -> green2d, source.py:821-1044) and then deposited as two
    # ordinary mp.Source objects carrying amp_data (get_equiv_sources, source.py:780-812).
    # So nothing here is transcribed: `_lift_gaussian_beam_2d_source` calls MEEP's own
    # synthesis and routes the deposit through the plain amp_data path.
    #
    # The cell is test_gaussianbeam.py's own (14 um, PML 2, source at y = -4, beam
    # focus 7 um downstream rotated -40 degrees, w0 = 0.8) at resolution 15 rather
    # than 25, which is the only thing changed and is a cost cap.
    #
    # THE CONTROLS ARE THE TWO WAYS TO GET THIS PLAUSIBLY WRONG:
    #   electric_current_only — deposit K = nHat x H and skip N = -nHat x E. Love's
    #     equivalence needs BOTH for the sheet to radiate into one half-space; with
    #     one, the beam launches in both directions at half amplitude on a run that
    #     completes and looks entirely healthy.
    #   beam_3d_instead — the 3-D complex-point-source beam on the same declaration,
    #     which is the approximation this class exists to replace. If the two matched,
    #     this case would be measuring nothing that `beam_two_d_tm` does not.
    "gaussian_beam_2d": (
        dict(cell_size=mp.Vector3(14, 14, 0), resolution=15, k_point=False,
             boundary_layers=[mp.PML(2.0)],
             sources=[mp.GaussianBeam2DSource(
                 src=mp.GaussianSource(frequency=1.0, fwidth=0.2),
                 center=mp.Vector3(0, -4.0), size=mp.Vector3(14.0),
                 beam_x0=mp.Vector3(0, 7.0).rotate(mp.Vector3(0, 0, 1), np.radians(-40)),
                 beam_kdir=mp.Vector3(0, 1, 0).rotate(mp.Vector3(0, 0, 1), np.radians(-40)),
                 beam_w0=0.8, beam_E0=mp.Vector3(0, 0, 1))]),
        dict(until=8.0), "Ez",
        {"electric_current_only": "electric_current_only",
         "beam_3d_instead": dict(sources=[mp.GaussianBeam3DSource(
             src=mp.GaussianSource(frequency=1.0, fwidth=0.2),
             center=mp.Vector3(0, -4.0), size=mp.Vector3(14.0),
             beam_x0=mp.Vector3(0, 7.0).rotate(mp.Vector3(0, 0, 1), np.radians(-40)),
             beam_kdir=mp.Vector3(0, 1, 0).rotate(mp.Vector3(0, 0, 1), np.radians(-40)),
             beam_w0=0.8, beam_E0=mp.Vector3(0, 0, 1))])},
    ),
    # --- a DIAGONAL anisotropic D_conductivity ---------------------------------------
    # MEEP reads D_conductivity_diag.x/.y/.z for Dx/Dy/Dz (get_cnd,
    # meepgeom.cpp:1545-1559) and this engine's Fields._set_conductivity_side has always
    # stored sigma, condfac and condinv per component — it is what the mp.Absorber path
    # installs. Only the driver's public setter funnelled everything into one volume,
    # and that is what the refusal named.
    #
    # The three values are DELIBERATELY all different and one of them is ZERO: a lift
    # that took any single component's value would be caught, and the zero puts one
    # component on the bit-identical lossless path while its neighbours are lossy.
    # All three E components are driven for the same reason — a source on one component
    # alone would leave the other two rows untested.
    "anisotropic_conductivity": (
        dict(cell_size=mp.Vector3(3, 3, 3), resolution=12, k_point=False,
             boundary_layers=[mp.PML(0.6)],
             default_material=mp.Medium(epsilon=2.0,
                                        D_conductivity_diag=mp.Vector3(0.0, 1.0, 0.4)),
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=component,
                                center=mp.Vector3())
                      for component in (mp.Ex, mp.Ey, mp.Ez)]),
        dict(until=12.5), "Ey",
        {"conductivity_shared_from_x": "conductivity_shared_from_x",
         "conductivity_shared_from_y": "conductivity_shared_from_y"},
    ),
    # --- a STRUCTURED D_conductivity: media that differ in their loss -----------------
    # The uniform case above is a declaration the lift can read off a medium. This one
    # is not: sigma varies from point to point, so it comes back off MEEP's OWN D-row
    # chi1inv at a nonzero frequency (`_read_conductivity_volumes`). The read is exact —
    # the D row's instantaneous tensor is the identity, because structure::set_materials
    # fills chi1inv for E and H only (structure.cpp:374-384) — so monitor.cpp:339-343's
    # `1 + i*sigma/frequency` is all that is left in it.
    #
    # The control is the whole point: dropping the conductivity leaves a COMPLETE run
    # with the full signal and the same step count, four orders of magnitude away.
    "structured_conductivity": (
        dict(cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(0.5)],
             geometry=[mp.Block(mp.Vector3(1.2, 1.2, mp.inf), center=mp.Vector3(0.4, 0),
                                material=mp.Medium(epsilon=12, D_conductivity=0.7))],
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                center=mp.Vector3(-1.2, 0))]),
        dict(until=6.0), "Ez",
        {"conductivity_dropped": "conductivity_dropped"},
    ),
    # The same read, with the loss coming from a MaterialGrid's `damping` instead of
    # from a declared medium. Nothing here reimplements matgrid_val, tanh_projection,
    # beta or eta: MEEP adds u*(1-u)*damping into D_conductivity_diag itself
    # (meepgeom.cpp:623-626) and the D-row read picks it up because it is in the same
    # array. test_adjoint_solver.py:329 uses damping = pi*fcen; this uses pi*0.8, and
    # the recovered volume's maximum is damping/4 — the value at u = 1/2 — which the
    # test asserts on rather than leaving to the parity number alone.
    "material_grid_damping": (
        dict(cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(0.5)],
             geometry=[mp.Block(mp.Vector3(1.5, 1.5, mp.inf), center=mp.Vector3(0.4, 0),
                                material=_damped_material_grid())],
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                center=mp.Vector3(-1.2, 0))]),
        dict(until=6.0), "Ez",
        {"conductivity_dropped": "conductivity_dropped"},
    ),
    # A structured conductivity INSIDE an mp.Absorber. MEEP's conductivity[c][d] is the
    # material value with each absorber face's ramp ADDED INTO IT
    # (geom_epsilon::conductivity, meepgeom.cpp:1596-1625), so the read comes back as
    # the SUM and the driver must not compose the ramp a second time. The control is
    # exactly that double count, which is the bug a naive "install what you read" has.
    "structured_conductivity_in_an_absorber": (
        dict(cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
             boundary_layers=[mp.Absorber(0.5)],
             geometry=[mp.Block(mp.Vector3(1.2, 1.2, mp.inf), center=mp.Vector3(0.4, 0),
                                material=mp.Medium(epsilon=12, D_conductivity=0.7))],
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                center=mp.Vector3(-1.2, 0))]),
        dict(until=6.0), "Ez",
        {"absorber_ramp_added_twice": "absorber_ramp_added_twice"},
    ),
    # --- B_conductivity: the MAGNETIC half of the same loss term ---------------------
    # MEEP treats the two sides as ONE quantity: `get_cnd` is a single switch over
    # Dx/Dy/Dz and Bx/By/Bz (meepgeom.cpp:1545-1559), structure::set_materials installs
    # them through one FOR_D_AND_B loop (structure.cpp:376-378), and step_db passes
    # s->conductivity[cc][d_c] into the SAME STEP_CURL on either side
    # (step_db.cpp:125-127). This engine was already symmetric underneath — the
    # per-component condfac/condinv path in stepping._apply_curl has carried the B side
    # since mp.Absorber needed it — so what was missing was only a way to install a
    # MATERIAL's magnetic loss. That is why this was refused while an absorber's
    # identical B-side ramp was not.
    #
    # `b_conductivity_dropped` is the whole item: it is the run this package produced
    # BEFORE the lift installed anything, except that it now completes instead of being
    # refused. Measured 1.4756e-01 against a parity of 8.7850e-07.
    # `installed_on_the_d_side` is the plausible wrong convention — "a conductivity is a
    # conductivity" — and it is the control a same-magnitude cell cannot distinguish
    # from a working lift by its parity number alone.
    "b_conductivity": (
        dict(cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(0.5)],
             default_material=mp.Medium(epsilon=2.0, B_conductivity=0.5),
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                center=mp.Vector3(-1.2, 0))]),
        dict(until=6.0), "Ez",
        {"b_conductivity_dropped": "b_conductivity_dropped",
         "installed_on_the_d_side": "installed_on_the_d_side"},
    ),
    # A DIAGONAL ANISOTROPIC B_conductivity, the exact twin of anisotropic_conductivity
    # one side over. The three values are deliberately all different and one is ZERO, so
    # a lift that took any single component's value is caught and one B component stays
    # on the bit-identical lossless path while its neighbours are lossy. All three E
    # components are driven so that all three B components are live.
    "anisotropic_b_conductivity": (
        dict(cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(0.5)],
             default_material=mp.Medium(
                 epsilon=2.0, B_conductivity_diag=mp.Vector3(0.0, 1.0, 0.4)),
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=component,
                                center=mp.Vector3(-1.2, 0))
                      for component in (mp.Ez, mp.Ex, mp.Ey)]),
        dict(until=6.0), "Ez",
        {"b_shared_from_y": dict(default_material=mp.Medium(
            epsilon=2.0, B_conductivity_diag=mp.Vector3(1.0, 1.0, 1.0))),
         "b_shared_from_z": dict(default_material=mp.Medium(
             epsilon=2.0, B_conductivity_diag=mp.Vector3(0.4, 0.4, 0.4))),
         "b_conductivity_dropped": "b_conductivity_dropped"},
    ),
    # A material B_conductivity INSIDE an mp.Absorber. MEEP adds each absorber face's
    # profile onto the material's own get_cnd value "isotropically, for both magnetic
    # and electric conductivity" (geom_epsilon::conductivity, meepgeom.cpp:1596-1600 —
    # the comment is MEEP's), so the installed sigma is the SUM. `b_ramp_overwritten` is
    # the reading that keeps only the material's declaration where it has one, which is
    # what any "the medium wins" composition produces: a complete, smooth, LESS
    # absorbing run measured 7.8083e-02 away against a parity of 7.3982e-07.
    "b_conductivity_in_an_absorber": (
        dict(cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
             boundary_layers=[mp.Absorber(0.5)],
             default_material=mp.Medium(epsilon=2.0, B_conductivity=0.5),
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                center=mp.Vector3(-1.2, 0))]),
        dict(until=6.0), "Ez",
        {"b_ramp_overwritten": "b_ramp_overwritten",
         "b_conductivity_dropped": "b_conductivity_dropped"},
    ),
    # A source running straight through BOTH y PML layers while the cell's loss is
    # confined to an interior block. The geometric conductive-PML guard used to refuse
    # this outright; the defect it protects against needs a cell where the f_cond stage
    # and the split-field recurrence are both live, which a lossless layer has none of.
    # This is test_adjoint_solver::test_damping's own source shape.
    "structured_conductivity_source_through_a_lossless_pml": (
        dict(cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(0.5)],
             geometry=[mp.Block(mp.Vector3(1.2, 1.2, mp.inf), center=mp.Vector3(0.4, 0),
                                material=mp.Medium(epsilon=12, D_conductivity=0.7))],
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                center=mp.Vector3(-1.2, 0), size=mp.Vector3(0, 4))]),
        dict(until=6.0), "Ez", {},
    ),
    # A callable returning a dispersive medium whose poles MEEP never registers, so the
    # structure it builds is non-dispersive and the lift is right to step it that way.
    # test_material_dispersion.py's own callable, on a cell with spatial extent.
    "material_function_dispersive_dropped": (
        dict(cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(0.5)],
             material_function=dispersive_material, default_material=mp.air,
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                center=mp.Vector3(-1.2, 0))]),
        dict(until=6.0), "Ez",
        {"dispersion_installed_anyway": "dispersion_installed_anyway"},
    ),
    # --- mp.metal / mp.perfect_electric_conductor: MEEP's -1e20 sentinel, passed through --
    # NOT a PEC dispatch and NOT a mask. mp.metal is Medium(epsilon=-inf) where MEEP's
    # own `inf` is the literal 1.0e20 (meep/__init__.py:4231, :4438), so the medium that
    # reaches C++ carries epsilon = -1e20 exactly; it stays material_data::MEDIUM and
    # material_epsmu takes the ORDINARY sym_matrix_invert branch (meepgeom.cpp:785-796),
    # giving chi1inv = -1e-20. `is_metal` (meepgeom.cpp:307-326) is consulted only to
    # SKIP subpixel averaging (meepgeom.cpp:1093-1094), a decision already baked into
    # the chi1inv this lift reads back. So the ordinary E = (D - sum P) * inv_eps
    # reproduces MEEP's perfect conductor verbatim, and the right implementation is
    # pass-through of MEEP's own number.
    #
    # metal-cavity-ldos.py's own cell, reduced: a metal shell around an air cavity with
    # a slot, which is the geometry that makes the wall CONDITION visible rather than a
    # metal block sitting in a corner. The control is the one relaxation that looks
    # reasonable and is measurably wrong — install the sentinel cells as vacuum, which
    # produces a complete run with the full signal and the same step count.
    "pec_cavity_2d": (
        dict(cell_size=mp.Vector3(3.2, 3.2, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(0.6)],
             geometry=[mp.Block(mp.Vector3(1.2, 1.2, mp.inf), material=mp.metal),
                       mp.Block(mp.Vector3(1.0, 1.0, mp.inf), material=mp.air),
                       mp.Block(center=mp.Vector3(0.5), size=mp.Vector3(0.2, 0.3, mp.inf),
                                material=mp.air)],
             sources=[mp.Source(mp.GaussianSource(math.sqrt(0.5), fwidth=0.2),
                                component=mp.Ez, center=mp.Vector3())]),
        dict(until=8.0), "Ez",
        {"pec_installed_as_vacuum": "pec_installed_as_vacuum"},
    ),
    # The 3-D geometry, so the fix is not pinned only on a 2-D TM cell where the metal
    # touches exactly one field component's wall condition.
    "pec_box_3d": (
        dict(cell_size=mp.Vector3(3.0, 3.0, 3.0), resolution=10, k_point=False,
             boundary_layers=[mp.PML(0.5)],
             geometry=[mp.Block(mp.Vector3(1.6, 1.6, 1.6), material=mp.metal),
                       mp.Block(mp.Vector3(1.2, 1.2, 1.2), material=mp.air),
                       mp.Block(center=mp.Vector3(0.7, 0, 0),
                                size=mp.Vector3(0.4, 0.3, 0.3), material=mp.air)],
             sources=[mp.Source(mp.GaussianSource(0.7, fwidth=0.4), component=mp.Ez,
                                center=mp.Vector3())]),
        dict(until=6.0), "Ez",
        {"pec_installed_as_vacuum": "pec_installed_as_vacuum"},
    ),
    # A metal slab running INTO the absorber, which is where the sentinel and the PML's
    # graded conductivity share cells. Nothing in the constitutive update special-cases
    # either, and this is what says so.
    "pec_touching_pml": (
        dict(cell_size=mp.Vector3(4.0, 4.0, 0), resolution=20, k_point=False,
             boundary_layers=[mp.PML(0.8)],
             geometry=[mp.Block(center=mp.Vector3(0, 1.2),
                                size=mp.Vector3(mp.inf, 0.4, mp.inf), material=mp.metal)],
             sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.5), component=mp.Ez,
                                center=mp.Vector3(0, -0.5))]),
        dict(until=8.0), "Ez",
        {"pec_installed_as_vacuum": "pec_installed_as_vacuum"},
    ),
    # THE DEGENERATE CELL — every point a perfect conductor — which takes the UNIFORM
    # epsilon path rather than the structured one, and used to be accepted by the gate
    # and then raised inside `_lift_epsilon`. Both engines step fields of order 1e-19
    # here (E = D * -1e-20), and they must agree on them: "everything is ~0" is not the
    # same claim as "the two ~0s are the same ~0".
    "pec_uniform_cell": (
        dict(cell_size=mp.Vector3(2, 2, 0), resolution=10, k_point=False,
             default_material=mp.metal,
             sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), component=mp.Ez,
                                center=mp.Vector3())]),
        dict(until=6.0), "Ez", {},
    ),
    # The OTHER half the same transcription unblocks, and the one the corpus row for
    # oblique-source.py names: an oblique eigenmode launch on real storage. Identical
    # to `eigenmode_oblique_waveguide` except that it declares the storage mode the
    # SCRIPT declares — none, so MEEP steps it real. The obliquity is an in-plane phase
    # ramp (mpb.cpp:255), so this reaches the same projection the beam does by a
    # completely different route: MPB's solved profile rather than an analytic evaluator.
    "eigenmode_oblique_waveguide_real_fields": (
        dict(force_complex_fields=False, cell_size=mp.Vector3(10, 10, 0), resolution=20,
             k_point=False, boundary_layers=[mp.PML(thickness=2.0)],
             geometry=[mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, 1.0, mp.inf),
                                e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), np.radians(20)),
                                e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), np.radians(20)),
                                material=mp.Medium(epsilon=12))],
             sources=[mp.EigenModeSource(
                 mp.GaussianSource(0.15, fwidth=0.03),
                 center=mp.Vector3(), size=mp.Vector3(y=3.0),
                 direction=mp.NO_DIRECTION, eig_band=1, eig_parity=mp.ODD_Z,
                 eig_kpoint=mp.Vector3(x=1).rotate(mp.Vector3(z=1), np.radians(20)))]),
        dict(until=40.0), "Ez",
        {"axis_solve": "axis_solve", "no_phase": "no_phase"},
    ),
    # MEEP 1-D is the Z AXIS with x and y invariant, and its grid_volume holds only
    # Ex/Dx and Hy/By (vec.hpp grid_volume::has_field). Its array is 1-D.
    "one_d_metallic": (
        dict(dimensions=1, cell_size=mp.Vector3(0, 0, 4),
             sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35))),
        dict(until=4.0), "Ex",
        # Two invariant axes, so a missing nosize_direction clause is the resolution
        # SQUARED — a hundredfold at resolution 10.
        {"amplitude_x_resolution_squared": dict(
            sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35),
                       amplitude=RESOLUTION ** 2))},
    ),
    "one_d_pml": (
        dict(dimensions=1, cell_size=mp.Vector3(0, 0, 6), boundary_layers=[mp.PML(0.5)],
             sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35))),
        dict(until=4.0), "Ex", {},
    ),
    # NOT a reduced run: mp.Vector3(0, 0, L) at dimensions=3 is how several of MEEP's
    # own 1-D examples are written, and MEEP builds a 3-D cell whose x and y hold ONE
    # CELL each ("Working in 3D dimensions. Computational cell is 0.1 x 0.1 x 4").
    # Those unit directions are then Periodic by MEEP's own default (fields.cpp), not
    # metallic, which is what the lift transcribes.
    "unit_axes_three_d": (
        dict(cell_size=mp.Vector3(0, 0, 4),
             sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35))),
        dict(until=4.0), "Ex", {},
    ),
    # THE SAME CELL WITH A BLOCH PHASE ON THE ZERO-EXTENT AXIS — MEEP's own
    # test_refl_angular / test_boundaries_1D shape, where the k that sets the angle of
    # incidence sits on an axis holding ONE CELL. Nothing about the stepping changes
    # (the raw Yee storage matched MEEP at 3.0e-07 throughout); what changes is the
    # READBACK, because MEEP's default slice has zero extent on a one-cell axis and is
    # therefore COLLAPSED — the mean of the two centred samples straddling it, one of
    # which is a lattice image carrying conj(eikna). Selecting a plane instead was
    # exact at k = 0 and short by tan(pi*kx*dx) here: 9.67e-03 on MEEP's own
    # test_reflectance_angular_1_20_6 at resolution 200, HALVING with resolution
    # rather than converging (1.93e-02 / 9.67e-03 / 4.84e-03 at 100 / 200 / 400).
    # kx is on the zero-extent x and ky is zero on the zero-extent y, so the case
    # also pins that a collapsed axis with no phase stays untouched.
    "unit_axes_three_d_bloch": (
        dict(cell_size=mp.Vector3(0, 0, 4), k_point=mp.Vector3(0.616, 0, 0.3),
             force_complex_fields=False,
             sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35))),
        dict(until=4.0), "Ex",
        # A converter that accepted the k and dropped it produces exactly this run.
        {"k_zero": dict(cell_size=mp.Vector3(0, 0, 4), k_point=mp.Vector3(0, 0, 0),
                        force_complex_fields=True,
                        sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35)))},
    ),
    "unit_axes_three_d_structured": (
        dict(cell_size=mp.Vector3(0, 0, 4),
             sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35)),
             geometry=[mp.Block(size=mp.Vector3(mp.inf, mp.inf, 1.0),
                                center=mp.Vector3(0, 0, 0.5), material=mp.Medium(epsilon=4.0))]),
        dict(until=4.0), "Ex",
        {"no_geometry": dict(geometry=[])},
    ),
    "mirror_z_odd": (
        dict(symmetries=[mp.Mirror(mp.Z, phase=-1)], sources=cw(center=mp.Vector3(0, 0, 0))),
        dict(until=1.0), "Ez", {},
    ),
    "mirror_x_odd": (
        dict(cell_size=mp.Vector3(4, 2, 2), symmetries=[mp.Mirror(mp.X, phase=-1)],
             sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, 0))),
        dict(until=1.0), "Ex", {},
    ),
    "mirror_y_even": (
        dict(cell_size=mp.Vector3(2, 4, 2), symmetries=[mp.Mirror(mp.Y)],
             sources=cw(component=mp.Ez, center=mp.Vector3(0, 0, 0))),
        dict(until=1.0), "Ez", {},
    ),
    # A source OFF the plane, in the stored half. MEEP images it implicitly
    # (use_symmetry=false, sources.cpp:487) and this used to be refused outright,
    # which cost perturbation_theory_2d.py, solve-cw.py and antenna_pec_ground_plane.py
    # a lift. `unfolded_pair` is the control that earns the case: the same physics
    # written out longhand, which the lifted folded run must MATCH rather than differ
    # from — so it is asserted as a parity number below, not as a separation.
    # The cell is 6 rather than 4 along x for the same reason `mirror_x_odd`'s is 4
    # rather than 2: a folded axis is terminated with a zero ghost at its far face,
    # so the case has to stop before the field arrives there or it measures the
    # termination instead of the source. At 0.7 the source is 2.3 from that face,
    # and `until=1.0` keeps it 1.3 short (in a 4-wide cell it was 0.3 PAST it, and
    # the case read 4.40e-04 — the far-face residual, not the fold).
    "mirror_x_even_off_plane": (
        dict(cell_size=mp.Vector3(6, 2, 2), symmetries=[mp.Mirror(mp.X)],
             sources=cw(component=mp.Ez, center=mp.Vector3(0.7, 0, 0))),
        dict(until=1.0), "Ez", {},
    ),
    # The same, driving a component the plane makes ODD. On the plane that cancels
    # against its own image and is refused; off the plane it is an ordinary
    # antisymmetric pair, and MEEP runs it.
    "mirror_x_even_off_plane_odd_component": (
        dict(cell_size=mp.Vector3(6, 2, 2), symmetries=[mp.Mirror(mp.X)],
             sources=cw(component=mp.Ex, center=mp.Vector3(0.7, 0, 0))),
        dict(until=1.0), "Ex", {},
    ),
    # A source in the DISCARDED half beside one in the stored half: MEEP deposits the
    # first and silently drops the second, and the fold then regenerates exactly that
    # partner. This is the shape all three corpus scripts are written in — the mirror
    # pair spelled out longhand next to an mp.Mirror — so the lift has to drop it too
    # or the partner is deposited twice.
    "mirror_x_pair_written_longhand": (
        dict(cell_size=mp.Vector3(6, 2, 2), symmetries=[mp.Mirror(mp.X)],
             sources=cw(component=mp.Ez, center=mp.Vector3(0.7, 0, 0))
             + cw(component=mp.Ez, center=mp.Vector3(-0.7, 0, 0))),
        dict(until=1.0), "Ez", {},
    ),
    # --- a FULL-WIDTH source on a folded axis whose far face is LIVE ---------------
    # The corpus folded-live-face family (binary_grating.py, chirped_pulse.py,
    # diffracted_planewave.py, binary_grating_phasemap.py): {PML on X only,
    # Mirror(Y), Y periodic at k = 0, source spanning the whole of Y}. MEEP's
    # `loop_in_chunks` keys its lattice-shift loop on the boundary being Periodic
    # (loop_in_chunks.cpp:393), not on the symmetry, so the far end of the request
    # deposits its edge weight onto the top of the stored window through the
    # ishift = -1 image. Clipping without that image pass left the second-mirror
    # row HALF its current (deposit profile [1, ..., 1, 0.5] against MEEP's
    # uniform 1.0) and put spurious energy on the mirror-odd component — the
    # corpus family read 9.5e-03 … 6.0e-02 whole-volume against windowed-source
    # controls at 1e-06. Three cases because three different rows carry the
    # boundary weight: an EVEN count puts a shift-0 sample ON the plane (0.5 +
    # imaged 0.5), an ODD count puts a shift-1 sample there (the Hz case), and an
    # ODD count's shift-0 component overhangs the window by one rung — MEEP owns
    # that rung as the redundant image slot past the plane (vec.cpp:445-462), this
    # engine serves it by reflection and must DROP the rung rather than clamp it
    # onto the top stored row (the clamp double-counts once the image pass runs).
    "full_width_source_on_folded_live_face_even": (
        dict(cell_size=mp.Vector3(6, 0.32, 0), resolution=50,
             boundary_layers=[mp.PML(1.0, direction=mp.X)],
             symmetries=[mp.Mirror(mp.Y)],
             sources=cw(component=mp.Ez, center=mp.Vector3(-1.8, 0, 0),
                        size=mp.Vector3(0, 0.32, 0))),
        dict(until=4.0), "Ez", {},
    ),
    "full_width_source_on_folded_live_face_odd": (
        dict(cell_size=mp.Vector3(6, 0.30, 0), resolution=50,
             boundary_layers=[mp.PML(1.0, direction=mp.X)],
             symmetries=[mp.Mirror(mp.Y, phase=-1)],
             sources=cw(component=mp.Hz, center=mp.Vector3(-1.8, 0, 0),
                        size=mp.Vector3(0, 0.30, 0))),
        dict(until=4.0), "Hz", {},
    ),
    "full_width_source_on_folded_live_face_odd_shift0": (
        dict(cell_size=mp.Vector3(6, 0.30, 0), resolution=50,
             boundary_layers=[mp.PML(1.0, direction=mp.X)],
             symmetries=[mp.Mirror(mp.Y)],
             sources=cw(component=mp.Ez, center=mp.Vector3(-1.8, 0, 0),
                        size=mp.Vector3(0, 0.30, 0))),
        dict(until=4.0), "Ez", {},
    ),
    # --- an ODD folded PERIODIC axis whose TOP CELL is not vacuum -------------------
    # MEEP's halved chunk owns and steps a shift-0 sample at `big_corner` (doubled
    # n_full + 1 at an odd count — vec.cpp:445-462 `owns`, allocated by
    # vec.cpp:293-296 `update_ntot`), which is ONE HALF-CELL ABOVE the second
    # mirror at doubled n_full. This engine used to leave that cell unstored and
    # serve reads of it by reflecting about the second mirror, which is exact only
    # where the MEDIUM is symmetric about that plane. Nothing makes it so at an odd
    # count: MEEP anchors the absorber at the window top (structure.cpp:226 passes
    # `user_volume.boundary_location(side, d)`, i.e. `loc(Ez, ntot() - 1)`,
    # vec.cpp:692-700), so `pml_x` (structure.cpp:625-628) grades sigma about a wall
    # half a cell above the plane the reflection assumes. Both cases below are the
    # defect, one with the absorber and one without it — the second exists because
    # the absorber is not the mechanism, an asymmetric medium is.
    #
    # Measured whole-volume complex L2 against CPU MEEP's own folded run, with the
    # pre-fix engine restored / with the cell stored — these two cases exactly:
    #   odd_fold_absorber_on_the_folded_axis          1.0097e-02 -> 3.0705e-07
    #   odd_fold_material_step_under_the_window_top   9.9514e-01 -> 3.3826e-07
    #   even_fold_material_step_under_the_window_top  3.5141e-07 -> 3.5141e-07
    # the even control identical to the last digit, before and after. A standalone
    # sweep of the absorber depth read 3.55e-02 / 1.32e-02 / 8.62e-03 at dpml
    # 0.5 / 1.0 / 1.5 — the defect drains with absorber depth because a deeper
    # layer puts less sigma gradient across the half-cell the reflection misplaces.
    "odd_fold_absorber_on_the_folded_axis": (
        dict(cell_size=mp.Vector3(6, 4.1, 0), resolution=10,
             boundary_layers=[mp.PML(1.0)], symmetries=[mp.Mirror(mp.Y)],
             sources=cw(component=mp.Ez, center=mp.Vector3(-2.0, 0, 0),
                        size=mp.Vector3(0, 4.1, 0))),
        dict(until=6.0), "Ez", {},
    ),
    "odd_fold_material_step_under_the_window_top": (
        # ny_full = 41, dx = 0.1: the window is [-2.0, +2.1], the second mirror at
        # +2.05 and `big_corner` at +2.10. The slab's top edge at +2.05 is the
        # medium discontinuity BETWEEN them — the reflection's exact failure — and
        # no absorber touches the folded axis at all.
        dict(cell_size=mp.Vector3(6, 4.1, 0), resolution=10,
             boundary_layers=[mp.PML(1.0, direction=mp.X)],
             symmetries=[mp.Mirror(mp.Y)],
             geometry=[mp.Block(center=mp.Vector3(0, 1.775),
                                size=mp.Vector3(mp.inf, 0.55, mp.inf),
                                material=mp.Medium(epsilon=4.0))],
             sources=cw(component=mp.Ez, center=mp.Vector3(-2.0, 0, 0),
                        size=mp.Vector3(0, 4.1, 0))),
        dict(until=6.0), "Ez", {},
    ),
    "even_fold_material_step_under_the_window_top": (
        # The even control of the case above: ny_full = 40, where `big_corner` IS
        # the second mirror and the reflection was always exact. This one measured
        # 2.62e-07 while its odd sibling read 9.74e-01, which is what makes the
        # defect a parity statement rather than a geometry statement.
        dict(cell_size=mp.Vector3(6, 4.0, 0), resolution=10,
             boundary_layers=[mp.PML(1.0, direction=mp.X)],
             symmetries=[mp.Mirror(mp.Y)],
             geometry=[mp.Block(center=mp.Vector3(0, 1.725),
                                size=mp.Vector3(mp.inf, 0.45, mp.inf),
                                material=mp.Medium(epsilon=4.0))],
             sources=cw(component=mp.Ez, center=mp.Vector3(-2.0, 0, 0),
                        size=mp.Vector3(0, 4.0, 0))),
        dict(until=6.0), "Ez", {},
    ),
    "lorentz": (
        dict(default_material=mp.Medium(epsilon=2.0, E_susceptibilities=[
            mp.LorentzianSusceptibility(frequency=1.1, gamma=0.05, sigma=0.6)])),
        dict(until=3.0), "Ez",
        {"non_dispersive": dict(default_material=mp.Medium(epsilon=2.0))},
    ),
    "drude": (
        dict(default_material=mp.Medium(epsilon=1.5, E_susceptibilities=[
            mp.DrudeSusceptibility(frequency=1.0, gamma=0.2, sigma=0.4)])),
        dict(until=3.0), "Ez",
        {"non_dispersive": dict(default_material=mp.Medium(epsilon=1.5))},
    ),
    "conductivity": (
        dict(default_material=mp.Medium(epsilon=2.25, D_conductivity=0.35)),
        dict(until=3.0), "Ez",
        {"lossless": dict(default_material=mp.Medium(epsilon=2.25))},
    ),
    "anisotropic": (
        dict(default_material=mp.Medium(epsilon_diag=mp.Vector3(2.0, 3.0, 4.0))),
        dict(until=3.0), "Ez",
        # An isotropic lift that took only one diagonal entry lands here.
        {"isotropic_ez": dict(default_material=mp.Medium(epsilon=4.0))},
    ),
    "chi3": (
        dict(cell_size=mp.Vector3(2, 2, 2), default_material=mp.Medium(epsilon=2.0, chi3=1.0),
             sources=cw(center=mp.Vector3(0, 0, 0.05), size=mp.Vector3(1.0, 1.0, 0.0),
                        amplitude=0.5)),
        dict(until=3.0), "Ez",
        {"linear": dict(cell_size=mp.Vector3(2, 2, 2), default_material=mp.Medium(epsilon=2.0),
                        sources=cw(center=mp.Vector3(0, 0, 0.05), size=mp.Vector3(1.0, 1.0, 0.0),
                                   amplitude=0.5))},
    ),
    "chi2": (
        dict(cell_size=mp.Vector3(2, 2, 2), default_material=mp.Medium(epsilon=2.0, chi2=1.0),
             sources=cw(center=mp.Vector3(0, 0, 0.05), size=mp.Vector3(1.0, 1.0, 0.0),
                        amplitude=0.5)),
        dict(until=3.0), "Ez",
        {"linear": dict(cell_size=mp.Vector3(2, 2, 2), default_material=mp.Medium(epsilon=2.0),
                        sources=cw(center=mp.Vector3(0, 0, 0.05), size=mp.Vector3(1.0, 1.0, 0.0),
                                   amplitude=0.5))},
    ),
    "gaussian_source": (
        dict(sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.5), component=mp.Ez,
                                center=POINT)]),
        dict(until=3.0), "Ez", {},
    ),
    "gaussian_source_width_spelling": (
        # MEEP stores width = max(width, 1/fwidth) whichever spelling was used, so
        # this must reproduce the fwidth=0.5 case to the last bit.
        dict(sources=[mp.Source(mp.GaussianSource(frequency=1.0, width=2.0), component=mp.Ez,
                                center=POINT)]),
        dict(until=3.0), "Ez", {},
    ),
    "gaussian_until_after_sources": (
        dict(sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=1.0, cutoff=2.0),
                                component=mp.Ez, center=POINT)]),
        dict(until_after_sources=2.0), "Ez", {},
    ),
    # MEEP's plus/minus-omega superposition (dipole_in_vacuum_cyl_off_axis.py):
    # GaussianSource(-f) beside GaussianSource(+f) beats into a REAL sine-carrier
    # current — sources.cpp stores freq raw (:72-96) and rotates exp(-i*2*pi*f*t)
    # with its sign (:106). `until` reaches past the envelope peak (cutoff*width =
    # 12.5), so the number is about the pulse, not its turn-on tail. The control is
    # the abs()-lift: BOTH carriers at +f, a smooth complete field rotating one way.
    "gaussian_pm_pair": (
        dict(sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.4),
                                component=mp.Ez, center=POINT),
                      mp.Source(mp.GaussianSource(frequency=-1.0, fwidth=0.4),
                                component=mp.Ez, center=POINT)]),
        dict(until=16.0), "Ez",
        {"positive_pair": dict(sources=[
            mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.4),
                      component=mp.Ez, center=POINT),
            mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.4),
                      component=mp.Ez, center=POINT)])},
    ),
    "magnetic_source": (
        dict(sources=cw(component=mp.Hy, center=mp.Vector3(0, 0, -1.0))),
        dict(until=3.0), "Hy", {},
    ),
    "amp_func_sheet": (
        # The source centre is off both transverse axes, so "relative to centre"
        # and "absolute" are different arguments to the profile above.
        dict(sources=cw(center=mp.Vector3(0.2, -0.1, -1.05), size=mp.Vector3(1.6, 1.6, 0.0),
                        amp_func=gaussian_beam_profile)),
        dict(until=3.0), "Ez",
        # A dropped amp_func gives a flat sheet of the same total extent: a
        # complete, smooth, entirely wrong beam.
        {"flat_sheet": dict(sources=cw(center=mp.Vector3(0.2, -0.1, -1.05),
                                       size=mp.Vector3(1.6, 1.6, 0.0)))},
    ),
    # The ARRAY spelling of the same sheet. MEEP does not treat amp_data as a second
    # kind of source: add_volume_source copies the array into two static buffers and
    # calls the ordinary amp_func overload with `amp_file_func` (sources.cpp:394-417),
    # so what this case pins is `amp_interpolation`'s transcription of that function —
    # in particular the half-cell registration inside `map_coordinates`, which
    # `mutate_from_meep`'s `amp_data_map_coordinates_no_half_cell` breaks on purpose.
    "amp_data_sheet": (
        dict(sources=cw(center=mp.Vector3(0.2, -0.1, -1.05), size=mp.Vector3(1.6, 1.6, 0.0),
                        amp_data=amp_sheet_data())),
        dict(until=3.0), "Ez",
        {"flat_sheet": dict(sources=cw(center=mp.Vector3(0.2, -0.1, -1.05),
                                       size=mp.Vector3(1.6, 1.6, 0.0)))},
    ),
    "continuous_ramp": (
        # MEEP's tanh turn-on: width and slowness both reach the waveform, and a
        # converter that dropped either would emit an abrupt CW instead.
        dict(sources=[mp.Source(mp.ContinuousSource(frequency=1.0, width=1.0, slowness=2.0),
                                component=mp.Ez, center=POINT)]),
        dict(until=3.0), "Ez",
        {"no_ramp": dict(sources=cw())},
    ),
    "complex_amplitude": (
        dict(sources=cw(center=POINT, amplitude=0.5 + 0.8j)),
        dict(until=3.0), "Ez",
        {"real_amplitude": dict(sources=cw(center=POINT, amplitude=1.0))},
    ),
    "integrated_source": (
        dict(sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.5, is_integrated=True),
                                component=mp.Ez, center=POINT)]),
        dict(until=3.0), "Ez",
        {"not_integrated": dict(sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.5),
                                                   component=mp.Ez, center=POINT)])},
    ),
    "custom_source": (
        dict(sources=[mp.Source(mp.CustomSource(src_func=ramp, start_time=0.0, end_time=10.0),
                                component=mp.Ez, center=POINT)]),
        dict(until=3.0), "Ez", {},
    ),
    "odd_cell_counts": (
        # 15, 15 and 21 cells: MEEP's origin sits half a cell higher on an odd axis
        # (grid_volume::icenter rounds down to even), and taking -L/2 there is the
        # historical half-cell defect. The source is at no round coordinate either.
        dict(cell_size=mp.Vector3(1.5, 1.5, 2.1),
             sources=cw(center=mp.Vector3(0.13, -0.07, 0.23))),
        dict(until=2.0), "Ez", {},
    ),
    "real_fields": (
        # MEEP's own default storage mode, reached by omitting force_complex_fields.
        dict(force_complex_fields=False),
        dict(until=3.0), "Ez", {},
    ),
    "courant_non_default": (
        # A control must OVERRIDE the case, so it names Courant explicitly: an
        # empty control dict merges to the case itself and compares a run with
        # itself, which passes any bound and proves nothing.
        dict(Courant=0.3), dict(until=3.0), "Ez",
        {"default_courant": dict(Courant=0.5)},
    ),
    "resolution_non_default": (
        dict(resolution=14), dict(until=2.0), "Ez", {},
    ),
    # --- structured cells: MEEP rasterizes and smooths, the lift reads chi1inv back.
    "structured_slab": (
        dict(geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.8),
                                material=mp.Medium(epsilon=6.0))]),
        dict(until=3.0), "Ez",
        {
            # The geometry never installed: a smooth, complete, empty-cell run.
            "no_geometry": dict(geometry=[]),
            # THE REGISTRATION CONTROL. The same slab moved half a cell (dz = 0.05
            # at resolution 10). Both runs are physical, both are smooth, and the
            # only thing that separates them is whether the permittivity was
            # sampled at each component's own Yee point or half a cell away — the
            # exact defect class that cost 4.8e-02 on a flux and 6.6e-02 on a
            # grating before. If the lift's sampling slipped by half a cell, it
            # would match THIS reference instead of the one above.
            "half_cell_shifted": dict(geometry=[
                mp.Block(mp.Vector3(mp.inf, mp.inf, 0.8), center=mp.Vector3(0, 0, 0.05),
                         material=mp.Medium(epsilon=6.0))]),
        },
    ),
    # --- the same slab through MEEP's per-point epsilon routes -------------------
    #
    # These four used to be REFUSED by name, on the argument that a route MEEP
    # evaluates per point cannot be read back per component. The argument was wrong:
    # `_lift_epsilon_structured` reads MEEP's own rasterized chi1inv off sim.fields
    # and never asks what built it, so a route is invisible to it. What replaced the
    # refusal is a check on the RESULT — `_probe_material_callable` before init and
    # `_require_epsilon_only_structure` after it.
    #
    # `half_cell_shifted` is the load-bearing control here, and more so than for a
    # geometry case: these routes have no declared media, so the declared-media SPAN
    # check inside the structured lift is skipped for them (there is no bracket to
    # skip). What still catches a registration slip is `_lattice_indices`' integer
    # pin against `gv.iyee_shift`/`little_corner` — which is what the
    # `half_cell_shifted_sampling` gate is refused BY, by name — plus this control:
    # the same slab moved dz/2, a perfectly physical run that a half-cell-slipped
    # lift would reproduce instead. `_require_bulk_matches_meep` sits behind both as
    # a second witness; its power against a SHIFT is geometry-dependent and measured
    # absent on at least one cell (see its docstring), so it is not quoted here.
    "material_function_slab": (
        dict(material_function=slab_material), dict(until=3.0), "Ez",
        # Spelled out rather than left empty: control overrides MERGE onto the case's,
        # so `dict()` would re-run the case itself and score 0 while distinguishing
        # nothing.
        {
            "no_geometry": dict(material_function=None),
            "half_cell_shifted": dict(material_function=slab_material_shifted),
        },
    ),
    "epsilon_func_slab": (
        # The NUMBER-returning shape of the same route. MEEP tells the two apart with
        # a `.eps` flag it only sets while building the structure, so the converter
        # cannot read the flag and inspects the return value instead — which is what
        # this case exercises that `material_function_slab` does not.
        dict(epsilon_func=slab_epsilon), dict(until=3.0), "Ez",
        {"no_geometry": dict(epsilon_func=None)},
    ),
    "epsilon_array_slab": (
        # An ndarray as the default material — MEEP's `epsilon_input_file` route with
        # the file read already done (python/simulation.py:2015-2024 puts both in the
        # same slot). The declaration names NO medium at all here, so this is the case
        # where `_materials_of` comes back empty and `_representative_medium` supplies
        # vacuum's material model.
        dict(default_material=_slab_array()), dict(until=3.0), "Ez",
        {
            "no_geometry": dict(default_material=mp.Medium()),
            "half_cell_shifted": dict(default_material=_slab_array(shift=SLAB_SHIFT)),
        },
    ),
    "structured_three_media": (
        dict(geometry=[
            mp.Block(mp.Vector3(1.0, mp.inf, mp.inf), center=mp.Vector3(-0.4, 0, 0),
                     material=mp.Medium(epsilon=4.0)),
            mp.Block(mp.Vector3(0.6, mp.inf, mp.inf), center=mp.Vector3(0.5, 0, 0),
                     material=mp.Medium(epsilon=9.0))]),
        dict(until=3.0), "Ez",
        # Collapsing the two blocks into one medium is what a lift that read only
        # the first geometry object would do.
        {"one_medium": dict(geometry=[mp.Block(mp.Vector3(1.0, mp.inf, mp.inf),
                                               center=mp.Vector3(-0.4, 0, 0),
                                               material=mp.Medium(epsilon=4.0))])},
    ),
    "structured_anisotropic": (
        dict(geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.8),
                                material=mp.Medium(epsilon_diag=mp.Vector3(2.0, 3.0, 4.0)))]),
        dict(until=3.0), "Ez",
        # chi1inv is read per component, so an isotropic collapse is visible.
        {"isotropic_ez": dict(geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.8),
                                                 material=mp.Medium(epsilon=4.0))])},
    ),
    "structured_curved_no_averaging": (
        # With eps_averaging=False MEEP point-samples the cylinder and the tensor
        # really is diagonal — the control geometry for the smoothed cases below,
        # and the diagonal-only route through the same structured reader.
        dict(eps_averaging=False,
             geometry=[mp.Cylinder(radius=0.5, height=mp.inf,
                                   material=mp.Medium(epsilon=12.0))]),
        dict(until=3.0), "Ez",
        {"no_geometry": dict(eps_averaging=False, geometry=[])},
    ),
    # CURVED GEOMETRY WITH THE AVERAGING ON — the case that used to be the single
    # largest refusal in MEEP's own corpus (13 of 57 scripts). MEEP's anisotropic
    # averaging tilts chi1inv off the diagonal at every point of the cylinder's
    # surface; the lift reads the full row back and the engine's stencil consumes
    # it. The control is the SAME geometry point-sampled (eps_averaging=False):
    # a lift that quietly dropped the off-diagonal entries — or the averaging —
    # reproduces something close to THAT run, not MEEP's smoothed one.
    "two_d_cylinder_smoothed": (
        dict(cell_size=mp.Vector3(4, 4, 0), boundary_layers=[mp.PML(0.5)],
             sources=cw(component=mp.Ez, center=mp.Vector3(0.05, -1.35, 0)),
             geometry=[mp.Cylinder(radius=0.5, height=mp.inf,
                                   material=mp.Medium(epsilon=12.0))]),
        dict(until=6.0), "Ez",
        {"point_sampled": dict(
            eps_averaging=False,
            cell_size=mp.Vector3(4, 4, 0), boundary_layers=[mp.PML(0.5)],
            sources=cw(component=mp.Ez, center=mp.Vector3(0.05, -1.35, 0)),
            geometry=[mp.Cylinder(radius=0.5, height=mp.inf,
                                  material=mp.Medium(epsilon=12.0))]),
         "no_geometry": dict(
            cell_size=mp.Vector3(4, 4, 0), boundary_layers=[mp.PML(0.5)],
            sources=cw(component=mp.Ez, center=mp.Vector3(0.05, -1.35, 0)),
            geometry=[])},
    ),
    # The same feature reached through a TE polarization: in 2-D a tilted in-plane
    # interface couples Ex and Ey (the chi1inv[Ex][Y] / chi1inv[Ey][X] entries),
    # which a TM run never reads — the polarization the cylinder case cannot see.
    "two_d_rotated_block_smoothed_te": (
        dict(cell_size=mp.Vector3(4, 4, 0), boundary_layers=[mp.PML(0.5)],
             sources=cw(component=mp.Hz, center=mp.Vector3(0.05, -1.35, 0)),
             geometry=[mp.Block(mp.Vector3(1.4, 0.9, mp.inf),
                                e1=mp.Vector3(1, 1, 0).unit(), e2=mp.Vector3(-1, 1, 0).unit(),
                                material=mp.Medium(epsilon=4.0))]),
        dict(until=6.0), "Hz",
        {"point_sampled": dict(
            eps_averaging=False,
            cell_size=mp.Vector3(4, 4, 0), boundary_layers=[mp.PML(0.5)],
            sources=cw(component=mp.Hz, center=mp.Vector3(0.05, -1.35, 0)),
            geometry=[mp.Block(mp.Vector3(1.4, 0.9, mp.inf),
                               e1=mp.Vector3(1, 1, 0).unit(), e2=mp.Vector3(-1, 1, 0).unit(),
                               material=mp.Medium(epsilon=4.0))])},
    ),
    # A uniform ROTATED tensor medium — no geometry at all, so the whole cell is
    # the coupling. The control drops epsilon_offdiag: same diagonal, coupling
    # thrown away, which is the run the old blanket refusal existed to prevent.
    "uniform_epsilon_offdiag": (
        dict(default_material=mp.Medium(epsilon_diag=mp.Vector3(2.0, 2.5, 3.0),
                                        epsilon_offdiag=mp.Vector3(0.35, 0.2, 0.15))),
        dict(until=3.0), "Ex",
        {"diagonal_only": dict(default_material=mp.Medium(
            epsilon_diag=mp.Vector3(2.0, 2.5, 3.0)))},
    ),
    # THE CORPUS CONFIGURATION: a smoothed cylinder under a mirror plane —
    # ring.py, holey-wvg-cavity.py and seven more declare exactly this shape.
    # Under Mirror(Y) the cylinder's chi1inv[Ex][Y]/chi1inv[Ey][X] coupling is
    # ODD across the plane, so this is the folded ODD-parity case end to end:
    # MEEP folds its own run, the lift reads the folded chi1inv off the stored
    # half, and the engine steps it with the mirror ghosts. The tensor stencil's
    # fold is separately pinned at 8.3e-13 (test_tensor_epsilon); this case adds
    # MEEP's folded rasterization and the folded registration on top.
    # k_point=False throughout: the corpus scripts declare no k_point, so MEEP
    # gives them PEC walls — and a METALLIC folded far face is exactly the regime
    # the fold is exact in (the zero ghost IS the wall). Under the periodic
    # default the same case sits on the fold's documented zero-ghost far-face
    # limitation instead (7.2e-04 tracking the face residual, tensor-independent).
    "two_d_cylinder_smoothed_mirror": (
        dict(k_point=False, cell_size=mp.Vector3(4, 4, 0),
             boundary_layers=[mp.PML(0.5)], symmetries=[mp.Mirror(mp.Y)],
             sources=cw(component=mp.Ez, center=mp.Vector3(-1.35, 0, 0)),
             geometry=[mp.Cylinder(radius=0.5, height=mp.inf,
                                   material=mp.Medium(epsilon=12.0))]),
        dict(until=6.0), "Ez",
        {"point_sampled": dict(
            eps_averaging=False,
            k_point=False, cell_size=mp.Vector3(4, 4, 0),
            boundary_layers=[mp.PML(0.5)], symmetries=[mp.Mirror(mp.Y)],
            sources=cw(component=mp.Ez, center=mp.Vector3(-1.35, 0, 0)),
            geometry=[mp.Cylinder(radius=0.5, height=mp.inf,
                                  material=mp.Medium(epsilon=12.0))])},
    ),
    # The uniform tensor under a fold: Mirror(X) with a chi_yz-only coupling —
    # the EVEN parity combination, nonzero ON the plane, which is exactly the
    # row a fold-plane mask would destroy (measured 2.0e-02 in the engine).
    "uniform_epsilon_offdiag_mirror": (
        dict(k_point=False, cell_size=mp.Vector3(2, 4, 2),
             symmetries=[mp.Mirror(mp.X)],
             sources=cw(component=mp.Ez, center=mp.Vector3(0, 0.05, 0.05)),
             default_material=mp.Medium(epsilon_diag=mp.Vector3(2.0, 2.5, 3.0),
                                        epsilon_offdiag=mp.Vector3(0.0, 0.0, 0.15))),
        dict(until=3.0), "Ez",
        {"diagonal_only": dict(
            k_point=False, cell_size=mp.Vector3(2, 4, 2),
            symmetries=[mp.Mirror(mp.X)],
            sources=cw(component=mp.Ez, center=mp.Vector3(0, 0.05, 0.05)),
            default_material=mp.Medium(epsilon_diag=mp.Vector3(2.0, 2.5, 3.0)))},
    ),
    "structured_odd_cells": (
        # 15/15/21 cells and a block on no round coordinate: MEEP's origin sits
        # half a cell higher on an odd axis, so this is where a registration that
        # measures from -L/2 instead of grid.axis_origin comes apart.
        dict(cell_size=mp.Vector3(1.5, 1.5, 2.1),
             geometry=[mp.Block(mp.Vector3(0.53, 0.71, 0.9), center=mp.Vector3(0.1, -0.05, 0.13),
                                material=mp.Medium(epsilon=6.0))],
             sources=cw(center=mp.Vector3(0.13, -0.07, -0.55))),
        dict(until=2.0), "Ez",
        {"half_cell_shifted": dict(
            cell_size=mp.Vector3(1.5, 1.5, 2.1),
            geometry=[mp.Block(mp.Vector3(0.53, 0.71, 0.9), center=mp.Vector3(0.15, -0.05, 0.13),
                               material=mp.Medium(epsilon=6.0))],
            sources=cw(center=mp.Vector3(0.13, -0.07, -0.55)))},
    ),
    "structured_pml": (
        dict(cell_size=mp.Vector3(3, 3, 6), boundary_layers=[mp.PML(1.0)],
             geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.8),
                                material=mp.Medium(epsilon=6.0))],
             sources=cw(center=mp.Vector3(0, 0, -1.55))),
        dict(until=3.0), "Ez", {},
    ),
    "structured_mirror": (
        dict(symmetries=[mp.Mirror(mp.Z, phase=-1)],
             geometry=[mp.Block(mp.Vector3(1, 1, 1), material=mp.Medium(epsilon=4.0))],
             sources=cw(center=mp.Vector3(0, 0, 0))),
        dict(until=1.0), "Ez", {},
    ),
    "structured_bloch": (
        # A folded axis stores half the cell, so the sampled coordinates are the
        # folded half's; a Bloch axis carries a phase across the wrap. Both change
        # where the permittivity has to be read, so both are pinned with geometry.
        dict(cell_size=mp.Vector3(2, 2, 2), k_point=mp.Vector3(0.137, 0, 0),
             force_complex_fields=False,
             geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.7),
                                material=mp.Medium(epsilon=6.0))],
             sources=cw(center=mp.Vector3(0, 0, 0.05))),
        dict(until=3.0), "Ez",
        {"k_zero": dict(cell_size=mp.Vector3(2, 2, 2), k_point=mp.Vector3(0, 0, 0),
                        force_complex_fields=True,
                        geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.7),
                                           material=mp.Medium(epsilon=6.0))],
                        sources=cw(center=mp.Vector3(0, 0, 0.05)))},
    ),
    "structured_shared_lorentz": (
        # Structured in epsilon while every medium shares one dispersive model —
        # the only combination MEEP's chi1inv can supply, since its per-point sigma
        # volumes are not readable.
        dict(default_material=mp.Medium(epsilon=1.0, E_susceptibilities=[
                 mp.LorentzianSusceptibility(frequency=1.3, gamma=0.05, sigma=0.4)]),
             geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.8), material=mp.Medium(
                 epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(frequency=1.3, gamma=0.05, sigma=0.4)]))]),
        dict(until=3.0), "Ez",
        {"non_dispersive": dict(
            default_material=mp.Medium(epsilon=1.0),
            geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.8),
                               material=mp.Medium(epsilon=4.0))])},
    ),
    # MATERIAL GRIDS. A mp.MaterialGrid is not a material this engine understands and
    # never becomes one: MEEP evaluates it into an ordinary medium_struct at every point
    # (meepgeom.cpp:845-864) and interpolates epsilon linearly between medium1 and
    # medium2 (meepgeom.cpp:569-583), so what reaches the stepper is a per-point chi1inv
    # tensor and nothing else — exactly what the structured lift already reads. These
    # cases pin that, one per branch MEEP can take.
    #
    # The controls matter more here than usual. The declared-media SPAN CHECK inside
    # _lift_epsilon_structured is deliberately skipped for a smoothed grid, because
    # MEEP's own voxel average extrapolates u past [0, 1] and leaves the endpoint
    # bracket (get_uproj_w, meepgeom.cpp:1208-1218 — MEEP's own get_epsilon() reports
    # 0.9730..6.3529 on a 1..6 grid at beta=1000). `half_cell_shifted` is what takes over
    # that guard's job: it is the registration slip the span check existed to catch, and
    # it has to be far outside the bar the parity number is inside.
    "material_grid_point_sampled": (
        dict(geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0),
                                material=_ramped_grid(do_averaging=False))]),
        dict(until=3.0), "Ez",
        {
            # The grid never installed: a smooth, complete, empty-cell run.
            "no_grid": dict(geometry=[]),
            # The same ramp moved half a cell (dz = 0.05 at resolution 10).
            "half_cell_shifted": dict(geometry=[
                mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0), center=mp.Vector3(0, 0, 0.05),
                         material=_ramped_grid(do_averaging=False))]),
            # Collapsing the grid to ONE of its endpoint media is what a lift that took
            # the pair as a homogeneous material rather than a bracket would do.
            "endpoint_medium_only": dict(geometry=[
                mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0), material=mp.Medium(epsilon=6.0))]),
        },
    ),
    # do_averaging=True is a DIFFERENT MEEP code path, not a refinement of the one above:
    # it routes through fallback_chi1inv_row's one-dimensional adaptive integral along
    # the voxel diagonal (meepgeom.cpp:1352-1383) instead of get_material_pt's point
    # sample. The point-sampled case is its control.
    "material_grid_smoothed": (
        dict(geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0),
                                material=_ramped_grid(do_averaging=True))]),
        dict(until=3.0), "Ez",
        {
            "point_sampled": dict(geometry=[
                mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0),
                         material=_ramped_grid(do_averaging=False))]),
            "half_cell_shifted": dict(geometry=[
                mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0), center=mp.Vector3(0, 0, 0.05),
                         material=_ramped_grid(do_averaging=True))]),
        },
    ),
    # beta drives tanh_projection (meepgeom.cpp:497-502), the level-set thresholding an
    # adjoint design uses; at beta=1000 the graded ramp becomes a near-binary interface.
    # Its control is the SAME grid unprojected, which is a materially different structure.
    "material_grid_projected": (
        dict(geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0),
                                material=_ramped_grid(do_averaging=True, beta=1000.0, eta=0.5))]),
        dict(until=3.0), "Ez",
        {"unprojected": dict(geometry=[
            mp.Block(mp.Vector3(mp.inf, mp.inf, 2.0), material=_ramped_grid(do_averaging=True))])},
    ),
    # A grid as default_material rather than an object's material: a different route into
    # _materials_of (the DEFAULT slot, which is also the one _lift_material reads its
    # representative medium from) and the shape test_material_grid.py uses.
    "material_grid_default_material": (
        dict(default_material=_ramped_grid(do_averaging=False)),
        dict(until=3.0), "Ez",
        # Both controls have to be spelled out: an EMPTY control dict merges to the case
        # itself and would score 0, passing as a control while distinguishing nothing.
        {
            # The grid dropped entirely, which is what taking mp.Simulation's own
            # vacuum default would give.
            "vacuum_default": dict(default_material=mp.Medium(epsilon=1.0)),
            # The grid collapsed to one endpoint — a lift that read the pair as a
            # homogeneous material rather than as the bracket of an interpolation.
            "endpoint_medium_only": dict(default_material=mp.Medium(epsilon=6.0)),
        },
    ),
}


def relative_l2(candidate, reference):
    a = np.asarray(candidate, dtype=np.complex128).ravel()
    b = np.asarray(reference, dtype=np.complex128).ravel()
    denominator = float(np.linalg.norm(b))
    if denominator == 0.0:
        raise SystemExit("the CPU-MEEP reference is identically zero; the comparison is empty")
    if float(np.linalg.norm(a)) == 0.0:
        raise SystemExit("the lifted run is identically zero; an empty run must not score as exact")
    return float(np.linalg.norm(a - b) / denominator)


class _AbsAmplitude:
    """An EigenmodeData whose profile has lost its phase — the `no_phase` degradation."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def amplitude(self, point, component):
        return complex(abs(self._inner.amplitude(point, component)))


def degrade_lift(sim, mode):
    """Break the LIFT in one specific, plausible way; returns a restore callable.

    Most controls in this table are reference-side overrides — a different, physical
    run the lifted one must be far from. These cannot be: they are mistakes a converter
    makes while the MEEP input is unchanged, so the only way to measure them is to make
    the mistake on purpose.

    ``axis_solve`` is the one that matters for the oblique eigenmode cases. MEEP's
    NO_DIRECTION splits the SHEET NORMAL from the SOLVE DIRECTION (mpb.cpp:875 vs
    :876-884, and a wholly different Newton step at :605-612); collapsing them back is
    the natural one-line "fix", and on a rotated waveguide it converges, returns the
    same frequency, a group velocity within 0.7% and a peak within 1%, while launching
    at the wrong angle. ``no_phase`` strips the in-plane ramp that IS the obliquity
    (mpb.cpp:255).

    For the beam cases, ``hx_is_ey`` writes the INTENDED physics at
    sources.cpp:691/:716 in place of the uninitialized read MEEP actually performs,
    ``flip_sheet_signs`` / ``flip_other_sheet_signs`` invert one of the two
    equivalent-current PAIRS (both spellings exist because each polarization drives
    only one pair — the eigenmode mutation battery established that a single-pair flip
    survives a one-polarization suite), ``reduce_beam_vectors`` applies the
    dimensional reduction that is correct for a source's centre and size and wrong for
    ``beam_x0`` / ``beam_kdir``, and ``drop_beam_phase`` makes the beam amplitude real
    before it is deposited — the wrong way to put a beam on real float32 storage.
    """
    if mode in ("axis_solve", "no_phase"):
        original = sim.get_eigenmode

        def wrapper(frequency, direction, where, band, kpoint, **kwargs):
            if mode == "axis_solve" and direction == mp.NO_DIRECTION:
                direction = mp.X  # the sheet normal, which is not the solve direction
            edata = original(frequency, direction, where, band, kpoint, **kwargs)
            return _AbsAmplitude(edata) if mode == "no_phase" else edata

        sim.get_eigenmode = wrapper
        return lambda: setattr(sim, "get_eigenmode", original)

    if mode == "no_kz_phasefix":
        # MEEP's special_kz_phasefix lives in add_eigenmode_source, not in
        # fields::get_eigenmode, so a lift that calls the solver directly gets the
        # UNFIXED mode and must reapply the turn. Skipping it is the whole defect.
        original = from_meep._eigenmode_special_kz_turn
        from_meep._eigenmode_special_kz_turn = lambda module, sim, component: 1.0
        return lambda: setattr(from_meep, "_eigenmode_special_kz_turn", original)

    if mode == "full_3d_sample_point":
        # The mode is sampled through a 3-D mp.vec whatever the run's dimensionality
        # (EigenmodeData.amplitude, simulation.py:872-874), so a half-integer-z
        # component's own Yee offset would reach eigenmode_amplitude as a real z
        # displacement and pick up exp(i*2*pi*Gk[2]*pz) with Gk[2] = beta. MEEP's own
        # D2 vec never has a z at all. This control keeps the offset.
        original = from_meep._eigenmode_sample_point
        from_meep._eigenmode_sample_point = lambda module, centre, offsets, invariant: (
            module.Vector3(*(centre[axis] + offsets[axis] for axis in range(3))))
        return lambda: setattr(from_meep, "_eigenmode_sample_point", original)

    if mode == "hx_is_ey":
        original = gaussian_beam._gb_hx
        gaussian_beam._gb_hx = lambda gb_Ey: gb_Ey
        return lambda: setattr(gaussian_beam, "_gb_hx", original)

    if mode in ("flip_sheet_signs", "flip_other_sheet_signs"):
        original = from_meep._eigenmode_current_sheets
        pair = (1, 2) if mode == "flip_sheet_signs" else (0, 3)

        def flipped(module, normal):
            sheets = list(original(module, normal))
            for index in pair:
                sheets[index] = (sheets[index][0], sheets[index][1], -sheets[index][2])
            return tuple(sheets)

        from_meep._eigenmode_current_sheets = flipped
        return lambda: setattr(from_meep, "_eigenmode_current_sheets", original)

    if mode == "drop_beam_phase":
        # THE control for a REAL-fields beam, and the only one that is about the
        # projection rather than about the beam. `Re(a*s) = Re(a)Re(s) - Im(a)Im(s)`
        # is what MEEP's step.cpp:307 deposits; `Re(a)*s` — "real storage cannot carry
        # a complex amplitude, so make the amplitude real" — is the one-line fix that
        # looks equivalent, keeps the right envelope, and launches the wrong wavefront.
        original = from_meep.beam_fields

        def real_amplitude(*args, **kwargs):
            return np.real(original(*args, **kwargs)).astype(np.complex128)

        from_meep.beam_fields = real_amplitude
        return lambda: setattr(from_meep, "beam_fields", original)

    if mode in ("conductivity_shared_from_x", "conductivity_shared_from_y"):
        # THE PRE-ANISOTROPIC ENGINE, restored: one sigma volume shared by all three D
        # components, taken from whichever one the converter happened to read. Both
        # spellings are controls because neither is obviously wrong from inside the
        # code — `conductivity[0]` was the previous line, and `[1]` is the same
        # arbitrary pick. Measured on this cell: 2.22e-01 and 2.17e-01 against a parity
        # of 3.95e-07, both complete runs with the full signal.
        axis = 0 if mode.endswith("_x") else 1
        original = from_meep._lift_material

        def shared(driver, medium, **kwargs):
            original(driver, medium, **kwargs)
            driver.set_conductivity(
                float(from_meep._vector3(medium.D_conductivity_diag)[axis]))

        from_meep._lift_material = shared
        return lambda: setattr(from_meep, "_lift_material", original)

    if mode == "b_conductivity_dropped":
        # THE PRE-PHASE-0 ENGINE, restored: the lift installs the electric loss and
        # nothing on the magnetic side, which is exactly what `_lift_material` did while
        # a mp.Medium(B_conductivity=...) was refused at the gate. Nothing about the run
        # says the magnetic loss vanished — it completes, with the full signal and the
        # same step count.
        from meep_gpu import driver as driver_module

        original = driver_module.FdtdDriver.set_b_conductivity

        def nothing(self, b_conductivity, _original=original):
            return _original(self, None)

        driver_module.FdtdDriver.set_b_conductivity = nothing
        return lambda: setattr(driver_module.FdtdDriver, "set_b_conductivity", original)

    if mode == "installed_on_the_d_side":
        # "A conductivity is a conductivity." MEEP's own get_cnd is ONE switch over both
        # sides (meepgeom.cpp:1545-1559), which is precisely what makes reading
        # B_conductivity_diag onto the D components look harmless: the same number, the
        # same recurrence, the same three volumes. It is a different material.
        original = from_meep._lift_material

        def wrong_side(driver, medium, **kwargs):
            original(driver, medium, **kwargs)
            value = from_meep._vector3(medium.B_conductivity_diag)
            if any(entry != 0.0 for entry in value):
                driver.set_b_conductivity(None)
                driver.set_conductivity(value[0])

        from_meep._lift_material = wrong_side
        return lambda: setattr(from_meep, "_lift_material", original)

    if mode == "b_ramp_overwritten":
        # THE COMPOSITION READ AS A PRECEDENCE. MEEP ADDS the absorber's profile onto the
        # material's own conductivity (meepgeom.cpp:1596-1625); "the medium's declaration
        # wins where it has one" is the other natural reading, and it is invisible on
        # every cell whose material is lossless — which is every absorber case that
        # existed before this one.
        from meep_gpu import driver as driver_module

        original = driver_module.FdtdDriver._compose_conductivity_side

        def material_wins(self, components, material, absorber, _original=original):
            if material:
                return material
            return _original(self, components, material, absorber)

        driver_module.FdtdDriver._compose_conductivity_side = material_wins
        return lambda: setattr(
            driver_module.FdtdDriver, "_compose_conductivity_side", original)

    if mode == "conductivity_dropped":
        # THE PRE-READ ENGINE, restored: the epsilon lift reads chi1inv at frequency 0,
        # where monitor.cpp never reaches the conductivity branch, so a structured loss
        # simply vanishes. This is the control that makes the whole item necessary — the
        # run COMPLETES, with the full signal and the same step count, and nothing in it
        # says a lossy cell was stepped lossless. It is the same defect the material
        # grid's damping refusal used to name.
        original = from_meep._read_conductivity_volumes

        def nothing(mp_module, sim, driver, progress_cb=None, _original=original):
            volumes = _original(mp_module, sim, driver, progress_cb)
            return {name: np.zeros_like(volume) for name, volume in volumes.items()}

        from_meep._read_conductivity_volumes = nothing
        return lambda: setattr(from_meep, "_read_conductivity_volumes", original)

    if mode == "absorber_ramp_added_twice":
        # `_read_conductivity_volumes` returns MEEP's OWN conductivity[c][d], which the
        # absorber's ramp has already been added into (meepgeom.cpp:1596-1625). Composing
        # the driver's reconstructed ramp on top of it counts the absorber twice — the
        # exact bug in a naive "install what you read", and one that produces a stable,
        # smooth, MORE absorbing run rather than a failure.
        # Patched on the DRIVER's setter, not on the reader: the reader runs first and
        # `set_conductivity` sets the flag from its own argument afterwards, so a hook
        # that cleared it earlier would be silently overwritten and the control would
        # measure the parity case again. (It did, at exactly the same number, which is
        # what caught it.)
        from meep_gpu import driver as driver_module

        original = driver_module.FdtdDriver.set_conductivity

        def compose_again(self, d_conductivity, absorber_included=False, _original=original):
            return _original(self, d_conductivity, absorber_included=False)

        driver_module.FdtdDriver.set_conductivity = compose_again
        return lambda: setattr(driver_module.FdtdDriver, "set_conductivity", original)

    if mode == "dispersion_installed_anyway":
        # The other direction of the callable-dispersion question: MEEP DROPS a pole a
        # MATERIAL_USER callable returns (add_susceptibilities walks media only,
        # meepgeom.cpp:1816-1830), so installing it here would step a dispersive cell
        # MEEP is stepping non-dispersive. Reading the callable's declaration as if it
        # were a declared medium is the plausible wrong move.
        original = from_meep._lift_material

        def with_dispersion(driver, medium, **kwargs):
            original(driver, medium, **kwargs)
            for term in dispersive_material(mp.Vector3()).E_susceptibilities:
                driver.add_susceptibility(
                    from_meep._driver_susceptibility(term),
                    float(from_meep._vector3(term.sigma_diag)[0]))

        from_meep._lift_material = with_dispersion
        return lambda: setattr(from_meep, "_lift_material", original)

    if mode == "electric_current_only":
        # THE HALF-DEPOSIT. `get_equiv_sources` returns Love's equivalent PAIR — the
        # electric current K = nHat x H and the magnetic current N = -nHat x E
        # (source.py:786-794) — and it is the pair that makes the sheet radiate into
        # one half-space. Keeping only the electric half is the plausible converter
        # bug (one source drives one component), and it launches the beam in BOTH
        # directions at half amplitude on a run that completes with the full signal.
        original = from_meep._meep_equivalent_beam_sources

        def electric_only(module, sim, source):
            electric = (module.Ex, module.Ey, module.Ez)
            return [s for s in original(module, sim, source) if s.component in electric]

        from_meep._meep_equivalent_beam_sources = electric_only
        return lambda: setattr(from_meep, "_meep_equivalent_beam_sources", original)

    if mode == "pec_installed_as_vacuum":
        # THE NAIVE RELAXATION, and the reason the sentinel is carried through rather
        # than "handled": a metal cell's permittivity is unreadable as a dielectric, so
        # substitute the background. The run completes, the signal is full, the step
        # count matches, and only the parity number says the cavity has no walls.
        from meep_gpu import driver as driver_module

        original = driver_module.FdtdDriver.set_epsilon_components

        def vacuum_metal(self, volumes, **kwargs):
            patched = {
                name: np.where(np.asarray(values) < -1e19, 1.0, np.asarray(values))
                for name, values in volumes.items()
            }
            return original(self, patched, **kwargs)

        driver_module.FdtdDriver.set_epsilon_components = vacuum_metal
        return lambda: setattr(
            driver_module.FdtdDriver, "set_epsilon_components", original)

    if mode == "reduce_beam_vectors":
        original = from_meep.beam_fields

        def reduced(offsets, x0, kdir, *args, **kwargs):
            dimensions = args[-1] if args else kwargs["dimensions"]
            if dimensions == 2:
                x0, kdir = tuple(x0[:2]) + (0.0,), tuple(kdir[:2]) + (0.0,)
            return original(offsets, x0, kdir, *args, **kwargs)

        from_meep.beam_fields = reduced
        return lambda: setattr(from_meep, "beam_fields", original)

    if mode == "b_ghosts_filled_before_the_source":
        # THE DEFECT ITSELF, restored verbatim. MEEP's magnetic half is
        # step_db(B_stuff) / step_source(B_stuff) / step_boundaries(B_stuff)
        # (step.cpp:67-72); this engine used to fill the B-side mirror ghosts at the
        # end of step_B instead, one sub-step early, so every ghost carried the image
        # of its owned cell as of BEFORE that cell was driven. The result is a
        # complete, smooth, stable field whose only flaw is one permanently
        # source-stale row per folded axis — nothing about the run announces it.
        from meep_gpu import driver as driver_module

        original_step = driver_module.step_B
        original_fill = driver_module.fill_symmetry_bc_B

        def step_B_then_fill(fields, pml=None):
            original_step(fields, pml)
            if fields.grid.has_symmetry():
                original_fill(fields)

        driver_module.step_B = step_B_then_fill
        driver_module.fill_symmetry_bc_B = lambda fields: None

        def restore_slot():
            driver_module.step_B = original_step
            driver_module.fill_symmetry_bc_B = original_fill

        return restore_slot

    if mode == "absorber_electric_only":
        # THE MOST TEMPTING WRONG ABSORBER: build the D-side ramp and skip the B-side
        # one, because "conductivity" reads as electric. MEEP's FOR_D_AND_B loop
        # (structure.cpp:377-379) with geom_epsilon::has_conductivity returning true for
        # every component (meepgeom.cpp:1570-1573) is what makes both halves exist. The
        # run stays stable and the boundary still absorbs — just not by the same amount.
        from meep_gpu import driver as driver_module

        original = driver_module.FdtdDriver.set_absorber

        def electric_only(self, layers):
            original(self, layers)
            self.fields.set_b_conductivity(None)

        driver_module.FdtdDriver.set_absorber = electric_only
        return lambda: setattr(driver_module.FdtdDriver, "set_absorber", original)

    if mode == "absorber_shared_sigma":
        # ONE sigma volume for all six components instead of six, the compromise the
        # uniform-conductivity path could afford. MEEP point-samples each component at
        # its OWN Yee position (structure.cpp:868-895), which for a GRADED ramp puts
        # neighbouring components half a cell apart in the profile.
        from meep_gpu import driver as driver_module

        original = driver_module.FdtdDriver.set_absorber

        def shared_sigma(self, layers):
            original(self, layers)
            if not self._absorber_sigma:
                return
            one = self._absorber_sigma["Dx"]
            self._absorber_sigma = {name: one for name in self._absorber_sigma}
            self._install_conductivity()

        driver_module.FdtdDriver.set_absorber = shared_sigma
        return lambda: setattr(driver_module.FdtdDriver, "set_absorber", original)

    if mode == "bfast_dropped":
        # The pre-2026-08-04 blind spot, restored exactly: the gate accepts the run,
        # the lift reads `bfast_scaled_k`, and then the k is zeroed before anything
        # steps — so `Grid.bfast_active` is False, no f_bfast is allocated and the
        # ordinary update equations run. Nothing errors and nothing looks odd; the
        # run is a converged NORMAL-INCIDENCE answer to an oblique question.
        from meep_gpu import grid as grid_module

        original = grid_module.Grid._resolve_bfast

        def dropped(self):
            original(self)
            self.bfast_scaled_k = (0.0, 0.0, 0.0)

        grid_module.Grid._resolve_bfast = dropped
        return lambda: setattr(grid_module.Grid, "_resolve_bfast", original)

    raise SystemExit("unknown lift degradation: " + mode)


def reference_for(overrides, run_kwargs, component):
    sim = simulation(**overrides)
    if "until" in run_kwargs:
        sim.run(until=run_kwargs["until"])
    else:
        sim.run(until_after_sources=run_kwargs["until_after_sources"])
    return np.asarray(sim.get_array(component=MEEP_COMPONENTS[component])), sim


def run_case(case):
    overrides, run_kwargs, component, controls = CASES[case]
    sim = simulation(**overrides)
    verdict = gpu_compatibility(sim)
    if not verdict.supported:
        raise SystemExit("case refused by gpu_compatibility: " + "; ".join(verdict.reasons))
    result = run_on_gpu(sim, prefer_gpu=False, **run_kwargs)
    lifted = result.get_array(component)
    # THE SAME sim object now runs on CPU MEEP: the two codes cannot have been
    # handed different inputs by a transcription slip in this table.
    if "until" in run_kwargs:
        sim.run(until=run_kwargs["until"])
    else:
        sim.run(until_after_sources=run_kwargs["until_after_sources"])
    reference = np.asarray(sim.get_array(component=MEEP_COMPONENTS[component]))
    payload = {
        "shape": list(lifted.shape),
        "reference_shape": list(reference.shape),
        "steps": result.steps,
        "meep_time": result.meep_time,
        "meep_round_time": float(sim.round_time()),
        "is_real": bool(sim.fields.is_real),
        "controls": {},
    }
    if list(lifted.shape) != list(reference.shape):
        payload["parity"] = None
        return payload
    payload["parity"] = relative_l2(lifted, reference)
    for name, control_overrides in controls.items():
        if isinstance(control_overrides, str):
            # A LIFT-side degradation rather than a reference-side override: the same
            # MEEP input, converted wrong on purpose, against the same CPU-MEEP answer.
            degraded = simulation(**overrides)
            restore = degrade_lift(degraded, control_overrides)
            try:
                candidate = run_on_gpu(degraded, prefer_gpu=False, **run_kwargs)
            except RuntimeError as exc:  # MEEP itself refuses to produce this run.
                payload["controls"][name] = float("inf")
                payload.setdefault("control_aborts", {})[name] = str(exc)
                continue
            finally:
                restore()
            payload["controls"][name] = relative_l2(
                candidate.get_array(component), reference)
            continue
        merged = dict(overrides)
        merged.update(control_overrides)
        control_reference, _ = reference_for(merged, run_kwargs, component)
        payload["controls"][name] = relative_l2(lifted, control_reference)
    return payload


# --- refusals: (overrides, the substring the reason must contain) -----------------

REFUSALS = {
    # "no_k_point" used to live here, quoting a 1.28e+00 gap. It is now MEEP's
    # default PEC box, lifted as boundaries="metallic" — see the metallic_* parity
    # cases and the metallic_boundaries_are_lifted gate.
    # A 2-D or 1-D cell is no longer refused — see the two_d_* / one_d_* parity cases
    # and the reduced_dimension_* gates. What stays refused on a reduced cell is a fold
    # of the axis it makes invariant, which MEEP steps while printing "WARNING vol
    # mismatch" and double-counting its own loop volume.
    "mirror_on_the_invariant_axis": (
        dict(cell_size=mp.Vector3(2, 2, 0), symmetries=[mp.Mirror(mp.Z)]),
        "translationally invariant"),
    # A mirror on one axis with a Bloch phase on ANOTHER is SERVED (2026-08-08;
    # 2.8e-06 against CPU MEEP in test_driver_vs_meep). Its positive control is the
    # corpus rather than a unit case: five python/tests rows lift through THIS gate
    # with exactly that pairing and reproduce 37 of 37 MEEP-authored assertions
    # (results/bloch_mirror_2026-08-08). What stays refused is a phase on the axis the
    # plane folds, and it is refused HERE rather than by FdtdDriver's constructor
    # halfway through the lift.
    "bloch_on_the_folded_axis": (
        dict(symmetries=[mp.Mirror(mp.X)], k_point=mp.Vector3(0.3, 0, 0),
             force_complex_fields=True),
        "Brillouin-zone interior"),
    # Cylindrical itself LIFTS now (test_cylindrical.py end to end); what stays
    # refused are the per-constraint spellings the mode cannot serve.
    "cylindrical_nonint_m": (
        dict(dimensions=mp.CYLINDRICAL, cell_size=mp.Vector3(2, 0, 4), m=0.5),
        "not an integer"),
    # accurate_fields_near_cylorigin is IMPLEMENTED now (stepping._cylindrical_axis_rows,
    # and the cylindrical_accurate_origin_m2 parity case in test_cylindrical.py). What is
    # refused is running it OUTSIDE its stability bound: without the default treatment's
    # zero rows the near-axis update needs Courant <= 1/(|m| + 0.5) (MEEP's own comment
    # on the branch this selects), and above it the run does not crash — it grows
    # smoothly and hands back something field-shaped. MEEP's own test_pml_cyl sets
    # `sim.Courant = 1/(|m| + 0.6)` by hand for exactly this reason. m=3 at MEEP's
    # default Courant 0.5 is 0.5 against a bound of 0.2857.
    "cylindrical_accurate_origin_unstable_courant": (
        dict(dimensions=mp.CYLINDRICAL, cell_size=mp.Vector3(2, 0, 4), m=3,
             accurate_fields_near_cylorigin=True),
        "1/(|m| + 0.5)"),
    # mp.Absorber itself LIFTS now (the three absorber_* parity cases above); what
    # stays refused are the two spellings meep_gpu.absorber has no term for.
    "absorber_custom_profile": (
        dict(boundary_layers=[mp.Absorber(0.5, pml_profile=lambda u: u * u * u)]),
        "custom pml_profile"),
    "absorber_cylindrical": (
        dict(dimensions=mp.CYLINDRICAL, cell_size=mp.Vector3(2, 0, 4),
             boundary_layers=[mp.Absorber(0.5)]),
        "cylindrical (Dcyl) cell"),
    # A STRUCTURED conductivity on a cylindrical cell. The read that recovers it
    # (`_read_conductivity_volumes`) is measured on the Cartesian lattice only; its Dcyl
    # registration is not, and a graded loss volume installed half a cell out steps a
    # smooth, complete, wrong run. Refused at the GATE until it is measured — a uniform
    # D_conductivity on the same cell is a declaration rather than a read and still
    # lifts, which is what makes this refusal narrow rather than "no loss on Dcyl".
    "cylindrical_structured_conductivity": (
        dict(dimensions=mp.CYLINDRICAL, cell_size=mp.Vector3(2, 0, 4),
             geometry=[mp.Block(mp.Vector3(0.6, mp.inf, 1.0), center=mp.Vector3(0.5, 0, 0),
                                material=mp.Medium(epsilon=6.0, D_conductivity=0.4))]),
        "D_conductivity varies from point to point"),
    "custom_pml_profile": (dict(boundary_layers=[mp.PML(0.5, pml_profile=lambda u: u * u * u)]),
                           "custom pml_profile"),
    "pml_r_asymptotic": (dict(boundary_layers=[mp.PML(0.5, R_asymptotic=1e-8)]), "R_asymptotic"),
    "pml_mean_stretch": (dict(boundary_layers=[mp.PML(0.5, mean_stretch=1.5)]), "mean_stretch"),
    # A source against the absorber is ADMITTED on EVERY grid now.
    # `source_spans_the_pml` and `source_on_the_wrapped_pml_seam` moved from this
    # table to parity CASES when the Cartesian defects were fixed and measured (the
    # f_u deposit mirror and the integrated-dipole withdraw slot; evidence packet
    # §1.5-1.7), and `cylindrical_source_in_pml` followed when the Dcyl sweep
    # measured the same truth table transferring and the mirror landing every
    # broken case at the floor (sources.py deposit-mirror comment). What the sweep
    # found INSTEAD — an integer-r-stencil source reaching the r-max metallic wall,
    # whose wall ring the old clamp folded onto the last interior ring — was
    # refused here as `cylindrical_source_on_r_wall` until the deposition learned
    # MEEP's store-then-zero wall rule (the `metal_high` drop in
    # `sources._build_source_points`); that case is a parity CASE now too,
    # zone_plate.py's own spelling, measured at the interior controls' floor.
    #
    # The fold's two remaining refusals. A source OFF the plane in the stored half is
    # a parity case now (mirror_x_even_off_plane*), and one in the discarded half is
    # DROPPED as MEEP drops it — so what is left is the pair of cases where following
    # MEEP would return a clean, complete, empty or cancelled field.
    #
    # Every source discarded: MEEP deposits none of them (use_symmetry=false,
    # sources.cpp:487) and steps zero fields, returning a peak of exactly 0. Measured
    # on a 10x6 cell: the identical run with the source at +1.1 peaks at 1.9e+00.
    "every_source_in_the_discarded_half": (dict(
        symmetries=[mp.Mirror(mp.X)],
        sources=cw(center=mp.Vector3(-0.7, 0.0, -1.05))),
        "every source"),
    # An odd component ON the plane, where the image cancels it. The driver has always
    # raised on this; the PRE-FLIGHT did not, so it was the same accepted-then-raised
    # contract break the PML check had.
    "odd_component_on_the_mirror_plane": (dict(
        cell_size=mp.Vector3(4, 2, 2), symmetries=[mp.Mirror(mp.X)],
        sources=cw(component=mp.Ex, center=mp.Vector3(0.0, 0.0, 0.0))),
        "parity -1"),
    # A MONITOR in the discarded half, which is the opposite decision from a source
    # there: MEEP loops DFT monitors WITH the symmetry (dft.cpp:231 takes
    # A monitor region in the half a fold discards is SERVED now (the reflected
    # gather, `dft.folded_axis_sites`); what stays refused by name is a region the
    # fold serves NOTHING of — beyond the user window and its one lattice image,
    # where MEEP's loop_in_chunks would find no chunk and the accumulator would
    # report zeros that read as a measurement.
    "monitor_wholly_outside_the_fold": (dict(
        cell_size=mp.Vector3(2, 4, 2), symmetries=[mp.Mirror(mp.Y)],
        sources=cw(component=mp.Ez, center=mp.Vector3(0, 0, 0))),
        "wholly outside what the fold serves"),
    # `pml_fractional_cells` used to live here, refusing mp.PML(0.55) at resolution 10
    # for being 5.5 cells thick. It is a PARITY case now (CASES above): MEEP snaps only
    # the extent, to a half cell, and meep_gpu.pml reproduces that. Deleted rather than
    # re-anchored, as `bloch_on_a_pml_axis_accepted` was — there is no refusal left.
    # A structured cell is lifted (see the structured_* parity cases), and since the
    # off-diagonal chi1inv rows are ingested, so is curved and rotated geometry —
    # folded or not (two_d_cylinder_smoothed / two_d_cylinder_smoothed_mirror /
    # two_d_rotated_block_smoothed_te; the tensor stencil's driven fold-equivalence
    # is 8.3e-13..4.7e-12 across both coefficient parities).
    # A CONDUCTIVITY difference, not a susceptibility one: media differing only in
    # E_susceptibilities LIFT on a sigma-reader MEEP (meep-sigma-reader.patch), so
    # that spelling would stop being a refusal the moment this suite runs in the
    # patched env. Per-point conductivity still has no reader on any MEEP.
    # Media differing in chi3 — and it has to be chi3 (or chi2) now, because the three
    # entries this refusal used to be demonstrated on are all recoverable per point:
    # epsilon_diag and epsilon_offdiag off chi1inv at frequency 0, and
    # D_conductivity_diag off the D row at a nonzero one. MEASURED that chi2/chi3 are
    # not: adding chi2=5, chi3=20 to a block moves chi1inv by exactly 0.000e+00 on the
    # E row AND on the D row, at frequency 0 and at 0.9. MEEP keeps them in their own
    # arrays (meep.hpp:591) and publishes no reader for either, so there is nothing to
    # read back and nothing in chi1inv to separate them from the permittivity.
    "media_differ_in_chi3": (dict(
        default_material=mp.Medium(epsilon=1.0),
        geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 1.0), material=mp.Medium(
            epsilon=4.0, chi3=0.5))]),
        "differ in more than their permittivity"),
    # sim.set_boundary writes a PEC wall straight into fields.boundaries, which
    # MEEP exposes only as an opaque pointer — so an initialized simulation is
    # refused wholesale rather than lifted as the periodic run it may not be.
    "already_initialized": (dict(), "already been initialized"),
    "set_boundary_metallic": (dict(), "already been initialized"),
    # A material_function / epsilon_func / ndarray / epsilon_input_file is NOT refused
    # for being one any more — the four are parity cases now (material_function_slab,
    # epsilon_func_slab, epsilon_array_slab). What is refused is what such a route
    # PUTS IN THE STRUCTURE: the lift reads MEEP's chi1inv, which carries the
    # instantaneous permittivity and nothing else, so a callable returning a medium
    # with a susceptibility, a conductivity, a permeability or a nonlinearity would
    # have that half of the material silently dropped.
    # A callable returning a dispersive medium is refused ONLY when MEEP would actually
    # register the pole. `geom_epsilon::add_susceptibilities` (meepgeom.cpp:1816-1830)
    # discovers poles by walking geometry / extra_materials / default_material behind an
    # `is_medium` test, and a MATERIAL_USER callable is none of those — so the callable
    # alone LIFTS (the material_function_dispersive_dropped parity case; MEEP's own
    # get_epsilon reads 2.25 at frequency 0 and at 0.9). Naming the same medium in
    # extra_materials registers the pole, MEEP steps it per point, and this lift has no
    # per-point evaluator for a callable's sigma. That is the case refused here.
    "material_function_dispersive": (
        dict(material_function=dispersive_material,
             extra_materials=[mp.Medium(epsilon=2.0, E_susceptibilities=[
                 mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])]),
        "MEEP DOES register"),
    # And the cell-level version: a route cannot be mixed with declared media that
    # differ in more than permittivity, because `_require_epsilon_only_structure`
    # reads the whole cell at once and could not attribute the difference.
    "material_function_with_dispersive_geometry": (
        dict(material_function=slab_material,
             geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.5), material=mp.Medium(
                 epsilon=4.0, D_conductivity=0.5))]),
        "AND media that differ in more than their permittivity"),
    # A plain MaterialGrid used to be refused HERE, for being a material kind rather
    # than an mp.Medium. It is a parity case now (material_grid_* in CASES, at the
    # structured floor): _materials_of expands a grid into the two endpoint media MEEP
    # interpolates between, and nothing of the grid survives into the stepper except the
    # per-point chi1inv the structured lift already reads. What replaced the blanket
    # refusal are the three things the endpoint pair genuinely cannot speak for.
    #
    # PER-POINT DAMPING is LIFTED now — see the material_grid_damping PARITY case. It
    # was refused here while the lift read chi1inv only at frequency 0, where MEEP's
    # conductivity branch is never reached, and the run COMPLETED and was wrong:
    # 1.0531e-02 relative at damping=0.3, 1.7316e-02 at 0.5 and 9.1963e-02 at pi (what
    # test_adjoint_solver uses), against 2.9431e-07 at damping=0. What closed it is
    # `_read_conductivity_volumes`, which reads the same D_conductivity_diag array back
    # off the D-ROW chi1inv at a NONZERO frequency, damping included because MEEP put it
    # there. What is still refused is a NEGATIVE damping: u*(1-u)*damping is then a
    # negative conductivity, i.e. gain, which grows the field while every magnitude
    # still looks reasonable.
    "material_grid_negative_damping": (dict(default_material=mp.MaterialGrid(
        mp.Vector3(4, 4, 4), mp.Medium(epsilon=1), mp.Medium(epsilon=4), damping=-0.3)),
        "damping=-0.3"),
    # ENDPOINT DISPERSION, refused as unmeasurable rather than unimplementable. MEEP
    # concatenates the endpoints' terms and interpolates each sigma by (1-u)/u, but the
    # path would not activate on MEEP 1.33.0 single: get_epsilon_point returned
    # eps(0.45) == eps(0) at every point (delta 0.0000) while eps_inf graded correctly,
    # against 5.1130 of movement for the same pole on a plain medium.
    "material_grid_dispersive_endpoint": (dict(default_material=mp.MaterialGrid(
        mp.Vector3(4, 4, 4), mp.Medium(epsilon=1),
        mp.Medium(epsilon=4, E_susceptibilities=[
            mp.LorentzianSusceptibility(frequency=0.7, gamma=0.0, sigma=3.0)]))),
        "E_susceptibilities"),
    # AN ENDPOINT FIELD MEEP ITSELF DISCARDS, and the one that would be silently wrong in
    # the OTHER direction: md->medium is default-constructed (material_data.cpp:47-50)
    # and only E_susceptibilities are merged into it (typemap_utils.cpp:528-547), so
    # chi2/chi3, mu and B_conductivity on an endpoint never reach the structure — while
    # _lift_material would install chi3 from the representative medium and step a
    # nonlinear cell MEEP is stepping linear. Measured with chi3=20 on both endpoints:
    # MEEP's grid run is BIT-IDENTICAL (0.0000e+00) to the same grid with no chi3 at all,
    # while the same claim on a plain medium moves 1.4093e+00.
    "material_grid_endpoint_chi3": (dict(default_material=mp.MaterialGrid(
        mp.Vector3(4, 4, 4), mp.Medium(epsilon=1, chi3=0.4), mp.Medium(epsilon=4, chi3=0.4))),
        "NEVER READS"),
    "rotate4": (dict(symmetries=[mp.Rotate4(mp.Z)]), "Rotate4"),
    "magnetic_material": (dict(default_material=mp.Medium(epsilon=2.0, mu=2.0)), "mu_diag"),
    # A uniform epsilon_offdiag medium LIFTS, folded or not — see the
    # uniform_epsilon_offdiag and uniform_epsilon_offdiag_mirror parity cases.
    # "anisotropic_conductivity" used to live here. A DIAGONAL anisotropic
    # D_conductivity_diag is supported now (the anisotropic_conductivity PARITY case):
    # MEEP is per-component here too (get_cnd, meepgeom.cpp:1545-1559) and
    # Fields._set_conductivity_side always stored sigma per component — only the
    # driver's public spelling was missing. What stays refused is the ROTATED case,
    # which genuinely has no diagonal representation.
    "rotated_conductivity": (dict(default_material=mp.Medium(
        epsilon=2.0, D_conductivity_offdiag=mp.Vector3(0.1, 0.0, 0.0))),
        "rotated conductivity"),
    "noisy_susceptibility": (dict(default_material=mp.Medium(epsilon=2.0, E_susceptibilities=[
        mp.NoisyLorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.5, noise_amp=0.01)])),
        "NoisyLorentzianSusceptibility"),
    "gyrotropic_susceptibility": (dict(default_material=mp.Medium(
        epsilon=2.0, E_susceptibilities=[mp.GyrotropicLorentzianSusceptibility(
            frequency=1.0, gamma=0.1, sigma=0.5, bias=mp.Vector3(0, 0, 1))])),
        "GyrotropicLorentzianSusceptibility"),
    "magnetic_susceptibility": (dict(default_material=mp.Medium(epsilon=2.0, H_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.5)])),
        "H_susceptibilities"),
    # "b_conductivity" used to live here. A mp.Medium(B_conductivity=...) is a PARITY
    # case now (b_conductivity / anisotropic_b_conductivity / b_conductivity_in_an_absorber):
    # MEEP's get_cnd answers for Bx/By/Bz through the same switch it answers Dx/Dy/Dz
    # with (meepgeom.cpp:1545-1559), and Fields._set_conductivity_side has stored the B
    # volumes since mp.Absorber needed them, so only a public way to install a MATERIAL's
    # magnetic loss was missing. What stays refused is the ROTATED tensor, for the same
    # reason it is refused on the D side and with one fewer escape: there is no B-row
    # read to recover a diagonal from.
    "rotated_b_conductivity": (dict(default_material=_rotated_b_conductivity_medium()),
                               "B_conductivity_offdiag="),
    # "eigenmode_no_direction_real_fields" used to live here, and a plain
    # EigenModeSource before that. Neither is refused now: the oblique launch is
    # eigenmode_oblique_planewave / eigenmode_oblique_waveguide, and REAL storage is
    # beam_two_d_tm_real_fields / eigenmode_oblique_waveguide_real_fields. The gate was
    # the injection layer declining to project a phased spatial amplitude onto float32;
    # that layer now transcribes MEEP's own unconditional Re(A) (step.cpp:307), which
    # is measured to BE MEEP's answer rather than a reduction of it.
    #
    # "gaussian_beam_source" used to live here too. mp.GaussianBeamSource is a PARITY
    # case now (beam_two_d_tm / beam_two_d_te / beam_three_d). What stays refused are
    # the three shapes the reconstruction genuinely cannot express.
    #
    # A 2-D beam MEEP itself renders sourceless: with beam_E0 both in and out of plane,
    # its own parity checks (sources.cpp:501-504) reject all four sheets and it steps a
    # run with no source at all, returning a complete, smooth, identically zero field.
    # Measured on CPU MEEP: peak 0.0 across all six components (test_gaussian_beam.py).
    "gaussian_beam_both_polarizations_2d": (dict(
        cell_size=mp.Vector3(6, 6, 0), k_point=False, boundary_layers=[mp.PML(1.0)],
        sources=[mp.GaussianBeamSource(
            mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, -1),
            size=mp.Vector3(3, 0), beam_x0=mp.Vector3(0, 1), beam_kdir=mp.Vector3(0, 1),
            beam_w0=0.8, beam_E0=mp.Vector3(1, 0, 1))]),
        "BOTH an in-plane and an out-of-plane"),
    # A non-surface volume: MEEP takes the sheet normal from normal_direction(where),
    # which is defined only for a dim-1 volume and aborts otherwise (vec.cpp:227-247).
    "gaussian_beam_volume": (dict(sources=[mp.GaussianBeamSource(
        mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, 0, -1.0),
        size=mp.Vector3(1, 1, 1), beam_x0=mp.Vector3(0, 0, 0),
        beam_kdir=mp.Vector3(0, 0, 1), beam_w0=0.8, beam_E0=mp.Vector3(1, 0, 0))]),
        "zero-extent resolved axes"),
    # mp.GaussianBeam2DSource LIFTS now (the gaussian_beam_2d parity case, and MEEP's
    # own test_gaussianbeam assertions driven on this engine). What stays refused are
    # the four shapes its Python synthesis cannot express — each one a place where MEEP
    # itself raises, returns NaN, or reads a coordinate it never wrote.
    "gaussian_beam_2d_plane": (dict(
        cell_size=mp.Vector3(6, 6, 0), k_point=False, boundary_layers=[mp.PML(1.0)],
        sources=[mp.GaussianBeam2DSource(
            mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, -1),
            size=mp.Vector3(3, 3), beam_x0=mp.Vector3(0, 1), beam_kdir=mp.Vector3(0, 1),
            beam_w0=0.8, beam_E0=mp.Vector3(0, 0, 1))]),
        "line source, not a plane"),
    "gaussian_beam_2d_point": (dict(
        cell_size=mp.Vector3(6, 6, 0), k_point=False, boundary_layers=[mp.PML(1.0)],
        sources=[mp.GaussianBeam2DSource(
            mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, -1),
            beam_x0=mp.Vector3(0, 1), beam_kdir=mp.Vector3(0, 1),
            beam_w0=0.8, beam_E0=mp.Vector3(0, 0, 1))]),
        "UnboundLocalError"),
    "gaussian_beam_2d_in_3d": (dict(
        sources=[mp.GaussianBeam2DSource(
            mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, 0, -1.0),
            size=mp.Vector3(2, 2, 0), beam_x0=mp.Vector3(0, 0, 1),
            beam_kdir=mp.Vector3(0, 0, 1), beam_w0=0.8, beam_E0=mp.Vector3(1, 0, 0))]),
        "two-dimensional by construction"),
    # The fold seam: `_check_source_folds` and `_fold_drops_source` read sim.sources,
    # where this source is still ONE undivided declaration, while MEEP deposits the
    # derived (Ez, Hx) pair whose parities about the plane differ per component.
    "gaussian_beam_2d_under_symmetry": (dict(
        cell_size=mp.Vector3(6, 6, 0), k_point=False, boundary_layers=[mp.PML(1.0)],
        symmetries=[mp.Mirror(mp.X)],
        sources=[mp.GaussianBeam2DSource(
            mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, -1),
            size=mp.Vector3(3, 0), beam_x0=mp.Vector3(0, 1), beam_kdir=mp.Vector3(0, 1),
            beam_w0=0.8, beam_E0=mp.Vector3(0, 0, 1))]),
        "mirror symmetry"),
    # --- the PEC sentinel's OWN boundary ------------------------------------------
    # mp.metal LIFTS now (the pec_* parity cases). These two are what the relaxation
    # must NOT swallow with it, and they are the whole reason it is written as a test
    # against MEEP's -1e20 BY VALUE rather than as "a negative permittivity is fine".
    "negative_epsilon": (dict(geometry=[mp.Block(mp.Vector3(1, 1, 1),
                                                 material=mp.Medium(epsilon=-3.0))]),
                         "no stable leapfrog"),
    # mp.perfect_magnetic_conductor is the SAME -1e20 sentinel on the OTHER side of the
    # constitutive relation (mu, not epsilon), and the B-side machinery is not built.
    "perfect_magnetic_conductor": (
        dict(default_material=mp.perfect_magnetic_conductor), "mu_diag"),
    # amp_data on a POINT source. The array itself lifts (amp_data_sheet in CASES);
    # a spatial profile with no extent to be a profile of does not, exactly as an
    # amp_func with no extent does not.
    "amp_data_on_point": (dict(sources=cw()), "zero-size (point) source"),
    "amp_func_on_point": (dict(sources=cw(amp_func=gaussian_beam_profile)),
                          "zero-size (point) source"),
    "amp_func_file": (dict(sources=cw(size=mp.Vector3(1, 1, 0))), "amp_func_file"),
    "no_sources": (dict(sources=[]), "declares no sources"),
    "geometry_center": (dict(geometry_center=mp.Vector3(0.1, 0, 0)), "geometry_center"),
    "courant_too_large": (dict(Courant=0.9), "CFL stability limit"),
    # BFAST on a Cartesian grid is STEPPED now (the bfast_fixed_angle parity case, and
    # test_refl_angular's own 35.7-deg assertion driven end to end). What stays refused
    # is BFAST on a CYLINDRICAL grid, and the reason is upstream rather than here:
    # MEEP's Dcyl branch nulls f_p (R) and f_m (Z) by hand at step_db.cpp:87/:96 while
    # have_p/have_m stay true, which makes the single-operand branch of step_bfast
    # reachable with a nonzero k — and that branch, step_generic.cpp:376, is written
    # `F[i] = k1 * (g1[i+s1] + g1[i])` with the `- F[i]` its seven siblings all carry
    # MISSING, so the bilinear derivative degenerates into a bare two-point sum. On a
    # Cartesian grid the omission is dead (g2 is NULL only when have_m is false, and
    # the guard then forces k1 = 0). Refused at the GATE, and by Grid._resolve_bfast
    # if anything reaches the driver directly.
    "bfast_cylindrical": (
        dict(dimensions=mp.CYLINDRICAL, cell_size=mp.Vector3(2, 0, 4),
             bfast_scaled_k=(0.3, 0, 0)),
        "step_generic.cpp:376"),
    # special_kz was here. It is STEPPED now — the four special_kz_* parity cases and
    # the implicit_special_kz half of the reduced-dimension gate — so the refusal is
    # gone rather than re-pointed: the beta term is folded into the curl
    # (stepping._special_kz_beta_term) in both storage modes.
    # This slot has now been re-pointed TWICE, each time because the thing it refused
    # became supported: first `dft_monitor` (add_flux / add_dft_fields, now migrated),
    # then `near2far_monitor` (now accumulated here and evaluated by MEEP). What is
    # left is the near2far option this engine genuinely does not reproduce — nperiods
    # replicates the near surface over lattice periods when the far field is
    # evaluated, which changes what the STORED data means.
    "near2far_nperiods": (dict(), "replicates the near surface"),
    "already_stepped": (dict(), "already been stepped"),
    # The frequency-sign contract, closed at the GATE so it can never be an
    # accepted-then-throws again: a NEGATIVE carrier lifts (the conjugate carrier,
    # `gaussian_pm_pair` above), zero has no carrier to correct — MEEP's own
    # amplitude factor divides by it (sources.cpp:104) and steps an infinite
    # current without a word — and the driver refuses it, so the gate must too.
    "zero_frequency_source": (
        dict(sources=[mp.Source(mp.GaussianSource(0.0, fwidth=0.4), component=mp.Ez,
                                center=POINT)]),
        "finite and nonzero"),
    # "zero_frequency_monitor" used to live here. A DC monitor MIGRATES now and is
    # measured against MEEP's own accumulator
    # (test_a_dc_dft_monitor_accumulates_what_meeps_own_accumulator_does): update_dft
    # has no branch on omega (dft.cpp:266-269) and the one DC-sensitive line in the
    # file is the automatic-decimation guard, whose RESOLVED value this engine reads
    # off the chunk rather than re-deriving. The SOURCE half above stays refused and
    # is a different fact: a zero-frequency source divides MEEP's own amplitude
    # correction 1/(-2*pi*i*f) (sources.cpp:104).
    # A negative carrier on a SYNTHESIZED source is a different boundary: the MPB
    # solve this lift re-runs (mpb.cpp:839) is measured at positive frequencies
    # only, so it refuses by name instead of handing back an unpinned mode.
    "negative_frequency_eigenmode": (
        dict(sources=[mp.EigenModeSource(mp.GaussianSource(-1.0, fwidth=0.4),
                                         center=mp.Vector3(0, 0, -1.0),
                                         size=mp.Vector3(2, 2, 0))]),
        "positive frequencies only"),
    # THE SHAPE of a synthesized source, the eigenmode half. `gaussian_beam_volume`
    # above is the beam half and was gated from the start; the eigenmode half was
    # not, and the gap was measured on 2026-08-08: a 2-D point EigenModeSource under
    # mp.Mirror(mp.Y) reported supported=True with ZERO reasons, and the lift then
    # raised a bare `RuntimeError: meep: Could not determine normal direction for
    # given grid_volume` out of the `sim.init_sim()` it calls — MEEP's own abort,
    # naming no feature, which is the failure mode `_check_source_folds`'s docstring
    # forbids in its worse form. Both arms of `len(flat) != 1` are pinned because
    # they are different declarations reaching the same undefined plane: a POINT has
    # every resolved axis flat (the shape actually measured), a VOLUME has none.
    "eigenmode_point": (
        dict(sources=[mp.EigenModeSource(mp.GaussianSource(1.0, fwidth=0.4),
                                         center=mp.Vector3(0, 0, -1.0),
                                         size=mp.Vector3(0, 0, 0))]),
        "flat resolved axes"),
    "eigenmode_volume": (
        dict(sources=[mp.EigenModeSource(mp.GaussianSource(1.0, fwidth=0.4),
                                         center=mp.Vector3(0, 0, -1.0),
                                         size=mp.Vector3(1, 1, 1))]),
        "flat resolved axes"),
    # The one shape in that family MEEP really does resolve, and the reason the
    # refusal carries two evidence clauses instead of one: on a special_kz 2-D cell
    # `fields::normal_direction` gives a source with extent in BOTH x and y a Z
    # normal (dft.cpp:817-819) — a sheet normal to the axis a 2-D grid does not
    # carry. Refusing it is a NARROWING, not a transcription of a MEEP refusal, and
    # the message has to say so; the substring asserted is the clause that does.
    "eigenmode_volume_special_kz": (
        dict(cell_size=mp.Vector3(2, 2, 0), k_point=mp.Vector3(0, 0, 0.4),
             kz_2d="complex",
             sources=[mp.EigenModeSource(mp.GaussianSource(1.0, fwidth=0.4),
                                         center=mp.Vector3(0, 0), size=mp.Vector3(1, 1),
                                         eig_parity=mp.ODD_Z)]),
        "MEEP does resolve this one"),
    # The beam synthesis (k = 2*pi*f*n through gaussianbeam::get_fields) shares
    # the same boundary and the same refusal wording.
    "negative_frequency_beam": (
        dict(sources=[mp.GaussianBeamSource(mp.ContinuousSource(frequency=-1.0),
                                            center=mp.Vector3(0, 0, -1.0),
                                            size=mp.Vector3(2, 2, 0),
                                            beam_x0=mp.Vector3(0, 0, 0),
                                            beam_kdir=mp.Vector3(0, 0, 1),
                                            beam_w0=0.5,
                                            beam_E0=mp.Vector3(1, 0, 0))]),
        "positive frequencies only"),
    # `bloch_on_a_pml_axis` used to live here. It is GONE, not repaired: a Bloch phase
    # on an absorbing axis is now STEPPED, because measuring it settled the question
    # the refusal had guessed at. MEEP's use_bloch makes every direction Periodic once
    # any k_point is given and a PML is graded material underneath that wrap, so the
    # axis does repeat; the refused configuration reproduces CPU MEEP at 2.82e-07
    # against 2.63e-07 with the absorber removed. It is covered as a PARITY case
    # (`bloch_on_pml_axis`), not a refusal. MEEP's pw-source.py is this shape.
}


def run_refusal(case):
    overrides, _expected = REFUSALS[case]
    overrides = dict(overrides)
    sim = simulation(**overrides)
    if case == "amp_data_on_point":
        sim.sources[0].amp_data = np.ones((4, 4, 4), dtype=np.complex128)
    if case == "amp_func_file":
        sim.sources[0].amp_func_file = "profile.h5:dataset"
    if case == "near2far_nperiods":
        sim.add_near2far(1.0, 0.2, 3, mp.Near2FarRegion(center=mp.Vector3(0, 0, 1.0),
                                                        size=mp.Vector3(1, 1, 0)),
                         nperiods=2)
    if case == "monitor_wholly_outside_the_fold":
        # y = -9 in a (2, 4, 2) cell folded about y: below the user window and
        # more than one lattice image away, so no symmetry image reaches it —
        # MEEP's loop would create no chunk there either. (y = -1.0, in the
        # discarded half proper, is SERVED now: the fold-gather parity cases.)
        sim.add_near2far(1.0, 0.2, 3, mp.Near2FarRegion(center=mp.Vector3(0, -9.0, 0),
                                                        size=mp.Vector3(1, 0, 1)))
    if case == "already_stepped":
        sim.run(until=0.5)
    if case == "already_initialized":
        sim.init_sim()
    if case == "set_boundary_metallic":
        # MEEP's own antenna_pec_ground_plane_1D.py does exactly this. set_boundary
        # forces init_sim itself, so the refusal is reached through the
        # already-initialized gate — which is the point: the metallic wall is not
        # readable afterwards, so the whole post-init state is untrusted.
        sim.set_boundary(mp.Low, mp.Z, mp.Metallic)
    verdict = gpu_compatibility(sim)
    payload = {"supported": verdict.supported, "reasons": list(verdict.reasons)}
    try:
        driver = lift_simulation(sim, prefer_gpu=False)
    except MeepSimulationNotLiftable as exc:
        payload["raised"] = "MeepSimulationNotLiftable"
        payload["message"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - the test asserts on the type it got.
        payload["raised"] = type(exc).__name__
        payload["message"] = str(exc)
    else:
        driver.close()
        payload["raised"] = None
        payload["message"] = ""
    return payload


# --- the gates a first-line check has to fail before anything reaches them ---------


def run_gate(case):
    """Reach one deep gate by disabling the check that normally shadows it.

    Every gate here guards a wrong answer that LOOKS RIGHT: a permittivity read
    one whole cell over, or half a cell over, or with the off-diagonal thrown
    away, all step to a smooth complete field. They are unreachable while the
    first-line checks work, which is exactly why they need forcing — a gate that
    is never exercised is a gate nobody knows is broken.
    """
    from meep_gpu import from_meep

    cylinder = [mp.Cylinder(radius=0.5, height=mp.inf, material=mp.Medium(epsilon=12.0))]
    slab = [mp.Block(mp.Vector3(mp.inf, mp.inf, 0.8), material=mp.Medium(epsilon=6.0))]
    original = from_meep.component_coordinates

    if case == "pml_below_a_quarter_cell_is_no_layer":
        # MEEP's use_pml scans for a point with x > 0 and returns before allocating
        # sigma when it finds none (structure.cpp found_pml). Below a quarter of a cell
        # the snapped extent is 0 half-cells, so there is no absorber at all — not a
        # thin one. The lift drops such a layer for the same reason it drops one on an
        # invariant axis: installing it would grade nothing but WOULD switch the run
        # into PML field storage, a mode MEEP has no counterpart for here.
        #
        # Reported as a pair, because "no PML" is only the right answer on one side of
        # the threshold and a lift that dropped every thin layer would look identical
        # on the first case alone. At resolution 10, 0.02 length units is 0.2 cells
        # (extent 0) and 0.03 is 0.3 cells (extent 1, MEEP's thinnest real layer).
        report = {"case": case, "raised": None, "message": ""}
        for label, thickness in (("below", 0.02), ("above", 0.03)):
            sim = simulation(boundary_layers=[mp.PML(thickness, direction=mp.Z)])
            driver = lift_simulation(sim, prefer_gpu=False)
            layer = getattr(driver, "pml", None)
            report[label + "_has_pml"] = layer is not None
            report[label + "_faces"] = (
                None if layer is None
                else [[float(side) for side in axis] for axis in layer.thickness_by_face]
            )
            report[label + "_graded_cells"] = (
                0 if layer is None
                else int(np.count_nonzero(layer.get_sigma_profile("z")))
            )
            driver.close()
        return report

    # `source_placement_agrees_with_the_driver` used to live here: an 11-probe
    # gate holding gpu_compatibility and FdtdDriver.add_source to one wall
    # predicate (sources.radial_wall_source_conflict). The predicate is deleted —
    # the deposition now drops a wall-plane ring exactly as MEEP's zero_metal
    # leaves its own wall deposit inert (the `metal_high` transcription in
    # `sources._build_source_points`) — so every probe is admitted by both sides
    # and the agreement is pinned POSITIVELY by the parity cases instead:
    # `cylindrical_source_on_r_wall` (the wall, through the lift, against CPU
    # MEEP) and `cylindrical_source_in_pml` / `source_spans_the_pml` (the
    # absorber admissions the gate also carried).
    if case == "offdiagonal_chi1inv_ingested":
        # The scan that used to refuse a cylinder now INGESTS its off-diagonal
        # rows. This gate pins two facts a field comparison could only infer:
        # that the rows a cylinder actually installs are the in-plane pair and
        # nothing else, and that the lift's slot registration is MEEP's own.
        # The registration pin is the symmetric-row identity: chi1inv[Ex][Y] at
        # Ex slot (i,j,k) and chi1inv[Ey][X] at Ey slot (i,j,k) are rows of the
        # SAME symmetric tensor at the SAME integer node — MEEP samples every
        # off-diagonal entry at `here - shift1` (anisotropic_averaging.cpp:
        # 248-257) — so they are equal entry for entry; a half-cell misreading
        # of the registration breaks the identity on every point of the curved
        # interface, where the coefficient varies cell to cell.
        sim = simulation(cell_size=mp.Vector3(4, 4, 0), k_point=mp.Vector3(),
                         sources=cw(component=mp.Ez, center=mp.Vector3(0.05, -1.0, 0)),
                         geometry=cylinder)
        driver = lift_simulation(sim, prefer_gpu=False)
        report = {"case": case, "raised": None, "message": ""}
        rows = {name: driver.fields.chi1inv_offdiagonal_for(name)
                for name in ("Ex", "Ey", "Ez")}
        report["nonzero_rows"] = sorted(
            f"{row}:{partner}" for row, partners in rows.items() for partner in partners
        )
        xy = np.asarray(rows["Ex"].get("Ey", np.zeros(driver.shape)))
        yx = np.asarray(rows["Ey"].get("Ex", np.zeros(driver.shape)))
        report["max_offdiag"] = float(max(np.abs(xy).max(), np.abs(yx).max()))
        report["row_symmetry_max_diff"] = float(np.abs(xy - yx).max())
        driver.close()
        return report
    elif case == "half_cell_shifted_sampling":
        # Half a cell puts every sample on the WRONG PARITY of MEEP's integer
        # lattice, where grid_volume::index silently reads a neighbour.
        def shifted(grid, component):
            return tuple(axis + 0.5 * grid.dx for axis in original(grid, component))
        from_meep.component_coordinates = shifted
        sim = simulation(geometry=slab)
    elif case == "whole_cell_shifted_sampling":
        # A WHOLE cell keeps the parity legal, so the parity pin passes and the
        # lattice lookup succeeds: this is the shift that would otherwise install a
        # structure displaced by one cell and step it without a word.
        def shifted(grid, component):
            return tuple(axis + grid.dx for axis in original(grid, component))
        from_meep.component_coordinates = shifted
        sim = simulation(geometry=slab)
    elif case == "offlattice_sampling":
        # Three tenths of a cell is not a lattice point at EITHER parity, so the
        # registration pin's integer comparison never gets a meaningful value to
        # compare. This is the one the drift check exists for, and the reason it
        # raises instead of letting round_vec snap the coordinate to a neighbour.
        def shifted(grid, component):
            return tuple(axis + 0.3 * grid.dx for axis in original(grid, component))
        from_meep.component_coordinates = shifted
        sim = simulation(geometry=slab)
    elif case == "wrong_material_installed":
        # Not a registration error at all: the coordinates are right and the VALUES
        # are not MEEP's. The registration pin compares integers and cannot see
        # this; the cross-check against MEEP's own cell-centred diagnostic is the
        # only thing in the lift that ever compares a permittivity to MEEP's. It is
        # called directly because a correct lift never reaches it with bad values.
        sim = simulation(geometry=slab)
        driver = lift_simulation(sim, prefer_gpu=False)
        payload = {"case": case}
        try:
            from_meep._require_bulk_matches_meep(
                sim, driver,
                {name: np.full(driver.shape, 2.0) for name in ("Ex", "Ey", "Ez")},
            )
        except Exception as exc:  # noqa: BLE001 - the test asserts on the type it got.
            payload["raised"] = type(exc).__name__
            payload["message"] = str(exc)
        else:
            payload["raised"] = None
            payload["message"] = ""
        driver.close()
        return payload
    elif case == "boundaries_are_lifted_from_k_point":
        # No stub at all: this gate reports what the converter DECIDED, so the
        # boundary condition is pinned as a declaration rather than inferred from a
        # field comparison that a coincidence could satisfy.
        report = {"case": case, "raised": None, "message": "", "lifts": {}}
        for label, k_point in (("metallic", False), ("periodic_k0", mp.Vector3(0, 0, 0)),
                               ("periodic_bloch", mp.Vector3(0.2, 0, 0))):
            sim = simulation(k_point=k_point, force_complex_fields=True)
            verdict = gpu_compatibility(sim)
            driver = lift_simulation(sim, prefer_gpu=False)
            report["lifts"][label] = {
                "supported": verdict.supported,
                "reasons": list(verdict.reasons),
                "boundaries": [list(pair) for pair in driver.boundaries],
                "metallic_axes": [bool(driver.grid.is_metallic(a)) for a in range(3)],
                "wraps": [bool(driver.grid.axis_wraps(a)) for a in range(3)],
                "k_point": list(driver.k_point),
                "meep_shape": list(np.asarray(sim.get_epsilon()).shape),
                "driver_shape": list(driver.shape),
            }
            driver.close()
        return report
    elif case == "reduced_dimensions_are_lifted":
        # Like the boundary gate above: this reports what the converter DECIDED about
        # every reduced spelling, so the dimensionality, the invariant axis, the
        # boundary it gave that axis, the source scaling and MEEP's array rank are all
        # pinned as declarations rather than inferred from a field comparison that a
        # coincidence could satisfy. The spellings differ from each other in exactly
        # the ways MEEP's own _infer_dimensions distinguishes.
        report = {"case": case, "raised": None, "message": "", "lifts": {}}
        spellings = (
            ("z_zero", dict(cell_size=mp.Vector3(2, 3, 0))),
            ("declared_2d", dict(dimensions=2, cell_size=mp.Vector3(2, 3, 4))),
            ("declared_1d", dict(dimensions=1, cell_size=mp.Vector3(0, 0, 4),
                                 sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35)))),
            # NOT reduced: MEEP's _infer_dimensions collapses only on cell_size.z, and
            # a nonzero k_point.z blocks even that. Both build 3-D runs with one-cell
            # axes, and MEEP says so ("Working in 3D dimensions").
            # k_point=False (no k_point at all) is what leaves MEEP's default Metallic
            # walls in place, and it is the point of this spelling: the one-cell x and y
            # must STILL come back periodic, because fields::fields overrides a unit
            # direction to Periodic whatever the boundary default says, while z — the
            # only axis with real extent — carries the PEC box. With the helper's usual
            # k_point=Vector3(0,0,0) every direction is periodic and the case pins nothing.
            ("x_and_y_zero_three_d", dict(cell_size=mp.Vector3(0, 0, 4), k_point=False,
                                          sources=cw(component=mp.Ex,
                                                     center=mp.Vector3(0, 0, -0.35)))),
            # kz_2d="3d" is what makes the k_point.z block the collapse, and it is NOT
            # the default. MEEP's constructor reads kz_2d ("complex" by default) the
            # moment cell_size.z == 0 and k_point.z != 0, and "complex" sets
            # special_kz=True, which puts `special_kz` INTO use_2d's disjunction and
            # collapses the cell after all ("Working in 2D dimensions", a 2-D epsilon
            # array). Only "3d" leaves special_kz False and builds the 3-D run this
            # spelling is here to pin. The default spelling — a 2-D grid carrying
            # that kz as `beta` — is pinned below, in implicit_special_kz.
            ("z_zero_with_kz", dict(cell_size=mp.Vector3(2, 3, 0),
                                    k_point=mp.Vector3(0, 0, 0.25),
                                    kz_2d="3d",
                                    force_complex_fields=True)),
        )
        for label, overrides in spellings:
            sim = simulation(**overrides)
            verdict = gpu_compatibility(sim)
            driver = lift_simulation(sim, prefer_gpu=False)
            report["lifts"][label] = {
                "supported": verdict.supported,
                "reasons": list(verdict.reasons),
                "dimensions": int(driver.grid.dimensions),
                "invariant": [bool(driver.grid.is_invariant(a)) for a in range(3)],
                "metallic_axes": [bool(driver.grid.is_metallic(a)) for a in range(3)],
                "wraps": [bool(driver.grid.axis_wraps(a)) for a in range(3)],
                "nosize": [bool(driver.grid.nosize_direction(a)) for a in range(3)],
                "k_point": list(driver.k_point),
                "driver_shape": list(driver.shape),
                "meep_shape": list(np.asarray(sim.get_epsilon()).shape),
                "meep_array_axes": list(driver.fields.meep_array_axes()),
                "courant_limit_ok": float(sim.Courant) <= 1.0,
            }
            driver.close()
        # dimensions=0 is the spelling that does NOT reduce, because MEEP cannot build
        # it either: _create_grid_volume routes it to vol1d alongside 1, but py_v3_to_vec
        # then rejects it inside init_sim. Verified against this MEEP install — the
        # script raises ValueError("Invalid dimensions in Volume: 0") before it steps.
        # So it is pinned as a REFUSAL; lifting it would accept what the oracle rejects.
        zero_d = simulation(dimensions=0, cell_size=mp.Vector3(0, 0, 4),
                            sources=cw(component=mp.Ex, center=mp.Vector3(0, 0, -0.35)))
        zero_verdict = gpu_compatibility(zero_d)
        report["zero_d"] = {
            "supported": zero_verdict.supported,
            "reasons": list(zero_verdict.reasons),
        }
        try:
            lift_simulation(zero_d, prefer_gpu=False).close()
        except Exception as exc:  # noqa: BLE001 - the test asserts on the type it got.
            report["zero_d"]["raised"] = type(exc).__name__
        # The SAME cell as z_zero_with_kz, minus kz_2d="3d". MEEP's default turns it
        # into special_kz — a 2-D grid carrying the out-of-plane kz as a PHASE, which
        # is a different physics mode, not a 3-D run. It lifts as a 2-D grid whose
        # `beta` holds that kz and whose `k_point` keeps a hard zero on z, which is
        # exactly MEEP's own split (fields constructor beta slot vs
        # use_bloch(Vector3(kx, ky)), python/simulation.py:2478-2504). Both halves are
        # pinned here: taking the kz as a Bloch component instead would phase a wrap
        # that does not exist, and dropping it would step an ordinary 2-D run.
        implicit_kz = simulation(cell_size=mp.Vector3(2, 3, 0),
                                 k_point=mp.Vector3(0, 0, 0.25),
                                 force_complex_fields=True)
        implicit_verdict = gpu_compatibility(implicit_kz)
        implicit_driver = lift_simulation(implicit_kz, prefer_gpu=False)
        report["implicit_special_kz"] = {
            "meep_special_kz": bool(implicit_kz.special_kz),
            "supported": implicit_verdict.supported,
            "reasons": list(implicit_verdict.reasons),
            "dimensions": int(implicit_driver.grid.dimensions),
            "beta": float(implicit_driver.grid.beta),
            "k_point": list(implicit_driver.k_point),
        }
        implicit_driver.close()
        return report
    elif case == "monitors_are_migrated":
        # A DFT monitor already attached to the mp.Simulation used to be a refusal.
        # It is now rebuilt on the driver, so the two things to prove are that it
        # ACCUMULATED (an un-migrated monitor is correctly shaped and all zero) and
        # that it accumulated exactly what the hand-added monitor the old refusal told
        # people to write would have.
        #
        # PML on z, and the plane deliberately NOT equidistant from the source through
        # the periodic wrap: with the cell wrapping, a plane the same distance forward
        # and backward sees the counter-propagating halves cancel to EXACTLY zero, a
        # correct answer indistinguishable from a monitor that never ran. Measured on
        # a (2,2,4) cell with the source at z=-1: plane at z=+1 (2.0 each way) gives
        # 0.0, and moving it to 0.7 / 0.3 / -0.4 gives 1.5e-2 / 3.7e-2 / 7.6e-2.
        fcen, df, nfreq = 1.0, 0.4, 5
        sim = mp.Simulation(
            cell_size=mp.Vector3(2.0, 2.0, 6.0), resolution=12.0, k_point=mp.Vector3(),
            force_complex_fields=True,
            boundary_layers=[mp.PML(1.0, direction=mp.Z)],
            sources=[mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                               component=mp.Ez, center=mp.Vector3(0, 0, -1.5),
                               size=mp.Vector3(1.0, 1.0, 0))],
        )
        flux = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
            center=mp.Vector3(0, 0, 1.5), size=mp.Vector3(1.0, 1.0, 0)))
        dft = sim.add_dft_fields([mp.Ez], fcen, df, 3, where=mp.Volume(
            center=mp.Vector3(0, 0, 1.0), size=mp.Vector3(1.0, 1.0, 0)))
        report = {"case": case, "supported": bool(gpu_compatibility(sim).supported),
                  "reasons": list(gpu_compatibility(sim).reasons)}
        driver = lift_simulation(sim, prefer_gpu=False)
        migrated = driver.migrated_monitors[id(flux)]
        # The spelling the refusal used to instruct, on the same driver and same run.
        # The migration now carries MEEP's RESOLVED automatic decimation off the live
        # chunk (`_resolved_decimation`; MEEP's get_fwidth-based rule resolves 8 where
        # the nominal-fwidth re-derivation gives 9), so the hand-added control uses
        # the same factor — the comparison is the specification transfer, not two
        # different automatic rules.
        manual = driver.add_flux_monitor(fcen=fcen, df=df, nfreq=nfreq,
                                         center=(0.0, 0.0, 1.5), size=(1.0, 1.0, 0.0),
                                         decimation_factor=int(migrated.decimation_factor))
        driver.run(until=25.0)
        migrated_flux = [float(v) for v in np.asarray(migrated.get_flux_spectrum())]
        report["flux"] = migrated_flux
        report["flux_matches_manual"] = bool(np.array_equal(
            np.asarray(migrated.get_flux_spectrum()), np.asarray(manual.get_flux_spectrum())))
        report["flux_is_all_zero"] = not any(v != 0.0 for v in migrated_flux)
        dft_array = np.asarray(driver.migrated_monitors[id(dft)].get_dft("Ez", 0))
        report["dft_max_abs"] = float(np.abs(dft_array).max())
        report["dft_shape"] = list(dft_array.shape)
        report["n_migrated"] = len(driver.migrated_monitors)
        report["frequencies"] = [float(f) for f in np.asarray(migrated.frequencies)]
        report["meep_frequencies"] = [float(f) for f in flux.freq]
        # The decimation MEEP was asked for lives in args, not on the object; a wrong
        # read shifts the spectrum by ~1.4e-06 rather than raising.
        report["decimation_readback"] = {}
        for asked in (0, 1, 3, 7):
            probe = mp.Simulation(cell_size=mp.Vector3(2.0, 2.0, 2.0), resolution=10.0,
                                  k_point=mp.Vector3(), force_complex_fields=True,
                                  sources=[])
            pf = probe.add_flux(fcen, df, nfreq, mp.FluxRegion(
                center=mp.Vector3(), size=mp.Vector3(1.0, 1.0, 0)), decimation_factor=asked)
            pd = probe.add_dft_fields([mp.Ez], fcen, df, 3, where=mp.Volume(
                center=mp.Vector3(), size=mp.Vector3(1.0, 1.0, 0)), decimation_factor=asked)
            # add_mode_monitor builds a DftFlux TOO, with yee_grid inserted before the
            # decimation, so the same class needs a different positional read. Both
            # spellings of yee_grid are probed because the wrong index lands on it.
            pm = probe.add_mode_monitor(fcen, df, nfreq, mp.ModeRegion(
                center=mp.Vector3(), size=mp.Vector3(0, 1.0, 0)), decimation_factor=asked)
            pmy = probe.add_mode_monitor(fcen, df, nfreq, mp.ModeRegion(
                center=mp.Vector3(), size=mp.Vector3(0, 1.0, 0)), yee_grid=True,
                decimation_factor=asked)
            report["decimation_readback"][str(asked)] = [
                from_meep._monitor_decimation(pf), from_meep._monitor_decimation(pd),
                from_meep._monitor_decimation(pm), from_meep._monitor_decimation(pmy)]
        driver.close()
        return report
    elif case == "dc_dft_monitor":
        # A DFT accumulator at frequency 0 — a running time integral of the field.
        # MEEP needs no branch for it and neither does this engine: `update_dft` builds
        # `dft_phase[i] = polar(1.0, omega[i]*time) * scale` with no test on omega
        # (dft.cpp:266-269), and at omega = 0 that factor is exactly 1. The ONE
        # DC-sensitive line in dft.cpp is the automatic-decimation guard
        # (dft.cpp:207-210), which requires `freq_max > 0` and falls through to a
        # factor of 1 otherwise — which is why BOTH shapes are measured here: DC alone
        # (freq_max == 0, factor 1) and DC mixed with a real frequency (factor > 1).
        report = {"case": case, "bins": {}, "decimation": {}, "supported": {}}
        for label, freqs in (("dc_only", [0.0]), ("mixed", [0.0, 0.8])):
            def build(freqs=freqs):
                sim = mp.Simulation(
                    cell_size=mp.Vector3(3, 3, 0), resolution=10,
                    boundary_layers=[mp.PML(0.6)], force_complex_fields=True,
                    default_material=mp.Medium(epsilon=2.0),
                    sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4),
                                       component=mp.Ez, center=mp.Vector3())])
                return sim, sim.add_dft_fields([mp.Ez], freqs, center=mp.Vector3(),
                                               size=mp.Vector3(1.5, 1.5, 0))

            sim, monitor = build()
            verdict = gpu_compatibility(sim)
            report["supported"][label] = [bool(verdict.supported), list(verdict.reasons)]
            until = (400 - 0.5) * (0.5 / 10)
            result = run_on_gpu(sim, until=until, prefer_gpu=False)
            reference, reference_monitor = build()
            reference.run(until=until)
            report["decimation"][label] = int(from_meep._resolved_decimation(monitor))
            for index, frequency in enumerate(freqs):
                ours = np.asarray(result.get_dft_array(monitor, "Ez", index),
                                  dtype=np.complex128)
                theirs = np.asarray(
                    reference.get_dft_array(reference_monitor, mp.Ez, index),
                    dtype=np.complex128)
                report["bins"][label + ":" + str(frequency)] = [
                    float(np.linalg.norm(theirs)),
                    float(np.linalg.norm(ours - theirs)),
                    list(ours.shape), list(theirs.shape),
                ]
        return report
    elif case == "near2far_is_migrated":
        # The near surface accumulates here (per-timestep, over the grid); the
        # Green's-function far-field evaluation stays MEEP's. So the test that matters
        # is the FAR FIELD, evaluated by MEEP from this engine's data.
        #
        # Run with AND without a PML. That is not redundant: a PML splits the cell into
        # 27 structure chunks, the surface then spans 6 dft chunks per component, and
        # get_dft_data returns their concatenation instead of one contiguous box. The
        # two arrangements have the SAME total size, so only a value comparison
        # separates them — the no-PML case passed at 9.6e-08 while the PML case was
        # 1.27e+00 wrong.
        fcen, df, nfreq = 1.0, 0.3, 3
        far_points = [mp.Vector3(2.5, 0.4, 1.2), mp.Vector3(-1.8, 1.1, 3.0),
                      mp.Vector3(0.3, -2.2, 4.0)]

        def build(pml, decimation):
            sim = mp.Simulation(
                cell_size=mp.Vector3(2.0, 2.0, 4.0), resolution=12.0, k_point=mp.Vector3(),
                force_complex_fields=True,
                boundary_layers=[mp.PML(0.5)] if pml else [],
                sources=[mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                                   component=mp.Ez, center=mp.Vector3(0.05, -0.05, -0.5))],
            )
            monitor = sim.add_near2far(fcen, df, nfreq, mp.Near2FarRegion(
                center=mp.Vector3(0.0, 0.0, 0.6), size=mp.Vector3(1.0, 0.8, 0)),
                decimation_factor=decimation)
            return sim, monitor

        from meep.simulation import NearToFarData
        # The far-field GRID readback (get_farfields), exercised beside the point
        # readback because it walks a different MEEP code path (_get_farfields_array).
        far_grid = dict(center=mp.Vector3(0.2, -0.1, 3.4), size=mp.Vector3(1.6, 1.2, 0),
                        resolution=4.0)
        report = {"case": case, "runs": {}}
        for pml in (True, False):
            for decimation in (0, 1):
                label = f"{'pml' if pml else 'nopml'}_dec{decimation}"
                sim, monitor = build(pml, decimation)
                verdict = gpu_compatibility(sim)
                driver = lift_simulation(sim, prefer_gpu=False)
                migration = driver.migrated_monitors[id(monitor)]
                resolved = migration.entries[0][2].decimation_factor
                driver.run(until=18.0)
                # A separately-run CPU MEEP simulation supplies both the chunk layout
                # and the reference; the layout is read off the live object, never
                # reconstructed.
                cpu, cpu_monitor = build(pml, decimation)
                cpu.run(until=18.0)
                ours = migration.packed(mp, cpu_monitor)
                driver.close()
                theirs = np.asarray(cpu.get_near2far_data(cpu_monitor).F)
                # NEGATIVE CONTROL: the same values in a deliberately wrong order.
                # Each chunk's segment is dft[Nfreq*site + freq] (dft.cpp update_dft);
                # transposing every chunk to freq-major keeps the total size, the set
                # of values and even each chunk's boundaries identical, so only a
                # comparison sensitive to the ORDER can tell them apart.
                segments, cursor, chunk = [], 0, cpu_monitor.swigobj.F
                while chunk is not None:
                    count = int(chunk.N) * nfreq
                    segments.append(ours[cursor:cursor + count]
                                    .reshape(int(chunk.N), nfreq).T.ravel())
                    cursor += count
                    chunk = chunk.next_in_dft
                transposed = np.concatenate(segments)
                assert cursor == ours.size
                loaded, loaded_monitor = build(pml, decimation)
                loaded.init_sim()
                loaded.load_near2far_data(loaded_monitor, NearToFarData(F=ours))
                our_far = np.asarray([loaded.get_farfield(loaded_monitor, p)
                                      for p in far_points]).ravel()
                our_grid = loaded.get_farfields(loaded_monitor, **far_grid)
                meep_far = np.asarray([cpu.get_farfield(cpu_monitor, p)
                                       for p in far_points]).ravel()
                meep_grid = cpu.get_farfields(cpu_monitor, **far_grid)
                grid_ours = np.concatenate([np.asarray(our_grid[k]).ravel()
                                            for k in sorted(our_grid)])
                grid_meep = np.concatenate([np.asarray(meep_grid[k]).ravel()
                                            for k in sorted(meep_grid)])
                # Re-load the same simulation with the transposed pack; a second load
                # overwrites the chunk data, so this needs no fourth simulation.
                loaded.load_near2far_data(loaded_monitor, NearToFarData(F=transposed))
                bad_far = np.asarray([loaded.get_farfield(loaded_monitor, p)
                                      for p in far_points]).ravel()
                report["runs"][label] = {
                    "supported": bool(verdict.supported),
                    "reasons": list(verdict.reasons),
                    "chunks": len(migration.entries),
                    "resolved_decimation": int(resolved),
                    "meep_resolved": int(from_meep._resolved_decimation(cpu_monitor)),
                    "stored_size": int(ours.size),
                    "meep_stored_size": int(theirs.size),
                    "stored_error": float(np.linalg.norm(ours - theirs)
                                          / np.linalg.norm(theirs)),
                    "far_error": float(np.linalg.norm(our_far - meep_far)
                                       / np.linalg.norm(meep_far)),
                    "farfields_error": float(np.linalg.norm(grid_ours - grid_meep)
                                             / np.linalg.norm(grid_meep)),
                    "transposed_stored_error": float(np.linalg.norm(transposed - theirs)
                                                     / np.linalg.norm(theirs)),
                    "transposed_far_error": float(np.linalg.norm(bad_far - meep_far)
                                                  / np.linalg.norm(meep_far)),
                    "far_is_all_zero": bool(np.abs(our_far).max() == 0.0),
                }
        return report
    elif case == "near2far_cylindrical_is_migrated":
        # The Dcyl half of the case above. Everything the Cartesian one pins is pinned
        # again -- chunk layout, per-chunk memory order, decimation, far field -- plus
        # the two things only a cylindrical run has: the 2*pi*r ring measure on the
        # radial axis and the r = 0 axis clip. The z caps span r = 0 deliberately, so
        # the clip is on the measured path rather than merely implemented.
        fcen, df, nfreq = 1.0, 0.2, 2
        far_points = [mp.Vector3(40.0, 0, 30.0), mp.Vector3(12.0, 0, -55.0),
                      mp.Vector3(60.0, 0, 3.0)]

        def build(resolution, m):
            sim = mp.Simulation(
                cell_size=mp.Vector3(2.0, 0, 3.0), resolution=resolution,
                dimensions=mp.CYLINDRICAL, m=m, force_complex_fields=True,
                boundary_layers=[mp.PML(0.5)],
                sources=[mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                                   component=mp.Er, center=mp.Vector3(0.35, 0, -0.4))],
            )
            monitor = sim.add_near2far(
                fcen, df, nfreq,
                # A closed surface of revolution: two z caps and the r wall. The caps
                # run from the axis outward, so their radial ladder starts below r = 0
                # for the half-integer components and ON the axis for the integer ones.
                mp.Near2FarRegion(center=mp.Vector3(0.6, 0, 0.7),
                                  size=mp.Vector3(1.2, 0, 0), weight=+1),
                mp.Near2FarRegion(center=mp.Vector3(0.6, 0, -0.7),
                                  size=mp.Vector3(1.2, 0, 0), weight=-1),
                mp.Near2FarRegion(center=mp.Vector3(1.2, 0, 0.0),
                                  size=mp.Vector3(0, 0, 1.4), weight=+1),
                decimation_factor=1,
            )
            return sim, monitor

        def loop_weight(index, count, s0, s1, e0, e1):
            # vec.hpp:372-380 IVEC_LOOP_WEIGHT1x, one axis.
            if 1 < index < count - 2:
                return 1.0
            return {0: s0, 1: s1, count - 1: e0, count - 2: e1}.get(index, 1.0)

        def meep_site_weights(chunk):
            """MEEP's own per-site weight for one Dcyl chunk, off the live object.

            dft.cpp:277 folds in `IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV0 + dV1*loop_i2)`,
            and vec.cpp:262-271 fixes the Dcyl loop order as (P, R, Z) — so loop_i1 is
            the absent phi (n = 1, and `vec s0c(gv.dim, 1.0)` leaves its P slot at 1.0),
            loop_i2 is R and loop_i3 is Z. Nothing here is re-derived: dV0, dV1 and the
            four boundary vectors are read from the chunk MEEP built.
            """
            nr = (chunk.ie.r() - chunk._is.r()) // 2 + 1
            nz = (chunk.ie.z() - chunk._is.z()) // 2 + 1
            radial = np.array([
                loop_weight(i, nr, chunk.s0.r(), chunk.s1.r(), chunk.e0.r(), chunk.e1.r())
                * (chunk.dV0 + chunk.dV1 * i) for i in range(nr)])
            axial = np.array([
                loop_weight(k, nz, chunk.s0.z(), chunk.s1.z(), chunk.e0.z(), chunk.e1.z())
                for k in range(nz)])
            return radial[:, None] * axial[None, :]

        def weights_vs_meep(entries, cpu_monitor):
            """Worst per-site weight deviation from MEEP's own, and the sites compared.

            A stored-data comparison confounds the weights with the stepper; this one
            does not. Every chunk MEEP built is matched to the accumulator that answers
            for it, by component and containment, exactly as `_emit_chunk` matches them.
            """
            names = from_meep._component_names(mp, True)
            worst, sites = 0.0, 0
            chunk = cpu_monitor.swigobj.F
            while chunk is not None:
                component = names[int(chunk.c)]
                theirs_w = meep_site_weights(chunk)
                nr, nz = theirs_w.shape
                for name, _weight, acc, _source in entries:
                    if name != component:
                        continue
                    try:
                        lo0 = acc.index_for_doubled(0, int(chunk._is.r()))
                        lo2 = acc.index_for_doubled(2, int(chunk._is.z()))
                    except ValueError:
                        continue
                    first0, first2 = acc.first_index(0), acc.first_index(2)
                    stop0, stop2 = acc._slices[0].stop, acc._slices[2].stop
                    if not (first0 <= lo0 and lo0 + nr <= stop0
                            and first2 <= lo2 and lo2 + nz <= stop2):
                        continue
                    ours_w = np.asarray(acc._weights)[
                        lo0 - first0: lo0 - first0 + nr, 0,
                        lo2 - first2: lo2 - first2 + nz]
                    worst = max(worst, float(np.abs(ours_w - theirs_w).max()
                                             / np.abs(theirs_w).max()))
                    sites += int(ours_w.size)
                    break
                chunk = chunk.next_in_dft
            return worst, sites

        def centred_radius_ring(origin_doubled, parity, start, count, weights, dx):
            """The CONTROL: the ring at the cell-centre radius, not the Yee radius.

            `Near2FarMonitor._face_weights` legitimately uses `(index + 0.5)*dx` because
            that monitor samples cell CENTRES; a raw-Yee accumulator does not, and using
            it here moves every shift-0 component (Ep, Hr, Ez) half a cell outward — a
            slip that leaves the far field smooth and plausible.
            """
            rows = np.arange(count, dtype=np.float64) + start
            return weights * (2.0 * np.pi * (rows + 0.5) * dx)

        from meep_gpu import dft as dft_module
        from meep.simulation import NearToFarData
        report = {"case": case, "runs": {}}
        for resolution in (20, 40):
            for m in (0, 1, 2):
                label = f"res{resolution}_m{m}"
                sim, monitor = build(resolution, m)
                verdict = gpu_compatibility(sim)
                driver = lift_simulation(sim, prefer_gpu=False)
                migration = driver.migrated_monitors[id(monitor)]
                driver.run(until=12.0)
                cpu, cpu_monitor = build(resolution, m)
                cpu.run(until=12.0)
                ours = migration.packed(mp, cpu_monitor)
                theirs = np.asarray(cpu.get_near2far_data(cpu_monitor).F)
                weight_error, sites_compared = weights_vs_meep(migration.entries, cpu_monitor)
                driver.close()

                # The control lifts the SAME simulation with the ring rebuilt at the
                # centred radius and re-measures, so the number below is what this
                # comparison would have reported had the half-cell slip been taken.
                original_ring = dft_module._cylindrical_ring_weights
                dft_module._cylindrical_ring_weights = centred_radius_ring
                try:
                    slipped_sim, slipped_monitor = build(resolution, m)
                    slipped_driver = lift_simulation(slipped_sim, prefer_gpu=False)
                    slipped_entries = slipped_driver.migrated_monitors[
                        id(slipped_monitor)].entries
                    control_error, _ = weights_vs_meep(slipped_entries, cpu_monitor)
                    slipped_driver.close()
                finally:
                    dft_module._cylindrical_ring_weights = original_ring

                # --- the far field, MEEP's own evaluation of this engine's surface ---
                segments, cursor, chunk = [], 0, cpu_monitor.swigobj.F
                while chunk is not None:
                    count = int(chunk.N) * nfreq
                    segments.append(ours[cursor:cursor + count]
                                    .reshape(int(chunk.N), nfreq).T.ravel())
                    cursor += count
                    chunk = chunk.next_in_dft
                transposed = np.concatenate(segments)
                loaded, loaded_monitor = build(resolution, m)
                loaded.init_sim()
                loaded.load_near2far_data(loaded_monitor, NearToFarData(F=ours))
                our_far = np.asarray([loaded.get_farfield(loaded_monitor, p)
                                      for p in far_points]).ravel()
                meep_far = np.asarray([cpu.get_farfield(cpu_monitor, p)
                                       for p in far_points]).ravel()
                loaded.load_near2far_data(loaded_monitor, NearToFarData(F=transposed))
                bad_far = np.asarray([loaded.get_farfield(loaded_monitor, p)
                                      for p in far_points]).ravel()
                report["runs"][label] = {
                    "supported": bool(verdict.supported),
                    "reasons": list(verdict.reasons),
                    "chunks": len(migration.entries),
                    "stored_size": int(ours.size),
                    "meep_stored_size": int(theirs.size),
                    "sites_compared": sites_compared,
                    "weight_error": float(weight_error),
                    "centred_radius_weight_error": float(control_error),
                    "stored_error": float(np.linalg.norm(ours - theirs)
                                          / np.linalg.norm(theirs)),
                    "far_error": float(np.linalg.norm(our_far - meep_far)
                                       / np.linalg.norm(meep_far)),
                    "transposed_stored_error": float(np.linalg.norm(transposed - theirs)
                                                     / np.linalg.norm(theirs)),
                    "transposed_far_error": float(np.linalg.norm(bad_far - meep_far)
                                                  / np.linalg.norm(meep_far)),
                    "far_is_all_zero": bool(np.abs(our_far).max() == 0.0),
                    "meep_far_norm": float(np.linalg.norm(meep_far)),
                }
                print(f"[child] near2far cyl {label}: weights={weight_error:.2e} "
                      f"stored={report['runs'][label]['stored_error']:.2e} "
                      f"far={report['runs'][label]['far_error']:.2e}", flush=True)
        return report
    elif case == "n2f_point_region_and_flux_box_corners":
        # The two near2far shapes the corpus exposed and the 3-D test above cannot
        # see, each with the measured defect it pins:
        #
        # * A POINT Near2FarRegion on a 1-D-emulated (sx, 0, 0) cell —
        #   binary_grating_n2f.py. MEEP resolves its normal through the nosize pad
        #   (dft.cpp:806-824) and loops the one-pixel periodic y axis over its TWO
        #   lattice images: one dft chunk per image, identical is/ie, single-site
        #   y weights w0 and w1 = 1 - w0 (measured 0.0 and 1.0). This engine holds
        #   ONE plane at the summed weight 1, so each chunk must be emitted scaled
        #   by its own s0 — the `image_ignored` control emits both at full weight,
        #   which doubles the surface (measured: far field exactly 1.0 relative).
        # * A FOUR-FACE box with ±1 region weights — antenna-radiation.py without
        #   its symmetries. Its corner site lies in TWO regions carrying the same
        #   component at OPPOSITE stored weights, and matching by containment alone
        #   emits the corner with the other face's sign (measured: exactly 2.0e+00
        #   relative on that one single-site chunk, 3.6e-02 on the stored total).
        #   The `corner_filter_dropped` control is that defect re-applied.
        from meep.simulation import NearToFarData

        def build_point():
            resolution = 25
            sx = 8.5
            fmin, fmax = 1 / 0.6, 1 / 0.4
            fcen, df = 0.5 * (fmin + fmax), fmax - fmin
            sim = mp.Simulation(
                resolution=resolution, cell_size=mp.Vector3(sx),
                boundary_layers=[mp.PML(1.0, direction=mp.X)],
                k_point=mp.Vector3(), default_material=mp.Medium(index=1.5),
                sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez,
                                   center=mp.Vector3(-0.5 * sx + 1.0 + 1.5))],
            )
            monitor = sim.add_near2far(
                fcen, df, 5, mp.Near2FarRegion(center=mp.Vector3(0.5 * sx - 1.0 - 1.5)))
            return sim, monitor

        def build_box():
            sim = mp.Simulation(
                resolution=20, cell_size=mp.Vector3(6.0, 6.0, 0),
                boundary_layers=[mp.PML(1.0)],
                sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                   center=mp.Vector3(), component=mp.Ez)],
            )
            monitor = sim.add_near2far(
                1.0, 0, 1,
                mp.Near2FarRegion(center=mp.Vector3(0, 2.0), size=mp.Vector3(4.0, 0)),
                mp.Near2FarRegion(center=mp.Vector3(0, -2.0), size=mp.Vector3(4.0, 0),
                                  weight=-1),
                mp.Near2FarRegion(center=mp.Vector3(2.0, 0), size=mp.Vector3(0, 4.0)),
                mp.Near2FarRegion(center=mp.Vector3(-2.0, 0), size=mp.Vector3(0, 4.0),
                                  weight=-1),
            )
            return sim, monitor

        report = {"case": case, "runs": {}}
        specs = {
            # (N-0.5)*dt keeps both codes at exactly N steps (until = N*dt exactly is
            # the ambiguous stop, MEEP taking N+1); both runs reach well past the
            # source (point: last_source_time ~ 12 at t = 24; box: ~ 50 at t = 55).
            "point": (build_point, 1200,
                      [mp.Vector3(1e6, 0.2e6), mp.Vector3(1e6, -0.1e6), mp.Vector3(1e6, 0)]),
            "box": (build_box, 2200,
                    [mp.Vector3(1000 * np.cos(t), 1000 * np.sin(t))
                     for t in np.linspace(0.0, 2 * np.pi, 8)]),
        }
        for label, (build, num_steps, far_points) in specs.items():
            sim, monitor = build()
            verdict = gpu_compatibility(sim)
            driver = lift_simulation(sim, prefer_gpu=False)
            migration = driver.migrated_monitors[id(monitor)]
            until = (num_steps - 0.5) * sim.Courant / sim.resolution
            driver.run(until=until)
            cpu, cpu_monitor = build()
            cpu.run(until=until)
            ours = migration.packed(mp, cpu_monitor)
            theirs = np.asarray(cpu.get_near2far_data(cpu_monitor).F)
            # CONTROLS, each the natural wrong implementation re-applied:
            # image_ignored drops the per-lattice-image weight (the one-pixel y axis
            # emits both image chunks in full), corner_unfiltered matches
            # accumulators by containment alone — neither stored weight nor MEEP's
            # equivalent-source label `vc` allowed to separate two claimants.
            grid = migration.entries[0][2].grid
            grid.nosize_direction = lambda axis: False  # Instance attr shadows the method.
            try:
                image_ignored = migration.packed(mp, cpu_monitor)
            finally:
                del grid.nosize_direction  # Restore the class method.
            real_emit = type(migration)._emit_chunk

            def emit_unfiltered(self, component, low, high, expected_sites,
                                image_weight=1.0, stored_weight=None, flips=(),
                                source_component=None):
                return real_emit(self, component, low, high, expected_sites,
                                 image_weight=image_weight, stored_weight=None,
                                 flips=flips, source_component=None)

            type(migration)._emit_chunk = emit_unfiltered
            try:
                corner_unfiltered = migration.packed(mp, cpu_monitor)
            finally:
                type(migration)._emit_chunk = real_emit
            loaded, loaded_monitor = build()
            loaded.init_sim()

            def far(packed):
                loaded.load_near2far_data(loaded_monitor, NearToFarData(F=packed))
                return np.asarray([loaded.get_farfield(loaded_monitor, p)
                                   for p in far_points]).ravel()

            our_far = far(ours)
            meep_far = np.asarray([cpu.get_farfield(cpu_monitor, p)
                                   for p in far_points]).ravel()
            far_norm = float(np.linalg.norm(meep_far))
            report["runs"][label] = {
                "supported": bool(verdict.supported),
                "reasons": list(verdict.reasons),
                "steps": [int(driver.step_count), int(cpu.fields.t)],
                "stored_size": int(ours.size),
                "meep_stored_size": int(theirs.size),
                "stored_error": float(np.linalg.norm(ours - theirs)
                                      / np.linalg.norm(theirs)),
                "far_error": float(np.linalg.norm(our_far - meep_far) / far_norm),
                "image_ignored_far_error": float(
                    np.linalg.norm(far(image_ignored) - meep_far) / far_norm),
                "corner_unfiltered_stored_error": float(
                    np.linalg.norm(corner_unfiltered - theirs) / np.linalg.norm(theirs)),
                "meep_far_norm": far_norm,
            }
            driver.close()
            print(f"[child] {label}: stored={report['runs'][label]['stored_error']:.2e} "
                  f"far={report['runs'][label]['far_error']:.2e} "
                  f"image_ignored={report['runs'][label]['image_ignored_far_error']:.2e} "
                  f"corner={report['runs'][label]['corner_unfiltered_stored_error']:.2e}",
                  flush=True)
        return report
    elif case == "n2f_corner_claimant_filters":
        # WHICH accumulator answers for a chunk two near2far regions both contain.
        # Three shapes, each isolating one of the two filters that decide it:
        #
        # * duplicate_plane: TWO regions on the same plane at different weights. MEEP's
        #   equivalent-source label `vc` is identical on both (same normal, same j), so
        #   only the stored weight separates their chunks.
        # * double_mirror: antenna-radiation.py's four-face box WITH Mirror(X)+Mirror(Y)
        #   at resolution 50 — MEEP's own test_antenna_radiation::test_poynting_theorem
        #   cell. The fold splits the right face's ZERO-WEIGHT ghost column into a
        #   one-site chunk at the box corner, which the bottom face also contains, and
        #   the two carry stored weights (1+0j) and (1-0j): equal in value, different
        #   only in the sign of the zero imaginary part. Either filter alone separates
        #   them; with both dropped the bottom face answers at FULL weight where MEEP
        #   stores exactly zero.
        # * complex_weights: the same cell with region weights 1+1j and -1-1j, for which
        #   the tangential-pair signs make the two claimants' stored weights identical
        #   BIT FOR BIT. Nothing about the weight can separate them and only `vc` does.
        real_same_stored_weight = from_meep._same_stored_weight

        def abs_compared(entry_weight, chunk_weight):
            """The pre-fix filter: `abs(a - b)`, which cannot see a signed zero."""
            return abs(complex(entry_weight) - chunk_weight) <= 1e-12 * (
                1.0 + abs(chunk_weight))

        def build_pair(weights, symmetries, resolution):
            sim = mp.Simulation(
                resolution=resolution, cell_size=mp.Vector3(6.0, 6.0, 0),
                boundary_layers=[mp.PML(1.0)], symmetries=list(symmetries),
                sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                   center=mp.Vector3(), component=mp.Ez)],
            )
            monitor = sim.add_near2far(1.0, 0, 1, *[
                mp.Near2FarRegion(center=mp.Vector3(*center), size=mp.Vector3(*size),
                                  weight=weight)
                for center, size, weight in weights])
            return sim, monitor

        def coincident(_unused=None):
            # Same plane twice, weights 1 and 3: identical component, identical vc,
            # identical containment. Only the stored weight tells the chunks apart.
            return build_pair(
                [((0, 2.0, 0), (4.0, 0, 0), 1), ((0, 2.0, 0), (4.0, 0, 0), 3)], (), 10)

        def folded_box(weight):
            half = 2.0
            return build_pair(
                [((0, half, 0), (4.0, 0, 0), weight),
                 ((0, -half, 0), (4.0, 0, 0), -weight),
                 ((half, 0, 0), (0, 4.0, 0), weight),
                 ((-half, 0, 0), (0, 4.0, 0), -weight)],
                (mp.Mirror(mp.X), mp.Mirror(mp.Y)), 50)

        report = {"case": case, "runs": {}}
        specs = {
            "duplicate_plane": (lambda: coincident(), 600),
            "double_mirror": (lambda: folded_box(1), 1200),
            "complex_weights": (lambda: folded_box(1 + 1j), 1200),
        }
        for label, (build, num_steps) in specs.items():
            sim, monitor = build()
            verdict = gpu_compatibility(sim)
            driver = lift_simulation(sim, prefer_gpu=False)
            migration = driver.migrated_monitors[id(monitor)]
            until = (num_steps - 0.5) * sim.Courant / sim.resolution
            driver.run(until=until)
            cpu, cpu_monitor = build()
            cpu.run(until=until)
            ours = np.asarray(migration.packed(mp, cpu_monitor))
            theirs = np.asarray(cpu.get_near2far_data(cpu_monitor).F)
            real_emit = type(migration)._emit_chunk

            def without(weight=True, source=True):
                """`packed` with one or both claimant filters switched off."""
                def emit(self, component, low, high, expected_sites, image_weight=1.0,
                         stored_weight=None, flips=(), source_component=None):
                    return real_emit(
                        self, component, low, high, expected_sites,
                        image_weight=image_weight,
                        stored_weight=stored_weight if weight else None,
                        flips=flips,
                        source_component=source_component if source else None)
                type(migration)._emit_chunk = emit
                try:
                    return np.asarray(migration.packed(mp, cpu_monitor))
                finally:
                    type(migration)._emit_chunk = real_emit

            weight_dropped = without(weight=False)
            source_dropped = without(source=False)
            both_dropped = without(weight=False, source=False)
            from_meep._same_stored_weight = abs_compared
            try:
                abs_compared_data = np.asarray(migration.packed(mp, cpu_monitor))
                # The pre-fix code exactly: abs() comparison, and no `vc` filter at all.
                pre_fix = without(source=False)
            finally:
                from_meep._same_stored_weight = real_same_stored_weight
            norm = float(np.linalg.norm(theirs))

            def relative(data):
                return float(np.linalg.norm(data - theirs) / norm)

            # The entry weights must be MEEP's chunk weights BIT FOR BIT, signed zeros
            # included — that is what makes the comparison below exact rather than lucky.
            chunk, entry_weights, chunk_weights, sources, chunk_sources = (
                cpu_monitor.swigobj.F, [], [], [], [])
            names = from_meep._component_names(mp, False)
            while chunk is not None:
                chunk_weights.append(complex(chunk.stored_weight))
                chunk_sources.append(names.get(int(chunk.vc)))
                chunk = chunk.next_in_dft
            for _name, weight, _accumulator, source in migration.entries:
                entry_weights.append(complex(weight))
                sources.append(source)
            unmatched_weight = [
                (repr(w), [repr(e) for e in entry_weights])
                for w in chunk_weights
                if not any(e.real == w.real and e.imag == w.imag
                           and math.copysign(1.0, e.imag) == math.copysign(1.0, w.imag)
                           for e in entry_weights)
            ]
            report["runs"][label] = {
                "supported": bool(verdict.supported),
                "reasons": list(verdict.reasons),
                "steps": [int(driver.step_count), int(cpu.fields.t)],
                "stored_size": int(ours.size), "meep_stored_size": int(theirs.size),
                "stored_error": relative(ours),
                "weight_filter_dropped": relative(weight_dropped),
                "source_filter_dropped": relative(source_dropped),
                "both_filters_dropped": relative(both_dropped),
                "abs_weight_comparison": relative(abs_compared_data),
                "pre_fix": relative(pre_fix),
                "meep_stored_norm": norm,
                "chunk_weights_all_matched_with_signed_zeros": not unmatched_weight,
                "unmatched_weights": unmatched_weight[:4],
                "chunk_sources_are_entry_sources": (
                    set(chunk_sources) <= set(sources) and None not in chunk_sources),
            }
            driver.close()
            row = report["runs"][label]
            print(f"[child] n2f claimants {label}: stored={row['stored_error']:.2e} "
                  f"weight_dropped={row['weight_filter_dropped']:.2e} "
                  f"vc_dropped={row['source_filter_dropped']:.2e} "
                  f"both={row['both_filters_dropped']:.2e} "
                  f"abs={row['abs_weight_comparison']:.2e} "
                  f"pre_fix={row['pre_fix']:.2e}", flush=True)
        return report
    elif case == "monitor_region_negative_size":
        # MEEP's `volume` SORTS its corners (vec.cpp:163-167) and its Python layer
        # builds one from `center -/+ size/2` (simulation.py:443-449), so a negative
        # size component names the same span as its absolute value. MEEP's own
        # test_cavity_farfield.py writes `size=2*dpml - sx` = -10.4. Each monitor kind
        # is built twice, once each way, and must agree with MEEP AND with itself.
        def build(sign, kind):
            sim = mp.Simulation(
                resolution=10, cell_size=mp.Vector3(8, 8, 0),
                boundary_layers=[mp.PML(1.0)],
                sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2), mp.Ez,
                                   center=mp.Vector3())])
            if kind == "flux":
                monitor = sim.add_flux(
                    1.0, 0, 1,
                    mp.FluxRegion(center=mp.Vector3(0, 2), size=mp.Vector3(sign * 4.0, 0)),
                    decimation_factor=1)
            elif kind == "near2far":
                monitor = sim.add_near2far(
                    1.0, 0, 1,
                    mp.Near2FarRegion(center=mp.Vector3(0, 2),
                                      size=mp.Vector3(sign * 4.0, 0)),
                    decimation_factor=1)
            else:
                monitor = sim.add_dft_fields(
                    [mp.Ez], 1.0, 0, 1, center=mp.Vector3(),
                    size=mp.Vector3(sign * 4.0, sign * 4.0), decimation_factor=1)
            return sim, monitor

        def read(sign, kind, num_steps):
            sim, monitor = build(sign, kind)
            verdict = gpu_compatibility(sim)
            driver = lift_simulation(sim, prefer_gpu=False)
            until = (num_steps - 0.5) * sim.Courant / sim.resolution
            driver.run(until=until)
            migrated = driver.migrated_monitors[id(monitor)]
            cpu, cpu_monitor = build(sign, kind)
            cpu.run(until=until)
            if kind == "flux":
                ours = np.asarray(migrated.get_flux_spectrum(), dtype=np.complex128)
                theirs = np.asarray(mp.get_fluxes(cpu_monitor), dtype=np.complex128)
            elif kind == "near2far":
                ours = np.asarray(migrated.packed(mp, cpu_monitor))
                theirs = np.asarray(cpu.get_near2far_data(cpu_monitor).F)
            else:
                ours = np.asarray(migrated.get_dft_spectrum("Ez"))[0, :, :, 0]
                theirs = np.asarray(cpu.get_dft_array(cpu_monitor, mp.Ez, 0))
            driver.close()
            return bool(verdict.supported), list(verdict.reasons), ours, theirs

        report = {"case": case, "runs": {}}
        for kind in ("flux", "near2far", "fields"):
            supported, reasons, positive, meep_positive = read(+1, kind, 600)
            _s, _r, negative, meep_negative = read(-1, kind, 600)

            def relative(ours, theirs):
                denominator = float(np.linalg.norm(theirs))
                return float(np.linalg.norm(np.asarray(ours).ravel()
                                            - np.asarray(theirs).ravel()) / denominator)

            report["runs"][kind] = {
                "supported": supported, "reasons": reasons,
                "positive_vs_meep": relative(positive, meep_positive),
                "negative_vs_meep": relative(negative, meep_negative),
                "negative_vs_positive": float(np.linalg.norm(
                    np.asarray(negative).ravel() - np.asarray(positive).ravel())),
                "meep_negative_vs_positive": float(np.linalg.norm(
                    np.asarray(meep_negative).ravel() - np.asarray(meep_positive).ravel())),
                "size": int(np.asarray(positive).size),
            }
            row = report["runs"][kind]
            print(f"[child] negative size {kind}: +{row['positive_vs_meep']:.2e} "
                  f"-{row['negative_vs_meep']:.2e} "
                  f"identical={row['negative_vs_positive']:.3e}", flush=True)
        return report
    elif case == "flux_region_volume":
        # MEEP lets a FluxRegion carry extent along its OWN normal — its
        # test_visualization.py builds `FluxRegion(size=(4,4,4), direction=mp.X)`.
        # `_add_fluxish_stuff` passes the volume through unexamined, so the monitor
        # integrates the normal Poynting component over a volume rather than across a
        # surface, with MEEP's dV picking up one 1/a per extended direction. Each
        # volume is measured beside the flat plane it degenerates to.
        def build(dims, center, size, direction):
            cell = mp.Vector3(8, 8, 8) if dims == 3 else mp.Vector3(8, 8, 0)
            sim = mp.Simulation(
                resolution=10, cell_size=cell, boundary_layers=[mp.PML(1.0)],
                sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2), mp.Ez,
                                   center=mp.Vector3())])
            monitor = sim.add_flux(
                1.0, 0, 1,
                mp.FluxRegion(center=mp.Vector3(*center), size=mp.Vector3(*size),
                              direction=direction),
                decimation_factor=1)
            return sim, monitor

        report = {"case": case, "runs": {}}
        specs = {
            "plane_2d": (2, (2, 0, 0), (0, 4, 0), mp.X, 400),
            "volume_2d_x": (2, (2, 0, 0), (2, 4, 0), mp.X, 400),
            "volume_2d_y": (2, (0, 1, 0), (4, 4, 0), mp.Y, 400),
            "plane_3d": (3, (2, 0, 0), (0, 4, 4), mp.X, 200),
            "volume_3d": (3, (2, 0, 0), (4, 4, 4), mp.X, 200),
        }
        for label, (dims, center, size, direction, num_steps) in specs.items():
            sim, monitor = build(dims, center, size, direction)
            verdict = gpu_compatibility(sim)
            driver = lift_simulation(sim, prefer_gpu=False)
            until = (num_steps - 0.5) * sim.Courant / sim.resolution
            driver.run(until=until)
            ours = float(np.real(driver.migrated_monitors[id(monitor)].get_flux_spectrum()[0]))
            driver.close()
            cpu, cpu_monitor = build(dims, center, size, direction)
            cpu.run(until=until)
            theirs = float(mp.get_fluxes(cpu_monitor)[0])
            report["runs"][label] = {
                "supported": bool(verdict.supported), "reasons": list(verdict.reasons),
                "meep": theirs, "ours": ours,
                "relative": abs(ours - theirs) / abs(theirs) if theirs else None,
                "extends_along_normal": float(size["xyz".index(
                    {mp.X: "x", mp.Y: "y", mp.Z: "z"}[direction])]),
            }
            row = report["runs"][label]
            print(f"[child] flux volume {label}: meep={theirs:.6e} ours={ours:.6e} "
                  f"rel={row['relative']:.2e}", flush=True)
        return report
    elif case == "n2f_gate_lift_agreement":
        # The contract this family exists to enforce: `gpu_compatibility.supported`
        # must equal `lift_simulation` not raising, on the exact shapes that violated
        # it — a point region (was: supported then MeepSimulationNotLiftable from
        # _region_normal), an oversized face (was: supported then a bare ValueError
        # from YeeRegionDFT), and a genuinely ambiguous point region in a full 2-D
        # cell, which MEEP itself aborts on and both sides must refuse.
        def point_region_sim():
            sim = mp.Simulation(
                resolution=25, cell_size=mp.Vector3(8.5),
                boundary_layers=[mp.PML(1.0, direction=mp.X)], k_point=mp.Vector3(),
                sources=[mp.Source(mp.GaussianSource(2.0, fwidth=0.8), component=mp.Ez,
                                   center=mp.Vector3(-2.0))],
            )
            sim.add_near2far(2.0, 0.8, 3, mp.Near2FarRegion(center=mp.Vector3(2.0)))
            return sim

        def oversized_face_sim():
            sim = mp.Simulation(
                resolution=20, cell_size=mp.Vector3(6.0, 6.0, 0),
                boundary_layers=[mp.PML(1.0)],
                sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                   center=mp.Vector3(), component=mp.Ez)],
            )
            sim.add_near2far(1.0, 0, 1, mp.Near2FarRegion(
                center=mp.Vector3(0, 2), size=mp.Vector3(18.0, 0)))
            return sim

        def ambiguous_point_sim():
            sim = mp.Simulation(
                resolution=20, cell_size=mp.Vector3(6.0, 6.0, 0),
                boundary_layers=[mp.PML(1.0)],
                sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                   center=mp.Vector3(), component=mp.Ez)],
            )
            sim.add_near2far(1.0, 0, 1, mp.Near2FarRegion(center=mp.Vector3(0, 2)))
            return sim

        report = {"case": case, "runs": {}}
        for label, build in (("point_region", point_region_sim),
                             ("oversized_face", oversized_face_sim),
                             ("ambiguous_point", ambiguous_point_sim)):
            sim = build()
            verdict = gpu_compatibility(sim)
            try:
                driver = lift_simulation(sim, prefer_gpu=False)
            except MeepSimulationNotLiftable as refusal:
                outcome, detail = "refused", str(refusal)
            except Exception as exc:  # noqa: BLE001 - a bare error IS the defect.
                outcome, detail = f"bare {type(exc).__name__}", str(exc)
            else:
                outcome, detail = "lifted", ""
                driver.close()
            report["runs"][label] = {
                "supported": bool(verdict.supported),
                "reasons": list(verdict.reasons),
                "lift_outcome": outcome,
                "detail": detail[:500],
            }
            print(f"[child] {label}: supported={verdict.supported} lift={outcome}",
                  flush=True)
        return report
    elif case == "multi_region_flux_is_summed":
        # `sim.add_flux(fcen, df, nfreq, *FluxRegions)` — the flux-box idiom — is one
        # MEEP object over SEVERAL planes, and `mp.get_fluxes` returns their weighted
        # sum. It reached `add_flux_monitor` with `normal_direction = NO_DIRECTION` and
        # died on "Unknown flux direction 5" after the gate had said supported, which is
        # the failure mode the pre-flight exists to prevent.
        fcen, df, nfreq = 1.0, 0.3, 3

        def build(regions):
            sim = mp.Simulation(
                cell_size=mp.Vector3(4.0, 4.0, 0), resolution=12.0, dimensions=2,
                force_complex_fields=True, boundary_layers=[mp.PML(0.5)],
                sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez,
                                   center=mp.Vector3(0.05, -0.05))],
            )
            return sim, sim.add_flux(fcen, df, nfreq, *regions)

        # The run must outlast the source. A Gaussian at fwidth = 0.3 peaks near t = 17,
        # and a run stopped at 16 measures the exponential turn-on tail: MEASURED 7.5e-02
        # on the SAME comparison that reads 1.3e-07 at t = 80, for both the single-plane
        # and the summed case alike. The number would have looked like a summation bug.
        until = 80.0
        both = [mp.FluxRegion(center=mp.Vector3(0, 1.0), size=mp.Vector3(2.0, 0), weight=+1),
                mp.FluxRegion(center=mp.Vector3(0, -1.0), size=mp.Vector3(2.0, 0), weight=-1)]
        report = {"case": case, "runs": {}}
        for label, regions in (("top", both[:1]), ("bottom", both[1:]), ("both", both)):
            sim, flux = build(regions)
            verdict = gpu_compatibility(sim)
            driver = lift_simulation(sim, prefer_gpu=False)
            migration = driver.migrated_monitors[id(flux)]
            kind = type(migration).__name__
            driver.run(until=until)
            ours = np.asarray(migration.get_flux_spectrum(), dtype=float)
            # The control is the behaviour this replaces: the first region alone, which
            # `_monitor_region` silently returned for a monitor with two.
            control = (np.asarray(migration.parts[0][1].get_flux_spectrum(), dtype=float)
                       if kind == "FluxMigration" else ours)
            driver.close()
            cpu, cpu_flux = build(regions)
            cpu.run(until=until)
            theirs = np.asarray(mp.get_fluxes(cpu_flux), dtype=float)
            report["runs"][label] = {
                "supported": bool(verdict.supported),
                "reasons": list(verdict.reasons),
                "kind": kind,
                "meep_normal_direction": int(flux.normal_direction),
                "error": float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs)),
                "first_region_only_error": float(
                    np.linalg.norm(control - theirs) / np.linalg.norm(theirs)),
            }
            print(f"[child] multi-region flux {label}: {kind} "
                  f"err={report['runs'][label]['error']:.2e}", flush=True)
        return report
    elif case == "hosted_step_functions":
        # MEEP's own step-function wrappers, hosted on this engine's stepper and
        # compared against stock MEEP running the identical wrappers on the identical
        # cell for the identical time. The wrappers are closures written against
        # mp.Simulation (sim.round_time(), sim.fields.last_source_time(),
        # sim.get_field_point(), sim.sources), so what is being measured is whether the
        # facade answers each of those the way MEEP's Simulation does.
        #
        # Three quantities, each of which a plausible facade gets wrong:
        #   * the Harminv SAMPLE COUNT and data_dt — a step function called once too
        #     often, or once too few, still returns a smooth spectrum;
        #   * the at_every FIRE TIMES, exactly — this is the float32 rounding in
        #     fields::round_time (meep.hpp:1892). Carrying the double instead moves a
        #     fire by one step wherever t*dt straddles the threshold;
        #   * the recovered mode frequency, which is what the whole apparatus is for.
        report = {"case": case}
        # MEEP's own ring resonator (python/tests/test_ring.py), with the padding and
        # the PML trimmed so the cell is 100x100 rather than 160x160. A resonator,
        # because the point of hosting mp.Harminv is that it recovers a MODE, and an
        # open cell with no cavity gives it nothing to find.
        n, w, r, pad, dpml = 3.4, 1.0, 1.0, 2.0, 1.0
        sxy = 2 * (r + w + pad + dpml)
        fcen, df = 0.15, 0.1
        geometry = [mp.Cylinder(radius=r + w, material=mp.Medium(index=n)),
                    mp.Cylinder(radius=r)]
        sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez,
                             mp.Vector3(r + 0.1))]

        def build():
            return mp.Simulation(cell_size=mp.Vector3(sxy, sxy), resolution=10,
                                 geometry=geometry, sources=sources,
                                 symmetries=[mp.Mirror(mp.Y)],
                                 boundary_layers=[mp.PML(dpml)])

        def wrappers():
            probe = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
            fires = []
            # A two-argument leaf, so it also exercises the 'finish' todo MEEP's
            # _every forwards; a one-argument leaf would only ever see 'step'.
            def record(sim, todo):
                if todo == "step":
                    fires.append(sim.round_time())
            return probe, fires, [mp.after_sources(probe), mp.at_every(0.7, record)]

        our_probe, our_fires, our_funcs = wrappers()
        # A DRIVER-NATIVE step function alongside MEEP's, in the same run and on the
        # same point. meep_gpu.Harminv asks for `sim.meep_time()` and
        # `sim.get_field_point("Ez", (x, y, z))` — this engine's component NAME and a
        # plain triple, not mp.Ez and an mp.Vector3 — so it pins that the facade the
        # step functions now see has not displaced the driver they used to be handed.
        from meep_gpu import Harminv as DriverHarminv

        native = DriverHarminv(c="Ez", pt=(r + 0.1, 0.0, 0.0), fcen=fcen, df=df)
        result = run_on_gpu(build(), prefer_gpu=False, until=400.0,
                            step_functions=list(our_funcs) + [native])
        report["engine_steps"] = int(result.steps)
        report["engine_time"] = float(result.meep_time)
        result.close()
        report["native_samples"] = len(native.data)
        report["native_modes"] = [[float(m.freq), float(m.Q)] for m in native.modes]

        their_probe, their_fires, their_funcs = wrappers()
        control = build()
        control.run(*their_funcs, until=400.0)
        report["meep_steps"] = int(control.fields.t)
        report["meep_time"] = float(control.round_time())

        report["engine_samples"] = len(our_probe.data)
        report["meep_samples"] = len(their_probe.data)
        report["engine_data_dt"] = float(our_probe.data_dt)
        report["meep_data_dt"] = float(their_probe.data_dt)
        report["engine_fires"] = [float(t) for t in our_fires]
        report["meep_fires"] = [float(t) for t in their_fires]
        report["engine_modes"] = [[float(m.freq), float(m.Q)] for m in our_probe.modes]
        report["meep_modes"] = [[float(m.freq), float(m.Q)] for m in their_probe.modes]
        # The sampled SIGNAL itself, not just what harminv made of it: an
        # aligned-but-empty record would agree on every count above.
        ours = np.asarray(our_probe.data, dtype=complex)
        theirs = np.asarray(their_probe.data, dtype=complex)
        report["signal_max"] = float(np.abs(ours).max()) if ours.size else 0.0
        report["signal_error"] = (
            float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs))
            if ours.size == theirs.size and theirs.size and np.linalg.norm(theirs) > 0 else None
        )
        print(f"[child] hosted step functions: {report['engine_samples']} vs "
              f"{report['meep_samples']} samples, signal {report['signal_error']}", flush=True)
        return report
    elif case == "unhosted_step_function_is_refused":
        # The failure mode this refusal exists for: the mp.Simulation behind a lifted
        # run is INITIALIZED (lift_simulation calls init_sim) and never stepped, so any
        # fallthrough to it answers with a zero field of the right shape. A named
        # refusal is the only way the caller learns the difference.
        from meep_gpu import StepFunctionNotHosted

        report = {"case": case}
        sim = mp.Simulation(
            cell_size=mp.Vector3(4, 4), resolution=10,
            sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), mp.Ez, mp.Vector3())],
            boundary_layers=[mp.PML(0.5)])
        seen = {}

        def reach_for_the_writer(sim_obj, todo):
            if todo == "step" and "raised" not in seen:
                try:
                    seen["value"] = sim_obj.output_volume
                    seen["raised"] = False
                except StepFunctionNotHosted as refusal:
                    seen["raised"] = True
                    seen["message"] = str(refusal)
                except AttributeError as refusal:
                    seen["raised"] = "AttributeError"
                    seen["message"] = str(refusal)

        result = run_on_gpu(sim, prefer_gpu=False, until=2.0,
                            step_functions=[reach_for_the_writer])
        result.close()
        report["raised"] = seen.get("raised")
        report["message"] = seen.get("message", "")
        return report
    elif case == "two_run_normalization_idiom":
        # MEEP's dominant flux idiom: run once WITHOUT the structure, save the flux
        # transform, run again WITH it having loaded the saved transform negated, and
        # read the reflected power off the difference. It is the shape of
        # test_bend_flux, test_refl_angular, test_binary_grating and most of the
        # examples corpus, and it was undriveable here because MEEP's own
        # `load_minus_flux_data` reaches `flux.E`, whose lazy `swigobj` calls
        # `init_sim()` — after which the lift refuses an initialized simulation.
        #
        # A Fresnel slab, because its answer is known independently of both codes.
        n1, n2 = 1.4, 3.5
        t_pml, length_z = 1.0, 7.0
        size_z = length_z + 2 * t_pml
        frequency_min, frequency_max = 1 / 0.8, 1 / 0.4
        fcen = 0.5 * (frequency_min + frequency_max)
        df = frequency_max - frequency_min
        # Resolution 200 is MEEP's own choice in test_refl_angular.py, and it is
        # what the Fresnel comparison needs: the analytic agreement is grid
        # dispersion limited and scales as 1/resolution**2 — measured 48% at
        # res 40, 6.3% at 100, 1.5% at 200, all of them shared by CPU MEEP,
        # against MEEP's own 3% bar. A cheaper grid would have had to drop the
        # independent oracle and keep only the code-to-code comparison.
        nfreq, resolution, until = 11, 200.0, 120.0
        monitor_z = -0.5 * size_z + t_pml + 0.25 * length_z

        def build(with_slab):
            geometry = ([mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5 * size_z),
                                  center=mp.Vector3(z=0.25 * size_z),
                                  material=mp.Medium(index=n2))]
                        if with_slab else [])
            sim = mp.Simulation(
                resolution=resolution, cell_size=mp.Vector3(z=size_z), dimensions=3,
                default_material=mp.Medium(index=n1), boundary_layers=[mp.PML(t_pml)],
                Courant=0.5, geometry=geometry,
                sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ex,
                                   center=mp.Vector3(z=-0.5 * size_z + t_pml))])
            return sim, sim.add_flux(fcen, df, nfreq,
                                     mp.FluxRegion(center=mp.Vector3(z=monitor_z)))

        # --- CPU MEEP's own two-run idiom, unmodified ---
        empty, empty_flux = build(False)
        empty.run(until=until)
        meep_empty_data = empty.get_flux_data(empty_flux)
        meep_empty = np.asarray(mp.get_fluxes(empty_flux), dtype=float)
        slab, slab_flux = build(True)
        slab.load_minus_flux_data(slab_flux, meep_empty_data)
        slab.run(until=until)
        meep_reflectance = -np.asarray(mp.get_fluxes(slab_flux), dtype=float) / meep_empty

        # --- the same idiom through this engine ---
        our_empty_sim, our_empty_flux = build(False)
        first = run_on_gpu(our_empty_sim, until=until, prefer_gpu=False)
        our_data = first.get_flux_data(our_empty_flux)
        our_empty = np.asarray(first.get_flux_spectrum(our_empty_flux), dtype=float)
        first.close()

        # The load names THIS simulation's own flux object — the one its own
        # `add_flux` returned, never the normalization run's — so the second run is
        # built inside the helper.
        def measured(kind):
            sim_, flux_ = build(True)
            load = {} if kind == "none" else {kind: [(flux_, our_data)]}
            result = run_on_gpu(sim_, until=until, prefer_gpu=False, **load)
            spectrum = np.asarray(result.get_flux_spectrum(flux_), dtype=float)
            result.close()
            return -spectrum / our_empty

        ours = measured("minus_flux_data")
        # Two controls, both of which complete and both of which are wrong:
        #   "none"      - no normalization at all, the raw net flux.
        #   "flux_data" - MEEP's load WITHOUT the negation (load_flux_data), so the
        #                 second run ends at E2 + E1 rather than E2 - E1. This is the
        #                 sign error the whole idiom turns on.
        unnormalized = measured("none")
        unnegated = measured("flux_data")

        analytic = ((n1 - n2) / (n1 + n2)) ** 2
        report = {
            "case": case,
            "nfreq": nfreq,
            "empty_flux_error": float(np.linalg.norm(our_empty - meep_empty)
                                      / np.linalg.norm(meep_empty)),
            "reflectance_error": float(np.linalg.norm(ours - meep_reflectance)
                                       / np.linalg.norm(meep_reflectance)),
            "fresnel_error": float(np.max(np.abs(ours - analytic)) / analytic),
            "meep_fresnel_error": float(np.max(np.abs(meep_reflectance - analytic)) / analytic),
            "unnormalized_error": float(np.linalg.norm(unnormalized - meep_reflectance)
                                        / np.linalg.norm(meep_reflectance)),
            "unnegated_error": float(np.linalg.norm(unnegated - meep_reflectance)
                                     / np.linalg.norm(meep_reflectance)),
            "meep_reflectance": meep_reflectance.tolist(),
            "ours": ours.tolist(),
            "analytic": analytic,
        }
        print(f"[child] two-run normalization: ours vs MEEP "
              f"{report['reflectance_error']:.2e}, vs Fresnel {report['fresnel_error']:.2e}, "
              f"controls {report['unnormalized_error']:.2e} / "
              f"{report['unnegated_error']:.2e}", flush=True)
        return report
    elif case == "bulk_check_is_wired_in":
        # Defining a check and calling it are different facts, and a mutation that
        # deleted the CALL survived a suite that exercised the FUNCTION. This
        # replaces the cross-check with one that always raises, so a lift that
        # reaches the end without raising has stopped invoking it.
        def tripwire(sim_, driver_, volumes_, declined=None):
            raise RuntimeError("bulk cross-check reached")
        from_meep._require_bulk_matches_meep = tripwire
        sim = simulation(geometry=slab)
    elif case == "wall_exclusion_disabled_spread_still_refuses":
        # The exclusion (`_unowned_wall_planes`) and the declared-media spread
        # check are a pair: the exclusion may remove ONLY the wall planes MEEP
        # measurably declined, and the check must still refuse everything else.
        # Forcing the mask empty on ring_gds's class (non-vacuum background,
        # metallic walls) makes the declined-default vacuum visible to the check
        # again — the pre-fix failure verbatim — so a lift that completes here
        # has either lost the check's teeth or started excluding more than the
        # measured planes.
        original_mask = from_meep._unowned_wall_planes

        def no_exclusion(*args, **kwargs):
            return np.zeros_like(original_mask(*args, **kwargs))

        from_meep._unowned_wall_planes = no_exclusion
        sim = simulation(k_point=False, cell_size=mp.Vector3(2, 3, 0),
                         default_material=mp.Medium(index=1.4),
                         sources=cw(center=mp.Vector3(0.05, -0.15, 0)),
                         geometry=[mp.Block(mp.Vector3(0.6, mp.inf, mp.inf),
                                            center=mp.Vector3(0.3, 0, 0),
                                            material=mp.Medium(index=3.4))])
    elif case == "wall_exclusion_policies_on_a_bulk_wall_cell":
        # THE SECOND CONSUMER of the same exclusion, and the choice that decides it.
        # A tilted block crossing the low-x wall of a vacuum cell puts ONE bulk cell
        # (i=0, j=2 at these settings) where all three sampled components read the
        # declined vacuum default while MEEP's own cell-centred diagnostic — served
        # from stored chunk data no per-point public reader will hand back
        # (fields::get_chi1inv substitutes 1.0, monitor.cpp:181-183;
        # structure::get_chi1inv returns 0.0, monitor.cpp:238) — reports the real
        # smoothed material. The bulk cross-check then refuses a lift that read the
        # structure correctly. This is the class that refused
        # test_mode_decomposition.py::test_oblique_waveguide_backward_mode.
        #
        # The cell is chosen so the mask POLICY is measurable, not just its presence:
        # Ex's x window opens at corner+1 and is genuinely owned on the whole i=0
        # column, so the failing cell is declined for Ey and Ez ONLY. Every policy
        # below is run against the identical simulation and the identical check; only
        # the mask handed to `_require_bulk_matches_meep` changes.
        real_mask = from_meep._unowned_wall_planes
        real_bulk = from_meep._require_bulk_matches_meep
        masks = []

        def recording_mask(*args, **kwargs):
            mask = real_mask(*args, **kwargs)
            masks.append(np.asarray(mask).copy())
            return mask

        angle = math.radians(35.0)
        def wall_cell():
            return simulation(
                k_point=False, cell_size=mp.Vector3(2, 2, 0), resolution=20,
                force_complex_fields=False,
                sources=cw(center=mp.Vector3(0.05, -0.15, 0)),
                geometry=[mp.Block(
                    center=mp.Vector3(0, 0.25, 0), size=mp.Vector3(mp.inf, 0.6, mp.inf),
                    e1=mp.Vector3(1, 0, 0).rotate(mp.Vector3(0, 0, 1), angle),
                    e2=mp.Vector3(0, 1, 0).rotate(mp.Vector3(0, 0, 1), angle),
                    material=mp.Medium(index=3.5))])

        report = {"case": case, "raised": None, "message": "", "policies": {}}
        for policy in ("union", "none", "ex_only", "intersection"):
            masks.clear()
            from_meep._unowned_wall_planes = recording_mask

            def bulk(sim_, driver_, volumes_, declined=None, _policy=policy):
                chosen = {
                    "union": declined,
                    "none": None,
                    "ex_only": masks[0] if masks else None,
                    "intersection": (masks[0] & masks[1] & masks[2]) if len(masks) == 3 else None,
                }[_policy]
                return real_bulk(sim_, driver_, volumes_, declined=chosen)

            from_meep._require_bulk_matches_meep = bulk
            try:
                driver = lift_simulation(wall_cell(), prefer_gpu=False)
            except Exception as exc:  # noqa: BLE001 - the test asserts on the type.
                report["policies"][policy] = {"raised": type(exc).__name__,
                                              "message": str(exc)}
            else:
                driver.close()
                report["policies"][policy] = {"raised": None, "message": ""}
            if policy == "union":
                report["mask_sizes"] = [int(mask.sum()) for mask in masks]
                report["union_size"] = int((masks[0] | masks[1] | masks[2]).sum())
                report["intersection_size"] = int((masks[0] & masks[1] & masks[2]).sum())
                report["cells"] = int(masks[0].size)
        from_meep._unowned_wall_planes = real_mask
        from_meep._require_bulk_matches_meep = real_bulk
        return report

    elif case == "conductivity_read_is_exact_and_self_checked":
        # `elif`, not `if`: this branch belongs to the SAME chain as every case above
        # it. Starting a second chain here strands every case that falls through to
        # the shared `lift_simulation` tail — it reaches this `if`, misses, and hits
        # the `else` below as "unknown gate". Measured: five gate cases
        # (half_cell_shifted_sampling, whole_cell_shifted_sampling,
        # offlattice_sampling, bulk_check_is_wired_in and
        # wall_exclusion_disabled_spread_still_refuses) failed that way.
        #
        # The per-point D_conductivity read, on both halves of what makes it a READ.
        #
        # THE VALUE. Two cells whose true answer is known in closed form: a declared
        # D_conductivity=0.7 block, and a MaterialGrid whose damping puts
        # u*(1-u)*damping into the same array, whose maximum is damping/4 at u = 1/2.
        # Neither number is a tolerance on a field — they are the declaration itself,
        # recovered.
        #
        # THE SELF-CHECKS, forced. `_read_conductivity_volumes` takes `sim` only for
        # `sim.fields` and `sim.resolution`, so a shim in front of `get_chi1inv` puts
        # the exact wrong MEEP in its way without touching MEEP itself:
        #   * `identity_broken` — a MEEP that ALLOCATED chi1inv on the D row, so the
        #     frequency-0 answer is a permittivity instead of the literal 1.0. Every
        #     recovered sigma would then be eps*sigma, and the run would complete.
        #   * `frequency_dependent` — a D row carrying something that varies with
        #     frequency, which sigma_D never does.
        #   * `negative` — gain wearing loss's name.
        report = {"case": case}

        class _FieldShim:
            def __init__(self, fields, answer):
                self.gv = fields.gv
                self._answer = answer

            def get_chi1inv(self, component, direction, iloc, frequency):
                return self._answer(frequency)

        class _SimShim:
            def __init__(self, sim, answer):
                self.resolution = sim.resolution
                self.fields = _FieldShim(sim.fields, answer)

        def block_cell():
            return mp.Simulation(
                cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
                boundary_layers=[mp.PML(0.5)],
                geometry=[mp.Block(mp.Vector3(1.2, 1.2, mp.inf), center=mp.Vector3(0.4, 0),
                                   material=mp.Medium(epsilon=12, D_conductivity=0.7))],
                sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                   center=mp.Vector3(-1.2, 0))])

        def grid_cell():
            return mp.Simulation(
                cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
                boundary_layers=[mp.PML(0.5)],
                geometry=[mp.Block(mp.Vector3(1.5, 1.5, mp.inf), center=mp.Vector3(0.4, 0),
                                   material=_damped_material_grid())],
                sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=mp.Ez,
                                   center=mp.Vector3(-1.2, 0))])

        for label, builder, declared in (("block", block_cell, 0.7),
                                         ("grid", grid_cell, _MATERIAL_GRID_DAMPING / 4.0)):
            driver = lift_simulation(builder(), prefer_gpu=False)
            maxima = [float(np.asarray(driver.fields.conductivity_for(name)).max())
                      for name in ("Dx", "Dy", "Dz")]
            report[label + "_declared"] = float(declared)
            report[label + "_recovered_max"] = maxima
            report[label + "_min"] = min(
                float(np.asarray(driver.fields.conductivity_for(name)).min())
                for name in ("Dx", "Dy", "Dz"))
            driver.close()

        answers = {
            # frequency 0 answers with a permittivity rather than the literal 1.0
            "identity_broken": lambda frequency: (1.0 / 12.0 if frequency == 0
                                                  else 1.0 / (1.0 + 1j * 0.7 / frequency)),
            # sigma_read = frequency * 0.7/0.9, so the two probes disagree
            "frequency_dependent": lambda frequency: (
                1.0 if frequency == 0 else 1.0 / (1.0 + 1j * 0.7 / 0.9)),
            "negative": lambda frequency: (1.0 if frequency == 0
                                           else 1.0 / (1.0 - 1j * 0.7 / frequency)),
        }
        report["self_checks"] = {}
        for label, answer in answers.items():
            sim = block_cell()
            driver = lift_simulation(sim, prefer_gpu=False)
            try:
                from_meep._read_conductivity_volumes(mp, _SimShim(sim, answer), driver)
            except Exception as exc:  # noqa: BLE001 - the test asserts on the type.
                report["self_checks"][label] = {"raised": type(exc).__name__,
                                                "message": str(exc)}
            else:
                report["self_checks"][label] = {"raised": None, "message": ""}
            driver.close()
        return report

    else:
        raise SystemExit("unknown gate: " + case)

    payload = {"case": case}
    try:
        driver = lift_simulation(sim, prefer_gpu=False)
    except Exception as exc:  # noqa: BLE001 - the test asserts on the type it got.
        payload["raised"] = type(exc).__name__
        payload["message"] = str(exc)
    else:
        driver.close()
        payload["raised"] = None
        payload["message"] = ""
    return payload


def strip_phantom(array):
    """Drop MEEP's trailing all-zero rows from a folded get_dft_array.

    Under symmetry, ``get_dft_component_dims`` initializes its corner scan from
    the FOLDED fields volume (dft.cpp:1061-1062: ``min_corner = round_vec(
    v.get_max_corner()) + 1`` with ``v`` the halved volume); for a region wholly
    below the fold the init max-corner survives the chunk scan, so the returned
    array carries trailing all-zero rows the metadata does not report (measured:
    (10, 8) returned, 6 coordinates, columns 7-8 identically zero). The true
    rows are compared; the phantom is MEEP's readback artifact, not data.
    """
    array = np.asarray(array)
    for axis in range(array.ndim):
        while array.shape[axis] > 1:
            tail = array.take([-1], axis=axis)
            if np.abs(tail).max() == 0.0:
                array = array.take(range(array.shape[axis] - 1), axis=axis)
            else:
                break
    return array


def relative(mine, theirs):
    mine, theirs = np.asarray(mine), np.asarray(theirs)
    if mine.shape != theirs.shape:
        return f"SHAPE {mine.shape} vs {theirs.shape}"
    norm = float(np.linalg.norm(theirs.ravel()))
    return float(np.linalg.norm((mine - theirs).ravel()) / norm) if norm else 0.0


def run_fold_monitors(case):
    """Monitors served through the reflected gather, against MEEP on the same object."""
    report = {"case": case}
    if case in ("even", "odd_phase", "odd_count", "metallic"):
        phase = -1 if case == "odd_phase" else 1
        src = "Hz" if case == "odd_phase" else "Ez"
        ny = 1.5 if case == "odd_count" else 2.0
        k_point = False if case == "metallic" else mp.Vector3()
        sim = mp.Simulation(
            cell_size=mp.Vector3(2.0, ny, 0), resolution=10,
            sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                               component=MEEP_COMPONENTS[src],
                               center=mp.Vector3(0.17, 0.23))],
            symmetries=[mp.Mirror(mp.Y, phase=phase)], k_point=k_point)
        regions = {
            "straddle": sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
                                           1.0, 0, 1, center=mp.Vector3(0.3, -0.2),
                                           size=mp.Vector3(0.8, 1.0)),
            "below": sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
                                        1.0, 0, 1, center=mp.Vector3(0.3, -0.6),
                                        size=mp.Vector3(0.8, 0.4)),
        }
        fluxes = {
            "y_below": sim.add_flux(1.0, 0, 1, mp.FluxRegion(
                center=mp.Vector3(0, -0.7), size=mp.Vector3(1.2, 0))),
            "y_above": sim.add_flux(1.0, 0, 1, mp.FluxRegion(
                center=mp.Vector3(0, 0.7), size=mp.Vector3(1.2, 0))),
            "x_straddle": sim.add_flux(1.0, 0, 1, mp.FluxRegion(
                center=mp.Vector3(0.6, 0), size=mp.Vector3(0, 1.4))),
        }
        result = run_on_gpu(sim, until=30.0, prefer_gpu=False)
        sim.run(until=30.0)
        live = ("Ez", "Hx", "Hy") if src == "Ez" else ("Ex", "Ey", "Hz")
        dft_errors = {}
        for name, monitor in regions.items():
            ours = result.monitors[id(monitor)]
            for component in live:
                theirs = strip_phantom(sim.get_dft_array(
                    monitor, MEEP_COMPONENTS[component], 0))
                mine = np.asarray(ours.get_dft(component))[:, :, 0]
                dft_errors[f"{name}.{component}"] = relative(mine, theirs)
        flux_values = {}
        for name, monitor in fluxes.items():
            theirs = float(mp.get_fluxes(monitor)[0])
            mine = float(np.asarray(result.monitors[id(monitor)].get_flux_spectrum())[0])
            flux_values[name] = {"ours": mine, "meep": theirs,
                                 "error": abs(mine - theirs) / abs(theirs)}
        result.close()
        report.update({"dft_errors": dft_errors, "flux": flux_values})
        return report
    if case in ("near2far_box", "near2far_two_mirrors", "near2far_odd_count"):
        if case == "near2far_two_mirrors":
            symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)]
            src, ny = "Hz", 2.0
            regions = [mp.Near2FarRegion(center=mp.Vector3(0, 0.7), size=mp.Vector3(1.2, 0)),
                       mp.Near2FarRegion(center=mp.Vector3(-0.6, 0.35),
                                         size=mp.Vector3(0, 0.7), weight=-1.0)]
            farpoints = [(-3.0, 25.0, 0), (10.0, 16.0, 0)]
        else:
            symmetries = [mp.Mirror(mp.Y)]
            src = "Ez"
            ny = 1.5 if case == "near2far_odd_count" else 2.0
            if case == "near2far_box":
                regions = [
                    mp.Near2FarRegion(center=mp.Vector3(0, 0.7), size=mp.Vector3(1.2, 0)),
                    mp.Near2FarRegion(center=mp.Vector3(0, -0.7), size=mp.Vector3(1.2, 0),
                                      weight=-1.0),
                    mp.Near2FarRegion(center=mp.Vector3(0.6, 0), size=mp.Vector3(0, 1.4)),
                    mp.Near2FarRegion(center=mp.Vector3(-0.6, 0), size=mp.Vector3(0, 1.4),
                                      weight=-1.0),
                ]
            else:
                regions = [mp.Near2FarRegion(center=mp.Vector3(0, -0.7),
                                             size=mp.Vector3(1.2, 0), weight=-1.0)]
            farpoints = [(0.3, -20.0, 0), (18.0, 4.0, 0)]
        sim = mp.Simulation(
            cell_size=mp.Vector3(2.0, ny, 0), resolution=10,
            sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                               component=MEEP_COMPONENTS[src],
                               center=mp.Vector3(0.17, 0.23))],
            symmetries=symmetries, k_point=mp.Vector3())
        n2f = sim.add_near2far(1.0, 0, 1, *regions)
        result = run_on_gpu(sim, until=30.0, prefer_gpu=False)
        sim.run(until=30.0)
        theirs_packed = np.asarray(sim.get_near2far_data(n2f).F)
        theirs_far = [np.asarray(sim.get_farfield(n2f, mp.Vector3(*p))) for p in farpoints]
        ours_packed = np.asarray(result.monitors[id(n2f)].packed(mp, n2f))
        report["packed_error"] = relative(ours_packed, theirs_packed)
        result.load_near2far(sim, n2f)
        report["farfield_errors"] = [
            relative(np.asarray(sim.get_farfield(n2f, mp.Vector3(*p))), reference)
            for p, reference in zip(farpoints, theirs_far)
        ]
        result.close()
        return report
    if case == "integrated_source_fold":
        # finite_grating.py's combination: an is_integrated source sheet across the
        # folded axis, with a PML. MEEP itself folds it exactly (folded vs unfolded
        # MEEP measured 1.7e-07..1.0e-06), so the bar is the ordinary fold floor.
        sim = mp.Simulation(
            cell_size=mp.Vector3(6.0, 4.0, 0), resolution=10,
            boundary_layers=[mp.PML(1.0)],
            sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2, is_integrated=True),
                               component=mp.Ez, center=mp.Vector3(-2.0, 0),
                               size=mp.Vector3(0, 4.0))],
            geometry=[mp.Block(center=mp.Vector3(1.0, 0), size=mp.Vector3(1.0, 1.0, mp.inf),
                               material=mp.Medium(epsilon=4.0))],
            symmetries=[mp.Mirror(mp.Y)], k_point=mp.Vector3())
        result = run_on_gpu(sim, until=25.0, prefer_gpu=False)
        sim.run(until=25.0)
        report["field_errors"] = {
            name: relative(np.asarray(result.get_array(name), dtype=complex),
                           np.asarray(sim.get_array(component=MEEP_COMPONENTS[name]),
                                      dtype=complex))
            for name in ("Ez", "Hx", "Hy")
        }
        result.close()
        return report
    if case == "mie_flux_box_3d":
        # mie_scattering.py's shape at toy size: two mirrors (one odd), an
        # is_integrated planewave, six flux planes — two wholly in discarded halves.
        r = 0.4
        sim = mp.Simulation(
            cell_size=mp.Vector3(1.4, 1.4, 1.4), resolution=10,
            sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4, is_integrated=True),
                               component=mp.Ez, center=mp.Vector3(-0.5, 0, 0),
                               size=mp.Vector3(0, 1.4, 1.4))],
            geometry=[mp.Sphere(radius=0.25, center=mp.Vector3(0.1, 0, 0),
                                material=mp.Medium(epsilon=6.0))],
            symmetries=[mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)],
            k_point=mp.Vector3())
        specs = {
            "x1": dict(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2 * r, 2 * r)),
            "x2": dict(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2 * r, 2 * r)),
            "y1": dict(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r, 0, 2 * r)),
            "y2": dict(center=mp.Vector3(y=+r), size=mp.Vector3(2 * r, 0, 2 * r)),
            "z1": dict(center=mp.Vector3(z=-r), size=mp.Vector3(2 * r, 2 * r, 0)),
            "z2": dict(center=mp.Vector3(z=+r), size=mp.Vector3(2 * r, 2 * r, 0)),
        }
        monitors = {name: sim.add_flux(1.0, 0, 1, mp.FluxRegion(**spec))
                    for name, spec in specs.items()}
        result = run_on_gpu(sim, until=20.0, prefer_gpu=False)
        sim.run(until=20.0)
        report["flux"] = {}
        for name, monitor in monitors.items():
            theirs = float(mp.get_fluxes(monitor)[0])
            mine = float(np.asarray(result.monitors[id(monitor)].get_flux_spectrum())[0])
            report["flux"][name] = {"ours": mine, "meep": theirs,
                                    "error": abs(mine - theirs) / abs(theirs)}
        result.close()
        return report
    if case == "dz_yee_grid_fold":
        # absorbed_power_density.py's monitor: yee_grid=True [Dz, Ez] over a region
        # straddling the fold, on a dispersionless medium (D = eps*E, still a distinct
        # stored array). Compared per component against get_dft_array.
        sim = mp.Simulation(
            cell_size=mp.Vector3(2.0, 2.0, 0), resolution=10,
            sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                               center=mp.Vector3(0.17, 0.23))],
            geometry=[mp.Cylinder(radius=0.3, center=mp.Vector3(0.2, 0),
                                  material=mp.Medium(epsilon=4.0))],
            symmetries=[mp.Mirror(mp.Y)], k_point=mp.Vector3())
        monitor = sim.add_dft_fields([mp.Dz, mp.Ez], 1.0, 0, 1,
                                     center=mp.Vector3(0.2, -0.3),
                                     size=mp.Vector3(0.8, 1.0), yee_grid=True)
        result = run_on_gpu(sim, until=30.0, prefer_gpu=False)
        sim.run(until=30.0)
        ours = result.monitors[id(monitor)]
        report["dft_errors"] = {}
        for component, constant in (("Dz", mp.Dz), ("Ez", mp.Ez)):
            theirs = strip_phantom(sim.get_dft_array(monitor, constant, 0))
            mine = np.asarray(ours.get_dft(component))[:, :, 0]
            report["dft_errors"][component] = relative(mine, theirs)
        result.close()
        return report
    if case == "invariant_extent_flux":
        # coupler.py's 2a: a flux region with a z extent on a 2-D cell. MEEP ignores
        # the extent (a reduced volume has no z slot — measured identical fluxes at
        # 0.0e+00), so the migration flattens it; the driver's direct-API refusal is
        # separate and untouched.
        sim = mp.Simulation(
            cell_size=mp.Vector3(2.0, 2.0, 0), resolution=10,
            sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                               center=mp.Vector3(0.17, 0.23))],
            k_point=mp.Vector3())
        flux = sim.add_flux(1.0, 0, 1, mp.FluxRegion(
            center=mp.Vector3(0.6, 0, 0), size=mp.Vector3(0, 1.2, 0.44)))
        fields = sim.add_dft_fields([mp.Ez], 1.0, 0, 1, center=mp.Vector3(0.3, -0.2, 0),
                                    size=mp.Vector3(0.8, 1.0, 0.44))
        verdict = gpu_compatibility(sim)
        report["supported"] = bool(verdict.supported)
        report["reasons"] = list(verdict.reasons)
        result = run_on_gpu(sim, until=30.0, prefer_gpu=False)
        sim.run(until=30.0)
        theirs = float(mp.get_fluxes(flux)[0])
        mine = float(np.asarray(result.monitors[id(flux)].get_flux_spectrum())[0])
        report["flux"] = {"ours": mine, "meep": theirs,
                          "error": abs(mine - theirs) / abs(theirs)}
        report["dft_error"] = relative(
            np.asarray(result.monitors[id(fields)].get_dft("Ez"))[:, :, 0],
            strip_phantom(sim.get_dft_array(fields, mp.Ez, 0)))
        result.close()
        return report
    raise SystemExit("unknown fold_monitors case: " + case)


# --- result readers: get_dft_array and the flux packer --------------------------------


def _readers_dft_sim(yee_grid, symmetric, monitors):
    fcen = 0.4
    sim = mp.Simulation(
        cell_size=mp.Vector3(4.0, 4.0, 0), resolution=15,
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=0.2), mp.Ez,
                           mp.Vector3(-0.7, 0.0 if symmetric else 0.3))],
        boundary_layers=[mp.PML(1.0)], force_complex_fields=True,
        geometry=[mp.Cylinder(radius=0.4, material=mp.Medium(index=2.5))],
        symmetries=[mp.Mirror(mp.Y, phase=+1)] if symmetric else [])
    objects = [sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=mp.Vector3(*c),
                                  size=mp.Vector3(*s), yee_grid=yee_grid)
               for c, s in monitors]
    return sim, objects


def _readers_flux_sim(symmetric, mode_monitor):
    fcen = 0.25
    sim = mp.Simulation(
        cell_size=mp.Vector3(8.0, 6.0, 0), resolution=20,
        geometry=[mp.Block(size=mp.Vector3(mp.inf, 1.0, mp.inf),
                           material=mp.Medium(index=3.4))],
        sources=[mp.EigenModeSource(src=mp.GaussianSource(fcen, fwidth=0.1),
                                    center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4.0),
                                    eig_band=1, eig_parity=mp.ODD_Z + mp.EVEN_Y)],
        boundary_layers=[mp.PML(1.0)],
        symmetries=[mp.Mirror(mp.Y)] if symmetric else [])
    add = sim.add_mode_monitor if mode_monitor else sim.add_flux
    mon = add(fcen, 0, 1, mp.FluxRegion(center=mp.Vector3(2.5, 0), size=mp.Vector3(0, 4.0)))
    return sim, mon


def _readers_yee_flux_sim():
    """The same cell, with the centred and the yee_grid mode monitor on ONE region."""
    sim, centred = _readers_flux_sim(symmetric=False, mode_monitor=True)
    yee = sim.add_mode_monitor(0.25, 0, 1, mp.ModeRegion(center=mp.Vector3(2.5, 0),
                                                         size=mp.Vector3(0, 4.0)),
                               yee_grid=True)
    return sim, centred, yee


def _readers_1d_sim(size_z):
    sim = mp.Simulation(resolution=100, force_complex_fields=True,
                        cell_size=mp.Vector3(0, 0, 12.0),
                        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.1), mp.Ex,
                                           mp.Vector3(0, 0, -5.0))],
                        boundary_layers=[mp.PML(1.0)], k_point=False, dimensions=1)
    objects = [sim.add_dft_fields([mp.Ex], 1.0, 0, 1, center=mp.Vector3(0, 0, c),
                                  size=mp.Vector3(0, 0, s), yee_grid=yee)
               for c, s, yee in size_z]
    return sim, objects


# Region requests that exercise every branch of the read-time collapse: a box (no
# collapse), a zero-extent axis whose coordinate is OFF the lattice (two cells and
# real interpolation weights), and a point (both axes collapsed).
_READER_REGIONS = [((0.0, 0.0, 0.0), (1.6, 1.2, 0.0)),
                   ((0.0, 0.234804, 0.0), (1.6, 0.0, 0.0)),
                   ((-0.234804, 0.432973, 0.0), (0.0, 0.0, 0.0))]


def run_readers(case):
    report = {}
    if case == "conductivity_offdiag_is_discarded_by_meep":
        # NOT a lift measurement. It establishes what MEEP does with an off-diagonal
        # conductivity, which is the premise the D-side and B-side refusals rest on.
        def cell(offdiag=(0.0, 0.0, 0.0), diagonal=0.0):
            medium = mp.Medium(epsilon=2.0, D_conductivity=diagonal,
                               D_conductivity_offdiag=mp.Vector3(*offdiag))
            return mp.Simulation(
                cell_size=mp.Vector3(4, 4, 0), resolution=20, k_point=False,
                boundary_layers=[mp.PML(0.5)], default_material=medium,
                sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.4), component=c,
                                   center=mp.Vector3(-1.2, 0))
                         for c in (mp.Ez, mp.Ex, mp.Ey)])

        fields = {}
        for label, kwargs in (("none", {}), ("offdiag", dict(offdiag=(0.5, 0.0, 0.0))),
                              ("diagonal", dict(diagonal=0.5))):
            sim = cell(**kwargs)
            sim.run(until=6.0)
            fields[label] = np.asarray(sim.get_array(component=mp.Ez))
        report["offdiag_vs_none"] = relative(fields["offdiag"], fields["none"])
        report["diagonal_vs_none"] = relative(fields["diagonal"], fields["none"])
        # Which keyword reaches B_conductivity_offdiag depends on the release:
        # python/geom.py:450 assigns it from the D KEYWORD on MEEP 1.33.0 and from
        # the B keyword on 1.34.0. The release rides back for the parent to judge.
        report.update(meep_build())
        keyword_b = mp.Medium(epsilon=2.0,
                              B_conductivity_offdiag=mp.Vector3(0.1, 0.0, 0.0))
        keyword_d = mp.Medium(epsilon=2.0,
                              D_conductivity_offdiag=mp.Vector3(0.1, 0.0, 0.0))
        report["b_keyword_reaches_b"] = [
            float(v) for v in from_meep._vector3(keyword_b.B_conductivity_offdiag)]
        report["d_keyword_reaches_b"] = [
            float(v) for v in from_meep._vector3(keyword_d.B_conductivity_offdiag)]
        return report
    if case in ("dft_array_centred", "dft_array_yee", "dft_array_mirrored"):
        yee = case == "dft_array_yee"
        symmetric = case == "dft_array_mirrored"
        sim, objects = _readers_dft_sim(yee, symmetric, _READER_REGIONS)
        sim.run(until=30.0)
        theirs = [np.asarray(sim.get_dft_array(o, mp.Ez, 0)) for o in objects]
        sim2, objects2 = _readers_dft_sim(yee, symmetric, _READER_REGIONS)
        result = run_on_gpu(sim2, prefer_gpu=False, until=30.0)
        mine = [np.asarray(result.get_dft_array(o, "Ez", 0)) for o in objects2]
        result.close()
        report["shapes_match"] = [list(t.shape) == list(m.shape) for t, m in zip(theirs, mine)]
        report["errors"] = [relative(m, t) if t.shape == m.shape else None
                            for t, m in zip(theirs, mine)]
        report["shapes"] = [list(t.shape) for t in theirs]
        return report
    if case == "dft_array_far_face_1d":
        # The whole-cell request and a face-touching one, against an interior control
        # that was already exact. `_gathered_component` used to drop the Yee-to-centre
        # average for the WHOLE axis when its extended index list was clipped at a
        # non-mirrored far face, which is a half-cell displacement of every sample.
        requests = [((0.0, 12.0, False)), ((3.0, 6.0, False)),
                    ((-3.0, 6.0, False)), ((0.0, 8.0, False)), ((0.0, 12.0, True))]
        sim, objects = _readers_1d_sim(requests)
        sim.run(until=40.0)
        theirs = [np.asarray(sim.get_dft_array(o, mp.Ex, 0)) for o in objects]
        sim2, objects2 = _readers_1d_sim(requests)
        result = run_on_gpu(sim2, prefer_gpu=False, until=40.0)
        mine = [np.asarray(result.get_dft_array(o, "Ex", 0)) for o in objects2]
        result.close()
        report["shapes_match"] = [list(t.shape) == list(m.shape) for t, m in zip(theirs, mine)]
        report["errors"] = [relative(m, t) if t.shape == m.shape else None
                            for t, m in zip(theirs, mine)]
        return report
    if case in ("flux_packer", "flux_packer_mirrored", "flux_packer_reduced"):
        symmetric = case in ("flux_packer_mirrored", "flux_packer_reduced")
        mode_monitor = case != "flux_packer_reduced"
        sim, mon = _readers_flux_sim(symmetric, mode_monitor)
        sim.run(until=40.0)
        report["meep_steps"] = int(sim.fields.t)
        report["meep_time"] = float(sim.meep_time())
        theirs = sim.get_flux_data(mon)
        their_alpha = np.asarray(sim.get_eigenmode_coefficients(
            mon, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y).alpha).ravel()
        report["meep_flux"] = float(mp.get_fluxes(mon)[0])
        sim2, mon2 = _readers_flux_sim(symmetric, mode_monitor)
        result = run_on_gpu(sim2, prefer_gpu=False, until=40.0)
        report["engine_steps"] = int(result.steps)
        report["engine_time"] = float(result.meep_time)
        mine = result.pack_flux_data(mp, mon2)
        # THE CONTROL for taking the measure from MEEP's own chunk: pack the SAME
        # chunk list with this engine's region weights instead, by making the reader
        # of the chunk's s0/s1/e0/e1 hand back nothing so `FluxMonitor.packed` falls
        # back to `_weights * _measure`. The two models agree on the region MEEP was
        # asked for and part company on `S.reduce`'s halved one, whose new edges carry
        # new edge weights — so this must be small on the two full-plane cases and
        # large on the reduced one, wrong at only a handful of sites.
        saved_reader = from_meep._meep_chunk_weight
        from_meep._meep_chunk_weight = lambda chunk, component, directions: None
        try:
            control = result.pack_flux_data(mp, mon2)
        finally:
            from_meep._meep_chunk_weight = saved_reader
        result.close()
        report["E_error"] = relative(np.asarray(mine.E), np.asarray(theirs.E))
        report["H_error"] = relative(np.asarray(mine.H), np.asarray(theirs.H))
        report["E_size"] = int(np.asarray(mine.E).size)
        control_E = np.asarray(control.E)
        their_E = np.asarray(theirs.E)
        report["control_engine_weight_error"] = relative(control_E, their_E)
        report["control_engine_weight_sites_off"] = int(np.count_nonzero(
            ~np.isclose(control_E, their_E, rtol=1e-4, atol=0.0)))
        report["control_engine_weight_ratios"] = sorted({
            round(float(abs(their_E[i] / control_E[i])), 6)
            for i in np.flatnonzero(~np.isclose(control_E, their_E, rtol=1e-4, atol=0.0))
        })
        sim3, mon3 = _readers_flux_sim(symmetric, mode_monitor)
        sim3.init_sim()
        sim3.load_flux_data(mon3, mine)
        loaded_alpha = np.asarray(sim3.get_eigenmode_coefficients(
            mon3, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y).alpha).ravel()
        report["alpha_error"] = relative(loaded_alpha, their_alpha)
        report["alpha_meep"] = [[value.real, value.imag] for value in their_alpha]
        report["flux_error"] = abs(
            mp.get_fluxes(mon3)[0] - report["meep_flux"]) / abs(report["meep_flux"])
        # And the un-loaded control: MEEP's own reader on a simulation nothing was
        # written into returns zero, so a passing alpha cannot be MEEP answering itself.
        sim4, mon4 = _readers_flux_sim(symmetric, mode_monitor)
        sim4.init_sim()
        empty = np.asarray(sim4.get_eigenmode_coefficients(
            mon4, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y).alpha).ravel()
        report["control_unloaded_alpha_max"] = float(np.abs(empty).max())
        return report
    if case in ("force_diagonal", "force_diagonal_decimated"):
        # MEEP's own test_force.py setup, verbatim: a 2-D Ez Gaussian in a 10x10 cell
        # at resolution 20, and the diagonal ForceRegion the test declares. The
        # STOPPING CONDITION is run on the control first and its simulation time reused
        # verbatim, so neither code is measuring the other's turn-on tail.
        decimation = 10 if case == "force_diagonal_decimated" else 1

        def force_sim():
            s = mp.Simulation(
                resolution=20, cell_size=mp.Vector3(10, 10),
                boundary_layers=[mp.PML(1.0)],
                sources=[mp.Source(src=mp.GaussianSource(1.0, fwidth=1.0),
                                   center=mp.Vector3(), component=mp.Ez)],
            )
            region = mp.ForceRegion(mp.Vector3(y=1.27), direction=mp.Y,
                                    size=mp.Vector3(4.38))
            return s, s.add_force(1.0, 0, 1, region, decimation_factor=decimation)

        report.update(meep_build())
        sim, force = force_sim()
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-6))
        report["meep_force"] = float(mp.get_forces(force)[0])
        report["meep_time"] = float(sim.meep_time())
        report["meep_steps"] = int(sim.fields.t)
        sim2, force2 = force_sim()
        result = run_on_gpu(sim2, prefer_gpu=False, until=report["meep_time"])
        report["engine_force"] = float(result.get_forces(force2)[0])
        report["engine_steps"] = int(result.steps)
        report["engine_time"] = float(result.meep_time)
        # The store/load round trip test_force performs, on OUR accumulators: save,
        # scale to nonsense, load back, and the force must be the saved one again.
        monitor = result.monitor_for(force2)
        saved = result.get_force_data(force2)
        monitor.scale_dfts(3.0)
        report["scaled_force"] = float(result.get_forces(force2)[0])
        result.load_force_data(force2, saved)
        report["reloaded_force"] = float(result.get_forces(force2)[0])
        result.close()
        return report
    if case == "energy_folded":
        # MEEP's own test_dft_energy.py setup: a Mirror(Y) run whose EnergyRegion
        # straddles the fold, which is the case add_dft_energy serves through
        # loop_in_chunks' use_symmetry (dft.cpp:723 does NOT call S.reduce).
        def energy_sim():
            s = mp.Simulation(
                resolution=20, cell_size=mp.Vector3(10, 5),
                geometry=[mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                                   material=mp.Medium(epsilon=12))],
                boundary_layers=[mp.PML(1)],
                sources=[mp.EigenModeSource(
                    src=mp.GaussianSource(frequency=0.15, fwidth=0.03),
                    center=mp.Vector3(-3), size=mp.Vector3(y=5), eig_band=1,
                    eig_parity=mp.ODD_Z + mp.EVEN_Y, eig_match_freq=True)],
                symmetries=[mp.Mirror(direction=mp.Y)],
            )
            region = mp.EnergyRegion(center=mp.Vector3(3), size=mp.Vector3(y=5))
            return (s,
                    s.add_energy(0.15, 0, 1, region, decimation_factor=1),
                    s.add_energy(0.15, 0, 1, region, decimation_factor=10))

        report.update(meep_build())
        sim, energy, energy10 = energy_sim()
        sim.run(until_after_sources=100)
        report["meep_time"] = float(sim.meep_time())
        report["meep_steps"] = int(sim.fields.t)
        report["meep_electric"] = float(mp.get_electric_energy(energy)[0])
        report["meep_magnetic"] = float(mp.get_magnetic_energy(energy)[0])
        report["meep_total"] = float(mp.get_total_energy(energy)[0])
        sim2, energy2, energy10_2 = energy_sim()
        result = run_on_gpu(sim2, prefer_gpu=False, until=report["meep_time"])
        report["engine_electric"] = float(result.get_electric_energy(energy2)[0])
        report["engine_magnetic"] = float(result.get_magnetic_energy(energy2)[0])
        report["engine_total"] = float(result.get_total_energy(energy2)[0])
        report["engine_electric_decimated"] = float(result.get_electric_energy(energy10_2)[0])
        report["engine_magnetic_decimated"] = float(result.get_magnetic_energy(energy10_2)[0])
        report["engine_steps"] = int(result.steps)
        report["engine_time"] = float(result.meep_time)
        result.close()
        return report
    if case == "flux_packer_yee_refusal":
        # STOCK MEEP first, on ONE run carrying both monitors: the refusal below is
        # only worth having if the yee_grid monitor answers a different question, and
        # that is MEEP's own claim to make, not this engine's.
        sim, centred, yee = _readers_yee_flux_sim()
        sim.run(until=40.0)
        parity = mp.ODD_Z + mp.EVEN_Y
        for label, mon in (("centred", centred), ("yee", yee)):
            alpha = np.asarray(sim.get_eigenmode_coefficients(
                mon, [1], eig_parity=parity).alpha).ravel()
            report[f"meep_{label}_flux"] = float(mp.get_fluxes(mon)[0])
            report[f"meep_{label}_alpha_abs"] = [float(abs(value)) for value in alpha]
            sites = 0
            chunk = mon.swigobj.E
            while chunk is not None:
                sites += int(chunk.N)
                chunk = chunk.next_in_dft
            report[f"meep_{label}_E_sites"] = sites
        sim2, centred2, yee2 = _readers_yee_flux_sim()
        result = run_on_gpu(sim2, prefer_gpu=False, until=10.0)
        for label, mon in (("centred", centred2), ("yee", yee2)):
            try:
                result.pack_flux_data(mp, mon)
                report[f"{label}_refused"] = False
                report[f"{label}_reasons"] = []
            except MeepSimulationNotLiftable as refusal:
                report[f"{label}_refused"] = True
                report[f"{label}_reasons"] = list(refusal.reasons)
        result.close()
        return report
    raise SystemExit("unknown readers case: " + case)


mode, case, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "parity":
    data = run_case(case)
elif mode == "readers":
    data = run_readers(case)
elif mode == "refusal":
    data = run_refusal(case)
elif mode == "gate":
    data = run_gate(case)
elif mode == "fold_monitors":
    data = run_fold_monitors(case)
else:
    raise SystemExit("unknown mode: " + mode)
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(data, handle)
'''


def _child(tmp_path, mode: str, case: str) -> dict:
    """Run one case in its own process and return the JSON it wrote.

    The subprocess exists for the same platform reason ``test_driver_vs_meep``'s
    oracle does — meep and torch cannot share a process — but does more work
    here: the converter needs a live ``mp.Simulation``, so the lift, both runs
    and the comparison all happen in the child and only numbers come back.
    """
    script_path = tmp_path / "from_meep_case.py"
    script_path.write_text(_CHILD_SCRIPT, encoding="utf-8")
    output_path = tmp_path / f"{mode}_{case}.json"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONPATH"] = os.pathsep.join(
        [_PACKAGE_PARENT] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [sys.executable, str(script_path), mode, case, str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"from_meep child '{mode}/{case}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-3000:]}\nstderr tail:\n{completed.stderr[-3000:]}"
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


# --- cross-validation against CPU MEEP --------------------------------------------

# case -> (parity bound, {control: lower bound the control must EXCEED}).
# Every bound is the engine's established floor for that feature class, loosened
# only by the margin the measured number needs; the measured values are in the
# docstring of test_lifted_simulation_matches_cpu_meep.
_PARITY_BOUNDS = {
    "plain": (1e-6, {}),
    "metallic": (1e-6, {}),
    "metallic_long": (1e-6, {}),
    "metallic_odd_counts": (1e-6, {}),
    "metallic_pml": (1e-6, {}),
    "metallic_mirror_x": (1e-6, {}),
    "metallic_real_fields": (1e-6, {}),
    "metallic_lorentz": (1e-5, {"non_dispersive": 0.1}),
    "pml_uniform": (1e-6, {}),
    "pml_z_low_only": (1e-6, {"high_face": 0.1}),
    # mp.Absorber. The controls are the ways to get a graded D+B conductivity
    # plausibly wrong, and every one of them produces a stable, smooth, absorbing run:
    #   as_pml        — substitute a matched layer of the same thickness
    #   electric_only — build the D ramp and skip the B ramp
    #   shared_sigma  — one sigma volume for all six components instead of six
    # Their bounds are the measured separations backed off by roughly an order.
    "absorber_uniform": (1e-6, {"as_pml": 1e-2, "electric_only": 1e-2,
                                "shared_sigma": 1e-3}),
    "absorber_z_low_only": (1e-6, {"high_face": 0.1}),
    "absorber_and_pml_on_different_faces": (1e-6, {"pml_everywhere": 1e-2}),
    # Measured 6.3253e-07 / 9.6206e-07 standalone (whole-volume complex Ez, until 6);
    # the guard's former refusals, admitted by the f_u mirror + withdraw fixes.
    "source_spans_the_pml": (5e-6, {}),
    "source_on_the_wrapped_pml_seam": (5e-6, {}),
    # Measured 4.9019e-07 standalone; 9.5837e-01 with the withdraw slot stubbed.
    "integrated_source_in_the_pml": (5e-6, {}),
    # Measured 1.1016e-06 standalone; 4.08e+00 was the same configuration's class
    # in the Dcyl sweep before the f_u mirror ran on cylindrical grids.
    "cylindrical_source_in_pml": (5e-6, {}),
    # The former wall refusal, admitted by the metal_high drop in
    # `sources._build_source_points`: the wall ring's edge weight is discarded
    # exactly as MEEP's zero_metal leaves its own wall deposit inert. The
    # bare-wall class measured 7.1e-02 (Ep plane) / 1.00e+00 (last-half-cell
    # point) with the old clamp, floors of 6.9e-07 … 9.6e-07 with the drop.
    "cylindrical_source_on_r_wall": (5e-6, {}),
    # A fractional-cell layer, against the two whole-cell absorbers that would have
    # been substituted for it. The control bounds are the measured separations backed
    # off by roughly an order of magnitude: they are what makes this a test of the
    # fraction rather than of the PML in general. Measured 7.95e-07 with the neighbours
    # at 6.26e-02 / 6.51e-02, and 1.16e-06 at resolution 71 with 8.75e-04 / 9.27e-04 —
    # the separation narrows with the layer's depth in cells, as it must: 35.5 is a
    # smaller fractional perturbation of 35 than 5.5 is of 5.
    "pml_fractional_cells": (1e-6, {"rounded_down": 1e-2, "rounded_up": 1e-2}),
    # The non-half-cell case. Its job is the prefac DIVISOR, which the two cases above
    # cannot see: both sit at 0.5*N == c exactly. Substituting the snapped extent for
    # the raw thickness in prefac (a 4.0000% peak-sigma shift here) takes this case from
    # 7.34e-07 to 6.80e-03 while leaving both of those two at their own floors — measured.
    "pml_fractional_cells_uneven": (1e-6, {"rounded_down": 1e-2, "rounded_up": 1e-2}),
    "pml_fractional_cells_res71": (5e-6, {"rounded_down": 1e-4, "rounded_up": 1e-4}),
    "bloch": (1e-6, {"k_zero": 0.1}),
    # Measured 2.82e-07 against CPU MEEP; the control (same run without the x
    # absorber) is the run a converter that dropped the phase would produce.
    "bloch_on_pml_axis": (5e-6, {"no_x_absorber": 1e-3}),
    # Reduced dimensions. The bounds are the engine's own 3-D floors for the same
    # feature class, which is the claim being made: a 2-D run is not an approximation
    # of anything, it is the same arithmetic over one fewer axis.
    "two_d_tm": (1e-6, {"amplitude_x_resolution": 0.5}),
    "two_d_te": (1e-6, {}),
    "two_d_metallic_tm": (1e-6, {}),
    "two_d_metallic_te": (1e-6, {}),
    "two_d_pml": (1e-6, {}),
    "two_d_pml_on_the_invariant_axis": (1e-6, {}),
    "two_d_lorentz": (1e-5, {"non_dispersive": 0.1}),
    "two_d_mirror_x": (1e-6, {}),
    "two_d_bloch": (1e-6, {"k_zero": 0.1}),
    # special_kz. The Ex pair measures the TE-from-TM coupling alone and its control
    # is the same run at -beta (measured separations ~2.0, i.e. the field negated);
    # the Ez pair measures the driven component and its control is the run with no
    # beta at all. Both storage modes carry the same bound because real storage is
    # not an approximation of complex here — it is MEEP's implicit-i bookkeeping over
    # the identical arithmetic.
    "special_kz_complex": (1e-6, {"beta_sign_flipped": 0.5}),
    "special_kz_real_imag": (1e-6, {"beta_sign_flipped": 0.5}),
    "special_kz_complex_driven": (1e-6, {"no_beta": 0.1}),
    "special_kz_real_imag_driven": (1e-6, {"no_beta": 0.1}),
    "two_d_structured": (1e-6, {"no_geometry": 0.1}),
    # BFAST. Bar and control floors are the measured numbers backed off; the two
    # controls are the same physical mistake (a normal-incidence run) reached from
    # the reference side and from the engine side. See the case's own comment.
    "bfast_fixed_angle": (5e-6, {"no_bfast": 0.5, "bfast_dropped_on_lift": 0.5}),
    # BFAST + a nonzero Bloch k, the pair MEEP allows and the corpus never pairs.
    # Measured 3.81e-07 on Ex (Ez 7.05e-07, Hy 6.70e-07 on the same run) with the
    # shear-dropped control at 2.41e-01 on Ex.
    "bfast_with_bloch_k": (5e-6, {"no_bfast": 0.1}),
    # Measured 6.31e-07 (wall rows in the comparison; no_geometry control
    # 9.56e-01); before the wall-plane exclusion this case did not lift at all —
    # the spread check refused the declined-default vacuum below the SiO2
    # background.
    "structured_nonvacuum_background": (1e-6, {"no_geometry": 0.1}),
    "two_d_odd_counts": (1e-6, {}),
    "two_d_declared_with_z_extent": (1e-6, {}),
    # Measured 6.23e-07 with the peak ratio 1.0000; the plain-sheet control is the
    # degraded lift the old refusal warned about and sits at O(1).
    "eigenmode_source_waveguide": (5e-6, {"plain_ez_sheet": 0.1}),
    "eigenmode_source_waveguide_te": (5e-6, {}),
    # special_kz through an eigenmode source. Bounds are the eigenmode class's own
    # 5e-06; the controls are the two beta-specific ways to inject a smooth wrong mode.
    "eigenmode_source_special_kz": (5e-6, {"full_3d_sample_point": 1e-3}),
    "eigenmode_source_special_kz_real_imag": (5e-6, {"no_kz_phasefix": 0.1,
                                                     "full_3d_sample_point": 1e-3}),
    # The folded TM launch, on the same 5e-6 its unfolded twin holds — the claim being
    # that a fold costs a magnetic sheet nothing. Its control is the step-order defect
    # restored (1.27e-02), and the bar is deliberately only 1e-4: one stale ghost row
    # is a SMALL number next to every other control in this file — the smallest here by
    # two orders — which is the whole reason it went unseen. That is what the case
    # is for.
    #
    # THE PARITY NUMBER JITTERS AND THE BAR IS SET FOR IT: six repeat runs measured
    # 2.19 / 2.36 / 2.42 / 2.51 / 3.71 / 3.93 e-07 where the unfolded twin reproduces
    # 6.22876257082791e-07 to every digit. The MPB solve is iterative and its
    # convergence is not bit-reproducible run to run (the sheet profile moves in its
    # last digits), and the folded run's Hy reference norm is the smaller of the two,
    # so it shows. 5e-6 leaves 13x headroom over the worst seen; the control's own
    # spread is in the 8th digit (1.2728e-02), so the separation is never in doubt.
    "eigenmode_source_waveguide_folded": (5e-6, {"b_ghosts_filled_before_the_source": 1e-4}),
    # Oblique (NO_DIRECTION). The bar is the same 5e-6 the normal-launch eigenmode
    # cases sit on — the synthesis is byte-identical, only the solve differs — and
    # `axis_solve` is the control that earns the case: it is the natural wrong
    # implementation, and on the waveguide it produces a complete, converged,
    # right-frequency field that is four decades away from parity.
    "eigenmode_oblique_planewave": (5e-6, {"axis_solve": 0.1, "no_phase": 0.1,
                                           "plain_ez_sheet": 1.0}),
    "eigenmode_oblique_waveguide": (5e-6, {"axis_solve": 0.1, "no_phase": 0.1,
                                           "plain_ez_sheet": 0.5}),
    # Gaussian beams. The 3-D bar is 1e-6 (measured 2.75e-07); the 2-D TE case sits at
    # 2.38e-06, one Yee-offset class higher, so it takes 5e-6 like the eigenmode cases.
    # `hx_is_ey` is the control that earns the whole feature: it is the reading a
    # careful engineer writes down, and 2.0e-03 against 2.8e-07 is what says the
    # comparison can see it.
    "beam_two_d_tm": (1e-6, {"flip_sheet_signs": 0.5, "reduce_beam_vectors": 1.0,
                             "plain_ez_sheet": 0.5}),
    "beam_two_d_te": (5e-6, {"flip_other_sheet_signs": 0.5, "plain_ey_sheet": 0.5}),
    "beam_three_d": (1e-6, {"hx_is_ey": 1e-4, "flip_sheet_signs": 0.5,
                            "plain_ex_sheet": 0.5}),
    # THE SAME BEAM ON REAL STORAGE, CW and pulsed. The bar is the 2-D beam's own 1e-6:
    # the claim is that `real(A)` costs nothing, so the real case has to hold the bar
    # its complex twin holds, not a looser one. `drop_beam_phase` is the control that
    # earns both — `Re(a)*s` in place of `Re(a*s)`, the one-line "make the amplitude
    # real" fix, which keeps the envelope and launches the wrong wavefront.
    "beam_real_fields": (1e-6, {"drop_beam_phase": 0.1, "plain_ez_sheet": 0.1}),
    "beam_real_fields_pulse": (1e-6, {"drop_beam_phase": 0.1, "plain_ez_sheet": 0.1}),
    # mp.GaussianBeam2DSource — MEEP's own Python synthesis, called rather than
    # transcribed, so the bar is the plain-source floor. The controls are the half
    # deposit (Love's equivalence needs BOTH currents) and the 3-D approximation this
    # class exists to replace.
    "gaussian_beam_2d": (1e-6, {"electric_current_only": 0.1, "beam_3d_instead": 0.1}),
    # mp.metal / mp.perfect_electric_conductor — MEEP's -1e20 permittivity sentinel,
    # carried through unchanged. `pec_installed_as_vacuum` is the naive relaxation,
    # which runs to completion with the full signal and the same step count.
    # A diagonal anisotropic D_conductivity, on the plain-source floor. The two
    # controls are the pre-anisotropic engine's shared volume, taken from either of two
    # equally arbitrary components.
    "anisotropic_conductivity": (1e-6, {"conductivity_shared_from_x": 0.02,
                                        "conductivity_shared_from_y": 0.02}),
    # B_conductivity — the MAGNETIC half of MEEP's one conductivity term, on the
    # plain-source floor. Measured 8.7850e-07 with the magnetic loss dropped at
    # 1.4756e-01 and the same sigma installed on the D side at 1.1329e-01; the
    # anisotropic twin 5.6547e-07 against 2.0611e-01 / 1.0452e-01 for a shared volume
    # taken from either of two equally arbitrary components; and inside an mp.Absorber
    # 7.3982e-07 against 7.8083e-02 when the material's declaration REPLACES the face
    # ramp instead of adding to it.
    "b_conductivity": (1e-6, {"b_conductivity_dropped": 1e-2,
                              "installed_on_the_d_side": 1e-2}),
    "anisotropic_b_conductivity": (1e-6, {"b_shared_from_y": 1e-2,
                                          "b_shared_from_z": 1e-2,
                                          "b_conductivity_dropped": 1e-2}),
    "b_conductivity_in_an_absorber": (1e-6, {"b_ramp_overwritten": 1e-2,
                                             "b_conductivity_dropped": 1e-2}),
    # A STRUCTURED D_conductivity, read per point off MEEP's D-row chi1inv. Measured
    # 3.3888e-07 against 1.2185e-02 with the volume zeroed — four and a half orders,
    # and the control run is complete, smooth and the same length.
    "structured_conductivity": (1e-6, {"conductivity_dropped": 1e-3}),
    # The MaterialGrid damping, read out of the same array MEEP puts it in. Measured
    # 3.1668e-07 against 1.6377e-02 with the volume zeroed, and 8.4773e-07 for the
    # identical cell at damping = 0 — so the lifted damped run is at the undamped
    # cell's own floor.
    "material_grid_damping": (1e-6, {"conductivity_dropped": 1e-3}),
    # The absorber composition. Measured 2.2353e-07 against 6.0919e-02 when the ramp is
    # added a second time.
    "structured_conductivity_in_an_absorber": (1e-6, {"absorber_ramp_added_twice": 1e-2}),
    # A source through two LOSSLESS PML layers with the loss confined to the interior.
    # Measured 2.5964e-07, against a lossless control on the same cell at 3.3453e-07 and
    # 1.6419e-02 for the same source when the loss reaches into the layer — which is
    # still refused, and is what `_conductivity_reaches_a_pml` separates.
    "structured_conductivity_source_through_a_lossless_pml": (1e-6, {}),
    # A callable returning poles MEEP never registers. Measured 5.3347e-07; installing
    # them anyway steps a dispersive cell MEEP is stepping non-dispersive.
    "material_function_dispersive_dropped": (1e-6, {"dispersion_installed_anyway": 1e-2}),
    "pec_cavity_2d": (1e-6, {"pec_installed_as_vacuum": 0.1}),
    "pec_box_3d": (1e-6, {"pec_installed_as_vacuum": 0.1}),
    # The slab case's control bound is lower than its siblings' on purpose: the metal
    # here is a slab in an otherwise empty cell rather than a cavity WALL, so removing
    # it changes less of the field (measured 1.72e-01 against 7.39e-01 and 3.60e-01).
    "pec_touching_pml": (1e-6, {"pec_installed_as_vacuum": 0.05}),
    "pec_uniform_cell": (1e-6, {}),
    # The oblique eigenmode half, on the same 5e-6 its complex twin holds.
    "eigenmode_oblique_waveguide_real_fields": (5e-6, {"axis_solve": 0.1,
                                                       "no_phase": 0.1}),
    "one_d_metallic": (1e-6, {"amplitude_x_resolution_squared": 0.5}),
    "one_d_pml": (1e-6, {}),
    "unit_axes_three_d": (1e-6, {}),
    # The zero-extent axis WITH a Bloch phase on it. Measured 2.50e-07 against CPU
    # MEEP 1.33.0; 1.9597e-01 with the collapse of the one-cell axis taking one plane
    # instead of the mean of the two the slice straddles — tan(pi*kx*dx) = 1.9597e-01
    # at resolution 10, to five digits — which is the defect this case exists for.
    # The k_zero control is the run a converter that dropped the phase would produce.
    "unit_axes_three_d_bloch": (1e-6, {"k_zero": 0.1}),
    "unit_axes_three_d_structured": (1e-6, {"no_geometry": 0.1}),
    "mirror_z_odd": (1e-6, {}),
    "mirror_x_odd": (1e-6, {}),
    "mirror_y_even": (1e-6, {}),
    # Off-plane sources under a fold. The bound is the fold's own floor, which is the
    # claim: an off-plane source is not an approximation of anything, it is the same
    # arithmetic with the image supplied by the fold instead of by the caller.
    # Measured on a 10x6 probe cell before these landed: MEEP folded against MEEP's
    # unfolded parity-signed pair 1.6e-07..3.1e-07, and this engine against MEEP's
    # folded answer 3.8e-07..9.3e-07, across Ez/Ex/Ey at both phases.
    "mirror_x_even_off_plane": (1e-6, {}),
    "mirror_x_even_off_plane_odd_component": (1e-6, {}),
    # Measured 7.95e-07 / 1.10e-06 / 8.17e-07 standalone (whole-volume complex,
    # until 4); the corpus folded-live-face family read 9.5e-03 … 6.0e-02 before
    # the lattice-image pass in `sources._build_source_points` (mutations
    # `folded_live_lattice_images_dropped`, `folded_live_overhang_clamped_back`).
    "full_width_source_on_folded_live_face_even": (5e-6, {}),
    "full_width_source_on_folded_live_face_odd": (5e-6, {}),
    "full_width_source_on_folded_live_face_odd_shift0": (5e-6, {}),
    # The ODD folded-periodic top cell (MEEP's `big_corner`, one half-cell above
    # the second mirror). Bounds are the fold's own floor — measured 3.07e-07 /
    # 3.38e-07 / 3.51e-07 — where before the cell was stored the two odd cases
    # read 1.0097e-02 and 9.9514e-01 and the even control was unchanged to the
    # last digit. See the CASES entries for the MEEP citations.
    "odd_fold_absorber_on_the_folded_axis": (5e-6, {}),
    "odd_fold_material_step_under_the_window_top": (5e-6, {}),
    "even_fold_material_step_under_the_window_top": (5e-6, {}),
    "mirror_x_pair_written_longhand": (1e-6, {}),
    "lorentz": (1e-5, {"non_dispersive": 0.1}),
    "drude": (1e-5, {"non_dispersive": 0.1}),
    "conductivity": (1e-5, {"lossless": 0.05}),
    "anisotropic": (1e-6, {"isotropic_ez": 0.05}),
    "chi3": (1e-6, {"linear": 0.05}),
    "chi2": (1e-6, {"linear": 0.005}),
    "gaussian_source": (1e-6, {}),
    "gaussian_source_width_spelling": (1e-6, {}),
    "gaussian_until_after_sources": (1e-6, {}),
    # Measured 1.86e-07 (320 steps matched both sides); the abs()-lift control —
    # both carriers positive, a different current entirely (one complex rotation
    # where the pair beats into a real sine) — measured 5.26e-01 away.
    "gaussian_pm_pair": (1e-6, {"positive_pair": 0.1}),
    "magnetic_source": (1e-6, {}),
    "amp_func_sheet": (1e-6, {"flat_sheet": 0.05}),
    "amp_data_sheet": (1e-6, {"flat_sheet": 0.05}),
    "complex_amplitude": (1e-6, {"real_amplitude": 0.1}),
    "integrated_source": (1e-5, {"not_integrated": 0.1}),
    "custom_source": (1e-6, {}),
    "continuous_ramp": (1e-6, {"no_ramp": 0.05}),
    "odd_cell_counts": (1e-6, {}),
    "real_fields": (1e-6, {}),
    "courant_non_default": (1e-6, {"default_courant": 0.05}),
    "resolution_non_default": (1e-6, {}),
    "structured_slab": (1e-6, {"no_geometry": 0.1, "half_cell_shifted": 0.005}),
    # The per-point epsilon routes, held to the SAME structured floor as the geometry
    # spelling of the same slab above — not a widened one. They read exactly the same
    # chi1inv volumes; only MEEP's way of filling them differs.
    "material_function_slab": (1e-6, {"no_geometry": 0.1, "half_cell_shifted": 0.005}),
    "epsilon_func_slab": (1e-6, {"no_geometry": 0.1}),
    "epsilon_array_slab": (1e-6, {"no_geometry": 0.1, "half_cell_shifted": 0.005}),
    "structured_three_media": (1e-6, {"one_medium": 0.05}),
    "structured_anisotropic": (1e-6, {"isotropic_ez": 0.01}),
    "structured_curved_no_averaging": (1e-6, {"no_geometry": 0.1}),
    # Curved/rotated geometry with the averaging ON — off-diagonal chi1inv read
    # back and stepped. The point_sampled control is the same geometry without
    # the averaging: the run a lift that dropped the tensor would resemble.
    "two_d_cylinder_smoothed": (5e-6, {"point_sampled": 5e-3, "no_geometry": 0.1}),
    "two_d_rotated_block_smoothed_te": (5e-6, {"point_sampled": 5e-3}),
    "uniform_epsilon_offdiag": (5e-6, {"diagonal_only": 0.05}),
    "two_d_cylinder_smoothed_mirror": (5e-6, {"point_sampled": 5e-3}),
    "uniform_epsilon_offdiag_mirror": (5e-6, {"diagonal_only": 0.01}),
    "structured_odd_cells": (1e-6, {"half_cell_shifted": 0.001}),
    "structured_pml": (1e-6, {}),
    "structured_mirror": (1e-6, {}),
    "structured_bloch": (1e-6, {"k_zero": 0.1}),
    "structured_shared_lorentz": (1e-5, {"non_dispersive": 0.05}),
    # MaterialGrid, at the ordinary structured floor. Measured 2.9431e-07 point-sampled,
    # 2.5961e-07 smoothed, 2.5445e-07 projected and 2.0515e-07 as default_material,
    # against 2.7525e-07 for a plain-medium control on the same cell — so 1e-6 is the
    # same bar every other structured case is held to, not a widened one.
    #
    # `half_cell_shifted` is the load-bearing control: the declared-media span check
    # inside _lift_epsilon_structured is skipped for a smoothed grid (MEEP's own voxel
    # average leaves the endpoint bracket), and this is what takes over its job. The
    # bar is set well above the parity number and well below the control's measured
    # distance, so a lift that sampled half a cell off would match the control instead.
    "material_grid_point_sampled": (1e-6, {"no_grid": 0.1, "half_cell_shifted": 0.002,
                                           "endpoint_medium_only": 0.05}),
    "material_grid_smoothed": (1e-6, {"point_sampled": 0.001, "half_cell_shifted": 0.002}),
    "material_grid_projected": (1e-6, {"unprojected": 0.005}),
    "material_grid_default_material": (1e-6, {"vacuum_default": 0.05,
                                              "endpoint_medium_only": 0.05}),
}


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", sorted(_PARITY_BOUNDS))
def test_lifted_simulation_matches_cpu_meep(tmp_path, case):
    """One ``mp.Simulation``, lifted and stepped here, must reproduce MEEP stepping it.

    Complex relative L2 over the WHOLE volume in MEEP's own ``get_array`` layout,
    so a global phase error, a half-cell registration slip and a dropped boundary
    plane are all visible — magnitudes alone would hide every one of them.

    Measured (this host, MEEP 1.29.0, NumPy backend), against the engine's
    established floors of 2.41e-07 (no PML), 4.00e-07 (uniform PML),
    2.3e-07…6.3e-07 (Bloch), 3.2e-06 (Lorentz), 4.6e-07 (Drude), 2.7e-07
    (conductivity) and 1.2e-07…1.7e-07 (diagonal epsilon). Re-measured in full
    at the 2026-08-06 re-baselining against pristine MEEP 1.33.0 single: all 87
    cases within a few percent of the digits below, except integrated_source
    (1.13e-07 → 3.02e-07 — engine-side, the source-in-absorber withdraw rework)
    and the eigenmode waveguide pair (6.23e-07/3.13e-07 → 5.81e-07/3.58e-07,
    riding the #3163/#3167 MPB-path version splits). Parity first, then
    the control it must NOT match::

        plain                           2.33e-07
        pml_uniform                     4.00e-07   (the uniform-PML floor exactly)
        pml_z_low_only                  4.66e-07   high_face        2.15e-01
        pml_fractional_cells            7.95e-07   rounded_down     6.26e-02
                                                   rounded_up       6.51e-02
        pml_fractional_cells_uneven     7.34e-07   rounded_down     2.03e-02
                                                   rounded_up       1.46e-01
        pml_fractional_cells_res71      1.16e-06   rounded_down     8.75e-04
                                                   rounded_up       9.27e-04
        bloch                           2.90e-07   k_zero           8.76e-01
        metallic                        2.22e-07
        metallic_long                   2.35e-07   (t = 6, many wall reflections)
        metallic_odd_counts             4.24e-07
        metallic_pml                    7.52e-07
        metallic_mirror_x               2.05e-07   (folded AND walled, far face live)
        metallic_real_fields            2.09e-07
        metallic_lorentz                3.40e-06   non_dispersive   1.09e+00
        mirror_z_odd                    1.68e-07
        mirror_x_odd                    1.68e-07
        mirror_y_even                   1.67e-07
        lorentz                         3.41e-06   non_dispersive   1.11e+00
        drude                           9.32e-07   non_dispersive   1.26e+00
        conductivity                    1.60e-07   lossless         3.43e-01
        anisotropic                     1.53e-07   isotropic_ez     5.21e-01
        chi3                            1.99e-07   linear           1.15e-01
        chi2                            2.29e-07   linear           1.61e-01
        gaussian_source                 1.19e-07
        gaussian_source_width_spelling  1.19e-07   (identical to the fwidth case)
        gaussian_until_after_sources    2.24e-07
        magnetic_source                 2.62e-07
        amp_func_sheet                  2.25e-07   flat_sheet       1.13e+00
        amp_data_sheet                  1.88e-07   flat_sheet       1.03e+00
        complex_amplitude               2.52e-07   real_amplitude   9.43e-01
        integrated_source               1.13e-07   not_integrated   1.59e-01
        custom_source                   1.13e-07
        odd_cell_counts                 1.43e-07
        real_fields                     2.45e-07
        courant_non_default             2.27e-07   default_courant  2.53e-01
        resolution_non_default          4.70e-07

    And the structured cells, where MEEP rasterizes and smooths the geometry and
    the lift reads its per-component ``chi1inv`` back::

        structured_slab                 2.53e-07   no_geometry      4.16e-01
                                                   half_cell_shifted 3.46e-01
        material_function_slab          3.60e-07   no_geometry      4.73e-01
                                                   half_cell_shifted 3.84e-01
        epsilon_func_slab               3.60e-07   no_geometry      4.73e-01
        epsilon_array_slab              2.53e-07   no_geometry      4.16e-01
                                                   half_cell_shifted 3.38e-01
        structured_three_media          1.69e-07   one_medium       5.79e-01
        structured_anisotropic          2.60e-07   isotropic_ez     1.93e-01
        structured_curved_no_averaging  2.37e-07   no_geometry      1.13e+00
        structured_odd_cells            1.89e-07   half_cell_shifted 1.22e-01
        structured_pml                  5.84e-07
        structured_mirror               1.07e-07
        structured_bloch                1.76e-07   k_zero           1.94e-01
        structured_shared_lorentz       1.05e-06   non_dispersive   1.30e+00

    And the TENSOR cells — curved and rotated geometry with MEEP's anisotropic
    averaging ON, the off-diagonal chi1inv rows read back and stepped::

        two_d_cylinder_smoothed         3.36e-07   point_sampled    3.99e-01
                                                   no_geometry      5.01e-01
        two_d_rotated_block_smoothed_te 6.99e-07   point_sampled    1.94e-01
        uniform_epsilon_offdiag         3.84e-07   diagonal_only    8.73e-01
        two_d_cylinder_smoothed_mirror  3.20e-07   point_sampled    3.89e-01
        uniform_epsilon_offdiag_mirror  1.54e-07   diagonal_only    1.45e-01

    The two _mirror rows are the corpus configuration (ring.py and eight more:
    no k_point, so MEEP's PEC box; PML; mp.Mirror over the cylinder) — the fold
    carries the tensor's ODD coupling parity in the first and the EVEN one, live
    ON the plane, in the second. Both sit on the engine's metallic-fold floor
    because a metallic folded far face is exact; under the periodic default the
    same cylinder case instead measures the fold's documented zero-ghost
    far-face limitation (7.2e-04, tensor-independent, the far-face guard's
    regime) — which is why these cases pin the boundary they do.

    And the REDUCED-DIMENSION cells, where MEEP drops an axis entirely and returns
    an array of lower rank. Both polarizations are listed because a boundary or an
    amplitude that is right for TM and wrong for TE is invisible in either alone::

        two_d_tm                        1.58e-07   amplitude_x_resolution  9.00e-01
        two_d_te                        2.87e-07
        two_d_metallic_tm               1.44e-07
        two_d_metallic_te               2.77e-07
        two_d_pml                       2.98e-07
        two_d_pml_on_the_invariant_axis 2.98e-07   (bit-identical to two_d_pml)
        two_d_lorentz                   5.53e-07   non_dispersive   1.21e+00
        two_d_mirror_x                  1.56e-07
        two_d_bloch                     1.44e-07   k_zero           1.09e+00
        two_d_structured                3.72e-07   no_geometry      1.08e+00
        two_d_odd_counts                1.36e-07
        two_d_declared_with_z_extent    1.58e-07   (identical to two_d_tm)
        eigenmode_source_waveguide      6.23e-07   plain_ez_sheet   1.03e+00
        eigenmode_source_waveguide_te   3.13e-07   (the other sheet pair: Ey+Hz)
        eigenmode_source_waveguide_..   2.2e-07    b_ghosts_early   1.27e-02
          ..._folded (compared on Hy)    ..3.9e-07  over six runs; see the note below

    And the SYNTHESIZED sources, where MEEP builds the current distribution at setup
    and the lift re-runs that synthesis rather than reading the deposited current
    back. Their controls are LIFT-side degradations, not reference-side overrides:
    the MEEP input is unchanged and the converter is broken on purpose, because that
    is the only way to measure a mistake a converter makes rather than a different
    physical run::

        eigenmode_oblique_planewave     7.45e-07   axis_solve       MEEP ABORTS
                                                   no_phase         1.33e+00
                                                   plain_ez_sheet   1.76e+01
        eigenmode_oblique_waveguide     1.22e-06   axis_solve       4.08e-01
                                                   no_phase         1.47e+00
                                                   plain_ez_sheet   2.01e+00
        beam_two_d_tm                   4.43e-07   flip_sheet_signs 2.00e+00
                                                   reduce_beam_vec  8.39e+00
                                                   plain_ez_sheet   9.91e-01
        beam_two_d_te                   2.38e-06   flip_other_signs 2.00e+00
                                                   plain_ey_sheet   1.48e+00
        beam_three_d                    2.75e-07   hx_is_ey         2.02e-03
                                                   flip_sheet_signs 2.00e+00
                                                   plain_ex_sheet   1.52e+00

    And the same two synthesized sources on REAL float32 storage — MEEP's default,
    and what a script that declares no ``force_complex_fields`` gets. Both carry a
    spatial amplitude that is complex BY CONSTRUCTION (the beam's wavefront curvature
    and Gouy phase; the oblique mode's in-plane ramp), which the injection layer used
    to refuse rather than project. It now transcribes MEEP's own unconditional
    ``real(A)`` (step.cpp:307), and these are the cases that say what that is worth::

        beam_real_fields                5.37e-07   drop_beam_phase  6.60e-01
                                                   plain_ez_sheet   1.24e+00
        beam_real_fields_pulse          2.25e-07   drop_beam_phase  7.01e-01
                                                   plain_ez_sheet   1.14e+00
        eigenmode_oblique_..._real_..   1.19e-06   axis_solve       3.71e-01
                                                   no_phase         1.45e+00

    Each sits on the bound its COMPLEX twin sits on, which is the claim: ``real(A)``
    is not a lossy reduction of the complex run, it is that run's real part. Measured
    directly rather than inferred — on the same cell, this engine's real array equals
    ``Re(`` its complex array ``)`` to exactly 0.0, and so does CPU MEEP's. The pulsed
    case is not redundant with the CW one: the argument for projecting is that
    ``Re(A e^{-iwt})`` turns a spatial phase into a per-point time shift, which is the
    right launch at monochromatic steady state — a GaussianSource is neither, and the
    numbers were sampled from t = 0.5 upward for the same reason.

    THE EXACT 2-D BEAM, which is a third mechanism rather than a variant of the two
    above: ``mp.GaussianBeam2DSource`` performs its whole Hankel-function synthesis in
    Python (``get_fields`` -> ``green2d``, source.py:821-1044) and deposits an ordinary
    ``mp.Source`` PAIR carrying ``amp_data``, so this lift CALLS MEEP's synthesis
    rather than transcribing it::

        gaussian_beam_2d                2.81e-07   electric_current_only 6.89e-01
                                                   beam_3d_instead       4.38e+00

    ``electric_current_only`` is the control that earns it. ``get_equiv_sources``
    returns Love's equivalent PAIR — ``K = nHat x H`` and ``N = -nHat x E``
    (source.py:786-794) — and MEASURED on this cell, keeping only the electric half
    leaves a run with the full signal and the same step count that is 6.89e-01 from
    MEEP: the sheet radiates in BOTH directions at half amplitude. ``beam_3d_instead``
    is the 3-D complex-point-source approximation on the same declaration, 4.38e+00
    away, which is what says this case measures the 2-D synthesis and not "a beam".

    MEEP'S OWN ASSERTIONS on this source are reproduced separately and are the
    stronger evidence: ``test_gaussianbeam.py::TestGaussianBeamSource.test_gaussian_beam``
    drives 2/2 through this engine (10 000 steps), with MEEP's focus-energy fraction
    coming out at 0.98630 (the 2-D beam) and 0.99062 (the 3-D one) against its own
    ``assertGreater(frac, 0.98)``.

    A DIAGONAL ANISOTROPIC ``D_conductivity``, which needed no new per-step work at
    all — ``Fields._set_conductivity_side`` has always stored sigma, condfac and
    condinv PER COMPONENT (it is what ``mp.Absorber`` installs) and MEEP reads
    ``D_conductivity_diag.x/.y/.z`` for Dx/Dy/Dz (``get_cnd``, meepgeom.cpp:1545-1559).
    Only the driver's public setter funnelled the three into one volume::

        anisotropic_conductivity        4.81e-07   conductivity_shared_from_x 3.38e-01
                                                   conductivity_shared_from_y 1.11e-01

    The declared sigma is (0.0, 1.0, 0.4) — three different values, one of them zero,
    so a lift that read any single component is caught and the zero component stays on
    the bit-identical lossless path while its neighbours are lossy. Both controls are
    the pre-anisotropic engine restored, differing only in which arbitrary component
    the shared volume is taken from. Whole-volume over all three E components the
    parity is 3.95e-07, against isotropic calibrations of 5.01e-07 (sigma 1.0),
    3.09e-07 (0.4) and 3.58e-07 (lossless) on the same cell — the anisotropic path
    sits on the floor the isotropic one already held. ``D_conductivity_offdiag``
    stays refused (``rotated_conductivity``) and genuinely has no diagonal form.

    THE PERFECT ELECTRIC CONDUCTOR, which is not a fourth mechanism at all and that is
    the point. ``mp.metal`` is ``Medium(epsilon=-inf)`` where MEEP's ``inf`` is the
    literal 1.0e20, so the medium carries epsilon = -1e20; it stays
    ``material_data::MEDIUM``, ``material_epsmu`` inverts it on the ORDINARY
    ``sym_matrix_invert`` branch (meepgeom.cpp:785-796) to chi1inv = -1e-20, and the
    ordinary ``E = (D - sum P) * inv_eps`` drives E to ~0 there. The lift carries
    MEEP's own number through unchanged — no dispatch, no mask, no substituted zero::

        pec_cavity_2d                   3.67e-07   pec_installed_as_vacuum 7.39e-01
        pec_box_3d                      2.55e-07   pec_installed_as_vacuum 3.60e-01
        pec_touching_pml                4.39e-07   pec_installed_as_vacuum 1.72e-01
        pec_uniform_cell                1.47e-07   (no control)

    ``pec_installed_as_vacuum`` is the whole reason the refusal existed and the reason
    it is relaxed BY VALUE rather than by sign: substituting the background for the
    unreadable cells is the obvious repair, and it produces a complete run with the
    full signal and the same step count that only the parity number distinguishes.
    The separate ``negative_epsilon`` and ``perfect_magnetic_conductor`` refusal cases
    are the other half of that: epsilon = -3 still has no stable leapfrog, and
    ``mp.perfect_magnetic_conductor`` is the same -1e20 sentinel on the mu side, where
    the constitutive machinery is not built.

    ``pec_uniform_cell`` is the degenerate member — every cell a perfect conductor,
    which takes the UNIFORM epsilon path rather than the structured one. Both engines
    step fields of order 1e-19 there and they agree on them to 1.47e-07; "everything is
    ~0" is not the same claim as "the two ~0s are the same ~0". It is included because
    that path used to be accepted by the gate and then raise inside ``_lift_epsilon``.

    ``drop_beam_phase`` is the control that earns the beam pair. It is not the same
    mistake as ``no_phase``: rather than discarding the phase outright it makes the
    amplitude real BEFORE the product, ``Re(a)*s`` in place of ``Re(a*s)`` — the
    one-line "real storage cannot carry a complex amplitude" fix, which keeps the
    envelope, keeps the beam looking like a beam, and launches the wrong wavefront.

    ``axis_solve`` is the sharpest control in this file. MEEP's NO_DIRECTION splits
    the sheet NORMAL from the SOLVE direction, and collapsing them is the natural
    one-line implementation. On the planewave it aborts inside MEEP (its ``eig_vol``
    is a one-pixel slab, and only the oblique branch skips the containment check,
    mpb.cpp:371) — asserted as an abort. On the waveguide, whose ``eig_vol`` defaults
    to ``where``, it CONVERGES: same frequency to 8 digits, group velocity within
    0.7%, peak |Ez| within 1%, and 4.08e-01 wrong. Nothing cheap tells it from a
    working lift.

    ``hx_is_ey`` is the beam's equivalent. MEEP reads an uninitialized buffer slot as
    ``gb_Hx`` (sources.cpp:691/:716) and deposits exactly zero; writing the field it
    was plainly meant to be is what a careful engineer does, and it costs 0.74% at
    the sheet — four orders above this package's parity band and entirely invisible
    to any suite built around gaussian-beam.py's own E0 = zhat, which is why
    ``beam_three_d`` exists. The two sign-flip spellings are both here because each
    polarization drives only ONE pair of the four sheets: ``flip_sheet_signs`` is
    2.00e+00 in the TM and 3-D cases and invisible in the TE one, and
    ``flip_other_sheet_signs`` is the reverse.
        one_d_metallic                  1.75e-07   amplitude_x_res^2 9.90e-01
        one_d_pml                       3.71e-07
        unit_axes_three_d               1.74e-07   (3-D, x and y one cell each)
        unit_axes_three_d_structured    2.16e-07   no_geometry      7.96e-01

    Every parity number is between 1.1e-07 and 3.4e-06; every control is between
    1.1e-01 and 1.3e+00. There is no case in this table where the two are within
    five orders of magnitude of each other.

    The ``metallic_*`` cases carry no explicit control because the SHAPE assertion
    below is already one, and a sharper one than any field comparison: with
    ``k_point`` unset MEEP returns N points per axis and with it N + 1, so a lift
    that stepped the periodic problem cannot line up at all. The field-level control
    — the two boundary conditions being 1.28e+00 apart on the same cell — is measured
    directly in ``test_driver_vs_meep.py``.

    ``half_cell_shifted`` is the control that matters most for the structured
    cases: the same geometry moved by dx/2, which is a perfectly physical run and
    the one a lift with a registration slip would reproduce. It sits at 3.46e-01
    and 1.22e-01 against parity numbers six orders of magnitude smaller.

    THE CONTROLS ARE THE TEST. Every one of them is a complete, smooth,
    finite field that a broken converter would return without complaint: the
    Bloch control is the run a converter that accepted ``k_point`` and ignored it
    produces, ``high_face`` is MEEP's High = 0 / Low = 1 enum read the intuitive
    way, ``linear`` is a ``chi3`` that never reached the constitutive update, and
    ``flat_sheet`` is a dropped ``amp_func``. The parity bound alone cannot tell
    any of them from a working lift; the control bound is what does.
    """
    bound, control_bounds = _PARITY_BOUNDS[case]
    data = _child(tmp_path, "parity", case)
    print(
        f"[from_meep] {case:32} parity {data['parity']!r} "
        f"controls {data['controls']} steps {data['steps']}",
        flush=True,
    )
    assert data["shape"] == data["reference_shape"], (
        f"{case}: the lifted array is {data['shape']} where CPU MEEP's is "
        f"{data['reference_shape']}; the two codes built different grids."
    )
    assert data["parity"] is not None and data["parity"] < bound, (
        f"{case}: complex relative L2 against CPU MEEP is {data['parity']:.3e}, above the "
        f"{bound:.0e} bar for this feature class."
    )
    for name, floor in control_bounds.items():
        measured = data["controls"][name]
        assert measured > floor, (
            f"{case}: the '{name}' control is only {measured:.3e} away from the lifted field "
            f"(needs > {floor:.0e}). This case cannot distinguish a working lift from one that "
            f"dropped the feature, so its parity number means nothing."
        )


@requires_meep
@skip_without_meep
def test_until_after_sources_stops_where_meep_stops(tmp_path):
    """``run_on_gpu(until_after_sources=T)`` must end at MEEP's own ``last_source_time + T``.

    MEEP resolves the keyword in ``Simulation._run_sources_until`` as
    ``(last_source_time() - round_time()) + T``, and a Gaussian's last source time
    is ``start_time + 2*width*cutoff`` — not the peak. Reading it as the peak, or
    as the raw ``T``, produces a shorter run whose field is a perfectly ordinary
    earlier snapshot of the same simulation, so the check is that the two clocks
    agree to well under one time step and that the fields then match.
    """
    data = _child(tmp_path, "parity", "gaussian_until_after_sources")
    # fwidth 1.0, cutoff 2.0, start_time 0 -> width 1.0, MEEP's last_source_time 4.0.
    expected = 4.0 + 2.0
    assert abs(data["meep_time"] - expected) < 0.05, (
        f"the lifted run stopped at t={data['meep_time']:g} where MEEP's "
        f"last_source_time + 2.0 is {expected:g}."
    )
    assert abs(data["meep_time"] - data["meep_round_time"]) < 1e-9, (
        f"the two clocks disagree: driver t={data['meep_time']:g}, MEEP "
        f"t={data['meep_round_time']:g}."
    )
    assert data["parity"] < 1e-6, f"fields differ by {data['parity']:.3e} at the shared stop time."


@requires_meep
@skip_without_meep
def test_real_field_mode_is_taken_from_meeps_own_storage(tmp_path):
    """The lift must read ``fields.is_real``, not ``force_complex_fields``.

    MEEP promotes a run to complex storage on its own for a nonzero ``k_point``
    even when ``force_complex_fields`` is left False, so the constructor argument
    is not the storage mode. Building a real driver for a run MEEP is stepping
    complex would refuse the Bloch phase outright — a loud failure — but the
    reverse (a complex driver for MEEP's real default) would silently double the
    memory of a run the caller sized. Both directions are pinned here.
    """
    real = _child(tmp_path, "parity", "real_fields")
    assert real["is_real"], "the real_fields case did not actually run MEEP in real storage"
    assert real["parity"] < 1e-6, f"real-mode lift is {real['parity']:.3e} from CPU MEEP"
    bloch = _child(tmp_path, "parity", "bloch")
    assert not bloch["is_real"], "MEEP did not promote the nonzero-k run to complex storage"


# --- refusals ---------------------------------------------------------------------


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", sorted(
    # The expected substrings live in the child's REFUSALS table; the parent reads
    # the same names so the two cannot drift apart.
    [
        "mirror_on_the_invariant_axis", "bloch_on_the_folded_axis",
        "cylindrical_nonint_m", "cylindrical_accurate_origin_unstable_courant",
        "absorber_custom_profile", "absorber_cylindrical", "custom_pml_profile",
        "cylindrical_structured_conductivity",
        "pml_r_asymptotic", "pml_mean_stretch",
        "every_source_in_the_discarded_half", "odd_component_on_the_mirror_plane",
        "monitor_wholly_outside_the_fold",
        "media_differ_in_chi3",
        "already_initialized", "set_boundary_metallic",
        "material_function_dispersive", "material_function_with_dispersive_geometry",
        "rotate4", "magnetic_material",
        "material_grid_negative_damping", "material_grid_dispersive_endpoint",
        "material_grid_endpoint_chi3",
        "rotated_conductivity", "noisy_susceptibility",
        "gyrotropic_susceptibility", "magnetic_susceptibility", "rotated_b_conductivity",
        "gaussian_beam_both_polarizations_2d", "gaussian_beam_volume",
        "gaussian_beam_2d_plane", "gaussian_beam_2d_point",
        "gaussian_beam_2d_in_3d", "gaussian_beam_2d_under_symmetry",
        "negative_epsilon", "perfect_magnetic_conductor",
        "eigenmode_point", "eigenmode_volume", "eigenmode_volume_special_kz",
        "amp_data_on_point", "amp_func_on_point",
        "amp_func_file", "no_sources", "geometry_center", "courant_too_large",
        "bfast_cylindrical",
        "near2far_nperiods", "already_stepped",
        "zero_frequency_source",
        "negative_frequency_eigenmode", "negative_frequency_beam",
    ]
))
def test_unsupported_features_are_refused_by_name(tmp_path, case):
    """Everything the converter cannot reproduce must be reported AND must not run.

    Two assertions per case, because they fail differently. ``gpu_compatibility``
    reporting the feature is what lets a caller choose a backend before paying for
    a run; ``lift_simulation`` raising is what stops a run that was dispatched
    anyway. A converter with the first and not the second is exactly the failure
    this package exists to avoid — the report says "unsupported", the run happens
    regardless, and the field it returns looks like a field.

    The substring asserted is the feature's own name, so a refusal that fires for
    an unrelated reason (a typo elsewhere in the register, a cell that happens to
    be illegal too) does not pass as this one.
    """
    expected = _REFUSAL_SUBSTRINGS[case]
    data = _child(tmp_path, "refusal", case)
    print(f"[from_meep] refuse {case:28} reasons={len(data['reasons'])}", flush=True)
    assert not data["supported"], (
        f"{case}: gpu_compatibility reported this simulation as supported. Reasons: "
        f"{data['reasons']}"
    )
    joined = "\n".join(data["reasons"])
    assert expected in joined, (
        f"{case}: no reason names the unsupported feature (expected {expected!r}).\n{joined}"
    )
    assert data["raised"] == "MeepSimulationNotLiftable", (
        f"{case}: lift_simulation raised {data['raised']!r} instead of "
        f"MeepSimulationNotLiftable — a refused simulation must not build a driver."
    )
    assert expected in data["message"], (
        f"{case}: the exception does not carry the reason.\n{data['message']}"
    )


# Kept beside the parametrization above rather than inside it so the list of cases
# and the list of expected phrases are visibly the same set.
_REFUSAL_SUBSTRINGS = {
    "mirror_on_the_invariant_axis": "translationally invariant",
    "bloch_on_the_folded_axis": "Brillouin-zone interior",
    "cylindrical_nonint_m": "not an integer",
    "cylindrical_accurate_origin_unstable_courant": "1/(|m| + 0.5)",
    "absorber_custom_profile": "custom pml_profile",
    "absorber_cylindrical": "cylindrical (Dcyl) cell",
    "cylindrical_structured_conductivity": "D_conductivity varies from point to point",
    "custom_pml_profile": "custom pml_profile",
    "pml_r_asymptotic": "R_asymptotic",
    "pml_mean_stretch": "mean_stretch",
    "every_source_in_the_discarded_half": "every source",
    "monitor_wholly_outside_the_fold": "wholly outside what the fold",
    "odd_component_on_the_mirror_plane": "parity -1",
    "media_differ_in_chi3": "differ in more than their permittivity",
    "already_initialized": "already been initialized",
    "set_boundary_metallic": "already been initialized",
    "material_function_dispersive": "MEEP DOES register",
    "material_function_with_dispersive_geometry":
        "AND media that differ in more than their permittivity",
    "material_grid_negative_damping": "damping=-0.3",
    "material_grid_dispersive_endpoint": "E_susceptibilities",
    "material_grid_endpoint_chi3": "NEVER READS",
    "rotate4": "Rotate4",
    "magnetic_material": "mu_diag",
    "rotated_conductivity": "rotated conductivity",
    "noisy_susceptibility": "NoisyLorentzianSusceptibility",
    "gyrotropic_susceptibility": "GyrotropicLorentzianSusceptibility",
    "magnetic_susceptibility": "H_susceptibilities",
    "rotated_b_conductivity": "B_conductivity_offdiag=",
    "gaussian_beam_both_polarizations_2d": "BOTH an in-plane and an out-of-plane",
    "gaussian_beam_volume": "zero-extent resolved axes",
    "gaussian_beam_2d_plane": "line source, not a plane",
    "gaussian_beam_2d_point": "UnboundLocalError",
    "gaussian_beam_2d_in_3d": "two-dimensional by construction",
    "gaussian_beam_2d_under_symmetry": "mirror symmetry",
    "negative_epsilon": "no stable leapfrog",
    "perfect_magnetic_conductor": "mu_diag",
    "eigenmode_point": "flat resolved axes",
    "eigenmode_volume": "flat resolved axes",
    "eigenmode_volume_special_kz": "MEEP does resolve this one",
    "amp_data_on_point": "zero-size (point) source",
    "amp_func_on_point": "zero-size (point) source",
    "amp_func_file": "amp_func_file",
    "no_sources": "declares no sources",
    "geometry_center": "geometry_center",
    "courant_too_large": "CFL stability limit",
    "bfast_cylindrical": "step_generic.cpp:376",
    "near2far_nperiods": "replicates the near surface",
    "already_stepped": "already been stepped",
    "zero_frequency_source": "finite and nonzero",
    "negative_frequency_eigenmode": "positive frequencies only",
    "negative_frequency_beam": "positive frequencies only",
}


# test_the_curved_geometry_refusal_carries_its_measured_cost used to live here: the
# refusal it argued for is gone — curved geometry lifts through the off-diagonal
# chi1inv ingestion (two_d_cylinder_smoothed / two_d_rotated_block_smoothed_te), and
# the numbers it quoted (5.7e-02 diagonal-only, 1.84e-07 point-sampled) are now the
# parity cases' controls rather than a message's argument.


@requires_meep
@skip_without_meep
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("half_cell_shifted_sampling", "registration slip of 0.5 cell"),
        ("whole_cell_shifted_sampling", "registration slip of 1 cell"),
        ("offlattice_sampling", "not on MEEP's integer lattice"),
        ("wrong_material_installed", "in the BULK"),
        ("bulk_check_is_wired_in", "bulk cross-check reached"),
    ],
)
def test_a_mis_sampled_permittivity_is_caught_before_a_single_step(tmp_path, case, expected):
    """Force each deep gate by disabling the first-line check that shadows it.

    These are the whole reason the structured path is allowed to exist. Each
    disables one layer and checks that the next one still refuses:

    * ``half_cell_shifted_sampling`` — every sample coordinate is moved half a
      cell. That lands on the wrong parity of MEEP's integer lattice, where
      ``grid_volume::index`` does not complain, it reads the NEIGHBOUR.
    * ``whole_cell_shifted_sampling`` — a full cell, which keeps the PARITY legal.
      This is the shift that installs a structure displaced by exactly one cell
      and steps it without a word, and it is the reason the registration pin is an
      equality against ``gv.little_corner() + gv.iyee_shift(c)`` rather than a
      parity test: a parity test passes this one. (So does the bulk cross-check
      against MEEP's cell-centred diagnostic — a thick slab moved by one cell is
      still uniform in its own interior — which is why that check is a second
      witness and not the pin.)
    * ``offlattice_sampling`` — three tenths of a cell, which is not a lattice
      point at either parity. This is what the drift check is for, and the reason
      it raises instead of letting ``round_vec`` snap to a neighbour.
    * ``wrong_material_installed`` — not a registration error at all: right
      coordinates, wrong VALUES. The integer pin cannot see it, so the cross-check
      against MEEP's own diagnostic is called directly with a permittivity MEEP
      did not build. It is the only place in the lift where a permittivity is
      compared to MEEP's rather than a coordinate to MEEP's.
    * ``bulk_check_is_wired_in`` — defining a check and calling it are different
      facts. Deleting the CALL survived a suite that exercised the FUNCTION, so
      the cross-check is replaced with one that always raises and the lift has to
      hit it.

    A gate nobody can reach is a gate nobody knows is broken, which is why these
    are forced rather than left to the first-line checks. Three of them exist
    *because* a mutation battery found the checks unreachable: deleting the drift
    check, deleting the bulk cross-check, and deleting its call site all left the
    suite green (``parity/meep_gpu/mutate_from_meep.py``), and these cases are
    what turned that 37/39 into 39/39.
    """
    data = _child(tmp_path, "gate", case)
    assert data["raised"] is not None, (
        f"gate '{case}': the lift completed and would have stepped a permittivity that is not "
        f"the one MEEP built. Nothing downstream reports that."
    )
    assert expected in data["message"], (
        f"gate '{case}': raised {data['raised']} but the message does not name the defect "
        f"({expected!r}):\n{data['message']}"
    )


@requires_meep
@skip_without_meep
def test_the_declared_span_check_keeps_its_teeth_without_the_wall_exclusion(tmp_path):
    """Force the wall-plane exclusion empty; the spread check must refuse again.

    FIX-B's exclusion removes exactly the lattice planes MEEP measurably declines
    (``grid_volume::owns`` needs ``o > 0``, vec.cpp:445-463; the unowned diagonal
    defaults to vacuum, monitor.cpp:180-183) from the declared-media spread check.
    That check is the only place the lift ever compares a sampled permittivity
    against the declaration, so the exclusion must not be able to disarm it: with
    the mask forced empty on ring_gds's own class (SiO2 background, Si block,
    metallic walls) the declined-default vacuum is visible again and the lift must
    die exactly as it did before the fix — naming the spread. A lift that
    completes here would mean the check lost its teeth, and the
    ``declared_span_check_removed`` mutation is anchored on this test.
    """
    data = _child(tmp_path, "gate", "wall_exclusion_disabled_spread_still_refuses")
    assert data["raised"] == "ValueError", (
        f"with the wall-plane exclusion disabled the lift completed "
        f"(raised={data['raised']!r}); the declared-media spread check has lost its teeth."
    )
    assert "outside the range" in data["message"], (
        f"the refusal does not name the declared-media span:\n{data['message']}"
    )


@requires_meep
@skip_without_meep
def test_the_bulk_cross_check_excludes_the_union_of_the_declined_wall_planes(tmp_path):
    """The wall exclusion at its SECOND consumer, and the union is the load-bearing part.

    ``sim.get_epsilon()`` is an array slice over stored chunk data; both per-point
    public readers refuse a point no chunk owns (``fields::get_chi1inv`` substitutes
    literal vacuum, monitor.cpp:181-183; ``structure::get_chi1inv`` returns 0.0,
    monitor.cpp:238). So on the low-boundary plane of a non-wrapping axis the
    diagnostic knows the material and the reader cannot be made to say it, and
    requiring bulk equality there requires the impossible. Measured on
    test_mode_decomposition.py::test_oblique_waveguide_backward_mode, which this cell
    is the small twin of: one cell of 54639 carried the whole 4.364e-03 refusal, and
    excluding the declined planes leaves 2.50991e-07 against a 1.225e-03 tolerance
    with the lift then matching CPU MEEP 1.33.0 at parity_rel_l2 = 1.1391e-06 over
    2000 steps (time_matched, full signal) — which is what says the substituted vacuum
    is inert rather than merely tolerated.

    FOUR POLICIES, one simulation, one check. The mask handed to
    ``_require_bulk_matches_meep`` is the only thing that varies:

    * ``union`` (what the lift does) — the lift completes.
    * ``none`` — the pre-fix state. Must refuse "in the BULK", or the exclusion is
      being credited for something the check never had.
    * ``ex_only`` — "mask the component you compare", stated outright. The check
      classifies and compares ``stacked[0]``, so this is the intuitive reading, and
      it is WRONG: at the failing cell Ex's window opens at corner+1 and is genuinely
      owned, while Ey's and Ez's open at the corner and are declined. Must still
      refuse.
    * ``intersection`` — ``&`` where the lift writes ``|``, one character. The
      intersection of the three masks is the corner cell alone. Must still refuse.

    The last two are the negative controls that matter: both leave a mask in place,
    both look like the fix, and both put a correctly-read structure back into a
    refusal. They are what the ``bulk_wall_exclusion_*`` mutations are anchored on.
    """
    data = _child(tmp_path, "gate", "wall_exclusion_policies_on_a_bulk_wall_cell")
    policies = data["policies"]
    assert policies["union"]["raised"] is None, (
        f"the lift refused a structure it read correctly: "
        f"{policies['union']['raised']}: {policies['union']['message']}"
    )
    for policy in ("none", "ex_only", "intersection"):
        outcome = policies[policy]
        assert outcome["raised"] == "ValueError", (
            f"mask policy '{policy}' let the lift complete (raised={outcome['raised']!r}); "
            f"the bulk cross-check has stopped seeing the wall cell, so the union in "
            f"`_lift_epsilon_structured` is no longer what makes the difference and this "
            f"test would pass with the exclusion masked any way at all."
        )
        assert "in the BULK" in outcome["message"], (
            f"mask policy '{policy}' refused for a different reason than the bulk "
            f"cross-check:\n{outcome['message']}"
        )
    # The geometry claim the union rests on, as numbers: Ex is declined on one wall row
    # (its x window opens at corner+1), Ey on the other, Ez on both. The union is the
    # two rows; the intersection is the single corner cell, which is why `&` is not a
    # near-miss of `|` but a different mask entirely.
    ex, ey, ez = data["mask_sizes"]
    side = int(round(data["cells"] ** 0.5))
    assert (ex, ey, ez) == (side, side, 2 * side - 1), (
        f"the declined wall planes are not the measured ones on a {side}x{side} cell "
        f"(Ex {ex}, Ey {ey}, Ez {ez}; expected {side}, {side}, {2 * side - 1})"
    )
    assert data["union_size"] == 2 * side - 1 and data["intersection_size"] == 1, (
        f"union {data['union_size']} / intersection {data['intersection_size']} on a "
        f"{side}x{side} cell: the two policies are supposed to differ by a whole wall row."
    )


@requires_meep
@skip_without_meep
def test_a_pml_thinner_than_a_quarter_cell_is_lifted_as_no_pml_at_all(tmp_path):
    """MEEP builds no absorber below a quarter cell, and neither may the lift.

    ``structure_chunk::use_pml`` scans for a point with ``x > 0`` and returns before
    allocating sigma when it finds none (structure.cpp:648-654). ``half_cell_extent``
    is 0 for anything under 0.25 cells, so that scan cannot succeed: the layer is not
    thin, it is absent.

    Installing it anyway would grade nothing — every coefficient exactly 1 — but would
    still put the run into PML field storage, where E and H are served from the split
    auxiliaries. That is a mode difference with no MEEP counterpart, reached silently,
    on a configuration MEEP accepts without a word.

    Both sides of the threshold are measured, because "dropped" is only correct below
    it: at resolution 10 a 0.02-unit layer is 0.2 cells (extent 0 half-cells) and a
    0.03-unit one is 0.3 cells (extent 1) — MEEP's thinnest layer that is really there.
    A lift that dropped every thin layer, or ceiled every one to a half cell, fails on
    exactly one of the two.
    """
    data = _child(tmp_path, "gate", "pml_below_a_quarter_cell_is_no_layer")
    assert data["raised"] is None, f"a sub-quarter-cell PML no longer lifts: {data['message']}"
    assert data["below_has_pml"] is False, (
        f"a 0.2-cell layer installed an absorber ({data['below_faces']}); MEEP allocates no "
        f"sigma array for it at all, and installing one switches the run into PML storage."
    )
    assert data["above_has_pml"] is True and data["above_graded_cells"] == 1, (
        f"a 0.3-cell layer must install MEEP's thinnest real absorber — one graded integer "
        f"sample per face — got has_pml={data['above_has_pml']} with "
        f"{data['above_graded_cells']} graded cells and faces {data['above_faces']}."
    )


#: The MEEP release whose python/geom.py:450 assigns ``B_conductivity_offdiag`` from
#: the B keyword. 1.33.0 (and 1.31.0) assign it from the D keyword.
_GEOM_450_CORRECTED_IN = (1, 34, 0)


def _meep_release(version) -> tuple:
    """``"1.34.0"`` -> ``(1, 34, 0)``: the leading numeric fields of a MEEP version."""
    import re  # noqa: PLC0415

    fields = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    assert fields, f"MEEP reported a version this test cannot read: {version!r}"
    return tuple(int(field or 0) for field in fields.groups())


@requires_meep
@skip_without_meep
def test_an_offdiagonal_conductivity_never_reaches_meeps_stepper(tmp_path):
    """The premise both rotated-conductivity refusals rest on, MEASURED not assumed.

    The refusal used to argue that a rotated tensor "couples the components in
    step_D". IT DOES NOT — MEEP never receives it. ``medium_struct`` carries
    ``D_conductivity_diag`` and ``B_conductivity_diag`` and no off-diagonal at all
    (src/material_data.hpp:78-79), and the Python-to-C++ conversion copies exactly
    those two (python/typemap_utils.cpp:815-816). So the declaration is DISCARDED,
    which puts it in the same class as the MaterialGrid endpoint fields: refusing it
    is still right, but because lifting it would install a loss MEEP is not stepping,
    not because this engine cannot express a coupling MEEP applies.

    The second half pins which keyword reaches ``B_conductivity_offdiag`` on the
    MEEP in hand, because the refusal's NOTE states it per release. On MEEP 1.33.0
    ``mp.Medium.__init__`` assigns
    ``self.B_conductivity_offdiag = Vector3(*D_conductivity_offdiag)``
    (python/geom.py:450), the D keyword rather than the B one: a
    ``B_conductivity_offdiag=`` argument is silently dropped and a
    ``D_conductivity_offdiag=`` argument silently sets both sides. MEEP 1.34.0 assigns
    each keyword to its own attribute. A release that behaves otherwise than the NOTE
    says for it fails here.
    """
    data = _child(tmp_path, "readers", "conductivity_offdiag_is_discarded_by_meep")
    print(f"[from_meep] offdiag_discard offdiag {data['offdiag_vs_none']:.4e} "
          f"diagonal {data['diagonal_vs_none']:.4e}", flush=True)
    assert data["offdiag_vs_none"] == 0.0, (
        f"MEEP's field MOVED by {data['offdiag_vs_none']:.4e} when D_conductivity_offdiag "
        f"was declared, so it is no longer discarded and the refusal's stated reason — "
        f"and the whole class it was filed under — need re-deriving against the source."
    )
    assert data["diagonal_vs_none"] > 1e-2, (
        f"the DIAGONAL control moved only {data['diagonal_vs_none']:.4e}; this cell cannot "
        f"tell a discarded conductivity from a conductivity that does nothing here, so the "
        f"zero above means nothing."
    )
    version = data["meep_version"]
    if _meep_release(version) >= _GEOM_450_CORRECTED_IN:
        assert data["b_keyword_reaches_b"] == [0.1, 0.0, 0.0], (
            f"on MEEP {version} mp.Medium(B_conductivity_offdiag=...) does not reach the "
            f"attribute ({data['b_keyword_reaches_b']}); the refusal's NOTE says 1.34.0 "
            f"assigns each keyword to its own attribute. Re-read python/geom.py:450.")
        assert data["d_keyword_reaches_b"] == [0.0, 0.0, 0.0], (
            f"on MEEP {version} mp.Medium(D_conductivity_offdiag=...) aliases onto "
            f"B_conductivity_offdiag ({data['d_keyword_reaches_b']}) again; the refusal's "
            f"NOTE says 1.34.0 does not. Re-read python/geom.py:450.")
    else:
        assert data["b_keyword_reaches_b"] == [0.0, 0.0, 0.0], (
            f"on MEEP {version} mp.Medium(B_conductivity_offdiag=...) now reaches the "
            f"attribute ({data['b_keyword_reaches_b']}); the refusal's NOTE says 1.33.0 and "
            f"1.31.0 discard it (python/geom.py:450). Correct the NOTE.")
        assert data["d_keyword_reaches_b"] == [0.1, 0.0, 0.0], (
            f"on MEEP {version} mp.Medium(D_conductivity_offdiag=...) no longer aliases "
            f"onto B_conductivity_offdiag ({data['d_keyword_reaches_b']}); the refusal's "
            f"NOTE says 1.33.0 and 1.31.0 do (python/geom.py:450). Correct the NOTE.")


@requires_meep
@skip_without_meep
def test_the_conductivity_read_recovers_the_declaration_and_checks_its_own_premise(tmp_path):
    """The per-point ``D_conductivity`` read, against closed-form answers and its own gates.

    A field-parity number cannot say this. Both cells below have a conductivity whose
    correct value is known EXACTLY from the declaration, so the recovered volume is
    compared against that rather than against a tolerance:

    * a declared ``D_conductivity=0.7`` block — recovered to float32, which is
      0.699999988, not 0.7;
    * a ``MaterialGrid`` whose ``damping`` MEEP turns into ``u*(1-u)*damping`` per point
      (meepgeom.cpp:623-626). Its maximum is ``damping/4``, attained wherever the design
      passes through u = 1/2, and NOTHING in this package computes that — the value is
      read back out of the array MEEP wrote it into, which is the whole point of taking
      the D row instead of reimplementing ``matgrid_val`` and ``tanh_projection``.

    Then the two premises the read stands on, each forced through a shim in front of
    ``get_chi1inv`` so the wrong MEEP can be presented without touching MEEP:

    * ``identity_broken`` — the D row's instantaneous tensor is NOT the identity.
      ``structure::set_materials`` fills ``chi1inv`` under FOR_ELECTRIC/MAGNETIC only
      (structure.cpp:374-384), so on stock MEEP it is; a MEEP that allocated it would
      make every recovered sigma ``eps*sigma`` — 8.4 instead of 0.7 on this very cell —
      in a run that completes and looks smooth. This is the check that makes the read a
      read rather than an assumption.
    * ``frequency_dependent`` — sigma_D does not vary with frequency
      (monitor.cpp:339-343 multiplies by ``1 + i*sigma/frequency``), so two probes must
      agree. Measured agreement on the real cells is 1.11e-16 absolute against a
      tolerance of 1e-9 relative, so the check has seven orders of headroom.
    * ``negative`` — gain wearing loss's name, which grows the field while every
      magnitude still looks reasonable.
    """
    data = _child(tmp_path, "gate", "conductivity_read_is_exact_and_self_checked")
    for label, tolerance in (("block", 1e-7), ("grid", 1e-5)):
        declared = data[label + "_declared"]
        recovered = data[label + "_recovered_max"]
        print(f"[from_meep] conductivity {label}: declared {declared:.9g} recovered "
              f"{['%.9g' % value for value in recovered]}", flush=True)
        for axis, value in enumerate(recovered):
            assert abs(value - declared) <= tolerance * max(declared, 1.0), (
                f"{label}: D component {axis} recovered a peak conductivity of {value!r} where "
                f"the declaration gives {declared!r}. The read is exact up to MEEP's float32 "
                f"storage; a difference this large means it is reading something else — the "
                f"permittivity-scaled E row gives eps*sigma, and a frequency-0 read gives 0."
            )
        assert data[label + "_min"] >= 0.0, (
            f"{label}: the recovered volume holds a negative conductivity "
            f"({data[label + '_min']!r}), which is gain."
        )
    expected = {
        "identity_broken": "not the literal 1.0",
        "frequency_dependent": "disagrees with itself",
        "negative": "NEGATIVE D_conductivity",
    }
    for label, phrase in expected.items():
        outcome = data["self_checks"][label]
        assert outcome["raised"] == "MeepSimulationNotLiftable", (
            f"the {label} premise was not caught: _read_conductivity_volumes returned "
            f"{outcome['raised']!r} instead of refusing. Each of these produces a complete, "
            f"smooth, wrong run if it is installed."
        )
        assert phrase in outcome["message"], (
            f"the {label} refusal does not name what it caught (expected {phrase!r}).\n"
            f"{outcome['message']}"
        )


# `test_source_placement_is_pre_flighted_from_the_same_geometry_the_driver_checks`
# and its `_SOURCE_PLACEMENT_VERDICTS` table used to live here, holding the
# pre-flight and the driver to one wall predicate. The predicate is deleted with
# the defect it guarded (the `metal_high` drop in `sources._build_source_points`
# is the fix, measured at the floor bare and through the lift), so there is no
# placement refusal left for the two sides to agree on: the admission is pinned
# by the `cylindrical_source_on_r_wall` parity case and the deposition
# arithmetic by test_sources' metallic-wall drop test.


@requires_meep
@skip_without_meep
def test_offdiagonal_chi1inv_is_ingested_with_meeps_own_registration(tmp_path):
    """A smoothed cylinder installs exactly the in-plane coupling rows, registered as MEEP's.

    Three pins, none of which a field comparison could isolate:

    * WHICH rows: a 2-D cylinder's surface normals are in-plane, so the tensor
      tilt is the xy block — chi1inv[Ex][Y] and chi1inv[Ey][X] — and the Ez row
      stays diagonal. Extra rows mean the lift is reading tilt that is not there;
      missing rows mean the coupling was dropped.
    * HOW BIG: the installed coupling reaches the 0.203-class magnitude the old
      refusal measured, so the ingestion is carrying the real signal and not a
      rounding residue.
    * WHERE: the symmetric-row identity. MEEP samples every off-diagonal entry at
      the component's own site minus half a cell along its own axis — the integer
      node (anisotropic_averaging.cpp:248-257) — so chi1inv[Ex][Y] at Ex slot
      (i,j,k) and chi1inv[Ey][X] at Ey slot (i,j,k) describe the SAME node and are
      equal entry for entry. A half-cell misreading of that registration breaks
      the identity on every interface cell, where the coefficient changes fastest.
    """
    data = _child(tmp_path, "gate", "offdiagonal_chi1inv_ingested")
    assert data["raised"] is None, (
        f"the smoothed cylinder no longer lifts: {data['message']}"
    )
    assert data["nonzero_rows"] == ["Ex:Ey", "Ey:Ex"], (
        f"a 2-D cylinder must install exactly the in-plane coupling pair, got "
        f"{data['nonzero_rows']}."
    )
    assert data["max_offdiag"] > 0.05, (
        f"the installed coupling peaks at {data['max_offdiag']:.3g}; the cylinder's measured "
        f"tilt is 0.203-class, so the ingestion lost the signal."
    )
    assert data["row_symmetry_max_diff"] < 1e-10, (
        f"chi1inv[Ex][Y] and chi1inv[Ey][X] differ by {data['row_symmetry_max_diff']:.3e} at "
        f"the same slot; the two are the same symmetric tensor at the same node, so any gap "
        f"is a registration misreading."
    )


# --- structure, with no MEEP involved ----------------------------------------------


def test_importing_the_package_does_not_import_meep():
    """``import meep_gpu`` must stay free of MEEP even though the converter needs it.

    The whole package is usable on a machine that has never had MEEP installed,
    and every other module here is written to keep it that way. A module-scope
    ``import meep`` in ``from_meep.py`` would break that silently for everyone —
    the failure would be an ImportError from ``import meep_gpu`` itself, at which
    point nothing in the package works.

    Run in a bare child rather than through :func:`_child`: the case script
    imports MEEP itself, so ``sys.modules`` there can say nothing about what
    ``import meep_gpu`` pulled in. This one imports only the package.
    """
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONPATH"] = os.pathsep.join(
        [_PACKAGE_PARENT] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [sys.executable, "-c",
         "import sys, meep_gpu; print('meep' in sys.modules); print(bool(meep_gpu.gpu_compatibility))"],
        capture_output=True, text=True, env=environment, timeout=300,
    )
    assert completed.returncode == 0, f"importing meep_gpu failed:\n{completed.stderr[-3000:]}"
    imported, exported = completed.stdout.split()[:2]
    assert imported == "False", (
        "importing meep_gpu pulled in meep; from_meep.py must import it lazily, inside the "
        "functions that read an mp.Simulation."
    )
    assert exported == "True", "meep_gpu.gpu_compatibility is not exported from the package root."


def test_compatibility_report_is_a_reason_list_not_a_boolean():
    """``GpuCompatibility`` must carry why, not only whether.

    Modelled on ``GpuFdtdBackend.compatibility`` in the host application: a
    dispatcher choosing between backends needs the reasons to report or work
    around, and collapsing them to False makes every distinct refusal the same
    dead end.
    """
    supported = GpuCompatibility(supported=True, reasons=())
    refused = GpuCompatibility(supported=False, reasons=("first thing", "second thing"))
    assert bool(supported) and not bool(refused)
    assert "first thing" in str(refused) and "second thing" in str(refused)
    assert refused.reasons == ("first thing", "second thing")


def test_get_array_carries_the_bloch_phase_on_the_duplicate_plane():
    """MEEP's index-0 plane is one lattice vector down and carries ``conj(bloch_phase)``.

    Measured directly against ``sim.get_array`` at k = (0.1234567, 0, 0):
    ``ez[0] / ez[-1] = 0.019392 - 0.999812j``, which is ``conj(exp(2j*pi*k*L))``
    to every digit. ``Fields.to_meep_array`` prepends a plain copy instead, so
    ``GpuRunResult.get_array`` applies the factor itself — 1.98e-01 against CPU
    MEEP without it, 2.90e-07 with it, on a volume that is otherwise exact.

    No MEEP is needed to pin the relation: it is a statement about the array this
    package produces, checked against the driver's own ``bloch_phase``.
    """
    driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, force_complex_fields=True,
                        k_point=(0.1234567, 0.0, 0.0))
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                       "size": (0.0, 0.0, 0.0)})
    driver.run(until=1.0)
    raw = np.asarray(driver.fields.to_meep_array("Ez"))
    result = GpuRunResult(driver=driver, steps=driver.step_count,
                          meep_time=driver.meep_time(), wall_time_s=0.0)
    array = result.get_array("Ez")
    phase = driver.grid.bloch_phase(0)

    assert np.array_equal(raw[0], raw[-1]), (
        "Fields.to_meep_array no longer prepends a plain duplicate; the correction in "
        "GpuRunResult.get_array must be deleted rather than left to double-apply."
    )
    expected = (raw[-1] * np.conjugate(phase)).astype(array.dtype)
    assert array.dtype == raw.dtype, "get_array promoted the stored dtype."
    assert np.array_equal(array[0], expected), (
        "get_array did not apply conj(bloch_phase) to the prepended plane."
    )
    # Only that one plane moves; a factor applied to the wrong axis or to the whole
    # volume would still leave the interior looking perfectly reasonable.
    assert np.array_equal(array[1:], raw[1:]), "get_array altered the interior of the volume."
    driver.close()


def test_get_array_refuses_to_double_apply_the_bloch_correction():
    """The tripwire must fire if ``Fields`` ever learns the factor itself.

    Applying ``conj(bloch_phase)`` twice squares it on one plane and leaves every
    other cell exact, which is the quietest possible way for this correction to
    become wrong. It is guarded by a bit-equality check rather than by a comment.
    """
    driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10, force_complex_fields=True,
                        k_point=(0.25, 0.0, 0.0))
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.0, 0.0, 0.05),
                       "size": (0.0, 0.0, 0.0)})
    driver.run(until=1.0)
    result = GpuRunResult(driver=driver, steps=driver.step_count,
                          meep_time=driver.meep_time(), wall_time_s=0.0)
    already_phased = driver.fields.to_meep_array("Ez")
    already_phased[0] = already_phased[0] * 0.5  # Stand in for a fixed _add_boundary_cells.
    driver.fields.to_meep_array = lambda component: already_phased  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="no longer prepends a plain duplicate"):
        result.get_array("Ez")
    driver.close()


class _Vector3:
    """The three attributes ``_vector3`` reads off an ``mp.Vector3``."""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _StubSimulation:
    """Just enough of an initialized ``mp.Simulation`` for the two array-side gates.

    Both gates run AFTER ``gpu_compatibility`` has passed, so nothing can reach
    them through the public API without first defeating the declaration analysis.
    They are the second line — the one that catches a material route the register
    does not know about — and a second line that is only reachable through a
    first-line bug is exactly the kind of check that rots unnoticed. A stub gets
    at them directly, and needs no MEEP to do it.
    """

    def __init__(self, epsilon, cell_size=(2.0, 2.0, 2.0), resolution=10.0):
        self._epsilon = epsilon
        self.cell_size = _Vector3(*cell_size)
        self.resolution = resolution

    def get_epsilon(self):
        return self._epsilon


def test_a_grid_meep_built_differently_is_refused_before_stepping():
    """The (N+1)-per-unfolded-axis shape is a PIN, not an assumption.

    If MEEP's cell count and this engine's ``meep_cell_count`` ever disagree —
    a length that lands on a rounding boundary, a resolution MEEP snaps
    differently — every coordinate in the lifted run is registered against the
    wrong lattice and the field still looks like a field. Half a cell was worth
    4.8e-02 on a transmitted flux the last time it happened.
    """
    from .from_meep import _require_grids_agree

    driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10)
    _require_grids_agree(_StubSimulation(np.ones((21, 21, 21))), driver)  # The right shape passes.
    for wrong in ((20, 21, 21), (21, 22, 21), (21, 21, 20)):
        with pytest.raises(ValueError, match="built different grids"):
            _require_grids_agree(_StubSimulation(np.ones(wrong)), driver)
    driver.close()


def test_a_folded_axis_expects_meeps_full_count_with_no_duplicate_plane():
    """A mirror-folded axis is not periodic, so MEEP prepends nothing there.

    Expecting ``n_full + 1`` on every axis refuses every symmetric simulation
    outright — which is loud, but it is the shape rule itself that is wrong, and
    the same rule read the other way would accept a genuinely mismatched grid.
    """
    from .from_meep import _require_grids_agree
    from .grid import Mirror

    driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10,
                        symmetry=(Mirror("Z", -1),))
    _require_grids_agree(_StubSimulation(np.ones((21, 21, 20))), driver)  # nz_full, no duplicate.
    with pytest.raises(ValueError, match="built different grids"):
        _require_grids_agree(_StubSimulation(np.ones((21, 21, 21))), driver)
    driver.close()


def test_a_structured_volume_reaching_the_lift_is_refused_by_the_array_itself():
    """The uniformity gate must fire on MEEP's rasterized array, not only on the declaration.

    ``gpu_compatibility`` judges the DECLARED materials, which is what makes it
    cheap enough to call before building anything. A permittivity that reaches
    the structure by a route the register did not see — any material path added to
    a future MEEP — would pass that analysis and then be installed at some
    registration or other, at the 1.16e-01 the module docstring records. This is
    the check that stops it.

    Two routes that USED to be this check's whole subject no longer reach it: a
    MaterialGrid is expanded into its endpoint media, and a per-point epsilon route
    (a material callable, an ndarray, an ``epsilon_input_file``) is carried by
    ``_epsilon_routes`` and sent down the structured path unconditionally. The check
    stays because the class it guards is "a route this module does not know about",
    and that class is never empty.
    """
    from .from_meep import _lift_epsilon

    driver = FdtdDriver(cell_size=(2.0, 2.0, 2.0), resolution=10)
    medium = type("Medium", (), {"epsilon_diag": _Vector3(2.0, 2.0, 2.0)})()
    uniform = np.full((21, 21, 21), 2.0)
    _lift_epsilon(_StubSimulation(uniform), driver, medium)  # Uniform installs cleanly.

    structured = uniform.copy()
    structured[:, :, 10:] = 4.0
    with pytest.raises(ValueError, match="not uniform"):
        _lift_epsilon(_StubSimulation(structured), driver, medium)

    # And a uniform volume that is not the DECLARED material is refused too: MEEP
    # built something other than what the register read, so installing the
    # declaration would step a permittivity the reference run does not have.
    with pytest.raises(ValueError, match="declared material is not the one MEEP built"):
        _lift_epsilon(_StubSimulation(np.full((21, 21, 21), 3.0)), driver, medium)
    driver.close()


def test_run_on_gpu_requires_exactly_one_stopping_condition():
    """Neither or both is a caller error, and must not pick one silently."""
    from .from_meep import run_on_gpu

    with pytest.raises(ValueError, match="exactly one of until or until_after_sources"):
        run_on_gpu(object(), until=1.0, until_after_sources=1.0)
    with pytest.raises(ValueError, match="exactly one of until or until_after_sources"):
        run_on_gpu(object())


@requires_meep
@skip_without_meep
def test_the_boundary_condition_is_lifted_from_k_point_and_reported_as_a_declaration(tmp_path):
    """``sim.k_point`` decides the whole cell's outer boundary, and the lift says which.

    THE CONDITION IS ALL-OR-NOTHING, because MEEP makes it so:
    ``Simulation._init_fields`` calls ``fields::use_bloch`` for EVERY direction under a
    plain ``if self.k_point:`` (python/simulation.py), and ``use_bloch`` overwrites both
    faces of the axis it is given with ``Periodic`` (boundaries.cpp). ``k_point=False``,
    the default, therefore leaves ``fields::fields``'s ``Metallic`` standing on all six
    faces (fields.cpp), and ``k_point=mp.Vector3()`` — which is TRUTHY — turns the whole
    cell periodic. There is no declared-argument route to a mixed cell, which is why
    ``_lift_boundaries`` returns one condition rather than a per-axis table even though
    the engine underneath takes one.

    This gate reads the decision back off the built driver rather than inferring it from
    a field, because a field comparison can be satisfied by a coincidence and a
    declaration cannot. MEEP's own epsilon shape is asserted alongside it: N per axis
    for the metallic box and N + 1 for the periodic one, which is the same fact seen
    from MEEP's side.
    """
    data = _child(tmp_path, "gate", "boundaries_are_lifted_from_k_point")
    lifts = data["lifts"]
    for label, lift in sorted(lifts.items()):
        print(f"[from_meep] boundaries {label:15} {lift['boundaries']} "
              f"meep_eps {lift['meep_shape']} driver {lift['driver_shape']}", flush=True)
        assert lift["supported"], f"{label} was refused: {lift['reasons']}"

    metallic = lifts["metallic"]
    assert metallic["boundaries"] == [["metallic", "metallic"]] * 3
    assert metallic["metallic_axes"] == [True, True, True]
    assert metallic["wraps"] == [False, False, False]
    assert metallic["k_point"] == [0.0, 0.0, 0.0]
    # MEEP returns exactly N per axis for a PEC box; the driver stores exactly N.
    assert metallic["meep_shape"] == metallic["driver_shape"]

    for label in ("periodic_k0", "periodic_bloch"):
        periodic = lifts[label]
        assert periodic["boundaries"] == [["periodic", "periodic"]] * 3, label
        assert periodic["metallic_axes"] == [False, False, False], label
        assert periodic["wraps"] == [True, True, True], label
        # ... and MEEP's array gains its duplicate plane per axis, which is the whole
        # reason the metallic case needed a shape rule of its own.
        assert periodic["meep_shape"] == [n + 1 for n in periodic["driver_shape"]], label

    # k_point=mp.Vector3() is TRUTHY, so an all-zero vector is a PERIODIC run and not a
    # spelling of the default. Reading it as falsy would silently wall every k = 0 run
    # in the suite — 1.28e+00 wrong, and the same shape mismatch as above.
    assert lifts["periodic_k0"]["boundaries"] != metallic["boundaries"]


@requires_meep
@skip_without_meep
def test_reduced_dimensions_are_lifted_the_way_meep_infers_them(tmp_path):
    """Every reduced spelling must reach the dimensionality MEEP builds, and no other.

    Four facts are pinned per spelling, each a wrong answer that would otherwise be a
    smooth complete field:

    * WHICH axes are invariant. MEEP's assignment is fixed by
      ``start_at_direction``/``stop_at_direction`` (vec.hpp): 2-D is the X-Y plane and
      1-D is the Z axis. Guessing 1-D as "the x axis" rotates every source component
      and both polarizations by 90 degrees and still steps.
    * WHICH spellings reduce at all. ``Simulation._infer_dimensions`` collapses only on
      ``cell_size.z``, and only when ``k_point.z`` is zero. ``mp.Vector3(0, 0, L)`` at
      ``dimensions=3`` is a 3-D run with two ONE-CELL axes ("Working in 3D dimensions.
      Computational cell is 0.1 x 0.1 x 4"), and a nonzero ``k_point.z`` on a
      zero-thickness cell is a 3-D run with a one-cell z axis carrying a Bloch phase.
      The rule this replaced took ``min(declared, 3 - flat_axes)`` and reported 1-D for
      both.
    * WHAT boundary the invariant and unit axes get. MEEP has no ``fields::boundaries``
      entry for a direction it does not loop over, and forces a unit direction Periodic
      whatever ``k_point`` says (fields.cpp, "unit directions are periodic by
      default"). Metallic instead zeroes every component whose Yee shift on that axis
      is 0 — the entire TE polarization of a 2-D run — and returns a smooth exact zero:
      measured 1.000e+00 against CPU MEEP, against 1.805e-07 with the wrap.
    * WHETHER the delta-function amplitude scaling applies there. MEEP skips it on a
      one-pixel periodic direction (``fields::nosize_direction``), and skipping the
      skip multiplies every 2-D source by the resolution and every 1-D source by its
      square.

    ``dimensions=0`` is pinned here as a REFUSAL rather than a reduction. The routing
    half of the story is real — ``_create_grid_volume`` sends ``dims == 0 or dims == 1``
    to the same ``vol1d`` — but it is only half: ``py_v3_to_vec`` then rejects the 0
    inside ``init_sim``, so the script never steps on CPU MEEP either. Verified against
    this MEEP install, which raises ``ValueError: Invalid dimensions in Volume: 0``.
    Accepting it would mean lifting a simulation the oracle itself cannot build, so the
    engine declines it by name and says to declare ``dimensions=1``.
    """
    data = _child(tmp_path, "gate", "reduced_dimensions_are_lifted")
    lifts = data["lifts"]
    for label, lift in sorted(lifts.items()):
        print(f"[from_meep] reduced {label:22} dims={lift['dimensions']} "
              f"invariant={lift['invariant']} driver={lift['driver_shape']} "
              f"meep_eps={lift['meep_shape']}", flush=True)
        assert lift["supported"], f"{label} was refused: {lift['reasons']}"

    for label in ("z_zero", "declared_2d"):
        two_d = lifts[label]
        assert two_d["dimensions"] == 2, label
        assert two_d["invariant"] == [False, False, True], (
            f"{label}: MEEP 2-D is the X-Y plane with z invariant (vec.hpp "
            f"start_at_direction/stop_at_direction), got {two_d['invariant']}"
        )
        assert two_d["driver_shape"][2] == 1, label
        assert two_d["metallic_axes"][2] is False, (
            f"{label}: the invariant axis was walled. A PEC there zeroes Ex, Ey and Hz "
            f"— the whole TE polarization — and returns an exact smooth zero for it."
        )
        assert two_d["wraps"][2] is True, label
        assert two_d["nosize"] == [False, False, True], (
            f"{label}: nosize_direction must fire on the invariant axis and nowhere "
            f"else; got {two_d['nosize']}, which scales every source by the resolution."
        )
        assert two_d["meep_array_axes"] == [0, 1], label
        assert len(two_d["meep_shape"]) == 2, (
            f"{label}: MEEP's own array is {two_d['meep_shape']}, so it has already "
            f"dropped the axis it does not have; a 3-D readback would broadcast against "
            f"it instead of comparing."
        )

    implicit = data["implicit_special_kz"]
    assert implicit["meep_special_kz"] is True, (
        "MEEP no longer auto-enables special_kz for a z=0 cell with k_point.z != 0; "
        "simulation.py sets it from kz_2d, which defaults to 'complex'. If that "
        "changed, _infer_dimensions' use_2d disjunction needs rechecking too."
    )
    assert implicit["supported"], (
        f"the default kz_2d spelling is stepped as a 2-D run carrying beta; it must not "
        f"be refused. Reasons: {implicit['reasons']}"
    )
    assert implicit["dimensions"] == 2, (
        f"special_kz collapses to 2-D (use_2d's disjunction carries the flag), got "
        f"dimensions={implicit['dimensions']}."
    )
    assert implicit["beta"] == 0.25, (
        f"k_point.z must reach the driver as `beta`, MEEP's own fields-constructor slot; "
        f"got beta={implicit['beta']!r}."
    )
    assert implicit["k_point"][2] == 0.0, (
        f"the z component must NOT also ride as a Bloch phase — MEEP hands use_bloch only "
        f"Vector3(kx, ky) — but the lifted k_point is {implicit['k_point']}."
    )

    zero_d = data["zero_d"]
    assert not zero_d["supported"], (
        "dimensions=0 was accepted, but CPU MEEP raises 'Invalid dimensions in Volume: 0' "
        "for the same script — lifting it accepts what the oracle cannot build."
    )
    assert any("dimensions=0" in reason for reason in zero_d["reasons"]), (
        f"the refusal must name the declaration it rejects; got {zero_d['reasons']}"
    )
    assert any("dimensions=1" in reason for reason in zero_d["reasons"]), (
        f"the refusal must say what to declare instead; got {zero_d['reasons']}"
    )
    assert zero_d.get("raised") == "MeepSimulationNotLiftable", (
        f"a refused spelling must also DECLINE to build, not just report; "
        f"got {zero_d.get('raised')}"
    )

    for label in ("declared_1d",):
        one_d = lifts[label]
        assert one_d["dimensions"] == 1, f"{label}: MEEP 1-D is vol1d over cell_size.z"
        assert one_d["invariant"] == [True, True, False], (
            f"{label}: MEEP 1-D is the Z AXIS with x and y invariant (vol1d takes "
            f"cell_size.z), got {one_d['invariant']}"
        )
        assert one_d["driver_shape"][:2] == [1, 1], label
        assert one_d["metallic_axes"][:2] == [False, False], label
        assert one_d["nosize"] == [True, True, False], label
        assert one_d["meep_array_axes"] == [2], label
        assert len(one_d["meep_shape"]) == 1, label

    # NOT reduced, and the two ways of not being reduced differ from each other.
    unit = lifts["x_and_y_zero_three_d"]
    assert unit["dimensions"] == 3, (
        "mp.Vector3(0, 0, L) at dimensions=3 is a 3-D run: MEEP's vol3d turns a zero "
        "extent into ONE CELL and prints 'Working in 3D dimensions'."
    )
    assert unit["invariant"] == [False, False, False]
    assert unit["driver_shape"][:2] == [1, 1]
    assert unit["metallic_axes"] == [False, False, True], (
        "a unit direction is Periodic by MEEP's own default whatever k_point says "
        "(fields.cpp), so only z — the axis with real extent — carries the PEC box."
    )
    assert unit["nosize"] == [True, True, False]
    assert unit["meep_array_axes"] == [2], (
        "MEEP's array_slice keeps only the directions of more than one point "
        "(array_slice.cpp `if (n > 1)`), so this run's array is 1-D like MEEP's."
    )

    phased = lifts["z_zero_with_kz"]
    assert phased["dimensions"] == 3, (
        "a nonzero k_point.z blocks the 2-D collapse in Simulation._infer_dimensions; "
        "MEEP builds a 3-D run with a one-cell z axis carrying the Bloch phase."
    )
    assert phased["invariant"] == [False, False, False]
    assert phased["k_point"][2] == 0.25, (
        "the z phase must survive the lift: this run is not reduced, so there is no "
        "invariant axis for _lift_k_point to zero it on."
    )


def test_near2far_accumulates_here_and_meep_evaluates_the_far_field(tmp_path):
    """The near surface is this engine's; the far-field integral stays MEEP's.

    That split is the package's design test applied to one feature: the near-surface
    DFT runs inside the per-timestep loop over the grid, so it belongs here; the
    Green's-function evaluation runs once over a handful of host numbers, so
    reimplementing it would add a second thing to keep correct and buy no speed. The
    data is written back through ``load_near2far_data``, after which every far-field
    call is MEEP's own.

    Both PML settings are exercised, and that is the assertion with teeth. MEEP splits
    a cell with a PML into 27 structure chunks; the surface then spans 6 dft chunks per
    component and ``get_dft_data`` returns their CONCATENATION rather than one
    contiguous box. Both arrangements hold exactly the same number of values (3720
    here), so a size check passes either way — the no-PML case agreed at 9.6e-08 while
    the PML case was 1.27e+00 wrong, and only comparing values revealed it.

    Decimation is checked against MEEP's RESOLVED factor, not the requested one:
    ``args`` records 0 for "automatic", and MEEP's documented formula gives 10 on this
    case where MEEP itself resolved 9.

    Measured floors (4 runs: PML on/off x decimation 0/1): stored near data
    2.15e-07…4.85e-07 vs MEEP's own ``get_near2far_data``; ``get_farfield`` at three
    off-axis points 3.20e-08…1.27e-07; the ``get_farfields`` GRID readback (the
    separate ``_get_farfields_array`` path) 6.92e-08…3.06e-07. The NEGATIVE CONTROL
    pins that these comparisons resolve the per-chunk memory order: the same values
    with every chunk transposed from site-major to freq-major — identical size,
    identical value set, identical chunk boundaries — score 1.41e+00…1.42e+00 stored
    and 9.87e-01…1.03e+00 in the far field, seven orders above the parity floors.
    """
    data = _child(tmp_path, "gate", "near2far_is_migrated")

    for label, run in sorted(data["runs"].items()):
        print(f"[from_meep] near2far {label:12} chunks={run['chunks']} "
              f"dec={run['resolved_decimation']} stored={run['stored_error']:.2e} "
              f"far={run['far_error']:.2e} grid={run['farfields_error']:.2e} "
              f"transposed={run['transposed_far_error']:.2e}", flush=True)

        assert run["supported"], f"{label}: near2far was refused: {run['reasons']}"
        assert run["chunks"] == 4, (
            f"{label}: a z-normal region needs the four tangential accumulators "
            f"(Ex, Ey, Hx, Hy); got {run['chunks']}"
        )
        assert run["stored_size"] == run["meep_stored_size"], (
            f"{label}: packed {run['stored_size']} values against MEEP's "
            f"{run['meep_stored_size']}"
        )
        assert not run["far_is_all_zero"], (
            f"{label}: every far-field value is zero — the surface accumulated nothing, "
            f"which a size check cannot distinguish from a correct run"
        )
        assert run["resolved_decimation"] == run["meep_resolved"], (
            f"{label}: decimation {run['resolved_decimation']} against MEEP's resolved "
            f"{run['meep_resolved']}. A mismatch changes which steps are accumulated and "
            f"the weight on each — measured 2.2e-02 against 2.4e-07 once they agree."
        )
        assert run["stored_error"] < 5e-06, (
            f"{label}: stored near-field data {run['stored_error']:.3e} from CPU MEEP. "
            f"If only the PML cases fail, MEEP's chunk decomposition changed and "
            f"Near2FarMigration.packed must follow it."
        )
        assert run["far_error"] < 5e-06, (
            f"{label}: far fields {run['far_error']:.3e} from CPU MEEP's own evaluation "
            f"of the same surface"
        )
        assert run["farfields_error"] < 5e-06, (
            f"{label}: get_farfields grid {run['farfields_error']:.3e} from CPU MEEP — "
            f"the grid readback (_get_farfields_array) disagrees where the point "
            f"readback agreed"
        )
        # The negative control gives the comparison its teeth: the SAME values in a
        # deliberately wrong order (each chunk transposed from site-major to
        # freq-major) must NOT match. Size, value set and chunk boundaries are all
        # identical, so anything that passes the transposed pack is not checking
        # the layout at all.
        assert run["transposed_stored_error"] > 1e-01, (
            f"{label}: the transposed pack scored {run['transposed_stored_error']:.3e} "
            f"against MEEP's stored data — the stored comparison no longer resolves "
            f"the per-chunk memory order"
        )
        assert run["transposed_far_error"] > 1e-01, (
            f"{label}: the transposed pack's far field scored "
            f"{run['transposed_far_error']:.3e} — the far-field comparison no longer "
            f"resolves the per-chunk memory order"
        )


@requires_meep
@skip_without_meep
def test_point_near2far_and_flux_box_corners_match_meep(tmp_path):
    """The two near2far shapes the 3-D case cannot see, each pinned by its control.

    **The point region** (binary_grating_n2f.py): ``Near2FarRegion(center=pt)`` with
    size (0, 0, 0) on a 1-D-emulated ``(sx, 0, 0)`` cell. MEEP accepts it — the
    normal resolves through the nosize pad (dft.cpp:806-824) to mp.X (measured 0) —
    and loops the one-pixel periodic y axis over its two lattice images, one dft
    chunk per image with single-site weights w0 and w1 = 1 - w0 (measured 0.0 and
    1.0). This engine stores ONE y plane at the summed weight 1, and emits each MEEP
    chunk scaled by that chunk's own s0. The ``image_ignored`` control emits both
    images in full, which doubles the surface: a smooth, plausible far field at
    exactly 2x — measured 1.00e+00 relative while parity sits at 5.05e-07.

    **The flux-box corners** (antenna-radiation.py's four faces, ±1 weights): the
    corner site lies in TWO regions carrying the same component at OPPOSITE stored
    weights, and MEEP keeps one single-site chunk per region there. Matching
    accumulators by containment alone hands the corner the other face's sign —
    measured exactly 2.000e+00 on that one chunk, 3.6e-02 on the stored total,
    8.5e-03 in the far field, while every other chunk of the run sat at 1e-06. The
    ``corner_unfiltered`` control is that defect re-applied; the fix matches the
    chunk's own ``stored_weight``, which is this engine's entry weight by
    construction.

    Measured floors (this host, MEEP 1.33.0 single precision, NumPy backend):
    stored 9.33e-07 (point) / 4.83e-07 (box); far field 5.05e-07 / 1.66e-07;
    controls 1.00e+00 (image_ignored, point) and 4.26e-02 (corner_unfiltered, box).
    Each control equals the parity number on the OTHER case — the box has no
    one-pixel axis and the point region has no corner — which is why both cases
    exist. Both runs step (N - 0.5)*dt so this engine and CPU MEEP take exactly N
    steps.
    """
    data = _child(tmp_path, "gate", "n2f_point_region_and_flux_box_corners")

    for label, run in sorted(data["runs"].items()):
        print(f"[from_meep] n2f {label:6} steps={run['steps']} "
              f"stored={run['stored_error']:.2e} far={run['far_error']:.2e} "
              f"image_ignored={run['image_ignored_far_error']:.2e} "
              f"corner={run['corner_unfiltered_stored_error']:.2e}", flush=True)
        assert run["supported"], f"{label}: refused: {run['reasons']}"
        assert run["steps"][0] == run["steps"][1], (
            f"{label}: this engine took {run['steps'][0]} steps to CPU MEEP's "
            f"{run['steps'][1]}; the parity number compares different times."
        )
        assert run["stored_size"] == run["meep_stored_size"], (
            f"{label}: packed {run['stored_size']} values against MEEP's "
            f"{run['meep_stored_size']}"
        )
        assert run["meep_far_norm"] > 0.0, (
            f"{label}: MEEP's own far field is all zero; the run never accumulated "
            f"signal and the parity numbers below are about nothing"
        )
        assert run["stored_error"] < 5e-6, (
            f"{label}: stored near-field data {run['stored_error']:.3e} from CPU MEEP"
        )
        assert run["far_error"] < 5e-6, (
            f"{label}: far field {run['far_error']:.3e} from CPU MEEP's own "
            f"evaluation of the same surface"
        )
    # The controls, where each defect is visible: the doubled surface on the
    # one-pixel axis (point case), the flipped corner sign (box case).
    point, box = data["runs"]["point"], data["runs"]["box"]
    assert point["image_ignored_far_error"] > 1e-1, (
        f"point: emitting both lattice-image chunks in full scored "
        f"{point['image_ignored_far_error']:.3e} — the comparison can no longer see "
        f"a doubled surface, so its parity number means nothing"
    )
    assert box["corner_unfiltered_stored_error"] > 1e-2, (
        f"box: dropping the stored-weight filter scored "
        f"{box['corner_unfiltered_stored_error']:.3e} — the comparison can no longer "
        f"see the corner-sign defect it exists to pin"
    )


def test_near2far_stored_weight_is_meeps_signed_zero():
    """``s * w->weight`` built MEEP's way — componentwise, so the zero keeps its sign.

    ``s`` is a ``double`` in ``fields::add_dft_near2far`` (near2far.cpp:643-647), so
    C++ takes ``operator*(const T &, const complex<T> &)`` and scales the parts
    separately: ``-1 * (1+0j)`` is ``(-1.0, -0.0)``. Python's ``float * complex``
    promotes the scalar and runs the full product, whose imaginary part is
    ``(-1)*0.0 + 0.0*1.0 = +0.0``. The values are equal, ``==`` says so, and
    ``abs(a - b)`` is exactly ``0.0`` — but MEEP uses that sign to label WHICH near2far
    region a chunk came from, so getting it wrong makes two faces of a box
    indistinguishable at their shared corner.

    Pinned as a unit rule and not only end to end, because the end-to-end failure it
    causes needs a fold to become visible (see
    ``test_near2far_corner_claimants_are_separated``) and would otherwise go unnoticed
    on every cell without one.
    """
    from . import from_meep as module
    import math as _math

    for weight, sign, expected in (
        (complex(1.0), +1.0, (1.0, +0.0)),
        (complex(1.0), -1.0, (-1.0, -0.0)),
        (complex(-1.0), +1.0, (-1.0, +0.0)),
        (complex(-1.0), -1.0, (1.0, -0.0)),
    ):
        built = module._near2far_stored_weight(sign, weight)
        assert (built.real, built.imag) == expected, (
            f"s={sign} w={weight}: built {built!r}, MEEP's C++ builds {expected}"
        )
        assert _math.copysign(1.0, built.imag) == _math.copysign(1.0, expected[1]), (
            f"s={sign} w={weight}: the imaginary zero's sign is MEEP's region label; "
            f"built {built.imag!r} where MEEP has {expected[1]!r}"
        )
    # The two that a promoted (Python) product would collapse together.
    collapsed = -1.0 * complex(1.0)
    assert _math.copysign(1.0, collapsed.imag) == +1.0, (
        "Python's promoted product no longer differs from the componentwise one, so "
        "this rule has stopped describing anything — re-derive it before deleting."
    )


def test_same_stored_weight_separates_signed_zeros():
    """The claimant filter must read ``(1+0j)`` and ``(1-0j)`` as DIFFERENT labels.

    They are equal in value and their difference has modulus exactly ``0.0``, which is
    why the ``abs(a - b)`` form this replaced could not separate the two near2far
    regions meeting at a flux-box corner. Compared per part instead, with the sign of
    a zero significant.
    """
    from . import from_meep as module

    assert module._same_stored_weight(complex(1.0, 0.0), complex(1.0, 0.0))
    assert module._same_stored_weight(complex(-1.0, -0.0), complex(-1.0, -0.0))
    assert not module._same_stored_weight(complex(1.0, 0.0), complex(1.0, -0.0)), (
        "(1+0j) and (1-0j) compared equal; abs(a-b) is 0.0 for this pair, which is "
        "exactly the collision this filter exists to break"
    )
    assert not module._same_stored_weight(complex(-1.0, 0.0), complex(-1.0, -0.0))
    assert not module._same_stored_weight(complex(1.0, 0.0), complex(-1.0, 0.0))
    # A genuine rounding difference still passes; only the zero's sign is exact.
    assert module._same_stored_weight(complex(2.0, 3.0), complex(2.0 + 1e-15, 3.0))
    assert not module._same_stored_weight(complex(2.0, 3.0), complex(2.0 + 1e-6, 3.0))


def test_near2far_source_component_is_meeps_vc():
    """``c0 = direction_component(i == 0 ? Hx : Ex, fd[1 - j])`` — near2far.cpp:641-642.

    MEEP records it on every chunk as ``vc`` (dft.cpp:113), which makes it a label for
    the region's NORMAL that this engine can match a chunk against. The table below is
    MEEP's cyclic transverse order (X -> (Y, Z), Y -> (Z, X), Z -> (X, Y)) walked by
    hand; the live agreement with ``chunk.vc`` is measured in
    ``test_near2far_corner_claimants_are_separated``.
    """
    from . import from_meep as module

    expected = {
        # normal -> {(family, j): c0}, from fd[1 - j] of that normal's transverse pair.
        0: {("E", 0): "Hz", ("E", 1): "Hy", ("H", 0): "Ez", ("H", 1): "Ey"},
        1: {("E", 0): "Hx", ("E", 1): "Hz", ("H", 0): "Ex", ("H", 1): "Ez"},
        2: {("E", 0): "Hy", ("E", 1): "Hx", ("H", 0): "Ey", ("H", 1): "Ex"},
    }
    for normal, table in expected.items():
        transverse = module._NEAR2FAR_TRANSVERSE[normal]
        for (family, j), c0 in table.items():
            got = module._near2far_source_component(family, j, transverse)
            assert got == c0, (
                f"normal {normal}, {family}_fd{j}: c0 is {got}, MEEP's "
                f"direction_component({'Hx' if family == 'E' else 'Ex'}, "
                f"fd[{1 - j}]) is {c0}"
            )


@requires_meep
@skip_without_meep
def test_near2far_corner_claimants_are_separated(tmp_path):
    """Which accumulator answers for a chunk TWO near2far regions both contain.

    One component appears once per region, so containment alone is ambiguous wherever
    regions meet, and MEEP keeps one chunk per region there. Two labels ride on the
    chunk and both are used: its ``stored_weight`` and its ``vc`` (the equivalent-source
    component, which names the region's normal). Three cells, each isolating one:

    * **duplicate_plane** — two regions on the SAME plane at weights 1 and 3. Their
      ``vc`` is identical, so only the weight separates them: dropping the weight
      filter measured **6.32e-01** against a **3.45e-07** floor.
    * **double_mirror** — antenna-radiation.py's four-face box with Mirror(X) and
      Mirror(Y) at resolution 50, which is MEEP's own
      ``test_antenna_radiation::test_poynting_theorem`` cell. The fold splits the right
      face's ZERO-WEIGHT ghost column (MEEP's ``s0.x = 0`` interpolation row just
      inside the plane) into a one-site chunk at the corner, and the bottom face
      contains it too. The two chunks' stored weights are ``(1+0j)`` and ``(1-0j)`` —
      equal in value, different only in the sign of the zero imaginary part — so the
      old ``abs(a - b)`` filter could not tell them apart and the bottom face answered
      at FULL weight where MEEP stores exactly zero. Measured here at **1.84e-02** for
      the pre-fix implementation and **3.75e-02** with both labels dropped, against a
      **7.07e-07** floor; each filter ALONE brings it back to that floor. On the
      script's own 2500-step run the same defect read 2.69e-02 on the packed data,
      7.47e-03 of absolute error on that single site, and 1.40e-03 in the far field.
    * **complex_weights** — the same cell with region weights ``1+1j`` and ``-1-1j``,
      for which the tangential-pair signs make the two claimants' stored weights
      identical BIT FOR BIT. No reading of the weight can separate them; only ``vc``
      does — measured **1.84e-02** with it dropped and 7.11e-07 with it.

    The controls are the wrong implementations re-applied on the same accumulated data,
    so a control that stops being worse means this test has stopped measuring anything.
    """
    data = _child(tmp_path, "gate", "n2f_corner_claimant_filters")

    for label, run in sorted(data["runs"].items()):
        print(f"[from_meep] n2f claimants {label:16} stored={run['stored_error']:.2e} "
              f"weight_dropped={run['weight_filter_dropped']:.2e} "
              f"vc_dropped={run['source_filter_dropped']:.2e} "
              f"both={run['both_filters_dropped']:.2e} "
              f"abs={run['abs_weight_comparison']:.2e} "
              f"pre_fix={run['pre_fix']:.2e}", flush=True)
        assert run["supported"], f"{label}: refused: {run['reasons']}"
        assert run["steps"][0] == run["steps"][1], (
            f"{label}: this engine took {run['steps'][0]} steps to CPU MEEP's "
            f"{run['steps'][1]}; the parity number compares different times."
        )
        assert run["stored_size"] == run["meep_stored_size"], (
            f"{label}: packed {run['stored_size']} values against MEEP's "
            f"{run['meep_stored_size']}"
        )
        assert run["meep_stored_norm"] > 0.0, f"{label}: MEEP's own data is all zero"
        assert run["stored_error"] < 5e-6, (
            f"{label}: packed near-field data {run['stored_error']:.3e} from "
            f"get_near2far_data"
        )
        assert run["chunk_weights_all_matched_with_signed_zeros"], (
            f"{label}: MEEP chunk weights with no bit-identical entry weight — the "
            f"engine is not building `s * w->weight` the way MEEP does: "
            f"{run['unmatched_weights']}"
        )
        assert run["chunk_sources_are_entry_sources"], (
            f"{label}: a chunk's vc names an equivalent-source component no entry "
            f"claims, so the source filter is matching on a label this engine and "
            f"MEEP do not agree about"
        )

    # Each filter must be independently sufficient where it applies, and the pre-fix
    # combination must still be visibly wrong — otherwise nothing here is being tested.
    duplicate = data["runs"]["duplicate_plane"]
    assert duplicate["weight_filter_dropped"] > 1e-2, (
        f"duplicate_plane: dropping the stored-weight filter scored "
        f"{duplicate['weight_filter_dropped']:.3e}; two coincident regions at "
        f"different weights are no longer being separated by their weight"
    )
    assert duplicate["source_filter_dropped"] < 5e-6, (
        f"duplicate_plane: the weight alone should carry this case, but dropping the "
        f"vc filter scored {duplicate['source_filter_dropped']:.3e}"
    )
    folded = data["runs"]["double_mirror"]
    assert folded["both_filters_dropped"] > 1e-3, (
        f"double_mirror: with both claimant filters dropped the packed data scored "
        f"{folded['both_filters_dropped']:.3e}; the corner collision this cell exists "
        f"to pin is no longer reachable, so the parity number above proves nothing"
    )
    assert folded["pre_fix"] > 1e-3, (
        f"double_mirror: the pre-fix implementation (abs() comparison, no vc filter) "
        f"scored {folded['pre_fix']:.3e} — it was measured at 2.69e-02"
    )
    for filtered in ("weight_filter_dropped", "source_filter_dropped"):
        assert folded[filtered] < 5e-6, (
            f"double_mirror: {filtered} scored {folded[filtered]:.3e}; either label "
            f"alone is enough to separate the corner's two claimants here"
        )
    complex_weights = data["runs"]["complex_weights"]
    assert complex_weights["source_filter_dropped"] > 1e-3, (
        f"complex_weights: dropping the vc filter scored "
        f"{complex_weights['source_filter_dropped']:.3e}; with these weights the two "
        f"claimants' stored weights are identical bit for bit, so only vc can "
        f"separate them and this case has stopped proving that it does"
    )
    assert complex_weights["weight_filter_dropped"] < 5e-6, (
        f"complex_weights: vc alone should carry this case, but dropping the weight "
        f"filter scored {complex_weights['weight_filter_dropped']:.3e}"
    )


@requires_meep
@skip_without_meep
def test_monitor_region_negative_size_is_meeps_own_volume(tmp_path):
    """A NEGATIVE region size is the same volume, because MEEP's volume sorts corners.

    ``Volume.__init__`` builds ``mp.volume(center - size/2, center + size/2)``
    (python/simulation.py:443-449) and ``volume::volume`` takes
    ``min_corner = min(vec1, vec2)``, ``max_corner = max(vec1, vec2)``
    (src/vec.cpp:163-167), so MEEP never sees a reversed bound and ``size.x = -10.4``
    is ``+10.4`` about the same centre. Scripts write it — MEEP's own
    ``test_cavity_farfield.py`` gives a ``Near2FarRegion``
    ``size=mp.Vector3(2*dpml - sx)`` = -10.4 — and passing the sign through reached
    this engine's monitors as an inverted span and raised
    (``bounds on axis 0 are reversed: (5.2, -5.2)``) on a simulation the gate had
    accepted. Both of ``test_cavity_farfield``'s cases refused to lift for that alone;
    both now lift and reproduce 6 of MEEP's own 6 assertions.

    All three monitor kinds are checked, because each reached a different refusal, and
    the two spellings must be BIT-IDENTICAL rather than merely close — MEEP's are.
    """
    data = _child(tmp_path, "gate", "monitor_region_negative_size")

    for kind, run in sorted(data["runs"].items()):
        print(f"[from_meep] negative size {kind:9} +size={run['positive_vs_meep']:.2e} "
              f"-size={run['negative_vs_meep']:.2e} "
              f"|-size - +size|={run['negative_vs_positive']:.3e}", flush=True)
        assert run["supported"], f"{kind}: refused: {run['reasons']}"
        assert run["size"] > 0, f"{kind}: nothing was accumulated"
        assert run["positive_vs_meep"] < 5e-6, (
            f"{kind}: +size {run['positive_vs_meep']:.3e} from MEEP"
        )
        assert run["negative_vs_meep"] < 5e-6, (
            f"{kind}: -size {run['negative_vs_meep']:.3e} from MEEP — the negative "
            f"component is not being read as the same span"
        )
        assert run["negative_vs_positive"] == 0.0, (
            f"{kind}: the two spellings differ by {run['negative_vs_positive']:.3e}; "
            f"MEEP's own two differ by {run['meep_negative_vs_positive']:.3e}, so they "
            f"describe one volume and this engine must too"
        )
        assert run["meep_negative_vs_positive"] == 0.0, (
            f"{kind}: MEEP itself now distinguishes the two spellings "
            f"({run['meep_negative_vs_positive']:.3e}); its corner-sorting rule has "
            f"changed and this engine's reading has to be re-derived, not relaxed"
        )


@requires_meep
@skip_without_meep
def test_flux_region_may_extend_along_its_own_normal(tmp_path):
    """MEEP's flux VOLUME: a declared normal is allowed to carry extent.

    ``_add_fluxish_stuff`` puts a ``FluxRegion``'s volume into the list unexamined and
    ``add_dft_flux`` accumulates the four tangential components over whatever it is, so
    the monitor integrates the normal Poynting component over a volume rather than
    across a surface, with ``include_dV_and_interp_weights`` picking up one ``1/a`` per
    extended direction. MEEP's own ``test_visualization.py`` builds one
    (``FluxRegion(size=(4,4,4), direction=mp.X)``), and this engine used to accept the
    simulation at the gate and then raise ``Flux plane normal X must be the flat axis``
    — both of that file's cases, an accepted-but-does-not-lift gap.

    No special case was needed for it: the ladder treats the normal axis like any other
    and ``FluxMonitor._measure`` already takes one ``dx`` per extended axis, which is
    MEEP's own ``dV0`` rule. Each volume is measured beside the plane it degenerates to.
    An INFERRED normal with extent still raises, because there the axis would be a guess.
    """
    data = _child(tmp_path, "gate", "flux_region_volume")

    for label, run in sorted(data["runs"].items()):
        print(f"[from_meep] flux volume {label:12} meep={run['meep']:.6e} "
              f"ours={run['ours']:.6e} rel={run['relative']:.2e} "
              f"normal extent={run['extends_along_normal']:g}", flush=True)
        assert run["supported"], f"{label}: refused: {run['reasons']}"
        assert abs(run["meep"]) > 0.0, f"{label}: MEEP's own flux is zero"
        assert run["relative"] < 5e-6, (
            f"{label}: flux {run['relative']:.3e} from mp.get_fluxes over the same "
            f"region"
        )
    assert any(run["extends_along_normal"] > 0.0 for run in data["runs"].values()), (
        "no case in this test carries extent along its normal any more, so it has "
        "stopped testing the flux volume it exists for"
    )


@requires_meep
@skip_without_meep
def test_near2far_gate_and_lift_take_one_decision(tmp_path):
    """``gpu_compatibility.supported`` must equal ``lift_simulation`` not raising.

    The contract this family of fixes enforces, on the exact shapes that violated
    it (both measured on the corpus sweep):

    * a POINT ``Near2FarRegion`` on a 1-D-emulated cell — was supported=True then
      ``MeepSimulationNotLiftable`` out of ``_region_normal``, because the nosize
      pad (dft.cpp:806-824) was not transcribed and the resolver was only called
      from the migrations, never the gate;
    * an OVERSIZED face (wider than the cell) — was supported=True then a bare
      ``ValueError`` out of ``YeeRegionDFT.__post_init__``. MEEP itself accepts and
      clips it (measured: identical far fields at 0.0e+00), which this engine does
      not reproduce, so the agreeing verdict is refusal at the gate, by name;
    * an AMBIGUOUS point region in a full 2-D cell, where MEEP itself aborts
      ("Could not determine normal direction") — both sides must refuse, and this
      case is what keeps the nosize pad from over-resolving.
    """
    data = _child(tmp_path, "gate", "n2f_gate_lift_agreement")
    expected = {"point_region": "lifted", "oversized_face": "refused",
                "ambiguous_point": "refused"}
    for label, run in sorted(data["runs"].items()):
        print(f"[from_meep] gate/lift {label:15} supported={run['supported']} "
              f"lift={run['lift_outcome']}", flush=True)
        assert run["supported"] == (run["lift_outcome"] == "lifted"), (
            f"{label}: gpu_compatibility said supported={run['supported']} but the "
            f"lift {run['lift_outcome']} ({run['detail'][:200]}) — the pre-flight "
            f"verdict and the lift disagree, which is the one outcome the gate "
            f"exists to prevent."
        )
        assert run["lift_outcome"] == expected[label], (
            f"{label}: expected {expected[label]}, got {run['lift_outcome']} "
            f"({run['detail'][:200]})"
        )
        if not run["supported"]:
            assert run["reasons"], f"{label}: refused without naming a reason"


def test_cylindrical_near2far_accumulates_here_and_meep_evaluates_the_far_field(tmp_path):
    """The Dcyl near surface is this engine's too — ring measure, axis clip and all.

    Same split as the Cartesian case above, and the same three comparisons, on a
    ``dimensions=mp.CYLINDRICAL`` cell with a PML (which splits it into chunks) at
    resolutions 20 and 40 and azimuthal numbers m = 0, 1, 2. The near-to-far surface is
    a closed surface of revolution — the r wall plus two z caps — and the caps run from
    the AXIS outward, so the r = 0 clip is on the measured path rather than merely
    implemented.

    **The weights are compared to MEEP's own, with no field in the way.** For every chunk
    MEEP built, its per-site weight is reconstructed from that chunk's OWN
    ``dV0``/``dV1``/``s0``/``s1``/``e0``/``e1`` (dft.cpp:277, vec.hpp:381-383) and
    compared with this engine's ``_weights`` over the same sites. Measured 1.52e-16 …
    1.57e-16 over 586 sites (resolution 20) and 1031 (resolution 40) — exact to double
    precision, not merely close.

    The CONTROL for that comparison is the half-cell slip this package keeps getting
    bitten by: the same lift with the ring rebuilt at the CELL-CENTRE radius
    ``(index + 0.5)*dx`` instead of the component's own Yee radius, which is what
    ``Near2FarMonitor._face_weights`` legitimately uses because that monitor samples
    centres. It scores 2.17e-02 at resolution 20 and 1.06e-02 at resolution 40 — exactly
    ``0.5 / (r_max * a)``, half a cell measured against the outermost ring — fourteen
    orders above the floor above, and it moves only the shift-0 components (Ep, Hr, Ez),
    because Er/Hp/Hz sit at half-integer r where the two radii coincide.

    Measured floors over the six runs: stored near data 2.19e-07 … 3.16e-07 against
    MEEP's own ``get_near2far_data`` (the complex64 accumulator's floor), and
    ``get_farfield`` at three far points, evaluated by MEEP from this engine's data,
    4.73e-08 … 9.78e-08. The transposed-pack negative control scores 9.81e-01 … 1.15e+00
    stored and 6.15e-01 … 8.77e-01 in the far field.

    **The far-field agreement does NOT converge with resolution, and that is the point.**
    The standalone :class:`~.dft.Near2FarMonitor` differs from ``sim.get_farfield`` by
    2.09e-02 at resolution 20 falling to 5.33e-03 at 40 — second order — because it
    samples CELL CENTRES and MEEP samples Yee sites. This path has no such difference:
    ``YeeRegionDFT`` stores MEEP's own sites, so the residual is the stepper's float32
    floor at both resolutions (9.78e-08 -> 7.13e-08 for m = 0) and there is nothing left
    to converge. A second-order trend here would mean the packing had introduced a
    sampling offset of its own.
    """
    data = _child(tmp_path, "gate", "near2far_cylindrical_is_migrated")

    for label, run in sorted(data["runs"].items()):
        print(f"[from_meep] near2far cyl {label:11} sites={run['sites_compared']} "
              f"weights={run['weight_error']:.2e} "
              f"(centred-radius control {run['centred_radius_weight_error']:.2e}) "
              f"stored={run['stored_error']:.2e} far={run['far_error']:.2e}", flush=True)

        assert run["supported"], f"{label}: cylindrical near2far was refused: {run['reasons']}"
        assert run["chunks"] == 12, (
            f"{label}: three regions need four tangential accumulators each — a Dcyl "
            f"z-normal cap takes (Er, Ep, Hr, Hp) and the r wall (Ep, Ez, Hp, Hz), "
            f"near2far.cpp:596-620; got {run['chunks']}"
        )
        assert run["stored_size"] == run["meep_stored_size"], (
            f"{label}: packed {run['stored_size']} values against MEEP's "
            f"{run['meep_stored_size']}"
        )
        assert run["sites_compared"] > 0, (
            f"{label}: no chunk was matched to an accumulator, so the weight comparison "
            f"below asserted nothing"
        )
        assert not run["far_is_all_zero"], (
            f"{label}: every far-field value is zero — the surface accumulated nothing"
        )
        assert run["weight_error"] < 1e-12, (
            f"{label}: per-site weights {run['weight_error']:.3e} from MEEP's own "
            f"dV0 + dV1*loop_i2 times its own boundary ladder. These are compared with "
            f"no field in the way, so this is the ring measure and the axis clip alone."
        )
        assert run["centred_radius_weight_error"] > 1e-03, (
            f"{label}: the centred-radius control scored "
            f"{run['centred_radius_weight_error']:.3e} — the weight comparison no longer "
            f"resolves the half-cell choice between the Yee radius and the cell centre, "
            f"which is the whole reason it exists"
        )
        assert run["stored_error"] < 5e-06, (
            f"{label}: stored near-field data {run['stored_error']:.3e} from CPU MEEP"
        )
        assert run["far_error"] < 5e-06, (
            f"{label}: far fields {run['far_error']:.3e} from CPU MEEP's own evaluation "
            f"of the same surface"
        )
        assert run["transposed_stored_error"] > 1e-01, (
            f"{label}: the transposed pack scored {run['transposed_stored_error']:.3e} "
            f"against MEEP's stored data — the comparison no longer resolves the "
            f"per-chunk memory order"
        )
        assert run["transposed_far_error"] > 1e-01, (
            f"{label}: the transposed pack's far field scored "
            f"{run['transposed_far_error']:.3e}"
        )

    # The residual must NOT fall with resolution: this path samples MEEP's own Yee
    # sites, so there is no sampling offset to converge away, and one appearing would
    # be the packing introducing an offset of its own.
    for m in (0, 1, 2):
        coarse = data["runs"][f"res20_m{m}"]["far_error"]
        fine = data["runs"][f"res40_m{m}"]["far_error"]
        assert fine > 0.2 * coarse, (
            f"m={m}: the far-field residual fell from {coarse:.3e} to {fine:.3e} when "
            f"the resolution doubled. That is the signature of a SAMPLING difference "
            f"(the standalone cell-centre monitor converges at second order, 2.09e-02 -> "
            f"5.33e-03); through this path there should be none, only the float32 floor."
        )


@requires_meep
@skip_without_meep
def test_meeps_own_step_functions_are_hosted_on_this_stepper(tmp_path):
    """``mp.after_sources(mp.Harminv(...))`` and ``mp.at_every`` must run here as they run in MEEP.

    MEEP's step-function wrappers are closures written against ``mp.Simulation``:
    ``after_sources`` calls ``sim.fields.last_source_time()`` and ``sim.round_time()``
    (``python/simulation.py:5045-5051``), ``at_every`` calls ``sim.round_time()`` and
    ``sim.fields.dt`` (``:5119-5123``), ``Harminv`` calls ``sim.meep_time()``,
    ``sim.get_field_point()`` (``:1149-1153``) and ``sim.sources`` (``:1158``).
    ``run_on_gpu`` steps a driver, not a ``Simulation``, so without a facade the first
    of those raises ``AttributeError: 'Fields' object has no attribute
    'last_source_time'`` and the whole family — 29 rows of MEEP's own test corpus —
    is undriveable.

    Both codes run the SAME wrappers on the SAME cell for the same 120 time units, and
    four things are compared:

    * the number of samples ``Harminv`` collected and its ``data_dt``, EXACTLY. A
      facade whose clock is off by a step collects one sample too many or too few and
      still returns a smooth spectrum;
    * every time ``at_every(0.7, ...)`` fired, EXACTLY. This is the one place the
      float32 rounding in ``fields::round_time`` (``src/meep.hpp:1892``) is
      observable: MEEP tests ``t >= tlast + dt - 0.5*fields.dt`` against the ROUNDED
      time, and carrying the double through instead moves a fire by one step wherever
      ``t*dt`` and its float32 rounding straddle the threshold;
    * the sampled signal itself, so that agreeing counts cannot be agreeing on an
      empty record;
    * the mode ``Harminv`` recovered from it.

    Measured — MEEP's own ring resonator (``python/tests/test_ring.py``) at resolution
    10 with the padding trimmed to a 100x100 cell, one ``mp.Mirror(mp.Y)``, ``until=400``,
    NumPy backend, MEEP 1.33.0 single precision:

    * 8000 steps to t=400.0 on both codes;
    * **6001** Harminv samples on both, ``data_dt`` **0.049999999999954525** on both
      (the same bits, not merely close);
    * **571** ``at_every(0.7)`` fires on both, at identical times — and the times are
      visibly the float32 ones, ``0.699999988079071``, ``399.70001220703125``, which
      is the rounding this test exists to pin;
    * sampled field record **8.94e-06** relative, peak |Ez| 0.0758;
    * mode 0 frequency **0.11806894693** here against MEEP's **0.11806901108**,
      5.4e-07 relative (Q 78.045 both).
    """
    data = _child(tmp_path, "gate", "hosted_step_functions")
    print(f"[from_meep] hosted step functions: samples {data['engine_samples']}/"
          f"{data['meep_samples']}, fires {len(data['engine_fires'])}/"
          f"{len(data['meep_fires'])}, signal {data['signal_error']:.2e}", flush=True)

    assert data["engine_steps"] == data["meep_steps"] and (
        abs(data["engine_time"] - data["meep_time"]) < 1e-9), (
        f"the two codes did not step the same run: {data['engine_steps']} steps to "
        f"t={data['engine_time']:g} here, {data['meep_steps']} to "
        f"t={data['meep_time']:g} in MEEP"
    )
    assert data["engine_samples"] == data["meep_samples"], (
        f"mp.Harminv collected {data['engine_samples']} samples here and "
        f"{data['meep_samples']} in MEEP; the step function is called a different "
        f"number of times, or after_sources opens at a different time"
    )
    assert data["engine_samples"] > 400, (
        f"only {data['engine_samples']} samples were collected, which is too few for "
        f"the count to be evidence of anything (the source's own tail is 100 of the "
        f"120 time units, so after_sources should open around step 2400 of 2880)"
    )
    assert abs(data["engine_data_dt"] - data["meep_data_dt"]) < 1e-12, (
        f"Harminv's own sample interval differs: {data['engine_data_dt']!r} here, "
        f"{data['meep_data_dt']!r} in MEEP"
    )
    assert data["engine_fires"] == data["meep_fires"], (
        f"mp.at_every fired at different times: {len(data['engine_fires'])} fires here "
        f"vs {len(data['meep_fires'])} in MEEP, first difference at "
        f"{next((i for i, (a, b) in enumerate(zip(data['engine_fires'], data['meep_fires'])) if a != b), None)}. "
        f"round_time() is float(t*dt) — single precision — not the double."
    )
    assert data["signal_max"] > 1e-6, (
        f"the collected signal peaks at {data['signal_max']:.3e}; a record of zeros "
        f"would agree with MEEP's on every count above"
    )
    assert data["signal_error"] < 5e-5, (
        f"the sampled field record is {data['signal_error']:.3e} from MEEP's (measured "
        f"8.94e-06 over 6001 samples of an 8000-step resonator run)"
    )
    # The driver-native step function that shared the run: `meep_gpu.Harminv` asks with
    # this engine's component NAME and a plain triple, and is not wrapped in
    # after_sources, so it must see every step plus the sample before the first.
    assert data["native_samples"] == data["engine_steps"] + 1, (
        f"the driver-native step function collected {data['native_samples']} samples over "
        f"{data['engine_steps']} steps; MEEP's protocol samples at t0 and after every step, "
        f"so it must be steps + 1"
    )
    assert data["native_modes"], (
        "the driver-native step function ran but recovered no mode, so the facade's "
        "get_field_point may be answering it with something constant"
    )
    assert data["engine_modes"] and data["meep_modes"], "neither code recovered a mode"
    ours, theirs = data["engine_modes"][0][0], data["meep_modes"][0][0]
    assert abs(ours - theirs) / abs(theirs) < 1e-4, (
        f"the recovered mode frequency is {ours!r} here and {theirs!r} in MEEP"
    )


@requires_meep
@skip_without_meep
def test_a_step_function_the_facade_cannot_host_is_refused_by_name(tmp_path):
    """Reaching past the facade must raise, not silently read the un-stepped simulation.

    ``lift_simulation`` calls ``sim.init_sim()``, so the ``mp.Simulation`` behind a
    lifted run is fully built and its fields are all zero. A facade that fell back to
    it would answer ``sim.output_volume``, ``sim.get_array(...)`` or
    ``sim.filename_prefix`` with a real, correctly shaped, entirely wrong value —
    exactly the silent-zero failure this package refuses everywhere else.

    ``StepFunctionNotHosted`` is deliberately not an ``AttributeError``: MEEP's step
    functions reach state by attribute access, and an ``AttributeError`` would be
    swallowed by every ``getattr(sim, name, default)`` in the calling code.
    """
    data = _child(tmp_path, "gate", "unhosted_step_function_is_refused")
    assert data["raised"] is True, (
        f"reaching for sim.output_volume from a hosted step function returned "
        f"{data['raised']!r} instead of raising StepFunctionNotHosted; an "
        f"AttributeError here would be swallowed by MEEP's own getattr defaults"
    )
    assert "output_volume" in data["message"], (
        f"the refusal must name what was asked for; got {data['message']!r}"
    )


@requires_meep
@skip_without_meep
def test_two_run_normalization_idiom_matches_meeps_own(tmp_path):
    """The dominant flux idiom in MEEP's corpus, driven end to end and pinned twice.

    Run once WITHOUT the structure, save the flux transform, run again WITH it having
    loaded the saved transform NEGATED, and read the reflected power off the
    difference. That is ``test_bend_flux``, ``test_refl_angular``,
    ``test_binary_grating``, ``test_mode_decomposition`` and most of the examples
    corpus, and it is where MEEP's published normalized transmission and reflection
    constants live.

    It could not be driven here at all. MEEP's ``sim.load_minus_flux_data`` reaches
    ``flux.E``, whose lazy ``swigobj`` property calls ``init_sim()``, and an
    initialized simulation is a refusal of this lift — the boundary conditions a
    caller can install on one are invisible to any reader (``gpu_compatibility``).
    The load is therefore declared to ``run_on_gpu`` and applied to this engine's own
    accumulators after the lift, leaving MEEP untouched; 16 rows of MEEP's test
    corpus were blocked on exactly this.

    A Fresnel slab, so the answer is known independently of both codes. Measured
    (resolution 200, 11 frequencies over 0.4-0.8 um, n1 = 1.4, n2 = 3.5):

    * the normalization run's raw spectrum against ``mp.get_fluxes``: **1.2e-06**
    * this engine's reflectance against MEEP's own two-run reflectance: **5.3e-04**
      — larger than the raw-flux floor because the subtraction removes ~84% of the
      field and reports the residue, which is the idiom's own conditioning rather
      than a defect
    * against the analytic normal-incidence Fresnel value: **1.46e-02** for this
      engine and **1.53e-02** for CPU MEEP on the same grid, against MEEP's own
      3% bar in ``test_refl_angular.py``. That residue is grid dispersion and
      falls as 1/resolution**2 (48% at res 40, 6.3% at 100, 1.5% at 200) — shared
      by both codes, which is what makes it physics and not a lift.

    Two controls, both of which complete and return a smooth, plausible reflectance:

    * **no normalization at all** — the raw net flux: **5.5e+00** relative;
    * **the load WITHOUT the negation** (``load_flux_data``, MEEP's own non-minus
      spelling), which ends the second run at ``E2 + E1`` instead of ``E2 - E1``:
      **2.2e+01** relative. That is the sign the whole idiom turns on.

    The convention itself — subtract the FIELDS, both sides, not the powers — is
    pinned at the accumulator in
    ``test_dft.py::test_load_minus_flux_data_subtracts_THE_FIELDS_not_the_power``,
    where a case exists on which each corruption reads a different number.
    """
    data = _child(tmp_path, "gate", "two_run_normalization_idiom")
    print(f"[from_meep] two-run normalization: empty {data['empty_flux_error']:.2e}, "
          f"reflectance {data['reflectance_error']:.2e}, Fresnel {data['fresnel_error']:.2e} "
          f"(MEEP's own {data['meep_fresnel_error']:.2e}), controls "
          f"unnormalized {data['unnormalized_error']:.2e}, "
          f"unnegated {data['unnegated_error']:.2e}", flush=True)

    assert data["empty_flux_error"] < 5e-06, (
        f"the normalization run itself must be at the flux floor before its "
        f"difference means anything; got {data['empty_flux_error']:.3e}"
    )
    assert data["reflectance_error"] < 5e-03, (
        f"the normalized reflectance must reproduce MEEP's own two-run answer; got "
        f"{data['reflectance_error']:.3e}"
    )
    assert data["fresnel_error"] < 0.03, (
        f"and it must satisfy the analytic Fresnel value at MEEP's own tolerance "
        f"(test_refl_angular uses 0.03); got {data['fresnel_error']:.3e} against "
        f"MEEP's {data['meep_fresnel_error']:.3e}"
    )
    assert data["unnegated_error"] > 1e-01, (
        f"loading WITHOUT the negation must be a plainly different answer, or this "
        f"test cannot see a dropped sign; got {data['unnegated_error']:.3e}"
    )
    assert data["unnormalized_error"] > 1e-03, (
        f"and skipping the normalization entirely must be distinguishable too; got "
        f"{data['unnormalized_error']:.3e}"
    )


def test_multi_region_add_flux_is_summed_the_way_meep_sums_it(tmp_path):
    """A flux BOX is one MEEP object over several planes, and every plane must be lifted.

    ``sim.add_flux(fcen, df, nfreq, *FluxRegions)`` is the standard flux-box idiom, and
    ``mp.get_fluxes`` returns the regions' weighted sum: MEEP appends every region's
    chunks to one ``dft_flux`` and ``dft_flux::flux()`` sums the whole list
    (dft.cpp:533-556), each region's ``weight`` folded into its own E chunks.

    The lift answered for the FIRST region only, and the failure was loud but late.
    MEEP stores ``NO_DIRECTION`` as the monitor's ``normal_direction`` for exactly the
    multi-region case ("if the volume list has > 1 entry, store NO_DIRECTION",
    dft.cpp:639-641), so the lift reached ``add_flux_monitor`` and raised
    ``Unknown flux direction 5`` — after ``gpu_compatibility`` had already reported the
    simulation supported, which is the one outcome the pre-flight exists to prevent.

    The region WEIGHT is the second half of the same gap, and this test found it: a bare
    ``driver.add_flux_monitor`` has no weight, so the ``weight=-1`` face of a flux box —
    lifted alone, it is still a one-region monitor — returned ``+flux`` where MEEP
    returns ``-flux``, measured at exactly 2.0 relative. A single region keeps the bare
    monitor only at the default weight; anything else goes through the summing wrapper,
    which is why the ``bottom`` case below is a ``FluxMigration`` with one part.

    Measured against CPU MEEP's own ``mp.get_fluxes`` on the same run: 1.33e-07 for the
    top plane alone, 1.18e-07 for the weighted bottom plane alone, and 1.25e-07 for the
    two-region monitor — against a control of 5.16e-01 for the first region alone, which
    is what the summed number would have been had the second plane stayed unlifted.

    The run reaches t = 80 deliberately. At t = 16 the same three comparisons read
    7.5e-02, 7.4e-02 and 7.4e-02: a Gaussian at ``fwidth = 0.3`` peaks near t = 17, so a
    short run measures the source's exponential turn-on tail and nothing else. Every one
    of those numbers would have read as a summation bug.
    """
    data = _child(tmp_path, "gate", "multi_region_flux_is_summed")

    for label, run in sorted(data["runs"].items()):
        print(f"[from_meep] multi-region flux {label:7} {run['kind']:15} "
              f"normal_direction={run['meep_normal_direction']} "
              f"err={run['error']:.2e}", flush=True)
        assert run["supported"], f"{label}: refused: {run['reasons']}"
        assert run["error"] < 5e-06, (
            f"{label}: flux spectrum {run['error']:.3e} from CPU MEEP's mp.get_fluxes"
        )

    single = data["runs"]["top"]
    assert single["kind"] == "FluxMonitor", (
        "a one-region monitor must still migrate to a bare driver monitor — that is "
        "every case the flux path was built and measured against, and wrapping it would "
        "change what `result.get_flux_spectrum` hands back for all of them"
    )
    assert single["meep_normal_direction"] in (0, 1, 2)

    box = data["runs"]["both"]
    assert box["kind"] == "FluxMigration"
    assert box["meep_normal_direction"] == 5, (
        "MEEP stores NO_DIRECTION for a multi-region flux monitor; if it no longer does, "
        "the per-region resolution below is answering a question nobody is asking"
    )
    assert box["first_region_only_error"] > 1e-01, (
        f"the first region alone scored {box['first_region_only_error']:.3e} against "
        f"MEEP's sum — the comparison no longer notices a dropped plane, which is the "
        f"defect this test was written for"
    )


def test_dft_monitors_on_the_simulation_are_migrated_rather_than_refused(tmp_path):
    """A monitor MEEP already holds is rebuilt on the driver, and it accumulates.

    The DFT was never the missing piece — it is this engine's, it runs inside the
    per-timestep loop and it runs on the GPU. What was missing was carrying the
    SPECIFICATION across, so a script that had already called ``sim.add_flux`` was
    refused rather than lifted. Monitors were the single largest blocker in the
    corpus survey (31 of 57 scripts, and the only blocker for 12 of them).

    Three things are asserted, in the order they can go wrong:

    1. **It accumulated at all.** An un-migrated monitor is not an error — it is a
       correctly shaped, correctly typed spectrum of zeros, which is why the refusal
       existed. Shape agreement proves nothing here.
    2. **It accumulated the SAME thing** as the hand-added monitor the old refusal
       told callers to write. Bit equality, on one driver and one run, so nothing
       about the physics is being compared — only the specification transfer.
    3. **The decimation came across.** MEEP keeps it positionally in ``args`` rather
       than as a named attribute, so reading it is version-fragile, and getting it
       wrong does not raise: it shifts the spectrum by ~1.4e-06. Measured while
       building this: a missed read left migrated and manual differing at 1.42e-06
       on an otherwise identical run.
    """
    data = _child(tmp_path, "gate", "monitors_are_migrated")

    assert data["supported"], f"monitors are still refused: {data['reasons']}"
    assert data["n_migrated"] == 2, data["n_migrated"]

    assert not data["flux_is_all_zero"], (
        "the migrated flux monitor returned an all-zero spectrum — it was rebuilt but "
        "never accumulated, which is exactly the outcome the refusal used to prevent."
    )
    assert data["dft_max_abs"] > 0.0, (
        f"the migrated DFT monitor accumulated nothing (max |Ez| = {data['dft_max_abs']})"
    )
    assert data["flux_matches_manual"], (
        f"the migrated monitor and a hand-added one disagree on the same run: "
        f"{data['flux']}. The specification did not transfer exactly."
    )
    assert data["frequencies"] == data["meep_frequencies"], (
        f"frequencies drifted in migration: {data['frequencies']} vs MEEP's "
        f"{data['meep_frequencies']}. They are taken off the monitor rather than "
        f"re-derived from fcen/df/nfreq precisely so this cannot happen."
    )
    for asked, reads in sorted(data["decimation_readback"].items()):
        flux_read, fields_read, mode_read, mode_yee_read = reads
        assert flux_read == int(asked) and fields_read == int(asked), (
            f"decimation_factor={asked} read back as DftFlux={flux_read}, "
            f"DftFields={fields_read}. MEEP keeps it positionally in args; if its "
            f"signature moved, _DECIMATION_ARG_INDEX must move with it."
        )
        # `add_mode_monitor` builds a DftFlux with `yee_grid` where `add_flux` puts
        # the decimation (simulation.py:3510 against :3531-3534), so reading index 2
        # for both landed on a BOOL. The type guard turned that into 0 — MEEP's own
        # default — which is why the mis-read stayed invisible for every script that
        # left the factor alone. Both yee_grid spellings are checked because False is
        # what index 2 holds in the common case and True in the adjoint one.
        assert mode_read == int(asked) and mode_yee_read == int(asked), (
            f"add_mode_monitor(decimation_factor={asked}) read back as {mode_read} "
            f"(yee_grid=False) and {mode_yee_read} (yee_grid=True). Its DftFlux "
            f"carries the factor at args[3], not args[2] — the engine would step its "
            f"accumulator at MEEP's automatic decimation instead of the one asked for."
        )




@requires_meep
@skip_without_meep
def test_a_dc_dft_monitor_accumulates_what_meeps_own_accumulator_does(tmp_path):
    """Frequency 0 is a DFT bin like any other, and the assertion is ABSOLUTE.

    ``update_dft`` has no branch on omega (dft.cpp:266-269): the phase factor is
    ``polar(1.0, omega*time) * scale``, which at omega = 0 is exactly ``scale``, and
    the accumulate is the same multiply-add. So a DC monitor is a running time
    integral of the field and needs no new code — what it needed was for the gate and
    ``dft.normalize_frequencies`` to stop refusing it by name.

    THE RESIDUAL IS COMPARED ABSOLUTELY, NOT RELATIVELY, and that is the measurement
    rather than a softened bar. On this cell (3x3, resolution 10, eps 2, PML 0.6, a
    Gaussian Ez dipole, 400 steps) the DC bin's own magnitude is 0.0189 while the
    0.8 bin's is 21.4 — a factor of 1100 — because a lossless pulse's time integral
    is a near-total cancellation. The absolute residuals are the SAME:

        dc_only  f=0      |meep| 1.891e-02   abs 1.348e-06   rel 7.13e-05
        mixed    f=0      |meep| 1.939e-02   abs 2.305e-06   rel 1.19e-04
        mixed    f=0.8    |meep| 2.140e+01   abs 2.610e-06   rel 1.22e-07

    Both accumulators sit on the same ~2e-06 float32 accumulation floor; the DC bin's
    larger RELATIVE number is its own cancellation and says nothing about the rule.
    Quoting only the relative figure would report a DC accumulator as three orders
    worse than an AC one, which it is not.

    THE DECIMATION IS THE ONLY DC-SENSITIVE LINE IN dft.cpp, so both shapes are run.
    ``fields::add_dft`` sets the automatic factor to
    ``floor(1/(dt*(freq_max+src_freq_max)))`` only when ``freq_max > 0``, and falls
    through to 1 otherwise (dft.cpp:207-210). Measured here: 1 for the DC-only monitor
    and 10 for the mixed one. This engine reads the factor MEEP RESOLVED off the live
    chunk (``_resolved_decimation``) instead of re-deriving that rule, which is why
    the DC case is correct without a line of new code — and this is the assertion that
    would notice if it ever started re-deriving it.
    """
    data = _child(tmp_path, "gate", "dc_dft_monitor")

    for label, (supported, reasons) in sorted(data["supported"].items()):
        assert supported, f"a DC monitor is still refused ({label}): {reasons}"
    assert data["decimation"] == {"dc_only": 1, "mixed": 10}, (
        f"MEEP's resolved decimation came back as {data['decimation']}; dft.cpp:207-210 "
        f"falls through to 1 when freq_max == 0 and applies the automatic rule "
        f"otherwise, so a DC-only monitor must read 1 and the mixed one must not."
    )
    for key, (signal, residual, shape, reference_shape) in sorted(data["bins"].items()):
        assert shape == reference_shape, (key, shape, reference_shape)
        assert signal > 0.0, f"{key}: MEEP's own accumulator is empty, so nothing is pinned"
        assert residual < 1e-5, (
            f"{key}: the migrated accumulator differs from MEEP's own by {residual:.4e} "
            f"absolute (|meep| = {signal:.4e}). Measured 1.35e-06 / 2.31e-06 / 2.61e-06 "
            f"across the three bins — one float32 accumulation floor, shared by the DC "
            f"bin and the 0.8 bin whose magnitudes differ by a factor of 1100."
        )
    dc_residual = data["bins"]["mixed:0.0"][1]
    ac_residual = data["bins"]["mixed:0.8"][1]
    assert dc_residual < 10.0 * ac_residual, (
        f"the DC bin's absolute residual ({dc_residual:.4e}) is more than ten times the "
        f"AC bin's ({ac_residual:.4e}) on the same run and the same monitor. They share "
        f"an accumulator and a decimation, so they must share a floor; a gap here means "
        f"the DC bin is taking a different path, not that DC is harder."
    )


@requires_meep
@skip_without_meep
def test_structured_dispersive_media_lift_through_the_sigma_reader():
    """Media differing in E_susceptibilities lift on a sigma-reader MEEP — end to end.

    The consumer of meep-sigma-reader.patch (the design notes (meep-sigma-reader-plan)):
    a cell with a Lorentz block (eps 4, f=1, gamma=0.1, sigma=0.4) and a Drude block
    (eps 2, f=2, gamma=0.05, sigma=0.7) in vacuum — three media, differing exactly in
    epsilon_diag and E_susceptibilities — accepted by the gate, its per-point sigma
    volumes read back through ``fields.get_susceptibility_sigma`` at each component's
    own Yee sites and installed with ``add_susceptibility(sigma={volumes})``.
    Measured on landing (patched MEEP 1.33.0, single precision): Ez/Hx/Hy at
    **1.06e-07..1.49e-07** against CPU MEEP stepping the same cell, with the
    terms-dropped control at **1.9e-01** — six orders of margin. The reader itself is
    validated separately by parity/meep_gpu/validate_sigma_reader.py (enumeration,
    bulk, point-sampling-vs-smoothing, the chi1inv(freq) consistency identity at
    1.6e-09, and symmetry-folded reads).

    SKIPPED on an unpatched MEEP, where there is no reader for this test to be about.
    That is no longer the same as the cell being refused: the same cell lifts there
    through the chi1inv RECOVERY instead (``meep_gpu.sigma_recovery``, pinned end to
    end by ``test_sigma_recovery.py`` with the capability probe forced False, and
    compared point by point against this reader at 5.7e-08). What the
    ``media_differ_in_chi3`` refusal still covers on every MEEP is a chi2/chi3
    difference, which has no reader anywhere — measured invisible in chi1inv at exactly
    0.000e+00 on both the E and the D row. A CONDUCTIVITY difference is no longer part
    of it: `_read_conductivity_volumes` recovers that from the D row.
    """
    import meep as mp

    from .from_meep import gpu_compatibility, run_on_gpu

    if not hasattr(mp.fields, "get_susceptibility_sigma"):
        pytest.skip("this MEEP lacks the sigma reader; build one with MEEP_SIGMA_PATCH=1 "
                    "via parity/meep_gpu/build_meep_133_macos.sh")

    def build():
        geometry = [
            mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                     material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                         mp.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=0.4)])),
            mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                     material=mp.Medium(epsilon=2.0, E_susceptibilities=[
                         mp.DrudeSusceptibility(frequency=2.0, gamma=0.05, sigma=0.7)])),
        ]
        return mp.Simulation(cell_size=mp.Vector3(4.0, 4.0, 0.0), resolution=10,
                             geometry=geometry, force_complex_fields=True,
                             sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                                component=mp.Ez,
                                                center=mp.Vector3(0, -1.2, 0))])

    verdict = gpu_compatibility(build())
    assert verdict.supported, f"gate refused a sigma-readable cell: {verdict.reasons}"

    result = run_on_gpu(build(), until=6.0, prefer_gpu=False)
    try:
        ours = {name: np.asarray(result.get_array(name)) for name in ("Ez", "Hx", "Hy")}
    finally:
        result.close()

    reference = build()
    reference.run(until=6.0)
    for name, mine in ours.items():
        theirs = np.asarray(reference.get_array(component=getattr(mp, name), cmplx=True))
        error = float(np.linalg.norm(mine - theirs) / np.linalg.norm(theirs))
        assert error < 5e-6, f"structured-dispersive {name} rel L2 {error:.3e}"


@requires_meep
@skip_without_meep
def test_point_dft_region_readback_is_finite_at_any_alignment():
    """A POINT dft region reads back finite as-accumulated cells at both alignments.

    The upstream contrast this pins against (port reference §5b, found 2026-08-04):
    MEEP 1.29 returns an UNINITIALIZED buffer for a half-cell-aligned point region
    (measured 5.8e+28), and MEEP 1.33 — post-#3010, whose regression test covers
    only 1-D — returns exactly 0+0j in a 2-D cell where the surrounding field is
    ~1e-4. This engine's `get_dft_region` contract is the region's cells AS
    ACCUMULATED, so the assertion here is finiteness, a live magnitude, and the
    documented uncollapsed shape — the guard against ever importing either broken
    collapse convention.
    """
    import meep as mp

    from .from_meep import run_on_gpu

    sim = mp.Simulation(cell_size=mp.Vector3(2.0, 2.0, 0.0), resolution=10,
                        force_complex_fields=True,
                        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                           component=mp.Ez, center=mp.Vector3(-0.4, 0.3, 0))])
    monitors = {
        "aligned": sim.add_dft_fields([mp.Ez], 1.0, 0, 1,
                                      center=mp.Vector3(0.2, 0.1), size=mp.Vector3()),
        "half_aligned": sim.add_dft_fields([mp.Ez], 1.0, 0, 1,
                                           center=mp.Vector3(0.25, 0.15), size=mp.Vector3()),
    }
    result = run_on_gpu(sim, until=4.0, prefer_gpu=False)
    try:
        for label, monitor in monitors.items():
            values = np.asarray(result.get_dft_region(monitor, "Ez", 0))
            assert np.isfinite(values).all(), (
                f"{label} point region read back non-finite values {values.ravel()[:4]} — "
                f"the uninitialized-buffer class")
            peak = float(np.abs(values).max())
            assert 1e-8 < peak < 1e2, (
                f"{label} point region |peak| = {peak:.3e}: outside any plausible field "
                f"scale for this run (an exact 0 here is the post-#3010 2-D collapse bug, "
                f"a huge value the 1.29 uninitialized read)")
    finally:
        result.close()


# --- monitors served through the mirror fold's reflected gather ---------------------
#
# MEEP accumulates DFT monitors THROUGH the symmetry: `add_dft` takes
# `loop_in_chunks`' use_symmetry=true default where `add_volume_source` passes
# false (dft.cpp:231 vs sources.cpp:487), so a region in the half the fold
# discards reports the same surface as the unfolded run. These cases pin the
# engine's reflected gather (`dft.folded_axis_sites`) against MEEP on the same
# stepped object: dft_fields arrays, flux planes with their signs, near2far
# packed data and far fields, at both mirror parities and both stored-count
# parities. The bounds are ~5-10x the measured values (dft 1.0e-07..1.3e-06,
# flux 4.0e-08..3.0e-06, packed 1.7e-07..2.1e-07, farfields 2.6e-08..7.7e-07);
# before the gather existed the same requests either raised or were refused at
# the gate, and a clamped-to-the-quadrant flux read 89% of the right answer.

_FOLD_DFT_BOUND = 1e-05
_FOLD_FLUX_BOUND = 3e-05


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["even", "odd_phase", "odd_count", "metallic"])
def test_fold_gather_dft_and_flux_match_meep(tmp_path, case):
    """Straddling and wholly-below regions match get_dft_array / get_fluxes."""
    report = _child(tmp_path, "fold_monitors", case)
    for label, error in report["dft_errors"].items():
        assert isinstance(error, float) and error < _FOLD_DFT_BOUND, (
            f"[{case}] dft {label}: {error} (bound {_FOLD_DFT_BOUND})")
    for label, row in report["flux"].items():
        assert row["error"] < _FOLD_FLUX_BOUND, (
            f"[{case}] flux {label}: ours {row['ours']:+.6e} meep {row['meep']:+.6e} "
            f"rel {row['error']:.3e}")
    below, above = report["flux"]["y_below"], report["flux"]["y_above"]
    # The discarded-half plane is the sign-flipped mirror of its image plane —
    # MEEP's phase_shift for the Poynting normal (vec.cpp:1371-1379). A gather
    # that lost the per-component parity would flip or double one of the two.
    assert abs(below["ours"] + above["ours"]) <= 1e-05 * abs(above["ours"]), (
        f"[{case}] folded flux pair is not sign-symmetric: "
        f"{below['ours']:+.6e} vs {above['ours']:+.6e}")


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["near2far_box", "near2far_two_mirrors", "near2far_odd_count"])
def test_fold_gather_near2far_matches_meep(tmp_path, case):
    """Discarded-half near surfaces: packed chunk data and the far fields it yields.

    `near2far_two_mirrors` reaches the sn = 3 double image (both planes odd), and
    `near2far_box` the four-face box whose corner chunks pin the stored-weight
    disambiguation alongside the reflected ones.
    """
    report = _child(tmp_path, "fold_monitors", case)
    assert isinstance(report["packed_error"], float) and report["packed_error"] < 5e-06, (
        f"[{case}] packed near2far data: {report['packed_error']} vs get_near2far_data")
    for index, error in enumerate(report["farfield_errors"]):
        assert error < 5e-06, f"[{case}] farfield point {index}: {error}"


@requires_meep
@skip_without_meep
def test_integrated_source_through_the_fold_matches_meep(tmp_path):
    """finite_grating.py's class: is_integrated x mirror fold steps at the fold floor.

    MEEP itself folds the combination exactly (folded vs unfolded MEEP measured
    1.7e-07..1.0e-06 with the source sheet crossing the plane and a PML), so the
    old blanket refusal in `VolumeSource.__post_init__` protected against
    nothing; the suspected repair-pass interaction writes ghost D =
    parity * (f - dipole), which IS MEEP's f_minus_p at the image row.
    """
    report = _child(tmp_path, "fold_monitors", "integrated_source_fold")
    for name, error in report["field_errors"].items():
        assert error < 1e-05, f"integrated-fold {name}: {error}"


@requires_meep
@skip_without_meep
def test_mie_shaped_flux_box_under_two_mirrors_matches_meep(tmp_path):
    """mie_scattering.py's shape at toy size: six planes, two in discarded halves."""
    report = _child(tmp_path, "fold_monitors", "mie_flux_box_3d")
    for label, row in report["flux"].items():
        assert row["error"] < _FOLD_FLUX_BOUND, (
            f"mie flux {label}: ours {row['ours']:+.6e} meep {row['meep']:+.6e} "
            f"rel {row['error']:.3e}")
    for pair in (("y1", "y2"), ("z1", "z2")):
        low, high = (report["flux"][name]["ours"] for name in pair)
        assert abs(low + high) <= 1e-05 * abs(high), (
            f"folded plane pair {pair} not sign-symmetric: {low:+.6e} vs {high:+.6e}")


@requires_meep
@skip_without_meep
def test_yee_grid_dft_fields_with_dz_across_the_fold_matches_meep(tmp_path):
    """absorbed_power_density.py's monitor: yee_grid=True [Dz, Ez] straddling the fold.

    `yee_grid=True` is use_centered_grid=false — per-component chunks on the
    component's OWN lattice — so the migration rebuilds it as raw-Yee
    accumulators (`DftFieldsYeeMigration`), not the centred monitor; and Dz is a
    stored array of this engine like any other of the twelve.
    """
    report = _child(tmp_path, "fold_monitors", "dz_yee_grid_fold")
    for component, error in report["dft_errors"].items():
        assert isinstance(error, float) and error < _FOLD_DFT_BOUND, (
            f"yee-grid {component}: {error}")


@requires_meep
@skip_without_meep
def test_flux_region_extent_on_an_invariant_axis_is_meeps_own_flat_plane(tmp_path):
    """coupler.py's 2a: a z extent on a 2-D region is MEEP's flat region, exactly.

    MEEP cannot even store the extent — a reduced volume has no z slot — and
    measured fluxes with and without it are identical at 0.0e+00. The migration
    flattens it (`_meep_region_extents`); the gate must say supported and
    the lift must serve the same numbers as the flat request.
    """
    report = _child(tmp_path, "fold_monitors", "invariant_extent_flux")
    assert report["supported"] is True, report["reasons"]
    assert report["flux"]["error"] < _FOLD_FLUX_BOUND, report["flux"]
    assert isinstance(report["dft_error"], float) and report["dft_error"] < _FOLD_DFT_BOUND


# --- result readers: get_dft_array and the flux packer ----------------------------

# The engine's established single-precision DFT floor on these cells, measured:
# get_dft_array 1.7e-07 .. 3.9e-07 over the centred / Yee / mirrored cases and
# 2.8e-07 .. 2.9e-06 on the 1-D ladder; the packed flux 5.9e-07 .. 9.6e-07.
_READER_DFT_BOUND = 1e-05
_READER_FLUX_BOUND = 1e-05


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["dft_array_centred", "dft_array_yee", "dft_array_mirrored"])
def test_get_dft_array_reduces_the_way_meep_reduces(tmp_path, case):
    """``result.get_dft_array`` is ``sim.get_dft_array``, shape included.

    Three requests per case — a box, a zero-extent line whose coordinate is OFF the
    lattice, and a point — so the read-time collapse is exercised on zero, one and
    two axes. The SHAPE is asserted separately because the uncollapsed region and
    MEEP's answer still BROADCAST against each other: a comparison that only
    subtracted them would produce a plausible wrong number rather than fail.

    The ``yee`` case is the one that separates two rules a single implementation
    would conflate. ``add_dft_fields(yee_grid=True)`` registers on the COMPONENT's own
    Yee lattice, so the interpolation fraction of a collapsed axis is measured against
    that lattice and not against the cell centres: the centred fraction gives 1.67e-01
    on the line and 2.04e-01 on the point, against 1.63e-07 and 2.27e-07 for the
    component's own.
    """
    report = _child(tmp_path, "readers", case)
    assert all(report["shapes_match"]), (report["shapes_match"], report["shapes"])
    for index, error in enumerate(report["errors"]):
        assert isinstance(error, float) and error < _READER_DFT_BOUND, (
            f"{case} region {index} (shape {report['shapes'][index]}): {error}")


@requires_meep
@skip_without_meep
def test_a_dft_region_reaching_a_terminated_far_face_keeps_its_yee_average(tmp_path):
    """The far plane a cell-centre average consumes, on the GATHERED path.

    ``_sliced_component`` appends the plane past a non-wrapping far face for every
    axis; ``_gathered_component`` used to append it only for a MIRROR-FOLDED one. Any
    monitor on the gathered path — which in a 1-D run is every monitor, because the
    two invariant axes are marked periodic — therefore lost the Yee-to-centre average
    for the WHOLE axis as soon as its region touched the high face, and
    ``_apply_yee_interpolation`` skips silently when the extended slice is not longer
    than the region.

    Measured against CPU MEEP's own ``get_dft_array`` on ``test_planewave_1D``'s cell
    (resolution 100, PML 1.0, Ex at f = 1): the whole-cell region read **3.166e-02**
    wrong and a high-face-touching one **3.149e-02**, both a uniform phase factor
    ``exp(-i*2*pi*f*dx/2)`` = 1.000965 - 0.031437j — half a cell of displacement —
    while the low-face and interior twins in the SAME run were already exact at
    2.8e-07 and 3.0e-07. Those two are kept here as the control: a fix that moved
    them would be a different defect, not this one.
    """
    report = _child(tmp_path, "readers", "dft_array_far_face_1d")
    assert all(report["shapes_match"]), report["shapes_match"]
    labels = ["whole cell", "high face", "low face", "interior", "whole cell (yee_grid)"]
    for label, error in zip(labels, report["errors"]):
        assert isinstance(error, float) and error < _READER_DFT_BOUND, f"{label}: {error}"


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["flux_packer", "flux_packer_mirrored",
                                  "flux_packer_reduced"])
def test_pack_flux_data_serves_meeps_own_eigenmode_reader(tmp_path, case):
    """``pack_flux_data`` + MEEP's unchanged ``get_eigenmode_coefficients``.

    The claim is the strongest form available: MEEP's own MPB evaluator, this
    engine's fields. It is checked four ways, because each alone can be satisfied by
    a mistake.

    * the packed E and H against MEEP's own ``get_flux_data`` on the same object,
      element for element in MEEP's own chunk order;
    * the coefficient MEEP's reader returns from the loaded data against the one it
      returns from its own run, and the flux it reports from the same data;
    * a CONTROL that returns the right shape, the right H and a plausible E —
      packing the same chunk list with this engine's own region weights instead of
      the weight read off MEEP's chunk;
    * a second control, the un-loaded simulation, whose ``alpha`` is 0 + 0j — so a
      passing number cannot be MEEP quietly answering itself.

    THE THREE CASES ARE THREE DIFFERENT CHUNK LISTS, not one setup run three times.

    * ``flux_packer`` — no symmetry: one list, ``sn = 0`` throughout.
    * ``flux_packer_mirrored`` — ``add_mode_monitor`` under ``Mirror(Y)``. It sets
      ``use_symmetry = false`` on the ``dft_flux``, but that flag never reaches the
      chunk loop (``add_dft`` calls ``loop_in_chunks`` without it, dft.cpp:231), so
      its chunks still carry ``sn = 1`` and must be flipped into user space. The full
      plane, 164 sites.
    * ``flux_packer_reduced`` — ``add_flux`` under the same mirror, which registers
      through ``symmetry::reduce`` (``use_symmetry = true``): 84 sites, a contiguous
      HALF of the plane, with ``stored_weight`` doubled. MEEP carries the halving
      itself — ``sqrt(S.multiplicity())`` in ``cscale`` (mpb.cpp:996) — and refuses
      the one case that is genuinely ambiguous, a folded monitor against an
      unpolarised mode (mpb.cpp:934-936, ``parity == 0``).

    MEASURED (pristine MEEP 1.33.0 single precision, env ``reference``, NumPy backend;
    both codes stepped 1600 steps to t = 40.0 in every case):

    ============================  ======  ========  ========  ========  ========
    case                          sites   E         H         alpha     flux
    ============================  ======  ========  ========  ========  ========
    ``flux_packer``               164     9.63e-07  5.91e-07  1.84e-07  9.88e-08
    ``flux_packer_mirrored``      164     9.42e-07  6.42e-07  2.56e-07  3.24e-08
    ``flux_packer_reduced``        84     9.41e-07  6.18e-07  2.59e-07  2.73e-08
    ============================  ======  ========  ========  ========  ========

    THE CONTROL IS WHAT DISTINGUISHES THE THIRD CASE, and its shape is the evidence
    for taking the measure off the chunk. On the two full-plane lists this engine's
    own region weights reproduce MEEP's stored E exactly — 0 sites differ by more
    than 1e-4 relative — because the two models bracket the SAME region. On the
    reduced list they score **3.203e-01**, wrong at exactly **4 of 84 sites**, by
    factors of 0.875 and 0.125: the halved region's new edge weights, which are that
    chunk's own ``s0.y`` and ``s1.y``. So the old coverage-total refusal's headline
    number (3.20e-01) was real but its stated cause — the full plane packed into a
    short list — was wrong; the mapping was right and four edge weights were not.
    """
    report = _child(tmp_path, "readers", case)
    assert report["engine_steps"] == report["meep_steps"], (
        f"the two codes stepped different spans — MEEP {report['meep_steps']} steps to "
        f"t={report['meep_time']}, this engine {report['engine_steps']} to "
        f"t={report['engine_time']}.")
    assert abs(report["meep_flux"]) > 1e-6, (
        f"MEEP's own flux through the plane is {report['meep_flux']}: nothing was "
        f"measured and a matching zero would prove nothing.")
    assert report["E_error"] < _READER_FLUX_BOUND, report
    assert report["H_error"] < _READER_FLUX_BOUND, report
    assert report["alpha_error"] < _READER_FLUX_BOUND, report
    assert report["flux_error"] < _READER_FLUX_BOUND, report
    reduced = case == "flux_packer_reduced"
    assert (report["E_size"] == 84) == reduced, (
        f"{case}: MEEP's chunk list carried {report['E_size']} sites; the reduced "
        f"add_flux list must be 84 and a full plane 164.")
    if reduced:
        assert report["control_engine_weight_error"] > 0.2, (
            "the chunk's own integration weight is not doing anything: packing "
            "S.reduce's half plane with this engine's region weights instead scored "
            f"{report['control_engine_weight_error']}, and it must be about 3.2e-01.")
        assert report["control_engine_weight_sites_off"] == 4, (
            "the control must be wrong at exactly the 4 sites of the chunk straddling "
            f"the mirror plane, not {report['control_engine_weight_sites_off']} — a "
            f"different count means a different defect.")
        assert all(abs(ratio - 0.875) < 1e-3 or abs(ratio - 0.125) < 1e-3
                   for ratio in report["control_engine_weight_ratios"]), (
            "the control's wrong sites must be off by the halved region's new edge "
            f"weights 0.875 and 0.125, got {report['control_engine_weight_ratios']}.")
    else:
        assert report["control_engine_weight_sites_off"] == 0, (
            f"{case}: this engine's own region weights must reproduce MEEP's stored E "
            f"on a FULL-plane list, and they differ at "
            f"{report['control_engine_weight_sites_off']} sites — the two models are "
            f"supposed to part company only once S.reduce has halved the region.")
    assert report["control_unloaded_alpha_max"] == 0.0, (
        "MEEP's reader returned a non-zero coefficient from a simulation nothing was "
        f"loaded into ({report['control_unloaded_alpha_max']}), so the driven number is "
        f"not evidence about this engine")


@requires_meep
@skip_without_meep
def test_a_yee_grid_mode_monitor_is_refused_by_name(tmp_path):
    """``add_mode_monitor(..., yee_grid=True)`` is a different monitor, not a re-layout.

    ``yee_grid=True`` is ``centered_grid = False`` (simulation.py:3550), and
    ``add_dft`` then brackets every chunk on ITS OWN component lattice rather than
    ``Centered`` (dft.cpp:231). This engine's flux plane samples cell centres for all
    six components, so it holds neither the sites nor the values that list wants.

    The refusal is checked against MEEP's OWN two answers, taken from ONE run
    carrying both monitors on the same region — because "these are different
    monitors" is a claim about MEEP, and if the two agreed the right fix would be to
    re-lay-out the centred plane rather than refuse. Measured (stock MEEP 1.33.0
    single precision, 1600 steps to t = 40.0):

    * flux ``1.9906e-04`` centred against ``2.1776e-04`` yee — **9.393 % apart**;
    * coefficient magnitudes 0.9925x forward and 1.0146x backward;
    * 164 E sites centred against 81 yee, because Ez alone lives on the doubled EVEN
      x lattice with a single x site while the centred plane brackets two.

    Without the named refusal the symptom is a lattice error one frame down —
    ``doubled coordinate -60 on axis 0 is not on this plane's centred lattice``,
    which is ``2 * resolution * (-1.0)``, the Ez Yee site — true, but describing the
    symptom rather than the cause.
    """
    report = _child(tmp_path, "readers", "flux_packer_yee_refusal")
    gap = abs(report["meep_yee_flux"] - report["meep_centred_flux"]) / abs(
        report["meep_centred_flux"])
    assert gap > 0.01, (
        f"MEEP's yee_grid and centred monitors agree to {gap:.3e} on this run, so the "
        f"refusal is not defending anything — re-layout would be the right answer.")
    assert report["meep_yee_E_sites"] < report["meep_centred_E_sites"], (
        f"the yee list holds {report['meep_yee_E_sites']} E sites and the centred one "
        f"{report['meep_centred_E_sites']}; the per-component lattice is what makes "
        f"them differ and this run did not exercise it.")
    assert report["yee_refused"] is True, report
    assert any("yee_grid" in reason for reason in report["yee_reasons"]), (
        report["yee_reasons"])
    assert report["centred_refused"] is False, (
        f"the default centred monitor on the SAME run must still pack: "
        f"{report['centred_reasons']}")


# --- force and energy monitors (MEEP dft_force / dft_energy) ----------------------

# test_force.py's own published constant, and MEEP 1.33.0 single precision's answer
# for the same setup on this machine. Both are recorded because they are different
# claims: the first is the oracle MEEP's developers wrote down, the second is what
# stock MEEP produces here today, and a parity number is only meaningful against the
# second.
_FORCE_PUBLISHED = -0.11039089113393187

# THE PARITY BOUNDS FOLLOW THE PRECISION OF THE MEEP COMPARED WITH. The engine steps
# single precision whatever MEEP was built with; against a double-precision MEEP
# (every conda-forge pymeep build) the comparison also measures float32 against
# float64 rounding, which the single-precision bounds were never set for. The child
# reports the precision it compared with (``meep_precision``).
#
# MEASURED against double-precision MEEP (one Apple silicon host, 2026-09-28; 6 runs
# per case: MEEP 1.33.0 double in 2 environments and MEEP 1.34.0 double in 1, 2 runs
# in each, with NumPy 2.2.6, 2.4.6 and 2.5.3; identical digits in 6 of 6):
#
# * force, decimation 1: 2.895e-07 relative; decimation 10: 8.545e-08.
# * energy (the mirror-plane test below): electric 1.233e-05, magnetic 1.074e-05,
#   total 1.153e-05.
#
# MEEP's own single- and double-precision builds (1.33.0) differ on the same setups by
# 2.756e-07 (force, decimation 1) and by 9.742e-06, 1.224e-05 and 1.100e-05 (energy),
# the size of the gaps above: they are rounding, not the reduction. At decimation 10
# the two builds differ by 1.789e-08 and the engine is nearer the double build
# (8.545e-08) than the single one (1.034e-07). Each double-precision bound is twice
# the worst measured value, rounded up to one significant figure: force 6e-07 and
# 2e-07 (bound over worst: 2.07 and 2.34), energy 3e-05 for all three (2.43, 2.79,
# 2.60). The single-precision bounds are unchanged, and test_force.py's published
# constant is held to 5e-08 on both.
_DOUBLE_PRECISION_BOUNDS = {
    "force_diagonal": 6e-7,
    "force_diagonal_decimated": 2e-7,
    "energy": 3e-5,
}


@requires_meep
@skip_without_meep
def test_a_diagonal_force_region_reproduces_meeps_own_stress_tensor(tmp_path):
    """MEEP's ``test_force.py`` setup, driven through the migration, at both decimations.

    The diagonal branch of ``add_dft_force`` (stress.cpp:178-185) registers six raw-Yee
    chunks with ``sqrt_dV_and_interp_weights = true`` and reduces them as
    ``sum_d Re(weight1_d) * |F_d|^2`` (stress.cpp:90-114). The ``sqrt`` is the subtlety
    this test exists for: MEEP folds ``sqrt(w)`` in per step so the weight lands on
    ``|F|^2`` exactly once, which is why the reduction goes through
    ``YeeRegionDFT.weighted_square`` and not through ``packed`` — the latter would
    apply the weight to the transform and square it.

    MEASURED (pristine MEEP 1.33.0, single precision, env ``reference``; the stopping
    condition is run on the control and its simulation time reused verbatim, so both
    codes step the same span — 4002 steps to t = 100.05):

    * decimation 1: MEEP ``-0.11039092155897505``, this engine
      ``-0.11039092309326226`` — 1.39e-08 relative, and 3.20e-08 from the published
      constant, inside ``assertAlmostEqual``'s own ``places=7`` bar of 5e-08.
    * decimation 10: MEEP ``-0.11039086353242147``, this engine
      ``-0.11039085212407206`` — 1.03e-07 relative. The two engine values differ by
      7.10e-08, inside test_force's ``places=6`` bar of 5e-07.

    Both of MEEP's own assertions are reproduced end to end by
    ``parity/meep_gpu/drive_meep_test_assertions.py``; this test pins the numbers so a
    regression is a failure here rather than a silent drift in that survey.
    """
    for case, single_bound in (("force_diagonal", 1e-7), ("force_diagonal_decimated", 5e-7)):
        report = _child(tmp_path, "readers", case)
        precision = report["meep_precision"]
        bound = {"single": single_bound,
                 "double": _DOUBLE_PRECISION_BOUNDS[case]}[precision]
        assert report["engine_steps"] == report["meep_steps"], (
            f"{case}: the two codes stepped different spans — MEEP {report['meep_steps']} "
            f"steps to t={report['meep_time']}, this engine {report['engine_steps']} to "
            f"t={report['engine_time']}. A time mismatch is not an accumulator defect and "
            f"must not be read as one.")
        assert abs(report["meep_force"]) > 0.1, (
            f"{case}: MEEP's own force is {report['meep_force']}, so nothing was measured "
            f"and a matching zero would prove nothing.")
        error = abs(report["engine_force"] - report["meep_force"]) / abs(report["meep_force"])
        assert error < bound, (
            f"{case}: force {report['engine_force']} against MEEP's "
            f"{report['meep_force']} ({precision} precision) is {error:.3e} relative, "
            f"over {bound:.0e}.")
        assert abs(report["engine_force"] - _FORCE_PUBLISHED) < 5e-8, (
            f"{case}: force {report['engine_force']} is "
            f"{abs(report['engine_force'] - _FORCE_PUBLISHED):.3e} from test_force.py's "
            f"published {_FORCE_PUBLISHED}, which fails its own assertAlmostEqual "
            f"(places=7, bar 5e-08).")
        # The store/load round trip, on THIS engine's accumulators. MEEP's own
        # get_force_data reads `force.diag`, which is empty on a lifted run, so without
        # a counterpart the round trip would save zeros, load zeros, and the force
        # would still come out right — a no-op dressed as a round trip.
        # Loosely, because the scaled transform is re-rounded to complex64 exactly as
        # MEEP's `scale_dft` re-rounds its own `complex<realnum>`: measured 4.0e-09
        # relative, which is float32's epsilon and not a scaling error.
        scaled_error = abs(report["scaled_force"] - 9.0 * report["engine_force"]) / abs(
            9.0 * report["engine_force"])
        assert scaled_error < 1e-7, (
            f"{case}: scale_dfts(3) must scale the force by 9 (the reduction is "
            f"quadratic), got {report['scaled_force']} from {report['engine_force']} "
            f"— {scaled_error:.3e} relative.")
        assert report["reloaded_force"] == report["engine_force"], (
            f"{case}: the saved transform did not come back — {report['reloaded_force']} "
            f"against {report['engine_force']}.")


@requires_meep
@skip_without_meep
def test_an_energy_region_straddling_a_mirror_plane_reproduces_meeps_own_densities(tmp_path):
    """MEEP's ``test_dft_energy.py`` setup: ``Mirror(Y)``, region across the fold.

    ``add_dft_energy`` registers four families per field direction — E and H with
    ``include_dV_and_interp_weights``, D and B without — and reduces them as
    ``0.5*Re(conj(E)*D)`` and ``0.5*Re(conj(H)*B)`` (dft.cpp:672-741). The weight
    riding only the E/H side is what this pins: weighting both would square the
    measure and give an energy that is neither MEEP's nor a constant factor from it.

    The FOLD is the second pin. Unlike ``add_dft_force``, ``add_dft_energy`` does not
    call ``symmetry::reduce`` — it copies the volume list (dft.cpp:723) — so
    ``loop_in_chunks`` visits both images and MEEP returns the full-domain energy.
    This engine's reflected gather is that visit, and the region here straddles the
    plane, so a stored-half integral would be visibly short rather than subtly off.

    MEASURED (pristine MEEP 1.33.0, single precision, env ``reference``; both codes
    stepped 17334 steps to t = 433.35, MEEP's own ``until_after_sources=100`` end):

    * electric  MEEP ``2000.8391586171483``  engine ``2000.83359375``   2.78e-06
    * magnetic  MEEP ``2010.1892565620592``  engine ``2010.192431640625`` 1.58e-06
    * total     MEEP ``4011.0284151792075``  engine ``4011.0260253906``   5.96e-07

    all inside the corpus parity band (3.12e-07..6.38e-06). The decimated pair lands
    ``|e - e_10| = 0.0227`` and ``|m - m_10| = 0.0205``, both inside
    test_dft_energy's own ``places=1`` bar of 0.05.

    That test's four assertions all sit after ``sim.get_eigenmode_coefficients``,
    which needs the flux transform in MEEP's layout. Its monitor is an ``add_flux``
    one on a mirrored cell, so the chunk list is ``symmetry::reduce``'s half plane;
    since ``pack_flux_data`` takes each chunk's own integration weight it serves that
    list, and all four are now DELIVERED by the assertion survey.
    """
    report = _child(tmp_path, "readers", "energy_folded")
    precision = report["meep_precision"]
    bound = {"single": 6.4e-6, "double": _DOUBLE_PRECISION_BOUNDS["energy"]}[precision]
    band = {"single": "the corpus parity band",
            "double": "the double-precision bound"}[precision]
    assert report["engine_steps"] == report["meep_steps"], (
        f"the two codes stepped different spans — MEEP {report['meep_steps']} steps to "
        f"t={report['meep_time']}, this engine {report['engine_steps']} to "
        f"t={report['engine_time']}.")
    assert report["meep_electric"] > 1.0, (
        f"MEEP's own electric energy is {report['meep_electric']}: nothing was measured.")
    for name in ("electric", "magnetic", "total"):
        theirs = report[f"meep_{name}"]
        mine = report[f"engine_{name}"]
        error = abs(mine - theirs) / abs(theirs)
        assert error < bound, (
            f"{name} energy {mine} against MEEP's {theirs} ({precision} precision) is "
            f"{error:.3e} relative, over {bound:.1e} — outside {band}.")
    assert abs((report["engine_electric"] + report["engine_magnetic"])
               - report["engine_total"]) < 1e-9, (
        "total() must be electric() + magnetic() exactly (dft.cpp:703-714), got "
        f"{report['engine_total']} against "
        f"{report['engine_electric'] + report['engine_magnetic']}.")
    for name in ("electric", "magnetic"):
        gap = abs(report[f"engine_{name}"] - report[f"engine_{name}_decimated"])
        assert gap < 0.05, (
            f"decimation 10 moved the {name} energy by {gap}, outside test_dft_energy's "
            f"own places=1 bar of 0.05.")


# The mirror-parity predicate, held to ONE verdict on both sides of the lift.
#
# `sources.mirror_parity_offenders` is the predicate; `gpu_compatibility` reports it
# before anything is built and `FdtdDriver.add_source` raises it mid-lift. The two used
# to be independent transcriptions of one rule and they drifted the moment either moved:
# the driver's half was narrowed from "on the plane AND the component is odd" to "...AND
# the profile cannot carry it", the gate's copy was not, and the gate then reported
# `supported=False` for declarations that lift and step correctly.
#
# Both drift directions are contract failures, and they fail differently:
#
#   gate LENIENT, driver strict  -> `supported=True` and then a bare ValueError from
#       inside the lift. This is the one that actually happened to the five
#       EigenModeSource rows: the gate read `source.component`, which an eigenmode
#       source leaves at ALL_COMPONENTS, so the rule never ran on the four sheets
#       mpb.cpp:874-900 expands it into.
#   gate STRICT, driver lenient  -> a run reported unsupported that this engine
#       reproduces at its fold floor. Silent, and it costs coverage rather than
#       correctness, which is why it went unnoticed.
#
# So agreement alone is not the assertion — two sides that both refuse everything agree
# perfectly. Each case carries its own expected verdict, and the accepted and refused
# sets are both non-empty and both measured.
_PARITY_TRUTH_TABLE = (
    # (label, phase, source builder, foldable?, why the verdict is that and not the other)
    ("uniform_Ey_sheet_crossing", +1, "uniform_odd", False,
     "a constant is EVEN about every plane, so it cannot carry the parity -1 that "
     "Mirror(Y,+1) gives Ey; stock MEEP 1.33.0 runs it and lands 0.569-2.829 relative "
     "from its own unfolded answer (results/source_parity_predicate_2026-08-08)"),
    ("Ey_point_on_the_plane", +1, "point_odd", False,
     "zero extent makes the parity condition constrain the source against itself, and "
     "Ey is unshifted on y under the EVEN plane so the site really is its own image: "
     "J = -J admits only J = 0. Measured 1.006 relative in stock MEEP"),
    ("Ey_sheet_crossing_with_amp_func", +1, "profiled_odd", True,
     "an extended profile CAN be odd about the plane; sin(pi x/4) measures 3.71e-07 "
     "folded against unfolded, five orders below the uniform sheet beside it"),
    ("Ez_uniform_sheet_crossing_even_component", +1, "uniform_even", True,
     "Mirror(Y,+1) leaves Ez even, so no parity is demanded of the profile at all — "
     "this is what keeps the rule from reading as a ban on sources at y = 0"),
    ("Ey_point_off_the_plane", +1, "point_off", True,
     "off the plane the image is a DISTINCT current, not a canceller; MEEP's folded "
     "run equals its unfolded pair at 2.9e-07 and this engine reproduces it at 9.3e-07"),
    ("eigenmode_sheet_crossing_the_plane", +1, "eigenmode", True,
     "THE FIVE ROWS. Every EigenModeSource MEEP itself writes drives two sheets the "
     "plane makes odd (the Love pair, mpb.cpp:886-900), so a per-component rule refuses "
     "all of them; the sheets carry MPB's mode profile and fold correctly, measured "
     "1.0e-06 against CPU MEEP on test_special_kz and test_mode_coeffs"),
)


def _parity_table_source(mp, kind):
    envelope = mp.GaussianSource(1.0, fwidth=0.5)
    if kind == "uniform_odd":
        return mp.Source(envelope, component=mp.Ey, center=mp.Vector3(0, 0), size=mp.Vector3(0, 4))
    if kind == "point_odd":
        return mp.Source(envelope, component=mp.Ey, center=mp.Vector3(0, 0))
    if kind == "profiled_odd":
        return mp.Source(envelope, component=mp.Ey, center=mp.Vector3(0, 0),
                         size=mp.Vector3(0, 4), amp_func=lambda p: complex(p.y))
    if kind == "uniform_even":
        return mp.Source(envelope, component=mp.Ez, center=mp.Vector3(0, 0), size=mp.Vector3(0, 4))
    if kind == "point_off":
        return mp.Source(envelope, component=mp.Ey, center=mp.Vector3(0, 1.1))
    if kind == "eigenmode":
        return mp.EigenModeSource(envelope, center=mp.Vector3(-1.5, 0), size=mp.Vector3(0, 5))
    raise AssertionError(kind)


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("label,phase,kind,foldable,why", _PARITY_TRUTH_TABLE,
                         ids=[row[0] for row in _PARITY_TRUTH_TABLE])
def test_the_mirror_parity_gate_and_the_driver_reach_the_same_verdict(label, phase, kind,
                                                                      foldable, why):
    """One predicate, two callers, six declarations spanning both of its answers.

    Run IN PROCESS rather than through this file's ``_CHILD_SCRIPT`` harness: that
    harness exists to step both codes on one object and compare fields, and nothing
    here steps anything. Each case builds a simulation, asks the gate, and lifts —
    ``lift_simulation`` calls ``sim.init_sim()`` and stops — so the cost is one
    rasterization of a 6x6 cell at resolution 10 per case.
    """
    import meep as mp

    from meep_gpu import gpu_compatibility, lift_simulation

    sim = mp.Simulation(
        cell_size=mp.Vector3(6, 6, 0), resolution=10,
        symmetries=[mp.Mirror(mp.Y, phase=phase)],
        sources=[_parity_table_source(mp, kind)],
        geometry=[mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), material=mp.Medium(epsilon=12))],
    )

    verdict = gpu_compatibility(sim)
    gate_refused = [reason for reason in verdict.reasons if "cannot carry the parity" in reason]

    driver_refused = None
    try:
        driver = lift_simulation(sim, prefer_gpu=False)
    except ValueError as exc:
        if "cannot carry the parity" not in str(exc):
            raise
        driver_refused = str(exc)
    else:
        driver.close()

    assert bool(gate_refused) == bool(driver_refused), (
        f"{label}: the gate and the driver disagree — gate_refused={bool(gate_refused)}, "
        f"driver_refused={bool(driver_refused)}. The pre-flight and the lift must apply "
        f"one predicate; this is the drift `sources.mirror_parity_offenders` exists to "
        f"make impossible."
    )
    assert bool(gate_refused) is (not foldable), (
        f"{label}: expected foldable={foldable} and both sides said "
        f"refused={bool(gate_refused)}. They agree, but on the wrong answer — {why}."
    )
    if not foldable:
        # The refusal is evidence a reader acts on, so it must carry MEEP's mechanism
        # and not the cancellation the old wording claimed on both sides of the lift.
        for message in (gate_refused[0], driver_refused):
            assert "use_symmetry=false" in message and "sources.cpp:487" in message, message
            assert "cancel" not in message.lower(), message


@requires_meep
@skip_without_meep
def test_the_parity_gate_expands_an_eigenmode_source_into_its_four_sheets():
    """The gate must see the sheets, not ``ALL_COMPONENTS`` — the five rows' actual bug.

    ``mp.EigenModeSource.component`` defaults to ``ALL_COMPONENTS``, which names no
    field component, so a gate that reads that attribute asks the parity rule about
    nothing and always answers "supported". The driver then expands the declaration
    into the four Love-equivalence sheets (mpb.cpp:874-900) and raises on the odd ones.

    Pinned as the EXPANSION rather than as the verdict, because the verdict alone is
    satisfied by the bug: reading ALL_COMPONENTS also produces "supported" here. What
    separates the two is whether the four sheets are the ones the rule was asked about,
    and — the part a wrong convention would get wrong — that the odd pair among them is
    non-empty. A plane normal to the mode's own normal ALWAYS makes two of the four odd,
    which is exactly why a per-component predicate refused every eigenmode source MEEP
    writes.
    """
    import meep as mp

    from .fields import mirror_parity
    from .from_meep import (
        _component_names,
        _declared_source_sheets,
        _effective_dimensions,
        _is_cylindrical,
    )

    source = mp.EigenModeSource(mp.GaussianSource(1.0, fwidth=0.5),
                                center=mp.Vector3(-1.5, 0), size=mp.Vector3(0, 5))
    sim = mp.Simulation(cell_size=mp.Vector3(6, 6, 0), resolution=10,
                        symmetries=[mp.Mirror(mp.Y)], sources=[source])
    dimensions = _effective_dimensions(mp, sim)
    names = _component_names(mp, _is_cylindrical(sim))
    sheets = _declared_source_sheets(mp, sim, source, dimensions, names)

    components = sorted(component for component, _profile in sheets)
    assert components == ["Ey", "Ez", "Hy", "Hz"], (
        f"a mode plane normal to x installs K = nHat x H and N = -nHat x E — the four "
        f"sheets Ez, Ey, Hz, Hy (mpb.cpp:886-900); got {components}"
    )
    assert all(has_profile for _component, has_profile in sheets), (
        "every eigenmode sheet carries MPB's mode field as its profile, which is what "
        "lets the parity rule accept it without the gate ever running MPB"
    )
    odd = [component for component, _profile in sheets if mirror_parity(component, 1, +1) < 0]
    assert odd, (
        "the Y plane must make some of the four sheets odd — if it made none, the five "
        "refused rows would never have been refused and this test would be vacuous"
    )
    assert sorted(odd) == ["Ey", "Hz"], (
        f"under Mirror(Y,+1) the odd pair is Ey/Hz (and under phase=-1 it is Ez/Hy), so "
        f"a per-component rule refuses every eigenmode source whose plane crosses it; "
        f"got {odd}"
    )


# --- the migrated-monitor map's lifetime contract -------------------------------------
#
# `GpuRunResult` resolves a MEEP monitor to this engine's counterpart. That lookup was
# once `{id(meep monitor): ours}` holding no reference to the MEEP objects, and an `id`
# is an ADDRESS: CPython frees an object and hands the same address to the next
# allocation of the same size. So a first run's map could answer for a SECOND run's
# monitor, and answer with a real spectrum from the wrong run.
#
# It did. MEEP's `test_binary_grating_oblique_0_0_0` — a two-run normalization, driven
# through `parity/meep_gpu/drive_meep_test_assertions.py`, whose module-level
# `mp.get_fluxes` override asks EACH driven result in turn and takes the first that does
# not raise — returned the first run's transmission spectrum bit for bit for the second
# run's monitor, so the normalized transmittance came out exactly 1.0. Nine reruns in
# isolation fired twice (`parity/meep_gpu/results/bg_oblique_rep1`..`rep8`).
#
# The tests below pin the LIFETIME, because that is the property that was missing.
# Nothing about "the map can be read" would have caught this: the map read perfectly,
# and returned the wrong run's answer.


class _StandInMonitor:
    """A monitor-shaped object with no dependencies, for the allocator probe.

    Address reuse is CPython's allocator, not MEEP's: the probe needs an object whose
    instances are all the same size, which is every plain Python class. The real
    ``mp.DftFlux`` is exercised too, in the parametrized case below, because that is the
    class the defect actually bit.
    """

    # `__weakref__` because the mutation battery's `migrated_monitor_held_weakly` swaps
    # the container's strong reference for a `weakref.ref`, and a stand-in that could not
    # be weakly referenced would make that mutation crash instead of reproducing the
    # defect it exists to reproduce.
    __slots__ = ("tag", "__weakref__")

    def __init__(self, tag):
        self.tag = tag


def _probe_address_reuse(make, keep=None, attempts=256):
    """Free one object, then allocate until something lands on its address.

    Returns ``(address, reused_or_None, still_held)``. ``keep`` is handed the first
    object before every reference to it is dropped — that is where the map under test
    records it, and whether the map OWNS it is exactly what this probe measures.

    Each attempt is retained rather than dropped, so the loop keeps asking for FRESH
    memory instead of cycling one freed block; CPython's pool is LIFO, so an unowned
    address is normally handed back on the first attempt (measured 49 times in 50, for
    both a plain class and ``mp.DftFlux``), and the loop is here so that "normally" is
    not what the test rests on.
    """
    first = make()
    address = id(first)
    if keep is not None:
        keep(first)
    del first
    gc.collect()
    still_held = []
    for _ in range(attempts):
        candidate = make()
        if id(candidate) == address:
            return address, candidate, still_held
        still_held.append(candidate)
    return address, None, still_held


@pytest.mark.parametrize("kind", ["stand_in", "meep_dft_flux"])
def test_migrated_monitors_pins_the_address_of_the_meep_monitors_it_holds(kind):
    """The container OWNS its MEEP monitors, so their ids can never be handed out again.

    THE DETERMINISTIC HALF of the two-run defect. The hazard needs an address to be
    recycled; the fix makes recycling impossible rather than making a recycled address
    survivable, so what is asserted here is that the address is never handed out again —
    over 256 fresh allocations of the same class, immediately after every external
    reference was dropped and a collection forced.

    That is a real assertion and not a tautology dressed as one: the companion test
    below runs the SAME probe against the pre-fix shape (an ``id``-keyed dict with no
    reference) and gets the address back, usually on the first attempt.
    """
    if kind == "meep_dft_flux":
        mp = pytest.importorskip("meep", reason="CPU MEEP is not installed")
        # A DftObj is a plain Python holder — `func` and `args` are stored and only
        # called if a C++ attribute is reached — so this is the real class the defect
        # bit, with no simulation, no fields and no C++ allocation behind it.
        def make():
            return mp.DftFlux(lambda *args: None, [[1.0], []])
    else:
        def make():
            return _StandInMonitor("first")

    monitors = MigratedMonitors()
    address, reused, _held = _probe_address_reuse(
        make, keep=lambda monitor: monitors.add(monitor, "run one's monitor"))

    assert reused is None, (
        f"a MEEP monitor recorded in a MigratedMonitors was collected anyway and address "
        f"{address:#x} was handed to a later allocation. The container must hold a STRONG "
        f"reference to every monitor it was built from — that is the whole of the fix, and "
        f"without it a later run's monitor resolves onto this run's entry and reads back "
        f"the wrong spectrum without raising."
    )
    assert len(monitors) == 1 and monitors.pairs()[0][1] == "run one's monitor", (
        "the entry must still be there and still be reachable; a container that pinned "
        "the address by forgetting the pair would pass the assertion above and be useless"
    )
    original = monitors.pairs()[0][0]
    assert id(original) == address and monitors[address] == "run one's monitor", (
        "the mapping key is computed from the live object the container owns, so it must "
        "still name the address that was recorded before the drop"
    )


def test_a_bare_id_map_hands_back_the_first_runs_monitor_once_the_address_is_recycled():
    """The CONTROL: the pre-fix shape, reproduced in-process, returning the wrong monitor.

    Without this the test above would only say "an object that is referenced stays
    alive". This says what the reference is FOR: run the identical probe against
    ``{id(monitor): ours}`` — the map as it was — and a freshly allocated monitor, which
    belongs to no run at all, resolves to the first run's entry. No exception, no empty
    result: a real monitor, for the wrong run.

    Skipped rather than failed if the allocator declines to reuse the block, because the
    reuse is CPython's behaviour and not this package's contract. It has not declined in
    any run of this test; the skip exists so a future allocator change is reported as
    "the control could not be established" instead of as a defect.
    """
    bare: dict[int, str] = {}
    address, reused, _held = _probe_address_reuse(
        lambda: _StandInMonitor("a monitor"),
        keep=lambda monitor: bare.__setitem__(id(monitor), "run one's monitor"))
    if reused is None:
        pytest.skip("this interpreter did not recycle the freed address in 256 attempts, "
                    "so the pre-fix hazard could not be reproduced here")

    assert bare.get(id(reused)) == "run one's monitor", (
        f"the pre-fix map was expected to mis-resolve here: address {address:#x} was freed "
        f"and handed to a brand-new monitor, and the stale key should have matched it"
    )
    # And the shipped container, given the same recycled object, refuses it.
    owned = MigratedMonitors()
    owned.add(_StandInMonitor("run one"), "run one's monitor")
    assert owned.resolve(reused) is None, (
        "MigratedMonitors resolves by identity, so an object it was never given must not "
        "resolve at all — whatever address it happens to occupy"
    )


def test_a_second_runs_monitor_never_resolves_against_the_first_runs_result():
    """The reader's own fall-through, which is where the wrong spectrum was returned.

    ``drive_meep_test_assertions.make_module_reader`` overrides ``mp.get_fluxes`` and
    asks EVERY driven result in turn, taking the first that does not raise ``KeyError``.
    The first run is asked first. So the contract is not merely "the second result knows
    its own monitor" but "the FIRST result disowns the second run's monitor" — and it
    must disown it by raising, because a returned ``None`` would be read as a spectrum.
    """
    mp = pytest.importorskip("meep", reason="CPU MEEP is not installed")

    class _Spectrum:
        def __init__(self, values):
            self.values = values

        def get_flux_spectrum(self):
            return self.values

    class _Driver:  # Only `migrated_monitors` is read; nothing here steps anything.
        def __init__(self, monitors):
            self.migrated_monitors = monitors

    def flux():
        return mp.DftFlux(lambda *args: None, [[1.0], []])

    first_flux, second_flux = flux(), flux()
    first_monitors, second_monitors = MigratedMonitors(), MigratedMonitors()
    first_monitors.add(first_flux, _Spectrum([1.0, 1.0, 1.0]))
    second_monitors.add(second_flux, _Spectrum([0.25, 0.5, 0.75]))
    first = GpuRunResult(driver=_Driver(first_monitors), steps=1, meep_time=1.0, wall_time_s=0.0)
    second = GpuRunResult(driver=_Driver(second_monitors), steps=1, meep_time=1.0, wall_time_s=0.0)

    with pytest.raises(KeyError):
        first.monitor_for(second_flux)
    with pytest.raises(KeyError):
        second.monitor_for(first_flux)

    # The reader's loop, spelled the way the harness spells it.
    answered = None
    for result in (first, second):
        try:
            answered = result.get_flux_spectrum(second_flux)
        except KeyError:
            continue
        break
    assert answered == [0.25, 0.5, 0.75], (
        f"the second run's monitor must be answered by the second run. Getting "
        f"{answered} — the first run's spectrum — is the observed defect: on "
        f"test_binary_grating_oblique_0_0_0 it made the normalized transmittance "
        f"exactly 1.0, which is a plausible number nothing downstream questions."
    )


def test_two_live_monitors_of_one_run_never_resolve_to_each_other():
    """Distinct monitors stay distinct, and an unknown one is a refusal, not a guess."""
    reflected, transmitted = _StandInMonitor("refl"), _StandInMonitor("tran")
    monitors = MigratedMonitors()
    monitors.add(reflected, "ours: reflected")
    monitors.add(transmitted, "ours: transmitted")

    assert monitors.resolve(reflected) == "ours: reflected"
    assert monitors.resolve(transmitted) == "ours: transmitted"
    assert monitors.resolve(_StandInMonitor("refl")) is None, (
        "resolution is by IDENTITY; an equal-looking monitor from somewhere else is not "
        "this run's monitor"
    )
    assert dict(monitors) == {id(reflected): "ours: reflected",
                              id(transmitted): "ours: transmitted"}, (
        "the mapping protocol stays available for callers that spell the lookup as "
        "`map[id(monitor)]`, and it is safe because the container owns the objects"
    )
