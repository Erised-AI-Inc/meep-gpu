"""
Lift a constructed ``mp.Simulation`` onto this package's GPU time stepper.

This is the entry point the package exists for. A MEEP user writes the simulation
they already write — cell, geometry, sources, boundary layers, symmetries — and
``run_on_gpu(sim, until=...)`` steps it here instead of in MEEP's inner loop.
Everything MEEP does well stays MEEP's: it rasterizes the geometry, applies its
subpixel smoothing, resolves ``mp.Medium`` and the material library, and reports
the permittivity volume this module reads back through ``sim.get_epsilon()``.

**The one property that matters more than any feature: a simulation this module
cannot reproduce exactly must RAISE, never quietly run something else.** An FDTD
run that was set up wrong does not crash — it returns a smooth, finite, entirely
plausible field for a system nobody asked for, and no downstream spectrum says
so. Every refusal below was measured rather than assumed, and each message carries
the number:

* ``k_point`` left unset is **no longer a refusal** — it is MEEP's default
  *perfect electric conductor* box, and it is lifted as
  ``boundaries="metallic"`` (:func:`_lift_boundaries`). It used to be the second
  most common refusal in MEEP's own example corpus, 37 of the 57 surveyed scripts,
  and the **1.28e+00** complex relative L2 it quoted is what the boundary is worth:
  the two conditions are different simulations, not different accuracies, and MEEP
  returns a different array shape for each (N vs N+1 per axis, both handled).
  ``k_point`` is all-or-nothing in MEEP's declared arguments, so a lifted run is
  metallic on every axis or periodic on every axis; a per-axis mix is expressible
  on :class:`~.driver.FdtdDriver` directly but has nothing readable to be lifted
  from.
* A permittivity read from the wrong array. ``sim.get_epsilon()`` is a
  *diagnostic*: it is the average eigenvalue of the permittivity tensor,
  bilinearly interpolated from the nearest Yee points onto cell centres, so at an
  interface it reports a blend of three components registered half a cell apart
  (an eps 1 / eps 4 step reads 1.333 there, which is ``3/(1/1 + 1/1 + 1/4)`` and
  belongs to no component). Reinstalling that array reproduces CPU MEEP to
  **1.16e-01 … 3.43e-01** depending on how it is registered — every registration
  wrong by five orders of magnitude, and every one of them a smooth plausible
  field. A structured cell is therefore lifted from
  ``fields.get_chi1inv(component, direction, iloc, 0)``, which reports the stored
  per-component tensor entry itself, at each component's own Yee point: measured
  **1.5e-07 … 3.2e-07** against CPU MEEP over blocks, slabs, multi-material cells
  and both cell-count parities (:func:`_lift_epsilon_structured`).
* A permittivity tensor that is not diagonal. MEEP's subpixel averaging is
  anisotropic, so a surface whose normal is not a coordinate axis puts real weight
  in the off-diagonal ``chi1inv`` entries this engine has nowhere to store —
  measured **5.7e-02** (cylinder) and **2.5e-02** (rotated block) if only the
  diagonal is kept. Every off-diagonal entry is read at every point and a nonzero
  one refuses the lift. ``eps_averaging=False`` makes MEEP point-sample the same
  geometry, which lifts at **1.84e-07**.
* A 2-D or 1-D cell is **no longer a refusal** — it is lifted, with the missing
  axes carried as translational invariance rather than as a thin slab. MEEP's
  2-D is an infinite structure with every d/dz identically zero, and MEEP reaches
  it by leaving z out of ``LOOP_OVER_DIRECTIONS`` (``stride(Z) = 0``); this
  engine reaches it as one periodic cell on that axis, whose difference is the
  same sample twice. The two are not merely close: **MEEP's own 2-D run and the
  same problem as a 3-D cell one pixel deep in z agree to 0.0e+00**, and this
  engine reproduces both polarizations of a 2-D run against CPU MEEP at
  **1.4e-07 … 7.7e-07** — periodic, metallic, absorbing, dispersive, folded, and
  at odd cell counts. It was the largest single blocker in MEEP's example
  corpus, 47 of the 57 surveyed scripts. A **cylindrical** cell is lifted as
  well, as MEEP's own Dcyl discretization (r-phi-z curls, the 1/r metric and the
  m-dependent axis boundary) under the x -> r, y -> phi, z -> z mapping.
* An ``mp.Simulation`` that has already been initialized. ``sim.set_boundary(side,
  direction, cond)`` and ``sim.fields.use_bloch(direction, k)`` rewrite
  ``fields.boundaries`` per face, and MEEP exposes it only as an opaque pointer — so
  a lift cannot tell which condition each of the six faces ended up with, and would
  confidently step whichever one ``sim.k_point`` implies. Now that BOTH conditions
  are supported the ambiguity is worse, not better: the **1.28e+00** gap is still
  the cost of guessing. Found in MEEP's own example corpus
  (``antenna_pec_ground_plane_1D.py``), which is why it is checked.

A medium that is uniform, or structured in ``epsilon_diag`` alone — isotropic or
diagonally anisotropic, with any combination of Lorentz/Drude susceptibilities,
``D_conductivity``, ``chi2`` and ``chi3`` — is reproduced at the engine's own
established floors, which is what the cross-validation suite in
``test_from_meep.py`` pins.

A cell whose media differ in their ``E_susceptibilities`` as well needs the
per-point sigma volumes, and there are three ways to get them
(:func:`_lift_susceptibilities_structured` dispatches, and records which one ran
on ``driver.sigma_lift`` / :attr:`GpuRunResult.sigma_lift`).

* On a MEEP carrying ``fields.get_susceptibility_sigma``
  (``parity/meep_gpu/meep-sigma-reader.patch``) they are READ exactly, one number
  per Yee point. Preferred wherever it exists.
* On any stock MEEP the default is a declared-geometry LOOKUP
  (:func:`_lookup_sigma_volumes`, :mod:`~.sigma_lookup`). MEEP point-samples sigma
  from the geometry tree and never subpixel-averages it
  (``Subpixel_Smoothing.md:153``; measured, at a block edge placed off-grid, sigma
  reads binary ``{0, 0.400000}`` where the permittivity at the same points reads
  4.0000 / 3.2800 / 1.0000), so the sigma at a Yee point is the declared sigma of
  the medium containing that point. Every point's value is then CHECKED back
  against MEEP: the ``eps_inf`` read from MEEP, the looked-up sigma, and MEEP's own
  dispersion formula must reproduce ``get_chi1inv(c, d, iloc, w)`` at ``N + 1``
  frequencies. That check is what makes
  it a read rather than a reimplementation — it catches the coordinate the
  containment test is evaluated at, object precedence, ``default_material``, and
  the one-ulp disagreement between MEEP's exported ``is_point_in_object`` and the
  predicate its own tree search uses, without any of them being separately
  reasoned about. Measured against the exact reader over every Yee point of six
  cells: **1.49e-08 … 3.28e-08** of each term's own largest sigma.
* Where the lookup declines or fails its check, the sigmas are RECOVERED from the
  public ``get_chi1inv(c, d, iloc, frequency)`` alone, by sampling the same points
  at several frequencies and solving MEEP's
  ``eps(w) = eps_inf + SUM_n sigma_n * chi_n(w)`` (:mod:`~.sigma_recovery`, which
  refuses rather than returning a number it cannot cross-validate). It uses no
  geometry at all, so every dependency the lookup's check exists to police is
  absent from it. Agrees with the reader to **5.7e-08** on a Lorentz+Drude
  two-block cell and **1.4e-06** on a 6-pole ``meep.materials.Au`` cell
  (``test_sigma_recovery.py``), and costs ``2*(N+1) + 3`` chi1inv calls per point
  per component against the reader's one.

Registration, the historical failure mode, is pinned rather than assumed. On a
PERIODIC axis ``sim.get_array``'s grid point ``j`` sits at
``grid.axis_origin(d) + (j - 0.5)*dx``, for both cell-count parities (verified
against ``sim.get_array_metadata()``), so driver cell ``i`` is MEEP index ``i + 1``
and MEEP index 0 is the periodic duplicate. On a METALLIC axis there is no
duplicate to place: MEEP's point ``j`` sits at ``grid.axis_origin(d) + (j + 0.5)*dx``
and driver cell ``i`` is MEEP index ``i``. :meth:`GpuRunResult.get_array` returns
the whole thing in MEEP's own convention through
``Fields.to_meep_array``, so a lifted result is compared against
``sim.get_array(component=...)`` with no slicing at all.

MEEP is imported lazily, inside the functions. ``import meep_gpu`` keeps working
on a machine that has never had MEEP installed, which is what
``test_package_boundary.py`` and the rest of the package assume.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import functools
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy

from .backends import to_numpy
from .dft import (
    FluxDftData,
    FluxMonitor,
    ForceDftData,
    MeepChunkWeight,
    _axis_cell_count,
    _boundary_weight_ladder,
    _load_flux_planes,
    folded_axis_sites,
)
from .dispersion import (
    DRUDE,
    E_COMPONENTS,
    LORENTZIAN,
    Susceptibility,
    component_coordinates,
)
from .absorber import AbsorberLayer
from .amp_interpolation import amp_func_from_array
from .driver import FdtdDriver, PEC_EPSILON_SENTINEL, is_pec_chi1inv, is_pec_epsilon
from .gaussian_beam import beam_fields
from .grid import INVARIANT_AXES_BY_DIMENSIONS, Mirror, meep_cell_count
from .pml import FaceThickness, half_cell_extent
from .sigma_lookup import (
    VERIFICATION_TOLERANCE,
    SigmaLookupRefused,
    plan_verification,
    sigma_of_medium,
)
from .sigma_recovery import SigmaRecoveryRefused, SusceptibilityPole, plan_recovery

AXIS_NAMES = ("x", "y", "z")  # Axis order of every per-axis table below.
# MEEP's boundary_side enum (meep.hpp): High = 0, Low = 1. The order is NOT the
# obvious one, and reading it the obvious way puts a one-sided absorber on the
# wrong face of an otherwise correct run.
MEEP_SIDE_HIGH = 0
MEEP_SIDE_LOW = 1
MEEP_ALL = -1  # mp.ALL / mp.ALL_DIRECTIONS: "every direction" / "every side".
MEEP_CYLINDRICAL = -2  # mp.CYLINDRICAL — a token, not a dimension count.
# MEEP's `direction` enum, vec.hpp:79: `X = 0, Y, Z, R, P, NO_DIRECTION`. R and P are
# NOT axis indices — a Dcyl run's R is this engine's axis 0 and its Z is axis 2, and P
# names the invariant phi axis, which no surface has as a normal.
MEEP_X = 0
MEEP_Y = 1
MEEP_Z = 2
MEEP_R = 3  # mp.R, the radial direction of a Dcyl run (mp.P = 4 is never a PML direction).
MEEP_P = 4
# mp.AUTOMATIC = -1 (meep/__init__.py), and it is NOT a member of that enum: it is the
# sentinel `EigenModeSource.add_source` tests with `if self.direction < 0` before asking
# `fields::normal_direction` to infer the direction from the source volume's shape.
MEEP_AUTOMATIC = -1
_PML_DEFAULT_R_ASYMPTOTIC = 1e-15  # MEEP's mp.PML default, and the only value pml.PML builds.
_PML_DEFAULT_MEAN_STRETCH = 1.0  # kappa = 1: the driver's UPML has no stretch term.
# Probe points for MEEP's default quadratic PML profile (mp.PML: lambda u: u*u).
# A profile is accepted only if it reproduces u**2 at every one of these.
_PML_PROFILE_PROBE = (0.0, 0.125, 1.0 / 3.0, 0.5, 0.75, 0.9375, 1.0)
_PML_PROFILE_TOLERANCE = 1e-12
# Largest relative spread across sim.get_epsilon() still called homogeneous. The
# array is float32-backed, so exact equality is not available even for a single
# medium filling the cell.
_EPSILON_UNIFORM_TOLERANCE = 1e-6
_COURANT_LIMIT_3D = 1.0 / math.sqrt(3.0)  # The 3-D CFL bound FdtdDriver enforces.
# Largest off-diagonal chi1inv still called diagonal. MEEP leaves the array
# unallocated and returns exactly 0 unless its anisotropic averaging wrote to it,
# and the smallest value a real slanted interface produced in testing was 0.086
# (a rotated block), so any threshold in this range separates the two populations.
# chi1inv is float32-backed, hence a bound above its rounding rather than at zero.
_CHI1INV_OFFDIAG_TOLERANCE = 1e-6
# How far a sample coordinate may sit from MEEP's integer lattice (units of dx/2).
_LATTICE_TOLERANCE = 1e-7


class MeepSimulationNotLiftable(ValueError):
    """An ``mp.Simulation`` this module will not run, and the reasons why.

    A ``ValueError`` so that a caller who wraps :func:`lift_simulation` in the
    ordinary way still catches it, and its own class so that a caller who wants to
    distinguish "MEEP built something we cannot reproduce" from "this argument is
    malformed" can. ``reasons`` is the same tuple :func:`gpu_compatibility`
    returns, so the pre-flight check and the failure report never disagree.
    """

    def __init__(self, reasons: Sequence[str]):
        self.reasons = tuple(reasons)
        listed = "\n".join(f"  - {reason}" for reason in self.reasons)
        super().__init__(
            f"This mp.Simulation cannot be lifted onto the GPU stepper "
            f"({len(self.reasons)} reason(s)):\n{listed}\n"
            f"Nothing was run. Call meep_gpu.gpu_compatibility(sim) to get this list "
            f"without building anything, or run the simulation on CPU MEEP."
        )


@dataclass(frozen=True)
class GpuCompatibility:
    """What :func:`gpu_compatibility` reports: a verdict plus every blocking reason.

    Deliberately a list of reasons rather than a boolean, following
    ``GpuFdtdBackend.compatibility`` in the host application: a caller deciding
    between backends needs to know *what* is unsupported so it can say so or work
    around it, and a single ``False`` turns every distinct refusal into the same
    dead end. ``supported`` is exactly ``not reasons``.
    """

    supported: bool
    reasons: tuple[str, ...]

    def __bool__(self) -> bool:  # `if gpu_compatibility(sim):` reads as the verdict.
        return self.supported

    def __str__(self) -> str:
        if self.supported:
            return "GpuCompatibility(supported): every feature of this simulation can be lifted."
        listed = "\n".join(f"  - {reason}" for reason in self.reasons)
        return f"GpuCompatibility(unsupported), {len(self.reasons)} reason(s):\n{listed}"


def collapse_dft_region(region, first_site_doubled, resolution, center, size, invariant=(False,) * 3):
    """MEEP's read-time reduction of a DFT region — the whole of what ``get_dft_array`` adds.

    ``sim.add_dft_fields`` stores its chunks with ``include_dV_and_interp_weights =
    false`` (dft.cpp:904-917), so the accumulated numbers carry no measure at all;
    everything below happens at READ time, in two steps that this one function does
    together because MEEP does them together:

    1. ``dft_chunk::process_dft_component`` (dft.cpp:996-1006, 1013) multiplies each
       sample by ``interp_w = IVEC_LOOP_WEIGHT(s0i, s1i, e0i, e1i, 1.0)``, a copy of
       the boundary weights with EVERY NON-EMPTY DIRECTION SET TO 1 (dft.cpp:975-985).
       Only an axis whose requested extent is zero keeps a weight.
    2. ``collapse_array`` (array_slice.cpp:553-594) sums each such axis away, and
       ``get_dft_component_dims`` never gave a rank slot to an axis that came out one
       cell wide (dft.cpp:198-203), so those are dropped too.

    The weights are MEEP's zero-extent branch of ``compute_boundary_weights``
    (loop_in_chunks.cpp:274-287): with ``is``/``ie`` the doubled lattice coordinates of
    the first and last sampled site,

        ``w0 = 1 - c*a + is/2``   applied to the first site (``s0``),
        ``w1 = 1 + c*a - ie/2``   applied to the last  (``s1``, i.e. ``i == 1``
                                  of a two-site ladder — vec.hpp:372-380),

    which for the two cells bracketing an off-lattice coordinate are ``1 - f`` and
    ``f``: plain linear interpolation onto the requested plane. THE LATTICE IS THE
    COMPONENT'S OWN. ``loop_in_chunks`` shifts the bracket by
    ``iyee_shift(Centered) - iyee_shift(cgrid)`` before flooring (loop_in_chunks.cpp:350-357),
    and ``add_dft`` passes ``use_centered_grid ? Centered : c`` (dft.cpp:231), so a
    ``yee_grid=True`` monitor brackets on the component's Yee sites. Deriving the
    fraction from the cell centres instead is right for a centred monitor and wrong for
    a Yee one — measured 1.67e-01 and 2.04e-01 relative against CPU MEEP on a
    zero-thickness line and a point monitor of Ez, against 1.63e-07 and 2.27e-07 with
    the component's own lattice. ``first_site_doubled`` is therefore supplied by the
    caller from the accumulator that holds the samples, not recomputed here.

    An INVARIANT axis of a reduced-dimension run is not a direction MEEP has at all
    (``LOOP_OVER_DIRECTIONS(gv.dim, d)``), so it carries no weight and is simply
    dropped: its single plane is bookkeeping, not a coordinate.

    Args:
        region: ``(nx, ny, nz)`` block as accumulated, already unfolded if the run
            has mirror symmetry.
        first_site_doubled: Per-axis doubled lattice coordinate of index 0 of
            ``region`` — the sample's own, on its own Yee lattice.
        resolution: The grid's ``a``.
        center: The requested region's centre, MEEP's ``where`` centre.
        size: The requested region's extents, as MEEP resolved them
            (:func:`_meep_region_extents`); an axis is collapsed exactly when this
            is zero.
        invariant: Per-axis flag for the axes this run does not resolve.

    Returns:
        The array ``sim.get_dft_array`` would return, in ``numpy.complex128``.
    """
    reduced = numpy.asarray(region, dtype=numpy.complex128)
    if reduced.ndim != 3:
        raise ValueError(
            f"collapse_dft_region takes a (nx, ny, nz) block; got shape {reduced.shape}.")
    a = float(resolution)
    for axis in (2, 1, 0):
        count = int(reduced.shape[axis])
        if invariant[axis]:
            reduced = numpy.take(reduced, 0, axis=axis)
            continue
        if float(size[axis]) != 0.0:
            if count == 1:
                reduced = numpy.take(reduced, 0, axis=axis)
            continue
        if count > 2:
            raise ValueError(
                f"axis {axis} has a zero requested extent but {count} stored samples; MEEP's "
                f"vec2diel_floor/ceil bracket of one coordinate spans one or two sites, so this "
                f"region did not come from the request it is being collapsed against.")
        low = float(first_site_doubled[axis])
        w0 = 1.0 - float(center[axis]) * a + 0.5 * low
        if count == 1:
            weights = numpy.asarray([w0], dtype=numpy.complex128)
        else:
            w1 = 1.0 + float(center[axis]) * a - 0.5 * (low + 2.0)
            weights = numpy.asarray([w0, w1], dtype=numpy.complex128)
        reduced = numpy.tensordot(reduced, weights, axes=([axis], [0]))
    return reduced


class MigratedMonitors(Mapping):
    """Every MEEP monitor a lift rebuilt, resolvable from the object a script holds.

    A read-only mapping ``{id(meep monitor): this engine's monitor}`` — and, the part
    that is load-bearing rather than convenient, the OWNER of a strong reference to
    every MEEP monitor it was built from.

    **THE STRONG REFERENCE IS THE CONTRACT.** An ``id`` is an address, not an identity:
    CPython frees an object and hands the same address to the next allocation of the
    same size. A map that stored only the integer therefore stops describing anything
    the moment its monitors are collected — and the second run of the TWO-RUN
    NORMALIZATION IDIOM is exactly where that happens, because MEEP's own spelling
    drops the first simulation (and with it the first run's monitors) before building
    the second. Measured: MEEP's ``test_binary_grating_oblique_0_0_0`` driven through
    ``parity/meep_gpu/drive_meep_test_assertions.py`` returned the FIRST run's
    transmission spectrum, bit for bit, for the SECOND run's monitor, so the normalized
    transmittance came out exactly 1.0; re-run nine times in isolation it fired twice
    (``parity/meep_gpu/results/bg_oblique_rep1``..``rep8``). Nothing raised and nothing
    looked wrong: a real spectrum from the wrong run is as smooth and plausible as the
    right one, which is the failure class this package refuses everywhere else.

    The storage is therefore the PAIR list, and nothing is keyed on anything. Lookups
    come in two spellings, both safe for that reason:

    * :meth:`resolve` takes the MEEP object and matches it with ``is``. No address is
      involved at any point, so the hazard above cannot be expressed. Every reader
      inside this module goes through it.
    * ``mapping[id(monitor)]`` is the mapping protocol, kept because a caller holding
      the object may spell the lookup either way. It is safe HERE and only here: every
      key is computed on demand from a live object this container owns, so no key can
      ever name a freed address.

    Not a dict keyed on the monitor itself, and NOT because of hashability — measured on
    the installed MEEP 1.33.0, ``DftFlux``/``DftNear2Far``/``DftForce``/``DftEnergy``/
    ``DftFields`` all inherit ``object.__hash__`` and would work as keys there. The
    constraint the package writes down (``run_on_gpu``'s ``flux_data`` /
    ``minus_flux_data`` take a SEQUENCE OF PAIRS rather than a dict) is that MEEP does
    not promise it in every release, and a pair list does not have to care either way.
    A ``weakref.WeakKeyDictionary`` is not the alternative for a different and stronger
    reason: a weak reference is precisely the one that does NOT pin the address, so it
    would reproduce this defect with extra machinery.
    """

    __slots__ = ("_pairs",)

    def __init__(self, pairs: Sequence[tuple[Any, Any]] = ()):
        self._pairs: list[tuple[Any, Any]] = list(pairs)

    # THE TWO LINES THE CLASS EXISTS FOR, kept adjacent and spelled plainly. `add`
    # stores the MEEP OBJECT (never its `id`), which is the strong reference; `resolve`
    # compares with `is`, so no address enters the lookup. The mutation battery swaps
    # the pair for a `weakref.ref` — every live lookup keeps working and only the
    # lifetime changes — and requires a test to notice
    # (`parity/meep_gpu/mutate_from_meep.py::migrated_monitor_held_weakly`).
    def add(self, meep_monitor, migrated):
        """Record one ``(MEEP monitor, ours)`` pair, the OBJECT and not its id; returns ours."""
        self._pairs.append((meep_monitor, migrated))
        return migrated

    def resolve(self, meep_monitor):
        """Ours for this MEEP monitor, or ``None``; matched by IDENTITY, not by address."""
        for original, migrated in self._pairs:
            if original is meep_monitor:
                return migrated
        return None

    def pairs(self) -> tuple:
        """The ``(MEEP monitor, ours)`` pairs, in migration order."""
        return tuple(self._pairs)

    def __getitem__(self, key: int):
        for original, migrated in self._pairs:
            if id(original) == key:
                return migrated
        raise KeyError(key)

    def __iter__(self):
        return iter([id(original) for original, _ in self._pairs])

    def __len__(self) -> int:
        return len(self._pairs)

    def __repr__(self) -> str:
        return "MigratedMonitors({})".format(", ".join(
            f"{type(original).__name__}->{type(migrated).__name__}"
            for original, migrated in self._pairs))


@dataclass(frozen=True)
class GpuRunResult:
    """The outcome of :func:`run_on_gpu`: the stepped driver plus what the run cost.

    The driver is handed back rather than closed, because everything a caller
    wants next lives on it — the fields, ``get_epsilon``, any monitor they added
    through a ``prepare`` hook. Call :meth:`close` (or ``result.driver.close()``)
    when finished; on a GPU that is what returns the device memory to the pool.
    """

    driver: FdtdDriver
    steps: int  # Time steps taken by this call.
    meep_time: float  # Simulation time reached, MEEP's Simulation.meep_time().
    wall_time_s: float  # Wall-clock seconds inside FdtdDriver.run.

    @property
    def monitors(self) -> MigratedMonitors:
        """Every MEEP monitor the lift rebuilt — see :class:`MigratedMonitors`.

        The container itself, not a ``dict`` copy of it. The copy used to be defensive
        (a mapping nobody can mutate), but a plain ``dict`` keyed by ``id`` drops the
        strong references that make those keys mean anything, and the container is
        already read-only. :class:`MigratedMonitors` says what the references are for.
        """
        return getattr(self.driver, "migrated_monitors", None) or MigratedMonitors()

    @property
    def lift_record(self):
        """The MEEP build the lift read and the precision the engine steps, or None.

        ``{"meep_version", "meep_precision", "engine_precision"}``, the mapping
        :func:`lift_simulation` attaches to the driver as ``driver.lift_record``.
        """
        return getattr(self.driver, "lift_record", None)

    @property
    def sigma_lift(self):
        """Which route produced the per-point susceptibility sigma, or None.

        :class:`StructuredSigmaLift` when the cell's media differed in their
        ``E_susceptibilities`` and the per-point volumes had to come off MEEP;
        None for every other cell, where the dispersion is one medium's declared
        terms and no volume exists.
        """
        return getattr(self.driver, "sigma_lift", None)

    def monitor_for(self, meep_monitor):
        """The driver-side monitor rebuilt from a MEEP monitor object.

        Resolved by IDENTITY (:meth:`MigratedMonitors.resolve`), never by ``id``: the
        object handed in may be a LATER run's monitor allocated at a collected earlier
        one's address, and an address match there returns the earlier run's spectrum
        without raising. That is not hypothetical — see :class:`MigratedMonitors`.

        Raises rather than returning None for an unknown object: the thing a caller
        would do with a None here is read a spectrum off it, and the failure would
        surface as an empty result — the exact outcome the migration exists to stop.
        The KeyError is also a load-bearing signal for a caller holding SEVERAL runs'
        results, which asks each in turn and falls through on it.
        """
        found = self.monitors.resolve(meep_monitor)
        if found is None:
            raise KeyError(
                f"{type(meep_monitor).__name__} was not migrated onto this run. Only "
                f"{sorted(MIGRATABLE_MONITORS)} are rebuilt, and only monitors that were "
                f"already attached to the mp.Simulation when it was lifted."
            )
        return found

    def get_flux_spectrum(self, meep_flux) -> Any:
        """The flux spectrum for a migrated ``sim.add_flux`` monitor.

        The counterpart of ``mp.get_fluxes(flux)``, keyed by the object the script
        already holds, so reading results back does not require knowing this engine's
        monitor types.

        Named for what it returns, and deliberately NOT ``get_flux_data``: MEEP
        spells those two apart — ``mp.get_fluxes(flux)`` is the spectrum,
        ``sim.get_flux_data(flux)`` is the opaque transform for the normalization
        idiom — and this class now matches, so a script can be transliterated a line
        at a time without either name meaning something else here.
        """
        return self.monitor_for(meep_flux).get_flux_spectrum()

    def get_forces(self, meep_force) -> Any:
        """The force spectrum for a migrated ``sim.add_force`` monitor.

        The counterpart of ``mp.get_forces(force)`` (python/simulation.py:6066),
        keyed by the object the script already holds. Only the DIAGONAL branch of
        MEEP's stress tensor is reproduced — :class:`~.dft.ForceMonitor` names the
        rest, and :func:`_migrate_force` refuses it at migration time rather than
        returning a number nobody could check.
        """
        return self.monitor_for(meep_force).get_force_spectrum()

    def get_force_data(self, meep_force) -> Any:
        """This run's force transforms, MEEP's ``sim.get_force_data(force)``.

        Opaque, like MEEP's; hand it back to :meth:`load_force_data`. It exists so a
        script's store/load round trip runs on THIS engine's accumulators — MEEP's own
        method reads ``force.offdiag1``/``diag``, which are empty on a lifted run, so
        the round trip would save and reload zeros while the answer came from
        somewhere else entirely.
        """
        return self.monitor_for(meep_force).get_dft_data()

    def load_force_data(self, meep_force, data) -> None:
        """MEEP's ``sim.load_force_data(force, fdata)``, on this engine's accumulators."""
        self.monitor_for(meep_force).load_dft_data(data)

    def get_electric_energy(self, meep_energy) -> Any:
        """``mp.get_electric_energy(energy)`` for a migrated ``sim.add_energy`` monitor."""
        return self.monitor_for(meep_energy).get_electric_energy_spectrum()

    def get_magnetic_energy(self, meep_energy) -> Any:
        """``mp.get_magnetic_energy(energy)`` for a migrated ``sim.add_energy`` monitor."""
        return self.monitor_for(meep_energy).get_magnetic_energy_spectrum()

    def get_total_energy(self, meep_energy) -> Any:
        """``mp.get_total_energy(energy)`` — electric plus magnetic, MEEP's own sum."""
        return self.monitor_for(meep_energy).get_total_energy_spectrum()

    def get_flux_data(self, meep_flux) -> Any:
        """The accumulated transforms, MEEP's ``sim.get_flux_data(flux)``.

        Opaque, like MEEP's: hand it to ``run_on_gpu(..., minus_flux_data=[(flux2,
        data)])`` on a later simulation carrying the same flux region. That is the
        TWO-RUN NORMALIZATION IDIOM — run once without the structure, save, run again
        with it, subtract — transliterated from::

            sim.run(...); data = sim.get_flux_data(flux)      # normalization run
            sim2.load_minus_flux_data(flux2, data); sim2.run(...)

        to::

            r = run_on_gpu(sim, until_after_sources=...); data = r.get_flux_data(flux)
            run_on_gpu(sim2, until_after_sources=..., minus_flux_data=[(flux2, data)])

        Why it is not simply ``sim2.load_minus_flux_data(flux2, data)`` followed by a
        lift: MEEP's method reaches ``flux.E``, whose lazy ``swigobj`` property calls
        ``init_sim()``, and an initialized simulation is a documented refusal of this
        lift (:func:`gpu_compatibility` — a caller can rewrite ``fields.boundaries``
        per face through an opaque pointer no reader here can see). The load is
        therefore declared to ``run_on_gpu`` and applied to this engine's own
        accumulators after the lift, which leaves MEEP untouched.

        See :meth:`~.dft.FluxMonitor.load_minus_dft_data` for what the minus means.
        """
        return self.monitor_for(meep_flux).get_dft_data()

    def load_near2far(self, sim, meep_near2far) -> Any:
        """Write this run's near-surface DFT back into MEEP's own ``DftNear2Far``.

        After this, every far-field call is MEEP's unchanged —
        ``sim.get_farfields(obj, ...)``, ``sim.get_farfield(...)``,
        ``sim.output_farfields(...)`` — because the object now holds the fields this
        engine accumulated. That split is deliberate and is the package's design test:
        the near-surface DFT runs inside the per-timestep loop over the grid, so it
        belongs here; the Green's-function far-field evaluation runs once, on a handful
        of host numbers, so reimplementing it would add a second thing to keep correct
        and buy no speed.

        ``sim`` must be the simulation the monitor came from, and must still be
        initialized — ``load_near2far_data`` writes into the live C++ chunks.
        """
        migration = self.monitor_for(meep_near2far)
        import meep as mp  # Deferred: MEEP is optional to import this module.
        from meep.simulation import NearToFarData
        sim.load_near2far_data(meep_near2far, NearToFarData(F=migration.packed(mp, meep_near2far)))
        return meep_near2far

    def pack_flux_data(self, mp, meep_flux) -> Any:
        """This run's flux transform as MEEP's own ``FluxData``, ready for ``load_flux_data``.

        The near2far counterpart (:meth:`load_near2far`) exists so MEEP's unchanged
        far-field evaluator can run on this engine's fields. This is the same move for
        the flux side, and it buys the one reader that matters most:
        ``sim.get_eigenmode_coefficients``. That routine splits cleanly (src/mpb.cpp:925-1004)
        — the mode comes from MPB and depends only on epsilon and frequency, and the
        ONLY place the stepped fields enter is ``get_mode_flux_overlap(mode_data, flux,
        nf, ...)`` against the ``dft_flux``'s accumulated E and H chunks. So::

            sim.load_flux_data(flux, result.pack_flux_data(mp, flux))
            coeffs = sim.get_eigenmode_coefficients(flux, [1])   # MEEP's own, unchanged

        Verified as a mechanism before the packer was written, on CPU MEEP alone: run
        A normally, save ``sim.get_flux_data(mon)``, build a fresh identical
        simulation, ``init_sim()``, ``load_flux_data``, then call MEEP's unchanged
        ``get_eigenmode_coefficients`` — ``alpha`` differed from run A's by
        **0.000e+00** (bit-identical), with ``vgrp`` and ``kdom`` identical, while the
        same fresh simulation with NOTHING loaded returned ``alpha = 0 + 0j``. So the
        loaded data is demonstrably what the reader reads, and any residual is
        attributable to this packer alone.

        THE LAYOUT IS NOT THIS ENGINE'S TO CHOOSE, exactly as in
        :meth:`Near2FarMigration.packed`: it is whatever MEEP's own chunk
        decomposition produced, so it is read off the live object — ``swigobj.E`` and
        ``swigobj.H`` walked through ``next_in_dft``, each chunk's ``is``/``ie``
        naming its sites in MEEP's doubled lattice, emitted in ``LOOP_OVER_IVECS``
        order with sites major and frequencies minor, and concatenated.

        Scope, refused BY NAME rather than approximated:

        * a multi-region flux box (``FluxMigration``), whose planes MEEP keeps as one
          chunk list built by prepending;
        * a monitor built with ``add_mode_monitor(..., yee_grid=True)``, whose chunks
          sit on each component's OWN lattice rather than one centred one — a
          different monitor, not a different layout (see the refusal for the numbers);
        * a cylindrical monitor, whose chunks name R and Z.

        A lattice ``shift`` is CARRIED, not refused — see ``_packed_flux_chunk``.

        A SYMMETRY-REDUCED chunk list is served too, which is why ``sim.add_flux`` and
        ``sim.add_mode_monitor`` are both usable here. ``add_flux`` registers its
        region through ``S.reduce`` (dft.cpp:627-632 passes ``use_symmetry = true``),
        which halves the region on the flipped direction and doubles the stored weight
        (vec.cpp:1437-1456), so MEEP's list covers only half of the plane this engine
        holds. That is a SUBSET, not an approximation, and MEEP itself says so:
        measured on ONE run carrying both monitors on the SAME region, ``add_flux``
        gives 102 E sites at ``stored_weight`` -2 against ``add_mode_monitor``'s 200
        at -1, and MEEP's own answers from the two agree to **3.159e-08** on flux and
        1.674e-08 / 8.596e-07 on the two coefficients. What the halved region does
        change is its own EDGE weights, which is why the measure comes from the chunk
        (:class:`~.dft.MeepChunkWeight`) rather than from this engine's model of the
        region.

        A chunk MEEP created through a SYMMETRY IMAGE (``sn > 0``) IS served, and the
        reason it has to be is worth recording because the natural reading of MEEP's
        source says otherwise. ``add_mode_monitor`` forces ``use_symmetry = false``
        (dft.cpp:835-839) — but that flag is only stored on the ``dft_flux`` object;
        the chunks come from ``add_dft``, whose ``loop_in_chunks`` call passes no
        ``use_symmetry`` at all and therefore takes the default TRUE (dft.cpp:231). So
        every flux monitor on a mirrored cell has symmetry-image chunks, measured:
        ``test_mode_coeffs``'s ``add_mode_monitor`` under ``Mirror(Y)`` produces an Ez
        chunk with ``sn = 1``. They are handled the way
        :meth:`Near2FarMigration.packed` handles them — the chunk's bounds are carried
        into user space through the flip about doubled 0, and the flipped axes are
        emitted in descending stored order, with no extra sign on the VALUES because
        MEEP bakes ``S.phase_shift(c, sn)`` into the reflected chunk's scale
        (dft.cpp:169-171) and this monitor's reflected gather folds the same parity in
        (:meth:`~.dft.FluxMonitor._integrate_flux`).
        """
        from meep.simulation import FluxData  # Deferred: MEEP is optional to import this module.

        monitor = self.monitor_for(meep_flux)
        if not isinstance(monitor, FluxMonitor):
            raise MeepSimulationNotLiftable([
                f"this flux monitor migrated to {type(monitor).__name__}, not a single plane; "
                f"MEEP keeps a multi-region flux box as ONE chunk list built by prepending "
                f"(dft.cpp:627-631), and this packer lays out one plane."
            ])
        swig = getattr(meep_flux, "swigobj", None)
        if swig is None or getattr(swig, "E", None) is None:
            raise MeepSimulationNotLiftable([
                "this DftFlux has no accumulated chunk list (swigobj.E); the simulation must "
                "have been initialized before its data can be laid out."
            ])
        mode_args = list(getattr(meep_flux, "args", ()) or ())
        if _is_mode_monitor(meep_flux) and len(mode_args) > 2 and bool(mode_args[2]):
            # NAMED, because the symptom is otherwise a cryptic lattice error one frame
            # down ("doubled coordinate -60 on axis 0 is not on this plane's centred
            # lattice", which is 2*resolution*(-1.0), the Ez YEE site of a monitor at
            # x = -1.0). `yee_grid=True` is `centered_grid = false`
            # (simulation.py:3550), and `add_dft` then brackets each chunk on ITS OWN
            # component lattice instead of Centered (dft.cpp:231), so an Ez chunk sits
            # at doubled EVEN coordinates with a single x site where Hy sits at doubled
            # ODD ones. This engine's flux plane is one CENTRED lattice for all six
            # components (`FluxMonitor.site_doubled`), and the difference is not
            # cosmetic: measured in STOCK MEEP alone, the two monitors on the SAME
            # region of the same run report flux 9.393 % apart (1.9906e-04 centred
            # against 2.1776e-04 yee) with coefficient magnitudes 0.9925x forward and
            # 1.0146x backward, and the yee list holds 81 E sites where the centred
            # one holds 164. So substituting one for the other is a wrong answer, not
            # a re-layout.
            raise MeepSimulationNotLiftable([
                "this monitor was built with `add_mode_monitor(..., yee_grid=True)`, so "
                "MEEP registered each component on its OWN Yee lattice rather than one "
                "centred one (centered_grid = not yee_grid, simulation.py:3550; "
                "dft.cpp:231 passes `use_centered_grid ? Centered : c`). This engine's "
                "flux plane samples cell centres for all six components, and the two "
                "monitors do not answer the same question — measured in stock MEEP, "
                "9.393 % apart on flux and 0.9925x / 1.0146x on the two coefficients. "
                "Use the default `yee_grid=False`."
            ])
        grid = self.driver.grid
        if getattr(grid, "cylindrical", False):
            raise MeepSimulationNotLiftable([
                "a cylindrical flux monitor's chunks name MEEP's R and Z directions, not X/Y/Z; "
                "this packer lays out a Cartesian plane."
            ])
        names = _component_names(mp, False)
        invariant = tuple(grid.invariant_axes)
        directions = (mp.X, mp.Y, mp.Z)
        mirrored_axes = _grid_mirror_axes(grid)
        # THE OVERLAP CHECK. MEEP's chunk list is a SUBSET of this engine's plane, not
        # necessarily a cover of it: `add_flux` takes `use_symmetry = true`, so the
        # region it registers is `S.reduce`'s half (vec.cpp:1437-1456 halves the
        # flipped direction and doubles the weight), while this monitor holds all of
        # it. Measured on `test_dft_energy`'s `Mirror(Y)` cell, one run carrying both
        # monitors: `add_flux` gives 102 E sites at `stored_weight` -2 where
        # `add_mode_monitor` gives 200 at -1, and MEEP's own answers from the two agree
        # to 3.159e-08 on flux and 1.674e-08 / 8.596e-07 on the two coefficients. So a
        # short list is a well-defined request and is served; what must not happen is
        # two chunks claiming the SAME user site, which would double-count one and drop
        # another. That is checked per component, on the boxes each chunk mapped to.
        #
        # The genuinely ambiguous case is MEEP's OWN refusal, not this one: a folded
        # monitor cannot be decomposed against an unpolarised mode, so
        # `fields::get_eigenmode_coefficients` aborts when
        # `flux.use_symmetry && S.multiplicity() > 1 && parity == 0` (mpb.cpp:934-936).
        # Both surviving corpus rows pass `eig_parity = ODD_Z + EVEN_Y`, so they are
        # legal MEEP, and MEEP's own bookkeeping carries the halving:
        # `csc = sqrt((flux.use_symmetry ? S.multiplicity() : 1.0) / |normfac|)`
        # (mpb.cpp:996) puts the missing half back, which is why the two agree above.
        sides = []
        for head in (swig.E, swig.H):
            parts, claimed, chunk = [], {}, head
            while chunk is not None:
                component = names.get(int(chunk.c))
                values, box = self._packed_flux_chunk(monitor, chunk, names, directions,
                                                      invariant, mirrored_axes)
                for earlier in claimed.setdefault(component, []):
                    if all(box[axis][0] <= earlier[axis][1] and earlier[axis][0] <= box[axis][1]
                           for axis in range(3)):
                        raise MeepSimulationNotLiftable([
                            f"two of MEEP's {component} flux chunks claim the same grid sites "
                            f"— {earlier} and {box} in this plane's own indices. One site would "
                            f"be written twice and another left at zero, so the layout cannot "
                            f"be reproduced."
                        ])
                claimed[component].append(box)
                parts.append(values)
                chunk = chunk.next_in_dft
            sides.append(numpy.concatenate(parts) if parts
                         else numpy.zeros(0, dtype=numpy.complex128))
        return FluxData(E=sides[0], H=sides[1])

    def _packed_flux_chunk(self, monitor, chunk, names, directions, invariant, mirrored_axes=()):
        """This engine's samples for one MEEP flux chunk, in that chunk's own bounds.

        Returns ``(values, box)`` — the flat chunk-ordered stream, and the INCLUSIVE
        plane-relative ``((x0, x1), (y0, y1), (z0, z1))`` it was taken from, which is
        what :meth:`pack_flux_data` compares between chunks to catch two chunks
        claiming one site.
        """
        component = names.get(int(chunk.c))
        if component is None:
            raise MeepSimulationNotLiftable([
                f"a flux chunk carries MEEP component {int(chunk.c)}, which is not one of the "
                f"six this engine accumulates."
            ])
        # A chunk MEEP created through a symmetry image stores its is/ie in STORED-chunk
        # coordinates; the sites it serves live at `S.transform(iv, sn) + shift`
        # (dft.cpp:942-943). The accumulator holds the REQUESTED span, so the bounds are
        # carried into user space — sign flipped about doubled 0 on each axis sn flips —
        # and the flipped axes are emitted in descending stored order.
        flips = _sn_flip_axes(mirrored_axes, int(getattr(chunk, "sn", 0) or 0))
        low, high = [], []
        for axis, direction in enumerate(directions):
            if invariant[axis]:
                # A reduced run's ivec has no slot for this direction at all; the
                # accumulator's single plane is bookkeeping, not a coordinate.
                low.append(monitor.first_index(axis))
                high.append(monitor.first_index(axis))
                continue
            # A LATTICE SHIFT is carried, not refused. MEEP loops the lattice images
            # whose translated cell intersects the request, so a plane spanning a whole
            # periodic axis comes back as a main chunk plus a small one at
            # `shift = ±L` (measured on test_diffracted_planewave's grating: an Ez
            # chunk at shift 260 on y). The chunk's sites live at
            # `S.transform(iv, sn) + shift` (dft.cpp:942-943), which is user space —
            # and this monitor's accumulator holds the REQUESTED span in exactly that
            # space, out-of-range indices included, with the Bloch factor already
            # folded in by `_register_volume`'s lattice gather (MEEP folds the matching
            # `shift_phase` into the chunk's own scale). So the bounds simply carry the
            # shift, the same statement `Near2FarMigration.packed` makes.
            shift = int(chunk.shift.in_direction(direction))
            stored_lo = int(chunk._is.in_direction(direction))
            stored_hi = int(chunk.ie.in_direction(direction))
            if axis in flips:
                stored_lo, stored_hi = -stored_hi, -stored_lo
            stored_lo += shift
            stored_hi += shift
            low.append(monitor.index_for_doubled(axis, stored_lo))
            high.append(monitor.index_for_doubled(axis, stored_hi))
        starts = [monitor.first_index(axis) for axis in range(3)]
        stops = [monitor.stop_index(axis) for axis in range(3)]
        if any(low[axis] < starts[axis] or high[axis] >= stops[axis] for axis in range(3)):
            raise MeepSimulationNotLiftable([
                f"MEEP's {component} flux chunk spans grid indices {low}..{high} and this "
                f"engine's plane holds {starts}..{[stop - 1 for stop in stops]}; the two "
                f"registrations disagree, so the data would be laid out wrongly."
            ])
        sites = 1
        for axis in range(3):
            sites *= high[axis] - low[axis] + 1
        if sites != int(chunk.N):
            raise MeepSimulationNotLiftable([
                f"the {component} flux chunk spans {sites} sites by its bounds but MEEP reports "
                f"N={int(chunk.N)}; the index mapping disagrees with MEEP's own chunk."
            ])
        box = tuple((low[axis] - starts[axis], high[axis] - starts[axis]) for axis in range(3))
        try:
            stored_weight = complex(chunk.stored_weight)
        except Exception:  # noqa: BLE001 - an unreadable weight is a refusal, never a guess.
            raise MeepSimulationNotLiftable([
                f"the {component} flux chunk does not expose its stored_weight; MEEP divides "
                f"the stored value by exactly that number when it reads it back "
                f"(dft.cpp:1004), so the layout cannot be reproduced without it."
            ]) from None
        return (monitor.packed(component, box=box, extra_scale=stored_weight,
                               invariant=invariant, flip=tuple(flips),
                               chunk_weight=_meep_chunk_weight(chunk, component, directions)),
                box)

    def get_dft_region(self, meep_fields, component: str, freq_index: int = 0) -> Any:
        """One component of a migrated ``sim.add_dft_fields`` monitor, AS ACCUMULATED.

        Deliberately NOT named ``get_dft_array``, because it is not a drop-in
        counterpart of ``sim.get_dft_array`` and pretending otherwise is a trap. MEEP
        reduces at read time (``array_slice.cpp collapse_array``): an axis whose
        REQUESTED extent was zero is interpolated and summed away, and an axis that
        came out one cell wide is dropped from the shape. This returns the region the
        engine accumulated, so a zero-size z gives ``(nx, ny, 2)`` where MEEP gives
        ``(nx, ny)``.

        That difference matters because the shapes still BROADCAST: subtracting the two
        yields an array rather than an error, so a comparison written as though they
        match produces a plausible wrong number instead of failing. The collapse is
        :meth:`get_dft_array`, which composes this with :func:`collapse_dft_region`.
        """
        return self.monitor_for(meep_fields).get_dft(component, freq_index)

    def get_dft_array(self, meep_fields, component: str, freq_index: int = 0) -> Any:
        """MEEP's ``sim.get_dft_array(dft_obj, c, nf)`` — the region, reduced as MEEP reduces it.

        :meth:`get_dft_region` hands back the block this engine accumulated;
        ``sim.get_dft_array`` hands back that block after ``process_dft_component``
        has multiplied in the EMPTY-DIMENSION interpolation weights and
        ``collapse_array`` has summed those axes away (dft.cpp:1283-1291,
        array_slice.cpp:525-597). This composes the two, so a script that reads a DFT
        region through MEEP can be transliterated a line at a time.

        A ``yee_grid=True`` monitor is not a special case of the centred one and the
        difference is not small: the interpolation fraction is measured against the
        REQUESTED COMPONENT'S OWN Yee lattice (``loop_in_chunks`` builds ``is``/``ie``
        with ``cgrid = c``, loop_in_chunks.cpp:350-357), not against the cell centres.
        Measured on a 2-D x-y run at resolution 15 with a zero-thickness line monitor
        of Ez: the centred fraction gives **1.67e-01** relative against CPU MEEP and
        the component's own gives **1.63e-07**. ``test_planewave_1D`` uses
        ``yee_grid=True`` in three of its five sub-cases, so the corner is load-bearing.

        A run with mirror symmetry needs no unfolding here, which is worth stating
        because the opposite is the natural guess. MEEP's ``add_dft_fields`` loops the
        symmetry images (``use_symmetry`` defaults true in ``loop_in_chunks``, and
        ``add_dft`` takes that default — dft.cpp:231), so its array covers the whole
        REQUESTED volume; this engine's centred monitor covers the same requested span
        for the same reason, through ``_register_volume``'s reflected gather, and its
        region indices are already full-domain (they run negative below the plane).
        Calling :meth:`~.dft.DFTMonitor.get_dft_full` on top of that would mirror an
        already-mirrored region.
        """
        migration = self.monitor_for(meep_fields)
        center, size = _monitor_region(meep_fields)
        grid = self.driver.grid
        size = _meep_region_extents(grid, size)
        if isinstance(migration, DftFieldsYeeMigration):
            accumulator = migration.accumulators.get(component)
            if accumulator is None:
                raise ValueError(
                    f"Component '{component}' is not monitored. Monitored: "
                    f"{', '.join(sorted(migration.accumulators))}")
            region = numpy.asarray(to_numpy(migration.get_dft(component, freq_index)))
            # `site_doubled` indexes the GRID, not the accumulator's own block, so the
            # first stored sample is `first_index`, not 0 (`_emit_chunk` reads the pair
            # the same way round).
            first_doubled = tuple(accumulator.site_doubled(axis, accumulator.first_index(axis))
                                  for axis in range(3))
        else:
            region = numpy.asarray(to_numpy(migration.get_dft(component, freq_index)))
            first_doubled = tuple(migration.site_doubled(axis, migration.region[2 * axis])
                                  for axis in range(3))
        return collapse_dft_region(region, first_doubled, grid.resolution, center, size,
                                   invariant=tuple(grid.invariant_axes))

    def get_array(self, component: str) -> Any:
        """One field component in MEEP's own ``sim.get_array`` layout.

        Same shape, same registration, same unfolding of any mirror symmetry, so::

            numpy.allclose(result.get_array("Ez"), sim.get_array(component=mp.Ez))

        is the comparison, with no slicing and no index arithmetic. That is
        :meth:`~.fields.Fields.to_meep_array`, which is where the convention is
        defined and cross-validated, plus the two corrections below. The result
        is a host array on either backend, and a private copy, so a caller cannot
        reach engine state through it.

        THE METALLIC CORRECTION. MEEP's array gains its extra plane per axis from the
        PERIODIC image, so an axis with no lattice vector gains none: with ``k_point``
        unset MEEP returns ``N`` points per axis at ``axis_origin + (j + 0.5)*dx``,
        against ``N + 1`` at ``axis_origin + (j - 0.5)*dx`` for the periodic run of the
        same cell (both read off ``sim.get_array_metadata``). ``Fields
        ._add_boundary_cells`` prepends on every unfolded axis, so the duplicate is
        dropped again here — an exact slice, no arithmetic, which is why the tripwire
        below can insist it really was a duplicate. Like the Bloch correction this
        belongs in ``fields.py`` and is applied here so that this method's contract
        holds today.

        THE BLOCH CORRECTION, and why it is here rather than in ``Fields``.
        MEEP's array prepends one plane per periodic axis at
        ``axis_origin - dx/2``, which is one lattice vector BELOW the last plane
        and therefore carries ``conj(bloch_phase)`` times it — measured directly
        against ``sim.get_array`` at k = (0.1234567, 0, 0):
        ``ez[0] / ez[-1] = 0.019392 - 0.999812j``, which is
        ``conj(exp(2j*pi*k*L))`` to every digit.
        ``Fields._add_boundary_cells`` prepends a plain copy instead, so its array
        is **1.98e-01** wrong against CPU MEEP on a Bloch run (and exact,
        3.02e-07, everywhere else). That gap is latent today — nothing but this
        method calls ``to_meep_array`` at a nonzero ``k_point`` — and it belongs
        in ``fields.py``; it is corrected here so that this method's contract
        holds, and guarded by the tripwire below so the correction cannot be
        applied twice once ``Fields`` carries it.
        """
        fields = self.driver.fields
        grid = self.driver.grid
        values = numpy.array(to_numpy(fields.to_meep_array(component)), copy=True)
        # `to_meep_array` has already dropped every axis MEEP does not report
        # (Fields.meep_array_axes), so an axis NUMBER has to be translated into a
        # POSITION in the returned array before either correction below can slice it.
        # Using the number directly would apply the x correction to z on a 1-D run.
        position_of = {axis: position for position, axis in enumerate(fields.meep_array_axes())}
        for axis in range(3):
            if axis not in position_of:
                continue  # MEEP reports no such axis; there is no plane to correct.
            if not grid.is_metallic(axis) or grid.is_mirrored(axis):
                continue  # No wall here, or a folded axis, which is never prepended to.
            axis = position_of[axis]
            first = (slice(None),) * axis + (0,)
            last = (slice(None),) * axis + (-1,)
            if not numpy.array_equal(values[first], values[last]):
                raise RuntimeError(
                    f"Fields.to_meep_array no longer prepends a plain duplicate on the "
                    f"{AXIS_NAMES[axis]} axis, so it has learned the metallic convention itself. "
                    f"Delete the slice in GpuRunResult.get_array — applying it twice would drop a "
                    f"real plane of field off the low face of a PEC axis and leave the rest of the "
                    f"volume exact."
                )
            values = values[(slice(None),) * axis + (slice(1, None),)]
        for axis in range(3):
            # A nonzero phase implies complex storage — FdtdDriver refuses Bloch on
            # real fields outright — so the multiply below cannot drop an imaginary
            # part into a float32 array.
            phase = grid.bloch_phase(axis)
            if phase is None or grid.is_mirrored(axis) or axis not in position_of:
                continue  # No lattice vector on this axis, no prepended plane, or no axis at all.
            axis = position_of[axis]
            first = (slice(None),) * axis + (0,)
            last = (slice(None),) * axis + (-1,)
            if not numpy.array_equal(values[first], values[last]):
                raise RuntimeError(
                    f"Fields.to_meep_array no longer prepends a plain duplicate on the "
                    f"{AXIS_NAMES[axis]} axis, so it has learned the Bloch factor itself. Delete "
                    f"the correction in GpuRunResult.get_array — applying it twice would square "
                    f"conj(bloch_phase) on that plane and leave the rest of the volume exact, "
                    f"which is the quietest possible way to be wrong."
                )
            values[first] = values[first] * numpy.conjugate(phase)
        return values

    def get_field(self, component: str, cell_centered: bool = True) -> Any:
        """One field component in the driver's own layout (no boundary duplicate).

        ``result.get_array(c)[1:, 1:, 1:]`` on an unfolded run, and the raw stored
        quadrant when a mirror plane is active.
        """
        return self.driver.get_field(component, cell_centered=cell_centered)

    def close(self) -> None:  # Release the driver's arrays (and, on GPU, its pool blocks).
        self.driver.close()


def _import_meep():  # Import MEEP on demand, with a message that says what to install.
    """Return the ``meep`` module, or raise saying this package does not require it.

    Imported here rather than at module scope on purpose: every other module in
    this package runs on a machine with neither MEEP nor CuPy, and
    ``test_package_boundary.py`` exists to keep it that way. Only the functions
    that lift an ``mp.Simulation`` need MEEP, and they are the only ones that pay
    for it.
    """
    try:
        import meep  # noqa: PLC0415 - deliberately lazy; see the docstring.
    except ImportError as exc:  # pragma: no cover - exercised only without MEEP installed.
        raise ImportError(
            "meep_gpu.from_meep needs CPU MEEP importable to read an mp.Simulation "
            "(conda install -c conda-forge pymeep). The rest of meep_gpu — FdtdDriver "
            "and everything under it — does not, and stays usable without it."
        ) from exc
    return meep


#: What the engine steps, whatever the MEEP build: ``float32`` fields, ``complex64``
#: for a complex run.
ENGINE_PRECISION = "single"

#: The line a lift from a double-precision MEEP prints, once per process. The
#: figures are relative L2 differences of the final field between the NumPy
#: reference and MEEP 1.33.0 built in double precision: 4 simulations on 1 host,
#: 2026-09-28.
DOUBLE_PRECISION_NOTICE = (
    "meep_gpu: this MEEP build is double precision and the engine steps single "
    "precision (float32 fields, complex64 for a complex run). The lift proceeds. "
    "Expect agreement with a MEEP run of the same simulation at the level of "
    "single-precision rounding, not of double precision: final fields differed by "
    "1.5e-6 to 5.1e-4 (relative L2) on the 4 simulations compared")

_PRECISION_ANNOUNCED: set = set()


def _meep_build(mp) -> dict:  # The MEEP build a lift read: its version and its precision.
    """``{"meep_version", "meep_precision", "engine_precision"}``, for the lift's record.

    ``meep_precision`` is ``"single"`` or ``"double"`` as ``mp.is_single_precision()``
    answers, and ``None`` where this MEEP has no such function or it raises: a
    precision that was not read is recorded as unread and nothing is printed for it.
    Never raises, and refuses nothing.
    """
    precision = None
    try:
        precision = "single" if mp.is_single_precision() else "double"
    except Exception:  # noqa: BLE001 - an unreadable build is recorded, not refused
        pass
    return {
        "meep_version": getattr(mp, "__version__", None),
        "meep_precision": precision,
        "engine_precision": ENGINE_PRECISION,
    }


def _announce_precision(build: Mapping) -> None:  # One line to stderr per process, double only.
    if build.get("meep_precision") != "double":
        return
    if DOUBLE_PRECISION_NOTICE in _PRECISION_ANNOUNCED:
        return
    _PRECISION_ANNOUNCED.add(DOUBLE_PRECISION_NOTICE)
    print(DOUBLE_PRECISION_NOTICE, file=sys.stderr, flush=True)


def _component_names(mp, cylindrical: bool = False) -> dict:  # MEEP's component integers -> this engine's names.
    """The six current-carrying components, by MEEP's integer constant.

    Built per call from the module rather than pinned as literals: MEEP's
    ``component`` enum interleaves the cylindrical directions (Ez is 4, not 2),
    so a hard-coded table is a silent mis-mapping waiting for an enum change.

    In cylindrical mode the six live components are Er/Ep/Ez/Hr/Hp/Hz, carried
    on this engine's axes under the x -> r, y -> phi mapping the whole port uses
    (step_db.cpp's Dcyl loops against the Cartesian shift table); Ex/Ey/Hx/Hy
    are then absent from the map exactly as MEEP's ``coordinate_mismatch``
    rejects them in Dcyl.
    """
    if cylindrical:
        return {
            mp.Er: "Ex", mp.Ep: "Ey", mp.Ez: "Ez",
            mp.Hr: "Hx", mp.Hp: "Hy", mp.Hz: "Hz",
        }
    return {
        mp.Ex: "Ex", mp.Ey: "Ey", mp.Ez: "Ez",
        mp.Hx: "Hx", mp.Hy: "Hy", mp.Hz: "Hz",
    }


def _vector3(value) -> tuple[float, float, float]:  # mp.Vector3 -> a plain (x, y, z).
    return (float(value.x), float(value.y), float(value.z))


def _is_cylindrical(sim) -> bool:  # Either spelling of a Dcyl run.
    """True for a cylindrical simulation, whichever way the script declared it.

    MEEP's constructor stores ``dimensions=mp.CYLINDRICAL`` verbatim and flips
    ``is_cylindrical=True`` (rewriting dimensions to 2) only later, inside
    ``_fit_volume_to_simulation`` — so before anything touches a Volume, one
    flag is set and the other is not, and which one depends on how far the
    script has run. Reading either alone misses the other spelling.
    """
    return bool(getattr(sim, "is_cylindrical", False)) or int(sim.dimensions) == MEEP_CYLINDRICAL


def _is_medium(mp, material) -> bool:  # A plain mp.Medium, not a grid / function / subclass.
    """True only for ``mp.Medium`` itself.

    A material may be a callable, a string or an ndarray, none of which resolve
    to one homogeneous medium, and MEEP's own ``Medium`` subclasses carry state
    this module reads separately — so the EXACT type is tested rather than an
    ``isinstance``, and a subclass added to a future MEEP is refused by name
    instead of being lifted as its base's defaults.

    ``mp.MaterialGrid`` is NOT one of those subclasses: python/geom.py:578
    declares ``class MaterialGrid:`` with no base at all and its runtime MRO is
    ``(MaterialGrid, object)``, so it has no ``epsilon_diag`` to read. It never
    reaches here anyway — :func:`_materials_of` expands each grid into the two
    endpoint media MEEP interpolates between, so what this function sees is
    already a plain medium or something that was never a medium at all.
    """
    return type(material) is mp.Medium


def _is_material_grid(mp, material) -> bool:  # mp.MaterialGrid itself, not a subclass of it.
    """True only for ``mp.MaterialGrid``, and deliberately not for a subclass.

    Same discipline as :func:`_is_medium` and for the same reason: a subclass
    could carry per-point state on top of the grid's, and falling through to
    the grid's own handling would run the base object under the subclass's
    name. A subclass is neither a medium nor a grid here, so it lands on the
    material-kind refusal in :func:`_check_materials` — which is the safe side.
    """
    grid_type = getattr(mp, "MaterialGrid", None)
    return grid_type is not None and type(material) is grid_type


def _medium_signature(medium) -> tuple:  # Everything about a medium that must match another.
    """A hashable summary of every material property this module reads or refuses.

    Two media with the same signature are the same material to this engine, so a
    ``geometry`` object whose material repeats the ``default_material`` does not
    make the cell inhomogeneous. Every field MEEP's ``Medium`` carries appears
    here — including the ones that are refused — because a difference in a
    refused field is still a difference, and collapsing it would let two distinct
    materials pass as one.
    """
    return (
        _vector3(medium.epsilon_diag), _vector3(medium.epsilon_offdiag),
        _vector3(medium.mu_diag), _vector3(medium.mu_offdiag),
        _vector3(medium.E_chi2_diag), _vector3(medium.E_chi3_diag),
        _vector3(medium.H_chi2_diag), _vector3(medium.H_chi3_diag),
        _vector3(medium.D_conductivity_diag), _vector3(medium.D_conductivity_offdiag),
        _vector3(medium.B_conductivity_diag), _vector3(medium.B_conductivity_offdiag),
        tuple(_susceptibility_signature(term) for term in medium.E_susceptibilities),
        tuple(_susceptibility_signature(term) for term in medium.H_susceptibilities),
    )


# Signature entries a STRUCTURED cell is allowed to differ in, because each one is
# RECOVERED per point off MEEP's own rasterized structure rather than taken from a
# declared medium. Every other entry has no reader and must agree across the cell.
#
#   0  epsilon_diag      chi1inv's diagonal at frequency 0   (_lift_epsilon_structured)
#   1  epsilon_offdiag   chi1inv's off-diagonal at frequency 0, installed through
#                        set_epsilon_components(chi1inv_offdiagonal=...) — the same read,
#                        already measured on curved and rotated cells
#   8  D_conductivity_diag   the D-ROW chi1inv at a NONZERO frequency
#                        (_read_conductivity_volumes; monitor.cpp:339-343)
#
# Index 10 (B_conductivity_diag) is deliberately NOT here even though a magnetic loss
# IS lifted now (FdtdDriver.set_b_conductivity): the read that recovers a per-point
# conductivity walks the D row, and there is no B-row equivalent, so a B_conductivity
# that DIFFERS from medium to medium is still unrecoverable. What lifts is the whole
# cell's shared declaration, taken off the representative medium in _lift_material.
#
# Index 9 (D_conductivity_offdiag) is deliberately NOT here: a rotated conductivity
# tensor has no diagonal representation in this engine's loss term at all, so
# recovering it per point would not help. Index 12 (E_susceptibilities) is recoverable
# too but takes its own branch, because its route carries extra preconditions.
PER_POINT_SIGNATURE_INDICES = (0, 1, 8)
SUSCEPTIBILITY_SIGNATURE_INDEX = 12


def _signature_without(signature: tuple, indices: tuple) -> tuple:
    """``signature`` with the named entries dropped — the medium-comparison key."""
    return tuple(entry for index, entry in enumerate(signature) if index not in indices)


def _susceptibility_signature(term) -> tuple:  # One susceptibility, for the medium comparison.
    return (
        type(term).__name__,
        float(getattr(term, "frequency", 0.0)),
        float(getattr(term, "gamma", 0.0)),
        _vector3(term.sigma_diag), _vector3(term.sigma_offdiag),
        float(getattr(term, "noise_amp", 0.0)),
    )


def _has_sigma_reader(mp) -> bool:  # Does this MEEP carry fields::get_susceptibility_sigma?
    """Probe the CLASS, never an instance: the gate must not call ``init_sim``.

    ``meep-sigma-reader.patch`` adds the exact per-point sigma reader; a MEEP built
    without it has no way to report sigma at all, and the lift falls back to
    :mod:`~.sigma_recovery`. One probe, used by both the gate and the lift, so the
    pre-flight verdict and the route actually taken cannot disagree.
    """
    return hasattr(mp.fields, "get_susceptibility_sigma")


def _susceptibility_equivalence_key(term) -> tuple:  # MEEP's own "same chain entry?" test.
    """The fields ``susceptibility_equiv`` compares — deliberately NOT sigma.

    meepgeom.cpp:1633-1647 merges two declared susceptibilities into ONE entry of
    the structure's chain when they agree on ``bias``, ``frequency``, ``gamma``,
    ``alpha``, ``noise_amp``, ``drude``, ``saturated_gyrotropy``, ``is_file``,
    ``transitions`` and ``initial_populations`` — sigma and the object id are
    explicitly excluded, because sigma is the per-POINT quantity the chain entry
    carries a volume of. Of those fields, only ``frequency``, ``gamma`` and the
    Drude flag can vary across the kinds this module accepts: the gate refuses the
    noisy, gyrotropic and multilevel subclasses BY TYPE, which fixes ``noise_amp``,
    ``alpha``, ``bias``, ``saturated_gyrotropy`` and the transition lists at their
    defaults, and nothing here loads a susceptibility from file.

    This is the same question :func:`_susceptibility_signature` asks and a
    different answer, on purpose. That one INCLUDES sigma, because it decides
    whether two media are the same material; this one excludes it, because it
    decides how many unknowns the recovery has. Using the medium signature here
    would give two poles where MEEP built one — two identical basis columns, which
    :func:`~.sigma_recovery.plan_recovery` then refuses as singular rather than
    splitting one material's sigma arbitrarily between them.
    """
    return (
        type(term).__name__ == "DrudeSusceptibility",
        float(getattr(term, "frequency", 0.0)),
        float(getattr(term, "gamma", 0.0)),
    )


def _unique_susceptibilities(materials) -> tuple:
    """Every distinct susceptibility the cell's media declare, in declaration order.

    The list the recovery solves against: one unknown sigma per entry, per point.
    Order is the order first seen, which is stable but deliberately NOT claimed to
    match MEEP's chain order — the recovery never reads the chain, and each term is
    installed on the driver as its own polarization state.
    """
    seen: dict[tuple, Any] = {}
    for medium in materials:
        for term in getattr(medium, "E_susceptibilities", ()) or ():
            seen.setdefault(_susceptibility_equivalence_key(term), term)
    return tuple(seen.values())


def _recovery_pole(term) -> SusceptibilityPole:  # One mp.*Susceptibility as a recovery unknown.
    return SusceptibilityPole(
        frequency=float(term.frequency),
        gamma=float(term.gamma),
        drude=type(term).__name__ == "DrudeSusceptibility",
    )


def _driver_susceptibility(term) -> Susceptibility:  # One mp.*Susceptibility as an engine term.
    return Susceptibility(
        frequency=float(term.frequency),
        gamma=float(term.gamma),
        kind=DRUDE if type(term).__name__ == "DrudeSusceptibility" else LORENTZIAN,
    )


def _declared_materials(sim) -> list:  # Every material object the cell names, defaults first.
    """The declaration verbatim — grids, callables, strings and all.

    :func:`_materials_of` is the one callers want; this is the traversal under
    it, shared with :func:`_material_grids_of` so the two cannot disagree about
    what the cell contains.

    ``material_function``, ``epsilon_func`` and ``epsilon_input_file`` are listed
    here because MEEP lists them here: ``Simulation._init_structure``
    (python/simulation.py:2015-2024) assigns whichever one is set to
    ``self.default_material`` and builds the structure from that single slot. They
    are appended rather than substituted so the traversal reads the same before and
    after ``init_sim`` — afterwards the same object is ALSO the default material,
    which is why the result is deduplicated by identity.
    """
    materials = [sim.default_material]
    for obj in getattr(sim, "geometry", ()) or ():
        materials.append(getattr(obj, "material", None))
    materials.extend(getattr(sim, "extra_materials", ()) or ())
    for attribute in ("material_function", "epsilon_func", "epsilon_input_file"):
        route = getattr(sim, attribute, None)
        if route is not None and route != "":
            materials.append(route)
    seen: list = []
    for material in materials:
        if not any(material is entry for entry in seen):
            seen.append(material)
    return seen


def _materials_of(mp, sim) -> list:  # Every material the cell could contain, defaults first.
    """Every material the cell can hold, AS PLAIN MEDIA, defaults first.

    A ``mp.MaterialGrid`` is expanded into its two endpoint media rather than
    carried through as itself, because MEEP never steps a grid either:
    ``geom_epsilon::get_material_pt`` evaluates it into an ordinary
    ``medium_struct`` at every point (meepgeom.cpp:845-864) and
    ``epsilon_material_grid`` (meepgeom.cpp:569-583) interpolates the
    permittivity LINEARLY between ``medium_1`` and ``medium_2`` by u in [0, 1].
    The endpoint pair therefore BRACKETS every permittivity the grid can
    produce, which is exactly what the declared-media span check in
    :func:`_lift_epsilon_structured` needs — it holds over the pair with no
    change there — and nothing of the grid survives into the stepper except the
    per-point ``chi1inv`` tensor that check guards.

    The expansion is also what makes the rest of the material machinery apply to
    a grid at all: :func:`_medium_signature` reads ``epsilon_diag`` off its
    argument and a grid has none.

    WHAT THE PAIR DOES NOT SPEAK FOR, and why each is refused rather than
    approximated (:func:`_check_material_grids`): the per-point ``damping``
    conductivity, which is not a property of either endpoint, and the endpoint
    fields MEEP itself DISCARDS — ``md->medium`` is default-constructed
    (material_data.cpp:47-50) and only ``E_susceptibilities`` are merged into it
    (typemap_utils.cpp:528-547), so mu, chi2/chi3, ``B_conductivity`` and the
    magnetic susceptibilities declared on an endpoint never reach the structure.
    """
    materials = []
    for material in _declared_materials(sim):
        if _is_material_grid(mp, material):
            materials.extend((material.medium1, material.medium2))
        elif _epsilon_route_kind(material) is None:
            materials.append(material)
        # else: a per-point epsilon route names no medium at all — see
        # `_epsilon_route_kind`. It is dropped here rather than carried, because
        # every consumer of this list reads `epsilon_diag` off its entries and a
        # callable, an array or a filename has none. `_epsilon_routes` is the list
        # that speaks for them.
    return materials


# --- per-point epsilon routes ---------------------------------------------------------
#
# Four spellings of one thing: a permittivity MEEP evaluates POINT BY POINT into the
# structure instead of naming a medium.
#
#     mp.Simulation(material_function=f)     f(Vector3) -> mp.Medium
#     mp.Simulation(epsilon_func=f)          f(Vector3) -> float
#     mp.Simulation(epsilon_input_file=...)  an HDF5 permittivity volume
#     default_material / a geometry object's material set to a callable or an ndarray
#
# MEEP itself does not distinguish them for long: `Simulation._init_structure`
# (python/simulation.py:2015-2024) drops whichever is set into `default_material`, and
# `geom_epsilon::get_material_pt` evaluates the result at every quadrature point.
#
# THEY USED TO BE REFUSED BY NAME, on the argument that "this module cannot read the
# result back per component". That argument was wrong when it was written and the
# sibling refusal three lines below it said so: `_lift_epsilon_structured` reads
# MEEP's OWN rasterized `chi1inv` off `sim.fields`, one Yee point at a time, and never
# asks how the structure was built. A route is invisible to it.
#
# So the gate now refuses what the route PRODUCED rather than the route itself. Two
# checks, one cheap and one exact:
#
#   * `_probe_material_callable` samples a callable across the cell and requires every
#     medium it returns to be permittivity and nothing else. Pre-flight, so it must
#     not initialize anything; a sample is all it can be.
#   * `_require_epsilon_only_structure` compares MEEP's own rasterized epsilon and mu
#     at frequency 0 against the same arrays at a nonzero frequency, after `init_sim`
#     and before the first step. `structure_chunk::get_chi1inv_at_pt`
#     (monitor.cpp:263-355) folds every susceptibility AND the conductivity into the
#     nonzero-frequency answer, so that comparison is an EXACT whole-cell test for
#     dispersion, loss and magnetism — not a sample.
#
# What escapes both: `chi2`/`chi3`. MEEP keeps the nonlinear volumes outside
# `chi1inv` entirely and publishes no reader for them, so a nonlinearity returned by a
# callable is caught by the probe or not at all. Named in the refusal text, and the
# reason the probe checks the nonlinear fields too.
#
# MEASURED (main reference env, MEEP 1.33.0 single precision, NumPy backend), Linf over
# the whole Ez plane against pristine MEEP stepping the same object for the same
# `until`, both codes at equal step counts:
#
#   test_user_defined_material.py::test_user_material_func              3.896e-04
#   test_user_defined_material.py::test_geometric_obj_with_user_material 3.896e-04
#   test_user_defined_material.py::test_epsilon_func                    1.156e-03
#   test_user_defined_material.py::test_geometric_obj_with_epsilon_func 1.156e-03
#   test_simulation.py::test_epsilon_input_file                         2.896e-05
#   test_simulation.py::test_numpy_epsilon                              2.896e-05
#
# Those look large against this package's 1e-07 floor and none of it is the route's
# doing: the cells ring down (the control's own peak falls from 1.044e-01 at t=50 to
# 1.088e-03 at t=200), so a peak-normalized Linf inflates with the stopping time. The
# SAME cell spelled as ordinary geometry — a route the gate has always accepted —
# measures 2.252e-06 / 2.078e-05 / 1.557e-04 / 2.395e-03 at t = 20 / 50 / 100 / 200,
# i.e. 6x WORSE at t=200 than the material_function route on the identical structure.
# The point read that MEEP's own assertions use is the number that matters, and it
# holds to MEEP's own published places — see `test_from_meep`'s route cases.

EPSILON_ROUTE_CALLABLE = "callable"  # material_function / epsilon_func / a callable material.
EPSILON_ROUTE_ARRAY = "array"        # a NumPy permittivity volume as the material.
EPSILON_ROUTE_FILE = "file"          # epsilon_input_file.

# How densely a material callable is sampled by the pre-flight probe, per live axis.
# The probe is a KIND check (is what this returns permittivity and nothing else?),
# not a bound, so it is capped rather than run at MEEP's own resolution: the exact
# statement about the built structure is `_require_epsilon_only_structure`, which
# reads every point and costs two array reads.
_MATERIAL_PROBE_SAMPLES = {1: 4096, 2: 64, 3: 16}
# The frequency the rasterized structure is re-read at. Any nonzero value separates a
# dispersive or lossy chi1inv from an instantaneous one (monitor.cpp:263-355); 1.0 is
# in the middle of the band MEEP's own units make natural, and a pole sitting exactly
# on it makes the answer diverge, which this check reads as a difference either way.
_MATERIAL_PROBE_FREQUENCY = 1.0
_EPSILON_ROUTE_RESPONSE_TOLERANCE = 1e-6


def _epsilon_route_kind(material) -> str | None:
    """Which per-point epsilon route this declared material is, if it is one at all.

    The classification is MEEP's own, not a guess: ``pymaterial_to_material``
    (python/typemap_utils.cpp:560-621) accepts exactly five things in a material slot
    and says so in its own abort — "Expected a Medium, a Material Grid, a function, or
    a filename". Of those, a CALLABLE becomes ``make_user_material``, a STRING becomes
    ``make_file_material`` (which is what ``epsilon_input_file`` is: a filename in the
    material slot, python/simulation.py:2023-2024) and an ndarray becomes a
    ``MATERIAL_FILE`` carrying ``epsilon_data``. Those three are the per-point routes.

    Returns ``None`` for anything that names a medium — an ``mp.Medium``, an
    ``mp.MaterialGrid``, or something unrecognized that the material-kind refusal in
    :func:`_check_materials` should still catch by name.
    """
    if material is None or isinstance(material, str):
        # A string here is `epsilon_input_file` (the only string MEEP's own
        # `_init_structure` ever puts in the material slot); `None` is a geometry
        # object with no material, which names no medium either.
        return EPSILON_ROUTE_FILE if isinstance(material, str) and material else None
    if isinstance(material, numpy.ndarray):
        return EPSILON_ROUTE_ARRAY
    if callable(material):
        return EPSILON_ROUTE_CALLABLE
    return None


def _epsilon_routes(mp, sim) -> tuple[str, ...]:
    """Every per-point epsilon route this simulation declares, deduplicated.

    The one question three separate decisions ask: whether the gate has a route to
    validate, whether the lift must take the structured path even though the declared
    media agree, and whether the declared-media span is a bound on the sampled
    permittivity (it is not — there is no declaration to bound it with).
    """
    kinds = []
    for material in _declared_materials(sim):
        if _is_material_grid(mp, material):
            continue
        kind = _epsilon_route_kind(material)
        if kind is not None and kind not in kinds:
            kinds.append(kind)
    return tuple(kinds)


def _material_probe_points(mp, sim):
    """Cell-interior sample points for :func:`_probe_material_callable`.

    Cell centres of a coarse lattice over the declared cell, with a hard zero on every
    axis this run makes invariant — MEEP drops those coordinates before a material
    function ever sees them (``py_v3_to_vec``), so sampling them would ask the callable
    about a coordinate that does not exist.
    """
    dimensions = _effective_dimensions(mp, sim)
    invariant = _invariant_axes(dimensions)
    live = [axis for axis in range(3) if axis not in invariant]
    per_axis = _MATERIAL_PROBE_SAMPLES.get(len(live), 16)
    extents = _vector3(sim.cell_size)
    axis_values = []
    for axis in range(3):
        if axis in invariant:
            axis_values.append([0.0])
            continue
        length = float(extents[axis])
        if length == 0.0:
            axis_values.append([0.0])
            continue
        axis_values.append([
            -0.5 * length + (index + 0.5) * length / per_axis for index in range(per_axis)
        ])
    return [mp.Vector3(x, y, z)
            for x in axis_values[0] for y in axis_values[1] for z in axis_values[2]]


def _probe_material_callable(mp, sim, material, label: str) -> list[str]:
    """Sample a material callable across the cell; report anything but pure permittivity.

    MEEP accepts two shapes and this accepts both, because it cannot tell them apart
    before ``init_sim``: a ``material_function`` returns an ``mp.Medium``, an
    ``epsilon_func`` returns a number, and the ``.eps`` flag that distinguishes them
    (python/simulation.py:2017-2021, python/geom.py:1095-1098) is set on the
    Simulation-level spelling only when the structure is built. The return value is
    inspected instead.

    Returns the refusal sentences, empty when every sample is permittivity alone.
    """
    reasons: list[str] = []
    registered = _registered_susceptibility_keys(mp, sim)
    for point in _material_probe_points(mp, sim):
        try:
            value = material(point)
        except BaseException as exc:  # noqa: BLE001 — the caller's own function, reported as theirs.
            reasons.append(
                f"{label} raised {type(exc).__name__}: {exc} when this gate sampled it at "
                f"{tuple(_vector3(point))}. MEEP evaluates it at every quadrature point of "
                f"the cell, so a point inside the cell it cannot answer is not liftable."
            )
            return reasons
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            epsilon = (float(value),) * 3
        elif _is_medium(mp, value):
            epsilon = _vector3(value.epsilon_diag)
            problem = _epsilon_only_problem(value, registered=registered)
            if problem:
                reasons.append(
                    f"{label} returns a medium carrying {problem} at "
                    f"{tuple(_vector3(point))}. A per-point material route reaches this "
                    f"engine as MEEP's rasterized chi1inv and nothing else, so only the "
                    f"instantaneous permittivity of what it returns would be stepped — "
                    f"the rest would silently vanish. Declare that material as geometry, "
                    f"where the lift reads it."
                )
                return reasons
        else:
            reasons.append(
                f"{label} returned {type(value).__name__} at {tuple(_vector3(point))}; MEEP's "
                f"material routes return an mp.Medium (material_function) or a number "
                f"(epsilon_func)."
            )
            return reasons
        for axis, entry in enumerate(epsilon):
            if not math.isfinite(entry) or entry <= 0.0:
                reasons.append(
                    f"{label} returned epsilon[{AXIS_NAMES[axis]}]={entry!r} at "
                    f"{tuple(_vector3(point))}; a non-positive instantaneous permittivity has "
                    f"no stable leapfrog."
                )
                return reasons
    return reasons


def _registered_susceptibility_keys(mp, sim) -> frozenset:
    """The susceptibility poles MEEP will actually BUILD A CHAIN ENTRY FOR.

    ``geom_epsilon::add_susceptibilities`` (meepgeom.cpp:1816-1830) discovers poles by
    walking ``geometry``, ``extra_materials`` and ``default_material``, each behind an
    ``is_medium(...)`` test. A ``MATERIAL_USER`` callable is not a medium, so a pole a
    callable RETURNS never reaches that list — and ``geom_epsilon::sigma_row``
    (meepgeom.cpp:1682-1704) then has nothing to match it against, because it compares
    the point's medium terms against ``current_pol->user_s``, an entry of exactly that
    list. The pole vanishes from the structure entirely.

    MEASURED on ``test_material_dispersion.py``'s own callable (eps 2.25, two
    Lorentzians), MEEP 1.33.0 single: ``sim.get_epsilon(frequency=0)`` and
    ``get_epsilon(frequency=0.9)`` are both 2.25 with the callable alone; adding the
    same medium to ``extra_materials`` moves eps(0.9) to 3.762491+3.543005e-05j, which
    is bit-identical to declaring that medium outright. So the list below is the exact
    dividing line between a pole that steps and a pole that does not.
    """
    keys = set()
    # `_declared_materials` already walks exactly MEEP's three slots — default_material,
    # every geometry object's material, and extra_materials. The `is_medium` filter is
    # MEEP's and matters: a MaterialGrid is a MATERIAL_GRID, not a MEDIUM
    # (meepgeom.cpp:297-303), so its endpoints' poles are not registered either — which
    # is independently why `_check_material_grids` measured endpoint dispersion having
    # no effect on the structure at all.
    for medium in _declared_materials(sim):
        if not _is_medium(mp, medium):
            continue
        for attribute in ("E_susceptibilities", "H_susceptibilities"):
            for term in getattr(medium, attribute, ()) or ():
                keys.add((attribute, _susceptibility_equivalence_key(term)))
    return frozenset(keys)


def _epsilon_only_problem(medium, registered: frozenset = frozenset()) -> str | None:
    """Name the first material property on ``medium`` that is not its permittivity.

    The predicate a per-point route has to satisfy, and deliberately stricter than the
    one an ordinary structured cell satisfies: a declared medium's susceptibilities can
    be looked up per point in the geometry that declares them (:mod:`~.sigma_lookup`),
    and a callable declares no geometry to look anything up in.

    ``registered`` is :func:`_registered_susceptibility_keys` — the poles MEEP will
    build a chain entry for. A susceptibility the callable returns that is NOT in that
    set is not a problem at all: **MEEP itself drops it**, so refusing it would refuse
    a vanishing that cannot occur. One that IS in the set is refused, and the refusal
    is then about the right thing — MEEP would step that pole per point, and this lift
    has no per-point evaluator for a callable's sigma. The whole-cell backstop
    :func:`_require_epsilon_only_structure` stands behind both readings and is what
    catches an ``extra_materials`` spelling this sample never sees.
    """
    if any(value != 0.0 for value in _vector3(medium.epsilon_offdiag)):
        return "an off-diagonal permittivity"
    if _vector3(medium.mu_diag) != (1.0, 1.0, 1.0) or any(
        value != 0.0 for value in _vector3(medium.mu_offdiag)
    ):
        return "a magnetic permeability"
    for attribute, description in (("E_susceptibilities", "an E susceptibility (dispersion)"),
                                   ("H_susceptibilities",
                                    "an H susceptibility (magnetic dispersion)")):
        for term in getattr(medium, attribute, ()) or ():
            if (attribute, _susceptibility_equivalence_key(term)) in registered:
                return (f"{description} that MEEP DOES register, because a declared material "
                        f"or extra_materials entry carries the same pole")
    for attribute in ("D_conductivity_diag", "D_conductivity_offdiag",
                      "B_conductivity_diag", "B_conductivity_offdiag"):
        if any(value != 0.0 for value in _vector3(getattr(medium, attribute))):
            return f"a {attribute.rsplit('_', 1)[0]}"
    for attribute in ("E_chi2_diag", "E_chi3_diag", "H_chi2_diag", "H_chi3_diag"):
        if any(value != 0.0 for value in _vector3(getattr(medium, attribute))):
            return f"a nonlinearity ({attribute})"
    return None


def _require_epsilon_only_structure(sim, routes: tuple[str, ...]) -> None:
    """After ``init_sim``: prove MEEP's structure carries permittivity and nothing else.

    The exact half of the per-point-route gate, and the reason the pre-flight probe is
    allowed to be a sample. ``structure_chunk::get_chi1inv_at_pt`` (monitor.cpp:263-355)
    answers frequency 0 with the stored instantaneous tensor and any other frequency
    with the same tensor plus every susceptibility's ``chi1(frequency, sigma)`` and a
    ``1 + i*sigma_D/frequency`` conductivity factor. ``sim.get_epsilon(frequency)`` and
    ``sim.get_mu(frequency)`` walk that function over the whole cell in one C++ pass
    (array_slice.cpp:388-403), so two reads apiece decide the question for EVERY point:

    * ``|eps(f) - eps(0)|`` nonzero anywhere means a dispersion or a conductivity
      reached the D side of the structure.
    * ``mu(0) != 1`` means a permeability; ``|mu(f) - mu(0)|`` nonzero means a magnetic
      dispersion or a ``B_conductivity``.

    Only called when a per-point route is present, and it is a statement about the
    WHOLE cell, so a run mixing such a route with declared dispersive geometry is
    refused here rather than being asked to attribute the difference. That refusal is
    the gate's too (:func:`_check_materials`); this is the backstop for a route the
    gate could only sample.

    ``chi2``/``chi3`` are NOT covered — MEEP keeps the nonlinear volumes out of
    ``chi1inv`` and publishes no reader for them, so a nonlinearity returned by a
    material callable is caught by the pre-flight probe or not at all.
    """
    for reader, name, expected in (
        (sim.get_epsilon, "permittivity", None),
        (getattr(sim, "get_mu", None), "permeability", 1.0),
    ):
        if reader is None:
            continue
        instantaneous = numpy.asarray(reader(frequency=0))
        if expected is not None:
            deviation = float(numpy.abs(instantaneous - expected).max())
            if deviation > _EPSILON_ROUTE_RESPONSE_TOLERANCE:
                raise ValueError(
                    f"a per-point epsilon route ({', '.join(routes)}) built a structure whose "
                    f"rasterized {name} departs from {expected:g} by {deviation:.3g}. This lift "
                    f"installs MEEP's chi1inv and nothing else, so that material response would "
                    f"be dropped; nothing was stepped."
                )
        dispersive = numpy.asarray(reader(frequency=_MATERIAL_PROBE_FREQUENCY))
        response = float(numpy.abs(dispersive - instantaneous).max())
        if response > _EPSILON_ROUTE_RESPONSE_TOLERANCE * max(
            1.0, float(numpy.abs(instantaneous).max())
        ):
            raise ValueError(
                f"a per-point epsilon route ({', '.join(routes)}) built a structure whose "
                f"{name} at frequency {_MATERIAL_PROBE_FREQUENCY:g} differs from its "
                f"instantaneous value by {response:.3g} — a susceptibility or a conductivity "
                f"reached the structure (structure_chunk::get_chi1inv_at_pt, monitor.cpp:"
                f"263-355). Only the instantaneous permittivity is lifted through this route, "
                f"so the rest would silently vanish; nothing was stepped."
            )


def _material_grids_of(mp, sim) -> list:  # Every mp.MaterialGrid the cell declares.
    """The grids themselves, which :func:`_materials_of` has already expanded away."""
    return [material for material in _declared_materials(sim)
            if _is_material_grid(mp, material)]


def _flat_axes(sim) -> tuple[int, ...]:  # Axes the declared cell gives no extent at all.
    """Which axes of ``cell_size`` are exactly zero.

    Reported for the messages, not used to decide the dimensionality: MEEP's own
    rule (:func:`_effective_dimensions`) tests ``cell_size.z`` alone, and a zero x
    or y extent is a ONE-CELL 3-D axis rather than a reduction (``vol3d``:
    ``(xsize == 0) ? 1 : ...``). Confirmed against MEEP, which prints "Working in
    3D dimensions. Computational cell is 0.1 x 0.1 x 4" for ``mp.Vector3(0, 0, 4)``
    at ``dimensions=3``.
    """
    return tuple(
        axis for axis, length in enumerate(_vector3(sim.cell_size)) if float(length) == 0.0
    )


def _effective_dimensions(mp, sim) -> int:
    """The dimensionality MEEP will actually build, before ``init_sim`` is called.

    A transcription of ``Simulation._infer_dimensions`` (python/simulation.py),
    which is the only rule that decides it::

        if self.dimensions == 3:
            zero_z = self.cell_size.z == 0
            return 2 if zero_z and (not k or self.special_kz or k.z == 0) else 3
        elif self.dimensions == 2 and self.is_cylindrical:
            return mp.CYLINDRICAL
        return self.dimensions

    Three things in it are easy to get wrong from the outside, and each was
    checked against MEEP rather than assumed:

    * **Only ``cell_size.z`` collapses.** A zero x or y extent is not a reduction
      at all — ``vol3d`` turns it into a ONE-CELL axis of length ``dx`` and MEEP
      says "Working in 3D dimensions" — so a rule that tested every axis (as this
      function used to) reports 1-D for a cell MEEP builds in 3-D.
    * **``dimensions`` wins outright when it is 1 or 2.** ``dimensions=2`` with
      ``cell_size=mp.Vector3(2, 3, 4)`` is a 2-D run whose z extent MEEP simply
      never reads: measured 0.0e+00 against the same script on a ``(2, 3, 0)``
      cell. Taking ``min(declared, from_cell)`` — as this function used to — gets
      that case right by accident and ``dimensions=1`` wrong, since MEEP's
      ``vol1d`` reads ``cell_size.z`` and ignores x and y.
    * **A nonzero ``k_point.z`` blocks the collapse only when ``kz_2d="3d"``.** That
      is the case this returns 3 for: ``mp.Vector3(2, 3, 0)`` with
      ``k_point=mp.Vector3(0, 0, 0.1)`` and ``kz_2d="3d"`` is a 3-D run with a
      one-cell z axis carrying a Bloch phase. It is NOT the default. ``kz_2d``
      defaults to ``"complex"``, and MEEP's constructor reads it the moment
      ``cell_size.z == 0 and k_point.z != 0``, setting ``special_kz = True`` (and
      forcing complex fields). ``special_kz`` is itself a term in ``use_2d``'s
      disjunction, so the default spelling collapses to 2-D after all — MEEP prints
      "Working in 2D dimensions" and hands back a 2-D epsilon array. This function
      reproduces ``use_2d`` term for term; the ``special_kz`` mode it implies (a 2-D
      grid carrying the out-of-plane kz as an analytic phase) is lifted as the
      driver's ``beta``, NOT as a Bloch component — see :func:`_lift_k_point`, which
      zeroes z exactly as MEEP's ``use_bloch(Vector3(kx, ky))`` does.
    """
    if sim.is_cylindrical or sim.dimensions == getattr(mp, "CYLINDRICAL", -2):
        return getattr(mp, "CYLINDRICAL", -2)
    declared = int(sim.dimensions)
    if declared != 3:
        return declared
    wavevector = sim.k_point
    phased_z = bool(wavevector) and not getattr(sim, "special_kz", False) and (
        float(getattr(wavevector, "z", 0.0)) != 0.0
        if all(hasattr(wavevector, name) for name in ("x", "y", "z"))
        else True  # An unreadable k_point is reported elsewhere; do not collapse on it.
    )
    return 2 if float(_vector3(sim.cell_size)[2]) == 0.0 and not phased_z else 3


def _invariant_axes(dimensions: int) -> tuple[int, ...]:  # Axes a reduced run carries as invariant.
    """The axes MEEP drops for this ``dimensions``, in x, y, z order.

    ``dimensions=0`` maps onto 1 here only to keep this helper total. It is NOT a
    supported spelling, and the comment this replaces claimed the opposite: MEEP's
    ``_create_grid_volume`` does route ``dims == 0 or dims == 1`` to the same
    ``vol1d``, but the routing is where the resemblance ends — ``py_v3_to_vec``
    rejects the 0 inside ``init_sim``, so the script raises
    ``ValueError: Invalid dimensions in Volume: 0`` and never steps on CPU MEEP.
    The compatibility register refuses it by name before anything reaches here.

    ``dimensions == mp.CYLINDRICAL`` (-2, MEEP's token rather than a count) makes
    the phi axis the invariant one: volcyl reads cell_size.x as r and .z as z,
    and the exp(i*m*phi) dependence is analytic — one cell on axis 1, exactly as
    the driver spells it (``invariant = (1,) if cylindrical``).
    """
    if dimensions == MEEP_CYLINDRICAL:
        return (1,)
    return INVARIANT_AXES_BY_DIMENSIONS.get(1 if dimensions == 0 else dimensions, ())


def _unit_cell_axes(sim, dimensions: int) -> tuple[int, ...]:
    """The RESOLVED axes MEEP's grid builds with exactly one cell.

    MEEP treats such a direction differently in two places, and both matter to a
    lift, so the set is named once here rather than recomputed at each:

    * it carries no boundary layer — ``boundary_region::apply`` installs a face only
      when ``user_volume.num_direction(d) > 1`` (structure.cpp:208-218, and
      ``check_ok`` repeats the guard at :240), so ``mp.PML(t)`` skips it whether it
      was asked for implicitly or by name;
    * it is forced PERIODIC whatever ``k_point`` says — ``fields::fields``,
      "unit directions are periodic by default" (fields.cpp:80-85), for every
      direction that has both boundaries and is not ``R``.

    Not a corner case: it is the ``mp.Vector3(0, 0, L)`` idiom several of MEEP's own
    1-D examples are written in without declaring ``dimensions=1``, and the
    ``mp.Vector3(sr, 0, 0)`` idiom every Dcyl example with no z extent is written in
    (``ring-cyl.py``, ``perturbation_theory.py``). An INVARIANT axis is excluded
    because MEEP has no such direction at all — a separate clause of the same guards
    (``has_direction``), and one this converter already handles by itself.
    """
    invariant = _invariant_axes(dimensions)
    resolution = float(sim.resolution)
    return tuple(
        axis for axis, length in enumerate(_reduced_cell_size(sim, dimensions))
        if axis not in invariant and meep_cell_count(length, resolution) == 1
    )


def _reduced_vector(vector, dimensions: int) -> tuple[float, float, float]:
    """An ``mp.Vector3`` with a hard zero on every axis this run makes invariant.

    What a source's centre and size become on the way to the driver (see
    :func:`_lift_source` for why the zero is written rather than passed through), and
    therefore what :func:`_check_source_folds` has to place against the mirror
    planes for the pre-flight verdict and the lift to be about the same source.
    """
    invariant = _invariant_axes(dimensions)
    return tuple(
        0.0 if axis in invariant else value for axis, value in enumerate(_vector3(vector))
    )


def _reduced_cell_size(sim, dimensions: int) -> tuple[float, float, float]:
    """``sim.cell_size`` as the grid MEEP will actually build from it.

    Two rewrites, both transcribed rather than inferred:

    * an INVARIANT axis is forced to exactly 0. MEEP ignores whatever is there
      (``vol2d`` reads x and y, ``vol1d`` reads z), so the zero is written here
      rather than passed through: the driver requires it, which is what stops a
      caller of the ENGINE from declaring a thickness for an infinite direction,
      while a lift of a MEEP script that declared one still reproduces the run MEEP
      would have built. Measured: ``dimensions=2`` on a ``(2, 3, 4)`` cell steps
      0.0e+00 away from the same script on ``(2, 3, 0)``.
    * a RESOLVED axis given zero extent becomes ONE CELL, ``dx`` long, because that
      is what ``vol3d`` does: ``(xsize == 0) ? 1 : (int)(xsize*a + 0.5)``. MEEP
      prints "Working in 3D dimensions. Computational cell is 0.1 x 0.1 x 4" for
      ``mp.Vector3(0, 0, 4)`` at ``dimensions=3``, and its ``nosize_direction``
      then treats that axis as an emulated lower dimension for source placement.
      This is the ``mp.Vector3(0, 0, L)`` idiom several of MEEP's own 1-D examples
      are written in without declaring ``dimensions=1``.
    """
    invariant = _invariant_axes(dimensions)
    return tuple(
        0.0 if axis in invariant
        else (1.0 / float(sim.resolution) if float(length) == 0.0 else float(length))
        for axis, length in enumerate(_vector3(sim.cell_size))
    )


# --- the compatibility register ---------------------------------------------------


def gpu_compatibility(sim) -> GpuCompatibility:
    """Report every feature of ``sim`` this module cannot lift, without building anything.

    Ask before running. Nothing here initializes the simulation, allocates a
    structure, or mutates ``sim`` in any way, so it is cheap enough to call on a
    dispatch path — the epsilon volume is judged from the declared materials
    rather than from a rasterization. :func:`lift_simulation` calls this first and
    then re-checks the rasterized volume independently, so a material route this
    function does not know about is still caught before a single step is taken.

    ONE THING IT DOES CALL: a ``material_function`` / ``epsilon_func`` / callable
    material is SAMPLED here, at a bounded lattice of cell-interior points
    (:data:`_MATERIAL_PROBE_SAMPLES`), to ask whether what it returns is a
    permittivity and nothing else. That is the caller's own function running, so it
    is not free and it is not guaranteed side-effect-free; it is also strictly fewer
    evaluations than MEEP's own ``set_epsilon`` will make of it, and there is no other
    way to answer the question without building the structure.

    Args:
        sim: A constructed ``meep.Simulation``. It need not have been initialized.

    Returns:
        :class:`GpuCompatibility` — ``supported`` plus one sentence per blocker,
        each naming the feature and, where one exists, the workaround.
    """
    mp = _import_meep()
    reasons: list[str] = []
    _check_simulation_shape(mp, sim, reasons)
    _check_boundaries(mp, sim, reasons)
    _check_symmetries(mp, sim, reasons)
    # Placing each source against its mirror planes needs the cell, the
    # resolution and the folds to be readable in the first place; when any of
    # the three checks above objected, whichever of them is unreadable has
    # already been named and there is no second geometry to measure against.
    geometry_is_resolvable = not reasons
    _check_sources(mp, sim, reasons)
    if geometry_is_resolvable:
        _check_source_folds(mp, sim, reasons)
    _check_materials(mp, sim, reasons)
    _check_outputs(sim, reasons)
    return GpuCompatibility(supported=not reasons, reasons=tuple(reasons))


def _check_simulation_shape(mp, sim, reasons: list[str]) -> None:  # Cell, clock, units, boundaries-as-conditions.
    """Dimensionality, cell placement, resolution, Courant, k_point, and MPI."""
    dimensions = _effective_dimensions(mp, sim)
    cylindrical = dimensions == getattr(mp, "CYLINDRICAL", -2) or sim.is_cylindrical
    if cylindrical:
        # Cylindrical IS lifted: MEEP's Dcyl under the x -> r, y -> phi, z -> z
        # mapping (the design notes (fdtd-cylindrical-plan)), validated component by
        # component against CPU MEEP in test_cylindrical.py. What the mode cannot
        # serve is refused here by name, mirroring Grid._resolve_cylindrical.
        m_value = getattr(sim, "m", 0)
        if float(m_value) != float(int(m_value)):
            reasons.append(
                f"m={m_value!r} is not an integer: this engine's phi axis is one cell whose "
                f"wrap phase is exp(2*pi*i*m), single-valued only for integer m. MEEP's "
                f"non-integer m (wedge problems) has no lift."
            )
        if getattr(sim, "accurate_fields_near_cylorigin", False) and abs(int(m_value)) >= 2:
            # The branch itself is implemented (stepping._cylindrical_axis_rows); what is
            # refused is running it OUTSIDE its stability bound. MEEP's own comment on the
            # branch this flag selects puts that bound at ~1/(|m| + 0.5), against ~0.62
            # with the default's zero rows, and MEEP's own test_pml_cyl sets
            # `sim.Courant = 1/(|m| + 0.6)` by hand for exactly this reason. Refused
            # rather than warned because a cylindrical instability is a smooth growing
            # mode: the run completes and hands back something field-shaped.
            limit = 1.0 / (abs(int(m_value)) + 0.5)
            if float(sim.Courant) > limit:
                reasons.append(
                    f"accurate_fields_near_cylorigin=True with m={int(m_value)} needs "
                    f"Courant <= 1/(|m| + 0.5) = {limit:g}, and this run declares "
                    f"{float(sim.Courant):g}. Without the default treatment's zero rows near "
                    f"the axis the near-axis update is unstable above that bound (MEEP's own "
                    f"comment, src/step_db.cpp), and the instability is a smooth growing mode "
                    f"rather than a crash. Lower sim.Courant, or leave the flag unset."
                )
        if tuple(getattr(sim, "symmetries", ()) or ()):
            reasons.append(
                "mirror symmetry in cylindrical coordinates is not implemented: the axis "
                "already carries the exp(i*m*phi) reduction, and a z mirror over Dcyl has "
                "no lift here."
            )
        wavevector = sim.k_point
        if wavevector and all(hasattr(wavevector, name) for name in ("x", "y", "z")):
            k_r, k_p, _ = _vector3(wavevector)
            if k_r != 0.0 or k_p != 0.0:
                reasons.append(
                    f"k_point={_vector3(wavevector)} has a nonzero r or phi component; in "
                    f"Dcyl only a z Bloch phase is meaningful (the r axis ends at the axis "
                    f"and the PEC wall, phi is carried by m)."
                )
    elif dimensions not in (1, 2, 3):
        zero_note = (
            " MEEP cannot build this one either: _create_grid_volume routes dimensions=0 to "
            "vol1d, and py_v3_to_vec then rejects it with 'Invalid dimensions in Volume: 0' "
            "inside init_sim, so the script does not run on CPU MEEP as written. Declare "
            "dimensions=1."
            if dimensions == 0 else ""
        )
        reasons.append(
            f"dimensions={sim.dimensions!r} is not one of MEEP's Cartesian values (1, 2 or 3); "
            f"this engine has no reduction for it.{zero_note}"
        )
    elif dimensions != 3:
        # 1-D and 2-D ARE lifted: the missing axes are carried as translational
        # invariance, which is exactly what MEEP means by them, and the engine reaches
        # it as one periodic cell per invariant axis — MEEP's own lower-dimensional
        # emulation (fields.cpp nosize_direction). Verified against CPU MEEP: a 2-D run
        # and the same physics as a 3-D cell one pixel deep in z step to 0.0e+00 of each
        # other IN MEEP, and this engine reproduces both TE and TM at 1.4e-07…7.7e-07.
        # A reduced run does constrain what else it may declare; the checks are below
        # and in Grid, and each says what it would otherwise have done silently.
        pass
    if not cylindrical and getattr(sim, "m", 0):
        reasons.append(
            f"m={sim.m!r} is the cylindrical azimuthal mode number and has no meaning on a "
            f"Cartesian grid."
        )
    # BFAST — the broadband fixed-angle source technique — IS lifted: the extra
    # additive pass inside step_db (STEP_BFAST, step_db.cpp:129-142) is built as
    # stepping._bfast_term, folded into the curl on the same gathered operands, with
    # one f_bfast IIR state per D and B component. What stays refused is the ONE
    # geometry MEEP's own code cannot express consistently, mirrored here from
    # Grid._resolve_bfast so it fires at the gate rather than as a lift throw.
    #
    # History worth keeping: this was found SILENTLY ACCEPTED on 2026-08-04 —
    # gpu_compatibility returned supported=True with an empty reason list for a
    # bfast_scaled_k run, which would have been stepped with the ordinary update
    # equations and returned a smooth, complete, WRONG field (a fixed-angle run
    # degenerates into a normal-incidence one, whose Fresnel reflectance at the
    # corpus row's 35.7 deg is 0.18367 against BFAST's 0.124120 — 48 % away and
    # perfectly converged).
    bfast = tuple(getattr(sim, "bfast_scaled_k", None) or (0.0, 0.0, 0.0))
    if any(float(value) != 0.0 for value in bfast) and cylindrical:
        reasons.append(
            f"bfast_scaled_k={bfast} is not representable on a cylindrical grid: MEEP's Dcyl "
            f"branch nulls f_p (R) and f_m (Z) by hand at step_db.cpp:87/:96 while have_p and "
            f"have_m stay true, which makes the single-operand branch of step_bfast reachable "
            f"with a nonzero k — and that branch (step_generic.cpp:376) omits the '- F[i]' its "
            f"seven siblings all carry, so the bilinear time derivative degenerates into a bare "
            f"two-point sum. Transcribing an upstream inconsistency nobody has ever run would "
            f"be worse than refusing it. Run the cylindrical case at one frequency with a "
            f"k_point instead."
        )
    centre = _vector3(sim.geometry_center)
    if any(value != 0.0 for value in centre):
        reasons.append(
            f"geometry_center={centre} is not the origin: this engine's grid is always centred "
            f"on (0, 0, 0) (Grid.axis_origin), so every source centre, monitor region and PML "
            f"face would land at the wrong absolute coordinate. Shift the geometry and the "
            f"source centres instead of the cell."
        )
    resolution = sim.resolution
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float, numpy.integer, numpy.floating)):
        reasons.append(
            f"resolution={resolution!r} is not a single number; this engine's grid has one "
            f"resolution for all three axes (Grid.dx = 1 / resolution)."
        )
    elif not math.isfinite(float(resolution)) or float(resolution) <= 0.0:
        reasons.append(f"resolution={resolution!r} must be finite and positive.")
    courant = float(sim.Courant)
    # The CFL bound counts the axes the run differences, so a reduced run gets the
    # looser one its own stability analysis gives it (1/sqrt(2) in 2-D, 1 in 1-D).
    # Holding every run to 1/sqrt(3) would refuse a perfectly stable 2-D script at
    # MEEP's own suggested Courant for one.
    # Dcyl differences r and z — a 2-D stencil — and MEEP's own cylindrical
    # default Courant is 0.5, under the 2-D bound.
    resolved = 2 if cylindrical else (1 if dimensions == 0 else dimensions)
    limit = _COURANT_LIMIT_3D if resolved not in (1, 2, 3) else 1.0 / math.sqrt(resolved)
    if not math.isfinite(courant) or courant <= 0.0 or courant > limit:
        reasons.append(
            f"Courant={courant:g} is outside (0, {limit:.6f}], the {resolved}-D CFL stability "
            f"limit 1/sqrt({resolved}) this engine enforces."
        )
    wavevector = sim.k_point
    if not wavevector:
        # MEEP's default: fields::fields leaves every face Metallic and nothing calls
        # use_bloch (fields.cpp, simulation.py `if self.k_point:`). Lifted as
        # boundaries="metallic", cross-validated at the engine's own floors — this used
        # to be a refusal, and the 1.28e+00 gap it quoted is what the boundary is worth.
        pass
    elif not all(hasattr(wavevector, name) for name in ("x", "y", "z")):
        # MEEP's own py_v3_to_vec reads .x/.y/.z, so a tuple would fail there too —
        # but this function must report rather than raise AttributeError from a
        # pre-flight check the caller ran precisely to avoid a failure.
        reasons.append(
            f"k_point={wavevector!r} is not an mp.Vector3; MEEP reads .x/.y/.z off it."
        )
    else:
        for axis, value in enumerate(_vector3(wavevector)):
            if not math.isfinite(value):
                reasons.append(f"k_point component {AXIS_NAMES[axis]} must be finite, got {value!r}.")
    if mp.count_processors() > 1:
        reasons.append(
            f"this MEEP process is one of {mp.count_processors()} MPI ranks; the GPU stepper is "
            f"single-process and would step the whole cell on every rank. Run the lifting process "
            f"without mpirun."
        )
    if getattr(sim, "fields", None) is not None:
        if sim.round_time() > 0.0:
            reasons.append(
                f"this simulation has already been stepped to t={sim.round_time():g}; the lift "
                f"starts from zero fields and would silently discard that history. Lift a freshly "
                f"constructed Simulation, or call sim.reset_meep() first."
            )
        reasons.append(
            "this simulation has already been initialized (sim.fields exists), and everything a "
            "caller can do to an initialized MEEP simulation is INVISIBLE to this converter. "
            "sim.set_boundary(side, direction, cond) and sim.fields.use_bloch(direction, k) both "
            "rewrite fields.boundaries per FACE, which MEEP exposes only as an opaque pointer, so "
            "no reader here can tell which condition each of the six faces ended up with — and "
            "this engine now steps both of them, which makes the ambiguity worse rather than "
            "better: it would confidently run whichever one sim.k_point implies. The two differ "
            "by 1.28e+00 complex relative L2 on the same cell. sim.fields.set_epsilon, "
            "load_structure and initialize_field are equally unreadable. Lift a freshly "
            "constructed Simulation, or call sim.reset_meep() first (it rebuilds from the "
            "declared arguments, which ARE readable); a whole-cell metallic box needs nothing "
            "special — just leave k_point unset, which is MEEP's default."
        )


def _check_boundaries(mp, sim, reasons: list[str]) -> None:  # mp.PML layers -> the driver's per-face table.
    """Every boundary layer must be a default-profile ``mp.PML``, at any thickness.

    The thickness itself is no longer constrained: a fractional cell count is what
    MEEP builds and what ``meep_gpu.pml`` now builds (:func:`~.pml.half_cell_extent`).
    What is still refused by name is a layer this engine's UPML has no term for — an
    ``mp.Absorber``, a non-default ``R_asymptotic`` or ``mean_stretch``, a custom
    profile — or a Dcyl face that is not a face.
    """
    claimed: dict[tuple[int, int], float] = {}
    for layer in getattr(sim, "boundary_layers", ()) or ():
        label = f"{type(layer).__name__}(thickness={getattr(layer, 'thickness', '?')!r})"
        if isinstance(layer, mp.Absorber):
            # An Absorber IS an mp.PML by inheritance (python/simulation.py:308), so this
            # branch must come first and must consume the layer — everything below builds
            # the UPML face table, and an absorber installs no face at all.
            _check_absorber_layer(mp, sim, layer, label, reasons)
            continue
        if not isinstance(layer, mp.PML):
            reasons.append(f"boundary layer {layer!r} is not an mp.PML.")
            continue
        if float(getattr(layer, "R_asymptotic", _PML_DEFAULT_R_ASYMPTOTIC)) != _PML_DEFAULT_R_ASYMPTOTIC:
            reasons.append(
                f"{label} sets R_asymptotic={layer.R_asymptotic!r}; meep_gpu.pml.PML is built at "
                f"MEEP's default {_PML_DEFAULT_R_ASYMPTOTIC:g} and FdtdDriver.setup_pml does not "
                f"expose it, so the grading prefactor -ln(R)/(4*dx*integral) would be wrong. "
                f"Construct meep_gpu.pml.PML(R_asymptotic=...) by hand on a driver instead."
            )
        if float(getattr(layer, "mean_stretch", _PML_DEFAULT_MEAN_STRETCH)) != _PML_DEFAULT_MEAN_STRETCH:
            reasons.append(
                f"{label} sets mean_stretch={layer.mean_stretch!r}; this engine's UPML fixes "
                f"kappa = 1 (sigma-only grading, no coordinate stretch) and carries no term for it."
            )
        if not _is_quadratic_profile(layer):
            reasons.append(
                f"{label} carries a custom pml_profile; this engine grades sigma as u**order with "
                f"an integer order (pml.PML.order) and cannot evaluate an arbitrary callable per "
                f"cell. MEEP's default quadratic profile is the only one lifted — for u**3 or "
                f"u**4, call driver.setup_pml(cells, order=3) yourself."
            )
        thickness = float(layer.thickness)
        if not math.isfinite(thickness) or thickness <= 0.0:
            reasons.append(f"{label} must have a finite positive thickness.")
            continue
        # A thickness that is not a whole number of cells used to be refused here. It is
        # not a limitation of MEEP's and it is no longer one of this engine's: MEEP
        # snaps only the layer's EXTENT, to the nearest half cell, and normalizes the
        # profile by the raw thickness (structure.cpp pml_x / use_pml), which
        # meep_gpu.pml reproduces. `mp.PML(0.5)` at resolution 71 — 35.5 cells, three
        # corpus scripts' spelling — now lifts and steps.
        cylindrical = _is_cylindrical(sim)
        declared = int(getattr(layer, "direction", MEEP_ALL))
        if cylindrical:
            # Dcyl faces: mp.R maps to this engine's axis 0 (high side only — the
            # low side is the axis, not a boundary; vec.cpp has_boundary), mp.Z to
            # axis 2, and an undirected layer covers exactly those three faces.
            allowed = {MEEP_R: 0, 2: 2}
            directions = (0, 2) if declared == MEEP_ALL else (allowed.get(declared, -1),)
            if declared not in (MEEP_ALL, MEEP_R, 2):
                reasons.append(
                    f"{label} names direction={layer.direction!r}, which is not mp.R or mp.Z — "
                    f"the only faces a Dcyl cell has (phi is carried by m, not by cells)."
                )
                continue
        else:
            directions = (0, 1, 2) if declared == MEEP_ALL else (declared,)
        for direction in directions:
            if direction not in (0, 1, 2):
                reasons.append(
                    f"{label} names direction={layer.direction!r}, which is not mp.X, mp.Y or mp.Z."
                )
                continue
            side = int(getattr(layer, "side", MEEP_ALL))
            sides = (MEEP_SIDE_LOW, MEEP_SIDE_HIGH) if side == MEEP_ALL else (side,)
            if cylindrical and direction == 0:
                if side == MEEP_ALL:
                    sides = (MEEP_SIDE_HIGH,)  # The axis is not a boundary; nothing to absorb there.
                elif side == MEEP_SIDE_LOW:
                    reasons.append(
                        f"{label} asks for a PML on the LOW side of r — the axis. That is not a "
                        f"boundary (vec.cpp has_boundary excludes it in Dcyl); an absorber there "
                        f"would eat the on-axis field."
                    )
                    continue
            for one_side in sides:
                if one_side not in (MEEP_SIDE_LOW, MEEP_SIDE_HIGH):
                    reasons.append(f"{label} names side={layer.side!r}, which is not mp.Low or mp.High.")
                    continue
                key = (direction, one_side)
                if key in claimed and claimed[key] != thickness:
                    reasons.append(
                        f"two boundary layers claim the {AXIS_NAMES[direction]} "
                        f"{'low' if one_side == MEEP_SIDE_LOW else 'high'} face with different "
                        f"thicknesses ({claimed[key]!r} and {thickness!r}); this engine holds one "
                        f"thickness per face."
                    )
                claimed[key] = thickness


def _check_absorber_layer(mp, sim, layer, label: str, reasons: list[str]) -> None:  # mp.Absorber gate.
    """What an ``mp.Absorber`` still may not be. It is no longer refused for being one.

    An absorber is a graded scalar conductivity on D **and** B, not a matched layer
    (python/simulation.py:316), and :mod:`~.absorber` now builds exactly that — six
    per-component sigma volumes, bit-identical to MEEP's in 1-D and 2-D. What remains
    refused:

    * A **custom** ``pml_profile``. The profile enters twice — as ``P(u)`` in the table
      and through ``adaptive_integration(P)`` in ``prefac`` (meepgeom.cpp:755-762) — and
      :data:`~.absorber.QUADRATIC_PROFILE_INTEGRAL` hardcodes the quadratic's 1/3.
    * A **cylindrical** cell. MEEP's ``LOOP_OVER_DIRECTIONS(Dcyl, d)`` covers R and Z,
      including a Z axis holding a single cell — which is where every Dcyl script's
      ``mp.Vector3(sr, 0, 0)`` lands — and there the whole axis would be graded from a
      ``geometry_edge`` of half a pixel. That is a different absorber from the one the
      Cartesian faces get, and nothing here has measured it.
    * A face on the LOW side of a mirror-folded axis, for the reason the PML path names:
      cell 0 is the mirror plane, a boundary condition rather than a place to grade.

    ``mean_stretch`` is deliberately NOT refused even at a non-default value: MEEP's
    absorber path never reads it. ``set_cond_profile`` takes ``(L, dx, P, data, R)``
    and the boundary region an ``Absorber`` would have joined is skipped outright, so
    MEEP itself ignores the argument. ``R_asymptotic`` IS honoured — it is one factor
    in ``prefac`` and :func:`~.absorber.absorber_profile` takes it — unlike the PML
    path, which has no place to put it.
    """
    if _is_cylindrical(sim):
        reasons.append(
            f"{label} is an mp.Absorber on a cylindrical (Dcyl) cell. MEEP grades every "
            f"direction of the grid volume, which in Dcyl includes a z axis that most "
            f"scripts give a single cell; the resulting absorber is not the Cartesian one "
            f"and this engine has not measured it. Use mp.PML, or run the case on CPU MEEP."
        )
        return
    if not _is_quadratic_profile(layer):
        reasons.append(
            f"{label} carries a custom pml_profile. MEEP feeds it to the profile table AND "
            f"to the adaptive integration behind prefac = -ln(R)/(4*L*integral) "
            f"(meepgeom.cpp:744-763); meep_gpu.absorber hardcodes the quadratic profile's "
            f"integral of 1/3 and would grade a different layer entirely."
        )
    thickness = float(getattr(layer, "thickness", 0.0))
    if not math.isfinite(thickness) or thickness <= 0.0:
        reasons.append(f"{label} must have a finite positive thickness.")
        return
    # MEEP's table is N + 1 entries with N = int(L/dx + 0.5) at dx = HALF a pixel, and
    # its lookup divides by N. Below half a half-pixel N is 0 and MEEP divides by zero;
    # refused here by name rather than raised from inside the profile builder mid-lift.
    if int(thickness * 2.0 * float(sim.resolution) + 0.5) < 1:
        reasons.append(
            f"{label} spans {thickness * 2.0 * float(sim.resolution):.4g} half-pixels at "
            f"resolution {sim.resolution!r}, which MEEP tabulates as N = 0 — and its own "
            f"lookup then divides by N (meepgeom.cpp:1601). Use a thicker layer."
        )
        return
    dimensions = _effective_dimensions(mp, sim)
    declared = int(getattr(layer, "direction", MEEP_ALL))
    if declared != MEEP_ALL and declared not in (0, 1, 2):
        reasons.append(
            f"{label} names direction={layer.direction!r}, which is not mp.X, mp.Y or mp.Z."
        )
        return
    side = int(getattr(layer, "side", MEEP_ALL))
    if side != MEEP_ALL and side not in (MEEP_SIDE_LOW, MEEP_SIDE_HIGH):
        reasons.append(f"{label} names side={layer.side!r}, which is not mp.Low or mp.High.")
        return
    if side == MEEP_SIDE_LOW:
        folded_axes = _folded_axes(mp, sim, dimensions)
        axes = (0, 1, 2) if declared == MEEP_ALL else (declared,)
        for axis in axes:
            if axis in folded_axes and axis not in _invariant_axes(dimensions):
                reasons.append(
                    f"{label} asks for an absorber on the LOW side of the mirror-folded "
                    f"{AXIS_NAMES[axis]} axis. Cell 0 of a folded axis straddles the mirror "
                    f"plane — a boundary condition, not a place to grade a conductivity — and "
                    f"the stored chunk holds only the upper half, so the layer MEEP steps "
                    f"there is the IMAGE of the high-side one, not a second absorber."
                )


def _folded_axes(mp, sim, dimensions: int) -> tuple[int, ...]:  # Axes an mp.Mirror halves.
    """Which axes ``sim.symmetries`` folds, read before any grid exists.

    The gate runs before a driver is built, so it cannot ask ``grid.is_mirrored``.
    ``mp.Mirror.direction`` is the axis index directly (mp.X = 0), and a symmetry on an
    invariant axis reaches nothing — the caller filters those.
    """
    del dimensions
    axes = []
    for symmetry in getattr(sim, "symmetries", ()) or ():
        if type(symmetry).__name__ != "Mirror":
            continue
        direction = int(getattr(symmetry, "direction", -1))
        if direction in (0, 1, 2):
            axes.append(direction)
    return tuple(sorted(set(axes)))


def _resolve_absorber_layers(mp, sim, dimensions: int) -> tuple[AbsorberLayer, ...]:
    """``sim.boundary_layers`` -> the per-face ``AbsorberLayer`` list the driver installs.

    Face resolution follows MEEP's own loop (meepgeom.cpp:2017-2029): one profile per
    (direction, side) that the layer names, over the directions the GRID VOLUME has.
    Two differences from :func:`_resolve_pml_faces` are deliberate and both come from
    MEEP:

    * **No one-cell-axis drop.** ``boundary_region::apply`` installs a PML face only
      where ``num_direction(d) > 1`` (structure.cpp:208-218); the absorber never goes
      through ``boundary_region`` at all, so that guard does not exist for it. On a
      Cartesian cell a resolved axis with one cell cannot arise from a nonzero extent,
      and the cylindrical case where it does is refused by the gate.
    * **No folded-axis low-face drop.** The sigma is evaluated at each component's real
      coordinate against the FULL cell's ``geometry_edge``, so a folded axis — whose
      stored cells start one cell below the mirror plane — simply never satisfies the
      low-side test. Nothing has to be removed by hand; the gate refuses an explicit
      low-side request so that a user WRITING one is told, but an ordinary undirected
      ``mp.Absorber(t)`` beside an ``mp.Mirror`` resolves to the high face alone by
      arithmetic.

    MEEP's ``boundary_side`` enum is ``High = 0, Low = 1``. Read the intuitive way it
    puts the layer on the opposite face, which absorbs perfectly stably against a
    boundary the user did not name.
    """
    invariant = set(_invariant_axes(dimensions))
    layers: list[AbsorberLayer] = []
    for layer in getattr(sim, "boundary_layers", ()) or ():
        if not isinstance(layer, mp.Absorber):
            continue
        declared = int(getattr(layer, "direction", MEEP_ALL))
        axes = (0, 1, 2) if declared == MEEP_ALL else (declared,)
        side = int(getattr(layer, "side", MEEP_ALL))
        sides = (MEEP_SIDE_LOW, MEEP_SIDE_HIGH) if side == MEEP_ALL else (side,)
        for axis in axes:
            if axis in invariant:
                continue  # MEEP loops over the grid volume's directions; this is not one.
            for one_side in sides:
                layers.append(AbsorberLayer(
                    thickness=float(layer.thickness),
                    axis=axis,
                    side="low" if one_side == MEEP_SIDE_LOW else "high",
                    R_asymptotic=float(getattr(layer, "R_asymptotic", _PML_DEFAULT_R_ASYMPTOTIC)),
                ))
    return tuple(layers)


def _lift_absorber(mp, sim, driver, dimensions: int = 3) -> None:  # The resolved faces -> set_absorber.
    """Install every ``mp.Absorber`` face on the driver, invariant axes dropped.

    Runs BEFORE :func:`_lift_pml` so that a mixed run — MEEP allows PML on some faces
    and an absorber on others — has its conductivity in place when ``setup_pml``
    allocates the ``f_cond`` histories that the conductive PML recurrence needs.
    """
    layers = _resolve_absorber_layers(mp, sim, dimensions)
    if not layers:
        return
    folded = tuple(driver.grid.is_mirrored(axis) for axis in range(3))
    kept = tuple(
        layer for layer in layers
        if not (folded[layer.axis] and layer.side == "low")
    )
    driver.set_absorber(kept)


def _is_quadratic_profile(layer) -> bool:  # Is this MEEP's default pml_profile, lambda u: u*u?
    """Evaluate the profile rather than compare the callable.

    MEEP stores the profile as a plain Python function, so identity against the
    module-level default breaks the moment a user writes the same lambda out by
    hand — and comparing code objects would accept a different function that
    happens to compile alike. Sampling the actual mapping is what the grading
    depends on, so it is what gets checked.
    """
    profile = getattr(layer, "pml_profile", None)
    if profile is None:
        return True  # No profile attribute at all: MEEP's built-in quadratic.
    try:
        values = [complex(profile(u)) for u in _PML_PROFILE_PROBE]
    except Exception:  # noqa: BLE001 - any failure means we cannot certify the profile.
        return False
    return all(
        value.imag == 0.0 and abs(value.real - u * u) <= _PML_PROFILE_TOLERANCE
        for u, value in zip(_PML_PROFILE_PROBE, values)
    )


def _check_symmetries(mp, sim, reasons: list[str]) -> None:  # mp.Mirror -> meep_gpu.Mirror.
    for symmetry in getattr(sim, "symmetries", ()) or ():
        if not isinstance(symmetry, mp.Mirror):
            reasons.append(
                f"symmetry {type(symmetry).__name__} is not implemented: this engine folds mirror "
                f"planes only (meep_gpu.Mirror, i.e. mp.Mirror). Rotational symmetry maps a cell "
                f"onto itself through a transform the Yee lattice does not preserve axis by axis, "
                f"so it is a different fold, not a variant of this one. Drop it (the run is still "
                f"correct, just larger) or keep the job on CPU MEEP."
            )
            continue
        direction = int(symmetry.direction)
        if direction not in (0, 1, 2):
            reasons.append(
                f"mp.Mirror(direction={symmetry.direction!r}) is not normal to mp.X, mp.Y or mp.Z."
            )
            continue
        phase = complex(symmetry.phase)
        if phase.imag != 0.0 or phase.real not in (1.0, -1.0):
            reasons.append(
                f"mp.Mirror(direction={AXIS_NAMES[direction]}, phase={symmetry.phase!r}) has a "
                f"phase that is not +1 or -1; a mirror plane is an involution, so only those two "
                f"reproduce the field after two reflections."
            )
        if direction in _invariant_axes(_effective_dimensions(mp, sim)):
            reasons.append(
                f"mp.Mirror(direction={AXIS_NAMES[direction]}) folds an axis this "
                f"{_effective_dimensions(mp, sim)}-D run makes translationally invariant. A fold "
                f"reflects one half of an axis onto the other and an invariant axis has no "
                f"halves — MEEP steps such a run while printing 'WARNING vol mismatch' and "
                f"double-counting its own loop volume, which is not a result to reproduce. Fold "
                f"one of the axes the run resolves, or drop the plane (the answer is the same, "
                f"the run is just larger)."
            )
        # k on the FOLDED axis. This is the one mirror/Bloch pairing still refused, and
        # it belongs at the gate rather than as a throw out of FdtdDriver's constructor
        # halfway through the lift. A phase on an axis this run does NOT fold is served
        # (measured at 2.8e-06 against CPU MEEP; see
        # `FdtdDriver._require_bloch_is_representable`), so the test is per axis.
        wavevector = getattr(sim, "k_point", None)
        if wavevector and all(hasattr(wavevector, name) for name in ("x", "y", "z")):
            component = _vector3(wavevector)[direction]
            if math.isfinite(component) and component != 0.0:
                reasons.append(
                    f"k_point.{AXIS_NAMES[direction].lower()}={component!r} is nonzero on the axis "
                    f"mp.Mirror(direction={AXIS_NAMES[direction]}) folds. In the Brillouin-zone "
                    f"interior a mirror plane forces the field to be even or odd about itself, "
                    f"which only k = 0 satisfies. The zone EDGE is compatible (MEEP #3155 adds "
                    f"tests/symmetry.cpp::test_periodic_mirror_zone_edge for exactly that) but "
                    f"needs the lattice phase taken over the full, unfolded cell and is not "
                    f"implemented here. Drop this plane — the answer is the same, the run is just "
                    f"larger — or zero that k component."
                )


def _check_sources(mp, sim, reasons: list[str]) -> None:  # mp.Source -> the driver's source dict.
    sources = list(getattr(sim, "sources", ()) or ())
    if not sources:
        reasons.append(
            "this simulation declares no sources, so the lifted run would step zero fields "
            "forever and return an all-zero volume — which reads as a completed run rather than "
            "as an empty one. Add a source, or step FdtdDriver directly if you meant to seed the "
            "fields yourself (FdtdDriver.set_field)."
        )
    cylindrical = _is_cylindrical(sim)
    names = _component_names(mp, cylindrical)
    for index, source in enumerate(sources):
        label = f"sources[{index}]"
        if type(source) is not mp.Source:
            handlers = REALIZABLE_SOURCES.get(type(source).__name__)
            if handlers is not None:
                # Not refused: MEEP's synthesis is a setup-time solve (an MPB mode
                # solve, or an analytic beam evaluation) plus four plain
                # add_volume_source deposits, and the lift re-runs exactly that pair,
                # so only the variants that construction cannot express are refused —
                # per variant, with the alternative named.
                handlers[0](mp, sim, source, label, reasons)
                _check_source_time(mp, source.src, label, reasons)
                continue
            reasons.append(
                f"{label} is an {type(source).__name__}, not a plain mp.Source. "
                + _SPECIAL_SOURCE_NOTES.get(
                    type(source).__name__,
                    "Only mp.Source is lifted; anything that synthesizes its own current "
                    "distribution at setup belongs on CPU MEEP.",
                )
            )
            continue
        if source.component not in names:
            family = "mp.Er/Ep/Ez/Hr/Hp/Hz" if cylindrical else "mp.Ex/Ey/Ez/Hx/Hy/Hz"
            reasons.append(
                f"{label} drives component {source.component!r}, which is not one of "
                f"{family} — the only currents this coordinate system injects."
            )
        _check_source_profile(mp, sim, source, label, reasons)
        _check_source_time(mp, source.src, label, reasons)


def _source_profile_kind(source) -> str | None:  # MEEP's own precedence, not ours.
    """Which spatial profile MEEP would actually use for this source.

    ``Source.add_source`` (python/source.py:145-159) is an if/elif chain:
    ``amp_func_file`` wins, then ``amp_func``, then ``amp_data``, and the losers are
    never looked at again. Declaring two is not an error in MEEP and it is not one
    here — reading them in a different order, or refusing a combination MEEP accepts,
    would be the engine disagreeing with the code it lifts.
    """
    if getattr(source, "amp_func_file", ""):
        return "amp_func_file"
    if getattr(source, "amp_func", None) is not None:
        return "amp_func"
    if getattr(source, "amp_data", None) is not None:
        return "amp_data"
    return None


def _check_source_profile(mp, sim, source, label: str, reasons: list[str]) -> None:
    """The spatial profile of one ``mp.Source``: a callable, an array, or an HDF5 file.

    ``amp_data`` is no longer refused. MEEP does not treat it as a second kind of
    source — ``add_volume_source`` copies the array into two static buffers and calls
    the ordinary amp_func overload with ``amp_file_func`` (src/sources.cpp:394-417),
    so the array route IS the callable route with MEEP's own interpolator in front of
    it. That interpolator is transcribed in :mod:`~.amp_interpolation` and measured
    bit-identical to MEEP's on ``test_source.py``'s own construction (difference
    exactly 0.0; see that module for the table).

    ``amp_func_file`` stays refused, and for a packaging reason rather than a
    numerical one: it is the same array one HDF5 read earlier
    (src/sources.cpp:421-460), and this package carries no HDF5 dependency at all
    (``test_package_boundary`` enforces it). Read the ``.re``/``.im`` dataset pair
    yourself and pass ``amp_data`` — measured 1.164e-10 apart on MEEP's own fixture,
    which is the file's float32 storage and not the arithmetic.
    """
    kind = _source_profile_kind(source)
    if kind is None:
        return
    if kind == "amp_func_file":
        reasons.append(
            f"{label} sets amp_func_file={source.amp_func_file!r}; reading MEEP's HDF5 "
            f"format is a host-application concern (the package deliberately has no h5py "
            f"dependency). MEEP itself only uses the file to build an amp_data array "
            f"(sources.cpp:421-460), so load the '<dataset>.re' / '<dataset>.im' pair "
            f"yourself and pass amp_data=<complex array> — that route IS lifted, through "
            f"MEEP's own interpolator (meep_gpu.amp_interpolation)."
        )
        return
    if all(extent == 0.0 for extent in _vector3(source.size)):
        reasons.append(
            f"{label} has an {kind} on a zero-size (point) source; a spatial profile needs "
            f"an extent to be a profile of. Give the source a size, or fold the value into "
            f"`amplitude`."
        )
    if kind != "amp_data":
        return
    if _is_cylindrical(sim):
        # amp_file_func reads p.x() and p.y() unconditionally (sources.cpp:365-367) and
        # sizes itself from amp_func_vol->in_direction(X) in the Dcyl branch as well
        # (sources.cpp:360-363) — but a Dcyl `vec` carries its coordinates in the R and Z
        # slots, and nothing in the coordinate system ever writes X or Y. Whatever MEEP
        # samples there is not the radial profile the caller drew, so there is no
        # behaviour here worth reproducing.
        reasons.append(
            f"{label} sets amp_data in cylindrical coordinates; MEEP's amp_file_func reads "
            f"p.x() and p.y() (sources.cpp:365-367), which a Dcyl vec does not carry — the "
            f"array is not sampled over (r, z) there. Pass an amp_func of (r, z) instead."
        )
        return
    array = numpy.asarray(source.amp_data)
    if array.ndim > 3 or array.size == 0:
        reasons.append(
            f"{label} sets amp_data of shape {tuple(array.shape)}; MEEP's interpolator "
            f"indexes exactly three non-empty axes (fields.cpp:792-812)."
        )
        return
    if not numpy.all(numpy.isfinite(array)):
        reasons.append(f"{label} sets an amp_data array holding a non-finite value.")


# Sources MEEP synthesizes at setup whose synthesis this engine RE-RUNS through its
# own deposition machinery. The setup-time half (an MPB mode solve) stays MEEP's —
# exactly the work this package delegates — and the deposition half is
# `_build_source_points`, this package's transcription of the same `add_volume_source`
# MEEP's synthesis calls, so the lifted table matches MEEP's by construction.
#
# EigenModeSource is lifted by `_lift_eigenmode_source`, which re-runs MEEP's own
# synthesis — the MPB solve via sim.get_eigenmode, the four equivalent-current
# deposits via this engine's transcription of add_volume_source — rather than reading
# the deposited current back. Measured against CPU MEEP stepping its own eigenmode
# source: complex Ez 6.2e-07 over the full cell (2-D waveguide, peak ratio 1.0000,
# identical forward/backward ratio) and 3.8e-07 / 1.0e-06 Ex/Ey (3-D square
# waveguide), and 7.4e-07 / 1.2e-06 for the two NO_DIRECTION oblique launches — the
# latter 1.2e-06 on REAL float32 storage as well, where the mode's in-plane phase ramp
# is projected the way MEEP projects it (step.cpp:307). The variants that construction
# cannot express — a DiffractedPlanewave band, 1-D, eig_match_freq=False on a custom
# waveform — are refused per source in `_check_eigenmode_source`, not blanket-refused
# here.
#
# GaussianBeamSource / GaussianBeam3DSource are lifted by `_lift_gaussian_beam_source`
# on the same pattern, with the analytic beam evaluator transcribed in
# `.gaussian_beam` (validated against MEEP's own compiled `gaussianbeam::get_fields`
# to 9.2e-14) standing in for the MPB solve.
#
# GaussianBeam2DSource is a THIRD shape, and it needs no transcription at all: despite
# the name it is not the 3-D beam but an exact 2-D Hankel-function Green's function
# performed ENTIRELY IN PYTHON before add_source is reached (`get_fields` ->
# `green2d`, source.py:821-1044) and then deposited as ordinary mp.Source objects
# carrying amp_data (`get_equiv_sources`, source.py:780-812). So
# `_lift_gaussian_beam_2d_source` CALLS MEEP's synthesis — setup work, delegated by
# design — and routes each deposited plain source through the ordinary amp_data path,
# which reaches MEEP's own interpolator through `.amp_interpolation` (measured
# bit-identical, difference exactly 0.0). It is the one lift route that needs scipy,
# and it needs MEEP's rather than a dependency of this package; the gate names it.
#
# The table itself is built at the bottom of this module, where both lifters exist.


_SPECIAL_SOURCE_NOTES = {
    "IndexedSource": (
        "mp.IndexedSource carries a precomputed per-point amplitude table from the adjoint "
        "machinery; it is an internal MEEP type rather than a user-facing source."
    ),
}


def _check_source_time(mp, src, label: str, reasons: list[str]) -> None:  # The waveform half of a source.
    # The carrier frequency must be finite and NONZERO — either sign. A negative
    # frequency is lifted as MEEP's own conjugate carrier (sources.cpp:72-96 stores
    # ``freq`` raw; sources.cpp:106 rotates ``exp(-i*2*pi*f*(t-t0))`` with it), the
    # plus/minus-omega superposition dipole_in_vacuum_cyl_off_axis.py drives. Zero
    # divides MEEP's amplitude correction ``1/(-2*pi*i*f)`` (sources.cpp:104):
    # MEEP steps an infinite current without a word, and the driver refuses it —
    # so the gate must refuse it too, by name, or the verdict and the lift disagree.
    if isinstance(src, (mp.GaussianSource, mp.ContinuousSource)):
        frequency = float(src.frequency)
        if not math.isfinite(frequency) or frequency == 0.0:
            reasons.append(
                f"{label} is a {type(src).__name__} with frequency={src.frequency!r}; the "
                f"carrier frequency must be finite and nonzero (a NEGATIVE one is lifted "
                f"as MEEP's conjugate carrier, but zero divides the amplitude correction "
                f"1/(-2*pi*i*f), sources.cpp:104)."
            )
    if isinstance(src, mp.GaussianSource):
        width = float(src.width)
        if not math.isfinite(width) or width <= 0.0:
            reasons.append(
                f"{label} is a GaussianSource with width={src.width!r}; this engine takes "
                f"fwidth = 1/width and needs it finite and positive."
            )
        return
    if isinstance(src, mp.ContinuousSource):
        return
    if isinstance(src, mp.CustomSource):
        if not callable(getattr(src, "src_func", None)):
            reasons.append(f"{label} is a CustomSource whose src_func is not callable.")
        return
    reasons.append(
        f"{label} has time dependence {type(src).__name__}, which is not one of "
        f"mp.ContinuousSource, mp.GaussianSource or mp.CustomSource."
    )


def _fold_drops_source(mp, grid, source, dimensions: int) -> bool:  # Does the fold discard this one outright?
    """Is this source in the half a mirror plane discards — the half MEEP never deposits in?

    MEEP adds sources with ``use_symmetry=false`` (``add_volume_source``,
    sources.cpp:487) and stores the upper half of a folded axis, so a source declared
    in the lower half intersects no stored chunk and drives NOTHING. It is not imaged
    and it is not refused; it is simply absent from the run MEEP steps.

    Dropped here rather than passed on, for exactly the reason an invariant-axis PML
    layer is (:func:`_resolve_pml_faces`): ``sources._validate_symmetry`` raises on
    it, which is the right answer for someone WRITING a driver by hand — they have no
    MEEP to imitate and a source that drives nothing is a mistake — and the wrong one
    for a converter reproducing a run where MEEP ignored it.

    It is not a corner case, and the scripts that hit it show why the drop is the
    faithful reading rather than a leniency: ``perturbation_theory_2d.py``,
    ``solve-cw.py`` and ``antenna_pec_ground_plane.py`` each declare the mirror PAIR
    explicitly — a source at ``+(r + 0.1)`` and its partner at ``-(r + 0.1)`` with
    ``amplitude=-1`` — alongside ``mp.Mirror``. MEEP keeps the first and discards the
    second, and the fold then regenerates exactly that discarded partner, sign
    included. Dropping it reproduces the pair; keeping it would deposit the partner
    twice.

    What is still refused is the case where this leaves the run with NO source at
    all — an exactly-zero volume that reads as a completed run. That check is in
    :func:`_check_source_folds`, where it can see every source at once.
    """
    if type(source) is not mp.Source and type(source).__name__ not in REALIZABLE_SOURCES:
        return False
    centre = _reduced_vector(source.center, dimensions)
    tolerance = grid.dx / 10
    return any(
        grid.is_mirrored(axis) and centre[axis] < -tolerance
        for axis in range(3)
    )


def _declared_source_sheets(mp, sim, source, dimensions: int, names: dict) -> list:
    """The (component, has_amp_func) pairs one declaration actually deposits.

    THE GATE USED TO READ ``source.component`` and stop there. For an
    ``mp.EigenModeSource`` that attribute is ``ALL_COMPONENTS``, which is not a field
    component at all, so the parity rule never ran on the four sheets MEEP expands it
    into (mpb.cpp:874-900) and the pre-flight said ``supported`` for five of MEEP's
    own ``python/tests`` cases the driver then raised on mid-lift — the exact contract
    failure the docstring below forbids. Expanding here is what lets the gate answer
    the same question :func:`sources._validate_symmetry_parity` answers.

    ``has_amp_func`` is what the parity rule needs to know about the profile, and each
    kind knows it without evaluating anything: a plain ``mp.Source`` carries one only
    if the user gave it one (in any of MEEP's three spellings), while an eigenmode or
    beam sheet is BY CONSTRUCTION a profile — MPB's mode field, or the Gaussian beam's
    — so it can carry the plane's parity and the rule accepts it. That is the whole
    reason the gate need not run MPB to decide: the predicate reads the presence of a
    profile, never its values.
    """
    kind = type(source).__name__
    if type(source) is mp.Source:
        # Each of MEEP's three spellings has its OWN "absent" value and they are not the
        # same value: `amp_func` and `amp_data` default to None but `amp_func_file`
        # defaults to the EMPTY STRING (mp.Source.__init__). Testing all three against
        # None reports a profile on every plain source ever written, which silently
        # turned the parity rule's uniform-sheet branch off — the branch that refuses
        # the crossing sheet stock MEEP answers 57%-283% wrong.
        has_amp_func = (
            getattr(source, "amp_func", None) is not None
            or getattr(source, "amp_data", None) is not None
            or bool(getattr(source, "amp_func_file", "") or "")
        )
        component = names.get(getattr(source, "component", None))
        return [(component, has_amp_func)] if component else []
    if kind in ("EigenModeSource", "GaussianBeamSource"):
        try:
            normal_axis = _source_normal_axis(sim, source, dimensions)
        except MeepSimulationNotLiftable:
            # Not a surface source, so it has no sheets and the FOLD rule has nothing to
            # judge. Returning [] is now a DEFERRAL, not a limitation: the shape is
            # refused by name one check earlier, in `_check_eigenmode_source` /
            # `_check_gaussian_beam_source`, so the verdict this contributes to already
            # carries a reason and the gate/lift gap this branch used to document is
            # closed. (It stayed open for a while because closing it is a new gate
            # refusal, and a new refusal moves the corpus survey's accepted counts —
            # re-measured 2026-08-08: it moves none of them.) Nothing here re-refuses
            # it, because the fold rule genuinely has no opinion about a shape.
            return []
        if kind == "EigenModeSource":
            keep = _eigenmode_component_filter(mp, source, dimensions)
        else:
            keep = _beam_component_filter(mp, source, normal_axis, dimensions)
        driven = [component
                  for component, _mode, _sign in _eigenmode_current_sheets(mp, normal_axis)
                  if keep(component)]
        return [(names[component], True) for component in driven if component in names]
    return []


def _check_source_folds(mp, sim, reasons: list[str]) -> None:  # Every source, against every mirror plane.
    """Report here what ``FdtdDriver`` would refuse mid-lift: a source the fold discards.

    The pre-flight must never say ``supported=True`` and then watch the driver
    raise a bare ValueError mid-lift — the contract every placement check in
    this register serves. Here the axis rule is the mirror fold: MEEP
    stores the upper half of a folded axis and deposits with ``use_symmetry=false``
    (sources.cpp:487), so a source declared in the LOWER half reaches no stored chunk
    and drives nothing — MEEP returns a peak of exactly 0 rather than complaining
    (measured). ``sources._validate_symmetry`` raises on it; without this, the
    pre-flight said supported and the raise came from inside ``add_source``.

    A source ON such a plane whose PROFILE cannot carry the parity the fold demands is
    refused by the second rule, and reported from the same declarations before anything
    is built. That rule is :func:`sources.mirror_parity_offenders`, called here rather
    than restated: this function used to keep its own copy, worded as "the source
    cancels against its own image and the implied full-domain current is zero … wrong
    by ~1e0", and every clause of that is measurably false. MEEP deposits with
    ``use_symmetry=false`` (sources.cpp:487) so no J ever meets a -J; what it does is
    keep the owned half and image the rest with the opposite sign, which reproduces the
    declared run exactly when the declared current already has that parity. Sharing the
    predicate is also what keeps the two verdicts from drifting apart, which they did:
    the copy here stayed at "on the plane AND the component is odd" after the driver's
    half was narrowed to the profile, leaving the gate refusing extended sources that
    lift and step correctly.

    Off the plane neither rule applies: see ``sources._validate_symmetry_parity`` for
    the measured equivalence that earned that.
    """
    from .sources import mirror_parity_message, mirror_parity_offenders

    planes = {}
    for symmetry in getattr(sim, "symmetries", ()) or ():
        planes[int(symmetry.direction)] = int(complex(symmetry.phase).real) or 1
    if not planes:
        return
    dimensions = _effective_dimensions(mp, sim)
    dx = 1.0 / float(sim.resolution)
    names = _component_names(mp, _is_cylindrical(sim))
    liftable = [
        (index, source) for index, source in enumerate(getattr(sim, "sources", ()) or ())
        if type(source) is mp.Source or type(source).__name__ in REALIZABLE_SOURCES
    ]
    phases = [planes.get(axis) for axis in range(3)]
    dropped = []
    for index, source in liftable:
        centre = _reduced_vector(source.center, dimensions)
        if any(axis in (0, 1, 2) and centre[axis] < -dx / 10 for axis in planes):
            dropped.append(index)  # In the discarded half — DROPPED, not refused; see below.
            continue
        size = _reduced_vector(getattr(source, "size", mp.Vector3()), dimensions)
        for component, has_amp_func in _declared_source_sheets(mp, sim, source, dimensions, names):
            offenders = mirror_parity_offenders(component, phases, centre, size, has_amp_func, dx)
            if offenders:
                reasons.append(f"sources[{index}]: " + mirror_parity_message(component, offenders))
    # A source in the discarded half is DROPPED, not refused — see `_fold_drops_source`
    # for why that is MEEP's own semantics rather than a leniency. What is refused is
    # the case that leaves the run with nothing: every source discarded means MEEP
    # steps zero fields and returns an exactly-zero volume, which is the same empty
    # run `_check_sources` already refuses when no source was declared at all.
    if dropped and len(dropped) == len(liftable):
        reasons.append(
            f"every source ({len(dropped)} of them) sits in the half its mirror plane "
            f"DISCARDS — this engine stores the upper half of a folded axis, as MEEP does. "
            f"MEEP does not refuse this: sources are added with use_symmetry=false "
            f"(sources.cpp:487), so none of them lands in a stored chunk, and the run returns "
            f"a field whose peak is exactly 0 while reading as complete (measured). Declare "
            f"them at the mirrored coordinates, which under these planes is the same physical "
            f"configuration."
        )


def _check_materials(mp, sim, reasons: list[str]) -> None:  # Every mp.Medium the cell can hold.
    """Media may differ in ``epsilon_diag`` and in nothing else, and must be representable."""
    routes = _epsilon_routes(mp, sim)
    _check_epsilon_routes(mp, sim, routes, reasons)

    materials = _materials_of(mp, sim)
    non_media = [material for material in materials if not _is_medium(mp, material)]
    if non_media:
        kinds = sorted({type(material).__name__ for material in non_media})
        reasons.append(
            f"the cell contains material(s) of type {kinds}, which name no medium this module "
            f"can read. Two kinds of material that USED to land here no longer do and are not "
            f"what this refusal is about: a MaterialGrid, which _materials_of expands into its "
            f"two endpoint media (so a grid reaching here means an endpoint is not a plain "
            f"mp.Medium), and a per-point epsilon route — a callable, an ndarray or an "
            f"epsilon_input_file — which _epsilon_routes carries and _check_epsilon_routes "
            f"validates against what MEEP actually built."
        )
        return
    _check_material_grids(mp, sim, reasons)
    _check_structured_conductivity(mp, sim, materials, reasons)
    signatures = {_medium_signature(material) for material in materials}
    if len(signatures) > 1:
        # A structured cell is lifted by reading MEEP's own rasterized structure back
        # per point. THREE of the signature's entries are recovered that way and may
        # therefore differ from medium to medium — :data:`PER_POINT_SIGNATURE_INDICES`
        # names them and says which read recovers each. What is left has no reader, so
        # every medium still has to agree on it; E_susceptibilities are the fourth
        # recoverable entry and take the branch below, because their route needs
        # preconditions the other three do not.
        without_per_point = {
            _signature_without(signature, PER_POINT_SIGNATURE_INDICES)
            for signature in signatures
        }
        if len(without_per_point) > 1:
            # Index 12 of the signature is E_susceptibilities. Dropping it too asks
            # "is (per-point-recoverable, E dispersion) the only difference?" — and that
            # case now lifts on ANY MEEP. Where fields::get_susceptibility_sigma exists
            # (meep-sigma-reader.patch, plan in the design notes (meep-sigma-reader-plan))
            # the per-point sigma volumes are read exactly; where it does not, they
            # are looked up in the declared geometry and checked point by point
            # (meep_gpu.sigma_lookup), with the chi1inv inversion
            # (meep_gpu.sigma_recovery) behind that. The INVERSION needs a few extra
            # preconditions neither of the other two does — checked below, and only
            # when it is the route the lift would actually take, so that the
            # pre-flight verdict and the route agree. Nothing here initializes anything.
            beyond_eps_and_dispersion = {
                _signature_without(
                    signature, PER_POINT_SIGNATURE_INDICES + (SUSCEPTIBILITY_SIGNATURE_INDEX,))
                for signature in signatures
            }
            if len(beyond_eps_and_dispersion) == 1:
                # Media differ only in the recoverable entries and E_susceptibilities.
                # Every medium's terms must be liftable kinds; materials[0]'s are checked
                # by _check_medium below.
                for index, material in enumerate(materials[1:], start=1):
                    for tindex, term in enumerate(material.E_susceptibilities):
                        _check_susceptibility(
                            mp, term, f"media[{index}].E_susceptibilities[{tindex}]", reasons)
                if not _has_sigma_reader(mp):
                    _check_sigma_is_recoverable(mp, materials, reasons)
            else:
                reasons.append(
                    f"the cell contains {len(signatures)} distinct media that differ in more than "
                    f"their permittivity, their D_conductivity_diag and their E_susceptibilities "
                    f"(chi2/chi3, mu, B_conductivity_diag, a ROTATED conductivity). A "
                    f"structured cell is lifted by reading MEEP's own rasterized structure back "
                    f"per point — the full chi1inv tensor at frequency 0 for the permittivity, "
                    f"the D-row chi1inv at a nonzero frequency for D_conductivity_diag "
                    f"(monitor.cpp:339-343), and the susceptibility sigma volumes beside them "
                    f"(read exactly where fields.get_susceptibility_sigma exists, recovered from "
                    f"chi1inv otherwise). MEEP's per-point chi2/chi3 volumes have NO reader on "
                    f"any MEEP and are invisible in chi1inv (measured: adding chi2=5, chi3=20 to "
                    f"a block moves chi1inv by exactly 0.000e+00 on the E row and on the D row, "
                    f"at frequency 0 and at 0.9), and the read reaches the D row only, so a "
                    f"B_conductivity_diag that VARIES from medium to medium has no reader either "
                    f"— a UNIFORM one is a declaration rather than a read and does lift, through "
                    f"FdtdDriver.set_b_conductivity. Build the run on FdtdDriver "
                    f"directly and install the volumes with set_epsilon_components(...), "
                    f"set_conductivity(...) and add_susceptibility(sigma=<volume>) if you need one."
                )
                return
            for index, material in enumerate(materials[1:], start=1):
                # The per-medium half of _check_medium's conductivity checks. The
                # representative medium is checked there; the others reach the engine
                # only through the per-point read, so what is refused of them is exactly
                # what that read cannot carry.
                _check_conductivity_of(material, f"media[{index}].D_conductivity", reasons)
                _check_conductivity_of(material, f"media[{index}].B_conductivity", reasons)
        _check_structured_geometry(mp, sim, reasons)
    for index, material in enumerate(materials):
        # Every medium's epsilon_diag is installed, so every one has to be sound;
        # the rest of the material model is shared and checked once below.
        for axis, value in enumerate(_vector3(material.epsilon_diag)):
            if is_pec_epsilon(value):
                # mp.metal / mp.perfect_electric_conductor, admitted BY VALUE. See
                # driver.PEC_EPSILON_SENTINEL: MEEP inverts this ordinarily to
                # chi1inv = -1e-20 and steps the ordinary constitutive update with it,
                # so the lift carries MEEP's own number through and adds no physics.
                # mp.perfect_magnetic_conductor is NOT admitted here and must not be: it
                # is mu = -1e20 with epsilon = 1, refused by _check_medium's mu_diag
                # branch below because the B-side constitutive machinery is not built.
                continue
            if not math.isfinite(value) or value <= 0.0:
                reasons.append(
                    f"media[{index}].epsilon_diag[{AXIS_NAMES[axis]}]={value!r} must be finite and "
                    f"strictly positive; a non-positive instantaneous permittivity has no stable "
                    f"leapfrog."
                )
    _check_medium(mp, _representative_medium(mp, materials), reasons, sim=sim)


def _representative_medium(mp, materials):
    """The medium whose non-permittivity model the whole cell shares.

    ``materials[0]`` — the default material — except when every declared material is a
    per-point epsilon route and there is no ``materials[0]`` to take. Vacuum is the
    right stand-in there and not a guess: such a route installs a permittivity and
    NOTHING else (:func:`_require_epsilon_only_structure` proves it against the built
    structure), so the conductivity, susceptibility and nonlinearity the caller of this
    function goes on to install are vacuum's by construction.
    """
    return materials[0] if materials else mp.Medium()


def _check_epsilon_routes(mp, sim, routes: tuple[str, ...], reasons: list[str]) -> None:
    """The pre-flight half of the per-point-route gate: what the route RETURNS.

    Nothing about the route's spelling is refused any more — see the section comment
    above :data:`EPSILON_ROUTE_CALLABLE`. What is refused is a route that would put
    something other than an instantaneous permittivity into the structure, because
    that is all :func:`_lift_epsilon_structured` reads back.

    Three checks, in increasing order of what they can see:

    * A callable is SAMPLED across the cell (:func:`_probe_material_callable`). Every
      medium it returns must be permittivity alone.
    * An ndarray is read outright — it is a permittivity volume by construction, so
      only its values need to be sound.
    * ``epsilon_input_file`` cannot be read here at all: this package carries no HDF5
      dependency. Nothing is checked pre-flight; :func:`_require_epsilon_only_structure`
      and the per-point positivity check inside the structured lift see the values
      after MEEP has read them, before a single step.

    The last refusal is about the CELL rather than the route: a per-point route
    together with declared media that differ in more than their permittivity would
    make the structure's dispersion unattributable — the exact backstop compares one
    whole-cell epsilon against another and cannot say which half a difference came
    from. Both spellings are supported; mixing them in one cell is not.
    """
    if not routes:
        return
    for material in _declared_materials(sim):
        if _is_material_grid(mp, material):
            continue  # A grid is expanded into its endpoint media; _check_material_grids has it.
        kind = _epsilon_route_kind(material)
        if kind == EPSILON_ROUTE_CALLABLE:
            label = f"the material callable {getattr(material, '__name__', repr(material))}"
            reasons.extend(_probe_material_callable(mp, sim, material, label))
        elif kind == EPSILON_ROUTE_ARRAY:
            array = numpy.asarray(material, dtype=numpy.float64)
            if not array.size or not numpy.all(numpy.isfinite(array)) or numpy.any(array <= 0.0):
                reasons.append(
                    f"an ndarray permittivity of shape {tuple(array.shape)} is declared as a "
                    f"material, and it is empty or holds a non-positive or non-finite value; a "
                    f"non-positive instantaneous permittivity has no stable leapfrog."
                )
    media = _materials_of(mp, sim)
    signatures = {_medium_signature(material) for material in media if _is_medium(mp, material)}
    if len({signature[1:] for signature in signatures}) > 1:
        reasons.append(
            f"this cell declares a per-point epsilon route ({', '.join(routes)}) AND media that "
            f"differ in more than their permittivity. The route is lifted by reading MEEP's "
            f"rasterized chi1inv, which carries permittivity alone, and the check that proves "
            f"nothing else reached the structure reads the whole cell at once — so it cannot "
            f"attribute a dispersion to the route or to the declared medium. Use one spelling "
            f"or the other."
        )


# Everything a medium can declare that MEEP's material-grid evaluation NEVER reads,
# as (attribute, the value that means "not declared"). `md->medium` is
# default-constructed for a MATERIAL_GRID (material_data.cpp:47-50, epsilon 1 / mu 1 /
# every chi and conductivity zero) and the ONLY thing merged into it from the two
# endpoints is their E_susceptibilities list (typemap_utils.cpp:528-547);
# `epsilon_material_grid` then writes exactly epsilon_diag, epsilon_offdiag, the
# susceptibility sigmas and D_conductivity_diag per point (meepgeom.cpp:569-604).
# Nothing else on an endpoint medium reaches the structure at all.
_MATERIAL_GRID_IGNORED_FIELDS: tuple[tuple[str, tuple], ...] = (
    ("mu_diag", (1.0, 1.0, 1.0)),
    ("mu_offdiag", (0.0, 0.0, 0.0)),
    ("E_chi2_diag", (0.0, 0.0, 0.0)),
    ("E_chi3_diag", (0.0, 0.0, 0.0)),
    ("H_chi2_diag", (0.0, 0.0, 0.0)),
    ("H_chi3_diag", (0.0, 0.0, 0.0)),
    ("B_conductivity_diag", (0.0, 0.0, 0.0)),
    ("B_conductivity_offdiag", (0.0, 0.0, 0.0)),
)


def _check_material_grids(mp, sim, reasons: list[str]) -> None:  # What the endpoint pair cannot say.
    """The three things a ``mp.MaterialGrid`` can carry that its endpoint media do not.

    :func:`_materials_of` expands every grid into ``medium1``/``medium2``, which
    is sound for the permittivity — MEEP interpolates that linearly between them
    (meepgeom.cpp:569-583) and reading the result back per component is what
    :func:`_lift_epsilon_structured` already does. Measured against pristine CPU
    MEEP stepping the same object, on a 4x4x8 grid ramped 1 -> 6 in z: 2.9431e-07
    with ``do_averaging=False``, 2.5961e-07 with it on, 2.5445e-07 at
    ``beta=1000`` (the level-set projection) and 2.0515e-07 for a grid used as
    ``default_material`` — against 2.7525e-07 for a plain-medium control on the
    same cell, so a lifted grid sits at the engine's ordinary structured floor
    rather than merely inside a band.

    What the pair cannot speak for is refused here rather than approximated.

    * **``damping`` — NO LONGER REFUSED, and the reason is worth keeping.**
      meepgeom.cpp:623-626 adds ``u*(1-u)*damping`` to ``D_conductivity_diag``
      at every point — a conductivity that belongs to neither endpoint and
      varies across the cell. The epsilon lift reads ``chi1inv`` at frequency 0,
      where monitor.cpp never reaches the conductivity branch, so a damped grid
      lifted through that read alone COMPLETES AND STEPS WRONG rather than
      failing: measured 1.0531e-02 relative at ``damping=0.3``, 1.7316e-02 at
      ``damping=0.5`` and 9.1963e-02 at ``damping=pi``, every one with the full
      signal and the same step count, against 2.9431e-07 for the identical cell
      at ``damping=0``. Four to five orders of magnitude, no symptom.
      ``test_adjoint_solver.py:329`` uses ``damping=np.pi*fcen``.

      What closed it is :func:`_read_conductivity_volumes`, which reads the SAME
      ``D_conductivity_diag`` array back off MEEP's D-row chi1inv at a NONZERO
      frequency — the damping rides it because MEEP put it there, so nothing
      here reimplements ``matgrid_val``, ``tanh_projection``, beta or eta.
      Measured on a random 30x30 design at ``damping=pi*0.8``: the recovered
      volume peaks at 0.628318, which is ``damping/4`` to six figures — the
      maximum of ``u*(1-u)*damping``, attained where ``u = 1/2`` — and the
      stepped parity goes 1.6377e-02 (damping dropped) -> 3.1668e-07
      (recovered), against 8.4773e-07 for the same cell at ``damping=0``. Only
      a NEGATIVE damping is still refused, as gain.
    * **Endpoint dispersion.** MEEP concatenates the two endpoints'
      ``E_susceptibilities`` (typemap_utils.cpp:528-547) and interpolates each
      sigma by ``(1-u)``/``u`` (meepgeom.cpp:585-596). Refused as UNMEASURABLE,
      not as unimplementable: on MEEP 1.33.0 single the path could not be
      activated at all. With a Lorentzian (f0=0.7, gamma=0, sigma=3) on
      ``medium2``, ``sim.get_epsilon_point`` reported eps(0.45) EQUAL to eps(0)
      at every sampled point — delta 0.0000 at z = -0.8, 0.0 and +0.8 — while
      the same reader on a plain medium carrying that pole moved 5.1130
      (6.0000 -> 11.1130). The eps_inf grading was correct on the same cell
      (1.2086 -> 3.4981 -> 5.7845), so the grid itself is built and only the
      susceptibility never lands. A parity number taken there would score a
      non-dispersive cell as a dispersive one, which is the exact failure this
      module exists to prevent. No corpus row needs it: every MaterialGrid in
      MEEP's test corpus uses frequency-independent media.
    * **Endpoint fields MEEP itself DISCARDS.** mu, chi2/chi3 and
      ``B_conductivity`` on an endpoint never reach the structure (see
      :data:`_MATERIAL_GRID_IGNORED_FIELDS`), and this is the one that would be
      a SILENT wrong answer in the other direction: ``_lift_material`` installs
      chi2/chi3 from the representative medium, so a grid whose endpoints both
      declare ``chi3`` would run a nonlinear cell that MEEP is stepping linear.
      MEASURED on a 2x2x4 cell at resolution 10, ``chi3=20`` on both endpoints:
      MEEP's grid run is BIT-IDENTICAL to the same grid with no chi3 at all
      (0.0000e+00) and bit-identical to a plain ``Medium(epsilon=4)``
      (0.0000e+00), while the plain-medium control ``Medium(chi3=20)`` against
      ``Medium()`` moves 1.4093e+00. Refusing is the only honest reading: the
      declaration says one thing and MEEP does another, and silently siding with
      either would run a cell nobody asked for.
    """
    for index, grid in enumerate(_material_grids_of(mp, sim)):
        label = f"MaterialGrid[{index}]"
        damping = float(getattr(grid, "damping", 0.0) or 0.0)
        if damping < 0.0:
            reasons.append(
                f"{label} has damping={damping!r}. MEEP adds u*(1-u)*damping to "
                f"D_conductivity_diag at every point (meepgeom.cpp:623-626); a NEGATIVE damping "
                f"is a negative conductivity, which is gain rather than loss and grows the field "
                f"while every magnitude still looks reasonable. A positive damping is lifted — "
                f"the volume is read back off MEEP's own D-row chi1inv rather than reconstructed "
                f"from the grid's u array (_read_conductivity_volumes)."
            )
        for name in ("medium1", "medium2"):
            medium = getattr(grid, name, None)
            if medium is None:
                continue
            if getattr(medium, "E_susceptibilities", ()):
                reasons.append(
                    f"{label}.{name} declares E_susceptibilities. MEEP concatenates the two "
                    f"endpoints' terms (typemap_utils.cpp:528-547) and interpolates each sigma "
                    f"by (1-u)/u (meepgeom.cpp:585-596), but that route could not be ACTIVATED "
                    f"on MEEP 1.33.0 single: with a Lorentzian (f0=0.7, sigma=3) on medium2, "
                    f"sim.get_epsilon_point returned eps(0.45) EQUAL to eps(0) at every sampled "
                    f"point (delta 0.0000) even though eps_inf graded correctly across the "
                    f"region (1.2086 -> 5.7845), while a plain medium carrying the identical "
                    f"pole moved 5.1130. So the declaration has no measurable effect on the "
                    f"structure and any parity number taken here would be scoring a "
                    f"non-dispersive cell as a dispersive one. Refused until it can be measured."
                )
            for field, unset in _MATERIAL_GRID_IGNORED_FIELDS:
                value = getattr(medium, field, None)
                if value is not None and _vector3(value) != unset:
                    reasons.append(
                        f"{label}.{name} declares {field}={_vector3(value)}, which MEEP's "
                        f"material-grid evaluation NEVER READS: md->medium is "
                        f"default-constructed (material_data.cpp:47-50) and only "
                        f"E_susceptibilities are merged into it (typemap_utils.cpp:528-547), so "
                        f"only epsilon, sigma and D_conductivity are written per point "
                        f"(meepgeom.cpp:569-604). The discard is MEASURED on its chi3 instance: "
                        f"with chi3=20 on both endpoints MEEP's grid run is BIT-IDENTICAL "
                        f"(0.0000e+00) both to the same grid with no chi3 and to a plain "
                        f"Medium(epsilon=4), while the same claim on a plain medium moves "
                        f"1.4093e+00. Lifting the endpoint verbatim would install a material "
                        f"MEEP is not stepping; drop the field from the endpoint medium if "
                        f"MEEP ignoring it is what you meant."
                    )
            if getattr(medium, "H_susceptibilities", ()):
                reasons.append(
                    f"{label}.{name} declares H_susceptibilities, which MEEP's material-grid "
                    f"evaluation never reads (see the {_MATERIAL_GRID_IGNORED_FIELDS[0][0]} "
                    f"refusal); magnetic dispersion is not implemented here either."
                )


def _check_structured_conductivity(mp, sim, materials, reasons: list[str]) -> None:
    """The one cell shape the per-point conductivity read is NOT measured on.

    :func:`_read_conductivity_volumes` is written and measured for the Cartesian
    lattice only. The Dcyl spelling — ``mp.iveccyl`` with ``Dr``/``Dp``/``Dz`` —
    is the same three lines the epsilon lift and the sigma lookup already carry, so
    it is very probably right; "very probably right" is exactly the standard this
    package refuses to install a graded loss volume on, because a wrong registration
    there completes and is wrong rather than failing. Refused at the GATE, by name,
    until it is measured against CPU MEEP on a cylindrical cell.

    A UNIFORM conductivity on a cylindrical cell is untouched by this: it is a
    declaration, not a read, and :func:`_lift_material` still installs it.
    """
    if _effective_dimensions(mp, sim) != MEEP_CYLINDRICAL:
        return
    if not _conductivity_structured(mp, sim, materials):
        return
    reasons.append(
        "this cylindrical (Dcyl) cell's D_conductivity varies from point to point — the "
        "media differ in D_conductivity_diag, or a MaterialGrid declares a damping "
        "(meepgeom.cpp:623-626). The volume would be recovered by reading MEEP's own "
        "D-row chi1inv back per Yee point (_read_conductivity_volumes), and that read is "
        "MEASURED on the Cartesian lattice only. Its Dcyl registration is unmeasured, and "
        "a graded loss installed half a cell out steps a smooth, complete, wrong run — "
        "which is the failure this package exists to prevent, so it is refused rather "
        "than attempted. A UNIFORM D_conductivity on a cylindrical cell is unaffected and "
        "still lifts."
    )


def _check_sigma_is_recoverable(mp, materials, reasons: list[str]) -> None:
    """What the chi1inv INVERSION needs beyond what the exact sigma reader needs.

    Run on any MEEP without ``fields.get_susceptibility_sigma`` and any cell whose
    media differ in their dispersion — which is both compatibility routes, the
    declared-geometry lookup and the inversion behind it, because the lookup can
    decline at lift time and hand the cell to the inversion. Refusing here is
    therefore refusing the pair.

    Two preconditions, both cheap and declaration-only (this function must not
    initialize anything):

    * **No ``D_conductivity``.** The recovery reads ``1/get_chi1inv(c, d, iloc, w)``
      and solves ``eps(w) = eps_inf + SUM_n sigma_n*chi_n(w)`` for the sigmas. A
      conductivity is NOT a term of that sum: monitor.cpp:340-344 multiplies the
      whole permittivity by ``1 + i*sigma_D/f`` after the susceptibilities are added,
      so every sample is scaled by a factor the basis has no column for. The
      cross-validation at the held-out frequency would catch it — that is what it is
      for — but it would report a pole-list disagreement, which is not what is wrong.
      The exact reader is unaffected (it reads the sigma arrays themselves, and the
      epsilon lift reads chi1inv at frequency 0, where monitor.cpp never reaches the
      conductivity branch), so this is a recovery-path refusal only.

      NOT closed by :func:`_read_conductivity_volumes`, though it could be. That read
      now recovers sigma_D exactly, so the factor is KNOWN and dividing it out of every
      sample would make the basis fit again. Deliberately not done here: re-deriving the
      inversion is a change to the numerics of :mod:`~.sigma_recovery`, which is pinned
      by its own measured residuals, and no corpus row needs the combination. Refusing
      a case the package could in principle serve is the honest state to leave it in,
      and this sentence is the record of that.
    * **A basis that is not singular.** :func:`~.sigma_recovery.plan_recovery`
      depends on nothing but the poles, so the whole conditioning question is
      answerable here, before ``init_sim``: a pole set the recovery would refuse at
      lift time is refused at the pre-flight instead, with the same message.

    The recovery's other precondition — every pole needs a finite POSITIVE frequency,
    since sigma comes back as ``c/w0**2`` — is already enforced for every medium in
    the cell by :func:`_check_susceptibility`, which is why it is not repeated here
    (``test_sigma_recovery.py::test_the_gate_refuses_a_zero_frequency_susceptibility_in_any_medium``
    verifies that rather than trusting it).
    """
    # ANY medium's, not materials[0]'s: since D_conductivity_diag became a per-point
    # recoverable entry (PER_POINT_SIGNATURE_INDICES) the media no longer have to agree
    # on it, so reading the representative alone would miss a lossy block beside a
    # lossless default and hand the inversion the very samples it cannot fit.
    conductivity = next(
        (_vector3(material.D_conductivity_diag) for material in materials
         if any(value != 0.0 for value in _vector3(material.D_conductivity_diag))),
        None,
    )
    if conductivity is not None:
        reasons.append(
            f"D_conductivity_diag={conductivity} together with media that differ in their "
            f"E_susceptibilities: this MEEP has no fields.get_susceptibility_sigma, so the "
            f"per-point sigma volumes have to come from chi1inv — and MEEP multiplies the "
            f"permittivity it reports there by (1 + i*sigma_D/f) after summing the "
            f"susceptibilities (monitor.cpp:340-344), which the recovery basis has no term for. "
            f"Drop the conductivity, or build a MEEP carrying the sigma reader (MEEP_SIGMA_PATCH=1 "
            f"via parity/meep_gpu/build_meep_133_macos.sh), which reads sigma directly and is "
            f"unaffected."
        )
    terms = _unique_susceptibilities(materials)
    kinds = (mp.LorentzianSusceptibility, mp.DrudeSusceptibility)
    if not terms or any(type(term) not in kinds for term in terms):
        return  # A kind this module cannot model at all; _check_susceptibility names it.
    try:
        plan_recovery([_recovery_pole(term) for term in terms])
    except SigmaRecoveryRefused as refusal:
        reasons.append(
            f"the per-point susceptibility sigma cannot be recovered from chi1inv on this MEEP: "
            f"{refusal}"
        )


def _is_axis_aligned_block(mp, obj) -> bool:  # A Block whose faces are normal to x, y and z.
    """Only an axis-aligned ``mp.Block`` is certain to leave chi1inv diagonal.

    MEEP's subpixel averaging is *anisotropic*: at an interface it builds the
    effective inverse-permittivity tensor in the frame of the surface normal, so a
    normal that is not a coordinate axis produces OFF-DIAGONAL ``chi1inv``. This
    engine stores a diagonal permittivity, and installing only the diagonal of a
    rotated tensor is a smooth, plausible, wrong run — measured at 5.7e-02
    (cylinder) and 2.5e-02 (rotated block) complex relative L2 against CPU MEEP.
    """
    if type(obj) is not mp.Block:
        return False
    axes = (_vector3(obj.e1), _vector3(obj.e2), _vector3(obj.e3))
    return all(
        sum(1 for value in axis if value != 0.0) == 1 for axis in axes
    ) and len({tuple(1 if value else 0 for value in axis) for axis in axes}) == 3


def _check_structured_geometry(mp, sim, reasons: list[str]) -> None:  # Shapes that tilt chi1inv.
    """Refuse curved or rotated geometry ONLY where a mirror fold makes it unliftable.

    Curved and rotated geometry itself is lifted: MEEP's anisotropic averaging
    puts real weight in the off-diagonal ``chi1inv`` entries at any surface whose
    normal is not a coordinate axis, and :func:`_lift_epsilon_structured` now
    reads those entries alongside the diagonal and installs the full row
    (measured against CPU MEEP stepping the same cell: uniform tensors at
    1.9e-07..4.6e-07 across periodic/metallic/Bloch/PML/chi3, curved and rotated
    structured cells in test_from_meep's parity cases).

    A mirror-folded grid takes the tensor unchanged. That was refused for one
    revision on the argument that the coupling coefficient's parity under the
    fold (the product of two components' parities) had no carrier in the ghost
    machinery — and the argument dissolved under measurement: the stencil never
    ghosts the coefficient at all (the partner-axis shift touches the field
    before the multiply, the own-axis shift goes UP, away from the fold), so
    driven fold-equivalence against the full domain is exact for every parity
    combination — 3.1e-12 even, 8.3e-13 odd, 4.7e-12 both
    (``stepping._mask_metallic_wall_coupling`` carries the numbers). Nothing is
    refused here any more; the function remains as the documented site of that
    decision.
    """
    del mp, sim, reasons  # Curved/rotated geometry lifts folded and unfolded alike.


def _check_medium(mp, medium, reasons: list[str], sim=None) -> None:  # One mp.Medium against the driver's material model.
    del sim  # A tensor medium lifts folded and unfolded alike (fold-equivalence 8e-13..5e-12).
    if _vector3(medium.mu_diag) != (1.0, 1.0, 1.0) or any(
        value != 0.0 for value in _vector3(medium.mu_offdiag)
    ):
        reasons.append(
            f"mu_diag={_vector3(medium.mu_diag)} / mu_offdiag={_vector3(medium.mu_offdiag)}: this "
            f"engine steps B = mu_0 H with mu = 1 everywhere (stepping.update_H divides by nothing), "
            f"so a magnetic material would be run as a non-magnetic one."
        )
    if medium.H_susceptibilities:
        reasons.append(
            "H_susceptibilities (magnetic dispersion) are not implemented: the ADE machinery here "
            "is on the D side only."
        )
    if any(value != 0.0 for value in _vector3(medium.H_chi2_diag)) or any(
        value != 0.0 for value in _vector3(medium.H_chi3_diag)
    ):
        reasons.append("magnetic nonlinearity (H_chi2 / H_chi3) is not implemented.")
    _check_conductivity_of(medium, "D_conductivity", reasons)
    _check_conductivity_of(medium, "B_conductivity", reasons)
    # epsilon_diag is validated in _check_materials, for EVERY medium in the cell
    # rather than only this representative one.
    for index, term in enumerate(medium.E_susceptibilities):
        _check_susceptibility(mp, term, f"E_susceptibilities[{index}]", reasons)


def _check_conductivity_of(medium, label: str, reasons: list[str]) -> None:  # One medium's D or B loss.
    """The two things this engine refuses of a conductivity, for ANY medium in the cell.

    ``label`` names the property and IS the attribute prefix — ``"D_conductivity"`` or
    ``"B_conductivity"``, optionally with a ``media[i].`` qualifier in front for the
    structured case. The two sides take the same two refusals because MEEP treats them
    as one quantity: ``get_cnd`` is a single switch over Dx/Dy/Dz and Bx/By/Bz
    (meepgeom.cpp:1545-1559), ``structure::set_materials`` installs them in one
    ``FOR_D_AND_B`` loop (structure.cpp:376-378), and ``step_db`` passes
    ``s->conductivity[cc][d_c]`` into the same ``STEP_CURL`` on either side
    (step_db.cpp:125-127).

    Split out of :func:`_check_medium` because a structured cell's media are allowed
    to DIFFER in ``D_conductivity_diag`` (the volume is read per point off MEEP's own
    D-row chi1inv, :func:`_read_conductivity_volumes`), so the non-representative
    media reach the engine too and have to be checked as well. They may NOT differ in
    ``B_conductivity_diag`` — there is no B-row read — and the signature check refuses
    that before this runs.

    A DIAGONAL anisotropic conductivity is supported on both sides: MEEP is
    per-component here and so is this engine —
    :meth:`~.fields.Fields._set_conductivity_side` stores sigma, condfac and condinv
    per component, which is what the ``mp.Absorber`` path installs, and
    :meth:`~.driver.FdtdDriver.set_conductivity` /
    :meth:`~.driver.FdtdDriver.set_b_conductivity` take the matching
    ``{'Dx','Dy','Dz'}`` / ``{'Bx','By','Bz'}`` mapping.
    """
    side = "D" if label.endswith("D_conductivity") else "B"
    offdiag = _vector3(getattr(medium, f"{side}_conductivity_offdiag"))
    if any(value != 0.0 for value in offdiag):
        reasons.append(
            f"{label}_offdiag={offdiag} is a rotated conductivity tensor, and MEEP DOES "
            f"NOT READ IT. Its C++ medium_struct carries only D_conductivity_diag and "
            f"B_conductivity_diag (material_data.hpp:78-79); the Python-to-C++ conversion "
            f"copies exactly those two and no off-diagonal at all "
            f"(typemap_utils.cpp:815-816), and get_cnd answers per component from the "
            f"diagonal (meepgeom.cpp:1545-1559). MEASURED on pristine MEEP 1.33.0: a cell "
            f"with D_conductivity_offdiag=(0.5,0,0) steps BIT-IDENTICALLY (0.0000e+00) to "
            f"the same cell without it, while the same 0.5 on the diagonal moves the field "
            f"1.4446e-01. Lifting the declaration verbatim would install a loss MEEP is not "
            f"stepping, and dropping it silently would run a material the caller did not "
            f"describe; a rotated tensor has no diagonal representation in this engine's "
            f"loss term either (one sigma per {side} component, "
            f"Fields._set_conductivity_side). Drop the field if MEEP ignoring it is what "
            f"you meant. A diagonal anisotropic {side}_conductivity_diag IS supported. "
            f"NOTE on the B spelling, which depends on the MEEP release: in MEEP 1.33.0 "
            f"(and 1.31.0) mp.Medium's constructor assigns B_conductivity_offdiag from "
            f"the D KEYWORD (python/geom.py:450, "
            f"`self.B_conductivity_offdiag = Vector3(*D_conductivity_offdiag)`), so there "
            f"a B_conductivity_offdiag= keyword is silently discarded and a "
            f"D_conductivity_offdiag= keyword sets both; MEEP 1.34.0 assigns each keyword "
            f"to its own attribute."
        )
    diagonal = _vector3(getattr(medium, f"{side}_conductivity_diag"))
    if any(entry < 0.0 for entry in diagonal):
        reasons.append(f"{label}={diagonal} is negative, which is gain rather than loss.")


def _check_susceptibility(mp, term, label: str, reasons: list[str]) -> None:  # One mp.*Susceptibility.
    """Only plain Lorentzian and Drude terms; the noisy/gyrotropic subclasses are refused BY TYPE.

    ``mp.NoisyLorentzianSusceptibility`` subclasses ``mp.LorentzianSusceptibility``,
    so an ``isinstance`` test accepts it and would run a noiseless material under a
    noisy name — the noise term simply vanishing. The exact type is checked instead.
    """
    kind = type(term)
    if kind is not mp.LorentzianSusceptibility and kind is not mp.DrudeSusceptibility:
        reasons.append(
            f"{label} is an {kind.__name__}; this engine implements the plain Lorentzian and Drude "
            f"terms only (meep_gpu.dispersion.Susceptibility). Noisy, gyrotropic and "
            f"multilevel-atom susceptibilities each add state and a different update to the "
            f"per-timestep loop, and dropping to their noiseless / non-gyrotropic core would run a "
            f"different material under the same name."
        )
        return
    if any(value != 0.0 for value in _vector3(term.sigma_offdiag)):
        reasons.append(
            f"{label} has sigma_offdiag={_vector3(term.sigma_offdiag)}; the polarization state here "
            f"is diagonal (one sigma per E component)."
        )
    frequency = float(term.frequency)
    gamma = float(term.gamma)
    if not math.isfinite(frequency) or frequency <= 0.0:
        reasons.append(
            f"{label} has frequency={term.frequency!r}; MEEP's f_n = omega_n/(2*pi) must be finite "
            f"and strictly positive (the whole term's numerator is sigma*omega_n**2, so zero drives "
            f"nothing at all and would score as a perfectly transparent material)."
        )
    if not math.isfinite(gamma) or gamma < 0.0:
        reasons.append(
            f"{label} has gamma={term.gamma!r}; a negative gamma is gain (Im eps < 0 for f > 0), "
            f"which is unconditionally unstable in the time domain."
        )
    for axis, value in enumerate(_vector3(term.sigma_diag)):
        if not math.isfinite(value) or value < 0.0:
            reasons.append(
                f"{label} has sigma_diag[{AXIS_NAMES[axis]}]={value!r}; sigma must be finite and "
                f"non-negative."
            )


# MEEP monitor classes this lift can rebuild on the driver. The value is the reader a
# caller uses afterwards, quoted in the refusal for anything NOT in here so the message
# says what is missing rather than just that something is.
MIGRATABLE_MONITORS: dict[str, str] = {
    "DftFlux": "result.get_flux_spectrum(obj) / monitor.get_flux_spectrum(); the "
               "normalization idiom's transform through result.get_flux_data(obj) and "
               "run_on_gpu(minus_flux_data=...)",
    "DftFields": "result.get_dft_region(obj, component) / monitor.get_dft_spectrum(name)",
    "DftNear2Far": "result.load_near2far(sim, obj), then MEEP's own sim.get_farfields(obj, ...)",
    "DftForce": "result.get_forces(obj) / monitor.get_force_spectrum()",
    "DftEnergy": "result.get_electric_energy(obj) / get_magnetic_energy(obj) / "
                 "get_total_energy(obj)",
}

# The two transverse directions to each near2far normal, in MEEP's cyclic order
# (near2far.cpp add_dft_near2far). The order is load-bearing twice over: it decides
# WHICH components are accumulated, and the j index decides each component's sign.
_NEAR2FAR_TRANSVERSE: dict[int, tuple[int, int]] = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


def _near2far_stored_weight(sign: float, weight: complex) -> complex:
    """MEEP's ``s * w->weight``, built the way MEEP builds it: COMPONENTWISE.

    ``s`` is a ``double`` in ``fields::add_dft_near2far`` (near2far.cpp:643-647), so
    the product goes through ``operator*(const T &, const complex<T> &)`` — it scales
    the real and imaginary parts separately. Python's ``float * complex`` promotes the
    scalar and runs the full complex product ``(ac - bd) + (ad + bc)i`` instead, and
    the two disagree on the SIGN OF A ZERO: for ``s = -1``, ``w = 1+0j`` the
    componentwise form gives ``(-1.0, -0.0)`` — which is what MEEP stores — while the
    promoted form gives ``(-1.0, +0.0)``, because ``-0.0 + 0.0`` is ``+0.0``.

    That sign is not decoration. It is the only thing separating two chunks at a flux
    box's corner (see :meth:`Near2FarMigration._emit_chunk`), so the entry weight has
    to carry the same one MEEP's chunk does.
    """
    return complex(sign * weight.real, sign * weight.imag)


def _near2far_source_component(family: str, j: int, transverse: tuple[int, int]) -> str:
    """MEEP's equivalent-source component ``c0``, this engine's name for it.

    ``c0 = direction_component(i == 0 ? Hx : Ex, fd[1 - j])`` (near2far.cpp:641-642):
    the OTHER family on the OTHER transverse direction. MEEP records it on every chunk
    as ``vc`` (dft.cpp:113, ``vc = data->vc``; near2far.cpp:646 passes ``c0`` into
    ``add_dft``'s ``vc`` slot), which makes it a per-region-normal label this engine can
    match a chunk against without inspecting coordinates.

    Named on this engine's axes (x/y/z, which are r/phi/z on a ``Dcyl`` cell) so it
    compares directly against ``_component_names``' reading of ``chunk.vc``.
    """
    first, second = transverse
    return ("H" if family == "E" else "E") + "xyz"[second if j == 0 else first]


def _same_stored_weight(entry_weight: complex, chunk_weight: complex) -> bool:
    """Is this entry's stored weight MEEP's chunk weight, SIGNED ZEROS INCLUDED?

    Compared per part rather than through ``abs(a - b)``, because the difference of two
    weights that differ only in the sign of a zero is exactly ``0.0``: ``(1+0j)`` and
    ``(1-0j)`` are ``==`` in Python and their difference has zero modulus, yet they are
    MEEP's labels for two DIFFERENT near2far regions meeting at a corner.
    """
    for ours, theirs in ((entry_weight.real, chunk_weight.real),
                         (entry_weight.imag, chunk_weight.imag)):
        if abs(ours - theirs) > 1e-12 * (1.0 + abs(theirs)):
            return False
        if ours == 0.0 and theirs == 0.0 and math.copysign(1.0, ours) != math.copysign(1.0, theirs):
            return False
    return True


@dataclass
class Near2FarMigration:
    """One MEEP ``DftNear2Far`` rebuilt as raw-Yee accumulators on the driver.

    THE TWO THINGS THAT MADE THIS HARD, recorded because both are invisible in a
    shape check and cost a long hunt:

    1. **MEEP's chunk decomposition owns the layout.** A cell with a PML splits into
       27 structure chunks here; the near surface then spans 6 dft chunks per
       component, and ``get_dft_data`` returns their CONCATENATION rather than one
       contiguous box. The totals are identical either way (308 sites for one
       component, 3720 across the four), so every size assertion passes while the
       values are scrambled — and no axis permutation or index offset can reconcile
       one box with six. :meth:`packed` therefore reads the layout off the live chunk
       list instead of reconstructing it. MEEP documents the same constraint from the
       other side: ``load_near2far_data`` requires "the same chunk layout".
    2. **``args`` holds the REQUESTED decimation, not the resolved one.** With MEEP's
       default of 0 it says only "automatic", and re-deriving the rule is unreliable —
       MEEP's documented ``floor(1/(dt*(freq_max + src_freq_max)))`` gives 10 on the
       probe case while MEEP resolved 9, because ``src_time::get_fwidth()`` is not the
       nominal fwidth. Measured 2.2e-02 with the factor mismatched against 2.4e-07
       once it agrees, so ``_resolved_decimation`` asks the chunk.

    Established before either was found, and worth keeping: the accumulator itself is
    right. At one interior site its value matched a DFT computed by hand from CPU
    MEEP's own field time series at that exact position —
    ``2.323522e-06+1.616721e-06j`` against ``2.323518e-06+1.616717e-06j``.

    ``entries`` is in MEEP's CREATION order — per region, E_fd0, E_fd1, H_fd0,
    H_fd1, regions in the user's order — which is the REVERSE of the emitted
    order, because MEEP's dft chunks prepend (``next_in_dft = data->dft_chunks``,
    dft.cpp:128) and ``get_dft_data`` walks the resulting list head-first, each
    chunk contributing its ``N * Nfreq`` values in storage order
    (``_get_dft_data`` / ``_load_dft_data``, python/meep.i:479-513). That is why
    :meth:`packed` walks MEEP's live list rather than ``entries``. Measured on
    CPU MEEP: a z-normal region's flat data is Hy, Hx, Ey, Ex, each segment
    idx-major and frequency-minor (``dft[Nomega * idx_dft + i]``, dft.cpp:298),
    and the four segments have DIFFERENT lengths because each component's Yee
    lattice intersects the region differently.

    Each entry's ``stored_weight`` is the tangential-pair sign times the region's
    own weight — the ``s * w->weight`` MEEP folds into the chunk's scale: with
    ``(fd0, fd1)`` the cyclic transverse pair, E_fd0 gets ``-w``, E_fd1 ``+w``,
    H_fd0 ``+w``, H_fd1 ``-w`` (``s = j==0 ? +1 : -1``, flipped for electric
    components), built by :func:`_near2far_stored_weight` so its signed zeros are
    MEEP's. ``source_component`` is MEEP's equivalent-source label ``c0`` for the
    same entry (:func:`_near2far_source_component`), which is what the chunk carries
    as ``vc``.
    """

    entries: tuple  # ((component, stored_weight, YeeRegionDFT, source_component), ...) creation order.
    frequencies: tuple
    cylindrical: bool = False  # A Dcyl chunk's bounds live in R/Z; phi is not a stored direction.

    def packed(self, mp, meep_near2far) -> numpy.ndarray:
        """The flat array MEEP's ``load_near2far_data`` expects — per MEEP CHUNK.

        The layout is not this engine's to choose: it is whatever MEEP's own chunk
        decomposition produced, so it is read off the live object rather than
        reconstructed. Walking ``swigobj.F`` through ``next_in_dft`` gives each chunk's
        component and its ``is``/``ie`` bounds in MEEP's doubled integer lattice, and
        each chunk's samples are emitted in that order — the ``LOOP_OVER_IVECS``
        order between those corners (vec.hpp:151-168: axis 0 outermost to axis 2
        innermost, C order over (x, y, z) in 3-D), sites major and frequencies
        minor (``update_dft``, dft.cpp:265-307), which is exactly what
        ``_load_dft_data`` (python/meep.i:497-513) writes back per chunk.

        Reconstructing it instead — emitting one contiguous box per component — is
        correct only on an unsplit grid. A cell with a PML splits into 27 structure
        chunks here, the near surface spans 6 dft chunks per component, and the
        concatenation of six sub-boxes has the SAME TOTAL SIZE as one box covering the
        same sites. That is why the mistake survives every shape check: totals match,
        and no axis permutation or index offset can reconcile the two.
        """
        head = getattr(getattr(meep_near2far, "swigobj", None), "F", None)
        if head is None:
            raise MeepSimulationNotLiftable([
                "this DftNear2Far has no accumulated chunk list (swigobj.F); the "
                "simulation must have been initialized before its data can be laid out."
            ])
        names = _component_names(mp, self.cylindrical)
        # A chunk's `is`/`ie` are ivecs of the run's own dimensionality: `Dcyl` has only
        # R and Z (vec.hpp:98-102 — `start_at_direction(Dcyl) = Z`, `stop_at_direction` = P),
        # so the X and Y slots hold nothing and reading them would silently pin every chunk
        # to doubled coordinate 0 on two axes. `None` marks phi, whose single stored plane
        # `_emit_chunk` supplies from the accumulator instead of from a coordinate.
        directions = (mp.R, None, mp.Z) if self.cylindrical else (mp.X, mp.Y, mp.Z)
        # A one-pixel periodic axis (MEEP's lower-dimensional emulation) makes MEEP's
        # loop_in_chunks visit the region once per LATTICE IMAGE of the single cell:
        # one dft chunk per image, with identical is/ie and the zero-extent split
        # weights w0 / w1 = 1 - w0 in the chunk's own s0 (measured on
        # binary_grating_n2f's point region: two chunks per component, s0.y = 0.0 and
        # 1.0, on arm64; x86-64 MEEP makes one chunk at 1.0 for a region exactly on the
        # site, because the arm64 build fuses the multiply-add in its grid rounding).
        # This engine holds ONE plane there at the summed weight 1, so each
        # MEEP chunk is emitted scaled by its own s0 on that axis — emitting the full
        # plane for both images would double the surface.
        grid = self.entries[0][2].grid if self.entries else None
        nosize = getattr(grid, "nosize_direction", None)
        image_axes = tuple(
            (axis, direction) for axis, direction in enumerate(directions)
            if direction is not None and callable(nosize) and nosize(axis)
        )
        mirrored_axes = _grid_mirror_axes(grid)
        parts, chunk, walked = [], head, 0
        while chunk is not None:
            walked += 1
            component = names.get(int(chunk.c))
            if component is None:
                raise MeepSimulationNotLiftable([
                    f"a near2far chunk carries MEEP component {int(chunk.c)}, which is not "
                    f"one of Ex/Ey/Ez/Hx/Hy/Hz."
                ])
            # A chunk MEEP created through a symmetry image (sn > 0) or a lattice
            # shift stores its `is`/`ie` in STORED-chunk coordinates; the sites it
            # serves live at `S.transform(iv, sn) + shift` (dft.cpp:942-943, the
            # same transform get_dft_array places them by). The accumulators here
            # hold the REQUESTED span, so the chunk's bounds are carried into user
            # space — sign flipped about doubled 0 on each axis sn flips, then the
            # shift — and the flipped axes are emitted in descending stored order.
            flips = _sn_flip_axes(mirrored_axes, int(getattr(chunk, "sn", 0) or 0))
            low, high = [], []
            for axis, direction in enumerate(directions):
                if direction is None:
                    low.append(None)
                    high.append(None)
                    continue
                stored_lo = int(chunk._is.in_direction(direction))
                stored_hi = int(chunk.ie.in_direction(direction))
                shift = int(chunk.shift.in_direction(direction))
                if axis in flips:
                    low.append(-stored_hi + shift)
                    high.append(-stored_lo + shift)
                else:
                    low.append(stored_lo + shift)
                    high.append(stored_hi + shift)
            image_weight = 1.0
            for axis, direction in image_axes:
                try:
                    # A single-site ladder takes s0 (IVEC_LOOP_WEIGHT1's loop_i <= 1
                    # branch); the one-cell axis can never hold more than one site.
                    image_weight *= float(chunk.s0.in_direction(direction))
                except Exception as unreadable:  # noqa: BLE001 - silence here doubles the surface.
                    raise MeepSimulationNotLiftable([
                        f"the near2far chunk for {component} does not expose its "
                        f"boundary weight s0 ({unreadable!r}); on a one-pixel periodic "
                        f"axis MEEP stores one chunk per lattice image and the weight "
                        f"is the only thing separating them, so the layout cannot be "
                        f"reproduced without it."
                    ]) from unreadable
            try:
                chunk_weight = complex(chunk.stored_weight)
            except Exception:  # noqa: BLE001 - an unreadable weight falls back to containment alone.
                chunk_weight = None
            # `vc` is MEEP's equivalent-source component c0 for the region that built
            # this chunk (dft.cpp:113, near2far.cpp:641-647). It names the region's
            # NORMAL, which is what tells a corner chunk's two claimants apart. Read
            # with an explicit sentinel, never `or`: MEEP's component enum starts at
            # Ex = 0 (vec.hpp), so a falsy vc is a real component.
            chunk_vc = getattr(chunk, "vc", None)
            chunk_source = None if chunk_vc is None else names.get(int(chunk_vc))
            parts.append(self._emit_chunk(component, low, high, int(chunk.N),
                                          image_weight=image_weight,
                                          stored_weight=chunk_weight, flips=flips,
                                          source_component=chunk_source))
            chunk = chunk.next_in_dft
        if not parts:
            return numpy.zeros(0, dtype=numpy.complex128)
        return numpy.concatenate(parts)

    def _emit_chunk(self, component, low, high, expected_sites, image_weight=1.0,
                    stored_weight=None, flips=(), source_component=None) -> numpy.ndarray:
        """This engine's samples for one MEEP chunk, in that chunk's own bounds.

        The accumulator is found by component, SOURCE COMPONENT, STORED WEIGHT and
        containment. Containment because one component appears once per near2far region
        and only the region that actually covers these sites may answer for them. The
        other two because containment alone is ambiguous at a flux-box CORNER: the
        corner site of antenna-radiation.py's box lies in BOTH the top face's region
        and the right face's, and MEEP keeps one chunk per region there. Matching the
        first containing accumulator emitted the corner with the other face's sign —
        measured: exactly 2.000e+00 relative on that one single-site chunk while every
        other chunk of the run sat at 1e-06.

        The weight alone does not separate them, and the way it fails is silent.
        Adjacent faces of a box carry OPPOSITE region weights, so at the corner shared
        by a ``weight=+1`` face and a ``weight=-1`` face the two chunks' stored weights
        are ``s1 * (+1)`` and ``s2 * (-1)`` with ``s1 = -s2`` — numerically the same
        number, differing only in THE SIGN OF THE ZERO IMAGINARY PART, e.g. ``(1+0j)``
        against ``(1-0j)``. ``abs(a - b)`` is then exactly ``0.0`` and the first
        containing accumulator answers for both, emitting one face's data twice.
        Measured on antenna-radiation.py: 1.778e-02 on the packed data against
        ``get_near2far_data``, and on ``test_antenna_radiation``'s double-mirror cell at
        resolution 50 a single one-site chunk — MEEP's own zero-weight ghost site just
        inside the right face — answered by the BOTTOM face at full weight, 7.47e-03
        of absolute error against MEEP's exact zero, for 2.69e-02 on the packed data
        and 1.40e-03 on the far field. :func:`_same_stored_weight` compares the parts,
        so ``+0.0`` and ``-0.0`` are the different labels MEEP means them to be, and
        :func:`_near2far_stored_weight` builds this engine's entry weights the way
        MEEP builds the chunk's so the two zeros agree.

        ``source_component`` is MEEP's own ``vc`` — the equivalent-source component
        ``c0``, which names the region's NORMAL (near2far.cpp:641-647) — and it
        separates the same corner without arithmetic: the shared Ez site is E_fd1 of
        the x-normal face (``c0 = Hy``) and E_fd0 of the y-normal face (``c0 = Hx``).
        Both filters are applied; either alone would resolve the measured cases, and
        together the corner cannot be misattributed by a weight collision OR by two
        faces that happen to share one. ``None`` on either (an unreadable weight, or a
        MEEP without ``vc``) drops just that filter.

        An axis whose bound is ``None`` is one the run does not store — phi on a `Dcyl`
        cell — and takes the accumulator's single plane. That plane is bookkeeping, not a
        coordinate: `__post_init__` gives an invariant axis one site at weight 1 with no
        ladder, exactly as MEEP's reduced loop has no such axis to index.

        An axis the engine stores as ONE cell takes that single plane too, whatever
        doubled coordinate the chunk reports there. Two cases, one rule: an INVARIANT
        axis of a 2-D run appears in the chunk's ivec with an unset slot (a D2 ivec
        has no Z), and a one-pixel PERIODIC axis's owned site sits at MEEP's own
        registration — the HIGH-face row for a shift-0 component (owned corner
        ``little + 2 - shift``), or a lattice image of it — which is the same stored
        plane under the wrap but not the same doubled coordinate this engine's
        ``index_for_doubled`` inverts. ``image_weight`` is that axis's per-image
        split weight, read off the chunk by :meth:`packed`.

        ``low``/``high`` arrive in USER coordinates (the caller has applied the
        chunk's symmetry transform and shift), so containment works unchanged for
        a chunk MEEP registered through a mirror image: the accumulator holds the
        requested span, below-plane sites included. ``flips`` are the sn-flipped
        axes; the chunk loops its STORED sites ascending, which is the requested
        span descending there, so those axes are emitted reversed
        (:meth:`~.dft.YeeRegionDFT.packed`'s ``flip``). The VALUES carry no extra
        sign: MEEP bakes ``S.phase_shift(c, sn)`` into the reflected chunk's scale
        (dft.cpp:169-171) and the accumulator's fold weights carry the same
        parity, so the two sides agree site for site — measured 1.7e-07..2.1e-07
        against ``get_near2far_data`` over discarded-half regions, a four-face box
        with two faces in discarded halves, a double odd mirror (sn = 3) and an
        odd stored count, with far fields from the loaded data at 2.6e-08..7.7e-07
        of MEEP's own (test_driver_vs_meep.py pins the cases).
        """
        single_plane = tuple(_axis_cell_count(self.entries[0][2].grid, axis) == 1
                             for axis in range(3)) if self.entries else (False,) * 3

        def _index(accumulator, axis, doubled):
            if doubled is None or single_plane[axis]:
                return accumulator.first_index(axis)
            return accumulator.index_for_doubled(axis, doubled)

        for name, weight, accumulator, source in self.entries:
            if name != component:
                continue
            if source_component is not None and source != source_component:
                continue
            if stored_weight is not None and not _same_stored_weight(complex(weight),
                                                                     stored_weight):
                continue
            try:
                lo = [_index(accumulator, axis, low[axis]) for axis in range(3)]
                hi = [_index(accumulator, axis, high[axis]) for axis in range(3)]
            except ValueError:
                continue
            starts = [accumulator.first_index(axis) for axis in range(3)]
            stops = [accumulator._slices[axis].stop for axis in range(3)]
            if any(lo[a] < starts[a] or hi[a] >= stops[a] for a in range(3)):
                continue
            sites = 1
            for axis in range(3):
                sites *= hi[axis] - lo[axis] + 1
            if sites != expected_sites:
                raise MeepSimulationNotLiftable([
                    f"near2far chunk for {component} spans {sites} sites by its bounds but "
                    f"MEEP reports N={expected_sites}; the index mapping disagrees with "
                    f"MEEP's own chunk, so the data would be laid out wrongly."
                ])
            box = tuple((lo[a] - starts[a], hi[a] - starts[a]) for a in range(3))
            return accumulator.packed(extra_scale=weight * image_weight, box=box,
                                      flip=tuple(flips))
        raise MeepSimulationNotLiftable([
            f"no migrated near2far region covers the {component} chunk at doubled bounds "
            f"{low}..{high} (user coordinates, sn flips {tuple(flips)}, stored weight "
            f"{stored_weight}, source component {source_component}); MEEP's chunk "
            f"decomposition reaches sites this lift did not accumulate."
        ])


def _resolve_region_normal(
    region_center, region_size, declared_direction, kind,
    cylindrical, invariant, nosize,
) -> int:
    """A flux-ish region's normal axis, MEEP's way: declared if given, else inferred.

    MEEP (near2far.cpp): ``nd = component_direction(w->c)`` — the direction the
    Python layer baked in from the region's ``direction`` — and only when that is
    ``NO_DIRECTION`` does ``fields::normal_direction(w->v)`` infer it from the
    volume. The inference is TWO stages (dft.cpp:806-824), both transcribed:

    1. ``volume::normal_direction()`` (vec.cpp:227-256): exactly one zero-extent
       axis among the RESOLVED ones — an invariant axis of a 2-D run cannot be a
       surface normal, because MEEP's reduced loop has no such axis at all.
    2. Only when that is still ``NO_DIRECTION``, the nosize pad: every axis that is
       a one-pixel periodic emulation of a whole dimension (``nosize_direction``,
       fields.cpp:730-737; one-cell directions are forced periodic at
       fields.cpp:80-85) and on which the region has zero extent is padded by 0.1 —
       i.e. removed from the flat set — and the inference is retried. That is what
       resolves ``Near2FarRegion(center=pt)`` on a 1-D-emulated ``(sx, 0, 0)`` cell
       to ``mp.X`` (measured: ``fields.normal_direction`` returned 0 == mp.X for
       binary_grating_n2f's point region). The residue stays refused, exactly where
       MEEP itself aborts ("Could not determine normal direction", dft.cpp:820-821).

    The stage order matters: a region with real extent whose only flat axis IS the
    nosize one resolves in stage 1 — MEEP pads only after direct inference fails.

    ``FluxRegion`` and ``Near2FarRegion`` are the SAME class upstream
    (python/simulation.py:568, ``Near2FarRegion = FluxRegion``) and MEEP resolves both
    through the same ``_add_fluxish_stuff`` line —
    ``d = self.fields.normal_direction(v) if d0 < 0 else d0`` — so one resolver serves
    both here too; ``kind`` only names the region in the refusal.

    The declared constant is read in the run's OWN coordinate system. MEEP's
    ``direction`` enum is ``X=0, Y, Z, R, P, NO_DIRECTION`` (vec.hpp:79), so a Dcyl
    region declares ``mp.R = 3`` — which is not an axis index at all, and which the
    Cartesian reading below silently sent to the zero-extent inference. On a
    cylindrical grid R is this engine's axis 0 and Z its axis 2; ``mp.P`` is refused,
    because the phi axis of a Dcyl cell is invariant and cannot carry a surface normal
    (MEEP would build the surface, but its own ``coordinate_mismatch`` rejects the
    resulting components), and X/Y do not exist there at all.

    ``invariant`` and ``nosize`` are per-axis predicates rather than a grid, so the
    gate (:func:`_check_outputs`) can run the SAME rule from declarations alone —
    the pre-flight verdict and the migration must be one decision, not two.
    """
    if cylindrical:
        if declared_direction == MEEP_R:
            return 0
        if declared_direction == MEEP_P:
            raise MeepSimulationNotLiftable([
                f"a {kind} on a cylindrical run declares direction=mp.P; phi is the "
                f"invariant axis of a Dcyl cell, so no surface has it as a normal. A "
                f"cylindrical surface is the r wall (mp.R) or a z cap (mp.Z)."
            ])
        if declared_direction == MEEP_Z:
            return 2
        if declared_direction in (MEEP_X, MEEP_Y):
            raise MeepSimulationNotLiftable([
                f"a {kind} on a cylindrical run declares direction="
                f"{'mp.X' if declared_direction == MEEP_X else 'mp.Y'}, which is not a "
                f"direction a Dcyl cell has; use mp.R or mp.Z."
            ])
    elif declared_direction in (0, 1, 2):
        return int(declared_direction)
    flat = [axis for axis in range(3)
            if float(region_size[axis]) == 0.0 and not invariant(axis)]
    if len(flat) == 1:
        return flat[0]
    padded = [axis for axis in flat if not nosize(axis)]
    if len(padded) == 1:
        return padded[0]
    raise MeepSimulationNotLiftable([
        f"a {kind} at center {tuple(region_center)} size {tuple(region_size)} "
        f"has {len(flat)} zero-extent resolved axes ({len(padded)} after MEEP's "
        f"one-pixel-emulation pad, dft.cpp:806-824), so its normal is ambiguous — "
        f"MEEP itself aborts here ('Could not determine normal direction'); "
        f"declare it with {kind}(direction=...)."
    ])


def _region_normal(grid, region_center, region_size, declared_direction, kind="Near2FarRegion") -> int:
    """:func:`_resolve_region_normal` with its predicates read off a built grid."""
    invariant = getattr(grid, "is_invariant", None)
    nosize = getattr(grid, "nosize_direction", None)
    return _resolve_region_normal(
        region_center, region_size, declared_direction, kind,
        cylindrical=bool(getattr(grid, "cylindrical", False)),
        invariant=lambda axis: bool(callable(invariant) and invariant(axis)),
        nosize=lambda axis: bool(callable(nosize) and nosize(axis)),
    )


def _gate_region_geometry(mp, sim, reasons: list[str]) -> None:  # The migration's region decisions, pre-flighted.
    """Resolve every flux/near2far region's normal — and a near2far region's bounds — at gate time.

    Both checks exist because their absence was a measured contract violation:
    ``gpu_compatibility`` said supported and ``lift_simulation`` then raised —

    * binary_grating_n2f.py's ``Near2FarRegion(center=pt)`` (a POINT region MEEP
      accepts, resolving its normal through the nosize pad) died in
      ``_region_normal``, which was called only from the migrations;
    * a near2far face wider than the cell (which MEEP silently clips to the sites
      it owns — measured: a face 3x the cell width gives far fields identical to
      the cell-wide face at 0.0e+00 rel L2) died in ``YeeRegionDFT.__post_init__``
      with a bare ValueError.

    The normal is resolved with the SAME rule the migration uses
    (:func:`_resolve_region_normal`), with the predicates computed from
    declarations: invariant axes from :func:`_invariant_axes`, one-pixel-emulation
    axes from :func:`_unit_cell_axes` (both already gate-safe). For a single-region
    ``DftFlux``, MEEP has already resolved and stored ``normal_direction``, exactly
    as ``_migrate_flux`` reads it; multi-region monitors carry it per region.

    The bounds pre-flight replays the migration's ladder per axis at BOTH Yee
    parities (the four components of a near2far region span both on every axis)
    and refuses, by name, a region whose ladder reaches past the stored grid or
    lies wholly below it — the cases the constructor raises for, with a
    mirror-folded axis replayed through the same reflected-gather feasibility the
    constructor uses (``dft.folded_axis_sites``). Skipped per axis where the
    constructor's own single-plane branches apply (invariant axes; zero-extent
    unit axes).
    """
    try:
        dimensions = _effective_dimensions(mp, sim)
        cylindrical = dimensions == MEEP_CYLINDRICAL or _is_cylindrical(sim)
        invariant_axes = set(_invariant_axes(dimensions))
        unit_axes = {
            axis for axis in _unit_cell_axes(sim, dimensions)
            if not (cylindrical and axis == 0)  # MEEP never forces R periodic (fields.cpp:82).
        }
        cell = _reduced_cell_size(sim, dimensions)
        resolution = float(sim.resolution)
    except Exception:  # noqa: BLE001 - an unreadable shape is already reported by the shape check.
        return
    counts = tuple(
        1 if axis in invariant_axes else meep_cell_count(cell[axis], resolution)
        for axis in range(3)
    )
    mirrored_axes = set()
    if not cylindrical:  # Grid refuses Cartesian mirrors on a Dcyl cell outright.
        for symmetry in getattr(sim, "symmetries", ()) or ():
            direction = getattr(symmetry, "direction", None)
            if direction is not None and int(direction) in (0, 1, 2):
                mirrored_axes.add(int(direction))
    # MEEP's boundary table is all-or-nothing off the declaration: use_bloch runs for
    # every direction under a plain `if self.k_point:` and everything else is Metallic
    # (the same reading `_lift_boundaries` documents). The fold replay needs it
    # because a folded periodic axis stores the far plane and wraps its window while
    # a folded metallic one terminates on the wall.
    metallic = not bool(getattr(sim, "k_point", None))
    for monitor in list(getattr(sim, "dft_objects", ()) or ()):
        kind = type(monitor).__name__
        region_kind = {
            "DftNear2Far": "Near2FarRegion", "DftFlux": "FluxRegion",
            "DftForce": "ForceRegion", "DftEnergy": "EnergyRegion",
        }.get(kind)
        if region_kind is None:
            continue
        # A DftForce carries two refusals that are properties of the RUN rather than
        # of any one region, and both belong here so `gpu_compatibility` reports them
        # instead of `lift_simulation` raising after it said supported. See
        # `_migrate_force` for the source behind each.
        if kind == "DftForce" and (mirrored_axes or cylindrical):
            reasons.append(
                "a DftForce monitor on a run that declares "
                + ("a cylindrical cell" if cylindrical else "a symmetry")
                + ": MEEP "
                + ("loops Z, R, P over the field directions there (vec.hpp:147-149) and "
                   "the diagonal terms pick up the 2*pi*r ring measure"
                   if cylindrical else
                   "reduces the force region list through symmetry::reduce "
                   "(stress.cpp:161, vec.cpp:1416-1470) before registering a chunk")
                + ", which is not reproduced here and has no measured case."
            )
            continue
        for region in list(getattr(monitor, "regions", ()) or ()):
            center = _vector3(region.center)
            # |size| for the same reason the migration takes it — MEEP's volume sorts
            # its corners, so a negative component names the same span
            # (:func:`_meep_region_extents`). The normal rule below only tests for a
            # zero extent and so is sign-blind either way; the absolute value is here
            # to keep the gate's reading of a region and the migration's ONE reading.
            size = tuple(abs(value) for value in _vector3(region.size))
            # Read the REGION's declaration, never `monitor.normal_direction`: DftObj
            # is lazy, and touching any of its properties calls the real `add_flux`,
            # which "immediately initialize[s] the structure and fields" (DftObj's own
            # docstring) — after which the lift's initialized-simulation guard refuses
            # a run it accepted yesterday (measured: 15 previously-lifting corpus
            # scripts refused). For a single region the stored normal IS this rule's
            # answer — MEEP resolves it per region in `_add_fluxish_stuff`, declared
            # direction else the padded inference — so nothing is lost by not asking.
            raw_direction = getattr(region, "direction", -1)
            declared = int(-1 if raw_direction is None else raw_direction)  # 0 is mp.X, not "unset".
            resolver = functools.partial(
                _resolve_region_normal,
                cylindrical=cylindrical,
                invariant=lambda axis: axis in invariant_axes,
                nosize=lambda axis: axis in unit_axes,
            )
            try:
                if kind == "DftForce":
                    # TWO resolutions, MEEP's two: fd from the declaration (falling
                    # back to the inference) and nd always from the geometry. Only
                    # the diagonal branch is reproduced, so they must agree — see
                    # `_force_region_axes`, which is the same rule the migration runs.
                    force_direction, normal = _force_region_axes(
                        center, size, declared, resolver,
                    )
                    if force_direction != normal:
                        reasons.append(
                            f"a ForceRegion at center {tuple(center)} declares direction "
                            f"{('x', 'y', 'z')[force_direction]} on a surface whose normal "
                            f"is {('x', 'y', 'z')[normal]}, which takes MEEP's OFF-DIAGONAL "
                            f"stress-tensor branch (stress.cpp:168-177); that branch has no "
                            f"case in MEEP's own tests or examples and so no oracle, and is "
                            f"not reproduced here."
                        )
                        continue
                else:
                    resolver(center, size, declared, region_kind)
            except MeepSimulationNotLiftable as refusal:
                reasons.extend(refusal.reasons)
                continue
            if kind != "DftNear2Far":
                continue
            _gate_near2far_bounds(
                center, size, resolution, counts, cylindrical,
                invariant_axes, unit_axes, mirrored_axes, metallic, reasons,
            )


def _gate_near2far_bounds(
    center, size, resolution, counts, cylindrical, invariant_axes, unit_axes,
    mirrored_axes, metallic, reasons: list[str],
) -> None:
    """Replay ``YeeRegionDFT.__post_init__``'s ladder feasibility from declarations.

    The constructor clips a ladder's LOW overhang to MEEP's owned corner (that is
    MEEP's own loop, ``iscS = max(is - shifti, iscoS)``, loop_in_chunks.cpp — the
    lift accepts it) and raises for the two remaining cases: a ladder wholly below
    the owned sites, and one reaching past the stored high face. Both are named
    here so the gate's verdict and the lift agree. MEEP itself never refuses — on
    a non-periodic axis it clips the high side too (``iecS = min(ie - shifti,
    iecoS)``; measured: a face 3x the cell width matches the cell-wide face at
    0.0e+00) and on a periodic axis it wraps onto lattice images
    (loop_in_chunks.cpp:390-421) — but this engine's raw-Yee accumulator gathers
    neither the boundary row MEEP owns past the stored interior nor the wrapped
    images, so the honest verdict is a refusal with the workaround named.

    A MIRROR-FOLDED axis replays the constructor's other branch instead: the
    reflected gather (``dft.folded_axis_sites``, MEEP's use_symmetry=true loop)
    serves sites below the owned corner from their stored images and clips only
    what no image reaches, so the one refusal left there is a region the fold
    serves NOTHING of. ``counts`` stay the FULL cell counts; the folded stored
    count is derived here exactly as ``Grid`` derives it (``halve()`` plus the
    far-plane cell a periodic even fold stores), so the verdict and the
    registration are one decision.
    """
    for axis in range(3):
        if axis in invariant_axes:
            continue
        low = center[axis] - 0.5 * abs(size[axis])
        high = center[axis] + 0.5 * abs(size[axis])
        if axis in unit_axes and high == low:
            continue  # The constructor's single-plane branch; no ladder to check.
        stored = counts[axis]
        mirrored = axis in mirrored_axes
        if mirrored:
            n_full = counts[axis]
            stored = n_full - n_full // 2 + 1 + (
                1 if (not metallic and n_full % 2 == 0) else 0
            )
        origin = (0 if (cylindrical and axis == 0)
                  else -2 if mirrored
                  else -(stored - stored % 2))
        for parity in (0, 1):
            is_doubled, ie_doubled, _ = _boundary_weight_ladder(
                low, high, resolution, parity=parity
            )
            count = (ie_doubled - is_doubled) // 2 + 1
            start = (is_doubled - origin - parity) // 2
            first_owned = 0 if (cylindrical and axis == 0) else 1 - parity
            if mirrored and start < first_owned:
                _, indices, _ = folded_axis_sites(
                    start, count, parity,
                    n_full=counts[axis], stored=stored, metallic=metallic,
                )
                if indices.size == 0:
                    reasons.append(
                        f"a Near2FarRegion spanning [{low:g}, {high:g}] on the folded "
                        f"{AXIS_NAMES[axis]} axis lies wholly outside what the fold "
                        f"serves: no stored row, reflected image or lattice image "
                        f"reaches it."
                    )
                    break
                continue
            dropped = max(0, first_owned - start)
            if dropped >= count:
                reasons.append(
                    f"a Near2FarRegion spanning [{low:g}, {high:g}] on the "
                    f"{AXIS_NAMES[axis]} axis lies wholly below the sites MEEP owns "
                    f"there; there is no surface for it to accumulate."
                )
                break
            # The low clip moves the start up by exactly what it drops, so the
            # ladder's top is start + count whether or not anything was clipped.
            if start + count > stored:
                reasons.append(
                    f"a Near2FarRegion spanning [{low:g}, {high:g}] on the "
                    f"{AXIS_NAMES[axis]} axis reaches past the stored grid "
                    f"({stored} cells). MEEP accepts this — it clips a non-periodic "
                    f"axis to the sites it owns (loop_in_chunks' iecS clamp; measured "
                    f"identical far fields at 0.0e+00) and wraps a periodic one onto "
                    f"lattice images — but this engine's near2far accumulator has "
                    f"neither gather yet. Shrink the region to the cell interior."
                )
                break


def _check_outputs(sim, reasons: list[str]) -> None:  # Monitors MEEP owns that the lift cannot rebuild.
    """Refuse only the monitor kinds this engine cannot reproduce.

    A MEEP monitor object is bound to MEEP's own field arrays and accumulates inside
    MEEP's step loop. That loop never runs on a lift, so an un-migrated monitor ends
    the run EMPTY — a correctly shaped, correctly typed spectrum of zeros, which is
    the failure this package refuses to hand back.

    The DFT itself was never the missing piece: it is this engine's, it runs in the
    per-timestep loop, and it runs on the GPU. What was missing was carrying the
    *specification* across, which is mechanical — MEEP's object publishes the exact
    frequency list, the region and the components. :func:`migrate_monitors` does that
    now, so every kind in :data:`MIGRATABLE_MONITORS` — flux, fields, near2far, force
    and energy — is lifted rather than refused.

    Everything else still refuses BY NAME, as do the OPTIONS a migratable kind can
    carry that this engine does not reproduce (an off-diagonal force branch, a force
    under a symmetry). Quietly dropping one would reproduce exactly the
    spectrum-of-zeros this check exists to prevent.
    """
    monitors = list(getattr(sim, "dft_objects", ()) or ())
    # A region in the half a mirror fold discards is NOT refused here any more: MEEP
    # accumulates DFT monitors through the symmetry (dft.cpp:231 takes
    # loop_in_chunks' use_symmetry=true default, where sources.cpp:487 passes
    # false), and the engine now serves them the same way — the reflected gather
    # (`dft.folded_axis_sites`) registered by `_register_volume` and
    # `YeeRegionDFT`. The refusal that stood here cost five corpus scripts a lift
    # (absorbed_power_density, antenna-radiation, antenna_pec_ground_plane,
    # cavity-farfield, mie_scattering).
    _gate_region_geometry(_import_meep(), sim, reasons)
    # Options a migratable KIND can carry that this engine still does not reproduce.
    # They belong here, in the pre-flight, and not only in the migration: a check that
    # lives solely in `_migrate_near2far` lets `gpu_compatibility` report the script as
    # supported and then kills the lift with a bare error, which is the one outcome
    # this pre-flight exists to prevent (the same slip was found on 1-D cells, on
    # set_boundary, and on Bloch-plus-PML — see the parity matrix's findings).
    for monitor in monitors:
        _check_monitor_frequencies(monitor, reasons)
        if type(monitor).__name__ == "DftFields":
            _check_dft_fields_components(monitor, reasons)
        if type(monitor).__name__ != "DftNear2Far":
            continue
        periods = int(getattr(monitor, "nperiods", 1) or 1)
        if periods != 1:
            reasons.append(
                f"add_near2far(..., nperiods={periods}) replicates the near surface over "
                f"lattice periods when the far field is evaluated, which changes what the "
                f"stored near data means; only nperiods=1 is reproduced here."
            )
    unsupported = [m for m in monitors if type(m).__name__ not in MIGRATABLE_MONITORS]
    if unsupported:
        kinds = sorted({type(m).__name__ for m in unsupported})
        reasons.append(
            f"this simulation carries {len(unsupported)} DFT monitor(s) ({kinds}) this engine "
            f"cannot rebuild; only {sorted(MIGRATABLE_MONITORS)} are migrated. Their accumulation "
            f"is not implemented here, and a lifted run would finish with every one of them empty "
            f"— a full spectrum of zeros rather than an error. Either drop them before lifting and "
            f"evaluate them on CPU MEEP, or keep the run on CPU MEEP."
        )


def _check_dft_fields_components(monitor, reasons: list[str]) -> None:
    """Pre-flight a DftFields component list against what the migration accumulates.

    The migration takes the twelve stored field components (E/D/H/B, Cartesian or
    cylindrical) and raises for anything else — MEEP's ``add_dft_fields`` also
    accepts ``mp.Dielectric`` and the derived components, which have no stored
    array here. Named at the gate so the verdict and the lift are one decision.
    """
    mp = _import_meep()
    args = getattr(monitor, "args", None)
    if not isinstance(args, (list, tuple)) or not args:
        return
    try:
        requested = [int(component) for component in args[0]]
    except (TypeError, ValueError):
        return  # A shifted signature; the migration's own validation still stands.
    names = _monitor_component_names(mp, False)
    names.update(_monitor_component_names(mp, True))  # Either coordinate family passes.
    unknown = [component for component in requested if component not in names]
    if unknown:
        labels = ", ".join(str(mp.component_name(component)) for component in unknown)
        reasons.append(
            f"a DftFields monitor requests component(s) [{labels}], which are not among "
            f"the twelve stored field components (E/D/H/B); derived or material "
            f"components are not accumulated here."
        )


def _monitor_region(monitor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The (center, size) MEEP recorded for a monitor, from its own region object.

    Read from ``regions[0]`` rather than re-derived from ``where``: ``where`` is a
    ``meep::volume`` in MEEP's internal coordinates whose corners have already been
    snapped to the grid, so rebuilding a center and size from it would hand the driver
    a region a half cell away from the one the user asked for. The region object is
    what the user passed.
    """
    regions = list(getattr(monitor, "regions", ()) or ())
    if not regions:
        raise MeepSimulationNotLiftable([
            f"a {type(monitor).__name__} monitor carries no region object, so the volume it "
            f"covers cannot be read back and rebuilt."
        ])
    region = regions[0]
    return _vector3(region.center), _vector3(region.size)


# Where each monitor class keeps its requested decimation inside ``args``. MEEP does
# not expose it as a named attribute — DftFlux.__init__ only lifts args[0] and args[1]
# onto the object — so a positional read is the only route, and MEEP's own classes
# index args positionally for the same reason. Guarded, and pinned by a test, because
# a wrong read here does NOT raise: it shifts the accumulated spectrum by ~1.4e-06,
# which is small enough to pass an eyeball and wrong enough to fail parity.
#   DftFlux  : add_flux(fcen, df, nfreq, freq, FluxRegions, decimation_factor=0)
#   DftFields: add_dft_fields(cs, fcen, df, nfreq, freq, where, center, size,
#              yee_grid, decimation_factor, persist)
#   DftNear2Far: add_near2far(fcen, df, nfreq, freq, Near2FarRegions, nperiods,
#                decimation_factor)  -> args = [freq, nperiods, regions, decimation]
#   DftForce:  add_force(fcen, df, nfreq, freq, ForceRegions, decimation_factor)
#              -> DftForce(self._add_force, [freq, forces, decimation_factor])
#              (python/simulation.py:3392-3399)
#   DftEnergy: add_energy(...) -> DftEnergy(self._add_energy, [freq, energys,
#              decimation_factor]) (python/simulation.py:3134-3137)
# Both carry DftFlux's own three-slot shape, hence the same index.
_DECIMATION_ARG_INDEX: dict[str, int] = {
    "DftFlux": 2, "DftFields": 6, "DftNear2Far": 3, "DftForce": 2, "DftEnergy": 2,
}
# `add_mode_monitor` builds a DftFlux with `yee_grid` inserted before the decimation
# (simulation.py:3531-3534), so the same class carries the factor one slot further
# along. See `_is_mode_monitor`.
_MODE_MONITOR_DECIMATION_ARG_INDEX = 3


def _is_mode_monitor(monitor) -> bool:
    """Was this ``DftFlux`` built by ``add_mode_monitor`` rather than ``add_flux``?

    The two build the SAME class with DIFFERENT argument lists —
    ``DftFlux(self._add_flux, [freq, fluxes, decimation_factor])``
    (simulation.py:3510) against
    ``DftFlux(self._add_mode_monitor, [freq, fluxes, yee_grid, decimation_factor])``
    (simulation.py:3531-3534) — so every positional read of ``args`` past index 1
    has to know which one it is holding. ``DftObj`` records the bound method it will
    call as ``func`` (simulation.py:651-655), which is the discriminator MEEP itself
    keeps; the argument COUNT would work too but would silently agree with a future
    signature change instead of failing.
    """
    return getattr(getattr(monitor, "func", None), "__name__", "") == "_add_mode_monitor"


def _monitor_decimation(monitor) -> int:
    """The decimation MEEP was ASKED for; 0 means automatic and is MEEP's default.

    Zero is forwarded as zero rather than turned into 1: this driver implements the
    same source+monitor bandwidth rule, so 0 reproduces whatever MEEP resolved, while
    forcing 1 makes the engine accumulate on steps MEEP skipped.

    ``add_mode_monitor`` puts ``yee_grid`` where ``add_flux`` puts the decimation, so
    its slot is one further along (:func:`_is_mode_monitor`). Reading index 2 for
    both landed on the ``yee_grid`` BOOL, which the type guard below turned into 0 —
    MEEP's default, and therefore right whenever the script left the factor alone and
    silently wrong the moment it did not. The guard is kept, because it is what makes
    a future signature move degrade to MEEP's default rather than to ``int(False)``,
    but it is no longer what this case relies on.
    """
    kind = type(monitor).__name__
    index = _DECIMATION_ARG_INDEX.get(kind)
    if kind == "DftFlux" and _is_mode_monitor(monitor):
        index = _MODE_MONITOR_DECIMATION_ARG_INDEX
    args = list(getattr(monitor, "args", ()) or ())
    if index is None or index >= len(args):
        return 0
    value = args[index]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0  # A shifted signature landed something else here; take MEEP's default.
    return int(value)


# Where each monitor class keeps its resolved frequency list inside ``args`` —
# python/simulation.py binds ``nfreqs = len(args[0])`` for DftFlux (:693) and
# DftNear2Far (:768) and ``len(args[4])`` for DftFields (:842). Positional for the
# same reason as ``_DECIMATION_ARG_INDEX``: the object's ``freq`` PROPERTY is lazy
# (``swigobj_attr``) and touching it initializes the structure and fields, after
# which the initialized-simulation guard refuses a run the gate accepted.
_FREQUENCY_ARG_INDEX: dict[str, int] = {
    "DftFlux": 0, "DftFields": 4, "DftNear2Far": 0, "DftForce": 0, "DftEnergy": 0,
}


def _check_monitor_frequencies(monitor, reasons: list[str]) -> None:
    """Pre-flight a migratable monitor's declared frequencies against the DFT's rule.

    The accumulator (``dft.normalize_frequencies``) takes any FINITE frequency —
    negative is MEEP's conjugate accumulator, ``omega`` stored raw at dft.cpp:219-221
    and ``abs()``-ed only for the decimation bound at dft.cpp:201-206, which is how
    the minus-frequency near2far of dipole_in_vacuum_cyl_off_axis.py migrates, and
    zero is the running time integral (``polar(1.0, 0) == 1``, dft.cpp:266-269, no
    branch on omega anywhere in ``update_dft``) — but a non-finite entry raises there,
    at migration time, after ``gpu_compatibility`` said supported. Naming it here
    keeps the gate's verdict and the lift's one decision.
    """
    kind = type(monitor).__name__
    index = _FREQUENCY_ARG_INDEX.get(kind)
    args = getattr(monitor, "args", None)
    if index is None or not isinstance(args, (list, tuple)) or len(args) <= index:
        return  # Not a migratable kind, or a signature this table does not know.
    try:
        declared = [float(value) for value in args[index]]
    except (TypeError, ValueError):
        return  # A shifted signature; the migration's own validation still stands.
    bad = [value for value in declared if not math.isfinite(value)]
    if bad:
        reasons.append(
            f"a {kind} monitor asks for frequency {bad[0]!r}; the DFT accumulator "
            f"takes any finite frequency (zero is the running time integral, negative "
            f"is the conjugate phase), but a non-finite one is not a frequency."
        )


def _resolved_decimation(monitor) -> int:
    """The decimation MEEP actually settled on, read off its own chunk; 0 if unavailable.

    ``args`` carries what the SCRIPT asked for, and 0 there means "automatic" rather
    than a factor. The chunk carries what MEEP resolved that to, and re-deriving the
    rule is not reliable: MEEP's documented ``floor(1/(dt*(freq_max+src_freq_max)))``
    uses ``src_time::get_fwidth()``, which is not the nominal fwidth — measured on a
    2x2 probe, MEEP resolved 8 where the nominal-fwidth rule gives 9, and the
    monitors then accumulate on different steps with different quadrature weights
    (5-20 % apart on that probe while the stepped fields agreed at 1e-06). Every
    migratable kind keeps its list under its own name — ``F`` on a DftNear2Far,
    ``E`` (and ``H``) on a DftFlux, ``E``/``D``/``H``/``B`` on a DftEnergy,
    ``diag``/``offdiag1``/``offdiag2`` on a DftForce, ``chunks`` on a DftFields —
    so each is tried. Returns 0 when no chunk list exists yet, leaving the caller on
    the requested value.
    """
    swigobj = getattr(monitor, "swigobj", None)
    for attribute in ("F", "E", "diag", "chunks"):
        head = getattr(swigobj, attribute, None)
        reader = getattr(head, "get_decimation_factor", None)
        if reader is None:
            continue
        try:
            return int(reader())
        except Exception:  # noqa: BLE001 - an unreadable chunk is not a reason to fail the lift.
            continue
    return 0


def _grid_mirror_axes(grid) -> tuple[int, ...]:
    """The grid's mirror planes' axis indices, in DECLARATION order.

    Declaration order is what ``symmetry::transform`` decomposes ``sn`` by: the
    python layer chains ``sym = sym + mirror(s.direction, gv) * phase`` over
    ``self.symmetries`` in order (python/simulation.py ``_create_symmetries``),
    ``operator+`` appends to the chain's tail, and ``transform(d, n)`` peels
    ``n % g`` for the head and ``n / g`` for the rest (vec.cpp:1251-1284) — so
    bit i of ``sn`` is the i-th DECLARED plane. Measured on
    ``symmetries=[Mirror(X), Mirror(Y)]``: the chunk for a region reflected in
    both carries sn=3, in Y alone sn=2. ``Grid.symmetry`` preserves the declared
    order (``_normalize_mirrors`` appends), so it is read here.
    """
    planes = getattr(grid, "symmetry", ()) or ()
    axes = []
    for plane in planes:
        axis_index = getattr(plane, "axis_index", None)
        if axis_index is None and getattr(plane, "axis", None) in ("X", "Y", "Z"):
            axis_index = ("X", "Y", "Z").index(plane.axis)
        if axis_index is not None:
            axes.append(int(axis_index))
    return tuple(axes)


def _sn_flip_axes(mirror_axes: tuple[int, ...], sn: int) -> tuple[int, ...]:
    """Which axes a chunk's symmetry index flips — sn's bits over the declared planes."""
    return tuple(
        axis for bit, axis in enumerate(mirror_axes) if (sn >> bit) & 1
    )


def _meep_chunk_weight(chunk, component: str, directions) -> MeepChunkWeight:
    """One live ``dft_chunk``'s integration weight, read rather than modelled.

    Every member here is PUBLIC on MEEP's own ``dft_chunk`` (src/meep.hpp:1196-1205)
    and reachable through the SWIG wrapper, so the weight
    ``process_dft_component`` divides back out (dft.cpp:1003-1006) can be reproduced
    exactly instead of re-derived from the region this engine registered. The two
    derivations agree on the requested region and part company on a ``S.reduce``'d
    one, whose halved extent has new edges — see :class:`~.dft.MeepChunkWeight`.

    ``dV1`` is refused rather than approximated: it is the cylindrical measure's
    radial term, ``dV0 + dV1 * loop_i2`` (dft.cpp:996), and a Dcyl flux monitor is
    already refused a few frames up because its chunks name R and Z.
    """
    try:
        vectors = {name: tuple(float(getattr(chunk, name).in_direction(d)) for d in directions)
                   for name in ("s0", "s1", "e0", "e1")}
        dV0 = float(chunk.dV0)
        dV1 = float(chunk.dV1)
        include = bool(chunk.include_dV_and_interp_weights)
        sqrt_weights = bool(chunk.sqrt_dV_and_interp_weights)
    except Exception:  # noqa: BLE001 - an unreadable weight is a refusal, never a guess.
        raise MeepSimulationNotLiftable([
            f"the {component} flux chunk does not expose its integration weight "
            f"(s0/s1/e0/e1/dV0/dV1); MEEP divides the stored value by exactly "
            f"IVEC_LOOP_WEIGHT of those when it reads it back (dft.cpp:1003-1006), so "
            f"the layout cannot be reproduced without them."
        ]) from None
    if dV1 != 0.0:
        raise MeepSimulationNotLiftable([
            f"the {component} flux chunk carries dV1={dV1}, MEEP's radial measure term "
            f"`dV0 + dV1*loop_i2` (dft.cpp:996); that is a cylindrical chunk and this "
            f"packer lays out a Cartesian plane."
        ])
    return MeepChunkWeight(s0=vectors["s0"], s1=vectors["s1"], e0=vectors["e0"],
                           e1=vectors["e1"], dV0=dV0, include=include,
                           sqrt_weights=sqrt_weights)


def _meep_region_extents(grid, size) -> tuple[float, float, float]:
    """A declared monitor-region size AS MEEP READS IT. Two rules, one home.

    **A NEGATIVE component is its own absolute value.** MEEP's Python layer builds a
    region's volume as ``mp.volume(center - size/2, center + size/2)``
    (simulation.py:443-449) and the C++ constructor SORTS the corners —
    ``min_corner = min(vec1, vec2); max_corner = max(vec1, vec2)`` (vec.cpp:163-167) —
    so ``size.x = -10.4`` is the same volume as ``+10.4`` about the same centre and
    MEEP never sees a reversed bound. Scripts write it: MEEP's own
    ``test_cavity_farfield.py`` gives a ``Near2FarRegion`` ``size=2*dpml - sx``, which
    is ``-10.4``. Passing the sign through instead reached this engine's monitors as
    an inverted span and raised — an accepted-by-the-gate simulation that could not
    lift, for a spelling MEEP treats as ordinary.

    **An INVARIANT axis's extent is zero.** A reduced ``grid_volume`` has no such
    direction, so the ``volume`` MEEP builds carries the resolved axes alone and a
    declared extent there has nowhere to be written (vol2d takes x and y only).
    Measured: a 2-D flux region with a 0.44 z extent returns a flux identical to the
    flat region's at **0.000e+00**, and the same for a dft_fields region's array. The
    scripts that write it are not misusing MEEP — ``coupler.py``'s port volumes come
    from ``GDSII_vol(zmin, zmax)`` with the real device thickness, reused unchanged
    for the 2-D variant.

    Both are MEEP's leniency reproduced for a MIGRATED region, exactly as
    ``_reduced_cell_size`` does for ``cell_size``. The driver's own guards
    (``_require_region_flat_on_invariant_axes``, and the non-negative-size checks on
    ``add_flux_monitor`` / ``add_dft_monitor``) keep refusing the DIRECT API spelling,
    where no MEEP semantics stand behind either.
    """
    is_invariant = getattr(grid, "is_invariant", None)
    if not callable(is_invariant):
        return tuple(abs(float(value)) for value in size)
    return tuple(
        0.0 if is_invariant(axis) else abs(float(size[axis])) for axis in range(3)
    )


# MEEP component integers -> monitor component names: the six current-carrying
# components plus D and B, which add_dft_fields accepts and this engine stores
# (absorbed_power_density.py monitors [Dz, Ez]). Deliberately SEPARATE from
# `_component_names`, which also vets SOURCE components — a D "source" is not a
# current the stepper can inject, and widening that map would accept one.
def _monitor_component_names(mp, cylindrical: bool = False) -> dict:
    names = dict(_component_names(mp, cylindrical))
    if cylindrical:
        names.update({
            mp.Dr: "Dx", mp.Dp: "Dy", mp.Dz: "Dz",
            mp.Br: "Bx", mp.Bp: "By", mp.Bz: "Bz",
        })
    else:
        names.update({
            mp.Dx: "Dx", mp.Dy: "Dy", mp.Dz: "Dz",
            mp.Bx: "Bx", mp.By: "By", mp.Bz: "Bz",
        })
    return names


@dataclass
class DftFieldsYeeMigration:
    """A ``yee_grid=True`` DftFields rebuilt as one raw-Yee accumulator per component.

    ``add_dft_fields(..., yee_grid=True)`` is ``use_centered_grid=false`` — each
    component's chunk is registered on its OWN Yee lattice with no cell-centre
    averaging (dft.cpp:231 passes the component itself as cgrid), which is exactly
    :class:`~.dft.YeeRegionDFT`'s registration; the centred :class:`~.dft.DFTMonitor`
    cannot stand in for it. ``get_dft_spectrum`` returns the raw accumulated block
    in the accumulator's own (nf, cx, cy, cz) layout — the values
    ``get_dft_array`` reports before its empty-dimension collapse.
    """

    accumulators: dict  # {engine component name: YeeRegionDFT}
    frequencies: tuple

    def get_dft_spectrum(self, component: str):
        accumulator = self.accumulators.get(component)
        if accumulator is None:
            raise ValueError(
                f"Component '{component}' is not monitored. Monitored: "
                f"{', '.join(sorted(self.accumulators))}"
            )
        return accumulator._dft

    def get_dft(self, component: str, freq_index: int = 0):
        return self.get_dft_spectrum(component)[int(freq_index)]


@dataclass
class FluxMigration:
    """A ``sim.add_flux`` monitor whose several regions are rebuilt as several planes.

    ``sim.add_flux(fcen, df, nfreq, *FluxRegions)`` takes a LIST of regions — the flux-box
    idiom — and ``mp.get_fluxes`` returns their weighted sum, because MEEP appends every
    region's chunks to one ``dft_flux`` and ``dft_flux::flux()`` sums the whole chunk list
    (dft.cpp:533-556), with each region's ``weight`` folded into its own E chunks
    (``where->weight * (1 - 2i)``, dft.cpp:629).

    A single driver flux monitor cannot stand in for that, and the failure was neither
    quiet nor early: MEEP stores ``NO_DIRECTION`` as the monitor's ``normal_direction``
    for exactly the multi-region case ("if the volume list has > 1 entry, store
    NO_DIRECTION", dft.cpp:639-641), so the lift reached ``add_flux_monitor`` and died
    on ``Unknown flux direction 5`` after `gpu_compatibility` had already said supported.
    Each region therefore gets its own plane, resolved through :func:`_region_normal`
    exactly as MEEP resolves it per region, and this object sums them back.

    A region ``weight`` with an imaginary part is refused rather than folded in: MEEP
    scales the E chunk by the complex weight and the flux is ``real(conj(E) * H)``, so a
    complex weight does not factor out of the sum the way a real one does.
    """

    parts: tuple  # ((weight, FluxMonitor), ...) in the script's own region order.
    frequencies: tuple

    def get_flux_spectrum(self):
        """The weighted sum over regions — the counterpart of ``mp.get_fluxes(flux)``."""
        total = numpy.zeros(len(self.frequencies), dtype=numpy.float64)
        for weight, monitor in self.parts:
            total += weight * numpy.asarray(monitor.get_flux_spectrum(), dtype=numpy.float64)
        return total

    def get_dft_data(self) -> FluxDftData:
        """Every region's transforms, in ``parts`` order — the script's region order.

        Save and load both walk ``parts``, so a plane is always paired with its own
        data. MEEP's flat ``FluxData`` orders the same content differently — its
        chunk list is built by PREPENDING (``add_dft_flux`` threads the running head
        back in as ``chunk_next``, dft.cpp:627-631, and every new chunk takes it as
        ``next_in_dft``, dft.cpp:128) so the last region registered comes first —
        and that order does not have to be matched here, because
        this container never crosses into MEEP. What DOES have to hold is that the
        two ends agree, which ``_load_flux_planes`` checks plane by plane.
        """
        return FluxDftData(planes=tuple(
            record for _weight, monitor in self.parts
            for record in monitor.get_dft_data().planes
        ))

    def scale_dfts(self, scale: complex) -> None:  # dft_flux::scale_dfts over every region.
        for _weight, monitor in self.parts:
            monitor.scale_dfts(scale)

    def load_dft_data(self, data: FluxDftData) -> None:  # MEEP's load_flux_data, region by region.
        _load_flux_planes(data, tuple(monitor for _weight, monitor in self.parts),
                          "FluxMigration.load_dft_data")

    def load_minus_dft_data(self, data: FluxDftData) -> None:
        """MEEP's ``load_minus_flux_data``; see :meth:`~.dft.FluxMonitor.load_minus_dft_data`.

        The region WEIGHTS are not applied here and must not be. MEEP folds each
        region's weight into that region's own E chunks at registration
        (dft.cpp:629), so its saved transform carries it; this engine applies the
        weight at READ time instead (:meth:`get_flux_spectrum`), so its saved
        transform does not. Either way the negation is the plain ``scale_dfts(-1)``
        of dft.cpp:588 and the two agree, because a REAL weight factors straight out
        of ``Re[(w*E2 - w*E1) x conj(H2 - H1)]`` — which is exactly why
        :func:`_migrate_flux` refuses a complex one.
        """
        self.load_dft_data(data)
        self.scale_dfts(-1.0)


def _migrate_flux(monitor, driver, frequencies, decimation) -> Any:
    """Rebuild a ``DftFlux`` on the driver: one plane per region, summed at read time.

    A monitor that is ONE region at the default weight — the common case, and every case
    the flux path was built and measured against — is migrated to a bare driver monitor
    exactly as before, so nothing about it changes. A second region, or a region weight
    that is not 1, makes it a :class:`FluxMigration`, because ``driver.add_flux_monitor``
    has no weight of its own: MEEP folds ``where->weight`` into that region's E chunks
    (dft.cpp:629) and a bare monitor would return ``+flux`` where MEEP returns ``-flux``
    for the ``weight=-1`` face of a flux box (measured: exactly 2.0 relative).

    The per-region normal is resolved the way MEEP resolves it (``_add_fluxish_stuff``:
    the region's declared direction, else ``fields::normal_direction`` on its volume)
    rather than read off ``monitor.normal_direction``, which is the whole monitor's and is
    ``NO_DIRECTION`` as soon as there are two regions.
    """
    regions = list(getattr(monitor, "regions", ()) or ())
    if not regions:
        raise MeepSimulationNotLiftable([
            "a DftFlux monitor carries no region object, so the plane it measures cannot "
            "be read back and rebuilt."
        ])
    # Prefer the factor MEEP RESOLVED over the requested one, exactly as the
    # near2far migration does: `decimation=0` says only "automatic", and the
    # nominal-fwidth re-derivation disagrees with MEEP's own get_fwidth-based one
    # (measured 8 vs 9 on the flux probe; see `_resolved_decimation`).
    resolved = _resolved_decimation(monitor)
    if resolved:
        decimation = resolved
    if len(regions) == 1 and complex(getattr(regions[0], "weight", 1.0) or 1.0) == 1.0:
        center, size = _monitor_region(monitor)
        size = _meep_region_extents(driver.grid, size)
        # normal_direction is MEEP's own resolved axis for the plane; passing it rather
        # than inferring from the zero extent keeps a monitor whose region is flat on more
        # than one axis from being guessed at. On a Dcyl run it is mp.R / mp.Z, which is
        # not an axis index, so it goes through the same resolver as a declared direction.
        declared = int(monitor.normal_direction)
        normal = _region_normal(driver.grid, center, size, declared, kind="FluxRegion")
        return driver.add_flux_monitor(
            frequencies=frequencies, center=center, size=size,
            direction=normal, decimation_factor=decimation,
        )
    parts = []
    for region in regions:
        center = _vector3(region.center)
        size = _meep_region_extents(driver.grid, _vector3(region.size))
        weight = complex(getattr(region, "weight", 1.0) or 1.0)
        if weight.imag:
            raise MeepSimulationNotLiftable([
                f"a FluxRegion carries a complex weight {weight}; MEEP scales that "
                f"region's E chunks by it and the flux is real(conj(E)*H), so the weight "
                f"does not factor out of the sum. Only real region weights are "
                f"reproduced here."
            ])
        normal = _region_normal(
            driver.grid, center, size, getattr(region, "direction", -1), kind="FluxRegion"
        )
        parts.append((weight.real, driver.add_flux_monitor(
            frequencies=frequencies, center=center, size=size,
            direction=normal, decimation_factor=decimation,
        )))
    return FluxMigration(parts=tuple(parts), frequencies=tuple(frequencies))


@dataclass(frozen=True)
class ForceMigration:
    """A ``sim.add_force`` monitor whose several regions are rebuilt as several surfaces.

    ``dft_force::force()`` walks the WHOLE chunk list (stress.cpp:101-114), and
    ``add_dft_force`` appends every region's chunks to the same three lists
    (stress.cpp:163-190), so a multi-region force monitor's spectrum is the sum over
    regions — each with its own ``nd``, its own ``fd`` and therefore its own branch.
    One monitor per region, summed here, is the exact counterpart.
    """

    monitors: tuple
    frequencies: tuple

    def get_force_spectrum(self):
        """The sum over regions — the counterpart of ``mp.get_forces(force)``."""
        total = numpy.zeros(len(self.frequencies), dtype=numpy.float64)
        for monitor in self.monitors:
            total += numpy.asarray(monitor.get_force_spectrum(), dtype=numpy.float64)
        return total

    def get_dft_data(self) -> ForceDftData:
        """Every region's transforms, flattened in ``monitors`` order.

        Save and load both walk that order, so a region is always paired with its own
        data; the container never crosses into MEEP, so MEEP's own chunk ordering does
        not have to be matched, only the two ends of the round trip.
        """
        return ForceDftData(blocks=tuple(
            block for monitor in self.monitors for block in monitor.get_dft_data().blocks
        ))

    def load_dft_data(self, data: ForceDftData) -> None:
        """MEEP's ``load_force_data``, region by region, in ``monitors`` order."""
        blocks = list(getattr(data, "blocks", ()) or ())
        expected = sum(len(monitor._accumulators) for monitor in self.monitors)
        if len(blocks) != expected:
            raise ValueError(
                f"ForceMigration.load_dft_data: the data carries {len(blocks)} "
                f"accumulator block(s) and these {len(self.monitors)} region(s) hold "
                f"{expected}."
            )
        offset = 0
        for monitor in self.monitors:
            count = len(monitor._accumulators)
            monitor.load_dft_data(ForceDftData(blocks=tuple(blocks[offset:offset + count])))
            offset += count

    def scale_dfts(self, scale: complex) -> None:  # dft_force::scale_dfts over every region.
        for monitor in self.monitors:
            monitor.scale_dfts(scale)


@dataclass(frozen=True)
class EnergyMigration:
    """A ``sim.add_energy`` monitor whose several regions are rebuilt as several volumes.

    ``dft_energy::electric()`` and ``magnetic()`` walk the whole chunk list
    (dft.cpp:672-701) and ``add_dft_energy`` appends every region's chunks to the same
    four (dft.cpp:724-737), so a multi-region energy monitor sums over regions.
    """

    monitors: tuple
    frequencies: tuple

    def _sum(self, reader: str):
        total = numpy.zeros(len(self.frequencies), dtype=numpy.float64)
        for monitor in self.monitors:
            total += numpy.asarray(getattr(monitor, reader)(), dtype=numpy.float64)
        return total

    def get_electric_energy_spectrum(self):  # mp.get_electric_energy(obj)
        return self._sum("get_electric_energy_spectrum")

    def get_magnetic_energy_spectrum(self):  # mp.get_magnetic_energy(obj)
        return self._sum("get_magnetic_energy_spectrum")

    def get_total_energy_spectrum(self):  # mp.get_total_energy(obj)
        return self._sum("get_total_energy_spectrum")


def _force_region_axes(region_center, region_size, declared, resolver) -> tuple[int, int]:
    """A force region's ``(fd, nd)``, MEEP's two DIFFERENT resolutions.

    ``fd = component_direction(where->c)`` (stress.cpp:172) and ``where->c`` is
    ``direction_component(Sx, d)`` with ``d`` the region's DECLARED direction, falling
    back to ``fields::normal_direction`` only when nothing was declared
    (``_add_fluxish_stuff``, python/simulation.py:3808-3817).

    ``nd = normal_direction(where->v)`` (stress.cpp:171) is ALWAYS the geometric
    inference — the declared direction never reaches it. That asymmetry is what
    selects the diagonal or off-diagonal branch, and reading one resolution for both
    (as the flux migration reasonably does, having only one) would put a declared
    ``ForceRegion(direction=mp.X)`` on a y-normal surface into the diagonal branch it
    does not belong to.

    ``resolver`` is :func:`_resolve_region_normal` with its predicates already bound,
    so the gate and the migration run one rule from two different sources of truth.
    """
    normal = resolver(region_center, region_size, -1, "ForceRegion")
    if declared is None or int(declared) < 0:
        return normal, normal
    return resolver(region_center, region_size, int(declared), "ForceRegion"), normal


def _migrate_force(monitor, driver, frequencies, decimation) -> Any:
    """Rebuild a ``DftForce`` on the driver: one surface per region, summed at read time.

    Transcribed from ``fields::add_dft_force`` (src/stress.cpp:153-191); see
    :class:`~.dft.ForceMonitor` for the registration and the reduction, and for why
    only the diagonal branch is reproduced.

    A run carrying ANY symmetry is refused here rather than folded. ``add_dft_force``
    opens with ``S.reduce(where_)`` (stress.cpp:161), which rewrites the region list
    before any chunk exists — merging equivalent volumes, adding ``phase_shift``
    weights, halving a self-redundant volume and deleting one whose weight cancels
    (vec.cpp:1416-1470). Neither ``add_dft_flux`` nor ``add_dft_energy`` does any of
    it, so the reflected gather that serves those two is NOT the same contract here,
    and no corpus case measures a folded force.
    """
    regions = list(getattr(monitor, "regions", ()) or ())
    if not regions:
        raise MeepSimulationNotLiftable([
            "a DftForce monitor carries no region object, so the surface it integrates "
            "cannot be read back and rebuilt."
        ])
    if getattr(driver.grid, "symmetry", ()):
        raise MeepSimulationNotLiftable([
            "a DftForce monitor on a run declaring a symmetry: MEEP reduces the force "
            "region list through symmetry::reduce (stress.cpp:161, vec.cpp:1416-1470) "
            "before registering a single chunk — merging symmetry-equivalent volumes, "
            "adding phase-shifted weights, halving a self-redundant volume and deleting "
            "one whose weight cancels — which neither add_dft_flux nor add_dft_energy "
            "does. That rewrite is not reproduced here and no measured case exists for "
            "it; drop the symmetries, or evaluate the force on CPU MEEP."
        ])
    if bool(getattr(driver.grid, "cylindrical", False)):
        raise MeepSimulationNotLiftable([
            "a DftForce monitor on a cylindrical run: MEEP's LOOP_OVER_FIELD_DIRECTIONS "
            "runs Z, R, P there (vec.hpp:147-149) and the diagonal terms pick up the "
            "2*pi*r ring measure under the sqrt_dV weight. No case in MEEP's tests or "
            "examples exercises it, so there is no oracle to measure it against."
        ])
    resolved = _resolved_decimation(monitor)
    if resolved:
        decimation = resolved
    resolver = functools.partial(_region_normal, driver.grid)
    monitors = []
    for region in regions:
        center = _vector3(region.center)
        size = _meep_region_extents(driver.grid, _vector3(region.size))
        weight = complex(getattr(region, "weight", 1.0) or 1.0)
        force_direction, normal = _force_region_axes(
            center, size, getattr(region, "direction", -1), resolver,
        )
        if force_direction != normal:
            raise MeepSimulationNotLiftable([
                f"a ForceRegion at center {tuple(center)} declares direction "
                f"{('x', 'y', 'z')[force_direction]} on a surface whose normal is "
                f"{('x', 'y', 'z')[normal]}, which takes MEEP's OFF-DIAGONAL "
                f"stress-tensor branch (stress.cpp:168-177). That branch is not "
                f"reproduced here: no case in MEEP's own tests or examples reaches it, "
                f"so a transcription would have no oracle behind it."
            ])
        monitors.append(driver.add_force_monitor(
            frequencies=frequencies, center=center, size=size,
            force_direction=force_direction, normal=normal, weight=weight,
            decimation_factor=decimation,
        ))
    if len(monitors) == 1:
        return monitors[0]
    return ForceMigration(monitors=tuple(monitors), frequencies=tuple(frequencies))


def _migrate_energy(monitor, driver, frequencies, decimation) -> Any:
    """Rebuild a ``DftEnergy`` on the driver: one volume per region, summed at read time.

    Transcribed from ``fields::add_dft_energy`` (src/dft.cpp:716-741); see
    :class:`~.dft.EnergyMonitor`.

    An ``EnergyRegion``'s ``weight`` is NOT applied, because MEEP does not apply it:
    ``add_dft_energy`` passes ``stored_weight = 1.0`` on all four chunk families and
    never reads ``where->weight`` (dft.cpp:722-729), unlike ``add_dft_flux``
    (dft.cpp:629) and ``add_dft_force`` (stress.cpp:168-185). A script that writes one
    gets MEEP's answer here, which is the contract — but a script that EXPECTS it to
    bite would be surprised by MEEP too, so it is named rather than silently dropped.

    Unlike force, a symmetry needs no refusal: ``add_dft_energy`` copies the volume
    list unchanged (dft.cpp:723) and ``add_dft``'s ``loop_in_chunks`` visits each
    image, which is exactly what ``_register_volume``'s reflected gather does — the
    measured case (test_dft_energy) is itself a region straddling a ``Mirror(Y)``.
    """
    regions = list(getattr(monitor, "regions", ()) or ())
    if not regions:
        raise MeepSimulationNotLiftable([
            "a DftEnergy monitor carries no region object, so the region it integrates "
            "cannot be read back and rebuilt."
        ])
    resolved = _resolved_decimation(monitor)
    if resolved:
        decimation = resolved
    monitors = []
    for region in regions:
        center = _vector3(region.center)
        size = _meep_region_extents(driver.grid, _vector3(region.size))
        monitors.append(driver.add_energy_monitor(
            frequencies=frequencies, center=center, size=size,
            decimation_factor=decimation,
        ))
    if len(monitors) == 1:
        return monitors[0]
    return EnergyMigration(monitors=tuple(monitors), frequencies=tuple(frequencies))


def _migrate_near2far(mp, monitor, driver, frequencies, decimation) -> Near2FarMigration:
    """Rebuild a ``DftNear2Far`` as raw-Yee accumulators, in MEEP's creation order.

    Transcribed from ``fields::add_dft_near2far`` (near2far.cpp:569-653). Per
    region, MEEP loops i over {E, H} and j over the two transverse directions of
    the normal, in the cyclic order ``X->(Y,Z)``, ``Y->(Z,X)``, ``Z->(X,Y)``
    (near2far.cpp:596-620), and adds a chunk for
    ``component = direction_component(i == 0 ? Ex : Hx, fd[j])`` with
    ``s = (j == 0 ? +1 : -1)``, negated again when the component is electric, and
    ``s * w->weight`` folded into the chunk's scale (near2far.cpp:636-647). That
    is four chunks per region: E_fd0, E_fd1, H_fd0, H_fd1.

    Two labels ride along with each entry because ``_emit_chunk`` needs them to tell
    two regions apart where they meet: the stored weight built COMPONENTWISE, MEEP's
    way, so its signed zeros are MEEP's (:func:`_near2far_stored_weight`), and the
    equivalent-source component ``c0`` MEEP records on every chunk as ``vc``
    (:func:`_near2far_source_component`).

    ``nperiods > 1`` is refused rather than approximated: it makes MEEP replicate
    the surface over lattice periods when evaluating the far field, which changes
    what the STORED data means, and reproducing the stored data is the whole
    contract here.
    """
    # Prefer the factor MEEP RESOLVED over the one the script requested. With
    # `decimation_factor=0` (MEEP's default) the requested value says only "automatic",
    # and re-deriving the rule is not reliable: MEEP's documented formula
    # `floor(1/(dt*(freq_max + src_freq_max)))` gives 10 for this probe while MEEP
    # itself resolved 9, because `src_time::get_fwidth()` is not the nominal fwidth.
    # A mismatched factor is not a rounding difference — it changes which timesteps are
    # accumulated and the quadrature weight on each, measured 2.2e-02 against 2.4e-07
    # once the factors agree. The chunk knows the answer, so ask it.
    resolved = _resolved_decimation(monitor)
    if resolved:
        decimation = resolved
    if int(getattr(monitor, "nperiods", 1) or 1) != 1:
        raise MeepSimulationNotLiftable([
            f"add_near2far(..., nperiods={monitor.nperiods}) replicates the near surface "
            f"over lattice periods; only nperiods=1 is reproduced here."
        ])
    entries = []
    for region in list(getattr(monitor, "regions", ()) or ()):
        center = _vector3(region.center)
        size = _meep_region_extents(driver.grid, _vector3(region.size))
        weight = complex(getattr(region, "weight", 1.0) or 1.0)
        normal = _region_normal(driver.grid, center, size, getattr(region, "direction", -1))
        first, second = _NEAR2FAR_TRANSVERSE[normal]
        bounds = tuple(
            (center[axis] - 0.5 * size[axis], center[axis] + 0.5 * size[axis])
            for axis in range(3)
        )
        for family in ("E", "H"):
            for j, transverse in enumerate((first, second)):
                component = family + "xyz"[transverse]
                sign = 1.0 if j == 0 else -1.0
                if family == "E":
                    sign = -sign
                entries.append((
                    component,
                    _near2far_stored_weight(sign, weight),
                    driver.add_yee_region_dft(
                        component=component, bounds=bounds,
                        frequencies=frequencies, decimation_factor=decimation,
                    ),
                    _near2far_source_component(family, j, (first, second)),
                ))
    return Near2FarMigration(
        entries=tuple(entries),
        frequencies=tuple(frequencies),
        cylindrical=bool(getattr(driver.grid, "cylindrical", False)),
    )


def _monitor_frequencies(monitor) -> list[float]:
    """The exact frequency list, taken from the monitor rather than re-derived.

    MEEP stores the resolved frequencies on the object. Recomputing them from
    ``fcen``/``df``/``nfreq`` would reintroduce MEEP's own endpoint convention
    (``nfreq == 1`` collapses to fcen alone, otherwise the samples span the full
    width inclusively) and any drift between the two is a monitor measuring
    different frequencies than the script asked for.
    """
    return [float(f) for f in monitor.freq]


def migrate_monitors(mp, sim, driver) -> MigratedMonitors:
    """Rebuild every migratable MEEP monitor on the driver; return a :class:`MigratedMonitors`.

    Each entry is recorded as the PAIR ``(meep monitor, ours)``, which is what keeps
    the MEEP object alive for as long as the run's results are readable — read that
    class's docstring for what happens when it is not.

    Frequencies, region and decimation come off the MEEP object, so a migrated monitor
    measures what the script asked for rather than what a re-spelling guessed.
    """
    migrated = MigratedMonitors()
    names = _component_names(mp, _is_cylindrical(sim))
    for monitor in list(getattr(sim, "dft_objects", ()) or ()):
        kind = type(monitor).__name__
        if kind not in MIGRATABLE_MONITORS:
            continue  # _check_outputs already refused these by name.
        frequencies = _monitor_frequencies(monitor)
        decimation = _monitor_decimation(monitor)
        # The two multi-region kinds first: each reads its own region list, because
        # `_monitor_region` answers for one region only and a monitor's second region
        # is invisible to it.
        if kind == "DftNear2Far":
            migrated.add(monitor,
                         _migrate_near2far(mp, monitor, driver, frequencies, decimation))
            continue
        if kind == "DftFlux":
            migrated.add(monitor, _migrate_flux(monitor, driver, frequencies, decimation))
            continue
        if kind == "DftForce":
            migrated.add(monitor, _migrate_force(monitor, driver, frequencies, decimation))
            continue
        if kind == "DftEnergy":
            migrated.add(monitor, _migrate_energy(monitor, driver, frequencies, decimation))
            continue
        # DftFields.
        center, size = _monitor_region(monitor)
        size = _meep_region_extents(driver.grid, size)
        args = list(getattr(monitor, "args", ()) or ())
        requested = list(args[0]) if args else []
        monitor_names = _monitor_component_names(mp, _is_cylindrical(sim))
        components = [monitor_names[int(c)] for c in requested if int(c) in monitor_names]
        unknown = [int(c) for c in requested if int(c) not in monitor_names]
        if unknown:
            raise MeepSimulationNotLiftable([
                f"a DftFields monitor requests MEEP component(s) {unknown}, which are not "
                f"among the twelve stored field components (E/D/H/B) — derived components "
                f"are not accumulated here."
            ])
        resolved = _resolved_decimation(monitor)
        if resolved:
            decimation = resolved
        # args[5] is use_centered_grid (python/simulation.py add_dft_fields builds
        # [cs, where, center, size, freq, use_centered_grid, decimation, persist]);
        # yee_grid=True is its negation.
        yee_grid = len(args) > 5 and args[5] is False
        if yee_grid:
            bounds = tuple(
                (center[axis] - 0.5 * size[axis], center[axis] + 0.5 * size[axis])
                for axis in range(3)
            )
            migrated.add(monitor, DftFieldsYeeMigration(
                accumulators={
                    component: driver.add_yee_region_dft(
                        component=component, bounds=bounds,
                        frequencies=frequencies, decimation_factor=decimation,
                    )
                    for component in components
                },
                frequencies=tuple(frequencies),
            ))
            continue
        migrated.add(monitor, driver.add_dft_monitor(
            frequencies=frequencies, components=tuple(components),
            center=center, size=size, decimation_factor=decimation,
        ))
    return migrated


# --- the lift ---------------------------------------------------------------------


def lift_simulation(
    sim,
    *,
    prefer_gpu: bool = True,
    gpu_id: int = 0,
    progress_cb: Callable[[int, int], None] | None = None,
) -> FdtdDriver:
    """Build an :class:`~.driver.FdtdDriver` that reproduces ``sim``, ready to step.

    Calls ``sim.init_sim()`` — MEEP's own grid construction, geometry
    rasterization and subpixel smoothing — then reads the resulting permittivity
    back through ``sim.get_epsilon()`` and transfers the cell, the clock, the
    sources, the boundary layers, the symmetries and the material model onto a
    driver. Nothing is stepped; add monitors, then call
    :meth:`~.driver.FdtdDriver.run`, or use :func:`run_on_gpu` for the one-call
    path.

    ``sim`` is initialized but not otherwise modified: no fields are stepped and
    no MEEP state is reset, so the same object can still be run on CPU afterwards
    for a comparison (which is exactly what ``test_from_meep.py`` does).

    Args:
        sim: A constructed ``meep.Simulation``.
        prefer_gpu: Run on this host's GPU: CUDA through CuPy, or Metal on an
            Apple GPU, where any configuration no released arm covers steps the
            host CPU and says so; raises before ``sim.init_sim()`` when the host
            has neither. Defaults to True, as :func:`run_on_gpu` does, so a bare
            ``lift_simulation(sim)`` is a GPU driver. Pass False for the NumPy
            reference, which is the same driver on every host and never
            dispatches a kernel whatever ``MEEP_GPU_DISPATCH`` says.
            (:class:`~.driver.FdtdDriver` built directly defaults to the
            reference.)
        gpu_id: CUDA device index, used only when ``prefer_gpu`` is set; must be 0
            on an Apple GPU.
        progress_cb: ``(planes_done, planes_total)`` during the per-point
            sampling of a STRUCTURED cell — the permittivity, and then the
            susceptibility sigma when the media differ in their dispersion. That
            sampling is the only part of a lift that can take appreciable time:
            MEEP's ``chi1inv`` reader is one call per Yee point per tensor entry,
            about 0.9 us each, so a 1e6-cell grid is a few seconds and a
            300-cubed grid is a few minutes; both compatibility sigma routes are
            slower still per point, since their calls go through monitor.cpp's
            frequency branch and the lookup adds a geometry containment test.
            Each pass is reported as its own 0..total sequence — two of them
            normally, and three when the lookup declines partway and the
            inversion runs behind it. No progress is printed by this package.

    Returns:
        The configured :class:`~.driver.FdtdDriver`. Its ``lift_record`` names the
        MEEP version and precision the lift read (``meep_precision`` is
        ``"single"``, ``"double"`` or None when unread) and the precision the
        engine steps, which is single whatever the MEEP build; a lift from a
        double-precision MEEP proceeds and says so once per process on stderr.

    Raises:
        MeepSimulationNotLiftable: With every reason :func:`gpu_compatibility`
            found, before anything is built. The rasterized volume is then checked
            independently — shape against the grid MEEP built, uniformity against
            the declared medium — so a material route the declaration analysis
            does not know about still fails here rather than stepping.
        RuntimeError: ``prefer_gpu`` (the default here) on a host with neither a
            CUDA device nor an Apple GPU, raised before ``sim.init_sim()``; the
            message names ``prefer_gpu=False`` as the NumPy reference and says the
            GPU is what a call passing no ``prefer_gpu`` asks for. A bare
            ``lift_simulation(sim)`` therefore raises on a CPU-only host.
        ValueError: ``prefer_gpu`` (the default here) with ``gpu_id`` other than 0
            on an Apple GPU, also before ``sim.init_sim()``.
    """
    if prefer_gpu:
        # A HOST FACT, checked before MEEP builds anything: a GPU request on a host
        # with no GPU route (or gpu_id != 0 on an Apple GPU) fails here in
        # milliseconds instead of after init_sim. FdtdDriver's own resolution below
        # stays the authority.
        from .backends import _gpu_route  # noqa: PLC0415

        try:
            _gpu_route(gpu_id)
        except RuntimeError as refusal:
            # The GPU is the DEFAULT of both MEEP entry points, so the caller this
            # reaches may have requested nothing; the engine's own text is kept
            # whole and the entry points' default is named after it.
            raise RuntimeError(
                f"{refusal} lift_simulation and run_on_gpu run on the GPU by "
                "default: prefer_gpu=True is what a call passing no prefer_gpu "
                "asks for."
            ) from None
    mp = _import_meep()
    verdict = gpu_compatibility(sim)
    if not verdict.supported:
        raise MeepSimulationNotLiftable(verdict.reasons)
    # THE MEEP BUILD'S PRECISION, read and recorded and never refused. The engine
    # steps single precision whatever it says; a double-precision build is told so
    # once per process, with the agreement to expect.
    build = _meep_build(mp)
    _announce_precision(build)

    dimensions = _effective_dimensions(mp, sim)
    cylindrical = dimensions == MEEP_CYLINDRICAL
    sim.init_sim()  # MEEP builds the grid, rasterizes the geometry, and smooths it.
    driver = FdtdDriver(
        cell_size=_reduced_cell_size(sim, dimensions),
        dimensions=2 if cylindrical else (1 if dimensions == 0 else dimensions),
        cylindrical=cylindrical,
        m=int(getattr(sim, "m", 0)) if cylindrical else 0,
        # The user-facing flag, carried verbatim; MEEP stores its INVERSE internally
        # (fields.zero_fields_near_cylorigin = not accurate_..., simulation.py:2483).
        accurate_fields_near_cylorigin=(
            bool(getattr(sim, "accurate_fields_near_cylorigin", False)) if cylindrical
            else False),
        resolution=float(sim.resolution),
        courant=float(sim.Courant),
        # The ground truth for the storage mode, not the constructor argument: MEEP
        # promotes a run to complex on its own for a nonzero k_point, and reading
        # sim.force_complex_fields instead would build a real driver for a run MEEP
        # is stepping complex.
        force_complex_fields=not sim.fields.is_real,
        symmetry=_lift_symmetry(sim),
        k_point=_lift_k_point(sim, dimensions),
        # MEEP's special_kz: k_point.z leaves the Bloch wavevector entirely and goes
        # into the fields constructor's own `beta` slot (simulation.py:2478-2487),
        # where it becomes an analytic exp(i*2*pi*beta*z) rather than a wrap phase.
        # `_lift_k_point` has already zeroed z, which is MEEP's use_bloch(Vector3(kx, ky)).
        beta=float(getattr(sim.k_point, "z", 0.0)) if getattr(sim, "special_kz", False) else 0.0,
        # MEEP's bfast_scaled_k, read straight off the Simulation attribute
        # (simulation.py:1537) — which is exactly the object handed to `mp.fields` at
        # :2486, so nothing derives it and nothing else reads it. Independent of the
        # k_point: MEEP passes them to two different constructor slots.
        bfast_scaled_k=tuple(
            float(value)
            for value in (getattr(sim, "bfast_scaled_k", None) or (0.0, 0.0, 0.0))
        ),
        boundaries=_lift_boundaries(sim, dimensions),
        prefer_gpu=prefer_gpu,
        gpu_id=gpu_id,
    )
    # Recorded on the driver, the convention of sigma_lift and migrated_monitors
    # below, and surfaced as :attr:`GpuRunResult.lift_record`.
    driver.lift_record = build
    try:
        _require_grids_agree(sim, driver)
        materials = _materials_of(mp, sim)
        medium = _representative_medium(mp, materials)
        # A per-point epsilon route takes the structured path unconditionally. The
        # declared media cannot decide it: a material_function or an epsilon_input_file
        # names no medium at all, so the signature set collapses to one and the uniform
        # path would be chosen for a cell that is nothing of the kind. (It would not
        # step it wrongly — _lift_epsilon re-checks MEEP's rasterized array and raises —
        # but it would refuse a cell this engine can lift.)
        routes = _epsilon_routes(mp, sim)
        if routes:
            _require_epsilon_only_structure(sim, routes)
        if routes or len({_medium_signature(material) for material in materials}) > 1:
            _lift_epsilon_structured(mp, sim, driver, materials, progress_cb, routes=routes)
        else:
            _lift_epsilon(sim, driver, medium)
        # The sigma-structured case: media differ in their E_susceptibilities, so
        # no single medium's terms describe the cell — the per-point volumes come
        # off MEEP instead, read exactly through the sigma-reader patch or
        # recovered from chi1inv where that reader does not exist (the gate has
        # already refused every other kind of difference, and every precondition
        # the recovery route needs beyond the reader's).
        sigma_structured = _media_differ_in_susceptibilities(materials)
        # The conductivity twin of the line above: D_conductivity varies across the
        # cell, so no medium's declaration describes it and the volume is read back off
        # MEEP's own D-row chi1inv instead. Installed AFTER _lift_absorber, because what
        # comes back is MEEP's material + absorber sum.
        conductivity_structured = _conductivity_structured(mp, sim, materials)
        _lift_material(driver, medium, skip_susceptibilities=sigma_structured,
                       skip_conductivity=conductivity_structured)
        if sigma_structured:
            # Recorded on the driver rather than returned: lift_simulation hands back
            # one value, and a caller who wants to know which of the three sigma
            # routes ran already holds the driver. Same convention as
            # migrated_monitors below.
            driver.sigma_lift = _lift_susceptibilities_structured(
                mp, sim, driver, materials, progress_cb)
        _lift_absorber(mp, sim, driver, dimensions)
        if conductivity_structured:
            read = _read_conductivity_volumes(mp, sim, driver, progress_cb)
            driver.set_conductivity(
                {"D" + name[1:]: read[name] for name in E_COMPONENTS},
                # MEEP's conductivity[c][d] already holds material + absorber
                # (meepgeom.cpp:1596-1625), so the ramp must not be added again.
                absorber_included=True,
            )
        _lift_pml(mp, sim, driver, dimensions)
        for source in sim.sources:
            if _fold_drops_source(mp, driver.grid, source, dimensions):
                continue
            handlers = REALIZABLE_SOURCES.get(type(source).__name__)
            if handlers is not None:
                for spec in handlers[1](mp, sim, source, dimensions):
                    driver.add_source(spec)
            else:
                driver.add_source(_lift_source(mp, source, dimensions))
        # Monitors LAST: they read the grid and the storage mode, both of which the
        # steps above settle. Attached to the driver rather than returned separately so
        # lift_simulation keeps its single-value contract, and so a caller who reaches
        # for a migrated monitor finds it on the object they already hold.
        driver.migrated_monitors = migrate_monitors(mp, sim, driver)
    except Exception:
        driver.close()  # Never leave a half-built driver holding device memory.
        raise
    return driver


def _require_grids_agree(sim, driver) -> None:  # Pin the cell count and the array convention.
    """Refuse unless MEEP's own epsilon array has the shape this grid implies.

    THE REGISTRATION PIN. ``sim.get_array`` puts grid point ``j`` at
    ``axis_origin(d) + (j - 0.5)*dx`` and returns ``n_full + 1`` points on every
    unfolded axis (the extra plane at index 0 being the periodic duplicate of the
    last), which is verified against ``sim.get_array_metadata()``. If MEEP's cell
    count and this engine's ``meep_cell_count`` ever disagree — a cell length that
    lands on a rounding boundary, a resolution MEEP snaps differently — every
    coordinate in the run is off, and the field still looks like a field. Half a
    cell here was worth 4.8e-02 on a transmitted flux the last time it happened
    (Grid.origin_doubled), so it is checked rather than trusted.

    A mirror-folded axis at an EVEN count gains no duplicate plane, and MEEP
    returns exactly its FULL cell count there. At an ODD count the folded window
    sits half a cell above the requested cell and the convention splits by
    boundary kind — measured on MEEP 1.33.0 (N=21, resolution 10): metallic stays
    at ``N`` (y in [-0.95, +1.05]; the requested -L/2 site lies below the window
    and a wall serves no image), periodic gains one plane at -L/2 exactly
    (y in [-1.05, +1.05], ``N + 1`` points), the lattice-wrap image of the top
    stored plane and an exact duplicate of it. ``Fields._add_boundary_cells``
    follows the same rule. A METALLIC axis
    gains none either, for the same reason and measured the same way: with
    ``k_point`` unset MEEP returns ``N`` per axis where a periodic run returns
    ``N + 1`` (verified against ``get_array_metadata``, whose PEC coordinates start at
    ``axis_origin + dx/2`` instead of ``axis_origin - dx/2``). This pin is therefore
    also the tripwire on the boundary lift itself: get the condition wrong and the
    shapes disagree here, before a single step.
    """
    epsilon = numpy.asarray(sim.get_epsilon())
    grid = driver.grid
    counts = (grid.nx_full, grid.ny_full, grid.nz_full)
    # MEEP's array has FEWER AXES than three whenever one of them holds a single
    # point: array_slice.cpp keeps only the entries above 1, so a 2-D epsilon is
    # (nx[+1], ny[+1]), a 1-D one is (nz[+1],), and a 3-D cell with a one-pixel x axis
    # is (ny+1, nz+1). Comparing against a 3-D expectation would fail every such lift
    # on a shape that is in fact exactly right. `Fields.meep_array_axes` is the one
    # definition of which axes survive; reading it here keeps this pin and the
    # readback from ever disagreeing about the convention.
    active = driver.fields.meep_array_axes()

    def expected_points(axis: int) -> int:
        if grid.is_mirrored(axis):
            # Folded: full count — plus the odd-window wrap plane on a periodic axis.
            odd_wrap = counts[axis] % 2 == 1 and not grid.is_metallic(axis)
            return counts[axis] + (1 if odd_wrap else 0)
        if grid.is_metallic(axis) or grid.is_axis(axis) or getattr(grid, "cylindrical", False):
            return counts[axis]
        return counts[axis] + 1

    expected = tuple(expected_points(axis) for axis in active)
    if epsilon.shape != expected:
        reduced = (
            f" MEEP reports {', '.join(AXIS_NAMES[axis] for axis in active)} only "
            f"(dimensions={grid.dimensions}, cell counts {counts}); an axis of one point is not "
            f"one of its array's dimensions."
            if len(active) < 3 else ""
        )
        raise ValueError(
            f"MEEP's epsilon array is {epsilon.shape} where this grid implies {expected} "
            f"(cell counts {counts} at resolution {grid.resolution:g}, plus MEEP's one duplicate "
            f"plane per axis that is both UNFOLDED and PERIODIC — or folded PERIODIC at an ODD "
            f"count, whose shifted window makes the requested -L/2 site the wrap image of the top "
            f"stored plane).{reduced} The two codes have built "
            f"different grids, so every "
            f"coordinate in the lifted run — source centres, PML faces, monitor planes — would be "
            f"registered against the wrong lattice while still producing a smooth field. This is a "
            f"bug in the lift, not in your simulation; please report the cell_size / resolution "
            f"that produced it."
        )
    for axis in range(3):
        if grid.is_invariant(axis):
            continue  # MEEP never measures this axis; vol2d takes x and y, vol1d takes z.
        length = _vector3(sim.cell_size)[axis]
        if length == 0.0 and counts[axis] == 1:
            continue  # MEEP's vol3d turns a zero extent into ONE cell: (xsize == 0) ? 1 : ...
        if meep_cell_count(length, float(sim.resolution)) != counts[axis]:
            raise ValueError(
                f"cell count mismatch on {AXIS_NAMES[axis]}: MEEP's int(size*a + 0.5) gives "
                f"{meep_cell_count(length, float(sim.resolution))} for length {length!r}, this grid "
                f"holds {counts[axis]}."
            )
    for axis in range(3):
        if grid.is_invariant(axis) and counts[axis] != 1:
            raise ValueError(
                f"the {AXIS_NAMES[axis]} axis is invariant at dimensions={grid.dimensions} but "
                f"holds {counts[axis]} cells; an invariant axis stands for a whole infinite "
                f"direction with exactly one."
            )


def _lift_symmetry(sim) -> tuple:  # mp.Mirror(direction, phase) -> meep_gpu.Mirror(axis, phase).
    return tuple(
        Mirror("XYZ"[int(symmetry.direction)], int(complex(symmetry.phase).real))
        for symmetry in getattr(sim, "symmetries", ()) or ()
    )


def _lift_k_point(sim, dimensions: int) -> tuple[float, float, float]:  # sim.k_point, zeroed where MEEP drops it.
    """``sim.k_point`` with a hard zero on every axis this run makes invariant.

    MEEP drops the component rather than applying it: ``use_bloch`` runs over
    ``LOOP_OVER_DIRECTIONS``, so a ``k_point`` given to a 2-D simulation phases x and
    y and its z entry has no direction to belong to. The zero is written here so a
    lifted 2-D script with ``k_point=mp.Vector3(0.3, 0, 0.2)`` reproduces MEEP rather
    than hitting the engine's refusal for a phase on an invariant axis — the engine's
    refusal protects a caller who WROTE that phase, not a converter reproducing a run
    where it was never in effect.

    A nonzero ``k_point.z`` on a zero-thickness cell reaches this in the
    ``special_kz`` spellings ONLY (``kz_2d="complex"`` and ``"real/imag"``), and the
    zero written here is MEEP's own: ``_init_fields`` passes that z to the fields
    constructor's ``beta`` slot and hands ``use_bloch`` only ``Vector3(kx, ky)``
    (python/simulation.py:2478-2504), so it is an analytic phase and not a wrap.
    :func:`lift_simulation` picks it back up as the driver's ``beta``. Under
    ``kz_2d="3d"`` it never reaches this at all: it stops the 2-D collapse in
    ``_infer_dimensions`` and MEEP builds a 3-D run instead, which is what
    :func:`_effective_dimensions` reports.
    """
    if not sim.k_point:
        return (0.0, 0.0, 0.0)
    invariant = _invariant_axes(dimensions)
    return tuple(
        0.0 if axis in invariant else value
        for axis, value in enumerate(_vector3(sim.k_point))
    )


def _lift_boundaries(sim, dimensions: int = 3):  # sim.k_point -> the driver's outer-boundary declaration.
    """``"periodic"`` when this simulation has a ``k_point``, ``"metallic"`` when it does not.

    A REDUCED run gets a per-axis mapping instead of the bare condition, naming only
    the axes it resolves. The invariant axis keeps the engine's periodic default,
    which is not a boundary choice at all — MEEP has no ``fields::boundaries`` entry
    for a direction it does not loop over, and a one-cell periodic axis is how MEEP
    itself emulates one (fields.cpp ``nosize_direction``). Handing it ``"metallic"``
    along with the real axes would zero every component whose Yee shift on it is 0,
    which for a 2-D cell is the whole TE polarization; see
    :meth:`~.grid.Grid._require_boundaries_leave_invariant_axes_alone`.

    THE WHOLE CONDITION IS ``sim.k_point``, and it is all-or-nothing because MEEP
    makes it so. ``fields::fields`` sets ``boundaries[b][d] = Metallic`` on every face
    it has one for (fields.cpp), and the only thing that overwrites it is
    ``fields::use_bloch``, which ``Simulation._init_fields`` calls — for EVERY
    direction at once — under a plain ``if self.k_point:`` (python/simulation.py).
    ``k_point=mp.Vector3()`` is truthy, so an all-zero vector still turns the whole
    cell periodic; ``k_point=False``, the default, leaves the whole cell a perfect
    electric conductor box.

    There is therefore no declared-argument route to a per-axis mix. The only one MEEP
    offers is ``sim.set_boundary`` / ``sim.fields.use_bloch`` on an ALREADY-INITIALIZED
    simulation, which writes into ``fields.boundaries`` — a pointer this converter
    cannot read, and the reason an initialized simulation is refused outright
    (:func:`_check_simulation_shape`). The engine itself takes a per-axis table, so a
    mixed run is expressible by building ``FdtdDriver`` directly; it just cannot be
    LIFTED, because there is nothing readable to lift it from. (A reduced run is the
    one exception, and only in appearance: its invariant axis is not a face MEEP
    declared anything about.)
    """
    condition = "periodic" if sim.k_point else "metallic"
    # MEEP fields.cpp:80-85, "unit directions are periodic by default": a direction of
    # ONE CELL is forced Periodic whatever k_point says. It is not a corner case — it is
    # the mp.Vector3(0, 0, L) idiom several of MEEP's own 1-D examples are written in
    # without declaring dimensions=1, where vol3d turns each zero extent into one cell.
    # Measured: with those axes lifted metallic the engine returns an exact zero
    # (1.000e+00 against CPU MEEP); with them periodic it reproduces MEEP at 1.805e-07.
    unit = _unit_cell_axes(sim, dimensions)
    if dimensions == MEEP_CYLINDRICAL or _is_cylindrical(sim):
        # Only z is a declarable pair in Dcyl: r is (AXIS, METALLIC) by
        # construction in the Grid, phi is the invariant axis. MEEP's cylindrical
        # default is the PEC box (no k_point machinery unless k.z is set).
        #
        # The unit rule reaches HERE TOO, and this branch used to skip it. MEEP's
        # guard excludes only `d != R`, so a Dcyl cell declared mp.Vector3(sr, 0, 0)
        # — ring-cyl.py's and perturbation_theory.py's spelling, where z is one cell
        # — has its z forced Periodic exactly as a Cartesian unit axis does. Lifted
        # metallic instead, the driver refused outright ("the z axis is declared
        # 'metallic' and holds exactly one cell"), which is what stopped both scripts
        # from lifting at all.
        return {"z": "periodic"} if 2 in unit else {"z": condition}
    invariant = _invariant_axes(dimensions)
    if not invariant and not unit:
        return condition
    return {
        AXIS_NAMES[axis]: condition
        for axis in range(3)
        if axis not in invariant and axis not in unit
    }


def _lift_epsilon(sim, driver, medium) -> None:
    """Install the instantaneous permittivity, and verify it against MEEP's own array.

    ``sim.get_epsilon()`` at MEEP's default frequency 0 is eps_INFINITY — the
    ``omega -> inf`` limit, with neither the susceptibilities nor the conductivity
    folded in — which is exactly what :meth:`~.driver.FdtdDriver.set_epsilon`
    takes. It is read here as the *check*: the compatibility register has already
    established that one medium fills the cell, and this confirms MEEP agrees,
    cell by cell, after its own rasterization and smoothing. A volume that is not
    uniform after all raises rather than being installed at some registration or
    other (see the module docstring for what that costs).

    The values installed are the medium's own ``epsilon_diag``, because MEEP's
    array cannot supply them: for an anisotropic medium ``get_epsilon()`` reports
    ``3/(1/ex + 1/ey + 1/ez)``, one number where the driver needs three. The
    harmonic mean is compared against the array instead, so the declared medium
    and MEEP's structure are still cross-checked in that case.
    """
    epsilon = numpy.asarray(sim.get_epsilon(), dtype=numpy.float64)
    spread = float(epsilon.max() - epsilon.min())
    scale = float(numpy.abs(epsilon).mean())
    if scale <= 0.0 or spread > _EPSILON_UNIFORM_TOLERANCE * scale:
        raise ValueError(
            f"MEEP's rasterized permittivity is not uniform (it spans {epsilon.min():.6g} to "
            f"{epsilon.max():.6g}), so this cell is structured even though its declared materials "
            f"are not. Neither of the two routes that used to land here still can: a "
            f"MaterialGrid is expanded into its endpoint media, so a grid that varies at all "
            f"takes the structured route and one whose endpoints match interpolates to a "
            f"genuinely uniform cell; and a per-point epsilon route (a material callable, an "
            f"ndarray, an epsilon_input_file) takes the structured route unconditionally. So "
            f"something built this structure that lift_simulation did not see. A structured "
            f"cell cannot be "
            f"lifted through THIS path: sim.get_epsilon() is the "
            f"trace-averaged, cell-centred DIAGNOSTIC array, and reinstalling it reproduces CPU "
            f"MEEP to 1.16e-01 at best against 3.06e-07 for the same structure sampled per "
            f"component. Build the run on FdtdDriver and install the geometry with "
            f"set_epsilon_components(...) or set_epsilon_smoothed(...)."
        )
    diagonal = _vector3(medium.epsilon_diag)
    # getattr: the minimal media the refusal tests hand this path carry only
    # epsilon_diag, and an absent off-diagonal is a zero one.
    raw_offdiagonal = getattr(medium, "epsilon_offdiag", None)
    offdiagonal = _vector3(raw_offdiagonal) if raw_offdiagonal is not None else (0.0, 0.0, 0.0)
    # The full symmetric tensor and its inverse — for a diagonal medium this
    # reduces to the per-axis reciprocals exactly, so one formula serves both.
    tensor = numpy.array([
        [diagonal[0], offdiagonal[0], offdiagonal[1]],
        [offdiagonal[0], diagonal[1], offdiagonal[2]],
        [offdiagonal[1], offdiagonal[2], diagonal[2]],
    ], dtype=numpy.float64)
    eigenvalues = numpy.linalg.eigvalsh(tensor)
    # A cell that is ENTIRELY a perfect conductor reaches here — ``default_material=
    # mp.metal`` with no geometry — and is carried through like any other declared
    # medium, because that is what MEEP does with it: the tensor inverts ordinarily to
    # -1e-20 on the diagonal and the ordinary constitutive update drives E to ~0. It is
    # degenerate but it is not a refusal, and it must not become a LIFT throw when the
    # gate has already accepted it (the gate reads the same epsilon_diag).
    if float(eigenvalues.min()) <= 0.0 and not is_pec_epsilon(float(eigenvalues.min())):
        raise MeepSimulationNotLiftable([
            f"the declared permittivity tensor (epsilon_diag={diagonal}, "
            f"epsilon_offdiag={offdiagonal}) has a non-positive eigenvalue "
            f"({float(eigenvalues.min()):.6g}); it is not a passive dielectric and neither "
            f"engine steps it stably."
        ])
    chi1inv = numpy.linalg.inv(tensor)
    harmonic_mean = 3.0 / float(numpy.trace(chi1inv))
    measured = float(epsilon.mean())
    # abs() on the SCALE too, not only on the residual. A perfect conductor's harmonic
    # mean is -1e20, and a tolerance of `1e-5 * harmonic_mean` is then NEGATIVE, so the
    # comparison fires on every input including an exact match — the same sign accident
    # the bulk classifier makes. For a passive dielectric the two spellings are
    # identical. MEASURED on default_material=mp.metal: get_epsilon reports
    # -1.00000002e+20 against the declared -1e+20, a 2e-8 relative gap that is the
    # float32 storage round trip and nothing else.
    if abs(measured - harmonic_mean) > 1e-5 * abs(harmonic_mean):
        raise ValueError(
            f"MEEP reports a uniform permittivity of {measured:.9g} where the declared medium "
            f"(epsilon_diag={diagonal}, epsilon_offdiag={offdiagonal}) implies "
            f"{harmonic_mean:.9g} (get_epsilon returns the average eigenvalue, 3/trace(eps^-1)). "
            f"The declared material is not the one MEEP built, so the lift would install a "
            f"permittivity the reference run does not have."
        )
    shape = driver.shape
    if any(offdiagonal):
        # A uniform ROTATED tensor: install each row's effective diagonal (the
        # reciprocal of the chi1inv diagonal entry — NOT eps_cc) plus the raw
        # off-diagonal chi1inv entries, exactly the split MEEP's own
        # step_update_EDHB consumes. Cross-validated against CPU MEEP stepping
        # the same medium at 1.9e-07..4.6e-07 over periodic, metallic, Bloch,
        # PML and chi3 variants (test_tensor_epsilon.py).
        volumes = {
            component: numpy.full(shape, 1.0 / chi1inv[axis, axis], dtype=numpy.float64)
            for axis, component in enumerate(E_COMPONENTS)
        }
        rows = {
            row: {
                partner: numpy.full(shape, chi1inv[i, j], dtype=numpy.float64)
                for j, partner in enumerate(E_COMPONENTS)
                if j != i and chi1inv[i, j] != 0.0
            }
            for i, row in enumerate(E_COMPONENTS)
        }
        driver.set_epsilon_components(volumes, chi1inv_offdiagonal=rows)
        return
    if len(set(diagonal)) == 1:
        driver.set_epsilon(numpy.full(shape, diagonal[0], dtype=numpy.float64))
        return
    driver.set_epsilon_components({
        component: numpy.full(shape, diagonal[axis], dtype=numpy.float64)
        for axis, component in enumerate(E_COMPONENTS)
    })


def _lift_epsilon_structured(mp, sim, driver, materials, progress_cb, routes=()) -> None:
    """Install a spatially varying permittivity by reading MEEP's own ``chi1inv``.

    THE ROUTE, and why this one. MEEP rasterizes the geometry and applies its
    subpixel smoothing into a per-component, per-direction inverse-permittivity
    tensor. ``sim.get_epsilon()`` does NOT report it — that array is the average
    eigenvalue, bilinearly interpolated onto cell centres, and reinstalling it
    reproduces CPU MEEP to 1.16e-01 at best. ``fields.get_chi1inv(c, d, iloc, 0)``
    reports the stored value itself, one Yee point at a time, and reinstalling
    THAT reproduces CPU MEEP at the engine's floor: measured 1.5e-07 (two blocks),
    2.2e-07 (slab), 2.8e-07 (a finite block, corners included) and 3.2e-07 on odd
    cell counts with off-lattice faces.

    THE TRAP, and the pin against it. The lookup takes MEEP's integer lattice,
    whose unit is ``dx/2``, and ``grid_volume::index`` assumes the ivec already
    carries the component's own parity — an even coordinate where ``Ez`` wants an
    odd one is not rejected, it silently reads a NEIGHBOURING cell. So the
    coordinates are required to be exact lattice points (no rounding is permitted
    to fix anything up) and their parity is checked against MEEP's own
    ``gv.iyee_shift(component)``, which is the same convention
    ``dispersion.component_coordinates`` implements. A half-cell slip therefore
    raises here, at setup, instead of running a structure shifted by half a cell.

    THE WALL PLANES. The stored low-boundary rows of a metallic axis are lattice
    points MEEP owns with NO chunk (``grid_volume::owns`` needs ``o > 0``), so
    ``get_chi1inv`` answers its outside-the-cell vacuum default there rather than
    a material — measured, and excluded from BOTH checks that would otherwise
    compare it against something real (the declared-media span check below and
    :func:`_require_bulk_matches_meep`) by :func:`_unowned_wall_planes`, which
    carries the whole story. The bulk check takes the UNION of the three
    components' masks, for the reason argued at the accumulation below.

    THE TENSOR. MEEP's averaging is anisotropic, so a surface whose normal is not
    a coordinate axis puts real weight in the OFF-diagonal ``chi1inv`` entries —
    measured 0.203 for a cylinder and 0.086 for a rotated block, against exactly
    0.0 for every axis-aligned case. Every one of the six off-diagonal entries is
    read at every point ALONGSIDE the diagonal and installed as the row's coupling
    volumes (``FdtdDriver.set_epsilon_components(chi1inv_offdiagonal=...)``); the
    values are taken verbatim because MEEP's registration convention (the row
    entries at the component's own loop slots, physically the integer node —
    anisotropic_averaging.cpp:248-257) is the same one the engine's stencil
    consumes. What used to be the refusal is now the ingestion; the case a tensor
    still cannot serve — a mirror-folded grid, where the coupling coefficient is
    odd across the plane — is refused at the pre-flight
    (:func:`_check_structured_geometry`) and again at install.
    """
    grid = driver.grid
    fields = sim.fields
    gv = fields.gv
    get_chi1inv = fields.get_chi1inv
    lattice_per_unit = 2.0 * float(sim.resolution)  # MEEP's integer lattice unit is dx/2.
    cylindrical = bool(getattr(grid, "cylindrical", False))
    if cylindrical:
        # The Dcyl spelling of the same three constitutive rows: this engine's
        # (Ex, Ey, Ez) are MEEP's (Er, Ep, Ez) under x -> r, y -> phi, and the
        # grid_volume's ivecs are two-dimensional (r, z) — mp.iveccyl — with no
        # phi slot at all, so the invariant-axis placeholder index is dropped at
        # construction rather than carried as a strideless direction.
        directions = {"Ex": mp.R, "Ey": mp.P, "Ez": mp.Z}
        components = {"Ex": mp.Er, "Ey": mp.Ep, "Ez": mp.Ez}
        ivec = lambda ix, jy, kz: mp.iveccyl(ix, kz)  # noqa: E731 — the one construction seam.
    else:
        directions = {"Ex": mp.X, "Ey": mp.Y, "Ez": mp.Z}
        components = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez}
        ivec = mp.ivec
    declared = [value for material in materials for value in _vector3(material.epsilon_diag)]
    # THE PERFECT CONDUCTOR IS NOT AN ENDPOINT OF AN AVERAGE, so it comes off both
    # sides of this test rather than widening it. A PEC cell is not a subpixel average
    # of two materials — MEEP refuses to average one at all (meepgeom.cpp:1093-1094,
    # `is_metal(...) goto noavg`) — so the premise "an average lies between the
    # materials it averages" simply does not speak about it. Left in, the sentinel
    # makes `low` -1e20, and the sampled value fails the tolerance anyway by the
    # float32 round trip: it comes back -1.0000000317344784e+20 against a bound of
    # -1e+20 with a tolerance of 1e-4 * high. MEASURED firing on the metal cavity.
    declared = [value for value in declared if not is_pec_epsilon(value)]
    low, high = (min(declared), max(declared)) if declared else (0.0, 0.0)
    # THE ONE CELL WHERE THE DECLARED BRACKET IS NOT A BOUND, and it is MEEP's own
    # arithmetic that breaks it rather than anything here. A MaterialGrid smoothed
    # through `fallback_chi1inv_row` builds its voxel average from
    # `u_proj = tanh_projection(uval + |grad u| * x0, beta, eta)` with x0 running over
    # the WHOLE voxel diameter (meepgeom.cpp:1208-1218) and never clamps u to [0, 1],
    # so `meps = (1-u)*eps1 + u*eps2` linearly EXTRAPOLATES past both endpoints. The
    # spread test below would then refuse a correctly-read cell.
    #
    # MEASURED on a 4x4x8 grid ramped 1 -> 6 in z, 2x2x4 cell at resolution 10, against
    # pristine CPU MEEP stepping the same object: the sampled permittivity reaches 7.71
    # on Ex (the meps row, n_r = 0) and 0.759 on Ez (the 1/minveps row, n_r = 1) — both
    # outside [1, 6] — while the stepped-field parity is 2.5961e-07 with
    # the smoothing on and 2.5445e-07 at beta=1000, against 2.9431e-07 for the same
    # grid point-sampled. All three time_matched with the full signal. MEEP's OWN
    # get_epsilon() diagnostic says the same thing independently at beta=1000: it spans
    # (0.9730, 6.3529) on a cell whose declared endpoints are 1 and 6. So the read is
    # right and the premise is wrong, for this branch only.
    #
    # Scoped to EXACTLY MEEP's branch condition (meepgeom.cpp:1078-1086: a MATERIAL_GRID
    # with `do_averaging`, reached only when the structure was built with anisotropic
    # averaging at all), so every other cell — including a point-sampled grid, which
    # stays inside its endpoints because `matgrid_val` interpolates clipped weights and
    # `tanh_projection` maps [0, 1] onto itself — keeps the check unchanged. What
    # replaces it for these cells is measurement, not nothing: the material_grid_*
    # parity cases in test_from_meep carry a half-cell-shifted control, which is the
    # registration slip this spread test was written to catch.
    smoothed_grids = [grid for grid in _material_grids_of(mp, sim)
                      if getattr(grid, "do_averaging", False)] if getattr(
                          sim, "eps_averaging", True) else []
    # THE OTHER CELL WHERE THERE IS NO BRACKET AT ALL. A per-point epsilon route names
    # no medium — a material_function returns one per POINT, an ndarray and an
    # epsilon_input_file name values rather than materials — so `declared` above speaks
    # only for whatever plain media the cell also happens to declare, and on
    # test_simulation.py's `test_numpy_epsilon` it is empty. There is nothing to bound
    # the sampled permittivity with, and inventing a bound by sampling the callable
    # would only compare this lift's sampling against its own.
    #
    # What the spread test exists to catch is a registration slip, and that is caught
    # here regardless by `_require_bulk_matches_meep` below — which cross-checks every
    # bulk cell against MEEP's own centred diagnostic and is, by its own docstring, the
    # check the historical half-cell defects would have failed. The per-point
    # positivity and finiteness checks above also stand unchanged.
    bracketed = not smoothed_grids and not routes and bool(declared)

    partner_names = {direction: name for name, direction in directions.items()}
    volumes: dict[str, Any] = {}
    coupling_rows: dict[str, dict[str, Any]] = {}
    declined_any = None  # UNION of the three components' declined masks; see below.
    total = 3 * grid.nx
    done = 0
    for name in E_COMPONENTS:
        component = components[name]
        own = directions[name]
        others = [axis for axis in directions.values() if axis != own]
        indices = _lattice_indices(gv, grid, name, component, lattice_per_unit)
        counts = tuple(len(axis) for axis in indices)
        values = numpy.empty(counts, dtype=numpy.float64)
        coupling = {axis: numpy.empty(counts, dtype=numpy.float64) for axis in others}
        if not cylindrical:
            # A Dcyl ivec has no phi slot to probe: the invariant axis is dropped
            # at construction, so there is no stride for this check to measure.
            _require_invariant_axes_are_strideless(grid, gv, indices, component, own, get_chi1inv, ivec, name)
        for i, ix in enumerate(indices[0]):
            for j, jy in enumerate(indices[1]):
                for k, kz in enumerate(indices[2]):
                    location = ivec(ix, jy, kz)
                    values[i, j, k] = get_chi1inv(component, own, location, 0).real
                    for axis in others:
                        entry = complex(get_chi1inv(component, axis, location, 0))
                        if abs(entry.imag) > _CHI1INV_OFFDIAG_TOLERANCE:
                            raise ValueError(
                                f"MEEP reports a COMPLEX off-diagonal chi1inv on {name} "
                                f"({entry!r} at lattice point ({ix}, {jy}, {kz})); the inverse "
                                f"tensor of a lossless structure is real, so this cell holds a "
                                f"material model the lift does not understand."
                            )
                        coupling[axis][i, j, k] = entry.real
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)
        # Non-positive is refused, EXCEPT MEEP's perfect-conductor sentinel: a metal
        # cell reads chi1inv = -1e-20 (driver.PEC_CHI1INV_SENTINEL), which inverts to
        # the -1e20 permittivity MEEP itself declared and which the ordinary
        # constitutive update drives to ~0. Everything else non-positive stays refused.
        if not numpy.all(numpy.isfinite(values)) or numpy.any(
            (values <= 0.0) & ~is_pec_chi1inv(values)
        ):
            raise ValueError(
                f"MEEP reports a non-positive or non-finite chi1inv on {name}; the lift cannot "
                f"invert it to a permittivity."
            )
        volumes[name] = 1.0 / values
        for axis, entries in coupling.items():
            if not numpy.all(numpy.isfinite(entries)):
                raise ValueError(
                    f"MEEP reports a non-finite off-diagonal chi1inv on {name}; nothing was stepped."
                )
            if numpy.any(entries):
                coupling_rows.setdefault(name, {})[partner_names[axis]] = entries
        declined = _unowned_wall_planes(sim, gv, grid, component, own, indices, values, ivec)
        # THE UNION, and it has to be the union. The bulk cross-check below classifies and
        # compares `stacked[0]` — Ex — so masking each component with its OWN mask would
        # leave the case that motivated this untouched: on
        # test_mode_decomposition.py::test_oblique_waveguide_backward_mode (35-degree
        # rotated index-3.5 block, 240x240 at resolution 20) the single offending cell is
        # (i=0, j=22), where Ex's window opens at corner+1 and is genuinely OWNED, while
        # Ey's and Ez's open AT the corner and are both declined. The cell is excludable
        # only because Ey and Ez are unreadable there — the cell-centred diagnostic mixes
        # all three, so one unreadable component poisons the comparison for the cell.
        declined_any = declined if declined_any is None else (declined_any | declined)
        # Guarded on KEEP, not on the mask: a component whose every sampled point
        # was declined (a one-pixel metallic axis can put the whole volume on the
        # corner plane) has no MEEP-answered sample to check — the full array of
        # literal defaults then faces the span test rather than an empty .min().
        keep = ~declined
        sampled_volume = volumes[name][keep] if keep.any() else volumes[name]
        # The same exclusion the declared bracket took, on the sampled side: a PEC cell
        # is not an average and has no bracket to lie inside. A component that is ALL
        # metal has nothing left to bound, so it drops out of the test entirely rather
        # than comparing an empty array's min().
        sampled_volume = sampled_volume[~is_pec_epsilon(sampled_volume)]
        if sampled_volume.size == 0:
            continue
        spread = (float(sampled_volume.min()), float(sampled_volume.max()))
        # `bracketed` is false for the two cells where the declared media are not a bound
        # on the sampled permittivity — a smoothed MaterialGrid, where MEEP extrapolates
        # past both endpoints, and a per-point epsilon route, where there is no declared
        # bracket at all. Both are argued above, at `smoothed_grids`.
        if bracketed and (spread[0] < low - 1e-4 * high or spread[1] > high + 1e-4 * high):
            raise ValueError(
                f"the permittivity sampled for {name} spans {spread}, outside the range "
                f"[{low:g}, {high:g}] spanned by the declared media. A subpixel average lies "
                f"between the materials it averages, so a value outside them means this lift read "
                f"the wrong lattice points; nothing was stepped."
            )
    _require_bulk_matches_meep(sim, driver, volumes, declined=declined_any)
    driver.set_epsilon_components(volumes, chi1inv_offdiagonal=coupling_rows or None)


def _unowned_wall_planes(sim, gv, grid, component, own, indices, values, ivec):
    """Mask of sampled lattice points MEEP answers with its outside-the-cell default.

    ``grid_volume::owns`` requires a point STRICTLY above the chunk's little corner
    (``o > 0`` per axis, vec.cpp:445-463), so the global low-boundary plane of a
    non-wrapping axis is owned by no chunk, and ``fields::get_chi1inv`` then
    "default[s] to vacuum outside computational cell" — literal 1.0 on the
    diagonal (monitor.cpp:180-183) — rather than reading the stored ghost. This
    grid stores that plane (its N-cell window opens at the corner on every
    shift-0 axis), so a structured lift over a NON-vacuum background reads a
    wall row of vacuum where the declaration has none: measured on ring_gds.py
    (Si ring in SiO2, metallic walls behind PML), the set of points reading
    exactly 1.0 was bit-for-bit the set ``structure::get_chi1inv`` reports
    unowned — the whole j=0 row for Ex, i=0 for Ey, both for Ez, 425/425/851
    points — with every interior prism sample in range.

    The mask feeds two CHECKS — the declared-media spread check (per component)
    and :func:`_require_bulk_matches_meep` (the union over the three, because
    that one classifies and compares Ex while the unreadable component at a
    failing cell may be Ey or Ez) — and nothing else; the installed value
    stays MEEP's own answer. It is inert by MEEP's own arithmetic: an unowned
    plane is never timestepped there (``update_eh`` walks owned points), its
    tangential E is the wall condition both engines pin, and every
    previously-measured structured lift (vacuum backgrounds, where 1.0 IS the
    declared medium) keeps its installed volumes bit-for-bit.

    Membership is MEASURED per point, not derived from the geometry: only a
    candidate that reads exactly 1.0 from ``fields::get_chi1inv`` AND 0.0 from
    ``structure::get_chi1inv`` — which runs the same symmetry loop but neither
    translates across a periodic face (no ``locate_point_in_user_volume``) nor
    substitutes a default (monitor.cpp:226-240) — is masked. A wrapped low plane
    is translated to its image by the fields read (real value, not 1.0), a
    folded one is rescued by ``S.transform`` in both reads, and a genuine
    vacuum cell reads 1.0 from BOTH, so all three stay inside the spread check.
    Candidates are restricted to the corner planes (``indices[axis][0] ==
    little_corner``, shift-0 axes only), which bounds the extra SWIG reads to
    the wall rows.
    """
    mask = numpy.zeros(values.shape, dtype=bool)
    structure = getattr(sim, "structure", None)
    if structure is None:
        return mask
    corner = gv.little_corner()
    if getattr(grid, "cylindrical", False):
        corner_by_axis = (corner.r(), None, corner.z())
    else:
        corner_by_axis = (corner.x(), corner.y(), corner.z())
    for axis in range(3):
        if grid.is_invariant(axis) or corner_by_axis[axis] is None:
            continue
        if not len(indices[axis]) or indices[axis][0] != corner_by_axis[axis]:
            continue  # This component's window opens above the corner (Yee shift 1).
        plane = [slice(None)] * 3
        plane[axis] = slice(0, 1)
        for point in numpy.argwhere(values[tuple(plane)] == 1.0):
            i, j, k = (int(value) for value in point)
            location = ivec(indices[0][i], indices[1][j], indices[2][k])
            if complex(structure.get_chi1inv(component, own, location, 0)) == 0.0:
                mask[i, j, k] = True
    return mask


def _lattice_indices(gv, grid, name: str, component, lattice_per_unit: float) -> tuple:
    """This component's Yee coordinates as MEEP integer-lattice indices, pinned to MEEP's grid.

    Returns three ``int`` lists, one per axis, ready for ``mp.ivec``.

    THE PIN, and why it is an equality rather than a tolerance. MEEP's integer
    lattice has unit ``dx/2``, and its own ``grid_volume`` states where this
    component's first point sits::

        little_corner()[d] + iyee_shift(component)[d]

    — verified against ``little_owned_corner``, which MEEP derives as
    ``little_corner + 2 - iyee_shift``, on plain, odd-cell-count and
    mirror-folded grids. Every sample index here is required to match that
    exactly and to step by exactly 2, so the whole family of registration errors
    is one integer comparison: half a cell lands on the wrong parity (where
    ``grid_volume::index`` does not complain — it reads the NEIGHBOUR), a whole
    cell lands on the right parity and the wrong lattice point, and a wrong
    origin shifts the run. All three still step to a smooth complete field, so
    none of them can be left to be noticed downstream.

    Raising rather than rounding is the same decision: ``round_vec`` would snap an
    off-lattice coordinate onto a neighbouring point and hand back a perfectly
    plausible permittivity for the wrong cell.
    """
    shift = gv.iyee_shift(component)
    corner = gv.little_corner()
    limit = gv.big_corner()
    if getattr(grid, "cylindrical", False):
        # A Dcyl ivec answers .r()/.z() only; the phi entries are placeholders
        # (this engine's axis 1) that the iveccyl construction never reads.
        wanted = (shift.r(), 0, shift.z())
        first_expected = (corner.r() + wanted[0], 0, corner.z() + wanted[2])
        last_allowed = (limit.r(), 0, limit.z())
    else:
        wanted = (shift.x(), shift.y(), shift.z())
        first_expected = (corner.x() + wanted[0], corner.y() + wanted[1], corner.z() + wanted[2])
        last_allowed = (limit.x(), limit.y(), limit.z())
    indices = []
    for axis, coordinates in enumerate(component_coordinates(grid, name)):
        if grid.is_invariant(axis):
            # MEEP has no lattice on this axis at all: it is not in LOOP_OVER_DIRECTIONS,
            # so `iyee_shift` reports 0 there (measured: Ez's shift is (0, 0, 0) in a D2
            # grid_volume, not (0, 0, 1)), `little_corner` and `big_corner` are both 0,
            # and `stride` is 0 — every index reads the same data. This engine's own
            # coordinate for the axis is therefore not a lattice point of MEEP's and
            # must not be pinned against one; MEEP's own corner is used verbatim, and
            # _require_invariant_axes_are_strideless is what checks that the choice
            # genuinely cannot matter rather than assuming it.
            indices.append([first_expected[axis]])
            continue
        exact = numpy.asarray(to_numpy(coordinates), dtype=numpy.float64) * lattice_per_unit
        rounded = numpy.rint(exact)
        drift = float(numpy.abs(exact - rounded).max()) if rounded.size else 0.0
        if drift > _LATTICE_TOLERANCE:
            raise ValueError(
                f"{name}'s Yee coordinates on {AXIS_NAMES[axis]} are not on MEEP's integer lattice "
                f"(worst drift {drift:g} of a half cell). This engine's grid and MEEP's have "
                f"diverged, so every sampled permittivity would belong to a neighbouring cell "
                f"while still looking like a structure. Nothing was stepped."
            )
        rounded = rounded.astype(numpy.int64)
        if int(rounded[0]) != first_expected[axis]:
            raise ValueError(
                f"{name}'s first sample point on {AXIS_NAMES[axis]} is lattice index "
                f"{int(rounded[0])} where MEEP's own grid puts it at {first_expected[axis]} "
                f"(gv.little_corner + gv.iyee_shift). That is a registration slip of "
                f"{abs(int(rounded[0]) - first_expected[axis]) / 2:g} cell(s): the permittivity "
                f"would be read off a displaced structure and stepped without complaint, because "
                f"MEEP's grid_volume::index accepts any ivec and simply returns whatever cell it "
                f"lands in. Nothing was stepped."
            )
        steps = numpy.unique(numpy.diff(rounded)) if rounded.size > 1 else numpy.array([2])
        if steps.tolist() != [2]:
            raise ValueError(
                f"{name}'s sample points on {AXIS_NAMES[axis]} step by {steps.tolist()} lattice "
                f"units where one cell is exactly 2. Nothing was stepped."
            )
        values = [int(value) for value in rounded]
        overhang = values[-1] - last_allowed[axis] if values else 0
        if overhang == 1 and _lattice_axis_stores_far_ghost(grid, axis):
            # The far ghost slot of a folded PERIODIC axis (either parity): the
            # stored array runs one shift-1 slot past MEEP's big_corner
            # (Grid.stored_cells vs owned_cells), and MEEP's structure chunks do
            # not cover that point — grid_volume::index would silently read
            # whatever cell the ivec lands in. Its material value is the mirror
            # image about the second mirror at doubled n_full: chi1inv's
            # diagonal is even under the reflection (the off-diagonals, which
            # are odd, are refused on any folded grid before this runs), so the
            # slot samples the image point 2*n_full - p. In practice the fill
            # pass overwrites the slot's field every sub-step, but the
            # constitutive update still divides by the epsilon stored there, so
            # the value must be the image's, not a neighbour's.
            # The second mirror in MEEP's half-cell lattice: the fold plane
            # (icenter = little_corner + 2 on a halved axis) plus half the
            # lattice vector, which is n_full half-cell units (L/2 = n_full *
            # dx/2). The ghost slot at mirror + 1 images mirror - 1.
            mirror = (first_expected[axis] - wanted[axis]) + 2 \
                + (grid.nx_full, grid.ny_full, grid.nz_full)[axis]
            values[-1] = 2 * mirror - values[-1]
        elif overhang > 0:
            raise ValueError(
                f"{name}'s last sample point on {AXIS_NAMES[axis]} is lattice index "
                f"{values[-1]}, past MEEP's own big_corner {last_allowed[axis]}. "
                f"Nothing was stepped."
            )
        indices.append(values)
    return tuple(indices)


def _lattice_axis_stores_far_ghost(grid, axis: int) -> bool:
    """Does this axis store the far ghost slot past MEEP's corner? (Folded periodic, even N.)"""
    reader = getattr(grid, "owned_cells", None)
    if not callable(reader):
        return False
    return grid.is_mirrored(axis) and grid.stored_cells(axis) > reader(axis)


def _require_invariant_axes_are_strideless(
    grid, gv, indices, component, direction, get_chi1inv, ivec, name: str
) -> None:
    """Check that moving along an invariant axis really does read the same MEEP sample.

    :func:`_lattice_indices` takes MEEP's own corner for an invariant axis on the
    grounds that ``grid_volume::num_changed`` leaves ``stride(d) = 0`` for a direction
    outside ``LOOP_OVER_DIRECTIONS``, so the index cannot select anything. That is a
    statement about MEEP's internals, and it is exactly the kind of statement that
    stops being true one release later — at which point every reduced lift would read
    the permittivity off one plane of a structure it thinks is uniform, and step a
    smooth field for it.

    So it is measured: the same point is read at the declared index and two cells away
    on the invariant axis, and the three must be bit-identical. Six reads per lift,
    once per component.
    """
    if not grid.has_invariant:
        return
    for axis in range(3):
        if not grid.is_invariant(axis):
            continue
        base = [indices[0][0], indices[1][0], indices[2][0]]
        reference = get_chi1inv(component, direction, ivec(*base), 0)
        for offset in (2, -2):
            probe = list(base)
            probe[axis] = base[axis] + offset
            value = get_chi1inv(component, direction, ivec(*probe), 0)
            if value != reference:
                raise ValueError(
                    f"MEEP's chi1inv for {name} CHANGES along the {AXIS_NAMES[axis]} axis, which "
                    f"dimensions={grid.dimensions} makes translationally invariant: index "
                    f"{base[axis]} gives {reference!r} and {probe[axis]} gives {value!r}. This "
                    f"engine samples the invariant axis at MEEP's own corner because "
                    f"grid_volume::stride is 0 for a direction outside LOOP_OVER_DIRECTIONS, so "
                    f"the index cannot select anything — that no longer holds, and the lift would "
                    f"otherwise install one arbitrary plane of a structure it believes is uniform. "
                    f"Nothing was stepped."
                )


def _restore_dropped_axes(array, driver):  # A reduced MEEP array back to this engine's rank.
    """Re-insert the length-1 axes MEEP's ``array_slice`` dropped, so a 3-D index works.

    Only the shape changes: MEEP reports one axis per direction of more than one point
    (a 2-D ``get_epsilon`` is ``(nx+1, ny+1)``), while every array this engine indexes
    carries all three. ``reshape`` is used rather than ``atleast_3d``/``expand_dims`` at
    a guessed position, because which axes are missing is known exactly —
    ``Fields.meep_array_axes``, the same definition the readback uses — and in the
    ORDER it gives, which is not always ascending: a Dcyl array is (z, r), so the
    transpose back into this engine's x, y, z order happens here rather than being
    assumed away. Written against the tuple instead of against Dcyl's usual two axes
    because that pair is not fixed either — a cell with one z cell reports ``(0,)``,
    r alone, and the hardcoded ``(2, 0)`` reshape this replaced raised IndexError on
    ``ring-cyl.py``'s own array.
    """
    keep = driver.fields.meep_array_axes()
    if len(keep) == 3:
        return array
    # Positions of `keep`, sorted by the axis each names: the permutation that puts
    # MEEP's order into this engine's. Identity wherever `keep` is already ascending.
    ascending = sorted(range(len(keep)), key=keep.__getitem__)
    ordered = array.transpose(ascending) if len(keep) > 1 else array
    shape = [1, 1, 1]
    for position, axis in enumerate(sorted(keep)):
        shape[axis] = ordered.shape[position]
    return ordered.reshape(tuple(shape))


def _require_bulk_matches_meep(sim, driver, volumes, declined=None) -> None:
    """Cross-check the sampled volumes against MEEP's own cell-centred diagnostic.

    ``sim.get_epsilon()`` is useless for *installing* a structure, but it is a
    genuinely independent witness for *checking* one. Away from an interface the
    permittivity is locally constant, so its trace average and its bilinear
    interpolation onto the cell centre are both that same constant — which means
    every cell where the three sampled components agree with each other must also
    agree with MEEP's diagnostic, exactly.

    WHAT THIS CHECK IS AND IS NOT, restated after measurement. It is NOT the
    registration pin. The pin is :func:`_lattice_indices`' integer equality
    against ``gv.little_corner() + gv.iyee_shift(c)``, which is what refuses the
    half-cell, whole-cell and off-lattice gates by name (see
    ``test_a_mis_sampled_permittivity_is_caught_before_a_single_step``, whose
    expected messages all come from there). This docstring used to claim it was
    "the registration check the historical half-cell defects would have failed";
    that claim was MEASURED FALSE on the oblique-waveguide cell (35-degree rotated
    index-3.5 block, 240x240 at resolution 20) with the pin bypassed: a +0.5-cell
    x shift changes 53 of 172800 sampled values and this check does not fire, and a
    +1-cell shift fires only on the wall artifact below (bit-identical error, and
    2.51e-07 away from the walls). So it is a SECOND WITNESS, and its measured
    power is against right coordinates with WRONG VALUES — the
    ``wrong_material_installed`` gate, where a permittivity MEEP never built is
    handed straight to this function and refused "in the BULK". Whether it also
    catches a shift is geometry-dependent and is not claimed here.

    TWO KNOWN BLIND SPOTS, both deliberate.

    THE UNOWNED WALL PLANES, excluded via ``declined``. ``fields::get_chi1inv``
    substitutes literal vacuum for a point no chunk ``owns`` (monitor.cpp:181-183,
    "default to vacuum outside computational cell") while ``structure::get_chi1inv``
    returns 0.0 there (monitor.cpp:238), so the global low-boundary plane of a
    non-wrapping axis is unreadable through EITHER public reader — while
    ``sim.get_epsilon()``'s array slice reaches the stored chunk data and reports the
    real material. Requiring bulk equality there requires the impossible.
    :func:`_unowned_wall_planes` measures the set; the caller passes the UNION over
    the three components, because this function classifies and compares Ex and the
    cell that fails may be owned for Ex and declined for Ey/Ez. Measured on the
    oblique-waveguide cell: 479 of 57600 cells masked, one of them the whole
    4.364e-03 error, 54218 of 54639 bulk cells (99.2 %) still checked, residual
    2.50991e-07 against a 1.225e-03 tolerance — and the resulting lift matches CPU
    MEEP 1.33.0 at parity_rel_l2 = 1.1391e-06 over 2000 steps, time_matched, which
    is what says the substituted vacuum is inert (MEEP never timesteps an unowned
    point, and the tangential E there is the wall condition both engines pin).
    The exclusion gives up 421 checked cells and, on that geometry, no measured
    detection power — the two shift controls above are not caught either way.

    THE ON-LATTICE INTERFACE, left in place as a known false positive: an interface
    that lands EXACTLY on the Yee sampling planes of a cell defeats the bulk classifier —
    all three per-component samples read the same pure material while MEEP's
    centre diagnostic is subpixel-smoothed, so a genuinely correct lift is
    refused. Measured on zone_plate.py forced to resolution 18 (the parity
    harness's cell cap): the ring faces at z = ±0.25 land at z*a = ±4.5, half-
    integer — on the lattice — and the check reports 0.3215 across 773 cells that
    are all one row from a ring face, every one holding OUR pure 2.25 against
    MEEP's smoothed 1.588 whose z-neighbours are 2.25 and 1.0. At the script's
    own resolution 25 (z*a = ±6.25, off-lattice) the same lift passes clean.
    Excluding reference-non-constant neighbourhoods would remove the false
    positive but also the interface-adjacent rows that give this check its power
    against one-cell registration shifts, so the sharp check stays; a capped
    parity row that dies here is a cap artifact, and the row should say so.
    """
    grid = driver.grid
    reference = _restore_dropped_axes(
        numpy.asarray(sim.get_epsilon(), dtype=numpy.float64), driver
    )
    reported = driver.fields.meep_array_axes()
    stacked = numpy.stack([volumes[name] for name in E_COMPONENTS])
    # A folded periodic axis stores one cell past MEEP's own window at either
    # count parity (``big_corner`` and its ghost slot, ``Grid.stored_cells``
    # vs ``owned_cells``). That cell's centre sits past the diagnostic's last
    # tic, and its material is the imaged interior row ``_lattice_indices``
    # already samples — so the bulk comparison clips to the owned cells rather
    # than reading past MEEP's window.
    clip = tuple(
        slice(0, grid.owned_cells(axis)) if _lattice_axis_stores_far_ghost(grid, axis)
        else slice(None)
        for axis in range(3)
    )
    stacked = stacked[(slice(None), *clip)]
    sampled = stacked[0]
    # Where each of our cells sits in MEEP's array is READ from MEEP, not derived:
    # get_array_metadata returns the diagnostic's own coordinate tics, and a folded
    # axis keeps the whole cell there while this grid stores only its half. Deriving
    # the offset instead would be one more place to be half a cell wrong.
    if getattr(grid, "cylindrical", False):
        # MEEP's get_array_metadata refuses Dcyl outright ("does not support
        # cylindrical coordinates"), so the window cannot be READ there. It does
        # not need to be: a Dcyl epsilon array has no duplicate planes and no
        # folded halves (measured (nz, nr) exactly, _require_grids_agree), so
        # our cell (i, k) IS MEEP's entry (i, k) after _restore_dropped_axes —
        # zero offset by the same registration the get_array parity pins at
        # 2.6e-07.
        window = [slice(0, sampled.shape[axis]) for axis in range(3)]
        metadata = None
    else:
        metadata = sim.get_array_metadata()
        window = []
    for axis in range(3):
        if metadata is None:
            break
        if axis not in reported:
            # MEEP reports a single tic of 0.0 for a direction it does not have, and one
            # plane for a direction of one cell — neither is a coordinate to register
            # against, because there is only one plane to choose. Take it without matching.
            window.append(slice(0, 1))
            continue
        tics = numpy.atleast_1d(numpy.asarray(metadata[axis], dtype=numpy.float64))
        first = float(to_numpy(grid.axis_origin(axis))) + 0.5 * grid.dx
        offset = int(numpy.argmin(numpy.abs(tics - first)))
        if (
            abs(tics[offset] - first) > 1e-6 * grid.dx
            or offset + sampled.shape[axis] > tics.size
        ):
            raise ValueError(
                f"this grid's first {AXIS_NAMES[axis]} cell centre {first:.9g} is not one of MEEP's "
                f"epsilon coordinates (nearest {tics[offset]:.9g}); the two grids are registered "
                f"differently and nothing was stepped."
            )
        window.append(slice(offset, offset + sampled.shape[axis]))
    diagnostic = reference[tuple(window)]
    # A bulk cell is one whose whole 3x3x3 neighbourhood holds a single permittivity
    # in all three components. The neighbourhood, not the cell: each component's Yee
    # point is half a cell off the centre on its own axis, so MEEP's interpolation
    # onto the centre reaches diagonally into the adjacent cells, and a face-only
    # test calls a cell bulk that is still reading an interface. (An anisotropic
    # medium has no bulk cell by this definition — the trace average matches no
    # component — so this check simply stands down there.)
    low = high = stacked
    for axis in range(1, 4):
        low = numpy.minimum(numpy.minimum(low, numpy.roll(low, 1, axis)), numpy.roll(low, -1, axis))
        high = numpy.maximum(
            numpy.maximum(high, numpy.roll(high, 1, axis)), numpy.roll(high, -1, axis)
        )
    interior = (high.max(axis=0) - low.min(axis=0)) <= 1e-6 * high.max(axis=0)
    # THE PERFECT-CONDUCTOR NEIGHBOURHOOD, excluded explicitly — the third blind spot,
    # and the same argument as the unowned wall planes above: the two witnesses cannot
    # agree there, so requiring it requires the impossible. ``fields::get_eps`` is
    # ``nc / sum_c chi1inv(c)`` (monitor.cpp:202-212), a SUM over the electric
    # components, so a 2-D TM metal cell whose Ez chi1inv is -1e-20 and whose Ex/Ey are
    # 1.0 reads exactly 1.5 in the diagnostic while the stored per-component
    # permittivity is -1e20. Both readings are right; they measure different things.
    #
    # It is excluded BY NAME rather than left to the bulk classifier, which stands down
    # there only by accident: with all three components at -1e20 the classifier's
    # tolerance ``1e-6 * high.max()`` evaluates to -1e14 and ``0 <= -1e14`` is False, so
    # the cell is called non-bulk for an arithmetic reason rather than a physical one.
    # A single float32 wobble, or a metal that is anisotropic in only one component,
    # would flip that accident and refuse a correctly-read cell.
    interior &= ~is_pec_epsilon(stacked).any(axis=0)
    if declined is not None:
        # The same `clip` the volumes took: the mask is sampled on the component
        # lattice, one entry per stored cell, so a folded axis's far ghost has to
        # come off it too or the shapes disagree.
        interior &= ~numpy.asarray(declined)[clip]
    if not interior.any():
        return  # Every cell reads an interface (or an anisotropic medium): nothing to check.
    error = numpy.abs(diagnostic[interior] - sampled[interior]).max()
    scale = float(numpy.abs(sampled[interior]).max())
    if error > 1e-4 * scale:
        raise ValueError(
            f"the sampled permittivity disagrees with MEEP's own cell-centred diagnostic by "
            f"{error:.6g} in the BULK, where the two must be identical (a locally constant "
            f"permittivity interpolates to itself). The sampling is registered against a "
            f"different lattice than MEEP's structure — a half-cell shift moves every interface "
            f"and still produces a smooth field. Nothing was stepped."
        )


# The two frequencies the conductivity read is taken at. Any nonzero pair works —
# monitor.cpp's conductivity factor is `1 + i*sigma/frequency`, so `w * Im(1/chi1inv)`
# returns sigma itself at every one of them — and taking TWO is the self-check: a
# disagreement means the D row carried something frequency-dependent, which sigma_D is
# not. Measured agreement on the cells this was built against: 1.1e-16 absolute on a
# sigma = 0.7 block (this implementer's own read), 3.3e-17 and 5.6e-16 on the recon's
# block and graded-grid cells.
_CONDUCTIVITY_PROBE_FREQUENCIES = (0.9, 0.3)
# Relative, because sigma spans four orders of magnitude across the cases: 0.7 for a
# declared lossy medium, 0.785 for a MaterialGrid damping and 98.6 inside an
# mp.Absorber. Nine orders below the tightest real defect and seven above the measured
# round-off.
_CONDUCTIVITY_AGREEMENT_TOLERANCE = 1e-9
D_CONDUCTIVITY_SIGNATURE_INDEX = 8


def _conductivity_structured(mp, sim, materials) -> bool:
    """True when ``D_conductivity`` VARIES across the cell, so it has to be read per point.

    Two ways a cell gets there, and the second is why this is not simply a signature
    comparison: a ``mp.MaterialGrid`` with ``damping`` puts ``u*(1-u)*damping`` into
    ``D_conductivity_diag`` at every point (meepgeom.cpp:623-626) while BOTH its
    endpoint media declare a conductivity of zero, so no signature difference exists to
    find. Only called after the gate accepted the simulation.
    """
    signatures = {_medium_signature(material) for material in materials}
    if len({signature[D_CONDUCTIVITY_SIGNATURE_INDEX] for signature in signatures}) > 1:
        return True
    return any(float(getattr(grid, "damping", 0.0) or 0.0) != 0.0
               for grid in _material_grids_of(mp, sim))


def _read_conductivity_volumes(mp, sim, driver, progress_cb=None) -> dict:
    """Read MEEP's per-point ``D_conductivity`` back off its own D-row ``chi1inv``.

    THE READ, transcribed rather than inferred. ``structure_chunk::get_chi1inv_at_pt``
    (monitor.cpp:263-355) answers a NONZERO frequency by rebuilding the tensor for
    whichever family the component belongs to — and for a D component that family is
    ``comp_list = {Dx, Dy, Dz}`` with ``my_stuff = D_stuff``. Two things are then true
    of the D row and of nothing else:

    * ``chi1inv[Dx][X]`` is never allocated. ``structure::set_materials``
      (structure.cpp:374-384) fills ``chi1inv`` under ``FOR_ELECTRIC_COMPONENTS`` /
      ``FOR_MAGNETIC_COMPONENTS`` and fills ``conductivity[c][d]`` under
      ``FOR_D_AND_B(c)``. So the instantaneous D tensor is the IDENTITY, and the
      permittivity — the thing that makes ``chi1inv`` on the E row inseparable from
      the loss — is simply not in this row.
    * ``chiP[D_stuff]`` is empty for the same reason, so no susceptibility is summed
      in either.

    What survives is exactly monitor.cpp:339-343's
    ``eps = (1 + i*conductivity/frequency) * 1``, inverted once more on the way out.
    Hence ``sigma_D = w * Im(1/chi1inv(D_c, own, iloc, w))``, EXACTLY, with no
    inversion, no lookup and no patched MEEP. Measured on a declared
    ``D_conductivity=0.7`` block at resolution 20: recovered 0.699999988 (float32
    storage), worst error 1.19e-08 over the plane, and 1.11e-16 between w = 0.3 and
    w = 0.9.

    THE TWO SELF-CHECKS, because both premises above are places to be confidently
    wrong and neither is argued into correctness here:

    * **The identity check.** ``get_chi1inv(D_c, own, iloc, 0)`` must be exactly 1.0 —
      monitor.cpp:268 returns that literal when ``chi1inv[c][d]`` is null. A MEEP that
      allocated the D row would answer with a permittivity instead, and every recovered
      sigma would silently be ``eps * sigma`` rather than ``sigma``. Nothing in the
      stepped field would say so; the run would complete, smooth and wrong.
    * **The two-frequency check.** sigma_D is frequency-INDEPENDENT by construction, so
      the same read at a second frequency must return the same number. A D-side
      susceptibility chain — which this row is not supposed to have — would break it.

    Either failure refuses with the value and the Yee point, the way
    :class:`~.sigma_lookup.SigmaVerification` refuses, rather than installing a
    doubtful volume.

    WHAT THE READ INCLUDES, and it is not a defect: ``mp.Absorber``. MEEP's
    ``geom_epsilon::conductivity`` adds each absorber face's ramp INTO the same
    ``conductivity[c][d]`` array (meepgeom.cpp:1596-1625), so what comes back is
    material + absorber summed by MEEP itself — measured peak 98.565 on an
    ``mp.Absorber(0.5)`` cell. That is strictly more faithful than reconstructing the
    ramp, and it is why the caller passes ``absorber_included=True``. ``mp.PML`` does
    NOT appear (measured 0 of 6400 points nonzero): its profile lives in
    ``structure_chunk::sigma``, a different array this read never touches.

    Returns:
        ``{'Ex': volume, 'Ey': ..., 'Ez': ...}`` in the driver's grid shape, ready for
        :meth:`~.driver.FdtdDriver.set_conductivity` (which keys on the D components,
        so the caller renames).

    Raises:
        MeepSimulationNotLiftable: when either self-check fails.
    """
    fields = sim.fields
    gv = fields.gv
    grid = driver.grid
    get_chi1inv = fields.get_chi1inv
    lattice_per_unit = 2.0 * float(sim.resolution)
    # Cylindrical is refused at the gate (_check_structured_conductivity): the Dcyl
    # spelling of this read is UNMEASURED, and an unmeasured registration on a graded
    # loss volume is exactly the kind of thing that completes and is wrong.
    directions = {"Ex": mp.X, "Ey": mp.Y, "Ez": mp.Z}
    e_components = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez}
    d_components = {"Ex": mp.Dx, "Ey": mp.Dy, "Ez": mp.Dz}
    probe = _CONDUCTIVITY_PROBE_FREQUENCIES

    volumes: dict[str, Any] = {}
    total = max(3 * grid.nx, 1)
    done = 0
    for name in E_COMPONENTS:
        own = directions[name]
        d_component = d_components[name]
        # The E component's Yee shift, because a D component shares it — MEEP stores
        # D at the same lattice point as its E partner, which is what makes
        # `_lattice_indices`' integer pin the right registration for both.
        indices = _lattice_indices(gv, grid, name, e_components[name], lattice_per_unit)
        counts = tuple(len(axis) for axis in indices)
        values = numpy.zeros(counts, dtype=numpy.float64)
        for i, ix in enumerate(indices[0]):
            for j, jy in enumerate(indices[1]):
                for k, kz in enumerate(indices[2]):
                    location = mp.ivec(int(ix), int(jy), int(kz))
                    instantaneous = complex(get_chi1inv(d_component, own, location, 0))
                    if instantaneous != 1.0:
                        raise MeepSimulationNotLiftable([
                            f"MEEP answers the diagonal chi1inv of {name}'s D component at "
                            f"lattice point ({ix}, {jy}, {kz}) and frequency 0 with "
                            f"{instantaneous!r}, not the literal 1.0 that monitor.cpp:268 returns "
                            f"for an unallocated row. The per-point D_conductivity read assumes "
                            f"the instantaneous D tensor is the IDENTITY — structure::set_materials "
                            f"fills chi1inv for E and H only (structure.cpp:374-384) — and every "
                            f"recovered sigma on a MEEP that allocated it would be eps*sigma "
                            f"rather than sigma, in a run that completes and looks smooth. "
                            f"Nothing was stepped."
                        ])
                    sampled = []
                    for frequency in probe:
                        entry = complex(get_chi1inv(d_component, own, location, frequency))
                        if entry == 0.0 or not numpy.isfinite(entry):
                            raise MeepSimulationNotLiftable([
                                f"MEEP reports chi1inv = {entry!r} on {name}'s D row at lattice "
                                f"point ({ix}, {jy}, {kz}) and frequency {frequency:g}; the "
                                f"conductivity read inverts it and cannot."
                            ])
                        sampled.append(frequency * (1.0 / entry).imag)
                    scale = max(abs(sampled[0]), abs(sampled[1]), 1.0)
                    if abs(sampled[0] - sampled[1]) > _CONDUCTIVITY_AGREEMENT_TOLERANCE * scale:
                        raise MeepSimulationNotLiftable([
                            f"the D_conductivity read disagrees with itself on {name} at lattice "
                            f"point ({ix}, {jy}, {kz}): {sampled[0]!r} at frequency {probe[0]:g} "
                            f"against {sampled[1]!r} at {probe[1]:g}, a relative difference of "
                            f"{abs(sampled[0] - sampled[1]) / scale:.3e} against a tolerance of "
                            f"{_CONDUCTIVITY_AGREEMENT_TOLERANCE:g}. sigma_D is "
                            f"frequency-INDEPENDENT (monitor.cpp:339-343 multiplies by "
                            f"1 + i*sigma/frequency), so a disagreement means this row carried "
                            f"something else — a D-side susceptibility chain, which this read has "
                            f"no term for. Nothing was stepped."
                        ])
                    values[i, j, k] = sampled[0]
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)
        if not numpy.all(numpy.isfinite(values)):
            raise MeepSimulationNotLiftable([
                f"MEEP reports a non-finite D_conductivity on {name}; nothing was stepped."
            ])
        # Float64 round-off around a genuine zero reads as a value of order 1e-17 of
        # either sign; the driver refuses a negative conductivity outright, and it is
        # right to. Clip only what is round-off — anything a resolvable distance below
        # zero is gain and must reach that refusal.
        floor = -_CONDUCTIVITY_AGREEMENT_TOLERANCE * max(float(numpy.abs(values).max()), 1.0)
        if numpy.any(values < floor):
            raise MeepSimulationNotLiftable([
                f"MEEP reports a NEGATIVE D_conductivity on {name} (worst "
                f"{float(values.min()):g}); a negative conductivity is gain rather than loss and "
                f"grows the field while every magnitude still looks reasonable. Nothing was "
                f"stepped."
            ])
        volumes[name] = numpy.clip(values, 0.0, None)
    return volumes


def _lift_material(driver, medium, skip_susceptibilities: bool = False,
                   skip_conductivity: bool = False) -> None:  # Susceptibilities, conductivity, chi2/chi3.
    """Transfer the dispersive and nonlinear parts of the one medium.

    Order matters only in that everything here must precede the first step, which
    the driver enforces itself. ``sigma`` and the chi's go across per component
    when the medium is anisotropic in them and as a scalar when it is not: a
    scalar costs no array at all and is exactly registration-free, so the uniform
    case stays byte-identical to a hand-built run.

    ``skip_susceptibilities`` is the sigma-structured path: the media differ in
    their E_susceptibilities, so no single medium's terms describe the cell and
    :func:`_lift_susceptibilities_structured` reads the per-point volumes off
    MEEP instead. ``skip_conductivity`` is its exact twin one property over: the
    media differ in ``D_conductivity_diag``, or a MaterialGrid's ``damping``
    varies it across a cell whose media do not differ at all, so
    :func:`_read_conductivity_volumes` reads that per point instead. The chi's
    are still shared (the gate refused otherwise) and still come from the
    representative medium here.

    The two skips are independent flags rather than one "structured" flag
    because the cases are: ``test_adjoint_solver::test_damping`` skips the
    conductivity and not the susceptibilities, and a Drude/Lorentz interface
    skips the susceptibilities and not the conductivity.
    """
    if not skip_susceptibilities:
        for term in medium.E_susceptibilities:
            driver.add_susceptibility(
                _driver_susceptibility(term), _scalar_or_components(_vector3(term.sigma_diag)))
    conductivity = () if skip_conductivity else _vector3(medium.D_conductivity_diag)
    if any(value != 0.0 for value in conductivity):
        # Scalar when the three agree — which keeps every existing run bit-identical,
        # because one array object then goes under all three D keys — and the
        # per-component mapping when they do not. MEEP reads D_conductivity_diag per D
        # component (get_cnd, meepgeom.cpp:1545-1559), so the mapping is its shape.
        driver.set_conductivity(
            conductivity[0] if len(set(conductivity)) == 1
            else {"Dx": conductivity[0], "Dy": conductivity[1], "Dz": conductivity[2]})
    # The MAGNETIC half, installed exactly the same way and NEVER skipped: the
    # structured route reads MEEP's D row and reaches no B component at all, so a
    # B_conductivity is always the medium's own declaration (the gate refuses media
    # that differ in it). MEEP reads B_conductivity_diag.x/.y/.z for Bx/By/Bz through
    # the same get_cnd switch that answers for the D side (meepgeom.cpp:1545-1559),
    # so the per-component mapping is its shape here too.
    b_conductivity = _vector3(medium.B_conductivity_diag)
    if any(value != 0.0 for value in b_conductivity):
        driver.set_b_conductivity(
            b_conductivity[0] if len(set(b_conductivity)) == 1
            else {"Bx": b_conductivity[0], "By": b_conductivity[1], "Bz": b_conductivity[2]})
    chi2 = _vector3(medium.E_chi2_diag)
    chi3 = _vector3(medium.E_chi3_diag)
    if any(value != 0.0 for value in chi2) or any(value != 0.0 for value in chi3):
        # MEEP keeps the pair together (structure.cpp allocates a zero partner for
        # whichever is set), and so does the driver, so both go across even when one
        # is identically zero.
        driver.set_chi2(_scalar_or_components(chi2))
        driver.set_chi3(_scalar_or_components(chi3))


def _scalar_or_components(diagonal: tuple[float, float, float]):  # A float, or the per-component mapping.
    if len(set(diagonal)) == 1:
        return diagonal[0]
    return {component: diagonal[axis] for axis, component in enumerate(E_COMPONENTS)}


def _media_differ_in_susceptibilities(materials) -> bool:  # The sigma-structured predicate.
    """True when the cell's media differ in E_susceptibilities (and possibly epsilon).

    Only called after the compatibility gate accepted the simulation, so any
    difference the lift has no reader for (the chi's, mu, a rotated
    conductivity) has already been refused; this just re-derives which of the
    accepted structured shapes the cell is. Dropping
    :data:`PER_POINT_SIGNATURE_INDICES` — permittivity, off-diagonal
    permittivity and ``D_conductivity_diag``, each recovered by its own read —
    leaves E_susceptibilities as the only entry that can still differ.
    """
    signatures = {_medium_signature(material) for material in materials}
    return len({
        _signature_without(signature, PER_POINT_SIGNATURE_INDICES)
        for signature in signatures
    }) > 1


SIGMA_ROUTE_READER = "reader"
SIGMA_ROUTE_LOOKUP = "declared-geometry lookup"
SIGMA_ROUTE_INVERSION = "chi1inv inversion"

# Objects whose containment the lookup will scan per Yee point before handing the
# cell to the inversion instead.
#
# THERE IS NO COST CROSSOVER, and saying otherwise would be the comfortable version.
# The lookup asks MEEP's own is_point_in_object once per object per point until one
# answers yes — a LINEAR scan where MEEP's internal geometry tree is logarithmic —
# and it still pays for N+1 frequency samples on top, so it is MORE expensive than
# the inversion at every object count, including zero. Measured per call on this
# machine: is_point_in_object 4.38 us against get_chi1inv's 0.43 us at w = 0 and
# 0.64 us at w > 0. Measured end to end on the same cells, lookup / inversion wall
# time for the whole sigma pass:
#
#     objects     1     2     4     8    16    32
#     1 pole   1.70  1.96  2.33  5.18  6.57 10.28
#     2 poles  1.38  1.86  2.53  4.23  9.72 14.63
#
# The lookup is the default anyway because it is the more ACCURATE route (3.3x
# closer to the exact reader on a 2-pole cell, 41x on a 6-pole one), and this
# constant is what bounds the price of that: at 8 objects the worst measured
# multiple is 5.2x, and past it a cell is better served by the inversion. Both
# routes are correct at every object count, which is why this is a dispatch
# threshold and not a refusal.
_LOOKUP_MAX_OBJECTS = 8


@dataclass(frozen=True)
class StructuredSigmaLift:
    """Which route produced the per-point sigma volumes, and what it measured.

    Attached to the driver as ``driver.sigma_lift`` and surfaced on
    :attr:`GpuRunResult.sigma_lift`, because "which route ran" is not cosmetic: the
    three differ in what they are exact to (the reader exactly, the lookup to
    MEEP's float32 storage, the inversion to 4-6 significant digits) and in what
    they refuse. A test that means to exercise one of them has no other way to know
    it did, and a caller comparing two lifts has no other way to know why they
    differ.
    """

    route: str  # One of the SIGMA_ROUTE_* constants.
    term_count: int
    verified_points: int  # Yee points the lookup checked against MEEP; 0 on the other routes.
    verification_frequencies: tuple  # The frequencies it checked at; empty elsewhere.
    worst_residual: float  # Worst relative disagreement over those points; nan elsewhere.
    tolerance: float
    fallback_reason: str = ""  # Why the preferred route did not run, when one did not.

    def __str__(self) -> str:
        if self.route != SIGMA_ROUTE_LOOKUP:
            trailer = f" (after the lookup declined: {self.fallback_reason})" if self.fallback_reason else ""
            return f"sigma via the {self.route}, {self.term_count} term(s){trailer}"
        return (
            f"sigma via the {self.route}, {self.term_count} term(s), verified at "
            f"{self.verified_points} Yee points and {len(self.verification_frequencies)} "
            f"frequencies, worst disagreement {self.worst_residual:.3e} "
            f"(limit {self.tolerance:g})"
        )


def _lift_susceptibilities_structured(mp, sim, driver, materials, progress_cb=None):
    """Install the per-point susceptibility sigma volumes, by whichever route MEEP allows.

    The sigma twin of :func:`_lift_epsilon_structured`, and the one place the three
    routes meet. All of them produce the same pair — an engine
    :class:`~.dispersion.Susceptibility` per term and a ``{component: (nterms, nx,
    ny, nz)}`` mapping of sigma volumes at each component's own Yee points — so
    everything downstream of here is single-path:

    * :func:`_sigma_volumes_through_the_reader` READS them, one number per point,
      where MEEP carries ``fields.get_susceptibility_sigma``. Exact, one call per
      point per term, and preferred wherever it exists.
    * :func:`_lookup_sigma_volumes` LOOKS them up in the declared geometry and
      CHECKS every point against MEEP's ``chi1inv`` at ``N + 1`` frequencies. MEEP
      point-samples sigma rather than averaging it (Subpixel_Smoothing.md:153), so
      the lookup is exact to MEEP's float32 storage — measured against the exact
      reader over every Yee point, **1.49e-08 … 3.28e-08** of each term's own
      largest sigma, which is 3.3x closer than the inversion on a 2-pole cell and
      41x closer on a 6-pole one. It is the default here because of that, and NOT
      because it is cheaper — it is the more expensive route at every object count
      (:data:`_LOOKUP_MAX_OBJECTS` carries the measured curve).
    * :func:`_recover_sigma_volumes` RECOVERS them from the public ``chi1inv``
      reader with no geometry at all. Agrees with the reader to 5.7e-08 (two-block
      Lorentz+Drude) and 1.4e-06 (6-pole ``meep.materials.Au``). It is the FALLBACK
      rather than a lesser route: it depends on nothing the lookup depends on — not the
      containment predicate, not object precedence, not periodic images — so the
      whole class of failures the lookup's per-point check exists to catch is
      invisible to it, and it brings its own independent cross-validation at a
      held-out frequency. Falling back is therefore evidence-preserving, not a
      silent degrade: the lookup's measured disagreement is carried into
      :class:`StructuredSigmaLift` and reported, and if the inversion refuses too,
      both refusals are raised together.

    The volumes go across as float32, matching MEEP's own single-precision storage
    and the driver's field dtype; the arithmetic on both compatibility routes is
    float64 throughout and only the installed copy is narrowed.

    Returns:
        The :class:`StructuredSigmaLift` record of the route actually taken.
    """
    record = None
    if _has_sigma_reader(mp):
        terms, volumes = _sigma_volumes_through_the_reader(mp, sim, driver, progress_cb)
        record = StructuredSigmaLift(
            route=SIGMA_ROUTE_READER, term_count=len(terms), verified_points=0,
            verification_frequencies=(), worst_residual=float("nan"),
            tolerance=float("nan"),
        )
    else:
        declined = _why_the_lookup_cannot_run(mp, sim, materials)
        if not declined:
            try:
                terms, volumes, record = _lookup_sigma_volumes(
                    mp, sim, driver, materials, progress_cb)
            except SigmaLookupRefused as refusal:
                declined = str(refusal)
        if declined:
            try:
                terms, volumes = _recover_sigma_volumes(mp, sim, driver, materials, progress_cb)
            except SigmaRecoveryRefused as refusal:
                raise SigmaRecoveryRefused(
                    f"neither compatibility route produced a per-point sigma for this cell. The "
                    f"declared-geometry lookup declined: {declined}\nThe chi1inv inversion then "
                    f"refused: {refusal}"
                ) from refusal
            record = StructuredSigmaLift(
                route=SIGMA_ROUTE_INVERSION, term_count=len(terms), verified_points=0,
                verification_frequencies=(), worst_residual=float("nan"),
                tolerance=float("nan"), fallback_reason=declined,
            )
    for index, term in enumerate(terms):
        driver.add_susceptibility(term, {
            name: numpy.ascontiguousarray(volumes[name][index], dtype=numpy.float32)
            for name in E_COMPONENTS
        })
    return record


def _why_the_lookup_cannot_run(mp, sim, materials) -> str:
    """One sentence naming what stops the declared-geometry lookup, or "" if nothing does.

    Checked BEFORE the lookup runs, so that a cell it structurally cannot serve
    goes straight to the inversion instead of paying for a whole grid of
    containment tests and then declining. Every case here is a property of the
    declaration, not of a measurement.

    Three of them, and only three:

    * **A material that is not a plain ``mp.Medium``.** The lookup reads
      ``E_susceptibilities`` off the object the geometry search returns; a
      callable or an epsilon array resolves to a different structure per point
      and has no declared sigma to take. (The compatibility gate refuses these
      outright, so this is a guard rather than a route.)
    * **A ``MaterialGrid``, asked about SEPARATELY.** :func:`_materials_of`
      expands a grid into its two endpoint media, so the loop above no longer
      SEES one and the guard above silently stopped covering it. It is the same
      answer for a stronger reason: MEEP concatenates the endpoints' terms and
      interpolates each sigma by the per-point weight (meepgeom.cpp:585-596), so
      the endpoints' DECLARED sigmas are not the cell's sigmas anywhere except
      at u = 0 and u = 1, and looking them up would install the wrong volume
      across the whole graded region. The grid's dispersion is refused at the
      gate today (:func:`_check_material_grids`), which makes this unreachable —
      it is here so that relaxing that refusal routes the cell to the inversion
      rather than to a lookup that would answer confidently and wrongly.
    * **A pole set the per-point check could not see through.** Two poles that are
      indistinguishable over the sampled band let an error in one sigma be
      cancelled by an error in the other, which would make the check accept a wrong
      lookup; :func:`~.sigma_lookup.plan_verification` refuses that pole set and
      the reason is carried here verbatim.
    * **More objects than the linear containment scan is worth**
      (:data:`_LOOKUP_MAX_OBJECTS`). A dispatch decision on measured cost, not a
      correctness one.
    """
    if _material_grids_of(mp, sim):
        return (
            "the cell contains a MaterialGrid, whose per-point sigma is its endpoints' terms "
            "interpolated by the local weight (meepgeom.cpp:585-596) rather than either "
            "endpoint's declared sigma, so there is nothing to look up per point."
        )
    for material in _materials_of(mp, sim):
        if not _is_medium(mp, material):
            return (
                f"the cell contains a material of type {type(material).__name__} rather than a "
                f"plain mp.Medium, which carries no declared E_susceptibilities to look up."
            )
    objects = tuple(getattr(sim, "geometry", ()) or ())
    if len(objects) > _LOOKUP_MAX_OBJECTS:
        return (
            f"the cell declares {len(objects)} geometry objects, past the {_LOOKUP_MAX_OBJECTS} "
            f"this route will linearly scan per Yee point. The lookup is never the cheaper of the "
            f"two — is_point_in_object is 4.38 us against get_chi1inv's 0.64 us — and its cost "
            f"grows with the object count where the inversion's does not: measured 1.4-2.0x the "
            f"inversion at one or two objects, 5.2x at eight, 14.6x at thirty-two. Both routes "
            f"are correct here; this one picks the cheaper."
        )
    try:
        plan_verification([_recovery_pole(term) for term in _unique_susceptibilities(materials)])
    except SigmaLookupRefused as refusal:
        return str(refusal)
    return ""


def _declared_material_at(mp, sim, point, objects, wrap):
    """The medium MEEP's geometry tree holds at ``point`` — libctl's own search, reproduced.

    ``material_of_unshifted_point_in_tree_inobject`` (libctl geom.c:1705) returns
    the material of the first object in the tree that contains the point, and
    ``default_material`` when none does. ``create_geom_box_tree0`` (geom.c:1579)
    stores the objects in REVERSE declaration order, which is what MEEP's manual
    means by "when objects overlap, later objects in the list take precedence" —
    so this walks ``sim.geometry`` backwards and stops at the first hit.

    ``wrap`` is the lattice-vector list for MEEP's ``ensure_periodicity``, which
    libctl applies as extra shifted copies of each object (``LOOP_PERIODIC`` in the
    same constructor) and MEEP enables only when a ``k_point`` is set
    (``simulation.py``: ``self.ensure_periodicity and not not self.k_point``). The
    unshifted position is tried for every object first and the shifted ones only
    if nothing claimed the point, which is not libctl's interleaving — libctl would
    let a wrapped copy of a later object beat an unwrapped earlier one. That
    difference is left rather than reproduced because it is exactly what the
    per-point check catches: a cell where it matters fails verification and goes to
    the inversion. Reproducing it would multiply the containment tests by the shift
    count (seven per object in 3-D) at EVERY point, where this ordering pays for
    them only where nothing claimed the point at all.

    The containment predicate is MEEP's exported ``is_point_in_object``, which is
    libctl's ``point_in_objectp`` — the same ``point_in_fixed_pobjectp`` that
    ``tree_search`` calls, WITHOUT the bounding-box gate that precedes it there. On
    a point lying exactly on a face the two can disagree at one ulp. That is not
    reasoned around either; it is checked, per point, against MEEP's permittivity.
    """
    vector = mp.Vector3(*point)
    for obj in reversed(objects):
        if mp.is_point_in_object(vector, obj):
            return obj.material
    for shift in wrap:
        shifted = mp.Vector3(point[0] - shift[0], point[1] - shift[1], point[2] - shift[2])
        for obj in reversed(objects):
            if mp.is_point_in_object(shifted, obj):
                return obj.material
    return sim.default_material


def _lattice_wrap_vectors(mp, sim) -> tuple:
    """The lattice shifts libctl replicates objects by, or empty when it does not.

    MEEP passes ``ensure_periodicity and not not k_point`` into
    ``set_materials_from_geometry`` (python/simulation.py:2075), so a cell with no
    ``k_point`` gets no periodic copies at all and this is empty. When it is not,
    one shift per axis in each direction is generated — enough for any object that
    fits inside the cell, which is every object for which the replication changes an
    answer at a Yee point.
    """
    if not (getattr(sim, "ensure_periodicity", True) and sim.k_point):
        return ()
    lengths = _vector3(sim.cell_size)
    shifts = []
    for axis, length in enumerate(lengths):
        if float(length) == 0.0:
            continue
        for sign in (1.0, -1.0):
            shift = [0.0, 0.0, 0.0]
            shift[axis] = sign * float(length)
            shifts.append(tuple(shift))
    return tuple(shifts)


def _yee_point_location(mp, gv, component, location) -> tuple:
    """Where MEEP itself put this Yee point, as the ``vector3`` its geometry search sees.

    THE SNAP, and why it is not the same number as this engine's own coordinate.
    MEEP fills each sigma array inside ``LOOP_OVER_VOL(gv, c, i)`` and passes
    ``IVEC_LOOP_LOC(gv, here)`` — the grid_volume's own location for the loop's
    ivec — to ``sigma_row`` (anisotropic_averaging.cpp:336-338). That location is
    ``ivec * dx/2`` computed in MEEP's arithmetic. This engine's coordinate for the
    same point is ``axis_origin + i*dx`` computed in its own, and
    :func:`_lattice_indices` already proves the two agree to within 1e-7 of a half
    cell — which is not the same as agreeing, and the difference is entirely
    load-bearing: containment is decided by ``fabs(proj) <= 0.5*size`` (libctl
    geom.c:305), so a point one ulp outside a face is in a different medium than a
    point exactly on it.

    Measured on a block whose faces sit at x = -1/3 and y = -1/3 with dx = 1/30 —
    coordinates no binary float represents — the two coordinates put **81 of 10800**
    Yee points in different media
    (``test_sigma_lookup.py::test_this_engines_own_yee_coordinate_differs_from_meeps_snap_at_an_unrepresentable_face``),
    and moving the lookup off the snapped coordinate by that one ulp fails the
    per-point check at **1.8e-01** where the snapped one passes at **1.9e-08**. So
    the snapped location is taken from MEEP and never recomputed here.

    The missing dimensions are zeroed rather than filled from this engine's grid,
    because that is what ``vec_to_vector3`` does (meepgeom.cpp:238-263): a 2-D run
    hands the geometry z = 0, a 1-D run x = y = 0, and a cylindrical run
    ``(r, 0, z)``.
    """
    where = gv.loc(component, gv.index(component, location))
    dim = int(gv.dim)
    if dim == 3:  # meep::Dcyl — vec_to_vector3 puts r in x and leaves phi at 0.
        coordinates = (where.r(), 0.0, where.z())
    elif dim == 0:  # meep::D1 — only z is a coordinate.
        coordinates = (0.0, 0.0, where.z())
    elif dim == 1:  # meep::D2 — z is forced to 0, not taken from the grid.
        coordinates = (where.x(), where.y(), 0.0)
    else:
        coordinates = (where.x(), where.y(), where.z())
    return coordinates


def _periodic_lattice_period(user_volume, sim, cylindrical: bool):
    """The integer lattice vector per axis where MEEP's boundary is Periodic, else None.

    ``fields::fields`` leaves every face Metallic and only ``use_bloch`` sets
    ``boundaries[High][d] = Periodic``, which ``simulation.py`` calls exactly when
    ``sim.k_point`` is truthy — so a run either wraps on every Cartesian direction
    or on none, the same all-or-nothing :func:`_lift_boundaries` reads.

    Taken off MEEP's ``user_volume`` — the one ``locate_point_in_user_volume``
    itself measures against, which is the FULL cell even when ``fields.gv`` is a
    mirror-folded half — and as ``big_corner - little_corner`` rather than from the
    declared cell size, because it must be MEEP's integer lattice vector exactly
    (``ilattice_vector``) and not a rounding of a float length; measured on a 4.0
    cell at resolution 10, both give 80.

    Cylindrical returns None on every axis: the r axis ends at the coordinate
    singularity and the PEC wall rather than wrapping, and a Dcyl structured
    dispersive cell is refused before it reaches here anyway (its chi1inv is not
    diagonal at nonzero frequency).
    """
    if cylindrical or not sim.k_point:
        return (None, None, None)
    low, high = user_volume.little_corner(), user_volume.big_corner()
    spans = (high.x() - low.x(), high.y() - low.y(), high.z() - low.z())
    return tuple(span if span > 0 else None for span in spans)


def _meep_reads_here(gv, user_volume, build_ivec, coordinates, periods, mirrors):
    """The lattice point MEEP will actually read, or None where it reads vacuum.

    ``fields::get_chi1inv`` and the sigma reader both do the same two things before
    they touch any data, and this is those two things:

    1. ``locate_point_in_user_volume`` (boundaries.cpp) wraps the point up by the
       lattice vector along any PERIODIC direction while it sits at or below the
       user volume's low corner (and down again if it is more than one lattice
       vector above it)::

           if (boundaries[High][d] == Periodic && there[d] <= little_corner[d])
             while (there[d] <= little_corner[d]) there += ilattice_vector(d);
           else if (... there[d] - ilattice[d] > little_corner[d])
             while (...) there -= ilattice_vector(d);

    2. the caller then searches ``chunks[i]->gv.owns(S.transform(iloc, sn))`` over
       every symmetry image, and answers vacuum if none owns it. The mirror image is
       a NEGATION of the lattice coordinate, because ``geometry_center`` is required
       to be the origin (the gate refuses anything else) so the fold plane is
       lattice coordinate 0. Measured: on a ``mp.Mirror(mp.Y)`` cell whose folded
       grid_volume starts at y = -2, the enumerated plane y = -2 is unowned and
       MEEP reads it at y = +2; without this step the lookup calls it vacuum and the
       per-point check scores **1.2e+00**.

    A mirror-folded axis and a periodic one never co-occur — ``FdtdDriver`` refuses
    Bloch together with a mirror by name — so the two steps cannot interact.

    Returns the ``ivec`` to read, which is ``None`` when MEEP owns no image of the
    point and answers vacuum-with-no-susceptibility there.
    """
    wrapped = list(coordinates)
    if any(period is not None for period in periods):
        # Read the corner only on a run that actually wraps: a Dcyl ivec answers
        # .r()/.z() and aborts on .x(), and _periodic_lattice_period returns all
        # None there, so this guard is what keeps a cylindrical lift out of it.
        low = user_volume.little_corner()
        corner = (low.x(), low.y(), low.z())
        for axis, period in enumerate(periods):
            if period is None:
                continue
            while wrapped[axis] <= corner[axis]:
                wrapped[axis] += period
            while wrapped[axis] - period > corner[axis]:
                wrapped[axis] -= period
    candidates = [tuple(wrapped)]
    for axis in mirrors:
        candidates += [tuple(-value if index == axis else value
                             for index, value in enumerate(candidate))
                       for candidate in candidates]
    for candidate in candidates:
        location = build_ivec(*candidate)
        if gv.owns(location):
            return location
    return None


def _lookup_sigma_volumes(mp, sim, driver, materials, progress_cb=None) -> tuple:
    """Look sigma up in the declared geometry, and CHECK every point against MEEP.

    THE PREMISE, measured rather than assumed. MEEP never subpixel-averages a
    susceptibility sigma — ``Subpixel_Smoothing.md:153`` says so and the code agrees
    (``anisotropic_averaging.cpp:336`` point-samples ``sigma_row`` at the Yee point
    while the permittivity beside it goes through the full averaging machinery).
    Measured directly: at a block edge placed OFF the grid (face at x = -0.037,
    dx = 0.05) sigma reads binary ``{0, 0.400000}`` at consecutive points while the
    permittivity at the same points reads 4.0000 / 3.2800 / 1.0000. So the sigma at
    a Yee point IS the declared sigma of the medium containing that point, and this
    function reproduces ``geom_epsilon::sigma_row`` (meepgeom.cpp:1655-1705):
    resolve the medium at the snapped location, take
    ``sigma_diag[component_index(c)]`` off whichever declared susceptibility
    ``susceptibility_equiv`` matches the chain entry, zero if none does.

    THE CHECK, which is what makes it a READ. Each of the pieces above is a place to
    be confidently wrong — the coordinate the containment test is evaluated at
    (:func:`_yee_point_location`), which object wins an overlap, whether libctl's
    bounding-box gate and MEEP's exported predicate agree at a face, whether the
    point is one MEEP owns at all. None of them is argued into correctness here.
    Instead every point's looked-up sigma is put back into MEEP's own dispersion
    formula together with the ``eps_inf`` READ from MEEP, and compared against
    ``get_chi1inv(c, d, iloc, w)`` at ``N + 1`` frequencies
    (:mod:`~.sigma_lookup`). Worst measured disagreement on the cells this was built
    against: **1.9e-08 … 2.4e-07**, against **1.8e-01** for the one-ulp coordinate
    defect. A cell that fails goes to the inversion, which shares none of these
    dependencies.

    THE OWNERSHIP GATE. ``fields::get_chi1inv`` and the sigma reader are the same
    two steps: wrap the point into the user volume along any PERIODIC direction
    (``locate_point_in_user_volume``, boundaries.cpp), then search the chunks for
    one that ``owns`` the result and, finding none, answer *vacuum with no
    susceptibility* (monitor.cpp:182-184 returns 1 on the diagonal; the reader's
    counterpart returns 0). :func:`_meep_reads_here` is those two steps, and both
    halves are load-bearing:

    * ``_lattice_indices`` enumerates the full closed lattice, so about 5 % of its
      points — the low plane of each axis — are outside the owned region. MEEP holds
      no sigma there whatever the geometry says. Measured on a ``meep.materials.Au``
      block flush with the low x face: **21 of 1281** points disagree with the exact
      reader without the gate, **0 of 1281** with it.
    * On a PERIODIC axis that same low plane is not vacuum at all — MEEP wraps it to
      the high face and reads the material there. Gating on ownership alone (which
      is what this function did first) zeroes the sigma of a dispersive object that
      reaches the cell edge of a ``k_point`` run, and the check catches it at
      **1.2e-01**. The wrap is transcribed rather than inferred, and it is
      self-checking: the point it produces has to be one MEEP owns, or the answer
      falls through to vacuum exactly as MEEP's does.

    The declared-geometry search then runs at the WRAPPED point, because that is the
    location MEEP passed to ``sigma_row`` when it filled the cell being read.

    Args:
        materials: every medium the cell can hold, ``_materials_of`` order.

    Returns:
        ``(terms, volumes, record)`` — the first two exactly as
        :func:`_sigma_volumes_through_the_reader` returns them, plus the
        :class:`StructuredSigmaLift` describing what was verified.

    Raises:
        SigmaLookupRefused: when any point's looked-up sigma fails to reproduce
            MEEP's permittivity. Caught by :func:`_lift_susceptibilities_structured`,
            which then runs the inversion and records why.
        MeepSimulationNotLiftable: when the premise itself does not hold — a
            chi1inv tensor that is not diagonal, where ``1/chi1inv[c][c]`` is not
            ``eps_cc`` and neither the check nor the inversion can be written.
    """
    declared = _unique_susceptibilities(materials)
    keys = [_susceptibility_equivalence_key(term) for term in declared]
    check = plan_verification([_recovery_pole(term) for term in declared])
    terms = tuple(_driver_susceptibility(term) for term in declared)
    objects = tuple(getattr(sim, "geometry", ()) or ())
    wrap = _lattice_wrap_vectors(mp, sim)

    fields = sim.fields
    gv = fields.gv
    grid = driver.grid
    get_chi1inv = fields.get_chi1inv
    lattice_per_unit = 2.0 * float(sim.resolution)
    cylindrical = bool(getattr(grid, "cylindrical", False))
    if cylindrical:
        # The Dcyl spelling, as in _lift_epsilon_structured: this engine's
        # (Ex, Ey, Ez) are MEEP's (Er, Ep, Ez), whose component_index is still
        # (0, 1, 2) — vec.hpp:459-466 — so the sigma row selection is unchanged.
        directions = {"Ex": mp.R, "Ey": mp.P, "Ez": mp.Z}
        components = {"Ex": mp.Er, "Ey": mp.Ep, "Ez": mp.Ez}
        ivec = lambda ix, jy, kz: mp.iveccyl(ix, kz)  # noqa: E731 — the one construction seam.
    else:
        directions = {"Ex": mp.X, "Ey": mp.Y, "Ez": mp.Z}
        components = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez}
        ivec = mp.ivec
    user_volume = fields.user_volume
    periods = _periodic_lattice_period(user_volume, sim, cylindrical)
    mirrors = tuple(axis for axis in range(3) if grid.is_mirrored(axis))

    volumes: dict[str, Any] = {}
    frequencies = [float(value) for value in check.frequencies]
    worst = 0.0
    verified = 0
    total = max(3 * grid.nx, 1)
    done = 0
    for axis, name in enumerate(E_COMPONENTS):
        component = components[name]
        own = directions[name]
        others = [entry for entry in directions.values() if entry != own]
        indices = _lattice_indices(gv, grid, name, component, lattice_per_unit)
        counts = tuple(len(entry) for entry in indices)
        sigma = numpy.zeros((len(declared),) + counts, dtype=numpy.float64)
        instantaneous = numpy.empty(counts, dtype=numpy.float64)
        measured = numpy.empty((len(frequencies),) + counts, dtype=numpy.complex128)
        by_medium: dict[Any, numpy.ndarray] = {}
        for i, ix in enumerate(indices[0]):
            for j, jy in enumerate(indices[1]):
                for k, kz in enumerate(indices[2]):
                    location = ivec(int(ix), int(jy), int(kz))
                    for entry_axis in others:
                        # At the first VERIFICATION frequency, not at 0, and the
                        # difference is not cosmetic: the stored off-diagonals can be
                        # zero while monitor.cpp's frequency branch still returns a
                        # non-diagonal tensor (measured on a Dcyl cell — 0.0 at
                        # frequency 0, 0.5j at frequency 0.5, because that branch
                        # rebuilds the tensor from the Cartesian Ex/Ey/Ez rows, which
                        # a cylindrical structure does not fill). Checking at 0 would
                        # let that through to a residual of nan.
                        entry = complex(get_chi1inv(component, entry_axis, location,
                                                    frequencies[0]))
                        if abs(entry) > _CHI1INV_OFFDIAG_TOLERANCE:
                            raise MeepSimulationNotLiftable([
                                f"MEEP's chi1inv tensor is not diagonal at lattice point "
                                f"({ix}, {jy}, {kz}) — row {name}, off-diagonal entry {entry!r} at "
                                f"frequency {frequencies[0]:g} — "
                                f"and this MEEP has no fields.get_susceptibility_sigma, so the "
                                f"susceptibility sigma has to be checked against 1/chi1inv[c][c], "
                                f"which is eps_cc only for a diagonal tensor (monitor.cpp inverts "
                                f"the whole 3x3). A curved or rotated interface breaks that "
                                f"premise for the lookup and for the chi1inv inversion alike. The "
                                f"permittivity itself lifts here — the off-diagonal rows are "
                                f"ingested — but the dispersion cannot. Set eps_averaging=False, "
                                f"keep the dispersive shapes axis-aligned, or build a MEEP "
                                f"carrying the sigma reader (MEEP_SIGMA_PATCH=1 via "
                                f"parity/meep_gpu/build_meep_133_macos.sh)."
                            ])
                    entry = complex(get_chi1inv(component, own, location, 0))
                    if entry == 0.0 or not numpy.isfinite(entry):
                        raise MeepSimulationNotLiftable([
                            f"MEEP reports chi1inv = {entry!r} for {name} at lattice point "
                            f"({ix}, {jy}, {kz}); the lookup inverts it to a permittivity and "
                            f"cannot."
                        ])
                    instantaneous[i, j, k] = (1.0 / entry).real
                    reading = location if gv.owns(location) else _meep_reads_here(
                        gv, user_volume, ivec, (int(ix), int(jy), int(kz)), periods, mirrors)
                    if reading is not None:
                        medium = _declared_material_at(
                            mp, sim, _yee_point_location(mp, gv, component, reading),
                            objects, wrap)
                        row = by_medium.get(id(medium))
                        if row is None:
                            row = sigma_of_medium(
                                medium, keys, axis, _susceptibility_equivalence_key)
                            by_medium[id(medium)] = row
                        sigma[:, i, j, k] = row
                    # No image of the point is owned: MEEP answers vacuum at every
                    # frequency, so the looked-up sigma stays at its zero.
                    for index, frequency in enumerate(frequencies):
                        value = complex(get_chi1inv(component, own, location, frequency))
                        if value == 0.0 or not numpy.isfinite(value):
                            raise MeepSimulationNotLiftable([
                                f"MEEP reports chi1inv = {value!r} for {name} at lattice point "
                                f"({ix}, {jy}, {kz}), frequency {frequency:g}; the check inverts "
                                f"it to a permittivity and cannot."
                            ])
                        measured[index, i, j, k] = 1.0 / value
            done += 1
            if progress_cb is not None:
                progress_cb(min(done, total), total)
        worst = max(worst, check.require(
            instantaneous, sigma, measured, where=name,
            locate=lambda flat, counts=counts, indices=indices, name=name: (
                "lattice point ({}, {}, {})".format(*(
                    indices[a][value] for a, value in enumerate(numpy.unravel_index(flat, counts))
                ))
            ),
        ))
        verified += int(numpy.prod(counts))
        volumes[name] = sigma
    return terms, volumes, StructuredSigmaLift(
        route=SIGMA_ROUTE_LOOKUP,
        term_count=len(terms),
        verified_points=verified,
        verification_frequencies=tuple(frequencies),
        worst_residual=worst,
        tolerance=VERIFICATION_TOLERANCE,
    )


def _sigma_volumes_through_the_reader(mp, sim, driver, progress_cb=None) -> tuple:
    """Read MEEP's per-point susceptibility sigma volumes exactly.

    Through the reader the sigma patch adds (``fields::get_susceptibility_sigma``
    — see the design notes (meep-sigma-reader-plan) and parity/meep_gpu/
    meep-sigma-reader.patch). For each entry ``n`` of MEEP's E_stuff chain:

    * the term's physics comes from ``get_susceptibility_params(E_stuff, n)`` —
      the in-memory twin of ``dump_params``, ``(kind_tag, id, omega_0, gamma,
      drude_flag)`` — not from matching declared python objects by position:
      meepgeom deduplicates chain entries over exactly these fields
      (``susceptibility_equiv``), so the payload identifies the term uniquely
      and the emergent chain order never becomes a dependency;
    * one volume per E component is sampled at that component's own Yee points
      through the same ``_lattice_indices`` registration pin the epsilon lift
      uses — valid because sigma shares chi1inv's registration (the diagonal
      entry sits at the component's own site, anisotropic_averaging.cpp), and
      MEEP never subpixel-averages sigma, so the values read back are exactly
      the declared per-medium constants;
    * the off-diagonal row entries are audited and refused when nonzero — the
      engine's polarization state is diagonal — rather than silently dropped.

    All-zero volumes are still returned and still installed: the driver's own
    ``driven()`` check keeps them inert, and installing every chain entry keeps
    the enumeration cross-checkable against MEEP's.

    Returns:
        ``(terms, volumes)`` — the engine terms in MEEP's own chain order, and
        ``{component: (nterms, nx, ny, nz)}`` sigma sampled at each component's
        own Yee points.
    """
    fields = sim.fields
    gv = fields.gv
    grid = driver.grid
    lattice_per_unit = 2.0 * float(sim.resolution)
    components = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez}
    directions = {"Ex": mp.X, "Ey": mp.Y, "Ez": mp.Z}
    count = fields.get_num_susceptibilities(mp.E_stuff)
    terms: list[Susceptibility] = []
    stacks: dict[str, list] = {name: [] for name in E_COMPONENTS}
    total = max(3 * count * grid.nx, 1)
    done = 0
    for n in range(count):
        row = list(fields.get_susceptibility_params(mp.E_stuff, n))
        # Lorentzian-family payload: {4, id, omega_0, gamma, no_omega_0_denominator}.
        # Anything else on the chain is a kind the gate should have refused.
        if len(row) != 5 or row[0] != 4.0:
            raise MeepSimulationNotLiftable([
                f"susceptibility chain entry {n} reports params {row}, which is not the "
                f"plain Lorentzian/Drude payload (kind tag 4, five values); the "
                f"compatibility gate should have refused this material."
            ])
        terms.append(Susceptibility(
            frequency=float(row[2]),
            gamma=float(row[3]),
            kind=DRUDE if row[4] else LORENTZIAN,
        ))
        for name in E_COMPONENTS:
            component = components[name]
            own = directions[name]
            others = [axis for axis in (mp.X, mp.Y, mp.Z) if axis != own]
            indices = _lattice_indices(gv, grid, name, component, lattice_per_unit)
            counts = tuple(len(axis) for axis in indices)
            values = numpy.empty(counts, dtype=numpy.float64)
            for i, ix in enumerate(indices[0]):
                for j, jy in enumerate(indices[1]):
                    for k, kz in enumerate(indices[2]):
                        location = mp.ivec(ix, jy, kz)
                        values[i, j, k] = fields.get_susceptibility_sigma(
                            component, own, location, n)
                        for axis in others:
                            entry = fields.get_susceptibility_sigma(component, axis, location, n)
                            if entry != 0.0:
                                raise MeepSimulationNotLiftable([
                                    f"susceptibility entry {n} carries a nonzero off-diagonal "
                                    f"sigma[{name}][{('x', 'y', 'z')[int(axis)]}] = {entry!r} at "
                                    f"lattice point ({ix}, {jy}, {kz}); the polarization state "
                                    f"here is diagonal (one sigma per E component). The gate "
                                    f"checks declared sigma_offdiag, so a nonzero read here "
                                    f"means the declaration and the rasterized structure "
                                    f"disagree — refusing rather than dropping the coupling."
                                ])
                done += 1
                if progress_cb is not None:
                    progress_cb(min(done, total), total)
            stacks[name].append(values)
    volumes = {
        name: (numpy.stack(entries) if entries
               else numpy.empty((0,) + tuple(driver.shape), dtype=numpy.float64))
        for name, entries in stacks.items()
    }
    return tuple(terms), volumes


def _recover_sigma_volumes(mp, sim, driver, materials, progress_cb=None) -> tuple:
    """Recover the same sigma volumes from ``get_chi1inv`` alone — the STOCK-MEEP route.

    MEEP evaluates ``eps(w) = eps_inf + SUM_n sigma_n*chi_n(w)`` at every point when
    ``get_chi1inv`` is asked for a nonzero frequency (monitor.cpp:263-352), with
    ``chi_n`` fixed by the pole and ``sigma_n`` the per-point unknown. Sampling that
    at several frequencies and solving the linear system therefore recovers sigma
    with no C++ change at all. :mod:`~.sigma_recovery` owns the numerics — the
    reparameterization that makes the basis conditionable, the sampling band, and the
    cross-validation at a held-out frequency that makes a wrong pole list a refusal
    rather than a plausible number.

    What this function owns is everything MEEP-side, and each piece is a trap:

    * **The unknowns are MEEP's chain entries, not the declared objects.** Two media
      may declare the same pole with different sigma — one chain entry, two sigma
      values, one unknown per point. :func:`_unique_susceptibilities` deduplicates the
      way ``susceptibility_equiv`` does (meepgeom.cpp:1633-1647) for exactly that
      reason.
    * **The registration is the epsilon lift's**, ``_lattice_indices`` and its
      ``iyee_shift`` parity pin, because these are the same ``chi1inv`` entries at the
      same Yee points — sampled at a nonzero frequency instead of at 0.
    * **``mp.ivec`` must be called with three arguments.** ``mp.ivec(a, b)`` resolves
      to the SWIG ``(ndim, val)`` overload, which yields a vector that reads vacuum
      everywhere and then segfaults (the design notes (meep-sigma-reader-plan) §7).
    * **``get_chi1inv(c, d, iloc, w)`` at nonzero ``w`` returns an entry of the
      INVERSE tensor** — monitor.cpp inverts, adds the susceptibilities, and inverts
      back — so ``1/get_chi1inv(c, own, ...)`` is ``eps_cc`` only where that tensor is
      DIAGONAL. A curved or rotated interface makes it anisotropic (measured 0.203 for
      a cylinder, 0.086 for a rotated block, against exactly 0 for every axis-aligned
      case), and the recovery premise fails there. Every off-diagonal entry is read at
      the first sampling frequency and a nonzero one refuses the lift. This is a
      RECOVERY-path refusal only: the reader path reads sigma directly and the epsilon
      lift ingests the off-diagonal rows, so both are unaffected.

    Raises:
        SigmaRecoveryRefused: propagated with its measured message intact from
            :func:`~.sigma_recovery.plan_recovery` (a singular basis) or
            :meth:`~.sigma_recovery.SigmaRecoveryPlan.recover` (a pole list that
            disagrees with the structure MEEP rasterized). Never swallowed: a
            recovery that cannot be cross-validated returns no number at all.

    Returns:
        The same ``(terms, volumes)`` pair :func:`_sigma_volumes_through_the_reader`
        returns, so the install step is shared.
    """
    declared = _unique_susceptibilities(materials)
    plan = plan_recovery([_recovery_pole(term) for term in declared])
    terms = tuple(_driver_susceptibility(term) for term in declared)
    # The held-out frequency rides along as the last sample: one pass over the grid,
    # and plan.recover splits it back off. It took no part in the fit.
    sampled = [float(value) for value in plan.frequencies] + [float(plan.holdout_frequency)]
    fields = sim.fields
    gv = fields.gv
    grid = driver.grid
    lattice_per_unit = 2.0 * float(sim.resolution)
    if bool(getattr(grid, "cylindrical", False)):
        # The Dcyl spelling, as in _lift_epsilon_structured: this engine's
        # (Ex, Ey, Ez) are MEEP's (Er, Ep, Ez), and a Dcyl ivec is (r, z) with no
        # phi slot at all.
        directions = {"Ex": mp.R, "Ey": mp.P, "Ez": mp.Z}
        components = {"Ex": mp.Er, "Ey": mp.Ep, "Ez": mp.Ez}
        ivec = lambda ix, jy, kz: mp.iveccyl(ix, kz)  # noqa: E731 — the one construction seam.
    else:
        directions = {"Ex": mp.X, "Ey": mp.Y, "Ez": mp.Z}
        components = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez}
        ivec = mp.ivec
    volumes: dict[str, Any] = {}
    total = max(3 * grid.nx, 1)
    done = 0
    for name in E_COMPONENTS:
        component = components[name]
        own = directions[name]
        others = [axis for axis in directions.values() if axis != own]
        indices = _lattice_indices(gv, grid, name, component, lattice_per_unit)
        counts = tuple(len(axis) for axis in indices)
        samples = numpy.empty((len(sampled),) + counts, dtype=numpy.complex128)
        for i, ix in enumerate(indices[0]):
            for j, jy in enumerate(indices[1]):
                for k, kz in enumerate(indices[2]):
                    location = ivec(int(ix), int(jy), int(kz))
                    for axis in others:
                        entry = complex(fields.get_chi1inv(component, axis, location, sampled[0]))
                        if abs(entry) > _CHI1INV_OFFDIAG_TOLERANCE:
                            raise MeepSimulationNotLiftable([
                                f"MEEP's chi1inv tensor is not diagonal at lattice point "
                                f"({ix}, {jy}, {kz}) — row {name}, off-diagonal entry {entry!r} at "
                                f"frequency {sampled[0]:g} — and this MEEP has no "
                                f"fields.get_susceptibility_sigma, so the susceptibility sigma has "
                                f"to be recovered from chi1inv. That recovery reads eps_cc as "
                                f"1/chi1inv[c][c], which holds only for a diagonal tensor "
                                f"(monitor.cpp inverts the whole 3x3), so a curved or rotated "
                                f"interface breaks its premise. The permittivity itself lifts here "
                                f"— the off-diagonal rows are ingested — but the dispersion cannot. "
                                f"Set eps_averaging=False, keep the dispersive shapes "
                                f"axis-aligned, or build a MEEP carrying the sigma reader "
                                f"(MEEP_SIGMA_PATCH=1 via "
                                f"parity/meep_gpu/build_meep_133_macos.sh)."
                            ])
                    for index, frequency in enumerate(sampled):
                        entry = complex(fields.get_chi1inv(component, own, location, frequency))
                        if entry == 0.0 or not numpy.isfinite(entry):
                            raise MeepSimulationNotLiftable([
                                f"MEEP reports chi1inv = {entry!r} for {name} at lattice point "
                                f"({ix}, {jy}, {kz}), frequency {frequency:g}; the recovery "
                                f"inverts it to a permittivity and cannot."
                            ])
                        samples[index, i, j, k] = 1.0 / entry
            done += 1
            if progress_cb is not None:
                progress_cb(min(done, total), total)
        volumes[name] = plan.recover(samples[:-1], samples[-1])
    _settle_recovered_sigma(volumes, terms, plan)
    return terms, volumes


# The largest GAIN a recovered sigma may carry, in permittivity units, before the
# lift refuses. Measured on the cells this path was built against, as
# max|sigma_negative| * w0**2 * max|basis(w)| over the sampled band — i.e. how much
# permittivity the negative value is actually worth at the pole's strongest sampled
# frequency, which is scale-free in a way a bound on sigma itself is not (Drude
# sigma runs to 4e+21 and Lorentz sigma to 0.4 in the same material):
# **9.2e-16** on the Lorentz+Drude two-block cell and **1.4e-15** on
# meep.materials.Au, against eps of order 1. This bound is nine orders above that
# floor and six below anything a real gain medium would carry.
_RECOVERED_GAIN_TOLERANCE = 1e-6


def _settle_recovered_sigma(volumes, terms, plan) -> None:
    """Refuse a recovered sigma that is negative beyond the solve's noise; zero the rest.

    The held-out cross-check tests the MODEL, at one frequency, against MEEP. It
    does not test the SIGN of each recovered coefficient, and a negative sigma is
    gain: :func:`~.dispersion.normalize_sigma` refuses one outright ("a negative
    sigma is gain and would grow the field while every magnitude still looked
    reasonable"), and :func:`_check_susceptibility` refuses a declared one, so a
    recovered one must not slip between them.

    Every point where MEEP holds sigma = 0 — vacuum, or a pole belonging to some
    other medium — recovers as a least-squares residue of either sign rather than
    as an exact zero, so a bound is unavoidable. The negatives are clamped to
    exactly the zero MEEP holds there, which is a repair of the solve's noise
    against a KNOWN value rather than a guess; anything larger is refused with the
    number, because at that size the pole decomposition, not the arithmetic, is
    what disagrees.
    """
    for index, term in enumerate(terms):
        pole = plan.poles[index]
        # Per unit of the solved c = sigma*w0**2, the most permittivity this pole
        # can carry anywhere in the sampled band.
        strength = float(numpy.max(numpy.abs(pole.basis(plan.frequencies))))
        worst = min(float(numpy.min(volumes[name][index])) for name in E_COMPONENTS)
        carried = abs(min(worst, 0.0)) * pole.scale * strength
        if carried > _RECOVERED_GAIN_TOLERANCE:
            raise SigmaRecoveryRefused(
                f"the recovered sigma for the {term.kind} term f0={term.frequency:g} reaches "
                f"{worst:.3e}, worth {carried:.3e} of permittivity at the pole's strongest "
                f"sampled frequency (limit {_RECOVERED_GAIN_TOLERANCE:g}). A negative sigma is "
                f"gain, not loss, and this engine would step it as a growing mode. The solve "
                f"(condition number {plan.condition_number:.2e} over {plan.sample_count} "
                f"frequencies) passed its held-out cross-check, so the model fits MEEP's "
                f"permittivity — it is the decomposition into these poles that does not. Use a "
                f"MEEP carrying fields.get_susceptibility_sigma, which reads each sigma directly."
            )
        for name in E_COMPONENTS:
            numpy.maximum(volumes[name][index], 0.0, out=volumes[name][index])


def _resolve_pml_faces(mp, sim, dimensions: int, folded) -> Optional[FaceThickness]:
    """Accumulate the boundary layers into one per-axis, per-side thickness in cells.

    Split out of :func:`_lift_pml` when the retired source/absorber pre-flight
    needed the answer BEFORE a driver existed; the split stays because the lesson
    does — reading the layers a second way is how that pre-flight and the lift
    came to disagree about MEEP's own ``gaussian-beam.py``, so any future
    pre-driver consumer must take THIS table rather than re-derive one. ``folded``
    is passed rather than read off a grid because a pre-driver caller derives it
    from ``sim.symmetries`` while the lift reads the grid it just built, and both
    mean "a mirror plane halves this axis".

    Returns None when no face absorbs anything, which is what makes ``_lift_pml`` a
    no-op and leaves a source unconstrained.

    A layer on an INVARIANT axis is dropped, because MEEP drops it:
    ``_create_boundary_region_from_boundary_layers`` builds the region over the
    grid_volume's own directions, so ``mp.PML(0.5, direction=mp.Z)`` on a 2-D cell
    reaches nothing. Measured rather than reasoned — the same 2-D script with and
    without that layer steps to **0.0e+00** relative L2 of itself in MEEP. It is
    dropped here rather than passed on because ``setup_pml`` refuses an absorber on an
    invariant axis, which is the right answer for someone WRITING one and the wrong
    one for a converter reproducing a run where MEEP ignored it.

    A layer on an axis holding exactly ONE CELL is dropped for the same reason and by
    the other clause of the same guard: ``boundary_region::apply`` installs a face
    only when ``user_volume.num_direction(d) > 1`` (structure.cpp:208-218). That axis
    is resolved — MEEP has the direction, it is simply one cell deep — so the
    invariant rule above does not reach it, and every Dcyl script written
    ``mp.Vector3(sr, 0, 0)`` has one. Left in, ``mp.PML(32)`` on ``ring-cyl.py``'s
    cell put a **640-cell absorber on a 1-cell z axis**, whose "interior" then ran
    from +31.975 to -31.975 and swallowed the source. Measured on that cell at its own
    resolution: MEEP's undirected layer and the same layer named ``direction=mp.R``
    alone step to **0.0e+00** of each other, and so do ``mp.R`` plus an EXPLICITLY
    named ``mp.Z`` layer — the z faces are not there on either spelling.
    :func:`_unit_cell_axes` names the set.

    MEEP's layers are additive over faces: ``mp.PML(1.0)`` covers all six,
    ``mp.PML(1.0, direction=mp.Z)`` covers two, and a list of both spellings is
    resolved face by face. The result is handed to ``setup_pml`` as
    ``{'x': (low, high), ...}``, which is the same table MEEP's
    ``boundary_region`` holds.

    Note MEEP's ``boundary_side`` enum is ``High = 0, Low = 1``. Reading it the
    intuitive way puts a one-sided absorber on the opposite face, which produces a
    perfectly stable run against a boundary the user did not ask to absorb.

    THE CELL COUNT IS NOT ROUNDED. ``thickness * resolution`` goes across as it comes
    out: ``mp.PML(0.5)`` at resolution 71 is 35.5 cells, which MEEP builds and
    :class:`~.pml.PML` now builds too. Rounding it to 35 or 36 — which this converter
    did, behind a refusal, until the layer could carry a fraction — lands 8.75e-04 and
    9.27e-04 from CPU MEEP where the exact thickness lands at 1.16e-06 (the
    ``pml_fractional_cells_res71`` parity case and its two controls). The
    product is also the exact number MEEP quantizes: it forms
    ``(int)(dx * (2 * a) + 0.5)`` from the raw thickness, and ``2 * fl(t*r)`` equals
    ``fl(t * (2*r))`` because scaling by a power of two is exact, so the snapped extent
    is bit-identical on both sides of the conversion.
    """
    invariant = _invariant_axes(dimensions)
    # Both clauses of MEEP's own guard, in one set: the direction it does not have,
    # and the direction it has exactly one cell of.
    unreachable = set(invariant) | set(_unit_cell_axes(sim, dimensions))
    cylindrical = _is_cylindrical(sim) or dimensions == MEEP_CYLINDRICAL
    faces = [[0, 0], [0, 0], [0, 0]]  # (low, high) per axis, the driver's own order.
    active = False
    for layer in getattr(sim, "boundary_layers", ()) or ():
        if isinstance(layer, mp.Absorber):
            # AN ABSORBER IS AN mp.PML BY INHERITANCE (python/simulation.py:308) and must
            # be skipped here explicitly. MEEP skips it in exactly this place —
            # `_create_boundary_region_from_boundary_layers` opens with
            # `if isinstance(layer, Absorber): continue` (python/simulation.py:4956) — so
            # the face keeps whatever wall it had and no sigma is graded onto it. Letting
            # it through would install a PML *as well as* the conductivity: doubly
            # absorbing, perfectly stable, and wrong by the whole of the PML.
            continue
        cells = float(layer.thickness) * float(sim.resolution)
        if half_cell_extent(cells) == 0:
            # Below a quarter of a cell MEEP's use_pml finds no point with x > 0 and
            # returns before allocating sigma (structure.cpp found_pml) — the layer
            # simply is not there. Dropped for the same reason an invariant-axis layer
            # is: passing it on would install an inert absorber, which grades nothing
            # but does switch the run into PML field storage, a mode difference MEEP
            # has no counterpart for.
            continue
        declared = int(getattr(layer, "direction", MEEP_ALL))
        if cylindrical:
            # mp.R -> axis 0 (high face only — the axis is not a boundary), mp.Z ->
            # axis 2; an undirected layer covers exactly those three faces. The
            # compatibility gate already refused every other spelling by name.
            directions = (0, 2) if declared == MEEP_ALL else ({MEEP_R: 0, 2: 2}[declared],)
        else:
            directions = (0, 1, 2) if declared == MEEP_ALL else (declared,)
        side = int(getattr(layer, "side", MEEP_ALL))
        sides = (MEEP_SIDE_LOW, MEEP_SIDE_HIGH) if side == MEEP_ALL else (side,)
        for direction in directions:
            if direction in unreachable:
                continue  # MEEP installs no face here; measured a no-op both ways.
            for one_side in sides:
                if cylindrical and direction == 0 and one_side == MEEP_SIDE_LOW:
                    continue  # The axis face; the gate refused an explicit request already.
                faces[direction][0 if one_side == MEEP_SIDE_LOW else 1] = cells
                active = active or cells > 0
    if not active:
        return None
    # A mirror-folded axis keeps only its HIGH face: MEEP's boundary_region is
    # declared over the full user volume, but the folded chunk stores the upper
    # half, where the low-half absorber is the mirror IMAGE of the high one —
    # the same layer, folded. The driver refuses a low-face absorber on a folded
    # axis outright (cell 0 is the mirror plane, a boundary condition and not a
    # place to grade sigma), so the drop here is what makes MEEP's ordinary
    # ``mp.PML(t)`` + ``mp.Mirror`` spelling — ring.py's, among nine corpus
    # scripts' — arrive as the run MEEP actually steps.
    for axis in range(3):
        if axis not in invariant and folded[axis]:
            faces[axis][0] = 0
    return tuple((faces[axis][0], faces[axis][1]) for axis in range(3))


def _lift_pml(mp, sim, driver, dimensions: int = 3) -> None:  # The resolved table -> setup_pml.
    """Install :func:`_resolve_pml_faces`'s table on the driver, invariant axes dropped."""
    faces = _resolve_pml_faces(
        mp, sim, dimensions, tuple(driver.grid.is_mirrored(axis) for axis in range(3)))
    if faces is None:
        return
    invariant = _invariant_axes(dimensions)
    driver.setup_pml({
        name: faces[axis]
        for axis, name in enumerate(AXIS_NAMES)
        if axis not in invariant
    })


def _flat_resolved_axes(source, dimensions: int) -> list[int]:
    """The resolved axes a source declares no extent on — the surface-normal candidates.

    ONE predicate, three callers: :func:`_source_normal_axis` (the lift's resolver),
    :func:`_check_eigenmode_source` and :func:`_check_gaussian_beam_source` (the two
    pre-flights). Kept as a function rather than recopied because the gate and the
    lift disagreeing about what counts as a surface is exactly the drift that left an
    ``mp.EigenModeSource`` reported ``supported`` and then raised mid-lift.

    An INVARIANT axis is not a candidate: a 2-D run has no z to be flat on, so the
    surface a 2-D source declares is a LINE in x/y, and the same declaration in 3-D is
    a plane. This is stage 1 of MEEP's own inference (``volume::normal_direction``,
    vec.cpp:227-256), and the axes it excludes are the ones MEEP's reduced loop does
    not visit.
    """
    invariant = set(_invariant_axes(dimensions))
    return [axis for axis, value in enumerate(_vector3(source.size))
            if axis not in invariant and float(value) == 0.0]


def _source_normal_axis(sim, source, dimensions: int) -> int:
    """The axis a surface source is flat on — the one its equivalent currents straddle.

    The raise here is a BACKSTOP, not the message a caller is meant to read: both
    synthesized-source kinds run :func:`_flat_resolved_axes` at the gate
    (:func:`_check_eigenmode_source`, :func:`_check_gaussian_beam_source`), which is
    what makes the pre-flight verdict and the lift one decision rather than two, and
    each of those names its own MEEP abort site.
    """
    flat = _flat_resolved_axes(source, dimensions)
    if len(flat) != 1:
        raise MeepSimulationNotLiftable([
            f"{type(source).__name__} at center {_vector3(source.center)} has {len(flat)} "
            f"zero-extent resolved axes, so the plane its equivalent currents straddle is "
            f"ambiguous; only a surface source is realized here."
        ])
    return flat[0]


def _eigenmode_current_sheets(mp, normal: int) -> tuple:
    """MEEP's four equivalent-current sheets for a mode plane with this normal.

    mpb.cpp:874-900 — with ``n`` the normal axis and ``np1``/``np2`` the two
    transverse axes, the Love-equivalence currents are the electric current
    K = nHat x H and the magnetic current N = -nHat x E, deposited as four
    ``add_volume_source`` calls: (driven component, mode component sampled, sign).
    """
    np1, np2 = (normal + 1) % 3, (normal + 2) % 3
    cE = (mp.Ex, mp.Ey, mp.Ez)
    cH = (mp.Hx, mp.Hy, mp.Hz)
    return (
        (cE[np2], cH[np1], +1.0),
        (cE[np1], cH[np2], -1.0),
        (cH[np2], cE[np1], -1.0),
        (cH[np1], cE[np2], +1.0),
    )


def _eigenmode_component_filter(mp, source, dimensions: int):
    """Which of the four sheets MEEP actually installs — sources.cpp:495-506.

    ``source.component`` defaults to ALL_COMPONENTS (== Centered), which keeps all
    four; a single component keeps only its own sheet, with the D/B spellings folded
    onto E/H first (mpb.cpp:873-874). In 2-D the Z-parity bits of ``eig_parity``
    drop the other polarization's sheets: ODD_Z keeps the TM components, EVEN_Z the
    TE (``is_tm``, fields.cpp:458-468 — TM is Ez/Dz/Hx/Hy/Bx/By).
    """
    fold = {mp.Dx: mp.Ex, mp.Dy: mp.Ey, mp.Dz: mp.Ez,
            mp.Bx: mp.Hx, mp.By: mp.Hy, mp.Bz: mp.Hz}
    c0 = fold.get(int(source.component), int(source.component))
    parity = int(getattr(source, "eig_parity", 0) or 0)
    has_tm = bool(parity & mp.ODD_Z)
    has_te = bool(parity & mp.EVEN_Z)
    tm = {mp.Ez, mp.Hx, mp.Hy}

    def keep(component) -> bool:
        if c0 != mp.ALL_COMPONENTS and component != c0:
            return False
        if dimensions == 2:
            if has_te and component in tm:
                return False
            if has_tm and component not in tm:
                return False
        return True

    return keep


def _check_eigenmode_source(mp, sim, source, label: str, reasons: list) -> None:
    """Pre-flight for the eigenmode variants the reconstruction does not express.

    Everything here is an attribute check — the mode solve itself is deferred to
    the lift, so ``gpu_compatibility`` stays cheap. Each refusal names the variant
    and the nearest expressible alternative.
    """
    if _is_cylindrical(sim):
        reasons.append(
            f"{label} is an EigenModeSource on a cylindrical run; MPB solves modes on a "
            f"Cartesian lattice and MEEP's own Dcyl eigenmode path is not the four-sheet "
            f"reconstruction this lift performs. Inject the mode profile with a plain "
            f"mp.Source(amp_func=...), or run on CPU MEEP."
        )
        return
    dimensions = _effective_dimensions(mp, sim)
    if dimensions == 1:
        reasons.append(
            f"{label} is an EigenModeSource in a 1-D run, which has no transverse "
            f"cross-section for MPB to solve a mode on; the 1-D 'mode' is a plane wave, "
            f"which a plain mp.Source expresses directly."
        )
    elif int(getattr(source, "direction", MEEP_AUTOMATIC)) not in (MEEP_X, MEEP_Y, MEEP_Z):
        # THE SHAPE, and only on the branch where the shape is what resolves the normal.
        # A DECLARED direction=mp.X/Y/Z gives MEEP the normal outright (mpb.cpp:875,
        # ``n = d == X ? 0 : ...``) and `_lift_eigenmode_source` takes the same route,
        # so nothing about the volume is read and nothing here may refuse it.
        #
        # Everything else was the last gate/lift gap in this register, and it had TWO
        # raise sites rather than one, both measured 2026-08-08:
        #   * `_source_normal_axis`'s own MeepSimulationNotLiftable, mid-lift — the
        #     shape reaches it whenever MEEP's resolution succeeded and this engine's
        #     did not, which is the special_kz case below.
        #   * MEEP's abort, EARLIER than that, because `_lift_eigenmode_source` calls
        #     `sim.init_sim()` and MEEP resolves the normal while adding the source.
        #     An mp.EigenModeSource with size=(0,0) in a 2-D cell under mp.Mirror(mp.Y)
        #     returned supported=True with ZERO reasons and then raised a bare
        #     `RuntimeError: meep: Could not determine normal direction for given
        #     grid_volume` — worse than the MeepSimulationNotLiftable the contract
        #     forbids, because it names no feature at all. The same RuntimeError comes
        #     out of CPU MEEP's own init_sim for a point AND for a 2x2 volume, at both
        #     a zero and a nonzero eig_kpoint.
        #
        # TWO SHAPES REACH PAST THAT ABORT, which is why the sentence below claims it
        # for the default path only and carries a second evidence clause:
        #   * an explicit direction=mp.NO_DIRECTION skips the Python-side resolution
        #     entirely and mpb.cpp:876-884 takes the FIRST zero-extent axis. Nothing
        #     good follows: measured on a point source with eig_kpoint=(1,0), MPB's own
        #     eigensolver hard-fails ("crazy number detected in trace", eigensolver.c:393)
        #     and takes the interpreter with it — so the abort clause is still the
        #     honest summary, and this path is refused with it.
        #   * dft.cpp:817-819 resolves a source with extent in BOTH x and y to Z on a
        #     special_kz 2-D run. That one MEEP really does step, so it gets its own
        #     clause: this is a NARROWING, not a transcription of a MEEP refusal.
        #     Refusing it is still required — the reconstruction resolves the four
        #     sheets on a resolved normal, and letting the gate pass it on would put
        #     the raise back inside the lift.
        flat = _flat_resolved_axes(source, dimensions)
        if len(flat) != 1:
            size = _vector3(source.size)
            beta = (float(getattr(sim.k_point, "z", 0.0))
                    if getattr(sim, "special_kz", False) and getattr(sim, "k_point", None)
                    else 0.0)
            if dimensions == 2 and beta and float(size[0]) > 0 and float(size[1]) > 0:
                evidence = (
                    "MEEP does resolve this one, and to an axis this engine has no grid "
                    "for: a special_kz 2-D cell gives a source with extent in BOTH x and "
                    "y a Z normal (dft.cpp:817-819; measured, fields.normal_direction "
                    "returned 2 == mp.Z and init_sim completed). That is a sheet normal "
                    "to the invariant axis, which the four-sheet reconstruction here "
                    "builds on a RESOLVED normal only."
                )
            else:
                evidence = (
                    "MEEP does not step it either: on the default "
                    "direction=mp.AUTOMATIC, EigenModeSource.add_source resolves the "
                    "normal through fields::normal_direction (source.py:645-648), which "
                    "aborts on anything that is not a surface — 'Could not determine "
                    "normal direction for given grid_volume' (dft.cpp:821), measured as "
                    "a RuntimeError out of init_sim for a point and for a volume alike."
                )
            reasons.append(
                f"{label} is an EigenModeSource of size {size}, which leaves {len(flat)} "
                f"flat resolved axes rather than one — it is not a surface, so the plane "
                f"its four equivalent-current sheets straddle (mpb.cpp:874-900) is "
                f"undefined. {evidence} Give the source a line (2-D) or plane (3-D) "
                f"extent, or name the axis with direction=mp.X/mp.Y/mp.Z."
            )
    if not isinstance(source.eig_band, int):
        reasons.append(
            f"{label} sets eig_band={source.eig_band!r} (a DiffractedPlanewave, not a band "
            f"index); the diffracted-planewave synthesis is a different construction from the "
            f"MPB band solve this lift re-runs, and is not reproduced. Keep this run on CPU MEEP."
        )
    # An oblique launch (direction=NO_DIRECTION) on a run MEEP steps with REAL fields
    # was refused here, because the obliquity IS a complex in-plane phase ramp
    # (mpb.cpp:255 multiplies by ``std::polar(1.0, TWOPI * Gk . p0)``) and the injection
    # layer would not project it. It no longer needs a gate: `sources._step_source_values`
    # now transcribes MEEP's own unconditional ``real(A)`` (step.cpp:307) instead of
    # refusing, and oblique-source.py's own cell — the corpus script that was blocked —
    # steps to 1.2e-06 against CPU MEEP on REAL storage, the same number its complex twin
    # (`eigenmode_oblique_waveguide`) reads, with the real array equal to Re(the complex
    # array) to exactly 0.0. The measurement table is in `_step_source_values`.
    if not source.eig_match_freq and isinstance(source.src, mp.CustomSource):
        reasons.append(
            f"{label} sets eig_match_freq=False with a CustomSource waveform; MEEP retunes "
            f"the carrier to the fixed-k mode's frequency (mpb.cpp:864), which a caller-"
            f"supplied waveform has no carrier to retune. Match the frequency, or use a "
            f"gaussian/continuous src."
        )
    if isinstance(source.src, mp.CustomSource) and \
            not float(getattr(source.src, "center_frequency", 0.0) or 0.0):
        reasons.append(
            f"{label} drives an EigenModeSource with a CustomSource whose center_frequency "
            f"is 0; the MPB solve runs at exactly that frequency (mpb.cpp:839), so the mode "
            f"it would return is meaningless. Set center_frequency to the waveform's carrier."
        )
    solve_frequency = float(getattr(source.src, "frequency", None)
                            or getattr(source.src, "center_frequency", 0.0) or 0.0)
    if solve_frequency < 0.0:
        # A NEGATIVE carrier is lifted for plain mp.Source (the conjugate carrier);
        # here the frequency is also the MPB solve target (mpb.cpp:839), and neither
        # MEEP's synthesis nor this re-run of it has been measured there — refusing
        # by name beats handing back an unpinned mode.
        reasons.append(
            f"{label} is an EigenModeSource whose waveform frequency is negative "
            f"({solve_frequency}); the MPB mode solve this lift re-runs is measured at "
            f"positive frequencies only. Drive the conjugate carrier with a plain "
            f"mp.Source, or keep the run on CPU MEEP."
        )
    if getattr(source, "amp_func_file", ""):
        reasons.append(
            f"{label} sets amp_func_file={source.amp_func_file!r}; this engine takes a Python "
            f"amp_func only (the package deliberately has no h5py dependency)."
        )
    if getattr(source, "amp_data", None) is not None:
        reasons.append(
            f"{label} sets amp_data; MEEP interpolates that array with its own scheme, which "
            f"this engine does not reproduce. Wrap your own interpolation in amp_func instead."
        )


def _eigenmode_special_kz_turn(mp, sim, mode_component) -> complex:  # MEEP's source-side phasefix.
    """The quarter turn ``special_kz_phasefix`` puts on one mode component, or 1.

    MEEP source: ``special_kz_phasefix`` (mpb.cpp:295-304), called from
    ``add_eigenmode_source`` at mpb.cpp:847 with ``phase_flip=true`` whenever
    ``is_real && beta != 0`` — Ez, Hx and Hy multiplied by -i and nothing else.

    It lives in ``add_eigenmode_source``, NOT in ``fields::get_eigenmode``, so a lift
    that calls the solver directly (which is the only way to reach the mode from
    Python) bypasses it and has to reapply it here. The reason it exists: a
    real-storage beta run implicitly stores i*(TM fields) (step_db.cpp:148-160), so
    the current that drives those three has to be turned to match. Complex storage
    carries no implicit i and takes no turn.

    Measured on a 2-D epsilon-12 waveguide at kz = 0.2, ``kz_2d="real/imag"``: without
    it the run is 1.45 (Ez) / 1.39 (Ex) / 1.47 (Hy) away from CPU MEEP — a complete,
    smooth, stable field with a quarter-turn phase error — against 6.4e-07 with it.
    """
    if not (getattr(sim, "special_kz", False)
            and getattr(getattr(sim, "fields", None), "is_real", False)):
        return 1.0
    return -1j if mode_component in (mp.Ez, mp.Hx, mp.Hy) else 1.0


def _eigenmode_sample_point(mp, centre, offsets, invariant):  # Where the mode is sampled.
    """The absolute point one equivalent-current site samples the solved mode at.

    A module-level function so the mutation battery and the `full_3d_sample_point`
    control can replace it; see :func:`_lift_eigenmode_source` for the measurement.
    """
    return mp.Vector3(*(0.0 if axis in invariant else centre[axis] + offsets[axis]
                        for axis in range(3)))


def _lift_eigenmode_source(mp, sim, source, dimensions):
    """One mp.EigenModeSource -> driver source specs, by re-running MEEP's own synthesis.

    MEEP's ``add_eigenmode_source`` (mpb.cpp:830-904) is an MPB mode solve followed by
    four plain ``add_volume_source`` calls whose amp_func samples the solved mode
    (``eigenmode_amplitude``). Both halves are reachable from Python — the solve as
    ``sim.get_eigenmode`` with the same arguments ``EigenModeSource.add_source`` passes
    (source.py:639-689), the profile as ``EigenmodeData.amplitude`` — and this engine's
    ``_build_source_points`` is the transcription of ``add_volume_source`` itself, so
    handing it the same four (component, ±amplitude, profile) requests reproduces
    MEEP's deposition table. Measured stepping against CPU MEEP's own eigenmode
    source: complex Ez 6.2e-07 over the full cell for a 2-D waveguide (peak ratio
    1.0000, forward/backward ratio identical to the printed digits), and 3.8e-07 /
    1.0e-06 (Ex/Ey) for a 3-D square waveguide.

    The coordinate conventions cancel exactly as they do inside MEEP:
    ``EigenmodeData.amplitude`` takes ABSOLUTE positions (the where-relative shift
    ``add_eigenmode_source`` applies to ``edata->center`` at mpb.cpp:853 is skipped by
    ``fields::get_eigenmode``, so the Python object keeps ``eig_vol.center()`` and
    ``eigenmode_amplitude`` subtracts it, mpb.cpp:200-206), while the driver hands
    ``amp_func`` offsets relative to the source centre — the convention MEEP uses for
    a user ``A(p)``, which is why a caller-supplied ``amp_func`` multiplies on top
    unchanged (mpb.cpp:253).

    Reading the DEPOSITED current back instead (``sim.get_source``) cannot work and
    is not a fallback: ``get_source_slice`` registers every raw src_vol amplitude at
    the CENTRED location of its cell (array_slice.cpp:195-249, ``iloc(Dielectric)``),
    so each component's own Yee-site geometry is destroyed before Python ever sees
    it. Re-injecting that read put the electric sheet half a cell off its true
    plane, where the deposition ladder re-split it — preserving the total current
    (peak ratio 1.0031) while breaking the one-way cancellation (4.5–5:1 forward/
    backward against MEEP's 6.15:1, rel L2 5.9e-01) — the measured dead end that
    motivated this construction.
    """
    if not sim._is_initialized:
        sim.init_sim()  # get_eigenmode needs the structure; nothing is stepped.
    reasons: list[str] = []
    _check_eigenmode_source(mp, sim, source, type(source).__name__, reasons)
    if reasons:
        raise MeepSimulationNotLiftable(reasons)

    # The SHEET NORMAL and the SOLVE DIRECTION are two different quantities, equal
    # for every axis-aligned launch and different for an oblique one.
    #
    #   * the normal is the axis the four Love-equivalence sheets straddle. For an
    #     axis direction it IS that axis (mpb.cpp:875, ``n = d == X ? 0 : ...``);
    #     for NO_DIRECTION MEEP takes it from the SOURCE VOLUME's zero-extent axis
    #     instead (mpb.cpp:876-884), which is what `_source_normal_axis` computes,
    #     and which is also what `EigenModeSource.add_source` computes for
    #     AUTOMATIC (source.py:642-645, ``fields::normal_direction``).
    #   * the solve direction is what `fields::get_eigenmode` branches on, and
    #     NO_DIRECTION selects a genuinely different MPB solve: the lattice axis is
    #     ROTATED onto k (mpb.cpp:409-421), ``eig_vol`` need not contain ``where``
    #     (mpb.cpp:371 — oblique-planewave.py's eig_vol is a one-pixel slab inside a
    #     10 um source line, so no containment pre-flight may be added here), and the
    #     Newton step rewrites all three reciprocal components against
    #     ``kdir = kcart/|kcart|`` rather than only k[d] (mpb.cpp:485-497, :605-612).
    #
    # Collapsing the two — passing the normal axis as the solve direction — is the
    # measured silent-wrongness case: on a 20-deg rotated waveguide the axis-aligned
    # solve CONVERGES, returns the same frequency to 8 digits, a group velocity
    # within 0.7% and a peak |Ez| within 0.55%, and is 3.7e-01 wrong (its Newton step
    # holds k.y at sin(20 deg) and moves only k.x, launching at 38.3 deg). See the
    # `axis_solve` control on the oblique parity cases.
    direction = int(getattr(source, "direction", mp.AUTOMATIC))
    axes = (mp.X, mp.Y, mp.Z)
    if 0 <= direction < len(axes):
        normal = direction
        solve_direction = axes[direction]
    else:
        normal = _source_normal_axis(sim, source, dimensions)
        solve_direction = mp.NO_DIRECTION if direction == mp.NO_DIRECTION else axes[normal]
    where = mp.Volume(source.center, source.size, dims=sim.dimensions,
                      is_cylindrical=sim.is_cylindrical)
    eig_vol = source.eig_vol
    if eig_vol is None:  # source.py:650-656, without mutating the caller's source.
        eig_vol = mp.Volume(source.eig_lattice_center, source.eig_lattice_size,
                            sim.dimensions, is_cylindrical=sim.is_cylindrical)
    # mpb.cpp:839 real(src.frequency()) — a CustomSource spells it center_frequency
    # (source.py:342-372), and _check_eigenmode_source refused it already when zero.
    frequency = getattr(source.src, "frequency", None)
    if frequency is None:
        frequency = getattr(source.src, "center_frequency", 0.0)
    edata = sim.get_eigenmode(
        float(frequency), solve_direction, where,
        int(source.eig_band), source.eig_kpoint, eig_vol=eig_vol,
        match_frequency=bool(source.eig_match_freq), parity=int(source.eig_parity),
        resolution=float(source.eig_resolution),
        eigensolver_tol=float(source.eig_tolerance),
    )

    base = _lift_source(mp, source, dimensions, skip_component=True)
    if not source.eig_match_freq:
        # mpb.cpp:864 — the carrier is retuned to the mode the fixed-k solve found;
        # the envelope keeps its width (gaussian_src_time::set_frequency sets freq only).
        base["frequency"] = float(edata.freq)
    user_amp = base.pop("amp_func", None)
    base_amplitude = complex(base.get("amplitude", 1.0))
    centre = base["center"]
    names = _component_names(mp)
    keep = _eigenmode_component_filter(mp, source, dimensions)

    # THE INVARIANT AXIS CARRIES NO OFFSET INTO THE MODE SAMPLE. MEEP's own D2 run
    # hands ``eigenmode_amplitude`` a D2 ``vec``, whose LOOP_OVER_DIRECTIONS never
    # visits Z (mpb.cpp:203-208), while ``EigenmodeData.amplitude`` builds a 3-D
    # ``mp.vec(x, y, z)`` from whatever Python passes — so a half-integer-z component's
    # own +/- dx/2 Yee offset, which this driver's amp_func legitimately reports, would
    # reach the mode as a real z displacement and pick up ``exp(i*2*pi*Gk[2]*pz)``.
    # Gk[2] carries beta (mpb.cpp:394), so the pair of half-cell offsets averages to
    # cos(pi*beta*dx) and the whole injected mode comes out that factor small.
    # MEASURED on a 2-D epsilon-12 waveguide at resolution 20, best-fit complex scale
    # of the lifted field against CPU MEEP's: 1 - s = 3.088e-05 at kz = 0.05,
    # 4.933e-04 at kz = 0.2 and 1.973e-03 at kz = 0.4 — exactly quadratic in beta and
    # exactly 1 - cos(pi*beta*dx), constant in time (t = 15 and t = 40 agree to five
    # digits), with the residual after removing the scale already at the 5.6e-07 floor.
    # Invisible at beta = 0, where Gk[2] is zero, which is why every earlier eigenmode
    # case reads its floor.
    invariant = _invariant_axes(dimensions)

    def profile(mode_component):  # eigenmode_amplitude x optional user A, per sheet.
        turn = _eigenmode_special_kz_turn(mp, sim, mode_component)

        def amp_func(dx, dy, dz):
            point = _eigenmode_sample_point(mp, centre, (dx, dy, dz), invariant)
            value = complex(edata.amplitude(point, mode_component))
            # Applied to the MODE, before any user A: MEEP fixes the fft_data itself
            # and `eigenmode_amplitude` then multiplies the caller's A on top
            # (mpb.cpp:253).
            value *= turn
            if user_amp is not None:
                value *= complex(user_amp(dx, dy, dz))
            return value
        return amp_func

    specs = []
    for driven, mode_component, sign in _eigenmode_current_sheets(mp, normal):
        if not keep(driven):
            continue
        spec = dict(base)
        spec["component"] = names[driven]
        spec["amplitude"] = sign * base_amplitude
        spec["amp_func"] = profile(mode_component)
        specs.append(spec)
    if not specs:
        raise MeepSimulationNotLiftable([
            f"{type(source).__name__} at center {_vector3(source.center)}: the component/"
            f"parity filter (component={source.component!r}, eig_parity="
            f"{source.eig_parity!r}) leaves none of the four equivalent-current sheets, so "
            f"the source would inject nothing and the lifted run would be silently "
            f"sourceless."
        ])
    return specs


def _beam_polarizations(mp, source) -> tuple[bool, bool]:
    """``has_tm`` / ``has_te`` for a beam — sources.cpp:539-540, off beam_E0 alone."""
    E0 = _vector3(source.beam_E0)
    return abs(complex(E0[2])) > 0, (abs(complex(E0[0])) > 0 or abs(complex(E0[1])) > 0)


def _beam_component_filter(mp, source, normal: int, dimensions: int):
    """Which of the four beam sheets survive ``add_volume_source_check``, sources.cpp:495-506.

    Deliberately NOT ``_eigenmode_component_filter``: they share the ``gv.has_field``
    and ``component_direction(c) == d`` guards, but the D2 parity split is driven by
    ``beam_E0`` here (sources.cpp:539-540) and by ``eig_parity`` there, and the beam
    ignores ``source.component`` entirely — ``add_volume_source`` passes ``c0 = c`` on
    every call (sources.cpp:544-556), so that guard is a no-op for a beam where for an
    eigenmode source it is the user's single-component request.
    """
    has_tm, has_te = _beam_polarizations(mp, source)
    tm = {mp.Ez, mp.Hx, mp.Hy}  # is_tm, fields.cpp:458-469
    directions = {mp.Ex: 0, mp.Ey: 1, mp.Ez: 2, mp.Hx: 0, mp.Hy: 1, mp.Hz: 2}

    def keep(component) -> bool:
        if directions[component] == normal:  # sources.cpp:499
            return False
        if dimensions == 2:  # sources.cpp:500-504
            if has_te and component in tm:
                return False
            if has_tm and component not in tm:
                return False
        return True

    return keep


def _check_gaussian_beam_source(mp, sim, source, label: str, reasons: list) -> None:
    """Pre-flight for the beam variants the reconstruction does not express."""
    if _is_cylindrical(sim):
        reasons.append(
            f"{label} is a Gaussian beam on a cylindrical run; MEEP's own gaussianbeam "
            f"constructor aborts on Dcyl (sources.cpp:562), so there is nothing to reproduce."
        )
        return
    dimensions = _effective_dimensions(mp, sim)
    if dimensions not in (2, 3):
        reasons.append(
            f"{label} is a Gaussian beam in a {dimensions}-D run; the beam is defined on a "
            f"line (2-D) or a plane (3-D) and MEEP's D1 grid_volume holds only Ex and Hy, "
            f"neither of which the four equivalent-current sheets drive."
        )
        return
    flat = _flat_resolved_axes(source, dimensions)
    if len(flat) != 1:
        reasons.append(
            f"{label} is a Gaussian beam whose volume has {len(flat)} zero-extent resolved "
            f"axes; MEEP takes the sheet normal from ``normal_direction(where)`` "
            f"(vec.cpp:227-247), which is defined only for a surface, and aborts otherwise. "
            f"Give the source a line (2-D) or plane (3-D) extent."
        )
        return
    beam_frequency = float(getattr(source.src, "frequency", None)
                           or getattr(source.src, "center_frequency", 0.0) or 0.0)
    if not beam_frequency:
        # source.py:759 takes the beam frequency from the waveform, and every scale in
        # gaussianbeam::get_fields is built from k = 2*pi*f*n: at f = 0 the Rayleigh
        # range collapses and the Eorig normalization (sources.cpp:735-739) is 0/0.
        # MEEP would build it and return NaN; refusing names the cause instead.
        reasons.append(
            f"{label} is a Gaussian beam whose waveform has frequency 0 (a CustomSource "
            f"with no center_frequency, most likely); the beam's wavenumber, Rayleigh "
            f"range and on-axis normalization are all built from it, and at 0 the "
            f"normalization is 0/0. Set the waveform's carrier frequency."
        )
        return
    if beam_frequency < 0.0:
        # Same boundary as the eigenmode refusal: a negative carrier is lifted for
        # plain mp.Source, but the analytic beam evaluator (k = 2*pi*f*n through
        # gaussianbeam::get_fields) is transcribed and measured at positive
        # frequencies only.
        reasons.append(
            f"{label} is a Gaussian beam whose waveform frequency is negative "
            f"({beam_frequency}); the beam synthesis this lift re-runs is measured at "
            f"positive frequencies only. Drive the conjugate carrier with a plain "
            f"mp.Source, or keep the run on CPU MEEP."
        )
        return
    has_tm, has_te = _beam_polarizations(mp, source)
    if not (has_tm or has_te):
        reasons.append(
            f"{label} is a Gaussian beam with beam_E0={_vector3(source.beam_E0)}; a zero "
            f"polarization vector leaves MEEP's own has_te/has_tm both false and the beam "
            f"field identically zero (sources.cpp:681, :709)."
        )
    elif dimensions == 2 and has_tm and has_te:
        # NOT a silent no-op: MEEP's own D2 parity checks drop ALL FOUR sheets here
        # (sources.cpp:501-504), so MEEP steps a sourceless run that still returns a
        # clean, smooth, entirely zero field. Refusing is the only honest answer.
        reasons.append(
            f"{label} is a 2-D Gaussian beam with beam_E0={_vector3(source.beam_E0)}, which "
            f"has BOTH an in-plane and an out-of-plane component. MEEP's 2-D parity checks "
            f"(sources.cpp:501-504) then reject every one of the four equivalent-current "
            f"sheets, so MEEP itself steps this run with no source at all and returns an "
            f"all-zero field. Split it into a TE beam (beam_E0 in x/y) and a TM beam "
            f"(beam_E0 along z)."
        )


def _lift_gaussian_beam_source(mp, sim, source, dimensions):
    """One mp.GaussianBeamSource -> driver source specs, by re-running MEEP's own synthesis.

    Same shape as :func:`_lift_eigenmode_source`, and for the same reason: MEEP's
    ``add_volume_source(src, where, beam)`` (sources.cpp:526-567) is an analytic beam
    evaluator plus four plain ``add_volume_source`` deposits, whose sheet table is
    BIT-FOR-BIT the eigenmode source's (``_eigenmode_current_sheets``, mpb.cpp:874-900
    vs sources.cpp:544-557 — the same Love-equivalence currents with the same signs).
    So the evaluator is transcribed once in :mod:`.gaussian_beam` (validated against
    MEEP's own compiled ``get_fields`` through ctypes to 9.2e-14 relative across all
    three of its numeric branches) and the deposition is this package's
    ``_build_source_points``, already the transcription of the same C++ routine.

    THREE THINGS MEEP DOES HERE ARE NOT WHAT A PHYSICIST WOULD WRITE, and each is a
    measured fact rather than a reading:

    * ``eps`` and ``mu`` come from the INITIALIZED structure at the source centre
      (``fields::get_eps``/``get_mu``, source.py:760-765), not from the default
      medium. They set k, the Rayleigh range and the whole envelope.
    * ``source.amplitude`` and ``source.amp_func`` are SILENTLY DROPPED — MEEP never
      passes them (source.py:751-773 builds the beam and calls
      ``add_volume_source(src, where, beam)`` with neither). Measured on CPU MEEP:
      ``amplitude=5.0`` and ``amp_func=lambda p: 3.0`` each leave the deposited
      current ratio at exactly 1.000000. Routing this through ``_lift_source``'s
      ``sign * base_amplitude`` — which is what ``_lift_eigenmode_source`` correctly
      does, because an EigenModeSource DOES honour ``amplitude`` — would silently
      rescale the run.
    * ``beam_x0`` and ``beam_kdir`` keep their z components in a 2-D cell
      (``py_v3_to_vec``, simulation.py:130), and the beam evaluator reads them. The
      reduction ``_lift_source`` correctly applies to a source's centre and size is
      wrong here: measured 5.0e-01 (kdir.z) and 4.5e-01 (x0.z) relative change in the
      deposited currents.

    ``GaussianBeam2DSource`` has its OWN lifter and must not be folded into this one
    by name: it is a different mechanism entirely — a Python Hankel-function synthesis
    (source.py:821-1044) deposited as ordinary ``mp.Source`` objects carrying
    ``amp_data`` — and shares only the word "beam". See
    :func:`_lift_gaussian_beam_2d_source`.
    """
    if not sim._is_initialized:
        sim.init_sim()  # get_eps/get_mu need the structure; nothing is stepped.
    reasons: list[str] = []
    _check_gaussian_beam_source(mp, sim, source, type(source).__name__, reasons)
    if reasons:
        raise MeepSimulationNotLiftable(reasons)

    cylindrical = sim.is_cylindrical
    centre_vec = mp.py_v3_to_vec(sim.dimensions, source.center, cylindrical)
    eps = float(sim.fields.get_eps(centre_vec).real)   # source.py:760-762
    mu = float(sim.fields.get_mu(centre_vec).real)     # source.py:763-765
    frequency = float(source.src.swigobj.frequency().real)  # source.py:759

    # ALL THREE components, at both dimensionalities. This is where the reduction
    # `_lift_source` applies to a source's centre and size would be wrong: py_v3_to_vec
    # builds a D2 vec from x and y and then sets its z slot EXPLICITLY
    # (simulation.py:126-130), and `gaussianbeam::get_fields` reads that slot. Measured
    # in a 2-D cell: dropping beam_kdir.z changes the deposited currents by 5.0e-01 and
    # dropping beam_x0.z by 4.5e-01 (8.39e+00 end to end, the `reduce_beam_vectors`
    # control). Which of the three the C++ then uses varies per operation — see the
    # module docstring of `.gaussian_beam`.
    beam_x0 = tuple(float(value) for value in _vector3(source.beam_x0))
    beam_kdir = tuple(float(value) for value in _vector3(source.beam_kdir))
    beam_E0 = tuple(complex(value) for value in _vector3(source.beam_E0))

    normal = _source_normal_axis(sim, source, dimensions)
    base = _lift_source(mp, source, dimensions, skip_component=True)
    base.pop("amp_func", None)  # MEEP never passes it — measured ratio exactly 1.000000
    names = _component_names(mp)
    keep = _beam_component_filter(mp, source, normal, dimensions)

    def profile(slot):  # EH[slot] at the offset, exactly gaussianbeam_ampfunc's switch.
        def amp_func(dx, dy, dz):
            return complex(beam_fields(
                ((dx, dy, dz),), beam_x0, beam_kdir, float(source.beam_w0), frequency,
                eps, mu, beam_E0, dimensions)[0, slot])
        return amp_func

    # sources.cpp:544-557 — the same four sheets, and the same slot of EH each samples.
    slots = {mp.Ex: 0, mp.Ey: 1, mp.Ez: 2, mp.Hx: 3, mp.Hy: 4, mp.Hz: 5}
    specs = []
    for driven, sampled, sign in _eigenmode_current_sheets(mp, normal):
        if not keep(driven):
            continue
        spec = dict(base)
        spec["component"] = names[driven]
        # `sign` ALONE, deliberately: MEEP passes +/-1.0 as the sheet amplitude and
        # never source.amplitude (sources.cpp:544-556 vs source.py:776-777), measured
        # at a deposited-current ratio of exactly 1.000000 for amplitude=5.0. The
        # eigenmode lift's `sign * base_amplitude` is correct THERE and wrong here.
        spec["amplitude"] = sign
        spec["amp_func"] = profile(slots[sampled])
        specs.append(spec)
    if not specs:
        raise MeepSimulationNotLiftable([
            f"{type(source).__name__} at center {_vector3(source.center)}: MEEP's own "
            f"component filter (sources.cpp:495-506) leaves none of the four equivalent-"
            f"current sheets, so the source would inject nothing and the lifted run would "
            f"be silently sourceless."
        ])
    return specs


def _check_gaussian_beam_2d_source(mp, sim, source, label: str, reasons: list) -> None:
    """Pre-flight for mp.GaussianBeam2DSource — everything decidable without sim.fields.

    The synthesis itself is MEEP's and stays MEEP's (see
    :func:`_lift_gaussian_beam_2d_source`), so what is checked here is only the shapes
    ``GaussianBeam2DSource.get_fields``/``add_source`` cannot express — each one a
    place where MEEP raises, aborts, or silently reads a coordinate it never wrote.
    """
    if _is_cylindrical(sim):
        # get_fields builds its complex point source out of `center.x() + beam_x0.x()`
        # and `center.y() + beam_x0.y()` (source.py:851-853) and its 2-D Green's
        # function out of the same two slots. A Dcyl vec carries (r, z) and nothing
        # ever writes x or y, so those reads are not the radial profile the caller drew.
        reasons.append(
            f"{label} is a GaussianBeam2DSource on a cylindrical run; its Green's function "
            f"is built from the x and y slots of the source centre (source.py:851-853), "
            f"which a Dcyl vec does not carry."
        )
        return
    dimensions = _effective_dimensions(mp, sim)
    if dimensions != 2:
        # green2d is a genuinely two-dimensional Hankel-function Green's function — it
        # sums only rhat[0] and rhat[1] (source.py:995-1000) — so there is no third
        # dimension for it to be right in.
        reasons.append(
            f"{label} is a GaussianBeam2DSource in a {dimensions}-D run; the class is the "
            f"exact 2-D beam and its Green's function (green2d, source.py:983-1044) is "
            f"two-dimensional by construction. Use mp.GaussianBeam3DSource in 3-D."
        )
        return
    size = _reduced_vector(getattr(source, "size", mp.Vector3()), dimensions)
    if size[0] and size[1]:
        # MEEP's own refusal, moved to the gate: get_fields raises a bare Exception
        # here (source.py:844-847) after this lift has already said supported.
        reasons.append(
            f"{label} is a GaussianBeam2DSource whose size is {size[:2]}; MEEP raises "
            f"'GaussianBeam2DSource should be a line source, not a plane' "
            f"(source.py:844-847). Set size.x or size.y to zero."
        )
        return
    if not size[0] and not size[1]:
        # add_source picks nHat from `if size.x(): ... elif size.y(): ...`
        # (source.py:1059-1063) with NO else, so a point source leaves nHat unbound and
        # MEEP dies with an UnboundLocalError rather than a diagnosis.
        reasons.append(
            f"{label} is a GaussianBeam2DSource with no extent; the sheet normal comes "
            f"from whichever of size.x/size.y is nonzero (source.py:1059-1063) and MEEP "
            f"raises UnboundLocalError when neither is. Give the source a line extent."
        )
        return
    kdir = _vector3(source.beam_kdir)
    if not float(kdir[0]) and not float(kdir[1]):
        # `kdir = kdir / np.abs(kdir)` (source.py:855-856) on the complex in-plane
        # direction, which is 0/0 for a beam aimed along z, and every subsequent field
        # is NaN. MEEP returns the NaNs.
        reasons.append(
            f"{label} is a GaussianBeam2DSource with beam_kdir={kdir}, which has no "
            f"in-plane component; MEEP normalizes the complex direction "
            f"kdir.x + 1j*kdir.y by its own magnitude (source.py:855-856) and returns "
            f"NaN for a purely out-of-plane beam."
        )
    w0 = source.beam_w0
    if w0 is None or not (float(w0) > 0.0):
        # z0 = k*w0**2/2 places the complex point source; at w0 = 0 it collapses onto
        # the beam axis and the Hankel functions are evaluated at their singularity.
        reasons.append(
            f"{label} is a GaussianBeam2DSource with beam_w0={w0!r}; the complex point "
            f"source sits at z0 = k*w0**2/2 (source.py:854) and a non-positive waist "
            f"puts it on the Hankel singularity."
        )
    frequency = float(getattr(source.src, "frequency", None)
                      or getattr(source.src, "center_frequency", 0.0) or 0.0)
    if not frequency:
        reasons.append(
            f"{label} is a GaussianBeam2DSource whose waveform has frequency 0; k = "
            f"2*pi*f*sqrt(eps*mu) (source.py:833) is then zero, the point-source offset "
            f"z0 collapses and the Hankel arguments are all zero. Set the waveform's "
            f"carrier frequency."
        )
    try:  # noqa: SIM105 — the import IS the check.
        import scipy.special  # noqa: F401
    except ImportError:
        # The ONE route in this package that needs scipy, and it needs MEEP's rather
        # than its own: get_fields imports hankel1e/hankel2e/jve at call time
        # (source.py:822). This package declares no scipy dependency, so name it here
        # instead of letting the ImportError arrive from inside the lift.
        reasons.append(
            f"{label} is a GaussianBeam2DSource and scipy is not importable; MEEP's own "
            f"synthesis imports scipy.special (hankel1e, hankel2e, jve) at "
            f"source.py:822. Install scipy, or use mp.GaussianBeam3DSource."
        )
    if getattr(sim, "symmetries", ()) or ():
        # THE FOLD SEAM, refused by name rather than half-built. `_check_source_folds`
        # and `_fold_drops_source` run against `sim.sources`, which holds the BEAM
        # object — one declaration with no component — while what is actually deposited
        # is the derived (Ez, Hx) pair, each with its own parity about the plane. The
        # fold rule would judge the wrong object. Nothing in the corpus combines the
        # two, so this is a named refusal rather than a build.
        reasons.append(
            f"{label} is a GaussianBeam2DSource in a run with a mirror symmetry; the "
            f"fold rules read sim.sources, where this source is still one undivided "
            f"declaration, while MEEP deposits the derived equivalent-current pair whose "
            f"parities about the plane differ per component. Drop the symmetry — the "
            f"answer is the same, the run is just larger."
        )


def _lift_gaussian_beam_2d_source(mp, sim, source, dimensions):
    """One mp.GaussianBeam2DSource -> driver source specs, by running MEEP's own synthesis.

    NOT the 3-D beam under another name, and not a transcription either. Where
    :func:`_lift_gaussian_beam_source` re-runs a compiled C++ evaluator this package
    has transcribed, this one has nothing to transcribe: ``GaussianBeam2DSource``
    performs its ENTIRE synthesis in Python before ``add_source`` is reached
    (``get_fields`` -> ``green2d``, source.py:821-1044, on scipy's Hankel functions)
    and then deposits ordinary ``mp.Source`` objects carrying ``amp_data``
    (``get_equiv_sources``, source.py:780-812). So the synthesis is CALLED rather than
    reproduced — which is the package's own rule, delegate setup to MEEP — and only
    the deposit is lifted, through the plain-source path that every other ``amp_data``
    source already takes.

    MEASURED on ``test_gaussianbeam.py``'s own configuration (14x14 cell, resolution
    25, dpml 2, rot_angle -40, beam_w0 0.8, beam_E0 along z): ``get_equiv_sources``
    returns exactly TWO plain sources — ``mp.Ez`` (an electric current) and ``mp.Hx``
    (a magnetic current) — both on the same 14x0 line at (0, -4, 0), each carrying an
    ``amp_data`` array of shape (700, 1, 1) complex128 with ``amplitude`` 1+0j. Both
    components are in ``driver.SOURCE_COMPONENTS``, and ``amp_data`` reaches the
    driver through :mod:`.amp_interpolation`, measured bit-identical to MEEP's own
    interpolator (difference exactly 0.0).

    THE PAIR IS THE PHYSICS, not a redundancy. Love's equivalence needs both the
    electric current ``K = nHat x H`` and the magnetic current ``N = -nHat x E``
    (source.py:786-794) for the sheet to radiate into one half-space only; deposit
    either alone and the beam launches in BOTH directions at half amplitude, on a run
    that completes and looks entirely healthy. That is the mutation this lift is
    pinned against.
    """
    if not sim._is_initialized:
        sim.init_sim()  # get_fields reads get_eps/get_mu at the centre; nothing is stepped.
    reasons: list[str] = []
    _check_gaussian_beam_2d_source(mp, sim, source, type(source).__name__, reasons)
    if reasons:
        raise MeepSimulationNotLiftable(reasons)

    equivalent = _meep_equivalent_beam_sources(mp, sim, source)
    if not equivalent:
        # `get_equiv_sources` keeps a component only when `np.sum(np.abs(...))` is
        # nonzero (source.py:809-811), so a polarization that produces no current at
        # all would leave the lifted run silently sourceless.
        raise MeepSimulationNotLiftable([
            f"{type(source).__name__} at center {_vector3(source.center)}: MEEP's own "
            f"equivalent-current synthesis (source.py:780-812) returns no nonzero "
            f"current for beam_E0={_vector3(source.beam_E0)}, so the lifted run would be "
            f"silently sourceless."
        ])
    return [_lift_source(mp, plain, dimensions) for plain in equivalent]


def _meep_equivalent_beam_sources(mp, sim, source):
    """MEEP's own (Ez, Hx, ...) deposit for one 2-D beam, obtained by calling MEEP.

    Transcribed only in the sense that ``add_source`` (source.py:1054-1068) is three
    statements and cannot be called directly — it would deposit into MEEP's own
    ``fields`` object. The three are reproduced verbatim; everything they call is
    MEEP's.
    """
    import numpy as np  # MEEP's own np.sign semantics on the direction cosine.

    fields = source.get_fields(sim)  # source.py:1056
    size = mp.py_v3_to_vec(sim.dimensions, source.size, sim.is_cylindrical)  # :1058
    if size.x():  # :1059-1063 — the branch order is MEEP's, and it is not symmetric.
        normal = mp.Vector3(0, 1) * np.sign(source.beam_kdir.y)
    else:
        normal = mp.Vector3(1, 0) * np.sign(source.beam_kdir.x)
    return list(mp.get_equiv_sources(fields, normal, source.src, source.center, source.size))


# Sources MEEP synthesizes at setup whose synthesis this engine re-runs: class name ->
# (pre-flight, lifter). Keyed by NAME because this module must not import meep. See the
# commentary above `_SPECIAL_SOURCE_NOTES` for what each entry costs and what it omits.
REALIZABLE_SOURCES: dict = {
    "EigenModeSource": (_check_eigenmode_source, _lift_eigenmode_source),
    "GaussianBeamSource": (_check_gaussian_beam_source, _lift_gaussian_beam_source),
    "GaussianBeam3DSource": (_check_gaussian_beam_source, _lift_gaussian_beam_source),
    "GaussianBeam2DSource": (_check_gaussian_beam_2d_source, _lift_gaussian_beam_2d_source),
}


def _lift_source(mp, source, dimensions: int = 3, skip_component: bool = False) -> dict:  # One mp.Source -> the driver's add_source mapping.
    """Translate one source, waveform included, into the driver's boundary-protocol dict.

    The waveform mapping is MEEP's own arithmetic rather than its attribute names:
    ``mp.GaussianSource`` stores ``width = max(width, 1/fwidth)`` whichever
    spelling the user chose, and the driver takes ``fwidth``, so the inverse is
    taken here and both spellings arrive identically. ``cutoff`` and
    ``start_time`` cross unchanged, which puts the peak at
    ``start_time + cutoff*width`` on both sides (MEEP's
    ``gaussian_src_time`` peak_time = 0.5*(start + end) with end =
    start + 2*width*cutoff).

    ``amp_func`` is rewrapped, not forwarded: MEEP calls it with one ``Vector3``
    and this engine calls it with three floats. Both pass the offset RELATIVE to
    the source centre (MEEP ``sources.cpp`` ``src_vol_chunkloop``;
    ``sources._build_source_points`` here), so only the calling convention changes.

    On a REDUCED run the centre and size are zeroed on every invariant axis, because
    MEEP drops those coordinates before it ever sees them: ``py_v3_to_vec`` builds a
    D2 ``vec`` from x and y alone, so a 2-D source declared at ``mp.Vector3(x, y, 7.3)``
    is the same source as one at ``mp.Vector3(x, y, 0)`` — measured 0.0e+00 apart in
    MEEP. This engine would place it identically either way (every position on an
    invariant axis names the same cell), but a nonzero SIZE would not be harmless:
    ``add_volume_source``'s delta scaling keys on the extent being exactly zero.
    """
    reduce_vector = lambda vector: _reduced_vector(vector, dimensions)  # noqa: E731
    names = _component_names(mp, dimensions == MEEP_CYLINDRICAL)
    src = source.src
    data: dict[str, Any] = {
        # A synthesized source has no single component — MEEP expands it into several
        # equivalent currents — so the caller supplies one per realized component and
        # this lookup is skipped rather than guessing at `source.component`.
        "component": None if skip_component else names[source.component],
        "center": reduce_vector(source.center),
        "size": reduce_vector(source.size),
        "amplitude": complex(source.amplitude),
        "is_integrated": bool(src.is_integrated),
    }
    profile = _source_profile_kind(source)
    if profile == "amp_func":
        meep_amp_func = source.amp_func

        def amp_func(x: float, y: float, z: float) -> complex:  # MEEP's one-Vector3 convention.
            return meep_amp_func(mp.Vector3(x, y, z))

        data["amp_func"] = amp_func
    elif profile == "amp_data":
        # The array route, which MEEP itself turns into an amp_func before it reaches
        # the deposition (sources.cpp:394-417). The extent handed to the interpolator is
        # the REDUCED size for the same reason the source's own is: `amp_file_func`
        # switches on `amp_func_vol->dim` (sources.cpp:349-364) and leaves every axis
        # the coordinate system does not carry at size 0, which is what a zero here
        # means to `amp_func_from_array`.
        data["amp_func"] = amp_func_from_array(source.amp_data, data["size"])
    if isinstance(src, mp.GaussianSource):
        data["source_type"] = "gaussian"
        data["frequency"] = float(src.frequency)
        data["fwidth"] = 1.0 / float(src.width)  # MEEP stores width; the driver takes its inverse.
        data["start_time"] = float(src.start_time)
        data["cutoff"] = float(src.cutoff)
        return data
    if isinstance(src, mp.ContinuousSource):
        data["source_type"] = "continuous"
        data["frequency"] = float(src.frequency)
        data["start_time"] = float(src.start_time)
        data["end_time"] = float(src.end_time)
        data["width"] = float(src.width)
        data["slowness"] = float(src.slowness)
        return data
    data["source_type"] = "custom"
    data["src_func"] = src.src_func
    data["start_time"] = float(src.start_time)
    data["end_time"] = float(src.end_time)
    return data


# --- the one-call path ------------------------------------------------------------


class StepFunctionNotHosted(RuntimeError):
    """A step function asked the hosting facade for something the driver cannot answer.

    Deliberately NOT an ``AttributeError``: MEEP's step functions reach their state
    by plain attribute access, and an ``AttributeError`` here would be swallowed by
    every ``getattr(sim, name, default)`` and ``hasattr`` in the calling code, which
    turns "this engine cannot host that" into "the simulation does not have one" and
    then into a plausible wrong answer.

    It is also why :class:`_SimulationFacade` never falls back to the ``mp.Simulation``
    it was built from. That object is initialized and NOT stepped — its fields are all
    zero — so a fallthrough would answer a step function with a silent zero field in
    the exact shape of a successful read.
    """


class _MeepFieldsFacade:
    """``sim.fields`` as MEEP's step functions read it, backed by the driver's clock.

    Three members, each the one MEEP's own wrappers touch:

    * ``dt`` — ``fields::dt``, MEEP's time step, read by ``at_every``'s
      ``-0.5 * sim.fields.dt`` half-step slack (``python/simulation.py:5119-5123``)
      and by ``Harminv._analyze_harminv`` as the fallback sample interval
      (``python/simulation.py:1184``).
    * ``t`` — ``fields::t``, the integer step count (``src/meep.hpp:1892-1893``
      forms both times from it), read by ``stop_when_dft_decayed``
      (``python/simulation.py:5385``) and by ``Simulation.timestep``.
    * ``last_source_time()`` — deferred to MEEP'S OWN ``fields`` object, the same
      authority :func:`run_on_gpu` already uses to resolve ``until_after_sources``
      (a Gaussian ends at ``start_time + 2*width*cutoff``, and MEEP owns that
      arithmetic). Both codes therefore agree about when a pulse has finished, which
      is the whole point of ``mp.after_sources``.

    Everything else raises :class:`StepFunctionNotHosted`.
    """

    def __init__(self, driver: FdtdDriver, sim):
        self._driver = driver
        self._sim = sim

    @property
    def dt(self) -> float:  # fields::dt.
        return float(self._driver.dt)

    @property
    def t(self) -> int:  # fields::t, the integer step count.
        return int(self._driver.step_count)

    def last_source_time(self) -> float:  # MEEP's own answer; see the class docstring.
        return float(self._sim.fields.last_source_time())

    def __getattr__(self, name: str):
        raise StepFunctionNotHosted(
            f"a step function asked for sim.fields.{name}, which this engine's run does not "
            f"host. Only dt, t and last_source_time() are carried across; the mp.Simulation "
            f"behind this run was initialized but never stepped, so answering from it would "
            f"return a zero field rather than a refusal. Run this step function on CPU MEEP, "
            f"or drive the driver directly."
        )


class _SimulationFacade:
    """``mp.Simulation`` as MEEP's step functions read it, backed by a stepped driver.

    ``run_on_gpu`` steps an :class:`~.driver.FdtdDriver`, not the ``mp.Simulation``.
    MEEP's step-function wrappers, though, are closures written against a
    ``Simulation``: ``mp.after_sources`` calls ``sim.fields.last_source_time()`` and
    ``sim.round_time()`` (``python/simulation.py:5045-5051``), ``mp.at_every`` calls
    ``sim.round_time()`` and ``sim.fields.dt`` (``:5119-5123``), and ``mp.Harminv``
    calls ``sim.meep_time()``, ``sim.get_field_point()`` (``:1149-1153``),
    ``sim.sources`` (``:1158``) and ``sim.run_index`` (``:5522``). Handing them the
    driver raises ``AttributeError: 'Fields' object has no attribute
    'last_source_time'``; handing them the un-stepped ``mp.Simulation`` would be
    worse, because it answers every one of those with a plausible zero.

    So this is the translation layer, one member per call MEEP's own wrappers make,
    and :class:`StepFunctionNotHosted` for everything else. ``__getattr__`` delegates
    to the DRIVER, which is what keeps a step function written against this engine —
    :class:`~.harminv.Harminv`, say — working unchanged through the same path.

    ROUND_TIME IS NOT MEEP_TIME. ``fields::round_time`` is ``float(t * dt)``,
    the simulation time rounded to SINGLE precision, while ``fields::time`` is the
    double (``src/meep.hpp:1892-1893``). MEEP rounds deliberately — the docstring on
    ``Simulation.meep_time`` (``python/simulation.py:2613-2624``) says it is so that
    a ``time < T`` termination test does not land on a different time step from
    machine to machine — and every threshold test in MEEP's step functions
    (``at_every``, ``after_sources``, ``after_time``) is written against the rounded
    number. Carrying the double through instead would move a sample by one step
    wherever ``t*dt`` and its float32 rounding fall on opposite sides of a
    threshold, so the rounding is reproduced here rather than smoothed over.
    """

    def __init__(self, driver: FdtdDriver, sim, mp):
        self._driver = driver
        self._sim = sim
        self._names = _component_names(mp, _is_cylindrical(sim))
        self.fields = _MeepFieldsFacade(driver, sim)
        # Simulation.run_index counts the run() calls one Simulation has made
        # (python/simulation.py:1529, incremented at :2859) and is used only to label
        # `display_run_data`'s printed rows, e.g. "harminv0:, ...". A facade hosts
        # exactly one run, so it is the first one.
        self.run_index = 0

    @property
    def sources(self):  # Harminv._check_freqs walks these to warn about its own band.
        return list(getattr(self._sim, "sources", None) or [])

    def meep_time(self) -> float:  # Simulation.meep_time -> fields::time, the double.
        return float(self._driver.meep_time())

    def round_time(self) -> float:  # Simulation.round_time -> fields::round_time, float32.
        return float(numpy.float32(self._driver.meep_time()))

    def timestep(self) -> int:  # Simulation.timestep -> fields::t.
        return int(self._driver.step_count)

    def get_field_point(self, c=None, pt=None) -> complex:
        """``Simulation.get_field_point`` (``python/simulation.py:2675-2681``).

        MEEP's own reader is ``fields::get_field_from_comp`` -> ``fields::get_field``
        (``src/monitor.cpp:127``), which :meth:`~.driver.FdtdDriver.get_field_point`
        transcribes stencil for stencil. A component may arrive either as MEEP's
        integer constant (from ``mp.Harminv(c=mp.Ez, ...)``) or already as this
        engine's name (from a step function written against the driver), and a point
        either as an ``mp.Vector3`` or as a plain triple, so both spellings pass.
        """
        name = c if isinstance(c, str) else self._names.get(c)
        if name is None:
            raise StepFunctionNotHosted(
                f"a step function asked for component {c!r} at {pt!r}; this run publishes "
                f"{sorted(set(self._names.values()))} and nothing else. A derived component "
                f"(energy density, Poynting flux) is not a stored field and has no reader here."
            )
        point = (pt.x, pt.y, pt.z) if hasattr(pt, "x") else tuple(pt)
        return complex(self._driver.get_field_point(name, tuple(float(v) for v in point)))

    def __getattr__(self, name: str):
        # Only reached for a name this class does not define; delegation to the driver
        # is what lets a step function written against FdtdDriver run here unchanged.
        try:
            return getattr(self._driver, name)
        except AttributeError:
            raise StepFunctionNotHosted(
                f"a step function asked for sim.{name}, which neither this facade nor "
                f"FdtdDriver carries (output_volume, filename_prefix, last_eps_filename, "
                f"set_materials, dft_objects and the HDF5 writers are the usual ones). The "
                f"mp.Simulation behind this run was initialized but never stepped, so "
                f"answering from it would return an un-stepped value rather than a refusal."
            ) from None


def _host_step_functions(step_functions: Sequence[Callable[..., Any]], facade: _SimulationFacade):
    """Rebind each step function onto ``facade`` instead of the driver.

    The arity dispatch is :func:`~.driver._resolve_step_functions`, MEEP's
    ``_eval_step_func`` (``python/simulation.py:5015-5024``), reused rather than
    transcribed a second time — two copies of that rule are two chances for a
    one-argument step function to be called on ``'finish'``. The result is a
    two-argument callable, which the driver's own resolution then passes through
    unchanged.
    """
    from .driver import _resolve_step_functions

    return tuple(
        (lambda driver, todo, call=resolved, host=facade: call(host, todo))
        for resolved in _resolve_step_functions(step_functions)
    )


def _load_declared_flux_data(driver, requests: Sequence[tuple[Any, Any]], *, minus: bool) -> None:
    """Apply ``run_on_gpu``'s ``flux_data`` / ``minus_flux_data`` to migrated monitors.

    Resolved by IDENTITY against this run's own monitors — the same route
    :meth:`GpuRunResult.monitor_for` takes, and for the same reason
    (:class:`MigratedMonitors`) — and refused by name when the object was never
    migrated: the alternative is a normalization run that silently subtracts nothing
    and reports the raw spectrum as a reflectance. This is the SECOND run of the
    two-run idiom, so it is precisely where a first run's collected monitor could have
    handed its address to the object being looked up here.
    """
    migrated = getattr(driver, "migrated_monitors", None) or MigratedMonitors()
    for entry in requests or ():
        try:
            meep_flux, data = entry
        except (TypeError, ValueError):
            raise TypeError(
                f"flux_data / minus_flux_data take [(meep_flux, data), ...] pairs; got "
                f"{entry!r}."
            ) from None
        monitor = migrated.resolve(meep_flux)
        if monitor is None:
            raise KeyError(
                f"{type(meep_flux).__name__} was not migrated onto this run, so there is "
                f"nothing to load its saved transform into. Pass the flux object belonging "
                f"to THIS simulation — the one returned by its own sim.add_flux — not the "
                f"normalization run's."
            )
        if not hasattr(monitor, "load_minus_dft_data"):
            raise TypeError(
                f"{type(meep_flux).__name__} migrated to a {type(monitor).__name__}, which "
                f"holds no flux transform to load into. Only sim.add_flux monitors take "
                f"flux data, exactly as in MEEP."
            )
        if minus:
            monitor.load_minus_dft_data(data)
        else:
            monitor.load_dft_data(data)


def run_on_gpu(
    sim,
    until: float | None = None,
    until_after_sources: float | None = None,
    *,
    prefer_gpu: bool = True,
    gpu_id: int = 0,
    prepare: Callable[[FdtdDriver], None] | None = None,
    flux_data: Sequence[tuple[Any, Any]] = (),
    minus_flux_data: Sequence[tuple[Any, Any]] = (),
    step_functions: Sequence[Callable[..., Any]] = (),
    progress_cb: Callable[[int, int], None] | None = None,
    lift_progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_interval: int = 100,
) -> GpuRunResult:
    """Lift ``sim`` and step it, MEEP's ``sim.run(...)`` with the loop on the GPU.

    The two stopping conditions are MEEP's, with MEEP's meanings::

        run_on_gpu(sim, until=200)                 # sim.run(until=200)
        run_on_gpu(sim, until_after_sources=50)    # sim.run(until_after_sources=50)

    ``until_after_sources`` is resolved exactly as ``Simulation._run_sources_until``
    resolves it — ``last_source_time() + value``, read from MEEP's own
    ``sim.fields`` so the two codes cannot disagree about when a pulse has
    finished. Note this is NOT the driver's keyword of the same name, which is
    ``stop_when_fields_decayed``'s trailing window; for that criterion, lift the
    simulation and call :meth:`~.driver.FdtdDriver.run` yourself.

    Args:
        sim: A constructed ``meep.Simulation``.
        until: Simulation time to run to, MEEP's ``run(until=...)``.
        until_after_sources: Simulation time to run past the last source's end.
        prefer_gpu: Run on this host's GPU: CUDA through CuPy, or Metal on an
            Apple GPU, where any configuration no released arm covers steps the
            host CPU and says so. Defaults to True, as :func:`lift_simulation`
            does: both MEEP entry points run on the GPU unless told otherwise.
            Pass False for the NumPy reference, which is the same driver on
            every host and never dispatches.
        gpu_id: CUDA device index; must be 0 on an Apple GPU.
        prepare: Called with the configured driver after the lift and before the
            first step — where DFT and flux monitors go, since MEEP's own are not
            carried across (:func:`gpu_compatibility` refuses a simulation that
            has any).
        flux_data: ``[(meep_flux, data), ...]`` — MEEP's
            ``sim.load_flux_data(flux, data)``, applied to the migrated monitors
            after the lift and before the first step. ``data`` is what an earlier
            run's :meth:`GpuRunResult.get_flux_data` returned. A SEQUENCE of pairs
            rather than a dict because MEEP's monitor objects are not hashable in
            every release.
        minus_flux_data: as ``flux_data``, but negated after loading — MEEP's
            ``sim.load_minus_flux_data``, and the two-run normalization idiom's
            second run. See :meth:`~.dft.FluxMonitor.load_minus_dft_data` for what
            the minus subtracts and why it is not a difference of powers.
        step_functions: MEEP's step-function list, forwarded to
            :meth:`~.driver.FdtdDriver.run`; both ``f(sim)`` and ``f(sim, todo)``
            are dispatched as MEEP dispatches them. The object handed to them is
            a :class:`_SimulationFacade` over the stepped driver — MEEP's own
            wrappers (``mp.after_sources``, ``mp.at_every``, ``mp.Harminv``) are
            closures written against ``sim.round_time()``,
            ``sim.fields.last_source_time()`` and ``sim.get_field_point()``, and
            the facade answers each of those from the driver's clock and fields.
            A step function written against :class:`~.driver.FdtdDriver` keeps
            working through the same path, because the facade delegates every
            other attribute to the driver. Anything neither carries — the HDF5
            writers, ``set_materials``, ``output_volume`` — raises
            :class:`StepFunctionNotHosted` rather than reaching the
            ``mp.Simulation``, whose fields are initialized and never stepped.
        progress_cb: ``(completed_steps, total_steps)``, polled every
            ``progress_interval`` steps. No progress is printed by this package.
        lift_progress_cb: ``(planes_done, planes_total)`` during the permittivity
            sampling that precedes the first step on a structured cell. A separate
            argument from ``progress_cb`` because it counts a different thing over
            a different total, and a caller rendering a bar wants to know which.
        cancel_check: Polled on the same interval; a truthy return raises
            :class:`~.driver.FdtdCancelled`.
        progress_interval: Steps between ``progress_cb`` / ``cancel_check`` polls.

    Returns:
        :class:`GpuRunResult` — the stepped driver plus the step count, the
        simulation time reached, and the wall time inside the step loop. The
        driver is still open; ``result.close()`` releases it.

    Raises:
        MeepSimulationNotLiftable: Nothing in ``sim`` was run; see
            :func:`gpu_compatibility`.
        ValueError: Neither or both stopping conditions given; or ``prefer_gpu``
            (the default here) with ``gpu_id`` other than 0 on an Apple GPU,
            raised before ``sim.init_sim()``.
        RuntimeError: ``prefer_gpu`` (the default here) on a host with neither a
            CUDA device nor an Apple GPU, raised before ``sim.init_sim()``; the
            message names ``prefer_gpu=False`` as the NumPy reference. A bare
            ``run_on_gpu(sim, until=...)`` therefore raises on a CPU-only host.
    """
    given = [
        name for name, value in (("until", until), ("until_after_sources", until_after_sources))
        if value is not None
    ]
    if len(given) != 1:
        raise ValueError(
            f"run_on_gpu takes exactly one of until or until_after_sources, got {given or 'none'} "
            f"(MEEP's own run() takes them the same way)."
        )
    mp = _import_meep()  # For the facade's component table; the lift imports its own.
    driver = lift_simulation(
        sim, prefer_gpu=prefer_gpu, gpu_id=gpu_id, progress_cb=lift_progress_cb
    )
    try:
        if prepare is not None:
            prepare(driver)
        # After `prepare` so a caller's own monitors are already on the driver, and
        # before the first step because MEEP loads into the accumulator the run then
        # adds to (python/simulation.py:3629-3649); loading afterwards would discard
        # the run instead of subtracting from it.
        _load_declared_flux_data(driver, flux_data, minus=False)
        _load_declared_flux_data(driver, minus_flux_data, minus=True)
        if until is not None:
            end_time = float(until)
        else:
            # MEEP simulation.py _run_sources_until: the stop is measured from the
            # last source's end, and MEEP's own fields object is the authority on
            # when that is (a Gaussian ends at start_time + 2*width*cutoff).
            end_time = float(sim.fields.last_source_time()) + float(until_after_sources)
            if not math.isfinite(end_time):
                raise ValueError(
                    f"until_after_sources={until_after_sources!r} resolves to t={end_time!r} "
                    f"because a source never stops emitting (a ContinuousSource with no end_time "
                    f"has MEEP's ~1e20 default). Give the source an end_time, or use until=."
                )
        before = driver.step_count
        wall_time = driver.run(
            until=end_time,
            step_functions=_host_step_functions(
                step_functions, _SimulationFacade(driver, sim, mp)),
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            progress_interval=progress_interval,
        )
    except Exception:
        driver.close()
        raise
    return GpuRunResult(
        driver=driver,
        steps=driver.step_count - before,
        meep_time=driver.meep_time(),
        wall_time_s=wall_time,
    )
