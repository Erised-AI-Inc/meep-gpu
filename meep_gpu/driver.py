"""
``FdtdDriver`` owns a single FDTD run end to end: it builds the Yee grid and field
storage, optionally wraps them in UPML, holds the current sources and the DFT /
flux monitors, and advances everything through MEEP's leapfrog update order.

It is the object host adapters or direct callers drive. Construct it with the cell
geometry, hand it a permittivity volume, add sources and monitors, then call
``run``. Inputs are plain Python / NumPy values in MEEP natural units (c = 1,
frequency = 1 / wavelength, lengths measured in the same unit the resolution
counts pixels of, PML thickness in grid cells). Outputs are host NumPy arrays
(``get_field``, ``get_epsilon``) plus the monitor objects created here, which
carry their own DFT and flux accumulations.

This module consolidates the original source bundle's two rival drivers — the
``GPUBackend`` facade and the self-contained ``Simulation`` orchestrator — into
one class: the post-injection D-side symmetry fill is called, epsilon shape
validation covers every axis, monitor default regions resolve at run start so
monitor/PML creation order no longer matters, ``until`` must be a finite time,
and unknown component / source-kind / axis strings raise instead of silently
selecting a default. Progress reporting and cancellation are callbacks, never
prints, so the host layer owns all user-visible output.

Any number of monitors may coexist; each keeps its own region, frequency list
and accumulators, and every one is updated at the same point of every step.
Monitor frequencies are given as a scalar, a sequence, or MEEP's broadband
``fcen`` / ``df`` / ``nfreq`` triple, which expands the way MEEP's own
``linspace`` does (including its ``nfreq = 1`` midpoint case).

Sources come in both field families, and ``step`` injects them in two slots, as
MEEP does (step.cpp:64-100): magnetic currents after ``step_B`` at ``t``,
electric currents after ``step_D`` at ``t + dt/2``. Each source reports its own
slot through ``field_type``, so the choice is never inferred from the call site.
Getting it wrong is a pure phase error — 1.6e-01 against CPU MEEP where the
correct slot gives 2.9e-07 — and no magnitude comparison can see it. Waveforms
are ``continuous``, ``gaussian`` or a caller-supplied ``custom`` ``src_func``,
each optionally ``is_integrated``; the combinations the dedicated electric
classes cannot express are composed onto ``sources.VolumeSource``.

Two run-level safety rules come from direct CPU-MEEP PML comparisons. A source
whose extent reaches into the absorber is ill-posed — its deviation from a boundary-free run grows from 6e-4
while merely touching the layer to 1e-1 three cells inside — so ``add_source``
and ``setup_pml`` reject the overlap from either creation order, face by face.
And the PML thickness is validated per axis against the faces that axis actually
carries, so a folded axis (whose lower face is a mirror plane, not an absorber)
is not refused a layer it can hold.

``setup_pml`` takes a scalar covering all six faces or a per-axis / per-side
table (``{"z": 8}``, ``{"z": (8, 0)}``, ``(0, 0, 8)``), MEEP's
``mp.PML(t, direction=..., side=...)``. Only the faces named absorb; the rest of
the domain keeps the boundary it would have had without a PML at all.

``run`` stops on a step count, a finite simulation time, or MEEP's field-decay
criterion (``stop_when_fields_decayed``); the decay form always carries an
explicit ``max_time`` ceiling and raises there rather than looping forever, and
a probe that never sees a nonzero field can never satisfy the criterion.

``symmetry`` is MEEP's ``mp.Mirror(direction, phase)``, spelled as
:class:`~.grid.Mirror` objects or as the bare axis name that is shorthand for the
even plane. Any of X, Y and Z may be folded, at either phase, in any combination:
each plane halves its axis and so halves the memory and the work. THE PLANE, NOT
THE AXIS, IS WHAT EVERY CONSUMER READS — the source-parity rule, the lattice fold,
the DFT and flux unfolding, the PML face table and the far-face guard all resolve
the parity through :func:`~.fields.mirror_parity` at the plane's declared phase,
so an odd plane is the same fold with every sign inverted rather than a second
code path. That is what makes a current along a plane's own normal — an ``Ex``
dipole on ``x = 0``, which an even X mirror makes odd and therefore cancels
against its own image — foldable by declaring ``Mirror('X', -1)``, and it is why
the source rule refuses exactly the components the DECLARED plane makes odd
rather than a fixed list.

Boundaries are chosen per axis through ``boundaries=``: periodic (the default) or
metallic, MEEP's perfect-electric-conductor wall, and mirror-folded where a plane
says so. An absorber decides none of it — it does not terminate an axis, it damps
what crosses it, which is MEEP's arrangement too (``fields::fields`` starts every
face ``Metallic``, ``fields::use_bloch`` overwrites the axis with ``Periodic``, and
``structure_chunk::use_pml`` only grades a conductivity underneath whichever of the
two is in force). A metallic axis zeroes the samples that sit ON its walls —
tangential E and normal B, MEEP's ``on_metal_boundary`` — and reads a zero ghost
past either face, because there is no field beyond a perfect conductor; the other
axes of the same run may stay periodic or Bloch-phased. ``k_point`` makes a
periodic axis
Bloch-periodic: the field one lattice vector up an axis is
``exp(i*2*pi*k_d*L_d)`` times the field at the original point, MEEP's convention
down to the units of ``k`` (2*pi/distance, ``mp.Simulation(k_point=...)``). It is
the gating capability for photonic crystals, gratings and band structure, and it
composes with a per-axis PML: ``k_point=(kx, ky, 0)`` with ``setup_pml({"z": n})``
is a grating, Bloch-periodic across the period and absorbing along the
propagation axis. Four combinations are refused rather than approximated,
because each would produce a plausible-looking field for a system the caller did
not ask for: real fields (which cannot carry the phase at all), a mirror symmetry
(whose plane forces the field to be even or odd, i.e. k = 0), a Bloch phase on
an axis whose own faces absorb (that axis terminates rather than wraps), and a
Bloch phase on a metallic axis (a wall is not a lattice vector — MEEP's
``use_bloch`` cannot express that pair either).

``set_chi2`` / ``set_chi3`` are ``mp.Medium(chi2=..., chi3=...)``: MEEP's
INSTANTANEOUS nonlinear susceptibilities, which need no auxiliary field and no
history and are unrelated to ``add_susceptibility`` beyond landing in the same
constitutive update. They are what makes second-harmonic generation, the Pockels
and Kerr effects, self-focusing, third-harmonic generation and four-wave mixing
expressible at all — a linear engine produces none of them, so their absence is
not an accuracy question. They compose with PML, dispersion and diagonal epsilon
because MEEP applies them inside the function that handles all three, and a run
whose chi2 and chi3 are identically zero steps the linear kernel byte for byte.
The one thing this engine adds over MEEP is a refusal: the Pade approximant MEEP
substitutes for the cubic solve has a pole, and a field strong enough to cross it
comes back finite, smooth and sign-flipped, so ``run`` measures the distance to it
(``nonlinear_margin``) and raises ``FdtdNonlinearityOutOfRange`` instead.

Array backend: every array operation goes through ``grid.xp`` — NumPy by
default (the reference path, testable on any machine), CuPy when a device is
requested. This module never imports cupy or meep. Fields are float32 or
complex64; epsilon, inverse epsilon, and PML coefficients are always float32.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import inspect
import math
import time as time_module
from typing import Any, Callable, Mapping, Optional, Sequence


from . import host_writes
from .backends import resolve_backend, to_numpy
from .dft import (
    DFTMonitor,
    EnergyMonitor,
    FluxMonitor,
    ForceMonitor,
    YeeRegionDFT,
    _dot_pair_reduction,
    meep_frequency_span,
    normalize_frequencies,
    yee_shifts,
)
# Spelled apart from the driver's own method of the same name, which is the public
# spelling and delegates here (src/integrate.cpp is one function with two outputs).
from .dft import integrate_field_function as dft_integrate_field_function
from .dispersion import (
    E_COMPONENTS,
    PolarizationState,
    Susceptibility,
    conductivity_factor,
)
from .deposit_repair import _deposit_index
from .fastpath import (SYNC_UPDATE_H_PASS, last_dispatch_report, plan_fast_path,
                       reference_record)
from .absorber import AbsorberLayer, absorber_conductivity, describe_layers
from .fields import Fields, mirror_parity
from .grid import (
    INVARIANT_AXES_BY_DIMENSIONS,
    MIRROR_AXES,
    Grid,
    Mirror,
    MirrorSpec,
    index_tolerance,
)
from .pml import (
    PML,
    absorbing_cells,
    half_cell_extent,
    normalize_pml_thickness,
)
from .smoothing import (
    DEFAULT_FILL_RULE,
    DEFAULT_SUPERSAMPLE,
    SmoothedEpsilon,
    epsilon_variation_along,
    smooth_inverse_epsilon,
)
from .sources import (
    FIELD_TYPE_B,
    ContinuousEnvelope,
    ContinuousSource,
    CustomEnvelope,
    ExtendedSource,
    GaussianEnvelope,
    GaussianPulsedSource,
    VolumeSource,
)
from .stepping import (
    B_COMPONENTS,
    D_COMPONENTS,
    NonlinearMargin,
    fill_folded_far_ghosts_B,
    fill_folded_far_ghosts_D,
    fill_symmetry_bc_B,
    fill_symmetry_bc_D,
    nonlinear_margin,
    step_B,
    step_D,
    update_E,
    update_H,
    update_P,
    zero_metal_B,
    zero_metal_D,
)

FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Dx", "Dy", "Dz", "Bx", "By", "Bz")
PRIMARY_COMPONENTS = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")  # The arrays the curl steps own.
# All twelve stored components are monitorable — MEEP's add_dft_fields takes any of
# them (absorbed_power_density.py monitors [Dz, Ez]) and the DFT machinery reads
# them uniformly: `yee_shifts` and `mirror_parity` carry the D/B rows, `Fields.
# get_component` serves the arrays, and the E/D-vs-H/B stagger is `_is_electric`'s.
MONITOR_COMPONENTS = FIELD_COMPONENTS
ELECTRIC_SOURCE_COMPONENTS = ("Ex", "Ey", "Ez")  # Electric currents, injected after step_D.
MAGNETIC_SOURCE_COMPONENTS = ("Hx", "Hy", "Hz")  # Magnetic currents, injected after step_B.
SOURCE_COMPONENTS = ELECTRIC_SOURCE_COMPONENTS + MAGNETIC_SOURCE_COMPONENTS
SOURCE_KINDS = ("continuous", "gaussian", "custom")
AXIS_NAMES = ("X", "Y", "Z")

# THE PERFECT-ELECTRIC-CONDUCTOR SENTINEL, and it is MEEP's number rather than this
# engine's convention. ``mp.metal`` and ``mp.perfect_electric_conductor`` are both
# ``Medium(epsilon=-inf)`` (meep/__init__.py:4438-4439 on 1.33.0; python/meep.i:1898 in
# the source tree), and MEEP's ``inf`` is the literal ``1.0e20``
# (meep/__init__.py:4231, python/meep.i:1691) — so the permittivity that reaches the
# C++ side is exactly -1e20 at construction. NOTHING clamps it; there is no -inf to
# clamp. On the C++ side such a medium stays ``material_data::MEDIUM`` and
# ``material_epsmu`` takes the ORDINARY ``sym_matrix_invert`` branch
# (meepgeom.cpp:785-796), giving chi1inv = -1e-20 on the diagonal. ``is_metal``
# (meepgeom.cpp:307-326) is consulted only to skip subpixel averaging
# (meepgeom.cpp:1093-1094, ``goto noavg``), a decision already baked into the chi1inv
# this engine reads back.
#
# So a PEC region is NOT a separate update here, exactly as it is not one in MEEP: the
# ordinary ``E = (D - sum P) * inv_eps`` with inv_eps = -1e-20 drives E to ~0 in those
# cells, and carrying MEEP's own number through unchanged is bit-comparable by
# construction. Every validator below that would otherwise refuse a non-positive
# permittivity relaxes for THIS VALUE BY NAME and keeps refusing every other one.
PEC_EPSILON_SENTINEL = -1e20  # mp.metal's epsilon_diag entry, verbatim.
PEC_CHI1INV_SENTINEL = -1e-20  # Its ordinary inverse, which is what MEEP stores.
# The float32 round trip is measured, not assumed: reading the sentinel back through
# the stored float32 volume and inverting gives -1.0000000317344784e+20, so both
# predicates are magnitude tests against a decade-wide margin, never equality.
_PEC_EPSILON_CEILING = 0.1 * PEC_EPSILON_SENTINEL  # -1e19
_PEC_CHI1INV_FLOOR = 10.0 * PEC_CHI1INV_SENTINEL  # -1e-19


def is_pec_epsilon(values: Any):  # Elementwise on a NumPy/CuPy array, or on a float.
    """True where a permittivity is MEEP's PEC sentinel rather than a material."""
    return values < _PEC_EPSILON_CEILING


def is_pec_chi1inv(values: Any):  # Elementwise; the inverse-side twin of the above.
    """True where an inverse permittivity is MEEP's PEC sentinel.

    Strictly negative, so neither ``0.0`` nor ``-0.0`` counts. That is deliberate:
    ``structure::get_chi1inv`` answers 0.0 for an unowned point (monitor.cpp:238) and
    ``material_epsmu``'s ``PERFECT_METAL`` branch stores -0.0 (meepgeom.cpp:799-801),
    and neither inverts to a finite permittivity. The first is handled by
    ``_unowned_wall_planes`` and the second is unreachable from the Python frontend,
    where both PEC spellings are ``MEDIUM`` with the -1e20 diagonal.
    """
    return (values < 0.0) & (abs(values) < abs(_PEC_CHI1INV_FLOOR))

def _reject_complex(values: Any, what: str, remedy: str):  # Refuse complex input where a real quantity is stored.
    """Guard a real-valued setter against a silent ``Im`` drop.

    NumPy and CuPy discard the imaginary part on a real cast with nothing louder
    than a ``ComplexWarning``, which leaves the run indistinguishable from one
    that was handed the lossless quantity in the first place — a confident wrong
    answer rather than a failure. Callers probe with ``xp.asarray(x)`` (no dtype)
    and pass the result here before coercing.
    """
    if values.dtype.kind == "c":
        raise ValueError(
            f"{what} must be real-valued, got {values.dtype}. Casting would silently discard the "
            f"imaginary part and simulate a different material than the one supplied. {remedy}"
        )


_COURANT_LIMIT_3D = 1.0 / math.sqrt(3.0)  # 3-D CFL stability bound on dt = courant / resolution.


def _courant_limit(dimensions: int) -> float:  # CFL bound for a run that resolves this many axes.
    """The Courant bound ``1/sqrt(n)`` for an ``n``-dimensional run.

    The bound counts the axes whose finite difference is not identically zero, which
    is what the von Neumann analysis of the Yee update sums over. A reduced run has
    fewer of them — MEEP 2-D differences x and y only — so its stable step is larger:
    1/sqrt(2) = 0.7071 in 2-D and 1 in 1-D, against 1/sqrt(3) = 0.5774 in 3-D.

    Keying it to ``dimensions`` rather than to "how many axes happen to hold one cell"
    is deliberate. A one-cell PERIODIC axis of a 3-D run also differences to zero and
    would tolerate the larger step, but saying so would make the stability limit a
    function of a cell count, and a caller who later widened that axis by one cell
    would have their previously accepted Courant number become unstable with nothing
    but the cell size changed.
    """
    return 1.0 / math.sqrt(dimensions)
# How much of the run's peak field the FAR face of a mirror-folded axis may carry before
# the fold stops standing for the full domain — for a NONLINEAR run, whose
# transverse sums still read a zero far ghost; the linear paths reflect through
# the stored second mirror and serve a live face at the fold floor. See
# _require_folded_far_face_is_quiet.
_FOLDED_FAR_FACE_LIMIT = 1e-2
_STEP_TOLERANCE = 1e-9  # Guards ceil() against float noise in the "run while t < until" comparison.
_CONDUCTIVE_PML_SOURCE_CLEARANCE_CELLS = 2.0  # Validated f_cond source margin at each absorbing face.
# How often run() samples the field energy while any susceptibility is live, and how far
# above the post-source peak it may climb before the run is declared divergent. A pole at
# |z| = 1.001 grows by 1e9 over 20 000 steps, so 20 catches it long before it can
# contaminate a DFT — and the guard exists because the failure mode of a dispersive run is
# a large, smooth, entirely finite wrong answer rather than a NaN.
_STABILITY_CHECK_INTERVAL = 20
_STABILITY_GROWTH_LIMIT = 1e3
# Largest max(|c2| + |c3|) a nonlinear run may reach. Derived, not tuned:
# 1 + 2*c2 + 3*c3 >= 1 - 3*(|c2| + |c3|), so below 1/3 the Pade denominator cannot
# have crossed zero. See _NonlinearityGuard.
_NONLINEAR_EXPANSION_LIMIT = 1.0 / 3.0
_COMMON_SOURCE_KEYS = frozenset(
    {"component", "center", "size", "amplitude", "amp_func", "source_type", "is_integrated"}
)
_CONTINUOUS_SOURCE_KEYS = frozenset({"frequency", "start_time", "end_time", "width", "slowness"})
_GAUSSIAN_SOURCE_KEYS = frozenset({"frequency", "fwidth", "start_time", "cutoff"})
# A custom waveform carries no carrier frequency of its own — 'frequency' is refused
# rather than accepted and ignored, which would silently detune nothing at all.
_CUSTOM_SOURCE_KEYS = frozenset({"src_func", "start_time", "end_time"})
_SOURCE_KEYS_FOR_KIND = {
    "continuous": _CONTINUOUS_SOURCE_KEYS,
    "gaussian": _GAUSSIAN_SOURCE_KEYS,
    "custom": _CUSTOM_SOURCE_KEYS,
}


class FdtdCancelled(RuntimeError):
    """Raised inside :meth:`FdtdDriver.run` when the caller's ``cancel_check`` asks to stop.

    Engine-local on purpose: host integrations map it onto their own
    cancellation contract without creating an application dependency.
    """


def _as_xyz(name: str, value: Sequence[float]) -> tuple[float, float, float]:  # Validate a 3-vector argument.
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must have three components (x, y, z), got {values!r}.")
    coords = []
    for axis, entry in zip(AXIS_NAMES, values):
        number = float(entry)
        if not math.isfinite(number):
            raise ValueError(f"{name} component {axis} must be finite, got {entry!r}.")
        coords.append(number)
    return (coords[0], coords[1], coords[2])


def _normalize_symmetry(symmetry: Sequence[MirrorSpec]) -> tuple[Mirror, ...]:  # Validate the requested mirror planes.
    """Turn the driver's ``symmetry`` argument into :class:`~.grid.Mirror` planes.

    THE DESCRIPTOR EVERY CONSUMER READS. A mirror plane is an axis AND a phase —
    MEEP's ``mp.Mirror(direction, phase)`` — so this returns the pair rather than
    the bare axis name it used to, and every consumer below (the source-parity
    rule, the lattice fold, the DFT unfolding, the PML face table) resolves the
    phase through :meth:`~.grid.Grid.mirror_phase` instead of assuming +1. A bare
    axis name is still accepted and still means MEEP's own default even plane, so
    ``symmetry=("X",)`` is ``symmetry=(Mirror("X", +1),)`` to the last bit.

    :class:`~.grid.Grid` normalizes the same way and is the one definition of a
    plane; this runs first so an unusable request fails on the constructor's own
    argument, before a backend is resolved or a single array is allocated.
    """
    planes: list[Mirror] = []
    for entry in tuple(symmetry):
        if isinstance(entry, Mirror):
            plane = entry
        else:
            axis = str(entry).strip().upper()
            if axis not in MIRROR_AXES:
                raise ValueError(
                    f"Unknown symmetry axis {entry!r}. A mirror plane is normal to one of "
                    f"{list(MIRROR_AXES)}: name the axis for an even plane (MEEP's default "
                    f"phase), or pass Mirror(axis, phase) to declare an odd one."
                )
            plane = Mirror(axis)
        clash = next((existing for existing in planes if existing.axis == plane.axis), None)
        if clash is not None:
            raise ValueError(
                f"Duplicate symmetry plane normal to {plane.axis}: {clash!r} and {plane!r} in "
                f"{tuple(symmetry)!r}. One axis carries one plane."
            )
        planes.append(plane)
    return tuple(planes)


def _resolve_step_functions(
    step_functions: Sequence[Callable[..., Any]],
) -> tuple[Callable[["FdtdDriver", str], None], ...]:
    """Normalize MEEP's two step-function spellings into one ``(sim, todo)`` callable each.

    MEEP's ``_eval_step_func`` (simulation.py:4997-5006) dispatches on the argument
    count: a two-argument function is called for every ``todo``, a one-argument one
    only for ``'step'``. Anything else raises there and raises here.

    MEEP reads ``func.__code__.co_argcount`` with a special case for its own callable
    objects; :func:`inspect.signature` covers both without the special case and also
    covers ``functools.partial``, which MEEP's own ``get_num_args`` gets wrong.
    """
    resolved: list[Callable[["FdtdDriver", str], None]] = []
    for function in tuple(step_functions):
        if not callable(function):
            raise ValueError(
                f"step_functions entries must be callable, got {type(function).__name__}."
            )
        target = function if inspect.isfunction(function) or inspect.ismethod(function) else (
            function if inspect.isbuiltin(function) else getattr(function, "__call__", function)
        )
        try:
            parameters = inspect.signature(target).parameters.values()
        except (TypeError, ValueError) as exc:  # A C callable with no introspectable signature.
            raise ValueError(
                f"Cannot tell how many arguments the step function {function!r} takes, so it "
                f"cannot be dispatched as MEEP dispatches one. Wrap it in a plain "
                f"def f(sim, todo) or def f(sim)."
            ) from exc
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is parameter.empty
        ]
        count = len(positional)
        if any(parameter.kind is parameter.VAR_POSITIONAL for parameter in parameters):
            count = 2  # *args takes whatever it is given; MEEP's protocol hands it two.
        if count == 2:
            resolved.append(function)
        elif count == 1:
            resolved.append(
                lambda driver, todo, call=function: call(driver) if todo == "step" else None
            )
        else:
            raise ValueError(
                f"Step function {getattr(function, '__name__', function)!r} takes {count} "
                f"required arguments; MEEP's protocol is f(sim) or f(sim, todo)."
            )
    return tuple(resolved)


def _face_index(axis: int, index: int) -> tuple[Any, ...]:  # Index tuple selecting one slab of a 3-D array.
    return (slice(None),) * axis + (index,)


def _periodic_axes(grid: Grid) -> tuple[bool, bool, bool]:  # Axes a monitor region may wrap across.
    """Per-axis wrap flags for a grid: the symmetry and the declared boundary, nothing else.

    This is the monitor-side half of ``stepping._boundary_kinds`` and must agree with
    it exactly, because a monitor measures the field the stepping produced. Every
    unmirrored axis wraps, absorber or not: MEEP's boundary condition comes from
    ``fields::use_bloch`` (boundaries.cpp), which sets ``boundaries[side][d] =
    Periodic`` on every direction of any ``mp.Simulation`` carrying a ``k_point`` and
    never consults the layer, and ``loop_in_chunks`` takes its lattice shift on exactly
    that flag. A PML is a material graded underneath it.

    Clipping an absorbing axis instead — on the reasoning that its wrapped plane sits
    inside the layer where the field has been attenuated away — is measurably not
    MEEP's answer, because "attenuated" is not "absent". A flux plane spanning the full
    cross section of a cell whose transverse axes carry a 4-cell layer read 1.4e-02
    (x absorbing) and 2.3e-02 (x and y absorbing) wrong per frequency against CPU MEEP
    while its wrap was dropped, against 1.4e-07 / 1.6e-07 with the wrap taken — the same
    band as the identical plane on a non-absorbing axis (2.0e-07).

    A mirrored axis does not wrap: MEEP reaches the far half through the symmetry
    transform, which this engine's flux does not unfold. That is read per axis from
    ``Grid.is_mirrored``, so a folded Z terminates exactly as a folded X does — the
    hard-coded ``False`` this used to carry in the Z slot would have handed a folded
    Z run the periodic wrap of an axis that has no lattice vector.

    A metallic axis does not wrap either, and MEEP agrees for the plainest possible
    reason: ``locate_point_in_user_volume`` translates only across a face it marked
    ``Periodic`` (boundaries.cpp), and ``use_bloch`` is the only thing that marks one.
    Both are now one call — ``Grid.axis_wraps`` is the single definition of "has a
    lattice vector" — so the monitor side and ``stepping._boundary_kinds`` cannot
    drift apart about a PEC wall the way the Z slot once drifted about a fold.
    """
    return tuple(grid.axis_wraps(axis) for axis in range(3))


def _positive_frequency(frequency: Any) -> float:  # Validate a bandwidth / probe frequency.
    value = float(frequency)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"frequency must be finite and positive, got {frequency!r}.")
    return value


def _finite_nonzero_frequency(frequency: Any) -> float:  # Validate a source carrier frequency.
    """A source's centre frequency: finite and nonzero, EITHER sign.

    MEEP stores the carrier frequency raw (``gaussian_src_time``, sources.cpp:72-96
    — no sign restriction) and its dipole rotates ``exp(-i*2*pi*f*(t-t0))``
    (sources.cpp:106), so a negative frequency is the conjugate carrier — the
    plus/minus-omega superposition ``dipole_in_vacuum_cyl_off_axis.py`` is built
    on. Zero divides the amplitude correction ``1/(-2*pi*i*f)`` (sources.cpp:104)
    and is refused; MEEP itself would step an infinite current without a word.
    ``fwidth`` stays :func:`_positive_frequency`: it is a bandwidth, and a
    negative one would put the envelope peak before the start time.
    """
    value = float(frequency)
    if not math.isfinite(value) or value == 0.0:
        raise ValueError(
            f"source frequency must be finite and nonzero (negative selects the "
            f"conjugate carrier), got {frequency!r}."
        )
    return value


def _decimation_request(value: Any) -> int:
    """Validate MEEP's high-level monitor decimation request.

    Zero means automatic selection; positive integers are explicit. The low-level
    monitor classes accept positive values only because they have no source list
    from which to resolve zero.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"decimation_factor must be zero (automatic) or a positive whole number, got {value!r}."
        )
    try:
        factor = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"decimation_factor must be zero (automatic) or a positive whole number, got {value!r}."
        ) from None
    if factor != value or factor < 0:
        raise ValueError(
            f"decimation_factor must be zero (automatic) or a positive whole number, got {value!r}."
        )
    return factor


def _no_withdraw(fields: Any) -> None:  # Sources without an integrated path leave nothing standing in D/B.
    return None


def _envelope_last_time(envelope: Any) -> float:  # When a VolumeSource's waveform stops emitting.
    """Latest time a source envelope still contributes, MEEP's ``src_time::last_time``.

    Mirrors :meth:`FdtdDriver._last_source_time` for the envelope-carrying sources.
    An envelope this does not recognise raises rather than being assumed finished:
    the decay stop gates on this number, and an envelope wrongly reported as done
    lets the criterion fire while the source is still driving the field.
    """
    if isinstance(envelope, GaussianEnvelope):
        return float(envelope.peak_time + envelope.cutoff * envelope.width)
    if isinstance(envelope, (ContinuousEnvelope, CustomEnvelope)):
        return float(envelope.end_time)
    raise ValueError(
        f"Unknown source envelope {type(envelope).__name__}; cannot tell when it stops emitting."
    )


def _normalize_frequencies(
    frequencies: Any = None,
    frequency: Any = None,
    fcen: Any = None,
    df: Any = None,
    nfreq: Any = None,
) -> tuple[float, ...]:  # Resolve a monitor's frequency tuple from the accepted spellings.
    """Return the monitor frequency tuple for exactly one of the three spellings.

    Accepted, one at a time: ``frequencies`` as a scalar or a sequence, the
    equivalent scalar-or-sequence ``frequency``, or MEEP's ``fcen`` / ``df`` /
    ``nfreq`` triple. Mixing them raises rather than picking a winner, since a monitor
    that quietly ignored one of two given frequency specifications would produce a
    plausible spectrum at the wrong frequencies.

    What each spelling *means* is dft.py's to define, and this defers to it:
    ``meep_frequency_span`` expands the triple (including MEEP's ``nfreq = 1``
    midpoint case) and ``normalize_frequencies`` validates the explicit lists. This
    function only arbitrates between the spellings, which is the driver's own API
    surface.
    """
    spellings = []
    if frequencies is not None:
        spellings.append("frequencies")
    if frequency is not None:
        spellings.append("frequency")
    if fcen is not None or df is not None or nfreq is not None:
        spellings.append("fcen/df/nfreq")
    if len(spellings) != 1:
        raise ValueError(
            "A monitor needs exactly one frequency specification — 'frequencies' (a number or a "
            f"sequence), the equivalent 'frequency', or 'fcen' + 'df' + 'nfreq' — got {spellings or 'none'}."
        )
    if spellings[0] == "fcen/df/nfreq":
        if fcen is None or df is None or nfreq is None:
            raise ValueError(
                f"MEEP-style monitor frequencies need all of fcen, df and nfreq; got fcen={fcen!r}, "
                f"df={df!r}, nfreq={nfreq!r}."
            )
        return meep_frequency_span(fcen, df, nfreq)
    return normalize_frequencies(frequency, frequencies)


class _FieldDecayProbe:
    """MEEP's ``stop_when_fields_decayed``: trailing-window peak of one field point.

    Translation of ``stop_when_fields_decayed`` (meep/simulation.py) plus the
    "after sources" gate ``_run_sources_until`` wraps around it. The peak of
    ``|f|**2`` at the probe point is collected over a trailing window of
    ``window`` time units; at each window boundary the window peak is compared
    against the largest window peak seen so far, and the run stops once

        window_peak <= running_peak * decay_by

    Sampling the peak over a window rather than the instantaneous value is what
    keeps a field that merely passes through zero from reading as decayed.

    The probe is read after each step rather than before it, as MEEP's step loop
    does, so the windows are offset by one step from MEEP's; over a window of
    thousands of steps that moves no decision.

    Two departures from MEEP, both deliberate. The peak is tracked as ``|f|`` and
    squared in host double precision at the window boundary, so a probe far below
    float32's smallest normal cannot underflow its own decay ratio to zero. And
    the criterion is refused outright while the running peak is still zero: MEEP's
    ``0 <= 0 * decay_by`` is true, so a run whose probe never sees any field —
    wrong probe point, source that never fires — terminates immediately and
    reports a perfectly decayed run. Here it runs to the caller's ceiling and
    raises instead.
    """

    def __init__(self, xp, read_magnitude, decay_by, window, start_time, source_end_time):
        self._xp = xp
        self._read = read_magnitude
        self._decay_by = decay_by
        self._window = window
        self._window_start = start_time
        self._source_end_time = source_end_time
        self._window_peak = None  # Running max of |f| over the current window, kept on the device.
        self._running_peak = 0.0
        self._last_ratio = None

    def update(self, time: float) -> bool:  # Sample the probe; True once the field has decayed.
        sample = self._read()
        # xp.maximum keeps the window peak where the field lives: on CuPy this is a
        # device-side compare, so the per-step probe costs no host transfer.
        self._window_peak = sample if self._window_peak is None else self._xp.maximum(self._window_peak, sample)
        if time <= self._window_start + self._window:
            return False
        window_peak = float(self._window_peak)  # One host transfer per window, not per step.
        self._window_peak = None
        self._window_start = time
        self._running_peak = max(self._running_peak, window_peak)
        if self._running_peak <= 0.0:
            return False  # Nothing was ever measured; an empty run must not pass as a decayed one.
        self._last_ratio = (window_peak / self._running_peak) ** 2
        if time < self._source_end_time:
            return False  # MEEP gates the criterion on the sources having finished.
        return self._last_ratio <= self._decay_by

    @property
    def ratio(self) -> float | None:  # Last measured |f|^2 window peak over the running peak.
        return self._last_ratio

    @property
    def peak(self) -> float:  # Largest |f| any completed window has seen.
        return self._running_peak


class FdtdDivergence(RuntimeError):
    """A dispersive run whose field energy grew past every physical explanation.

    Separate from :class:`~.dispersion.DispersionInstability`, which is raised at
    setup by a closed-form pole test. This one fires mid-run, for the instabilities
    that test cannot see — a susceptibility interacting with an absorber, a
    conductivity, or a boundary. Both exist because a diverging dispersive FDTD run
    does not crash: it returns a large, smooth, entirely finite wrong answer, and a
    normalised transmission spectrum computed from it still looks like a curve.
    """


class FdtdNonlinearityOutOfRange(RuntimeError):
    """A chi2/chi3 run whose field left the range MEEP's Pade approximant describes.

    Raised mid-run, not at setup, because the quantity that leaves the range is the
    FIELD and not the material: the same chi3 is unremarkable at one source amplitude
    and past the pole at ten times it. Separate from :class:`FdtdDivergence`, which
    watches energy growth in a passive dispersive medium, because the failure is
    different in kind — nothing here diverges. Past the pole of
    ``u = (1 + c2 + 2*c3)/(1 + 2*c2 + 3*c3)`` the recovered E is finite, smooth,
    plausible and of the WRONG SIGN, and a spectrum computed from it is a spectrum.
    """


class _NonlinearityGuard:
    """Hold a nonlinear run to the range MEEP's Pade approximant is a valid expansion in.

    Samples :func:`~.stepping.nonlinear_margin` on the same interval the energy guard
    uses and raises once the expansion parameter ``max(|c2| + |c3|)`` reaches
    ``limit``. The default limit is derived rather than chosen:
    ``1 + 2*c2 + 3*c3 >= 1 - 3*(|c2| + |c3|)``, so ``|c2| + |c3| < 1/3`` is exactly
    the condition that keeps the approximant's denominator on the positive side of
    its pole, and 1/3 is therefore the largest limit that can be defended at all.

    MEEP performs no such check and will return the post-pole field. Doing the same
    here would put a silent wrong answer at ordinary-looking inputs — a user who
    doubles a source amplitude gets a number back either way.
    """

    def __init__(self, driver: "FdtdDriver", interval: int = _STABILITY_CHECK_INTERVAL,
                 limit: float = _NONLINEAR_EXPANSION_LIMIT):
        self._driver = driver
        self._interval = max(1, int(interval))
        self._limit = float(limit)

    def update(self) -> None:  # Sample every `interval` steps; raise once the Pade range is left.
        if self._driver.step_count % self._interval:
            return
        margin = nonlinear_margin(self._driver.fields, self._driver.pml)
        if margin is None or margin.expansion < self._limit:
            return
        raise FdtdNonlinearityOutOfRange(
            f"The instantaneous nonlinearity left the range MEEP's Pade approximant describes after "
            f"{self._driver.step_count} steps: max(|c2| + |c3|) reached {margin.expansion:.3g} on "
            f"{margin.component} (limit {self._limit:.3g}), and the approximant's denominator "
            f"1 + 2*c2 + 3*c3 fell to {margin.denominator:.3g} — its pole is at 0, past which the "
            f"recovered E flips sign while staying finite and smooth. c2 = D*chi2*inv_eps^2 and "
            f"c3 = |D|^2*chi3*inv_eps^3, so lower the source amplitude, lower chi2/chi3, or raise "
            f"eps. Note the chi2/chi3 power series itself is not physical at this strength either "
            f"(MEEP step_generic.cpp:537-541), so a more exact cubic solve would not rescue it."
        )


class _DivergenceGuard:
    """Watch the field energy of a dispersive run and raise the moment it runs away.

    MEEP aborts on a non-finite energy density every step (step.cpp:137-138). NaN
    arrives late, though — the dangerous mode grows smoothly for thousands of steps
    first — so this also enforces a physical bound: in a passive medium
    (gamma >= 0, sigma >= 0) under periodic or absorbing boundaries, the field energy
    after the last source has stopped emitting is non-increasing. Any sustained
    growth there is numerical, whatever it looks like.

    The peak is tracked while the sources are still emitting and the ceiling applies
    only afterwards, so a run that is simply still being driven is never accused.
    """

    def __init__(self, driver: "FdtdDriver", interval: int = _STABILITY_CHECK_INTERVAL,
                 growth_limit: float = _STABILITY_GROWTH_LIMIT):
        self._driver = driver
        self._interval = max(1, int(interval))
        self._growth_limit = float(growth_limit)
        self._source_end = driver._last_source_time()
        self._peak = 0.0

    def update(self) -> None:  # Sample every `interval` steps; raise on non-finite or runaway energy.
        if self._driver.step_count % self._interval:
            return
        energy = self._driver._field_energy()
        if not math.isfinite(energy):
            raise FdtdDivergence(
                f"Field energy is not finite ({energy}) after {self._driver.step_count} steps of a "
                f"dispersive run. The likeliest cause is the susceptibility with the largest "
                f"discrete pole: {self._driver._worst_susceptibility()}. Raise the resolution, "
                f"lower the Courant factor, or refit the material."
            )
        if self._driver.time <= self._source_end:
            self._peak = max(self._peak, energy)
            return
        if self._peak > 0.0 and energy > self._growth_limit * self._peak:
            raise FdtdDivergence(
                f"Field energy grew to {energy:.3e} after {self._driver.step_count} steps, "
                f"{energy / self._peak:.3e} times the peak {self._peak:.3e} reached while the "
                f"sources were still emitting (they stopped at t={self._source_end:g}, it is now "
                f"t={self._driver.time:g}). A passive medium under these boundaries cannot gain "
                f"energy after the drive stops, so this run is diverging — it would still have "
                f"returned a smooth, finite, entirely wrong field. Most likely responsible: "
                f"{self._driver._worst_susceptibility()}."
            )
        self._peak = max(self._peak, energy)


class FdtdDriver:
    """FDTD run driver: grid + fields + PML + sources + monitors, stepped in MEEP's order.

    Attributes:
        grid: :class:`Grid` holding the resolution, dt/dx, symmetry, and array module.
        fields: :class:`Fields` holding D/B (always) and E/H plus PML auxiliaries (with PML).
        pml: :class:`PML` coefficient set, or None when the run has no absorbing boundary.
        step_count: Completed time steps; simulation time is ``step_count * grid.dt``.
        xp: The resolved array module (NumPy or CuPy).
        gpu: What ``prefer_gpu`` resolved to: ``"cuda"`` (CuPy), ``"metal"`` (Metal
            kernels over NumPy host arrays), or ``None`` for the NumPy reference,
            which never consults a kernel table.
    """

    def __init__(
        self,
        cell_size: Sequence[float],
        resolution: float,
        courant: float = 0.5,
        force_complex_fields: bool = True,
        symmetry: Sequence[MirrorSpec] = (),
        k_point: Sequence[float] = (0.0, 0.0, 0.0),
        boundaries: Any = None,
        dimensions: int = 3,
        cylindrical: bool = False,
        m: int = 0,
        accurate_fields_near_cylorigin: bool = False,
        beta: float = 0.0,
        bfast_scaled_k: Sequence[float] = (0.0, 0.0, 0.0),
        prefer_gpu: bool = False,
        gpu_id: int = 0,
    ):
        """Build the grid and field storage for one run.

        Args:
            cell_size: (Lx, Ly, Lz) domain size in MEEP length units.
            resolution: Grid points per unit length; dx = 1 / resolution.
            courant: Courant factor; dt = courant / resolution.
            force_complex_fields: Complex64 fields when True (the default here),
                float32 when False. Both modes are cross-validated against CPU MEEP —
                real mode against MEEP's *own* default real storage (plain 2.5e-7,
                uniform PML 3.6e-7, Lorentz 2.8e-6) — and a real run is bit-for-bit the
                real part of the same run in complex mode, which halves the field
                allocation. Note this default is the opposite of MEEP's: complex is kept
                as the default here because every accuracy floor in the suite was
                measured in it. A nonzero ``k_point``, a complex source amplitude and a
                complex ``amp_func`` all require complex fields and are refused in real
                mode rather than silently losing their imaginary part.
            symmetry: Mirror planes, spelled as MEEP's ``mp.Mirror(direction, phase)``
                pair: :class:`~.grid.Mirror` objects, or the bare axis name that is
                shorthand for MEEP's own default even plane. Any of 'X', 'Y' and 'Z'
                may be folded, at either phase, and any combination of them —
                ``symmetry=(Mirror('X', -1), 'Z')`` is an odd X plane with an even Z
                one. An odd plane is the same fold with every component's parity
                inverted (:func:`~.fields.mirror_parity`), which is what lets a
                current along the plane's own normal — an ``Ex`` dipole on ``x = 0``,
                refused under an even X mirror — be folded at all. Each plane halves
                its axis to MEEP's ``n_full - n_full // 2 + 1`` owned cells, plus
                the ``big_corner`` plane and its ghost slot on a periodic axis
                (MEEP's ``halve()`` and ``update_ntot``,
                even and odd full counts alike — an odd count is MEEP's own fold
                about a grid point, with the window shifted half a cell up), so it
                divides both the memory and the work by two.
            k_point: Bloch wavevector (kx, ky, kz) in MEEP's units of 2*pi/distance —
                the same numbers ``mp.Simulation(k_point=mp.Vector3(...))`` takes — so
                the field one lattice vector up axis d is ``exp(i*2*pi*k_d*L_d)`` times
                the field at the original point. All zeros (the default) is plain
                periodicity and steps bit-identically to a run built without the
                argument at all. See :meth:`_require_bloch_is_representable` for the
                combinations this refuses.
            boundaries: Outer boundary condition per axis — ``"periodic"`` (the
                default) or ``"metallic"``, a perfect electric conductor. Spellable as
                one condition for all three axes, a ``{"x": "metallic"}`` mapping, or a
                3-sequence; see :func:`~.grid._normalize_boundaries`. ``None`` (the
                default) is periodic everywhere and steps bit-identically to a run built
                without the argument at all.

                **This default is the opposite of MEEP's, on purpose.** MEEP starts
                every face ``Metallic`` and only ``k_point`` turns an axis periodic
                (fields.cpp, boundaries.cpp ``use_bloch``), so ``mp.Simulation(...)``
                with no ``k_point`` is a PEC box — which is what
                ``from_meep.lift_simulation`` translates into
                ``boundaries="metallic"``. This driver keeps periodic as its own default
                because every floor in its suite was measured in it; flipping it would
                change the answer of every existing caller silently. The two are
                1.28e+00 complex relative L2 apart on a 2x2x4 cell at resolution 10 with
                a CW point source and no absorber, so the choice is a different
                simulation, not a different accuracy.

                Combines freely with a PML (the layer absorbs before the wall reflects,
                which is the usual MEEP arrangement), with ``symmetry`` (MEEP's own
                default box is folded AND metallic), and with a periodic or Bloch-phased
                run on the other axes. A nonzero ``k_point`` on a metallic axis is
                refused: a PEC wall gives that axis no lattice vector for a phase to
                describe, and MEEP cannot express it either.
            dimensions: 3 (the default), 2 or 1 — MEEP's ``dimensions``. A reduced run
                carries its missing axes as TRANSLATIONAL INVARIANCE, which is what
                MEEP means by 2-D and what makes the TE (Ex, Ey, Hz) and TM
                (Ez, Hx, Hy) polarizations decouple: every field is constant along the
                missing axis and every derivative along it is identically zero. It is
                NOT a thin slab — a slab of finite thickness radiates out of plane and
                a 2-D run cannot.

                ``dimensions=2`` makes **z** invariant (MEEP's X-Y plane) and
                ``dimensions=1`` makes **x and y** invariant (MEEP's z axis); the
                assignment is MEEP's ``LOOP_OVER_DIRECTIONS`` and not a choice made
                here. ``cell_size`` must be exactly 0 on each invariant axis, and that
                axis then holds one cell with a periodic wrap — MEEP's own
                lower-dimensional emulation (fields.cpp ``nosize_direction``), which
                steps to the same bytes as MEEP's genuine 2-D on the same problem.
                A metallic wall, a mirror plane, a Bloch phase, a PML or a monitor
                extent on an invariant axis are each refused; see
                :class:`~.grid.Grid` for what they would silently do.
            beta: MEEP's out-of-plane 2-D wavevector — ``fields::beta``, the mode
                ``mp.Simulation(kz_2d="complex" | "real/imag")`` selects on a
                zero-thickness cell with a nonzero ``k_point.z``. The invariant axis
                then carries an ANALYTIC ``exp(i*2*pi*beta*z)`` dependence instead of
                grid points, so ``d/dz`` is the exact factor ``i*2*pi*beta`` and the
                curl gains an ``i*beta*zhat x`` cross product that couples the TE and
                TM polarizations (:func:`~.stepping._special_kz_beta_term`). It costs
                no storage and no extra pass, and it is NOT a Bloch phase: nothing
                wraps, and ``k_point`` must keep a hard zero on z (MEEP hands
                ``use_bloch`` only ``Vector3(kx, ky)``, simulation.py:2498-2504).
                Legal only at ``dimensions=2``. Works in BOTH storage modes — real
                storage is MEEP's implicit-i-on-TM trick, which is the whole point of
                the ``"real/imag"`` spelling — but is refused in real storage together
                with an off-diagonal epsilon, as MEEP refuses it.
            bfast_scaled_k: MEEP's ``bfast_scaled_k`` — the BROADBAND FIXED-ANGLE
                SOURCE TECHNIQUE, i.e. the scaled wavevector ``n * sin(theta)`` of the
                incidence direction. It shears time by ``t -> t - k.r/c``, adding
                ``+d/dt (k x E)`` to dB/dt and ``-d/dt (k x H)`` to dD/dt
                (:func:`~.stepping._bfast_term`), which makes a planewave at a FIXED
                angle transversely uniform at every frequency, so ordinary periodic
                boundaries serve a whole BAND at one angle instead of one frequency.
                Costs one auxiliary array per D and B component. Refused on a
                cylindrical grid; a nonzero ``k_point`` alongside it is allowed, as
                MEEP allows it.
            prefer_gpu: Run on this host's GPU: CUDA through CuPy, or Metal kernels
                over NumPy host arrays on an Apple GPU; raises when the host has
                neither. False (the default) is the NumPy reference: host arrays and
                the array path, with no kernel table consulted whatever
                ``MEEP_GPU_DISPATCH`` says.
            gpu_id: CUDA device index, used only when ``prefer_gpu`` is set; must be 0
                on an Apple GPU.
        """
        cell = _as_xyz("cell_size", cell_size)
        # Cylindrical resolves r and z with phi invariant — an axis table no
        # `dimensions` value spells; the Grid re-validates the same facts.
        invariant = (1,) if cylindrical else INVARIANT_AXES_BY_DIMENSIONS.get(dimensions, ())
        if any(
            length <= 0.0 for axis, length in enumerate(cell) if axis not in invariant
        ):
            raise ValueError(
                f"cell_size must be positive along every axis this run resolves, got {cell}"
                + (
                    f" at dimensions={dimensions!r}, which makes "
                    f"{', '.join(AXIS_NAMES[axis] for axis in invariant)} invariant (and requires "
                    f"exactly 0 there)."
                    if invariant else "."
                )
            )
        resolution_value = float(resolution)
        if not math.isfinite(resolution_value) or resolution_value <= 0.0:
            raise ValueError(f"resolution must be finite and positive, got {resolution!r}.")
        courant_value = float(courant)
        # The CFL bound counts the axes this run differences, so a reduced run gets the
        # larger limit MEEP's own stability analysis gives it (1/sqrt(2) in 2-D, 1 in
        # 1-D). `Grid` validates `dimensions` itself; an unrecognized value falls back
        # to the 3-D bound here so the message a caller sees names the real problem.
        courant_limit = _courant_limit(dimensions) if dimensions in INVARIANT_AXES_BY_DIMENSIONS else _COURANT_LIMIT_3D
        if not math.isfinite(courant_value) or courant_value <= 0.0 or courant_value > courant_limit:
            raise ValueError(
                f"courant must lie in (0, {courant_limit:.6f}] — the {dimensions}-D CFL stability "
                f"limit 1/sqrt({dimensions}) — got {courant!r}."
            )
        planes = _normalize_symmetry(symmetry)
        wavevector = _as_xyz("k_point", k_point)
        self.force_complex_fields = bool(force_complex_fields)
        self._require_bloch_is_representable(wavevector, planes)
        self.xp, self.gpu = resolve_backend(prefer_gpu=prefer_gpu, gpu_id=gpu_id)
        self.grid = Grid(
            resolution=resolution_value,
            cell_size=cell,
            courant=courant_value,
            symmetry=planes,
            k_point=wavevector,
            boundaries=boundaries,
            dimensions=dimensions,
            # MEEP Dcyl under the x -> r, y -> phi, z -> z mapping; the Grid
            # carries the mode's constraints and refusals
            # (the design notes (fdtd-cylindrical-plan)).
            cylindrical=bool(cylindrical),
            m=m,
            # MEEP's accurate_fields_near_cylorigin: the |m| >= 2 near-axis branch that
            # holds only the r = 0 row at zero instead of |m| rows, at the cost of a
            # Courant bound of ~1/(|m| + 0.5) that the Grid enforces.
            accurate_fields_near_cylorigin=bool(accurate_fields_near_cylorigin),
            # MEEP's special_kz: the out-of-plane wavevector of a 2-D run, carried as
            # an analytic phase rather than as a third axis (fields.cpp:546-549 owns
            # the one restriction, which the Grid re-states).
            beta=float(beta),
            # MEEP's bfast_scaled_k: the broadband fixed-angle shear, a `fields`
            # constructor slot (python/simulation.py:2486) that touches neither the
            # structure nor the sources — see stepping._bfast_term.
            bfast_scaled_k=_as_xyz("bfast_scaled_k", bfast_scaled_k),
            xp=self.xp,
        )
        if self.grid.cylindrical and self.grid.m != 0 and not self.force_complex_fields:
            # The same refusal _cylindrical_imr_term raises at the first step,
            # moved to where MEEP decides storage (fields.cpp:824): at build
            # time, before any setup work is spent on an impossible run.
            raise ValueError(
                f"cylindrical m={self.grid.m} needs complex fields: the i*m/r coupling "
                f"puts the partner's field a quarter turn out of phase, which float32 "
                f"storage cannot carry (MEEP promotes the same combination silently, "
                f"simulation.py:2517-2522). Build the run with force_complex_fields=True."
            )
        self.fields = Fields(self.grid, force_complex_fields=self.force_complex_fields)
        self.pml: PML | None = None
        self.step_count = 0
        # Fast-path plan cache (fastpath.plan_fast_path): built at the next step(),
        # once the configuration has frozen; None is the array path. Stale means
        # "not planned yet" — construction, and every invalidate_fast_path() call a
        # material mutator makes, leaves it True so the next step re-plans.
        self._fast_path = None
        self._fast_path_stale = True
        # THIS driver's own dispatch artifact, captured at its own freeze. A refused
        # driver holds no plan to serve a report from, and reading the module-level
        # last record instead made it serve ANOTHER driver's: a refused driver
        # reported decision='dispatched', step_path='fused' while stepping the array
        # path, because a second driver had frozen in between. One slot per driver,
        # and a driver that has not frozen yet has None.
        self._fast_path_record = None
        self._sources: list[Any] = []
        self._dft_monitors: list[DFTMonitor] = []
        self._flux_monitors: list[FluxMonitor] = []
        self._pending_monitor_regions: list[tuple[DFTMonitor, Any, Any]] = []
        # MEEP's high-level decimation_factor=0 is resolved from the complete source
        # list. Keeping these pending until the first step preserves this driver's
        # supported monitor-before-source ordering without changing the factor after
        # any sample has been accumulated.
        self._automatic_decimation_monitors: list[DFTMonitor | FluxMonitor] = []
        # (Susceptibility, sigma-as-given) for every added term, kept alongside the
        # PolarizationState so get_epsilon(frequency) and the stability report can talk
        # about the material rather than about its arrays.
        self._susceptibilities: list[tuple[Susceptibility, Any]] = []
        # The two independent sources of conductivity, kept apart so either can be set
        # without discarding the other and so their SUM is what reaches the fields —
        # MEEP adds the absorber's ramp to the material's sigma rather than replacing it
        # (meepgeom.cpp:1596-1625, "add their conductivities to cond_val"), measured as
        # such: a 1x material sigma plus an absorber steps bit-identically to the same
        # material function evaluated at 1x + absorber.
        # Per D component, because MEEP is: `get_cnd` reads D_conductivity_diag.x/.y/.z
        # for Dx/Dy/Dz (meepgeom.cpp:1545-1559). None means "no material loss at all";
        # an isotropic conductivity stores the SAME array object under all three keys,
        # which is what keeps such a run bit-identical to the pre-anisotropic engine.
        self._material_d_conductivity: dict[str, Any] | None = None
        # The MAGNETIC half, per B component, with exactly the same meaning. MEEP's
        # get_cnd is one switch over both sides (meepgeom.cpp:1545-1559) and
        # structure::set_materials installs them through the same FOR_D_AND_B loop
        # (structure.cpp:376-378), so the two are independent stores of one shape.
        self._material_b_conductivity: dict[str, Any] | None = None
        # True only when set_conductivity was told its volumes already carry the
        # absorber's ramp — see set_conductivity's `absorber_included`.
        self._material_conductivity_includes_absorber = False
        self._absorber_layers: tuple[AbsorberLayer, ...] = ()
        self._absorber_sigma: dict[str, Any] = {}
        # chi2/chi3 as the caller last gave them. MEEP keeps the pair together, so
        # set_chi2 reinstalls the stored chi3 alongside its own argument and vice versa;
        # 0.0 is "not set", which installs nothing at all.
        self._chi2: Any = 0.0
        self._chi3: Any = 0.0
        # (component, point) -> the zero-argument sampler get_field_point runs. Only the
        # stencil is cached; the arrays it reads are looked up per call, so nothing here
        # can outlive a reset() or a change of storage mode.
        self._point_readers: dict[tuple[str, tuple[float, float, float]], Callable[[], Any]] = {}
        # MEEP's `synchronized_magnetic_fields` counter and the arrays it stands over
        # (energy_and_flux.cpp:146-186): nesting is a no-op and only the outermost
        # restore puts B, H and their PML auxiliaries back.
        self._synchronized_magnetic: int = 0
        self._magnetic_backup: dict[str, Any] = {}
        self._closed = False

    def _require_bloch_is_representable(
        self, wavevector: tuple[float, float, float], planes: tuple[Mirror, ...]
    ):  # Refuse the k_point combinations this engine cannot run.
        """Reject a nonzero ``k_point`` combined with real fields or a mirror on its own axis.

        Bloch periodicity multiplies the wrapped field by ``exp(i*2*pi*k_d*L_d)``, so
        a real-field run has nowhere to put the phase. MEEP refuses the same pairing
        outright (boundaries.cpp ``use_bloch``: "Can't use real fields with bloch
        boundary conditions"), and it is refused here rather than silently promoted
        to complex, which would quietly double the memory of a run the caller sized.

        What is refused about symmetry, and what is not:

        * A nonzero k on a mirror plane's OWN axis is refused. Inside the Brillouin
          zone the plane makes the field even or odd about itself, which is a
          statement that the field repeats with no phase — only k_d = 0 satisfies it,
          so anything else is a contradiction rather than an approximation. The ZONE
          EDGE is the one exception, and it is refused here too: MEEP supports it
          (a474edde / #3155, whose ``tests/symmetry.cpp::test_periodic_mirror_zone_edge``
          folds Y at k_y = zone edge, "compatible with Mirror(Y) (fields are
          anti-periodic in Y)", with ``eikna`` taken over ``user_volume`` — the FULL
          cell — rather than the symmetry-halved ``gv``). No corpus row asks for it
          and this engine has not implemented the full-cell period, so it stays a
          refusal; the message says "zone interior" so it does not read as a claim
          that the pairing is impossible.
        * A mirror on one axis with a Bloch phase on ANOTHER is supported, and is
          ordinary MEEP: the two phases compose multiplicatively with no cross term.
          MEEP accumulates the lattice phase over the shift loop
          (loop_in_chunks.cpp:415 ``ph *= pow(eikna[d], ishift.in_direction(d))``,
          gated at :393 on ``boundaries[High][...] == Periodic``) and multiplies in
          the symmetry phase separately (``symmetry::phase_shift``, vec.cpp:1347),
          the two combining only at loop_in_chunks.cpp:202
          ``phases[nc] = shift_phase * S.phase_shift(...)``.

          This engine composes them the same way because the readback is PER AXIS.
          :meth:`~.fields.Fields.to_cell_center` sends a mirrored axis down the fold
          branch (zero ghost on a metallic termination, ``mirror_parity *
          result[reflect_row]`` on a periodic one) and every unfolded axis down
          :meth:`~.fields.Fields._wrapped_neighbour`, which applies
          ``grid.bloch_phase``; :meth:`~.from_meep.GpuRunResult.get_array` applies
          ``conj(bloch_phase)`` to the prepended plane and skips mirrored axes. The
          folded axis needs no Bloch factor precisely because the branch above forces
          its k to zero.

          MEASURED on this tree (whole-volume complex relative L2, all six
          components, MEEP ``get_array`` layout): fold(Y,-1) with k_x = 3.5
          **1.030e-06**, against controls of 1.021e-06 for the fold alone and
          1.028e-06 for k_x = 3.5 unfolded; fold(Y,-1) with k_x = 0.1234567
          9.542e-07; fold(X,+1) with k = (0, 0.3, 0.2) in 3-D 1.321e-07 against
          1.294e-07 unfolded; 2-D with ``special_kz`` beta and fold(Y,+1) 4.412e-07
          against 4.510e-07 unfolded. Dropping the phase from ``_wrapped_neighbour``
          moves the folded run to 2.728e-01 and moves the folded and unfolded legs by
          the SAME amount (0.2727518730 against 0.2727518579), which is the evidence
          that the fold routes its Bloch axis through the unfolded code and adds no
          separate hazard.

        The PML pairing is refused separately, in :meth:`setup_pml`, because the layer
        is added after construction — and only per axis: a Bloch phase on an axis with
        no absorber is exactly the grating configuration and is supported.
        """
        bloch_axes = [
            AXIS_NAMES[axis] for axis in range(3) if wavevector[axis] != 0.0
        ]
        if not bloch_axes:
            return
        if not self.force_complex_fields:
            raise ValueError(
                f"k_point={wavevector} is nonzero on {bloch_axes}, which needs complex fields to "
                f"carry the boundary phase exp(i*2*pi*k*L); this run was built with "
                f"force_complex_fields=False. MEEP refuses the same combination."
            )
        if not planes:
            return
        clashing = [plane for plane in planes if wavevector[plane.axis_index] != 0.0]
        if clashing:
            raise ValueError(
                f"k_point={wavevector} is nonzero on the mirror-symmetric axis / axes {clashing}: "
                f"in the Brillouin zone interior a mirror plane forces the field to be even or odd "
                f"about it, which requires that axis's k component to be exactly 0. Drop the "
                f"symmetry or zero that component. (The zone EDGE is compatible with a mirror — "
                f"MEEP #3155 — but needs the lattice phase taken over the full, unfolded cell and "
                f"is not implemented here.) A Bloch phase on an axis that is NOT folded is "
                f"supported and needs no change."
            )

    def set_epsilon(self, eps: Any):  # Install the scalar permittivity volume and its inverse.
        """Set the (real, isotropic, non-dispersive) permittivity array.

        Accepts either this grid's own shape ``(nx, ny, nz)`` or MEEP's
        ``get_epsilon()`` shape ``(nx+1, ny+1, nz+1)``; every axis is checked, so a
        half-matching array raises instead of slipping through (port reference §5.8
        — the bundle's second branch was unreachable).

        Registration, RESOLVED by a non-uniform-epsilon oracle (the Bloch grating in
        ``test_driver_integration.py`` and ``test_driver_vs_meep.py``): ``inv_eps[i,
        j, k]`` must hold the permittivity at the INTEGER Yee position
        ``grid.axis_origin(d) + i*dx`` on each axis — not at the cell centre, and not
        measured from ``-L/2``, which is the origin only when that axis's cell count
        is even (:meth:`Grid.origin_doubled`). Measured on a grating transmission
        spectrum at resolution 15: integer sampling from MEEP's origin 3.0e-07,
        cell-centre sampling 6.7e-02, integer sampling from -L/2 6.6e-02.

        Passing MEEP's own ``(N+1)`` edge-registered array still takes the bundle's
        ``[:-1, :-1, :-1]`` trim, which lands half a cell either side of that
        position; ``0.5 * (eps[:-1] + eps[1:])`` would hit it. That resampling is not
        done here — the trim is what every array-fed number was measured with — so
        prefer supplying an ``(nx, ny, nz)`` array sampled at the position above.

        This is the shared-array shorthand: all three D components use the same
        registration. For diagonal anisotropy, or an isotropic geometry sampled
        separately at the three E-component Yee positions, use
        :meth:`set_epsilon_components`.

        UNDER A MIRROR FOLD, PREFER :meth:`set_epsilon_components`. One shared array
        indexed at the integer Yee position is half a cell off for the component whose
        Yee shift on the folded axis is 1 — ``Ex`` under an X plane sits at
        ``p_i + dx/2`` and is multiplied by the permittivity at ``p_i``, so the two
        cells straddling the plane (at -dx/2 and +dx/2) take eps(-dx) and eps(0). Those
        are not mirror images of each other, so a spatially varying epsilon installed
        this way is not exactly symmetric about the plane even when the geometry is,
        and it is the FULL-domain run that drifts: the folded run enforces the symmetry
        exactly through its ghost cell. Measured on a 3-D cell at resolution 10 with an
        X fold and epsilon graded ACROSS the plane, folded against unfolded on the raw
        D arrays: 1.9e-03 after 12 steps and 1.2e-02 after 20, growing, and identically
        0.0 for a uniform medium, for grading along either UNFOLDED axis, and for the
        same graded medium installed through :meth:`set_epsilon_components` at each
        component's own coordinates (:func:`~.dispersion.component_coordinates`). The
        fold itself is exact in every one of those cases; what varies is whether the
        two runs were given the same medium.
        """
        self._require_open()
        trimmed = self._coerce_epsilon_array(eps, "Epsilon")
        epsilon = self.xp.ascontiguousarray(trimmed)
        inverse = (1.0 / epsilon).astype(self.xp.float32)
        self.fields.set_isotropic_epsilon_volume(epsilon, inverse)
        self.invalidate_fast_path()  # New material arrays: any cached plan pointer is stale.

    def set_epsilon_components(
        self,
        epsilon: Mapping[str, Any],
        chi1inv_offdiagonal: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        """Install per-component permittivity volumes, optionally with a full tensor row.

        The ``epsilon`` mapping must contain exactly ``Ex``, ``Ey`` and ``Ez``.
        Uniform diagonal anisotropy is represented by three different constant
        arrays; a spatially varying isotropic material reaches MEEP's Yee
        registration by sampling the same scalar geometry separately at each
        component's coordinates (see :func:`dispersion.component_coordinates`).
        Each diagonal value is an EFFECTIVE permittivity: its reciprocal is the
        diagonal chi1inv entry, which for a rotated medium is ``(eps^-1)_cc``
        and not ``eps_cc``.

        ``chi1inv_offdiagonal`` carries MEEP's off-diagonal ``chi1inv[c][d]``
        rows — ``{row component: {partner component: volume}}`` — taken DIRECTLY
        (they are inverse-tensor entries, not permittivities: free to be
        negative, zero, or tiny, and never inverted here). MEEP registers them
        at the row component's Yee site minus half a cell along its own axis
        (the integer node, anisotropic_averaging.cpp:248-257), and the stencil
        in ``stepping._offdiagonal_terms`` consumes them slot-for-slot with the
        same convention, so callers hand over exactly what ``fields.get_chi1inv``
        reports and never restate the registration. Rows that are identically
        zero are dropped, reducing byte for byte to the diagonal engine; a mirror-folded
        grid takes any surviving row UNCHANGED (the stencil never ghosts the coefficient, so
        the fold's field-parity ghosts are the whole story — 8.3e-13..4.7e-12 equivalence).
        """
        self._require_open()
        if not isinstance(epsilon, Mapping):
            raise ValueError(
                "set_epsilon_components() requires a mapping with Ex, Ey, and Ez arrays."
            )
        expected = {"Ex", "Ey", "Ez"}
        unknown = sorted(set(epsilon) - expected)
        missing = sorted(expected - set(epsilon))
        if unknown or missing:
            raise ValueError(
                "epsilon component mapping must contain exactly Ex, Ey, and Ez; "
                f"missing={missing}, unexpected={unknown}."
            )
        epsilon_components = {
            component: self.xp.ascontiguousarray(
                self._coerce_epsilon_array(
                    epsilon[component],
                    f"Epsilon[{component!r}]",
                )
            )
            for component in ("Ex", "Ey", "Ez")
        }
        inverse_components = {
            component: (1.0 / values).astype(self.xp.float32)
            for component, values in epsilon_components.items()
        }
        offdiagonal = None
        if chi1inv_offdiagonal is not None:
            offdiagonal = {
                row: {
                    partner: self._coerce_offdiagonal_array(
                        values, f"chi1inv_offdiagonal[{row!r}][{partner!r}]"
                    )
                    for partner, values in (partners or {}).items()
                }
                for row, partners in chi1inv_offdiagonal.items()
            }
        self.fields.set_epsilon_volumes(
            epsilon_components, inverse_components, chi1inv_offdiagonal=offdiagonal
        )
        self.invalidate_fast_path()  # New material arrays: any cached plan pointer is stale.

    def _coerce_offdiagonal_array(self, values: Any, label: str):
        """Copy and validate one off-diagonal chi1inv volume.

        Unlike a permittivity it may be negative or zero — it is an
        inverse-tensor entry — so the only constraints are realness, finiteness
        and the stored grid shape (MEEP's boundary-padded shape is trimmed the
        same way the diagonal's is).
        """
        probe = self.xp.asarray(values)
        _reject_complex(
            probe,
            label,
            "The inverse-permittivity tensor of a lossless medium is real; a complex entry is "
            "loss or dispersion, which this engine expresses through set_conductivity(...) or "
            "add_susceptibility(...), never through the tensor.",
        )
        array = self.xp.array(probe, dtype=self.xp.float32, order="C", copy=True)
        if array.ndim != 3:
            raise ValueError(f"{label} must be a 3-D array, got {array.ndim} dimensions.")
        grid_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        meep_shape = tuple(n + 1 for n in grid_shape)
        shape = tuple(int(n) for n in array.shape)
        if shape == grid_shape:
            trimmed = array
        elif shape == meep_shape:
            trimmed = array[:-1, :-1, :-1]
        else:
            raise ValueError(
                f"{label} shape {shape} matches neither the grid shape {grid_shape} nor MEEP's "
                f"boundary-padded shape {meep_shape}."
            )
        if not bool(self.xp.all(self.xp.isfinite(trimmed))):
            raise ValueError(f"{label} must be finite in every cell.")
        return self.xp.ascontiguousarray(trimmed)

    def _coerce_epsilon_array(self, epsilon: Any, label: str):
        """Copy, validate, and trim one epsilon volume to the stored grid."""
        # Copy rather than adopt: the caller's array must not stay wired into the
        # material, and a later mutation of theirs must not silently change physics.
        probe = self.xp.asarray(epsilon)
        _reject_complex(
            probe,
            label,
            "A complex permittivity is loss or dispersion: pass its real part here and express the "
            "imaginary part with set_conductivity(...) for broadband loss, or add_susceptibility(...) "
            "for a Lorentz/Drude model.",
        )
        values = self.xp.array(probe, dtype=self.xp.float32, order="C", copy=True)
        if values.ndim != 3:
            raise ValueError(f"{label} must be a 3-D array, got {values.ndim} dimensions.")
        grid_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        meep_shape = tuple(n + 1 for n in grid_shape)
        shape = tuple(int(n) for n in values.shape)
        if shape == grid_shape:
            trimmed = values
        elif shape == meep_shape:
            trimmed = values[:-1, :-1, :-1]
        else:
            raise ValueError(
                f"{label} shape {shape} matches neither the grid shape {grid_shape} nor MEEP's "
                f"boundary-padded shape {meep_shape}."
            )
        # Strictly positive, OR the PEC sentinel by name. Relaxing this to "negative is
        # fine" would admit an epsilon = -3 cell, whose leapfrog has no stable step at
        # any Courant factor; the sentinel is admitted because it is MEEP's own
        # spelling of a perfect conductor and steps to ~0 rather than growing.
        unusable = ~(self.xp.isfinite(trimmed) & ((trimmed > 0.0) | is_pec_epsilon(trimmed)))
        if bool(self.xp.any(unusable)):
            raise ValueError(
                f"{label} must be finite and strictly positive in every cell (or MEEP's "
                f"perfect-conductor sentinel {PEC_EPSILON_SENTINEL:g}, which mp.metal and "
                f"mp.perfect_electric_conductor install)."
            )
        return trimmed

    def set_epsilon_smoothed(
        self,
        epsilon: Any,
        component: str,
        supersample: Any = DEFAULT_SUPERSAMPLE,
        variation_tolerance: float = 0.0,
        fill_rule: str = DEFAULT_FILL_RULE,
    ) -> SmoothedEpsilon:
        """Install a subpixel-smoothed permittivity — MEEP's ``eps_averaging=True``.

        A staircased material interface drops FDTD to first-order accuracy and costs
        percent-level error at any resolution a user would run (measured 3.9e-01 on a
        sharp grating against CPU MEEP, dispersive or not — six orders above every
        other floor in this engine). MEEP repairs it by averaging the inverse
        permittivity TENSOR over each voxel — ``<1/eps>`` for the direction normal
        to the interface (the harmonic mean of eps) and ``1/<eps>`` for the two
        parallel ones (the arithmetic mean, inverted), with the normal read per
        voxel from a sphere-quadrature gradient (:mod:`.smoothing`, transcribed from
        ``anisotropic_averaging.cpp``). This method computes that tensor row for
        ``component`` and installs its diagonal entry as the run's scalar
        ``inv_eps``.

        At the defaults it closes that gap to 1.5e-06 on the same grating — the
        engine's own reproduction floor for the case, and independent of
        ``supersample`` — because the voxel fill fraction comes from the local
        interface PLANE in closed form rather than from a sampling sum. See
        ``fill_rule`` below, and ``planar_fraction`` on the returned report for how
        much of the interface got that treatment.

        WHICH OF THE TWO MEANS YOU ACTUALLY GET: the arithmetic one, always. The
        invariance this method requires (below) forces the epsilon gradient along
        ``component``'s own axis to vanish exactly, so ``n_d = 0``, the diagonal
        collapses to ``1/<eps>``, and the interface normal has no influence on the
        installed array at all. That is not a shortcut but the correct effective
        medium for this configuration — a structure invariant along ``d`` driven in
        ``E_d`` has E tangential to every interface it owns, and MEEP computes the
        same ``1/<eps>`` there. The harmonic branch belongs to the other
        polarization and needs the per-component tensor this engine cannot store, so
        it is verified in :mod:`.smoothing` and reached by nothing.

        THE SCALAR CONTRACT. This engine stores ONE ``inv_eps`` volume shared by all
        three D components, so it can carry exactly one row of MEEP's tensor. That
        reduction is exact — not approximate — when both of these hold:

        * epsilon is invariant along ``component``'s own axis, so the interface
          normal never tilts into that axis, this row's off-diagonal entries are
          identically zero, and the row's diagonal entry is the whole story;
        * the run drives only the polarization containing ``component`` (for a
          y-invariant structure and ``component="Ey"``: an Ey source, which keeps
          Dx and Dz identically zero, so the values the shared array would apply
          to them multiply nothing).

        That is precisely the TE grating / photonic-crystal configuration the
        per-axis PML + Bloch support exists for. The invariance is CHECKED — sampled
        on the same fine lattice the means use — and violation raises rather than
        shipping a smooth, plausible, few-percent-wrong field. The polarization half
        of the contract cannot be checked here (sources may be added later); it is
        the caller's, and stays in this docstring.

        Args:
            epsilon: A vectorized callable ``epsilon(x, y, z)`` — evaluated at
                coordinates wrapped into the periodic cell, like MEEP's
                ``ensure_periodicity`` — or a supersampled array built on
                :func:`.smoothing.fine_coordinates`. The callable form also gets
                MEEP-exact point sampling at voxel centres where the scheme declines
                to average.
            component: Which E component's chi1inv row the scalar carries: ``"Ex"``,
                ``"Ey"`` or ``"Ez"`` — the polarization the run will drive.
            supersample: Midpoint samples per axis per voxel for the voxel means — a
                scalar or a per-axis ``(sx, sy, sz)`` triple. 1 on an axis the
                structure does not vary along is exact and free. NOTE: 1 on
                ``component``'s own axis narrows the invariance check to variation
                BETWEEN cells along it (one sample per voxel is still taken), so a
                feature thinner than one cell along that axis passes unseen — there,
                the triple form is the caller asserting the invariance themselves.

                Under the default ``fill_rule`` this no longer sets the accuracy of a
                planar interface. It sets DETECTION — a voxel whose ``s**3`` samples
                are all equal is declared homogeneous and returned untouched, so a
                feature thinner than ``dx/s`` is invisible — and it is the accuracy of
                the voxels the planar fit declines. Under ``fill_rule="sampled"`` it is
                still the dominant error term, first order in ``1/s``: 16 leaves
                3.4e-2 on the grating below, 256 reaches 2.1e-3.
            variation_tolerance: Largest relative variation of epsilon along
                ``component``'s own axis to accept. The default 0.0 is exact
                invariance, which any callable or float64 array that simply ignores
                that coordinate satisfies bit-for-bit; pass a small tolerance only
                for an array whose builder carries numerical noise along that axis.
            fill_rule: ``"planar"`` takes each voxel's fill fraction and interface
                normal from the local interface PLANE — located by bisection and
                integrated in closed form — which is what MEEP does analytically for a
                geometric object and removes the ``1/s`` quantization outright. On the
                grating below it takes the worst frequency from 3.4e-2 (sampled,
                ``s`` = 16) to 1.2e-3, and the per-voxel material from 3.5e-2 to
                1.4e-15. ``"sampled"`` is the midpoint sums alone. ``"auto"``, the
                default, is ``"planar"`` for a callable epsilon and ``"sampled"`` for
                an array — an array has nothing between its fine samples to bisect, and
                asking for ``"planar"`` with one raises rather than downgrading
                quietly. CHECK ``planar_fraction`` on the result: it is the share of
                averaged voxels that got the analytic answer, and the rest are back on
                ``supersample``.

        Returns:
            The :class:`~.smoothing.SmoothedEpsilon` that was installed — its
            ``interface_fraction``, ``planar_fraction``, ``smoothed`` mask, ``normal``
            and full tensor ``row`` say what the smoothing did. CHECK
            ``unresolved_fraction``: it is the fraction of voxels where the fine
            lattice found two materials and MEEP's 8-point shortcut declined to
            average anyway, which is the one way this can be wrong rather than merely
            coarse (a bar corner measured 53 % off MEEP's value there).
            ``offdiagonal_magnitude`` cannot warn — the invariance this method
            requires forces it to round-off.

        Raises:
            ValueError: epsilon varying along ``component``'s own axis beyond
                ``variation_tolerance``; a mirror-folded grid (the stored quadrant is
                not a period, so the voxel means would wrap onto the wrong material);
                and every input defect :func:`.smoothing.smooth_inverse_epsilon`
                refuses — unknown component, bad supersample or fill rule, mis-shaped
                or mis-sampled array, ``fill_rule="planar"`` with an array epsilon,
                non-finite or non-positive epsilon.

        MEEP does not smooth susceptibility sigma, the D conductivity, or mu, and
        neither does this engine: a dispersive material is smoothed in its
        INSTANTANEOUS eps_infinity alone (install it here, then add the terms with
        :meth:`add_susceptibility` and point-sampled sigma), which is the first
        limitation MEEP's own Subpixel_Smoothing.md states. Smoothing can only move
        eps between the sampled values, so the CFL bound checked when a
        susceptibility is added cannot be tightened by re-smoothing afterwards.
        """
        self._require_open()
        variation_bound = float(variation_tolerance)
        if not math.isfinite(variation_bound) or variation_bound < 0.0:
            raise ValueError(
                f"variation_tolerance must be a finite non-negative relative bound, got "
                f"{variation_tolerance!r}."
            )
        variation = epsilon_variation_along(self.grid, epsilon, component, supersample)
        if variation > variation_bound:
            axis_name = component[1]  # "Ex" -> "x": the component's own axis.
            raise ValueError(
                f"set_epsilon_smoothed(component={component!r}) requires epsilon to be invariant "
                f"along the {axis_name} axis, but it varies by {variation:.3e} (relative, "
                f"tolerance {variation_bound:g}) on the sampled lattice. The engine's single "
                f"scalar inv_eps can carry {component}'s smoothed tensor row exactly only for a "
                f"{axis_name}-invariant structure driven in the {component} polarization; "
                f"anything else needs the off-diagonal entries this engine cannot store. Pick "
                f"the component that lies along the structure's invariant axis, or run CPU MEEP "
                f"for a fully 3-D-varying material."
            )
        smoothed = smooth_inverse_epsilon(
            self.grid, epsilon, component, supersample, fill_rule=fill_rule
        )
        installed_epsilon = self.xp.ascontiguousarray(smoothed.epsilon)
        installed_inverse = self.xp.ascontiguousarray(smoothed.inverse_epsilon)
        self.fields.set_isotropic_epsilon_volume(installed_epsilon, installed_inverse)
        self.invalidate_fast_path()  # The smoothing route replaces the material arrays too.
        return smoothed

    def add_susceptibility(self, susceptibility: Susceptibility, sigma: Any):  # Add one Lorentz/Drude term.
        """Add one Lorentz or Drude susceptibility to the material.

        This is ``mp.Medium(E_susceptibilities=[...])`` /
        ``meep::structure::add_susceptibility``. :meth:`set_epsilon` supplies the
        INSTANTANEOUS permittivity eps_infinity and this adds one term of MEEP's
        sum_n, so the full model is::

            eps(omega) = eps_inf + sum_n sigma_n * chi_n(omega)

        Order between terms is free, but every term must be added before the first
        step: the recurrence carries a two-step history and a coefficient table fixed
        by dt, and adding a term to a run already in flight would start it from a
        history it never lived through.

        ``sigma`` takes a scalar (uniform — no array, exactly registration-free), one
        volume replicated to all three components at the same integer registration
        ``set_epsilon`` documents, or a ``{'Ex':..., 'Ey':..., 'Ez':...}`` mapping
        sampled at each component's own Yee position, which is the MEEP-exact form
        (MEEP never subpixel-averages sigma: doc/docs/Materials.md:60,
        anisotropic_averaging.cpp:333-345 point-samples it while :252-257 averages
        chi1inv). Build the volumes with :func:`~.dispersion.sample_region` rather
        than re-deriving the offsets.

        Refused, each with the configuration in the message: a term whose discrete
        poles leave the unit circle (:meth:`~.dispersion.Susceptibility.require_stable`);
        a Courant factor too large for this eps_infinity; and any call after stepping
        has begun or after :meth:`close`.

        A susceptibility overlapping a PML is NOT refused. doc/docs/Materials.md:79
        warns that the combination "may produce instabilities" and recommends an
        absorber, which this engine does not have — but refusing outright would block
        the cases the feature exists for (measuring n(omega) through a PML-terminated
        slab; a Drude half-space backed by an absorber), and the combination is
        cross-validated against CPU MEEP at the recorded PML floor. The runtime
        divergence guard in :meth:`run` is what stands behind it.
        """
        self._require_open()
        if not isinstance(susceptibility, Susceptibility):
            raise ValueError(
                f"add_susceptibility takes a dispersion.Susceptibility, got "
                f"{type(susceptibility).__name__}. Noisy, gyrotropic and multilevel-atom kinds are "
                f"not implemented on this engine; those jobs belong on CPU MEEP."
            )
        if self.step_count > 0:
            raise RuntimeError(
                f"add_susceptibility must be called before the first step; this run has already "
                f"taken {self.step_count}. The polarization carries a two-step history and its "
                f"coefficients are fixed by dt, so a term added mid-run would begin from a history "
                f"it never lived through. Call reset() first, or build a new driver."
            )
        susceptibility.require_stable(self.grid.dt)
        self._require_courant_supports_epsilon(susceptibility)
        state = PolarizationState(susceptibility, sigma, self.grid, self.fields._field_dtype())
        if state.driven():
            # A term whose sigma is identically zero drives nothing, so it needs no
            # stored E — and must not switch the mode on, or get_E would start
            # returning an array update_E has no reason to write. MEEP's needs_P is
            # false for a trivial sigma for the same reason. This is what makes the
            # zero-strength case byte-identical rather than merely close.
            self.fields.enable_field_storage()
        self.fields.polarizations.append(state)
        self._susceptibilities.append((susceptibility, sigma))
        self.invalidate_fast_path()  # Dispersion changes update_E's semantics and the coverage.
        return state

    def _require_courant_supports_epsilon(self, susceptibility: Susceptibility):  # CFL against eps_infinity.
        """Reject a Courant factor the INSTANTANEOUS index cannot carry.

        doc/docs/Materials.md:73: ``S < n_min/sqrt(#dimensions)``. The fastest
        numerical modes see eps_infinity, not eps(omega), so a Drude metal with
        eps_inf = 1 at Courant 0.5 is fine while the same metal on eps_inf = 0.8 is
        not — 0.5 > sqrt(0.8)/sqrt(3) = 0.516 is the boundary. MEEP checks the same
        quantity and warns (python/simulation.py, "Epsilon < 1 may require adjusting
        the Courant parameter"); this raises, because the resulting instability is a
        smooth growing mode rather than a crash.

        Checked here rather than in ``set_epsilon`` so that no existing non-dispersive
        run changes behaviour: without a susceptibility the engine has never enforced
        it and has no measurement that says it must.
        """
        # PEC cells are excluded, and by name. Their permittivity is MEEP's -1e20
        # sentinel, whose sqrt is not a number at all — but a perfect conductor carries
        # no propagating mode, so it sets no CFL bound either; the bound belongs to the
        # dielectric the wave actually travels in. A cell holding both a metal and a
        # dispersive medium reaches here (metal-cavity-ldos.py does not, having no
        # susceptibility), and without this it would raise out of math.sqrt.
        minima = []
        for component in E_COMPONENTS:
            values = self.fields.epsilon_for(component)
            dielectric = values[~is_pec_epsilon(values)]
            if dielectric.size:
                minima.append(float(self.xp.min(dielectric)))
        if not minima:
            return  # Every cell is a perfect conductor: nothing propagates, nothing to bound.
        minimum_eps = min(minima)
        limit = math.sqrt(minimum_eps) / math.sqrt(3.0)
        if self.grid.courant <= limit:
            return
        raise ValueError(
            f"courant={self.grid.courant:g} exceeds the CFL limit sqrt(min eps_inf)/sqrt(3) = "
            f"{limit:g} for this run's instantaneous permittivity (min eps_inf = {minimum_eps:g} "
            f"over the cells that are not perfect conductors), "
            f"so adding the {susceptibility.kind} susceptibility f0={susceptibility.frequency:g} "
            f"would step an unstable grid. eps here is eps_INFINITY, not eps(omega) — the fastest "
            f"numerical modes see the instantaneous response. Lower the Courant factor or raise "
            f"eps_inf."
        )

    def _require_existing_sources_clear_of_conductive_pml(self) -> None:
        """Validate all current sources before enabling conductivity inside a PML."""
        if self.pml is None:
            return
        for source in self._sources:
            self._require_source_clear_of_conductive_pml(
                source.center, source.size, self.pml.thickness_by_face
            )

    def set_conductivity(self, d_conductivity: Any, absorber_included: bool = False):  # Install MEEP's frequency-independent D conductivity.
        """Set MEEP's ``D_conductivity``: narrowband loss without an auxiliary equation.

        This is ``mp.Medium(D_conductivity=...)``. It multiplies the WHOLE
        permittivity by ``1 + i*sigma_D/omega`` (python/geom.py ``_get_epsmu``), i.e.
        ``Im eps = eps_inf * sigma_D / omega``, so to hit a known complex index at one
        frequency, ``sigma_D = omega * Im[eps] / Re[eps]``. MEEP's worked example:
        eps = 3.4 + 0.101i at f = 0.42 is ``D_conductivity = 2*pi*0.42*0.101/3.4``.

        It is deliberately frequency-DEPENDENT in that form. A constant Im eps is
        unconditionally unstable — every time-domain run contains both signs of
        omega, so one of them would be gain — which is why MEEP offers this shape and
        not a constant. Conductivities are absorptive at every frequency, cost one
        float32 volume and no auxiliary state, and have no stability question at all.

        Accepts a scalar, a volume in this grid's shape or MEEP's ``(N+1)`` padded
        shape (trimmed as ``set_epsilon`` trims), or a ``{'Dx': ..., 'Dy': ...,
        'Dz': ...}`` mapping of either — the same three spellings ``set_chi2`` and
        ``add_susceptibility`` take. A conductivity that is identically zero, or
        ``None``, removes it entirely rather than installing a volume of zeros, which
        keeps the plain ``step_D`` bit-identical instead of merely equal to within a
        multiply by 1.0.

        THE PER-COMPONENT MAPPING IS MEEP'S OWN SHAPE, not a convenience: ``get_cnd``
        reads ``D_conductivity_diag.x/.y/.z`` for Dx/Dy/Dz respectively
        (meepgeom.cpp:1545-1559), and :meth:`~.fields.Fields._set_conductivity_side`
        has always stored ``sigma``, ``condfac`` and ``condinv`` per component — it is
        what the ``mp.Absorber`` path installs. A scalar still goes in under all three
        keys as ONE array object, so every existing run stays bit-identical.

        The registration is the shared-scalar compromise ``inv_eps`` carries: MEEP
        point-samples sigma_D at each D component's OWN Yee position
        (structure.cpp:868-895, no subpixel averaging), and a VOLUME supplied here is
        sampled once for whichever components it is given to. Uniform conductivity is
        exactly registration-free.

        ``absorber_included`` says the supplied volumes ALREADY carry whatever
        :meth:`set_absorber` would add, so :meth:`_install_conductivity` must not add
        it a second time. There is exactly one caller and one reason:
        :func:`~.from_meep._read_conductivity_volumes` reads sigma_D back off MEEP's
        own ``conductivity[c][d]`` array, and MEEP fills that array with the material
        value and then ADDS each absorber face's ramp into it
        (``geom_epsilon::conductivity``, meepgeom.cpp:1596-1625) — so what comes back
        is the SUM. Measured: a cell whose only loss is ``mp.Absorber(0.5)`` reads a
        peak sigma_D of 98.565 on the D row, against 0 for the identical cell with
        ``mp.PML(0.5)``, whose profile lives in a separate ``sigma`` array the read
        never touches. Installing the read total and suppressing the ramp is strictly
        more faithful than reconstructing it, because the read IS MEEP's own sum. The
        B side is untouched by this flag: no D-row read can see it, so it still comes
        from :meth:`set_absorber`'s own reconstruction.

        With a PML, MEEP's three-stage ``f_cond`` / ``f_u`` / D recurrence is
        reproduced directly from step_generic.cpp. The extra three field histories
        are allocated only for that combination, so conductivity-only and PML-only
        jobs retain their existing storage and arithmetic paths.
        """
        self._require_open()
        if self.step_count > 0:
            raise RuntimeError(
                f"set_conductivity must be called before the first step; this run has already "
                f"taken {self.step_count}. Its coefficients are fixed by dt and D already carries "
                f"the loss history of the steps taken. Call reset() first, or build a new driver."
            )
        if isinstance(d_conductivity, dict):
            unknown = sorted(set(d_conductivity) - set(D_COMPONENTS))
            if unknown:
                raise ValueError(
                    f"D conductivity mapping carries {unknown}; the keys are the D components "
                    f"{list(D_COMPONENTS)}. MEEP reads D_conductivity_diag.x/.y/.z for Dx/Dy/Dz "
                    f"respectively (get_cnd, meepgeom.cpp:1545-1559)."
                )
            per_component = {
                component: self._coerce_conductivity(d_conductivity.get(component))
                for component in D_COMPONENTS
            }
        else:
            shared = self._coerce_conductivity(d_conductivity)
            # ONE array object under all three keys when the conductivity is shared, so
            # a scalar or single-volume run keeps exactly the storage and the arithmetic
            # it had before the per-component spelling existed.
            per_component = {component: shared for component in D_COMPONENTS}
        if any(volume is not None for volume in per_component.values()):
            self._require_existing_sources_clear_of_conductive_pml()
        self._material_d_conductivity = (
            per_component if any(volume is not None for volume in per_component.values())
            else None  # Identically zero everywhere: the plain path, bit-identical.
        )
        self._material_conductivity_includes_absorber = bool(absorber_included)
        self._install_conductivity()
        self.invalidate_fast_path()

    def set_b_conductivity(self, b_conductivity: Any):  # Install MEEP's frequency-independent B conductivity.
        """Set MEEP's ``B_conductivity``: the MAGNETIC half of the same loss term.

        ``mp.Medium(B_conductivity=...)``. Everything :meth:`set_conductivity` says
        about sigma_D holds here one side over, because in MEEP it is literally the
        same code: ``get_cnd`` is ONE switch that answers for Dx/Dy/Dz and Bx/By/Bz
        alike (meepgeom.cpp:1545-1559), ``structure::set_materials`` installs both
        through the same ``FOR_D_AND_B(c) { if (mat.has_conductivity(c)) ... }`` loop
        (structure.cpp:376-378), and ``step_db`` hands ``s->conductivity[cc][d_c]``,
        ``s->condinv[cc][d_c]`` and ``f_cond[cc][cmp]`` to the SAME ``STEP_CURL``
        whichever side ``cc`` belongs to (step_db.cpp:125-127). There is no separate
        magnetic recurrence to transcribe.

        This engine was already symmetric underneath:
        :meth:`~.fields.Fields.set_b_conductivity` and the per-component ``condfac`` /
        ``condinv`` lookup in ``stepping._apply_curl`` have carried the B side since
        ``mp.Absorber`` needed it. What was missing was a public way to install a
        MATERIAL's magnetic loss, which is why a ``mp.Medium(B_conductivity=...)`` was
        refused at the gate while an absorber's identical B-side ramp was not.

        The two sources ADD rather than replace, exactly as on the D side and for the
        same reason: MEEP's ``geom_epsilon::conductivity`` starts from the material's
        ``get_cnd`` value and adds each absorber face's profile onto it
        "isotropically, for both magnetic and electric conductivity"
        (meepgeom.cpp:1596-1625, and the comment is MEEP's own). :meth:`set_absorber`
        and this setter therefore compose in :meth:`_install_conductivity` rather than
        overwriting one another.

        Takes the same three spellings as :meth:`set_conductivity` with the B keys:
        a scalar, one volume, or ``{'Bx': ..., 'By': ..., 'Bz': ...}``. Zero and
        ``None`` remove the loss entirely rather than installing a volume of zeros,
        so a lossless run keeps its bit-identical ``step_B``.

        There is no ``absorber_included`` twin. The flag exists on the D side because
        :func:`~.from_meep._read_conductivity_volumes` reads MEEP's own
        ``conductivity[c][d]`` array back, which already carries the ramp; no such
        read reaches the B row, so a magnetic conductivity always arrives here as the
        material's own declared value.
        """
        self._require_open()
        if self.step_count > 0:
            raise RuntimeError(
                f"set_b_conductivity must be called before the first step; this run has already "
                f"taken {self.step_count}. Its coefficients are fixed by dt and B already carries "
                f"the loss history of the steps taken. Call reset() first, or build a new driver."
            )
        if isinstance(b_conductivity, dict):
            unknown = sorted(set(b_conductivity) - set(B_COMPONENTS))
            if unknown:
                raise ValueError(
                    f"B conductivity mapping carries {unknown}; the keys are the B components "
                    f"{list(B_COMPONENTS)}. MEEP reads B_conductivity_diag.x/.y/.z for Bx/By/Bz "
                    f"respectively (get_cnd, meepgeom.cpp:1545-1559)."
                )
            per_component = {
                component: self._coerce_conductivity(b_conductivity.get(component), "B")
                for component in B_COMPONENTS
            }
        else:
            shared = self._coerce_conductivity(b_conductivity, "B")
            per_component = {component: shared for component in B_COMPONENTS}
        self._material_b_conductivity = (
            per_component if any(volume is not None for volume in per_component.values())
            else None  # Identically zero everywhere: the plain path, bit-identical.
        )
        self._install_conductivity()
        self.invalidate_fast_path()

    def _coerce_conductivity(self, value: Any, side: str = "D"):  # One sigma volume, or None for "no loss here".
        """Validate one conductivity entry and return its float32 volume, or ``None``.

        ``None`` and an identically zero value both return ``None`` rather than a
        volume of zeros, which is what keeps a lossless component's ``step_D``
        bit-identical instead of merely equal to within a multiply by 1.0.

        ``side`` is ``"D"`` or ``"B"`` and only names the quantity in the messages;
        the validation is identical because MEEP's is (one ``get_cnd``, one
        ``STEP_CURL``), and a negative value is gain on either side.
        """
        if value is None:
            return None
        probe = self.xp.asarray(value)
        _reject_complex(
            probe,
            f"{side} conductivity",
            f"MEEP's sigma_{side} is a real, frequency-independent loss rate; a complex value "
            f"usually means an Im(epsilon) was converted with the wrong sign or factor. Supply "
            f"the real rate.",
        )
        values = self.xp.asarray(probe, dtype=self.xp.float64)
        if values.ndim == 0:
            uniform = float(values)
            if not math.isfinite(uniform) or uniform < 0.0:
                raise ValueError(
                    f"{side} conductivity must be finite and >= 0, got {value!r}; a negative "
                    f"conductivity is gain and grows the field while every magnitude still looks "
                    f"reasonable."
                )
            if uniform == 0.0:
                return None
            return self.xp.full(self.grid.shape, uniform, dtype=self.xp.float32)
        if values.ndim != 3:
            raise ValueError(
                f"{side} conductivity must be a scalar or a 3-D array, got {values.ndim} dimensions.")
        grid_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        meep_shape = tuple(n + 1 for n in grid_shape)
        shape = tuple(int(n) for n in values.shape)
        if shape == grid_shape:
            trimmed = values
        elif shape == meep_shape:
            trimmed = values[:-1, :-1, :-1]
        else:
            raise ValueError(
                f"{side} conductivity shape {shape} matches neither the grid shape {grid_shape} nor "
                f"MEEP's boundary-padded shape {meep_shape}."
            )
        if not bool(self.xp.all(self.xp.isfinite(trimmed))) or bool(self.xp.any(trimmed < 0.0)):
            raise ValueError(
                f"{side} conductivity must be finite and >= 0 in every cell; a negative value is gain.")
        if not bool(self.xp.any(trimmed != 0.0)):
            return None
        return trimmed.astype(self.xp.float32)

    def set_absorber(self, layers: Any):  # mp.Absorber: a graded D AND B conductivity on named faces.
        """Install MEEP's ``mp.Absorber`` faces as six graded conductivity volumes.

        ``layers`` is a sequence of :class:`~.absorber.AbsorberLayer` — one per
        (axis, side) face, which is how MEEP stores them too
        (``geom_epsilon::set_cond_profile`` is called once per direction and side,
        meepgeom.cpp:2017-2029). An empty sequence removes the absorber.

        **This installs no boundary condition.** MEEP excludes an ``Absorber`` from
        the boundary region entirely (python/simulation.py:4956), so the face keeps
        whatever wall it had — periodic, metallic or mirrored — and this method
        deliberately does not touch :attr:`pml`. A run may carry both: PML on some
        faces, absorber on others, each per face.

        Six volumes rather than one, and both field types rather than one: see
        :mod:`~.absorber` for the source and the measurements. The absorber's sigma
        ADDS to any material ``D_conductivity`` already set, and either may be set
        first.
        """
        self._require_open()
        if self.step_count > 0:
            raise RuntimeError(
                f"set_absorber must be called before the first step; this run has already taken "
                f"{self.step_count}. Its condfac/condinv are fixed by dt and D and B already "
                f"carry the loss history of the steps taken. Call reset() first, or build a new "
                f"driver."
            )
        resolved = tuple(layers or ())
        for layer in resolved:
            if not isinstance(layer, AbsorberLayer):
                raise TypeError(
                    f"set_absorber takes AbsorberLayer instances, got {type(layer).__name__}; "
                    f"the (axis, side) resolution — including MEEP's High = 0, Low = 1 enum — "
                    f"belongs to the caller so no reader here has to guess it."
                )
            if self.grid.is_invariant(layer.axis):
                raise ValueError(
                    f"{describe_layers([layer])} names the invariant {AXIS_NAMES[layer.axis]} axis, "
                    f"which this run does not resolve. MEEP's own loop is over the grid volume's "
                    f"directions (LOOP_OVER_DIRECTIONS(gv.dim, d), meepgeom.cpp:2019), so it "
                    f"builds no profile there either — drop the layer rather than installing an "
                    f"absorber that grades nothing."
                )
        self._absorber_layers = resolved
        self._absorber_sigma = absorber_conductivity(self.grid, resolved)
        self._install_conductivity()
        self.invalidate_fast_path()

    @property
    def absorber_layers(self) -> tuple[AbsorberLayer, ...]:  # The resolved mp.Absorber faces.
        return self._absorber_layers

    def _install_conductivity(self) -> None:  # Material sigma + absorber sigma -> the field arrays.
        """Compose the two conductivity sources and hand the sum to :class:`~.fields.Fields`.

        MEEP's ``geom_epsilon::conductivity`` starts from the material's own
        ``cond_val`` and ADDS each absorber face's contribution to it
        (meepgeom.cpp:1596-1625), so the two never replace one another. Measured: a
        material function carrying 1x a sigma pattern steps bit-identically to the
        absorber alone, and 2x bit-identically to absorber-plus-1x, with 1x and 2x
        8.05e-06 apart — the two really do add, and the reconstruction that shows it
        is live rather than a silent no-op.

        With only a material sigma and no anisotropy, ONE array object is installed
        under all three D keys — the same array the fields held before this composition
        existed, so a uniform or single-volume conductivity run is bit-identical and
        costs no extra storage. An absorber, or an anisotropic ``D_conductivity_diag``,
        makes the three differ; the composition is per component either way, because
        MEEP's is (``get_cnd`` reads ``D_conductivity_diag.x/.y/.z`` for Dx/Dy/Dz,
        meepgeom.cpp:1545-1559, and each face's ramp is added onto that component's own
        value).

        BOTH SIDES COMPOSE THE SAME WAY, because MEEP's one loop does: the absorber
        ramp is added "isotropically, for both magnetic and electric conductivity"
        (meepgeom.cpp:1596-1600, MEEP's own comment), so a
        ``mp.Medium(B_conductivity=...)`` inside an ``mp.Absorber`` is the SUM of the
        two and not either one of them. The only asymmetry is
        ``_material_conductivity_includes_absorber``, which is a fact about the D-row
        READ (:func:`~.from_meep._read_conductivity_volumes` recovers MEEP's already
        summed array) and has no B-row counterpart, because no read reaches the B row.
        """
        absorber = self._absorber_sigma
        electric_ramp = (
            None  # Already inside the volumes the D-row read recovered; adding it doubles it.
            if self._material_conductivity_includes_absorber and self._material_d_conductivity
            else absorber
        )
        self.fields.set_d_conductivity(self._compose_conductivity_side(
            D_COMPONENTS, self._material_d_conductivity, electric_ramp))
        self.fields.set_b_conductivity(self._compose_conductivity_side(
            B_COMPONENTS, self._material_b_conductivity, absorber))

    def _compose_conductivity_side(self, components, material, absorber):
        """One side's material sigma plus the absorber's ramp, or the material's own object.

        Returns exactly what was passed in when there is no ramp to add — the SAME dict
        object, so the shared-array identity a scalar conductivity relies on survives and
        such a run stays bit-identical rather than merely equal.
        """
        if not absorber:
            return material
        own_volumes = material or {}
        composed: dict[str, Any] = {}
        for component in components:
            ramp = absorber.get(component)
            own = own_volumes.get(component)
            if own is None:
                composed[component] = ramp
            elif ramp is None:
                composed[component] = own
            else:
                # The absorber ramp is float64 so this sum rounds ONCE into float32,
                # as MEEP's single `cnd[i] = C.conductivity(c, here)` does.
                composed[component] = own.astype(self.xp.float64) + ramp
        return composed

    def set_chi2(self, chi2: Any):  # MEEP's mp.Medium(chi2=...): the instantaneous Pockels nonlinearity.
        """Set the instantaneous second-order susceptibility chi(2).

        This is ``mp.Medium(chi2=...)`` / ``structure::set_chi2``. The constitutive
        relation it joins is MEEP's::

            D = eps*E + chi2*E^2 + chi3*|E|^2 E

        with everything on the right INSTANTANEOUS — no auxiliary field, no ODE, no
        memory of previous steps, and no relationship to the dispersion machinery of
        :meth:`add_susceptibility` beyond sharing the constitutive update they both
        land in. It is the second-harmonic / Pockels term: a medium driven at f
        radiates at 2f, which a linear engine cannot produce at all.

        ``chi2`` takes the same three spellings ``sigma`` takes in
        :meth:`add_susceptibility` — a scalar (uniform, allocation-free, exactly
        registration-free), one volume replicated to all three components, or a
        ``{'Ex':..., 'Ey':..., 'Ez':...}`` mapping sampled at each component's own Yee
        position (``dispersion.sample_region``), which is the MEEP-exact form for a
        structured material. MEEP point-samples chi2 at the component's own position
        and never subpixel-averages it (structure.cpp:832-849).

        The sign is free. A negative chi2 is a reversed Pockels coefficient, an
        ordinary material — unlike a negative conductivity or sigma, which are gain
        and are refused.

        Chi2 and chi3 travel together in MEEP, which allocates a zero partner for
        whichever of the two you set (structure.cpp:815-826) because
        ``step_update_EDHB`` branches on one and dereferences both. Setting either
        here installs both, and setting BOTH to identically zero removes the pair, so
        the run steps the linear kernel byte for byte rather than multiplying by a
        Pade factor that happens to evaluate to 1.0.
        """
        self._install_nonlinearity(chi2, self._chi3)

    def set_chi3(self, chi3: Any):  # MEEP's mp.Medium(chi3=...): the instantaneous Kerr nonlinearity.
        """Set the instantaneous third-order (Kerr) susceptibility chi(3).

        This is ``mp.Medium(chi3=...)`` / ``structure::set_chi3``, the term behind
        self-phase modulation, self-focusing, third-harmonic generation and
        four-wave mixing. See :meth:`set_chi2` for the constitutive relation, the
        accepted spellings, the sign convention and the both-or-neither rule; the
        only difference is which power of E the term carries.

        MEEP's own conversion (doc/docs/Units_and_Nonlinearity.md), for reading a
        value out of the literature or checking one against a measurement: the AC
        Kerr coefficient is ``n2 = 3*chi3 / (4*n0^2)`` with ``n0 = sqrt(eps)``, and
        the effective index of a plane wave of amplitude ``E0`` shifts by
        ``dn = 3*chi3*E0^2 / (8*n0)``. The literature carries several conflicting
        conventions for both quantities; that pair is the one this engine and MEEP
        implement.

        A chi3 large enough — or a field strong enough — to leave the Pade
        approximant's valid range is refused mid-run rather than returned, because
        past its pole the recovered E is large, smooth, finite and backwards. See
        :meth:`nonlinear_margin`.
        """
        self._install_nonlinearity(self._chi2, chi3)

    def _install_nonlinearity(self, chi2: Any, chi3: Any) -> None:
        """Normalize and install the chi2/chi3 pair, switching stored E on if it is live.

        Refused once stepping has begun, for the same reason ``add_susceptibility`` is:
        installing a nonlinearity switches E from computed-on-demand to STORED, and the
        array it allocates is zeros. Between that call and the next ``update_E`` every
        ``get_field`` would return an empty volume for a run that is carrying a field —
        which reads as a run that produced nothing rather than as a mode switch.
        """
        self._require_open()
        second = self._normalize_nonlinear(chi2, "chi2")
        third = self._normalize_nonlinear(chi3, "chi3")
        if self.step_count > 0 and (self.fields.has_nonlinearity or any(
                not self.fields.is_trivial_chi(value)
                for mapping in (second, third) for value in mapping.values())):
            raise RuntimeError(
                f"set_chi2/set_chi3 must be called before the first step; this run has already "
                f"taken {self.step_count}. Installing a nonlinearity switches E into storage, and "
                f"the freshly allocated array is zeros until the next update_E — so every readback "
                f"in between would report an empty field for a run that is carrying one. Call "
                f"reset() first, or build a new driver."
            )
        was_nonlinear = self.fields.has_nonlinearity
        self.fields.set_nonlinear_volumes(second, third)
        self.invalidate_fast_path()  # chi2/chi3 volumes swapped; nonlinearity changes coverage.
        self._chi2 = chi2
        self._chi3 = chi3
        if self.fields.has_nonlinearity:
            # E is no longer D*inv_eps under a nonlinearity, so it MUST be stored:
            # get_E would otherwise recompute the linear field and quietly delete every
            # harmonic the run exists to produce. Same obligation a susceptibility has
            # (MEEP update_eh.cpp:166-171 allocates f[ec] whenever chi1inv exists, and
            # set_chi2/set_chi3 force chi1inv into existence: structure.cpp:800-804).
            self.fields.enable_field_storage()
        if self.fields.has_nonlinearity != was_nonlinear:
            # A nonlinearity flips _automatic_decimation_factor's resolution
            # (dft.cpp:207-210 resolves one on a nonlinear run), so monitors that
            # asked for decimation_factor=0 must be re-resolved. MEEP never faces
            # this ordering — its dft objects are created on `fields`, after the
            # structure's chi2/chi3 already exist, so add_dft always sees the
            # final nonlinearity; here chi may arrive after the monitor. The
            # refresh is always legal: installation is refused once stepping has
            # begun (above), and no monitor accumulates before the first step.
            self._refresh_automatic_decimation()

    def _normalize_nonlinear(self, value: Any, label: str) -> dict[str, Any]:
        """Resolve a scalar / one volume / per-component mapping into a per-component dict.

        Deliberately the same three spellings ``dispersion.normalize_sigma`` accepts,
        so a caller who has built sigma volumes with ``sample_region`` can build chi2
        and chi3 the identical way. Validation differs in exactly one respect: sign is
        unconstrained here (see :meth:`set_chi2`).
        """
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - set(E_COMPONENTS))
            if unknown:
                raise ValueError(
                    f"{label} keys must be electric components {E_COMPONENTS}, got unexpected "
                    f"{unknown}. Magnetic (H-side) nonlinearity is not implemented on this engine; "
                    f"that job belongs on CPU MEEP."
                )
            return {name: self._coerce_nonlinear(value.get(name, 0.0), f"{label}[{name!r}]")
                    for name in E_COMPONENTS}
        shared = self._coerce_nonlinear(value, label)
        return {name: shared for name in E_COMPONENTS}

    def _coerce_nonlinear(self, value: Any, label: str) -> Any:
        """Validate one chi2/chi3 entry and trim a volume to the stored grid."""
        probe = self.xp.asarray(value)
        _reject_complex(
            probe,
            label,
            "MEEP's chi2 and chi3 are real, instantaneous susceptibilities; a complex value usually "
            "means a frequency-domain nonlinear coefficient was passed through unconverted. Supply "
            "the real instantaneous value.",
        )
        if probe.ndim == 0:
            number = float(probe)
            if not math.isfinite(number):
                raise ValueError(f"{label} must be finite, got {value!r}.")
            return number
        values = self.xp.array(probe, dtype=self.xp.float32, order="C", copy=True)
        if values.ndim != 3:
            raise ValueError(f"{label} must be a scalar or a 3-D array, got {values.ndim} dimensions.")
        grid_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        meep_shape = tuple(n + 1 for n in grid_shape)
        shape = tuple(int(n) for n in values.shape)
        if shape == grid_shape:
            trimmed = values
        elif shape == meep_shape:
            trimmed = values[:-1, :-1, :-1]
        else:
            raise ValueError(
                f"{label} shape {shape} matches neither the grid shape {grid_shape} nor MEEP's "
                f"boundary-padded shape {meep_shape}."
            )
        if not bool(self.xp.all(self.xp.isfinite(trimmed))):
            raise ValueError(f"{label} must be finite in every cell.")
        return trimmed

    def nonlinear_margin(self) -> NonlinearMargin | None:
        """How far this run's current field is from the Pade approximant's pole.

        None when the run carries no chi2/chi3. Otherwise a
        :class:`~.stepping.NonlinearMargin`: the smallest ``1 + 2*c2 + 3*c3`` on the
        grid, the largest ``|c2| + |c3|``, and which component produced the latter.

        The bound to read it against is ``expansion < 1/3``, which is not a tuned
        constant: ``1 + 2*c2 + 3*c3 >= 1 - 3*(|c2| + |c3|)``, so anything below 1/3
        keeps the denominator strictly positive and the run on the linear side of the
        pole. :meth:`run` enforces exactly that on the same sampling interval the
        divergence guard uses, and raises :class:`FdtdNonlinearityOutOfRange` rather
        than returning the large, smooth, sign-flipped field that lies past it.
        """
        self._require_open()
        return nonlinear_margin(self.fields, self.pml)

    def setup_pml(self, thickness_cells: Any, order: int = 2):  # Build UPML coefficients and switch fields into PML mode.
        """Enable a PML, uniform on all six faces or specified per axis and per side.

        The thickness is in cells, not length units — the bridge layer converts
        (``thickness_um * resolution``) before calling. It need not be a WHOLE number
        of cells: MEEP's ``mp.PML(0.5)`` at resolution 71 is 35.5 cells, and MEEP
        builds exactly that, snapping only the layer's extent to the nearest half cell
        (:func:`~.pml.half_cell_extent`). Rounding such a request to a whole cell
        builds a different absorber, measurably: a 5.5-cell one-sided layer reproduces
        CPU MEEP at **5.63e-07** while the 5- and 6-cell layers sit 3.26e-02 and
        2.89e-02 from that same reference, and at resolution 71 a 35.5-cell layer
        reproduces at 1.16e-06 against 8.75e-04 / 9.27e-04 for its two neighbours.
        A scalar covers every
        face, as it always has; :func:`~.pml.normalize_pml_thickness` documents the
        per-axis and per-side spellings, of which the one this feature exists for is
        ``{"z": 8}``: absorb along the propagation axis while x and y stay periodic,
        which is what makes a Bloch-periodic grating or photonic crystal a legal run.
        ``{"z": (8, 0)}`` absorbs at the low z face only, MEEP's
        ``mp.PML(t, direction=mp.Z, side=mp.Low)``.

        What a one-sided layer does NOT do is terminate the opposite face. That face
        still wraps — MEEP's does too — so a wave leaving it re-enters the cell at the
        absorber's OUTER edge, where sigma is at its maximum. That is a legal MEEP
        configuration and this engine now reproduces it: **5.5e-07** against CPU MEEP
        with a live wave on the un-absorbed face, where it used to read 1.2e-01. The
        difference was not the physics of entering a graded layer backwards, which
        both codes share; it was that the engine graded the wrapped lattice point from
        the wrong wall (:meth:`~.pml.PML._graded_sigma`), and that plane is precisely
        where the wrap re-enters. Every per-side arrangement now lands in the same
        3e-07…8e-07 band as the two-sided ones — one-sided with the far face quiet
        2.9e-07, asymmetric (10, 4) 4.9e-07, uniform 4.0e-07, z-only 4.5e-07
        in the package's CPU-MEEP parity suite.

        Field storage is switched to PML mode immediately (E/H plus the twelve
        auxiliaries), never mid-run: ``enable_pml_storage`` changes what ``get_E`` /
        ``get_H`` mean, so flipping it after stepping has begun would silently change
        physics. Calling this after the first step therefore raises.

        The thickness is validated per axis against the faces that axis actually
        carries — the lower face of a mirror-folded X/Y axis is the mirror plane, not
        an absorber, and holds nothing — matching :meth:`PML.__post_init__`.
        Validating with ``min(nx, ny, nz)`` and two faces everywhere, as this method
        used to, refuses legal configurations (an 8-cell layer on a folded 16-cell
        axis holds comfortably, and so does a z-only layer on a 4-cell-wide x axis).

        Sources already added are re-checked against the new absorber here, so the
        overlap guard of :meth:`add_source` holds whichever order the two are called in.

        Bloch and PML now coexist, but only where they do not contradict each other:
        a nonzero ``k_point`` component is refused on an axis that carries an absorber
        on either face, because the layer terminates that axis instead of repeating it
        and the phase would describe a lattice vector the run does not have. Axes with
        no absorber keep their (Bloch-)periodic wrap, so ``k_point=(0.3, 0, 0)`` with
        ``{"z": 8}`` is the supported grating configuration and ``k_point=(0.3, 0, 0)``
        with a scalar PML still raises.

        A request that absorbs nowhere — a scalar 0, or a per-face table of zeros —
        raises rather than installing an inert layer: the caller asked for an absorber
        and would have got a fully periodic run whose boundaries look absorbing in the
        source. (A :class:`PML` object built directly with zero thickness is a
        different matter: it is simply inert, and steps bit-identically to no PML.)
        """
        self._require_open()
        if self.step_count > 0:
            raise RuntimeError(
                "setup_pml() must be called before stepping: enabling PML storage mid-run changes "
                "the field-access semantics and would discard the E/H history."
            )
        request = normalize_pml_thickness(thickness_cells)
        faces = request.faces
        # The test is on the SNAPPED extent, not the raw request. MEEP quantizes a
        # layer's depth to whole half-cells ((int)(dx*2a + 0.5), structure.cpp:625-628),
        # so anything under a quarter of a cell grades ZERO samples and absorbs
        # nowhere — while a raw `low > 0` test waves it through. Measured on the
        # request this guard was extended for: setup_pml(0.2) used to return with
        # is_active True, a repr quoting prefac=1295.20, and sigma identically zero
        # on every axis, stepping 3.293e-07 from the same run with no PML at all —
        # an absorber that absorbs nowhere, wearing the clothes of one that does.
        # `_lift_pml` already dropped exactly this case; enforcing it on one entry
        # point out of three is what let it through here.
        if not any(half_cell_extent(low) > 0 or half_cell_extent(high) > 0
                   for low, high in faces):
            deepest = max(max(low, high) for low, high in faces)
            raise ValueError(
                f"PML thickness {thickness_cells!r} absorbs on no face at all: the deepest face "
                f"asks for {deepest:g} cells, which MEEP snaps to zero half-cells "
                f"((int)({deepest:g}*2 + 0.5) == 0), so every graded sample would be zero. A PML "
                f"that absorbs nowhere is a fully periodic run; omit setup_pml() rather than "
                f"installing one. The smallest layer that grades anything is 0.25 cells."
            )
        for axis in range(3):
            low, high = faces[axis]
            if (low > 0 or high > 0) and self.grid.is_invariant(axis):
                raise ValueError(
                    f"the PML puts (low={low}, high={high}) cells on the {AXIS_NAMES[axis]} axis, "
                    f"which dimensions={self.grid.dimensions} makes translationally invariant. An "
                    f"invariant axis has no face to absorb at — it is infinite and uniform, and "
                    f"the one cell that stands for it would be entirely inside the absorber. MEEP "
                    f"drops such a layer silently (an mp.PML(direction=mp.Z) on a 2-D cell steps "
                    f"0.0e+00 away from the same run without it, measured), so nothing is lost by "
                    f"putting the layer on the axes this run resolves "
                    f"({', '.join(AXIS_NAMES[a] for a in range(3) if not self.grid.is_invariant(a))})."
                )
        for axis in range(3):
            # Validated against MEEP's OWNED window, not the stored count: a folded
            # periodic axis stores one cell past the wall (the big_corner plane
            # and its ghost slot), and a layer that covers the
            # whole owned window has no interior however many ghost rows sit past it.
            owned_cells = self.grid.owned_cells(axis)
            low, high = faces[axis]
            folded = self.grid.is_mirrored(axis)
            # Counted in whole cells actually graded, so the message says what it means
            # for a fractional face too; absorbing_cells is the identity on whole ones.
            covered = absorbing_cells(high) + (0 if folded else absorbing_cells(low))
            if covered >= owned_cells:
                mirror_note = " (its lower face is the mirror plane and carries no PML)" if folded else ""
                raise ValueError(
                    f"PML thickness (low={low}, high={high}) leaves no interior on the "
                    f"{AXIS_NAMES[axis]} axis of a {self.grid.shape} grid{mirror_note}: the "
                    f"absorbing face(s) would cover {covered} of its {owned_cells} owned cells."
                )
        # No monitor-ordering check is needed here. A monitor's wrap flags come from
        # :func:`_periodic_axes`, which reads the symmetry alone — an absorber removes no
        # wrap (it is a material, not a boundary condition), so installing one cannot
        # invalidate a plane that was registered before it. The check that used to stand
        # here refused a legal ordering whenever the layer happened to cover the axis a
        # monitor spanned.
        order_value = int(order)
        if order_value != order or order_value < 1:
            raise ValueError(f"PML order must be a positive whole number, got {order!r}.")
        pml = PML(self.grid, thickness=thickness_cells, order=order_value)
        if self.fields.has_conductivity:
            for source in self._sources:
                self._require_source_clear_of_conductive_pml(
                    source.center, source.size, pml.thickness_by_face
                )
        self.pml = pml
        self.fields.enable_pml_storage()
        for source in self._sources:
            # Which deposition cells sit in MEEP's unsplit edge chunks changes with
            # the layer, so every source re-derives its f_u mirror from this one.
            source.bind_pml(pml)
        self.invalidate_fast_path()  # PML tables and storage mode both enter the plan.

    def _conductivity_reaches_a_pml(self, faces) -> bool:
        """Is any installed D or B conductivity nonzero ANYWHERE inside an absorbing face?

        The predicate :meth:`_require_source_clear_of_conductive_pml` is really about.
        Its defect needs a cell where the conductive ``f_cond`` stage and the PML's
        split-field recurrence are BOTH live; where sigma is zero throughout every
        layer, no such cell exists and the source's position in the layer cannot
        matter.

        MEASURED (``component_coordinates`` is not used here; see the index note
        below), on a source spanning the full transverse extent so that it runs
        straight through both y PML layers — test_damping's own source shape — against
        pristine CPU MEEP stepping the same cell:

        =========================================  ==========
        cell                                       rel-L2
        =========================================  ==========
        lossless control                           3.3453e-07
        sigma = 0.7 in a central block only        2.5964e-07
        sigma = 0.7 everywhere, PML included       1.6419e-02
        =========================================  ==========

        Five orders between the middle row and the last, and the middle row is AT the
        lossless control's floor. So the geometric guard was refusing a case it has no
        evidence against, and the last row is the one it exists for.

        Deliberately a WHOLE-REGION test rather than a source-footprint one: the
        footprint version would need the source box folded through every mirror plane
        to be sound, and this question does not need that precision.

        Written on INDEX ranges rather than on coordinates, and PADDED by one cell on
        each side. ``component_coordinates``' own docstring names the half-cell slip
        that a ``-L/2`` comparison introduces on an odd cell count; a question whose
        answer is "is there any loss in this layer" does not need that resolution, and
        the pad makes the mask a superset of the layer. A mirrored axis likewise takes
        the larger of its two face thicknesses. Both choices can only OVER-report, so
        the refusal can only become more conservative, never less.
        """
        xp = self.xp
        counts = (self.grid.nx, self.grid.ny, self.grid.nz)
        # Both sides, because the defect is the injection-into-a-conductive-cell one and
        # a magnetic current lands in B exactly as an electric one lands in D. It changes
        # nothing for an mp.Absorber (its ramp is nonzero on both sides in the same cells,
        # so the D loop already answered yes) and bites only for a material's own
        # B_conductivity carried into a layer with no electric loss at all.
        for d_component in D_COMPONENTS + B_COMPONENTS:
            sigma = self.fields.conductivity_for(d_component)
            if sigma is None:
                continue
            inside = None
            for axis in range(3):
                low_cells, high_cells = faces[axis]
                if self.grid.is_mirrored(axis):
                    low_cells = high_cells = max(low_cells, high_cells)
                count = counts[axis]
                axis_mask = xp.zeros(count, dtype=bool)
                if low_cells:
                    axis_mask[:min(int(math.ceil(low_cells)) + 1, count)] = True
                if high_cells:
                    axis_mask[max(count - int(math.ceil(high_cells)) - 1, 0):] = True
                shape = [1, 1, 1]
                shape[axis] = -1
                axis_mask = axis_mask.reshape(tuple(shape))
                inside = axis_mask if inside is None else (inside | axis_mask)
            if inside is not None and bool(xp.any(inside & (sigma != 0.0))):
                return True
        return False

    def _require_source_clear_of_conductive_pml(self, center, size, faces) -> None:
        """Keep a source two cells inside a PML when D conductivity is LIVE THERE.

        MEEP injects the current into D after the conductive ``f_cond`` history has
        advanced. The array implementation matches that recurrence to 3e-7 when the
        source stencil is at least two cells inside the layer edge, but a sheet that
        touches the edge differs by 3.7e-2 because MEEP's internal chunk boundary
        owns the zero-sigma edge differently. Refuse the unvalidated overlap instead
        of returning that smooth, plausible wrong field.

        The overlap is only unvalidated when the conductivity is actually nonzero
        inside a layer — see :meth:`_conductivity_reaches_a_pml` for the measurement
        that separates the two cases by five orders. A per-point conductivity confined
        to the interior of the cell, which is what every MaterialGrid ``damping`` and
        every lossy inclusion is, leaves the layers lossless and takes no refusal.
        """
        if not self._conductivity_reaches_a_pml(faces):
            return
        margin = _CONDUCTIVE_PML_SOURCE_CLEARANCE_CELLS * self.grid.dx
        lengths = (self.grid.Lx, self.grid.Ly, self.grid.Lz)
        folded = tuple(self.grid.is_mirrored(axis) for axis in range(3))
        violations: list[tuple[float, int, str, float, float]] = []
        for axis in range(3):
            low_cells, high_cells = faces[axis]
            if folded[axis]:
                low_cells = high_cells
            low = center[axis] - size[axis] / 2.0
            high = center[axis] + size[axis] / 2.0
            if low_cells:
                limit = -lengths[axis] / 2.0 + low_cells * self.grid.dx + margin
                violations.append((limit - low, axis, "low", low, limit))
            if high_cells:
                limit = lengths[axis] / 2.0 - high_cells * self.grid.dx - margin
                violations.append((high - limit, axis, "high", high, limit))
        offending = [entry for entry in violations if entry[0] > 1e-12]
        if not offending:
            return
        amount, axis, side, endpoint, limit = max(offending)
        raise ValueError(
            f"A source in a run combining D conductivity with PML must stay at least "
            f"{_CONDUCTIVE_PML_SOURCE_CLEARANCE_CELLS:g} grid cells ({margin:g} length units) "
            f"inside every absorbing face. Its {AXIS_NAMES[axis]}-{side} endpoint {endpoint:g} "
            f"crosses the validated limit {limit:g} by {amount:g}. The f_cond recurrence matches "
            f"CPU MEEP to 3e-7 beyond this margin; a source touching the layer edge differs by "
            f"3.7e-2. Move or shrink the source."
        )

    def add_source(self, source_data: Mapping[str, Any]):  # Create one current source from the boundary-protocol dict.
        """Add a source described by the backend boundary protocol.

        Keys: ``component`` (any of :data:`SOURCE_COMPONENTS`), ``center``, ``size``,
        ``amplitude``, ``amp_func`` ((x, y, z) -> complex, extended sources only),
        ``is_integrated``, and ``source_type`` ('continuous' — the default —
        'gaussian', or 'custom'). Continuous and gaussian sources need ``frequency``
        — finite and nonzero, either sign: a negative one is MEEP's conjugate
        carrier (sources.cpp:106 rotates ``exp(-i*2*pi*f*t)`` with ``f`` stored raw);
        gaussian ones also need ``fwidth`` and take ``start_time`` / ``cutoff``, and
        continuous ones forward ``start_time`` / ``end_time`` / ``width`` /
        ``slowness`` (MEEP's turn-on ramp) when present. A ``custom`` source instead
        takes ``src_func`` (t -> complex, required) with optional ``start_time`` /
        ``end_time``, and refuses ``frequency``: a caller-supplied waveform carries no
        carrier of its own, so accepting one would mean ignoring it. A zero ``size``
        selects the point-source path, any extent selects the extended-source path.

        ``component`` selects the field family, and with it the injection slot: the
        electric components are injected between ``step_D`` and ``update_E`` at
        ``t + dt/2``, the magnetic ones between ``step_B`` and ``update_H`` at ``t``
        (MEEP step.cpp:64-100). :meth:`step` reads each source's ``field_type`` to
        place it, so the two families cannot be injected in the same slot.

        Magnetic components, ``is_integrated`` and ``custom`` waveforms are served by
        :class:`~.sources.VolumeSource`; the electric non-integrated CW and gaussian
        cases keep their original dedicated classes, whose output they reproduce.

        Unknown keys, components, and source types raise: a mistyped key that is
        silently ignored produces a physically wrong run that still looks successful.
        A source whose extent reaches a metallic wall — the cylindrical r-max
        wall included — is ADMITTED: the deposition drops the ladder ring that
        lands on the wall plane exactly as MEEP's ``zero_metal`` makes its own
        wall deposit inert (the ``metal_high`` transcription in
        ``sources._build_source_points``, measured at the interior controls'
        floor on Dcyl and Cartesian PEC walls alike). A source inside a PML is
        likewise admitted on every grid: the two defects the old geometric
        refusal masked are fixed and measured on Cartesian and Dcyl alike
        (evidence packet §1.5-1.7 and the Dcyl sweep in the ``sources``
        deposit-mirror comment).
        """
        self._require_open()
        if not isinstance(source_data, Mapping):
            raise ValueError(f"add_source() takes a mapping of source parameters, got {type(source_data).__name__}.")
        kind = str(source_data.get("source_type", "continuous"))
        if kind not in SOURCE_KINDS:
            raise ValueError(f"Unknown source_type {kind!r}. Supported kinds: {SOURCE_KINDS}.")
        allowed = _COMMON_SOURCE_KEYS | _SOURCE_KEYS_FOR_KIND[kind]
        unknown = sorted(set(source_data) - allowed)
        if unknown:
            raise ValueError(f"Unknown source keys {unknown} for source_type={kind!r}; allowed: {sorted(allowed)}.")
        if "component" not in source_data:
            raise ValueError("A source needs at least 'component'.")
        component = str(source_data["component"])
        if component not in SOURCE_COMPONENTS:
            raise ValueError(
                f"Unknown source component {component!r}. Supported source components: "
                f"{SOURCE_COMPONENTS} (electric {ELECTRIC_SOURCE_COMPONENTS} drive D, "
                f"magnetic {MAGNETIC_SOURCE_COMPONENTS} drive B)."
            )
        if kind == "custom":
            if source_data.get("src_func") is None:
                raise ValueError("A custom source requires 'src_func', a callable func(t) -> complex.")
            frequency = None
        else:
            if "frequency" not in source_data:
                raise ValueError(f"A {kind} source needs 'frequency'.")
            frequency = _finite_nonzero_frequency(source_data["frequency"])
        is_integrated = bool(source_data.get("is_integrated", False))
        center = _as_xyz("center", source_data.get("center", (0.0, 0.0, 0.0)))
        size = _as_xyz("size", source_data.get("size", (0.0, 0.0, 0.0)))
        if any(extent < 0.0 for extent in size):
            raise ValueError(f"Source size must be non-negative along every axis, got {size}.")
        if self.pml is not None and self.fields.has_conductivity:
            self._require_source_clear_of_conductive_pml(
                center, size, self.pml.thickness_by_face
            )
        amplitude = complex(source_data.get("amplitude", 1.0))
        amp_func = source_data.get("amp_func", None)
        if amp_func is not None and not callable(amp_func):
            raise ValueError("amp_func must be callable (x, y, z) -> complex.")
        # The dedicated electric classes are pinned against MEEP by name and stay the
        # path for the cases they cover; anything they cannot express — a magnetic
        # component, an integrated source, a caller-supplied waveform — is composed
        # from an envelope and handed to VolumeSource, whose spatial machinery is
        # ExtendedSource's.
        needs_volume_source = (
            component in MAGNETIC_SOURCE_COMPONENTS or is_integrated or kind == "custom"
        )
        if needs_volume_source:
            source = self._build_volume_source(
                source_data, kind, component, center, size, amplitude, amp_func, frequency, is_integrated
            )
            self._sources.append(source)
            # THE SOURCE LIST IS A PLAN INPUT. `plan_fast_path` hands it to the
            # fused-pair predicates, which cannot read it off Fields and refuse
            # outright when it is not declared; a pair admitted for one source set
            # and then run against another would compute its constitutive half
            # against a deposit nothing repaired. Adding a source after the freeze
            # is legal, so it re-plans -- the same rule the material mutators obey.
            self.invalidate_fast_path()
            if self.pml is not None:
                source.bind_pml(self.pml)
            self._refresh_automatic_decimation()
            return source
        arguments: dict[str, Any] = {
            "grid": self.grid,
            "frequency": frequency,
            "component": component,
            "center": center,
            "size": size,
            "amplitude": amplitude,
        }
        if kind == "gaussian":
            if source_data.get("fwidth") is None:
                raise ValueError("A gaussian source requires 'fwidth' (frequency width).")
            arguments["fwidth"] = _positive_frequency(source_data["fwidth"])
            arguments["amp_func"] = amp_func
            for key in ("start_time", "cutoff"):
                if key in source_data:
                    arguments[key] = float(source_data[key])
            source = GaussianPulsedSource(**arguments)
        elif all(extent == 0.0 for extent in size):
            if amp_func is not None:
                raise ValueError("amp_func applies to extended sources; a point source has no spatial profile.")
            for key in sorted(_CONTINUOUS_SOURCE_KEYS - {"frequency"}):
                if key in source_data:
                    arguments[key] = float(source_data[key])
            source = ContinuousSource(**arguments)
        else:
            arguments["amp_func"] = amp_func
            for key in sorted(_CONTINUOUS_SOURCE_KEYS - {"frequency"}):
                if key in source_data:
                    arguments[key] = float(source_data[key])
            source = ExtendedSource(**arguments)
        self._sources.append(source)
        self.invalidate_fast_path()  # A plan input; see the note above.
        if self.pml is not None:
            source.bind_pml(self.pml)
        self._refresh_automatic_decimation()
        return source

    def _build_volume_source(
        self, source_data, kind, component, center, size, amplitude, amp_func, frequency, is_integrated
    ) -> VolumeSource:  # Compose the envelope for a magnetic, integrated, or custom-waveform source.
        """Build the general source for the cases the dedicated electric classes cannot express.

        The envelope carries the waveform and ``is_integrated``; ``VolumeSource``
        carries the component, the spatial extent and the injection slot. A point
        source is ``size=(0, 0, 0)`` here as everywhere, which selects MEEP's
        ``loop_in_chunks`` deposition rather than the trilinear point path — the two
        agree, and it is the one cross-validated for magnetic components.
        """
        envelope_arguments: dict[str, Any] = {"is_integrated": is_integrated}

        def forward(*keys):  # Copy the given float keys through to the envelope when present.
            for key in keys:
                if key in source_data:
                    envelope_arguments[key] = float(source_data[key])

        if kind == "custom":
            src_func = source_data["src_func"]
            if not callable(src_func):
                raise ValueError(
                    f"A custom source's 'src_func' must be callable (t) -> complex, got {type(src_func).__name__}."
                )
            forward("start_time", "end_time")
            envelope = CustomEnvelope(func=src_func, **envelope_arguments)
        elif kind == "gaussian":
            if source_data.get("fwidth") is None:
                raise ValueError("A gaussian source requires 'fwidth' (frequency width).")
            forward("start_time", "cutoff")
            envelope = GaussianEnvelope(
                frequency=frequency, fwidth=_positive_frequency(source_data["fwidth"]), **envelope_arguments
            )
        else:
            forward("start_time", "end_time", "width", "slowness")
            envelope = ContinuousEnvelope(frequency=frequency, **envelope_arguments)
        if amp_func is not None and all(extent == 0.0 for extent in size):
            raise ValueError("amp_func applies to extended sources; a point source has no spatial profile.")
        return VolumeSource(
            grid=self.grid,
            component=component,
            center=center,
            size=size,
            envelope=envelope,
            amplitude=amplitude,
            amp_func=amp_func,
        )

    def add_dft_monitor(
        self,
        frequencies: Any = None,
        components: Sequence[str] = ("Ex", "Ey", "Ez"),
        center: Sequence[float] | None = None,
        size: Sequence[float] | None = None,
        *,
        frequency: Any = None,
        fcen: float | None = None,
        df: float | None = None,
        nfreq: int | None = None,
        decimation_factor: int = 1,
    ) -> DFTMonitor:  # Accumulate DFT fields over a region, at one or more frequencies.
        """Add a DFT field monitor at one frequency or a list of them.

        ``frequencies`` takes a single frequency or a sequence of them; ``fcen`` +
        ``df`` + ``nfreq`` is MEEP's broadband spelling and expands through
        :func:`_meep_linspace`. The scalar ``frequency`` keyword is the older
        spelling and means the same as a scalar ``frequencies``.

        With an explicit ``center`` / ``size`` the monitor covers that volume. Without
        one it covers the PML interior when the run has a PML, else the whole domain —
        and that default is resolved when stepping starts, not here, so adding the
        monitor before or after :meth:`setup_pml` gives the same region (port
        reference §5.14; the bundle's order-sensitive defaulting silently produced
        full-domain monitors that included the absorbing layers).

        ``decimation_factor=0`` selects MEEP's automatic safe sampling interval
        from the bandwidth-limited sources and this monitor's highest frequency.
        A positive value explicitly samples every nth global timestep and applies
        MEEP's matching ``n*dt`` quadrature weight. This package retains its
        established explicit default of one for backward compatibility; callers
        seeking MEEP's high-level default should pass zero. Any number of monitors
        may be added; each owns its region and accumulators and is considered at
        the same point in every step.
        """
        self._require_open()
        monitor_frequencies = _normalize_frequencies(frequencies, frequency, fcen, df, nfreq)
        component_names = tuple(str(name) for name in components)
        if not component_names:
            raise ValueError("A DFT monitor needs at least one field component.")
        unknown = [name for name in component_names if name not in MONITOR_COMPONENTS]
        if unknown:
            raise ValueError(f"Unknown DFT monitor components {unknown}. Supported: {MONITOR_COMPONENTS}.")
        if (center is None) != (size is None):
            raise ValueError("A DFT monitor region needs both 'center' and 'size', or neither.")
        region_center = _as_xyz("center", center) if center is not None else None
        region_size = _as_xyz("size", size) if size is not None else None
        if region_size is not None and any(extent < 0.0 for extent in region_size):
            raise ValueError(f"Monitor size must be non-negative along every axis, got {region_size}.")
        if region_size is not None:
            self._require_region_flat_on_invariant_axes(region_size, "DFT monitor")
        request = _decimation_request(decimation_factor)
        resolved_decimation = (
            self._automatic_decimation_factor(monitor_frequencies)
            if request == 0
            else request
        )
        monitor = DFTMonitor(self.grid, frequencies=monitor_frequencies, components=component_names,
                             periodic=self._monitor_periodic_axes(),
                             decimation_factor=resolved_decimation)
        self._dft_monitors.append(monitor)
        if request == 0:
            self._automatic_decimation_monitors.append(monitor)
        self._pending_monitor_regions.append((monitor, region_center, region_size))
        return monitor

    def add_yee_region_dft(
        self,
        component: str,
        bounds: Any,
        frequencies: Any = None,
        decimation_factor: int = 0,
    ) -> "YeeRegionDFT":
        """Accumulate the raw-Yee DFT of one component over one region.

        This is the near-to-far chunk (`use_centered_grid=false` in MEEP's
        `add_dft_near2far`), not a general-purpose monitor: no cell-centre
        interpolation, no unfolding, and the region must sit inside the stored grid.
        `migrate_monitors` is the caller; the accumulated chunk is read back through
        :meth:`YeeRegionDFT.packed` in MEEP's own stored layout so it can be loaded
        into a CPU-MEEP ``DftNear2Far`` for far-field evaluation.

        ``decimation_factor=0`` resolves MEEP's automatic source+monitor bandwidth
        rule, exactly as :meth:`add_dft_monitor` does — the near2far data carries a
        ``dt*decimation`` quadrature weight, so matching MEEP's resolved factor is
        part of matching its stored values, not an optimisation.
        """
        self._require_open()
        monitor_frequencies = _normalize_frequencies(frequencies=frequencies)
        request = _decimation_request(decimation_factor)
        resolved = (
            self._automatic_decimation_factor(monitor_frequencies)
            if request == 0
            else request
        )
        accumulator = YeeRegionDFT(
            self.grid, component=component, bounds=tuple(bounds),
            frequencies=monitor_frequencies, decimation_factor=resolved,
        )
        self._dft_monitors.append(accumulator)
        if request == 0:
            self._automatic_decimation_monitors.append(accumulator)
        return accumulator

    def add_energy_monitor(
        self,
        frequencies: Any = None,
        center: Sequence[float] | None = None,
        size: Sequence[float] | None = None,
        *,
        frequency: Any = None,
        fcen: float | None = None,
        df: float | None = None,
        nfreq: int | None = None,
        decimation_factor: int = 0,
    ) -> EnergyMonitor:
        """Accumulate the electric and magnetic energy density over one region.

        MEEP's ``sim.add_energy(fcen, df, nfreq, EnergyRegion(...))``; read back
        through :meth:`~.dft.EnergyMonitor.electric` / ``magnetic`` / ``total``,
        MEEP's ``mp.get_electric_energy`` and friends. Frequencies and
        ``decimation_factor`` are spelled exactly as in :meth:`add_flux_monitor`.

        The region may be a volume, a plane, a line or a point; ``_measure`` picks
        up one ``dx`` per axis with extent, MEEP's own ``dV0`` rule.
        """
        self._require_open()
        monitor_frequencies = _normalize_frequencies(frequencies, frequency, fcen, df, nfreq)
        if center is None or size is None:
            raise ValueError(
                "An energy monitor needs both 'center' and 'size' — the region it integrates."
            )
        region_center = _as_xyz("center", center)
        region_size = _as_xyz("size", size)
        if any(extent < 0.0 for extent in region_size):
            raise ValueError(
                f"Energy region size must be non-negative along every axis, got {region_size}."
            )
        self._require_region_flat_on_invariant_axes(region_size, "energy region")
        request = _decimation_request(decimation_factor)
        resolved = (
            self._automatic_decimation_factor(monitor_frequencies)
            if request == 0
            else request
        )
        monitor = EnergyMonitor(
            grid=self.grid,
            frequencies=monitor_frequencies,
            center=region_center,
            size=region_size,
            periodic=self._monitor_periodic_axes(),
            decimation_factor=resolved,
        )
        self._dft_monitors.append(monitor)
        if request == 0:
            self._automatic_decimation_monitors.append(monitor)
        return monitor

    def add_force_monitor(
        self,
        frequencies: Any = None,
        center: Sequence[float] | None = None,
        size: Sequence[float] | None = None,
        force_direction: int | None = None,
        normal: int | None = None,
        *,
        weight: complex = 1.0,
        frequency: Any = None,
        fcen: float | None = None,
        df: float | None = None,
        nfreq: int | None = None,
        decimation_factor: int = 0,
    ) -> ForceMonitor:
        """Accumulate the Maxwell-stress-tensor force over one region.

        MEEP's ``sim.add_force(fcen, df, nfreq, ForceRegion(...))``; read back
        through :meth:`~.dft.ForceMonitor.get_force_spectrum`, MEEP's
        ``mp.get_forces``.

        ``force_direction`` is MEEP's ``fd`` — the axis of the force, which a
        ``ForceRegion`` must declare because it has no relation to the surface's
        orientation — and ``normal`` is MEEP's ``nd``, the geometric normal of the
        region's volume. They must agree: only the DIAGONAL branch of the stress
        tensor is reproduced, and :class:`~.dft.ForceMonitor` names why.
        """
        self._require_open()
        monitor_frequencies = _normalize_frequencies(frequencies, frequency, fcen, df, nfreq)
        if center is None or size is None:
            raise ValueError(
                "A force monitor needs both 'center' and 'size' — the surface it integrates."
            )
        region_center = _as_xyz("center", center)
        region_size = _as_xyz("size", size)
        self._require_region_flat_on_invariant_axes(
            tuple(abs(extent) for extent in region_size), "force region"
        )
        request = _decimation_request(decimation_factor)
        resolved = (
            self._automatic_decimation_factor(monitor_frequencies)
            if request == 0
            else request
        )
        monitor = ForceMonitor(
            grid=self.grid,
            frequencies=monitor_frequencies,
            center=region_center,
            size=region_size,
            force_direction=force_direction,
            normal=normal,
            weight=weight,
            decimation_factor=resolved,
        )
        self._dft_monitors.append(monitor)
        if request == 0:
            self._automatic_decimation_monitors.append(monitor)
        return monitor

    def add_flux_monitor(
        self,
        frequencies: Any = None,
        center: Sequence[float] | None = None,
        size: Sequence[float] | None = None,
        direction: int | None = None,
        *,
        frequency: Any = None,
        fcen: float | None = None,
        df: float | None = None,
        nfreq: int | None = None,
        decimation_factor: int = 1,
    ) -> FluxMonitor:  # Accumulate Poynting flux through one axis-aligned plane.
        """Add a flux monitor on an axis-aligned plane, at one frequency or a list of them.

        Frequencies are spelled exactly as in :meth:`add_dft_monitor`: a scalar, a
        sequence, or MEEP's ``fcen`` / ``df`` / ``nfreq`` triple.
        ``decimation_factor`` has the same automatic-zero / explicit-positive
        semantics as :meth:`add_dft_monitor`.

        ``direction`` is the plane normal (0 = X, 1 = Y, 2 = Z). When omitted it is
        taken from the flat axis of ``size``; a volume with no flat axis raises rather
        than defaulting to Z, since a silently reinterpreted plane yields a plausible
        but meaningless power number. A DECLARED normal may carry extent — that is
        MEEP's flux volume, and it integrates the normal Poynting component over the
        volume rather than across a surface (see the refusal site below).
        """
        self._require_open()
        monitor_frequencies = _normalize_frequencies(frequencies, frequency, fcen, df, nfreq)
        if center is None or size is None:
            raise ValueError("A flux monitor needs both 'center' and 'size' — the plane it measures.")
        plane_center = _as_xyz("center", center)
        plane_size = _as_xyz("size", size)
        if any(extent < 0.0 for extent in plane_size):
            raise ValueError(f"Flux plane size must be non-negative along every axis, got {plane_size}.")
        self._require_region_flat_on_invariant_axes(plane_size, "flux plane")
        # An invariant axis is flat by construction and can never be the normal: no power
        # crosses it (see the refusal below), and letting it win the auto-detection would
        # hand a 1-D run an x-normal plane where it wanted a z-normal one.
        flat_axes = [
            axis for axis in range(3)
            if plane_size[axis] < self.grid.dx and not self.grid.is_invariant(axis)
        ]
        if direction is None:
            if not flat_axes:
                raise ValueError(
                    f"Flux plane size {plane_size} has no flat axis; give a size below one cell "
                    f"({self.grid.dx:g}) on the normal axis, or pass an explicit direction."
                )
            normal = flat_axes[0]
        else:
            normal = int(direction)
            if normal not in (0, 1, 2):
                raise ValueError(f"Unknown flux direction {direction!r}; use 0 (X), 1 (Y), or 2 (Z).")
            if self.grid.is_invariant(normal):
                raise ValueError(
                    f"a flux plane normal to {AXIS_NAMES[normal]} measures the power crossing an "
                    f"axis that dimensions={self.grid.dimensions} makes translationally invariant, "
                    f"which is identically zero for every field this run can hold: the normal "
                    f"Poynting component pairs the two in-plane E components with the two in-plane "
                    f"H components, and a reduced run has one of each pair equal to zero in both "
                    f"polarizations. The monitor would accumulate, report 0.0, and look like a "
                    f"measurement. Measure across one of the axes this run resolves "
                    f"({', '.join(AXIS_NAMES[a] for a in range(3) if not self.grid.is_invariant(a))})."
                )
            # A DECLARED normal may have extent: MEEP allows a flux VOLUME and its own
            # test_visualization.py builds one (`FluxRegion(size=(4,4,4),
            # direction=mp.X)`). `_add_fluxish_stuff` (python/simulation.py) puts the
            # region's volume into the list unexamined and `add_dft_flux` accumulates
            # the four tangential components over whatever it is, so the monitor
            # integrates the normal Poynting over a volume rather than a surface. This
            # engine needs no special case for it: the ladder treats the normal axis
            # like any other (:meth:`~.dft.FluxMonitor._setup_region`) and `_measure`
            # already picks up one dx per extended axis, MEEP's own dV0 rule. Measured
            # against `mp.get_fluxes` on the same region, decimation pinned on both
            # sides: 2-D (2,4,0) normal X 9.40e-08, 2-D (4,4,0) normal Y 2.57e-09, 2-D
            # (3,3,0) normal X off-centre 6.54e-09, 3-D (4,4,4) normal X 6.28e-08 —
            # the same class as the flat controls beside them (1.43e-07, 3.86e-08).
            # The INFERRED case above still raises, because there a normal with extent
            # means nothing was declared and the axis would be a guess.
        request = _decimation_request(decimation_factor)
        resolved_decimation = (
            self._automatic_decimation_factor(monitor_frequencies)
            if request == 0
            else request
        )
        monitor = FluxMonitor(
            grid=self.grid,
            frequencies=monitor_frequencies,
            center=plane_center,
            size=plane_size,
            direction=normal,
            periodic=self._monitor_periodic_axes(),
            decimation_factor=resolved_decimation,
        )
        self._flux_monitors.append(monitor)
        if request == 0:
            self._automatic_decimation_monitors.append(monitor)
        return monitor

    def _automatic_decimation_factor(self, monitor_frequencies: Sequence[float]) -> int:
        """Resolve MEEP ``fields::add_dft`` automatic decimation.

        This is the ``decimation_factor == 0`` block in ``src/dft.cpp:195-216``
        (MEEP 1.33.0): ``floor(1 / (dt * (freq_max + src_freq_max)))``, bounded
        below by one, where a Gaussian source contributes ``abs(frequency) +
        fwidth/2``. Continuous and custom sources have zero declared bandwidth
        and therefore resolve to one when they are the only source types. In a
        mixed source list, MEEP's implementation derives the factor from the
        Gaussian members.

        A NONLINEAR run resolves to one. MEEP gates the bandwidth formula on
        ``!has_nonlinearities(false)`` (dft.cpp:207-210): chi2/chi3 pump power
        into harmonics and mixing products OUTSIDE the source band, so a Nyquist
        bound derived from source bandwidth no longer covers the spectrum the
        accumulators see. MEEP's predicate (fields.cpp:681-686 ->
        structure.cpp:548-556) is true when any owned chunk carries a non-NULL
        instantaneous ``chi2[c]`` / ``chi3[c]`` array for any component, or when
        any polarizable susceptibility chain reports nonlinear — which only
        ``multilevel_susceptibility`` does (meep.hpp:350; the base class walks
        the chain returning false, meep.hpp:108, so Lorentz/Drude dispersion is
        NOT nonlinear to this test). Of those clauses this engine mirrors
        exactly the reachable subset: instantaneous chi2/chi3 via
        ``fields.has_nonlinearity`` (:meth:`set_chi2` / :meth:`set_chi3`).
        Multilevel-atom susceptibilities are refused at
        :meth:`add_susceptibility`, and the cross-chunk / cross-process
        reductions (``is_mine`` in structure.cpp:550, ``min_to_all`` in
        dft.cpp:215) are unreachable in a single-domain engine. One deliberate
        divergence rides on the engine's trivial-pair removal (:meth:`set_chi2`):
        MEEP keeps the allocated-but-zeroed chi arrays of an overwritten
        nonlinearity, so its guard stays latched after zeroing, where this
        engine deletes an identically-zero pair and returns to the bandwidth
        factor — consistent with its stepping the linear kernel byte for byte.
        """
        if self.fields.has_nonlinearity:
            # dft.cpp:207-210: the `!has_nonlinearities(false)` clause fails, so
            # the else-branch resolves one — sample every step.
            return 1
        source_frequency_max = 0.0
        for source in self._sources:
            if isinstance(source, GaussianPulsedSource):
                frequency, width = source.frequency, source.fwidth
            elif isinstance(source, VolumeSource) and isinstance(source.envelope, GaussianEnvelope):
                frequency, width = source.envelope.frequency, source.envelope.fwidth
            else:
                continue
            source_frequency_max = max(
                source_frequency_max,
                abs(float(frequency)) + 0.5 * float(width),
            )
        monitor_frequency_max = max(
            (abs(float(frequency)) for frequency in monitor_frequencies),
            default=0.0,
        )
        if monitor_frequency_max <= 0.0 or source_frequency_max <= 0.0:
            return 1
        return max(
            1,
            int(math.floor(
                1.0 / (self.grid.dt * (monitor_frequency_max + source_frequency_max))
            )),
        )

    def _refresh_automatic_decimation(self) -> None:
        """Refresh unresolved automatic monitors while no sample has been taken.

        Called when the resolution's inputs change after a ``decimation_factor=0``
        monitor already exists: a source joins the list (:meth:`add_source`), or a
        nonlinearity is installed or removed (:meth:`_install_nonlinearity`).
        Every monitor's ``_configure_decimation_factor`` refuses the change after
        accumulation has started, so a refresh that arrives too late is loud.
        """
        for monitor in self._automatic_decimation_monitors:
            monitor._configure_decimation_factor(
                self._automatic_decimation_factor(monitor.frequencies)
            )

    def _folded_far_face_ratio(self) -> tuple[int, float]:  # (axis, |field| there / |field| anywhere).
        """Largest primary-field amplitude on the far face of a folded axis, relative to the run's peak.

        Returns ``(-1, 0.0)`` when no axis is folded, and skips a folded axis whose
        declared boundary is METALLIC: there the zero ghost is not an approximation of
        MEEP's parity-weighted image, it IS the boundary condition, so how much field
        reaches that face says nothing about the fold's accuracy. On a folded
        PERIODIC axis the far plane is now a stored, stepped degree of freedom
        (MEEP's ``big_corner``; ``Grid.stored_cells`` vs ``owned_cells``
        and ``stepping.fill_folded_far_ghosts_*``), so a live face is the normal
        state of a correct run — the only consumer left is the nonlinear guard
        (:meth:`_require_folded_far_face_is_quiet`), whose transverse sums still
        read a zero face there.

        Reads the stored D and B arrays rather than E/H so the measurement costs one
        reduction per component and never allocates a derived volume.
        """
        xp = self.xp
        folded = [
            axis for axis in range(3)
            if self.grid.is_mirrored(axis) and not self.grid.is_metallic(axis)
        ]
        if not folded:
            return -1, 0.0
        peak, worst_axis, worst_face = 0.0, -1, 0.0
        for name in PRIMARY_COMPONENTS:
            array = getattr(self.fields, name, None)
            if array is None:
                continue
            peak = max(peak, float(xp.max(xp.abs(array))))
            for axis in folded:
                face = float(xp.max(xp.abs(array[_face_index(axis, -1)])))
                if face > worst_face:
                    worst_axis, worst_face = axis, face
        if peak <= 0.0:
            return worst_axis, 0.0
        return worst_axis, worst_face / peak

    def _require_folded_far_face_is_quiet(self):
        """Refuse a NONLINEAR folded run whose periodic far face has become live.

        A mirror plane at x = 0 under periodic boundaries implies a SECOND mirror at
        x = +/-L/2: MEEP reaches the ghost past the folded half by a lattice translation
        followed by the symmetry transform (boundaries.cpp ``connect_the_chunks`` ->
        ``locate_component_point``), which returns the parity-weighted image of an
        interior plane. The linear engine now reproduces that treatment in full:
        the plane at the window top is a stored, stepped degree of freedom
        (``Grid.stored_cells`` — one cell past ``owned_cells`` at either count
        parity, MEEP's own ``num + 1`` allocation), the shift-1
        ghost slot past it is imaged by ``stepping.fill_folded_far_ghosts_*``,
        and reads past the stored top reflect through ``_shift_up``'s
        parity-weighted rule. Measured with the face fully live (2-D 4x2 cell at
        res 16, PML X only, Y periodic at Gamma, Gaussian f=1 fwidth=0.5,
        until=8: the recon configuration whose CPU-MEEP folded-vs-unfolded floor
        is 1.26e-07/1.14e-07): see test_driver_vs_meep's folded-live-face cases
        for the pinned numbers.

        WHAT STILL CANNOT SERVE A LIVE FACE: the nonlinear transverse sums.
        ``stepping._nonlinear_transverse_sums`` shifts PRODUCTS of components,
        which carry no single mirror parity, so their far-face read on a folded
        periodic axis keeps the zero ghost (``_shift_up`` with no
        ``reflect_row``). While that face is dark the zero is exact and a folded
        nonlinear run is at its usual floor; once it is live the sums would be
        assembled from a wall MEEP does not have. So this refusal now fires only
        when a chi2/chi3 is installed — the linear paths are measured and free.

        **A metallic folded axis is never measured here**, because there the zero
        ghost is exactly right for the sums too: MEEP holds the window-top plane
        at zero (``find_metals`` / ``zero_metal``), which is what the zero ghost
        says. ``symmetry=('X',)`` with ``boundaries='metallic'`` — MEEP's own
        default arrangement — runs to the floor with the wave hitting the wall
        as hard as it likes, nonlinearity included.
        """
        if not self.fields.has_nonlinearity:
            return
        axis, ratio = self._folded_far_face_ratio()
        if axis < 0 or ratio <= _FOLDED_FAR_FACE_LIMIT:
            return
        raise RuntimeError(
            f"The far {AXIS_NAMES[axis]} face of this mirror-folded NONLINEAR run carries "
            f"{ratio:.3e} of the run's peak field, above the {_FOLDED_FAR_FACE_LIMIT:g} the "
            f"nonlinear fold can stand for. The linear stencils reflect that face through the "
            f"second mirror a periodic cell with a mirror at 0 has at +/-L/2, but the chi2/chi3 "
            f"transverse sums shift component PRODUCTS, which carry no single mirror parity and "
            f"still read a zero ghost there — the two systems agree only while that face is "
            f"dark. Re-run without symmetry={self.grid.symmetry}, absorb that face and stop "
            f"before the wave reaches it, or — if the walls you meant were MEEP's default "
            f"perfect conductors rather than periodic images — build the run with "
            f"boundaries='metallic', where the zero ghost IS the boundary condition and this "
            f"measurement does not apply."
        )

    def _require_region_flat_on_invariant_axes(self, size, description: str) -> None:
        """Refuse a monitor region that claims an extent along a translationally invariant axis.

        The surface measure is what makes this a correctness matter rather than a
        tidiness one. Both monitor kinds take ``dA = dx ** (number of axes the region
        extends along)`` — MEEP's own ``dV`` for the requested region — so a 2-D flux
        line asked for as ``size=(0, w, dx)`` instead of ``size=(0, w, 0)`` is
        integrated as if it were an area and comes back a factor of ``dx`` too large,
        which is a factor of 10 at resolution 10 and looks like an ordinary number.

        MEEP cannot even be asked the question: a reduced ``grid_volume`` has no z
        direction, so ``volume`` objects built for it carry x and y alone and the
        extent has nowhere to be written. Zero is therefore the only extent an
        invariant axis has, and asking for another is a request this engine declines
        rather than reinterprets.
        """
        if not self.grid.has_invariant:
            return
        offenders = [
            AXIS_NAMES[axis] for axis in range(3)
            if self.grid.is_invariant(axis) and float(size[axis]) != 0.0
        ]
        if not offenders:
            return
        raise ValueError(
            f"{description} size {tuple(size)} claims an extent along {offenders}, which "
            f"dimensions={self.grid.dimensions} makes translationally invariant. Such an axis is "
            f"infinite and uniform, so a region has no thickness along it — and the surface "
            f"measure counts every axis with a nonzero extent, so this request would be "
            f"integrated over one dimension too many and come back a factor of dx "
            f"({self.grid.dx:g}) out per offending axis. Pass 0 on {offenders}."
        )

    def _monitor_periodic_axes(self) -> tuple[bool, bool, bool]:  # Axes a monitor region may wrap across.
        """Which faces a monitor may continue across, for this run's boundaries.

        MEEP takes a lattice shift where ``boundaries[High][d] == Periodic``
        (loop_in_chunks.cpp), and with a ``k_point`` — which every run this engine
        models carries — that is every direction, absorber or not. So a monitor reaching
        past a face is wrapped on every unmirrored axis, exactly as the stepping wraps
        it (:func:`~.stepping._boundary_kinds`); the two must not disagree, or a monitor
        measures a boundary the run does not have. A mirror axis is the one exception:
        MEEP reaches the far half through the symmetry transform, which this engine's
        flux does not unfold.

        The rule is :func:`_periodic_axes`, which carries the measured cost of dropping
        the wrap on an absorbing axis (1.4e-02 to 2.3e-02 on a flux plane, against
        1.4e-07 with it).
        """
        return _periodic_axes(self.grid)

    def invalidate_fast_path(self) -> None:
        """Drop the cached fast-path plan; the next ``step()`` re-plans from scratch.

        THE ONE centralized invalidation seam: every mutator that swaps or writes a
        material array calls this — never a per-mutator convention. Plain
        :meth:`set_epsilon` and the smoothing route REPLACE material arrays and are
        legal after stepping has begun, so any cached flattened inverse-epsilon
        pointer a fast-path plan holds goes stale through paths a per-feature list
        would miss; one funnel makes "did you invalidate?" a greppable question.
        Invalidation is cheap (a re-plan on the next step), so mutators call it
        unconditionally rather than reasoning about whether a plan could exist.
        """
        self._fast_path = None
        self._fast_path_stale = True

    @property
    def active_step_path(self) -> str:
        """Which stepping implementation the FROZEN configuration runs: "array" or "fused".

        TWO-VALUED, and it stays two-valued: ``benchmark_gpu.py`` hard-codes
        ``choices=("array", "fused")`` and five results harnesses read it as a
        boolean, so this is a contract rather than a label. What "fused" MEANS is
        the part that moved — it is now "at least one sub-step will launch a
        KERNEL", not "a plan object exists".
        The distinction is load-bearing: the no-PML null family's product replaces
        ``update_H``/``update_E`` with a plan that does what the array path does,
        namely nothing, so a plan-object test would have a NumPy laptop with an
        inert layer reporting "fused" for a run that launches nothing at all.

        READ IT AFTER A STEP, not before. The plan is built at the configuration
        freeze inside ``step()``, so this reads "array" before the first step of a
        run that will dispatch, and again between ``invalidate_fast_path()`` and the
        next step. That is not a lie about the future — nothing has decided yet —
        but it is why ``benchmark_gpu.py`` asserts the requested path AFTER warmup,
        and why a harness that reads this first would under-report a fused run as
        array. Making the property plan on demand was refused deliberately: the
        freeze compiles kernels, and a property with that side effect is worse than
        a property that answers about the frozen state.

        The benchmark harness asserts the requested path against this property so a
        predicate regression cannot silently masquerade as a perf regression.
        ``MEEP_GPU_FUSED=0`` forces "array" for bisection AT THE FREEZE: it is
        read when the plan is built, so exporting it before the process starts (or
        before the first step) is what a bisection does. Setting it mid-run changes
        nothing until something calls ``invalidate_fast_path()`` — measured: two
        further launches after the switch was set, and 'array' only after an
        explicit invalidation.

        A driver built with ``prefer_gpu=False`` reads "array" always: it is the
        NumPy reference, and its freeze never consults a kernel table.
        """
        plan = self._fast_path
        return "fused" if plan is not None and plan.dispatches_a_kernel else "array"

    def fast_path_report(self):
        """This configuration's dispatch artifact — which slots ran where, and why.

        The record a user reads instead of the source to answer "did I get
        kernels, which ones, on what certification, and if not why not". Present
        whether or not a plan was built: a refusal carries the single named reason
        it was refused for. ``MEEP_GPU_DISPATCH_LOG`` additionally appends one
        JSON object per configuration freeze, so a long run's dispatch state is
        readable with ``tail`` on the machine that owns the job.

        ALWAYS THIS DRIVER'S OWN FREEZE, dispatched or refused, and that is a
        correction: it used to fall through to the module-level last record
        whenever no plan was held, which is one process-wide slot that every
        freeze overwrites. Measured, a refused driver then reported
        ``decision='dispatched'``, ``step_path='fused'`` and a dispatched slot list
        while its own ``active_step_path`` read ``"array"`` — the artifact
        inverted, not merely stale — because a second driver had frozen a
        dispatching plan in between. The 13 material mutators that call
        ``invalidate_fast_path`` make re-freezes routine, so this was reachable
        without anything unusual.

        ``None`` before the first ``step()``: no configuration has frozen yet, so
        this driver has no verdict to report and will not borrow one.

        A driver built with ``prefer_gpu=False`` reports its freeze as the NumPy
        reference: ``reference_driver`` True, ``table`` None, ``refused_because``
        :data:`~meep_gpu.fastpath.REFERENCE_REASON`, and the ``environment`` block
        (hardware, candidate tables, table preference) marked ``not_read``. The
        ``enable`` and ``kill_switch`` blocks still report what ``MEEP_GPU_DISPATCH``
        and ``MEEP_GPU_FUSED`` were set to, ``effective`` and ``vetoes`` included;
        those describe the values and decide nothing on a reference driver.
        """
        plan = self._fast_path
        if plan is not None and not self._fast_path_stale:
            return plan.report()
        record = self._fast_path_record
        if record is None:
            return None
        if self._fast_path_stale:
            return dict(record, superseded="a material mutation invalidated this "
                                           "plan; the next step() re-freezes and "
                                           "re-decides")
        return record

    def step(self):  # Advance one time step in MEEP's fields::step() order.
        """Perform one FDTD time step.

        MEEP step.cpp fields::step() order, reproduced exactly:
            1. ``step_B``   — B -= dt * curl(E)
            2. magnetic sources — B -= dt * K(t)
            2a. ``fill_symmetry_bc_B`` — mirror-fill the non-owned B cells
            2b. ``zero_metal_B`` — clear the B samples on any metallic wall
            3. ``update_H`` — H = B / mu
            4. ``step_D``   — D += dt * curl(H)
            5. electric sources — D -= dt * J(t + dt/2)
            6. ``fill_symmetry_bc_D`` — mirror-fill the non-owned D cells
            6b. ``zero_metal_D`` — clear the D samples on any metallic wall
            7. ``update_E`` — E = (D - sum P) / epsilon_infinity
            7b. ``update_P`` — advance every polarization from the W of the step just closed
            8. step_count += 1, then monitors accumulate at the post-increment time.

        Step 7b is MEEP's ``update_pols(E_stuff)`` immediately after
        ``update_eh(E_stuff)`` (step.cpp:105-114). Entering the step the arrays hold
        P^n, step 7 consumes P^n to form E^n, and step 7b then writes P^(n+1). The
        monitors run after 7b and read the STORED E, which is still E^n — that only
        stays true because E is stored, which is why any susceptibility switches the
        storage on.

        MEASURED, against the expectation that swapping 7 and 7b would detune the
        resonance: it does not, and the reason is worth writing down. Advancing P
        first shifts BOTH indices by one — the polarization is driven by the previous
        step's W, and the stored E then subtracts the already-advanced P — and the two
        shifts cancel exactly, so the swapped order is the same scheme with the array
        holding a different generation of P between the two calls. Verified: with the
        calls swapped, every dispersive CPU-MEEP case still lands on its recorded
        floor (3.2e-06 no-PML, 5.7e-06 uniform PML). MEEP's order is kept because it
        is MEEP's, and because the equivalence holds only while nothing reads P
        between them; ``test_the_two_orders_of_update_E_and_update_P_agree`` pins it.

        The two source slots are MEEP's, not a convenience: step.cpp:64-100 calls
        ``calc_sources(time())`` before ``step_boundaries(B_stuff)`` and
        ``calc_sources(time() + dt/2)`` before the D half, so the two field families
        see currents half a step apart. Injecting a magnetic source in the electric
        slot is a pure phase error — measured 1.6e-01 against CPU MEEP, where the
        correct slot gives 2.9e-07 — and is invisible to a magnitude comparison.
        Each source reports which slot it belongs to through ``field_type``.

        Step 6 is the port fix from reference §5.2: MEEP fills D boundaries after
        ``step_source``, and the bundle exported ``fill_symmetry_bc_D`` for exactly
        this slot but never called it from either driver, so symmetric runs left the
        mirror-plane D cells stale — which the PML+symmetry kernels explicitly rely
        on this call to repair.

        Step 2a is the SAME fix on the magnetic side, and it is the same slot for the
        same reason: step.cpp:67-72 is ``step_db(B_stuff)``, ``step_source(B_stuff)``,
        ``step_boundaries(B_stuff)`` — the D half at :95-103 verbatim, one field type
        earlier. The B fill used to run at the end of ``step_B``, a sub-step before the
        magnetic currents land, so every mirror ghost held the image of its owned cell
        as of BEFORE that cell was driven. Only a run with a magnetic current can see
        it, and on a Y fold only ``By`` has a ghost row to be stale (Yee shift 0 on the
        folded axis), which is why it surfaced as an eigenmode-source defect on ``Hy``
        alone: mode-decomposition.py at resolution 17 read Hy 2.72e-03 with Ez and Hx
        at 1.32e-06 / 2.51e-06, and 1.62e-06 on Hy with the fill in this slot (whole
        volume 2.309e-03 -> 1.802e-06, Ez and Hx bit-unchanged).

        Steps 2b and 6b are the metallic-wall passes, and their SLOT is the same fact
        one field type later: MEEP runs ``step_boundaries`` after ``step_source`` on
        each side (step.cpp:64-105), so a current deposited onto a perfect conductor
        is injected first and wiped second. Running either one earlier — inside
        ``step_B`` or before the sources — leaves that current standing on the wall,
        where it radiates. It is invisible in every run whose sources are off the wall,
        which is why a mutation deleting the B-side call survived the whole CPU-MEEP
        metallic cross-validation until a test put an Hx current on an x wall.
        """
        self._require_open()
        if self._automatic_decimation_monitors:
            self._refresh_automatic_decimation()
            # Freeze factors at the first accumulated timestep, matching MEEP's
            # one-time resolution when a DFT object is initialized.
            self._automatic_decimation_monitors.clear()
        if self._pending_monitor_regions:
            self._resolve_monitor_regions()
        if self._fast_path_stale:
            # The configuration freezes at the first step (and re-freezes after any
            # invalidate_fast_path()): plan once here, not per step, so kernel
            # warm-up and coefficient caching land in setup rather than in step 1.
            # Keep THIS freeze's artifact, dispatched or refused. The module-level
            # record is one process-wide slot that the next driver to freeze
            # overwrites; a driver that reads it later reports that driver's verdict.
            if self.gpu is not None:
                self._fast_path = plan_fast_path(self.fields, self.pml, self.grid,
                                                 self._sources)
                self._fast_path_record = last_dispatch_report()
            else:
                # prefer_gpu=False IS THE NUMPY REFERENCE: it consults no kernel
                # table, whatever the environment says, and its record says so. The
                # record is the one this call built, taken from its return value
                # rather than read back out of the process-wide slot.
                self._fast_path = None
                self._fast_path_record = reference_record()
            self._fast_path_stale = False
        # Read ONCE, after the freeze. Every sub-step below asks the same object,
        # so a step cannot run half on kernels built from one configuration and
        # half on kernels built from another. `fast.dispatch` answers False for
        # every slot the plan does not carry, which is the array path.
        fast = self._fast_path
        dt = self.grid.dt
        magnetic = [source for source in self._sources if source.field_type == FIELD_TYPE_B]
        electric = [source for source in self._sources if source.field_type != FIELD_TYPE_B]
        # MEEP's integrated source never rides the curl ladder: its dipole enters at
        # the constitutive read (update_eh.cpp:126-138) and the primary array MEEP
        # steps holds no offset. Each integrated source here withdraws its standing
        # offset before step_B/step_D so the ladder — PML rescale, conductivity —
        # sees exactly MEEP's f, then reinjects the fresh whole dipole in its slot.
        for source in magnetic:
            getattr(source, "withdraw", _no_withdraw)(self.fields)
        if fast is None or not fast.dispatch("step_B", self.fields):
            step_B(self.fields, self.pml)
        for source in magnetic:
            source.inject(self.fields, self.time)
        # THE FILL SUB-STEP IS TWO CONSULTS AND ONE UNCONSULTED PASS BETWEEN THEM.
        # ``fastpath.DRIVER_SLOTS``' ``fill_B`` names the near-symmetry pass and the
        # folded-far pass together, but the wall clear sits between them in MEEP's
        # own order, so a single consult could only answer for a span the composer
        # does not own. The near site is consulted with the SLOT name and the far
        # site with the driver's own PASS name; each answer is total for its own
        # pass, and an adapter that carries neither name answers False to both,
        # which is the array path. ``zero_metal_B`` stays behind NO consult, which
        # is the fact ``deposit_repair`` reads when it says the field is injected,
        # filled and cleared by the second consult of a fused pair.
        if fast is None or not fast.dispatch("fill_B", self.fields):
            fill_symmetry_bc_B(self.fields)
        zero_metal_B(self.fields)  # MEEP step_boundaries(B_stuff), AFTER step_source.
        # Far-plane image of a folded periodic even axis.
        if fast is None or not fast.dispatch("fill_folded_far_ghosts_B", self.fields):
            fill_folded_far_ghosts_B(self.fields)
        if fast is None or not fast.dispatch("update_H", self.fields):
            update_H(self.fields, self.pml)
        for source in electric:
            getattr(source, "withdraw", _no_withdraw)(self.fields)
        if fast is None or not fast.dispatch("step_D", self.fields):
            step_D(self.fields, self.pml)
        source_time = self.time + 0.5 * dt
        if electric and self.fields.has_conductivity:
            self._inject_electric_through_conductivity(electric, source_time)
        else:
            for source in electric:
                source.inject(self.fields, source_time)
        # The electric half of the two-consult fill seam above, in the same slot one
        # field type later.
        if fast is None or not fast.dispatch("fill_D", self.fields):
            fill_symmetry_bc_D(self.fields)
        zero_metal_D(self.fields)  # MEEP step_boundaries(D_stuff), AFTER step_source.
        # Far-plane image of a folded periodic even axis.
        if fast is None or not fast.dispatch("fill_folded_far_ghosts_D", self.fields):
            fill_folded_far_ghosts_D(self.fields)
        if fast is None or not fast.dispatch("update_E", self.fields):
            update_E(self.fields, self.pml)
        if fast is None or not fast.dispatch("update_P", self.fields):
            update_P(self.fields, self.pml)
        self.step_count += 1
        for monitor in self._dft_monitors:
            monitor.update(self.fields, self.time, self.step_count)
        for monitor in self._flux_monitors:
            monitor.update(self.fields, self.time, self.step_count)

    def _inject_electric_through_conductivity(self, electric, source_time):  # MEEP's cndinv-scaled currents.
        """Inject the electric currents into a lossy D, scaled by ``condinv``.

        **The current is scaled by condinv.** MEEP step.cpp:294-317 multiplies each
        injected current by ``s->condinv[c][d]`` when a conductivity is present, so a
        source deposits ``dt*J*condinv`` rather than ``dt*J`` — the same implicit
        half-step the conductive curl update applies to D.

        Once in D the deposit is then damped by ``step_D`` like every other part of D,
        which is what a conductivity means. Under PML, MEEP likewise leaves the
        source out of ``f_cond`` and writes the ``condinv``-scaled increment directly
        into D after the three-stage curl recurrence.

        ``sources`` is not this file's to edit, so the scaling is applied by
        difference — but ONLY at the deposit cells the sources publish
        (``deposit_repair._deposit_index`` reads the same tables): snapshot the target
        D component at those indices, inject, then replay the rescale per deposit cell
        in the same operand order. Exact at every deposit cell, untouched everywhere
        else, and paid only by runs that have both a conductivity and an electric
        source.

        **Why sparse, not whole-volume.** This used to be three whole-volume passes
        (``array -= before; array *= condinv; array += before``). Those are the
        identity at every non-deposit finite value EXCEPT that they canonicalise
        ``-0.0`` to ``+0.0`` (``x - x`` is ``+0.0``, and ``+0.0 + (-0.0)`` is
        ``+0.0``) at every cell of the component — a rewrite stock MEEP never
        performs, because step.cpp:294-317 scales only the injected current at the
        source's own points. The whole-volume canonicalisation is what forced the
        fused electric pairs on three backends to refuse conductive rows by name
        (their launches compute E from pre-rewrite values, so ``-0.0``/``+0.0``
        word diffs appeared at cells no deposit repair can name). The sparse replay
        leaves non-deposit ``-0.0`` words as ``-0.0``, matching MEEP. On the CuPy
        backend the whole-volume passes were lossier still — device arithmetic
        flushes subnormals, so every non-deposit subnormal word of the component
        was silently zeroed each conductive-injection step; measured on an RTX
        A6000 (2026-09-01): replaying sparsely, 54 signed-zero + 68 subnormal
        words of 588 survive that the passes destroyed, with the deposit cells
        byte-identical.

        A source that publishes no deposit table at all (no ``_point_ix`` attribute
        — no in-tree electric source class is one) falls back to the whole-volume
        difference passes for its component, because an unnameable deposit must
        still be scaled: correctness of the physics outranks the signed-zero words.

        **An integrated source is NOT scaled**: MEEP's ``step_source`` skips it
        (step.cpp:300) — its ``cndinv`` multiply never runs — and the dipole instead
        reaches the field at the constitutive read (update_eh.cpp:126-138),
        downstream of the conductivity, fresh and undamped. The dipole an integrated
        source leaves standing in D here is withdrawn before ``step_D`` (see
        :meth:`step` and :meth:`.sources.VolumeSource.withdraw`), so the conductive
        recurrence never touches it either; the injection below is the whole of what
        it needs. Measured against CPU MEEP (2-D 8x8 cell, res 10, eps 2.25, uniform
        sigma_D = 0.4, CW Ez is_integrated point source, until 15, complex Ez
        relative L2 over the volume): 2.0417e-07 with the withdraw slot, where the
        pre-withdraw telescoping — which this driver used to refuse by name — reads
        6.3802e-02 on the same run.
        """
        integrated = [source for source in electric if source.is_integrated]
        scaled = [source for source in electric if not source.is_integrated]
        for source in integrated:
            source.inject(self.fields, source_time)
        if not scaled:
            return
        by_target: dict = {}
        for source in scaled:
            by_target.setdefault("D" + source.component[1], []).append(source)
        xp = self.fields.grid.xp
        sparse = {}  # name -> (ix, iy, iz, before-values at those cells)
        dense = {}   # name -> whole-volume snapshot (unpublished-deposit fallback only)
        for name, sources_for in sorted(by_target.items()):
            if self.fields.condinv_for(name) is None:
                continue  # This component is lossless; MEEP's cndinv multiply never runs.
            array = getattr(self.fields, name)
            columns = []
            for source in sources_for:
                index = _deposit_index(source)
                if index is not None:
                    columns.append(index)
                elif not hasattr(source, "_point_ix"):
                    dense[name] = array.copy()  # Publishes no table; see the docstring.
                    break
            else:
                if not columns:
                    continue  # Every source on this component deposits into zero cells.
                if len(columns) == 1:
                    ix, iy, iz = columns[0]
                else:
                    ix = xp.concatenate([column[0] for column in columns])
                    iy = xp.concatenate([column[1] for column in columns])
                    iz = xp.concatenate([column[2] for column in columns])
                sparse[name] = (ix, iy, iz, array[ix, iy, iz])
        for source in scaled:
            source.inject(self.fields, source_time)
        # Replay the rescale per deposit cell, in the whole-volume passes' exact
        # operand order, so the bytes at every deposit cell are unchanged from the
        # retired composition. Cells two sources both hit gather the same
        # post-injection value and scatter the same result, so duplicates are safe.
        for name, (ix, iy, iz, before) in sparse.items():
            # SPARSE, through the residency door: a raw fetch, a gather of the deposit
            # cells, the same arithmetic in the same order, a scatter. Under a held
            # Metal residency the array is sealed and the old fancy-index store raised
            # (measured on conductive_2d, 2026-09-23); on the array path these are the
            # plain NumPy read and write. Duplicates in the cell list are safe here for
            # the reason the comment above gives -- every duplicate scatters the same
            # value -- so no uniqueness check is owed.
            array = host_writes.raw(self.fields, name)
            linear = host_writes.ravel_points(ix, iy, iz, array.shape)
            values = host_writes.gather(array, linear)
            values -= before
            # MEEP's s->condinv[c][d], read at the deposit cells only.
            values *= self.fields.condinv_for(name)[ix, iy, iz]
            values += before
            host_writes.scatter(array, linear, values)
        for name, before in dense.items():
            # Whole-volume by nature (no deposit table to be sparse over), and
            # unreachable from any in-tree source class; declared as the whole-volume
            # write it is so a held residency syncs it rather than sealing it.
            array = host_writes.acquire(getattr(self.fields, name))
            array -= before
            array *= self.fields.condinv_for(name)  # The component's OWN volume.
            array += before

    def _field_energy(self) -> float:  # sum |E|^2 eps + |B|^2, the divergence guard's scalar.
        """One scalar summarising the field, for the runtime divergence guard.

        MEEP checks ``D_EnergyDensity`` at the cell centre every step and aborts on a
        non-finite value (step.cpp:137-138). A single point is enough to catch a NaN
        but not enough to catch the dangerous case — a mode growing at 1.001 per step
        that finishes large, smooth and finite — so this reduces over the whole
        volume instead.
        """
        xp = self.xp
        total = 0.0
        for component in E_COMPONENTS:
            values = self.fields.get_E(component)
            total += float(
                xp.sum(
                    (values.real**2 + values.imag**2)
                    * self.fields.epsilon_for(component)
                )
            )
        for component in ("Bx", "By", "Bz"):
            values = getattr(self.fields, component)
            total += float(xp.sum(values.real**2 + values.imag**2))
        return total

    def _worst_susceptibility(self) -> str:  # The term most likely to be responsible, for the message.
        if not self._susceptibilities:
            return "none"
        worst = max(
            (term for term, _ in self._susceptibilities),
            key=lambda term: term.stability(self.grid.dt).pole_magnitude,
        )
        return worst.stability(self.grid.dt).describe()

    def run(
        self,
        num_steps: int | None = None,
        until: float | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_interval: int = 100,
        *,
        step_functions: Sequence[Callable[..., Any]] = (),
        until_after_sources: float | None = None,
        decay_by: float | None = None,
        decay_component: str | None = None,
        decay_point: Sequence[float] | None = None,
        max_time: float | None = None,
    ) -> float:  # Step the simulation, reporting progress and honouring cancellation.
        """Run to a stopping condition and return the elapsed wall time.

        Exactly one stopping condition is required:

        * ``num_steps`` — that many time steps.
        * ``until`` — up to a finite simulation time strictly beyond the current one.
          Infinite times raise (port reference §5.12).
        * ``until_after_sources`` — MEEP's field-decay stop. The value is the trailing
          window length in simulation time, and the run ends once the peak ``|f|**2``
          at ``decay_point`` over a window has fallen to ``decay_by`` times the largest
          window peak seen so far, never before the sources have finished emitting.
          ``decay_by``, ``decay_component``, ``decay_point`` and ``max_time`` are all
          required with it; this is MEEP's
          ``run(until_after_sources=stop_when_fields_decayed(window, c, pt, decay_by))``
          flattened into keywords.

        The decay stop cannot loop forever. ``max_time`` is the caller's ceiling and
        raises :class:`RuntimeError` when the field has not decayed by then, reporting
        the last measured ratio — a run whose probe never saw a field, or whose PML
        cannot absorb what the source keeps feeding it, must fail loudly rather than
        hang or report a decayed field it never measured. A continuous source, whose
        ``end_time`` defaults to ~1e20, can never finish before the ceiling and is
        rejected up front rather than after a long run.

        ``step_functions`` is MEEP's step-function list (``python/simulation.py``
        ``_run_until``): each entry is called once at the run's starting time and again
        after every step, then once with ``'finish'`` when the run ends normally. Both
        of MEEP's spellings are accepted, ``f(sim)`` and ``f(sim, todo)``, dispatched on
        the argument count exactly as ``_eval_step_func`` does — a one-argument function
        is called only for ``'step'``. That is what lets ``mp.Harminv``'s counterpart
        here (:class:`~.harminv.Harminv`) be handed straight to ``run``::

            probe = Harminv(c="Ez", pt=(0.13, 0.07, 0.0), fcen=0.30, df=0.20)
            driver.run(num_steps=4000, step_functions=[probe])
            probe.modes[0].Q

        The ordering is MEEP's: a step function sees the fields at ``t``, then the step
        to ``t + dt`` happens, so N steps produce N+1 samples uniformly spaced by ``dt``
        starting at the time ``run`` was entered. They are NOT called when the run
        raises — a cancelled or diverged run has no ``'finish'``, so an analysis step
        function leaves its previous result rather than one built from a truncated
        record.

        ``progress_cb(completed_steps, total_steps)`` and ``cancel_check()`` are polled
        every ``progress_interval`` steps and after the final step; under a decay stop
        ``total_steps`` is the ceiling's step count, the only total known in advance.
        A truthy ``cancel_check`` raises :class:`FdtdCancelled`, leaving the driver's
        state at the step it reached. Nothing is printed; the caller owns all output.
        """
        self._require_open()
        conditions = [
            name
            for name, value in (
                ("num_steps", num_steps),
                ("until", until),
                ("until_after_sources", until_after_sources),
            )
            if value is not None
        ]
        if len(conditions) != 1:
            raise ValueError(
                f"run() takes exactly one of num_steps, until, or until_after_sources; got "
                f"{conditions or 'none'}."
            )
        interval = int(progress_interval)
        if interval != progress_interval or interval < 1:
            raise ValueError(f"progress_interval must be a positive whole number of steps, got {progress_interval!r}.")
        decay: _FieldDecayProbe | None = None
        if until_after_sources is not None:
            decay, total = self._prepare_decay_stop(
                window=until_after_sources,
                decay_by=decay_by,
                decay_component=decay_component,
                decay_point=decay_point,
                max_time=max_time,
            )
        else:
            unexpected = [
                name
                for name, value in (
                    ("decay_by", decay_by),
                    ("decay_component", decay_component),
                    ("decay_point", decay_point),
                    ("max_time", max_time),
                )
                if value is not None
            ]
            if unexpected:
                raise ValueError(
                    f"{unexpected} configure the field-decay stop and need until_after_sources; "
                    f"this run stops on {conditions[0]}."
                )
            if until is not None:
                end_time = float(until)
                if not math.isfinite(end_time):
                    raise ValueError(
                        f"until must be a finite simulation time, got {until!r}; pass a step count for open-ended runs."
                    )
                if end_time <= self.time:
                    raise ValueError(
                        f"until={end_time} is not beyond the current simulation time {self.time}; nothing to run."
                    )
                total = int(math.ceil((end_time - self.time) / self.grid.dt - _STEP_TOLERANCE))
            else:
                total = int(num_steps)
                if total != num_steps or total < 1:
                    raise ValueError(f"num_steps must be a positive whole number of steps, got {num_steps!r}.")
        started = time_module.perf_counter()
        guard = _DivergenceGuard(self) if self.fields.has_polarizations else None
        # Independent of the energy guard, and needed on its own: a nonlinear run that
        # crosses the Pade pole does not gain energy, it returns a sign-flipped field.
        nonlinear = _NonlinearityGuard(self) if self.fields.has_nonlinearity else None
        steppers = _resolve_step_functions(step_functions)
        for stepper in steppers:  # MEEP samples t BEFORE the step, so the record opens here.
            stepper(self, "step")
        for index in range(total):
            self.step()
            if guard is not None:
                guard.update()
            if nonlinear is not None:
                nonlinear.update()
            for stepper in steppers:
                stepper(self, "step")
            completed = index + 1
            decayed = decay is not None and decay.update(self.time)
            if completed % interval == 0 or completed == total or decayed:
                if cancel_check is not None and cancel_check():
                    raise FdtdCancelled(f"FDTD run cancelled after {self.step_count} steps.")
                if progress_cb is not None:
                    progress_cb(completed, total)
            if decayed:
                self._require_folded_far_face_is_quiet()
                for stepper in steppers:
                    stepper(self, "finish")
                return time_module.perf_counter() - started
        if decay is not None:
            if decay.peak <= 0.0:
                measured = f"the probe never measured a nonzero |{decay_component}|"
            elif decay.ratio is None:
                measured = "no decay window completed"
            else:
                measured = f"last window ratio {decay.ratio:.3e}"
            raise RuntimeError(
                f"Fields had not decayed to {decay_by:g} of their peak at the max_time ceiling "
                f"{float(max_time):g} ({measured}, peak |{decay_component}| {decay.peak:.3e} at "
                f"{tuple(decay_point)}). Raise max_time, loosen decay_by, or check that the probe point "
                f"sees the field at all — a probe that never measures anything can never decay."
            )
        self._require_folded_far_face_is_quiet()
        for stepper in steppers:
            stepper(self, "finish")
        return time_module.perf_counter() - started

    def _prepare_decay_stop(
        self, window: float, decay_by: Any, decay_component: Any, decay_point: Any, max_time: Any
    ) -> tuple[_FieldDecayProbe, int]:  # Validate the decay arguments and build the probe plus its ceiling.
        """Build the field-decay probe and the ceiling step count, validating every argument.

        The ceiling is mandatory and is checked against the sources first: MEEP's
        ``_run_sources_until`` gates the criterion on ``last_source_time()``, so a
        source still emitting at ``max_time`` — a continuous source at its ~1e20
        default ``end_time``, say — could never satisfy it, and saying so here beats
        discovering it after the full ceiling has been stepped.
        """
        window_length = float(window)
        if not math.isfinite(window_length) or window_length <= 0.0:
            raise ValueError(
                f"until_after_sources is the trailing decay window in simulation time and must be "
                f"finite and positive, got {window!r}."
            )
        missing = [
            name
            for name, value in (
                ("decay_by", decay_by),
                ("decay_component", decay_component),
                ("decay_point", decay_point),
                ("max_time", max_time),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"A field-decay stop needs {missing} as well as until_after_sources: the probed "
                f"component and point, the decay fraction, and the ceiling the run may not exceed."
            )
        ratio = float(decay_by)
        if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError(
                f"decay_by must lie in (0, 1) — the fraction of the peak |field|^2 the trailing window "
                f"must fall to, MEEP's squared-magnitude convention — got {decay_by!r}."
            )
        component = str(decay_component)
        probe_point = _as_xyz("decay_point", decay_point)
        ceiling = float(max_time)
        if not math.isfinite(ceiling):
            raise ValueError(f"max_time must be a finite simulation time, got {max_time!r}.")
        if ceiling <= self.time:
            raise ValueError(
                f"max_time={ceiling} is not beyond the current simulation time {self.time}; nothing to run."
            )
        source_end = self._last_source_time()
        if source_end > ceiling:
            raise ValueError(
                f"The sources are still emitting at max_time={ceiling:g} (the last one ends at "
                f"{source_end:g}), so the fields can never be measured after the sources finish. "
                f"A continuous source runs to its end_time, ~1e20 by default: give it a finite "
                f"end_time, use a gaussian source, or raise max_time."
            )
        probe = _FieldDecayProbe(
            self.xp,
            self._point_magnitude_reader(component, probe_point),
            decay_by=ratio,
            window=window_length,
            start_time=self.time,
            source_end_time=source_end,
        )
        total = int(math.ceil((ceiling - self.time) / self.grid.dt - _STEP_TOLERANCE))
        return probe, total

    def _last_source_time(self) -> float:  # Latest time any source still emits (MEEP fields::last_source_time).
        """Return the time after which no source contributes, MEEP's ``last_source_time``.

        MEEP's ``gaussian_src_time::last_time`` is ``peak_time + cutoff``; the cutoff
        used here is the requested ``cutoff * width`` rather than the shrunk-to-
        representable one the source stores, which can only be later, never earlier.
        Continuous sources end at ``end_time`` — ~1e20 unless the caller set one, so a
        decay stop refuses them. A custom waveform is opaque: MEEP takes its
        ``end_time`` at face value and so does this, which for the ~1e20 default means
        a decay stop refuses it too rather than guessing when an arbitrary function
        goes quiet.

        A source whose stopping time cannot be determined raises: silently treating it
        as finished would let the decay criterion fire while it was still emitting, and
        silently treating it as endless would refuse a legitimate run.
        """
        last = 0.0
        for source in self._sources:
            if isinstance(source, GaussianPulsedSource):
                last = max(last, source.peak_time + source.cutoff * source.width)
            elif isinstance(source, (ContinuousSource, ExtendedSource)):
                last = max(last, float(source.end_time))
            elif isinstance(source, VolumeSource):
                last = max(last, _envelope_last_time(source.envelope))
            else:
                raise ValueError(f"Unknown source type {type(source).__name__}; cannot tell when it stops emitting.")
        return last

    def _point_magnitude_reader(self, component: str, point: tuple[float, float, float]):
        """Return a zero-argument callable giving ``|component|`` at the cell nearest ``point``.

        Built once per run for the decay probe, which samples one point every step:
        routing that through :meth:`get_field` would interpolate — and, without PML,
        allocate — the whole volume on every step, and every sample would cross back to
        the host on the GPU backend. The value returned is an ``xp`` scalar, so the
        probe's running maximum stays on the device between window boundaries.

        The sample is the nearest Yee point rather than MEEP's trilinear
        ``get_field_point``. Both are fixed linear functionals of the field, and the
        criterion is a ratio of one functional against itself over time, so the choice
        changes which point is watched, not whether the decay is detected.
        """
        if component not in FIELD_COMPONENTS:
            raise ValueError(f"Unknown decay_component {component!r}. Supported: {FIELD_COMPONENTS}.")
        shifts = yee_shifts(component)
        index = []
        for axis, axis_name in enumerate(("x", "y", "z")):
            lower, weight = self.grid.position_to_index(point[axis], axis_name, iyee_shift=shifts[axis])
            nearest = lower + (1 if weight >= 0.5 else 0)
            index.append(max(0, min(self.grid.shape[axis] - 1, nearest)))
        ix, iy, iz = index
        xp = self.xp
        stored = component in PRIMARY_COMPONENTS or self.pml is not None  # E/H are arrays only under PML.
        partner = ("D" if component[0] == "E" else "B") + component[1]

        def read_magnitude():  # |f| at the probe cell, without materializing a derived volume.
            if stored:
                return xp.abs(getattr(self.fields, component)[ix, iy, iz])
            if component[0] == "E":  # E = D / eps, evaluated at the one cell.
                inverse = self.fields.inverse_epsilon_for(component)
                return xp.abs(
                    getattr(self.fields, partner)[ix, iy, iz] * inverse[ix, iy, iz]
                )
            return xp.abs(getattr(self.fields, partner)[ix, iy, iz])  # mu = 1, so H is numerically B.

        return read_magnitude

    def meep_time(self) -> float:  # MEEP Simulation.meep_time(): simulation time, not wall clock.
        """Current simulation time — MEEP's ``Simulation.meep_time()`` (simulation.py:2607).

        The same number as the :attr:`time` property, carried under MEEP's name because
        MEEP's step-function protocol asks for it by that name: ``mp.Harminv`` calls
        ``sim.meep_time()`` to work out its own sample interval, so a step function
        written against MEEP finds what it expects here.
        """
        self._require_open()
        return self.time

    def get_field_point(self, component: str, point: Sequence[float]) -> complex:
        """One field component at one point — MEEP's ``Simulation.get_field_point``.

        This is ``fields::get_field(component, vec)`` (monitor.cpp:127): the eight
        surrounding Yee samples of that component, trilinearly interpolated onto
        ``point`` with ``grid_volume::interpolate``'s weights (vec.cpp:558-612,
        ``0.5*(1 -+ dv)`` per axis), accumulated in double precision as MEEP
        accumulates them. The return is always ``complex``, as MEEP's is, with a zero
        imaginary part in a real-fields run.

        A ``beta`` run (MEEP's ``special_kz``) additionally multiplies the accumulated
        sum by ``exp(i*2*pi*beta*z)`` for any sample asked for off the z = 0 plane,
        which is where the whole analytic third axis lives: ``monitor.cpp:139-140``,
        the ``gv.dim == D2 && loc.in_direction(Z) != 0`` branch.

        ``point`` is in USER COORDINATES — the full, unfolded cell, exactly as MEEP
        takes it. A point in the half a mirror plane folded away is reflected onto the
        stored quadrant and multiplied by that component's parity
        (``fields::get_field`` -> ``S.phase_shift``, monitor.cpp:145-157); a point past
        the end of a periodic axis is translated back by a lattice vector and
        multiplied by the Bloch factor (``locate_point_in_user_volume``). Both folds
        are exact, not approximate: the mirror maps the component's Yee lattice onto
        itself.

        WHERE THIS REFUSES AND MEEP DOES NOT. MEEP returns 0.0 for a point that lies
        in no chunk — off the end of a non-periodic axis, say — which is a field value
        indistinguishable from a real null. This raises instead. Nothing else about
        the sampling differs.

        COST. One call gathers eight elements; on the GPU backend that is one small
        kernel plus one device-to-host synchronization, so sampling every step of a
        long run is a real (if modest) cost — the same trade MEEP makes, where each
        call is an MPI all-reduce. The stencil (indices, weights, fold phase) is built
        once per (component, point) and cached; the arrays behind it are re-resolved
        on every call, so a :meth:`reset` or a change of storage mode cannot leave a
        stale reader behind.
        """
        self._require_open()
        return complex(self._point_field_reader(component, _as_xyz("point", point))())

    def _point_field_reader(self, component: str, point: tuple[float, float, float]):
        """The cached zero-argument sampler behind :meth:`get_field_point`."""
        key = (component, point)
        reader = self._point_readers.get(key)
        if reader is None:
            reader = self._build_point_field_reader(component, point)
            self._point_readers[key] = reader
        return reader

    def _build_point_field_reader(self, component: str, point: tuple[float, float, float]):
        """Resolve one point into a 2x2x2 slice, its weights, and the fold phase.

        Geometry only: nothing here captures a field array or a storage mode, both of
        which the returned closure looks up when it runs. That is deliberate — a
        reader cached across :meth:`reset` would otherwise hand back the pre-reset
        arrays, and a reader built before ``setup_pml`` would keep computing E from D
        after E became stored. Neither would raise; both would just be wrong.
        """
        if component not in FIELD_COMPONENTS:
            raise ValueError(f"Unknown field component {component!r}. Supported: {FIELD_COMPONENTS}.")
        shifts = yee_shifts(component)
        xp = self.xp
        axis_indices: list[Any] = []
        axis_weights: list[Any] = []
        for axis in range(3):
            indices, weights = self._point_stencil_along(component, axis, point[axis], shifts[axis])
            axis_indices.append(xp.asarray(indices, dtype=xp.int64))
            axis_weights.append(xp.asarray(weights, dtype=xp.complex128))
        gather = (
            axis_indices[0][:, None, None],
            axis_indices[1][None, :, None],
            axis_indices[2][None, None, :],
        )
        # MEEP accumulates the eight terms into a complex<double>; carrying the weights
        # in complex128 promotes the product the same way, so a point sample is not held
        # to the field's own single precision — and the fold phases (a mirror parity, a
        # Bloch factor) ride on the per-leg weights, where a wrapped leg needs its own.
        weights = (
            axis_weights[0][:, None, None]
            * axis_weights[1][None, :, None]
            * axis_weights[2][None, None, :]
        )
        primary = component in PRIMARY_COMPONENTS
        partner = ("D" if component[0] == "E" else "B") + component[1]
        # The special_kz readback phase, monitor.cpp:139-140: a 2-D run with beta != 0
        # carries exp(i*2*pi*beta*z) analytically, so a sample asked for at z != 0 is
        # the stored z = 0 value times that factor. MEEP applies it AFTER the
        # interpolation, over the whole accumulated sum, and only for a D2 grid; folded
        # into the weight product here, which is the same multiplication.
        if self.grid.beta != 0.0 and point[2] != 0.0:
            turn = 2.0 * math.pi * self.grid.beta * point[2]
            weights = weights * complex(math.cos(turn), math.sin(turn))  # std::polar(1.0, turn)

        def read_point():  # sum_i w_i * f_i over the 2x2x2 stencil, without materializing a volume.
            if primary or component[0] == "H" or self.fields.stores_E:
                # D/B are always arrays and H is numerically B (mu = 1); E is an array
                # only under PML, a susceptibility or a nonlinearity.
                name = component if (primary or component[0] == "E") else partner
                values = getattr(self.fields, name)[gather]
            else:  # E = D * inv_eps, evaluated on the eight cells rather than the volume.
                values = (
                    getattr(self.fields, partner)[gather]
                    * self.fields.inverse_epsilon_for(component)[gather]
                )
            return xp.sum(weights * values)

        return read_point

    def _point_stencil_along(
        self, component: str, axis: int, coordinate: float, iyee_shift: int
    ) -> tuple[list[int], list[complex]]:
        """One axis of the interpolation stencil: the cells to read and their weights.

        The weights are MEEP's ``grid_volume::interpolate`` (vec.cpp:571-581) — linear
        in the fractional index, ``1 - w`` on the cell below and ``w`` on the one above
        — times whatever phase the fold onto the stored array costs:

        * **Mirror.** A folded axis stores only ``x >= 0``; its plane is at the origin,
          which is where ``stepping._fill_symmetry_ghost_cells`` puts it when it writes
          ``cell 0 = parity * cell 2``. Reflecting maps the component's Yee lattice onto
          itself — the samples sit at ``(i - 1/2)*dx`` or ``(i - 1)*dx``, both sets
          symmetric about 0 — so the reflected point takes the same weights and carries
          ``fields.mirror_parity``, MEEP's ``S.phase_shift`` (monitor.cpp:145-157). A
          point past the FAR end of a folded axis has no image at all and raises, since
          a mirror is a boundary condition and not a lattice vector.
        * **Bloch.** A periodic axis is unbounded: the fractional index runs off either
          end and each leg is reduced modulo the stored count, carrying
          ``bloch_phase ** (leg // n)`` — the engine's one statement of that convention
          (``Grid.bloch_phase``: ``f(x + L) = phase * f(x)``). Per LEG rather than per
          point, because the common case is a point in the outer half cell whose two
          legs land on opposite faces; translating the whole point could not express it
          and would refuse an ordinary request.
        """
        if self.grid.is_mirrored(axis):
            parity: complex = 1.0 + 0.0j
            if coordinate < 0.0:
                coordinate = -coordinate
                parity = float(mirror_parity(component, axis, self.grid.mirror_phase(axis)))
            low, upper = self.grid.position_to_index(
                coordinate, ("x", "y", "z")[axis], iyee_shift=iyee_shift
            )
            if self.grid.stored_cells(axis) < 2:  # No neighbour to weigh a folded axis against.
                return [low], [parity]
            return [low, low + 1], [parity * (1.0 - upper), parity * upper]
        count = self.grid.stored_cells(axis)
        fractional = (
            (coordinate - self.grid.axis_origin(axis)) / self.grid.dx - 0.5 * iyee_shift
        )
        if not math.isfinite(fractional):
            raise ValueError(
                f"point coordinate {coordinate!r} on the {('x', 'y', 'z')[axis]} axis is not finite."
            )
        below = math.floor(fractional)
        upper = fractional - below
        wrap = self.grid.bloch_phase(axis)
        indices: list[int] = []
        weights: list[complex] = []
        for leg, weight in ((below, 1.0 - upper), (below + 1, upper)):
            images = leg // count  # Lattice vectors between this leg and its stored image.
            factor = 1.0 + 0.0j if wrap is None else complex(wrap) ** images
            indices.append(leg - images * count)
            weights.append(weight * factor)
        return indices, weights

    def get_field(self, component: str, cell_centered: bool = True) -> Any:  # Host NumPy copy of one field component.
        """Return a field component as a host NumPy array.

        ``cell_centered`` interpolates off the Yee positions to cell centres, matching
        MEEP's ``get_array`` sampling; the raw form returns the stored Yee array. The
        result is always a private copy, so callers cannot mutate engine state through
        it.

        The cell-centring is ``Fields.to_cell_center``, which carries the Bloch
        factor on the wrapped plane of each interpolated axis itself: the last plane
        averages with a cell one lattice vector away, and MEEP applies the same
        factor when it fills an array slice from a shifted chunk
        (loop_in_chunks.cpp: ``ph *= pow(eikna[d], ishift)``). Keeping that in
        ``Fields`` rather than here means ``to_meep_array`` and any direct reader
        get the same answer this method does. At k = 0 every factor is None and the
        multiply is skipped, so the readback is bit-identical to the pre-Bloch one.
        """
        self._require_open()
        if component not in FIELD_COMPONENTS:
            raise ValueError(f"Unknown field component {component!r}. Supported: {FIELD_COMPONENTS}.")
        if not cell_centered:
            array = self.fields.get_component(component)
        else:
            array = self.fields.to_cell_center(component)
        values = to_numpy(array)
        return values.copy() if values is array else values

    def set_field(self, component: str, array: Any):  # Overwrite one primary (D or B) field from an array.
        """Set a primary field array, e.g. to seed an initial condition.

        Only D and B are settable: E and H are derived (or, under PML, accumulated)
        and writing them would be discarded by the next constitutive update. The
        array is copied, never adopted — stepping writes in place, so adopting the
        caller's buffer would let one driver's run mutate whatever else references it
        (two drivers seeded from slices of one array would silently share memory).
        """
        self._require_open()
        if component not in PRIMARY_COMPONENTS:
            raise ValueError(
                f"Only the primary fields are settable, got {component!r}; supported: {PRIMARY_COMPONENTS}."
            )
        target = getattr(self.fields, component)
        probe = self.xp.asarray(array)
        if target.dtype.kind != "c":
            _reject_complex(
                probe,
                f"A {component} seed for a real-field run",
                "This run stores real fields, so the imaginary part has nowhere to go. Seed the real "
                "part, or build the driver with force_complex_fields=True.",
            )
        values = self.xp.array(probe, dtype=target.dtype, order="C", copy=True)
        if tuple(values.shape) != tuple(target.shape):
            raise ValueError(f"{component} array shape {tuple(values.shape)} does not match grid shape {tuple(target.shape)}.")
        setattr(self.fields, component, values)
        self.invalidate_fast_path()  # The primary array was REBOUND, not written in place.

    def get_epsilon(self, frequency: float | None = None, component: str = "Ez") -> Any:
        """Host NumPy copy of the permittivity volume, instantaneous or at one frequency.

        ``frequency=None`` returns eps_infinity as float32 — today's behaviour, and
        the only meaningful answer for a non-dispersive run. A frequency returns the
        complex ``mp.Simulation.get_epsilon(frequency)`` quantity::

            eps(f) = (eps_inf + sum_n sigma_n * chi_n(f)) * (1 + i*sigma_D/(2*pi*f))

        with ``chi_n`` the CONTINUUM susceptibility (python/geom.py
        ``_get_epsmu``), not the discrete one the engine steps — this is the material
        model as MEEP reports it, so it is a stepping-free parity target that catches
        a units error, a dropped 2*pi or a flipped Drude sign before a single field
        has moved.

        ``component`` selects which sigma registration is summed. It matters only for
        the per-component ``{'Ex':..., 'Ey':..., 'Ez':...}`` sigma form, where the
        three volumes genuinely differ by half a cell at an interface; the scalar and
        single-array forms alias one value into all three.
        """
        self._require_open()
        if component not in E_COMPONENTS:
            raise ValueError(f"component must be one of {E_COMPONENTS}, got {component!r}.")
        if frequency is None:
            material = self.fields.epsilon_for(component)
            values = to_numpy(material)
            return values.copy() if values is material else values
        probe = _positive_frequency(frequency)
        import numpy  # Local: the result is a host complex volume, never a device array.

        epsilon = numpy.asarray(
            to_numpy(self.fields.epsilon_for(component)),
            dtype=numpy.complex128,
        ).copy()
        for state in self.fields.polarizations:
            sigma = state.sigma[component]
            chi = state.susceptibility.chi1(probe)
            if isinstance(sigma, float):
                epsilon += sigma * chi
            else:
                epsilon += numpy.asarray(to_numpy(sigma), dtype=numpy.float64) * chi
        # The D component sharing this E component's Yee position is the one whose sigma
        # multiplies it — MEEP's conductivity is per component, and an absorber's is
        # graded, so reading Dx's volume for Ez would be a half-cell registration error.
        sigma_volume = self.fields.conductivity_for("D" + component[1])
        if sigma_volume is not None:
            sigma_d = numpy.asarray(to_numpy(sigma_volume), dtype=numpy.float64)
            # MEEP multiplies the WHOLE permittivity, susceptibilities included
            # (python/geom.py _get_epsmu); the factor lives in one place so this and
            # any analytic target cannot drift apart.
            epsilon = epsilon * (1.0 + (conductivity_factor(1.0, probe) - 1.0) * sigma_d)
        return epsilon

    # --- field-function reducers (src/integrate.cpp, src/energy_and_flux.cpp) ----------
    #
    # MEEP's instantaneous, read-time measures of the field: an arbitrary user integrand
    # over a volume, and the five ``*_in_box`` quantities built on it. They are not
    # monitors — nothing accumulates, nothing has to be registered before the run — so
    # every one of them is answered from whatever the field arrays hold right now.
    #
    # The volume is spelled ``center``/``size`` like every other region this driver takes;
    # :meth:`total_volume` is MEEP's ``fields.total_volume()`` and is the default.

    def total_volume(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """MEEP's ``fields::total_volume()`` as this driver's ``(center, size)``.

        NOT the declared cell, and not ``surroundings()`` either. fields.cpp:716-724::

            volume gv0 = gv.interior();
            volume v = gv0;
            for (int n = 1; n < S.multiplicity(); ++n) v = v | S.transform(gv0, n);

        ``gv`` is the STEPPED grid volume — halved on every mirrored axis — and
        ``interior()`` (vec.cpp:289-291) is ``[little_corner, big_corner - 2]`` in doubled
        units, i.e. ONE CELL SHORT of the cell at the top. So an unfolded axis of a
        10-unit cell at resolution 20 runs ``[-5.0, +4.95]``, not ``[-5, +5]``: the
        declared cell measured 0.05 wider than MEEP's own "total volume", and using the
        declared one made ``sim.modal_volume_in_box()`` disagree with
        ``sim.fields.modal_volume_in_box(sim.fields.total_volume())`` by 3.1e-06 — a
        failure of ``test_simulation.TestSimulation.test_modal_volume_in_box``, whose
        whole point is that the two spellings agree.

        A MIRRORED axis is then unioned with its reflection about the plane, which is what
        makes the folded 10-unit axis ``[-4.95, +4.95]`` — symmetric, and one cell short at
        BOTH ends. Measured against ``sim.fields.total_volume()`` on five cells: folded
        10x10 at resolution 20, the same cell unfolded, 16x8 at resolution 10, a 2 x 1.5
        cell whose y count is ODD, and a 1-D z cell — exact on every axis of every one.
        """
        self._require_open()
        centers, sizes = [], []
        for axis in range(3):
            if self.grid.is_invariant(axis):
                centers.append(0.0)
                sizes.append(0.0)
                continue
            low = self.grid.origin_doubled(axis)
            high = low + 2 * self.grid.stored_cells(axis) - 2
            if self.grid.is_mirrored(axis):
                reach = max(abs(low), abs(high))
                low, high = -reach, reach
            if self.grid.is_axis(axis):  # Dcyl clips r at the axis (fields.cpp:722).
                low = max(low, 0)
            low = low * 0.5 * self.grid.dx
            high = high * 0.5 * self.grid.dx
            centers.append(0.5 * (low + high))
            sizes.append(high - low)
        return tuple(centers), tuple(sizes)

    def _reducer_region(self, center, size):  # (center, size), defaulting to the whole cell.
        if center is None and size is None:
            return self.total_volume()
        if center is None or size is None:
            raise ValueError(
                "a reducer volume needs both a center and a size, or neither (the whole cell); "
                f"got center={center!r}, size={size!r}."
            )
        center = tuple(float(value) for value in center)
        size = tuple(float(value) for value in size)
        if len(center) != 3 or len(size) != 3:
            raise ValueError(
                f"a reducer volume takes a three-component center and size, got "
                f"center={center!r}, size={size!r}."
            )
        return center, size

    def integrate_field_function(self, components, func, center=None, size=None) -> complex:
        """MEEP's ``Simulation.integrate_field_function`` — ``fields::integrate``.

        ``func(location, *values)`` is called once per grid point ``loop_in_chunks``
        visits, with ``location`` an ``(x, y, z)`` tuple and one Python ``complex`` per
        entry of ``components``, in order. ``"Dielectric"`` and ``"Permeability"`` are
        accepted alongside the twelve field components and are NOT field reads — see
        :func:`~.dft._material_trace`.
        """
        self._require_open()
        region_center, region_size = self._reducer_region(center, size)
        integral, _ = dft_integrate_field_function(
            self.fields, self.grid, components, func, region_center, region_size)
        return integral

    def max_abs_field_function(self, components, func, center=None, size=None) -> float:
        """MEEP's ``Simulation.max_abs_field_function`` — ``fields::max_abs``.

        The maximum of ``|func|`` over the same points :meth:`integrate_field_function`
        sums, UNWEIGHTED and taken before the integration weight (integrate.cpp:123). It
        shares that method's single pass, because MEEP's ``max_abs`` is that pass
        (integrate.cpp:221-227).
        """
        self._require_open()
        region_center, region_size = self._reducer_region(center, size)
        _, maximum = dft_integrate_field_function(
            self.fields, self.grid, components, func, region_center, region_size)
        return maximum

    def electric_energy_in_box(self, center=None, size=None) -> float:
        """MEEP's ``fields::electric_energy_in_box`` — energy_and_flux.cpp:85-89.

        ``sum over d of 0.5 * Re(integrate(2, {E_d, D_d}, conj(f0)*f1))``. Each pair
        shares one Yee shift, so each integral runs on ITS OWN raw lattice; this is the
        measure that cannot be rebuilt from a cell-centred array.
        """
        return self._energy_in_box(("Ex", "Ey", "Ez"), "D", center, size)

    def magnetic_energy_in_box(self, center=None, size=None) -> float:
        """MEEP's ``fields::magnetic_energy_in_box`` — energy_and_flux.cpp:91-95.

        The mirror of :meth:`electric_energy_in_box` over the (H_d, B_d) pairs. Like
        MEEP's, it does NOT synchronize the magnetic fields on its own — only
        :meth:`field_energy_in_box` and :meth:`flux_in_box` do.
        """
        return self._energy_in_box(("Hx", "Hy", "Hz"), "B", center, size)

    def _energy_in_box(self, family, primary: str, center, size) -> float:
        self._require_open()
        region_center, region_size = self._reducer_region(center, size)
        total = 0.0
        for component in family:
            total += 0.5 * _dot_pair_reduction(
                self.fields, self.grid, component, primary + component[1],
                region_center, region_size)
        return total

    def field_energy_in_box(self, center=None, size=None) -> float:
        """MEEP's ``fields::field_energy_in_box`` — energy_and_flux.cpp:54-59.

        Electric plus magnetic, with the magnetic half measured on SYNCHRONIZED fields
        and the electric half on the stored ones. The asymmetry is MEEP's: B and H are
        half a step behind E and D, so only the magnetic half needs the half-step
        average, and taking it changes the number (measured 2.5e-02 relative on
        test_wvg_src's near-field flux).
        """
        self._require_open()
        self.synchronize_magnetic_fields()
        try:
            magnetic = self.magnetic_energy_in_box(center, size)
        finally:
            self.restore_magnetic_fields()
        return self.electric_energy_in_box(center, size) + magnetic

    def flux_in_box(self, direction: int, center=None, size=None) -> float:
        """MEEP's ``fields::flux_in_box`` — energy_and_flux.cpp:189-222.

        The integral of ``Re[E x H]`` along ``direction`` (0 = x, 1 = y, 2 = z) over the
        box, on synchronized magnetic fields. Each (E, H) pair straddles two lattices, so
        every one of these integrals takes the CENTERED branch.
        """
        self._require_open()
        self.synchronize_magnetic_fields()
        try:
            return self._flux_in_box_wrong_h(direction, center, size)
        finally:
            self.restore_magnetic_fields()

    # energy_and_flux.cpp:193-206 — which (E, H) pair carries which sign, per direction.
    _FLUX_PAIRS = {
        0: (("Ey", "Hz"), ("Ez", "Hy")),
        1: (("Ez", "Hx"), ("Ex", "Hz")),
        2: (("Ex", "Hy"), ("Ey", "Hx")),
    }

    def _flux_in_box_wrong_h(self, direction: int, center, size) -> float:
        """``flux_in_box_wrongH``: the same integral WITHOUT the half-step synchronization."""
        if direction not in self._FLUX_PAIRS:
            raise ValueError(f"flux direction must be 0 (x), 1 (y) or 2 (z), got {direction!r}.")
        region_center, region_size = self._reducer_region(center, size)
        total = 0.0
        for index, (electric, magnetic) in enumerate(self._FLUX_PAIRS[direction]):
            total += _dot_pair_reduction(
                self.fields, self.grid, electric, magnetic, region_center, region_size
            ) * (1 - 2 * index)
        return total

    def electric_energy_max_in_box(self, center=None, size=None) -> float:
        """MEEP's ``fields::electric_energy_max_in_box`` — energy_and_flux.cpp:254-283.

        ``0.5 * max|E . D|`` over the grid points, through ``max_abs`` on all six of
        (Ex, Ey, Ez, Dx, Dy, Dz) at once — which do NOT share a Yee shift, so this one
        runs on the CENTERED lattice while :meth:`electric_energy_in_box` runs on three
        raw ones. MEEP's own comment (energy_and_flux.cpp:245-253) says why: averaging a
        discontinuous product is what makes a modal volume noisy, and it accepted that
        rather than pay for a better average.
        """
        self._require_open()
        region_center, region_size = self._reducer_region(center, size)
        _, maximum = dft_integrate_field_function(
            self.fields, self.grid,
            ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz"),
            lambda _location, ex, ey, ez, dx, dy, dz: (
                (ex.conjugate() * dx).real + (ey.conjugate() * dy).real
                + (ez.conjugate() * dz).real
            ),
            region_center, region_size,
        )
        return maximum * 0.5

    def modal_volume_in_box(self, center=None, size=None) -> float:
        """MEEP's ``fields::modal_volume_in_box`` — energy_and_flux.cpp:288-290.

        ``electric_energy_in_box / electric_energy_max_in_box``, the Purcell-effect modal
        volume. An UNSTEPPED run answers ``nan`` here and so does MEEP: both halves are
        exactly zero. That is reproduced rather than guarded, because a guard would hand
        back a finite modal volume for a simulation that has no field in it.
        """
        self._require_open()
        region_center, region_size = self._reducer_region(center, size)
        energy = self.electric_energy_in_box(region_center, region_size)
        peak = self.electric_energy_max_in_box(region_center, region_size)
        if peak == 0.0:
            # C++ `0.0/0.0` is a quiet NaN and Python's is a ZeroDivisionError. MEEP's
            # answer for a field-free simulation is the NaN, and it is an answer a test
            # can assert against (`assertAlmostEqual(x, nan)` FAILS); raising instead
            # would turn a reproducible number into an exception nobody asked for.
            return float("nan") if energy == 0.0 else math.copysign(float("inf"), energy)
        return energy / peak

    # --- the magnetic half-step (energy_and_flux.cpp:146-178) --------------------------

    _SYNC_FIELDS = ("Bx", "By", "Bz", "Hx", "Hy", "Hz")
    # MEEP's BACKUP/RESTORE list for the half-step, energy_and_flux.cpp:96-140. The
    # f_bfast entries are NOT optional and NOT cosmetic: :113 backs them up and :130
    # restores them, so a synchronization that advanced B without putting them back
    # would leave the BFAST IIR one step ahead of the fields — and because that
    # recursion is marginally stable (its homogeneous mode is (-1)^n, undamped
    # forever), the error would never decay. One flux call would poison the whole run.
    _SYNC_AUXILIARY = ("fu_Bx", "fu_By", "fu_Bz",
                       "f_cond_Bx", "f_cond_By", "f_cond_Bz",
                       "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
                       "f_w_Hx", "f_w_Hy", "f_w_Hz")

    def synchronize_magnetic_fields(self) -> None:
        """Advance B and H half a step and average with where they were — MEEP's own.

        B and H are stored half a time step behind E and D, so any product of the two —
        a Poynting flux, a total field energy — is a product of fields at different
        times. MEEP's answer (energy_and_flux.cpp:146-172) is to take the NEXT magnetic
        half-update and average it with the current arrays, giving B and H at the
        electric fields' own time, then put the originals back.

        The half-update is exactly the B side of :meth:`step`, in that method's order,
        because it is the same code in MEEP: ``calc_sources(time())``, ``step_db``,
        ``step_source``, ``step_boundaries``, then ``calc_sources(time() + dt/2)`` and
        ``update_eh(H_stuff)``. The PML and conductivity auxiliaries the B ladder
        advances are backed up and RESTORED but never averaged, which is MEEP's
        ``backup_component`` / ``average_with_backup`` split (:96-140): only ``f``
        itself is averaged.

        REFCOUNTED like MEEP's, so nesting is a no-op: ``field_energy_in_box`` calls
        this and then ``magnetic_energy_in_box``, and a caller who has already
        synchronized must not get a second half-step.

        MEASURED on test_wvg_src's own cell (16 x 8, resolution 10, complex fields,
        eigenmode ContinuousSource at 0.15, until=200 = 4000 steps exactly, steady-state
        CW so no turn-on tail): the near-field flux is -1.775e-03 synchronized against
        -1.731e-03 unsynchronized, 2.5e-02 apart, and MEEP's published constant is the
        synchronized one to 3.6e-08.
        """
        self._require_open()
        self._synchronized_magnetic += 1
        if self._synchronized_magnetic > 1:
            return
        xp = self.xp
        backup = {}
        for name in self._SYNC_FIELDS + self._SYNC_AUXILIARY:
            array = getattr(self.fields, name, None)
            if array is not None:
                backup[name] = xp.array(array, copy=True)
        self._magnetic_backup = backup
        # The SAME dispatch the step's magnetic half takes, for the same reason it
        # takes it: this is `step`'s B/H half verbatim, and a run where one
        # sub-step sometimes launches and sometimes does not is a composition no
        # gate measured. It does NOT re-plan — a stale plan reads as no plan — and
        # the backup above copies OUT and restores in place, so the plan's cached
        # device views survive the synchronization.
        fast = None if self._fast_path_stale else self._fast_path
        magnetic = [source for source in self._sources if source.field_type == FIELD_TYPE_B]
        for source in magnetic:
            getattr(source, "withdraw", _no_withdraw)(self.fields)
        if fast is None or not fast.dispatch("step_B", self.fields):
            step_B(self.fields, self.pml)
        for source in magnetic:
            source.inject(self.fields, self.time)  # MEEP's step_source(B_stuff).
        # step_boundaries(B_stuff) — after the currents land, exactly as `step` runs
        # it, INCLUDING the two fill consults. A half-step that filled on the array
        # path while the step filled on kernels would be the mixed composition this
        # method's own comment above refuses.
        if fast is None or not fast.dispatch("fill_B", self.fields):
            fill_symmetry_bc_B(self.fields)
        zero_metal_B(self.fields)
        if fast is None or not fast.dispatch("fill_folded_far_ghosts_B", self.fields):
            fill_folded_far_ghosts_B(self.fields)
        # THE ONE CONSULT ON THIS PATH THAT IS NOT ``step``'s, and the name is the
        # difference. ``update_H`` is the last magnetic slot and therefore the seam
        # where a product may reach forward into the ELECTRIC half — a weld spanning
        # ``update_H`` and ``step_D`` is correct in ``step`` and a silent wrong answer
        # here, because the backup above is ``_SYNC_FIELDS`` + ``_SYNC_AUXILIARY`` and
        # ``D``, ``fu_D`` and ``f_cond_D`` are in NEITHER. Such a product would advance
        # the electric state inside a half-step nothing can undo, on every
        # ``flux_in_box`` and ``field_energy_in_box`` call, and the run would carry it
        # to the end.
        #
        # So this site asks for ``update_H_synchronize`` and the plan answers only if
        # what it holds at ``update_H`` stays inside the magnetic half-step
        # (``fastpath.SYNC_PATH_SLOTS``); anything wider answers False and the array
        # ``update_H`` below runs, which advances the magnetic constitutive and nothing
        # else. A NAME rather than an argument, for the reason
        # ``fastpath.FAR_FILL_OWNERS`` gives: every other implementer of this consult —
        # the ``DispatchAdapter`` shims, and a frozen copy inside a gate's result
        # directory — answers False to a name it does not carry, which is the array
        # path, where a second method would raise ``AttributeError``.
        if fast is None or not fast.dispatch(SYNC_UPDATE_H_PASS, self.fields):
            update_H(self.fields, self.pml)  # update_eh(H_stuff) at time() + dt/2.
        for name in self._SYNC_FIELDS:
            array = getattr(self.fields, name, None)
            if array is not None and name in backup:
                # A WHOLE-VOLUME HOST WRITE, declared as one. This runs once per
                # energy or flux query, not per step, so the whole copy it costs
                # under a held residency is the "one full bracket per query" the
                # residency plan already concedes -- and without the declaration the
                # in-place average raised on a sealed array (measured on every held
                # case of the 2026-09-23 preflight, through the gate's
                # second-consult control).
                host_writes.acquire(array)
                array *= 0.5
                array += 0.5 * backup[name]

    def restore_magnetic_fields(self) -> None:
        """Undo :meth:`synchronize_magnetic_fields` — MEEP's ``restore_magnetic_fields``.

        Refcounted: only the outermost call restores, and a call with nothing
        outstanding is a no-op, exactly as energy_and_flux.cpp:174-186.
        """
        self._require_open()
        if not self._synchronized_magnetic:
            return
        self._synchronized_magnetic -= 1
        if self._synchronized_magnetic:
            return
        for name, saved in (self._magnetic_backup or {}).items():
            array = getattr(self.fields, name, None)
            if array is not None:
                host_writes.acquire(array)[...] = saved
        self._magnetic_backup = {}

    def susceptibility_report(self) -> tuple[str, ...]:  # Every term's discrete-pole verdict, for a log.
        """One :class:`~.dispersion.StabilityReport` line per added susceptibility."""
        self._require_open()
        return tuple(
            term.stability(self.grid.dt).describe() for term, _ in self._susceptibilities
        )

    def bytes_per_cell(self) -> int:  # Storage this run costs per cell, susceptibilities included.
        """Per-cell memory budget, for a dispatch predicate to consult BEFORE allocating.

        A six-term metal fit costs 240-360 B/cell on top of a 56 B/cell (no PML) or
        200 B/cell (PML) baseline, so a job that sizes itself on the field arrays
        alone can dispatch to GPU, OOM, and fall back only after the setup has been
        paid for.
        """
        self._require_open()
        return self.fields.bytes_per_cell()

    def reset(self):  # Return fields, monitors, and the clock to their initial state.
        """Clear the fields, the monitors, the sources' own state, and the clock.

        An integrated source carries the dipole offset it has already applied across
        steps, so it has to be reset with the clock: without this a re-run starts by
        subtracting the increment from the *old* final dipole to the new initial one,
        which produces a wrong but entirely plausible field.
        """
        self._require_open()
        self.fields.reset()
        self.invalidate_fast_path()  # Field arrays may have been reallocated with the state.
        self.step_count = 0
        for source in self._sources:
            source_reset = getattr(source, "reset", None)
            if source_reset is not None:
                source_reset()
        for monitor in self._dft_monitors:
            monitor.reset()
        for monitor in self._flux_monitors:
            monitor.reset()

    def close(self):  # Drop array references so the backend allocator can reclaim the memory.
        """Release every array this driver owns; the driver is unusable afterwards.

        The device allocator keeps freed blocks in its pool, so the pool is drained
        too when running on GPU — a long FDTD run otherwise holds gigabytes past the
        end of the job.
        """
        if self.fields is not None:
            self.fields.polarizations.clear()  # P/P_prev are the largest arrays a dispersive run holds.
            if self.fields.scratch is not None:
                self.fields.scratch.clear()  # The step loop's pooled temporaries, up to ~8 volumes.
        self.fields = None
        self.pml = None
        self.invalidate_fast_path()  # A plan holds device views; drop them with everything else.
        self._chi2 = 0.0  # Drop any chi2/chi3 volume the caller handed over.
        self._chi3 = 0.0
        self._susceptibilities.clear()
        self._sources.clear()
        self._dft_monitors.clear()
        self._flux_monitors.clear()
        self._pending_monitor_regions.clear()
        self._automatic_decimation_monitors.clear()
        self._point_readers.clear()  # These hold small index arrays on the device.
        self._closed = True
        memory_pool = getattr(self.xp, "get_default_memory_pool", None)
        if memory_pool is not None:
            memory_pool().free_all_blocks()

    @property
    def time(self) -> float:  # Current simulation time (MEEP: t * dt).
        return self.step_count * self.grid.dt

    @property
    def shape(self) -> tuple[int, int, int]:  # Stored grid shape (halved on symmetry axes).
        return (self.grid.nx, self.grid.ny, self.grid.nz)

    @property
    def dt(self) -> float:  # Time step, courant / resolution.
        return self.grid.dt

    @property
    def dx(self) -> float:  # Grid spacing, 1 / resolution.
        return self.grid.dx

    @property
    def k_point(self) -> tuple[float, float, float]:  # Bloch wavevector, MEEP units of 2*pi/distance.
        return self.grid.k_point

    @property
    def boundaries(self) -> tuple[tuple[str, str], ...]:  # Declared (low, high) condition per axis.
        """The normalized outer-boundary table, MEEP's ``fields::boundaries`` for this run.

        One ``(low, high)`` pair per axis in x, y, z order. A mirror plane is NOT
        reported here — it is a separate mechanism in MEEP too — so a folded metallic
        axis reads ``("metallic", "metallic")`` and ``grid.is_mirrored`` tells you it
        is also folded.
        """
        return self.grid.boundaries

    def _resolve_monitor_regions(self):  # Apply deferred DFT-monitor region defaults once stepping starts.
        """Give every region-less DFT monitor the PML interior, face by face.

        A layer requested as a scalar goes through ``DFTMonitor.set_region_from_pml``
        unchanged, so every default region measured under the uniform PML is the same
        region that method has always produced, whatever it decides about a folded
        axis's lower face.

        A per-face layer goes through ``DFTMonitor.set_region_from_faces`` instead,
        which trims each face by its own thickness: putting the scalar entry point's
        single thickness on a periodic x axis would drop live interior, and on a cell
        only a few cells wide across (a grating period) it would leave no region at
        all.

        Every region — the explicit volumes as much as the defaults — is resolved
        against the boundaries in force NOW, not the ones the monitor was stamped
        with when it was added. ``add_dft_monitor`` records the run's wrap flags at
        add time, and a PML installed afterwards turns a wrapping face into an
        absorbing one; carrying the stale answer here made the same request resolve
        to different cells depending on the order the two calls were made in, which
        is precisely what deferring the region to step time exists to prevent.
        Measured on a 2x2x3 cell at resolution 10 with ``{"z": 6}`` and a monitor
        centred at z = 1.3 with size 0.6: PML first registered z cells 24..30, monitor
        first registered 24..32 — two extra cells continued past a face this absorber
        terminates, carrying field from the far end of the cell (inside the opposite
        layer) under labels a caller would read as the +z continuation.
        """
        pending = self._pending_monitor_regions
        self._pending_monitor_regions = []
        periodic = self._monitor_periodic_axes()
        for monitor, center, size in pending:
            if center is not None and size is not None:
                monitor.set_region_from_volume(center, size, periodic=periodic)
                continue
            # The default regions carry no ladder and never gather, but `periodic`
            # still decides whether a far-face Yee average is taken from the wrap, so
            # it is refreshed on the same rule that chose the region.
            monitor.periodic = periodic
            if self.pml is None:
                continue
            if self.pml.uniform_thickness is not None:
                monitor.set_region_from_pml(self.pml.uniform_thickness)
            else:
                monitor.set_region_from_faces(self.pml.thickness_by_face)

    def _require_open(self):  # Guard every entry point against use after close().
        if self._closed:
            raise RuntimeError("This FdtdDriver has been closed; build a new one to run again.")

    def __repr__(self) -> str:
        backend = ("numpy reference" if self.gpu is None
                   else f"{self.xp.__name__}, gpu={self.gpu}")
        symmetry = f", symmetry={self.grid.symmetry}" if self.grid.symmetry else ""
        bloch = f", k_point={self.grid.k_point}" if self.grid.has_bloch else ""
        state = ", closed" if self._closed else ""
        return (
            f"FdtdDriver(shape={self.grid.shape}, backend={backend}, steps={self.step_count}"
            f"{symmetry}{bloch}{state})"
        )
