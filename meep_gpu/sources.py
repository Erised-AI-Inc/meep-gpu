"""
Current sources for the standalone backend-neutral FDTD engine.

This module owns everything between a user-level source specification (component,
frequency, center, size, amplitude, optional spatial amplitude function) and the
per-timestep current the stepping loop subtracts from the primary fields. Four
source shapes are provided: `ContinuousSource` (CW point source),
`ExtendedSource` (CW sheet/volume source), `GaussianPulsedSource` (the same
spatial machinery with a Gaussian temporal envelope), and `VolumeSource` — the
general form, which takes any electric *or* magnetic component and any temporal
envelope object. All four place their current through one function,
`_source_geometry`: a translation of MEEP's `add_volume_source` and
`loop_in_chunks` -- Simpson boundary weights, owned-corner clipping, and one
pass per lattice shift wherever the request reaches a periodic face, each with
its Bloch factor (loop_in_chunks.cpp:390-418 keys that loop on the boundary
being Periodic, so a mirror-folded axis with a periodic outer face takes it too).
`_src_vol_pieces` cuts each shift's points per chunk of a division into MEEP's
`src_vol` pieces; `_summed_table` is the source's own table, every cell once.
`create_gaussian_beam_source` is a factory that dresses an `ExtendedSource` with
a converging Gaussian-beam amplitude function.

The temporal waveform is a separate object so it can be reused across the two
field families and replaced wholesale: `ContinuousEnvelope`, `GaussianEnvelope`
and `CustomEnvelope` translate MEEP's `continuous_src_time`, `gaussian_src_time`
and `custom_src_time`. Each carries MEEP's `is_integrated` flag.

WHERE THE CURRENT COMES FROM. A lifted run forms every current and dipole through
the MEEP simulation's OWN `src_time` object: the lift binds it to the source (or to
a `VolumeSource`'s envelope) with `bind_meep_src_time`, and `dipole(t)` /
`current(t, dt)` then return `src_time::dipole` / `src_time::current` (meep.hpp,
sources.cpp) as MEEP's build evaluates them. The words of a carrier depend on which
libm entry the C++ compiler lands on (`__sincos_stret` on Darwin, glibc `sincos` /
`cexp` on Linux) and on the C++ runtime's complex division, none of which this
package can see; MEEP's own object answers for all three. The package's own
evaluators (`_continuous_dipole`, `_gaussian_dipole`, `_custom_dipole`) serve a
driver built without MEEP, in MEEP's operation order, and carry no claim of
identity with MEEP's words. `current` is MEEP's forward difference
`(dipole(t + dt) - dipole(t)) / dt` (meep.hpp:983-985), formed per part.

THE DEPOSIT LIST IS MEEP'S `src_vol` LIST. MEEP splits each source into one piece
per (lattice shift, chunk) it touches and appends the piece to that chunk's list,
unless an entry of the list has the same component, the same shared `src_time`
instance and the same index vector, in which case the amplitudes are summed into it
in double (`fields_chunk::add_source`, fields.cpp:576-585; `src_vol::combinable` /
`add_amplitudes_from`, sources.cpp:179-188; equal src_times share one instance,
`src_time::add_to` / `src_times_equal`, sources.cpp:33-52). The merged pair then
deposits once, rounding once; two pieces that share a cell without sharing their
index vector -- the two ends of a request across a periodic seam -- deposit that cell
twice, rounding twice. :func:`bind_src_vols` builds the list from the driver's sources
and its chunk division and gives each source the entries it created, with the summed
amplitudes; the source deposits them in order.

Inputs: a `Grid` (which supplies the array module `xp`, resolution, dt, cell
counts and symmetry flags) plus the source specification. Output: a set of grid
points with precomputed complex128 amplitudes, plus `dipole(t)` / `current(t, dt)`
waveforms; `inject(fields, time)` applies `f -= real((amp * current(t)) * dt)` at
those points in double precision, rounding each stored word once, which is MEEP's
`step_source` (step.cpp:295-317; `_deposit` carries the arithmetic).

Role in the step sequence — MEEP drives the two field families at different
half-steps (step.cpp:64-100), and `field_type` says which slot a source belongs
in:

    'D' (Ex/Ey/Ez, Dx/Dy/Dz): injected after `step_D`, before `update_E`,
        evaluated at `time() + 0.5*dt`.
    'B' (Hx/Hy/Hz, Bx/By/Bz): injected after `step_B`, before `update_H`,
        evaluated at `time()` — half a step earlier. Getting this wrong is a
        pure phase error, invisible to any magnitude-only comparison.

The `amp_factor = 1/(-2*pi*i*f)` convention makes the user-supplied amplitude a
*current* amplitude, since `current()` is the forward difference of `dipole()`
(MEEP `src_time::current`).

An `is_integrated` source is MEEP's polarization source. MEEP's `step_source` skips
it (step.cpp:300), and `update_eh` builds `f_minus_p`, a copy of D (or B), subtracts
`real(amp[j] * dipole)` from it per point in double, in list order, and computes E
(or H) from that copy; D and B themselves never hold the dipole
(update_eh.cpp:66-140). The dipole time is `time() + dt` for E and `time() + dt/2`
for H (step.cpp:74, :105). :func:`f_minus_p_begin` and :func:`f_minus_p_end` are that
copy, done in place around the driver's constitutive consult: the words at the
integrated points are saved, replaced by `f_minus_p`'s words (and imaged into the
ghost and wall cells by the same boundary passes the driver ran, so a pointwise
constitutive update at a ghost reads the image of `f_minus_p`), and restored after
the update, with the boundary passes run again so every ghost holds MEEP's D. An
integrated source's `inject` deposits nothing. This replaces a standing offset that
was subtracted into D/B in the injection slot and withdrawn before the next curl:
`(f - d) + d` is not `f` in float32, and MEEP never rounds D through the dipole
(phase 0 measured the offset form against MEEP's own formula at 137 of 280
point-steps where MEEP's formula holds at 280 of 280).

Real-valued fields — MEEP's default — are injected into as well. MEEP's
`step_source` writes `f[c][0][i] -= real(A)` and only touches the imaginary plane
`f[c][1][i]` when the run is complex (step.cpp:295-317), so the real-field
current is the real part of the SAME product `(amp[j] * current(t)) * dt`
this module already forms, on the same float64 operands. That is not an
approximation of the complex run: the
curl, the constitutive relation and the PML recurrence all carry real
coefficients, so a real run reproduces the real part of the complex one exactly
— measured 0.0 (bitwise) between CPU MEEP's own two modes, and 0.0 here (see
`test_injection_into_real_fields_takes_the_real_part_of_the_current` in
test_sources.py and `test_real_stepping_is_the_real_part_of_complex_stepping`
in test_stepping.py; the absolute anchor is
`test_real_field_run_matches_cpu_meeps_default_mode`, against MEEP run in its
real default).

That holds for a *spatial* amplitude with phase too — a complex `amplitude`, an
`amp_func` returning complex values, a Gaussian beam's wavefront curvature, an
oblique mode's in-plane ramp. `real(A)` is taken there with no condition
attached, which is MEEP's own step.cpp:307 and, measured, MEEP's own answer:
`_deposit` carries the table. This module used to refuse that pairing
on the grounds that `Re(a)Re(s) - Im(a)Im(s)` re-reads a spatial phase as a
per-point time shift; it does, and the resulting field is still exactly the real
part of the complex run, so the refusal only declined runs MEEP completes. The
temporal current stays complex in both modes and its real part is taken by
design: that IS the real-field convention, not a dropped term.

Units are MEEP natural units throughout (c = 1, frequency = 1/wavelength); the
micrometre conversion belongs to the bridge layer above. Real field arrays are
float32, complex field arrays complex64; the deposition amplitudes and the deposit
arithmetic are double, as MEEP's are. No array is created here except through
`grid.xp`, so the module runs unchanged on NumPy and CuPy.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .grid import METALLIC
from . import host_writes

if TYPE_CHECKING:
    from .fields import Fields
    from .grid import Grid

# Electric components, mapped to the D array they write into. `ContinuousSource`,
# `ExtendedSource` and `GaussianPulsedSource` drive the D update only and reject
# everything else rather than silently redirecting it; magnetic components go
# through `VolumeSource`.
_D_ARRAY_FOR_COMPONENT = {
    'Ex': 'Dx', 'Ey': 'Dy', 'Ez': 'Dz',
    'Dx': 'Dx', 'Dy': 'Dy', 'Dz': 'Dz',
}

# Magnetic components, mapped to the B array they write into. MEEP's src_vol
# folds B components onto their H names (sources.cpp:168-174) and step_source
# then writes the B array, so both spellings name the same current.
_B_ARRAY_FOR_COMPONENT = {
    'Hx': 'Bx', 'Hy': 'By', 'Hz': 'Bz',
    'Bx': 'Bx', 'By': 'By', 'Bz': 'Bz',
}

_ARRAY_FOR_COMPONENT = {**_D_ARRAY_FOR_COMPONENT, **_B_ARRAY_FOR_COMPONENT}

# Which half-step family a component belongs to; see the module docstring for the
# two injection slots this selects between.
FIELD_TYPE_D = 'D'
FIELD_TYPE_B = 'B'
_FIELD_TYPE_FOR_COMPONENT = {
    **{name: FIELD_TYPE_D for name in _D_ARRAY_FOR_COMPONENT},
    **{name: FIELD_TYPE_B for name in _B_ARRAY_FOR_COMPONENT},
}

_MEEP_SLOWNESS = 3.0  # MEEP continuous_src_time default turn-on slowness.
_MEEP_ENDLESS = 1e20  # MEEP continuous_src_time default end_time (effectively never).
_MEEP_CUSTOM_START = -1e20  # MEEP CustomSource default start_time (turn on at -infinity).
_MEEP_CUSTOM_END = 1e20     # MEEP CustomSource default end_time (never turn off).


def _amp_factor(frequency: float) -> complex:  # MEEP sources.cpp:104 (1.33.0) — makes the current amplitude ~1 at the peak.
    """``1/(-2*pi*i*f)``, with the frequency's own sign.

    MEEP stores the carrier frequency raw — ``gaussian_src_time`` keeps ``freq = f``
    with no sign restriction (sources.cpp:72-96) — and both the amplitude
    correction here and the carrier ``exp(-i*2*pi*f*t)`` simply take the sign with
    it, so a NEGATIVE frequency is the conjugate carrier. That is a real MEEP
    usage, not an edge case: the plus/minus-omega superposition
    (``dipole_in_vacuum_cyl_off_axis.py`` drives ``GaussianSource(-f)`` beside
    ``GaussianSource(+f)``) builds a real current from the pair and separates the
    two rotations with monitors at ``+f`` and ``-f``. Zero is refused loudly:
    MEEP's own arithmetic divides by it and steps an infinite current without a
    word.
    """
    if not math.isfinite(frequency) or frequency == 0.0:
        raise ValueError(
            f"Source frequency must be finite and nonzero, got {frequency} "
            f"(negative selects MEEP's conjugate carrier; zero divides the "
            f"amplitude correction 1/(-2*pi*i*f))"
        )
    return 1.0 / (-2j * np.pi * frequency)


def _iyee_shift(component: str, d: int) -> int:  # MEEP vec.hpp:1132-1140 — half-cell offset of a component along axis d.
    """Half-cell offset of a component along one axis, in MEEP's doubled units.

    Electric components carry the shift on their own axis, magnetic components on
    the two other axes (MEEP's `d == component_direction` / `d != ...` test), and
    'Centered' carries it everywhere.
    """
    if d not in (0, 1, 2):
        raise ValueError(f"Unknown axis index {d!r}; expected 0 (X), 1 (Y) or 2 (Z)")
    if component == 'Centered':
        return 1
    if component in ('Ex', 'Dx'):
        return 1 if d == 0 else 0
    if component in ('Ey', 'Dy'):
        return 1 if d == 1 else 0
    if component in ('Ez', 'Dz'):
        return 1 if d == 2 else 0
    if component in ('Hx', 'Bx'):
        return 0 if d == 0 else 1
    if component in ('Hy', 'By'):
        return 0 if d == 1 else 1
    if component in ('Hz', 'Bz'):
        return 0 if d == 2 else 1
    raise ValueError(
        f"Unsupported source component {component!r}; expected one of "
        f"{sorted(_ARRAY_FOR_COMPONENT)}"
    )


def _validate_component(component: str) -> str:  # Reject anything an electric current source cannot drive.
    name = _D_ARRAY_FOR_COMPONENT.get(component)
    if name is None:
        raise ValueError(
            f"Unsupported source component {component!r}; expected one of "
            f"{sorted(_D_ARRAY_FOR_COMPONENT)}. Magnetic components need VolumeSource."
        )
    return name


def _validate_any_component(component: str) -> str:  # Reject anything either field family can drive.
    name = _ARRAY_FOR_COMPONENT.get(component)
    if name is None:
        raise ValueError(
            f"Unsupported source component {component!r}; expected one of "
            f"{sorted(_ARRAY_FOR_COMPONENT)}"
        )
    return name


def _field_type_for(component: str) -> str:  # 'D' or 'B' — which half-step slot a component is injected in.
    field_type = _FIELD_TYPE_FOR_COMPONENT.get(component)
    if field_type is None:
        raise ValueError(
            f"Unsupported source component {component!r}; expected one of "
            f"{sorted(_ARRAY_FOR_COMPONENT)}"
        )
    return field_type


def _d_array_for(fields: 'Fields', component: str) -> Any:  # Resolve the D array a component drives; unknown names raise.
    # RAW: every caller hands this array to :func:`_deposit`, which moves
    # only the source cells. A barriered fetch would sync the whole volume first.
    return host_writes.raw(fields, _validate_component(component))


def _array_for(fields: 'Fields', component: str) -> Any:  # Resolve the D or B array a component drives.
    return host_writes.raw(fields, _validate_any_component(component))


def _condinv_at(fields: 'Fields', array_name: str) -> Any:  # MEEP's s->condinv[c][d] for the deposit, or None.
    """The ``condinv`` volume of ``array_name``, the array a non-integrated source deposits into.

    MEEP's ``step_source`` reads ``s->condinv[c][component_direction(sv.c)]`` for the
    deposited component of EITHER family -- D for an electric source, B for a magnetic
    one (step.cpp:299-307) -- and scales the current by it inside the one double
    expression :func:`_deposit` forms. ``array_name`` is the name the caller resolved
    its array by, so the scale is read for exactly the array written. ``None`` is MEEP's
    NULL: no conductivity on that component, no multiply. A ``Fields`` stand-in without
    conductivity support (the unit tests' stub) is lossless.
    """
    getter = getattr(fields, "condinv_for", None)
    if getter is None:
        return None
    return getter(array_name)


def _axis_is_mirrored(grid: 'Grid', axis: int) -> bool:  # Is this axis mirror-folded?
    """Defer to ``Grid.is_mirrored``, the one definition of "folded".

    The fallback serves the stub grids in test_sources.py, which model the
    registration conventions and carry no methods; it reads the same per-axis
    flags ``Grid`` publishes, including ``sym_z``, so a folded Z is never
    silently reported as an unfolded axis the way a hard-coded ``False`` in the
    Z slot did.
    """
    reader = getattr(grid, 'is_mirrored', None)
    if callable(reader):
        return bool(reader(axis))
    return bool(getattr(grid, ('sym_x', 'sym_y', 'sym_z')[axis], False))


def _nosize_direction(grid: 'Grid', axis: int) -> bool:  # MEEP fields::nosize_direction, per axis.
    """Defer to ``Grid.nosize_direction``: is this a one-pixel periodic axis?

    The one place the delta-function amplitude scaling asks whether a zero extent
    means "at this point" or "along the whole of this direction". The fallback
    reproduces the same test from the attributes a stub grid in ``test_sources.py``
    carries — cell count 1 and neither folded nor walled — so a stub cannot silently
    take the 3-D branch on an axis it declared reduced.
    """
    reader = getattr(grid, 'nosize_direction', None)
    if callable(reader):
        return bool(reader(axis))
    counts = (getattr(grid, 'nx_full', None), getattr(grid, 'ny_full', None),
              getattr(grid, 'nz_full', None))
    if counts[axis] != 1:
        return False
    metallic = getattr(grid, 'metallic_axes', (False, False, False))
    return not _axis_is_mirrored(grid, axis) and not bool(metallic[axis])


def _axis_mirror_phase(grid: 'Grid', axis: int) -> Optional[int]:  # Declared phase of this axis's plane, or None.
    """Defer to ``Grid.mirror_phase``; ``None`` where the axis carries no plane.

    The fallback is MEEP's own default phase for a stub grid that knows only
    which axes are folded, which is exactly what those stubs declare.
    """
    reader = getattr(grid, 'mirror_phase', None)
    if callable(reader):
        return reader(axis)
    return 1 if _axis_is_mirrored(grid, axis) else None


def _validate_symmetry(grid: 'Grid', center: Tuple[float, float, float]) -> None:  # Sources must sit in the half a fold stores.
    """A source centre must lie in the half a folded axis STORES, not only on the plane.

    THE RULE USED TO BE "ON THE PLANE", and that was stricter than MEEP by a whole
    class of run. MEEP adds sources with ``use_symmetry=false``
    (``add_volume_source``, sources.cpp:487) and says so in its own comment above
    ``src_vol_chunkloop``: only the untransformed chunks are deposited into,
    "since the transformed versions are implicit". So a source anywhere in the
    stored half is legal, and its mirror image is supplied by the fold — which is
    exactly what a user declaring ``mp.Mirror`` is asserting about their problem.
    Measured on a 10x6 cell at resolution 10 with a 2 um layer: MEEP's folded run
    with ONE source at x = +1.1 equals MEEP's UNFOLDED run with that source and its
    parity-signed image at x = -1.1 to **1.6e-07..3.1e-07** over five
    (component, phase) combinations, and this engine reproduces MEEP's folded
    answer at **3.8e-07..9.3e-07** — its ordinary fold floor. Refusing them cost
    three of MEEP's own examples a lift (``perturbation_theory_2d.py``,
    ``solve-cw.py``, ``antenna_pec_ground_plane.py``, all with the source at
    +1.1 or +1.25 and none of them on a plane).

    WHAT IS STILL REFUSED is the OTHER half, and there the refusal is worth more
    than it was: MEEP stores the upper half, so a source declared in the lower one
    intersects no stored chunk and is deposited NOWHERE. Measured on the same cell:
    the identical run with the source at x = -1.1 returns a field whose peak is
    **0.000e+00** — a complete, clean, entirely zero result that reads as a
    finished run. Reproducing that silently is the degenerate-returns failure this
    codebase refuses on principle, so it raises and names the fix: declare the
    source at the mirrored coordinate, which is the same physical configuration.

    The rule is per axis and reads ``Grid.is_mirrored``, so a folded Z is checked
    exactly as a folded X is. It used to name X and Y only, which meant a Z fold
    accepted a source anywhere along z and folded it about a plane it did not sit on.
    """
    tol = grid.dx / 10
    for axis in range(3):
        if not _axis_is_mirrored(grid, axis) or center[axis] >= -tol:
            continue
        name = ('x', 'y', 'z')[axis]
        raise ValueError(
            f"With {name.upper()} symmetry the stored half is {name} >= 0, and this source is at "
            f"{name}={center[axis]:g} — in the half the fold discards. MEEP does not refuse this: it "
            f"adds sources with use_symmetry=false (sources.cpp:487), finds no stored chunk to "
            f"deposit into, and returns a field whose peak is exactly 0 — a complete, smooth, "
            f"entirely empty run that looks finished (measured). Declare the source at "
            f"{name}={-center[axis]:g} instead; under this plane that is the same physical "
            f"configuration, and its image at {name}={center[axis]:g} is supplied by the fold."
        )


def _validate_symmetry_parity(grid: 'Grid', component: str,
                              center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                              size: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                              amp_func: Optional[Callable[[float, float, float], complex]] = None,
                              ) -> None:
    """Refuse a source on a mirror plane that CANNOT carry the parity that plane demands.

    WHAT MEEP DOES — the mechanism, from the source, because the docstring that
    stood here asserted a different one. MEEP neither forbids this nor cancels it.
    It CLIPS. ``fields::add_volume_source`` deposits with
    ``loop_in_chunks(..., /*use_symmetry=*/false)`` (sources.cpp:487, and the
    comment at :239-244 says why: "we only find the intersection of the grid_volume
    with the untransformed chunks (since the transformed versions are implicit)"),
    and with ``use_symmetry`` false the ``sn`` loop runs once (loop_in_chunks.cpp:377),
    so only chunks that UNTRANSFORMED intersect the source volume are visited. Those
    chunks tile the halved grid_volume (structure.cpp:158-182, vec.cpp:1070-1075).
    The part of the source in the discarded half is therefore never deposited at
    all, and the missing half is supplied at readback as
    ``S.phase_shift(c, sn)`` times the owned value (boundaries.cpp:282-303,
    vec.cpp:1347-1369). No ``J`` ever meets a ``-J``; the "the fold cancels it
    against its own mirror image, ``J - J = 0``" claim that stood here is supported
    by no line of MEEP's source, and stock MEEP's answer for the case it described
    is not zero (measured below). There is no check and no warning anywhere in
    ``src/sources.cpp`` or ``python/simulation.py``: MEEP projects silently.

    So the current MEEP really simulates is ``J`` on the owned half plus
    ``parity * J(mirror)`` on the other, and the folded run reproduces the run that
    was written down **iff the declared current already has that parity**:

        J(mirror p) == mirror_parity(component, axis, phase) * J(p)

    THE PREDICATE IS THE PROFILE, NOT THE CENTRE. This used to judge a source by its
    centre alone, which applied a plane-centred-dipole argument to an extended sheet
    and refused five of MEEP's own ``python/tests`` cases (``test_mode_coeffs``
    x3, ``test_special_kz`` x2, 38 MEEP-authored assertions) whose folded answers are
    measurably RIGHT. Measured in stock MEEP 1.33.0, folded vs unfolded at the pulse
    peak over a shared sub-volume, 10x6 cell at resolution 10, all five sources
    centred on x = 0 and all driving a component the declared plane makes odd
    (``parity/meep_gpu/results/source_parity_predicate_2026-08-08/``,
    ``profile_predicate.jsonl``):

        zero-extent point,   Mirror(X, -1)          rel L2  1.006
        uniform sheet |x|<=2, Mirror(X, -1)         rel L2  2.829
        odd sheet sin(pi x/4), Mirror(X, -1)        rel L2  3.71e-07   <- foldable
        uniform sheet |x|<=2, Mirror(X, +1)         rel L2  0.569
        odd sheet sin(pi x/4), Mirror(X, +1)        rel L2  4.05e-07   <- foldable

    Extent alone is NOT the predicate — the two uniform sheets span the folded axis
    and are 57% and 283% wrong. What separates the pairs is that ``sin(pi x/4)`` is
    odd about x = 0 and the uniform profile is even.

    WHAT THIS REFUSES is the two cases that provably cannot carry an odd parity,
    both cheap to decide and both measured above:

    * ZERO EXTENT along the folded axis. The condition then constrains the source
      against itself. Where the component is unstaggered (the odd set of an odd
      plane) the sample sits on the plane, is its own image, and ``J = -J`` admits
      only ``J = 0``; where it is staggered (the odd set of an EVEN plane —
      ``iyee_shift`` gives a half-pixel along axis ``a`` exactly for ``{Ea, Hb, Hc}``,
      vec.hpp:1132-1140) half the deposit falls in the discarded half and is dropped.
      Stock MEEP runs both and returns a smooth, plausible, finished-looking field:
      1.006 for the unstaggered case above, and 7.65 (max |Ex| 5.081 folded vs 0.761
      unfolded) for a point Ex dipole at x = 0 under ``mp.Mirror(mp.X)``.
    * A UNIFORM PROFILE over a nonzero extent, i.e. no ``amp_func``. A constant is
      even about every plane, so an odd parity is impossible by inspection — 0.569
      and 2.829 above.

    WHAT THIS DOES NOT CHECK, stated plainly because the gap is real: an extended
    source WITH an ``amp_func`` is accepted without evaluating whether that profile
    carries the plane's parity. Deciding that per source is not enough, and the
    measurement says why. On the five rows above, the profile-parity residual
    ``|| A(-u) - parity*A(u) || / || A(u) ||`` and each sheet's share of the
    declaration's largest deposited amplitude are (same directory,
    ``profile_parity.jsonl`` and ``sheet_weights.jsonl``):

        test_special_kz   Ey  residual 1.6e-15   share 7.3e-02   <- odd to machine precision
        test_special_kz   Hz  residual 1.997     share 3.1e-06
        test_mode_coeffs  Ey  residual 1.926     share 9.3e-09
        test_mode_coeffs  Hz  residual 1.451     share 5.2e-09

    Every sheet carrying real current has the right parity; every wrong-parity sheet
    is numerical dust from an ``eig_parity``-unset eigenmode expansion, eight orders
    down. A residual test alone would refuse all five rows on that dust. The
    complete predicate is the residual WEIGHTED by the sheet's share of its
    declaration, and a per-source check cannot see the siblings that set the scale —
    it belongs to whatever builds the four equivalent-current sheets together
    (``from_meep._lift_eigenmode_source``) or to a driver-level gate. Deliberately
    not built here; the numbers above are the whole basis for it.

    OFF the plane nothing is refused, and that has its own measurement: ``J`` at
    ``+x0`` and ``-J`` at ``-x0`` are two distinct currents, so MEEP's folded run with
    one source at +1.1 equals its unfolded run with the pair to 2.9e-07 and this
    engine reproduces the folded answer at 9.3e-07.

    THE REFUSED SET IS READ FROM THE PLANE, not from MEEP's default.
    ``mirror_parity`` multiplies every component's parity by the declared phase, so
    the two phases of one axis are exactly complementary:

        Mirror('X', +1)   odd: Ex, Hy, Hz    even: Ey, Ez, Hx
        Mirror('X', -1)   odd: Ey, Ez, Hx    even: Ex, Hy, Hz

    which is the whole reason MEEP exposes ``phase``: a current along the plane's OWN
    normal — an ``Ex`` dipole on ``x = 0`` — is odd under an even mirror and can only
    be folded by declaring ``phase=-1``.

    Every offending plane is reported, not the first, and each names its whole odd
    set: one ``mp.EigenModeSource`` expands into four component sheets and this used
    to raise on whichever was constructed first, which read as a single-component
    defect and sent an investigation to the wrong component.
    """
    offenders = mirror_parity_offenders(
        component,
        [_axis_mirror_phase(grid, axis) for axis in range(3)],
        center, size, amp_func is not None, grid.dx,
    )
    if offenders:
        raise ValueError(mirror_parity_message(component, offenders))


def mirror_parity_offenders(component: str, phases, center, size,
                            has_amp_func: bool, dx: float) -> list:
    """THE predicate, once, for both the driver and the pre-flight gate.

    Returns one ``(axis, phase, why)`` per mirror plane this declaration cannot
    carry the parity of, empty when the declaration is foldable. ``phases[axis]``
    is the declared phase of the mirror on that axis, or ``None`` for an unfolded
    axis. ``has_amp_func`` — not the callable — is deliberately the whole of what
    is read about the profile, because that is the most the GATE can know: it
    decides before MPB has run, so an ``mp.EigenModeSource``'s profile does not
    exist yet to be sampled.

    WHY THIS IS A SEPARATE FUNCTION and not two copies of one rule. It used to be
    two copies, and they disagreed the moment the driver's half was narrowed: the
    gate kept "on the plane AND the component is odd" while
    :func:`_validate_symmetry_parity` moved to "…AND the profile cannot carry it".
    A gate STRICTER than the driver reports ``supported=False`` for a run that
    lifts and steps correctly, which is the same contract failure as the reverse
    (``from_meep._check_source_folds``, "the pre-flight must never say supported and
    then watch the driver raise") pointing the other way. One function makes the
    two agree by construction; ``mutate_from_meep``'s
    ``source_parity_gate_reads_component_only`` mutation pins that they still do.
    """
    from .fields import (  # Local: fields imports nothing from here, and this keeps that one-way.
        FIELD_COMPONENTS,
        mirror_parity,
    )

    if component not in FIELD_COMPONENTS:
        return []  # Component validity is not this function's job; the callers already checked it.
    tol = dx / 10
    offenders = []
    for axis in range(3):
        phase = phases[axis]
        if phase is None or mirror_parity(component, axis, phase) > 0:
            continue
        if abs(center[axis]) > tol:
            continue  # Off the plane: the image is a distinct current, not a canceller.
        if size[axis] <= tol:
            why = (f"it has no extent along {'xyz'[axis]} (size[{'xyz'[axis]}]={size[axis]:g}), so the "
                   f"parity condition constrains it against itself and admits only zero current")
        elif not has_amp_func:
            why = (f"its profile over |{'xyz'[axis]}| <= {size[axis] / 2:g} is uniform (no amp_func) "
                   f"while a constant is EVEN about every plane, so no extent can give it parity -1")
        else:
            continue  # Extended with a profile: the profile can carry the plane's parity.
        offenders.append((axis, phase, why))
    return offenders


def mirror_parity_message(component: str, offenders: list) -> str:
    """The refusal text, shared so the gate reports what the driver would raise."""
    from .fields import mirror_parity

    parts = []
    for axis, phase, why in offenders:
        axis_name = ('X', 'Y', 'Z')[axis]
        kind = 'even' if phase > 0 else 'odd'
        odd_here = [name for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
                    if mirror_parity(name, axis, phase) < 0]
        even_here = [name for name in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
                     if mirror_parity(name, axis, phase) > 0]
        parts.append(
            f"the {kind} {axis_name} plane Mirror({axis_name!r}, phase={phase:+d}) gives it parity -1, "
            f"and {why}. That plane makes {', '.join(odd_here)} odd and leaves "
            f"{', '.join(even_here)} even, so a multi-component declaration on it "
            f"(an eigenmode source expands into four sheets) is refused on every odd one"
        )
    return (
        f"A {component} source centred on the mirror plane cannot carry the parity the fold "
        f"demands: " + "; also ".join(parts) + ". MEEP does not refuse this — it deposits with "
        f"use_symmetry=false (sources.cpp:487), keeps only the half it stores and images the rest "
        f"with the opposite sign, so the run completes and returns a smooth, plausible field that "
        f"is 57%-283% away from the equivalent full-domain run (measured, stock MEEP 1.33.0). "
        f"Give the source an extent across the plane with an amp_func whose profile is ODD about "
        f"it, drive one of the components the plane leaves even, declare the opposite phase, or "
        f"drop the symmetry and run the full domain."
    )


def _continuous_dipole(time: float, omega: float, amp_factor: complex, start_time: float,
                       end_time: float, width: float, slowness: float) -> complex:
    """MEEP sources.cpp continuous_src_time::dipole — CW carrier with tanh turn-on/off.

    Verified against meep 1.29.0's `continuous_src_time.dipole` to the last bit
    (including the float32 comparison MEEP uses on the time window). `width = 0`
    reproduces MEEP's default hard switch-on; a nonzero `width` suppresses the
    broadband switch-on transient over roughly `slowness` widths.

    The bundle this module was ported from omitted the ramp entirely (port
    reference §5.4); it is restored here as the canonical behavior.
    """
    rtime = np.float32(time)  # MEEP: float rtime = float(time)
    if rtime < np.float32(start_time) or rtime > np.float32(end_time):
        return 0.0 + 0j
    carrier = complex(np.exp(-1j * omega * time))
    if width == 0.0:
        return carrier * amp_factor
    ts = (time - start_time) / width - slowness
    te = (end_time - time) / width - slowness
    return carrier * amp_factor * (1.0 + math.tanh(ts)) * (1.0 + math.tanh(te)) * 0.25


def _gaussian_cutoff_time(cutoff: float, width: float) -> float:
    """MEEP sources.cpp:93-95 — shrink the hard cut until the envelope there is representable.

    The result is rounded to float32 exactly as MEEP does, so the cut is
    insensitive to roundoff on both sides of the comparison.
    """
    cutoff_time = cutoff * width
    while math.exp(-cutoff_time * cutoff_time / (2 * width * width)) < 1e-100:
        cutoff_time *= 0.9
    return float(np.float32(cutoff_time))


def _gaussian_dipole(time: float, omega: float, amp_factor: complex, peak_time: float,
                     width: float, cutoff_time: float) -> complex:
    """MEEP sources.cpp:98-107 gaussian_src_time::dipole — Gaussian envelope about the peak.

    The carrier phase is referenced to `time - peak_time`, not to absolute time,
    and the envelope is cut hard outside `cutoff_time`. The envelope's argument is
    MEEP's expression in MEEP's order, `-tt * tt / (2 * width * width)`
    (sources.cpp:106), evaluated in double: `-0.5 * (tt / width) ** 2` is the same
    function and a different double.
    """
    t_rel = time - peak_time
    if np.float32(abs(t_rel)) > cutoff_time:
        return 0.0 + 0j
    envelope = math.exp(-t_rel * t_rel / (2 * width * width))
    phase = complex(np.exp(-1j * omega * t_rel))
    return phase * envelope * amp_factor


def _custom_dipole(time: float, func: Callable[[float], complex], start_time: float,
                   end_time: float) -> complex:
    """MEEP meep.hpp:1072-1078 custom_src_time::dipole — the caller's waveform, gated by a time window.

    MEEP stores the window bounds as float32 and compares a float32 time against
    them, so a caller-supplied start or end that is not exactly representable
    gates on the same side of the boundary as CPU MEEP does.
    """
    rtime = np.float32(time)
    if rtime < np.float32(start_time) or rtime > np.float32(end_time):
        return 0.0 + 0j
    return complex(func(time))


def _vec2diel_floor(pt: float, a: float) -> int:
    """MEEP loop_in_chunks.cpp:233-241 — nearest LOWER ivec on the doubled dielectric grid."""
    return 1 + 2 * int(np.floor(pt * a - 0.5))


def _vec2diel_ceil(pt: float, a: float) -> int:
    """MEEP loop_in_chunks.cpp:242-250 — nearest UPPER ivec on the doubled dielectric grid."""
    return 1 + 2 * int(np.ceil(pt * a - 0.5))


def _compute_boundary_weights(where_min: float, where_max: float, is_d: int, ie_d: int,
                              a: float) -> Tuple[float, float, float, float]:
    """MEEP loop_in_chunks.cpp:257-297 — Simpson integration weights (s0, s1, e0, e1) for one direction.

    The `elif` ordering below is semantically load-bearing: a zero-extent source
    can also satisfy `num_doubled == 2`, and MEEP reaches the zero-extent branch
    first. Reordering silently changes point-source amplitudes.
    """
    # MEEP lines 260-262
    w0 = 1.0 - where_min * a + 0.5 * is_d
    w1 = 1.0 + where_max * a - 0.5 * ie_d

    num_doubled = ie_d - is_d

    # MEEP lines 263-296 - Simpson weight selection
    if num_doubled >= 3 * 2:  # ie >= is + 6 (3+ cells)
        s0 = w0 * w0 / 2
        s1 = 1 - (1 - w0) * (1 - w0) / 2
        e0 = w1 * w1 / 2
        e1 = 1 - (1 - w1) * (1 - w1) / 2
    elif num_doubled == 2 * 2:  # ie == is + 4 (2 cells)
        s0 = w0 * w0 / 2
        s1 = 1 - (1 - w0) * (1 - w0) / 2 - (1 - w1) * (1 - w1) / 2
        e0 = w1 * w1 / 2
        e1 = s1
    elif where_min == where_max:  # Zero extent (point/line source)
        s0 = w0
        s1 = w1
        e0 = w1
        e1 = w0
    elif num_doubled == 1 * 2:  # ie == is + 2 (1 cell)
        s0 = w0 * w0 / 2 - (1 - w1) * (1 - w1) / 2
        e0 = w1 * w1 / 2 - (1 - w0) * (1 - w0) / 2
        s1 = e0
        e1 = s0
    else:
        s0 = s1 = e0 = e1 = 1.0

    return s0, s1, e0, e1


def _ivec_loop_weight_1d(i: int, n: int, s0: float, s1: float, e0: float, e1: float) -> float:
    """MEEP vec.hpp:372-378 IVEC_LOOP_WEIGHT1x — integration weight of position i in a run of n.

    The test order mirrors MEEP's nested ternary (is, then is+2, then ie, then
    ie-2), which matters when the run is shorter than four positions.
    """
    if i > 1 and i < n - 2:
        return 1.0
    elif i == 0:
        return s0
    elif i == 1:
        return s1
    elif i == n - 1:
        return e0
    elif i == n - 2:
        return e1
    else:
        return 1.0


class _SourcePass(NamedTuple):
    """One lattice shift of a source request: one ``ishift`` of MEEP's loop_in_chunks.

    ``ishift`` is the lattice shift and ``shifti`` its doubled translation
    (loop_in_chunks.cpp:409-418). ``amp`` is MEEP's ``data->amp * conj(shift_phase)``
    (sources.cpp:259) as (real, imag). ``points`` lists every point the shifted request
    places on the window's owned lattice, in ``LOOP_OVER_IVECS`` order (x outermost, z
    innermost): its CHUNK-frame doubled coordinate per axis (``None`` on an axis MEEP
    does not loop over), the stored cell it lands on (``None`` where MEEP's own deposit
    is inert, the metallic wall plane), and ``A(rel_loc)`` as (real, imag).
    """

    ishift: Tuple[int, int, int]
    shifti: Tuple[int, int, int]
    amp: Tuple[float, float]
    points: Tuple[Tuple[Tuple[Optional[int], Optional[int], Optional[int]],
                        Optional[Tuple[int, int, int]], Tuple[float, float]], ...]


class _SourceGeometry(NamedTuple):
    """Everything ``loop_in_chunks`` needs to cut a source request into MEEP's ``src_vol`` pieces.

    Built once, when the source is placed (:func:`_source_geometry`), from the grid alone;
    :func:`_src_vol_pieces` cuts it per chunk of a division without calling the source's
    amplitude function again, so a capsule replay asks it only the questions the
    construction asked. ``directions`` is MEEP's ``LOOP_OVER_DIRECTIONS`` as package axes,
    ``yucky`` its three ``LOOP_OVER_IVECS`` directions (outer to inner) with ``None`` for a
    direction the dimension does not carry. ``is_``/``ie`` are the doubled bounds and
    ``s0``/``s1``/``e0``/``e1`` the boundary weights of :func:`_compute_boundary_weights`,
    per axis (unused off ``directions``). ``user_low`` is
    ``user_volume.little_owned_corner(c)`` per axis, and ``window`` the grid's
    ``(little_corner, big_corner)``, doubled: the box of the one-chunk division.
    """

    component: str
    directions: Tuple[int, ...]
    yucky: Tuple[Optional[int], Optional[int], Optional[int]]
    iyee: Tuple[int, int, int]
    is_: Tuple[int, int, int]
    ie: Tuple[int, int, int]
    s0: Tuple[float, float, float]
    s1: Tuple[float, float, float]
    e0: Tuple[float, float, float]
    e1: Tuple[float, float, float]
    user_low: Tuple[int, int, int]
    window: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    cylindrical: bool
    passes: Tuple[_SourcePass, ...]


def _meep_directions(grid: 'Grid') -> Tuple[Tuple[int, ...], Tuple[Optional[int], Optional[int], Optional[int]]]:
    """MEEP's ``LOOP_OVER_DIRECTIONS`` and ``yucky_dir`` order for this grid, as package axes.

    vec.hpp ``LOOP_OVER_DIRECTIONS`` walks the ``direction`` enum (X, Y, Z, R, P) from
    ``start_at_direction`` to ``stop_at_direction``: D1 is Z; D2 is X, Y; D3 is X, Y, Z;
    Dcyl is Z then R (package z, then package x). vec.cpp ``yucky_dir`` gives the three
    nested loop directions, outermost first: D2 is (Z, X, Y), Dcyl is (P, R, Z), D1 and D3
    are (X, Y, Z); a direction the dimension does not carry loops once.
    """
    if bool(getattr(grid, "cylindrical", False)):
        return (2, 0), (None, 0, 2)
    dimensions = int(getattr(grid, "dimensions", 3))
    if dimensions == 1:
        return (2,), (None, None, 2)
    if dimensions == 2:
        return (0, 1), (None, 0, 1)
    return (0, 1, 2), (0, 1, 2)


def _source_geometry(grid: 'Grid', component: str, center: Tuple[float, float, float],
                     size: Tuple[float, float, float], amplitude: complex,
                     amp_func: Optional[Callable[[float, float, float], complex]],
                     ) -> _SourceGeometry:
    """MEEP's ``add_volume_source`` up to the chunk loop, every lattice shift's points evaluated once.

    Line by line from sources.cpp:456-489 (``add_volume_source``), loop_in_chunks.cpp:349-418
    (the bounds, the boundary weights and the lattice-shift loop) and sources.cpp:243-312
    (``src_vol_chunkloop``: the location, the amplitude function and the owned test). What
    depends on the chunk division -- the owned-corner clip of each (shift, chunk) pair,
    the boundary weights it keeps and the narrow-range minima -- is left to
    :func:`_src_vol_pieces`; this function enumerates each shift's points over the whole
    window, which is the union of the chunks' owned ranges.

    THE SHIFTS STAY SEPARATE. A request that reaches a periodic face is placed once per
    lattice shift, and each (shift, chunk) pair is its own ``src_vol`` (sources.cpp:311);
    two of them meet on one cell exactly at a periodic seam (the request's two ends, one
    lattice vector apart) and MEEP deposits each separately, rounding twice
    (step.cpp:309-314), unless their index vectors are equal and ``add_source`` merges
    them. Summing the two amplitudes into one deposit, as this module did, rounds once:
    the one-ulp seam row of every full-width source across a periodic axis.

    THE LOCATION IS MEEP'S. ``IVEC_LOOP_LOC`` forms ``(0.5 * is + i) * inva`` in the chunk
    frame and ``src_vol_chunkloop`` then adds ``shift * (0.5 * inva)`` (vec.hpp:365-369,
    sources.cpp:270): two roundings on an image, where the source-frame form rounds once.
    An axis the dimension does not carry has no coordinate at all: the Python amplitude
    function receives 0 there (python/typemap_utils.cpp ``vec2py``), and the axis is one
    point of weight 1 (``s0c``'s ``vec(gv.dim, 1.0)``, loop_in_chunks.cpp:451).

    THE PRODUCTS ARE MEEP'S. ``data.amp *= gv.a`` and ``IVEC_LOOP_WEIGHT * amp`` scale per
    part; ``data->amp * conj(shift_phase)`` and the product with ``A`` are the C++
    ``complex<double>`` multiply, ``(ac - bd, ad + bc)`` with every product rounded
    (:func:`readback.cmul`), and the shift phase's power is the host C++ runtime's
    (:func:`readback.cpow_int`). A Python complex multiply is not that: CPython's on
    Apple silicon contracts it to a fused multiply-add (measured: 87,460 of 200,000
    random products differ from the separately rounded form).

    EVERY OWNED POINT IS KEPT, its amplitude zero or not: MEEP's ``src_vol`` lists them all
    and its deposit is not word-neutral on a stored ``-0`` (see :func:`_deposit`).
    """
    from . import readback  # noqa: PLC0415 - readback imports fields, which imports nothing from here

    _validate_any_component(component)
    a = grid.resolution  # MEEP: gv.a
    inva = grid.dx       # MEEP: gv.inva = 1/a
    directions, yucky = _meep_directions(grid)
    resolved = tuple(d in directions for d in range(3))

    where_min = [center[d] - size[d] / 2 for d in range(3)]
    where_max = [center[d] + size[d] / 2 for d in range(3)]

    # MEEP sources.cpp:467-476 add_volume_source -- the request is measured against the
    # cell before anything else happens to it. Wider than one cell plus one pixel and
    # MEEP aborts; wider by less than a pixel and MEEP snaps it back to exactly one cell,
    # the minimum moving DOWN by half the excess and the maximum placed one cell length
    # above the new minimum. `Lx`/`Ly`/`Lz` are the full cell lengths on a folded axis
    # too, which is the length MEEP compares against (its user_volume spans the cell).
    lengths = (grid.Lx, grid.Ly, grid.Lz)
    for d in directions:
        span = where_max[d] - where_min[d]
        if span > lengths[d] + inva:
            raise ValueError(
                f"Source size {size[d]} along {('x', 'y', 'z')[d]} exceeds the cell length "
                f"{lengths[d]} by more than one cell ({inva}); MEEP's add_volume_source aborts on "
                f"exactly this request (sources.cpp:469, \"Source width > cell width\"). A source "
                f"wider than one lattice period deposits several of its own lattice images onto "
                f"the same cells, so it would run and be over-driven rather than fail. Clamp the "
                f"size to the cell."
            )
        if span > lengths[d]:
            excess = span - lengths[d]
            where_min[d] = where_min[d] - excess * 0.5
            where_max[d] = where_min[d] + lengths[d]

    # MEEP sources.cpp:481-484: `data.amp *= gv.a` per zero-size direction the dimension
    # carries, except a one-pixel periodic one (`nosize_direction`, Grid.nosize_direction:
    # MEEP's lower-dimensional emulation, where a zero extent means the whole direction).
    # complex<double> *= double scales each part.
    amp = (float(complex(amplitude).real), float(complex(amplitude).imag))
    for d in directions:
        if where_max[d] - where_min[d] == 0.0 and not _nosize_direction(grid, d):
            amp = (amp[0] * a, amp[1] * a)

    # MEEP loop_in_chunks.cpp:352-359: the bounds on the component's lattice and the
    # boundary weights of the ORIGINAL where.
    iyee = tuple(_iyee_shift(component, d) for d in range(3))
    iyee_c = [_iyee_shift('Centered', d) - iyee[d] for d in range(3)]
    is_ = [0, 0, 0]
    ie = [0, 0, 0]
    s0 = [1.0, 1.0, 1.0]
    s1 = [1.0, 1.0, 1.0]
    e0 = [1.0, 1.0, 1.0]
    e1 = [1.0, 1.0, 1.0]
    for d in directions:
        yee_c = iyee_c[d] * 0.5 * inva
        is_[d] = _vec2diel_floor(where_min[d] + yee_c, a) - iyee_c[d]
        ie[d] = _vec2diel_ceil(where_max[d] + yee_c, a) - iyee_c[d]
        s0[d], s1[d], e0[d], e1[d] = _compute_boundary_weights(
            where_min[d], where_max[d], is_[d], ie[d], a)

    # The window: MEEP's little_corner / big_corner (vec.hpp, `Grid.origin_doubled` and
    # `Grid.big_corner_doubled` -- `owned_cells`, not the stored count, so a folded periodic
    # axis's far ghost slot is never deposited into). Its owned range is
    # `[little_corner + 2 - iyee, big_corner - iyee]` (vec.hpp:1102-1107), with the
    # cylindrical axis row clawed back to 0 (vec.cpp:432-436). The user volume is the full
    # cell (a folded axis's full little corner lies below the halved window, so the
    # `max` against it in loop_in_chunks.cpp:425 never binds there).
    origin_reader = getattr(grid, "origin_doubled", None)
    corner_reader = getattr(grid, "big_corner_doubled", None)
    axis_reader = getattr(grid, "is_axis", None)
    full_counts = (grid.nx_full, grid.ny_full, grid.nz_full)
    stored_counts = (grid.nx, grid.ny, grid.nz)
    io = [0, 0, 0]
    big = [0, 0, 0]
    user_low = [0, 0, 0]
    for d in range(3):
        if callable(origin_reader):
            io[d] = int(origin_reader(d))
        elif _axis_is_mirrored(grid, d):
            io[d] = -2
        else:
            io[d] = -(full_counts[d] - full_counts[d] % 2)
        if callable(corner_reader):
            big[d] = int(corner_reader(d))
        else:
            big[d] = io[d] + 2 * stored_counts[d]
        user_io = -(full_counts[d] - full_counts[d] % 2) if _axis_is_mirrored(grid, d) else io[d]
        user_low[d] = user_io + 2 - iyee[d]
        if callable(axis_reader) and axis_reader(d) and user_low[d] == 2:
            user_low[d] = 0
    window_low = [_chunk_little_owned(io[d], iyee[d], d, grid) for d in range(3)]
    window_high = [big[d] - iyee[d] for d in range(3)]

    # Which axes MEEP loops lattice shifts over: `boundaries[High][d] == Periodic`
    # (loop_in_chunks.cpp:391), which is every wrapping axis and also a mirror-folded
    # axis whose outer face is periodic (the second mirror at +L/2 is a lattice image).
    wraps_reader = getattr(grid, "axis_wraps", None)
    metallic_reader = getattr(grid, "is_metallic", None)
    wraps = tuple(bool(wraps_reader(d)) if callable(wraps_reader) else not _axis_is_mirrored(grid, d)
                  for d in range(3))
    folded_live = tuple(
        _axis_is_mirrored(grid, d) and callable(metallic_reader) and not metallic_reader(d)
        and callable(corner_reader)
        for d in range(3))
    periodic = tuple(resolved[d] and (wraps[d] or folded_live[d]) for d in range(3))
    # The plane ON a metallic high wall is an owned entry MEEP deposits into and then
    # zeroes before anything reads it (`find_metals` registers it, `step_boundaries` zeroes
    # metals first, boundaries.cpp:187-201, step.cpp:245): the exact image here is no
    # deposit at all. `boundaries` is read defensively for the stub grids in test_sources.py.
    boundaries_reader = getattr(grid, "boundaries", None)
    metal_high = tuple(
        (not wraps[d]) and boundaries_reader is not None and boundaries_reader[d][1] == METALLIC
        for d in range(3))

    def cell_of(d: int, p: int) -> Optional[int]:  # The stored cell of an owned doubled coordinate.
        ladder = (p - io[d]) // 2
        count = stored_counts[d]
        if wraps[d]:
            return ladder % count
        if metal_high[d] and ladder >= count:
            return None  # the metallic wall plane: MEEP's deposit there is zeroed unread
        if not 0 <= ladder < count:
            raise RuntimeError(
                f"source deposition on axis {('x', 'y', 'z')[d]}: owned doubled coordinate {p} "
                f"is stored row {ladder}, outside the {count} stored rows -- a registration "
                f"slip between the window [{window_low[d]}, {window_high[d]}] and the grid, not "
                f"a caller error")
        return ladder

    # MEEP loop_in_chunks.cpp:391-404: the shift range over which the window's surroundings
    # intersect the request, `gv[ivec] = ivec * (0.5 * inva)` and the lattice vector
    # `gv[ilattice_vector(d)]` (boundaries.cpp:121-148), whose length is the FULL cell's.
    half_inva = 0.5 * inva
    min_ishift = [0, 0, 0]
    max_ishift = [0, 0, 0]
    period = [0, 0, 0]
    for d in directions:
        if not periodic[d]:
            continue
        period[d] = 2 * full_counts[d]
        lattice = abs(period[d] * half_inva)
        min_ishift[d] = int(math.floor((where_min[d] - big[d] * half_inva) / lattice))
        max_ishift[d] = int(math.ceil((where_max[d] - io[d] * half_inva) / lattice))

    centre = [(where_min[d] + where_max[d]) * 0.5 for d in range(3)]  # sources.cpp:486
    passes: List[_SourcePass] = []
    ishift = list(min_ishift)
    while True:
        shifti = tuple(period[d] * ishift[d] for d in range(3))
        # loop_in_chunks.cpp:412-417: ph = 1; ph *= pow(eikna[d], ishift[d]) per direction,
        # libc++'s pow(complex, int) = exp(n * log(z)); sources.cpp:259 amp * conj(ph).
        ph = (1.0, 0.0)
        for d in directions:
            ph = readback.cmul(ph, readback.cpow_int(readback.eikna(grid, d), ishift[d]))
        pass_amp = readback.cmul(amp, (ph[0], -ph[1]))
        ranges = []
        for d in range(3):
            if not resolved[d]:
                ranges.append([None])
                continue
            low = max(is_[d] - shifti[d], window_low[d], user_low[d])
            high = min(ie[d] - shifti[d], window_high[d])
            ranges.append(list(range(low, high + 1, 2)) if low <= high else [])
        points = []
        for px in ranges[0]:
            for py in ranges[1]:
                for pz in ranges[2]:
                    coords = (px, py, pz)
                    rel = [0.0 if resolved[d] else _invariant_sample_offset(component, d, inva)
                           for d in range(3)]
                    cell = [0, 0, 0]
                    inert = False
                    for d in directions:
                        # IVEC_LOOP_LOC then `loc += shift * (0.5 * inva)`, then rel_loc.
                        loc = (0.5 * coords[d]) * inva
                        loc = loc + shifti[d] * half_inva
                        rel[d] = loc - centre[d]
                        stored = cell_of(d, coords[d])
                        if stored is None:
                            inert = True
                        else:
                            cell[d] = stored
                    if amp_func is None:
                        value = (1.0, 0.0)  # sources.cpp:219-222, `one`
                    else:
                        result = complex(amp_func(rel[0], rel[1], rel[2]))
                        value = (float(result.real), float(result.imag))
                    points.append((coords, None if inert else (cell[0], cell[1], cell[2]), value))
        if points:
            passes.append(_SourcePass(tuple(ishift), shifti, pass_amp, tuple(points)))
        # loop_in_chunks.cpp:525-532: the odometer, first direction fastest.
        for d in directions:
            if ishift[d] + 1 <= max_ishift[d]:
                ishift[d] += 1
                break
            ishift[d] = min_ishift[d]
        else:
            break

    return _SourceGeometry(
        component=component, directions=directions, yucky=yucky, iyee=iyee,
        is_=tuple(is_), ie=tuple(ie), s0=tuple(s0), s1=tuple(s1), e0=tuple(e0), e1=tuple(e1),
        user_low=tuple(user_low), window=(tuple(io), tuple(big)),
        cylindrical=bool(getattr(grid, "cylindrical", False)), passes=tuple(passes))


def _invariant_sample_offset(component: str, axis: int, inva: float) -> float:  # vec2py's 0.
    """The coordinate an amplitude function is handed on an axis the dimension does not carry: 0.

    ``src_vol_chunkloop`` builds ``rel_loc`` as a ``vec`` of the run's dimension, and
    ``vec2py`` (python/typemap_utils.cpp) gives the Python function 0 on every axis that
    dimension lacks -- z in 2-D, phi in Dcyl, x and y in 1-D -- whatever the component's
    Yee offset along it. A module-level function so the lift's mutation battery can plant
    the offset back (``full_3d_sample_point`` in test_from_meep.py): sampling a
    half-integer-z component at its own +-dx/2 costs a mode carrying kz a factor of
    ``cos(pi * kz * dx)`` and is invisible at kz = 0.
    """
    return 0.0


def _chunk_little_owned(lo: int, iyee: int, axis: int, grid: Any) -> int:
    """``grid_volume::little_owned_corner`` on one axis of a box starting at doubled ``lo``.

    ``little_corner + 2 - iyee_shift(c)`` (vec.hpp:1102-1104), and on the cylindrical
    axis a box whose origin is r = 0 owns its axis row: a value of 2 becomes 0
    (vec.cpp:432-436).
    """
    value = lo + 2 - iyee
    axis_reader = getattr(grid, "is_axis", None)
    if callable(axis_reader) and axis_reader(axis) and lo == 0 and value == 2:
        return 0
    return value


def _src_vol_pieces(geometry: _SourceGeometry, boxes: Any, grid: Any,
                    ) -> List[Tuple[int, List[Tuple[int, int, int]], List[complex]]]:
    """MEEP's ``src_vol`` pieces of one source, in ``add_source`` order: (chunk, cells, amplitudes).

    The chunk half of loop_in_chunks.cpp:420-516 and sources.cpp:243-312: for every
    lattice shift (the odometer order the geometry kept) and every chunk in index order,
    the request shifted into the chunk frame is clipped to the chunk's owned corners
    (:func:`_chunk_little_owned`, ``big_owned_corner = big_corner - iyee``), the boundary
    weights are kept where the clipped ends meet the request's ends IN THE SOURCE FRAME
    (``iscS += shifti`` precedes the comparisons) and the narrow-range minima are taken;
    each point's amplitude is ``(IVEC_LOOP_WEIGHT * amp) * A`` -- the weight nested as the
    macro nests it, ``w3 * (w2 * (1 * w1))`` over the yucky directions, scaling ``amp``
    per part, then libc++'s complex multiply by ``A``. A non-empty intersection is one
    piece. ``boxes`` is ``ChunkLayout.boxes`` (doubled ``(lo, hi)`` per chunk), or ``None``
    for one chunk spanning the window. A point on a metallic wall plane is placed and
    dropped (its deposit is inert); a piece left empty by that is not returned.
    """
    from . import readback  # noqa: PLC0415

    directions = geometry.directions
    if boxes is None:
        boxes = (geometry.window,)
    pieces: List[Tuple[int, List[Tuple[int, int, int]], List[complex]]] = []
    for source_pass in geometry.passes:
        shifti = source_pass.shifti
        for chunk, (lo, hi) in enumerate(boxes):
            isc = [0, 0, 0]
            iec = [0, 0, 0]
            empty = False
            for d in directions:
                low = max(geometry.user_low[d], _chunk_little_owned(int(lo[d]), geometry.iyee[d], d, grid))
                isc[d] = max(geometry.is_[d] - shifti[d], low)
                iec[d] = min(geometry.ie[d] - shifti[d], int(hi[d]) - geometry.iyee[d])
                if isc[d] > iec[d]:
                    empty = True
                    break
            if empty:
                continue
            weights = {}
            for d in directions:
                start, end = isc[d] + shifti[d], iec[d] + shifti[d]
                ws0 = ws1 = we0 = we1 = 1.0
                if start == geometry.is_[d]:
                    ws0, ws1 = geometry.s0[d], geometry.s1[d]
                elif start == geometry.is_[d] + 2:
                    ws0 = geometry.s1[d]
                if end == geometry.ie[d]:
                    we0, we1 = geometry.e0[d], geometry.e1[d]
                elif end == geometry.ie[d] - 2:
                    we0 = geometry.e1[d]
                if end == start:
                    w = min(ws0, we0)
                    ws0 = we0 = ws1 = we1 = w
                elif end == start + 2:
                    w = min(ws0, we1)
                    ws0 = we1 = w
                    w = min(ws1, we0)
                    ws1 = we0 = w
                elif end == start + 4:
                    w = min(ws1, we1)
                    ws1 = we1 = w
                weights[d] = (ws0, ws1, we0, we1, (iec[d] - isc[d]) // 2 + 1)
            cells: List[Tuple[int, int, int]] = []
            amps: List[complex] = []
            for coords, cell, value in source_pass.points:
                if any(not isc[d] <= coords[d] <= iec[d] for d in directions):
                    continue
                factors = []
                for d in geometry.yucky:
                    if d is None or d not in weights:
                        factors.append(1.0)
                        continue
                    ws0, ws1, we0, we1, n = weights[d]
                    factors.append(_ivec_loop_weight_1d((coords[d] - isc[d]) // 2, n, ws0, ws1, we0, we1))
                weight = factors[2] * (factors[1] * (1.0 * factors[0]))  # vec.hpp:381-383
                if cell is None:
                    continue
                scaled = (weight * source_pass.amp[0], weight * source_pass.amp[1])
                product = readback.cmul(scaled, value)
                cells.append(cell)
                amps.append(complex(product[0], product[1]))
            if cells:
                pieces.append((chunk, cells, amps))
    return pieces


def _summed_table(geometry: _SourceGeometry, grid: 'Grid',
                  ) -> Tuple[List[Tuple[int, int, int]], List[complex]]:
    """Every cell the source deposits into, once, with its pieces' amplitudes summed in order.

    The source's own table: the cells of every piece of the one-chunk division (which are
    the cells of every piece of any division), each listed once in first-placement order,
    its amplitude the sum of its pieces' in placement order. It is what the deposit-repair
    bracket reads for the cells a source writes (``_point_ix``) and the deposit a source
    makes before it is bound to a division; :func:`bind_src_vols` replaces the deposit by
    MEEP's list, where a cell two pieces reach is deposited twice.
    """
    accumulated: Dict[Tuple[int, int, int], complex] = {}
    for _chunk, cells, amps in _src_vol_pieces(geometry, None, grid):
        for cell, value in zip(cells, amps):
            previous = accumulated.get(cell)
            accumulated[cell] = value if previous is None else complex(
                previous.real + value.real, previous.imag + value.imag)
    return list(accumulated), list(accumulated.values())


def _build_source_points(grid: 'Grid', component: str, center: Tuple[float, float, float],
                         size: Tuple[float, float, float], amplitude: complex,
                         amp_func: Optional[Callable[[float, float, float], complex]],
                         ) -> Tuple[List[Tuple[int, int, int]], List[complex]]:
    """A source request's own table (:func:`_summed_table` of :func:`_source_geometry`)."""
    return _summed_table(_source_geometry(grid, component, center, size, amplitude, amp_func), grid)


def _index_arrays(grid: 'Grid', indices: List[Tuple[int, int, int]],
                  amplitudes: List[complex]) -> Tuple[Any, Any, Any, Any]:
    """Pack (index, amplitude) pairs into backend arrays for the deposit.

    The amplitudes are packed in COMPLEX128: they are MEEP's ``amp[j]``, which is a
    ``complex<double>`` (``src_vol::amp``, meep_internals.hpp:52-69), and
    :func:`_deposit` multiplies them by the current in double. Packing them in
    complex64, as this module did before the deposit moved to double, rounded each one
    to float before the product MEEP forms in double.
    """
    ix, iy, iz = _point_index_arrays(grid, indices)
    return ix, iy, iz, grid.xp.asarray(amplitudes, dtype=grid.xp.complex128)


def _point_index_arrays(grid: 'Grid', indices: List[Tuple[int, int, int]]) -> Tuple[Any, Any, Any]:
    """The three backend index arrays of a deposition table, refused if a cell repeats.

    MEEP's deposit loop (step.cpp:309-314) subtracts point by point, so a cell listed
    twice is subtracted twice and rounded twice; the vectorized deposit gathers each
    cell once and would keep only one of the two differences. Every table this module
    builds lists each cell once -- ``_build_source_points`` sums the amplitudes of the
    rungs that fold onto one cell -- and a table that does not is refused here, by
    name, rather than deposited differently from MEEP.
    """
    xp = grid.xp
    linear = [(x * grid.ny + y) * grid.nz + z for (x, y, z) in indices]
    if len(set(linear)) != len(linear):
        raise ValueError(
            "a source's deposition table lists the same cell more than once; MEEP "
            "subtracts each listed point separately (step.cpp:309-314, one rounding per "
            "point), which a vectorized deposit cannot reproduce, so the table is refused")
    ix = xp.asarray([p[0] for p in indices], dtype=xp.int64)
    iy = xp.asarray([p[1] for p in indices], dtype=xp.int64)
    iz = xp.asarray([p[2] for p in indices], dtype=xp.int64)
    return ix, iy, iz


def _injects_real(array: Any, component: str) -> bool:  # Real storage? Validated.
    """True when this array takes only ``real(A)``, False when it also takes ``imag(A)``.

    Resolves the two supported storage widths, complex64 and float32, and refuses any
    other with one message for every source class.
    """
    if array.dtype == np.complex64:
        return False
    if array.dtype == np.float32:
        return True
    raise ValueError(
        f"Source injection into {component} requires a complex64 or float32 primary array, got "
        f"{array.dtype}. Build the Fields with force_complex_fields=True (complex64) or "
        f"False (float32)."
    )


#: Flat index arrays for a source's point list, keyed by the identity of its
#: three index arrays and holding references so the ids stay unique. A source's
#: points are built once and injected every step, so this is a per-step hit.
_POINT_LINEAR: Dict[Tuple[int, int, int], Tuple[Any, Any, Any, Any]] = {}


def _linear_points(array: Any, ix: Any, iy: Any, iz: Any) -> Any:
    """C-order flat indices of ``array[ix, iy, iz]``, cached, checked unique once.

    The deposit gathers, subtracts and scatters each listed cell once, so a repeated
    index would keep one of its differences where MEEP's loop subtracts both; and a
    device scatter with a repeated index is unordered. Every table this engine builds
    lists each cell once (:func:`_point_index_arrays` refuses one that does not); this
    re-checks it the first time a table reaches an array rather than trusting it.
    """
    key = (id(ix), id(iy), id(iz))
    hit = _POINT_LINEAR.get(key)
    if hit is not None and hit[0] is ix and hit[1] is iy and hit[2] is iz:
        return hit[3]
    linear = host_writes.ravel_points(ix, iy, iz, array.shape)
    if host_writes.index_module(linear).unique(linear).size != linear.size:
        raise ValueError(
            "a source deposits into the same cell more than once; MEEP subtracts each "
            "listed point separately (one rounding per point) and the vectorized deposit "
            "would keep only one of them, so it is refused")
    _POINT_LINEAR[key] = (ix, iy, iz, linear)
    return linear


def _is_plus_zero(value: float) -> bool:
    """``value`` is ``+0.0``: equal to zero AND its sign bit clear (``-0.0 == 0.0`` too)."""
    return value == 0.0 and math.copysign(1.0, value) > 0.0


#: Per deposition table, keyed by the identity of its amplitude array: whether every
#: amplitude times the current ``(+0, +0)`` forms ``+0`` parts, on real storage and on
#: complex storage (:func:`_silent_amplitude_planes`). A weak reference to the array
#: guards the key against reuse and keeps nothing alive.
_SILENT_AMPLITUDES: Dict[int, Tuple[Any, bool, bool]] = {}

#: Per (``condinv`` volume, deposition cells), under the same weak-reference guard:
#: whether every ``condinv`` word the deposit reads at those cells is finite with its
#: sign bit clear. A volume the fields replace is read again; none is kept alive.
_SILENT_CONDINV: Dict[Tuple[int, int, int, int], Tuple[Tuple[Any, ...], bool]] = {}


def _host_values(values: Any) -> np.ndarray:  # A NumPy copy of a device array; NumPy passes through.
    get = getattr(values, "get", None)
    return np.asarray(get() if callable(get) else values)


def _silent_amplitude_planes(amplitudes: Any) -> Tuple[bool, bool]:
    """Does each amplitude times the current ``(+0, +0)`` give ``+0`` parts?

    For a finite amplitude ``a`` the products of :func:`_deposit` are zeros carrying the
    sign of their amplitude part, and IEEE 754 (6.3, round to nearest) gives a sum of two
    zeros their common sign and ``+0`` otherwise. So ``re = (a_re * +0) - (a_im * +0)`` is
    ``-0`` exactly where ``a_re`` has its sign bit set and ``a_im`` does not (the point is
    FLAGGED in the real plane), and ``im = (a_re * +0) + (a_im * +0)`` is ``-0`` exactly
    where both have it (flagged in the imaginary plane). An infinite or NaN part gives
    NaN. Returns (real storage, complex storage): True when every amplitude is finite and
    no point is flagged in a plane that storage takes.
    """
    parts = _host_values(amplitudes).astype(np.complex128).reshape(-1)
    re, im = parts.real, parts.imag
    if not bool(np.all(np.isfinite(re) & np.isfinite(im))):
        return False, False
    negative_re, negative_im = np.signbit(re), np.signbit(im)
    real_plane = not bool(np.any(negative_re & ~negative_im))
    return real_plane, real_plane and not bool(np.any(negative_re & negative_im))


def _silent_amplitudes(amps: Any, keep_real: bool) -> bool:
    """:func:`_silent_amplitude_planes` of this table, read once (one host copy) and kept."""
    hit = _SILENT_AMPLITUDES.get(id(amps))
    if hit is None or hit[0]() is not amps:
        try:
            reference = weakref.ref(amps)
        except TypeError:  # no weak reference to it: keep nothing, and the table deposits
            return False
        hit = (reference, *_silent_amplitude_planes(amps))
        _SILENT_AMPLITUDES[id(amps)] = hit
    return hit[1] if keep_real else hit[2]


def _silent_condinv(cndinv: Any, ix: Any, iy: Any, iz: Any) -> bool:
    """Is every ``condinv`` word read at the cells finite, with its sign bit clear?

    Then ``(+-0) * double(condinv)`` keeps the zero's sign (``+0`` included: ``(+0) * (+0)
    = +0``). Read once per (volume, cells) and kept: a ``condinv`` volume is formed once
    per conductivity and never written in place (``Fields._set_conductivity_side``
    installs a new array each time), so a replaced conductivity is a new key.
    """
    if cndinv is None:
        return True
    held = (cndinv, ix, iy, iz)
    key = (id(cndinv), id(ix), id(iy), id(iz))
    hit = _SILENT_CONDINV.get(key)
    if hit is not None and all(ref() is obj for ref, obj in zip(hit[0], held)):
        return hit[1]
    try:
        references = tuple(weakref.ref(obj) for obj in held)
    except TypeError:  # no weak reference to one of them: keep nothing, and the table deposits
        return False
    words = _host_values(cndinv[ix, iy, iz])
    verdict = bool(np.all(np.isfinite(words) & ~np.signbit(words)))
    _SILENT_CONDINV[key] = (references, verdict)
    return verdict


def _silent_deposit_is_identity(amps: Any, ix: Any, iy: Any, iz: Any, dt: Optional[float],
                                cndinv: Any, keep_real: bool) -> bool:
    """Does this table's deposit of the current ``(+0, +0)`` subtract ``+0`` at every point?

    True exactly when ``dt`` is None or finite and positive, every amplitude is finite
    and none is flagged in a plane the storage takes (:func:`_silent_amplitude_planes`),
    and every ``condinv`` word at the cells is finite with its sign bit clear
    (:func:`_silent_condinv`): then ``* dt`` and ``* double(condinv)`` keep each ``+0``,
    and every subtrahend :func:`_deposit` would form is ``+0``. The verdicts are static
    per table and read once, from host copies, so a step reads no device word for them.
    """
    if dt is not None:
        step = float(dt)
        if not (math.isfinite(step) and step > 0.0):
            return False
    return _silent_amplitudes(amps, keep_real) and _silent_condinv(cndinv, ix, iy, iz)


def _deposit(array: Any, ix: Any, iy: Any, iz: Any, amps: Any, factor: complex,
             dt: Optional[float], component: str, cndinv: Any = None) -> None:
    """MEEP's source deposit at a source's points, in MEEP's double precision.

    The statement transcribed is ``fields_chunk::step_source`` (step.cpp:295-317)::

        A = sv.current(j) * dt                      // * double(cndinv[i]) with a conductivity
        f[c][0][i] -= real(A);  if (!is_real) f[c][1][i] -= imag(A);

    with ``sv.current(j) = amp[j] * src_t->current()`` (meep_internals.hpp:69), all
    ``complex<double>``. Per point, in MEEP's order and as separate float64 ufuncs:

    * the complex product first, as libc++ forms it on its finite branch:
      ``re = (amp_re * c_re) - (amp_im * c_im)`` and, for complex storage,
      ``im = (amp_re * c_im) + (amp_im * c_re)``;
    * then ``* dt``, then ``* double(cndinv[i])`` when ``cndinv`` is given;
    * then :func:`host_writes.subtract_double`: the stored word widened to double,
      ``real(A)`` subtracted once, rounded once to float32 (and the imaginary word
      against ``imag(A)``).

    NEVER A COMPLEX UFUNC MULTIPLY. A complex128 multiply is one ufunc with two
    products and an add inside it, and a host that contracts it fuses them: NumPy on an
    x86 host with AVX-512 matches ``fma(ar, br, -(ai*bi))`` in 0 of 2,000,000 words
    and the separately rounded form in 491,977 (phase-0 complex family,
    ``numpy_cmul_check_gpu_host.txt``). MEEP's coff and portable builds round every
    product. Real-by-real float64 multiplies and adds are one operation each and cannot
    fuse on any host.

    ONE EARLY RETURN, ONLY WHERE IT IS AN IDENTITY. MEEP's loop has none
    (step.cpp:309-314), and a zero deposit is not word-neutral in general:
    ``(-0) - (-0) = +0``, so a stored ``-0`` meets a ``real(A)`` of ``-0`` (a negative
    real amplitude and a silent current) and becomes ``+0`` in MEEP. It is neutral
    exactly when every subtrahend is ``+0``: ``float(double(f) - (+0))`` is ``f`` for
    every float32 word, both zeros included (``(-0) - (+0) = -0``). So the deposit
    returns before forming anything only when the current is bitwise ``(+0, +0)`` --
    both sign bits read, never ``== 0``, which a ``-0`` part passes -- the array is the
    C-contiguous array the door accepts (so a refusal by name stays one), and
    :func:`_silent_deposit_is_identity` holds for the table: ``dt`` finite and positive
    (or None), every amplitude finite, the ``condinv`` words at its cells finite with
    the sign bit clear, and no FLAGGED point. With ``J = (+0, +0)`` the real line is
    ``(a_re * +0) - (a_im * +0)``, which is ``-0`` where ``a_re`` has its sign bit set
    and ``a_im`` does not, and the imaginary line is ``(a_re * +0) + (a_im * +0)``,
    ``-0`` where both have it. Every other deposit, a flagged table's silent one
    included, runs the statement below, in list order. Under flush the widening also
    reads a subnormal ``f`` as a zero of its sign, so there the identity also needs no
    subnormal at the table's cells; a flushing run stores none (every arithmetic result
    is flushed, and the fills and wall clears copy, negate or zero; pinned on a run
    that decays through signed zeros, ``test_source_deposit.py``). A threshold
    (``abs(current) < eps``) or the source's ``last_time()`` is never the predicate: a
    tail current of ~1e-100 is not zero, and its deposit gives a stored zero the sign
    of ``-real(A)``.

    ``factor`` is the time factor the amplitudes multiply: the current for MEEP's
    ``step_source``, with ``dt`` the step; for an integrated source's ``f_minus_p``
    (:func:`f_minus_p_begin`) it is the dipole, with ``dt`` and ``cndinv`` None, which
    forms ``real(amp[j] * dipole)`` as ``update_eh.cpp:131-134`` subtracts it from
    ``f_minus_p``. ``cndinv`` is the target component's ``condinv`` volume
    (``s->condinv[c][d]``, read at the deposit cells and widened exactly), or None
    where the component is lossless -- MEEP's NULL pointer, which skips the multiply.
    Where a conductive component's sigma is zero its ``condinv`` word is exactly 1.0,
    so the multiply is exact there and reading the volume at every point is MEEP's
    per-chunk choice cell for cell.

    THE REAL PART IS TAKEN UNCONDITIONALLY, INCLUDING WHEN THE PER-POINT AMPLITUDE
    CARRIES PHASE. MEEP takes ``real(A)`` with no condition attached (step.cpp:307),
    and a run that refused a phased amplitude on real storage would not be stricter
    than MEEP, only a run MEEP completes and this one does not. A Gaussian beam's
    amplitude is complex BY CONSTRUCTION (wavefront curvature and the Gouy phase), and
    every oblique eigenmode launch carries an in-plane phase ramp. MEASURED against CPU
    MEEP stepping the same ``mp.Simulation``, complex relative L2 over the whole
    volume, before the deposit moved to double:

    ============================  ==========  ============  ===========
    real-field run                parity      |A| launch    complex twin
    ============================  ==========  ============  ===========
    gaussian-beam.py cell, CW     1.9e-07 …   7.5e-01       —
                                  7.1e-07
    the same cell, GaussianSource 1.8e-07 …   7.5e-01 …     —
                                  5.0e-07     9.2e-01
    2-D TM / TE beam              1.6e-07 …   1.0e+00 …     —
                                  2.0e-06     1.5e+00
    3-D beam (`beam_three_d`)     1.5e-07,    —             1.4e-07,
                                  2.8e-07                   2.8e-07
    oblique-source.py's cell      1.2e-06     —             1.2e-06
    ============================  ==========  ============  ===========

    The real run's ``real(A)`` line is the complex run's first line word for word (the
    same float64 operations on the same operands), so a real run is the real part of
    the complex run. The suite carries the subtler spelling of the mistake,
    ``drop_beam_phase`` -- ``Re(a)*s`` in place of ``Re(a*s)`` -- at 6.6e-01 and
    7.0e-01 (``test_from_meep.py``, cases ``beam_real_fields`` /
    ``beam_real_fields_pulse`` / ``eigenmode_oblique_waveguide_real_fields``).
    """
    keep_real = _injects_real(array, component)
    linear = _linear_points(array, ix, iy, iz)
    current = complex(factor)
    c_re, c_im = current.real, current.imag
    if (_is_plus_zero(c_re) and _is_plus_zero(c_im)
            and bool(getattr(getattr(array, "flags", None), "c_contiguous", False))
            and _silent_deposit_is_identity(amps, ix, iy, iz, dt, cndinv, keep_real)):
        return
    amp_re, amp_im = amps.real, amps.imag
    real_part = amp_re * c_re
    real_part -= amp_im * c_im
    if dt is not None:
        real_part *= float(dt)
    scale = None
    if cndinv is not None:
        scale = cndinv[ix, iy, iz].astype(host_writes.index_module(real_part).float64)
        real_part *= scale
    if keep_real:
        host_writes.subtract_double(array, linear, real_part)
        return
    imag_part = amp_re * c_im
    imag_part += amp_im * c_re
    if dt is not None:
        imag_part *= float(dt)
    if scale is not None:
        imag_part *= scale
    host_writes.subtract_double(array, linear, real_part, imag_part)


# ---------------------------------------------------------------------------
# MEEP's src_vol list: per chunk, merged in double
# ---------------------------------------------------------------------------


class SrcVolTable(NamedTuple):
    """One entry of MEEP's ``src_vol`` list that a source deposits.

    ``ix`` / ``iy`` / ``iz`` / ``amps`` as :func:`_index_arrays` packs them; ``points``
    is their length. A source deposits a tuple of these in order (:func:`_deposit_tables`).
    """

    ix: Any
    iy: Any
    iz: Any
    amps: Any
    points: int


def _own_table(source: Any) -> Tuple[List[Tuple[int, int, int]], List[complex]]:
    """A source's own (indices, amplitudes): every point it places, its amplitude each."""
    if hasattr(source, "_point_indices"):
        return list(source._point_indices or ()), list(source._point_weights or ())
    return list(getattr(source, "_indices", None) or ()), list(getattr(source, "_amplitudes", None) or ())


def _deposit_tables(source: Any) -> Tuple[SrcVolTable, ...]:
    """What ``source`` deposits now, in order: its bound entries, else its own table.

    The bound entries are the ones :func:`bind_src_vols` gave it -- possibly none, when
    every piece was absorbed into an earlier source's entry. A source never bound (built
    and injected without a driver) deposits its own table. Empty when there is nothing
    to deposit.
    """
    bound = getattr(source, "_src_vol", None)
    if bound is not None:
        return tuple(table for table in bound if table.points)
    ix = getattr(source, "_point_ix", None)
    if ix is None:
        return ()
    # ``size``, not ``shape[0]``: a table whose index parts are scalars is one point.
    return (SrcVolTable(ix, source._point_iy, source._point_iz, source._point_amps,
                        int(getattr(ix, "size", 1))),)


def _point_chunks(layout: Any, grid: 'Grid', component: str,
                  indices: List[Tuple[int, int, int]]) -> np.ndarray:
    """MEEP's chunk of each point of one component: ``grid_volume::owns`` (vec.cpp:445-462).

    The stored row ``i`` of a component with Yee shift ``s`` is the doubled coordinate
    ``origin + 2*i + s`` (``chunk_map.AxisFrame``); a wrapping axis translates a
    coordinate on or below the window's low face by the period, as ``ChunkLayout``
    resolves its own rows. ``-1`` where no chunk owns the point.
    """
    coordinates = []
    for axis, frame in enumerate(layout.frames):
        rows = np.asarray([point[axis] for point in indices], dtype=np.int64)
        points = frame.origin + 2 * rows + _iyee_shift(component, axis)
        if frame.wraps:
            low, high = frame.window
            period = high - low
            points = np.where(points <= low, points + period, points)
            points = np.where(points > high, points - period, points)
        coordinates.append(points)
    return np.asarray(layout.chunk_of(coordinates, cylindrical=bool(getattr(grid, "cylindrical", False))))


def source_time_key(source: Any) -> Tuple[Any, ...]:
    """Which shared MEEP ``src_time`` instance a source deposits through, as data.

    MEEP keeps one instance per class of equal src_times (``src_time::add_to`` and
    ``src_times_equal``: ``is_equal`` both ways and the same ``is_integrated``,
    sources.cpp:33-52), and two pieces combine only through the same instance. The
    key is computed from the source's own parameters, never from a bound MEEP object,
    so a driver rebuilt from a capsule merges exactly as the lifted one did. A source
    class this module does not define is a src_time of its own.
    """
    method = getattr(source, "src_time_key", None)
    if method is None:
        return ("object", id(source))
    return method()


def _source_pieces(source: Any, layout: Any, grid: 'Grid',
                   ) -> List[Tuple[int, List[Tuple[int, int, int]], List[complex]]]:
    """One source's MEEP ``src_vol`` pieces on ``layout``: (chunk, cells, amplitudes), in order.

    A source placed by :func:`_source_geometry` is cut per (lattice shift, chunk) by
    :func:`_src_vol_pieces`. One that carries only a table (a class this module does not
    define, or a stub) is cut per chunk, its cells in table order.
    """
    geometry = getattr(source, "_src_geometry", None)
    if geometry is not None:
        return _src_vol_pieces(geometry, layout.boxes, grid)
    indices, amps = _own_table(source)
    if not indices:
        return []
    component = str(source.component)
    chunks = _point_chunks(layout, grid, component, indices)
    order: Dict[int, List[int]] = {}
    for position, chunk in enumerate(chunks.tolist()):
        order.setdefault(int(chunk), []).append(position)
    return [(chunk, [indices[j] for j in positions], [complex(amps[j]) for j in positions])
            for chunk, positions in order.items()]


def bind_src_vols(sources: Any, layout: Any, grid: 'Grid') -> Dict[str, int]:
    """Give every source the entries it deposits in MEEP's ``src_vol`` lists.

    MEEP adds a source as one piece per (lattice shift, chunk) its request reaches
    (``loop_in_chunks`` with ``src_vol_chunkloop``, sources.cpp:243-312), and
    ``fields_chunk::add_source`` (fields.cpp:576-585) appends the piece to that chunk's
    list unless an entry already in the list has the same component (the D and B
    spellings fold onto E and H, sources.cpp:167-173), the same shared src_time and the
    same index vector; then the piece's amplitudes are summed into the entry,
    ``amp[i] += other.amp[i]`` in ``complex<double>`` (sources.cpp:183-188), and the pair
    deposits once. Pieces that share a cell without sharing their index vector -- the
    two ends of a request across a periodic seam -- stay two entries, and the cell is
    deposited twice (step.cpp:309-314).

    ``step_source`` walks each chunk's list in order. Every entry is created by a source
    no later than any source merged into it, the cells of different chunks are disjoint,
    and a source's own entries are created in its piece order; so the package's step,
    which deposits the sources in their order and each source's entries in creation
    order, subtracts every cell's terms in MEEP's order. Each source's ``_src_vol``
    becomes the tuple of entries it created, with the merged amplitudes (empty when an
    earlier source absorbed all of its pieces). The integrated sources are listed and
    merged the same way; their entries are what :func:`f_minus_p_begin` subtracts.

    ``layout`` is the run's ``chunk_map.ChunkLayout``. Returns counts: ``pieces``,
    ``merged`` (pieces absorbed into an earlier entry), ``entries`` and ``sources_rebound``.
    """
    sources = tuple(sources or ())
    chunk_entries: Dict[Tuple[int, Tuple[Any, ...]], Dict[str, Any]] = {}
    created: List[Dict[str, Any]] = []
    pieces = merged = 0
    for index, source in enumerate(sources):
        component = str(getattr(source, "component", ""))
        if component not in _ARRAY_FOR_COMPONENT:
            continue
        array = _ARRAY_FOR_COMPONENT[component]
        time_key = source_time_key(source)
        for chunk, cells, amps in _source_pieces(source, layout, grid):
            pieces += 1
            key = (chunk, (array, time_key, tuple(cells)))
            entry = chunk_entries.get(key)
            if entry is None:
                entry = {"owner": index, "cells": cells, "amps": list(amps)}
                chunk_entries[key] = entry
                created.append(entry)
                continue
            entry["amps"] = [complex(a.real + b.real, a.imag + b.imag)
                             for a, b in zip(entry["amps"], amps)]
            merged += 1
    owned: Dict[int, List[SrcVolTable]] = {}
    for entry in created:
        ix, iy, iz, packed = _index_arrays(grid, entry["cells"], entry["amps"])
        owned.setdefault(entry["owner"], []).append(SrcVolTable(ix, iy, iz, packed, len(entry["cells"])))
    rebound = 0
    for index, source in enumerate(sources):
        if not hasattr(source, "_src_vol"):
            continue
        source._src_vol = tuple(owned.get(index, ()))
        rebound += 1
    return {"pieces": pieces, "merged": merged, "entries": len(created), "sources_rebound": rebound}


# ---------------------------------------------------------------------------
# f_minus_p: the integrated sources' dipole, at the constitutive update
# ---------------------------------------------------------------------------


class FMinusP(NamedTuple):
    """The words :func:`f_minus_p_begin` replaced: per array, the flat cells and their words."""

    field_type: str
    saved: Tuple[Tuple[str, Any, Any], ...]


def _integrated_tables(sources: Any, field_type: str) -> List[Tuple[Any, SrcVolTable]]:
    """The integrated sources' entries of one field family, in deposit order."""
    chosen = []
    for source in tuple(sources or ()):
        if str(getattr(source, "field_type", "")) != field_type:
            continue
        if not bool(getattr(source, "is_integrated", False)):
            continue
        for table in _deposit_tables(source):
            chosen.append((source, table))
    return chosen


def has_integrated(sources: Any, field_type: str) -> bool:
    """Does any integrated source of this field family deposit into the run?"""
    return bool(_integrated_tables(sources, field_type))


def _boundary_passes(fields: 'Fields', field_type: str) -> None:
    """``step_boundaries`` of one family as the driver runs it after the deposit.

    The mirror fill, the wall clear and the folded far fill, in the driver's order
    (``FdtdDriver.step``). Each writes a component's ghost or wall cells from that
    component's own owned cells (or zeroes them), so running them again changes only
    the cells that image, or lie on, a cell written since they last ran.
    """
    from . import stepping  # noqa: PLC0415 - stepping imports nothing from here

    if field_type == FIELD_TYPE_B:
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
    else:
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)


def f_minus_p_begin(fields: 'Fields', sources: Any, field_type: str,
                    time: float) -> Optional[FMinusP]:
    """Write MEEP's ``f_minus_p`` words at the integrated sources' points; return what they replaced.

    MEEP (update_eh.cpp:66-140) copies D (B) into ``f_minus_p`` and, for every
    integrated ``src_vol`` entry of the family in list order, subtracts
    ``real(amp[j] * dipole)`` per point (and ``imag`` from the imaginary part of complex
    storage): a ``realnum -= double``, so one subtraction in double and one rounding,
    with no ``dt`` and no conductivity. The constitutive update then reads
    ``f_minus_p`` in place of D at every cell. ``time`` is the slot ``calc_sources``
    evaluated: ``time() + dt`` for E and ``time() + dt/2`` for H (step.cpp:74, :105).

    Here the copy is made in place: the words at the integrated cells are saved, then
    each entry's dipole is subtracted from them through :func:`_deposit` (``dt`` None),
    and the family's boundary passes run again so every ghost and wall cell the driver
    images from, or clears at, an integrated cell holds what it would hold had D been
    ``f_minus_p`` when they ran: the pointwise constitutive update at a ghost then
    reads the image of ``f_minus_p``, and a wall cell stays the wall's zero. Call it
    after the family's boundary passes and immediately before the constitutive
    consult; :func:`f_minus_p_end` undoes it. ``None`` when no integrated source of the
    family deposits (nothing was written).

    NOT FOR A RUN WITH A LIVE POLARIZATION: MEEP subtracts P first and the dipole after
    (update_eh.cpp:120-134), while the package's constitutive update subtracts P from
    the replaced word, ``(D - dipole) - P``. The words then differ from MEEP's in the
    last bit, as they did with the standing offset this replaces.
    """
    entries = _integrated_tables(sources, field_type)
    if not entries:
        return None
    grouped: Dict[str, List[Any]] = {}
    for source, table in entries:
        name = _ARRAY_FOR_COMPONENT[str(source.component)]
        array = host_writes.raw(fields, name)
        grouped.setdefault(name, []).append(host_writes.ravel_points(table.ix, table.iy, table.iz,
                                                                     array.shape))
    saved = []
    for name, parts in grouped.items():
        array = host_writes.raw(fields, name)
        xp = host_writes.index_module(parts[0])
        linear = xp.unique(xp.concatenate(parts)) if len(parts) > 1 else parts[0]
        saved.append((name, linear, host_writes.gather(array, linear)))
    # MEEP's src_t->dipole() is the step's one cached value (src_time::update, called once
    # per step by calc_sources), whatever the number of entries reading it.
    dipoles: Dict[int, complex] = {}
    for source, table in entries:
        array = _array_for(fields, source.component)
        if id(source) not in dipoles:
            dipoles[id(source)] = source.dipole(time)
        _deposit(array, table.ix, table.iy, table.iz, table.amps, dipoles[id(source)], None,
                 source.component)
    _boundary_passes(fields, field_type)
    return FMinusP(field_type, tuple(saved))


def f_minus_p_end(fields: 'Fields', held: Optional[FMinusP]) -> None:
    """Restore the words :func:`f_minus_p_begin` replaced, and image them again.

    MEEP's D (B) never held ``f_minus_p``: the integrated cells take their saved words
    back and the family's boundary passes run again, so every ghost and wall cell holds
    MEEP's D (B) too. ``None`` is a no-op.
    """
    if held is None:
        return
    for name, linear, words in held.saved:
        host_writes.scatter(host_writes.raw(fields, name), linear, words)
    _boundary_passes(fields, held.field_type)


def _forward_difference(dipole: Callable[[float], complex], time: float, dt: float) -> complex:
    """MEEP's ``src_time::current`` (meep.hpp:983-985): ``(dipole(t + dt) - dipole(t)) / dt``.

    ``complex<double> - complex<double>`` and ``complex<double> / double`` act per part,
    so each part is one subtraction and one division. Spelled per part here because a
    Python ``complex / float`` promotes the divisor to a complex and runs Smith's
    quotient, which adds ``imag * 0.0`` to the real part (a ``-0`` real part comes back
    ``+0``).
    """
    late = complex(dipole(time + dt))
    early = complex(dipole(time))
    return complex((late.real - early.real) / dt, (late.imag - early.imag) / dt)


def _f32(value: float) -> float:  # MEEP stores some window bounds as float(x) in a double.
    return float(np.float32(value))


class _MeepSrcTimeBinding:
    """The lifted simulation's own MEEP ``src_time``, when the lift bound one.

    Mixed into every class whose ``dipole`` / ``current`` a deposit calls: the dedicated
    electric source classes and the three envelopes (a :class:`VolumeSource` binds its
    envelope). A class using it declares the dataclass field ``_meep_src_time``: the
    MEEP object (``src.swigobj``, or the clone ``add_eigenmode_source`` retunes), which
    a capsule does not carry. A capsule of a lifted run therefore loads only with the
    currents its lift recorded (``capsule._require_evaluable_sources``), never with the
    package's own waveform in their place.

    Bound, ``dipole(t)`` and ``current(t, dt)`` return MEEP's own
    ``src_time::dipole`` / ``src_time::current``, the values ``calc_sources`` caches
    for ``step_source`` and ``update_eh`` (step.cpp:278-283; meep.hpp:968-975). The
    carrier's words depend on the libm entry and the complex division the C++ build
    landed on, which only MEEP's own object knows (phase 0, P0-d3: on Linux the
    package's ``math.cos`` / ``math.sin`` carrier misses 11 of 8,833 rows that glibc's
    ``sincos`` / ``cexp`` reproduce).
    """

    def bind_meep_src_time(self, src_time: Any) -> None:
        """Form every dipole and current through ``src_time`` from now on."""
        if not (callable(getattr(src_time, "dipole", None))
                and callable(getattr(src_time, "current", None))):
            raise ValueError(
                f"bind_meep_src_time takes MEEP's src_time (dipole(t) and current(t, dt)), "
                f"got {type(src_time).__name__}")
        integrated = getattr(src_time, "is_integrated", None)
        if integrated is not None and bool(integrated) != bool(self.is_integrated):
            raise ValueError(
                f"the bound src_time has is_integrated={bool(integrated)} and this "
                f"{type(self).__name__} has is_integrated={bool(self.is_integrated)}; MEEP's "
                f"step_source and update_eh select the slot on that flag")
        self._meep_src_time = src_time

    def _meep_dipole(self, time: float) -> complex:
        return complex(self._meep_src_time.dipole(float(time)))

    def _meep_current(self, time: float, dt: float) -> complex:
        return complex(self._meep_src_time.current(float(time), float(dt)))


class SourceEnvelope:
    """
    Base class for the temporal part of a source — MEEP's `src_time`.

    A subclass supplies `dipole(time)`; `current(time, dt)` defaults to MEEP's
    forward difference (meep.hpp:983-985) and may be overridden, as
    `CustomEnvelope` does.

    `is_integrated` selects MEEP's second injection path: an integrated source is
    skipped by `step_source` (step.cpp:300) and its *dipole* is subtracted inside
    `update_eh` instead, so what reaches the field is the time integral of the
    current. MEEP's Python layer defaults every source to False, and so does this.
    """

    is_integrated: bool = False

    def dipole(self, time: float) -> complex:  # Dipole moment at one time; subclasses override.
        raise NotImplementedError("A SourceEnvelope subclass must implement dipole(time)")

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current — forward difference.
        return _forward_difference(self.dipole, time, dt)


@dataclass
class ContinuousEnvelope(_MeepSrcTimeBinding, SourceEnvelope):
    """
    CW carrier with MEEP's tanh turn-on/turn-off ramp — `continuous_src_time`.

    Same waveform as `ContinuousSource` / `ExtendedSource` carry inline; this is
    the form `VolumeSource` composes with, so a magnetic or integrated CW source
    reuses the exact waveform the electric classes are pinned against.
    """

    frequency: float
    width: float = 0.0            # Turn-on/turn-off ramp width (MEEP default 0 = no ramp).
    slowness: float = _MEEP_SLOWNESS
    start_time: float = 0.0
    end_time: float = _MEEP_ENDLESS
    is_integrated: bool = False

    _omega: float = field(default=0.0, repr=False, init=False)
    _amp_factor: complex = field(default=1j, repr=False, init=False)
    _meep_src_time: Any = field(default=None, repr=False, init=False, compare=False)

    def __post_init__(self):  # Resolve the carrier constants; a zero/non-finite frequency raises.
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:128-140 continuous_src_time::dipole.
        if self._meep_src_time is not None:
            return self._meep_dipole(time)
        return _continuous_dipole(
            time, self._omega, self._amp_factor,
            self.start_time, self.end_time, self.width, self.slowness,
        )

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current.
        if self._meep_src_time is not None:
            return self._meep_current(time, dt)
        return _forward_difference(self.dipole, time, dt)

    def src_time_key(self) -> Tuple[Any, ...]:  # MEEP's src_times_equal, as data (sources.cpp:33-36, :145-151).
        return _continuous_key(self.frequency, self.width, self.start_time, self.end_time,
                               self.slowness, self.is_integrated)


@dataclass
class GaussianEnvelope(_MeepSrcTimeBinding, SourceEnvelope):
    """
    Gaussian pulse about a peak — MEEP's `gaussian_src_time`.

    MEEP's conventions, not the textbook ones: `width = 1/fwidth`,
    `peak_time = start_time + cutoff*width`, carrier phase referenced to
    `t - peak_time`, and a hard cut outside `cutoff*width` (shrunk until the
    envelope there is representable).
    """

    frequency: float
    fwidth: float
    start_time: float = 0.0
    cutoff: float = 5.0           # Envelope extent in widths (MEEP GaussianSource default).
    is_integrated: bool = False

    _omega: float = field(default=0.0, repr=False, init=False)
    _amp_factor: complex = field(default=1j, repr=False, init=False)
    _width: float = field(default=0.0, repr=False, init=False)
    _peak_time: float = field(default=0.0, repr=False, init=False)
    _cutoff_time: float = field(default=0.0, repr=False, init=False)
    _meep_src_time: Any = field(default=None, repr=False, init=False, compare=False)

    def __post_init__(self):  # Resolve width, peak and cut; a zero frequency or non-positive fwidth raises.
        if self.fwidth <= 0.0:
            raise ValueError(f"Gaussian source fwidth must be positive, got {self.fwidth}")
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104
        self._width = 1.0 / self.fwidth
        self._peak_time = self.start_time + self.cutoff * self._width
        self._cutoff_time = _gaussian_cutoff_time(self.cutoff, self._width)

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:98-107 gaussian_src_time::dipole.
        if self._meep_src_time is not None:
            return self._meep_dipole(time)
        return _gaussian_dipole(time, self._omega, self._amp_factor,
                                self._peak_time, self._width, self._cutoff_time)

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current.
        if self._meep_src_time is not None:
            return self._meep_current(time, dt)
        return _forward_difference(self.dipole, time, dt)

    def src_time_key(self) -> Tuple[Any, ...]:  # MEEP's src_times_equal, as data (sources.cpp:33-36, :117-124).
        return _gaussian_key(self.frequency, self._width, self._peak_time, self._cutoff_time,
                             self.is_integrated)

    @property
    def peak_time(self) -> float:  # Time of the envelope peak.
        return self._peak_time

    @property
    def width(self) -> float:  # Temporal width (1/fwidth), the envelope standard deviation.
        return self._width


@dataclass
class CustomEnvelope(_MeepSrcTimeBinding, SourceEnvelope):
    """
    Caller-supplied waveform — MEEP's `custom_src_time` / Python `CustomSource`.

    `func(t) -> complex` is evaluated in MEEP natural units and gated to
    `[start_time, end_time]`. Which quantity it represents follows MEEP
    (meep.hpp:1066-1071): with `is_integrated=False` — the default — the
    function *is* the current, so it is injected directly rather than
    differenced; with `is_integrated=True` it is the dipole moment and the
    current becomes its forward difference.

    A function that returns zero deposits a zero, which MEEP's loop subtracts like
    any other value (step.cpp:309-314); the field changes only where a stored ``-0``
    meets it.
    """

    func: Callable[[float], complex]
    start_time: float = _MEEP_CUSTOM_START
    end_time: float = _MEEP_CUSTOM_END
    is_integrated: bool = False

    _meep_src_time: Any = field(default=None, repr=False, init=False, compare=False)

    def __post_init__(self):  # A non-callable waveform is a caller bug, not a zero source.
        if not callable(self.func):
            raise ValueError(
                f"CustomEnvelope needs a callable func(t) -> complex, got {type(self.func).__name__}"
            )

    def dipole(self, time: float) -> complex:  # MEEP meep.hpp:1072-1078 custom_src_time::dipole.
        if self._meep_src_time is not None:
            return self._meep_dipole(time)
        return _custom_dipole(time, self.func, self.start_time, self.end_time)

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp:1066-1071 custom_src_time::current.
        if self._meep_src_time is not None:
            return self._meep_current(time, dt)
        if self.is_integrated:
            return _forward_difference(self.dipole, time, dt)
        return self.dipole(time)

    def src_time_key(self) -> Tuple[Any, ...]:
        """MEEP's ``custom_src_time::is_equal`` (sources.cpp:154-161), as data.

        MEEP compares the function by identity and the window as stored, plus the
        ``center_frequency`` and ``fwidth`` a Python ``CustomSource`` carries; this
        envelope carries neither, so two custom sources that share a function and a
        window but declare different centre frequencies would merge here and not in
        MEEP. A lift passes each ``CustomSource``'s own function, so sharing one is
        the caller's explicit choice.
        """
        return ("custom", id(self.func), _f32(self.start_time), _f32(self.end_time),
                bool(self.is_integrated))


def _continuous_key(frequency: float, width: float, start_time: float, end_time: float,
                    slowness: float, integrated: bool) -> Tuple[Any, ...]:
    """``continuous_src_time::is_equal`` (sources.cpp:145-151) plus ``is_integrated``.

    MEEP stores the window as ``float(st)`` / ``float(et)`` (meep.hpp:1038-1039).
    """
    return ("continuous", complex(frequency), float(width), _f32(start_time), _f32(end_time),
            float(slowness), bool(integrated))


def _gaussian_key(frequency: float, width: float, peak_time: float, cutoff_time: float,
                  integrated: bool) -> Tuple[Any, ...]:
    """``gaussian_src_time::is_equal`` (sources.cpp:117-124) plus ``is_integrated``."""
    return ("gaussian", float(frequency), float(width), float(peak_time), float(cutoff_time),
            bool(integrated))


@dataclass
class ContinuousSource(_MeepSrcTimeBinding):
    """
    Continuous-wave point source.

    Translation of MEEP's continuous_src_time (sources.cpp); extended
    (sheet/volume) geometries belong to ExtendedSource. The temporal waveform —
    including MEEP's tanh turn-on/turn-off ramp — matches meep 1.29.0 exactly;
    `width = 0` is MEEP's default hard switch-on, and a nonzero `width` suppresses
    the switch-on transient over roughly `slowness` widths.

    The spatial placement is `_build_source_points`, the same translation of
    MEEP's `loop_in_chunks` that every other source class uses, applied to a
    zero-size volume. It used to be an independent trilinear 8-neighbour stencil
    built on `Grid.position_to_index`, and the two agree exactly wherever the
    stencil stays inside the axis — for a zero-extent direction MEEP's Simpson
    ladder collapses to the same pair of linear weights — but they part company at
    the ends, which is where a point source at a cell face lives:

    * The stencil could not fold onto the lattice. A request past the last sample
      of a wrapping axis was placed by extrapolating off the end instead of onto
      the image at the far face, and the negative half of the extrapolation was
      then dropped by a positive-weight filter, leaving ~1.5x the requested
      current on one cell (2x at an integer-placed face).
    * The stencil could not carry the Bloch phase, so even the cells it did reach
      would have been unphased.

    Measured against CPU MEEP, complex relative L2 over the Ez volume of a
    1 x 1 x 2 cell at resolution 10: 5.4e-1 at a z cell face and 1.2e0 at an x
    cell face before, 2.2e-7 worst after, at both cell parities and with a Bloch phase.
    """
    grid: 'Grid'
    frequency: float
    component: str = 'Ex'
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: complex = 1.0 + 0j
    width: float = 0.0            # Turn-on/turn-off ramp width (MEEP default 0 = no ramp).
    slowness: float = _MEEP_SLOWNESS
    start_time: float = 0.0
    end_time: float = _MEEP_ENDLESS

    # Internal state — the deposition table, in MEEP's src_vol form: one entry per
    # touched cell, each amplitude carrying the user amplitude, its interpolation weight,
    # the delta-function scaling `a` per zero-size axis, and any Bloch factor the
    # lattice fold picked up.
    _omega: float = field(default=0.0, repr=False, init=False)
    _amp_factor: complex = field(default=1j, repr=False, init=False)
    _point_indices: list = field(default=None, repr=False, init=False)
    _point_weights: list = field(default=None, repr=False, init=False)
    # The same table packed for the deposit, built once (`_pack_points`).
    _point_ix: Any = field(default=None, repr=False, init=False)
    _point_iy: Any = field(default=None, repr=False, init=False)
    _point_iz: Any = field(default=None, repr=False, init=False)
    _point_amps: Any = field(default=None, repr=False, init=False)
    # The table this source deposits in MEEP's merged src_vol list (`bind_src_vols`),
    # or None for its own table; and the lift's MEEP src_time (`bind_meep_src_time`).
    _src_vol: Any = field(default=None, repr=False, init=False, compare=False)
    # The request as MEEP's loop_in_chunks cuts it (`_source_geometry`), kept so a
    # division can be bound later without evaluating the amplitude function again.
    _src_geometry: Any = field(default=None, repr=False, init=False, compare=False)
    _meep_src_time: Any = field(default=None, repr=False, init=False, compare=False)

    def __post_init__(self):  # Resolve the waveform constants and the deposition stencil.
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104
        self._setup_distribution()

    def _pack_points(self) -> None:  # Pack the deposition table for the deposit.
        """Build the backend index/amplitude arrays :meth:`inject` deposits through.

        The weights are packed as they are, in complex128, and :func:`_deposit` forms
        MEEP's ``(amp[j] * current) * dt`` from them: the product first, then ``dt``.
        The user ``amplitude`` is inside each weight, as MEEP folds it into ``amp[j]``
        (sources.cpp:243-312: ``IVEC_LOOP_WEIGHT * (amplitude * a) * A``), and the
        current is the unit waveform's. Carrying it in :meth:`dipole` instead, as this
        class once did, forms ``w * (amplitude * J)`` where MEEP forms
        ``(w * amplitude) * J``, which differs in the last bit of the double product
        (phase 0: 119 of 800 rows of one CW source in the sources family).
        """
        indices = self._point_indices
        if not indices:
            self._point_ix = self._point_iy = self._point_iz = None
            self._point_amps = None
            return
        xp = self.grid.xp
        self._point_ix, self._point_iy, self._point_iz = _point_index_arrays(self.grid, indices)
        self._point_amps = xp.asarray(self._point_weights, dtype=xp.complex128)

    def _setup_distribution(self):  # Point sources only; extended geometry has its own class.
        if all(s == 0.0 for s in self.size):
            self._setup_point_source()
        else:
            raise NotImplementedError("Use ExtendedSource for non-point sources")

    def _setup_point_source(self):  # MEEP loop_in_chunks deposition for a zero-size volume.
        """Place the point through MEEP's own machinery, lattice fold included.

        `_build_source_points` is called with the user amplitude, which MEEP folds into
        every per-point amplitude; everything else it returns — the Simpson weights
        (which for a zero-extent direction are the two linear interpolation weights,
        or a single 1.0 when the request lands exactly on a sample), the `a` per
        zero-size axis of MEEP sources.cpp:481-484, and `conj(bloch_phase ** shift)`
        on any cell reached by wrapping — belongs in the per-point amplitude too.

        The symmetry check is the same pair every other source class runs. A point
        source used to skip the on-the-plane half of it, so a CW point requested off
        an active mirror plane was placed in the stored quadrant as if it were on it —
        one source where the fold means two.
        """
        _validate_component(self.component)
        _validate_symmetry(self.grid, self.center)
        # No amp_func on this class at all (the driver refuses one on a point source), so a
        # nonzero `size` here is a UNIFORM sheet — the profile the parity rule refuses by
        # inspection. Passing `size` is what tells it which of the two cases this is.
        _validate_symmetry_parity(self.grid, self.component, self.center, self.size)
        self._src_geometry = _source_geometry(
            self.grid, self.component, self.center, self.size, complex(self.amplitude), None,
        )
        self._point_indices, self._point_weights = _summed_table(self._src_geometry, self.grid)
        self._pack_points()

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:128-140 — the user amplitude is in the point amps.
        if self._meep_src_time is not None:
            return self._meep_dipole(time)
        return _continuous_dipole(
            time, self._omega, self._amp_factor,
            self.start_time, self.end_time, self.width, self.slowness,
        )

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current — forward difference.
        if self._meep_src_time is not None:
            return self._meep_current(time, dt)
        return _forward_difference(self.dipole, time, dt)

    def src_time_key(self) -> Tuple[Any, ...]:  # MEEP's src_times_equal, as data.
        return _continuous_key(self.frequency, self.width, self.start_time, self.end_time,
                               self.slowness, False)

    @property
    def field_type(self) -> str:  # Electric source: injected between step_D and update_E.
        return FIELD_TYPE_D

    @property
    def is_integrated(self) -> bool:  # This class only implements MEEP's non-integrated path.
        return False

    def inject(self, fields: 'Fields', time: float):  # MEEP step.cpp step_source — subtract the current from D.
        tables = _deposit_tables(self)
        if not tables:
            return
        dt = self.grid.dt
        J = self.current(time, dt)
        D = _d_array_for(fields, self.component)
        cndinv = _condinv_at(fields, _validate_component(self.component))
        # One deposit per entry of MEEP's src_vol list, in its order: a point that two
        # pieces reach (a periodic seam) is subtracted twice. The delta-function scaling
        # is already inside each amplitude -- MEEP folds it in too.
        for table in tables:
            _deposit(D, table.ix, table.iy, table.iz, table.amps, J, dt, self.component, cndinv)

    def __repr__(self) -> str:
        return (f"ContinuousSource(freq={self.frequency:.4f}, "
                f"component={self.component}, center={self.center})")


@dataclass
class ExtendedSource(_MeepSrcTimeBinding):
    """
    Continuous-wave sheet or volume source.

    The spatial machinery is a translation of MEEP's loop_in_chunks: doubled
    (ivec) bounds via vec2diel_floor/ceil, Simpson boundary weights with MEEP's
    exact case ladder, owned-corner clipping against the stored (possibly
    symmetry-halved) grid, and per-point amplitudes sampled once at setup. The
    temporal waveform is the same continuous_src_time as ContinuousSource,
    including the tanh ramp.

    `amp_func(x_rel, y_rel, z_rel) -> complex` is evaluated at each point's
    offset from the source center; all three offsets are passed.
    """
    grid: 'Grid'
    frequency: float
    component: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    amplitude: complex = 1.0 + 0j
    amp_func: Optional[Callable[[float, float, float], complex]] = None
    width: float = 0.0            # Turn-on/turn-off ramp width (MEEP default 0 = no ramp).
    slowness: float = _MEEP_SLOWNESS
    start_time: float = 0.0
    end_time: float = _MEEP_ENDLESS

    # Internal state — the analog of MEEP's src_vol index_array / amps_array
    _omega: float = field(default=0.0, repr=False, init=False)
    _amp_factor: complex = field(default=1j, repr=False, init=False)
    _indices: List[Tuple[int, int, int]] = field(default=None, repr=False, init=False)
    _amplitudes: List[complex] = field(default=None, repr=False, init=False)
    _point_ix: Any = field(default=None, repr=False, init=False)
    _point_iy: Any = field(default=None, repr=False, init=False)
    _point_iz: Any = field(default=None, repr=False, init=False)
    _point_amps: Any = field(default=None, repr=False, init=False)
    _n_source_points: int = field(default=0, repr=False, init=False)
    _src_vol: Any = field(default=None, repr=False, init=False, compare=False)
    # The request as MEEP's loop_in_chunks cuts it (`_source_geometry`), kept so a
    # division can be bound later without evaluating the amplitude function again.
    _src_geometry: Any = field(default=None, repr=False, init=False, compare=False)
    _meep_src_time: Any = field(default=None, repr=False, init=False, compare=False)

    def __post_init__(self):  # MEEP sources.cpp:456-489 add_volume_source.
        _validate_component(self.component)  # Electric components only; magnetic ones go through VolumeSource.
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104
        self._validate_symmetry()
        self._setup_source_points()

    def _validate_symmetry(self):  # A source must sit in the stored half, and carry the fold's parity.
        _validate_symmetry(self.grid, self.center)
        _validate_symmetry_parity(self.grid, self.component, self.center, self.size, self.amp_func)

    def _setup_source_points(self):  # Build the per-point index/amplitude tables.
        self._src_geometry = _source_geometry(
            self.grid, self.component, self.center, self.size, self.amplitude, self.amp_func
        )
        self._indices, self._amplitudes = _summed_table(self._src_geometry, self.grid)
        self._setup_index_arrays()

    def _setup_index_arrays(self):  # Pack the tables into backend arrays for vectorized injection.
        self._n_source_points = len(self._indices)
        if self._n_source_points == 0:
            self._point_ix = self._point_iy = self._point_iz = self._point_amps = None
            return
        (self._point_ix, self._point_iy,
         self._point_iz, self._point_amps) = _index_arrays(self.grid, self._indices, self._amplitudes)

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:128-140 — the user amplitude is folded into the point amps.
        if self._meep_src_time is not None:
            return self._meep_dipole(time)
        return _continuous_dipole(
            time, self._omega, self._amp_factor,
            self.start_time, self.end_time, self.width, self.slowness,
        )

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current — forward difference.
        if self._meep_src_time is not None:
            return self._meep_current(time, dt)
        return _forward_difference(self.dipole, time, dt)

    def src_time_key(self) -> Tuple[Any, ...]:  # MEEP's src_times_equal, as data.
        return _continuous_key(self.frequency, self.width, self.start_time, self.end_time,
                               self.slowness, False)

    @property
    def field_type(self) -> str:  # Electric source: injected between step_D and update_E.
        return FIELD_TYPE_D

    @property
    def is_integrated(self) -> bool:  # This class only implements MEEP's non-integrated path.
        return False

    def inject(self, fields: 'Fields', time: float):  # MEEP step.cpp:295-317 — f -= real((amp * current) * dt).
        D = _d_array_for(fields, self.component)
        tables = _deposit_tables(self)
        if not tables:
            return
        dt = self.grid.dt
        J = self.current(time, dt)
        cndinv = _condinv_at(fields, _validate_component(self.component))
        for table in tables:  # MEEP's src_vol entries, in list order.
            _deposit(D, table.ix, table.iy, table.iz, table.amps, J, dt, self.component, cndinv)

    def __repr__(self) -> str:
        return (f"ExtendedSource(freq={self.frequency:.4f}, "
                f"component={self.component}, center={self.center}, size={self.size})")


@dataclass
class GaussianPulsedSource(_MeepSrcTimeBinding):
    """
    Gaussian-pulsed sheet or volume source, for broadband runs.

    Spatially identical to ExtendedSource; temporally this is MEEP's
    gaussian_src_time (sources.cpp:85-126):

        dipole(t) = exp(-(t - peak)^2 / (2*width^2)) * exp(-i*omega*(t - peak)) * amp_factor

    with `width = 1/fwidth`, `peak = start_time + cutoff*width`, and a hard cut
    at `|t - peak| > cutoff*width`. Both conventions differ from the bundle this
    module was ported from (port reference §5.5), which used
    `width = 1/(2*pi*fwidth)` and referenced the carrier phase to absolute time;
    the MEEP conventions above are now canonical and are verified bit-for-bit
    against meep 1.29.0.
    """
    grid: 'Grid'
    frequency: float
    fwidth: float
    component: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    amplitude: complex = 1.0 + 0j
    amp_func: Optional[Callable[[float, float, float], complex]] = None
    start_time: float = 0.0
    cutoff: float = 5.0           # Envelope extent in widths (MEEP GaussianSource default).

    # Internal state
    _omega: float = field(default=0.0, repr=False, init=False)
    _amp_factor: complex = field(default=1j, repr=False, init=False)
    _width: float = field(default=0.0, repr=False, init=False)
    _peak_time: float = field(default=0.0, repr=False, init=False)
    _cutoff_time: float = field(default=0.0, repr=False, init=False)
    _indices: List[Tuple[int, int, int]] = field(default=None, repr=False, init=False)
    _amplitudes: List[complex] = field(default=None, repr=False, init=False)
    _point_ix: Any = field(default=None, repr=False, init=False)
    _point_iy: Any = field(default=None, repr=False, init=False)
    _point_iz: Any = field(default=None, repr=False, init=False)
    _point_amps: Any = field(default=None, repr=False, init=False)
    _n_source_points: int = field(default=0, repr=False, init=False)
    _src_vol: Any = field(default=None, repr=False, init=False, compare=False)
    # The request as MEEP's loop_in_chunks cuts it (`_source_geometry`), kept so a
    # division can be bound later without evaluating the amplitude function again.
    _src_geometry: Any = field(default=None, repr=False, init=False, compare=False)
    _meep_src_time: Any = field(default=None, repr=False, init=False, compare=False)

    def __post_init__(self):  # Resolve the pulse constants and the spatial tables.
        if self.fwidth <= 0.0:
            raise ValueError(f"Gaussian source fwidth must be positive, got {self.fwidth}")
        _validate_component(self.component)  # Electric components only; magnetic ones go through VolumeSource.
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104

        # MEEP sources.cpp:94-95 — width = 1/fwidth, peak = midpoint of [start, start + 2*cutoff*width]
        self._width = 1.0 / self.fwidth
        self._peak_time = self.start_time + self.cutoff * self._width
        self._cutoff_time = _gaussian_cutoff_time(self.cutoff, self._width)

        self._validate_symmetry()
        self._setup_source_points()

    def _validate_symmetry(self):  # A source must sit in the stored half, and carry the fold's parity.
        _validate_symmetry(self.grid, self.center)
        _validate_symmetry_parity(self.grid, self.component, self.center, self.size, self.amp_func)

    def _setup_source_points(self):  # Same spatial machinery as ExtendedSource.
        self._src_geometry = _source_geometry(
            self.grid, self.component, self.center, self.size, self.amplitude, self.amp_func
        )
        self._indices, self._amplitudes = _summed_table(self._src_geometry, self.grid)
        self._setup_index_arrays()

    def _setup_index_arrays(self):  # Pack the tables into backend arrays for vectorized injection.
        self._n_source_points = len(self._indices)
        if self._n_source_points == 0:
            self._point_ix = self._point_iy = self._point_iz = self._point_amps = None
            return
        (self._point_ix, self._point_iy,
         self._point_iz, self._point_amps) = _index_arrays(self.grid, self._indices, self._amplitudes)

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:109-120 gaussian_src_time::dipole.
        if self._meep_src_time is not None:
            return self._meep_dipole(time)
        return _gaussian_dipole(time, self._omega, self._amp_factor,
                                self._peak_time, self._width, self._cutoff_time)

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current — forward difference.
        if self._meep_src_time is not None:
            return self._meep_current(time, dt)
        return _forward_difference(self.dipole, time, dt)

    def src_time_key(self) -> Tuple[Any, ...]:  # MEEP's src_times_equal, as data.
        return _gaussian_key(self.frequency, self._width, self._peak_time, self._cutoff_time, False)

    @property
    def field_type(self) -> str:  # Electric source: injected between step_D and update_E.
        return FIELD_TYPE_D

    @property
    def is_integrated(self) -> bool:  # This class only implements MEEP's non-integrated path.
        return False

    def inject(self, fields: 'Fields', time: float):  # MEEP step.cpp:295-317 — f -= real((amp * current) * dt).
        D = _d_array_for(fields, self.component)
        tables = _deposit_tables(self)
        if not tables:
            return
        dt = self.grid.dt
        J = self.current(time, dt)
        cndinv = _condinv_at(fields, _validate_component(self.component))
        # Every table is handed to `_deposit` outside the pulse too: it returns early only
        # where MEEP's subtraction would leave every word, never on a stored -0 it flips.
        for table in tables:  # MEEP's src_vol entries, in list order.
            _deposit(D, table.ix, table.iy, table.iz, table.amps, J, dt, self.component, cndinv)

    @property
    def peak_time(self) -> float:  # Time of the envelope peak.
        return self._peak_time

    @property
    def width(self) -> float:  # Temporal width (1/fwidth), the envelope standard deviation.
        return self._width

    def __repr__(self) -> str:
        return (f"GaussianPulsedSource(freq={self.frequency:.4f}, "
                f"fwidth={self.fwidth:.4f}, component={self.component})")


@dataclass
class VolumeSource:
    """
    General current source: any field family, any temporal envelope.

    The spatial machinery is `ExtendedSource`'s — MEEP's `loop_in_chunks`, driven
    entirely by the component's Yee shift — so a point, sheet or volume source is
    selected by `size` exactly as there. What this class adds is the two axes the
    electric CW/Gaussian classes do not cover:

    *Magnetic components.* `Hx`/`Hy`/`Hz` (and their `Bx`/`By`/`Bz` spellings)
    inject into B with the same sign MEEP uses for both families
    (step.cpp:295-312). They belong in a different slot of the step sequence,
    which `field_type` reports — see the module docstring.

    *Envelopes.* `envelope` is a `SourceEnvelope` (or a bare `func(t) -> complex`,
    wrapped in a `CustomEnvelope`), so a caller-supplied waveform is expressible
    alongside the built-in CW and Gaussian forms, and `is_integrated` rides on it.

    `amp_func(x_rel, y_rel, z_rel) -> complex` is evaluated at each point's offset
    from the source center, as in `ExtendedSource`.
    """

    grid: 'Grid'
    component: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    envelope: Any
    amplitude: complex = 1.0 + 0j
    amp_func: Optional[Callable[[float, float, float], complex]] = None

    # Internal state — the analog of MEEP's src_vol index_array / amps_array, and the
    # table this source deposits in MEEP's merged src_vol list (`bind_src_vols`).
    _field_type: str = field(default=FIELD_TYPE_D, repr=False, init=False)
    _indices: List[Tuple[int, int, int]] = field(default=None, repr=False, init=False)
    _amplitudes: List[complex] = field(default=None, repr=False, init=False)
    _point_ix: Any = field(default=None, repr=False, init=False)
    _point_iy: Any = field(default=None, repr=False, init=False)
    _point_iz: Any = field(default=None, repr=False, init=False)
    _point_amps: Any = field(default=None, repr=False, init=False)
    _n_source_points: int = field(default=0, repr=False, init=False)
    _src_vol: Any = field(default=None, repr=False, init=False, compare=False)
    # The request as MEEP's loop_in_chunks cuts it (`_source_geometry`), kept so a
    # division can be bound later without evaluating the amplitude function again.
    _src_geometry: Any = field(default=None, repr=False, init=False, compare=False)

    def __post_init__(self):  # MEEP sources.cpp:456-489 add_volume_source, for either field family.
        _validate_any_component(self.component)
        self._field_type = _field_type_for(self.component)
        if callable(self.envelope) and not isinstance(self.envelope, SourceEnvelope):
            self.envelope = CustomEnvelope(func=self.envelope)
        if not isinstance(self.envelope, SourceEnvelope):
            raise ValueError(
                f"VolumeSource needs a SourceEnvelope or a callable func(t) -> complex, "
                f"got {type(self.envelope).__name__}"
            )
        _validate_symmetry(self.grid, self.center)
        _validate_symmetry_parity(self.grid, self.component, self.center, self.size, self.amp_func)
        # An is_integrated source runs UNDER MIRROR SYMMETRY too. Its f_minus_p words are
        # written at the owned points and imaged into the ghost rows by the driver's own
        # fill passes (`f_minus_p_begin`), so the ghost E a pointwise constitutive update
        # produces is the parity image of the owned E, the value MEEP's E-side
        # step_boundaries copies; the restore runs the passes again, so every ghost D
        # then holds MEEP's D. MEEP itself folds the combination exactly (measured
        # folded vs unfolded MEEP, an integrated full-height Ez sheet through a
        # Mirror(Y) with PML: 1.7e-07 with PML on x, 1.8e-07 with PML everywhere). A
        # blanket refusal cost finite_grating.py, absorbed_power_density.py and
        # mie_scattering.py their lifts.
        self._setup_source_points()

    def _setup_source_points(self):  # Build the per-point index/amplitude tables.
        self._src_geometry = _source_geometry(
            self.grid, self.component, self.center, self.size, self.amplitude, self.amp_func
        )
        self._indices, self._amplitudes = _summed_table(self._src_geometry, self.grid)
        self._n_source_points = len(self._indices)
        if self._n_source_points == 0:
            self._point_ix = self._point_iy = self._point_iz = self._point_amps = None
            return
        (self._point_ix, self._point_iy,
         self._point_iz, self._point_amps) = _index_arrays(self.grid, self._indices, self._amplitudes)

    @property
    def field_type(self) -> str:  # 'D' (inject after step_D) or 'B' (inject after step_B).
        return self._field_type

    @property
    def is_integrated(self) -> bool:  # Whether the envelope injects the time-integrated current.
        return self.envelope.is_integrated

    def dipole(self, time: float) -> complex:  # Envelope dipole; the user amplitude lives in the point amps.
        return self.envelope.dipole(time)

    def current(self, time: float, dt: float) -> complex:  # Envelope current (MEEP src_time::current or an override).
        return self.envelope.current(time, dt)

    def bind_meep_src_time(self, src_time: Any) -> None:  # The lift's MEEP src_time, on the envelope.
        """Form this source's dipole and current through MEEP's own ``src_time``.

        Bound on the envelope, which is the object this class asks: see
        :class:`_MeepSrcTimeBinding`.
        """
        if not isinstance(self.envelope, _MeepSrcTimeBinding):
            raise ValueError(
                f"this VolumeSource's envelope ({type(self.envelope).__name__}) cannot take "
                f"a MEEP src_time")
        self.envelope.bind_meep_src_time(src_time)

    def src_time_key(self) -> Tuple[Any, ...]:  # MEEP's src_times_equal, as data (the envelope's).
        key = getattr(self.envelope, "src_time_key", None)
        if key is None:
            # An envelope class this module does not define: one instance per source,
            # as MEEP's add_to would find no equal src_time to share.
            return ("envelope", id(self.envelope), bool(self.is_integrated))
        return key()

    def inject(self, fields: 'Fields', time: float):  # Subtract this step's current from D or B.
        """Apply one timestep of this source: MEEP's ``step_source`` (step.cpp:295-317).

        ``f -= real((amp[j] * current(time)) * dt)`` per point, with ``time`` the slot
        time of the component's field family. An integrated source deposits NOTHING
        here: ``step_source`` skips it (step.cpp:300) and its dipole enters at the
        constitutive update through ``f_minus_p`` (:func:`f_minus_p_begin`).
        """
        array = _array_for(fields, self.component)
        if self.envelope.is_integrated:
            return
        tables = _deposit_tables(self)
        if not tables:
            return
        dt = self.grid.dt
        J = self.envelope.current(time, dt)
        cndinv = _condinv_at(fields, _validate_any_component(self.component))
        # MEEP's step_source: every entry of its src_vol list, in list order, handed to
        # `_deposit` on a silent envelope too (it skips only a word-neutral table).
        for table in tables:
            _deposit(array, table.ix, table.iy, table.iz, table.amps, J, dt, self.component, cndinv)

    def __repr__(self) -> str:
        return (f"VolumeSource(component={self.component}, center={self.center}, "
                f"size={self.size}, envelope={type(self.envelope).__name__})")


def create_gaussian_beam_source(
    grid: 'Grid',
    frequency: float,
    waist_radius: float,
    source_z: float,
    focus_position: Tuple[float, float, float],
    source_size: Tuple[float, float],
    n_medium: float = 1.0,
    component: str = 'Ex',
    amplitude: complex = 1.0 + 0j,
) -> ExtendedSource:
    """Build a converging Gaussian beam as a z-normal sheet source.

    The sheet carries the beam's transverse Gaussian profile evaluated at the
    source plane plus the spherical phase that focuses it at `focus_position`.
    `waist_radius` is the waist at focus and `wavelength = 1/frequency` is the
    vacuum wavelength (natural units).
    """
    fx, fy, fz = focus_position
    wavelength = 1.0 / frequency
    distance_to_focus = fz - source_z

    z_R = n_medium * np.pi * waist_radius ** 2 / wavelength
    w_at_source = waist_radius * np.sqrt(1 + (distance_to_focus / z_R) ** 2)
    k_m = 2 * np.pi * n_medium / wavelength

    def amp_func(x: float, y: float, z: float) -> complex:  # Transverse profile plus converging spherical phase.
        del z  # A z-normal sheet has no z extent; the offset is always zero here.
        r_sq = (x - fx) ** 2 + (y - fy) ** 2
        gaussian_amp = np.exp(-r_sq / w_at_source ** 2)
        spherical_dist = np.sqrt(distance_to_focus ** 2 + r_sq)
        phase = -k_m * (spherical_dist - distance_to_focus)
        return complex(gaussian_amp * np.exp(1j * phase))

    return ExtendedSource(
        grid=grid,
        frequency=frequency,
        component=component,
        center=(0.0, 0.0, source_z),
        size=(source_size[0], source_size[1], 0.0),
        amplitude=amplitude,
        amp_func=amp_func,
    )
