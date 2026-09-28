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
`_build_source_points`: a translation of MEEP's `loop_in_chunks` including
Simpson boundary weights, symmetry-chunk clipping, the lattice fold that
carries a request past a periodic face back onto the cell with its Bloch phase,
and — on a mirror-folded axis whose outer boundary is periodic — MEEP's
lattice-shift image passes, which land a full-width request's far edge weight
on the top of the stored window (loop_in_chunks.cpp:390-415 keys that loop on
the boundary being Periodic, not on the symmetry).
`create_gaussian_beam_source` is a factory that dresses an `ExtendedSource` with
a converging Gaussian-beam amplitude function.

The temporal waveform is a separate object so it can be reused across the two
field families and replaced wholesale: `ContinuousEnvelope`, `GaussianEnvelope`
and `CustomEnvelope` translate MEEP's `continuous_src_time`, `gaussian_src_time`
and `custom_src_time`. Each carries MEEP's `is_integrated` flag.

Inputs: a `Grid` (which supplies the array module `xp`, resolution, dt, cell
counts and symmetry flags) plus the source specification. Output: a set of grid
points with precomputed complex amplitudes, plus `dipole(t)` / `current(t, dt)`
waveforms; `inject(fields, time)` applies `D -= dt * current(t) * amp` at those
points, which is MEEP's `step_source` (step.cpp:295-298).

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

An `is_integrated` source injects the time-integrated current instead. MEEP
excludes it from `step_source` and subtracts the dipole inside `update_eh`
(update_eh.cpp:126-140, `f_minus_p = D - dipole`); this module reaches the same
E and H by subtracting the whole fresh `dipole` into D or B in the injection
slot and WITHDRAWING it again before the next curl ladder runs
(`VolumeSource.withdraw`, called by the driver ahead of step_D / step_B), so the
array holds exactly `f - dipole` when the constitutive update reads it and
exactly MEEP's f everywhere else. It used to subtract the *increment*
`dipole(t) - dipole(t_previous)` and leave the offset standing, which telescopes
to the same thing ONLY while nothing rescales the array between injections — the
split PML recurrence's dsigu damping and a D conductivity both do, and both
measured broken (4.59e-01 and 6.38e-02 against CPU MEEP) before the withdraw
slot existed, at the floor (9e-07, 2e-07) after. The equivalence still needs the
mirror-plane repair pass not to copy offset-bearing cells, so an integrated
source on a symmetric grid raises rather than running slightly wrong.

Real-valued fields — MEEP's default — are injected into as well. MEEP's
`step_source` writes `f[c][0][i] -= real(A)` and only touches the imaginary plane
`f[c][1][i]` when the run is complex (step.cpp:295-312), so the real-field
current is the real part of the SAME complex product `amp[j] * current(t) * dt`
this module already forms. That is not an approximation of the complex run: the
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
`_step_source_values` carries the table. This module used to refuse that pairing
on the grounds that `Re(a)Re(s) - Im(a)Im(s)` re-reads a spatial phase as a
per-point time shift; it does, and the resulting field is still exactly the real
part of the complex run, so the refusal only declined runs MEEP completes. The
temporal current stays complex in both modes and its real part is taken by
design: that IS the real-field convention, not a dropped term.

Units are MEEP natural units throughout (c = 1, frequency = 1/wavelength); the
micrometre conversion belongs to the bridge layer above. Real arrays are
float32, complex arrays complex64. No array is created here except through
`grid.xp`, so the module runs unchanged on NumPy and CuPy.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .grid import METALLIC
from . import host_writes

if TYPE_CHECKING:
    from .fields import Fields
    from .grid import Grid

# Debug flag: when True, source setup re-checks the linear-index uniqueness
# invariant that vectorized injection relies on (see _setup_index_arrays).
DEBUG_VALIDATE_SOURCE_INDICES = False

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
    # RAW: every caller hands this array to :func:`_sparse_subtract`, which moves
    # only the source cells. A barriered fetch would sync the whole volume first.
    return host_writes.raw(fields, _validate_component(component))


def _array_for(fields: 'Fields', component: str) -> Any:  # Resolve the D or B array a component drives.
    return host_writes.raw(fields, _validate_any_component(component))


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
    and the envelope is cut hard outside `cutoff_time`.
    """
    t_rel = time - peak_time
    if np.float32(abs(t_rel)) > cutoff_time:
        return 0.0 + 0j
    envelope = math.exp(-0.5 * (t_rel / width) ** 2)
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


def _build_source_points(grid: 'Grid', component: str, center: Tuple[float, float, float],
                         size: Tuple[float, float, float], amplitude: complex,
                         amp_func: Optional[Callable[[float, float, float], complex]],
                         ) -> Tuple[List[Tuple[int, int, int]], List[complex]]:
    """Translate a source volume into (index, amplitude) pairs, following MEEP's loop_in_chunks.

    Line-by-line from sources.cpp:478-489 (add_volume_source), loop_in_chunks.cpp:349-359
    (bounds), :418-489 (chunk intersection and weight adjustment), sources.cpp:243-312
    (src_vol_chunkloop) and vec.hpp:151-168, 365-383 (LOOP_OVER_IVECS, IVEC_LOOP_LOC,
    IVEC_LOOP_WEIGHT).

    Every step is driven by the component's Yee shift, so this serves magnetic
    components unchanged; the callers decide which components they accept.
    """
    _validate_any_component(component)
    a = grid.resolution  # MEEP: gv.a
    inva = grid.dx       # MEEP: gv.inva = 1/a

    cx, cy, cz = center
    sx, sy, sz = size

    # Source bounds (the MEEP "where" volume)
    where_min = [cx - sx / 2, cy - sy / 2, cz - sz / 2]
    where_max = [cx + sx / 2, cy + sy / 2, cz + sz / 2]

    # MEEP sources.cpp:467-476 add_volume_source — the request is measured against the
    # cell BEFORE anything else happens to it. Wider than one cell plus one pixel and
    # MEEP aborts; wider than one cell by less than a pixel and MEEP snaps it back to
    # exactly one cell. Neither is cosmetic here: the ladder below folds every rung onto
    # the cell modulo the lattice, so a request spanning more than one period lands two
    # or more lattice images on the same cells and SUMS them, which is a silently
    # over-driven source that still returns a smooth field. `Lx`/`Ly`/`Lz` are the full
    # cell lengths on a folded axis too, which is the length MEEP compares against
    # (its user_volume spans the whole cell, not the stored quadrant).
    lengths = (grid.Lx, grid.Ly, grid.Lz)
    for d in range(3):
        span = where_max[d] - where_min[d]
        if span > lengths[d] + inva:
            raise ValueError(
                f"Source size {size[d]} along {('x', 'y', 'z')[d]} exceeds the cell length "
                f"{lengths[d]} by more than one cell ({inva}); MEEP's add_volume_source aborts on "
                f"exactly this request (sources.cpp:469, \"Source width > cell width\"). A source "
                f"wider than one lattice period deposits several of its own lattice images onto "
                f"the same cells and sums them, so it would run and be over-driven rather than "
                f"fail. Clamp the size to the cell."
            )
        if span > lengths[d]:
            # MEEP sources.cpp:471-474, transcribed with its asymmetry intact: the minimum
            # moves DOWN by half the excess and the maximum is then placed one cell length
            # above the NEW minimum, so the window shrinks to exactly one period and slides
            # down by the full excess rather than staying centred on the request.
            excess = span - lengths[d]
            where_min[d] = where_min[d] - excess * 0.5
            where_max[d] = where_min[d] + lengths[d]

    # MEEP sources.cpp:481-484 — delta-function scaling: amp *= gv.a per zero-size
    # direction, EXCEPT one that MEEP's `nosize_direction` reports:
    #
    #     if (where.in_direction(d) == 0.0 && !nosize_direction(d)) data.amp *= gv.a;
    #
    # A one-pixel PERIODIC axis is MEEP's own lower-dimensional emulation, so a zero
    # extent there means "the whole of this direction", not "a delta function at this
    # point", and no 1/length correction is due. Dropping that clause scales every
    # source in a 2-D or 1-D run by the resolution — a factor of 10 at res=10 — and
    # the field it produces is smooth, stable and exactly ten times too large.
    # `Grid.nosize_direction` is the transcription; it is a no-op on any axis with
    # more than one cell, so a 3-D run reaches the same product it always did.
    amp = amplitude
    for d in range(3):
        if size[d] == 0.0 and not _nosize_direction(grid, d):
            amp = amp * a

    # MEEP loop_in_chunks.cpp:352-353
    #   vec yee_c(gv.yee_shift(Centered) - gv.yee_shift(cgrid));
    #   ivec iyee_c(gv.iyee_shift(Centered) - gv.iyee_shift(cgrid));
    iyee_c = [_iyee_shift('Centered', d) - _iyee_shift(component, d) for d in range(3)]
    yee_c = [iyee_c[d] * 0.5 * inva for d in range(3)]

    # MEEP loop_in_chunks.cpp:354-356
    #   volume wherec(where + yee_c);
    #   ivec is(vec2diel_floor(wherec.get_min_corner(), gv.a, ...) - iyee_c);
    #   ivec ie(vec2diel_ceil(wherec.get_max_corner(), gv.a, ...) - iyee_c);
    wherec_min = [where_min[d] + yee_c[d] for d in range(3)]
    wherec_max = [where_max[d] + yee_c[d] for d in range(3)]
    is_orig = [_vec2diel_floor(wherec_min[d], a) - iyee_c[d] for d in range(3)]
    ie_orig = [_vec2diel_ceil(wherec_max[d], a) - iyee_c[d] for d in range(3)]

    # MEEP loop_in_chunks.cpp:359 — compute_boundary_weights uses the ORIGINAL where, not wherec
    s0_orig = [0.0, 0.0, 0.0]
    s1_orig = [0.0, 0.0, 0.0]
    e0_orig = [0.0, 0.0, 0.0]
    e1_orig = [0.0, 0.0, 0.0]
    for d in range(3):
        s0, s1, e0, e1 = _compute_boundary_weights(
            where_min[d], where_max[d], is_orig[d], ie_orig[d], a
        )
        s0_orig[d] = s0
        s1_orig[d] = s1
        e0_orig[d] = e0
        e1_orig[d] = e1

    # MEEP loop_in_chunks.cpp:418-446 — chunk intersection.
    # Sources are added with use_symmetry=false (sources.cpp:487), so a symmetric
    # run stores only the positive quadrant and the source volume is clipped to it:
    #   symmetric axis  -> little_corner = io = -2 (MEEP vec.cpp halve(), icenter-2),
    #                      big_corner = n_full (doubled coords)
    #   full axis       -> little_corner = io = -n_full, big_corner = +n_full
    # vec.hpp:1102-1107:
    #   little_owned_corner(c) = little_corner + 2 - iyee_shift(c)
    #   big_owned_corner(c)    = big_corner - iyee_shift(c)
    # MEEP's little corner rounds the cell count DOWN TO EVEN (vec.cpp icenter(),
    # applied by Simulation._create_grid_volume's center_origin), so a centred axis
    # starts at -(n_full - n_full % 2) — -L/2 only for an even count. `Grid.
    # origin_doubled` is the one definition and carries the measured cost of the
    # parity; the fallback serves the stub grids in test_sources.py.
    iyee_comp = [_iyee_shift(component, d) for d in range(3)]
    symmetric = tuple(_axis_is_mirrored(grid, d) for d in range(3))
    full_counts = (grid.nx_full, grid.ny_full, grid.nz_full)
    origin_reader = getattr(grid, "origin_doubled", None)
    corner_reader = getattr(grid, "big_corner_doubled", None)
    io = [0, 0, 0]
    big_corner = [0, 0, 0]
    for d in range(3):
        if callable(origin_reader):
            io[d] = origin_reader(d)
        elif symmetric[d]:
            io[d] = -2
        else:
            io[d] = -(full_counts[d] - full_counts[d] % 2)
        # MEEP vec.hpp big_corner(): io + 2 * halve()'s num. That is Grid.
        # owned_cells, NOT the stored count: a folded PERIODIC axis at an even
        # full count stores one cell past MEEP's corner (the second-mirror plane
        # and its ghost slot), and depositing into that ghost slot would double
        # the current the far-plane fill then images there. The fallback serves
        # the stub grids in test_sources.py, whose stored counts are MEEP's.
        if callable(corner_reader):
            big_corner[d] = corner_reader(d)
        else:
            big_corner[d] = io[d] + 2 * (grid.nx, grid.ny, grid.nz)[d]

    little_owned = [io[d] + 2 - iyee_comp[d] for d in range(3)]
    big_owned = [big_corner[d] - iyee_comp[d] for d in range(3)]
    # MEEP vec.cpp:432-436 little_owned_corner — in Dcyl with the r origin at 0,
    # an owned corner that lands at doubled r=2 claws back to 0: the axis row of
    # an integer-r component (Ez, Ep, Hz) IS owned there, which is the ownership
    # statement behind the per-m axis rules that step it. The generic corner
    # (little_owned_corner0) excludes it, and with only that rule a source at
    # r=0 on Ez or Ep clips to nothing and injects zero — a silent no-op run
    # (measured rel L2 1.0, every component identically zero, before this).
    axis_reader = getattr(grid, "is_axis", None)
    if callable(axis_reader):
        for d in range(3):
            if axis_reader(d) and little_owned[d] == 2:
                little_owned[d] = 0

    # MEEP loop_in_chunks.cpp:392-446 — the lattice-shift loop. A source volume that
    # reaches a PERIODIC face is NOT clipped there: MEEP loops over the lattice shifts
    # whose translated cell still intersects the request, deposits each translated copy
    # on the cell it folds onto, and scales it by ``conj(eikna ** ishift)``
    # (sources.cpp:259, `amp = data->amp * conj(shift_phase)` — the source is specified
    # in the unfolded frame, so representing it inside the fundamental cell divides by
    # the Bloch factor rather than multiplying).
    #
    # Clipping instead — the whole request ladder truncated at the owned corner, with
    # the overhanging points clamped back onto the last cell — silently lost the
    # wrapped plane AND produced duplicate indices, which the fancy-index subtraction
    # in `_inject_points` drops all but one of. Measured against CPU MEEP on the
    # standard grating source (an Ey sheet spanning the full periodic cross section of
    # a 1 x 1 x 4 cell at resolution 15, PML in z only, transmitted flux at five
    # frequencies): 1.29e-01 at k = 0, where the missing plane is a full 1/15 of the
    # injected current, and 9.5e-03 at kx = 0.4, where the wrapped plane's own Bloch
    # factor happens to cancel most of what was dropped. Both complete and return a
    # plausible transmission spectrum; only the absolute power is wrong, and by an
    # amount that depends on k.
    #
    # An unfolded face the ladder can reach is necessarily periodic: an absorber
    # removes no wrap (it is a material, not a boundary condition — the wrapped-seam
    # deposit is measured, test_from_meep's `source_on_the_wrapped_pml_seam`).
    # `Grid.axis_wraps` is the one definition of which axes have a lattice vector; the
    # fallback expression is the same rule spelled out for the stub grids in
    # test_sources.py, which model the registration conventions and nothing else.
    #
    # A FOLDED axis is clipped to MEEP's owned corners — MEEP's own shift-0 pass,
    # `max/min` against the chunk's owned corners (loop_in_chunks.cpp:440,446) —
    # but when its outer boundary is PERIODIC the fold does not end the story:
    # `loop_in_chunks` keys the shift range on `boundaries[High][d] == Periodic`
    # (loop_in_chunks.cpp:393), NOT on the symmetry, so a mirror-folded periodic
    # axis still loops over lattice images of the request, each clipped to the
    # same owned window with its boundary weights read in the SOURCE frame
    # (`iscS += shifti` before the weight comparison, loop_in_chunks.cpp:452-469).
    # That image pass is how a full-width source stays uniform up to the second
    # mirror at +L/2: the direct pass leaves the top row its edge weight (0.5
    # with the volume endpoint ON the sample, 7/8 one rung inside), and the
    # ishift = -1 image of the volume's OTHER end lands exactly there carrying
    # the rest. Dropping the image pass was this engine's folded-live-face
    # divergence family: measured on the binary_grating configuration {PML on X,
    # Mirror(Y), Y periodic at k = 0, full-width Ez/Hz source}, the deposit
    # profile read [1, ..., 1, 0.5] against MEEP's uniform 1.0, the missing
    # half-row surfaced as spurious energy on the mirror-odd component (Hx up to
    # 3.8e+05 relative on a component MEEP leaves at roundoff), and whole-volume
    # parity sat at 1e-02..6e-02 where the windowed-source control holds 1e-06
    # (the `full_width_source_on_folded_live_face_*` parity cases in
    # test_from_meep.py; the per-row totals are pinned MEEP-free in
    # test_sources.py).
    wraps_reader = getattr(grid, "axis_wraps", None)
    if callable(wraps_reader):
        lattice = tuple(bool(wraps_reader(d)) for d in range(3))
    else:
        lattice = tuple(not _axis_is_mirrored(grid, d) for d in range(3))
    # The folded axes whose far face is LIVE (outer boundary periodic, so a second
    # mirror at +L/2 rather than a wall): exactly the axes whose lattice images
    # must be deposited. `is_metallic` is read defensively for the stub grids in
    # test_sources.py, which keep the historical clip-only behaviour.
    metallic_reader = getattr(grid, "is_metallic", None)
    folded_live = tuple(
        _axis_is_mirrored(grid, d)
        and callable(metallic_reader) and not metallic_reader(d)
        and callable(corner_reader)
        for d in range(3)
    )
    iscS = [is_orig[d] if lattice[d] else max(is_orig[d], little_owned[d]) for d in range(3)]
    iecS = [ie_orig[d] if lattice[d] else min(ie_orig[d], big_owned[d]) for d in range(3)]

    # MEEP loop_in_chunks.cpp:448-489 — adjust boundary weights where the chunk
    # boundary does not coincide with the source boundary.
    s0c = [1.0, 1.0, 1.0]
    s1c = [1.0, 1.0, 1.0]
    e0c = [1.0, 1.0, 1.0]
    e1c = [1.0, 1.0, 1.0]
    for d in range(3):
        # MEEP lines 454-462: chunk start at the source start keeps the source weights
        if iscS[d] == is_orig[d]:
            s0c[d] = s0_orig[d]
            s1c[d] = s1_orig[d]
        elif iscS[d] == is_orig[d] + 2:
            s0c[d] = s1_orig[d]  # first boundary cell skipped: s1 becomes the new s0
        # MEEP lines 463-469: chunk end at the source end keeps the source weights
        if iecS[d] == ie_orig[d]:
            e0c[d] = e0_orig[d]
            e1c[d] = e1_orig[d]
        elif iecS[d] == ie_orig[d] - 2:
            e0c[d] = e1_orig[d]  # last boundary cell dropped: e1 becomes the new e0
        # MEEP lines 470-489: narrow ranges
        if iecS[d] == iscS[d]:
            w = min(s0c[d], e0c[d])  # single point
            s0c[d] = w
            e0c[d] = w
            s1c[d] = w
            e1c[d] = w
        elif iecS[d] == iscS[d] + 2:
            w = min(s0c[d], e1c[d])  # two points: cross-adjust
            s0c[d] = w
            e1c[d] = w
            w = min(s1c[d], e0c[d])
            s1c[d] = w
            e0c[d] = w
        elif iecS[d] == iscS[d] + 4:
            w = min(s1c[d], e1c[d])  # three points: adjust the inner weights
            s1c[d] = w
            e1c[d] = w

    # MEEP vec.hpp:153 — loop_n = (ie - is) / 2 + 1, computed per pass inside
    # `deposit_pass` below: the direct pass and a lattice-image pass clip to
    # different windows, and an empty one skips itself rather than the whole
    # request (a request that intersects nothing returns no points naturally).

    # Per-axis stored cell count and wrap factor, for folding a ladder point that sits
    # outside the cell onto its lattice image. `grid.bloch_phase` is None at k = 0,
    # where every factor is exactly 1 and the fold is a plain modulo.
    # `bloch_phase` is read defensively: the source tests drive this through stub grids
    # that model the registration conventions and nothing else, and a grid without one
    # is plain-periodic, where every wrap factor is 1.
    cell_counts = (grid.nx, grid.ny, grid.nz)
    phase_reader = getattr(grid, "bloch_phase", None)
    wrap_phases = [
        phase_reader(d) if (lattice[d] and callable(phase_reader)) else None for d in range(3)
    ]

    # MEEP stores one more sample per axis than this engine's cell-count arrays,
    # so the plane ON a metallic high wall is a real array entry there — and an
    # OWNED one: `grid_volume::owns` (vec.cpp:445-462) includes the big corner,
    # and `big_owned_corner(c) = big_corner() - iyee_shift(c)` (vec.hpp) equals
    # the wall plane for an integer-stencil component, so `src_vol_chunkloop`
    # deposits its Simpson edge weight onto that entry (sources.cpp:264-267, the
    # `owns(iloc)` skip passes). The deposit is then INERT: `find_metals`
    # registers every owned point satisfying `on_metal_boundary` — the High test
    # is `here == big_corner` on a Metallic face (boundaries.cpp:187-201,
    # :311-340) — and `step_boundaries` zeroes them FIRST (step.cpp:245, "Do the
    # metals first!") in the same half-step in which `step_source` deposited
    # (step.cpp:100-103), before any curl or constitutive update reads the array.
    # This engine keeps the wall as the unstored zero ghost, so the exact image
    # of MEEP's store-then-zero is to DROP a ladder point landing on the wall
    # plane. It used to clamp that point onto the last interior cell instead,
    # which SUMS the wall's edge weight into a live ring: measured on the Dcyl
    # 3x6 cell at res 10 (CW, until 20, no absorber, whole-volume complex rel L2
    # against CPU MEEP 1.33.0) a full-radius Ep plane read 7.1e-02, Ez 1.28e-01,
    # Hr 1.6e-02, and a POINT Ep in the last half-open cell read 1.00e+00 — its
    # w0 + w1 = 1 collapsed onto one ring, exactly double MEEP's effective w0 —
    # and on a Cartesian PEC box a full-width Ey line read 3.2e-02; with the
    # drop, every one of those sits at its interior control's floor (6.8e-07 …
    # 1.0e-06). Half-integer-stencil components have big_owned_corner one short
    # of the wall and can never reach this branch, which is why they measured
    # exact all along. `boundaries` is read defensively for the stub grids in
    # test_sources.py, which model periodic/folded registration and no walls.
    boundaries_reader = getattr(grid, "boundaries", None)
    metal_high = [
        (not lattice[d]) and boundaries_reader is not None
        and boundaries_reader[d][1] == METALLIC
        for d in range(3)
    ]

    accumulated: dict[Tuple[int, int, int], complex] = {}
    center_vec = [(where_min[d] + where_max[d]) * 0.5 for d in range(3)]  # MEEP sources.cpp:486

    def deposit_pass(pass_iscS: List[int], pass_iecS: List[int],
                     pass_s0: List[float], pass_s1: List[float],
                     pass_e0: List[float], pass_e1: List[float],
                     shifti: Tuple[int, int, int]) -> None:
        """One (lattice shift, chunk) ladder — MEEP sources.cpp:265-273 src_vol_chunkloop.

        ``pass_iscS``/``pass_iecS`` are CHUNK-frame doubled bounds; ``shifti`` is
        the doubled lattice translation, so the source-frame location — where the
        amplitude function is sampled — is the chunk-frame loc plus
        ``shifti * dx/2`` (sources.cpp:270, ``loc += shift * (0.5 * inva)``).
        """
        pass_n = [(pass_iecS[d] - pass_iscS[d]) // 2 + 1 for d in range(3)]
        if any(n <= 0 for n in pass_n):
            return
        for i1 in range(pass_n[0]):
            for i2 in range(pass_n[1]):
                for i3 in range(pass_n[2]):
                    # MEEP vec.hpp:365-369 IVEC_LOOP_LOC: loc = (0.5*loop_is + loop_i) * inva,
                    # then sources.cpp:270 restores the source frame with the shift.
                    loc = [
                        (0.5 * (pass_iscS[0] + shifti[0]) + i1) * inva,
                        (0.5 * (pass_iscS[1] + shifti[1]) + i2) * inva,
                        (0.5 * (pass_iscS[2] + shifti[2]) + i3) * inva,
                    ]
                    # MEEP vec.hpp:372-383 IVEC_LOOP_WEIGHT with the adjusted weights
                    weight = (
                        _ivec_loop_weight_1d(i1, pass_n[0], pass_s0[0], pass_s1[0], pass_e0[0], pass_e1[0]) *
                        _ivec_loop_weight_1d(i2, pass_n[1], pass_s0[1], pass_s1[1], pass_e0[1], pass_e1[1]) *
                        _ivec_loop_weight_1d(i3, pass_n[2], pass_s0[2], pass_s1[2], pass_e0[2], pass_e1[2])
                    )
                    # MEEP sources.cpp:272-273 — amplitude sampled at the point's offset
                    # from the source center. The offset is passed in full: unlike the
                    # bundle, which computed z and dropped it, amp_func receives all
                    # three components so z-varying volume sources are expressible.
                    rel_loc = [loc[d] - center_vec[d] for d in range(3)]
                    if amp_func is not None:
                        A = amp_func(rel_loc[0], rel_loc[1], rel_loc[2])
                    else:
                        A = 1.0
                    amp_val = weight * amp * A

                    # MEEP vec.hpp:161-163 — idx = (ivec - little_corner) / 2, then the
                    # lattice fold: ladder cell j + n sits one lattice vector above cell j,
                    # so it deposits on cell j mod n carrying conj(bloch_phase ** shift).
                    cell = [0, 0, 0]
                    dropped = False
                    for d, ladder_index in enumerate(
                            ((pass_iscS[0] + 2 * i1 - io[0]) // 2,
                             (pass_iscS[1] + 2 * i2 - io[1]) // 2,
                             (pass_iscS[2] + 2 * i3 - io[2]) // 2)):
                        count = cell_counts[d]
                        if lattice[d]:
                            shift = ladder_index // count
                            cell[d] = ladder_index - shift * count
                            if shift and wrap_phases[d] is not None:
                                amp_val = amp_val * np.conj(wrap_phases[d] ** shift)
                        elif metal_high[d] and ladder_index >= count:
                            # The metallic wall plane — MEEP's zero_metal makes this
                            # deposit inert (the metal_high comment above carries the
                            # transcription and the measured cost of clamping it).
                            dropped = True
                            break
                        elif folded_live[d] and not 0 <= ladder_index < count:
                            # Unreachable, and stated rather than absorbed. A
                            # folded-live axis clips its ladder to MEEP's owned
                            # corners above (`little_owned` / `big_owned`), and
                            # `Grid.stored_cells` now covers every one of them at
                            # both count parities — MEEP's num + 1 allocation,
                            # whose top shift-0 sample sits AT `big_corner` and is
                            # owned (vec.cpp:445-462). Dropping the rung was this
                            # engine's answer while that slot went unstored at an
                            # odd count; clamping it onto the top stored row was
                            # the other wrong answer (it double-counts once the
                            # image pass below deposits the row's own
                            # lattice-shift contribution). Either would now be a
                            # silently lost current, so say so instead.
                            raise RuntimeError(
                                f"source deposition on folded axis {('x', 'y', 'z')[d]}: ladder "
                                f"rung {ladder_index} is outside the {count} stored cells "
                                f"after clipping to MEEP's owned corners "
                                f"[{little_owned[d]}, {big_owned[d]}] — a registration slip, "
                                f"not a caller error."
                            )
                        else:
                            # A folded METALLIC axis is clipped, not continued; the
                            # owned-corner bounds above already keep it in range, so
                            # this only guards against an off-by-one reaching the
                            # array.
                            cell[d] = max(0, min(count - 1, ladder_index))
                    if dropped:
                        continue

                    key = (cell[0], cell[1], cell[2])
                    # Two ladder points can fold onto one cell (a source spanning the whole
                    # cell contributes at both faces). MEEP's step_source subtracts them one
                    # after another, so they add; `D[ix, iy, iz] -= amps` would keep only the
                    # last, so they are summed here and each cell appears exactly once —
                    # the uniqueness invariant `_index_arrays` depends on.
                    accumulated[key] = accumulated.get(key, 0j) + amp_val

    # The direct pass — MEEP's ishift = 0.
    deposit_pass(iscS, iecS, s0c, s1c, e0c, e1c, (0, 0, 0))

    # MEEP loop_in_chunks.cpp:390-415 — the nonzero lattice shifts of a folded
    # PERIODIC axis. The range is MEEP's own: every ishift for which the shifted
    # chunk surroundings still intersect the request (:396-398, floor/ceil over
    # the physical volumes; the reduced grid's surroundings run from the origin
    # to `big_corner`, vec.cpp `surroundings()`). Each image is clipped to the
    # owned window and its boundary weights are read in the SOURCE frame
    # (:452-469), so the far end of a full-width request hands its edge weight to
    # the top of the stored window — the deposit MEEP's shift pass makes and the
    # clip-only path lost. The phase factor is `conj(eikna ** ishift)` = 1: a
    # folded axis refuses a nonzero Bloch phase outright
    # (`FdtdDriver._require_bloch_is_representable`).
    #
    # UNFOLDED periodic axes deliberately keep their shift list at {0}: their
    # images are already served inside `deposit_pass` by the modulo fold with
    # `conj(bloch_phase ** shift)`, the equivalent of MEEP's shift passes when
    # the chunk spans the whole lattice period (measured, test_from_meep's
    # `source_on_the_wrapped_pml_seam`).
    shift_lists: List[List[int]] = [[0], [0], [0]]
    for d in range(3):
        if not folded_live[d] or lengths[d] <= 0.0:
            continue
        gvs_min = io[d] * 0.5 * inva
        gvs_max = big_corner[d] * 0.5 * inva
        low = int(math.floor((where_min[d] - gvs_max) / lengths[d]))
        high = int(math.ceil((where_max[d] - gvs_min) / lengths[d]))
        shift_lists[d] = list(range(low, high + 1))
        if 0 not in shift_lists[d]:
            shift_lists[d].append(0)
    for combo in itertools.product(*shift_lists):
        if combo == (0, 0, 0):
            continue  # The direct pass above.
        shifti = tuple(combo[d] * 2 * full_counts[d] for d in range(3))
        pass_iscS: List[int] = []
        pass_iecS: List[int] = []
        pass_s0: List[float] = []
        pass_s1: List[float] = []
        pass_e0: List[float] = []
        pass_e1: List[float] = []
        empty = False
        for d in range(3):
            if combo[d] == 0:
                pass_iscS.append(iscS[d])
                pass_iecS.append(iecS[d])
                pass_s0.append(s0c[d])
                pass_s1.append(s1c[d])
                pass_e0.append(e0c[d])
                pass_e1.append(e1c[d])
                continue
            # MEEP loop_in_chunks.cpp:440,446 — the shifted request against the
            # owned corners: iscS = max(is - shifti, iscoS), iecS = min(ie - shifti, iecoS).
            low_bound = max(is_orig[d] - shifti[d], little_owned[d])
            high_bound = min(ie_orig[d] - shifti[d], big_owned[d])
            if high_bound < low_bound:
                empty = True
                break
            # MEEP loop_in_chunks.cpp:452-489 — weights compared in the source
            # frame (`iscS += shifti` precedes the comparisons), then the
            # narrow-range minima on the clipped extent.
            source_start = low_bound + shifti[d]
            source_end = high_bound + shifti[d]
            ws0 = ws1 = we0 = we1 = 1.0
            if source_start == is_orig[d]:
                ws0 = s0_orig[d]
                ws1 = s1_orig[d]
            elif source_start == is_orig[d] + 2:
                ws0 = s1_orig[d]
            if source_end == ie_orig[d]:
                we0 = e0_orig[d]
                we1 = e1_orig[d]
            elif source_end == ie_orig[d] - 2:
                we0 = e1_orig[d]
            if high_bound == low_bound:
                w = min(ws0, we0)
                ws0 = ws1 = we0 = we1 = w
            elif high_bound == low_bound + 2:
                w = min(ws0, we1)
                ws0 = we1 = w
                w = min(ws1, we0)
                ws1 = we0 = w
            elif high_bound == low_bound + 4:
                w = min(ws1, we1)
                ws1 = we1 = w
            pass_iscS.append(low_bound)
            pass_iecS.append(high_bound)
            pass_s0.append(ws0)
            pass_s1.append(ws1)
            pass_e0.append(we0)
            pass_e1.append(we1)
        if empty:
            continue
        deposit_pass(pass_iscS, pass_iecS, pass_s0, pass_s1, pass_e0, pass_e1, shifti)

    indices: List[Tuple[int, int, int]] = []
    amplitudes: List[complex] = []
    for key, value in accumulated.items():
        if abs(value) > 1e-20:
            indices.append(key)
            amplitudes.append(value)
    return indices, amplitudes


def _index_arrays(grid: 'Grid', indices: List[Tuple[int, int, int]],
                  amplitudes: List[complex]) -> Tuple[Any, Any, Any, Any]:
    """Pack (index, amplitude) pairs into backend arrays for vectorized injection.

    Injection uses a fancy-index subtraction `D[ix, iy, iz] -= values`, which is
    correct only because the points are unique: they come from a loop over
    distinct doubled ivecs, and the index map `(ivec - little_corner) / 2` is
    injective over the clipped range. Duplicate indices would silently drop all
    but one contribution (NumPy fancy-index assignment does not accumulate), so
    `DEBUG_VALIDATE_SOURCE_INDICES` re-checks the invariant.
    """
    ix, iy, iz = _point_index_arrays(grid, indices)
    return ix, iy, iz, grid.xp.asarray(amplitudes, dtype=grid.xp.complex64)


def _point_index_arrays(grid: 'Grid', indices: List[Tuple[int, int, int]]) -> Tuple[Any, Any, Any]:
    """The three backend index arrays of a deposition table, uniqueness re-checked.

    Split out of :func:`_index_arrays` because the point source packs its amplitudes
    in DOUBLE precision (see ``ContinuousSource._pack_points``) while every extended
    source packs them in complex64; the index half is the same for both.
    """
    xp = grid.xp
    ix = xp.asarray([p[0] for p in indices], dtype=xp.int64)
    iy = xp.asarray([p[1] for p in indices], dtype=xp.int64)
    iz = xp.asarray([p[2] for p in indices], dtype=xp.int64)
    if DEBUG_VALIDATE_SOURCE_INDICES:
        linear = [(x * grid.ny + y) * grid.nz + z for (x, y, z) in indices]
        assert len(set(linear)) == len(linear), "source points must have unique linear indices"
    return ix, iy, iz


def _step_source_values(array: Any, amps: Any, scale: complex,
                        component: str) -> Any:  # What step_source subtracts, in the array's dtype.
    """MEEP step.cpp:295-312 — the current to subtract, matched to the storage dtype.

    Complex storage takes the whole product `amp * current * dt`; real storage takes
    its real part, which is MEEP's own `f[c][0][i] -= real(A)` with the imaginary
    plane `f[c][1][i]` absent. The product is formed identically either way, so the
    real path is bit-for-bit the real part of the complex one and the complex path
    is byte-unchanged from before real mode existed.

    THE REAL PART IS TAKEN UNCONDITIONALLY, INCLUDING WHEN THE PER-POINT AMPLITUDE
    CARRIES PHASE. This engine used to refuse that input instead — the argument being
    that `Re(a*s) = Re(a)Re(s) - Im(a)Im(s)` re-reads a spatial phase as a per-point
    time shift, which the stored real field cannot record. The argument is true and
    the refusal was still wrong, for the reason this package exists: MEEP takes
    `real(A)` here with no condition attached (step.cpp:307), so a run that refuses is
    not stricter than MEEP, it is a run MEEP completes and this one does not. A
    Gaussian beam's amplitude is complex BY CONSTRUCTION (wavefront curvature and the
    Gouy phase), so the refusal declined essentially every real-field beam script —
    including MEEP's own ``gaussian-beam.py`` — and every oblique eigenmode launch,
    whose obliquity IS an in-plane phase ramp.

    MEASURED against CPU MEEP stepping the same ``mp.Simulation``, complex relative L2
    over the whole volume, with the refusal stubbed out:

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

    Sampled at t = 0.5 upward, so the early times where the "per-point time shift"
    reading is at its weakest — the run is nowhere near monochromatic steady state —
    are covered, and they are the TIGHTEST numbers in the table rather than the
    loosest. The last two columns are what settles it: on the same cell the real run
    and the complex run agree with CPU MEEP equally well, and the real array is
    ``Re(`` the complex array ``)`` to EXACTLY 0.0 — for this engine and for MEEP
    itself alike. So a real-field run with a phased amplitude is not a corrupted
    complex run; it is the real part of one, which is what real-field mode means.

    The control column is what a run that DID lose the phase looks like: launching
    |A| in place of A sits at 0.75…1.5. The suite carries the subtler spelling of the
    same mistake, ``drop_beam_phase`` — ``Re(a)*s`` in place of ``Re(a*s)``, which is
    the one-line "real storage cannot carry a complex amplitude" fix — at 6.6e-01 and
    7.0e-01 (``test_from_meep.py``, cases ``beam_real_fields`` /
    ``beam_real_fields_pulse`` / ``eigenmode_oblique_waveguide_real_fields``).
    """
    keep_real = _injects_real(array, component)
    product = amps * np.complex64(scale)
    return product.real if keep_real else product  # MEEP step.cpp:307 f[c][0][i] -= real(A).


def _injects_real(array: Any, component: str) -> bool:  # Real storage? Validated.
    """True when this array takes `real(A)`, False when it takes the whole complex product.

    Resolves the same three dtype cases as :func:`_step_source_values` — complex64,
    float32, anything else — for the scalar loop `ContinuousSource` uses, so both
    injection paths refuse an unsupported dtype with one message.
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

    UNIQUENESS IS THE ONE PLACE A SPARSE WRITE COULD DIFFER FROM NUMPY. Fancy
    assignment with a repeated index is last-wins on NumPy and unordered on a
    device ``index_copy_``. MEEP's source tables list each cell once, and every
    table this engine builds does too; this checks it the first time a table is
    seen rather than trusting it.
    """
    key = (id(ix), id(iy), id(iz))
    hit = _POINT_LINEAR.get(key)
    if hit is not None and hit[0] is ix and hit[1] is iy and hit[2] is iz:
        return hit[3]
    linear = host_writes.ravel_points(ix, iy, iz, array.shape)
    if host_writes.index_module(linear).unique(linear).size != linear.size:
        raise ValueError(
            "a source deposits into the same cell more than once; a sparse write "
            "cannot reproduce NumPy's last-wins fancy assignment there, so it is "
            "refused rather than left to the device's ordering")
    _POINT_LINEAR[key] = (ix, iy, iz, linear)
    return linear


def _sparse_subtract(array: Any, ix: Any, iy: Any, iz: Any, values: Any) -> None:
    """``array[ix, iy, iz] -= values``, moving only those cells.

    Bit-for-bit what the fancy in-place subtraction does: one float32 subtraction per
    word at the deposit cells, formed on whichever side owns the words. Under a held
    residency that is the device (``Residency.subtract_index``, a few cells, not the
    volume, and no host wait); on the array path it is the plain NumPy operation.
    """
    flags = getattr(array, "flags", None)
    if flags is None or not flags.c_contiguous:
        # A flat index is a C-order index only on a C-contiguous array; on any other
        # layout ``reshape(-1)`` copies and a scatter through it lands nowhere. The
        # engine allocates every primary C-contiguous, so this is the guard
        # ``deposit_repair._sparse_linear`` carries, not a route: the fancy in-place
        # subtraction, declared as the whole-volume write it is.
        host_writes.acquire(array)[ix, iy, iz] -= values
        return
    linear = _linear_points(array, ix, iy, iz)
    # THE DIFFERENCE IS FORMED WHERE THE WORDS LIVE. Gathering the cells to the host
    # and scattering ``current - values`` back was word-identical but was the one
    # blocking device->host read a held step made (``profile_held_step_res40_patched``,
    # 2026-09-24: 2.6-2.7 ms of a 4.6 ms step at 4.1M cells); the door subtracts on the
    # device under a hold and is NumPy's own in-place fancy subtraction otherwise.
    host_writes.subtract(array, linear, values)


def _inject_points(D: Any, ix: Any, iy: Any, iz: Any, amps: Any, scale: complex,
                   component: str) -> None:
    """MEEP step.cpp:295-298 step_source — `f[c][0][i] -= current * dt` at the source points.

    `D` is the primary array the component drives: the D array for an electric
    component, the B array for a magnetic one (MEEP subtracts with the same sign
    in both families).
    """
    _sparse_subtract(D, ix, iy, iz, _step_source_values(D, amps, scale, component))


# --- deposits that must also land in f_u ------------------------------------------
#
# MEEP does not run one PML curl recurrence everywhere: step_db selects a branch PER
# STRUCTURE CHUNK from sigsize (step_db.cpp:56-59), and the domain is split at the
# PML boundaries (measured 3x3 chunks for a 2-D cell with PML on all faces), so an
# edge-strip chunk that absorbs only on a component's dsig axis runs the UNSPLIT
# recurrence — f itself carries the damping, f_u is never touched
# (step_generic.cpp:86-240). This engine runs the both-active split recurrence over
# the whole grid. The two are algebraically identical while ``f == f_u`` — the
# invariant step_db.cpp:71-75 establishes by memcpy — and step_source is what breaks
# it: MEEP deposits into f alone, and in the unsplit chunk that deposit is damped by
# (kappa-sigma)/(kappa+sigma) every step, while the split recurrence damps f only on
# the dsigu axis and leaves the deposit standing. Mirroring the deposit into f_u
# wherever MEEP's chunking would run the unsplit branch restores the invariant and
# with it the recurrence MEEP actually runs there. Measured (12x12 um cell, res 20,
# CW Ez half-buried in a 2 um x-face layer, whole-volume complex relative L2 against
# CPU MEEP): 4.9581e-01 as shipped -> 5.6106e-07 mirrored, with the corner, y-PML
# and interior controls bit-unchanged (the design notes (absorber-placement-evidence)
# §1.5-1.6). The integrated path never runs through step_source in MEEP (step.cpp:300)
# and never mirrors here.
#
# CYLINDRICAL grids run the same mirror since the Dcyl sweep measured that the
# mechanism transfers unchanged. MEEP's Dcyl step_db selects its branch from the
# same per-chunk sigsize read (step_db.cpp:44-59 serves every dimensionality), and
# under this engine's axis mapping (x = R, y = P, z = Z) the curl tables' dsig/dsigu
# labels EQUAL the Dcyl cycle_direction start = 2 binding (vec.hpp:583-589): for
# Ez, dsig = R and dsigu = P — and P never carries sigma, so an Ez deposit under
# the r layer always lands in an unsplit chunk, Dcyl CORNERS included (unlike the
# Cartesian corner row of the truth table). Measured on a 3x6 Dcyl cell, res 10/20,
# CW, until ~20, whole-volume complex relative L2, m in {0, 1, 2}: Ez mid-r-layer
# 4.08e+00 -> 3.12e-06, corner 7.05e+01 -> 2.04e-05, Ep mid-z-layer 1.26e+00 ->
# 4.50e-06, a buried z line 5.65e+00 -> 1.77e-06, a partial radial span 6.36e-01 ->
# 2.36e-06; every interior/edge/decoupled-axis control bit-unchanged, and res 20
# reads 8.17e+00 -> 8.52e-06 on the mid-layer case.

# Ladder axes per primary array, from stepping's curl tables (the one home of the
# dsig/dsigu binding, MEEP vec.hpp cycle_direction with start = 1 resp. 2).
AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _pml_ladder_axes(target: str) -> Tuple[str, str]:  # (dsig, dsigu) for one primary array.
    from .stepping import B_CURL_TERMS, D_CURL_TERMS  # Deferred: stepping imports fields at module load.
    for term in (*D_CURL_TERMS, *B_CURL_TERMS):
        if term.target == target:
            return term.dsig, term.dsigu
    raise ValueError(f"No curl term drives {target!r}")


def _unsplit_deposit_flags(pml: Any, grid: 'Grid', component: str,
                           indices: List[Tuple[int, int, int]]) -> Optional[List[bool]]:
    """Which deposition points MEEP's chunking hands to the UNSPLIT dsig-only recurrence.

    Per point: sigma bites on the component's dsig axis AND the cell is outside every
    chunk that carries the dsigu-axis layer (``sigsize[dsigu] == 1`` there,
    step_db.cpp:56-59). Sigma is read at the Yee offset the component's curl
    coefficients use (integer for D, half-integer for B), from the same grading pass
    that filled them (:meth:`~.pml.PML.sigma_bites`); chunk membership is
    :meth:`~.pml.PML.in_pml_chunk`, which is NOT "sigma bites on dsigu": the layer's
    chunk owns one cell past its inner edge, whose own sigma is exactly zero and
    which MEEP still steps with the both-active branch. Measured on the 12x12 case:
    mirroring that boundary cell reads 4.59e-01 where leaving it split reads 7.3e-07.

    Returns None when nothing needs mirroring: no absorber, or every flag False.

    CYLINDRICAL grids take the same path with no special case: the curl tables'
    dsig/dsigu labels under the x = R, y = P, z = Z mapping equal MEEP's Dcyl
    ``cycle_direction`` ``start = 2`` binding (vec.hpp:583-589), and the phi axis
    (y) never carries a layer, so ``in_pml_chunk('y')`` is all-False and an Ez
    deposit under the r layer is mirrored everywhere sigma bites — corners
    included, which is what the Dcyl sweep measured (module comment above:
    7.05e+01 -> 2.04e-05 on the corner, controls bit-unchanged).
    """
    if pml is None or not getattr(pml, "is_active", False):
        return None
    target = _ARRAY_FOR_COMPONENT[component]
    dsig, dsigu = _pml_ladder_axes(target)
    half_integer = target[0] == 'B'
    bites_dsig = pml.sigma_bites(dsig, half_integer)
    chunked_dsigu = pml.in_pml_chunk(dsigu, half_integer)
    a_sig, a_sigu = AXIS_INDEX[dsig], AXIS_INDEX[dsigu]
    flags = [
        bool(bites_dsig[point[a_sig]]) and not bool(chunked_dsigu[point[a_sigu]])
        for point in indices
    ]
    return flags if any(flags) else None


def _bind_fu_mirror(source: Any, pml: Any) -> None:
    """(Re)compute one index-array source's packed f_u mirror selection, or clear it.

    Called by the driver whenever the absorber changes (``setup_pml``) or a source
    is added under an existing one (``add_source``), so the selection always
    describes the layer the run will actually step. An integrated source never
    mirrors: MEEP's step_source skips it (step.cpp:300) and its dipole reaches the
    field at the constitutive read instead.
    """
    source._fu_ix = source._fu_iy = source._fu_iz = source._fu_amps = None
    if source._n_source_points == 0 or getattr(source, "is_integrated", False):
        return
    flags = _unsplit_deposit_flags(pml, source.grid, source.component, source._indices)
    if flags is None:
        return
    xp = source.grid.xp
    selection = xp.asarray([i for i, keep in enumerate(flags) if keep], dtype=xp.int64)
    source._fu_ix = source._point_ix[selection]
    source._fu_iy = source._point_iy[selection]
    source._fu_iz = source._point_iz[selection]
    source._fu_amps = source._point_amps[selection]


def _inject_fu_mirror(source: Any, fields: 'Fields', scale: complex) -> None:
    """Mirror this step's deposit into f_u at the selected points, same value, same sign."""
    if source._fu_amps is None:
        return
    # RAW, not the barriered attribute: under a held residency the barrier acquired
    # the WHOLE ``fu_*`` volume every step for a deposit that touches a few cells
    # (1.0 copies out and 0.3 in per step on ``pml_3d`` at res 12, 2026-09-24).
    fu = host_writes.raw(fields, "fu_" + _ARRAY_FOR_COMPONENT[source.component])
    if fu is None:
        return  # PML storage not allocated: nothing steps the split recurrence yet.
    _inject_points(fu, source._fu_ix, source._fu_iy, source._fu_iz,
                   source._fu_amps, scale, source.component)


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
        return (self.dipole(time + dt) - self.dipole(time)) / dt


@dataclass
class ContinuousEnvelope(SourceEnvelope):
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

    def __post_init__(self):  # Resolve the carrier constants; a zero/non-finite frequency raises.
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:128-140 continuous_src_time::dipole.
        return _continuous_dipole(
            time, self._omega, self._amp_factor,
            self.start_time, self.end_time, self.width, self.slowness,
        )


@dataclass
class GaussianEnvelope(SourceEnvelope):
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

    def __post_init__(self):  # Resolve width, peak and cut; a zero frequency or non-positive fwidth raises.
        if self.fwidth <= 0.0:
            raise ValueError(f"Gaussian source fwidth must be positive, got {self.fwidth}")
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104
        self._width = 1.0 / self.fwidth
        self._peak_time = self.start_time + self.cutoff * self._width
        self._cutoff_time = _gaussian_cutoff_time(self.cutoff, self._width)

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:98-107 gaussian_src_time::dipole.
        return _gaussian_dipole(time, self._omega, self._amp_factor,
                                self._peak_time, self._width, self._cutoff_time)

    @property
    def peak_time(self) -> float:  # Time of the envelope peak.
        return self._peak_time

    @property
    def width(self) -> float:  # Temporal width (1/fwidth), the envelope standard deviation.
        return self._width


@dataclass
class CustomEnvelope(SourceEnvelope):
    """
    Caller-supplied waveform — MEEP's `custom_src_time` / Python `CustomSource`.

    `func(t) -> complex` is evaluated in MEEP natural units and gated to
    `[start_time, end_time]`. Which quantity it represents follows MEEP
    (meep.hpp:1066-1071): with `is_integrated=False` — the default — the
    function *is* the current, so it is injected directly rather than
    differenced; with `is_integrated=True` it is the dipole moment and the
    current becomes its forward difference.

    A function that returns zero injects exactly zero: the scale reaches the
    field as an exact 0 and the write is skipped entirely, so a silent envelope
    cannot leave round-off behind and score as a working source.
    """

    func: Callable[[float], complex]
    start_time: float = _MEEP_CUSTOM_START
    end_time: float = _MEEP_CUSTOM_END
    is_integrated: bool = False

    def __post_init__(self):  # A non-callable waveform is a caller bug, not a zero source.
        if not callable(self.func):
            raise ValueError(
                f"CustomEnvelope needs a callable func(t) -> complex, got {type(self.func).__name__}"
            )

    def dipole(self, time: float) -> complex:  # MEEP meep.hpp:1072-1078 custom_src_time::dipole.
        return _custom_dipole(time, self.func, self.start_time, self.end_time)

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp:1066-1071 custom_src_time::current.
        if self.is_integrated:
            return super().current(time, dt)
        return self.dipole(time)


@dataclass
class ContinuousSource:
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
    # touched cell, each amplitude carrying its interpolation weight, the delta-function
    # scaling `a` per zero-size axis, and any Bloch factor the lattice fold picked up.
    _omega: float = field(default=0.0, repr=False, init=False)
    _amp_factor: complex = field(default=1j, repr=False, init=False)
    _point_indices: list = field(default=None, repr=False, init=False)
    _point_weights: list = field(default=None, repr=False, init=False)
    _fu_flags: Optional[List[bool]] = field(default=None, repr=False, init=False)
    # The same table packed for the fancy-index deposit, built once (`_pack_points`).
    _point_ix: Any = field(default=None, repr=False, init=False)
    _point_iy: Any = field(default=None, repr=False, init=False)
    _point_iz: Any = field(default=None, repr=False, init=False)
    _point_dt_weights: Any = field(default=None, repr=False, init=False)
    _fu_ix: Any = field(default=None, repr=False, init=False)
    _fu_iy: Any = field(default=None, repr=False, init=False)
    _fu_iz: Any = field(default=None, repr=False, init=False)
    _fu_dt_weights: Any = field(default=None, repr=False, init=False)

    def __post_init__(self):  # Resolve the waveform constants and the deposition stencil.
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104
        self._setup_distribution()

    def bind_pml(self, pml: Any) -> None:  # Which stencil cells mirror into f_u (module comment above).
        self._fu_flags = (
            _unsplit_deposit_flags(pml, self.grid, self.component, self._point_indices)
            if self._point_indices else None
        )
        self._pack_points()

    def _pack_points(self) -> None:  # Pack the deposition table for the fancy-index deposit.
        """Build the backend index/amplitude arrays :meth:`inject` subtracts through.

        This replaces a per-cell Python loop that indexed the primary array with three
        SCALARS per point. On NumPy that loop was genuinely cheap — measured 0.004 s of
        a 2.474 s profiled run, 0.2%, and the comment it carried ("a plain loop is
        cheaper than packing index arrays") was true there. On a device it is not: each
        scalar read-modify-write is its own kernel launch, so a source touching eight
        cells issued eight to sixteen launches per timestep, plus a host-side
        ``np.complex64`` per point. Every other source class already deposits through
        :func:`_inject_points`; this was the odd one out.

        THE WEIGHTS ARE STORED PRE-MULTIPLIED BY dt, IN DOUBLE PRECISION, and that is
        what makes the change bit-identical rather than merely close. The loop formed
        ``np.complex64(dt * weight * J)`` — the whole product in Python's complex
        double, rounded to complex64 ONCE at the end. Packing ``dt * weight`` here (the
        same expression, evaluated once instead of every step) and multiplying by ``J``
        in complex128 before the single cast reproduces that rounding exactly; packing
        the weights as complex64 the way the extended sources do would have rounded
        twice and moved the last bit.
        """
        indices = self._point_indices
        if not indices:
            self._point_ix = self._point_iy = self._point_iz = None
            self._point_dt_weights = None
            self._fu_ix = self._fu_iy = self._fu_iz = self._fu_dt_weights = None
            return
        xp = self.grid.xp
        dt = self.grid.dt
        scaled = [dt * weight for weight in self._point_weights]
        self._point_ix, self._point_iy, self._point_iz = _point_index_arrays(self.grid, indices)
        self._point_dt_weights = xp.asarray(scaled, dtype=xp.complex128)
        keep = [index for index, flag in enumerate(self._fu_flags or ()) if flag]
        if not keep:
            self._fu_ix = self._fu_iy = self._fu_iz = self._fu_dt_weights = None
            return
        self._fu_ix, self._fu_iy, self._fu_iz = _point_index_arrays(
            self.grid, [indices[index] for index in keep])
        self._fu_dt_weights = xp.asarray([scaled[index] for index in keep],
                                         dtype=xp.complex128)

    def _setup_distribution(self):  # Point sources only; extended geometry has its own class.
        if all(s == 0.0 for s in self.size):
            self._setup_point_source()
        else:
            raise NotImplementedError("Use ExtendedSource for non-point sources")

    def _setup_point_source(self):  # MEEP loop_in_chunks deposition for a zero-size volume.
        """Place the point through MEEP's own machinery, lattice fold included.

        `_build_source_points` is called with a unit amplitude because this class
        keeps the user amplitude in `dipole()`; everything else it returns — the
        Simpson weights (which for a zero-extent direction are the two linear
        interpolation weights, or a single 1.0 when the request lands exactly on a
        sample), the `a` per zero-size axis of MEEP sources.cpp:481-484, and
        `conj(bloch_phase ** shift)` on any cell reached by wrapping — belongs in the
        per-point amplitude.

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
        self._point_indices, self._point_weights = _build_source_points(
            self.grid, self.component, self.center, self.size, 1.0 + 0j, None,
        )
        self._pack_points()  # Re-packed by bind_pml once the f_u mirror set is known.

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:128-140 continuous_src_time::dipole.
        return self.amplitude * _continuous_dipole(
            time, self._omega, self._amp_factor,
            self.start_time, self.end_time, self.width, self.slowness,
        )

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current — forward difference.
        return (self.dipole(time + dt) - self.dipole(time)) / dt

    @property
    def field_type(self) -> str:  # Electric source: injected between step_D and update_E.
        return FIELD_TYPE_D

    @property
    def is_integrated(self) -> bool:  # This class only implements MEEP's non-integrated path.
        return False

    def inject(self, fields: 'Fields', time: float):  # MEEP step.cpp step_source — subtract the current from D.
        if self._point_dt_weights is None:
            return
        dt = self.grid.dt
        J = self.current(time, dt)
        D = _d_array_for(fields, self.component)
        # A zero-size volume touches at most 8 cells, each exactly once (two ladder
        # rungs that fold onto the same cell were summed at setup), so one fancy-index
        # subtraction covers the whole stencil. The delta-function scaling is already
        # inside each weight — MEEP folds it into the amplitude too. Real storage takes
        # the real part of the same complex product, which is MEEP's
        # `f[c][0][i] -= real(A)`.
        keep_real = _injects_real(D, self.component)
        values = self._deposit_values(self._point_dt_weights, J, keep_real)
        _sparse_subtract(D, self._point_ix, self._point_iy, self._point_iz, values)
        if self._fu_dt_weights is None:
            return
        # RAW, then the sparse door -- the same route the primary deposit above takes.
        # Fetching ``fu_*`` through the read barrier under a held residency acquired the
        # WHOLE volume every step (a source point inside the PML at res 8/12/48 of
        # ``pml_3d``: 3.35 ms/step of the 14.5 at 7.1M cells, measured 2026-09-24,
        # ``results/round_2026-09-24_night_drivers/profile_held_step.log``), and the
        # in-place fancy subtraction was the one host write of a held array that
        # bypassed the doors. ``_sparse_subtract`` forms the same difference in the
        # same dtype and moves only the deposit cells.
        mirror = host_writes.raw(fields, "fu_" + _ARRAY_FOR_COMPONENT[self.component])
        if mirror is None:
            return  # PML storage not allocated: nothing steps the split recurrence yet.
        _sparse_subtract(mirror, self._fu_ix, self._fu_iy, self._fu_iz,
                         self._deposit_values(self._fu_dt_weights, J, keep_real))

    def _deposit_values(self, dt_weights: Any, J: complex, keep_real: bool) -> Any:
        """``complex64(dt * weight * J)`` per point — see :meth:`_pack_points` for the rounding."""
        xp = self.grid.xp
        values = (dt_weights * complex(J)).astype(xp.complex64)
        return values.real if keep_real else values

    def __repr__(self) -> str:
        return (f"ContinuousSource(freq={self.frequency:.4f}, "
                f"component={self.component}, center={self.center})")


@dataclass
class ExtendedSource:
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
    _fu_ix: Any = field(default=None, repr=False, init=False)
    _fu_iy: Any = field(default=None, repr=False, init=False)
    _fu_iz: Any = field(default=None, repr=False, init=False)
    _fu_amps: Any = field(default=None, repr=False, init=False)

    def __post_init__(self):  # MEEP sources.cpp:456-489 add_volume_source.
        _validate_component(self.component)  # Electric components only; magnetic ones go through VolumeSource.
        self._omega = 2 * np.pi * self.frequency
        self._amp_factor = _amp_factor(self.frequency)  # MEEP sources.cpp:104
        self._validate_symmetry()
        self._setup_source_points()

    def bind_pml(self, pml: Any) -> None:  # Which stencil cells mirror into f_u (module comment above).
        _bind_fu_mirror(self, pml)

    def _validate_symmetry(self):  # A source must sit in the stored half, and carry the fold's parity.
        _validate_symmetry(self.grid, self.center)
        _validate_symmetry_parity(self.grid, self.component, self.center, self.size, self.amp_func)

    def _setup_source_points(self):  # Build the per-point index/amplitude tables.
        self._indices, self._amplitudes = _build_source_points(
            self.grid, self.component, self.center, self.size, self.amplitude, self.amp_func
        )
        self._setup_index_arrays()

    def _setup_index_arrays(self):  # Pack the tables into backend arrays for vectorized injection.
        self._n_source_points = len(self._indices)
        if self._n_source_points == 0:
            self._point_ix = self._point_iy = self._point_iz = self._point_amps = None
            return
        (self._point_ix, self._point_iy,
         self._point_iz, self._point_amps) = _index_arrays(self.grid, self._indices, self._amplitudes)

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:128-140 — the user amplitude is folded into the point amps.
        return _continuous_dipole(
            time, self._omega, self._amp_factor,
            self.start_time, self.end_time, self.width, self.slowness,
        )

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current — forward difference.
        return (self.dipole(time + dt) - self.dipole(time)) / dt

    @property
    def field_type(self) -> str:  # Electric source: injected between step_D and update_E.
        return FIELD_TYPE_D

    @property
    def is_integrated(self) -> bool:  # This class only implements MEEP's non-integrated path.
        return False

    def inject(self, fields: 'Fields', time: float):  # MEEP step.cpp:295-298 — D -= dt * current * stored amplitude.
        D = _d_array_for(fields, self.component)
        if self._n_source_points == 0:
            return
        dt = self.grid.dt
        curr = self.current(time, dt)
        _inject_points(D, self._point_ix, self._point_iy, self._point_iz,
                       self._point_amps, dt * curr, self.component)
        _inject_fu_mirror(self, fields, dt * curr)

    def __repr__(self) -> str:
        return (f"ExtendedSource(freq={self.frequency:.4f}, "
                f"component={self.component}, center={self.center}, size={self.size})")


@dataclass
class GaussianPulsedSource:
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
    _fu_ix: Any = field(default=None, repr=False, init=False)
    _fu_iy: Any = field(default=None, repr=False, init=False)
    _fu_iz: Any = field(default=None, repr=False, init=False)
    _fu_amps: Any = field(default=None, repr=False, init=False)

    def bind_pml(self, pml: Any) -> None:  # Which stencil cells mirror into f_u (module comment above).
        _bind_fu_mirror(self, pml)

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
        self._indices, self._amplitudes = _build_source_points(
            self.grid, self.component, self.center, self.size, self.amplitude, self.amp_func
        )
        self._setup_index_arrays()

    def _setup_index_arrays(self):  # Pack the tables into backend arrays for vectorized injection.
        self._n_source_points = len(self._indices)
        if self._n_source_points == 0:
            self._point_ix = self._point_iy = self._point_iz = self._point_amps = None
            return
        (self._point_ix, self._point_iy,
         self._point_iz, self._point_amps) = _index_arrays(self.grid, self._indices, self._amplitudes)

    def dipole(self, time: float) -> complex:  # MEEP sources.cpp:109-120 gaussian_src_time::dipole.
        return _gaussian_dipole(time, self._omega, self._amp_factor,
                                self._peak_time, self._width, self._cutoff_time)

    def current(self, time: float, dt: float) -> complex:  # MEEP meep.hpp src_time::current — forward difference.
        return (self.dipole(time + dt) - self.dipole(time)) / dt

    @property
    def field_type(self) -> str:  # Electric source: injected between step_D and update_E.
        return FIELD_TYPE_D

    @property
    def is_integrated(self) -> bool:  # This class only implements MEEP's non-integrated path.
        return False

    def inject(self, fields: 'Fields', time: float):  # MEEP step.cpp:295-298 — D -= dt * current * stored amplitude.
        D = _d_array_for(fields, self.component)
        if self._n_source_points == 0:
            return
        dt = self.grid.dt
        curr = self.current(time, dt)
        if abs(curr) < 1e-30:
            return  # Outside the pulse: nothing to add, and the fancy-index write is not free.
        _inject_points(D, self._point_ix, self._point_iy, self._point_iz,
                       self._point_amps, dt * curr, self.component)
        _inject_fu_mirror(self, fields, dt * curr)

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

    # Internal state — the analog of MEEP's src_vol index_array / amps_array,
    # plus the running dipole offset the integrated path has already applied.
    _field_type: str = field(default=FIELD_TYPE_D, repr=False, init=False)
    _indices: List[Tuple[int, int, int]] = field(default=None, repr=False, init=False)
    _amplitudes: List[complex] = field(default=None, repr=False, init=False)
    _point_ix: Any = field(default=None, repr=False, init=False)
    _point_iy: Any = field(default=None, repr=False, init=False)
    _point_iz: Any = field(default=None, repr=False, init=False)
    _point_amps: Any = field(default=None, repr=False, init=False)
    _n_source_points: int = field(default=0, repr=False, init=False)
    _applied_dipole: complex = field(default=0j, repr=False, init=False)
    _fu_ix: Any = field(default=None, repr=False, init=False)
    _fu_iy: Any = field(default=None, repr=False, init=False)
    _fu_iz: Any = field(default=None, repr=False, init=False)
    _fu_amps: Any = field(default=None, repr=False, init=False)

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
        # An is_integrated source runs UNDER MIRROR SYMMETRY too. The suspicion that
        # stood here — the mirror-plane repair copies cells while they hold f - dipole
        # — is not a defect: the repair writes ghost D = parity * (f - dipole) at the
        # source's own image cells, which is exactly MEEP's f_minus_p there (the ghost
        # E the constitutive update then produces equals parity * E of the image row,
        # the same value MEEP's E-side step_boundaries copies), and `withdraw` restores
        # the owned source points before the next curl while the ghost rows are
        # refreshed by the next repair pass before anything reads them. MEEP itself
        # folds the combination exactly — measured folded vs unfolded MEEP, an
        # integrated full-height Ez sheet through a Mirror(Y) with PML: 1.7e-07 (PML on
        # x) and 1.8e-07 (PML everywhere) — and this engine now reproduces the folded
        # run at its ordinary fold floor (test_sources.py pins the number). The blanket
        # refusal cost finite_grating.py, absorbed_power_density.py and
        # mie_scattering.py their lifts.
        self._setup_source_points()

    def _setup_source_points(self):  # Build the per-point index/amplitude tables.
        self._indices, self._amplitudes = _build_source_points(
            self.grid, self.component, self.center, self.size, self.amplitude, self.amp_func
        )
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

    def reset(self):  # Forget the applied dipole offset, for a driver that restarts the clock.
        self._applied_dipole = 0j

    def bind_pml(self, pml: Any) -> None:  # Which stencil cells mirror into f_u (module comment above).
        _bind_fu_mirror(self, pml)

    def withdraw(self, fields: 'Fields') -> None:
        """Return the standing integrated-source dipole to D/B before the curl ladder runs.

        MEEP's integrated source never sits in the primary array while ``step_db``
        runs: the dipole is subtracted from ``f_minus_p`` inside ``update_eh``
        (update_eh.cpp:126-138), downstream of the whole ladder, fresh and undamped
        every step. This engine stores the offset IN the array between the injection
        slot and the constitutive read, so any per-cell rescale the ladder applies —
        the split PML recurrence's ``(kappa-sigma)/(kappa+sigma)`` on the component's
        dsigu axis, a D conductivity's ``condfac*condinv`` — would decay an offset
        MEEP never exposes to it. Measured before this method existed: a CW Ez
        ``is_integrated`` source in a y-face layer read 4.59e-01 against CPU MEEP
        where its x-face twin (dsigu sigma zero) read 6.6e-07, corners 1.28e-01
        (the design notes (absorber-placement-evidence) §1.6, Defect A).

        The driver calls this immediately before ``step_D`` (or ``step_B`` for a
        magnetic component): the offset is added back, the array re-enters the
        ladder holding exactly MEEP's f everywhere, and :meth:`inject` afterwards
        subtracts the fresh whole ``dipole`` — so between injection and the
        constitutive read the array holds ``f - dipole``, which is what MEEP's
        ``update_eh`` computes, and outside that window it holds MEEP's f exactly.
        """
        if not self.envelope.is_integrated or self._n_source_points == 0:
            return
        if self._applied_dipole == 0:
            return
        array = _array_for(fields, self.component)
        _inject_points(array, self._point_ix, self._point_iy, self._point_iz,
                       self._point_amps, -self._applied_dipole, self.component)
        self._applied_dipole = 0j

    def inject(self, fields: 'Fields', time: float):  # Subtract this step's contribution from D or B.
        """Apply one timestep of this source.

        Non-integrated (MEEP step.cpp:295-312): ``f -= dt * current(time)``, with
        ``time`` the slot time of the component's field family.

        Integrated (MEEP update_eh.cpp:126-140): MEEP forms ``f_minus_p = f -
        dipole`` and feeds that to the constitutive update, half a step after the
        injection slot; the dipole time lands on ``calc_sources(time() + dt)`` for
        the electric family and ``time() + dt/2`` for the magnetic one, which both
        equal this slot's ``time + 0.5*dt``. The driver's ``withdraw`` call before
        the curl ladder has already returned the previous offset, so
        ``_applied_dipole`` is zero here and the whole fresh dipole is subtracted;
        the array then holds exactly ``f - dipole(t)`` when the constitutive update
        reads it.
        """
        array = _array_for(fields, self.component)
        if self._n_source_points == 0:
            return
        dt = self.grid.dt
        if self.envelope.is_integrated:
            offset = self.envelope.dipole(time + 0.5 * dt)
            scale = offset - self._applied_dipole
            self._applied_dipole = offset
        else:
            scale = dt * self.envelope.current(time, dt)
        if scale == 0:
            return  # A silent envelope writes nothing at all, so the fields stay exactly zero.
        _inject_points(array, self._point_ix, self._point_iy, self._point_iz,
                       self._point_amps, scale, self.component)
        if not self.envelope.is_integrated:
            _inject_fu_mirror(self, fields, scale)

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
