"""
Frequency-domain monitors for the MEEP-compatible FDTD engine: `DFTMonitor` accumulates the
discrete Fourier transform of chosen field components over a rectangular grid
region, and `FluxMonitor` accumulates all six components on a plane and reduces
them to a Poynting (power) flux. Both monitor a *set* of frequencies — a
transmission or reflection spectrum is one monitor, one pass over the run.

Both monitors are driven from the stepping loop by
`update(fields, time, current_step)` once per timestep, after the step count has
been incremented, and accumulate

    dft += exp(+i*omega*t) * (dt * decimation_factor / sqrt(2*pi))
           * field_at_cell_center

with E sampled at `time` and H/B sampled at `time - 0.5*dt` — the leapfrog
stagger MEEP's `fields_chunk::update_dfts` uses. With decimation factor ``d``,
only steps whose global index satisfies ``step % d == 0`` accumulate, exactly
as MEEP's `dft.cpp`; the factor of ``d`` in the quadrature weight preserves the
integral normalization. Field values are interpolated from their Yee positions
to cell centers first, so every component of a monitor shares one index region
and one coordinate set. Every monitored frequency is accumulated in the same
pass, by broadcasting the per-frequency phase factor over the field slice;
nothing loops over frequency per timestep.

A third monitor, `Near2FarMonitor`, is built ON those two rather than beside them:
it registers one `DFTMonitor` per near-field patch, converts what they accumulate
into the equivalent surface currents `J = n x H` and `M = -n x E` of the
surface-equivalence theorem, and radiates them through the free-space dyadic
Green's function (MEEP's `dft_near2far`). That is how a radiation pattern, a
scattering cross-section or an antenna response leaves a run whose cell is only a
wavelength or two across. It reuses the same fractional-cell surface weighting a
flux plane uses, so the patch integrates over the area it was asked for wherever it
falls relative to the grid, and it refuses the run shapes whose near field is not
the field on one surface — a Bloch phase (the far field is a phased sum over
lattice images, MEEP's `Nperiods`), a mirror-folded grid (the stored quadrant holds
part of every patch), a surface inside the PML, and a surface that does not enclose
the sources.

Both field monitors register through one function, `_register_volume`, and its cells and
their integration weights both come from `_axis_ladder`, MEEP's `loop_in_chunks`
bracket of the requested bounds. A volume therefore lands on the same cells
whichever monitor is asked for it, and a region is never registered on one set of
cells and weighted for another. A flux plane is placed at the coordinate it was
asked for and not at the nearest row of cell centers: a request between two rows
is bracketed by both and linearly interpolated, which is MEEP's own zero-thickness
rule. A region reaching past a *periodic* face is continued into the neighbouring
lattice image with its Bloch factor rather than clipped there, which is MEEP's
lattice-shift loop; on an absorbing face it is clipped, and refused outright when
nothing of it is left inside the cell.

Accumulators therefore carry a leading frequency axis: `dft[component]` has
shape `(nf, nx, ny, nz)`. `get_dft(component, freq_index)` returns the
`(nx, ny, nz)` slice for one frequency — unchanged in meaning from the
single-frequency monitors this replaced — and `get_dft_spectrum(component)`
returns the whole stack. `FluxMonitor.get_flux(freq_index)` and
`get_flux_spectrum()` are the flux equivalents. Because `nf` multiplies the
footprint of every accumulator, allocation is checked against
`MAX_ACCUMULATOR_BYTES` and refused loudly rather than attempted.

Inputs: a `Grid` (supplies the array module `xp`, spacing, timestep, symmetry
flags and coordinate axes), a `Fields` object exposing `get_component(name)`,
one or more monitor frequencies in MEEP natural units, and either an index
region or a physical (center, size) volume. Outputs: complex64 DFT arrays per
component, `get_dft_full` quadrant reconstruction when mirror symmetry is
active, an intensity map, and per-frequency flux.

Normalization is the caller's job: after a run of length `runtime`, the
steady-state amplitude of a monochromatic field is
`A = DFT * sqrt(2*pi) / runtime`.

Units are MEEP natural units throughout (c = 1, frequency = 1/wavelength);
host applications perform any external unit conversion.
Real fields are float32 and complex fields complex64; DFT accumulators are
always complex64 and the flux surface weights float32.

A complex accumulator over REAL fields is MEEP's own arrangement, not an
inconsistency: `update_dft` branches on `numcmp = fc->f[c][1] ? 2 : 1` and, when
the run is real, accumulates `{fr*phase.real(), fr*phase.imag()}` into the same
complex array (dft.cpp:294-305). Two things follow, and both matter to a caller
reading a spectrum. The field arrays must NOT be promoted to satisfy the
accumulator — the whole memory saving of a real run is in those arrays, and
`phase * field` produces complex64 from float32 without touching the operand.
And the accumulated value is HALVED relative to the complex run, because
`Re(A e^-iwt)` carries half its energy in the +w line and half in -w: measured
against CPU MEEP's own two modes, `dft_complex / dft_real = 2.00` plus a
counter-rotating residual that decays with run length. Real and complex spectra
are each self-consistent and each match CPU MEEP in the same mode; they are not
interchangeable, and a normalisation run must use the mode its measurement did.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .backends import to_numpy
from .pml import absorbing_cells

if TYPE_CHECKING:  # Type-only imports; no runtime dependency between engine modules.
    from .fields import Fields
    from .grid import Grid

# Yee offsets in units of dx/2 (MEEP vec.hpp iyee_shift). 1 means the component
# already sits at a half-cell offset in that direction (cell centre for the
# purposes of the 0.5*(f[i]+f[i+1]) average), 0 means it needs interpolation.
IYEE_SHIFTS: dict[str, tuple[int, int, int]] = {
    "Ex": (1, 0, 0), "Ey": (0, 1, 0), "Ez": (0, 0, 1),
    "Dx": (1, 0, 0), "Dy": (0, 1, 0), "Dz": (0, 0, 1),
    "Hx": (0, 1, 1), "Hy": (1, 0, 1), "Hz": (1, 1, 0),
    "Bx": (0, 1, 1), "By": (1, 0, 1), "Bz": (1, 1, 0),
}
ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")
MAGNETIC_COMPONENTS = ("Hx", "Hy", "Hz")
AXIS_NAMES = ("x", "y", "z")  # Axis order of every (center, size) triple and of the field arrays.

COMPLEX64_BYTES = 8  # Every DFT accumulator is complex64; nf multiplies this footprint.
MAX_ACCUMULATOR_BYTES = 4 * 1024**3  # 4 GiB of accumulators per monitor, summed over components.


def _slab(axis: int, index) -> tuple:  # Index tuple selecting along one axis only.
    """`arr[_slab(axis, i)]` is `arr[..., i, ...]` with `i` on `axis`.

    The same helper `fields._slab` is, so the axis-by-axis unfolding here reads
    the way the field-array unfolding there does.
    """
    return (slice(None),) * axis + (index,)


def meep_frequency_span(fcen: float, df: float, nfreq: int) -> tuple[float, ...]:
    """Expand MEEP's (fcen, df, nfreq) monitor spec into explicit frequencies.

    MEEP source: `meep.simulation.fix_dft_args` — `nfreq` points spanning
    `fcen - df/2` to `fcen + df/2`, except that a single-point spectrum is
    `[fcen]` and not the lower edge `fcen - df/2` that `linspace(..., 1)` would
    give. That special case is the whole reason this lives here: a one-frequency
    monitor written as (fcen, df, 1) must land on the centre frequency, and a
    half-bandwidth offset there is invisible in a plot and wrong in every number.

    Args:
        fcen: Centre frequency in MEEP natural units.
        df: Full bandwidth spanned by the monitor.
        nfreq: Number of frequency points, at least 1.

    Returns:
        The monitored frequencies, ascending.
    """
    count = int(nfreq)
    if count != nfreq or count < 1:
        raise ValueError(f"nfreq must be a positive whole number of frequency points, got {nfreq!r}.")
    width = float(df)
    if not math.isfinite(width) or width < 0.0:
        raise ValueError(f"df must be a finite, non-negative bandwidth, got {df!r}.")
    if count == 1:
        return normalize_frequencies(fcen)
    return normalize_frequencies(np.linspace(float(fcen) - 0.5 * width, float(fcen) + 0.5 * width, count))


def normalize_frequencies(
    frequency: float | Sequence[float] | None = None,
    frequencies: float | Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Resolve a monitor's frequency argument(s) into an ordered tuple.

    Either spelling is accepted — `frequency` for the single-frequency call
    shape that predates spectra, `frequencies` for a sequence — and either may
    carry a scalar or a sequence. Passing both is allowed only when they describe
    the same set — a caller that spells one monitor's frequencies twice is
    redundant, not wrong. Two *different* sets raise, because choosing between
    them would accumulate a spectrum nobody asked for.

    Args:
        frequency: Scalar or sequence, or None when `frequencies` carries the set.
        frequencies: Scalar or sequence, or None when `frequency` carries the set.

    Returns:
        The monitored frequencies as a tuple of floats, in the order given.
    """
    if frequency is not None and frequencies is not None:
        from_frequency = normalize_frequencies(frequency=frequency)
        from_frequencies = normalize_frequencies(frequencies=frequencies)
        if from_frequency != from_frequencies:
            raise ValueError(
                f"Conflicting monitor frequencies: frequency={frequency!r} resolves to "
                f"{from_frequency} but frequencies={frequencies!r} resolves to {from_frequencies}. "
                f"Pass one or the other."
            )
        return from_frequency
    given = frequency if frequency is not None else frequencies
    if given is None:
        raise ValueError("A monitor needs at least one frequency; pass frequencies=[...] or frequency=f.")
    if isinstance(given, Sequence) and not isinstance(given, (str, bytes)):
        values = list(given)
    elif isinstance(given, np.ndarray):
        values = [given.item()] if given.ndim == 0 else list(given.ravel())
    else:
        values = [given]  # A scalar, or something that is about to fail float() loudly.
    if not values:
        raise ValueError("A monitor needs at least one frequency; the given sequence is empty.")
    resolved: list[float] = []
    seen: set[float] = set()
    duplicates: list[float] = []
    for entry in values:
        # Text is rejected before float() sees it: "1e15" parses cleanly and would
        # install a plausible-looking frequency that nobody meant to ask for.
        if isinstance(entry, (str, bytes)):
            raise ValueError(f"Monitor frequencies must be numbers, not text; got {entry!r} in {given!r}.")
        try:
            value = float(entry)
        except (TypeError, ValueError):
            raise ValueError(f"Monitor frequencies must be numbers, got {entry!r} in {given!r}.") from None
        # Finite and NONZERO, either sign. MEEP stores a monitor frequency raw —
        # ``omega[i] = 2*pi*freq[i]`` (dft.cpp:219-221) — accumulates
        # ``polar(1.0, omega*t)`` with that sign (dft.cpp:268-270), and takes
        # ``abs(freq)`` for its decimation bound (dft.cpp:201-206), so a negative
        # frequency is an anticipated input: the conjugate accumulator, which the
        # plus/minus-omega near2far pair in dipole_in_vacuum_cyl_off_axis.py
        # depends on.
        #
        # ZERO IS ADMITTED TOO, and it needs no branch anywhere. ``update_dft`` builds
        # ``dft_phase[i] = polar(1.0, omega[i]*time) * scale`` with NO test on omega
        # (dft.cpp:266-269) and accumulates the same multiply-add; at omega = 0 the
        # phase factor is exactly 1 and the accumulator is a running time integral of
        # the field. The one DC-sensitive line in the whole file is the AUTOMATIC
        # decimation guard (dft.cpp:207-210): it requires ``freq_max > 0`` and falls
        # through to a factor of 1 otherwise. This engine reads the factor MEEP
        # RESOLVED off the chunk (``_resolved_decimation``) rather than re-deriving
        # that rule, so a DC monitor is already right here for free.
        if not math.isfinite(value):
            raise ValueError(
                f"Monitor frequencies must be finite (zero accumulates a running time "
                f"integral, negative accumulates the conjugate phase), got "
                f"{entry!r} in {given!r}."
            )
        if value in seen:
            duplicates.append(value)
        seen.add(value)
        resolved.append(value)
    if duplicates:
        raise ValueError(
            f"Monitor frequencies must be distinct; {sorted(set(duplicates))} appear more than once "
            f"in {resolved}."
        )
    return tuple(resolved)


def _frequency_index(frequencies: tuple[float, ...], freq_index: Any) -> int:
    """Validate a frequency index against a monitor's frequency set.

    Negative indices are rejected rather than wrapped: `get_flux(-1)` quietly
    returning the top of the band instead of failing is the kind of plausible
    wrong number this engine's error policy exists to prevent.
    """
    try:
        index = int(freq_index)
    except (TypeError, ValueError):
        raise ValueError(f"freq_index must be a whole number, got {freq_index!r}.") from None
    if index != freq_index:
        raise ValueError(f"freq_index must be a whole number, got {freq_index!r}.")
    if not 0 <= index < len(frequencies):
        raise ValueError(
            f"freq_index {freq_index} is out of range for the {len(frequencies)} monitored "
            f"frequencies {frequencies}."
        )
    return index


def _check_accumulator_budget(shape, num_frequencies: int, num_components: int, description: str):
    """Refuse an accumulator allocation that is implausibly large, before attempting it.

    A DFT monitor costs `nf * nx * ny * nz * 8 bytes` per component, so widening
    a working single-frequency monitor into a hundred-point spectrum multiplies
    its footprint by a hundred. Past `MAX_ACCUMULATOR_BYTES` (4 GiB per monitor,
    summed over components) that is refused with the arithmetic spelled out,
    rather than left to the allocator — on GPU an out-of-memory abort mid-run
    loses the whole job, and on host it swaps.
    """
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    total_bytes = cells * num_frequencies * num_components * COMPLEX64_BYTES
    if total_bytes > MAX_ACCUMULATOR_BYTES:
        raise ValueError(
            f"{description} would allocate {total_bytes / 1024**3:.2f} GiB "
            f"({num_frequencies} frequencies x {num_components} components x {tuple(int(n) for n in shape)} "
            f"cells x {COMPLEX64_BYTES} bytes), above the "
            f"{MAX_ACCUMULATOR_BYTES / 1024**3:.0f} GiB per-monitor limit (dft.MAX_ACCUMULATOR_BYTES). "
            f"Monitor fewer frequencies, fewer components, or a smaller region."
        )


def _positive_decimation(value: Any) -> int:
    """Validate an explicit MEEP DFT decimation factor."""
    if isinstance(value, bool):
        raise ValueError(
            f"decimation_factor must be a positive whole number of steps, got {value!r}."
        )
    try:
        factor = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"decimation_factor must be a positive whole number of steps, got {value!r}."
        ) from None
    if factor != value or factor < 1:
        raise ValueError(
            f"decimation_factor must be a positive whole number of steps, got {value!r}."
        )
    return factor


def _phase_factors(grid, omegas, scale: float, time: float):
    """Per-frequency DFT phase factors for E/D and for H/B, shaped to broadcast.

    Returns two `(nf, 1, 1, 1)` complex64 arrays holding
    `exp(i*omega*t) * dt/sqrt(2*pi)`, sampled at `time` for the electric
    components and at `time - 0.5*dt` for the magnetic ones (MEEP's leapfrog
    stagger). Both come from one host expression and one transfer, so a GPU run
    pays a single small copy per timestep rather than one per frequency, and the
    `(nf, 1, 1, 1)` shape lets the accumulate broadcast over the field slice
    instead of looping.
    """
    sample_times = np.array([time, time - 0.5 * grid.dt], dtype=np.float64)
    factors = np.exp(1j * np.outer(sample_times, omegas)) * scale
    both = grid.xp.asarray(factors.astype(np.complex64)).reshape(2, omegas.size, 1, 1, 1)
    return both[0], both[1]


def _frequency_summary(frequencies: tuple[float, ...]) -> str:  # Compact frequency set for a repr.
    if len(frequencies) == 1:
        return f"{frequencies[0]:.4f}"
    return f"{len(frequencies)}x[{min(frequencies):.4f}..{max(frequencies):.4f}]"


def yee_shifts(component: str) -> tuple[int, int, int]:  # Yee shift triple for a component, loudly.
    if component not in IYEE_SHIFTS:
        raise ValueError(
            f"Unknown field component '{component}'. "
            f"Valid components: {', '.join(sorted(IYEE_SHIFTS))}"
        )
    return IYEE_SHIFTS[component]


def _is_electric(component: str) -> bool:  # True for E/D components, which sample at time().
    return component[0] in ("E", "D")


def _apply_yee_interpolation(arr, shifts, target_shape):
    """Average a sliced field from its Yee positions onto cell centres.

    MEEP source: fields.cpp get_array() yee2cent interpolation. In every
    direction whose Yee shift is 0 the field is averaged as
    `0.5 * (f[i] + f[i+1])`, which consumes the one extra plane the caller
    sliced in that direction. Where the region abuts the grid edge no extra
    plane could be sliced, the shape already matches `target_shape`, and the
    average is skipped — the same edge behaviour as the source implementation.

    Args:
        arr: Field slice, extended by one cell in each interpolated direction.
        shifts: (shift_x, shift_y, shift_z) Yee shifts for the component.
        target_shape: Region shape (x1-x0, y1-y0, z1-z0) to reduce to.

    Returns:
        Array with shape equal to `target_shape`.
    """
    interpolated = arr
    shift_x, shift_y, shift_z = shifts
    target_nx, target_ny, target_nz = target_shape
    if shift_x == 0 and interpolated.shape[0] > target_nx:
        interpolated = 0.5 * (interpolated[:-1, :, :] + interpolated[1:, :, :])
    if shift_y == 0 and interpolated.shape[1] > target_ny:
        interpolated = 0.5 * (interpolated[:, :-1, :] + interpolated[:, 1:, :])
    if shift_z == 0 and interpolated.shape[2] > target_nz:
        interpolated = 0.5 * (interpolated[:, :, :-1] + interpolated[:, :, 1:])
    return interpolated


def _raw_factors(fields, name):
    """``Fields.component_factors(name)``, reproduced on RAW fetches.

    The same rule, the same arrays, in the same order -- ``(D, inv_eps)`` for an
    E component that is not stored and has a full inverse-epsilon volume, otherwise
    the stored component (E, or H under PML, or B for H without PML) -- but fetched
    without the read barrier, because the caller reads only a box of each and a
    barriered fetch would sync the whole volume first. The one branch this cannot
    reproduce raw is a scalar inverse epsilon, where the product is formed over the
    whole grid by ``get_E``; that branch keeps the barriered path and stays exact.
    """
    from . import host_writes  # noqa: PLC0415

    if name in ("Ex", "Ey", "Ez") and not fields._stored_E:
        displacement = host_writes.raw(fields, "D" + name[1])
        inverse = fields.inverse_epsilon_for(name)
        if getattr(inverse, "shape", None) == getattr(displacement, "shape", None):
            return (displacement, inverse)
        return (fields.get_component(name),)
    if name in ("Ex", "Ey", "Ez"):
        return (host_writes.raw(fields, name),)
    if name in ("Hx", "Hy", "Hz"):
        return (host_writes.raw(fields, name if fields._pml_active else "B" + name[1]),)
    return (host_writes.raw(fields, name),)


def _sliced_factors_product(fields, name, index):
    """``Fields.sliced_component(name, index)`` through the residency door.

    Bit-for-bit the same: each factor's box is a copy of the same cells, and the
    product is formed in the same order. Under a held residency the boxes come off
    the device as boxes; the whole-volume sync this replaces was, on a 3-D grid with
    one flux monitor, six volumes a step.
    """
    from . import host_writes  # noqa: PLC0415

    # THE ORIGINAL PATH WHEN NOTHING HOLDS. Reproducing ``component_factors`` raw
    # is only worth anything under a held residency, and the device-free tests
    # hand this function stub ``Fields`` objects that carry ``sliced_component``
    # and nothing else; asking those for ``_stored_E`` is an AttributeError. So
    # the raw form is taken only when a backend is actually holding arrays, and
    # the shipped method -- byte-identical by construction -- otherwise.
    if not host_writes.holding():
        return fields.sliced_component(name, index)
    factors = _raw_factors(fields, name)
    block = host_writes.gather_box(factors[0], index)
    for factor in factors[1:]:
        block = block * host_writes.gather_box(factor, index)
    return block


def _sliced_component(fields, grid, component, region, periodic=(False, False, False)):
    """Cell-centred slice of one component over a region.

    The Yee-to-centre average consumes one extra plane in every direction whose
    Yee shift is 0. Where the region ends at the last stored cell that plane is one
    lattice vector away, and on a *periodic* axis it is fetched — wrapped, with its
    Bloch factor — rather than given up.

    Giving it up is what this used to do, and it does not cost the far plane alone:
    the extended slice then has the region's own length, `_apply_yee_interpolation`
    sees nothing to consume, and the average is dropped for the WHOLE region on that
    axis. Measured against CPU MEEP's `get_dft_array` in a periodic cell, a slab
    whose upper z face is the cell face came back 2.7e-01 wrong overall and 4.7e-01
    wrong on its first plane — a plane nowhere near the boundary. The default
    region of a DFT monitor in a run with no PML is the whole grid, so every
    interpolated axis of the default monitor was affected.

    `periodic` says which axes may be continued; the driver derives it from the
    symmetry alone (see `FdtdDriver._monitor_periodic_axes`) — an absorber removes no
    wrap, because MEEP's boundaries come from `use_bloch` and a PML is a material
    graded underneath them.

    A MIRROR-FOLDED axis has no lattice vector, so the plane past its far face is
    not fetched from a lattice image. What stands past it depends on the outer
    boundary: over METALLIC walls the fold terminates on the zero the wall holds
    (`_append_metallic_planes`); over PERIODIC boundaries the second mirror at
    doubled ``n_full`` reflects, and the plane past the stored top is the
    parity-weighted image row `stepping._far_reflect_rows` names
    (`_append_reflected_planes`) — the same neighbour `fields.to_cell_center`
    supplies there. Either plane is appended for the same reason the periodic
    plane is fetched above: without it the extended slice has the region's own
    length, `_apply_yee_interpolation` finds nothing to consume, and THE AVERAGE IS
    DROPPED FOR THE WHOLE AXIS — the quadrant then holds raw Yee samples on the
    folded axis while every other axis is cell-centred, and nothing about the array
    says so. Measured on a 3 x 3 x 2 cell at resolution 10 with an X mirror, the
    default whole-grid monitor, against the same cells of the unfolded run: Ey
    6.55e-01 and Ez 5.02e-01 wrong (Ex, whose x Yee shift is 1 and which therefore
    needs no average on that axis, was 0.0), against 0.0 for all three once the
    plane is supplied. A region stopping short of the far face was already exact,
    which is why this survived — it is only the default region, the most ordinary
    request there is, that reached the far face.
    """
    x0, x1, y0, y1, z0, z1 = region
    target_shape = (x1 - x0, y1 - y0, z1 - z0)
    shifts = yee_shifts(component)
    bounds = ((x0, x1), (y0, y1), (z0, z1))
    counts = (grid.nx, grid.ny, grid.nz)
    reaches_far_face = [
        axis for axis in range(3)
        if shifts[axis] == 0 and bounds[axis][1] + 1 > counts[axis]
    ]
    wrapping = [axis for axis in reaches_far_face if periodic[axis]]
    # The far plane of EVERY non-wrapping axis is appended below: a mirror fold
    # over metallic walls and a plain metallic axis terminate on the PEC zero,
    # while a mirror fold over periodic boundaries reflects through its second
    # mirror. Selecting only the folded ones here — as this used to — dropped
    # the Yee-to-centre average for the WHOLE axis of any region reaching a PEC far
    # face: measured on a 2x2x4 metallic cell at res 10 (Gaussian Ez, 3-freq flux
    # against mp.get_fluxes), a full-cross-section z-plane one cell inside the high
    # wall read 2.02x MEEP (worst rel 1.02e+00; 4.6e-01 / 6.6e-01 at z = 1.90 /
    # 1.925), and every full-width transverse extent read ~1e-02 — repaired to
    # 4e-08 … 2e-07 by this one condition with the periodic twin and the low-wall
    # planes unmoved (the design notes (absorber-placement-evidence) §2.3, §3.3).
    terminated = [axis for axis in reaches_far_face if not periodic[axis]]
    metallic = [axis for axis in terminated
                if not _axis_is_mirrored(grid, axis) or grid.is_metallic(axis)]
    reflected = [axis for axis in terminated if axis not in metallic]
    if not wrapping:
        extended = tuple(
            min(bounds[axis][1] + (1 if shifts[axis] == 0 else 0), counts[axis]) for axis in range(3)
        )
        # A view over stored storage, and the block of `D * inv_eps` where E is
        # derived — `Fields.sliced_component` multiplies the two SLICES rather than
        # the two volumes, which is where a PML-free run's monitor cost lived.
        # Every downstream operation allocates a fresh array either way, so no
        # defensive copy is needed here.
        sliced = _sliced_factors_product(
            fields, component, (slice(x0, extended[0]), slice(y0, extended[1]), slice(z0, extended[2])))
        sliced = _append_reflected_planes(grid, sliced, reflected, component,
                                          tuple(bounds[axis][0] for axis in range(3)))
        return _apply_yee_interpolation(_append_metallic_planes(grid, sliced, metallic),
                                        shifts, target_shape)

    # Same reason the contiguous branch above slices through `Fields.sliced_component`:
    # where E is derived, indexing the two FACTORS beats forming the product over the
    # whole grid. Here the axes are taken one at a time, so the factors are carried in
    # lockstep and multiplied once the last axis is gathered. The Bloch phases are held
    # back to the same point IN THE SAME ORDER: each one depends on its own axis alone,
    # and a `take` along a different axis cannot permute that axis, so deferring them
    # multiplies every cell by the same factors in the same sequence.
    layers = fields.component_factors(component)
    deferred_phases = []
    for axis in range(3):
        low, high = bounds[axis]
        span = high - low + (1 if shifts[axis] == 0 else 0)
        if axis not in wrapping:
            block = (slice(None),) * axis + (slice(low, min(low + span, counts[axis])),)
            layers = tuple(layer[block] for layer in layers)
            continue
        indices, phases = _lattice_gather_device(grid, axis, low, span,
                                                 _axis_bloch_phase(grid, axis))
        layers = tuple(layer.take(indices, axis=axis) for layer in layers)
        if phases is not None:
            deferred_phases.append(phases)
    sampled = layers[0]
    for layer in layers[1:]:
        sampled = sampled * layer
    for phases in deferred_phases:
        sampled = sampled * phases
    sampled = _append_reflected_planes(grid, sampled, reflected, component,
                                       tuple(bounds[axis][0] for axis in range(3)))
    return _apply_yee_interpolation(_append_metallic_planes(grid, sampled, metallic),
                                    shifts, target_shape)


def _append_metallic_planes(grid, sampled, axes):  # Zero plane past the far face of each folded axis.
    """Give the Yee-to-centre average the plane a metallic termination holds at zero.

    A folded METALLIC axis (and a plain metallic one) stops at its window's top
    wall with a zero MEEP itself holds there (find_metals / zero_metal), so the
    cell-centre average of its last cell is `0.5 * (f[n-1] + 0)`. Supplying the
    zero explicitly is what keeps `_apply_yee_interpolation` averaging over the
    WHOLE axis instead of silently skipping it; the alternative loses the average
    everywhere on that axis, not just at the face.
    """
    for axis in axes:
        shape = list(sampled.shape)
        shape[axis] = 1
        zero = grid.xp.zeros(tuple(shape), dtype=sampled.dtype)
        sampled = grid.xp.concatenate((sampled, zero), axis=axis)
    return sampled


def _append_reflected_planes(grid, sampled, axes, component, lows):
    """Give the Yee-to-centre average the reflected plane of a folded PERIODIC axis.

    Past the stored top of such an axis stands the parity-weighted image of
    stored row ``n_full - stored + 2`` about the second mirror at doubled
    ``n_full`` — the row `stepping._far_reflect_rows` derives and
    `fields.to_cell_center` averages against. ``lows`` are the region's stored
    start indices, which locate that stored row inside the already-sliced array
    (the plane is taken from `sampled` itself, so an axis appended earlier in the
    loop contributes its plane to a later axis's corner exactly as the sequential
    unfold does). The parity multiply allocates a fresh plane, so `sampled`'s
    underlying storage is never written.
    """
    from .fields import mirror_parity  # Local: keeps the dft -> fields import one-way.

    for axis in axes:
        n_full = grid.shape_full[axis]
        reflect_row = n_full - grid.stored_cells(axis) + 2 - lows[axis]
        if reflect_row < 0 or reflect_row >= sampled.shape[axis]:
            raise ValueError(
                f"a region on folded axis {axis} reaches the far face but not the "
                f"reflect row {reflect_row + lows[axis]} its cell-centre average needs"
            )
        parity = mirror_parity(component, axis, grid.mirror_phase(axis))
        plane = parity * sampled[(slice(None),) * axis + (slice(reflect_row, reflect_row + 1),)]
        sampled = grid.xp.concatenate((sampled, plane), axis=axis)
    return sampled


def _boundary_weight_ladder(where_min, where_max, resolution, parity=1):
    """Return the MEEP fractional-cell integration weights for one direction.

    MEEP source: loop_in_chunks.cpp lines 233-297 (`vec2diel_floor`,
    `vec2diel_ceil`, `compute_boundary_weights`). The monitor's physical bounds
    are converted to MEEP's doubled integer coordinates, and the four
    Simpson-style boundary weights (s0, s1, e0, e1) are selected by the same
    case ladder — the `elif` order is semantically load-bearing and is
    reproduced verbatim.

    `parity` anchors the ladder to a grid: 1 (the default, and the only value any
    caller used before near-to-far) is the centred ("dielectric") grid whose sites
    sit at odd doubled coordinates; 0 is a Yee family whose sites on this axis sit
    at even doubled coordinates — a component whose iyee shift here is 0. MEEP
    makes the same distinction through the `cgrid` argument of `loop_in_chunks`
    (`Centered` for a DFT-fields region, the component itself for a near2far or
    flux chunk); the floor/ceil anchoring is the only thing parity changes, and
    the case ladder below is common because MEEP's `compute_boundary_weights`
    receives `is`/`ie` already anchored and never looks back at the grid.

    **The arithmetic ORDER is transcribed, not simplified.** `vec2diel_floor` and
    `vec2diel_ceil` are hard-wired to the odd lattice, so MEEP never floors on the
    component's own grid: it shifts the requested volume onto the centred grid,
    floors there, and shifts the integer result back (loop_in_chunks.cpp:352-356)::

        volume wherec(where + yee_c);   // yee_c = yee_shift(Centered) - yee_shift(cgrid)
        ivec is(vec2diel_floor(wherec.get_min_corner(), gv.a, 0) - iyee_c);

    Algebraically that is `parity + 2*floor(where*a - parity/2)`, which is what this
    engine computed. The two differ in FLOATING POINT on a surface that lands exactly
    on the lattice — the coordinate a user is most likely to type. At `where = -0.7`,
    `a = 20`, parity 0: MEEP's order evaluates `(where + 0.5/a)*a - 0.5` as
    `-13.999999999999998`, whose ceil is -13 and whose ladder is TWO cells; the direct
    form evaluates `where*a` as exactly `-14.0` and gets ONE. MEEP's own near2far chunk
    for Ep on that surface reports ``is``/``ie`` of -28/-26, and a one-cell ladder
    cannot be matched to it at all — the lift fails outright with "no migrated near2far
    region covers this chunk". The second cell carries weight `w1 = 0`, so the two
    ladders integrate the same field; only the layout differs, and the layout is MEEP's.

    `w0` and `w1` are the fractional overlaps of the first and last cell with
    the requested extent, so the resulting weight vector integrates to exactly
    `(where_max - where_min) * resolution` cells regardless of where the monitor
    falls relative to the grid. They are computed from the UNSHIFTED bounds, because
    `compute_boundary_weights` reads `where`, not `wherec`.

    Args:
        where_min: Lower physical bound of the monitor in this direction.
        where_max: Upper physical bound of the monitor in this direction.
        resolution: Grid points per unit length (MEEP `gv.a`).
        parity: 1 for odd-doubled sites (centred), 0 for even-doubled sites.

    Returns:
        (is_doubled, ie_doubled, (s0, s1, e0, e1)) — the inclusive doubled-coordinate
        bounds of the weight ladder and its four boundary weights.
    """
    a = resolution
    if parity not in (0, 1):
        raise ValueError(f"parity must be 0 or 1, got {parity!r}")
    yee_c = (1 - parity) * 0.5 / a  # The shift onto the centred grid, in length units.
    is_doubled = 1 + 2 * math.floor((where_min + yee_c) * a - 0.5) - (1 - parity)
    ie_doubled = 1 + 2 * math.ceil((where_max + yee_c) * a - 0.5) - (1 - parity)
    w0 = 1.0 - where_min * a + 0.5 * is_doubled
    w1 = 1.0 + where_max * a - 0.5 * ie_doubled
    num_doubled = ie_doubled - is_doubled
    if num_doubled >= 3 * 2:  # Three or more cells: independent start and end tapers.
        s0 = w0 * w0 / 2
        s1 = 1 - (1 - w0) * (1 - w0) / 2
        e0 = w1 * w1 / 2
        e1 = 1 - (1 - w1) * (1 - w1) / 2
    elif num_doubled == 2 * 2:  # Two cells: the two tapers share the interior weight.
        s0 = w0 * w0 / 2
        s1 = 1 - (1 - w0) * (1 - w0) / 2 - (1 - w1) * (1 - w1) / 2
        e0 = w1 * w1 / 2
        e1 = s1
    elif where_min == where_max:  # Zero extent: the surface splits between adjacent cells.
        s0, s1, e0, e1 = w0, w1, w1, w0
    elif num_doubled == 1 * 2:  # One cell: start and end tapers overlap.
        s0 = w0 * w0 / 2 - (1 - w1) * (1 - w1) / 2
        e0 = w1 * w1 / 2 - (1 - w0) * (1 - w0) / 2
        s1 = e0
        e1 = s0
    else:
        s0 = s1 = e0 = e1 = 1.0
    return is_doubled, ie_doubled, (s0, s1, e0, e1)


def _ivec_loop_weight(index, count, s0, s1, e0, e1):  # MEEP vec.hpp IVEC_LOOP_WEIGHT1x position weight.
    if 1 < index < count - 2:
        return 1.0
    if index == 0:
        return s0
    if index == 1:
        return s1
    if index == count - 1:
        return e0
    if index == count - 2:
        return e1
    return 1.0


def _first_owned_index(grid, axis: int, parity: int) -> int:
    """The lowest stored index MEEP would accumulate on one axis — its owned corner.

    `loop_in_chunks` never loops below the chunk's owned corner: it clips the requested
    ladder with ``ivec iscS(max(is - shifti, iscoS))`` (loop_in_chunks.cpp:439-442) where
    ``iscoS`` is bounded below by ``user_volume.little_owned_corner(cgrid)``. That corner
    is (vec.hpp:1102-1104)::

        little_owned_corner0(c) = little_corner() + one_ivec(dim) * 2 - iyee_shift(c)

    which in this engine's stored index — site ``j`` sits at ``origin + 2j + parity`` —
    is ``(origin + 2 - parity - origin - parity)/2 = 1 - parity``, independent of the
    grid: the boundary row of a component whose shift on this axis is 0 is written by
    the boundary condition, not owned, and is not integrated over.

    The one exception is the cylindrical AXIS, and it is an explicit special case in
    MEEP (vec.cpp:432-436)::

        if (dim == Dcyl && origin.r() == 0.0 && iloc.r() == 2) iloc.set_direction(R, 0);

    so on a Dcyl r axis touching r = 0 the shift-0 components (Ep, Hr, Ez) DO own their
    r = 0 row — which is then weighted by ``2*pi*0 = 0`` and contributes nothing, but is
    still stored and still counts toward the chunk's ``N``. Equivalently, on that axis
    the rule is "drop every site at doubled coordinate < 0" for both parities.

    Measured against MEEP's own chunk list on `disc_extraction_efficiency.py` (resolution
    50, cell 5 x 2.3, 4 chunks): the r-wall region's z ladder starts at doubled -116 for
    Hz/Ep (shift 0) and -115 for Hp/Ez (shift 1), and MEEP reports ``is.z`` of -112 and
    -113 respectively — ``1 - parity`` in both cases — while the z-cap region's r ladder
    starts at doubled 0 for Hr/Ep and 1 for Er/Hp, which is the axis exception.
    """
    if bool(getattr(grid, "cylindrical", False)) and axis == 0:
        return 0
    return 1 - int(parity)


def _clip_to_owned(first_owned, start, count, weights, what):
    """Drop the ladder's leading sites that fall below the owned corner, unrenormalized.

    The survivors keep the weights they already had. MEEP applies the start taper only
    where the chunk's own start still coincides with the ladder's — ``iscS == is`` gives
    ``(s0c, s1c) = (s0, s1)``, the single-site fallback ``iscS == is + 2`` gives
    ``s0c = s1``, and past that both stay 1.0 (loop_in_chunks.cpp:451-460) — which is
    exactly what dropping leading entries of ``[s0, s1, 1, ..., 1, e1, e0]`` produces.
    The clipped weight is simply lost, and matching MEEP means losing it too.

    Confirmed against MEEP's reported per-chunk weights on `disc_extraction_efficiency.py`:
    Er's z-cap ladder loses one r site and MEEP reports ``s0.r = 0.8750``, which is that
    ladder's ``s1``; Hp's r-wall ladder loses one z site and MEEP reports ``s0.z = 0.9783``,
    which is ITS ``s1`` (``1 - (1 - 0.7917)**2 / 2``).
    """
    dropped = first_owned - start
    if dropped <= 0:
        return start, count, weights
    if dropped >= count:
        raise ValueError(
            f"{what} lies entirely outside the sites MEEP owns on this axis: its ladder "
            f"spans stored indices [{start}, {start + count}) and the first owned index is "
            f"{first_owned}. There is no surface there for MEEP to accumulate."
        )
    return first_owned, count - dropped, weights[dropped:]


def _cylindrical_ring_weights(origin_doubled, parity, start, count, weights, dx):
    """Multiply one raw-Yee ladder axis by MEEP's Dcyl ring measure ``2*pi*r``.

    **The ring measure.** loop_in_chunks.cpp:505-512 builds the integration volume as
    ``dV = dV0 + dV1 * loop_i2``, where ``loop_i2`` is the R loop index (vec.cpp:262-271
    fixes the Dcyl loop order as P, R, Z, so ``loop_i2`` is R and ``loop_i1`` is the
    absent phi), with::

        dV1 = dV0 * 2 * pi * gv.inva;
        dV0 *= 2 * pi * fabs((S.transform(chunks[i]->gv[isc], sn) + shift).in_direction(R));

    so ``dV = inva**n_extent * 2*pi*(r(isc) + loop_i2 * inva)`` — the plain Cartesian
    ``dV0`` times ``2*pi*r`` at THIS SAMPLE'S OWN RADIUS, advancing one cell per row
    from the radius of the chunk's first sample. `update_dft` then folds it in through
    ``IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV0 + dV1*loop_i2)`` (dft.cpp:277).

    The radius is the COMPONENT's Yee radius, ``(origin_doubled + 2*j + parity)/2 * dx``,
    because ``isc`` is built on the ``cgrid = c`` lattice (loop_in_chunks.cpp:341-357 with
    ``use_centered_grid = false``, dft.cpp:231) and ``gv[iv]`` is that doubled coordinate
    in length units. It is NOT ``(start + row + 0.5)*dx``: that is the CELL-CENTRE radius
    :meth:`Near2FarMonitor._face_weights` uses because that monitor samples centres, and
    using it here would put every even-parity component (Ep, Hr, Ez) half a cell too far
    out and every odd-parity one (Er, Hp, Hz) half a cell too far in — a half-cell slip
    that leaves the far field smooth, plausible, and wrong.

    Verified against MEEP's own chunk list on `disc_extraction_efficiency.py` (resolution
    50): a z-cap region flat in z reports ``dV1 = 0.00251327 = (1/50)*2*pi*(1/50)`` and, for
    Er whose first owned site is at doubled r = 1, ``dV0 = 0.00125664 = (1/50)*2*pi*0.01``;
    for Hr and Ep, whose first site is the axis row at doubled r = 0, ``dV0`` is exactly 0.

    :func:`_clip_to_owned` has already removed the sites below the axis, so every radius
    here is non-negative and MEEP's ``fabs`` is a no-op.

    Returns:
        The weights with the ring folded in.
    """
    doubled = origin_doubled + 2 * (start + np.arange(count, dtype=np.int64)) + parity
    return weights * (2.0 * np.pi * (doubled.astype(np.float64) * 0.5 * dx))


def _axis_is_mirrored(grid, axis) -> bool:  # Is this axis mirror-folded?
    """Defer to `Grid.is_mirrored`, the one definition of "folded".

    The fallback serves the stub grids in test_dft.py, which model the
    registration conventions and carry no methods; it reads the same per-axis
    flags `Grid` publishes, `sym_z` included, so a folded Z can never be reported
    as an ordinary periodic axis the way a hard-coded `False` in the Z slot did.
    """
    reader = getattr(grid, "is_mirrored", None)
    if callable(reader):
        return bool(reader(axis))
    if axis not in (0, 1, 2):
        raise ValueError(f"Axis must be 0 (X), 1 (Y) or 2 (Z); got {axis!r}")
    return bool(getattr(grid, ("sym_x", "sym_y", "sym_z")[axis], False))


def _axis_mirror_phase(grid, axis):  # Declared phase of this axis's plane, or None.
    """Defer to `Grid.mirror_phase`; `None` where the axis carries no plane.

    The fallback is MEEP's own default phase for a stub grid that knows only which
    axes are folded, which is all those stubs declare.
    """
    reader = getattr(grid, "mirror_phase", None)
    if callable(reader):
        return reader(axis)
    return 1 if _axis_is_mirrored(grid, axis) else None


def _axis_origin_doubled(grid, axis):
    """Doubled-coordinate origin of a grid axis, so cell j sits at `origin + 2j + 1`.

    Defers to `Grid.origin_doubled`, which is the one definition; the fallback is
    for the stub grids in test_dft.py, which model the registration conventions and
    carry no methods. MEEP's `center_origin` rounds the cell count DOWN TO EVEN, so
    a centred axis starts at `-(n_full - n_full % 2)` — `-L/2` only for an even
    count — and a symmetry-halved axis at -2, placing cell 0 across the mirror
    plane (MEEP vec.cpp `halve()` / `icenter()`). Every axis halves the same way,
    Z included.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"Axis must be 0 (X), 1 (Y) or 2 (Z); got {axis!r}")
    reader = getattr(grid, "origin_doubled", None)
    if callable(reader):
        return reader(axis)
    if _axis_is_mirrored(grid, axis):
        return -2
    full = (grid.nx_full, grid.ny_full, grid.nz_full)[axis]
    return -(full - full % 2)


def _axis_ladder(grid, axis, where_min, where_max):
    """Cells MEEP's weight ladder covers on one axis, and the ladder's taper weights.

    MEEP source: loop_in_chunks.cpp `loop_in_chunks` — `is`/`ie` come from
    `vec2diel_floor`/`vec2diel_ceil` on the requested bounds and the same pair is
    then handed to `compute_boundary_weights`, so in MEEP the sampled cells and
    their weights are two halves of one decision. They are here too: this is the
    only place either is derived, and every caller — the region a monitor
    allocates and the weights it applies — reads both from this one return.
    Deriving them separately is what let a flux plane land on one set of cells
    and be weighted for another (see `FluxMonitor._setup_region`).

    Ladder point `i` sits at doubled coordinate `is_doubled + 2i` and grid cell
    `j` at `origin + 2j + 1`, which fixes the ladder's first cell index. The
    returned range may start below 0 or end past the axis: it is what the
    request covers, not what this grid stores, and clamping is the caller's.

    Args:
        grid: Computational grid; supplies the resolution and axis registration.
        axis: 0 (X), 1 (Y) or 2 (Z).
        where_min: Lower physical bound of the monitor on this axis.
        where_max: Upper physical bound; equal to `where_min` for a flat axis.

    Returns:
        (first_cell, cell_count, (s0, s1, e0, e1)) — the ladder's first stored-grid
        cell index, how many cells it spans, and its four boundary weights.
    """
    is_doubled, ie_doubled, taper = _boundary_weight_ladder(where_min, where_max, grid.resolution)
    cell_count = (ie_doubled - is_doubled) // 2 + 1
    first_cell = int(round((is_doubled - _axis_origin_doubled(grid, axis) - 1) / 2.0))
    return first_cell, cell_count, taper


def _axis_bloch_phase(grid, axis):  # grid.bloch_phase(axis) where the grid has one, else None.
    """Bloch wrap factor for one axis, tolerating a grid that predates Bloch support.

    The monitors are also driven by the test stubs, which model the registration
    conventions and nothing else, so the phase is read defensively; a grid without
    a ``bloch_phase`` is plain-periodic and every wrap factor is 1.
    """
    reader = getattr(grid, "bloch_phase", None)
    return reader(axis) if callable(reader) else None


def _axis_cell_count(grid, axis):  # Stored cell count on one axis.
    return (grid.nx, grid.ny, grid.nz)[axis]


def _lattice_gather(grid, axis, first_cell, cell_count, phase):
    """Stored-grid indices and per-cell Bloch factors for a run of ladder cells.

    MEEP source: loop_in_chunks.cpp lines 390-415. A monitor whose requested extent
    reaches past a *periodic* face is not clipped there — MEEP loops over the
    lattice shifts whose translated cell still intersects the request and
    multiplies each shifted copy by ``pow(eikna[d], ishift)``. Ladder cell
    ``first_cell + i`` therefore samples stored cell ``(first_cell + i) mod n`` and
    carries ``bloch_phase ** ((first_cell + i) // n)``, which is the same statement
    one cell at a time: cell ``j + n`` sits one lattice vector up from cell ``j``,
    and ``grid.bloch_phase`` is by definition the factor between them.

    Returns:
        (indices, phases) — an int array into the stored axis and the matching
        complex factors, or None when every factor is 1 (no Bloch phase, or no
        cell actually wrapped).
    """
    count = _axis_cell_count(grid, axis)
    raw = np.arange(first_cell, first_cell + cell_count, dtype=np.int64)
    shifts = np.floor_divide(raw, count)
    indices = raw - shifts * count
    if phase is None or not shifts.any():
        return indices, None
    factors = np.array([complex(phase) ** int(shift) for shift in shifts], dtype=np.complex64)
    return indices, factors


# One entry per distinct (backend, axis, axis length, ladder, Bloch phase) a process
# has gathered through. Bounded because a run has a handful of monitors and each has
# one ladder per wrapping axis; the reset below is a guard against a long-lived
# process sweeping thousands of geometries, not an expected path.
_LATTICE_GATHER_DEVICE: dict = {}
_LATTICE_GATHER_DEVICE_LIMIT = 1024


def _lattice_gather_device(grid, axis, first_cell, cell_count, phase):
    """:func:`_lattice_gather`'s result as BACKEND arrays, built once per run, not per step.

    The host build is pure — it depends on the axis length, the ladder and the Bloch
    phase, every one of them fixed when the monitor is registered — but it sat inside
    ``_sliced_component``'s wrapping branch and therefore ran on EVERY timestep, per
    component, per wrapping axis. Two things there cost more than they look:

      * ``[complex(phase) ** int(shift) for shift in shifts]`` is a PYTHON LOOP OVER
        THE AXIS LENGTH — the only true per-cell Python loop left anywhere in the
        per-timestep path. Measured 45 us at span 241, 199 us at 1201 and 826 us at
        4801, i.e. 0.17 us per cell, and it does not shrink on a GPU because none of
        it is array work.
      * the two ``xp.asarray`` calls that followed it are HOST-TO-DEVICE COPIES of
        arrays that never change, one per component per step.

    Both are run invariants, so they are built once here — the same thing
    ``_MirrorGather`` already does for the folded-axis tables. The phase vector comes
    back already shaped ``(1, 1, 1)`` with the axis extent in place, so the caller
    multiplies without reshaping.
    """
    key = (id(grid.xp), int(axis), _axis_cell_count(grid, axis), int(first_cell),
           int(cell_count), None if phase is None else complex(phase))
    entry = _LATTICE_GATHER_DEVICE.get(key)
    if entry is None:
        if len(_LATTICE_GATHER_DEVICE) >= _LATTICE_GATHER_DEVICE_LIMIT:
            _LATTICE_GATHER_DEVICE.clear()
        indices, factors = _lattice_gather(grid, axis, first_cell, cell_count, phase)
        shape = [1, 1, 1]
        shape[axis] = int(cell_count)
        entry = (grid.xp.asarray(indices),
                 None if factors is None else grid.xp.asarray(factors).reshape(tuple(shape)))
        _LATTICE_GATHER_DEVICE[key] = entry
    return entry


MIRROR_SITE_DIRECT = 0  # Stored as declared; factor 1.
MIRROR_SITE_REFLECTED = 1  # Image under a mirror; factor mirror_parity(component).
MIRROR_SITE_ZERO = 2  # A PEC wall row MEEP allocates and holds at zero; factor 0.


def folded_axis_sites(
    first_site: int,
    count: int,
    parity: int,
    *,
    n_full: int,
    stored: int,
    metallic: bool,
):
    """Map requested Yee sites on a mirror-folded axis onto stored rows — MEEP's sn loop.

    THE ONE TRANSCRIPTION of how ``loop_in_chunks`` serves a monitor site the fold
    did not store, fixed at registration time exactly as MEEP fixes it: the chunk a
    site lands in, the symmetry image ``sn`` that reaches it and the sign that image
    carries are all decided in ``fields::add_dft`` (dft.cpp:231) before a single
    step runs. Per site, in MEEP's own composition order — lattice shift first,
    then the mirror (loop_in_chunks.cpp:306 "we apply the symmetry first to the
    chunk, *then* the shift"; read from the site's side that is shift, then S):

    1. **Wrap into the user volume** (loop_in_chunks.cpp:390-415). A periodic axis
       translates the site by one lattice vector when it lies outside the owned
       window ``[little_owned_corner, big_owned_corner]`` of the FULL user volume
       (vec.hpp:1102-1107: ``little_corner + 2 - iyee_shift(c)``). A folded axis
       carries no Bloch phase — the grid refuses k on a mirrored axis — so the
       wrap factor is exactly 1. A metallic axis takes no shift at all
       (``boundaries[High][d] == Periodic`` gates ``min_ishift``); its window-edge
       parity-0 row is the PEC wall MEEP allocates and zeroes (``update_ntot``'s
       num + 1 slot, ``zero_metal``), served here as :data:`MIRROR_SITE_ZERO`,
       and anything beyond is served by no chunk — dropped, exactly as MEEP's
       ``iscS = max(is - shifti, iscoS)`` clamp drops it.
    2. **Reflect about the plane at doubled 0** (``symmetry::transform``,
       vec.cpp:1287-1301, with ``i_symmetry_point = icenter() = 0``): a site below
       the plane maps to its image ``d -> -d`` and the value picks up
       ``S.phase_shift(c, sn)`` — per component, ``fields.mirror_parity``. The
       sn=0 pass serves sites at and above the component's owned corner
       (``1 - iyee_shift``); the sn=1 pass is clamped one row below it by MEEP's
       ``isym`` bound (loop_in_chunks.cpp:426-445, ``i_symmetry_point -
       off_sym_shift``), so every site is served exactly once. Measured chunk
       anatomy, y = -0.7 region under Mirror(Y) at resolution 10: Ez (parity 0)
       sn=1 chunk at doubled y 12..14, Hx (parity 1) at 13..15 — the reflections
       of the request's -14..-12 / -15..-13 brackets.
    3. **Reflect through the second mirror** where the stored rows still end short
       of the wrapped-and-reflected site: under periodic boundaries a mirror at 0
       implies a second mirror at doubled ``n_full`` (translation + reflection —
       the same plane ``stepping._shift_up``'s reflect rule reads through), whose
       image carries the same parity factor. This is the odd-full-count window
       top: measured, the same y = -0.7 region on ``n_full = 15`` produces sn=0
       chunks at doubled +15/+16 — the below-window sites -15/-14 wrapped up by
       the doubled lattice vector 30 — where the even count's land at the far
       plane by reflection alone; both compositions are taken here.

    Args:
        first_site: Requested first site index, in the folded axis's stored-index
            convention — site ``j`` of parity ``p`` sits at doubled coordinate
            ``-2 + 2*j + p`` (``grid_volume::halve()`` sets the origin two half
            cells below the plane). May be negative: that is the request this
            function exists to serve.
        count: How many consecutive sites the request covers.
        parity: The component's Yee shift on this axis — 0 or 1. The centred
            (cell-centre) lattice is parity 1.
        n_full: Unfolded cell count of the axis.
        stored: Stored (folded) cell count — ``owned_cells`` plus the
            ``big_corner`` cell a periodic fold stores at either parity.
        metallic: Whether the axis is PEC-terminated; a folded axis that is not
            metallic is periodic (the second-mirror machinery exists only there).

    Returns:
        ``(dropped_low, indices, codes)`` — how many leading sites no chunk
        serves (MEEP's owned clamp; the survivors keep their own weights, exactly
        as :func:`_clip_to_owned` keeps them), an int64 array of stored row
        indices for the kept sites, and an int8 code per kept site
        (:data:`MIRROR_SITE_DIRECT` / ``_REFLECTED`` / ``_ZERO``). Trailing
        unservable sites are likewise dropped; a gap in the middle is impossible
        (the servable set is an interval) and raises as a registration slip.
    """
    if parity not in (0, 1):
        raise ValueError(f"parity must be 0 or 1, got {parity!r}")
    origin_full = -(n_full - n_full % 2)  # MEEP icenter(): round_down_to_even.
    lattice = 2 * n_full
    little_owned = origin_full + 2 - parity
    big_owned = origin_full + lattice - parity
    def serve(doubled: int):
        """(stored index, code) for one requested doubled coordinate, or None."""
        flips = 0
        if doubled < little_owned or doubled > big_owned:
            if metallic:
                # No lattice vector to shift by. The window-edge parity-0 row is
                # the PEC wall (allocated, zeroed by MEEP's zero_metal); anything
                # beyond it is served by no chunk at all.
                if parity == 0 and doubled == origin_full:
                    return 0, MIRROR_SITE_ZERO  # Value 0; the index is inert.
                return None
            doubled += lattice if doubled < little_owned else -lattice
            if doubled < little_owned or doubled > big_owned:
                return None  # More than one lattice image away; nothing serves it.
        if doubled < -parity:  # Below the plane: the sn image, with its parity sign.
            doubled = -doubled
            flips ^= 1
        offset = doubled + 2 - parity
        if offset % 2:
            raise RuntimeError(
                f"folded-axis site map: doubled {doubled} is not on the parity-"
                f"{parity} lattice — a registration slip, not a caller error."
            )
        index = offset // 2
        if index >= stored:
            if metallic:
                # The high PEC wall row (parity 0 only); held at zero.
                if parity == 0 and index == stored:
                    return stored - 1, MIRROR_SITE_ZERO
                return None
            doubled = lattice - doubled  # The second mirror at doubled n_full.
            flips ^= 1
            offset = doubled + 2 - parity
            if offset % 2 or not 0 <= offset // 2 < stored:
                return None
            index = offset // 2
        return index, MIRROR_SITE_REFLECTED if flips else MIRROR_SITE_DIRECT

    mapped: list[tuple[int, int]] = []  # (stored index, code) per kept site.
    dropped_low = 0
    dropped_high = 0
    for position in range(count):
        served = serve(-2 + 2 * (first_site + position) + parity)
        if served is None:
            if mapped:
                dropped_high += 1
            else:
                dropped_low += 1
            continue
        if dropped_high:
            raise RuntimeError(
                "folded-axis site map: an unservable site inside the request — the "
                "servable set must be an interval, so this is a mapping slip."
            )
        mapped.append(served)
    indices = np.asarray([index for index, _ in mapped], dtype=np.int64)
    codes = np.asarray([code for _, code in mapped], dtype=np.int8)
    return dropped_low, indices, codes


def _axis_is_metallic(grid, axis) -> bool:  # PEC-terminated axis, stub-tolerant.
    """Defer to ``Grid.is_metallic``; the stub grids in test_dft.py carry no methods."""
    reader = getattr(grid, "is_metallic", None)
    return bool(reader(axis)) if callable(reader) else False


def _axis_full_count(grid, axis) -> int:  # Unfolded cell count of one axis.
    return int((grid.nx_full, grid.ny_full, grid.nz_full)[axis])


def _folded_axis_sites(grid, axis, first_site, count, parity):
    """:func:`folded_axis_sites` with its scalars read off a grid."""
    return folded_axis_sites(
        first_site, count, parity,
        n_full=_axis_full_count(grid, axis),
        stored=_axis_cell_count(grid, axis),
        metallic=_axis_is_metallic(grid, axis),
    )


class _MirrorGather:
    """One folded axis's registered site map, ready for per-component sampling.

    The per-site CODES are component-independent — which sites are images is grid
    geometry — while the factor a code carries is not: a reflected site multiplies
    by ``mirror_parity(component, axis, phase)``, which is ±1 per component. Both
    factor vectors are therefore precomputed once (``even`` for parity +1
    components, ``odd`` for parity -1) and sampling picks by sign; a vector that
    would multiply by all ones is ``None`` so the hot path skips the multiply.

    ``base`` serves components whose Yee shift on the axis is 1 (their sites sit
    at the cell-centre spacing); ``extended`` serves shift-0 components, whose
    cell-centre average consumes one more site — the same split
    :func:`_register_volume`'s Bloch entries make.
    """

    __slots__ = ("base", "extended", "base_even", "base_odd",
                 "extended_even", "extended_odd")

    def __init__(self, xp, base_indices, base_codes, extended_indices, extended_codes):
        self.base = xp.asarray(base_indices)
        self.extended = xp.asarray(extended_indices)
        self.base_even, self.base_odd = self._factor_pair(xp, base_codes)
        self.extended_even, self.extended_odd = self._factor_pair(xp, extended_codes)

    @staticmethod
    def _factor_pair(xp, codes):
        zero = codes == MIRROR_SITE_ZERO
        reflected = codes == MIRROR_SITE_REFLECTED
        even = np.where(zero, 0.0, 1.0).astype(np.float32)
        odd = np.where(zero, 0.0, np.where(reflected, -1.0, 1.0)).astype(np.float32)
        return (
            xp.asarray(even) if zero.any() else None,
            xp.asarray(odd) if (zero.any() or reflected.any()) else None,
        )

    def factors(self, component_parity: int, interpolates: bool):
        """The multiply vector for one component, or ``None`` for all-ones."""
        if component_parity > 0:
            return self.extended_even if interpolates else self.base_even
        return self.extended_odd if interpolates else self.base_odd


def _fractional_cell_weights(grid, axis, where_min, where_max, index_start, cell_count):
    """Build the 1-D fractional-cell weight vector for one monitor axis.

    Combines `_axis_ladder` with the region's own start index so each cell of the
    monitor region receives the weight MEEP would apply through
    `include_dV_and_interp_weights`. Region cells outside the ladder (which only
    happens when the region was clamped to the grid) get weight 0, so the weights
    always integrate to the stored part of the requested physical extent.
    """
    xp = grid.xp
    ladder_start_index, ladder_count, (s0, s1, e0, e1) = _axis_ladder(
        grid, axis, where_min, where_max
    )
    weights = np.zeros(cell_count, dtype=np.float32)
    for ladder_index in range(ladder_count):
        cell = ladder_start_index + ladder_index - index_start
        if 0 <= cell < cell_count:
            weights[cell] = _ivec_loop_weight(ladder_index, ladder_count, s0, s1, e0, e1)
    return xp.asarray(weights, dtype=xp.float32)


def _monitor_volume_bounds(center, size, description):
    """Validate a monitor's physical volume and return its per-axis (min, max) bounds.

    A negative extent would put `vec2diel_floor` above `vec2diel_ceil` and hand back
    an inverted ladder — a region and a taper for a volume nobody could describe —
    so it is refused here rather than resolved into something plausible.
    """
    if center is None or size is None:
        raise ValueError(
            f"{description} needs both a center and a size; got center={center!r}, size={size!r}."
        )
    center_values, size_values = tuple(center), tuple(size)
    if len(center_values) != 3 or len(size_values) != 3:
        raise ValueError(
            f"{description} needs a three-component (x, y, z) center and size; got "
            f"center={center!r}, size={size!r}."
        )
    if any(not math.isfinite(float(value)) for value in center_values):
        raise ValueError(f"{description} must have a finite center on every axis, got {center!r}.")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in size_values):
        raise ValueError(
            f"{description} must have a finite, non-negative size on every axis, got {size!r}."
        )
    return tuple(
        (float(center_values[axis]) - float(size_values[axis]) / 2,
         float(center_values[axis]) + float(size_values[axis]) / 2)
        for axis in range(3)
    )


def _register_volume(grid, center, size, periodic, description):
    """Cells a monitor volume covers, and the gather that continues it past a wrapping face.

    One registration for both monitor kinds. Every axis takes its cells from
    `_axis_ladder`, i.e. from MEEP's own `vec2diel_floor` / `vec2diel_ceil` bracket
    of the requested bounds — the same pair `compute_boundary_weights` tapers — so a
    DFT region and a flux plane asked for the same volume land on the same cells,
    and both land where MEEP puts them.

    The DFT path used to derive its own indices instead, from a fixed half-cell
    margin and `round()`. MEEP's outward rounding costs a face no margin at all when
    it lands exactly on a row of cell centres (`vec2diel_floor` reaches that row and
    stops), while `round()` on the padded coordinate lands a cell lower or not
    depending on the *parity* of the cell index — so the region was one cell wider
    per centre-aligned face, sometimes. Measured against CPU MEEP's `get_dft_array`
    at resolution 10 in a 2x2x3 cell: a monitor of size 2.1 registered 20 cells
    against MEEP's 22, while the same monitor at size 1.9 agreed exactly. A shape
    that tracks parity is not a shape a caller can reason about.

    An axis whose ladder — or the one extra plane a Yee-to-centre average consumes
    on it — reaches past a *wrapping* face is continued into the neighbouring
    lattice image rather than clipped there, which is MEEP's lattice-shift loop
    (loop_in_chunks.cpp lines 390-415, `_lattice_gather`). Clipping is what the DFT
    path did, and it is not visible in the result: the region simply comes back
    smaller, holding the cells that happened to be in range. Measured against CPU
    MEEP's `get_dft_array` in a 2x2x3 periodic cell at resolution 10, before and
    after — complex relative L2 on Ex at k = 0 / at k = (0.15, -0.25, 0.35), or the
    shapes where the two could not be compared at all:

        past the +z face       (6, 6, 5) vs (6, 6, 7)    ->  1.1e-07 / 1.3e-07
        past the -z face       (6, 6, 5) vs (6, 6, 7)    ->  1.1e-07 / 1.3e-07
        centred on the +z face (6, 6, 3) vs (6, 6, 6)    ->  1.1e-07 / 1.2e-07
        full cross-section     (20,20, 7) vs (22,22, 6)  ->  1.1e-07 / 1.2e-07
        the whole cell         (20,20,30) vs (22,22,32)  ->  1.1e-07 / 1.2e-07
        wider than the cell    (20, 6, 7) vs (32, 6, 6)  ->  1.0e-07 / 1.1e-07
        flat plane on a face   4.9e-01 (values)          ->  4.3e-07 / 2.3e-07

    The last of those is the one that returned a number rather than a wrong shape: a
    zero-thickness region asked for at the cell face was clamped onto the last row
    of cell centres half a cell away and reported that row's field as the face's.

    `periodic` says which axes may be continued; the driver derives it from the run's
    boundaries (`FdtdDriver._monitor_periodic_axes`) — MEEP takes a lattice shift where
    `boundaries[High][d] == Periodic`, which `use_bloch` makes every unmirrored
    direction, absorber or not. A MIRRORED axis does not wrap; a request reaching
    below its plane is served through the reflected gather instead
    (`folded_axis_sites` — MEEP's `loop_in_chunks` symmetry loop, which serves DFT
    monitors through the fold because `add_dft` takes the `use_symmetry=true`
    default where `add_volume_source` passes false, dft.cpp:231 vs sources.cpp:487):
    the region keeps its requested cells, below-plane cells sample their stored
    image times the component's `mirror_parity`, and only what no image reaches is
    clipped. Measured against MEEP's own folded runs, the served regions match the
    unfolded control at the fold's own floor (1.1e-07..1.8e-06; see
    test_dft.py's fold-gather cases for the pinned values).

    Args:
        grid: Computational grid; supplies the resolution, registration and wrap phases.
        center: (x, y, z) centre of the requested volume, in MEEP coordinates.
        size: (sx, sy, sz) extent; 0 on an axis means a zero-thickness request there.
        periodic: Which axes may be continued into the neighbouring lattice image.
        description: How to name this monitor in an error message.

    Returns:
        (region, gather, wrapped) — the half-open index region; the per-axis
        (base indices, extended indices, base phases, extended phases) gather that
        `_gathered_component` samples through, or `None` when the region is a
        contiguous slice of the stored grid; and which axes were continued.
    """
    bounds = _monitor_volume_bounds(center, size, description)
    for axis in range(3):
        if periodic[axis] and _axis_is_mirrored(grid, axis):
            # MEEP reaches the far half of a mirrored axis through the symmetry
            # transform, never through a lattice shift: its lower face is the
            # symmetry plane and its upper face the physical edge of the folded
            # half. Continuing it periodically would sample the folded quadrant as
            # if it were the neighbouring cell — plausible values from the wrong
            # half of the domain. Nothing in the engine asks for that (the driver
            # and FluxMonitor both exclude mirrored axes), so it is a caller error.
            raise ValueError(
                f"{description} is marked periodic on the {AXIS_NAMES[axis]} axis, which is "
                f"mirror-folded. A mirrored axis is reached by the symmetry transform, not by a "
                f"lattice shift; continuing it periodically would sample the folded half as "
                f"though it were the next cell."
            )
    ladders = [_axis_ladder(grid, axis, *bounds[axis]) for axis in range(3)]
    # A mirrored axis whose ladder reaches below the plane is served through the
    # reflected gather (`folded_axis_sites`) — MEEP's use_symmetry=true loop — not
    # clamped at cell 0 the way it once was.
    mirrored_below = tuple(
        _axis_is_mirrored(grid, axis) and ladders[axis][0] < 0 for axis in range(3)
    )
    # The +1 is the plane a Yee-to-centre average consumes: a region ending on the
    # last row of cell centres needs the wrapped neighbour to interpolate a
    # component whose Yee shift is 0 on that axis, even though its own ladder is in
    # range.
    wrapped = tuple(
        bool(periodic[axis])
        and (ladders[axis][0] < 0
             or ladders[axis][0] + ladders[axis][1] + 1 > _axis_cell_count(grid, axis))
        for axis in range(3)
    )
    if not any(wrapped) and not any(mirrored_below):
        region = _clamp_region(
            grid, tuple(value for first, count, _ in ladders for value in (first, first + count))
        )
        _require_populated_region(grid, region, description)
        return region, None, wrapped

    starts, counts, gathered = [], [], []
    for axis in range(3):
        first_cell, cell_count, _ = ladders[axis]
        stored = _axis_cell_count(grid, axis)
        if mirrored_below[axis]:
            # The reflected gather: requested cells keep their own (requested-space)
            # indices and weights; each samples the stored row its symmetry image
            # lands on, with the per-component parity applied at sample time —
            # registration decides everything, exactly as MEEP's add_dft does.
            dropped, base_indices, base_codes = _folded_axis_sites(
                grid, axis, first_cell, cell_count, parity=1
            )
            if base_indices.size == 0:
                raise ValueError(
                    f"{description} lies wholly outside what the folded {AXIS_NAMES[axis]} "
                    f"axis can serve: no stored row, reflected image or lattice image "
                    f"reaches it. A monitor there would accumulate nothing and report "
                    f"zeros, which read as a real measurement."
                )
            start = first_cell + dropped
            extension = _folded_axis_sites(grid, axis, start, int(base_indices.size) + 1, parity=0)
            if extension[0] or extension[1].size != base_indices.size + 1:
                raise RuntimeError(
                    f"{description}: the folded {AXIS_NAMES[axis]} axis serves cell centres "
                    f"the parity-0 site map cannot cover — a mapping slip, not a caller error."
                )
            starts.append(start)
            counts.append(int(base_indices.size))
            gathered.append(_MirrorGather(grid.xp, base_indices, base_codes,
                                          extension[1], extension[2]))
            continue
        if wrapped[axis]:
            last = first_cell + cell_count - 1
            # MEEP would tile as many images as the request covers, but a monitor
            # several cells outside the cell is a caller error far more often than a
            # deliberate many-image integral, and answering with an image nobody
            # asked about is the same silent-plausible-answer failure as zero.
            if first_cell // stored < -1 or last // stored > 1:
                raise ValueError(
                    f"{description} does not intersect the {grid.shape} grid or either of its "
                    f"adjacent periodic images on the {AXIS_NAMES[axis]} axis: it asks for cells "
                    f"{first_cell}..{last} of {stored}. Reporting a lattice image that far away "
                    f"would read as a real measurement of the cell."
                )
            phase = _axis_bloch_phase(grid, axis)
            base_indices, base_phases = _lattice_gather(grid, axis, first_cell, cell_count, phase)
            extended = _lattice_gather(grid, axis, first_cell, cell_count + 1, phase)
            starts.append(first_cell)
            counts.append(cell_count)
            gathered.append((base_indices, extended[0], base_phases, extended[1]))
            continue
        low = max(0, min(stored, first_cell))
        high = max(0, min(stored, first_cell + cell_count))
        if high - low < 1:
            raise ValueError(
                f"{description} does not intersect the {grid.shape} grid on the "
                f"{AXIS_NAMES[axis]} axis. A monitor outside the cell would accumulate nothing "
                f"and report zeros, which read as a real measurement."
            )
        starts.append(low)
        counts.append(high - low)
        gathered.append((np.arange(low, high, dtype=np.int64),
                         np.arange(low, min(high + 1, stored), dtype=np.int64), None, None))
    xp = grid.xp
    region = (starts[0], starts[0] + counts[0], starts[1], starts[1] + counts[1],
              starts[2], starts[2] + counts[2])
    gather = tuple(
        entry if isinstance(entry, _MirrorGather) else
        (xp.asarray(entry[0]), xp.asarray(entry[1]),
         None if entry[2] is None else xp.asarray(entry[2]),
         None if entry[3] is None else xp.asarray(entry[3]))
        for entry in gathered
    )
    return region, gather, wrapped


def _require_populated_region(grid, region, description):
    """Refuse a region that resolved to no cells on some axis.

    An empty accumulator is the worst kind of answer this engine can give: every
    reduction over it returns 0 — zero power, zero field, a flat spectrum — and
    nothing in the number says it came from a monitor that missed the cell.
    """
    x0, x1, y0, y1, z0, z1 = region
    shape = (x1 - x0, y1 - y0, z1 - z0)
    if min(shape) < 1:
        raise ValueError(
            f"{description} does not intersect the {grid.shape} grid: it resolves to {shape} "
            f"cells. A monitor outside the cell would accumulate nothing and report zeros, "
            f"which read as a real measurement."
        )


def _gathered_component(fields, grid, component, gather, region):
    """Cell-centred sample of one component over a region that crosses a periodic face.

    The wrapping counterpart of `_sliced_component`: each axis is gathered through
    its index list instead of sliced, the wrapped cells carry their Bloch factor,
    and the Yee average then runs on the gathered array exactly as it does on a
    contiguous slice — the extra plane it consumes is simply the next ladder cell,
    wrapped like any other.

    A MIRROR-FOLDED axis takes one of two shapes here. An axis whose region stays
    at or above the plane keeps the same exception `_sliced_component` makes:
    `_register_volume` clips its extended index list at the last stored cell
    (there is no lattice image to gather), so a region that reaches the far face
    gets an extended list no longer than its base one and
    `_apply_yee_interpolation` drops the average for the whole axis. The plane
    past that face is the metallic zero over metallic walls and the second
    mirror's parity-weighted image over periodic boundaries, appended here so the
    average runs — see `_append_metallic_planes` / `_append_reflected_planes` for
    the measured cost of leaving it out. An axis whose region reaches BELOW the
    plane carries a :class:`_MirrorGather` instead: its index lists already name
    the stored row each requested site's symmetry image lands on (including any
    far-face plane, as a code rather than an append), and the per-site factor is
    the component's own `fields.mirror_parity` — MEEP's ``S.phase_shift(c, sn)``,
    which `add_dft_chunkloop` bakes into the reflected chunk's scale
    (dft.cpp:169-171).
    """
    from .fields import mirror_parity  # Local: keeps the dft -> fields import one-way.

    shifts = yee_shifts(component)
    # The factors of the component, gathered in lockstep and multiplied at the end —
    # see `Fields.component_factors` for the measured cost of forming `D * inv_eps`
    # over the whole grid to keep a plane, and `_sliced_component`'s wrapping branch
    # for why holding the per-axis phase and parity factors back is the same value in
    # the same order.
    layers = fields.component_factors(component)
    deferred_factors = []
    target_shape = (region[1] - region[0], region[3] - region[2], region[5] - region[4])
    metallic = []
    reflected = []
    lows = (region[0], region[2], region[4])
    for axis in range(3):
        entry = gather[axis]
        interpolates = shifts[axis] == 0
        if isinstance(entry, _MirrorGather):
            parity = mirror_parity(component, axis, _axis_mirror_phase(grid, axis))
            factors = entry.factors(parity, interpolates)
            take_indices = entry.extended if interpolates else entry.base
            layers = tuple(layer.take(take_indices, axis=axis) for layer in layers)
            if factors is not None:
                shape = [1, 1, 1]
                shape[axis] = factors.shape[0]
                deferred_factors.append(factors.reshape(tuple(shape)))
            continue
        base, extended, base_phase, extended_phase = entry
        take_indices = extended if interpolates else base
        take_phase = extended_phase if interpolates else base_phase
        if interpolates and extended.shape[0] == base.shape[0]:
            # The extended list was clipped at a far face that cannot be continued —
            # `_register_volume` builds it as `arange(low, min(high + 1, stored))` — so
            # without a plane appended here `_apply_yee_interpolation` finds nothing to
            # consume and DROPS THE AVERAGE FOR THE WHOLE AXIS, not just its last cell.
            #
            # This used to require `_axis_is_mirrored`, and the missing half — a plain
            # NON-mirrored axis terminated by a wall — is the same defect
            # `_sliced_component` records repairing on its own path, reached whenever
            # some OTHER axis puts the monitor on the gathered path. A 1-D run does
            # exactly that: `_monitor_periodic_axes` marks the two invariant axes
            # periodic, so every monitor gathers, and a region reaching the high z face
            # lost its z average entirely. Measured against CPU MEEP's own
            # `get_dft_array` on test_planewave_1D's cell (resolution 100, 1-D, PML 1.0,
            # Ex at f=1): the whole-cell region read **3.17e-02** wrong and a region
            # touching only the high face **3.15e-02**, both as a uniform phase factor
            # of `exp(-i*2*pi*f*dx/2)` = 1.000965 - 0.031437j — a half-cell displacement
            # of every sample, over an array whose low-face twin was exact at 2.8e-07.
            (metallic if (not _axis_is_mirrored(grid, axis) or grid.is_metallic(axis))
             else reflected).append(axis)
        layers = tuple(layer.take(take_indices, axis=axis) for layer in layers)
        if take_phase is not None:
            shape = [1, 1, 1]
            shape[axis] = take_phase.shape[0]
            deferred_factors.append(take_phase.reshape(tuple(shape)))
    sampled = layers[0]
    for layer in layers[1:]:
        sampled = sampled * layer
    for factor in deferred_factors:
        sampled = sampled * factor
    sampled = _append_reflected_planes(grid, sampled, reflected, component, lows)
    return _apply_yee_interpolation(_append_metallic_planes(grid, sampled, metallic),
                                    shifts, target_shape)


def _axis_coordinates(grid, axis, start, stop):
    """Host cell-centre coordinates for a run of region cells, continued past the faces.

    A region continued into a neighbouring lattice image holds cell indices outside
    `[0, n)`, where a plain slice of `grid.x` silently truncates — or, for a negative
    start, silently returns the far end of the axis and hands the caller coordinates
    that do not belong to its data. Cell `j` sits at `axis[0] + j*dx` for every `j`,
    including outside the stored range, so the coordinates are built from that and
    run monotonically across the face, which is where the samples actually are.
    """
    origin = float(to_numpy((grid.x, grid.y, grid.z)[axis])[0])
    return (origin + np.arange(start, stop, dtype=np.float64) * grid.dx).astype(np.float32)


@dataclass
class YeeRegionDFT:
    """Raw-Yee DFT accumulator over one region of one component — the near2far chunk.

    Translation of the `dft_chunk` MEEP's `add_dft_near2far` creates (near2far.cpp,
    dft.cpp): `use_centered_grid=false`, so the stored samples are the component's own
    Yee sites with NO cell-centre averaging — which is why :class:`DFTMonitor`, whose
    `_sample` interpolates to centres, cannot stand in for it. Accumulation is
    otherwise MEEP's: every `decimation_factor`-th step adds
    `exp(i*omega*t) * dt*decimation/sqrt(2*pi) * f`, with E components sampled at
    `time` and H components at `time - dt/2` (`_phase_factors`, the same stagger the
    other monitors pin against CPU MEEP).

    What MEEP folds in per step and this class applies once at :meth:`packed` — the
    `include_dV_and_interp_weights` factor — is time-independent, so pulling it out of
    the loop is algebraically exact: `sum_t phase*(w*f) == w * sum_t phase*f`. The
    weight per site is the per-axis interpolation-ladder product times `dV`, where dV
    is one factor of `1/resolution` per axis the region has EXTENT on (measured on
    CPU MEEP: a plane's interior sites imply 1e-2 = (1/10)^2 at resolution 10, with
    the straddling/half-weight and fractional-edge tiers on top).

    Per-axis registration anchors the ladder to the COMPONENT's Yee parity
    (`_boundary_weight_ladder(..., parity=yee_shifts(c)[axis])`), which is the
    `cgrid=c` half of MEEP's `loop_in_chunks` that the centred monitors never
    exercise. An invariant axis of a reduced-dimension run contributes its single
    stored plane at weight 1 and no dV factor, because MEEP's 2-D loop has no such
    axis at all. A one-pixel PERIODIC axis with a zero-extent request — MEEP's
    emulation of a missing dimension (`nosize_direction`, fields.cpp:730-737) —
    contributes the same single plane at weight 1: MEEP splits it over the two
    lattice images of the one cell, whose weights sum to exactly 1.

    Every axis is clipped to the sites MEEP OWNS (:func:`_first_owned_index`,
    :func:`_clip_to_owned`) — a ladder reaching past the cell's low face loses those
    sites, and the survivors keep their untouched weights. On a CYLINDRICAL grid the
    radial axis additionally carries MEEP's ring measure, ``2*pi*r`` at each sample's own
    Yee radius (:func:`_cylindrical_ring_weights`), and its owned corner is the axis row
    itself rather than one cell in. The phase stagger, the dV power and the packed layout
    are unchanged: `Dcyl` differs from `D2` only in `loop_in_chunks`'s ``dV0``/``dV1``.

    A MIRROR-FOLDED axis whose ladder reaches below the component's owned corner is
    served through the reflected gather (:func:`folded_axis_sites` — the same
    ``use_symmetry=true`` loop that creates MEEP's sn > 0 chunks, dft.cpp:231):
    the accumulator keeps the REQUESTED sites, each sampling the stored row its
    symmetry image lands on, and the image's sign — ``S.phase_shift(c, sn)``,
    which MEEP bakes into the reflected chunk's scale (dft.cpp:169-171) — is
    folded into that axis's weight vector, exact because it is time-independent
    and real (±1 for a Cartesian mirror at either declared phase). What comes
    back from :meth:`packed` therefore equals MEEP's own chunk data on both sides
    of the plane: the reflected chunk's dft is ``sum_t phase*parity*raw`` and so
    is ours.

    The remaining scope refusals are the constructor's: on an UNFOLDED axis a
    region reaching past the stored high face, or wrapping a periodic axis, is
    refused — MEEP itself clips the one and wraps the other (see the constructor's
    error text for the measured basis), but this accumulator implements only the
    low-side owned-corner clip there, which IS MEEP's own loop. The caller
    supplies explicit physical bounds per axis.
    """

    grid: Any
    component: str
    bounds: tuple  # ((x0, x1), (y0, y1), (z0, z1)) physical bounds, min <= max.
    frequencies: Sequence[float] = ()
    decimation_factor: int = 1

    def __post_init__(self):
        self.component = str(self.component)
        shifts = yee_shifts(self.component)
        self.frequencies = tuple(float(f) for f in self.frequencies)
        if not self.frequencies:
            raise ValueError("YeeRegionDFT needs at least one frequency.")
        self._omegas = 2 * np.pi * np.asarray(self.frequencies, dtype=np.float64)
        self.decimation_factor = _positive_decimation(self.decimation_factor)
        self._scale = self.grid.dt * self.decimation_factor / math.sqrt(2 * math.pi)
        self._update_calls = 0

        from .fields import mirror_parity  # Local: keeps the dft -> fields import one-way.

        starts, counts, axis_weights = [], [], []
        takes: list[Any] = [None, None, None]
        extent_axes = 0
        invariant = getattr(self.grid, "is_invariant", None)
        cylindrical = bool(getattr(self.grid, "cylindrical", False))
        for axis in range(3):
            low, high = (float(b) for b in self.bounds[axis])
            if high < low:
                raise ValueError(f"bounds on axis {axis} are reversed: ({low}, {high}).")
            if callable(invariant) and invariant(axis):
                # MEEP's reduced-dimension loop has no such axis: one stored plane,
                # weight 1, and no dV factor from it.
                starts.append(0)
                counts.append(1)
                axis_weights.append(np.ones(1, dtype=np.float64))
                continue
            nosize = getattr(self.grid, "nosize_direction", None)
            if (high == low and callable(nosize) and nosize(axis)
                    and _axis_cell_count(self.grid, axis) == 1):
                # A one-pixel periodic axis is MEEP's own lower-dimensional emulation
                # (fields.cpp:730-737), and a region with no extent there means the
                # whole direction. Running the ownership ladder instead clips away the
                # axis's only stored site (`first_owned = 1 - parity` removes index 0
                # for a shift-0 component — the measured failure mode of
                # binary_grating_n2f's point region). MEEP loops the zero-extent split
                # pair over the two lattice images of the single cell, one dft chunk
                # per image, whose single-site weights are w0 and w1 = 1 - w0 (measured
                # on that region: two chunks per component, is.y = ie.y, s0.y = 0.0 and
                # 1.0) — so the one stored plane carries their sum, exactly 1, and no
                # dV factor (MEEP's own chunks report dV0 = 1.0 for the point region).
                starts.append(0)
                counts.append(1)
                axis_weights.append(np.ones(1, dtype=np.float64))
                continue
            parity = shifts[axis]
            is_doubled, ie_doubled, taper = _boundary_weight_ladder(
                low, high, self.grid.resolution, parity=parity
            )
            count = (ie_doubled - is_doubled) // 2 + 1
            origin = _axis_origin_doubled(self.grid, axis)
            start_doubled = is_doubled - origin - parity
            if start_doubled % 2:
                raise RuntimeError(
                    f"{self.component} axis {axis}: ladder start {is_doubled} is not on "
                    f"this component's Yee lattice (origin {origin}, parity {parity}) — "
                    f"a registration slip, not a caller error."
                )
            start = start_doubled // 2
            weights = np.asarray(
                [_ivec_loop_weight(i, count, *taper) for i in range(count)],
                dtype=np.float64,
            )
            if (_axis_is_mirrored(self.grid, axis)
                    and start < _first_owned_index(self.grid, axis, parity)):
                # Below the component's owned corner on a folded axis: MEEP serves
                # these sites through the symmetry loop's reflected chunks, so this
                # accumulator gathers each requested site's stored image and folds
                # the image's sign — S.phase_shift(c, sn), per component — into
                # this axis's weights, exact because it is time-independent.
                dropped, image_indices, image_codes = _folded_axis_sites(
                    self.grid, axis, start, count, parity
                )
                if image_indices.size == 0:
                    raise ValueError(
                        f"{self.component} region [{low}, {high}] on axis {axis} lies "
                        f"wholly outside what the folded axis serves: no stored row, "
                        f"reflected image or lattice image reaches it."
                    )
                start += dropped
                count = int(image_indices.size)
                weights = weights[dropped:dropped + count]
                parity_sign = mirror_parity(
                    self.component, axis, _axis_mirror_phase(self.grid, axis)
                )
                weights = weights * np.where(
                    image_codes == MIRROR_SITE_ZERO, 0.0,
                    np.where(image_codes == MIRROR_SITE_REFLECTED, float(parity_sign), 1.0),
                )
                takes[axis] = self.grid.xp.asarray(image_indices)
                starts.append(start)
                counts.append(count)
                axis_weights.append(weights)
                if high > low:
                    extent_axes += 1
                continue
            # MEEP loops only over the sites it owns, so a ladder reaching past the cell's
            # low face is clipped there rather than refused — and on a Dcyl r axis that
            # clip is what keeps a cap spanning r = 0 from asking for a row below the axis.
            start, count, weights = _clip_to_owned(
                _first_owned_index(self.grid, axis, parity),
                start, count, weights,
                f"{self.component} region [{low}, {high}] on axis {axis}",
            )
            if cylindrical and axis == 0:
                weights = _cylindrical_ring_weights(
                    origin, parity, start, count, weights, self.grid.dx
                )
            stored = _axis_cell_count(self.grid, axis)
            if start + count == stored + 1 and self.grid.is_metallic(axis):
                # THE PEC WALL ROW MEEP OWNS PAST THE STORED INTERIOR. On a
                # metallic axis MEEP's owned corner for a shift-0 component is one
                # site ABOVE the last interior one — the wall itself — and that site
                # is identically zero for the tangential field a PEC wall holds
                # there. Measured on `add_dft_fields(yee_grid=True)` over the whole
                # cell of a 1-D run (resolution 50, cell 12, PML 1.0, Ex at f = 1):
                # MEEP's array is 600 long over doubled z sites -598..600 of a
                # 600-cell grid — index 1 through index 600 — and its last element is
                # EXACTLY 0.0 + 0.0j while the interior maxes at 4.206. The low
                # site (index 0, doubled -600) is the one MEEP drops, which
                # `_clip_to_owned` above already reproduces.
                #
                # The site is served the way the folded path serves its zero image
                # (MIRROR_SITE_ZERO): a valid stored index at weight 0, so the sample
                # costs one gather and contributes exactly nothing. Reproducing
                # MEEP's SHAPE is the point — a script that reads the array and
                # builds coordinates from `len(array)` (test_planewave_1D does
                # exactly that) mislabels every sample if the length is one short.
                #
                # Only a one-site overflow on a PEC axis is served. A larger
                # overflow is a region genuinely outside the cell, and a periodic
                # axis wraps onto lattice images this accumulator does not gather;
                # both keep the refusal below.
                takes[axis] = self.grid.xp.asarray(np.concatenate([
                    np.arange(start, stored, dtype=np.int64),
                    np.asarray([max(start, 0)], dtype=np.int64),
                ]))
                weights = np.asarray(weights, dtype=np.float64).copy()
                weights[-1] = 0.0
                starts.append(start)
                counts.append(count)
                axis_weights.append(weights)
                if high > low:
                    extent_axes += 1
                continue
            if start < 0 or start + count > stored:
                raise ValueError(
                    f"{self.component} region [{low}, {high}] on axis {axis} needs Yee "
                    f"sites [{start}, {start + count}) of a grid storing {stored}. MEEP "
                    f"itself accepts this — it clips a non-periodic axis to the sites "
                    f"it owns (loop_in_chunks' iecS clamp; measured: a face 3x the "
                    f"cell width gives far fields identical to the cell-wide face at "
                    f"0.0e+00) and wraps a periodic axis onto lattice images — but "
                    f"this accumulator gathers the wrapped images for neither, and "
                    f"serves the boundary row only for the ONE-site overflow of a PEC "
                    f"axis (where MEEP's own value is identically zero), so the region "
                    f"is refused; the gate names it pre-flight."
                )
            starts.append(start)
            counts.append(count)
            axis_weights.append(weights)
            if high > low:
                extent_axes += 1

        self._slices = tuple(slice(s, s + c) for s, c in zip(starts, counts))
        self._takes = tuple(takes) if any(entry is not None for entry in takes) else None
        volume_element = (1.0 / float(self.grid.resolution)) ** extent_axes
        self._weights = (
            axis_weights[0][:, None, None]
            * axis_weights[1][None, :, None]
            * axis_weights[2][None, None, :]
            * volume_element
        )
        self._dft = self.grid.xp.zeros(
            (len(self.frequencies), counts[0], counts[1], counts[2]),
            dtype=self.grid.xp.complex64,  # MEEP accumulates in complex<realnum>.
        )

    def _configure_decimation_factor(self, factor: int) -> None:
        """Driver-resolved automatic decimation; same contract as DFTMonitor's."""
        if self._update_calls:
            raise RuntimeError(
                "Cannot change a Yee-region DFT's decimation factor after accumulation "
                "has started."
            )
        self.decimation_factor = _positive_decimation(factor)
        self._scale = self.grid.dt * self.decimation_factor / math.sqrt(2 * math.pi)

    def update(self, fields, time, current_step=None):
        """Accumulate one step, on MEEP's decimation and stagger conventions."""
        if current_step is None:
            self._update_calls += 1
            step = self._update_calls
        else:
            step = int(current_step)
            self._update_calls = step
        if step % self.decimation_factor:
            return
        phase_electric, phase_magnetic = _phase_factors(
            self.grid, self._omegas, self._scale, time
        )
        phase = phase_electric if _is_electric(self.component) else phase_magnetic
        raw = fields.get_component(self.component)
        if self._takes is None:
            raw = raw[self._slices]
        else:
            for axis in range(3):
                indexer = self._takes[axis]
                if indexer is None:
                    raw = raw[_slab(axis, self._slices[axis])]
                else:
                    raw = raw.take(indexer, axis=axis)
        self._dft += phase * raw

    def first_index(self, axis: int) -> int:  # This accumulator's first stored index on an axis.
        return self._slices[axis].start

    def site_doubled(self, axis: int, index: int) -> int:
        """Doubled coordinate of a stored index, in MEEP's own integer lattice.

        Index ``j`` on this axis sits at ``origin_doubled + 2j + yee_shift``, verified
        pointwise against ``sim.get_field_point`` at 3.4e-07…6.5e-06 for Ex/Ey/Hx/Hy.
        This is the inverse used to line our samples up with a MEEP chunk's ``is``/``ie``.
        """
        return _axis_origin_doubled(self.grid, axis) + 2 * index + yee_shifts(self.component)[axis]

    def index_for_doubled(self, axis: int, doubled: int) -> int:
        """Stored index holding a given doubled coordinate; inverse of :meth:`site_doubled`."""
        offset = doubled - _axis_origin_doubled(self.grid, axis) - yee_shifts(self.component)[axis]
        if offset % 2:
            raise ValueError(
                f"{self.component}: doubled coordinate {doubled} on axis {axis} is not on "
                f"this component's Yee lattice."
            )
        return offset // 2

    def weighted_square(self, index: int) -> float:
        """``sum_sites |w| * |F|^2`` at one frequency — MEEP's ``sqrt_dV`` reduction.

        The DIAGONAL stress-tensor terms are the one consumer that must not go
        through :meth:`packed`. MEEP builds those chunks with
        ``sqrt_dV_and_interp_weights = true`` (stress.cpp:178-185), so
        ``update_dft`` folds ``sqrt(IVEC_LOOP_WEIGHT)`` in per step
        (dft.cpp:296-300) and the reduction ``real(extra_weight * F * conj(F))``
        (stress.cpp:90-99) therefore carries the weight EXACTLY ONCE. This
        accumulator stores the unweighted sum, so applying the full ``_weights``
        to ``|F|^2`` here is algebraically identical — while ``packed`` applies it
        LINEARLY to the transform, which would square it and give a force off by a
        factor of the cell weight on every partially covered site.

        ``abs`` because a MIRROR-FOLDED axis puts ``S.phase_shift(c, sn) = ±1``
        into ``_weights`` (see the class docstring): that sign belongs to the
        transform, and under ``|F|^2`` it squares away. Taking it unsigned here
        keeps the two facts in one place instead of letting a negative weight
        subtract a positive-definite quantity. (No corpus row exercises a folded
        force region — :func:`~.from_meep._migrate_force` refuses one, because
        MEEP additionally rewrites the region list through ``symmetry::reduce``,
        vec.cpp:1416-1470 — so this is the analytic half of a claim that is not
        yet measured.)

        The product and the sum run in FLOAT64 because MEEP's do: ``stress_sum``
        promotes each stored ``complex<realnum>`` sample to ``complex<double>``
        explicitly before multiplying (stress.cpp:95-97), unlike
        ``dft_energy::electric``, which multiplies in ``realnum`` and promotes only
        the accumulator (dft.cpp:686). Reducing in float32 here instead was measured
        at 3.82e-08 relative against MEEP where the promoted reduction gives 7.4e-09
        — still inside the corpus band, but it spends most of test_force's own
        ``places=7`` margin (5e-08) on arithmetic MEEP does not do.
        """
        xp = self.grid.xp
        weights = xp.asarray(np.abs(self._weights))  # float64, MEEP's own reduction width.
        block = self._dft[int(index)]
        real = block.real.astype(xp.float64)
        imaginary = block.imag.astype(xp.float64)
        return float(xp.sum(weights * (real * real + imaginary * imaginary)))

    def packed(self, extra_scale=1.0, box=None, flip=()) -> np.ndarray:
        """The accumulated data in MEEP's stored layout: idx-major, frequency-minor.

        That order is `dft[Nomega * idx_dft + i]` (`update_dft`, dft.cpp:265-307),
        with `idx_dft` counting sites in `LOOP_OVER_IVECS` order (vec.hpp:151-168)
        — axis 0 outermost to axis 2 innermost, so the C-order ravel of this
        accumulator's (x, y, z) block after the frequency axis is moved last.

        `extra_scale` carries the near2far `stored_weight` — the tangential-pair sign
        times the region's own weight — which MEEP bakes into `scale` per step and
        which is likewise time-independent.

        `box`, when given, is ``((x0, x1), (y0, y1), (z0, z1))`` of INCLUSIVE
        accumulator-relative indices, and emits only that sub-box. MEEP splits a
        grid into chunks (27 structure chunks for a cell with a PML, where an
        unsplit cell has 1), and ``get_dft_data`` returns the CONCATENATION of
        every chunk the region touches, not one contiguous box. Emitting one box
        is right only when the grid is unsplit — where it matches CPU MEEP element
        for element at 9.6e-08 — and wrong, while still having the correct total
        size, as soon as a PML splits it.

        `flip` lists axes to emit in DESCENDING index order. A chunk MEEP created
        through a symmetry image (sn > 0) loops its STORED sites ascending, which
        is the requested span descending on each flipped axis; this accumulator
        stores the requested span, so the flip recovers MEEP's own storage order.
        The VALUES need no extra sign: the reflected chunk's dft is
        ``sum_t phase * S.phase_shift(c, sn) * raw`` (dft.cpp:169-171), and the
        parity folded into this accumulator's weights is the same factor.
        """
        host = to_numpy(self._dft)  # (nf, cx, cy, cz)
        weighted = host * self._weights[None, :, :, :] * complex(extra_scale)
        if box is not None:
            weighted = weighted[
                :,
                box[0][0]: box[0][1] + 1,
                box[1][0]: box[1][1] + 1,
                box[2][0]: box[2][1] + 1,
            ]
        for axis in flip:
            weighted = np.flip(weighted, axis=1 + int(axis))
        return np.moveaxis(weighted, 0, -1).reshape(-1).astype(np.complex128)

    def reset(self) -> None:
        self._dft = self.grid.xp.zeros_like(self._dft)
        self._update_calls = 0


@dataclass
class DFTMonitor:
    """Multi-frequency DFT accumulator over a rectangular grid region.

    Translation of MEEP's `dft_chunk` (dft.cpp lines 200-400), including explicit
    positive-integer decimation, with the accumulated field interpolated to cell
    centres. Every monitored frequency is accumulated in one pass over each
    sampled timestep.

    Attributes:
        grid: Computational grid; supplies the array module and geometry.
        frequency: Monitor frequency (or frequencies) in MEEP natural units
            (1/wavelength). After construction this holds the first monitored
            frequency — the one `get_dft`/`get_intensity` return by default.
        region: (x0, x1, y0, y1, z0, z1) half-open index region; defaults to the
            whole grid.
        components: Field components to accumulate.
        frequencies: The monitored frequencies; the alternative spelling of
            `frequency`, and after construction always the full tuple. The
            accumulators' leading axis runs over these, in this order.
        periodic: Which axes wrap, as (x, y, z). A region whose Yee-to-centre
            average needs the plane past the last stored cell takes it from the
            neighbouring lattice image, with its Bloch factor; a non-wrapping
            axis terminates on a wall MEEP holds at exactly zero, so the average
            there consumes that zero (see `_sliced_component` for the measured
            2.02x flux defect that dropping the average used to be). Defaults to
            no wrapping axis; `FdtdDriver.add_dft_monitor` passes the run's
            actual boundaries.
        decimation_factor: Accumulate every nth global timestep with an ``n*dt``
            quadrature weight, matching MEEP's explicit decimation setting.
    """

    grid: "Grid"
    frequency: float | Sequence[float] | None = None
    region: tuple[int, int, int, int, int, int] | None = None
    components: tuple[str, ...] = ("Ex", "Ey", "Ez")
    frequencies: float | Sequence[float] | None = None
    periodic: tuple[bool, bool, bool] = (False, False, False)
    decimation_factor: int = 1

    _omegas: Any = field(default=None, repr=False, init=False)
    _scale: float = field(default=0.0, repr=False, init=False)
    _update_calls: int = field(default=0, repr=False, init=False)
    _volume: tuple | None = field(default=None, repr=False, init=False)
    _wrapped: tuple[bool, bool, bool] = field(default=(False, False, False), repr=False, init=False)
    _gather: tuple | None = field(default=None, repr=False, init=False)
    _dft: dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):  # Validate frequencies and components, then allocate the accumulators.
        if not self.components:
            raise ValueError("DFTMonitor requires at least one field component")
        for component in self.components:
            yee_shifts(component)
        self.frequencies = normalize_frequencies(self.frequency, self.frequencies)
        self.frequency = self.frequencies[0]  # The default freq_index=0 accessor's frequency.
        self._omegas = 2 * np.pi * np.asarray(self.frequencies, dtype=np.float64)
        self.decimation_factor = _positive_decimation(self.decimation_factor)
        # MEEP dft.cpp add_dft: dt_factor = dt/sqrt(2*pi) * decimation_factor.
        self._scale = (
            self.grid.dt * self.decimation_factor / math.sqrt(2 * math.pi)
        )
        if len(tuple(self.periodic)) != 3:
            raise ValueError(f"DFTMonitor periodic must be (x, y, z) flags, got {self.periodic!r}.")
        if self.region is None:
            self.region = (0, self.grid.nx, 0, self.grid.ny, 0, self.grid.nz)
        self._reinit_accumulators()

    def _configure_decimation_factor(self, factor: int) -> None:
        """Install a driver-resolved automatic factor before accumulation starts.

        A monitor constructed directly has no source list, so its public constructor
        deliberately accepts only an explicit positive factor. ``FdtdDriver`` owns
        the higher-level MEEP-compatible ``decimation_factor=0`` policy and calls
        this method after considering every source and monitor frequency.
        """
        if self._update_calls:
            raise RuntimeError(
                "Cannot change a DFT monitor's decimation factor after accumulation has started."
            )
        self.decimation_factor = _positive_decimation(factor)
        self._scale = (
            self.grid.dt * self.decimation_factor / math.sqrt(2 * math.pi)
        )

    @property
    def num_frequencies(self) -> int:  # Length of the accumulators' leading frequency axis.
        return len(self.frequencies)

    def set_region_from_pml(self, pml_thickness: float):
        """Shrink the region to the interior of a uniform PML.

        The thickness is a cell count and need not be whole (``mp.PML(0.5)`` at
        resolution 71 is 35.5 cells). A face is trimmed by
        :func:`~.pml.absorbing_cells` — the whole cells it actually grades, which is
        the thickness itself whenever that is whole. Truncating with ``int(35.5)``
        instead leaves the second-deepest graded cell inside what this region calls
        the PML interior, and the spectrum that comes back is smooth, plausible and
        low; :meth:`Near2FarMonitor._require_clear_of_pml` records 7.3% and 44% of
        radiated power lost to the same class of mistake.

        A *mirrored* axis is trimmed at its upper face only: its lower face is the
        symmetry plane, which carries no absorber (the same rule
        `FdtdDriver.setup_pml` validates its thickness against). Trimming it there
        cost more than the cells it dropped. The stored quadrant's cell 0 is the one
        that straddles the mirror plane, and `get_dft_full` unfolds about exactly
        that cell — so starting the region `thickness` cells in made the unfolding
        mirror about a plane `thickness` cells off the symmetry plane and hand back
        a full-domain field that looked like one. Measured on a 2x2x3 cell at
        resolution 10 with X and Y mirrors and a 4-cell PML: the default monitor
        registered x cells 4..7 of the 11 stored, and `get_dft_full` returned a
        4-cell "full domain" spanning x = -0.55..0.55 of a 2.0-wide cell instead of
        the 12 interior cells it should have. Nothing about the result said so.
        """
        if not isinstance(pml_thickness, (int, float, np.integer, np.floating)) or pml_thickness < 0:
            raise ValueError(f"PML thickness must be a non-negative cell count, got {pml_thickness!r}.")
        thickness = absorbing_cells(pml_thickness)
        self._volume = None
        self._gather = None
        self._wrapped = (False, False, False)
        low = tuple(
            0 if _axis_is_mirrored(self.grid, axis) else thickness for axis in range(3)
        )
        self.region = (
            low[0], self.grid.nx - thickness,
            low[1], self.grid.ny - thickness,
            low[2], self.grid.nz - thickness,
        )
        _require_populated_region(
            self.grid, self.region,
            f"A DFT monitor defaulting to the interior of a {pml_thickness:g}-cell PML",
        )
        self._reinit_accumulators()

    def set_region_from_faces(self, faces):
        """Shrink the region to the interior of a per-face PML, face by face.

        The per-face counterpart of :meth:`set_region_from_pml`, for a layer whose
        six faces do not share one thickness (``{"z": 8}``, ``{"z": (8, 4)}``). Each
        face is trimmed by ITS OWN thickness, so an axis carrying no absorber keeps
        every one of its cells: putting the scalar entry point's single thickness on
        a periodic x axis would discard live interior, and on a cell only a grating
        period wide it would leave no region at all.

        A mirror-folded axis may not name its lower face. Cell 0 of the stored
        quadrant straddles the mirror plane, which carries no absorber, and it is the
        cell :meth:`get_dft_full` unfolds about — trimming there returns a
        "full domain" mirrored about the wrong plane, which is exactly the defect
        :meth:`set_region_from_pml` documents. :class:`~.pml.PML` already zeroes that
        face, so this refusal never fires on a layer built through it; it exists so a
        hand-made table cannot reintroduce the defect in silence.

        Args:
            faces: ``((x_low, x_high), (y_low, y_high), (z_low, z_high))`` absorber
                thickness in cells — :attr:`~.pml.PML.thickness_by_face`. Whole or
                fractional; each face is trimmed by :func:`~.pml.absorbing_cells` of
                its own count, for the reason :meth:`set_region_from_pml` records.
        """
        if len(tuple(faces)) != 3 or any(len(tuple(axis_faces)) != 2 for axis_faces in faces):
            raise ValueError(
                f"A per-face PML table must be ((x_low, x_high), (y_low, y_high), "
                f"(z_low, z_high)) cell counts, got {faces!r}."
            )
        if any(side < 0 for axis_faces in faces for side in axis_faces):
            raise ValueError(f"PML face thickness must be non-negative cell counts, got {faces!r}.")
        table = tuple(tuple(absorbing_cells(side) for side in axis_faces) for axis_faces in faces)
        for axis in range(3):
            # Triggered on the REQUESTED thickness, not the whole cells it grades: a
            # named low face on a folded axis is the mistake, whatever its depth.
            if _axis_is_mirrored(self.grid, axis) and faces[axis][0] > 0:
                raise ValueError(
                    f"A DFT monitor cannot take the interior of a PML {faces[axis][0]:g} cells deep "
                    f"at the {AXIS_NAMES[axis]} low face: that axis is mirror-folded, so cell 0 "
                    f"straddles the mirror plane get_dft_full unfolds about. Trimming there "
                    f"returns a full domain mirrored about the wrong plane."
                )
        self._volume = None
        self._gather = None
        self._wrapped = (False, False, False)
        counts = (self.grid.nx, self.grid.ny, self.grid.nz)
        self.region = tuple(
            value
            for axis in range(3)
            for value in (table[axis][0], counts[axis] - table[axis][1])
        )
        _require_populated_region(
            self.grid, self.region,
            f"A DFT monitor defaulting to the interior of a PML with faces {table}",
        )
        self._reinit_accumulators()

    def set_region_from_volume(self, center: tuple, size: tuple, periodic=None):
        """Register the monitor on the cells MEEP's axis ladder gives this volume.

        The registration itself lives in `_register_volume`, which is also what a
        flux plane is built from, so the two monitor kinds cannot drift apart: the
        same volume yields the same cells whichever one is asked for it, and both
        place where MEEP's `loop_in_chunks` places. See that function for the
        centre-aligned-face and periodic-continuation defects this replaced, with
        their measured before/after numbers.

        Args:
            center: (x, y, z) centre of the requested volume, in MEEP coordinates.
            size: (sx, sy, sz) extent; 0 on an axis is a zero-thickness request.
            periodic: Which axes may be continued into the neighbouring lattice
                image, overriding the monitor's own `periodic` for this
                registration. The driver resolves its regions when stepping starts
                rather than when the monitor is added, so it has a fresher answer
                than the one stored at construction — a PML installed in between
                turns a wrapping face into an absorbing one.
        """
        wrapping = self.periodic if periodic is None else periodic
        if len(tuple(wrapping)) != 3:
            raise ValueError(f"DFT monitor periodic must be (x, y, z) flags, got {periodic!r}.")
        self.periodic = tuple(bool(flag) for flag in wrapping)
        self.region, self._gather, self._wrapped = _register_volume(
            self.grid, center, size, self.periodic,
            f"DFT region centred at {tuple(center)} with size {tuple(size)}",
        )
        self._volume = (tuple(float(value) for value in center),
                        tuple(float(value) for value in size))
        self._reinit_accumulators()

    def _reinit_accumulators(self):  # Allocate zeroed (nf, nx, ny, nz) complex64 accumulators.
        xp = self.grid.xp
        x0, x1, y0, y1, z0, z1 = self.region
        shape = (x1 - x0, y1 - y0, z1 - z0)
        if min(shape) < 1:
            raise ValueError(
                f"DFT monitor region {self.region} spans {shape} cells of a {self.grid.shape} "
                f"grid. An accumulator with no cells reduces to zeros on every accessor, which "
                f"read as a real measurement rather than as a monitor that registered nothing."
            )
        _check_accumulator_budget(
            shape, self.num_frequencies, len(self.components), f"DFT monitor over region {self.region}"
        )
        for component in self.components:
            self._dft[component] = xp.zeros((self.num_frequencies,) + shape, dtype=xp.complex64)

    def update(
        self,
        fields: "Fields",
        time: float,
        current_step: int | None = None,
    ):
        """Accumulate one timestep into every component's DFT, for every frequency.

        MEEP source: dft.cpp `update_dft` lines 265-306, driven from
        `fields_chunk::update_dfts` with the post-increment time. E and D are
        sampled at `time`; H and B are sampled half a step earlier, at
        `time - 0.5*dt`, matching the leapfrog stagger. MEEP accumulates only
        when ``current_step % decimation_factor == 0`` and multiplies the DFT
        weight by that same factor. The field slice is read once per component
        and the `(nf, 1, 1, 1)` phase broadcasts over it, so the cost of a wide
        spectrum is the accumulate, not the field access.

        Args:
            fields: Field container exposing `get_component`.
            time: Current simulation time (MEEP `time()`).
            current_step: Global post-increment step index. When omitted, direct
                monitor use counts calls starting at one; the driver always
                supplies its global step count.
        """
        if current_step is None:
            self._update_calls += 1
            step = self._update_calls
        else:
            step = int(current_step)
            if step != current_step or step < 1:
                raise ValueError(
                    f"current_step must be a positive whole number, got {current_step!r}."
                )
            self._update_calls = step
        if step % self.decimation_factor:
            return
        phase_electric, phase_magnetic = _phase_factors(self.grid, self._omegas, self._scale, time)
        for component in self.components:
            centered = self._sample(fields, component)
            phase = phase_electric if _is_electric(component) else phase_magnetic
            self._dft[component] += phase * centered

    def _sample(self, fields, component):  # Cell-centred field over the region, sliced or gathered.
        """A region inside the cell is sliced; one that crosses a wrapping face is gathered.

        The contiguous path — and every number ever measured against it — is
        untouched by the wrap support. An index region (the whole-grid default, or
        the PML interior) has no ladder and so never gathers; its far-face Yee
        average still comes from `_sliced_component`'s own wrap.
        """
        if self._gather is None:
            return _sliced_component(fields, self.grid, component, self.region, self.periodic)
        return _gathered_component(fields, self.grid, component, self._gather, self.region)

    def _require_monitored(self, component: str):  # Accumulator stack for a component, loudly.
        if component not in self._dft:
            raise ValueError(
                f"Component '{component}' is not monitored. "
                f"Monitored components: {', '.join(self.components)}"
            )
        return self._dft[component]

    def get_dft(self, component: str, freq_index: int = 0):
        """Accumulated DFT of one component at one monitored frequency.

        Returns the `(nx, ny, nz)` slice — a view into the accumulator, so writes
        through it reach the monitor's state, as they did before the frequency
        axis existed.
        """
        stack = self._require_monitored(component)
        return stack[_frequency_index(self.frequencies, freq_index)]

    def get_dft_spectrum(self, component: str):  # Full (nf, nx, ny, nz) accumulator for one component.
        return self._require_monitored(component)

    def site_doubled(self, axis: int, index: int) -> int:
        """Doubled coordinate of a GRID index on this axis, in MEEP's integer lattice.

        The counterpart of :meth:`YeeRegionDFT.site_doubled` for a CENTRED monitor:
        every sample sits at a cell centre, ``origin_doubled + 2*index + 1``, whatever
        component it holds. That ``+ 1`` in place of the component's own Yee shift is
        the whole difference between the two, and it is why a read-time collapse (the
        interpolation onto a zero-thickness plane, :func:`~.from_meep.collapse_dft_region`)
        cannot use one rule for both.

        ``index`` is a grid index, the same coordinate ``region`` is expressed in, so
        the first stored sample is ``site_doubled(axis, region[2*axis])``. On a
        mirror-folded axis that index runs NEGATIVE below the plane — the reflected
        gather keeps the requested span in requested-space indices — and the formula
        holds there unchanged, because ``origin_doubled`` is the halved axis's own
        origin of -2 and cell 0 straddles the plane.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"Axis must be 0 (X), 1 (Y) or 2 (Z); got {axis!r}")
        return _axis_origin_doubled(self.grid, axis) + 2 * int(index) + 1

    def _require_unfoldable(self):
        """Refuse to unfold a region that does not reach the mirror plane.

        `_reconstruct_from_quadrant` mirrors about the region's own cell 0, which is
        only the symmetry plane when the region starts at stored cell 0 — the cell
        `halve()` places straddling it. A monitor that sits entirely in the positive
        half unfolds about its own near face instead, and the result is not merely
        displaced: on a 2x2x3 cell at resolution 10 with X and Y mirrors, a region on
        x cells 5..9 (x = 0.45..0.75) came back as six cells whose coordinates ran
        [-0.75, -0.65, 0.45, 0.55, 0.65, 0.75] — two of the four mirrored cells
        missing, a gap in the middle of a supposedly uniform axis, and one cell
        written twice so it carried the positive value under a mirrored label.

        A region that misses the plane has no rectangular full-domain image at all
        (its reflection and itself are two disjoint blocks), so there is nothing to
        return and this raises rather than inventing one.
        """
        if not self.grid.has_symmetry():
            return
        for axis in range(3):
            if not _axis_is_mirrored(self.grid, axis):
                continue
            if self.region[2 * axis] < 0:
                raise ValueError(
                    f"This monitor's region already reaches {AXIS_NAMES[axis]} cell "
                    f"{self.region[2 * axis]}, below the {AXIS_NAMES[axis].upper()} mirror "
                    f"plane — it was registered through the reflected gather and holds the "
                    f"requested span in full-domain coordinates. There is no quadrant here "
                    f"to unfold; read get_dft()/get_intensity() directly."
                )
            if self.region[2 * axis] != 0:
                raise ValueError(
                    f"This monitor's region starts at {AXIS_NAMES[axis]} cell "
                    f"{self.region[2 * axis]} of the stored quadrant, so it does not include the "
                    f"cell straddling the {AXIS_NAMES[axis].upper()} mirror plane and has no "
                    f"rectangular full-domain image: its reflection is a separate block. Read "
                    f"get_dft()/get_intensity() on the quadrant, or widen the region to the "
                    f"mirror plane."
                )

    def _component_parities(self, component: str):
        """Parity of one component about each axis's plane — `None` where there is none.

        The ONE place this monitor decides a sign. Every entry comes from
        `fields.mirror_parity`, MEEP's `symmetry::phase_shift`, evaluated at the
        plane's OWN declared phase (`Grid.mirror_phase`) rather than at MEEP's
        default +1. Reading the even X/Y table instead leaves every even run
        correct and mirrors an odd one with the wrong sign — a full, plausible
        full-domain array that is the negative of the truth on half its cells.
        """
        from .fields import mirror_parity  # Deferred: keeps module import order free.

        parities = []
        for axis in range(3):
            phase = _axis_mirror_phase(self.grid, axis)
            parities.append(None if phase is None else mirror_parity(component, axis, phase))
        return tuple(parities)

    def get_dft_full(self, component: str, freq_index: int = 0):
        """Return the DFT unfolded to the full domain when mirror symmetry is active.

        The stored quadrant is mirrored into the remaining quadrants with the
        per-component mirror parities from `fields.mirror_parity` (MEEP
        symmetry.cpp `phase_shift`): `f(S_d x) = parity_d * f(x)` for each folded
        axis d, so a doubly mirrored corner carries the product of the two.

        Any of X, Y and Z may be folded, at either phase; the axes are unfolded
        one at a time (`_reconstruct_from_quadrant`), so one rule serves one, two
        or three planes.

        A region that does not reach the mirror plane is refused rather than
        unfolded about its own near face (`_require_unfoldable`).
        """
        quadrant = self.get_dft(component, freq_index)
        if not self.grid.has_symmetry():
            return quadrant
        self._require_unfoldable()
        return self._reconstruct_from_quadrant(quadrant, self._component_parities(component))

    def _mirrored_cells(self, axis, quadrant_count):
        """Cells the mirrored (negative) side contributes when a quadrant unfolds.

        ``min(quadrant_count - 1, grid n_full // 2)``. Every stored cell except
        the straddling cell 0 has at most one lower image, and the grid's full
        window keeps only ``n_full // 2`` cells below the plane. At an even full
        count the clip never binds (a whole-axis quadrant has exactly
        ``n_full//2`` non-straddling cells); at an odd one the top stored cell —
        centre exactly at +L/2 — has an image half a cell BELOW MEEP's shifted
        window ``[-(N-1)dx/2, (N+1)dx/2]`` and is clipped.

        The full count comes from the GRID because the quadrant shape alone
        cannot supply it: parity is genuinely lost at ``halve()`` — a quadrant of
        ``m`` cells could have come from ``n_full = 2m - 2`` or ``2m - 3`` — so
        the old inverse ``n_full = 2 * (m - 1)`` was right only at even counts.
        A region narrower than the axis is unaffected either way: its top stored
        cell sits strictly inside the window and the ``m - 1`` bound binds first.
        """
        full = (self.grid.nx_full, self.grid.ny_full, self.grid.nz_full)[axis]
        return min(quadrant_count - 1, full // 2)

    def _full_shape(self, quadrant_shape):
        """Full-domain shape a quadrant of this shape unfolds to.

        Per mirrored axis: the quadrant's ``count - 1`` cells at or above the
        plane (the straddling cell counts as the plane's own) plus the mirrored
        cells of :meth:`_mirrored_cells` — ``2 * (count - 1)`` at even full
        counts, one less where the odd-count clip binds. Applied per axis, Z
        included: every axis halves the same way.
        """
        return tuple(
            self._positive_cells(axis, count) + self._mirrored_cells(axis, count)
            if _axis_is_mirrored(self.grid, axis) else count
            for axis, count in enumerate(quadrant_shape)
        )

    def _positive_cells(self, axis, quadrant_count):
        """Quadrant cells at or above the plane that have a full-domain home.

        ``count - 1`` (the straddling cell counts as the plane's own) except on a
        folded PERIODIC axis, where the quadrant's last cell is the stored
        ``big_corner`` cell (`Grid.stored_cells` one past `owned_cells`, at
        either count parity): its centre sits past the window top, so it
        has no full-domain cell to land on and is dropped — the same clip
        `fields._reconstruct_full_domain` takes. The grid bound
        ``n_full - centre`` states that directly for a whole-axis quadrant and
        never binds on a narrower region.
        """
        full = (self.grid.nx_full, self.grid.ny_full, self.grid.nz_full)[axis]
        return min(quadrant_count - 1,
                   full - self._mirrored_cells(axis, quadrant_count))

    def _reconstruct_from_quadrant(self, quadrant, parities):
        """Mirror a positive-quadrant array into the full domain, one axis at a time.

        Quadrant cell 0 straddles the mirror plane at -0.5*dx, so on each folded
        axis the positive block starts at `c - 1` (with `c` the mirrored-cell
        count of `_mirrored_cells` — `n_full // 2` for a whole-axis region, both
        count parities) and the mirrored block covers `[0, c)` from
        `quadrant[1:c+1]` reversed; the stored cells past `c` — the top of the
        window and MEEP's `big_corner` cell above it — have no image inside
        MEEP's window and are left out). The two writes overlap in the straddling cell,
        where `parity * quadrant[1]` and `quadrant[0]` describe the same physical
        cell — identical for any field that actually obeys the symmetry.

        Unfolding axis by axis in X, Y, Z order is what lets one rule serve one,
        two or three planes: after unfolding X the array is full in X and still
        folded in Y, so unfolding Y multiplies the already-phased X half by
        `parity_y` and the doubly mirrored corner ends up carrying
        `parity_x * parity_y`. That reproduces the hand-written four-quadrant
        version BIT FOR BIT, overlapping writes included — the same statement
        `fields._reconstruct_full_domain` makes for the field arrays.

        Args:
            quadrant: Stored DFT block, halved on each folded axis.
            parities: Per-axis component parity, `None` on an unfolded axis.
        """
        xp = self.grid.xp
        result = quadrant
        for axis in range(3):
            parity = parities[axis]
            if parity is None:
                continue
            n_quadrant = result.shape[axis]
            centre = self._mirrored_cells(axis, n_quadrant)  # First positive-side cell.
            kept = 1 + self._positive_cells(axis, n_quadrant)  # Rows consumed, cell 0 included.
            n_full = (kept - 1) + centre
            shape = list(result.shape)
            shape[axis] = n_full
            full = xp.zeros(tuple(shape), dtype=result.dtype)
            full[_slab(axis, slice(centre - 1, centre - 1 + kept))] = (
                result[_slab(axis, slice(0, kept))]
            )
            full[_slab(axis, slice(0, centre))] = parity * xp.flip(
                result[_slab(axis, slice(1, centre + 1))], axis=axis
            )
            result = full
        return result

    def get_intensity(self, freq_index: int = 0):  # Sum |E|^2 over the monitored electric components.
        xp = self.grid.xp
        index = _frequency_index(self.frequencies, freq_index)
        x0, x1, y0, y1, z0, z1 = self.region
        intensity = xp.zeros((x1 - x0, y1 - y0, z1 - z0), dtype=xp.float32)
        for component in ELECTRIC_COMPONENTS:
            if component in self._dft:
                intensity += xp.abs(self._dft[component][index]) ** 2
        return intensity

    def get_intensity_full(self, freq_index: int = 0):  # Sum |E|^2 after unfolding to the full domain.
        xp = self.grid.xp
        index = _frequency_index(self.frequencies, freq_index)
        if not self.grid.has_symmetry():
            return self.get_intensity(index)
        intensity = None
        for component in ELECTRIC_COMPONENTS:
            if component in self._dft:
                magnitude_squared = xp.abs(self.get_dft_full(component, index)) ** 2
                intensity = magnitude_squared if intensity is None else intensity + magnitude_squared
        if intensity is None:
            # No electric components monitored: return the shape the populated
            # path would have produced, derived from the region quadrant. The
            # unfoldability check runs here too — the populated branch reaches it
            # through get_dft_full, and this one must not be the way a region that
            # misses the mirror plane gets a full-domain shape anyway.
            self._require_unfoldable()
            x0, x1, y0, y1, z0, z1 = self.region
            shape = self._full_shape((x1 - x0, y1 - y0, z1 - z0))
            return xp.zeros(shape, dtype=xp.float32)
        return intensity

    def get_coordinates(self):  # Host coordinate axes spanning the monitor region.
        """Cell-centre coordinates of the region, continued past a wrapping face.

        Built from the axis origin rather than sliced out of `grid.x`, because a
        region continued into the neighbouring lattice image holds indices outside
        the stored range and a slice there returns the wrong cells without saying so
        (`_axis_coordinates`).
        """
        x0, x1, y0, y1, z0, z1 = self.region
        return (
            _axis_coordinates(self.grid, 0, x0, x1),
            _axis_coordinates(self.grid, 1, y0, y1),
            _axis_coordinates(self.grid, 2, z0, z1),
        )

    def get_coordinates_full(self):
        """Host coordinate axes matching `get_dft_full` / `get_intensity_full`.

        The mirrored block of the reconstruction is `quadrant[1:c+1]` reversed
        (`c` from `_mirrored_cells`) and overwrites the straddling cell, so the
        unfolded axis holds the mirrored positions of `quadrant[2:c+1]` followed
        by the quadrant's own positions — one cell shorter per mirrored axis than
        naively mirroring the whole quadrant, which would hand callers more
        coordinates than data, and one shorter again where the odd-count clip
        drops the top stored cell's out-of-window image.

        Refused for a region that does not reach the mirror plane, for the same
        reason `get_dft_full` is: there is no such axis to describe.
        """
        axes = list(self.get_coordinates())
        if not self.grid.has_symmetry():
            return tuple(axes)
        self._require_unfoldable()
        for axis in range(3):  # Per axis, Z included: every plane mirrors its own coordinates.
            if _axis_is_mirrored(self.grid, axis):
                centre = self._mirrored_cells(axis, axes[axis].size)
                kept = 1 + self._positive_cells(axis, axes[axis].size)  # Drop the plane cell.
                axes[axis] = np.concatenate(
                    [-np.flip(axes[axis][2:centre + 1]), axes[axis][:kept]])
        return tuple(axes)

    def reset(self):  # Zero every accumulator without reallocating.
        for accumulator in self._dft.values():
            accumulator.fill(0)
        self._update_calls = 0

    def __repr__(self) -> str:
        x0, x1, y0, y1, z0, z1 = self.region
        shape = (x1 - x0, y1 - y0, z1 - z0)
        return (
            f"DFTMonitor(freq={_frequency_summary(self.frequencies)}, shape={shape}, "
            f"components={self.components}, decimation={self.decimation_factor})"
        )


@dataclass(frozen=True)
class MeepChunkWeight:
    """One MEEP ``dft_chunk``'s own integration weight, read off the live chunk.

    MEEP does not store a DFT sample raw: ``dft_chunk::update_dft`` folds
    ``w = IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV0 + dV1*loop_i2)`` into every stored
    element when ``include_dV_and_interp_weights`` is set (dft.cpp:276-281), and
    ``process_dft_component`` divides it straight back out when it reads
    (dft.cpp:1003-1006). So the number in the array is the field TIMES that weight,
    and a packer that supplies a different weight supplies a different field.

    This engine has its own model of the same measure — ``_axis_ladder``'s bracket
    of the requested bounds — and the two agree on the region MEEP was ASKED for.
    They do not agree once ``symmetry::reduce`` has halved the region
    (vec.cpp:1437-1456), because the halved region has NEW edges and therefore new
    edge weights. Measured on a ``Mirror(Y)`` waveguide cell: packing ``add_flux``'s
    reduced list with the engine's own weights differs from MEEP's stored E at
    exactly 4 of its 84 sites — the chunk straddling the mirror plane — by 1/0.875
    and 1/0.125, that chunk's own ``s0.y`` and ``s1.y``, for 3.203e-01 overall,
    while both FULL-plane lists on the same geometry agree at 0 sites. So the weight
    is not modelled here at all; it is transcribed from the chunk MEEP built.

    ``s0``/``s1``/``e0``/``e1`` are per-axis ``(x, y, z)`` triples of the
    like-named public ``vec`` members (src/meep.hpp:1196-1205), each read through
    ``in_direction``; a direction the run does not have reads back 1.0, which is
    the identity this product needs. ``include`` and ``sqrt_weights`` are the
    chunk's two flags, and ``dV0`` its cell measure.
    """

    s0: tuple
    s1: tuple
    e0: tuple
    e1: tuple
    dV0: float
    include: bool
    sqrt_weights: bool

    def grid_for(self, shape) -> np.ndarray:
        """The per-site weight for a block of ``shape`` ``(nx, ny, nz)`` sites.

        ``IVEC_LOOP_WEIGHT`` (src/meep/vec.hpp:379-383) is a product of one
        :func:`_ivec_loop_weight` factor per loop axis times ``dV``, and each axis's
        factor depends only on that axis's position and site count — so the block is
        the outer product of three vectors. The loop index runs along the STORED
        chunk ascending, which is the order :meth:`FluxMonitor.packed` has already
        flipped its block into by the time this is applied.
        """
        axes = []
        for axis in range(3):
            count = int(shape[axis])
            axes.append(np.array(
                [_ivec_loop_weight(index, count, self.s0[axis], self.s1[axis],
                                   self.e0[axis], self.e1[axis])
                 for index in range(count)], dtype=np.float64))
        weight = (axes[0][:, None, None] * axes[1][None, :, None]
                  * axes[2][None, None, :]) * float(self.dV0)
        return np.sqrt(weight) if self.sqrt_weights else weight


@dataclass(frozen=True)
class FluxPlaneDft:
    """One flux plane's accumulated E and H transforms, detached from the run.

    MEEP's counterpart is one region's worth of the flat arrays inside a
    ``FluxData`` (``python/simulation.py:3619-3627``); the split into a record per
    plane is this engine's, because :class:`FluxMonitor` keeps a named
    ``(nf, nx, ny, nz)`` block per component rather than MEEP's single
    chunk-ordered stream.

    The arrays are host copies taken at read time, so the monitor they came from
    may be closed, reset or stepped further without changing them — which is what
    makes the normalization idiom's first run disposable.
    """

    E: dict  # {"Ex"/"Ey"/"Ez": (nf, nx, ny, nz) complex host array}
    H: dict  # {"Hx"/"Hy"/"Hz": (nf, nx, ny, nz) complex host array}
    frequencies: tuple


@dataclass(frozen=True)
class FluxDftData:
    """What ``sim.get_flux_data(flux)`` returns, for a flux monitor of this engine.

    Opaque by design, exactly as MEEP declares its own ``FluxData`` to be
    ("should be considered opaque", ``python/simulation.py:3619-3624``): the only
    supported use is handing it back to :meth:`FluxMonitor.load_dft_data` or
    :meth:`FluxMonitor.load_minus_dft_data` on a monitor built the same way in a
    later run.

    ``planes`` has one entry per flux plane. A monitor built from a single
    ``mp.FluxRegion`` has exactly one; MEEP's multi-region flux box (rebuilt here
    as ``from_meep.FluxMigration``) has one per region, in the script's own region
    order. That order is this container's own convention — save and load both walk
    the same list, and :func:`_load_flux_planes` checks the pairing plane by plane
    — and it deliberately does not try to match MEEP's, whose single ``dft_flux``
    chunk list is built by prepending (dft.cpp:627-631, :128) and therefore runs
    backwards. Nothing here is ever handed to MEEP, so nothing has to.
    """

    planes: tuple


def _load_flux_planes(data: Any, planes: Sequence["FluxMonitor"], description: str) -> None:
    """Overwrite each plane's accumulators from ``data``, MEEP's ``load_flux_data``.

    MEEP's ``Simulation.load_flux_data`` (``python/simulation.py:3629-3638``) hands
    each side to ``_load_dft_data`` (``python/meep.i:497-511``), which REPLACES the
    stored transform element by element — it does not add to it — and aborts when
    the incoming length does not match the monitor's own. Both properties are
    reproduced here.

    The checks are stricter than MEEP's on purpose. MEEP compares one total element
    count per side; this compares the plane count, the component set, each block's
    shape and the frequency list, because this engine stores the transform as a
    named block per component rather than one flat stream, and a mismatched block
    would BROADCAST instead of failing — handing back a plausible normalized
    spectrum computed against the wrong plane. A stricter check can only turn a
    silent wrong number into a refusal.
    """
    if not isinstance(data, FluxDftData):
        raise TypeError(
            f"{description} takes the object a flux monitor's get_dft_data() returned "
            f"(meep_gpu.dft.FluxDftData), got {type(data).__name__}. MEEP's own FluxData "
            f"holds its chunk-ordered arrays and is not interchangeable with this one."
        )
    if len(data.planes) != len(planes):
        raise ValueError(
            f"{description} was given data for {len(data.planes)} flux plane(s) but this "
            f"monitor has {len(planes)}. Load from a monitor built with the same "
            f"FluxRegion list."
        )
    for index, (record, monitor) in enumerate(zip(data.planes, planes)):
        where = f"{description}, plane {index}"
        if len(record.frequencies) != monitor.num_frequencies:
            raise ValueError(
                f"{where}: the data holds {len(record.frequencies)} frequencies and this "
                f"monitor accumulates {monitor.num_frequencies}."
            )
        for saved, mine in zip(record.frequencies, monitor.frequencies):
            if not math.isclose(float(saved), float(mine), rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    f"{where}: the data was accumulated at frequencies {record.frequencies} "
                    f"and this monitor accumulates {monitor.frequencies}. Subtracting one "
                    f"from the other would pair different frequencies."
                )
        sides = (
            ("E", ELECTRIC_COMPONENTS, record.E, monitor._dft_E),
            ("H", MAGNETIC_COMPONENTS, record.H, monitor._dft_H),
        )
        for side, names, block, target in sides:
            if set(block) != set(names):
                raise ValueError(
                    f"{where}: the data's {side} components are {sorted(block)}, expected "
                    f"{sorted(names)}."
                )
            for name in names:
                incoming = np.asarray(block[name])
                if incoming.shape != tuple(target[name].shape):
                    raise ValueError(
                        f"{where}: {name} in the data has shape {incoming.shape} and this "
                        f"monitor's plane is {tuple(target[name].shape)}. The two runs must "
                        f"agree on the cell and on the flux region (MEEP requires the same, "
                        f"python/simulation.py:3629-3636)."
                    )
        # Written only once every block of this plane has been validated, so a refusal
        # leaves the monitor untouched rather than half-loaded.
        xp = monitor.grid.xp
        for _side, names, block, target in sides:
            for name in names:
                target[name][...] = xp.asarray(block[name], dtype=target[name].dtype)


def _volume_periodic_axes(grid, periodic, kind: str) -> tuple[bool, bool, bool]:
    """Which axes a sampling region may be continued across; the caller's answer or the rule.

    A mirrored axis never wraps: its lower face is the symmetry plane and its upper
    face is the physical edge of the folded half, and MEEP would reach those cells
    through the symmetry transform rather than a lattice shift. Everything else
    wraps, absorber or not — this fallback and the flags
    :meth:`FdtdDriver.add_flux_monitor` passes are the same rule
    (:func:`~.driver._periodic_axes`), so a monitor built without the driver's flags
    samples the boundaries the stepping actually applied.

    One home for the rule because three volume monitors now need it — flux, energy
    and the off-diagonal force branch — and a monitor that resolved it differently
    would register on a different set of cells than the plane beside it.
    """
    if periodic is None:
        return _wrapping_axes(grid)
    resolved = tuple(bool(flag) for flag in periodic)
    if len(resolved) != 3:
        raise ValueError(f"{kind} periodic must be (x, y, z) flags, got {periodic!r}.")
    return resolved


def _volume_boundary_weights(grid, center, size, region):
    """Fractional-cell integration weights for a registered volume, shaped like it.

    MEEP folds these into the chunks built with ``include_dV_and_interp_weights``,
    where each cell contributes in proportion to its overlap with the requested
    volume. All three axes get the ``_axis_ladder`` vector — partial weights on the
    two cells at each end, 1.0 in the interior for an axis with extent; ``w0`` and
    ``w1`` summing to 1.0 for a flat one, which is a linear interpolation onto the
    requested coordinate rather than an integration.

    The FLAT axis is weighted through exactly that path, which is what registers a
    plane falling between two sample planes. Weighting only the axes with extent —
    and so summing the flat one at weight 1.0 — was correct only while it was one
    cell wide, and is half of the snapping defect :meth:`FluxMonitor._setup_region`
    describes (measured 7.2e-02 against CPU MEEP, against a 5% bar, with the same
    number returned for every request across a full cell).

    Because each extended axis's weight vector sums to ``extent * resolution`` and a
    flat one's to 1, the weighted sum times ``dx**extent_axes`` equals the requested
    measure exactly, for any position.

    Shared by :class:`FluxMonitor` and :class:`EnergyMonitor` so a region has ONE
    set of weights whichever reduction consumes it.
    """
    x0, x1, y0, y1, z0, z1 = region
    starts = (x0, y0, z0)
    counts = (x1 - x0, y1 - y0, z1 - z0)
    axis_weights = []
    for axis in range(3):
        extent = size[axis]
        where_min = center[axis] - extent / 2
        where_max = center[axis] + extent / 2
        axis_weights.append(_fractional_cell_weights(
            grid, axis, where_min, where_max, starts[axis], counts[axis]
        ))
    if getattr(grid, "cylindrical", False):
        # MEEP loop_in_chunks.cpp:505-515 — in Dcyl the integration measure is
        # dV = dV0 + dV1*iloopR with dV1 = dV0_cartesian * 2*pi*inva and dV0
        # picking up 2*pi*r at the loop's starting centered-grid point: every
        # sample is weighted by the circumference of its own ring, linearly in
        # the r index. The Cartesian dV0 factor is `_measure`; the ring factor
        # multiplies the r ladder here, at each region row's centre coordinate
        # (row j is stored cell x0+j, centred at (x0+j+0.5)*dx — the axis row
        # sits at r = dx/2, never 0, so no sample is zero-weighted). Without
        # this a z-normal plane integrates plain dr and undercounts by the
        # mean 2*pi*r of its extent (measured 0.47-0.49x MEEP on an r in
        # [0, 1.2] plane). A flat-in-r monitor gets the same factor on its
        # two interpolation cells, whose w0/w1 sum then reproduces 2*pi*r at
        # the requested radius exactly (2*pi*r is linear in r).
        xp = grid.xp
        rows = xp.arange(counts[0], dtype=xp.float32)
        ring = 2.0 * np.pi * (starts[0] + rows + 0.5) * grid.dx
        axis_weights[0] = axis_weights[0] * ring.astype(xp.float32)
    return (
        axis_weights[0][:, None, None]
        * axis_weights[1][None, :, None]
        * axis_weights[2][None, None, :]
    )


@dataclass
class FluxMonitor:
    """Multi-frequency Poynting-flux monitor over a planar region.

    Accumulates the DFT of all six field components on a plane, at every
    monitored frequency, and reduces them to the normal component of
    `Re(E x conj(H))`, integrated over the surface. A transmission spectrum is
    one monitor: `get_flux_spectrum()` returns the power at each frequency.

    MEEP source: dft.cpp `dft_flux`; meep.hpp `dft_flux` definition. MEEP builds
    the E-side `dft_chunk`s with `include_dV_and_interp_weights = true` and the
    H-side without, so the surface measure is applied exactly once — reproduced
    here by weighting only the E DFTs with `_compute_boundary_weights`.

    The plane is placed where it was asked for, not at the nearest sample plane:
    a requested coordinate that falls between two rows of cell centres is
    bracketed by both and the flux integrand is linearly interpolated onto it,
    which is MEEP's own zero-thickness rule (`_setup_region`). Nothing is snapped
    and nothing needs reporting back to the caller.

    Attributes:
        grid: Computational grid.
        frequency: Monitor frequency (or frequencies) in MEEP natural units.
            After construction this holds the first monitored frequency — the one
            `get_flux` returns by default.
        center: (x, y, z) centre of the flux plane; any coordinate, on or off the
            grid's sample planes.
        size: (sx, sy, sz) plane size; the normal direction should be 0.
        direction: Normal direction 0=X, 1=Y, 2=Z; auto-detected when None.
        weight: Scalar multiplier applied to the integrated flux.
        frequencies: The monitored frequencies; the alternative spelling of
            `frequency`, and after construction always the full tuple.
        periodic: Which axes wrap, as (x, y, z). A plane whose sampling reaches
            past a wrapping face is continued into the neighbouring lattice image
            with its Bloch factor, exactly as MEEP's `loop_in_chunks` does; on a
            non-wrapping axis the request is clipped at the face. Defaults to
            every unmirrored axis, which is right for a run with no absorber;
            the driver overrides it because a PML terminates its faces instead.
    """

    grid: "Grid"
    frequency: float | Sequence[float] | None = None
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] | None = None
    direction: int | None = None
    weight: float = 1.0
    frequencies: float | Sequence[float] | None = None
    periodic: tuple[bool, bool, bool] | None = None
    decimation_factor: int = 1

    _omegas: Any = field(default=None, repr=False, init=False)
    _scale: float = field(default=0.0, repr=False, init=False)
    _update_calls: int = field(default=0, repr=False, init=False)
    _region: tuple[int, int, int, int, int, int] | None = field(
        default=None, repr=False, init=False
    )
    _weights: Any = field(default=None, repr=False, init=False)
    _measure: float = field(default=0.0, repr=False, init=False)
    _wrapped: tuple[bool, bool, bool] = field(default=(False, False, False), repr=False, init=False)
    _gather: tuple | None = field(default=None, repr=False, init=False)
    _dft_E: dict[str, Any] = field(default_factory=dict, repr=False, init=False)
    _dft_H: dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):  # Resolve the frequencies and surface normal, then build the region and weights.
        self.frequencies = normalize_frequencies(self.frequency, self.frequencies)
        self.frequency = self.frequencies[0]  # The default freq_index=0 accessor's frequency.
        self._omegas = 2 * np.pi * np.asarray(self.frequencies, dtype=np.float64)
        self.decimation_factor = _positive_decimation(self.decimation_factor)
        self._scale = (
            self.grid.dt * self.decimation_factor / math.sqrt(2 * math.pi)
        )
        for name, value in (("center", self.center), ("size", self.size)):
            if value is None:
                raise ValueError(f"FluxMonitor requires a plane {name}; got None.")
            if len(tuple(value)) != 3:
                raise ValueError(f"FluxMonitor {name} must have three components (x, y, z), got {value!r}.")
        # A negative extent would invert the weight ladder's bounds and hand back a
        # plausible flux for a plane nobody could describe.
        if any(not math.isfinite(extent) or extent < 0.0 for extent in self.size):
            raise ValueError(f"FluxMonitor size must be finite and non-negative on every axis, got {self.size!r}.")
        if any(not math.isfinite(coordinate) for coordinate in self.center):
            raise ValueError(f"FluxMonitor center must be finite on every axis, got {self.center!r}.")
        if self.direction is None:
            if self.size[0] == 0 or self.size[0] < self.grid.dx:
                self.direction = 0
            elif self.size[1] == 0 or self.size[1] < self.grid.dx:
                self.direction = 1
            else:
                self.direction = 2
        if self.direction not in (0, 1, 2):
            raise ValueError(
                f"Flux direction must be 0 (X), 1 (Y) or 2 (Z); got {self.direction!r}"
            )
        self._setup_region()

    def _configure_decimation_factor(self, factor: int) -> None:
        """Install a driver-resolved automatic factor before accumulation starts."""
        if self._update_calls:
            raise RuntimeError(
                "Cannot change a flux monitor's decimation factor after accumulation has started."
            )
        self.decimation_factor = _positive_decimation(factor)
        self._scale = (
            self.grid.dt * self.decimation_factor / math.sqrt(2 * math.pi)
        )

    @property
    def num_frequencies(self) -> int:  # Length of the accumulators' leading frequency axis.
        return len(self.frequencies)

    def _setup_region(self):
        """Map the flux plane to grid indices and precompute its surface weights.

        Every axis — the plane's normal as much as its two in-plane axes — takes
        its cells from `_axis_ladder`, i.e. from MEEP's own `vec2diel_floor` /
        `vec2diel_ceil` bracket of the requested bounds. An axis with extent
        therefore spans `extent*a + 2` cells as before, and a flat axis spans the
        one or two cells that bracket the requested coordinate: one when the
        coordinate lands on a sample plane, two when it falls between them.

        The two-cell case is the whole point. A zero-thickness axis is MEEP's
        case 4 (loop_in_chunks.cpp, "as (3), but a = b: interpolation, not
        integration"), whose weights `w0` and `w1` linearly interpolate the flux
        integrand onto the requested plane. This engine used to round that axis
        to a single index instead, which snapped a plane requested at a round
        multiple of `dx` — z = 1.0 at resolution 10, the coordinate a user is
        most likely to type — half a cell away, in silence: measured 7.2e-02
        against CPU MEEP, against a 5% bar, and the same number returned for
        every request across a full cell. Interpolating instead holds a swept
        cell and a half to 9.3e-07 at its worst position, from 7.2e-02
        (test_driver_integration.py's positional sweep).

        The normal axis is not special-cased anywhere: it is flat only in the
        sense that its requested extent is zero, and the ladder handles that
        through the same path as any other axis.

        An axis whose ladder — or the one extra plane a Yee average consumes on
        it — reaches past a *wrapping* face is continued into the neighbouring
        lattice image rather than clipped there, which is MEEP's lattice-shift
        loop (`_lattice_gather`). Clipping instead silently dropped whatever the
        far face carried: measured against CPU MEEP in a periodic cell, a plane on
        the last row of cell centres read 9.1e-02 wrong, one a fifth of a cell
        further out 3.1e-01, one on the face itself 6.0e-01, and a plane spanning
        the full periodic cross-section — the standard grating measurement —
        5.1e-02. All of them completed and returned a plausible power.
        """
        description = f"Flux plane centred at {tuple(self.center)} with size {tuple(self.size)}"
        self._region, self._gather, self._wrapped = _register_volume(
            self.grid, self.center, self.size, self._resolve_periodic_axes(), description,
        )
        # One weight vector per axis, from the region the ladder just produced. A
        # wrapped axis keeps every ladder cell, so its region start *is* the ladder
        # start and `_fractional_cell_weights` reproduces the ladder taper exactly;
        # a clipped one starts inside the ladder and the cells outside it get 0.
        # Deriving the weights from the region, in one place, is what stops a plane
        # landing on one set of cells and being weighted for another.
        self._weights = self._compute_boundary_weights()
        xp = self.grid.xp
        x0, x1, y0, y1, z0, z1 = self._region
        shape = (x1 - x0, y1 - y0, z1 - z0)
        # MEEP loop_in_chunks.cpp: dV0 picks up one factor of 1/a per direction the
        # monitor actually extends along, so a plane integrates over an area and a
        # flat-in-two-directions monitor over a line. Reading it off the request
        # rather than assuming dx^2 keeps a degenerate monitor comparable with MEEP
        # instead of off by a factor of the grid spacing.
        self._measure = self.grid.dx ** sum(1 for extent in self.size if extent > 0)
        _check_accumulator_budget(
            shape,
            self.num_frequencies,
            len(ELECTRIC_COMPONENTS) + len(MAGNETIC_COMPONENTS),
            f"Flux monitor on the plane centred at {self.center}",
        )
        for component in ELECTRIC_COMPONENTS:
            self._dft_E[component] = xp.zeros((self.num_frequencies,) + shape, dtype=xp.complex64)
        for component in MAGNETIC_COMPONENTS:
            self._dft_H[component] = xp.zeros((self.num_frequencies,) + shape, dtype=xp.complex64)

    def _resolve_periodic_axes(self) -> tuple[bool, bool, bool]:
        """Which axes a sampling region may be continued across.

        A mirrored axis never wraps: its lower face is the symmetry plane and its
        upper face is the physical edge of the folded half, and MEEP would reach
        those cells through the symmetry transform rather than a lattice shift.
        Everything else wraps, absorber or not — the fallback here and the answer
        :meth:`FdtdDriver.add_flux_monitor` passes are the same rule
        (:func:`~.driver._periodic_axes`), so a monitor built without the driver's
        flags samples the boundaries the stepping actually applied.
        """
        return _volume_periodic_axes(self.grid, self.periodic, "FluxMonitor")

    def _compute_boundary_weights(self):
        """Fractional-cell surface weights for the flux plane, shaped like the region.

        MEEP integrates a flux plane with `include_dV_and_interp_weights` on the
        E-side chunk, which folds in the `loop_in_chunks` boundary weights: each
        cell contributes in proportion to its overlap with the requested plane.
        All three axes get the `_axis_ladder` vector — partial weights on the two
        cells at each end, 1.0 in the interior for an axis with extent; `w0` and
        `w1` summing to 1.0 for a flat one, which is a linear interpolation onto
        the requested coordinate rather than an integration.

        The normal axis is weighted through exactly that path, which is what
        registers a plane that falls between two sample planes. Weighting only
        the in-plane axes — and so summing the normal axis with weight 1.0 —
        was correct only while the normal axis was one cell wide, and it is the
        other half of the snapping defect `_setup_region` describes.

        Because each in-plane weight vector sums to `extent * resolution` and the
        normal one to 1, the weighted sum times `dx^2` equals the requested plane
        area exactly, for any plane position. This replaces an earlier empirical
        `0.75 - 0.25*offset` edge weighting whose error depended on where the
        plane sat in the original implementation:
        fit only looked accurate for monitors whose field was negligible at the
        plane edges, and mis-scaled the area by up to ~15% for a plane carrying
        field out to its boundary.
        """
        return _volume_boundary_weights(self.grid, self.center, self.size, self._region)

    def update(
        self,
        fields: "Fields",
        time: float,
        current_step: int | None = None,
    ):
        """Accumulate one timestep of all six components on the flux plane.

        Same MEEP timing as `DFTMonitor.update`: E at `time`, H at
        `time - 0.5*dt`, both interpolated to cell centres first, and every
        monitored frequency accumulated in the same pass.

        A plane entirely inside the cell is sliced, exactly as before; one that
        crosses a wrapping face is gathered instead, so the contiguous path — and
        every number measured against it — is untouched by the wrap support.
        """
        if current_step is None:
            self._update_calls += 1
            step = self._update_calls
        else:
            step = int(current_step)
            if step != current_step or step < 1:
                raise ValueError(
                    f"current_step must be a positive whole number, got {current_step!r}."
                )
            self._update_calls = step
        if step % self.decimation_factor:
            return
        phase_electric, phase_magnetic = _phase_factors(self.grid, self._omegas, self._scale, time)
        for component in ELECTRIC_COMPONENTS:
            self._dft_E[component] += phase_electric * self._sample(fields, component)
        for component in MAGNETIC_COMPONENTS:
            self._dft_H[component] += phase_magnetic * self._sample(fields, component)

    def _sample(self, fields, component):  # Cell-centred field over the plane, sliced or gathered.
        if self._gather is None:
            return _sliced_component(fields, self.grid, component, self._region)
        return _gathered_component(fields, self.grid, component, self._gather, self._region)

    def get_flux(self, freq_index: int = 0) -> float:
        """Integrate the normal Poynting component over the plane at one frequency.

        `S = Re(E x conj(H))`, so
        `S_x = Re(Ey*conj(Hz) - Ez*conj(Hy))`,
        `S_y = Re(Ez*conj(Hx) - Ex*conj(Hz))`,
        `S_z = Re(Ex*conj(Hy) - Ey*conj(Hx))`.

        The surface weights are applied to the E DFTs only (MEEP applies the
        measure once, on the E-side chunk). The overall normalization is carried
        by the `dt/sqrt(2*pi)` DFT scale, so no extra factor of 1/2 appears here.

        Args:
            freq_index: Which monitored frequency to reduce; 0 is the first.

        Returns:
            Total flux at that frequency; positive means power flowing along +direction.
        """
        return self._integrate_flux(_frequency_index(self.frequencies, freq_index))

    def get_flux_spectrum(self):
        """Power through the plane at every monitored frequency.

        The reduction runs once per frequency because it is a read-time call at
        the end of a run, not the hot path — the per-timestep accumulation is
        what is vectorized over frequency — and reducing each frequency's own
        contiguous block keeps a one-frequency monitor's number identical to what
        the single-frequency monitor produced.

        Returns:
            Host float64 array of shape (nf,), aligned with `frequencies`.
        """
        return np.array(
            [self._integrate_flux(index) for index in range(self.num_frequencies)],
            dtype=np.float64,
        )

    def _integrate_flux(self, index: int) -> float:  # Weighted normal Poynting sum for one frequency.
        """MEEP returns the FULL-DOMAIN flux under symmetry, and so does this.

        ``loop_in_chunks`` visits each symmetry image of the monitor volume, so
        ``mp.get_fluxes`` reads the same number folded or not. A plane reaching
        below a mirror plane samples through `_register_volume`'s reflected gather:
        each below-plane cell holds ``mirror_parity(c)`` times its image's field, so
        the ``E x conj(H)`` product carries ``phase_shift(E) * phase_shift(H)`` —
        MEEP's derived-component ``phase_shift(Sx..Sz)`` (vec.cpp:1371-1379) —
        which flips the sign of the normal component exactly where MEEP flips it.
        Measured against MEEP's own folded runs (even and odd mirrors, even and
        odd stored counts): a plane wholly in the discarded half reports the
        SIGN-FLIPPED mirror of its image plane bit for bit, and matches the
        unfolded control at 5.0e-07..3.7e-06; a straddling plane at 1e-07. Before
        the gather existed this refused instead — the stored-half integral is 89%
        of the right answer for one fold, 79% for two, and no fixed factor
        recovers it.
        """
        xp = self.grid.xp
        dA = self._measure  # dx per axis the monitor extends along; dx^2 for a plane.
        Ex = self._dft_E["Ex"][index] * self._weights
        Ey = self._dft_E["Ey"][index] * self._weights
        Ez = self._dft_E["Ez"][index] * self._weights
        Hx, Hy, Hz = (self._dft_H[name][index] for name in MAGNETIC_COMPONENTS)
        if self.direction == 0:
            normal_poynting = xp.real(Ey * xp.conj(Hz) - Ez * xp.conj(Hy))
        elif self.direction == 1:
            normal_poynting = xp.real(Ez * xp.conj(Hx) - Ex * xp.conj(Hz))
        else:
            normal_poynting = xp.real(Ex * xp.conj(Hy) - Ey * xp.conj(Hx))
        return float(xp.sum(normal_poynting) * dA * self.weight)

    def first_index(self, axis: int) -> int:  # This plane's first stored grid index on an axis.
        return self._region[2 * axis]

    def stop_index(self, axis: int) -> int:  # One past this plane's last stored grid index.
        return self._region[2 * axis + 1]

    def site_doubled(self, axis: int, index: int) -> int:
        """Doubled coordinate of a GRID index on this axis, in MEEP's integer lattice.

        A flux plane samples CELL CENTRES for every one of its six components —
        ``add_dft_flux`` passes ``centered_grid`` (true by default) so
        ``loop_in_chunks`` runs on the ``Centered`` lattice whatever component the
        chunk carries (dft.cpp:627-632, loop_in_chunks.cpp:350-352) — so this is the
        same ``origin_doubled + 2*index + 1`` for all six, unlike
        :meth:`YeeRegionDFT.site_doubled`, whose lattice is the component's own.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"Axis must be 0 (X), 1 (Y) or 2 (Z); got {axis!r}")
        return _axis_origin_doubled(self.grid, axis) + 2 * int(index) + 1

    def index_for_doubled(self, axis: int, doubled: int) -> int:
        """Grid index holding a given doubled coordinate; inverse of :meth:`site_doubled`."""
        offset = int(doubled) - _axis_origin_doubled(self.grid, axis) - 1
        if offset % 2:
            raise ValueError(
                f"doubled coordinate {doubled} on axis {axis} is not on this plane's centred "
                f"lattice."
            )
        return offset // 2

    def packed(self, component: str, box=None, extra_scale=1.0,
               invariant=(False, False, False), flip=(), chunk_weight=None) -> np.ndarray:
        """One component's transform in MEEP's own stored layout — the ``dft_flux`` chunk.

        MEEP's counterpart of :meth:`YeeRegionDFT.packed`, and the two differ in
        exactly the way MEEP's two ``add_dft`` calls differ (dft.cpp:627-632):

        * an **E** component is registered with ``include_dV_and_interp_weights =
          true``, so MEEP's stored numbers already carry ``w = IVEC_LOOP_WEIGHT(s0,
          s1, e0, e1, dV0 + dV1*loop_i2)`` per site (dft.cpp:276-281). This engine
          accumulates the field RAW and applies its own measure at read time
          (:meth:`_integrate_flux` multiplies by ``_weights`` and ``_measure``), so
          the measure has to be put back on the way out;
        * an **H** component is registered with ``include_dV_and_interp_weights =
          false`` and carries no measure at all, here or there.

        WHICH measure is the subtlety, and it is not this engine's to choose.
        ``chunk_weight`` is a :class:`MeepChunkWeight` read off the live chunk, and
        when it is given it is the ONLY measure applied — this monitor's own
        ``_weights * _measure`` is not consulted, and the chunk's ``include`` flag,
        not the component's electric/magnetic kind, decides whether a weight rides at
        all. The two models agree on the region MEEP was ASKED for and disagree once
        ``symmetry::reduce`` has halved it (see :class:`MeepChunkWeight`), so the
        engine's own model is kept only as this file's CONTROL: passing
        ``chunk_weight=None`` for an E component reverts to it, and on an
        ``add_flux`` list under ``Mirror(Y)`` that scores 3.203e-01 against MEEP's
        stored E, wrong at exactly 4 of 84 sites, while staying exact on the
        full-plane lists.

        ``extra_scale`` is MEEP's own ``stored_weight`` for the chunk — the region
        weight times the tangential pair's ``(1 - 2i)`` sign, which is ``+1`` for the
        first component of the pair and ``-1`` for the second — read off the live
        chunk rather than re-derived from the normal direction, because
        ``process_dft_component`` divides the stored value by exactly that number
        when MEEP reads it back (dft.cpp:1004).

        ``box`` is ``((x0, x1), (y0, y1), (z0, z1))`` of INCLUSIVE region-relative
        indices; the emitted order is sites major and frequencies minor, sites in
        ``LOOP_OVER_IVECS`` order (the C-order ravel of the (x, y, z) block), which
        is what ``_load_dft_data`` writes back per chunk (python/meep.i:497-513).
        The chunk weight is applied AFTER ``box`` and ``flip``, because MEEP's loop
        index runs along the stored chunk ascending and ``flip`` is what puts this
        block into that order.

        ``invariant`` names the axes the run does not resolve, and it is not
        cosmetic. An invariant axis is a direction MEEP's reduced ``grid_volume``
        does not have at all, while this engine registers the flat request there
        through the SAME ladder as any other axis and gets TWO half-weight planes
        holding the identical field (measured on a 2-D run: ``_weights`` = 0.25 per
        site where MEEP's is 0.5, and the accumulator's z axis is 2 cells wide). So
        the FIELD is taken from the first of those planes — they are identical — and
        the two half weights are never used: MEEP's chunk has no such direction, its
        ``s0`` there reads back 1.0, and its ``dV0`` already carries the whole
        measure (measured on the same 2-D cell: ``dV0`` = 0.05 = 1/resolution).
        Under the control path, where the engine's own weights are used instead, that
        axis's weights are SUMMED rather than sliced, because
        ``sum_k w_k * f == (sum_k w_k) * f``; taking one plane's weight there emits
        exactly half of MEEP's stored E.
        """
        electric = _is_electric(component)
        block = (self._dft_E if electric else self._dft_H).get(component)
        if block is None:
            raise ValueError(
                f"'{component}' is not one of this flux monitor's components "
                f"({', '.join(ELECTRIC_COMPONENTS + MAGNETIC_COMPONENTS)})."
            )
        field = to_numpy(block).astype(np.complex128)
        measure = None
        if chunk_weight is None and electric:
            measure = to_numpy(self._weights).astype(np.float64) * self._measure
        for axis in range(3):
            if not invariant[axis]:
                continue
            if measure is not None:
                measure = measure.sum(axis=axis, keepdims=True)
            field = field[_slab(axis + 1, slice(0, 1))]
        weighted = field * complex(extra_scale)
        if measure is not None:
            weighted = weighted * measure[None, :, :, :]
        if box is not None:
            weighted = weighted[
                :,
                box[0][0]: box[0][1] + 1,
                box[1][0]: box[1][1] + 1,
                box[2][0]: box[2][1] + 1,
            ]
        for axis in flip:
            weighted = np.flip(weighted, axis=1 + int(axis))
        if chunk_weight is not None and chunk_weight.include:
            weighted = weighted * chunk_weight.grid_for(weighted.shape[1:])[None, :, :, :]
        return np.moveaxis(weighted, 0, -1).reshape(-1)

    def get_dft_data(self) -> FluxDftData:
        """This plane's accumulated transforms, MEEP's ``sim.get_flux_data(flux)``.

        Host copies, so the run they came from can be closed. Handing the result to
        :meth:`load_minus_dft_data` on an identically built monitor in a second run
        is the normalization idiom — see that method for what the minus means.
        """
        return FluxDftData(planes=(FluxPlaneDft(
            E={name: np.array(to_numpy(block), copy=True) for name, block in self._dft_E.items()},
            H={name: np.array(to_numpy(block), copy=True) for name, block in self._dft_H.items()},
            frequencies=tuple(self.frequencies),
        ),))

    def scale_dfts(self, scale: complex) -> None:
        """Multiply every accumulated transform by ``scale``, in place.

        ``dft_flux::scale_dfts`` (dft.cpp:588-591) scales the E-side and the H-side
        chunk lists, and ``dft_chunk::scale_dft`` (dft.cpp:401-405) multiplies every
        stored element by the scale. BOTH sides, which is what makes the minus
        convention below subtract a FIELD rather than a power.
        """
        factor = complex(scale)
        for accumulator in self._dft_E.values():
            accumulator *= factor
        for accumulator in self._dft_H.values():
            accumulator *= factor

    def load_dft_data(self, data: FluxDftData) -> None:
        """Replace this plane's transforms with ``data``, MEEP's ``load_flux_data``.

        Replaces — it does not add (``python/meep.i:497-511`` assigns each element).
        The accumulation of the run that follows adds on top of whatever is left
        here, which is how the minus form below becomes a subtraction.
        """
        _load_flux_planes(data, (self,), "FluxMonitor.load_dft_data")

    def load_minus_dft_data(self, data: FluxDftData) -> None:
        """``load_dft_data`` then negate, MEEP's ``load_minus_flux_data``.

        THE MINUS CONVENTION, which is the whole normalization idiom and is easy to
        get quietly wrong. MEEP's ``Simulation.load_minus_flux_data``
        (``python/simulation.py:3642-3649``) loads the saved transform and then calls
        ``flux.scale_dfts(complex(-1.0))``, so the accumulator STARTS the second run
        at ``-E1`` and ``-H1`` and finishes it at ``E2 - E1`` and ``H2 - H1``. The
        reported flux is then ``Re[(E2-E1) x conj(H2-H1)]``.

        That is NOT ``flux2 - flux1``. Subtracting the two powers would keep the
        incident wave's interference with the reflected one; subtracting the FIELDS
        removes the incident wave itself and leaves the reflected wave's own power,
        which is what a reflectance is. Both quantities are smooth, finite and of
        the same order, so a run that subtracted the wrong one still completes and
        still plots — hence the pin in ``test_from_meep.py`` and the mutation
        ``minus_flux_data_subtracts_power_not_fields``.

        Negating only one side (E but not H) is the same class of quiet error: it
        flips the sign of the cross term instead of removing it.
        """
        self.load_dft_data(data)
        self.scale_dfts(-1.0)

    def reset(self):  # Zero every accumulator without reallocating.
        for accumulator in self._dft_E.values():
            accumulator.fill(0)
        for accumulator in self._dft_H.values():
            accumulator.fill(0)
        self._update_calls = 0

    def __repr__(self) -> str:
        return (
            f"FluxMonitor(freq={_frequency_summary(self.frequencies)}, "
            f"direction={['X', 'Y', 'Z'][self.direction]}, "
            f"decimation={self.decimation_factor})"
        )


# The four families ``add_dft_energy`` registers, paired as the reduction consumes
# them: the WEIGHTED member first (MEEP gives ``include_dV_and_interp_weights=true``
# to E and H and false to D and B, dft.cpp:722-729), so the surface/volume measure is
# applied exactly once per pair, the same "once" trick FluxMonitor plays with E and H.
_ENERGY_ELECTRIC_PAIRS = (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz"))
_ENERGY_MAGNETIC_PAIRS = (("Hx", "Bx"), ("Hy", "By"), ("Hz", "Bz"))
_ENERGY_COMPONENTS = tuple(
    name for pair in _ENERGY_ELECTRIC_PAIRS + _ENERGY_MAGNETIC_PAIRS for name in pair
)


@dataclass
class EnergyMonitor:
    """Multi-frequency electric/magnetic energy-density monitor over a region.

    MEEP's ``dft_energy`` (``fields::add_dft_energy``, src/dft.cpp:716-741). Per
    region MEEP loops ``LOOP_OVER_FIELD_DIRECTIONS`` (vec.hpp:147-149 — X, Y, Z on a
    Cartesian cell, Z, R, P on a Dcyl one, which is the same three axes this engine
    stores either way) and registers FOUR chunks per direction: ``E_d`` and ``H_d``
    with ``include_dV_and_interp_weights = true``, ``D_d`` and ``B_d`` without. The
    reductions are

    * ``electric()`` = ``sum_sites 0.5 * real(conj(E_d) * D_d)`` (dft.cpp:672-686),
    * ``magnetic()`` = ``sum_sites 0.5 * real(conj(H_d) * B_d)`` (dft.cpp:688-701),
    * ``total()``    = their sum (dft.cpp:703-714),

    summed over the three directions and over every chunk. So this is structurally
    :class:`FluxMonitor` with four component families instead of two, and it reuses
    that class's registration verbatim — :func:`_register_volume`,
    :func:`_volume_boundary_weights`, :func:`_volume_periodic_axes`,
    :func:`_sliced_component`/:func:`_gathered_component` — rather than growing a
    second reading of the same region.

    Timing is the stagger every monitor here shares: ``update_dfts`` picks
    ``is_H_or_B(c) ? timeH : timeE`` (dft.cpp:253-261), so E and D sample at
    ``time`` and H and B at ``time - dt/2``, which is exactly :func:`_is_electric`.

    **Under a mirror fold this returns the FULL-DOMAIN energy, like MEEP.** Unlike
    ``add_dft_force``, ``add_dft_energy`` does NOT call ``symmetry::reduce`` — it
    copies the volume list unchanged (dft.cpp:723) — and ``add_dft``'s
    ``loop_in_chunks`` still runs with ``use_symmetry = true``, so MEEP visits each
    image of the region. This engine's ``_register_volume`` reflected gather is the
    same visit, and the folded case is the one that was MEASURED (see the class's
    test): test_dft_energy's region straddles its ``Mirror(Y)`` plane.

    **A region ``weight`` is IGNORED, because MEEP ignores it.** ``add_dft_energy``
    passes ``stored_weight = 1.0`` on all four chunks (dft.cpp:722-729) and never
    reads ``where->weight``, unlike ``add_dft_flux`` (dft.cpp:629) and
    ``add_dft_force`` (stress.cpp:168-185). Reproducing that is the contract; the
    migration says so where a script could be surprised by it.

    Attributes:
        grid: Computational grid.
        frequencies: Monitored frequencies, MEEP natural units.
        center: (x, y, z) centre of the energy region.
        size: (sx, sy, sz) region size; any axis may be zero.
        periodic: Which axes wrap, as (x, y, z); the driver's flags, or None for
            :func:`_wrapping_axes`.
        decimation_factor: Accumulate every Nth step, MEEP's own rule.
    """

    grid: "Grid"
    frequencies: float | Sequence[float] | None = None
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] | None = None
    frequency: float | Sequence[float] | None = None
    periodic: tuple[bool, bool, bool] | None = None
    decimation_factor: int = 1

    _omegas: Any = field(default=None, repr=False, init=False)
    _scale: float = field(default=0.0, repr=False, init=False)
    _update_calls: int = field(default=0, repr=False, init=False)
    _region: tuple[int, int, int, int, int, int] | None = field(
        default=None, repr=False, init=False
    )
    _weights: Any = field(default=None, repr=False, init=False)
    _measure: float = field(default=0.0, repr=False, init=False)
    _gather: tuple | None = field(default=None, repr=False, init=False)
    _wrapped: tuple[bool, bool, bool] = field(default=(False, False, False), repr=False, init=False)
    _dft: dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        self.frequencies = normalize_frequencies(self.frequency, self.frequencies)
        self.frequency = self.frequencies[0]
        self._omegas = 2 * np.pi * np.asarray(self.frequencies, dtype=np.float64)
        self.decimation_factor = _positive_decimation(self.decimation_factor)
        self._scale = self.grid.dt * self.decimation_factor / math.sqrt(2 * math.pi)
        for name, value in (("center", self.center), ("size", self.size)):
            if value is None:
                raise ValueError(f"EnergyMonitor requires a region {name}; got None.")
            if len(tuple(value)) != 3:
                raise ValueError(
                    f"EnergyMonitor {name} must have three components (x, y, z), got {value!r}."
                )
        if any(not math.isfinite(extent) or extent < 0.0 for extent in self.size):
            raise ValueError(
                f"EnergyMonitor size must be finite and non-negative on every axis, "
                f"got {self.size!r}."
            )
        if any(not math.isfinite(coordinate) for coordinate in self.center):
            raise ValueError(
                f"EnergyMonitor center must be finite on every axis, got {self.center!r}."
            )
        self.center = tuple(float(value) for value in self.center)
        self.size = tuple(float(value) for value in self.size)
        description = f"Energy region centred at {self.center} with size {self.size}"
        self._region, self._gather, self._wrapped = _register_volume(
            self.grid, self.center, self.size,
            _volume_periodic_axes(self.grid, self.periodic, "EnergyMonitor"), description,
        )
        self._weights = _volume_boundary_weights(
            self.grid, self.center, self.size, self._region
        )
        x0, x1, y0, y1, z0, z1 = self._region
        shape = (x1 - x0, y1 - y0, z1 - z0)
        # MEEP's dV0 picks up one factor of 1/a per direction the region extends
        # along, so a volume integrates over a volume and a plane over an area.
        self._measure = self.grid.dx ** sum(1 for extent in self.size if extent > 0)
        _check_accumulator_budget(
            shape, self.num_frequencies, len(_ENERGY_COMPONENTS), description,
        )
        xp = self.grid.xp
        for component in _ENERGY_COMPONENTS:
            self._dft[component] = xp.zeros((self.num_frequencies,) + shape, dtype=xp.complex64)

    def _configure_decimation_factor(self, factor: int) -> None:
        """Install a driver-resolved automatic factor before accumulation starts."""
        if self._update_calls:
            raise RuntimeError(
                "Cannot change an energy monitor's decimation factor after accumulation "
                "has started."
            )
        self.decimation_factor = _positive_decimation(factor)
        self._scale = self.grid.dt * self.decimation_factor / math.sqrt(2 * math.pi)

    @property
    def num_frequencies(self) -> int:  # Length of the accumulators' leading frequency axis.
        return len(self.frequencies)

    def update(self, fields: "Fields", time: float, current_step: int | None = None):
        """Accumulate one timestep of all twelve components over the region."""
        if current_step is None:
            self._update_calls += 1
            step = self._update_calls
        else:
            step = int(current_step)
            if step != current_step or step < 1:
                raise ValueError(
                    f"current_step must be a positive whole number, got {current_step!r}."
                )
            self._update_calls = step
        if step % self.decimation_factor:
            return
        phase_electric, phase_magnetic = _phase_factors(self.grid, self._omegas, self._scale, time)
        for component in _ENERGY_COMPONENTS:
            phase = phase_electric if _is_electric(component) else phase_magnetic
            self._dft[component] += phase * self._sample(fields, component)

    def _sample(self, fields, component):  # Cell-centred field over the region, sliced or gathered.
        if self._gather is None:
            return _sliced_component(fields, self.grid, component, self._region)
        return _gathered_component(fields, self.grid, component, self._gather, self._region)

    def _reduce(self, pairs, index: int) -> float:
        """``measure * sum_pairs sum_sites 0.5*real(conj(w*first) * second)``.

        The weight rides the FIRST member of each pair because that is the one MEEP
        registers with ``include_dV_and_interp_weights = true`` — E against D, H
        against B. Weighting both would apply the measure twice and hand back an
        energy scaled by the cell weight squared, which on a region whose ends are
        partially covered is neither MEEP's answer nor a constant factor from it.
        """
        xp = self.grid.xp
        total = 0.0
        for weighted, plain in pairs:
            total += float(xp.sum(0.5 * xp.real(
                xp.conj(self._dft[weighted][index] * self._weights) * self._dft[plain][index]
            )))
        return total * self._measure

    def electric(self, freq_index: int = 0) -> float:
        """MEEP's ``dft_energy::electric()`` at one frequency; ``mp.get_electric_energy``."""
        return self._reduce(_ENERGY_ELECTRIC_PAIRS, _frequency_index(self.frequencies, freq_index))

    def magnetic(self, freq_index: int = 0) -> float:
        """MEEP's ``dft_energy::magnetic()`` at one frequency; ``mp.get_magnetic_energy``."""
        return self._reduce(_ENERGY_MAGNETIC_PAIRS, _frequency_index(self.frequencies, freq_index))

    def total(self, freq_index: int = 0) -> float:
        """Electric plus magnetic, MEEP's ``dft_energy::total()`` (dft.cpp:703-714)."""
        return self.electric(freq_index) + self.magnetic(freq_index)

    def get_electric_energy_spectrum(self):  # Host float64 (nf,), aligned with `frequencies`.
        return np.array(
            [self._reduce(_ENERGY_ELECTRIC_PAIRS, i) for i in range(self.num_frequencies)],
            dtype=np.float64,
        )

    def get_magnetic_energy_spectrum(self):  # Host float64 (nf,), aligned with `frequencies`.
        return np.array(
            [self._reduce(_ENERGY_MAGNETIC_PAIRS, i) for i in range(self.num_frequencies)],
            dtype=np.float64,
        )

    def get_total_energy_spectrum(self):  # Host float64 (nf,); electric + magnetic per frequency.
        return self.get_electric_energy_spectrum() + self.get_magnetic_energy_spectrum()

    def reset(self):  # Zero every accumulator without reallocating.
        for accumulator in self._dft.values():
            accumulator.fill(0)
        self._update_calls = 0

    def __repr__(self) -> str:
        return (
            f"EnergyMonitor(freq={_frequency_summary(self.frequencies)}, "
            f"center={self.center}, size={self.size}, "
            f"decimation={self.decimation_factor})"
        )


@dataclass(frozen=True)
class ForceDftData:
    """The accumulated force transforms — this engine's ``sim.get_force_data(force)``.

    Opaque, like MEEP's ``ForceData``: hand it back to
    :meth:`ForceMonitor.load_dft_data`. It is deliberately NOT MEEP's flat
    chunk-concatenation layout — this container never crosses into MEEP, and the only
    contract is that the two ends of the round trip agree, which the loader checks
    accumulator by accumulator.
    """

    blocks: tuple  # One host complex128 array per accumulator, in registration order.


@dataclass
class ForceMonitor:
    """Multi-frequency Maxwell-stress-tensor force over a region — DIAGONAL terms.

    MEEP's ``dft_force`` (``fields::add_dft_force``, src/stress.cpp:153-191). Per
    region MEEP resolves two directions: ``nd = normal_direction(where->v)``, the
    geometric normal of the volume, and ``fd = component_direction(where->c)``, which
    the Python layer built from the region's DECLARED ``direction``
    (``_add_fluxish_stuff``, python/simulation.py:3808-3817). They select two very
    different registrations, and only the diagonal one is reproduced here.

    **DIAGONAL** (``fd == nd``, stress.cpp:178-185). Per field direction ``d``,
    ``weight1 = where->weight * (d == fd ? +0.5 : -0.5)``, and both ``E_d`` and
    ``H_d`` are registered with ``include_dV_and_interp_weights = true``,
    ``sqrt_dV_and_interp_weights = true``, ``extra_weight = weight1`` and
    ``use_centered_grid = FALSE`` — raw Yee sites, which is why the accumulator is
    :class:`YeeRegionDFT` and not the centred :class:`DFTMonitor`. The reduction is
    ``stress_sum(diag, diag)`` (stress.cpp:90-114), i.e.
    ``sum_sites real(weight1 * F * conj(F))`` = ``Re(weight1) * |F|^2``, so a
    complex region weight is harmless here: its imaginary part cannot survive.

    The ``sqrt`` is the subtle part. MEEP folds ``sqrt(IVEC_LOOP_WEIGHT)`` in PER
    STEP (dft.cpp:296-300), so ``|F|^2`` carries the interpolation weight exactly
    ONCE. This accumulator stores the raw sum and applies the full weight to
    ``|F|^2`` at read time (:meth:`YeeRegionDFT.weighted_square`), which is
    algebraically the same because the weight is real and time-independent —
    ``sum_t phase*sqrt(w)*f`` squared is ``w * |sum_t phase*f|^2``.
    :meth:`YeeRegionDFT.packed` must NOT be used: it multiplies the TRANSFORM by the
    weight, which squares it under ``|F|^2``.

    **OFF-DIAGONAL** (``fd != nd``, stress.cpp:168-177) is REFUSED by name. It pairs
    two centred accumulators per family — ``E_fd`` weighted against ``E_nd``
    unweighted, likewise for H — through ``stress_sum(offdiag1, offdiag2)``. No case
    in MEEP's own test corpus or examples reaches it (test_force and
    parallel-wvgs-force.py are both diagonal), so it has no measured oracle, and a
    transcription with no measurement behind it would return a plausible force
    nobody could check. Refusing names the gap instead.

    **A SYMMETRY is REFUSED by name**, and this is the one place force differs from
    every other monitor here. ``add_dft_force`` opens with ``S.reduce(where_)``
    (stress.cpp:161), which does not merely visit the images: it MERGES volumes onto
    their symmetry-equivalent partners, adds ``phase_shift``-scaled weights, HALVES a
    volume redundant with itself, and DELETES any whose weight cancels to zero
    (vec.cpp:1416-1470). That rewrites the region before a single chunk exists.
    ``add_dft_flux`` and ``add_dft_energy`` do none of it. Reproducing it needs a
    measured case and there is none.

    Attributes:
        grid: Computational grid.
        frequencies: Monitored frequencies, MEEP natural units.
        center: (x, y, z) centre of the force region.
        size: (sx, sy, sz) region size.
        force_direction: The axis of the force being computed (MEEP's ``fd``), 0/1/2.
        normal: The region's geometric normal axis (MEEP's ``nd``), 0/1/2.
        weight: The region's ``weight``; only its real part can survive the
            diagonal reduction, exactly as in MEEP.
        decimation_factor: Accumulate every Nth step, MEEP's own rule.
    """

    grid: "Grid"
    frequencies: float | Sequence[float] | None = None
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] | None = None
    force_direction: int | None = None
    normal: int | None = None
    weight: complex = 1.0
    frequency: float | Sequence[float] | None = None
    decimation_factor: int = 1

    _accumulators: tuple = field(default=(), repr=False, init=False)
    _axis_weights: tuple = field(default=(), repr=False, init=False)

    def __post_init__(self):
        self.frequencies = normalize_frequencies(self.frequency, self.frequencies)
        self.frequency = self.frequencies[0]
        self.decimation_factor = _positive_decimation(self.decimation_factor)
        for name, value in (("center", self.center), ("size", self.size)):
            if value is None:
                raise ValueError(f"ForceMonitor requires a region {name}; got None.")
            if len(tuple(value)) != 3:
                raise ValueError(
                    f"ForceMonitor {name} must have three components (x, y, z), got {value!r}."
                )
        self.center = tuple(float(value) for value in self.center)
        self.size = tuple(abs(float(value)) for value in self.size)
        for name, axis in (("force_direction", self.force_direction), ("normal", self.normal)):
            if axis not in (0, 1, 2):
                raise ValueError(
                    f"ForceMonitor {name} must be 0 (X), 1 (Y) or 2 (Z); got {axis!r}."
                )
        self.force_direction = int(self.force_direction)
        self.normal = int(self.normal)
        if self.force_direction != self.normal:
            raise ValueError(
                f"a force region whose declared direction ({AXIS_NAMES[self.force_direction]}) "
                f"differs from its surface normal ({AXIS_NAMES[self.normal]}) takes MEEP's "
                f"OFF-DIAGONAL stress-tensor branch (stress.cpp:168-177), which pairs two "
                f"centred accumulators per family rather than squaring one raw-Yee "
                f"accumulator. That branch is not reproduced here: no case in MEEP's own "
                f"tests or examples exercises it, so there is no oracle to measure a "
                f"transcription against and it would return a force nobody could check."
            )
        weight = complex(self.weight)
        self.weight = weight
        # stress.cpp:180-184 — weight1 = where->weight * (+0.5 on the force axis, -0.5
        # on the other two), and stress_sum takes real(weight1 * F * conj(F)), so only
        # the real part can survive. Precomputed per axis so the reduction is one
        # multiply per accumulator.
        self._axis_weights = tuple(
            weight.real * (0.5 if axis == self.force_direction else -0.5)
            for axis in range(3)
        )
        bounds = tuple(
            (self.center[axis] - 0.5 * self.size[axis], self.center[axis] + 0.5 * self.size[axis])
            for axis in range(3)
        )
        self._accumulators = tuple(
            (axis, YeeRegionDFT(
                self.grid, component=family + "xyz"[axis], bounds=bounds,
                frequencies=self.frequencies, decimation_factor=self.decimation_factor,
            ))
            for axis in range(3)
            for family in ("E", "H")
        )

    def _configure_decimation_factor(self, factor: int) -> None:
        """Install a driver-resolved automatic factor before accumulation starts."""
        self.decimation_factor = _positive_decimation(factor)
        for _axis, accumulator in self._accumulators:
            accumulator._configure_decimation_factor(self.decimation_factor)

    @property
    def num_frequencies(self) -> int:  # Length of every accumulator's leading frequency axis.
        return len(self.frequencies)

    def update(self, fields: "Fields", time: float, current_step: int | None = None):
        """Accumulate one timestep into all six raw-Yee accumulators."""
        for _axis, accumulator in self._accumulators:
            accumulator.update(fields, time, current_step)

    def get_force(self, freq_index: int = 0) -> float:
        """MEEP's ``dft_force::force()`` at one frequency; ``mp.get_forces``."""
        index = _frequency_index(self.frequencies, freq_index)
        return float(sum(
            self._axis_weights[axis] * accumulator.weighted_square(index)
            for axis, accumulator in self._accumulators
        ))

    def get_force_spectrum(self):
        """Force at every monitored frequency, host float64 (nf,)."""
        return np.array(
            [self.get_force(index) for index in range(self.num_frequencies)],
            dtype=np.float64,
        )

    def get_dft_data(self) -> ForceDftData:
        """This monitor's accumulated transforms; MEEP's ``sim.get_force_data(force)``.

        The counterpart of :meth:`FluxMonitor.get_dft_data`, and it exists for the
        same reason: MEEP's own method reads ``force.offdiag1`` and friends, which on
        a run this engine stepped are EMPTY, so a test that round-trips its force data
        through MEEP would save zeros, load zeros, and still read the right answer out
        of this monitor — a no-op dressed as a store/load. Serving the round trip from
        here makes the test exercise what it says it exercises.
        """
        return ForceDftData(blocks=tuple(
            to_numpy(accumulator._dft).astype(np.complex128)
            for _axis, accumulator in self._accumulators
        ))

    def load_dft_data(self, data: ForceDftData) -> None:
        """MEEP's ``load_force_data``: replace every accumulator's transform.

        Validated accumulator by accumulator BEFORE anything is written, so a
        mismatched save leaves the monitor untouched rather than half-loaded.
        """
        blocks = tuple(getattr(data, "blocks", ()) or ())
        if len(blocks) != len(self._accumulators):
            raise ValueError(
                f"ForceMonitor.load_dft_data: the data carries {len(blocks)} "
                f"accumulator block(s) and this monitor has {len(self._accumulators)}. "
                f"The two runs must agree on the cell and on the force region."
            )
        for block, (_axis, accumulator) in zip(blocks, self._accumulators):
            if tuple(np.shape(block)) != tuple(accumulator._dft.shape):
                raise ValueError(
                    f"ForceMonitor.load_dft_data: a block of shape "
                    f"{tuple(np.shape(block))} does not fit this monitor's "
                    f"{tuple(accumulator._dft.shape)}."
                )
        xp = self.grid.xp
        for block, (_axis, accumulator) in zip(blocks, self._accumulators):
            accumulator._dft = xp.asarray(np.asarray(block), dtype=accumulator._dft.dtype)

    def scale_dfts(self, scale: complex) -> None:  # dft_force::scale_dfts (stress.cpp:145-149).
        for _axis, accumulator in self._accumulators:
            accumulator._dft = accumulator._dft * complex(scale)

    def reset(self):  # Zero every accumulator without reallocating.
        for _axis, accumulator in self._accumulators:
            accumulator.reset()

    def __repr__(self) -> str:
        return (
            f"ForceMonitor(freq={_frequency_summary(self.frequencies)}, "
            f"direction={AXIS_NAMES[self.force_direction]}, "
            f"decimation={self.decimation_factor})"
        )


def _clamp_region(grid, region):  # Clamp a half-open index region to the grid extent.
    x0, x1, y0, y1, z0, z1 = region
    return (
        max(0, min(grid.nx, x0)), max(0, min(grid.nx, x1)),
        max(0, min(grid.ny, y0)), max(0, min(grid.ny, y1)),
        max(0, min(grid.nz, z0)), max(0, min(grid.nz, z1)),
    )


def _wrapping_axes(grid) -> tuple[bool, bool, bool]:
    """Axes a monitor region may be continued across when the caller names none.

    Every unmirrored axis wraps, absorber or not: MEEP takes a lattice shift where
    ``boundaries[High][d] == Periodic``, which ``use_bloch`` makes every unmirrored
    direction, and a PML is a material graded underneath that boundary rather than a
    different boundary (:func:`~.driver._periodic_axes` is the driver's copy of the
    same rule). A mirrored axis never wraps — MEEP reaches its far half through the
    symmetry transform — and that is read per axis, so a folded Z is terminated
    rather than handed the wrap of an axis it no longer has.
    """
    return tuple(not _axis_is_mirrored(grid, axis) for axis in range(3))


# ---------------------------------------------------------------------------
# Near-to-far-field transformation (MEEP src/near2far.cpp)
# ---------------------------------------------------------------------------

# Which E and H components are tangential to a plane whose normal is the given axis.
# Only these enter the equivalent currents: n x E and n x H annihilate the normal
# component, so registering it would accumulate a field nothing reads. MEEP's
# `add_dft_near2far` registers the same two E and two H components per face.
_TANGENTIAL_AXES = ((1, 2), (2, 0), (0, 1))

# Relative tolerance the cylindrical ring quadrature is doubled to, MEEP's own default
# for every entry point that takes one (src/meep.hpp:1363, 1369, 1373, 1377). It became
# a caller parameter in MEEP #3064; before that (1.29 and earlier) it was hardcoded to
# the same 1e-3 alongside the pre-#3047 convergence test, which normalized by the
# COHERENT sum instead of the integrand's L1 norm and so tightened with |m|. Two MEEP
# versions therefore do not run the same quadrature at the same nominal tolerance —
# quote the version with any number measured against one.
_GREENCYL_TOL = 1e-3


def _axis_cross(axis: int, vector):
    """``e_axis x V`` for a vector held as a three-tuple of arrays.

    ``e_a x V = sum_b V_b (e_a x e_b) = sum_bc eps_abc V_b e_c``, so the component
    along ``a`` vanishes and the other two are a signed swap of the tangential pair.
    Written out rather than fetched from a cross-product helper because the operands
    are whole ``(nf, N)`` accumulator stacks, not 3-vectors.
    """
    first, second = _TANGENTIAL_AXES[axis]
    out = [None, None, None]
    out[axis] = None  # Identically zero; the caller must not read it.
    out[first] = -vector[second]
    out[second] = vector[first]
    return out


@dataclass(frozen=True)
class Near2FarRegion:
    """One planar patch of a near-field surface, in MEEP's ``Near2FarRegion`` shape.

    Attributes:
        center: (x, y, z) centre of the patch, in MEEP coordinates.
        size: (sx, sy, sz) extent; exactly one axis must be zero (or thinner than a
            cell) and that axis is the patch normal.
        weight: Complex multiplier applied to this patch's equivalent currents. Its
            SIGN carries the outward normal, exactly as in MEEP: the nominal normal is
            the +axis, so the low face of a box takes ``weight=-1`` and the high face
            ``weight=+1``. :meth:`Near2FarMonitor.box` fills these in.
        direction: Normal axis 0 (X), 1 (Y) or 2 (Z); taken from the flat axis of
            ``size`` when omitted, and refused when that is ambiguous — a patch whose
            normal was guessed radiates a plausible pattern in the wrong direction.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    weight: complex = 1.0
    direction: int | None = None

    def normal_axis(self, cell_size: float) -> int:
        """Resolve the patch normal, refusing an ambiguous one.

        ``cell_size`` is the grid spacing: an axis thinner than one cell is flat for
        this purpose, the same test :class:`FluxMonitor` and
        :meth:`~.driver.FdtdDriver.add_flux_monitor` apply.
        """
        extents = tuple(float(value) for value in self.size)
        if len(extents) != 3 or len(tuple(self.center)) != 3:
            raise ValueError(
                f"A Near2FarRegion needs a three-component (x, y, z) center and size; got "
                f"center={self.center!r}, size={self.size!r}."
            )
        if any(not math.isfinite(value) or value < 0.0 for value in extents):
            raise ValueError(
                f"Near2FarRegion size must be finite and non-negative on every axis, got {self.size!r}."
            )
        if any(not math.isfinite(float(value)) for value in self.center):
            raise ValueError(
                f"Near2FarRegion center must be finite on every axis, got {self.center!r}."
            )
        flat = [axis for axis in range(3) if extents[axis] < cell_size]
        if self.direction is None:
            if len(flat) != 1:
                raise ValueError(
                    f"Near2FarRegion size {self.size!r} has {len(flat)} axes thinner than one "
                    f"cell ({cell_size:g}); a near-field patch is a plane with exactly one. "
                    f"Give a flat normal axis, or pass an explicit direction."
                )
            return flat[0]
        axis = int(self.direction)
        if axis not in (0, 1, 2):
            raise ValueError(f"Near2FarRegion direction must be 0 (X), 1 (Y) or 2 (Z); got {self.direction!r}.")
        if axis not in flat:
            raise ValueError(
                f"Near2FarRegion normal {AXIS_NAMES[axis]} must be the flat axis, but size "
                f"{self.size!r} extends {extents[axis]:g} along it (one cell is {cell_size:g})."
            )
        return axis


@dataclass(frozen=True)
class _Near2FarFace:  # One registered patch: its monitor, its normal, and its weight.
    monitor: "DFTMonitor"
    region: Near2FarRegion
    axis: int
    weight: complex


class Near2FarMonitor:
    """Near-to-far-field transformation over one or more planar near-field patches.

    Translation of MEEP's ``dft_near2far`` (src/near2far.cpp). The tangential E and H
    on the near-field surface are accumulated at every monitored frequency, converted
    to the equivalent surface currents of the surface-equivalence (Love) theorem,

        J = n x H        (electric surface current)
        M = -n x E       (magnetic surface current)

    with ``n`` the OUTWARD normal, and then radiated into the unbounded homogeneous
    medium through the free-space dyadic Green's function. The result is the field
    the enclosed sources produce at any point outside the surface — a radiation
    pattern, a scattering cross-section, an antenna or metasurface response —
    computed from a near-field box small enough to fit inside the absorbing layer.

    **The Green's function.** With MEEP's ``exp(-i*omega*t)`` phasor convention (its
    DFT accumulates ``exp(+i*omega*t)``, so an outgoing wave carries ``exp(+i*k*r)``),
    ``g(r) = exp(i*k*r) / (4*pi*r)``, ``k = omega*sqrt(eps*mu)``, and

        E = integral[ i*omega*mu * Ghat . J  -  dg/dr * (rhat x M) ] dA
        H = integral[ i*omega*eps * Ghat . M  +  dg/dr * (rhat x J) ] dA

        Ghat = g * [ (1 + (i*k*r - 1)/(k*r)^2) * I
                     + (3 - 3*i*k*r - (k*r)^2)/(k*r)^2 * rhat rhat ]
        dg/dr = g * (i*k - 1/r)

    ``Ghat`` is the FULL dyadic, not its far-field limit, so the transformation is
    exact at any distance — which is what MEEP's ``get_farfield`` promises and what
    makes a moderate-radius analytic comparison meaningful. Its ``k*r -> infinity``
    limit is ``g * (I - rhat rhat)``, the transverse projector, and its ``k*r -> 0``
    limit is the static dipole dyadic; both limits are asserted in the tests.

    **The surface measure.** MEEP builds the near2far chunks with
    ``include_dV_and_interp_weights``, so each cell carries the ``loop_in_chunks``
    fractional-cell weight times the area element. Reproduced here by the same
    ``_fractional_cell_weights`` ladder :class:`FluxMonitor` uses, times ``dx**2``:
    a patch therefore integrates over exactly the area it was asked for, wherever it
    falls relative to the grid, and a patch requested between two rows of cell centres
    is bracketed by both and interpolated rather than snapped onto one.

    **Where the currents live.** This engine samples every component at CELL CENTRES
    (:func:`_apply_yee_interpolation`), so ``J`` and ``M`` are co-located and the two
    equivalent currents describe the same point. MEEP evaluates each component at its
    own Yee position instead, half a cell apart. Both are consistent discretizations
    of the same integral and both converge to it; they differ at second order in
    ``k*dx``, which is the size of the residual in the CPU-MEEP cross-validation and
    is measured there at two resolutions to show it falling.

    **Cylindrical runs.** On a Dcyl grid the same surface currents are radiated through
    MEEP's ``greencyl`` instead (:meth:`_greencyl`): a near-field sample there is a RING
    of current at its own radius carrying the run's ``exp(i*m*phi)`` dependence, and its
    far field is the phi integral of the dyadic above as the source rotates around the
    axis. The near-field surface is :meth:`cylinder` — the r wall and the two z caps,
    with no third pair of faces — each sample carries MEEP's ``2*pi*r`` ring weight, and
    a far point is spelled ``(r, 0, z)`` because MEEP evaluates on the ``phi = 0``
    half-plane, which is what makes the returned six-vector read as
    ``(Er, Ep, Ez, Hr, Hp, Hz)``. Radiating a cylindrical near field through the
    Cartesian dyadic instead is not an approximation of this: measured against CPU MEEP
    on the same stored near data it is 5.9 relative, i.e. the whole answer.

    **What is NOT checked.** The equivalence theorem holds for a surface that encloses
    every source. :meth:`box` builds a closed one and
    :meth:`require_sources_enclosed` refuses a run whose sources are not inside it. An
    open surface — a single aperture plane, the usual metasurface measurement — is
    only valid in the half-space it faces and only when it captures the whole
    radiated field; the constructor refuses one unless the caller passes
    ``closed=False``, which is the acknowledgement, not a default.

    Args:
        grid: Computational grid; supplies the array module, spacing and registration.
        frequencies: Monitor frequency or frequencies, in MEEP natural units.
        regions: The near-field patches. Six of them, from :meth:`box`, is the closed
            case.
        eps: Relative permittivity of the homogeneous medium the far field radiates
            into. Must match the medium at the near-field surface; MEEP's
            ``dft_near2far`` carries the same pair.
        mu: Relative permeability of that medium.
        closed: Whether ``regions`` is asserted to be a closed surface. Left ``None``
            it is derived — six axis-aligned patches forming a box — and a surface
            that is not one is refused. Pass ``False`` to work with an open aperture
            deliberately.
        periodic: Which axes a patch may be continued across, as (x, y, z); defaults
            to :func:`_wrapping_axes`. Ignored on the driver path, where the driver
            resolves each region against the boundaries in force when stepping starts.
        decimation_factor: Accumulate every nth global timestep, as
            :class:`DFTMonitor`.
        _face_factory: How to create each patch's :class:`DFTMonitor`. The default
            builds and registers one directly; :meth:`on_driver` substitutes the
            driver's own ``add_dft_monitor`` so the driver owns the stepping.
    """

    def __init__(
        self,
        grid: "Grid",
        frequencies: float | Sequence[float] | None = None,
        regions: Sequence[Near2FarRegion] = (),
        *,
        frequency: float | Sequence[float] | None = None,
        eps: float = 1.0,
        mu: float = 1.0,
        closed: bool | None = None,
        periodic: tuple[bool, bool, bool] | None = None,
        decimation_factor: int = 1,
        greencyl_tol: float = _GREENCYL_TOL,
        _face_factory=None,
    ):
        self.grid = grid
        self.cylindrical = bool(getattr(grid, "cylindrical", False))
        self.m = float(getattr(grid, "m", 0.0) or 0.0)
        self.greencyl_tol = float(greencyl_tol)
        if self.cylindrical and not (math.isfinite(self.greencyl_tol) and self.greencyl_tol > 0.0):
            raise ValueError(
                f"greencyl_tol is the relative tolerance the ring quadrature is doubled to "
                f"(MEEP's dft_near2far::farfield argument, default {_GREENCYL_TOL:g}); it must be "
                f"finite and positive, got {greencyl_tol!r}."
            )
        self.frequencies = normalize_frequencies(frequency, frequencies)
        if any(value < 0.0 for value in self.frequencies):
            # The raw-Yee ACCUMULATOR side is sign-agnostic (the shared DFT phase,
            # dft.cpp:268-270), and that is the path the mp.Simulation migration
            # takes for a negative-frequency near2far — MEEP then evaluates the far
            # field itself via load_near2far. This class's own evaluators
            # (green3d / _greencyl) are transcribed and MEASURED at omega > 0 only,
            # so a negative frequency is refused here rather than radiated unpinned.
            raise ValueError(
                f"Near2FarMonitor evaluates far fields with Green's functions measured at "
                f"positive frequencies only, got {self.frequencies}. Accumulate the near "
                f"field at a negative frequency with add_dft_monitor (or lift the "
                f"mp.Simulation, whose migration stores raw accumulators) and hand the "
                f"far-field evaluation to MEEP via load_near2far."
            )
        self.frequency = self.frequencies[0]
        self.eps = float(eps)
        self.mu = float(mu)
        if not math.isfinite(self.eps) or self.eps <= 0.0 or not math.isfinite(self.mu) or self.mu <= 0.0:
            raise ValueError(
                f"The medium a near-to-far transformation radiates into must have finite positive "
                f"eps and mu, got eps={eps!r}, mu={mu!r}."
            )
        patches = tuple(regions)
        if not patches:
            raise ValueError(
                "A Near2FarMonitor needs at least one near-field patch; use Near2FarMonitor.box() "
                "for the closed surface that the equivalence theorem actually requires."
            )
        for patch in patches:
            if not isinstance(patch, Near2FarRegion):
                raise ValueError(
                    f"Near2FarMonitor regions must be Near2FarRegion objects, got "
                    f"{type(patch).__name__}."
                )
        self.closed = self._resolve_closed(patches, closed, self.cylindrical)
        self.decimation_factor = _positive_decimation(decimation_factor)
        self._drives_faces = _face_factory is None
        factory = _face_factory if _face_factory is not None else self._register_face_monitor
        wrapping = _wrapping_axes(grid) if periodic is None else tuple(bool(flag) for flag in periodic)
        if len(wrapping) != 3:
            raise ValueError(f"Near2FarMonitor periodic must be (x, y, z) flags, got {periodic!r}.")
        self._periodic = wrapping
        self._require_transformable_run()
        faces = []
        for patch in patches:
            axis = patch.normal_axis(self.grid.dx)
            if self.cylindrical and axis == 1:
                # A phi-normal patch is a half-plane cutting every ring, and its
                # equivalent currents are not what the exp(i*m*phi) near field on this
                # grid holds: the phi axis is invariant, so the patch would be one
                # sample wide and would radiate a wedge, smoothly and wrongly. MEEP
                # itself only tables R and Z normals in Dcyl (near2far.cpp:605-618).
                raise NotImplementedError(
                    f"A phi-normal near-field patch (center={patch.center!r}, size={patch.size!r}) "
                    f"has no meaning on a cylindrical grid: phi is the invariant axis, and MEEP's "
                    f"Dcyl near2far tables only R and Z normals (near2far.cpp:605-618). A "
                    f"cylindrical near-field surface is the r wall plus the two z caps."
                )
            components = tuple(
                family + AXIS_NAMES[tangent]
                for family in ("E", "H")
                for tangent in _TANGENTIAL_AXES[axis]
            )
            monitor = factory(patch, components)
            faces.append(_Near2FarFace(monitor=monitor, region=patch, axis=axis,
                                       weight=complex(patch.weight)))
        self._faces = tuple(faces)

    # -- construction ------------------------------------------------------

    def _require_transformable_run(self):
        """Refuse the two run shapes whose near field is not the field on one surface.

        **A Bloch phase.** With ``k_point != 0`` the near-field surface is one period of
        an infinite lattice, and the far field is the sum over every lattice image with
        its own phase — MEEP's ``Nperiods`` argument to ``add_dft_near2far``, which this
        engine does not yet carry. Radiating one period alone is an aperture diffraction
        pattern of a single unit cell: smooth, physically shaped, and not the grating's
        response. Nothing in the numbers distinguishes it.

        **Mirror symmetry.** A folded run stores one quadrant, so a near-field box
        registered on it covers only the stored half of every mirrored axis. The
        equivalent currents of a closed surface would then be missing the faces that
        live in the folded half, and the surface would not be closed at all — while
        still returning six patches' worth of plausible pattern.
        :meth:`DFTMonitor.get_dft_full` unfolds a monitor region, but a near-field patch
        on the mirror plane is not a region with a rectangular full-domain image, so
        that path does not apply.
        """
        grid = self.grid
        if getattr(grid, "has_symmetry", None) is not None and grid.has_symmetry():
            raise ValueError(
                f"A near-to-far transformation cannot run on a mirror-folded grid "
                f"(symmetry={getattr(grid, 'symmetry', '?')}): the stored quadrant holds only part "
                f"of every near-field patch, so the surface is not closed and its equivalent "
                f"currents are missing the folded half. Re-run without symmetry."
            )
        phases = [_axis_bloch_phase(grid, axis) for axis in range(3)]
        if any(phase is not None and abs(complex(phase) - 1.0) > 1e-12 for phase in phases):
            raise ValueError(
                f"A near-to-far transformation cannot run at k_point="
                f"{getattr(grid, 'k_point', '?')}: the near-field surface is one period of an "
                f"infinite lattice and the far field is the phased sum over every image (MEEP's "
                f"Nperiods), which this engine does not implement. Radiating a single period "
                f"instead returns the aperture pattern of one unit cell, which looks like a "
                f"radiation pattern."
            )

    def _register_face_monitor(self, patch: Near2FarRegion, components):
        """Build one patch's DFT monitor and register it on the requested plane now.

        The standalone path: this monitor owns its accumulators and is stepped through
        :meth:`update`. The region is resolved immediately, because nothing else will.
        """
        monitor = DFTMonitor(
            self.grid,
            frequencies=self.frequencies,
            components=components,
            periodic=self._periodic,
            decimation_factor=self.decimation_factor,
        )
        monitor.set_region_from_volume(patch.center, patch.size, periodic=self._periodic)
        return monitor

    @staticmethod
    def _resolve_closed(patches, declared, cylindrical: bool = False):
        """Decide whether the patches form a closed surface, and refuse a silent open one.

        A surface that does not enclose the sources still produces a smooth,
        plausible-looking pattern — the field of the currents that happen to be on it —
        with nothing in the numbers to say the theorem it came from did not apply. So
        an open surface has to be asked for: ``closed=False`` is the caller's
        acknowledgement that the result is a half-space aperture integral, and
        anything else is refused.
        """
        if declared is not None:
            return bool(declared)
        if cylindrical:
            # A closed surface in Dcyl is not a six-faced box: the surface of revolution
            # that encloses a volume is the r wall plus the two z caps, and the "face"
            # at r = 0 is a ring of zero circumference that carries zero weight
            # (2*pi*r = 0 there — measured on MEEP's own stored weights). Deriving
            # closure from six Cartesian faces cannot see that, so on this grid the
            # caller states it.
            raise ValueError(
                "On a cylindrical grid a closed near-field surface is the r wall plus the two "
                "z caps (the r = 0 'face' has zero circumference and contributes nothing), which "
                "is not the six-face box this derivation recognizes. Pass closed=True when the "
                "patches enclose the sources, or closed=False for an open aperture."
            )
        if _is_closed_box(patches):
            return True
        raise ValueError(
            f"These {len(patches)} near-field patches do not form a closed box, so the "
            f"surface-equivalence theorem does not apply to them and the far field they "
            f"produce is an aperture integral over one half-space, not the radiation of the "
            f"enclosed sources. Use Near2FarMonitor.box() for a closed surface, or pass "
            f"closed=False to work with an open aperture deliberately."
        )

    @classmethod
    def box(
        cls,
        grid: "Grid",
        frequencies: float | Sequence[float] | None = None,
        center: Sequence[float] = (0.0, 0.0, 0.0),
        size: Sequence[float] | None = None,
        **kwargs,
    ) -> "Near2FarMonitor":
        """Enclose a volume in the six-face near-field box the theorem asks for.

        The low face of each axis takes ``weight=-1`` and the high face ``weight=+1``,
        which is MEEP's spelling of the outward normal (its ``Near2FarRegion`` has no
        normal of its own; the nominal normal is the +axis and the weight's sign turns
        it around). Getting one of those signs wrong flips one face's contribution
        without changing anything else about the run, so the box is built here rather
        than left to a caller to spell out six times.
        """
        if getattr(grid, "cylindrical", False):
            raise NotImplementedError(
                "A six-face box is not a closed surface on a cylindrical grid — two of its faces "
                "would be phi-normal, and the invariant phi axis has no such patch. Use "
                "Near2FarMonitor.cylinder() for the r wall plus the two z caps."
            )
        return cls(grid, frequencies=frequencies, regions=_box_regions(center, size),
                   closed=True, **kwargs)

    @classmethod
    def cylinder(
        cls,
        grid: "Grid",
        frequencies: float | Sequence[float] | None = None,
        *,
        radius: float,
        height: float,
        z_center: float = 0.0,
        **kwargs,
    ) -> "Near2FarMonitor":
        """Enclose a volume of revolution: the r wall and the two z caps.

        The cylindrical spelling of :meth:`box`. There is no third pair of faces —
        the surface is closed in the (r, z) half-plane, and the "face" at ``r = 0``
        is a ring of zero circumference whose weight is exactly zero (measured on
        MEEP's own stored near2far weights), so it neither exists nor is missing.
        Each cap takes the outward sign in z; the wall's outward normal is +r.
        """
        return cls(grid, frequencies=frequencies,
                   regions=_cylinder_regions(radius, height, z_center),
                   closed=True, **kwargs)

    @classmethod
    def on_driver(
        cls,
        driver,
        frequencies: float | Sequence[float] | None = None,
        regions: Sequence[Near2FarRegion] | None = None,
        *,
        center: Sequence[float] | None = None,
        size: Sequence[float] | None = None,
        decimation_factor: int = 1,
        **kwargs,
    ) -> "Near2FarMonitor":
        """Register a near-field surface on a running driver, through its public API.

        Each patch becomes one of the driver's own DFT monitors
        (``FdtdDriver.add_dft_monitor``), so the driver steps it, resolves its region
        against the boundaries in force when stepping starts, and applies its
        decimation policy — no separate monitor list and no second update path that
        could double-count or be forgotten. :meth:`update` is refused on the result
        for exactly that reason.

        ``center`` + ``size`` builds the closed box; ``regions`` takes explicit
        patches. The sources are checked to lie inside a closed box
        (:meth:`require_sources_enclosed`) before the monitor is returned.
        """
        if (regions is None) == (center is None and size is None):
            raise ValueError(
                "Near2FarMonitor.on_driver() takes either a near-field box (center and size) or an "
                "explicit list of regions, not both and not neither."
            )
        if regions is None:
            if center is None or size is None:
                raise ValueError("A near-field box needs both 'center' and 'size'.")
            patches = _box_regions(center, size)
        else:
            patches = tuple(regions)

        def factory(patch, components):  # One driver-owned DFT monitor per near-field patch.
            return driver.add_dft_monitor(
                frequencies=frequencies,
                components=components,
                center=patch.center,
                size=patch.size,
                decimation_factor=decimation_factor,
            )

        monitor = cls(
            driver.grid, frequencies=frequencies, regions=patches,
            decimation_factor=decimation_factor, _face_factory=factory,
            **({"closed": True} if regions is None else {}),
            **kwargs,
        )
        monitor.require_sources_enclosed(driver)
        monitor.require_surface_clear_of_pml(driver)
        return monitor

    # -- accumulation ------------------------------------------------------

    @property
    def num_frequencies(self) -> int:  # Length of every accumulator's leading frequency axis.
        return len(self.frequencies)

    @property
    def face_monitors(self) -> tuple:  # The per-patch DFT monitors, in the order the patches were given.
        return tuple(face.monitor for face in self._faces)

    def update(self, fields: "Fields", time: float, current_step: int | None = None):
        """Accumulate one timestep into every patch's DFT.

        Refused on a driver-registered monitor: those patches are already in the
        driver's monitor list and stepping them again would accumulate every sampled
        timestep twice — a far field wrong by a clean factor of two, which reads as a
        calibration question rather than as a bug.
        """
        if not self._drives_faces:
            raise RuntimeError(
                "This Near2FarMonitor's patches are registered on an FdtdDriver, which already "
                "updates them once per step. Updating it here as well would accumulate every "
                "timestep twice and scale the far field by two."
            )
        for face in self._faces:
            face.monitor.update(fields, time, current_step)

    def reset(self):  # Zero every patch's accumulators without reallocating.
        for face in self._faces:
            face.monitor.reset()

    def require_sources_enclosed(self, driver, tolerance: float = 0.0):
        """Refuse a closed near-field box that does not contain every source.

        The equivalence theorem replaces the sources INSIDE the surface by currents on
        it. A source outside contributes to the near fields anyway — it is simply part
        of the incident field there — and the transformation then radiates it as though
        it were inside, producing a pattern with no defect in it to see. Nothing about
        the returned numbers distinguishes that case, which is why it is a refusal.

        An open surface (``closed=False``) is exempt: the caller has already
        acknowledged that the enclosure premise does not hold.

        Args:
            driver: The run whose ``_sources`` are checked; each is a dataclass with
                ``center`` and ``size``.
            tolerance: Slack in MEEP length units allowed at each face. Zero means a
                source touching the box is already refused.
        """
        if not self.closed:
            return
        low, high = self.enclosure_bounds()
        for source in getattr(driver, "_sources", ()):
            center = tuple(float(value) for value in getattr(source, "center", (0.0, 0.0, 0.0)))
            size = tuple(float(value) for value in getattr(source, "size", (0.0, 0.0, 0.0)))
            for axis in range(3):
                source_low = center[axis] - 0.5 * size[axis]
                source_high = center[axis] + 0.5 * size[axis]
                if source_low < low[axis] - tolerance or source_high > high[axis] + tolerance:
                    raise ValueError(
                        f"A {getattr(source, 'component', '?')} source spanning "
                        f"{source_low:g}..{source_high:g} on the {AXIS_NAMES[axis]} axis is not "
                        f"inside the near-field box {low[axis]:g}..{high[axis]:g}. The "
                        f"surface-equivalence theorem only replaces the sources the surface "
                        f"encloses; a source outside it would be radiated as though it were "
                        f"inside, and the far field would look exactly as plausible."
                    )

    def require_surface_clear_of_pml(self, driver, tolerance_cells: float = 0.5):
        """Refuse a near-field surface that sits inside the absorbing layer.

        A patch inside a PML samples a field the absorber has already attenuated and
        rotated, and radiates THAT as though it were the physical near field. The far
        field comes back smooth, correctly shaped, and low by an amount that depends
        on how deep the patch went — the single most likely way to get a wrong
        radiation pattern out of a correct transformation, and nothing in the numbers
        says so. Measured on a 2.4-cube at resolution 15 with a 9-cell PML (interior
        +-0.6), sphere-integrating the far field of one Ez dipole: a box of size 1.0
        radiates 2.0933, size 1.2 (faces exactly on the interior edge) 2.1129, size 1.4
        — three cells inside the layer — 1.9401, and size 1.6 just 1.1643. A 7.3% and
        then a 44% loss, on a pattern that stays the same clean dipole lobe throughout.

        The line is the one the retired source/absorber predicate drew, kept here
        because for a MONITOR the physics still demands it: each face's absorber
        begins one own-thickness inside it, a half-cell allowance covers the Yee
        snapping of the ladder, and a face with no absorber constrains nothing.
        (Sources are ADMITTED inside the layer now — the deposit-mirror and
        withdraw fixes in ``sources`` are measured on Cartesian and Dcyl alike —
        but a near-field patch has no analogous fix: it reads an attenuated field,
        which is not an engine defect but the absorber doing its job.)
        """
        pml = getattr(driver, "pml", None)
        if pml is None:
            return
        faces = pml.thickness_by_face
        grid = self.grid
        lengths = (grid.Lx, grid.Ly, grid.Lz)
        tolerance = float(tolerance_cells) * grid.dx
        low, high = self.enclosure_bounds()
        for axis in range(3):
            low_cells, high_cells = faces[axis]
            # The axis's own origin, not -L/2: the cylindrical r axis starts at exactly
            # 0 (Grid.axis_origin, MEEP volcyl io=(0,0)), and measuring its interior
            # from -L/2 puts every r patch "outside the absorber" by a full radius.
            start = grid.axis_origin(axis)
            interior_low = start + low_cells * grid.dx
            interior_high = start + lengths[axis] - high_cells * grid.dx
            below = low_cells > 0 and low[axis] < interior_low - tolerance
            above = high_cells > 0 and high[axis] > interior_high + tolerance
            if below or above:
                raise ValueError(
                    f"The near-field surface spans {low[axis]:g}..{high[axis]:g} on the "
                    f"{AXIS_NAMES[axis]} axis, outside the PML interior "
                    f"{interior_low:g}..{interior_high:g}. A patch inside the absorber samples an "
                    f"already-attenuated field and radiates it as the physical near field, which "
                    f"returns a correctly shaped pattern at the wrong power."
                )

    def enclosure_bounds(self):
        """(low, high) corner of the box the patches bound, as two (x, y, z) tuples."""
        lows, highs = [], []
        for axis in range(3):
            edges = [face.region.center[axis] for face in self._faces]
            spans = [0.5 * face.region.size[axis] for face in self._faces]
            lows.append(min(edge - span for edge, span in zip(edges, spans)))
            highs.append(max(edge + span for edge, span in zip(edges, spans)))
        return tuple(lows), tuple(highs)

    # -- far-field evaluation ---------------------------------------------

    def _require_registered(self):
        """Refuse to read a far field from patches whose region is still the default.

        On the driver path a patch's region is resolved when stepping starts. Read
        before then, every patch would still cover the whole grid, and the "surface"
        integral would be a volume integral over the entire cell — a large, smooth,
        entirely wrong number.
        """
        for face in self._faces:
            if face.monitor._volume is None:
                raise ValueError(
                    "This near-field surface has not been registered on the grid yet: its patches "
                    "still cover the whole cell. Step the run at least once (the driver resolves "
                    "monitor regions when stepping starts) before reading a far field."
                )

    def _face_weights(self, face: _Near2FarFace):
        """Fractional-cell surface weights for one patch, shaped like its region.

        The same ladder :meth:`FluxMonitor._compute_boundary_weights` applies, from the
        same :func:`_fractional_cell_weights`: partial weights on the two cells at each
        end of an axis with extent, and ``w0`` / ``w1`` summing to one on the flat axis,
        which interpolates the currents onto the requested plane instead of snapping
        them to the nearest row of cell centres.
        """
        monitor = face.monitor
        x0, x1, y0, y1, z0, z1 = monitor.region
        starts = (x0, y0, z0)
        counts = (x1 - x0, y1 - y0, z1 - z0)
        center, size = monitor._volume
        weights = []
        for axis in range(3):
            weights.append(_fractional_cell_weights(
                self.grid, axis,
                center[axis] - 0.5 * size[axis], center[axis] + 0.5 * size[axis],
                starts[axis], counts[axis],
            ))
        if self.cylindrical:
            # The ring measure. MEEP stores every Dcyl near2far sample already
            # multiplied by 2*pi*r (loop_in_chunks.cpp:505-512: dV0 picks up
            # 2*pi*|r| at the chunk's first sample and dV1 advances it per r row),
            # and greencyl then divides that 2*pi*r back out — its header says so
            # in as many words (near2far.cpp:277-278) — leaving the ring's own
            # `r dphi` measure in the phi quadrature. Carrying the same factor here
            # is what lets :meth:`_greencyl` be MEEP's quadrature verbatim, weight
            # 1/N per point and no 2*pi anywhere in it.
            #
            # r is each sample's OWN radius, not the patch's: 2*pi*r is linear in r,
            # so a flat-in-r patch's two interpolation cells reproduce 2*pi*r at the
            # requested radius exactly, while collapsing both onto the plane's radius
            # leaves the summed current identical and only moves where each ring sits
            # (measured 2.2e-04 in MEEP's own far field — smooth, plausible, and
            # invisible to any 1e-3 assertion). This engine samples at cell centres,
            # so r = (start + row + 0.5)*dx is never zero and no axis row is clipped.
            rows = np.arange(counts[0], dtype=np.float64)
            ring = 2.0 * np.pi * (starts[0] + rows + 0.5) * self.grid.dx
            weights[0] = weights[0] * self.grid.xp.asarray(ring.astype(np.float32))
        return (weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :])

    def _surface_currents(self):
        """Every patch's cell positions and weighted equivalent currents, flattened.

        Returns ``(positions, J, M)`` with ``positions`` of shape ``(N, 3)`` and the
        currents ``(nf, N, 3)``, already carrying the patch weight, the fractional-cell
        surface weight and the area element — so the far-field sum is a plain sum over
        ``N`` and the whole surface is one contiguous reduction rather than six.

        Cells whose surface weight is zero are dropped. They contribute nothing by
        construction, and dropping them also removes the only way a cell of a patch
        that was clipped at a face could sit at the observation point and divide by
        zero distance.
        """
        self._require_registered()
        xp = self.grid.xp
        positions_blocks, current_blocks, magnetic_blocks = [], [], []
        for face in self._faces:
            monitor = face.monitor
            weights = to_numpy(self._face_weights(face)).astype(np.float64)
            keep = weights.reshape(-1) != 0.0
            if not keep.any():
                continue
            x, y, z = monitor.get_coordinates()
            if self.cylindrical:
                # Axis 1 is phi, invariant and one sample wide; its stored coordinate
                # is a grid bookkeeping value, not an angle. Every near-field sample
                # of a Dcyl run sits on the phi = 0 half-plane, and :meth:`_greencyl`
                # rotates it around the axis itself. Carrying the bookkeeping value
                # through would displace every ring off its own plane.
                y = np.zeros(np.shape(np.asarray(y)), dtype=np.float64)
            coordinates = np.broadcast_arrays(
                np.asarray(x, dtype=np.float64)[:, None, None],
                np.asarray(y, dtype=np.float64)[None, :, None],
                np.asarray(z, dtype=np.float64)[None, None, :],
            )
            positions_blocks.append(np.stack(
                [axis_coordinates.reshape(-1)[keep] for axis_coordinates in coordinates], axis=1
            ))
            # MEEP loop_in_chunks dV0: one factor of 1/a per direction the patch
            # extends along. A near-field patch is a plane, so dA = dx**2 -- read off
            # the request rather than assumed, so a degenerate patch is comparable
            # with MEEP instead of off by a power of the grid spacing.
            _, size = monitor._volume
            measure = self.grid.dx ** sum(1 for extent in size if extent > 0)
            scale = xp.asarray(
                (weights.reshape(-1)[keep] * measure).astype(np.float64)
            )[None, :] * face.weight
            electric, magnetic = [None, None, None], [None, None, None]
            for tangent in _TANGENTIAL_AXES[face.axis]:
                name = AXIS_NAMES[tangent]
                electric[tangent] = monitor.get_dft_spectrum("E" + name).reshape(
                    self.num_frequencies, -1
                )[:, xp.asarray(keep)]
                magnetic[tangent] = monitor.get_dft_spectrum("H" + name).reshape(
                    self.num_frequencies, -1
                )[:, xp.asarray(keep)]
            # J = n x H and M = -n x E, with the patch weight carrying the sign of the
            # outward normal (MEEP's Near2FarRegion convention).
            crossed_h = _axis_cross(face.axis, magnetic)
            crossed_e = _axis_cross(face.axis, electric)
            zeros = xp.zeros(
                (self.num_frequencies, int(keep.sum())), dtype=xp.complex128
            )
            current_blocks.append(xp.stack([
                zeros if crossed_h[axis] is None
                else xp.asarray(crossed_h[axis], dtype=xp.complex128) * scale
                for axis in range(3)
            ], axis=2))
            magnetic_blocks.append(xp.stack([
                zeros if crossed_e[axis] is None
                else -xp.asarray(crossed_e[axis], dtype=xp.complex128) * scale
                for axis in range(3)
            ], axis=2))
        if not positions_blocks:
            raise ValueError(
                "Every near-field patch resolved to zero surface weight, so there is no surface to "
                "integrate over. A far field of zeros would read as a run that radiated nothing."
            )
        return (
            xp.asarray(np.concatenate(positions_blocks, axis=0)),
            xp.concatenate(current_blocks, axis=1),
            xp.concatenate(magnetic_blocks, axis=1),
        )

    def _radiate(self, positions, currents, magnetic_currents, points):
        """Radiate the equivalent currents to a set of observation points.

        MEEP source: near2far.cpp ``green3d`` / ``farfield_lowlevel``. One observation
        point at a time, vectorized over ``(nf, N)``: the ``N`` surface cells dominate,
        and holding one point's ``(nf, N)`` intermediates costs the same whether the
        caller asked for one far-field point or ten thousand.
        """
        xp = self.grid.xp
        omega = xp.asarray(2 * np.pi * np.asarray(self.frequencies, dtype=np.float64))
        wavenumber = omega * math.sqrt(self.eps * self.mu)
        result = np.zeros((points.shape[0], self.num_frequencies, 6), dtype=np.complex128)
        for index in range(points.shape[0]):
            offset = xp.asarray(points[index], dtype=xp.float64)[None, :] - positions  # (N, 3)
            distance = xp.sqrt(xp.sum(offset * offset, axis=1))  # (N,)
            closest = float(xp.min(distance))
            if closest < 0.5 * self.grid.dx:                # Not just the exact singularity. The surface cells are half a cell
                # apart at best, so inside that radius one cell's 1/r**3 term swamps the
                # sum and the "far field" is a property of which cell happened to be
                # nearest. An exact-zero test would not even catch the exact case: the
                # cell coordinates come back as float32, so a point named at a cell
                # centre misses it by ~1e-8 and returns an enormous finite number.
                raise ValueError(
                    f"The far-field point {tuple(float(value) for value in points[index])} lies on "
                    f"the near-field surface — {closest:g} from the nearest surface cell, less "
                    f"than the half-cell ({0.5 * self.grid.dx:g}) inside which the Green's "
                    f"function is singular. The transformation is defined off the surface."
                )
            electric, magnetic = _green3d(
                xp, offset / distance[:, None], distance,
                omega[:, None], wavenumber[:, None], self.eps, self.mu,
                currents, magnetic_currents,
            )
            result[index, :, 0:3] = to_numpy(xp.sum(electric, axis=1))
            result[index, :, 3:6] = to_numpy(xp.sum(magnetic, axis=1))
        return result

    def _greencyl(self, positions, currents, magnetic_currents, points):
        """Radiate a cylindrical near field: MEEP's ``greencyl`` (near2far.cpp:275-349).

        Every near-field sample of a Dcyl run is not a point current but a RING of
        current at its own radius, carrying the run's ``exp(i*m*phi)`` dependence.
        ``greencyl`` is the ring's Green's function, built by integrating
        :func:`_green3d` as the source rotates around the axis::

            EH(x) = (1/2pi) Integral[ green3d(x, x0(phi), R(phi).p) e^(i m phi) ] dphi

        with ``x0(phi) = (r0 cos phi, r0 sin phi, z0)`` and ``R(phi)`` rotating the
        source current out of the ``phi = 0`` half-plane the near field is stored on:
        ``r_hat = cos(phi) x_hat + sin(phi) y_hat`` and
        ``phi_hat = cos(phi) y_hat - sin(phi) x_hat`` (near2far.cpp:316-339, where the
        two are spelled as two ``green3d`` calls with ``cos``/``sin`` amplitudes; a
        z-directed current does not rotate). The ``1/2pi`` is not a normalization
        choice: the stored samples already carry MEEP's ``2*pi*r`` weight
        (:meth:`_face_weights`), and dividing it back out here leaves the ring's own
        ``r dphi`` measure — near2far.cpp:277-278 says exactly this.

        **The far point.** MEEP evaluates at ``x_3d = (x.r(), 0, x.z())``
        (near2far.cpp:282), i.e. on the ``phi = 0`` half-plane, which is what makes the
        returned six-vector read as ``(Er, Ep, Ez, Hr, Hp, Hz)``: there ``r_hat = x_hat``
        and ``phi_hat = y_hat`` (python/simulation.py:3225-3226 documents it).

        **The quadrature.** Equally spaced trapezoid in phi — the integrand is smooth
        and periodic, so it converges exponentially — starting at ``N0 = 16 + 4|m|``
        points and doubling until the sum stops moving: ``sumdiff <= sumabs * tol``
        with ``sumabs`` the L1 norm of the INTEGRAND accumulated across doublings, not
        of the coherent sum (MEEP #3047; the older criterion used ``|EH|``, which nearly
        cancels at large ``|m|`` and made the test over-strict). Previous points are
        re-used by halving the accumulated sum and summing only odd indices thereafter,
        and ``N`` is capped at 65536.

        **Where this deviates from MEEP, deliberately.** MEEP runs the doubling
        separately for each stored (cell, component) — one ``greencyl`` call per DFT
        chunk entry — while this runs it per (cell, frequency) with the whole
        ``(J, M)`` pair as one vector source. The quadrature points, weights and
        rotation are identical and the integrals are equal by linearity; only the
        stopping index can differ, because the convergence test sees the L1 norm of the
        summed integrand rather than of each component's separately. That is bounded by
        ``tol``, and the tests measure it: at ``tol=1e-8`` on both sides the far fields
        agree to the ``float64`` floor.

        Each (cell, frequency) converges on its own schedule, exactly as MEEP's
        per-call loop does, so a converged ring is frozen and dropped from the work
        list rather than carried to the next doubling.
        """
        xp = self.grid.xp
        cells = positions.shape[0]
        frequency_count = self.num_frequencies
        rows = frequency_count * cells
        omega = xp.asarray(2 * np.pi * np.asarray(self.frequencies, dtype=np.float64))
        wavenumber = omega * math.sqrt(self.eps * self.mu)
        omega_rows = xp.repeat(omega, cells)  # Frequency-major, matching the reshape below.
        wavenumber_rows = xp.repeat(wavenumber, cells)
        source_r = xp.tile(positions[:, 0], frequency_count)
        source_z = xp.tile(positions[:, 2], frequency_count)
        electric_rows = currents.reshape(rows, 3)
        magnetic_rows = magnetic_currents.reshape(rows, 3)
        quadrature_points = 16 + int(4 * abs(self.m))

        result = np.zeros((points.shape[0], frequency_count, 6), dtype=np.complex128)
        for index in range(points.shape[0]):
            far_r = float(points[index][0])
            far_z = float(points[index][2])
            # The ring passes closest to the far point at phi = 0, so the (r, z)
            # separation is the whole singularity test — the same half-cell floor the
            # Cartesian path applies, for the same reason.
            plane = xp.sqrt((far_r - positions[:, 0]) ** 2 + (far_z - positions[:, 2]) ** 2)
            closest = float(xp.min(plane))
            if closest < 0.5 * self.grid.dx:
                raise ValueError(
                    f"The far-field point (r={far_r:g}, z={far_z:g}) lies on the near-field "
                    f"surface — {closest:g} from the nearest source ring, less than the half-cell "
                    f"({0.5 * self.grid.dx:g}) inside which the Green's function is singular. The "
                    f"transformation is defined off the surface."
                )
            total = xp.zeros((rows, 6), dtype=xp.complex128)
            integrand_norm = xp.zeros(rows, dtype=xp.float64)
            live = xp.arange(rows)
            count = quadrature_points
            spacing = 2.0 / quadrature_points  # near2far.cpp:292 — halved before first use.
            while count <= 65536:
                spacing *= 0.5
                angle_step = spacing * 2 * np.pi
                # Re-use the previous quadrature by halving what it accumulated, then
                # sum only the odd points after the first pass (near2far.cpp:297-303).
                partial = total[live] * 0.5
                norm = integrand_norm[live] * 0.5
                first = 1 if count > quadrature_points else 0
                stride = 2 if count > quadrature_points else 1
                radius = source_r[live]
                height = source_z[live]
                electric_live = electric_rows[live]
                magnetic_live = magnetic_rows[live]
                omega_live = omega_rows[live]
                wavenumber_live = wavenumber_rows[live]
                for step in range(first, count, stride):
                    angle = step * angle_step
                    cosine, sine = math.cos(angle), math.sin(angle)
                    amplitude = complex(np.exp(1j * self.m * angle)) * spacing
                    offset = xp.stack([
                        far_r - radius * cosine,
                        -radius * sine,
                        far_z - height,
                    ], axis=1)
                    distance = xp.sqrt(xp.sum(offset * offset, axis=1))
                    electric_source = _rotate_about_axis(xp, electric_live, cosine, sine) * amplitude
                    magnetic_source = _rotate_about_axis(xp, magnetic_live, cosine, sine) * amplitude
                    electric, magnetic = _green3d(
                        xp, offset / distance[:, None], distance,
                        omega_live, wavenumber_live, self.eps, self.mu,
                        electric_source, magnetic_source,
                    )
                    contribution = xp.concatenate([electric, magnetic], axis=1)
                    partial = partial + contribution
                    norm = norm + xp.sum(xp.abs(contribution), axis=1)
                change = xp.sum(xp.abs(total[live] - partial), axis=1)
                total[live] = partial
                integrand_norm[live] = norm
                live = live[change > norm * self.greencyl_tol]
                if live.size == 0:
                    break
                count *= 2
            summed = xp.sum(total.reshape(frequency_count, cells, 6), axis=1)
            result[index] = to_numpy(summed)
        return result

    def _radiated_stack(self, points):  # (npoints, nf, 6) for a validated point array.
        requested = np.asarray(points, dtype=np.float64)
        single = requested.ndim == 1
        stacked = requested.reshape(1, 3) if single else requested
        if stacked.ndim != 2 or stacked.shape[1] != 3:
            raise ValueError(
                f"Far-field points must be one (x, y, z) triple or an (npoints, 3) array, got "
                f"shape {requested.shape}."
            )
        if not np.isfinite(stacked).all():
            raise ValueError("Far-field points must be finite; got a non-finite coordinate.")
        positions, currents, magnetic_currents = self._surface_currents()
        if not self.cylindrical:
            return self._radiate(positions, currents, magnetic_currents, stacked), single
        # On a Dcyl grid a far point is (r, 0, z): MEEP evaluates at phi = 0
        # (near2far.cpp:282), and that is what makes the returned six-vector read as
        # (Er, Ep, Ez, Hr, Hp, Hz) rather than Cartesian components. A point off that
        # half-plane is the same field rotated, and returning it in a basis the caller
        # would read as cylindrical is the silent kind of wrong, so it is refused.
        if np.any(stacked[:, 1] != 0.0):
            offending = stacked[np.argmax(stacked[:, 1] != 0.0)]
            raise ValueError(
                f"A cylindrical far-field point is given as (r, 0, z) — MEEP evaluates on the "
                f"phi = 0 half-plane, which is what makes the result (Er, Ep, Ez, Hr, Hp, Hz). "
                f"Got {tuple(float(value) for value in offending)}, whose middle (phi) "
                f"coordinate is not zero."
            )
        if np.any(stacked[:, 0] < 0.0):
            offending = stacked[np.argmax(stacked[:, 0] < 0.0)]
            raise ValueError(
                f"A cylindrical far-field point needs r >= 0; got "
                f"{tuple(float(value) for value in offending)}."
            )
        return self._greencyl(positions, currents, magnetic_currents, stacked), single

    def farfields(self, points, freq_index: int = 0):
        """Far fields at a set of observation points, at one monitored frequency.

        The ``freq_index=0`` default is the same convention as :meth:`DFTMonitor.get_dft`
        and :meth:`FluxMonitor.get_flux`; :meth:`farfield_spectrum` is the whole stack.

        Args:
            points: ``(npoints, 3)`` observation coordinates, or one ``(3,)`` point.
            freq_index: Which monitored frequency; 0 is the first.

        Returns:
            Host complex128 ``(npoints, 6)`` array of ``(Ex, Ey, Ez, Hx, Hy, Hz)``. A
            single ``(3,)`` point drops the leading axis.
        """
        radiated, single = self._radiated_stack(points)
        chosen = radiated[:, _frequency_index(self.frequencies, freq_index), :]
        return chosen[0] if single else chosen

    def farfield_spectrum(self, points):
        """Far fields at every monitored frequency.

        Returns:
            Host complex128 ``(npoints, nf, 6)`` array, the frequency axis aligned with
            :attr:`frequencies`. A single ``(3,)`` point drops the leading axis.
        """
        radiated, single = self._radiated_stack(points)
        return radiated[0] if single else radiated

    def farfield(self, point, freq_index: int = 0):
        """Far field at one observation point, as ``(Ex, Ey, Ez, Hx, Hy, Hz)``.

        The single-point spelling of :meth:`farfields`, matching MEEP's
        ``Simulation.get_farfield``.
        """
        return self.farfields(np.asarray(point, dtype=np.float64).reshape(3), freq_index=freq_index)

    def farfield_poynting(self, points, freq_index: int = 0):
        """Time-averaged Poynting vector of the far field at each point.

        ``Re(E x conj(H))``, the same convention as :meth:`FluxMonitor.get_flux`: the
        factor of one half lives in the ``dt/sqrt(2*pi)`` DFT scale, so a far-field
        Poynting integrated over a closed surface is directly comparable with a flux
        monitor's power on the same run and no extra factor enters either side.

        Returns:
            Host float64 array of shape ``(npoints, 3)``; a single point drops the axis.
        """
        fields = self.farfields(points, freq_index=freq_index)
        single = fields.ndim == 1
        stacked = fields.reshape(1, 6) if single else fields
        electric, magnetic = stacked[:, 0:3], np.conj(stacked[:, 3:6])
        poynting = np.real(np.cross(electric, magnetic))
        return poynting[0] if single else poynting

    def radiation_pattern(self, theta, phi, radius: float, freq_index: int = 0, center=None):
        """Radiant intensity ``r**2 * S_r`` on a sphere around the near-field surface.

        The quantity a radiation pattern actually is: power per unit solid angle,
        which for an outgoing spherical wave is independent of ``radius`` once the
        observation sphere is in the far zone. Its independence of ``radius`` is
        therefore a property of the result, not an assumption of the calculation —
        the transformation uses the full dyadic Green's function at whatever radius it
        is given — and the tests use exactly that to measure the ``1/r`` falloff.

        Args:
            theta: Polar angles from the +z axis, in radians; broadcast against ``phi``.
            phi: Azimuthal angles from the +x axis, in radians.
            radius: Sphere radius, measured from ``center``.
            freq_index: Which monitored frequency.
            center: Sphere centre; the centre of the near-field surface when omitted.

        Returns:
            Host float64 array shaped like the broadcast of ``theta`` and ``phi``.
        """
        polar, azimuth = np.broadcast_arrays(
            np.asarray(theta, dtype=np.float64), np.asarray(phi, dtype=np.float64)
        )
        span = float(radius)
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError(f"A radiation-pattern sphere needs a finite positive radius, got {radius!r}.")
        if self.cylindrical:
            # The sphere is swept in theta alone: a Dcyl far point lives on the phi = 0
            # half-plane, where (r_hat, phi_hat, z_hat) coincides with (x, y, z), so
            # the direction below is already the cylindrical one and the Poynting
            # projection needs no change. Its centre is ON the axis — taking the
            # midpoint of the r extent, as the Cartesian default does, would put the
            # sphere's centre at r = radius/2 and tilt every angle.
            if np.any(azimuth != 0.0):
                raise ValueError(
                    "A cylindrical radiation pattern is swept in theta on the phi = 0 half-plane "
                    "(the run's own exp(i*m*phi) carries the azimuth analytically); pass phi=0."
                )
            if np.any(polar < 0.0) or np.any(polar > np.pi):
                raise ValueError(
                    "A cylindrical radiation pattern needs theta in [0, pi]; outside it the "
                    "sphere point has r < 0, which is the phi = pi half-plane, not this one."
                )
            if center is None:
                low, high = self.enclosure_bounds()
                origin = np.array([0.0, 0.0, 0.5 * (low[2] + high[2])], dtype=np.float64)
            else:
                origin = np.asarray(center, dtype=np.float64).reshape(3)
                if origin[0] != 0.0 or origin[1] != 0.0:
                    raise ValueError(
                        f"A cylindrical radiation-pattern sphere is centred on the axis; got "
                        f"center={tuple(float(value) for value in origin)}."
                    )
        elif center is None:
            low, high = self.enclosure_bounds()
            origin = np.array([(low[axis] + high[axis]) / 2 for axis in range(3)], dtype=np.float64)
        else:
            origin = np.asarray(center, dtype=np.float64).reshape(3)
        direction = np.stack([
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ], axis=-1)
        points = origin + span * direction.reshape(-1, 3)
        poynting = self.farfield_poynting(points, freq_index=freq_index)
        radial = np.sum(poynting * direction.reshape(-1, 3), axis=1)
        return (span * span * radial).reshape(polar.shape)

    def __repr__(self) -> str:
        return (
            f"Near2FarMonitor(freq={_frequency_summary(self.frequencies)}, "
            f"patches={len(self._faces)}, closed={self.closed}, "
            f"decimation={self.decimation_factor})"
        )


def _box_regions(center, size) -> tuple[Near2FarRegion, ...]:
    """The six outward-facing patches of an axis-aligned near-field box.

    The low face of each axis takes ``weight=-1`` and the high face ``weight=+1``,
    MEEP's spelling of the outward normal. One flipped sign turns one face's
    contribution inward and changes nothing else that a caller could see.
    """
    center_values = tuple(float(value) for value in center)
    if size is None:
        raise ValueError("A near-field box needs its size; there is no default volume.")
    size_values = tuple(float(value) for value in size)
    if len(center_values) != 3 or len(size_values) != 3:
        raise ValueError(
            f"A near-field box takes a three-component (x, y, z) center and size; got "
            f"center={center!r}, size={size!r}."
        )
    if any(not math.isfinite(value) for value in center_values):
        raise ValueError(f"A near-field box must have a finite center on every axis, got {center!r}.")
    if any(not math.isfinite(value) or value <= 0.0 for value in size_values):
        raise ValueError(
            f"A near-field box needs a positive extent on every axis, got size={size!r}. A box flat "
            f"on one axis is two coincident faces whose equal and opposite contributions cancel to "
            f"a far field of zeros."
        )
    regions = []
    for axis in range(3):
        face_size = list(size_values)
        face_size[axis] = 0.0
        for sign in (-1, +1):
            face_center = list(center_values)
            face_center[axis] += sign * 0.5 * size_values[axis]
            regions.append(Near2FarRegion(
                center=tuple(face_center), size=tuple(face_size),
                weight=float(sign), direction=axis,
            ))
    return tuple(regions)


def _green3d(xp, unit, distance, omega, wavenumber, eps, mu, electric_current, magnetic_current):
    """The free-space dyadic Green's function of MEEP's ``green3d`` (near2far.cpp:133-187).

    Written as the fields of a co-located electric and magnetic point current, which is
    what a near-field cell carries::

        E = i*omega*mu * Ghat . J  -  dg/dr * (rhat x M)
        H = i*omega*eps * Ghat . M  +  dg/dr * (rhat x J)

        Ghat = g * [ (1 + (i*k*r - 1)/(k*r)^2) I + (3 - 3*i*k*r - (k*r)^2)/(k*r)^2 rhat rhat ]
        g = exp(i*k*r) / (4*pi*r),   dg/dr = g * (i*k - 1/r)

    MEEP spells the same thing as ``expfac * (term1*p + term2*rhat)``: its
    ``expfac = f0 * polar(k*n/(4*pi*r), k*r + pi/2)`` is ``i*k*n*g*f0``, so after the
    ``/= eps`` of the electric branch it is ``i*omega*mu*g*f0``; its ``term1`` is the
    isotropic coefficient above, its ``term2/(p.rhat)`` the radial one, and
    ``expfac*term3/Z`` is ``g*(i*k - 1/r)``. Both the ``1/r^2`` and ``1/r^3`` terms are
    kept, so the transformation is exact at any distance rather than in the far limit.

    Shapes broadcast: ``unit`` and the currents carry a trailing 3-axis, ``distance``,
    ``omega`` and ``wavenumber`` its leading shape. Shared by the Cartesian
    transformation and the per-quadrature-point evaluation inside :meth:`_greencyl`.
    """
    kr = wavenumber * distance
    green = xp.exp(1j * kr) / (4 * np.pi * distance)
    dyadic_isotropic = 1.0 + (1j * kr - 1.0) / (kr * kr)
    dyadic_radial = (3.0 - 3j * kr - kr * kr) / (kr * kr)
    gradient = green * (1j * wavenumber - 1.0 / distance)
    radial_j = xp.sum(unit * electric_current, axis=-1)
    radial_m = xp.sum(unit * magnetic_current, axis=-1)
    dyad_j = (dyadic_isotropic[..., None] * electric_current
              + dyadic_radial[..., None] * unit * radial_j[..., None])
    dyad_m = (dyadic_isotropic[..., None] * magnetic_current
              + dyadic_radial[..., None] * unit * radial_m[..., None])
    cross_m = _cross_last_axis(xp, unit, magnetic_current)
    cross_j = _cross_last_axis(xp, unit, electric_current)
    electric = (1j * omega[..., None] * mu * green[..., None] * dyad_j
                - gradient[..., None] * cross_m)
    magnetic = (1j * omega[..., None] * eps * green[..., None] * dyad_m
                + gradient[..., None] * cross_j)
    return electric, magnetic


def _rotate_about_axis(xp, vector, cosine, sine):
    """Carry a ``(r, phi, z)`` vector at ``phi = 0`` around to azimuth ``phi``, in Cartesian.

    ``r_hat = cos(phi) x_hat + sin(phi) y_hat`` and
    ``phi_hat = cos(phi) y_hat - sin(phi) x_hat`` (near2far.cpp:316-339, where MEEP
    spells the rotation as the ``cos``/``sin`` amplitudes of two Cartesian ``green3d``
    calls); a z-directed current does not rotate.
    """
    return xp.stack([
        vector[..., 0] * cosine - vector[..., 1] * sine,
        vector[..., 0] * sine + vector[..., 1] * cosine,
        vector[..., 2],
    ], axis=-1)


def _cylinder_regions(radius, height, z_center) -> tuple[Near2FarRegion, ...]:
    """The closed surface of revolution: the r wall and the two z caps.

    The cylindrical counterpart of :func:`_box_regions`. The caps take ``weight=-1``
    below and ``+1`` above (MEEP's spelling of the outward normal); the wall's outward
    normal is ``+r``, so it takes ``+1``. There is no fourth patch — the ``r = 0``
    "face" is a ring of zero circumference and MEEP weights it exactly zero.
    """
    span, tall = float(radius), float(height)
    middle = float(z_center)
    if not math.isfinite(span) or span <= 0.0 or not math.isfinite(tall) or tall <= 0.0:
        raise ValueError(
            f"A cylindrical near-field surface needs a positive radius and height, got "
            f"radius={radius!r}, height={height!r}."
        )
    if not math.isfinite(middle):
        raise ValueError(f"A cylindrical near-field surface needs a finite z_center, got {z_center!r}.")
    return (
        Near2FarRegion(center=(0.5 * span, 0.0, middle - 0.5 * tall),
                       size=(span, 0.0, 0.0), weight=-1.0, direction=2),
        Near2FarRegion(center=(0.5 * span, 0.0, middle + 0.5 * tall),
                       size=(span, 0.0, 0.0), weight=+1.0, direction=2),
        Near2FarRegion(center=(span, 0.0, middle),
                       size=(0.0, 0.0, tall), weight=+1.0, direction=0),
    )


def _cross_last_axis(xp, left, right):  # Cross product over the trailing 3-axis of two stacks.
    return xp.stack([
        left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1],
        left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2],
        left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0],
    ], axis=-1)


def _is_closed_box(patches, tolerance: float = 1e-9) -> bool:
    """Whether six planar patches tile the surface of one axis-aligned box.

    Two patches per axis, on opposite faces, each spanning the other two axes' full
    extent and carrying the outward sign (``weight`` negative on the low face,
    positive on the high one). Anything else — five faces, a face short of its
    neighbours, two faces with the same sign — leaves a hole the enclosed field
    escapes through, so it is not treated as closed.
    """
    if len(patches) != 6:
        return False
    by_axis: dict[int, list] = {0: [], 1: [], 2: []}
    for patch in patches:
        flat = [axis for axis in range(3) if float(patch.size[axis]) == 0.0]
        if patch.direction is not None:
            axis = int(patch.direction)
            if axis not in flat:
                return False
        elif len(flat) == 1:
            axis = flat[0]
        else:
            return False
        by_axis[axis].append(patch)
    if any(len(entries) != 2 for entries in by_axis.values()):
        return False
    bounds = {}
    for axis, entries in by_axis.items():
        low = [patch for patch in entries if complex(patch.weight).real < 0]
        high = [patch for patch in entries if complex(patch.weight).real > 0]
        if len(low) != 1 or len(high) != 1:
            return False
        if float(low[0].center[axis]) >= float(high[0].center[axis]):
            return False
        bounds[axis] = (float(low[0].center[axis]), float(high[0].center[axis]))
    for axis, entries in by_axis.items():
        for patch in entries:
            for other in range(3):
                if other == axis:
                    continue
                span = float(patch.size[other])
                expected = bounds[other][1] - bounds[other][0]
                centre = float(patch.center[other])
                midpoint = 0.5 * (bounds[other][0] + bounds[other][1])
                if abs(span - expected) > tolerance or abs(centre - midpoint) > tolerance:
                    return False
    return True


# --------------------------------------------------------------------------------------
# Field-function reducers — MEEP's ``fields::integrate`` / ``fields::max_abs``
# --------------------------------------------------------------------------------------
#
# src/integrate.cpp is one loop with two outputs: the weighted sum of a user integrand
# over the points ``loop_in_chunks`` visits, and the UNWEIGHTED maximum of the same
# integrand over the same points. Everything below is the sampling half of that loop —
# which sites MEEP visits, what value each carries, what weight it gets and where it is
# — expressed once so the six public reducers (:meth:`~.driver.FdtdDriver.
# integrate_field_function` and the five ``*_in_box`` measures of
# src/energy_and_flux.cpp) all read the same sites.
#
# THE ONE THING THAT IS NOT THE DFT MONITORS' PATH, and the reason this exists at all:
# ``loop_in_chunks`` is called with ``cgrid``, and ``cgrid`` is Centered only when the
# requested components do NOT all share a Yee shift (integrate.cpp:135-143). A pair like
# (Ex, Dx) DOES share one, so ``electric_energy_in_box`` loops the component's own raw
# Yee lattice and reads raw Yee values — no cell-centre average anywhere. Measured on the
# folded 10x10 cell of test_field_functions at resolution 20, integrating |Ez|^2 over the
# whole cell at t = 20: ``cs=[Ez]`` (cgrid = Ez, raw) gives 9.321e-13 and ``cs=[Ez,Hx]``
# (cgrid = Centered) gives 3.382e-13 — a factor of 2.756 apart, with and without symmetry
# alike. Reconstructing an energy from the centred array is wrong by 64%-99%.
#
# The geometry of that difference is measured too. Integrating the CONSTANT 1 over the
# same whole cell:
#
#     cgrid = Centered   ->  99.75015625     (parity 1 on both axes)
#     cgrid = Hx         ->  99.6253125      (parity 1 on x, 0 on y)
#     cgrid = Ez         ->  99.500625       (parity 0 on both axes)
#
# which factors exactly as the per-axis weight sums 199.75 (parity 1) and 199.5 (parity
# 0, whose 201-point ladder loses one end to the owned-corner clip): 9.9875**2,
# 9.975*9.9875 and 9.975**2. So the parity rule is per axis and it is
# ``iyee_shift(cgrid)`` on that axis — the same number that decides whether
# ``yee2cent_offsets`` averages there.

_REDUCER_MATERIALS = ("Dielectric", "Permeability")  # Not field reads; see `_material_trace`.
_CENTERED_SHIFT = (1, 1, 1)  # iyee_shift(Dielectric) / (Centered) — vec.hpp:1132-1140.


def reducer_component_shift(component: str) -> tuple[int, int, int]:
    """``gv.iyee_shift(c)`` for anything a reducer can be asked for.

    ``Dielectric`` and ``Permeability`` are all-ones (vec.hpp:1135-1138), i.e. the
    centred lattice, which is why a request for the permittivity alone still samples
    where the cell centres are.
    """
    if component in _REDUCER_MATERIALS:
        return _CENTERED_SHIFT
    return yee_shifts(component)


def reducer_cgrid(components: Sequence[str]) -> str | None:
    """MEEP's ``cgrid`` for one ``fields::integrate`` call — integrate.cpp:133-143.

    ``None`` stands for ``Centered``: the components do not all share a Yee shift, so
    every field read is the four-point ``yee2cent`` average and the loop lattice is the
    cell centres. Otherwise the loop runs on ``components[0]``'s OWN lattice and the
    field values are raw.
    """
    if not components:
        return None
    first = reducer_component_shift(components[0])
    for component in components[1:]:
        if reducer_component_shift(component) != first:
            return None
    return components[0]


@dataclass
class _ReducerAxis:
    """One axis of a reducer's sample plan: where MEEP loops, and with what weight.

    ``indices``/``codes``/``phases`` name the stored row each requested site is served
    from (a mirror image, a lattice image or itself) exactly as ``_register_volume``'s
    gather does; ``extended_*`` is the same map one site longer on the PARITY-0 lattice,
    which is the extra plane a ``yee2cent`` average consumes and is built only for a
    Centered loop.
    """

    indices: Any
    codes: Any  # int8 MIRROR_SITE_* per site, or None when every site is direct.
    phases: Any  # Complex Bloch factor per site, or None when every factor is 1.
    weights: Any  # IVEC_LOOP_WEIGHT per site, before dV0.
    coords: Any  # Physical coordinate of each site (the REQUESTED lattice, not the stored one).
    extended_indices: Any
    extended_codes: Any
    extended_phases: Any
    parity: int  # iyee_shift(cgrid) on this axis: which lattice the loop runs on.
    invariant: bool  # A direction this grid does not have; MEEP never loops it.


def _reducer_axis_map(grid, axis: int, first_site: int, count: int, parity: int):
    """(dropped_low, indices, codes, phases) for `count` sites of one parity on one axis.

    Three shapes, all of them already transcribed elsewhere in this file:

    * a MIRROR-FOLDED axis goes through :func:`folded_axis_sites`, MEEP's ``sn`` loop,
      which returns the stored row each site's symmetry image lands on plus the code
      saying whether it is direct, reflected or the PEC wall MEEP holds at zero;
    * a WRAPPING axis is continued into the neighbouring lattice image with its Bloch
      factor (loop_in_chunks.cpp:390-415), the rule :func:`_lattice_gather` states;
    * a WALLED axis is clipped at ``little_owned_corner`` below and at the last stored
      row above, which is MEEP's ``iscS = max(is - shifti, iscoS)`` clamp
      (loop_in_chunks.cpp:439-460) — the survivors keep their own ladder weights,
      exactly as :func:`_clip_to_owned` keeps them.
    """
    stored = _axis_cell_count(grid, axis)
    if _axis_is_mirrored(grid, axis):
        dropped, indices, codes = _folded_axis_sites(grid, axis, first_site, count, parity)
        # A LEADING zero-coded site is the low PEC wall of the FULL cell, and
        # `loop_in_chunks` never loops it: the clamp is
        # `iscoS = max(user_volume.little_owned_corner(cgrid), ...)`
        # (loop_in_chunks.cpp:421-424, 441) and `little_owned_corner0` is
        # `little_corner + 2 - iyee_shift(cgrid)` (vec.hpp:1102-1104), which for a
        # parity-0 cgrid is one site ABOVE that wall. `folded_axis_sites` serves it as a
        # zero because a DFT monitor's chunk covers it and stores a zero there, and for
        # a FIELD the two are the same answer — a zero-valued site adds nothing. For a
        # general integrand they are not: `f(loc, 0)` is whatever the caller's function
        # says. Measured on the folded 10x10 cell at resolution 20, integrating the
        # CONSTANT 1 with cgrid = Hx: 99.875 with the site visited against MEEP's
        # 99.6253125 without it, an exact half-cell of ladder weight.
        while int(indices.size) and int(codes[0]) == MIRROR_SITE_ZERO:
            dropped += 1
            indices, codes = indices[1:], codes[1:]
        if int(indices.size) == 0:
            raise ValueError(
                f"the requested volume covers no site MEEP loops on the folded "
                f"{AXIS_NAMES[axis]} axis. An integral there would sum nothing and report "
                f"0, which reads as a measurement."
            )
        return dropped, indices, codes, None
    raw = np.arange(first_site, first_site + count, dtype=np.int64)
    if _wrapping_axes(grid)[axis] and not _axis_is_metallic(grid, axis) and not grid.is_axis(axis):
        shifts = np.floor_divide(raw, stored)
        indices = raw - shifts * stored
        phase = _axis_bloch_phase(grid, axis)
        if phase is None or not shifts.any():
            return 0, indices, None, None
        factors = np.array([complex(phase) ** int(shift) for shift in shifts],
                           dtype=np.complex128)
        return 0, indices, None, factors
    first_owned = _first_owned_index(grid, axis, parity)
    # MEEP owns up to `big_owned_corner(c) = big_corner - iyee_shift(c)` (vec.hpp:1107),
    # which in stored indices is `stored - parity`: a parity-0 component owns ONE MORE
    # site than it has stored rows, the far PEC wall that `zero_metal` holds at zero.
    # It is served as a zero rather than dropped because MEEP visits it — it contributes
    # nothing to an energy (0 times 0) but it is a point `max_abs` takes a maximum over
    # and a point a general integrand is evaluated at.
    last_owned = stored - parity
    low = max(0, first_owned - first_site)
    high = min(count, last_owned + 1 - first_site)
    if high <= low:
        raise ValueError(
            f"the requested volume covers no site MEEP owns on the {AXIS_NAMES[axis]} axis: "
            f"its parity-{parity} ladder spans stored sites "
            f"[{first_site}, {first_site + count}) and this axis owns "
            f"[{first_owned}, {last_owned}]. An integral there would sum nothing and report "
            f"0, which reads as a measurement."
        )
    kept = raw[low:high]
    codes = np.where(kept >= stored, MIRROR_SITE_ZERO, MIRROR_SITE_DIRECT).astype(np.int8)
    return low, np.clip(kept, 0, stored - 1), (codes if (kept >= stored).any() else None), None


def _reducer_extension_map(grid, axis: int, start: int, count: int):
    """The parity-0 READ map a ``yee2cent`` average consumes — NOT clamped to the owned corner.

    MEEP clamps the LOOP to ``little_owned_corner(cgrid)`` and then reads ``idx + off``
    freely (integrate.cpp:111-118). Those offsets reach the chunk's boundary slots, which
    over a metallic wall hold the zero ``zero_metal`` puts there — the same plane
    :func:`_append_metallic_planes` appends for the DFT monitors, and the same reason:
    without it the average has nothing to consume and is silently dropped for the whole
    axis.
    """
    stored = _axis_cell_count(grid, axis)
    if _axis_is_mirrored(grid, axis):
        dropped, indices, codes = _folded_axis_sites(grid, axis, start, count, 0)
        if dropped or int(indices.size) != count:
            raise RuntimeError(
                f"the {AXIS_NAMES[axis]} axis serves cell centres its parity-0 site map cannot "
                f"cover ({dropped} dropped, {int(indices.size)} of {count} kept) — a mapping "
                f"slip, not a caller error."
            )
        return indices, codes, None
    raw = np.arange(start, start + count, dtype=np.int64)
    if _wrapping_axes(grid)[axis] and not _axis_is_metallic(grid, axis) and not grid.is_axis(axis):
        shifts = np.floor_divide(raw, stored)
        indices = raw - shifts * stored
        phase = _axis_bloch_phase(grid, axis)
        if phase is None or not shifts.any():
            return indices, None, None
        return indices, None, np.array([complex(phase) ** int(shift) for shift in shifts],
                                       dtype=np.complex128)
    outside = (raw < 0) | (raw >= stored)
    codes = np.where(outside, MIRROR_SITE_ZERO, MIRROR_SITE_DIRECT).astype(np.int8)
    return np.clip(raw, 0, stored - 1), (codes if outside.any() else None), None


def _reducer_axis_plan(grid, axis: int, where_min: float, where_max: float,
                       parity: int) -> _ReducerAxis:
    """Sites, weights and coordinates MEEP's ``loop_in_chunks`` visits on one axis.

    An INVARIANT axis is not one of ``gv.dim``'s directions, so MEEP has no loop for it
    at all: ``LOOP_OVER_IVECS`` never iterates it, ``IVEC_LOOP_WEIGHT`` carries no factor
    for it, ``yee2cent_offsets`` takes no offset along it (vec.cpp:333-344 loops
    ``LOOP_OVER_DIRECTIONS(dim, d)``) and ``vec2py`` reports its coordinate as 0
    (typemap_utils.cpp:127-147). One site, weight 1, coordinate 0 — not a two-cell
    interpolation onto a plane that does not exist.
    """
    if bool(grid.is_invariant(axis)):
        zero = np.zeros(1, dtype=np.int64)
        return _ReducerAxis(zero, None, None, np.ones(1, dtype=np.float64),
                            np.zeros(1, dtype=np.float64), zero, None, None,
                            parity=parity, invariant=True)
    is_doubled, ie_doubled, taper = _boundary_weight_ladder(
        where_min, where_max, grid.resolution, parity
    )
    count = (ie_doubled - is_doubled) // 2 + 1
    offset = is_doubled - _axis_origin_doubled(grid, axis) - parity
    if offset % 2:
        raise RuntimeError(
            f"reducer ladder on the {AXIS_NAMES[axis]} axis starts at doubled {is_doubled}, "
            f"which is not on the parity-{parity} lattice of an axis whose origin is "
            f"{_axis_origin_doubled(grid, axis)} — a transcription slip, not a caller error."
        )
    first_site = offset // 2
    weights = np.array([_ivec_loop_weight(index, count, *taper) for index in range(count)],
                       dtype=np.float64)
    coords = (is_doubled + 2.0 * np.arange(count, dtype=np.float64)) * 0.5 * grid.dx
    dropped, indices, codes, phases = _reducer_axis_map(grid, axis, first_site, count, parity)
    kept = int(indices.size)
    weights = weights[dropped:dropped + kept]
    coords = coords[dropped:dropped + kept]
    extended_indices, extended_codes, extended_phases = indices, codes, phases
    if parity == 1:
        # The plane a Yee-to-centre average consumes: the SAME requested sites plus one,
        # read on the parity-0 lattice. Built on every parity-1 axis, not only for
        # ``cgrid == Centered``: the MATERIAL integrands average through
        # ``yee2cent_offsets`` unconditionally (integrate.cpp:73-74 sits outside the
        # ``if (cgrid == Centered)`` branch at :70), so a call for ``[Dielectric]``
        # alone — whose cgrid is Dielectric, not Centered — still consumes it.
        start = first_site + dropped
        extended_indices, extended_codes, extended_phases = _reducer_extension_map(
            grid, axis, start, kept + 1
        )
    return _ReducerAxis(indices, codes, phases, weights, coords,
                        extended_indices, extended_codes, extended_phases,
                        parity=parity, invariant=False)


def _reducer_site_factors(xp, codes, parity: int):
    """Per-site multiplier for one mirror-code vector, or None when every factor is 1."""
    if codes is None:
        return None
    zero = codes == MIRROR_SITE_ZERO
    reflected = codes == MIRROR_SITE_REFLECTED
    if not zero.any() and not (reflected.any() and parity != 1):
        return None
    factors = np.where(zero, 0.0, np.where(reflected, float(parity), 1.0))
    return xp.asarray(factors.astype(np.float64))


def _reducer_sample(arrays, grid, component: str, axes, interpolate: bool, material: bool):
    """One component's value at every site of a reducer plan, as a 3-D array.

    ``arrays`` answers ``get_component(name)`` — :class:`~.fields.Fields` for a field read
    and :class:`_InverseChi1Arrays` for the two material integrands, which are not field
    reads at all (integrate.cpp:85-108).

    ``material`` changes TWO things, both of them about signs and walls rather than about
    where the samples are. The first is the mirror parity. The second is the
    :data:`MIRROR_SITE_ZERO` code: the plane past a PEC wall holds a zero FIELD, which is
    what ``zero_metal`` puts there, but it does not hold a zero MATERIAL — MEEP's
    ``chi1inv`` arrays are filled over the chunk's whole ``ntot``, boundary slot included,
    with the medium evaluated at that position (structure.cpp ``set_chi1inv``). So a
    material sample ignores the code and reads the nearest stored row instead, which is
    the same medium wherever the cell's outermost cells are uniform — true of every cell
    terminated by a PML in a homogeneous background, and NOT verified for a cell whose
    two opposite faces hold different media.

    Beyond that, ``material`` array is a
    scalar property of a structure the symmetry declares invariant, so its reflected
    image is itself and the factor is +1; the component's own ``S.phase_shift`` is the
    sign a FIELD picks up, and applying it to ``chi1inv`` would flip the sign of half the
    trace. It does NOT change the Yee shift: ``chi1inv[Ex][X]`` lives at Ex's own
    position and MEEP averages it with ``yee2cent_offsets(Ex)``, the SAME offsets it
    averages the field with (integrate.cpp:73-74 against :80-83). Taking the centred
    shift here instead skipped the average entirely and read raw per-component chi1inv —
    measured 1.7e-02 to 3.6e-01 wrong against MEEP on the eps=12 waveguide cell of
    test_wvg_src, and exact on vacuum, where nothing can tell the two apart.
    """
    from .fields import mirror_parity  # Local: keeps the dft -> fields import one-way.

    shifts = yee_shifts(component)
    sampled = arrays.get_component(component)
    for axis in range(3):
        plan = axes[axis]
        interpolates = interpolate and shifts[axis] == 0 and not plan.invariant
        if interpolates and plan.parity != 1:
            raise RuntimeError(
                f"a Yee-to-centre average was asked for on the {AXIS_NAMES[axis]} axis, whose "
                f"loop lattice is parity {plan.parity}; MEEP averages onto cell centres only, "
                f"and the parity-0 read map this needs was never built."
            )
        indices = plan.extended_indices if interpolates else plan.indices
        codes = plan.extended_codes if interpolates else plan.codes
        phases = plan.extended_phases if interpolates else plan.phases
        sampled = sampled.take(grid.xp.asarray(indices), axis=axis)
        phase = _axis_mirror_phase(grid, axis)
        parity = (1 if (material or phase is None)
                  else mirror_parity(component, axis, phase))
        for factors in (_reducer_site_factors(grid.xp, None if material else codes, parity),
                        None if phases is None else grid.xp.asarray(phases)):
            if factors is None:
                continue
            shape = [1, 1, 1]
            shape[axis] = factors.shape[0]
            sampled = sampled * factors.reshape(tuple(shape))
        if interpolates:
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(0, -1)
            upper[axis] = slice(1, None)
            sampled = 0.5 * (sampled[tuple(lower)] + sampled[tuple(upper)])
    return sampled


class _InverseChi1Arrays:
    """``fc->s->chi1inv[c][component_direction(c)]`` for one field family, by component name.

    Serves :func:`_reducer_sample` the same way :class:`~.fields.Fields` does, so the
    Dielectric integrand samples through ONE site map and one interpolation rule rather
    than a second copy of them. MEEP reads the diagonal entry only
    (integrate.cpp:88, ``chi1inv[iecs[k]][ieds[k]]`` with ``ieds[k] =
    component_direction(iecs[k])``), which is what makes the trace below a trace.

    ``get_component`` returns ``None`` where MEEP's pointer would be NULL, which is the
    ``else tr += 4`` branch of integrate.cpp:91-94 / :103-106 — a constant 1, contributed
    without sampling anything.
    """

    def __init__(self, fields, magnetic: bool):
        self._fields = fields
        self._magnetic = magnetic

    def get_component(self, component: str):
        if self._magnetic:
            return None  # This engine steps mu = 1: MEEP's `invmu` pointer is NULL.
        return self._fields.inverse_epsilon_for(component)


_REDUCER_ELECTRIC = ("Ex", "Ey", "Ez")  # FOR_ELECTRIC_COMPONENTS order (meep.hpp).
_REDUCER_MAGNETIC = ("Hx", "Hy", "Hz")  # FOR_MAGNETIC_COMPONENTS order.


def _material_trace(fields, grid, component: str, axes):
    """MEEP's Dielectric / Permeability integrand value at every site of a plan.

    integrate.cpp:85-108, transcribed::

        tr = sum over the field family's components of the FOUR-POINT SUM of chi1inv
        fvals = (4 * ninveps) / tr

    which is ``ninveps`` divided by the mean of the family's cell-centred inverse
    permittivities — the harmonic-style average MEEP reports as "the dielectric
    function", NOT the permittivity of any one component. Each four-point sum is four
    times :func:`_reducer_sample`'s centred average of the same array, so the two
    factors of four cancel and the expression below is the same quantity in the same
    order.

    MEASURED against MEEP on test_wvg_src's eps=12 waveguide cell (16 x 8, resolution 10,
    subpixel smoothing on): **exact to 1.2e-09 or better** on every volume that stops
    short of the cell's outermost row — a slab through the waveguide, an off-grid box, a
    flat line inside the 0.1-thick air slot — and **2.4e-03** on the WHOLE cell.

    The whole-cell gap is one row wide and it is a limitation of the lift, not of this
    loop. A shift-0 component's lowest site sits ON the PEC wall, and the per-component
    ``chi1inv`` this engine holds was read through MEEP's PUBLIC ``fields.get_chi1inv``,
    which serves owned points; the wall row is below ``little_owned_corner`` and comes
    back as vacuum, while MEEP's own internal array holds the medium there and
    ``integrate`` reads it. Measured: that row alone accounts for the whole difference
    (every interior row agrees to 1.2e-07, the float32 storage's own resolution), and it
    is exactly 0 on a vacuum cell, where the wall row IS vacuum — which is why
    ``test_field_functions`` cannot see it.
    """
    magnetic = component == "Permeability"
    family = _REDUCER_MAGNETIC if magnetic else _REDUCER_ELECTRIC
    arrays = _InverseChi1Arrays(fields, magnetic)
    shape = tuple(int(axis.indices.size) for axis in axes)
    total = grid.xp.zeros(shape, dtype=grid.xp.float32)
    for name in family:
        if arrays.get_component(name) is None:
            total = total + 1.0  # MEEP's `tr += 4` default, divided through by the 4.
            continue
        total = total + _reducer_sample(arrays, grid, name, axes, interpolate=True, material=True)
    return len(family) / total


def _reducer_volume_bounds(grid, center, size):
    """Per-axis (min, max) of the requested volume, with MEEP's leniency on a flat axis."""
    bounds = []
    for axis in range(3):
        extent = abs(float(size[axis]))  # MEEP's volume constructor sorts its corners.
        middle = float(center[axis])
        bounds.append((middle - extent / 2, middle + extent / 2))
    return tuple(bounds)


def integrate_field_function(fields, grid, components: Sequence[str], func, center, size):
    """MEEP's ``fields::integrate`` — src/integrate.cpp:131-201, with its ``max_abs``.

    Returns ``(integral, max_abs)``: the weighted sum of ``func`` over the points
    ``loop_in_chunks`` visits, and the UNWEIGHTED maximum of ``|func|`` over the same
    points. Both come from one pass because MEEP computes them in one pass
    (integrate.cpp:122-124) — ``max_abs`` is literally ``integrate(..., &maxabs)`` —
    and computing the maximum separately would be a second chance to disagree about
    which points those are.

    ``func`` is called as MEEP's SWIG bridge calls it (python/meep.i:158-188
    ``py_field_func_wrap``): the location first as a plain ``(x, y, z)`` tuple's worth
    of floats packaged by the caller, then one Python ``complex`` per requested
    component, in the order requested.

    THE WEIGHT is ``IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV0)`` — the per-axis fractional
    ladder times ``dV0 = inva ** (number of the grid's OWN directions the request has
    extent along)`` (loop_in_chunks.cpp:499-506). An axis with zero extent therefore
    contributes two sites whose weights sum to 1 and no factor of ``inva``: a linear
    interpolation onto the requested plane, not an integration across it. That is what
    ``test_field_functions``'s ``max_abs`` over a 1 x 0 line depends on — collapsing the
    flat axis to one row changes which points the maximum is taken over.
    """
    if bool(getattr(grid, "cylindrical", False)):
        raise NotImplementedError(
            "the field-function reducers are transcribed for Cartesian grids only; a Dcyl "
            "run needs loop_in_chunks' ring measure (dV0 + dV1*loop_i2) and MEEP's (r, z) "
            "location convention, neither of which is validated here."
        )
    names = [str(component) for component in components]
    for name in names:
        reducer_component_shift(name)  # Validates; raises on an unknown component.
    cgrid = reducer_cgrid(names)
    centered = cgrid is None
    shift = _CENTERED_SHIFT if centered else reducer_component_shift(cgrid)
    bounds = _reducer_volume_bounds(grid, center, size)
    axes = [
        _reducer_axis_plan(grid, axis, bounds[axis][0], bounds[axis][1], shift[axis])
        for axis in range(3)
    ]
    values = []
    for name in names:
        if name in _REDUCER_MATERIALS:
            values.append(_material_trace(fields, grid, name, axes))
        else:
            values.append(_reducer_sample(fields, grid, name, axes, centered, material=False))
    extents = sum(
        1 for axis in range(3)
        if not axes[axis].invariant and bounds[axis][1] > bounds[axis][0]
    )
    weight = (
        axes[0].weights[:, None, None] * axes[1].weights[None, :, None]
        * axes[2].weights[None, None, :] * (grid.dx ** extents)
    )
    host = [np.asarray(to_numpy(value)) for value in values]
    shape = tuple(len(axis.coords) for axis in axes)
    for index, block in enumerate(host):
        if block.shape != shape:
            raise RuntimeError(
                f"component {names[index]!r} sampled to {block.shape} where the plan visits "
                f"{shape} sites — the site map and the sampler have gone out of step."
            )
    total = 0.0 + 0.0j
    maxabs = 0.0
    coords = [np.asarray(axis.coords, dtype=np.float64) for axis in axes]
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                location = (float(coords[0][i]), float(coords[1][j]), float(coords[2][k]))
                integrand = complex(func(location, *(complex(block[i, j, k]) for block in host)))
                magnitude = abs(integrand)
                if magnitude > maxabs:
                    maxabs = magnitude
                total += integrand * float(weight[i, j, k])
    return total, maxabs


def _dot_pair_reduction(fields, grid, first: str, second: str, center, size) -> float:
    """``real(integrate(2, {first, second}, dot_integrand, where))`` — energy_and_flux.cpp:61-65.

    The pair shares a Yee shift wherever MEEP forms one (E with D, H with B), so this is
    the RAW-lattice branch of :func:`integrate_field_function`: no cell-centre average
    anywhere, and a per-axis ladder anchored on the component's own parity.
    """
    integral, _ = integrate_field_function(
        fields, grid, (first, second),
        lambda _location, a, b: (a.conjugate() * b).real,
        center, size,
    )
    return float(integral.real)
