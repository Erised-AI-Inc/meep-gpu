"""``mp.Absorber`` — the graded scalar conductivity MEEP puts on a face instead of a PML.

An absorber is NOT a perfectly matched layer and this module does not pretend it
is one. MEEP says so in ``mp.Absorber``'s own docstring (python/simulation.py:316):
it is "simply a scalar electric **and** magnetic conductivity that turns on
gradually within the layer", reflectionless only in the limit of a thick layer.
Two consequences drive everything here:

1. **The face keeps whatever wall it had.** MEEP excludes an ``Absorber`` from the
   boundary region outright (``_create_boundary_region_from_boundary_layers``,
   python/simulation.py:4956), so no PML sigma is installed on that face at all;
   the axis stays periodic, metallic, or mirrored exactly as it was. An absorber
   is a MATERIAL, not a boundary condition.
2. **Both conductivities, six volumes.** ``structure::set_materials`` runs
   ``FOR_D_AND_B(c) { if (mat.has_conductivity(c)) set_conductivity(c, mat); }``
   (structure.cpp:377-379), and ``geom_epsilon::has_conductivity``
   (meepgeom.cpp:1570-1573) answers true for EVERY component the moment any
   absorber profile exists. Dropping the magnetic half is not a small error:
   measured against pristine MEEP 1.33.0, electric-only lands 4.08e-03 (1-D) and
   6.46e-01 (2-D) relative Linf, magnetic-only 3.55e-03 / 5.43e-01 — three to six
   orders above this package's parity band.

Registration is per component and NOT shared. ``structure_chunk::set_conductivity``
(structure.cpp:868-895) point-samples sigma at each component's OWN Yee location
with ``multby = 0`` for a D/B component — raw sigma, never multiplied by chi1inv
and never subpixel-averaged. For a GRADED profile one shared volume is measurably
wrong: emulating it by snapping the sample position costs 1.51e-02 (node-snapped)
or 1.07e-01 (centre-snapped) relative Linf on a 2-D absorber case. Hence
:func:`absorber_conductivity` returns six arrays.

MEEP source, transcribed:

- ``geom_epsilon::set_cond_profile`` (meepgeom.cpp:744-763) — the profile table.
- ``set_materials_from_geom_epsilon`` (meepgeom.cpp:2017-2029) — the call, with
  ``dx = gv.inva * 0.5``, i.e. HALF a pixel, once per (direction, side).
- ``geom_epsilon::conductivity`` (meepgeom.cpp:1596-1625) — the interpolated
  lookup, accumulated over directions, against ``geometry_edge`` = cell/2
  (meepgeom.cpp:1981).
- ``structure_chunk::update_condinv`` (structure.cpp:697-699) — ``condinv``.
- ``step_generic.cpp:46-53, 86-100`` — the update the sigma feeds.

Verified bit-identical (0.000000e+00 relative Linf) against pristine MEEP 1.33.0
single precision in 1-D (res 20, cell z = 10, ``mp.Absorber(1)``, 2400 steps) and
in 2-D (res 13, cell 8 x 8, ``mp.Absorber(0.7)`` — a deliberately non-integer
9.1-cell layer whose profile table has N = 18).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy

from .fields import IYEE_SHIFTS

# MEEP's mp.PML defaults, inherited by mp.Absorber (python/simulation.py:227-229).
DEFAULT_R_ASYMPTOTIC = 1e-15

# adaptive_integration of DefaultPMLProfile u -> u*u over [0, 1] (meepgeom.cpp:755-757).
# MEEP integrates numerically to 1e-9 relative tolerance; the exact value is 1/3 and
# the quadrature reproduces it to well inside float64, so the constant stands in for
# the quadrature rather than repeating it. A NON-quadratic pml_profile changes this
# number and is refused by the converter rather than integrated here.
QUADRATIC_PROFILE_INTEGRAL = 1.0 / 3.0

#: The component order the six conductivity volumes are keyed by.
CONDUCTIVITY_COMPONENTS: Tuple[str, ...] = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")

_SIDES = ("low", "high")


@dataclass(frozen=True)
class AbsorberLayer:
    """One ``mp.Absorber`` face: a thickness in LENGTH units on one side of one axis.

    ``thickness`` is MEEP's ``L`` — a physical length, not a cell count, because that
    is what the profile table divides by (``prefac = -ln(R)/(4*L*integral)`` and
    ``ui = N*(x - edge)/L``). ``axis`` is 0/1/2 for x/y/z and ``side`` is ``"low"`` or
    ``"high"``; MEEP's own ``boundary_side`` enum is ``High = 0, Low = 1``, which the
    converter resolves before it gets here so no reader downstream has to remember it.

    MEEP builds one profile per (direction, side) pair — an undirected
    ``mp.Absorber(t)`` on a 3-D cell is six of these — and the lookup ADDS their
    contributions (meepgeom.cpp:1596-1625 loops over directions and adds to
    ``cond_val``), so overlapping layers in a corner accumulate rather than compete.
    """

    thickness: float
    axis: int
    side: str
    R_asymptotic: float = DEFAULT_R_ASYMPTOTIC

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError(f"absorber axis must be 0, 1 or 2, got {self.axis!r}")
        if self.side not in _SIDES:
            raise ValueError(f"absorber side must be 'low' or 'high', got {self.side!r}")
        if not math.isfinite(self.thickness) or self.thickness <= 0.0:
            raise ValueError(
                f"absorber thickness must be finite and > 0, got {self.thickness!r}; a "
                f"zero-thickness layer has no profile table at all (N = 0 divides by zero "
                f"in MEEP's own u = N*(x - edge)/L)."
            )
        if not math.isfinite(self.R_asymptotic) or not 0.0 < self.R_asymptotic < 1.0:
            raise ValueError(
                f"absorber R_asymptotic must lie in (0, 1), got {self.R_asymptotic!r}; the "
                f"grading prefactor is -ln(R), which is negative — i.e. GAIN — for R >= 1."
            )


def absorber_profile(
    thickness: float,
    resolution: float,
    R_asymptotic: float = DEFAULT_R_ASYMPTOTIC,
    profile_integral: float = QUADRATIC_PROFILE_INTEGRAL,
) -> Tuple[int, numpy.ndarray]:
    """MEEP's ``geom_epsilon::set_cond_profile`` (meepgeom.cpp:744-763), verbatim.

    Returns ``(N, prof)`` with ``prof`` of length ``N + 1``::

        dx     = gv.inva * 0.5 = 1 / (2 * resolution)      # HALF a pixel
        N      = int(L / dx + 0.5)
        prefac = -ln(R) / (4 * L * profile_integral)
        prof[i] = prefac * P(i / N)                        # P(u) = u*u

    **``dx`` is half a pixel, not a pixel.** The table is sampled at twice the grid
    resolution because the Yee lattice is: a component sits on either the integer or
    the half-integer sub-lattice, and MEEP wants a table entry for both. Reading
    ``dx = 1/resolution`` halves N and coarsens every interpolated value.

    ``N`` is the only quantity that is quantized. ``L`` itself is not: ``prefac`` and
    the lookup's ``ui = N*(x - edge)/L`` both divide by the RAW requested thickness,
    so a 9.1-cell layer (L = 0.7 at resolution 13) gets N = 18 and a profile whose
    argument runs slightly past 1 at the wall. Rounding L to a whole or half cell is
    the plausible-looking correction that makes a range of thicknesses collapse onto
    one absorber; MEEP does not do it, and neither does this.
    """
    if resolution <= 0.0 or not math.isfinite(resolution):
        raise ValueError(f"resolution must be finite and > 0, got {resolution!r}")
    half_pixel = 0.5 / resolution  # gv.inva * 0.5
    count = int(thickness / half_pixel + 0.5)
    if count < 1:
        raise ValueError(
            f"absorber thickness {thickness!r} spans {thickness / half_pixel:.4g} half-pixels "
            f"at resolution {resolution!r}, which MEEP tabulates as N = {count}; its lookup "
            f"divides by N. Use a thicker layer or a finer resolution."
        )
    prefac = (-math.log(R_asymptotic)) / (4.0 * thickness * profile_integral)
    index = numpy.arange(count + 1, dtype=numpy.float64)
    return count, prefac * (index / count) ** 2


def _sigma_along_axis(
    coordinates: numpy.ndarray,
    layer: AbsorberLayer,
    half_length: float,
    resolution: float,
) -> numpy.ndarray:
    """One layer's contribution at the given coordinates — meepgeom.cpp:1596-1625.

    High side (``edge = geometry_edge - L``, active where ``x >= edge``)::

        ui = N * (x - edge) / L

    Low side (``edge = L - geometry_edge``, active where ``x <= edge``)::

        ui = N * (edge - x) / L

    then, in both cases, ``i = int(ui)`` and a LINEAR interpolation between
    ``prof[i]`` and ``prof[i+1]``, clamped to ``prof[N]`` once ``i >= N``. The clamp
    is what a fractional layer needs: ``ui`` can exceed N at the wall.

    ``half_length`` is ``geometry_edge`` for this axis — the FULL cell length over
    two (meepgeom.cpp:1981, ``geometry_edge = geometry_lattice.size * 0.5``), not the
    stored half of a mirror-folded axis. A folded axis needs no special case here: its
    stored coordinates start one cell below the mirror plane, so the low-side test
    ``x <= L - geometry_edge`` is simply never true for them, which is exactly the
    absorber MEEP steps on the folded chunk.
    """
    count, profile = absorber_profile(
        layer.thickness, resolution, layer.R_asymptotic)
    thickness = float(layer.thickness)
    if layer.side == "high":
        edge = half_length - thickness
        inside = coordinates >= edge
        raw = count * (coordinates - edge) / thickness
    else:
        edge = thickness - half_length
        inside = coordinates <= edge
        raw = count * (edge - coordinates) / thickness
    contribution = numpy.zeros(coordinates.shape, dtype=numpy.float64)
    if not bool(numpy.any(inside)):
        return contribution
    argument = numpy.where(inside, raw, 0.0)
    lower = numpy.floor(argument).astype(numpy.int64)  # C's int() on a non-negative value.
    fraction = argument - lower
    saturated = lower >= count
    lower = numpy.minimum(lower, count - 1)
    interpolated = profile[lower] * (1.0 - fraction) + profile[lower + 1] * fraction
    contribution[...] = numpy.where(inside, numpy.where(saturated, profile[count], interpolated), 0.0)
    return contribution


def _component_coordinates(grid: Any, axis: int, shift: int,
                           indices: numpy.ndarray) -> numpy.ndarray:
    """Physical coordinates of a Yee sub-lattice along one axis, in MEEP's own arithmetic.

    ``grid_volume::operator[]`` forms the position as ``doubled * (0.5 * inva)`` from
    ONE integer (vec.hpp), where ``doubled = little_corner + 2*i + iyee_shift``. Built
    the same way here, in float64, rather than from ``grid.x/y/z`` — those are float32
    cell centres and carry a second rounding, which is enough to move a cell across
    the ``int(ui)`` boundary of the profile lookup right at the absorber's inner edge.
    """
    doubled = grid.origin_doubled(axis) + 2 * numpy.asarray(indices, dtype=numpy.int64) + shift
    return doubled * (0.5 * float(grid.dx))


def _owned_indices(grid: Any, axis: int, shift: int) -> numpy.ndarray:
    """MEEP's OWNED index for each stored cell of one axis — the seam rule.

    ``little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)`` (vec.hpp:1102-1104),
    so along one direction ``step_curl`` runs from ``i >= 1 - shift``: a half-integer
    component is stepped from index 0, an INTEGER one only from index 1. Index 0 of an
    integer component is not stepped at all — on a wrapping axis ``step_boundaries``
    fills it from the owned copy one lattice vector up, at index N, and MEEP's arrays
    carry that extra plane (they are ``N + 1`` deep where this engine stores ``N``).

    So the coefficient governing the seam is the one at ``+L/2``, not the one at
    ``-L/2``. For anything periodic the two are equal and none of this is visible. A
    ONE-SIDED absorber is where they differ by the whole of sigma_max, and it is
    measurable: a low-only ``mp.Absorber(1)`` on a periodic 6-unit cell drifts from
    2.9e-06 at t = 2 to 1.7e-04 at t = 4 to **1.5e-01** at t = 8 against CPU MEEP when
    index 0 takes its own coordinate's sigma, and sits at 2.4e-07 / 3.4e-07 / 4.0e-07
    when it takes index N's. The high-only layer is the same story at 2.8e-01 -> 3.2e-07.
    Two-sided layers are bit-unchanged either way, which is exactly why every
    symmetric case passed while this one did not.

    A folded or metallic axis does not wrap, so it has no index-N copy and this rule
    does not apply there — cell 0 of a folded axis is the mirror image of cell 2, and a
    metallic wall is held at zero outright.
    """
    stored = (grid.nx, grid.ny, grid.nz)[axis]
    if shift != 0 or not grid.axis_wraps(axis):
        return numpy.arange(stored, dtype=numpy.int64)
    return numpy.concatenate((
        numpy.array([stored], dtype=numpy.int64),
        numpy.arange(1, stored, dtype=numpy.int64),
    ))


def absorber_conductivity(
    grid: Any,
    layers: Sequence[AbsorberLayer],
) -> Dict[str, Any]:
    """Six sigma volumes — one per D and B component — for a set of absorber faces.

    Keys are :data:`CONDUCTIVITY_COMPONENTS`; every value is a float64 array of
    ``grid.shape`` on ``grid.xp``. All six are returned even when some are identically
    zero, because MEEP allocates all six too: ``has_conductivity`` is answered per
    absorber-exists, not per component (meepgeom.cpp:1570-1573).

    Float64, not float32, because a caller may still have to ADD a material sigma
    before storing. MEEP evaluates ``cnd[i] = C.conductivity(c, here)`` once in double
    and rounds once into its realnum array; returning the ramp already rounded would
    make the sum round twice. :class:`~.fields.Fields` does the single cast.

    Each component is sampled at its OWN Yee position (structure.cpp:868-895). Along
    one axis only two sub-lattices exist, so the per-axis lookup is evaluated twice
    and the six volumes are assembled by broadcasting — sigma is separable, being a
    SUM over directions (meepgeom.cpp's ``cond_val +=``) of a function of that axis's
    coordinate alone.

    Returns an empty dict when ``layers`` is empty, which is the caller's signal to
    leave the conductivity-free stepping path alone.
    """
    if not layers:
        return {}
    xp = grid.xp
    resolution = 1.0 / float(grid.dx)
    half_lengths = (grid.Lx / 2.0, grid.Ly / 2.0, grid.Lz / 2.0)
    # per_axis[axis][shift] -> 1-D float64 sigma contribution, or None where nothing absorbs.
    per_axis: list[list[Any]] = [[None, None], [None, None], [None, None]]
    for layer in layers:
        axis = layer.axis
        for shift in (0, 1):
            coordinates = _component_coordinates(
                grid, axis, shift, _owned_indices(grid, axis, shift))
            contribution = _sigma_along_axis(
                coordinates, layer, half_lengths[axis], resolution)
            existing = per_axis[axis][shift]
            per_axis[axis][shift] = contribution if existing is None else existing + contribution
    volumes: Dict[str, Any] = {}
    shape = tuple(grid.shape)
    for component in CONDUCTIVITY_COMPONENTS:
        shifts = IYEE_SHIFTS[component]
        total = numpy.zeros(shape, dtype=numpy.float64)
        for axis in range(3):
            contribution = per_axis[axis][shifts[axis]]
            if contribution is None:
                continue
            broadcast = [1, 1, 1]
            broadcast[axis] = shape[axis]
            total = total + contribution.reshape(broadcast)
        volumes[component] = xp.asarray(total)
    return volumes


def describe_layers(layers: Iterable[AbsorberLayer]) -> str:  # One log line for a lift.
    """A short, stable summary of the resolved faces, for a lift's provenance note."""
    parts = [
        f"{'xyz'[layer.axis]}{'-' if layer.side == 'low' else '+'}={layer.thickness:g}"
        for layer in layers
    ]
    return "absorber(" + ", ".join(parts) + ")" if parts else "absorber(none)"
