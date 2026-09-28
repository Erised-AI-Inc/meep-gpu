"""
``PML`` precomputes the uniaxial perfectly-matched-layer coefficients the field
update steps multiply into the curl and constitutive relations. It is a
translation of MEEP's ``structure.cpp use_pml()`` grading plus the coefficient
packing that ``step_generic.cpp step_curl`` and ``step_update_EDHB`` consume.

Inputs: a Grid and a PML thickness in grid cells — a scalar covering every face,
or a per-axis / per-side specification (plus the polynomial order and asymptotic
reflection, both at MEEP's defaults). The count need not be whole: ``mp.PML(0.5)``
at resolution 71 is 35.5 cells, which MEEP builds exactly. Outputs: eighteen 1-D
float32 coefficient vectors reshaped for broadcasting against 3-D field arrays —
``{kms, sinv, kps} x {x, y, z} x {integer, half-integer}`` — where
``kms = kappa - sigma``, ``sinv = 1/(kappa + sigma)``, ``kps = kappa + sigma``.

Pairing rule (easy to break silently): the D-field curl update and the H-field
constitutive update read the integer-position sets (``kms_x`` ...); the B-field
curl update and the E-field constitutive update read the half-integer sets
(``kms_x_h`` ...).

Thickness is per face. ``PML(grid, 8)`` is the shorthand for eight cells on all
six, and :func:`normalize_pml_thickness` also accepts ``{"z": 8}`` (absorb in z,
leave x and y periodic — the grating / photonic-crystal case), ``{"z": (8, 0)}``
(absorb at the low z face only), and the positional ``(0, 0, 8)``. MEEP spells
the same thing as ``mp.PML(thickness, direction=mp.Z, side=mp.Low)``, a linked
list of per-(direction, side) ``boundary_region`` entries; the normalized table
here is that list flattened. Each face is graded from its OWN thickness, as
MEEP's ``structure_chunk::use_pml`` is called once per side with that side's
``dx``, so ``prefac`` differs between the two faces of an asymmetric axis.

The lower X/Y face carries no PML when that axis is mirror-symmetric: the mirror
plane is a boundary condition, not an absorber. The scalar shorthand means "every
face that can carry one" and drops it silently, as it always has; a per-side
request that explicitly names that face raises instead of being quietly ignored.

Grading, verbatim from MEEP: sigma-only quadratic profile ``u**order``, kappa
fixed at 1.0 (mean_stretch = 1.0), no CFS/alpha term, R_asymptotic = 1e-15,
``prefac = -ln(R) / (4 * dx_pml * 1/(order+1))``, and the dt/2 factor baked into
the stored sigma. Units are MEEP natural units (c = 1, frequency = 1/wavelength).

MEEP source references:
- structure.cpp: use_pml(), pml_x(), pml_quadratic_profile(), boundary_region
  and its ``apply`` / ``check_ok`` (the per-direction, per-side thickness table
  and the ``thick[d][Low] + thick[d][High] > interior`` fit rule reproduced here)
- step_generic.cpp: step_curl() PML update formula
- meep_internals.hpp: KSTRIDE_DEF/DEF_k sigma indexing (k = s + 2*j over a
  2*N+2 array at half-cell resolution, s the component's iyee_shift in the
  sigma direction); the integer (s=0) and half-integer (s=1) slices are
  pre-extracted here.

Update equations (step_generic.cpp step_curl, dsig and dsigu active, no
conductivity):

    fprev = fu[i]
    fu[i] = ((kap - sig) * fu[i] - dtdx * (g1[i+s1] - g1[i] + g2[i] - g2[i+s2])) * siginv
    f[i]  = siginvu * ((kapu - sigu) * f[i] + fu[i] - fprev)

Coordinate mapping (structure.cpp pml_x). MEEP grades by distance to the wall on
its half-cell index i, one expression covering both faces and both Yee offsets:

    N_half = int(2 * n_pml + 0.5)                 # the layer EXTENT, in half cells
    n      = |i - i_wall|,  i = 2*j + s,  i_wall = 0 or 2*N
    u      = (N_half - n) / (2 * n_pml)           # graded where n < N_half

for array cell j at Yee shift s (0 integer, 1 half-integer) on a grid of N cells.
Expanding it per face gives u = (n_pml - j - s/2) / n_pml at the lower wall and
u = (j + s/2 - (N - n_pml)) / n_pml at the upper one — note the sign of the s/2
term flips, so the upper face is NOT the index-mirror of the lower face.

TWO QUANTITIES, ONLY ONE OF THEM SNAPPED. MEEP quantizes the layer's extent to a
whole half-cell (that is the whole content of the int() above; the distance n is
already a whole half-cell because the wall sits on a lattice point), and it
quantizes nothing else: the profile argument divides by the RAW requested
thickness, and so does prefac. A whole number of cells is the special case
N_half == 2*n_pml, where u collapses to 1 - n/(2*n_pml) and tops out at exactly 1
on the wall; a fractional one does not, and u reaches N_half/(2*n_pml) — 1.008755
at 26.27 cells, 0.973320 at 8.733 cells — which is where MEEP evaluates its
profile. Renormalising to (N_half - n)/N_half so that u ends at 1 is the
plausible-looking correction that collapses every thickness inside one half-cell
bin onto a single absorber; 35.4 and 35.5 cells are different absorbers in MEEP.

Rounding the request to a whole cell instead is the substitution this table used
to force, and it is measurable: against CPU MEEP a 5.5-cell one-sided layer
reproduces at 5.63e-07 while the 5- and 6-cell absorbers sit 3.26e-02 and 2.89e-02
from the same reference, and at resolution 71 a 35.5-cell layer reproduces at
1.16e-06 against 8.75e-04 / 9.27e-04 for its neighbours. Both of those thicknesses
are exact half-cell counts, where 0.5*N_half == n_pml and the two candidate
divisors of prefac coincide; a NON-half-cell case is what separates them, and
pml_fractional_cells_uneven (5.2 cells) does it at 7.34e-07 against 6.80e-03 for
the snapped divisor. Taking the extent from
round()/numpy.round is wrong only at an exact even half-cell count, because those
round half to EVEN while MEEP's int(v + 0.5) rounds half UP. Below a quarter of a
cell N_half is 0 and MEEP allocates no sigma at all (structure.cpp found_pml),
which this table reproduces by grading nothing.

One correction to that index on a WRAPPING axis: MEEP owns the half-cell indices
1..2*N, not 0..2*N-1, so the integer set is 2, 4, ... 2*N and index 0 is the
non-owned periodic image of 2*N. Cell 0 stores that lattice point and is graded
from the UPPER wall accordingly. Both accountings give u = 1 whenever the two
faces are equally deep, which is why every uniform layer is unaffected and why
only one-sided and unequal-thickness layers ever saw the difference — see
:meth:`PML._graded_sigma` for the measured cost.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import numpy

if TYPE_CHECKING:
    from .grid import Grid

BROADCAST_SHAPES = {"x": (-1, 1, 1), "y": (1, -1, 1), "z": (1, 1, -1)}  # Against (nx, ny, nz) fields.
AXIS_NAMES = ("x", "y", "z")  # Axis order of the per-face thickness table.
SIDE_NAMES = ("low", "high")  # MEEP's boundary_side, in the order each axis's faces are stored.

# A thickness is a cell count, whole or fractional: `int` whenever the request was a
# whole number of cells (which keeps every pre-existing table and repr exactly as it
# was — NOT the stored bytes, which are the same either way: measured, 84 of 84
# coefficient arrays identical when the same whole request is carried as a float),
# `float` otherwise.
FaceThickness = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def half_cell_extent(cells: float) -> int:  # MEEP: (int)(dx * (2*a) + 0.5), in HALF cells.
    """How deep one face reaches, in whole half-cells — the only quantity MEEP snaps.

    Transcribed from ``pml_x`` (structure.cpp:625-628, byte-identical in v1.29.0 and
    v1.33.0)::

        (int)(dx * (2 * a) + 0.5)

    with ``dx`` the requested thickness in length units and ``a`` the resolution, so
    ``dx * (2 * a) == 2 * cells``. That identity is exact rather than approximate for
    the converter's route into this function: it hands over ``cells = thickness *
    resolution``, and ``2 * fl(t*r) == fl(t*(2*r))`` because scaling by a power of two
    is exact. Nothing else about the layer is quantized — see the module docstring.

    Two rounding traps, both silent:

    * The C cast truncates, so ``v + 0.5`` is ROUND-HALF-UP. ``round`` and
      ``numpy.round`` round half to EVEN: ``round(70.5) == 70`` where MEEP gives 71.
      They disagree only at an exact even half-cell count — rare, and invisible at
      every whole and half cell the suite otherwise exercises.
    * Below a quarter of a cell this returns 0, and MEEP then builds no PML at all:
      ``use_pml`` scans for a point with ``x > 0``, finds none, and returns before
      allocating sigma (structure.cpp:648-654). Ceiling to one half cell would absorb
      where MEEP does not; raising would refuse a run MEEP accepts.
    """
    return int(2.0 * float(cells) + 0.5)


def absorbing_cells(cells: float) -> int:  # Whole array cells one face touches: ceil(N_half / 2).
    """How many whole array cells one face grades — what the interior consumers need.

    The integer-position sample of array cell ``j`` sits at half-cell distance ``2*j``
    from the wall and is graded while ``2*j < N_half``, so the face reaches
    ``ceil(N_half / 2)`` cells. Equal to the thickness itself for a whole-cell face,
    which is why every pre-existing interior bound is untouched.

    Truncating with ``int(cells)`` instead — which is what ``PML.interior_slice`` and
    ``DFTMonitor.set_region_from_pml`` did while the count was always whole — leaves
    the cell carrying the second-largest sigma of a 5.5-cell layer inside the region
    called "the PML interior". The DFT spectrum that comes back is smooth, plausible
    and low; the near-to-far guard in dft.py records 7.3% and 44% of radiated power
    lost to exactly that class of mistake.
    """
    return (half_cell_extent(cells) + 1) // 2


class PMLThickness(NamedTuple):
    """A normalized PML thickness request: the per-face table plus how it was spelled.

    Attributes:
        faces: ``((x_low, x_high), (y_low, y_high), (z_low, z_high))`` in grid cells.
        uniform: The scalar the whole layer was requested with, or None when the
            caller named faces individually. The distinction is not cosmetic —
            the scalar shorthand silently drops the face a mirror plane occupies,
            and it is the spelling every pre-existing PML number was measured
            with, so paths that must stay byte-identical branch on it.
    """

    faces: FaceThickness
    uniform: Optional[float]


def _face_cells(value: Any, where: str) -> float:  # One face's thickness: a non-negative cell count.
    """Coerce one face's thickness to a non-negative cell count, or raise naming the face.

    Whole requests come back as ``int`` and fractional ones as ``float``. The split is
    deliberate rather than cosmetic: it keeps ``repr``, the per-face table and every
    downstream integer consumer on exactly the values they held before fractional
    thicknesses were accepted, so a whole-cell layer cannot pick up a float-formatted
    message or a float-typed slice bound on the way through.

    ``True`` is an ``int`` in Python and would silently become a one-cell layer, so
    booleans are rejected by name rather than by their numeric value. Non-finite is
    rejected separately from the type, because ``inf`` passes ``isinstance`` and
    ``half_cell_extent`` would carry it into an ``int()`` that raises somewhere far
    from the request that caused it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, numpy.integer, numpy.floating)):
        raise ValueError(
            f"PML thickness for {where} must be a number of grid cells, got {value!r}."
        )
    number = float(value)
    if not numpy.isfinite(number):
        raise ValueError(
            f"PML thickness for {where} must be a finite number of grid cells, got {value!r}."
        )
    if number < 0:
        raise ValueError(f"PML thickness for {where} must be non-negative, got {value!r}.")
    return int(number) if number == int(number) else number


def _axis_index(key: Any) -> Optional[int]:  # Axis position of 'x'/'X'/0, or None when unrecognised.
    if isinstance(key, str):
        lowered = key.lower()
        return AXIS_NAMES.index(lowered) if lowered in AXIS_NAMES else None
    if isinstance(key, (int, numpy.integer)) and not isinstance(key, bool) and 0 <= int(key) <= 2:
        return int(key)
    return None


def _as_plain(value: Any) -> Any:  # NumPy arrays become lists so the sequence paths see them.
    return value.tolist() if isinstance(value, numpy.ndarray) else value


def _normalize_axis_entry(value: Any, axis_name: str) -> tuple[float, float]:  # One axis's (low, high) faces.
    """Expand one axis's entry into ``(low, high)`` cell counts.

    Accepts a scalar (both sides), a ``(low, high)`` pair, or a mapping keyed by
    ``low`` / ``high`` where an omitted side means zero — MEEP's
    ``mp.PML(t, direction=d)`` versus ``mp.PML(t, direction=d, side=mp.Low)``.
    """
    value = _as_plain(value)
    if isinstance(value, Mapping):
        sides = [0, 0]
        for key, entry in value.items():
            name = str(key).lower()
            if name not in SIDE_NAMES:
                raise ValueError(
                    f"Unknown PML side {key!r} on the {axis_name} axis; expected any of "
                    f"{list(SIDE_NAMES)}."
                )
            sides[SIDE_NAMES.index(name)] = _face_cells(entry, f"the {axis_name} {name} face")
        return (sides[0], sides[1])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pair = tuple(value)
        if len(pair) != 2:
            raise ValueError(
                f"PML thickness for the {axis_name} axis must be a scalar or a (low, high) pair, "
                f"got {value!r} with {len(pair)} entries."
            )
        return (
            _face_cells(pair[0], f"the {axis_name} low face"),
            _face_cells(pair[1], f"the {axis_name} high face"),
        )
    cells = _face_cells(value, f"the {axis_name} axis")
    return (cells, cells)


def normalize_pml_thickness(spec: Any) -> PMLThickness:
    """Expand a PML thickness request into the per-face table the layer is built from.

    This is the one place the accepted spellings are defined, so the driver and the
    coefficient table cannot drift apart on what a request means. In grid cells,
    whole or fractional (``{"z": 35.5}`` is ``mp.PML(0.5)`` at resolution 71):

        8                      -> eight cells on all six faces (the historical form)
        {"z": 8}               -> eight cells on both z faces, none on x or y
        {"z": (8, 0)}          -> eight cells at the low z face only
        {"z": {"high": 8}}     -> eight cells at the high z face only
        (0, 0, 8)              -> positional per axis, same as {"z": 8}
        ((0, 0), (0, 0), (8, 4)) -> positional per axis and side

    Axis keys are ``x``/``y``/``z`` (either case) or 0/1/2; side keys are ``low``
    and ``high``, MEEP's ``mp.Low`` / ``mp.High``. Anything else raises: a mistyped
    axis key that defaulted to zero would build a run with no absorber where the
    caller asked for one, and the field would still look like a field.
    """
    spec = _as_plain(spec)
    if isinstance(spec, Mapping):
        faces = [(0, 0), (0, 0), (0, 0)]
        seen: dict[int, Any] = {}
        for key, value in spec.items():
            axis = _axis_index(key)
            if axis is None:
                raise ValueError(
                    f"Unknown PML axis key {key!r}; expected any of {list(AXIS_NAMES)} (or 0, 1, 2). "
                    f"Sides are named inside an axis entry, e.g. {{'z': {{'low': 8}}}}."
                )
            if axis in seen:
                raise ValueError(
                    f"PML axis {AXIS_NAMES[axis]} is specified twice, as {seen[axis]!r} and "
                    f"{key!r}; give each axis once."
                )
            seen[axis] = key
            faces[axis] = _normalize_axis_entry(value, AXIS_NAMES[axis])
        return PMLThickness((faces[0], faces[1], faces[2]), None)
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        entries = tuple(spec)
        if len(entries) != 3:
            raise ValueError(
                f"A positional PML thickness must have one entry per axis (x, y, z), got {spec!r} "
                f"with {len(entries)} entries. Name the axes instead, e.g. {{'z': 8}}."
            )
        return PMLThickness(
            tuple(_normalize_axis_entry(entries[axis], AXIS_NAMES[axis]) for axis in range(3)),
            None,
        )
    cells = _face_cells(spec, "every face")
    return PMLThickness(((cells, cells), (cells, cells), (cells, cells)), cells)


@dataclass
class PML:
    """MEEP-compatible uniaxial PML (UPML) coefficient table.

    Translation of MEEP's structure.cpp use_pml(). Stores, per axis and per Yee
    offset, the three combinations the update kernels need:
        kms  = kap - sig,  sinv = 1/(kap + sig),  kps = kap + sig
    with sig = 0.5 * dt * prefac * profile(u) and kap = 1.0, shaped for
    broadcasting with 3-D field arrays.

    Attributes (set by ``__post_init__``, alongside the coefficient arrays):
        thickness_by_face: ``((x_low, x_high), (y_low, y_high), (z_low, z_high))``
            in grid cells — the authoritative geometry of the layer.
        uniform_thickness: The scalar the layer was requested with, or None for a
            per-face request. Callers that must reproduce the historical uniform
            behaviour byte for byte branch on this.
        thickness: The DEEPEST face, in cells. It is the back-compatible scalar —
            ``pml.thickness != 0`` still means "this layer absorbs somewhere" —
            and is the right conservative depth for a whole-run margin, but it is
            not the thickness of any particular face. Read
            :attr:`thickness_by_face` (or :meth:`axis_faces`) for that.

    Cell counts may be fractional. They are ints whenever the request was whole, so a
    layer spelled in whole cells carries exactly the values, types and coefficient
    bytes it always has; see :func:`half_cell_extent` for what MEEP does with the rest.
    """

    grid: "Grid"
    thickness: Any  # Cells: a scalar for all six faces, or a per-axis/per-side spec.
    order: int = 2  # Polynomial grading order (MEEP default 2, quadratic)
    R_asymptotic: float = 1e-15  # MEEP default asymptotic reflection
    # Derived in __post_init__, but declared as fields so that two layers differing only
    # in how their thickness is distributed do not compare equal: `thickness` collapses
    # to the deepest face, and a generated __eq__ built on it alone would call a z-only
    # layer equal to a uniform one.
    thickness_by_face: FaceThickness = field(
        default=((0, 0), (0, 0), (0, 0)), init=False, repr=False)
    uniform_thickness: Optional[float] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):  # Validate the layer geometry and build every coefficient set.
        if self.order < 1:
            raise ValueError(f"PML grading order must be at least 1, got {self.order}")
        if not 0.0 < self.R_asymptotic < 1.0:
            raise ValueError(
                f"R_asymptotic must lie in (0, 1), got {self.R_asymptotic}"
            )
        request = normalize_pml_thickness(self.thickness)
        self.uniform_thickness = request.uniform
        self.thickness_by_face = self._resolve_mirror_faces(request)
        # MEEP structure.cpp boundary_region::check_ok: the two faces of one axis may
        # not overlap (thick[d][Low] + thick[d][High] > interior fails there too).
        # Compared on the RAW thicknesses, as MEEP compares them — it checks lengths
        # against gv.interior(), never a snapped extent — so a pair summing to exactly
        # the axis is legal here for the same reason it is there.
        for axis, n_cells in enumerate((self.grid.nx, self.grid.ny, self.grid.nz)):
            low, high = self.thickness_by_face[axis]
            if low + high > n_cells:
                raise ValueError(
                    f"PML thickness (low={low}, high={high}) does not fit on the "
                    f"{AXIS_NAMES[axis]} axis ({n_cells} cells)"
                )
        self.thickness = max(max(faces) for faces in self.thickness_by_face)
        self._compute_coefficients()

    def _resolve_mirror_faces(self, request: PMLThickness) -> FaceThickness:
        """Drop the lower face on a mirrored axis, or refuse it if it was named.

        Cell 0 of a folded axis lies on the mirror plane, which is a boundary
        condition rather than an outer wall; an absorber there would eat the
        mirrored half of the domain. The scalar shorthand means "every face that
        can carry a layer" and skips it, as it always has. A per-side request that
        names that face is a different statement — the caller asked for an absorber
        at a specific place that cannot hold one — and raises rather than being
        honoured somewhere else or silently discarded.

        The rule is per axis and reads ``Grid.is_mirrored``, so a folded Z drops
        its low face exactly as a folded X does; enumerating X and Y alone left a
        folded Z run graded from BOTH walls, absorbing across the mirror plane.
        """
        faces = list(request.faces)
        for axis in range(3):
            low, high = faces[axis]
            if not self.grid.is_mirrored(axis) or low == 0:
                continue
            if request.uniform is None:
                raise ValueError(
                    f"PML requested at the {AXIS_NAMES[axis]} low face ({low} cells), but the "
                    f"{AXIS_NAMES[axis].upper()} axis is mirror-symmetric: cell 0 lies on the "
                    f"mirror plane, which is a boundary condition, not an absorber. Ask for the "
                    f"high face only, e.g. {{'{AXIS_NAMES[axis]}': {{'high': {low}}}}}."
                )
            faces[axis] = (0, high)  # Scalar shorthand: every face that can carry one.
        return (faces[0], faces[1], faces[2])

    def axis_faces(self, axis: int) -> tuple[float, float]:  # (low, high) thickness in cells for one axis.
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0 (X), 1 (Y), or 2 (Z), got {axis!r}")
        return self.thickness_by_face[axis]

    def axis_has_pml(self, axis: int) -> bool:  # True when either face of this axis absorbs.
        low, high = self.axis_faces(axis)
        return low > 0 or high > 0

    @property
    def is_active(self) -> bool:  # True when any of the six faces carries an absorber.
        """Whether this layer absorbs anywhere.

        A layer of zero thickness on every face is not an error — it is simply no
        absorber — and the stepping kernels take the no-PML path on it, so a
        zero-thickness table steps bit-identically to ``pml=None``.
        """
        return any(low > 0 or high > 0 for low, high in self.thickness_by_face)

    def prefac(self, cells: Optional[float] = None) -> float:  # MEEP: -ln(R) / (4 * dx_pml * profile_integral).
        """Grading prefactor for one face's thickness (default: the deepest face).

        MEEP calls ``structure_chunk::use_pml`` once per (direction, side) with that
        side's own ``dx``, so ``prefac`` is a property of a FACE, not of the layer:
        the shallower face of an asymmetric axis is graded more steeply so that both
        reach the same asymptotic reflection.

        The thickness here is the RAW request, never :func:`half_cell_extent`'s snapped
        version — structure.cpp:635 divides by the ``dx`` the caller asked for. Using
        the extent instead scales sigma by ``c / (0.5 * N_half)``, and since
        ``|0.5 * N_half - c| <= 0.25`` by construction that is an error of up to a
        QUARTER of a cell in ``1/dx`` — not half, and not a fixed percentage: it decays
        like ``0.25 / c``, so a thin layer is where it bites. Measured peak-sigma shifts:
        +5.0000% at 2.1 cells, +4.0000% at 5.2, -3.6364% at 5.3, +0.2506% at
        cavity-farfield.py's 66.667 cells and -0.0402% at cylinder_cross_section.py's
        27.489. It is exactly ZERO at every whole cell and at every exact half cell,
        35.5 at resolution 71 among them, because ``0.5 * N_half == c`` there — which is
        why the substitution has to be measured somewhere else. Somewhere else is
        ``pml_fractional_cells_uneven`` in test_from_meep.py: 5.2 cells against CPU MEEP
        goes from 7.34e-07 to 6.80e-03 under it, while both exact-half-cell parity cases
        are bit-identical under it — on the converter's own route ``mp.PML(0.55)`` at
        resolution 10 and ``mp.PML(0.5)`` at 71 give ``0.5 * N_half == c`` exactly, so
        the two divisors are the same double. In the asymptotic reflection the same 4.0000% is
        ``R -> R**1.04``, which at ``R = 1e-15`` is a factor 3.98 (measured).
        """
        n_pml = self.thickness if cells is None else cells
        dx_pml = n_pml * self.grid.dx
        if dx_pml <= 0:
            return 0.0
        profile_integral = 1.0 / (self.order + 1)  # integral_0^1 u^order du
        return float(-numpy.log(self.R_asymptotic) / (4.0 * dx_pml * profile_integral))

    def _graded_sigma(
        self, n_cells: int, faces: tuple[float, float], offset: float, scale_factor: float,
        wraps: bool = True, wall_high: Optional[int] = None,
    ) -> numpy.ndarray:
        """Fill one axis with the graded conductivity ``scale_factor * prefac * profile(u)``.

        Translation of the sigma-filling loop in structure.cpp use_pml(). MEEP
        stores sigma over 2*N+2 half-cell entries; this fills only the entries for
        the requested offset (0.0 = integer positions / even indices, 0.5 =
        half-integer positions / odd indices).

        Both faces come from MEEP's single ``pml_x`` grading, the distance from the
        wall measured on the half-cell index ``i = 2*j + offset*2``:

            N_half = int(2 * n_pml + 0.5)                    # the snapped extent
            u      = (N_half - |i - i_wall|) / (2 * n_pml)   # where |i - i_wall| < N_half

        with ``i_wall`` 0 (lower) or ``wall_high`` (upper) — ``2 * n_cells`` by
        default, which is the window top everywhere except a folded PERIODIC axis
        (either count parity), where the stored array runs one cell past MEEP's
        corner (the ``big_corner`` plane and its ghost slot;
        ``Grid.owned_cells`` vs ``Grid.stored_cells``) and the wall is
        ``2 * owned_cells`` (:meth:`_wall_high`). Deriving the upper face as
        the index-mirror of the lower one instead is wrong by a full cell at integer
        positions. The two faces are not index mirrors on a Yee lattice, while the
        half-integer sets, sitting half a cell inside both walls, are. Keep the one
        expression: the per-face algebra is what drifted apart before.

        THE EXTENT AND THE NORMALIZATION MOVE SEPARATELY. Only the numerator is
        quantized; the denominator and ``prefac`` carry the raw thickness
        (structure.cpp:635,683). What a "reasonable" generalization loses is that ``u``
        is not bounded by 1: it reaches ``N_half / (2 * n_pml)``, which is 1.008755 at
        26.27 cells and 0.973320 at 8.733, and renormalising it back to 1 collapses
        every thickness inside one half-cell bin onto a single absorber. That, together
        with ``prefac``, is the whole of the fractional information — which is why
        ``test_pml_grading_matches_meeps_distance_to_wall_rule`` now draws thicknesses
        on either side of a half cell (5.2 rounds its extent down, 5.3 up) rather than
        only half-integers, where a floor() and a ceil() each look right half the time.

        THE EXTENT TEST IS NOT INDEPENDENTLY OBSERVABLE. ``|i - i_wall| < N_half`` is
        MEEP's own ``x > 0`` (structure.cpp:682) written on the quantized numerator, and
        for the ``u`` above it is the SAME predicate: measured over 1,600,000 samples
        spanning 4000 thicknesses from 0.01 to 40.00 cells, ``distance < extent`` and
        ``u > 0`` never disagree once. It differs from the PRE-fractional test
        ``distance < 2 * n_pml`` only at the sample sitting exactly at distance
        ``N_half``, which the raw test admits and the snapped one does not whenever
        ``N_half < 2*n_pml``; and that difference is unobservable too: over the same
        sweep (8000 sigma arrays,
        both Yee offsets) the two rules disagree about membership at 3816 samples and
        produce ZERO differing sigma arrays, because the disputed sample carries
        ``u == 0.0`` exactly and ``0.0 ** order`` is 0 for any ``order >= 1``. Keep the
        snapped form because it is MEEP's, not because a test can tell.

        WHOLE CELLS MUST NOT MOVE. ``u`` is written as
        ``1 + ((N_half - 2*n_pml) - n) / (2*n_pml)`` rather than as the plainer
        ``(N_half - n) / (2*n_pml)``: for a whole-cell face ``N_half - 2*n_pml`` is
        exactly 0.0 and the rest is the pre-fractional expression operation for
        operation, so the whole-cell path provably never enters the new arithmetic at
        all. That is a DEFENSIVE spelling, not a pinned one, and the distinction is
        measured: the plainer form really is a different double — 2952 of 4236
        whole-cell sigma arrays swept over three resolutions, three cells and every
        (low, high) pair up to 8, and 247 of the first 256 integer thicknesses in ``u``
        itself — but the float32 store absorbs every one of them. 0 of 388,600 stored
        float32 entries differ, 0 of the 21 combinations
        ``test_a_whole_cell_layer_is_byte_identical_under_fractional_support`` pins
        differ, and substituting the plain form here passes all 142 tests in
        test_grid_pml.py. The byte pins hold the whole-cell path against a change big
        enough to reach float32 (2.5e-06 relative in ``R``, measured); they do not hold
        it against this reassociation, and claiming they do would be claiming a guard
        that is not there.

        WHICH LATTICE POINT INDEX 0 IS (``wraps``). On a wrapping axis MEEP owns the
        half-cell indices ``1 .. 2*N``, not ``0 .. 2*N - 1``: ``LOOP_OVER_VOL`` runs
        from ``little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)`` to
        ``big_corner()`` (meep/vec.hpp:1102, step_db.cpp:124), so the half-integer
        samples are 1, 3, ... 2N-1 and the INTEGER ones are 2, 4, ... 2N. Index 0 —
        the little corner itself — is never owned; it is the non-owned periodic image
        of index 2N. This array stores that lattice point once, at cell 0, so cell 0
        of the integer set must be graded as MEEP's ``2*N``: from the UPPER wall, with
        the upper face's own thickness and prefac. Grading it from the lower wall
        instead is invisible whenever both faces are equally deep — u = 1 either way —
        which is every uniform layer and therefore every pinned number this table had.
        It is worth a factor of ``1/(kappa+sigma)`` at that one plane otherwise, and
        the plane in question is exactly where a one-sided layer's wrap re-enters the
        absorber: measured against CPU MEEP, a one-sided z layer with a live wave on
        the un-absorbed face went from 1.16e-01 to 5.5e-07, and an asymmetric (10, 4)
        two-sided layer from 9.9e-05 to 4.9e-07.

        A mirror-folded axis does NOT wrap, and its cell 0 straddles the mirror plane
        rather than imaging the far face; MEEP's halved ``grid_volume`` puts the owned
        integer cells at 2, 4, ... 2N of a grid whose owned window is
        ``n_full - n_full//2 + 1`` cells (a folded periodic axis stores one
        more past the wall; the wall itself is ``_wall_high``), so
        the plain ``2*j`` index is already MEEP's there and cell 0 is simply unowned
        (``stepping._mask_non_owned_cells`` zeroes its curl and the symmetry pass
        overwrites it). Pass ``wraps=False`` for such an axis. The flag must stay in
        step with ``stepping._boundary_kinds``, which is the other half of the same
        statement: every unmirrored axis wraps, absorber or not.

        Each face carries its own ``n_pml`` and therefore its own ``prefac``. The
        two graded regions cannot overlap (``__post_init__`` refuses ``low + high >
        n_cells``, and rounding the two extents apart cannot cross that bound), so
        writing them in turn — MEEP assigns rather than accumulates too — leaves
        neither face able to overwrite the other's cells.
        """
        if offset not in (0.0, 0.5):
            raise ValueError(f"PML Yee offset must be 0.0 or 0.5, got {offset!r}")
        sig = numpy.zeros(n_cells, dtype=numpy.float32)
        half_index = 2 * numpy.arange(n_cells) + int(2 * offset)  # MEEP's i = 2*j + s.
        if wraps:
            # Move the wrapped representative from [0, 2N-1] onto MEEP's owned [1, 2N].
            # Only index 0 moves, and only for the integer set: the half-integer set
            # already starts at 1, so this is a no-op there.
            half_index = (half_index - 1) % (2 * n_cells) + 1
        if wall_high is None:
            wall_high = 2 * n_cells
        for wall, n_pml in ((0, faces[0]), (wall_high, faces[1])):
            if n_pml <= 0:  # This face carries no absorber; MEEP simply never calls use_pml for it.
                continue
            scale = scale_factor * self.prefac(n_pml)
            extent = half_cell_extent(n_pml)  # MEEP: (int)(dx*(2a) + 0.5), in half cells.
            distance = numpy.abs(half_index - wall)  # MEEP: (int)(|bloc - here|*(2a) + 0.5).
            overhang = extent - 2.0 * n_pml  # Exactly 0.0 for a whole-cell face.
            u = 1.0 + (overhang - distance) / (2.0 * n_pml)
            inside = distance < extent  # MEEP's "if (x > 0)", on the quantized numerator.
            sig[inside] = scale * u[inside] ** self.order
        return sig

    def _wall_high(self, axis: int) -> int:  # Upper-wall half-cell index for one axis.
        """The window top in half-cell indices — ``2 * Grid.owned_cells``, not ``2 * stored``.

        The two differ only on a folded PERIODIC axis (either count parity), whose
        stored array carries the ``big_corner`` plane AND its ghost slot past
        MEEP's ``big_corner``; grading from the stored end there would slide the
        whole upper absorber one cell out of MEEP's positions. The ``getattr``
        fallback serves the stub grids of the test suite, whose stored counts are
        MEEP's own.
        """
        reader = getattr(self.grid, "owned_cells", None)
        if callable(reader):
            return 2 * reader(axis)
        return 2 * (self.grid.nx, self.grid.ny, self.grid.nz)[axis]

    def axis_wraps(self, axis: int) -> bool:  # Is this axis periodic rather than terminated?
        """Whether one axis wraps, which decides where its index 0 lattice point sits.

        Deferred to ``Grid.axis_wraps`` — the one definition — rather than restated
        as ``not is_mirrored``. The restatement was written while mirrors were the
        only non-wrapping axis and silently mis-graded the CYLINDRICAL radial axis:
        its integer position 0 is the AXIS, not the periodic image of the outer
        wall, but the wrap grading put the high-face layer's peak sigma there.
        Measured at m=1 with an r-high absorber: the H-side constitutive along r —
        the first user of the integer-position r coefficients, since m=0 never
        exercises them — read kps[0] = 2.295 where the axis needs exactly 1, and
        the stored Hr on the axis row came out 2.3x and growing against a stored
        Br that was exact to 1e-13. Only a wrapping axis has an index 0 that is
        the periodic image of its far face, which is what :meth:`_graded_sigma`
        needs to know; a folded axis terminates at its plane and the radial axis
        at r = 0.
        """
        return self.grid.axis_wraps(axis)

    def _compute_sigma_profile(
        self, n_cells: int, faces: tuple[float, float], offset: float, wraps: bool,
        wall_high: Optional[int] = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Compute the PML coefficient triple for one axis and one Yee offset.

        Args:
            n_cells: Total cells on this axis.
            faces: (low, high) PML thickness in cells for this axis.
            offset: 0.0 for integer positions, 0.5 for half-integer positions.
            wraps: Whether this axis is periodic (see :meth:`_graded_sigma`).
            wall_high: Upper-wall half-cell index (:meth:`_wall_high`); defaults
                to ``2 * n_cells`` inside :meth:`_graded_sigma`.

        Returns:
            (kappa_minus_sigma, sigma_inverse, kappa_plus_sigma)
        """
        # MEEP: sig[d][idx] = 0.5 * dt * prefac * profile(u); kap[d][idx] = 1.0 for
        # mean_stretch = 1.0, so kappa keeps its interior value everywhere.
        sig = self._graded_sigma(n_cells, faces, offset, 0.5 * self.grid.dt, wraps,
                                 wall_high)
        kap = numpy.ones(n_cells, dtype=numpy.float32)

        sigma_inverse = 1.0 / (kap + sig)  # MEEP: siginv[d][idx] = 1 / (kap + sig)
        kappa_minus_sigma = kap - sig  # step_curl:  (kap[k] - sig[k]) * f
        kappa_plus_sigma = kap + sig  # update_eh:  f += (kap+sig)*fw - (kap-sig)*fwprev
        return kappa_minus_sigma, sigma_inverse, kappa_plus_sigma

    def _reshape_for_broadcast(self, arr: numpy.ndarray, axis: str) -> Any:
        """Reshape a 1-D coefficient vector for broadcasting along one field axis.

        Returns an ``xp`` (NumPy or CuPy) float32 array of shape (n,1,1) for 'x',
        (1,n,1) for 'y', or (1,1,n) for 'z'.
        """
        shape = BROADCAST_SHAPES.get(axis)
        if shape is None:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        xp = self.grid.xp
        return xp.asarray(arr, dtype=xp.float32).reshape(shape)

    def _compute_coefficients(self):  # Build the integer and half-integer sets for all axes.
        """Populate every coefficient attribute, matching MEEP's use_pml() layout.

        MEEP references: structure.cpp use_pml(), meep/vec.hpp iyee_shift(),
        meep_internals.hpp KSTRIDE_DEF/DEF_k.

        Integer positions (iyee_shift = 0) serve the D-field curl PML and the
        H-field constitutive update; half-integer positions (iyee_shift = 1) serve
        the B-field curl PML and the E-field constitutive update, because B-fields
        are half-integer in both transverse directions and E-fields are
        half-integer in their own component direction.

        Every axis reads its own ``(low, high)`` entry, which ``_resolve_mirror_faces``
        has already zeroed at a mirror plane, so an axis the caller left out of the
        request simply gets sigma = 0 everywhere and its coefficients collapse to
        the identity (kms = kps = 1, sinv = 1) — the pass-through the interior uses.
        """
        faces_x, faces_y, faces_z = self.thickness_by_face
        wrap_x, wrap_y, wrap_z = (self.axis_wraps(axis) for axis in range(3))

        wall_x, wall_y, wall_z = (self._wall_high(axis) for axis in range(3))

        kms_x, sinv_x, kps_x = self._compute_sigma_profile(self.grid.nx, faces_x, 0.0, wrap_x, wall_x)
        kms_y, sinv_y, kps_y = self._compute_sigma_profile(self.grid.ny, faces_y, 0.0, wrap_y, wall_y)
        kms_z, sinv_z, kps_z = self._compute_sigma_profile(self.grid.nz, faces_z, 0.0, wrap_z, wall_z)
        self.kms_x = self._reshape_for_broadcast(kms_x, "x")
        self.sinv_x = self._reshape_for_broadcast(sinv_x, "x")
        self.kps_x = self._reshape_for_broadcast(kps_x, "x")
        self.kms_y = self._reshape_for_broadcast(kms_y, "y")
        self.sinv_y = self._reshape_for_broadcast(sinv_y, "y")
        self.kps_y = self._reshape_for_broadcast(kps_y, "y")
        self.kms_z = self._reshape_for_broadcast(kms_z, "z")
        self.sinv_z = self._reshape_for_broadcast(sinv_z, "z")
        self.kps_z = self._reshape_for_broadcast(kps_z, "z")

        kms_x_h, sinv_x_h, kps_x_h = self._compute_sigma_profile(self.grid.nx, faces_x, 0.5, wrap_x, wall_x)
        kms_y_h, sinv_y_h, kps_y_h = self._compute_sigma_profile(self.grid.ny, faces_y, 0.5, wrap_y, wall_y)
        kms_z_h, sinv_z_h, kps_z_h = self._compute_sigma_profile(self.grid.nz, faces_z, 0.5, wrap_z, wall_z)
        self.kms_x_h = self._reshape_for_broadcast(kms_x_h, "x")
        self.sinv_x_h = self._reshape_for_broadcast(sinv_x_h, "x")
        self.kps_x_h = self._reshape_for_broadcast(kps_x_h, "x")
        self.kms_y_h = self._reshape_for_broadcast(kms_y_h, "y")
        self.sinv_y_h = self._reshape_for_broadcast(sinv_y_h, "y")
        self.kps_y_h = self._reshape_for_broadcast(kps_y_h, "y")
        self.kms_z_h = self._reshape_for_broadcast(kms_z_h, "z")
        self.sinv_z_h = self._reshape_for_broadcast(sinv_z_h, "z")
        self.kps_z_h = self._reshape_for_broadcast(kps_z_h, "z")

    def interior_slice(self):
        """Index bounds of the non-absorbing interior as (x0, x1, y0, y1, z0, z1).

        Each face is trimmed by its own thickness, so a face carrying no absorber —
        an axis left out of the request, or the lower face of a mirrored axis, whose
        cell 0 is the mirror plane — keeps every one of its cells. Trimming a
        periodic axis by the deepest layer in the run would discard live interior.

        A fractional face is trimmed by :func:`absorbing_cells`, the whole cells it
        actually grades, which is ``ceil`` of the half-cell extent and equals the
        thickness itself whenever that is whole. Truncating instead would leave the
        deepest-but-one graded cell of a 5.5-cell layer inside the "interior".
        """
        (x_low, x_high), (y_low, y_high), (z_low, z_high) = (
            tuple(absorbing_cells(side) for side in axis) for axis in self.thickness_by_face
        )
        # The high trim is measured from the WALL, which sits at the owned count
        # (`_wall_high` / Grid.owned_cells): the stored rows a folded periodic
        # axis keeps past it — the big_corner plane and its ghost slot —
        # are inside the absorber, not interior.
        owned = tuple(self._wall_high(axis) // 2 for axis in range(3))
        return (
            x_low, owned[0] - x_high,
            y_low, owned[1] - y_high,
            z_low, owned[2] - z_high,
        )

    def sigma_bites(self, axis: str, half_integer: bool = False) -> numpy.ndarray:
        """Per stored cell: does this axis's graded sigma bite there, at this Yee offset?

        Host boolean array, built by the same :meth:`_graded_sigma` pass that fills
        the stored coefficient sets — so this predicate and the stepping recurrence
        cannot disagree about where the layer acts. ``half_integer`` selects the Yee
        offset exactly as the kernels do: the D-side curl reads the integer sets, the
        B-side curl the half-integer ones (module docstring, pairing rule).

        MEEP's grading puts sigma EXACTLY zero at the sample sitting at the layer's
        snapped extent (``u = 0`` there, structure.cpp:682), so a cell "touching the
        inner edge" reads False here with no tolerance — which is why the source
        truth table in the absorber-placement evidence packet sorts that row as exact.
        """
        if axis not in AXIS_NAMES:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        index = AXIS_NAMES.index(axis)
        n_cells = (self.grid.nx, self.grid.ny, self.grid.nz)[index]
        sigma = self._graded_sigma(
            n_cells, self.thickness_by_face[index], 0.5 if half_integer else 0.0, 1.0,
            self.axis_wraps(index), self._wall_high(index),
        )
        return sigma != 0.0

    def in_pml_chunk(self, axis: str, half_integer: bool = False) -> numpy.ndarray:
        """Per stored cell: is its sample OWNED by a chunk that CARRIES this axis's layer?

        MEEP selects the PML recurrence per structure chunk from ``sigsize``
        (step_db.cpp:56-59), and ``sigsize[d] > 1`` for every cell of a chunk that
        overlaps the layer — including cells whose own sigma is exactly zero. The
        chunk is the per-side effort volume ``structure::use_pml`` registers
        (structure.cpp:509-524)::

            pml_volume.set_num_direction(d, int(dx * user_volume.a + 1 + 0.5));

        ``span = int(cells + 1.5)`` array cells measured from the wall, where
        ``choose_chunkdivision`` splits. Ownership then follows MEEP's owned-corner
        arithmetic (vec.hpp:1100-1104, ``little_owned_corner0 = little_corner + 2 -
        iyee_shift``): a chunk spanning doubled indices ``[c0, c1]`` owns the
        integer samples ``c0 + 2 .. c1`` and the half-integer samples ``c0 + 1 ..
        c1 - 1``, so the INTEGER lattice plane on the chunk's own upper boundary
        belongs to it, not to its interior neighbour. Per Yee offset on this axis:

        * integer (shift 0): low face owns cells ``1 .. span``; high face owns the
          top ``span`` planes — stored cells ``n - span + 1 .. n - 1`` plus, on a
          wrapping axis, stored cell 0 (the image of the wall plane it owns).
        * half-integer (shift 1): low face owns cells ``0 .. span - 1``; high face
          ``n - span .. n - 1``.

        Measured discriminators (12x12 um cell, res 20, PML(2) all faces, CW Ez at
        x = -5, whole-volume complex relative L2 against CPU MEEP): a deposit at
        y = -4.0 (cell 40, sigma exactly 0) and one at y = -3.975 (cells 40 + 41)
        both step the BOTH-ACTIVE recurrence — mirroring either reads 2.3e-01 …
        4.6e-01 where leaving them split reads ~7e-07 — while y = -3.9 (cell 42)
        steps the unsplit one: 4.9583e-01 unmirrored, 5.6135e-07 mirrored. This is
        the boundary :meth:`sigma_bites` cannot see, because those cells' own sigma
        is zero on every reading.
        """
        if axis not in AXIS_NAMES:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        index = AXIS_NAMES.index(axis)
        n_cells = (self.grid.nx, self.grid.ny, self.grid.nz)[index]
        low, high = self.thickness_by_face[index]
        inside = numpy.zeros(n_cells, dtype=bool)
        if low > 0:
            span = min(n_cells, int(low + 1.5))
            if half_integer:
                inside[0:span] = True
            else:
                inside[1: min(n_cells, span + 1)] = True
        if high > 0:
            span = min(n_cells, int(high + 1.5))
            if half_integer:
                inside[n_cells - span:] = True
            else:
                inside[max(0, n_cells - span + 1):] = True
                if self.axis_wraps(index):
                    inside[0] = True  # The stored image of the high wall plane.
        return inside

    def get_sigma_profile(self, axis: str) -> numpy.ndarray:
        """Raw sigma profile (prefac * profile(u), without the 0.5*dt factor) for one axis.

        Sampled at integer positions from that axis's own (low, high) faces, so it
        describes exactly the grading baked into the stored coefficients — including
        a face that carries nothing, which reads back as identically zero, and
        including cell 0, which on a wrapping axis is graded from the UPPER wall
        because that is the lattice point MEEP owns there (:meth:`_graded_sigma`).
        Host NumPy array, intended for diagnostics and plotting.
        """
        if axis not in AXIS_NAMES:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        index = AXIS_NAMES.index(axis)
        n_cells = (self.grid.nx, self.grid.ny, self.grid.nz)[index]
        return self._graded_sigma(n_cells, self.thickness_by_face[index], 0.0, 1.0,
                                  self.axis_wraps(index), self._wall_high(index))

    def __repr__(self) -> str:
        uniform = all(faces == (self.thickness, self.thickness) for faces in self.thickness_by_face)
        if uniform:
            layout = f"thickness={self.thickness}"
            grading = f"prefac={self.prefac():.2f}"
        else:
            layout = "faces=" + ", ".join(
                f"{AXIS_NAMES[axis]}({low},{high})"
                for axis, (low, high) in enumerate(self.thickness_by_face)
            )
            # Each face has its own prefac, so quoting one number would name a grading
            # that most of the layer does not use.
            depths = sorted({face for faces in self.thickness_by_face for face in faces if face})
            grading = "prefac=" + "/".join(f"{self.prefac(cells):.2f}" for cells in depths)
        return f"PML({layout}, order={self.order}, {grading}, R={self.R_asymptotic:.0e})"
