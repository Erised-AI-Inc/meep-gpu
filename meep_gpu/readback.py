"""
MEEP's field readback, word for word: ``get_array``, the stored-layout cell centres, and
``get_field_point``.

Every field value this engine hands a user as a READING is formed here, as MEEP 1.33.0
forms it, from the package's stored OWNED words only:

* ``GpuRunResult.get_array(c)`` and ``get_array(c, center=..., size=...)`` -- MEEP's
  ``sim.get_array`` (:func:`get_array`);
* ``FdtdDriver.get_field(c, cell_centered=True)`` -- the same centred words at every
  stored cell centre, in the stored layout (:func:`cell_center`);
* ``FdtdDriver.get_field_point(c, pt)`` -- MEEP's ``fields::get_field(c, vec)``
  (:class:`PointStencil`).

**The centred sum** (array_slice.cpp:413-415)::

    f[k] = 0.25 * (fc->f[cS[i]][k][idx] + fc->f[cS[i]][k][idx + off[2 * i]] +
                   fc->f[cS[i]][k][idx + off[2 * i + 1]] +
                   fc->f[cS[i]][k][idx + off[2 * i] + off[2 * i + 1]]);

three adds left to right in ``realnum`` (float32), the ``0.25 *`` in double; real and
imaginary parts summed separately. The offsets are ``yee2cent_offsets`` (vec.cpp:333-344):
``offset1`` is the stride of the FIRST direction in ``LOOP_OVER_DIRECTIONS`` order on
which the component is not Yee-shifted and ``offset2`` the second, or 0 when there is
one, so a one-axis average is ``((a + b) + a) + b``. The order is X, Y, Z, and Z then R
in Dcyl (vec.hpp:98-102).

**The words.** The four words are MEEP's CHUNK words: owned words are the package's
stored owned cells; a not-owned word is rebuilt from owned words by the connection
rules ``step_boundaries`` and ``zero_metal`` apply (step.cpp:171-221: CONNECT_COPY,
CONNECT_NEGATE ``= -word``, CONNECT_PHASE a ``complex<realnum>`` product with the phase
rounded to float, boundaries.cpp:362-366, :495; metal planes ``0.0``,
boundaries.cpp:310-313). The package's own ghost storage is never read: its ghosts carry
the package's fill arithmetic, which is not MEEP state.

**B reads the owner's H in a chunk where H aliases B** (:class:`HAliasRule`). MEEP
allocates H as an alias of B (fields.cpp:483-494) and gives H its own array only in a
chunk with ``chi1inv[Hc][d_c]``, ``have_f_minus_p`` or ``dsigw`` on the component's own
axis (update_eh.cpp:162-172). ``step_boundaries(H_stuff)`` runs after
``step_boundaries(B_stuff)`` (step.cpp:64-90), so in an aliasing chunk every not-owned
B word is the connected H word. After the run's FIRST step the other chunks hold no
connected B either: the connections the first B pass uses were built while every chunk
still aliased, and ``connect_the_chunks`` sends no B between two aliasing chunks
(boundaries.cpp:350-404); they are rebuilt only after the first ``update_eh`` allocates H
(update_eh.cpp:55, step.cpp:226). So at step 1 a not-owned B word of a chunk with H of its
own is still the zero it was allocated with.

**The store** (array_slice.cpp:416)::

    fields[i] = IVEC_LOOP_WEIGHT(s0i, s1i, e0i, e1i, 1.0) * complex<double>(f[0], f[1]) * ph[i];

weight, then the phase ``ph = shift_phase * S.phase_shift(c, sn)``, in double, one
rounding to ``complex<realnum>`` per part. ``shift_phase`` is
``prod_d pow(eikna[d], ishift[d])`` over every direction (loop_in_chunks.cpp:412-417):
the periodic duplicate plane of a Bloch axis carries ``pow(eikna, -1)`` there, formed in
the host C++ runtime's arithmetic (:func:`cpow_int`). An empty direction keeps its interpolation
weights and is collapsed by ``collapse_array``'s float ``+=`` (array_slice.cpp:582-587).

**The point reader** (monitor.cpp:127-157): ``grid_volume::interpolate``'s weights
(vec.cpp:558-612: the per-axis halving, the total-weight correction, the 1e-13 cut,
``stupidsort``, the equal-weight reset), then ``res += w * get_field(c, iloc)`` in
double, in order, with each leg located by ``locate_point_in_user_volume``
(boundaries.cpp:203-226) and phased ``S.phase_shift(c, sn) * kphase``. A leg no chunk
owns reads ``0.0``, as MEEP's does.

STORAGE CONVENTION ON A WRAPPING AXIS. MEEP owns ``(lc, lc + 2N]`` of a Yee-shift-0
component on an unfolded periodic axis, and so does the package's storage
(``Grid.stored_window_low``): on an axis with a Bloch phase stored cell 0 IS the owned
top word ``lc + 2N``, and the word at ``lc`` is MEEP's not-owned word, the CONNECT_PHASE
image of cell 0 by ``1.0 * conj(eikna)`` (combined over every axis on which the point is
below the window, X, Y, Z order, in ``complex<double>`` and rounded once), which
:func:`chunk_words` forms on an extended plane. At ``k = 0`` the two words are the same
(a COPY), and stored cell 0 is read as either.

NumPy only: arrays are copied to the host first, as ``get_array`` always did.

MEEP source references: array_slice.cpp:330-420, :582-587; loop_in_chunks.cpp:229-298,
:352-457; vec.cpp:333-344, :445-462, :512-612, :1155-1369; monitor.cpp:127-157;
boundaries.cpp:203-226, :310-313, :350-404, :362-366, :495; step.cpp:64-90, :171-221;
fields.cpp:483-494, :716-724; update_eh.cpp:70-101, :162-172; libc++ ``<complex>``
(``operator*``, ``log``, ``exp``, ``pow``).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ctypes
import ctypes.util
import itertools
import math
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy

from . import part_products
from .backends import to_numpy
from .fields import IYEE_SHIFTS, mirror_parity

_F32 = numpy.float32
_F64 = numpy.float64
_SMALL = 1e-13  # boundaries.cpp:359-360 phase_isclose; vec.cpp:560 interpolate's SMALL

#: A ``complex<double>`` as its two parts, Python floats, signed zeros kept.
ComplexParts = Tuple[float, float]

B_COMPONENTS = ("Bx", "By", "Bz")


def _slab(axis: int, index: Any) -> Tuple[Any, ...]:
    return (slice(None),) * axis + (index,)


def direction_axes(grid: Any) -> Tuple[int, ...]:
    """Package axes in MEEP's ``LOOP_OVER_DIRECTIONS`` order (vec.hpp:98-102, :141-144).

    D1/D2/D3 start at X (Z for D1); Dcyl starts at Z and stops before P, so it is Z then
    R, with the package's x carrying r and y the invariant phi.
    """
    if getattr(grid, "cylindrical", False):
        return (2, 0)
    return tuple(axis for axis in range(3) if not grid.is_invariant(axis))


# ---- C++ complex<double> arithmetic (libc++ / libstdc++), contraction off -----------------

class _SinCos(ctypes.Structure):
    _fields_ = [("sin", ctypes.c_double), ("cos", ctypes.c_double)]


def _load_libm() -> Any:
    """The C math library MEEP's own ``<complex>`` calls resolve to on this host."""
    if sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    else:
        name = ctypes.util.find_library("m")
        try:
            library = ctypes.CDLL(name or "libm.so.6")
        except OSError as exc:
            raise OSError("no C math library found for MEEP's complex<double> transcription "
                          "(ctypes.util.find_library('m') and libm.so.6)") from exc
    for function in ("hypot", "atan2"):
        getattr(library, function).restype = ctypes.c_double
        getattr(library, function).argtypes = [ctypes.c_double, ctypes.c_double]
    for function in ("log", "exp", "cos", "sin"):
        getattr(library, function).restype = ctypes.c_double
        getattr(library, function).argtypes = [ctypes.c_double]
    if sys.platform == "darwin":
        library.__sincos_stret.restype = _SinCos
        library.__sincos_stret.argtypes = [ctypes.c_double]
    elif hasattr(library, "sincos"):
        library.sincos.restype = None
        library.sincos.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    return library


_LIBM: Any = None


def _libm() -> Any:
    global _LIBM
    if _LIBM is None:
        _LIBM = _load_libm()
    return _LIBM


def sincos(x: float) -> Tuple[float, float]:
    """``(sin x, cos x)`` as MEEP's compiled ``<complex>`` gets them.

    libc++'s ``exp(complex)`` and ``polar`` take a ``sin`` and a ``cos`` of one argument,
    which the compiler fuses into one call: ``__sincos_stret`` on Darwin (MEASURED on
    MEEP 1.33.0's ``eikna``: a separate ``sin`` differs in its last bit on 1 of 408 Bloch
    axes, ``__sincos_stret`` on none), glibc's ``sincos`` elsewhere (INFERRED from
    ``__builtin_cexp``; not measured here).
    """
    libm = _libm()
    if sys.platform == "darwin":
        result = libm.__sincos_stret(float(x))
        return (float(result.sin), float(result.cos))
    if hasattr(libm, "sincos"):
        sin_out, cos_out = ctypes.c_double(), ctypes.c_double()
        libm.sincos(float(x), ctypes.byref(sin_out), ctypes.byref(cos_out))
        return (float(sin_out.value), float(cos_out.value))
    return (float(libm.sin(float(x))), float(libm.cos(float(x))))


def cmul(a: ComplexParts, b: ComplexParts) -> ComplexParts:
    """``complex<double> * complex<double>`` for finite operands: ``(ac - bd, ad + bc)``.

    One Python float operation at a time, so nothing contracts, as in the contraction-off
    MEEP build (libc++ ``operator*``; compiler-rt ``__muldc3``).
    """
    ac = a[0] * b[0]
    bd = a[1] * b[1]
    ad = a[0] * b[1]
    bc = a[1] * b[0]
    x = ac - bd
    y = ad + bc
    if math.isnan(x) and math.isnan(y):
        raise ArithmeticError("the complex multiply's NaN-recovery branch is not transcribed")
    return (x, y)


def _clog(z: ComplexParts) -> ComplexParts:
    """libc++ ``log(complex)`` = ``(log(abs(z)), arg(z))``; ``abs`` is ``hypot``, ``arg`` ``atan2``."""
    libm = _libm()
    return (libm.log(libm.hypot(z[0], z[1])), libm.atan2(z[1], z[0]))


def _cexp(z: ComplexParts) -> ComplexParts:
    """libc++ ``exp(complex)`` for a finite real part, its ``imag == 0`` branch included."""
    libm = _libm()
    if z[1] == 0:
        return (libm.exp(z[0]), math.copysign(0.0, z[1]))
    if math.isinf(z[0]):
        raise ArithmeticError("the infinite-real exp branch is not transcribed")
    e = libm.exp(z[0])
    sin_i, cos_i = sincos(z[1])
    return (e * cos_i, e * sin_i)


# libgcc2.c's __divdc3 constants (GCC >= 12): DBL_MIN, DBL_EPSILON, DBL_MAX / 2.
_DBL_MIN = 2.2250738585072014e-308
_RMIN2 = 2.220446049250313e-16
_RMINSCAL = 1.0 / _RMIN2
_RBIG = 1.7976931348623157e308 / 2
_RMAX2 = _RBIG * _RMIN2


def _divdc3(a: float, b: float, c: float, d: float) -> ComplexParts:
    """``(a + ib) / (c + id)`` as libgcc's ``__divdc3`` forms it (finite operands).

    The division GCC emits for ``complex<double>`` without ``-ffast-math``: Smith's
    algorithm with libgcc's power-of-two rescalings and its alternate order for a
    subnormal ratio. One Python float operation at a time, as the uncontracted build.
    """
    if abs(c) < abs(d):
        if abs(d) >= _RBIG:
            a, b, c, d = a / 2, b / 2, c / 2, d / 2
        if abs(d) < _RMIN2:
            a, b, c, d = a * _RMINSCAL, b * _RMINSCAL, c * _RMINSCAL, d * _RMINSCAL
        elif ((abs(a) < _DBL_MIN and abs(b) < _RMAX2 and abs(d) < _RMAX2)
              or (abs(b) < _DBL_MIN and abs(a) < _RMAX2 and abs(d) < _RMAX2)):
            a, b, c, d = a * _RMINSCAL, b * _RMINSCAL, c * _RMINSCAL, d * _RMINSCAL
        ratio = c / d
        denom = (c * ratio) + d
        if abs(ratio) > _DBL_MIN:
            return (((a * ratio) + b) / denom, ((b * ratio) - a) / denom)
        return (((c * (a / d)) + b) / denom, ((c * (b / d)) - a) / denom)
    if abs(c) >= _RBIG:
        a, b, c, d = a / 2, b / 2, c / 2, d / 2
    if abs(c) < _RMIN2:
        a, b, c, d = a * _RMINSCAL, b * _RMINSCAL, c * _RMINSCAL, d * _RMINSCAL
    elif ((abs(a) < _DBL_MIN and abs(b) < _RMAX2 and abs(c) < _RMAX2)
          or (abs(b) < _DBL_MIN and abs(a) < _RMAX2 and abs(c) < _RMAX2)):
        a, b, c, d = a * _RMINSCAL, b * _RMINSCAL, c * _RMINSCAL, d * _RMINSCAL
    ratio = d / c
    denom = (d * ratio) + c
    if abs(ratio) > _DBL_MIN:
        return (((b * ratio) + a) / denom, (b - (a * ratio)) / denom)
    return ((a + (d * (b / c))) / denom, (b - (d * (a / c))) / denom)


def _cpow_unsigned(x: ComplexParts, n: int) -> ComplexParts:
    """libstdc++'s ``__complex_pow_unsigned``: repeated squaring, ``complex<T>(1)`` for 0."""
    y = x if n % 2 else (1.0, 0.0)
    n >>= 1
    while n:
        x = cmul(x, x)
        if n % 2:
            y = cmul(y, x)
        n >>= 1
    return y


def cpow_int(x: ComplexParts, n: int) -> ComplexParts:
    """``pow(complex<double>, int)`` as the host's C++ runtime forms it.

    libc++ (MEEP built on Apple silicon) has no integer overload: ``pow(x,
    complex<double>(n))`` is ``exp(y * log(x))`` (measured there, P0-g). ``x = (1, +0)``,
    the ``eikna`` of every axis without a Bloch phase, gives ``(1, +0)`` for every ``n``
    (``log`` is ``(0, +0)``, the product ``(+-0, +0)``, and ``exp``'s ``imag == 0``
    branch returns ``(1, copysign(0, +0))``); that case is answered without the math
    library, so a run with no Bloch phase reads without one.

    libstdc++ (MEEP built with GCC) takes ``pow(const complex<T>&, int)``: repeated
    squaring, and ``complex<T>(1) /`` the power for a negative exponent, the division
    being libgcc's ``__divdc3`` (:func:`_divdc3`). The two forms differ in the last bit
    of most phased powers: against a probe compiled by the Linux reference's own
    toolchain (conda-forge GCC 15.2; parity/meep_gpu/results/bit_parity_0_10/dev/wp/RG/
    scripts/gpu_host/cpow_check.py), the libstdc++ form here matches 21,000 of 21,000
    random unit-modulus cases at n = -3..3 and the libc++ form 6,472. Which runtime is
    the host's: Darwin's is libc++, everything else is taken as libstdc++ (the host rule
    of :func:`sincos`).
    """
    if sys.platform != "darwin":
        if n < 0:
            power = _cpow_unsigned(x, -n)
            return _divdc3(1.0, 0.0, power[0], power[1])
        return _cpow_unsigned(x, n)
    if x[0] == 1.0 and x[1] == 0.0 and not math.copysign(1.0, x[1]) < 0:
        return (1.0, 0.0)
    return _cexp(cmul((float(n), 0.0), _clog(x)))


def eikna(grid: Any, axis: int) -> ComplexParts:
    """MEEP's ``fields::eikna[d]`` for one package axis: ``Grid.bloch_phase``, or ``1.0``."""
    phase = grid.bloch_phase(axis)
    if phase is None:
        return (1.0, 0.0)  # fields.cpp:46, and use_bloch's exp((+-0, 0)) = (1, +0) at k = 0
    phase = complex(phase)
    return (float(phase.real), float(phase.imag))


def shift_phase(grid: Any, ishift: Sequence[int]) -> ComplexParts:
    """``ph = 1.0; ph *= pow(eikna[d], ishift[d])`` over every direction (loop_in_chunks.cpp:412-417)."""
    ph = (1.0, 0.0)
    for axis in direction_axes(grid):
        ph = cmul(ph, cpow_int(eikna(grid, axis), int(ishift[axis])))
    return ph


# ---- MEEP's chunk words --------------------------------------------------------------------

def _parts(array: Any) -> List[numpy.ndarray]:
    """MEEP's per-component realnum arrays ``fc->f[c][0]`` and, if complex, ``fc->f[c][1]``."""
    host = numpy.asarray(to_numpy(array))
    if host.dtype.kind == "c":
        return [numpy.array(host.real, dtype=_F32, copy=True),
                numpy.array(host.imag, dtype=_F32, copy=True)]
    if host.dtype != _F32:
        raise TypeError(f"the MEEP-order reader expects float32 storage, got {host.dtype}")
    return [numpy.array(host, dtype=_F32, copy=True)]


def _connect(src: Sequence[numpy.ndarray], phase: ComplexParts) -> List[numpy.ndarray]:
    """One not-owned word from its owned source, step.cpp:171-221 by ``connect_phase``.

    ``connect_phase_from_phase`` (boundaries.cpp:362-366): within 1e-13 of +1 a copy, of
    -1 a unary minus per realnum, otherwise a ``complex<realnum>`` product with the phase
    rounded to float (boundaries.cpp:495), formed with separate float32 operations
    (:func:`~.part_products.connect_parts`, the one transcription the stepping shares).
    """
    return [numpy.asarray(w, dtype=_F32) for w in part_products.connect_parts(src, phase)]


def stored_window_axes(grid: Any, component: str) -> Tuple[int, ...]:
    """Axes on which ``component``'s stored cell 0 is MEEP's owned top word ``lc + 2N``.

    ``Grid.stored_window_low == 1``: a Yee-shift-0 component on an axis with a Bloch
    phase (the module docstring's storage convention).
    """
    shifts = IYEE_SHIFTS[component]
    return tuple(axis for axis in range(3) if grid.stored_window_low(axis, shifts[axis]) == 1)


def ghost_phase(grid: Any, axes: Sequence[int]) -> ComplexParts:
    """``thephase`` of the not-owned word below the window on ``axes``.

    ``locate_point_in_user_volume`` (boundaries.cpp:203-226) starts from ``1.0`` and
    multiplies by ``conj(eikna[d])`` for every direction on which the point is
    translated up into the user volume, in direction order, in ``complex<double>``.
    """
    ph = (1.0, 0.0)
    for axis in direction_axes(grid):
        if axis in axes:
            e = eikna(grid, axis)
            ph = cmul(ph, (e[0], -e[1]))
    return ph


def _extend_window(grid: Any, parts: List[numpy.ndarray], axes: Sequence[int]) -> List[numpy.ndarray]:
    """MEEP's chunk words over ``[lc, lc + 2N]`` on every stored-window axis: one plane more.

    On each axis in ``axes`` index ``i`` of the result is the word at doubled
    ``lc + 2i``: indices ``1 .. N - 1`` the stored cells, ``N`` the owned top word (stored
    cell 0), and ``0`` the not-owned word at ``lc`` -- the CONNECT image of the owned
    word at the same point translated up on exactly the axes where the index is 0, by
    :func:`ghost_phase` of those axes (one product per point, as MEEP forms a corner).
    """
    ext = list(parts)
    for axis in axes:
        ext = [numpy.concatenate((p, p[_slab(axis, slice(0, 1))]), axis=axis) for p in ext]
    out = [numpy.array(e, copy=True) for e in ext]
    for size in range(1, len(axes) + 1):
        for below in itertools.combinations(axes, size):
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            for axis in axes:
                if axis in below:
                    dst[axis] = 0
                    src[axis] = ext[0].shape[axis] - 1
                else:
                    dst[axis] = slice(1, None)
                    src[axis] = slice(1, None)
            words = _connect([e[tuple(src)] for e in ext], ghost_phase(grid, below))
            for o, w in zip(out, words):
                o[tuple(dst)] = w
    return out


def _parity(component: str, grid: Any, axis: int) -> int:
    return mirror_parity(component, axis, grid.mirror_phase(axis))


def _metal_terminated(grid: Any, axis: int) -> bool:
    """Is the sample one past the stored top a ``zero_metal`` word (boundaries.cpp:310-313)?"""
    if grid.is_axis(axis):
        return True  # Dcyl r_max: the PEC wall
    return bool(grid.is_metallic(axis))


def chunk_words(fields: Any, component: str, array: Any) -> List[numpy.ndarray]:
    """The words MEEP's chunk holds at every stored cell of one component.

    On a stored-window axis (:func:`stored_window_axes`) the result is one plane LONGER
    than the storage, ``[lc, lc + 2N]`` (:func:`_extend_window`); :func:`centred_sums`
    crops the sums back to the stored cells.

    Owned cells: the package's stored words. The not-owned cells a centred sum reaches:

    * a folded axis's low ghost (stored cell 0 of a component with Yee shift 0 there,
      doubled -2): ``connect_the_chunks`` connects it to its mirror image, stored cell
      ``MIRROR_SOURCE_INDEX`` = 2 at doubled +2, with the component's parity, COPY or
      NEGATE. Axes in X, Y, Z order; a doubly unowned corner then carries the product
      (negation is exact and involutive);
    * a metallic low wall on an unfolded axis (stored cell 0 of a shift-0 component):
      MEEP allocates the plane and never steps or connects it, and ``zero_metal`` holds
      it at +0. Applied after the mirror ghosts, so a metal point wins over a
      connection, as ``connect_the_chunks`` skips metal points.
    """
    grid = fields.grid
    shifts = IYEE_SHIFTS[component]
    parts = _parts(array)
    for axis in range(3):
        if not grid.is_mirrored(axis) or shifts[axis] != 0:
            continue
        source = [p[_slab(axis, 2)] for p in parts]
        image = _connect(source, (float(_parity(component, grid, axis)), 0.0))
        for p, value in zip(parts, image):
            p[_slab(axis, 0)] = value
    for axis in range(3):
        if shifts[axis] != 0 or grid.is_invariant(axis) or grid.is_mirrored(axis) or grid.is_axis(axis):
            continue
        if grid.is_metallic(axis):
            for p in parts:
                p[_slab(axis, 0)] = 0.0
    window = stored_window_axes(grid, component)
    return _extend_window(grid, parts, window) if window else parts


def _up(fields: Any, component: str, parts: Sequence[numpy.ndarray], axis: int) -> List[numpy.ndarray]:
    """``f[idx + stride(axis)]`` at every stored cell: the next word up, as MEEP's chunk holds it.

    The top stored plane's neighbour is MEEP's not-owned word one past it: +0 past a
    metal-terminated face (PEC wall, folded metallic far face, Dcyl r_max); on a folded
    PERIODIC far face the image about the second mirror at doubled ``n_full``, stored row
    ``n_full - n_q + 2`` (``stepping._far_reflect_rows``) with the mirror parity; on an
    unfolded periodic axis the owned top word ``lc + 2N``: on a stored-window axis the
    extended plane :func:`chunk_words` already carries (its last plane is then a
    placeholder no sum reads), and at ``k = 0`` a COPY of stored cell 0.
    """
    grid = fields.grid
    out = []
    for p in parts:
        q = numpy.empty_like(p)
        q[_slab(axis, slice(None, -1))] = p[_slab(axis, slice(1, None))]
        out.append(q)
    far = _slab(axis, -1)
    if axis in stored_window_axes(grid, component):
        for q, p in zip(out, parts):
            q[far] = p[far]
        return out
    if _metal_terminated(grid, axis):
        for q in out:
            q[far] = 0.0
        return out
    if grid.is_mirrored(axis):
        row = grid.shape_full[axis] - parts[0].shape[axis] + 2
        image = _connect([p[_slab(axis, row)] for p in parts], (float(_parity(component, grid, axis)), 0.0))
    else:
        image = _connect([p[_slab(axis, 0)] for p in parts], (1.0, 0.0))
    for q, value in zip(out, image):
        q[far] = value
    return out


# ---- B reads the owner's H where H aliases B -----------------------------------------------

class HAliasRule:
    """Which chunks hold H as an alias of B, per B component (update_eh.cpp:162-172).

    ``layout`` is the run's :class:`~.chunk_map.ChunkLayout`; ``aliases[c][k]`` is true when
    chunk ``k`` holds H as B's alias for B component ``c``: no ``sigsize[d_c] > 1`` on the
    component's own axis and no integrated B-family source point in the chunk
    (``have_f_minus_p``, update_eh.cpp:70-101, chunk-wide). The package refuses the third
    trigger, ``chi1inv[Hc][d_c]`` (mu != 1), at the lift. In an aliasing chunk every
    not-owned B word the centred sum takes is the connected H word.
    """

    def __init__(self, layout: Any, aliases: Mapping[str, numpy.ndarray], separate: numpy.ndarray,
                 first_step: bool = False):
        self.layout = layout
        self.aliases = {name: numpy.asarray(aliases[name], dtype=bool) for name in B_COMPONENTS}
        self.separate = numpy.asarray(separate, dtype=bool)
        self.first_step = bool(first_step)
        self._masks: Dict[Tuple[str, Tuple[int, ...]], Tuple[List[numpy.ndarray], List[numpy.ndarray]]] = {}

    def masks(self, grid: Any, component: str, axes: Sequence[int],
              shape: Tuple[int, ...]) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
        """Per word of the centred sum: does the reading chunk take the H word there, or a zero?

        The reading chunk is the owner of the centred point (doubled ``origin + 2i + 1``
        on every axis). Where that chunk does not own the word's point, an aliasing chunk
        takes the H word; after the first step a chunk with H of its own takes the zero it
        was allocated with. Words in ``[f, n1]`` or ``[f, n1, n2, n12]`` order.
        """
        key = (component, tuple(shape))
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        centre = []
        for axis in range(3):
            view = [1, 1, 1]
            view[axis] = shape[axis]
            line = grid.origin_doubled(axis) + 2 * numpy.arange(shape[axis], dtype=numpy.int64) + 1
            centre.append(line.reshape(view))
        cylindrical = bool(getattr(grid, "cylindrical", False))
        reader = self.layout.chunk_of(centre, cylindrical)
        flags = self.aliases[component]
        aliasing = numpy.where(reader >= 0, flags[numpy.where(reader >= 0, reader, 0)], False)
        offsets = {1: [(-1,), (+1,)], 2: [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]}[len(axes)]
        take_h, take_zero = [], []
        for offs in offsets:
            coords = list(centre)
            for axis, off in zip(axes, offs):
                coords[axis] = centre[axis] + off
            notowned = ~self.layout.chunk_owns(reader, coords, cylindrical) & (reader >= 0)
            take_h.append(numpy.broadcast_to(aliasing & notowned, shape))
            stale = notowned & ~aliasing if self.first_step else numpy.zeros_like(notowned)
            take_zero.append(numpy.broadcast_to(stale, shape))
        self._masks[key] = (take_h, take_zero)
        return take_h, take_zero


def h_alias_rule(layout: Any, grid: Any, sources: Sequence[Any] = (), first_step: bool = False) -> HAliasRule:
    """The run's :class:`HAliasRule`, from its chunk layout and its registered sources.

    ``first_step`` is true for the state after the run's first step (see the module
    docstring: the not-owned B words of a chunk with H of its own are still zero then).
    """
    n_chunks = len(layout.boxes)
    separate = numpy.zeros(n_chunks, dtype=bool)
    for source in sources:
        if getattr(source, "field_type", None) != "B" or not bool(getattr(source, "is_integrated", False)):
            continue
        shifts = IYEE_SHIFTS[source.component]
        points = getattr(source, "_indices", None) or ()
        for index in points:
            coords = [numpy.asarray(grid.origin_doubled(axis) + 2 * int(index[axis]) + shifts[axis])
                      for axis in range(3)]
            owner = int(layout.chunk_of(coords, bool(getattr(grid, "cylindrical", False))))
            if owner >= 0:  # loop_in_chunks clips a src_vol to its chunk's owned corners
                separate[owner] = True
    pml = numpy.asarray(layout.pml_flags, dtype=bool).reshape(n_chunks, 3)
    aliases = {name: ~(pml[:, axis] | separate) for axis, name in enumerate(B_COMPONENTS)}
    return HAliasRule(layout, aliases, separate, first_step=first_step)


def _needs_alias_rule(fields: Any, component: str) -> bool:
    """A B read depends on the chunk layout only where H has storage of its own."""
    return component in B_COMPONENTS and bool(getattr(fields, "_pml_active", False))


# ---- the centred sums ----------------------------------------------------------------------

def _four_words(fields: Any, component: str, parts: List[numpy.ndarray],
                axes: Sequence[int]) -> List[List[numpy.ndarray]]:
    grid = fields.grid
    if len(axes) == 0:
        return [parts]
    if len(axes) == 1:
        return [parts, _up(fields, component, parts, axes[0])]
    if len(axes) == 2:
        n1 = _up(fields, component, parts, axes[0])
        n2 = _up(fields, component, parts, axes[1])
        n12 = _up(fields, component, n1, axes[1])
        if _metal_terminated(grid, axes[0]):  # the corner past a metal face is a zero_metal word
            for q in n12:
                q[_slab(axes[0], -1)] = 0.0
        return [parts, n1, n2, n12]
    raise RuntimeError(f"weird yee shift for component {component}")  # vec.cpp:337


def _cropped(words: List[List[numpy.ndarray]], shape: Tuple[int, ...]) -> List[List[numpy.ndarray]]:
    """Each word's planes at the stored cells: drops the extended plane of a stored-window axis."""
    crop = tuple(slice(0, int(n)) for n in shape)
    return [[part[crop] for part in word] for word in words]


def centred_sums(fields: Any, component: str, alias: Optional[HAliasRule] = None) -> List[numpy.ndarray]:
    """``0.25 * (sum)`` as float64 per realnum part, at every stored cell centre (array_slice.cpp:413-415).

    Index ``i`` on each axis is the centred point at doubled ``origin + 2*i + 1``.
    """
    grid = fields.grid
    stored = fields.get_component(component)
    parts = chunk_words(fields, component, stored)
    shifts = IYEE_SHIFTS[component]
    axes = [axis for axis in direction_axes(grid) if shifts[axis] == 0]
    words = _cropped(_four_words(fields, component, parts, axes), stored.shape)
    if axes and _needs_alias_rule(fields, component):
        if alias is None:
            raise ValueError(
                f"reading {component} needs the run's chunk layout: H has its own storage, and in "
                f"a chunk where MEEP holds H as B's alias every not-owned {component} word is the "
                f"owner's H word (update_eh.cpp:162-172). Read it through the driver "
                f"(FdtdDriver.get_field, GpuRunResult.get_array), or pass readback.h_alias_rule(...).")
        h_parts = chunk_words(fields, component, fields.get_component("H" + component[1]))
        h_words = _cropped(_four_words(fields, component, h_parts, axes), stored.shape)
        take_h, take_zero = alias.masks(grid, component, axes, tuple(stored.shape))
        words = [[numpy.where(zero, _F32(0.0), numpy.where(take, ph, pb)).astype(_F32, copy=False)
                  for pb, ph in zip(word_b, word_h)]
                 for word_b, word_h, take, zero in zip(words, h_words, take_h, take_zero)]
    if len(words) == 1:
        sums = [((f + f) + f) + f for f in words[0]]
    elif len(words) == 2:
        sums = [((f + a) + f) + a for f, a in zip(words[0], words[1])]
    else:
        sums = [((f + b) + c) + d for f, b, c, d in zip(*words)]
    return [s.astype(_F64) * 0.25 for s in sums]


def _times_phase(q: Sequence[numpy.ndarray], weight: Any, pr: Any, pi: Any) -> List[numpy.ndarray]:
    """``weight * complex<double>(f[0], f[1]) * ph`` in double (array_slice.cpp:416), separate ops.

    ``f[1]`` is 0 when the component has no imaginary array (array_slice.cpp:417-418).
    Returns float64 (re, im); the caller rounds to realnum once.
    """
    qr = q[0]
    qi = q[1] if len(q) > 1 else numpy.zeros_like(qr)
    wr = weight * qr
    wi = weight * qi
    re = (wr * pr) - (wi * pi)
    im = (wr * pi) + (wi * pr)
    return [re, im]


def _store(re: numpy.ndarray, im: numpy.ndarray, complex_storage: bool) -> numpy.ndarray:
    if not complex_storage:
        return re.astype(_F32)
    out = numpy.empty(re.shape, dtype=numpy.complex64)
    out.real = re.astype(_F32)
    out.imag = im.astype(_F32)
    return out


def cell_center(fields: Any, component: str, alias: Optional[HAliasRule] = None) -> numpy.ndarray:
    """The stored-layout cell-centred array (``FdtdDriver.get_field(..., cell_centered=True)``).

    Each value is the word MEEP's ``get_array`` forms at that point in its owned frame:
    weight 1 and ``ph = shift_phase * phase_shift(c, 0)`` with every ``ishift`` 0, which
    is ``(1, +0)`` without a Bloch phase and ``(1, +-0)`` with one (``pow(eikna, 0)``
    keeps the sign of ``0 * arg(eikna)``). A host array, float32 or complex64 as the
    storage is.
    """
    q = centred_sums(fields, component, alias)
    ph = cmul(shift_phase(fields.grid, (0, 0, 0)), phase_shift(fields.grid, component, 0))
    re, im = _times_phase(q, 1.0, ph[0], ph[1])
    return _store(re, im, len(q) > 1)


# ---- MEEP's symmetry phases, signed zeros included -----------------------------------------

#: One symmetry of MEEP's chain: (the package axes it flips, the reflected-mask bit it
#: answers to, its ``complex<double>`` phase ``ph``).
ChainEntry = Tuple[Tuple[int, ...], int, ComplexParts]


def _chain(grid: Any) -> List[ChainEntry]:
    """The symmetry chain ``fields::S`` as MEEP builds it, in list order.

    Cartesian: simulation.py ``_create_symmetries`` adds ``mirror(d, gv) * phase`` per
    declared plane; ``operator*`` does ``s.ph *= p`` from ``ph = 1.0`` (vec.cpp:1155-1165,
    :1239-1243). Dcyl: ``S = S + r_to_minus_r_symmetry(m)`` (fields.cpp:42), which flips R
    and P (package x and y) with ``ph = -1.0`` for odd ``m``, ``1.0`` for even
    (vec.cpp:1141-1153); the package refuses mirrors on a cylindrical grid.
    """
    out: List[ChainEntry] = []
    for plane in grid.symmetry:
        declared = complex(plane.phase)
        axis = int(plane.axis_index)
        out.append(((axis,), axis, cmul((1.0, 0.0), (float(declared.real), float(declared.imag)))))
    if getattr(grid, "cylindrical", False):
        m = int(getattr(grid, "m", 0))
        out.append(((0, 1), 0, (-1.0 if m & 1 else 1.0, 0.0)))
    return out


def _transform(chain: Sequence[ChainEntry], d: int, n: int, k: int = 0) -> Tuple[bool, ComplexParts]:
    """``symmetry::transform(d, n)`` (vec.cpp:1251-1284) for a chain of g = 2 symmetries
    that flip directions and permute none."""
    if n == 0 or k >= len(chain):
        return (False, (1.0, 0.0))
    nme, nrest = n % 2, n // 2
    if nme == 0:
        if nrest == 0:
            return (False, (1.0, 0.0))
        return _transform(chain, d, nrest, k + 1)
    axes, _, ph = chain[k]
    flipped = d in axes
    if k + 1 < len(chain) and nrest:
        inner_flipped, inner_phase = _transform(chain, d, nrest, k + 1)
        return (inner_flipped != flipped, cmul(inner_phase, ph))
    return (flipped, cmul((1.0, 0.0), ph))


def phase_shift(grid: Any, component: str, reflected: int) -> ComplexParts:
    """``symmetry::phase_shift(c, n)`` (vec.cpp:1347-1369) for the image reflecting the
    package axes in bit mask ``reflected``; ``-phase`` is a unary minus on both parts.

    A pseudovector (H, B) also flips once per flipped direction (``FOR_DIRECTIONS``); no
    symmetry here permutes directions, so the handedness term never fires.
    """
    chain = _chain(grid)
    n = sum(1 << k for k, (_, bit, _) in enumerate(chain) if reflected >> bit & 1)
    direction = "xyz".index(component[1])
    flip, phase = _transform(chain, direction, n)
    if component[0] in "HB":
        for d in range(3):
            if _transform(chain, d, n)[0]:
                flip = not flip
    return (-phase[0], -phase[1]) if flip else phase


# ---- the whole-cell centred lattice --------------------------------------------------------

def _unfold(a: numpy.ndarray, axis: int, n_full: int, centre: int) -> numpy.ndarray:
    shape = list(a.shape)
    shape[axis] = n_full
    full = numpy.zeros(tuple(shape), dtype=a.dtype)
    full[_slab(axis, slice(centre, None))] = a[_slab(axis, slice(1, 1 + n_full - centre))]
    full[_slab(axis, centre - 1)] = a[_slab(axis, 0)]
    full[_slab(axis, slice(None, centre - 1))] = a[_slab(axis, slice(centre, 1, -1))]
    return full


class CentredLattice:
    """Every centred point MEEP's ``get_array`` can visit, with its double sum and its phase.

    ``q`` holds the parent-frame ``0.25 * sum`` (float64 per part); ``reflected`` the bit
    mask of folded axes on which the point is a mirror image; ``shifted`` the bit mask of
    axes on which it is the periodic duplicate one lattice vector down (``ishift = -1``);
    ``phase_r`` / ``phase_i`` MEEP's ``ph = shift_phase * S.phase_shift(cS, sn)`` as
    ``complex<double>`` parts with their signed zeros; ``ivec[axis]`` the doubled
    coordinate of each index on the reported axes.

    Built by the stored layout's unfolding rule: on a folded axis the stored centre at
    doubled -1 is MEEP's owned extra pixel and lands unphased, and doubled +3 and up land
    mirrored with the parity. Then the periodic duplicate plane, below the first: on an
    unfolded axis that is not metallic, and on a folded PERIODIC axis of odd count (whose
    odd window sits half a cell above the requested cell). A metallic axis, the Dcyl r
    axis and an invariant axis have none.
    """

    def __init__(self, fields: Any, component: str, alias: Optional[HAliasRule] = None):
        grid = fields.grid
        self.grid = grid
        self.component = component
        self.complex = numpy.asarray(to_numpy(fields.get_component(component))).dtype.kind == "c"
        q = centred_sums(fields, component, alias)
        reflected = numpy.zeros(q[0].shape, dtype=numpy.uint8)
        shifted = numpy.zeros(q[0].shape, dtype=numpy.uint8)
        ivec: Dict[int, numpy.ndarray] = {}
        for axis in range(3):
            if grid.is_invariant(axis):
                continue
            ivec[axis] = grid.origin_doubled(axis) + 2 * numpy.arange(q[0].shape[axis]) + 1
        for axis in range(3):
            if grid.mirror_phase(axis) is None:
                continue
            n_full = grid.shape_full[axis]
            centre = n_full // 2
            q = [_unfold(a, axis, n_full, centre) for a in q]
            reflected = _unfold(reflected, axis, n_full, centre)
            shifted = _unfold(shifted, axis, n_full, centre)
            reflected[_slab(axis, slice(None, centre - 1))] |= numpy.uint8(1 << axis)
            ivec[axis] = 2 * numpy.arange(n_full) - 2 * centre + 1
        for axis in range(3):
            if grid.is_invariant(axis) or grid.is_axis(axis):
                continue
            if grid.is_mirrored(axis):
                prepend = grid.shape_full[axis] % 2 == 1 and not grid.is_metallic(axis)
            else:
                prepend = not grid.is_metallic(axis)
            if not prepend:
                continue
            q = [numpy.concatenate((a[_slab(axis, slice(-1, None))], a), axis=axis) for a in q]
            reflected = numpy.concatenate((reflected[_slab(axis, slice(-1, None))], reflected), axis=axis)
            low = shifted[_slab(axis, slice(-1, None))] | numpy.uint8(1 << axis)
            shifted = numpy.concatenate((low, shifted), axis=axis)
            ivec[axis] = numpy.concatenate(([ivec[axis][0] - 2], ivec[axis]))
        self.q = q
        self.reflected = reflected
        self.shifted = shifted
        self.phase_r = numpy.empty(reflected.shape, dtype=_F64)
        self.phase_i = numpy.empty(reflected.shape, dtype=_F64)
        key = reflected.astype(numpy.uint16) | (shifted.astype(numpy.uint16) << 8)
        for value in numpy.unique(key):
            mask, shift_bits = int(value) & 0xFF, int(value) >> 8
            ishift = [-1 if shift_bits >> axis & 1 else 0 for axis in range(3)]
            ph = cmul(shift_phase(grid, ishift), phase_shift(grid, component, mask))
            self.phase_r[key == value] = ph[0]
            self.phase_i[key == value] = ph[1]
        self.ivec = ivec


def _vec2diel_floor(coordinate: float, a: float) -> int:  # loop_in_chunks.cpp:229-236
    return 1 + 2 * int(math.floor(coordinate * a - .5))


def _vec2diel_ceil(coordinate: float, a: float) -> int:  # loop_in_chunks.cpp:238-245
    return 1 + 2 * int(math.ceil(coordinate * a - .5))


def total_volume(grid: Any) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """MEEP's ``fields::total_volume()`` as ``(center, size)`` (fields.cpp:716-724).

    ``gv.interior()`` (vec.cpp:289-291) is ``[little_corner, big_corner - 2]`` doubled, one
    cell short of the top, on the STEPPED (halved) grid; a mirrored axis is unioned with
    its reflection about the plane; Dcyl clips r at the axis (fields.cpp:722).
    """
    centers, sizes = [], []
    for axis in range(3):
        if grid.is_invariant(axis):
            centers.append(0.0)
            sizes.append(0.0)
            continue
        low = grid.origin_doubled(axis)
        high = low + 2 * grid.stored_cells(axis) - 2
        if grid.is_mirrored(axis):
            reach = max(abs(low), abs(high))
            low, high = -reach, reach
        if grid.is_axis(axis):
            low = max(low, 0)
        low = low * 0.5 * grid.dx
        high = high * 0.5 * grid.dx
        centers.append(0.5 * (low + high))
        sizes.append(high - low)
    return (centers[0], centers[1], centers[2]), (sizes[0], sizes[1], sizes[2])


def get_array(fields: Any, component: str, alias: Optional[HAliasRule] = None,
              center: Optional[Sequence[float]] = None,
              size: Optional[Sequence[float]] = None) -> numpy.ndarray:
    """``sim.get_array(component=c)``, or with ``center=``/``size=``, in MEEP's words.

    The volume is MEEP's: ``fields::total_volume()`` by default (:func:`total_volume`),
    else the corners ``center -/+ size * 0.5`` as ``mp.Volume`` forms them. Per reported
    axis the centred points visited are the lattice points in ``[vec2diel_floor(min),
    vec2diel_ceil(max)]`` (loop_in_chunks.cpp:352-356). A non-empty direction takes
    weight 1 (array_slice.cpp:363-368). An empty one keeps ``compute_boundary_weights``'
    interpolation weights (loop_in_chunks.cpp:257-298, the ``min == max`` branch,
    reconciled per chunk at :441-457: the low point ``w0``, the high point ``w1``, a lone
    point the ``min`` of its pair) and, when it spans two points, is summed by
    ``collapse_array``'s float ``+=`` from zero (array_slice.cpp:582-587); a lone point is
    not in the uncollapsed rank and is not summed. Axes of one point leave the rank
    (``get_array_slice_dimensions``' ``n > 1`` filter), and the rest come in direction
    order: (z, r) in Dcyl.

    Raises:
        ValueError: only one of ``center``/``size`` given, or a volume that misses the
            component's centred lattice on some axis.
        NotImplementedError: a ``center``/``size`` volume on a cylindrical grid (only the
            whole cell is transcribed there).
    """
    grid = fields.grid
    if center is None and size is None:
        center, size = total_volume(grid)
    elif center is None or size is None:
        raise ValueError("pass both center and size, or neither (the whole cell)")
    elif getattr(grid, "cylindrical", False):
        raise NotImplementedError(
            "a center/size get_array on a cylindrical grid is not transcribed; read the whole "
            "cell (no center, no size) and slice it")
    center = tuple(float(v) for v in center)
    size = tuple(float(v) for v in size)
    if len(center) != 3 or len(size) != 3:
        raise ValueError(f"center and size take three components, got {center!r} and {size!r}")
    lattice = CentredLattice(fields, component, alias)
    a = float(grid.resolution)
    order = direction_axes(grid)
    pick: List[numpy.ndarray] = [numpy.array([0])] * 3
    wline: Dict[int, numpy.ndarray] = {}
    collapse: List[int] = []
    for axis in order:
        lo = center[axis] - size[axis] * 0.5
        hi = center[axis] + size[axis] * 0.5
        lo, hi = min(lo, hi), max(lo, hi)
        i_s = _vec2diel_floor(lo, a)
        i_e = _vec2diel_ceil(hi, a)
        points = lattice.ivec[axis]
        keep = numpy.nonzero((points >= i_s) & (points <= i_e))[0]
        if keep.size == 0:
            raise ValueError(f"the volume misses the {'xyz'[axis]} centred lattice of {component}")
        pick[axis] = keep
        if lo != hi:
            continue
        w0 = 1. - lo * a + 0.5 * i_s  # loop_in_chunks.cpp:260-261, evaluated left to right
        w1 = 1. + hi * a - 0.5 * i_e
        if keep.size == 2:
            wline[axis] = numpy.array([min(w0, w0), min(w1, w1)], dtype=_F64)
            collapse.append(axis)
        elif keep.size == 1:
            visited = int(points[keep[0]])
            if i_e == i_s:
                w = min(w0, w1)
            elif visited == i_s:
                w = min(w0, w0)
            else:
                w = min(w1, w1)
            wline[axis] = numpy.array([w], dtype=_F64)
        else:
            raise RuntimeError("an empty dimension spanning more than two centred planes")
    grid_ix = numpy.ix_(*pick)
    q = [part[grid_ix] for part in lattice.q]
    phase_r = lattice.phase_r[grid_ix]
    phase_i = lattice.phase_i[grid_ix]
    # IVEC_LOOP_WEIGHT (vec.hpp:381-383): w3 * (w2 * (dV * w1)), dV = 1.0, over the loop
    # directions in order; 1.0 on every non-empty direction.
    weight: Any = None
    for axis in order:
        factor: Any = _F64(1.0)
        if axis in wline:
            shape = [1, 1, 1]
            shape[axis] = wline[axis].size
            factor = wline[axis].reshape(shape)
        weight = (_F64(1.0) * factor) if weight is None else (factor * weight)
    if weight is None:
        weight = _F64(1.0)
    re, im = _times_phase(q, weight, phase_r, phase_i)
    store = [re.astype(_F32), im.astype(_F32)]
    if collapse:  # collapse_array: reduced[...] += full[...] in realnum, row-major
        shape = list(store[0].shape)
        reduced = tuple(1 if axis in collapse else shape[axis] for axis in range(3))
        acc = [numpy.zeros(reduced, dtype=_F32), numpy.zeros(reduced, dtype=_F32)]
        ranges = [range(shape[axis]) if axis in collapse else (None,) for axis in range(3)]
        for combo in itertools.product(*ranges):
            sel = tuple(slice(combo[axis], combo[axis] + 1) if axis in collapse else slice(None)
                        for axis in range(3))
            acc = [acc[k] + store[k][sel] for k in range(2)]
        store = acc
    kept = [axis for axis in order if store[0].shape[axis] > 1]
    drop = tuple(axis for axis in range(3) if axis not in kept)
    if drop:
        store = [s.squeeze(axis=drop) for s in store]
    if len(kept) > 1:
        positions = sorted(kept)
        permutation = [positions.index(axis) for axis in kept]
        if permutation != sorted(permutation):
            store = [numpy.transpose(s, permutation) for s in store]
    if not lattice.complex:
        return numpy.array(store[0], dtype=_F32, copy=True)  # a point volume stays 0-d, as MEEP's
    out = numpy.empty(store[0].shape, dtype=numpy.complex64)
    out.real = store[0]
    out.imag = store[1]
    return out


# ---- the point reader: fields::get_field(component, vec), monitor.cpp:127-157 -------------

def _my_round(x: float) -> int:  # meep_internals.hpp:29
    return int(math.floor(math.fabs(x) + 0.5) * (-1 if x < 0 else 1))


def interpolate(grid: Any, component: str, loc: Sequence[float]) -> Tuple[List[Dict[int, int]], List[float]]:
    """``grid_volume::interpolate(c, loc, locs, weights)``, vec.cpp:558-612, in its own order.

    Doubled integer coordinates per package axis (x carries r in Dcyl); the direction loop
    is MEEP's (:func:`direction_axes`). Weights below 1e-13 are zeroed, ``stupidsort``
    moves weights below 2e-15 to the end, and equal weights are reset to ``1/n``.
    """
    a = float(grid.resolution)
    inva = 1.0 / a  # vec.cpp:306
    half = 0.5 * inva  # operator[](ivec): p * (0.5 * inva), vec.hpp:1055
    iyee = IYEE_SHIFTS[component]
    dirs = direction_axes(grid)
    p = {d: (float(loc[d]) - iyee[d] * half) * a for d in dirs}
    middle = {d: int(math.floor(p[d])) * 2 + 1 + iyee[d] for d in dirs}
    midv = {d: middle[d] * half for d in dirs}
    dv = {d: (float(loc[d]) - midv[d]) * (2 * a) for d in dirs}
    base = {d: _my_round(midv[d] * 2 * a) for d in dirs}  # round_vec, vec.cpp:32-36
    locs = [dict(base) for _ in range(8)]
    weights = [1.0] * 8
    have = 1
    for d in dirs:
        for i in range(have):
            locs[have + i] = dict(locs[i])
            weights[have + i] = weights[i]
            locs[i][d] = middle[d] - 1
            weights[i] *= 0.5 * (1.0 - dv[d])
            locs[have + i][d] = middle[d] + 1
            weights[have + i] *= 0.5 * (1.0 + dv[d])
        have *= 2
    for i in range(have, 8):
        weights[i] = 0.0
    total = 0.0
    for i in range(have):
        total += weights[i]
    for i in range(have):
        weights[i] += (1.0 - total) * (1.0 / have)
    for i in range(have):
        if weights[i] < 0.0:
            if -weights[i] >= _SMALL * 1e5:
                raise RuntimeError(f"large negative interpolation weight[{i}] = {weights[i]:e}")
            weights[i] = 0.0
        elif weights[i] < _SMALL:
            weights[i] = 0.0
    start, length = 0, have  # stupidsort (vec.cpp:512-526) over the first `have` entries
    while length:
        if math.fabs(weights[start]) < 2e-15:
            weights[start] = weights[start + length - 1]
            locs[start] = locs[start + length - 1]
            weights[start + length - 1] = 0.0
            locs[start + length - 1] = {d: 0 for d in dirs}
        else:
            start += 1
        length -= 1
    same = True
    i = 0
    while i < 8 and weights[i]:
        if weights[i] != weights[0]:
            same = False
        i += 1
    if same:
        count = 0
        while count < 8 and weights[count]:
            count += 1
        for i in range(count):
            weights[i] = 1.0 / count
    return locs, weights


def _locate(grid: Any, component: str, iloc: Mapping[int, int]) -> Tuple[Optional[Tuple[int, int, int]], ComplexParts]:
    """``fields::get_field(c, ivec)``'s geometry (monitor.cpp:145-157).

    Returns the stored index of the owning word (``None`` when no chunk owns the point,
    where MEEP returns ``0.0``) and ``S.phase_shift(c, sn) * kphase``. MEEP's owned top
    point ``lc + 2N`` of a Yee-shift-0 component on an unfolded periodic axis is stored
    cell 0 (the module docstring's storage convention; at ``k = 0`` the same word).
    """
    iyee = IYEE_SHIFTS[component]
    kphase = (1.0, 0.0)
    pos = dict(iloc)
    for d in pos:  # locate_point_in_user_volume, boundaries.cpp:203-226
        n_full = grid.shape_full[d]
        io = -(n_full - n_full % 2) if not grid.is_axis(d) else 0  # user_volume io (centred)
        lattice = 2 * n_full
        if grid.is_metallic(d) or grid.is_axis(d):
            continue
        e = eikna(grid, d)
        if pos[d] <= io:
            while pos[d] <= io:
                pos[d] += lattice
                kphase = cmul(kphase, (e[0], -e[1]))
        elif pos[d] - lattice > io:
            while pos[d] - lattice > io:
                pos[d] -= lattice
                kphase = cmul(kphase, e)
    reflected = 0
    for d in pos:
        if grid.is_mirrored(d) and pos[d] <= -2:  # not owned by the halved chunk (io = -2)
            pos[d] = -pos[d]
            reflected |= 1 << d
        elif grid.is_axis(d) and pos[d] < 0:  # Dcyl: no chunk owns r < 0; r_to_minus_r does
            pos[d] = -pos[d]
            reflected |= 1 << d
    index = []
    for axis in range(3):
        if axis not in pos:
            index.append(0)
            continue
        j2 = pos[axis] - grid.origin_doubled(axis) - iyee[axis]
        if j2 % 2:
            raise RuntimeError("interpolation point off the component's Yee lattice")
        j = j2 // 2
        n_stored = grid.stored_cells(axis)
        o = pos[axis] - (-2 if grid.is_mirrored(axis) else grid.origin_doubled(axis))
        limit = 2 * (grid.owned_cells(axis) if grid.is_mirrored(axis) else grid.shape_full[axis])
        if not (0 < o <= limit) and not (grid.is_axis(axis) and o == 0):
            return None, (0.0, 0.0)  # owned by no chunk: monitor.cpp:157 returns 0.0
        unfolded_periodic = not (grid.is_metallic(axis) or grid.is_axis(axis) or grid.is_mirrored(axis))
        if unfolded_periodic and iyee[axis] == 0 and j == n_stored:
            j = 0  # MEEP's owned top point lc + 2N: stored cell 0 (the storage convention)
        if not 0 <= j < n_stored:
            return None, (0.0, 0.0)
        index.append(j)
    ps = phase_shift(grid, component, reflected)  # S.phase_shift(c, sn)
    return (index[0], index[1], index[2]), cmul(ps, kphase)


class PointStencil:
    """One ``(component, point)``'s ``get_field`` geometry: legs, weights, phases.

    Geometry only, built once per (component, point) and cacheable: the words are read
    at call time (:meth:`read`), so a reset or a change of storage cannot leave a stale
    value behind.
    """

    def __init__(self, grid: Any, component: str, point: Sequence[float]):
        self.grid = grid
        self.component = component
        self.point = tuple(float(v) for v in point)
        locs, weights = interpolate(grid, component, self.point)
        self.legs: List[Tuple[float, Optional[Tuple[int, int, int]], ComplexParts]] = []
        for argh in range(8):
            if not weights[argh]:
                break
            index, phase = _locate(grid, component, locs[argh])
            self.legs.append((weights[argh], index, phase))
        indexed = [leg[1] for leg in self.legs if leg[1] is not None]
        self._gather = (tuple(numpy.asarray([index[axis] for index in indexed], dtype=numpy.int64)
                              for axis in range(3)) if indexed else None)
        self._device_gather: Dict[str, Any] = {}
        self.special_kz: Optional[ComplexParts] = None
        beta = float(getattr(grid, "beta", 0.0) or 0.0)
        if beta != 0.0 and self.point[2] != 0.0:  # beta is legal on a 2-D grid only (Grid)
            turn = 2 * math.pi * beta * self.point[2]  # monitor.cpp:139-140, left to right
            sin_t, cos_t = sincos(turn)
            self.special_kz = (1.0 * cos_t, 1.0 * sin_t)  # std::polar(1.0, turn)

    def _words(self, fields: Any) -> Tuple[List[ComplexParts], bool]:
        """The owning words, in leg order, as ``complex<double>`` parts; and whether storage is complex."""
        if self._gather is None:
            return [], False  # every leg lies in no chunk: nothing to read
        xp = fields.grid.xp
        key = getattr(xp, "__name__", str(id(xp)))
        index = self._device_gather.get(key)
        if index is None:
            index = self._gather if xp is numpy else tuple(xp.asarray(a) for a in self._gather)
            self._device_gather[key] = index
        values = numpy.asarray(to_numpy(fields.sliced_component(self.component, index)))
        if values.dtype.kind == "c":
            re = numpy.asarray(values.real, dtype=_F32)
            im = numpy.asarray(values.imag, dtype=_F32)
            return [(float(re[i]), float(im[i])) for i in range(values.shape[0])], True
        re = numpy.asarray(values, dtype=_F32)
        return [(float(re[i]), 0.0) for i in range(values.shape[0])], False

    def read(self, fields: Any) -> complex:
        """``res += w[argh] * get_field(c, ilocs[argh])`` in order (monitor.cpp:135-140)."""
        words, _ = self._words(fields)
        stream = iter(words)
        rr, ri = 0.0, 0.0
        for weight, index, phase in self.legs:
            if index is None:
                value = (0.0, 0.0)
            else:
                value = cmul(phase, next(stream))
            rr = rr + weight * value[0]
            ri = ri + weight * value[1]
        if self.special_kz is not None:
            rr, ri = cmul((rr, ri), self.special_kz)
        return complex(rr, ri)
