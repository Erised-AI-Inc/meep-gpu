"""MEEP's ``amp_data`` interpolation, transcribed — the array spelling of an amp_func.

``mp.Source(amp_data=<array>)`` is not a second kind of source. MEEP turns it into
an ordinary ``amp_func`` and forgets the array ever existed:
``Source.add_source`` (python/source.py:145-159) hands the array to
``fields::add_volume_source(c, src, where, arr, dim1, dim2, dim3, amp)``
(src/sources.cpp:394-417), which copies it into two file-static buffers and calls
``add_volume_source(c, src, where_, amp_file_func, amp)`` — the SAME entry point a
user ``amp_func`` reaches. ``amp_func_file`` is the same story one step earlier:
src/sources.cpp:421-460 reads the two HDF5 datasets and calls the array overload.

So the array route needs no new machinery in the stepper, the deposition or the
source table. It needs exactly one function — MEEP's ``amp_file_func``
(src/sources.cpp:345-374) and the two helpers under it,
``meep::linear_interpolate`` (src/fields.cpp:792-812) and
``meep::map_coordinates`` (src/fields.cpp:759-789) — evaluated in Python at lift
time and handed to :mod:`~.sources` as an ordinary amp_func.

The transcription is deliberately literal, quirks included, because the point is to
be MEEP rather than to be reasonable:

* ``map_coordinates`` folds ``r > 1`` to ``1 - r``, not to ``2 - r``. That is not a
  mirror reflection and it is very probably an upstream slip, but a source volume
  never samples outside [0, 1] (``r = 0.5 + p/size`` with ``|p| <= size/2``) so the
  branch is unreachable from ``amp_file_func``, and reproducing it costs nothing.
* ``mirrorindex`` reflects a negative index as ``-1 - i``, not as ``-i``. With
  ``n == 1`` the naive form would index ``-1``; the real one stays at 0.
* The array is read ROW-MAJOR as ``data[((x*ny + y)*nz + z)]``, HDF5's order and
  therefore C-contiguous NumPy's, so a NumPy array of shape ``(nx, ny, nz)`` indexes
  directly.

MEASURED against pristine MEEP 1.33.0 (single precision) on ``test_source.py``'s own
``TestAmpFileFunc`` construction — a 1x1 cell at resolution 60, a 0.3 x 0.2 source at
(0.1, 0.2), a 100x200x1 complex amp_data, ``Ez`` read at the origin after t=200:

    MEEP amp_data                    7.165555143728852e-05
    MEEP amp_func = this transcription 7.165555143728852e-05   difference 0.0 EXACTLY
    MEEP amp_func_file (the same data through HDF5)  7.16554350219667e-05
                                                    difference 1.164e-10

The last gap is the fixture's float32 storage, not the arithmetic: MEEP reads the
file as ``realnum``.

WHAT IS NOT HERE, and why. ``amp_func_file`` itself is not read by this package —
``meep_gpu`` carries no HDF5 dependency (``test_package_boundary`` enforces it), so
the file stays the caller's to open. :func:`~.from_meep.gpu_compatibility` says so by
name and points at this module: read the ``<dataset>.re`` / ``<dataset>.im`` pair
yourself and pass ``amp_data``, and the two routes agree to the digits above.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy


def mirror_index(i: int, n: int) -> int:
    """``meep::mirrorindex`` (src/fields.cpp:755), verbatim.

    Mirror boundary conditions for an index outside ``0..n-1``. The negative branch
    is ``-1 - i`` rather than ``-i``: index -1 reflects to 0, which is what keeps a
    single-entry axis (``n == 1``) in range.
    """
    if i >= n:
        return 2 * n - 1 - i
    return -1 - i if i < 0 else i


def map_coordinates(
    rx: float, ry: float, rz: float, nx: int, ny: int, nz: int
) -> tuple[int, int, int, int, int, int, float, float, float]:
    """``meep::map_coordinates`` (src/fields.cpp:759-789) with ``do_fabs=true``.

    Returns ``(x1, y1, z1, x2, y2, z2, dx, dy, dz)`` — the two bracketing indices per
    axis and the already-absolute weights of the second one.
    """
    rx = -rx if rx < 0.0 else (1.0 - rx if rx > 1.0 else rx)
    ry = -ry if ry < 0.0 else (1.0 - ry if ry > 1.0 else ry)
    rz = -rz if rz < 0.0 else (1.0 - rz if rz > 1.0 else rz)
    # int() in C truncates toward zero; the arguments here are non-negative after the
    # fold above, so Python's int() on a float is the same operation.
    x1 = mirror_index(int(rx * nx), nx)
    y1 = mirror_index(int(ry * ny), ny)
    z1 = mirror_index(int(rz * nz), nz)
    dx = rx * nx - x1 - 0.5
    dy = ry * ny - y1 - 0.5
    dz = rz * nz - z1 - 0.5
    x2 = mirror_index(x1 + 1 if dx >= 0.0 else x1 - 1, nx)
    y2 = mirror_index(y1 + 1 if dy >= 0.0 else y1 - 1, ny)
    z2 = mirror_index(z1 + 1 if dz >= 0.0 else z1 - 1, nz)
    return x1, y1, z1, x2, y2, z2, abs(dx), abs(dy), abs(dz)


def linear_interpolate(rx: float, ry: float, rz: float, data: Any) -> float:
    """``meep::linear_interpolate`` (src/fields.cpp:792-812) over a ``(nx, ny, nz)`` array.

    ``data`` replaces MEEP's ``(pointer, nx, ny, nz, stride)``: the shape carries the
    counts and NumPy's own indexing carries the row-major ``D(x, y, z)`` macro, so
    stride 1 — the only stride ``amp_file_func`` ever passes — is implicit.
    """
    nx, ny, nz = data.shape
    x1, y1, z1, x2, y2, z2, dx, dy, dz = map_coordinates(rx, ry, rz, nx, ny, nz)
    return float(
        (
            (data[x1, y1, z1] * (1.0 - dx) + data[x2, y1, z1] * dx) * (1.0 - dy)
            + (data[x1, y2, z1] * (1.0 - dx) + data[x2, y2, z1] * dx) * dy
        )
        * (1.0 - dz)
        + (
            (data[x1, y1, z2] * (1.0 - dx) + data[x2, y1, z2] * dx) * (1.0 - dy)
            + (data[x1, y2, z2] * (1.0 - dx) + data[x2, y2, z2] * dx) * dy
        )
        * dz
    )


def amp_func_from_array(
    values: Any, size: Sequence[float]
) -> Callable[[float, float, float], complex]:
    """``amp_file_func`` (src/sources.cpp:345-374) bound to one array and one volume.

    Args:
        values: The ``amp_data`` volume, any shape broadcastable to three axes. Real
            and imaginary parts are interpolated separately, as MEEP interpolates its
            two ``double`` buffers separately.
        size: The extent of the SOURCE volume on ``(x, y, z)``, already reduced to the
            run's dimensionality — an axis MEEP's coordinate system does not carry
            reads zero there, which is exactly what ``amp_file_func``'s switch over
            ``amp_func_vol->dim`` leaves those axes at. A zero extent pins that axis's
            fractional coordinate at 0 rather than dividing by it.

    Returns:
        ``f(x, y, z) -> complex`` taking the offset from the source CENTRE — the
        convention MEEP calls ``A(rel_loc)`` with (``src_vol_chunkloop``,
        src/sources.cpp:272-273) and the one :mod:`~.sources` already uses.
    """
    array = numpy.asarray(values)
    if array.ndim > 3:
        raise ValueError(
            f"amp_data has {array.ndim} axes; MEEP's interpolator indexes exactly three "
            f"(src/fields.cpp:792-812)."
        )
    array = array.reshape(array.shape + (1,) * (3 - array.ndim))
    real = numpy.ascontiguousarray(array.real, dtype=numpy.float64)
    imaginary = numpy.ascontiguousarray(array.imag, dtype=numpy.float64)
    x_size, y_size, z_size = (float(value) for value in size)

    def amp_func(x: float, y: float, z: float) -> complex:
        rx = 0.0 if x_size == 0.0 else 0.5 + x / x_size
        ry = 0.0 if y_size == 0.0 else 0.5 + y / y_size
        rz = 0.0 if z_size == 0.0 else 0.5 + z / z_size
        return complex(
            linear_interpolate(rx, ry, rz, real),
            linear_interpolate(rx, ry, rz, imaginary),
        )

    return amp_func
