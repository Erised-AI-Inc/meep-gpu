"""The radial sum of the CUDA cylindrical curls: a column-serial scan kernel, and its host side.

WHAT IT COMPUTES. MEEP's Z update on a Dcyl grid needs ``(1/r)·d(r·Fp)/dr`` where the
ordinary curl computes ``dFp/dr``, and MEEP builds it as ``f_rderiv_int``, a running sum
along r restarted in EVERY STRUCTURE CHUNK over that chunk's own rows (step_db.cpp:99-119).
The array path transcribes it as ``stepping.cylindrical_rderiv_difference``; this module is
that function on the device, word for word::

    inc[g]  = rinv[g-1] * ((Fp[g]*w[g]) - (Fp[g-1]*w[g-1]))         g >= 1
    F_s[s]  = +0,   F_s[g] = F_s[g-1] + inc[g]                         g > s

with ``w[g] = realnum(g + ir0)`` and ``rinv[g-1] = realnum(1.0 / ((g + ir0) - 0.5))``
(``stepping._cylindrical_rderiv_rows``, CALLED here and bound as two device rows, so the
kernel performs no division), each operation rounded once, the sum sequential along r. A
target cell reads the sum of the chunk that owns it: ``F_s[j+1] - F_s[j]`` for Bz (forward,
half-integer rows ``j``) and ``F_s[g-1] - F_s[g]`` for Dz (MEEP's negated stride, integer
rows ``g``). The cut row between two chunks therefore holds two values -- the lower chunk's
full sum and the upper chunk's zero -- so the kernel writes the DIFFERENCE, a volume shaped
like the target, and never a prefix array. Rows no segment writes keep ``+0`` (the array
path's ``out[...] = 0``).

WHICH CHUNK OWNS A ROW is the array path's own answer, ``stepping._cylindrical_scan_starts``
over the division it steps with (the lifted simulation's, else the one-rank division of the
active layer), bound as an ``int32`` line (each target row's chunk start) and its distinct
values as the segment list. A division whose r cuts differ along z has no such line -- the
array path then keys the sum per cell -- and is refused by name
(:class:`RadialScanRefusal`; bit_parity_0_10 DESIGN.md D8).

THE SCAN IS COLUMN-SERIAL (DESIGN.md D26). One lane per column of stored words -- a
(phi, z) column, and one stored part of it on complex storage, because a C-contiguous
complex64 volume viewed as float32 words is one column per part -- walks the rows in order.
A parallel scan (a tree reduction, or a library ``cumsum``) associates the sum differently
and is another float32 word on most rows; the array path's own CuPy branch is the same row
loop for that reason. Nothing here may replace the loop with one.

THE B SIDE'S WALL ROW. Ep is extended by MEEP's stored wall row (``nr + 1`` rows, the wall's
Ep held at ``+0``): the kernel reads row ``nr`` as ``+0``, so the last row's forward
difference is the local ``0 - F[nr-1]`` MEEP reads, not minus a sum over a zero ghost. The
D side's backward difference reads rows ``g - 1`` and ``g`` and needs no wall row.

NOTHING HERE IMPORTS CUPY at module scope: the kernel text, the row builders and the
starts line are pure host work, exercisable at the merge bar (a host interpreter executes
the text against the array path in ``test_cylindrical_chunk_forms``). ``cupy`` is imported
at the one compile site.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Tuple

#: ``ir0 = origin_r*a + 0.5*iyee_shift(Fp).in_direction(R)``. The r origin IS the axis
#: here, so the whole value is the half-shift: Ep sits at the node (r-shift 0) and Hp half
#: a cell out (r-shift 1). Swapped, it is a half-cell error in the radial weights and is
#: silent -- a smooth, plausible, slightly wrong field.
PREFIX_IR0: Dict[str, float] = {"step_B": 0.0, "step_D": 0.5}

#: Which sub-step's sum runs over MEEP's stored ZERO WALL ROW: only the B side, whose Bz
#: curl is a forward difference (``stepping.step_B``).
PREFIX_WALL_ROW: Dict[str, bool] = {"step_B": True, "step_D": False}

#: Which stored component each sub-step's sum is built from: Ep on the B side and Hp on
#: the D side, which are ``Ey`` and ``Hy`` under this engine's ``x -> r, y -> phi,
#: z -> z`` mapping (grid.py).
PREFIX_COMPONENT: Dict[str, str] = {"step_B": "Ey", "step_D": "Hy"}

#: The target whose whole curl is ``dtdx`` times the difference: Bz and Dz.
RADIAL_TARGET: Dict[str, str] = {"step_B": "Bz", "step_D": "Dz"}

#: Which side differences backward (D's negated stride).
RADIAL_BACKWARD: Dict[str, bool] = {"step_B": False, "step_D": True}

#: The scan kernel's entry point and the options it is compiled with. ``--fmad=false`` is
#: correctness: ``rinv * (here - prev)`` added to the running sum is a contraction candidate
#: the array path rounds twice.
RADIAL_SCAN_KERNEL = "cyl_radial_difference"
RADIAL_SCAN_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: Lanes per block; one lane per column of stored words.
RADIAL_SCAN_THREADS = 128

#: The kernel partition (``test_kernel_partition``). The scan is new with this module and
#: has no gate verdict yet. Spelled as a dict WITHOUT a type annotation: the partition
#: readers walk the syntax tree, and an annotated assignment is an ``ast.AnnAssign`` they
#: do not match.
CERTIFIED_KERNELS = ()
UNCERTIFIED_KERNELS = {
    "cyl_radial_difference":
        "gated by parity/meep_gpu/gate_cuda_cylindrical_scan.py under both float32 "
        "subnormal policies; the certification block that would move this name into "
        "CERTIFIED_KERNELS is cut from that gate's artifacts in the certification round",
}

#: The column-serial scan, ``stepping.cylindrical_rderiv_difference`` with
#: ``_cylindrical_segment_scan``, transcribed. ``rows_src`` is ``rows_stored + 1`` on the B
#: side (its extra row is the wall, read as ``+0``) and ``rows_stored`` on the D side;
#: ``n_target`` is ``rows_src - 1`` forward and ``rows_src`` backward, which keeps every
#: written row inside the target. Pure ASCII (NVRTC reads the source through the locale).
RADIAL_SCAN_SOURCE = r'''
// stepping.cylindrical_rderiv_difference (MEEP step_db.cpp:99-119): the per-chunk radial
// sum of (1/r)d(r*Fp)/dr and the difference a target cell reads from ITS chunk's sum.
// One lane per column of stored words (per part on complex storage), rows in order.
extern "C" __global__ void cyl_radial_difference(
    const float* __restrict__ fp, const float* __restrict__ w,
    const float* __restrict__ rinv, const int* __restrict__ starts,
    const int* __restrict__ segs, float* __restrict__ out,
    int rows_stored, int rows_src, int n_target, int ncol, int n_segs, int backward
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncol) return;

    // out[...] = 0: a row no segment writes keeps +0.
    for (int g = 0; g < n_target; ++g) {
        out[g * ncol + idx] = 0.0f;
    }
    // Per chunk start s: F = +0 at row s, then one rounded add per row. The chunk's sum
    // runs past its end too; those rows are never read (their starts name another chunk).
    for (int t = 0; t < n_segs; ++t) {
        const int s = segs[t];
        float total = 0.0f;
        float prev = w[s] * (s < rows_stored ? fp[s * ncol + idx] : 0.0f);
        for (int g = s + 1; g < rows_src; ++g) {
            float here = w[g] * (g < rows_stored ? fp[g * ncol + idx] : 0.0f);
            float after = total + (rinv[g - 1] * (here - prev));
            // Forward (Bz): row g - 1 reads F[g] - F[g-1]. Backward (Dz): row g reads
            // F[g-1] - F[g]. Only the rows whose chunk starts at s take this sum.
            const int row = backward ? g : g - 1;
            if (starts[row] == s) {
                out[row * ncol + idx] = backward ? (total - after) : (after - total);
            }
            total = after;
            prev = here;
        }
    }
}
'''


class RadialScanRefusal(ValueError):
    """A division the scan kernel cannot read: the array path steps it."""


def _require_sub_step(sub_step: str) -> None:
    if sub_step not in PREFIX_IR0:
        raise ValueError(f"sub_step must be one of {sorted(PREFIX_IR0)}, got {sub_step!r}")


def prefix_rows(sub_step: str, radial_rows: int) -> int:
    """How many rows this sub-step's sum runs over -- ``nr`` or ``nr + 1`` (the wall)."""
    _require_sub_step(sub_step)
    return int(radial_rows) + (1 if PREFIX_WALL_ROW[sub_step] else 0)


def radial_rows(sub_step: str, rows_stored: int) -> Tuple[Any, Any]:
    """``(w, rinv)``: the scan's two invariant row vectors, host float32, flat.

    ``stepping._cylindrical_rderiv_rows`` -- the array path's own builder, called, not
    re-derived -- over the sum's source rows (``rows_stored + 1`` on the B side, Ep and its
    wall row) at the side's ``ir0``.
    """
    import numpy  # noqa: PLC0415 - host-side

    from ..stepping import _cylindrical_rderiv_rows  # noqa: PLC0415 - NumPy only

    rows_src = prefix_rows(sub_step, rows_stored)
    weights, rinv = _cylindrical_rderiv_rows(numpy, rows_src, PREFIX_IR0[sub_step],
                                             numpy.dtype(numpy.float32))
    return (numpy.ascontiguousarray(weights.reshape(-1), dtype=numpy.float32),
            numpy.ascontiguousarray(rinv.reshape(-1), dtype=numpy.float32))


def radial_starts(fields: Any, pml: Any, sub_step: str) -> Tuple[Any, Any]:
    """``(starts, segments)``: per target row its chunk's first stored r row, and the
    distinct starts, as host int32 vectors.

    ``stepping._cylindrical_scan_starts`` of the target (Bz or Dz) -- the array path's own
    function, over the division it steps with. ``None`` (one chunk, starting at the axis)
    becomes the all-zero line, which selects the same rows. A per-cell answer (a division
    whose r cuts differ along z) is refused by name: no per-row line describes it.
    """
    import numpy  # noqa: PLC0415 - host-side

    from ..stepping import _cylindrical_scan_starts, _host_ints  # noqa: PLC0415

    _require_sub_step(sub_step)
    target = RADIAL_TARGET[sub_step]
    n_target = int(getattr(fields, target).shape[0])
    starts = _cylindrical_scan_starts(fields, pml, target)
    if starts is None:
        line = numpy.zeros(n_target, dtype=numpy.int32)
    else:
        values = numpy.asarray(_host_ints(starts))
        if values.ndim != 3 or values.shape[1:] != (1, 1) or values.shape[0] != n_target:
            raise RadialScanRefusal(
                f"the chunk division's r cuts differ along z (the array path keys "
                f"{target}'s radial sum per cell, shape {values.shape}); the column-serial "
                f"scan reads one chunk start per r row, so the array path steps it")
        line = numpy.ascontiguousarray(values.reshape(-1), dtype=numpy.int32)
    segments = numpy.ascontiguousarray(numpy.unique(line), dtype=numpy.int32)
    if int(segments.min()) < 0 or int(segments.max()) >= n_target:
        raise RadialScanRefusal(f"a chunk start {segments.tolist()} lies outside the "
                                f"{n_target} rows of {target}")
    return line, segments


def radial_scan_tables(fields: Any, pml: Any, sub_step: str) -> Dict[str, Any]:
    """Everything the scan binds for one sub-step, as host arrays and ints.

    Grid invariants (the radial extent, the side, the division), so a plan builds them
    once. The device copies are the caller's (:func:`radial_scan_device_tables`).
    """
    _require_sub_step(sub_step)
    source = getattr(fields, PREFIX_COMPONENT[sub_step])
    rows_stored = int(source.shape[0])
    weights, rinv = radial_rows(sub_step, rows_stored)
    starts, segments = radial_starts(fields, pml, sub_step)
    return {"w": weights, "rinv": rinv, "starts": starts, "segs": segments,
            "rows_stored": rows_stored, "rows_src": prefix_rows(sub_step, rows_stored),
            "n_target": int(getattr(fields, RADIAL_TARGET[sub_step]).shape[0]),
            "backward": bool(RADIAL_BACKWARD[sub_step])}


def radial_scan_device_tables(tables: Dict[str, Any], xp: Any) -> Dict[str, Any]:
    """:func:`radial_scan_tables` with its four vectors moved onto ``xp``'s device."""
    out = dict(tables)
    for name in ("w", "rinv", "starts", "segs"):
        out[name] = xp.ascontiguousarray(xp.asarray(tables[name]))
    return out


def get_radial_scan_kernel(source: str = None) -> Any:
    """The compiled scan, through the shared memo (source and policy in the key)."""
    import cupy as cp  # noqa: PLC0415 - the one compile site

    from .compile_cache import get_or_compile, kernel_cache_key  # noqa: PLC0415

    text = RADIAL_SCAN_SOURCE if source is None else source
    key = kernel_cache_key(RADIAL_SCAN_KERNEL, False, RADIAL_SCAN_OPTIONS, text)
    return get_or_compile(
        key, lambda: cp.RawKernel(text, RADIAL_SCAN_KERNEL, options=RADIAL_SCAN_OPTIONS))


def _words(array: Any, xp: Any) -> Any:
    """A C-contiguous volume's float32 words: itself, or its complex64 parts interleaved."""
    if not array.flags.c_contiguous:
        raise ValueError("the radial scan reads a C-contiguous volume")
    if array.dtype.kind == "c":
        return array.view(xp.float32)
    if array.dtype != xp.float32:
        raise ValueError(f"the radial scan reads float32 words, got {array.dtype}")
    return array


def launch_radial_scan(kernel: Any, source: Any, tables: Dict[str, Any], out: Any,
                       xp: Any) -> Any:
    """One scan launch over ``source`` (Ep or Hp) into ``out`` (shaped like the target).

    ``tables`` are :func:`radial_scan_device_tables`'. The grid is sized from the COLUMN
    count, one lane per column of words.
    """
    import numpy  # noqa: PLC0415 - scalar types

    words = _words(source, xp)
    out_words = _words(out, xp)
    rows_stored = int(tables["rows_stored"])
    if int(words.shape[0]) != rows_stored or int(out_words.shape[0]) != int(tables["n_target"]):
        raise ValueError(
            f"the scan was built for {rows_stored} source rows and {tables['n_target']} "
            f"target rows, handed {words.shape[0]} and {out_words.shape[0]}")
    ncol = int(words.size) // rows_stored
    if int(out_words.size) // int(tables["n_target"]) != ncol:
        raise ValueError("the source and the difference do not share a column layout")
    blocks = (ncol + RADIAL_SCAN_THREADS - 1) // RADIAL_SCAN_THREADS
    kernel((blocks,), (RADIAL_SCAN_THREADS,), (
        words, tables["w"], tables["rinv"], tables["starts"], tables["segs"], out_words,
        numpy.int32(rows_stored), numpy.int32(tables["rows_src"]),
        numpy.int32(tables["n_target"]), numpy.int32(ncol),
        numpy.int32(int(tables["segs"].size)), numpy.int32(1 if tables["backward"] else 0),
    ))
    return out


def radial_difference(fields: Any, pml: Any, sub_step: str, tables: Dict[str, Any] = None,
                      out: Any = None, scratch: Any = None, kernel: Any = None) -> Any:
    """This launch's radial difference, the volume Bz's / Dz's curl is ``dtdx`` times.

    ``stepping.cylindrical_rderiv_difference`` on the device, over the CURRENT Ep (B side)
    or Hp (D side). ``tables`` may be handed in by a plan that built them once
    (:func:`radial_scan_device_tables`); ``out`` by a caller that owns the buffer, else it
    is the pool's ``cyl_radial_kernel`` slot (valid until the next call) or a fresh volume.
    """
    _require_sub_step(sub_step)
    xp = fields.grid.xp
    source = getattr(fields, PREFIX_COMPONENT[sub_step])
    if tables is None:
        tables = radial_scan_device_tables(radial_scan_tables(fields, pml, sub_step), xp)
    shape = (int(tables["n_target"]),) + tuple(int(n) for n in source.shape[1:])
    if out is None:
        out = (xp.empty(shape, dtype=source.dtype) if scratch is None
               else scratch.take("cyl_radial_kernel", shape, source.dtype))
    return launch_radial_scan(get_radial_scan_kernel() if kernel is None else kernel,
                              source, tables, out, xp)
