"""Raw-CUDA device code for the three in-seam passes: the wall wipe and two ghost fills.

WHAT WAS MISSING. Between a curl and the constitutive update that closes it, the
driver runs three passes (driver.py:3284-3286 on the B side, :3300-3302 on the
D side)::

    step_B -> magnetic sources -> fill_symmetry_bc_B
                               -> zero_metal_B
                               -> fill_folded_far_ghosts_B -> update_H

Measured 2026-08-21, counting the NON-TEST modules of each kernel track that
name the pass -- and, for the CUDA column, that carry DEVICE CODE for it:

    pass                      triton  metal  cuda (naming / with device code)
    zero_metal                  19      21     1 / 0
    fill_symmetry               12      17     2 / 0
    fill_folded_far_ghosts      11      17     0 / 0

The three CUDA hits are PROSE and were read rather than counted: five copies of
``// NOTE: No cell masking for PML case - _fill_symmetry_bc_B handles non-owned
cells`` in ``step_curl_kernels.py``, one sentence in its own module comment about
``zero_metal_*`` running in the seam, and one refusal string in ``coverage.py``.
Not one line of device code among them, so on the hand-CUDA track every one of
these passes was host-side array work.
``step_curl_kernels.py`` says it outright beside its PML masking -- "No cell
masking for PML case - _fill_symmetry_bc_B handles non-owned cells" -- which is
the seam stated as a dependency: the kernel leaves a plane wrong on purpose and
a HOST pass repairs it before the next sub-step reads it. Until these exist as
device code a CUDA sub-step returns to the host between the curl and the
constitutive BY CONSTRUCTION, and there is nothing for a fused pair to absorb.

=============================================================================
THIS IS NOT A FUSED PRODUCT AND MUST NOT BE READ AS ONE
=============================================================================

Nothing here fuses. Each pass is its own launch, gated on its own, against the
array-path function it transcribes. What this file supplies is the PARTS a fused
CUDA pair would have to absorb -- and the parts, measured separately, are the
only thing that would make such a fusion a transcription rather than a rewrite.

Nor does anything DISPATCH them: no module in ``meep_gpu/`` imports
``cuda_kernels`` at all (``test_package_boundary.py`` pins the absence in both
directions). Certification here is a statement about six kernels, not about any
run.

=============================================================================
WHAT IS TRANSCRIBED, AND FROM WHERE
=============================================================================

Every rule below cites ``../stepping.py`` by name and line, read off the file on
2026-08-21 (the module was renamed that morning, so the numbers were re-read
rather than inherited):

* the wall wipe        ``stepping._zero_metal`` (:2206-2247), driver call sites
                       ``zero_metal_B`` (driver.py:3285) and ``zero_metal_D``
                       (:3301)
* the near ghost fill  ``stepping._fill_symmetry_ghost_cells`` (:1426-1450) and
                       ``_write_mirror_ghost`` (:1449), behind
                       ``fill_symmetry_bc_B`` (:1399) / ``_D`` (:1387), called at
                       driver.py:3284 and :3300
* the far ghost fill   ``stepping._fill_folded_far_ghosts`` (:1489-1541) behind
                       ``fill_folded_far_ghosts_B`` (:1484) / ``_D`` (:1478),
                       called at driver.py:3286 and :3302
* which axis has a ghost at all  ``stepping._stored_past_owned`` (:1454-1469)
* which stored row it images     ``stepping._far_reflect_rows`` (:1661-1692)
* the per-component sign         ``fields.mirror_parity`` (fields.py:117-157)
* the near source row            ``stepping.MIRROR_SOURCE_INDEX`` (:159)

The predicate and every per-axis derivation live in the CUPY-FREE sibling
``in_seam_coverage.py``, so they are importable, exercisable and mutable on a
laptop; this module is the device text and the launch.

=============================================================================
THE FOUR THINGS THAT DECIDE BIT-IDENTITY HERE
=============================================================================

1. THE PARITY IS A MULTIPLY BY EXACTLY +/-1.0f, AND IT IS A MULTIPLY.
   The array path writes ``phase * field[...]`` with ``phase`` a Python int
   (stepping.py:1497, :1585); under NumPy/CuPy weak promotion that is a float32
   elementwise MULTIPLY, not a copy and not a sign flip.

   A COPY IS NOT THE SAME BYTES UNDER THE FLUSH POLICY, which is why the
   ``phase == +1`` case is NOT special-cased into an assignment. Under
   ``"flush"`` CuPy compiles with ``-ftz=true`` (subnormal_policy.py header),
   so ``1.0f * x`` FLUSHES a subnormal operand to zero on both paths; a copy
   would preserve it and diverge on exactly the values the policy exists to
   decide. The gate arms this as ``copy_instead_of_multiply``, whose verdict is
   EXPECTED TO DIFFER BY POLICY and is reported as a measurement rather than
   scored as a clause.

   Negation is likewise spelled ``-phase`` on the SCALAR, once, and the parity
   then multiplies -- never ``-(phase * x)``, which would put a ``neg.f32``
   after an operation the array path has no counterpart for.

2. THE AXES ARE APPLIED IN X, Y, Z ORDER, ONE LAUNCH EACH, FOR THE TWO FILLS.
   ``_fill_symmetry_ghost_cells`` walks ``for term ... for axis in range(3)``,
   so a component unowned on two planes has its corner written twice and carries
   the PRODUCT of both parities -- the doubly mirrored value MEEP's
   ``connect_the_chunks`` produces. A grid of programs cannot promise that order
   inside one launch, so ``in_seam_coverage.plan`` returns a list and
   :func:`run_pass` walks it. The gate plants ``reverse_axis_order``.

   ZERO_METAL IS ONE LAUNCH FOR ALL THREE AXES, and that is a different fact
   rather than a laxer one: its write is the constant ``+0.0f`` and it reads
   nothing, so two axes' planes commute exactly.

3. THE THREE PASSES ARE THREE LAUNCHES AND MUST NOT BE MERGED. They are not
   independent: on a grid with a mirrored X and a metallic Y, ``Dz`` is
   near-filled on X (plane ``Dz[0, :, :]``) and wall-wiped on Y (plane
   ``Dz[:, 0, :]``), and the two planes SHARE the edge ``Dz[0, 0, :]``. The
   driver runs the fill first and the wipe second, so that edge ends at zero.
   One launch doing both is a race on that edge. The gate's fixture carries
   exactly that configuration.

4. THE FAR FILL'S IMAGE ROW IS DERIVED, NEVER BAKED. ``n_full - stored + 2`` is
   ``stored - 2`` at an even full count and ``stored - 3`` at an odd one, because
   ``big_corner`` IS the second mirror at an even count and sits one half-cell
   above it at an odd one. The fixed ``n - 2`` reflects about the window top
   instead of about the mirror -- a whole cell wrong on every odd-count run, and
   invisible to a sweep that carries one count parity. The gate plants
   ``reflect_row_n_minus_two`` and carries both parities.

=============================================================================
THE B/D INVERSION, TRANSCRIBED AFRESH ON BOTH SIDES
=============================================================================

``IYEE_SHIFTS`` (fields.py:214-219) makes the two families exact opposites::

    Bx (0,1,1)  By (1,0,1)  Bz (1,1,0)   -> shift 0 on its OWN axis only
    Dx (1,0,0)  Dy (0,1,0)  Dz (0,0,1)   -> shift 0 on the OTHER TWO axes

so on one folded axis ``a`` the near fill (``shifts[a] == 0``) reaches ONE B
component and TWO D components, and the far fill (``shifts[a] == 1``) reaches
TWO B components and ONE D component. THE D SIDE IS NOT THE B SIDE MIRRORED and
was written from the table, not from the B kernel: the fusion track measured
this inversion this week and it is precisely where an argument by analogy is
plausible and wrong.

=============================================================================
PLATFORM FACTS THIS FILE DEPENDS ON (CUDA path)
=============================================================================

* ``--fmad=false`` is carried for CONSISTENCY WITH THE TRACK, not because these
  kernels contract anything: there is no ``a*b + c`` anywhere below -- every
  device expression is a single multiply or a store. The gate's unguarded
  control is therefore EXPECTED to be identical here, and the record says so
  rather than implying the guard bought something.
* THE DEVICE STRINGS ARE PURE ASCII, a compile requirement rather than a style
  rule: ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source with a bare
  ``open(..., 'w')``, so the bytes go through the interpreter's LOCALE encoding
  -- ASCII under C/POSIX, which is what a non-interactive shell on the
  validation host gets. Two em-dashes in a comment once killed a kernel at its
  first launch with ``UnicodeEncodeError``. ``test_in_seam_passes.py`` scans and
  performs the encode.
* DIVISION: the index decomposition divides on INTEGERS (``t / nz``), never on
  floats, so the ``div.rn.f32`` / ptxas-FTZ exception on record does not apply.
* SIGNED ZERO: CUDA does not canonicalize it. ``-phase`` on the host-supplied
  scalar is exact for +/-1.0f and nothing below relies on the answer for zero.

=============================================================================
A KERNEL WITH A GATE VERDICT MOVES SETS
=============================================================================

``CERTIFIED_KERNELS`` needs a record block in ``certification.json``;
``UNCERTIFIED_KERNELS`` names why not. A kernel in neither set fails
``test_in_seam_passes.py``, and so does one in ``CERTIFIED_KERNELS`` with no
record.
"""

import cupy as cp
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid

# The predicate and the per-axis derivations live in the CuPy-free sibling so
# they stay importable on a machine with no GPU. The by-path fallback is for the
# bit-identity probe, which loads these modules outside the package.
try:
    from .in_seam_coverage import (FAMILIES, FAMILY_COMPONENTS,
                                   MIRROR_SOURCE_INDEX, PASSES, covers, plan,
                                   plane_extent)
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    _spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_in_seam_coverage",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "in_seam_coverage.py"))
    _in_seam_coverage = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_in_seam_coverage)
    FAMILIES = _in_seam_coverage.FAMILIES
    FAMILY_COMPONENTS = _in_seam_coverage.FAMILY_COMPONENTS
    MIRROR_SOURCE_INDEX = _in_seam_coverage.MIRROR_SOURCE_INDEX
    PASSES = _in_seam_coverage.PASSES
    covers = _in_seam_coverage.covers
    plan = _in_seam_coverage.plan
    plane_extent = _in_seam_coverage.plane_extent

try:
    from . import compile_cache
except ImportError:  # loaded by path, outside the package
    import importlib.util as _importlib_util
    import os as _os

    _cache_spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_compile_cache",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "compile_cache.py"))
    compile_cache = _importlib_util.module_from_spec(_cache_spec)
    _cache_spec.loader.exec_module(compile_cache)

#: Byte-identical to the array path with a gate verdict behind it, cut 2026-08-21
#: on an RTX A6000 under BOTH float32 subnormal policies. The evidence is
#: ``certification.json``'s ``in_seam_2026-08-21`` block; the artifacts are
#: ``parity/meep_gpu/results/cuda_in_seam_2026-08-21/``. Each pass was gated ON ITS
#: OWN against the ``stepping`` function it transcribes -- 20 cases for the wipe, 72
#: for the near fill, 52 for the far fill, every one identical at one launch and
#: over 60 -- and NONE of those verdicts is a verdict about the three composed.
#: An entry here WITHOUT a record entry is the failure ``test_in_seam_passes.py``
#: exists to catch.
CERTIFIED_KERNELS = ("zero_metal_B", "zero_metal_D",
                     "fill_symmetry_B", "fill_symmetry_D",
                     "fill_folded_far_B", "fill_folded_far_D")

#: Shipped but not gated, each with the reason. EMPTY, and the partition test
#: requires every shipped kernel to be in exactly one of the two sets -- so a
#: kernel added to this file without a verdict fails there rather than shipping
#: unmeasured. Spelled as a plain dict WITHOUT a type annotation: the partition
#: test reads it off the syntax tree, without importing the module, and an
#: annotated assignment is an ``ast.AnnAssign``, which that reader does not match.
UNCERTIFIED_KERNELS = {}


# =============================================================================
# THE DEVICE CODE
# =============================================================================
#
# THE PLANE DECOMPOSITION, ONCE. Every kernel below addresses ONE FACE of a
# C-contiguous (nx, ny, nz) volume. For axis `a` the face holds every cell whose
# `a` coordinate is fixed, and one thread owns one cell of it:
#
#     axis 0 (x):  plane = ny*nz   stride = ny*nz   base = t
#     axis 1 (y):  plane = nx*nz   stride = nz      base = (t/nz)*(ny*nz) + (t%nz)
#     axis 2 (z):  plane = nx*ny   stride = 1       base = (t/ny)*(ny*nz) + (t%ny)*nz
#
# `base` is the linear offset of the face's cell at coordinate 0 along `a`, and
# `stride` steps one cell along `a`, so row `r` of the face is `base + r*stride`.
# The divisions are INTEGER divisions on the thread index; nothing here divides a
# float.
#
# WHY THE AXIS IS A RUNTIME ARGUMENT and not a compile-time constant: one device
# string per (pass, family) rather than three, so one digest covers every fold
# orientation and the certification record cannot end up pinning a per-axis
# variant nobody ran. The branch is uniform across the whole launch.

_PLANE_PRELUDE = r'''
// The face decomposition, shared by all six kernels. `axis` is uniform across
// the launch. Returns the face's element count; `base` and `stride` are the
// linear offset of row 0 and the step of one row along `axis`.
__device__ __forceinline__ int face_geometry(
    int t, int nx, int ny, int nz, int axis, int* base, int* stride
) {
    if (axis == 0) {
        *stride = ny * nz;
        *base = t;
        return ny * nz;
    }
    if (axis == 1) {
        *stride = nz;
        *base = (t / nz) * (ny * nz) + (t % nz);
        return nx * nz;
    }
    *stride = 1;
    *base = (t / ny) * (ny * nz) + (t % ny) * nz;
    return nx * ny;
}
'''

# ---------------------------------------------------------------------------
# Pass 1: the metallic wall wipe
# ---------------------------------------------------------------------------
#
# stepping._zero_metal (stepping.py:2206-2247):
#
#     if not grid.has_metallic: return
#     walled = [a for a in range(3) if grid.is_metallic(a) and not grid.is_mirrored(a)]
#     for name in components:
#         shifts = IYEE_SHIFTS[name]
#         for axis in walled:
#             if shifts[axis] == 0:
#                 array[_face(axis, 0)] = 0
#
# STORED CELL 0 IS THE LOW WALL. The HIGH wall is MEEP's slot N, which this
# engine does not store: `_shift_up` supplies it as the zero ghost, so the pair
# of walls is complete without the extra plane and this kernel writes ONE face
# per walled axis.
#
# THE WALLED SET EXCLUDES A FOLDED METALLIC AXIS and the host decides that
# (`in_seam_coverage.zero_metal_axes`). Writing a wall there does not impose a
# boundary, it destroys the fold: 1.28e+00 complex relative L2 against CPU MEEP
# with the fold's ghost zeroed, against 2.0e-07 with the skip in place.
#
# THE VALUE IS `+0.0f`. The array path assigns the Python int 0 into a float32
# array, which is +0.0 and never -0.0; the gate plants `write_negative_zero`.
#
# ONE LAUNCH, THREE AXES, and the three flags are runtime ints. A store of a
# constant reads nothing, so a cell on two walled faces is written zero twice in
# either order.

_zero_metal_B_kernel_code = _PLANE_PRELUDE + r'''
extern "C" __global__ void zero_metal_B(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    int nx, int ny, int nz, int wall_x, int wall_y, int wall_z
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int base, stride;

    // IYEE_SHIFTS: Bx (0,1,1), By (1,0,1), Bz (1,1,0). A B component has Yee
    // shift 0 on its OWN axis and nowhere else, so exactly one component sits on
    // each wall: Bx on the x wall, By on y, Bz on z.
    if (wall_x) {
        int plane = face_geometry(t, nx, ny, nz, 0, &base, &stride);
        if (t < plane) Bx[base] = 0.0f;
    }
    if (wall_y) {
        int plane = face_geometry(t, nx, ny, nz, 1, &base, &stride);
        if (t < plane) By[base] = 0.0f;
    }
    if (wall_z) {
        int plane = face_geometry(t, nx, ny, nz, 2, &base, &stride);
        if (t < plane) Bz[base] = 0.0f;
    }
}
'''

_zero_metal_D_kernel_code = _PLANE_PRELUDE + r'''
extern "C" __global__ void zero_metal_D(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    int nx, int ny, int nz, int wall_x, int wall_y, int wall_z
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int base, stride;

    // IYEE_SHIFTS: Dx (1,0,0), Dy (0,1,0), Dz (0,0,1). A D component has Yee
    // shift 0 on the OTHER TWO axes, so TWO components sit on each wall -- the
    // exact inverse of the B family above, read off the table and not off it.
    // x wall: Dy and Dz.  y wall: Dx and Dz.  z wall: Dx and Dy.
    if (wall_x) {
        int plane = face_geometry(t, nx, ny, nz, 0, &base, &stride);
        if (t < plane) { Dy[base] = 0.0f; Dz[base] = 0.0f; }
    }
    if (wall_y) {
        int plane = face_geometry(t, nx, ny, nz, 1, &base, &stride);
        if (t < plane) { Dx[base] = 0.0f; Dz[base] = 0.0f; }
    }
    if (wall_z) {
        int plane = face_geometry(t, nx, ny, nz, 2, &base, &stride);
        if (t < plane) { Dx[base] = 0.0f; Dy[base] = 0.0f; }
    }
}
'''

# ---------------------------------------------------------------------------
# Pass 2: the near mirror-ghost fill
# ---------------------------------------------------------------------------
#
# stepping._fill_symmetry_ghost_cells (stepping.py:1426-1450):
#
#     for term in terms:
#         for axis in range(3):
#             if parities[axis] is None or shifts[axis] != 0: continue
#             field[_face(axis, 0)] = mirror_parity(target, axis, phase) * field[_face(axis, 2)]
#
# OWNERSHIP is MEEP's little_owned_corner0(c) = little_corner + 2 - iyee_shift(c);
# on a halved grid (io = -2) that is -iyee_shift, so cell 0 is unowned exactly
# where the component's Yee shift on the axis is 0. The source row is
# MIRROR_SOURCE_INDEX = 2 (stepping.py:160).
#
# THE PARITY IS `+phase` AND NOTHING ELSE ON THIS PASS. mirror_parity(c, a, ph)
# == ph * (1 - 2*iyee[c][a]) -- measured over all 72 component/axis/phase
# combinations against fields.mirror_parity -- and this pass only ever touches
# shift-0 components, so the factor is the plane's declared phase itself.
#
# ONE LAUNCH PER FOLDED AXIS, X then Y then Z. The corner of a doubly unowned
# component carries the PRODUCT of both parities because the second axis's fill
# reads a plane the first already wrote, and that is only reproducible if the
# axes are ordered.

_fill_symmetry_B_kernel_code = _PLANE_PRELUDE + r'''
extern "C" __global__ void fill_symmetry_B(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    int nx, int ny, int nz, int axis, float phase
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int base, stride;
    int plane = face_geometry(t, nx, ny, nz, axis, &base, &stride);
    if (t >= plane) return;

    // Yee shift 0 on `axis` -> parity = phase * (1 - 2*0) = phase.
    // B family: exactly the component whose own axis IS `axis` (Bx (0,1,1) on x,
    // By (1,0,1) on y, Bz (1,1,0) on z).
    float* f = (axis == 0) ? Bx : ((axis == 1) ? By : Bz);

    // MIRROR_SOURCE_INDEX = 2: cell 0 images stored cell 2.
    // A MULTIPLY, never a copy -- see note 1 in the module header.
    f[base] = phase * f[base + 2 * stride];
}
'''

_fill_symmetry_D_kernel_code = _PLANE_PRELUDE + r'''
extern "C" __global__ void fill_symmetry_D(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    int nx, int ny, int nz, int axis, float phase
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int base, stride;
    int plane = face_geometry(t, nx, ny, nz, axis, &base, &stride);
    if (t >= plane) return;

    // Yee shift 0 on `axis` -> parity = phase, the same factor as the B kernel.
    // D family: the TWO components whose own axis is NOT `axis` -- Dx (1,0,0) is
    // shift 0 on y and z, Dy (0,1,0) on x and z, Dz (0,0,1) on x and y. TWO
    // arrays here where the B kernel has one; that inversion is the table's, and
    // is written from it.
    float* g0 = (axis == 0) ? Dy : Dx;              // x -> Dy, y -> Dx, z -> Dx
    float* g1 = (axis == 2) ? Dy : Dz;              // x -> Dz, y -> Dz, z -> Dy

    g0[base] = phase * g0[base + 2 * stride];
    g1[base] = phase * g1[base + 2 * stride];
}
'''

# ---------------------------------------------------------------------------
# Pass 3: the folded-periodic far ghost fill
# ---------------------------------------------------------------------------
#
# stepping._fill_folded_far_ghosts (stepping.py:1489-1541):
#
#     axes = [a for a in range(3) if _stored_past_owned(grid, a)]
#     for term in terms:
#         for axis in axes:
#             if shifts[axis] != 1: continue
#             field[_face(axis, -1)] = mirror_parity(target, axis, phase) * field[_face(axis, reflect_row)]
#
# WHERE THE PASS RUNS AT ALL: only on a FOLDED PERIODIC axis, at either full-count
# parity. That is `stepping._stored_past_owned` (:1454-1469) -- the one place
# `Grid.stored_cells` exceeds `Grid.owned_cells`, because the stored array carries
# MEEP's big_corner plane AND the shift-1 slot half a cell past it (update_ntot's
# num + 1, vec.cpp:293-296) that connect_the_chunks fills. A folded METALLIC axis
# has no such slot and keeps its unstored zero ghost.
#
# THE PARITY IS `-phase` ON THIS PASS. Only shift-1 components are touched, so
# mirror_parity = phase * (1 - 2*1) = -phase. Spelled as a negation of the SCALAR,
# once, and then multiplied -- never as a negation of the product.
#
# THE IMAGE ROW IS `n_full - stored + 2`, DERIVED ON THE HOST
# (`stepping._far_reflect_rows`, :1661-1692), and it is `stored - 2` at an even
# full count and `stored - 3` at an odd one. Both parities are swept; the fixed
# `n - 2` is planted as a mutation.
#
# THE COMPONENT SETS ARE THE NEAR FILL'S, EXCHANGED BETWEEN THE FAMILIES, and
# that is the fusion track's measured warning: on the D side the near fill reaches
# two components and the far one, and on the B side it is the other way round.
# Written from IYEE_SHIFTS on each side rather than mirrored from the other.

_fill_folded_far_B_kernel_code = _PLANE_PRELUDE + r'''
extern "C" __global__ void fill_folded_far_B(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    int nx, int ny, int nz, int axis, float phase, int reflect_row
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int base, stride;
    int plane = face_geometry(t, nx, ny, nz, axis, &base, &stride);
    if (t >= plane) return;

    // _face(axis, -1): the LAST stored row of this axis.
    int last = ((axis == 0) ? nx : ((axis == 1) ? ny : nz)) - 1;

    // Yee shift 1 on `axis` -> parity = phase * (1 - 2*1) = -phase.
    float parity = -phase;

    // B family: the TWO components whose own axis is NOT `axis` carry shift 1
    // there -- By and Bz on x, Bx and Bz on y, Bx and By on z. Two arrays here
    // where the D kernel below has one.
    float* g0 = (axis == 0) ? By : Bx;              // x -> By, y -> Bx, z -> Bx
    float* g1 = (axis == 2) ? By : Bz;              // x -> Bz, y -> Bz, z -> By

    g0[base + last * stride] = parity * g0[base + reflect_row * stride];
    g1[base + last * stride] = parity * g1[base + reflect_row * stride];
}
'''

_fill_folded_far_D_kernel_code = _PLANE_PRELUDE + r'''
extern "C" __global__ void fill_folded_far_D(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    int nx, int ny, int nz, int axis, float phase, int reflect_row
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int base, stride;
    int plane = face_geometry(t, nx, ny, nz, axis, &base, &stride);
    if (t >= plane) return;

    int last = ((axis == 0) ? nx : ((axis == 1) ? ny : nz)) - 1;
    float parity = -phase;

    // D family: exactly the component whose own axis IS `axis` carries shift 1
    // there -- Dx (1,0,0) on x, Dy (0,1,0) on y, Dz (0,0,1) on z. ONE array,
    // where the B kernel has two, and the near fill has the counts the other way
    // round on both families.
    float* f = (axis == 0) ? Dx : ((axis == 1) ? Dy : Dz);

    f[base + last * stride] = parity * f[base + reflect_row * stride];
}
'''

# NVRTC compile options. ``--fmad=false`` is carried for consistency with every
# other kernel on this track; NOTHING BELOW CONTRACTS -- there is no ``a*b + c``
# in any of the six device strings -- so the gate's unguarded control is expected
# to be identical here and the record says so rather than implying the guard
# bought an answer. Spelled here rather than imported so that loading this file by
# path cannot pick up a different tuple than the one the gate compiled.
_COMPILE_OPTIONS = ('--fmad=false',)

#: Lanes per block, matching the rest of the track (``_REAL_PML_THREADS`` in
#: ``step_curl_kernels``, ``_CONSTITUTIVE_THREADS`` in ``constitutive_kernels``,
#: ``triton_kernels/kernels.DEFAULT_BLOCK``). One element per lane over ONE FACE,
#: so a launch is two orders of magnitude smaller than a volume launch and the
#: block size is not a tuning question here.
_IN_SEAM_THREADS = 256

#: The device text of every kernel this module ships, by name. Rebuilt per call
#: where it is consumed (see :func:`_get_kernel`); this mapping is the one the
#: record and the tests read.
KERNEL_NAMES = (
    "zero_metal_B", "zero_metal_D",
    "fill_symmetry_B", "fill_symmetry_D",
    "fill_folded_far_B", "fill_folded_far_D",
)

#: Which kernel each (pass, family) launches.
KERNEL_FOR: Dict[Tuple[str, str], str] = {
    ("zero_metal", "B"): "zero_metal_B",
    ("zero_metal", "D"): "zero_metal_D",
    ("fill_symmetry", "B"): "fill_symmetry_B",
    ("fill_symmetry", "D"): "fill_symmetry_D",
    ("fill_folded_far", "B"): "fill_folded_far_B",
    ("fill_folded_far", "D"): "fill_folded_far_D",
}


def _code_map() -> Dict[str, str]:
    """The six device strings, READ FRESH ON EVERY CALL.

    That is load-bearing for the same reason it is in the sibling modules: the
    source is part of the compile memo key, so the map has to be read before a
    key exists to miss on, and the gate mutates kernels by assigning over the
    module-level source strings. A map memoized at first call would hand back the
    pre-mutation string forever -- a leg reporting a pass for a mutation it never
    applied.
    """
    return {
        "zero_metal_B": _zero_metal_B_kernel_code,
        "zero_metal_D": _zero_metal_D_kernel_code,
        "fill_symmetry_B": _fill_symmetry_B_kernel_code,
        "fill_symmetry_D": _fill_symmetry_D_kernel_code,
        "fill_folded_far_B": _fill_folded_far_B_kernel_code,
        "fill_folded_far_D": _fill_folded_far_D_kernel_code,
    }


def device_sources() -> Dict[str, str]:
    """The shipped device text by kernel name — what a record block pins."""
    return dict(_code_map())


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    Shared memo with the sibling modules -- the cache is keyed on the source
    string, so two modules cannot collide. The gate drives this between guard
    sets, where the option tuple is overridden from outside, a change no memo key
    can see.
    """
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str):
    """Compile on first use, memoized on (name, options, policy, source)."""
    code = _code_map()[name]
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def _arrays(fields: "Fields", family: str) -> Tuple[Any, Any, Any]:
    if family not in FAMILY_COMPONENTS:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    return tuple(getattr(fields, name)  # type: ignore[return-value]
                 for name in FAMILY_COMPONENTS[family])


def _blocks(elements: int) -> int:
    return (int(elements) + _IN_SEAM_THREADS - 1) // _IN_SEAM_THREADS


def launch_zero_metal(fields: "Fields", family: str,
                      walls: Sequence[int]) -> Dict[str, Any]:
    """``stepping.zero_metal_B``/``_D`` in ONE launch.

    ``walls`` are ``in_seam_coverage.zero_metal_axes`` -- the question
    ``_zero_metal`` asks (the grid's own declaration, with a folded metallic axis
    excluded), NOT the ghost-rule resolution ``_boundary_kinds`` performs. Getting
    those two confused is a wall written across a fold.

    Returns the launch geometry rather than None so a gate can assert that
    something was actually launched; a pass that ran zero blocks and reported
    success is the vacuity this whole track guards against.
    """
    arrays = _arrays(fields, family)
    nx, ny, nz = (int(n) for n in arrays[0].shape)
    if not any(int(bool(w)) for w in walls):
        return {"launched": False, "reason": "no walled axis on this grid"}
    elements = max(ny * nz, nx * nz, nx * ny)
    blocks = _blocks(elements)
    _get_kernel(KERNEL_FOR[("zero_metal", family)])(
        (blocks,), (_IN_SEAM_THREADS,),
        (arrays[0], arrays[1], arrays[2],
         np.int32(nx), np.int32(ny), np.int32(nz),
         np.int32(int(bool(walls[0]))), np.int32(int(bool(walls[1]))),
         np.int32(int(bool(walls[2])))))
    return {"launched": True, "blocks": blocks, "elements": elements}


def launch_fill_symmetry(fields: "Fields", family: str, axis: int,
                         phase: int) -> Dict[str, Any]:
    """``stepping.fill_symmetry_bc_B``/``_D`` for ONE folded axis.

    ONE AXIS PER CALL because the array path's corner carries the product of both
    parities and only an ordered walk reproduces it. :func:`run_pass` walks
    ``in_seam_coverage.plan``, which is that order.
    """
    arrays = _arrays(fields, family)
    nx, ny, nz = (int(n) for n in arrays[0].shape)
    elements = plane_extent((nx, ny, nz), int(axis))
    blocks = _blocks(elements)
    _get_kernel(KERNEL_FOR[("fill_symmetry", family)])(
        (blocks,), (_IN_SEAM_THREADS,),
        (arrays[0], arrays[1], arrays[2],
         np.int32(nx), np.int32(ny), np.int32(nz),
         np.int32(int(axis)), np.float32(phase)))
    return {"launched": True, "blocks": blocks, "elements": elements,
            "axis": int(axis), "phase": int(phase)}


def launch_fill_folded_far(fields: "Fields", family: str, axis: int, phase: int,
                           reflect_row: int) -> Dict[str, Any]:
    """``stepping.fill_folded_far_ghosts_B``/``_D`` for ONE folded-periodic axis.

    ``reflect_row`` MUST come from ``in_seam_coverage.folded_far_rows``
    (``stepping._far_reflect_rows``, ``n_full - stored + 2``). It is ``stored - 2``
    at an even full count and ``stored - 3`` at an odd one; a baked ``n - 2``
    reflects about the window top instead of about the second mirror, which is a
    whole cell wrong on every odd-count run.
    """
    arrays = _arrays(fields, family)
    nx, ny, nz = (int(n) for n in arrays[0].shape)
    stored = (nx, ny, nz)[int(axis)]
    row = int(reflect_row)
    if not (0 <= row <= stored - 2):
        raise ValueError(
            f"reflect_row {row} is outside axis {axis}'s stored extent {stored} "
            f"or lands on the slot being written; it must come from "
            f"in_seam_coverage.folded_far_rows (n_full - stored + 2)")
    elements = plane_extent((nx, ny, nz), int(axis))
    blocks = _blocks(elements)
    _get_kernel(KERNEL_FOR[("fill_folded_far", family)])(
        (blocks,), (_IN_SEAM_THREADS,),
        (arrays[0], arrays[1], arrays[2],
         np.int32(nx), np.int32(ny), np.int32(nz),
         np.int32(int(axis)), np.float32(phase), np.int32(row)))
    return {"launched": True, "blocks": blocks, "elements": elements,
            "axis": int(axis), "phase": int(phase), "reflect_row": row}


def run_pass(pass_name: str, fields: "Fields", family: str, *,
             grid: Optional["Grid"] = None,
             launches: Optional[Sequence[Dict[str, Any]]] = None) -> Tuple[Dict[str, Any], ...]:
    """Run one whole in-seam pass on one family, in the array path's own order.

    SUPPLY ``grid`` AND THE LAUNCH LIST IS DERIVED HERE, by ``in_seam_coverage.plan``
    -- the same function the predicate's derivations come from, so the launcher and
    the predicate cannot disagree about which axes a pass visits, what parity each
    carries, or which row the far ghost images.

    ``launches`` IS THE GATE'S DOOR and is keyword-only and named for what it is:
    the gate feeds DELIBERATELY WRONG plans (a reversed axis order, a baked
    ``n - 2`` image row, a flipped parity), and a launcher that could not be handed
    its own plan could not arm those mutations. Passing both, or neither, is
    refused: a caller with a grid AND a plan has two answers to one question.

    THE THREE PASSES ARE NEVER MERGED HERE. They share planes -- a near fill on a
    mirrored X and a wall wipe on a metallic Y meet on one edge of ``Dz`` -- and
    the driver's order (fill, wipe, far fill) is what decides that edge.
    """
    if pass_name not in PASSES:
        raise ValueError(f"pass must be one of {PASSES}, got {pass_name!r}")
    if (grid is None) == (launches is None):
        raise ValueError(
            "pass exactly one of grid (the launch list is derived from it) or "
            "launches (the gate supplies its own, including wrong ones); got "
            f"grid={'set' if grid is not None else 'None'} and "
            f"launches={'set' if launches is not None else 'None'}")
    if launches is None:
        launches = plan(pass_name, grid)
    out = []
    for entry in launches:
        if pass_name == "zero_metal":
            out.append(launch_zero_metal(fields, family, entry["axes"]))
        elif pass_name == "fill_symmetry":
            out.append(launch_fill_symmetry(fields, family, entry["axis"],
                                            entry["phase"]))
        else:
            out.append(launch_fill_folded_far(fields, family, entry["axis"],
                                              entry["phase"],
                                              entry["reflect_row"]))
    return tuple(out)


def covers_pass(pass_name: str, fields: "Fields", grid: "Grid",
                family: str) -> Tuple[bool, str]:
    """The predicate, re-exported: this module is the slice's public face."""
    return covers(pass_name, fields, grid, family)
