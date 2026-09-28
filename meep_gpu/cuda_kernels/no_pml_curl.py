"""The NO-ABSORBER curl family for the hand-CUDA track: three kernels and a predicate.

WHAT COLLAPSES WHEN THE ABSORBER GOES AWAY. ``stepping._apply_curl``
(stepping.py:489-539) has four tails. With ``_pml_is_active(pml)`` False and
``fields.condfac_for(target)`` None, three of them fall away and the whole
sub-step tail is one line (stepping.py:539)::

    target -= curl

There is no ``fu`` auxiliary (``_require_pml_storage`` is not called and every
``fu_*`` is None), no ``kms``/``sinv`` pair, and no split-field recurrence. The
CURL ITSELF IS UNCHANGED -- the term table, the ghost rule, the grouping and the
wall masks -- which is why each block below is the certified
``step_{B,D}_pml_real`` block VERBATIM down to the last line, with only the
tail replaced. That is deliberate and is the whole safety argument for this file:
a reader can diff it against ``step_curl_kernels.py`` and see that nothing in the
arithmetic moved.

WHAT THE B SUB-STEP ACTUALLY DIFFERENCES, WHICH IS NOT ALWAYS ``fields.Ex``
--------------------------------------------------------------------------
This is the finding that reshaped the family, and it was MEASURED before it was
written down. ``stepping.step_B`` reads its operands through
``_component_snapshot`` -> ``_read_component`` (stepping.py:2425-2454), whose E
branch has TWO arms::

    if fields.stores_E or fields.scratch is None:
        return fields.get_E(name)                       # stepping.py:2440-2441
    ...
    scratch.xp.multiply(displacement, inverse, out=target)   # :2450

and ``Fields.get_E`` (fields.py:1009-1036) is itself two arms: the stored array
when ``_stored_E``, and a freshly computed ``D * inv_eps`` when not. WITHOUT AN
ABSORBER AND WITHOUT A SUSCEPTIBILITY NOTHING STORES E -- ``fields.Ex`` is None --
so the B curl differences a DERIVED volume, cell by cell ``D_c * inv_eps_c``.

Measured on a real (6, 8, 10) ``Grid``/``Fields`` pair with an inhomogeneous
inverse epsilon drawn in [0.2, 0.9), boundaries (periodic, metallic, periodic),
courant 0.5: ``stepping.step_B(fields, None)`` is bit-identical, as uint32 words,
to the forward-difference curl taken OF ``D * inv_eps`` -- 480/480 words moved on
Bx and Bz, 420/480 on By (the metallic wall row is masked). A kernel binding
``Ex``/``Ey``/``Ez`` cannot serve that configuration at all, and on the 186-row
corpus it is the MAJORITY of the no-absorber rows: five of the eight real-storage
ones have ``stores_E`` False.

So this file carries THREE kernels, not two:

* ``step_B_no_pml_real``          -- E is STORED; binds Ex/Ey/Ez.
* ``step_B_no_pml_real_derived``  -- E is DERIVED; binds Dx/Dy/Dz and the
  three inverse-epsilon volumes, and forms ``D * inv_eps`` at the point of use.
* ``step_D_no_pml_real``          -- binds the B arrays (see below).

The derived kernel is the ONLY block here that is not a verbatim copy: its two
device helpers, ``derived_at`` and ``derived_up``, are ``shift_up`` composed with
the multiply, in that order. THE ORDER IS THE ARITHMETIC. The array path forms
the whole derived volume and then shifts it with the far-face rule, so a
non-periodic far ghost is an exact ``0.0f`` -- NOT ``D_ghost * inv_eps_ghost`` --
and a periodic wrap carries the WRAPPED cell's inverse epsilon, not the reading
cell's. Both are armed as source mutations in the gate
(``derived_far_ghost_wraps``, ``derived_wrap_uses_own_inv_eps``) because reading
them off is not measuring them.

CONTRACTION IS A HAZARD ON THE DERIVED ARM ON PAPER, AND WAS MEASURED ABSENT ON
ONE DEVICE. ``sf - f1`` is a difference of two PRODUCTS there, which is a textbook
fused-multiply-add candidate, and the array path rounds the multiply and the
subtraction separately -- so ``--fmad=false`` is carried as CORRECTNESS rather
than as tuning. What the gate then measured is that it changed nothing: on an RTX
A6000 under NVRTC 11.6, the UNGUARDED build was bit-identical to the guarded one
on all 112 comparable cases at the inexact courant, derived arm included
(:data:`NO_PML_CURL_ADMISSION`, ``contraction_guard``). The flag is therefore
kept and its effect is NOT claimed: this run carries no evidence that it changed
an answer here, and none that a different compiler would leave it alone.

THE D SUB-STEP DIFFERENCES THE B ARRAYS, AND THAT IS NOT AN OPTIMISATION.
``Fields.get_H`` (fields.py:1164-1187) returns the B ARRAY ITSELF without an
absorber: mu = 1, H is never stored, ``fields.Hx`` is None. Measured on the same
fixture, ``stepping.step_D(fields, None)`` is bit-identical to the backward
curl of Bx/By/Bz. A predicate that delegates its storage inventory to the
certified curl predicate refuses every no-absorber run for "Hx is not allocated";
the inventory below is this file's own for exactly that reason.

WHAT THIS IS FOR, AND WHAT THE CORPUS SAYS IT IS WORTH -- WHICH IS LESS THAN THE
ROW COUNT SUGGESTS. The union census carries 14 rows with no active absorber. Six
are complex storage and belong to a different kernel. Of the eight real-storage
rows, three carry a conductivity (MEEP's ``Absorber`` boundary layer IS a
conductivity, not a PML), which routes ``_apply_curl`` to
``_apply_conductive_update`` (stepping.py:536-537) -- a different recurrence this
family refuses by name. The reachable curl slots are therefore FIVE at ``step_D``
and FIVE at ``step_B`` (one stored-E, four derived), not the 28 a "14 rows x 2
sub-steps" reading suggests, and the census MEASURED exactly those ten:
595 -> 605 of 759 slots, 113 -> 117 rows covered at every sub-step, with the
before figure taken by running the PREVIOUS round's analyzer over the SAME data so
the delta is one predicate's and not a difference of two rounds
(:data:`NO_PML_CURL_ADMISSION`, ``slots_before``/``slots_after``).

The constitutive halves of those rows are a separate question and are NOT served
here: under an inactive layer ``stepping.update_H`` returns at :944-945 and
``update_E`` at :984 before reading an array, so those are a null plan rather
than a kernel -- which is what ``no_pml_constitutive.py`` already carries.

THE CONDUCTIVITY IS REFUSED, not silently absorbed. ``_apply_curl``'s second tail
(:507-508) routes a component carrying ``condfac`` to
``_apply_conductive_update`` even with no absorber, and that is a different
recurrence. It is read PER COMPONENT, so a run where one component is lossy and
two are not takes two different tails within one sub-step -- which one kernel
cannot serve, and which the delegated predicate refuses per sub-step by name.

NO DISPATCH. Nothing in ``meep_gpu`` imports ``cuda_kernels``; these kernels are
reachable only from the gate and the laptop tests. ``covers_no_pml_curl``
returning True licenses a MEASUREMENT, not a production step. What has and has not
been measured on a device is :data:`NO_PML_CURL_ADMISSION` -- as of 2026-08-20,
224/224 licensed cases bit-identical to ``stepping`` at one launch and 224/224 at
60, per policy, under both float32 subnormal policies on an RTX A6000, with 25
source mutations CAUGHT, 24 NULL CONFIRMED, none escaped, and the release verdict
shown to refuse a planted defect. Nothing here may be read as a device verdict
while that record's ``host`` field is None.

IMPORTABLE WITHOUT CUPY, DELIBERATELY. The predicate is consumed by the coverage
census, which runs on a laptop. ``cupy`` is imported INSIDE the launchers, never
at module scope, and the shared prelude is READ out of the sibling's source
rather than imported from it. ``cylindrical_kernels.py`` took the other route --
``import cupy as cp`` at scope, predicate exiled to ``cylindrical_coverage.py`` --
and either is fine; what is not fine is a predicate no census machine can call.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, Dict, Optional, Tuple

from . import coverage as _coverage
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)


def _sibling_prelude() -> str:
    """``step_curl_kernels._REAL_PML_PRELUDE``, READ rather than imported or copied.

    The certified curl module imports ``cupy`` at module scope, so importing it
    would make this file -- predicate and all -- unimportable on any machine
    without a device, and the coverage census runs on exactly such a machine.
    Copying the prelude instead would give the two files two sets of bytes that
    are equal only until someone edits one: the ghost helpers and the boundary
    codes below ARE the certified ones, and a silent divergence in ``shift_up``
    would be a wrong answer that no diff of this file would show.

    Parsing the sibling's source for its own string literal has neither problem.
    It is the sibling's bytes by construction, and a rename or a restructure
    there fails loudly here instead of quietly forking the arithmetic.
    """
    source = (pathlib.Path(__file__).with_name("step_curl_kernels.py")
              .read_text(encoding="utf-8"))
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Name) and target.id == "_REAL_PML_PRELUDE"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            return node.value.value
    raise RuntimeError(
        "step_curl_kernels.py no longer assigns a string literal to "
        "_REAL_PML_PRELUDE; the no-PML curl pair shares that prelude verbatim "
        "and cannot be built without it")


_REAL_PML_PRELUDE = _sibling_prelude()

#: The prelude's ``pml_apply`` helper is DEAD CODE in every kernel below -- none
#: of them calls it, because the whole point of the family is that the tail is
#: not the split-field recurrence. It is carried anyway, because the prelude is
#: shared verbatim and forking it to drop three lines would be the exact silent
#: divergence :func:`_sibling_prelude` exists to prevent. The gate turns that
#: liability into evidence: every mutation of ``pml_apply`` (``drop_fu_store``,
#: ``read_fprev_after_store``, ``swap_dsig_dsigu``, ``reload_fu_from_memory``) is
#: scored as a MUST-BE-UNCAUGHT null, and their being uncaught is the measurement
#: that this family really does not route through that recurrence.
_PRELUDE_HELPER_IS_DEAD_HERE = "pml_apply"

__all__ = (
    "CERTIFIED_KERNELS",
    "NO_PML_KERNELS",
    "NO_PML_CURL_ADMISSION",
    "UNCERTIFIED_KERNELS",
    "covers_no_pml_curl",
    "kernel_source",
    "no_pml_curl_arm",
    "step_no_pml_curl",
)

#: The three kernels this module carries, in ``STEP_ORDER`` order then arm order.
NO_PML_KERNELS: Tuple[str, ...] = ("step_B_no_pml_real",
                                   "step_B_no_pml_real_derived",
                                   "step_D_no_pml_real")

#: Byte-identical to the array path with a device verdict behind it. The evidence is
#: :data:`NO_PML_CURL_ADMISSION` and ``certification.json``'s
#: ``cuda_no_pml_curl_2026-08-21`` block, which names the same artifact directory; a
#: name here without a record block is the failure ``test_kernel_partition.py``
#: exists to catch.
CERTIFIED_KERNELS = ("step_B_no_pml_real", "step_B_no_pml_real_derived",
                     "step_D_no_pml_real")

#: Shipped but not gated. EMPTY, and the partition requires every shipped kernel to
#: be in exactly one of the two sets, so a kernel added to this file without a gate
#: verdict fails there rather than shipping unmeasured. Spelled as a dict WITHOUT a
#: type annotation for the reason ``constitutive_kernels.py`` records: the partition
#: readers walk the syntax tree so they run where there is no CuPy, and an annotated
#: assignment is an ``ast.AnnAssign`` the plain-assignment readers do not match --
#: annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}

#: Which kernel serves which ``(sub_step, arm)``. ``step_D`` has ONE arm because
#: it never reads E: its sources are the B arrays whatever E storage the run has.
ARM_KERNELS: Dict[Tuple[str, str], str] = {
    ("step_B", "stored_E"): "step_B_no_pml_real",
    ("step_B", "derived_E"): "step_B_no_pml_real_derived",
    ("step_D", "magnetic"): "step_D_no_pml_real",
}


# B side, STORED E: forward differences, iyee from fields.IYEE_SHIFTS -- Bx (0,1,1),
# By (1,0,1), Bz (1,1,0). Every mask line is the certified kernel's.
_step_B_no_pml_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void step_B_no_pml_real(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    int nx, int ny, int nz, float dtdx,
    int bc_x, int bc_y, int bc_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Bx: curl_x = dEz/dy - dEy/dz; iyee=(0,1,1) -> cell 0 on x, and on a folded
    // PERIODIC axis the top plane on y and z.
    {
        float f1 = Ez[idx];
        float sf = shift_up(Ez, idx, j, ny, sy, bc_y);
        float f2 = Ey[idx];
        float ss = shift_up(Ey, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        Bx[idx] = Bx[idx] - curl;
    }

    // By: curl_y = dEx/dz - dEz/dx; iyee=(1,0,1) -> cell 0 on y, top plane on z and x.
    {
        float f1 = Ex[idx];
        float sf = shift_up(Ex, idx, k, nz, sz, bc_z);
        float f2 = Ez[idx];
        float ss = shift_up(Ez, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        By[idx] = By[idx] - curl;
    }

    // Bz: curl_z = dEy/dx - dEx/dy; iyee=(1,1,0) -> cell 0 on z, top plane on x and y.
    {
        float f1 = Ey[idx];
        float sf = shift_up(Ey, idx, i, nx, sx, bc_x);
        float f2 = Ex[idx];
        float ss = shift_up(Ex, idx, j, ny, sy, bc_y);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        Bz[idx] = Bz[idx] - curl;
    }
}
'''

# B side, DERIVED E. Identical to the block above with every ``Ec[idx]`` replaced
# by ``derived_at(Dc, iec, idx)`` and every ``shift_up(Ec, ...)`` by
# ``derived_up(Dc, iec, ...)``. The two helpers are ``_read_component``'s multiply
# COMPOSED WITH ``_shift_up``, in that order -- see the module header for why the
# order is the arithmetic and not a style choice.
# The citation inside this prelude is gate-pinned device text (the admission
# record digests it); the line it names is stepping.py:2450 on the current tree.
_DERIVED_PRELUDE = r'''
// stepping._read_component's derived E at one cell: get_E's un-stored branch is
// D * inv_eps (fields.py:1035), and the scratch branch (stepping.py:2395) is the
// same product into a pooled buffer. D ON THE LEFT, as both write it.
__device__ __forceinline__ float derived_at(
    const float* __restrict__ d, const float* __restrict__ ie, int idx
) {
    return d[idx] * ie[idx];
}

// shift_up APPLIED TO THE DERIVED VOLUME. The array path derives the whole
// volume first and shifts it after, so the far-face ghost of a non-periodic axis
// is an exact zero rather than the ghost cell's product, and a periodic wrap
// carries the WRAPPED cell's inverse epsilon. Both are armed as mutations.
__device__ __forceinline__ float derived_up(
    const float* __restrict__ d, const float* __restrict__ ie,
    int idx, int ia, int na, int stride, int bc
) {
    if (ia + 1 < na) return d[idx + stride] * ie[idx + stride];
    return (bc == BC_PERIODIC) ? d[idx - ia * stride] * ie[idx - ia * stride] : 0.0f;
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 2395->2450

_step_B_no_pml_real_derived_kernel_code = (
    _REAL_PML_PRELUDE + _DERIVED_PRELUDE + r'''
extern "C" __global__ void step_B_no_pml_real_derived(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    const float* __restrict__ Dx, const float* __restrict__ Dy,
    const float* __restrict__ Dz,
    const float* __restrict__ iex, const float* __restrict__ iey,
    const float* __restrict__ iez,
    int nx, int ny, int nz, float dtdx,
    int bc_x, int bc_y, int bc_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Bx: curl_x = dEz/dy - dEy/dz; iyee=(0,1,1) -> cell 0 on x, and on a folded
    // PERIODIC axis the top plane on y and z.
    {
        float f1 = derived_at(Dz, iez, idx);
        float sf = derived_up(Dz, iez, idx, j, ny, sy, bc_y);
        float f2 = derived_at(Dy, iey, idx);
        float ss = derived_up(Dy, iey, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        Bx[idx] = Bx[idx] - curl;
    }

    // By: curl_y = dEx/dz - dEz/dx; iyee=(1,0,1) -> cell 0 on y, top plane on z and x.
    {
        float f1 = derived_at(Dx, iex, idx);
        float sf = derived_up(Dx, iex, idx, k, nz, sz, bc_z);
        float f2 = derived_at(Dz, iez, idx);
        float ss = derived_up(Dz, iez, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        By[idx] = By[idx] - curl;
    }

    // Bz: curl_z = dEy/dx - dEx/dy; iyee=(1,1,0) -> cell 0 on z, top plane on x and y.
    {
        float f1 = derived_at(Dy, iey, idx);
        float sf = derived_up(Dy, iey, idx, i, nx, sx, bc_x);
        float f2 = derived_at(Dx, iex, idx);
        float ss = derived_up(Dx, iex, idx, j, ny, sy, bc_y);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        Bz[idx] = Bz[idx] - curl;
    }
}
''')

# D side: MEEP's negated strides (backward differences), iyee -- Dx (1,0,0),
# Dy (0,1,0), Dz (0,0,1). The three source pointers are the B ARRAYS: without an
# absorber ``Fields.get_H`` returns them, and the array path differences them.
_step_D_no_pml_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void step_D_no_pml_real(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    int nx, int ny, int nz, float dtdx,
    int bc_x, int bc_y, int bc_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Dx: curl_x = dHz/dy - dHy/dz; iyee=(1,0,0) -> cell 0 on y and z, and on a
    // folded PERIODIC axis the top plane on x -- an axis this stencil never
    // differences along. The mask is about which cells MEEP's owned loop VISITS,
    // not about where the stencil reaches.
    {
        float f1 = Hz[idx];
        float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);
        float f2 = Hy[idx];
        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        Dx[idx] = Dx[idx] - curl;
    }

    // Dy: curl_y = dHx/dz - dHz/dx; iyee=(0,1,0) -> cell 0 on x and z, top plane on y.
    {
        float f1 = Hx[idx];
        float sf = shift_dn(Hx, idx, k, nz, sz, bc_z);
        float f2 = Hz[idx];
        float ss = shift_dn(Hz, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        Dy[idx] = Dy[idx] - curl;
    }

    // Dz: curl_z = dHy/dx - dHx/dy; iyee=(0,0,1) -> cell 0 on x and y, top plane on z.
    {
        float f1 = Hy[idx];
        float sf = shift_dn(Hy, idx, i, nx, sx, bc_x);
        float f2 = Hx[idx];
        float ss = shift_dn(Hx, idx, j, ny, sy, bc_y);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        Dz[idx] = Dz[idx] - curl;
    }
}
'''

#: The device text, by kernel name. A MUTABLE MODULE GLOBAL on purpose: the gate's
#: source-mutation battery rewrites entries here and re-launches, and
#: :func:`_get_kernel` re-reads this map on every call so a rewritten string is
#: seen. A map memoized at first call would hand back the pre-mutation bytes
#: forever -- "a leg reporting a pass for a mutation it never applied", which the
#: sibling track hit three times and which the source-in-the-memo-key design
#: exists to remove.
_SOURCES: Dict[str, str] = {
    "step_B_no_pml_real": _step_B_no_pml_real_kernel_code,
    "step_B_no_pml_real_derived":
        _step_B_no_pml_real_derived_kernel_code,
    "step_D_no_pml_real": _step_D_no_pml_real_kernel_code,
}

#: ``--fmad=false`` is CORRECTNESS here, not tuning. On the stored arm the
#: candidate is ``dtdx * ((sf - f1) + (f2 - ss))``, which the array path rounds in
#: three separate operations; on the DERIVED arm ``sf`` and ``f1`` are themselves
#: products, so ``sf - f1`` is a textbook FMA candidate the array path rounds
#: twice. The gate runs an unguarded control at the inexact courant and records
#: whether the flag changed an answer, rather than asserting that it must.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: The block size, matching ``step_curl_kernels._REAL_PML_THREADS``. NOT read out
#: of the sibling the way the prelude is, and it does not need to be: every kernel
#: here guards on ``idx >= nx*ny*nz`` and touches only its own cell's output, so a
#: different value changes throughput and no bit. The prelude is a different case
#: entirely -- ``shift_up`` IS the arithmetic -- which is why one is shared and the
#: other is a number.
_THREADS = 256


def kernel_source(name: str) -> str:
    """The device text for one kernel, by name.

    Read through a function rather than exported as a module global so a gate
    that mutates the text has one seam to patch and the mutation cannot miss a
    second copy.
    """
    if name not in _SOURCES:
        raise ValueError(f"no such no-PML kernel {name!r}; have {sorted(_SOURCES)}")
    return _SOURCES[name]


def set_kernel_source(name: str, source: str) -> None:
    """Replace one kernel's device text -- the gate's mutation seam.

    Paired with :func:`clear_kernel_cache`: the compile memo keys on the SOURCE,
    so a mutated body is a miss and reaches NVRTC without the clear, but the
    clear is what makes the accounting ("how many constructions came from the
    mutated bytes") exact.
    """
    if name not in _SOURCES:
        raise ValueError(f"no such no-PML kernel {name!r}; have {sorted(_SOURCES)}")
    _SOURCES[name] = source


def clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return _clear_cache()


def _get_kernel(name: str):
    """Compile (or fetch) one kernel. ``cupy`` is imported HERE, never at scope."""
    import cupy as cp  # noqa: PLC0415 - a device-only import in a laptop-importable module

    code = kernel_source(name)
    key = _kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def step_no_pml_curl(fields: Any, sub_step: str, boundary_codes, dtdx: float,
                     arm: Optional[str] = None,
                     inverse_epsilon: Optional[Tuple[Any, Any, Any]] = None) -> str:
    """Launch the no-absorber curl for one sub-step. Returns the kernel launched.

    ``boundary_codes`` is the triple ``coverage.real_curl_boundary_codes`` builds
    -- the SAME function the certified pair's launcher calls, so the launcher and
    the predicate cannot disagree about what a folded axis resolves to.

    ``arm`` is normally left None and taken from the run
    (:func:`no_pml_curl_arm`); a gate passes it explicitly to launch a specific
    kernel against a fixture it built for that arm.

    ``inverse_epsilon`` is likewise a GATE SEAM and nothing else: the three
    volumes normally come from ``fields.inverse_epsilon_for``, and a host-mutation
    leg substitutes corrupted ones here so the defect it plants is on the POINTERS
    a launch binds rather than on the fields object every other leg shares.
    """
    import numpy as np  # noqa: PLC0415 - only the scalar types are needed

    arm = arm or no_pml_curl_arm(fields, sub_step)
    name = ARM_KERNELS[(sub_step, arm)]
    # THE REFUSAL COMES BEFORE THE COMPILE, so a configuration this family does not
    # serve costs an exception rather than an NVRTC call -- and so the check is
    # reachable on a host with no CuPy, which is where its test runs.
    if sub_step == "step_D" and getattr(fields, "_pml_active", False):
        raise ValueError(
            "Fields is in PML storage mode, so get_H would return the stored H "
            "rather than the B array this family differences; ask "
            "covers_no_pml_curl first -- it refuses this configuration rather "
            "than raising")
    kernel = _get_kernel(name)

    if sub_step == "step_B":
        targets = (fields.Bx, fields.By, fields.Bz)
        if arm == "stored_E":
            sources = (fields.Ex, fields.Ey, fields.Ez)
            extra: Tuple[Any, ...] = ()
        else:
            sources = (fields.Dx, fields.Dy, fields.Dz)
            extra = tuple(inverse_epsilon) if inverse_epsilon is not None else tuple(
                fields.inverse_epsilon_for(component)
                for component in ("Ex", "Ey", "Ez"))
    else:
        # ``get_H``'s no-absorber alias, spelled out. Bound DIRECTLY rather than
        # through ``get_H`` so the launcher binds exactly the arrays
        # :func:`_array_inventory` checked -- ``get_H`` branches on
        # ``Fields._pml_active`` (fields.py:1182-1187) and would hand back a
        # STORED H on a Fields left in PML storage mode. That configuration is
        # already refused by the predicate (its ``fu_D*`` clause fires, because
        # ``enable_pml_storage`` allocates the auxiliaries with the H arrays), and
        # the guard above raises on it -- the same split
        # ``real_curl_boundary_codes`` takes, where the launch-side face raises on
        # exactly what the predicate refuses.
        targets = (fields.Dx, fields.Dy, fields.Dz)
        sources = (fields.Bx, fields.By, fields.Bz)
        extra = ()

    nx, ny, nz = targets[0].shape
    blocks = (nx * ny * nz + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,), (
        targets[0], targets[1], targets[2],
        sources[0], sources[1], sources[2],
        *extra,
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ))
    return name


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def no_pml_curl_arm(fields: Any, sub_step: str) -> str:
    """Which arm a run takes at this sub-step. Never a refusal -- a classification.

    ``step_D`` has one arm because it never reads E. ``step_B`` follows
    ``fields.stores_E``, which is the exact flag ``stepping._read_component``
    branches on (stepping.py:2440), so the arm cannot disagree with what the
    array path will do.
    """
    if sub_step == "step_D":
        return "magnetic"
    if sub_step != "step_B":
        raise ValueError(
            f"sub_step must be 'step_B' or 'step_D', got {sub_step!r}")
    return "stored_E" if getattr(fields, "stores_E", False) else "derived_E"


def _array_inventory(fields: Any, grid: Any, sub_step: str, arm: str):
    """The arrays THIS family's kernel binds, checked as the kernel will use them.

    THIS IS NOT A DUPLICATE OF THE CERTIFIED PREDICATE'S ARRAY LOOP. That loop
    asks for the eighteen arrays a PML run has; a no-absorber run has neither the
    stored H nor any ``fu``, and on the derived arm it has no stored E either. The
    list below is exactly what :func:`step_no_pml_curl` passes as a pointer, and
    nothing else -- which is the only list whose contiguity, dtype and shape a
    launch depends on.
    """
    xp = getattr(grid, "xp", None)
    try:
        shape = tuple(grid.shape)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return None, f"grid shape could not be read: {type(exc).__name__}: {exc}"

    if sub_step == "step_B":
        targets = ("Bx", "By", "Bz")
        sources = ("Ex", "Ey", "Ez") if arm == "stored_E" else ("Dx", "Dy", "Dz")
    else:
        targets = ("Dx", "Dy", "Dz")
        # Spelled as B, because that is the array the kernel receives. Asking for
        # "Hx" here is the refusal this family exists to stop inheriting.
        sources = ("Bx", "By", "Bz")

    for name in targets + sources:
        problem = _coverage._array_problem(name, getattr(fields, name, None),
                                           xp, shape)
        if problem is not None:
            return None, problem

    if sub_step == "step_B" and arm == "derived_E":
        # THE INVERSE-EPSILON VOLUMES ARE POINTERS TOO, and their DTYPE decides
        # the arithmetic rather than just the launch: the array path's derived
        # buffer is ``result_type(D, inv_eps)`` (stepping.py:2449), so a float64
        # inverse epsilon makes the whole curl a float64 curl and this float32
        # kernel would be a different computation, not a different rounding.
        for component in ("Ex", "Ey", "Ez"):
            try:
                inverse = fields.inverse_epsilon_for(component)
            except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
                return None, (f"fields could not be asked for {component}'s "
                              f"inverse epsilon: {type(exc).__name__}: {exc}")
            problem = _coverage._array_problem(f"inverse epsilon for {component}",
                                               inverse, xp, shape)
            if problem is not None:
                return None, problem

    # THE AUXILIARY MUST BE ABSENT, not merely unused. This family writes no
    # ``fu``; an ALLOCATED one means something else built it and this step would
    # leave it stale, which is a wrong answer on the NEXT sub-step rather than
    # this one.
    for name in (f"fu_{stem}" for stem in targets):
        if getattr(fields, name, None) is not None:
            return None, (f"{name} is allocated while the layer is inactive: "
                          f"this family writes no auxiliary, so something else "
                          f"built it and this step would leave it stale")
    return arm, None


class _ActiveLayerProxy:
    """The run's layer, answering ``is_active`` True so the sibling's clauses run.

    Every clause of ``covers_real_pml_curl`` except the absorber one is about the
    grid, the storage and the fields, and those are unchanged by the absorber's
    absence. Satisfying that one clause AT THE INPUT is the same technique the
    coverage census uses for the CuPy-backend clause, and for the same reason:
    these predicates short-circuit, so there is no accumulated refusal list to
    filter afterwards.

    A layer that is genuinely absent (``pml is None``) still cannot answer the
    coefficient questions; ``getattr(proxy, "kms_x", None)`` then yields None and
    the sibling refuses with a ``pml.``-prefixed reason, which is the ONE refusal
    this file forgives -- see :func:`covers_no_pml_curl`.
    """

    __slots__ = ("_layer",)

    def __init__(self, layer: Any) -> None:
        self._layer = layer

    @property
    def is_active(self) -> bool:
        return True

    def __getattr__(self, item: str) -> Any:
        if self._layer is None:
            raise AttributeError(
                f"no absorber object exists on this run, so {item!r} cannot be "
                f"answered; the no-PML curl predicate asks the certified "
                f"predicate only about the grid, the storage and the fields")
        return getattr(self._layer, item)


class _NoAbsorberFieldsProxy:
    """The run's fields, answering the sibling's PML-shaped storage questions.

    WHY A PROXY AND NOT A STRING FILTER. The obvious way to reuse
    ``covers_real_pml_curl`` is to call it and forgive the refusals a no-absorber
    run legitimately earns. That is UNSOUND, and measurably so: the predicate
    SHORT-CIRCUITS, and ``stores_E`` is clause 12 of 16. Forgiving its refusal
    silently skips the chi2/chi3 clause, the three-dimensionality clause and the
    int32 index-range clause that follow it -- three real refusals turned into
    admissions by a filter that never claimed to touch them. Answering the
    question AT THE INPUT instead lets every later clause run.

    WHAT IS TRUE HERE AND WHAT IS A STAND-IN, named one by one:

    * ``Hx``/``Hy``/``Hz`` -> the B arrays. NOT a stand-in: ``Fields.get_H``
      returns exactly these without an absorber (fields.py:1184-1186), and
      ``stepping.step_D`` differences exactly these -- measured bit-identical.
    * ``stores_E`` -> True. On the stored arm this is the truth. On the derived
      arm it is an ASSERTION THAT THIS FILE HAS ANSWERED THE QUESTION ITSELF:
      ``_array_inventory`` has already checked D and the three inverse-epsilon
      volumes, which is what the derived curl actually reads.
    * ``Ex``/``Ey``/``Ez`` on the derived arm -> the D arrays, a STAND-IN. The
      sibling's clause asks dtype, shape and contiguity of the array the curl
      differences; the array the curl differences is the derived buffer, whose
      shape is D's and whose dtype is ``result_type(D, inv_eps)``
      (stepping.py:2449). D therefore answers the shape and contiguity question
      exactly, and the DTYPE question is answered by this file's own check on the
      inverse-epsilon volumes -- because that is the operand D's dtype alone
      cannot speak for.
    * ``fu_*`` -> the target arrays, a STAND-IN, for the same reason and with the
      same accounting: the real question -- "is one allocated?" -- is
      :func:`_array_inventory`'s, and it is asked in the opposite direction
      (allocated is a REFUSAL here).

    Everything else forwards, so a clause added to the sibling is inherited
    rather than skipped.
    """

    __slots__ = ("_fields",)

    _H_ALIAS = {"Hx": "Bx", "Hy": "By", "Hz": "Bz"}
    _DERIVED_ALIAS = {"Ex": "Dx", "Ey": "Dy", "Ez": "Dz"}
    _FU_ALIAS = {"fu_Bx": "Bx", "fu_By": "By", "fu_Bz": "Bz",
                 "fu_Dx": "Dx", "fu_Dy": "Dy", "fu_Dz": "Dz"}

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    @property
    def stores_E(self) -> bool:
        return True

    def __getattr__(self, item: str) -> Any:
        fields = object.__getattribute__(self, "_fields")
        if item in self._H_ALIAS:
            return getattr(fields, self._H_ALIAS[item], None)
        if item in self._FU_ALIAS:
            return getattr(fields, self._FU_ALIAS[item], None)
        if item in self._DERIVED_ALIAS and not getattr(fields, "stores_E", False):
            return getattr(fields, self._DERIVED_ALIAS[item], None)
        return getattr(fields, item)


#: The ONE refusal from the certified predicate this file forgives, matched on the
#: prefix ``_coefficient_vector_problem`` gives every message it produces
#: (coverage.py:520-534). It is forgiven because a run with no absorber object has
#: no coefficient vectors and these kernels read none -- and it is SAFE to forgive
#: because it is the LAST clause the sibling evaluates, so nothing is skipped
#: behind it. ``test_forgiven_refusal_is_the_last_clause`` measures that ordering
#: rather than trusting it; if the sibling ever moves the coefficient loop up, that
#: test fails and this file has to be rewritten, not silently widened.
_FORGIVEN_REFUSAL_PREFIX = "pml."

#: WHAT THIS FAMILY'S OWN GATE SWEPT, where the sibling's is wider, and why the
#: difference is refused here rather than inherited.
#:
#: :func:`covers_no_pml_curl` DELEGATES to ``coverage.covers_real_pml_curl`` and
#: inherits every clause it states. That is the design, and it is sound while the
#: two families' evidence covers the same shapes -- delegation can only make this
#: predicate NARROWER than the one it borrows its arithmetic from, never wider.
#:
#: ON 2026-08-20 THE SIBLING BECAME WIDER THAN ITS OWN GATE. Two of its refusals
#: were retired on device rounds that ran the PML kernels
#: (``coverage.CURL_FOLD_ADMISSION``'s third fold plane and
#: ``coverage.CURL_NONLINEAR_ADMISSION``'s chi2/chi3), and delegation would have
#: carried both here on the strength of a measurement of DIFFERENT KERNELS.
#: ``gate_cuda_no_pml_curl.py``'s own case table stops at ``"XY"`` -- two
#: simultaneous planes -- and installs no chi anywhere, so this family has run
#: neither shape.
#:
#: BOTH REFUSALS COST ZERO SLOTS, which is why this is a bookkeeping correction
#: and not a retreat: the corpus's triply-folded row (``TestLDOS.test_ldos_3D``)
#: and both of its instantaneous-nonlinearity rows (``3rd-harm-1d.py``,
#: ``Test3rdHarm1d.test_3rd_harm_1d``) are PML-ACTIVE, so this family is never
#: asked about them. Raising either number means adding the spec to this family's
#: gate and running it.
NO_PML_CURL_NARROWER_THAN_THE_SIBLING: dict = {
    "gate": "parity/meep_gpu/gate_cuda_no_pml_curl.py",
    "folded_planes_swept": 2,
    "chi2_chi3_swept": False,
    "sibling_admission_not_inherited": (
        "coverage.CURL_FOLD_ADMISSION (three simultaneous fold planes, "
        "2026-08-20)",
        "coverage.CURL_NONLINEAR_ADMISSION (instantaneous chi2/chi3, 2026-08-20)",
    ),
    "slots_this_costs": 0,
    "why_zero": ("every corpus row carrying either shape is PML-active, so the "
                 "no-absorber family is not asked about one of them"),
}


def covers_no_pml_curl(fields: Any, pml: Any, grid: Any, sub_step: str) -> tuple:
    """Whether the no-absorber curl kernels may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. Returns ``(covered, reason)``.

    THE SUB-STEP ARGUMENT IS REQUIRED for the reason
    :func:`coverage.covers_real_pml_curl`'s is: the conductivity is read PER
    TARGET COMPONENT in ``stepping._apply_curl`` (:479), so a D-side conductivity
    routes ``step_D`` to a recurrence this family does not implement and leaves
    ``step_B`` an ordinary curl. Refusing both would give up a slot that is
    served; admitting both would serve one that is not. It ALSO chooses the arm:
    ``step_B`` reads E and ``step_D`` does not, which is why ``stores_E`` is a
    question for one of them and irrelevant to the other -- an over-refusal this
    predicate carried until it was measured on the corpus.

    DELEGATION, NOT A SECOND CLAUSE SET. Every question except the absorber and
    the storage inventory is the certified curl predicate's, asked through it, so
    this family cannot drift to a weaker standard than the one it borrows its
    arithmetic from. The two proxies are how the questions that ARE this file's
    get answered at the input rather than filtered out of the output; see
    :class:`_NoAbsorberFieldsProxy` for why the filter would be unsound.
    """
    if sub_step not in ("step_B", "step_D"):
        raise ValueError(
            f"sub_step must be 'step_B' or 'step_D', got {sub_step!r}")

    if pml is not None and bool(getattr(pml, "is_active", False)):
        return False, ("an active absorber is installed: the curl tail is the "
                       "split-field recurrence, which step_*_pml_real "
                       "serves and this family does not")

    # THE TWO CLAUSES THIS FAMILY OWNS BECAUSE THE SIBLING NO LONGER STATES THEM.
    # Delegation inherits every refusal the sibling makes; it cannot inherit one
    # the sibling has retired. Both of the sibling's 2026-08-20 widenings rest on
    # device rounds that ran the PML kernels, and this family's gate has swept
    # neither shape -- see :data:`NO_PML_CURL_NARROWER_THAN_THE_SIBLING`, which
    # also records that both refusals cost zero slots on this corpus.
    #
    # THEY ARE ASKED BEFORE DELEGATION, not filtered out of its answer, for the
    # reason :class:`_NoAbsorberFieldsProxy` gives at length: the sibling
    # short-circuits, so anything decided after the fact skips whatever lay
    # behind it.
    try:
        folded = sum(1 for axis in range(3) if bool(grid.is_mirrored(axis)))
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        return False, (f"grid could not be asked which axes are mirrored: "
                       f"{type(exc).__name__}: {exc}")
    cap = NO_PML_CURL_NARROWER_THAN_THE_SIBLING["folded_planes_swept"]
    if folded > cap:
        return False, (f"{folded} mirror planes at once: this family's gate swept "
                       f"{cap} simultaneous planes, so a {folded}-plane fold would "
                       f"be inherited from the PML curl's measurement rather than "
                       f"measured on these kernels")
    if getattr(fields, "_chi2_components", None) or getattr(
            fields, "_chi3_components", None):
        return False, ("instantaneous chi2/chi3: the PML curl's null was measured "
                       "on its own kernels and this family's gate installs no chi, "
                       "so admitting one here would be a transfer between kernels")

    arm = no_pml_curl_arm(fields, sub_step)
    arm, problem = _array_inventory(fields, grid, sub_step, arm)
    if problem is not None:
        return False, problem

    covered, reason = _coverage.covers_real_pml_curl(
        _NoAbsorberFieldsProxy(fields), _ActiveLayerProxy(pml), grid, sub_step)
    if not covered and not str(reason or "").startswith(_FORGIVEN_REFUSAL_PREFIX):
        return False, reason
    return True, ""


#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, AND WHAT HAS NOT.
#:
#: ``host`` is None until a device leg has run. Nothing in this file may be read
#: as a device verdict while it is None --
#: ``test_no_pml_curl.test_admission_record_is_all_or_nothing`` is what keeps the
#: two states from blurring, and it is the same rule
#: ``no_pml_constitutive.NULL_CONSTITUTIVE_CONFIRMATION`` carries.
#: THE DEVICE TEXT IS PINNED BY DIGEST, NOT THE FILE. A record written after a
#: gate ran cannot also be inside the bytes that gate stamped, and pinning the
#: whole file would make every comment edit read as an un-gated kernel change.
#: What must not have moved is the three device strings, so those are what the
#: record carries and what ``test_no_pml_curl`` re-measures against
#: :func:`kernel_source`.
NO_PML_CURL_ADMISSION: dict = {
    "gate": "parity/meep_gpu/gate_cuda_no_pml_curl.py",
    "artifacts": "parity/meep_gpu/results/cuda_no_pml_curl_2026-08-21_rename",
    "artifact_sha256": {
        "keep": "8d721cb44ccb895d0564eff03fb36d99c0f7c376dd2a23499f03d96f02e5a306",
        "flush": "8c8e6ddc547cff16453b9ea0b24f537d75170a076fdeecdf6ef4ab3ba34a94a1",
        "falsify": "2577a4b6e519823ab42e16ba1c94b3b8245667018f3274c5187be263a7033ea5",
    },
    "recorded_utc": "2026-08-21T23:11:49Z",
    "host": "the GPU host",
    "device": "NVIDIA RTX A6000 (index 6, verified empty), CuPy 13.5.1, NVRTC 11.6",
    # PER POLICY, and identical under both. 464 cases are scored in the sweep; the
    # 224 below are the LICENSED arm (the shipped code resolution), which is the
    # only triple a dispatch could produce. The other 240 are the two
    # misdeclaration controls and are meant to diverge.
    "cases_scored": 464,
    "single_launch_identical": 224,
    "multi_step_identical": 224,
    "multi_step_budget": 60,
    "arms_swept": ("stored_E", "derived_E", "magnetic"),
    "arm_case_counts": {"step_B/stored_E": 56, "step_B/derived_E": 56,
                        "step_D/magnetic": 112},
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "courants": (0.5, 0.35),
    "value_classes": ("uniform", "subnormal_band"),
    "absorber_shapes_swept": ("none", "inert_layer"),
    "shapes_swept": ("3-D", "2-D", "1-D"),
    "walls_swept": ("periodic", "metallic", "mixed", "mirror fold at both terminations"),
    # THE TWO POLICIES ARE DIFFERENT BUILDS, not one build named twice: the NVRTC
    # observer recorded 65 calls and 47 distinct binaries on each leg, with
    # ``any_ftz_true_reached_nvrtc`` FALSE under keep and TRUE under flush, and the
    # same source hashing to different binaries. A policy leg whose binaries
    # matched the other's would be one leg reported as two.
    "nvrtc_calls_per_leg": 65,
    "distinct_binaries_per_leg": 47,
    "ftz_true_reached_nvrtc": {"keep": False, "flush": True},
    "kernel_source_sha256": {
        "step_B_no_pml_real":
            "8b1c1c65329dd2480759f12b9cbe3c0573bb80fde4786ef6da40b0d8230556d2",
        "step_B_no_pml_real_derived":
            "cb923ef8e80f0c4f6577b88f12786cb83143500549d2cd086b887859a8269690",
        "step_D_no_pml_real":
            "9528f69b0164eec37256ce494871c78b569af4d1b7fa76bcfdae4f26954179bc",
    },
    "source_mutations_caught": 25,
    "source_mutations_null_confirmed": 24,
    "source_mutations_escaped": 0,
    "host_mutations_caught": ("reverse_inverse_epsilon_volumes",
                              "alias_inverse_epsilon_to_Ez"),
    "host_mutations_null_confirmed": ("identity_rebind",),
    # THE FOUR DEAD-PRELUDE NULLS, and what their silence measures. The shared
    # ``_REAL_PML_PRELUDE`` compiles ``pml_apply`` into all three kernels and none
    # of them calls it; ``drop_fu_store``, ``read_fprev_after_store``,
    # ``swap_dsig_dsigu`` and ``reload_fu_from_memory`` therefore arm, at one site
    # each, and came back UNCAUGHT on every leg under both policies. That is the
    # measurement that this family's tail really is ``target -= curl`` and not the
    # split-field recurrence.
    "dead_prelude_nulls_confirmed": ("drop_fu_store", "read_fprev_after_store",
                                     "swap_dsig_dsigu", "reload_fu_from_memory"),
    # THREE MUTATIONS THAT ARE INERT FOR STRUCTURAL REASONS, each measured red
    # first and then given a NAME rather than a filter. Every one is a fact about
    # a degenerate axis rather than about the kernel:
    #  * a folded PERIODIC axis's top-plane mask drops the plane that consumes the
    #    changed far ghost, so ``derived_far_ghost_wraps`` cannot bite there;
    #  * an INVARIANT axis's periodic wrap returns the reading cell, so the wrap's
    #    inverse-epsilon index is unobservable;
    #  * the C-order and Fortran-order index decompositions agree cell for cell on
    #    a (1, 1, 24) grid.
    "structural_nulls_confirmed": ("derived_far_ghost_wraps_on_a_folded_axis",
                                   "derived_wrap_uses_own_inv_eps_on_an_invariant_axis",
                                   "fortran_order_index_decomposition_on_a_flat_grid"),
    "misdeclaration_controls": {
        # A folded PERIODIC axis handed BC_METALLIC. The two codes share a ghost
        # rule, so this changes ONLY the top-plane mask -- and every one of the
        # 6842 differing words lay on the mask-delta planes, none elsewhere.
        "mirror_as_metallic": {"cases": 64, "diverged": 64,
                               "confined_to_mask_delta": True},
        # The blunt one: no mask and a wrapping ghost everywhere. 175 of 176
        # diverged; the one that did not is a 1-D grid whose resolved triple the
        # substitution barely moves.
        "all_periodic": {"cases": 176, "diverged": 175,
                         "confined_to_mask_delta": False},
    },
    # THE VERDICT WAS SHOWN TO GO RED, three independent ways on the record itself
    # (a licensed case flipped to diverging, a must-be-caught mutation flipped to
    # ESCAPED, the misdeclaration control silenced) and once for real: the
    # falsification leg ran a SEPARATE tree with the derive multiply deleted from
    # ``derived_at``/``derived_up``, mutation battery OFF, and the gate REFUSED it
    # (rc 1) with the derived arm 0/5 at one launch and 0/5 at 60 -- while the
    # stored and magnetic arms stayed 5/5 and 10/10 identical, which is the plant
    # being localized rather than smeared.
    "verdict_flips_against_planted_defect": True,
    "falsification_leg": {"plant": "the derive multiply deleted from derived_at "
                                   "and derived_up", "released": False,
                          "derived_arm_identical": "0/5 single, 0/5 multi",
                          "other_arms_identical": "5/5 and 10/10, unaffected"},
    # THE CONTRACTION GUARD IS CARRIED AND ITS EFFECT IS NOT CLAIMED. The reading
    # said ``sf - f1`` on the derived arm is an FMA candidate; measured, the
    # UNGUARDED build was bit-identical on all 112 comparable cases at the inexact
    # courant (0.35), on every arm, under both policies. ``--fmad=false`` stays
    # because the hazard is real in principle; this run is evidence that NVRTC 11.6
    # on an A6000 did not take it, and evidence of nothing else.
    "contraction_guard": {
        "flag": "--fmad=false",
        "comparable_cases": 112,
        "unguarded_diverged": 0,
        "reading": "MEASURED DECORATIVE on this compiler and this body",
    },
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-20_no_pml_curl"),
    # RECOMPUTED as a CONTROLLED comparison rather than a difference of two
    # rounds: the previous round's analyzer (whose UNION_FAMILIES does not know
    # this family) run over THIS round's data gives 595/759, and the same data
    # with the family added gives 605/759. The two differ in one line of
    # configuration, so +10 slots and +4 rows-covered-at-every-sub-step is this
    # family's contribution and nothing else's. Reading 605 - 557 against the
    # 2026-08-20_all_families figure would credit this family with other
    # families' work that landed between the two runs.
    "slots_before": 595,
    "slots_after": 605,
    "rows_covered_at_every_sub_step_before": 113,
    "rows_covered_at_every_sub_step_after": 117,
    "slots_gained": 10,
    "rows_served": ("material-dispersion.py (stored_E)",
                    "TestMaterialDispersion.test_material_dispersion_with_user_material",
                    "TestMedium.test_check_material_frequencies",
                    "TestSimulation.test_harminv_warnings",
                    "TestEigenModeSource.test_amp_func_change_sources"),
    "derived_arm_share_of_step_B_slots": "4 of 5",
    "what_it_does_not_license": (
        "a conductivity. MEEP's Absorber boundary layer IS a conductivity, and "
        "three of the eight real-storage no-absorber corpus rows carry one; "
        "_apply_curl routes them to _apply_conductive_update (stepping.py:536-537), "
        "a different recurrence no kernel here implements. Those are the 6 curl "
        "slots this family refuses and the largest thing it did not build.",
        "complex storage. Six of the fourteen no-absorber corpus rows are "
        "force_complex_fields runs and belong to the complex pair; this predicate "
        "refuses them for 'Bx is complex64, not float32'.",
        "the constitutive sub-steps, which under an inactive layer are a null "
        "plan (no_pml_constitutive.py), not a kernel.",
        "cylindrical coordinates, a Bloch phase, BFAST, special_kz, chi2/chi3 and "
        "a third simultaneous mirror plane, each refused by an inherited clause "
        "and untested here.",
        "any throughput claim: the gate is a correctness gate and times nothing.",
        "any dispatch. Nothing in meep_gpu imports cuda_kernels, so a True verdict "
        "licenses a MEASUREMENT and not a production step.",
    ),
}
