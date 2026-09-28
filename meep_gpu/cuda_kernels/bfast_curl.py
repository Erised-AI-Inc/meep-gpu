"""The BFAST (``grid.bfast_active``) curl pair for the hand-CUDA track.

WHAT BFAST IS (stepping.py:826-932; MEEP step_db.cpp:129-142 +
step_generic.cpp:335-471). The broadband fixed-angle source technique is the
time-sheared substitution ``t -> t - k.r/c``, under which a planewave at a FIXED
incidence angle is transversely uniform at EVERY frequency, so plain periodic
boundaries are exact for a whole band. In the stepper it is exactly::

    dB/dt = -curl(E) + d/dt (k x E)
    dD/dt = +curl(H) - d/dt (k x H)

THE ARITHMETIC, TRANSCRIBED. Per component, over the SAME four operands the curl
already gathered (``CurlOperands``, stepping.py:1591-1604)::

    total   = k1 * (shifted_first + first) - k2 * (shifted_second + second)  # :890-891
    advance = total - 2.0 * F_prev                                           # :895
    <mask>  = _mask_non_owned_cells(advance, grid, term.iyee)                # :902
    F      += advance                                                        # :903
    return  -advance                                                         # :904

and the caller adds that to the curl BEFORE the curl's own mask
(stepping.py:392-396 / :475-478 against :397 / :479). ``F_n = S_n - F_{n-1}`` with
output ``F_n - F_{n-1}`` is the z-domain filter ``(1 - z^-1)/(1 + z^-1)`` -- the
Tustin derivative -- so ``advance`` is ``dt * d/dt`` of the two-point average, and
NO ``dtdx`` and no explicit ``dt`` appear anywhere: MEEP passes ``dtdx`` to
``step_bfast`` and the body never reads it.

WHY THE SUM AND NOT THE DIFFERENCE, and why that matters to a kernel: the curl
DIFFERENCES the same operands and BFAST SUMS them, so an invariant axis (whose
shifted operand is the cell itself) contributes an exact zero to the curl and
``2*g`` here. That is the single easiest place to lose the term silently, and it is
why the ``have_p``/``have_m`` gates are applied on the HOST, in
:func:`bfast_curl_coefficients`, rather than being re-derived in the kernel.

THE k ASSIGNMENT (step_db.cpp:129-136; stepping.py:908-913), the subtle part::

    k1 = have_m ? bfast_scaled_k[component_index(c_m)] : 0   // multiplies g1 = f_p
    k2 = have_p ? bfast_scaled_k[component_index(c_p)] : 0   // multiplies g2 = f_m
    if (ft == D_stuff) { k1 = -k1; k2 = -k2; }

Each k is indexed by the OTHER partner's OWN DIRECTION -- ``component_index``
(vec.hpp:445), not the direction its derivative is taken along. For Bx (first = Ez,
second = Ey) that is ``k1 = k_y`` on Ez and ``k2 = k_z`` on Ey, i.e.
``(k x E)_x = k_y E_z - k_z E_y``. Getting it wrong merely moves the term to the
wrong pair of components, silently, which is why MEEP wrote its "puts k1 in
direction of g2" comments and why :data:`BFAST_TERMS` is pinned against
``stepping.B_CURL_TERMS`` by the laptop test rather than trusted.

THE STATE IS THE HAZARD. ``f_bfast_*`` is a per-component IIR state whose
homogeneous mode ``(-1)^n`` is UNDAMPED forever, so a kernel that gets the field
right and the state wrong is correct for exactly one launch and wrong from the
second. The state is therefore an OUTPUT this family's gate compares on every case,
beside the field and the PML auxiliary -- and the mask is applied to ``advance``
BEFORE the state absorbs it, because MEEP writes ``F`` only inside its owned-cell
loop.

WHAT THIS IS FOR, stated narrowly and measured. On the 2026-08-20 union census
exactly ONE corpus row carries BFAST and is refused for it at every one of its four
sub-steps: ``tests/TestReflectanceAngular.test_reflectance_angular_2_35_7``
(``bfast_scaled_k = (0.8169576958985646, 0, 0)``, 1x1x1800, all-periodic, no fold,
courant 0.10567952354605308, real storage). Replaying it with the BFAST clause
satisfied at the input measured every other clause of both shipped real-storage
predicates already passing, so this family's ceiling is 4 slots and not more.

THE FOLD IS REFUSED BY NAME, and that costs nothing. The certified curl pair
admits a mirror fold since 2026-08-20 on a device verdict of its own, but that
verdict is about a stencil that DIFFERENCES its ghosts; BFAST SUMS them, so the
"the folded ghost values are dead" half of the argument does not transfer and would
have to be measured separately. No corpus row pairs BFAST with a fold, so refusing
it is free -- and a refusal that costs nothing is the one place a family should
never widen by argument.

NO DISPATCH. Nothing in ``meep_gpu`` imports ``cuda_kernels``; these kernels are
reachable only from ``parity/meep_gpu/gate_cuda_bfast.py`` and the laptop tests, so
:func:`covers_bfast_curl` returning True licenses a MEASUREMENT and not a
production step.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, Dict, Tuple

from . import coverage as _coverage


def _sibling_prelude() -> str:
    """``step_curl_kernels._REAL_PML_PRELUDE``, READ rather than imported or copied.

    Same reason as ``no_pml_curl._sibling_prelude`` and
    ``special_kz_curl._sibling_prelude``: importing the certified module would pull
    ``cupy`` in and make this file unimportable where the coverage census runs,
    and copying the prelude would fork ``shift_up`` / ``shift_dn`` / ``pml_apply``
    silently. Parsing the sibling for its own literal has neither problem.
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
        "_REAL_PML_PRELUDE; the BFAST curl pair shares that prelude verbatim and "
        "cannot be built without it")


_REAL_PML_PRELUDE = _sibling_prelude()

__all__ = (
    "BFAST_ADMISSION",
    "BFAST_KERNELS",
    "BFAST_TERMS",
    "CERTIFIED_KERNELS",
    "UNCERTIFIED_KERNELS",
    "bfast_curl_coefficients",
    "covers_bfast_constitutive",
    "covers_bfast_curl",
    "kernel_source",
)

#: The two kernels this module carries, in ``STEP_ORDER`` order.
BFAST_KERNELS: Tuple[str, ...] = ("step_B_bfast_real",
                                  "step_D_bfast_real")

#: Byte-identical to the array path with a device verdict behind it. The evidence
#: is :data:`BFAST_ADMISSION` and ``certification.json``'s ``cuda_bfast_2026-08-21``
#: block, which names the same two artifacts; a name here without a record block is
#: the failure ``test_kernel_partition.py`` exists to catch.
CERTIFIED_KERNELS = ("step_B_bfast_real", "step_D_bfast_real")

#: Shipped but not gated. EMPTY, and the partition requires every shipped kernel to
#: be in exactly one of the two sets, so a kernel added to this file without a gate
#: verdict fails there rather than shipping unmeasured. Spelled as a dict WITHOUT a
#: type annotation for the reason ``constitutive_kernels.py`` records: the partition
#: readers walk the syntax tree so they run where there is no CuPy, and an annotated
#: assignment is an ``ast.AnnAssign`` the plain-assignment readers do not match --
#: annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}

_AXIS_NAMES: Tuple[str, str, str] = ("x", "y", "z")

#: ``sub_step -> ((target, first, first_axis, second, second_axis), ...)``, in the
#: kernel's block order. A TRANSCRIPTION of ``stepping.B_CURL_TERMS`` /
#: ``D_CURL_TERMS`` (stepping.py:214-223), restated here because this module must
#: import nothing from the engine to stay evaluable on the census host -- and
#: PINNED against those tables by ``test_bfast_curl.py``, because the k assignment
#: below reads ``first``/``second`` and confusing them is a silent wrong answer
#: rather than a crash.
BFAST_TERMS: Dict[str, Tuple[Tuple[str, str, int, str, int], ...]] = {
    "step_B": (("Bx", "Ez", 1, "Ey", 2),
               ("By", "Ex", 2, "Ez", 0),
               ("Bz", "Ey", 0, "Ex", 1)),
    "step_D": (("Dx", "Hz", 1, "Hy", 2),
               ("Dy", "Hx", 2, "Hz", 0),
               ("Dz", "Hy", 0, "Hx", 1)),
}


def _bfast_axis(component: str) -> int:
    """``stepping._bfast_axis`` (stepping.py:814-823), transcribed.

    x -> 0, y -> 1, z -> 2 from a component name's LAST LETTER. ``component_index``
    (vec.hpp:445) indexes ``bfast_scaled_k`` by the component's OWN direction, not
    by the direction its derivative is taken along.
    """
    return _AXIS_NAMES.index(component[-1].lower())


def bfast_curl_coefficients(grid: Any, sub_step: str
                            ) -> Tuple[Tuple[float, float], ...]:
    """``((k1, k2), (k1, k2), (k1, k2))`` for one sub-step's three targets.

    ``stepping._bfast_term``'s host half (stepping.py:907-914), transcribed::

        have_p = not grid.is_invariant(term.first_axis)
        have_m = not grid.is_invariant(term.second_axis)
        k1 = bfast[_bfast_axis(term.second)] if have_m else 0.0
        k2 = bfast[_bfast_axis(term.first)]  if have_p else 0.0
        if not magnetic: k1, k2 = -k1, -k2          # MEEP's ft == D_stuff
        ... dtype.type(k1) * (...)                  # :890-891, ONE round each

    THE INVARIANCE GATES ARE HOST-SIDE and that is the transcription's point: they
    are ``figure_out_step_plan``'s ``have_p``/``have_m`` (fields.cpp:428-455), false
    exactly when the derivative direction is not one the grid resolves. The curl
    would swallow that case by itself -- an invariant axis makes its DIFFERENCE an
    exact zero -- but the BFAST SUM of the same two samples is ``2*g``, so a kernel
    that re-derived the gate from the shifted operand would be wrong on a reduced
    dimension grid and right everywhere else.

    The negation for the D side is taken in float64 and the result rounded ONCE, as
    stepping.py rounds it: float64 negation and float32 rounding commute exactly, so
    the two spellings cannot differ, but the literal order is kept.
    """
    if sub_step not in BFAST_TERMS:
        raise ValueError(f"sub_step must be 'step_B' or 'step_D', got {sub_step!r}")
    import numpy  # noqa: PLC0415

    magnetic = sub_step == "step_B"
    bfast = tuple(float(value) for value in grid.bfast_scaled_k)
    out = []
    for _target, first, first_axis, second, second_axis in BFAST_TERMS[sub_step]:
        have_p = not bool(grid.is_invariant(first_axis))
        have_m = not bool(grid.is_invariant(second_axis))
        k1 = bfast[_bfast_axis(second)] if have_m else 0.0
        k2 = bfast[_bfast_axis(first)] if have_p else 0.0
        if not magnetic:
            k1, k2 = -k1, -k2
        out.append((float(numpy.float32(k1)), float(numpy.float32(k2))))
    return tuple(out)


# B side: the CERTIFIED ``step_B_pml_real`` block for block, with the BFAST
# insert between the curl and the masks. The four operands are the ones the curl
# already loaded -- ``f1``/``sf`` are MEEP's g1 pair and ``f2``/``ss`` its g2 pair
# (``CurlOperands``: first, shifted_first, second, shifted_second) -- so the term
# costs no new load.
#
# THE MASK IS APPLIED TWICE, ON PURPOSE, AND TO TWO DIFFERENT THINGS. The array path
# masks ``advance`` (stepping.py:929) before the state absorbs it and before it is
# handed back, then masks the ``curl`` it was added to (stepping.py:397). Masking
# only the curl would leave the STATE stepping cells MEEP's owned loop never
# visits -- invisible for one launch, and permanent afterwards, because the
# ``(-1)^n`` mode never decays.
#
# ``curl - advance`` CARRIES ``curl + (-advance)``: IEEE-754 defines subtraction as
# addition of the negation, so the single subtract is the same bits on every input.
_step_B_bfast_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void step_B_bfast_real(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    float* __restrict__ fb_Bx, float* __restrict__ fb_By, float* __restrict__ fb_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    float k1_a, float k2_a, float k1_b, float k2_b, float k1_c, float k2_c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Bx: curl_x = dEz/dy - dEy/dz; dsig=y, dsigu=z; iyee=(0,1,1).
    // BFAST: k1 = k_y on Ez (g1), k2 = k_z on Ey (g2) -- (k x E)_x.
    {
        float f1 = Ez[idx];
        float sf = shift_up(Ez, idx, j, ny, sy, bc_y);
        float f2 = Ey[idx];
        float ss = shift_up(Ey, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        float total = (k1_a * (sf + f1)) - (k2_a * (ss + f2));
        float bprev = fb_Bx[idx];
        float advance = total - (2.0f * bprev);
        if (bc_x == BC_METALLIC && i == 0) advance = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) advance = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) advance = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) advance = 0.0f;
        fb_Bx[idx] = bprev + advance;
        curl = curl - advance;
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        pml_apply(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // By: curl_y = dEx/dz - dEz/dx; dsig=z, dsigu=x; iyee=(1,0,1).
    // BFAST: k1 = k_z on Ex (g1), k2 = k_x on Ez (g2).
    {
        float f1 = Ex[idx];
        float sf = shift_up(Ex, idx, k, nz, sz, bc_z);
        float f2 = Ez[idx];
        float ss = shift_up(Ez, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        float total = (k1_b * (sf + f1)) - (k2_b * (ss + f2));
        float bprev = fb_By[idx];
        float advance = total - (2.0f * bprev);
        if (bc_y == BC_METALLIC && j == 0) advance = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) advance = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) advance = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) advance = 0.0f;
        fb_By[idx] = bprev + advance;
        curl = curl - advance;
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        pml_apply(By, fu_By, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Bz: curl_z = dEy/dx - dEx/dy; dsig=x, dsigu=y; iyee=(1,1,0).
    // BFAST: k1 = k_x on Ey (g1), k2 = k_y on Ex (g2). EVERY component carries a
    // term here -- unlike special_kz, whose z component gets none.
    {
        float f1 = Ey[idx];
        float sf = shift_up(Ey, idx, i, nx, sx, bc_x);
        float f2 = Ex[idx];
        float ss = shift_up(Ex, idx, j, ny, sy, bc_y);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        float total = (k1_c * (sf + f1)) - (k2_c * (ss + f2));
        float bprev = fb_Bz[idx];
        float advance = total - (2.0f * bprev);
        if (bc_z == BC_METALLIC && k == 0) advance = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) advance = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) advance = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) advance = 0.0f;
        fb_Bz[idx] = bprev + advance;
        curl = curl - advance;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        pml_apply(Bz, fu_Bz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }
}
'''

# D side: MEEP's negated strides, integer-position coefficients, iyee -- Dx
# (1,0,0), Dy (0,1,0), Dz (0,0,1). The D-side sign flip (``k1 = -k1; k2 = -k2``,
# step_db.cpp:136) is taken on the HOST in :func:`bfast_curl_coefficients`, so the
# two kernel bodies differ only where the certified pair differs.
_step_D_bfast_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void step_D_bfast_real(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    float* __restrict__ fb_Dx, float* __restrict__ fb_Dy, float* __restrict__ fb_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    float k1_a, float k2_a, float k1_b, float k2_b, float k1_c, float k2_c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Dx: curl_x = dHz/dy - dHy/dz; dsig=y, dsigu=z; iyee=(1,0,0).
    {
        float f1 = Hz[idx];
        float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);
        float f2 = Hy[idx];
        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        float total = (k1_a * (sf + f1)) - (k2_a * (ss + f2));
        float bprev = fb_Dx[idx];
        float advance = total - (2.0f * bprev);
        if (bc_y == BC_METALLIC && j == 0) advance = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) advance = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) advance = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) advance = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) advance = 0.0f;
        fb_Dx[idx] = bprev + advance;
        curl = curl - advance;
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        pml_apply(Dx, fu_Dx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // Dy: curl_y = dHx/dz - dHz/dx; dsig=z, dsigu=x; iyee=(0,1,0).
    {
        float f1 = Hx[idx];
        float sf = shift_dn(Hx, idx, k, nz, sz, bc_z);
        float f2 = Hz[idx];
        float ss = shift_dn(Hz, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        float total = (k1_b * (sf + f1)) - (k2_b * (ss + f2));
        float bprev = fb_Dy[idx];
        float advance = total - (2.0f * bprev);
        if (bc_x == BC_METALLIC && i == 0) advance = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) advance = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) advance = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) advance = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) advance = 0.0f;
        fb_Dy[idx] = bprev + advance;
        curl = curl - advance;
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Dz: curl_z = dHy/dx - dHx/dy; dsig=x, dsigu=y; iyee=(0,0,1).
    {
        float f1 = Hy[idx];
        float sf = shift_dn(Hy, idx, i, nx, sx, bc_x);
        float f2 = Hx[idx];
        float ss = shift_dn(Hx, idx, j, ny, sy, bc_y);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        float total = (k1_c * (sf + f1)) - (k2_c * (ss + f2));
        float bprev = fb_Dz[idx];
        float advance = total - (2.0f * bprev);
        if (bc_x == BC_METALLIC && i == 0) advance = 0.0f;
        if (bc_y == BC_METALLIC && j == 0) advance = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) advance = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) advance = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) advance = 0.0f;
        fb_Dz[idx] = bprev + advance;
        curl = curl - advance;
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }
}
'''

_SOURCES: Dict[str, str] = {
    "step_B_bfast_real": _step_B_bfast_real_kernel_code,
    "step_D_bfast_real": _step_D_bfast_real_kernel_code,
}

#: ``sub_step -> kernel name``, so a caller does not spell an if.
KERNEL_FOR_SUB_STEP: Dict[str, str] = {
    "step_B": "step_B_bfast_real",
    "step_D": "step_D_bfast_real",
}

#: The IIR state each sub-step advances, in block order. It is an OUTPUT, not a
#: scratch: the gate compares it on every case.
BFAST_STATE: Dict[str, Tuple[str, str, str]] = {
    "step_B": ("f_bfast_Bx", "f_bfast_By", "f_bfast_Bz"),
    "step_D": ("f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"),
}


def kernel_source(name: str) -> str:
    """The device text for one kernel, by name. One seam for a gate to mutate."""
    if name not in _SOURCES:
        raise ValueError(f"no such BFAST kernel {name!r}; have {sorted(_SOURCES)}")
    return _SOURCES[name]


def set_kernel_source(name: str, source: str) -> None:
    """Replace one kernel's device text -- the gate's mutation seam, and only that."""
    if name not in _SOURCES:
        raise ValueError(f"no such BFAST kernel {name!r}; have {sorted(_SOURCES)}")
    _SOURCES[name] = source


#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, AND WHAT HAS NOT.
#:
#: Filled in from ``parity/meep_gpu/results/cuda_bfast_2026-08-21_rename_r2/`` -- two legs,
#: one per float32 subnormal policy, on ONE verified-empty RTX A6000. Both
#: released. Nothing here licenses a DISPATCH: no shipped module imports
#: ``cuda_kernels``.
BFAST_ADMISSION: Dict[str, Any] = {
    "gate": "parity/meep_gpu/gate_cuda_bfast.py",
    "artifacts": ("parity/meep_gpu/results/cuda_bfast_2026-08-21_rename_r2/keep/gate.json",
                  "parity/meep_gpu/results/cuda_bfast_2026-08-21_rename_r2/flush/gate.json"),
    "host": "the GPU host",
    "device": "NVIDIA RTX A6000 (cc 8.6), CuPy 13.5.1, NVRTC 11.6",
    #: WHAT MOVED IN THIS FILE SINCE THE RUN, and why the verdict survives it.
    #: Declared rather than left to be inferred from a file hash that no longer
    #: matches, and the survival claim is MEASURED, not argued: the sha256 of every
    #: string :func:`kernel_source` returns is a key in BOTH artifacts'
    #: ``nvrtc_binary_report.binary_sha256_by_source``, i.e. a source the gate's own
    #: observer saw handed to NVRTC. ``test_special_kz_bfast_admission.py``
    #: re-performs that comparison on every run and refuses the declaration if a
    #: device string has moved.
    "post_gate_record_edits": (
        "2026-08-28: this module gained CERTIFIED_KERNELS, UNCERTIFIED_KERNELS and "
        "the two names in __all__. Host-side only; no device string moved.",),
    "device_bound_by": ("nvrtc_binary_report.binary_sha256_by_source, carried in "
                        "both artifacts"),
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "courants": (0.5, 0.35),
    "value_classes": ("uniform", "subnormal_band"),
    "multi_step_budget": 60,
    # The CURL, per policy: 48 scored cases, all bit-identical at one launch and
    # at 60 -- FIELD, PML AUXILIARY AND ``f_bfast`` IIR STATE, the state compared
    # as an output on every case.
    "curl_cases_scored": 48,
    "curl_single_launch_identical": 48,
    "curl_multi_step_identical": 48,
    "curl_full_rank_cases": 40,
    "curl_invariant_axis_cases": 8,
    "state_compared_as_an_output": True,
    "constitutive_cases_scored": {"H": 24, "E": 24},
    "constitutive_identical": {"H": 24, "E": 24},
    "source_mutations_caught": 30,
    "source_mutations_null_confirmed": 8,
    "host_mutations_caught": 5,
    "host_mutations_null_confirmed": 1,
    "fmad_false_is_load_bearing": {"comparable": 24, "diverged": 24},
    "what_it_does_not_license": (
        "any dispatch: nothing in meep_gpu imports cuda_kernels.",
        "A MIRROR FOLD. Refused by name and NOT swept: the certified pair's fold "
        "verdict rests on the folded ghost VALUES being dead, which is a claim "
        "about a stencil that differences its ghosts, and BFAST sums them. No "
        "corpus row pairs the two, so the refusal costs nothing.",
        "complex storage: Fields._ensure_bfast_storage allocates one COMPLEX "
        "state per component and these kernels index float32.",
        "cylindrical BFAST: Grid itself refuses it (grid.py:699+).",
        "a conductivity, beta, a Bloch phase or dispersion: refused by inherited "
        "clauses, untested here.",
        "any throughput claim: this is a correctness gate and times nothing.",
    ),
}


# ---------------------------------------------------------------------------
# COMPILATION AND LAUNCH
# ---------------------------------------------------------------------------
#
# ``cupy`` is imported INSIDE these functions, never at module scope, so the
# predicate above stays callable on the census laptop.

#: ``--fmad=false`` is CORRECTNESS on this pair and not tuning. Beside the two
#: contraction candidates the certified kernels carry, BFAST adds two more of
#: exactly the shape a compiler fuses: ``k1*(sf+f1) - k2*(ss+f2)`` is a
#: multiply feeding a subtract of another multiply, and ``total - 2.0f*bprev`` is
#: a multiply feeding a subtract. The array path rounds every one of those
#: separately (stepping.py:917-922). The gate substitutes the empty tuple as a
#: control and reports whether the guard changed an answer.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: The block size the certified pair launches at, restated rather than imported.
_THREADS = 256


def _get_kernel(sub_step: str):
    """The compiled kernel for one sub-step, through the shared memo.

    Source read through :func:`kernel_source` PER CALL, so a body rewritten by a
    gate's mutation leg is a memo MISS and reaches NVRTC -- the defect the sibling
    track hit three times is a leg reporting a pass for a mutation it never
    applied.
    """
    import cupy as cp  # noqa: PLC0415
    from .compile_cache import get_or_compile, kernel_cache_key  # noqa: PLC0415

    name = KERNEL_FOR_SUB_STEP[sub_step]
    code = kernel_source(name)
    key = kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    from .compile_cache import clear_kernel_cache  # noqa: PLC0415

    return clear_kernel_cache()


def step_bfast(sub_step: str, fields: Any, tables: Dict[str, Any],
               boundary_codes: Any, dtdx: float,
               coefficients: Tuple[Tuple[float, float], ...]) -> None:
    """One BFAST curl sub-step, in place, in one launch.

    ``tables`` are the six flattened kms/sinv views from
    ``step_curl_kernels.real_pml_curl_tables`` -- HALF-INTEGER for ``step_B``,
    INTEGER for ``step_D``. ``coefficients`` is :func:`bfast_curl_coefficients`'
    three ``(k1, k2)`` pairs, in block order; the six scalars are bound LAST and
    the three IIR state pointers immediately after the auxiliaries, so the
    argument list is the certified one with two appends a reader can diff.
    """
    import numpy  # noqa: PLC0415

    targets = (("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz"))
    sources = (("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Hx", "Hy", "Hz"))
    volumes = [getattr(fields, name) for name in targets]
    auxiliaries = [getattr(fields, "fu_" + name) for name in targets]
    states = [getattr(fields, name) for name in BFAST_STATE[sub_step]]
    operands = [getattr(fields, name) for name in sources]
    nx, ny, nz = volumes[0].shape
    blocks = (nx * ny * nz + _THREADS - 1) // _THREADS
    scalars = [numpy.float32(value) for pair in coefficients for value in pair]
    _get_kernel(sub_step)((blocks,), (_THREADS,), (
        volumes[0], volumes[1], volumes[2],
        auxiliaries[0], auxiliaries[1], auxiliaries[2],
        states[0], states[1], states[2],
        operands[0], operands[1], operands[2],
        numpy.int32(nx), numpy.int32(ny), numpy.int32(nz), numpy.float32(dtdx),
        tables["kms_x"], tables["sinv_x"],
        tables["kms_y"], tables["sinv_y"],
        tables["kms_z"], tables["sinv_z"],
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
        *scalars,
    ))


class _BfastFreeGrid:
    """The run's grid, answering ``bfast_active = False`` and forwarding the rest.

    Satisfying AT THE INPUT the single clause this family inverts -- the technique
    ``no_pml_curl._ActiveLayerProxy`` uses, and for the same reason: the shipped
    predicates SHORT-CIRCUIT, so there is no accumulated refusal list to filter
    afterwards. ``grid.bfast_active`` is read in exactly one place in each shipped
    predicate (``coverage._grid_facts``:429, consumed at :721 and :1049).

    ``bfast_scaled_k`` is NOT masked: nothing in the certified predicates reads it,
    and a proxy that hid it would hide it from this module's own clauses too.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)

    @property
    def bfast_active(self) -> bool:
        return False

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), item)


def _bfast_reasons(fields: Any, grid: Any, targets: Tuple[str, ...]) -> Any:
    """The clauses THIS family owns, in place of the one it inverts. None if clear.

    1. **BFAST must be ACTIVE.** A ``bfast_active`` False run belongs to the
       certified pair and the array path never enters the term
       (stepping.py:392/:475). Admitting one here would put two families on one
       slot, which the union census reports as a FINDING, not as coverage.
    2. **real storage.** ``Fields._ensure_bfast_storage`` allocates ONE COMPLEX
       array per component under complex storage (fields.py:632-651) and these
       kernels index float32; the certified predicate refuses complex storage
       anyway, so this clause names the family that owns the configuration rather
       than describing the width.
    3. **the state must be allocated, float32, contiguous, grid-shaped.** The
       array path RAISES when it is missing (stepping.py:917-920) -- a run this
       family could not serve either -- and a kernel handed a differently shaped
       one would corrupt memory rather than refuse.
    4. **no mirror fold.** See the module docstring: the certified pair's fold
       verdict is about a stencil that DIFFERENCES its ghosts and BFAST SUMS them,
       so it does not transfer, and no corpus row pairs the two.
    """
    try:
        active = bool(getattr(grid, "bfast_active", False))
        scaled = tuple(float(v) for v in (getattr(grid, "bfast_scaled_k", ()) or ()))
        mirrored = tuple(bool(grid.is_mirrored(axis)) for axis in range(3))
        symmetric = bool(grid.has_symmetry())
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return (f"the grid could not answer a question this predicate has to ask: "
                f"{type(exc).__name__}: {exc}")
    if not active:
        return ("grid.bfast_active is False: this pair exists only for BFAST runs, "
                "and a non-BFAST run belongs to the certified "
                "step_*_pml_real")
    if len(scaled) != 3:
        return (f"grid.bfast_scaled_k is {scaled!r}, not a three-component vector; "
                f"the kernel binds one (k1, k2) pair per target from it")
    if getattr(fields, "force_complex_fields", False):
        return ("complex64 storage with BFAST: Fields._ensure_bfast_storage "
                "allocates one COMPLEX state per component (fields.py:632-651) and "
                "these kernels index float32")
    if symmetric or any(mirrored):
        return ("mirror symmetry with BFAST: the certified pair's fold verdict is "
                "about a stencil that DIFFERENCES its ghosts, and BFAST SUMS the "
                "same operands -- so the 'folded ghost values are dead' half of "
                "that argument does not transfer and has not been measured here")
    xp = getattr(grid, "xp", None)
    try:
        shape = tuple(grid.shape)
    except Exception as exc:  # noqa: BLE001
        return f"the grid could not be asked for its shape: {type(exc).__name__}: {exc}"
    for target in targets:
        name = "f_bfast_" + target
        state = getattr(fields, name, None)
        if state is None:
            return (f"{name} is not allocated: the BFAST IIR state is an OUTPUT of "
                    f"this sub-step and stepping raises without it "
                    f"(stepping.py:917-920)")
        problem = _coverage._array_problem(name, state, xp, shape)
        if problem is not None:
            return problem
    return None


def covers_bfast_curl(fields: Any, pml: Any, grid: Any, sub_step: str) -> tuple:
    """Whether ``step_{B,D}_bfast_real`` may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. Returns ``(covered, reason)``.

    THE SUB-STEP ARGUMENT IS REQUIRED for the reason
    :func:`coverage.covers_real_pml_curl`'s is -- the conductivity is read PER
    TARGET COMPONENT (stepping.py:508) -- and for a second reason of this family's
    own: the IIR state is per sub-step, so ``step_B`` and ``step_D`` do not stand or
    fall on the same arrays.

    DELEGATION, NOT A SECOND CLAUSE SET. Every question except the BFAST clause and
    the state inventory is the CERTIFIED curl predicate's, asked through it on a
    proxy grid whose ``bfast_active`` reads False.
    """
    if sub_step not in ("step_B", "step_D"):
        raise ValueError(f"sub_step must be 'step_B' or 'step_D', got {sub_step!r}")
    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    own = _bfast_reasons(fields, grid, targets)
    if own is not None:
        return False, own
    return _coverage.covers_real_pml_curl(fields, pml, _BfastFreeGrid(grid), sub_step)


def covers_bfast_constitutive(fields: Any, pml: Any, grid: Any, side: str) -> tuple:
    """Whether the CERTIFIED constitutive pair may serve ``update_H``/``update_E``
    on a BFAST run. ``side`` is ``"H"`` or ``"E"``; returns ``(covered, reason)``.

    THIS FAMILY BUILDS NO CONSTITUTIVE KERNEL. ``stepping.update_H`` (:907-923) and
    ``update_E`` (:926-993) reach ``_apply_constitutive_pml`` (:2065) and none of
    them reads ``grid.bfast_scaled_k`` or any ``f_bfast_*`` array: BFAST enters the
    step loop at stepping.py:392-396 and :475-478 and nowhere else. So the admission
    is the shipped predicate's with the BFAST clause inverted, and the ARITHMETIC is
    ``constitutive_kernels``' own, untouched.

    THAT READING IS NOT THE EVIDENCE, and the state is why it must not be. BFAST is
    the one auxiliary in this stepper whose homogeneous mode never decays, so a
    constitutive sub-step that touched it -- or that a BFAST run's own history moved
    differently -- would be wrong from launch two rather than launch one.
    ``gate_cuda_bfast.py`` runs the certified pair against the array path on a grid
    whose BFAST is live and whose curls have already advanced the state, and the two
    sides are asked SEPARATELY.

    The BFAST STATE INVENTORY IS NOT ASKED HERE. These sub-steps do not read or
    write ``f_bfast_*``, so requiring it would refuse a configuration on the absence
    of an array the sub-step never touches -- the same mistake in the other
    direction from admitting one by attribute absence.
    """
    if side not in ("H", "E"):
        raise ValueError(f"side must be 'H' or 'E', got {side!r}")
    own = _bfast_reasons(fields, grid, ())
    if own is not None:
        return False, own
    return _coverage.covers_real_pml_constitutive(
        fields, pml, _BfastFreeGrid(grid), side)
