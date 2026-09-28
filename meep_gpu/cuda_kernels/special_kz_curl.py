"""The special_kz (``grid.beta != 0``) curl pair for the hand-CUDA track.

WHAT BETA IS (grid.py:668-697 ``_resolve_beta``; stepping.py:754-811). A 2-D run
carries ``exp(i*2*pi*beta*z)`` analytically, so ``d/dz`` on the invariant axis is
the EXACT factor ``i*2*pi*beta`` -- an ``i*beta*zhat x`` cross product folded into
the curl (MEEP step_db.cpp:148-176). MEEP calls it ``special_kz`` and reaches it
from ``mp.Simulation(kz_2d=...)`` on a cell with ``cell_size.z == 0`` and
``k_point.z != 0``.

THE FAMILY ADDS ONE TERM TO THE TWO CURL SUB-STEPS AND CHANGES NOTHING ELSE:

* call sites -- ``step_B`` (stepping.py:384-391): Bx adds the Ey-CENTRE term at
  sign +1 (:389), By the Ex-CENTRE term at sign -1 (:391), Bz nothing;
  ``step_D`` (:438-445): Dx <- Hy at +1 (:443), Dy <- Hx at -1 (:445), Dz nothing.
  ``cc`` runs over ``d_c`` in {X, Y} only, which is why the z component gets none;
* coefficient (stepping.py:797) -- ``sign * 2*pi * grid.beta * grid.dt``. NO
  ``dtdx``: this is an analytic derivative, not a finite difference (:758-762).
  Under REAL storage the ``+-1j`` factor at :771-772 is not taken at all --
  ``partner_values.dtype.kind`` is ``'f'`` -- so BOTH sub-steps use the SAME plain
  real coefficient. That is MEEP's "implicitly store i*(TM fields)" trick
  (step_db.cpp:148-160), an interpretation of the same arithmetic rather than a
  second code path;
* rounding (stepping.py:811) -- ``partner_values.dtype.type(coefficient)``, rounded
  ONCE on the host, then ``return -(c * partner)``: the negation is of the PRODUCT,
  in curl sign convention, because the caller subtracts;
* position -- added to ``curl`` AFTER ``_curl_from_operands`` (:342 / :429) and
  BEFORE ``_mask_non_owned_cells`` (:369 / :450) and ``_apply_curl`` (:374 / :455),
  so the increment rides the SAME split-field kms/sinv ladder as the finite
  difference curl, exactly and linearly;
* partner -- the SAME-CELL component snapshot (:293 electric, :408 magnetic). Per
  ``B_CURL_TERMS`` / ``D_CURL_TERMS`` (stepping.py:214-223) every beta partner is a
  CENTRE operand the certified curl already loads: Bx takes ``f2`` (the second
  source's centre, Ey), By takes ``f1`` (the first source's centre, Ex), Dx takes
  ``f2`` (Hy) and Dy ``f1`` (Hx). The term therefore costs no new pointer and no
  extra memory traffic -- and the load reused must be the UNSHIFTED centre one,
  never an ``sf``/``ss`` shifted operand, which is a gate mutation;
* no new state array, no ghost rule, no fold parity, no wrap. ``update_H`` /
  ``update_E`` (``_apply_constitutive_pml``, stepping.py:2112) read nothing
  beta-dependent, which is why :func:`covers_special_kz_constitutive` below is an
  ADMISSION over the CERTIFIED constitutive pair rather than a third kernel.

WHAT THIS IS FOR, stated narrowly and measured. On the 2026-08-20 union census
(``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_all_families/``)
exactly two REAL-storage corpus rows carry a nonzero beta and are refused for it
at every one of their four sub-steps: ``examples/refl-angular-kz2d.py``
(beta = 0.3321611318837033, 1200x1x1, no fold) and
``tests/TestSpecialKz.test_eigsrc_kz_1_real_imag`` (beta = 0.2, 420x212x1, folded
Y with a PERIODIC termination). Replaying both with the beta clause satisfied at
the input measured every other clause of both shipped real-storage predicates
already passing, so this family's ceiling is 8 slots and not more. The other four
beta rows in the corpus are COMPLEX storage
(``TestSpecialKz.test_special_kz``, ``test_eigsrc_kz_0_complex``, three
``TestEigCoeffs.test_binary_grating_special_kz_*``) and are refused by the complex
family for storage or for the fold, not for beta; nothing here reaches them.

REAL STORAGE ONLY, AND OFF-DIAGONAL EPSILON IS REFUSED BY NAME. With real storage
and a beta run ``stepping._special_kz_beta_term`` RAISES on an off-diagonal
epsilon (stepping.py:800-810) and MEEP aborts on the same combination
(fields.cpp:548-549): the implicit i on the TM half cancels only while mu and
epsilon leave TE and TM uncoupled. The refusal is family-wide rather than E-side
only, because the run itself dies inside the FIRST beta curl.

NO DISPATCH. Nothing in ``meep_gpu`` imports ``cuda_kernels``; these kernels are
reachable only from ``parity/meep_gpu/gate_cuda_special_kz.py`` and the laptop
tests, so :func:`covers_special_kz_curl` returning True licenses a MEASUREMENT and
not a production step.
"""

from __future__ import annotations

import ast
import math
import pathlib
from typing import Any, Dict, Tuple

from . import coverage as _coverage


def _sibling_prelude() -> str:
    """``step_curl_kernels._REAL_PML_PRELUDE``, READ rather than imported or copied.

    The certified curl module imports ``cupy`` at module scope, so importing it
    would make this file -- predicate and all -- unimportable on any machine
    without a device, and the coverage census runs on exactly such a machine.
    Copying the prelude instead would give the two files two sets of bytes that are
    equal only until someone edits one: the ghost helpers, the three boundary codes
    and ``pml_apply`` below ARE the certified ones, and a silent divergence in
    ``shift_up`` would be a wrong answer no diff of this file would show.

    Parsing the sibling's source for its own string literal has neither problem,
    and it is the technique ``no_pml_curl.py`` already uses for the same reason.
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
        "_REAL_PML_PRELUDE; the special_kz curl pair shares that prelude verbatim "
        "and cannot be built without it")


_REAL_PML_PRELUDE = _sibling_prelude()

__all__ = (
    "CERTIFIED_KERNELS",
    "SPECIAL_KZ_ADMISSION",
    "SPECIAL_KZ_KERNELS",
    "UNCERTIFIED_KERNELS",
    "beta_curl_coefficients",
    "covers_special_kz_constitutive",
    "covers_special_kz_curl",
    "kernel_source",
)

#: The two kernels this module carries, in ``STEP_ORDER`` order.
SPECIAL_KZ_KERNELS: Tuple[str, ...] = ("step_B_special_kz_real",
                                       "step_D_special_kz_real")

#: Byte-identical to the array path with a device verdict behind it. The evidence is
#: :data:`SPECIAL_KZ_ADMISSION` and ``certification.json``'s
#: ``cuda_special_kz_2026-08-21`` block, which names the same two artifacts; a name
#: here without a record block is the failure ``test_kernel_partition.py`` exists to
#: catch.
CERTIFIED_KERNELS = ("step_B_special_kz_real", "step_D_special_kz_real")

#: Shipped but not gated. EMPTY, and the partition requires every shipped kernel to
#: be in exactly one of the two sets, so a kernel added to this file without a gate
#: verdict fails there rather than shipping unmeasured. Spelled as a dict WITHOUT a
#: type annotation for the reason ``constitutive_kernels.py`` records: the partition
#: readers walk the syntax tree so they run where there is no CuPy, and an annotated
#: assignment is an ``ast.AnnAssign`` the plain-assignment readers do not match --
#: annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}


def beta_curl_coefficients(beta: float, dt: float) -> Tuple[float, float]:
    """``(plus, minus)`` -- ``stepping._special_kz_beta_term``'s coefficient, twice.

    Transcribed from stepping.py:797 and :811, with the ``+-1j`` branch at :798-799
    NOT taken because this family is real storage only::

        coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt   # :770, float64
        partner_values.dtype.type(coefficient)                     # :784, ONE round

    TWO SCALARS, ONE PER SIGN, rather than one negated in the kernel. float64
    negation and float32 rounding commute exactly, so the two spellings cannot
    differ -- but binding both keeps the transcription literal per call site
    instead of resting on that identity, and it is what the gate's
    ``swap_beta_signs`` host mutation has to corrupt.

    ``numpy`` is imported in the BODY so this module stays stdlib-only at import
    and can be read on the census host; the rounding is deliberately spelled with
    the same object stepping.py uses (``dtype.type``, i.e. ``numpy.float32``)
    rather than with ``struct``, so the two cannot round differently.
    """
    import numpy  # noqa: PLC0415

    return tuple(float(numpy.float32(sign * 2.0 * math.pi * float(beta) * float(dt)))
                 for sign in (1.0, -1.0))


# B side: the CERTIFIED ``step_B_pml_real`` block for block, with the beta
# insert between the curl and the masks -- the array path's own order (:361/:363
# after :342, before :369). Bx's partner is ``f2`` (the Ey CENTRE the stencil
# already loaded) at sign +1, By's is ``f1`` (Ex centre) at sign -1, and Bz gets
# nothing at all.
#
# ``curl - (beta * g)`` CARRIES ``curl + (-(beta * g))``. stepping.py:811 negates
# the PRODUCT and :389 adds it; IEEE-754 defines subtraction as addition of the
# negation, so the single subtract is the same bits on every input, signed zeros
# included. ``--fmad=false`` is what keeps it from contracting into the dtdx
# multiply that precedes it.
_step_B_special_kz_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void step_B_special_kz_real(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    float beta_plus, float beta_minus
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
    // Beta partner: the Ey CENTRE, which is f2, at sign +1 (stepping.py:361).
    {
        float f1 = Ez[idx];
        float sf = shift_up(Ez, idx, j, ny, sy, bc_y);
        float f2 = Ey[idx];
        float ss = shift_up(Ey, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        curl = curl - (beta_plus * f2);
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        pml_apply(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // By: curl_y = dEx/dz - dEz/dx; dsig=z, dsigu=x; iyee=(1,0,1).
    // Beta partner: the Ex CENTRE, which is f1, at sign -1 (stepping.py:363).
    {
        float f1 = Ex[idx];
        float sf = shift_up(Ex, idx, k, nz, sz, bc_z);
        float f2 = Ez[idx];
        float ss = shift_up(Ez, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        curl = curl - (beta_minus * f1);
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        pml_apply(By, fu_By, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Bz: curl_z = dEy/dx - dEx/dy; dsig=x, dsigu=y; iyee=(1,1,0).
    // NO BETA TERM: MEEP's `cc` loop runs over d_c in {X, Y} only
    // (step_db.cpp:148-176), and stepping.py:356-363 has no Bz branch.
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
        pml_apply(Bz, fu_Bz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 361->389, 363->391, 356-363->384-391

# D side: MEEP's negated strides, integer-position coefficients, iyee -- Dx
# (1,0,0), Dy (0,1,0), Dz (0,0,1). SAME SIGNS AS THE B SIDE: the ``+-1j`` that
# distinguishes them lives only in complex storage (stepping.py:798-799), and this
# family is real. Dx's partner is the Hy centre (f2) at +1 (:472), Dy's the Hx
# centre (f1) at -1 (:445), Dz none.
_step_D_special_kz_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void step_D_special_kz_real(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    float beta_plus, float beta_minus
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
    // Beta partner: the Hy CENTRE, which is f2, at sign +1 (stepping.py:443).
    {
        float f1 = Hz[idx];
        float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);
        float f2 = Hy[idx];
        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        curl = curl - (beta_plus * f2);
        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == nx - 1) curl = 0.0f;
        pml_apply(Dx, fu_Dx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // Dy: curl_y = dHx/dz - dHz/dx; dsig=z, dsigu=x; iyee=(0,1,0).
    // Beta partner: the Hx CENTRE, which is f1, at sign -1 (stepping.py:445).
    {
        float f1 = Hx[idx];
        float sf = shift_dn(Hx, idx, k, nz, sz, bc_z);
        float f2 = Hz[idx];
        float ss = shift_dn(Hz, idx, i, nx, sx, bc_x);
        float curl = dtdx * ((sf - f1) + (f2 - ss));
        curl = curl - (beta_minus * f1);
        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;
        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;
        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;
        if (bc_z == BC_MIRROR_PERIODIC && k == 0) curl = 0.0f;
        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;
        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Dz: curl_z = dHy/dx - dHx/dy; dsig=x, dsigu=y; iyee=(0,0,1). NO BETA TERM.
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
        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 443->472, 445->474

_SOURCES: Dict[str, str] = {
    "step_B_special_kz_real": _step_B_special_kz_real_kernel_code,
    "step_D_special_kz_real": _step_D_special_kz_real_kernel_code,
}

#: ``sub_step -> kernel name``, so a caller does not spell an if.
KERNEL_FOR_SUB_STEP: Dict[str, str] = {
    "step_B": "step_B_special_kz_real",
    "step_D": "step_D_special_kz_real",
}


def kernel_source(name: str) -> str:
    """The device text for one kernel, by name.

    Read through a function rather than exported as a module global so a gate that
    mutates the text has ONE seam to patch and the mutation cannot miss a second
    copy.
    """
    if name not in _SOURCES:
        raise ValueError(
            f"no such special_kz kernel {name!r}; have {sorted(_SOURCES)}")
    return _SOURCES[name]


def set_kernel_source(name: str, source: str) -> None:
    """Replace one kernel's device text -- the gate's mutation seam, and only that.

    A gate mutates by rewriting the source and re-launching; the compile memo keys
    on the source string, so a rewritten body is a miss and reaches NVRTC. Exposed
    as a function for the same reason :func:`kernel_source` is: one seam, no second
    copy to forget.
    """
    if name not in _SOURCES:
        raise ValueError(
            f"no such special_kz kernel {name!r}; have {sorted(_SOURCES)}")
    _SOURCES[name] = source


#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, AND WHAT HAS NOT.
#:
#: Filled in from ``parity/meep_gpu/results/cuda_special_kz_2026-08-21_rename_r2/`` -- two
#: legs, one per float32 subnormal policy, on ONE verified-empty RTX A6000. Both
#: released. Nothing here may be read as licensing a DISPATCH: no shipped module
#: imports ``cuda_kernels``, so a True from the predicates above licenses a
#: MEASUREMENT.
SPECIAL_KZ_ADMISSION: Dict[str, Any] = {
    "gate": "parity/meep_gpu/gate_cuda_special_kz.py",
    "artifacts": ("parity/meep_gpu/results/cuda_special_kz_2026-08-21_rename_r2/keep/gate.json",
                  "parity/meep_gpu/results/cuda_special_kz_2026-08-21_rename_r2/flush/gate.json"),
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
    # The CURL, per policy: 72 scored cases, all bit-identical at one launch AND
    # at 60, split 32 unfolded / 40 folded (Y and X, both terminations, both
    # plane parities).
    "curl_cases_scored": 72,
    "curl_single_launch_identical": 72,
    "curl_multi_step_identical": 72,
    "curl_folded_cases": 40,
    "curl_unfolded_cases": 32,
    # The CONSTITUTIVE sides, scored SEPARATELY on beta runs whose two curls had
    # already moved the state -- the arm that turns "update_H/update_E read
    # nothing beta-dependent" from a reading into a measurement.
    "constitutive_cases_scored": {"H": 36, "E": 36},
    "constitutive_identical": {"H": 36, "E": 36},
    "source_mutations_caught": 30,
    "source_mutations_null_confirmed": 10,
    "host_mutations_caught": 5,
    "host_mutations_null_confirmed": 1,
    # MEASURED, not assumed: at the inexact courant the UNGUARDED build diverged
    # on every case where the guarded build was identical.
    "fmad_false_is_load_bearing": {"comparable": 36, "diverged": 36},
    # A MEASURED REFUSAL, and worth as much as an admission. Swapping the beta
    # partner for its SHIFTED companion is UNDETECTABLE here: both partners'
    # only shifted companion is the neighbour along Z, and every beta grid has
    # nz == 1, so the two are the same word. Scored 0/12 CAUGHT across both
    # policies and recorded as a null rather than deleted.
    "beta_partner_shifted_is_structurally_inert": True,
    "what_it_does_not_license": (
        "any dispatch: nothing in meep_gpu imports cuda_kernels.",
        "COMPLEX storage with beta. Four of the six beta corpus rows are complex "
        "(TestSpecialKz.test_special_kz, test_eigsrc_kz_0_complex and three "
        "TestEigCoeffs.test_binary_grating_special_kz legs); all four were "
        "replayed and all four are refused by name for storage.",
        "a z-thick grid: Grid refuses a z extent at dimensions=2, so nz == 1 on "
        "every grid this family can be handed and the z ghost rule and z wall "
        "mask are exercised at one cell only. They are the certified pair's and "
        "are gated on thick grids elsewhere.",
        "off-diagonal epsilon with beta in real storage: stepping RAISES and MEEP "
        "aborts, so there is no such run; refused by name, not measured.",
        "a conductivity, BFAST, a Bloch phase, cylindrical coordinates or three "
        "simultaneous fold planes: refused by inherited clauses, untested here.",
        "any throughput claim: this is a correctness gate and times nothing.",
    ),
}


# ---------------------------------------------------------------------------
# COMPILATION AND LAUNCH
# ---------------------------------------------------------------------------
#
# ``cupy`` is imported INSIDE these functions, never at module scope, so the
# predicate above stays callable on the census laptop. ``compile_cache`` is
# CuPy-free and is imported at the top.

#: ``--fmad=false`` is CORRECTNESS on this pair and not tuning. The certified
#: kernels need it for ``(fu*kms) - curl`` and ``(f*kms_u) + fu_new``; this pair
#: adds a THIRD contraction candidate of exactly the shape a compiler fuses --
#: ``curl - (beta * g)``, a multiply feeding a subtract, where the array path
#: rounds the product (stepping.py:811) and the addition (:389) separately. The
#: gate substitutes the empty tuple as a control and reports whether the guard
#: changed an answer here, rather than asserting that it must.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: The block size the certified pair launches at, restated rather than imported
#: for the same reason the prelude is read rather than imported.
_THREADS = 256


def _get_kernel(sub_step: str):
    """The compiled kernel for one sub-step, through the shared memo.

    The memo key carries the subnormal policy in force AND the source string
    actually compiled, so neither a late policy install nor a body rewritten
    through :func:`set_kernel_source` can be served an earlier binary. The source
    is read through :func:`kernel_source` PER CALL for the same reason
    ``step_curl_kernels._get_kernel`` rebuilds its map per call: a memo above the
    seam would hand back the pre-mutation string forever, and a leg reporting a
    pass for a mutation it never applied is worse than no leg.
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


def step_special_kz(sub_step: str, fields: Any, tables: Dict[str, Any],
                    boundary_codes: Any, dtdx: float,
                    beta_coefficients: Tuple[float, float]) -> None:
    """One special_kz curl sub-step, in place, in one launch.

    ``tables`` are the six flattened kms/sinv views from
    ``step_curl_kernels.real_pml_curl_tables`` -- HALF-INTEGER for ``step_B``
    (stepping.py:404), INTEGER for ``step_D`` (:486). ``beta_coefficients`` is
    :func:`beta_curl_coefficients`' ``(plus, minus)``; the two scalars are bound
    LAST so the argument list is the certified one with an append, which is what
    lets a reader diff the two launches.
    """
    import numpy  # noqa: PLC0415

    targets = (("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz"))
    sources = (("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Hx", "Hy", "Hz"))
    volumes = [getattr(fields, name) for name in targets]
    auxiliaries = [getattr(fields, "fu_" + name) for name in targets]
    operands = [getattr(fields, name) for name in sources]
    nx, ny, nz = volumes[0].shape
    blocks = (nx * ny * nz + _THREADS - 1) // _THREADS
    plus, minus = beta_coefficients
    _get_kernel(sub_step)((blocks,), (_THREADS,), (
        volumes[0], volumes[1], volumes[2],
        auxiliaries[0], auxiliaries[1], auxiliaries[2],
        operands[0], operands[1], operands[2],
        numpy.int32(nx), numpy.int32(ny), numpy.int32(nz), numpy.float32(dtdx),
        tables["kms_x"], tables["sinv_x"],
        tables["kms_y"], tables["sinv_y"],
        tables["kms_z"], tables["sinv_z"],
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
        numpy.float32(plus), numpy.float32(minus),
    ))


class _BetaFreeGrid:
    """The run's grid, answering ``beta = 0.0`` and forwarding everything else.

    Satisfying AT THE INPUT the single clause this family inverts is the technique
    ``no_pml_curl._ActiveLayerProxy`` uses and the coverage census uses for the
    CuPy-backend clause, and for the same reason: the shipped predicates
    SHORT-CIRCUIT, so there is no accumulated refusal list to filter afterwards.

    ``grid.beta`` is read in exactly one place in each shipped predicate --
    ``coverage._grid_facts`` (:430), consumed at :723 and :1051 -- so the proxy
    changes the answer to that clause and to nothing else.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)

    @property
    def beta(self) -> float:
        return 0.0

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), item)


def _beta_reasons(fields: Any, grid: Any) -> Any:
    """The clauses THIS family owns, in place of the one it inverts. None if clear.

    Three, and each is a wrong answer rather than a crash if it is dropped:

    1. **beta must be NONZERO.** A beta = 0 run belongs to the certified pair, and
       the array path never enters the term (stepping.py:384/:467). Admitting one
       here would put two families on one slot, which the union census reports as a
       FINDING rather than as extra coverage.
    2. **real storage.** A complex-storage beta run takes the ``+-1j`` branch at
       stepping.py:798-799 and a different kernel; the certified predicate refuses
       complex storage anyway, so this clause is belt and braces WITH a reason --
       it names the family that owns the configuration instead of describing the
       storage.
    3. **no off-diagonal epsilon.** ``_special_kz_beta_term`` RAISES on it in real
       storage (stepping.py:800-810) and MEEP aborts on the same combination
       (fields.cpp:548-549). The refusal is FAMILY-WIDE, not E-side only: the run
       dies inside the first beta CURL, so there is no such run to serve at any
       sub-step.

    ``grid.dimensions != 2`` and ``cylindrical`` are refused here too, RESTATED
    rather than inferred from ``Grid._resolve_beta``'s constructor guard: a
    predicate that reasons "the constructor would have refused it" is a predicate
    that admits by argument from absence, and the cylindrical clause in particular
    is one the certified predicate happens to carry and this one must not rely on
    it carrying.
    """
    try:
        beta = float(getattr(grid, "beta", 0.0))
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return (f"the grid could not be asked for beta: {type(exc).__name__}: {exc}")
    if beta == 0.0:
        return ("grid.beta is zero: this pair exists only for special_kz runs, and "
                "a beta = 0 run belongs to the certified step_*_pml_real")
    if getattr(fields, "force_complex_fields", False):
        return ("complex64 storage with beta: the +-1j coefficient branch "
                "(stepping.py:798-799) is the complex family's, not this pair's")
    if getattr(fields, "has_offdiagonal_epsilon", False):
        return ("off-diagonal epsilon with beta in REAL storage: "
                "stepping._special_kz_beta_term raises (stepping.py:800-810) and "
                "MEEP aborts (fields.cpp:548-549) -- the implicit i on the TM half "
                "cancels only while mu and epsilon leave TE and TM uncoupled, so "
                "there is no such run to step")
    if bool(getattr(grid, "cylindrical", False)):
        return ("cylindrical (Dcyl) coordinates with beta: MEEP aborts "
                "(fields.cpp:546-547) and Grid._resolve_beta refuses the pairing "
                "at construction (grid.py:668-697)")
    try:
        dimensions = int(getattr(grid, "dimensions", 0))
    except Exception as exc:  # noqa: BLE001
        return (f"the grid could not be asked for dimensions: "
                f"{type(exc).__name__}: {exc}")
    if dimensions != 2:
        return (f"grid dimensions={dimensions} is not the effective-2-D grid beta "
                f"requires (grid.py:668-697; MEEP fields.cpp:546-547, \"Nonzero "
                f"beta unsupported in dimensions other than 2\")")
    return None


def covers_special_kz_curl(fields: Any, pml: Any, grid: Any, sub_step: str) -> tuple:
    """Whether ``step_{B,D}_special_kz_real`` may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. Returns ``(covered, reason)``.

    THE SUB-STEP ARGUMENT IS REQUIRED for the reason
    :func:`coverage.covers_real_pml_curl`'s is: the conductivity is read PER TARGET
    COMPONENT in ``stepping._apply_curl`` (:479), so a D-side conductivity routes
    ``step_D`` to a recurrence this pair does not implement and leaves ``step_B``
    an ordinary curl.

    DELEGATION, NOT A SECOND CLAUSE SET. Every question except the beta clause is
    the CERTIFIED curl predicate's, asked through it on a proxy grid whose beta
    reads zero, so this family cannot drift to a weaker standard than the one it
    borrows its arithmetic and its masks from -- and a clause added there is
    inherited here rather than silently skipped. The FOLD in particular is admitted
    exactly where the certified predicate admits it, through the same
    ``real_curl_boundary_codes`` split, because the beta term reuses a CENTRE
    register the fold never touches. That composition is the gate's folded arm, not
    an argument made here.
    """
    if sub_step not in ("step_B", "step_D"):
        raise ValueError(
            f"sub_step must be 'step_B' or 'step_D', got {sub_step!r}")
    own = _beta_reasons(fields, grid)
    if own is not None:
        return False, own
    return _coverage.covers_real_pml_curl(fields, pml, _BetaFreeGrid(grid), sub_step)


def covers_special_kz_constitutive(fields: Any, pml: Any, grid: Any,
                                   side: str) -> tuple:
    """Whether the CERTIFIED constitutive pair may serve ``update_H``/``update_E``
    on a beta run. ``side`` is ``"H"`` or ``"E"``; returns ``(covered, reason)``.

    THIS FAMILY BUILDS NO CONSTITUTIVE KERNEL, and that is the finding rather than
    an omission. ``stepping.update_H`` (:907-923) and ``update_E`` (:926-993) reach
    ``_apply_constitutive_pml`` (:2065) and neither they nor it read ``grid.beta``,
    ``bfast_scaled_k`` or any array the beta term writes: beta enters the step loop
    at stepping.py:384-391 and :467-474 and nowhere else. So the admission is the
    shipped predicate's with the beta clause inverted, and the ARITHMETIC is
    ``constitutive_kernels``' own, untouched.

    THAT READING IS NOT THE EVIDENCE. ``gate_cuda_special_kz.py`` carries a
    constitutive arm that runs the certified pair against ``stepping.update_H`` /
    ``update_E`` on a grid whose beta is nonzero, at both sub-steps and both
    subnormal policies, and a beta run whose two curls have already moved the
    state. A sub-step admitted because nothing was found to read beta, with no
    device leg, would be admitted by argument from absence -- which is what the
    Metal track's tranche-6 split (that family's two remaining open slots were
    exactly one row's constitutive sides, while its two curls were served) exists
    to warn about. The two sub-steps are asked SEPARATELY here for the same reason.
    """
    if side not in ("H", "E"):
        raise ValueError(f"side must be 'H' or 'E', got {side!r}")
    own = _beta_reasons(fields, grid)
    if own is not None:
        return False, own
    return _coverage.covers_real_pml_constitutive(
        fields, pml, _BetaFreeGrid(grid), side)
