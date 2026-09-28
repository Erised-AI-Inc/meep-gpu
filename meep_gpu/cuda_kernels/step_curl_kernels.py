"""Raw-CUDA fast-path kernels: the fused alternative to the array path in stepping.py.
step_curl.py - STRICT line-by-line translation of MEEP's curl updates

MEEP source files:
- step_generic.cpp: step_curl function
- step_db.cpp: fields_chunk::step_db() function
- step_generic.cpp: step_update_EDHB function

=============================================================================
MEEP CURL FORMULA (step_generic.cpp):
=============================================================================

    f[i] -= dtdx * (g1[i + s1] - g1[i] + g2[i] - g2[i + s2])

MEEP iterates over ALL owned cells (PLOOP_OVER_IVECS from is to ie).
At grid boundaries, g1[i+s1] or g2[i+s2] accesses the ghost zone.

=============================================================================
STRIDE NEGATION FOR D-FIELDS (step_db.cpp):
=============================================================================

    if (ft == D_stuff) {
        stride_p = -stride_p;
        stride_m = -stride_m;
    }

For B-field updates: use forward differences (positive strides)
For D-field updates: use backward differences (negated strides)

=============================================================================
BOUNDARY CONDITIONS:
=============================================================================

MEEP handles boundary conditions through step_boundaries() calls between
each sub-step. For periodic BC (k_point or force_complex_fields), values
wrap around from one side to the other. For metallic BC, ghost zones are
zero.

The plain kernels wrap neighbor access with modular indexing (`(i+1) % n`),
the semantics of cp.roll() in the array path. This matches MEEP's behavior
when periodic BC are active (e.g. when force_complex_fields=True with default
k_point=0). NOTE the PML kernels instead clamp the outer X/Y faces
metallically — WRONG against MEEP's wrap, measured at 6.31e-04 vs 4.00e-07
relative L2 on the uniform-PML sheet case (stepping.py:110-117); reworking
that to the wrap is Phase 3 of the revival plan.

=============================================================================
MIRROR SYMMETRY BOUNDARY CONDITIONS:
=============================================================================

When X-Y mirror symmetry is active, we only compute the positive quadrant.
At the symmetry boundary (i=0 or j=0), the "ghost cell" at i=-1 is:

    ghost[-1] = phase * f[1]

where phase is +1 or -1 depending on the field component's symmetry.

For the derivative at i=0:
    Forward difference: f[0] - f[-1] = f[0] - phase * f[1]
    Backward difference: f[1] - f[0] (normal, no ghost needed)

MEEP source: symmetry.cpp

=============================================================================
GPU KERNEL FUSION:
=============================================================================

Each kernel fuses several CuPy array operations into one launch to cut
kernel-launch overhead:

- Fused triple-curl: 21 operations → 1 kernel for all 3 components
- Fused curl + PML recurrence: ~42 operations → 1 kernel
- Fused constitutive PML update: ~18 operations → 1 kernel

The original bundle's own header estimated the fusion at 84 kernels/step →
8-12 kernels/step (~1.5-1.9x) — an unverified prior until measured through the
benchmark harness with the bit-identity gate passed.

This file holds ONLY the kernel strings, their launch wrappers, and the compile
cache. The dispatch decision lives in ``meep_gpu/fastpath.py`` (which imports
this module lazily, and only on a CuPy backend); the array path in
``../stepping.py`` is the reference and the correctness oracle. The in-file
fallback stepping implementation this file used to carry — a second, drifted
copy of what ``stepping.py`` now is — was deleted outright, with the
never-reachable generic kernels, in Phase 0 of
``the design notes (fdtd-fused-cuda-kernel-plan)``.
"""

import cupy as cp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..fields import Fields
    from ..pml import PML

# The coverage predicate lives in a CUPY-FREE sibling, ``coverage.py``, so it can
# be imported, exercised and mutated on a machine with no GPU — the reason is
# written out in that module's header. It is imported back here because this
# module is the slice's public face: ``kernels.covers_real_pml_curl`` is what the
# lifted-case validator calls, and the launch wrappers below need ``BC_CODES``.
try:
    from .coverage import (BC_CODES, REAL_CURL_BC_CODES, covers_real_pml_curl,
                           real_pml_boundary_kinds)
    from .coverage import real_curl_boundary_codes as real_curl_boundary_codes_from_grid
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    _coverage_spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_coverage",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "coverage.py"))
    _coverage = _importlib_util.module_from_spec(_coverage_spec)
    _coverage_spec.loader.exec_module(_coverage)
    BC_CODES = _coverage.BC_CODES
    REAL_CURL_BC_CODES = _coverage.REAL_CURL_BC_CODES
    covers_real_pml_curl = _coverage.covers_real_pml_curl
    real_pml_boundary_kinds = _coverage.real_pml_boundary_kinds
    real_curl_boundary_codes_from_grid = _coverage.real_curl_boundary_codes

# The compile memo is a CUPY-FREE sibling too, for the same reason and by the
# same by-path fallback: what it keys on decides WHICH BINARY a caller is handed,
# and that decision has to be exercisable where there is no device. See
# ``compile_cache.py``'s header for why the policy lives in the KEY rather than
# in a clear-on-install hook.
try:
    from . import compile_cache
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    _cache_spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_compile_cache",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "compile_cache.py"))
    compile_cache = _importlib_util.module_from_spec(_cache_spec)
    _cache_spec.loader.exec_module(compile_cache)

# =============================================================================
# CERTIFIED, AND DEAD — WHICH OF THE FOURTEEN KERNELS BELOW MAY BE WIRED
# =============================================================================
#
# READ THIS BEFORE WIRING ANYTHING FROM THIS FILE. Two of the fourteen kernels
# are certified. The other twelve are not, and the recommendation of record
# (``the design notes (cuda-kernel-track-disposition)`` §3) is that they should not
# be repaired, not be wired, and not be counted as coverage. They are kept, not
# deleted, and this block is why the decision is visible here rather than only
# inside a 2026-08-04 JSON.
#
# THE ARITHMETIC ON THE FOUR PLAIN KERNELS, read first-hand out of
# ``parity/meep_gpu/results/fused_kernel_bit_identity.json`` (2026-08-04, 64
# cases: 4 option sets x {step_B, step_D} x {float32, complex64} x 2 shapes x 2
# dtdx): ``vs_array_order`` is FALSE on all 64 — 0/16 per option set, INCLUDING
# under the shipped ``--fmad=false``. They reach 16/16 only against a reference
# recomputed in the kernel's own left-to-right grouping (``vs_kernel_order``).
# Kernels 5-12 have never been compared to the array path at any tolerance: that
# record's ``bit_identity`` block contains only ``step_B``/``step_D``.
#
# WHAT REPAIRING THEM WOULD BUY. The union of every corpus row the twelve could
# EVER serve — if all six documented defects in README.md were fixed and all
# twelve were certified — is 26 rows of the 186 engine-accepted lifted rows, and
# a certified Triton family already covers every one of the 26. That ceiling is
# derived in the disposition §3.1 from
# ``parity/meep_gpu/results/predicate_coverage_2026-08-14_allnine/``; the covering
# family is named per kernel below. The cost of the alternative is twelve
# predicates, twelve gates and twelve mutation batteries, every one device-only.
#
# None of the twelve has a coverage predicate, so there is no fail-closed refusal
# written anywhere for the configurations they would silently mis-serve. That is
# the operative reason they cannot simply be switched on.
#
# These two tuples are the file's half of a partition that
# ``test_certification_record.py`` pins against ``certification.json``: the
# record names exactly the certified kernels, every other kernel shipped in this
# file carries an entry here, and the two sets together are all fourteen. A
# kernel added without a gate verdict fails that test rather than shipping
# unmeasured.

#: Certified byte-identical to the array path — ``certification.json`` holds the
#: gate verdict, its shapes and its limits.
CERTIFIED_KERNELS = ("step_B_pml_real", "step_D_pml_real")

#: Every other kernel in this file, mapped to the certified Triton family that
#: already covers the corpus surface it targets. Being on this list means: not
#: certified against the array path, no coverage predicate, not to be wired.
UNCERTIFIED_KERNELS = {}

# =============================================================================
# FUSED CUDA KERNELS
# =============================================================================





# =============================================================================
# SYMMETRY-AWARE FUSED KERNELS
# =============================================================================
#
# These kernels handle X-Y mirror symmetry boundary conditions:
# - Forward diff (shift=-1): At far boundary, use metallic BC (0)
# - Backward diff (shift=+1): At near boundary, use mirror BC (phase * f[mirror_idx])
# - After computing curl, zero out non-owned cells based on target iyee_shift
#
# For step_B (forward diff):
#   Bx: dEz/dy - dEy/dz, iyee_Bx=(0,1,1), phases: Ez_y, 1
#   By: dEx/dz - dEz/dx, iyee_By=(1,0,1), phases: 1, Ez_x
#   Bz: dEy/dx - dEx/dy, iyee_Bz=(1,1,0), phases: Ey_x, Ex_y
#
# For step_D (backward diff):
#   Dx: dHz/dy - dHy/dz, iyee_Dx=(1,0,0), phases: Hz_y, 1
#   Dy: dHx/dz - dHz/dx, iyee_Dy=(0,1,0), phases: 1, Hz_x
#   Dz: dHy/dx - dHx/dy, iyee_Dz=(0,0,1), phases: Hy_x, Hx_y



# =============================================================================
# FUSED PML KERNELS
# =============================================================================
#
# These kernels combine curl computation + PML update into single kernel launches.
# Each component uses different PML coefficient axes:
#
# For step_B (forward diff):
#   Bx: dsig=Y, dsigu=Z → kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]
#   By: dsig=Z, dsigu=X → kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]
#   Bz: dsig=X, dsigu=Y → kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]
#
# For step_D (backward diff):
#   Dx: dsig=Y, dsigu=Z → kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]
#   Dy: dsig=Z, dsigu=X → kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]
#   Dz: dsig=X, dsigu=Y → kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]
#
# PML update formula (step_generic.cpp):
#   fprev = fu[idx]
#   fu[idx] = (kms * fu[idx] - curl) * sinv
#   field[idx] = sinv_u * (kms_u * field[idx] + fu[idx] - fprev)



# =============================================================================
# SYMMETRY-AWARE FUSED PML KERNELS
# =============================================================================
#
# These kernels combine symmetry BC handling + curl + PML update in single kernel.
# They handle:
# - Metallic BC (zero) at far boundary for forward diff (step_B)
# - Mirror BC (phase * f[mirror_idx]) at near boundary for backward diff (step_D)
# - Cell ownership masking based on iyee_shift
# - PML absorption formula: fu_new = (kms * fu - curl) * sinv
#                           B = sinv_u * (kms_u * B + fu_new - fprev)
#
# For step_B with symmetry:
#   Bx: dsig=Y, dsigu=Z, iyee_Bx=(0,1,1) -> mask i=0 if sym_x
#   By: dsig=Z, dsigu=X, iyee_By=(1,0,1) -> mask j=0 if sym_y
#   Bz: dsig=X, dsigu=Y, iyee_Bz=(1,1,0) -> no masking
#
# For step_D with symmetry:
#   Dx: dsig=Y, dsigu=Z, iyee_Dx=(1,0,0) -> mask j=0 if sym_y
#   Dy: dsig=Z, dsigu=X, iyee_Dy=(0,1,0) -> mask i=0 if sym_x
#   Dz: dsig=X, dsigu=Y, iyee_Dz=(0,0,1) -> mask i=0 or j=0 if sym





# Kernels compile lazily on first use and are memoized in ``compile_cache``,
# keyed on (name, is_complex, options, subnormal policy, source) — NOT on
# (name, is_complex), which is what it was until 2026-08-15 and which served a
# kernel compiled under one float32 subnormal policy to a caller running under
# another. That module's header carries the reasoning, including the half of the
# problem a memo key CANNOT fix: CuPy's own on-disk cache key is computed above
# the seam ``subnormal_policy`` strips ``-ftz=true`` at, so a gate runner must
# also take a fresh policy-token-carrying ``CUPY_CACHE_DIR`` per leg.

# NVRTC compile options, pinned here beside the code map because they are part of
# these kernels' CORRECTNESS, not their performance.
#
# NVRTC contracts ``a*b + c`` into a fused multiply-add by default, which rounds
# once where the array path in ``../stepping.py`` rounds twice. Measured on an RTX
# A6000 (CuPy 13.5.1, NVRTC 11.6, ``parity/meep_gpu/probe_fused_kernel_bit_identity.py``
# and ``probe_fused_kernel_grouping.py``, results committed beside them):
#
#   * Default options emit ``mul.f32``/``add.f32``/``sub.f32`` — contractible by
#     ptxas — plus 3 ``fma.rn.f32`` NVRTC contracts itself in each complex curl
#     kernel. With ``--fmad=false`` every f32 op carries ``.rn``, which the PTX
#     ISA defines as non-contractible, and the ``fma.rn.f32`` count is 0.
#   * The arithmetic consequence, AND IT IS CONDITIONAL — read the condition, it
#     is what the claim rests on. ``--fmad=false`` reaches EXACT bit-identity
#     against the array path ONLY FOR A KERNEL WHOSE STENCIL IS PARENTHESIZED TO
#     STEPPING.PY'S TREE. Exactly two of the fourteen kernels below are so
#     parenthesized — ``step_B_pml_real`` and ``step_D_pml_real``
#     (numbered note 3, "THE STENCIL IS PARENTHESISED TO
#     ``stepping._curl_from_operands``", in the real-field PML section header
#     below, for why) — and they are the two in ``CERTIFIED_KERNELS``.
#     For them the 2026-08-09/10 gate measured 120/120 identical under
#     ``--fmad=false`` and 0/120 under the default options.
#     THE FOUR PLAIN KERNELS ARE NOT PARENTHESIZED, AND THEY DO NOT REACH IT:
#     measured 0/16 against the array path under ``--fmad=false`` and 0/16 under
#     every other guard tried (``results/fused_kernel_bit_identity.json``,
#     2026-08-04, ``vs_array_order`` false on all 64 cases). The 16/16 that record
#     also carries is against a DIFFERENT reference — one recomputed in the
#     kernel's own left-to-right grouping (``vs_kernel_order``) — which measures
#     the contraction guard, not agreement with the oracle. Reading either "16/16"
#     as bit-identity against the array path reads a pass where the artifact
#     records a fail.
#   * A dtdx that is a power of two hides the whole effect (0.5 scaling is exact,
#     so contracted and uncontracted round identically); the divergence only shows
#     with a dtdx that is not exactly representable, which is the realistic case.
#     Both dtdx values are swept in the 2026-08-04 record above (0.5 and 0.35);
#     the 2026-08-09/10 PML gate sweeps both at synthetic shapes, but its six
#     real-corpus lifted cases ran at 0.5 ONLY, so the guard has never been
#     exercised at corpus scale (disposition §1.4).
#
# Both ``--fmad=false`` and ``-fmad=false`` are accepted by NVRTC 11.6; all
# fourteen registered kernels compile with either (``grep -c 'extern "C"
# __global__'`` over this file returns 14; the "twelve" this line used to say
# predates the two real-field PML kernels added 2026-08-10).
_COMPILE_OPTIONS = ('--fmad=false',)


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    Kept because the bit-identity probe drives it between guard sets, where the
    compiler ITSELF is substituted and the option tuple this module passes is
    overridden from outside — a change the memo key cannot see. Nothing else
    depends on it being called: the policy, the options and the source are all IN
    the key, so a compile taken under different conditions misses rather than
    hits.
    """
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str, is_complex: bool = False):
    """Get compiled kernel, compiling on first use.

    The memo key carries the subnormal policy in force and the source string
    actually compiled, so neither a late policy install nor a mutated kernel can
    be served an earlier binary (``compile_cache.py``).

    THE CODE MAP IS REBUILT PER CALL AND THAT IS LOAD-BEARING, not an oversight —
    it was inside the cache-miss branch until 2026-08-15, and it left because the
    SOURCE is now part of the memo key, so the map has to be read before a key
    exists to miss on. Rebuilding it is what makes a REWRITTEN module-level
    source string visible: ``probe_fused_kernel_bit_identity`` mutates kernels by
    assigning over ``module._step_B_pml_real_kernel_code`` and re-launching,
    and a map memoized at first call would hand back the pre-mutation string
    forever — a leg reporting a pass for a mutation it never applied, which is
    the defect the sibling track hit three times and the reason the source is in
    the key at all. ``test_compile_cache.py`` pins it; a memo here would need
    every mutation site in two probes to remember to drop it.

    THE COST, MEASURED RATHER THAN ASSUMED (laptop, 200 000 iterations each):
    the 14-entry map literal is 0.784 us, ``compile_policy_token()`` 0.71 us
    uninstalled / 0.74 us installed, the key build plus memo hit 0.34 us — about
    1.9 us against the 0.042 us of the pre-2026-08-15 ``(name, is_complex)``
    lookup. Two calls per timestep, against ~5.7 ms per timestep at 160³ and the
    measured 716.7 Mcells/s array path: 6.7e-4 of one step. Caching the token per
    process was considered and REFUSED for the same reason as the map — a token
    frozen at first launch cannot see a later policy install, which is precisely
    the staleness this key exists to remove.
    """
    code_map = {
        ('step_B_pml_real', False): (_step_B_pml_real_kernel_code, 'step_B_pml_real'),
        ('step_D_pml_real', False): (_step_D_pml_real_kernel_code, 'step_D_pml_real'),
    }
    code, func_name = code_map[(name, is_complex)]
    key = compile_cache.kernel_cache_key(name, is_complex, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, func_name, options=_COMPILE_OPTIONS))






















# =============================================================================
# REAL-FIELD PML CURL — the configuration 54 of the 57 corpus scripts declare
# =============================================================================
#
# Every _pml kernel above is complex-only, so the single most common real
# configuration — real storage with an absorbing layer — had no kernel at all.
# These two are that kernel. They replace exactly ``stepping.step_B`` and
# ``stepping.step_D`` and nothing else; source injection, zero_metal_*, the
# symmetry fills, update_H/update_E and update_P keep running on the array path.
#
# FOUR THINGS DIFFER FROM THE COMPLEX KERNELS ABOVE, and each of them is a
# wrong answer rather than a crash if it is dropped:
#
# 1. THE BOUNDARY KIND IS AN ARGUMENT, NOT A LAYOUT. The complex kernels clamp
#    the outer X/Y faces metallically and wrap Z. That is a hard-coded guess:
#    ``stepping._boundary_kinds`` resolves each axis independently, and an axis
#    that merely absorbs still WRAPS (an absorber grades a conductivity on top
#    of the boundary; it does not become a mirror). A kernel that assumes the
#    complex layout is bit-identical on a 2-D XY-PML sheet, where the guess
#    happens to coincide, and silently wrong on a 3-D run and on every metallic
#    Z. ``bc_x/bc_y/bc_z`` carry the resolved kind per axis instead.
# 2. THE METALLIC CURL MASK (``stepping._mask_non_owned_cells``). Cell 0 of a
#    component whose Yee shift is 0 on a metallic axis sits ON the wall, which
#    MEEP never steps. The curl is zeroed there and the recurrence still runs,
#    so the auxiliary integrates a zero and decays instead of accumulating a
#    curl assembled from a ghost the cell does not own. ``iyee`` is a
#    compile-time constant per component, so the mask is one comparison.
# 3. THE FOLDED-PERIODIC TOP-PLANE MASK, the second half of that same function
#    and the one this pair carried for its first nine days without. It is
#    numbered separately because it masks a DIFFERENT plane for a DIFFERENT
#    reason and the two never coincide — see the block below note 4.
# 4. THE STENCIL IS PARENTHESISED TO ``stepping._curl_from_operands``:
#
#        dtdx * ((sf - f1) + (f2 - ss))
#
#    NOT ``dtdx * (sf - f1 + f2 - ss)``, which C associates left-to-right as
#    ``((sf - f1) + f2) - ss``. Float addition is not associative and the two
#    differ in the last bits — measured, and the reason the bit-identity probe
#    carries both groupings as separate references. The complex kernels above
#    are written in the left-to-right form; these are not.
#
# ``_COMPILE_OPTIONS``' ``--fmad=false`` is load-bearing here for the same
# reason it is above, and for one more: ``(fu*kms) - curl`` and
# ``(f*kms_u) + fu_new`` are both FMA candidates the array path rounds twice.
#
# =============================================================================
# A MIRROR FOLD — THE THIRD BOUNDARY CODE, AND WHY IT IS ONLY A MASK
# =============================================================================
#
# ``stepping._boundary_kinds`` (stepping.py:2192-2194) resolves a folded axis to
# MIRROR, ahead of its declared condition. A fold changes TWO things a curl
# depends on — the ghost rule and the ownership mask — so it was refused outright
# until each half was measured. What the two 2026-08-19/20 device gates found is
# that only one half reaches an output word:
#
# THE GHOST VALUES ARE DEAD. ``stepping._shift_down``'s MIRROR branch writes
# ``parity * field[2]`` into stored cell 0 (stepping.py:1870-1873,
# ``MIRROR_SOURCE_INDEX = 2``) and ``_shift_up``'s writes
# ``parity * field[reflect_row]`` into the last stored slot of a folded PERIODIC
# axis (:1772-1780), or an exact zero on a folded METALLIC one (:1781-1783).
# Each ghost has exactly ONE consumer plane and BOTH consumer planes are dropped
# by ``_mask_non_owned_cells`` (:1865-1900): ``_shift_down`` is the D sub-step's
# and its ghost lands at stored cell 0, whose target has Yee shift 0 there — the
# cell the mirrored branch of the mask drops; ``_shift_up`` is the B sub-step's
# and its ghost lands at the last stored slot, whose target has Yee shift 1
# there — the cell the fold's own mask drops. So the parity arithmetic never
# reaches a surviving word, and ``BC_MIRROR_PERIODIC`` takes the METALLIC zero
# for both faces rather than carrying a parity this kernel has no per-component
# constant for. THE ZERO IS NOT THE BOUNDARY CONDITION HERE; the mask is, and
# the two mutations ``drop_folded_periodic_near_mask`` /
# ``drop_folded_periodic_top_mask`` are what stop that from being an assertion.
#
# THE MASK IS NOT DEAD, AND IT SPLITS BY TERMINATION.
# ``_mask_non_owned_cells`` zeroes cell 0 on a mirrored axis for every component
# whose Yee shift there is 0 — which a folded METALLIC axis already inherits from
# ``BC_METALLIC``, arithmetic identical, and which a folded PERIODIC axis needs
# stated because it is not metallic. And, ONLY where ``_stored_past_owned``
# (stepping.py:1503-1518) is true — a folded PERIODIC axis, at either full-count
# parity, whose stored array carries MEEP's not-owned allocation slot past
# ``big_corner`` — it also zeroes the LAST stored slot for every component whose
# Yee shift there is 1. On a folded METALLIC axis that top plane IS stepped and
# must NOT be masked.
#
# THAT SPLIT WAS MEASURED, NOT ARGUED (``gate_cuda_folded_curl.py``,
# results/cuda_folded_curl_2026-08-19/, the GPU host, 256 cases per policy): with the
# folded axis handed ``BC_METALLIC``, a folded METALLIC axis was ALREADY exact at
# 48/48 single-launch and 48/48 at 60 launches under both float32 subnormal
# policies, and a folded PERIODIC axis diverged 0/64 with 19,649 differing words
# of which 19,649 lay on the mask-delta plane and ZERO anywhere else — always the
# same entry, ``(folded axis, -1)``, oracle-masked and kernel-kept. The
# top-plane branch below is that specification, and nothing else. DO NOT
# GENERALISE A FOLD VERDICT ACROSS FAMILIES: ``covers_real_pml_constitutive``
# took the fold for free because its sub-step reads its own cell; a curl reads
# NEIGHBOURS, and ``covers_real_pml_offdiag_constitutive`` still refuses a fold
# with no device evidence of its own.
#
# Ghost rules, transcribed from ``stepping._shift_up`` / ``_shift_down``. Only
# PERIODIC rolls; METALLIC and the folded-PERIODIC code take the zero face, and
# CYL_AXIS has no code at all (refused by :func:`covers_real_pml_curl`). Real
# storage cannot carry a Bloch phase, so ``phases`` is (None, None, None) by
# construction in this slice:
#
#   up   PERIODIC: g[i+1 == n ? 0   : i+1]     otherwise: i+1 >= n ? 0 : g[i+1]
#   down PERIODIC: g[i   == 0 ? n-1 : i-1]     otherwise: i   == 0 ? 0 : g[i-1]
#
# An invariant axis (n == 1, always PERIODIC) needs no case of its own: the
# wrap returns the same cell and the difference is an exact zero, which is what
# MEEP's ``stride(d) = 0`` computes for a direction it does not have.

_REAL_PML_PRELUDE = r'''
#define BC_PERIODIC 0
#define BC_METALLIC 1
// A MIRROR-FOLDED axis whose termination is PERIODIC -- stepping._boundary_kinds
// resolves it to "mirror" and stepping._stored_past_owned says the stored array
// carries one slot past MEEP's owned window. A folded METALLIC axis is handed
// BC_METALLIC instead and has no code of its own: its ghost rule, its cell-0 mask
// and its (absent) top-plane mask are METALLIC's, arithmetic for arithmetic, and
// that coincidence was measured 48/48 rather than assumed.
#define BC_MIRROR_PERIODIC 2

// stepping._shift_up: the neighbour one cell up, with that axis's far-face rule.
__device__ __forceinline__ float shift_up(
    const float* __restrict__ g, int idx, int ia, int na, int stride, int bc
) {
    if (ia + 1 < na) return g[idx + stride];
    return (bc == BC_PERIODIC) ? g[idx - ia * stride] : 0.0f;
}

// stepping._shift_down: the neighbour one cell below, with that axis's near-face rule.
__device__ __forceinline__ float shift_dn(
    const float* __restrict__ g, int idx, int ia, int na, int stride, int bc
) {
    if (ia > 0) return g[idx - stride];
    return (bc == BC_PERIODIC) ? g[idx + (na - 1) * stride] : 0.0f;
}

// stepping._apply_pml_update, in its in-place order:
//   fu *= kms; fu -= curl; fu *= sinv;
//   f *= kms_u; f += fu; f -= fu_previous; f *= sinv_u
__device__ __forceinline__ void pml_apply(
    float* __restrict__ f, float* __restrict__ fu, int idx, float curl,
    float kms, float sinv, float kms_u, float sinv_u
) {
    float fprev = fu[idx];
    float fu_new = ((fprev * kms) - curl) * sinv;
    fu[idx] = fu_new;
    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;
}
'''

# B side: forward differences, half-integer PML coefficients, iyee from
# fields.IYEE_SHIFTS — Bx (0,1,1), By (1,0,1), Bz (1,1,0). Only the axis whose
# shift is 0 can carry the wall mask, so each component masks cell 0 on one axis
# — and, on a folded PERIODIC axis, the last stored slot on the OTHER TWO, where
# its shift is 1. The two masks are per-axis complements: no component ever
# carries both on the same axis, which is why the branch is written as two
# statements rather than one clamped index.
_step_B_pml_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void step_B_pml_real(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
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

    // Bx: curl_x = dEz/dy - dEy/dz; dsig=y, dsigu=z; iyee=(0,1,1) -> cell 0 on x,
    // and on a folded PERIODIC axis the top plane on y and z.
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
        pml_apply(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // By: curl_y = dEx/dz - dEz/dx; dsig=z, dsigu=x; iyee=(1,0,1) -> cell 0 on y,
    // top plane on z and x.
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
        pml_apply(By, fu_By, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Bz: curl_z = dEy/dx - dEx/dy; dsig=x, dsigu=y; iyee=(1,1,0) -> cell 0 on z,
    // top plane on x and y.
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
'''

# D side: MEEP's negated strides (backward differences), integer-position PML
# coefficients, iyee — Dx (1,0,0), Dy (0,1,0), Dz (0,0,1). Two axes carry a
# zero shift on each component here, so each masks cell 0 on two — and, on a
# folded PERIODIC axis, the last stored slot on the ONE remaining axis, which is
# the axis this component's stencil never differences along. That the top-plane
# mask fires on an axis the component reads no neighbour on is not a slip: the
# mask is about which cells MEEP's owned loop VISITS, not about where the
# stencil reaches (stepping._mask_non_owned_cells is per-axis, per-component).
_step_D_pml_real_kernel_code = _REAL_PML_PRELUDE + r'''
extern "C" __global__ void step_D_pml_real(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
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

    // Dx: curl_x = dHz/dy - dHy/dz; dsig=y, dsigu=z; iyee=(1,0,0) -> cell 0 on y and
    // z, and on a folded PERIODIC axis the top plane on x -- an axis this stencil
    // never differences along.
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
        pml_apply(Dx, fu_Dx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // Dy: curl_y = dHx/dz - dHz/dx; dsig=z, dsigu=x; iyee=(0,1,0) -> cell 0 on x and
    // z, top plane on y.
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
        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Dz: curl_z = dHy/dx - dHx/dy; dsig=x, dsigu=y; iyee=(0,0,1) -> cell 0 on x and
    // y, top plane on z.
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
'''

# ``BC_CODES`` and ``REAL_CURL_BC_CODES`` — the boundary kinds this slice serves,
# as the kernels' integer codes — are imported from ``coverage.py`` at the top of
# this module, together with ``real_pml_boundary_kinds``,
# ``real_curl_boundary_codes`` and ``covers_real_pml_curl``.

_REAL_PML_THREADS = 256


def real_pml_curl_tables(pml: 'PML', half_integer: bool) -> dict:
    """Flatten one sub-step's six PML coefficient vectors into cached device views.

    Called ONCE per frozen configuration, not per launch. The complex wrappers
    above call ``cp.ascontiguousarray(...ravel())`` on every launch, which is six
    device allocations per sub-step and twelve per timestep for tables that never
    change (port-reference defect 5.14). ``PML._reshape_for_broadcast`` already
    stores float32 arrays of shape (n,1,1)/(1,n,1)/(1,1,n), so ``reshape(-1)`` is a
    view and this costs nothing to hold.

    ``half_integer`` selects the sub-lattice: the B curl reads the half-integer
    positions (``kms_x_h``) and the D curl the integer ones (``kms_x``). Getting
    that backwards is a half-cell error, not a crash.
    """
    suffix = "_h" if half_integer else ""
    tables = {}
    for axis in ("x", "y", "z"):
        for name in ("kms", "sinv"):
            attribute = f"{name}_{axis}{suffix}"
            flat = getattr(pml, attribute).reshape(-1)
            if flat.dtype != cp.float32:
                raise ValueError(
                    f"{attribute} is {flat.dtype}; the real-field PML kernels index "
                    f"float32 coefficient vectors.")
            if not flat.flags.c_contiguous:
                raise ValueError(
                    f"{attribute} did not flatten to a contiguous view; the kernel "
                    f"indexes it as a bare vector.")
            tables[f"{name}_{axis}"] = flat
    return tables


def real_curl_boundary_codes(grid) -> tuple:
    """The three integer codes this pair's kernels take, resolved off the GRID.

    IT TAKES THE GRID, NOT THE KIND STRINGS, and that is the whole reason the
    function was replaced rather than extended. ``stepping._boundary_kinds``
    resolves a folded axis to ``"mirror"`` at BOTH terminations (stepping.py:2193
    — the fold outranks the declaration), and the two terminations need DIFFERENT
    codes: a folded METALLIC axis is ``BC_METALLIC`` and a folded PERIODIC one is
    ``BC_MIRROR_PERIODIC``, because only the second stores the slot past MEEP's
    owned window that the top-plane mask drops. No function of the kind strings
    alone can tell them apart, so the previous ``real_pml_boundary_codes(kinds)``
    could not have expressed this admission — it raised on ``"mirror"``, which
    was correct while nothing served a fold and is a CRASHED RUN now that the
    predicate admits one.

    Delegates the whole split to :func:`coverage.real_curl_boundary_codes`, which
    is the copy ``covers_real_pml_curl`` itself consults, so the predicate and the
    launcher cannot answer differently about the same grid. RAISES on a grid the
    predicate refuses — callers gate on the predicate first, and a raise here
    means they did not.
    """
    codes, refusal = real_curl_boundary_codes_from_grid(grid)
    if refusal is not None:
        raise ValueError(
            f"this grid has no boundary-code triple for the real-field PML curl "
            f"kernels: {refusal}. Ask covers_real_pml_curl first; it refuses this "
            f"configuration rather than raising.")
    return tuple(np.int32(code) for code in codes)


def _step_fused_pml_real(kernel, targets, auxiliaries, sources,
                         tables: dict, boundary_codes, dtdx: float):
    """Shared launch for both sides: the argument list differs only in which arrays."""
    nx, ny, nz = targets[0].shape
    blocks = (nx * ny * nz + _REAL_PML_THREADS - 1) // _REAL_PML_THREADS
    kernel((blocks,), (_REAL_PML_THREADS,), (
        targets[0], targets[1], targets[2],
        auxiliaries[0], auxiliaries[1], auxiliaries[2],
        sources[0], sources[1], sources[2],
        np.int32(nx), np.int32(ny), np.int32(nz),
        np.float32(dtdx),
        tables["kms_x"], tables["sinv_x"],
        tables["kms_y"], tables["sinv_y"],
        tables["kms_z"], tables["sinv_z"],
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ))


def _step_B_fused_pml_real(fields: 'Fields', tables: dict, boundary_codes,
                           dtdx: float):
    """``stepping.step_B`` for real fields under a PML, in one launch.

    ``tables`` are the HALF-INTEGER coefficient views from
    :func:`real_pml_curl_tables` (``half_integer=True``); the sources are the
    STORED ``Ex/Ey/Ez``, which a PML run always has.
    """
    _step_fused_pml_real(
        _get_kernel('step_B_pml_real', False),
        (fields.Bx, fields.By, fields.Bz),
        (fields.fu_Bx, fields.fu_By, fields.fu_Bz),
        (fields.Ex, fields.Ey, fields.Ez),
        tables, boundary_codes, dtdx)


def _step_D_fused_pml_real(fields: 'Fields', tables: dict, boundary_codes,
                           dtdx: float):
    """``stepping.step_D`` for real fields under a PML, in one launch.

    ``tables`` are the INTEGER coefficient views from
    :func:`real_pml_curl_tables` (``half_integer=False``).
    """
    _step_fused_pml_real(
        _get_kernel('step_D_pml_real', False),
        (fields.Dx, fields.Dy, fields.Dz),
        (fields.fu_Dx, fields.fu_Dy, fields.fu_Dz),
        (fields.Hx, fields.Hy, fields.Hz),
        tables, boundary_codes, dtdx)


# =============================================================================
# COVERAGE PREDICATE — MOVED
# =============================================================================
#
# ``covers_real_pml_curl`` and its helpers now live in ``coverage.py``, which
# imports ``__future__`` and ``typing`` and nothing else, and are imported back
# at the top of this module so every existing caller keeps working. The move is
# the point: this module imports ``cupy`` at scope, so while the predicate lived
# here it could not be evaluated on a machine without a GPU and its whole test
# file collapsed to one sanctioned skip. The refusal reasoning, and why every
# entry in it is a silent wrong answer rather than a crash, is written out in
# that module's header.
