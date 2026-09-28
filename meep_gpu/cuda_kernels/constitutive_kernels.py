"""Raw-CUDA real-storage constitutive kernels: ``update_H`` and ``update_E``.

THE ZERO-ROW SURFACE. Before this file the hand-CUDA track covered 84 of 759
predicate slots and **zero rows at every sub-step**, for one reason: no
real-storage constitutive kernel existed. ``update_H``, ``update_E`` and
``update_P`` all sat at zero rows, so every row the certified curl pair admits
still had to hand two of its four sub-steps back to the array path. These two
kernels are the first half of that gap — the ordinary real-storage PML case,
which is what 54 of the 57 corpus scripts declare.

WHY A NEW MODULE AND NOT ``step_curl_kernels.py``. That file's fourteen device
strings are frozen: ``certification.json`` pins each certified string by digest,
pins all fourteen concatenated, and pins the whole-file digest, and
``test_certification_record.py`` fails on a laptop the moment any of the three
moves. Adding a kernel there would have forced a record edit that no device run
backs. Nothing in this file changes a byte of that one.

WHAT IS TRANSCRIBED, AND FROM WHERE. Every line below cites ``../stepping.py``
by name and line, exactly as the certified curl pair does:

* the term table and which axis each component reads
                        ``stepping.H_CONSTITUTIVE_TERMS`` / ``E_CONSTITUTIVE_TERMS``
                        (stepping.py:227-228)
* the sub-lattice       ``stepping.update_H`` (:948, ``half_integer=False``) vs
                        ``stepping.update_E`` (:986, ``half_integer=True``)
* the H source          ``stepping.update_H`` (:921) passes ``fields.B*`` directly —
                        mu = 1 is baked into the array path, not applied here
* the E source          ``stepping.update_E`` (:981-984):
                        ``source * fields.inverse_epsilon_for(component)``
* the recurrence        ``stepping._apply_constitutive_pml`` (:2065-2096)

THE REFERENCE FOR THE DECISIONS — tile shape, launch geometry, vectorization —
is the sibling Triton kernel ``triton_kernels/kernels.py:constitutive_step``,
which is certified. The ARITHMETIC is written from ``stepping.py`` regardless.
The two agree on all four decisions that matter here: one element per program
lane, a flat 1-D grid over ``nx*ny*nz``, 256 lanes per block
(``kernels.DEFAULT_BLOCK = 256``, ``step_curl_kernels._REAL_PML_THREADS = 256``),
and no vectorized load — every access is a scalar ``float`` at the linear index,
because the coefficient index is a per-axis *lookup* and a vector load would
straddle the ``k`` decomposition.

=============================================================================
DO NOT TRANSCRIBE ``update_H_pml_complex`` / ``update_E_pml_complex``
=============================================================================

They are the obvious template and they are the wrong one. Three defects, all
read first-hand out of ``step_curl_kernels.py`` and all silent:

1. THE GROUPING IS FLATTENED. Both write
   ``f[idx] += kps_val * fw_new - kms_val * fw_prev`` — that is
   ``f + (kps*src - kms*prev)``. The array path is TWO SEPARATE ACCUMULATIONS,
   ``field += kps*fw`` then ``field -= kms*fw_previous`` (stepping.py:2133-2134
   and the scratch branch :2140-2143), i.e. ``((f + kps*src) - kms*prev)``.
   Float addition is not associative; these are different float32 numbers. It
   is the same defect class that measures the four plain curl kernels at 0/16
   against the array path.
2. THE E SIDE BINDS ONE EPSILON VOLUME FOR ALL THREE COMPONENTS. Its host
   wrapper passes ``fields.inv_eps``, which ``Fields.set_epsilon_volumes``
   re-points at ``self._inv_eps_components["Ez"]`` (fields.py:1259-1260) as a
   "representative material array". On any run whose three components do not
   alias one volume — a diagonal anisotropic epsilon — Ex and Ey are then
   updated with **Ez's** inverse permittivity. The comment directly above that
   line says constitutive updates always use the component-aware accessors; the
   complex kernel does not. This file binds three pointers.
3. THE E SIDE PUTS INVERSE EPSILON ON THE LEFT
   (``complex<float>(inv_eps_val, 0.0f) * Dx[idx]``) where stepping.py:1011
   writes ``source * fields.inverse_epsilon_for(component)``, D on the left.
   This one is INERT on the bits — float32 multiply is bitwise commutative, and
   the sibling track measured exactly this null at 30/30 identical
   (``DISPERSIVE_NULL_MUTATIONS = ("inv_eps_left",)``). It is listed because a
   reader checking defect 2 will look straight at it, and because transcription
   order is kept here for discipline, not because it changes a bit.

=============================================================================
PLATFORM FACTS THIS FILE DEPENDS ON (CUDA path, adjudicated 2026-08-15)
=============================================================================

* ``--fmad=false`` is CORRECTNESS, not tuning. ``a*b + c`` is contracted by
  NVRTC into ``fma.rn.f32``, which rounds once where the array path rounds
  twice. Both accumulations below (``+ kps*src`` and ``- kms*prev``) are FMA
  candidates and the E side's ``D*inv_eps`` feeds a third. Under the guard the
  PTX census over all fourteen kernels of the sibling module found zero
  ``fma.rn.f32`` and zero contractible mul/add/sub.
* SIGNED ZERO: CUDA does NOT canonicalize it the way Triton does. ``-x`` lowers
  to ``neg.f32``, not to ``0.0f - x``, so a negated addend needs no workaround
  on this path. SAY IT OUT LOUD rather than rely on it — the asymmetry is a
  trap for anyone porting a body between the two tracks in either direction,
  and this kernel is written so that it never depends on the answer: the
  subtraction is spelled ``a - kms*prev``, never ``a + (-(kms*prev))``.
* THE DEVICE STRINGS ARE PURE ASCII, and that is a compile requirement rather
  than a style rule. ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source
  to a ``.cu`` file with a bare ``open(..., 'w')`` (compiler.py:368), so the
  bytes go through the interpreter's LOCALE encoding — ASCII under C/POSIX, which
  is what a non-interactive shell on the validation host gets. MEASURED
  2026-08-15: two em-dashes in the comment block of the E kernel below made it
  die at its first launch with ``UnicodeEncodeError ... position 2535``, while
  the pure-ASCII H kernel compiled beside it. Nothing was wrong with the
  arithmetic; the kernel simply could not be handed to the compiler.
  ``test_constitutive_pml_real.py`` now pins this for the device strings of BOTH
  modules, by scanning and by performing the encode itself.
* DIVISION: none. This sub-step divides nowhere, which sidesteps the one PTX
  exception on record — ptxas expands ``div.rn.f32`` into a Newton-Raphson
  sequence whose range checks carry ``.FTZ`` in SASS regardless of the PTX
  modifier, so a PTX audit does not fully enforce a subnormal *keep* policy
  where a division appears. If a future constitutive variant ever divides (an
  epsilon reciprocal computed in-kernel rather than looked up, say), that
  exception applies to it and this note is where it was written down.

=============================================================================
CERTIFIED 2026-08-15 — AND WHAT THAT DOES AND DOES NOT MEAN
=============================================================================

Both kernels are byte-identical to the array path, per sub-step and over 60
consecutive launches, on an RTX A6000, under BOTH float32 subnormal policies,
with the two policies proven to have compiled DISTINCT binaries. The evidence
block is ``certification.json``'s ``constitutive_2026-08-15``; the numbers there
are derived from the run's own artifacts by
``results/cuda_constitutive_recut_2026-08-15/summarize_constitutive_recut.py``.

WHAT IT DOES NOT MEAN. NOTHING DISPATCHES THEM, and the reason is stronger than
a planner declining to: NO MODULE IN ``meep_gpu/`` IMPORTS ``cuda_kernels`` AT
ALL. ``fastpath.py`` is the Triton dispatcher and never names this package;
``test_package_boundary.py`` pins that absence in both directions. The only
importers are this directory's tests and the parity probes. So certification here
is a statement about two kernels and not about any run, and wiring them is a
separate piece of work that does not exist yet.

(An earlier draft of this header said dispatch was blocked because
``fastpath.plan_fast_path`` "returns None on every branch". That was wrong twice:
that function dispatches the five wired TRITON families and returns plans all the
time, and it has nothing to do with this package either way.)

The predicate exists so that when these ARE wired, the configurations they must
not serve are refused by name rather than by omission, and
:func:`covers_real_pml_constitutive` is the only thing that will stand between
the two facts.

A kernel with a gate verdict moves from ``UNCERTIFIED_KERNELS`` to
``CERTIFIED_KERNELS`` and gains a record entry; a kernel in neither set fails
``test_constitutive_pml_real.py``.
"""

import cupy as cp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..fields import Fields
    from ..pml import PML

# The predicate lives in the CUPY-FREE sibling ``coverage.py`` so it stays
# importable, exercisable and mutable on a machine with no GPU. Imported back
# here because this module is the slice's public face; the by-path fallback is
# for the bit-identity probe, which loads these modules outside the package.
try:
    from .coverage import (CONSTITUTIVE_SIDES, constitutive_sub_lattice,
                           covers_real_pml_constitutive)
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    _coverage_spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_coverage",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "coverage.py"))
    _coverage = _importlib_util.module_from_spec(_coverage_spec)
    _coverage_spec.loader.exec_module(_coverage)
    CONSTITUTIVE_SIDES = _coverage.CONSTITUTIVE_SIDES
    constitutive_sub_lattice = _coverage.constitutive_sub_lattice
    covers_real_pml_constitutive = _coverage.covers_real_pml_constitutive

# The compile memo is a CuPy-free sibling too, and the key carries the subnormal
# policy in force plus the source string actually compiled — so neither a late
# policy install nor a mutated kernel can be served an earlier binary.
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

#: Byte-identical to the array path with a gate verdict behind it, cut
#: 2026-08-15 on an RTX A6000 under BOTH float32 subnormal policies. The evidence
#: is ``certification.json``'s ``constitutive_2026-08-15`` block; the artifacts
#: are ``parity/meep_gpu/results/cuda_constitutive_recut_2026-08-15/``. An entry
#: added here WITHOUT a record entry is the failure
#: ``test_constitutive_pml_real.py`` exists to catch, and it still is.
CERTIFIED_KERNELS = ("update_H_pml_real", "update_E_pml_real")

#: Shipped but not gated. EMPTY, and the partition test requires that every
#: shipped kernel be in exactly one of the two sets — so a kernel added to this
#: file without a gate verdict fails there rather than shipping unmeasured.
#: Spelled as a dict rather than through a shared constant, and WITHOUT a type
#: annotation: the partition test reads it off the syntax tree, without importing
#: the module, because that is the only way to read it on a host with no CuPy —
#: and an annotated assignment is an ``ast.AnnAssign``, which that reader does not
#: match. Annotating it made the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}


# =============================================================================
# THE DEVICE CODE
# =============================================================================
#
# FOUR THINGS DECIDE BIT-IDENTITY HERE. The first three are the sibling Triton
# kernel's numbered notes, measured there and re-stated because they are
# properties of the TRANSCRIPTION, not of the language it is written in. The
# fourth is this file's own.
#
# 1. THE GROUPING IS TWO SEPARATE ACCUMULATIONS, LEFT TO RIGHT.
#
#        a = f[idx];
#        a = a + kps * src;        // stepping.py:2133  field += kps * fw
#        f[idx] = a - kms * prev;  // stepping.py:2134  field -= kms * fw_previous
#
#    NOT ``f[idx] + (kps*src - kms*prev)``, which is what the complex kernels in
#    the sibling module write and a different float32 number. The scratch branch
#    of ``_apply_constitutive_pml`` (:2089-2096) is the same tree by a different
#    route — ``multiply(kps, fw, out=product); field += product; multiply(kms,
#    fw_previous, out=product); field -= product`` — so both array-path branches
#    agree and there is no ambiguity to resolve.
#
# 2. THE COEFFICIENT INDEX IS THE COMPONENT'S OWN AXIS. ``kps_x``/``kms_x`` for
#    component 0, ``_y`` for 1, ``_z`` for 2 — MEEP's ``dsigw``, the absorption a
#    component accumulates along the direction it points in
#    (``H_CONSTITUTIVE_TERMS``/``E_CONSTITUTIVE_TERMS``, stepping.py:227-228).
#    It is NOT the dsig/dsigu cycle the curl recurrence uses, and getting the two
#    confused is a smooth, converged, entirely wrong absorber.
#
# 3. ``prev`` IS READ BEFORE ``fw`` IS WRITTEN. The array path copies fw first
#    (``fw_previous = fw.copy()``, :2084; ``scratch.copy_of("previous", fw)``,
#    :2090) for exactly this reason. Storing before loading makes ``prev`` the
#    value just written, and is wrong only where ``kms != 0`` — i.e. INSIDE THE
#    PML ONLY — which reads as a slightly weaker absorber, not as a bug.
#
# 4. THERE IS NO OWNERSHIP MASK AND NO GHOST RULE, AND THAT IS TRANSCRIBED, NOT
#    FORGOTTEN. ``_apply_constitutive_pml`` writes every cell of the volume, and
#    this sub-step reads no neighbour so it needs no ``shift_up``/``shift_dn``.
#    Anyone porting the curl kernel's shape into this one will reach for both.
#    Neither belongs here.
#
#    THE MASK'S CALL SITES, READ OFF THE FILE RATHER THAN REMEMBERED. This note
#    used to say ``_mask_non_owned_cells`` (:1865) was "called from ``_apply_curl``
#    and from nowhere else", and that was wrong twice: there are THREE call sites
#    and none of them is ``_apply_curl``. They are ``step_B`` (:369), ``step_D``
#    (:450) and ``_bfast_term`` (:902) — two curl sub-steps and the BFAST advance,
#    which sits directly above ``update_H``. The conclusion is unchanged (no
#    constitutive function masks anything, and the predicate refuses BFAST
#    outright), but a sentence about another file's structure is exactly the kind
#    this tranche insists on reading off the source, so
#    ``test_constitutive_pml_real.py`` now does.
#
#    That does NOT make the sub-step boundary-agnostic, which is why the
#    predicate still refuses a fold and a cylindrical axis: both change the
#    STORED EXTENT, and the extent is what turns ``i`` into a coefficient index.

_REAL_CONSTITUTIVE_PRELUDE = r'''
// stepping._apply_constitutive_pml (stepping.py:2065), MEEP's step_update_EDHB
// with dsigw active:
//
//     realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
//     fw[i] = g[i] * u[i];                   // B for H, D*inv_eps for E
//     f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
//
// The two accumulations are kept SEPARATE and LEFT-TO-RIGHT (note 1 above);
// `prev` is loaded before the store (note 3). The intermediate `a` is a
// register rather than a second read-modify-write of f[idx]: a float32 stored
// to global memory and loaded back is the identity on the bits, which the
// sibling track measured as its own null mutation (reload_fu_from_memory,
// 60/60 identical) on the curl's equivalent expression.
__device__ __forceinline__ void constitutive_apply(
    float* __restrict__ f, float* __restrict__ fw, int idx, float src,
    float kps, float kms
) {
    float prev = fw[idx];
    fw[idx] = src;
    float a = f[idx] + kps * src;
    f[idx] = a - kms * prev;
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 2065->2112

# H side: source is B directly (mu = 1 is in the array path, stepping.py:949),
# INTEGER-position coefficients (:948, half_integer=False).
_update_H_pml_real_kernel_code = _REAL_CONSTITUTIVE_PRELUDE + r'''
extern "C" __global__ void update_H_pml_real(
    float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
    float* __restrict__ f_w_Hx, float* __restrict__ f_w_Hy,
    float* __restrict__ f_w_Hz,
    const float* __restrict__ Bx, const float* __restrict__ By,
    const float* __restrict__ Bz,
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // THE OWN-CELL HOIST (register view): every word this thread reads at idx,
    // loaded before the first store. The certified constitutive_apply steps the
    // registers (f = &field, fw = &aux, idx = 0); the write-back stores them.
    float w_x = f_w_Hx[idx];
    float h_x = Hx[idx];
    float w_y = f_w_Hy[idx];
    float h_y = Hy[idx];
    float w_z = f_w_Hz[idx];
    float h_z = Hz[idx];
    float src_x = Bx[idx];
    float src_y = By[idx];
    float src_z = Bz[idx];

    // Hx <- Bx, dsigw = x.  Hy <- By, dsigw = y.  Hz <- Bz, dsigw = z.
    constitutive_apply(&h_x, &w_x, 0, src_x, kps_x[i], kms_x[i]);
    constitutive_apply(&h_y, &w_y, 0, src_y, kps_y[j], kms_y[j]);
    constitutive_apply(&h_z, &w_z, 0, src_z, kps_z[k], kms_z[k]);

    // The write-back, in the certified store order (auxiliary, then field).
    f_w_Hx[idx] = w_x;
    Hx[idx] = h_x;
    f_w_Hy[idx] = w_y;
    Hy[idx] = h_y;
    f_w_Hz[idx] = w_z;
    Hz[idx] = h_z;
}
'''

# E side: source is D * inv_eps with D ON THE LEFT (stepping.py:1011-1013),
# HALF-INTEGER coefficients (:1015, half_integer=True), and THREE inverse-epsilon
# pointers — one per component, from Fields.inverse_epsilon_for. They may all
# alias one volume (set_isotropic_epsilon_volume) and the kernel does not care;
# what it must not do is bind Ez's volume for all three, which is what
# update_E_pml_complex does through fields.inv_eps (fields.py:1259-1260).
_update_E_pml_real_kernel_code = _REAL_CONSTITUTIVE_PRELUDE + r'''
extern "C" __global__ void update_E_pml_real(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    const float* __restrict__ Dx, const float* __restrict__ Dy,
    const float* __restrict__ Dz,
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // THE OWN-CELL HOIST (register view): every word this thread reads at idx,
    // loaded before the first store. The certified constitutive_apply steps the
    // registers (f = &field, fw = &aux, idx = 0); the write-back stores them.
    float w_x = f_w_Ex[idx];
    float e_x = Ex[idx];
    float w_y = f_w_Ey[idx];
    float e_y = Ey[idx];
    float w_z = f_w_Ez[idx];
    float e_z = Ez[idx];

    // NOT __restrict__ on the three inv_eps pointers, deliberately: an isotropic
    // run hands the same device pointer three times (fields.py:1323-1325), and
    // restrict on mutually aliasing arguments is a promise the caller cannot
    // keep. They are read-only, so nothing is lost but the promise.
    // All three products are formed BEFORE any store, where the array path forms
    // each inside its own loop iteration (stepping.py:969-989). Exact, not merely
    // close: this sub-step reads D and inv_eps and writes E and f_w_E, four
    // disjoint sets, so no store can reach an operand of a later product. The
    // OFF-DIAGONAL row product is precisely where that stops being true -- it
    // reads the other components' volumes -- which is why the predicate refuses
    // it and why it is a kernel of its own rather than a flag on this one.
    float src_x = Dx[idx] * inv_eps_Ex[idx];   // stepping.py:982: source * inv_eps
    float src_y = Dy[idx] * inv_eps_Ey[idx];
    float src_z = Dz[idx] * inv_eps_Ez[idx];

    // Ex <- Dx*inv_eps, dsigw = x.  Ey, dsigw = y.  Ez, dsigw = z.
    constitutive_apply(&e_x, &w_x, 0, src_x, kps_x[i], kms_x[i]);
    constitutive_apply(&e_y, &w_y, 0, src_y, kps_y[j], kms_y[j]);
    constitutive_apply(&e_z, &w_z, 0, src_z, kps_z[k], kms_z[k]);

    // The write-back, in the certified store order (auxiliary, then field).
    f_w_Ex[idx] = w_x;
    Ex[idx] = e_x;
    f_w_Ey[idx] = w_y;
    Ey[idx] = e_y;
    f_w_Ez[idx] = w_z;
    Ez[idx] = e_z;
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 969-989->998-1018, 982->1011

# NVRTC compile options — CORRECTNESS, not performance. Identical to the sibling
# module's, and spelled here rather than imported so that loading this file by
# path (which the probe does) cannot pick up a different tuple than the one the
# gate compiled. ``test_constitutive_pml_real.py`` pins the two spellings equal.
_COMPILE_OPTIONS = ('--fmad=false',)

#: Lanes per block. Both tracks landed on 256 independently — the sibling
#: module's ``_REAL_PML_THREADS`` and ``triton_kernels/kernels.DEFAULT_BLOCK``.
#: One element per lane, flat 1-D grid; the sub-step is element-wise, so there is
#: no tile to shape and no reuse to exploit.
_CONSTITUTIVE_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    Shared memo with the sibling module — the cache is keyed on the source
    string, so two modules cannot collide. The probe drives this between guard
    sets, where the compiler ITSELF is substituted and the option tuple is
    overridden from outside, a change no memo key can see.
    """
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str):
    """Compile on first use, memoized on (name, options, policy, source).

    THE CODE MAP IS REBUILT PER CALL AND THAT IS LOAD-BEARING, for the same
    reason it is in the sibling module: the source is part of the memo key, so
    the map has to be read before a key exists to miss on, and the bit-identity
    probe mutates kernels by assigning over the module-level source string. A
    map memoized at first call would hand back the pre-mutation string forever —
    a leg reporting a pass for a mutation it never applied.
    """
    code_map = {
        'update_H_pml_real': (_update_H_pml_real_kernel_code,
                              'update_H_pml_real'),
        'update_E_pml_real': (_update_E_pml_real_kernel_code,
                              'update_E_pml_real'),
    }
    code, func_name = code_map[name]
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, func_name, options=_COMPILE_OPTIONS))


def real_constitutive_tables(pml: 'PML', half_integer: bool) -> dict:
    """Flatten one side's six kps/kms coefficient vectors into cached device views.

    Called ONCE per frozen configuration, not per launch — the complex wrappers
    in the sibling module call ``cp.ascontiguousarray(...ravel())`` on every
    launch, which is six device allocations per sub-step for tables that never
    change (port-reference defect 5.14). ``PML._reshape_for_broadcast`` already
    stores float32 arrays of shape (n,1,1)/(1,n,1)/(1,1,n), so ``reshape(-1)`` is
    a view and holding it costs nothing.

    ``half_integer`` selects the Yee sub-lattice, and it is the argument this
    function exists to make explicit: ``update_H`` reads the INTEGER positions
    (stepping.py:948) and ``update_E`` the half-integer ones (:1015). Swapped, it
    is a half-cell error in the absorber profile — converged, smooth and wrong —
    which is why the gate carries ``swap_constitutive_sublattice`` as a host
    mutation rather than trusting the call sites.
    """
    suffix = "_h" if half_integer else ""
    tables = {}
    for axis in ("x", "y", "z"):
        for name in ("kps", "kms"):
            attribute = f"{name}_{axis}{suffix}"
            flat = getattr(pml, attribute).reshape(-1)
            if flat.dtype != cp.float32:
                raise ValueError(
                    f"{attribute} is {flat.dtype}; the real constitutive kernels "
                    f"index float32 coefficient vectors.")
            if not flat.flags.c_contiguous:
                raise ValueError(
                    f"{attribute} did not flatten to a contiguous view; the kernel "
                    f"indexes it as a bare vector.")
            tables[f"{name}_{axis}"] = flat
    return tables


def constitutive_tables_for(side: str, pml: 'PML') -> dict:
    """One side's coefficient views, with the sub-lattice chosen for the caller.

    ``real_constitutive_tables`` takes ``half_integer`` as an argument, which means
    every call site is one more place the H/E pairing can be got backwards — and
    backwards is a half-cell error in the absorber profile, not a crash. This asks
    :func:`constitutive_sub_lattice`, the same function the predicate asks, so the
    two cannot disagree.

    IT IS REACHED FROM :func:`update_fused_pml_real`, and until 2026-08-15 it was
    not reached from anywhere. The launcher took a caller-supplied ``tables`` dict
    and passed it straight through, so the docstring above it claiming the launcher
    and the predicate "cannot disagree" described a convention, not an enforced
    property — the half-cell error was one argument away on the live entry point,
    ``real_constitutive_tables``' dtype and contiguity guards were unreachable, and
    the test asserting this function's structure inspected code nothing called.
    """
    return real_constitutive_tables(pml, constitutive_sub_lattice(side))


def _launch(kernel, targets, auxiliaries, sources, inverse_epsilon, tables: dict):
    """Shared launch: the two sides differ only in their arrays and one argument.

    ``inverse_epsilon`` is None on the H side, where the source is B itself
    (mu = 1, stepping.py:949), and a three-tuple on the E side.
    """
    nx, ny, nz = targets[0].shape
    blocks = (nx * ny * nz + _CONSTITUTIVE_THREADS - 1) // _CONSTITUTIVE_THREADS
    arguments = [
        targets[0], targets[1], targets[2],
        auxiliaries[0], auxiliaries[1], auxiliaries[2],
        sources[0], sources[1], sources[2],
    ]
    if inverse_epsilon is not None:
        arguments.extend(inverse_epsilon)
    arguments.extend([
        np.int32(nx), np.int32(ny), np.int32(nz),
        tables["kps_x"], tables["kms_x"],
        tables["kps_y"], tables["kms_y"],
        tables["kps_z"], tables["kms_z"],
    ])
    kernel((blocks,), (_CONSTITUTIVE_THREADS,), tuple(arguments))


def _update_H_fused_pml_real(fields: 'Fields', tables: dict):
    """``stepping.update_H`` for real fields under a PML, in one launch.

    ``tables`` are the INTEGER coefficient views from
    :func:`real_constitutive_tables` (``half_integer=False``, stepping.py:948).
    The source is ``Bx/By/Bz`` directly: ``stepping.update_H`` (:949) passes
    ``getattr(fields, source)`` with ``source`` from ``H_CONSTITUTIVE_TERMS``,
    and mu = 1 is already baked into that choice — there is no permeability
    volume to multiply by and introducing one here would be a second engine.
    """
    _launch(_get_kernel('update_H_pml_real'),
            (fields.Hx, fields.Hy, fields.Hz),
            (fields.f_w_Hx, fields.f_w_Hy, fields.f_w_Hz),
            (fields.Bx, fields.By, fields.Bz),
            None, tables)


def _update_E_fused_pml_real(fields: 'Fields', tables: dict):
    """``stepping.update_E`` for real fields under a PML, in one launch.

    ``tables`` are the HALF-INTEGER views (``half_integer=True``,
    stepping.py:1015). The source is ``D * inv_eps`` formed in-kernel, with the
    per-component inverse permittivity from ``Fields.inverse_epsilon_for``
    (fields.py:1337) — three pointers, never ``fields.inv_eps``, which is the Ez
    view (fields.py:1259-1260).

    This wrapper is valid only where ``displacement_minus_polarization`` returns
    the D array itself, i.e. where no susceptibility drives the component
    (fields.py:1097-1098). :func:`covers_real_pml_constitutive` refuses a
    registered polarization by name for exactly that reason.
    """
    _launch(_get_kernel('update_E_pml_real'),
            (fields.Ex, fields.Ey, fields.Ez),
            (fields.f_w_Ex, fields.f_w_Ey, fields.f_w_Ez),
            (fields.Dx, fields.Dy, fields.Dz),
            (fields.inverse_epsilon_for("Ex"),
             fields.inverse_epsilon_for("Ey"),
             fields.inverse_epsilon_for("Ez")),
            tables)


#: Launch one side by name, for callers that hold the side as data (the probe,
#: and any future planner). ``CONSTITUTIVE_SIDES`` and
#: :func:`constitutive_sub_lattice` come from ``coverage.py``, so the predicate
#: and the launcher cannot disagree about which sub-lattice a side reads.
_LAUNCHERS = {"H": _update_H_fused_pml_real, "E": _update_E_fused_pml_real}


def update_fused_pml_real(side: str, fields: 'Fields', pml: 'PML' = None, *,
                          tables: dict = None):
    """Run ``update_H`` (side='H') or ``update_E`` ('E') in one launch.

    SUPPLY ``pml`` AND THE SUB-LATTICE IS DECIDED HERE, from the side, by the same
    :func:`constitutive_sub_lattice` the predicate asks — which is what makes "the
    launcher and the predicate cannot disagree about which table a side reads" an
    enforced property of this function rather than a convention a caller may keep.
    It did not used to be: ``tables`` was a positional parameter passed straight
    through to :func:`_launch`, ``constitutive_tables_for`` had no callers at all,
    and the half-cell error the whole pairing exists to prevent was one argument
    away on the only entry point anything actually calls.

    ``tables`` IS THE GATE'S DOOR and stays, keyword-only and named for what it is.
    ``probe_fused_kernel_bit_identity.py`` feeds synthetic tables that no ``PML``
    produces, and its host mutation ``swap_constitutive_sublattice`` feeds
    DELIBERATELY MIS-PAIRED ones — a gate that could not hand in its own tables
    could not arm that mutation. Passing both, or neither, is refused: a caller
    that supplies a layer AND a table set has two answers to a question with one.
    """
    if side not in _LAUNCHERS:
        raise ValueError(
            f"side must be one of {sorted(_LAUNCHERS)}, got {side!r}")
    if (pml is None) == (tables is None):
        raise ValueError(
            "pass exactly one of pml (the tables are derived from the side) or "
            "tables (the gate supplies its own, including mis-paired ones); "
            f"got pml={'set' if pml is not None else 'None'} and "
            f"tables={'set' if tables is not None else 'None'}")
    if tables is None:
        tables = constitutive_tables_for(side, pml)
    return _LAUNCHERS[side](fields, tables)
