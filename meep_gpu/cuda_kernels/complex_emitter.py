"""The COMPLEX-storage PML device sources, emitted per expansion arm -- no CuPy.

WHAT THIS FAMILY IS. Complex64 field storage under a real (split-field) PML: the
four sub-steps ``step_B`` / ``step_D`` / ``update_H`` / ``update_E`` when the run
carries ``force_complex_fields`` or a nonzero ``k_point``. Today's hand-CUDA track
handles real float32 storage only, and ``coverage.covers_real_pml_curl`` and
``coverage.covers_real_pml_constitutive`` both refuse this configuration by name
("complex64 storage: the recurrence is the same but the storage is not"). That
refusal is exactly right about the two REAL kernels and says nothing about
whether a complex kernel could serve the run; this module is the kernel, and
``coverage.covers_real_pml_complex_curl`` /
``coverage.covers_real_pml_complex_constitutive`` are the written refusal for the
configurations it must NOT serve.

WHY AN EMITTER RATHER THAN FOUR STRING CONSTANTS. The complex multiply has TWO
licensable transcriptions -- ``FMA_V1`` and ``NAIVE`` -- and which one reproduces
the reference implementation's bytes is a MEASURED PLATFORM FACT, not a choice
(see ARM below). So the device code is a function of the arm, the thing a test has
to check is a FUNCTION, and a function has to be imported to be called. It goes
where ``coverage.py``, ``compile_cache.py`` and ``offdiag_emitter.py`` already are:
a stdlib-only sibling that runs at the merge bar on a machine with no GPU, on this
package's rule that the parts whose failure mode is a SILENT WRONG ANSWER are the
parts that must be exercisable on a laptop.

NOTHING HERE IMPORTS CUPY OR NUMPY, and nothing here imports ``coverage`` either:
this family's tables are its own.

=============================================================================
THE REFERENCE, AND WHY IT IS TRANSCRIBED RATHER THAN DERIVED
=============================================================================

Two certified siblings already write this arithmetic and this module copies
theirs rather than deriving a third:

* Triton ``triton_kernels/complex_fields.py`` -- ``bloch_pml_curl_step``
  (:510-737) and ``bloch_constitutive_step`` (:739-838), with the three multiply
  helpers ``_rotate_field_left`` (:433-449), ``_mul_field_left`` (:452-485) and
  ``_mul_coefficient_left`` (:488-507);
* Metal ``metal_kernels/complex_fields.py``, the same tranche in MSL.

The array path they both transcribe is ``stepping.py``, and every line below
carries its citation. The transcription is the Triton body with THREE changes and
only three, which is what its own docstring says the complex kernel is against the
real one:

1. every field access is a float32 WORD PAIR at ``2*idx`` / ``2*idx + 1``
   (complex64 is bit-layout (re, im) interleaved, so a C-contiguous complex volume
   IS a C-contiguous float32 volume of shape ``(nx, ny, nz, 2)``);
2. the wrapped lane of each phased axis's shifted operand is rotated by the Bloch
   factor BEFORE the difference -- the slot between ``_rolled`` and the subtract
   (stepping.py:1814-1818 up, :1865-1869 down, both through ``_apply_bloch_phase``
   :1893-1909);
3. every real-coefficient multiply is the ZERO-IMAGINARY COMPLEX PRODUCT per the
   arm, never a bare float multiply.

=============================================================================
THE ARM. A MEASURED PLATFORM FACT, CARRIED IN A PROBE ARTIFACT
=============================================================================

``numpy``/``cupy`` do not expose a scalar-times-complex loop: ``np.multiply``
carries only 'FF->F' for complex, so EVERY real-coefficient multiply on the array
path is a FULL complex multiply with a zero-imaginary operand. Whether the
platform's complex multiply is FUSED (``FMA_V1``: rounded inner products, fused
outer add) or separately rounded (``NAIVE``) is a property of the compiled
dispatch and differs between platforms -- measured at ~25% of words apart on
random pairs. It is therefore bound from a PROBE ARTIFACT, and the arbiter is
``triton_kernels.complex_fields.expansion_license``.

THIS MODULE DOES NOT RE-IMPLEMENT THAT RULE and does not import it either. The
arm arrives as an ARGUMENT -- :func:`complex_source` takes it, the launcher takes
it, and ``coverage.covers_real_pml_complex_curl`` takes the licence VERDICT and
checks its shape without re-deriving it. Two spellings of an arbiter is one too
many (the Triton module says so at its own ``ONE RULE, SEVERAL PATTERN SETS``
paragraph, having just merged two drifted copies); a stdlib-only module reaching
across into the Triton package for it would be worse than either. So the boundary
is: the GATE produces the verdict, this package consumes an arm.

THE LICENCE IS POLICY-CONDITIONAL. An arm classified under one float32 subnormal
policy reproduces the platform's bytes under THAT policy only, which is why the
gate cuts a probe per policy and why a certification that spans both policies has
to name the arm it bound in each.

=============================================================================
WHAT IS COMPILE-TIME AND WHAT IS NOT -- this family's whole design decision
=============================================================================

The Triton kernel carries EIGHT compile-time axes on the curl (BACKWARD, three
boundary codes, three phase flags, EXPANSION). This emitter splits them exactly as
``offdiag_emitter`` splits its twelve, by what they do to the ARITHMETIC:

* the three BOUNDARY codes and the three PHASE flags are BRANCH axes -- they
  select an index, a predicated zero, or whether one multiply happens on one plane
  -- so they are ordinary runtime ``int`` arguments, exactly as the certified
  ``step_B_pml_real`` already takes ``bc_x``/``bc_y``/``bc_z``
  (step_curl_kernels.py:1687). Sixty-four variants collapse to one source.
  A branch and a ``tl.where`` select store the same bytes: the rotation is applied
  on the wrapped lane and nowhere else in both spellings, and Triton uses the
  select only because it has no per-lane scalar branch.
* BACKWARD and the H/E source rule are STRUCTURAL -- different stencils, different
  masks, a different coefficient sub-lattice, and on the E side an extra volume
  read -- so they are four separate kernels, exactly as the certified real pair is
  two;
* EXPANSION is an ARITHMETIC axis. It changes which products are fused and how the
  zero cross terms are spelled, so it stays compile-time and is this emitter's
  only parameter.

Four kernels times two arms is eight sources; :func:`corpus_digest` hashes all
eight so one changed character anywhere moves one number.

=============================================================================
FIVE THINGS DECIDE BIT-IDENTITY HERE
=============================================================================

1. THE ZERO CROSS TERMS ARE CARRIED LITERALLY AND MUST NOT BE FOLDED.
   With the field on the left, ``z * (c + 0j)`` is
   ``re = fma(z_re, c, (z_im * 0.0f) * -1.0f)``, ``im = fma(z_re, 0.0f, z_im * c)``.
   Plane-wise ``{re*c, im*c}`` is byte-wrong on signed zeros: the cross term
   carries the sign of the field's OTHER word into an addend that is exactly zero,
   and ``fma(x, y, -0.0f)`` differs from ``fma(x, y, +0.0f)`` wherever ``x*y`` is
   exactly zero -- which a thin absorber's deepest ``kms = kappa - sigma`` reaches
   from all-zero state. Constant-folding ``z_im * 0.0f`` to a literal ``0.0f``
   would destroy that sign, so its survival through NVRTC is a GATE question with
   a dedicated mutation, not a construction guarantee.

2. OPERAND ORDER IS NORMATIVE UNDER FMA_V1, so the array path's orientations are
   transcribed literally and each has its own helper:
   ``dtdx`` scalar LEFT (stepping.py:1682, ``xp.multiply(dtdx, total)``);
   ``fu *= kms`` / ``fu *= sinv`` / ``field *= kms_u`` / ``field *= sinv_u`` field
   LEFT (:1929-1935); ``kps * fw`` and ``kms * fw_previous`` coefficient LEFT
   (:2086-2087, :2093-2095); ``D * inv_eps`` D LEFT (:982-984); the phase multiply
   plane LEFT with the phase pre-rounded to complex64 (:1862).

3. NEGATION IS ``* -1.0f`` AND THAT SPELLING IS DELIBERATE, though on THIS platform
   it is belt and braces rather than the load-bearing fact it is in Triton. Triton
   lowers unary ``-x`` as ``0.0 - x`` (semantic.py:386-391), which canonicalizes a
   ``-0.0`` addend to ``+0.0`` before the fma; CUDA lowers ``-x`` to ``neg.f32``
   and does NOT canonicalize signed zeros (adjudicated for the ADE family,
   ade_kernels.py's platform-facts block). Both spellings are therefore
   IEEE-exact here. ``* -1.0f`` is kept so the CUDA and Triton bodies read as the
   same transcription, and ``test_complex_pml.py`` pins the absence of unary minus
   on any float path so a future edit cannot introduce the Triton hazard by
   copying it back.

4. THE CURL'S PLANE-WISE GROUPING IS ``((sf - f1) + (f2 - ss))``, NOT the
   left-to-right ``((sf - f1) + f2) - ss`` C would associate from an unparenthesised
   sum. ``stepping._curl_from_operands`` (:1601-1636) forms
   ``dtdx * ((shifted_first - first) + (second - shifted_second))`` and float32
   addition is not associative. Complex add and subtract are component-wise, so the
   grouping is the real kernel's per plane.

5. NO FMA CONTRACTION. ``--fmad=false`` is CORRECTNESS, not tuning: the PML
   recurrence has contraction candidates at ``(fu*kms) - curl`` and
   ``(f*kms_u) + fu_new``, and the constitutive tail at both accumulations. The
   FMA_V1 arm's fusions are spelled with ``__fmaf_rn``, which is a single
   ``fma.rn.f32`` whatever the contraction flag says -- so the flag removes the
   accidental fusions and leaves the transcribed ones.

=============================================================================
PLATFORM REQUIREMENTS
=============================================================================

* THE DEVICE STRINGS ARE PURE ASCII, a compile requirement rather than a style
  rule: ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a
  bare ``open(..., 'w')`` (compiler.py:368), so the bytes go through the
  interpreter's LOCALE encoding -- ASCII under C/POSIX, which is what a
  non-interactive shell on the validation host gets. Two em-dashes in a comment
  killed a sibling kernel at its first launch on 2026-08-15.
* NO DIVISION anywhere: the one PTX exception on record is ptxas expanding
  ``div.rn.f32`` into a sequence whose range checks carry ``.FTZ`` in SASS
  regardless of the PTX modifier, which cannot arise in a divide-free family. The
  reciprocal the recurrence needs is ``sinv``, computed host-side by ``PML``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

#: The two transcription arms, and the integer codes the host binds them by. The
#: SAME mapping as ``triton_kernels.complex_fields.EXPANSIONS``; restated rather
#: than imported for the reason every table in this package is restated (a
#: stdlib-only module imports nothing), and pinned equal to the Triton spelling by
#: ``test_complex_pml.py`` so the two cannot drift.
EXPANSIONS: Dict[str, int] = {"NAIVE": 0, "FMA_V1": 1}

#: Arm code -> arm name, the inverse of :data:`EXPANSIONS`.
EXPANSION_NAMES: Dict[int, str] = {code: name for name, code in EXPANSIONS.items()}

#: The four kernels this family emits, keyed by the sub-step they replace. The
#: value is ``(kernel name, is the D/backward side, does the source carry inv_eps)``
#: -- the two structural axes, written out so a launcher and a test read the same
#: table.
KERNELS: Dict[str, Tuple[str, bool, bool]] = {
    "step_B": ("step_B_pml_complex_bloch", False, False),
    "step_D": ("step_D_pml_complex_bloch", True, False),
    "update_H": ("update_H_pml_complex_bloch", False, False),
    "update_E": ("update_E_pml_complex_bloch", False, True),
}

#: The two curl sub-steps and the two constitutive sides, as the launcher and the
#: predicates name them.
CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")
CONSTITUTIVE_SIDES: Tuple[str, ...] = ("H", "E")

#: Which Yee sub-lattice each sub-step's coefficient vectors come from.
#: ``stepping._curl_coefficients`` (:2418): the B curl reads HALF-INTEGER positions
#: and the D curl INTEGER ones. ``stepping._constitutive_coefficients`` (:2428):
#: ``update_H`` reads INTEGER (stepping.py:948) and ``update_E`` HALF-INTEGER
#: (:1015). Swapped, it is a half-cell error in the absorber profile -- converged,
#: smooth and wrong -- which is why the gate carries it as a host mutation.
HALF_INTEGER: Dict[str, bool] = {
    "step_B": True, "step_D": False, "H": False, "E": True}

# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

#: Word-pair addressing, complex add/sub, and the ghost rules -- everything that
#: does not depend on the arm.
_HEAD = r'''
#define BC_PERIODIC 0
#define BC_METALLIC 1

// complex64 as a float32 WORD PAIR. A C-contiguous complex volume is a
// C-contiguous float32 volume with a doubled last axis, so complex cell w is
// words 2*w and 2*w+1. Every index below is a COMPLEX CELL index; the doubling
// happens here and nowhere else.
typedef struct { float re; float im; } cf;

__device__ __forceinline__ cf cf_load(const float* g, int idx) {
    cf z;
    z.re = g[2 * idx];
    z.im = g[2 * idx + 1];
    return z;
}

__device__ __forceinline__ void cf_store(float* g, int idx, cf z) {
    // Two word stores replace the array path's single complex store. Non-semantic:
    // the re and im planes of one cell do not alias.
    g[2 * idx] = z.re;
    g[2 * idx + 1] = z.im;
}

// Complex add and subtract are component-wise -- measured 0 of 2^17 word
// mismatches against the array path's, which is what licenses the plane-wise
// grouping of the curl below.
__device__ __forceinline__ cf cf_add(cf a, cf b) {
    cf o;
    o.re = a.re + b.re;
    o.im = a.im + b.im;
    return o;
}

__device__ __forceinline__ cf cf_sub(cf a, cf b) {
    cf o;
    o.re = a.re - b.re;
    o.im = a.im - b.im;
    return o;
}

// The array path assigns the INTEGER 0 to a complex64 array (stepping.py:1785,
// :1896), which is the word pair (+0.0f, +0.0f) and not a sign-carrying zero.
__device__ __forceinline__ cf cf_zero() {
    cf o;
    o.re = 0.0f;
    o.im = 0.0f;
    return o;
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1785->1832, 1896->1943

#: The three multiply orientations, per arm. This is the whole of what the arm
#: changes, and the three helpers are one-for-one with the Triton reference's.
_ARM_SOURCE: Dict[int, str] = {
    EXPANSIONS["FMA_V1"]: r'''
// ---- EXPANSION = FMA_V1 ---------------------------------------------------
// The fused form: inner products separately rounded, outer add fused. The
// addend's negation is spelled (a*b) * -1.0f, never unary minus (module note 3).
// __fmaf_rn is a single fma.rn.f32 whatever --fmad says, so the transcribed
// fusions survive the flag that removes the accidental ones.

// One full complex multiply, FIELD on the left -- stepping.py:1862,
// shifted[plane] *= shifted.dtype.type(phase).
__device__ __forceinline__ cf rotate_field_left(cf g, cf p) {
    cf o;
    o.re = __fmaf_rn(g.re, p.re, (g.im * p.im) * -1.0f);
    o.im = __fmaf_rn(g.re, p.im, g.im * p.re);
    return o;
}

// (z) * (c + 0j) with the FIELD on the left -- fu *= kms and its family
// (stepping.py:1929-1935), D * inv_eps with D left (:982-984).
__device__ __forceinline__ cf mul_field_left(cf z, float c) {
    cf o;
    o.re = __fmaf_rn(z.re, c, (z.im * 0.0f) * -1.0f);
    o.im = __fmaf_rn(z.re, 0.0f, z.im * c);
    return o;
}

// (c + 0j) * (z) with the COEFFICIENT on the left -- kps*fw and kms*fw_previous
// (stepping.py:2086-2087, :2093-2095), and the scalar-left multiply(dtdx, total)
// (:1635). Note the zero-sign asymmetry against mul_field_left: here the re cross
// term is 0.0f * z.im, the sign of the field's imag through the LEFT zero, and it
// is the coefficient product that is fused.
__device__ __forceinline__ cf mul_coefficient_left(float c, cf z) {
    cf o;
    o.re = __fmaf_rn(c, z.re, (0.0f * z.im) * -1.0f);
    o.im = __fmaf_rn(c, z.im, 0.0f * z.re);
    return o;
}
''',  # stepping.py live lines for the frozen device-text citation(s) in this string: 1862->1909, 1929-1935->1976-1982, 982-984->1011-1013, 2086-2087->2133-2134, 2093-2095->2140-2142, 1635->1682
    EXPANSIONS["NAIVE"]: r'''
// ---- EXPANSION = NAIVE ----------------------------------------------------
// The separately-rounded scalar-loop form a non-SIMD dispatch takes. Same
// operands, same orientations, same zero cross terms; only the fusion differs.

__device__ __forceinline__ cf rotate_field_left(cf g, cf p) {
    cf o;
    o.re = (g.re * p.re) - (g.im * p.im);
    o.im = (g.re * p.im) + (g.im * p.re);
    return o;
}

__device__ __forceinline__ cf mul_field_left(cf z, float c) {
    cf o;
    o.re = (z.re * c) - (z.im * 0.0f);
    o.im = (z.re * 0.0f) + (z.im * c);
    return o;
}

__device__ __forceinline__ cf mul_coefficient_left(float c, cf z) {
    cf o;
    o.re = (c * z.re) - (0.0f * z.im);
    o.im = (c * z.im) + (0.0f * z.re);
    return o;
}
''',
}

#: The ghost rules and the two recurrences -- arm-independent in shape, but they
#: CALL the arm's helpers, so they follow the arm block in the emitted source.
_TAIL = r'''
// stepping._shift_up (stepping.py:1725-1786), PERIODIC and METALLIC branches
// only: MIRROR and CYL_AXIS are refused by the predicate. The Bloch factor is
// applied to the WRAPPED LANE and nowhere else, which is exactly
// _apply_bloch_phase multiplying the single plane _face(axis, -1) of the rolled
// buffer (:1767-1771 through :1862). ph = 0 skips the multiply entirely rather
// than doing it against 1+0j, which is what keeps k = 0 bit-identical to the
// plain complex engine (:1768-1770).
//
// An invariant axis (na == 1) takes the wrap branch with idx - ia*stride == idx,
// which is what xp.roll computes there -- and IS a wrap lane, so a phase on a
// collapsed axis is applied. Same as the array path.
__device__ __forceinline__ cf cshift_up(
    const float* g, int idx, int ia, int na, int stride, int bc, int ph, cf phase
) {
    if (ia + 1 < na) return cf_load(g, idx + stride);
    if (bc == BC_METALLIC) return cf_zero();
    cf w = cf_load(g, idx - ia * stride);
    if (ph) w = rotate_field_left(w, phase);
    return w;
}

// stepping._shift_down (:1788-1845). The near face wraps DOWN by one lattice
// vector, so it carries the CONJUGATE of the up-going factor (:1818-1822,
// phase.conjugate()); the host passes the conjugate in, so this body is the
// mirror of cshift_up and nothing here negates anything. Taking the same factor
// in both directions is the classic sign error -- every magnitude stays
// plausible and only the phase moves, which is what a band structure is made of.
__device__ __forceinline__ cf cshift_dn(
    const float* g, int idx, int ia, int na, int stride, int bc, int ph, cf phase
) {
    if (ia > 0) return cf_load(g, idx - stride);
    if (bc == BC_METALLIC) return cf_zero();
    cf w = cf_load(g, idx + (na - 1) * stride);
    if (ph) w = rotate_field_left(w, phase);
    return w;
}

// stepping._apply_pml_update (stepping.py:1905-1935), in its in-place order:
//   fprev = fu.copy(); fu *= kms; fu -= curl; fu *= sinv;
//   field *= kms_u; field += fu; field -= fprev; field *= sinv_u
// Every multiply is the zero-imaginary product with the FIELD on the left; the
// two subtractions and the addition are plane-wise between them. The loaded
// fprev IS the array path's copy, taken before the store.
__device__ __forceinline__ void pml_apply(
    float* f, float* fu, int idx, cf curl,
    float kms, float sinv, float kms_u, float sinv_u
) {
    cf fprev = cf_load(fu, idx);
    cf fu_new = mul_field_left(cf_sub(mul_field_left(fprev, kms), curl), sinv);
    cf_store(fu, idx, fu_new);
    cf a = mul_field_left(cf_load(f, idx), kms_u);
    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u));
}

// stepping._apply_constitutive_pml (stepping.py:2065-2096), MEEP's
// step_update_EDHB with dsigw active:
//   fwprev = fw[i]; fw[i] = g[i]*u[i];
//   f[i] += (kap+sig)*fw[i] - (kap-sig)*fwprev
// TWO SEPARATE ACCUMULATIONS, LEFT TO RIGHT -- never flattened to
// f + (kps*src - kms*prev), which is a different float32 number and is what the
// twelve uncertified complex kernels in step_curl_kernels.py write. `prev` is
// read BEFORE the fw store (:2084, :2090); the wrong order is wrong only where
// kms != 0, i.e. INSIDE THE PML ONLY, which reads as a slightly weaker absorber
// rather than as a bug.
__device__ __forceinline__ void constitutive_apply(
    float* f, float* fw, int idx, cf src, float kps, float kms
) {
    cf prev = cf_load(fw, idx);
    cf_store(fw, idx, src);
    cf a = cf_load(f, idx);
    a = cf_add(a, mul_coefficient_left(kps, src));
    a = cf_sub(a, mul_coefficient_left(kms, prev));
    cf_store(f, idx, a);
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1725-1786->1772-1833, 1905-1935->1952-1982, 2065-2096->2112-2143

#: The curl kernels. ``__SIDE__`` is B or D, ``__SHIFT__`` the ghost helper, and
#: the three per-component blocks differ only in their stencil and their mask, so
#: they are written out rather than macro-ed: the term table IS the physics and a
#: reader has to be able to check it against ``stepping.B_CURL_TERMS`` by eye.
_CURL_TEMPLATE = r'''
extern "C" __global__ void __NAME__(
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ u0, float* __restrict__ u1, float* __restrict__ u2,
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    int nx, int ny, int nz, float dtdx,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    // Strides in COMPLEX CELLS. cf_load does the word doubling.
    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    cf px; px.re = pxr; px.im = pxi;
    cf py; py.re = pyr; py.im = pyi;
    cf pz; pz.re = pzr; pz.im = pzi;

    // Target 0: curl_x = d(g2)/dy - d(g1)/dz; dsig=y, dsigu=z; mask __AXES0__.
    {
        cf f_1 = cf_load(g2, idx);
        cf sf = __SHIFT__(g2, idx, j, ny, sy, bc_y, ph_y, py);
        cf f_2 = cf_load(g1, idx);
        cf ss = __SHIFT__(g1, idx, k, nz, sz, bc_z, ph_z, pz);
        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));
__MASK0__
        pml_apply(f0, u0, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // Target 1: curl_y = d(g0)/dz - d(g2)/dx; dsig=z, dsigu=x; mask __AXES1__.
    {
        cf f_1 = cf_load(g0, idx);
        cf sf = __SHIFT__(g0, idx, k, nz, sz, bc_z, ph_z, pz);
        cf f_2 = cf_load(g2, idx);
        cf ss = __SHIFT__(g2, idx, i, nx, sx, bc_x, ph_x, px);
        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));
__MASK1__
        pml_apply(f1, u1, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Target 2: curl_z = d(g1)/dx - d(g0)/dy; dsig=x, dsigu=y; mask __AXES2__.
    {
        cf f_1 = cf_load(g1, idx);
        cf sf = __SHIFT__(g1, idx, i, nx, sx, bc_x, ph_x, px);
        cf f_2 = cf_load(g0, idx);
        cf ss = __SHIFT__(g0, idx, j, ny, sy, bc_y, ph_y, py);
        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));
__MASK2__
        pml_apply(f2, u2, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }
}
'''

#: The constitutive kernels. ``__SCALE__`` is the E side's D*inv_eps product, empty
#: on the H side where the source is B itself (mu = 1 is already baked into
#: ``H_CONSTITUTIVE_TERMS``; stepping.py:949 passes ``getattr(fields, source)``).
_CONSTITUTIVE_TEMPLATE = r'''
extern "C" __global__ void __NAME__(
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
__EPS_PARAMS__    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // NO ghost rule, NO ownership mask and NO PHASE, and all three are
    // transcribed rather than forgotten: _apply_constitutive_pml writes every
    // cell of the volume, reads no neighbour, and the constitutive sub-step never
    // sees the wrap (stepping.py:907-993). Anyone porting the curl kernel's shape
    // into this one will reach for all three. None belongs here.
    //
    // Component 0 takes its coefficient pair from axis x, 1 from y, 2 from z --
    // MEEP's dsigw, the absorption a component accumulates along the direction it
    // points in (H_/E_CONSTITUTIVE_TERMS, stepping.py:226-227). It is NOT the
    // dsig/dsigu cycle the curl uses, and confusing the two is a smooth,
    // converged, entirely wrong absorber.
    cf s0 = cf_load(g0, idx);
    cf s1 = cf_load(g1, idx);
    cf s2 = cf_load(g2, idx);
__SCALE__
    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i]);
    constitutive_apply(f1, w1, idx, s1, kps_y[j], kms_y[j]);
    constitutive_apply(f2, w2, idx, s2, kps_z[k], kms_z[k]);
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 226-227->227-228

#: The E side's extra parameters. NOT ``__restrict__``, deliberately: an isotropic
#: run hands the same device pointer three times (fields.py:1323-1325), and
#: restrict on mutually aliasing arguments is a promise the caller cannot keep.
#: They are read-only, so nothing is lost but the promise. They are float32
#: VOLUMES indexed by the COMPLEX CELL index -- one real coefficient per cell,
#: shared by both planes -- and NEVER word-doubled: ``inv_eps`` stays float32 under
#: complex storage (stepping.py:41-50 against fields.py:1203-1204).
_EPS_PARAMS = ("    const float* inv_eps_0, const float* inv_eps_1,\n"
               "    const float* inv_eps_2,\n")

#: The E side's source product. D on the LEFT (stepping.py:1011-1013,
#: ``source * fields.inverse_epsilon_for(component)``).
_SCALE_BODY = """    s0 = mul_field_left(s0, inv_eps_0[idx]);
    s1 = mul_field_left(s1, inv_eps_1[idx]);
    s2 = mul_field_left(s2, inv_eps_2[idx]);
"""

#: The ownership mask, per target and per side. ``stepping._mask_non_owned_cells``
#: (:1865-1902) zeroes cell 0 of every axis on which the target's Yee shift is 0
#: and which is metallic; a mirror and the cylindrical axis are refused by the
#: predicate, so METALLIC is the only live branch. ``fields.IYEE_SHIFTS``
#: (fields.py:214-219): Bx (0,1,1), By (1,0,1), Bz (1,1,0);
#: Dx (1,0,0), Dy (0,1,0), Dz (0,0,1). So each B target masks on ONE axis and each
#: D target on TWO -- which is the asymmetry a reader should check by eye.
_MASK_AXES: Dict[bool, Tuple[Tuple[str, ...], ...]] = {
    False: (("x",), ("y",), ("z",)),            # step_B: Bx, By, Bz
    True: (("y", "z"), ("x", "z"), ("x", "y")),  # step_D: Dx, Dy, Dz
}

_AXIS_INDEX = {"x": "i", "y": "j", "z": "k"}


def _mask_lines(axes: Sequence[str]) -> str:
    """The ``curl = cf_zero()`` guards for one target, one per masked axis."""
    return "\n".join(
        f"        if (bc_{axis} == BC_METALLIC && {_AXIS_INDEX[axis]} == 0) "
        f"curl = cf_zero();" for axis in axes)


def normalized_expansion(expansion) -> int:
    """The arm as its integer code, from a code or a name; refuses anything else.

    A WRONG ARM IS A WRONG ANSWER, not a crash: both arms compile, both run, and
    they differ in the last bits of about a quarter of the words. So this refuses
    by name rather than defaulting, and there is no default anywhere in this
    family -- ``complex_source`` requires the argument.
    """
    if isinstance(expansion, str):
        if expansion not in EXPANSIONS:
            raise ValueError(
                f"expansion must be one of {sorted(EXPANSIONS)}, got "
                f"{expansion!r}; the arm is a measured platform fact and there is "
                f"no default")
        return EXPANSIONS[expansion]
    if isinstance(expansion, bool) or not isinstance(expansion, int):
        raise ValueError(
            f"expansion must be an arm name or its integer code, got "
            f"{expansion!r}")
    if expansion not in EXPANSION_NAMES:
        raise ValueError(
            f"expansion code {expansion!r} is not one of "
            f"{sorted(EXPANSION_NAMES)}")
    return expansion


def complex_source(sub_step: str, expansion) -> str:
    """The device source for one sub-step under one expansion arm.

    ``sub_step`` is a key of :data:`KERNELS`: ``step_B``, ``step_D``, ``update_H``
    or ``update_E``. ``expansion`` is an arm name or its code -- required, never
    defaulted.
    """
    if sub_step not in KERNELS:
        raise ValueError(
            f"sub_step must be one of {sorted(KERNELS)}, got {sub_step!r}")
    arm = normalized_expansion(expansion)
    name, backward, scaled = KERNELS[sub_step]
    prelude = _HEAD + _ARM_SOURCE[arm] + _TAIL
    if sub_step in CURL_SUB_STEPS:
        body = _CURL_TEMPLATE.replace("__NAME__", name)
        body = body.replace("__SHIFT__", "cshift_dn" if backward else "cshift_up")
        for target, axes in enumerate(_MASK_AXES[backward]):
            # The AXES placeholder is a comment and the MASK placeholder is code;
            # they are two names because one substitution serving both would splice
            # a multi-line guard into the middle of a // comment and silently
            # comment out the first guard.
            body = body.replace(f"__AXES{target}__", "+".join(axes))
            body = body.replace(f"__MASK{target}__", _mask_lines(axes))
        return prelude + body
    body = _CONSTITUTIVE_TEMPLATE.replace("__NAME__", name)
    body = body.replace("__EPS_PARAMS__", _EPS_PARAMS if scaled else "")
    body = body.replace("__SCALE__", _SCALE_BODY if scaled else "")
    return prelude + body


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Four sub-steps times two arms is eight sources; a single changed character
    anywhere in this module moves this value. The per-source digests of the arm a
    gate actually launches belong in that gate's artifact, which is where a
    certification record reads them from.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for sub_step in sorted(KERNELS):
        for name in sorted(EXPANSIONS):
            digest.update(f"{sub_step}|{name}".encode("ascii"))
            digest.update(complex_source(sub_step, name).encode("utf-8"))
    return digest.hexdigest()


def shipped_kernel_names(source: str):
    """Every kernel NVRTC could be asked to compile in ``source``, from the text."""
    import re  # noqa: PLC0415 - stdlib, imported at the one call site

    return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))
