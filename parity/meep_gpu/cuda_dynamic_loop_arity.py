"""THE DYNAMIC-LOOP GATE's subject: one sub-step, written twice, one axis apart.

WHAT THIS MEASURES, AND WHY IT IS WORTH A DEVICE LEG THAT BUYS NO COVERAGE.

Every hand-CUDA family after the constitutive pair has to decide what to do with
the *arity* axes — the compile-time constants that control HOW MANY TERMS an
unrolled inner sum has (``NP*`` pole count, ``R**`` row count, ``DRIVEN*``,
``ZERO_ROWS``). The *branch* axes are already settled and cost nothing: the
shipped ``step_B_pml_real`` takes ``int bc_x, bc_y, bc_z`` as ordinary
runtime arguments (``cuda_kernels/step_curl_kernels.py:1687``) and is certified.
An arity axis is different in kind. Folding it into a runtime argument turns a
fully-unrolled sum into a DYNAMIC LOOP, and a loop and an unrolled sum can differ
in the ASSOCIATION ORDER of their additions. Float addition is not associative,
so on a track whose entire currency is bit-identity that is a CORRECTNESS
question, not a performance one.

It had never been measured. This module is the subject of the measurement:
the same sub-step compiled two ways, differing in nothing but that.

THE AXIS IS ``NP`` — THE PER-COMPONENT LIVE POLE COUNT — AND IT WAS CHOSEN BY
COUNTING THE CORPUS, not by convenience. From the 210 rows of
``results/cuda_predicate_coverage_2026-08-15_constitutive/{examples,tests,
tests_param_matched}.jsonl``, read through each row's ``polarization`` record
(the per-state ``driven`` component list, which is
``PolarizationState._driven`` — dispersion.py:640-642 — so it is the LIVE count,
not the declared one):

    NP per component   rows   what drives it
    ----------------   ----   -------------------------------------------------
             1            5   absorbed_power_density.py, and the four 3-D
                              TestLoadDump rows (one Lorentzian)
             2            1   material-dispersion.py (two Lorentzians)
             5            6   absorber-1d.py, TestAbsorber.test_absorber, and
                              the four 2-D TestLoadDump rows
             6            3   the three stochastic_emitter scripts (Ag: 1 Drude
                              + 5 Lorentzian)

Four distinct arities over 15 corpus rows, and the count is uniform across
Ex/Ey/Ez in every one of them (the measured triples are exactly (1,1,1), (2,2,2),
(5,5,5) and (6,6,6)). That is a REAL axis whose arity REALLY VARIES — a synthetic
axis nothing drives would make this measurement worthless — and the sweep below
covers 0..8 so the answer is not read off two points.

IT IS ALSO THE AXIS WHOSE SUM IS ALREADY KNOWN TO BE ORDER-SENSITIVE, which is
what makes it the honest choice rather than the flattering one. The sibling
Triton module's recon measured, on real Grid/Fields/PML cases:

* ``D - (P0 + P1)`` instead of ``((D - P0) - P1)`` — caught 20/20 at two poles,
  0/10 at one pole (``triton_kernels/dispersive_update_e.py``, control 1);
* ``((D - P1) - P0)`` — the same terms, reordered — caught 20/20 at two poles.

So the quantity under test is one where regrouping and reordering DO move bits.
A "bit-identical" verdict here therefore cannot be explained by the sum being
insensitive; the controls below re-establish that on this leg's own operands
rather than inheriting it.

WHAT IS TRANSCRIBED, AND FROM WHERE. The sub-step is ``stepping.update_E``
(stepping.py:954) with an active PML and at least one registered susceptibility.
Per component the array path runs:

    source       = fields.displacement_minus_polarization(c)   stepping.py:1010
    constitutive = source * fields.inverse_epsilon_for(c)      stepping.py:1011
    kps, kms     = _constitutive_coefficients(pml, axis, half_integer=True)
                                                               stepping.py:1015
    _apply_constitutive_pml(E_c, constitutive, kps, kms, f_w_c)
                                                               stepping.py:1016

and ``displacement_minus_polarization`` (fields.py:1079-1105) is a COPY of D
followed by one in-place subtraction per driving state, in
``fields.polarizations`` order:

    scratch[...] = D_c                                         fields.py:1102
    for state in contributors: state.subtract_into(c, scratch) fields.py:1103-1104
    #   PolarizationState.subtract_into is `target -= self.P[component]`
    #                                                          dispersion.py:693

with ``_apply_constitutive_pml`` (stepping.py:2112-2143) being

    fw_previous = fw.copy(); fw[...] = source
    field += kps * fw ; field -= kms * fw_previous

i.e. TWO SEPARATE ACCUMULATIONS, ``prev`` read before the store. So the contract,
per component, is exactly:

    s    = ((D_c - P_0[c]) - P_1[c]) - ...      left to right, NP terms
    src  = s * inv_eps_c                        D-minus-P on the LEFT
    prev = f_w_c ; f_w_c = src
    E_c  = (E_c + kps_h[a]*src) - kms_h[a]*prev a = the component's OWN axis

The ARITY AXIS IS THE ``s`` LINE AND NOTHING ELSE. Everything downstream of it is
byte-for-byte the certified ``constitutive_apply`` from
``cuda_kernels/constitutive_kernels.py``, reproduced here rather than imported so
that this file compiles standalone on a host that loads it by path — and pinned
equal to that module's text by ``test_cuda_dynamic_loop_arity.py``, so the two
cannot drift into being different measurements.

=============================================================================
THE VARIANTS
=============================================================================

``unrolled``            NP is a compile-time ``#define``; the subtraction chain is
                        written out. 24 pole pointers, unused slots bound to the
                        component's own D and never read. THIS IS WHAT THE
                        SHIPPED KERNELS DO, and one NVRTC compile per (NP0,NP1,
                        NP2) triple is precisely the "variant cache" the
                        reversal condition prices.
``dynamic``             NP is ``int np0, np1, np2``; the chain is a
                        ``for (int p = 0; p < np; ++p) s = s - P[p][idx];`` over a
                        DEVICE ARRAY OF POINTERS. One compile covers every arity.
``dynamic_notrips``     ``dynamic`` with the trip-count store removed. The NULL
                        that proves the counter is inert on the arithmetic.
``dynamic_pragma``      ``dynamic`` with ``#pragma unroll`` on the loop. Asks the
                        compiler to unroll a runtime-bounded loop, which is the
                        likeliest place a reassociation could enter.

and three CONTROLS, each of which MUST be caught by the same byte comparison
that reports the verdict — a gate that cannot fail certifies nothing:

``dynamic_short``       the loop runs ``np - 1`` times. Armed at NP >= 1. This is
                        the "one iteration short" control the trip-count claim
                        needs.
``dynamic_reversed``    ``for (int p = np - 1; p >= 0; --p)`` — the SAME count of
                        the SAME operations in the opposite ASSOCIATION ORDER.
                        Armed at NP >= 2. This is the sharpest control on the
                        board: if reversed is caught and forward is not, the
                        comparison is provably sensitive to exactly the property
                        under test, at exactly this arity, on exactly these
                        operands.
``dynamic_presummed``   ``acc = P0 + P1 + ...; s = D - acc``. Armed at NP >= 2.
                        The defect the sibling track caught 20/20.

=============================================================================
OPERAND CLASSES -- AND A DESIGN ASSUMPTION THAT MEASUREMENT OVERTURNED
=============================================================================

This module was first written with a purpose-built ``wide_exponent`` class as
the headline, on the argument that terms spread over many decades "absorb and
cancel, and that is what makes association order bite". THAT ARGUMENT IS WRONG,
and it was caught by measuring it in float32 NumPy on a laptop before any device
time was spent. Forward-vs-reversed and forward-vs-presummed disagreement as a
fraction of OUTPUT words -- the quantity the gate compares, not the pole sum --
shape 16x16x24, one seed per arity:

    class            NP=2          NP=3          NP=5          NP=6
                  rev / pre     rev / pre     rev / pre     rev / pre
    uniform      .234 / .235   .319 / .312   .436 / .407   .460 / .435
    wide (8 dec) .068 / .068   .125 / .104   .223 / .136   .265 / .143
    subnormal    .058 / .057   .116 / .095   .219 / .134   .260 / .151

``uniform(-1, 1)`` -- the sibling gates' plain class, the one the wide class was
invented to improve on -- is the MOST order-sensitive at every arity, by a
factor of three or more at NP=2. Two further candidates were measured on the
pole sum and also lost to it: a narrow 1.5-decade spread and a deliberately
cancelling class of near-equal magnitudes with alternating signs.

THE REASON, and it is worth writing down because it recurs on the folded and
complex families: association order bites on CANCELLATION, not on ABSORPTION.
Once one addend dominates another by more than 24 bits, adding the small one is
a no-op in ANY order, so a wide spread of exponents makes most of the sum order-
INSENSITIVE. Terms of comparable magnitude with mixed signs cancel, the partial
sum loses its leading bits, and where the next addend's low bits land then
depends on when it arrived. The class that looks adversarial is the weak one
here, and the plain one is the strong one.

The headline is therefore ``uniform``, chosen by that table rather than by
argument. The other two are kept as coverage, not as the verdict:

* ``wide_exponent`` exercises the ABSORPTION regime, which is real and which the
  corpus reaches, and where a reassociation would be hardest to see. It is the
  weaker discriminator and the headline does not rest on it.
* ``subnormal_band`` is the sibling gates' policy class, reported SEPARATELY and
  never merged into the headline, for the reason those modules give at length:
  in that band it is the ARRAY PATH that leaves IEEE (CuPy appends
  ``-ftz=true`` to every NVRTC compile), so a divergence there is not a kernel
  disagreement. Here both sides are kernels under one option tuple, so the class
  is still informative -- but it is kept separate so this leg's headline means
  what the other legs' headlines mean.

THE VACUITY FLOOR is stated on the headline class: every ARMED control must be
caught on at least :data:`CONTROL_CATCH_FLOOR` of the ELIGIBLE words, or the leg
is reporting on operands that cannot tell two association orders apart and its
"identical" means nothing. ELIGIBLE, not all, and that distinction matters for
the asymmetric triples: a control armed at NP >= 2 can only move the components
whose OWN arity is at least 2, so scoring it across all six output arrays would
manufacture a failure out of a case like (0, 1, 2). The eligible set is computed
per case and recorded in the artifact.

NOTHING HERE IS A SHIPPED KERNEL. This module compiles no coverage, exports no
predicate, and is imported by no engine path. It exists to answer one question
about the cost model of tasks #54 and the complex family.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)


# ---------------------------------------------------------------------------
# The axis, measured
# ---------------------------------------------------------------------------

#: The per-component live pole counts the 210-row corpus census ACTUALLY drives,
#: with the number of rows at each. Read out of the coverage artifacts named in
#: the module docstring; ``test_cuda_dynamic_loop_arity.py`` re-derives it from
#: those files so this constant cannot go stale silently.
CORPUS_ARITIES: Dict[int, int] = {1: 5, 2: 1, 5: 6, 6: 3}

#: The most poles one component may carry — the sibling Triton module's
#: ``MAX_POLES`` (``triton_kernels/dispersive_update_e.py``), kept equal because
#: the unrolled variant below is that kernel's chain in CUDA and a different
#: ceiling would be a different kernel.
MAX_POLES = 8

#: The full sweep. 0 is included because it is the degenerate arity — the loop
#: must not execute and the body must reduce to the non-dispersive form — and
#: 3, 4, 7, 8 because a two-point answer is not an answer: the divergence this
#: gate is looking for may appear only past some length.
SWEEP_ARITIES: Tuple[int, ...] = tuple(range(MAX_POLES + 1))

#: Shapes. Two are not multiples of the 256-lane block, so the tail guard runs;
#: one is 1-D-like and one 2-D-like, which is what 54 of the 57 corpus scripts
#: declare; and the largest is big enough that a per-word verdict is a real
#: sample rather than an anecdote.
SWEEP_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 2048),        # 1-D, block-aligned
    (1, 24, 40),         # 2-D, 960 elements: the tail guard runs
    (7, 13, 29),         # fully oblique, 2639 elements: tail guard, odd strides
    (16, 16, 24),        # 3-D, 6144 elements
)


# ---------------------------------------------------------------------------
# Operand classes
# ---------------------------------------------------------------------------

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "wide_exponent", "subnormal_band")

#: The class the headline verdict is read from, CHOSEN BY THE TABLE IN THE MODULE
#: DOCSTRING and not by argument: ``uniform`` separates association orders three
#: times more often than the purpose-built wide class at NP=2, and more often at
#: every other arity too.
HEADLINE_VALUE_CLASS = "uniform"

#: Exact zero, negative zero, the smallest normal float32, the largest subnormal
#: and the smallest subnormal in both signs — the sibling probe's needles
#: (``probe_fused_kernel_bit_identity.SUBNORMAL_NEEDLES``), which
#: ``uniform(-1, 1)`` provably never draws.
SUBNORMAL_NEEDLES = np.array(
    [0.0, -0.0, 1.1754944e-38, -1.1754944e-38, 1.1754942e-38, 1e-45, -1e-45],
    dtype=np.float32)

#: The fraction of ELIGIBLE words at which an armed control must be caught before
#: the headline is allowed to mean anything.
#:
#: SET FROM THE OUTPUT-LEVEL MEASUREMENT, AND THAT CORRECTION IS THE POINT.
#: This floor was first set at 0.25 from a laptop calibration of the raw pole sum
#: ``s``, which reads 0.324 at NP=2. The device leg then failed sixteen legs --
#: every NP=2 case on every shape, systematically -- because THE GATE DOES NOT
#: COMPARE ``s``. It compares E and f_w_E, and the arithmetic between them
#: collapses differences:
#:
#:     NP=2, uniform, 16x16x24, forward vs reversed
#:       the sum s                                    0.324
#:       f_w = s * inv_eps                            0.301   the multiply rounds
#:                                                            distinct s together
#:       E   = (E + kps*src) - kms*prev               0.158   two accumulations
#:                                                            against a same-size E
#:       both, which is what the gate scores           0.229
#:
#: THE TRANSFERABLE LESSON, and it applies to every mutation on this project: a
#: control's catch rate must be calibrated on THE QUANTITY THE GATE COMPARES, not
#: on the intermediate the mutation targets. Calibrating on the intermediate
#: overstated this gate's discrimination by 41% at its weakest armed arity.
#:
#: 0.20 is the corrected floor. The measured minimum over the whole device sweep
#: at NP=2 is 0.2283 and every NP>=3 case is above 0.32, so the floor binds only
#: where the discrimination is genuinely weakest and still leaves margin.
CONTROL_CATCH_FLOOR = 0.20

#: An ABSOLUTE floor beside the fractional one, because a fraction of a small
#: case is a small number of words and "a fifth of sixty" is not evidence. Every
#: shape in the sweep clears this comfortably; it exists so that adding a tiny
#: shape later cannot quietly weaken the claim.
CONTROL_CATCH_MINIMUM_WORDS = 200

#: The fraction of live words the SUBJECT must move away from its input before a
#: leg counts. Zero-init is a fixed point of a constitutive sub-step, so a leg
#: that compared two kernels which both wrote nothing would report a perfect
#: match. Checked per case, in every variant, against the pre-launch E and f_w.
#:
#: PER CLASS, because the classes are not equally movable and one number would be
#: either toothless on the normal classes or a false failure on the band. In the
#: subnormal band a product can underflow to a zero the input already held, so a
#: word can legitimately fail to move; on the normal classes it effectively
#: cannot, and a launch that left a tenth of its output alone there is broken.
MOVED_WORD_FLOOR_BY_CLASS: Dict[str, float] = {
    "uniform": 0.90,
    "wide_exponent": 0.90,
    "subnormal_band": 0.40,
}


def _band_draw(shape: Tuple[int, int, int], rng) -> np.ndarray:
    """One array seeded across the float32 subnormal band, needles forced in.

    The sibling probe's ``subnormal_band_hosts`` draw, one array at a time. Kept
    to the same two-pass shape (magnitudes, then one cell in sixteen forced to a
    needle) so the two legs' band class is the same class.
    """
    exponents = rng.uniform(-45.0, -30.0, size=shape)
    signs = np.where(rng.integers(0, 2, size=shape) == 0, -1.0, 1.0)
    values = (signs * np.power(10.0, exponents)).astype(np.float32)
    picks = rng.integers(0, 16, size=shape) == 0
    choice = SUBNORMAL_NEEDLES[rng.integers(0, SUBNORMAL_NEEDLES.size, size=shape)]
    return np.where(picks, choice, values).astype(np.float32)


def _wide_draw(shape: Tuple[int, int, int], rng, decades: float = 8.0) -> np.ndarray:
    """One array across ~2*``decades`` of exponent, mixed signs, no subnormals.

    THE POINT OF THE CLASS. A sum of terms that all share an exponent is nearly
    order-insensitive: the partial sums stay in one binade and every regrouping
    rounds the same way. Spreading the terms across many decades makes each
    partial sum absorb the small terms differently depending on when they arrive,
    which is exactly the sensitivity this gate needs its operands to have.

    Bounded to normal numbers on purpose: the subnormal band is its own class and
    mixing the two would make a caught control unattributable.
    """
    exponents = rng.uniform(-decades, decades, size=shape)
    signs = np.where(rng.integers(0, 2, size=shape) == 0, -1.0, 1.0)
    return (signs * np.power(10.0, exponents)).astype(np.float32)


def _uniform_draw(shape: Tuple[int, int, int], rng) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def draw(shape: Tuple[int, int, int], rng, value_class: str) -> np.ndarray:
    if value_class == "uniform":
        return _uniform_draw(shape, rng)
    if value_class == "wide_exponent":
        return _wide_draw(shape, rng)
    if value_class == "subnormal_band":
        return _band_draw(shape, rng)
    raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")


def operand_census(arrays: Dict[str, np.ndarray]) -> Dict[str, int]:
    """How many subnormals, signed zeros and exponent decades a leg REALLY holds.

    A leg whose operands cannot contain the class it claims to test measures
    nothing and reports a pass while doing it. The exponent SPREAD is in here
    because it is this leg's own precondition: a wide class that came out narrow
    would silently become the uniform class.
    """
    census = {"subnormal": 0, "negative_zero": 0, "zero": 0, "n": 0}
    smallest_normal = np.float32(1.1754944e-38)
    lo, hi = np.inf, 0.0
    for array in arrays.values():
        flat = np.asarray(array, dtype=np.float32).ravel()
        census["n"] += int(flat.size)
        magnitude = np.abs(flat)
        nonzero = magnitude[magnitude > 0]
        census["subnormal"] += int(np.count_nonzero(nonzero < smallest_normal))
        census["zero"] += int(np.count_nonzero(flat == 0.0))
        census["negative_zero"] += int(np.count_nonzero(
            np.signbit(flat) & (flat == 0.0)))
        if nonzero.size:
            lo = min(lo, float(nonzero.min()))
            hi = max(hi, float(nonzero.max()))
    census["exponent_decades"] = (
        0.0 if hi == 0.0 or not np.isfinite(lo) else
        float(np.log10(hi) - np.log10(lo)))
    return census


# ---------------------------------------------------------------------------
# The device sources
# ---------------------------------------------------------------------------
#
# ONE PRELUDE, TWO SIGNATURES, SEVEN BODIES. The prelude is the certified
# constitutive apply, character for character; the two signatures differ only in
# how the pole volumes arrive; the bodies differ only in the `s` line.
#
# PURE ASCII IS A COMPILE REQUIREMENT ON THIS PATH, not a style rule.
# ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a bare
# ``open(..., 'w')`` (compiler.py:368), so the bytes go through the interpreter's
# LOCALE encoding — ASCII under C/POSIX, which is what a non-interactive shell on
# the validation host gets. Two em-dashes in a comment killed the constitutive E
# kernel at its first launch on 2026-08-15. The test scans these strings and
# performs the encode itself.

_PRELUDE = r'''
// stepping._apply_constitutive_pml (stepping.py:2065), MEEP's step_update_EDHB
// with dsigw active:
//
//     realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
//     fw[i] = g[i] * u[i];                   // B for H, D*inv_eps for E
//     f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;
//
// The two accumulations are kept SEPARATE and LEFT-TO-RIGHT; `prev` is loaded
// before the store. This is the certified body from
// cuda_kernels/constitutive_kernels.py::_REAL_CONSTITUTIVE_PRELUDE, reproduced
// rather than imported so this file compiles standalone, and pinned equal to it
// by test_cuda_dynamic_loop_arity.py.
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

#: The linear-index decomposition, identical in every variant. Written once so a
#: divergence can never be attributed to two spellings of ``i``, ``j``, ``k``.
_INDEX_BLOCK = r'''
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
'''

_UNROLLED_TEMPLATE = r'''
#define NP0 __NP0__
#define NP1 __NP1__
#define NP2 __NP2__

extern "C" __global__ void dispersive_update_E_unrolled(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    const float* __restrict__ Dx, const float* __restrict__ Dy,
    const float* __restrict__ Dz,
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    const float* __restrict__ Px0, const float* __restrict__ Px1,
    const float* __restrict__ Px2, const float* __restrict__ Px3,
    const float* __restrict__ Px4, const float* __restrict__ Px5,
    const float* __restrict__ Px6, const float* __restrict__ Px7,
    const float* __restrict__ Py0, const float* __restrict__ Py1,
    const float* __restrict__ Py2, const float* __restrict__ Py3,
    const float* __restrict__ Py4, const float* __restrict__ Py5,
    const float* __restrict__ Py6, const float* __restrict__ Py7,
    const float* __restrict__ Pz0, const float* __restrict__ Pz1,
    const float* __restrict__ Pz2, const float* __restrict__ Pz3,
    const float* __restrict__ Pz4, const float* __restrict__ Pz5,
    const float* __restrict__ Pz6, const float* __restrict__ Pz7,
    int* __restrict__ trips,
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z
) {
__INDEX__
    // fields.displacement_minus_polarization (fields.py:1102-1104): a copy of D
    // followed by one in-place subtraction per driving state, left to right, in
    // fields.polarizations order. NOT pre-summed: D - (P0 + P1) is a different
    // float32 number and the sibling track caught it 20/20 at two poles.
    float sx = Dx[idx];
#if NP0 > 0
    sx = sx - Px0[idx];
#endif
#if NP0 > 1
    sx = sx - Px1[idx];
#endif
#if NP0 > 2
    sx = sx - Px2[idx];
#endif
#if NP0 > 3
    sx = sx - Px3[idx];
#endif
#if NP0 > 4
    sx = sx - Px4[idx];
#endif
#if NP0 > 5
    sx = sx - Px5[idx];
#endif
#if NP0 > 6
    sx = sx - Px6[idx];
#endif
#if NP0 > 7
    sx = sx - Px7[idx];
#endif

    float sy = Dy[idx];
#if NP1 > 0
    sy = sy - Py0[idx];
#endif
#if NP1 > 1
    sy = sy - Py1[idx];
#endif
#if NP1 > 2
    sy = sy - Py2[idx];
#endif
#if NP1 > 3
    sy = sy - Py3[idx];
#endif
#if NP1 > 4
    sy = sy - Py4[idx];
#endif
#if NP1 > 5
    sy = sy - Py5[idx];
#endif
#if NP1 > 6
    sy = sy - Py6[idx];
#endif
#if NP1 > 7
    sy = sy - Py7[idx];
#endif

    float sz = Dz[idx];
#if NP2 > 0
    sz = sz - Pz0[idx];
#endif
#if NP2 > 1
    sz = sz - Pz1[idx];
#endif
#if NP2 > 2
    sz = sz - Pz2[idx];
#endif
#if NP2 > 3
    sz = sz - Pz3[idx];
#endif
#if NP2 > 4
    sz = sz - Pz4[idx];
#endif
#if NP2 > 5
    sz = sz - Pz5[idx];
#endif
#if NP2 > 6
    sz = sz - Pz6[idx];
#endif
#if NP2 > 7
    sz = sz - Pz7[idx];
#endif

    // stepping.py:982 - source * inv_eps, D on the LEFT. Three inverse-epsilon
    // pointers, one per component (Fields.inverse_epsilon_for); NOT __restrict__,
    // because an isotropic run hands the same device pointer three times.
    float src_x = sx * inv_eps_Ex[idx];
    float src_y = sy * inv_eps_Ey[idx];
    float src_z = sz * inv_eps_Ez[idx];

    // The trip count this variant WOULD have run, so the two variants report the
    // same quantity through the same output. A store to a disjoint int array;
    // it cannot reach a float operand.
    trips[idx] = NP0 + NP1 + NP2;

    // Ex takes its half-integer coefficients from axis x, Ey from y, Ez from z:
    // the component's OWN axis (stepping.E_CONSTITUTIVE_TERMS, stepping.py:227).
    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);
    constitutive_apply(Ey, f_w_Ey, idx, src_y, kps_y[j], kms_y[j]);
    constitutive_apply(Ez, f_w_Ez, idx, src_z, kps_z[k], kms_z[k]);
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011, 227->228

_DYNAMIC_TEMPLATE = r'''
extern "C" __global__ void dispersive_update_E_dynamic(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    const float* __restrict__ Dx, const float* __restrict__ Dy,
    const float* __restrict__ Dz,
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    const float* const* __restrict__ Px,
    const float* const* __restrict__ Py,
    const float* const* __restrict__ Pz,
    int np0, int np1, int np2,
    int* __restrict__ trips,
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z
) {
__INDEX__
    int executed = 0;

    // The SAME chain as the unrolled variant, with the count as a runtime bound
    // and the pole volumes reached through a device array of pointers. That
    // indirection is forced: eight named arguments cannot be indexed by a
    // runtime p. It moves ADDRESSES, not values -- the float32 loaded at each
    // step is the same word from the same array in the same order.
    float sx = Dx[idx];
__LOOP_X__

    float sy = Dy[idx];
__LOOP_Y__

    float sz = Dz[idx];
__LOOP_Z__

    float src_x = sx * inv_eps_Ex[idx];
    float src_y = sy * inv_eps_Ey[idx];
    float src_z = sz * inv_eps_Ez[idx];

__TRIPS__
    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);
    constitutive_apply(Ey, f_w_Ey, idx, src_y, kps_y[j], kms_y[j]);
    constitutive_apply(Ez, f_w_Ez, idx, src_z, kps_z[k], kms_z[k]);
}
'''


def _forward_loop(component: str, bound: str, pragma: str = "") -> str:
    return (f"{pragma}    for (int p = 0; p < {bound}; ++p) {{\n"
            f"        s{component} = s{component} - P{component}[p][idx];\n"
            f"        ++executed;\n"
            f"    }}")


def _reversed_loop(component: str, bound: str) -> str:
    return (f"    for (int p = {bound} - 1; p >= 0; --p) {{\n"
            f"        s{component} = s{component} - P{component}[p][idx];\n"
            f"        ++executed;\n"
            f"    }}")


def _presummed_loop(component: str, bound: str) -> str:
    return (f"    float acc_{component} = 0.0f;\n"
            f"    for (int p = 0; p < {bound}; ++p) {{\n"
            f"        acc_{component} = acc_{component} + P{component}[p][idx];\n"
            f"        ++executed;\n"
            f"    }}\n"
            f"    s{component} = s{component} - acc_{component};")


#: The loop body each dynamic variant substitutes, as a callable of
#: (component letter, runtime bound expression). Named as data so the test can
#: assert the set has not quietly lost a control.
_DYNAMIC_BODIES = {
    "dynamic": lambda c, b: _forward_loop(c, b),
    "dynamic_notrips": lambda c, b: _forward_loop(c, b),
    "dynamic_pragma": lambda c, b: _forward_loop(c, b, "#pragma unroll\n"),
    "dynamic_short": lambda c, b: _forward_loop(c, f"({b} - 1)"),
    "dynamic_reversed": lambda c, b: _reversed_loop(c, b),
    "dynamic_presummed": lambda c, b: _presummed_loop(c, b),
}

#: Which variants are the SUBJECT of the verdict and which are controls that MUST
#: be caught. ``dynamic_notrips`` is neither: it is the null that proves the trip
#: counter does not perturb the arithmetic.
SUBJECT_VARIANTS: Tuple[str, ...] = ("dynamic", "dynamic_pragma")
NULL_VARIANTS: Tuple[str, ...] = ("dynamic_notrips",)
CONTROL_VARIANTS: Tuple[str, ...] = (
    "dynamic_short", "dynamic_reversed", "dynamic_presummed")

#: The arity at or above which each control is ARMED. Below it the control is
#: mathematically the same computation and a "not caught" is the correct answer,
#: not a miss -- reporting it as a miss is how a gate acquires a fake failure.
#: ``dynamic_short`` differs as soon as there is one term to drop.
#: ``dynamic_reversed`` and ``dynamic_presummed`` are exact identities at NP <= 1.
CONTROL_ARMED_AT: Dict[str, int] = {
    "dynamic_short": 1,
    "dynamic_reversed": 2,
    "dynamic_presummed": 2,
}


def unrolled_source(np0: int, np1: int, np2: int) -> str:
    """The fully-unrolled kernel at one (NP0, NP1, NP2) triple.

    The arity is a ``#define`` IN THE SOURCE rather than a ``-D`` in the option
    tuple, and that is deliberate: it keeps the compile options of the two
    variants LITERALLY IDENTICAL, so the only thing that differs between the two
    binaries under comparison is the body. It also makes the per-variant NVRTC
    compile visible for what it is -- the variant cache the reversal condition
    prices.
    """
    for value in (np0, np1, np2):
        if not 0 <= value <= MAX_POLES:
            raise ValueError(f"arity {value} outside 0..{MAX_POLES}")
    body = (_UNROLLED_TEMPLATE
            .replace("__NP0__", str(np0))
            .replace("__NP1__", str(np1))
            .replace("__NP2__", str(np2))
            .replace("__INDEX__", _INDEX_BLOCK))
    return _PRELUDE + body


def dynamic_source(variant: str) -> str:
    """One dynamic-loop variant. The arity never appears; it is a kernel argument."""
    if variant not in _DYNAMIC_BODIES:
        raise ValueError(f"unknown dynamic variant {variant!r}")
    make = _DYNAMIC_BODIES[variant]
    trips = "" if variant == "dynamic_notrips" else "    trips[idx] = executed;\n"
    body = (_DYNAMIC_TEMPLATE
            .replace("__INDEX__", _INDEX_BLOCK)
            .replace("__LOOP_X__", make("x", "np0"))
            .replace("__LOOP_Y__", make("y", "np1"))
            .replace("__LOOP_Z__", make("z", "np2"))
            .replace("__TRIPS__", trips))
    return _PRELUDE + body


ALL_VARIANTS: Tuple[str, ...] = ("unrolled",) + tuple(_DYNAMIC_BODIES)


def source_for(variant: str, arities: Tuple[int, int, int]) -> str:
    if variant == "unrolled":
        return unrolled_source(*arities)
    return dynamic_source(variant)


def source_digest(text: str) -> str:
    """sha256 of a DEVICE STRING, encoded ASCII deliberately.

    The encode is a guard, not an implementation detail: a device string that
    cannot be encoded ASCII cannot be handed to NVRTC on a C-locale host, so
    digesting it should fail here rather than at first launch. Use
    :func:`file_digest` for a source FILE, whose prose is not so constrained --
    and which is where this function's first caller went wrong, dying on an
    em-dash in its own module docstring.
    """
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def file_digest(path: str) -> str:
    """sha256 of a file's BYTES. No encoding assumption, because none applies."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# ---------------------------------------------------------------------------
# Compile options
# ---------------------------------------------------------------------------

#: SPELLED HERE, not imported, for the reason the constitutive module gives:
#: loading a kernel module by path must not be able to pick up a different tuple
#: than the one the gate compiled under. ``test_cuda_dynamic_loop_arity.py``
#: pins this equal to ``constitutive_kernels._COMPILE_OPTIONS`` and to
#: ``step_curl_kernels._REAL_PML_COMPILE_OPTIONS``.
#:
#: ``--fmad=false`` is CORRECTNESS here even more directly than in the sibling
#: modules: the accumulations ``a + kps*src`` and ``a - kms*prev`` and the E
#: side's ``s * inv_eps`` are all contraction candidates, and an FMA rounds once
#: where the array path rounds twice. It also removes one confound from THIS
#: measurement -- a loop and an unrolled chain could otherwise be contracted
#: differently, and the verdict would be about contraction rather than about
#: association.
COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block. The sibling modules' 256, unchanged: one element per lane,
#: flat 1-D grid, element-wise sub-step.
THREADS = 256


# ---------------------------------------------------------------------------
# The array-path reference
# ---------------------------------------------------------------------------

def reference_update_E(state: Dict[str, np.ndarray],
                       arities: Tuple[int, int, int],
                       kps: Sequence[np.ndarray],
                       kms: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """``stepping.update_E`` under PML with poles, in float32 NumPy.

    Transcribed from the four lines cited in the module docstring, and VALIDATED
    against ``stepping.update_E`` itself on real ``Fields``/``PML``/
    ``PolarizationState`` objects by ``test_cuda_dynamic_loop_arity.py``. This
    function is not the gate's verdict -- the gate compares two KERNELS -- but a
    third leg that reports which of them, if either, also matches the array path
    is worth having for free, and a reference nobody validated is worth nothing.

    ``kps``/``kms`` are the three per-axis vectors ALREADY RESHAPED for
    broadcasting on their own axis, which is the shape ``PML._reshape_for_broadcast``
    stores them in.
    """
    components = (("Ex", "Dx", "Px", 0),
                  ("Ey", "Dy", "Py", 1),
                  ("Ez", "Dz", "Pz", 2))
    out: Dict[str, np.ndarray] = {}
    for index, (field, source_name, pole_prefix, axis) in enumerate(components):
        # fields.py:1102-1104 -- a copy, then one in-place subtraction per state.
        s = np.array(state[source_name], dtype=np.float32, copy=True)
        for pole in range(arities[index]):
            s -= state[f"{pole_prefix}{pole}"]
        src = (s * state["inv_eps_" + field]).astype(np.float32)
        fw_name = "f_w_" + field
        prev = np.array(state[fw_name], dtype=np.float32, copy=True)
        out[fw_name] = src
        # stepping.py:2133-2134 -- two separate accumulations, left to right.
        value = (state[field] + (kps[axis] * src)).astype(np.float32)
        value = (value - (kms[axis] * prev)).astype(np.float32)
        out[field] = value
    return out


def coefficient_tables(shape: Tuple[int, int, int], rng
                       ) -> Dict[str, List[np.ndarray]]:
    """Six per-axis kps/kms vectors, GRADED rather than uniform.

    A table of ones hides the coefficient indexing completely, so a kernel that
    read the wrong axis would still match. The tables are drawn per axis length
    and broadcast by the reference along that axis; the kernel indexes them with
    ``i``, ``j``, ``k``.
    """
    tables: Dict[str, List[np.ndarray]] = {}
    for name, base in (("kps", 1.0), ("kms", 0.5)):
        vectors = []
        for length in shape:
            values = (base + rng.uniform(0.05, 0.45, size=length)).astype(np.float32)
            vectors.append(values)
        tables[name] = vectors
    return tables


#: How ``PML._reshape_for_broadcast`` stores a per-axis vector: length on its own
#: axis, 1 on the other two.
BROADCAST_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (-1, 1, 1), (1, -1, 1), (1, 1, -1))


def broadcast_tables(vectors: Sequence[np.ndarray]) -> List[np.ndarray]:
    """The three per-axis vectors reshaped for broadcasting on their own axis."""
    return [np.asarray(vectors[axis], dtype=np.float32).reshape(BROADCAST_SHAPES[axis])
            for axis in range(3)]


def make_case(shape: Tuple[int, int, int], arities: Tuple[int, int, int],
              value_class: str, seed: int) -> Dict[str, Any]:
    """Every host array one case needs, plus its operand census.

    THE POLE VOLUMES ARE ALWAYS ALLOCATED AT FULL ``MAX_POLES``, whatever the
    arity, and are always filled with live values. Binding an unused slot to a
    zero volume would make the short control undetectable at the boundary and
    would make "the loop stopped early" look like "the loop read a zero".
    """
    rng = np.random.default_rng(seed)
    state: Dict[str, np.ndarray] = {}
    for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        state[name] = draw(shape, rng, value_class)
    # Inverse epsilon is a MATERIAL array, not a field: it is positive and O(1)
    # in every class, because a run whose 1/eps was subnormal is not a run.
    for name in ("inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez"):
        state[name] = rng.uniform(0.2, 1.0, size=shape).astype(np.float32)
    for prefix in ("Px", "Py", "Pz"):
        for pole in range(MAX_POLES):
            state[f"{prefix}{pole}"] = draw(shape, rng, value_class)

    tables = coefficient_tables(shape, rng)
    kps = broadcast_tables(tables["kps"])
    kms = broadcast_tables(tables["kms"])

    pole_names = [f"{prefix}{pole}" for prefix in ("Px", "Py", "Pz")
                  for pole in range(MAX_POLES)]
    census = operand_census({name: state[name] for name in
                             ["Dx", "Dy", "Dz", "Ex", "Ey", "Ez"] + pole_names})
    return {
        "shape": tuple(shape),
        "arities": tuple(arities),
        "value_class": value_class,
        "seed": seed,
        "state": state,
        "kps_vectors": tables["kps"],
        "kms_vectors": tables["kms"],
        "kps_broadcast": kps,
        "kms_broadcast": kms,
        "operand_census": census,
    }


def case_key(shape: Tuple[int, int, int], arities: Tuple[int, int, int],
             value_class: str) -> str:
    return (f"{shape[0]}x{shape[1]}x{shape[2]}|np={arities[0]},{arities[1]},"
            f"{arities[2]}|{value_class}")


def sweep_cases(shapes: Sequence[Tuple[int, int, int]] = SWEEP_SHAPES,
                arities: Sequence[int] = SWEEP_ARITIES,
                value_classes: Sequence[str] = VALUE_CLASSES,
                asymmetric: bool = True) -> List[Dict[str, Any]]:
    """The full product, as plain descriptors -- no arrays, no device, no seed use.

    Separated from :func:`make_case` so the case list can be counted, printed and
    asserted on a laptop before an hour of device time is spent on it.

    THE UNIFORM TRIPLE IS WHAT THE CORPUS ACTUALLY DRIVES -- every one of the 15
    dispersive rows has the same live pole count on Ex, Ey and Ez. A handful of
    ASYMMETRIC triples are added anyway, because an anisotropic sigma makes them
    reachable (dispersion.py:640-642 drops a component whose sigma is identically
    zero) and because a per-component bound is where a shared-loop-counter defect
    would hide.
    """
    cases: List[Dict[str, Any]] = []
    seed = 0
    for value_class in value_classes:
        for shape in shapes:
            for arity in arities:
                seed += 1
                cases.append({"shape": tuple(shape),
                              "arities": (arity, arity, arity),
                              "value_class": value_class,
                              "seed": 1000 + seed})
            if asymmetric:
                for triple in ((0, 1, 2), (6, 0, 3), (1, 5, 8), (8, 8, 1)):
                    seed += 1
                    cases.append({"shape": tuple(shape),
                                  "arities": triple,
                                  "value_class": value_class,
                                  "seed": 1000 + seed})
    return cases


# ---------------------------------------------------------------------------
# Byte comparison
# ---------------------------------------------------------------------------

OUTPUT_NAMES: Tuple[str, ...] = (
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The two output arrays each component owns, in component order. Used to score a
#: control on the components it can actually reach.
COMPONENT_OUTPUTS: Tuple[Tuple[str, str], ...] = (
    ("Ex", "f_w_Ex"), ("Ey", "f_w_Ey"), ("Ez", "f_w_Ez"))


def eligible_outputs(arities: Sequence[int], armed_at: int) -> Tuple[str, ...]:
    """The output arrays a control armed at ``armed_at`` can possibly move.

    A reordering or a regrouping of a component's pole sum cannot touch another
    component's arrays -- the sub-step is element-wise and per component -- so a
    control armed at NP >= 2 is simply INAPPLICABLE to a component whose arity is
    0 or 1. Scoring it there dilutes the catch rate with words it was never able
    to move, which on an asymmetric triple like (0, 1, 2) is the difference
    between a 0.33 catch rate and a 0.11 one, and therefore between a pass and a
    manufactured failure of the vacuity floor.
    """
    names: List[str] = []
    for index, (field, fw) in enumerate(COMPONENT_OUTPUTS):
        if arities[index] >= armed_at:
            names.extend((field, fw))
    return tuple(names)


def as_words(array: np.ndarray) -> np.ndarray:
    """A float32 array's UINT32 WORDS. NaN payloads and signed zeros included.

    ``==`` on float32 says 0.0 == -0.0 and says NaN != NaN. Neither is the
    question here; the question is whether the two kernels wrote the same bytes.
    """
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def _ordered(words: np.ndarray) -> np.ndarray:
    """float32 bit patterns mapped to a MONOTONE integer line.

    The standard total order: a non-negative pattern keeps its value, a negative
    one is reflected through ``INT32_MIN``. So an ulp distance means the same
    thing across the sign boundary as it does inside one binade.

    ``+0.0`` AND ``-0.0`` COLLAPSE TO THE SAME KEY, which is correct and is
    deliberately not worked around: they are the same number and zero ulps apart.
    The WORD comparison is what reports them as different, and it does; the ulp
    figure answers "by how much", to which the honest answer there is "by
    nothing, but the bytes differ".
    """
    signed = words.astype(np.int64) - np.int64(0x100000000) * (
        (words >> np.uint32(31)).astype(np.int64))
    return np.where(signed < 0, np.int64(-0x80000000) - signed, signed)


def compare_words(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray],
                  names: Sequence[str] = OUTPUT_NAMES) -> Dict[str, Any]:
    """Word-level disagreement between two output sets, per array and in total.

    Reports ULP distance as well as a count, because "how far apart" is the
    question the follow-up in the reversal condition turns on: a one-ulp
    disagreement and a catastrophic one call for different answers.
    """
    total = 0
    differing = 0
    per_array: Dict[str, int] = {}
    max_ulp = 0
    for name in names:
        a = as_words(left[name]).ravel()
        b = as_words(right[name]).ravel()
        if a.shape != b.shape:
            raise ValueError(f"{name}: shapes differ, {a.shape} vs {b.shape}")
        mask = a != b
        count = int(np.count_nonzero(mask))
        per_array[name] = count
        differing += count
        total += int(a.size)
        if count:
            distance = np.abs(_ordered(a[mask]) - _ordered(b[mask]))
            max_ulp = max(max_ulp, int(distance.max()))
    return {"total_words": total, "differing_words": differing,
            "identical": differing == 0, "per_array": per_array,
            "max_ulp": max_ulp,
            "differing_fraction": (differing / total) if total else 0.0}


def moved_words(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                names: Sequence[str] = OUTPUT_NAMES) -> Dict[str, Any]:
    """How much of the output a launch actually WROTE, against its own input.

    THE VACUITY FLOOR. A kernel that returned its inputs untouched would compare
    identical to another kernel that did the same, and the leg would report a
    perfect match having measured nothing. This is checked in EVERY variant of
    EVERY case, not once per sweep.
    """
    result = compare_words(before, after, names)
    return {"moved_words": result["differing_words"],
            "total_words": result["total_words"],
            "moved_fraction": result["differing_fraction"]}
