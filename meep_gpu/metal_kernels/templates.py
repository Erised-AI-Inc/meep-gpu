"""The one import surface a Metal kernel family assembles its source from.

WHAT THIS MODULE IS. Two things, and the split is deliberate:

1. **A re-export of the certified emitters**, not a new home for them.
   :func:`ghost`, :func:`ownership_mask`, :func:`substitute`,
   :func:`contraction_pragma`, :func:`source_sha256` and the boundary codes stay
   defined in :mod:`.shaders` and are re-exported here. Two reasons, both hard:
   ``test_metal_kernels.test_the_contraction_directive_is_spelled_exactly_once``
   pins the contraction directive's home to ``shaders.py`` and fails the build on a
   second spelling anywhere in the package; and moving a CERTIFIED emitter would
   re-emit every fingerprinted source string through a new code path to save an
   import — EIGHTEEN per contraction mode (sixteen curl specialisations and two
   constitutive sides), thirty-six rows in ``fingerprints.json``. The trade is
   obvious in one direction.
2. **New shared fragments** this round licenses: the ``n_elem`` guard, the flat
   index decode, the complex ``float2`` arithmetic, and a column-serial scan
   skeleton for the cylindrical port.

WHAT DELIBERATELY IS NOT HERE: ``shaders._CURL_TEMPLATE`` and
``shaders._CONSTITUTIVE_TEMPLATE``. The certified pair keeps its own copy of both
templates and both emitters; new families import the fragments. That is duplication
with a stated reason, and
``test_metal_kernels.test_the_checked_in_fingerprints_match_the_tree`` asserts every
one of the THIRTY-SIX checked-in kernel-source sha256s is unchanged, which makes the
byte-neutrality of this whole extraction MEASURED rather than asserted. Re-measured
2026-08-15 after the ``zero`` parameter landed on :func:`ownership_mask`: 36 of 36
unchanged.

THE COMPLEX ARITHMETIC IS MEASURED, NOT INHERITED (2026-08-15, this host,
torch 2.10.0 / numpy 2.4.3 / macOS 26.2 arm64):

* the reference — ``numpy``'s own complex64 multiply — takes the **FMA_V1** arm:
  ``re = fma(a_re, b_re, -(a_im*b_im))``, ``im = fma(a_re, b_im, +(a_im*b_re))``.
  Measured on 8,192-cell random vectors, the FMA arm matches 0/16,384 words and the
  four-rounded-product arm misses 4,095/16,384. On an EXHAUSTIVE 256-pattern
  signed-zero/finite table the two arms AGREE (every product there is exact), so
  neither class alone classifies and the gate runs both. Metal's explicit ``fma()``
  SURVIVES the file-scope contraction directive (:func:`contraction_pragma`, spelled
  once in :mod:`.shaders`) — that directive stops IMPLICIT contraction only, which is
  what makes the fused arm spellable at all;
* the **zero cross terms are load-bearing** and are spelled literally. On the
  exhaustive 64-pattern tables, folding ``z_im * 0.0f`` to a literal misses 12/128
  words in EITHER orientation (field left and coefficient left alike, re-measured
  2026-08-15 by ``gate_metal_complex`` leg ``zero_cross_terms``), and a plane-wise
  ``{re*c, im*c}`` fast path misses 24/128. An earlier revision of this line gave
  24/128 for the folded coefficient-left form; that is the plane-wise number, and
  the coefficient-left fold had never been measured at all. RANDOM DATA
  DISCRIMINATES NONE OF THEM (0/16,384 for all three), which is why the gate's zero
  tables are exhaustive rather than sampled;
* **negation is spelled ``-x``.** On the same exhaustive table ``-x`` and
  ``x * -1.0f`` are both exact (0/512) while ``0.0f - x`` misses 36/512. The Triton
  track carries ``(a*b) * -1.0`` because *Triton* lowers unary minus as ``0.0 - x``
  and canonicalizes signed zeros; **that reason does not reproduce on Metal**, so
  the workaround is NOT inherited. The refuted spelling is a must-catch gate
  mutation rather than a comment.

THE 31-BINDING CEILING is a measured platform limit, not a style rule: buffer
attribute indices must be 0..30 and a 32nd binding is a COMPILE ERROR ("'buffer'
attribute parameter is out of bounds: must be between 0 and 30"). A complex curl
binding re/im planes separately needs 35 buffers, so ``float2`` volumes are FORCED
rather than preferred. See :data:`.device.MAX_BUFFER_BINDINGS`; the complex gate
compiles the split signature and requires the failure.
"""

from __future__ import annotations

from typing import Dict

# EVERY ONE OF THESE IS IMPORTED, NOT RE-SPELLED — see the module docstring for the
# two reasons. `shaders` does not import this module, so the direction is one-way
# and there is no cycle: certified emitters flow outward to the new families.
from .shaders import (  # noqa: F401 - re-exported deliberately
    CONTRACT_FAST,
    CONTRACT_MODES,
    CONTRACT_OFF,
    METALLIC,
    PERIODIC,
    contraction_pragma,
    ghost,
    ownership_mask,
    source_sha256,
    substitute,
)

__all__ = [
    "CONTRACT_FAST", "CONTRACT_MODES", "CONTRACT_OFF", "METALLIC", "PERIODIC",
    "contraction_pragma", "ghost", "ownership_mask", "source_sha256", "substitute",
    "GUARD", "DECODE_IJK", "EXPANSIONS", "EXPANSION_ARMS", "COMPLEX_ZERO",
    "complex_helpers", "COLUMN_SERIAL_SCAN",
]


# ---------------------------------------------------------------------------
# Index and guard fragments
# ---------------------------------------------------------------------------

#: The dispatch is sized from the FIRST tensor argument's element count, so a volume
#: wider than the sub-step's own extent would step cells the array path does not
#: own. Every kernel in this package opens with this guard, and an ``n_elem`` bound
#: is what makes the guard checkable rather than a comment about grid sizing.
GUARD = "    if (idx >= n_elem) { return; }"

#: C-contiguous (nx, ny, nz) flat index -> (i, j, k). Integer and exact: no float
#: rounds on this path, so a spelling difference here cannot change a bit.
DECODE_IJK = """    int nyi = int(ny), nzi = int(nz);
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;"""


# ---------------------------------------------------------------------------
# Complex (float2) arithmetic
# ---------------------------------------------------------------------------

#: The two transcription arms of the reference complex multiply, named as the probe
#: artifact spells them. Which arm a platform takes is a compiled-dispatch fact, so
#: a family BINDS THE ARM FROM A MEASURED ARTIFACT and never from a guess; a missing
#: or disagreeing probe is a coverage refusal by name.
EXPANSIONS: Dict[str, int] = {"NAIVE": 0, "FMA_V1": 1}

#: Host-side arm names in probe-artifact spelling, in code order.
EXPANSION_ARMS = ("NAIVE", "FMA_V1")

#: The array path assigns a complex zero to BOTH planes when it masks
#: (stepping.py:1943, :1949), so a complex ownership mask writes this, not ``0.0f``.
COMPLEX_ZERO = "float2(0.0f, 0.0f)"

_COMPLEX_HELPERS: Dict[str, str] = {
    "FMA_V1": r"""// --- complex multiply: FMA_V1 arm (MEASURED against numpy on this host) -----
// 0/16384 words on random data where the four-rounded-product arm misses 4095.
// Negation is `-x`, a sign-bit operation here; `0.0f - x` misses 36/512 on the
// exhaustive signed-zero table and is a gate mutation, not a comment. The zero
// cross terms are LITERAL: folding them misses 12/128 (field left) and 24/128
// (coefficient left), and random data catches neither.
static inline float2 c_mul(float2 z, float2 p) {                  // z * p, z LEFT
    return float2(fma(z.x, p.x, -(z.y * p.y)),
                  fma(z.x, p.y,  (z.y * p.x)));
}
static inline float2 c_mul_field_left(float2 z, float c) {        // z * (c + 0j)
    return float2(fma(z.x, c,    -(z.y * 0.0f)),
                  fma(z.x, 0.0f,  (z.y * c)));
}
static inline float2 c_mul_coefficient_left(float c, float2 z) {  // (c + 0j) * z
    return float2(fma(c, z.x, -(0.0f * z.y)),
                  fma(c, z.y,  (0.0f * z.x)));
}""",
    "NAIVE": r"""// --- complex multiply: NAIVE arm -------------------------------------------
// Four separately rounded products and two rounded adds. A REAL arm, not a
// strawman: which form the REFERENCE takes is a platform fact, and a host whose
// numpy dispatched a scalar loop would bind here instead. The probe decides.
static inline float2 c_mul(float2 z, float2 p) {
    return float2((z.x * p.x) - (z.y * p.y),
                  (z.x * p.y) + (z.y * p.x));
}
static inline float2 c_mul_field_left(float2 z, float c) {
    return float2((z.x * c)    - (z.y * 0.0f),
                  (z.x * 0.0f) + (z.y * c));
}
static inline float2 c_mul_coefficient_left(float c, float2 z) {
    return float2((c * z.x) - (0.0f * z.y),
                  (c * z.y) + (0.0f * z.x));
}""",
}


def complex_helpers(expansion: str) -> str:
    """The three complex-multiply helpers for one measured expansion arm.

    THE THREE ORIENTATIONS ARE SEPARATE FUNCTIONS ON PURPOSE. Operand order changes
    bytes under a fused expansion, so the array path's orientations are normative
    and transcribed literally: ``dtdx`` scalar LEFT (stepping.py:1682), ``fu *= kms``
    field LEFT (:1976-1982), ``kps*fw`` / ``kms*fw_previous`` coefficient LEFT
    (:2086-2087, :2093-2095), ``D * inv_eps`` D LEFT (:982-984), the Bloch phase
    multiply plane LEFT with the phase pre-rounded to complex64 (:1862).
    """
    if expansion not in _COMPLEX_HELPERS:
        raise ValueError(f"expansion must be one of {tuple(_COMPLEX_HELPERS)}, "
                         f"got {expansion!r}")
    return _COMPLEX_HELPERS[expansion]


# ---------------------------------------------------------------------------
# Column-serial scan — the cylindrical prefix skeleton
# ---------------------------------------------------------------------------

#: A prefix reduction along the FIRST axis, one thread per (j, k) column, summed
#: strictly serially from i = 0 upward.
#:
#: WHY A HAND SCAN, and this is a LICENCE rather than a preference: a parallel or
#: blocked scan reassociates the sum and float32 addition is not associative. The
#: Triton cylindrical tranche measured ``torch.cumsum`` against ``numpy.cumsum`` at
#: 2691/4096, 283/555 and 3210/4096 differing words while a column-serial reference
#: sat at 0/4096. Any family needing a radial prefix builds on this and its gate
#: carries that control, or the choice looks like taste.
#:
#: NOT CONSUMED BY ANY FAMILY IN THIS TRANCHE. It is here so the cylindrical port
#: inherits the fragment and the reason together rather than rediscovering both.
COLUMN_SERIAL_SCAN = r"""
kernel void __NAME__(
    device float*       out     [[buffer(0)]],
    device const float* src     [[buffer(1)]],
    constant uint&      nx      [[buffer(2)]],
    constant uint&      ny      [[buffer(3)]],
    constant uint&      nz      [[buffer(4)]],
    constant uint&      n_cols  [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_cols) { return; }
    int nzi = int(nz), nyi = int(ny), nxi = int(nx);
    int k = int(idx) % nzi;
    int j = int(idx) / nzi;
    if (j >= nyi) { return; }
    int nyz = nyi * nzi;
    float acc = 0.0f;
    for (int i = 0; i < nxi; ++i) {
        int o = i * nyz + j * nzi + k;
        acc = acc + src[o];          // STRICTLY SERIAL: no blocked reassociation.
        out[o] = acc;
    }
}
"""
