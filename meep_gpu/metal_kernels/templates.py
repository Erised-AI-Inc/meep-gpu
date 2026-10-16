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

THE COMPLEX ARITHMETIC IS MEEP'S, PART BY PART. MEEP stores a complex field as two
``realnum`` arrays and forms no complex-by-complex product while it steps
(:mod:`meep_gpu.part_products`, which the array path routes every product through):

* a real coefficient times a field is one float32 product PER PART, ``float2(c *
  z.x, c * z.y)`` -- the ``DOCMP`` loops (step_db.cpp:47, update_eh.cpp:149). Both
  orientation helpers below spell exactly that; IEEE multiplication is commutative,
  so the orientation in a helper's name records the array path's spelling and moves
  no bit;
* a Bloch connection is ``complex<realnum>(thephase) * word`` with every product and
  every sum rounded on its own (step.cpp:183-192, libc++ ``operator*`` in the
  contraction-off build): :func:`c_mul`, with the phase as its second operand. IEEE
  multiplication and addition of two operands are commutative, so ``(z.x*p.x) -
  (z.y*p.y)``, ``(z.x*p.y) + (z.y*p.x)`` is ``part_products.connect_parts``' word.
  Within 1e-13 of +1 MEEP copies and of -1 it negates per part; those two are host
  decisions the emitting family bakes in, never a multiply by ``(+/-1, 0)``;
* negation is ``-x``, a sign-bit operation on this host (0/512 on the exhaustive
  signed-zero table, where ``0.0f - x`` misses 36/512).

THERE IS ONE BODY. An earlier arm copied NumPy's fused complex product and its
zero-cross-term real-coefficient helpers copied NumPy's complex-by-``(c + 0j)`` loop;
the array path carries neither since it forms products per part, so no arm is
chosen and nothing is measured to choose one.

THE 31-BINDING CEILING is a measured platform limit, not a style rule: buffer
attribute indices must be 0..30 and a 32nd binding is a COMPILE ERROR ("'buffer'
attribute parameter is out of bounds: must be between 0 and 30"). A complex curl
binding re/im planes separately needs 35 buffers, so ``float2`` volumes are FORCED
rather than preferred. See :data:`.device.MAX_BUFFER_BINDINGS`; the complex gate
compiles the split signature and requires the failure.
"""

from __future__ import annotations

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
    "GUARD", "DECODE_IJK", "COMPLEX_ZERO",
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

#: The array path assigns a complex zero to BOTH planes when it masks
#: (``stepping._mask_non_owned_cells``), so a complex ownership mask writes this, not
#: ``0.0f``.
COMPLEX_ZERO = "float2(0.0f, 0.0f)"

_COMPLEX_HELPERS = r"""// --- complex arithmetic, part by part (meep_gpu.part_products) -------------
// A realnum coefficient times a field: one rounded product per part, MEEP's DOCMP
// loops. The two names record the array path's operand order; multiplication is
// commutative, so both are the same word.
static inline float2 c_mul_field_left(float2 z, float c) {        // z * c, per part
    return float2(z.x * c, z.y * c);
}
static inline float2 c_mul_coefficient_left(float c, float2 z) {  // c * z, per part
    return float2(c * z.x, c * z.y);
}
// A Bloch connection, complex<realnum>(p) * z with every product and sum rounded on
// its own (libc++ operator*, contraction off): part_products.connect_parts.
static inline float2 c_mul(float2 z, float2 p) {
    return float2((z.x * p.x) - (z.y * p.y),
                  (z.x * p.y) + (z.y * p.x));
}"""


def complex_helpers() -> str:
    """The complex helpers: part-wise real products and MEEP's unfused connection.

    The array path's orientations are recorded in the helper names and are not
    load-bearing: ``dtdx`` scalar LEFT (``stepping._curl_from_operands``),
    ``kms * f`` (``_apply_pml_update``),
    ``kps * fw`` / ``kms * fw_previous`` (``_apply_constitutive_pml``), ``D * inv_eps``
    (``update_E``) -- each ``part_products.scale``, one product per part.
    """
    return _COMPLEX_HELPERS


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
