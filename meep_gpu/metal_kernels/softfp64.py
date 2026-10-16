"""Software IEEE-754 binary64 for Metal kernels: MEEP's double-precision steps on float32 fields.

WHY THIS EXISTS. MEEP stores single-precision fields but writes a few of its formulas
with a ``double`` operand, so C++ promotes the float32 words, performs the operation in
double and rounds once on the store:

* the source deposit, ``f[c][0][i] -= real(A)`` (step.cpp:309-314): the stored word is
  widened, ONE double subtraction runs, ONE rounding brings it back
  (``host_writes.subtract_double``);
* the off-diagonal constitutive row, ``fw = gs*us + OFFDIAG + OFFDIAG``
  (step_generic.cpp:580-583): each OFFDIAG is the double literal ``0.25`` times a
  float32 sum, the row is summed left to right in double and rounded once
  (``stepping._offdiagonal_terms`` / ``_rounded_row``);
* the nonlinear ``Dsqr = gs*gs + 0.0625*(g1s*g1s + g2s*g2s)`` (step_generic.cpp:646-654):
  the two float32 terms widened, the transverse one scaled by the double ``0.0625``,
  added in double and rounded once (``stepping._nonlinear_dsqr``).

The Apple GPU has no float64. These helpers carry a binary64 value as its 64-bit pattern
in a ``ulong`` and perform each operation with integer arithmetic, one IEEE operation per
helper, rounded to nearest even: the words they return are the words the host's own
float64 unit returns for the same statement. Owner ruling R7 (2026-10-08) chose this over
refusing those rows to the array path.

THE API (MSL, every helper ``static inline``; a binary64 operand or result is a ``ulong``
bit pattern, a float32 one a ``uint`` bit pattern):

* ``ulong sd_widen(uint w, bool fz)`` -- float32 -> binary64 (C++'s promotion, FCVT);
* ``ulong sd_widen_f(float v, bool fz)`` -- the same from a float register;
* ``ulong sd_scale_pow2(ulong x, int k, bool fz)`` -- ``x * 2**k``, the product with a
  double power-of-two literal: ``k = -2`` is MEEP's ``0.25``, ``k = -4`` its ``0.0625``;
  ``2**k`` must be a normal binary64 (``-1022 <= k <= 1023``);
* ``ulong sd_add(ulong x, ulong y, bool fz)`` -- ``x + y``;
* ``ulong sd_sub(ulong x, ulong y, bool fz)`` -- ``x - y``;
* ``uint sd_narrow(ulong x, bool fz)`` -- binary64 -> float32, the store's rounding;
* ``float sd_narrow_f(ulong x, bool fz)`` -- the same into a float register;
* ``uint sd_subtract_from_word(uint w, ulong x, bool fz)`` -- the deposit:
  ``narrow(sub(widen(w), x))``;
* ``ulong sd_daz(ulong x)`` -- the flush policy's input rule alone, exported for a
  consumer that receives a binary64 operand from the host.

``fz`` IS THE SUBNORMAL POLICY, and Metal ships one. The MPS executor flushes natively
and cannot be made to keep (:mod:`.subnormal`: ``ATTAINABLE == (FLUSH,)``), so a Metal
kernel runs only under ``flush`` and every shipped call passes ``fz = true``. The array
path it must equal then runs on a host whose FPU is in that mode, and the helpers
reproduce what that host does, MEASURED on this arm64 host (FPCR.FZ set by the package's
own fenv lever; ``backends.subnormals_flushed()`` True):

* an INPUT subnormal, float32 or binary64, reads as a zero of its sign: the widening of
  ``0x80000001`` is ``-0.0``, and ``0.0 - 2**-1074`` is ``+0.0``;
* an OUTPUT whose exact value lies below the format's smallest normal is a zero of its
  sign, decided BEFORE rounding: a binary64 in ``[2**-126 * (1 - 2**-25), 2**-126)``
  narrows to ``+-0`` where gradual underflow gives ``+-2**-126``. That is phase-0 item
  P0-d1's band, where arm64 FZ and x86 FTZ disagree (x86 decides tininess after
  rounding); the Apple GPU's own float32 multiply rounds it as arm64 does
  (``phase0/P0-d/d1/ftz_ops``).

``fz = false`` is IEEE gradual underflow, the ``keep`` policy: the control, and what the
deposit door uses when it is handed a host that keeps (``device.Residency.subtract_double_index``
measures the host's mode at every call, because its specification is the array path's
statement under whatever mode the host is in).

EXACTNESS SCOPE: every binary64 and float32 input, finite or not. Subnormals are formed
and rounded exactly (or flushed under ``fz``); infinities follow IEEE; NaNs follow the
arm64 rules the host applies -- the first signalling NaN operand, else the second, else
the first quiet one, else the second, returned quietened with its payload; an invalid
operation (``inf - inf``) returns the default NaN ``0x7FF8000000000000``; ``x - NaN``
returns that NaN un-negated; widening keeps a NaN's payload shifted up 29 bits with the
quiet bit set, narrowing keeps its top 22 fraction bits. Rounding is to nearest even
only; no other rounding mode is offered or needed.

WHAT IS DELIBERATELY ABSENT: a multiply of two binary64 values, and an unrounded float32
x float32 product. No array-path statement a Metal kernel transcribes forms either:
``gs*us``, the off-diagonal pair products and the ``|D|^2`` squares are float32 products,
each rounded to float32 BEFORE it is widened (``stepping._offdiagonal_terms``: ``(part *
inverse).astype(float64)``; ``stepping._nonlinear_dsqr``: ``(scaled * scaled).astype``),
and the only double multiplies are the power-of-two literals, which
:c:func:`sd_scale_pow2` covers. (The off-diagonal row times a nonlinear factor, the one
double-by-double product ``_nonlinear_constitutive`` forms, has no Metal family: a
nonlinear run with an anisotropic epsilon steps on the array path.)

HOW A KERNEL TAKES IT UP. Splice :data:`MSL_HELPERS` after ``#include <metal_stdlib>``,
``using namespace metal;`` and the contraction directive of :mod:`.shaders` (or call
:func:`library_source`), then call the helpers. The fragment carries an include guard,
so two families spliced into one source define the helpers once. It contains no
floating-point arithmetic -- every operation is on integers, the only float contact
being ``as_type`` reinterpretations -- so the contraction mode cannot change a word (the
device test compiles both modes and compares them), and no fast-math or denormal mode of
the Metal compiler reaches it. Store a result through a ``uint`` view (``sd_narrow``) where
a keep-policy subnormal must survive; under ``fz`` no result is subnormal.

THE NUMPY TRANSCRIPTION. :func:`widen`, :func:`scale_pow2`, :func:`add`, :func:`sub`,
:func:`narrow`, :func:`subtract_from_word` and :func:`daz` below are the helpers
transcribed statement for statement onto ``numpy.uint64`` arrays, branch for branch as
masks. They exist so the arithmetic can be checked on a host with no GPU: the host tests
compare them with the host's float64 unit in both modes, and the device test compares the
MSL with them word for word, so a disagreement says which side moved.

No torch here, and NumPy only inside the transcription's functions.
"""

from __future__ import annotations

from typing import Any

#: The include guard's macro; a source that splices the fragment twice compiles it once.
GUARD = "MEEP_GPU_SOFTFP64"

#: The helpers, as one MSL fragment. See the module docstring for the contract.
MSL_HELPERS = r"""
#ifndef MEEP_GPU_SOFTFP64
#define MEEP_GPU_SOFTFP64
// ---- software binary64 on ulong words (meep_gpu.metal_kernels.softfp64) ---------------
// One IEEE operation per helper, round to nearest even. fz = the flush policy: an input
// subnormal reads as a zero of its sign, a result below the smallest normal (decided
// before rounding) is a zero of its sign. fz = false is gradual underflow.
constant ulong SD_SIGN = 0x8000000000000000ul;
constant ulong SD_EXP = 0x7FF0000000000000ul;
constant ulong SD_FRAC = 0x000FFFFFFFFFFFFFul;
constant ulong SD_ONE = 0x0010000000000000ul;          // 2**52, the implicit bit
constant ulong SD_QUIET = 0x0008000000000000ul;        // the quiet-NaN bit
constant ulong SD_DEFAULT_NAN = 0x7FF8000000000000ul;  // an invalid operation's result

// The flush policy's input rule: a binary64 subnormal reads as a zero of its sign.
static inline ulong sd_daz(ulong x) {
    return ((x & SD_EXP) == 0ul) ? (x & SD_SIGN) : x;
}

static inline bool sd_is_nan(ulong x) {
    return (x & ~SD_SIGN) > SD_EXP;
}

// Two operands, at least one a NaN: the first signalling NaN, else the second, else the
// first quiet one, else the second; quietened, payload kept.
static inline ulong sd_pick_nan(ulong x, ulong y) {
    bool x_signals = sd_is_nan(x) && (x & SD_QUIET) == 0ul;
    bool y_signals = sd_is_nan(y) && (y & SD_QUIET) == 0ul;
    ulong r = x_signals ? x : (y_signals ? y : (sd_is_nan(x) ? x : y));
    return r | SD_QUIET;
}

// Round and pack. m holds the magnitude with its leading one at bit 55 and three bits
// below the binary64 ulp (guard, round, sticky: bit 0 is set when anything nonzero lies
// below it); e is the biased exponent of that leading one, which may be < 1.
static inline ulong sd_round_pack(ulong sign, int e, ulong m, bool fz) {
    if (e < 1) {
        if (fz) { return sign; }
        int r = 1 - e;
        m = (r >= 58) ? 1ul
                      : ((m >> ulong(r)) | (((m & ((1ul << ulong(r)) - 1ul)) != 0ul) ? 1ul : 0ul));
        e = 0;
    }
    ulong g = m & 7ul;
    m >>= 3;
    if (g > 4ul || (g == 4ul && (m & 1ul) != 0ul)) { m += 1ul; }
    if (e == 0) { return sign | m; }                   // a subnormal, or 2**-1022 if it carried
    if ((m >> 53) != 0ul) { m >>= 1; e += 1; }
    if (e >= 2047) { return sign | SD_EXP; }
    return sign | (ulong(e) << 52) | (m & SD_FRAC);
}

// float32 word -> binary64, exact. A NaN is quietened with its payload shifted up 29 bits.
static inline ulong sd_widen(uint w, bool fz) {
    ulong s = ulong(w >> 31) << 63;
    uint e = (w >> 23) & 0xFFu;
    uint m = w & 0x7FFFFFu;
    if (e - 1u >= 0xFEu) {                               // exponent 0 or 255
        if (e == 0xFFu) {
            return s | SD_EXP | (ulong(m) << 29) | ((m != 0u) ? SD_QUIET : 0ul);
        }
        if (m == 0u || fz) { return s; }
        uint p = 31u - clz(m);                           // the top set bit, 0..22
        return s | (ulong(874u + p) << 52) | ((ulong(m) << (52u - p)) & SD_FRAC);
    }
    return s | (ulong(e + 896u) << 52) | (ulong(m) << 29);
}

static inline ulong sd_widen_f(float v, bool fz) {
    return sd_widen(as_type<uint>(v), fz);
}

// x * 2**k, where 2**k is a normal binary64 literal (MEEP's 0.25 is k = -2, 0.0625 k = -4).
static inline ulong sd_scale_pow2(ulong x, int k, bool fz) {
    if (fz) { x = sd_daz(x); }
    ulong ax = x & ~SD_SIGN;
    int e = int(ax >> 52);
    if (e >= 1 && e <= 2046 && e + k >= 1 && e + k <= 2046) {
        return x + (ulong(long(k)) << 52);               // normal in, normal out: exact
    }
    if (ax > SD_EXP) { return x | SD_QUIET; }
    if (ax == SD_EXP || ax == 0ul) { return x; }
    ulong m = ax & SD_FRAC;
    if (e == 0) {
        int shift = int(clz(m)) - 11;                   // the leading one to bit 52
        m <<= ulong(shift);
        e = 1 - shift;
    } else {
        m |= SD_ONE;
    }
    return sd_round_pack(x & SD_SIGN, e + k, m << 3, fz);
}

static inline ulong sd_add(ulong x, ulong y, bool fz) {
    if (fz) { x = sd_daz(x); y = sd_daz(y); }
    ulong ax = x & ~SD_SIGN, ay = y & ~SD_SIGN;
    if (ax - SD_ONE >= SD_EXP - SD_ONE || ay - SD_ONE >= SD_EXP - SD_ONE) {
        // an operand that is not a normal number: a zero, a subnormal, an infinity, a NaN
        if (ax > SD_EXP || ay > SD_EXP) { return sd_pick_nan(x, y); }
        if (ax == SD_EXP) {
            return (ay == SD_EXP && ((x ^ y) & SD_SIGN) != 0ul) ? SD_DEFAULT_NAN : x;
        }
        if (ay == SD_EXP) { return y; }
        if (ay == 0ul) { return (ax == 0ul) ? (x & y) : x; }  // (-0) + (-0) = -0, else +0
        if (ax == 0ul) { return y; }
    }
    if (ax < ay) { ulong t = x; x = y; y = t; t = ax; ax = ay; ay = t; }
    int ex = int(ax >> 52), ey = int(ay >> 52);
    ulong mx = ax & SD_FRAC, my = ay & SD_FRAC;
    if (ex == 0) { ex = 1; } else { mx |= SD_ONE; }      // a subnormal: exponent 1, no implicit bit
    if (ey == 0) { ey = 1; } else { my |= SD_ONE; }
    mx <<= 3;                                             // room for guard, round and sticky
    my <<= 3;
    int d = ex - ey;
    if (d >= 58) {
        my = 1ul;                                         // all of y is sticky
    } else if (d > 0) {
        ulong lost = my & ((1ul << ulong(d)) - 1ul);
        my = (my >> ulong(d)) | ((lost != 0ul) ? 1ul : 0ul);
    }
    ulong m;
    if (((x ^ y) & SD_SIGN) == 0ul) {
        m = mx + my;
        if ((m >> 56) != 0ul) { m = (m >> 1) | (m & 1ul); ex += 1; }
    } else {
        m = mx - my;
        if (m == 0ul) { return 0ul; }                     // an exact cancellation is +0
    }
    int shift = int(clz(m)) - 8;                          // the leading one to bit 55
    m <<= ulong(shift);
    return sd_round_pack(x & SD_SIGN, ex - shift, m, fz);
}

// x - y. A NaN y is returned as it is (quietened), not negated.
static inline ulong sd_sub(ulong x, ulong y, bool fz) {
    if (sd_is_nan(x) || sd_is_nan(y)) { return sd_pick_nan(x, y); }
    return sd_add(x, y ^ SD_SIGN, fz);
}

// binary64 -> float32, the store's one rounding. A NaN keeps its top 22 fraction bits.
static inline uint sd_narrow(ulong x, bool fz) {
    uint s = uint(x >> 32) & 0x80000000u;
    ulong ax = x & ~SD_SIGN;
    if (ax >= SD_EXP) {
        return (ax > SD_EXP) ? (s | 0x7FC00000u | uint((ax >> 29) & 0x3FFFFFul)) : (s | 0x7F800000u);
    }
    int e = int(ax >> 52) - 896;                          // the float32 biased exponent
    if (e <= 0 && (fz || e <= -24)) { return s; }         // flushed, or below half the least subnormal
    ulong m = (ax & SD_FRAC) | SD_ONE;                    // 53 significant bits
    int shift = 29;
    if (e <= 0) { shift += 1 - e; e = 0; }                // gradual underflow
    ulong q = m >> ulong(shift);
    ulong rem = m & ((1ul << ulong(shift)) - 1ul);
    ulong midpoint = 1ul << ulong(shift - 1);
    if (rem > midpoint || (rem == midpoint && (q & 1ul) != 0ul)) { q += 1ul; }
    if (e == 0) { return s | uint(q); }                   // a subnormal, or 2**-126 if it carried
    if ((q >> 24) != 0ul) { q >>= 1; e += 1; }
    if (e >= 255) { return s | 0x7F800000u; }
    return s | (uint(e) << 23) | (uint(q) & 0x7FFFFFu);
}

static inline float sd_narrow_f(ulong x, bool fz) {
    return as_type<float>(sd_narrow(x, fz));
}

// MEEP's source deposit on one stored word: float(double(f) - x).
static inline uint sd_subtract_from_word(uint w, ulong x, bool fz) {
    return sd_narrow(sd_sub(sd_widen(w, fz), x, fz), fz);
}
// ---- end software binary64 ------------------------------------------------------------
#endif
"""

#: Each public helper's MSL signature, as the fragment spells it. The test file checks
#: the fragment against this table, and the consuming families' lanes read it: a rename
#: here is a rename there, in the same change.
SIGNATURES = (
    "static inline ulong sd_widen(uint w, bool fz)",
    "static inline ulong sd_widen_f(float v, bool fz)",
    "static inline ulong sd_scale_pow2(ulong x, int k, bool fz)",
    "static inline ulong sd_add(ulong x, ulong y, bool fz)",
    "static inline ulong sd_sub(ulong x, ulong y, bool fz)",
    "static inline uint sd_narrow(ulong x, bool fz)",
    "static inline float sd_narrow_f(ulong x, bool fz)",
    "static inline uint sd_subtract_from_word(uint w, ulong x, bool fz)",
    "static inline ulong sd_daz(ulong x)",
)

#: MEEP's two double literals as the ``k`` of :c:func:`sd_scale_pow2`.
QUARTER = -2      # step_generic.cpp:580, OFFDIAG's 0.25
SIXTEENTH = -4    # step_generic.cpp:646-654, Dsqr's 0.0625


def library_source(kernels: str, contract: str = "off") -> str:
    """A complete MSL source: header, the contraction directive, :data:`MSL_HELPERS`, ``kernels``.

    The directive comes from :func:`.shaders.contraction_pragma`, its one spelling in this
    package; ``off`` is the shipped mode.
    """
    from . import shaders  # noqa: PLC0415

    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            + shaders.contraction_pragma(contract) + "\n" + MSL_HELPERS + kernels)


# ---------------------------------------------------------------------------
# The NumPy transcription: the same statements on uint64 arrays
# ---------------------------------------------------------------------------
#
# Each function below is the MSL helper of the same name, statement for statement; a
# branch is a mask, and the arrays a branch does not select are carried through with
# values that keep every shift below 64 bits. Inputs are uint32 (float32 words) or uint64
# (binary64 words) arrays of one shape; outputs are fresh arrays of the same shape.

def _np():
    import numpy as np  # noqa: PLC0415

    return np


def _consts():
    np = _np()
    return {
        "SIGN": np.uint64(0x8000000000000000),
        "EXP": np.uint64(0x7FF0000000000000),
        "FRAC": np.uint64(0x000FFFFFFFFFFFFF),
        "ONE": np.uint64(0x0010000000000000),
        "QUIET": np.uint64(0x0008000000000000),
        "DEFAULT_NAN": np.uint64(0x7FF8000000000000),
    }


def _clz64(v: Any) -> Any:
    """``clz`` of every element of a uint64 array (64 for a zero element)."""
    np = _np()
    v = v.astype(np.uint64, copy=True)
    n = np.zeros(v.shape, dtype=np.int64)
    for step in (32, 16, 8, 4, 2, 1):
        high = (v >> np.uint64(step)) != 0
        v = np.where(high, v >> np.uint64(step), v)
        n += np.where(high, step, 0)
    return np.where(v != 0, 63 - n, 64)


def _shr(v: Any, s: Any) -> Any:
    """``v >> s`` for shift counts 0..63 (an array or a scalar)."""
    np = _np()
    return v >> np.asarray(s).astype(np.uint64)


def _shl(v: Any, s: Any) -> Any:
    np = _np()
    return v << np.asarray(s).astype(np.uint64)


def _low_mask(s: Any) -> Any:
    """``(1 << s) - 1`` for 0 <= s <= 63."""
    np = _np()
    return _shl(np.ones(np.shape(s), dtype=np.uint64), s) - np.uint64(1)


def daz(x: Any) -> Any:
    """:c:func:`sd_daz`."""
    np = _np()
    c = _consts()
    x = np.asarray(x, dtype=np.uint64)
    return np.where((x & c["EXP"]) == 0, x & c["SIGN"], x)


def is_nan(x: Any) -> Any:
    """:c:func:`sd_is_nan`."""
    c = _consts()
    return (x & ~c["SIGN"]) > c["EXP"]


def pick_nan(x: Any, y: Any) -> Any:
    """:c:func:`sd_pick_nan`."""
    np = _np()
    c = _consts()
    x_signals = is_nan(x) & ((x & c["QUIET"]) == 0)
    y_signals = is_nan(y) & ((y & c["QUIET"]) == 0)
    r = np.where(x_signals, x, np.where(y_signals, y, np.where(is_nan(x), x, y)))
    return r | c["QUIET"]


def _round_pack(sign: Any, e: Any, m: Any, fz: bool) -> Any:
    """:c:func:`sd_round_pack` (``e`` int64, ``m`` uint64 with its leading one at bit 55)."""
    np = _np()
    c = _consts()
    e = np.asarray(e, dtype=np.int64).copy()
    m = np.asarray(m, dtype=np.uint64).copy()
    tiny = e < 1
    flushed = tiny & bool(fz)
    if not fz:
        r = np.where(tiny, 1 - e, 0)
        safe_r = np.clip(r, 0, 57)
        lost = (m & _low_mask(safe_r)) != 0
        shifted = _shr(m, safe_r) | lost.astype(np.uint64)
        m = np.where(tiny, np.where(r >= 58, np.uint64(1), shifted), m)
        e = np.where(tiny, 0, e)
    g = m & np.uint64(7)
    m = m >> np.uint64(3)
    up = (g > 4) | ((g == 4) & ((m & np.uint64(1)) != 0))
    m = m + up.astype(np.uint64)
    sub = e == 0
    carry = (~sub) & ((m >> np.uint64(53)) != 0)
    m = np.where(carry, m >> np.uint64(1), m)
    e = np.where(carry, e + 1, e)
    over = (~sub) & (e >= 2047)
    packed = sign | (np.clip(e, 0, 2047).astype(np.uint64) << np.uint64(52)) | (m & c["FRAC"])
    out = np.where(sub, sign | m, np.where(over, sign | c["EXP"], packed))
    return np.where(flushed, sign, out)


def widen(w: Any, fz: bool) -> Any:
    """:c:func:`sd_widen` on uint32 float32 words."""
    np = _np()
    c = _consts()
    w = np.asarray(w, dtype=np.uint32).astype(np.uint64)
    s = (w >> np.uint64(31)) << np.uint64(63)
    e = (w >> np.uint64(23)) & np.uint64(0xFF)
    m = w & np.uint64(0x7FFFFF)
    special = s | c["EXP"] | (m << np.uint64(29)) | np.where(m != 0, c["QUIET"], np.uint64(0))
    p = 63 - _clz64(np.where(m == 0, np.uint64(1), m))      # the top set bit (m == 0 unused)
    sub = s | ((np.uint64(874) + p.astype(np.uint64)) << np.uint64(52)) | \
        (_shl(m, np.clip(52 - p, 0, 63)) & c["FRAC"])
    sub = np.where((m == 0) | bool(fz), s, sub)
    normal = s | ((e + np.uint64(896)) << np.uint64(52)) | (m << np.uint64(29))
    return np.where(e == 0xFF, special, np.where(e == 0, sub, normal))


def scale_pow2(x: Any, k: int, fz: bool) -> Any:
    """:c:func:`sd_scale_pow2`."""
    np = _np()
    c = _consts()
    x = np.asarray(x, dtype=np.uint64)
    if fz:
        x = daz(x)
    ax = x & ~c["SIGN"]
    e = (ax >> np.uint64(52)).astype(np.int64)
    exact = (e >= 1) & (e <= 2046) & (e + int(k) >= 1) & (e + int(k) <= 2046)
    moved = x + (np.int64(k).astype(np.uint64) << np.uint64(52))
    m = ax & c["FRAC"]
    shift = np.clip(_clz64(np.where(m == 0, np.uint64(1), m)) - 11, 0, 63)
    sub = e == 0
    m = np.where(sub, _shl(m, shift), m | c["ONE"])
    e = np.where(sub, 1 - shift, e)
    out = _round_pack(x & c["SIGN"], e + int(k), m << np.uint64(3), fz)
    out = np.where((ax == c["EXP"]) | (ax == 0), x, out)
    out = np.where(ax > c["EXP"], x | c["QUIET"], out)
    return np.where(exact, moved, out)


def add(x: Any, y: Any, fz: bool) -> Any:
    """:c:func:`sd_add`."""
    np = _np()
    c = _consts()
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    if fz:
        x, y = daz(x), daz(y)
    ax, ay = x & ~c["SIGN"], y & ~c["SIGN"]
    nan = (ax > c["EXP"]) | (ay > c["EXP"])
    x_inf = ax == c["EXP"]
    y_inf = ay == c["EXP"]
    inf_out = np.where(x_inf, np.where(y_inf & (((x ^ y) & c["SIGN"]) != 0), c["DEFAULT_NAN"], x), y)
    y_zero = ay == 0
    x_zero = ax == 0
    zero_out = np.where(y_zero, np.where(x_zero, x & y, x), y)
    swap = ax < ay
    x, y = np.where(swap, y, x), np.where(swap, x, y)
    ax, ay = np.where(swap, ay, ax), np.where(swap, ax, ay)
    ex = (ax >> np.uint64(52)).astype(np.int64)
    ey = (ay >> np.uint64(52)).astype(np.int64)
    mx = np.where(ex == 0, ax & c["FRAC"], (ax & c["FRAC"]) | c["ONE"]) << np.uint64(3)
    my = np.where(ey == 0, ay & c["FRAC"], (ay & c["FRAC"]) | c["ONE"]) << np.uint64(3)
    ex = np.where(ex == 0, 1, ex)
    ey = np.where(ey == 0, 1, ey)
    d = ex - ey
    safe_d = np.clip(d, 0, 57)
    lost = (my & _low_mask(safe_d)) != 0
    aligned = _shr(my, safe_d) | lost.astype(np.uint64)
    my = np.where(d >= 58, np.uint64(1), np.where(d > 0, aligned, my))
    same = ((x ^ y) & c["SIGN"]) == 0
    total = mx + my
    carry = (total >> np.uint64(56)) != 0
    total = np.where(carry, (total >> np.uint64(1)) | (total & np.uint64(1)), total)
    ex = np.where(same & carry, ex + 1, ex)
    difference = mx - my
    m = np.where(same, total, difference)
    cancelled = (~same) & (difference == 0)
    shift = np.clip(_clz64(np.where(m == 0, np.uint64(1), m)) - 8, 0, 63)
    m = _shl(m, shift)
    out = _round_pack(x & c["SIGN"], ex - shift, m, fz)
    out = np.where(cancelled, np.uint64(0), out)
    out = np.where(y_zero | x_zero, zero_out, out)
    out = np.where(x_inf | y_inf, inf_out, out)
    return np.where(nan, pick_nan(np.where(swap, y, x), np.where(swap, x, y)), out)


def sub(x: Any, y: Any, fz: bool) -> Any:
    """:c:func:`sd_sub`."""
    np = _np()
    c = _consts()
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    nan = is_nan(x) | is_nan(y)
    return np.where(nan, pick_nan(x, y), add(x, y ^ c["SIGN"], fz))


def narrow(x: Any, fz: bool) -> Any:
    """:c:func:`sd_narrow`, returning uint32 float32 words."""
    np = _np()
    c = _consts()
    x = np.asarray(x, dtype=np.uint64)
    s = ((x >> np.uint64(32)) & np.uint64(0x80000000))
    ax = x & ~c["SIGN"]
    nan_out = s | np.uint64(0x7FC00000) | ((ax >> np.uint64(29)) & np.uint64(0x3FFFFF))
    inf_out = s | np.uint64(0x7F800000)
    e = (ax >> np.uint64(52)).astype(np.int64) - 896
    zero = (e <= 0) & (bool(fz) | (e <= -24))
    m = (ax & c["FRAC"]) | c["ONE"]
    shift = np.where(e <= 0, 29 + 1 - e, 29)
    shift = np.clip(shift, 1, 63)
    e = np.where(e <= 0, 0, e)
    q = _shr(m, shift)
    rem = m & _low_mask(shift)
    midpoint = _shl(np.ones(m.shape, dtype=np.uint64), shift - 1)
    up = (rem > midpoint) | ((rem == midpoint) & ((q & np.uint64(1)) != 0))
    q = q + up.astype(np.uint64)
    sub_out = s | q
    carry = (q >> np.uint64(24)) != 0
    q = np.where(carry, q >> np.uint64(1), q)
    e = np.where(carry, e + 1, e)
    normal = s | (np.clip(e, 0, 255).astype(np.uint64) << np.uint64(23)) | (q & np.uint64(0x7FFFFF))
    normal = np.where(e >= 255, inf_out, normal)
    out = np.where(e == 0, sub_out, normal)
    out = np.where(zero, s, out)
    out = np.where(ax == c["EXP"], inf_out, out)
    out = np.where(ax > c["EXP"], nan_out, out)
    return out.astype(np.uint32)


def subtract_from_word(w: Any, x: Any, fz: bool) -> Any:
    """:c:func:`sd_subtract_from_word`."""
    return narrow(sub(widen(w, fz), x, fz), fz)
