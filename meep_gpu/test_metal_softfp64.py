"""The software binary64 helpers of the Metal table, word for word against the host's float64 unit.

:mod:`meep_gpu.metal_kernels.softfp64` carries MEEP's double-precision steps on 64-bit
integer words because the Apple GPU has no float64. Its specification is the host's own
float64 arithmetic under the subnormal mode the host is in, so every comparison here is a
WORD comparison (uint64 for a binary64, uint32 for a float32, sign of zero and NaN
payload included) against NumPy float64 on this host, with the FPU keeping and then
driven to flush (``subnormal_policy._drive_host_fpu``; restored after every leg).

Three legs, so a disagreement names the side that moved:

* HOST: the NumPy transcription of the helpers against the host's float64 unit, on
  every operand class below, in both modes. Runs anywhere.
* DEVICE: the MSL helpers launched on the Apple GPU against the host's float64 unit on
  the same classes, at least 4,000,000 triples per class in each mode, and against the
  transcription word for word.
* CONTROLS: each class is checked to construct what it claims (a class that never
  produced a tie cannot certify the tie rule), and each planted defect in the MSL (and a
  representative set in the transcription) must be caught.

The operand classes. The six of the off-diagonal row's measurement (the R7 ruling's
evidence, ``r1 = widen(a) + 2**k * widen(t1)``, ``r2 = r1 + 2**k * widen(t2)``,
``out = narrow(r2)``, terms present at random), each run with ``k = -2`` (MEEP's 0.25) and
``k = -4`` (the ``|D|^2`` 0.0625):

  spread       independent exponents over 2**+-30, both signs
  cancel       t1 = -4a(1 + d), d down to 2**-24: deep cancellation in the first add
  gap_29_31    t1/4 29..31 bits below a: the sum needs 53..55 bits, so the double add
               rounds and about half its cases are exact double ties
  float_ties   a + t1/4 on a float32 midpoint: the narrowing's own ties
  zeros        +0 and -0 planted in each operand, and exact cancellations
  subnormal    operands and results in the float32 subnormal band

plus ``band``: ``r`` exactly ``2**-126 * (1 - 2**-25)`` from NORMAL float32 operands, the
bottom of phase-0 item P0-d1's band, where gradual underflow rounds up to ``2**-126`` and
the flush policy (tininess before rounding) gives ``+-0``. And three classes for the
single operations (widen, add, sub, scale, narrow, and the deposit
``narrow(sub(widen(w), x))``):

  bits         random binary64 and float32 patterns biased to zeros, subnormals,
               infinities and NaNs (signalling and quiet), extreme exponents
  deposit      stored words of every class against ``real(A)`` doubles over the whole
               binary64 range (subnormals included), with planted float32 midpoints
  band         ``w - x`` and ``narrow(x)`` inside and around P0-d1's band

The deposit's reference is :func:`meep_gpu.host_writes._rounded_difference`, the array
path's own door, so the helper is pinned to the statement it replaces.
"""

from __future__ import annotations

import hashlib
import platform
import re
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

from conftest import requires_resource_skip  # noqa: E402

from meep_gpu import backends, host_writes, subnormal_policy
from meep_gpu.metal_kernels import shaders, softfp64

#: Triples per class in the device leg (the battery's floor), and the slice the
#: transcription is word-checked against the MSL on.
DEVICE_TRIPLES = 4_000_000
TRANSCRIPTION_TRIPLES = 250_000
MUTANT_TRIPLES = 200_000
HOST_TRIPLES = 100_000
CHUNK = 1_000_000

ROW_CLASSES = ("spread", "cancel", "gap_29_31", "float_ties", "zeros", "subnormal", "band")
OP_CLASSES = ("bits", "deposit", "band")
POLICIES = ("keep", "flush")
ROW_SCALES = (softfp64.QUARTER, softfp64.SIXTEENTH)
OP_SCALES = {"bits": -60, "deposit": softfp64.QUARTER, "band": softfp64.SIXTEENTH}

TWO = np.float64(2.0)


def _say(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} softfp64: {message}", flush=True)


# ---------------------------------------------------------------------------
# The host's mode, driven and restored
# ---------------------------------------------------------------------------

class host_mode:
    """``with host_mode("flush"):`` drives this thread's FPU and restores it on exit."""

    def __init__(self, policy: str) -> None:
        self.flush = policy == "flush"

    def __enter__(self) -> bool:
        if platform.machine().lower() not in ("arm64", "aarch64"):
            # The specification is THIS host's float64 unit as the Metal table's host has
            # it: arm64 picks NaN operands and decides the flush policy's tininess its own
            # way (before rounding), and an x86 host differs on both by design.
            requires_resource_skip("arm64_host", "the helpers are specified by an arm64 "
                                   "host's float64 unit")
        self.before = backends.subnormals_flushed()
        if self.before != self.flush:
            subnormal_policy._drive_host_fpu(self.flush)  # noqa: SLF001
        if backends.subnormals_flushed() != self.flush:
            requires_resource_skip("host_fpu_mode", "this host's FPU mode cannot be driven")
        return self.flush

    def __exit__(self, *exc: Any) -> None:
        if backends.subnormals_flushed() != self.before:
            subnormal_policy._drive_host_fpu(self.before)  # noqa: SLF001
        assert backends.subnormals_flushed() == self.before, "the host FPU mode was not restored"


# ---------------------------------------------------------------------------
# Operand classes
# ---------------------------------------------------------------------------

def _floats(rng, n: int, lo: int, hi: int) -> np.ndarray:
    """float32 values with a uniform 24-bit significand and exponent in [lo, hi], random sign."""
    mant = (rng.integers(0, 1 << 23, size=n, dtype=np.int64) | (1 << 23)).astype(np.float64)
    exps = rng.integers(lo, hi + 1, size=n)
    sign = np.where(rng.integers(0, 2, size=n) == 0, 1.0, -1.0)
    return (sign * mant * np.ldexp(1.0, exps - 23)).astype(np.float32)


def make_rows(name: str, rng, n: int) -> Tuple[np.ndarray, ...]:
    """``(a, t1, t2, flags)`` for one row class: float32 words (uint32) and presence flags."""
    if name == "spread":
        a, t1, t2 = (_floats(rng, n, -30, 30) for _ in range(3))
    elif name == "cancel":
        a = _floats(rng, n, -10, 10)
        d = np.ldexp(1.0, -rng.integers(1, 25, size=n)) * rng.choice([-1.0, 1.0], size=n)
        t1 = (-4.0 * a.astype(np.float64) * (1.0 + d)).astype(np.float32)
        t2 = _floats(rng, n, -40, 10)
    elif name == "gap_29_31":
        a = _floats(rng, n, -5, 5)
        ea = np.frexp(a.astype(np.float64))[1]
        terms = []
        for lo, hi in ((29, 31), (28, 32)):
            gap = rng.integers(lo, hi + 1, size=n)
            mant = (rng.integers(0, 1 << 23, size=n, dtype=np.int64) | (1 << 23)).astype(np.float64)
            sign = rng.choice([-1.0, 1.0], size=n)
            terms.append((sign * mant * np.ldexp(1.0, ea - 24 - gap + 2)).astype(np.float32))
        t1, t2 = terms
    elif name == "float_ties":
        a = _floats(rng, n, -8, 8)
        ulp = np.spacing(np.abs(a)).astype(np.float64)
        k = rng.choice([-1.0, 1.0], size=n) * (2.0 * rng.integers(0, 4, size=n) + 1.0)
        t1 = (2.0 * ulp * k).astype(np.float32)          # a + t1/4 = a + k*ulp/2: a midpoint
        t2 = np.where(rng.integers(0, 4, size=n) == 0, np.float32(0.0),
                      (2.0 * ulp * rng.choice([-1.0, 1.0], size=n)).astype(np.float32))
    elif name == "zeros":
        a = _floats(rng, n, -6, 6)
        t1 = _floats(rng, n, -4, 8)
        t2 = _floats(rng, n, -4, 8)
        for values in (a, t1, t2):
            pick = rng.integers(0, 3, size=n)
            values[pick == 0] = np.float32(0.0)
            values[pick == 1] = np.float32(-0.0)
        pick = rng.integers(0, 4, size=n) == 0
        t1[pick] = (-4.0 * a[pick].astype(np.float64)).astype(np.float32)
    elif name == "subnormal":
        a = _floats(rng, n, -150, -118)
        t1 = _floats(rng, n, -150, -116)
        t2 = _floats(rng, n, -150, -116)
    elif name == "band":
        # a = 2**-126 + j 2**-149 and t1 = -(2**-149 + j 2**-147), both NORMAL and exact
        # float32 values for 2**21 <= j < 2**22 (|t1| stays below 2**-125, where the
        # spacing is still 2**-149), so a + t1/4 = 2**-126 - 2**-151 exactly.
        j = rng.integers(1 << 21, 1 << 22, size=n).astype(np.float64)
        sign = rng.choice([-1.0, 1.0], size=n)
        a = (sign * (np.ldexp(1.0, -126) + j * np.ldexp(1.0, -149))).astype(np.float32)
        t1 = (-sign * (np.ldexp(1.0, -149) + j * np.ldexp(1.0, -147))).astype(np.float32)
        pick = rng.integers(0, 4, size=n)
        t2 = np.select([pick == 0, pick == 1, pick == 2],
                       [np.full(n, 0.0, np.float32), np.full(n, -0.0, np.float32),
                        (sign * np.ldexp(1.0, -149)).astype(np.float32)],   # back up to 2**-126
                       _floats(rng, n, -128, -124))
    else:
        raise ValueError(name)
    flags = rng.integers(0, 4, size=n).astype(np.uint32)
    if name == "band":
        flags |= np.uint32(1)
    return (a.view(np.uint32), t1.view(np.uint32), t2.view(np.uint32), flags)


def _bits64(rng, n: int) -> np.ndarray:
    raw = rng.integers(0, 1 << 62, size=n, dtype=np.int64).astype(np.uint64) * np.uint64(4) \
        + rng.integers(0, 4, size=n, dtype=np.int64).astype(np.uint64)
    frac = raw & np.uint64(0x000FFFFFFFFFFFFF)
    sign = rng.integers(0, 2, size=n).astype(np.uint64) << np.uint64(63)
    pick = rng.integers(0, 8, size=n)
    exps = np.select(
        [pick == 0, pick == 1, pick == 2, pick == 3, pick == 4],
        [np.zeros(n, np.uint64), np.full(n, 0x7FF, np.uint64),
         rng.integers(1, 60, size=n).astype(np.uint64),
         rng.integers(1023 - 160, 1023 + 140, size=n).astype(np.uint64),
         rng.integers(1990, 2047, size=n).astype(np.uint64)],
        (raw >> np.uint64(52)) & np.uint64(0x7FF))
    out = sign | (exps << np.uint64(52)) | frac
    out[rng.integers(0, n, n // 50)] = np.uint64(0)
    out[rng.integers(0, n, n // 50)] = np.uint64(0x8000000000000000)
    out[rng.integers(0, n, n // 100)] = np.uint64(0x7FF0000000000000)    # +inf
    out[rng.integers(0, n, n // 100)] = np.uint64(0xFFF0000000000000)    # -inf
    return out


def _bits32(rng, n: int) -> np.ndarray:
    words = rng.integers(0, 1 << 32, size=n, dtype=np.int64).astype(np.uint32)
    pick = rng.integers(0, 4, size=n)
    words = np.where(pick == 0, words & np.uint32(0x807FFFFF), words)   # subnormals and zeros
    words = np.where(pick == 1, words | np.uint32(0x7F800000), words)   # NaNs
    words[rng.integers(0, n, n // 100)] = np.uint32(0x7F800000)          # +inf
    words[rng.integers(0, n, n // 100)] = np.uint32(0xFF800000)          # -inf
    return words


def _stored_words(rng, n: int) -> np.ndarray:
    """Stored float32 words: physical magnitudes, subnormals, both zeros, a few specials."""
    words = _floats(rng, n, -20, 10).view(np.uint32).copy()
    pick = rng.integers(0, 16, size=n)
    words[pick == 0] = 0
    words[pick == 1] = 0x80000000
    sub = rng.integers(1, 1 << 23, size=n).astype(np.uint32) | (rng.integers(0, 2, size=n).astype(np.uint32) << 31)
    words = np.where(pick == 2, sub, words)
    words = np.where(pick == 3, _floats(rng, n, -126, -110).view(np.uint32), words)
    words = np.where(pick == 4, _bits32(rng, n), words)
    return words


def make_ops(name: str, rng, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(w, x, y)`` for one single-operation class: float32 words and two binary64 words."""
    if name == "bits":
        return _bits32(rng, n), _bits64(rng, n), _bits64(rng, n)
    if name == "deposit":
        w = _stored_words(rng, n)
        mant = rng.random(n) + 1.0
        x = rng.choice([-1.0, 1.0], size=n) * mant * np.ldexp(1.0, rng.integers(-1074, 8, size=n))
        pick = rng.integers(0, 8, size=n)
        x[pick == 0] = 0.0
        x[pick == 1] = -0.0
        x = np.where(pick == 2, mant * np.ldexp(1.0, rng.integers(-60, 4, size=n)), x)
        # Planted float32 midpoints: w - x = w + k * ulp(w) / 2 with k odd.
        wf = w.view(np.float32)
        finite = np.isfinite(wf)
        ulp = np.spacing(np.abs(np.where(finite, wf, np.float32(1.0)))).astype(np.float64)
        k = rng.choice([-1.0, 1.0], size=n) * (2.0 * rng.integers(0, 3, size=n) + 1.0)
        x = np.where((pick == 3) & finite, -k * ulp / 2.0, x)
        y = np.where(finite, wf, np.float32(0.0)).astype(np.float64)
        y = np.where(rng.integers(0, 2, size=n) == 0, y, -y * mant)
        return w, x.view(np.uint64), y.view(np.uint64)
    if name == "band":
        low = np.ldexp(1.0, -126) * (1.0 - np.ldexp(1.0, -25))
        sign = rng.choice([-1.0, 1.0], size=n)
        inside = low + rng.random(n) * (np.ldexp(1.0, -126) - low)
        around = np.ldexp(1.0, -126) * (1.0 + (rng.random(n) - 0.5) * np.ldexp(1.0, -20))
        pick = rng.integers(0, 4, size=n)
        x = sign * np.select([pick == 0, pick == 1], [inside, around],
                             np.ldexp(1.0, -151) * rng.random(n))
        w = np.where(pick >= 2, (sign * np.ldexp(1.0, -126)).astype(np.float32),
                     _floats(rng, n, -127, -124)).view(np.uint32)
        y = sign * np.select([pick == 0, pick == 1], [np.ldexp(1.0, -151) * rng.random(n), -inside],
                             np.ldexp(1.0, -126) * (1.0 + rng.random(n) * np.ldexp(1.0, -24)))
        return w, x.view(np.uint64), y.view(np.uint64)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# References: the host's float64 unit, in whatever mode the host is in
# ---------------------------------------------------------------------------

def reference_rows(a, t1, t2, flags, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The row in NumPy float64, on this host's unit in its current mode."""
    scale = TWO ** k
    r = a.view(np.float32).astype(np.float64)
    q1 = scale * t1.view(np.float32).astype(np.float64)
    q2 = scale * t2.view(np.float32).astype(np.float64)
    r1 = np.where((flags & 1) != 0, r + q1, r)
    r2 = np.where((flags & 2) != 0, r1 + q2, r1)
    return r1.view(np.uint64), r2.view(np.uint64), r2.astype(np.float32).view(np.uint32)


def reference_ops(w, x, y, k: int) -> Dict[str, np.ndarray]:
    xf, yf = x.view(np.float64), y.view(np.float64)
    with np.errstate(all="ignore"):
        return {
            "widen": w.view(np.float32).astype(np.float64).view(np.uint64),
            "add": (xf + yf).view(np.uint64),
            "sub": (xf - yf).view(np.uint64),
            "scale": (TWO ** k * xf).view(np.uint64),
            "narrow": xf.astype(np.float32).view(np.uint32),
            "deposit": host_writes._rounded_difference(  # noqa: SLF001 - the array path's door
                w.view(np.float32), xf, None).view(np.uint32),
        }


def transcribed_rows(a, t1, t2, flags, k: int, fz: bool) -> Tuple[np.ndarray, ...]:
    r = softfp64.widen(a, fz)
    r1 = np.where((flags & 1) != 0,
                  softfp64.add(r, softfp64.scale_pow2(softfp64.widen(t1, fz), k, fz), fz), r)
    r2 = np.where((flags & 2) != 0,
                  softfp64.add(r1, softfp64.scale_pow2(softfp64.widen(t2, fz), k, fz), fz), r1)
    return r1, r2, softfp64.narrow(r2, fz)


def transcribed_ops(w, x, y, k: int, fz: bool) -> Dict[str, np.ndarray]:
    return {
        "widen": softfp64.widen(w, fz),
        "add": softfp64.add(x, y, fz),
        "sub": softfp64.sub(x, y, fz),
        "scale": softfp64.scale_pow2(x, k, fz),
        "narrow": softfp64.narrow(x, fz),
        "deposit": softfp64.subtract_from_word(w, x, fz),
    }


def _differ(got: Dict[str, np.ndarray], want: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {name: int(np.count_nonzero(got[name] != want[name])) for name in want}


def _rows_as_dict(r1, r2, out) -> Dict[str, np.ndarray]:
    return {"r1": r1, "r2": r2, "out": out}


# ---------------------------------------------------------------------------
# The device leg's kernels and launcher
# ---------------------------------------------------------------------------

BATTERY_KERNELS = r"""
kernel void sd_rows(device ulong* r1o           [[buffer(0)]],
                    device ulong* r2o           [[buffer(1)]],
                    device uint* outo           [[buffer(2)]],
                    device const uint* a        [[buffer(3)]],
                    device const uint* t1       [[buffer(4)]],
                    device const uint* t2       [[buffer(5)]],
                    device const uint* flags    [[buffer(6)]],
                    constant uint& n            [[buffer(7)]],
                    constant int& k             [[buffer(8)]],
                    constant uint& fz           [[buffer(9)]],
                    uint g [[thread_position_in_grid]])
{
    if (g >= n) { return; }
    bool z = fz != 0u;
    uint f = flags[g];
    ulong r = sd_widen(a[g], z);
    if ((f & 1u) != 0u) { r = sd_add(r, sd_scale_pow2(sd_widen(t1[g], z), k, z), z); }
    r1o[g] = r;
    if ((f & 2u) != 0u) { r = sd_add(r, sd_scale_pow2(sd_widen(t2[g], z), k, z), z); }
    r2o[g] = r;
    outo[g] = sd_narrow(r, z);
}

kernel void sd_ops(device ulong* widened        [[buffer(0)]],
                   device ulong* summed         [[buffer(1)]],
                   device ulong* differenced    [[buffer(2)]],
                   device ulong* scaled         [[buffer(3)]],
                   device uint* narrowed        [[buffer(4)]],
                   device uint* deposited       [[buffer(5)]],
                   device ulong* widened_f      [[buffer(6)]],
                   device float* narrowed_f     [[buffer(7)]],
                   device const uint* w         [[buffer(8)]],
                   device const ulong* x        [[buffer(9)]],
                   device const ulong* y        [[buffer(10)]],
                   constant uint& n             [[buffer(11)]],
                   constant int& k              [[buffer(12)]],
                   constant uint& fz            [[buffer(13)]],
                   uint g [[thread_position_in_grid]])
{
    if (g >= n) { return; }
    bool z = fz != 0u;
    widened[g] = sd_widen(w[g], z);
    summed[g] = sd_add(x[g], y[g], z);
    differenced[g] = sd_sub(x[g], y[g], z);
    scaled[g] = sd_scale_pow2(x[g], k, z);
    narrowed[g] = sd_narrow(x[g], z);
    deposited[g] = sd_subtract_from_word(w[g], x[g], z);
    widened_f[g] = sd_widen_f(as_type<float>(w[g]), z);
    narrowed_f[g] = sd_narrow_f(x[g], z);
}
"""


def battery_source(contract: str = shaders.CONTRACT_OFF,
                   mutation: Tuple[str, str] = None) -> str:
    helpers = softfp64.MSL_HELPERS
    if mutation is not None:
        old, new = mutation
        assert helpers.count(old) == 1, (old, helpers.count(old))
        helpers = helpers.replace(old, new)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            + shaders.contraction_pragma(contract) + "\n" + helpers + BATTERY_KERNELS)


def require_mps() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("torch", "PyTorch is not installed")
    if not torch.backends.mps.is_available():
        requires_resource_skip("mps_device", "the helpers are launched on an Apple GPU")
    return torch


_LIBRARIES: Dict[Any, Any] = {}


def compiled(contract: str = shaders.CONTRACT_OFF, mutation: Tuple[str, str] = None) -> Any:
    key = (contract, mutation)
    if key not in _LIBRARIES:
        torch = require_mps()
        _LIBRARIES[key] = torch.mps.compile_shader(battery_source(contract, mutation))
    return _LIBRARIES[key]


def _dev(torch, array: np.ndarray) -> Any:
    view = array.view(np.int64) if array.dtype == np.uint64 else array.view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(view)).to("mps")


def device_rows(library, a, t1, t2, flags, k: int, fz: bool) -> Tuple[np.ndarray, ...]:
    torch = require_mps()
    n = int(a.size)
    r1 = torch.empty(n, dtype=torch.int64, device="mps")
    r2 = torch.empty(n, dtype=torch.int64, device="mps")
    out = torch.empty(n, dtype=torch.int32, device="mps")
    library.sd_rows(r1, r2, out, _dev(torch, a), _dev(torch, t1), _dev(torch, t2),
                    _dev(torch, flags), n, int(k), int(fz), threads=n)
    torch.mps.synchronize()
    return (r1.cpu().numpy().view(np.uint64), r2.cpu().numpy().view(np.uint64),
            out.cpu().numpy().view(np.uint32))


def device_ops(library, w, x, y, k: int, fz: bool) -> Dict[str, np.ndarray]:
    torch = require_mps()
    n = int(w.size)
    wide = [torch.empty(n, dtype=torch.int64, device="mps") for _ in range(4)]
    narrow = [torch.empty(n, dtype=torch.int32, device="mps") for _ in range(2)]
    wide_f = torch.empty(n, dtype=torch.int64, device="mps")
    narrow_f = torch.empty(n, dtype=torch.float32, device="mps")
    library.sd_ops(*wide, *narrow, wide_f, narrow_f, _dev(torch, w), _dev(torch, x),
                   _dev(torch, y), n, int(k), int(fz), threads=n)
    torch.mps.synchronize()
    names = ("widen", "add", "sub", "scale")
    out = {name: tensor.cpu().numpy().view(np.uint64) for name, tensor in zip(names, wide)}
    out["narrow"] = narrow[0].cpu().numpy().view(np.uint32)
    out["deposit"] = narrow[1].cpu().numpy().view(np.uint32)
    out["widen_f"] = wide_f.cpu().numpy().view(np.uint64)
    out["narrow_f"] = narrow_f.cpu().numpy().view(np.uint32)
    return out


def _seed(*labels: Any) -> int:
    """A fixed seed per (kind, class, mode, scale): the same operands on every run and host."""
    return int.from_bytes(hashlib.sha256(repr(("softfp64",) + labels).encode()).digest()[:8], "little")


def run_class(kind: str, name: str, policy: str, triples: int, *,
              library: Any = None, against: str = "host",
              k: int = None) -> Dict[str, Any]:
    """Count differing words for one class in one mode, in chunks of :data:`CHUNK`.

    ``kind`` is ``rows`` or ``ops``; ``against`` is ``host`` (the float64 unit) or
    ``transcription``; ``library`` None compares the transcription itself with the host.
    Returns per-output differing counts, the word count and the seconds taken.

    THE OPERANDS ARE ALWAYS BUILT WITH THE HOST KEEPING. A class built under flush would
    lose its own subnormals in the arithmetic that constructs them (``mant * 2**-1074``
    is a product, and the flushing FPU returns zero), and the subnormal legs would then
    certify nothing; only the reference statement runs in the mode under test.
    """
    rng = np.random.default_rng(_seed(kind, name, policy, k))
    counts: Dict[str, int] = {}
    words = 0
    t0 = time.time()
    fz = policy == "flush"
    done = 0
    while done < triples:
        n = min(CHUNK, triples - done)
        with host_mode("keep"):
            operands = make_rows(name, rng, n) if kind == "rows" else make_ops(name, rng, n)
        if kind == "rows":
            if against == "host":
                with host_mode(policy):
                    want = _rows_as_dict(*reference_rows(*operands, k))
            else:
                want = _rows_as_dict(*transcribed_rows(*operands, k, fz))
            got = (_rows_as_dict(*transcribed_rows(*operands, k, fz)) if library is None
                   else _rows_as_dict(*device_rows(library, *operands, k, fz)))
        else:
            if against == "host":
                with host_mode(policy):
                    want = reference_ops(*operands, k)
            else:
                want = transcribed_ops(*operands, k, fz)
            got = (transcribed_ops(*operands, k, fz) if library is None
                   else device_ops(library, *operands, k, fz))
            if library is not None:
                want = dict(want, widen_f=want["widen"], narrow_f=want["narrow"])
        for output, count in _differ(got, want).items():
            counts[output] = counts.get(output, 0) + count
        words += n * len(want)
        done += n
    return {"differing": counts, "total_differing": sum(counts.values()), "words": words,
            "triples": triples, "seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# HOST: the fragment's shape
# ---------------------------------------------------------------------------

def test_the_fragment_spells_each_signature_of_the_api_once():
    for signature in softfp64.SIGNATURES:
        assert softfp64.MSL_HELPERS.count(signature + " {") == 1, signature
    spelled = re.findall(r"^static inline \w+ (sd_\w+)\(", softfp64.MSL_HELPERS, re.M)
    public = {re.search(r"(sd_\w+)\(", s).group(1) for s in softfp64.SIGNATURES}
    internal = {"sd_is_nan", "sd_pick_nan", "sd_round_pack"}
    assert set(spelled) == public | internal and len(spelled) == len(set(spelled)), spelled


def test_the_fragment_is_guarded_and_does_no_floating_point_arithmetic():
    """Integers only: the float type appears in the two register forms' signatures and
    their reinterpretations, nowhere else, so no contraction or fast-math mode reaches it."""
    text = softfp64.MSL_HELPERS
    assert text.lstrip().startswith(f"#ifndef {softfp64.GUARD}\n#define {softfp64.GUARD}")
    assert text.rstrip().endswith("#endif")
    code = "\n".join(line.split("//")[0].rstrip() for line in text.splitlines())
    float_lines = [line.strip() for line in code.splitlines() if "float" in line]
    assert float_lines == [
        "static inline ulong sd_widen_f(float v, bool fz) {",
        "static inline float sd_narrow_f(ulong x, bool fz) {",
        "return as_type<float>(sd_narrow(x, fz));",
    ], float_lines
    assert "double" not in code and not re.search(r"\d\.\d*f?\b", code.replace("0x", ""))


def test_library_source_carries_the_shipped_contraction_directive():
    source = softfp64.library_source("kernel void k() {}\n")
    assert shaders.contraction_pragma(shaders.CONTRACT_OFF) in source
    assert source.index(shaders.contraction_pragma(shaders.CONTRACT_OFF)) < source.index(softfp64.GUARD)


# ---------------------------------------------------------------------------
# HOST: the transcription against the host's float64 unit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", ROW_CLASSES)
def test_the_transcription_forms_the_row_as_the_host_does(name, policy):
    for k in ROW_SCALES:
        result = run_class("rows", name, policy, HOST_TRIPLES, k=k)
        _say(f"host rows {name} k={k} {policy}: {result['total_differing']} / {result['words']}")
        assert result["total_differing"] == 0, (name, k, policy, result)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", OP_CLASSES)
def test_the_transcription_performs_each_operation_as_the_host_does(name, policy):
    result = run_class("ops", name, policy, HOST_TRIPLES, k=OP_SCALES[name])
    _say(f"host ops {name} {policy}: {result['total_differing']} / {result['words']}")
    assert result["total_differing"] == 0, (name, policy, result)


# ---------------------------------------------------------------------------
# CONTROLS: each class constructs what it claims
# ---------------------------------------------------------------------------

def _census_rows(name: str, n: int = 50_000) -> Dict[str, int]:
    rng = np.random.default_rng(_seed("census", name))
    with host_mode("keep"):
        a, t1, t2, flags = make_rows(name, rng, n)
        r1, r2, out = reference_rows(a, t1, t2, flags, softfp64.QUARTER)
    r1f, r2f = r1.view(np.float64), r2.view(np.float64)
    af = a.view(np.float32).astype(np.float64)
    q1 = 0.25 * t1.view(np.float32).astype(np.float64)
    term = (flags & 1) != 0
    # TwoSum's error: nonzero exactly when the double addition rounded.
    bb = r1f - af
    err = (af - (r1f - bb)) + (q1 - bb)
    lo = np.nextafter(r2f.astype(np.float32), np.float32(-np.inf)).astype(np.float64)
    hi = np.nextafter(r2f.astype(np.float32), np.float32(np.inf)).astype(np.float64)
    mid = (r2f == (lo + r2f.astype(np.float32).astype(np.float64)) / 2) | \
        (r2f == (hi + r2f.astype(np.float32).astype(np.float64)) / 2)
    words32 = np.concatenate([a, t1, t2])
    subnormal32 = ((words32 & 0x7F800000) == 0) & ((words32 & 0x007FFFFF) != 0)
    out_sub = ((out & 0x7F800000) == 0) & ((out & 0x007FFFFF) != 0)
    band_lo = np.ldexp(1.0, -126) * (1.0 - np.ldexp(1.0, -25))
    return {
        "rounded_adds": int(np.count_nonzero(term & (err != 0))),
        "float32_midpoints": int(np.count_nonzero(mid)),
        "deep_cancellations": int(np.count_nonzero(term & (np.abs(r1f) < np.abs(af) * 2.0 ** -20))),
        "negative_zero_results": int(np.count_nonzero(r2 == np.uint64(1 << 63))),
        "positive_zero_results": int(np.count_nonzero(r2 == 0)),
        "subnormal_operands": int(np.count_nonzero(subnormal32)),
        "subnormal_results": int(np.count_nonzero(out_sub)),
        "band_results": int(np.count_nonzero((np.abs(r2f) >= band_lo) & (np.abs(r2f) < np.ldexp(1.0, -126)))),
    }


#: Per class, the census entry it exists for and a floor of about half what 50,000
#: triples measured on this host (spread 6,337 rounded adds; cancel 4,639 deep
#: cancellations; gap_29_31 10,369 rounded adds; float_ties 24,968 midpoints; zeros
#: 19,072 / 7,000 positive / negative zero results; subnormal 27,654 subnormal results;
#: band 12,501 results in the band).
CENSUS_FLOORS = {
    "spread": ("rounded_adds", 3_000),
    "cancel": ("deep_cancellations", 2_000),
    "gap_29_31": ("rounded_adds", 5_000),
    "float_ties": ("float32_midpoints", 10_000),
    "zeros": ("positive_zero_results", 5_000),
    "subnormal": ("subnormal_results", 10_000),
    "band": ("band_results", 5_000),
}


@pytest.mark.parametrize("name", ROW_CLASSES)
def test_each_row_class_constructs_what_it_claims(name):
    census = _census_rows(name)
    key, floor = CENSUS_FLOORS[name]
    assert census[key] >= floor, (name, census)
    if name == "zeros":
        assert census["negative_zero_results"] >= 2_000, census
    if name == "subnormal":
        assert census["subnormal_operands"] >= 50_000, census


def test_the_band_is_where_the_two_policies_part():
    """On the band class every narrowed word differs between keep and flush, and on the
    deposit class the subnormal stored words do: the policy is load-bearing there."""
    rng = np.random.default_rng(7)
    with host_mode("keep"):
        a, t1, t2, flags = make_rows("band", rng, 20_000)
    words = {}
    for policy in POLICIES:
        with host_mode(policy):
            words[policy] = reference_rows(a, t1, t2, flags, softfp64.QUARTER)[2]
    in_band = (t2.view(np.float32) == 0) | ((flags & 2) == 0)
    assert np.count_nonzero(in_band) > 5_000
    assert np.all(words["keep"][in_band] & 0x7FFFFFFF == 0x00800000)
    assert np.all(words["flush"][in_band] & 0x7FFFFFFF == 0)


def test_the_op_classes_reach_every_kind_of_binary64():
    rng = np.random.default_rng(8)
    with host_mode("keep"):
        w, x, y = make_ops("bits", rng, 50_000)
    xf = x.view(np.float64)
    exponent = (x >> np.uint64(52)) & np.uint64(0x7FF)
    fraction = x & np.uint64(0x000FFFFFFFFFFFFF)
    assert np.count_nonzero((exponent == 0) & (fraction != 0)) > 1_000      # subnormals
    assert np.count_nonzero(np.isinf(xf)) > 500 and np.count_nonzero(np.isnan(xf)) > 1_000
    both = np.isinf(xf) & np.isinf(y.view(np.float64)) & ((x ^ y) >= np.uint64(1 << 63))
    assert np.count_nonzero(both) > 0                                        # inf - inf, inf + -inf
    signalling = np.isnan(xf) & ((x & np.uint64(1 << 51)) == 0)
    assert np.count_nonzero(signalling) > 500
    with host_mode("keep"):
        w, x, y = make_ops("deposit", rng, 50_000)
    xf = x.view(np.float64)
    assert np.count_nonzero((xf != 0) & (np.abs(xf) < np.ldexp(1.0, -1022))) > 50
    with host_mode("keep"), np.errstate(all="ignore"):
        deposited = host_writes._rounded_difference(w.view(np.float32), xf, None)  # noqa: SLF001
        wide = w.view(np.float32).astype(np.float64) - xf
    finite = np.isfinite(w.view(np.float32))
    tie = finite & (wide != deposited.astype(np.float64)) & \
        (np.abs(wide - deposited.astype(np.float64)) == np.spacing(np.abs(deposited)).astype(np.float64) / 2)
    assert np.count_nonzero(tie) > 1_000, np.count_nonzero(tie)


# ---------------------------------------------------------------------------
# CONTROLS: the host comparison catches planted defects in the transcription
# ---------------------------------------------------------------------------

def _truncating_round_pack(sign, e, m, fz):
    return _ORIGINAL_ROUND_PACK(sign, e, np.asarray(m, dtype=np.uint64) & ~np.uint64(7), fz)


_ORIGINAL_ROUND_PACK = softfp64._round_pack  # noqa: SLF001

HOST_MUTANTS: Dict[str, Tuple[str, Callable[..., Any]]] = {
    "round_truncates": ("_round_pack", _truncating_round_pack),
    "no_daz": ("daz", lambda x: np.asarray(x, dtype=np.uint64)),
    "widen_ignores_the_policy": ("widen", (lambda original: lambda w, fz: original(w, False))(softfp64.widen)),
    "narrow_ignores_the_policy": ("narrow", (lambda original: lambda x, fz: original(x, False))(softfp64.narrow)),
}


@pytest.mark.parametrize("mutant", sorted(HOST_MUTANTS))
def test_the_host_comparison_catches_each_planted_defect(mutant, monkeypatch):
    attribute, replacement = HOST_MUTANTS[mutant]
    monkeypatch.setattr(softfp64, attribute, replacement)
    caught = 0
    for policy in POLICIES:
        for name in ("gap_29_31", "subnormal", "band"):
            caught += run_class("rows", name, policy, 20_000, k=softfp64.QUARTER)["total_differing"]
        caught += run_class("ops", "deposit", policy, 20_000, k=softfp64.QUARTER)["total_differing"]
    assert caught > 0, mutant


# ---------------------------------------------------------------------------
# DEVICE: the MSL on the Apple GPU
# ---------------------------------------------------------------------------

#: Planted defects in the MSL, each an exact substring replacement: the device leg must
#: report differing words under every one, or it is not measuring that rule.
MUTANTS: Dict[str, Tuple[str, str]] = {
    "round_truncates": ("if (g > 4ul || (g == 4ul && (m & 1ul) != 0ul)) { m += 1ul; }",
                        "if (false) { m += 1ul; }"),
    "round_ties_away": ("if (g > 4ul || (g == 4ul && (m & 1ul) != 0ul)) { m += 1ul; }",
                        "if (g >= 4ul) { m += 1ul; }"),
    "add_drops_sticky": ("my = (my >> ulong(d)) | ((lost != 0ul) ? 1ul : 0ul);",
                         "my = (my >> ulong(d));"),
    "subnormal_result_drops_sticky": (
        ": ((m >> ulong(r)) | (((m & ((1ul << ulong(r)) - 1ul)) != 0ul) ? 1ul : 0ul));",
        ": (m >> ulong(r));"),
    "narrow_truncates": ("if (rem > midpoint || (rem == midpoint && (q & 1ul) != 0ul)) { q += 1ul; }",
                         "if (false) { q += 1ul; }"),
    "narrow_ties_away": ("if (rem > midpoint || (rem == midpoint && (q & 1ul) != 0ul)) { q += 1ul; }",
                         "if (rem >= midpoint) { q += 1ul; }"),
    "narrow_rounds_the_band_under_flush": ("if (e <= 0 && (fz || e <= -24)) { return s; }",
                                           "if (e <= 0 && e <= -24) { return s; }"),
    "narrow_flushes_under_keep": ("if (e <= 0 && (fz || e <= -24)) { return s; }",
                                  "if (e <= 0) { return s; }"),
    "widen_keeps_under_flush": ("if (m == 0u || fz) { return s; }", "if (m == 0u) { return s; }"),
    "widen_subnormal_exponent": ("ulong(874u + p)", "ulong(875u + p)"),
    "add_reads_a_subnormal_y_under_flush": ("if (fz) { x = sd_daz(x); y = sd_daz(y); }",
                                            "if (fz) { x = sd_daz(x); }"),
    "cancel_negative_zero": ("if (m == 0ul) { return 0ul; }", "if (m == 0ul) { return x & SD_SIGN; }"),
    "zero_sum_sign": ("return (ax == 0ul) ? (x & y) : x;", "return (ax == 0ul) ? (x | y) : x;"),
    "sub_negates_a_nan": ("if (sd_is_nan(x) || sd_is_nan(y)) { return sd_pick_nan(x, y); }",
                          "if (false) { return sd_pick_nan(x, y); }"),
    "second_quiet_nan_first": ("(sd_is_nan(x) ? x : y));", "(sd_is_nan(y) ? y : x));"),
    "narrow_drops_the_nan_payload": ("0x7FC00000u | uint((ax >> 29) & 0x3FFFFFul)", "0x7FC00000u"),
}


def test_the_helpers_compile_in_the_shipped_mode_and_the_mode_moves_no_word():
    """Integer-only helpers: the fast contraction mode is a recorded NULL control."""
    off = compiled(shaders.CONTRACT_OFF)
    fast = compiled(shaders.CONTRACT_FAST)
    rng = np.random.default_rng(3)
    for name in ("gap_29_31", "float_ties", "subnormal", "band"):
        with host_mode("keep"):
            a, t1, t2, flags = make_rows(name, rng, 200_000)
        for fz in (False, True):
            got_off = device_rows(off, a, t1, t2, flags, softfp64.QUARTER, fz)
            got_fast = device_rows(fast, a, t1, t2, flags, softfp64.QUARTER, fz)
            for x, y in zip(got_off, got_fast):
                assert np.array_equal(x, y), (name, fz)
    with host_mode("keep"):
        w, x, y = make_ops("bits", rng, 200_000)
    for fz in (False, True):
        assert all(np.array_equal(p, q) for p, q in zip(
            device_ops(off, w, x, y, -60, fz).values(), device_ops(fast, w, x, y, -60, fz).values()))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", ROW_CLASSES)
def test_the_device_forms_the_row_as_the_host_does(name, policy):
    library = compiled()
    for k in ROW_SCALES:
        result = run_class("rows", name, policy, DEVICE_TRIPLES, library=library, k=k)
        _say(f"device rows {name} k={k} {policy}: {result['total_differing']} / {result['words']} "
             f"words ({result['triples']} triples, {result['seconds']} s)")
        assert result["total_differing"] == 0, (name, k, policy, result)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", OP_CLASSES)
def test_the_device_performs_each_operation_as_the_host_does(name, policy):
    result = run_class("ops", name, policy, DEVICE_TRIPLES, library=compiled(), k=OP_SCALES[name])
    _say(f"device ops {name} {policy}: {result['total_differing']} / {result['words']} words "
         f"({result['seconds']} s) {result['differing']}")
    assert result["total_differing"] == 0, (name, policy, result)


@pytest.mark.parametrize("policy", POLICIES)
def test_the_msl_is_the_transcription_word_for_word(policy):
    library = compiled()
    total = 0
    for name in ROW_CLASSES:
        for k in ROW_SCALES:
            result = run_class("rows", name, policy, TRANSCRIPTION_TRIPLES, library=library,
                               against="transcription", k=k)
            total += result["total_differing"]
            _say(f"msl vs transcription rows {name} k={k} {policy}: {result['total_differing']} / {result['words']}")
    for name in OP_CLASSES:
        result = run_class("ops", name, policy, TRANSCRIPTION_TRIPLES, library=library,
                           against="transcription", k=OP_SCALES[name])
        total += result["total_differing"]
        _say(f"msl vs transcription ops {name} {policy}: {result['total_differing']} / {result['words']}")
    assert total == 0


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_each_planted_defect_in_the_msl_is_caught_on_the_device(mutant):
    library = compiled(mutation=MUTANTS[mutant])
    caught: Dict[str, int] = {}
    for policy in POLICIES:
        for name in ROW_CLASSES:
            result = run_class("rows", name, policy, MUTANT_TRIPLES, library=library,
                               k=softfp64.QUARTER)
            caught[f"rows/{name}/{policy}"] = result["total_differing"]
        for name in OP_CLASSES:
            result = run_class("ops", name, policy, MUTANT_TRIPLES, library=library,
                               k=OP_SCALES[name])
            caught[f"ops/{name}/{policy}"] = result["total_differing"]
    _say(f"mutant {mutant}: {sum(caught.values())} differing words; "
         f"{ {key: value for key, value in caught.items() if value} }")
    assert sum(caught.values()) > 0, (mutant, caught)


# ---------------------------------------------------------------------------
# THE DEPOSIT DOOR (step M11): Residency.subtract_double_index
# ---------------------------------------------------------------------------
#
# ``host_writes.subtract_double`` hands a held deposit to the residency that owns the
# array. On the device the door is ``door_scatter_sub`` (``sd_subtract_from_word``); its
# specification is the array path's own statement, ``host_writes._rounded_difference``,
# run on the same words in the same host mode. Each case below deposits into a held
# (DEVICE_OWNED) mirror through the engine's door and compares every word of the device
# copy with that statement applied to a host copy.

DOOR_CELLS = 1 << 20
DOOR_DEPOSITS = 4          # disjoint index sets per mirror, so 4M words per real case


def _door_case(torch, dtype, policy: str, name: str, device: str = "mps",
               cells: int = DOOR_CELLS, deposits: int = DOOR_DEPOSITS) -> Dict[str, int]:
    """Deposit ``deposits`` times into a held mirror; count words that differ from the array path."""
    from meep_gpu.metal_kernels.device import DEVICE_OWNED, Residency

    rng = np.random.default_rng(_seed("door", name, policy, np.dtype(dtype).name, device))
    width = 2 if dtype == np.complex64 else 1
    with host_mode("keep"):
        words, _, _ = make_ops(name, rng, cells * width)
        payloads = [make_ops(name, rng, cells * width)[1] for _ in range(deposits)]
    host = words.view(np.float32).view(dtype).copy()
    reference = host.copy()
    order = rng.permutation(cells)
    sets = np.array_split(order, deposits)
    residency = Residency(device=device)
    residency.mirror("field", host, dtype=dtype)
    residency.arm_hold(["field"])
    residency.sync_out(["field"])
    assert residency.state_of("field") == DEVICE_OWNED
    host_writes.register(residency)
    try:
        before = (residency.sparse_gathers, residency.copies_out, residency.double_subtracts)
        with host_mode(policy), np.errstate(all="ignore"):  # the bits class carries NaN and inf
            for linear, payload in zip(sets, payloads):
                linear = np.ascontiguousarray(linear, dtype=np.int64)
                parts = payload.view(np.float64).reshape(-1, width)[: linear.size]
                real64 = np.ascontiguousarray(parts[:, 0])
                imag64 = np.ascontiguousarray(parts[:, 1]) if width == 2 else None
                host_writes.subtract_double(host, linear, real64, imag64)
                flat = reference.reshape(-1)
                flat[linear] = host_writes._rounded_difference(flat[linear], real64, imag64)  # noqa: SLF001
        after = (residency.sparse_gathers, residency.copies_out, residency.double_subtracts)
        assert residency.state_of("field") == DEVICE_OWNED
        device_words = residency.tensor("field").cpu().numpy().reshape(-1).view(np.uint32)
    finally:
        host_writes.unregister(residency)
    want = np.ascontiguousarray(reference).reshape(-1).view(np.uint32)
    return {"words": int(want.size), "differing": int(np.count_nonzero(device_words != want)),
            "sparse_gathers": after[0] - before[0], "copies_out": after[1] - before[1],
            "door_launches": after[2] - before[2], "deposits": deposits}


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("dtype", (np.float32, np.complex64), ids=("real", "complex"))
@pytest.mark.parametrize("name", OP_CLASSES)
def test_the_held_deposit_is_the_array_paths_statement_and_reads_nothing_back(name, dtype, policy):
    torch = require_mps()
    result = _door_case(torch, dtype, policy, name)
    _say(f"door {name} {np.dtype(dtype).name} {policy}: {result['differing']} / {result['words']} "
         f"words; gathers {result['sparse_gathers']}, copies out {result['copies_out']}, "
         f"door launches {result['door_launches']} of {result['deposits']} deposits")
    assert result["differing"] == 0, result
    assert (result["sparse_gathers"], result["copies_out"], result["door_launches"]) == \
        (0, 0, result["deposits"]), result


def test_a_deposit_on_a_host_current_mirror_is_formed_on_the_host_and_moves_no_state():
    torch = require_mps()
    from meep_gpu.metal_kernels.device import CLEAN_BOTH, Residency

    rng = np.random.default_rng(12)
    with host_mode("keep"):
        words, x, _ = make_ops("deposit", rng, 4096)
    host = words.view(np.float32).copy()
    reference = host.copy()
    residency = Residency()
    residency.mirror("field", host)
    host_writes.register(residency)
    try:
        linear = rng.permutation(4096)[:1000].astype(np.int64)
        real64 = x.view(np.float64)[:1000].copy()
        host_writes.subtract_double(host, linear, real64)
        reference[linear] = host_writes._rounded_difference(reference[linear], real64, None)  # noqa: SLF001
        assert residency.state_of("field") == CLEAN_BOTH
        assert residency.double_subtracts == 0 and residency.sparse_gathers == 0
        np.testing.assert_array_equal(host.view(np.uint32), reference.view(np.uint32))
        assert residency.verify() == {}
    finally:
        host_writes.unregister(residency)
    del torch


def test_the_door_refuses_the_storage_the_array_path_refuses():
    require_mps()
    from meep_gpu.metal_kernels.device import Residency

    linear = np.array([0, 1], dtype=np.int64)
    for dtype, imag64, match in ((np.float32, np.zeros(2), "no imaginary plane"),
                                 (np.complex64, None, "needs imag")):
        host = np.zeros(8, dtype=dtype)
        residency = Residency()
        residency.mirror("field", host, dtype=dtype)
        residency.arm_hold(["field"])
        residency.sync_out(["field"])
        with pytest.raises(ValueError, match=match):
            residency.subtract_double_index(host, linear, np.zeros(2), imag64)
        residency.release_hold()


def test_the_cpu_twin_of_the_door_is_the_array_paths_statement():
    """A residency on the CPU (the twin a gate builds) runs the array path's statements."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("torch", "PyTorch is not installed")
    for dtype in (np.float32, np.complex64):
        for policy in POLICIES:
            result = _door_case(torch, dtype, policy, "deposit", device="cpu", cells=1 << 14)
            assert result["differing"] == 0 and result["door_launches"] == result["deposits"], result


@pytest.mark.parametrize("mutant", ("narrow_rounds_the_band_under_flush", "widen_keeps_under_flush",
                                    "add_reads_a_subnormal_y_under_flush", "narrow_ties_away"))
def test_a_planted_defect_in_the_door_is_caught(mutant, monkeypatch):
    """The door's own comparison sees each planted helper defect (compiled into the door)."""
    torch = require_mps()
    from meep_gpu.metal_kernels import device

    old, new = MUTANTS[mutant]
    assert device._DOOR_SOURCE.count(old) == 1  # noqa: SLF001
    monkeypatch.setattr(device, "_DOOR_SOURCE", device._DOOR_SOURCE.replace(old, new))  # noqa: SLF001
    monkeypatch.setattr(device, "_DOOR_LIBRARY", {})  # noqa: SLF001 - recompiled from the mutant
    differing = 0
    for name in OP_CLASSES:
        differing += _door_case(torch, np.float32, "flush", name, cells=1 << 16)["differing"]
    assert differing > 0, mutant
