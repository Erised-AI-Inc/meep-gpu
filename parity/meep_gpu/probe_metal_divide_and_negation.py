"""THE DIVIDE GATE: is Metal's ``/`` IEEE round-to-nearest float32 on this host?

WHY THIS EXISTS, AND WHY IT BLOCKS THE NONLINEAR FAMILY. Every Metal family
shipped so far is division-free, and ``shaders.py``'s transcription rules say so in
as many words: "Neither body divides, calls ``sqrt``, or takes a min or a max,
which is why this pair is the lowest-risk one to certify first" (shaders.py:44-46).
The chi2/chi3 Pade factor is the FIRST quotient this package will emit::

    u = ((1 + c2) + 2*c3) / ((1 + 2*c2) + 3*c3)      # stepping.py:1054-1056

so the platform's division semantics stop being someone else's problem.

THE SIBLING TRACK MEASURED THIS AND GOT A BAD ANSWER, WHICH IS THE REASON THIS
PROBE IS NOT OPTIONAL. On Triton/NVPTX the plain ``/`` lowers to ``div.full.f32``
(~2 ulp) and EVERY nonlinear sweep case diverged while the linear arm stayed
byte-identical; ``tl.math.div_rn`` (``div.rn.ftz.f32``) fixed it
(``triton_kernels/nonlinear_update_e.py``, docstring point 4). That is a fact about
one compiler and it transfers to Metal exactly as far as the CUDA arity result
transferred — not at all. Metal has THREE spellings (``/``, ``fast::divide``,
``precise::divide``) and which one the plain operator resolves to under
``torch.mps.compile_shader``'s default options is UNMEASURED on this host.

WHAT THIS PROBE CANNOT DO, STATED RATHER THAN PAPERED OVER. ``compile_shader``
exposes no AIR, no GPU ISA and no optimisation report (``metal_kernels/device.py:
46-62``), so this cannot be settled the way the CUDA round settled its arity
question — by reading the generated code. Everything here is BEHAVIOURAL: it
catches a wrong ANSWER, never a wrong INSTRUCTION. That is this backend's standing
certification gap and it stays stated on every claim built on top of this one.

THE FOUR LEGS:

1. ``divide`` — THE HEADLINE. ``a / b`` in a Metal kernel against
   ``numpy.float32`` division, over five operand classes, uint32 words. The
   ``pade_band`` class is the one that matters: both operands anchored at 1.0,
   which is exactly where the nonlinear quotient lives and exactly where a 2-ulp
   divide hides best. ``fast::divide`` and ``precise::divide`` are measured BESIDE
   it as named alternatives, so "the plain operator is the right spelling" is a
   measurement with a refuted twin rather than a preference.

2. ``negation`` — ``-x`` vs ``0.0f - x`` vs ``x * -1.0f``, RE-MEASURED here rather
   than inherited from ``shaders.py``'s 2026-08-15 census or from Triton's
   canonicalization workaround. The nonlinear body has no negation site, so this
   leg's job is to confirm the package's recorded fact still holds on this
   toolchain and to say plainly that the family does not depend on it.

3. ``signed_zero`` — the exhaustive special-value cross product through the
   operations this family actually performs: multiply, add, square and divide. A
   signed zero reaching the Pade numerator is not hypothetical: ``c2`` carries
   ``D_c`` linearly, and ``D_c`` is exactly ``+0.0`` on a metallic wall plane that
   ``zero_metal_D`` cleared.

4. ``host_scalar_rounding`` — NOT a device question, and load-bearing anyway. The
   plan wants to bind chi2/chi3 as pre-rounded float32 scalars. That is only legal
   if ``f32_array * python_float`` and ``f32_array * numpy.float32(python_float)``
   are the same words under this NumPy. NEP 50 says the Python float is weak and
   the loop runs in float32; this leg MEASURES it, because a remembered fact about
   a dependency is a hypothesis.

Artifacts: one JSON plus a JSONL row per case, written after every case (rule 7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

#: "This host cannot certify this claim" — distinct from 1, which means it ran and
#: the bytes differ. A CI that conflated them would read an unrunnable probe as a
#: passing one.
EXIT_CANNOT_CERTIFY = 75

#: How many lanes each value class fills. One dispatch per (variant, class).
LANES = 1 << 16


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_row(row: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Word comparison
# ---------------------------------------------------------------------------

def _words(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)


def differing(left: np.ndarray, right: np.ndarray) -> int:
    """uint32 word inequality, NEVER ``allclose``."""
    return int(np.count_nonzero(_words(left) != _words(right)))


def max_ulp(left: np.ndarray, right: np.ndarray) -> int:
    """Signed-magnitude ULP distance. NaN and inf lanes are excluded by the caller."""
    a = _words(left).view(np.int32).astype(np.int64)
    b = _words(right).view(np.int32).astype(np.int64)
    floor = np.int64(np.iinfo(np.int32).min)
    a = np.where(a < 0, floor - a, a)
    b = np.where(b < 0, floor - b, b)
    return int(np.max(np.abs(a - b))) if a.size else 0


def subnormal_mask(array: np.ndarray) -> np.ndarray:
    """Lanes whose word is a nonzero subnormal — the band Metal flushes."""
    w = _words(array)
    return ((w & 0x7F800000) == 0) & ((w & 0x007FFFFF) != 0)


def subnormal_count(array: np.ndarray) -> int:
    """Nonzero words whose exponent field is zero — the band Metal flushes."""
    return int(np.count_nonzero(subnormal_mask(array)))


# ---------------------------------------------------------------------------
# The shader sources
# ---------------------------------------------------------------------------

#: The shipped contraction directive, spelled the way ``shaders.py`` spells it. It
#: is passed in rather than hard-coded so the probe can measure BOTH modes and show
#: the verdict does not depend on the pragma.
_CONTRACT = "#pragma clang fp contract({mode})"

_BINARY_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void binary_op(
    device float*       out    [[buffer(0)]],
    device const float* a      [[buffer(1)]],
    device const float* b      [[buffer(2)]],
    constant uint&      n_elem [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
    float x = a[idx];
    float y = b[idx];
    out[idx] = __EXPR__;
}
"""

_UNARY_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void unary_op(
    device float*       out    [[buffer(0)]],
    device const float* a      [[buffer(1)]],
    constant uint&      n_elem [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
    float x = a[idx];
    out[idx] = __EXPR__;
}
"""

#: The whole Pade factor, as ``stepping.calc_nonlinear_u`` associates it
#: (stepping.py:1054-1056). Measured end to end rather than only per-operator,
#: because the quotient's operands are themselves rounded sums and a divide that
#: is correct on synthetic pairs can still meet operands the synthetic classes
#: never build.
_PADE_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void pade_u(
    device float*       out    [[buffer(0)]],
    device const float* gs     [[buffer(1)]],
    device const float* dsqr   [[buffer(2)]],
    device const float* us     [[buffer(3)]],
    device const float* chi2   [[buffer(4)]],
    device const float* chi3   [[buffer(5)]],
    constant uint&      n_elem [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
    float g = gs[idx];
    float d = dsqr[idx];
    float u = us[idx];
    float q2 = chi2[idx];
    float q3 = chi3[idx];
    float us_sq = u * u;
    float us_cu = (u * u) * u;
    float c2 = (g * q2) * us_sq;
    float c3 = (d * q3) * us_cu;
    float num = (1.0f + c2) + 2.0f * c3;
    float den = (1.0f + 2.0f * c2) + 3.0f * c3;
    out[idx] = __EXPR__;
}
"""

#: Every binary spelling measured, subject first. The plain operator is the SUBJECT;
#: the two namespaced ones are named alternatives, so a null result on them is
#: informative rather than a gap.
BINARY_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("divide_plain", "x / y"),
    ("divide_fast", "fast::divide(x, y)"),
    ("divide_precise", "precise::divide(x, y)"),
    ("multiply", "x * y"),
    ("add", "x + y"),
)

#: The three negation spellings the package has a recorded fact about.
UNARY_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("neg_unary", "-x"),
    ("neg_zero_minus", "0.0f - x"),
    ("neg_mul_minus_one", "x * -1.0f"),
    ("square", "x * x"),
)

#: The Pade quotient's three spellings.
PADE_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("pade_divide_plain", "num / den"),
    ("pade_divide_fast", "fast::divide(num, den)"),
    ("pade_divide_precise", "precise::divide(num, den)"),
)


def binary_source(expression: str, contract: str) -> str:
    return (_BINARY_TEMPLATE
            .replace("__CONTRACT__", _CONTRACT.format(mode=contract))
            .replace("__EXPR__", expression))


def unary_source(expression: str, contract: str) -> str:
    return (_UNARY_TEMPLATE
            .replace("__CONTRACT__", _CONTRACT.format(mode=contract))
            .replace("__EXPR__", expression))


def pade_source(expression: str, contract: str) -> str:
    return (_PADE_TEMPLATE
            .replace("__CONTRACT__", _CONTRACT.format(mode=contract))
            .replace("__EXPR__", expression))


# ---------------------------------------------------------------------------
# Operand classes
# ---------------------------------------------------------------------------

#: The special values the signed-zero leg crosses exhaustively. NaN is DELIBERATELY
#: absent from this list and measured in its own class: two implementations may
#: agree on "a NaN" while disagreeing on the payload bits, and a payload difference
#: reported as a divergence would be a false positive on the operator under test.
SPECIAL: Tuple[float, ...] = (
    0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
    float(np.float32(1.1754944e-38)),    # smallest normal
    float(-np.float32(1.1754944e-38)),
    float(np.float32(3.4028235e38)),     # largest finite
    float(-np.float32(3.4028235e38)),
    float(np.float32(1e-30)), float(np.float32(-1e-30)),
    float(np.float32(1e30)), float(np.float32(-1e30)),
)


def _pair_grid(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """The exhaustive cross product of ``values`` as two aligned vectors."""
    left = np.repeat(np.array(values, dtype=np.float32), len(values))
    right = np.tile(np.array(values, dtype=np.float32), len(values))
    return left, right


def binary_classes(seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The operand pairs each binary variant is measured on.

    ``pade_band`` IS THE CLASS THAT DECIDES THIS PROBE. Both operands are anchored
    at 1.0 with a perturbation in the ulp-visible range, which is the shape
    ``calc_nonlinear_u`` actually divides: a quotient near 1 has almost no exponent
    range to hide an error in, so a 2-ulp divide shows up here as a word difference
    and nowhere else in the classes below.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # The real shape: 1 + small, over the perturbation decades the Pade tree spans.
    scale = np.float32(10.0) ** rng.uniform(-7.0, -0.6, LANES).astype(np.float32)
    num = (np.float32(1.0) + scale * rng.standard_normal(LANES).astype(np.float32))
    scale2 = np.float32(10.0) ** rng.uniform(-7.0, -0.6, LANES).astype(np.float32)
    den = (np.float32(1.0) + scale2 * rng.standard_normal(LANES).astype(np.float32))
    out["pade_band"] = (num.astype(np.float32), den.astype(np.float32))

    # Wide dynamic range, both signs, no subnormals and no zeros.
    mag_a = np.float32(10.0) ** rng.uniform(-18.0, 18.0, LANES).astype(np.float32)
    mag_b = np.float32(10.0) ** rng.uniform(-18.0, 18.0, LANES).astype(np.float32)
    sign_a = np.where(rng.random(LANES) < 0.5, np.float32(-1.0), np.float32(1.0))
    sign_b = np.where(rng.random(LANES) < 0.5, np.float32(-1.0), np.float32(1.0))
    out["random_wide"] = ((mag_a * sign_a).astype(np.float32),
                          (mag_b * sign_b).astype(np.float32))

    # NEAR THE POLE: the denominator approaches zero, so u leaves the [0.998, 1.0]
    # band the settled chi3 science measured and the quotient's exponent moves.
    # This is the class that makes a divergence in a large-u regime byte-visible.
    small = (np.float32(10.0) ** rng.uniform(-8.0, -2.0, LANES).astype(np.float32))
    small = small * np.where(rng.random(LANES) < 0.5, np.float32(-1.0), np.float32(1.0))
    out["near_pole"] = ((np.float32(1.0)
                         + np.float32(0.1) * rng.standard_normal(LANES)
                         ).astype(np.float32), small.astype(np.float32))

    # Exact powers of two: division is EXACT here, so this class is a predicted
    # null. It is carried anyway, because a variant that diverged even here would
    # be doing something other than dividing.
    exponents = rng.integers(-60, 60, LANES)
    out["exact_powers"] = (
        np.ldexp(np.ones(LANES), rng.integers(-60, 60, LANES)).astype(np.float32),
        np.ldexp(np.ones(LANES), exponents).astype(np.float32))

    # The exhaustive special-value cross product, NaN excluded (see SPECIAL).
    out["signed_zero_exhaustive"] = _pair_grid(SPECIAL)
    return out


def unary_classes(seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    out: Dict[str, np.ndarray] = {}
    mag = np.float32(10.0) ** rng.uniform(-18.0, 18.0, LANES).astype(np.float32)
    sign = np.where(rng.random(LANES) < 0.5, np.float32(-1.0), np.float32(1.0))
    out["random_wide"] = (mag * sign).astype(np.float32)
    out["special_exhaustive"] = np.array(SPECIAL, dtype=np.float32)
    # The subnormal band, which this device FLUSHES and cannot be made not to. The
    # class is here so the flush is MEASURED on the negation spellings rather than
    # assumed to be uniform across them: a spelling that flushed differently would
    # be a different operation, which is the whole point of the leg.
    out["subnormal_band"] = np.ldexp(
        np.array(SPECIAL[:8], dtype=np.float64), -140).astype(np.float32)
    return out


def pade_operands(seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Operands for the end-to-end Pade leg, in the family's admitted domain.

    THE MAGNITUDE CLAUSE IS RESPECTED HERE. The settled science puts the crossing
    at ``chi3 * |chi1inv|^3 = 1e31`` and the family refuses above ``1e29``; these
    classes stay inside that, except ``near_clause_edge``, which sits just under it
    on purpose so the arithmetic at the admitted boundary is exercised rather than
    only the comfortable middle.
    """
    rng = np.random.default_rng(seed + 2)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    def pack(gs, dsqr, us, chi2, chi3):
        return {"gs": gs.astype(np.float32), "dsqr": dsqr.astype(np.float32),
                "us": us.astype(np.float32), "chi2": chi2.astype(np.float32),
                "chi3": chi3.astype(np.float32)}

    gs = rng.standard_normal(LANES).astype(np.float32)
    out["physical"] = pack(
        gs, (gs * gs) * np.float32(1.3), np.full(LANES, np.float32(0.4)),
        np.full(LANES, np.float32(1e-3)), np.full(LANES, np.float32(1e-2)))

    big = (rng.standard_normal(LANES) * 30.0).astype(np.float32)
    out["large_u"] = pack(
        big, big * big, np.full(LANES, np.float32(1.0)),
        np.full(LANES, np.float32(1e-2)), np.full(LANES, np.float32(3e-3)))

    # chi3 * |chi1inv|^3 = 1e29 exactly at the clause edge, with chi1inv = 1.
    out["near_clause_edge"] = pack(
        rng.standard_normal(LANES).astype(np.float32) * np.float32(1e-16),
        (rng.standard_normal(LANES).astype(np.float32) * np.float32(1e-16)) ** 2,
        np.full(LANES, np.float32(1.0)),
        np.full(LANES, np.float32(1e29)), np.full(LANES, np.float32(1e29)))

    # A wall plane: D is exactly +0.0 where zero_metal_D cleared it, so c2 is a
    # signed zero entering the numerator. Not hypothetical — this is what a
    # metallic run's face-0 lane holds.
    zeros = np.zeros(LANES, dtype=np.float32)
    zeros[1::2] = np.float32(-0.0)
    out["wall_zero"] = pack(
        zeros, np.abs(zeros), np.full(LANES, np.float32(0.7)),
        np.full(LANES, np.float32(1e-2)), np.full(LANES, np.float32(1e-2)))
    return out


# ---------------------------------------------------------------------------
# The NumPy references — the array path's own arithmetic, transcribed
# ---------------------------------------------------------------------------

def reference_binary(name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore",
                     under="ignore"):
        if name.startswith("divide"):
            return (x / y).astype(np.float32)
        if name == "multiply":
            return (x * y).astype(np.float32)
        if name == "add":
            return (x + y).astype(np.float32)
    raise ValueError(name)


def reference_unary(name: str, x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if name.startswith("neg"):
            return (-x).astype(np.float32)
        if name == "square":
            return (x * x).astype(np.float32)
    raise ValueError(name)


def reference_pade(operands: Dict[str, np.ndarray]) -> np.ndarray:
    """``stepping.calc_nonlinear_u`` (:1025-1027) verbatim, on float32 arrays.

    Transcribed rather than re-derived: ``c2 = di * chi2 * (chi1inv * chi1inv)``,
    ``c3 = dsqr * chi3 * (chi1inv * chi1inv * chi1inv)``, then
    ``(1 + c2 + 2*c3) / (1 + 2*c2 + 3*c3)``. Python's left-to-right association is
    what makes ``a * b * c`` mean ``(a*b)*c``, which is why the kernel spells the
    powers that way too.
    """
    di = operands["gs"]
    dsqr = operands["dsqr"]
    chi1inv = operands["us"]
    chi2 = operands["chi2"]
    chi3 = operands["chi3"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore",
                     under="ignore"):
        c2 = di * chi2 * (chi1inv * chi1inv)
        c3 = dsqr * chi3 * (chi1inv * chi1inv * chi1inv)
        return ((1 + c2 + 2 * c3) / (1 + 2 * c2 + 3 * c3)).astype(np.float32)


# ---------------------------------------------------------------------------
# The device runner
# ---------------------------------------------------------------------------

class Runner:
    """Compiles on demand, launches against fresh device copies, counts both."""

    def __init__(self) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
        self._cache: Dict[str, Any] = {}
        self.compiles = 0
        self.launches = 0

    def library(self, source: str) -> Any:
        library = self._cache.get(source)
        if library is None:
            library = self.torch.mps.compile_shader(source)
            self._cache[source] = library
            self.compiles += 1
        return library

    def launch(self, source: str, entry: str, inputs: Sequence[np.ndarray]
               ) -> Tuple[np.ndarray, int]:
        """One launch on FRESH copies. Returns ``(out, moved)``.

        The output buffer is pre-filled with a SENTINEL, never zeros: a kernel that
        wrote nothing would otherwise agree with any reference that also produced
        zeros, and the identity would be vacuous. ``moved`` is the vacuity floor.
        """
        torch = self.torch
        library = self.library(source)
        lanes = int(inputs[0].size)
        sentinel = np.full(lanes, np.float32(-7.5), dtype=np.float32)
        d_out = torch.from_numpy(sentinel.copy()).to("mps")
        device_inputs = [torch.from_numpy(np.ascontiguousarray(
            array, dtype=np.float32).reshape(-1).copy()).to("mps")
            for array in inputs]
        getattr(library, entry)(d_out, *device_inputs, lanes)
        torch.mps.synchronize()
        self.launches += 1
        out = d_out.cpu().numpy()
        moved = int(np.count_nonzero(out.view(np.uint32) != sentinel.view(np.uint32)))
        return out, moved


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def classify(device: np.ndarray, expected: np.ndarray) -> Dict[str, Any]:
    """Split a comparison into FLUSH lanes and everything else.

    THIS SPLIT IS THE WHOLE HONESTY OF THIS PROBE, and the first run of it was
    wrong without it. ``divide_plain`` reported 16 differing words on the special
    table and that read as a divide defect; every one of the sixteen turned out to
    be a lane whose IEEE result is SUBNORMAL, which this device flushes to a
    correctly-signed zero and cannot be made not to. ``multiply`` — an operation
    four certified families already ship — showed the identical signature on 8
    lanes, which is what proves the effect belongs to the platform's number system
    and not to the operator under test.

    So a row reports three counts, and the SUBJECT assertion reads only the third:

    * ``flushed`` — lanes where the reference result is a nonzero subnormal and the
      device produced the same-signed zero. Expected, unavoidable, and governed by
      the family's CHECKED subnormal-free precondition rather than by a tolerance;
    * ``nonfinite`` — lanes where either side is inf or NaN, excluded because a NaN
      payload difference is not a statement about the operator;
    * ``differing_normal`` — everything else. THIS is the number that decides
      whether the operator is IEEE round-to-nearest on this host.

    A lane whose reference is subnormal and whose device value is NOT the matching
    signed zero counts as ``differing_normal``: that would be a real defect wearing
    the flush's clothes, and it must not be absorbed by the exemption.
    """
    finite = _finite_mask(device, expected)
    sub_ref = subnormal_mask(expected)
    signed_zero = (_words(device) == (_words(expected) & np.uint32(0x80000000)))
    flushed = finite & sub_ref & signed_zero
    exempt = flushed | (~finite)
    words_differ = _words(device) != _words(expected)
    normal = words_differ & ~exempt
    return {
        "differing": int(np.count_nonzero(words_differ)),
        "differing_normal": int(np.count_nonzero(normal)),
        "flushed_to_signed_zero": int(np.count_nonzero(flushed)),
        "subnormal_reference_lanes": int(np.count_nonzero(sub_ref)),
        "nonfinite_lanes": int(np.count_nonzero(~finite)),
        "max_ulp_normal": max_ulp(device[finite & ~sub_ref],
                                  expected[finite & ~sub_ref]),
    }


def _record(payload: Dict[str, Any], out: str, jsonl: str, leg: str,
            row: Dict[str, Any]) -> None:
    payload["legs"].setdefault(leg, []).append(row)
    append_row(dict(row, leg=leg), jsonl)
    save(payload, out)


def leg_divide(runner: Runner, payload: Dict[str, Any], out: str, jsonl: str,
               seed: int, failures: List[str]) -> None:
    classes = binary_classes(seed)
    fast_divide_caught = 0
    for contract in ("off", "fast"):
        for name, expression in BINARY_VARIANTS:
            source = binary_source(expression, contract)
            for class_name, (x, y) in classes.items():
                device, moved = runner.launch(source, "binary_op", (x, y))
                expected = reference_binary(name, x, y)
                verdict = classify(device, expected)
                row = {
                    "variant": name, "expression": expression,
                    "contract": contract, "value_class": class_name,
                    "lanes": int(x.size), "moved_words": moved,
                    "subnormal_in": subnormal_count(x) + subnormal_count(y),
                }
                row.update(verdict)
                assert moved == x.size, (
                    f"VACUOUS: {name}/{class_name} moved {moved} of {x.size} "
                    f"words; a no-op agreeing with a no-op certifies nothing")
                # THE SUBJECT is the plain operator, and only its NORMAL-result
                # lanes are asserted: the flush is a platform fact this family
                # handles with a checked precondition, not with a tolerance.
                if name == "divide_plain" and verdict["differing_normal"]:
                    failures.append(
                        f"divide_plain/{class_name}/contract={contract}: "
                        f"{verdict['differing_normal']} differing normal-result "
                        f"words, max {verdict['max_ulp_normal']} ulp")
                if name == "divide_fast":
                    fast_divide_caught += verdict["differing_normal"]
                _record(payload, out, jsonl, "divide", row)
                log(f"[divide] {name:<16} {contract:<4} {class_name:<24} "
                    f"lanes={x.size:<6} diff={verdict['differing']:<6} "
                    f"normal={verdict['differing_normal']:<6} "
                    f"flushed={verdict['flushed_to_signed_zero']:<4} "
                    f"ulp={verdict['max_ulp_normal']}")

    # THE REFUTED SPELLING MUST ACTUALLY BE REFUTED. `fast::divide` is going to be
    # this family's must-catch source mutation, and a mutation that turns out to be
    # equivalent catches nothing — so the probe asserts here that the spelling
    # genuinely diverges rather than leaving the gate to discover it does not.
    assert fast_divide_caught > 0, (
        "fast::divide did not diverge from the plain operator anywhere: the "
        "refuted spelling is then not refuted, and a gate mutation planting it "
        "would certify nothing")
    payload["fast_divide_divergent_words"] = fast_divide_caught


def leg_negation(runner: Runner, payload: Dict[str, Any], out: str, jsonl: str,
                 seed: int, failures: List[str]) -> None:
    classes = unary_classes(seed)
    for contract in ("off", "fast"):
        for name, expression in UNARY_VARIANTS:
            source = unary_source(expression, contract)
            for class_name, x in classes.items():
                device, moved = runner.launch(source, "unary_op", (x,))
                expected = reference_unary(name, x)
                verdict = classify(device, expected)
                row = {
                    "variant": name, "expression": expression,
                    "contract": contract, "value_class": class_name,
                    "lanes": int(x.size), "moved_words": moved,
                    "subnormal_in": subnormal_count(x),
                }
                row.update(verdict)
                assert moved == x.size, (
                    f"VACUOUS: {name}/{class_name} moved {moved} of {x.size}")
                # `-x` IS THE PACKAGE'S RECORDED SPELLING and this leg re-measures
                # it rather than inheriting it. It must be exact on EVERY class
                # including the subnormal band, because a sign-bit flip is not an
                # arithmetic operation and has no band to flush.
                if name == "neg_unary" and verdict["differing"]:
                    failures.append(
                        f"neg_unary/{class_name}/contract={contract}: "
                        f"{verdict['differing']} differing words — the package's "
                        f"recorded `-x` fact does not hold on this toolchain")
                _record(payload, out, jsonl, "negation", row)
                log(f"[negation] {name:<20} {contract:<4} {class_name:<20} "
                    f"lanes={x.size:<6} diff={verdict['differing']:<6} "
                    f"normal={verdict['differing_normal']}")


def leg_pade(runner: Runner, payload: Dict[str, Any], out: str, jsonl: str,
             seed: int, failures: List[str]) -> None:
    classes = pade_operands(seed)
    order = ("gs", "dsqr", "us", "chi2", "chi3")
    contraction_divergence = 0
    for contract in ("off", "fast"):
        for name, expression in PADE_VARIANTS:
            source = pade_source(expression, contract)
            for class_name, operands in classes.items():
                inputs = tuple(operands[key] for key in order)
                device, moved = runner.launch(source, "pade_u", inputs)
                expected = reference_pade(operands)
                finite = _finite_mask(device, expected)
                verdict = classify(device, expected)
                row = {
                    "variant": name, "expression": expression,
                    "contract": contract, "value_class": class_name,
                    "lanes": int(inputs[0].size), "moved_words": moved,
                    "u_min": float(np.nanmin(expected[finite])) if finite.any() else None,
                    "u_max": float(np.nanmax(expected[finite])) if finite.any() else None,
                }
                row.update(verdict)
                assert moved == inputs[0].size, (
                    f"VACUOUS: {name}/{class_name} moved {moved}")
                # THE SHIPPED CONFIGURATION IS (plain divide, contract off). That
                # is the only combination asserted. contract=fast is measured as a
                # POSITIVE control on the pragma's necessity, not as a failure.
                if (name == "pade_divide_plain" and contract == "off"
                        and verdict["differing_normal"]):
                    failures.append(
                        f"pade_divide_plain/{class_name}/contract=off: "
                        f"{verdict['differing_normal']} differing normal-result "
                        f"words, max {verdict['max_ulp_normal']} ulp")
                if name == "pade_divide_plain" and contract == "fast":
                    contraction_divergence += verdict["differing_normal"]
                _record(payload, out, jsonl, "pade", row)
                log(f"[pade] {name:<22} {contract:<4} {class_name:<18} "
                    f"diff={verdict['differing']:<6} "
                    f"normal={verdict['differing_normal']:<6} "
                    f"ulp={verdict['max_ulp_normal']} "
                    f"u=[{row['u_min']}, {row['u_max']}]")

    # THE PRAGMA MUST BE LOAD-BEARING, MEASURED. `shaders.py` calls
    # `#pragma clang fp contract(off)` mandatory; on THIS expression that claim is
    # only worth something if turning it off actually changes the bytes. If the
    # two modes agreed everywhere, the pragma would be cargo on this family and
    # the docstring would be overstating its own guard.
    assert contraction_divergence > 0, (
        "contract(fast) produced identical bytes to contract(off) on the whole "
        "Pade expression: the mandatory-pragma claim is then untested here")
    payload["contraction_divergent_words"] = contraction_divergence


def leg_host_scalar_rounding(payload: Dict[str, Any], out: str, jsonl: str,
                             seed: int, failures: List[str]) -> None:
    """Is ``f32_array * python_float`` the same words as ``* numpy.float32(x)``?

    The plan wants to bind chi2/chi3 as pre-rounded float32 scalars, which is legal
    only if the array path's own Python-float multiply rounds the scalar ONCE to
    float32 and then multiplies in float32. Under NEP 50 the Python float is weak
    and the loop runs at float32 — but that is a remembered fact about a dependency
    until it is measured, and the Triton track's docstring point 3 records the
    OPPOSITE behaviour for a Python-float ``us`` in a DOUBLE power expression, so
    the two cases must not be conflated.
    """
    rng = np.random.default_rng(seed + 7)
    array = (rng.standard_normal(LANES) * 1e3).astype(np.float32)
    scalars = [1e-3, 0.1, 1.0 / 3.0, 1e29, 1e-30, 3.4028235e38, 7.7e-13]
    for scalar in scalars:
        weak = (array * scalar).astype(np.float32)
        prerounded = (array * np.float32(scalar)).astype(np.float32)
        diff = differing(weak, prerounded)
        row = {"scalar": repr(scalar), "lanes": int(array.size),
               "differing": diff,
               "float32_of_scalar": float(np.float32(scalar))}
        if diff:
            failures.append(
                f"host_scalar_rounding/{scalar!r}: {diff} differing words — a "
                f"pre-rounded float32 scalar is NOT the array path's arithmetic, "
                f"so chi2/chi3 must be bound as volumes rather than scalars")
        _record(payload, out, jsonl, "host_scalar_rounding", row)
        log(f"[host] scalar={scalar!r:<22} differing={diff}")

    # The counter-case the Triton track measured: a DOUBLE power of a Python-float
    # epsilon rounds once when it meets the array, and rebuilding it in float32 is
    # a different word. Recorded so the two cases are visibly distinguished rather
    # than the first result being over-generalised.
    us = 1.0 / 3.0
    double_power = (array * (us * us)).astype(np.float32)
    f32_power = (array * (np.float32(us) * np.float32(us))).astype(np.float32)
    diff = differing(double_power, f32_power)
    row = {"scalar": "counter_case: (us*us) double vs float32 power",
           "lanes": int(array.size), "differing": diff,
           "note": "NOT a failure: it is why this family requires a VOLUME "
                   "inverse epsilon, so the powers are formed in float32 on both "
                   "sides"}
    _record(payload, out, jsonl, "host_scalar_rounding", row)
    log(f"[host] counter-case double-vs-f32 power differing={diff}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def environment_stamp() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        from meep_gpu.metal_kernels.device import metal_frontend_version  # noqa: PLC0415

        record["metal_frontend"] = metal_frontend_version()
    except Exception as exc:  # noqa: BLE001
        record["metal_frontend"] = None
        record["metal_frontend_error"] = repr(exc)
    try:
        import torch  # noqa: PLC0415

        record["torch"] = str(torch.__version__)
        record["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        record["torch"] = None
        record["mps_available"] = False
        record["torch_error"] = repr(exc)
    return record


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default="")
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(list(argv))

    started = time.time()
    out = os.path.abspath(args.out)
    jsonl = os.path.splitext(out)[0] + ".jsonl"
    if os.path.exists(jsonl):
        os.remove(jsonl)
    wanted = tuple(name.strip() for name in args.legs.split(",") if name.strip())

    with open(os.path.abspath(__file__), "rb") as handle:
        self_sha = hashlib.sha256(handle.read()).hexdigest()

    payload: Dict[str, Any] = {
        "probe": "metal_divide_and_negation",
        "question": ("is Metal's plain `/` IEEE round-to-nearest float32 on this "
                     "host, and do this package's recorded negation and "
                     "signed-zero facts still hold on this toolchain?"),
        "evidence_class": (
            "BEHAVIOURAL ONLY. torch.mps.compile_shader exposes no AIR, no GPU ISA "
            "and no optimisation report, so this probe cannot read the generated "
            "code the way the CUDA sibling read its PTX. It certifies that the "
            "ANSWERS agree, never that the INSTRUCTIONS do."),
        "probe_sha256": self_sha,
        "seed": args.seed,
        "lanes_per_class": LANES,
        "environment": environment_stamp(),
        "legs": {},
    }
    save(payload, out)

    if not payload["environment"].get("mps_available"):
        payload["summary"] = {"status": "cannot-certify-here",
                              "reasons": ["no MPS device available"]}
        save(payload, out)
        log("[cannot-certify] no MPS device available")
        return EXIT_CANNOT_CERTIFY

    runner = Runner()
    failures: List[str] = []
    legs = (
        ("divide", lambda: leg_divide(runner, payload, out, jsonl, args.seed, failures)),
        ("negation", lambda: leg_negation(runner, payload, out, jsonl, args.seed, failures)),
        ("pade", lambda: leg_pade(runner, payload, out, jsonl, args.seed, failures)),
        ("host_scalar_rounding",
         lambda: leg_host_scalar_rounding(payload, out, jsonl, args.seed, failures)),
    )
    ran: List[str] = []
    for name, leg in legs:
        if wanted and name not in wanted:
            log(f"[skip] leg {name}")
            continue
        log(f"=== LEG {name} ===")
        leg()
        ran.append(name)

    compared = sum(len(rows) for rows in payload["legs"].values())
    payload["summary"] = {
        "status": "passed" if (not failures and compared) else "FAILED",
        "legs_run": ran,
        "cases": compared,
        "failures": failures,
        "compiles": runner.compiles,
        "launches": runner.launches,
        "elapsed_s": round(time.time() - started, 1),
    }
    save(payload, out)
    log(f"{'PASSED' if not failures and compared else 'FAILED'}: {compared} cases, "
        f"{len(failures)} failures, compiles={runner.compiles} "
        f"launches={runner.launches} -> {out}")
    return 0 if (not failures and compared) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
