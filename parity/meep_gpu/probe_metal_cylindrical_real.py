"""What the Metal cylindrical m = 0 family must MEASURE before it can be written.

Five questions, each of which decides a design choice rather than decorating one.
None of them is answered by another backend's measurement, and the module says so
where the temptation to inherit is strongest.

1. **THE DIVIDE** (:func:`leg_divide`). ``cylindrical_rderiv_prefix``
   (stepping.py:1286-1334) divides once per element: ``(weighted[i] -
   weighted[i-1]) / divisor[i-1]``. Whether Metal's ``/`` is the correctly-rounded
   float32 divide NumPy performs is a PLATFORM FACT and is unmeasured on this
   backend; ``metal_kernels/shaders.py:41-46`` records only that ``fast::divide``
   diverges. If ``/`` diverges too, the prefix cannot run on the device at all and
   the family must pay a host round trip per curl sub-step.

2. **THE SCAN** (:func:`leg_prefix`). The Triton m = 0 family put the scan on the
   ARRAY PATH and called it the round's decisive measurement
   (``triton_kernels/cylindrical_triton.py:32-56``): ``cupy.cumsum`` is not a
   sequential float32 accumulation (87,270 of 102,400 elements differ), so on that
   backend the ORACLE ITSELF was unreproducible and no device scan could match it.
   **THAT REASON DOES NOT TRANSFER AND ITS PREMISE IS INVERTED HERE.** This
   backend's engine holds NumPy (``metal_kernels/coverage.py:82``), and the same
   Triton paragraph records that ``numpy.cumsum`` IS byte-equal to a sequential
   accumulation. So the oracle a Metal kernel must reproduce is the sequential one,
   which a column-serial scan can reproduce by construction — and
   :data:`templates.COLUMN_SERIAL_SCAN` exists for exactly this. Whether it DOES is
   measured here, end to end against ``stepping.cylindrical_rderiv_prefix`` itself,
   at both ``ir0`` values the two sub-steps use and with the B side's zero wall row.

3. **NEGATION AND SIGNED ZERO** (:func:`leg_negation`). Re-measured on this
   backend rather than inherited. ``shaders.py:47-51`` records ``-x`` as a sign-bit
   operation here while Triton lowers it as ``0.0 - x``; this family's masks and its
   ``Dy[0] = 0`` rule put signed zeros live on the axis row, so the three spellings
   are measured again on THIS host, in THIS round, over an exhaustive table.

4. **THE r-AXIS NEAR GHOST** (:func:`leg_axis_ghost`). ``_shift_down``'s CYL_AXIS
   branch (stepping.py:1877-1889) writes a sign-flipped image of stored row 0. The
   Triton COMPLEX family measured a METALLIC substitution identical over 80/80 rows;
   that measurement is |m| >= 1 and complex, and this family is m = 0 and real. It
   is re-measured here on the ARRAY PATH, where the question actually lives — no
   Metal involved — by stepping two engines that differ only in that one ghost rule.

5. **THE OWNERSHIP MASK AND GHOST EMITTERS** (:func:`leg_emitters`). The claim the
   whole design rests on is that compiling the r axis as METALLIC reproduces BOTH
   the CYL_AXIS far ghost (identical by inspection, stepping.py:1828-1830) and the
   is_axis ownership mask (stepping.py:1945-1949) with NO new emitter. That is an
   assertion about ``shaders.ownership_mask``'s output, and it is checked against
   ``_mask_non_owned_cells``'s own behaviour on a real cylindrical grid rather than
   against a reading of it.

VACUITY. Every leg asserts it MOVED WORDS against a sentinel fill, and every leg
censuses subnormals in its inputs and outputs — Metal flushes them and cannot be
made not to, so a case that reaches the band is measuring the flush rather than the
question, and is reported as a WINDOW rather than hidden.

EVIDENCE CLASS, STATED. ``torch.mps.compile_shader`` exposes no AIR, no GPU ISA and
no optimisation report (``metal_kernels/device.py:46-62``), so every Metal leg here
certifies that the ANSWERS agree and never that the INSTRUCTIONS do. It catches a
wrong answer, not a wrong instruction. That is this backend's standing gap against
the Triton and CUDA tracks and it stays stated on every claim built on this probe.

Run::

    python -u probe_metal_cylindrical_real.py \
        --out results/metal_cylindrical_real_<date>/probe.json
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

from parity.meep_gpu import metal_gate_kit as kit  # noqa: E402

SEED = 20260816


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _torch():
    import torch  # noqa: PLC0415

    return torch


def _subnormal_words(array: Any) -> int:
    """How many float32 words sit in the subnormal band (exponent 0, mantissa != 0).

    Counted as WORDS rather than by ``numpy.isfinite`` tests, because the band is a
    bit pattern: exponent field zero with a nonzero mantissa. A case whose inputs or
    outputs reach the band is measuring Metal's flush rather than the question the
    leg asks, so every leg reports this beside its verdict.
    """
    word = kit.words(array)
    return int(np.count_nonzero(((word & 0x7F800000) == 0) & ((word & 0x007FFFFF) != 0)))


def classify(measured: Any, reference: Any) -> Dict[str, int]:
    """Split disagreeing words into the FLUSH class and everything else.

    THE SPLIT IS NOT A LOOSENING, IT IS THE ONLY WAY TO ASK THE QUESTION. Metal
    flushes float32 subnormals and exposes no lever to stop it — no pragma, no
    environment variable, no torch API — so a comparison that lumps the flush in
    with rounding cannot say whether an operation is correctly rounded; it can only
    say that the platform flushes, which is already known. A word is counted as a
    FLUSH only under all three conditions together: the reference is subnormal, the
    device produced a zero, and the two carry the SAME SIGN. Anything else is
    ``other`` and is what a leg asserts on.

    The signed-zero condition is load-bearing rather than tidy: a kernel that
    dropped a negation would also produce a zero where the reference is subnormal,
    and without the sign check that defect would be absorbed into the flush class
    and reported as a platform fact.
    """
    got = kit.words(measured)
    want = kit.words(reference)
    disagree = got != want
    reference_subnormal = ((want & 0x7F800000) == 0) & ((want & 0x007FFFFF) != 0)
    device_zero = (got & 0x7FFFFFFF) == 0
    same_sign = (got & 0x80000000) == (want & 0x80000000)
    flush = disagree & reference_subnormal & device_zero & same_sign
    return {"differing": int(np.count_nonzero(disagree)),
            "flushed": int(np.count_nonzero(flush)),
            "other": int(np.count_nonzero(disagree & ~flush))}


def _upload(array: Any) -> Any:
    torch = _torch()
    flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    return torch.from_numpy(flat).to(torch.device("mps"))


def _download(tensor: Any, shape: Tuple[int, ...]) -> Any:
    torch = _torch()
    torch.mps.synchronize()
    return tensor.cpu().numpy().reshape(shape)


def _sentinel(shape: Tuple[int, ...]) -> Any:
    """A fill no correct kernel can leave behind — the vacuity floor's needle.

    Zero-init would make a kernel that never ran compare equal to one that wrote
    zeros, which is the class of hollow pass this project refuses everywhere else.
    """
    return np.full(shape, np.float32(-7.5), dtype=np.float32)


def _compile(source: str, entry: str) -> Any:
    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    return getattr(compile_source(source), entry)


# ---------------------------------------------------------------------------
# Leg 1 — the divide
# ---------------------------------------------------------------------------

_DIVIDE_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(__MODE__)

kernel void divide_probe(
    device float*       out   [[buffer(0)]],
    device const float* a     [[buffer(1)]],
    device const float* b     [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) { return; }
    out[idx] = __EXPR__;
}
"""


def _divide_operands(rng: Any) -> Dict[str, Tuple[Any, Any]]:
    """Numerator/divisor pairs, in the classes the prefix actually produces.

    ``rderiv`` IS THE REAL ONE: the divisors are the float32-rounded
    ``(ir + ir0) - 0.5`` row constants ``cylindrical_rderiv_prefix`` builds
    (stepping.py:1310-1312) and the numerators are differences of weighted field
    rows. The other three classes exist so a divergence can be ATTRIBUTED — a
    correctly-rounded divide on random data that fails on the ladder would be a
    range fact, not a rounding one.
    """
    out: Dict[str, Tuple[Any, Any]] = {}

    # The real thing: divisors from both ir0 values, numerators from field-scale
    # differences.
    rows = 1024
    numerators: List[Any] = []
    divisors: List[Any] = []
    for ir0 in (0.0, 0.5):
        counts = np.arange(rows, dtype=np.float64) + ir0
        divisor = (counts[1:] - 0.5).astype(np.float32)
        numerator = rng.uniform(-1.0, 1.0, size=divisor.shape).astype(np.float32)
        numerators.append(numerator)
        divisors.append(divisor)
    out["rderiv"] = (np.concatenate(numerators), np.concatenate(divisors))

    # Uniform random over a wide exponent range.
    size = 65536
    mantissa = rng.uniform(1.0, 2.0, size=size).astype(np.float32)
    exponent = rng.integers(-40, 40, size=size)
    wide = (mantissa * np.float32(2.0) ** exponent.astype(np.float32)).astype(np.float32)
    signs = np.where(rng.integers(0, 2, size=size) == 0, np.float32(-1.0),
                     np.float32(1.0))
    out["wide_dynamic"] = ((wide * signs).astype(np.float32),
                           (wide[::-1] * signs[::-1]).astype(np.float32))

    out["random_band"] = (rng.uniform(-1.0, 1.0, size=size).astype(np.float32),
                          rng.uniform(0.5, 2.0, size=size).astype(np.float32))

    # Signed zeros and ones, exhaustively paired. +-0 / x and x / +-inf are exactly
    # the patterns an ownership-masked row produces, and the sign of the quotient's
    # zero is a word this project compares.
    edges = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 3.0, -3.0,
                      np.float32(np.finfo(np.float32).max),
                      np.float32(np.finfo(np.float32).tiny)], dtype=np.float32)
    left = np.repeat(edges, edges.size)
    right = np.tile(edges, edges.size)
    out["signed_edges"] = (left.astype(np.float32), right.astype(np.float32))
    return out


def leg_divide(payload: Dict[str, Any], out: str) -> None:
    """Is Metal's float32 ``/`` the divide NumPy performs? ``fast::divide`` as control."""
    rng = np.random.default_rng(SEED + 1)
    rows: List[Dict[str, Any]] = []
    operands = _divide_operands(rng)

    arms = (("plain", "a[idx] / b[idx]", True),
            ("fast_divide", "fast::divide(a[idx], b[idx])", None))
    for mode in ("off", "fast"):
        for arm, expr, must_match in arms:
            entry = _compile(
                _DIVIDE_SOURCE.replace("__MODE__", mode).replace("__EXPR__", expr),
                "divide_probe")
            for label, (left, right) in operands.items():
                with np.errstate(divide="ignore", invalid="ignore"):
                    reference = (left / right).astype(np.float32)
                got = _sentinel(left.shape)
                device = _upload(got)
                entry(device, _upload(left), _upload(right), int(left.size))
                measured = _download(device, left.shape)
                moved = kit.differing(measured, got)
                kit.assert_moved(moved, f"divide/{mode}/{arm}/{label}",
                                 floor=int(left.size))
                verdict = classify(measured, reference)
                row = {"contract": mode, "arm": arm, "class": label,
                       "words": int(left.size), "moved": moved,
                       "subnormal_in": _subnormal_words(left) + _subnormal_words(right),
                       "subnormal_reference": _subnormal_words(reference)}
                row.update(verdict)
                rows.append(row)
                kit.log(f"  divide {mode:<4} {arm:<12} {label:<14} "
                        f"differing={verdict['differing']}/{left.size} "
                        f"(flushed {verdict['flushed']}, other {verdict['other']})")
                if must_match:
                    assert verdict["other"] == 0, (
                        f"Metal '/' diverges from numpy float32 divide on {label} "
                        f"beyond the subnormal flush ({verdict['other']} words of "
                        f"{left.size}) under contract({mode}): the radial prefix "
                        f"CANNOT run on the device")
                payload["legs"]["divide"] = rows
                kit.save(payload, out)

    plain = [r for r in rows if r["arm"] == "plain"]
    fast = [r for r in rows if r["arm"] == "fast_divide"]
    payload["legs"]["divide"] = rows
    payload["verdicts"]["divide"] = {
        "plain_words": sum(r["words"] for r in plain),
        "plain_differing": sum(r["differing"] for r in plain),
        "plain_flushed": sum(r["flushed"] for r in plain),
        "plain_other": sum(r["other"] for r in plain),
        "fast_divide_other": sum(r["other"] for r in fast),
        "fast_divide_differing": sum(r["differing"] for r in fast),
        # The claim is NOT "the divide never disagrees" — it is "every disagreement
        # is the platform's subnormal flush, which every claim on this backend
        # already rides on a checked precondition against".
        "verdict": ("IDENTICAL-MODULO-FLUSH"
                    if sum(r["other"] for r in plain) == 0 else "DIVERGENT"),
    }
    kit.save(payload, out)


# ---------------------------------------------------------------------------
# Leg 2 — the scan
# ---------------------------------------------------------------------------

#: The candidate device prefix, built from :data:`templates.COLUMN_SERIAL_SCAN`'s
#: shape: one thread per (phi, z) column, summed STRICTLY SERIALLY from r = 0 up.
#: ``__WALL__`` is the B side's zero wall row (stepping.py:359-361) as a SOURCE
#: specialisation rather than a runtime branch, so no branch survives around a float
#: expression — the same discipline ``shaders.ghost`` follows.
_PREFIX_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(__MODE__)

kernel void cyl_rderiv_prefix(
    device float*       out     [[buffer(0)]],
    device const float* src     [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* divisor [[buffer(3)]],
    constant uint&      nx      [[buffer(4)]],
    constant uint&      ny      [[buffer(5)]],
    constant uint&      nz      [[buffer(6)]],
    constant uint&      n_cols  [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_cols) { return; }
    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int k = int(idx) % nzi;
    int j = int(idx) / nzi;
    if (j >= nyi) { return; }
    int nyz = nyi * nzi;
    int base = j * nzi + k;

    // increment[0] is an exact +0.0 (xp.zeros_like / increment[face] = 0), and the
    // cumsum's first output IS that zero. Written, not assumed: -0.0 here would be
    // a different word.
    float acc = 0.0f;
    out[base] = acc;
    float prev = __SRC0__ * weights[0];
    for (int i = 1; i < nxi; ++i) {
        int o = i * nyz + base;
        float w = __SRCI__ * weights[i];
        // ONE divide per element, matching stepping.py:1286 exactly: the difference
        // of two already-weighted rows over the row-constant divisor. Not
        // `* (1/divisor)` -- that is a different float32 number.
        float inc = (w - prev) / divisor[i - 1];
        // STRICTLY SERIAL: numpy.cumsum is out[i] = out[i-1] + in[i], and any
        // blocked or parallel scan reassociates a sum float32 addition does not
        // associate.
        acc = acc + inc;
        out[o] = acc;
        prev = w;
    }
}
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 1286->1315


def _prefix_source(mode: str, wall: bool) -> str:
    """The scan source for one contraction mode and one wall-row arm."""
    if wall:
        # The B side scans nr + 1 rows over an nr-row source: the extra top row is
        # the zero wall (stepping.py:360). Spelled as the literal 0.0f the array
        # path assigns, not as a shortened loop.
        src0 = "(0 < nxi - 1 ? src[base] : 0.0f)"
        srci = "(i < nxi - 1 ? src[i * nyz + base] : 0.0f)"
    else:
        src0 = "src[base]"
        srci = "src[i * nyz + base]"
    return (_PREFIX_SOURCE.replace("__MODE__", mode)
            .replace("__SRC0__", src0).replace("__SRCI__", srci))


def _host_prefix(source: Any, ir0: float, wall: bool) -> Tuple[Any, Any, Any, Any]:
    """The array path's own answer, plus the two row vectors the kernel binds.

    ``stepping.cylindrical_rderiv_prefix`` is CALLED, never re-derived: it is the
    oracle, and a transcription of it here would pin this probe against a second
    reading of the same lines rather than against the function the engine runs.
    """
    from meep_gpu import stepping  # noqa: PLC0415

    scanned = source
    if wall:
        rows = source.shape[0]
        scanned = np.zeros((rows + 1,) + source.shape[1:], dtype=source.dtype)
        scanned[0:rows] = source
        scanned[rows] = 0
    reference = stepping.cylindrical_rderiv_prefix(np, scanned, ir0)
    counts = np.arange(scanned.shape[0], dtype=np.float64) + ir0
    weights = counts.astype(np.float32)
    divisor = (counts[1:] - 0.5).astype(np.float32)
    return reference, scanned, weights, divisor


def leg_prefix(payload: Dict[str, Any], out: str) -> None:
    """A column-serial device scan against ``stepping.cylindrical_rderiv_prefix``."""
    rng = np.random.default_rng(SEED + 2)
    rows: List[Dict[str, Any]] = []
    shapes = ((20, 1, 40), (13, 1, 9), (320, 1, 64), (7, 1, 3))
    for mode in ("off", "fast"):
        for wall in (False, True):
            entry = _compile(_prefix_source(mode, wall), "cyl_rderiv_prefix")
            for shape in shapes:
                for ir0, side in ((0.0, "step_B"), (0.5, "step_D")):
                    for value_class in ("random_band", "wide_dynamic", "cancelling"):
                        rows.append(_one_prefix_case(
                            entry, rng, shape, ir0, side, wall, mode, value_class))
                        row = rows[-1]
                        kit.log(f"  prefix {mode:<4} wall={int(wall)} "
                                f"{str(shape):<12} ir0={ir0} {value_class:<12} "
                                f"differing={row['differing']}/{row['words']} "
                                f"(flushed {row['flushed']}, other {row['other']})")
                        if mode == "off":
                            assert row["other"] == 0, (
                                f"the column-serial device scan diverges from "
                                f"stepping.cylindrical_rderiv_prefix beyond the "
                                f"subnormal flush: {row}")
                        payload["legs"]["prefix"] = rows
                        kit.save(payload, out)

    shipped = [r for r in rows if r["contract"] == "off"]
    fast = [r for r in rows if r["contract"] == "fast"]
    payload["legs"]["prefix"] = rows
    payload["verdicts"]["prefix"] = {
        "shipped_words": sum(r["words"] for r in shipped),
        "shipped_differing": sum(r["differing"] for r in shipped),
        "shipped_flushed": sum(r["flushed"] for r in shipped),
        "shipped_other": sum(r["other"] for r in shipped),
        "cases": len(shipped),
        "subnormal_free_cases": sum(1 for r in shipped
                                    if not r["subnormal_in"]
                                    and not r["subnormal_reference"]),
        # STATED BECAUSE IT IS A NULL RESULT AND NULLS GET DROPPED. The contract
        # guard is NOT exercised by this kernel: the ``fast`` arm measured the same
        # zero as the shipped one, so this leg says nothing about the pragma. The
        # reason is visible in the source — ``w`` is live across the loop iteration
        # (it becomes the next ``prev``), so the multiply cannot be folded into the
        # subtract without materialising it anyway. The guard is still spelled on
        # this kernel, because the CURL kernel beside it genuinely needs it and one
        # spelling for the package is the rule; but a reader must not take this leg
        # as evidence that it bites here.
        "contract_guard_discriminates": bool(
            sum(r["other"] for r in fast) != sum(r["other"] for r in shipped)),
        "fast_other": sum(r["other"] for r in fast),
        "verdict": ("IDENTICAL-MODULO-FLUSH"
                    if sum(r["other"] for r in shipped) == 0 else "DIVERGENT"),
    }
    kit.save(payload, out)


def _one_prefix_case(entry: Any, rng: Any, shape, ir0: float, side: str,
                     wall: bool, mode: str, value_class: str) -> Dict[str, Any]:
    source = _seed_volume(rng, shape, value_class)
    reference, scanned, weights, divisor = _host_prefix(source, ir0, wall)

    got = _sentinel(reference.shape)
    device = _upload(got)
    n_cols = shape[1] * shape[2]
    entry(device, _upload(source), _upload(weights), _upload(divisor),
          int(reference.shape[0]), int(shape[1]), int(shape[2]), int(n_cols))
    measured = _download(device, reference.shape)

    moved = kit.differing(measured, got)
    kit.assert_moved(moved, f"prefix/{shape}/{ir0}/{value_class}",
                     floor=int(np.prod(reference.shape)))
    row = {"contract": mode, "wall": bool(wall), "shape": list(shape),
           "ir0": ir0, "side": side, "value_class": value_class,
           "words": int(np.prod(reference.shape)),
           "moved": moved,
           "subnormal_in": _subnormal_words(scanned),
           "subnormal_reference": _subnormal_words(reference),
           "reference_digest": kit.state_digest({"pfx": reference})}
    row.update(classify(measured, reference))
    return row


def _seed_volume(rng: Any, shape, value_class: str) -> Any:
    """Physical-band field values, in three classes with different failure reach."""
    if value_class == "random_band":
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if value_class == "wide_dynamic":
        # A geometric ladder along r: the prefix accumulates down the same axis, so
        # this is where a reassociated scan loses the small terms.
        ladder = (np.float32(10.0) ** np.linspace(-6.0, 2.0, shape[0])
                  .astype(np.float32))
        base = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        return (base * ladder.reshape(-1, 1, 1)).astype(np.float32)
    if value_class == "cancelling":
        # Alternating signs so the running sum passes through zero repeatedly; a
        # different summation order shows up here and nowhere else.
        base = rng.uniform(0.5, 1.5, size=shape).astype(np.float32)
        sign = np.where(np.arange(shape[0]) % 2 == 0, np.float32(1.0),
                        np.float32(-1.0)).reshape(-1, 1, 1)
        return (base * sign).astype(np.float32)
    raise ValueError(f"unknown value class {value_class!r}")


# ---------------------------------------------------------------------------
# Leg 3 — negation and signed zero
# ---------------------------------------------------------------------------

_NEGATION_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

kernel void negation_probe(
    device float*       out   [[buffer(0)]],
    device const float* x     [[buffer(1)]],
    constant uint&      n     [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) { return; }
    float v = x[idx];
    out[idx] = __EXPR__;
}
"""


def leg_negation(payload: Dict[str, Any], out: str) -> None:
    """The three negation spellings on THIS backend, re-measured not inherited.

    ``shaders.py:47-51`` already records the answer for the certified pair. It is
    measured again because this family puts signed zeros live on the axis row —
    ``Dy[0] = 0`` writes one, the ownership mask writes three more — and a family
    that INHERITED the spelling would be repeating a claim rather than making one.
    """
    patterns = np.array(
        [0.0, -0.0, 1.0, -1.0, np.float32(np.finfo(np.float32).tiny),
         -np.float32(np.finfo(np.float32).tiny),
         np.float32(np.finfo(np.float32).max), -np.float32(np.finfo(np.float32).max),
         3.5, -3.5, 1e-30, -1e-30, 1e30, -1e30],
        dtype=np.float32)
    rng = np.random.default_rng(SEED + 3)
    values = np.concatenate([
        patterns,
        rng.uniform(-1.0, 1.0, size=8192).astype(np.float32),
        # Deliberately in the band: this backend FLUSHES, so the census is what
        # makes the row honest rather than the row being dropped.
        (rng.uniform(-1.0, 1.0, size=256) * 1e-42).astype(np.float32),
    ]).astype(np.float32)
    reference = (-values).astype(np.float32)

    rows: List[Dict[str, Any]] = []
    for arm, expr in (("unary_minus", "-v"),
                      ("zero_minus", "0.0f - v"),
                      ("times_minus_one", "v * -1.0f")):
        entry = _compile(_NEGATION_SOURCE.replace("__EXPR__", expr),
                         "negation_probe")
        got = _sentinel(values.shape)
        device = _upload(got)
        entry(device, _upload(values), int(values.size))
        measured = _download(device, values.shape)
        moved = kit.differing(measured, got)
        kit.assert_moved(moved, f"negation/{arm}", floor=int(values.size) - 1)
        differing = kit.differing(measured, reference)
        rows.append({"arm": arm, "expression": expr, "words": int(values.size),
                     "differing": differing, "moved": moved,
                     "subnormal_in": _subnormal_words(values),
                     "subnormal_out": _subnormal_words(measured)})
        kit.log(f"  negation {arm:<16} {differing}/{values.size} differing "
                f"(subnormal in {rows[-1]['subnormal_in']}, "
                f"out {rows[-1]['subnormal_out']})")
        payload["legs"]["negation"] = rows
        kit.save(payload, out)

    payload["verdicts"]["negation"] = {
        row["arm"]: ("IDENTICAL" if row["differing"] == 0 else "DIVERGENT")
        for row in rows}
    kit.save(payload, out)


# ---------------------------------------------------------------------------
# Leg 4 — the r-axis near ghost, on the ARRAY PATH
# ---------------------------------------------------------------------------

def _build_engine(shape, courant: float, m: int = 0):
    """A real cylindrical ``Grid``/``Fields``/``PML`` on NumPy, at m = 0."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=m, boundaries={"z": "metallic"},
                courant=float(courant), xp=np)
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(f"Grid built {tuple(grid.shape)} for {tuple(shape)}")
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    thickness = {"x": (0, max(2, shape[0] // 4)), "z": max(2, shape[2] // 4)}
    return grid, fields, PML(grid=grid, thickness=thickness)


FIELD_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
               "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
               "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")


def _seed_engine(fields: Any, shape, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for name in FIELD_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def leg_axis_ghost(payload: Dict[str, Any], out: str) -> None:
    """Is the CYL_AXIS near ghost OBSERVABLE at m = 0, real storage?

    THE QUESTION LIVES ON THE ARRAY PATH, so no Metal runs here. Two engines are
    seeded identically and stepped by ``stepping.step_B``/``step_D``; the second has
    ``_shift_down``'s CYL_AXIS branch replaced by METALLIC's zero ghost, which is
    exactly what compiling ``BCX = METALLIC`` would give the kernel. If the two
    agree bitwise the substitution is licensed on THIS configuration; if they do
    not, the family needs a real near-ghost arm.

    Why it is plausibly unobservable, and why that is not enough: only ``Dy``
    (partner Hz) and ``Dz`` (partner Hy) take a shift-down along r, both have r-Yee
    shift 0 (``fields.IYEE_SHIFTS``), and ``_mask_non_owned_cells`` zeroes their
    curl at row 0 (stepping.py:1945-1949) — the one row the near ghost writes. That
    is an argument. This leg is the measurement.
    """
    from meep_gpu import stepping  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    original = stepping._shift_down
    for shape in ((20, 1, 40), (13, 1, 9)):
        for courant in (0.5, 0.3141592653589793):
            rows.append(_one_ghost_case(stepping, original, shape, courant))
            kit.log(f"  axis_ghost {str(shape):<12} courant={courant:.6g} "
                    f"differing={rows[-1]['differing']} "
                    f"moved={rows[-1]['moved']} "
                    f"ghost_rows_written={rows[-1]['ghost_rows_written']}")
            payload["legs"]["axis_ghost"] = rows
            kit.save(payload, out)

    payload["verdicts"]["axis_ghost"] = {
        "differing": sum(r["differing"] for r in rows),
        "words": sum(r["words"] for r in rows),
        "ghost_rows_written": sum(r["ghost_rows_written"] for r in rows),
        "verdict": ("UNOBSERVABLE" if sum(r["differing"] for r in rows) == 0
                    else "OBSERVABLE"),
    }
    kit.save(payload, out)


def _one_ghost_case(stepping: Any, original: Any, shape, courant: float
                    ) -> Dict[str, Any]:
    grid_a, fields_a, pml_a = _build_engine(shape, courant)
    grid_b, fields_b, pml_b = _build_engine(shape, courant)
    _seed_engine(fields_a, shape, SEED + 11)
    _seed_engine(fields_b, shape, SEED + 11)
    before = {name: np.array(getattr(fields_a, name)) for name in FIELD_NAMES}

    # Engine A: the shipped array path, CYL_AXIS ghost intact.
    for _ in range(4):
        stepping.step_B(fields_a, pml_a)
        stepping.update_H(fields_a, pml_a)
        stepping.step_D(fields_a, pml_a)
        stepping.update_E(fields_a, pml_a)

    # Engine B: the same path with the CYL_AXIS near ghost replaced by METALLIC's
    # zero, counting how many times the substitution actually fired. A count of
    # zero would mean the leg measured nothing at all.
    calls = {"count": 0}

    def metallic_near_ghost(xp, field, axis, boundary, component, phase=None,
                            mirror_phase=None, scratch=None,
                            scratch_tag="shift_down"):
        if boundary == stepping.CYL_AXIS:
            calls["count"] += 1
            boundary = stepping.METALLIC
            mirror_phase = None
        return original(xp, field, axis, boundary, component, phase, mirror_phase,
                        scratch, scratch_tag)

    stepping._shift_down = metallic_near_ghost
    try:
        for _ in range(4):
            stepping.step_B(fields_b, pml_b)
            stepping.update_H(fields_b, pml_b)
            stepping.step_D(fields_b, pml_b)
            stepping.update_E(fields_b, pml_b)
    finally:
        stepping._shift_down = original

    differing = sum(kit.differing(getattr(fields_a, n), getattr(fields_b, n))
                    for n in FIELD_NAMES)
    moved = sum(kit.differing(getattr(fields_a, n), before[n])
                for n in FIELD_NAMES)
    kit.assert_moved(moved, f"axis_ghost/{shape}/{courant}", floor=1000)
    kit.assert_census_floor(calls["count"], f"axis_ghost substitutions {shape}")
    return {"shape": list(shape), "courant": repr(courant), "steps": 4,
            "words": sum(int(getattr(fields_a, n).size) for n in FIELD_NAMES),
            "differing": differing, "moved": moved,
            "ghost_rows_written": calls["count"],
            "per_component": {n: kit.differing(getattr(fields_a, n),
                                               getattr(fields_b, n))
                              for n in FIELD_NAMES}}


# ---------------------------------------------------------------------------
# Leg 5 — the emitters, against the array path's own mask
# ---------------------------------------------------------------------------

def leg_emitters(payload: Dict[str, Any], out: str) -> None:
    """Does compiling the r axis as METALLIC reproduce the is_axis mask exactly?

    The whole design rests on reusing ``shaders.ownership_mask`` and
    ``shaders.ghost`` unchanged, with ``codes[0] = METALLIC`` standing in for
    CYL_AXIS. That is a claim about which CELLS get zeroed, so it is checked against
    ``_mask_non_owned_cells`` running on a real cylindrical grid — the function
    itself, not a reading of it.
    """
    from meep_gpu import stepping  # noqa: PLC0415
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    shape = (20, 1, 40)
    grid, fields, pml = _build_engine(shape, 0.5)
    kinds = stepping._boundary_kinds(grid, pml)
    rows: List[Dict[str, Any]] = []

    for sub_step, terms, backward in (("step_B", stepping.B_CURL_TERMS, False),
                                      ("step_D", stepping.D_CURL_TERMS, True)):
        # What the array path masks, per target: run the real function on a volume
        # of ones and read which cells came back zero.
        array_path: Dict[str, Any] = {}
        for term in terms:
            probe = np.ones(shape, dtype=np.float32)
            stepping._mask_non_owned_cells(probe, grid, term.iyee)
            array_path[term.target] = (probe == 0.0)

        # What the emitter would mask, decoded from the source it produces with the
        # r axis compiled as METALLIC.
        codes = tuple(shaders.METALLIC if kind in ("axis", "metallic")
                      else shaders.PERIODIC for kind in kinds)
        emitted = shaders.ownership_mask(codes, backward)
        kernel_mask = _decode_mask(emitted, shape, terms)

        agreed = all(np.array_equal(array_path[term.target],
                                    kernel_mask[term.target]) for term in terms)
        masked_cells = int(sum(int(array_path[t.target].sum()) for t in terms))
        kit.assert_census_floor(masked_cells, f"emitters/{sub_step} masked cells")
        rows.append({"sub_step": sub_step, "codes": list(codes),
                     "boundary_kinds": list(kinds),
                     "emitted": emitted, "agreed": bool(agreed),
                     "masked_cells": masked_cells,
                     "per_target": {t.target: int(array_path[t.target].sum())
                                    for t in terms}})
        kit.log(f"  emitters {sub_step} agreed={agreed} masked={masked_cells}")
        assert agreed, (
            f"shaders.ownership_mask with the r axis as METALLIC does NOT reproduce "
            f"_mask_non_owned_cells on a cylindrical grid for {sub_step}: "
            f"{rows[-1]}")
        payload["legs"]["emitters"] = rows
        kit.save(payload, out)

    payload["verdicts"]["emitters"] = {
        "agreed": all(r["agreed"] for r in rows),
        "sub_steps": [r["sub_step"] for r in rows]}
    kit.save(payload, out)


def _decode_mask(emitted: str, shape, terms) -> Dict[str, Any]:
    """Turn the emitter's ``curl<t> = at_<axis> ? 0.0f : curl<t>;`` lines into masks."""
    flags = {"at_x": np.zeros(shape, dtype=bool),
             "at_y": np.zeros(shape, dtype=bool),
             "at_z": np.zeros(shape, dtype=bool)}
    flags["at_x"][0, :, :] = True
    flags["at_y"][:, 0, :] = True
    flags["at_z"][:, :, 0] = True
    out = {term.target: np.zeros(shape, dtype=bool) for term in terms}
    order = [term.target for term in terms]
    for line in emitted.splitlines():
        line = line.strip()
        if not line.startswith("curl"):
            continue
        target = order[int(line[4])]
        flag = line.split("=", 1)[1].split("?", 1)[0].strip()
        out[target] |= flags[flag]
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS = (("divide", leg_divide),
        ("prefix", leg_prefix),
        ("negation", leg_negation),
        ("axis_ghost", leg_axis_ghost),
        ("emitters", leg_emitters))


def main(argv: List[str]) -> int:
    parser = kit.argument_parser(__doc__ or "")
    arguments = parser.parse_args(argv)
    out = os.path.abspath(arguments.out)
    started = time.time()

    payload: Dict[str, Any] = {
        "probe": "metal_cylindrical_real",
        "environment": kit.environment_stamp(),
        "legs": {},
        "verdicts": {},
        "evidence_class": (
            "BEHAVIOURAL ONLY. torch.mps.compile_shader exposes no AIR, no GPU ISA "
            "and no optimisation report (metal_kernels/device.py:46-62), so these "
            "legs certify that the ANSWERS agree and never that the INSTRUCTIONS "
            "do. They catch a wrong answer, not a wrong instruction."),
    }
    kit.save(payload, out)

    try:
        import torch  # noqa: PLC0415

        if not torch.backends.mps.is_available():
            return kit.cannot_certify(payload, out, ["no MPS device on this host"])
    except Exception as exc:  # noqa: BLE001
        return kit.cannot_certify(payload, out, [f"torch unavailable: {exc!r}"])

    ran = kit.run_legs(LEGS, payload, out, kit.wanted_legs(arguments.legs))
    verdicts = payload["verdicts"]
    certified = (
        verdicts.get("divide", {}).get("verdict")
        in (None, "IDENTICAL-MODULO-FLUSH")
        and verdicts.get("prefix", {}).get("verdict")
        in (None, "IDENTICAL-MODULO-FLUSH")
        and verdicts.get("emitters", {}).get("agreed") in (None, True))
    compared = sum(len(rows) for rows in payload["legs"].values())
    return kit.summarize(
        payload, out,
        claim=("Metal's float32 divide and a column-serial radial scan reproduce "
               "stepping.cylindrical_rderiv_prefix word for word, and the r axis "
               "compiled as METALLIC reproduces the CYL_AXIS ghost and the is_axis "
               "ownership mask on a real m = 0 Dcyl grid"),
        scope="m = 0, real float32 storage, active split-field PML, this host only",
        stated_weakness=payload["evidence_class"],
        started=started, legs_run=ran, compared=compared, certified=certified)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
