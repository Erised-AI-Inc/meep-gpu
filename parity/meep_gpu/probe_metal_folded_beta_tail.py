"""THE FOLDED-BETA SPELLING PROBE: what does THIS backend do with the beta tail?

WHY IT EXISTS. ``metal_kernels/folded_beta.py`` is a composition of two families
that are each already byte-certified on this host, and the whole point of writing
it as a family rather than as a flag is that a composition may still carry
arithmetic neither parent carries. The beta tail is the one expression the fold
puts in a new place — between the curl and TWO ownership masks instead of one — so
its spelling is re-measured here rather than inherited from ``special_kz``'s round
or, worse, from the Triton track's.

WHAT IS DELIBERATELY NOT INHERITED. Triton spells its negations ``* -1.0`` because
*Triton* lowers unary minus as ``0.0 - x`` and canonicalizes a zero's sign. That is
a fact about Triton. This probe asks the three spellings on Metal and reports what
it measured; ``metal_kernels/templates.py`` records the same answer from an earlier
round and this leg is the independent re-measurement, not a citation of it.

THE FOUR LEGS:

1. ``real_tail`` — the REAL arm's ``curl = curl - (beta * partner)``
   (``special_kz._REAL_BETA_INSERT``) against the array path's
   ``curl + (-(c32 * partner))`` (stepping.py:811 then :389). Five refutable
   respellings ride along, three expected null and two expected to bite.
2. ``complex_tail`` — the COMPLEX arm's
   ``curl = curl - c_mul(float2(bpr, bpi), partner)``
   (``special_kz._COMPLEX_BETA_INSERT``) against numpy's own complex64 product.
   The coefficient here is PURELY IMAGINARY (stepping.py:798-799 multiplies by
   ``+/-1j``), which is exactly the case where an orientation swap is NOT free —
   the leg measures that rather than assuming it either way.
3. ``uniform_vs_literal`` — the coefficient bound as ``constant float&`` against
   the same value baked into the source as a decimal literal. This is the leg that
   licenses the family's ONE runtime-versus-compile-time choice: the beta words
   stay bound uniforms and are never specialised into the string. The arity round
   (``probe_metal_dynamic_loop_arity.py``) measured the loop-shaped version of this
   question and got IDENTICAL; this measures the scalar-shaped version the
   folded-beta family actually makes.
4. ``mask_order`` — THE COMPOSITION'S OWN, and the one no parent can measure. The
   fold adds a SECOND ownership mask after the first, and the array path applies
   BOTH after the beta term (stepping.py:384-391 then :397, whose
   ``_mask_non_owned_cells`` carries the cell-0 arm at :1945-1949 and the top-plane
   arm at :1887-1897). A kernel that put the beta term after either mask would
   write a beta contribution into a cell the array path zeroes. The leg builds the
   three orders and requires the two wrong ones to DIVERGE, which is what makes the
   family's ordering claim refutable rather than decorative.

EVIDENCE CLASS, STATED. ``torch.mps.compile_shader`` exposes no AIR, no GPU ISA and
no optimisation report (``metal_kernels/device.py:46-62``), so nothing here reads
the emitted code. Every number below is BEHAVIOURAL: it catches a wrong answer, not
a wrong instruction. That is this backend's standing certification gap against the
Triton and CUDA tracks and it stays stated on every claim built on top of it.

SUBNORMALS. They flush on this platform and cannot be unflushed, so every leg
censuses its own inputs and outputs and the value tables are built BAND-FREE. A
case that entered the band would be measuring the flush rather than the spelling.
The census is reported per case, not asserted away.

Run::

    python -u probe_metal_folded_beta_tail.py \\
        --out <dir>/metal_folded_beta_tail.json \\
        --jsonl <dir>/metal_folded_beta_tail.jsonl

Progress is one line per case, unbuffered, and every case is appended to the JSONL
as it lands (the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)


def log(message: str) -> None:
    print(message, flush=True)


#: The contraction modes. ``off`` is the shipped one — the file-scope pragma is what
#: buys byte-identity on this platform at all — and ``fast`` is swept because a
#: divergence that appeared only under contraction would be a fact about the pragma
#: and not about the spelling, and the two must not be confused.
CONTRACT_MODES: Tuple[str, str] = ("off", "fast")

#: The smallest normal float32. Anything below it in magnitude and not exactly zero
#: is in the band this platform flushes.
SMALLEST_NORMAL = np.float32(np.ldexp(1.0, -126))

#: A BAND-FREE, OVERFLOW-FREE exhaustive table: both signed zeros and both signs at
#: five magnitudes spanning eighteen decades either side of one.
#:
#: THE ENDPOINTS ARE CHOSEN, NOT ARBITRARY, and the reason is the platform. Every
#: PRODUCT of two entries lies in ``[1e-18, 1e36]`` and every SUM of two products
#: stays under ``FLT_MAX``, so no lane can reach the subnormal band (which this
#: platform flushes, so a lane there would measure the flush rather than the
#: spelling) and no lane can reach an infinity (whose difference is a NaN, whose
#: PAYLOAD is unspecified — comparing NaN words across two compilers would
#: manufacture a divergence that is about neither spelling). A table carrying
#: ``FLT_MAX`` and ``FLT_MIN`` was the first draft and it did both.
TABLE: Tuple[float, ...] = (
    0.0, -0.0, 1.0, -1.0, 1.5, -1.5, 1e-9, -1e-9, 1e9, -1e9, 1e18, -1e18,
)

_PRAGMA = "#pragma clang fp contract({mode})"

_PREAMBLE = """
#include <metal_stdlib>
using namespace metal;

__PRAGMA__
"""


# ---------------------------------------------------------------------------
# LEG 1 — the real beta tail
# ---------------------------------------------------------------------------

#: One kernel, one spelling of the tail. ``curl``, ``coef`` and ``partner`` arrive
#: as three independent lanes so the exhaustive cross product is one launch.
_REAL_TAIL = _PREAMBLE + """
kernel void real_tail(
    device float*        out     [[buffer(0)]],
    device const float*  curl    [[buffer(1)]],
    device const float*  coefv   [[buffer(2)]],
    device const float*  partner [[buffer(3)]],
    constant float&      beta    [[buffer(4)]],
    constant uint&       lanes   [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= lanes) { return; }
    float curl0 = curl[idx];
    float c = coefv[idx];
    float b = partner[idx];
    (void)beta;
__BODY__
    out[idx] = curl0;
}
"""

#: The shipped spelling and every respelling this family could plausibly acquire.
#: ``expect`` is what the AUTHOR predicts, recorded BEFORE the run so a surprise is
#: visible as a surprise; the verdict column is what was measured.
#:
#: ``expect`` IS SCOPED TO ``contract=off``, WHICH IS THE SHIPPED MODE. Under
#: ``contract=fast`` the compiler is licensed to contract ``curl - (c * b)`` into an
#: fma and the shipped spelling then diverges from the array path just as the
#: explicit ``fused`` variant does — which is not a surprise, it is the measurement
#: that makes the file-scope pragma MANDATORY rather than advisory. Scoring
#: ``fast`` against these predictions reported that necessity as six failures.
REAL_SPELLINGS: Tuple[Tuple[str, str, str], ...] = (
    # label                     body                                    expect
    ("shipped",                 "    curl0 = curl0 - (c * b);",         "null"),
    ("add_negated",             "    curl0 = curl0 + (-(c * b));",      "null"),
    ("operand_swap",            "    curl0 = curl0 - (b * c);",         "null"),
    ("negate_times_minus_one",  "    curl0 = curl0 + ((c * b) * -1.0f);", "null"),
    ("negate_zero_minus",       "    curl0 = curl0 + (0.0f - (c * b));", "bites"),
    ("fused",                   "    curl0 = fma(-c, b, curl0);",       "bites"),
)


# ---------------------------------------------------------------------------
# LEG 2 — the complex beta tail
# ---------------------------------------------------------------------------

_COMPLEX_HELPERS_FMA = """
static inline float2 c_mul(float2 z, float2 p) {
    return float2(fma(z.x, p.x, -(z.y * p.y)),
                  fma(z.x, p.y,  (z.y * p.x)));
}
"""

_COMPLEX_TAIL = _PREAMBLE + _COMPLEX_HELPERS_FMA + """
kernel void complex_tail(
    device float2*        out     [[buffer(0)]],
    device const float2*  curl    [[buffer(1)]],
    device const float2*  partner [[buffer(2)]],
    constant float&       bpr     [[buffer(3)]],
    constant float&       bpi     [[buffer(4)]],
    constant uint&        lanes   [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= lanes) { return; }
    float2 curl0 = curl[idx];
    float2 b = partner[idx];
__BODY__
    out[idx] = curl0;
}
"""

#: ``expect`` scoped to ``contract=off`` and to the SHIPPED coefficient, which is
#: purely imaginary (stepping.py:798-799 multiplies by ``+/-1j``, so the real word is
#: a signed zero Python's own complex multiply produced).
#:
#: ``orientation_swap`` IS PREDICTED NULL FOR A STATED REASON, and the reason is
#: exactly why the ``general_coefficient`` sweep below exists. Under the FMA_V1
#: expansion the imaginary word is ``fma(z.x, p.y, z.y * p.x)``, which fuses a
#: DIFFERENT product depending on which operand is left — but with ``z.x`` an exact
#: signed zero, ``z.x * p.y`` is exact either way and the two spellings collapse.
#: That is an equivalence SCOPED TO A ZERO REAL WORD, not a general one, and the
#: probe sweeps a general coefficient to keep the scope measured rather than argued.
COMPLEX_SPELLINGS: Tuple[Tuple[str, str, str], ...] = (
    ("shipped",
     "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);", "null"),
    ("add_negated",
     "    curl0 = curl0 + (-c_mul(float2(bpr, bpi), b));", "null"),
    ("orientation_swap",
     "    curl0 = curl0 - c_mul(b, float2(bpr, bpi));", "null"),
    ("negate_zero_minus",
     "    curl0 = curl0 + (float2(0.0f, 0.0f) - c_mul(float2(bpr, bpi), b));",
     "bites"),
    ("imaginary_only_shortcut",
     "    curl0 = curl0 - float2(-(bpi * b.y), (bpi * b.x));", "bites"),
)

#: The two coefficient classes leg 2 sweeps. ``imaginary`` is the shipped one; the
#: ``general`` one exists ONLY so ``orientation_swap``'s null is reported with its
#: scope measured. A family that carried the null unscoped would licence an operand
#: swap in a kernel whose coefficient later stopped being purely imaginary.
COEFFICIENT_CLASSES: Tuple[str, str] = ("imaginary", "general")


# ---------------------------------------------------------------------------
# LEG 3 — a bound uniform against a baked literal
# ---------------------------------------------------------------------------

_UNIFORM_TAIL = _PREAMBLE + """
kernel void uniform_tail(
    device float*        out     [[buffer(0)]],
    device const float*  curl    [[buffer(1)]],
    device const float*  partner [[buffer(2)]],
    constant float&      beta    [[buffer(3)]],
    constant uint&       lanes   [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= lanes) { return; }
    (void)beta;
    float c = __COEF__;
    out[idx] = curl[idx] - (c * partner[idx]);
}
"""


# ---------------------------------------------------------------------------
# LEG 4 — the composition's own: where the beta term sits against TWO masks
# ---------------------------------------------------------------------------

#: A one-dimensional stand-in for the folded curl's tail: a cell-0 mask (``at``),
#: the fold's top-plane mask (``last``) and the beta term, in the three orders an
#: author could write them. Only the first is the array path's.
_MASK_ORDER = _PREAMBLE + """
kernel void mask_order(
    device float*        out     [[buffer(0)]],
    device const float*  curl    [[buffer(1)]],
    device const float*  partner [[buffer(2)]],
    constant float&      beta    [[buffer(3)]],
    constant uint&       lanes   [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= lanes) { return; }
    float curl0 = curl[idx];
    float b = partner[idx];
    bool at   = (idx == 0u);
    bool last = (idx == lanes - 1u);
__BODY__
    out[idx] = curl0;
}
"""

MASK_ORDERS: Tuple[Tuple[str, str, str], ...] = (
    ("beta_then_both_masks",
     "    curl0 = curl0 - (beta * b);\n"
     "    curl0 = at ? 0.0f : curl0;\n"
     "    curl0 = last ? 0.0f : curl0;", "null"),
    ("beta_after_cell_zero_mask",
     "    curl0 = at ? 0.0f : curl0;\n"
     "    curl0 = curl0 - (beta * b);\n"
     "    curl0 = last ? 0.0f : curl0;", "bites"),
    ("beta_after_both_masks",
     "    curl0 = at ? 0.0f : curl0;\n"
     "    curl0 = last ? 0.0f : curl0;\n"
     "    curl0 = curl0 - (beta * b);", "bites"),
    ("top_mask_omitted",
     "    curl0 = curl0 - (beta * b);\n"
     "    curl0 = at ? 0.0f : curl0;", "bites"),
)


# ---------------------------------------------------------------------------
# Machinery
# ---------------------------------------------------------------------------

def _source(template: str, body: str, contract: str, **extra: str) -> str:
    out = template.replace("__BODY__", body)
    for token, value in extra.items():
        out = out.replace(f"__{token}__", value)
    return out.replace("__PRAGMA__", _PRAGMA.format(mode=contract))


def _subnormal_words(array: np.ndarray) -> int:
    flat = np.ascontiguousarray(array).reshape(-1)
    flat = flat.view(np.float32) if flat.dtype == np.complex64 else flat
    magnitude = np.abs(np.ascontiguousarray(flat, dtype=np.float32))
    return int(np.count_nonzero((magnitude > 0.0) & (magnitude < SMALLEST_NORMAL)))


def _words(array: np.ndarray) -> np.ndarray:
    """The array as raw uint32 words — the only comparison this project accepts."""
    return np.ascontiguousarray(array).reshape(-1).view(np.uint32)


def _differing(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(_words(a) != _words(b)))


def _max_ulp(a: np.ndarray, b: np.ndarray) -> int:
    ai = _words(a).view(np.int32).astype(np.int64)
    bi = _words(b).view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(np.iinfo(np.int32).min) - ai, ai)
    bi = np.where(bi < 0, np.int64(np.iinfo(np.int32).min) - bi, bi)
    return int(np.max(np.abs(ai - bi))) if ai.size else 0


class Runner:
    """Compiles on demand and launches one variant against fresh device copies."""

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

    def run(self, source: str, entry: str, arrays: Sequence[np.ndarray],
            scalars: Sequence[Any], lanes: int, complex_out: bool = False
            ) -> Tuple[np.ndarray, int]:
        """One launch on FRESH device copies. Returns ``(out, moved_words)``.

        The output buffer is pre-filled with a SENTINEL rather than zeros: zero-init
        is a fixed point of most of this engine's expressions and two kernels that
        wrote nothing would compare identical.
        """
        torch = self.torch
        library = self.library(source)
        if complex_out:
            sentinel = np.full(lanes, np.complex64(-7.5 - 3.25j), dtype=np.complex64)
        else:
            sentinel = np.full(lanes, np.float32(-7.5), dtype=np.float32)
        d_out = torch.from_numpy(sentinel.copy()).to("mps")
        bound = [d_out] + [torch.from_numpy(np.ascontiguousarray(a).copy()).to("mps")
                           for a in arrays]
        getattr(library, entry)(*bound, *scalars)
        torch.mps.synchronize()
        self.launches += 1
        out = d_out.cpu().numpy()
        moved = int(np.count_nonzero(_words(out) != _words(sentinel)))
        return out, moved


def _cross(table: Sequence[float], width: int) -> List[np.ndarray]:
    """The exhaustive ``width``-fold cross product of ``table``, as lane arrays."""
    combos = np.array(list(itertools.product(table, repeat=width)), dtype=np.float32)
    return [np.ascontiguousarray(combos[:, i]) for i in range(width)]


def _record(sink: Any, row: Dict[str, Any]) -> None:
    if sink is not None:
        sink.write(json.dumps(row) + "\n")
        sink.flush()


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------

def leg_real_tail(runner: Runner, sink: Any) -> List[Dict[str, Any]]:
    """``curl - (c*b)`` against the array path, and five respellings."""
    curl, coef, partner = _cross(TABLE, 3)
    lanes = int(curl.size)
    # The array path: stepping.py:811 rounds `-(c * partner)` once and :389 adds it.
    reference = (curl + (-(coef * partner))).astype(np.float32)
    rows: List[Dict[str, Any]] = []
    for contract in CONTRACT_MODES:
        for label, body, expect in REAL_SPELLINGS:
            source = _source(_REAL_TAIL, body, contract)
            out, moved = runner.run(source, "real_tail",
                                    (curl, coef, partner), (0.0, lanes), lanes)
            differing = _differing(out, reference)
            row = {
                "leg": "real_tail", "contract": contract, "spelling": label,
                "expect": expect, "lanes": lanes, "words": lanes,
                "differing": differing, "max_ulp": _max_ulp(out, reference),
                "moved_words": moved,
                "verdict": "null" if differing == 0 else "bites",
                "subnormal_in": (_subnormal_words(curl) + _subnormal_words(coef)
                                 + _subnormal_words(partner)),
                "subnormal_out": _subnormal_words(out),
                "subnormal_reference": _subnormal_words(reference),
            }
            # The prediction is about the SHIPPED contraction mode only; see
            # REAL_SPELLINGS. Under `fast` the interesting fact is whether the
            # compiler contracted, which is reported rather than scored.
            row["as_expected"] = (row["verdict"] == expect if contract == "off"
                                  else None)
            rows.append(row)
            _record(sink, row)
            mark = ("as expected" if row["as_expected"] else
                    "SURPRISE" if row["as_expected"] is False else
                    "unscored (contract=fast)")
            log(f"  real_tail    contract={contract:4s} {label:24s} "
                f"differing={differing:6d}/{lanes} ulp={row['max_ulp']:>3d} "
                f"moved={moved} -> {row['verdict']:5s} ({mark})")
    return rows


def leg_complex_tail(runner: Runner, sink: Any) -> List[Dict[str, Any]]:
    """``curl - c_mul(coef, partner)``, over both coefficient classes."""
    re_c, im_c, re_p, im_p = _cross(TABLE, 4)
    curl = (re_c + 1j * im_c).astype(np.complex64)
    partner = (re_p + 1j * im_p).astype(np.complex64)
    lanes = int(curl.size)
    rows: List[Dict[str, Any]] = []
    for coefficient_class in COEFFICIENT_CLASSES:
        if coefficient_class == "imaginary":
            # stepping.py:797-799 — `sign * 2*pi*beta*dt` then `* (+/-1j)`, rounded
            # once at :811. The real word is a SIGNED ZERO produced by Python's own
            # complex multiply and is passed through, never synthesized.
            coefficient = np.complex64(complex(0.5 * 2.0 * np.pi * 0.2, 0.0) * 1j)
        else:
            # NOT a shipped coefficient. Its only job is to give the orientation
            # question a case where the real word is NOT an exact zero, so the
            # imaginary class's null is reported with a measured scope.
            coefficient = np.complex64(complex(-0.37218, 1.2566371))
        bpr = float(np.float32(coefficient.real))
        bpi = float(np.float32(coefficient.imag))
        reference = (curl + (-(coefficient * partner))).astype(np.complex64)
        for contract in CONTRACT_MODES:
            for label, body, expect in COMPLEX_SPELLINGS:
                source = _source(_COMPLEX_TAIL, body, contract)
                out, moved = runner.run(source, "complex_tail", (curl, partner),
                                        (bpr, bpi, lanes), lanes, complex_out=True)
                differing = _differing(out, reference)
                row = {
                    "leg": "complex_tail", "contract": contract, "spelling": label,
                    "coefficient_class": coefficient_class,
                    "expect": expect, "lanes": lanes, "words": 2 * lanes,
                    "differing": differing, "max_ulp": _max_ulp(out, reference),
                    "moved_words": moved,
                    "verdict": "null" if differing == 0 else "bites",
                    "coefficient": [bpr, bpi],
                    "subnormal_in": (_subnormal_words(curl)
                                     + _subnormal_words(partner)),
                    "subnormal_out": _subnormal_words(out),
                    "subnormal_reference": _subnormal_words(reference),
                }
                # Scored only on the SHIPPED contraction mode and the SHIPPED
                # coefficient class; the general class is a scope measurement.
                row["as_expected"] = (
                    row["verdict"] == expect
                    if contract == "off" and coefficient_class == "imaginary"
                    else None)
                rows.append(row)
                _record(sink, row)
                mark = ("as expected" if row["as_expected"] else
                        "SURPRISE" if row["as_expected"] is False else "unscored")
                log(f"  complex_tail contract={contract:4s} "
                    f"coef={coefficient_class:9s} {label:24s} "
                    f"differing={differing:6d}/{2*lanes} ulp={row['max_ulp']:>3d} "
                    f"moved={moved} -> {row['verdict']:5s} ({mark})")
    return rows


def leg_uniform_vs_literal(runner: Runner, sink: Any) -> List[Dict[str, Any]]:
    """A ``constant float&`` beta word against the same value baked as a literal."""
    curl, partner = _cross(TABLE, 2)
    lanes = int(curl.size)
    rows: List[Dict[str, Any]] = []
    # Four coefficients, including one that is not representable as a short decimal,
    # so the literal spelling has to carry the full round-trip.
    for beta_word in (np.float32(1.2566371), np.float32(-4.3005),
                      np.float32(0.10000000149011612), np.float32(-1.0)):
        value = float(beta_word)
        literal = f"{np.float32(value).item()!r}f"
        for contract in CONTRACT_MODES:
            bound_src = _source(_UNIFORM_TAIL, "", contract, COEF="beta")
            baked_src = _source(_UNIFORM_TAIL, "", contract, COEF=literal)
            bound, moved_a = runner.run(bound_src, "uniform_tail",
                                        (curl, partner), (value, lanes), lanes)
            baked, moved_b = runner.run(baked_src, "uniform_tail",
                                        (curl, partner), (value, lanes), lanes)
            reference = (curl - (beta_word * partner)).astype(np.float32)
            differing = _differing(bound, baked)
            row = {
                "leg": "uniform_vs_literal", "contract": contract,
                "beta": value, "literal": literal, "lanes": lanes, "words": lanes,
                "differing_bound_vs_baked": differing,
                "differing_bound_vs_numpy": _differing(bound, reference),
                "distinct_sources": int(bound_src != baked_src),
                "moved_words": min(moved_a, moved_b),
                "verdict": "null" if differing == 0 else "bites",
                "subnormal_out": _subnormal_words(bound),
            }
            rows.append(row)
            _record(sink, row)
            log(f"  uniform     contract={contract:4s} beta={value:<12g} "
                f"bound-vs-baked={differing:5d}/{lanes} "
                f"bound-vs-numpy={row['differing_bound_vs_numpy']:5d} "
                f"moved={row['moved_words']} -> {row['verdict']}")
    return rows


def leg_mask_order(runner: Runner, sink: Any) -> List[Dict[str, Any]]:
    """Where the beta term sits against the cell-0 mask AND the fold's top mask."""
    rng = np.random.default_rng(20260816)
    lanes = 4096
    curl = rng.uniform(-1.0, 1.0, size=lanes).astype(np.float32)
    partner = rng.uniform(-1.0, 1.0, size=lanes).astype(np.float32)
    beta = float(np.float32(1.2566371))
    rows: List[Dict[str, Any]] = []
    baseline = None
    for contract in CONTRACT_MODES:
        for label, body, expect in MASK_ORDERS:
            source = _source(_MASK_ORDER, body, contract)
            out, moved = runner.run(source, "mask_order", (curl, partner),
                                    (beta, lanes), lanes)
            if label == MASK_ORDERS[0][0] and contract == CONTRACT_MODES[0]:
                baseline = out
            differing = _differing(out, baseline)
            row = {
                "leg": "mask_order", "contract": contract, "order": label,
                "expect": expect, "lanes": lanes, "words": lanes,
                "differing_vs_array_path_order": differing,
                "moved_words": moved,
                "verdict": "null" if differing == 0 else "bites",
                "subnormal_out": _subnormal_words(out),
            }
            row["as_expected"] = (row["verdict"] == expect
                                  if contract == CONTRACT_MODES[0] else None)
            rows.append(row)
            _record(sink, row)
            mark = ("as expected" if row["as_expected"] else
                    "SURPRISE" if row["as_expected"] is False else "unscored")
            log(f"  mask_order   contract={contract:4s} {label:26s} "
                f"differing={differing:5d}/{lanes} moved={moved} "
                f"-> {row['verdict']:5s} ({mark})")
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="JSON summary path")
    parser.add_argument("--jsonl", default=None, help="per-case JSONL path")
    parser.add_argument("--legs", default="all",
                        help="comma-separated subset of "
                             "real_tail,complex_tail,uniform,mask_order")
    args = parser.parse_args(list(argv))

    started = time.time()
    import torch  # noqa: PLC0415

    log("METAL FOLDED-BETA SPELLING PROBE")
    log(f"  host           : {platform.machine()} {platform.system()} "
        f"{platform.mac_ver()[0] or platform.release()}")
    log(f"  torch          : {torch.__version__}   mps={torch.backends.mps.is_available()}")
    log(f"  numpy          : {np.__version__}")
    log(f"  table          : {len(TABLE)} band-free values, exhaustive cross products")
    log("  EVIDENCE CLASS : behavioural only — this backend exposes no generated "
        "code to audit")
    log("")

    if not torch.backends.mps.is_available():
        log("MPS is not available on this host; nothing can be measured.")
        return 2

    sink = open(args.jsonl, "w") if args.jsonl else None
    runner = Runner()
    wanted = ({"real_tail", "complex_tail", "uniform", "mask_order"}
              if args.legs == "all" else set(args.legs.split(",")))
    rows: List[Dict[str, Any]] = []
    try:
        if "real_tail" in wanted:
            log("LEG 1 — the REAL beta tail")
            rows += leg_real_tail(runner, sink)
        if "complex_tail" in wanted:
            log("LEG 2 — the COMPLEX beta tail (purely imaginary coefficient)")
            rows += leg_complex_tail(runner, sink)
        if "uniform" in wanted:
            log("LEG 3 — bound uniform vs baked literal")
            rows += leg_uniform_vs_literal(runner, sink)
        if "mask_order" in wanted:
            log("LEG 4 — the beta term against BOTH ownership masks")
            rows += leg_mask_order(runner, sink)
    finally:
        if sink is not None:
            sink.close()

    surprises = [r for r in rows if r.get("as_expected") is False]
    vacuous = [r for r in rows if r.get("moved_words", 1) == 0]
    summary = {
        "probe": "metal_folded_beta_tail",
        "host": {"machine": platform.machine(), "system": platform.system(),
                 "release": platform.mac_ver()[0] or platform.release(),
                 "torch": torch.__version__, "numpy": np.__version__},
        "evidence_class": ("behavioural only: torch.mps.compile_shader exposes no "
                           "AIR, no GPU ISA and no optimisation report, so this "
                           "probe catches a wrong answer and never a wrong "
                           "instruction"),
        "table": list(TABLE),
        "contract_modes": list(CONTRACT_MODES),
        "compiles": runner.compiles, "launches": runner.launches,
        "cases": len(rows), "surprises": surprises, "vacuous_cases": vacuous,
        "elapsed_s": round(time.time() - started, 2),
        "rows": rows,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(summary, handle, indent=1)
        log(f"\nwrote {args.out}")

    log("")
    log(f"cases {len(rows)}   compiles {runner.compiles}   "
        f"launches {runner.launches}   elapsed {summary['elapsed_s']}s")
    log(f"SURPRISES (measured verdict != predicted): {len(surprises)}")
    for row in surprises:
        log(f"  {row['leg']}/{row.get('spelling') or row.get('order')} "
            f"contract={row['contract']}: expected {row['expect']}, "
            f"measured {row['verdict']}")
    log(f"VACUOUS CASES (nothing written): {len(vacuous)}")
    return 0 if not vacuous else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
