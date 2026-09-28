"""THE DYNAMIC-LOOP GATE, Metal leg: is a runtime trip count bit-identical to an
unrolled compile-time twin?

WHY THIS EXISTS, AND WHY THE CUDA ANSWER DOES NOT SETTLE IT. The sibling
``probe_cuda_dynamic_loop_arity.py`` measured the same question on NVCC/PTX and
got IDENTICAL at every arity 0-8, and it could say WHY: the PTX showed NVCC
unrolling by four with a nounroll remainder over a strictly serial chain on one
accumulator, so no reassociation occurred. That is a fact about one compiler.

**THIS BACKEND HAS NO DISASSEMBLY.** ``torch.mps.compile_shader`` takes a source
string and returns callable entry points; it exposes no AIR, no GPU ISA and no
optimisation report, so the mechanism cannot be read the way the CUDA round read
it. Everything this probe reports is therefore BEHAVIOURAL EVIDENCE: it catches a
wrong ANSWER, never a wrong INSTRUCTION. That is the certification gap this
backend carries against the Triton and CUDA tracks and it is stated, not papered
over.

WHAT IS AT STAKE. Triton's ``cyl_complex_pml_curl_step`` carries an UNBOUNDED
``tl.constexpr`` axis (``triton_kernels/cylindrical_complex.py:725``), so its
variant count is ``8 x unbounded`` and no author can enumerate it. Metal
specialises by SOURCE SUBSTITUTION (``metal_kernels/shaders.py``), so an unbounded
compile-time axis is an unbounded set of source strings. If a runtime-valued axis
is bit-identical to the specialised one, the axis collapses to a bound uniform and
the family is finite; if it is not, every arity the corpus asks for must be
compiled separately and the port pays a variant per arity.

WHAT IT MEASURES, in the order it measures it:

1. ``accumulate`` - THE HEADLINE. A strictly serial float32 accumulation over an
   arity axis: ``acc = acc + coef[p]*state[p]``, one trip per pole, the shape the
   ADE ``update_P`` pole recurrence and the off-diagonal row mask both have. The
   subject is a loop whose bound comes from a ``constant uint&`` buffer; the twin
   is the same arithmetic emitted as straight-line source with literal indices.
   Compared as UINT32 WORDS.

2. ``threshold`` - WHAT CYLINDRICAL ACTUALLY NEEDS, which is NOT a sum. The
   Triton kernel spends ``ZERO_ROWS`` at exactly one site,
   ``cylindrical_complex.py:971`` ``near = i < ZERO_ROWS``: an INTEGER comparison
   bound on the row index, not a trip count and not a reduction. This leg measures
   the runtime-uniform spelling of that compare against the compile-time-constant
   one. No float rounds on that path, so the expectation is identity; the leg
   exists because "no float rounds" is a reading and this project measures.

ARMED CONTROLS, so that a null result is not vacuous:

* ``dynamic_reversed`` / ``unrolled_reversed`` - the same operations in the
  opposite order. If reversal is NOT caught then these operands cannot separate
  two association orders at all and the leg's "identical" is about the draw, not
  about the compiler. INAPPLICABLE at arity 0 and 1, where there is no order to
  reverse; the leg reports those as ``inapplicable`` rather than as passes.
* ``dynamic_short`` - trip count minus one. Catches a loop that ran the wrong
  number of times. Inapplicable at arity 0.
* ``threshold_off_by_one`` - the compare spelled ``<=``. Catches a threshold that
  is off by a row.
* ``dynamic_notrips`` - the subject WITHOUT its trip counter store. It must be
  bit-identical to ``dynamic``, which is what makes the trip counter admissible as
  evidence rather than a perturbation of the thing being measured.

VACUITY FLOORS, every leg, every case:

* ``moved_words`` > 0 against a pre-launch sentinel. Zero-init is a fixed point of
  most of this engine's sub-steps and two kernels that wrote nothing compare
  identical.
* the value classes are drawn in the PHYSICAL BAND and a ``cancelling`` class is
  swept deliberately, because a draw whose partial sums never cancel cannot
  separate association orders.
* the trip counts the dynamic kernels report must equal the arity they were given.
* NO SUBNORMALS may be produced. Subnormals flush on this platform and cannot be
  unflushed, so a case that entered the band would be measuring the flush and not
  the loop. Each case censuses its own inputs and outputs and refuses.

Run::

    python -u probe_metal_dynamic_loop_arity.py \\
        --out results/<dir>/metal_dynamic_loop_arity.json \\
        --jsonl results/<dir>/metal_dynamic_loop_arity.jsonl

Progress is one line per case on stdout, unbuffered, and every case is appended to
the JSONL as it lands, so an interrupted run keeps everything up to the failure
(the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The sweep axes
# ---------------------------------------------------------------------------

#: Arities swept. 0 and 1 are the degenerate ends the family must not trip over
#: (``zero_rows`` is 0 at |m| = 1 and a one-pole Lorentzian is the commonest ADE
#: row); 2, 3 and 5 are the |m| values ``perturbation_theory`` and ``ring-cyl``
#: actually ask for; 8 matches the largest arity the CUDA sibling swept, so the
#: two tracks' answers are comparable at the same point.
ARITIES: Tuple[int, ...] = (0, 1, 2, 3, 5, 8)

#: Lanes per launch. Every lane is an independent accumulation, so this is also
#: the number of uint32 words compared per (case, variant) pair.
LANES = 1 << 16

#: The contraction modes. ``off`` is the shipped one — the pragma is what buys
#: byte-identity on this platform at all — and ``fast`` is swept because a
#: divergence that appears only under contraction would be a fact about the
#: pragma rather than about the loop, and the two must not be confused.
CONTRACT_MODES: Tuple[str, ...] = ("off", "fast")

#: Fraction of words an armed control must disturb before the leg will call the
#: headline meaningful.
CONTROL_CATCH_FLOOR = 0.05

#: The smallest normal float32. Anything at or below it in magnitude (and not
#: exactly zero) is in the band this platform flushes, and a case that produced
#: one would be measuring the flush.
SMALLEST_NORMAL = np.float32(np.ldexp(1.0, -126))


# ---------------------------------------------------------------------------
# Shader sources
# ---------------------------------------------------------------------------

#: The contraction directive, spelled the way ``metal_kernels.shaders`` spells it.
#: This probe does not import that module — it must stand alone against a staged
#: tree — but the spelling is deliberately the same string.
_PRAGMA = "#pragma clang fp contract({mode})"

#: Substitution is by TOKEN REPLACEMENT, never ``str.format``: a shader source is
#: full of braces and ``format`` would try to read every one of them as a field.
#: ``metal_kernels.shaders.substitute`` makes the same choice for the same reason.
_PREAMBLE = """
#include <metal_stdlib>
using namespace metal;

__PRAGMA__
"""

#: THE SUBJECT: the trip count is read from a buffer, so no constant folding is
#: available to the compiler. The chain is STRICTLY SERIAL on one accumulator —
#: that is the property the CUDA round attributed its null to, and stating it here
#: keeps the two measurements about the same shape.
_DYNAMIC = _PREAMBLE + """
kernel void arity_dynamic(
    device float*        out    [[buffer(0)]],
    device const float*  base   [[buffer(1)]],
    device const float*  coef   [[buffer(2)]],
    device const float*  state  [[buffer(3)]],
    device uint*         trips  [[buffer(4)]],
    constant uint&       arity  [[buffer(5)]],
    constant uint&       lanes  [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= lanes) { return; }
    float acc = base[idx];
    uint n = 0;
    for (uint p = 0; p < arity; ++p) {
        acc = acc + coef[p * lanes + idx] * state[p * lanes + idx];
        n = n + 1;
    }
__TRIPSTORE__
    out[idx] = acc;
}
"""

#: THE TWIN: the same addresses, the same order, the same arithmetic, emitted as
#: straight-line source with LITERAL indices. This is what source substitution
#: would produce if the arity were specialised the way Triton specialises it.
_UNROLLED = _PREAMBLE + """
kernel void arity_unrolled(
    device float*        out    [[buffer(0)]],
    device const float*  base   [[buffer(1)]],
    device const float*  coef   [[buffer(2)]],
    device const float*  state  [[buffer(3)]],
    device uint*         trips  [[buffer(4)]],
    constant uint&       arity  [[buffer(5)]],
    constant uint&       lanes  [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= lanes) { return; }
    float acc = base[idx];
__BODY__
    trips[idx] = __ARITY__u;
    out[idx] = acc;
}
"""

#: The integer-threshold shape, which is what the cylindrical near-axis rule
#: really is (``cylindrical_complex.py:971``). Two spellings of one compare.
_THRESHOLD = _PREAMBLE + """
kernel void threshold_kernel(
    device float*        out    [[buffer(0)]],
    device const float*  base   [[buffer(1)]],
    device const float*  coef   [[buffer(2)]],
    device const float*  state  [[buffer(3)]],
    device uint*         trips  [[buffer(4)]],
    constant uint&       arity  [[buffer(5)]],
    constant uint&       lanes  [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= lanes) { return; }
    // `i` stands in for the radial row index the cylindrical kernel decomposes
    // out of the flat cell index; the rule zeroes every volume on rows within
    // ZERO_ROWS of the axis.
    uint i = idx % 64u;
    float v = base[idx] + coef[idx] * state[idx];
    bool near = __COMPARE__;
    out[idx] = near ? 0.0f : v;
    trips[idx] = near ? 1u : 0u;
}
"""


def _dynamic_source(contract: str, reversed_order: bool, short: bool,
                    store_trips: bool) -> str:
    """One runtime-trip-count variant."""
    body = _DYNAMIC
    if reversed_order:
        body = body.replace(
            "    for (uint p = 0; p < arity; ++p) {",
            "    for (uint q = 0; q < arity; ++q) {\n        uint p = arity - 1u - q;")
    if short:
        body = body.replace("p < arity;", "p + 1u < arity;")
    trip_store = "    trips[idx] = n;" if store_trips else "    trips[idx] = 0u;"
    return (body
            .replace("__TRIPSTORE__", trip_store)
            .replace("__PRAGMA__", _PRAGMA.format(mode=contract)))


def _unrolled_source(contract: str, arity: int, reversed_order: bool) -> str:
    """The straight-line twin at one compile-time arity."""
    order = range(arity - 1, -1, -1) if reversed_order else range(arity)
    lines = [f"    acc = acc + coef[{p}u * lanes + idx] * state[{p}u * lanes + idx];"
             for p in order]
    body = "\n".join(lines) or "    // arity 0: the accumulator is the seed"
    return (_UNROLLED
            .replace("__BODY__", body)
            .replace("__ARITY__", str(arity))
            .replace("__PRAGMA__", _PRAGMA.format(mode=contract)))


def _threshold_source(contract: str, compare: str) -> str:
    return (_THRESHOLD
            .replace("__COMPARE__", compare)
            .replace("__PRAGMA__", _PRAGMA.format(mode=contract)))


# ---------------------------------------------------------------------------
# Value classes
# ---------------------------------------------------------------------------

def _draw(value_class: str, arity: int, lanes: int, seed: int
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(base, coef, state)`` for one case, in the physical band.

    ``random_band`` spans three decades of magnitude with mixed signs — the
    ordinary case. ``cancelling`` is the one that MATTERS for this question: the
    terms are built so the running partial sum passes near zero, which is exactly
    where two association orders separate. A draw that never cancels would let a
    reassociating compiler pass unnoticed.
    """
    rng = np.random.default_rng(seed)
    shape = (max(arity, 1), lanes)
    if value_class == "random_band":
        mag = np.float32(10.0) ** rng.uniform(-3.0, 0.0, size=shape).astype(np.float32)
        coef = (mag * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=shape))
        state = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        base = rng.uniform(-1.0, 1.0, size=lanes).astype(np.float32)
    elif value_class == "cancelling":
        # Alternating large terms with a small tail: the partial sums cancel to
        # near zero and the residue depends on the order the terms arrive in.
        signs = np.where(np.arange(shape[0])[:, None] % 2 == 0, 1.0, -1.0)
        coef = (signs * rng.uniform(0.9, 1.1, size=shape)).astype(np.float32)
        state = (rng.uniform(0.9, 1.1, size=shape)
                 * (np.float32(10.0) ** np.arange(shape[0])[:, None].astype(np.float32)
                    * np.float32(1e-2))).astype(np.float32)
        base = rng.uniform(-1e-2, 1e-2, size=lanes).astype(np.float32)
    elif value_class == "wide_dynamic":
        # A geometric ladder of terms: adding small-to-large and large-to-small
        # give different answers, the classic order-sensitive summation.
        #
        # THE DECADE SPAN IS CHOSEN, NOT ARBITRARY. At -0.6 per pole the smallest
        # term at arity 8 is ~6e-5 against an accumulator of order one, which is
        # ~500 float32 ulps — so the ``dynamic_short`` control (which drops
        # exactly that term) still BITES. A steeper ladder would push the last
        # term below half an ulp, the control would legitimately miss, and the
        # leg would report a draw artefact as a compiler fact. ``base`` is held
        # small for the same reason: a base of order one would dominate the sum
        # and launder the order difference the class exists to expose.
        scale = (np.float32(10.0) ** (np.arange(shape[0])[:, None].astype(np.float32)
                                      * np.float32(-0.6)))
        coef = (scale * rng.uniform(0.5, 1.5, size=shape)).astype(np.float32)
        state = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        base = rng.uniform(-1e-3, 1e-3, size=lanes).astype(np.float32)
    else:
        raise ValueError(f"unknown value class {value_class!r}")
    coef = np.ascontiguousarray(coef, dtype=np.float32)
    state = np.ascontiguousarray(state, dtype=np.float32)
    base = np.ascontiguousarray(base, dtype=np.float32)
    if arity == 0:
        coef = coef[:1] * np.float32(0.0)
        state = state[:1] * np.float32(0.0)
    return base, coef, state


VALUE_CLASSES: Tuple[str, ...] = ("random_band", "cancelling", "wide_dynamic")


def _subnormal_words(array: np.ndarray) -> int:
    """How many words of ``array`` are in the band this platform flushes."""
    flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    magnitude = np.abs(flat)
    return int(np.count_nonzero((magnitude > 0.0) & (magnitude < SMALLEST_NORMAL)))


# ---------------------------------------------------------------------------
# The device leg
# ---------------------------------------------------------------------------

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

    def launch(self, source: str, entry: str, base: np.ndarray, coef: np.ndarray,
               state: np.ndarray, arity: int, lanes: int
               ) -> Tuple[np.ndarray, np.ndarray, int]:
        """One launch on FRESH device copies. Returns ``(out, trips, moved)``.

        The output buffer is pre-filled with a SENTINEL rather than zeros, so
        ``moved`` counts words the kernel actually wrote and a kernel that wrote
        nothing cannot report a vacuous identity.
        """
        torch = self.torch
        library = self.library(source)
        sentinel = np.full(lanes, np.float32(-7.5), dtype=np.float32)
        d_out = torch.from_numpy(sentinel.copy()).to("mps")
        d_base = torch.from_numpy(base.copy()).to("mps")
        d_coef = torch.from_numpy(coef.reshape(-1).copy()).to("mps")
        d_state = torch.from_numpy(state.reshape(-1).copy()).to("mps")
        d_trips = torch.from_numpy(np.full(lanes, 0xFFFFFFFF, dtype=np.uint32)).to("mps")
        getattr(library, entry)(d_out, d_base, d_coef, d_state, d_trips,
                                int(arity), int(lanes))
        torch.mps.synchronize()
        self.launches += 1
        out = d_out.cpu().numpy()
        trips = d_trips.cpu().numpy()
        moved = int(np.count_nonzero(out.view(np.uint32) != sentinel.view(np.uint32)))
        return out, trips, moved


def _differing(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))


def _max_ulp(a: np.ndarray, b: np.ndarray) -> int:
    """Signed-magnitude ULP distance, the way a float comparison should spell it."""
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(np.iinfo(np.int32).min) - ai, ai)
    bi = np.where(bi < 0, np.int64(np.iinfo(np.int32).min) - bi, bi)
    return int(np.max(np.abs(ai - bi))) if a.size else 0


def _reference(base: np.ndarray, coef: np.ndarray, state: np.ndarray,
               arity: int) -> np.ndarray:
    """The host's own left-to-right float32 accumulation.

    Reported beside the device numbers rather than asserted against them: NumPy
    and Metal are two compilers and this probe's question is about ONE of them
    against ITSELF. The reference is here so a reader can see the device answers
    are in the right neighbourhood, not as an oracle.
    """
    acc = base.astype(np.float32).copy()
    for p in range(arity):
        acc = (acc + (coef[p].astype(np.float32) * state[p].astype(np.float32))
               ).astype(np.float32)
    return acc


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def run_accumulate(runner: Runner, lanes: int, arities: Sequence[int],
                   emit) -> List[Dict[str, Any]]:
    """THE HEADLINE LEG: runtime trip count vs unrolled twin."""
    rows: List[Dict[str, Any]] = []
    seed = 20260816
    for contract in CONTRACT_MODES:
        for value_class in VALUE_CLASSES:
            for arity in arities:
                seed += 1
                started = time.time()
                base, coef, state = _draw(value_class, arity, lanes, seed)
                in_sub = (_subnormal_words(base) + _subnormal_words(coef)
                          + _subnormal_words(state))

                variants: Dict[str, Tuple[str, str]] = {
                    "dynamic": (_dynamic_source(contract, False, False, True),
                                "arity_dynamic"),
                    "dynamic_notrips": (_dynamic_source(contract, False, False, False),
                                        "arity_dynamic"),
                    "unrolled": (_unrolled_source(contract, arity, False),
                                 "arity_unrolled"),
                    "dynamic_reversed": (_dynamic_source(contract, True, False, True),
                                         "arity_dynamic"),
                    "unrolled_reversed": (_unrolled_source(contract, arity, True),
                                          "arity_unrolled"),
                    "dynamic_short": (_dynamic_source(contract, False, True, True),
                                      "arity_dynamic"),
                }
                results: Dict[str, Any] = {}
                for name, (source, entry) in variants.items():
                    out, trips, moved = runner.launch(source, entry, base, coef,
                                                      state, arity, lanes)
                    results[name] = {"out": out, "trips": trips, "moved": moved}

                subject = results["dynamic"]["out"]
                twin = results["unrolled"]["out"]
                out_sub = _subnormal_words(subject) + _subnormal_words(twin)

                row: Dict[str, Any] = {
                    "leg": "accumulate",
                    "contract": contract,
                    "value_class": value_class,
                    "arity": int(arity),
                    "lanes": int(lanes),
                    "words_compared": int(subject.size),
                    "differing_words": _differing(subject, twin),
                    "max_ulp": _max_ulp(subject, twin),
                    "moved_words": {k: v["moved"] for k, v in results.items()},
                    "trips_reported": {
                        k: sorted(set(int(t) for t in v["trips"]))
                        for k, v in results.items()},
                    "input_subnormal_words": in_sub,
                    "output_subnormal_words": out_sub,
                    "reference_differing_words": _differing(
                        subject, _reference(base, coef, state, arity)),
                    "controls": {},
                    "seconds": None,
                }
                for control in ("dynamic_reversed", "unrolled_reversed",
                                "dynamic_short"):
                    caught = _differing(subject, results[control]["out"])
                    applicable = arity >= (1 if control == "dynamic_short" else 2)
                    row["controls"][control] = {
                        "differing_words": caught,
                        "catch_fraction": caught / subject.size,
                        "applicable": applicable,
                        "verdict": ("inapplicable" if not applicable else
                                    "CAUGHT" if caught / subject.size
                                    >= CONTROL_CATCH_FLOOR else "MISSED"),
                    }
                row["notrips_differing_words"] = _differing(
                    subject, results["dynamic_notrips"]["out"])
                row["verdict"] = ("IDENTICAL" if row["differing_words"] == 0
                                  else "DIVERGENT")
                row["seconds"] = round(time.time() - started, 3)
                rows.append(row)
                emit(row)
                controls = ",".join(
                    f"{k.split('_')[-1]}={v['verdict']}"
                    for k, v in row["controls"].items())
                log(f"  accumulate contract={contract:4s} {value_class:12s} "
                    f"arity={arity} -> {row['verdict']} "
                    f"diff={row['differing_words']}/{row['words_compared']} "
                    f"ulp={row['max_ulp']} moved={row['moved_words']['dynamic']} "
                    f"trips={row['trips_reported']['dynamic']} "
                    f"controls[{controls}] ({row['seconds']}s)")
    return rows


def run_threshold(runner: Runner, lanes: int, thresholds: Sequence[int],
                  emit) -> List[Dict[str, Any]]:
    """The cylindrical near-axis shape: a runtime uniform vs a literal compare."""
    rows: List[Dict[str, Any]] = []
    seed = 40260816
    for contract in CONTRACT_MODES:
        for threshold in thresholds:
            seed += 1
            started = time.time()
            base, coef, state = _draw("random_band", 1, lanes, seed)
            dynamic_src = _threshold_source(contract, "i < arity")
            literal_src = _threshold_source(contract, f"i < {threshold}u")
            off_by_one_src = _threshold_source(contract, f"i <= {threshold}u")

            subject, s_trips, s_moved = runner.launch(
                dynamic_src, "threshold_kernel", base, coef[0], state[0],
                threshold, lanes)
            twin, t_trips, t_moved = runner.launch(
                literal_src, "threshold_kernel", base, coef[0], state[0],
                threshold, lanes)
            control, _, _ = runner.launch(
                off_by_one_src, "threshold_kernel", base, coef[0], state[0],
                threshold, lanes)

            caught = _differing(subject, control)
            zeroed = int(np.count_nonzero(s_trips == 1))
            row = {
                "leg": "threshold",
                "contract": contract,
                "threshold": int(threshold),
                "lanes": int(lanes),
                "words_compared": int(subject.size),
                "differing_words": _differing(subject, twin),
                "max_ulp": _max_ulp(subject, twin),
                "moved_words": {"dynamic": s_moved, "literal": t_moved},
                "rows_zeroed": zeroed,
                "output_subnormal_words": (_subnormal_words(subject)
                                           + _subnormal_words(twin)),
                "controls": {
                    "threshold_off_by_one": {
                        "differing_words": caught,
                        "catch_fraction": caught / subject.size,
                        "applicable": True,
                        "verdict": ("CAUGHT" if caught > 0 else "MISSED"),
                    },
                },
                "verdict": None,
                "seconds": None,
            }
            row["verdict"] = ("IDENTICAL" if row["differing_words"] == 0
                              else "DIVERGENT")
            row["seconds"] = round(time.time() - started, 3)
            rows.append(row)
            emit(row)
            log(f"  threshold  contract={contract:4s} zero_rows={threshold} -> "
                f"{row['verdict']} diff={row['differing_words']}/"
                f"{row['words_compared']} zeroed={zeroed} moved={s_moved} "
                f"control={row['controls']['threshold_off_by_one']['verdict']} "
                f"({row['seconds']}s)")
    return rows


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def summarize(rows: Sequence[Dict[str, Any]], runner: Runner) -> Dict[str, Any]:
    accumulate = [r for r in rows if r["leg"] == "accumulate"]
    threshold = [r for r in rows if r["leg"] == "threshold"]
    failures: List[str] = []

    for row in rows:
        if min(row["moved_words"].values()) <= 0:
            failures.append(
                f"VACUOUS: {row['leg']} {row.get('value_class', '')} "
                f"arity/threshold={row.get('arity', row.get('threshold'))} moved "
                f"{row['moved_words']} words — a kernel that wrote nothing "
                f"compares identical to another that wrote nothing")
        if row["output_subnormal_words"]:
            failures.append(
                f"SUBNORMAL: {row['leg']} produced "
                f"{row['output_subnormal_words']} words in the flushed band; "
                f"this case measures the flush, not the loop")
        for name, control in row["controls"].items():
            if control["verdict"] == "MISSED":
                failures.append(
                    f"CONTROL MISSED: {row['leg']} {name} at "
                    f"{row.get('value_class', '')} "
                    f"{row.get('arity', row.get('threshold'))} disturbed "
                    f"{control['differing_words']} words — this draw cannot "
                    f"separate two orders, so its null is about the draw")

    for row in accumulate:
        expected = [0] if row["arity"] == 0 else [row["arity"]]
        if row["trips_reported"]["dynamic"] != expected:
            failures.append(
                f"TRIP COUNT: arity {row['arity']} reported "
                f"{row['trips_reported']['dynamic']}, expected {expected}")
        if row["notrips_differing_words"]:
            failures.append(
                f"TRIP COUNTER PERTURBS: arity {row['arity']} "
                f"{row['contract']} {row['value_class']} — the counter store "
                f"changed {row['notrips_differing_words']} words, so the counter "
                f"is not admissible as evidence")

    divergent = [r for r in rows if r["verdict"] == "DIVERGENT"]
    applicable_controls = sum(
        1 for r in rows for c in r["controls"].values() if c["applicable"])
    caught_controls = sum(
        1 for r in rows for c in r["controls"].values()
        if c["verdict"] == "CAUGHT")

    return {
        "verdict": ("DIVERGENT" if divergent else
                    "INCONCLUSIVE" if failures else "IDENTICAL"),
        "cases": len(rows),
        "accumulate_cases": len(accumulate),
        "threshold_cases": len(threshold),
        "words_compared": sum(r["words_compared"] for r in rows),
        "differing_words": sum(r["differing_words"] for r in rows),
        "max_ulp": max([r["max_ulp"] for r in rows], default=0),
        "divergent_cases": [
            {k: r[k] for k in ("leg", "contract", "value_class", "arity",
                               "threshold", "differing_words") if k in r}
            for r in divergent],
        "armed_controls_applicable": applicable_controls,
        "armed_controls_caught": caught_controls,
        "compiles": runner.compiles,
        "launches": runner.launches,
        "failures": failures,
        "evidence_class": (
            "BEHAVIOURAL ONLY. torch.mps.compile_shader exposes no AIR, no GPU ISA "
            "and no optimisation report, so this probe cannot read the mechanism "
            "the way the CUDA sibling read its PTX. It certifies that the answers "
            "agree, never that the instructions do."),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="summary JSON path")
    parser.add_argument("--jsonl", default=None, help="per-case JSONL path")
    parser.add_argument("--lanes", type=int, default=LANES)
    parser.add_argument("--legs", default="accumulate,threshold")
    args = parser.parse_args(argv)

    legs = tuple(part.strip() for part in args.legs.split(",") if part.strip())
    handle = None
    if args.jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.jsonl)), exist_ok=True)
        handle = open(args.jsonl, "w")

    def emit(row: Dict[str, Any]) -> None:
        if handle is not None:
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()

    started = time.time()
    log(f"metal dynamic-loop arity probe: lanes={args.lanes} legs={legs}")
    log(f"host: {platform.machine()} macOS {platform.mac_ver()[0]}")
    runner = Runner()
    log(f"torch {runner.torch.__version__} mps={runner.torch.backends.mps.is_available()}")

    rows: List[Dict[str, Any]] = []
    if "accumulate" in legs:
        log("leg 1/2: accumulate (runtime trip count vs unrolled twin)")
        rows += run_accumulate(runner, args.lanes, ARITIES, emit)
    if "threshold" in legs:
        log("leg 2/2: threshold (runtime uniform vs literal compare)")
        rows += run_threshold(runner, args.lanes, (0, 1, 2, 3, 5, 8), emit)

    summary = summarize(rows, runner)
    summary["seconds"] = round(time.time() - started, 3)
    summary["host"] = {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "torch": runner.torch.__version__,
        "numpy": np.__version__,
    }
    if handle is not None:
        handle.close()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as out_handle:
            json.dump({"summary": summary, "rows": rows}, out_handle,
                      indent=2, default=str)

    log("")
    log(f"VERDICT: {summary['verdict']}")
    log(f"  cases            {summary['cases']} "
        f"({summary['accumulate_cases']} accumulate, "
        f"{summary['threshold_cases']} threshold)")
    log(f"  words compared   {summary['words_compared']}")
    log(f"  differing words  {summary['differing_words']}  "
        f"max ulp {summary['max_ulp']}")
    log(f"  armed controls   {summary['armed_controls_caught']} caught of "
        f"{summary['armed_controls_applicable']} applicable")
    log(f"  compiles/launches {summary['compiles']}/{summary['launches']}")
    for failure in summary["failures"]:
        log(f"  FAILURE: {failure}")
    log(f"  evidence: {summary['evidence_class']}")
    return 0 if summary["verdict"] == "IDENTICAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
