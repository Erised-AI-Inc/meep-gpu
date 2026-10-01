#!/usr/bin/env python3
"""Conformance gate: does EVERY executor actually obey the requested subnormal policy?

WHAT THIS REPLACES. The alternative to this gate is re-certifying all nine Triton
families under both policies — every gate, every probe, every byte. This gate makes
that unnecessary for the POLICY question specifically: a family's bytes depend on
the policy only through the float32 arithmetic its executors perform, so proving
that every executor obeys one requested policy, on every f32 op class the shipped
kernels use, is what "the policy is uniform" means. (It does not replace the
families' own bit-identity gates, which pin GROUPING; it removes the need to run
each of them twice.)

WHY THIS IS NOT OPTIONAL. A half-applied policy is a divergence generator, measured
rather than feared: the bfast composition is bit-identical on 7/7 cases and 80/80
steps under uniform keep, and diverges at step 1 with 92 differing floats of 2880
across 22 arrays when only Triton is flipped to flush. Three executors decide
independently — the host FPU (MEEP's FTZ/DAZ bits), CuPy's NVRTC binaries
(``-ftz=true``, appended unconditionally), Triton's PTX (IEEE by default) — so
"the flag is set" and "the arithmetic obeys it" are different claims and only the
second one matters.

METHOD, and why it is bit patterns and not literals. Every operand is built on the
host as ``numpy.uint32`` BIT PATTERNS, moved as integers, and reinterpreted as
float32 where it will be used; every verdict is read back as ``uint32`` with no
arithmetic between the operation under test and the readback. A float literal
would not survive: CPython parses literals with dtoa's ``strtod``, which finishes
in floating-point arithmetic, so a module compiled in a flushed process bakes
``0.0`` into ``co_consts`` where its source said ``5e-324`` — and the case would
then "pass" by comparing zero against zero. ``meep_gpu/backends.py`` documents the
same trap one level down and ``meep_gpu/test_backends.py`` forbids the literal.

Lane 0 of every 256-lane cell is an exact power-of-two case whose keep word and
flush word are single known values; lanes 1-255 carry random in-band mantissas so
the verdict is a count over 256 lanes and not one lucky word.

THE EXECUTOR x OP MATRIX is the output. Executors: host NumPy; CuPy ufunc; CuPy
``ElementwiseKernel``; CuPy ``RawKernel`` under this engine's exact
``('--fmad=false',)`` options; CuPy reductions; and Triton on every f32 op class
the shipped kernels use — plain ``+ - * -x``, ``tl.math.fma``, ``tl.math.div_rn``,
``tl.where``, ``!=`` (a comparison whose result is a store mask, 48 ``setp.neu.f32``
in ``conductive_pml_curl_step`` alone) — plus ``tl.cumsum``, which no shipped
kernel uses today and which is carried because it is the one scan class this
engine could reach for.

THREE CLASSES OF OP, and the distinctions are load-bearing. GOVERNED ops are
float32 arithmetic and must follow the policy. TRANSPORT ops (``tl.where``,
load/store) move bits without arithmetic — ``selp.f32`` and ``ld.global.f32``
carry no ``.ftz`` under any policy — so they must KEEP under both, and a transport
op that flushed would be a corruption, not a policy. And EXECUTOR-DEPENDENT ops
are governed on some executors and unreachable on others: ``-x`` is the measured
case, because x86 FTZ/DAZ are MXCSR bits over SSE ARITHMETIC and negation is a
sign flip (``xorps``), so ``np.negative`` keeps a subnormal while the device's
``neg.ftz.f32`` flushes it. :func:`expected_verdict` therefore takes the EXECUTOR
as well as the policy, and the exception is read from
``subnormal_policy.EXECUTOR_OP_EXCEPTIONS`` rather than invented here.

EVERY CELL IS TWO ASSERTIONS, not one. The structural verdict (all 256 lanes kept
/ all flushed / mixed) AND lane 0's exact word against the word that executor must
produce. A verdict alone cannot tell ``0 - x`` (``+0``) from ``neg`` (``-0``): both
read "flush" while the two executors disagree in the sign bit, which is a real
byte divergence between kernels that this gate exists to catch.

THE KNOWN GAP is a leg of its own. A plain ``/`` on floats lowers to
``div.full.f32`` as inline asm, which LLVM never sees, so the ``flush`` policy's
LLVM attribute cannot reach it. Under ``flush`` that compile is expected to be
REFUSED by ``subnormal_policy``'s PTX auditor; the leg asserts the refusal and
records it, so "the auditor is doing its job" is measured rather than assumed. No
shipped kernel contains a plain float ``/`` (the spelling is ``tl.math.div_rn``);
this leg exists so the day one appears, it fails here first.

Progress reporting: one flushed progress line per cell, and the JSON artifact is rewritten
atomically (tmp + fsync + os.replace) after every cell, so an interrupted run
keeps everything up to the failure.

EXIT CODES, because "green" has three failure modes here. 0 PASS. 1 FAIL — a cell
disagreed. 2 REFUSED — the policy would not install, which is a statement about
the host and not about the executors. 3 INCOMPLETE — a requested leg produced no
cells, which is NOT a pass: a green artifact with missing rows reads as "every
executor obeys the policy" while most of them were never asked. ``--legs`` also
scopes which policy executors get installed, so a host-only run on a machine that
merely has CuPy importable is not refused over CuPy's cache-directory convention.

Usage (the GPU host, one clear device)::

    CUDA_VISIBLE_DEVICES=3 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u gate_subnormal_policy.py --policy keep \\
        --out results/subnormal_conformance_<date>/gate.json

    CUDA_VISIBLE_DEVICES=3 \\
        python -u gate_subnormal_policy.py --policy flush \\
        --out results/subnormal_conformance_<date>/gate_flush.json

Laptop (host leg only; the device legs skip by name)::

    python -u gate_subnormal_policy.py --policy keep --out /tmp/conformance.json

Correctness only: this gate makes no throughput or timing claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_API_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from meep_gpu import subnormal_policy as sp  # noqa: E402

N_LANES = 256
SEED = 20260813

# --- bit patterns; never spell any of these as a float literal ---------------
BITS_2P_M75 = 0x1A000000        # 2^-75  (normal)
BITS_2P_M60 = 0x21800000        # 2^-60  (normal)
BITS_2P_M135 = 0x00004000       # 2^-135 (subnormal) — the canonical keep word
BITS_2P_M136 = 0x00002000       # 2^-136 (subnormal)
BITS_SMALLEST_NORMAL = 0x00800000   # 2^-126
BITS_1P25_SMALLEST = 0x00A00000     # 1.25 * 2^-126
BITS_2P_M128 = 0x00200000       # 2^-128 (subnormal)
BITS_ONE = 0x3F800000           # 1.0
SIGN = 0x80000000
MANT = 0x007FFFFF

#: The CUDA C source the RawKernel leg compiles, under the engine's own options.
RAW_KERNEL_SOURCE = r"""
extern "C" {
__global__ void dev_mul(const float* a, const float* b, float* o, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) o[i] = a[i] * b[i];
}
__global__ void dev_sub(const float* a, const float* b, float* o, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) o[i] = a[i] - b[i];
}
__global__ void dev_add(const float* a, const float* b, float* o, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}
__global__ void dev_neg(const float* a, const float* b, float* o, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) o[i] = -a[i];
}
__global__ void dev_div(const float* a, const float* b, float* o, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) o[i] = a[i] / b[i];
}
__global__ void dev_fma(const float* a, const float* b, float* o, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) o[i] = fmaf(a[i], b[i], 0.0f);
}
__global__ void dev_select(const float* a, const float* b, float* o, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) o[i] = (i >= 0) ? a[i] : b[i];
}
__global__ void dev_cmp(const float* a, const float* b, float* o, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) o[i] = (a[i] != b[i]) ? 1.0f : 0.0f;
}
}
"""

#: The engine's own NVRTC options for a RawKernel — the configuration under test,
#: not a convenient one. ``--fmad=false`` is what the array path compiles with.
ENGINE_RAWKERNEL_OPTIONS: Tuple[str, ...] = ("--fmad=false",)


def log(message: str) -> None:
    print(message, flush=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Case builders — pure, and unit-tested on the laptop
# ---------------------------------------------------------------------------


def build_cases(seed: int = SEED, lanes: int = N_LANES) -> List[Dict[str, Any]]:
    """Every op class, as uint32 operand arrays plus the two possible lane-0 words.

    Each case carries ``keep_lane0`` (the IEEE result) and ``flush_lane0`` (what a
    flushing executor produces), so a verdict is a comparison against two KNOWN
    words rather than against a reference computed by an executor whose own policy
    is the thing in question.

    ``governed`` marks float32 arithmetic, which must follow the policy.
    ``governed=False`` marks transport (select, load/store), which must KEEP under
    both policies because no PTX transport opcode carries ``.ftz``.
    """
    rng = np.random.default_rng(seed)

    def rand_mant(n: int) -> np.ndarray:
        return rng.integers(1, MANT + 1, size=n, dtype=np.uint32)

    def with_lane0(word: int, n: int) -> np.ndarray:
        return np.concatenate(([np.uint32(word)], rand_mant(n - 1))).astype(np.uint32)

    cases: List[Dict[str, Any]] = []

    # 1. product of two NORMAL operands whose exact result is subnormal (result FTZ).
    cases.append({
        "op": "mul_result_subnormal", "kind": "mul", "governed": True,
        "a": (np.uint32(BITS_2P_M75) | with_lane0(0, lanes)).astype(np.uint32),
        "b": np.full(lanes, BITS_2P_M60, dtype=np.uint32),
        "keep_lane0": BITS_2P_M135, "flush_lane0": 0x00000000,
        "why": "2^-75 * 2^-60 = 2^-135; the plainest result-flush discriminator",
    })

    # 2. cancellation of two NORMALS to a subnormal difference — the class that
    #    seeds the run-level divergence in the curl sub-steps.
    hi = (np.uint32(BITS_SMALLEST_NORMAL)
          | np.concatenate(([np.uint32(BITS_2P_M128 & MANT)], rand_mant(lanes - 1)))
          ).astype(np.uint32)
    cases.append({
        "op": "sub_cancellation_subnormal", "kind": "sub", "governed": True,
        "a": hi, "b": np.full(lanes, BITS_SMALLEST_NORMAL, dtype=np.uint32),
        "keep_lane0": BITS_2P_M128, "flush_lane0": 0x00000000,
        "why": "1.25*2^-126 - 2^-126 = 2^-128; cancellation is how the curl "
               "sub-steps manufacture subnormals from normal fields",
    })

    # 3. subnormal INPUT times one — flushed only if inputs are flushed (DAZ on the
    #    host; the .ftz modifier flushes inputs as well as results on the device).
    cases.append({
        "op": "mul_subnormal_input", "kind": "mul", "governed": True,
        "a": with_lane0(BITS_2P_M135, lanes),
        "b": np.full(lanes, BITS_ONE, dtype=np.uint32),
        "keep_lane0": BITS_2P_M135, "flush_lane0": 0x00000000,
        "why": "input flush (host DAZ / device .ftz on operands)",
    })

    # 4. signed-zero fingerprint: 0.0 + (-subnormal). Keep returns the operand;
    #    flushing launders the sign to +0.
    cases.append({
        "op": "add_signed_zero", "kind": "add", "governed": True,
        "a": np.zeros(lanes, dtype=np.uint32),
        "b": (np.uint32(SIGN) | with_lane0(BITS_2P_M135, lanes)).astype(np.uint32),
        "keep_lane0": SIGN | BITS_2P_M135, "flush_lane0": 0x00000000,
        "why": "keep preserves the negative sign; flush returns +0, so this case "
               "distinguishes the two by SIGN and not only by magnitude",
    })

    # 5. unary negate of a subnormal. THREE lowerings and three answers, which is
    #    why this case carries per-executor expectations: the host's np.negative
    #    is a sign flip that FTZ/DAZ never see (measured on the GPU host with the FPU
    #    flushing: 0x00004000 -> 0x80004000); CuPy's device kernels lower to
    #    neg.ftz.f32, which flushes to a SIGNED zero; and Triton's ``-x`` lowers to
    #    ``sub.rn.f32 0f00000000, %f``, whose flushed answer is 0 - 0 = +0.
    cases.append({
        "op": "neg_subnormal", "kind": "neg", "governed": True,
        "a": with_lane0(BITS_2P_M135, lanes),
        "b": np.full(lanes, BITS_ONE, dtype=np.uint32),
        "keep_lane0": SIGN | BITS_2P_M135, "flush_lane0": SIGN,
        "governed_on": ("cupy_ufunc", "cupy_elementwise", "cupy_rawkernel", "triton"),
        "flush_lane0_by_executor": {"triton": 0x00000000},
        "why": "neg.ftz.f32 flushes to a SIGNED zero, not +0 — a different flush "
               "word from every other case here; and the host does not flush it "
               "at all, because negation is a bitwise sign flip (xorps) and x86 "
               "FTZ/DAZ govern SSE arithmetic only",
    })

    # 6. fma whose product is subnormal and whose addend is zero — the shipped
    #    kernels' ``tl.math.fma`` op class.
    cases.append({
        "op": "fma_result_subnormal", "kind": "fma", "governed": True,
        "a": (np.uint32(BITS_2P_M75) | with_lane0(0, lanes)).astype(np.uint32),
        "b": np.full(lanes, BITS_2P_M60, dtype=np.uint32),
        "keep_lane0": BITS_2P_M135, "flush_lane0": 0x00000000,
        "why": "fma(2^-75, 2^-60, 0) = 2^-135; tl.math.fma is used 11 times in "
               "the shipped kernels",
    })

    # 7. round-to-nearest divide whose quotient is subnormal — ``tl.math.div_rn``.
    cases.append({
        "op": "div_rn_result_subnormal", "kind": "div", "governed": True,
        "a": with_lane0(BITS_2P_M135, lanes),
        "b": np.full(lanes, BITS_ONE, dtype=np.uint32),
        "keep_lane0": BITS_2P_M135, "flush_lane0": 0x00000000,
        "why": "a subnormal numerator over 1.0; div.rn.f32 is the shipped "
               "spelling (tl.math.div_rn) and already carries .ftz today via "
               "set_nvvm_reflect_ftz, which this gate records rather than hides",
    })

    # 8. TRANSPORT: select between a subnormal and a normal. selp.f32 has no .ftz
    #    form, so this must KEEP under both policies.
    cases.append({
        "op": "select_transport", "kind": "select", "governed": False,
        "a": with_lane0(BITS_2P_M135, lanes),
        "b": np.full(lanes, BITS_ONE, dtype=np.uint32),
        "keep_lane0": BITS_2P_M135, "flush_lane0": BITS_2P_M135,
        "why": "tl.where / selp.f32 moves bits and performs no arithmetic; a "
               "flushing select would be a corruption, not a policy",
    })

    # 9. COMPARISON. ``setp.neu.f32`` is not a rounding op, but it READS its
    #    operands, so flushing decides its answer: a subnormal compared against
    #    zero differs under keep and is equal under flush (host DAZ; the .ftz
    #    modifier on the device). This class is live in the shipped kernels —
    #    ``conductivity.py:307`` spells ``(km1 != 1.0) | (si1 != 1.0)`` and uses it
    #    as a STORE MASK, 48 ``setp.neu.f32`` in ``conductive_pml_curl_step`` — so
    #    a gate that did not test it was not covering the op classes it claimed.
    #    The result is a WORD, not a magnitude, so this case is classified by exact
    #    word equality on every lane rather than by subnormal structure.
    cases.append({
        "op": "cmp_subnormal_against_zero", "kind": "cmp", "governed": True,
        "verdict_mode": "word",
        "a": with_lane0(BITS_2P_M135, lanes),
        "b": np.zeros(lanes, dtype=np.uint32),
        "keep_lane0": BITS_ONE, "flush_lane0": 0x00000000,
        "why": "(subnormal != 0) is 1.0 under keep and 0.0 once the operands are "
               "flushed on read. The host answer rests on DAZ applying to SSE "
               "compares (CMPPS/UCOMISS read their source operands), so a "
               "disagreement in this cell is information about the host, not noise",
    })

    # 10. TRITON ONLY: the associative scan. NO SHIPPED KERNEL USES ``tl.cumsum``
    #    today — the only occurrences in ``triton_kernels/`` are prose in two
    #    comments — so this case is carried as the one SCAN class this engine could
    #    reach for, not as coverage of something in flight. Lane 0 of a scan is a
    #    PASS-THROUGH — the input, with no add performed — so the verdict is taken
    #    from lane 1 onward. Every lane holds 2^-136, so lane k's cumulative sum is
    #    (k+1)*2^-136 and stays subnormal through lane 255 (256*2^-136 = 2^-128).
    cases.append({
        "op": "cumsum_scan", "kind": "cumsum", "governed": True,
        "triton_only": True, "verdict_from": 1, "shipped": False,
        "a": np.full(lanes, BITS_2P_M136, dtype=np.uint32),
        "b": np.full(lanes, BITS_2P_M136, dtype=np.uint32),
        "keep_lane0": BITS_2P_M135, "flush_lane0": 0x00000000,
        "why": "tl.cumsum over 256 lanes of 2^-136; lane 1 is 2*2^-136 = 2^-135 "
               "under keep and zero once the scan's adds flush their inputs. "
               "Lane 0 is excluded because a scan does not add into it.",
    })
    return cases


def reduction_case() -> Dict[str, Any]:
    """One subnormal among 4095 zeros: what survives ``sum`` and ``max``.

    NOT a formality, and not a CUB kernel obeying CuPy's options. Measured: with
    the whole process flushing, ``cp.sum`` over this buffer returned ``0x00004000``
    — CuPy 13.5.1's default reduction accelerator is a CUB binary built when CuPy
    was built, which no compile-time option this policy can reach applies to.
    ``subnormal_policy.install_cupy_policy`` therefore switches that accelerator
    off under ``"flush"``, and THIS CASE IS THE MEASUREMENT that it worked.
    """
    buffer_bits = np.zeros(4096, dtype=np.uint32)
    buffer_bits[137] = BITS_2P_M135
    return {"op": "reduction", "governed": True, "buffer": buffer_bits,
            "keep_word": BITS_2P_M135, "flush_word": 0x00000000,
            "why": "one 2^-135 among 4095 zeros. Under 'keep' every route agrees "
                   "trivially (CUB keeps, and a stripped NVRTC kernel keeps). "
                   "Under 'flush' this cell is the only proof that reductions "
                   "left the CUB path and obey the policy like everything else"}


def verdict(out_bits: np.ndarray, case: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a cell's raw output words as keep / flush / mixed, on the HOST.

    Two modes. The default reads subnormal STRUCTURE: a kept subnormal has a zero
    exponent and a nonzero mantissa, a flushed one is a zero of either sign, and
    the count is over all 256 lanes so one lucky word cannot carry the verdict.
    ``verdict_mode="word"`` compares every lane against the case's two known words
    instead, for a case whose result is a flag rather than a magnitude (the
    comparison case: 1.0 or 0.0, neither of which is subnormal).

    ``verdict_from`` skips leading lanes a case cannot speak for — only the scan
    needs it, because lane 0 of a cumulative sum is a pass-through that performs
    no addition and would therefore report ``mixed`` on a perfectly uniform flush.
    """
    out_bits = out_bits[int(case.get("verdict_from", 0)):]
    lane0 = int(out_bits[0])
    if case.get("verdict_mode") == "word":
        kept = int((out_bits == np.uint32(case["keep_lane0"])).sum())
        zeroed = int((out_bits == np.uint32(case["flush_lane0"])).sum())
    else:
        exponent = (out_bits >> 23) & 0xFF
        mantissa = out_bits & MANT
        kept = int(((exponent == 0) & (mantissa != 0)).sum())
        zeroed = int(((out_bits & ~np.uint32(SIGN)) == 0).sum())
    other = len(out_bits) - kept - zeroed
    if kept == len(out_bits):
        name = "keep"
    elif zeroed == len(out_bits):
        name = "flush"
    else:
        name = "mixed"
    return {
        "verdict": name,
        "lane0": f"0x{lane0:08X}",
        "lane0_matches_keep": lane0 == int(case["keep_lane0"]),
        "lane0_matches_flush": lane0 == int(case["flush_lane0"]),
        "n_kept": kept, "n_zero": zeroed, "n_other": other,
    }


def expected_verdict(case: Dict[str, Any], policy: str, executor: str) -> str:
    """What this case must report on THIS executor under ``policy``.

    Three answers, and the executor axis is not decoration. Transport always
    keeps. A governed op follows the policy — unless the case names the executors
    it is governed on (``governed_on``), which is how a measured executor-level
    exception is expressed: ``-x`` on the host is a bitwise sign flip that x86
    FTZ/DAZ never see, so ``np.negative`` keeps a subnormal in a process where
    every other host op flushes. An earlier revision had no executor axis and
    therefore FAILED a correctly installed flush policy on that one cell.
    """
    if not case["governed"]:
        return "keep"
    governed_on = case.get("governed_on")
    if governed_on is not None and executor not in governed_on:
        return "keep"
    return "flush" if policy == sp.FLUSH else "keep"


def expected_lane0(case: Dict[str, Any], policy: str, executor: str) -> int:
    """The exact word lane 0 must carry on this executor under ``policy``.

    A structural verdict is not enough: Triton's ``-x`` lowers to
    ``sub.rn.f32 0f00000000, %f`` and flushes to ``+0``, while CuPy's lowers to
    ``neg.ftz.f32`` and flushes to ``-0``. Both read "flush"; they are different
    bytes, and telling them apart is the whole reason this gate reports words.
    """
    if expected_verdict(case, policy, executor) == "keep":
        return int(case["keep_lane0"])
    return int(case.get("flush_lane0_by_executor", {}).get(executor, case["flush_lane0"]))


def judge(bits: np.ndarray, case: Dict[str, Any], policy: str, executor: str) -> Dict[str, Any]:
    """One cell: the structural verdict AND lane 0's exact word, both required.

    ``agrees`` is the conjunction. An earlier revision computed the word match and
    then ignored it, so two executors producing different bytes for the same
    operation both passed.
    """
    payload = verdict(bits, case)
    expected = expected_verdict(case, policy, executor)
    word = expected_lane0(case, policy, executor)
    payload["expected"] = expected
    payload["expected_lane0"] = f"0x{word:08X}"
    payload["verdict_agrees"] = payload["verdict"] == expected
    payload["lane0_agrees"] = int(payload["lane0"], 16) == word
    payload["agrees"] = bool(payload["verdict_agrees"] and payload["lane0_agrees"])
    return payload


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


def _f32(u32: np.ndarray) -> np.ndarray:
    return u32.view(np.float32)


def exec_host_numpy(case: Dict[str, Any]) -> Optional[np.ndarray]:
    """The host path. ``fma`` has no NumPy ufunc; that cell is unsupported, by name."""
    a, b = _f32(case["a"]), _f32(case["b"])
    kind = case["kind"]
    if kind in ("fma", "cumsum"):
        return None
    if kind == "mul":
        out = np.multiply(a, b)
    elif kind == "sub":
        out = np.subtract(a, b)
    elif kind == "add":
        out = np.add(a, b)
    elif kind == "neg":
        out = np.negative(a)
    elif kind == "div":
        out = np.divide(a, b)
    elif kind == "select":
        out = np.where(np.ones(len(a), dtype=bool), a, b)
    elif kind == "cmp":
        # The comparison reads its operands, so DAZ decides it; the 1.0/0.0 the
        # result is encoded as are both normal and carry no policy of their own.
        out = np.where(np.not_equal(a, b), np.float32(1), np.float32(0))
    else:  # pragma: no cover - build_cases emits no other kind
        raise ValueError(f"no host executor for kind {kind!r}")
    return np.ascontiguousarray(out, dtype=np.float32).view(np.uint32)


def _to_device(cp, u32: np.ndarray):
    return cp.asarray(u32).view(cp.float32)


def exec_cupy_ufunc(cp, case: Dict[str, Any]) -> Optional[np.ndarray]:
    kind = case["kind"]
    if kind in ("fma", "cumsum"):
        return None
    a, b = _to_device(cp, case["a"]), _to_device(cp, case["b"])
    table = {"mul": cp.multiply, "sub": cp.subtract, "add": cp.add,
             "div": cp.divide}
    if kind in table:
        out = table[kind](a, b)
    elif kind == "neg":
        out = cp.negative(a)
    elif kind == "select":
        out = cp.where(cp.ones(len(case["a"]), dtype=bool), a, b)
    elif kind == "cmp":
        out = cp.where(cp.not_equal(a, b), cp.float32(1), cp.float32(0))
    else:  # pragma: no cover
        raise ValueError(f"no cupy ufunc for kind {kind!r}")
    return out.view(cp.uint32).get()


_ELEMENTWISE: Dict[str, Any] = {}


def exec_cupy_elementwise(cp, case: Dict[str, Any]) -> Optional[np.ndarray]:
    kind = case["kind"]
    expressions = {"mul": "o = a * b", "sub": "o = a - b", "add": "o = a + b",
                   "neg": "o = -a", "div": "o = a / b",
                   "fma": "o = fmaf(a, b, 0.0f)", "select": "o = (i >= 0) ? a : b",
                   "cmp": "o = (a != b) ? 1.0f : 0.0f"}
    if kind not in expressions:
        return None   # cumsum is a scan, not an elementwise op; Triton covers it
    if kind not in _ELEMENTWISE:
        _ELEMENTWISE[kind] = cp.ElementwiseKernel(
            "float32 a, float32 b", "float32 o", expressions[kind], f"conf_ew_{kind}")
    out = _ELEMENTWISE[kind](_to_device(cp, case["a"]), _to_device(cp, case["b"]))
    return out.view(cp.uint32).get()


_RAWKERNELS: Dict[str, Any] = {}


def exec_cupy_rawkernel(cp, case: Dict[str, Any]) -> Optional[np.ndarray]:
    kind = case["kind"]
    if kind == "cumsum":
        return None   # no single-kernel counterpart; the Triton leg carries it
    if kind not in _RAWKERNELS:
        _RAWKERNELS[kind] = cp.RawKernel(RAW_KERNEL_SOURCE, f"dev_{kind}",
                                         options=ENGINE_RAWKERNEL_OPTIONS)
    a, b = _to_device(cp, case["a"]), _to_device(cp, case["b"])
    out = cp.zeros(len(case["a"]), dtype=cp.float32)
    _RAWKERNELS[kind]((1,), (len(case["a"]),), (a, b, out, np.int32(len(case["a"]))))
    cp.cuda.runtime.deviceSynchronize()
    return out.view(cp.uint32).get()


def exec_cupy_reductions(cp, case: Dict[str, Any]) -> Dict[str, str]:
    """``sum`` / ``max`` / ``abs``-then-``max`` over one subnormal among zeros."""
    buffer_ = _to_device(cp, case["buffer"])
    results: Dict[str, str] = {}
    for name, value in (("sum", cp.sum(buffer_)),
                        ("max", cp.max(buffer_)),
                        ("abs_max", cp.max(cp.abs(buffer_)))):
        word = int(value.reshape(1).view(cp.uint32).get()[0])
        results[name] = f"0x{word:08X}"
    return results


# ---- Triton ---------------------------------------------------------------- #

class CupyPointer:
    """The CuPy-to-Triton pointer adapter — same shape as ``launch.CupyPointer``.

    A bare cupy array is rejected by Triton's launcher ("Pointer argument must be
    either uint64 or have data_ptr method"); this renames the address and exposes
    the dtype, and nothing is copied.
    """

    __slots__ = ("array",)

    def __init__(self, array: Any) -> None:
        self.array = array

    def data_ptr(self) -> int:
        return int(self.array.data.ptr)

    @property
    def dtype(self):
        return self.array.dtype


_TRITON_NS: Dict[str, Any] = {}


def triton_namespace() -> Dict[str, Any]:
    """One jitted kernel per f32 op class the shipped kernels use, plus the gap."""
    if _TRITON_NS:
        return _TRITON_NS
    import triton
    import triton.language as tl

    globals()["tl"] = tl  # A jitted body resolves names in the MODULE's globals.

    @triton.jit
    def t_mul(A, B, Out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(Out + offs, tl.load(A + offs, mask=m) * tl.load(B + offs, mask=m), mask=m)

    @triton.jit
    def t_sub(A, B, Out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(Out + offs, tl.load(A + offs, mask=m) - tl.load(B + offs, mask=m), mask=m)

    @triton.jit
    def t_add(A, B, Out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(Out + offs, tl.load(A + offs, mask=m) + tl.load(B + offs, mask=m), mask=m)

    @triton.jit
    def t_neg(A, B, Out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(Out + offs, -tl.load(A + offs, mask=m), mask=m)

    @triton.jit
    def t_fma(A, B, Out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m)
        b = tl.load(B + offs, mask=m)
        tl.store(Out + offs, tl.math.fma(a, b, a * 0.0), mask=m)

    @triton.jit
    def t_div(A, B, Out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m)
        b = tl.load(B + offs, mask=m)
        tl.store(Out + offs, tl.math.div_rn(a, b), mask=m)

    @triton.jit
    def t_select(A, B, Out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m)
        b = tl.load(B + offs, mask=m)
        tl.store(Out + offs, tl.where(offs >= 0, a, b), mask=m)

    @triton.jit
    def t_cmp(A, B, Out, n, BLOCK: tl.constexpr):
        # The shipped spelling: conductivity.py:307 builds a store mask this way.
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m)
        b = tl.load(B + offs, mask=m)
        tl.store(Out + offs, tl.where(a != b, 1.0, 0.0), mask=m)

    @triton.jit
    def t_cumsum(A, B, Out, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m, other=0.0)
        tl.store(Out + offs, tl.cumsum(a, axis=0), mask=m)

    @triton.jit
    def t_plain_div(A, B, Out, n, BLOCK: tl.constexpr):
        # THE KNOWN GAP, deliberately: a plain '/' lowers to div.full.f32 as
        # inline asm, which LLVM never sees. No shipped kernel spells it this way.
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m)
        b = tl.load(B + offs, mask=m)
        tl.store(Out + offs, a / b, mask=m)

    _TRITON_NS.update({"mul": t_mul, "sub": t_sub, "add": t_add, "neg": t_neg,
                       "fma": t_fma, "div": t_div, "select": t_select,
                       "cmp": t_cmp, "cumsum": t_cumsum, "plain_div": t_plain_div})
    return _TRITON_NS


def exec_triton(cp, case: Dict[str, Any], kernel_name: Optional[str] = None
                ) -> Tuple[np.ndarray, str]:
    """Launch one Triton kernel with a cleared cache; return bits and its PTX."""
    from meep_gpu.triton_kernels.kernels import ENABLE_FP_FUSION

    kernel = triton_namespace()[kernel_name or case["kind"]]
    kernel.cache.clear()
    a, b = _to_device(cp, case["a"]), _to_device(cp, case["b"])
    out = cp.zeros(len(case["a"]), dtype=cp.float32)
    kernel[(1,)](CupyPointer(a), CupyPointer(b), CupyPointer(out), len(case["a"]),
                 BLOCK=len(case["a"]), num_warps=4,
                 enable_fp_fusion=ENABLE_FP_FUSION)
    cp.cuda.runtime.deviceSynchronize()
    compiled = next(iter(next(iter(kernel.cache.values())).values()))
    return out.view(cp.uint32).get(), compiled.asm.get("ptx", "")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def save(record: Dict[str, Any], path: str) -> None:
    """Atomic rewrite after every cell: tmp + fsync + os.replace (progress reporting)."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(handle, "w") as sink:
        json.dump(record, sink, indent=2, sort_keys=True, default=str)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(tmp, path)


#: Which policy executors each leg needs installed. ``--legs host`` must not drag
#: in CuPy's cache-directory refusal for a leg that will never run: measured, the
#: gate's own documented laptop invocation exited 2 on a host that merely had CuPy
#: importable, because the install was unconditional.
LEG_EXECUTORS: Dict[str, str] = {
    "host": "host", "cupy_ufunc": "cupy", "cupy_elementwise": "cupy",
    "cupy_rawkernel": "cupy", "cupy_reduction": "cupy", "triton": "triton",
}

#: Which matrix rows each leg is responsible for producing. A leg that ran and
#: produced nothing is a hole, and a green artifact with a hole in it is worse
#: than a red one: measured, this gate returned PASS with 5 of its 6 executors
#: never executed and a stamp asserting they had attained the policy.
LEG_ROWS: Dict[str, str] = {
    "host": "host_numpy", "cupy_ufunc": "cupy_ufunc",
    "cupy_elementwise": "cupy_elementwise", "cupy_rawkernel": "cupy_rawkernel",
    "cupy_reduction": "cupy_reduction", "triton": "triton",
}


def run(policy: str, out_path: str, legs: Tuple[str, ...]) -> int:
    started = time.time()
    cases = build_cases()
    record: Dict[str, Any] = {
        "gate": "subnormal_policy_conformance",
        "requested_policy": policy,
        "started_utc": now(),
        "legs_requested": list(legs),
        "machine": os.uname().machine if hasattr(os, "uname") else "unknown",
        "cases": [{"op": c["op"], "kind": c["kind"], "governed": c["governed"],
                   "governed_on": list(c.get("governed_on") or ()) or "every executor",
                   "verdict_mode": c.get("verdict_mode", "subnormal_structure"),
                   "shipped": c.get("shipped", True),
                   "keep_lane0": f"0x{int(c['keep_lane0']):08X}",
                   "flush_lane0": f"0x{int(c['flush_lane0']):08X}",
                   "flush_lane0_by_executor": {
                       name: f"0x{int(word):08X}"
                       for name, word in (c.get("flush_lane0_by_executor") or {}).items()},
                   "why": c["why"]} for c in cases],
        "matrix": {},
        "skipped": {},
        "failures": [],
        "known_gap": {},
    }
    save(record, out_path)

    def cell(executor: str, op: str, payload: Dict[str, Any]) -> None:
        record["matrix"].setdefault(executor, {})[op] = payload
        if payload.get("agrees") is False:
            if payload.get("verdict_agrees") is False or "verdict_agrees" not in payload:
                record["failures"].append(
                    f"{executor}/{op}: {payload.get('verdict')} but "
                    f"{payload.get('expected')} was requested "
                    f"(lane0 {payload.get('lane0')}, expected "
                    f"{payload.get('expected_lane0')})")
            else:
                record["failures"].append(
                    f"{executor}/{op}: the verdict is {payload.get('verdict')} as "
                    f"requested but lane 0 is {payload.get('lane0')} where "
                    f"{payload.get('expected_lane0')} is the word this executor "
                    f"must produce — same class, different bytes")
        save(record, out_path)
        log(f"[cell] {executor:<20s} {op:<28s} verdict={payload.get('verdict'):<9s} "
            f"expected={payload.get('expected'):<9s} "
            f"{'OK' if payload.get('agrees') else 'DISAGREES'} "
            f"lane0={payload.get('lane0')} want={payload.get('expected_lane0')}")

    # ---- install the policy first; every executor below runs under it --------
    # Only the executors this run will actually measure: installing CuPy's half
    # for a --legs host run refuses on the cache-directory convention, for a leg
    # that was never requested.
    executors = tuple(sorted({LEG_EXECUTORS[leg] for leg in legs}))
    record["executors_installed"] = list(executors)
    try:
        stamp = sp.install_subnormal_policy(policy, strict=True, executors=executors)
        record["policy_stamp"] = stamp
    except (sp.SubnormalPolicyUnattainable, sp.SubnormalPolicyLocked) as exc:
        record["policy_stamp"] = sp.policy_stamp()
        record["failures"].append(f"install: {exc}")
        record["verdict"] = "REFUSED"
        record["finished_utc"] = now()
        save(record, out_path)
        log(f"[install] REFUSED: {exc}")
        return 2
    log(f"[install] policy={policy} stamp={stamp['policy']} "
        f"unattained={stamp['unattained']}")

    # ---- host NumPy ---------------------------------------------------------
    if "host" in legs:
        for case in cases:
            bits = None if case.get("triton_only") else exec_host_numpy(case)
            if bits is None:
                record["skipped"][f"host_numpy/{case['op']}"] = (
                    "this op class has no single NumPy operation (fma, cumsum); "
                    "the device legs carry it")
                save(record, out_path)
                log(f"[skip] host_numpy {case['op']}: no single NumPy operation")
                continue
            cell("host_numpy", case["op"], judge(bits, case, policy, "host_numpy"))

    # ---- CuPy ---------------------------------------------------------------
    cp = None
    if any(leg.startswith("cupy") for leg in legs):
        try:
            import cupy as cp_module
            if int(cp_module.cuda.runtime.getDeviceCount()) < 1:
                raise RuntimeError("no CUDA device visible")
            cp = cp_module
        except Exception as exc:
            record["skipped"]["cupy"] = f"CuPy unavailable: {exc}"
            save(record, out_path)
            log(f"[skip] cupy: {exc}")

    if cp is not None:
        executors = (("cupy_ufunc", exec_cupy_ufunc),
                     ("cupy_elementwise", exec_cupy_elementwise),
                     ("cupy_rawkernel", exec_cupy_rawkernel))
        for name, function in executors:
            if name not in legs:
                continue
            for case in cases:
                bits = None if case.get("triton_only") else function(cp, case)
                if bits is None:
                    record["skipped"][f"{name}/{case['op']}"] = "no such op on this executor"
                    save(record, out_path)
                    continue
                payload = judge(bits, case, policy, name)
                if name == "cupy_rawkernel":
                    payload["options"] = list(ENGINE_RAWKERNEL_OPTIONS)
                cell(name, case["op"], payload)

        if "cupy_reduction" in legs:
            reduction = reduction_case()
            results = exec_cupy_reductions(cp, reduction)
            expected_word = (f"0x{reduction['flush_word']:08X}" if policy == sp.FLUSH
                             else f"0x{reduction['keep_word']:08X}")
            for name, word in results.items():
                payload = {"verdict": "flush" if int(word, 16) == 0 else "keep",
                           "lane0": word, "expected_lane0": expected_word,
                           "expected": sp.FLUSH if policy == sp.FLUSH else sp.KEEP,
                           "agrees": word == expected_word}
                cell("cupy_reduction", f"reduce_{name}", payload)
            record["reduction_accelerators"] = sp.executor_report("cupy").get(
                "reduction_accelerators")
            save(record, out_path)

    # ---- Triton -------------------------------------------------------------
    # NOT nested inside "did CuPy import": a Triton leg that never ran because
    # CuPy was missing used to be recorded NOWHERE, and the artifact still read
    # PASS. The reason is written down whichever half is absent.
    if "triton" in legs:
        available = cp is not None
        if cp is None:
            record["skipped"]["triton"] = (
                "the Triton leg needs CuPy for device buffers, and CuPy is "
                f"unavailable: {record['skipped'].get('cupy', 'not imported')}")
        else:
            try:
                import triton  # noqa: F401
            except Exception as exc:
                available = False
                record["skipped"]["triton"] = f"Triton unavailable: {exc}"
        if not available:
            save(record, out_path)
            log(f"[skip] triton: {record['skipped']['triton']}")
        if available:
            for case in cases:
                # A REFUSAL IS A RESULT, not a crash. Under 'flush' the PTX
                # auditor raises out of Triton's compiler for any audited f32
                # instruction that lacks .ftz, and an uncaught raise here would
                # end the run with a traceback: no verdict, no matrix, and the
                # one cell that carries the information silently missing. It is
                # recorded as a cell that does not agree — the op class this
                # policy could not deliver, named — and the remaining classes
                # are still measured.
                try:
                    bits, ptx = exec_triton(cp, case)
                except sp.SubnormalPolicyUnattainable as exc:
                    cell("triton", case["op"], {
                        "verdict": "refused", "lane0": "n/a",
                        "expected": expected_verdict(case, policy, "triton"),
                        "expected_lane0": f"0x{expected_lane0(case, policy, 'triton'):08X}",
                        "verdict_agrees": False, "lane0_agrees": False,
                        "agrees": False,
                        "refusal": str(exc)[:800],
                        "note": "the PTX auditor refused this compile, so no bits "
                                "exist to judge; the policy is not attained for "
                                "this op class on this executor"})
                    continue
                payload = judge(bits, case, policy, "triton")
                payload["ptx_audit"] = {k: v for k, v in
                                        sp.audit_ptx(ptx, policy).items()
                                        if k != "violations"}
                cell("triton", case["op"], payload)

            # The known gap, as its own recorded leg.
            gap_case = dict(cases[0])
            gap_case["kind"] = "plain_div"
            gap_case["a"] = np.full(N_LANES, BITS_2P_M135, dtype=np.uint32)
            gap_case["b"] = np.full(N_LANES, BITS_ONE, dtype=np.uint32)
            try:
                bits, ptx = exec_triton(cp, gap_case, kernel_name="plain_div")
                audit = sp.audit_ptx(ptx, policy)
                record["known_gap"] = {
                    "refused": False,
                    "verdict": verdict(bits, gap_case)["verdict"],
                    "audited": audit["audited"],
                    "missing_ftz": audit["missing_ftz"],
                    "note": ("a plain '/' compiled without being refused: under "
                             "'flush' this means the auditor did not see "
                             "div.full.f32 and the gap is OPEN"
                             if policy == sp.FLUSH else
                             "a plain '/' compiled, which is the expected outcome "
                             "under 'keep' — there is nothing for the auditor to "
                             "refuse here; the gap is a flush-policy question"),
                }
                if policy == sp.FLUSH:
                    record["failures"].append(
                        "known_gap: a plain float '/' compiled under 'flush' "
                        "without the PTX auditor refusing it")
            except sp.SubnormalPolicyUnattainable as exc:
                record["known_gap"] = {"refused": True, "refusal": str(exc)[:600],
                                       "note": "the auditor refused the compile, "
                                               "which is the designed behaviour "
                                               "under 'flush'"}
            save(record, out_path)
            log(f"[gap] plain '/' refused={record['known_gap'].get('refused')}")
            record["triton_counters"] = sp.triton_counters()

    # ---- every requested leg must have produced cells -----------------------
    # A green artifact with missing rows is the worst output this gate can make:
    # it reads as "every executor obeys the policy" while most of them were never
    # asked. An unrun leg is INCOMPLETE, which is not PASS.
    empty = [leg for leg in legs if not record["matrix"].get(LEG_ROWS[leg])]
    record["legs_without_cells"] = empty
    for leg in empty:
        record["failures"].append(
            f"leg {leg!r} was requested and produced no cells "
            f"({record['skipped'].get(LEG_ROWS[leg]) or record['skipped'].get(leg) or 'no reason recorded'})")

    record["policy_stamp"] = sp.policy_stamp()
    record["elapsed_s"] = round(time.time() - started, 3)
    record["finished_utc"] = now()
    record["verdict"] = ("PASS" if not record["failures"]
                         else ("INCOMPLETE" if empty and len(record["failures"]) == len(empty)
                               else "FAIL"))
    save(record, out_path)

    log("")
    log(f"=== executor x op matrix (requested policy: {policy}) ===")
    ops = [c["op"] for c in cases]
    for executor in sorted(record["matrix"]):
        row = record["matrix"][executor]
        cells = " ".join(f"{op}={row[op]['verdict']}" for op in ops if op in row)
        extra = " ".join(f"{k}={v['verdict']}" for k, v in row.items() if k not in ops)
        log(f"  {executor:<20s} {cells} {extra}")
    for name, reason in sorted(record["skipped"].items()):
        log(f"  SKIPPED {name}: {reason}")
    for failure in record["failures"]:
        log(f"  FAILURE {failure}")
    log(f"=== {record['verdict']} in {record['elapsed_s']} s -> {out_path}")
    return 0 if record["verdict"] == "PASS" else (3 if record["verdict"] == "INCOMPLETE" else 1)


ALL_LEGS: Tuple[str, ...] = ("host", "cupy_ufunc", "cupy_elementwise",
                             "cupy_rawkernel", "cupy_reduction", "triton")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, choices=list(sp.POLICIES),
                        help="the policy every executor must be shown to obey")
    parser.add_argument("--out", required=True, help="JSON artifact path")
    parser.add_argument("--legs", default=",".join(ALL_LEGS),
                        help="comma-separated subset of " + ",".join(ALL_LEGS))
    args = parser.parse_args(argv)
    legs = tuple(leg.strip() for leg in args.legs.split(",") if leg.strip())
    unknown = [leg for leg in legs if leg not in ALL_LEGS]
    if unknown:
        parser.error(f"unknown legs {unknown}; choose from {list(ALL_LEGS)}")
    return run(args.policy, args.out, legs)


if __name__ == "__main__":
    raise SystemExit(main())
