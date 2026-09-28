"""BREADTH: every row mask, and the expansion arm no device leg has ever compiled.

WHAT THIS IS AND WHAT THE 26 GATES ARE. The gates establish byte identity against
``stepping``'s array path at one launch and at 60, under both float32 subnormal
policies, over uniform and subnormal-band value classes and both courants, with
every mutation scored CAUGHT or NULL CONFIRMED. What they do NOT establish is
stated in their own records, and two of those sentences are this file's whole
subject. From ``complex_offdiag_update_e.COMPLEX_OFFDIAG_UPDATE_E_ADMISSION``::

    "row_masks_exercised": ((1,1,1,1,1,1), (1,0,0,1,0,0), (1,1,0,0,0,0),
                            (0,0,1,0,0,0)),
    "expansion_arms_exercised": ("FMA_V1",),
    ... a verdict about the EMITTER at the corpus digest below, exercised on FOUR
    of its 63 row masks per tail and on ONE of its two expansion arms. A mask or
    an arm outside those is covered by the emitter's own tests and by that digest,
    not by a byte gate.

    "what_it_does_not_license": ... "the NAIVE expansion arm, which the emitter
    can still emit and no device leg has scored."

So: 4 of 63 masks per tail, and 1 of 2 arms. This file sweeps the other 59 and
launches the other arm.

=============================================================================
THE TWO BACKENDS, AND WHY THE NumPy ONE CANNOT RELEASE
=============================================================================

``--backend numpy`` (the default, and the only one run so far) is not a weaker
byte comparison; it is a DIFFERENT MEASUREMENT and the artifact says so in every
row. It runs the emitted device text through
``test_complex_offdiag_update_e.evaluate_source`` -- the merge bar's
TEXT-ANCHORED complex128 evaluator, imported rather than re-written -- against
``stepping.update_E`` on real ``Grid``/``Fields``/``PML`` objects, and scores a
RELATIVE deviation at the float32 input floor. That settles the whole silent
structural class per mask (a partner read at the wrong offset, a ghost taken in
the wrong direction, a parity sign, an unconjugated phase, a mask on the wrong
plane) and settles NOTHING about bytes: complex128 expresses neither the zero
cross terms nor the fused arm. ``release`` is therefore False on this backend
unconditionally, with the reason carried in the verdict rather than implied.

``--backend cuda`` swaps ONE object -- the comparator -- and the fixtures, the
seeds, the case keys, the floors, the plan and the artifact schema are the same
bytes of code. The device leg is a flag flip, and the launch itself is
``gate_cuda_complex_offdiag_update_e.KernelBackend.launch``, subclassed rather
than respelled, so this file adds no second spelling of how the family is
launched.

WHY THE EVALUATOR IS IMPORTED AND NOT COPIED. It is anchored: its arithmetic
shape comes from PINNED substrings of the shipped prelude
(``test_complex_offdiag_update_e.ANCHORS``) and every index expression, per-term
argument, accumulator start and final combination is PARSED out of the emitted
text. A second transcription here would be a second thing to keep in step, and
that module's own record says what that costs -- three planted assembly defects
once left 86 of 87 tests passing because the evaluator MIRRORED the emitter
instead of executing it.

=============================================================================
THE SAMPLING, DECLARED RATHER THAN PERFORMED QUIETLY
=============================================================================

63 masks x 2 tails x 2 arms x 2 policies x 23 specs x 2 courants x 3 value
classes is 69,552 cells. That product is not affordable on a contended box and a
sweep that silently samples it is worse than one that says where it looked. The
split taken here, and the reason for each:

  LEG ``rowmask``   ALL 63 masks x BOTH tails, at ONE arm (FMA_V1, the licensed
                    one), ONE value class (uniform), ONE process policy, and a
                    NAMED six-spec subset covering a fold on each axis, both
                    declared terminations, both plane parities, a Bloch phase and
                    a metallic wall. The mask axis is COMPLETE; everything else is
                    held at the point the gates already measured, so a divergence
                    here is attributable to the mask.
  LEG ``corner``    The FOUR masks the device already exercised, at BOTH arms and
                    the FULL spec list. The arm axis is COMPLETE at exactly the
                    masks whose FMA_V1 answer is already on record, so the NAIVE
                    number has a same-fixture FMA_V1 number beside it.
  LEG ``naive``     The arm question itself, in three parts: the TEXT (is the arm
                    confined to the prelude block, on all 63 x 2 sources), the
                    HELPERS (each arm's three multiply bodies, parsed out of the
                    shipped text and executed in float32 against the array path's
                    own complex64 expression for that site), and on a device the
                    whole kernel under BOTH ``--fmad`` option tuples.
  LEG ``falsify``   Planted defects, each required to flip the leg that owns it.

``coverage_declaration`` in the artifact carries the cell counts both ways --
swept and not swept -- computed from the plan that ran rather than typed.

=============================================================================
THE LICENSING SUBTLETY, AND WHICH SIDE OF IT THIS FILE IS ON
=============================================================================

The expansion licence binds an ARM. The shipped probe artifact
(``results/expansion_probe_2026-08-17/``) licenses FMA_V1 under both policies,
and ``covers_complex_offdiag_pml_update_e`` takes that verdict and checks its
shape -- so THE PLANNER WILL NEVER PLAN NAIVE, and that refusal is correct.

This file does not ask the planner. It calls the launcher directly with
``expansion="NAIVE"``, which is a MEASUREMENT of an arm the emitter can emit and
no device has scored -- not a dispatch of it. Every NAIVE row carries
``planner_bypassed: True`` and the predicate's verdict for the same fixture is
recorded beside it, so the artifact cannot be read as licensing the arm. Nothing
in ``meep_gpu/`` imports ``cuda_kernels`` at all, so no production step is
reachable from here in either case.

=============================================================================
WHAT WAS MEASURED ON THE LAPTOP AND WHY IT SHARPENS THE DEVICE LEG
=============================================================================

``numpy`` 2.4.3 on this arm64 host computes complex64 ``a * b`` BIT-IDENTICALLY
to the FMA_V1 spelling (0 of 400,000 words differ over a uniform draw) and NOT to
the NAIVE spelling (52,425 of 200,000 real words differ) nor to a
double-then-round evaluation (97,094 of 400,000). That is a fact about THIS
host's contracted numpy build, and it is recorded here as a second draw of the
same classification the device probe reached for CuPy -- not as evidence about
the GPU host, whose numpy and whose CuPy are different builds on a different ISA.

THAT PREDICTION WAS WRONG IN ONE PLACE AND THE LEG IS WHAT SAID SO. Reasoning
from the text alone gives: the two arms differ only in the three multiply
helpers, and of those only ``rotate_field_left`` is ever handed two genuinely
complex operands, so the zero-cross-term helpers must be exact on both arms and
NAIVE should diverge only on phased wrap lanes. MEASURED, over 20,000 operands
per row (``naive.helpers`` in the artifact), that holds on a uniform draw and
FAILS in the subnormal band:

    (50,000 operands per row, so 100,000 float32 words compared per row)

    FMA_V1  all three helpers, all four value classes    0 / 100,000 words
    NAIVE   rotate_field_left, uniform               26,267 / 100,000
    NAIVE   rotate_field_left, subnormal band        24,094 / 100,000
    NAIVE   rotate_field_left, subnormal x uniform    7,709 / 100,000
    NAIVE   rotate_field_left, signed zero            1,705 / 100,000
    NAIVE   mul_field_left, uniform                        0 / 100,000
    NAIVE   mul_field_left, SUBNORMAL BAND            12,008 / 100,000
    NAIVE   mul_field_left, subnormal x uniform          335 / 100,000
    NAIVE   mul_coefficient_left, SUBNORMAL BAND      24,109 / 100,000
    NAIVE   mul_coefficient_left, subnormal x uniform    705 / 100,000

The mechanism, read off the differing operands rather than guessed: the real
product UNDERFLOWS to ``-0.0f`` and the exact-zero cross term is also ``-0.0f``,
so NAIVE computes ``(-0.0f) - (-0.0f) = +0.0f`` while the fused form computes
``fmaf(z.re, c, +0.0f)`` over the still-nonzero exact product and rounds to
``-0.0f``. It is the SIGN OF AN UNDERFLOWING ZERO, which is the same hazard
``offdiag_emitter``'s own docstring names for the zero-padding fold -- and this
family's note 1 says a ``-0.0f`` total is reachable.

So the device leg's expectation is sharper AND different from the one that would
have been written from the text: NAIVE should reproduce FMA_V1 on unphased,
non-subnormal cases and diverge both on phased wrap lanes and anywhere the row
product underflows. A device run that finds NAIVE identical on a subnormal-band
case has found something.

RUNNING IT
==========
Laptop, no GPU, no device claim::

    PYTHONPATH=. python -u \\
        parity/meep_gpu/stress_cuda_breadth.py --backend numpy \\
        --out parity/meep_gpu/results/stress_breadth_<date>/numpy/stress.json

Device (ONE verified-idle GPU; the cache dir MUST carry the policy token because
CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/keep \\
        python -u parity/meep_gpu/stress_cuda_breadth.py --backend cuda \\
        --subnormal-policy keep --out $OUT/keep/stress.json

SHRUNK, for a contended box -- one GPU, well under an hour, and it keeps the
mask axis COMPLETE because that is the axis this leg exists for; what it cuts is
the specs, the second courant and the falsify population::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/keep \\
        python -u parity/meep_gpu/stress_cuda_breadth.py --backend cuda \\
        --product reduced --legs rowmask,naive,falsify \\
        --subnormal-policy keep --out $OUT/keep/stress.json

WHAT IT COSTS, with the basis rather than a guess. NVRTC dominates: the full plan
compiles about 429 distinct sources (126 for the 63x2 mask sweep, 32 for the
corner, 268 for the two emitter plants across two mask populations, 3 for the
body plants) and runs about 1,760 cases whose per-case time the sibling gate
measured at 20 ms. That gate's own artifact
(``results/cuda_complex_offdiag_update_e_2026-08-20/keep/gate.json``) puts 1,104
sweep cases plus 64 multi-step cases plus a 40-leg mutation battery at 346 s wall
with 8 sweep sources and ~40 mutation recompiles, which prices a compile of this
family's source at roughly 5-6 s. So the full plan is ~40 min of compiles plus
~1 min of cases: CALL IT 45 MINUTES ON ONE GPU FOR ONE POLICY, and twice that for
both (two processes, two cache dirs). The shrunk plan compiles ~241 sources:
~25 minutes. No throughput claim is made or possible either way -- these are
scheduling numbers, not measurements of the kernel.

WHAT THIS FILE DOES NOT ESTABLISH, restated in the artifact's verdict per run:
bytes on any mask or arm until ``--backend cuda`` has run; the NAIVE arm as a
LICENSED arm (the licence binds FMA_V1 and every NAIVE row here bypasses the
planner deliberately); the split-field TAIL on the numpy backend (the anchored
evaluator returns the constitutive value and never runs the recurrence, which is
the certified sibling's and the device gate's); anything at long horizon (one
launch per case) or at corpus scale (the largest fixture here is a few thousand
cells); and determinism, since every case is a single comparison against the
oracle and nothing is run twice.

One flushed line per case (the progress-reporting rule); every case is appended to
``cases.jsonl`` beside the artifact as it lands, and the artifact itself is
rewritten atomically every ``--flush-every`` cases and at every leg boundary, so
an interrupted run keeps everything up to the failure. Correctness only -- no
throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:  # the laptop can still import this file and run every leg
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402
# THE DEVICE GATE, IMPORTED FOR ITS MACHINE. Its fixture builder, its floors, its
# spec table, its mask table, its digest-seeded case keys, its mutation battery
# and its launcher are the things a stress leg must not respell: two spellings of
# "how this family is launched" is how a stress run ends up measuring a kernel the
# gate never compiled. It imports cleanly without CuPy.
import gate_cuda_complex_offdiag_update_e as gate  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter  # noqa: E402
from meep_gpu.cuda_kernels import complex_offdiag_update_e as family  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402
# THE LAPTOP EVALUATOR AND ITS ANCHORS, imported rather than re-written -- the
# precedent is gate_cuda_folded_offdiag_rowmask.py, which imports
# test_offdiag_constitutive_pml_real for exactly this reason.
from meep_gpu.cuda_kernels import (  # noqa: E402
    test_complex_offdiag_update_e as bench)

log = probe.log
to_host = probe.to_host
bit_compare = probe.bit_compare
combine = probe.combine

#: The gate's seed, deliberately. A stress leg drawing a different fixture from
#: the gate for the same case key would produce numbers nobody could set beside
#: the gate's.
SEED = gate.SEED

#: The structural bar the merge-bar leg already holds every emitted source to
#: (``test_structure_matches_the_oracle``). Restated rather than imported because
#: it is an assertion literal there; a test pins the two equal.
STRUCTURAL_FLOOR = 3e-6

#: Every row mask the emitter can emit: the 63 non-empty subsets of the six slots.
ALL_ROW_MASKS: Tuple[Tuple[int, ...], ...] = family.LIVE_ROW_MASKS

#: The four the device has actually launched, read out of the family's own
#: admission record rather than typed, so a record that moves moves this too.
EXERCISED_ROW_MASKS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(int(flag) for flag in mask)
    for mask in family.COMPLEX_OFFDIAG_UPDATE_E_ADMISSION["row_masks_exercised"])

#: The arms the record says a device has scored, same source, same reason.
EXERCISED_ARMS: Tuple[str, ...] = tuple(
    family.COMPLEX_OFFDIAG_UPDATE_E_ADMISSION["expansion_arms_exercised"])

#: Both arms the emitter can emit. ``NAIVE`` is the one this file exists to reach.
ARMS: Tuple[str, ...] = tuple(sorted(complex_emitter.EXPANSIONS))

#: The two tails, the family's own order.
TAILS: Tuple[str, ...] = family.ARMS

#: The spec subset the 63-mask sweep runs on, NAMED rather than sliced. A head
#: slice of ``gate.SPECS`` is all X folds and no store arm at all. Chosen so that
#: across the six: every axis is folded somewhere, both declared terminations
#: appear on a fold, both plane parities appear, a Bloch phase appears beside a
#: fold and on its own, a metallic wall appears (which no folded axis can carry --
#: a folded axis always reports ``wm = 0``), and both tails appear.
BREADTH_SPEC_LABELS: Tuple[str, ...] = (
    "pml_fold_X_metallic",            # fold X, metallic termination, even plane
    "pml_fold_Y_phased_X",            # fold Y, periodic termination, phase on X
    "pml_fold_Z_periodic_odd_plane",  # fold Z, odd plane parity
    "pml_unfolded_walls",             # no fold: the wall mask can bite
    "store_3d_phased",                # the store tail, phased on all three axes
    "store_metallic_walls",           # the store tail, walls and one phase
)

#: Courants swept. 0.5 is exactly representable in float32 and 0.35 is not; only
#: the second can distinguish a contracted expression from an uncontracted one,
#: and the courant also moves dt and therefore every PML coefficient, so it is a
#: real second draw of the tables.
COURANTS: Tuple[float, ...] = gate.COURANTS

#: The compile-option tuples. On this family the guard is NOT correctness -- the
#: gate measured its unguarded control 276/276 identical at the inexact courant
#: and NVRTC emitting identical PTX for both tuples on the LICENSED arm -- but it
#: IS expected to matter on NAIVE, whose PTX the same gate measured DISTINCT
#: between the tuples. That asymmetry is the whole reason the NAIVE arm's
#: contraction question is open where FMA_V1's is settled.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = tuple(
    (label, options) for label, options, _ in gate.GUARD_SETS)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# float32 arithmetic the way the device spells it
# ---------------------------------------------------------------------------
#
# ``__fmaf_rn`` is a single rounding. Emulating it as float64 multiply-add then a
# cast is DOUBLE ROUNDING and differs from the real thing on some operands, so the
# platform's own ``fmaf`` is called instead, through ctypes. Measured on this host
# over a uniform draw the two agreed 50,000/50,000 -- which is a reason to record
# the agreement, not a reason to trust the shortcut on a subnormal draw.

_LIBM = ctypes.CDLL(None)
_FMAF = _LIBM.fmaf
_FMAF.restype = ctypes.c_float
_FMAF.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_FMAF_VECTOR = np.frompyfunc(_FMAF, 3, 1)

F32_ZERO = np.float32(0.0)
F32_ONE = np.float32(1.0)


def fmaf(a: Any, b: Any, c: Any) -> np.ndarray:
    """``__fmaf_rn`` -- ONE rounding, from libm rather than from a float64 detour."""
    a32, b32, c32 = (np.asarray(v, dtype=np.float32) for v in (a, b, c))
    a32, b32, c32 = np.broadcast_arrays(a32, b32, c32)
    return _FMAF_VECTOR(a32, b32, c32).astype(np.float32)


class Cf:
    """A complex64 value as the device sees it: two independent float32 planes.

    NEVER ``re + 1j*im``. ``1j * im`` carries a real part of ``0.0 * im``, so the
    sum computes ``(-0.0) + (+0.0) = +0.0`` and destroys every negative zero -- in
    a leg whose entire subject is how the two arms spell their zero cross terms.
    """

    __slots__ = ("re", "im")

    def __init__(self, re: Any, im: Any) -> None:
        self.re = np.asarray(re, dtype=np.float32)
        self.im = np.asarray(im, dtype=np.float32)

    def as_complex64(self) -> np.ndarray:
        out = np.empty(np.shape(self.re), dtype=np.complex64)
        out.real = self.re
        out.imag = self.im
        return out


# ---------------------------------------------------------------------------
# THE ARM, READ OUT OF THE SHIPPED TEXT
# ---------------------------------------------------------------------------
#
# The three multiply helpers are PARSED from the emitted source and their two
# assignment expressions EXECUTED, so an edit to an arm is run by this leg rather
# than mirrored by a second copy of it that someone edited alongside. That is the
# one failure mode a hand transcription has, and this package has already paid for
# it once (see test_complex_offdiag_update_e's _GS/_TOTAL_INIT/_SRC note).

_HELPER_BODY = re.compile(
    r"__device__ __forceinline__ cf (?P<name>\w+)\((?P<params>[^)]*)\) \{\n"
    r"    cf o;\n"
    r"    o\.re = (?P<re>.+);\n"
    r"    o\.im = (?P<im>.+);\n"
    r"    return o;\n\}")

#: The three orientations, each with the ARRAY-PATH expression it transcribes and
#: the ``stepping`` line it comes from. The reference is numpy's own complex64
#: arithmetic on the same operands -- that IS the array path at that site, not a
#: model of it.
HELPER_SITES: Tuple[Dict[str, Any], ...] = (
    {"helper": "rotate_field_left", "operands": ("cf", "cf"),
     "site": "shifted[plane] *= shifted.dtype.type(phase) (stepping.py:1862)",  # stepping.py live lines for the frozen device-text citation(s) in this string: 1862->1909
     "why_it_can_differ": "two genuinely complex operands: the only site where "
                          "the arms' association is visible at all"},
    {"helper": "mul_field_left", "operands": ("cf", "float"),
     "site": "source * inverse_epsilon_for(c) (stepping.py:976), pair * "  # stepping.py live lines for the frozen device-text citation(s) in this string: 976->1005
             "coefficient (:1246)",
     "why_it_can_differ": "the cross terms are exact zeros, so both arms are "
                          "expected to be exact"},
    {"helper": "mul_coefficient_left", "operands": ("float", "cf"),
     "site": "kps*fw and kms*fw_prev (stepping.py:2086-2087), 0.25*(...) "  # stepping.py live lines for the frozen device-text citation(s) in this string: 2086-2087->2133-2134
             "(:1248), parity*_mirror_source(...) (:1873-1875)",
     "why_it_can_differ": "the cross terms are exact zeros, so both arms are "
                          "expected to be exact"},
)


def parse_helper(source: str, name: str) -> Dict[str, str]:
    """One multiply helper's two assignment expressions, out of the emitted text."""
    for match in _HELPER_BODY.finditer(source):
        if match["name"] == name:
            return {"params": match["params"], "re": match["re"],
                    "im": match["im"]}
    raise AssertionError(
        f"{name} is not in the emitted source in the shape this leg executes; "
        f"the arm block has been rewritten and this leg would otherwise measure "
        f"a body the kernel does not have")


def _to_python(expression: str) -> str:
    """The C expression as python, with the two float32 literals and the fma named.

    The ONLY rewrites are literal spellings and the fma call. Nothing reassociates
    and nothing is simplified: ``(g.im * p.im) * -1.0f`` stays two operations in
    that order, because that order is the transcription under test.
    """
    text = expression.replace("0.0f", "F32_ZERO").replace("1.0f", "F32_ONE")
    return text.replace("__fmaf_rn(", "fmaf(")


def evaluate_helper(body: Dict[str, str], operands: Dict[str, Any]) -> Cf:
    """Execute one parsed helper on float32 operands, in the arm's own spelling."""
    environment = dict(operands)
    environment.update({"F32_ZERO": F32_ZERO, "F32_ONE": F32_ONE, "fmaf": fmaf})
    re_value = eval(_to_python(body["re"]), {"__builtins__": {}}, environment)
    im_value = eval(_to_python(body["im"]), {"__builtins__": {}}, environment)
    return Cf(re_value, im_value)


#: The value classes the helper leg draws, and what each one is FOR.
#:
#: ``subnormal_band`` was the obvious choice and is measurably the WRONG one on
#: its own: every product of two band values underflows, so the whole reference is
#: zero-magnitude and what the comparison then distinguishes is the SIGN OF THE
#: ZERO and nothing else. That is a real question here -- this family's note 1 says
#: a ``-0.0f`` total is reachable and the two arms measurably disagree about it --
#: but it is not the rounding question, so the mixed class was added to ask that
#: one: one operand in the band and one ordinary puts the exact products IN and
#: AROUND the band, where a multiply rounds and an fma does not.
HELPER_VALUE_CLASSES: Tuple[str, ...] = (
    "uniform", "subnormal_band", "subnormal_times_uniform", "signed_zero")


def helper_operands(kind: Tuple[str, str], value_class: str, size: int,
                    rng) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """One draw of a helper's operands, plus the two array-path operands.

    Returns ``(environment, left_reference, right_reference)`` where the two
    references are the complex64 / float32 arrays the ARRAY PATH would multiply --
    the same numbers, in the form numpy's own operator takes.
    """
    def plane(cls: str) -> np.ndarray:
        if cls == "uniform":
            return rng.uniform(-1.0, 1.0, size=size).astype(np.float32)
        if cls == "subnormal_band":
            drawn = probe.subnormal_band_hosts(("v",), (size, 1, 1), rng)["v"]
            return np.asarray(drawn, dtype=np.float32).reshape(size)
        if cls == "signed_zero":
            values = rng.uniform(-1.0, 1.0, size=size)
            zeroed = rng.integers(0, 2, size=size) == 0
            signs = np.where(rng.integers(0, 2, size=size) == 0, -0.0, 0.0)
            return np.asarray(np.where(zeroed, signs, values), dtype=np.float32)
        raise ValueError(f"unknown value class {cls!r}")

    if value_class == "subnormal_times_uniform":
        left_class, right_class = "subnormal_band", "uniform"
    else:
        left_class = right_class = value_class

    if kind == ("cf", "cf"):
        left = Cf(plane(left_class), plane(left_class))
        right = Cf(plane(right_class), plane(right_class))
        return ({"g": left, "p": right},
                left.as_complex64(), right.as_complex64())
    if kind == ("cf", "float"):
        left = Cf(plane(left_class), plane(left_class))
        coefficient = plane(right_class)
        return {"z": left, "c": coefficient}, left.as_complex64(), coefficient
    if kind == ("float", "cf"):
        coefficient = plane(left_class)
        right = Cf(plane(right_class), plane(right_class))
        return {"c": coefficient, "z": right}, coefficient, right.as_complex64()
    raise ValueError(f"unknown operand kind {kind!r}")


def array_path_product(kind: Tuple[str, str], left: Any, right: Any) -> np.ndarray:
    """What ``numpy`` itself computes at that site, on those exact operands.

    THE REFERENCE IS THE OPERATOR, NOT A MODEL OF IT. Every real-coefficient
    multiply on the array path is a FULL complex multiply with a zero-imaginary
    operand, because numpy exposes no scalar-times-complex loop -- so the reference
    for all three orientations is one binary ``*`` with the operands in the
    array path's own order, which is what makes the orientation normative.
    """
    return left * right


# ---------------------------------------------------------------------------
# THE PLANTED DEFECTS
# ---------------------------------------------------------------------------
#
# A stress leg nobody has seen fail measures nothing. Each entry names the leg it
# must move and whether it must be CAUGHT or is a declared NULL, and the falsify
# leg refuses a run where any of them scores otherwise.
#
# ``ez_offset1_reads_the_offset0_partner`` IS THE BREADTH DEMONSTRATION and is
# built to be invisible to the four masks a device has launched. Read the per
# component live-slot states of those four:
#
#     (1,1,1,1,1,1) -> Ex(1,1) Ey(1,1) Ez(1,1)
#     (1,0,0,1,0,0) -> Ex(1,0) Ey(0,1) Ez(0,0)
#     (1,1,0,0,0,0) -> Ex(1,1) Ey(0,0) Ez(0,0)
#     (0,0,1,0,0,0) -> Ex(0,0) Ey(1,0) Ez(0,0)
#
# Ez is either FULLY live or FULLY dead in all four; the state "Ez carries exactly
# its offset-1 slot" is reached by 16 of the 63 masks and by NONE of the four. A
# defect confined to that state is a half-cell registration error -- smooth,
# converged, plausible and wrong -- that the entire device record to date would
# report as identical. The falsify leg runs it over the four AND over all 63 and
# reports both numbers, because the claim is comparative.


def _plant_ez_offset1(component: int, row_mask: Sequence[int], arm: str) -> str:
    """Ez's offset-1 term takes the offset-0 PARTNER AXIS, when it is alone.

    Applied to the emitter's own output for that component, so everything else --
    the volume, the coefficient, the own-axis up shift, the wall mask, the tail --
    is the shipped text. Only the DOWN leg and its near-face rule move from axis Y
    to axis X, which is precisely the ``dj`` where ``dk`` belongs error the
    emitter's docstring names as the reason the index expressions are assembled
    from a table.
    """
    text = _PRISTINE_COMPONENT_SOURCE(component, row_mask, arm)
    live = [offset for offset in (0, 1) if row_mask[2 * component + offset]]
    if component != 2 or live != [1]:
        return text
    text = text.replace("flat(i, dj, k, nyz, nz)", "flat(di, j, k, nyz, nz)")
    text = text.replace("flat(i, dj, uk, nyz, nz)", "flat(di, j, uk, nyz, nz)")
    return text.replace("at_y, bc_y, ph_y, dpy, gw_y,",
                        "at_x, bc_x, ph_x, dpx, gw_x,")


def _plant_ez_offset1_everywhere(component: int, row_mask: Sequence[int],
                                 arm: str) -> str:
    """The SAME rewrite with the mask condition removed -- the control.

    Its whole job is to show that the four exercised masks catch a defect of this
    SHAPE when it is not confined; without it, "the four masks missed it" could be
    a statement about the comparator rather than about the masks.
    """
    text = _PRISTINE_COMPONENT_SOURCE(component, row_mask, arm)
    if component != 2 or not any(row_mask[4:6]):
        return text
    text = text.replace("flat(i, dj, k, nyz, nz)", "flat(di, j, k, nyz, nz)")
    text = text.replace("flat(i, dj, uk, nyz, nz)", "flat(di, j, uk, nyz, nz)")
    return text.replace("at_y, bc_y, ph_y, dpy, gw_y,",
                        "at_x, bc_x, ph_x, dpx, gw_x,")


#: The emitter's own function, captured at import so a plant can be layered over
#: it and lifted off again. Patching the MODULE GLOBAL is what makes a planted
#: emitter reach ``complex_offdiag_source`` -- and therefore reach NVRTC on a
#: device leg through exactly the path a real launch takes.
_PRISTINE_COMPONENT_SOURCE = family._component_source

EMITTER_PLANTS: Dict[str, Callable[..., str]] = {
    "ez_offset1_reads_the_offset0_partner": _plant_ez_offset1,
    "ez_offset1_reads_the_offset0_partner_unconfined":
        _plant_ez_offset1_everywhere,
}


class planted_emitter:
    """Install one emitter plant for the duration of a block, then lift it."""

    def __init__(self, name: Optional[str]) -> None:
        self.name = name

    def __enter__(self):
        if self.name is not None:
            family._component_source = EMITTER_PLANTS[self.name]
        return self

    def __exit__(self, *_exc) -> bool:
        family._component_source = _PRISTINE_COMPONENT_SOURCE
        return False


#: Rewrites of the ARM BLOCK, for the helper leg. Each is applied to the emitted
#: text, so the leg executes the mutated body exactly as it would execute a
#: shipped one.
def _arm_plant_plane_wise(text: str) -> str:
    """The plane-wise complex multiply -- byte-wrong on signed zeros and on
    everything else besides. ``complex_emitter``'s note 1 is the derivation."""
    return text.replace(
        "    o.re = (g.re * p.re) - (g.im * p.im);\n"
        "    o.im = (g.re * p.im) + (g.im * p.re);",
        "    o.re = (g.re * p.re);\n"
        "    o.im = (g.im * p.im);")


def _arm_plant_sign_flip(text: str) -> str:
    """The classic sign error: every magnitude stays plausible, only the phase
    moves."""
    return text.replace("    o.re = (g.re * p.re) - (g.im * p.im);",
                        "    o.re = (g.re * p.re) + (g.im * p.im);")


def _arm_plant_commute_zero_cross(text: str) -> str:
    """THE DECLARED NULL: the exact-zero cross term with its operands swapped.

    float32 multiplication is commutative on every operand including signed zeros,
    so this must come back NULL CONFIRMED. A battery with no predicted null cannot
    show that its scoring distinguishes a catch from a no-op.
    """
    return text.replace("    o.re = (z.re * c) - (z.im * 0.0f);",
                        "    o.re = (z.re * c) - (0.0f * z.im);")


ARM_PLANTS: Dict[str, Dict[str, Any]] = {
    "naive_rotate_plane_wise": {
        "transform": _arm_plant_plane_wise, "arm": "NAIVE",
        "must_be_caught": True,
        "why": "the plane-wise spelling this family refuses by name"},
    "naive_rotate_sign_flip": {
        "transform": _arm_plant_sign_flip, "arm": "NAIVE",
        "must_be_caught": True,
        "why": "the cross-term sign: plausible magnitudes, wrong phase"},
    "naive_commute_the_zero_cross": {
        "transform": _arm_plant_commute_zero_cross, "arm": "NAIVE",
        "must_be_caught": False,
        "why": "PREDICTED NULL: float32 multiply is commutative, signed zeros "
               "included"},
}


#: Rewrites of the emitted BODY, for the structural sweep legs. The gate owns a
#: 30-strong battery of these; what is taken here is the two that a BREADTH sweep
#: is the right place to score -- one that must be caught on every mask, and one
#: the numpy comparator provably cannot see, carried so the artifact states the
#: limit rather than leaving it to be discovered.
def _body_plant_down_index(text: str) -> Tuple[str, int]:
    """The partner-axis DOWN index replaced by the home coordinate.

    THE COUNT IS A COUNT, not a boolean. ``sites`` is what the artifact reports as
    "the mutation was armed", and a 1 that means "at least one" cannot be told
    from a 1 that means "exactly one" -- which matters here, because on the full
    mask this index appears in two different components' terms.
    """
    return _count_replace(text, "flat(i, dj, k, nyz, nz)",
                          "flat(i, j, k, nyz, nz)")


def _body_plant_comment_only(text: str) -> Tuple[str, int]:
    """THE PREDICTED NULL: a comment line, nothing else."""
    return _count_replace(text, "    int nyz = ny * nz;",
                          "    // planted comment, no arithmetic\n"
                          "    int nyz = ny * nz;")


def _count_replace(text: str, needle: str, replacement: str) -> Tuple[str, int]:
    return text.replace(needle, replacement), text.count(needle)


def _body_plant_commute_row_sum(text: str) -> Tuple[str, int]:
    """The row accumulation's two operands swapped.

    DECLARED NULL ON THE NumPy BACKEND AND FOR A STATED REASON: the anchored
    evaluator computes the accumulation itself in live-slot order and only ASSERTS
    that each accumulated term is live, so it does not read the operand ORDER.
    float32 addition is bitwise commutative, so the device leg scores it null too
    -- but for a different reason, and the artifact records which.
    """
    out, count = re.subn(r"cf_add\(total_(E[xyz]), term_(E[xyz])_([01])\)",
                         r"cf_add(term_\2_\3, total_\1)", text)
    return out, count


BODY_PLANTS: Dict[str, Dict[str, Any]] = {
    "down_index_is_the_home_coordinate": {
        "transform": _body_plant_down_index, "must_be_caught": True,
        "why": "a half-cell registration error on the partner axis"},
    "comment_only": {
        "transform": _body_plant_comment_only, "must_be_caught": False,
        "why": "PREDICTED NULL: the scoring must distinguish a catch from a "
               "no-op"},
    "commute_the_row_sum": {
        "transform": _body_plant_commute_row_sum, "must_be_caught": False,
        "why": "PREDICTED NULL: float32 addition is bitwise commutative, and on "
               "the numpy backend the evaluator does not read the order at all"},
}


# ---------------------------------------------------------------------------
# THE TWO COMPARATORS
# ---------------------------------------------------------------------------

class EvaluatorBackend:
    """The laptop comparator: the emitted text, in complex128, against the oracle.

    IT DOES NOT CERTIFY AND SAYS SO IN EVERY ROW. ``certifies`` is False, ``kind``
    names the measurement, and :func:`build_verdict` refuses to release on it.
    """

    name = "numpy"
    certifies = False
    needs_tables = False
    kind = "structural_complex128_relative"
    comparison = ("relative deviation of the emitted source's constitutive value "
                  "from stepping.update_E, in complex128, floored at the float32 "
                  "input epsilon")

    def __init__(self) -> None:
        self.source_transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self.sources_used: Dict[str, str] = {}
        self.guard: Tuple[str, ...] = ()

    def set_guard(self, guard: Sequence[str]) -> None:
        """RECORDED, NOT APPLIED. There is no compiler on this leg, so the guard
        is a device axis this backend leaves unswept rather than one it pretends
        to hold."""
        self.guard = tuple(guard)

    def set_source_mutation(self, transform) -> None:
        self.source_transform = transform

    def pristine_source(self, arm: str, mask: Sequence[int], expansion) -> str:
        return family.complex_offdiag_source(arm, mask, expansion)

    def source_for(self, arm: str, mask: Sequence[int],
                   expansion) -> Tuple[str, int]:
        text = self.pristine_source(arm, mask, expansion)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def compare(self, oracle_after: Dict[str, np.ndarray], kernel_fields,
                arm: str, mask: Sequence[int], expansion, grid,
                launch_args) -> Dict[str, Any]:
        text, sites = self.source_for(arm, mask, expansion)
        self.sources_used[str((arm, tuple(mask), expansion))] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        codes, walls, flags, weights, down, up, _tables = launch_args
        try:
            got = bench.evaluate_source(
                text, kernel_fields, codes, walls, weights,
                (flags, down, up), arm, tuple(grid.shape))
        except Exception as exc:  # noqa: BLE001 - a refusal is a result
            return {"agrees": False, "error": f"{type(exc).__name__}: {exc}"[:600],
                    "worst_relative": None, "mutation_sites": sites}
        worst = 0.0
        per_component = {}
        for name in ("Ex", "Ey", "Ez"):
            reference = oracle_after[
                ("f_w_" + name) if arm == "pml" else name].astype(np.complex128)
            scale = max(float(np.max(np.abs(reference))), 1e-30)
            deviation = float(np.max(np.abs(got[name] - reference))) / scale
            per_component[name] = deviation
            worst = max(worst, deviation)
        return {"agrees": worst < STRUCTURAL_FLOOR, "worst_relative": worst,
                "per_component": per_component, "error": None,
                "mutation_sites": sites}

    def clear(self) -> int:
        self.source_transform = None
        return 0


class DeviceBackend(gate.KernelBackend):
    """The device comparator: the gate's own launcher, byte-compared.

    NOTHING IS RESPELLED. ``launch``, ``set_guard``, ``source_for`` and the
    override bookkeeping are the gate's; the only thing added is the comparison
    call, so this file cannot compile a kernel the gate would not have compiled.
    """

    name = "cupy"
    certifies = True
    needs_tables = True
    kind = "byte_identity_uint32"
    comparison = ("exact uint32 word comparison of every written volume against "
                  "stepping.update_E")

    def compare(self, oracle_after: Dict[str, np.ndarray], kernel_fields,
                arm: str, mask: Sequence[int], expansion, grid,
                launch_args) -> Dict[str, Any]:
        codes, walls, flags, weights, down, up, tables = launch_args
        try:
            self.launch(kernel_fields, arm, mask, expansion, tables, codes,
                        walls, flags, weights, down, up)
        except Exception as exc:  # noqa: BLE001 - a refusal is a result
            return {"agrees": False, "error": f"{type(exc).__name__}: {exc}"[:600],
                    "differing_words": None}
        parts = {name: bit_compare(oracle_after[name], getattr(kernel_fields, name))
                 for name in gate.outputs_for(arm)}
        verdict = combine(parts)
        return {"agrees": bool(verdict["bit_identical"]), "error": None,
                "differing_words": int(verdict["differing_floats"]),
                "total_words": int(verdict["total_floats"]),
                "max_ulp": verdict.get("max_ulp"),
                "per_component": {name: int(part["differing_floats"])
                                  for name, part in parts.items()}}


# ---------------------------------------------------------------------------
# ONE CASE
# ---------------------------------------------------------------------------

def folded_axis_absorbs(layer, grid) -> Dict[str, Any]:
    """Does the FOLDED axis's HALF-INTEGER profile beat the identity?

    ``kps = kms = 1`` is the interior pass-through, and an axis whose whole vector
    is that identity cannot distinguish a coefficient-index error on it -- which
    is precisely the axis the fold moved. ``gate.folded_axis_absorbs`` computes
    exactly this and reaches the vectors through ``complex_offdiag_tables``, which
    imports the certified sibling's kernel module and therefore ``cupy`` at scope;
    that makes it unreachable on the laptop leg.

    So the vectors are read off the ``PML`` directly -- WHICH IS WHAT
    ``real_constitutive_tables`` DOES (constitutive_kernels.py:426-431:
    ``getattr(pml, f"{name}_{axis}{suffix}").reshape(-1)``) -- and the sub-lattice
    suffix is ASKED of ``coverage.constitutive_sub_lattice("E")`` rather than
    spelled, because that function is the single place the E side's pairing with
    the half-integer table is decided and a second spelling of it is a half-cell
    error in the absorber profile: converged, smooth and wrong.
    """
    out: Dict[str, Any] = {"folded_axes": [], "max_deviation": 0.0,
                           "sub_lattice_half_integer": bool(
                               coverage.constitutive_sub_lattice("E"))}
    if layer is None or not getattr(layer, "is_active", False):
        out["meets_floor"] = True
        return out
    suffix = "_h" if out["sub_lattice_half_integer"] else ""
    for axis, name in enumerate("xyz"):
        if not grid.is_mirrored(axis):
            continue
        deviation = 0.0
        for stem in ("kps", "kms"):
            values = to_host(
                getattr(layer, f"{stem}_{name}{suffix}")).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        out["folded_axes"].append({"axis": axis, "name": name,
                                   "max_deviation_from_identity": deviation})
        out["max_deviation"] = max(out["max_deviation"], deviation)
    out["meets_floor"] = (not out["folded_axes"]) or out["max_deviation"] > 0.0
    return out


def case_key(spec: Dict[str, Any], mask: Sequence[int], expansion: str,
             courant: float, value_class: str, guard: str) -> str:
    digits = "".join(str(int(flag)) for flag in mask)
    return "|".join([spec["label"], f"m{digits}", expansion, f"C{courant}",
                     value_class, guard])


def one_case(backend, xp, spec: Dict[str, Any], mask: Sequence[int],
             expansion: str, courant: float, value_class: str,
             guard_label: str, guard: Sequence[str], licence: Dict[str, Any],
             policy_name: Optional[str]) -> Dict[str, Any]:
    """One frozen problem, run twice: the array path, then the emitted kernel.

    TWO FIXTURES FROM ONE SEED, not one fixture restored -- the gate's shape, and
    it removes the class of failure where a restore reproduced values but not the
    object graph. Both are asserted byte-identical before either is touched.
    """
    started = time.time()
    key = case_key(spec, mask, expansion, courant, value_class, guard_label)
    seed = gate.case_seed(key)
    arm = spec["arm"]
    mask = tuple(int(flag) for flag in mask)

    oracle_fields, oracle_layer, oracle_grid = gate.build(
        xp, spec, mask, courant, value_class, seed)
    kernel_fields, kernel_layer, grid = gate.build(
        xp, spec, mask, courant, value_class, seed)

    codes = family.complex_offdiag_boundary_codes(grid)
    walls = coverage.offdiag_wall_mask_flags(grid)
    weights = family.mirror_ghost_weights(grid)
    flags, down, up = family.bloch_phase_table(grid)
    # THE COEFFICIENT TABLES ARE A DEVICE ARGUMENT. ``complex_offdiag_tables``
    # imports the certified sibling's kernel module, which imports cupy at scope,
    # and the laptop comparator does not bind them at all: the anchored evaluator
    # returns the CONSTITUTIVE value and never runs the split-field tail. That is
    # a stated limit of this backend, not an omission -- the recurrence is the
    # certified sibling's and the device gate re-measures it.
    tables = (family.complex_offdiag_tables(kernel_layer)
              if arm == "pml" and backend.needs_tables else None)
    roles = coverage.offdiag_fold_roles(
        mask, tuple(bool(grid.is_mirrored(axis)) for axis in range(3)))

    case: Dict[str, Any] = {
        "key": key, "seed": seed, "label": spec["label"], "arm": arm,
        "row_mask": list(mask),
        "row_mask_is_device_exercised": mask in EXERCISED_ROW_MASKS,
        "expansion": expansion,
        "expansion_is_device_exercised": expansion in EXERCISED_ARMS,
        "courant": courant, "value_class": value_class, "guard": guard_label,
        "backend": backend.name, "measurement": backend.kind,
        "shape": [int(n) for n in grid.shape],
        "cells": int(np.prod([int(n) for n in grid.shape])),
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "codes": list(codes), "walls": list(walls),
        "phase_flags": list(flags),
        "ghost_weights": [float(v) for v in weights],
        "fold_reaches_the_partner_arm": bool(
            roles["fold_is_a_live_partner_axis"]),
        "fold_reaches_the_own_arm": bool(roles["fold_is_a_live_own_axis"]),
        "wall_mask_bites": bool(gate.wall_mask_bites(mask, walls)),
        "live_slots": int(sum(mask)),
    }

    # THE PREDICATE'S ANSWER, RECORDED AND NOT ACTED ON -- and on the NAIVE arm it
    # is the thing that makes this a measurement rather than a dispatch. The
    # licence binds FMA_V1, so the planner would never plan the arm this leg
    # launches; the launcher is reached directly and the row says so.
    verdict = (family.covers_complex_offdiag_pml_update_e if arm == "pml"
               else family.covers_complex_no_pml_offdiag_update_e)(
        kernel_fields, kernel_layer, grid, license=licence,
        subnormal_policy=policy_name)
    case["predicate_today"] = {"covered": bool(verdict[0]), "reason": verdict[1]}
    case["licensed_arm"] = licence.get("arm")
    case["planner_bypassed"] = expansion != licence.get("arm")

    case["fixtures_agree"] = gate.fixtures_agree(oracle_fields, kernel_fields, arm)
    case["coefficient_profile"] = gate.coefficient_profile(kernel_fields)
    case["folded_axis_absorbs"] = folded_axis_absorbs(kernel_layer, grid)
    case["mirror_ghost_is_live"] = gate.mirror_ghost_is_live(kernel_fields, grid)
    case["phase_is_live"] = gate.phase_is_live(grid, flags)
    for floor in ("fixtures_agree", "coefficient_profile", "folded_axis_absorbs",
                  "mirror_ghost_is_live", "phase_is_live"):
        if not case[floor]["meets_floor"]:
            case["skipped"] = f"{floor} is below its floor"
            case["seconds"] = round(time.time() - started, 3)
            return case

    frozen = gate.slot_hosts(oracle_fields, arm)

    # --- the oracle --------------------------------------------------------
    # ONE LAUNCH. The horizon axis belongs to the long-run leg; what is swept here
    # is breadth, and a multi-step sweep over 63 masks would buy a slower version
    # of the same statement.
    stepping.update_E(oracle_fields, oracle_layer)
    oracle = gate.slot_hosts(oracle_fields, arm)

    # NON-VACUITY, PER CASE AND BEFORE ANY COMPARISON. A zero-initialised
    # constitutive step is a FIXED POINT of this sub-step: every tree agrees on it,
    # so a deliberately wrong reference still reports identical. Half of an earlier
    # gate's cases could not fail for exactly this reason.
    moved = sum(int(np.count_nonzero(
        oracle[name].view(np.float32).view(np.uint32)
        != frozen[name].view(np.float32).view(np.uint32)))
        for name in gate.outputs_for(arm))
    case["oracle_moved_words"] = moved
    case["oracle_moved"] = moved > 0

    # AND THE COUPLING MUST BE LIVE. The row product has to differ from the bare
    # diagonal somewhere, or no defect in THIS family can bite and the case is
    # about the certified element-wise sibling instead.
    constitutive = (oracle["f_w_Ex"] if arm == "pml" else oracle["Ex"])
    diagonal = (to_host(oracle_fields.Dx).astype(np.complex128)
                * to_host(oracle_fields.inverse_epsilon_for("Ex")).astype(
                    np.float64))
    case["coupling_max_abs"] = float(np.max(np.abs(
        constitutive.astype(np.complex128) - diagonal)))
    case["coupling_is_live"] = case["coupling_max_abs"] > 0.0

    # --- the kernel, on the SECOND fixture ---------------------------------
    backend.set_guard(guard)
    result = backend.compare(oracle, kernel_fields, arm, mask, expansion, grid,
                             (codes, walls, flags, weights, down, up, tables))
    case.update({f"result_{name}": value for name, value in result.items()})
    case["agrees"] = bool(result["agrees"]) and result.get("error") is None
    case["seconds"] = round(time.time() - started, 3)
    return case


def case_is_valid(case: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """A case that could not distinguish anything is REFUSED, not passed."""
    if case.get("skipped"):
        return False, case["skipped"]
    if not case.get("oracle_moved"):
        return False, ("the array path changed no output word: a zero-initialised "
                       "constitutive step is a fixed point and a case that moved "
                       "nothing certifies nothing")
    if not case.get("coupling_is_live"):
        return False, ("the off-diagonal coupling is identically zero: this case "
                       "cannot distinguish any defect in this family")
    return True, None


# ---------------------------------------------------------------------------
# THE PLANS
# ---------------------------------------------------------------------------

def spec_by_label(label: str) -> Dict[str, Any]:
    for spec in gate.SPECS:
        if spec["label"] == label:
            return spec
    raise KeyError(label)


#: The two specs the SHRUNK form keeps: one folded PML grid and one store grid,
#: so both tails and the fold survive the cut.
REDUCED_BREADTH_SPEC_LABELS: Tuple[str, ...] = ("pml_fold_X_metallic",
                                                "store_3d_phased")


def rowmask_plan(product: str, licensed_arm: str) -> List[Dict[str, Any]]:
    """ALL 63 masks x both tails, at one arm, one value class, one policy.

    ``reduced`` CUTS THE SPECS AND THE COURANT AND NEVER THE MASKS. The mask axis
    is the one thing this leg exists to complete, and ``ALL_ROW_MASKS`` is ordered
    by the integer whose bits it is -- so a head slice of it is a slice of the
    SLOTS, not a sample of the masks: ``[:8]`` is every mask with slots 3, 4 and 5
    dead. A shrunk form that cut there would report ``every_row_mask_scored``
    False for a reason about the slice.
    """
    specs = [spec_by_label(label) for label in
             (BREADTH_SPEC_LABELS if product == "full"
              else REDUCED_BREADTH_SPEC_LABELS)]
    masks = ALL_ROW_MASKS
    courants = COURANTS if product == "full" else (gate.INEXACT_COURANT,)
    plan = []
    for mask in masks:
        for spec in specs:
            for courant in courants:
                plan.append({"spec": spec, "mask": mask, "expansion": licensed_arm,
                             "courant": courant, "value_class": "uniform",
                             "guard_label": "fmad_false",
                             "guard": ("--fmad=false",)})
    return plan


def corner_plan(product: str) -> List[Dict[str, Any]]:
    """The four device-exercised masks at BOTH arms, on the full spec list.

    THE ARM AXIS IS COMPLETE HERE and nowhere else, and it is complete at exactly
    the masks whose FMA_V1 answer is already on the device record -- so the NAIVE
    number has a same-fixture, same-seed FMA_V1 number to sit beside instead of a
    number from another run.
    """
    specs = (list(gate.SPECS) if product == "full"
             else [spec_by_label(label) for label in gate.REDUCED_SPEC_LABELS])
    courants = COURANTS if product == "full" else (gate.INEXACT_COURANT,)
    plan = []
    for mask in EXERCISED_ROW_MASKS:
        for expansion in ARMS:
            for spec in specs:
                for courant in courants:
                    for guard_label, guard in GUARD_SETS:
                        plan.append({
                            "spec": spec, "mask": mask, "expansion": expansion,
                            "courant": courant, "value_class": "uniform",
                            "guard_label": guard_label, "guard": guard})
    return plan


# ---------------------------------------------------------------------------
# THE SWEEP DRIVER
# ---------------------------------------------------------------------------

def run_sweep(name: str, plan: Sequence[Dict[str, Any]], backend, xp,
              results: Dict[str, Any], out_path: str, licence, policy_name,
              flush_every: int, jsonl) -> Dict[str, Any]:
    """Run one plan, one flushed line per case, every row written as it lands."""
    cases: List[Dict[str, Any]] = []
    leg = {"plan_size": len(plan), "cases": cases, "started_utc": now()}
    results[name] = leg
    save(results, out_path)
    started = time.time()
    for index, entry in enumerate(plan, start=1):
        case = one_case(backend, xp, entry["spec"], entry["mask"],
                        entry["expansion"], entry["courant"],
                        entry["value_class"], entry["guard_label"],
                        entry["guard"], licence, policy_name)
        valid, reason = case_is_valid(case)
        case["case_is_valid"] = valid
        case["invalid_reason"] = reason
        cases.append(case)
        if jsonl is not None:
            jsonl.write(json.dumps(case) + "\n")
            jsonl.flush()
        elapsed = time.time() - started
        rate = elapsed / index
        detail = (f"rel={case.get('result_worst_relative'):.2e}"
                  if case.get("result_worst_relative") is not None
                  else f"diff={case.get('result_differing_words')}")
        log(f"[{name}] case {index}/{len(plan)} {case['key']} "
            f"valid={valid} agrees={case.get('agrees')} {detail} "
            f"({elapsed:.1f}s elapsed, {rate:.2f}s/case, "
            f"~{rate * (len(plan) - index):.0f}s left)")
        if index % flush_every == 0:
            leg["summary"] = summarize(cases)
            save(results, out_path)
    leg["summary"] = summarize(cases)
    leg["finished_utc"] = now()
    leg["seconds"] = round(time.time() - started, 1)
    save(results, out_path)
    return leg


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [case for case in cases if case.get("case_is_valid")]
    agreeing = [case for case in scored if case.get("agrees")]
    masks = {tuple(case["row_mask"]) for case in scored}
    per_arm: Dict[str, Dict[str, int]] = {}
    for case in scored:
        bucket = per_arm.setdefault(case["expansion"], {"ran": 0, "agree": 0})
        bucket["ran"] += 1
        bucket["agree"] += int(bool(case.get("agrees")))
    per_tail: Dict[str, Dict[str, int]] = {}
    for case in scored:
        bucket = per_tail.setdefault(case["arm"], {"ran": 0, "agree": 0})
        bucket["ran"] += 1
        bucket["agree"] += int(bool(case.get("agrees")))
    worst = [case.get("result_worst_relative") for case in scored
             if case.get("result_worst_relative") is not None]
    return {
        "cases": len(cases), "scored": len(scored),
        "refused_as_vacuous": len(cases) - len(scored),
        "agreeing": len(agreeing),
        "disagreeing": [case["key"] for case in scored
                        if not case.get("agrees")][:40],
        "distinct_row_masks_scored": len(masks),
        "row_masks_scored_beyond_the_device_four": len(
            masks - set(EXERCISED_ROW_MASKS)),
        "per_arm": per_arm, "per_tail": per_tail,
        "worst_relative": max(worst) if worst else None,
        "folded_partner_arm_scored": sum(
            int(c["fold_reaches_the_partner_arm"]) for c in scored),
        "folded_own_arm_scored": sum(
            int(c["fold_reaches_the_own_arm"]) for c in scored),
        "wall_mask_bites_scored": sum(int(c["wall_mask_bites"]) for c in scored),
        "phased_scored": sum(int(any(c["phase_flags"])) for c in scored),
    }


# ---------------------------------------------------------------------------
# THE ARM LEG
# ---------------------------------------------------------------------------

def arm_text_leg() -> Dict[str, Any]:
    """Is the expansion arm CONFINED to the prelude, on all 63 x 2 sources?

    If it is, then a body measured on one arm is the same body on the other and
    the arm question reduces to three helper functions -- which is what makes the
    helper leg below a statement about the whole kernel rather than about a
    fragment. If it is not, that is a finding: the arm would then be an axis of
    the row product too, and 63 x 2 x 2 sources would each need their own launch.
    """
    preludes = {arm: family._prelude(arm) for arm in ARMS}
    arm_blocks = {arm: complex_emitter._ARM_SOURCE[
        complex_emitter.EXPANSIONS[arm]] for arm in ARMS}
    shared = {arm: preludes[arm].replace(arm_blocks[arm], "") for arm in ARMS}
    rows: List[Dict[str, Any]] = []
    confined = 0
    for tail in TAILS:
        for mask in ALL_ROW_MASKS:
            bodies = {}
            for arm in ARMS:
                text = family.complex_offdiag_source(tail, mask, arm)
                assert text.startswith(preludes[arm]), (tail, mask, arm)
                bodies[arm] = text[len(preludes[arm]):]
            same = bodies[ARMS[0]] == bodies[ARMS[1]]
            confined += int(same)
            if not same:
                rows.append({"tail": tail, "row_mask": list(mask),
                             "bodies_identical": False})
    return {
        "sources_compared": len(TAILS) * len(ALL_ROW_MASKS),
        "bodies_identical_across_arms": confined,
        "arm_is_confined_to_the_prelude": confined == len(TAILS) * len(
            ALL_ROW_MASKS),
        "shared_prelude_identical": shared[ARMS[0]] == shared[ARMS[1]],
        "fmaf_calls_per_arm": {arm: arm_blocks[arm].count("__fmaf_rn")
                               for arm in ARMS},
        #: The contraction question, as a TEXT count. A bare product feeding a
        #: bare add is what ``--fmad=true`` may contract; ``__fmaf_rn`` is a
        #: single ``fma.rn.f32`` whatever the flag says. The real measurement is
        #: the PTX and belongs to the device leg -- this is the reason to take it.
        "contractable_bare_pairs_per_arm": {
            arm: len(re.findall(r"\) [-+] \(", arm_blocks[arm]))
            for arm in ARMS},
        "exceptions": rows,
    }


def arm_helper_leg(size: int, seed: int, plant: Optional[str] = None
                   ) -> Dict[str, Any]:
    """Each arm's three multiply bodies, executed in float32 against the oracle.

    THE BODIES ARE PARSED OUT OF THE EMITTED SOURCE and executed; nothing here
    re-types them. The reference is numpy's own complex64 operator on the same
    operands with the array path's own operand order, which IS the array path at
    that site.

    WHAT A DIFFERING COUNT MEANS ON THIS HOST AND WHAT IT DOES NOT. It classifies
    THIS host's numpy build against the two spellings. It is a second draw of the
    same classification the device probe reached for CuPy on the GPU host, and it is
    not evidence about the GPU host: different ISA, different compiler, different
    contraction.
    """
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        source = family.complex_offdiag_source("pml", (1, 1, 1, 1, 1, 1), arm)
        if plant is not None and ARM_PLANTS[plant]["arm"] == arm:
            source = ARM_PLANTS[plant]["transform"](source)
        for site in HELPER_SITES:
            body = parse_helper(source, site["helper"])
            for value_class in HELPER_VALUE_CLASSES:
                # THE PLANT IS NOT IN THE SEED KEY, and that is a correction
                # this leg's own falsification forced. With the plant name in the
                # key the baseline and the planted run drew DIFFERENT OPERANDS,
                # so every "caught" was partly a statement about the draw --
                # measured 2026-08-20 when the declared-null
                # ``naive_commute_the_zero_cross`` came back CAUGHT (7257 -> 7034
                # words) purely because its seed had moved. Same key, same
                # operands, and the delta is the arithmetic.
                key = f"{arm}|{site['helper']}|{value_class}"
                rng = np.random.default_rng(
                    seed + int.from_bytes(
                        hashlib.sha256(key.encode("ascii")).digest()[:4], "big"))
                operands, left, right = helper_operands(
                    tuple(site["operands"]), value_class, size, rng)
                got = evaluate_helper(body, operands).as_complex64()
                reference = array_path_product(
                    tuple(site["operands"]), left, right)
                verdict = bit_compare(reference, got)
                # NON-VACUITY FOR THIS LEG TOO, and the FIRST spelling of it
                # was wrong in a way worth recording: "the reference has a nonzero
                # word" refuses the pure subnormal band, where every product
                # underflows and the reference is all zeros -- yet that draw is
                # exactly where the two arms disagree, because the zeros carry
                # DIFFERENT SIGNS. What makes a row able to fail is that the
                # reference is not a CONSTANT word, so that is what is measured;
                # the nonzero count is kept beside it because "the whole band
                # product underflowed" is itself a fact this leg establishes.
                words = np.ascontiguousarray(reference).view(np.float32)
                nonzero = int(np.count_nonzero(words))
                distinct = int(np.unique(words.view(np.uint32)).size)
                rows.append({
                    "arm": arm, "helper": site["helper"],
                    "value_class": value_class, "plant": plant,
                    "site": site["site"],
                    "words": int(verdict["total_floats"]),
                    "differing_words": int(verdict["differing_floats"]),
                    "identical": bool(verdict["bit_identical"]),
                    "max_ulp": verdict.get("max_ulp"),
                    "reference_nonzero_words": nonzero,
                    "reference_distinct_words": distinct,
                    "is_vacuous": distinct <= 1,
                })
    by_arm: Dict[str, Dict[str, int]] = {}
    for row in rows:
        bucket = by_arm.setdefault(row["arm"], {"rows": 0, "identical": 0,
                                                "differing_words": 0})
        bucket["rows"] += 1
        bucket["identical"] += int(row["identical"])
        bucket["differing_words"] += row["differing_words"]
    return {
        "operand_count_per_row": size, "rows": rows, "by_arm": by_arm,
        "host_array_path_matches": [arm for arm in ARMS
                                    if by_arm[arm]["differing_words"] == 0],
        "vacuous_rows": [row for row in rows if row["is_vacuous"]],
        "fma_emulation": "libm fmaf through ctypes (one rounding)",
    }


def fma_emulation_control(size: int, seed: int) -> Dict[str, Any]:
    """Is the float64 shortcut this leg REFUSES to use actually different?

    Recorded because "we used libm rather than a float64 detour" is only worth
    something with the delta beside it, and because the delta is expected to be
    zero on a uniform draw and need not be on a subnormal one.
    """
    out = {}
    for value_class in ("uniform", "subnormal_band"):
        rng = np.random.default_rng(seed + len(value_class))
        if value_class == "uniform":
            draw = [rng.uniform(-1, 1, size).astype(np.float32) for _ in range(3)]
        else:
            draw = [np.asarray(probe.subnormal_band_hosts(
                ("v",), (size, 1, 1), rng)["v"], dtype=np.float32).reshape(size)
                for _ in range(3)]
        exact = fmaf(*draw)
        detour = ((draw[0].astype(np.float64) * draw[1].astype(np.float64))
                  + draw[2].astype(np.float64)).astype(np.float32)
        out[value_class] = int(np.count_nonzero(
            exact.view(np.uint32) != detour.view(np.uint32)))
    return {"words": size, "float64_detour_differs_from_libm_fmaf": out}


# ---------------------------------------------------------------------------
# FALSIFICATION
# ---------------------------------------------------------------------------

def falsify_leg(backend, xp, licence, policy_name, product: str,
                out_path: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Plant each defect and require the leg that owns it to move.

    THE COMPARATIVE ONE IS THE POINT. ``ez_offset1_reads_the_offset0_partner`` is
    run over the FOUR masks a device has launched and over ALL 63, and both counts
    are reported: if the four catch it, this file's premise is wrong and the
    record should say so; if only the 63 catch it, that is the measured value of
    the breadth this leg buys.
    """
    legs: List[Dict[str, Any]] = []

    # ---- the emitter plants, over two mask populations --------------------
    # THE REDUCED POPULATION IS A STRIDE, NEVER A HEAD SLICE, and that is not a
    # style preference. ``LIVE_ROW_MASKS`` is ordered by the integer whose bits it
    # is, so ``[:24]`` is every mask with slot 5 DEAD -- exactly the slot the
    # confined plant needs live. A head slice would have made the shrunk form
    # report ``breadth_is_load_bearing: False`` for a reason about the slice.
    reduced = list(dict.fromkeys(list(ALL_ROW_MASKS[::3])
                                 + list(EXERCISED_ROW_MASKS)))
    populations = {
        "device_exercised_four": list(EXERCISED_ROW_MASKS),
        "all_63" if product == "full" else "strided_sample": list(
            ALL_ROW_MASKS if product == "full" else reduced),
    }
    #: One spec per tail, and a courant: the plant is a registration error on the
    #: Ez row and needs no fold to bite, so the population is what varies.
    plant_specs = [spec_by_label("pml_unfolded_walls"),
                   spec_by_label("store_3d_phased")]
    for plant_name in ("ez_offset1_reads_the_offset0_partner",
                       "ez_offset1_reads_the_offset0_partner_unconfined"):
        entry: Dict[str, Any] = {"plant": plant_name, "kind": "emitter",
                                 "populations": {}}
        for population, masks in populations.items():
            caught, scored, caught_masks = 0, 0, []
            with planted_emitter(plant_name):
                backend.clear()
                for mask in masks:
                    for spec in plant_specs:
                        case = one_case(
                            backend, xp, spec, mask, licence["arm"],
                            gate.INEXACT_COURANT, "uniform", "fmad_false",
                            ("--fmad=false",), licence, policy_name)
                        valid, _reason = case_is_valid(case)
                        if not valid:
                            continue
                        scored += 1
                        if not case.get("agrees"):
                            caught += 1
                            if list(mask) not in caught_masks:
                                caught_masks.append(list(mask))
            backend.clear()
            # CAN THIS POPULATION ARM THE PLANT AT ALL? The confined plant bites
            # only where Ez carries exactly its offset-1 slot, so a population
            # with no such mask scores 0 caught for a reason about the population
            # rather than about the defect. Counted from the masks, and reported,
            # so a zero is never read as a null.
            armable = [list(mask) for mask in masks
                       if not mask[4] and mask[5]]
            entry["populations"][population] = {
                "masks": len(masks), "cases_scored": scored,
                "cases_caught": caught,
                "distinct_masks_that_caught_it": caught_masks[:20],
                "mask_count_that_caught_it": len(caught_masks),
                "masks_in_the_state_the_confined_plant_targets": len(armable),
            }
            log(f"[falsify] {plant_name} on {population}: "
                f"{caught}/{scored} cases caught it "
                f"({len(caught_masks)} distinct masks)")
        legs.append(entry)
        results["falsify"] = {"legs": legs}
        save(results, out_path)

    # ---- the body plants, on the mask where every plant HAS a site --------
    #
    # THE FULL MASK, and that is a correction this leg's own first run forced:
    # on (1,0,0,1,0,0) no component carries two live slots, so the row-sum plant
    # rewrote NOTHING and was scored NULL CONFIRMED -- a null for a mutation that
    # was never applied, which is the exact failure "a leg reporting a pass for a
    # mutation it never introduced" that this track has recorded three times. A
    # zero-site plant is now scored UNARMED and fails the leg.
    for name, spec_entry in BODY_PLANTS.items():
        transform = spec_entry["transform"]
        backend.set_source_mutation(transform)
        case = one_case(backend, xp, spec_by_label("pml_fold_X_metallic"),
                        (1, 1, 1, 1, 1, 1), licence["arm"],
                        gate.INEXACT_COURANT, "uniform", "fmad_false",
                        ("--fmad=false",), licence, policy_name)
        backend.clear()
        sites = case.get("result_mutation_sites")
        caught = (not case.get("agrees")) or case.get("result_error") is not None
        armed = bool(sites)
        legs.append({
            "plant": name, "kind": "body", "why": spec_entry["why"],
            "must_be_caught": spec_entry["must_be_caught"],
            "mutation_sites": sites, "armed": armed,
            "outcome": ("UNARMED" if not armed
                        else "CAUGHT" if caught else "NULL_CONFIRMED"),
            "as_required": armed and (
                bool(caught) == bool(spec_entry["must_be_caught"])),
            "worst_relative": case.get("result_worst_relative"),
            "differing_words": case.get("result_differing_words"),
            "error": case.get("result_error"),
        })
        log(f"[falsify] body plant {name}: {legs[-1]['outcome']} "
            f"required_caught={spec_entry['must_be_caught']} "
            f"as_required={legs[-1]['as_required']} sites={sites}")
        results["falsify"] = {"legs": legs}
        save(results, out_path)

    # ---- the arm plants, on the helper leg --------------------------------
    baseline = arm_helper_leg(size=4096, seed=SEED)
    for name, spec_entry in ARM_PLANTS.items():
        planted = arm_helper_leg(size=4096, seed=SEED, plant=name)
        arm = spec_entry["arm"]
        # PER ROW, NOT ON THE TOTAL. A plant that moved two rows in opposite
        # directions would cancel in a sum and be scored a null it is not.
        def _rows(record):
            return {(row["helper"], row["value_class"]): row["differing_words"]
                    for row in record["rows"] if row["arm"] == arm}

        before, after = _rows(baseline), _rows(planted)
        moved_rows = sorted(key for key in before if before[key] != after[key])
        moved = bool(moved_rows)
        legs.append({
            "plant": name, "kind": "arm", "arm": arm,
            "why": spec_entry["why"],
            "must_be_caught": spec_entry["must_be_caught"],
            "baseline_differing_words": baseline["by_arm"][arm][
                "differing_words"],
            "planted_differing_words": planted["by_arm"][arm]["differing_words"],
            "rows_that_moved": ["|".join(key) for key in moved_rows],
            "outcome": "CAUGHT" if moved else "NULL_CONFIRMED",
            "as_required": bool(moved) == bool(spec_entry["must_be_caught"]),
        })
        log(f"[falsify] arm plant {name}: {legs[-1]['outcome']} "
            f"required_caught={spec_entry['must_be_caught']} "
            f"as_required={legs[-1]['as_required']} "
            f"{legs[-1]['baseline_differing_words']} -> "
            f"{legs[-1]['planted_differing_words']} words")
        results["falsify"] = {"legs": legs}
        save(results, out_path)

    summary = {
        "legs": legs,
        "body_and_arm_legs_as_required": all(
            leg["as_required"] for leg in legs if "as_required" in leg),
        "breadth_is_load_bearing": None,
    }
    for leg in legs:
        if leg.get("plant") != "ez_offset1_reads_the_offset0_partner":
            continue
        four = leg["populations"]["device_exercised_four"]
        wide_name = next(name for name in leg["populations"]
                         if name != "device_exercised_four")
        wide = leg["populations"][wide_name]
        # THE CLAIM IS COMPARATIVE AND IT IS SCORED THAT WAY: the four masks a
        # device has launched must MISS a defect the wider population CATCHES,
        # and the wider population must have been able to arm it in the first
        # place. Any of the three failing makes the clause False rather than
        # quietly True.
        summary["breadth_is_load_bearing"] = bool(
            four["cases_caught"] == 0 and wide["cases_caught"] > 0
            and wide["masks_in_the_state_the_confined_plant_targets"] > 0)
        summary["confined_plant_population"] = wide_name
        summary["confined_plant_caught_by_the_device_four"] = four["cases_caught"]
        summary["confined_plant_caught_by_the_wider_sweep"] = wide["cases_caught"]
        summary["confined_plant_armable_masks_in_the_device_four"] = four[
            "masks_in_the_state_the_confined_plant_targets"]
        summary["confined_plant_armable_masks_in_the_wider_sweep"] = wide[
            "masks_in_the_state_the_confined_plant_targets"]
    return summary


# ---------------------------------------------------------------------------
# THE DECLARATION, THE VERDICT AND THE ARTIFACT
# ---------------------------------------------------------------------------

def coverage_declaration(results: Dict[str, Any], product: str,
                         policy_name: Optional[str],
                         applied_guards: bool = False) -> Dict[str, Any]:
    """Which cells this run swept and which it did not, counted from the plans.

    A SWEEP THAT SILENTLY SAMPLES IS WORSE THAN ONE THAT DECLARES ITS SAMPLING,
    and the only way to be sure the declaration matches the run is to compute it
    from the cases that ran rather than from the intent that was typed.
    """
    swept_cells = set()
    for leg_name in ("rowmask", "corner"):
        for case in results.get(leg_name, {}).get("cases", []):
            if not case.get("case_is_valid"):
                continue
            swept_cells.add((tuple(case["row_mask"]), case["arm"],
                             case["expansion"], case["guard"], case["label"],
                             case["courant"], case["value_class"]))
    masks_seen = {cell[0] for cell in swept_cells}
    arms_seen = {cell[2] for cell in swept_cells}
    tails_seen = {cell[1] for cell in swept_cells}
    specs_seen = {cell[4] for cell in swept_cells}
    guards_seen = {cell[3] for cell in swept_cells}
    classes_seen = {cell[6] for cell in swept_cells}
    full_product = (len(ALL_ROW_MASKS) * len(TAILS) * len(ARMS) * 2
                    * len(gate.SPECS) * len(COURANTS)
                    * len(gate.VALUE_CLASSES))
    return {
        "product": product,
        "axes": {
            "row_mask": {"available": len(ALL_ROW_MASKS),
                         "swept": len(masks_seen),
                         "complete": len(masks_seen) == len(ALL_ROW_MASKS),
                         "beyond_the_device_four": len(
                             masks_seen - set(EXERCISED_ROW_MASKS))},
            "tail": {"available": len(TAILS), "swept": len(tails_seen),
                     "complete": len(tails_seen) == len(TAILS)},
            "expansion_arm": {
                "available": len(ARMS), "swept": sorted(arms_seen),
                "complete_at_every_mask": False,
                "complete_at_the_device_four": sorted(arms_seen) == sorted(ARMS),
                "note": "the arm axis is complete only at the four masks the "
                        "device already exercised; the 63-mask sweep runs the "
                        "licensed arm alone"},
            "guard": {"available": len(GUARD_SETS),
                      "labels_planned": sorted(guards_seen),
                      "applied": bool(applied_guards),
                      "note": ("a COMPILE-OPTION axis. On the numpy backend the "
                               "label is carried on the row and the option tuple "
                               "is never applied -- there is no compiler -- so "
                               "this axis is UNSWEPT there however many labels "
                               "appear above."
                               if not applied_guards else
                               "applied at NVRTC: the memo keys on the option "
                               "tuple, so a case cannot be served another "
                               "guard's binary")},
            "subnormal_policy": {
                "available": 2, "swept_in_this_process": policy_name,
                "note": "a process-level FPU state; both policies means running "
                        "this file twice, and the two artifacts are the record"},
            "spec": {"available": len(gate.SPECS), "swept": len(specs_seen),
                     "labels": sorted(specs_seen),
                     "not_swept": sorted(
                         {spec["label"] for spec in gate.SPECS} - specs_seen)},
            "value_class": {"available": len(gate.VALUE_CLASSES),
                            "swept": sorted(classes_seen),
                            "note": "the value-class axis is the device gate's "
                                    "and is held at uniform here; a complex128 "
                                    "comparator cannot say anything about a "
                                    "subnormal-band draw"},
            "steps": {"swept": 1,
                      "note": "one launch per case: the horizon axis belongs to "
                              "the long-run leg, not to a breadth sweep"},
        },
        "cells_swept": len(swept_cells),
        "cells_in_the_full_product": full_product,
        "fraction_of_the_full_product": round(
            len(swept_cells) / full_product, 6) if full_product else None,
    }


def build_verdict(results: Dict[str, Any], backend) -> Dict[str, Any]:
    """The release clauses, and on the numpy backend the refusal that overrides."""
    clauses: Dict[str, Any] = {}
    rowmask = results.get("rowmask", {}).get("summary") or {}
    corner = results.get("corner", {}).get("summary") or {}
    arm_text = results.get("naive", {}).get("text") or {}
    helpers = results.get("naive", {}).get("helpers") or {}
    falsify = results.get("falsify") or {}

    clauses["every_row_mask_scored"] = bool(
        rowmask.get("distinct_row_masks_scored") == len(ALL_ROW_MASKS))
    clauses["both_tails_scored"] = bool(
        set((rowmask.get("per_tail") or {})) == set(TAILS))
    rowmask_cases = results.get("rowmask", {}).get("cases", [])
    clauses["no_vacuous_case_scored"] = bool(
        not helpers.get("vacuous_rows")
        and (rowmask.get("scored") or 0) > 0
        and all(case.get("invalid_reason") for case in rowmask_cases
                if not case.get("case_is_valid")))
    clauses["rowmask_all_agree"] = bool(
        rowmask.get("scored") and rowmask.get("agreeing") == rowmask.get("scored"))
    clauses["corner_all_agree"] = bool(
        corner.get("scored") and corner.get("agreeing") == corner.get("scored"))
    clauses["arm_confined_to_the_prelude"] = bool(
        arm_text.get("arm_is_confined_to_the_prelude"))
    clauses["falsification_as_required"] = bool(
        falsify.get("body_and_arm_legs_as_required"))
    clauses["breadth_is_load_bearing"] = falsify.get("breadth_is_load_bearing")

    release = all(value is True for value in clauses.values())
    verdict: Dict[str, Any] = {"clauses": clauses, "release": bool(release)}
    if not backend.certifies:
        # THE OVERRIDE, AND IT IS UNCONDITIONAL. Every clause above can be True on
        # this backend and none of them is a byte claim: the comparator is a
        # complex128 evaluation against a complex64 array path, so the fused arm
        # and the zero cross terms -- the whole subject of the arm leg -- are out
        # of its reach by construction.
        verdict["release"] = False
        verdict["release_blocked_by"] = (
            f"backend {backend.name}: the comparison is {backend.comparison}. "
            f"That measures the TRANSCRIPTION and cannot measure bytes, so no "
            f"row mask and no expansion arm is certified by this artifact. The "
            f"device leg is the same plan with --backend cuda.")
    verdict["what_this_does_not_establish"] = (
        ("byte identity on any mask outside the four the device record names, "
         "under either arm, until --backend cuda has run;"
         if not backend.certifies else
         "the masks and arms this run's coverage_declaration lists as unswept;"),
        ("the split-field TAIL: the anchored evaluator returns the constitutive "
         "value and never runs the recurrence, which belongs to the certified "
         "sibling and to the device gate;"
         if not backend.certifies else
         "nothing about the tail beyond what these fixtures reached;"),
        "the NAIVE arm as a licensed arm: the expansion licence binds FMA_V1 and "
        "every NAIVE row here is a MEASUREMENT reached by bypassing the planner, "
        "recorded per case as planner_bypassed;",
        "anything at long horizon (one launch per case) or at corpus scale (the "
        "largest fixture here is a few thousand cells);",
        "determinism: every case is a single comparison against the oracle and "
        "nothing is run twice;",
        "any throughput claim.",
    )
    return verdict


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, STAMPED IMMEDIATELY BEFORE THE WRITE.

    The stamp goes in the three-line window above the write and not merely
    somewhere in the function: ``meep_gpu/test_gate_provenance.py`` reads that
    window, because anchoring on the first ``json.dump`` in a file once bound a
    stamp inside a refused early-exit branch while the releasing branch recorded
    no provenance at all.
    """
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = out_path + ".tmp"
    gate_provenance.stamp(results)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # MEASURED ON DEVICE 2026-08-21, not a style choice: without this the "flush"
    # policy is UNATTAINABLE for the HOST executor ("MEEP is not imported") and
    # the run refuses at launch. I judged this flag a "gate concept" during the
    # coherence pass and left it off three of the four harnesses; the device
    # disproved that in the first minute. Any harness that INSTALLS a policy
    # needs it, because resolving the host half of that policy reads what MEEP
    # left the process doing.
    parser.add_argument("--import-meep-for-host-policy",
                        action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="numpy")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        default="keep")
    parser.add_argument("--legs", default="rowmask,corner,naive,falsify")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--helper-operands", type=int, default=200000)
    args = parser.parse_args(argv)

    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())

    if args.backend == "cuda":
        if cp is None:
            raise SystemExit("--backend cuda needs CuPy; run it on a device host")
        # The flush leg needs MEEP imported first: install_subnormal_policy
        # refuses in a process that has not, because mp.set_zero_subnormals is
        # the only exposure of this process's FTZ/DAZ bits the package may use.
        meep_import = (probe.import_meep_for_host_policy()
                       if args.subnormal_policy == "flush" else
                       {"requested": False})
        from meep_gpu import subnormal_policy as policy_module  # noqa: PLC0415

        policy = policy_module.install_subnormal_policy(args.subnormal_policy,
                                                        strict=True)
        policy_name = args.subnormal_policy
        backend = DeviceBackend()
        xp = cp
    else:
        # NO POLICY IS INSTALLED ON THE LAPTOP LEG and that is recorded rather
        # than defaulted: the subnormal policy is a device-FPU fact, and a numpy
        # comparator in complex128 cannot see it either way.
        meep_import = {"requested": False}
        policy = None
        policy_name = None
        backend = EvaluatorBackend()
        xp = bench.XP

    gate._POLICY_NAME = args.subnormal_policy
    licence = gate._licence()

    results: Dict[str, Any] = {
        "harness": os.path.basename(__file__),
        "dimension": "BREADTH: every row mask, and the unexercised expansion arm",
        "backend": backend.name,
        "certifies": backend.certifies,
        "measurement": backend.kind,
        "comparison": backend.comparison,
        "subnormal_policy": policy,
        "requested_policy": args.subnormal_policy if backend.certifies else None,
        "meep_import_for_host_policy": meep_import,
        "product": args.product,
        "seed": SEED,
        "structural_floor": STRUCTURAL_FLOOR,
        "emitter_corpus_digest": family.corpus_digest(),
        "kernels": dict(family.KERNEL_NAMES),
        "expansion_licence": {
            "arm": licence.get("arm"), "basis": licence.get("basis"),
            "policy_token_the_record_was_read_under": args.subnormal_policy,
            "probe_patterns": list(licence.get("probe_patterns") or ()),
            "record": licence.get("record_path"),
            "note": "the licence binds ONE arm. Every NAIVE row in this artifact "
                    "bypasses the planner deliberately and is a measurement of "
                    "an arm no device leg has scored, not a dispatch of it.",
        },
        "row_masks_available": len(ALL_ROW_MASKS),
        "row_masks_the_device_has_exercised": [list(m) for m in
                                               EXERCISED_ROW_MASKS],
        "expansion_arms_the_device_has_exercised": list(EXERCISED_ARMS),
        "breadth_spec_labels": list(BREADTH_SPEC_LABELS),
        "started_utc": now(),
    }
    save(results, args.out)

    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                              "cases.jsonl")
    jsonl = open(jsonl_path, "a", encoding="utf-8")
    results["cases_jsonl"] = jsonl_path
    try:
        if "rowmask" in legs:
            plan = rowmask_plan(args.product, licence["arm"])
            spec_labels = sorted({entry["spec"]["label"] for entry in plan})
            courants = sorted({entry["courant"] for entry in plan})
            log(f"[rowmask] {len(plan)} cases: {len(ALL_ROW_MASKS)} masks x "
                f"{len(spec_labels)} specs x {len(courants)} courants, "
                f"arm {licence['arm']}; specs {spec_labels}")
            run_sweep("rowmask", plan, backend, xp, results, args.out, licence,
                      policy_name, args.flush_every, jsonl)
        if "corner" in legs:
            plan = corner_plan(args.product)
            log(f"[corner] {len(plan)} cases: {len(EXERCISED_ROW_MASKS)} "
                f"device-exercised masks x {len(ARMS)} arms x "
                f"{len(GUARD_SETS)} guards")
            leg = run_sweep("corner", plan, backend, xp, results, args.out,
                            licence, policy_name, args.flush_every, jsonl)
            if not backend.certifies:
                # SAID ON THE LEG, not left to a reader. A complex128 evaluator
                # is ARM-BLIND: both arms multiply the same numbers in the same
                # order there and only the float32 rounding differs. So a NAIVE
                # row agreeing here means the arm did not change the BODY -- which
                # the text leg already proves by string equality -- and says
                # nothing whatever about the arm's bytes.
                leg["what_the_arm_rows_mean_on_this_backend"] = (
                    "the NAIVE rows agree because the comparator is complex128 "
                    "and arm-blind; the arm question is measured by the helper "
                    "leg here and by the device leg's byte comparison there")
        if "naive" in legs:
            log("[naive] text leg: is the arm confined to the prelude?")
            text = arm_text_leg()
            log(f"[naive] bodies identical across arms on "
                f"{text['bodies_identical_across_arms']}/"
                f"{text['sources_compared']} sources; "
                f"__fmaf_rn per arm {text['fmaf_calls_per_arm']}; "
                f"contractable bare pairs {text['contractable_bare_pairs_per_arm']}")
            helpers = arm_helper_leg(size=args.helper_operands, seed=SEED)
            for row in helpers["rows"]:
                log(f"[naive] helper {row['arm']:<7} {row['helper']:<22} "
                    f"{row['value_class']:<15} "
                    f"{row['differing_words']}/{row['words']} words differ "
                    f"identical={row['identical']}")
            results["naive"] = {
                "text": text, "helpers": helpers,
                "fma_emulation_control": fma_emulation_control(20000, SEED),
                "host_note": ("the helper leg classifies THIS host's numpy build "
                              "against the two spellings; it is a second draw of "
                              "the classification the device probe reached for "
                              "CuPy, and it is not evidence about that host"),
            }
            save(results, args.out)
        if "falsify" in legs:
            results["falsify"] = falsify_leg(
                backend, xp, licence, policy_name, args.product, args.out,
                results)
            save(results, args.out)
    finally:
        jsonl.close()

    results["coverage_declaration"] = coverage_declaration(
        results, args.product, policy_name, applied_guards=backend.certifies)
    results["verdict"] = build_verdict(results, backend)
    results["source_digests"] = dict(backend.sources_used)
    results["finished_utc"] = now()
    save(results, args.out)

    verdict = results["verdict"]
    log(f"RELEASE={verdict['release']}")
    for name, value in verdict["clauses"].items():
        log(f"  {name}: {value}")
    if verdict.get("release_blocked_by"):
        log(f"  BLOCKED: {verdict['release_blocked_by']}")
    falsify = results.get("falsify") or {}
    if "falsify" in legs and not falsify.get("body_and_arm_legs_as_required"):
        log("FALSIFICATION FAILED: a planted defect did not score as required")
        return 2
    # THE EXIT CODE IS ABOUT THE CLAUSES, NOT ABOUT ``release``. On the numpy
    # backend ``release`` is False by construction, so returning on it would make
    # this leg unable to go green -- and a leg that always exits non-zero is one
    # nobody reads. What can still go RED here is a clause: a mask that failed to
    # score, a tail missing, a disagreement, a vacuous row.
    holds = all(value is True for value in verdict["clauses"].values())
    return 0 if holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
