"""The Dcyl COMPLEX H->D product: ``update_H`` + the pre-scan stage of the radial
prefix in ONE launch, the array path's ``cumsum`` untouched, the certified complex
cylindrical ``step_D`` curl in a second launch.

The complex-storage twin of :mod:`.cylindrical_real_fused_hd_pair`, on the sixteen
buildable rows of the ``(cylindrical complex -> cylindrical complex PML)`` H->D cell
(every m the complex family carries -- m = 0 under complex storage, |m| = 1, |m| >= 2
-- and both z terminations). Read that module first: the shape, the seam, the
withdraw clause and the arbitration are identical and are not restated here. What
this file adds is the COMPLEX arithmetic of the increment, and it is the one place
where the two families' spellings INVERT.

=============================================================================
THE TWO SPELLINGS THAT ARE MEASURED TO INVERT
=============================================================================

``cylindrical_rderiv_prefix`` runs on a complex64 ``f_p`` with float32 ``weights`` and
``divisor`` (stepping.py:1308, 1284-1302). CuPy's ``multiply`` and ``true_divide``
carry no mixed ``complex64 x float32`` loop, so the float32 operand is cast to
``complex<float>(x)`` with imaginary part ``+0.0f`` and the COMPLEX/COMPLEX operators
run. MEASURED on the device (lane record ``cupy_probe/FINDINGS.md``, 4,005,000 words
per case against the ufuncs themselves, then 14,387,104 words per policy inside the
fused stage):

1. **The multiply** ``Hy * w`` is the four-product complex form -- and the device
   CONTRACTS it. The uncontracted four-product transcription misses 6 of 4,005,000
   words under ``flush`` (the sign of a FLUSHED zero when ``Hy.re * 0.5`` underflows
   on row 0 with ``Hy.re < 0``, ``Hy.im < 0``); the spelling that is 0 of 4,005,000
   on every case under both policies is the FMA_V1 ``mul_field_left`` arrangement,
   ``re = fma(z.re, w, (z.im*0.0)*-1.0); im = fma(z.re, 0.0, z.im*w)`` with EXPLICIT
   fused multiply-adds. So :func:`_mul_field_left_fma` spells exactly that,
   REGARDLESS of which ``EXPANSION`` arm the constitutive half runs under: with
   floating-point contraction off an explicit fma survives and an implicit one never
   forms, so the arm of the certified constitutive cannot leak into the increment.
2. **The divide** ``/ d`` is CuPy's SCALED complex division
   (``cupy/_core/include/cupy/complex/arithmetic.h:96-110``) with ``rhs = (d, +0.0f)``
   and EVERY zero-valued term left in::

       s = |d| + |0|;  oos = 1/s;  ars = z.re*oos;  ais = z.im*oos;
       brs = d*oos;  bis = 0*oos;  s = brs*brs + bis*bis;  oos = 1/s;
       q.re = (ars*brs + ais*bis) * oos;   q.im = (ais*brs - ars*bis) * oos

   0 of 14,387,104 words under both policies. numpy's ``a * (float32(1)/d)`` -- the
   spelling the Metal verdict prescribes for complex storage, and CORRECT there
   because that backend's engine is NumPy -- is WRONG here: 793,105 / 820,455 words
   (keep / flush). Componentwise ``re/d, im/d`` -- the real family's ``/`` carried over
   -- is wrong on 3,584,917 / 3,591,312 words. "Tidying" the zero terms out is wrong
   on 15,598 / 60,038 words, all in the class where a product is a signed zero or
   underflows. Both reciprocals are correctly rounded (``tl.math.div_rn``): NVRTC's
   ``1.0f / s`` under the default precise division is an IEEE division and Triton's
   ``/`` is not (the real module says why).

So THREE things a careful reader would "correct" are each an armed mutation on this
product: the numpy reciprocal (from the sibling backend's verdict), the componentwise
divide (from the sibling family's comment), and the four-product multiply (from the
certified constitutive's own NAIVE arm). The last is byte-visible only on the
planted row-0 tiny-normal class under ``flush``, so the gate PLANTS that case; a
random battery alone did not reach it (0 of 14,387,104 with the naive spelling).

Row 0 is ``(+0.0, +0.0)`` -- the array path assigns the integer 0 -- and the subtract
is componentwise. Every NVRTC number above is the pre-scan stage measured in
isolation under ``cp.RawKernel``; the Triton-compiled kernel's own bytes are the
gate's ``increment_stage`` leg and are not inherited from that record.

=============================================================================
THE CONSTITUTIVE HALF IS LIFTED, THE CURL HALF IS LAUNCHED VERBATIM
=============================================================================

:func:`_h_cell_complex` is :func:`.complex_fields.bloch_constitutive_step`'s body
below its decode as a function of a cell, with exactly the substitutions
:data:`CONSTITUTIVE_LIFT_EDITS` names -- the ``SCALE`` branch resolved to the H side,
the three pointer families renamed to the pre-launch buffers, and each store replaced
by a REGISTER CAPTURE so the caller stores to scratch for its own cell and stores
nothing for the foreign recompute. :func:`certified_constitutive_tail` and
:func:`lifted_constitutive_tail` cut both at their anchors and the laptop test asserts
the two strings equal, so this is a LIFT and not a transcription; the arithmetic
inside -- every coefficient-left zero-imaginary product through
:func:`.complex_fields._mul_coefficient_left` under the run's ``EXPANSION`` -- is the
certified body's own, unchanged.

Launch 2 is :func:`.cylindrical_complex.cyl_complex_pml_curl_step` with
``BACKWARD = 1``, launched with the argument list
:class:`.cylindrical_complex.CylindricalComplexCurlPlan.run` passes and the freshly
written twins as its magnetic sources; the i*m/r rows, the axis-increment scalars and
``four_dtdx`` are built by that module's own host functions.

THE EXPANSION LICENCE IS POLICY-CONDITIONAL and it is the certified family's, not
this product's: every complex arm in this package was certified under ``keep``
(``complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY``) and the predicates this module
conjoins refuse under any other policy in force
(``complex_fields.expansion_certification_reasons``). Under ``flush`` the composer
installs nothing complex on these rows and this product's predicate refuses by that
name; what the gate can still measure there is the arithmetic from arrays with the
arm forced, and it says so in the record.

Import contract: importable WITHOUT Triton.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import cylindrical_complex as _cx
from . import cylindrical_real_fused_hd_pair as _real
from . import fused_hd_pair as _cartesian
from .. import withdraw_hoist as _withdraw_hoist
from .complex_fields import _resolve_expansion, _word_view
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

FAMILY = "cylindrical_fused_hd_pair"
SLOT = "update_H"
REPLACES: Tuple[str, ...] = ("update_H", "step_D")
SEAM: str = _withdraw_hoist.SEAM
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two arms this product implements on ``(update_H, step_D)`` -- the labels the
#: composer writes for ``cylindrical_complex_constitutive_coverage('H')`` and
#: ``cylindrical_complex_curl_coverage('step_D')``.
ARMS: Tuple[str, str] = ("cylindrical complex", "cylindrical complex PML")

ROTATED: Tuple[str, ...] = _real.ROTATED
IN_PLACE: Tuple[str, ...] = _real.IN_PLACE
CONSTITUTIVE_SOURCES: Tuple[str, ...] = _real.CONSTITUTIVE_SOURCES

#: The prefix's source and ir0, read off the complex family's own table.
PREFIX_COMPONENT: str = _cx.PREFIX[CURL_SUB_STEP]["component"]
PREFIX_IR0: float = float(_cx.PREFIX[CURL_SUB_STEP]["ir0"])
INCREMENT_TAG = _real.INCREMENT_TAG
PREFIX_TAG = _real.PREFIX_TAG

BACKWARD = 1
DEFAULT_BLOCK = 256
LAUNCHES_PER_RUN = 3
KERNEL_LAUNCHES_PER_RUN = 2
CARRIES_DEPOSIT_REPAIR = False
REPAIR_PATHS: Tuple[str, ...] = ()
HOISTS_THE_WITHDRAW = False
INSTALLABLE = False
INSTALLABLE_REASON = _real.INSTALLABLE_REASON.replace(
    "gate_triton_cylindrical_real_fused_hd_pair.py",
    "gate_triton_cylindrical_fused_hd_pair.py")

#: The four values the FMA_V1 ``mul_field_left`` arrangement fuses, as the increment
#: spells them: measured, not chosen (module docstring, item 1).
INCREMENT_SPELLING: Tuple[Tuple[str, str], ...] = (
    ("halo", "h_re, h_im = _h_tap_complex(1, i - 1, j, k, inner,"),
    ("weight_here", "wh_re, wh_im = _mul_field_left_fma(o1_re, o1_im, w_i)"),
    ("weight_below", "wb_re, wb_im = _mul_field_left_fma(h_re, h_im, w_im1)"),
    ("subtract_re", "d_re = wh_re - wb_re"),
    ("subtract_im", "d_im = wh_im - wb_im"),
    ("divide", "q_re, q_im = _div_coefficient_scaled(d_re, d_im, d_im1)"),
    ("row0_re", "q_re = tl.where(inner, q_re, 0.0)"),
    ("row0_im", "q_im = tl.where(inner, q_im, 0.0)"),
)

#: The scaled-divide statements, in order, exactly as :func:`_div_coefficient_scaled`
#: spells them; the laptop test pins each (zero terms KEPT, both reciprocals
#: correctly rounded) and the gate measures the algorithm against the array path.
DIVIDE_SPELLING: Tuple[str, ...] = (
    "s = tl.abs(d) + 0.0",
    "oos = tl.math.div_rn(1.0, s)",
    "ars = z_re * oos",
    "ais = z_im * oos",
    "brs = d * oos",
    "bis = 0.0 * oos",
    "s2 = (brs * brs) + (bis * bis)",
    "oos2 = tl.math.div_rn(1.0, s2)",
    "q_re = ((ars * brs) + (ais * bis)) * oos2",
    "q_im = ((ais * brs) - (ars * bis)) * oos2",
)

#: The multiply statements, exactly as :func:`_mul_field_left_fma` spells them.
MULTIPLY_SPELLING: Tuple[str, ...] = (
    "out_re = tl.math.fma(z_re, c, (z_im * 0.0) * -1.0)",
    "out_im = tl.math.fma(z_re, 0.0, z_im * c)",
)

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_LIFT_EDITS",
    "CONSTITUTIVE_SIDE", "CONSTITUTIVE_SOURCES", "CURL_SUB_STEP", "DEFAULT_BLOCK",
    "DIVIDE_SPELLING", "FAMILY", "HOISTS_THE_WITHDRAW", "INCREMENT_SPELLING",
    "INCREMENT_TAG", "INSTALLABLE", "INSTALLABLE_REASON", "IN_PLACE",
    "KERNEL_LAUNCHES_PER_RUN", "LAUNCHES_PER_RUN", "MULTIPLY_SPELLING",
    "PREFIX_COMPONENT", "PREFIX_IR0", "PREFIX_TAG", "REPAIR_PATHS", "REPLACES",
    "ROTATED", "SEAM", "SLOT",
    "CylindricalFusedHdPairPlan",
    "certified_constitutive_tail", "cyl_complex_update_H_increment_kernel",
    "cylindrical_fused_hd_pair_coverage", "explain_cylindrical_fused_hd_pair",
    "function_source", "increment_statements", "lifted_constitutive_tail",
    "plan_cylindrical_fused_hd_pair", "plan_cylindrical_fused_hd_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The machine-checked lift of the complex constitutive
# ---------------------------------------------------------------------------

#: Where the certified body ends its decode (``complex_fields.bloch_constitutive_step``).
DECODE_END = "    i = plane // ny\n"

#: Where :func:`_h_cell_complex` re-composes the flat index from its own coordinates.
H_CELL_DECODE_END = "        idx = i * nyz + j * nz + k\n"

#: Where :func:`_h_cell_complex` stops being lifted text and returns its registers.
H_CELL_RETURN_START = "        return (out0_re, out0_im, out1_re, out1_im, out2_re, out2_im,\n"

#: Every line of the certified body this lift does not carry verbatim, with the
#: reason. NO ARITHMETIC: pointer spellings, a resolved constexpr branch, and stores
#: that become register captures.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "if SCALE: ie = tl.load(eN + idx, ...); src = _mul_field_left(src, ie, ...)",
     "became": "(removed -- the constexpr branch is RESOLVED to SCALE = 0)",
     "why": "the H side reads B directly (mu = 1 baked in as the array path bakes it, "
            "stepping.py:907-923); the SCALE = 1 arm is text no launch of this product "
            "reaches and the three inverse-epsilon pointers leave with it."},
    {"line": "prev_re/prev_im = tl.load(wN + 2 * idx ...)",
     "became": "prev_re/prev_im = tl.load(wiN + 2 * idx ...)",
     "why": "the split-field history is read from the PRE-LAUNCH buffer, which "
            "nothing in this launch writes; on this side the newly written f_w_H IS B "
            "exactly, so an in-place read of a neighbour would hand B where B_prev is "
            "needed."},
    {"line": "src_re/src_im = tl.load(gN + 2 * idx ...)",
     "became": "src_re/src_im = tl.load(bN + 2 * idx ...)",
     "why": "one pointer rename: the magnetic flux density under its own name."},
    {"line": "tl.store(wN + 2 * idx, src_re, ...); tl.store(wN + 2 * idx + 1, src_im, ...)",
     "became": "sN_re = src_re; sN_im = src_im",
     "why": "the store moves to the caller, which addresses the SCRATCH for its own "
            "cell and stores nothing for a foreign recompute; the value is untouched."},
    {"line": "a_re/a_im = tl.load(fN + 2 * idx ...)",
     "became": "a_re/a_im = tl.load(hiN + 2 * idx ...)",
     "why": "the accumulator is read from the PRE-LAUNCH H, const for the dispatch -- "
            "what makes the foreign recompute a pure function of unwritten memory."},
    {"line": "tl.store(fN + 2 * idx, a_re, ...); tl.store(fN + 2 * idx + 1, a_im, ...)",
     "became": "outN_re = a_re; outN_im = a_im",
     "why": "the two accumulations, their order and their operands are untouched; "
            "only the destination moves to the caller."},
    {"line": "idx = tl.program_id(0) * BLOCK + ... ; i = plane // ny",
     "became": "(removed -- i, j and k are parameters; idx is composed from them)",
     "why": "the certified body becomes a function evaluated at an ARBITRARY cell; "
            "`live` arrives as the caller's per-lane validity so a masked-off halo "
            "dereferences nothing."},
)


def function_source(function_name: str, path: Optional[Path] = None) -> str:
    """One function's source text, by name, off a file -- never by import."""
    return _cartesian._source_of(  # noqa: SLF001 - the shared lift helper
        function_name,
        Path(__file__).with_name("complex_fields.py") if path is None else path)


def certified_constitutive_tail() -> str:
    """``bloch_constitutive_step``'s body below the decode with exactly the declared
    edits applied through :func:`.fused_hd_pair.needle` -- a drifted spelling RAISES."""
    tail = _cartesian._cut(function_source("bloch_constitutive_step"),  # noqa: SLF001
                           DECODE_END)
    needle = _cartesian.needle
    for target in range(3):
        trailer = "  # D LEFT (S:982-984)\n" if target == 0 else "\n"
        tail = needle(
            tail,
            f"if SCALE:\n"
            f"    ie = tl.load(e{target} + idx, mask=live, other=0.0)\n"
            f"    src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)"
            + trailer,
            "")
        tail = needle(tail, f"prev_re = tl.load(w{target} + 2 * idx",
                      f"prev_re = tl.load(wi{target} + 2 * idx")
        tail = needle(tail, f"prev_im = tl.load(w{target} + 2 * idx",
                      f"prev_im = tl.load(wi{target} + 2 * idx")
        tail = needle(tail, f"src_re = tl.load(g{target} + 2 * idx",
                      f"src_re = tl.load(b{target} + 2 * idx")
        tail = needle(tail, f"src_im = tl.load(g{target} + 2 * idx",
                      f"src_im = tl.load(b{target} + 2 * idx")
        tail = needle(
            tail,
            f"tl.store(w{target} + 2 * idx, src_re, mask=live)\n"
            f"tl.store(w{target} + 2 * idx + 1, src_im, mask=live)\n",
            f"s{target}_re = src_re\ns{target}_im = src_im\n")
        tail = needle(tail, f"a_re = tl.load(f{target} + 2 * idx",
                      f"a_re = tl.load(hi{target} + 2 * idx")
        tail = needle(tail, f"a_im = tl.load(f{target} + 2 * idx",
                      f"a_im = tl.load(hi{target} + 2 * idx")
        tail = needle(
            tail,
            f"tl.store(f{target} + 2 * idx, a_re, mask=live)\n"
            f"tl.store(f{target} + 2 * idx + 1, a_im, mask=live)\n",
            f"out{target}_re = a_re\nout{target}_im = a_im\n")
    for stem in ("f", "w", "g", "e"):
        for target in range(3):
            if f"{stem}{target} +" in tail:
                raise AssertionError(
                    f"the lifted complex update_H body still indexes {stem}{target}")
    if "SCALE" in tail or "tl.store(" in tail:
        raise AssertionError("the lifted complex update_H body still stores or branches")
    return tail.rstrip("\n") + "\n"


def lifted_constitutive_tail() -> str:
    """:func:`_h_cell_complex`'s own body between its two anchors, dedented."""
    source = function_source("_h_cell_complex", Path(__file__))
    return _cartesian._cut(source, H_CELL_DECODE_END,  # noqa: SLF001
                           stop=H_CELL_RETURN_START).rstrip("\n") + "\n"


def increment_statements() -> List[str]:
    """The launch-1 kernel's statements as they stand in the SOURCE, for the tests."""
    import ast  # noqa: PLC0415

    text = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "cyl_complex_update_H_increment", "_div_coefficient_scaled",
                "_mul_field_left_fma"):
            segment = ast.get_source_segment(text, node) or ""
            out.extend(line.strip() for line in segment.splitlines()
                       if line.strip() and not line.strip().startswith("#"))
    if not out:
        raise AssertionError("the module no longer defines its launch-1 kernel")
    return out


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the gate

    from .complex_fields import _mul_coefficient_left  # noqa: PLC0415

    @triton.jit
    def _h_cell_complex(i, j, k, live,
                        hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                        kp0, km0, kp1, km1, kp2, km2, ny, nz,
                        EXPANSION: tl.constexpr):
        """``complex_fields.bloch_constitutive_step`` for side H at ONE named cell.

        The body below the index line is the certified body's own text with exactly
        :data:`CONSTITUTIVE_LIFT_EDITS`; the laptop test asserts the two strings equal.
        Returns the three stepped H word pairs, then the three f_w_H word pairs.
        """
        nyz = ny * nz
        idx = i * nyz + j * nz + k

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z (dsigw).
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -----------------------------------------------------------
        prev_re = tl.load(wi0 + 2 * idx, mask=live, other=0.0)   # BEFORE the store.
        prev_im = tl.load(wi0 + 2 * idx + 1, mask=live, other=0.0)
        src_re = tl.load(b0 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b0 + 2 * idx + 1, mask=live, other=0.0)
        s0_re = src_re
        s0_im = src_im
        a_re = tl.load(hi0 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(hi0 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        out0_re = a_re
        out0_im = a_im

        # --- component 1 -----------------------------------------------------------
        prev_re = tl.load(wi1 + 2 * idx, mask=live, other=0.0)
        prev_im = tl.load(wi1 + 2 * idx + 1, mask=live, other=0.0)
        src_re = tl.load(b1 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b1 + 2 * idx + 1, mask=live, other=0.0)
        s1_re = src_re
        s1_im = src_im
        a_re = tl.load(hi1 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(hi1 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_1, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_1, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        out1_re = a_re
        out1_im = a_im

        # --- component 2 -----------------------------------------------------------
        prev_re = tl.load(wi2 + 2 * idx, mask=live, other=0.0)
        prev_im = tl.load(wi2 + 2 * idx + 1, mask=live, other=0.0)
        src_re = tl.load(b2 + 2 * idx, mask=live, other=0.0)
        src_im = tl.load(b2 + 2 * idx + 1, mask=live, other=0.0)
        s2_re = src_re
        s2_im = src_im
        a_re = tl.load(hi2 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(hi2 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_2, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_2, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        out2_re = a_re
        out2_im = a_im
        return (out0_re, out0_im, out1_re, out1_im, out2_re, out2_im,
                s0_re, s0_im, s1_re, s1_im, s2_re, s2_im)

    @triton.jit
    def _h_tap_complex(COMP: tl.constexpr, i, j, k, valid,
                       hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                       kp0, km0, kp1, km1, kp2, km2, ny, nz,
                       EXPANSION: tl.constexpr):
        """One FOREIGN magnetic word pair: the stepped component at another cell,
        an exact ``(+0.0, +0.0)`` where the caller's guard is False."""
        (o0_re, o0_im, o1_re, o1_im, o2_re, o2_im,
         _s0_re, _s0_im, _s1_re, _s1_im, _s2_re, _s2_im) = _h_cell_complex(
            i, j, k, valid, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, km0, kp1, km1, kp2, km2, ny, nz, EXPANSION)
        v_re = o0_re
        v_im = o0_im
        if COMP == 1:
            v_re = o1_re
            v_im = o1_im
        if COMP == 2:
            v_re = o2_re
            v_im = o2_im
        return tl.where(valid, v_re, 0.0), tl.where(valid, v_im, 0.0)

    @triton.jit
    def _mul_field_left_fma(z_re, z_im, c):
        """``z * (c + 0j)`` with the FIELD on the left, the FMA_V1 arrangement with
        EXPLICIT fused multiply-adds -- the measured spelling of CuPy's contracted
        ``complex<float> * complex<float>(c)`` (module docstring, item 1). Not gated
        on ``EXPANSION``: the arm of the constitutive half must not reach the increment.
        Negation is ``* -1.0``, never unary minus (complex_fields._mul_field_left)."""
        out_re = tl.math.fma(z_re, c, (z_im * 0.0) * -1.0)
        out_im = tl.math.fma(z_re, 0.0, z_im * c)
        return out_re, out_im

    @triton.jit
    def _div_coefficient_scaled(z_re, z_im, d):
        """``z / (d + 0j)`` as CuPy's scaled complex division spells it
        (``cupy/complex/arithmetic.h:96-110``), every zero-valued term KEPT and both
        reciprocals correctly rounded (module docstring, item 2)."""
        s = tl.abs(d) + 0.0
        oos = tl.math.div_rn(1.0, s)
        ars = z_re * oos
        ais = z_im * oos
        brs = d * oos
        bis = 0.0 * oos
        s2 = (brs * brs) + (bis * bis)
        oos2 = tl.math.div_rn(1.0, s2)
        q_re = ((ars * brs) + (ais * bis)) * oos2
        q_im = ((ais * brs) - (ars * bis)) * oos2
        return q_re, q_im

    @triton.jit
    def cyl_complex_update_H_increment(
        ho0, ho1, ho2,                  # SCRATCH out: the stepped Hx, Hy, Hz   (c8 as words)
        wo0, wo1, wo2,                  # SCRATCH out: the stepped f_w_H*        (c8 as words)
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz                  (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_H*                      (read-only)
        b0, b1, b2,                     # Bx, By, Bz                             (read-only)
        inc,                            # SCRATCH out: the radial increment      (c8 as words)
        wgt, dvs,                       # the array path's cached weights (nr), divisor (nr-1), f32
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's own axis, INTEGER lattice
        nx, ny, nz, n_elem,             # n_elem = COMPLEX cells
        EXPANSION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``update_H`` into scratch, plus the pre-scan stage of the radial prefix, on
        complex64 word pairs. Nothing written by this launch is read by it."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ============ THE CONSTITUTIVE: update_H at the program's own cell, to SCRATCH
        (o0_re, o0_im, o1_re, o1_im, o2_re, o2_im,
         s0_re, s0_im, s1_re, s1_im, s2_re, s2_im) = _h_cell_complex(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, km0, kp1, km1, kp2, km2, ny, nz, EXPANSION)
        tl.store(ho0 + 2 * idx, o0_re, mask=live)
        tl.store(ho0 + 2 * idx + 1, o0_im, mask=live)
        tl.store(ho1 + 2 * idx, o1_re, mask=live)
        tl.store(ho1 + 2 * idx + 1, o1_im, mask=live)
        tl.store(ho2 + 2 * idx, o2_re, mask=live)
        tl.store(ho2 + 2 * idx + 1, o2_im, mask=live)
        tl.store(wo0 + 2 * idx, s0_re, mask=live)
        tl.store(wo0 + 2 * idx + 1, s0_im, mask=live)
        tl.store(wo1 + 2 * idx, s1_re, mask=live)
        tl.store(wo1 + 2 * idx + 1, s1_im, mask=live)
        tl.store(wo2 + 2 * idx, s2_re, mask=live)
        tl.store(wo2 + 2 * idx + 1, s2_im, mask=live)

        # ============ THE INCREMENT: stepping.cylindrical_rderiv_prefix's four passes
        inner = live & (i >= 1)
        h_re, h_im = _h_tap_complex(1, i - 1, j, k, inner,
                                    hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                                    kp0, km0, kp1, km1, kp2, km2, ny, nz, EXPANSION)
        w_i = tl.load(wgt + i, mask=live, other=0.0)
        w_im1 = tl.load(wgt + i - 1, mask=inner, other=0.0)
        d_im1 = tl.load(dvs + i - 1, mask=inner, other=1.0)
        # xp.multiply(f_p, weights): complex64 * complex<float>(w), the FIELD on the
        # left, in the arrangement the device is measured to compute.
        wh_re, wh_im = _mul_field_left_fma(o1_re, o1_im, w_i)
        wb_re, wb_im = _mul_field_left_fma(h_re, h_im, w_im1)
        # xp.subtract(weighted[1:], weighted[:-1]): componentwise.
        d_re = wh_re - wb_re
        d_im = wh_im - wb_im
        # increment[1:] /= divisor: complex64 / complex<float>(d), the SCALED algorithm.
        q_re, q_im = _div_coefficient_scaled(d_re, d_im, d_im1)
        # increment[_face(0, 0)] = 0: the axis row is an exact (+0.0, +0.0).
        q_re = tl.where(inner, q_re, 0.0)
        q_im = tl.where(inner, q_im, 0.0)
        tl.store(inc + 2 * idx, q_re, mask=live)
        tl.store(inc + 2 * idx + 1, q_im, mask=live)

else:  # pragma: no cover - the laptop path
    cyl_complex_update_H_increment = None  # type: ignore[assignment]


def cyl_complex_update_H_increment_kernel() -> Any:
    """The shipped launch-1 kernel object, or a refusal naming the missing import."""
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the cylindrical complex H->D pair needs Triton to launch; the predicate, "
            f"the lift checks and the plan builder answer without it "
            f"({_TRITON_IMPORT_ERROR})")
    return cyl_complex_update_H_increment


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_fused_hd_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                       probe: Any = None) -> "_coverage.Coverage":
    """May ONE run span ``update_H`` -> the electric withdraw -> ``step_D`` on this
    complex Dcyl run? The two certified halves' OWN predicates, in the driver's order,
    plus the seam clauses -- nothing weakened, every reason prefixed by its half."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    constitutive = _cx.cylindrical_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, probe=probe)
    if not constitutive.covered:
        reasons.extend(f"cylindrical complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = _cx.cylindrical_complex_curl_coverage(fields, pml, CURL_SUB_STEP, probe=probe)
    if not curl.covered:
        reasons.extend(f"cylindrical complex curl half: {reason}"
                       for reason in curl.reasons)

    reasons.extend(_withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3315-3316); this product declares "
            f"HOISTS_THE_WITHDRAW = False because it declares INSTALLABLE = False, "
            f"so _install_fused_pair's withdraw-hoist branch is unreachable for it "
            f"and nothing would perform the withdraw before the launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES))

    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every run")
    for name in IN_PLACE + CONSTITUTIVE_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    if getattr(fields, "scratch", None) is None:
        reasons.append(
            "fields carries no StepScratch: the increment and the scan output are the "
            "array path's own pooled slots (cyl_increment, cyl_prefix) and the two "
            "invariant row vectors are its cached constants")
    return _coverage.Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_cylindrical_fused_hd_pair(fields: Any, pml: Any, sources: Any = None,
                                      probe: Any = None) -> "_coverage.Coverage":
    return cylindrical_fused_hd_pair_coverage(fields, pml, sources, probe=probe)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalFusedHdPairPlan(ScratchWeldPairPlan):
    """TWO launches and ONE array-module scan for ``update_H`` and ``step_D`` on
    complex64 word pairs. The complex twin of
    :class:`.cylindrical_real_fused_hd_pair.CylindricalRealFusedHdPairPlan`; see it."""

    __slots__ = ("dtdx", "backward", "bcz", "m_class", "zero_rows", "expansion",
                 "increment_scalars", "four_dtdx", "xp", "scratch", "_b", "_targets",
                 "_aux", "_curl_coeff", "_h_coeff", "_imr_rows", "_weights", "_divisor",
                 "_kernel", "_curl_kernel", "_pointer", "_flat", "cumsum_calls",
                 "kernel_launches", "_dtype")

    replaces = REPLACES
    launches_per_run = LAUNCHES_PER_RUN
    kernel_launches_per_run = KERNEL_LAUNCHES_PER_RUN

    def __init__(self, shape: Sequence[int], dtdx: float, m: int,
                 accurate_fields_near_cylorigin: bool, bcz: int, expansion: int,
                 block: int, fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], xp: Any, scratch: Any,
                 kernel: Any = None, curl_kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps)
        if int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not")
        if scratch is None:
            raise ValueError(
                "this plan takes the increment and the scan output from the engine's "
                "StepScratch and binds its cached row vectors; a None pool is refused")
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernels' 2 * idx addressing")
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bcz = int(bcz)
        self.m_class = _cx.m_class(m)
        self.zero_rows = _cx.zero_rows(m, accurate_fields_near_cylorigin)
        self.expansion = int(expansion)
        self.increment_scalars = _cx.axis_increment_scalars(m, dtdx)
        self.four_dtdx = _cx.four_dtdx_scalar(dtdx)
        self.xp = xp
        self.scratch = scratch
        self._pointer = CupyPointer
        self._flat = _flat
        self._dtype = targets[0].dtype
        if str(self._dtype) != "complex64":
            raise ValueError(f"this plan steps complex64 word pairs, not {self._dtype}")
        self._b = tuple(CupyPointer(_word_view(a)) for a in flux)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._curl_coeff = tuple(CupyPointer(_flat(a)) for a in curl_coefficients)
        if len(self._curl_coeff) != 6 or len(constitutive_coefficients) != 6:
            raise ValueError(
                "this plan binds six curl coefficient vectors (kms/sinv per axis) and "
                "six constitutive ones (kps, kms per axis, interleaved), all integer")
        self._h_coeff = tuple(CupyPointer(_flat(a)) for a in constitutive_coefficients)
        # The i*m/r rows: the complex family's own host-built invariants (zero rows at
        # m = 0, where the M_ZERO arm never loads them; bound so the launch signature
        # is one signature).
        self._imr_rows = tuple(
            CupyPointer(_word_view(xp.ascontiguousarray(
                _cx.imr_coefficient_row(xp, _cx.SUB_STEPS[CURL_SUB_STEP]["targets"][index],
                                        sign, m, dtdx, self.shape[0],
                                        self._dtype).reshape(-1))))
            for index, _register, sign in _cx.IMR_TERMS[CURL_SUB_STEP])
        weights, divisor = _real.rderiv_vectors(xp, scratch, self.shape[0],
                                                targets[0].real.dtype)
        if int(weights.shape[0]) != self.shape[0] or int(divisor.shape[0]) != self.shape[0] - 1:
            raise ValueError(
                f"the cached rderiv vectors are {weights.shape} / {divisor.shape} for a "
                f"{self.shape[0]}-row grid")
        self._weights = weights
        self._divisor = divisor
        self._check_aliases(twins, flux, targets, auxiliaries, curl_coefficients,
                            constitutive_coefficients)
        self._kernel = kernel
        self._curl_kernel = curl_kernel
        self.cumsum_calls = 0
        self.kernel_launches = 0

    def _increment(self) -> Any:
        return self.scratch.take(INCREMENT_TAG, self.shape, self._dtype)

    def _prefix_out(self) -> Any:
        return self.scratch.take(PREFIX_TAG, self.shape, self._dtype)

    def _check_aliases(self, twins, flux, targets, auxiliaries,
                       curl_coefficients, constitutive_coefficients) -> None:
        def address(array: Any) -> Optional[int]:
            data = getattr(array, "data", None)
            pointer = getattr(data, "ptr", None)
            if pointer is not None:
                return int(pointer)
            interface = getattr(array, "__array_interface__", None)
            if isinstance(interface, dict):
                return int(interface["data"][0])
            return None

        outputs: Dict[int, str] = {}
        named = [(name, twins[name]) for name in ROTATED]
        named += [(IN_PLACE[index], array) for index, array in enumerate(targets)]
        named += [(IN_PLACE[3 + index], array)
                  for index, array in enumerate(auxiliaries)]
        named += [(INCREMENT_TAG, self._increment()), (PREFIX_TAG, self._prefix_out())]
        for label, array in named:
            key = address(array)
            if key is None:
                continue
            if key in outputs:
                raise ValueError(
                    f"outputs {outputs[key]} and {label} are the same allocation; "
                    f"one run would write both")
            outputs[key] = label
        inputs: List[Tuple[str, Any]] = [
            (CONSTITUTIVE_SOURCES[index], array) for index, array in enumerate(flux)]
        inputs += [(f"curl_coefficient{index}", array)
                   for index, array in enumerate(curl_coefficients)]
        inputs += [(f"constitutive_coefficient{index}", array)
                   for index, array in enumerate(constitutive_coefficients)]
        inputs += [("rderiv_weights", self._weights), ("rderiv_divisor", self._divisor)]
        for label, array in inputs:
            key = address(array)
            if key is not None and key in outputs:
                raise ValueError(
                    f"input {label} aliases output {outputs[key]}: the run would read a "
                    f"volume it is writing")

    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        pointer = self._pointer
        scratch = [pointer(_word_view(array)) for array in writes]
        prior = [pointer(_word_view(array)) for array in reads]

        # LAUNCH 1 -- update_H to scratch, plus the increment.
        kernel = (self._kernel if self._kernel is not None
                  else cyl_complex_update_H_increment_kernel())
        increment = self._increment()
        kernel[self._grid](
            *scratch, *prior, *self._b, pointer(_word_view(increment)),
            pointer(self._flat(self._weights)), pointer(self._flat(self._divisor)),
            *self._h_coeff,
            nx, ny, nz, self.n_elem,
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        self.kernel_launches += 1

        # THE SCAN -- stepping.py:1332-1333, on the pool's own slot. The oracle.
        prefix = self.xp.cumsum(increment, axis=0, out=self._prefix_out())
        self.cumsum_calls += 1

        # LAUNCH 2 -- the certified complex cylindrical curl, verbatim, over the NEW H,
        # with the argument list `CylindricalComplexCurlPlan.run` passes (its default
        # launch carries no num_warps override, and neither does this call).
        curl = (self._curl_kernel if self._curl_kernel is not None
                else _cx.cyl_complex_pml_curl_step)
        (minus_dtdx, inc_b) = self.increment_scalars
        curl[self._grid](
            *self._targets, *self._aux,
            scratch[0], scratch[1], scratch[2],       # the stepped Hx, Hy, Hz
            pointer(_word_view(prefix)), *self._imr_rows,
            *self._curl_coeff,
            nx, ny, nz, self.n_elem, self.dtdx,
            minus_dtdx, inc_b[0], inc_b[1], self.four_dtdx,
            BACKWARD=self.backward,
            BCZ=self.bcz,
            M_CLASS=self.m_class,
            ZERO_ROWS=self.zero_rows,
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )
        self.kernel_launches += 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"CylindricalFusedHdPairPlan(shape={self.shape}, m_class={self.m_class}, "
                f"zero_rows={self.zero_rows}, bcz={self.bcz}, "
                f"expansion={self.expansion}, block={self.block})")


def _sub_lattice_suffixes() -> Tuple[str, str]:
    """``(curl suffix, constitutive suffix)`` from the shipped tables, asserted equal."""
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl = SUB_STEPS[CURL_SUB_STEP]["suffix"]
    constitutive = ("_h" if _coverage.CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
                    ["half_integer"] else "")
    if curl != constitutive:
        raise AssertionError(
            f"step_D reads the {curl or 'integer'!r} PML sub-lattice and update_H the "
            f"{constitutive or 'integer'!r} one; both halves take one lattice here")
    if int(SUB_STEPS[CURL_SUB_STEP]["backward"]) != BACKWARD:
        raise AssertionError("launch.SUB_STEPS and this product disagree about BACKWARD")
    return curl, constitutive


def plan_cylindrical_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None, curl_kernel: Any = None,
        probe: Any = None) -> Optional[CylindricalFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused."""
    if not cylindrical_fused_hd_pair_coverage(fields, pml, sources, probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    grid = fields.grid
    kinds = resolve(grid, pml)
    return CylindricalFusedHdPairPlan(
        grid.shape, grid.dt / grid.dx, int(grid.m),
        bool(grid.accurate_fields_near_cylorigin),
        1 if kinds[2] == "metallic" else 0, expansion,
        DEFAULT_BLOCK if block is None else block,
        fields, twin_table(fields, ROTATED),
        [getattr(fields, name) for name in CONSTITUTIVE_SOURCES],
        [getattr(fields, name) for name in IN_PLACE[:3]],
        [getattr(fields, name) for name in IN_PLACE[3:]],
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(pml, f"{stem}_{axis}{constitutive_suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        grid.xp, fields.scratch,
        kernel=kernel, curl_kernel=curl_kernel, num_warps=num_warps,
    )


def plan_cylindrical_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], dtdx: float, m: int,
        accurate_fields_near_cylorigin: bool, bcz: int, expansion: int,
        fields: Any, xp: Any, scratch: Any, block: Optional[int] = None,
        kernel: Any = None, curl_kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route. No predicate runs, and
    ``expansion`` arrives as the constexpr the caller resolved (or forced)."""
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    return CylindricalFusedHdPairPlan(
        shape, dtdx, m, accurate_fields_near_cylorigin, bcz, int(expansion),
        DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [constitutive_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        xp, scratch, kernel=kernel, curl_kernel=curl_kernel, num_warps=num_warps,
    )
