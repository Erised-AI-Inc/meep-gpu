"""COMPLEX (``float2``) radial prefix scan on Metal — ``stepping.cylindrical_rderiv_prefix``
over complex64 storage, column-serial on the device.

THE PREREQUISITE FOR THE SIXTEEN ``cylindrical complex`` H->D ROWS. Every one of them
pays, on every ``step_D``, a device->host->device round trip so the prefix can be
scanned by NumPy (:class:`.cylindrical_complex.CylindricalComplexCurlPlan.refresh_prefix`).
:mod:`.cylindrical_complex` :598-627 states the licence for moving that scan onto the
device and declines to use it: *"moving the scan to the device would change the answer
unless the device scan were column-serial AND the engine's own array module agreed
with it, and the engine here is NumPy."* Both halves of that conjunction hold on this
backend — ``coverage.MIRRORED_HOST_MODULE`` is ``numpy``, ``numpy.cumsum`` IS a strictly
serial accumulation (measured, complex64 included: 0 differing words of 1,675,980 over
52 cases against a blocked scan that misses 1,389,353), and the real family's
``cyl_rderiv_prefix`` already reproduces the oracle at 0 of 257,712. This module is the
complex64 port of that kernel.

=========================================================================
THE ONE BUILD-CRITICAL FACT: THE DIVIDE SPELLING INVERTS BETWEEN THE FAMILIES
=========================================================================

The real template (:data:`.cylindrical_real._PREFIX_TEMPLATE`) spells the row divide as
``(w - prev) / divisor[i - 1]`` and its own comment insists on ``/`` — correctly:
float32 ``x / d`` and ``x * (1.0f / d)`` are different numbers on ~25% of words, and the
real gate catches the reciprocal at 6,373 words. A reader porting that template to
``float2`` will carry the ``/`` over, and it is WRONG here on the complex field:
``numpy``'s ``complex64 / float32`` is not two float32 divides. The float32 divisor is
cast to complex64 ``(d, +0.0)`` and the COMPLEX divide loop runs::

    rat = in2i / in2r                       # = +0.0f / d = +0.0f
    scl = 1.0f / (in2r + in2i * rat)        # = 1.0f / d
    out_r = (in1r + in1i * rat) * scl       # = (re + im*0) * (1/d)
    out_i = (in1i - in1r * rat) * scl       # = (im - re*0) * (1/d)

so the complex increment is the RECIPROCAL MULTIPLY ``z * (1.0f / d)``, and the true
componentwise divide ``float2 / float`` misses ~25% of increment words — and, because a
wrong increment poisons every row below it, ~90% of prefix words (measured 558,305 of
617,680 on the corpus extents). Spelled here as :func:`c_div_real` and carried as an
armed mutation in the gate, because the real family's comment says the opposite for a
correct reason and this is the one line a careful reader will get wrong.

THE SIGNED-ZERO REFINEMENT, measured on the host before the kernel was written. The
loop above is NOT bit-identical to the plain reciprocal multiply on every INCREMENT:
``re + im*rat`` turns a ``-0.0`` real part into ``+0.0`` whenever ``im >= 0`` (1,027 of
8,192 words on a signed-zero-rich table; 0 of 8,192 on random data), and the same holds
for the weight multiply (``re*w - im*0`` vs ``re*w``: 1,007 of 8,192). Neither reaches
the prefix. The accumulator starts at an exact ``+0.0`` and IEEE addition never
produces ``-0.0`` from a ``+0.0`` accumulator (``+0 + -0 = +0``; an exact cancellation
rounds to ``+0``), so the sign of an exactly-zero increment is absorbed before it is
stored: the oracle's prefix over a whole ``+-0`` lattice is ``+0.0`` in every word
(160/160 and 52,500/52,500 measured). The kernel therefore ships the plain reciprocal
multiply, which the verdict named and which is bit-identical on the OUTPUT, and the
gate carries the literal-loop spelling and the componentwise multiply as MEASURED
EQUIVALENCES on the prefix rather than as beliefs. A consumer that ever read the
INCREMENT rather than the prefix would have to revisit this; nothing in the package
does.

=========================================================================
WHAT IS CARRIED FROM THE REAL SCAN, AND WHAT IS NOT
=========================================================================

* ``ir0``, the wall row and the row vectors are the REAL FAMILY'S TABLES, imported:
  :data:`.cylindrical_real.PREFIX_IR0`, :data:`.cylindrical_real.PREFIX_WALL_ROW`,
  :func:`.cylindrical_real.prefix_row_vectors` (which itself calls
  ``stepping._cylindrical_rderiv_weights``). ``cylindrical_rderiv_prefix`` builds the
  weights from ``f_p.real.dtype``, which is float32 for complex64 storage, so the two
  families bind the SAME two float32 vectors. Restating either table here would be a
  second place for a half-cell error to live.
* The weighted term is ``f_p * weights`` with the FIELD ON THE LEFT (stepping.py:1313,
  :1327), through the certified :func:`templates.complex_helpers` ``c_mul_field_left``
  for the probe-bound expansion arm — the same helper the complex curl uses. For a
  real coefficient the ``FMA_V1`` and ``NAIVE`` arms are the same bits (the cross term
  is an exact zero), and the gate measures both arms rather than asserting it.
* The accumulation is STRICTLY SERIAL from r = 0 upward, ``acc = acc + inc`` — the
  :data:`templates.COLUMN_SERIAL_SCAN` shape with a ``float2`` accumulator, one thread
  per (phi, z) column. A blocked scan is the gate's null control and must differ.
* Row 0 is written as an exact ``(+0.0, +0.0)`` (stepping.py:1314 ``zeros_like``;
  :1329), never assumed: a ``-0.0`` there is a different word.
* The B side's ZERO WALL ROW is a source specialisation exactly as on the real side:
  the scan covers ``nr + 1`` rows over an ``nr``-row source and the top row reads a
  literal complex zero (stepping.py:360). The D side scans ``Hy`` raw.
* EIGHT BINDINGS, the real scan's layout with ``float2`` in slots 0 and 1.
* The contraction guard is mandatory and is reached through :mod:`.templates`, the
  package's one spelling.

NO PLAN CLASS AND NO ARM. This module is a kernel and its emitter. The product that
consumes it — the two-launch complex H->D pair, launch 1 = the pointwise complex H
constitutive plus this scan over the recomputed ``Hy``, launch 2 = the certified
complex curl — is a separate module with its own gate; this one certifies the scan
alone against ``stepping.cylindrical_rderiv_prefix`` CALLED, never re-derived.

NO TORCH AT MODULE LEVEL, as everywhere in this package.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ..triton_kernels.launch import SUB_STEPS
from . import shaders, templates
from .cylindrical_real import PREFIX_IR0, PREFIX_WALL_ROW, prefix_row_vectors, scan_rows
from .device import compile_source

FAMILY = "cylindrical_complex_scan"

#: The kernel entry point's name — the real scan's with ``complex`` in it, so a
#: compiled library, a fingerprint label and a gate row cannot be mistaken for the
#: float32 kernel's.
KERNEL = "cyl_complex_rderiv_prefix"

#: out, src, weights, divisor, nx, ny, nz, n_cols — the real scan's eight, with the
#: two volumes ``float2``. Pinned by the suite against the emitted text.
BINDINGS = 8

#: WELDED. Empty means ``fingerprints.json`` carries
#: ``metal_cylindrical_complex_scan_device_gate``, minted from the released artifact
#: by ``parity/meep_gpu/mint_metal_weld.py``.
WELD_OWED: str = ""

__all__ = [
    "BINDINGS", "FAMILY", "KERNEL", "PREFIX_IR0", "PREFIX_WALL_ROW", "WELD_OWED",
    "COLUMN_SCAN_LOOP", "compile_cylindrical_complex_prefix",
    "cylindrical_complex_prefix_source", "enumerate_cylindrical_complex_scan_sources",
    "prefix_row_vectors", "scan_arguments", "scan_rows", "source_reads",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: ``numpy``'s ``complex64 / float32`` on the prefix: the reciprocal multiply. See the
#: module docstring for why this is NOT ``/`` here while it MUST be ``/`` on the real
#: scan, and for the signed-zero refinement the accumulator absorbs.
_DIVIDE_HELPER = r"""// --- complex64 / float32, as the array path's complex divide loop rounds it -----
// numpy casts the float32 divisor to (d, +0) and runs its complex divide:
//   rat = 0/d, scl = 1/(d + 0*rat), out = ((re + im*rat)*scl, (im - re*rat)*scl)
// which is z * (1.0f / d) on every word that reaches the prefix. NOT `z / d`: the
// componentwise true divide is a different float32 number on ~25% of increments and,
// through the running sum, on ~90% of prefix words. The real scan's `/` is right FOR
// THE REAL SCAN and wrong here; the gate arms the `/` spelling and it must be caught.
static inline float2 c_div_real(float2 z, float d) {
    return z * (1.0f / d);
}
"""

#: The serial column loop, shared by the standalone kernel below and by any fused
#: kernel whose column leader scans a value it recomputes rather than loads: the
#: ``__SRC0__``/``__SRCI__`` placeholders are the row-0 and row-``i`` COMPLEX source
#: expressions, exactly as :data:`.cylindrical_real._PREFIX_TEMPLATE` parameterises
#: them. Everything that decides a bit is in this fragment and nowhere else.
COLUMN_SCAN_LOOP = r"""    // increment[0] is an exact complex +0.0 (stepping.py:1285 zeros_like / :1300)
    // and the cumsum's first output IS that zero. Written, not assumed: a -0.0 word
    // here is a different word, and this family compares words.
    float2 acc = float2(0.0f, 0.0f);
    out[base] = acc;
    // weighted = f_p * weights, FIELD LEFT, exactly as stepping.py:1284/:1298 spells
    // it, through the certified field-left complex product.
    float2 prev = c_mul_field_left(__SRC0__, weights[0]);
    for (int i = 1; i < nxi; ++i) {
        int o = i * nyz + base;
        float2 w = c_mul_field_left(__SRCI__, weights[i]);
        // stepping.py:1286/:1301-1302 — the difference of two ALREADY-WEIGHTED rows,
        // then numpy's complex-by-real divide, which is the reciprocal multiply.
        float2 inc = c_div_real(w - prev, divisor[i - 1]);
        // STRICTLY SERIAL: numpy.cumsum is out[i] = out[i-1] + in[i], both planes.
        acc = acc + inc;
        out[o] = acc;
        prev = w;
    }"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 1285->1314, 1300->1329, 1284->1313, 1298->1327, 1286->1315, 1301-1302->1330-1331

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

__DIVIDE__

kernel void cyl_complex_rderiv_prefix(
    device float2*       out     [[buffer(0)]],
    device const float2* src     [[buffer(1)]],
    device const float*  weights [[buffer(2)]],
    device const float*  divisor [[buffer(3)]],
    constant uint&       nx      [[buffer(4)]],
    constant uint&       ny      [[buffer(5)]],
    constant uint&       nz      [[buffer(6)]],
    constant uint&       n_cols  [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    // One thread per (phi, z) column. The dispatch is sized from `out`, which is
    // nx times wider, so most threads return here; that is a stated throughput cost
    // of this kernel and not a correctness one.
    if (idx >= n_cols) { return; }
    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int k = int(idx) % nzi;
    int j = int(idx) / nzi;
    if (j >= nyi) { return; }
    int nyz = nyi * nzi;
    int base = j * nzi + k;
__LOOP__
}
"""


def source_reads(sub_step: str) -> Tuple[str, str]:
    """The ``(row 0, row i)`` complex source expressions for one sub-step.

    The B side's zero wall row is a SOURCE specialisation: the scan covers ``nr + 1``
    rows over an ``nr``-row source and the top row reads a literal complex zero, the
    value ``stepping.py:360`` assigns. Spelling it as a shortened loop instead would
    drop the wall row's own prefix entry, which is the value Bz's last-row forward
    difference reads. The D side reads the source raw.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if PREFIX_WALL_ROW[sub_step]:
        return ("(0 < nxi - 1 ? src[base] : float2(0.0f, 0.0f))",
                "(i < nxi - 1 ? src[i * nyz + base] : float2(0.0f, 0.0f))")
    return "src[base]", "src[i * nyz + base]"


def cylindrical_complex_prefix_source(sub_step: str, expansion: str,
                                      contract: str = shaders.CONTRACT_OFF) -> str:
    """The complex radial-scan source for one sub-step and one expansion arm.

    ``expansion`` is the PROBE-MEASURED complex-multiply arm the cylindrical complex
    family binds (``cylindrical_complex.expansion_from_probe``); it selects which
    ``c_mul_field_left`` text is emitted. For the real coefficient this kernel
    multiplies by, the two arms are the same bits, and the gate measures that rather
    than this docstring asserting it.
    """
    src0, srci = source_reads(sub_step)
    loop = templates.substitute(COLUMN_SCAN_LOOP, {"__SRC0__": src0, "__SRCI__": srci})
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__DIVIDE__": _DIVIDE_HELPER,
        "__LOOP__": loop,
    })


def compile_cylindrical_complex_prefix(sub_step: str, expansion: str,
                                       contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised complex radial-scan entry point for one sub-step."""
    return getattr(compile_source(
        cylindrical_complex_prefix_source(sub_step, expansion, contract)), KERNEL)


def enumerate_cylindrical_complex_scan_sources(
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation this module can emit — TWO per arm, keyed by a stable label."""
    return {f"{KERNEL}/{sub_step}/{expansion}":
            cylindrical_complex_prefix_source(sub_step, expansion, contract)
            for sub_step in ("step_B", "step_D")}


def scan_arguments(out: Any, src: Any, weights: Any, divisor: Any,
                   scan_shape: Sequence[int]) -> Tuple[Any, ...]:
    """The positional tuple the kernel takes, in signature order.

    ONE spelling of the argument order for the gate, the suite and any plan that
    launches this scan, so the eight bindings cannot be permuted at a call site.
    ``scan_shape`` is the SCAN volume's shape — ``(nr + 1, ny, nz)`` on B,
    ``(nr, ny, nz)`` on D — and ``n_cols`` is ``ny * nz`` of it.
    """
    rows, ny, nz = (int(n) for n in scan_shape)
    return (out, src, weights, divisor, rows, ny, nz, ny * nz)
