"""COMPLEX (``float2``) radial scan on Metal — ``stepping.cylindrical_rderiv_difference``
over complex64 storage, column-serial on the device.

MEEP sums ``f_rderiv_int`` per part (its ``DOCMP`` loop), and the array path forms every
step of the radial sum per part (:mod:`~meep_gpu.part_products`): the weighted term
``f_p * weights``, the difference of two weighted rows and the product with the
multiplied ``rinv`` are each one rounded real product or sum per part, and the
accumulation is a complex add, which is per part. So the complex scan IS the real scan
of :mod:`.cylindrical_real` with ``float2`` registers: :func:`column_scan_loop` derives it
from that module's loop by the declared type edits of :data:`FLOAT2_EDITS` and nothing
else (``float2 * float`` is one rounded product per part on this backend, as the real
scan's ``float * float`` is one). The chunk starts, the row vectors, ``ir0`` and the B
side's zero wall row are the real family's, imported.

NO PLAN CLASS AND NO ARM. This module is a kernel and its emitter. The product that
consumes it — the two-launch complex H->D pair (:mod:`.cylindrical_complex_fused_hd_pair`),
whose leading launch scans the recomputed ``Hy`` — carries its own gate; this one
certifies the scan alone against ``stepping.cylindrical_rderiv_difference`` CALLED,
never re-derived.

NO TORCH AT MODULE LEVEL, as everywhere in this package.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ..triton_kernels.launch import SUB_STEPS
from . import cylindrical_real as _real
from . import shaders, templates
from .cylindrical_real import PREFIX_IR0, PREFIX_WALL_ROW, prefix_row_vectors, scan_rows
from .device import compile_source

FAMILY = "cylindrical_complex_scan"

#: The kernel entry point's name — the real scan's with ``complex`` in it, so a
#: compiled library, a fingerprint label and a gate row cannot be mistaken for the
#: float32 kernel's.
KERNEL = "cyl_complex_rderiv_difference"

#: out, src, weights, rinv, starts, nx, ny, nz, n_cols — the real scan's nine, with the
#: two volumes ``float2``. Pinned by the suite against the emitted text.
BINDINGS = 9

#: WELDED. Empty means ``fingerprints.json`` carries
#: ``metal_cylindrical_complex_scan_device_gate``, minted from the released artifact
#: by ``parity/meep_gpu/mint_metal_weld.py``.
WELD_OWED: str = ""

#: The edits that turn the real scan's column loop into the complex one, each applied
#: exactly once per loop where it occurs (:func:`column_scan_loop`). TYPES AND ZERO
#: LITERALS ONLY: not one operator, operand or paren moves.
FLOAT2_EDITS: Tuple[Tuple[str, str], ...] = (
    ("    out[base] = 0.0f;\n", "    out[base] = float2(0.0f, 0.0f);\n"),
    ("    float w_prev = ", "    float2 w_prev = "),
    ("    float acc = 0.0f;", "    float2 acc = float2(0.0f, 0.0f);"),
    ("{ acc = 0.0f; owner = s; }", "{ acc = float2(0.0f, 0.0f); owner = s; }"),
    ("        float w = ", "        float2 w = "),
    (" : 0.0f) * weights[g + 1];", " : float2(0.0f, 0.0f)) * weights[g + 1];"),
    ("        float inc = ", "        float2 inc = "),
    ("        float nxt = ", "        float2 nxt = "),
)

__all__ = [
    "BINDINGS", "FAMILY", "FLOAT2_EDITS", "KERNEL", "PREFIX_IR0", "PREFIX_WALL_ROW",
    "WELD_OWED", "column_scan_loop", "compile_cylindrical_complex_scan",
    "cylindrical_complex_scan_source", "enumerate_cylindrical_complex_scan_sources",
    "prefix_row_vectors", "scan_arguments", "scan_rows", "source_reads",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: The two source reads of the real loop, per sub-step: row 0 and row ``g`` (the B
#: side's row ``g`` is ``g + 1``, guarded by the wall row's literal zero).
_SOURCE_READS: Dict[str, Tuple[str, str]] = {
    "step_B": ("src[base]", "src[(g + 1) * nyz + base]"),
    "step_D": ("src[base]", "src[g * nyz + base]"),
}


def source_reads(sub_step: str) -> Tuple[str, str]:
    """The ``(row 0, row g)`` source reads of one sub-step's column loop, as spelled.

    A fused kernel whose column leader scans a value it RECOMPUTES replaces exactly
    these two reads (and nothing else) in :func:`column_scan_loop`'s text.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    return _SOURCE_READS[sub_step]


def column_scan_loop(sub_step: str) -> str:
    """The complex column loop: the real scan's, with :data:`FLOAT2_EDITS` applied.

    Each edit that names text the real loop carries must land exactly once, so a real
    loop that changed under this family raises rather than emitting a mixed-type scan.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    loop = _real._SCAN_LOOPS[sub_step]  # noqa: SLF001 - the one spelling of the loop
    for old, new in FLOAT2_EDITS:
        count = loop.count(old)
        if count > 1:
            raise AssertionError(
                f"the real {sub_step} scan loop carries {old.strip()!r} {count} times; "
                f"the complex edit is ambiguous")
        loop = loop.replace(old, new)
    for read in source_reads(sub_step):
        if read not in loop:
            raise AssertionError(
                f"the real {sub_step} scan loop no longer reads {read!r}")
    if "    float " in loop or " 0.0f;" in loop.replace("float2(0.0f, 0.0f);", ""):
        raise AssertionError(
            f"the complex {sub_step} scan loop still declares a float register or "
            f"assigns a float zero: {loop!r}")
    return loop


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void cyl_complex_rderiv_difference(
    device float2*       out     [[buffer(0)]],
    device const float2* src     [[buffer(1)]],
    device const float*  weights [[buffer(2)]],
    device const float*  rinv    [[buffer(3)]],
    device const int*    starts  [[buffer(4)]],
    constant uint&       nx      [[buffer(5)]],
    constant uint&       ny      [[buffer(6)]],
    constant uint&       nz      [[buffer(7)]],
    constant uint&       n_cols  [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    // One thread per (phi, z) column. The dispatch is sized from `out`, which is
    // nx times wider, so most threads return here.
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


def cylindrical_complex_scan_source(sub_step: str,
                                    contract: str = shaders.CONTRACT_OFF) -> str:
    """The complex radial-scan source for one sub-step."""
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__LOOP__": column_scan_loop(sub_step),
    })


def compile_cylindrical_complex_scan(sub_step: str,
                                     contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised complex radial-scan entry point for one sub-step."""
    return getattr(compile_source(
        cylindrical_complex_scan_source(sub_step, contract)), KERNEL)


def enumerate_cylindrical_complex_scan_sources(
        contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation this module can emit — TWO, keyed by a stable label."""
    return {f"{KERNEL}/{sub_step}":
            cylindrical_complex_scan_source(sub_step, contract)
            for sub_step in ("step_B", "step_D")}


def scan_arguments(out: Any, src: Any, weights: Any, rinv: Any, starts: Any,
                   shape: Sequence[int]) -> Tuple[Any, ...]:
    """The positional tuple the kernel takes, in signature order.

    ONE spelling of the argument order for the gate, the suite and any plan that
    launches this scan. ``shape`` is the TARGET (and source) volume's ``(nr, ny, nz)``;
    the B side's wall row is the loop's literal, not a stored row.
    """
    rows, ny, nz = (int(n) for n in shape)
    return (out, src, weights, rinv, starts, rows, ny, nz, ny * nz)
