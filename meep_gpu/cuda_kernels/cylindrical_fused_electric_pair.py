"""The Dcyl COMPLEX hand-CUDA fused pair: ``step_D`` -> ``zero_metal_D`` -> ``update_E``.

THE CELL: ``D_to_E (cuda_cyl_complex/cylindrical complex, cuda_complex/complex)``. Both
halves are lifted from the emitted texts of their singles -- the cylindrical complex curl
(``cylindrical_complex_kernels.cylindrical_complex_source("step_D")``) and the complex
constitutive (``complex_emitter.complex_source("update_E")``) -- and this module
introduces no arithmetic of its own: every edit is a register capture, a seam, a rename or
the own-cell hoist, listed in :data:`LIFT_EDITS`. The derivations are
:mod:`.cylindrical_fused_magnetic_pair`'s, imported: the two pairs lift the same helpers.

WHAT ONE LAUNCH PERFORMS. :data:`REPLACES` is ``step_D`` -> ``zero_metal_D`` ->
``update_E``. The radial scan is the curl's own first launch over the current Hp; it hands
the difference in as ``rad``. The two mirror fills are absent by refusal.

THE REGISTER FOLLOWS EVERY STORE OF D: the PML form, the i*m/r pass on targets 0 and 2,
the per-m axis rules (|m| = 1: Dz on the axis row; |m| >= 2: the family on rows
``[0, zero_rows)``; m = 0: Dz's double ``4*Courant`` rule and Dp's zero) and the wall pass
``zero_metal_D`` (the OFF-DIAGONAL table, two components per walled axis), each a register
clear or capture beside the certified store. The m = 0 rule adds into the register the
lane already holds, the word it stored.

THE OWN-CELL HOIST, THE SEAM, THE ALIASING AND THE DEPOSIT REPAIR are the magnetic pair's,
on the D side: ``f_w_E`` and E behind the dsigw flag, ``fu_D`` behind the dsigu flag, D
unguarded; D bound exactly once; ``inv_eps`` bound three times and NOT ``__restrict__``
(an isotropic run hands one allocation three times). The two halves collide on
``kms_x/y/z`` (integer for the D curl, half-integer for ``update_E``), so the constitutive
group is ``kms_half_*``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

from . import bounds_block
from . import compile_cache
from . import complex_emitter
from . import cylindrical_complex_kernels
from .coverage import covers_real_pml_complex_constitutive
from .cylindrical_coverage import covers_pml_cylindrical_complex_curl
from .cylindrical_fused_magnetic_pair import (constitutive_prelude, curl_prelude,
                                              fused_prelude, line_starting, replace_once,
                                              split_body)
from .cylindrical_prefix import radial_difference
from .in_seam_coverage import zero_metal_axes

try:
    from . import complex_pml_kernels
except ImportError:  # a host with no CuPy: only the table builder is unavailable
    complex_pml_kernels = None

from .. import deposit_repair as _deposit_repair


# =============================================================================
# THE PARTITION -- PLAIN ASSIGNMENTS, NO ANNOTATION
# =============================================================================

#: The entry point. The record that certified its 0.9.2 text is re-cut in the combined
#: certification round (bit_parity_0_10 CERT-3).
CERTIFIED_KERNELS = (
    "fused_electric_pair_pml_cyl_complex",
)
UNCERTIFIED_KERNELS = {}

#: The bracket that repairs an in-seam electric deposit (``fused_pairs._install_fused_pair``).
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_cylindrical_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_electric_pair_pml_cyl_complex"

#: The driver passes ONE launch of this kernel performs, in driver order.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: Every line of certified device text this file did not lift verbatim, with the reason.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ cf pml_apply_reg_pre(",
     "why": "the magnetic pair's derivation (cylindrical_fused_magnetic_pair."
            "pml_apply_reg_pre_source): the stored value returned, the two own-cell "
            "words arguments, checked against the certified statements."},
    {"line": "__device__ __forceinline__ void fold_apply(",
     "became": "__device__ __forceinline__ cf fold_apply_reg(",
     "why": "the register follows the i*m/r pass's store."},
    {"line": "__device__ __forceinline__ void constitutive_apply(",
     "became": "__device__ __forceinline__ void constitutive_apply_pre(",
     "why": "the own-cell hoist: prev and the old E are arguments preloaded under the "
            "dsigw flag."},
    {"line": "    cf updated = dsigw ? cf_add(cf_load(f, idx), cf_sub(mul_coefficient_left("
             "kps, src), mul_coefficient_left(kms, prev)))",
     "became": "    cf updated = dsigw ? cf_add(fcur, cf_sub(mul_coefficient_left("
               "kps, src), mul_coefficient_left(kms, prev)))",
     "why": "the accumulation's one load of the old E is the argument the hoist "
            "preloaded under the dsigw flag (CONSTITUTIVE_LOADS: fcur is "
            "cf_load(f, idx)); the expression is otherwise the certified one."},
    {"line": "    <the curl body's chunk-record block> + <update_E's>",
     "became": "    <ONE block, bounds_block.bounds_block(both entry sets, presence xyz)>",
     "why": "one decode of the record for both halves (DESIGN.md D10)."},
    {"line": "        pml_apply(Dx, fu_Dx, idx, curl, ...);",
     "became": "        d0 = pml_apply_reg_pre(Dx, fu_Dx, idx, curl, ..., pre_fu_0, pre_d_0);",
     "why": "capture the register and take the preloaded words."},
    {"line": "            fold_apply(Dx, fu_Dx, idx, inc, ...);",
     "became": "            d0 = fold_apply_reg(Dx, fu_Dx, idx, inc, ...);",
     "why": "the register follows the i*m/r pass's store (targets 0 and 2)."},
    {"line": "        cf_store(Dz, idx, cf_zero());   (the |m| = 1 and |m| >= 2 axis rules)",
     "became": "        d2 = cf_zero(); cf_store(Dz, idx, d2);",
     "why": "the register follows the axis rules' stores; the fu stores keep their "
            "spelling and get no register (the constitutive half reads none)."},
    {"line": "        cf_store(Dz, idx, cf_add(cf_load(Dz, idx), double_product(four_courant, "
             "cf_load(Hy, idx))));",
     "became": "        d2 = cf_add(d2, double_product(four_courant, cf_load(Hy, idx))); "
               "cf_store(Dz, idx, d2);",
     "why": "the m = 0 rule adds into the register, which holds the word this lane "
            "stored for Dz (the reload it replaces reads that same word)."},
    {"line": "        array[_face(axis, 0)] = 0   (stepping._zero_metal)",
     "became": "    if (wall_x && i == 0) { d1 = cf_zero(); cf_store(Dy, idx, d1); ... }",
     "why": "zero_metal_D carried on the registers after the axis rules (the driver's "
            "order): the OFF-DIAGONAL, two components per walled axis."},
    {"line": "    cf s0 = cf_load(g0, idx);",
     "became": "    cf s0 = d0;",
     "why": "THE SEAM: the register instead of a reload of D. The inverse-permittivity "
            "products that follow are the certified lines, read and not edited."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i], in_h_x);",
     "became": "    constitutive_apply_pre(f0, w0, idx, s0, pre_w_0, pre_e_0, kps_x[i], "
               "kms_half_x[i], in_h_x);",
     "why": "the sub-lattice rename (kms_half_*) and the preloaded words."},
    {"line": "    cf d2 = cf_zero();",
     "became": "    cf d2 = cf_zero();\n    cf pre_w_0 = in_h_x ? cf_load(w0, idx) : cf_zero();"
               "   // ... twelve guarded lines",
     "why": "THE OWN-CELL HOIST, the magnetic pair's on the D side."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body", "certified_curl_body",
    "covers_cylindrical_fused_electric_pair", "cylindrical_fused_electric_pair_prelude",
    "cylindrical_fused_electric_pair_source", "cylindrical_fused_electric_pair_tables",
    "device_sources", "hoisted_loads", "inverse_epsilon_bindings",
    "launch_cylindrical_fused_electric_pair", "resolve_launch_arguments",
    "run_cylindrical_fused_electric_pair", "union_bounds_block", "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================

_DECODE_END = "    int i = idx / (ny * nz);\n"

#: The three carried registers and the D targets they follow, in target order.
_CARRIED: Tuple[str, ...] = ("0", "1", "2")
_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")

#: ``stepping._zero_metal``'s OFF-DIAGONAL table for the D side, as (flag, coordinate,
#: the two component indices wiped there): a D component has Yee shift 0 on the OTHER
#: two axes (``fields.IYEE_SHIFTS``), so on a Dcyl grid (phi, z) for Dr, (r, z) for Dp
#: and (r, phi) for Dz.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("wall_x", "i", ("1", "2")),
    ("wall_y", "j", ("0", "2")),
    ("wall_z", "k", ("0", "1")),
)

#: ``stepping._cylindrical_axis_zero_D``'s three branches as the curl emits them, and
#: their carried rewrites. Anchored as WHOLE BLOCKS: ``cf_store(Dz, idx, cf_zero());``
#: appears in two of them, and the fu stores must visibly survive untouched.
_AXIS_TAIL_M1 = (
    "    if (m_class == 1 && i == 0) {\n"
    "        cf_store(Dz, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M1_CARRIED = (
    "    if (m_class == 1 && i == 0) {\n"
    "        d2 = cf_zero(); cf_store(Dz, idx, d2);\n"
    "    }\n")
_AXIS_TAIL_M2 = (
    "    if (m_class == 2 && i < zero_rows) {\n"
    "        cf_store(Dx, idx, cf_zero());\n"
    "        cf_store(Dy, idx, cf_zero());\n"
    "        cf_store(Dz, idx, cf_zero());\n"
    "        cf_store(fu_Dx, idx, cf_zero());\n"
    "        cf_store(fu_Dy, idx, cf_zero());\n"
    "        cf_store(fu_Dz, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M2_CARRIED = (
    "    if (m_class == 2 && i < zero_rows) {\n"
    "        d0 = cf_zero(); cf_store(Dx, idx, d0);\n"
    "        d1 = cf_zero(); cf_store(Dy, idx, d1);\n"
    "        d2 = cf_zero(); cf_store(Dz, idx, d2);\n"
    "        cf_store(fu_Dx, idx, cf_zero());\n"
    "        cf_store(fu_Dy, idx, cf_zero());\n"
    "        cf_store(fu_Dz, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M0 = (
    "    if (m_class == 0 && i == 0) {\n"
    "        cf_store(Dz, idx, cf_add(cf_load(Dz, idx), double_product(four_courant, "
    "cf_load(Hy, idx))));\n"
    "        cf_store(Dy, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M0_CARRIED = (
    "    if (m_class == 0 && i == 0) {\n"
    "        // THE AXIS RULE, CARRIED: d2 holds the word this lane stored for Dz.\n"
    "        d2 = cf_add(d2, double_product(four_courant, cf_load(Hy, idx)));\n"
    "        cf_store(Dz, idx, d2);\n"
    "        d1 = cf_zero(); cf_store(Dy, idx, d1);\n"
    "    }\n")

#: The fused entry point. The ONLY hand-written device text in this module, and it is a
#: signature. D appears EXACTLY ONCE.
_SIGNATURE = r'''
extern "C" __global__ void fused_electric_pair_pml_cyl_complex(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 D.
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    // The radial difference the scan wrote, and the two i*m/r rinv rows.
    const float* __restrict__ rad,
    const float* __restrict__ imr0, const float* __restrict__ imr2,
    // E and its constitutive history, the constitutive template's own names.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    // NOT __restrict__: an isotropic run hands one allocation three times. Float32
    // volumes indexed by the COMPLEX CELL index.
    const float* inv_eps_0, const float* inv_eps_1, const float* inv_eps_2,
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx, double plus_courant, double four_courant,
    // The D curl's INTEGER split-field coefficients.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The E constitutive's HALF-INTEGER coefficients, its kms renamed kms_half_*.
    const float* __restrict__ kps_x, const float* __restrict__ kms_half_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_half_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_half_z,
    int bc_x, int bc_y, int bc_z,
    int m_class, int zero_rows,
    // zero_metal_D's three walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z,
    // THE CHUNK RECORD (chunk_map.chunk_bounds_record), int32[25].
    const int* __restrict__ cb
) {
'''


def cylindrical_fused_electric_pair_prelude() -> str:
    """The fused kernel's prelude: the magnetic pair's derived prelude
    (the same three helpers, the same derivations)."""
    return fused_prelude()


#: The chunk-record entries each lifted body decodes, as ``(entries, presence)``.
CURL_BLOCK: Tuple[Any, Tuple[int, ...]] = (
    tuple(bounds_block.curl_entries(("Dx", "Dy", "Dz"))), (0, 1, 2))
CONSTITUTIVE_BLOCK: Tuple[Any, Tuple[int, ...]] = (
    tuple(bounds_block.constitutive_entries(("Ex", "Ey", "Ez"))), ())


def union_bounds_block() -> str:
    """The fused kernel's ONE chunk-record block: both bodies' entries, one decode."""
    return bounds_block.bounds_block(CURL_BLOCK[0] + CONSTITUTIVE_BLOCK[0],
                                     CURL_BLOCK[1] + CONSTITUTIVE_BLOCK[1])


def curl_flags() -> Tuple[str, ...]:
    """Each D target's dsigu membership flag, in target order (the curl's own)."""
    return complex_emitter.CURL_FLAGS[True]


def constitutive_flags() -> Tuple[str, ...]:
    """Each E component's dsigw membership flag, x/y/z."""
    return complex_emitter.CONSTITUTIVE_FLAGS["E"]


def hoisted_loads() -> str:
    """The twelve own-cell loads (24 words), each behind the certified load's guard."""
    lines = [
        "    // --- the own-cell hoist: every word this thread reads at idx, issued",
        "    // before the kernel's first store, behind the certified load's guard",
        "    // (f_w_E and E behind the dsigw flag; fu behind the dsigu flag). ---",
    ]
    for register, flag in zip(_CARRIED, constitutive_flags()):
        lines.append(f"    cf pre_w_{register} = {flag} ? cf_load(w{register}, idx) : cf_zero();")
        lines.append(f"    cf pre_e_{register} = {flag} ? cf_load(f{register}, idx) : cf_zero();")
    for register, target, flag in zip(_CARRIED, _TARGETS, curl_flags()):
        lines.append(f"    cf pre_fu_{register} = {flag} ? cf_load(fu_{target}, idx) : cf_zero();")
        lines.append(f"    cf pre_d_{register} = cf_load({target}, idx);")
    return "\n".join(lines) + "\n"


def _certified_source() -> Any:
    """Refuse the emitter BY NAME where the cylindrical text is unreachable.

    THE SPLICE IS THE LIFT, so a missing half is not a degraded emit -- there is nothing
    to emit. Raising here keeps that fact one frame from the caller instead of surfacing
    as ``'NoneType' object has no attribute`` inside a string operation.
    :func:`covers_cylindrical_fused_electric_pair` needs none of it and still answers.
    """
    if cylindrical_complex_kernels is None:
        raise RuntimeError(
            "cylindrical_complex_kernels is not available, so there is no certified "
            "cylindrical text to splice. covers_cylindrical_fused_electric_pair needs "
            "none of it and still answers")
    return cylindrical_complex_kernels


def certified_curl_body() -> str:
    """``cyl_step_D_pml_complex``'s body, lifted: registers captured at every store of D,
    the union record block, the hoist."""
    body = split_body(_certified_source().cylindrical_complex_source("step_D"),
                      curl_prelude(), "cylindrical_complex_source('step_D')")
    body = bounds_block.strip_bounds_block(body, "the certified curl body", *CURL_BLOCK)
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END, "\n",
        union_bounds_block(),
        "    // The three carried registers, declared here: each component below is a\n"
        "    // braced scope, and a value declared inside one does not outlive it.\n"
        "    cf d0 = cf_zero();\n"
        "    cf d1 = cf_zero();\n"
        "    cf d2 = cf_zero();\n",
        hoisted_loads(),
        tail,
    ))
    for register, target in zip(_CARRIED, _TARGETS):
        prefix = f"        pml_apply({target}, fu_{target}, idx, curl, "
        call = line_starting(body, prefix, f"{target}'s pml_apply")
        if not call.endswith(");"):
            raise AssertionError(f"{target}'s pml_apply no longer closes on one line")
        body = body.replace(call, call.replace(
            f"pml_apply({target},", f"d{register} = pml_apply_reg_pre({target},", 1)[:-2]
            + f", pre_fu_{register}, pre_d_{register});", 1)
    for register, target in (("0", "Dx"), ("2", "Dz")):
        body = replace_once(body, f"            fold_apply({target}, fu_{target}, idx, inc, ",
                            f"            d{register} = fold_apply_reg({target}, fu_{target}, "
                            f"idx, inc, ", f"{target}'s i*m/r ladder call")
    body = replace_once(body, _AXIS_TAIL_M1, _AXIS_TAIL_M1_CARRIED,
                        "the |m| = 1 branch of stepping._cylindrical_axis_zero_D")
    body = replace_once(body, _AXIS_TAIL_M2, _AXIS_TAIL_M2_CARRIED,
                        "the |m| >= 2 branch of stepping._cylindrical_axis_zero_D")
    body = replace_once(body, _AXIS_TAIL_M0, _AXIS_TAIL_M0_CARRIED,
                        "the m = 0 branch of stepping._cylindrical_axis_zero_D")
    if re.search(r"\b(pml_apply|fold_apply)\(", body):
        raise AssertionError("a certified curl-helper call survived the capture rewrite")
    return body


def zero_metal_carry() -> str:
    """``zero_metal_D`` carried between the halves, after the axis rules (driver order)."""
    lines = [
        "\n    // stepping._zero_metal / in_seam_passes.zero_metal_D, CARRIED, after the\n"
        "    // axis rules above. The OFF-DIAGONAL: a D component sits ON the wall of each\n"
        "    // axis whose Yee shift is 0, which for D is the other two.\n"
    ]
    for flag, coordinate, registers in _ZERO_METAL_ROWS:
        stores = " ".join(
            f"d{register} = cf_zero(); cf_store({_TARGETS[int(register)]}, idx, d{register});"
            for register in registers)
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {stores} }}\n")
    return "".join(lines)


def certified_constitutive_body() -> str:
    """``update_E_pml_complex_bloch``'s body, lifted, reading the registers, hoisted."""
    body = split_body(complex_emitter.complex_source("update_E"),
                      constitutive_prelude(), "complex_source('update_E')")
    body = bounds_block.strip_bounds_block(body, "the certified constitutive body",
                                           *CONSTITUTIVE_BLOCK)
    tail = body.split(_DECODE_END, 1)[1]
    for register in _CARRIED:
        tail = replace_once(tail, f"    cf s{register} = cf_load(g{register}, idx);\n",
                            f"    cf s{register} = d{register};   // THE SEAM\n",
                            f"s{register}'s load")
    if "cf_load(g" in tail:
        raise AssertionError("a source reload survived the seam rewrite")
    # The inverse-permittivity products are the certified lines, READ and not edited.
    for register in _CARRIED:
        line_starting(tail, f"    s{register} = mul_field_left(s{register}, "
                            f"inv_eps_{register}[idx]);",
                      f"target {register}'s inverse-permittivity product")
    for register, axis, flag in zip(_CARRIED, ("x", "y", "z"), constitutive_flags()):
        coordinate = "ijk"[int(register)]
        old = (f"    constitutive_apply(f{register}, w{register}, idx, s{register}, "
               f"kps_{axis}[{coordinate}], kms_{axis}[{coordinate}], {flag});")
        tail = replace_once(
            tail, old,
            f"    constitutive_apply_pre(f{register}, w{register}, idx, s{register}, "
            f"pre_w_{register}, pre_e_{register}, kps_{axis}[{coordinate}], "
            f"kms_half_{axis}[{coordinate}], {flag});",
            f"target {register}'s constitutive_apply")
    if re.search(r"\bconstitutive_apply\(", tail):
        raise AssertionError("a certified constitutive call survived the hoist rewrite")
    return tail


def cylindrical_fused_electric_pair_source() -> str:
    """The whole fused kernel: prelude, signature, body. PURE ASCII."""
    source = "".join((
        fused_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        zero_metal_carry(),
        "\n    // --- update_E, its sources the registers above ---\n",
        certified_constitutive_body(),
        "}\n",
    ))
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """The source this family emits, keyed by kernel name -- for a digest."""
    return {KERNEL_NAME: cylindrical_fused_electric_pair_source()}


#: NVRTC compile options -- CORRECTNESS, identical to both halves'.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str = KERNEL_NAME):
    """Compile the fused kernel, memoized on (name, options, policy,
    source). The source is emitted per call: a gate mutates this family through it."""
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be compiled; "
            "the predicate and the emitter need no device and still run")
    code = cylindrical_fused_electric_pair_source()
    key = compile_cache.kernel_cache_key(name, True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

#: The three E components whose inverse permittivity ``update_E`` multiplies by.
_INVERSE_EPSILON_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")


def covers_cylindrical_fused_electric_pair(fields: Any, pml: Any, grid: Any,
                                           sources: Any = None
                                           ) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``zero_metal_D`` -> ``update_E``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own certified
    predicate refuses is refused here with that half's reason, prefixed so a reader can
    tell which side said it. What this predicate ADDS is the seam clauses: the source
    slot, the injection ROUTE, the two UNCARRIED fills, and the grid's ability to say
    which axes are walled.
    """
    covered, reason = covers_pml_cylindrical_complex_curl(
        fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "E")
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE TWO FILLS ARE REFUSED, NOT CARRIED. The curl half already refuses every folded
    # grid and Grid refuses a mirror on a Dcyl cell before that; the clause is stated
    # HERE anyway, because REPLACES is a claim about what one launch performed and it
    # may not rest on a clause in another module that a future device verdict could
    # licence away.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_D and "
                       "fill_folded_far_ghosts_D run inside it")
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry or any(folded):
        return False, ("a mirror plane is active: fill_symmetry_bc_D "
                       "(driver.py:3309) and fill_folded_far_ghosts_D (:3311) run "
                       "inside this seam and this pair carries neither")

    # THE INJECTION ROUTE, REFUSED BY NAME, and it is a D-seam clause with no B-seam
    # counterpart. `_inject_electric_through_conductivity` (driver.py:3305) snapshots the
    # D components, injects, and rescales only the increment by `condinv` (MEEP
    # step.cpp:294-317). BOTH COMPLEX HALVES ADMIT A CONDUCTIVITY BY NAME -- it is read
    # in `stepping._apply_curl` and nowhere else -- so unlike the real electric pair
    # this refusal is not inherited from a half and has to be made here. The deposit
    # repair has never been measured against the scaled route.
    if getattr(fields, "has_conductivity", False):
        return False, ("the engine carries a conductivity, so the driver deposits "
                       "this seam's electric sources through "
                       "_inject_electric_through_conductivity (driver.py:3305), which "
                       "rescales the increment by condinv; the deposit repair has no "
                       "verdict on that route")

    # THE SOURCE SEAM, CARRIED RATHER THAN REFUSED. An ELECTRIC source is injected
    # BETWEEN the two halves (driver.py:3305/:3308). IGNORANCE IS NEVER AN EMPTY SET:
    # `Fields` does not hold the source list, so a predicate that inferred "no sources"
    # from not being told would be the over-covering this clause prevents.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3308)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

    # zero_metal_D IS CARRIED, so the grid must be able to say which axes are walled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_D cannot be "
                           f"carried in registers")
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE STORED SHAPE THE KERNEL INDEXES MUST BE THE ONE THE WALL PLANE AND THE RADIAL
    # DIFFERENCE ARE DERIVED FROM. `nx, ny, nz` come from `fields.Dx.shape`; the wall is
    # stored cell 0 of a `grid.stored_cells` axis and the difference is target-shaped.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Dx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks Dx.shape "
                       f"{extents}; the wall plane and the radial difference are derived "
                       f"from the first and indexed into the second")

    # THE THREE INVERSE-PERMITTIVITY VOLUMES MUST BE ASKABLE AND MUST MATCH THE SHAPE
    # THIS LAUNCH WALKS. The constitutive half's own predicate already checks them
    # against `_grid_facts`' shape; this asks the same question against the extents THIS
    # LAUNCH walks (`fields.Dx.shape`), which is the array the flat idx indexes.
    accessor = getattr(fields, "inverse_epsilon_for", None)
    if not callable(accessor):
        return False, ("fields does not expose inverse_epsilon_for; update_E's three "
                       "inverse-permittivity volumes cannot be bound")
    for component in _INVERSE_EPSILON_COMPONENTS:
        try:
            volume = accessor(component)
        except Exception as exc:  # noqa: BLE001
            return False, (f"fields.inverse_epsilon_for({component!r}) raised: "
                           f"{type(exc).__name__}: {exc}")
        if volume is None:
            return False, f"fields.inverse_epsilon_for({component!r}) is None"
        shape = tuple(int(n) for n in getattr(volume, "shape", ()))
        if shape != extents:
            return False, (f"inverse_epsilon_for({component!r}) has shape {shape} but "
                           f"the launch walks {extents}; the kernel indexes it with "
                           f"this launch's flat index")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The volumes bound from ``Fields``, in SIGNATURE ORDER.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz",
    "fu_Dx", "fu_Dy", "fu_Dz",
    "Hx", "Hy", "Hz",
)
_CONSTITUTIVE_BINDINGS: Tuple[str, ...] = (
    "Ex", "Ey", "Ez",
    "f_w_Ex", "f_w_Ey", "f_w_Ez",
)

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def cylindrical_fused_electric_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads: the D curl
    INTEGER, ``update_E`` HALF-INTEGER (the magnetic pair's pairing mirrored)."""
    if complex_pml_kernels is None:
        raise RuntimeError("the CuPy-backed table builders are not importable on this host")
    return {"curl": cylindrical_complex_kernels.cylindrical_complex_curl_tables(pml, False),
            "constitutive": complex_pml_kernels.complex_constitutive_tables(pml, True)}


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three per-component inverse-permittivity volumes, in signature order
    (``Fields.inverse_epsilon_for``, never ``fields.inv_eps``, which is the Ez view)."""
    return tuple(  # type: ignore[return-value]
        fields.inverse_epsilon_for(component) for component in _INVERSE_EPSILON_COMPONENTS)


def assert_disjoint_bindings(fields: "Fields", tables: Dict[str, Dict[str, Any]],
                             radial: Any, imr_rows: Sequence[Any]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes. The three
    ``inv_eps`` volumes may alias each other (they are not restrict) but no restrict
    argument. Returns the allocations checked."""
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in _FIELD_BINDINGS + _CONSTITUTIVE_BINDINGS:
        visit(name, getattr(fields, name))
    visit("rad", radial)
    for index, row in enumerate(imr_rows):
        visit(f"imr{index}", row)
    for key in _CURL_TABLE_KEYS:
        visit(f"curl:{key}", tables["curl"][key])
    for key in _CONSTITUTIVE_TABLE_KEYS:
        visit(f"constitutive:{key}", tables["constitutive"][key])
    for component, volume in zip(_INVERSE_EPSILON_COMPONENTS,
                                 inverse_epsilon_bindings(fields)):
        pointer = int(volume.data.ptr)
        if pointer in bound:
            collisions.append(f"inv_eps_{component} and {bound[pointer]} are the same "
                              f"allocation, and the second is bound __restrict__")
    if collisions:
        raise ValueError(
            "the cylindrical fused electric pair binds every field and table argument "
            "__restrict__, and these arguments alias, which is undefined behaviour NVRTC "
            "miscompiles silently rather than diagnosing: " + "; ".join(collisions))
    return len(bound)


def resolve_launch_arguments(fields: "Fields", grid: "Grid", pml: "PML",
                             chunk_bounds: Any) -> Dict[str, Any]:
    """Everything one launch binds that is a reading of the frozen grid and layer:
    held by a plan, built once. ``chunk_bounds`` is the composition's record."""
    from .cylindrical_prefix import radial_scan_device_tables, radial_scan_tables  # noqa: PLC0415

    xp = grid.xp
    m = int(grid.m)
    return {"tables": cylindrical_fused_electric_pair_tables(pml),
            "boundary_codes": cylindrical_complex_kernels.cylindrical_complex_boundary_codes(grid),
            "imr_rows": cylindrical_complex_kernels.device_rows(
                cylindrical_complex_kernels.imr_rows_for("step_D", grid,
                                                         int(fields.Dx.shape[0])), xp),
            "courant": cylindrical_complex_kernels.axis_courant_words("step_D", grid),
            "m": m, "m_class": cylindrical_complex_kernels.m_class(m),
            "zero_rows": cylindrical_complex_kernels.zero_rows(
                m, bool(grid.accurate_fields_near_cylorigin)),
            "walls": zero_metal_axes(grid), "dtdx": float(grid.dt / grid.dx),
            "chunk_bounds": chunk_bounds, "layer": pml,
            "scan": radial_scan_device_tables(radial_scan_tables(fields, pml, "step_D"), xp)}


def launch_cylindrical_fused_electric_pair(
        fields: "Fields", arguments: Dict[str, Any], radial: Any = None,
        kernel: Optional[Any] = None, scan_kernel: Optional[Any] = None) -> Dict[str, Any]:
    """The scan, then all three of :data:`REPLACES` in ONE launch.

    ``arguments`` are :func:`resolve_launch_arguments`'. ``radial`` may be handed in by a
    gate; otherwise the scan runs here over the CURRENT Hp. ``kernel`` and
    ``scan_kernel`` are the gate's doors.
    """
    word_view = cylindrical_complex_kernels.word_view
    cb = cylindrical_complex_kernels.chunk_record_argument(arguments["chunk_bounds"])
    if radial is None:
        radial = radial_difference(fields, arguments["layer"], "step_D",
                                   tables=arguments["scan"],
                                   scratch=getattr(fields, "scratch", None),
                                   kernel=scan_kernel)
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    if tuple(int(n) for n in radial.shape) != (nx, ny, nz):
        raise ValueError(f"the radial difference is {tuple(radial.shape)}, the target "
                         f"{(nx, ny, nz)}")
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = arguments["tables"]["curl"]
    constitutive = arguments["tables"]["constitutive"]
    walls = arguments["walls"]
    plus, four = arguments["courant"]
    launch_arguments = tuple(
        word_view(getattr(fields, name)) for name in _FIELD_BINDINGS
    ) + (word_view(radial),) + tuple(arguments["imr_rows"]) + tuple(
        word_view(getattr(fields, name)) for name in _CONSTITUTIVE_BINDINGS
    ) + tuple(inverse_epsilon_bindings(fields)) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(arguments["dtdx"]),
        np.float64(plus), np.float64(four),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + tuple(np.int32(code) for code in arguments["boundary_codes"]) + (
        np.int32(int(arguments["m_class"])), np.int32(int(arguments["zero_rows"])),
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3)) + (cb,)
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), launch_arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "m_class": int(arguments["m_class"]), "zero_rows": int(arguments["zero_rows"])}


def run_cylindrical_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: "PML", *, chunk_bounds: Any,
        sources: Any = None,
        arguments: Optional[Dict[str, Any]] = None, radial: Any = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, scan, launch once.
    A refusal is RETURNED, not raised."""
    covered, reason = covers_cylindrical_fused_electric_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if arguments is None:
        arguments = resolve_launch_arguments(fields, grid, pml, chunk_bounds)
    if radial is None:
        radial = radial_difference(fields, arguments["layer"], "step_D",
                                   tables=arguments["scan"],
                                   scratch=getattr(fields, "scratch", None))
    assert_disjoint_bindings(fields, arguments["tables"], radial, arguments["imr_rows"])
    return launch_cylindrical_fused_electric_pair(fields, arguments, radial=radial,
                                                  kernel=kernel)
