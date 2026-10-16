"""The Dcyl COMPLEX hand-CUDA fused pair: ``step_B`` -> ``zero_metal_B`` -> ``update_H``.

THE CELL: ``B_to_H (cuda_cyl_complex/cylindrical complex, cuda_complex/complex)``. Both
halves are lifted from the emitted texts of their singles -- the cylindrical complex curl
(``cylindrical_complex_kernels.cylindrical_complex_source("step_B")``) and the complex
constitutive (``complex_emitter.complex_source("update_H")``) -- and this module
introduces no arithmetic of its own: every edit is a register capture, a seam, a rename or
the own-cell hoist, listed in :data:`LIFT_EDITS`.

WHAT ONE LAUNCH PERFORMS. :data:`REPLACES` is ``step_B`` -> ``zero_metal_B`` ->
``update_H``. The radial scan is the curl's own first launch (``cylindrical_prefix``): it
runs before this one over the current Ep and hands the difference in as ``rad``. The two
mirror fills are absent by refusal (the predicate refuses every folded grid, and ``Grid``
refuses a mirror on a Dcyl cell).

THE REGISTER FOLLOWS EVERY STORE OF B. The curl half's value of each component is not the
value of its PML form alone:

* targets 0 and 2 take the i*m/r pass after the form (``stepping._cylindrical_imr_increment``
  through ``_apply_folded_increment``), so the register is captured again at the pass's
  store (``fold_apply_reg``);
* the |m| >= 2 axis rule zeroes the three components (and their ``fu``) on rows
  ``[0, zero_rows)``, and the m = 0 rule zeroes Br on the axis row, after the pass;
* ``zero_metal_B`` (the driver's pass after ``step_B``) is carried after those.

Each is a register clear or capture BESIDE the certified store, so ``update_H`` reads the
word ``step_B`` and the wall pass left.

THE OWN-CELL HOIST. Every own-cell word the kernel reads -- ``f_w_H*``, ``H*``, ``fu_B*``,
``B*`` -- is issued before its first store, each behind the certified load's own guard
(``fu`` behind the component's dsigu flag, ``f_w_H`` and ``H`` behind its dsigw flag, the
one form that reads them; ``B`` unguarded, as every curl form reads it). The two helpers
that would have loaded them take the words as arguments and are DERIVED from the certified
text, their statements asserted equal to the certified helper's under the argument-to-load
map (:func:`derive`). The i*m/r pass's ladder reads ``B`` and ``fu`` back after the form
stored them: reloads of this lane's own writes.

THE START OF H. The run's first ``update_H`` starts H and ``f_w_H`` from B
(``stepping.separate_h_from_b``); B is final only after this launch, so on that launch
:func:`start_of_h` re-steps ``update_H`` on the array path over the stored B, as the
Cartesian pairs do.

ALIASING. B is bound EXACTLY ONCE; the constitutive half reads the registers and has no
source pointers. The two halves collide on ``kms_x/y/z`` (half-integer for the B curl,
integer for ``update_H``), so the constitutive group is ``kms_int_*``.
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
from .cylindrical_prefix import radial_difference
from .in_seam_coverage import zero_metal_axes

try:
    from . import complex_pml_kernels
except ImportError:  # a host with no CuPy: only the table builder is unavailable
    complex_pml_kernels = None

from .. import deposit_repair as _deposit_repair


# =============================================================================
# THE PARTITION -- PLAIN ASSIGNMENTS, NO ANNOTATION (the partition readers walk the
# syntax tree without importing the module)
# =============================================================================

#: The entry point. The record that certified its 0.9.2 text is re-cut in the combined
#: certification round (bit_parity_0_10 CERT-3).
CERTIFIED_KERNELS = (
    "fused_magnetic_pair_pml_cyl_complex",
)
UNCERTIFIED_KERNELS = {}

#: The bracket that repairs an in-seam magnetic deposit (``fused_pairs._install_fused_pair``
#: wires it; the flag and the wiring change together). The constitutive half is
#: ``complex_emitter``'s, which the repair inverts.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_cylindrical_fused_magnetic_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_magnetic_pair_pml_cyl_complex"

#: The driver passes ONE launch of this kernel performs, in driver order.
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: The sub-step slot the planner holds this on.
SLOT = "step_B"

#: Every line of certified device text this file did not lift verbatim, with the reason.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ cf pml_apply_reg_pre(",
     "why": "the constitutive half consumes the value in a register: the helper returns "
            "the word pair it stored, and takes its two own-cell words (fprev, fcur) as "
            "arguments the hoist preloaded under the same guards. DERIVED by anchored "
            "replacement; its statements equal the certified helper's under the "
            "argument-to-load map."},
    {"line": "__device__ __forceinline__ void fold_apply(",
     "became": "__device__ __forceinline__ cf fold_apply_reg(",
     "why": "the i*m/r pass stores the target again after its PML form; the helper "
            "returns the word pair it stored so the register follows it."},
    {"line": "__device__ __forceinline__ void constitutive_apply(",
     "became": "__device__ __forceinline__ void constitutive_apply_pre(",
     "why": "the own-cell hoist: prev and the old H are arguments the hoist preloaded "
            "under the dsigw flag. DERIVED and checked as pml_apply_reg_pre is."},
    {"line": "    cf updated = dsigw ? cf_add(cf_load(f, idx), cf_sub(mul_coefficient_left("
             "kps, src), mul_coefficient_left(kms, prev)))",
     "became": "    cf updated = dsigw ? cf_add(fcur, cf_sub(mul_coefficient_left("
               "kps, src), mul_coefficient_left(kms, prev)))",
     "why": "the accumulation's one load of the old H is the argument the hoist "
            "preloaded under the dsigw flag (CONSTITUTIVE_LOADS: fcur is "
            "cf_load(f, idx)); the expression is otherwise the certified one."},
    {"line": "    <the curl body's chunk-record block> + <update_H's>",
     "became": "    <ONE block, bounds_block.bounds_block(both entry sets, presence xyz)>",
     "why": "one decode of the record for both halves (DESIGN.md D10); "
            "bounds_block.strip_bounds_block refuses unless each is exactly the block "
            "its body carries."},
    {"line": "        pml_apply(Bx, fu_Bx, idx, curl, ...);",
     "became": "        b0 = pml_apply_reg_pre(Bx, fu_Bx, idx, curl, ..., pre_fu_0, pre_b_0);",
     "why": "capture the register and take the preloaded words; three lines, the "
            "argument lists otherwise untouched."},
    {"line": "            fold_apply(Bx, fu_Bx, idx, inc, ...);",
     "became": "            b0 = fold_apply_reg(Bx, fu_Bx, idx, inc, ...);",
     "why": "the register follows the i*m/r pass's store (targets 0 and 2)."},
    {"line": "        cf_store(Bx, idx, cf_zero());   (the |m| >= 2 and m = 0 axis rules)",
     "became": "        b0 = cf_zero(); cf_store(Bx, idx, b0);",
     "why": "the register follows the axis rules' stores; the guards, the rows and the "
            "fu stores are untouched."},
    {"line": "        array[_face(axis, 0)] = 0   (stepping._zero_metal)",
     "became": "    if (wall_x && i == 0) { b0 = cf_zero(); cf_store(Bx, idx, b0); }",
     "why": "zero_metal_B carried on the registers, after the axis rules (the driver's "
            "order). Same diagonal, same stored cell 0, same (+0, +0)."},
    {"line": "    cf s0 = cf_load(g0, idx);",
     "became": "    cf s0 = b0;",
     "why": "THE SEAM: the register instead of a reload of B, which is also what lets B "
            "be bound exactly once."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i], in_i_x);",
     "became": "    constitutive_apply_pre(f0, w0, idx, s0, pre_w_0, pre_h_0, kps_x[i], "
               "kms_int_x[i], in_i_x);",
     "why": "the sub-lattice rename (kms_int_*) and the preloaded words."},
    {"line": "    cf b2 = cf_zero();",
     "became": "    cf b2 = cf_zero();\n    cf pre_w_0 = in_i_x ? cf_load(w0, idx) : cf_zero();"
               "   // ... twelve guarded lines",
     "why": "THE OWN-CELL HOIST: every word this thread reads at its own idx is issued "
            "before the kernel's first store, behind the certified load's own guard. "
            "No other thread writes a word at this idx in this launch (no fill: every "
            "folded grid is refused)."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body", "certified_curl_body",
    "covers_cylindrical_fused_magnetic_pair", "cylindrical_fused_magnetic_pair_prelude",
    "cylindrical_fused_magnetic_pair_source", "cylindrical_fused_magnetic_pair_tables",
    "derive", "device_sources", "hoisted_loads", "launch_cylindrical_fused_magnetic_pair",
    "resolve_launch_arguments", "run_cylindrical_fused_magnetic_pair", "start_of_h",
    "union_bounds_block", "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================

_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"

#: The certified helpers' signatures and the anchors the derivations edit.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_FOLD_APPLY_SIGNATURE = "__device__ __forceinline__ void fold_apply(\n"
_CONSTITUTIVE_SIGNATURE = "__device__ __forceinline__ void constitutive_apply(\n"
_STORE_VALUE = "    cf_store(f, idx, value);\n"
_STORE_VALUE_RETURNED = (
    "    cf_store(f, idx, value);\n"
    "    // The value is additionally RETURNED, so the next half reads it from a\n"
    "    // register instead of reloading the word pair; the arithmetic and the store\n"
    "    // are the certified ones.\n"
    "    return value;\n")

#: The three carried registers and the B targets they follow, in target order.
_CARRIED: Tuple[str, ...] = ("0", "1", "2")
_TARGETS: Tuple[str, ...] = ("Bx", "By", "Bz")

#: ``stepping._zero_metal``'s diagonal for the B side, as (register, target, flag,
#: coordinate): each B component is wiped on its OWN axis (``fields.IYEE_SHIFTS``).
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("b0", "Bx", "wall_x", "i"),
    ("b1", "By", "wall_y", "j"),
    ("b2", "Bz", "wall_z", "k"),
)

#: ``stepping._cylindrical_axis_zero_B``'s m = 0 branch, as the curl emits it, and its
#: carried rewrite. Anchored as a WHOLE BLOCK: its one store also appears in the
#: |m| >= 2 branch.
_AXIS_TAIL_M0 = (
    "    if (m_class == 0 && i == 0) {\n"
    "        cf_store(Bx, idx, cf_zero());\n"
    "    }\n")
_AXIS_TAIL_M0_CARRIED = (
    "    // THE m = 0 AXIS TAIL, CARRIED. stepping._cylindrical_axis_zero_B zeroes Br on\n"
    "    // the axis row -- THE FIELD ONLY, never fu_Bx -- after the recurrence, so the\n"
    "    // register has to follow the store or update_H would read the pre-zeroing Bx.\n"
    "    if (m_class == 0 && i == 0) {\n"
    "        b0 = cf_zero(); cf_store(Bx, idx, b0);\n"
    "    }\n")

#: The fused entry point. The ONLY hand-written device text in this module, and it is a
#: signature. B appears EXACTLY ONCE.
_SIGNATURE = r'''
extern "C" __global__ void fused_magnetic_pair_pml_cyl_complex(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 B.
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    // The radial difference the scan wrote, and the two i*m/r rinv rows.
    const float* __restrict__ rad,
    const float* __restrict__ imr0, const float* __restrict__ imr2,
    // H and its constitutive history, the constitutive template's own names.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx, float m_word, double minus_courant,
    // The B curl's HALF-INTEGER split-field coefficients.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The H constitutive's INTEGER coefficients, its kms renamed kms_int_*.
    const float* __restrict__ kps_x, const float* __restrict__ kms_int_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_int_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_int_z,
    int bc_x, int bc_y, int bc_z,
    int m_class, int zero_rows,
    // zero_metal_B's three walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z,
    // THE CHUNK RECORD (chunk_map.chunk_bounds_record), int32[25].
    const int* __restrict__ cb
) {
'''


def split_body(source: str, prelude: str, name: str) -> str:
    """One emitted kernel string reduced to its body, or a named failure."""
    if not source.startswith(prelude):
        raise AssertionError(
            f"{name} no longer begins with the prelude this module lifts separately; "
            f"the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, not 1; "
            f"the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure."""
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line starting "
            f"{prefix!r}); this family LIFTS that line and cannot splice around its "
            f"absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the lift "
            f"of {what} would take an arbitrary one")
    return matches[0]


def replace_once(text: str, old: str, new: str, what: str) -> str:
    """``text`` with ``old`` replaced exactly once, or a named failure."""
    if text.count(old) != 1:
        raise AssertionError(
            f"the certified text carries {what} ({old!r}) {text.count(old)} times, not "
            f"once; the lift has no anchor")
    return text.replace(old, new, 1)


def curl_prelude() -> str:
    """The prelude ``cylindrical_complex_source`` builds, before its own template."""
    return complex_emitter.prelude() + cylindrical_complex_kernels._CYLINDRICAL_HELPERS


def constitutive_prelude() -> str:
    """The prelude ``complex_emitter.complex_source`` builds."""
    return complex_emitter.prelude()


def function_text(prelude: str, signature: str, what: str) -> str:
    """One helper's whole text in ``prelude``: its signature to its closing brace."""
    if prelude.count(signature) != 1:
        raise AssertionError(
            f"the prelude declares {what} {prelude.count(signature)} times, not once; "
            f"the derived helper has nothing to be derived from")
    text = prelude[prelude.index(signature):]
    end = text.find("\n}\n")
    if end < 0:
        raise AssertionError(f"{what} has no closing brace in the prelude")
    return text[: end + len("\n}\n")]


def _statements(text: str, loads: Dict[str, str], what: str) -> List[str]:
    """The helper's statements, every name of ``loads`` replaced by its load.

    A ``cf NAME = cf_load(P, idx);`` line or its guarded form ``cf NAME = FLAG ?
    cf_load(P, idx) : cf_zero();`` whose NAME the map names is a BINDING: it must bind
    exactly the map's load, and is dropped from the stream. Every other line is a
    statement and stays.
    """
    if text.count(") {\n") != 1:
        raise AssertionError(f"{what}: the parameter list no longer closes on one line")
    body = text.split(") {\n", 1)[1][: -len("}\n")]
    out: List[str] = []
    for line in body.splitlines():
        code = line.split("//", 1)[0].strip()
        if not code:
            continue
        bound = (re.fullmatch(r"cf (\w+) = (cf_load\(\w+, idx\));", code)
                 or re.fullmatch(r"cf (\w+) = (\w+ \? cf_load\(\w+, idx\) : cf_zero\(\));",
                                 code))
        if bound and bound.group(1) in loads:
            spelled = bound.group(2)
            if spelled.startswith("cf_load"):
                want = loads[bound.group(1)]
            else:
                want = loads[bound.group(1)][1:-1]
            if spelled != want:
                raise AssertionError(
                    f"{what}: {bound.group(1)} is bound to {spelled}, not to "
                    f"{loads[bound.group(1)]}; the load map names the wrong word")
            continue
        for name, load in loads.items():
            code = re.sub(rf"\b{name}\b", load, code)
        out.append(code)
    return out


def derive(helper: str, edits, loads: Dict[str, str], what: str) -> str:
    """``helper`` with ``edits`` applied, refused unless the derived statements equal the
    certified ones under the argument-to-load map ``loads``."""
    derived = helper
    for old, new, anchor in edits:
        derived = replace_once(derived, old, new, f"{what}'s {anchor}")
    certified = _statements(helper, loads, what)
    hoisted = _statements(derived, loads, what + " (derived)")
    if certified != hoisted:
        raise AssertionError(
            f"the derived {what} does not carry the certified arithmetic: "
            f"{certified!r} != {hoisted!r}")
    return derived


#: The own-cell load maps: each name a derivation turns into a parameter, bound to the
#: word its caller must preload (the guarded expression where the certified helper loads
#: under its membership flag).
PML_APPLY_LOADS: Dict[str, str] = {"fprev": "(dsigu ? cf_load(fu, idx) : cf_zero())",
                                   "fcur": "cf_load(f, idx)"}
CONSTITUTIVE_LOADS: Dict[str, str] = {"prev": "(dsigw ? cf_load(fw, idx) : cf_zero())",
                                      "fcur": "cf_load(f, idx)"}


def pml_apply_reg_source(prelude: str) -> str:
    """The certified complex ``pml_apply`` returning the value it stored: the signature and
    the closing store, the arithmetic untouched."""
    helper = function_text(prelude, _PML_APPLY_SIGNATURE, "pml_apply")
    helper = replace_once(helper, _PML_APPLY_SIGNATURE,
                          "__device__ __forceinline__ cf pml_apply_reg(\n",
                          "pml_apply's signature")
    return replace_once(helper, _STORE_VALUE, _STORE_VALUE_RETURNED,
                        "pml_apply's closing store")


def pml_apply_reg_pre_source(prelude: str) -> str:
    """:func:`pml_apply_reg_source` taking ``fprev`` / ``fcur`` as arguments -- derived from
    the text and checked against it under :data:`PML_APPLY_LOADS`."""
    helper = pml_apply_reg_source(prelude)
    return derive(helper, (
        ("__device__ __forceinline__ cf pml_apply_reg(\n",
         "__device__ __forceinline__ cf pml_apply_reg_pre(\n", "signature"),
        ("    float kms, float sinv, float kms_u, float sinv_u, int dsigu\n)",
         "    float kms, float sinv, float kms_u, float sinv_u, int dsigu,\n"
         "    cf fprev, cf fcur\n)", "parameter list"),
        ("    cf fprev = dsigu ? cf_load(fu, idx) : cf_zero();\n", "",
         "guarded load of fu"),
        ("    cf fcur = cf_load(f, idx);\n", "", "load of f"),
    ), PML_APPLY_LOADS, "pml_apply_reg")


def fold_apply_reg_source(prelude: str) -> str:
    """The family's ``fold_apply`` returning the value it stored: the signature and the
    closing store, the arithmetic untouched."""
    helper = function_text(prelude, _FOLD_APPLY_SIGNATURE, "fold_apply")
    helper = replace_once(helper, _FOLD_APPLY_SIGNATURE,
                          "__device__ __forceinline__ cf fold_apply_reg(\n",
                          "fold_apply's signature")
    return replace_once(helper, _STORE_VALUE, _STORE_VALUE_RETURNED,
                        "fold_apply's closing store")


def constitutive_apply_pre_source(prelude: str) -> str:
    """``complex_emitter``'s ``constitutive_apply`` taking ``prev`` / ``fcur`` as
    arguments -- derived from the text."""
    helper = function_text(prelude, _CONSTITUTIVE_SIGNATURE, "constitutive_apply")
    return derive(helper, (
        (_CONSTITUTIVE_SIGNATURE, "__device__ __forceinline__ void constitutive_apply_pre(\n",
         "signature"),
        ("    float* f, float* fw, int idx, cf src, float kps, float kms, int dsigw\n)",
         "    float* f, float* fw, int idx, cf src,\n"
         "    cf prev, cf fcur, float kps, float kms, int dsigw\n)", "parameter list"),
        ("    cf prev = dsigw ? cf_load(fw, idx) : cf_zero();\n", "", "guarded load of fw"),
        ("cf_add(cf_load(f, idx), ", "cf_add(fcur, ", "load of f"),
    ), CONSTITUTIVE_LOADS, "constitutive_apply")


def fused_prelude() -> str:
    """The curl prelude with its three helpers replaced by their derived twins.

    ``pml_apply``, ``fold_apply`` and ``constitutive_apply`` are each replaced IN PLACE by
    the derived text: every call takes the derived twin, so a kept copy would be dead text
    -- and a dead copy of a helper is where a first-site mutation needle lands vacuously.
    """
    prelude = curl_prelude()
    for signature, what, derived in (
            (_PML_APPLY_SIGNATURE, "pml_apply", pml_apply_reg_pre_source(prelude)),
            (_FOLD_APPLY_SIGNATURE, "fold_apply", fold_apply_reg_source(prelude)),
            (_CONSTITUTIVE_SIGNATURE, "constitutive_apply",
             constitutive_apply_pre_source(prelude))):
        prelude = replace_once(prelude, function_text(prelude, signature, what), derived,
                               f"{what}'s text")
    return prelude


def cylindrical_fused_magnetic_pair_prelude() -> str:
    """The fused kernel's prelude."""
    return fused_prelude()


#: The chunk-record entries each lifted body decodes, as ``(entries, presence)``.
CURL_BLOCK: Tuple[Any, Tuple[int, ...]] = (
    tuple(bounds_block.curl_entries(("Bx", "By", "Bz"))), (0, 1, 2))
CONSTITUTIVE_BLOCK: Tuple[Any, Tuple[int, ...]] = (
    tuple(bounds_block.constitutive_entries(("Hx", "Hy", "Hz"))), ())


def union_bounds_block() -> str:
    """The fused kernel's ONE chunk-record block: both bodies' entries, one decode."""
    return bounds_block.bounds_block(CURL_BLOCK[0] + CONSTITUTIVE_BLOCK[0],
                                     CURL_BLOCK[1] + CONSTITUTIVE_BLOCK[1])


def curl_flags() -> Tuple[str, ...]:
    """Each B target's dsigu membership flag, in target order (the curl's own)."""
    return complex_emitter.CURL_FLAGS[False]


def constitutive_flags() -> Tuple[str, ...]:
    """Each H component's dsigw membership flag, x/y/z."""
    return complex_emitter.CONSTITUTIVE_FLAGS["H"]


def hoisted_loads() -> str:
    """The twelve own-cell loads (24 words), each behind the certified load's guard."""
    lines = [
        "    // --- the own-cell hoist: every word this thread reads at idx, issued",
        "    // before the kernel's first store, behind the certified load's guard",
        "    // (f_w_H and H behind the dsigw flag; fu behind the dsigu flag). ---",
    ]
    for register, flag in zip(_CARRIED, constitutive_flags()):
        lines.append(f"    cf pre_w_{register} = {flag} ? cf_load(w{register}, idx) : cf_zero();")
        lines.append(f"    cf pre_h_{register} = {flag} ? cf_load(f{register}, idx) : cf_zero();")
    for register, target, flag in zip(_CARRIED, _TARGETS, curl_flags()):
        lines.append(f"    cf pre_fu_{register} = {flag} ? cf_load(fu_{target}, idx) : cf_zero();")
        lines.append(f"    cf pre_b_{register} = cf_load({target}, idx);")
    return "\n".join(lines) + "\n"


def _certified_source() -> Any:
    """Refuse the emitter BY NAME where the cylindrical text is unreachable.

    THE SPLICE IS THE LIFT, so a missing half is not a degraded emit -- there is nothing
    to emit. Raising here keeps that fact one frame from the caller instead of surfacing
    as ``'NoneType' object has no attribute`` inside a string operation.
    :func:`covers_cylindrical_fused_magnetic_pair` needs none of it and still answers.
    """
    if cylindrical_complex_kernels is None:
        raise RuntimeError(
            "cylindrical_complex_kernels is not available, so there is no certified "
            "cylindrical text to splice. covers_cylindrical_fused_magnetic_pair needs "
            "none of it and still answers")
    return cylindrical_complex_kernels


def certified_curl_body() -> str:
    """``cyl_step_B_pml_complex``'s body, lifted: registers captured at every store of B,
    the union record block, the hoist."""
    body = split_body(_certified_source().cylindrical_complex_source("step_B"),
                      curl_prelude(), "cylindrical_complex_source('step_B')")
    body = bounds_block.strip_bounds_block(body, "the certified curl body", *CURL_BLOCK)
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END, "\n",
        union_bounds_block(),
        "    // The three carried registers, declared here: each component below is a\n"
        "    // braced scope, and a value declared inside one does not outlive it.\n"
        "    cf b0 = cf_zero();\n"
        "    cf b1 = cf_zero();\n"
        "    cf b2 = cf_zero();\n",
        hoisted_loads(),
        tail,
    ))
    for register, target in zip(_CARRIED, _TARGETS):
        prefix = f"        pml_apply({target}, fu_{target}, idx, curl, "
        call = line_starting(body, prefix, f"{target}'s pml_apply")
        if not call.endswith(");"):
            raise AssertionError(f"{target}'s pml_apply no longer closes on one line")
        body = body.replace(call, call.replace(
            f"pml_apply({target},", f"b{register} = pml_apply_reg_pre({target},", 1)[:-2]
            + f", pre_fu_{register}, pre_b_{register});", 1)
    for register, target in (("0", "Bx"), ("2", "Bz")):
        body = replace_once(body, f"            fold_apply({target}, fu_{target}, idx, inc, ",
                            f"            b{register} = fold_apply_reg({target}, fu_{target}, "
                            f"idx, inc, ", f"{target}'s i*m/r ladder call")
    body = replace_once(body, _AXIS_TAIL_M0, _AXIS_TAIL_M0_CARRIED,
                        "the m = 0 branch of stepping._cylindrical_axis_zero_B")
    for register, target in zip(_CARRIED, _TARGETS):
        body = replace_once(body, f"        cf_store({target}, idx, cf_zero());\n",
                            f"        b{register} = cf_zero(); cf_store({target}, idx, "
                            f"b{register});   // THE AXIS TAIL, CARRIED\n",
                            f"the |m| >= 2 rule's store into {target}")
    if re.search(r"\b(pml_apply|fold_apply)\(", body):
        raise AssertionError("a certified curl-helper call survived the capture rewrite")
    return body


def zero_metal_carry() -> str:
    """``zero_metal_B`` carried between the halves, after the axis rules (driver order)."""
    lines = [
        "\n    // stepping._zero_metal / in_seam_passes.zero_metal_B, CARRIED, after the\n"
        "    // axis rules above (they are inside step_B; the wall pass follows it). The\n"
        "    // diagonal only: a B component sits ON the wall of its own axis.\n"
    ]
    for register, target, flag, coordinate in _ZERO_METAL_ROWS:
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {register} = cf_zero(); "
                     f"cf_store({target}, idx, {register}); }}\n")
    return "".join(lines)


def certified_constitutive_body() -> str:
    """``update_H_pml_complex_bloch``'s body, lifted, reading the registers, hoisted."""
    body = split_body(complex_emitter.complex_source("update_H"),
                      constitutive_prelude(), "complex_source('update_H')")
    body = bounds_block.strip_bounds_block(body, "the certified constitutive body",
                                           *CONSTITUTIVE_BLOCK)
    tail = body.split(_DECODE_END, 1)[1]
    for register in _CARRIED:
        tail = replace_once(tail, f"    cf s{register} = cf_load(g{register}, idx);\n",
                            f"    cf s{register} = b{register};   // THE SEAM\n",
                            f"s{register}'s load")
    if "cf_load(g" in tail:
        raise AssertionError("a source reload survived the seam rewrite")
    for register, axis, flag in zip(_CARRIED, ("x", "y", "z"), constitutive_flags()):
        coordinate = "ijk"[int(register)]
        old = (f"    constitutive_apply(f{register}, w{register}, idx, s{register}, "
               f"kps_{axis}[{coordinate}], kms_{axis}[{coordinate}], {flag});")
        tail = replace_once(
            tail, old,
            f"    constitutive_apply_pre(f{register}, w{register}, idx, s{register}, "
            f"pre_w_{register}, pre_h_{register}, kps_{axis}[{coordinate}], "
            f"kms_int_{axis}[{coordinate}], {flag});",
            f"target {register}'s constitutive_apply")
    if re.search(r"\bconstitutive_apply\(", tail):
        raise AssertionError("a certified constitutive call survived the hoist rewrite")
    return tail


def cylindrical_fused_magnetic_pair_source() -> str:
    """The whole fused kernel: prelude, signature, body. PURE ASCII."""
    source = "".join((
        fused_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        zero_metal_carry(),
        "\n    // --- update_H, its sources the registers above ---\n",
        certified_constitutive_body(),
        "}\n",
    ))
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """The source this family emits, keyed by kernel name -- for a digest."""
    return {KERNEL_NAME: cylindrical_fused_magnetic_pair_source()}


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
    code = cylindrical_fused_magnetic_pair_source()
    key = compile_cache.kernel_cache_key(name, True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_cylindrical_fused_magnetic_pair(fields: Any, pml: Any, grid: Any,
                                           sources: Any = None
                                           ) -> Tuple[bool, str]:
    """May ONE launch span ``step_B`` -> ``zero_metal_B`` -> ``update_H``?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own
    certified predicate refuses is refused here with that half's reason, prefixed so
    a reader can tell which side said it. What this predicate ADDS is the seam
    clauses: the source slot, the two UNCARRIED fills, and the grid's ability to say
    which axes are walled.
    """
    covered, reason = covers_pml_cylindrical_complex_curl(
        fields, pml, grid, "step_B")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "H")
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE SOURCE SEAM, CARRIED RATHER THAN REFUSED. A MAGNETIC source is injected
    # BETWEEN the two halves (driver.py:3293). IGNORANCE IS NEVER AN EMPTY SET:
    # `Fields` does not hold the source list, so a predicate that inferred "no
    # sources" from not being told would be the over-covering this clause prevents.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an "
            "empty magnetic source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the driver "
            f"injects it BETWEEN step_B and update_H (driver.py:3293)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

    # THE TWO FILLS ARE REFUSED, NOT CARRIED. The curl half already refuses every
    # folded grid and Grid refuses a mirror on a Dcyl cell before that; the clause is
    # stated HERE anyway, because REPLACES is a claim about what one launch performed
    # and it may not rest on a clause in another module that a future device verdict
    # could licence away.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_B and "
                       "fill_folded_far_ghosts_B run inside it")
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry or any(folded):
        return False, ("a mirror plane is active: fill_symmetry_bc_B "
                       "(driver.py:3294) and fill_folded_far_ghosts_B (:3296) run "
                       "inside this seam and this pair carries neither")

    # zero_metal_B IS CARRIED, so the grid must be able to say which axes are walled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_B cannot be "
                           f"carried in registers")
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE STORED SHAPE THE KERNEL INDEXES MUST BE THE ONE THE WALL PLANE AND THE
    # RADIAL DIFFERENCE ARE DERIVED FROM. `nx, ny, nz` come from `fields.Bx.shape`; the
    # wall is stored cell 0 of a `grid.stored_cells` axis and the difference is
    # target-shaped.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Bx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks "
                       f"Bx.shape {extents}; the wall plane and the radial difference "
                       f"are derived from the first and indexed into the second")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The volumes bound from ``Fields``, in SIGNATURE ORDER.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Bx", "By", "Bz",
    "fu_Bx", "fu_By", "fu_Bz",
    "Ex", "Ey", "Ez",
)
_CONSTITUTIVE_BINDINGS: Tuple[str, ...] = (
    "Hx", "Hy", "Hz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def cylindrical_fused_magnetic_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads: the B curl
    HALF-INTEGER, the H constitutive INTEGER."""
    if complex_pml_kernels is None:
        raise RuntimeError("the CuPy-backed table builders are not importable on this host")
    return {"curl": cylindrical_complex_kernels.cylindrical_complex_curl_tables(pml, True),
            "constitutive": complex_pml_kernels.complex_constitutive_tables(pml, False)}


def assert_disjoint_bindings(fields: "Fields", tables: Dict[str, Dict[str, Any]],
                             radial: Any, imr_rows: Sequence[Any]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes, the radial
    difference and the two i*m/r rows included. Returns the allocations checked."""
    bound: Dict[int, str] = {}
    collisions = []

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
    if collisions:
        raise ValueError(
            "the cylindrical fused magnetic pair binds every argument __restrict__, and "
            "these arguments alias, which is undefined behaviour NVRTC miscompiles "
            "silently rather than diagnosing: " + "; ".join(collisions))
    return len(bound)


def start_of_h(fields: "Fields", layer: Any) -> None:
    """MEEP's start of H after the run's first fused magnetic launch, on the array path:
    ``stepping.update_H`` over the B the launch stored (``separate_h_from_b``)."""
    from ..stepping import update_H  # noqa: PLC0415 - NumPy/CuPy array path

    update_H(fields, layer)


def resolve_launch_arguments(fields: "Fields", grid: "Grid", pml: "PML",
                             chunk_bounds: Any) -> Dict[str, Any]:
    """Everything one launch binds that is a reading of the frozen grid and layer:
    held by a plan, built once. ``chunk_bounds`` is the composition's record."""
    from .cylindrical_prefix import radial_scan_device_tables, radial_scan_tables  # noqa: PLC0415

    xp = grid.xp
    m = int(grid.m)
    return {"tables": cylindrical_fused_magnetic_pair_tables(pml),
            "boundary_codes": cylindrical_complex_kernels.cylindrical_complex_boundary_codes(grid),
            "imr_rows": cylindrical_complex_kernels.device_rows(
                cylindrical_complex_kernels.imr_rows_for("step_B", grid,
                                                         int(fields.Bx.shape[0])), xp),
            "courant": cylindrical_complex_kernels.axis_courant_words("step_B", grid),
            "m": m, "m_class": cylindrical_complex_kernels.m_class(m),
            "zero_rows": cylindrical_complex_kernels.zero_rows(
                m, bool(grid.accurate_fields_near_cylorigin)),
            "walls": zero_metal_axes(grid), "dtdx": float(grid.dt / grid.dx),
            "chunk_bounds": chunk_bounds, "layer": pml,
            "scan": radial_scan_device_tables(radial_scan_tables(fields, pml, "step_B"), xp)}


def launch_cylindrical_fused_magnetic_pair(
        fields: "Fields", arguments: Dict[str, Any], radial: Any = None,
        kernel: Optional[Any] = None, scan_kernel: Optional[Any] = None) -> Dict[str, Any]:
    """The scan, then all three of :data:`REPLACES` in ONE launch.

    ``arguments`` are :func:`resolve_launch_arguments`'. ``radial`` may be handed in by a
    gate that already computed it; otherwise the scan runs here over the CURRENT Ep (a
    cached difference would be a stale field one step later). ``kernel`` and
    ``scan_kernel`` are the gate's doors. The run's first launch starts H on the array
    path (:func:`start_of_h`) and refuses by name when handed no layer.
    """
    word_view = cylindrical_complex_kernels.word_view
    cb = cylindrical_complex_kernels.chunk_record_argument(arguments["chunk_bounds"])
    first = not bool(getattr(fields, "h_separated", True))
    if first and arguments.get("layer") is None:
        raise RuntimeError(
            "the run's first update_H starts H from B (stepping.separate_h_from_b) and "
            "this fused magnetic launch was handed no layer to start it with")
    if radial is None:
        radial = radial_difference(fields, arguments["layer"], "step_B",
                                   tables=arguments["scan"],
                                   scratch=getattr(fields, "scratch", None),
                                   kernel=scan_kernel)
    nx, ny, nz = (int(n) for n in fields.Bx.shape)
    if tuple(int(n) for n in radial.shape) != (nx, ny, nz):
        raise ValueError(f"the radial difference is {tuple(radial.shape)}, the target "
                         f"{(nx, ny, nz)}")
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = arguments["tables"]["curl"]
    constitutive = arguments["tables"]["constitutive"]
    walls = arguments["walls"]
    launch_arguments = tuple(
        word_view(getattr(fields, name)) for name in _FIELD_BINDINGS
    ) + (word_view(radial),) + tuple(arguments["imr_rows"]) + tuple(
        word_view(getattr(fields, name)) for name in _CONSTITUTIVE_BINDINGS
    ) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(arguments["dtdx"]),
        np.float32(float(arguments["m"])), np.float64(arguments["courant"][0]),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + tuple(np.int32(code) for code in arguments["boundary_codes"]) + (
        np.int32(int(arguments["m_class"])), np.int32(int(arguments["zero_rows"])),
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3)) + (cb,)
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), launch_arguments)
    if first:
        start_of_h(fields, arguments["layer"])
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES, "h_started": first,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "m_class": int(arguments["m_class"]), "zero_rows": int(arguments["zero_rows"])}


def run_cylindrical_fused_magnetic_pair(
        fields: "Fields", grid: "Grid", pml: "PML", *, chunk_bounds: Any,
        sources: Any = None,
        arguments: Optional[Dict[str, Any]] = None, radial: Any = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, scan, launch once.

    A refusal is RETURNED, not raised: the caller's answer to a configuration this
    product does not carry is the array path.
    """
    covered, reason = covers_cylindrical_fused_magnetic_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if arguments is None:
        arguments = resolve_launch_arguments(fields, grid, pml, chunk_bounds)
    if radial is None:
        radial = radial_difference(fields, arguments["layer"], "step_B",
                                   tables=arguments["scan"],
                                   scratch=getattr(fields, "scratch", None))
    assert_disjoint_bindings(fields, arguments["tables"], radial, arguments["imr_rows"])
    return launch_cylindrical_fused_magnetic_pair(fields, arguments, radial=radial,
                                                  kernel=kernel)
