"""The SCRATCH-OUTPUT lift both COMPLEX off-diagonal D/E welds are built from.

WHY THERE IS A SHARED MODULE AND NOT TWO COPIES, and it is
:mod:`.offdiag_stencil_weld`'s reason one storage width over:
:mod:`.folded_complex_offdiag_fused_electric_pair` and
:mod:`.complex_no_pml_offdiag_fused_electric_pair` occupy two different board cells
with two different curl arms and two different constitutive tails -- but the scratch
discipline, the per-cell resolution and the rotation are ONE piece of arithmetic, and
the one thing this track will not do is let two copies of a resolution drift apart.
Everything that decides a float lives here; each family module owns its predicate, its
signature, its cell and its record.

NOTHING HERE IMPORTS CuPy. The lift is string work over the certified device text and
the resolution is emitted from ``fields.IYEE_SHIFTS``, so both are exercisable at the
merge bar on a laptop -- the rule this package applies to every part whose failure mode
is a silent wrong answer rather than a crash.

=============================================================================
THE DESIGN IS THE REAL WELD'S, AND THE RECORDED REFUSAL DOES NOT REACH IT
=============================================================================

The board records both cells STENCIL-BLOCKED with

    "the constitutive half reads the curl half's IN-PLACE output at a cell this
     thread does not own. In one launch that cell is written by another block, and
     CUDA offers no grid-wide barrier inside an ordinary launch"

-- true, and true OF THE IN-PLACE WELD. :mod:`.offdiag_stencil_weld`'s docstring is
where that argument lives and it is not restated here. Both premises go the same way:
the curl half writes a LAUNCH-LOCAL SCRATCH so ``D``, ``fu_D`` and ``H`` are ``const``
for the whole launch, and every foreign tap is a RECOMPUTE from that unwritten state
through :func:`raw_step_D_cell_source`'s lift of the certified complex curl body.

=============================================================================
WHAT COMPLEX STORAGE CHANGES, AND WHAT WAS MEASURED BEFORE IT WAS SPELLED
=============================================================================

THE STORAGE ITSELF CHANGES NOTHING ABOUT THE ADDRESSING. A complex64 volume is bound
as its float32 word view and every index in this module is a COMPLEX CELL index;
``cf_load``/``cf_store`` do the word doubling and nothing else does. The Yee tables,
the redirect rows, the wall planes and the reflect rows are INDEX facts -- they mention
no dtype -- so :func:`.offdiag_stencil_weld.near_axes`, ``.far_axis``, ``.clear_axes``
and ``.fill_plan`` are IMPORTED rather than respelled, and one home keeps the two
storage widths from disagreeing about which plane is the ghost.

WHAT DOES CHANGE IS EVERY MULTIPLY, AND THERE ARE EXACTLY TWO OF THEM:

1. **THE PARITY IS A PART-WISE SCALE, AT BOTH SIGNS.** The array path writes a mirror
   ghost through ``host_writes.copy_face``, which forms ``part_products.scale(phase,
   plane)``: the real part and the imaginary part each multiplied by the float32
   parity, one rounding per part. The emitted line is the certified
   ``complex_emitter.mul_coefficient_left`` -- the same helper
   :mod:`.complex_fill_carry` and :mod:`.complex_electric_fill_carry` apply -- for
   EVERY imaged plane at EVERY sign, so the resolution performs the array path's
   multiply rather than an equivalent rewrite of it (a negation, or no line at +1).

2. **THE WALL CLEAR IS ``cf_zero()``, BOTH WORDS.** ``stepping._zero_metal`` clears
   through ``host_writes.zero_faces``, which assigns the INTEGER 0 to a complex64
   plane: the pair ``(+0.0f, +0.0f)`` and not a sign-carrying zero. A clear that
   touched only the real plane is a defect class this tree has paid for; the gate
   arms it.

THE ORDER OF THE CLEAR AND THE PARITIES IS LOAD-BEARING. ``mul_coefficient_left(-1.0f,
cf_zero())`` is ``(-0.0f, -0.0f)``, so a cleared plane that is then multiplied is NOT
the cleared plane. The array path's pass order (near fill, wall clear, far fill:
``fill_symmetry_bc_D``, ``zero_metal_D``, ``fill_folded_far_ghosts_D``) is therefore
transcribed exactly: the clear REPLACES the near arm and its parities, and the FAR
parity is applied to whatever the clear left. That is the composition
:mod:`.offdiag_stencil_weld` emits one storage width down.

=============================================================================
THE TWO ARMS
=============================================================================

``pml`` lifts the certified ``pml_apply`` (``complex_emitter``'s, transcribing
``stepping._apply_curl_regions``' two per-chunk forms) into a PURE form that stores
nothing: the split-field value comes back through an out-parameter and the
displacement is RETURNED. ``no_pml`` lifts ``no_pml_apply`` (``complex_no_pml_kernels``,
``target -= curl``) the same way; there is no ``fu`` at all, so the scratch is three
volumes rather than six and the resolution's raw call takes no auxiliary. Both edits
keep the right-hand side, its tree and its operand order character for character --
what moves is where the value goes.

BOTH ARMS BIND THE SAME CHUNK-FORM PACK. Each family's signature carries the run's
chunk record, the live rows' presence words and the two connection tables under the
names the pack and the certified constitutive body read, so one derivation serves
both: the no-absorber curl body decodes only the record's partner-presence word (no
chunk carries a layer without an absorber), the split-field one its membership rows
too.

THE OFF-DIAGONAL TERM IS THE CHUNK-FORM ONE (``complex_offdiag_update_e``): MEEP's
double row per part, the per-chunk presence words, and two connection tables -- the
pair's same-index operand on the PARTNER axis takes ``eikna`` (the D curl's own table,
so one kernel triple serves both halves) and the near product on the OWN axis takes
``conj(eikna)``. Every foreign sample it reads is a plain stored word on the array
path (the shifted operands are plain wraps), so the resolution recomputes the stored
word and the certified term applies the connections itself.

WHAT THIS MODULE DOES NOT DO: it plans nothing and launches nothing.
``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch, and both
families are opt-in through ``arms.plan_step(..., fuse=True)``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

try:
    from . import offdiag_stencil_weld as _real_weld
except ImportError:  # loaded by path, outside the package: the host probes
    import importlib.util as _importlib_util
    import os as _os

    def _load(stem: str) -> Any:
        here = _os.path.dirname(_os.path.abspath(__file__))
        spec = _importlib_util.spec_from_file_location(
            f"cuda_kernels_{stem}", _os.path.join(here, f"{stem}.py"))
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    _real_weld = _load("offdiag_stencil_weld")

#: THE INDEX FACTS ARE IMPORTED, NOT RESPELLED. Every one of these is a statement
#: about the Yee lattice or about a driver pass's stored rows, and none of them
#: mentions a dtype -- so the complex weld asks the same functions the real weld does
#: and the two storage widths cannot disagree about which plane a fill images.
D_TARGETS: Tuple[str, str, str] = _real_weld.D_TARGETS
NEAR_SOURCE_ROW: int = _real_weld.NEAR_SOURCE_ROW
near_axes = _real_weld.near_axes
far_axis = _real_weld.far_axis
clear_axes = _real_weld.clear_axes
fill_plan = _real_weld.fill_plan
split_body = _real_weld.split_body
the_line_starting = _real_weld.the_line_starting
close_call_with_the_pack = _real_weld.close_call_with_the_pack

# The certified curl text. Taken DEFENSIVELY, for the reason every module on this
# track takes it defensively: both modules import CuPy at scope, and the predicate
# half of every family built on this must answer on a host with no device.
try:
    from . import complex_emitter
except Exception:  # noqa: BLE001 - no CuPy, or no device toolchain
    complex_emitter = None  # type: ignore[assignment]
try:
    from . import complex_folded_kernels
except Exception:  # noqa: BLE001
    complex_folded_kernels = None  # type: ignore[assignment]
try:
    from . import complex_no_pml_kernels
except Exception:  # noqa: BLE001
    complex_no_pml_kernels = None  # type: ignore[assignment]
try:
    from . import complex_offdiag_update_e
except Exception:  # noqa: BLE001
    complex_offdiag_update_e = None  # type: ignore[assignment]
try:
    from . import coverage as _coverage
except Exception:  # noqa: BLE001
    _coverage = None  # type: ignore[assignment]

__all__ = [
    "ARMS", "CURL_LIFT_EDITS", "D_TARGETS", "NEAR_SOURCE_ROW",
    "RESOLUTION_LIFT_EDITS", "clear_axes", "close_call_with_the_pack",
    "constitutive_body", "constitutive_prelude_defines",
    "constitutive_prelude_helpers", "curl_prelude", "far_axis", "fill_plan",
    "near_axes", "normalized_arm",
    "raw_step_D_cell_source", "resolution_source", "split_body",
    "the_line_starting", "weld_args_construction", "weld_args_fields",
    "weld_args_struct",
]

#: The two arms this lift serves, in the order every table walks them. ``pml`` is
#: the split-field recurrence and ``no_pml`` the plain subtraction; the names are the
#: ones ``complex_offdiag_update_e.ARMS`` already uses for the constitutive tails, so
#: a product's curl arm and its constitutive tail are spelled with one vocabulary.
ARMS: Tuple[str, str] = ("pml", "no_pml")

_AXIS = ("x", "y", "z")
_COORD = ("i", "j", "k")
_EXTENT = ("nx", "ny", "nz")

#: Separates a certified kernel's signature from its body, and the marker that
#: identifies its single entry point. Every certified string on this track closes its
#: parameter list on its own line.
_GLOBAL_MARKER = 'extern "C" __global__ void '
_BODY_ANCHOR = "\n) {\n"

#: The certified complex curl body's thread preamble and its index decode -- the two
#: blocks a ``__device__`` function evaluated at an ARBITRARY cell cannot keep. The
#: STRIDE block between them is NOT touched: it is a function of the extents alone.
_THREAD_PREAMBLE = ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                    "    if (idx >= nx * ny * nz) return;\n")
_INDEX_DECODE = ("    int k = idx % nz;\n"
                 "    int j = (idx / nz) % ny;\n"
                 "    int i = idx / (ny * nz);\n")

#: ``complex_emitter``'s certified ``pml_apply``, and what the PURE form makes of it.
#: Anchored on the exact certified spelling so an upstream edit is a named failure.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_PURE = "__device__ __forceinline__ cf pml_apply_pure(\n"
_PML_APPLY_PARAMETERS = (
    "    float* f, float* fu, int idx, cf curl,\n"
    "    float kms, float sinv, float kms_u, float sinv_u, int dsigu\n)")
_PML_APPLY_PARAMETERS_PURE = (
    "    const float* f, const float* fu, int idx, cf curl,\n"
    "    float kms, float sinv, float kms_u, float sinv_u, int dsigu, cf* fu_out\n)")
_PML_APPLY_FU_STORE = "    if (dsigu) cf_store(fu, idx, fu_new);\n"
_PML_APPLY_FU_STORE_PURE = (
    "    // THE FIRST OF THE TWO STORES, HANDED BACK INSTEAD OF PERFORMED. The\n"
    "    // caller writes it to the SCRATCH volume, so nothing this launch reads\n"
    "    // is ever a word this launch wrote. The scratch replaces the stored split\n"
    "    // field at every cell, so on a cell outside the recurrence's chunks --\n"
    "    // where the certified helper leaves fu untouched -- it carries the\n"
    "    // pre-launch word.\n"
    "    *fu_out = dsigu ? fu_new : cf_load(fu, idx);\n")
_PML_APPLY_F_STORE = "    cf_store(f, idx, value);\n"
_PML_APPLY_F_STORE_PURE = (
    "    // THE SECOND STORE, LIKEWISE. Same right-hand side, same tree, same\n"
    "    // helpers; the value is RETURNED so the resolution can redirect it, the\n"
    "    // wall clear can replace it with cf_zero() and the constitutive can\n"
    "    // consume it from a register.\n"
    "    return value;\n")

#: ``complex_no_pml_kernels``' certified ``no_pml_apply``, and its PURE form. The
#: whole helper is three lines and the edit is the same one: the store leaves and the
#: subtraction is returned.
_NO_PML_APPLY = ("__device__ __forceinline__ void no_pml_apply(float* f, int idx, "
                 "cf curl) {\n"
                 "    cf_store(f, idx, cf_sub(cf_load(f, idx), curl));\n"
                 "}\n")
_NO_PML_APPLY_PURE = (
    "// THE PURE FORM. Same subtraction, same operand order, same helper; the value\n"
    "// is RETURNED and the caller writes it to the SCRATCH volume, so no thread in\n"
    "// this launch can observe another thread's store.\n"
    "__device__ __forceinline__ cf no_pml_apply_pure(const float* f, int idx,\n"
    "                                                cf curl) {\n"
    "    return cf_sub(cf_load(f, idx), curl);\n"
    "}\n")

#: Every line of certified CURL text the weld does not lift verbatim, with the reason.
#: DATA, not prose, so a gate and a host suite can assert the list. Keyed by arm,
#: because the two tails are different certified helpers.
CURL_LIFT_EDITS: Dict[str, Tuple[Dict[str, str], ...]] = {
    "pml": (
        {"line": "__device__ __forceinline__ void pml_apply(",
         "became": "__device__ __forceinline__ cf pml_apply_pure(",
         "why": "the weld writes a SCRATCH volume, so the certified helper must not "
                "store. It returns the displacement and hands the split-field value "
                "back through an out-parameter; f and fu become const because "
                "nothing in this launch writes them."},
        {"line": "    if (dsigu) cf_store(fu, idx, fu_new);",
         "became": "    *fu_out = dsigu ? fu_new : cf_load(fu, idx);",
         "why": "the store moves to the caller, which addresses the scratch. The "
                "value and the expression that produced it are untouched on the "
                "recurrence's cells; elsewhere the scratch, which replaces the "
                "stored split field at every cell, carries the pre-launch word the "
                "certified helper leaves there (the auxiliary contract)."},
        {"line": "    cf_store(f, idx, value);",
         "became": "    return value;",
         "why": "same right-hand side, same helpers, same operand orientation on "
                "both per-chunk forms; the result is returned rather than stored, "
                "because the resolution may redirect it, the wall clear may "
                "replace it with cf_zero(), and the constitutive half consumes the "
                "thread's own cell from a register."},
        {"line": "        pml_apply(f*, u*, idx, curl, ...);",
         "became": "        d_out[n] = pml_apply_pure(f*, u*, idx, curl, ..., "
                   "&fu_out[n]);",
         "why": "capture both values. The argument list, its order and the "
                "coefficient pairing are untouched; what is added is where the two "
                "results go. Three lines, one per component, each derived from the "
                "certified line by prefix and suffix rather than retyped."},
    ),
    "no_pml": (
        {"line": "__device__ __forceinline__ void no_pml_apply(float* f, int idx, "
                 "cf curl) { cf_store(f, idx, cf_sub(cf_load(f, idx), curl)); }",
         "became": "__device__ __forceinline__ cf no_pml_apply_pure(const float* f, "
                   "int idx, cf curl) { return cf_sub(cf_load(f, idx), curl); }",
         "why": "the whole helper is the store, so the PURE form returns the "
                "subtraction the certified body would have stored. The operands, "
                "their order and cf_sub are untouched -- and the certified module's "
                "own caveat is preserved by construction: this is the direct "
                "subtraction, never the split-field binding with unit coefficients, "
                "which differs at target = -0.0 and curl = +0.0."},
        {"line": "        no_pml_apply(f*, idx, curl);",
         "became": "        d_out[n] = no_pml_apply_pure(f*, idx, curl);",
         "why": "capture the value. The argument list and its order are untouched; "
                "what is added is where the result goes. There is no auxiliary on "
                "this arm, so nothing else is captured."},
    ),
}

#: The two blocks BOTH arms drop, spelled once. Appended to each arm's list by
#: :func:`curl_lift_edits`, which is what a suite and a gate assert.
_SHARED_CURL_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= nx * ny * nz) return;",
     "became": "    const int idx = (i * (ny * nz)) + (j * nz) + k;",
     "why": "THE WHOLE POINT OF THE LIFT. The certified body becomes a __device__ "
            "function evaluated at an ARBITRARY cell rather than at the thread's "
            "own, so the cell arrives as three coordinates and the flat index is "
            "COMPOSED from them by the same layout the decode inverts -- in COMPLEX "
            "CELLS, exactly as the certified body's own strides are. The bounds "
            "guard leaves with the thread index: every caller is either the "
            "kernel's own guarded thread or a redirect target the resolution proved "
            "in range (stored row NEAR_SOURCE_ROW on an axis the predicate requires "
            "more cells of, or a reflect row in [0, n)), and every ghosted tap is "
            "filtered by `index < 0` before it reaches here."},
    {"line": "    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
             "    int i = idx / (ny * nz);",
     "became": "(removed -- i, j and k are parameters)",
     "why": "the decode's inverse is now the caller's; keeping it would redeclare "
            "the three parameters and the kernel would not compile. The STRIDE block "
            "above it is NOT removed: sx/sy/sz are a function of the extents alone "
            "and every ghost helper the body calls indexes through them."},
)

#: Every line of certified CONSTITUTIVE text a family module rewrites. Shared here
#: because both families make the same edits for the same reasons; each module
#: asserts the ones it actually applied.
RESOLUTION_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    cf gs_E* = cf_load(D*, idx);",
     "became": "    cf gs_E* = v_*;",
     "why": "THE SEAM, own-cell half. The certified constitutive body opens each "
            "component by loading the flux density; the weld has already resolved "
            "the thread's own cell into a register, and that register is the "
            "post-fill, post-clear value the array path would have loaded."},
    {"line": "        D*, chi1inv_E*_E*, idx,",
     "became": "        <component tag>, v_*, chi1inv_E*_E*, idx,",
     "why": "THE SEAM, foreign half. The partner volume becomes a COMPONENT TAG plus "
            "the partner's own-cell register: the term's home sample is the register "
            "and its three foreign samples are recomputed through resolve_D. The "
            "coefficient argument, the home index, the three shifted indices, the "
            "ghost lane, the parity weight and both connections are untouched, and "
            "the argument pack is appended to the call."},
    {"line": "    cf near_pair = cf_add(pair_window(g, home, pa, pph, pphase), "
             "down_sample(g, down, dgl, dbc, w));",
     "became": "    cf near_pair = cf_add(cconnect(g_home, pa, pph, pphase), "
               "resolved_down_sample(comp, down, dgl, dbc, w, weld));",
     "why": "the same addition of the same two samples. The home one is a register "
            "connected by the same cconnect pair_window applies to its load (home "
            "is never the -1 ghost), and the foreign one is RECOMPUTED from "
            "pre-launch state instead of loaded from a volume another block is "
            "writing. resolved_down_sample is down_sample's body with the load "
            "replaced -- the index < 0 arm and the ghost-lane mirror weight are "
            "unchanged."},
    {"line": "    cf far_pair = cf_add(pair_window(g, up, pa, pph, pphase), "
             "down_sample(g, corner, dgl, dbc, w));",
     "became": "    cf far_pair = cf_add(resolved_pair_window(comp, up, pa, pph, "
               "pphase, weld), resolved_down_sample(comp, corner, dgl, dbc, w, "
               "weld));",
     "why": "the same, for the term's other two foreign samples. "
            "resolved_pair_window is pair_window's body with the load replaced. "
            "The coefficient multiply BETWEEN the two shifts, the own-axis "
            "connection on the near product, the ghosted coefficient at a "
            "zero-ghost up plane and the sum are untouched -- offdiag_term's last "
            "three lines are verbatim."},
)


def normalized_arm(arm: str) -> str:
    """``"pml"`` or ``"no_pml"``, or a named failure.

    REQUIRED AND NEVER DEFAULTED: the two arms are two different recurrences, and a
    wrong one is a wrong answer rather than a crash.
    """
    if arm not in ARMS:
        raise ValueError(
            f"arm must be one of {ARMS!r}, got {arm!r}; the split-field recurrence "
            f"and the plain subtraction are different arithmetic and this lift will "
            f"not pick one for a caller")
    return arm


def curl_lift_edits(arm: str) -> Tuple[Dict[str, str], ...]:
    """One arm's edit list, its own plus the two both arms make."""
    return CURL_LIFT_EDITS[normalized_arm(arm)] + _SHARED_CURL_EDITS


# ---------------------------------------------------------------------------
# The certified text
# ---------------------------------------------------------------------------

def _certified(arm: str) -> None:
    """Refuse BY NAME on a host with no device toolchain."""
    normalized_arm(arm)
    missing = [name for name, module in
               (("complex_emitter", complex_emitter),
                ("complex_folded_kernels", complex_folded_kernels),
                ("complex_no_pml_kernels", complex_no_pml_kernels))
               if module is None]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} is not importable on this host (it imports CuPy "
            f"at module scope), so there is no certified complex curl text to "
            f"splice. Both families' predicates need none of it and still answer")


def certified_curl_source(arm: str) -> str:
    """The certified curl kernel this arm lifts, whole.

    ``pml`` takes ``complex_folded_kernels.folded_source("step_D", ...)`` -- the
    FOLDED complex curl, because the only board cell this arm serves has a fold on it
    -- and ``no_pml`` takes ``complex_no_pml_kernels._curl_source("step_D", False,
    ...)``, the PLAIN (non-conductive) no-absorber curl. A conductive no-absorber run
    is refused BY NAME by that family's predicate clause rather than served with the
    wrong tail.
    """
    _certified(arm)
    if arm == "pml":
        return complex_folded_kernels.folded_source("step_D")
    return complex_no_pml_kernels._curl_source("step_D", False)


def _split_emitted(source: str, name: str) -> Tuple[str, str]:
    """``(prelude, body)`` of one emitted certified kernel, or a named failure."""
    if source.count(_GLOBAL_MARKER) != 1:
        raise AssertionError(
            f"{name} carries {source.count(_GLOBAL_MARKER)} global entry points, "
            f"not 1; the prelude cannot be identified")
    prelude, rest = source.split(_GLOBAL_MARKER, 1)
    if rest.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {rest.count(_BODY_ANCHOR)} signature terminators, not "
            f"1; the body anchor no longer identifies the signature")
    body = rest.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return prelude, body[: -len("}\n")]


def curl_prelude(arm: str) -> str:
    """The certified complex curl prelude with its tail helper made PURE.

    ``cf_load``/``cf_store``, ``cf_add``/``cf_sub``, ``cf_zero``, the three multiply
    orientations and both ghost helpers are lifted UNTOUCHED -- they read ``H`` and
    the pre-launch ``D``, which no thread writes.
    """
    prelude, _body = _split_emitted(certified_curl_source(arm),
                                    f"the certified {arm} complex step_D curl")
    if arm == "pml":
        for anchor, what in ((_PML_APPLY_SIGNATURE, "pml_apply's declaration"),
                             (_PML_APPLY_PARAMETERS, "pml_apply's parameter list"),
                             (_PML_APPLY_FU_STORE, "pml_apply's split-field store"),
                             (_PML_APPLY_F_STORE, "pml_apply's displacement store")):
            if prelude.count(anchor) != 1:
                raise AssertionError(
                    f"the certified complex curl prelude carries "
                    f"{prelude.count(anchor)} copies of {what}, not one; the pure "
                    f"rewrite has no anchor")
        prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_PURE, 1)
        prelude = prelude.replace(_PML_APPLY_PARAMETERS,
                                  _PML_APPLY_PARAMETERS_PURE, 1)
        prelude = prelude.replace(_PML_APPLY_FU_STORE, _PML_APPLY_FU_STORE_PURE, 1)
        prelude = prelude.replace(_PML_APPLY_F_STORE, _PML_APPLY_F_STORE_PURE, 1)
        return prelude
    if prelude.count(_NO_PML_APPLY) != 1:
        raise AssertionError(
            f"the certified no-absorber complex prelude carries "
            f"{prelude.count(_NO_PML_APPLY)} copies of no_pml_apply, not one; the "
            f"pure rewrite has no anchor")
    return prelude.replace(_NO_PML_APPLY, _NO_PML_APPLY_PURE, 1)


# ---------------------------------------------------------------------------
# The argument pack
# ---------------------------------------------------------------------------

#: The members BOTH arms carry, in emission order. A struct rather than forty
#: parameters threaded through four helpers: the members are built once from the
#: kernel's own arguments, are uniform across the block, and name exactly the state
#: the resolution reads.
_SHARED_WELD_ARGS: Tuple[Tuple[str, str], ...] = (
    ("const float* __restrict__", "f0"), ("const float* __restrict__", "f1"),
    ("const float* __restrict__", "f2"),
    ("const float* __restrict__", "g0"), ("const float* __restrict__", "g1"),
    ("const float* __restrict__", "g2"),
    ("int", "nx"), ("int", "ny"), ("int", "nz"), ("float", "dtdx"),
    ("int", "bc_x"), ("int", "bc_y"), ("int", "bc_z"),
    ("int", "ph_x"), ("int", "ph_y"), ("int", "ph_z"),
    ("float", "pxr"), ("float", "pxi"), ("float", "pyr"), ("float", "pyi"),
    ("float", "pzr"), ("float", "pzi"),
    ("int", "wall_x"), ("int", "wall_y"), ("int", "wall_z"),
    ("int", "near_x"), ("int", "near_y"), ("int", "near_z"),
    ("int", "reflect_x"), ("int", "reflect_y"), ("int", "reflect_z"),
    ("float", "near_phase_x"), ("float", "near_phase_y"),
    ("float", "near_phase_z"),
    ("float", "far_phase_x"), ("float", "far_phase_y"), ("float", "far_phase_z"),
    # The run's chunk record: the certified curl body's decode block, lifted with
    # it, reads the membership and the partner presence at the cell this function
    # is evaluated at -- a RECOMPUTED cell's chunk, not the thread's.
    ("const int* __restrict__", "cb"),
)

#: The split-field arm's extra members: the auxiliary volumes the recurrence reads
#: and the six INTEGER sub-lattice coefficient vectors.
_PML_WELD_ARGS: Tuple[Tuple[str, str], ...] = (
    ("const float* __restrict__", "u0"), ("const float* __restrict__", "u1"),
    ("const float* __restrict__", "u2"),
    ("const float* __restrict__", "kms_x"), ("const float* __restrict__", "sinv_x"),
    ("const float* __restrict__", "kms_y"), ("const float* __restrict__", "sinv_y"),
    ("const float* __restrict__", "kms_z"), ("const float* __restrict__", "sinv_z"),
)


def weld_args_fields(arm: str) -> Tuple[Tuple[str, str], ...]:
    """The pack's members for one arm, in emission order.

    THE FIELD AND AUXILIARY POINTERS KEEP THE CERTIFIED CURL BODY'S OWN NAMES
    (``f0``..``g2``, ``u0``..``u2``): the lifted body is spliced verbatim and reads
    them, so an unpack under any other name would be a rename inside certified text.
    """
    normalized_arm(arm)
    fields = list(_SHARED_WELD_ARGS)
    if arm == "pml":
        # Inserted directly after the three D volumes, which is where the certified
        # signature carries them.
        fields[3:3] = list(_PML_WELD_ARGS[:3])
        fields += list(_PML_WELD_ARGS[3:])
    return tuple(fields)


def weld_args_struct(arm: str) -> str:
    """The argument pack, declared."""
    lines = [
        "",
        "// THE PRE-LAUNCH STATE AND THE RUNTIME PLAN, packed once. Every member is",
        "// read-only for the whole launch: that is the property the recompute rests",
        "// on, and binding the field pointers const here is where it is stated in",
        "// the type system rather than in a comment.",
        "struct WeldArgs {",
    ]
    for kind, name in weld_args_fields(arm):
        lines.append(f"    {kind} {name};")
    lines.append("};")
    return "\n".join(lines) + "\n"


def weld_args_construction(arm: str, indent: str = "    ",
                           boundary_prefix: str = "bc_") -> str:
    """The struct, built from the kernel's own parameters. One line per member.

    ``boundary_prefix`` is ``"bc_"`` where one boundary triple serves both halves and
    ``"bc_"`` still where the constitutive binds its own ``cbc_*``: THE CURL'S TRIPLE
    IS ALWAYS THE ONE THE PACK CARRIES, because the pack feeds the lifted curl body
    and nothing else. It is a parameter so a family that renamed the CURL's triple
    would have to say so here rather than silently pack the constitutive's.

    THE CONNECTION TABLE NEEDS NO ALIAS. The D curl reads the not-owned word ABOVE
    the stored window (``eikna``, ``complex_pml_kernels.bloch_phase_arguments(grid,
    True)``), and the off-diagonal pair's same-index operand reads the same word on
    its partner axis -- ``complex_offdiag_update_e.bloch_connection_table``'s pair
    half is that very call. So one kernel triple (``ph_*``, ``p*r``/``p*i``) feeds
    both halves under the curl's own names; the near product's own-axis table
    (``oh_*``, ``o*r``/``o*i``) is the constitutive's alone and never enters the pack.
    """
    lines = [f"{indent}WeldArgs weld;"]
    for _kind, name in weld_args_fields(arm):
        if name in ("bc_x", "bc_y", "bc_z"):
            source = boundary_prefix + name[-1]
        else:
            source = name
        lines.append(f"{indent}weld.{name} = {source};")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def raw_step_D_cell_source(arm: str) -> str:
    """The certified complex ``step_D`` body as a ``__device__`` function of a CELL.

    THE ENTIRE BODY IS LIFTED. Only the two blocks that tie it to a thread index and
    the three tail call lines are edited, each through an exact anchor, so a change
    in the certified text is a named failure rather than a silently different kernel.
    The three component blocks -- their stencils, their ghost calls, their phase
    arguments, their fold masks and their coefficient pairings -- arrive character
    for character.

    ``d_out[c]`` is the RAW stepped displacement (before any in-seam pass). On the
    ``pml`` arm ``fu_out[c]`` is the split-field value, which no in-seam pass ever
    touches (``stepping._write_mirror_ghost`` writes ``field`` and never ``fu_field``,
    :1451; ``_zero_metal`` writes only the named components, :2247); the ``no_pml``
    arm has no auxiliary and takes no such parameter.
    """
    arm = normalized_arm(arm)
    _prelude, body = _split_emitted(certified_curl_source(arm),
                                    f"the certified {arm} complex step_D curl")
    for anchor, what in ((_THREAD_PREAMBLE, "the thread-index preamble"),
                         (_INDEX_DECODE, "the index decode")):
        if body.count(anchor) != 1:
            raise AssertionError(
                f"the certified {arm} complex curl body carries "
                f"{body.count(anchor)} copies of {what}, not one; the cell lift has "
                f"no anchor")
    body = body.replace(
        _THREAD_PREAMBLE,
        "    // THE CELL ARRIVES AS COORDINATES. The flat index is composed by the\n"
        "    // same layout the certified decode inverts -- in COMPLEX CELLS, which\n"
        "    // is what the strides below are in too; the bounds guard left with the\n"
        "    // thread index, and every caller is either the kernel's own guarded\n"
        "    // thread or a redirect target the resolution proved in range.\n"
        "    const int idx = (i * (ny * nz)) + (j * nz) + k;\n", 1)
    body = body.replace(_INDEX_DECODE, "", 1)

    for component in range(3):
        if arm == "pml":
            prefix = f"        pml_apply(f{component}, u{component}, idx, curl, "
            line = the_line_starting(
                body, prefix, f"target {component}'s certified pml_apply call")
            if not line.endswith(");"):
                raise AssertionError(
                    f"target {component}'s certified pml_apply call does not close "
                    f"on its own line ({line!r}); the capture has no suffix to take")
            captured = (f"        d_out[{component}] = pml_apply_pure("
                        + line[len("        pml_apply("):-len(");")]
                        + f", &fu_out[{component}]);")
        else:
            prefix = f"        no_pml_apply(f{component}, idx, curl"
            line = the_line_starting(
                body, prefix, f"target {component}'s certified no_pml_apply call")
            if line != f"        no_pml_apply(f{component}, idx, curl);":
                raise AssertionError(
                    f"target {component}'s certified no-absorber tail is spelled "
                    f"{line!r}; this lift captures the value that call would have "
                    f"stored and cannot do so on a line whose arguments have moved")
            captured = (f"        d_out[{component}] = no_pml_apply_pure("
                        f"f{component}, idx, curl);")
        body = body.replace(line, captured, 1)

    auxiliary = ", cf* fu_out" if arm == "pml" else ""
    signature = (
        "\n// The certified complex step_D curl, evaluated at ONE ARBITRARY CELL and\n"
        "// storing nothing. This is the function the constitutive half calls for\n"
        "// every foreign sample it needs: a pure function of the pre-launch D"
        + (", fu\n// and H" if arm == "pml" else " and\n// H") + ", none of which "
        "this launch writes, so a foreign evaluation cannot\n"
        "// depend on which block ran first.\n"
        "__device__ __forceinline__ void raw_step_D_cell(\n"
        "    int i, int j, int k, const WeldArgs& weld, cf* d_out"
        + auxiliary + "\n) {\n")
    unpack = ["    // The certified body's own names, bound to the pack."]
    scalars = [(kind, name) for kind, name in weld_args_fields(arm)
               if "*" not in kind]
    pointers = [(kind, name) for kind, name in weld_args_fields(arm)
                if "*" in kind]
    for kind, name in pointers:
        unpack.append(f"    {kind} {name} = weld.{name};")
    for kind, name in scalars:
        if name.startswith(("wall_", "near_", "reflect_", "far_phase")):
            continue           # the resolution's plan; the curl body never reads it
        unpack.append(f"    const {kind} {name} = weld.{name};")
    return signature + "\n".join(unpack) + "\n" + body + "}\n"


def resolution_source(arm: str) -> str:
    """``resolve_D`` -- the post-pass value of one D component at any cell, complex.

    Three specialized helpers plus two dispatchers and the two ghosted forms the
    constitutive's own samplers need. The AXIS SETS are compile time (they are the
    component's Yee shifts); WHICH of them is live is runtime, which is this track's
    standing rule -- one device string per family, so one certification digest covers
    every fold orientation and no per-axis variant can end up pinned that nobody ran.

    THE COMPOSITION IS THE ARRAY PATH'S PASS ORDER and under complex storage it moves
    bytes: see this module's docstring. The clear REPLACES the near arm rather than
    following it, because ``mul_coefficient_left(-1.0f, cf_zero())`` is
    ``(-0.0f, +0.0f)`` and the array path's clear is the last thing to touch that
    plane before the far fill reads it.

    THE TWO SAMPLERS ARE THE CHUNK-FORM TERM'S, with the load resolved: the
    off-diagonal term reads every foreign D sample as a plain stored word (the
    array path's shifted operands are plain wraps), and applies the partner-axis
    window and the down-sample mirror weight itself -- so ``resolved_pair_window``
    and ``resolved_down_sample`` are ``pair_window`` and ``down_sample`` with
    ``cf_load`` replaced by :func:`resolve_D_at` and nothing else moved.
    """
    arm = normalized_arm(arm)
    auxiliary = ", fu" if arm == "pml" else ""
    declare_fu = "        cf fu[3];\n" if arm == "pml" else ""
    blocks: List[str] = [
        "",
        "// stepping.MIRROR_SOURCE_INDEX (stepping.py:159): MEEP's io = -2 halved",  # stepping.py live lines for the frozen device-text citation(s) in this string: 159->160
        "// origin puts the ghost at stored cell 0 onto stored cell 2. The predicate",
        "// refuses a folded axis with no more than this many stored cells, which is",
        "// what makes the redirect below in range without a guard.",
        f"#define NEAR_SOURCE_ROW {NEAR_SOURCE_ROW}",
        "",
    ]
    for component, target in enumerate(D_TARGETS):
        own = far_axis(component)
        shifts = _real_weld.IYEE_SHIFTS[target]
        near = near_axes(component)
        lines = [
            f"// {target} {shifts}: the three in-seam passes, resolved per cell.",
            f"//   near fill  -- axes {tuple(_AXIS[a] for a in near)} (Yee shift 0),",
            f"//                 stored cell 0 images stored cell NEAR_SOURCE_ROW",
            f"//                 through mul_coefficient_left(+phase, .), the",
            f"//                 part-wise scale copy_face forms, at BOTH signs",
            f"//                 (stepping._write_mirror_ghost);",
            f"//   wall clear -- the SAME two axes (the same Yee test,",
            f"//                 stepping._zero_metal), stored cell 0 -> cf_zero(),",
            f"//                 and it runs AFTER the near fill (fill_symmetry_bc_D,",
            f"//                 then zero_metal_D) so a near ghost in a cleared plane",
            f"//                 is CLEARED -- (+0, +0), which is not the cleared",
            f"//                 plane times a parity, and is why the clear replaces",
            f"//                 the arm;",
            f"//   far fill   -- axis {_AXIS[own]} (Yee shift 1), the top stored plane",
            f"//                 images the reflect row through",
            f"//                 mul_coefficient_left(-phase, .), AFTER the clear",
            f"//                 (fill_folded_far_ghosts_D).",
            f"__device__ __forceinline__ cf resolve_{target}("
            f"int i, int j, int k, const WeldArgs& weld) {{",
            f"    const int far = (weld.reflect_{_AXIS[own]} >= 0)"
            f" && ({_COORD[own]} == weld.{_EXTENT[own]} - 1);",
            f"    if (far) {_COORD[own]} = weld.reflect_{_AXIS[own]};",
        ]
        cleared = " || ".join(f"(weld.wall_{_AXIS[a]} && {_COORD[a]} == 0)"
                              for a in near)
        lines += [
            "    cf value;",
            f"    if ({cleared}) {{",
            "        // The array path assigns the INTEGER 0 to a complex64 array",
            "        // here, which is the word pair (+0.0f, +0.0f).",
            "        value = cf_zero();",
            "    } else {",
        ]
        for axis in near:
            lines.append(
                f"        const int image_{_AXIS[axis]} ="
                f" weld.near_{_AXIS[axis]} && {_COORD[axis]} == 0;")
        source = []
        for axis in range(3):
            if axis in near:
                source.append(f"image_{_AXIS[axis]} ? NEAR_SOURCE_ROW"
                              f" : {_COORD[axis]}")
            else:
                source.append(_COORD[axis])
        lines += [
            "        cf raw[3];",
            declare_fu.rstrip("\n") if declare_fu else None,
            f"        raw_step_D_cell({source[0]}, {source[1]}, {source[2]},"
            f" weld, raw{auxiliary});",
            f"        value = raw[{component}];",
        ]
        lines = [line for line in lines if line is not None]
        if near:
            lines += [
                "        // ONE CERTIFIED mul_coefficient_left PER IMAGED PLANE,",
                "        // NESTED and in X, Y, Z order, because the fill writes",
                "        // plane after plane: a corner unowned on two planes carries",
                "        // the composition of both parities as two separate part-wise",
                "        // scales, never their pre-multiplied product, and at BOTH",
                "        // signs: the array path's multiply, not a rewrite of it.",
            ]
        for axis in near:                     # X, Y, Z order: the fill's own
            lines.append(
                f"        if (image_{_AXIS[axis]})"
                f" value = mul_coefficient_left(weld.near_phase_{_AXIS[axis]},"
                f" value);")
        lines += [
            "    }",
            f"    if (far) value = mul_coefficient_left(weld.far_phase_"
            f"{_AXIS[own]}, value);",
            "    return value;",
            "}",
            "",
        ]
        blocks.append("\n".join(lines))

    blocks.append("\n".join([
        "// The component tag is a compile-time constant at every call site, so this",
        "// chain is resolved by the compiler and no branch survives it.",
        "__device__ __forceinline__ cf resolve_D(int comp, int i, int j, int k,",
        "                                       const WeldArgs& weld) {",
        "    if (comp == 0) return resolve_Dx(i, j, k, weld);",
        "    if (comp == 1) return resolve_Dy(i, j, k, weld);",
        "    return resolve_Dz(i, j, k, weld);",
        "}",
        "",
        "// The same, addressed by the FLAT COMPLEX CELL index the certified",
        "// constitutive body already computes for its taps -- decoded by the layout",
        "// flat() composes.",
        "__device__ __forceinline__ cf resolve_D_at(int comp, int index,",
        "                                          const WeldArgs& weld) {",
        "    const int k = index % weld.nz;",
        "    const int j = (index / weld.nz) % weld.ny;",
        "    const int i = index / (weld.ny * weld.nz);",
        "    return resolve_D(comp, i, j, k, weld);",
        "}",
        "",
        "// pair_window()'s body with the load resolved: index -1 is the own-axis",
        "// zero ghost (+0, +0), and every other index is the post-pass value",
        "// recomputed from pre-launch state, then connected across the partner",
        "// axis's stored window on its plane 0 exactly as the certified sampler",
        "// connects the stored word (cwindow: cconnect of the load).",
        "__device__ __forceinline__ cf resolved_pair_window(",
        "    int comp, int index, int pa, int pph, cf pphase, const WeldArgs& weld",
        ") {",
        "    if (index < 0) return cf_zero();",
        "    return cconnect(resolve_D_at(comp, index, weld), pa, pph, pphase);",
        "}",
        "",
        "// down_sample()'s body with the load resolved and NOTHING ELSE MOVED: the",
        "// metallic zero, and the mirror weight ON THE GHOST LANE AND NOWHERE ELSE",
        "// through the certified part-wise scale. No connection: the shifted",
        "// operand is a plain wrap.",
        "__device__ __forceinline__ cf resolved_down_sample(",
        "    int comp, int index, int gl, int bc, float w, const WeldArgs& weld",
        ") {",
        "    if (index < 0) return cf_zero();",
        "    cf z = resolve_D_at(comp, index, weld);",
        "    if (gl && bc == BC_MIRROR) return mul_coefficient_left(w, z);",
        "    return z;",
        "}",
        "",
    ]))
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# The constitutive half -- ONE lift, both families
# ---------------------------------------------------------------------------
#
# THE TWO FAMILIES SPLICE THE SAME CONSTITUTIVE TEXT. Unlike the real-storage
# welds -- whose two cells take two DIFFERENT certified constitutive modules
# (``offdiag_constitutive_kernels`` and ``folded_offdiag_kernels``, with different
# helper names) -- both complex cells take ``complex_offdiag_update_e``, whose two
# arms differ only in the TAIL. So the prelude edits, the seam rewrites and the
# assertions are one piece of text here rather than two, and the arm reaches them
# as an argument.

#: Where the constitutive prelude's DEFINES end and its ``__device__`` helpers
#: begin: the comment block that opens ``coord_up``. The weld splices its own struct,
#: its cell lift and its resolution BETWEEN the two, because ``resolved_down_sample``
#: reads ``BC_MIRROR`` and ``offdiag_term`` calls ``resolved_down_sample`` -- so the
#: defines must precede the resolution and the resolution must precede the term. The
#: curl prelude (``complex_emitter``'s) already carries ``BC_PERIODIC`` and
#: ``BC_METALLIC``, and the constitutive prelude no longer repeats them, so nothing is
#: de-duplicated.
_CONSTITUTIVE_HELPERS_ANCHOR = "// stepping._shift_up AS _offdiagonal_terms CALLS IT"

#: ``offdiag_term``'s three anchors, in the certified spelling.
_TERM_PARAMETERS = (
    "    const float* g, const float* u, int home, int down, int up, int corner,\n"
    "    int pa, int dgl, int dbc, float w, int pph, cf pphase,\n"
    "    int oa, int oph, cf ophase\n)")
_TERM_PARAMETERS_RESOLVED = (
    "    int comp, cf g_home, const float* u, int home, int down, int up,\n"
    "    int corner,\n"
    "    int pa, int dgl, int dbc, float w, int pph, cf pphase,\n"
    "    int oa, int oph, cf ophase, const WeldArgs& weld\n)")
_TERM_NEAR = (
    "    cf near_pair = cf_add(pair_window(g, home, pa, pph, pphase),\n"
    "                          down_sample(g, down, dgl, dbc, w));\n")
_TERM_NEAR_RESOLVED = (
    "    // THE SEAM, near half. The home sample is the register this thread\n"
    "    // already resolved for the partner at its own cell -- the post-fill,\n"
    "    // post-clear value the array path would have loaded -- connected across\n"
    "    // the partner axis's window exactly as pair_window connects the load (home\n"
    "    // is never a ghost). The DOWN sample is a foreign cell, recomputed from\n"
    "    // pre-launch state and run through the SAME mirror rule on the SAME ghost\n"
    "    // lane. Same two operands, same order, same cf_add.\n"
    "    cf near_pair = cf_add(cconnect(g_home, pa, pph, pphase),\n"
    "                          resolved_down_sample(comp, down, dgl, dbc, w, weld));\n")
_TERM_FAR = (
    "    cf far_pair = cf_add(pair_window(g, up, pa, pph, pphase),\n"
    "                         down_sample(g, corner, dgl, dbc, w));\n")
_TERM_FAR_RESOLVED = (
    "    // THE SEAM, far half. Both samples are foreign; the UP leg keeps the\n"
    "    // own-axis zero ghost and the partner-axis window, the corner keeps the\n"
    "    // ghost lane and its rule. The coefficient multiply BETWEEN the two\n"
    "    // shifts, the own-axis connection on the near product, the ghosted\n"
    "    // coefficient at a zero-ghost up plane and the sum below are verbatim.\n"
    "    cf far_pair = cf_add(resolved_pair_window(comp, up, pa, pph, pphase, weld),\n"
    "                         resolved_down_sample(comp, corner, dgl, dbc, w, weld));\n")

#: The certified constitutive body's thread preamble and index decode -- the block
#: the WELD emits once, for both halves, and therefore drops from here.
_CONSTITUTIVE_PREAMBLE = ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                          "    if (idx >= nx * ny * nz) return;\n")
_CONSTITUTIVE_DECODE = ("    int nyz = ny * nz;\n"
                        "    int k = idx % nz;\n"
                        "    int j = (idx / nz) % ny;\n"
                        "    int i = idx / (ny * nz);\n")

#: The register each component's own-cell displacement lands in, in component
#: order. Spelled once so the diagonal rewrite and the partner rewrite cannot
#: disagree about which register carries which component.
OWN_CELL_REGISTERS: Tuple[str, str, str] = ("v_x", "v_y", "v_z")


def _first_code_match(pattern: str, body: str) -> Any:
    """The first match of ``pattern`` in ``body`` OUTSIDE a ``//`` comment.

    The certified emitter labels every component block and every term with the
    volume and the axis it transcribes (``// --- Ex: own axis x; source Dx``), and
    those comments are the audit trail this weld most wants to keep. A stray-token
    check that read them would either fire on every emission or force the comments
    out; this one asks the question it means -- does any COMPILED token still name
    a volume this launch does not bind.
    """
    for line in body.splitlines():
        code = line.split("//", 1)[0]
        found = re.search(pattern, code)
        if found is not None:
            return found.group(0)
    return None


def _certified_constitutive() -> None:
    """Refuse BY NAME where the certified constitutive text is unreachable."""
    if complex_offdiag_update_e is None or _coverage is None:
        raise RuntimeError(
            "complex_offdiag_update_e is not importable on this host, so there is "
            "no certified complex off-diagonal update_E text to splice. Both "
            "families' predicates need none of it and still answer")


def constitutive_prelude_defines() -> str:
    """``complex_offdiag_update_e``'s own defines (``BC_MIRROR``, ``MIRROR_ROW``).

    Emitted BEFORE the weld's struct and resolution, because both are read by
    :func:`resolution_source`'s ``resolved_down_sample`` and by ``coord_dn`` below it.
    """
    _certified_constitutive()
    own = complex_offdiag_update_e._OWN_PRELUDE
    if own.count(_CONSTITUTIVE_HELPERS_ANCHOR) != 1:
        raise AssertionError(
            f"the certified constitutive prelude carries "
            f"{own.count(_CONSTITUTIVE_HELPERS_ANCHOR)} copies of coord_up's "
            f"comment block, not one; the split between its defines and its helpers "
            f"has no anchor")
    return own.split(_CONSTITUTIVE_HELPERS_ANCHOR, 1)[0]


def constitutive_prelude_helpers() -> str:
    """The certified chunk helpers, then ``coord_up``, ``coord_dn``, both certified
    samplers and the RESOLVED ``offdiag_term``.

    The chunk helpers (``flat``, ``ghosted``, ``chunk_label``, ``chunk_row``,
    ``chunk_keeps``) are read out of ``offdiag_emitter`` by the certified family's own
    reader (``complex_offdiag_update_e._certified_helper``), in its order, so the weld
    carries the bytes the single-kernel family compiles rather than a copy.

    The two certified samplers are carried WHOLE and unused, on
    ``complex_offdiag_update_e``'s own precedent for carrying dead certified
    helpers: forking a prelude to drop them is the silent divergence indirection
    exists to prevent, and carrying them turns the liability into evidence -- a
    gate arms a mutation of each as a MUST-BE-UNCAUGHT null, and their silence is
    the measurement that this weld really does route every sample through the
    resolution instead.
    """
    _certified_constitutive()
    own = complex_offdiag_update_e._OWN_PRELUDE
    siblings = "\n\n".join(complex_offdiag_update_e._certified_helper(name)
                           for name in complex_offdiag_update_e.SIBLING_HELPERS)
    helpers = _CONSTITUTIVE_HELPERS_ANCHOR + own.split(
        _CONSTITUTIVE_HELPERS_ANCHOR, 1)[1]
    for anchor, what in ((_TERM_PARAMETERS, "offdiag_term's parameter list"),
                         (_TERM_NEAR, "offdiag_term's near-pair line"),
                         (_TERM_FAR, "offdiag_term's far-pair line")):
        if helpers.count(anchor) != 1:
            raise AssertionError(
                f"the certified constitutive prelude carries "
                f"{helpers.count(anchor)} copies of {what}, not one; this weld "
                f"LIFTS that text rather than retyping it and cannot splice around "
                f"its absence")
    helpers = helpers.replace(_TERM_PARAMETERS, _TERM_PARAMETERS_RESOLVED, 1)
    helpers = helpers.replace(_TERM_NEAR, _TERM_NEAR_RESOLVED, 1)
    helpers = helpers.replace(_TERM_FAR, _TERM_FAR_RESOLVED, 1)
    return "\n" + siblings + "\n\n" + helpers


def constitutive_body(arm: str, row_mask: Sequence[int],
                      boundary_prefix: str = "bc_") -> str:
    """The certified ``update_E`` body for one arm and row mask, seamed.

    FIVE EDITS, each through an exact anchor:

    1. the thread preamble and the index decode leave -- the weld emits them once
       for both halves, and splicing both would redeclare ``idx``, ``i``, ``j``,
       ``k`` and ``nyz``;
    2. the own-cell flux-density load becomes the register this thread already
       resolved (:data:`OWN_CELL_REGISTERS`);
    3. each live term's partner VOLUME becomes a component TAG plus that partner's
       own-cell register, and the argument pack is appended to the call;
    4. on the ``pml`` arm the constitutive's HALF-INTEGER ``kms_*`` is renamed
       ``kms_half_*`` -- the integer sub-lattice belongs to the curl and letting
       one shadow the other is half a cell in the absorber profile rather than a
       compile failure;
    5. where ``boundary_prefix`` is not ``"bc_"`` the constitutive's boundary reads
       are renamed onto its own triple. THE TWO READINGS GENUINELY DIFFER ON A
       FOLD -- the curl splits the two terminations and ``update_E`` serves both
       with ``BC_MIRROR`` -- so the folded family binds two triples and the
       unfolded one binds a single triple it cross-checks.

    Raises rather than returning a body that would compile and be quietly wrong:
    a surviving ``D*`` token would be an unbound identifier, and a surviving
    curl-triple read after a rename would be the wrong ghost rule on a plane.
    """
    _certified_constitutive()
    arm = normalized_arm(arm)
    mask = complex_offdiag_update_e.normalized_row_mask(row_mask)
    source = complex_offdiag_update_e.complex_offdiag_source(arm, mask)
    _prelude, body = _split_emitted(
        source, f"complex_offdiag_source({arm!r}, {tuple(mask)!r})")

    # (1) THE SHARED PREAMBLE, dropped.
    for anchor, what in ((_CONSTITUTIVE_PREAMBLE, "the thread-index preamble"),
                         (_CONSTITUTIVE_DECODE, "the index decode")):
        if body.count(anchor) != 1:
            raise AssertionError(
                f"the certified constitutive body carries {body.count(anchor)} "
                f"copies of {what}, not one; the weld emits that block once for "
                f"both halves and cannot drop it from here")
        body = body.replace(anchor, "", 1)

    # (5) THE CONSTITUTIVE'S OWN BOUNDARY TRIPLE, where the family binds two.
    if boundary_prefix != "bc_":
        for axis in "xyz":
            body = re.sub(rf"(?<![A-Za-z0-9_])bc_{axis}(?![A-Za-z0-9_])",
                          f"{boundary_prefix}{axis}", body)
    if boundary_prefix != "bc_":
        stray = _first_code_match(r"(?<![A-Za-z0-9_])bc_[xyz](?![A-Za-z0-9_])", body)
        if stray is not None:
            raise AssertionError(
                f"{stray!r} survived the rename onto the constitutive's "
                f"own boundary triple; on a fold the two readings of the same grid "
                f"differ by design and a body left reading the curl's would apply "
                f"the wrong ghost rule to a whole plane")

    # (2) THE SEAM, diagonal half.
    for component, term in enumerate(complex_offdiag_update_e.E_TERMS):
        name, volume, _own_axis = term
        line = the_line_starting(
            body, f"    cf gs_{name} = cf_load({volume}, idx);",
            f"{name}'s own-cell flux-density load")
        body = body.replace(
            line,
            f"    cf gs_{name} = {OWN_CELL_REGISTERS[component]};   "
            f"// the register this thread resolved for its own cell", 1)

    # (3) THE SEAM, foreign half: the partner argument and the pack, per live term.
    for slot, flag in enumerate(mask):
        if not flag:
            continue
        component, offset = divmod(slot, 2)
        partner = _coverage.OFFDIAG_TRANSVERSE_PARTNERS[component][offset]
        volume = complex_offdiag_update_e.PARTNER_VOLUMES[partner]
        coefficient = complex_offdiag_update_e.ROW_PARAMETERS[slot]
        tag = f"{complex_offdiag_update_e.E_TERMS[component][0]}_{offset}"
        line = the_line_starting(
            body, f"        {volume}, {coefficient}, idx,",
            f"the partner argument of the {coefficient} term")
        body = body.replace(
            line,
            f"        {partner}, {OWN_CELL_REGISTERS[partner]}, {coefficient}, idx,",
            1)
        body = close_call_with_the_pack(
            body, f"    cf term_{tag} = offdiag_term(", f"the {coefficient} term")

    # (4) THE SUB-LATTICE RENAME, matched as a WHOLE LINE.
    if arm == "pml":
        for component, term in enumerate(complex_offdiag_update_e.E_TERMS):
            name, _volume, own_axis = term
            axis = "xyz"[own_axis]
            index = "ijk"[own_axis]
            line = the_line_starting(
                body, f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},",
                f"{name}'s certified constitutive_apply call")
            wanted = (f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},"
                      f" kps_{axis}[{index}], kms_{axis}[{index}], in_h_{axis});")
            if line != wanted:
                raise AssertionError(
                    f"the certified constitutive body spells {name}'s tail "
                    f"{line!r}; this weld renames only the half-integer kms and "
                    f"cannot do so on a line whose argument list has moved")
            body = body.replace(
                line,
                f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},"
                f" kps_{axis}[{index}], kms_half_{axis}[{index}], in_h_{axis});", 1)

    stray = _first_code_match(r"(?<![A-Za-z0-9_])D[xyz](?![A-Za-z0-9_])", body)
    if stray is not None:
        raise AssertionError(
            f"{stray!r} survived into the seamed constitutive body; this "
            f"weld binds the flux density ONCE, as the curl's own f0/f1/f2, and "
            f"every read of it here is either the thread's own register or a "
            f"recompute through resolve_D -- a surviving volume name would be an "
            f"unbound identifier at compile time or, worse, a second binding of a "
            f"volume the launch is not writing")
    return body
