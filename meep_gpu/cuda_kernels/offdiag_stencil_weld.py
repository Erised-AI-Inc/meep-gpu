"""The SCRATCH-OUTPUT lift both off-diagonal D/E welds are built from.

WHY THERE IS A SHARED MODULE AND NOT TWO COPIES. :mod:`.offdiag_fused_electric_pair`
and :mod:`.folded_offdiag_fused_electric_pair` occupy two different board cells with
two different constitutive arms whose predicates are disjoint on the fold -- but the
CURL half, the scratch discipline, the per-cell resolution and the rotation are one
piece of arithmetic, and the one thing this track will not do is let two copies of a
resolution drift apart. Everything that decides a float lives here; each family module
owns its predicate, its signature, its cell and its record.

NOTHING HERE IMPORTS CuPy. The lift is string work over the certified device text and
the resolution is emitted from ``fields.IYEE_SHIFTS``, so both are exercisable at the
merge bar on a laptop -- which is the rule this package applies to every part whose
failure mode is a silent wrong answer rather than a crash.

=============================================================================
THE DESIGN, AND WHY THE RECORDED REFUSAL DOES NOT REACH IT
=============================================================================

The board records both cells STENCIL-BLOCKED with

    "the constitutive half reads the curl half's IN-PLACE output at a cell this
     thread does not own. In one launch that cell is written by another block, and
     CUDA offers no grid-wide barrier inside an ordinary launch"

-- true, and true OF THE IN-PLACE WELD. It has two premises, and this shape removes
both:

* **the curl half writes nothing in place.** ``D_new`` and ``fu_new`` go to a
  LAUNCH-LOCAL SCRATCH allocation, so ``D_old``, ``fu_old`` and ``H`` are ``const``
  for the whole launch and no thread can observe another thread's store;
* **the foreign read is not a read of another thread's output.** It is a RECOMPUTE
  from that same unwritten state, through :func:`raw_step_D_cell` -- the certified
  ``step_D_pml_real`` body, whole, evaluated at an arbitrary cell. A pure function of
  unwritten memory has no schedule to depend on.

The launcher then ROTATES the ``D``/``fu`` bindings, which is the choreography the
certified E->P path already ships (``ade_kernels.update_P_fused_pml_real`` rotates
``P``/``P_prev``/``_scratch`` per component, :414-424).

WHAT MAKES THE RECOMPUTE EXACT RATHER THAN CLOSE. ``step_D_pml_real`` at a cell reads
``H`` at that cell and its three backward neighbours, and ``D``/``fu`` AT THAT CELL
ONLY -- ``pml_apply`` touches ``f[idx]`` and ``fu[idx]`` and nothing else. So the
stepped value at any cell is a function of state this launch does not write, and
evaluating it twice gives the same bits by construction rather than by a tolerance.

WHY THE RECOMPUTE IS LOAD-BEARING, MEASURED. ``parity/meep_gpu/results/
cuda_offdiag_scratch_weld_2026-09-02/probe.json`` counts, per fixture, how many of the
constitutive's foreign taps land on a cell whose stepped value DIFFERS from its
pre-launch value: **227,892 of 227,892 across eleven fixtures.** Every single foreign
tap is a cell the in-place weld would read at an undefined time. That number is what
makes this a design rather than a formality.

=============================================================================
THE RESOLUTION -- WHAT ONE LAUNCH PERFORMS BETWEEN THE TWO HALVES
=============================================================================

The array path runs three passes between ``step_D`` and ``update_E``
(driver.py:3309, :3310, :3311), and the constitutive reads their OUTPUT. The weld
therefore does not read a stored D at all: it reads

    resolve_D(component, cell) =
        the raw stepped value at ONE redirected cell, times a product of per-axis
        parities applied ONE MULTIPLY AT A TIME, or an exact +0.0f where the wall
        clear owns the plane

with the redirect and the parities exactly as ``stepping._write_mirror_ghost``
(:1450-1451), ``._zero_metal`` (:2206-2247) and ``._fill_folded_far_ghosts``
(:1489-1534) write them, in the driver's order: near fill, then wall clear, then far
fill. A near ghost landing in a cleared plane is therefore CLEARED, and the far image
reads the post-clear, post-near value at its source row.

THAT COMPOSITION IS MEASURED, not argued. The probe above puts it against the driver's
own pass order over TWO COMPLETE STEPS on eleven fixtures -- unfolded one-row, all-rows
and walled; folded even, odd, odd-count, metallic-terminated, odd-walled, two-axis
mixed-phase and 3-D -- and reports zero differing words on every stored volume at step
1 and at step 2, with every reachable null control diverging and every unreachable one
recorded ``predicted_null`` WITH its reason. The sibling Metal probe
(``results/metal_scratch_weld_closed_form_2026-09-01``) measured the same resolution
independently; this one measures it on the CUDA arms and adds the per-cell purity
ledger and the race census.

PER-CELL PURITY IS THE PROPERTY THAT LICENSES THE RECOMPUTE, and the probe asserts it
rather than assuming it: every resolved cell consults EXACTLY ONE raw cell. A
resolution that needed a plane would not fit in a ``__device__`` function a thread can
call for a neighbour.

=============================================================================
THE PARITIES ARE HANDED IN, NOT NEGATED IN THE KERNEL
=============================================================================

``fields.mirror_parity(c, axis, phase)`` is ``phase * (1 - 2 * iyee[c][axis])``: a D
component's NEAR axes are its two shift-0 axes, so their weight is ``+phase``; its FAR
axis is its one shift-1 axis, so that weight is ``-phase``. Both are computed on the
HOST by ``in_seam_coverage.component_parity`` -- the same function the certified
in-seam passes use -- and bound as two float32 triples.

The kernel therefore performs NO negation. That is deliberate and is this platform's
own rule, stated in ``offdiag_emitter``: CUDA lowers ``-x`` to ``neg.f32`` rather than
to ``0.0f - x``, so it does not canonicalize signed zeros the way the sibling track's
platform does, and a weld that computed ``-phase`` in device code would be making a
signed-zero claim it does not need to make. Multiplying by a bound ``+/-1.0f`` is the
array path's own spelling (``phase * _mirror_source(...)``, a Python int times a
float32 array) and is what the probe measured.

=============================================================================
WHAT THIS MODULE DOES NOT DO
=============================================================================

It plans nothing and launches nothing on its own; it emits device text and resolves
tables. ``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch --
the two families are opt-in through ``arms.plan_step(..., fuse=True)`` like every other
product on this board.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .in_seam_coverage import (IYEE_SHIFTS, MIRROR_SOURCE_INDEX,
                                   component_parity, folded_far_rows,
                                   mirror_fill_phases, zero_metal_axes)
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

    _in_seam_coverage = _load("in_seam_coverage")
    IYEE_SHIFTS = _in_seam_coverage.IYEE_SHIFTS
    MIRROR_SOURCE_INDEX = _in_seam_coverage.MIRROR_SOURCE_INDEX
    component_parity = _in_seam_coverage.component_parity
    folded_far_rows = _in_seam_coverage.folded_far_rows
    mirror_fill_phases = _in_seam_coverage.mirror_fill_phases
    zero_metal_axes = _in_seam_coverage.zero_metal_axes

# The certified curl text. Taken DEFENSIVELY: the module imports CuPy at scope, and
# the predicate half of every family built on this must answer on a host with no
# device. The emitter refuses BY NAME where it is absent.
try:
    from . import step_curl_kernels
except Exception:  # noqa: BLE001 - no CuPy, or no device toolchain
    step_curl_kernels = None  # type: ignore[assignment]

__all__ = [
    "CURL_LIFT_EDITS", "D_TARGETS", "NEAR_SOURCE_ROW", "RESOLUTION_LIFT_EDITS",
    "WELD_ARGS_FIELDS", "clear_axes", "close_call_with_the_pack", "curl_prelude",
    "far_axis", "fill_plan",
    "near_axes", "raw_step_D_cell_source", "resolution_source", "split_body",
    "the_line_starting", "weld_args_construction", "weld_args_struct",
]

#: The three D volumes, in signature order; the curl body's own spelling.
D_TARGETS: Tuple[str, str, str] = ("Dx", "Dy", "Dz")

#: ``stepping.MIRROR_SOURCE_INDEX`` (stepping.py:160) -- MEEP's ``io = -2`` halved
#: origin, so the near ghost at stored cell 0 images stored cell 2. Spelled once, as
#: a ``#define`` in the emitted prelude, so the redirect and the predicate's
#: minimum-extent clause cannot disagree about the row.
NEAR_SOURCE_ROW = MIRROR_SOURCE_INDEX

_AXIS = ("x", "y", "z")
_COORD = ("i", "j", "k")
_EXTENT = ("nx", "ny", "nz")

#: Separates a certified kernel's signature from its body. Every certified string on
#: this track closes its parameter list on its own line.
_BODY_ANCHOR = "\n) {\n"

#: The certified curl body's thread preamble and its index decode -- the two blocks a
#: ``__device__`` function evaluated at an ARBITRARY cell cannot keep.
_THREAD_PREAMBLE = ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                    "    if (idx >= nx * ny * nz) return;\n")
_INDEX_DECODE = ("    int k = idx % nz;\n"
                 "    int j = (idx / nz) % ny;\n"
                 "    int i = idx / (ny * nz);\n")

#: The certified ``pml_apply``'s three anchors, and what the PURE form makes of each.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_PURE = "__device__ __forceinline__ float pml_apply_pure(\n"
_PML_APPLY_PARAMETERS = (
    "    float* __restrict__ f, float* __restrict__ fu, int idx, float curl,\n"
    "    float kms, float sinv, float kms_u, float sinv_u\n)")
_PML_APPLY_PARAMETERS_PURE = (
    "    const float* __restrict__ f, const float* __restrict__ fu, int idx,\n"
    "    float curl,\n"
    "    float kms, float sinv, float kms_u, float sinv_u, float* fu_out\n)")
_PML_APPLY_FU_STORE = "    fu[idx] = fu_new;\n"
_PML_APPLY_FU_STORE_PURE = (
    "    // THE FIRST OF THE TWO STORES, HANDED BACK INSTEAD OF PERFORMED. The\n"
    "    // caller writes it to the SCRATCH volume, so nothing this launch reads\n"
    "    // is ever a word this launch wrote.\n"
    "    *fu_out = fu_new;\n")
_PML_APPLY_F_STORE = "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
_PML_APPLY_F_STORE_PURE = (
    "    // THE SECOND STORE, LIKEWISE. Same right-hand side, same tree, same\n"
    "    // parenthesisation; the value is RETURNED so the resolution can redirect\n"
    "    // it, the wall clear can replace it and the constitutive can consume it\n"
    "    // from a register.\n"
    "    return (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n")

#: Every line of certified CURL text the weld does not lift verbatim, with the reason.
#: DATA, not prose, so a gate and a host suite can assert the list.
CURL_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ float pml_apply_pure(",
     "why": "the weld writes a SCRATCH volume, so the helper must not store. It "
            "returns the displacement and hands the split-field value back through "
            "an out-parameter; f and fu become const because nothing in this launch "
            "writes them."},
    {"line": "    fu[idx] = fu_new;",
     "became": "    *fu_out = fu_new;",
     "why": "the store moves to the caller, which addresses the scratch. The value "
            "and the expression that produced it are untouched."},
    {"line": "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;",
     "became": "    return (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;",
     "why": "same right-hand side, same tree, same operand order; the result is "
            "returned rather than stored, because the resolution may redirect it, "
            "the wall clear may replace it with an exact +0.0f, and the "
            "constitutive half consumes the thread's own cell from a register."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= nx * ny * nz) return;",
     "became": "    const int idx = (i * (ny * nz)) + (j * nz) + k;",
     "why": "THE WHOLE POINT OF THE LIFT. The certified body becomes a __device__ "
            "function evaluated at an ARBITRARY cell rather than at the thread's "
            "own, so the cell arrives as three coordinates and the flat index is "
            "COMPOSED from them by the same layout the decode inverts. The bounds "
            "guard leaves with the thread index: every caller is either the "
            "kernel's own guarded thread or a redirect target the resolution "
            "proved in range (stored row 2 on an axis the predicate requires more "
            "than 2 cells of, or a reflect row in [0, n)), and every ghosted tap "
            "is filtered by `index < 0` before it reaches here."},
    {"line": "    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
             "    int i = idx / (ny * nz);",
     "became": "(removed -- i, j and k are parameters)",
     "why": "the decode's inverse is now the caller's; keeping it would redeclare "
            "the three parameters and the kernel would not compile."},
    {"line": "        pml_apply(D*, fu_D*, idx, curl, ...);",
     "became": "        d_out[n] = pml_apply_pure(D*, fu_D*, idx, curl, ..., "
               "&fu_out[n]);",
     "why": "capture both values. The argument list, its order and the coefficient "
            "pairing are untouched; what is added is where the two results go. "
            "Three lines, one per component, each derived from the certified line "
            "by prefix and suffix rather than retyped."},
)

#: Every line of certified CONSTITUTIVE text a family module rewrites. Shared here
#: because both families make the same four edits for the same four reasons; each
#: module asserts the ones it actually applied.
RESOLUTION_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    float gs_E* = D*[idx];",
     "became": "    float gs_E* = v_*;",
     "why": "THE SEAM, own-cell half. The certified constitutive body opens each "
            "component by loading the flux density; the weld has already resolved "
            "the thread's own cell into a register, and that register is the "
            "post-fill, post-clear value the array path would have loaded."},
    {"line": "        D*, chi1inv_E*_E*, idx,",
     "became": "        <component tag>, v_*, chi1inv_E*_E*, idx,",
     "why": "THE SEAM, foreign half. The partner volume becomes a COMPONENT TAG "
            "plus the partner's own-cell register: the term's home sample is the "
            "register and its three foreign samples are recomputed through "
            "resolve_D. The coefficient argument, the home index and the three "
            "shifted indices are untouched."},
    {"line": "    constitutive_apply(E*, f_w_E*, idx, src_E*, kps_*[*], kms_*[*]);",
     "became": "    constitutive_apply(E*, f_w_E*, idx, src_E*, kps_*[*], "
               "kms_half_*[*]);",
     "why": "ONE RENAME, NO ARITHMETIC. kms_* is the INTEGER sub-lattice for the D "
            "curl and the HALF-INTEGER one for update_E; letting one shadow the "
            "other in a single scope is a half-cell error in the absorber profile "
            "rather than a compile failure. The same rename fused_electric_pair "
            "makes, spelled the same way."},
    {"line": "#define BC_PERIODIC 0\n#define BC_METALLIC 1",
     "became": "(removed from the constitutive prelude)",
     "why": "the curl prelude defines both, identically. A second IDENTICAL "
            "definition is legal and a second DIFFERENT one would be diagnosed "
            "only as a warning, so the weld carries exactly one of each and the "
            "removal is anchored on the certified spelling."},
)

#: The pack every ``__device__`` helper takes, in emission order. A struct rather
#: than thirty parameters threaded through four helpers: the members are built once
#: from the kernel's own arguments, are uniform across the block, and name exactly
#: the state the resolution reads.
WELD_ARGS_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("const float* __restrict__", "Dx"), ("const float* __restrict__", "Dy"),
    ("const float* __restrict__", "Dz"),
    ("const float* __restrict__", "fu_Dx"), ("const float* __restrict__", "fu_Dy"),
    ("const float* __restrict__", "fu_Dz"),
    ("const float* __restrict__", "Hx"), ("const float* __restrict__", "Hy"),
    ("const float* __restrict__", "Hz"),
    ("const float* __restrict__", "kms_x"), ("const float* __restrict__", "sinv_x"),
    ("const float* __restrict__", "kms_y"), ("const float* __restrict__", "sinv_y"),
    ("const float* __restrict__", "kms_z"), ("const float* __restrict__", "sinv_z"),
    ("int", "nx"), ("int", "ny"), ("int", "nz"), ("float", "dtdx"),
    ("int", "bc_x"), ("int", "bc_y"), ("int", "bc_z"),
    ("int", "wall_x"), ("int", "wall_y"), ("int", "wall_z"),
    ("int", "near_x"), ("int", "near_y"), ("int", "near_z"),
    ("int", "reflect_x"), ("int", "reflect_y"), ("int", "reflect_z"),
    ("float", "near_phase_x"), ("float", "near_phase_y"), ("float", "near_phase_z"),
    ("float", "far_phase_x"), ("float", "far_phase_y"), ("float", "far_phase_z"),
)


# ---------------------------------------------------------------------------
# The Yee tables, read rather than assumed
# ---------------------------------------------------------------------------

def near_axes(component: int) -> Tuple[int, ...]:
    """The axes ``fill_symmetry_bc_D`` can image for one D component.

    ``stepping._fill_symmetry_ghost_cells`` (:1440-1447) writes axis ``a`` for a
    component exactly when the plane's phase is declared and ``iyee[a] == 0``. The
    YEE TEST is what this returns; whether each axis is folded is the runtime
    ``near_*`` flag.
    """
    shifts = IYEE_SHIFTS[D_TARGETS[component]]
    return tuple(axis for axis in range(3) if shifts[axis] == 0)


def clear_axes(component: int) -> Tuple[int, ...]:
    """The axes ``zero_metal_D`` can clear one D component's stored cell 0 on.

    ``stepping._zero_metal`` (:2240-2247) clears axis ``a`` for a component exactly
    when that axis is walled and ``iyee[a] == 0`` -- the SAME Yee test the near fill
    uses, which is why a D component's near axes and its cleared axes are the same
    pair and why a near ghost can land in a cleared plane at all.
    """
    return near_axes(component)


def far_axis(component: int) -> int:
    """The one axis ``fill_folded_far_ghosts_D`` can image a D component's top plane on.

    ``stepping._fill_folded_far_ghosts`` (:1533-1534) skips a component whose
    ``iyee[axis] != 1``. Every D component has exactly one shift-1 axis; a table that
    drifted to give it none or two would make the far fill either dead or ambiguous,
    so this RAISES rather than returning a default.
    """
    shifts = IYEE_SHIFTS[D_TARGETS[component]]
    ones = [axis for axis in range(3) if shifts[axis] == 1]
    if len(ones) != 1:
        raise AssertionError(
            f"{D_TARGETS[component]} has Yee shifts {shifts}, which give it "
            f"{len(ones)} shift-1 axes; fields.IYEE_SHIFTS gives every D component "
            f"exactly one, and the far fill images that axis and no other")
    return ones[0]


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def _certified_curl() -> None:
    if step_curl_kernels is None:
        raise RuntimeError(
            "step_curl_kernels is not importable on this host (it imports CuPy at "
            "module scope), so there is no certified curl text to splice. The "
            "families' predicates need none of it and still answer")


def split_body(source: str, prelude: str, name: str) -> str:
    """One certified kernel string, reduced to its body.

    Strips the prelude it was built from, the signature that follows it and the
    closing brace. Raises rather than returning a truncated body: a splice that
    silently dropped a component would emit a kernel that steps two fields and
    reports success.
    """
    if not source.startswith(prelude):
        raise AssertionError(
            f"{name} no longer begins with the prelude this module lifts "
            f"separately; the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, not "
            f"1; the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def the_line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    A missing anchor is certified text that changed under this family, and splicing
    around it would emit a kernel that compiles and is quietly not the certified
    arithmetic; TWO matches would mean the anchor no longer identifies one statement.
    """
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this weld LIFTS that line rather than retyping "
            f"it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines; the lift of {what} would "
            f"take an arbitrary one")
    return matches[0]


def close_call_with_the_pack(body: str, opening: str, what: str) -> str:
    """Append ``, weld`` to the call whose first line is ``opening``.

    THE OTHER HALF OF THE TERM REWRITE. The resolved ``offdiag_term`` needs the
    argument pack, and the certified call spans four lines with its last argument on
    the closing one. Found STRUCTURALLY -- the opening line, then the first line
    below it that closes the call -- rather than by matching the corner index
    expression, because two row masks emit different corner expressions and a match
    on one of them would be an anchor that silently stops applying.

    Raises where the call does not close within the lines the certified emitter
    writes: a term whose shape moved is a term this weld must not splice. Serves both
    families -- the unfolded term closes on its corner index and the folded one on its
    ghost lane and weight, so the terminator is the statement's own ``);`` rather than
    either family's last argument.
    """
    lines = body.splitlines(keepends=True)
    starts = [n for n, line in enumerate(lines) if line.startswith(opening)]
    if len(starts) != 1:
        raise AssertionError(
            f"{opening!r} opens {len(starts)} calls in the certified body, not 1; "
            f"the pack argument for {what} would be appended to an arbitrary one")
    start = starts[0]
    for offset in range(start, min(start + 9, len(lines))):
        closing = lines[offset].rstrip("\n")
        if closing.endswith(");"):
            lines[offset] = closing[:-len(");")] + ", weld);\n"
            return "".join(lines)
    raise AssertionError(
        f"the certified call opened by {opening!r} does not close within eight "
        f"lines; this weld appends the argument pack to its closing line and "
        f"cannot find it")


def curl_prelude() -> str:
    """``step_curl_kernels._REAL_PML_PRELUDE`` with ``pml_apply`` made PURE.

    ``shift_up`` and ``shift_dn`` are lifted untouched -- they read ``H``, which no
    thread writes -- and so are the three boundary ``#define``s.
    """
    _certified_curl()
    prelude = step_curl_kernels._REAL_PML_PRELUDE
    for anchor, what in ((_PML_APPLY_SIGNATURE, "pml_apply's declaration"),
                         (_PML_APPLY_PARAMETERS, "pml_apply's parameter list"),
                         (_PML_APPLY_FU_STORE, "pml_apply's split-field store"),
                         (_PML_APPLY_F_STORE, "pml_apply's displacement store")):
        if prelude.count(anchor) != 1:
            raise AssertionError(
                f"the certified curl prelude carries {prelude.count(anchor)} copies "
                f"of {what}, not one; the pure rewrite has no anchor")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_PURE, 1)
    prelude = prelude.replace(_PML_APPLY_PARAMETERS, _PML_APPLY_PARAMETERS_PURE, 1)
    prelude = prelude.replace(_PML_APPLY_FU_STORE, _PML_APPLY_FU_STORE_PURE, 1)
    prelude = prelude.replace(_PML_APPLY_F_STORE, _PML_APPLY_F_STORE_PURE, 1)
    return prelude


def weld_args_struct() -> str:
    """The argument pack, declared."""
    lines = [
        "",
        "// THE PRE-LAUNCH STATE AND THE RUNTIME PLAN, packed once. Every member is",
        "// read-only for the whole launch: that is the property the recompute rests",
        "// on, and binding D/fu/H const here is where it is stated in the type",
        "// system rather than in a comment.",
        "struct WeldArgs {",
    ]
    for kind, name in WELD_ARGS_FIELDS:
        lines.append(f"    {kind} {name};")
    lines.append("};")
    return "\n".join(lines) + "\n"


def weld_args_construction(indent: str = "    ") -> str:
    """The struct, built from the kernel's own parameters. One line per member."""
    lines = [f"{indent}WeldArgs weld;"]
    for _kind, name in WELD_ARGS_FIELDS:
        lines.append(f"{indent}weld.{name} = {name};")
    return "\n".join(lines) + "\n"


def raw_step_D_cell_source() -> str:
    """The certified ``step_D_pml_real`` body as a ``__device__`` function of a CELL.

    THE ENTIRE BODY IS LIFTED. Only the two blocks that tie it to a thread index and
    the three ``pml_apply`` call lines are edited, each through an exact anchor, so a
    change in the certified text is a named failure rather than a silently different
    kernel. The three component blocks -- their curls, their masks, their coefficient
    pairings -- arrive character for character.

    Both results come back through out-parameters: ``d_out[c]`` is the RAW stepped
    displacement (before any in-seam pass) and ``fu_out[c]`` the split-field value,
    which no in-seam pass ever touches (``stepping._write_mirror_ghost`` writes
    ``field`` and never ``fu_field``, :1451; ``_zero_metal`` writes only the named
    components, :2247).
    """
    _certified_curl()
    body = split_body(step_curl_kernels._step_D_pml_real_kernel_code,
                      step_curl_kernels._REAL_PML_PRELUDE,
                      "_step_D_pml_real_kernel_code")
    for anchor, what in ((_THREAD_PREAMBLE, "the thread-index preamble"),
                         (_INDEX_DECODE, "the index decode")):
        if body.count(anchor) != 1:
            raise AssertionError(
                f"the certified curl body carries {body.count(anchor)} copies of "
                f"{what}, not one; the cell lift has no anchor")
    body = body.replace(
        _THREAD_PREAMBLE,
        "    // THE CELL ARRIVES AS COORDINATES. The flat index is composed by the\n"
        "    // same layout the certified decode inverts; the bounds guard left with\n"
        "    // the thread index, and every caller is either the kernel's own\n"
        "    // guarded thread or a redirect target the resolution proved in range.\n"
        "    const int idx = (i * (ny * nz)) + (j * nz) + k;\n", 1)
    body = body.replace(_INDEX_DECODE, "", 1)

    for component, target in enumerate(D_TARGETS):
        prefix = f"        pml_apply({target}, fu_{target}, idx, curl, "
        line = the_line_starting(body, prefix, f"{target}'s certified pml_apply call")
        if not line.endswith(");"):
            raise AssertionError(
                f"{target}'s certified pml_apply call does not close on its own "
                f"line ({line!r}); the capture has no suffix to take")
        captured = (f"        d_out[{component}] = pml_apply_pure("
                    + line[len("        pml_apply("):-len(");")]
                    + f", &fu_out[{component}]);")
        body = body.replace(line, captured, 1)

    signature = (
        "\n// The certified step_D_pml_real, evaluated at ONE ARBITRARY CELL and\n"
        "// storing nothing. This is the function the constitutive half calls for\n"
        "// every foreign sample it needs: a pure function of D_old, fu_old and H,\n"
        "// none of which this launch writes, so a foreign evaluation cannot depend\n"
        "// on which block ran first.\n"
        "__device__ __forceinline__ void raw_step_D_cell(\n"
        "    int i, int j, int k, const WeldArgs& weld, float* d_out, float* fu_out\n"
        ") {\n")
    unpack = ["    // The certified body's own names, bound to the pack.",
              "    const int nx = weld.nx; const int ny = weld.ny; const int nz = weld.nz;",
              "    const float dtdx = weld.dtdx;",
              "    const int bc_x = weld.bc_x; const int bc_y = weld.bc_y;",
              "    const int bc_z = weld.bc_z;"]
    for _kind, name in WELD_ARGS_FIELDS:
        if name.startswith(("D", "fu_D", "H", "kms_", "sinv_")):
            unpack.append(f"    const float* __restrict__ {name} = weld.{name};")
    return signature + "\n".join(unpack) + "\n" + body + "}\n"


def resolution_source() -> str:
    """``resolve_D`` -- the post-pass value of one D component at any cell.

    Three specialized helpers plus two dispatchers. The AXIS SETS are compile time
    (they are the component's Yee shifts); WHICH of them is live is runtime, which is
    this track's standing rule -- one device string per family, so one certification
    digest covers every fold orientation and no per-axis variant can end up pinned
    that nobody ran.
    """
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
        shifts = IYEE_SHIFTS[target]
        near = near_axes(component)
        lines = [
            f"// {target} {shifts}: the three in-seam passes, resolved per cell.",
            f"//   near fill  -- axes {tuple(_AXIS[a] for a in near)} (Yee shift 0),",
            f"//                 stored cell 0 images stored cell NEAR_SOURCE_ROW",
            f"//                 with weight +phase (stepping.py:1440-1451);",
            f"//   wall clear -- the SAME two axes (the same Yee test,",
            f"//                 stepping.py:2240-2247), stored cell 0 -> +0.0f, and",
            f"//                 it runs AFTER the near fill (driver.py:3309, :3310)",
            f"//                 so a near ghost in a cleared plane is cleared;",
            f"//   far fill   -- axis {_AXIS[own]} (Yee shift 1), the top stored plane",
            f"//                 images the reflect row with weight -phase, AFTER the",
            f"//                 clear (driver.py:3311).",
            f"__device__ __forceinline__ float resolve_{target}("
            f"int i, int j, int k, const WeldArgs& weld) {{",
            f"    const int far = (weld.reflect_{_AXIS[own]} >= 0)"
            f" && ({_COORD[own]} == weld.{_EXTENT[own]} - 1);",
            f"    if (far) {_COORD[own]} = weld.reflect_{_AXIS[own]};",
        ]
        cleared = " || ".join(f"(weld.wall_{_AXIS[a]} && {_COORD[a]} == 0)"
                              for a in near)
        lines += [
            "    float value;",
            f"    if ({cleared}) {{",
            "        // The array path writes an exact +0.0f into this plane.",
            "        value = 0.0f;",
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
            "        float raw[3];",
            "        float fu[3];",
            f"        raw_step_D_cell({source[0]}, {source[1]}, {source[2]},"
            f" weld, raw, fu);",
            f"        value = raw[{component}];",
        ]
        if near:
            lines.append(
                "        // ONE MULTIPLY PER IMAGED PLANE, NESTED and in X, Y, Z"
                " order, because")
            lines.append(
                "        // the fill writes plane after plane: a corner unowned on"
                " two planes")
            lines.append(
                "        // carries the product of both parities as two separate"
                " multiplies.")
        for axis in near:                     # X, Y, Z order: the fill's own
            lines.append(
                f"        if (image_{_AXIS[axis]})"
                f" value = weld.near_phase_{_AXIS[axis]} * value;")
        lines += [
            "    }",
            f"    if (far) value = weld.far_phase_{_AXIS[own]} * value;",
            "    return value;",
            "}",
            "",
        ]
        blocks.append("\n".join(lines))

    blocks.append("\n".join([
        "// The component tag is a compile-time constant at every call site, so this",
        "// chain is resolved by the compiler and no branch survives it.",
        "__device__ __forceinline__ float resolve_D(int comp, int i, int j, int k,",
        "                                          const WeldArgs& weld) {",
        "    if (comp == 0) return resolve_Dx(i, j, k, weld);",
        "    if (comp == 1) return resolve_Dy(i, j, k, weld);",
        "    return resolve_Dz(i, j, k, weld);",
        "}",
        "",
        "// The same, addressed by the FLAT index the certified constitutive body",
        "// already computes for its taps -- decoded by the layout flat() composes.",
        "__device__ __forceinline__ float resolve_D_at(int comp, int index,",
        "                                             const WeldArgs& weld) {",
        "    const int k = index % weld.nz;",
        "    const int j = (index / weld.nz) % weld.ny;",
        "    const int i = index / (weld.ny * weld.nz);",
        "    return resolve_D(comp, i, j, k, weld);",
        "}",
        "",
        "// ghosted()'s body with the load resolved: index -1 is the metallic zero",
        "// ghost the array path writes into that plane as an exact +0.0",
        "// (_shift_up:1782-1784, _shift_down:1826-1828), and every other index is",
        "// the post-pass value recomputed from pre-launch state.",
        "__device__ __forceinline__ float resolved_ghosted(int comp, int index,",
        "                                                 const WeldArgs& weld) {",
        "    return (index < 0) ? 0.0f : resolve_D_at(comp, index, weld);",
        "}",
        "",
    ]))
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# The runtime plan
# ---------------------------------------------------------------------------

def fill_plan(grid: Any) -> Dict[str, Tuple[Any, ...]]:
    """The three in-seam passes' runtime plan, as the flat per-axis triples.

    THE THREE READINGS ANSWER DIFFERENT QUESTIONS and confusing them is a plane of
    wrong values rather than a crash:

    * ``near[a]`` is 1 on a MIRROR-folded axis of EITHER termination -- the near
      fill's own condition is a declared mirror phase (stepping.py:1484-1485);
    * ``wall[a]`` is ``_zero_metal``'s declaration, with a folded axis EXCLUDED
      (stepping.py:2284-2286);
    * ``reflect[a]`` is ``fill_folded_far_ghosts_D``'s image row on a folded PERIODIC
      axis and ``-1`` where that pass does not run -- a SENTINEL, so a kernel that
      read it anyway would index outside the volume and be caught rather than
      silently imaging row 0.

    THE TWO PARITIES ARE COMPUTED BY THE ENGINE'S OWN HELPER, per component and per
    axis, and the per-axis collapse is ASSERTED rather than assumed: a D component's
    near axes are exactly its shift-0 axes (weight ``+phase``) and its far axis its
    one shift-1 axis (weight ``-phase``), so the weight depends on the axis alone --
    but only while ``fields.IYEE_SHIFTS`` says so, and a drifted table would put a
    wrong SIGN on a whole plane rather than raising.

    RAISES rather than returning a plan it cannot stand behind: a launcher handed a
    grid the predicate would have refused must not quietly build one.
    """
    phases = mirror_fill_phases(grid)
    rows = folded_far_rows(grid)
    walls = zero_metal_axes(grid)
    near_phase: List[float] = []
    far_phase: List[float] = []
    for axis in range(3):
        phase = phases[axis]
        if phase is not None and int(phase) not in (1, -1):
            raise ValueError(
                f"axis {axis} is folded but its mirror phase is {phase!r}; a plane's "
                f"parity is +1 or -1, and an even-mirror default standing in for a "
                f"plane that declared otherwise is a run wrong by twice the field "
                f"wherever the parity mattered")
        if phase is not None and bool(walls[axis]):
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction "
                f"(stepping.py:2284-2286) and this weld's resolution relies on the "
                f"two sets being disjoint")
        if rows[axis] is not None and phase is None:
            raise ValueError(
                f"axis {axis} carries a far reflect row {rows[axis]!r} and no mirror "
                f"phase; stepping._fill_folded_far_ghosts weights that image with "
                f"the plane's parity and cannot run without one")
        if phase is None:
            near_phase.append(0.0)
            far_phase.append(0.0)
            continue
        wanted_near = {component_parity("D", component, axis, int(phase))
                       for component in range(3) if axis in near_axes(component)}
        wanted_far = {component_parity("D", component, axis, int(phase))
                      for component in range(3) if far_axis(component) == axis}
        if len(wanted_near) != 1 or len(wanted_far) != 1:
            raise AssertionError(
                f"axis {axis} gives near parities {sorted(wanted_near)} and far "
                f"parities {sorted(wanted_far)}; fields.mirror_parity depends only "
                f"on the component's Yee shift on this axis, and the near set is "
                f"the shift-0 components and the far set the shift-1 one, so each "
                f"must be a single value. A drifted IYEE_SHIFTS would put a wrong "
                f"sign on a whole plane")
        near_phase.append(float(wanted_near.pop()))
        far_phase.append(float(wanted_far.pop()))
    return {
        "near": tuple(int(phases[axis] is not None) for axis in range(3)),
        "wall": tuple(int(bool(walls[axis])) for axis in range(3)),
        "reflect": tuple(-1 if rows[axis] is None else int(rows[axis])
                         for axis in range(3)),
        "near_phase": tuple(near_phase),
        "far_phase": tuple(far_phase),
    }
