"""The FIRST fused pair this track has put on a STENCIL cell: PML curl x off-diagonal.

THE CELL, AND WHAT WAS RECORDED ABOUT IT. ``D_to_E (cuda_curl/PML,
cuda_offdiag/off-diagonal)`` carries 16 corpus rows, 8 of which clear the source
injection on the driver fact alone, and the hand-CUDA board has recorded it
STENCIL-BLOCKED since the cell was first priced:

    "the constitutive half reads the curl half's IN-PLACE output at a cell this
     thread does not own. In one launch that cell is written by another block, and
     CUDA offers no grid-wide barrier inside an ordinary launch -- only cooperative
     groups, which caps the grid at the resident-block count."

Every word of that is true of the weld it describes, and the fusion-residue audit
(§1.1) found that the strongest design was never built. This module is that design:
the curl half writes a LAUNCH-LOCAL SCRATCH, the constitutive half takes its own cell
from a register and RECOMPUTES each foreign cell from pre-launch state through the
same ``__device__`` function, and the launcher rotates the ``D``/``fu`` bindings
afterwards. :mod:`.offdiag_stencil_weld` holds the lift and the argument for it; this
module owns the cell, the predicate, the signature and the record.

=============================================================================
WHAT ONE LAUNCH PERFORMS, AND WHAT IT REFUSES
=============================================================================

:data:`REPLACES` is ``step_D`` -> ``zero_metal_D`` -> ``update_E``: a contiguous run
of driver order (driver.py:3302, :3310, :3313). All 8 of the cell's reachable rows
drive ``zero_metal_D`` -- the board's own per-pass ledger for this cell is
``{"zero_metal_D": 8}`` -- so the wall clear is CARRIED, in the D family's
OFF-DIAGONAL table (two components per wall: x wall Dy and Dz, y wall Dx and Dz, z
wall Dx and Dy), and it is carried inside :func:`offdiag_stencil_weld.resolution_source`
rather than as a separate pass, because the constitutive half reads its output at
foreign cells too.

THE TWO MIRROR FILLS ARE NOT CARRIED AND CANNOT RUN. ``covers_real_pml_offdiag_
constitutive`` refuses a fold twice over (``has_symmetry`` and per axis,
coverage.py:1755-1791), so ``fill_symmetry_bc_D`` (driver.py:3309) and
``fill_folded_far_ghosts_D`` (:3311) are no-ops on every grid this predicate admits.
The resolution still SPELLS them -- the emitted ``resolve_D*`` carries the near and
far arms with their runtime flags bound to 0 -- because the same emitter serves
:mod:`.folded_offdiag_fused_electric_pair`, where they are live, and one device
string per family is this track's standing rule. The predicate asserts the flags are
dead rather than trusting the sibling clause.

=============================================================================
THE DEPOSIT IS REFUSED, AND THAT IS A MEASUREMENT, NOT A CHOICE
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is ``False``, and it is the one value this module could
declare. ``deposit_repair.repairable(fields, "D")`` refuses an off-diagonal chi1inv
row BY NAME (deposit_repair.py:686-691) -- the repair recomputes ``update_E`` at the
deposit cells, and with an off-diagonal row that output depends on cells a point
repair does not restore AND changes the answer at cells the deposit never touched.
Every row of this cell has such a row installed by construction (the constitutive
arm's clause (a) REQUIRES one), so the repair is unavailable here whatever this
module declared, and declaring ``True`` would bracket a launch with a repair that
refuses.

The ceiling that follows is exact: 8 of the cell's 16 seam-instances, the 8 the board
already prices as clearing the source seam.

=============================================================================
THE THREE ALIASING FACTS THE SIGNATURE IS SHAPED AROUND
=============================================================================

1. **``D`` and ``fu_D`` appear TWICE, and that is the design rather than a hazard.**
   The ``_out`` group is written and the bare group is read; they are DIFFERENT
   ALLOCATIONS, checked by base address in :func:`assert_scratch_is_disjoint` before
   any launch. Both carry ``__restrict__``, which is a promise the launcher keeps by
   allocating the scratch itself and refusing a caller-supplied one that collides.
2. **the three inverse-permittivity pointers are NOT ``__restrict__``**, lifted from
   the certified ``update_E`` rather than decided here: an isotropic run hands the
   same device pointer three times (fields.py:1321-1326).
3. **the row coefficients are NOT ``__restrict__`` either**, for
   ``offdiag_emitter``'s own reason: a row volume may legally alias another row's, an
   epsilon volume or a D volume.

=============================================================================
THE ONE NAME COLLISION, AND WHY IT MATTERS
=============================================================================

Both certified bodies were written against the same vocabulary and collide on
``kms_x/kms_y/kms_z`` -- the INTEGER sub-lattice for the D curl
(``real_pml_curl_tables(pml, half_integer=False)``) and the HALF-INTEGER one for
``update_E`` (``coverage.constitutive_sub_lattice("E")``). Both compile perfectly and
differ by half a cell in the absorber profile: converged, smooth and entirely wrong.
The constitutive group is therefore renamed ``kms_half_*``, the same rename
:mod:`.fused_electric_pair` makes at the same seam, and the three
``constitutive_apply`` lines are matched WHOLE so a moved argument list is a named
failure rather than a partial substitution.

``bc_x``/``bc_y``/``bc_z`` do NOT collide on this family, and that is CHECKED rather
than assumed: ``step_curl_kernels.real_curl_boundary_codes`` and
``offdiag_constitutive_kernels.offdiag_boundary_codes`` are two different readings of
the grid that agree on every unfolded one, and :func:`offdiag_fused_electric_pair_codes`
raises if they ever disagree instead of binding two vectors that a reader would have
to keep apart. The folded sibling, where they genuinely differ, binds both.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch. This
module ships a predicate, an emitter and a launcher; ``fused_pairs`` can plan it
opt-in (``arms.plan_step(..., fuse=True)``), and no shipped code path launches it.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import coverage as _coverage
from . import offdiag_emitter as _flat
from . import offdiag_stencil_weld as _weld
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import (covers_real_pml_curl, covers_real_pml_offdiag_constitutive,
                       offdiag_row_mask, offdiag_row_volumes,
                       offdiag_wall_mask_flags)
from .in_seam_coverage import zero_metal_axes

from .. import deposit_repair as _deposit_repair

# The certified curl's table and code helpers. Taken defensively for the reason every
# module on this track takes them defensively: the predicate is the half whose failure
# is a silent wrong answer, and it must answer on a host with no device.
try:
    from . import step_curl_kernels
except Exception:  # noqa: BLE001
    step_curl_kernels = None  # type: ignore[assignment]
try:
    from . import constitutive_kernels
except Exception:  # noqa: BLE001
    constitutive_kernels = None  # type: ignore[assignment]
try:
    from . import offdiag_constitutive_kernels
except Exception:  # noqa: BLE001
    offdiag_constitutive_kernels = None  # type: ignore[assignment]

#: BYTE-IDENTICAL TO THE ARRAY PATH WITH A DEVICE VERDICT BEHIND IT.
#: ``certification.json``'s ``cuda_offdiag_stencil_welds_2026-09-02`` block names the same
#: two artifacts this line rests on, and ``test_kernel_partition.py`` refuses a
#: name here that no record block claims.
#:
#: THE DECLARATION IS THE FINAL BYTES, AND THAT IS THIS CAMPAIGN'S RULE 4 RATHER
#: THAN a claim made ahead of its evidence: a gate certifies the module it
#: imported, so a module that moved its own name between the two sets AFTER the
#: run would leave the record bound to bytes that no longer ship -- which is
#: exactly the drift ``rebind_cuda_welds.py`` exists to catch. The run this line
#: rests on was cut against these bytes; the earlier pass that measured the
#: design is not what the record binds.
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "offdiag_fused_electric_pair_pml_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS: Dict[str, str] = {}

#: FALSE, AND MEASURED OFF ``deposit_repair.repairable`` RATHER THAN CHOSEN. Every row
#: of this cell has an off-diagonal chi1inv row installed -- the constitutive arm's
#: predicate REQUIRES one -- and ``repairable(fields, "D")`` refuses exactly that
#: (deposit_repair.py:686-691). A product declaring True here would bracket its launch
#: with a repair that refuses on every row it serves, which is a refusal wearing a
#: carry's name. :func:`covers_offdiag_fused_electric_pair` therefore refuses every
#: electric deposit through the shared seam clause, and the cell's ceiling is the 8
#: rows the board already prices as clearing the injection on the driver fact alone.
#:
#: THE HOST SUITE ASSERTS THIS AGAINST THE CENSUS rather than against this comment:
#: ``test_offdiag_stencil_welds.py`` replays the board's own selection over the
#: residual-welds census, takes this cell's rows, and asserts every one of them is
#: refused by ``deposit_repair.repairable`` for the off-diagonal clause.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_offdiag_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once. Deliberately NOT either half's name:
#: a third kernel wearing a certified one's name would make ``certification.json``'s
#: partition test and any NVRTC binary observation ambiguous about which body it saw.
KERNEL_NAME = "offdiag_fused_electric_pair_pml_real"

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3302, :3310, :3313). Declared rather than inferred from the two slots:
#: ``zero_metal_D`` is a third driver pass that no slot names.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

_ELECTRIC: Tuple[str, str, str] = ("Ex", "Ey", "Ez")
_D_TARGETS: Tuple[str, str, str] = _weld.D_TARGETS

#: The scratch volumes, in signature order. Named for the field each shadows.
SCRATCH_VOLUMES: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")

_FUSED_THREADS = 256

#: Imported by value from the two siblings so loading this file by path cannot pick up
#: a different tuple than the one a gate compiled.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: The certified constitutive prelude's boundary defines, which the CURL prelude
#: already supplies identically. Carried once; see :data:`LIFT_EDITS`.
_DUPLICATE_BC_DEFINES = "\n#define BC_PERIODIC 0\n#define BC_METALLIC 1\n"

#: ``offdiag_term``'s certified parameter list and its two ghosted-load lines -- the
#: three anchors that turn the certified helper into one that takes the home sample
#: from a register and RESOLVES the three foreign ones.
_TERM_PARAMETERS = (
    "    const float* g, const float* u, int home, int down, int up, int corner\n)")
_TERM_PARAMETERS_RESOLVED = (
    "    int comp, float g_home, const float* u, int home, int down, int up,\n"
    "    int corner, const WeldArgs& weld\n)")
_TERM_NEAR = "    float near_pair = g[home] + ghosted(g, down);\n"
_TERM_NEAR_RESOLVED = (
    "    // THE SEAM. The home sample is the register the thread already resolved\n"
    "    // for its own cell; the DOWN sample is a foreign cell, recomputed from\n"
    "    // pre-launch state. Same two operands, same order, same addition.\n"
    "    float near_pair = g_home + resolved_ghosted(comp, down, weld);\n")
_TERM_FAR = "    float far_pair = ghosted(g, up) + ghosted(g, corner);\n"
_TERM_FAR_RESOLVED = (
    "    float far_pair = resolved_ghosted(comp, up, weld)"
    " + resolved_ghosted(comp, corner, weld);\n")

#: Every line of certified text this file does not lift verbatim, with the reason.
#: DATA, not prose, so a gate and the host suite can assert the list rather than a
#: docstring. The CURL half's edits live in :data:`offdiag_stencil_weld.CURL_LIFT_EDITS`
#: because both families make them identically; these are this family's own.
LIFT_EDITS: Tuple[Dict[str, str], ...] = _weld.CURL_LIFT_EDITS + (
    {"line": "#define BC_PERIODIC 0\n#define BC_METALLIC 1",
     "became": "(removed from offdiag_emitter.PRELUDE)",
     "why": "the curl prelude defines both, identically and first. A second "
            "IDENTICAL definition is legal C and a second DIFFERENT one would be a "
            "warning rather than an error, so the weld carries exactly one of each "
            "and the removal is anchored on the certified spelling."},
    {"line": "    const float* g, const float* u, int home, int down, int up, "
             "int corner\n)",
     "became": "    int comp, float g_home, const float* u, int home, int down, "
               "int up,\n    int corner, const WeldArgs& weld\n)",
     "why": "the partner volume becomes a COMPONENT TAG and the home sample a "
            "register. The tag is a compile-time constant at every call site, so "
            "resolve_D's dispatch chain folds away; the register is the value this "
            "thread already resolved for its own cell, which is what the certified "
            "body's g[home] would have loaded."},
    {"line": "    float near_pair = g[home] + ghosted(g, down);",
     "became": "    float near_pair = g_home + resolved_ghosted(comp, down, weld);",
     "why": "the same addition of the same two samples; what changes is that the "
            "foreign one is RECOMPUTED from pre-launch state instead of loaded from "
            "a volume another block is writing. resolved_ghosted is ghosted's body "
            "with the load replaced -- the index < 0 metallic-zero arm is unchanged."},
    {"line": "    float far_pair = ghosted(g, up) + ghosted(g, corner);",
     "became": "    float far_pair = resolved_ghosted(comp, up, weld) + "
               "resolved_ghosted(comp, corner, weld);",
     "why": "the same, for the term's other two foreign samples. The 0.25f scale, "
            "the coefficient multiply BETWEEN the two shifts and the two "
            "accumulations are untouched -- offdiag_term's return line is verbatim."},
    {"line": "    float gs_E* = D*[idx];",
     "became": "    float gs_E* = v_*;",
     "why": "THE SEAM, diagonal half. The certified body opens each component by "
            "loading the flux density at its own cell; the weld has already resolved "
            "that cell into a register, and the register is the post-clear value the "
            "array path would have loaded. The multiply that follows, its operand "
            "ORDER (D on the left, stepping.py:1011) and the fact that all three "
            "products are formed before any store are untouched."},
    {"line": "        D*, chi1inv_E*_E*, idx,",
     "became": "        <tag>, v_*, chi1inv_E*_E*, idx,",
     "why": "the call-site half of the offdiag_term rewrite: the partner volume "
            "becomes its component tag plus that partner's own-cell register. The "
            "coefficient argument, the home index and the three shifted indices are "
            "untouched, so a mispaired partner would have to be introduced "
            "deliberately."},
    {"line": "        flat(...));   (the closing line of each offdiag_term call)",
     "became": "        flat(...), weld);",
     "why": "the other half of the same rewrite: the resolved helper needs the "
            "argument pack. Found STRUCTURALLY -- the term's opening line, then the "
            "first line below it that closes the call -- rather than by matching a "
            "corner index expression, which differs between row masks and would be "
            "an anchor that silently stopped applying."},
    {"line": "    constitutive_apply(E*, f_w_E*, idx, src_E*, kps_*[*], kms_*[*]);",
     "became": "    constitutive_apply(E*, f_w_E*, idx, src_E*, kps_*[*], "
               "kms_half_*[*]);",
     "why": "ONE RENAME, NO ARITHMETIC. kms_* is the INTEGER sub-lattice for the D "
            "curl and the HALF-INTEGER one for update_E, and letting one shadow the "
            "other in a single scope is a half-cell error in the absorber profile "
            "rather than a compile failure."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= nx * ny * nz) return;   (offdiag_emitter.TEMPLATE)",
     "became": "(the weld's own guarded preamble, once)",
     "why": "the constitutive TEMPLATE's own preamble and index decode are dropped "
            "because the weld emits them once for both halves; splicing both would "
            "redeclare idx, i, j and k and the kernel would not compile. The "
            "decode's arithmetic is the certified curl body's, character for "
            "character."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SCRATCH_VOLUMES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_scratch_is_disjoint", "constitutive_body", "covers_offdiag_fused_electric_pair",
    "device_sources", "kernel_source", "launch_offdiag_fused_electric_pair",
    "offdiag_fused_electric_pair_codes", "offdiag_fused_electric_pair_scratch",
    "offdiag_fused_electric_pair_tables", "prelude", "rotate_into_fields",
    "run_offdiag_fused_electric_pair", "signature",
]


# =============================================================================
# THE DEVICE CODE
# =============================================================================

def prelude() -> str:
    """Both certified preludes plus the weld's own three helpers.

    ORDER MATTERS AND IS DECLARED: the curl prelude first (it owns the two boundary
    defines and ``shift_up``/``shift_dn``), then the constitutive prelude with its
    duplicate defines removed and ``offdiag_term`` resolved, then the argument pack,
    the cell-wise curl and the resolution -- because ``offdiag_term`` calls
    ``resolved_ghosted``, which calls ``resolve_D``, which calls ``raw_step_D_cell``,
    and CUDA needs each declared before its caller.
    """
    constitutive = _flat.PRELUDE
    for anchor, what in ((_DUPLICATE_BC_DEFINES, "the duplicate boundary defines"),
                         (_TERM_PARAMETERS, "offdiag_term's parameter list"),
                         (_TERM_NEAR, "offdiag_term's near-pair line"),
                         (_TERM_FAR, "offdiag_term's far-pair line")):
        if constitutive.count(anchor) != 1:
            raise AssertionError(
                f"offdiag_emitter.PRELUDE carries {constitutive.count(anchor)} copies "
                f"of {what}, not one; this weld LIFTS that text rather than retyping "
                f"it and cannot splice around its absence")
    constitutive = constitutive.replace(_DUPLICATE_BC_DEFINES, "\n", 1)
    head, _, tail = constitutive.partition(_TERM_PARAMETERS)
    constitutive = head + _TERM_PARAMETERS_RESOLVED + tail
    constitutive = constitutive.replace(_TERM_NEAR, _TERM_NEAR_RESOLVED, 1)
    constitutive = constitutive.replace(_TERM_FAR, _TERM_FAR_RESOLVED, 1)
    return (_weld.curl_prelude()
            + _weld.weld_args_struct()
            + _weld.raw_step_D_cell_source()
            + _weld.resolution_source()
            + constitutive)


def signature(row_mask: Sequence[int]) -> str:
    """The kernel's parameter list, with only the LIVE row coefficients bound.

    The dead slots have no parameter, which is ``offdiag_emitter``'s own choice for
    its own reason: binding a dead slot to some other volume leaves a pointer aimed
    at an array the kernel must never touch.
    """
    mask = _flat.normalized_row_mask(row_mask)
    rows = "\n".join(f"    const float* {_flat.ROW_PARAMETERS[slot]},"
                     for slot, flag in enumerate(mask) if flag)
    return f'''
extern "C" __global__ void {KERNEL_NAME}(
    // THE SCRATCH OUTPUTS. A DIFFERENT ALLOCATION from the D/fu group below --
    // assert_scratch_is_disjoint checks it by base address before every launch --
    // and that separation is the whole design: nothing this launch reads is ever a
    // word this launch wrote, so the constitutive half's foreign samples are a pure
    // function of unwritten memory rather than a race with another block.
    float* __restrict__ Dx_out, float* __restrict__ Dy_out,
    float* __restrict__ Dz_out,
    float* __restrict__ fu_Dx_out, float* __restrict__ fu_Dy_out,
    float* __restrict__ fu_Dz_out,
    // THE PRE-LAUNCH STATE, read at ANY cell by ANY thread and written by none.
    const float* __restrict__ Dx, const float* __restrict__ Dy,
    const float* __restrict__ Dz,
    const float* __restrict__ fu_Dx, const float* __restrict__ fu_Dy,
    const float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    // E and its constitutive history: written at the thread's OWN cell only, which
    // is why they need no scratch.
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    // NOT __restrict__, lifted from the certified update_E rather than decided here:
    // an isotropic run hands the same device pointer three times
    // (fields.py:1321-1326), and restrict on mutually aliasing arguments is a
    // promise the caller cannot keep.
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    // NOT __restrict__ either: a row volume may legally alias another row's, an
    // epsilon volume or a D volume (offdiag_emitter's own reason).
{rows}
    int nx, int ny, int nz, float dtdx,
    // The D curl's INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // update_E's HALF-INTEGER pair. Its kms is renamed kms_half_*: the two
    // sub-lattices collide on the bare name in one scope, and a shadow there is a
    // half-cell error in the absorber profile, not a compile failure.
    const float* __restrict__ kps_x, const float* __restrict__ kms_half_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_half_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_half_z,
    // ONE set of boundary codes, and that is CHECKED rather than assumed:
    // real_curl_boundary_codes and offdiag_boundary_codes are two readings of the
    // grid that agree on every unfolded one, and this family admits nothing else.
    // offdiag_fused_electric_pair_codes raises where they disagree.
    int bc_x, int bc_y, int bc_z,
    // The COUPLING's wall mask (coverage.offdiag_wall_mask_flags): whether
    // _mask_metallic_wall_coupling zeroes a component's coupling total at face 0.
    int wm_x, int wm_y, int wm_z,
    // zero_metal_D's walls (in_seam_coverage.zero_metal_axes). A DIFFERENT question
    // from wm_*: that one masks the coupling TOTAL, this one clears the stored D.
    int wall_x, int wall_y, int wall_z,
    // The two mirror fills' plan. Dead on every grid this predicate admits -- the
    // constitutive arm refuses a fold -- and bound anyway, because the same
    // resolution serves the folded sibling and one device string per family is this
    // track's rule. The predicate asserts they are dead rather than trusting it.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float near_phase_x, float near_phase_y, float near_phase_z,
    float far_phase_x, float far_phase_y, float far_phase_z
) {{
'''


def constitutive_body(row_mask: Sequence[int]) -> str:
    """``update_E_pml_real_offdiag``'s body, lifted, with the four seam edits.

    Not a transcription: this is ``offdiag_emitter.offdiag_source(row_mask)`` itself,
    minus its prelude, its signature and its thread preamble, with every D read
    routed through the resolution and the sub-lattice rename applied.
    """
    mask = _flat.normalized_row_mask(row_mask)
    body = _weld.split_body(_flat.offdiag_source(mask), _flat.PRELUDE,
                            f"offdiag_source({tuple(mask)})")
    # (1) the constitutive TEMPLATE's own preamble and decode: the weld emits them
    #     once, above, for both halves.
    for anchor in ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                   "    if (idx >= nx * ny * nz) return;\n",
                   "    int nyz = ny * nz;\n    int k = idx % nz;\n"
                   "    int j = (idx / nz) % ny;\n    int i = idx / (ny * nz);\n"):
        if body.count(anchor) != 1:
            raise AssertionError(
                f"the certified off-diagonal body carries {body.count(anchor)} copies "
                f"of its thread preamble block, not one; the weld emits that block "
                f"once and cannot splice around a changed one")
        body = body.replace(anchor, "", 1)
    # (2) THE SEAM, diagonal half.
    for component, target in enumerate(_D_TARGETS):
        name = _ELECTRIC[component]
        line = _weld.the_line_starting(
            body, f"    float gs_{name} = {target}[idx];",
            f"{name}'s own-cell flux-density load")
        body = body.replace(
            line,
            f"    float gs_{name} = v_{'xyz'[component]};   "
            f"// the register this thread resolved for its own cell", 1)
    # (3) THE SEAM, foreign half: every offdiag_term call's partner argument.
    for slot, flag in enumerate(mask):
        if not flag:
            continue
        component, offset = divmod(slot, 2)
        partner = _coverage.OFFDIAG_TRANSVERSE_PARTNERS[component][offset]
        coefficient = _flat.ROW_PARAMETERS[slot]
        line = _weld.the_line_starting(
            body, f"        {_D_TARGETS[partner]}, {coefficient}, idx,",
            f"the partner argument of the {coefficient} term")
        body = body.replace(
            line,
            f"        {partner}, v_{'xyz'[partner]}, {coefficient}, idx,", 1)
        component_name = _ELECTRIC[component]
        body = _weld.close_call_with_the_pack(
            body, f"    float term_{component_name}_{offset} = offdiag_term(",
            f"the {coefficient} term")
    # (4) THE SUB-LATTICE RENAME. Matched as a WHOLE LINE so a moved argument list is
    #     a named failure rather than a partial substitution.
    for component in range(3):
        name = _ELECTRIC[component]
        axis = "xyz"[component]
        index = "ijk"[component]
        line = _weld.the_line_starting(
            body, f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},",
            f"{name}'s certified constitutive_apply call")
        wanted = (f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},"
                  f" kps_{axis}[{index}], kms_{axis}[{index}]);")
        if line != wanted:
            raise AssertionError(
                f"the certified off-diagonal body spells {name}'s constitutive tail "
                f"{line!r}; this weld renames only the half-integer kms and cannot "
                f"do so on a line whose argument list has moved")
        body = body.replace(
            line,
            f"    constitutive_apply({name}, f_w_{name}, idx, src_{name},"
            f" kps_{axis}[{index}], kms_half_{axis}[{index}]);", 1)
    return body


def kernel_source(row_mask: Sequence[int]) -> str:
    """The whole device source for one row mask."""
    mask = _flat.normalized_row_mask(row_mask)
    preamble = [
        "    // ONE bounds guard and ONE index decode for both halves -- the",
        "    // certified curl's own two blocks, kept here and dropped from the",
        "    // certified constitutive body, which emits the identical arithmetic",
        "    // plus the nyz its flat() needs.",
        "    const int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx >= nx * ny * nz) return;",
        "",
        "    const int nyz = ny * nz;",
        "    const int k = idx % nz;",
        "    const int j = (idx / nz) % ny;",
        "    const int i = idx / (ny * nz);",
        "",
        _weld.weld_args_construction(),
        "    // THE CURL HALF. Both results go to the SCRATCH; fu is the RAW stepped",
        "    // value at this cell because no in-seam pass ever touches it",
        "    // (stepping.py:1451 writes field and never fu_field; _zero_metal writes",
        "    // only the named components, :2247).",
        "    float d_raw[3];",
        "    float fu_raw[3];",
        "    raw_step_D_cell(i, j, k, weld, d_raw, fu_raw);",
        "    fu_Dx_out[idx] = fu_raw[0];",
        "    fu_Dy_out[idx] = fu_raw[1];",
        "    fu_Dz_out[idx] = fu_raw[2];",
        "",
        "    // THE THREE IN-SEAM PASSES, resolved for this thread's own cell. The",
        "    // same function the constitutive half calls for a foreign cell, so the",
        "    // own-cell value and a neighbour's are the same arithmetic by",
        "    // construction rather than by a second transcription.",
        "    const float v_x = resolve_Dx(i, j, k, weld);",
        "    const float v_y = resolve_Dy(i, j, k, weld);",
        "    const float v_z = resolve_Dz(i, j, k, weld);",
        "    Dx_out[idx] = v_x;",
        "    Dy_out[idx] = v_y;",
        "    Dz_out[idx] = v_z;",
        "    // The certified off-diagonal body follows, VERBATIM from its own",
        "    // neighbour-coordinate line down: di/ui, dj/uj, dk/uk, the wall-plane",
        "    // predicates and the three component blocks are lifted, not retyped.",
    ]
    return (prelude() + signature(mask) + "\n".join(preamble)
            + constitutive_body(mask) + "}\n")


def device_sources(row_mask: Sequence[int] = (1, 1, 1, 1, 1, 1)) -> Dict[str, str]:
    """``{kernel name: source}`` -- the shape every family on this track publishes."""
    return {KERNEL_NAME: kernel_source(row_mask)}


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Sixty-three row masks is too many to pin one digest each without burying the
    record and not too many to hash, so the whole enumeration hashes to one value: a
    single changed character anywhere in the weld or in either certified half moves
    it. The per-source digests of the masks a gate launches belong in that artifact.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for mask in _flat.LIVE_ROW_MASKS:
        digest.update(repr(mask).encode("ascii"))
        digest.update(kernel_source(mask).encode("utf-8"))
    return digest.hexdigest()


def _get_kernel(row_mask: Sequence[int], source: Optional[str] = None):
    """Compile on first use, memoized on (name, options, policy, source).

    THE ROW MASK REACHES THE KEY THROUGH THE SOURCE, which is already in it, so two
    specializations cannot be served each other's binary -- and neither can a
    mutation harness that rewrote the emitter's output be served the unmutated one.
    ``cupy`` is imported HERE, never at scope.
    """
    import cupy as cp  # noqa: PLC0415 - device-only, in a laptop-importable module

    code = kernel_source(row_mask) if source is None else source
    key = _kernel_cache_key(KERNEL_NAME, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# =============================================================================
# THE PREDICATE
# =============================================================================

def covers_offdiag_fused_electric_pair(fields: Any, pml: Any, grid: Any,
                                       sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``zero_metal_D`` -> ``update_E`` here?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal, this
    directory's convention.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own
    certified predicate refuses is refused here with that half's reason, prefixed so
    a reader can tell which side said it. What this predicate ADDS is four seam
    clauses: the two mirror fills must be dead, the two boundary readings must agree,
    the source slot must be empty, and the three stored extents must be one tuple.
    """
    covered, reason = covers_real_pml_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_offdiag_constitutive(fields, pml, grid)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE TWO MIRROR FILLS ARE REFUSED, NOT CARRIED, and the refusal is stated HERE
    # rather than inherited. The constitutive half above already refuses every folded
    # grid -- twice -- so this clause is unreachable today, and that is precisely why
    # it is written: the guarantee that fill_symmetry_bc_D (driver.py:3309) and
    # fill_folded_far_ghosts_D (:3311) do nothing inside this seam is what makes
    # REPLACES honest, and it may not depend on a clause in another module that a
    # future device verdict could licence away. The FOLDED sibling is the product for
    # that configuration.
    for name in ("is_mirrored", "has_symmetry"):
        if not callable(getattr(grid, name, None)):
            return False, (f"grid does not expose {name}; this seam cannot tell "
                           f"whether fill_symmetry_bc_D and fill_folded_far_ghosts_D "
                           f"run inside it")
    try:
        folded = tuple(bool(grid.is_mirrored(axis)) for axis in range(3))
        symmetry = bool(grid.has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry or any(folded):
        return False, ("a mirror plane is active: fill_symmetry_bc_D "
                       "(driver.py:3309) and fill_folded_far_ghosts_D (:3311) run "
                       "inside this seam, and this pair's constitutive arm is the "
                       "UNFOLDED one -- folded_offdiag_fused_electric_pair is the "
                       "product for that cell")

    # THE TWO BOUNDARY READINGS MUST AGREE, checked rather than assumed. This family
    # binds ONE bc_* triple to both halves; the readings are the same function of an
    # unfolded grid and different functions of a folded one, so the check is what
    # keeps the single binding honest if the fold clause above ever moves.
    try:
        offdiag_fused_electric_pair_codes(grid, pml)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"the two boundary readings this launch binds as one: "
                       f"{type(exc).__name__}: {exc}")

    # THE SOURCE SEAM, REFUSED RATHER THAN CARRIED. An ELECTRIC source is injected
    # BETWEEN the two halves (driver.py:3305/:3308), so a fused pair would compute
    # update_E against a pre-injection D. CARRIES_DEPOSIT_REPAIR is False -- and could
    # not be True, because deposit_repair.repairable refuses an off-diagonal chi1inv
    # row by name (deposit_repair.py:686-691) and every row of this cell has one.
    # IGNORANCE IS NEVER AN EMPTY SET: Fields does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be exactly the
    # over-covering this clause exists to prevent.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, "D",
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3305/:3308) and this "
            f"pair declares no deposit repair -- deposit_repair.repairable refuses an "
            f"off-diagonal chi1inv row, which every row of this cell carries"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

    # THE STORED SHAPE THE KERNEL INDEXES MUST BE ONE TUPLE, on three readings: the
    # launch walks Dx.shape, the certified curl's guard is nx*ny*nz and the certified
    # constitutive's is its own bounds test, and this kernel emits ONE guard. If those
    # ever disagreed the fused launch would run one half over a range its own
    # certified kernel would not have.
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        curl_extents = tuple(int(n) for n in fields.Dx.shape)
        constitutive_extents = tuple(int(n) for n in fields.Ex.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid or fields could not state the stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if not stored == curl_extents == constitutive_extents:
        return False, (f"grid.stored_cells is {stored}, Dx.shape is {curl_extents} "
                       f"and Ex.shape is {constitutive_extents}; this launch emits "
                       f"ONE bounds guard and may only do so where the three are one "
                       f"number")

    # THE SPLIT-FIELD STORE MUST EXIST AND BE SHAPED LIKE THE VOLUME IT SHADOWS. The
    # curl half's own predicate checks the D volumes; the scratch this product
    # allocates is shaped from the fu volumes, and an absent one is a launch with a
    # null pointer rather than a refusal.
    for name in ("fu_Dx", "fu_Dy", "fu_Dz"):
        volume = getattr(fields, name, None)
        if volume is None:
            return False, (f"{name} is not allocated: this pair writes the "
                           f"split-field recurrence into a scratch shaped like it, "
                           f"and step_D's recurrence has nowhere to land")
        if tuple(int(n) for n in getattr(volume, "shape", ())) != curl_extents:
            return False, (f"{name} has shape {tuple(volume.shape)} against Dx's "
                           f"{curl_extents}; the scratch this pair rotates in is "
                           f"shaped from it")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

def offdiag_fused_electric_pair_tables(pml: Any) -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads.

    THE SUB-LATTICE IS DECIDED HERE, not by the caller: while it was an argument,
    every call site was one more place the pairing could be got backwards, and
    backwards is a half-cell error rather than a crash. The D curl reads the INTEGER
    positions and ``update_E`` the HALF-INTEGER ones, through the same two functions
    the certified launchers ask.
    """
    if step_curl_kernels is None or constitutive_kernels is None:
        raise RuntimeError(
            "step_curl_kernels/constitutive_kernels are not importable on this host "
            "(they import CuPy at module scope), so the coefficient tables cannot be "
            "resolved. covers_offdiag_fused_electric_pair needs neither and still "
            "answers")
    return {
        "curl": step_curl_kernels.real_pml_curl_tables(pml, False),
        "constitutive": constitutive_kernels.real_constitutive_tables(
            pml, _coverage.constitutive_sub_lattice("E")),
    }


def offdiag_fused_electric_pair_codes(grid: Any, pml: Any = None) -> Tuple[int, ...]:
    """The ONE boundary triple this launch binds, with both readings cross-checked.

    ``coverage.real_curl_boundary_codes`` is the CURL's ghost-rule resolution and
    ``BC_CODES[real_pml_boundary_kinds(...)]`` the CONSTITUTIVE's -- the two the
    device launchers delegate to, taken from ``coverage`` directly so this clause
    answers on a host with no CuPy, which is where the census asks it. On an unfolded grid -- the only kind this family admits -- they
    are the same function of the same declaration and agree by construction. This
    binds one vector to both halves and RAISES where they disagree, rather than
    binding two and leaving a reader to keep them apart: a disagreement means the
    fold clause moved, and the folded sibling is the product for that.
    """
    # BOTH READINGS COME FROM ``coverage``, WHICH IMPORTS NOTHING. That is not a
    # convenience: this function is a PREDICATE CLAUSE, and the census asks every
    # predicate on a laptop with no CuPy. Routing it through the two device modules
    # -- which merely delegate here (``step_curl_kernels.real_curl_boundary_codes``
    # is ``coverage.real_curl_boundary_codes``, and ``offdiag_boundary_codes`` is
    # ``BC_CODES[kind] for kind in real_pml_boundary_kinds``) -- made the whole
    # predicate unanswerable off-device, which is a refusal wearing a measurement's
    # name. The two device functions are asserted equal to these in the host suite,
    # so the delegation cannot drift.
    # ``coverage.real_curl_boundary_codes`` FAILS CLOSED -- it returns
    # ``(codes, refusal)`` with exactly one None so a caller need not catch -- and
    # the refusal is carried here rather than swallowed: a grid whose fold
    # termination the curl cannot resolve is a grid this launch must not bind.
    codes, refusal = _coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(
            f"the curl's own boundary resolution refuses this grid: {refusal}")
    curl = tuple(int(code) for code in codes)
    constitutive = tuple(int(_coverage.BC_CODES[kind])
                         for kind in _coverage.real_pml_boundary_kinds(grid))
    if curl != constitutive:
        raise ValueError(
            f"the curl reads this grid's boundaries as {curl} and the off-diagonal "
            f"constitutive as {constitutive}; this pair binds ONE bc_* triple to both "
            f"halves and may only do so where they agree. They diverge exactly on a "
            f"FOLD, which folded_offdiag_fused_electric_pair serves")
    return curl


def offdiag_fused_electric_pair_scratch(fields: Any) -> Dict[str, Any]:
    """The six launch-local volumes, allocated once per frozen configuration.

    NOT ``StepScratch``. That pool is documented for "values consumed within the call
    that produced them" (fields.py:298-327), and these are the opposite: the rotation
    hands them to the driver as the live ``D``/``fu_D`` and keeps the retired pair for
    the next launch. Taking a pool slot and then rotating it out would break the
    pool's own tag invariant.

    UNINITIALIZED IS CORRECT. Every cell of all six is written by every launch --
    ``fu_*_out[idx]`` for every thread and ``D*_out[idx]`` for every thread -- so
    there is no cell whose prior contents a reader could observe. ``empty_like``
    rather than ``zeros_like`` for that reason and for the allocation cost the pool's
    own measurement records.
    """
    volumes: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        source = getattr(fields, name, None)
        if source is None:
            raise ValueError(
                f"fields.{name} is not allocated; this pair rotates a scratch shaped "
                f"like it and has nothing to shape one from")
        volumes[name] = _empty_like(source)
    return volumes


def _empty_like(array: Any) -> Any:
    """``xp.empty_like`` through the array's own module, never an imported one."""
    import cupy as cp  # noqa: PLC0415 - device-only

    return cp.empty_like(array)


def assert_scratch_is_disjoint(fields: Any, scratch: Dict[str, Any],
                               tables: Dict[str, Dict[str, Any]]) -> int:
    """THE CHECK THE WHOLE DESIGN RESTS ON, plus the ordinary restrict promise.

    Two things, and the first is not a convention:

    1. **no scratch volume is any bound input.** If ``Dx_out`` were ``Dx`` the launch
       would be the IN-PLACE weld the board refused -- every foreign recompute would
       read words other blocks had already overwritten, and the answer would be a
       schedule. This is the mutation ``scratch_aliased_to_storage`` on the device
       gate, and it is refused here so it cannot happen by accident.
    2. **no two ``__restrict__`` arguments are one allocation.** Two restrict pointers
       to one object is UB whatever the route to it, and NVRTC reorders across it
       without a diagnostic. The three ``inv_eps`` volumes are checked DIFFERENTLY and
       deliberately: they are not restrict-qualified, so they may alias each other --
       an isotropic run hands one pointer three times -- but may not alias a
       restrict-qualified argument.

    Returns the number of distinct allocations inspected, so a caller can assert that
    something was actually examined; a check that examined nothing and reported
    success is the vacuity this track guards against.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def address(array: Any) -> int:
        return int(array.data.ptr)

    def visit(label: str, array: Any) -> None:
        pointer = address(array)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in SCRATCH_VOLUMES:
        visit(f"{name}_out", scratch[name])
    for name in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
                 "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        visit(name, getattr(fields, name))
    for key, group in (("curl", tables["curl"]), ("constitutive", tables["constitutive"])):
        for table, volume in sorted(group.items()):
            visit(f"{key}:{table}", volume)
    if collisions:
        raise ValueError(
            "this pair binds every field, scratch and table argument __restrict__, "
            "and these arguments alias -- which is undefined behaviour NVRTC "
            "miscompiles silently rather than diagnosing, and which for the scratch "
            "group would make the launch the in-place weld the board refused: "
            + "; ".join(collisions))
    for component in _ELECTRIC:
        volume = fields.inverse_epsilon_for(component)
        pointer = address(volume)
        if pointer in bound:
            collisions.append(
                f"inv_eps_{component} and {bound[pointer]} are the same allocation, "
                f"and the second is bound __restrict__")
    if collisions:
        raise ValueError(
            "an inverse-permittivity volume aliases a __restrict__ argument: "
            + "; ".join(collisions))
    return len(bound)


def launch_offdiag_fused_electric_pair(
        fields: Any, scratch: Dict[str, Any], tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[int], wall_mask: Sequence[int],
        fills: Dict[str, Tuple[Any, ...]], dtdx: float, row_mask: Sequence[int],
        rows: Sequence[Any], kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch. THE CALLER ROTATES AFTERWARDS.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is: a gate
    compiles a deliberately broken copy of the shipped source and hands it here. A
    launcher that could not be handed its own kernel could not arm a single mutation,
    and every mutation leg would silently launch the shipped one and report the defect
    as uncaught.

    Returns the launch geometry rather than ``None`` so a gate can assert that
    something was actually launched.
    """
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments: List[Any] = [scratch[name] for name in SCRATCH_VOLUMES]
    arguments += [getattr(fields, name) for name in
                  ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz", "Hx", "Hy", "Hz",
                   "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")]
    arguments += [fields.inverse_epsilon_for(component) for component in _ELECTRIC]
    arguments += list(rows)
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx)]
    for axis in ("x", "y", "z"):
        arguments += [curl[f"kms_{axis}"], curl[f"sinv_{axis}"]]
    for axis in ("x", "y", "z"):
        arguments += [constitutive[f"kps_{axis}"], constitutive[f"kms_{axis}"]]
    arguments += [np.int32(int(code)) for code in boundary_codes]
    arguments += [np.int32(int(bool(flag))) for flag in wall_mask]
    arguments += [np.int32(int(flag)) for flag in fills["wall"]]
    arguments += [np.int32(int(flag)) for flag in fills["near"]]
    arguments += [np.int32(int(row)) for row in fills["reflect"]]
    arguments += [np.float32(float(value)) for value in fills["near_phase"]]
    arguments += [np.float32(float(value)) for value in fills["far_phase"]]
    (kernel or _get_kernel(row_mask))((blocks,), (_FUSED_THREADS,), tuple(arguments))
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "row_mask": tuple(int(flag) for flag in row_mask),
            "walls": tuple(int(flag) for flag in fills["wall"]),
            "coupling_wall_mask": tuple(int(bool(flag)) for flag in wall_mask),
            "fills": {key: tuple(fills[key])
                      for key in ("near", "reflect", "near_phase", "far_phase")}}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION. The scratch becomes the live D/fu; the retired pair becomes scratch.

    THE CERTIFIED E->P CHOREOGRAPHY, one seam earlier.
    ``ade_kernels.update_P_fused_pml_real`` writes a third buffer and rotates the
    names on the host afterwards (:414-424, transcribing dispersion.py:689-691); this
    is the same move on the D/fu pair, and it is what makes a scratch-output weld a
    step rather than a computation thrown away.

    THE CONSUMER AUDIT THIS RESTS ON. Every consumer of the flux density in this
    engine resolves it BY NAME at use time -- ``getattr(fields, "D" + name[1])``
    (stepping.py:2446), ``getattr(self.fields, component)`` (driver.py:3395, :3421,
    :3454, :3984, :4322…), ``fields.get_component(...)`` in every DFT and flux monitor
    -- so a rebinding between steps is invisible to all of them. That is not asserted
    from a grep: ``test_offdiag_stencil_welds.py`` drives a real ``FdtdDriver`` with
    sources and monitors for several steps with an equivalent rotation shim spliced in
    at exactly this point, and requires every stored volume AND every monitor
    accumulation to be byte-identical to the unrotated run.

    Returns the mapping the caller should keep as the NEXT launch's scratch.
    """
    retired: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        retired[name] = getattr(fields, name)
        setattr(fields, name, scratch[name])
    return retired


def run_offdiag_fused_electric_pair(fields: Any, grid: Any, pml: Any, dtdx: float, *,
                                    scratch: Optional[Dict[str, Any]] = None,
                                    sources: Any = None,
                                    tables: Optional[Dict[str, Dict[str, Any]]] = None,
                                    kernel: Optional[Any] = None,
                                    rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once, rotate.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``scratch``, ``tables``, ``kernel`` and ``rotate`` are the gate's doors,
    keyword-only. ``rotate=False`` is how the ``rotation_skipped`` mutation is armed:
    a launcher that always rotated could not measure what the rotation is worth.
    """
    covered, reason = covers_offdiag_fused_electric_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = offdiag_fused_electric_pair_tables(pml)
    if scratch is None:
        scratch = offdiag_fused_electric_pair_scratch(fields)
    assert_scratch_is_disjoint(fields, scratch, tables)
    mask = _flat.normalized_row_mask(offdiag_row_mask(fields))
    rows = [volume for volume in offdiag_row_volumes(fields) if volume is not None]
    record = launch_offdiag_fused_electric_pair(
        fields, scratch, tables, offdiag_fused_electric_pair_codes(grid, pml),
        offdiag_wall_mask_flags(grid), _weld.fill_plan(grid), dtdx, mask, rows,
        kernel)
    record["rotated"] = bool(rotate)
    record["scratch"] = rotate_into_fields(fields, scratch) if rotate else scratch
    return record


# The in-seam module is imported for one reason and it is worth naming: this file
# CARRIES ``zero_metal_D``, and a reader checking that claim should find the axis
# reading one attribute away rather than in another directory.
_ZERO_METAL_REFERENCE = zero_metal_axes
