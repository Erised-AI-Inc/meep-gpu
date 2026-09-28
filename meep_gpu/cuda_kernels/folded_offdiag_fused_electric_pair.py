"""The FOLDED stencil weld: PML curl x folded off-diagonal ``update_E``, one launch.

THE CELL, AND WHAT WAS RECORDED ABOUT IT. ``D_to_E (cuda_curl/PML,
cuda_folded_offdiag/folded off-diagonal)`` is the largest STENCIL-BLOCKED cell on the
hand-CUDA board -- 19 corpus rows, 9 of which clear the source injection on the driver
fact alone (``cavity-farfield.py``, ``cyl-ellipsoid.py``, ``holey-wvg-bands.py`` and
their test siblings) -- and it carried the same verdict its unfolded twin did: the
constitutive half reads the curl half's IN-PLACE output at a cell this thread does not
own. :mod:`.offdiag_stencil_weld` explains why the SCRATCH-OUTPUT shape removes both
premises of that verdict, and :mod:`.offdiag_fused_electric_pair` is the unfolded twin
of this module. What is this one's own is the FOLD.

=============================================================================
THE FOLD IS TWO DIFFERENT THINGS AND THE WELD CARRIES BOTH
=============================================================================

A folded axis reaches this seam twice over, through machinery that lives in two
different places, and a weld that conflated them would be smooth and wrong:

1. **the two mirror FILLS** (driver.py:3309, :3311) write the stored ``D`` before
   ``update_E`` runs -- stored cell 0 from stored cell 2 with weight ``+phase``, and
   the top plane of a folded PERIODIC axis from its reflect row with weight
   ``-phase``. Those are the array path's own passes, and this weld carries them
   inside :func:`offdiag_stencil_weld.resolution_source`: the value ``resolve_D``
   returns at ANY cell is already post-fill, post-clear, so both the thread's own
   register and every foreign recompute see what ``update_E`` would have loaded.
2. **``_shift_down``'s MIRROR ARM** is the CONSTITUTIVE's own reading of the same
   fold, one level down: the ghost BELOW stored cell 0 is stored row 2 weighted by
   ``mirror_parity`` (stepping.py:1870-1872). That is not a fill and does not touch
   the stored array; it is the certified ``folded_offdiag_kernels`` body's
   ``coord_dn``/``ghosted_mirror`` pair, and it is lifted UNTOUCHED except for where
   its load comes from.

The two compose exactly, and the composition is what the host probe measured: the
ghost lane redirects to stored row 2, and ``resolve_D`` at stored row 2 is the raw
stepped value there (row 2 is nobody's image), so the ghost weight applies once --
which is the array path's arithmetic on the post-fill array.

=============================================================================
THE TWO BOUNDARY READINGS GENUINELY DIFFER HERE, SO BOTH ARE BOUND
=============================================================================

The unfolded twin binds ONE ``bc_*`` triple and RAISES if the curl's reading and the
constitutive's disagree. On a fold they disagree by design:

* ``step_curl_kernels.real_curl_boundary_codes`` splits the two terminations --
  ``BC_MIRROR_PERIODIC`` for a folded PERIODIC axis (whose stored array carries a slot
  past MEEP's owned window and whose top plane the curl masks) and ``BC_METALLIC`` for
  a folded METALLIC one;
* ``folded_offdiag_kernels.folded_offdiag_boundary_codes`` hands BOTH terminations
  ``BC_MIRROR``, because ``update_E`` has no ownership mask and no reflect row --
  FACT 3 of that module's docstring, measured 144/144 and 96/96.

So this kernel binds ``bc_*`` for the curl and ``cbc_*`` for the constitutive, and the
three lines of the certified constitutive body that read the codes are renamed. That
is the one edit this family makes that its twin does not, and getting it wrong is a
wrong ghost rule on a whole plane rather than a crash.

=============================================================================
WHAT ONE LAUNCH PERFORMS
=============================================================================

:data:`REPLACES` is the whole contiguous D seam: ``step_D`` -> ``fill_symmetry_bc_D``
-> ``zero_metal_D`` -> ``fill_folded_far_ghosts_D`` -> ``update_E`` (driver.py:3302,
:3309, :3310, :3311, :3313). The board's per-pass ledger for this cell is
``{"fill_D": 9, "zero_metal_D": 2}`` -- every reachable row drives the near fill and
two also drive the wall clear -- and the far fill runs on every folded PERIODIC axis.

THE WALL CLEAR AND THE FOLD MEET, AND THE ORDER IS THE DRIVER'S. A D component's near
axes are exactly the axes ``zero_metal_D`` clears it on (the same Yee-shift-0 test),
so a near ghost can land in a cleared plane; the clear runs AFTER the near fill
(driver.py:3309 then :3310) and the resolution applies them in that order. The host
probe's ``clear_order_flipped`` null is armed on the one fixture that can arm it and
moves one word -- which is exactly the size of that question and exactly why it is
measured rather than reasoned about.

=============================================================================
THE DEPOSIT IS REFUSED, AND THAT IS A MEASUREMENT
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is ``False`` for its twin's reason and by the same
measurement: ``deposit_repair.repairable(fields, "D")`` refuses an off-diagonal
chi1inv row by name (deposit_repair.py:686-691), every row of this cell has one
installed by construction, and the FOLD makes the refusal stronger rather than weaker
-- ``deposit_repair.repair_cells`` would have to reconstruct the CLOSURE of a
deposit's mirror images through a stencil. The ceiling is the 9 rows the board already
prices as clearing the injection.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import coverage as _coverage
from . import folded_offdiag_kernels as _folded
from . import offdiag_emitter as _flat
from . import offdiag_stencil_weld as _weld
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import (covers_real_pml_curl, offdiag_row_mask, offdiag_row_volumes,
                       offdiag_wall_mask_flags)
from .in_seam_coverage import zero_metal_axes

from .. import deposit_repair as _deposit_repair

try:
    from . import step_curl_kernels
except Exception:  # noqa: BLE001
    step_curl_kernels = None  # type: ignore[assignment]
try:
    from . import constitutive_kernels
except Exception:  # noqa: BLE001
    constitutive_kernels = None  # type: ignore[assignment]

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
    "folded_offdiag_fused_electric_pair_pml_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS: Dict[str, str] = {}

#: FALSE, and measured off ``deposit_repair.repairable`` rather than chosen -- see the
#: module docstring. The host suite asserts it against the census for THIS cell's rows
#: rather than against this comment.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_folded_offdiag_fused_electric_pair"

KERNEL_NAME = "folded_offdiag_fused_electric_pair_pml_real"

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3302, :3309, :3310, :3311, :3313). Declared rather than inferred from
#: the two slots: three of the five are driver passes no slot names.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

SLOT = "step_D"

_ELECTRIC: Tuple[str, str, str] = ("Ex", "Ey", "Ez")
_D_TARGETS: Tuple[str, str, str] = _weld.D_TARGETS

SCRATCH_VOLUMES: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")

_FUSED_THREADS = 256
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

_DUPLICATE_BC_DEFINES = "\n#define BC_PERIODIC 0\n#define BC_METALLIC 1\n"

#: ``ghosted_mirror``'s two anchors: its signature (which gains the component tag and
#: the pack) and its one load line.
_GHOSTED_MIRROR_SIGNATURE = (
    "__device__ __forceinline__ float ghosted_mirror(const float* g, int index,\n"
    "                                                int mg, float w) {")
_GHOSTED_MIRROR_SIGNATURE_RESOLVED = (
    "__device__ __forceinline__ float ghosted_mirror(int comp, int index,\n"
    "                                                int mg, float w,\n"
    "                                                const WeldArgs& weld) {")
_GHOSTED_MIRROR_LOAD = "    float value = g[index];\n"
_GHOSTED_MIRROR_LOAD_RESOLVED = (
    "    // THE SEAM. The mirror ghost's SOURCE is stored row MIRROR_ROW, and its\n"
    "    // post-pass value is recomputed from pre-launch state rather than loaded\n"
    "    // from a volume this launch is writing. The weight below is applied to the\n"
    "    // result exactly as the certified body applies it -- ON THE GHOST LANE AND\n"
    "    // NOWHERE ELSE.\n"
    "    float value = resolve_D_at(comp, index, weld);\n")

#: ``folded_offdiag_term``'s three anchors.
_TERM_PARAMETERS = (
    "    const float* g, const float* u, int home, int down, int up, int corner,\n"
    "    int mg, float w\n)")
_TERM_PARAMETERS_RESOLVED = (
    "    int comp, float g_home, const float* u, int home, int down, int up,\n"
    "    int corner, int mg, float w, const WeldArgs& weld\n)")
_TERM_NEAR = "    float near_pair = g[home] + ghosted_mirror(g, down, mg, w);\n"
_TERM_NEAR_RESOLVED = (
    "    // THE SEAM. The home sample is the register this thread already resolved\n"
    "    // for its own cell; the DOWN sample is a foreign cell, recomputed and then\n"
    "    // weighted on the ghost lane exactly as before. Same two operands, same\n"
    "    // order, same addition.\n"
    "    float near_pair = g_home + ghosted_mirror(comp, down, mg, w, weld);\n")
_TERM_FAR = (
    "    float far_pair = ghosted(g, up) + ghosted_mirror(g, corner, mg, w);\n")
_TERM_FAR_RESOLVED = (
    "    // THE UP LEG STAYS UNWEIGHTED AND UNREDIRECTED -- it is the component's OWN\n"
    "    // axis, where the array path serves an exact zero on a fold (coord_up) --\n"
    "    // so it takes the plain resolved load, and the corner keeps the ghost lane.\n"
    "    float far_pair = resolved_ghosted(comp, up, weld)"
    " + ghosted_mirror(comp, corner, mg, w, weld);\n")

LIFT_EDITS: Tuple[Dict[str, str], ...] = _weld.CURL_LIFT_EDITS + (
    {"line": "#define BC_PERIODIC 0\n#define BC_METALLIC 1",
     "became": "(removed from folded_offdiag_kernels.prelude())",
     "why": "the curl prelude defines both, identically and first; BC_MIRROR stays, "
            "because only this family's constitutive half has it."},
    {"line": "__device__ __forceinline__ float ghosted_mirror(const float* g, "
             "int index,\n                                                int mg, "
             "float w) {",
     "became": "(int comp, int index, int mg, float w, const WeldArgs& weld)",
     "why": "the partner volume becomes a component tag and the pack arrives, so the "
            "ghost lane's SOURCE can be recomputed instead of loaded."},
    {"line": "    float value = g[index];",
     "became": "    float value = resolve_D_at(comp, index, weld);",
     "why": "THE SEAM on the ghost lane. The index < 0 metallic-zero arm above and "
            "the `mg ? (w * value) : value` weight below are untouched, so the "
            "unfolded reduction stays the structural claim the certified family's "
            "own gate measured (32/32)."},
    {"line": "    const float* g, const float* u, int home, int down, int up, "
             "int corner,\n    int mg, float w\n)",
     "became": "    int comp, float g_home, const float* u, int home, int down, "
               "int up,\n    int corner, int mg, float w, const WeldArgs& weld\n)",
     "why": "the partner volume becomes a COMPONENT TAG and the home sample a "
            "register -- the value this thread already resolved for its own cell, "
            "which is what the certified body's g[home] would have loaded."},
    {"line": "    float near_pair = g[home] + ghosted_mirror(g, down, mg, w);",
     "became": "    float near_pair = g_home + ghosted_mirror(comp, down, mg, w, "
               "weld);",
     "why": "the same addition of the same two samples; what changes is that the "
            "foreign one is RECOMPUTED. The ghost lane and its weight are unchanged."},
    {"line": "    float far_pair = ghosted(g, up) + ghosted_mirror(g, corner, mg, w);",
     "became": "    float far_pair = resolved_ghosted(comp, up, weld) + "
               "ghosted_mirror(comp, corner, mg, w, weld);",
     "why": "the same, for the term's other two foreign samples, keeping the "
            "asymmetry the certified body has: the UP leg is the component's own "
            "axis, unweighted and unredirected. The 0.25f scale, the coefficient "
            "multiply BETWEEN the two shifts and the two accumulations are untouched "
            "-- the return line is verbatim."},
    {"line": "        flat(...),\n        mg_*, gw_*);   (each term's closing line)",
     "became": "        mg_*, gw_*, weld);",
     "why": "the resolved helper needs the argument pack. Found STRUCTURALLY -- the "
            "term's opening line, then the first line below it that closes the "
            "statement -- rather than by matching an index expression that differs "
            "between row masks."},
    {"line": "    float gs_E* = D*[idx];",
     "became": "    float gs_E* = v_*;",
     "why": "THE SEAM, diagonal half. The register is the post-fill, post-clear "
            "value the array path would have loaded."},
    {"line": "        D*, chi1inv_E*_E*, idx,",
     "became": "        <tag>, v_*, chi1inv_E*_E*, idx,",
     "why": "the call-site half of the term rewrite. The coefficient argument, the "
            "home index, the three shifted indices and the ghost lane/weight pair "
            "are untouched."},
    {"line": "    int di = coord_dn(i, nx, bc_x), ui = coord_up(i, nx, bc_x);",
     "became": "    int di = coord_dn(i, nx, cbc_x), ui = coord_up(i, nx, cbc_x);",
     "why": "THE ONE EDIT THE UNFOLDED TWIN DOES NOT MAKE. The curl and the "
            "constitutive read a FOLDED axis differently -- the curl splits the two "
            "terminations (BC_MIRROR_PERIODIC / BC_METALLIC) because only one stores "
            "a slot past the owned window, while update_E has no ownership mask and "
            "serves both with BC_MIRROR -- so this kernel binds two triples and the "
            "constitutive body reads its own. Letting one shadow the other is a "
            "wrong ghost rule on a whole plane, not a compile failure. Three lines, "
            "one per axis."},
    {"line": "    int mg_x = (bc_x == BC_MIRROR) && at_x;",
     "became": "    int mg_x = (cbc_x == BC_MIRROR) && at_x;",
     "why": "the same rename on the ghost-lane predicate, which the certified body "
            "DERIVES FROM bc_* IN ONE PLACE precisely so the index redirect and the "
            "weight cannot disagree about which lane is the ghost. Renaming one and "
            "not the other would break exactly that property. Three lines."},
    {"line": "    constitutive_apply(E*, f_w_E*, idx, src_E*, kps_*[*], kms_*[*]);",
     "became": "    constitutive_apply(E*, f_w_E*, idx, src_E*, kps_*[*], "
               "kms_half_*[*]);",
     "why": "ONE RENAME, NO ARITHMETIC -- the integer/half-integer sub-lattice "
            "collision, half a cell in the absorber profile."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= nx * ny * nz) return;   (the constitutive TEMPLATE)",
     "became": "(the weld's own guarded preamble, once)",
     "why": "the weld emits the guard and the decode once for both halves; splicing "
            "both would redeclare idx, i, j and k."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SCRATCH_VOLUMES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_scratch_is_disjoint", "constitutive_body",
    "covers_folded_offdiag_fused_electric_pair", "device_sources",
    "folded_offdiag_fused_electric_pair_scratch",
    "folded_offdiag_fused_electric_pair_tables", "kernel_source", "prelude",
    "launch_folded_offdiag_fused_electric_pair", "rotate_into_fields",
    "run_folded_offdiag_fused_electric_pair", "signature",
]


# =============================================================================
# THE DEVICE CODE
# =============================================================================

def prelude() -> str:
    """Both certified preludes plus the weld's own helpers, in dependency order."""
    constitutive = _folded.prelude()
    anchors = ((_DUPLICATE_BC_DEFINES, "the duplicate boundary defines"),
               (_GHOSTED_MIRROR_SIGNATURE, "ghosted_mirror's signature"),
               (_GHOSTED_MIRROR_LOAD, "ghosted_mirror's load"),
               (_TERM_PARAMETERS, "folded_offdiag_term's parameter list"),
               (_TERM_NEAR, "folded_offdiag_term's near-pair line"),
               (_TERM_FAR, "folded_offdiag_term's far-pair line"))
    for anchor, what in anchors:
        if constitutive.count(anchor) != 1:
            raise AssertionError(
                f"folded_offdiag_kernels.prelude() carries "
                f"{constitutive.count(anchor)} copies of {what}, not one; this weld "
                f"LIFTS that text rather than retyping it and cannot splice around "
                f"its absence")
    constitutive = constitutive.replace(_DUPLICATE_BC_DEFINES, "\n", 1)
    constitutive = constitutive.replace(
        _GHOSTED_MIRROR_SIGNATURE, _GHOSTED_MIRROR_SIGNATURE_RESOLVED, 1)
    constitutive = constitutive.replace(
        _GHOSTED_MIRROR_LOAD, _GHOSTED_MIRROR_LOAD_RESOLVED, 1)
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
    """The kernel's parameter list, with only the LIVE row coefficients bound."""
    mask = _flat.normalized_row_mask(row_mask)
    rows = "\n".join(f"    const float* {_flat.ROW_PARAMETERS[slot]},"
                     for slot, flag in enumerate(mask) if flag)
    return f'''
extern "C" __global__ void {KERNEL_NAME}(
    // THE SCRATCH OUTPUTS. A DIFFERENT ALLOCATION from the D/fu group below --
    // assert_scratch_is_disjoint checks it by base address before every launch --
    // and that separation is the whole design: nothing this launch reads is ever a
    // word this launch wrote, so every foreign sample is a pure function of
    // unwritten memory rather than a race with another block.
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
    // E and its constitutive history: written at the thread's OWN cell only.
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    // NOT __restrict__: an isotropic run hands the same device pointer three times
    // (fields.py:1321-1326).
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    // NOT __restrict__ either: a row volume may legally alias another row's, an
    // epsilon volume or a D volume.
{rows}
    int nx, int ny, int nz, float dtdx,
    // The D curl's INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // update_E's HALF-INTEGER pair, its kms renamed: the two sub-lattices collide on
    // the bare name and a shadow there is half a cell in the absorber profile.
    const float* __restrict__ kps_x, const float* __restrict__ kms_half_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_half_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_half_z,
    // THE CURL's boundary codes (step_curl_kernels.real_curl_boundary_codes), which
    // SPLIT the two folded terminations: only a folded PERIODIC axis stores a slot
    // past MEEP's owned window, and only its top plane is masked.
    int bc_x, int bc_y, int bc_z,
    // THE CONSTITUTIVE's boundary codes (folded_offdiag_boundary_codes), which serve
    // BOTH folded terminations with one BC_MIRROR: update_E has no ownership mask
    // and no reflect row. A DIFFERENT reading of the same grid, and the one edit
    // this family makes that its unfolded twin does not.
    int cbc_x, int cbc_y, int cbc_z,
    // The COUPLING's wall mask (coverage.offdiag_wall_mask_flags). Zero on every
    // folded axis by construction: _mask_metallic_wall_coupling asks is_metallic AND
    // NOT is_mirrored (stepping.py:1253).
    int wm_x, int wm_y, int wm_z,
    // The mirror ghost's parity per axis (folded_offdiag_kernels.mirror_ghost_weights)
    // -- _shift_down's own weight, applied ON THE GHOST LANE AND NOWHERE ELSE.
    float gw_x, float gw_y, float gw_z,
    // zero_metal_D's walls (in_seam_coverage.zero_metal_axes). A DIFFERENT question
    // from wm_*: that masks the coupling TOTAL, this clears the stored D.
    int wall_x, int wall_y, int wall_z,
    // The two mirror FILLS' plan -- the other half of the fold, and the half that
    // touches the stored array. near_a is 1 on a MIRROR-folded axis of EITHER
    // termination; reflect_a is the far fill's image row on a folded PERIODIC axis
    // and -1 where that pass does not run, a SENTINEL so a source that read it
    // anyway would index outside the volume; the two phase triples are
    // in_seam_coverage.component_parity's own answers for the shift-0 and shift-1
    // components, so the kernel performs no negation.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float near_phase_x, float near_phase_y, float near_phase_z,
    float far_phase_x, float far_phase_y, float far_phase_z
) {{
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1253->1282


def constitutive_body(row_mask: Sequence[int]) -> str:
    """``update_E_pml_real_folded_offdiag``'s body, lifted, with this family's edits."""
    mask = _flat.normalized_row_mask(row_mask)
    body = _weld.split_body(_folded.folded_offdiag_source(mask), _folded.prelude(),
                            f"folded_offdiag_source({tuple(mask)})")
    for anchor in ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                   "    if (idx >= nx * ny * nz) return;\n",
                   "    int nyz = ny * nz;\n    int k = idx % nz;\n"
                   "    int j = (idx / nz) % ny;\n    int i = idx / (ny * nz);\n"):
        if body.count(anchor) != 1:
            raise AssertionError(
                f"the certified folded body carries {body.count(anchor)} copies of "
                f"its thread preamble block, not one; the weld emits that block once")
        body = body.replace(anchor, "", 1)

    # (1) THE CONSTITUTIVE'S OWN BOUNDARY CODES. Both the neighbour-coordinate line
    #     and the ghost-lane predicate, per axis, so the redirect and the weight keep
    #     reading ONE triple -- the property the certified body's own comment names.
    for axis, letter in enumerate("xyz"):
        coordinate = "ijk"[axis]
        extent = ("nx", "ny", "nz")[axis]
        down, up = ("di", "dj", "dk")[axis], ("ui", "uj", "uk")[axis]
        line = _weld.the_line_starting(
            body, f"    int {down} = coord_dn({coordinate}, {extent}, bc_{letter}),",
            f"axis {letter}'s neighbour-coordinate line")
        wanted = (f"    int {down} = coord_dn({coordinate}, {extent}, bc_{letter}), "
                  f"{up} = coord_up({coordinate}, {extent}, bc_{letter});")
        if line != wanted:
            raise AssertionError(
                f"the certified folded body spells axis {letter}'s neighbour "
                f"coordinates {line!r}; this weld renames only the boundary code and "
                f"cannot do so on a line whose shape has moved")
        body = body.replace(line, wanted.replace(f"bc_{letter}", f"cbc_{letter}"), 1)
        ghost = _weld.the_line_starting(
            body, f"    int mg_{letter} = (bc_{letter} == BC_MIRROR)",
            f"axis {letter}'s ghost-lane predicate")
        body = body.replace(
            ghost, ghost.replace(f"(bc_{letter} ==", f"(cbc_{letter} ==", 1), 1)

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

    # (3) THE SEAM, foreign half: the partner argument and the pack, per live term.
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
            line, f"        {partner}, v_{'xyz'[partner]}, {coefficient}, idx,", 1)
        body = _weld.close_call_with_the_pack(
            body,
            f"    float term_{_ELECTRIC[component]}_{offset} = folded_offdiag_term(",
            f"the {coefficient} term")

    # (4) THE SUB-LATTICE RENAME, matched as a WHOLE LINE.
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
                f"the certified folded body spells {name}'s constitutive tail "
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
        "    // value at this cell because NO in-seam pass touches it -- neither fill",
        "    // writes fu_field (stepping.py:1451, :1533) and _zero_metal writes only",  # stepping.py live lines for the frozen device-text citation(s) in this string: 1533->1580
        "    // the named components (:2247).",
        "    float d_raw[3];",
        "    float fu_raw[3];",
        "    raw_step_D_cell(i, j, k, weld, d_raw, fu_raw);",
        "    fu_Dx_out[idx] = fu_raw[0];",
        "    fu_Dy_out[idx] = fu_raw[1];",
        "    fu_Dz_out[idx] = fu_raw[2];",
        "",
        "    // ALL THREE IN-SEAM PASSES, resolved for this thread's own cell: the",
        "    // near fill, the wall clear and the far fill, in the driver's order.",
        "    // The same function the constitutive half calls for a foreign cell, so",
        "    // the own-cell value and a neighbour's are the same arithmetic by",
        "    // construction rather than by a second transcription.",
        "    const float v_x = resolve_Dx(i, j, k, weld);",
        "    const float v_y = resolve_Dy(i, j, k, weld);",
        "    const float v_z = resolve_Dz(i, j, k, weld);",
        "    Dx_out[idx] = v_x;",
        "    Dy_out[idx] = v_y;",
        "    Dz_out[idx] = v_z;",
        "    // The certified folded off-diagonal body follows, from its own",
        "    // neighbour-coordinate line down, with its boundary codes renamed to",
        "    // the constitutive triple and its D reads routed through the",
        "    // resolution. Everything else is lifted, not retyped.",
    ]
    return (prelude() + signature(mask) + "\n".join(preamble)
            + constitutive_body(mask) + "}\n")


def device_sources(row_mask: Sequence[int] = (1, 1, 1, 1, 1, 1)) -> Dict[str, str]:
    return {KERNEL_NAME: kernel_source(row_mask)}


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered."""
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    for mask in _flat.LIVE_ROW_MASKS:
        digest.update(repr(mask).encode("ascii"))
        digest.update(kernel_source(mask).encode("utf-8"))
    return digest.hexdigest()


def _get_kernel(row_mask: Sequence[int], source: Optional[str] = None):
    """Compile on first use, memoized on (name, options, policy, source)."""
    import cupy as cp  # noqa: PLC0415 - device-only

    code = kernel_source(row_mask) if source is None else source
    key = _kernel_cache_key(KERNEL_NAME, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# =============================================================================
# THE PREDICATE
# =============================================================================

def covers_folded_offdiag_fused_electric_pair(fields: Any, pml: Any, grid: Any,
                                              sources: Any = None
                                              ) -> Tuple[bool, str]:
    """May ONE launch span the whole folded D seam here?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. The curl half's own certified predicate
    and the constitutive arm's COMPOSITION predicate -- ``covers_folded_offdiag_
    composition``, which REQUIRES a real fold and is what makes this family disjoint
    from the unfolded twin on every corpus row -- are both asked in full. What this
    predicate ADDS is the seam clauses: the fill plan must be buildable, the source
    slot must be empty, and the three stored extents must be one tuple.
    """
    covered, reason = covers_real_pml_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _folded.covers_folded_offdiag_composition(fields, pml, grid)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE FILL PLAN MUST BE BUILDABLE, and its three refusals are this weld's, not
    # inherited: an unreadable plane phase would launch a NaN weight, an axis reported
    # both folded and walled would put the wall clear on a plane MEEP steps, and a far
    # reflect row without a phase is a pass that cannot run. Each is a plane of wrong
    # values rather than a crash, which is why the plan RAISES and this catches.
    try:
        plan = _weld.fill_plan(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"the in-seam fill plan: {type(exc).__name__}: {exc}")
    if not any(plan["near"]):
        return False, ("no axis carries a mirror fill: this product is the FOLDED "
                       "weld and offdiag_fused_electric_pair serves the unfolded "
                       "cell -- a fold-free grid admitted here would put two "
                       "products on one row")

    # THE CURL'S CODES MUST BE BUILDABLE TOO. The two readings differ on a fold BY
    # DESIGN and both are bound, so there is nothing to cross-check -- but a grid the
    # curl's own resolver refuses is a grid this launch cannot bind.
    try:
        folded_offdiag_fused_electric_pair_codes(grid, pml)
    except Exception as exc:  # noqa: BLE001
        return False, (f"the two boundary readings this launch binds: "
                       f"{type(exc).__name__}: {exc}")

    # THE SOURCE SEAM, REFUSED RATHER THAN CARRIED -- see CARRIES_DEPOSIT_REPAIR.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, "D",
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3305/:3308) and this "
            f"pair declares no deposit repair -- deposit_repair.repairable refuses an "
            f"off-diagonal chi1inv row, which every row of this cell carries, and on "
            f"a folded grid the repair would additionally have to reconstruct the "
            f"deposit's mirror images through a stencil"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

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
    for name in ("fu_Dx", "fu_Dy", "fu_Dz"):
        volume = getattr(fields, name, None)
        if volume is None:
            return False, (f"{name} is not allocated: this pair writes the "
                           f"split-field recurrence into a scratch shaped like it")
        if tuple(int(n) for n in getattr(volume, "shape", ())) != curl_extents:
            return False, (f"{name} has shape {tuple(volume.shape)} against Dx's "
                           f"{curl_extents}; the scratch this pair rotates in is "
                           f"shaped from it")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

def folded_offdiag_fused_electric_pair_tables(pml: Any) -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads."""
    if step_curl_kernels is None or constitutive_kernels is None:
        raise RuntimeError(
            "step_curl_kernels/constitutive_kernels are not importable on this host, "
            "so the coefficient tables cannot be resolved. The predicate needs "
            "neither and still answers")
    return {
        "curl": step_curl_kernels.real_pml_curl_tables(pml, False),
        "constitutive": _folded.folded_offdiag_constitutive_tables(pml),
    }


def folded_offdiag_fused_electric_pair_codes(grid: Any, pml: Any = None
                                             ) -> Dict[str, Tuple[int, ...]]:
    """BOTH boundary readings, kept apart. THEY DIFFER ON A FOLD AND THAT IS THE POINT.

    Returned as a mapping rather than two positional tuples so a caller cannot bind
    them the wrong way round: on a folded METALLIC axis the curl says ``BC_METALLIC``
    and the constitutive says ``BC_MIRROR``, and swapping them is a wrong ghost rule
    on a whole plane rather than a crash.
    """
    # BOTH READINGS COME FROM STDLIB-ONLY SIBLINGS, and that is a predicate
    # requirement rather than a convenience: the census asks this clause on a laptop
    # with no CuPy. ``coverage.real_curl_boundary_codes`` FAILS CLOSED, returning
    # ``(codes, refusal)`` with exactly one None, and the refusal is carried rather
    # than swallowed -- a grid whose fold termination the curl cannot resolve is a
    # grid this launch must not bind.
    codes, refusal = _coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(
            f"the curl's own boundary resolution refuses this grid: {refusal}")
    return {
        "curl": tuple(int(code) for code in codes),
        "constitutive": tuple(int(code) for code in
                              _folded.folded_offdiag_boundary_codes(grid, pml)),
    }


def folded_offdiag_fused_electric_pair_scratch(fields: Any) -> Dict[str, Any]:
    """The six launch-local volumes, allocated once per frozen configuration.

    NOT ``StepScratch`` -- that pool is for values consumed within the call that
    produced them (fields.py:298-327), and the rotation hands these to the driver as
    the live ``D``/``fu_D``. Uninitialized is correct: every cell of all six is
    written by every launch.
    """
    import cupy as cp  # noqa: PLC0415 - device-only

    volumes: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        source = getattr(fields, name, None)
        if source is None:
            raise ValueError(
                f"fields.{name} is not allocated; this pair rotates a scratch shaped "
                f"like it and has nothing to shape one from")
        volumes[name] = cp.empty_like(source)
    return volumes


def assert_scratch_is_disjoint(fields: Any, scratch: Dict[str, Any],
                               tables: Dict[str, Dict[str, Any]]) -> int:
    """No scratch volume is a bound input, and no two restrict arguments alias.

    THE FIRST CHECK IS THE DESIGN, not a convention: if ``Dx_out`` were ``Dx`` the
    launch would be the IN-PLACE weld the board refused, and every foreign recompute
    would read words other blocks had already overwritten. The fold makes it sharper
    -- one of the redirect sources is stored row 2, deep INSIDE the volume rather than
    at a face.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
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
    for key in ("curl", "constitutive"):
        for table, volume in sorted(tables[key].items()):
            visit(f"{key}:{table}", volume)
    if collisions:
        raise ValueError(
            "this pair binds every field, scratch and table argument __restrict__, "
            "and these arguments alias -- undefined behaviour NVRTC miscompiles "
            "silently, and for the scratch group the in-place weld the board "
            "refused: " + "; ".join(collisions))
    for component in _ELECTRIC:
        pointer = int(fields.inverse_epsilon_for(component).data.ptr)
        if pointer in bound:
            collisions.append(
                f"inv_eps_{component} and {bound[pointer]} are the same allocation, "
                f"and the second is bound __restrict__")
    if collisions:
        raise ValueError(
            "an inverse-permittivity volume aliases a __restrict__ argument: "
            + "; ".join(collisions))
    return len(bound)


def _validate_launch_arguments(shape: Sequence[int],
                               codes: Dict[str, Tuple[int, ...]],
                               wall_mask: Sequence[int],
                               weights: Sequence[float],
                               fills: Dict[str, Tuple[Any, ...]]) -> None:
    """Everything a launch would otherwise get silently WRONG rather than crash.

    The certified constitutive family's three checks (a NaN ghost weight, a folded
    axis that is also wall-masked, a folded axis too thin for the mirror row), plus
    the two this weld adds: the mirror ghost weight and the FAR fill parity are the
    same number by two routes (``mirror_ghost_weights`` asks
    ``mirror_parity("D"+axis, axis, phase)`` and the fill plan asks
    ``component_parity`` for the shift-1 component of that axis -- both are
    ``-phase``), and a folded axis must carry a near flag.
    """
    constitutive = codes["constitutive"]
    if len(constitutive) != 3 or any(
            int(code) not in _folded.FOLDED_BC_CODES.values()
            for code in constitutive):
        raise ValueError(
            f"the constitutive boundary codes must be three of "
            f"{sorted(set(_folded.FOLDED_BC_CODES.values()))}, got {constitutive!r}")
    for axis in range(3):
        if int(constitutive[axis]) != _folded.BC_MIRROR_CODE:
            continue
        if int(wall_mask[axis]):
            raise ValueError(
                f"axis {axis} is folded AND wall-masked; "
                f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                f"(stepping.py:1253)")  # stepping.py live lines for the frozen device-text citation(s) in this string: 1253->1282
        if float(weights[axis]) not in (1.0, -1.0):
            raise ValueError(
                f"axis {axis} is folded with ghost weight {weights[axis]!r}; the "
                f"mirror ghost carries mirror_parity('D'+axis, axis, phase) == "
                f"-phase, exactly +1.0 or -1.0")
        if int(shape[axis]) <= _folded.MIRROR_SOURCE_INDEX:
            raise ValueError(
                f"axis {axis} is folded with {shape[axis]} stored cells; the mirror "
                f"ghost images stored row {_folded.MIRROR_SOURCE_INDEX}")
        if not int(fills["near"][axis]):
            raise ValueError(
                f"axis {axis} reads BC_MIRROR to the constitutive half and carries "
                f"no near fill; fill_symmetry_bc_D writes stored cell 0 on every "
                f"mirrored axis (stepping.py:1484-1485) and the two readings of the "
                f"fold have diverged")
        if float(fills["far_phase"][axis]) != float(weights[axis]):
            raise ValueError(
                f"axis {axis}'s mirror ghost weight is {weights[axis]!r} and its far "
                f"fill parity {fills['far_phase'][axis]!r}; both are "
                f"mirror_parity of that axis's own shift-1 D component and must be "
                f"one number by two routes")


def launch_folded_offdiag_fused_electric_pair(
        fields: Any, scratch: Dict[str, Any], tables: Dict[str, Dict[str, Any]],
        codes: Dict[str, Tuple[int, ...]], wall_mask: Sequence[int],
        weights: Sequence[float], fills: Dict[str, Tuple[Any, ...]], dtdx: float,
        row_mask: Sequence[int], rows: Sequence[Any],
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch. THE CALLER ROTATES AFTERWARDS.

    ``kernel`` IS THE GATE'S DOOR: a gate compiles a deliberately broken copy of the
    shipped source and hands it here. A launcher that could not be handed its own
    kernel could not arm a single mutation.
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
    arguments += [np.int32(int(code)) for code in codes["curl"]]
    arguments += [np.int32(int(code)) for code in codes["constitutive"]]
    arguments += [np.int32(int(bool(flag))) for flag in wall_mask]
    arguments += [np.float32(float(value)) for value in weights]
    arguments += [np.int32(int(flag)) for flag in fills["wall"]]
    arguments += [np.int32(int(flag)) for flag in fills["near"]]
    arguments += [np.int32(int(row)) for row in fills["reflect"]]
    arguments += [np.float32(float(value)) for value in fills["near_phase"]]
    arguments += [np.float32(float(value)) for value in fills["far_phase"]]
    (kernel or _get_kernel(row_mask))((blocks,), (_FUSED_THREADS,), tuple(arguments))
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "row_mask": tuple(int(flag) for flag in row_mask),
            "boundary_codes": {key: tuple(value) for key, value in codes.items()},
            "coupling_wall_mask": tuple(int(bool(flag)) for flag in wall_mask),
            "ghost_weights": tuple(float(value) for value in weights),
            "fills": {key: tuple(fills[key]) for key in
                      ("near", "wall", "reflect", "near_phase", "far_phase")}}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION -- the certified E->P choreography, one seam earlier.

    See :func:`offdiag_fused_electric_pair.rotate_into_fields` for the consumer audit
    this rests on and the test that measures it.
    """
    retired: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        retired[name] = getattr(fields, name)
        setattr(fields, name, scratch[name])
    return retired


def run_folded_offdiag_fused_electric_pair(
        fields: Any, grid: Any, pml: Any, dtdx: float, *,
        scratch: Optional[Dict[str, Any]] = None, sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None, rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once, rotate."""
    covered, reason = covers_folded_offdiag_fused_electric_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = folded_offdiag_fused_electric_pair_tables(pml)
    if scratch is None:
        scratch = folded_offdiag_fused_electric_pair_scratch(fields)
    assert_scratch_is_disjoint(fields, scratch, tables)
    codes = folded_offdiag_fused_electric_pair_codes(grid, pml)
    wall_mask = offdiag_wall_mask_flags(grid)
    weights = _folded.mirror_ghost_weights(grid)
    fills = _weld.fill_plan(grid)
    _validate_launch_arguments(tuple(int(n) for n in fields.Dx.shape), codes,
                               wall_mask, weights, fills)
    mask = _flat.normalized_row_mask(offdiag_row_mask(fields))
    rows = [volume for volume in offdiag_row_volumes(fields) if volume is not None]
    record = launch_folded_offdiag_fused_electric_pair(
        fields, scratch, tables, codes, wall_mask, weights, fills, dtdx, mask, rows,
        kernel)
    record["rotated"] = bool(rotate)
    record["scratch"] = rotate_into_fields(fields, scratch) if rotate else scratch
    return record


_ZERO_METAL_REFERENCE = zero_metal_axes
