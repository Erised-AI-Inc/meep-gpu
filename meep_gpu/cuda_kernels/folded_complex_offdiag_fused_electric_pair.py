"""The FOLDED COMPLEX stencil weld: folded complex PML curl x off-diagonal ``update_E``.

THE CELL, AND WHAT WAS RECORDED ABOUT IT. ``D_to_E (cuda_complex_folded/folded
complex, cuda_complex_offdiag/complex off-diagonal PML)`` carries 3 corpus rows --
``examples:solve-cw.py``, ``tests:TestArrayMetadata.test_array_metadata`` and
``tests:TestHoleyWvgBands.test_fields_at_kx`` -- of which ONE clears the source
injection on the driver fact alone. It is one of the two cells the hand-CUDA board
still recorded STENCIL-BLOCKED after the real-storage welds landed, and
:mod:`.complex_offdiag_stencil_weld` is where the argument that the verdict does not
reach a SCRATCH-OUTPUT weld lives. This module is that shape's folded complex arm.

WHAT IS THIS FAMILY'S OWN, over and above the shared lift:

* **the two boundary readings genuinely differ and both are bound.** The curl splits
  the fold's two terminations (``BC_MIRROR_PERIODIC`` for a folded PERIODIC axis,
  whose stored array carries a slot past MEEP's owned window and whose top plane the
  curl masks; ``BC_METALLIC`` for a folded METALLIC one), while ``update_E`` has no
  ownership mask and no reflect row and serves BOTH with ``BC_MIRROR``. So this
  kernel binds ``bc_*`` for the curl and ``cbc_*`` for the constitutive, and the
  shared lift renames the constitutive's three reads. Letting one shadow the other is
  a wrong ghost rule on a whole plane rather than a compile failure. It is the same
  edit :mod:`.folded_offdiag_fused_electric_pair` makes one storage width over;
* **the fold reaches this seam twice over**, exactly as it does on the real weld: the
  two mirror FILLS write the stored ``D`` before ``update_E`` runs, and are carried
  inside :func:`complex_offdiag_stencil_weld.resolution_source`; and
  ``_shift_down``'s MIRROR ARM is the CONSTITUTIVE's own reading of the same fold one
  level down, lifted untouched except for where its load comes from. The two compose:
  the ghost lane redirects to stored row ``MIRROR_ROW``, ``resolve_D`` there is the
  raw stepped value (row 2 is nobody's image), so the ghost weight applies once --
  which is the array path's arithmetic on the post-fill array;
* **the parity multiplies are FULL COMPLEX PRODUCTS at BOTH signs.** That is the
  shared lift's measurement and its module docstring is where it is derived; what
  matters here is that this family is the one whose corpus rows actually fold, so it
  is the family those multiplies fire on.

=============================================================================
WHAT ONE LAUNCH PERFORMS
=============================================================================

:data:`REPLACES` is the whole contiguous D seam: ``step_D`` -> ``fill_symmetry_bc_D``
-> ``zero_metal_D`` -> ``fill_folded_far_ghosts_D`` -> ``update_E`` (driver.py:3317,
:3326, :3327, :3330, :3332). The curl half writes ``D_new``/``fu_new`` to a
LAUNCH-LOCAL SCRATCH, the constitutive half re-derives every foreign tap from
PRE-LAUNCH state through the certified curl body lifted inline, and
:func:`rotate_into_fields` rotates the ``D``/``fu_D`` bindings only AFTER the launch
returns. No thread reads a word the launch wrote.

=============================================================================
THE DEPOSIT IS REFUSED, AND THAT IS A MEASUREMENT
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is ``False`` for the real off-diagonal welds' reason
and by the same measurement: ``deposit_repair.repairable(fields, "D")`` refuses an
off-diagonal chi1inv row BY NAME (a point repair recomputes E at the deposit cell
from THAT cell's displacement, while this constitutive reads its NEIGHBOURS'), every
row of this cell has one installed by construction, and the fold makes the refusal
stronger rather than weaker. The ceiling is therefore the rows that clear the
injection on the driver fact alone -- 1 of this cell's 3.

=============================================================================
NOTHING HERE IS DISPATCH
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch; this
family is opt-in through ``arms.plan_step(..., fuse=True)`` like every other product
on this board.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import complex_emitter
from . import complex_folded_kernels as _folded
from . import complex_offdiag_stencil_weld as _weld
from . import complex_offdiag_update_e as _offdiag
from . import coverage as _coverage
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import offdiag_row_mask, offdiag_row_volumes, offdiag_wall_mask_flags
from .in_seam_coverage import zero_metal_axes

from .. import deposit_repair as _deposit_repair

try:  # device-only: the tables and the word view
    from .complex_pml_kernels import (bloch_phase_arguments, complex_curl_tables,
                                      word_view)
except ImportError:  # a host with no CuPy: only the launch is unavailable
    bloch_phase_arguments = complex_curl_tables = word_view = None


# =============================================================================
# THE PARTITION -- EMPTY ON THE CERTIFIED SIDE UNTIL A GATE MOVES IT
# =============================================================================

#: Byte-identical with a gate verdict AND a record behind it. RECORDED 2026-09-02 as
#: ``cuda_complex_offdiag_stencil_welds_2026-09-02`` in ``certification.json``. Both
#: legs RELEASED under BOTH float32 subnormal policies on the GPU host (NVIDIA RTX
#: A6000): 132 S1 cases bit-identical per COMPLETE driver step over 60 steps x 4
#: repeats x 3 value classes, 66 S2 cases identical at every block size from one warp
#: to 1024, 11 S3 cases identical at the PLANTED D SEAM -- the leg that carries the
#: adversarial signed-zero catalogue a complete step destroys before the seam is
#: reached -- and a 14-arm mutation battery two-sided, including the three
#: complex-specific arms and a confirmed null.
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "fused_electric_pair_pml_complex_folded_offdiag",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument. The
#: arithmetic was proved off-device FIRST, against the driver's own pass order, by
#: ``parity/meep_gpu/probe_cuda_complex_offdiag_scratch_weld.py``.
UNCERTIFIED_KERNELS: Dict[str, str] = {}

#: FALSE, and MEASURED off ``deposit_repair.repairable`` rather than chosen -- see
#: the module docstring. The host suite asserts it against the census for THIS
#: cell's rows rather than against this comment.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_folded_complex_offdiag_fused_electric_pair"

#: The ``extern "C"`` symbol NVRTC is asked for, spelled once.
KERNEL_NAME = "fused_electric_pair_pml_complex_folded_offdiag"

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3317, :3326, :3327, :3330, :3332). Declared rather than inferred from
#: the two slots: three of the five are driver passes no slot names.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: The six launch-local COMPLEX volumes the curl half writes and the launcher
#: rotates in afterwards. Named for the ``fields`` attributes they shadow.
SCRATCH_VOLUMES: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")

_ELECTRIC: Tuple[str, str, str] = ("Ex", "Ey", "Ez")

#: The curl arm this family's half of the shared lift takes.
ARM = "pml"

_FUSED_THREADS = 256
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate and the host suite can assert the list against
#: the emitted source rather than against a docstring. The curl half's entries and
#: the constitutive half's are the SHARED lift's own declarations -- imported by
#: value, so a family that drifted from the lift it calls would be caught here.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    _weld.curl_lift_edits(ARM) + _weld.RESOLUTION_LIFT_EDITS + (
        {"line": "    int di = coord_dn(i, nx, bc_x), ui = coord_up(i, nx, bc_x);",
         "became": "    int di = coord_dn(i, nx, cbc_x), ui = coord_up(i, nx, "
                   "cbc_x);",
         "why": "THE ONE EDIT THE UNFOLDED TWIN DOES NOT MAKE. The curl and the "
                "constitutive read a FOLDED axis differently -- the curl splits the "
                "two terminations (BC_MIRROR_PERIODIC / BC_METALLIC) because only "
                "one stores a slot past the owned window, while update_E has no "
                "ownership mask and serves both with BC_MIRROR -- so this kernel "
                "binds two triples and the constitutive body reads its own. Every "
                "read of the triple moves with it, which is what keeps the index "
                "redirect and the ghost-lane weight reading ONE code."},
        {"line": "    constitutive_apply(E*, f_w_E*, idx, src_E*, kps_*[*], "
                 "kms_*[*]);",
         "became": "    constitutive_apply(E*, f_w_E*, idx, src_E*, kps_*[*], "
                   "kms_half_*[*]);",
         "why": "ONE RENAME, NO ARITHMETIC. kms_* is the INTEGER sub-lattice for "
                "the D curl and the HALF-INTEGER one for update_E; letting one "
                "shadow the other in a single scope is a half-cell error in the "
                "absorber profile rather than a compile failure."},
        {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                 "    if (idx >= nx * ny * nz) return;   (BOTH certified bodies)",
         "became": "(the weld's own guarded preamble, once)",
         "why": "the weld emits the guard and the decode once for both halves; "
                "splicing both would redeclare idx, i, j, k and nyz."},
    ))

__all__ = [
    "ARM", "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SCRATCH_VOLUMES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_scratch_is_disjoint", "constitutive_body", "corpus_digest",
    "covers_folded_complex_offdiag_fused_electric_pair", "device_sources",
    "folded_complex_offdiag_fused_electric_pair_codes",
    "folded_complex_offdiag_fused_electric_pair_phases",
    "folded_complex_offdiag_fused_electric_pair_scratch",
    "folded_complex_offdiag_fused_electric_pair_tables", "kernel_source",
    "launch_folded_complex_offdiag_fused_electric_pair", "prelude",
    "rotate_into_fields", "run_folded_complex_offdiag_fused_electric_pair",
    "signature",
]


# =============================================================================
# THE DEVICE CODE
# =============================================================================

def prelude(expansion: Any) -> str:
    """Both certified preludes plus the weld's own helpers, in dependency order.

    THE ORDER IS FORCED AND IS NOT A STYLE CHOICE. ``resolved_down_sample`` reads
    ``BC_MIRROR``, which the constitutive prelude defines; ``offdiag_term`` calls
    ``resolved_down_sample``, which the resolution defines. So the constitutive
    prelude is spliced in TWO pieces with the weld's struct, cell lift and
    resolution between them, and each piece is found by an exact anchor in the
    shared lift rather than by position here.
    """
    return (_weld.curl_prelude(ARM, expansion)
            + _weld.constitutive_prelude_defines()
            + _weld.weld_args_struct(ARM)
            + _weld.raw_step_D_cell_source(ARM, expansion)
            + _weld.resolution_source(ARM)
            + _weld.constitutive_prelude_helpers())


def signature(row_mask: Sequence[int]) -> str:
    """The kernel's parameter list, with only the LIVE row coefficients bound."""
    mask = _offdiag.normalized_row_mask(row_mask)
    rows = "\n".join(f"    const float* {_offdiag.ROW_PARAMETERS[slot]},"
                     for slot, flag in enumerate(mask) if flag)
    return f'''
extern "C" __global__ void {KERNEL_NAME}(
    // THE SCRATCH OUTPUTS, as the float32 word views of complex64 volumes. A
    // DIFFERENT ALLOCATION from the group below -- assert_scratch_is_disjoint
    // checks it by base address before every launch -- and that separation is the
    // whole design: nothing this launch reads is ever a word this launch wrote, so
    // every foreign sample is a pure function of unwritten memory rather than a
    // race with another block.
    float* __restrict__ f0_out, float* __restrict__ f1_out,
    float* __restrict__ f2_out,
    float* __restrict__ u0_out, float* __restrict__ u1_out,
    float* __restrict__ u2_out,
    // THE PRE-LAUNCH STATE, read at ANY cell by ANY thread and written by none.
    // The certified curl body's own names, because it is spliced verbatim.
    const float* __restrict__ f0, const float* __restrict__ f1,
    const float* __restrict__ f2,
    const float* __restrict__ u0, const float* __restrict__ u1,
    const float* __restrict__ u2,
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    // E and its constitutive history: written at the thread's OWN cell only.
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    // NOT __restrict__: an isotropic run hands the same device pointer three times
    // (fields.py:1321-1326). FLOAT32 under complex storage, indexed by the COMPLEX
    // CELL index -- chi1inv and inv_eps are never word-doubled.
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    // NOT __restrict__ either: a row volume may legally alias another row's, an
    // epsilon volume or a D volume.
{rows}
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx,
    // The D curl's INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // update_E's HALF-INTEGER pair, its kms renamed: the two sub-lattices collide
    // on the bare name and a shadow there is half a cell in the absorber profile.
    const float* __restrict__ kps_x, const float* __restrict__ kms_half_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_half_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_half_z,
    // THE CURL's boundary codes (coverage.real_curl_boundary_codes), which SPLIT
    // the two folded terminations: only a folded PERIODIC axis stores a slot past
    // MEEP's owned window, and only its top plane is masked.
    int bc_x, int bc_y, int bc_z,
    // THE CONSTITUTIVE's boundary codes (complex_offdiag_boundary_codes), which
    // serve BOTH folded terminations with one BC_MIRROR: update_E has no ownership
    // mask and no reflect row. A DIFFERENT reading of the same grid.
    int cbc_x, int cbc_y, int cbc_z,
    // The COUPLING's wall mask (coverage.offdiag_wall_mask_flags). Zero on every
    // folded axis by construction: _mask_metallic_wall_coupling asks is_metallic
    // AND NOT is_mirrored (stepping.py:1253).
    int wm_x, int wm_y, int wm_z,
    // The Bloch flags, SHARED by both halves: 0 means the multiply is SKIPPED and
    // never done against 1 + 0j, which is what makes k = 0 bit-identical.
    int ph_x, int ph_y, int ph_z,
    // The mirror ghost's parity per axis (complex_offdiag_update_e.
    // mirror_ghost_weights) -- _shift_down's own weight, applied ON THE GHOST LANE
    // AND NOWHERE ELSE, through the certified mul_coefficient_left.
    float gw_x, float gw_y, float gw_z,
    // THE DOWN (CONJUGATE) BLOCH FACTORS, BOUND ONCE FOR BOTH HALVES. The curl is
    // the BACKWARD sub-step and takes the conjugate; the constitutive's partner-
    // axis DOWN shift takes the same conjugate. The launcher RAISES where the two
    // resolvers disagree rather than binding one of them twice.
    float dpxr, float dpxi, float dpyr, float dpyi, float dpzr, float dpzi,
    // THE UP (FORWARD) FACTORS -- the constitutive's own-axis shift alone.
    float upxr, float upxi, float upyr, float upyi, float upzr, float upzi,
    // zero_metal_D's walls (in_seam_coverage.zero_metal_axes). A DIFFERENT
    // question from wm_*: that masks the coupling TOTAL, this clears the stored D.
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


def constitutive_body(row_mask: Sequence[int], expansion: Any) -> str:
    """``update_E_pml_complex_offdiag``'s body, seamed by the shared lift.

    ``boundary_prefix`` is ``"cbc_"``: THE TWO READINGS DIFFER ON A FOLD AND BOTH
    ARE BOUND, which is this family's one edit over its unfolded sibling.
    """
    return _weld.constitutive_body(ARM, row_mask, expansion,
                                   boundary_prefix="cbc_")


def kernel_source(row_mask: Sequence[int], expansion: Any) -> str:
    """The whole device source for one row mask and one expansion arm. PURE ASCII."""
    mask = _offdiag.normalized_row_mask(row_mask)
    preamble = [
        "    // ONE bounds guard and ONE index decode for both halves -- the",
        "    // certified curl's own two blocks, kept here and dropped from the",
        "    // certified constitutive body, which emits the identical arithmetic",
        "    // plus the nyz its flat() needs. Every index here is a COMPLEX CELL.",
        "    const int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx >= nx * ny * nz) return;",
        "",
        "    const int nyz = ny * nz;",
        "    const int k = idx % nz;",
        "    const int j = (idx / nz) % ny;",
        "    const int i = idx / (ny * nz);",
        "",
        _weld.weld_args_construction(ARM),
        "    // THE CURL HALF. Both results go to the SCRATCH; fu is the RAW stepped",
        "    // value at this cell because NO in-seam pass touches it -- neither fill",
        "    // writes fu_field (stepping.py:1451, :1533) and _zero_metal writes only",  # stepping.py live lines for the frozen device-text citation(s) in this string: 1533->1580
        "    // the named components (:2247).",
        "    cf d_raw[3];",
        "    cf fu_raw[3];",
        "    raw_step_D_cell(i, j, k, weld, d_raw, fu_raw);",
        "    cf_store(u0_out, idx, fu_raw[0]);",
        "    cf_store(u1_out, idx, fu_raw[1]);",
        "    cf_store(u2_out, idx, fu_raw[2]);",
        "",
        "    // ALL THREE IN-SEAM PASSES, resolved for this thread's own cell: the",
        "    // near fill, the wall clear and the far fill, in the driver's order.",
        "    // The same function the constitutive half calls for a foreign cell, so",
        "    // the own-cell value and a neighbour's are the same arithmetic by",
        "    // construction rather than by a second transcription.",
        "    const cf v_x = resolve_Dx(i, j, k, weld);",
        "    const cf v_y = resolve_Dy(i, j, k, weld);",
        "    const cf v_z = resolve_Dz(i, j, k, weld);",
        "    cf_store(f0_out, idx, v_x);",
        "    cf_store(f1_out, idx, v_y);",
        "    cf_store(f2_out, idx, v_z);",
        "    // The certified complex off-diagonal body follows, from its own",
        "    // neighbour-coordinate line down, with its boundary codes renamed to",
        "    // the constitutive triple and its D reads routed through the",
        "    // resolution. Everything else is lifted, not retyped.",
    ]
    source = (prelude(expansion) + signature(mask) + "\n".join(preamble)
              + constitutive_body(mask, expansion) + "}\n")
    try:
        source.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AssertionError(
            f"the fused device source is not pure ASCII ({exc}); NVRTC's source "
            f"file is written through the locale encoding and this would fail at "
            f"first launch, not at import") from exc
    return source


def device_sources(row_mask: Sequence[int] = (1, 1, 1, 1, 1, 1)
                   ) -> Dict[str, str]:
    """Every source this family can emit for one row mask, keyed by arm name."""
    return {f"{KERNEL_NAME}:{name}": kernel_source(row_mask, name)
            for name in sorted(complex_emitter.EXPANSIONS)}


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered."""
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for mask in _offdiag.LIVE_ROW_MASKS:
        for name in sorted(complex_emitter.EXPANSIONS):
            digest.update(f"{mask!r}|{name}".encode("ascii"))
            digest.update(kernel_source(mask, name).encode("utf-8"))
    return digest.hexdigest()


def _get_kernel(row_mask: Sequence[int], expansion: Any,
                source: Optional[str] = None):
    """Compile on first use, memoized on (name, arm, options, policy, source)."""
    import cupy as cp  # noqa: PLC0415 - device-only

    code = kernel_source(row_mask, expansion) if source is None else source
    key = _kernel_cache_key(
        f"{KERNEL_NAME}_arm{complex_emitter.normalized_expansion(expansion)}",
        True, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# =============================================================================
# THE PREDICATE
# =============================================================================

def covers_folded_complex_offdiag_fused_electric_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None,
        license: Any = None, subnormal_policy: Any = None, *,
        curl_license: Any = None) -> Tuple[bool, str]:
    """May ONE launch span the whole folded complex D seam here?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. The curl half's own certified predicate
    -- ``covers_complex_folded_curl``, which REQUIRES a mirror fold under complex
    storage -- and the constitutive arm's ``covers_complex_offdiag_pml_update_e``
    are both asked in full. What this predicate ADDS is the seam clauses: the fill
    plan must be buildable, both boundary readings must resolve, the source slot
    must be empty and the stored extents must be one tuple.

    TWO LICENCES, ONE PER HALF, AND THAT IS NOT A CONVENIENCE. This is the first
    product on this board whose halves sit in DIFFERENT expansion-licence families:
    the folded complex curl binds ``registry.LICENSE_COMPLEX`` and the off-diagonal
    ``update_E`` binds ``LICENSE_COMPLEX_OFFDIAG``, whose arbiter is
    ``folded_complex.parity_expansion_license`` and classifies a FIFTH orientation
    the base four do not carry (the mirror parity as a coefficient on the left).
    Handing one verdict to both halves would silently declare that one record's
    pattern set answered for the other's, so ``curl_license`` is a separate
    argument; it defaults to ``license`` for a single-licence caller, and the
    composer passes both.
    """
    covered, reason = _folded.covers_complex_folded_curl(
        fields, pml, grid, "step_D",
        license if curl_license is None else curl_license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _offdiag.covers_complex_offdiag_pml_update_e(
        fields, pml, grid, license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE INJECTION ROUTE, REFUSED BY NAME. Both certified halves admit a
    # conductivity, so this refusal cannot be inherited from either and is made
    # here: the driver deposits an electric source through
    # _inject_electric_through_conductivity, which rescales by condinv, and this
    # product declares no deposit repair to have a verdict about that route with.
    if getattr(fields, "has_conductivity", False):
        return False, ("the engine carries a conductivity, so the driver deposits "
                       "this seam's electric sources through "
                       "_inject_electric_through_conductivity (driver.py:3319), "
                       "which rescales the increment by condinv; this pair carries "
                       "no deposit repair and has no verdict on that route")

    # THE FILL PLAN MUST BE BUILDABLE, and its three refusals are this weld's, not
    # inherited: an unreadable plane phase would launch a NaN weight, an axis
    # reported both folded and walled would put the wall clear on a plane MEEP
    # steps, and a far reflect row without a phase is a pass that cannot run. Each
    # is a plane of wrong values rather than a crash, which is why the plan RAISES
    # and this catches.
    try:
        plan = _weld.fill_plan(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, f"the in-seam fill plan: {type(exc).__name__}: {exc}"
    if not any(plan["near"]):
        return False, ("no axis carries a mirror fill: this product is the FOLDED "
                       "complex weld and its curl half requires a fold, so a "
                       "fold-free grid reaching here would mean the two readings "
                       "of the fold had diverged")

    # BOTH BOUNDARY READINGS MUST RESOLVE. They differ on a fold BY DESIGN and both
    # are bound, so there is nothing to cross-check here -- but a grid either
    # resolver refuses is a grid this launch cannot bind.
    try:
        folded_complex_offdiag_fused_electric_pair_codes(grid)
    except Exception as exc:  # noqa: BLE001
        return False, (f"the two boundary readings this launch binds: "
                       f"{type(exc).__name__}: {exc}")

    # THE ONE PHASE TRIPLE THAT FEEDS BOTH HALVES MUST BE ONE NUMBER BY BOTH
    # ROUTES. The certified curl reads the CONJUGATE factor for step_D and the
    # certified constitutive's DOWN factor is the same conjugate; binding one
    # where the other belonged is a phase error that leaves every magnitude
    # plausible, so it is refused rather than trusted.
    try:
        folded_complex_offdiag_fused_electric_pair_phases(grid)
    except Exception as exc:  # noqa: BLE001
        return False, (f"the Bloch table this launch binds once for both halves: "
                       f"{type(exc).__name__}: {exc}")

    # THE SOURCE SEAM, REFUSED RATHER THAN CARRIED -- see CARRIES_DEPOSIT_REPAIR.
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, "D",
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3319/:3322) and "
            f"this pair declares no deposit repair -- deposit_repair.repairable "
            f"refuses an off-diagonal chi1inv row, which every row of this cell "
            f"carries, and on a folded grid the repair would additionally have to "
            f"reconstruct the deposit's mirror images through a stencil"),
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
                       f"ONE bounds guard and may only do so where the three are "
                       f"one number")
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

_FIELD_BINDINGS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz", "Hx", "Hy", "Hz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
)
_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def folded_complex_offdiag_fused_electric_pair_tables(pml: Any
                                                      ) -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads.

    The D curl takes the INTEGER lattice (``complex_curl_tables(pml, False)``) and
    ``update_E`` the HALF-INTEGER one (``complex_offdiag_tables(pml)``, which asks
    ``coverage.constitutive_sub_lattice`` for itself).
    """
    if complex_curl_tables is None:
        raise RuntimeError(
            "complex_pml_kernels is not importable on this host, so the curl's "
            "coefficient tables cannot be resolved. The predicate needs none of "
            "them and still answers")
    return {"curl": complex_curl_tables(pml, False),
            "constitutive": _offdiag.complex_offdiag_tables(pml)}


def folded_complex_offdiag_fused_electric_pair_codes(grid: Any
                                                     ) -> Dict[str, Tuple[int, ...]]:
    """BOTH boundary readings, kept apart. THEY DIFFER ON A FOLD AND THAT IS THE POINT.

    Returned as a mapping rather than two positional tuples so a caller cannot bind
    them the wrong way round: on a folded METALLIC axis the curl says
    ``BC_METALLIC`` and the constitutive says ``BC_MIRROR``, and swapping them is a
    wrong ghost rule on a whole plane rather than a crash.

    BOTH READINGS COME FROM STDLIB-ONLY SIBLINGS, which is a predicate requirement
    rather than a convenience: the census asks this clause on a laptop with no
    CuPy. ``folded_complex_boundary_codes`` FAILS CLOSED, returning
    ``(codes, refusal)`` with exactly one None, and the refusal is carried rather
    than swallowed.
    """
    codes, refusal = _folded.folded_complex_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(
            f"the curl's own boundary resolution refuses this grid: {refusal}")
    return {"curl": tuple(int(code) for code in codes),
            "constitutive": tuple(int(code) for code in
                                  _offdiag.complex_offdiag_boundary_codes(grid))}


def folded_complex_offdiag_fused_electric_pair_phases(grid: Any) -> Dict[str, Any]:
    """The Bloch flags and both directions' factors, as ONE table for both halves.

    ``complex_offdiag_update_e.bloch_phase_table`` is the source, because it is the
    stdlib-plus-numpy spelling and returns BOTH directions; the device-only
    ``complex_pml_kernels.bloch_phase_arguments`` is asked as a CROSS-CHECK where
    it is importable, and a disagreement RAISES. The curl half reads only the DOWN
    (conjugate) triple, which is exactly what ``bloch_phase_arguments(grid, True)``
    hands its own kernels, so one binding serves both -- but only while the two
    spellings agree, and that is measured rather than assumed.
    """
    flags, down, up = _offdiag.bloch_phase_table(grid)
    if bloch_phase_arguments is not None:
        curl_flags, curl_values = bloch_phase_arguments(grid, True)
        if tuple(int(flag) for flag in curl_flags) != tuple(int(f) for f in flags):
            raise ValueError(
                f"the curl's Bloch flags {tuple(int(f) for f in curl_flags)!r} and "
                f"the constitutive's {tuple(int(f) for f in flags)!r} disagree; a "
                f"flag decides whether the multiply happens at all, and this "
                f"launch binds ONE triple for both halves")
        if tuple(float(v) for v in curl_values) != tuple(float(v) for v in down):
            raise ValueError(
                f"the curl's backward Bloch factors "
                f"{tuple(float(v) for v in curl_values)!r} and the constitutive's "
                f"DOWN factors {tuple(float(v) for v in down)!r} disagree; both "
                f"are conj(grid.bloch_phase(axis)) rounded to complex64 and this "
                f"launch binds one of them for both halves")
    return {"flags": tuple(int(flag) for flag in flags),
            "down": tuple(float(value) for value in down),
            "up": tuple(float(value) for value in up)}


def folded_complex_offdiag_fused_electric_pair_scratch(fields: Any) -> Dict[str, Any]:
    """The six launch-local complex volumes, allocated once per frozen configuration.

    NOT ``StepScratch`` -- that pool is for values consumed within the call that
    produced them (fields.py:298-327), and the rotation hands these to the driver
    as the live ``D``/``fu_D``. Uninitialized is correct: every cell of all six is
    written by every launch.
    """
    import cupy as cp  # noqa: PLC0415 - device-only

    volumes: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        source = getattr(fields, name, None)
        if source is None:
            raise ValueError(
                f"fields.{name} is not allocated; this pair rotates a scratch "
                f"shaped like it and has nothing to shape one from")
        volumes[name] = cp.empty_like(source)
    return volumes


def assert_scratch_is_disjoint(fields: Any, scratch: Dict[str, Any],
                               tables: Dict[str, Dict[str, Any]]) -> int:
    """No scratch volume is a bound input, and no two restrict arguments alias.

    THE FIRST CHECK IS THE DESIGN, not a convention: if ``f0_out`` were ``Dx`` the
    launch would be the IN-PLACE weld the board refused, and every foreign
    recompute would read words other blocks had already overwritten. The fold makes
    it sharper -- one of the redirect sources is stored row 2, deep INSIDE the
    volume rather than at a face.
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
    for name in _FIELD_BINDINGS:
        visit(name, getattr(fields, name))
    for key in _CURL_TABLE_KEYS:
        visit(f"curl:{key}", tables["curl"][key])
    for key in _CONSTITUTIVE_TABLE_KEYS:
        visit(f"constitutive:{key}", tables["constitutive"][key])
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
            int(code) not in _offdiag.FOLDED_BC_CODES.values()
            for code in constitutive):
        raise ValueError(
            f"the constitutive boundary codes must be three of "
            f"{sorted(set(_offdiag.FOLDED_BC_CODES.values()))}, got "
            f"{constitutive!r}")
    for axis in range(3):
        if int(constitutive[axis]) != _offdiag.BC_MIRROR_CODE:
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
        if int(shape[axis]) <= _offdiag.MIRROR_SOURCE_INDEX:
            raise ValueError(
                f"axis {axis} is folded with {shape[axis]} stored cells; the "
                f"mirror ghost images stored row "
                f"{_offdiag.MIRROR_SOURCE_INDEX}")
        if not int(fills["near"][axis]):
            raise ValueError(
                f"axis {axis} reads BC_MIRROR to the constitutive half and carries "
                f"no near fill; fill_symmetry_bc_D writes stored cell 0 on every "
                f"mirrored axis (stepping.py:1484-1485) and the two readings of "
                f"the fold have diverged")
        if float(fills["far_phase"][axis]) != float(weights[axis]):
            raise ValueError(
                f"axis {axis}'s mirror ghost weight is {weights[axis]!r} and its "
                f"far fill parity {fills['far_phase'][axis]!r}; both are "
                f"mirror_parity of that axis's own shift-1 D component and must be "
                f"one number by two routes")


def launch_folded_complex_offdiag_fused_electric_pair(
        fields: Any, scratch: Dict[str, Any], tables: Dict[str, Dict[str, Any]],
        codes: Dict[str, Tuple[int, ...]], wall_mask: Sequence[int],
        phases: Dict[str, Any], weights: Sequence[float],
        fills: Dict[str, Tuple[Any, ...]], dtdx: float,
        row_mask: Sequence[int], rows: Sequence[Any], expansion: Any,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch. THE CALLER ROTATES AFTERWARDS.

    ``kernel`` IS THE GATE'S DOOR: a gate compiles a deliberately broken copy of
    the shipped source and hands it here. A launcher that could not be handed its
    own kernel could not arm a single mutation.
    """
    if word_view is None:
        raise RuntimeError(
            "complex_pml_kernels is not importable on this host, so the complex "
            "volumes cannot be bound as their float32 word views")
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments: List[Any] = [word_view(scratch[name]) for name in SCRATCH_VOLUMES]
    arguments += [word_view(getattr(fields, name)) for name in _FIELD_BINDINGS]
    arguments += [fields.inverse_epsilon_for(component)
                  for component in _ELECTRIC]
    arguments += list(rows)
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx)]
    arguments += [curl[key] for key in _CURL_TABLE_KEYS]
    arguments += [constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS]
    arguments += [np.int32(int(code)) for code in codes["curl"]]
    arguments += [np.int32(int(code)) for code in codes["constitutive"]]
    arguments += [np.int32(int(bool(flag))) for flag in wall_mask]
    arguments += [np.int32(int(flag)) for flag in phases["flags"]]
    arguments += [np.float32(float(value)) for value in weights]
    arguments += [np.float32(float(value)) for value in phases["down"]]
    arguments += [np.float32(float(value)) for value in phases["up"]]
    arguments += [np.int32(int(flag)) for flag in fills["wall"]]
    arguments += [np.int32(int(flag)) for flag in fills["near"]]
    arguments += [np.int32(int(row)) for row in fills["reflect"]]
    arguments += [np.float32(float(value)) for value in fills["near_phase"]]
    arguments += [np.float32(float(value)) for value in fills["far_phase"]]
    (kernel or _get_kernel(row_mask, expansion))(
        (blocks,), (_FUSED_THREADS,), tuple(arguments))
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "row_mask": tuple(int(flag) for flag in row_mask),
            "arm": complex_emitter.normalized_expansion(expansion),
            "boundary_codes": {key: tuple(value) for key, value in codes.items()},
            "coupling_wall_mask": tuple(int(bool(flag)) for flag in wall_mask),
            "ghost_weights": tuple(float(value) for value in weights),
            "phase_flags": tuple(int(flag) for flag in phases["flags"]),
            "fills": {key: tuple(fills[key]) for key in
                      ("near", "wall", "reflect", "near_phase", "far_phase")}}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION -- the certified E->P choreography, one seam earlier.

    See :func:`offdiag_fused_electric_pair.rotate_into_fields` for the consumer
    audit this rests on and the test that measures it. The volumes handed back are
    the RETIRED ones, which the caller may reuse as the next launch's scratch.
    """
    retired: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        retired[name] = getattr(fields, name)
        setattr(fields, name, scratch[name])
    return retired


def run_folded_complex_offdiag_fused_electric_pair(
        fields: Any, grid: Any, pml: Any, dtdx: float, expansion: Any, *,
        scratch: Optional[Dict[str, Any]] = None, sources: Any = None,
        license: Any = None, subnormal_policy: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None, rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once, rotate."""
    covered, reason = covers_folded_complex_offdiag_fused_electric_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = folded_complex_offdiag_fused_electric_pair_tables(pml)
    if scratch is None:
        scratch = folded_complex_offdiag_fused_electric_pair_scratch(fields)
    assert_scratch_is_disjoint(fields, scratch, tables)
    codes = folded_complex_offdiag_fused_electric_pair_codes(grid)
    phases = folded_complex_offdiag_fused_electric_pair_phases(grid)
    wall_mask = offdiag_wall_mask_flags(grid)
    weights = _offdiag.mirror_ghost_weights(grid)
    fills = _weld.fill_plan(grid)
    _validate_launch_arguments(tuple(int(n) for n in fields.Dx.shape), codes,
                               wall_mask, weights, fills)
    mask = _offdiag.normalized_row_mask(offdiag_row_mask(fields))
    rows = [volume for volume in offdiag_row_volumes(fields) if volume is not None]
    record = launch_folded_complex_offdiag_fused_electric_pair(
        fields, scratch, tables, codes, wall_mask, phases, weights, fills, dtdx,
        mask, rows, expansion, kernel)
    record["rotated"] = bool(rotate)
    record["scratch"] = rotate_into_fields(fields, scratch) if rotate else scratch
    return record


#: Read by the host suite: the wall-clear planner this family and the shared lift
#: both consult, so a test can assert the identity rather than diff two spellings.
_ZERO_METAL_REFERENCE = zero_metal_axes
_COVERAGE_REFERENCE = _coverage
