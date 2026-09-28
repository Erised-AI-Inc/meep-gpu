"""The NO-ABSORBER COMPLEX stencil weld: complex no-PML curl x off-diagonal ``update_E``.

THE CELL, AND WHAT WAS RECORDED ABOUT IT. ``D_to_E (cuda_complex_no_pml/complex
no-PML curl, cuda_complex_offdiag/complex off-diagonal no-PML)`` carries 2 corpus
rows -- ``tests:TestMaterialGrid.test_matgrid_3d`` and
``tests:TestMaterialGrid.test_subpixel_smoothing`` -- of which ONE clears the source
injection on the driver fact alone, and it is the second of the two cells the
hand-CUDA board still recorded STENCIL-BLOCKED after the real-storage welds landed.
:mod:`.complex_offdiag_stencil_weld` carries the argument that the verdict is a
statement about the IN-PLACE weld and does not reach a SCRATCH-OUTPUT one; this
module is that shape's no-absorber arm, and :mod:`.folded_complex_offdiag_fused_
electric_pair` is its folded twin.

=============================================================================
WHAT IS SHORTER HERE, AND WHAT THAT REMOVES RATHER THAN CHANGES
=============================================================================

The seam is SHORTER than the folded twin's rather than differently filled, and every
difference is a subtraction:

* **there is no split-field recurrence at either end.** The curl tail is
  ``no_pml_apply`` -- ``target -= curl``, three lines -- so there is no ``fu``
  volume, the scratch is THREE volumes rather than six, and the resolution's raw
  call takes no auxiliary. The constitutive tail is ``stepping.py:1022``'s plain
  OVERWRITE, so there is no ``f_w_E``, no ``kps``/``kms`` pair and no sub-lattice
  rename to make: the INTEGER/HALF-INTEGER collision the PML welds all have to
  spell around does not arise because neither half binds a coefficient vector;
* **the fold cannot reach this cell.** ``complex_no_pml_kernels`` refuses a mirror
  fold BY NAME, so ``near`` and ``reflect`` are inert on every grid this predicate
  admits and the only in-seam pass with work to do is ``zero_metal_D``. The
  resolution still carries all three, because it is the SHARED emitter and one
  spelling of the composition is the whole point of sharing it -- what changes is
  that the two fill arms are provably dead here, which the host suite asserts on
  every admitted fixture rather than assuming;
* **one boundary triple serves both halves, and that is CHECKED rather than
  assumed.** The folded twin binds two because the curl splits the fold's two
  terminations while ``update_E`` serves both with ``BC_MIRROR``; with no fold in
  reach the two resolvers must agree, so this family binds ONE triple and
  :func:`complex_no_pml_offdiag_fused_electric_pair_codes` RAISES where
  ``complex_pml_kernels.complex_boundary_codes`` and
  ``complex_offdiag_update_e.complex_offdiag_boundary_codes`` disagree.

WHAT DOES NOT CHANGE is every multiply. The parity is a full complex product with
the coefficient on the LEFT at BOTH signs, the wall clear is ``cf_zero()`` in both
words, and the clear REPLACES the near arm rather than following it -- all three are
:mod:`.complex_offdiag_stencil_weld`'s measurements and its docstring is where they
are derived.

=============================================================================
THE CONDUCTIVE ARM IS REFUSED BY NAME
=============================================================================

``complex_no_pml_kernels`` emits TWO curl arms, plain and conductive, and this weld
lifts the PLAIN one (``_curl_source("step_D", False, ...)``). A conductive
no-absorber run is therefore refused BY NAME by :func:`covers_complex_no_pml_
offdiag_fused_electric_pair` rather than served with a tail that is not its own --
and the refusal is doubled, because on that route the driver also deposits electric
sources through ``_inject_electric_through_conductivity``.

=============================================================================
THE DEPOSIT IS REFUSED, AND THAT IS A MEASUREMENT
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is ``False`` for the real off-diagonal welds' reason
and by the same measurement: ``deposit_repair.repairable(fields, "D")`` refuses an
off-diagonal chi1inv row BY NAME -- a point repair recomputes E at the deposit cell
from THAT cell's displacement, while this constitutive reads its NEIGHBOURS' -- and
every row of this cell carries one by construction. The ceiling is the rows that
clear the injection on the driver fact alone: 1 of this cell's 2.

NOTHING HERE IS DISPATCH. ``meep_gpu.fastpath.plan_fast_path`` still returns ``None``
on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import complex_emitter
from . import complex_no_pml_kernels as _no_pml
from . import complex_offdiag_stencil_weld as _weld
from . import complex_offdiag_update_e as _offdiag
from . import coverage as _coverage
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import offdiag_row_mask, offdiag_row_volumes, offdiag_wall_mask_flags
from .in_seam_coverage import zero_metal_axes

from .. import deposit_repair as _deposit_repair

try:  # device-only: the boundary resolver's sibling and the word view
    from .complex_pml_kernels import (bloch_phase_arguments,
                                      complex_boundary_codes, word_view)
except ImportError:  # a host with no CuPy: only the launch is unavailable
    bloch_phase_arguments = complex_boundary_codes = word_view = None


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
    "fused_electric_pair_no_pml_complex_offdiag",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument. The
#: arithmetic was proved off-device FIRST, against the driver's own pass order, by
#: ``parity/meep_gpu/probe_cuda_complex_offdiag_scratch_weld.py``.
UNCERTIFIED_KERNELS: Dict[str, str] = {}

#: FALSE, and MEASURED off ``deposit_repair.repairable`` rather than chosen -- see
#: the module docstring. The host suite asserts it against the census for THIS
#: cell's rows rather than against this comment.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_complex_no_pml_offdiag_fused_electric_pair"

#: The ``extern "C"`` symbol NVRTC is asked for, spelled once.
KERNEL_NAME = "fused_electric_pair_no_pml_complex_offdiag"

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3317, :3327, :3332). THREE, not five: the two mirror fills cannot
#: reach a grid this family's curl half admits, so declaring them would claim a
#: pass this product can never have work for.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: The three launch-local COMPLEX volumes the curl half writes and the launcher
#: rotates in afterwards. THREE rather than six: there is no split-field auxiliary
#: on this arm.
SCRATCH_VOLUMES: Tuple[str, ...] = ("Dx", "Dy", "Dz")

_ELECTRIC: Tuple[str, str, str] = ("Ex", "Ey", "Ez")

#: The curl arm this family's half of the shared lift takes.
ARM = "no_pml"

_FUSED_THREADS = 256
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: Every line of certified device text this file did not lift verbatim, with the
#: reason. DATA, not prose, so a gate and the host suite can assert the list
#: against the emitted source rather than against a docstring. Both halves' entries
#: are the SHARED lift's own declarations, imported by value, so a family that
#: drifted from the lift it calls would be caught here.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    _weld.curl_lift_edits(ARM) + _weld.RESOLUTION_LIFT_EDITS + (
        {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                 "    if (idx >= nx * ny * nz) return;   (BOTH certified bodies)",
         "became": "(the weld's own guarded preamble, once)",
         "why": "the weld emits the guard and the decode once for both halves; "
                "splicing both would redeclare idx, i, j, k and nyz."},
    ))

__all__ = [
    "ARM", "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SCRATCH_VOLUMES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_scratch_is_disjoint", "constitutive_body",
    "complex_no_pml_offdiag_fused_electric_pair_codes",
    "complex_no_pml_offdiag_fused_electric_pair_phases",
    "complex_no_pml_offdiag_fused_electric_pair_scratch", "corpus_digest",
    "covers_complex_no_pml_offdiag_fused_electric_pair", "device_sources",
    "kernel_source", "launch_complex_no_pml_offdiag_fused_electric_pair",
    "prelude", "rotate_into_fields",
    "run_complex_no_pml_offdiag_fused_electric_pair", "signature",
]


# =============================================================================
# THE DEVICE CODE
# =============================================================================

def prelude(expansion: Any) -> str:
    """Both certified preludes plus the weld's own helpers, in dependency order.

    THE ORDER IS FORCED AND IS NOT A STYLE CHOICE -- see the folded twin's
    :func:`folded_complex_offdiag_fused_electric_pair.prelude`. Each piece is found
    by an exact anchor in the shared lift rather than by position here.
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
    // race with another block. THREE, not six: this arm has no auxiliary.
    float* __restrict__ f0_out, float* __restrict__ f1_out,
    float* __restrict__ f2_out,
    // THE PRE-LAUNCH STATE, read at ANY cell by ANY thread and written by none.
    // The certified curl body's own names, because it is spliced verbatim.
    const float* __restrict__ f0, const float* __restrict__ f1,
    const float* __restrict__ f2,
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    // E: written at the thread's OWN cell only. NO f_w_E -- the constitutive tail
    // on this arm is stepping.py:993's plain overwrite, which reads neither the
    // stored E nor a history, and binding one here would be the split-field
    // recurrence this family does not have.
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    // NOT __restrict__: an isotropic run hands the same device pointer three times
    // (fields.py:1321-1326). FLOAT32 under complex storage, indexed by the COMPLEX
    // CELL index -- chi1inv and inv_eps are never word-doubled.
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    // NOT __restrict__ either: a row volume may legally alias another row's, an
    // epsilon volume or a D volume.
{rows}
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx,
    // ONE boundary triple for BOTH halves. The folded twin binds two because the
    // curl splits the fold's terminations while update_E serves both with
    // BC_MIRROR; with no fold in reach the two resolvers must agree, and the
    // launcher RAISES where they do not rather than binding one of them twice.
    int bc_x, int bc_y, int bc_z,
    // The COUPLING's wall mask (coverage.offdiag_wall_mask_flags).
    int wm_x, int wm_y, int wm_z,
    // The Bloch flags, SHARED by both halves: 0 means the multiply is SKIPPED and
    // never done against 1 + 0j, which is what makes k = 0 bit-identical.
    int ph_x, int ph_y, int ph_z,
    // The mirror ghost's parity per axis. Bound because the certified constitutive
    // body reads it on the ghost lane; DEAD on every grid this family admits,
    // because its curl half refuses a mirror fold by name.
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
    // The two mirror FILLS' plan. Bound because the SHARED resolution reads it;
    // inert on every grid this family admits, and the host suite asserts that on
    // each admitted fixture rather than assuming it.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float near_phase_x, float near_phase_y, float near_phase_z,
    float far_phase_x, float far_phase_y, float far_phase_z
) {{
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 993->1022


def constitutive_body(row_mask: Sequence[int], expansion: Any) -> str:
    """``update_E_no_pml_complex_offdiag``'s body, seamed by the shared lift.

    ``boundary_prefix`` stays ``"bc_"``: ONE triple serves both halves here, and
    the launcher is where the two resolvers are cross-checked.
    """
    return _weld.constitutive_body(ARM, row_mask, expansion, boundary_prefix="bc_")


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
        "    // THE CURL HALF. The result goes to the SCRATCH; there is no",
        "    // auxiliary on this arm, so the raw call hands back one array.",
        "    cf d_raw[3];",
        "    raw_step_D_cell(i, j, k, weld, d_raw);",
        "",
        "    // THE IN-SEAM PASSES, resolved for this thread's own cell in the",
        "    // driver's order. The wall clear is the one with work to do here; the",
        "    // two fill arms are inert because this family's curl half refuses a",
        "    // mirror fold by name. The SAME function the constitutive half calls",
        "    // for a foreign cell, so the own-cell value and a neighbour's are the",
        "    // same arithmetic by construction rather than by a second",
        "    // transcription.",
        "    const cf v_x = resolve_Dx(i, j, k, weld);",
        "    const cf v_y = resolve_Dy(i, j, k, weld);",
        "    const cf v_z = resolve_Dz(i, j, k, weld);",
        "    cf_store(f0_out, idx, v_x);",
        "    cf_store(f1_out, idx, v_y);",
        "    cf_store(f2_out, idx, v_z);",
        "    // The certified complex off-diagonal body follows, from its own",
        "    // neighbour-coordinate line down, with its D reads routed through the",
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

def covers_complex_no_pml_offdiag_fused_electric_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None,
        license: Any = None, subnormal_policy: Any = None, *,
        curl_license: Any = None) -> Tuple[bool, str]:
    """May ONE launch span the whole no-absorber complex D seam here?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal.

    A CONJUNCTION, AND NOTHING IS WEAKENED. ``covers_complex_no_pml_curl(step_D)``
    and ``covers_complex_no_pml_offdiag_update_e`` are both asked in full,
    including the extended parity expansion licence the off-diagonal family binds.
    What this predicate ADDS is the seam clauses: the PLAIN curl arm (the
    conductive one is a different certified tail and is refused by name), the fill
    plan, the single boundary triple, the single Bloch table, the empty source slot
    and the stored extents.

    TWO LICENCES, ONE PER HALF -- see the folded twin's clause. The complex
    no-absorber curl binds ``registry.LICENSE_COMPLEX_NO_PML``, whose arbiter reads
    a SEVEN-pattern record, and the off-diagonal ``update_E`` binds
    ``LICENSE_COMPLEX_OFFDIAG``, whose arbiter carries the parity orientation.
    Neither record answers for the other, so ``curl_license`` is a separate
    argument; it defaults to ``license`` for a single-licence caller.
    """
    covered, reason = _no_pml.covers_complex_no_pml_curl(
        fields, pml, grid, "step_D",
        license if curl_license is None else curl_license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _offdiag.covers_complex_no_pml_offdiag_update_e(
        fields, pml, grid, license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE PLAIN CURL ARM, AND ONLY IT. complex_no_pml_kernels emits two tails and
    # this weld lifts _curl_source(..., False, ...); a conductive run is a
    # DIFFERENT certified recurrence, and serving it with this one would be a wrong
    # answer rather than a crash. The driver's condinv-scaled electric deposit sits
    # on the same route, so the refusal is doubled.
    if getattr(fields, "has_conductivity", False):
        return False, ("the engine carries a conductivity: complex_no_pml_kernels' "
                       "CONDUCTIVE curl arm is a different certified recurrence and "
                       "this weld lifts the plain one, and the driver additionally "
                       "deposits this seam's electric sources through "
                       "_inject_electric_through_conductivity (driver.py:3319), "
                       "which rescales the increment by condinv")

    # THE FILL PLAN MUST BE BUILDABLE, and its refusals are this weld's rather than
    # inherited: an unreadable plane phase would launch a NaN weight and an axis
    # reported both folded and walled would put the wall clear on a plane MEEP
    # steps. Each is a plane of wrong values rather than a crash, which is why the
    # plan RAISES and this catches.
    try:
        plan = _weld.fill_plan(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, f"the in-seam fill plan: {type(exc).__name__}: {exc}"
    if any(plan["near"]) or any(int(row) >= 0 for row in plan["reflect"]):
        return False, ("an axis carries a mirror fill or a folded far image: this "
                       "is the UNFOLDED complex weld, its curl half refuses a fold "
                       "by name, and a folded grid reaching here would mean the two "
                       "readings of the fold had diverged")

    # THE ONE BOUNDARY TRIPLE MUST BE ONE NUMBER BY BOTH ROUTES, and the folded
    # twin is where they are allowed to differ. A disagreement here is a wrong
    # ghost rule on a whole plane rather than a crash.
    try:
        complex_no_pml_offdiag_fused_electric_pair_codes(grid)
    except Exception as exc:  # noqa: BLE001
        return False, (f"the boundary triple this launch binds once for both "
                       f"halves: {type(exc).__name__}: {exc}")

    # THE SAME, FOR THE BLOCH TABLE. The certified curl reads the CONJUGATE factor
    # for step_D and the certified constitutive's DOWN factor is that conjugate;
    # binding one where the other belonged is a phase error that leaves every
    # magnitude plausible.
    try:
        complex_no_pml_offdiag_fused_electric_pair_phases(grid)
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
            f"carries"),
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
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The nine pointers this launch binds, in signature order: D (``f0/f1/f2``), the
#: curl's operands (``g0/g1/g2``) and E.
#:
#: THE CURL'S OPERANDS ARE SPELLED AS **B**, NOT H, AND THAT IS NOT A NAMING CHOICE.
#: ``Fields.enable_field_storage`` deliberately does not allocate H without an
#: absorber -- "``update_H`` writes nothing without PML, so a stored H would sit at
#: zero for the whole run while ``get_H`` returned it" (fields.py:653-668) -- and
#: ``Fields.get_H`` returns the B array itself there (:1164-1187). Every grid THIS
#: product's predicate admits is exactly that grid: its curl half refuses a Fields
#: in PML storage mode beside an inert layer BY NAME. So ``fields.Hx`` is ``None``
#: on every admitted row, and binding it would hand ``word_view`` a ``None`` at the
#: first launch. The released no-absorber sibling
#: (:mod:`.no_pml_complex_fused_electric_pair`, ``_CURL_SOURCES``) spells it the
#: same way for the same measured reason, and ``stepping.step_D`` differences
#: exactly these arrays.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ex", "Ey", "Ez",
)


def complex_no_pml_offdiag_fused_electric_pair_codes(grid: Any) -> Tuple[int, ...]:
    """The ONE boundary triple both halves read, or a named failure.

    Two independent resolvers are asked and their answers must be one tuple:
    ``complex_pml_kernels.complex_boundary_codes`` (the curl's) and
    ``complex_offdiag_update_e.complex_offdiag_boundary_codes`` (the
    constitutive's). They differ only where a fold is live, which this family's
    curl half refuses -- so a disagreement here means the fold refusal and the
    boundary resolution have diverged, and this launch must not bind either
    reading. Where the device-only curl resolver is unimportable (the census
    laptop) the constitutive's answer stands alone, which is the same rule the
    predicate's other stdlib-only clauses follow.
    """
    constitutive = tuple(int(code) for code in
                         _offdiag.complex_offdiag_boundary_codes(grid))
    if complex_boundary_codes is not None:
        curl = tuple(int(code) for code in complex_boundary_codes(grid))
        if curl != constitutive:
            raise ValueError(
                f"the curl's boundary codes {curl!r} and the constitutive's "
                f"{constitutive!r} disagree; the two readings of a grid differ only "
                f"on a FOLD, which this family's curl half refuses by name, so a "
                f"disagreement means the fold refusal and the boundary resolution "
                f"have diverged and neither reading may be bound")
    return constitutive


def complex_no_pml_offdiag_fused_electric_pair_phases(grid: Any) -> Dict[str, Any]:
    """The Bloch flags and both directions' factors, as ONE table for both halves.

    The folded twin's :func:`folded_complex_offdiag_fused_electric_pair.
    folded_complex_offdiag_fused_electric_pair_phases`, with the same cross-check
    and the same reason.
    """
    flags, down, up = _offdiag.bloch_phase_table(grid)
    if bloch_phase_arguments is not None:
        curl_flags, curl_values = bloch_phase_arguments(grid, True)
        if tuple(int(flag) for flag in curl_flags) != tuple(int(f) for f in flags):
            raise ValueError(
                f"the curl's Bloch flags {tuple(int(f) for f in curl_flags)!r} and "
                f"the constitutive's {tuple(int(f) for f in flags)!r} disagree; a "
                f"flag decides whether the multiply happens at all, and this launch "
                f"binds ONE triple for both halves")
        if tuple(float(v) for v in curl_values) != tuple(float(v) for v in down):
            raise ValueError(
                f"the curl's backward Bloch factors "
                f"{tuple(float(v) for v in curl_values)!r} and the constitutive's "
                f"DOWN factors {tuple(float(v) for v in down)!r} disagree; both are "
                f"conj(grid.bloch_phase(axis)) rounded to complex64 and this launch "
                f"binds one of them for both halves")
    return {"flags": tuple(int(flag) for flag in flags),
            "down": tuple(float(value) for value in down),
            "up": tuple(float(value) for value in up)}


def complex_no_pml_offdiag_fused_electric_pair_scratch(fields: Any) -> Dict[str, Any]:
    """The three launch-local complex volumes, allocated once per configuration.

    NOT ``StepScratch`` -- that pool is for values consumed within the call that
    produced them (fields.py:298-327), and the rotation hands these to the driver
    as the live ``D``. Uninitialized is correct: every cell of all three is written
    by every launch.
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


def assert_scratch_is_disjoint(fields: Any, scratch: Dict[str, Any]) -> int:
    """No scratch volume is a bound input, and no two restrict arguments alias.

    THE FIRST CHECK IS THE DESIGN, not a convention: if ``f0_out`` were ``Dx`` the
    launch would be the IN-PLACE weld the board refused, and every foreign
    recompute would read words other blocks had already overwritten.
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
    if collisions:
        raise ValueError(
            "this pair binds every field and scratch argument __restrict__, and "
            "these arguments alias -- undefined behaviour NVRTC miscompiles "
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


def _validate_launch_arguments(codes: Sequence[int], wall_mask: Sequence[int],
                               fills: Dict[str, Tuple[Any, ...]]) -> None:
    """Everything a launch would otherwise get silently WRONG rather than crash.

    Three clauses, and each is a plane of wrong values if it is dropped: a code
    outside the family's own table would select no branch of ``coord_dn``; a live
    mirror code would mean the fold refusal had been bypassed; and a live fill flag
    would mean the resolution had work this family's REPLACES does not declare.
    """
    if len(codes) != 3 or any(int(code) not in _offdiag.FOLDED_BC_CODES.values()
                              for code in codes):
        raise ValueError(
            f"the boundary codes must be three of "
            f"{sorted(set(_offdiag.FOLDED_BC_CODES.values()))}, got "
            f"{tuple(codes)!r}")
    for axis in range(3):
        if int(codes[axis]) == _offdiag.BC_MIRROR_CODE:
            raise ValueError(
                f"axis {axis} resolves BC_MIRROR: this family's curl half refuses a "
                f"mirror fold by name, and a launch that bound one would take the "
                f"constitutive's mirror ghost arm on a grid the curl stepped as "
                f"though unfolded")
        if int(fills["near"][axis]) or int(fills["reflect"][axis]) >= 0:
            raise ValueError(
                f"axis {axis} carries a mirror fill (near={fills['near'][axis]!r}, "
                f"reflect={fills['reflect'][axis]!r}); this product declares "
                f"{REPLACES!r} and does not claim the two fill passes")
        if int(fills["wall"][axis]) and int(wall_mask[axis]) and \
                int(codes[axis]) != _offdiag.FOLDED_BC_CODES["metallic"]:
            raise ValueError(
                f"axis {axis} is wall-clearing and wall-masked but resolves "
                f"{codes[axis]!r}; both passes ask stepping's is_metallic of the "
                f"same grid and this launch binds both answers")


def launch_complex_no_pml_offdiag_fused_electric_pair(
        fields: Any, scratch: Dict[str, Any], codes: Sequence[int],
        wall_mask: Sequence[int], phases: Dict[str, Any],
        weights: Sequence[float], fills: Dict[str, Tuple[Any, ...]], dtdx: float,
        row_mask: Sequence[int], rows: Sequence[Any], expansion: Any,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch. THE CALLER ROTATES AFTERWARDS.

    ``kernel`` IS THE GATE'S DOOR: a gate compiles a deliberately broken copy of
    the shipped source and hands it here.
    """
    if word_view is None:
        raise RuntimeError(
            "complex_pml_kernels is not importable on this host, so the complex "
            "volumes cannot be bound as their float32 word views")
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    arguments: List[Any] = [word_view(scratch[name]) for name in SCRATCH_VOLUMES]
    arguments += [word_view(getattr(fields, name)) for name in _FIELD_BINDINGS]
    arguments += [fields.inverse_epsilon_for(component)
                  for component in _ELECTRIC]
    arguments += list(rows)
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx)]
    arguments += [np.int32(int(code)) for code in codes]
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
            "boundary_codes": tuple(int(code) for code in codes),
            "coupling_wall_mask": tuple(int(bool(flag)) for flag in wall_mask),
            "ghost_weights": tuple(float(value) for value in weights),
            "phase_flags": tuple(int(flag) for flag in phases["flags"]),
            "fills": {key: tuple(fills[key]) for key in
                      ("near", "wall", "reflect", "near_phase", "far_phase")}}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION -- the certified E->P choreography, one seam earlier.

    Three volumes rather than six; the ones handed back are the RETIRED ones, which
    the caller may reuse as the next launch's scratch.
    """
    retired: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        retired[name] = getattr(fields, name)
        setattr(fields, name, scratch[name])
    return retired


def run_complex_no_pml_offdiag_fused_electric_pair(
        fields: Any, grid: Any, pml: Any, dtdx: float, expansion: Any, *,
        scratch: Optional[Dict[str, Any]] = None, sources: Any = None,
        license: Any = None, subnormal_policy: Any = None,
        kernel: Optional[Any] = None, rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once, rotate."""
    covered, reason = covers_complex_no_pml_offdiag_fused_electric_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if scratch is None:
        scratch = complex_no_pml_offdiag_fused_electric_pair_scratch(fields)
    assert_scratch_is_disjoint(fields, scratch)
    codes = complex_no_pml_offdiag_fused_electric_pair_codes(grid)
    phases = complex_no_pml_offdiag_fused_electric_pair_phases(grid)
    wall_mask = offdiag_wall_mask_flags(grid)
    weights = _offdiag.mirror_ghost_weights(grid)
    fills = _weld.fill_plan(grid)
    _validate_launch_arguments(codes, wall_mask, fills)
    mask = _offdiag.normalized_row_mask(offdiag_row_mask(fields))
    rows = [volume for volume in offdiag_row_volumes(fields) if volume is not None]
    record = launch_complex_no_pml_offdiag_fused_electric_pair(
        fields, scratch, codes, wall_mask, phases, weights, fills, dtdx, mask, rows,
        expansion, kernel)
    record["rotated"] = bool(rotate)
    record["scratch"] = rotate_into_fields(fields, scratch) if rotate else scratch
    return record


#: Read by the host suite: the wall-clear planner this family and the shared lift
#: both consult, so a test can assert the identity rather than diff two spellings.
_ZERO_METAL_REFERENCE = zero_metal_axes
_COVERAGE_REFERENCE = _coverage
