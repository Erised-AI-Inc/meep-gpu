"""The SPECIAL_KZ hand-CUDA fused pair: all five passes of the folded magnetic seam.

THE BOARD'S CELL: ``B_to_H (cuda_special_kz/real beta, cuda_special_kz/real
beta)`` -- 2 seam-instances of demand, BOTH clearing the source seam since the
2026-09-02 residue round (``examples:refl-angular-kz2d.py`` carries nothing in
the seam at all; ``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag`` deposits its
lifted eigenmode sheets between the halves, which the deposit repair carries --
see :data:`CARRIES_DEPOSIT_REPAIR` -- and is folded, which the ported fill carry
serves). Both halves are CERTIFIED singles (``cuda_special_kz_2026-08-21``,
re-gated 2026-08-30); this module is the weld and introduces no new arithmetic.

WHAT THE CURL HALF IS: the certified ``step_B_special_kz_real`` -- the root real PML
curl body with MEEP's beta pass after the x and y components' curls
(``folded_pass.beta_insert``: ``-(c * g)`` on the partner's stored word and the folded
ladder on the word ``pml_apply`` just stored; ``grid.beta != 0`` under REAL storage,
MEEP's "implicitly store i*(TM fields)" trick, the coefficient host-rounded once by
``special_kz_curl.beta_curl_coefficients``). The pass rides through the register
capture: the x and y registers are the values the LADDER returns (``fold_apply_reg``),
because ``update_H`` and the fills must read B after both passes. The chunk record
``cb`` is decoded ONCE for both halves (bit_parity_0_10 DESIGN.md D10), and the run's
first ``update_H`` starts H from B on the array path (``fused_magnetic_pair.start_of_h``).

THE TWO MIRROR FILLS ARE CARRIED SINCE 2026-09-02, BY THE REAL PAIR'S OWN
MACHINERY. The fold refusal that stood here rested on "a cell whose one folded
row cannot clear the source seam anyway" -- and that premise fell with the
deposit-repair flag (the row's eigenmode sheets DO publish point indices; see
:data:`CARRIES_DEPOSIT_REPAIR`). The carry is IMPORTED from
:mod:`.fused_magnetic_pair` rather than re-spelled: the special_kz curl body is
the certified real curl body plus the two beta statements, so every anchor the
real pair's ownership inversion splices on -- the ``pml_apply`` calls, the
decode line, the certified ``update_H`` statements -- is byte-identical here,
and the beta statements ride through the capture untouched. One spelling of the
carry, two real-storage products; where the certified sources genuinely share
the spelling, so do the welds.

THE ALIASING RULE IS THE SIBLINGS': B is bound EXACTLY ONCE.

NOTHING HERE IS DISPATCH; :mod:`.fused_pairs` can plan it opt-in only.
"""

from __future__ import annotations

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

# THE CERTIFIED TEXT AND THE SHARED CARRY. ``special_kz_curl`` is stdlib-only (it
# reads the shared real prelude off the sibling's source), so the curl half is
# importable everywhere; ``fused_magnetic_pair`` -- the real pair whose ownership
# inversion this weld splices -- takes its own CuPy-bound halves defensively and
# is importable everywhere too; ``constitutive_kernels`` imports CuPy at module
# scope and is taken defensively.
try:
    from . import special_kz_curl as _special_kz
    from . import fused_magnetic_pair as _real_pair
    from . import bounds_block
    from . import folded_pass as _folded_pass
    from .coverage import real_curl_boundary_codes
    from .in_seam_coverage import (covers_fill_folded_far, covers_fill_symmetry,
                                   zero_metal_axes)
    from . import compile_cache
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.machinery as _machinery
    import importlib.util as _importlib_util
    import os as _os
    import sys as _sys

    def _load(stem):
        """By-path loader with a synthetic package, so the loaded siblings'
        own relative imports resolve too."""
        here = _os.path.dirname(_os.path.abspath(__file__))
        package = "cuda_kernels_bypath"
        if package not in _sys.modules:
            spec = _machinery.ModuleSpec(package, None, is_package=True)
            shim = _importlib_util.module_from_spec(spec)
            shim.__path__ = [here]
            _sys.modules[package] = shim
        name = f"{package}.{stem}"
        if name in _sys.modules:
            return _sys.modules[name]
        spec = _importlib_util.spec_from_file_location(
            name, _os.path.join(here, f"{stem}.py"))
        module = _importlib_util.module_from_spec(spec)
        _sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


    _special_kz = _load("special_kz_curl")
    _real_pair = _load("fused_magnetic_pair")
    bounds_block = _load("bounds_block")
    _folded_pass = _load("folded_pass")
    real_curl_boundary_codes = _load("coverage").real_curl_boundary_codes
    _in_seam_coverage = _load("in_seam_coverage")
    covers_fill_folded_far = _in_seam_coverage.covers_fill_folded_far
    covers_fill_symmetry = _in_seam_coverage.covers_fill_symmetry
    zero_metal_axes = _in_seam_coverage.zero_metal_axes
    compile_cache = _load("compile_cache")

try:
    from . import constitutive_kernels
except ImportError:  # a host with no CuPy: only the emitter is unavailable
    constitutive_kernels = None

try:
    from .. import deposit_repair as _deposit_repair
except ImportError:  # loaded by path, outside the package
    import importlib.util as _importlib_util
    import os as _os

    _spec = _importlib_util.spec_from_file_location(
        "meep_gpu_deposit_repair",
        _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "deposit_repair.py"))
    _deposit_repair = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_deposit_repair)


# =============================================================================
# THE PARTITION -- EMPTY ON THE CERTIFIED SIDE UNTIL A GATE MOVES IT
# =============================================================================

#: Byte-identical with a gate verdict AND a record behind it.
#: TWO RECORDS STAND BEHIND THIS ONE NAME AND THE LATER ONE IS THE LIVE BYTES'.
#: ``cuda_special_kz_fused_magnetic_pair_2026-09-02`` certified the THREE-PASS
#: weld; the residue round then ported the real pair's fill carry and the
#: deposit bracket into this module, which CHANGES the emitted device text, so
#: the five-pass weld was gated in its own right and recorded as
#: ``cuda_special_kz_fused_magnetic_pair_2026-09-02_fillcarry``. Neither
#: verdict is about the other's bytes, which is why both blocks exist -- the
#: same shape ``cuda_fused_magnetic_pair_2026-08-27`` and
#: ``..._2026-08-29_fillcarry`` already have. Both legs of the fillcarry run
#: RELEASED under BOTH float32 subnormal policies on the GPU host: 14/14 cases
#: weld- and array-identical per complete driver step, 40/40 live mutations
#: caught of 65 scored, the carry floor's legs on every folded fixture, and
#: four deposit legs -- the cell's blocked row's shape, an in-seam MAGNETIC
#: source carried through the fold closure, with the unrepaired null control
#: DIVERGING on both fixtures.
CERTIFIED_KERNELS = (
    "fused_magnetic_pair_special_kz_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: TRUE, RE-MEASURED 2026-09-02 (fusion-residue audit §1.3): the False that
#: stood here rested on the premise that ``eigsrc_kz_1_real_imag`` "deposits
#: through an ``EigenModeSource``, which publishes no point index" -- FALSE of
#: the live tree. ``EigenModeSource`` never reaches the driver:
#: ``_lift_eigenmode_source`` re-runs MEEP's synthesis into equivalent-current
#: SHEETS deposited via ``add_source``, whose magnetic sheets are
#: ``VolumeSource`` objects, and every source class publishes
#: ``_point_ix/_point_iy/_point_iz`` plus per-cell amps at setup (only an empty
#: footprint yields ``None``). Re-measured on this tree before the flip: a
#: sheet magnetic ``VolumeSource`` with the lift's ``amp_func`` shape returns
#: index ARRAYS from ``deposit_repair._deposit_index`` and
#: ``seam_source_reasons(carries_repair=True)`` fires no publish-refusal. The
#: wiring is ``fused_pairs._install_fused_pair``'s -- the same bracket the
#: seven carrying products reach -- and ``repairable()`` decides per run.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_special_kz_fused_magnetic_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_magnetic_pair_special_kz_real"

#: The driver passes ONE launch performs, in driver order (driver.py:3291-3298).
#: The two mirror fills JOINED on 2026-09-02 with the ported carry; until then
#: they were refused by name and the cell's one folded row sat outside the
#: predicate.
REPLACES = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
            "fill_folded_far_ghosts_B", "update_H")

#: The sub-step slot the planner holds this on.
SLOT = "step_B"

#: Every line of certified device text this file did not lift verbatim. The
#: carry entries are :mod:`.fused_magnetic_pair`'s, shared verbatim -- one
#: spelling of the ownership inversion, two real-storage welds -- and the helper
#: entries are restated here because the lift leg checks each module's own
#: declaration against its own emitted source.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ float pml_apply_reg(... , int owned)",
     "why": "the constitutive half consumes the value in a register, and on a cell "
            "a fill images the SOURCE thread writes B -- so this thread must not "
            "read f[idx] either. The guard sits directly before the one load of f; "
            "the fu stage stays ABOVE it (step_B steps fu on every recurrence cell "
            "and neither fill touches it). Both forms below the guard and the store "
            "are the certified ones; the value is additionally returned."},
    {"line": "__device__ __forceinline__ void fold_apply(",
     "became": "__device__ __forceinline__ float fold_apply_reg(... , int owned)",
     "why": "the folded ladder, derived the same way from its own text "
            "(folded_pass.fold_apply_reg_source): the same guard above its one load "
            "of f, its fu stage above the guard, and the value returned. The plain "
            "helper leaves the prelude: no call is left."},
    {"line": "        pml_apply(Bx, fu_Bx, idx, curl, ...);\n"
             "        ...\n        fold_apply(Bx, fu_Bx, idx, inc_beta, ...);",
     "became": "        b_x = pml_apply_reg(Bx, fu_Bx, idx, curl, ..., own_x);\n"
               "        ...\n        b_x = fold_apply_reg(Bx, fu_Bx, idx, inc_beta, ..., "
               "own_x);",
     "why": "capture the register OWNED; argument lists otherwise untouched, the "
            "beta pass between them untouched (Bz has no beta pass). Three "
            "'float b_* = 0.0f;' declarations and the ownership flags are hoisted "
            "above the certified braced blocks."},
    {"line": "    constitutive_apply(Hx, f_w_Hx, idx, Bx[idx], kps_x[i], kms_x[i], "
             "in_i_x);",
     "became": "    if (own_x) { constitutive_apply(Hx, f_w_Hx, idx, b_x, kps_x[i], "
               "kms_int_x[i], in_i_x); <ghost blocks> }",
     "why": "THE SEAM plus the sub-lattice rename, the ownership guard and the "
            "ghost carry -- fused_magnetic_pair.certified_constitutive_body, "
            "imported: half-integer for the B curl, integer for the H "
            "constitutive."},
    {"line": "        array[_face(axis, 0)] = 0   (stepping._zero_metal)",
     "became": "    if (own_x && wall_x && i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }",
     "why": "zero_metal_B carried on the registers, GUARDED on own_* -- launch "
            "geometry, not arithmetic. Same B diagonal, same stored cell 0, "
            "same +0.0f; a cell a fill images belongs to its source thread."},
    {"line": "    field[_face(axis, 0)] = phase * field[_face(axis, 2)]   "
             "(stepping._write_mirror_ghost)",
     "became": "the SOURCE thread at stored 2 writes stored 0 from its register",
     "why": "THE OWNERSHIP INVERSION, imported from fused_magnetic_pair: the "
            "destination thread would have to read a word another block writes "
            "in this launch; the source thread already holds it. Same parity, "
            "same destination cell, same order relative to the wall clear."},
    {"line": "    array[_face(axis, -1)] = parity * array[_face(axis, reflect)]   "
             "(stepping._fill_folded_far_ghosts)",
     "became": "the SOURCE thread at the reflect row writes the top plane",
     "why": "The same inversion for the far fill, weight -phase against the "
            "near fill's +phase, applied AFTER the wall clear -- the driver's "
            "own order."},
    {"line": "    <step_B_special_kz_real's chunk-record block>\n"
             "    <update_H_pml_real's chunk-record block>",
     "became": "    <ONE block, fused_magnetic_pair.union_bounds_block()>",
     "why": "ONE DECODE OF THE RECORD PER KERNEL (DESIGN.md D10): both lifted bodies' "
            "blocks are removed by bounds_block.strip_bounds_block and the real pair's "
            "union block is emitted directly after the decode. A ghost the carry "
            "steps takes the membership of the row its coefficient entry is read at "
            "(the destination's)."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body",
    "certified_curl_body", "covers_special_kz_fused_magnetic_pair",
    "device_sources", "launch_special_kz_fused_magnetic_pair",
    "run_special_kz_fused_magnetic_pair",
    "special_kz_fused_magnetic_pair_fills",
    "special_kz_fused_magnetic_pair_source",
    "special_kz_fused_magnetic_pair_tables", "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================

_GLOBAL_MARKER = 'extern "C" __global__ void '
_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"
#: The certified helper's anchors and the OWNED rewrites -- the real pair's own four
#: (signature, parameter list, the guard above the one load of f, the returned store),
#: read from :mod:`.fused_magnetic_pair` so the two real-storage welds cannot drift
#: apart on the guard. Applied to ``pml_apply``'s own text only: the folded ladder ends
#: on the same store line.
_PML_APPLY_SIGNATURE = _real_pair._PML_APPLY_SIGNATURE
_PML_APPLY_OWNED_EDITS: Tuple[Tuple[str, str, str], ...] = (
    (_real_pair._PML_APPLY_SIGNATURE, _real_pair._PML_APPLY_SIGNATURE_REG,
     "its signature"),
    (_real_pair._PML_APPLY_PARAMETERS, _real_pair._PML_APPLY_PARAMETERS_OWNED,
     "its parameter list"),
    (_real_pair._PML_APPLY_FCUR, _real_pair._PML_APPLY_FCUR_OWNED,
     "its one load of the field"),
    (_real_pair._PML_APPLY_STORE, _real_pair._PML_APPLY_STORE_REG, "its closing store"),
)

_CARRIED: Tuple[str, ...] = ("x", "y", "z")

#: The fused entry point: the certified special_kz step_B signature with the H
#: group, the INTEGER constitutive tables, the wall flags, the beta scalars and
#: the two fills' runtime plan appended. B appears EXACTLY ONCE. The fills tail
#: is the real pair's own spelling (:mod:`.fused_magnetic_pair`), because the
#: carry it feeds is imported from there.
_SIGNATURE = r'''
extern "C" __global__ void fused_magnetic_pair_special_kz_real(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the
    // constitutive half reads it from a register; a second const __restrict__
    // binding of the same allocation would be UB NVRTC miscompiles silently.
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
    float* __restrict__ f_w_Hx, float* __restrict__ f_w_Hy,
    float* __restrict__ f_w_Hz,
    int nx, int ny, int nz, float dtdx,
    // The B curl's HALF-INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The H constitutive's INTEGER coefficients, its kms renamed kms_int_*.
    const float* __restrict__ kps_x, const float* __restrict__ kms_int_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_int_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_int_z,
    int bc_x, int bc_y, int bc_z,
    // The two host-rounded beta coefficients, one per call-site sign -- the
    // certified special_kz signature's own insert.
    float beta_plus, float beta_minus,
    // zero_metal_B's walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z,
    // THE TWO FILLS' RUNTIME PLAN, from in_seam_coverage.mirror_fill_phases and
    // .folded_far_rows -- the real pair's own tail. `near_a` is 1 on a
    // MIRROR-folded axis, where fill_symmetry_bc_B images stored 0 from stored
    // 2. `reflect_a` is fill_folded_far_ghosts_B's image row on a folded
    // PERIODIC axis and -1 where that pass does not run. `phase_a` is the
    // plane's declared mirror phase; the near fill weights by +phase and the
    // far fill by -phase.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float phase_x, float phase_y, float phase_z,
    // THE CHUNK RECORD (chunk_map.chunk_bounds_record), int32[25], decoded once.
    const int* __restrict__ cb
) {
'''


def _split_certified(source: str, name: str) -> Tuple[str, str]:
    """``(prelude, body)`` of one certified kernel string, or a named failure."""
    if source.count(_GLOBAL_MARKER) != 1:
        raise AssertionError(
            f"{name} carries {source.count(_GLOBAL_MARKER)} global entry points, "
            f"not 1; the prelude cannot be identified")
    prelude, rest = source.split(_GLOBAL_MARKER, 1)
    if rest.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {rest.count(_BODY_ANCHOR)} signature terminators, "
            f"not 1; the body anchor no longer identifies the signature")
    body = rest.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return prelude, body[: -len("}\n")]


def special_kz_fused_magnetic_pair_prelude() -> str:
    """The certified special_kz prelude (read through the family's own string, so a
    gate mutation there reaches this weld) with ``pml_apply`` and the folded ladder
    turned into OWNED values -- the real pair's four rewrites -- plus the untouched real
    constitutive prelude."""
    if constitutive_kernels is None:
        raise RuntimeError(
            "constitutive_kernels is not importable on this host (it imports "
            "CuPy at module scope), so there is no certified constitutive text "
            "to splice; the predicate needs none of it and still answers")
    prelude, _body = _split_certified(
        _special_kz.kernel_source("step_B_special_kz_real"),
        "step_B_special_kz_real")
    prelude = _folded_pass.rewrite_helper(prelude, _PML_APPLY_SIGNATURE,
                                          _PML_APPLY_OWNED_EDITS, "pml_apply")
    prelude = _folded_pass.with_fold_reg(prelude, owned=True)
    return prelude + constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE


def certified_curl_body() -> str:
    """``step_B_special_kz_real``'s body, captured OWNED via the real pair's carry.

    The registers, the ownership flags and the per-call ``own_*`` argument are the real
    pair's own splice (:mod:`.fused_magnetic_pair`); the beta passes are INSIDE the
    lifted body and ride through, their ladders captured into the same registers -- the
    splice asserts both arrived, because a curl body without them would compile and be
    the plain real pair under this family's name.
    """
    _prelude, body = _split_certified(
        _special_kz.kernel_source("step_B_special_kz_real"),
        "step_B_special_kz_real")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the "
            "carried registers have nowhere to be declared")
    body = bounds_block.strip_bounds_block(body, "step_B_special_kz_real",
                                           *_real_pair.CURL_BLOCK)
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END, "\n",
        _real_pair.union_bounds_block(),
        "    // The three carried registers. Declared HERE because each\n"
        "    // certified component below is a braced scope, and a value declared\n"
        "    // inside one does not outlive it.\n"
        "    float b_x = 0.0f;\n"
        "    float b_y = 0.0f;\n"
        "    float b_z = 0.0f;\n"
        "\n",
        _real_pair.ownership_declarations(),
        tail,
    ))
    for axis in _CARRIED:
        old = f"pml_apply(B{axis}, fu_B{axis}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} "
                f"{body.count(old)} times, not once; the capture has no anchor")
        body = body.replace(old, f"b_{axis} = pml_apply_reg(B{axis}, fu_B{axis}, ", 1)
        prefix = f"        b_{axis} = pml_apply_reg(B{axis}, fu_B{axis}, "
        call = _real_pair._line_starting(
            body, prefix, f"B{axis}'s captured pml_apply")
        if not call.endswith(");"):
            raise AssertionError(
                f"the certified curl body no longer closes B{axis}'s pml_apply "
                f"on one line ({call!r}); the ownership flag has no anchor to be "
                f"appended at")
        body = body.replace(call, f"{call[:-2]}, own_{axis});", 1)
    if "pml_apply(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be "
            "written to global memory and never read into the seam")
    body = _folded_pass.capture_fold_calls(
        body, {f"B{axis}, fu_B{axis}, ": (f"b_{axis}", f"own_{axis}")
               for axis in _CARRIED},
        "fold_apply", "fold_apply_reg", "the special_kz fused magnetic pair")
    if (body.count("float inc_beta = ") != 2
            or body.count("b_x = fold_apply_reg(") != 1
            or body.count("b_y = fold_apply_reg(") != 1):
        raise AssertionError(
            "the captured special_kz curl body no longer carries both beta passes "
            "captured into the x and y registers; the lift would ship the plain real "
            "pair under this family's name")
    return body


def zero_metal_carry() -> str:
    """``zero_metal_B`` on the registers, GUARDED on ``own_*`` -- the real
    pair's own emitter, imported rather than re-spelled so the two real-storage
    welds cannot drift apart on the carry."""
    return _real_pair.zero_metal_carry()


def certified_constitutive_body() -> str:
    """``update_H_pml_real``'s body with the seam, the rename, the ownership
    guard and the ghost carry -- the real pair's own lift, imported. The
    special_kz constitutive IS the certified ordinary ``update_H`` (beta enters
    the curls only, stepping.py:384-391), so the real pair's constitutive splice
    is byte-for-byte this family's too."""
    return _real_pair.certified_constitutive_body()


def special_kz_fused_magnetic_pair_source() -> str:
    """The whole fused kernel. PURE ASCII."""
    source = "".join((
        special_kz_fused_magnetic_pair_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        "\n",
        zero_metal_carry(),
        "\n    // --- update_H (stepping.update_H / _apply_constitutive_pml) ----\n"
        "    // Its three sources are the registers above, not a reload of B,\n"
        "    // and each runs again at every ghost cell this thread owns.\n",
        certified_constitutive_body(),
        "}\n",
    ))
    try:
        source.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AssertionError(
            f"the fused device source is not pure ASCII ({exc}); NVRTC's source "
            f"file is written through the locale encoding") from exc
    return source


def device_sources() -> Dict[str, str]:
    """The shipped device text by kernel name -- what a record block would pin."""
    return {KERNEL_NAME: special_kz_fused_magnetic_pair_source()}


_COMPILE_OPTIONS = ('--fmad=false',)
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str = KERNEL_NAME):
    """Compile on first use, memoized on (name, options, policy, source); the
    source is rebuilt per call because a gate mutates the family's strings."""
    if name != KERNEL_NAME:
        raise ValueError(f"this module ships one kernel, {KERNEL_NAME!r}, "
                         f"not {name!r}")
    if cp is None:
        raise RuntimeError(
            f"{KERNEL_NAME} cannot be compiled here: CuPy is not importable on "
            f"this host. The emitter and the predicate need no device and still "
            f"run; a LAUNCH does")
    code = special_kz_fused_magnetic_pair_source()
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_special_kz_fused_magnetic_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span the five passes of :data:`REPLACES`?

    A CONJUNCTION over the special_kz family's two certified predicates -- which
    REQUIRE ``grid.beta != 0`` under real storage, the clause that keeps this
    product disjoint from the real pair (refuses beta) and from every complex
    product (real storage) -- plus the seam clauses: the source slot (carried,
    :data:`CARRIES_DEPOSIT_REPAIR`) and the fill clauses the real pair's carry
    states (:mod:`.fused_magnetic_pair`'s own set, restated here so the
    guarantee does not rest on a clause another module could widen).
    """
    covered, reason = _special_kz.covers_special_kz_curl(
        fields, pml, grid, "step_B")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _special_kz.covers_special_kz_constitutive(
        fields, pml, grid, "H")
    if not covered:
        return False, f"constitutive half: {reason}"

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

    # THE TWO FILLS, CARRIED SINCE 2026-09-02 by the real pair's ownership
    # inversion. What the carry does NOT reach is refused BY NAME, through the
    # fills' own predicates -- `in_seam_coverage` is the single place each
    # fill's clauses live, so a clause added there reaches this seam without a
    # second edit.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_B and "
                       "fill_folded_far_ghosts_B run inside it")
    covered, reason = covers_fill_symmetry(fields, grid, "B")
    if not covered:
        return False, (f"fill_symmetry_bc_B runs inside this seam "
                       f"(driver.py:3294) and this carry cannot serve it: {reason}")
    covered, reason = covers_fill_folded_far(fields, grid, "B")
    if not covered:
        return False, (f"fill_folded_far_ghosts_B runs inside this seam "
                       f"(driver.py:3296) and this carry cannot serve it: {reason}")
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry != any(folded):
        return False, ("the grid reports has_symmetry() and no mirrored axis, or "
                       "the reverse; this carry is decided per axis and cannot be "
                       "read off a grid that disagrees with itself")

    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_B cannot be "
                           f"carried in registers")
    try:
        walls = zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE THREE CLAUSES THE CARRY ADDS, the real pair's own list restated.
    for axis in range(3):
        if folded[axis] and bool(walls[axis]):
            return False, (
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction and "
                f"the near carry writes no wall line at the ghost it images")
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Bx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks "
                       f"Bx.shape {extents}; the wall plane and the two fills' "
                       f"rows are derived from the first and indexed into the "
                       f"second")
    for axis in range(3):
        if folded[axis] and extents[axis] <= _real_pair.NEAR_SOURCE_INDEX:
            return False, (
                f"folded axis {axis} stores {extents[axis]} cells, so the near "
                f"fill's source row {_real_pair.NEAR_SOURCE_INDEX} does not "
                f"exist (stepping._mirror_source raises on it)")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

_FIELD_BINDINGS: Tuple[str, ...] = (
    "Bx", "By", "Bz",
    "fu_Bx", "fu_By", "fu_Bz",
    "Ex", "Ey", "Ez",
    "Hx", "Hy", "Hz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def special_kz_fused_magnetic_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads."""
    if constitutive_kernels is None:
        raise RuntimeError(
            "constitutive_kernels is not importable on this host; the tables "
            "cannot be resolved without it")
    from . import step_curl_kernels  # noqa: PLC0415 - device-only module
    return {
        "curl": step_curl_kernels.real_pml_curl_tables(pml, True),
        "constitutive": constitutive_kernels.real_constitutive_tables(pml, False),
    }


def beta_scalars(grid: "Grid") -> Tuple[float, float]:
    """``(beta_plus, beta_minus)``, host-rounded once by the certified family's
    own helper, so the weld and the single cannot round differently."""
    return _special_kz.beta_curl_coefficients(float(grid.beta), float(grid.dt))


def assert_disjoint_bindings(fields: "Fields",
                             tables: Dict[str, Dict[str, Any]]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes."""
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in _FIELD_BINDINGS:
        visit(name, getattr(fields, name))
    for key in _CURL_TABLE_KEYS:
        visit(f"curl:{key}", tables["curl"][key])
    for key in _CONSTITUTIVE_TABLE_KEYS:
        visit(f"constitutive:{key}", tables["constitutive"][key])
    if collisions:
        raise ValueError(
            "the special_kz fused magnetic pair binds every argument "
            "__restrict__, and these arguments alias, which is undefined "
            "behaviour NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def special_kz_fused_magnetic_pair_fills(grid: "Grid") -> Dict[str, Tuple[Any, ...]]:
    """The two fills' launch plan -- the real pair's own resolver, imported."""
    return _real_pair.fused_magnetic_pair_fills(grid)


def launch_special_kz_fused_magnetic_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], walls: Sequence[int],
        beta: Sequence[float], fills: Dict[str, Sequence[Any]], dtdx: float,
        kernel: Optional[Any] = None, *, chunk_bounds: Any,
        layer: Any = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch. ``kernel`` is the gate's door.

    ``chunk_bounds`` is the run's chunk record on the device, required
    (``step_curl_kernels.chunk_record_argument``). The run's first ``update_H`` starts
    H from B: when ``fields.h_separated`` is still False the launch is followed by
    ``fused_magnetic_pair.start_of_h`` with ``layer``, and a first launch handed no
    ``layer`` is refused by name before it launches.
    """
    from . import step_curl_kernels  # noqa: PLC0415 - device-only module

    cb = step_curl_kernels.chunk_record_argument(chunk_bounds)
    first = not bool(getattr(fields, "h_separated", True))
    if first and layer is None:
        raise RuntimeError(
            "the run's first update_H starts H from B (stepping.separate_h_from_b) and "
            "this fused magnetic launch was handed no layer to start it with")
    nx, ny, nz = (int(n) for n in fields.Bx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    plus, minus = beta
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + (
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
        np.float32(plus), np.float32(minus),
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3)
    ) + tuple(np.int32(int(fills["near"][axis])) for axis in range(3)) + tuple(
        np.int32(int(fills["reflect"][axis])) for axis in range(3)
    ) + tuple(np.float32(float(fills["phase"][axis])) for axis in range(3)) + (cb,)
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), arguments)
    if first:
        _real_pair.start_of_h(fields, layer)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES, "h_started": first,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "near": tuple(int(v) for v in fills["near"]),
            "reflect": tuple(int(v) for v in fills["reflect"]),
            "phase": tuple(float(v) for v in fills["phase"]),
            "beta_scalars": [float(plus), float(minus)]}


def run_special_kz_fused_magnetic_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        chunk_bounds: Any, sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once."""
    covered, reason = covers_special_kz_fused_magnetic_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = special_kz_fused_magnetic_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    # coverage.real_curl_boundary_codes is the FAIL-CLOSED face -- (codes,
    # refusal) -- and the predicate above consulted the same resolver, so the
    # launcher and the predicate cannot answer differently about one grid.
    codes, refusal = real_curl_boundary_codes(grid)
    if refusal is not None:
        return {"launched": False, "reason": refusal}
    return launch_special_kz_fused_magnetic_pair(
        fields, tables, tuple(np.int32(code) for code in codes),
        zero_metal_axes(grid), beta_scalars(grid),
        special_kz_fused_magnetic_pair_fills(grid), dtdx, kernel,
        chunk_bounds=chunk_bounds, layer=pml)
