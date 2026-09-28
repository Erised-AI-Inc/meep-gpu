"""The FOLDED-COMPLEX hand-CUDA fused pair: one launch for the whole magnetic seam.

THE BOARD'S LARGEST REMAINING BUILDABLE CELL:
``B_to_H (cuda_complex_folded/folded complex, cuda_complex/complex)`` -- 5
seam-instances of demand, 4 clearing the source seam, with ``fill_symmetry_bc_B``
live on all 4 and ``fill_folded_far_ghosts_B`` on 2
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-01_certclose/``). Both
halves are CERTIFIED singles as of 2026-09-01 (``cuda_complex_folded_2026-09-01``,
``complex_pml_2026-08-26``); this module is the weld and introduces no new
arithmetic.

ONE LAUNCH PERFORMS FIVE DRIVER PASSES, in driver order (driver.py:3291-3298)::

    step_B -> fill_symmetry_bc_B -> zero_metal_B -> fill_folded_far_ghosts_B
           -> update_H

which is the OWNERSHIP-INVERSION PORT the real pair made on 2026-08-28
(:mod:`.fused_magnetic_pair`), applied to COMPLEX STORAGE: the source thread
writes every imaged ghost from a live ``cf`` register, the destination thread is
carved out, no thread reads a word another writes, and no grid-wide barrier is
needed or used. The cf machinery -- the parity chain, the ownership flags, the
guarded ``pml_apply_reg`` -- lives in :mod:`.complex_fill_carry`, ONCE, because
:mod:`.complex_beta_fused_magnetic_pair` splices the same folded curl template
(the beta source IS the folded source plus the beta insert) and two spellings of
one carry is how they would come to differ.

WHAT IS LIFTED AND FROM WHERE. The curl half is
``complex_folded_kernels.folded_source("step_B", arm)`` -- the certified folded
complex curl, whose fold masks (``BC_MIRROR_PERIODIC`` at cell 0 and at the last
stored slot) ride through the capture untouched. The constitutive half is
``complex_emitter.complex_source("update_H", arm)`` -- the certified complex
constitutive the fold ADMITS directly
(``coverage.COMPLEX_CONSTITUTIVE_FOLD_ADMISSION``, 2026-08-20). The splice
anchors every certified line it edits and raises on a moved one.

THE MASKS AND THE CARRY COMPOSE, AND THE COMPOSITION IS THE ARRAY PATH'S OWN.
The folded curl's masks zero the CURL at exactly the cells the two fills image
(cell 0 of a folded axis; the last stored slot of a folded periodic one), so on
both paths the split-field auxiliary at those cells advances from a zero curl --
the weld's ``fu`` recurrence runs UNGUARDED above the ownership return, seeing
the same masked curl -- while the field word the recurrence would store there is
discarded on both paths: the array path's fill overwrites it, and this kernel's
owner thread never forms it.

THE ALIASING RULE IS THE SIBLINGS': B is bound EXACTLY ONCE (the constitutive
half has no source pointers -- it reads the registers), because two
``__restrict__`` pointers to one allocation is UB NVRTC miscompiles silently.

A FOLDED AXIS CARRIES NO BLOCH PHASE, twice over: the certified folded curl
predicate refuses one by name (``complex_folded_kernels._fold_reasons``), and the
carry's parity weights are therefore the bare ``+-phase`` -- the fills apply the
plane parity and never the wrap factor (``stepping._fill_folded_far_ghosts``
docstring: "no Bloch factor (a folded axis refuses a nonzero k outright)").

NOTHING HERE IS DISPATCH. ``fastpath.plan_fast_path`` returns ``None`` on every
branch; :mod:`.fused_pairs` can plan this opt-in (``arms.plan_step(...,
fuse=True)``) and no shipped code path launches it.
"""

from __future__ import annotations

try:  # a host with no CuPy: the emitter and the predicate still run
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

# THE CERTIFIED TEXT AND THE SHARED CARRY. All CuPy-free, so the emitter and the
# predicate run on the census laptop; the by-path fallback serves the bit-identity
# probe, which loads these files outside the package.
try:
    from . import complex_emitter
    from . import complex_fill_carry as _carry
    from . import complex_folded_kernels as _folded
    from .coverage import covers_real_pml_complex_constitutive
    from .in_seam_coverage import zero_metal_axes
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

    complex_emitter = _load("complex_emitter")
    _carry = _load("complex_fill_carry")
    _folded = _load("complex_folded_kernels")
    covers_real_pml_complex_constitutive = \
        _load("coverage").covers_real_pml_complex_constitutive
    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")

# The CuPy-BOUND launch helpers, taken separately and defensively so the
# predicate and the emitter above stay importable on the census laptop; a launch
# refuses by name when they are absent.
try:
    from .complex_pml_kernels import (bloch_phase_arguments,
                                      complex_constitutive_tables,
                                      complex_curl_tables, word_view)
except ImportError:  # a host with no CuPy: only the launch is unavailable
    bloch_phase_arguments = complex_constitutive_tables = None
    complex_curl_tables = word_view = None

# The source-seam clause, shared with every fused-pair predicate on all three
# tracks; imported rather than re-spelled (a per-product copy once asked about the
# wrong seam on Metal).
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
#
# Plain assignments with NO type annotation: the partition readers walk the syntax
# tree without importing the module, and an ``ast.AnnAssign`` is invisible to them.

#: Byte-identical to the array path with a gate verdict AND a ``certification.json``
#: record behind it. A fused product's bit-identity is a claim about the WELD, which
#: no verdict on either half establishes, so this stays empty until the fused gate
#: has run on a device.
#: RECORDED 2026-09-02 as ``cuda_complex_folded_fused_magnetic_pair_2026-09-02``
#: in ``certification.json``. Both legs RELEASED under BOTH float32 subnormal
#: policies on the GPU host: 16/16 cases weld- and array-identical per complete
#: driver step, 63/63 live mutations caught (the cf fill carry's own needles
#: included), the carry floor's degenerate and no-carry legs diverging on every
#: folded fixture, and the in-seam deposit carried through the fold closure with
#: its null control diverging.
CERTIFIED_KERNELS = (
    "fused_magnetic_pair_pml_complex_folded",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its fused launch with the deposit repair? TRUE, wired
#: through ``fused_pairs._install_fused_pair`` exactly as the siblings are (this
#: family has a row in ``fused_pairs.FUSED_PRODUCTS``); the flag and the wiring
#: move together or not at all (``deposit_repair.py``). THE FLAG IS LOAD-BEARING
#: ON THIS CELL, measured off the certclose census: four of the five rows are
#: electric-only sourced and clear on the driver fact alone, and the fifth --
#: ``TestHoleyWvgBands.test_fields_at_kx`` -- deposits a plain point ``Source``
#: with ``source_field_types == ['B']`` between the halves, which is exactly
#: what the bracket reconstructs. ON THIS CELL THE FOLD CLOSURE IS LIVE: a
#: deposit on a row a fill reads from is imaged into cells a point repair never
#: visits, and ``deposit_repair.repair_cells`` hands the two plans that closure
#: -- complex storage measured by ``test_deposit_repair.COMPLEX_CASES``, the
#: fold closure by the real pair's gate. What the shared clause still refuses BY
#: NAME: a source that does not publish its point index, and an unallocated
#: ``f_w_H*``.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_complex_folded_fused_magnetic_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_magnetic_pair_pml_complex_folded"

#: The driver passes ONE launch performs, in driver order (driver.py:3291-3298).
#: THE TWO FILLS ARE IN IT, carried by the ownership inversion; declared rather
#: than inferred.
REPLACES = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
            "fill_folded_far_ghosts_B", "update_H")

#: The sub-step slot the planner holds this on.
SLOT = "step_B"

#: Every line of certified device text this file did not lift verbatim, with the
#: reason -- DATA, so a gate can assert the list rather than a docstring. The
#: rewrites are :mod:`.complex_fill_carry`'s and are shared verbatim with the beta
#: weld; entries 1-4 are the complex pair's own edits with the ownership flag
#: added, 5-6 are the two carried passes.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ cf pml_apply_reg(... , int owned)",
     "why": "the constitutive half consumes the value in a register, and on a "
            "cell a fill images the SOURCE thread writes f -- so this thread "
            "must not read f[idx] either. The fu recurrence stays ABOVE the "
            "guard: step_B writes fu at every cell and neither fill touches it "
            "(stepping.py:1451). The expression, its parenthesisation and the "
            "store below the guard are the certified ones."},
    {"line": "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), "
             "fprev), sinv_u));",
     "became": "    if (!owned) return cf_zero();\n    cf a = ...;\n    cf value "
               "= mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u);\n"
               "    cf_store(f, idx, value);\n    return value;",
     "why": "same right-hand side, same tree, same store; the value is "
            "additionally NAMED so the constitutive half can read it from a "
            "register, and the ownership guard sits ABOVE the field load so a "
            "destination thread never reads a word another block writes. A "
            "float32 word pair stored to global and reloaded is the identity "
            "on the bits."},
    {"line": "        pml_apply(f0, u0, idx, curl, ...);",
     "became": "        b0 = pml_apply_reg(f0, u0, idx, curl, ..., own_0);",
     "why": "capture the register under the ownership flag; three lines, one per "
            "component, argument lists untouched. Three 'cf b* = cf_zero();' "
            "declarations are hoisted above the certified braced blocks."},
    {"line": "    cf s0 = cf_load(g0, idx);",
     "became": "    cf s0 = b0;",
     "why": "THE SEAM. The certified constitutive body opens each component with "
            "a reload of the flux density the curl just stored; this reads the "
            "register instead, and removes the constitutive half's only use of "
            "g0/g1/g2 -- which is what lets B be bound exactly once."},
    {"line": "    constitutive_apply(f0, w0, idx, s0, kps_x[i], kms_x[i]);",
     "became": "    if (own_0) { constitutive_apply(h0, w0, idx, s0, kps_x[i], "
               "kms_int_x[i]); <ghost blocks> }",
     "why": "TWO RENAMES (f0 is B in the curl body and H here; kms_x is the "
            "HALF-INTEGER sub-lattice for the B curl and the INTEGER one for "
            "update_H), the ownership guard, and the ghost carry: a source "
            "thread runs the certified statement AGAIN at every ghost it owns, "
            "at the composed parity chain and -- for a near image, which moves "
            "the very axis update_H indexes -- at the DESTINATION's coefficient "
            "entry."},
    {"line": "        array[_face(axis, 0)] = 0   (stepping._zero_metal, :2206)",
     "became": "    if (own_0 && wall_x && i == 0) { b0 = cf_zero(); "
               "cf_store(f0, idx, b0); }",
     "why": "zero_metal_B carried on the registers -- launch geometry, not "
            "arithmetic. Same diagonal, same stored cell 0, same (+0.0f, +0.0f) "
            "word pair; guarded on own_* because a cell a fill images is written "
            "by its source thread (the array path's :3296 overwrites :3295 "
            "there)."},
    {"line": "field[dst] = phase * field[src]   (stepping.py:1451, :1529-1532)",
     "became": "cf g0_yn_v = mul_coefficient_left(phase_x, b0); g0_yn_v = "
               "mul_coefficient_left((phase_y * -1.0f), g0_yn_v); ...",
     "why": "THE TWO FILLS, carried. Each parity is ONE certified "
            "mul_coefficient_left -- the coefficient is np.multiply's left "
            "operand -- applied in the driver's own composition order (near "
            "innermost, then far ascending), because under complex storage the "
            "chain order and the zero cross terms both move bytes: measured on "
            "the GPU host 2026-09-01 (plain scale wrong on signed zeros; "
            "pre-multiplied parities wrong on 4 of 8216 words)."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body",
    "certified_curl_body", "complex_folded_fused_magnetic_pair_fills",
    "complex_folded_fused_magnetic_pair_prelude",
    "complex_folded_fused_magnetic_pair_source",
    "complex_folded_fused_magnetic_pair_tables",
    "covers_complex_folded_fused_magnetic_pair", "device_sources",
    "launch_complex_folded_fused_magnetic_pair",
    "run_complex_folded_fused_magnetic_pair",
]


# =============================================================================
# THE LIFT
# =============================================================================

#: Splits an emitted single-kernel source into (prelude, signature, body). The
#: prelude holds only ``__device__`` helpers, so the ONE ``extern "C" __global__``
#: marker separates it from the kernel; the signature closes at the one
#: ``"\n) {\n"``.
_GLOBAL_MARKER = 'extern "C" __global__ void '
_BODY_ANCHOR = "\n) {\n"


def _split_emitted(source: str, name: str) -> Tuple[str, str]:
    """``(prelude, body)`` of one emitted certified kernel, or a named failure."""
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


#: The fused entry point. The ONLY hand-written device text in this module, and it
#: is a signature: no arithmetic lives here. B (f0/f1/f2) appears EXACTLY ONCE.
_SIGNATURE = r'''
extern "C" __global__ void fused_magnetic_pair_pml_complex_folded(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 B.
    // The curl half writes it, the fills image it and the constitutive half
    // reads it from a register; a second const __restrict__ binding of the same
    // allocation would be UB NVRTC miscompiles without a diagnostic.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ u0, float* __restrict__ u1, float* __restrict__ u2,
    // E, the curl's operands.
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    // H and its constitutive history.
    float* __restrict__ h0, float* __restrict__ h1, float* __restrict__ h2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx,
    // The B curl's HALF-INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The H constitutive's INTEGER coefficients, its kms renamed kms_int_*: the
    // two sub-lattices collide on the bare name in one scope, and a shadow there
    // is a half-cell error in the absorber profile, not a compile failure.
    const float* __restrict__ kps_x, const float* __restrict__ kms_int_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_int_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_int_z,
    // The FOLD-AWARE boundary codes (BC_MIRROR_PERIODIC included), from
    // complex_folded_kernels.folded_complex_boundary_codes.
    int bc_x, int bc_y, int bc_z,
    // The Bloch table. A folded axis never carries a phase (refused by name in
    // the curl half's own predicate); an unfolded periodic axis may.
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,
    // zero_metal_B's walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z,
    // THE TWO FILLS' RUNTIME PLAN, from complex_fill_carry.fills_plan. near_a is
    // 1 on a MIRROR-folded axis (either termination), where fill_symmetry_bc_B
    // images stored 0 from stored 2. reflect_a is fill_folded_far_ghosts_B's
    // image row on a folded PERIODIC axis and -1 where that pass does not run --
    // a SENTINEL, so a source that read it anyway would index outside the volume
    // and be caught rather than silently imaging row 0. phase_a is the plane's
    // declared mirror parity; the near fill weights by +phase (imaged components
    // have Yee shift 0 there) and the far fill by -phase (theirs is 1).
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float phase_x, float phase_y, float phase_z
) {
'''


def complex_folded_fused_magnetic_pair_prelude(arm) -> str:
    """The certified FOLDED complex prelude with ``pml_apply`` owned and valued.

    The folded prelude differs from the plain complex one by exactly the two
    prelude deltas ``complex_folded_kernels.DELTAS`` spells (the third boundary
    code and the fold ghost's metallic zero); it is read off the emitted folded
    source rather than rebuilt, so this module cannot drift from the sibling's
    transform. ``constitutive_apply`` is emitted untouched.
    """
    prelude, _body = _split_emitted(
        _folded.folded_source("step_B", arm), "folded_source('step_B')")
    return _carry.pml_apply_reg_prelude(prelude, FAMILY)


def certified_curl_body(arm) -> str:
    """``step_B_pml_complex_folded``'s body, captured OWNED via the shared carry."""
    _prelude, body = _split_emitted(
        _folded.folded_source("step_B", arm), "folded_source('step_B')")
    return _carry.captured_curl_body(body, FAMILY)


def certified_constitutive_body(arm) -> str:
    """``update_H_pml_complex_bloch``'s statements, seamed, renamed and guarded.

    The certified complex constitutive is emitted by ``complex_emitter`` (the
    fold admits it directly -- ``COMPLEX_CONSTITUTIVE_FOLD_ADMISSION``); its
    duplicate index decomposition is dropped here (the curl body above already
    declares ``idx``/``i``/``j``/``k``) and the shared carry applies the seam,
    the renames, the ownership guards and the ghost blocks.
    """
    _prelude, body = _split_emitted(
        complex_emitter.complex_source("update_H", arm),
        "complex_source('update_H')")
    decode = "    int i = idx / (ny * nz);\n"
    if decode not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own "
            "line; the duplicate decomposition cannot be identified")
    tail = body.split(decode, 1)[1]
    return _carry.constitutive_body_with_carry(tail, FAMILY)


def complex_folded_fused_magnetic_pair_source(arm) -> str:
    """The whole fused kernel for one expansion arm. PURE ASCII (a compile
    requirement: NVRTC's source file goes through the locale encoding)."""
    source = "".join((
        complex_folded_fused_magnetic_pair_prelude(arm),
        _SIGNATURE,
        certified_curl_body(arm),
        "\n",
        _carry.zero_metal_carry(),
        "\n    // --- update_H (the certified complex constitutive; its sources\n"
        "    // are the registers above, not a reload of B) -------------------\n",
        certified_constitutive_body(arm),
        "}\n",
    ))
    try:
        source.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AssertionError(
            f"the fused device source is not pure ASCII ({exc}); NVRTC's source "
            f"file is written through the locale encoding and this would fail at "
            f"first launch, not at import") from exc
    return source


def device_sources() -> Dict[str, str]:
    """Every source this family can emit, keyed by arm name -- for a digest."""
    return {name: complex_folded_fused_magnetic_pair_source(name)
            for name in sorted(complex_emitter.EXPANSIONS)}


#: NVRTC compile options -- CORRECTNESS, identical to both certified halves'.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane.
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm, name: str = KERNEL_NAME):
    """Compile the fused kernel under one arm, memoized on (name, options, policy,
    source). THE SOURCE IS EMITTED PER CALL: it is part of the memo key, and a
    gate mutates this family by monkeypatching the emitters it splices from."""
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be "
            "compiled; the predicate and the emitter need no device and still run")
    code = complex_folded_fused_magnetic_pair_source(arm)
    key = compile_cache.kernel_cache_key(
        f"{name}_arm{complex_emitter.normalized_expansion(arm)}", True,
        _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_complex_folded_fused_magnetic_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None,
        license: Any = None, subnormal_policy: Any = None) -> Tuple[bool, str]:
    """May ONE launch span the five passes of :data:`REPLACES`?

    A CONJUNCTION, AND NOTHING IS WEAKENED. The curl half's certified predicate
    (``covers_complex_folded_curl`` -- which REQUIRES a fold, refuses a Bloch
    phase on a folded axis, and delegates every other clause to the certified
    complex curl) and the constitutive half's
    (``covers_real_pml_complex_constitutive``, whose fold admission is its own
    device verdict) are asked first with their own reasons prefixed. What this
    predicate ADDS is the seam: the source slot, the two fills' complex clauses,
    and the wall/stored-shape facts the carry rests on.

    DISJOINT FROM ITS THREE SEAM SIBLINGS BY CONSTRUCTION: the real pair refuses
    complex64 storage, the plain complex pair refuses every folded grid, the
    cylindrical pair requires Dcyl, and this one REQUIRES a fold (through the curl
    half) while inheriting the beta refusal that keeps it off the beta pair's
    rows. Two admitters on one seam would leave it unfused naming both.
    """
    covered, reason = _folded.covers_complex_folded_curl(
        fields, pml, grid, "step_B", license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "H", license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE SOURCE SEAM, CARRIED RATHER THAN REFUSED -- with the fold closure live:
    # a deposit on a row a fill reads from is imaged into cells a point repair
    # never visits, and deposit_repair.repair_cells is the closure that follows
    # the images. IGNORANCE IS NEVER AN EMPTY SET: Fields does not hold the
    # source list, so an undeclared one is refused rather than assumed clean.
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

    # THE TWO FILLS ARE CARRIED, so their complex clauses -- and the three facts
    # the carry adds -- are asked through the shared module. One spelling, shared
    # with the beta weld.
    fill = _carry.complex_fill_reasons(fields, grid)
    if fill is not None:
        return False, fill
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

#: The fifteen complex volumes, in SIGNATURE ORDER.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Bx", "By", "Bz",
    "fu_Bx", "fu_By", "fu_Bz",
    "Ex", "Ey", "Ez",
    "Hx", "Hy", "Hz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: The curl's HALF-INTEGER coefficient keys and the constitutive's INTEGER ones,
#: in signature order. The pairing is the opposite of the D/E seam's; a swap is a
#: half-cell error in the absorber profile, converged, smooth and wrong.
_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def complex_folded_fused_magnetic_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads.

    Decided HERE, not by the caller: the B curl reads the HALF-INTEGER positions
    (stepping.py:2473) and ``update_H`` the INTEGER ones (:2483, :948).
    """
    return {"curl": complex_curl_tables(pml, True),
            "constitutive": complex_constitutive_tables(pml, False)}


def complex_folded_fused_magnetic_pair_fills(grid: "Grid") -> Dict[str, Tuple[Any, ...]]:
    """The two fills' launch plan -- :func:`complex_fill_carry.fills_plan`."""
    return _carry.fills_plan(grid)


def assert_disjoint_bindings(fields: "Fields",
                             tables: Dict[str, Dict[str, Any]]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes.

    Once per frozen configuration, off the launch path; returns the number of
    distinct allocations checked so a caller can assert something was inspected.
    """
    bound: Dict[int, str] = {}
    collisions = []

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
            "the folded-complex fused magnetic pair binds every argument "
            "__restrict__, and these arguments alias, which is undefined "
            "behaviour NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_complex_folded_fused_magnetic_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], phase_flags: Sequence[Any],
        phase_values: Sequence[Any], walls: Sequence[int],
        fills: Dict[str, Sequence[Any]], dtdx: float, arm: Any,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch.

    ``boundary_codes`` are ``complex_folded_kernels.folded_complex_boundary_codes``'
    (fold-aware; the plain complex resolver refuses a fold by name and is wrong
    for this kernel); ``walls`` are ``zero_metal_axes``' and ``fills`` is
    :func:`complex_folded_fused_magnetic_pair_fills`' -- three DIFFERENT readings
    of the grid, none derivable from another. ``kernel`` is the gate's door.
    """
    nx, ny, nz = (int(n) for n in fields.Bx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(word_view(getattr(fields, name)) for name in _FIELD_BINDINGS) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + tuple(np.int32(code) for code in boundary_codes) + tuple(
        np.int32(flag) for flag in phase_flags
    ) + tuple(np.float32(value) for value in phase_values) + tuple(
        np.int32(int(bool(walls[axis]))) for axis in range(3)
    ) + tuple(np.int32(int(fills["near"][axis])) for axis in range(3)) + tuple(
        np.int32(int(fills["reflect"][axis])) for axis in range(3)
    ) + tuple(np.float32(float(fills["phase"][axis])) for axis in range(3))
    (kernel or _get_kernel(arm))((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "near": tuple(int(v) for v in fills["near"]),
            "reflect": tuple(int(v) for v in fills["reflect"]),
            "phase": tuple(float(v) for v in fills["phase"]),
            "arm": complex_emitter.normalized_expansion(arm)}


def run_complex_folded_fused_magnetic_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, arm: Any, *,
        sources: Any = None, license: Any = None, subnormal_policy: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once.

    A refusal is RETURNED, not raised: the caller's correct response is the array
    path. ``tables`` and ``kernel`` are the gate's doors.
    """
    covered, reason = covers_complex_folded_fused_magnetic_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = complex_folded_fused_magnetic_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    codes, refusal = _folded.folded_complex_boundary_codes(grid)
    if refusal is not None:
        return {"launched": False, "reason": refusal}
    flags, values = bloch_phase_arguments(grid, False)
    return launch_complex_folded_fused_magnetic_pair(
        fields, tables, codes, flags, values, zero_metal_axes(grid),
        _carry.fills_plan(grid), dtdx, arm, kernel)
