"""The BFAST hand-CUDA fused pair: ``step_B`` -> ``zero_metal_B`` -> ``update_H``.

THE BOARD'S SMALLEST BUILDABLE CELL: ``B_to_H (cuda_bfast/BFAST, cuda_bfast/
BFAST)`` -- 1 seam-instance of demand, 1 clearing the source seam
(``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7``), carrying
NOTHING in the seam. Both halves are CERTIFIED singles
(``cuda_bfast_2026-08-21``, re-gated 2026-08-30); this module is the weld and
introduces no new arithmetic.

WHAT THE CURL HALF IS: the certified real PML curl body with the BFAST insert --
the IIR state advance (``fb_B*``), the masked ``advance``, and the
``curl - advance`` substitution, all INSIDE the lifted body and untouched by the
splice, exactly as the beta statements ride through the special_kz weld. The
state pointers and the six host-gated ``(k1, k2)`` scalars are the certified
signature's own appends, kept.

NO FILLS ARE CARRIED, AND THE ABSENCE IS A REFUSAL: the certified BFAST curl
admits a folded grid (its masks carry the fold arms), so this predicate refuses
one by name -- the ownership inversion that carries the fills is not ported
here, and this cell's one row is unfolded.

THE ALIASING RULE IS THE SIBLINGS': B is bound EXACTLY ONCE. The IIR state is
bound once too, writable, in the curl half only -- ``update_H`` reads nothing
BFAST-dependent (``covers_bfast_constitutive``'s own finding, gated on device).

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

# THE CERTIFIED TEXT. ``bfast_curl`` is stdlib-only (it reads the shared real
# prelude off the sibling's source); ``constitutive_kernels`` imports CuPy at
# module scope and is taken defensively.
try:
    from . import bfast_curl as _bfast
    from .coverage import real_curl_boundary_codes
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


    _bfast = _load("bfast_curl")
    real_curl_boundary_codes = _load("coverage").real_curl_boundary_codes
    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")

from . import own_cell_hoist

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

#: Byte-identical with a gate verdict AND a record behind it; empty until the
#: fused gate has run on a device.
#: RECORDED 2026-09-02 as ``cuda_bfast_fused_magnetic_pair_2026-09-02`` in
#: ``certification.json``. Both legs RELEASED under BOTH float32 subnormal
#: policies on the GPU host: 8/8 cases weld- and array-identical per complete driver
#: step (the 3-D fixture's full movement floor included), 21/21 live mutations
#: caught, and the complement source leg measuring the refusal and its
#: consequence.
CERTIFIED_KERNELS = (
    "fused_magnetic_pair_bfast_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: FALSE, MEASURED OFF THE CELL rather than defaulted: the cell's one row
#: declares ``source_field_types == ['D']`` (the certclose census's own
#: ``configuration`` block), so the magnetic seam is empty on every reachable
#: configuration and ``_install_fused_pair`` always takes its ``NoopPlan``
#: branch here. The shared clause below refuses any declared in-seam magnetic
#: source BY NAME instead, fail-closed.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_bfast_fused_magnetic_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_magnetic_pair_bfast_real"

#: The driver passes ONE launch performs (driver.py:3291, :3295, :3298). The two
#: mirror fills are ABSENT BY REFUSAL: this predicate refuses every folded grid.
REPLACES = ("step_B", "zero_metal_B", "update_H")

#: The sub-step slot the planner holds this on.
SLOT = "step_B"

#: Every line of certified device text this file did not lift verbatim.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ float pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the "
            "expression, its parenthesisation and the store to f[idx] are "
            "unchanged, so the global state it leaves is the certified one."},
    {"line": "        pml_apply(Bx, fu_Bx, idx, curl, ...);",
     "became": "        b_x = pml_apply_reg(Bx, fu_Bx, idx, curl, ...);",
     "why": "capture the register; three lines, argument lists untouched, the "
            "BFAST state advance and masks above them untouched. Three "
            "'float b_* = 0.0f;' declarations are hoisted above the certified "
            "braced blocks."},
    {"line": "    constitutive_apply(Hx, f_w_Hx, idx, Bx[idx], kps_x[i], kms_x[i]);",
     "became": "    constitutive_apply(Hx, f_w_Hx, idx, b_x, kps_x[i], kms_int_x[i]);",
     "why": "THE SEAM plus the sub-lattice rename -- half-integer for the B "
            "curl, integer for the H constitutive; a shadow is a half-cell "
            "error in the absorber profile."},
    {"line": "        array[_face(axis, 0)] = 0   (stepping._zero_metal, :2206)",
     "became": "    if (wall_x && i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }",
     "why": "zero_metal_B carried on the registers -- launch geometry, not "
            "arithmetic. Same B diagonal, same stored cell 0, same +0.0f. The "
            "IIR state is NOT masked: zero_metal_B passes B_COMPONENTS only "
            "(stepping.py:2250) and f_bfast_* is not in it."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "bfast_fused_magnetic_pair_source",
    "bfast_fused_magnetic_pair_tables", "certified_constitutive_body",
    "certified_curl_body", "covers_bfast_fused_magnetic_pair",
    "device_sources", "launch_bfast_fused_magnetic_pair",
    "run_bfast_fused_magnetic_pair", "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================

_GLOBAL_MARKER = 'extern "C" __global__ void '
_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ float pml_apply_reg(\n"
_PML_APPLY_STORE = "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
_PML_APPLY_STORE_REG = (
    "    // THE ONE EDIT TO THIS HELPER: the right-hand side, its\n"
    "    // parenthesisation and the store are the certified ones; the value is\n"
    "    // additionally NAMED so the constitutive half can read it from a\n"
    "    // register instead of reloading it from global memory.\n"
    "    float value = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
    "    f[idx] = value;\n"
    "    return value;\n")

_CARRIED: Tuple[str, ...] = ("x", "y", "z")
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("Bx", "wall_x", "i"),
    ("By", "wall_y", "j"),
    ("Bz", "wall_z", "k"),
)

#: The fused entry point: the certified BFAST step_B signature with the H group,
#: the INTEGER constitutive tables and the wall flags appended. B appears ONCE;
#: the IIR state pointers keep their certified position after the auxiliaries.
_SIGNATURE = r'''
extern "C" __global__ void fused_magnetic_pair_bfast_real(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the
    // constitutive half reads it from a register; a second const __restrict__
    // binding of the same allocation would be UB NVRTC miscompiles silently.
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    float* __restrict__ fb_Bx, float* __restrict__ fb_By, float* __restrict__ fb_Bz,
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
    // The six host-gated BFAST scalars, the certified signature's own append.
    float k1_a, float k2_a, float k1_b, float k2_b, float k1_c, float k2_c,
    // zero_metal_B's walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z
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


def bfast_fused_magnetic_pair_prelude() -> str:
    """The certified real prelude (through the BFAST family's own string, so a
    gate mutation there reaches this weld) with ``pml_apply`` turned into a
    value, plus the untouched real constitutive prelude."""
    if constitutive_kernels is None:
        raise RuntimeError(
            "constitutive_kernels is not importable on this host (it imports "
            "CuPy at module scope), so there is no certified constitutive text "
            "to splice; the predicate needs none of it and still answers")
    prelude, _body = _split_certified(
        _bfast.kernel_source("step_B_bfast_real"), "step_B_bfast_real")
    for anchor, what in ((_PML_APPLY_SIGNATURE, "pml_apply's signature"),
                         (_PML_APPLY_STORE, "pml_apply's closing store")):
        if prelude.count(anchor) != 1:
            raise AssertionError(
                f"the certified real prelude carries {what} "
                f"{prelude.count(anchor)} times, not once; the value rewrite has "
                f"no anchor")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_PML_APPLY_STORE, _PML_APPLY_STORE_REG, 1)
    return prelude + constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE


def certified_curl_body() -> str:
    """``step_B_bfast_real``'s body, lifted, with the registers captured and the
    three BFAST state advances asserted present."""
    _prelude, body = _split_certified(
        _bfast.kernel_source("step_B_bfast_real"), "step_B_bfast_real")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the "
            "carried registers have nowhere to be declared")
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END,
        "\n    // The three carried registers. Declared HERE because each\n"
        "    // certified component below is a braced scope, and a value declared\n"
        "    // inside one does not outlive it.\n"
        "    float b_x = 0.0f;\n"
        "    float b_y = 0.0f;\n"
        "    float b_z = 0.0f;\n"
        "\n",
        tail,
    ))
    for axis in _CARRIED:
        old = f"pml_apply(B{axis}, fu_B{axis}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} "
                f"{body.count(old)} times, not once; the capture has no anchor")
        body = body.replace(old, f"b_{axis} = pml_apply_reg(B{axis}, fu_B{axis}, ", 1)
    if "pml_apply(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be "
            "written to global memory and never read into the seam")
    if body.count("curl = curl - advance;") != 3:
        raise AssertionError(
            f"the captured BFAST curl body carries "
            f"{body.count('curl = curl - advance;')} state substitutions, not 3; "
            f"the lift would ship the plain real pair under this family's name")
    return body


def zero_metal_carry() -> str:
    """``zero_metal_B`` for all three targets, carried on the registers. The IIR
    state is NOT masked: ``zero_metal_B`` passes ``B_COMPONENTS`` only."""
    lines = [
        "    // --- zero_metal_B, carried in registers ---------------------------",
        "    // stepping._zero_metal (:2206-2245): the B DIAGONAL, stored cell 0,",
        "    // +0.0f. f_bfast_* is NOT in B_COMPONENTS and is not masked. A",
        "    // folded metallic axis carries no wall; every folded grid is refused",
        "    // outright by this product's predicate.",
    ]
    for target, flag, coordinate in _ZERO_METAL_ROWS:
        register = f"b_{target[1]}"
        lines.append(
            f"    if ({flag} && {coordinate} == 0) "
            f"{{ {register} = 0.0f; {target}[idx] = 0.0f; }}")
    return "\n".join(lines) + "\n"


def certified_constitutive_body() -> str:
    """``update_H_pml_real``'s body, lifted, with the seam and the rename applied."""
    if constitutive_kernels is None:
        raise RuntimeError(
            "constitutive_kernels is not importable on this host; there is no "
            "certified constitutive text to splice")
    _prelude, body = _split_certified(
        own_cell_hoist.unhoisted_kernel_code(
            constitutive_kernels._update_H_pml_real_kernel_code,
            "update_H_pml_real", "_update_H_pml_real_kernel_code"),
        "_update_H_pml_real_kernel_code")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer carries its decode "
            "prologue; the lift would redeclare the curl half's indices")
    body = body.split(_DECODE_END, 1)[1]
    for axis, coordinate in zip(_CARRIED, ("i", "j", "k")):
        source = f"B{axis}[idx]"
        if body.count(source) != 1:
            raise AssertionError(
                f"the certified H body reads {source} {body.count(source)} times, "
                f"not once; the seam has no anchor")
        body = body.replace(source, f"b_{axis}", 1)
        lattice = f"kms_{axis}[{coordinate}]"
        if body.count(lattice) != 1:
            raise AssertionError(
                f"the certified H body indexes {lattice} {body.count(lattice)} "
                f"times, not once; the sub-lattice rename has no anchor")
        body = body.replace(lattice, f"kms_int_{axis}[{coordinate}]", 1)
    return body


def bfast_fused_magnetic_pair_source() -> str:
    """The whole fused kernel. PURE ASCII."""
    source = "".join((
        bfast_fused_magnetic_pair_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        "\n",
        zero_metal_carry(),
        "\n    // --- update_H (stepping.update_H / _apply_constitutive_pml) ----\n"
        "    // Its three sources are the registers above, not a reload of B.\n",
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
    return {KERNEL_NAME: bfast_fused_magnetic_pair_source()}


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
    code = bfast_fused_magnetic_pair_source()
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_bfast_fused_magnetic_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_B`` -> ``zero_metal_B`` -> ``update_H``?

    A CONJUNCTION over the BFAST family's two certified predicates -- which
    REQUIRE ``grid.bfast_active`` and the per-sub-step IIR state, the clauses
    that keep this product disjoint from every seam sibling (the real pair
    refuses BFAST; every complex product refuses real storage) -- plus the seam
    clauses, including the fold refusal the fills' absence rests on.
    """
    covered, reason = _bfast.covers_bfast_curl(fields, pml, grid, "step_B")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _bfast.covers_bfast_constitutive(fields, pml, grid, "H")
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

    # THE TWO FILLS ARE REFUSED, NOT CARRIED, and the refusal is stated HERE:
    # the certified BFAST curl admits a fold, so without this clause the weld
    # would admit a folded row and silently skip the two fills inside its seam.
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
                       "(driver.py:3294) and fill_folded_far_ghosts_B (:3296) "
                       "run inside this seam and this pair carries neither -- "
                       "the ownership inversion that carries them is not ported "
                       "here")

    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_B cannot be "
                           f"carried in registers")
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Bx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks "
                       f"Bx.shape {extents}; the wall plane is derived from the "
                       f"first and indexed into the second")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

_FIELD_BINDINGS: Tuple[str, ...] = (
    "Bx", "By", "Bz",
    "fu_Bx", "fu_By", "fu_Bz",
    "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
    "Ex", "Ey", "Ez",
    "Hx", "Hy", "Hz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def bfast_fused_magnetic_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
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


def bfast_scalars(grid: "Grid") -> Tuple[Tuple[float, float], ...]:
    """The three ``(k1, k2)`` pairs for ``step_B``, from the certified family's
    own host-side transcription -- the invariance gates included."""
    return _bfast.bfast_curl_coefficients(grid, "step_B")


def assert_disjoint_bindings(fields: "Fields",
                             tables: Dict[str, Dict[str, Any]]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes -- the
    three IIR state pointers included, which no sibling binds."""
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
            "the BFAST fused magnetic pair binds every argument __restrict__, "
            "and these arguments alias, which is undefined behaviour NVRTC "
            "miscompiles silently rather than diagnosing: " + "; ".join(collisions))
    return len(bound)


def launch_bfast_fused_magnetic_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], walls: Sequence[int],
        coefficients: Sequence[Sequence[float]], dtdx: float,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch. ``kernel`` is the gate's door."""
    nx, ny, nz = (int(n) for n in fields.Bx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    scalars = [np.float32(value) for pair in coefficients for value in pair]
    if len(scalars) != 6:
        raise ValueError(
            f"BFAST step_B takes three (k1, k2) pairs, got {len(scalars)} scalars")
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + (
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
    ) + tuple(scalars) + tuple(
        np.int32(int(bool(walls[axis]))) for axis in range(3))
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "bfast_scalars": [float(value) for value in scalars]}


def run_bfast_fused_magnetic_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once."""
    covered, reason = covers_bfast_fused_magnetic_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = bfast_fused_magnetic_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    codes, refusal = real_curl_boundary_codes(grid)
    if refusal is not None:
        return {"launched": False, "reason": refusal}
    return launch_bfast_fused_magnetic_pair(
        fields, tables, tuple(np.int32(code) for code in codes),
        zero_metal_axes(grid), bfast_scalars(grid), dtdx, kernel)
