"""The REAL-Dcyl hand-CUDA fused pair: ``step_B`` -> ``zero_metal_B`` -> ``update_H``.

THE CELL: ``B_to_H (cuda_cylindrical/cylindrical, cuda_constitutive/ordinary)``. Both halves
are lifted from the texts of their singles -- ``cylindrical_kernels``' ``cyl_step_B_pml_real``
and ``constitutive_kernels``' ``update_H_pml_real`` (its statement form,
``own_cell_hoist.unhoisted_kernel_code``) -- and this module introduces no arithmetic: every
edit is a register capture, a seam or a rename, listed in :data:`LIFT_EDITS`.

THE REGISTER FOLLOWS THE STORE. The m = 0 axis rule (``stepping._cylindrical_axis_zero_B``)
stores +0 into Br on the axis row AFTER the recurrence, so the weld clears the register
beside that store; ``zero_metal_B`` is then carried after it, the driver's order.

THE RADIAL SCAN IS THE CURL'S FIRST LAUNCH (``cylindrical_prefix``): it runs before this
launch over the CURRENT Ep and hands the difference in as ``rad`` -- a cached one would be a
stale field one step later. ONE DECODE of the chunk record serves both halves (DESIGN.md
D10). The run's first ``update_H`` starts H from B (``stepping.separate_h_from_b``), so that
launch re-steps ``update_H`` on the array path over the stored B (:func:`start_of_h`), as
the Cartesian pairs do.

NO FILLS ARE CARRIED, BY REFUSAL: the predicate refuses every folded grid. B is bound EXACTLY
ONCE; the constitutive half reads the registers. ``kms_x/y/z`` collide between the halves
(half-integer for the B curl, integer for ``update_H``), so the constitutive group is
``kms_int_*``.
"""

from __future__ import annotations

try:  # a host with no CuPy: the predicate still runs; the emitter refuses by name
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..fields import Fields
    from ..grid import Grid
    from ..pml import PML

# THE CERTIFIED HALVES. These imports ARE the lift -- the device text below is
# spliced from these modules' own strings -- and they are DEFENSIVE because both
# import CuPy at module scope while the predicate (which touches neither) must run
# on the census laptop.
from . import bounds_block, own_cell_hoist

try:
    from . import constitutive_kernels, cylindrical_kernels
except ImportError:  # a host with no CuPy: only the emitter is unavailable
    constitutive_kernels = cylindrical_kernels = None

try:
    from .coverage import covers_real_pml_constitutive
    from .cylindrical_coverage import (covers_real_pml_cylindrical_curl,
                                       cylindrical_boundary_codes)
    from .coverage import real_pml_boundary_kinds
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


    _coverage = _load("coverage")
    covers_real_pml_constitutive = _coverage.covers_real_pml_constitutive
    real_pml_boundary_kinds = _coverage.real_pml_boundary_kinds
    _cylindrical_coverage = _load("cylindrical_coverage")
    covers_real_pml_cylindrical_curl = \
        _cylindrical_coverage.covers_real_pml_cylindrical_curl
    cylindrical_boundary_codes = _cylindrical_coverage.cylindrical_boundary_codes
    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")

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
#: RECORDED 2026-09-02 as ``cuda_cylindrical_real_fused_magnetic_pair_2026-09-02``
#: in ``certification.json``. Both legs RELEASED under BOTH float32 subnormal
#: policies on the GPU host: 8/8 cases weld- and array-identical per complete driver
#: step, 22/22 live mutations caught -- the axis-tail register follow armed as
#: its own needle (cyl_axis_tail_skips_the_register, CAUGHT) -- and the
#: complement source leg measuring the refusal and its consequence.
CERTIFIED_KERNELS = (
    "fused_magnetic_pair_pml_cyl_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: FALSE, MEASURED OFF THE CELL rather than defaulted, the same reading the
#: Triton twin (``triton_kernels/cylindrical_real_fused_magnetic_pair``) and the
#: CUDA no-absorber electric pair made on their own cells: all three of this
#: cell's rows declare ``source_field_types == ['D']`` (the certclose census's
#: own ``configuration`` block), so the MAGNETIC seam is empty on every one of
#: them and the cell already clears on the driver fact alone. With nothing to
#: bracket, ``_install_fused_pair`` always takes its ``NoopPlan`` branch here,
#: and the predicate below refuses any declared in-seam magnetic source BY NAME
#: through the shared clause -- fail-closed, never approximated.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cuda_cylindrical_real_fused_magnetic_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_magnetic_pair_pml_cyl_real"

#: The driver passes ONE launch performs (driver.py:3291, :3295, :3298). The two
#: mirror fills are ABSENT BY REFUSAL: the predicate refuses every folded grid,
#: and on an unfolded one both fills do nothing at all.
REPLACES = ("step_B", "zero_metal_B", "update_H")

#: The sub-step slot the planner holds this on.
SLOT = "step_B"

#: Every line of certified device text this file did not lift verbatim.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ float pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the helper "
            "returns the value it already stored. Both forms, the guards and the "
            "stores are unchanged, so the global state it leaves is the certified one."},
    {"line": "    <the curl body's chunk-record block> + <update_H's>",
     "became": "    <ONE block, bounds_block.bounds_block(both entry sets, presence xyz)>",
     "why": "one decode of the record for both halves (DESIGN.md D10); "
            "bounds_block.strip_bounds_block refuses unless each is exactly the block "
            "its body carries."},
    {"line": "        pml_apply(Bx, fu_Bx, idx, curl, ...);",
     "became": "        b_x = pml_apply_reg(Bx, fu_Bx, idx, curl, ...);",
     "why": "capture the register; three lines, one per component, argument lists "
            "untouched. Three 'float b_* = 0.0f;' declarations are hoisted above the "
            "braced blocks."},
    {"line": "    if (i == 0) Bx[idx] = 0.0f;",
     "became": "    if (i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }",
     "why": "the m = 0 axis rule (stepping._cylindrical_axis_zero_B) stores +0 into Br "
            "on the axis row AFTER the recurrence, so the weld follows the STORE, "
            "clearing the register beside it. The auxiliary is not touched."},
    {"line": "    constitutive_apply(Hx, f_w_Hx, idx, Bx[idx], kps_x[i], kms_x[i], in_i_x);",
     "became": "    constitutive_apply(Hx, f_w_Hx, idx, b_x, kps_x[i], kms_int_x[i], in_i_x);",
     "why": "THE SEAM (Bx[idx] -> b_x) plus the sub-lattice rename: both halves ship a "
            "vector spelled kms_a on DIFFERENT Yee sub-lattices."},
    {"line": "        array[_face(axis, 0)] = 0   (stepping._zero_metal)",
     "became": "    if (wall_x && i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }",
     "why": "zero_metal_B carried on the registers, AFTER the axis rule -- the driver's "
            "order. Same B diagonal, same stored cell 0, same +0.0f."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body",
    "certified_curl_body", "covers_cylindrical_real_fused_magnetic_pair",
    "cylindrical_real_fused_magnetic_pair_source",
    "cylindrical_real_fused_magnetic_pair_tables", "device_sources",
    "launch_cylindrical_real_fused_magnetic_pair",
    "run_cylindrical_real_fused_magnetic_pair", "zero_metal_carry",
]


# =============================================================================
# THE LIFT
# =============================================================================

_GLOBAL_MARKER = 'extern "C" __global__ void '
_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ float pml_apply_reg(\n"
_PML_APPLY_STORE = "    f[idx] = value;\n"
_PML_APPLY_STORE_REG = (
    "    f[idx] = value;\n"
    "    // THE ONE EDIT TO THIS HELPER'S ARITHMETIC: none. Both forms and the store\n"
    "    // are the certified ones; the value is additionally RETURNED so the\n"
    "    // constitutive half can read it from a register instead of reloading it.\n"
    "    return value;\n")

#: The certified m = 0 axis tail, and the register-following rewrite.
_AXIS_TAIL = "    if (i == 0) Bx[idx] = 0.0f;\n"
_AXIS_TAIL_REG = (
    "    // stepping._cylindrical_axis_zero_B, THE CERTIFIED STORE FOLLOWED: the\n"
    "    // register is cleared beside it so update_H reads the post-rule value. The\n"
    "    // auxiliary is untouched, as on the array path.\n"
    "    if (i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }\n")

_CARRIED: Tuple[str, ...] = ("x", "y", "z")
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("Bx", "wall_x", "i"),
    ("By", "wall_y", "j"),
    ("Bz", "wall_z", "k"),
)

#: The fused entry point: the certified cyl step_B signature with the H group,
#: the INTEGER constitutive tables and the wall flags appended. B appears ONCE.
_SIGNATURE = r'''
extern "C" __global__ void fused_magnetic_pair_pml_cyl_real(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the
    // constitutive half reads it from a register; a second const __restrict__
    // binding of the same allocation would be UB NVRTC miscompiles silently.
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    // The radial difference of Ep's per-chunk sum, written by the scan launch
    // before this one (cylindrical_prefix), shaped like the target.
    const float* __restrict__ rad,
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
    // zero_metal_B's walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z,
    // THE CHUNK RECORD (chunk_map.chunk_bounds_record), int32[25]: which cells take
    // which PML form, decoded once for both halves.
    const int* __restrict__ cb
) {
'''


def _certified_halves() -> None:
    """Refuse the emitter BY NAME on a host where the certified text is unreachable."""
    missing = [name for name, module in (
        ("cylindrical_kernels", cylindrical_kernels),
        ("constitutive_kernels", constitutive_kernels)) if module is None]
    if missing:
        raise RuntimeError(
            f"the certified halves {missing} are not importable on this host "
            f"(they import CuPy at module scope), so there is no certified text "
            f"to splice. covers_cylindrical_real_fused_magnetic_pair needs none "
            f"of them and still answers")


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


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure."""
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this family LIFTS that line rather than "
            f"retyping it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the "
            f"lift of {what} would take an arbitrary one")
    return matches[0]


def cylindrical_real_fused_magnetic_pair_prelude() -> str:
    """The certified REAL prelude with ``pml_apply`` turned into a value, plus the
    untouched real constitutive prelude."""
    _certified_halves()
    prelude, _body = _split_certified(
        cylindrical_kernels._cyl_step_B_pml_real_kernel_code,
        "_cyl_step_B_pml_real_kernel_code")
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


#: The chunk-record entries each lifted body decodes, as ``(entries, presence)``: the B
#: curl its half-integer lattice along each dsigu axis plus the presence word, update_H
#: its integer lattice along each own axis (``bounds_block.curl_entries`` /
#: ``constitutive_entries``).
CURL_BLOCK: Tuple[Any, Tuple[int, ...]] = (
    tuple(bounds_block.curl_entries(("Bx", "By", "Bz"))), (0, 1, 2))
CONSTITUTIVE_BLOCK: Tuple[Any, Tuple[int, ...]] = (
    tuple(bounds_block.constitutive_entries(("Hx", "Hy", "Hz"))), ())


def union_bounds_block() -> str:
    """The fused kernel's ONE chunk-record block: both bodies' entries, one decode."""
    return bounds_block.bounds_block(CURL_BLOCK[0] + CONSTITUTIVE_BLOCK[0],
                                     CURL_BLOCK[1] + CONSTITUTIVE_BLOCK[1])


def certified_curl_body() -> str:
    """``cyl_step_B_pml_real``'s body, lifted, registers captured, TAIL FOLLOWED."""
    _certified_halves()
    _prelude, body = _split_certified(
        cylindrical_kernels._cyl_step_B_pml_real_kernel_code,
        "_cyl_step_B_pml_real_kernel_code")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified curl body no longer decodes i on its own line; the "
            "carried registers have nowhere to be declared")
    body = bounds_block.strip_bounds_block(body, "_cyl_step_B_pml_real_kernel_code",
                                           *CURL_BLOCK)
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END, "\n",
        union_bounds_block(),
        "    // The three carried registers. Declared HERE because each\n"
        "    // certified component below is a braced scope, and a value declared\n"
        "    // inside one does not outlive it.\n"
        "    float b_x = 0.0f;\n"
        "    float b_y = 0.0f;\n"
        "    float b_z = 0.0f;\n",
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
    # THE AXIS TAIL, FOLLOWED. The certified store is kept and the register is
    # cleared beside it; a body without the tail is a different kernel and the
    # splice refuses it.
    if body.count(_AXIS_TAIL) != 1:
        raise AssertionError(
            f"the certified curl body carries the m = 0 axis tail "
            f"{body.count(_AXIS_TAIL)} times, not once; the weld follows the "
            f"certified STORE and cannot splice around its absence")
    body = body.replace(_AXIS_TAIL, _AXIS_TAIL_REG, 1)
    return body


def zero_metal_carry() -> str:
    """``zero_metal_B`` for all three targets, carried on the registers, AFTER the
    certified axis tail -- the driver's own order (driver.py:3295)."""
    lines = [
        "    // --- zero_metal_B, carried in registers ---------------------------",
        "    // stepping._zero_metal (:2206-2245): the B DIAGONAL, stored cell 0,",
        "    // +0.0f. AFTER the certified axis tail above, which is the driver's",
        "    // own order: the tail is part of step_B, the wall pass is :3295. A",
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
    _certified_halves()
    _prelude, body = _split_certified(
        own_cell_hoist.unhoisted_kernel_code(
            constitutive_kernels._update_H_pml_real_kernel_code,
            "update_H_pml_real", "_update_H_pml_real_kernel_code"),
        "_update_H_pml_real_kernel_code")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer carries its decode "
            "prologue; the lift would redeclare the curl half's indices")
    body = bounds_block.strip_bounds_block(body, "_update_H_pml_real_kernel_code",
                                           *CONSTITUTIVE_BLOCK)
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


def cylindrical_real_fused_magnetic_pair_source() -> str:
    """The whole fused kernel: certified prelude, signature, curl (tail followed),
    wall carry, constitutive. PURE ASCII."""
    source = "".join((
        cylindrical_real_fused_magnetic_pair_prelude(),
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
    return {KERNEL_NAME: cylindrical_real_fused_magnetic_pair_source()}


_COMPILE_OPTIONS = ('--fmad=false',)
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str = KERNEL_NAME):
    """Compile on first use, memoized on (name, options, policy, source). The
    source is rebuilt per call: it is spliced from the siblings' strings, and a
    gate mutates those."""
    if name != KERNEL_NAME:
        raise ValueError(f"this module ships one kernel, {KERNEL_NAME!r}, "
                         f"not {name!r}")
    if cp is None:
        raise RuntimeError(
            f"{KERNEL_NAME} cannot be compiled here: CuPy is not importable on "
            f"this host. The emitter's certified halves are unreachable too; the "
            f"predicate needs neither and still runs")
    code = cylindrical_real_fused_magnetic_pair_source()
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_cylindrical_real_fused_magnetic_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_B`` -> ``zero_metal_B`` -> ``update_H``?

    A CONJUNCTION over the two certified predicates --
    ``covers_real_pml_cylindrical_curl(step_B)`` (which REQUIRES a real m = 0
    Dcyl grid under an active PML) and ``covers_real_pml_constitutive(H)`` (the
    ordinary certified constitutive, which the Dcyl rows' composer selection
    already goes to) -- plus the seam clauses.

    DISJOINT FROM EVERY SEAM SIBLING ON THE GRID ALONE: every other magnetic-seam
    product either refuses a Dcyl grid by name or requires complex64 storage,
    and this one requires real-storage Dcyl.
    """
    covered, reason = covers_real_pml_cylindrical_curl(
        fields, pml, grid, "step_B")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = covers_real_pml_constitutive(fields, pml, grid, "H")
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE SOURCE SEAM, REFUSED RATHER THAN CARRIED: all three of this cell's
    # rows are electric-only sourced, so the magnetic seam is empty on every
    # reachable configuration and the flag is False by measurement. A declared
    # in-seam magnetic source is refused by name through the shared clause.
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

    # THE TWO FILLS ARE REFUSED, NOT CARRIED, and the refusal is stated HERE
    # rather than inherited, so the guarantee does not rest on a clause another
    # module could widen.
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
                       "the ownership inversion that carries them is the complex "
                       "welds' and is not ported here")

    # zero_metal_B IS CARRIED, so the grid must be able to say which axes are
    # walled; a grid that cannot answer would silently be treated as unwalled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_B cannot be "
                           f"carried in registers")
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # The stored shape the kernel indexes must be the one the wall plane is
    # derived from.
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

#: The fifteen field pointers in SIGNATURE ORDER; the prefix is bound separately
#: because it is resolved per launch.
_FIELD_BINDINGS: Tuple[str, ...] = (
    "Bx", "By", "Bz",
    "fu_Bx", "fu_By", "fu_Bz",
    "Ex", "Ey", "Ez",
)
_H_BINDINGS: Tuple[str, ...] = (
    "Hx", "Hy", "Hz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def cylindrical_real_fused_magnetic_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads: the B
    curl HALF-INTEGER (``cylindrical_curl_tables(pml, True)``), the H
    constitutive INTEGER (``real_constitutive_tables(pml, False)``)."""
    _certified_halves()
    return {
        "curl": cylindrical_kernels.cylindrical_curl_tables(pml, True),
        "constitutive": constitutive_kernels.real_constitutive_tables(pml, False),
    }


def assert_disjoint_bindings(fields: "Fields",
                             tables: Dict[str, Dict[str, Any]],
                             radial: Any) -> int:
    """Check the promise every ``__restrict__`` in the signature makes -- the radial
    difference included, which no Cartesian sibling binds."""
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in _FIELD_BINDINGS + _H_BINDINGS:
        visit(name, getattr(fields, name))
    visit("rad", radial)
    for key in _CURL_TABLE_KEYS:
        visit(f"curl:{key}", tables["curl"][key])
    for key in _CONSTITUTIVE_TABLE_KEYS:
        visit(f"constitutive:{key}", tables["constitutive"][key])
    if collisions:
        raise ValueError(
            "the cylindrical real fused magnetic pair binds every argument "
            "__restrict__, and these arguments alias, which is undefined "
            "behaviour NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def start_of_h(fields: "Fields", layer: Any) -> None:
    """MEEP's start of H after the run's first fused magnetic launch, on the array path:
    ``stepping.update_H`` over the B the launch stored (``separate_h_from_b``)."""
    from ..stepping import update_H  # noqa: PLC0415 - NumPy/CuPy array path

    update_H(fields, layer)


def launch_cylindrical_real_fused_magnetic_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]], radial: Any,
        boundary_codes: Sequence[Any], walls: Sequence[int], dtdx: float,
        kernel: Optional[Any] = None, *, chunk_bounds: Any,
        layer: Any = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch.

    ``radial`` is the CURRENT step's radial difference of Ep (the scan launch's,
    ``cylindrical_prefix.radial_difference``), resolved per launch by the caller and
    shaped like the target. ``chunk_bounds`` is the run's chunk record on the device,
    required. The run's first launch (``fields.h_separated`` False) starts H on the array
    path (:func:`start_of_h`) with ``layer``, and refuses by name when handed none.
    ``kernel`` is the gate's door.
    """
    from .cylindrical_complex_kernels import chunk_record_argument  # noqa: PLC0415

    cb = chunk_record_argument(chunk_bounds)
    first = not bool(getattr(fields, "h_separated", True))
    if first and layer is None:
        raise RuntimeError(
            "the run's first update_H starts H from B (stepping.separate_h_from_b) and "
            "this fused magnetic launch was handed no layer to start it with")
    nx, ny, nz = (int(n) for n in fields.Bx.shape)
    if tuple(int(n) for n in radial.shape) != (nx, ny, nz):
        raise ValueError(f"the radial difference is {tuple(radial.shape)}, the target "
                         f"{(nx, ny, nz)}")
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + (
        radial,
    ) + tuple(getattr(fields, name) for name in _H_BINDINGS) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + tuple(np.int32(code) for code in boundary_codes) + tuple(
        np.int32(int(bool(walls[axis]))) for axis in range(3)) + (cb,)
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), arguments)
    if first:
        start_of_h(fields, layer)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES, "h_started": first,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3))}


def run_cylindrical_real_fused_magnetic_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        chunk_bounds: Any, sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        radial: Any = None, scan: Any = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, scan, launch once.

    ``radial`` may be handed in by a gate that already computed it (so the same words go
    to both paths); otherwise the scan runs here over the CURRENT Ep, from ``scan``
    (``cylindrical_prefix.radial_scan_device_tables``) when a plan holds one.
    """
    from .cylindrical_prefix import radial_difference  # noqa: PLC0415

    covered, reason = covers_cylindrical_real_fused_magnetic_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    _certified_halves()
    if tables is None:
        tables = cylindrical_real_fused_magnetic_pair_tables(pml)
    if radial is None:
        radial = radial_difference(fields, pml, "step_B", tables=scan,
                                   scratch=getattr(fields, "scratch", None))
    codes = cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid)))
    return launch_cylindrical_real_fused_magnetic_pair(
        fields, tables, radial, tuple(np.int32(code) for code in codes),
        zero_metal_axes(grid), dtdx, kernel, chunk_bounds=chunk_bounds, layer=pml)
