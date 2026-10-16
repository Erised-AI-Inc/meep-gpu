"""The COMPLEX-BETA hand-CUDA fused pair: the folded-complex weld plus the beta insert.

THE BOARD'S CELL: ``B_to_H (cuda_complex_beta/complex beta, cuda_complex_beta/
complex beta)`` -- 4 seam-instances of demand, ALL FOUR clearing the source seam
since the 2026-09-02 flag flip (three D-sourced rows whose magnetic seam is
empty, plus ``eigsrc_kz_0_complex``, whose lifted eigenmode sheets the deposit
repair carries -- see :data:`CARRIES_DEPOSIT_REPAIR`), with
``fill_symmetry_bc_B`` live on 2 and ``fill_folded_far_ghosts_B`` on 2 (the beta
corpus is mostly FOLDED: 12 of its 16 complex slots were, which is why the beta
family was built OVER the folded one). Both halves are CERTIFIED singles as of
2026-09-01 (``cuda_complex_beta_2026-09-01``; the constitutive is the certified
complex pair under the beta family's own admission).

ONE LAUNCH PERFORMS THE SAME FIVE DRIVER PASSES as
:mod:`.complex_folded_fused_magnetic_pair`, THROUGH THE SAME SHARED CARRY
(:mod:`.complex_fill_carry`), and that sharing is exact rather than analogous:
``complex_beta_kernels.beta_source`` IS ``complex_folded_kernels.folded_source`` with
MEEP's beta pass inserted after the curl of targets 0 and 1 (``folded_pass.
complex_beta_insert``: ``-quarter_turn(c, g)`` part by part and the folded ladder on the
word pair ``pml_apply`` just stored), and the pass touches no fill, no ``pml_apply``
call and no constitutive statement, so every anchor the carry splices on is
byte-identical across the two families. One spelling of the carry, two products. The
pass rides through the capture: targets 0 and 1's registers are the values the LADDER
returns (``cfold_apply_reg``), because the fills and ``update_H`` read B after both
passes.

WHAT IS DIFFERENT, AND ONLY THIS: the curl body carries the two beta passes and the
signature carries the two host-rounded beta coefficients (``bp``/``bm``, real float32
words -- the quarter turn forms no complex coefficient); the predicate conjoins the BETA
family's two certified predicates, whose beta clause is what keeps this product
disjoint from the folded pair (that family inherits the certified complex predicate's
beta refusal) and from every real-storage product (complex64). The chunk record ``cb``
is decoded ONCE for both halves (bit_parity_0_10 DESIGN.md D10), and the run's first
``update_H`` starts H from B on the array path.
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

try:
    from . import complex_beta_kernels as _beta
    from . import complex_emitter
    from . import complex_fill_carry as _carry
    from . import complex_folded_kernels as _folded
    from . import folded_pass as _folded_pass
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

    _beta = _load("complex_beta_kernels")
    complex_emitter = _load("complex_emitter")
    _carry = _load("complex_fill_carry")
    _folded = _load("complex_folded_kernels")
    _folded_pass = _load("folded_pass")
    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")

# The CuPy-BOUND launch helpers, taken separately and defensively so the
# predicate and the emitter above stay importable on the census laptop; a launch
# refuses by name when they are absent.
try:
    from .complex_pml_kernels import (bloch_phase_arguments,
                                      chunk_record_argument,
                                      complex_constitutive_tables,
                                      complex_curl_tables, word_view)
except ImportError:  # a host with no CuPy: only the launch is unavailable
    bloch_phase_arguments = complex_constitutive_tables = chunk_record_argument = None
    complex_curl_tables = word_view = None

# The chunk-record decode's one spelling on this table (NumPy through chunk_map).
try:
    from . import bounds_block as _bounds_block
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    from meep_gpu.cuda_kernels import bounds_block as _bounds_block

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
#: RECORDED 2026-09-02 as ``cuda_complex_beta_fused_magnetic_pair_2026-09-02``
#: in ``certification.json``. Both legs RELEASED under BOTH float32 subnormal
#: policies on the GPU host: 12/12 cases weld- and array-identical per complete
#: driver step, 51/51 live mutations caught, the carry floor diverging on every
#: folded fixture and inert on the unfolded one, and the complement source leg
#: measuring both the refusal and its consequence.
CERTIFIED_KERNELS = (
    "fused_magnetic_pair_pml_complex_beta",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: TRUE, RE-MEASURED 2026-09-02 (fusion-residue audit §1.3): the False that
#: stood here rested on the premise that the cell's one B-sourced row
#: (``eigsrc_kz_0_complex``) "deposits through an ``EigenModeSource``, which
#: publishes no point index" -- and that premise is FALSE of the live tree.
#: ``EigenModeSource`` never reaches the driver: ``_lift_eigenmode_source``
#: re-runs MEEP's synthesis into equivalent-current SHEETS deposited via
#: ``add_source``, whose magnetic sheets are ``VolumeSource`` objects, and
#: every source class publishes ``_point_ix/_point_iy/_point_iz`` plus
#: per-cell amps at setup (``sources._setup_source_points``; only an empty
#: footprint yields ``None``). Re-measured on this tree before the flip: a
#: sheet magnetic ``VolumeSource`` with the lift's ``amp_func`` shape returns
#: index ARRAYS from ``deposit_repair._deposit_index`` and
#: ``seam_source_reasons(carries_repair=True)`` fires no publish-refusal,
#: while ``carries_repair=False`` refuses it -- so the flag, not the
#: machinery, was the blocker. The repair has no point-cardinality
#: assumption (flat index columns of arbitrary length; the line-deposit
#: cases in ``test_deposit_repair.py`` are the measured multi-cell
#: precedent), and no amplitude is computed from field state mid-step
#: (J(t) = host scalar x setup-time per-cell amps).
#:
#: The wiring is ``fused_pairs._install_fused_pair``'s -- the same
#: ``LeadingRepairPlan``/``TrailingRepairPlan`` bracket the seven carrying
#: products reach -- and ``repairable()`` decides per run. The three
#: D-sourced rows are untouched (their magnetic seam is empty); what this
#: buys is the cell's fourth row.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_complex_beta_fused_magnetic_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_magnetic_pair_pml_complex_beta"

#: The driver passes ONE launch performs, in driver order (driver.py:3291-3298).
REPLACES = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
            "fill_folded_far_ghosts_B", "update_H")

#: The sub-step slot the planner holds this on.
SLOT = "step_B"

#: The edits are :mod:`.complex_fill_carry`'s, shared verbatim with the folded
#: weld -- see :mod:`.complex_folded_fused_magnetic_pair`'s ``LIFT_EDITS`` -- plus the
#: two this family's folded ladder adds.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    cf fcur = cf_load(f, idx);  ...  cf_store(f, idx, value);   (pml_apply)",
     "became": "    if (!owned) return cf_zero();\n    cf fcur = cf_load(f, idx);\n"
               "    ...\n    cf_store(f, idx, value);\n    return value;",
     "why": "the ownership guard directly before the one load of the field, so a "
            "destination thread never reads a word another block writes; the value "
            "is additionally RETURNED so the constitutive half reads it from a "
            "register. Applied to pml_apply's own text: the folded ladder ends on "
            "the same store line."},
    {"line": "__device__ __forceinline__ void cfold_apply(",
     "became": "__device__ __forceinline__ cf cfold_apply_reg(... , int owned)",
     "why": "the folded ladder, derived the same way from its own text "
            "(folded_pass.cfold_apply_reg_source): the same guard above its one load "
            "of the field, its fu stage above the guard, the value returned. The "
            "plain helper leaves the prelude: no call is left."},
    {"line": "        cfold_apply(f0, u0, idx, inc_beta, ...);",
     "became": "        b0 = cfold_apply_reg(f0, u0, idx, inc_beta, ..., own_0);",
     "why": "capture the register after the beta pass, under the ownership flag; "
            "targets 0 and 1 (target 2 has no beta pass)."},
    {"line": "(the other shared rewrites)",
     "became": "see complex_folded_fused_magnetic_pair.LIFT_EDITS",
     "why": "the two welds splice one carry; the beta pass is inside the lifted curl "
            "body and its ladder is captured, nothing else of it is edited. The union "
            "chunk-record block is complex_folded_fused_magnetic_pair's."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body",
    "certified_curl_body", "complex_beta_fused_magnetic_pair_fills",
    "complex_beta_fused_magnetic_pair_prelude",
    "complex_beta_fused_magnetic_pair_source",
    "complex_beta_fused_magnetic_pair_tables",
    "covers_complex_beta_fused_magnetic_pair", "device_sources",
    "launch_complex_beta_fused_magnetic_pair",
    "run_complex_beta_fused_magnetic_pair", "union_bounds_block",
]


# =============================================================================
# THE LIFT
# =============================================================================

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


#: The fused entry point: the folded weld's signature plus the two host-rounded
#: beta coefficients, bound as two floats exactly where the certified beta curl
#: binds them (after the Bloch words). B appears EXACTLY ONCE.
_SIGNATURE = r'''
extern "C" __global__ void fused_magnetic_pair_pml_complex_beta(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 B.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ u0, float* __restrict__ u1, float* __restrict__ u2,
    // E, the curl's operands.
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    // H and its constitutive history.
    float* __restrict__ h0, float* __restrict__ h1, float* __restrict__ h2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    int nx, int ny, int nz, float dtdx,
    // The B curl's HALF-INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The H constitutive's INTEGER coefficients, its kms renamed kms_int_*.
    const float* __restrict__ kps_x, const float* __restrict__ kms_int_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_int_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_int_z,
    // Fold-aware boundary codes; Bloch table (a folded axis never carries a
    // phase -- the beta family inherits that refusal through the folded one).
    int bc_x, int bc_y, int bc_z,
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,
    // The two host-rounded beta coefficients, one per call-site sign, float32
    // words: the quarter turn forms no complex coefficient.
    float bp, float bm,
    // zero_metal_B's walled-axis flags.
    int wall_x, int wall_y, int wall_z,
    // The two fills' runtime plan -- complex_fill_carry.fills_plan's triples.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float phase_x, float phase_y, float phase_z,
    // THE CHUNK RECORD (chunk_map.chunk_bounds_record), int32[25], decoded once.
    const int* __restrict__ cb
) {
'''


def _pml_apply_owned_edits(carry: Any) -> Tuple[Tuple[str, str, str], ...]:
    """The shared carry's four ownership rewrites of ``pml_apply``, as anchored edits."""
    return (
        (carry._PML_APPLY_SIGNATURE, carry._PML_APPLY_SIGNATURE_REG, "its signature"),
        (carry._PML_APPLY_PARAMETERS, carry._PML_APPLY_PARAMETERS_OWNED,
         "its parameter list"),
        (carry._PML_APPLY_FCUR, carry._PML_APPLY_FCUR_OWNED, "its one load of the field"),
        (carry._PML_APPLY_STORE, carry._PML_APPLY_STORE_REG, "its closing store"),
    )


def complex_beta_fused_magnetic_pair_prelude() -> str:
    """The certified BETA prelude (folded prelude + the complex fold helpers) with
    ``pml_apply`` and the folded ladder owned and valued. The carry's rewrites apply to
    ``pml_apply``'s own text: the ladder closes on the same store line."""
    prelude, _body = _split_emitted(
        _beta.beta_source("step_B"), "beta_source('step_B')")
    prelude = _folded_pass.rewrite_helper(
        prelude, _carry._PML_APPLY_SIGNATURE, _pml_apply_owned_edits(_carry), "pml_apply")
    return _folded_pass.with_fold_reg(prelude, complex_storage=True, owned=True)


#: The chunk-record entries each lifted body decodes and the ONE union block -- the
#: folded weld's own (the beta curl is the folded curl plus the beta passes).
def union_bounds_block() -> str:
    """The fused kernel's ONE chunk-record block (DESIGN.md D10)."""
    from . import complex_folded_fused_magnetic_pair as _folded_pair  # noqa: PLC0415

    return _folded_pair.union_bounds_block()


def certified_curl_body() -> str:
    """``step_B_pml_complex_beta``'s body, captured OWNED via the shared carry, with the
    two beta passes captured into the target 0 and 1 registers.

    The beta passes are INSIDE the lifted body and ride through; the splice asserts
    they arrived, because a curl body without them would compile and be the folded
    kernel under this family's name.
    """
    from . import complex_folded_fused_magnetic_pair as _folded_pair  # noqa: PLC0415

    _prelude, body = _split_emitted(
        _beta.beta_source("step_B"), "beta_source('step_B')")
    captured = _carry.captured_curl_body(
        body, FAMILY, bounds=(_folded_pair.CURL_BLOCK[0], _folded_pair.CURL_BLOCK[1],
                              union_bounds_block()))
    captured = _folded_pass.capture_fold_calls(
        captured, {f"f{target}, u{target}, ": (f"b{target}", f"own_{target}")
                   for target in range(3)},
        "cfold_apply", "cfold_apply_reg", "the complex-beta fused magnetic pair")
    if (captured.count("cf inc_beta = ") != 2
            or captured.count("b0 = cfold_apply_reg(") != 1
            or captured.count("b1 = cfold_apply_reg(") != 1):
        raise AssertionError(
            "the captured beta curl body no longer carries both beta passes captured "
            "into the target 0 and 1 registers; the lift would ship the folded kernel "
            "under the beta family's name")
    return captured


def certified_constitutive_body() -> str:
    """The certified complex ``update_H`` statements, seamed, renamed, guarded.

    ``update_H`` reads nothing beta-dependent, so the constitutive half is the
    certified complex pair's -- the same text the folded weld splices -- under the beta
    family's own admission (``covers_complex_beta_constitutive``).
    """
    from . import complex_folded_fused_magnetic_pair as _folded_pair  # noqa: PLC0415

    _prelude, body = _split_emitted(
        complex_emitter.complex_source("update_H"),
        "complex_source('update_H')")
    decode = "    int i = idx / (ny * nz);\n"
    if decode not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own "
            "line; the duplicate decomposition cannot be identified")
    body = _bounds_block.strip_bounds_block(body, "the certified constitutive body",
                                            *_folded_pair.CONSTITUTIVE_BLOCK)
    tail = body.split(decode, 1)[1]
    return _carry.constitutive_body_with_carry(tail, FAMILY)


def complex_beta_fused_magnetic_pair_source() -> str:
    """The whole fused kernel. PURE ASCII."""
    source = "".join((
        complex_beta_fused_magnetic_pair_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        "\n",
        _carry.zero_metal_carry(),
        "\n    // --- update_H (the certified complex constitutive; its sources\n"
        "    // are the registers above, not a reload of B) -------------------\n",
        certified_constitutive_body(),
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
    """The shipped device text by kernel name -- what a record block would pin."""
    return {KERNEL_NAME: complex_beta_fused_magnetic_pair_source()}


_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str = KERNEL_NAME):
    """Compile, memoized on (name, options, policy, source); the
    source is emitted per call because a gate mutates the emitters it splices."""
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be "
            "compiled; the predicate and the emitter need no device and still run")
    code = complex_beta_fused_magnetic_pair_source()
    key = compile_cache.kernel_cache_key(
        name, True, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_complex_beta_fused_magnetic_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span the five passes of :data:`REPLACES`?

    A CONJUNCTION over the BETA family's two certified predicates --
    ``covers_complex_beta_curl`` (which REQUIRES ``grid.beta != 0`` under complex
    storage, admits folded and unfolded runs alike, and inherits every folded
    clause through the folded family) and ``covers_complex_beta_constitutive``
    (the certified complex pair under the beta admission) -- plus the seam: the
    source slot and the shared complex fill clauses.

    DISJOINT FROM EVERY SEAM SIBLING: the beta clause partitions it from the
    folded and plain complex pairs (both inherit the certified complex
    predicate's beta refusal), complex64 storage from the real products, and the
    Dcyl refusal (inherited) from the cylindrical ones.
    """
    covered, reason = _beta.covers_complex_beta_curl(
        fields, pml, grid, "step_B")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _beta.covers_complex_beta_constitutive(
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

    fill = _carry.complex_fill_reasons(fields, grid)
    if fill is not None:
        return False, fill
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


def complex_beta_fused_magnetic_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads."""
    return {"curl": complex_curl_tables(pml, True),
            "constitutive": complex_constitutive_tables(pml, False)}


def complex_beta_fused_magnetic_pair_fills(grid: "Grid") -> Dict[str, Tuple[Any, ...]]:
    """The two fills' launch plan -- :func:`complex_fill_carry.fills_plan`."""
    return _carry.fills_plan(grid)


def beta_coefficients(grid: "Grid") -> Tuple[float, float]:
    """The two host-rounded beta coefficients for the MAGNETIC side.

    ``complex_beta_kernels.beta_curl_coefficients(grid.beta, grid.dt, True)`` -- ONE
    rounding to float32 on the host per call site. Resolved here so the launcher and
    the certified single-kernel launcher cannot round differently.
    """
    return _beta.beta_curl_coefficients(
        float(grid.beta), float(grid.dt), _beta.MAGNETIC["step_B"])


def assert_disjoint_bindings(fields: "Fields",
                             tables: Dict[str, Dict[str, Any]]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes."""
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
            "the complex-beta fused magnetic pair binds every argument "
            "__restrict__, and these arguments alias, which is undefined "
            "behaviour NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_complex_beta_fused_magnetic_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], phase_flags: Sequence[Any],
        phase_values: Sequence[Any], beta_words: Sequence[Sequence[float]],
        walls: Sequence[int], fills: Dict[str, Sequence[Any]], dtdx: float,
        kernel: Optional[Any] = None, *, chunk_bounds: Any,
        layer: Any = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch.

    ``beta_words`` is :func:`beta_coefficients`' ``(bp, bm)``; the two floats are bound
    exactly where the certified beta curl binds them. ``kernel`` is the gate's door.
    ``chunk_bounds`` is the run's chunk record on the device, required. The run's first
    ``update_H`` starts H from B: when ``fields.h_separated`` is still False the launch
    is followed by ``stepping.update_H`` on the array path with ``layer``, and a first
    launch handed no layer is refused by name.
    """
    cb = chunk_record_argument(chunk_bounds)
    first = not bool(getattr(fields, "h_separated", True))
    if first and layer is None:
        raise RuntimeError(
            "the run's first update_H starts H from B (stepping.separate_h_from_b) and "
            "this fused magnetic launch was handed no layer to start it with")
    nx, ny, nz = (int(n) for n in fields.Bx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    bp, bm = beta_words
    arguments = tuple(word_view(getattr(fields, name)) for name in _FIELD_BINDINGS) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + tuple(np.int32(code) for code in boundary_codes) + tuple(
        np.int32(flag) for flag in phase_flags
    ) + tuple(np.float32(value) for value in phase_values) + (
        np.float32(bp), np.float32(bm),
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3)
    ) + tuple(np.int32(int(fills["near"][axis])) for axis in range(3)) + tuple(
        np.int32(int(fills["reflect"][axis])) for axis in range(3)
    ) + tuple(np.float32(float(fills["phase"][axis])) for axis in range(3)) + (cb,)
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), arguments)
    if first:
        from ..stepping import update_H  # noqa: PLC0415 - NumPy/CuPy array path

        update_H(fields, layer)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES, "h_started": first,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "near": tuple(int(v) for v in fills["near"]),
            "reflect": tuple(int(v) for v in fills["reflect"]),
            "phase": tuple(float(v) for v in fills["phase"]),
            "beta_words": [float(bp), float(bm)]}


def run_complex_beta_fused_magnetic_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        chunk_bounds: Any, sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once."""
    covered, reason = covers_complex_beta_fused_magnetic_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = complex_beta_fused_magnetic_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    codes, refusal = _folded.folded_complex_boundary_codes(grid)
    if refusal is not None:
        return {"launched": False, "reason": refusal}
    flags, values = bloch_phase_arguments(grid, False)
    return launch_complex_beta_fused_magnetic_pair(
        fields, tables, codes, flags, values, beta_coefficients(grid),
        zero_metal_axes(grid), _carry.fills_plan(grid), dtdx, kernel,
        chunk_bounds=chunk_bounds, layer=pml)
