"""The COMPLEX-BETA hand-CUDA fused pair on the D/E seam: the folded electric weld plus the beta insert.

THE BOARD'S CELL: ``D_to_E (cuda_complex_beta/complex beta, cuda_complex_beta/
complex beta)`` -- 4 seam-instances of demand, 0 clearing the source seam
before this product existed (``tests:TestEigCoeffs.test_binary_grating_special_
kz_0_13_2``/``_1_17_7`` and ``tests:TestSpecialKz.test_eigsrc_kz_0_complex``,
all FOLDED; ``tests:TestSpecialKz.test_special_kz``, unfolded -- every one
declares an ELECTRIC deposit inside this seam, ``eigsrc_kz_0_complex`` through
its lifted eigenmode sheets). What this module is worth is the BRACKET
(:data:`CARRIES_DEPOSIT_REPAIR`) plus the CARRY: it is the ELECTRIC TWIN of
the 2026-09-02 :mod:`.complex_beta_fused_magnetic_pair`, landed in the residue
round that priced it (fusion-residue audit §1.3: 12 electric-twin instances),
with the Metal electric twins as the CARRIES=True shape precedent.

ONE LAUNCH PERFORMS THE SAME FIVE DRIVER PASSES as
:mod:`.complex_folded_fused_electric_pair`, THROUGH THE SAME SHARED D-SIDE
COMPLEX CARRY (:mod:`.complex_electric_fill_carry`), and that sharing is exact
rather than analogous: ``complex_beta_kernels.beta_source`` IS
``complex_folded_kernels.folded_source`` with the beta insert applied, and the
insert -- one ``cf_sub(curl, ...)`` per beta target, above the masks --
touches no fill, no ``pml_apply`` call and no constitutive statement, so every
anchor the carry splices on is byte-identical across the two families.

WHAT IS DIFFERENT, AND ONLY THIS: the curl body carries the two beta
statements and the signature carries the two host-rounded complex coefficients
(``bp``/``bm``, the D side's own sign table -- ``beta_curl_coefficients(...,
MAGNETIC['step_D'])``); the predicate conjoins the BETA family's two certified
predicates, whose beta clause keeps this product disjoint from the folded
electric pair and from every real-storage product.

THE BETA ROWS SPLIT ON THE FOLD AND THE WELD SERVES BOTH SIDES: the fold
branches are runtime plan arguments, dead on ``test_special_kz``'s
all-periodic grid (where both fills are no-ops and every ownership flag is 1)
and live on the folded rows.

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

try:
    from . import complex_beta_kernels as _beta
    from . import complex_electric_fill_carry as _carry
    from . import complex_emitter
    from . import complex_folded_kernels as _folded
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
    _carry = _load("complex_electric_fill_carry")
    complex_emitter = _load("complex_emitter")
    _folded = _load("complex_folded_kernels")
    zero_metal_axes = _load("in_seam_coverage").zero_metal_axes
    compile_cache = _load("compile_cache")

# The CuPy-BOUND launch helpers, taken separately and defensively.
try:
    from .complex_pml_kernels import (bloch_phase_arguments,
                                      complex_constitutive_tables,
                                      complex_curl_tables, word_view)
except ImportError:  # a host with no CuPy: only the launch is unavailable
    bloch_phase_arguments = complex_constitutive_tables = None
    complex_curl_tables = word_view = None

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
#: RECORDED 2026-09-02 as ``cuda_complex_beta_fused_electric_pair_2026-09-02``
#: in ``certification.json``. Both legs RELEASED under BOTH float32 subnormal
#: policies on the GPU host: 12/12 cases weld- and array-identical per complete
#: driver step, 58/58 live mutations caught of 85 scored, the carry floor on
#: every folded fixture, and four deposit legs with the unrepaired null
#: control DIVERGING.
CERTIFIED_KERNELS = (
    "fused_electric_pair_pml_complex_beta",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: TRUE, MEASURED OFF THE CELL: all four rows declare an ELECTRIC deposit
#: between ``step_D`` and ``update_E`` (three plain Sources, and
#: ``eigsrc_kz_0_complex``'s lifted eigenmode SHEETS, which publish point
#: indices -- the premise re-measured in the residue round). Without the
#: bracket this product would serve ZERO rows and with it the whole cell. The
#: wiring is ``fused_pairs._install_fused_pair``'s; ``repairable()`` decides
#: per run, and the fold clauses in ``_folded_seam_reasons`` are LIVE on the
#: three folded rows. The MAGNETIC twin carries the same flag for the
#: complementary seam.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_complex_beta_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_electric_pair_pml_complex_beta"

#: The driver passes ONE launch performs, in driver order (driver.py:3302-3313).
REPLACES = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
            "fill_folded_far_ghosts_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: The edits are :mod:`.complex_electric_fill_carry`'s, shared verbatim with the
#: folded electric weld -- what this family adds over them is NOTHING: the beta
#: statements ride through the capture untouched, exactly as the fold masks do.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), "
             "fprev), sinv_u));",
     "became": "    if (!owned) return cf_zero();\n    cf a = ...;\n    cf value "
               "= mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u);\n"
               "    cf_store(f, idx, value);\n    return value;",
     "why": "same right-hand side, same tree, same store; the value is "
            "additionally NAMED so the constitutive half can read it from a "
            "register, and the ownership guard sits ABOVE the field load."},
    {"line": "(the other shared rewrites)",
     "became": "see complex_folded_fused_electric_pair.LIFT_EDITS",
     "why": "the two D-side complex welds splice one carry; the beta insert is "
            "inside the lifted curl body and is not edited by the splice at "
            "all. The store entry above is restated VERBATIM because the lift "
            "leg checks each module's own declaration against its own emitted "
            "source."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "beta_coefficients",
    "certified_constitutive_body", "certified_curl_body",
    "complex_beta_fused_electric_pair_fills",
    "complex_beta_fused_electric_pair_prelude",
    "complex_beta_fused_electric_pair_source",
    "complex_beta_fused_electric_pair_tables",
    "covers_complex_beta_fused_electric_pair", "device_sources",
    "inverse_epsilon_bindings", "launch_complex_beta_fused_electric_pair",
    "run_complex_beta_fused_electric_pair",
]


# =============================================================================
# THE LIFT
# =============================================================================

_GLOBAL_MARKER = 'extern "C" __global__ void '
_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"


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


#: The fused entry point: the folded electric weld's signature plus the two
#: host-rounded beta coefficients, bound as four floats exactly where the
#: certified beta curl binds them (after the Bloch words). D appears EXACTLY
#: ONCE.
_SIGNATURE = r'''
extern "C" __global__ void fused_electric_pair_pml_complex_beta(
    // THE SHARED VOLUME, BOUND ONCE, as the float32 word view of complex64 D.
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    float* __restrict__ u0, float* __restrict__ u1, float* __restrict__ u2,
    // H, the curl's operands.
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    // E and its constitutive history.
    float* __restrict__ h0, float* __restrict__ h1, float* __restrict__ h2,
    float* __restrict__ w0, float* __restrict__ w1, float* __restrict__ w2,
    // NOT __restrict__: an isotropic run hands the same device pointer three
    // times. FLOAT32 under complex storage, indexed by the COMPLEX CELL index.
    const float* inv_eps_0, const float* inv_eps_1, const float* inv_eps_2,
    // Extents are in COMPLEX CELLS; cf_load/cf_store do the word doubling.
    int nx, int ny, int nz, float dtdx,
    // The D curl's INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // The E constitutive's HALF-INTEGER coefficients, its kms renamed
    // kms_half_*. THE PAIRING IS THE MIRROR IMAGE OF THE B/H SEAM'S.
    const float* __restrict__ kps_x, const float* __restrict__ kms_half_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_half_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_half_z,
    // Fold-aware boundary codes; Bloch table (conjugated -- the backward
    // sub-step); a folded axis never carries a phase.
    int bc_x, int bc_y, int bc_z,
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,
    // The two complex64-rounded beta coefficients, one per call-site sign,
    // host-rounded ONCE and passed through -- the real word's signed zero
    // included, never synthesised in-kernel. THE D SIDE'S OWN SIGN TABLE
    // (beta_curl_coefficients(..., MAGNETIC['step_D'])).
    float bp_re, float bp_im, float bm_re, float bm_im,
    // zero_metal_D's walled-axis flags.
    int wall_x, int wall_y, int wall_z,
    // The two fills' runtime plan -- complex_electric_fill_carry.fills_plan.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float phase_x, float phase_y, float phase_z
) {
'''


def complex_beta_fused_electric_pair_prelude(arm) -> str:
    """The certified BETA prelude (folded prelude + the coefficient-left helper)
    with ``pml_apply`` owned and valued, via the shared rewrite."""
    prelude, _body = _split_emitted(
        _beta.beta_source("step_D", arm), "beta_source('step_D')")
    return _carry.pml_apply_reg_prelude(prelude, FAMILY)


def certified_curl_body(arm) -> str:
    """``step_D_pml_complex_beta``'s body, captured OWNED via the shared carry.

    The two beta statements and the beta registers are INSIDE the lifted body
    and ride through untouched; the splice asserts they arrived, because a curl
    body without them would compile and be the folded electric kernel under
    this family's name.
    """
    _prelude, body = _split_emitted(
        _beta.beta_source("step_D", arm), "beta_source('step_D')")
    captured = _carry.captured_curl_body(body, FAMILY)
    if captured.count("mul_imag_coefficient_left(") != 2:
        raise AssertionError(
            f"the captured beta curl body carries "
            f"{captured.count('mul_imag_coefficient_left(')} beta terms, not 2; "
            f"the lift would ship the folded electric kernel under the beta "
            f"family's name")
    return captured


def certified_constitutive_body(arm) -> str:
    """The certified complex ``update_E`` statements, seamed, renamed, guarded,
    with the ghost carry beside them -- the shared carry's own lift.
    ``update_E`` reads nothing beta-dependent (beta enters the step loop's
    curls only), so the constitutive half is the certified complex pair's under
    the beta family's own admission."""
    _prelude, body = _split_emitted(
        complex_emitter.complex_source("update_E", arm),
        "complex_source('update_E')")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own "
            "line; the duplicate decomposition cannot be identified")
    tail = body.split(_DECODE_END, 1)[1]
    return _carry.constitutive_body_with_carry(tail, FAMILY)


def complex_beta_fused_electric_pair_source(arm) -> str:
    """The whole fused kernel for one expansion arm. PURE ASCII."""
    source = "".join((
        complex_beta_fused_electric_pair_prelude(arm),
        _SIGNATURE,
        certified_curl_body(arm),
        "\n",
        _carry.zero_metal_carry(),
        "\n    // --- update_E (the certified complex constitutive; its sources\n"
        "    // are the registers above, not a reload of D, and each runs again\n"
        "    // at every ghost cell this thread owns) -------------------------\n",
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
    return {name: complex_beta_fused_electric_pair_source(name)
            for name in sorted(complex_emitter.EXPANSIONS)}


_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)
_FUSED_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(arm, name: str = KERNEL_NAME):
    """Compile under one arm, memoized on (name, options, policy, source); the
    source is emitted per call because a gate mutates the emitters it splices."""
    if cp is None:
        raise RuntimeError(
            "CuPy is not importable on this host, so this kernel cannot be "
            "compiled; the predicate and the emitter need no device and still run")
    code = complex_beta_fused_electric_pair_source(arm)
    key = compile_cache.kernel_cache_key(
        f"{name}_arm{complex_emitter.normalized_expansion(arm)}", True,
        _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_complex_beta_fused_electric_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None,
        license: Any = None, subnormal_policy: Any = None) -> Tuple[bool, str]:
    """May ONE launch span the five passes of :data:`REPLACES`?

    A CONJUNCTION over the BETA family's two certified predicates --
    ``covers_complex_beta_curl(step_D)`` (which REQUIRES ``grid.beta != 0``
    under complex storage, admits folded and unfolded runs alike) and
    ``covers_complex_beta_constitutive(E)`` -- plus the seam: the injection
    route, the source slot (carried) and the shared D-side complex fill
    clauses.

    DISJOINT FROM EVERY SEAM SIBLING: the beta clause partitions it from the
    folded and plain complex electric pairs (both inherit the certified complex
    predicate's beta refusal), complex64 storage from the real products, and
    the Dcyl refusal (inherited) from the cylindrical ones.
    """
    covered, reason = _beta.covers_complex_beta_curl(
        fields, pml, grid, "step_D", license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _beta.covers_complex_beta_constitutive(
        fields, pml, grid, "E", license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE INJECTION ROUTE, REFUSED BY NAME -- the complex electric pair's own
    # clause; the complex halves admit a conductivity by name, so it cannot be
    # inherited from a half.
    if getattr(fields, "has_conductivity", False):
        return False, ("the engine carries a conductivity, so the driver "
                       "deposits this seam's electric sources through "
                       "_inject_electric_through_conductivity (driver.py:3305), "
                       "which rescales the increment by condinv; the deposit "
                       "repair has no verdict on that route")

    # THE SOURCE SEAM, CARRIED RATHER THAN REFUSED (CARRIES_DEPOSIT_REPAIR).
    seam = _deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an "
            "empty electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3308)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR)
    if seam:
        return False, seam[0]

    fill = _carry.complex_electric_fill_reasons(fields, grid)
    if fill is not None:
        return False, fill

    # THE THREE INVERSE-PERMITTIVITY VOLUMES MUST BE ASKABLE AND MATCH THE
    # SHAPE THIS LAUNCH WALKS.
    try:
        extents = tuple(int(n) for n in fields.Dx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"fields could not state Dx's shape: "
                       f"{type(exc).__name__}: {exc}")
    accessor = getattr(fields, "inverse_epsilon_for", None)
    if not callable(accessor):
        return False, ("fields does not expose inverse_epsilon_for; update_E's "
                       "three inverse-permittivity volumes cannot be bound")
    for component in ("Ex", "Ey", "Ez"):
        try:
            volume = accessor(component)
        except Exception as exc:  # noqa: BLE001
            return False, (f"fields.inverse_epsilon_for({component!r}) raised: "
                           f"{type(exc).__name__}: {exc}")
        if volume is None:
            return False, f"fields.inverse_epsilon_for({component!r}) is None"
        shape = tuple(int(n) for n in getattr(volume, "shape", ()))
        if shape != extents:
            return False, (f"inverse_epsilon_for({component!r}) has shape "
                           f"{shape} but the launch walks {extents}; the kernel "
                           f"indexes it with this launch's flat index")
    return True, "covered"


# =============================================================================
# THE LAUNCH
# =============================================================================

_FIELD_BINDINGS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz",
    "fu_Dx", "fu_Dy", "fu_Dz",
    "Hx", "Hy", "Hz",
    "Ex", "Ey", "Ez",
    "f_w_Ex", "f_w_Ey", "f_w_Ez",
)
_INVERSE_EPSILON_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def complex_beta_fused_electric_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups: the D curl INTEGER, ``update_E`` HALF-INTEGER --
    the mirror image of the magnetic twin's pairing."""
    return {"curl": complex_curl_tables(pml, False),
            "constitutive": complex_constitutive_tables(pml, True)}


def complex_beta_fused_electric_pair_fills(grid: "Grid") -> Dict[str, Tuple[Any, ...]]:
    """The two fills' launch plan -- :func:`complex_electric_fill_carry.fills_plan`."""
    return _carry.fills_plan(grid)


def beta_coefficients(grid: "Grid") -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """The two host-rounded complex coefficients for the ELECTRIC side.

    ``complex_beta_kernels.beta_curl_coefficients(grid.beta, grid.dt,
    MAGNETIC['step_D'])`` -- ONE rounding to complex64 on the host, the real
    word's signed zero passed through. Resolved here so the launcher and the
    certified single-kernel launcher cannot round differently, and the D side's
    OWN sign table so a copy of the magnetic sibling cannot be silently wrong.
    """
    return _beta.beta_curl_coefficients(
        float(grid.beta), float(grid.dt), _beta.MAGNETIC["step_D"])


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three per-component inverse-permittivity volumes, in signature order."""
    return tuple(  # type: ignore[return-value]
        fields.inverse_epsilon_for(component)
        for component in _INVERSE_EPSILON_COMPONENTS)


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
    for component, volume in zip(_INVERSE_EPSILON_COMPONENTS,
                                 inverse_epsilon_bindings(fields)):
        pointer = int(volume.data.ptr)
        if pointer in bound:
            collisions.append(
                f"inv_eps_{component} and {bound[pointer]} are the same "
                f"allocation, and the second is bound __restrict__")
    if collisions:
        raise ValueError(
            "the complex-beta fused electric pair binds every field and table "
            "argument __restrict__, and these arguments alias, which is "
            "undefined behaviour NVRTC miscompiles silently rather than "
            "diagnosing: " + "; ".join(collisions))
    return len(bound)


def launch_complex_beta_fused_electric_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], phase_flags: Sequence[Any],
        phase_values: Sequence[Any], beta_words: Sequence[Sequence[float]],
        walls: Sequence[int], fills: Dict[str, Sequence[Any]], dtdx: float,
        arm: Any, kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch. ``kernel`` is the gate's door."""
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    (bp_re, bp_im), (bm_re, bm_im) = beta_words
    arguments = tuple(word_view(getattr(fields, name)) for name in _FIELD_BINDINGS
                      ) + tuple(inverse_epsilon_bindings(fields)) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + tuple(np.int32(code) for code in boundary_codes) + tuple(
        np.int32(flag) for flag in phase_flags
    ) + tuple(np.float32(value) for value in phase_values) + (
        np.float32(bp_re), np.float32(bp_im), np.float32(bm_re), np.float32(bm_im),
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3)
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
            "beta_words": [[float(bp_re), float(bp_im)],
                           [float(bm_re), float(bm_im)]],
            "arm": complex_emitter.normalized_expansion(arm)}


def run_complex_beta_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, arm: Any, *,
        sources: Any = None, license: Any = None, subnormal_policy: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once."""
    covered, reason = covers_complex_beta_fused_electric_pair(
        fields, pml, grid, sources, license, subnormal_policy)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = complex_beta_fused_electric_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    codes, refusal = _folded.folded_complex_boundary_codes(grid)
    if refusal is not None:
        return {"launched": False, "reason": refusal}
    # THE BACKWARD PHASE TABLE: `True` is the step_D sub-step's own conjugation.
    flags, values = bloch_phase_arguments(grid, True)
    return launch_complex_beta_fused_electric_pair(
        fields, tables, codes, flags, values, beta_coefficients(grid),
        zero_metal_axes(grid), _carry.fills_plan(grid), dtdx, arm, kernel)
