"""The SPECIAL_KZ hand-CUDA fused pair on the D/E seam: all five passes of the folded electric seam.

THE BOARD'S CELL: ``D_to_E (cuda_special_kz/real beta, cuda_special_kz/real
beta)`` -- 2 seam-instances of demand, 0 clearing the source seam before this
product existed (``examples:refl-angular-kz2d.py`` and
``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag`` both deposit ELECTRIC
sources inside this seam; the second is additionally FOLDED). What this module
is worth is the BRACKET (:data:`CARRIES_DEPOSIT_REPAIR`) plus the CARRY: it is
the ELECTRIC TWIN of :mod:`.special_kz_fused_magnetic_pair`, landed in the
2026-09-02 residue round that priced it (fusion-residue audit §1.3: 12
electric-twin instances), with the Metal electric twins as the CARRIES=True
shape precedent.

WHAT THE CURL HALF IS: the certified ``step_D_special_kz_real`` -- the real PML
backward curl body with the two beta statements above the masks (``grid.beta !=
0`` under REAL storage, MEEP's "implicitly store i*(TM fields)" trick, the
coefficient host-rounded once by ``special_kz_curl.beta_curl_coefficients``).
The beta statements ride through the register capture untouched; the splice
asserts they arrived, because a curl body without them would compile and be
the plain real electric pair under this family's name.

THE TWO MIRROR FILLS ARE CARRIED, BY THE REAL ELECTRIC PAIR'S OWN MACHINERY
(:mod:`.fused_electric_pair`): the D-side ownership inversion -- near set = the
TWO shift-0 axes reading the PRE-clear register, far set = the component's own
axis applied after the clear, the off-diagonal wall table between the two
parity products, exactly the composition the shipped weld measured
(``probe_cuda_electric_fill_carry_order.py``: 0 differing words for this
ordering, 9 and 11 for the naive one). The special_kz curl body is the
certified real curl body plus the beta statements, so every anchor that carry
splices on is byte-identical here and the machinery is IMPORTED rather than
re-spelled.

THE ALIASING RULE IS THE SIBLINGS': D is bound EXACTLY ONCE.

NOTHING HERE IS DISPATCH; :mod:`.fused_pairs` can plan it opt-in only.
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

# THE CERTIFIED TEXT AND THE SHARED CARRY. ``special_kz_curl`` is stdlib-only;
# ``fused_electric_pair`` -- the real D/E weld whose ownership inversion this
# weld splices -- takes its CuPy-bound halves defensively and is importable
# everywhere; ``constitutive_kernels`` imports CuPy at module scope and is
# taken defensively.
try:
    from . import special_kz_curl as _special_kz
    from . import fused_electric_pair as _real_pair
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
    _real_pair = _load("fused_electric_pair")
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
#: RECORDED 2026-09-02 as ``cuda_special_kz_fused_electric_pair_2026-09-02``
#: in ``certification.json``. Both legs RELEASED under BOTH float32 subnormal
#: policies on the GPU host: 14/14 cases weld- and array-identical per complete
#: driver step, 48/48 live mutations caught of 75 scored, the carry floor on
#: every folded fixture, and four deposit legs with the unrepaired null
#: control DIVERGING.
CERTIFIED_KERNELS = (
    "fused_electric_pair_special_kz_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: TRUE, MEASURED OFF THE CELL: both rows declare an ELECTRIC source between
#: ``step_D`` and ``update_E`` (``refl-angular-kz2d.py`` a point Ez;
#: ``eigsrc_kz_1_real_imag`` its lifted eigenmode SHEETS, which publish point
#: indices -- the premise re-measured in the residue round). Without the
#: bracket this product would serve ZERO rows and with it the whole cell. The
#: wiring is ``fused_pairs._install_fused_pair``'s; ``repairable()`` decides
#: per run.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_special_kz_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_electric_pair_special_kz_real"

#: The driver passes ONE launch performs, in driver order (driver.py:3302,
#: :3309, :3310, :3311, :3313). The two mirror fills are CARRIED by the real
#: electric pair's ownership inversion.
REPLACES = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
            "fill_folded_far_ghosts_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: Every line of certified device text this file did not lift verbatim. The
#: carry entries are :mod:`.fused_electric_pair`'s, shared verbatim -- one
#: spelling of the D-side ownership inversion, two real-storage electric welds.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ float pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the helper "
            "names and returns the float it already computed."},
    {"line": "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;",
     "became": "    if (!owned) return 0.0f;\n    float value = ...;\n"
               "    f[idx] = value;\n    return value;",
     "why": "THE OWNERSHIP GUARD, shared with fused_electric_pair: on a cell "
            "one of the two fills images, the SOURCE thread writes D -- so "
            "this thread must not read f[idx] either. The fu recurrence stays "
            "ABOVE the guard (step_D writes fu at every cell and neither fill "
            "touches it, stepping.py:1451)."},
    {"line": "        pml_apply(Dx, fu_Dx, idx, curl, ...);",
     "became": "        d_x = pml_apply_reg(Dx, fu_Dx, idx, curl, ..., own_x);",
     "why": "capture the register OWNED; three lines, argument lists otherwise "
            "untouched, the beta statements above them untouched."},
    {"line": "    float src_x = Dx[idx] * inv_eps_Ex[idx];   // stepping.py:982: "  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011
             "source * inv_eps",
     "became": "    float src_x = d_x * inv_eps_Ex[idx];   // stepping.py:982: "  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011
               "source * inv_eps",
     "why": "THE SEAM, the real electric pair's own: the register instead of "
            "the reload, which is what lets D be bound exactly once."},
    {"line": "    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);",
     "became": "    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], "
               "kms_half_x[i]);",
     "why": "ONE RENAME, NO ARITHMETIC -- the D seam's sub-lattice pairing."},
    {"line": "        Dy[base] = 0.0f; Dz[base] = 0.0f;   "
             "(in_seam_passes._zero_metal_D_kernel_code)",
     "became": "    int clr_y = 0; ... if (wall_x && i == 0) { clr_y = 1; "
               "clr_z = 1; }\n    if (own_y && clr_y) { d_y = 0.0f; "
               "Dy[idx] = d_y; }",
     "why": "the real electric pair's carried wall clear: the OFF-DIAGONAL "
            "table, the flag NAMED because the near fill carry reads it, "
            "guarded on own_*."},
    {"line": "    field[_face(axis, 0)] = phase * field[_face(axis, 2)]   "
             "(stepping._write_mirror_ghost:1450-1451)",
     "became": "the SOURCE thread at stored 2 writes stored 0 from the "
               "PRE-clear register",
     "why": "THE OWNERSHIP INVERSION, imported from fused_electric_pair -- "
            "fill_symmetry_bc_D (driver.py:3309) runs BEFORE zero_metal_D "
            "(:3310), so the near product is formed from pre_* and the clear "
            "is applied to the result, the driver's own order."},
    {"line": "    array[_face(axis, -1)] = parity * array[_face(axis, reflect)]   "
             "(stepping._fill_folded_far_ghosts:1524-1534)",
     "became": "the SOURCE thread at the reflect row writes the top plane",
     "why": "the same inversion for the far fill, weight -phase, applied LAST "
            "because the driver runs it after the clear."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "certified_constitutive_body",
    "certified_curl_body", "covers_special_kz_fused_electric_pair",
    "device_sources", "launch_special_kz_fused_electric_pair",
    "run_special_kz_fused_electric_pair", "special_kz_fused_electric_pair_fills",
    "special_kz_fused_electric_pair_source",
    "special_kz_fused_electric_pair_tables",
]


# =============================================================================
# THE LIFT
# =============================================================================

_GLOBAL_MARKER = 'extern "C" __global__ void '
_BODY_ANCHOR = "\n) {\n"
_DECODE_END = "    int i = idx / (ny * nz);\n"
#: The certified helper's anchors and the OWNED rewrites -- the real electric
#: pair's own three, restated so loading this file by path cannot pick up a
#: different guard than the one the gate compiled.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ float pml_apply_reg(\n"
_PML_APPLY_PARAMETERS = (
    "    float kms, float sinv, float kms_u, float sinv_u\n)")
_PML_APPLY_PARAMETERS_OWNED = (
    "    float kms, float sinv, float kms_u, float sinv_u, int owned\n)")
_PML_APPLY_STORE = "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
_PML_APPLY_STORE_REG = (
    "    // THE OWNERSHIP GUARD, AND IT IS ABOUT MEMORY TRAFFIC RATHER THAN\n"
    "    // ABOUT A DEAD REGISTER. On a cell a fill images, the SOURCE thread\n"
    "    // writes f -- so this thread must not read f[idx] either: that word is\n"
    "    // written by another block in this same launch and an ordinary CUDA\n"
    "    // launch has no grid-wide barrier at which the load would be defined.\n"
    "    // fu above is UNGUARDED: step_D writes it at every cell and neither\n"
    "    // fill touches it (stepping.py:1451).\n"
    "    if (!owned) return 0.0f;\n"
    "    // THE ONE EDIT TO THIS HELPER: the right-hand side, its\n"
    "    // parenthesisation and the store are the certified ones; the value is\n"
    "    // additionally NAMED so the constitutive half can read it from a\n"
    "    // register instead of reloading it from global memory.\n"
    "    float value = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;\n"
    "    f[idx] = value;\n"
    "    return value;\n")

_CARRIED: Tuple[str, ...] = ("x", "y", "z")
_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")

#: The fused entry point: the real electric pair's signature with this family's
#: kernel name and the two host-rounded beta coefficients inserted after the
#: boundary codes -- the certified special_kz step_D signature's own append.
#: D appears EXACTLY ONCE.
_SIGNATURE = r'''
extern "C" __global__ void fused_electric_pair_special_kz_real(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the
    // constitutive half reads it from a register; binding it a second time as
    // a const __restrict__ source would be UB NVRTC miscompiles silently.
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    // NOT __restrict__, lifted from the certified update_E rather than decided
    // here: an isotropic run hands the same device pointer three times.
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
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
    int bc_x, int bc_y, int bc_z,
    // The two host-rounded beta coefficients, one per call-site sign
    // (stepping.py:770/:784) -- the certified special_kz signature's own append.
    float beta_plus, float beta_minus,
    // zero_metal_D's three walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z,
    // THE TWO FILLS' RUNTIME PLAN -- the real electric pair's own tail.
    int near_x, int near_y, int near_z,
    int reflect_x, int reflect_y, int reflect_z,
    float phase_x, float phase_y, float phase_z
) {
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 770->797, 784->811


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


def special_kz_fused_electric_pair_prelude() -> str:
    """The certified real prelude (read through the special_kz family's own
    string) with ``pml_apply`` turned into an OWNED value -- the real electric
    pair's three rewrites -- plus the untouched real constitutive prelude."""
    if constitutive_kernels is None:
        raise RuntimeError(
            "constitutive_kernels is not importable on this host (it imports "
            "CuPy at module scope), so there is no certified constitutive text "
            "to splice; the predicate needs none of it and still answers")
    prelude, _body = _split_certified(
        _special_kz.kernel_source("step_D_special_kz_real"),
        "step_D_special_kz_real")
    for anchor, what in ((_PML_APPLY_SIGNATURE, "pml_apply's signature"),
                         (_PML_APPLY_PARAMETERS, "pml_apply's parameter list"),
                         (_PML_APPLY_STORE, "pml_apply's closing store")):
        if prelude.count(anchor) != 1:
            raise AssertionError(
                f"the certified real prelude carries {what} "
                f"{prelude.count(anchor)} times, not once; the ownership rewrite "
                f"has no anchor")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_PML_APPLY_PARAMETERS,
                              _PML_APPLY_PARAMETERS_OWNED, 1)
    prelude = prelude.replace(_PML_APPLY_STORE, _PML_APPLY_STORE_REG, 1)
    return prelude + constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE


def certified_curl_body() -> str:
    """``step_D_special_kz_real``'s body, captured OWNED via the real electric
    pair's carry, with the two beta statements asserted present."""
    _prelude, body = _split_certified(
        _special_kz.kernel_source("step_D_special_kz_real"),
        "step_D_special_kz_real")
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
        "    float d_x = 0.0f;\n"
        "    float d_y = 0.0f;\n"
        "    float d_z = 0.0f;\n"
        "\n",
        _real_pair.ownership_declarations(),
        tail,
    ))
    for axis, target in zip(_CARRIED, _TARGETS):
        old = f"pml_apply({target}, fu_{target}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} "
                f"{body.count(old)} times, not once; the capture has no anchor")
        body = body.replace(old, f"d_{axis} = pml_apply_reg({target}, fu_{target}, ", 1)
        prefix = f"        d_{axis} = pml_apply_reg({target}, fu_{target}, "
        call = _real_pair._line_starting(
            body, prefix, f"{target}'s captured pml_apply")
        if not call.endswith(");"):
            raise AssertionError(
                f"the certified curl body no longer closes {target}'s pml_apply "
                f"on one line ({call!r}); the ownership flag has no anchor to be "
                f"appended at")
        body = body.replace(call, f"{call[:-2]}, own_{axis});", 1)
    if "pml_apply(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be "
            "written to global memory and never read into the seam")
    if body.count("beta_plus") + body.count("beta_minus") < 2:
        raise AssertionError(
            "the captured special_kz curl body no longer carries both beta "
            "statements; the lift would ship the plain real electric pair under "
            "this family's name")
    return body


def certified_constitutive_body() -> str:
    """``update_E_pml_real``'s body with the seam, the rename, the ownership
    guard and the D-side ghost carry -- the real electric pair's own lift,
    imported. The special_kz constitutive IS the certified ordinary ``update_E``
    (beta enters the curls only, stepping.py:384-391)."""
    return _real_pair.certified_constitutive_body()


def special_kz_fused_electric_pair_source() -> str:
    """The whole fused kernel. PURE ASCII."""
    source = "".join((
        special_kz_fused_electric_pair_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        _real_pair.pre_clear_registers(),
        _real_pair.zero_metal_carry(),
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml) ----\n"
        "    // Its three sources are the registers above, not a reload of D,\n"
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
    return {KERNEL_NAME: special_kz_fused_electric_pair_source()}


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
    code = special_kz_fused_electric_pair_source()
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_special_kz_fused_electric_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span the five passes of :data:`REPLACES`?

    A CONJUNCTION over the special_kz family's two certified predicates --
    which REQUIRE ``grid.beta != 0`` under real storage, the clause that keeps
    this product disjoint from the real electric pair (refuses beta) and from
    every complex product (real storage) -- plus the seam clauses: the
    injection route, the source slot (carried), and the real electric pair's
    fill-carry clause set, restated so the guarantee does not rest on a clause
    another module could widen.
    """
    covered, reason = _special_kz.covers_special_kz_curl(
        fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _special_kz.covers_special_kz_constitutive(
        fields, pml, grid, "E")
    if not covered:
        return False, f"constitutive half: {reason}"

    # THE INJECTION ROUTE, REFUSED BY NAME -- the real electric pair's clause.
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

    # THE TWO FILLS, CARRIED by the real electric pair's inversion; what the
    # carry does NOT reach is refused BY NAME through the fills' own predicates.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_D and "
                       "fill_folded_far_ghosts_D run inside it")
    covered, reason = covers_fill_symmetry(fields, grid, "D")
    if not covered:
        return False, (f"fill_symmetry_bc_D runs inside this seam "
                       f"(driver.py:3309) and this carry cannot serve it: {reason}")
    covered, reason = covers_fill_folded_far(fields, grid, "D")
    if not covered:
        return False, (f"fill_folded_far_ghosts_D runs inside this seam "
                       f"(driver.py:3311) and this carry cannot serve it: {reason}")
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
            return False, (f"grid does not expose {name}; zero_metal_D cannot be "
                           f"carried in registers")
    try:
        walls = zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE THREE CLAUSES THE CARRY ADDS, the real electric pair's own list.
    for axis in range(3):
        if folded[axis] and bool(walls[axis]):
            return False, (
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction and "
                f"the near carry writes no wall line at the ghost it images")
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Dx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks "
                       f"Dx.shape {extents}; the wall plane and the two fills' "
                       f"rows are derived from the first and indexed into the "
                       f"second")
    for axis in range(3):
        if folded[axis] and extents[axis] <= _real_pair.NEAR_SOURCE_INDEX:
            return False, (
                f"folded axis {axis} stores {extents[axis]} cells, so the near "
                f"fill's source row {_real_pair.NEAR_SOURCE_INDEX} does not "
                f"exist (stepping._mirror_source raises on it)")

    # THE THREE INVERSE-PERMITTIVITY VOLUMES MUST BE ASKABLE AND MATCH THE
    # SHAPE THIS LAUNCH WALKS.
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


def special_kz_fused_electric_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads -- the
    real electric pair's own resolver, imported so the pairing cannot be got
    backwards here."""
    return _real_pair.fused_electric_pair_tables(pml)


def special_kz_fused_electric_pair_fills(grid: "Grid") -> Dict[str, Tuple[Any, ...]]:
    """The two fills' launch plan -- the real electric pair's own resolver."""
    return _real_pair.fused_electric_pair_fills(grid)


def beta_scalars(grid: "Grid") -> Tuple[float, float]:
    """``(beta_plus, beta_minus)``, host-rounded once by the certified family's
    own helper, so the weld and the single cannot round differently."""
    return _special_kz.beta_curl_coefficients(float(grid.beta), float(grid.dt))


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three per-component inverse-permittivity volumes, in signature order."""
    return tuple(  # type: ignore[return-value]
        fields.inverse_epsilon_for(component)
        for component in _INVERSE_EPSILON_COMPONENTS)


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
    for component, volume in zip(_INVERSE_EPSILON_COMPONENTS,
                                 inverse_epsilon_bindings(fields)):
        pointer = int(volume.data.ptr)
        if pointer in bound:
            collisions.append(
                f"inv_eps_{component} and {bound[pointer]} are the same "
                f"allocation, and the second is bound __restrict__")
    if collisions:
        raise ValueError(
            "the special_kz fused electric pair binds every field and table "
            "argument __restrict__, and these arguments alias, which is "
            "undefined behaviour NVRTC miscompiles silently rather than "
            "diagnosing: " + "; ".join(collisions))
    return len(bound)


def launch_special_kz_fused_electric_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], walls: Sequence[int],
        beta: Sequence[float], fills: Dict[str, Sequence[Any]], dtdx: float,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All five of :data:`REPLACES` in ONE launch. ``kernel`` is the gate's door."""
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    plus, minus = beta
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + tuple(
        inverse_epsilon_bindings(fields)
    ) + (
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx),
    ) + tuple(curl[key] for key in _CURL_TABLE_KEYS) + tuple(
        constitutive[key] for key in _CONSTITUTIVE_TABLE_KEYS
    ) + (
        boundary_codes[0], boundary_codes[1], boundary_codes[2],
        np.float32(plus), np.float32(minus),
    ) + tuple(np.int32(int(bool(walls[axis]))) for axis in range(3)
    ) + tuple(np.int32(int(fills["near"][axis])) for axis in range(3)) + tuple(
        np.int32(int(fills["reflect"][axis])) for axis in range(3)
    ) + tuple(np.float32(float(fills["phase"][axis])) for axis in range(3))
    (kernel or _get_kernel())((blocks,), (_FUSED_THREADS,), arguments)
    return {"launched": True, "blocks": blocks, "threads": _FUSED_THREADS,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "walls": tuple(int(bool(walls[axis])) for axis in range(3)),
            "near": tuple(int(v) for v in fills["near"]),
            "reflect": tuple(int(v) for v in fills["reflect"]),
            "phase": tuple(float(v) for v in fills["phase"]),
            "beta_scalars": [float(plus), float(minus)]}


def run_special_kz_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once."""
    covered, reason = covers_special_kz_fused_electric_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = special_kz_fused_electric_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    codes, refusal = real_curl_boundary_codes(grid)
    if refusal is not None:
        return {"launched": False, "reason": refusal}
    return launch_special_kz_fused_electric_pair(
        fields, tables, tuple(np.int32(code) for code in codes),
        zero_metal_axes(grid), beta_scalars(grid),
        special_kz_fused_electric_pair_fills(grid), dtdx, kernel)
