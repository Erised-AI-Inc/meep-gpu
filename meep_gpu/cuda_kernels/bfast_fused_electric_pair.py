"""The BFAST hand-CUDA fused pair on the D/E seam: ``step_D`` -> ``zero_metal_D`` -> ``update_E``.

THE BOARD'S CELL: ``D_to_E (cuda_bfast/BFAST, cuda_bfast/BFAST)`` -- 1
seam-instance of demand, 0 clearing the source seam before this product existed
(``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7`` declares
``source_field_types == ['D']``: an ELECTRIC deposit inside this seam). What
this module is worth is therefore the BRACKET (:data:`CARRIES_DEPOSIT_REPAIR`):
without a product on the cell the board prices the row on the driver fact
alone, whatever the fitness verdict says. It is the ELECTRIC TWIN of the
2026-09-02 :mod:`.bfast_fused_magnetic_pair`, landed in the residue round that
priced it (fusion-residue audit §1.3: 12 electric-twin instances), with the
Metal electric twins as the CARRIES=True shape precedent.

WHAT THE CURL HALF IS: the certified ``step_D_bfast_real`` -- the real PML curl
machinery with the three per-component IIR state advances (``total - 2*bprev``
into ``fb_D*``, the advance subtracted from the curl) that implement MEEP's
BFAST scaled-k substitution. The three state advances ride through the register
capture untouched; the splice asserts they arrived, because a curl body without
them would compile and be the plain real pair under this family's name.

THE TWO MIRROR FILLS ARE ABSENT BY REFUSAL, twice over: the certified BFAST
curl refuses a fold itself (the "folded ghost values are dead" half of the real
curl's argument does not transfer and has not been measured), and this
predicate restates the refusal so the guarantee that makes :data:`REPLACES`
honest does not rest on a clause another module could licence away.

``zero_metal_D`` IS CARRIED, and it is the OFF-DIAGONAL table -- two components
per walled axis -- never the B side's diagonal.

THE ALIASING RULE IS THE SIBLINGS': D is bound EXACTLY ONCE, the three
``inv_eps`` pointers are deliberately NOT ``__restrict__``, and the constitutive
``kms`` is renamed ``kms_half_*`` (the D seam's pairing: curl INTEGER,
``update_E`` HALF-INTEGER).

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

# THE CERTIFIED TEXT. ``bfast_curl`` is CuPy-free at module scope (its kernels
# compile on demand), so the curl half's source and predicate are importable
# everywhere; ``constitutive_kernels`` imports CuPy at module scope and is taken
# defensively.
try:
    from . import bfast_curl as _bfast
    from .coverage import (covers_real_pml_constitutive,
                           real_curl_boundary_codes)
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
    _coverage = _load("coverage")
    covers_real_pml_constitutive = _coverage.covers_real_pml_constitutive
    real_curl_boundary_codes = _coverage.real_curl_boundary_codes
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

#: Byte-identical with a gate verdict AND a record behind it.
#: RECORDED 2026-09-02 as ``cuda_bfast_fused_electric_pair_2026-09-02`` in
#: ``certification.json``. Both legs RELEASED under BOTH float32 subnormal
#: policies on the GPU host: 8/8 cases weld- and array-identical per complete
#: driver step, 24/24 live mutations caught of 33 scored, and four deposit
#: legs (an in-seam ELECTRIC source carried by the two repair plans, with the
#: unrepaired null control DIVERGING on both fixtures).
CERTIFIED_KERNELS = (
    "fused_electric_pair_bfast_real",
)
#: EMPTY since 2026-09-02; what emptied it was the run above, not an argument.
UNCERTIFIED_KERNELS = {}

#: TRUE, MEASURED OFF THE CELL: the cell's one row declares
#: ``source_field_types == ['D']`` -- an ELECTRIC deposit between ``step_D``
#: and ``update_E`` -- so without the bracket this product would serve ZERO
#: rows and with it the whole cell. The wiring is
#: ``fused_pairs._install_fused_pair``'s; ``repairable()`` decides per run. The
#: MAGNETIC twin declares False on the same row for the complementary
#: measurement: its magnetic seam is empty.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cuda_bfast_fused_electric_pair"

#: The kernel's entry-point symbol, spelled once.
KERNEL_NAME = "fused_electric_pair_bfast_real"

#: The driver passes ONE launch performs (driver.py:3302, :3310, :3313). The two
#: mirror fills are ABSENT BY REFUSAL: the predicate refuses every folded grid.
REPLACES = ("step_D", "zero_metal_D", "update_E")

#: The sub-step slot the planner holds this on.
SLOT = "step_D"

#: Every line of certified device text this file did not lift verbatim.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void pml_apply(",
     "became": "__device__ __forceinline__ float pml_apply_reg(",
     "why": "the constitutive half consumes the value in a register; the helper "
            "names and returns the expression it already computed. The "
            "expression, its parenthesisation and the store to f[idx] are "
            "unchanged, so the global state it leaves is the certified one."},
    {"line": "        pml_apply(Dx, fu_Dx, idx, curl, ...);",
     "became": "        d_x = pml_apply_reg(Dx, fu_Dx, idx, curl, ...);",
     "why": "capture the register; three lines, one per component, argument "
            "lists untouched, the three BFAST state advances above them "
            "untouched. Three 'float d_* = 0.0f;' declarations are hoisted "
            "above the certified braced blocks."},
    {"line": "    float src_x = Dx[idx] * inv_eps_Ex[idx];   // stepping.py:982: "  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011
             "source * inv_eps",
     "became": "    float src_x = d_x * inv_eps_Ex[idx];   // stepping.py:982: "  # stepping.py live lines for the frozen device-text citation(s) in this string: 982->1011
               "source * inv_eps",
     "why": "THE SEAM. The certified constitutive body opens each component by "
            "reloading the flux density the curl just stored; this reads the "
            "register instead, which is also what lets D be bound exactly "
            "once. The multiply and its operand order are untouched."},
    {"line": "    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);",
     "became": "    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], "
               "kms_half_x[i]);",
     "why": "ONE RENAME, NO ARITHMETIC. kms_x is the INTEGER sub-lattice for "
            "the D curl and the HALF-INTEGER one for update_E; a shadow is a "
            "half-cell error in the absorber profile."},
    {"line": "        Dy[base] = 0.0f; Dz[base] = 0.0f;   "
             "(in_seam_passes._zero_metal_D_kernel_code)",
     "became": "    if (wall_x && i == 0) { d_y = 0.0f; Dy[idx] = d_y; "
               "d_z = 0.0f; Dz[idx] = d_z; }",
     "why": "zero_metal_D carried on the registers, the OFF-DIAGONAL table -- "
            "two components per walled axis, never the B diagonal. Launch "
            "geometry re-spelled, arithmetic not; same +0.0f."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY", "KERNEL_NAME",
    "LIFT_EDITS", "REPLACES", "SLOT", "UNCERTIFIED_KERNELS",
    "assert_disjoint_bindings", "bfast_fused_electric_pair_source",
    "bfast_fused_electric_pair_tables", "certified_constitutive_body",
    "certified_curl_body", "covers_bfast_fused_electric_pair",
    "device_sources", "inverse_epsilon_bindings",
    "launch_bfast_fused_electric_pair", "run_bfast_fused_electric_pair",
    "zero_metal_carry",
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
_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_COORDINATE: Tuple[str, ...] = ("i", "j", "k")

#: ``in_seam_passes._zero_metal_D_kernel_code``'s component table -- the
#: OFF-DIAGONAL complement; see :mod:`.cylindrical_real_fused_electric_pair`.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("wall_x", "i", ("y", "z")),
    ("wall_y", "j", ("x", "z")),
    ("wall_z", "k", ("x", "y")),
)

#: The fused entry point: the certified BFAST step_D signature with the E group,
#: the inverse-permittivity volumes, the HALF-INTEGER constitutive tables and
#: the wall flags appended. D appears EXACTLY ONCE.
_SIGNATURE = r'''
extern "C" __global__ void fused_electric_pair_bfast_real(
    // THE SHARED VOLUME, BOUND ONCE. The curl half writes it and the
    // constitutive half reads it from a register; a second const __restrict__
    // binding of the same allocation would be UB NVRTC miscompiles silently.
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    // The BFAST IIR state, an OUTPUT the gate compares.
    float* __restrict__ fb_Dx, float* __restrict__ fb_Dy, float* __restrict__ fb_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    // E and its constitutive history.
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
    // The six host-rounded BFAST (k1, k2) pairs, one per component -- the
    // certified step_D signature's own append.
    float k1_a, float k2_a, float k1_b, float k2_b, float k1_c, float k2_c,
    // zero_metal_D's walled-axis flags, from in_seam_coverage.zero_metal_axes.
    int wall_x, int wall_y, int wall_z
) {
'''


def _certified_halves() -> None:
    """Refuse the emitter BY NAME on a host where the certified text is unreachable."""
    if constitutive_kernels is None:
        raise RuntimeError(
            "the certified half ['constitutive_kernels'] is not importable on "
            "this host (it imports CuPy at module scope), so there is no "
            "certified constitutive text to splice. "
            "covers_bfast_fused_electric_pair needs none of it and still answers")


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


def bfast_fused_electric_pair_prelude() -> str:
    """The certified real prelude (read through the BFAST family's own string,
    so a gate mutation there reaches this weld) with ``pml_apply`` turned into a
    value, plus the untouched real constitutive prelude."""
    _certified_halves()
    prelude, _body = _split_certified(
        _bfast.kernel_source("step_D_bfast_real"), "step_D_bfast_real")
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
    """``step_D_bfast_real``'s body, lifted, with the registers captured and the
    three BFAST state advances asserted present."""
    _prelude, body = _split_certified(
        _bfast.kernel_source("step_D_bfast_real"), "step_D_bfast_real")
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
        tail,
    ))
    for axis, target in zip(_CARRIED, _TARGETS):
        old = f"pml_apply({target}, fu_{target}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"the certified curl body calls {old.strip()!r} "
                f"{body.count(old)} times, not once; the capture has no anchor")
        body = body.replace(old, f"d_{axis} = pml_apply_reg({target}, fu_{target}, ", 1)
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
    """``zero_metal_D`` (driver.py:3310), carried between the two halves. The
    OFF-DIAGONAL table; the register is cleared beside the store because the
    constitutive half below reads the register."""
    lines = [
        "\n    // stepping._zero_metal (:2206-2245) / in_seam_passes.zero_metal_D,\n"
        "    // CARRIED. The OFF-DIAGONAL: a D component sits ON the wall of each\n"
        "    // axis whose Yee shift is 0, which for D is the other two -- so two\n"
        "    // components are wiped per walled axis. Stored cell 0 is the LOW\n"
        "    // wall; the high wall is the zero ghost shift_dn already supplies.\n"
        "    // A folded metallic axis carries no wall; every folded grid is\n"
        "    // refused outright by this product's predicate.\n"
    ]
    for flag, coordinate, axes in _ZERO_METAL_ROWS:
        stores = " ".join(
            f"d_{axis} = 0.0f; D{axis}[idx] = d_{axis};" for axis in axes)
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {stores} }}\n")
    return "".join(lines)


def certified_constitutive_body() -> str:
    """``update_E_pml_real``'s body, lifted, reading the registers -- the seam
    and the ``kms_half_*`` rename, no ownership guard and no ghost carry (this
    product refuses every folded grid)."""
    _certified_halves()
    _prelude, body = _split_certified(
        own_cell_hoist.unhoisted_kernel_code(
            constitutive_kernels._update_E_pml_real_kernel_code,
            "update_E_pml_real", "_update_E_pml_real_kernel_code"),
        "_update_E_pml_real_kernel_code")
    if _DECODE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer decodes i on its own "
            "line; the duplicate decomposition cannot be identified")
    tail = body.split(_DECODE_END, 1)[1]
    for axis, target in zip(_CARRIED, _TARGETS):
        anchor = f"    float src_{axis} = {target}[idx] * inv_eps_E{axis}[idx];"
        line = _line_starting(tail, anchor,
                              f"{target}'s inverse-permittivity product")
        if line.count(f"{target}[idx]") != 1:
            raise AssertionError(
                f"{target}[idx] appears {line.count(f'{target}[idx]')} times in "
                f"{line!r}; the seam substitution has no unambiguous token")
        tail = tail.replace(line, line.replace(f"{target}[idx]", f"d_{axis}", 1), 1)
    for target in _TARGETS:
        if f"{target}[idx]" in tail:
            raise AssertionError(
                f"a read of {target} survived the seam rewrite; the constitutive "
                f"half would need D bound a second time, which is the aliasing "
                f"hazard this signature exists to avoid")
    for axis, coordinate in zip(_CARRIED, _COORDINATE):
        prefix = f"    constitutive_apply(E{axis}, f_w_E{axis}, idx, src_{axis}, "
        call = _line_starting(tail, prefix, f"E{axis}'s constitutive_apply")
        expected = f"{prefix}kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);"
        if call != expected:
            raise AssertionError(
                f"E{axis}'s constitutive_apply is {call!r}, not the certified "
                f"{expected!r}; the sub-lattice rename would be applied to an "
                f"argument list this module has not read")
        tail = tail.replace(call,
                            call.replace(f"kms_{axis}[", f"kms_half_{axis}[", 1), 1)
    for axis, coordinate in zip(_CARRIED, _COORDINATE):
        if f"kms_{axis}[{coordinate}])" in tail:
            raise AssertionError(
                f"kms_{axis} survived the sub-lattice rename; it is the INTEGER "
                f"vector in the curl body and the HALF-INTEGER one here, and the "
                f"two would collide on the bare name")
    return tail


def bfast_fused_electric_pair_source() -> str:
    """The whole fused kernel. PURE ASCII."""
    source = "".join((
        bfast_fused_electric_pair_prelude(),
        _SIGNATURE,
        certified_curl_body(),
        zero_metal_carry(),
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml) ----\n"
        "    // Its three sources are the registers above, not a reload of D.\n",
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
    return {KERNEL_NAME: bfast_fused_electric_pair_source()}


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
    code = bfast_fused_electric_pair_source()
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# =============================================================================
# COVERAGE
# =============================================================================

def covers_bfast_fused_electric_pair(
        fields: Any, pml: Any, grid: Any, sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``step_D`` -> ``zero_metal_D`` -> ``update_E``?

    A CONJUNCTION over the BFAST family's two certified predicates -- which
    REQUIRE ``bfast_active``, the clause that keeps this product disjoint from
    every other electric-seam sibling (they all refuse it through the certified
    real predicate's own clause) -- plus the seam clauses: the injection route,
    the source slot (carried), the two refused fills, and the wall/shape facts.
    """
    covered, reason = _bfast.covers_bfast_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"
    covered, reason = _bfast.covers_bfast_constitutive(fields, pml, grid, "E")
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

    # THE TWO FILLS ARE REFUSED, NOT CARRIED. The certified BFAST curl refuses
    # a fold itself, so this clause is unreachable today -- and it is still
    # written, so the guarantee that the two fills do nothing inside this seam
    # does not rest on a clause in another module that a future device verdict
    # could licence away.
    mirrored = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored) or not callable(has_symmetry):
        return False, ("grid does not expose is_mirrored/has_symmetry; this seam "
                       "cannot tell whether fill_symmetry_bc_D and "
                       "fill_folded_far_ghosts_D run inside it")
    try:
        folded = tuple(bool(mirrored(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return False, (f"grid could not answer is_mirrored/has_symmetry: "
                       f"{type(exc).__name__}: {exc}")
    if symmetry or any(folded):
        return False, ("a mirror plane is active: fill_symmetry_bc_D "
                       "(driver.py:3309) and fill_folded_far_ghosts_D (:3311) "
                       "run inside this seam and this pair carries neither")

    # zero_metal_D IS CARRIED, so the grid must be able to say which axes are
    # walled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return False, (f"grid does not expose {name}; zero_metal_D cannot be "
                           f"carried in registers")
    try:
        zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                       f"{type(exc).__name__}: {exc}")
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Dx.shape)
    except Exception as exc:  # noqa: BLE001
        return False, (f"grid could not state its stored extents: "
                       f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return False, (f"grid.stored_cells is {stored} but the launch walks "
                       f"Dx.shape {extents}; the wall plane is derived from the "
                       f"first and indexed into the second")

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
    "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz",
    "Hx", "Hy", "Hz",
    "Ex", "Ey", "Ez",
    "f_w_Ex", "f_w_Ey", "f_w_Ez",
)
_INVERSE_EPSILON_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

_CURL_TABLE_KEYS: Tuple[str, ...] = (
    "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z")
_CONSTITUTIVE_TABLE_KEYS: Tuple[str, ...] = (
    "kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z")


def bfast_fused_electric_pair_tables(pml: "PML") -> Dict[str, Dict[str, Any]]:
    """Both coefficient groups, each from the sub-lattice its half reads: the D
    curl INTEGER, the E constitutive HALF-INTEGER -- the mirror image of the
    magnetic twin's pairing."""
    _certified_halves()
    from . import step_curl_kernels  # noqa: PLC0415 - device-only module
    return {
        "curl": step_curl_kernels.real_pml_curl_tables(pml, False),
        "constitutive": constitutive_kernels.real_constitutive_tables(pml, True),
    }


def inverse_epsilon_bindings(fields: "Fields") -> Tuple[Any, Any, Any]:
    """The three per-component inverse-permittivity volumes, in signature order."""
    return tuple(  # type: ignore[return-value]
        fields.inverse_epsilon_for(component)
        for component in _INVERSE_EPSILON_COMPONENTS)


def assert_disjoint_bindings(fields: "Fields",
                             tables: Dict[str, Dict[str, Any]]) -> int:
    """Check the promise every ``__restrict__`` in the signature makes. The
    three inv_eps volumes may alias EACH OTHER but not a restrict argument."""
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
            "the BFAST fused electric pair binds every field and table argument "
            "__restrict__, and these arguments alias, which is undefined "
            "behaviour NVRTC miscompiles silently rather than diagnosing: "
            + "; ".join(collisions))
    return len(bound)


def launch_bfast_fused_electric_pair(
        fields: "Fields", tables: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[Any], walls: Sequence[int],
        coefficients: Sequence[Sequence[float]], dtdx: float,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """All three of :data:`REPLACES` in ONE launch. ``kernel`` is the gate's door."""
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + _FUSED_THREADS - 1) // _FUSED_THREADS
    curl = tables["curl"]
    constitutive = tables["constitutive"]
    scalars = [np.float32(value) for pair in coefficients for value in pair]
    if len(scalars) != 6:
        raise ValueError(
            f"BFAST step_D takes three (k1, k2) pairs, got {len(scalars)} scalars")
    arguments = tuple(getattr(fields, name) for name in _FIELD_BINDINGS) + tuple(
        inverse_epsilon_bindings(fields)
    ) + (
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


def run_bfast_fused_electric_pair(
        fields: "Fields", grid: "Grid", pml: "PML", dtdx: float, *,
        sources: Any = None,
        tables: Optional[Dict[str, Dict[str, Any]]] = None,
        kernel: Optional[Any] = None) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once."""
    covered, reason = covers_bfast_fused_electric_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = bfast_fused_electric_pair_tables(pml)
    assert_disjoint_bindings(fields, tables)
    codes, refusal = real_curl_boundary_codes(grid)
    if refusal is not None:
        return {"launched": False, "reason": refusal}
    return launch_bfast_fused_electric_pair(
        fields, tables, tuple(np.int32(code) for code in codes),
        zero_metal_axes(grid),
        _bfast.bfast_curl_coefficients(grid, "step_D"), dtdx, kernel)
