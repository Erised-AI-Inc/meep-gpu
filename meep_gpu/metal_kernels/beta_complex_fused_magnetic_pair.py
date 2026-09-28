"""The COMPLEX-BETA fused Metal pair on the B seam: special_kz ``step_B`` -> ``update_H``.

THE B-SIDE TWIN of :mod:`.beta_complex_fused_electric_pair`, and the same
one-emitter swap over :mod:`.complex_fused_magnetic_pair`: the certified
complex/Bloch curl body is replaced by the certified COMPLEX BETA curl body —
:func:`.special_kz.beta_bloch_curl_source` with ``backward=False`` — and nothing
else about the construction changes. Same wall-clear carry (the B DIAGONAL), same
three-line seam, same constitutive lift, same plan shape.

THE CELL: ``tests:TestSpecialKz.test_special_kz`` B->H, curl arm ``special_kz
complex beta``, constitutive arm the 2026-08-19 ``special_kz complex beta``
restatement (the tranche-6 census predates that arm; the reclose artifact records
``plan_step`` selecting it — see the electric twin's docstring for the citation).
CUDA serves exactly this instance through ``cuda_special_kz_fused_magnetic_pair``,
which is the cross-backend proof of the weld shape.

THE SIGNATURE — 27 pointers plus one packed struct, 28 bindings, three under the
ceiling: the H constitutive brings no inverse-permeability volume (``H = B`` with
mu = 1 baked into the array path), which is the same physics that separates every
magnetic pair from its electric twin on this backend. The four beta words ride in
the packed struct as two more ``float2`` members.

THE SEAM IS EMPTY ON THE CORPUS ROW — ``test_special_kz`` declares no magnetic
source (its recorded ``plan_step.live`` carries no ``fill_B``) — so
:data:`CARRIES_DEPOSIT_REPAIR` stays False for the measured reason the BFAST and
real-beta magnetic pairs hold it False: the one corpus row this cell reaches is
electric-only, so the clause refuses nothing that exists, and a flag flipped
without a measured demand would be a claim with no row behind it. (The PLAIN
complex magnetic pair flipped True on its own cell's measured demand; no such
demand exists on this one.)

NOT WIRED AS AN ARM; reached through ``launch.FUSED_PAIR_ARMS`` instead.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import shaders, special_kz, templates
from .complex_fused_magnetic_pair import (
    _BODY_ANCHOR,
    _CURL_STORE,
    certified_constitutive_body,
    zero_metal_mask,
)
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? FALSE, and
#: while False the source-presence clause below refuses every in-seam magnetic
#: deposit by name. The one corpus row this cell reaches carries an ELECTRIC source
#: only, so the refusal costs the cell nothing — the same measured situation the
#: BFAST and real-beta magnetic pairs record. Flipping it is a claim about the PLAN
#: this module builds and is only ever changed in the same edit as that wiring.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "beta_complex_fused_magnetic_pair"

#: The slot this arm holds a row on, and the driver passes one launch performs.
SLOT = "step_B"
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: The shipped binding count: 27 pointers + one packed ``constant Params&``.
PACKED_BINDINGS = 28

#: What the same 27 pointers need with the five scalars, six phase words and four
#: beta words bound SEPARATELY. Over the ceiling by eleven; the gate compiles it
#: and requires the failure.
SEPARATE_SCALAR_BINDINGS = 42

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex-beta fused magnetic pair binds {PACKED_BINDINGS} buffers and "
        f"Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

#: The host record's total size in bytes — the electric twin's layout, unchanged.
PARAMS_ITEMSIZE = 64

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "PACKED_BINDINGS", "PARAMS_ITEMSIZE",
    "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalBetaComplexFusedMagneticPairPlan", "certified_beta_bloch_curl_body",
    "beta_complex_fused_magnetic_pair_source",
    "compile_beta_complex_fused_magnetic_pair",
    "metal_beta_complex_fused_magnetic_pair_coverage", "params_record_dtype",
    "plan_metal_beta_complex_fused_magnetic_pair",
    "refuted_separate_scalar_source", "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: THE BUFFER ORDER IS ``complex_fused_magnetic_pair``'s, POINTER FOR POINTER; only
#: the struct grows two ``float2`` beta members, unpacked into the certified curl
#: body's own scalar names before any lifted text runs.
_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE FIVE float2 MEMBERS COME FIRST, for the measured alignment reason
// `complex_fused_magnetic_pair` records.
struct Params {
    float2 px; float2 py; float2 pz; float2 bp; float2 bm;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
};

kernel void beta_complex_fused_magnetic_pair_step(
    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device float2*       u0      [[buffer(3)]],
    device float2*       u1      [[buffer(4)]],
    device float2*       u2      [[buffer(5)]],
    device const float2* g0      [[buffer(6)]],
    device const float2* g1      [[buffer(7)]],
    device const float2* g2      [[buffer(8)]],
    device float2*       h0      [[buffer(9)]],
    device float2*       h1      [[buffer(10)]],
    device float2*       h2      [[buffer(11)]],
    device float2*       w0      [[buffer(12)]],
    device float2*       w1      [[buffer(13)]],
    device float2*       w2      [[buffer(14)]],
    device const float*  kmx     [[buffer(15)]],
    device const float*  sinvx   [[buffer(16)]],
    device const float*  kmy     [[buffer(17)]],
    device const float*  sinvy   [[buffer(18)]],
    device const float*  kmz     [[buffer(19)]],
    device const float*  sinvz   [[buffer(20)]],
    device const float*  kp0     [[buffer(21)]],
    device const float*  km0     [[buffer(22)]],
    device const float*  kp1     [[buffer(23)]],
    device const float*  km1     [[buffer(24)]],
    device const float*  kp2     [[buffer(25)]],
    device const float*  km2     [[buffer(26)]],
    constant Params&     prm     [[buffer(27)]],
    uint idx [[thread_position_in_grid]])
{
    // THE PACKED ARGUMENTS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES, once,
    // before any of the lifted text runs. Everything below this line is then
    // character-for-character what `special_kz.beta_bloch_curl_source` and
    // `complex_fields.bloch_constitutive_source` emit, plus the wall clear.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float pxr = prm.px.x, pxi = prm.px.y;
    float pyr = prm.py.x, pyi = prm.py.y;
    float pzr = prm.pz.x, pzi = prm.pz.y;
    float bpr = prm.bp.x, bpi = prm.bp.y;
    float bmr = prm.bm.x, bmi = prm.bm.y;
    (void)pzr; (void)pzi;   // the invariant axis carries no phase; bound for one layout

__BODY__
}
"""


def certified_beta_bloch_curl_body(codes: Sequence[int], phased: Sequence[int],
                                   expansion: str,
                                   contract: str = shaders.CONTRACT_OFF) -> str:
    """The COMPLEX BETA ``step_B`` curl kernel's BODY, lifted from its own emitter.

    ``backward`` is False and never a parameter — this is the B seam — and
    ``has_beta`` is True and never a parameter either: a ``has_beta=False`` weld
    would be the plain complex product under a second name.
    """
    source = special_kz.beta_bloch_curl_source(codes, False, phased, expansion,
                                               True, contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified complex beta curl source no longer carries the body "
            "anchor; this family lifts that body and would otherwise splice a "
            "truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(
            "the certified complex beta curl source does not end with '}'")
    return body[: -len("}\n")]


def beta_complex_fused_magnetic_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundaries, phases, walls, arm, contraction).

    THE SPLICE IS :func:`.complex_fused_magnetic_pair
    .complex_fused_magnetic_pair_source`'S, step for step, with
    :func:`certified_beta_bloch_curl_body` where its ``certified_curl_body`` stands.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if len(phased) != 3:
        raise ValueError(f"phased must be a per-axis triple, got {phased!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")

    curl = certified_beta_bloch_curl_body(codes, phased, expansion, contract)
    # THE BETA TERM MUST BE PRESENT — see the electric twin for why its absence
    # would be the plain complex weld under another name.
    for statement in ("curl0 = curl0 - c_mul(float2(bpr, bpi), b);",
                      "curl1 = curl1 - c_mul(float2(bmr, bmi), a);"):
        if statement not in curl:
            raise AssertionError(
                f"the lifted complex beta curl body does not carry {statement!r}; "
                f"this weld would be the plain complex product under another name")
    if _CURL_STORE not in curl:
        raise AssertionError(
            "the certified complex beta curl body no longer stores the three "
            "targets on one line; the wall clear has no anchor to sit in front of")
    head, tail = curl.split(_CURL_STORE, 1)
    curl = "".join((
        head,
        "    // --- zero_metal_B, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The DIAGONAL for B: Bx on an x wall, By on y, Bz on z. A COMPLEX\n"
        "    // zero: the array path assigns one to both planes (S:1896, :1902).\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    magnetic = certified_constitutive_body(expansion, contract)
    # THE SEAM — the plain complex weld's three lines, unchanged.
    for target in range(3):
        old = f"    float2 src{target} = g{target}[ii];\n"
        if old not in magnetic:
            raise AssertionError(
                f"the certified complex H constitutive body no longer reads its "
                f"source as {old.strip()!r}; the seam has no anchor")
        magnetic = magnetic.replace(
            old,
            f"    // THE SEAM: the register step_B just wrote, not a reload of "
            f"B{'xyz'[target]}.\n"
            f"    float2 src{target} = v{target};\n")

    body = "".join((
        curl,
        "\n    // --- update_H (stepping.update_H / _apply_constitutive_pml:2065) --\n",
        magnetic,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": body,
    })


def compile_beta_complex_fused_magnetic_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, phases, walls, arm, mode)."""
    return compile_source(beta_complex_fused_magnetic_pair_source(
        codes, phased, zero_metal, expansion,
        contract)).beta_complex_fused_magnetic_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 42-binding separate-scalar signature this platform REFUSES."""
    volumes = [f"    device float2* q{index} [[buffer({index})]]"
               for index in range(15)]
    reals = [f"    device const float* r{index} [[buffer({index})]]"
             for index in range(15, 27)]
    scalars = [f"    constant uint& s{index} [[buffer({index})]]"
               for index in range(27, 31)]
    scalars.append("    constant float& dtdx [[buffer(31)]]")
    words = [f"    constant float& w{index} [[buffer({index})]]"
             for index in range(32, 42)]
    arguments = ",\n".join(volumes + reals + scalars + words)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            + shaders.contraction_pragma(contract) + "\n"
            "kernel void separate_scalar_beta_magnetic_pair(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    q0[idx] = float2(dtdx, r15[idx] + w32);\n}\n")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_beta_complex_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span complex-beta ``step_B`` -> wall -> ``update_H``?

    A conjunction of the two halves' OWN shipped predicates plus the seam clauses:

    * :func:`.special_kz.beta_bloch_pml_curl_coverage` on ``step_B``;
    * :func:`.special_kz.beta_run_complex_constitutive_coverage` on side ``"H"``.
    """
    reasons: List[str] = []

    curl = special_kz.beta_bloch_pml_curl_coverage(
        fields, pml, "step_B", residency, probe)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    magnetic = special_kz.beta_run_complex_constitutive_coverage(
        fields, pml, "H", residency, probe)
    if not magnetic.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in magnetic.reasons)

    # THE SOURCE SEAM. A MAGNETIC source is injected BETWEEN the two halves
    # (driver.py:3283-3284). IGNORANCE IS NEVER AN EMPTY SET.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283-3284)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS, restated by NAME for THIS seam.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_B and "
                f"stepping.fill_folded_far_ghosts_B both run inside this seam "
                f"(driver.py:3285-3287) and neither is carried by this family")

    # zero_metal_B is CARRIED, so the grid must be able to answer which axes are
    # walled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalBetaComplexFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch performing three driver passes on a complex beta run's B seam."""

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values",
                 "beta_words", "zero_metal", "shape", "n_elem", "dtdx",
                 "expansion", "params")

    family = "complex-beta fused PML B-curl/stored-H pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phased", "zero_metal", "expansion")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 beta_words: Sequence[Tuple[float, float]],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 expansion: str, params: Any, pointers: Sequence[Any],
                 functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.beta_words = tuple((float(re), float(im)) for re, im in beta_words)
        if len(self.beta_words) != 2:
            raise ValueError(f"beta_words must be the (plus, minus) pair, "
                             f"got {beta_words!r}")
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.expansion = str(expansion)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def params_record_dtype() -> Any:
    """The host record's dtype — the electric twin's layout, one home per seam side."""
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["px", "py", "pz", "bp", "bm",
                  "nx", "ny", "nz", "n_elem", "dtdx"],
        "formats": [("<f4", 2), ("<f4", 2), ("<f4", 2), ("<f4", 2), ("<f4", 2),
                    "<u4", "<u4", "<u4", "<u4", "<f4"],
        "offsets": [0, 8, 16, 24, 32, 40, 44, 48, 52, 56],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]],
                   beta_words: Sequence[Tuple[float, float]], device: str) -> Any:
    """The five scalars, three phases and two beta words as one 64-byte record."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    for name, (real, imag) in zip(("px", "py", "pz"), tuple(phase_values)):
        record[name] = (np.float32(real), np.float32(imag))
    for name, (real, imag) in zip(("bp", "bm"), tuple(beta_words)):
        record[name] = (np.float32(real), np.float32(imag))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_beta_complex_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        ) -> Optional[MetalBetaComplexFusedMagneticPairPlan]:
    """Build the complex-beta fused B/H plan, or ``None`` when the seam is refused.

    ``functions`` is the mutation seam and is FORWARDED; dropping it would silently
    disarm every mutation leg.
    """
    if not metal_beta_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    record = probe if probe is not None else special_kz.load_expansion_probe()
    expansion = special_kz.beta_expansion_from_probe(record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    kinds = _boundary_kinds(grid, pml)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    walls = zero_metal_axes(grid)
    # backward=False: this is the B seam, whose wrapped lane carries the
    # UNCONJUGATED phase.
    phased, phase_words = special_kz.bloch_phase_words(grid, kinds, False)
    phase_values = tuple((phase_words[2 * axis], phase_words[2 * axis + 1])
                         for axis in range(3))
    beta_words = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=True, complex_storage=True)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    flux = [bind_complex(name, getattr(fields, name)) for name in ("Bx", "By", "Bz")]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Bx", "By", "Bz")]
    electric = [bind_complex(name, getattr(fields, name))
                for name in ("Ex", "Ey", "Ez")]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in ("Hx", "Hy", "Hz")]
    workspace = [bind_complex("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Hx", "Hy", "Hz")]
    # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER
    # one — the opposite pairing to the D/E twin, and the kernel cannot tell.
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"))
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + electric + magnetic + workspace
                + curl_coefficients + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_beta_complex_fused_magnetic_pair(
                codes, phased, walls, expansion, mode)

    dtdx = grid.dt / grid.dx
    return MetalBetaComplexFusedMagneticPairPlan(
        residency, volumes, codes, phased, phase_values, beta_words, walls,
        grid.shape, dtdx, expansion,
        _params_tensor(grid.shape, dtdx, phase_values, beta_words,
                       residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"complex-beta fused magnetic pair cannot fill {slot}",))
    return metal_beta_complex_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("beta_probe"))


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalBetaComplexFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_beta_complex_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, probe=context.extra.get("beta_probe"))


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False`` — enumerable, never selectable."""
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "complex-beta fused magnetic B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="complex-beta fused magnetic B/H pair: ",
                          noun="complex-beta fused PML B-curl/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
