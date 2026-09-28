"""The COMPLEX-BETA fused Metal pair: special_kz ``step_D`` welded into ``update_E``.

:mod:`.complex_fused_electric_pair` WITH ONE EMITTER SWAPPED, exactly as
:mod:`.beta_fused_electric_pair` swaps one into :mod:`.fused_electric_pair` on the
real-storage side: the certified complex/Bloch curl body is replaced by the certified
COMPLEX BETA curl body — :func:`.special_kz.beta_bloch_curl_source` with
``backward=True`` — and nothing else about the construction changes. Same wall-clear
carry, same three-line seam, same constitutive lift, same plan shape.

THE CELL, from ``results/fusion_matrix_metal_2026-09-01_lastcells`` read with its own
staleness note: ``tests:TestSpecialKz.test_special_kz`` D->E, curl arm
``special_kz complex beta``. The tranche-6 census left ``update_E`` UNSELECTED there;
``results/metal_coverage_special_kz_reclose_2026-08-19/smoke/test_special_kz.json``
re-measured the row with the 2026-08-19 constitutive arm in place and records
``plan_step`` selecting ``special_kz complex beta`` at ALL FOUR slots — so the pair
exists, both arms are wired and certified, and the one thing missing was this weld.
CUDA serves the same row's B->H through ``cuda_special_kz_fused_magnetic_pair``,
which is what proves the weld shape ports.

WHY THE SWAP IS SOUND. The complex beta curl and the certified complex curl differ by
ONE INSERTED BLOCK — the beta term (``stepping._special_kz_beta_term``, S:727-784),
placed AFTER the dtdx curl and BEFORE the ownership mask, the array path's own fold
order (S:438-445 against S:450) — plus two ``float2`` beta coefficient words. The
body anchor, the wrapped-lane predicates, the phase rotation and the single store
line the wall clear splices in front of are all unchanged, and every one is ASSERTED
at emit time. The constitutive half needs nothing:
:func:`.special_kz.beta_run_complex_constitutive_coverage` is already the certified
complex constitutive predicate with the beta clause inverted (measured on this host:
the two constitutive sub-steps moved 7,319 words and ZERO differed between the corpus
row's beta and beta = 0), and this weld lifts the SAME certified body through
:func:`.complex_fused_electric_pair.certified_constitutive_body` — imported, not
copied.

THE SIGNATURE — 30 pointers plus one packed struct, EXACTLY ON THE CEILING::

      3 D  +  3 fu_D  +  3 H        (float2, the curl half)
    + 3 E  +  3 f_w_E               (float2, the constitutive half)
    + 3 inverse epsilon             (float32 — real per cell under complex storage)
    + 6 curl coefficients  +  6 constitutive coefficients   (float32)
    = 30 pointers, + one ``constant Params&`` = 31 bindings

31 IS :data:`.device.MAX_BUFFER_BINDINGS` — the count, not the headroom. The four
beta words ride in the SAME packed struct the five scalars and six phase words
already ride in (two more ``float2`` members), which is why this weld fits where the
separate-scalar spelling — :data:`SEPARATE_SCALAR_BINDINGS`, compiled and REFUSED by
the gate — cannot.

THE SEAM: the driver injects an ELECTRIC source between the halves
(driver.py:3294-3299) and the corpus row carries one (its ``plan_step.live`` names
``fill_D``), so :data:`CARRIES_DEPOSIT_REPAIR` is True and IS the row: with it False
this weld would compile, gate green on every arithmetic leg, and serve nothing. The
repair arithmetic on complex storage is the measurement
:mod:`.complex_fused_electric_pair` already stands on (``test_deposit_repair.py``'s
``COMPLEX_CASES``); nothing here adds to it.

NOT WIRED AS AN ARM, for the reason every fused pair records: ``plan_step`` assigns
at most one arm per slot and this product spans three. It registers ``wired=False``
and reaches its two slots through ``launch.FUSED_PAIR_ARMS`` instead.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import complex_fields, shaders, special_kz, templates
from .complex_fused_electric_pair import (
    _BODY_ANCHOR,
    _CURL_STORE,
    certified_constitutive_body,
    zero_metal_mask,
)
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: this cell the flag IS the row: ``tests:TestSpecialKz.test_special_kz`` declares an
#: electric source inside the seam (its recorded ``plan_step.live`` carries
#: ``fill_D``), so the same module with this False would serve nothing. The wiring
#: this claims is ``launch._install_fused_pair`` through this family's
#: ``FUSED_PAIR_ARMS`` row; the repair path is the split-field default — both halves
#: require an ACTIVE absorber, so the recurrence the seam inverts is
#: ``_apply_constitutive_pml``'s, never the plain overwrite.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "beta_complex_fused_electric_pair"

#: The sub-step slot this arm is registered on. It spans three; it holds a row on the
#: first, so a refusal is NAMED on the slot the fusion starts at.
SLOT = "step_D"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3292-3303). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The shipped binding count: 30 pointers + one packed ``constant Params&``. This is
#: MAX_BUFFER_BINDINGS exactly, the same zero margin the plain complex D/E pair
#: ships at.
PACKED_BINDINGS = 31

#: What the same 30 pointers need with the five scalars, six phase words and four
#: beta words bound SEPARATELY, the way the certified 30-binding complex beta curl
#: binds its own. Over the ceiling by eleven; the gate compiles it and requires the
#: failure.
SEPARATE_SCALAR_BINDINGS = 45

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex-beta fused electric pair binds {PACKED_BINDINGS} buffers and "
        f"Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}; the signature "
        f"cannot be built at all")

#: The host record's total size in bytes. Natural end is 60 (five ``float2`` + four
#: ``uint`` + one ``float``); Metal pads the struct to its 8-byte alignment, so the
#: record is padded to match rather than letting the tail word float.
PARAMS_ITEMSIZE = 64

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "PACKED_BINDINGS", "PARAMS_ITEMSIZE",
    "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalBetaComplexFusedElectricPairPlan", "certified_beta_bloch_curl_body",
    "beta_complex_fused_electric_pair_source",
    "compile_beta_complex_fused_electric_pair",
    "metal_beta_complex_fused_electric_pair_coverage", "params_record_dtype",
    "plan_metal_beta_complex_fused_electric_pair",
    "refuted_separate_scalar_source", "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: THE BUFFER ORDER IS ``complex_fused_electric_pair``'s, POINTER FOR POINTER. Only
#: the struct grows: two ``float2`` beta members after the three phases, unpacked
#: into the certified curl body's own scalar names (``bpr``/``bpi``/``bmr``/``bmi``)
#: before any lifted text runs.
_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE FIVE float2 MEMBERS COME FIRST, for the measured reason
// `complex_fused_magnetic_pair` records: Metal aligns float2 to 8 bytes, and with
// the scalars first the natural host record puts every phase one word early.
struct Params {
    float2 px; float2 py; float2 pz; float2 bp; float2 bm;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
};

kernel void beta_complex_fused_electric_pair_step(
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
    device const float*  e0      [[buffer(15)]],
    device const float*  e1      [[buffer(16)]],
    device const float*  e2      [[buffer(17)]],
    device const float*  kmx     [[buffer(18)]],
    device const float*  sinvx   [[buffer(19)]],
    device const float*  kmy     [[buffer(20)]],
    device const float*  sinvy   [[buffer(21)]],
    device const float*  kmz     [[buffer(22)]],
    device const float*  sinvz   [[buffer(23)]],
    device const float*  kp0     [[buffer(24)]],
    device const float*  km0     [[buffer(25)]],
    device const float*  kp1     [[buffer(26)]],
    device const float*  km1     [[buffer(27)]],
    device const float*  kp2     [[buffer(28)]],
    device const float*  km2     [[buffer(29)]],
    constant Params&     prm     [[buffer(30)]],
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
    """The COMPLEX BETA ``step_D`` curl kernel's BODY, lifted from its own emitter.

    Not a transcription: this is :func:`.special_kz.beta_bloch_curl_source`'s output
    with the ``#include``/helpers/signature preamble and the closing brace removed.
    ``backward`` is True and never a parameter — this is the D seam — and
    ``has_beta`` is True and never a parameter either: a ``has_beta=False`` weld
    would be the plain complex product under a second name, and that product already
    exists (:mod:`.complex_fused_electric_pair`).
    """
    source = special_kz.beta_bloch_curl_source(codes, True, phased, expansion,
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


def beta_complex_fused_electric_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundaries, phases, walls, arm, contraction).

    THE SPLICE IS :func:`.complex_fused_electric_pair
    .complex_fused_electric_pair_source`'S, step for step, with
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
    # THE BETA TERM MUST BE PRESENT. A lift that silently produced the has_beta=0
    # arm would be the plain complex weld wearing this family's name, and every
    # device leg would still pass at beta = 0 because a zero coefficient IS the
    # plain arithmetic.
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
        "    // --- zero_metal_D, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The OFF-DIAGONAL for D: Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z.\n"
        "    // A COMPLEX zero: the array path assigns one to both planes\n"
        "    // (S:1896, :1902).\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    electric = certified_constitutive_body(expansion, contract)
    # THE SEAM, and it is exactly three lines — the plain complex weld's, unchanged:
    # the certified complex E body's reload of the displacement becomes the live
    # register, the inverse-epsilon factor stays on the right and D stays on the
    # LEFT (stepping.py:1011-1013).
    for target in range(3):
        old = (f"    float2 src{target} = "
               f"c_mul_field_left(g{target}[ii], e{target}[ii]);\n")
        if old not in electric:
            raise AssertionError(
                f"the certified complex E constitutive body no longer reads its "
                f"source as {old.strip()!r}; the seam has no anchor")
        electric = electric.replace(
            old,
            f"    // THE SEAM: the register step_D just wrote, not a reload of "
            f"D{'xyz'[target]}.\n"
            f"    float2 src{target} = c_mul_field_left(v{target}, e{target}[ii]);\n")
    # THE CURL'S `g` IS H AND THE CONSTITUTIVE'S `g` WAS D. After the seam there must
    # be no `g` left in the electric half at all.
    for target in range(3):
        if f"g{target}[" in electric:
            raise AssertionError(
                f"the lifted complex E constitutive half still reads g{target}, "
                f"which in the fused signature is H{'xyz'[target]} and not "
                f"D{'xyz'[target]}")

    body = "".join((
        curl,
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml:2083) --\n",
        electric,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": body,
    })


def compile_beta_complex_fused_electric_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, phases, walls, arm, mode)."""
    return compile_source(beta_complex_fused_electric_pair_source(
        codes, phased, zero_metal, expansion,
        contract)).beta_complex_fused_electric_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 45-binding separate-scalar signature this platform REFUSES.

    The same 30 pointers with the five scalars, six phase words and four beta words
    each bound separately — the way the certified complex beta curl binds its own
    at its 30-binding size. Not shipped and not launchable: the gate compiles it
    and requires the failure, which is what turns
    :data:`SEPARATE_SCALAR_BINDINGS` from an argument into a measurement.
    """
    volumes = [f"    device float2* q{index} [[buffer({index})]]"
               for index in range(15)]
    reals = [f"    device const float* r{index} [[buffer({index})]]"
             for index in range(15, 30)]
    scalars = [f"    constant uint& s{index} [[buffer({index})]]"
               for index in range(30, 34)]
    scalars.append("    constant float& dtdx [[buffer(34)]]")
    words = [f"    constant float& w{index} [[buffer({index})]]"
             for index in range(35, 45)]
    arguments = ",\n".join(volumes + reals + scalars + words)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            + shaders.contraction_pragma(contract) + "\n"
            "kernel void separate_scalar_beta_pair(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    q0[idx] = float2(dtdx, r15[idx] + w35);\n}\n")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_beta_complex_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span complex-beta ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' OWN shipped predicates plus the seam clauses,
    which is :func:`.complex_fused_electric_pair
    .metal_complex_fused_electric_pair_coverage`'s construction with each half's
    predicate replaced by the special_kz complex arm's:

    * :func:`.special_kz.beta_bloch_pml_curl_coverage` on ``step_D``;
    * :func:`.special_kz.beta_run_complex_constitutive_coverage` on side ``"E"``.

    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it — the
    beta clause, the expansion-probe clause (the beta tranche's FIVE patterns, one
    more than the plain complex artifact's), the complex-storage clause and the
    fold clause are all INHERITED rather than restated.
    """
    reasons: List[str] = []

    curl = special_kz.beta_bloch_pml_curl_coverage(
        fields, pml, "step_D", residency, probe)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = special_kz.beta_run_complex_constitutive_coverage(
        fields, pml, "E", residency, probe)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM, and for this family it is the clause that decides whether the
    # cell is worth anything at all: the one corpus row carries an electric source.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294-3299)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS. Both halves already refuse a mirror plane (the folded beta
    # complex family owns that intersection); restated by NAME for THIS seam.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled; an unreadable answer would compile the wall away on a run that needs
    # it.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A polarization would make the constitutive source (D - sum P) rather than D.
    # The constitutive half already refuses it; restated because this family's
    # kernel bakes the plain product — the same restatement every D/E pair carries.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalBetaComplexFusedElectricPairPlan(KernelPlan):
    """ONE dispatch performing three driver passes on a complex beta run.

    ``launches_per_run`` is the base's 1: this plan unpacks one argument tuple and
    calls one function. ``runs`` is an alias of ``launches``.
    """

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values",
                 "beta_words", "zero_metal", "shape", "n_elem", "dtdx",
                 "expansion", "params")

    family = "complex-beta fused PML D-curl/stored-E pair"

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
        # The COMPLEX CELL count — the dispatch is sized from the first tensor
        # argument's element count, and a complex64 tensor's element count is its
        # cell count.
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.expansion = str(expansion)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def params_record_dtype() -> Any:
    """The host record's dtype — THE FIVE ``float2`` MEMBERS FIRST, itemsize 64.

    ONE HOME FOR THE LAYOUT, for the measured reason
    :func:`.complex_fused_electric_pair.params_record_dtype` gives. The offsets are
    stated explicitly; ``itemsize`` is stated because the natural record ends at 60
    bytes and Metal pads the struct to its 8-byte alignment.
    """
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
    """The five scalars, three phases and two beta words as one 64-byte record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every fused
    pair's ``_params_tensor`` gives. Built here, once, at plan time. Nothing on the
    launch path allocates.
    """
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


def plan_metal_beta_complex_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        ) -> Optional[MetalBetaComplexFusedElectricPairPlan]:
    """Build the complex-beta fused D/E plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal. ``functions`` is the mutation seam and is
    FORWARDED; dropping it would silently disarm every mutation leg. The phase and
    beta words come from the special_kz arm's OWN helpers
    (:func:`.special_kz.bloch_phase_words`,
    :func:`.special_kz.beta_curl_coefficients`), so the conjugation rule and the
    signed-zero-carrying coefficient rounding live in one place each.
    """
    if not metal_beta_complex_fused_electric_pair_coverage(
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
    # backward=True: this is the D seam, whose wrapped lane carries the CONJUGATE.
    phased, phase_words = special_kz.bloch_phase_words(grid, kinds, True)
    phase_values = tuple((phase_words[2 * axis], phase_words[2 * axis + 1])
                         for axis in range(3))
    beta_words = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=False, complex_storage=True)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    flux = [bind_complex(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Dx", "Dy", "Dz")]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in ("Hx", "Hy", "Hz")]
    electric = [bind_complex(name, getattr(fields, name))
                for name in ("Ex", "Ey", "Ez")]
    workspace = [bind_complex("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Ex", "Ey", "Ez")]
    # THE INVERSE EPSILON IS float32 UNDER COMPLEX STORAGE — one real coefficient
    # per cell, applied to both planes by `c_mul_field_left`.
    inverse_epsilon = [
        bind_real("inv_eps_" + name, fields.inverse_epsilon_for(name))
        for name in ("Ex", "Ey", "Ez")]
    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER
    # one — the opposite pairing to the B/H twin, and the kernel cannot tell.
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"))
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + magnetic + electric + workspace + inverse_epsilon
                + curl_coefficients + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_beta_complex_fused_electric_pair(
                codes, phased, walls, expansion, mode)

    dtdx = grid.dt / grid.dx
    return MetalBetaComplexFusedElectricPairPlan(
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
            False, (f"complex-beta fused electric pair cannot fill {slot}",))
    return metal_beta_complex_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("beta_probe"))


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalBetaComplexFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_beta_complex_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, probe=context.extra.get("beta_probe"))


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    Unwired for :func:`.complex_fused_electric_pair.register_arms`'s reason, and
    for the one specific to a weld that inherits wired halves: were it wired it
    would contend on ``step_D`` with ``special_kz_complex``'s own curl arm, which
    admits exactly the same configurations, and ``_select_slot`` would leave the
    slot UNSELECTED — taking the certified curl off the device as well. The absorb
    table (``launch.FUSED_PAIR_ARMS["beta_complex_fused_electric_pair"]``) is what
    reaches it.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "complex-beta fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="complex-beta fused electric D/E pair: ",
                          noun="complex-beta fused PML D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
