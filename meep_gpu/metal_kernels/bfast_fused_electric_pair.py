"""The fused electric D/E pair on a BFAST run: ``step_D`` -> ``update_E``, packed.

THE D-SIDE TWIN of :mod:`.bfast_fused_magnetic_pair`, and the first of the two cells
the 2026-09-01 board scored ``UNFUSABLE ON METAL`` that the SHIPPED
:mod:`.coefficient_pack` technique closes::

    D->E  (BFAST, BFAST)   tests:TestReflectanceAngular.test_reflectance_angular_2_35_7
        bfast_curl.curl 18p + shaders.constitutive_E 18p, sharing 3
        -> 33 pointers, ceiling 30      UNFUSABLE ON METAL   (over by 3)

CROSS-BACKEND FACT, from the residue audit: the board's ``belongs to the CUDA
track`` position is unfulfilled — CUDA's same cell is priced POINTWISE-BUILDABLE and
serves 0 (no BFAST fused electric pair with a deposit carry exists there) — so
before this module NO backend served this instance and the Metal-local pack was the
shortest route.

WHAT MOVED IS NOT THE CEILING. :data:`.device.MAX_BUFFER_BINDINGS` is 31 and stays
31; the gate measures it again rather than citing it. What changed is the
SIGNATURE: the curl half's six per-axis PML coefficient vectors are ONE buffer with
six element offsets carried in ``Params`` — see :mod:`.coefficient_pack` for why an
array pack is read-only by construction and costs nothing on this backend. The pair
binds 28 pointers where the unpacked shape needs 33, and both numbers are COMPILED
rather than counted (:data:`UNPACKED_POINTER_BINDINGS`, :data:`PACKED_BINDINGS`).
This is the THIRD cell the pack flips from UNFUSABLE to built — the two Dcyl D/E
pairs are the precedent — which is what makes packing a tool on this backend rather
than a rescue for one signature.

EVERYTHING ELSE IS THE TWO SIBLINGS' CONSTRUCTION, spliced from certified emitters:

* the curl half — :func:`.bfast_curl.bfast_curl_source` with ``backward=True`` and
  ``has_bfast=True``, lifted verbatim by :func:`certified_bfast_curl_body` exactly
  as the magnetic twin lifts the ``step_B`` arm. The Tustin tail's presence is
  ASSERTED at emit time so this weld can never silently become the ordinary product;
* the wall clear — :func:`.fused_electric_pair.zero_metal_mask`, the D
  OFF-DIAGONAL, imported rather than re-derived;
* the constitutive half — :func:`.fused_electric_pair.certified_constitutive_body`,
  the ordinary ``update_E``: a BFAST run's constitutive pair reads nothing
  BFAST-dependent (:func:`.bfast_curl.bfast_run_constitutive_coverage` is that
  finding, and it is this family's E-side predicate).

THE SEAM: the one corpus row declares an ELECTRIC source (its recorded
``plan_step.live`` carries ``fill_D``), so :data:`CARRIES_DEPOSIT_REPAIR` is True
and IS the row — the repair path is the split-field default, since BFAST requires
an ACTIVE absorber (the curl predicate's clause 3) and ``update_E`` is therefore
always ``_apply_constitutive_pml``'s recurrence.

NOT WIRED AS AN ARM; reached through ``launch.FUSED_PAIR_ARMS`` instead.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.bfast_curl import BFAST_STATE_NAMES, bfast_curl_coefficients
from ..triton_kernels.coverage import Coverage
from . import bfast_curl, coefficient_pack, shaders, templates
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .fused_electric_pair import (
    _BODY_ANCHOR,
    _CURL_STORE,
    certified_constitutive_body,
    zero_metal_mask,
)
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: this cell the flag IS the row: the one corpus instance declares an ELECTRIC
#: source inside the seam, so the same module with this at False would compile, gate
#: green on every arithmetic leg, and serve nothing. The wiring this claims is
#: ``launch._install_fused_pair`` through this family's ``FUSED_PAIR_ARMS`` row;
#: the repair path is the split-field default (BFAST requires an active absorber).
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "bfast_fused_electric_pair"

#: The slot this arm holds a row on, and the driver passes one launch performs.
SLOT = "step_D"
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The curl half's per-axis PML coefficient vectors, IN PACK ORDER — the order the
#: certified BFAST curl body declares them in. One buffer holds them end to end and
#: ``Params`` carries six element offsets.
PACKED_VECTORS: Tuple[str, ...] = ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz")

#: THREE BINDING COUNTS, ALL COMPILED BY THE GATE rather than argued from here:
#:
#: * :data:`SEPARATE_SCALAR_BINDINGS` — 33 pointers with the eleven scalars bound
#:   separately. Refused, and by a wide margin.
#: * :data:`UNPACKED_POINTER_BINDINGS` — 33 pointers plus ONE packed ``Params&``.
#:   THIS IS THE NUMBER THE BOARD'S VERDICT RECORDED: the fused pair as every other
#:   D/E product on this backend builds it, REFUSED at 34 bindings against 31.
#: * :data:`PACKED_BINDINGS` — the shipped shape, 28 pointers plus ``Params``.
SEPARATE_SCALAR_BINDINGS = 44
UNPACKED_POINTER_BINDINGS = 34
PACKED_BINDINGS = 29

#: The pointer counts, spelled apart from the binding counts because the ceiling is
#: a BINDING ceiling and the fusion matrix's sharing calibration is a POINTER count.
PACKED_POINTERS = 28
UNPACKED_POINTERS = 33

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the BFAST fused electric pair binds {PACKED_BINDINGS} buffers and "
        f"Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "PACKED_BINDINGS", "PACKED_POINTERS",
    "PACKED_VECTORS", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "UNPACKED_POINTERS", "UNPACKED_POINTER_BINDINGS",
    "MetalBfastFusedElectricPairPlan", "bfast_fused_electric_pair_source",
    "certified_bfast_curl_body", "compile_bfast_fused_electric_pair",
    "metal_bfast_fused_electric_pair_coverage",
    "plan_metal_bfast_fused_electric_pair",
    "refuted_separate_scalar_source", "refuted_unpacked_pointer_source",
    "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: THE BUFFER ORDER IS ``fused_electric_pair``'s, GROUP FOR GROUP, with two
#: differences and no others: the three BFAST state pointers sit immediately after
#: the curl's three sources (where the magnetic twin puts them), and the six curl
#: coefficient vectors have become the ONE ``cpml`` pack in the slot the first of
#: them held.
_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
    float k1_0; float k2_0; float k1_1; float k2_1; float k1_2; float k2_2;
__PACK_FIELDS__
};

kernel void bfast_fused_electric_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device float*       s0      [[buffer(9)]],
    device float*       s1      [[buffer(10)]],
    device float*       s2      [[buffer(11)]],
    device float*       e0      [[buffer(12)]],
    device float*       e1      [[buffer(13)]],
    device float*       e2      [[buffer(14)]],
    device float*       w0      [[buffer(15)]],
    device float*       w1      [[buffer(16)]],
    device float*       w2      [[buffer(17)]],
    device const float* ie0     [[buffer(18)]],
    device const float* ie1     [[buffer(19)]],
    device const float* ie2     [[buffer(20)]],
    device const float* cpml    [[buffer(21)]],
    device const float* kp0     [[buffer(22)]],
    device const float* km0     [[buffer(23)]],
    device const float* kp1     [[buffer(24)]],
    device const float* km1     [[buffer(25)]],
    device const float* kp2     [[buffer(26)]],
    device const float* km2     [[buffer(27)]],
    constant Params&    prm     [[buffer(28)]],
    uint idx [[thread_position_in_grid]])
{
    // THE ELEVEN SCALARS AND THE SIX PACKED VECTORS ARE UNPACKED INTO THE CERTIFIED
    // BODIES' OWN NAMES, once, before any of the lifted text runs. Everything below
    // this line is then character-for-character what `bfast_curl.bfast_curl_source`
    // and `shaders.constitutive_source` emit, plus the wall clear and the one seam
    // substitution.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float k1_0 = prm.k1_0, k2_0 = prm.k2_0;
    float k1_1 = prm.k1_1, k2_1 = prm.k2_1;
    float k1_2 = prm.k1_2, k2_2 = prm.k2_2;

    // --- the curl half's PML coefficient vectors, one allocation, six offsets ---
    // READ-ONLY: nothing on the device writes `cpml` (the BFAST state s0-2 is the
    // half's device-written set and keeps its own bindings for that reason).
__PACK_PROLOGUE__

__BODY__
}
"""


def certified_bfast_curl_body(codes: Sequence[int],
                              contract: str = shaders.CONTRACT_OFF) -> str:
    """The BFAST ``step_D`` curl kernel's BODY, lifted from its own emitter.

    ``backward`` is True and never a parameter — this is the D seam, and
    ``step_B``'s forward strides are the magnetic twin's product — and ``has_bfast``
    is True and never a parameter either: a ``has_bfast=False`` weld would be the
    ordinary product under a second name.
    """
    source = bfast_curl.bfast_curl_source(codes, True, contract, True)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified BFAST curl source no longer carries the body anchor; "
            "this family lifts that body and would otherwise splice a truncated "
            "kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified BFAST curl source does not end with '}'")
    return body[: -len("}\n")]


def bfast_fused_electric_pair_source(codes: Sequence[int],
                                     zero_metal: Sequence[bool],
                                     contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, walls, contraction mode).

    THE SPLICE IS :func:`.fused_electric_pair.fused_electric_pair_source`'S, step
    for step, with :func:`certified_bfast_curl_body` where its
    ``certified_curl_body`` stands and the pack prologue re-creating the six curl
    coefficient names.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")

    curl = certified_bfast_curl_body(codes, contract)
    # THE TAIL MUST BE PRESENT — see the magnetic twin for why its absence would be
    # the ordinary weld wearing this family's name.
    for statement in ("float st0 = s0[ii];",
                      "s0[ii] = st0 + adv0;",
                      "curl0 = curl0 - adv0;"):
        if statement not in curl:
            raise AssertionError(
                f"the lifted BFAST curl body does not carry {statement!r}; this "
                f"weld would be the ordinary product under another name")
    if _CURL_STORE not in curl:
        raise AssertionError(
            "the certified BFAST curl body no longer stores the three targets on "
            "one line; the wall clear has no anchor to sit in front of")
    head, tail = curl.split(_CURL_STORE, 1)
    curl = "".join((
        head,
        "    // --- zero_metal_D, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The OFF-DIAGONAL for D: Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z.\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    electric = certified_constitutive_body(contract)
    # THE SEAM — the plain D/E weld's three lines, unchanged: the reload becomes the
    # live register, the inverse-epsilon factor stays on the right and D stays on
    # the LEFT (stepping.py:1011-1013).
    for target in range(3):
        old = f"    float src{target} = g{target}[ii] * ie{target}[ii];\n"
        if old not in electric:
            raise AssertionError(
                f"the certified E constitutive body no longer reads its source as "
                f"{old.strip()!r}; the seam has no anchor")
        electric = electric.replace(
            old,
            f"    // THE SEAM: the register step_D just wrote, not a reload of "
            f"D{'xyz'[target]}.\n"
            f"    float src{target} = v{target} * ie{target}[ii];\n")
    # THE CURL'S `g` IS H AND THE CONSTITUTIVE'S `g` WAS D.
    for target in range(3):
        if f"g{target}[" in electric:
            raise AssertionError(
                f"the lifted E constitutive half still reads g{target}, which in "
                f"the fused signature is H{'xyz'[target]} and not D{'xyz'[target]}")

    body = "".join((
        curl,
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml:2083) --\n",
        electric,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__PACK_FIELDS__": coefficient_pack.params_fields(PACKED_VECTORS),
        "__PACK_PROLOGUE__": coefficient_pack.prologue("cpml", PACKED_VECTORS),
        "__BODY__": body,
    })


def compile_bfast_fused_electric_pair(codes: Sequence[int],
                                      zero_metal: Sequence[bool],
                                      contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, walls, mode)."""
    return compile_source(bfast_fused_electric_pair_source(
        codes, zero_metal, contract)).bfast_fused_electric_pair_step


#: The pointer names of the UNPACKED fused signature, in the order the two certified
#: halves declare them. Written once and consumed by both refuted emitters, so the
#: number this family says it was over by is the number both refusals are built
#: from.
_UNPACKED_WRITTEN: Tuple[str, ...] = ("f0", "f1", "f2", "u0", "u1", "u2",
                                      "s0", "s1", "s2",
                                      "e0", "e1", "e2", "w0", "w1", "w2")
_UNPACKED_READ: Tuple[str, ...] = ("g0", "g1", "g2",
                                   "ie0", "ie1", "ie2",
                                   "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                                   "kp0", "km0", "kp1", "km1", "kp2", "km2")


def _touching_body(written: Sequence[str], read: Sequence[str], guard: str,
                   scale: str) -> List[str]:
    """A body that READS every input and WRITES every output, so dead-code
    elimination cannot decide the answer — the SIGNATURE is what is measured."""
    touch = " + ".join(f"{name}[0]" for name in read)
    return [f"    {guard}",
            f"    float touch = {touch};",
            f"    f0[idx] = f0[idx] + touch;",
            *(f"    {name}[idx] = {name}[idx] * {scale};" for name in written)]


def refuted_unpacked_pointer_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 34-binding signature this platform REFUSES — 33 pointers plus ``Params``.

    THIS IS THE MEASUREMENT THE PACK EXISTS FOR: the fused pair built the way every
    other D/E product on this backend is built — eleven scalars packed into one
    ``constant Params&``, every vector its own pointer — over the ceiling by three.
    Not shipped and not launchable: the gate compiles it and requires the failure,
    which is what turns :data:`UNPACKED_POINTER_BINDINGS` from the board's argument
    into a measurement.
    """
    lines: List[str] = []
    slot = 0
    for name in _UNPACKED_WRITTEN:
        lines.append(f"    device float*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in _UNPACKED_READ:
        lines.append(f"    device const float* {name:<8}[[buffer({slot})]],")
        slot += 1
    if slot != UNPACKED_POINTERS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, UNPACKED_POINTERS))
    lines.append(f"    constant Params&    prm     [[buffer({slot})]],")
    slot += 1
    if slot != UNPACKED_POINTER_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, UNPACKED_POINTER_BINDINGS))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        templates.contraction_pragma(contract),
        "",
        "struct Params {",
        "    uint nx; uint ny; uint nz; uint n_elem; float dtdx;",
        "    float k1_0; float k2_0; float k1_1; float k2_1; float k1_2; float k2_2;",
        "};",
        "",
        "kernel void refuted_unpacked_pointers(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        *_touching_body(_UNPACKED_WRITTEN, _UNPACKED_READ,
                        "if (idx >= prm.n_elem) { return; }",
                        "prm.dtdx"),
        "}",
        "",
    ))


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 44-binding separate-scalar signature this platform REFUSES.

    Kept beside the pointer refusal so a reader cannot wonder whether the scalars
    were the problem: even with them packed the pointers alone are three over.
    """
    lines: List[str] = []
    slot = 0
    for name in _UNPACKED_WRITTEN:
        lines.append(f"    device float*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in _UNPACKED_READ:
        lines.append(f"    device const float* {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("nx", "ny", "nz", "n_elem"):
        lines.append(f"    constant uint&      {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("dtdx", "k1_0", "k2_0", "k1_1", "k2_1", "k1_2", "k2_2"):
        lines.append(f"    constant float&     {name:<8}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        templates.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalars(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        *_touching_body(_UNPACKED_WRITTEN, _UNPACKED_READ,
                        "if (idx >= n_elem) { return; }", "dtdx"),
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_bfast_fused_electric_pair_coverage(fields: Any, pml: Any,
                                             sources: Any = None,
                                             residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_D`` -> wall -> ``update_E`` on a BFAST run?

    The conjunction of the two halves' OWN shipped predicates plus the seam
    clauses, which is the magnetic twin's construction on the other seam:

    * :func:`.bfast_curl.bfast_pml_curl_coverage` on ``step_D``;
    * :func:`.bfast_curl.bfast_run_constitutive_coverage` on side ``"E"``.
    """
    reasons: List[str] = []

    curl = bfast_curl.bfast_pml_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = bfast_curl.bfast_run_constitutive_coverage(fields, pml, "E",
                                                          residency)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299) and the one corpus row carries one — the repair is not
    # an optimisation here; it is the product. IGNORANCE IS NEVER AN EMPTY SET.
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

    # THE STATE VOLUMES. The Tustin filter's per-component state is bound as three
    # POINTERS; the curl half's own predicate checks them, restated by NAME because
    # the weld is what BINDS them.
    for name in BFAST_STATE_NAMES["step_D"]:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated; this weld binds it directly")

    # THE TWO FILLS, restated by NAME for THIS seam.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A polarization would make the constitutive source (D - sum P) rather than D.
    # The constitutive half already refuses it; restated because this family's
    # kernel bakes the plain product.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalBfastFusedElectricPairPlan(KernelPlan):
    """ONE dispatch performing three driver passes on a BFAST run's D seam.

    ``launches_per_run`` is the base's 1: this plan unpacks one argument tuple and
    calls one function. ``runs`` is an alias of ``launches``.
    """

    __slots__ = ("residency", "volumes", "codes", "zero_metal", "shape", "dtdx",
                 "ks", "pack_layout", "params")

    family = "fused BFAST PML D-curl/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "zero_metal", "ks")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], zero_metal: Sequence[bool],
                 shape: Sequence[int], dtdx: float, ks: Sequence[float],
                 pack_layout: Any, params: Any, pointers: Sequence[Any],
                 functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        # Already f32-rounded by `bfast_curl_coefficients`; float() keeps the bits.
        self.ks = tuple(float(value) for value in ks)
        if len(self.ks) != 6:
            raise ValueError(f"ks must be the six (k1,k2) scalars, got {ks!r}")
        self.pack_layout = pack_layout
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def _params_tensor(shape: Sequence[int], dtdx: float, ks: Sequence[float],
                   offsets: Sequence[int], device: str) -> Any:
    """The eleven scalars AND the six pack offsets as one 68-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every fused
    pair's ``_params_tensor`` gives. THE OFFSET FIELDS ARE THE PACK'S WHOLE COST:
    six uints, in the pack's own order, and they are what buys back five buffer
    bindings. The field order here and the struct order in :data:`_TEMPLATE` are
    BOTH generated from :data:`PACKED_VECTORS`, so the two cannot drift into a
    reinterpretation.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    fields = ([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
               ("dtdx", "<f4"), ("k1_0", "<f4"), ("k2_0", "<f4"),
               ("k1_1", "<f4"), ("k2_1", "<f4"), ("k1_2", "<f4"), ("k2_2", "<f4")]
              + coefficient_pack.record_dtype_fields(PACKED_VECTORS))
    record = np.zeros(1, dtype=np.dtype(fields))
    nx, ny, nz = (int(n) for n in shape)
    offsets = tuple(int(offset) for offset in offsets)
    if len(offsets) != len(PACKED_VECTORS):  # pragma: no cover - arithmetic guard
        raise AssertionError((len(offsets), len(PACKED_VECTORS)))
    record[0] = ((nx, ny, nz, nx * ny * nz, np.float32(dtdx))
                 + tuple(np.float32(value) for value in ks) + offsets)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_bfast_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        params: Optional[Any] = None,
        ) -> Optional[MetalBfastFusedElectricPairPlan]:
    """Build the fused BFAST D/E plan, or ``None`` when the seam is refused.

    ``functions`` is the KERNEL mutation seam and ``params`` the HOST one — the
    packed offsets are host arithmetic, so a leg that corrupts one has to be able
    to hand this builder a struct the shipped code would not have written. Dropping
    either is a silent DISARMING.
    """
    if not metal_bfast_fused_electric_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    import numpy as np  # noqa: PLC0415
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))
    walls = zero_metal_axes(grid)
    # THE SIX SCALARS ARE THE CURL ARM'S OWN, called with the grid's OWN declared
    # invariance flags (never a shape test) and the electric side.
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    ks = bfast_curl_coefficients(grid.bfast_scaled_k, invariant, magnetic=False)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    flux = [bind(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Dx", "Dy", "Dz")]
    magnetic = [bind(name, getattr(fields, name)) for name in ("Hx", "Hy", "Hz")]
    state = [bind(name, getattr(fields, name))
             for name in BFAST_STATE_NAMES["step_D"]]
    stored = [bind(name, getattr(fields, name)) for name in ("Ex", "Ey", "Ez")]
    workspace = [bind("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Ex", "Ey", "Ez")]
    inverse_epsilon = [
        bind("inv_eps_" + name, fields.inverse_epsilon_for(name), constant=True)
        for name in ("Ex", "Ey", "Ez")]
    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER
    # one (SUB_STEPS['step_D']['suffix'] is '' and CONSTITUTIVE_SIDES['E']
    # ['half_integer'] is True) — the opposite pairing to the B/H twin.
    #
    # THE PACK IS THE CURL GROUP AND NOTHING ELSE. Six read-only vectors, one
    # buffer, six element offsets in Params. The constitutive group stays six
    # separate pointers because it does not have to move.
    curl_pack, pack_layout = coefficient_pack.packed_mirror(
        residency, "pml_pack:bfast_curl:step_D", np, PACKED_VECTORS,
        [getattr(pml, f"{stem}_{axis}") for axis in "xyz"
         for stem in ("kms", "sinv")])
    volumes.append("pml_pack:bfast_curl:step_D")
    constitutive_coefficients = [
        bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + magnetic + state + stored + workspace
                + inverse_epsilon + [curl_pack] + constitutive_coefficients)
    if len(pointers) != PACKED_POINTERS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(pointers), PACKED_POINTERS))
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_bfast_fused_electric_pair(codes, walls, mode)

    dtdx = grid.dt / grid.dx
    record = (_params_tensor(grid.shape, dtdx, ks, pack_layout.offsets,
                             residency.device) if params is None else params)
    return MetalBfastFusedElectricPairPlan(
        residency, volumes, codes, walls, grid.shape, dtdx, ks, pack_layout,
        record, pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"BFAST fused electric pair cannot fill {slot}",))
    return metal_bfast_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalBfastFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_bfast_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False`` — enumerable, never selectable."""
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "BFAST fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="BFAST fused electric D/E pair: ",
                          noun="fused BFAST PML D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
