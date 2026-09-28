"""The conductive PML ``step_D`` welded into the ORDINARY ``update_E``, both packed.

THE SECOND of the two cells the 2026-09-01 board scored ``UNFUSABLE ON METAL`` that
the SHIPPED :mod:`.coefficient_pack` technique closes — and the widest pack in the
tree: BOTH halves' coefficient vectors ride one buffer::

    D->E  (conductive PML curl, ordinary)   tests:TestAdjointSolver.test_damping
        conductive_pml.curl 24p + shaders.constitutive_E 18p, sharing 3
        -> 39 pointers, ceiling 30      UNFUSABLE ON METAL   (over by 9)

CROSS-BACKEND FACT, from the residue audit: the board's ``belongs to the CUDA
track`` position is unfulfilled — CUDA's same cell is priced POINTWISE-BUILDABLE
and serves 0 (no conductive-curl electric pair with a deposit carry exists there)
— so before this module NO backend served this instance.

WHAT MOVED IS NOT THE CEILING. One pack holds the TWELVE disjointly-named
read-only vectors of both halves — the curl's ``kmx``/``sinvx``/... and the
constitutive's ``kp0``/``km0``/... — with twelve element offsets in ``Params``,
saving eleven pointers: 39 -> 28, bound at :data:`PACKED_BINDINGS` = 29. The six
read-only conductivity VOLUMES (``cf``/``ci``) also meet the pack's
read-only-only rule but stay separate: the pack's members are PER-AXIS VECTORS
whose lengths are grid extents, and folding three-dimensional volumes in beside
them would make the offsets a mixed-geometry table for no binding need — 29 is
already two under the ceiling. The device-written ``f``/``u``/``c``/``w`` volumes
rightly stay unpacked (the read-only rule), as does everything the repair writes.

THE CURL HALF IS LIFTED WHOLE AND UNEDITED — decode, ghosts, loads, curl, mask,
per-target four-branch conductive tails, every in-place store — and the wall clear
plus the seam then work on a RELOAD: after the certified text stores each stepped
D component, the weld re-reads its own cell (the same thread, the value it just
wrote), masks the wall on the register, stores the masked value back, and hands
the register to the constitutive half. That is byte-for-byte the driver's own
order (step_D stores; zero_metal_D overwrites the wall plane; update_E reads), it
keeps the certified conductive tail text at ZERO edits, and the reload is the
declared-equivalence pattern the shipped gates already measure ("seam reloads the
flux it just stored").

THE SEAM'S CLAUSES. The one corpus row declares an ELECTRIC source, so
:data:`CARRIES_DEPOSIT_REPAIR` is True (split-field path — the absorber is
ACTIVE). The conductive whole-volume-rescale hazard that priced the no-PML
conductive cell at 0 of 2 is consulted here through the SAME clause, imported from
:mod:`.no_pml_conductive_fused_electric_pair` — one home — and with the driver's
sparse per-deposit-cell rescale landed that clause refuses only a scaled source
that publishes no deposit table, which no in-tree electric source class is.

NOT WIRED AS AN ARM; reached through ``launch.FUSED_PAIR_ARMS`` instead.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.conductivity import conductive_targets
from ..triton_kernels.coverage import Coverage
from . import coefficient_pack, conductive_pml, shaders, templates
from .coverage import constitutive_coverage, zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .fused_electric_pair import (
    _BODY_ANCHOR,
    certified_constitutive_body,
    zero_metal_mask,
)
from .no_pml_conductive_fused_electric_pair import (
    _scaled_conductive_injection_reasons,
)
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: this cell the flag IS the row: the one corpus instance declares an ELECTRIC
#: source inside the seam. The repair path is the split-field default — the curl
#: half requires an ACTIVE absorber, so the recurrence the seam inverts is always
#: ``_apply_constitutive_pml``'s.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "conductive_fused_electric_pair"

#: The slot this arm holds a row on, and the driver passes one launch performs.
SLOT = "step_D"
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: BOTH halves' read-only per-axis coefficient vectors, IN PACK ORDER: the curl's
#: six, then the constitutive's six. Disjointly named, so one prologue re-creates
#: all twelve under their certified names.
PACKED_VECTORS: Tuple[str, ...] = ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                                   "kp0", "km0", "kp1", "km1", "kp2", "km2")

#: THREE BINDING COUNTS, ALL COMPILED BY THE GATE:
#:
#: * :data:`SEPARATE_SCALAR_BINDINGS` — 39 pointers + 5 separate scalars.
#: * :data:`UNPACKED_POINTER_BINDINGS` — 39 pointers + one packed ``Params&``: the
#:   shape the board's ``UNFUSABLE ON METAL`` verdict priced, REFUSED at 40
#:   bindings against 31 (over by nine).
#: * :data:`PACKED_BINDINGS` — the shipped shape, 28 pointers plus ``Params``.
SEPARATE_SCALAR_BINDINGS = 44
UNPACKED_POINTER_BINDINGS = 40
PACKED_BINDINGS = 29

#: The pointer counts, spelled apart from the binding counts.
PACKED_POINTERS = 28
UNPACKED_POINTERS = 39

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the conductive fused electric pair binds {PACKED_BINDINGS} buffers and "
        f"Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "PACKED_BINDINGS", "PACKED_POINTERS",
    "PACKED_VECTORS", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "UNPACKED_POINTERS", "UNPACKED_POINTER_BINDINGS",
    "MetalConductiveFusedElectricPairPlan", "certified_conductive_curl_body",
    "compile_conductive_fused_electric_pair",
    "conductive_fused_electric_pair_source",
    "metal_conductive_fused_electric_pair_coverage",
    "plan_metal_conductive_fused_electric_pair",
    "refuted_separate_scalar_source", "refuted_unpacked_pointer_source",
    "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
__PACK_FIELDS__
};

kernel void conductive_fused_electric_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device float*       c0      [[buffer(6)]],
    device float*       c1      [[buffer(7)]],
    device float*       c2      [[buffer(8)]],
    device const float* cf0     [[buffer(9)]],
    device const float* cf1     [[buffer(10)]],
    device const float* cf2     [[buffer(11)]],
    device const float* ci0     [[buffer(12)]],
    device const float* ci1     [[buffer(13)]],
    device const float* ci2     [[buffer(14)]],
    device const float* g0      [[buffer(15)]],
    device const float* g1      [[buffer(16)]],
    device const float* g2      [[buffer(17)]],
    device float*       e0      [[buffer(18)]],
    device float*       e1      [[buffer(19)]],
    device float*       e2      [[buffer(20)]],
    device float*       w0      [[buffer(21)]],
    device float*       w1      [[buffer(22)]],
    device float*       w2      [[buffer(23)]],
    device const float* ie0     [[buffer(24)]],
    device const float* ie1     [[buffer(25)]],
    device const float* ie2     [[buffer(26)]],
    device const float* cpack   [[buffer(27)]],
    constant Params&    prm     [[buffer(28)]],
    uint idx [[thread_position_in_grid]])
{
    // THE FIVE SCALARS AND THE TWELVE PACKED VECTORS ARE UNPACKED INTO THE
    // CERTIFIED BODIES' OWN NAMES, once, before any of the lifted text runs.
    // Everything below this line is then character-for-character what
    // `conductive_pml.conductive_pml_curl_source` and `shaders.constitutive_source`
    // emit, plus the wall clear on a reload and the one seam substitution.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;

    // --- BOTH halves' coefficient vectors, one allocation, twelve offsets -------
    // READ-ONLY: nothing on the device writes `cpack`. The conductivity volumes
    // cf/ci are read-only too but are whole VOLUMES, not per-axis vectors, and
    // keep their own bindings; f/u/c/w are device-written and may never enter a
    // pack at all.
__PACK_PROLOGUE__

__BODY__
}
"""


def certified_conductive_curl_body(codes: Sequence[int],
                                   conductive: Sequence[bool],
                                   contract: str = shaders.CONTRACT_OFF) -> str:
    """The conductive PML ``step_D`` curl kernel's BODY, lifted from its own emitter.

    Not a transcription: this is
    :func:`.conductive_pml.conductive_pml_curl_source`'s own output with the
    ``#include``/signature preamble and the closing brace removed — INCLUDING every
    per-target four-branch tail and every in-place store, unedited. ``backward`` is
    True and never a parameter: this is the D seam, and the certified emitter is
    the one home of the branch structure.
    """
    source = conductive_pml.conductive_pml_curl_source(codes, True, conductive,
                                                       contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified conductive PML curl source no longer carries the body "
            "anchor; this family lifts that body and would otherwise splice a "
            "truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(
            "the certified conductive PML curl source does not end with '}'")
    return body[: -len("}\n")]


#: Per target, the certified tail's FINAL store text for each arm — the anchors the
#: splice requires so a restructured emitter reaches this family as an
#: AssertionError rather than as a silently un-stored component.
def _tail_store_anchors(index: int, conductive: bool) -> Tuple[str, ...]:
    if conductive:
        return (
            f"c{index}[ii] = c{index}_new; u{index}[ii] = u{index}_new; "
            f"f{index}[ii] = f{index}_new;",
            f"        f{index}[ii] = ((f{index}_previous * cf{index}[ii]) - "
            f"curl{index}) * ci{index}[ii];",
        )
    return (f"    u{index}[ii] = u{index}_new;",
            f"    f{index}[ii] = f{index}_new;")


def conductive_fused_electric_pair_source(
        codes: Sequence[int], conductive: Sequence[bool],
        zero_metal: Sequence[bool], contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundaries, sigma flags, walls, contraction).

    THE CURL TEXT IS UNEDITED; the weld appends a reload-mask-store epilogue and
    then the certified constitutive half with the three-line seam substitution
    reading the reloaded registers.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    conductive = tuple(bool(flag) for flag in conductive)
    if len(conductive) != 3:
        raise ValueError(f"conductive must be a per-axis triple, got {conductive!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")

    curl = certified_conductive_curl_body(codes, conductive, contract)
    for index in range(3):
        for anchor in _tail_store_anchors(index, conductive[index]):
            if anchor not in curl:
                raise AssertionError(
                    f"the certified conductive tail for target {index} no longer "
                    f"carries {anchor!r}; the reload epilogue would read a value "
                    f"nothing stored")

    epilogue = "\n".join((
        "",
        "    // --- zero_metal_D on a RELOAD (stepping._zero_metal:2232) ----------",
        "    // The certified tails above stored every stepped D in place; this",
        "    // thread re-reads its OWN cell (the value it just wrote), masks the",
        "    // wall on the register, stores the masked value back, and hands the",
        "    // register to the constitutive half. Same bytes, driver order:",
        "    // step_D stores, zero_metal_D overwrites, update_E reads.",
        "    float v0 = f0[ii];",
        "    float v1 = f1[ii];",
        "    float v2 = f2[ii];",
        zero_metal_mask(zero_metal),
        "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;",
        "",
    ))

    electric = certified_constitutive_body(contract)
    # THE SEAM — the plain D/E weld's three lines, unchanged.
    for target in range(3):
        old = f"    float src{target} = g{target}[ii] * ie{target}[ii];\n"
        if old not in electric:
            raise AssertionError(
                f"the certified E constitutive body no longer reads its source as "
                f"{old.strip()!r}; the seam has no anchor")
        electric = electric.replace(
            old,
            f"    // THE SEAM: the register the wall clear just settled, not a "
            f"reload of D{'xyz'[target]}.\n"
            f"    float src{target} = v{target} * ie{target}[ii];\n")
    # THE CURL'S `g` IS H AND THE CONSTITUTIVE'S `g` WAS D.
    for target in range(3):
        if f"g{target}[" in electric:
            raise AssertionError(
                f"the lifted E constitutive half still reads g{target}, which in "
                f"the fused signature is H{'xyz'[target]} and not D{'xyz'[target]}")

    body = "".join((
        curl,
        epilogue,
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml:2083) --\n",
        electric,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__PACK_FIELDS__": coefficient_pack.params_fields(PACKED_VECTORS),
        "__PACK_PROLOGUE__": coefficient_pack.prologue("cpack", PACKED_VECTORS),
        "__BODY__": body,
    })


def compile_conductive_fused_electric_pair(
        codes: Sequence[int], conductive: Sequence[bool],
        zero_metal: Sequence[bool], contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, sigma flags, walls, mode)."""
    return compile_source(conductive_fused_electric_pair_source(
        codes, conductive, zero_metal,
        contract)).conductive_fused_electric_pair_step


#: The pointer names of the UNPACKED fused signature, in the order the two
#: certified halves declare them.
_UNPACKED_WRITTEN: Tuple[str, ...] = ("f0", "f1", "f2", "u0", "u1", "u2",
                                      "c0", "c1", "c2",
                                      "e0", "e1", "e2", "w0", "w1", "w2")
_UNPACKED_READ: Tuple[str, ...] = ("cf0", "cf1", "cf2", "ci0", "ci1", "ci2",
                                   "g0", "g1", "g2",
                                   "ie0", "ie1", "ie2",
                                   "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                                   "kp0", "km0", "kp1", "km1", "kp2", "km2")


def _touching_body(written: Sequence[str], read: Sequence[str], guard: str,
                   scale: str) -> List[str]:
    touch = " + ".join(f"{name}[0]" for name in read)
    return [f"    {guard}",
            f"    float touch = {touch};",
            f"    f0[idx] = f0[idx] + touch;",
            *(f"    {name}[idx] = {name}[idx] * {scale};" for name in written)]


def refuted_unpacked_pointer_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 40-binding signature this platform REFUSES — 39 pointers plus ``Params``.

    THIS IS THE MEASUREMENT THE PACK EXISTS FOR: the shape the board's verdict
    priced, over the ceiling by nine.
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
        "struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };",
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
    """The 44-binding separate-scalar signature this platform REFUSES."""
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
    lines.append(f"    constant float&     dtdx    [[buffer({slot})]],")
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

def metal_conductive_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May ONE dispatch span conductive ``step_D`` -> wall -> ``update_E``?

    The conjunction of the two halves' OWN shipped predicates plus the seam
    clauses:

    * :func:`.conductive_pml.metal_conductive_pml_curl_coverage` on ``step_D``;
    * :func:`.coverage.constitutive_coverage` on side ``"E"`` — the ``ordinary``
      arm's own body, which is what the board's cell names.
    """
    reasons: List[str] = []

    curl = conductive_pml.metal_conductive_pml_curl_coverage(
        fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = constitutive_coverage(fields, pml, "E", residency)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves — on a
    # conductive run through _inject_electric_through_conductivity (driver.py:3306).
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3294-3299), and "
            f"on a conductive row through _inject_electric_through_conductivity "
            f"(driver.py:3306)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # THE CONDUCTIVE INJECTION'S REMAINING HAZARD — imported from the no-PML
    # conductive sibling, one home: with the driver's sparse rescale, only a scaled
    # source that publishes NO deposit table still routes through the whole-volume
    # fallback, and only that case is refused.
    reasons.extend(_scaled_conductive_injection_reasons(fields, sources))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

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
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalConductiveFusedElectricPairPlan(KernelPlan):
    """ONE dispatch performing three driver passes on a conductive PML run's D seam.

    ``launches_per_run`` is the base's 1. ``runs`` is an alias of ``launches``.
    """

    __slots__ = ("residency", "volumes", "codes", "conductive", "zero_metal",
                 "shape", "dtdx", "pack_layout", "params")

    family = "conductive fused PML D-curl/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "conductive", "zero_metal")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], conductive: Sequence[bool],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 pack_layout: Any, params: Any, pointers: Sequence[Any],
                 functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.conductive = tuple(bool(flag) for flag in conductive)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        self.pack_layout = pack_layout
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def _params_tensor(shape: Sequence[int], dtdx: float, offsets: Sequence[int],
                   device: str) -> Any:
    """The five scalars AND the twelve pack offsets as one 68-byte device record."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    fields = ([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
               ("dtdx", "<f4")]
              + coefficient_pack.record_dtype_fields(PACKED_VECTORS))
    record = np.zeros(1, dtype=np.dtype(fields))
    nx, ny, nz = (int(n) for n in shape)
    offsets = tuple(int(offset) for offset in offsets)
    if len(offsets) != len(PACKED_VECTORS):  # pragma: no cover - arithmetic guard
        raise AssertionError((len(offsets), len(PACKED_VECTORS)))
    record[0] = (nx, ny, nz, nx * ny * nz, np.float32(dtdx)) + offsets
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_conductive_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        params: Optional[Any] = None,
        ) -> Optional[MetalConductiveFusedElectricPairPlan]:
    """Build the conductive fused D/E plan, or ``None`` when the seam is refused.

    ``functions`` is the KERNEL mutation seam and ``params`` the HOST one — the
    packed offsets are host arithmetic. Dropping either is a silent DISARMING.
    """
    if not metal_conductive_fused_electric_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    import numpy as np  # noqa: PLC0415
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))
    walls = zero_metal_axes(grid)
    flags = conductive_targets(fields, "step_D")

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    # The dead history/conductivity slots of a LOSSLESS target alias the target
    # tensor, exactly as the certified conductive curl plan binds them
    # (conductive_pml.py:221-232): the lossless tail never touches c/cf/ci, so the
    # alias is never read and never written through.
    targets = ("Dx", "Dy", "Dz")
    flux = [bind(name, getattr(fields, name)) for name in targets]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in targets]
    history = [
        (bind("f_cond_" + name, getattr(fields, "f_cond_" + name))
         if flags[index] else flux[index])
        for index, name in enumerate(targets)]
    condfac = [
        (bind(f"condfac_{name}:pml", fields.condfac_for(name), constant=True)
         if flags[index] else flux[index])
        for index, name in enumerate(targets)]
    condinv = [
        (bind(f"condinv_{name}:pml", fields.condinv_for(name), constant=True)
         if flags[index] else flux[index])
        for index, name in enumerate(targets)]
    magnetic = [bind(name, getattr(fields, name)) for name in ("Hx", "Hy", "Hz")]
    stored = [bind(name, getattr(fields, name)) for name in ("Ex", "Ey", "Ez")]
    workspace = [bind("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Ex", "Ey", "Ez")]
    inverse_epsilon = [
        bind("inv_eps_" + name, fields.inverse_epsilon_for(name), constant=True)
        for name in ("Ex", "Ey", "Ez")]
    # ONE PACK OVER BOTH HALVES' TWELVE VECTORS: the curl's integer-lattice
    # kms/sinv and the constitutive's half-integer kps/kms, in PACKED_VECTORS
    # order. The two lattices stay distinguishable through the offsets, which are
    # generated from the same tuple the struct fields are.
    pack, pack_layout = coefficient_pack.packed_mirror(
        residency, "pml_pack:conductive_pair:step_D", np, PACKED_VECTORS,
        [getattr(pml, f"{stem}_{axis}") for axis in "xyz"
         for stem in ("kms", "sinv")]
        + [getattr(pml, f"{stem}_{axis}_h") for axis in "xyz"
           for stem in ("kps", "kms")])
    volumes.append("pml_pack:conductive_pair:step_D")

    pointers = (flux + auxiliary + history + condfac + condinv + magnetic
                + stored + workspace + inverse_epsilon + [pack])
    if len(pointers) != PACKED_POINTERS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(pointers), PACKED_POINTERS))
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_conductive_fused_electric_pair(
                codes, flags, walls, mode)

    dtdx = grid.dt / grid.dx
    record = (_params_tensor(grid.shape, dtdx, pack_layout.offsets,
                             residency.device) if params is None else params)
    return MetalConductiveFusedElectricPairPlan(
        residency, volumes, codes, flags, walls, grid.shape, dtdx, pack_layout,
        record, pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"conductive fused electric pair cannot fill {slot}",))
    return metal_conductive_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalConductiveFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_conductive_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False`` — enumerable, never selectable."""
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "conductive fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="conductive fused electric D/E pair: ",
                          noun="conductive fused PML D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
