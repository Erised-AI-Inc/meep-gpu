"""Metal ``update_E`` for the fold + tensor-row + dispersion intersection.

**THE ROW IS MEEP'S DOUBLE ROW, IN SOFTWARE BINARY64 (ruling R7).** The source term
is the off-diagonal row ``gs*us + OFFDIAG(...) + OFFDIAG(...)`` over the ``D - P0 -
P1 ...`` volumes, which the array path forms in float64 and rounds once
(``stepping._offdiagonal_terms``, ``_rounded_row``). The kernel forms it as the
off-diagonal single does, statement for statement and through the same emitter
(:func:`.offdiag_update_e._row_assembly`, :mod:`.softfp64` with ``fz = true``):
``gs*us`` in float32, widened; per live term its float32 pair sum ``(near*u[i]) +
(far*u[i+s])`` over the ordered ``D - P`` gathers, wall-masked, scaled by MEEP's
double ``0.25`` and added left to right where the cell's chunk keeps the term; one
rounding to float32. A component with no live row keeps the float32 ``gs*us``.

**PRESENCE IS PER CHUNK (KERNELS.md §2).** The six 27-bit presence words
(``OffdiagPresence kw``) ride in ``Params`` before the chunk record and are read by
the single's own block (:func:`.offdiag_update_e.presence_block`, on the folded
codes); the host builds them from the array path's presence
(:func:`.offdiag_update_e.presence_for`) and the predicate refuses by name a
division the record cannot carry them on.

**THE PML TAIL IS THE ARRAY PATH'S PER-CHUNK SELECT.** The chunk record rides in the
kernel's ``Params`` (``ChunkBounds cb``, :func:`.chunk_bounds.params_alias`), and the
tail is the base constitutive kernel's own, character for character: on a cell whose
chunk carries the component's own (``dsigw``) layer ``prev`` is read before ``f_w``
is written and ``E = E + ((kps*src) - (kms*prev))``; everywhere else ``E = src`` and
``f_w`` is neither read nor written (``stepping._apply_constitutive_regions``). The
membership locals and their bounds block are the E side's
(``shaders.constitutive_bounds_block("E")``,
``shaders.constitutive_membership_flags("E")``), the predicate the base kernel reads.

THE SCALARS MOVE INTO ``Params``. The shader binds 27 volumes; the four scalars
(``nx``, ``ny``, ``nz``, ``n_elem``), the presence words and the chunk record ride in
one ``constant Params&`` (:data:`BINDING_COUNT` 28 of
:data:`.device.MAX_BUFFER_BINDINGS` 31), so neither costs a binding.

The tensor row gathers each partner component at four Yee addresses.  With a
susceptibility, *every one* of those gathers is ``D - P0 - P1 - ...`` in registered
order; composing the folded tensor shader with the pointwise dispersive shader
would therefore be wrong.  Three groups of eight live pole volumes would exceed the
binding ceiling, so the plan keeps three persistent ``MAX_POLES x n_elem`` pole
packs, one per D component, and the fused shader reads each pack by lane.  This
preserves the ordered arithmetic without materialising ``D - sum(P)`` volumes.

The pack refresh has two modes.  The standalone gate has no ADE plan, so it stages
the initial NumPy pole arrays.  In a composed step, the ADE plan has already
registered and then rotated the physical P mirrors; each pack lane copies directly
from that live MPS mirror, without pulling stale P data through NumPy.  Neither mode
enables production driver dispatch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import (
    COVERED_SUSCEPTIBILITY_KINDS,
    ELECTRIC_COMPONENTS,
    Coverage,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
    _volume_reasons,
)
from . import arms, shaders, templates
from . import chunk_bounds as _chunk_bounds
from . import folded_offdiag_update_e as _folded
from . import offdiag_update_e as _offdiag
from . import symmetry as _symmetry
from .coverage import _residency_declaration_reasons
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source

FAMILY = "folded_offdiag_dispersive_update_e"
SLOT = "update_E"
LABEL = "folded off-diagonal dispersive PML E"
MAX_POLES = 8
E_TERMS = _offdiag.E_TERMS
ROW_SLOTS = _offdiag.ROW_SLOTS

#: Buffer bindings the shader declares: 27 volumes and one ``Params`` carrying the
#: four scalars, the presence words and the chunk record.
BINDING_COUNT = 28

#: The ``Params`` binding index: last, after the 27 volumes.
PARAMS_BUFFER = 27

if BINDING_COUNT > MAX_BUFFER_BINDINGS:  # pragma: no cover - platform invariant
    raise RuntimeError("the packed-pole signature exceeds the measured binding limit")

__all__ = [
    "BINDING_COUNT",
    "FAMILY",
    "LABEL",
    "MAX_POLES",
    "MetalFoldedOffdiagDispersivePlan",
    "PARAMS_BUFFER",
    "folded_offdiag_dispersive_coverage",
    "folded_offdiag_dispersive_source",
    "params_record_dtype",
    "params_words",
    "plan_folded_offdiag_dispersive",
    "register_arms",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
__SOFTFP64__
__RECORD_STRUCT__
__PRESENCE_STRUCT__
struct Params {
    uint nx;
    uint ny;
    uint nz;
    uint n_elem;
__PRESENCE_MEMBER__
__RECORD_MEMBER__
};
__POLE_HELPERS__
kernel void folded_offdiag_dispersive_step(
    device float* f0 [[buffer(0)]], device float* f1 [[buffer(1)]],
    device float* f2 [[buffer(2)]], device float* w0 [[buffer(3)]],
    device float* w1 [[buffer(4)]], device float* w2 [[buffer(5)]],
    device const float* g0 [[buffer(6)]], device const float* g1 [[buffer(7)]],
    device const float* g2 [[buffer(8)]], device const float* e0 [[buffer(9)]],
    device const float* e1 [[buffer(10)]], device const float* e2 [[buffer(11)]],
    device const float* p0 [[buffer(12)]], device const float* p1 [[buffer(13)]],
    device const float* p2 [[buffer(14)]], device const float* u01 [[buffer(15)]],
    device const float* u02 [[buffer(16)]], device const float* u11 [[buffer(17)]],
    device const float* u12 [[buffer(18)]], device const float* u21 [[buffer(19)]],
    device const float* u22 [[buffer(20)]], device const float* kp0 [[buffer(21)]],
    device const float* km0 [[buffer(22)]], device const float* kp1 [[buffer(23)]],
    device const float* km1 [[buffer(24)]], device const float* kp2 [[buffer(25)]],
    device const float* km2 [[buffer(26)]], constant Params& prm [[buffer(27)]],
    uint idx [[thread_position_in_grid]])
{
    // The scalars live in ONE binding; copying them back into the names the
    // certified emitters use lets GUARD and the decode below be their own text.
    uint nx = prm.nx;
    uint ny = prm.ny;
    uint nz = prm.nz;
    uint n_elem = prm.n_elem;
__RECORD_ALIAS__
__PRESENCE_ALIAS__
__GUARD__
__DECODE__

    // --- the chunk record: each component's own-axis membership at this cell -----
__BOUNDS__

__PRESENCE__
    int di = i - 1, dj = j - 1, dk = k - 1;
    int ui = i + 1, uj = j + 1, uk = k + 1;
    bool dvx = true, dvy = true, dvz = true;
    bool uvx = true, uvy = true, uvz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
    float kp_0 = kp0[i], km_0 = km0[i];
    float kp_1 = kp1[j], km_1 = km1[j];
    float kp_2 = kp2[k], km_2 = km2[k];
    const bool dw0 = __DW0__, dw1 = __DW1__, dw2 = __DW2__;

    // The PML tail is the base constitutive kernel's per-chunk select
    // (stepping._apply_constitutive_regions): on a dsigw cell `prev` is read BEFORE
    // `fw` is written and the right-hand side is formed first, f + ((kps*src) -
    // (kms*prev)) (step_generic.cpp:689, :697); elsewhere the word is src itself and
    // fw is not touched.
    float prev0 = dw0 ? w0[ii] : 0.0f;
__SRC0__
    if (dw0) { w0[ii] = src0; }
    float a0 = src0;
    if (dw0) { a0 = f0[ii] + ((kp_0 * src0) - (km_0 * prev0)); }
    f0[ii] = a0;

    float prev1 = dw1 ? w1[ii] : 0.0f;
__SRC1__
    if (dw1) { w1[ii] = src1; }
    float a1 = src1;
    if (dw1) { a1 = f1[ii] + ((kp_1 * src1) - (km_1 * prev1)); }
    f1[ii] = a1;

    float prev2 = dw2 ? w2[ii] : 0.0f;
__SRC2__
    if (dw2) { w2[ii] = src2; }
    float a2 = src2;
    if (dw2) { a2 = f2[ii] + ((kp_2 * src2) - (km_2 * prev2)); }
    f2[ii] = a2;
}
"""


def params_record_dtype() -> np.dtype:
    """``Params``: four ``uint`` scalars, the presence words (``OffdiagPresence kw``),
    then the chunk record (``ChunkBounds cb``).

    Every member is a 4-byte word, so the six presence words sit at byte 16, the
    record 4-aligned at byte 40 with no padding, and the struct is ``40 + 100`` bytes.
    The presence precedes the record as it does in the off-diagonal welds' ``Params``.
    """
    record = _chunk_bounds.record_dtype()
    presence = _offdiag.presence_record_dtype()
    names = ["nx", "ny", "nz", "n_elem", "kw", "cb"]
    formats: List[Any] = ["<u4", "<u4", "<u4", "<u4", presence, record]
    offsets = [0, 4, 8, 12, 16, 16 + presence.itemsize]
    return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                     "itemsize": 16 + presence.itemsize + record.itemsize})


def params_words(shape: Sequence[int], chunk_bounds: Any,
                 presence: Any) -> np.ndarray:
    """The packed ``Params`` as ``int32`` words, the record checked against ``shape``.

    Refused by name: a missing record or one that disagrees with the shape
    (:func:`.chunk_bounds.plan_words`), and missing or malformed presence words
    (:func:`.offdiag_update_e.presence_plan_words`). Both are required, with no
    default.
    """
    checked = _chunk_bounds.plan_words(chunk_bounds, shape, f"{FAMILY} Params")
    words = _offdiag.presence_plan_words(presence, f"{FAMILY} Params")
    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["kw"] = tuple(words)
    record["cb"] = _chunk_bounds.pack(checked)
    return np.frombuffer(record.tobytes(), dtype=np.int32).copy()


def _pole_helpers(counts: Sequence[int]) -> str:
    lines = []
    for component, count in enumerate(counts):
        chain = "\n".join(
            f"    value = value - p[{lane}u * n_elem + uint(index)];"
            for lane in range(int(count)))
        lines.append("\n".join((
            f"static inline float dmp{component}(device const float* g, "
            "device const float* p, int index, uint n_elem) {",
            "    float value = g[uint(index)];", chain, "    return value;", "}")))
    return "\n".join(lines)


def _term(component: int, partner_axis: int, coefficient: str, tag: str,
          negate: Sequence[int]) -> str:
    own_axis = E_TERMS[component][2]
    _down, _up, down_valid, _up_valid, _extent = _offdiag._TWO_WAY_NAMES[
        "xyz"[partner_axis]]
    _odown, own_up, _odown_valid, own_up_valid, _oextent = _offdiag._TWO_WAY_NAMES[
        "xyz"[own_axis]]
    dmp = f"dmp{partner_axis}(g{partner_axis}, p{partner_axis}, {{index}}, n_elem)"
    down = _offdiag._index({partner_axis: _down})
    up = _offdiag._index({own_axis: own_up})
    corner = _offdiag._index({own_axis: own_up, partner_axis: _down})
    at = ("at_x", "at_y", "at_z")[partner_axis]
    dn = f"{down_valid} ? {dmp.format(index=down)} : 0.0f"
    cn = f"({own_up_valid} && {down_valid}) ? {dmp.format(index=corner)} : 0.0f"
    near_sign = f"({at} ? -dn_{tag} : dn_{tag})" if negate[partner_axis] else f"dn_{tag}"
    corner_sign = f"({at} ? -cn_{tag} : cn_{tag})" if negate[partner_axis] else f"cn_{tag}"
    return "\n".join((
        f"    float dn_{tag} = {dn};",
        f"    float cn_{tag} = {cn};",
        f"    float near_{tag} = {dmp.format(index='ii')} + {near_sign};",
        f"    float far_{tag} = ({own_up_valid} ? {dmp.format(index=up)} : 0.0f) "
        f"+ {corner_sign};",
        f"    float unear_{tag} = {coefficient}[ii];",
        f"    float ufar_{tag} = {own_up_valid} ? {coefficient}[{up}] : 0.0f;",
        f"    float sum_{tag} = (near_{tag} * unear_{tag}) "
        f"+ (far_{tag} * ufar_{tag});"))


def _component_source(component: int, row_mask: Sequence[int],
                      walls: Sequence[int], negate: Sequence[int]) -> str:
    """One component's ``src{c}``: the float32 diagonal, or the double row.

    ``gs`` is the ordered ``D - P0 - P1 ...`` gather at the cell (``dmp{c}``, the
    array path's ``displacement_minus_polarization_volumes`` order) and ``us`` its
    inverse epsilon. A live arm forms each term's float32 pair sum (:func:`_term`)
    and the double row with the off-diagonal single's own emitter
    (:func:`.offdiag_update_e._row_assembly`: the per-term wall masks, the kept
    terms added left to right, one rounding), so the row text is that family's.
    """
    gs = f"dmp{component}(g{component}, p{component}, ii, n_elem)"
    us = f"e{component}[ii]"
    live = [offset for offset in (0, 1) if row_mask[2 * component + offset]]
    if not live:
        return f"    float src{component} = {gs} * {us};"
    lines = [f"    float gs{component} = {gs};",
             f"    float us{component} = {us};"]
    tags = []
    for offset in live:
        tag = f"{component}{offset}"
        lines.append(_term(component, _offdiag.TRANSVERSE_PARTNERS[component][offset],
                           _offdiag._ROW_BUFFERS[2 * component + offset], tag,
                           negate))
        tags.append(tag)
    lines.append(_offdiag._row_assembly(component, tags, walls))
    return "\n".join(lines)


def folded_offdiag_dispersive_source(row_mask: Sequence[int], codes: Sequence[int],
                                      walls: Sequence[int], negate: Sequence[int],
                                      counts: Sequence[int],
                                      contract: str = shaders.CONTRACT_OFF) -> str:
    row_mask, codes = tuple(map(int, row_mask)), tuple(map(int, codes))
    walls, negate, counts = tuple(map(int, walls)), tuple(map(int, negate)), tuple(map(int, counts))
    if len(row_mask) != 6 or not any(row_mask):
        raise ValueError("row_mask must contain at least one of the six tensor slots")
    if len(codes) != len(walls) != len(negate):
        raise ValueError("codes, walls, and negate must be axis triples")
    if len(counts) != 3 or any(count < 0 or count > MAX_POLES for count in counts):
        raise ValueError(f"counts must be three values in [0, {MAX_POLES}]")
    if not any(counts):
        raise ValueError("a dispersive intersection requires at least one live pole")
    flags = shaders.constitutive_membership_flags("E")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_MEMBER__": _chunk_bounds.params_member(),
        "__POLE_HELPERS__": _pole_helpers(counts),
        "__RECORD_ALIAS__": _chunk_bounds.params_alias(),
        **_offdiag.weld_presence_substitutions(),
        "__PRESENCE__": _offdiag.presence_block(row_mask, codes),
        "__GUARD__": templates.GUARD,
        "__BOUNDS__": shaders.constitutive_bounds_block("E"),
        "__DW0__": flags[0], "__DW1__": flags[1], "__DW2__": flags[2],
        "__DECODE__": """    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii = int(idx);
    int k = ii % nzi;
    int plane = ii / nzi;
    int j = plane % nyi;
    int i = plane / nyi;""",
        "__GHOST_X__": _folded._folded_two_way_ghost("x", codes[0]),
        "__GHOST_Y__": _folded._folded_two_way_ghost("y", codes[1]),
        "__GHOST_Z__": _folded._folded_two_way_ghost("z", codes[2]),
        "__SRC0__": _component_source(0, row_mask, walls, negate),
        "__SRC1__": _component_source(1, row_mask, walls, negate),
        "__SRC2__": _component_source(2, row_mask, walls, negate),
    })


def compile_folded_offdiag_dispersive(row_mask: Sequence[int], codes: Sequence[int],
                                      walls: Sequence[int], negate: Sequence[int],
                                      counts: Sequence[int],
                                      contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(folded_offdiag_dispersive_source(
        row_mask, codes, walls, negate, counts, contract)).folded_offdiag_dispersive_step


def _poles(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    states = tuple(getattr(fields, "polarizations", ()) or ())
    return {component: tuple(state for state in states
                             if _call(state, "drives", component, default=False))
            for component in ELECTRIC_COMPONENTS}


def _pole_reasons(fields: Any, shape: Sequence[int]) -> List[str]:
    reasons: List[str] = []
    outputs = {_offdiag._base_address(getattr(fields, name, None)): name
               for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")}
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        kind = getattr(getattr(state, "susceptibility", None), "kind", None)
        if kind not in COVERED_SUSCEPTIBILITY_KINDS:
            reasons.append(f"polarization {index} kind {kind!r} is not covered")
        for component in tuple(_call(state, "driven", default=()) or ()):
            if component not in ELECTRIC_COMPONENTS:
                reasons.append(f"polarization {index} drives {component!r}")
                continue
            for label in ("P", "P_prev"):
                array = (getattr(state, label, {}) or {}).get(component)
                if array is None:
                    reasons.append(f"polarization {index} {label}[{component}] is not allocated")
                    continue
                reasons.extend(_volume_reasons(
                    f"polarization {index} {label}[{component}]", array, shape))
                if label == "P" and _offdiag._base_address(array) in outputs:
                    reasons.append(f"polarization {index} P[{component}] aliases output "
                                   f"{outputs[_offdiag._base_address(array)]}")
        scratch = getattr(state, "_scratch", None)
        if scratch is None:
            reasons.append(f"polarization {index} scratch is not allocated")
        else:
            reasons.extend(_volume_reasons(f"polarization {index} scratch", scratch,
                                            shape))
    for component, states in _poles(fields).items():
        if len(states) > MAX_POLES:
            reasons.append(f"{component} is driven by {len(states)} poles, more than "
                           f"MAX_POLES={MAX_POLES}")
    return reasons


def folded_offdiag_dispersive_coverage(fields: Any, pml: Any,
                                        residency: Any = None,
                                        chunk_bounds: Any = None) -> Coverage:
    """May the folded tensor-row dispersive kernel step ``update_E`` for this pair?

    The folded off-diagonal grid clauses, a fold, a registered susceptibility, a live
    row slot, PML storage, stored E, the rows', the poles', the layout's, the inverse
    epsilon's and the half-integer coefficient tables' clauses, and last, on an
    otherwise admitted run, the per-chunk presence words
    (:func:`.offdiag_update_e.presence_reasons` on the folded codes, through
    ``chunk_bounds``, the record the composer binds; ``None`` derives it from
    ``stepping._chunk_layout``).
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    codes, reasons = _folded._folded_offdiag_grid_reasons(fields, pml, grid)
    reasons = list(reasons)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_susceptibility_reasons(fields))
    if not _symmetry._has_real_fold(grid):
        reasons.append("no mirror plane is active: this is the folded intersection")
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if not states:
        reasons.append("no susceptibility is registered: folded off-diagonal owns update_E")
    if not any(value is not None for value in _offdiag.row_volumes_for(fields)):
        reasons.append("no off-diagonal chi1inv row survived installation")
    if not bool(getattr(fields, "_pml_active", False)):
        reasons.append("Fields is not in PML storage mode")
    if not bool(getattr(fields, "stores_E", False)):
        reasons.append("E is recomputed from D rather than stored")
    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return Coverage(False, tuple(dict.fromkeys(reasons)))
    reasons.extend(_offdiag._row_reasons(fields, shape))
    reasons.extend(_pole_reasons(fields, shape))
    names = tuple(name for name, _source, _axis in E_TERMS)
    names += tuple("f_w_" + name for name, _source, _axis in E_TERMS)
    names += tuple(source for _name, source, _axis in E_TERMS)
    reasons.extend(_layout_reasons(fields, shape, names))
    reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and bool(getattr(pml, "is_active", False)):
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), ("_h",)))
    # (p) The per-chunk presence words, on the codes the kernel bakes, only on an
    #     otherwise admitted run (the clause reads the rows and the division the
    #     clauses above vet).
    if not reasons and codes is not None:
        reasons.extend(_offdiag.presence_reasons(fields, pml, codes, chunk_bounds))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalFoldedOffdiagDispersivePlan:
    """One launch per step, the four scalars, the presence words and the chunk record
    in ``Params``.

    ``chunk_bounds`` is the host record the plan binds, checked against the stored
    shape by :func:`.chunk_bounds.plan_words`, and ``presence`` the six words the rows
    are added with (:func:`.offdiag_update_e.presence_plan_words`); both required, with
    no default. The ``Params`` words are a plan-owned device tensor outside the
    residency registry, as every fused pair's ``Params`` is.
    """

    performs_device_work = True
    replaces_sub_steps = ("update_E",)

    def __init__(self, fields: Any, pml: Any, residency: Residency,
                 row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
                 negate: Sequence[int], poles: Mapping[str, Sequence[Any]],
                 functions: Mapping[str, Any], *, chunk_bounds: Any,
                 presence: Any) -> None:
        self.fields, self.pml, self.residency = fields, pml, residency
        self.shape = tuple(int(value) for value in fields.grid.shape)
        self.n_elem = int(np.prod(self.shape))
        self.params_words = params_words(self.shape, chunk_bounds, presence)
        self.chunk_bounds = _chunk_bounds.plan_words(chunk_bounds, self.shape,
                                                     "MetalFoldedOffdiagDispersivePlan")
        self.presence = _offdiag.presence_plan_words(
            presence, "MetalFoldedOffdiagDispersivePlan")
        self.row_mask, self.codes = tuple(row_mask), tuple(codes)
        self.walls, self.negate = tuple(walls), tuple(negate)
        self._poles = {component: tuple(values) for component, values in poles.items()}
        self.counts = tuple(len(self._poles[name]) for name, _source, _axis in E_TERMS)
        self._packs = {name: np.zeros((MAX_POLES, self.n_elem), dtype=np.float32)
                       for name, _source, _axis in E_TERMS}
        targets = [getattr(fields, name) for name, _source, _axis in E_TERMS]
        auxiliaries = [getattr(fields, "f_w_" + name) for name, _source, _axis in E_TERMS]
        sources = [getattr(fields, source) for _name, source, _axis in E_TERMS]
        inverse = [fields.inverse_epsilon_for(name) for name, _source, _axis in E_TERMS]
        rows = _offdiag.row_volumes_for(fields)
        coefficients = [getattr(pml, f"{stem}_{axis}_h")
                        for axis in "xyz" for stem in ("kps", "kms")]
        _offdiag.MetalOffdiagConstitutivePlan._require_no_aliasing(
            (targets, auxiliaries, sources, inverse, coefficients), rows)
        self._tensors: Dict[str, Any] = {}
        for name, array in zip(("Ex", "Ey", "Ez"), targets):
            self._tensors[name] = residency.mirror(name, array)
        for name, array in zip(("f_w_Ex", "f_w_Ey", "f_w_Ez"), auxiliaries):
            self._tensors[name] = residency.mirror(name, array)
        for name, array in zip(("Dx", "Dy", "Dz"), sources):
            self._tensors[name] = residency.mirror(name, array)
        for name, array in zip(("inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez"), inverse):
            self._tensors[name] = residency.mirror(name, array, constant=True)
        for index, ((row, partner), array) in enumerate(zip(ROW_SLOTS, rows)):
            # Every row slot has a static binding even when its specialization has
            # compiled the term out.  A dead slot must never borrow its partner D
            # volume: D is mutable across the step, while this binding is declared
            # constant, producing a private stale mirror.  The matching diagonal
            # inverse-epsilon volume has the required f32 shape and is immutable;
            # the emitted dead-row specialization cannot read its value.
            placeholder = inverse[index // 2]
            self._tensors[f"row:{row}:{partner}"] = residency.mirror(
                f"row:{row}:{partner}", array if array is not None else placeholder,
                constant=True)
        for (axis, stem), array in zip(((axis, stem) for axis in "xyz" for stem in ("kps", "kms")), coefficients):
            self._tensors[f"{stem}_{axis}"] = residency.mirror(
                f"pml:{stem}_{axis}_h", array, constant=True)
        for component, pack in self._packs.items():
            self._tensors[f"pack:{component}"] = residency.mirror(
                f"folded_offdiag_dispersive:pack:{component}", pack)
        import torch  # noqa: PLC0415

        self._params = torch.from_numpy(self.params_words.copy()).to(
            torch.device(str(getattr(residency, "device", "mps"))))
        self._functions = dict(functions)
        self.launches = self.runs = 0
        self.launches_per_run = 1
        self.volumes = tuple(("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                              "Dx", "Dy", "Dz"))

    def _refresh_packs(self) -> None:
        import torch  # noqa: PLC0415
        for component, pack in self._packs.items():
            states = self._poles[component]
            device_sources = tuple(
                self.residency.tensor_for_host(state.P[component])
                for state in states)
            destination = self._tensors[f"pack:{component}"].reshape(-1)
            if any(source is not None for source in device_sources):
                if any(source is None for source in device_sources):
                    raise RuntimeError(
                        f"{component} pole pack has a mixed host/device producer set; "
                        "a partial refresh would silently combine time levels")
                destination.zero_()
                for lane, source in enumerate(device_sources):
                    destination.narrow(0, lane * self.n_elem, self.n_elem).copy_(
                        source.reshape(-1))
                continue
            pack.fill(0.0)
            for lane, state in enumerate(states):
                pack[lane, :] = np.ascontiguousarray(state.P[component]).reshape(-1)
            destination.copy_(torch.from_numpy(pack.reshape(-1)))

    def run(self, contract: Optional[str] = None) -> None:
        mode = shaders.CONTRACT_OFF if contract is None else contract
        self._refresh_packs()
        function = self._functions[mode]
        tensors = self._tensors
        function(tensors["Ex"], tensors["Ey"], tensors["Ez"],
                 tensors["f_w_Ex"], tensors["f_w_Ey"], tensors["f_w_Ez"],
                 tensors["Dx"], tensors["Dy"], tensors["Dz"],
                 tensors["inv_eps_Ex"], tensors["inv_eps_Ey"], tensors["inv_eps_Ez"],
                 tensors["pack:Ex"], tensors["pack:Ey"], tensors["pack:Ez"],
                 *(tensors[f"row:{row}:{partner}"] for row, partner in ROW_SLOTS),
                 *(tensors[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")),
                 self._params)
        self.launches += 1
        self.runs += 1


def plan_folded_offdiag_dispersive(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        *, chunk_bounds: Any,
        ) -> Optional[MetalFoldedOffdiagDispersivePlan]:
    """Build the folded tensor-row dispersive plan, or return ``None``.

    None means REFUSED, and the reasons come from
    :func:`folded_offdiag_dispersive_coverage`. ``chunk_bounds`` is the record of the
    division the run steps with (the composer's ``StepContext.chunk_bounds``) and has
    no default; the presence words are built against it
    (:func:`.offdiag_update_e.presence_for`, on the folded codes).
    """
    if not folded_offdiag_dispersive_coverage(fields, pml, residency,
                                              chunk_bounds).covered:
        return None
    assert residency is not None and pml is not None
    codes, _reasons = _symmetry.folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - coverage already refuses
        return None
    presence = _offdiag.presence_for(fields, pml, codes, chunk_bounds)
    weights = _folded.mirror_ghost_weights(fields.grid)
    walls = _offdiag.wall_mask_axes(fields.grid)
    negate = _folded.negated_axes(codes, weights)
    poles = _poles(fields)
    counts = tuple(len(poles[name]) for name, _source, _axis in E_TERMS)
    selected = dict(functions or {})
    for mode in contract_variants:
        selected.setdefault(mode, compile_folded_offdiag_dispersive(
            tuple(value is not None for value in _offdiag.row_volumes_for(fields)),
            codes, walls, negate, counts, mode))
    return MetalFoldedOffdiagDispersivePlan(
        fields, pml, residency,
        tuple(value is not None for value in _offdiag.row_volumes_for(fields)),
        codes, walls, negate, poles, selected, chunk_bounds=chunk_bounds,
        presence=presence)


def _has_fold_rows_poles(context: Any) -> bool:
    try:
        fields = context.fields
        return bool(_symmetry._has_real_fold(fields.grid)
                    and tuple(getattr(fields, "polarizations", ()) or ())
                    and any(value is not None for value in _offdiag.row_volumes_for(fields)))
    except Exception:  # noqa: BLE001
        return False


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return (folded_offdiag_dispersive_coverage(context.fields, context.pml,
                                                context.residency,
                                                getattr(context, "chunk_bounds", None))
            if slot == SLOT else Coverage(False, (f"folded tensor dispersion cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalFoldedOffdiagDispersivePlan]:
    return (plan_folded_offdiag_dispersive(context.fields, context.pml,
                                            context.residency,
                                            context.contract_variants,
                                            chunk_bounds=context.chunk_bounds)
            if slot == SLOT else None)


def register_arms() -> Tuple[Any, ...]:
    # Wired: the row is the array path's double row (ruling R7, software binary64).
    return (arms.register(
        FAMILY, SLOT, LABEL, _arm_coverage, _arm_plan,
        prefix=f"{LABEL}: ", noun="folded tensor dispersive stored-E constitutive",
        gate=_has_fold_rows_poles, wired=True),)


ARMS = register_arms()
