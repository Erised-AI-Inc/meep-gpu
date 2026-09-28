"""Metal ``update_E`` for the fold + tensor-row + dispersion intersection.

The tensor row gathers each partner component at four Yee addresses.  With a
susceptibility, *every one* of those gathers is ``D - P0 - P1 - …`` in registered
order; composing the folded tensor shader with the pointwise dispersive shader
would therefore be wrong.  Metal exposes at most 31 bindings, while three groups
of eight live pole volumes would exceed that limit.  The plan keeps three persistent
``MAX_POLES × n_elem`` pole packs, one per D component, and the fused shader reads
each pack by lane.  This preserves the ordered arithmetic and keeps the signature
at the platform limit (31) without materialising ``D - sum(P)`` volumes.

The pack refresh has two modes.  The standalone gate has no ADE plan, so it stages
the initial NumPy pole arrays.  In a composed step, the ADE plan has already
registered and then rotated the physical P mirrors; each pack lane copies directly
from that live MPS mirror, without pulling stale P data through NumPy.  The latter
is separately whole-step gated; neither mode enables production driver dispatch.
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
BINDING_COUNT = 31

if BINDING_COUNT != MAX_BUFFER_BINDINGS:  # pragma: no cover - platform invariant
    raise RuntimeError("the packed-pole signature must consume the measured limit")

__all__ = [
    "BINDING_COUNT",
    "FAMILY",
    "LABEL",
    "MAX_POLES",
    "MetalFoldedOffdiagDispersivePlan",
    "folded_offdiag_dispersive_coverage",
    "folded_offdiag_dispersive_source",
    "plan_folded_offdiag_dispersive",
    "register_arms",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
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
    device const float* km2 [[buffer(26)]], constant uint& nx [[buffer(27)]],
    constant uint& ny [[buffer(28)]], constant uint& nz [[buffer(29)]],
    constant uint& n_elem [[buffer(30)]], uint idx [[thread_position_in_grid]])
{
__GUARD__
__DECODE__
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
    float prev0 = w0[ii];
__SRC0__
    w0[ii] = src0;
    float a0 = f0[ii]; a0 = a0 + kp_0 * src0; a0 = a0 - km_0 * prev0; f0[ii] = a0;
    float prev1 = w1[ii];
__SRC1__
    w1[ii] = src1;
    float a1 = f1[ii]; a1 = a1 + kp_1 * src1; a1 = a1 - km_1 * prev1; f1[ii] = a1;
    float prev2 = w2[ii];
__SRC2__
    w2[ii] = src2;
    float a2 = f2[ii]; a2 = a2 + kp_2 * src2; a2 = a2 - km_2 * prev2; f2[ii] = a2;
}
"""


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
        f"    float term_{tag} = 0.25f * ((near_{tag} * unear_{tag}) "
        f"+ (far_{tag} * ufar_{tag}));"))


def _component_source(component: int, row_mask: Sequence[int],
                      walls: Sequence[int], negate: Sequence[int]) -> str:
    gs = f"dmp{component}(g{component}, p{component}, ii, n_elem)"
    us = f"e{component}[ii]"
    live = [offset for offset in (0, 1) if row_mask[2 * component + offset]]
    if not live:
        return f"    float src{component} = {gs} * {us};"
    lines = []
    tags = []
    for offset in live:
        tag = f"{component}{offset}"
        lines.append(_term(component, _offdiag.TRANSVERSE_PARTNERS[component][offset],
                           _offdiag._ROW_BUFFERS[2 * component + offset], tag,
                           negate))
        tags.append(tag)
    lines.append(f"    float total{component} = term_{tags[0]};")
    for tag in tags[1:]:
        lines.append(f"    total{component} = total{component} + term_{tag};")
    lines.append(_offdiag._wall_mask(component, walls))
    lines.append(f"    float src{component} = ({gs} * {us}) + total{component};")
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
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__POLE_HELPERS__": _pole_helpers(counts), "__GUARD__": templates.GUARD,
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
                                        residency: Any = None) -> Coverage:
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    _codes, reasons = _folded._folded_offdiag_grid_reasons(fields, pml, grid)
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
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalFoldedOffdiagDispersivePlan:
    performs_device_work = True
    replaces_sub_steps = ("update_E",)

    def __init__(self, fields: Any, pml: Any, residency: Residency,
                 row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
                 negate: Sequence[int], poles: Mapping[str, Sequence[Any]],
                 functions: Mapping[str, Any]) -> None:
        self.fields, self.pml, self.residency = fields, pml, residency
        self.shape = tuple(int(value) for value in fields.grid.shape)
        self.n_elem = int(np.prod(self.shape))
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
                 *self.shape, self.n_elem)
        self.launches += 1
        self.runs += 1


def plan_folded_offdiag_dispersive(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalFoldedOffdiagDispersivePlan]:
    if not folded_offdiag_dispersive_coverage(fields, pml, residency).covered:
        return None
    assert residency is not None and pml is not None
    codes, _reasons = _symmetry.folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - coverage already refuses
        return None
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
        codes, walls, negate, poles, selected)


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
                                                context.residency)
            if slot == SLOT else Coverage(False, (f"folded tensor dispersion cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalFoldedOffdiagDispersivePlan]:
    return (plan_folded_offdiag_dispersive(context.fields, context.pml,
                                            context.residency,
                                            context.contract_variants)
            if slot == SLOT else None)


def register_arms() -> Tuple[Any, ...]:
    return (arms.register(
        FAMILY, SLOT, LABEL, _arm_coverage, _arm_plan,
        prefix=f"{LABEL}: ", noun="folded tensor dispersive stored-E constitutive",
        gate=_has_fold_rows_poles, wired=True),)


ARMS = register_arms()
