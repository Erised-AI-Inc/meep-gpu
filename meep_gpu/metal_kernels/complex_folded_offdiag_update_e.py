"""Complex folded tensor-row ``update_E`` with the active-PML stored-E tail.

This is distinct from the no-PML complex tensor shader: it stores the row product
in ``f_w_E`` before applying the two ordered PML accumulations.  The six complex
wrap phases are packed into one immutable ``float2`` buffer, keeping the signature
at 29 bindings while preserving the phase direction and operand order at each
partner-DOWN and own-UP shift.  Mirror parity is a coefficient-left complex
multiplication on the partner DOWN ghost only.

The native tensor gate and the experimental composer separately cover its folded
complex curl/fill/tensor-row residency composition. This does not enable production
driver dispatch.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import Coverage, _boundary_kinds, _coefficient_reasons, _inverse_epsilon_reasons
from . import shaders, templates
from . import complex_fields as _complex
from . import folded_complex as _folded_complex
from . import folded_offdiag_update_e as _folded_real
from . import offdiag_update_e as _offdiag
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .plans import KernelPlan

FAMILY = "complex_folded_offdiag_update_e"
SLOT = "update_E"
LABEL = "complex folded off-diagonal PML E"
BINDING_COUNT = 29
E_TERMS = _offdiag.E_TERMS
ROW_SLOTS = _offdiag.ROW_SLOTS

if BINDING_COUNT > MAX_BUFFER_BINDINGS:  # pragma: no cover - platform invariant
    raise RuntimeError("complex folded tensor-row signature exceeds Metal's binding ceiling")

__all__ = [
    "BINDING_COUNT", "FAMILY", "LABEL", "MetalComplexFoldedOffdiagPlan", "SLOT",
    "complex_folded_offdiag_source", "compile_complex_folded_offdiag",
    "metal_complex_folded_offdiag_coverage", "plan_metal_complex_folded_offdiag",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
__HELPERS__
kernel void complex_folded_offdiag_update_e(
    device float2* f0 [[buffer(0)]], device float2* f1 [[buffer(1)]],
    device float2* f2 [[buffer(2)]], device float2* w0 [[buffer(3)]],
    device float2* w1 [[buffer(4)]], device float2* w2 [[buffer(5)]],
    device const float2* g0 [[buffer(6)]], device const float2* g1 [[buffer(7)]],
    device const float2* g2 [[buffer(8)]], device const float* e0 [[buffer(9)]],
    device const float* e1 [[buffer(10)]], device const float* e2 [[buffer(11)]],
    device const float* u01 [[buffer(12)]], device const float* u02 [[buffer(13)]],
    device const float* u11 [[buffer(14)]], device const float* u12 [[buffer(15)]],
    device const float* u21 [[buffer(16)]], device const float* u22 [[buffer(17)]],
    device const float* kp0 [[buffer(18)]], device const float* km0 [[buffer(19)]],
    device const float* kp1 [[buffer(20)]], device const float* km1 [[buffer(21)]],
    device const float* kp2 [[buffer(22)]], device const float* km2 [[buffer(23)]],
    device const float2* ph [[buffer(24)]], constant uint& nx [[buffer(25)]],
    constant uint& ny [[buffer(26)]], constant uint& nz [[buffer(27)]],
    constant uint& n_elem [[buffer(28)]], uint idx [[thread_position_in_grid]])
{
__GUARD__
__DECODE__
    int nxi = int(nx), nyz = nyi * nzi;
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
    float2 prev0 = w0[ii];
__SRC0__
    w0[ii] = src0;
    float2 a0 = f0[ii]; a0 = a0 + c_mul_coefficient_left(kp_0, src0);
    a0 = a0 - c_mul_coefficient_left(km_0, prev0); f0[ii] = a0;
    float2 prev1 = w1[ii];
__SRC1__
    w1[ii] = src1;
    float2 a1 = f1[ii]; a1 = a1 + c_mul_coefficient_left(kp_1, src1);
    a1 = a1 - c_mul_coefficient_left(km_1, prev1); f1[ii] = a1;
    float2 prev2 = w2[ii];
__SRC2__
    w2[ii] = src2;
    float2 a2 = f2[ii]; a2 = a2 + c_mul_coefficient_left(kp_2, src2);
    a2 = a2 - c_mul_coefficient_left(km_2, prev2); f2[ii] = a2;
}
"""

_STRIDE = {0: "{} * nyz", 1: "{} * nzi", 2: "{}"}
_HOME = ("i", "j", "k")
_ROW_BUFFERS = ("u01", "u02", "u11", "u12", "u21", "u22")


def _index(shifts: Mapping[int, str]) -> str:
    return " + ".join(_STRIDE[axis].format(shifts.get(axis, _HOME[axis]))
                       for axis in range(3))


def _phase_down(axis: int, phased: Sequence[int], value: str) -> str:
    if not phased[axis]:
        return ""
    coordinate = ("i", "j", "k")[axis]
    return f"    {value} = ({coordinate} == 0) ? c_mul({value}, ph[{axis}]) : {value};"


def _phase_up(axis: int, phased: Sequence[int], value: str) -> str:
    extent = ("nxi", "nyi", "nzi")[axis]
    coordinate = ("i", "j", "k")[axis]
    return (f"    {value} = ({coordinate} == {extent} - 1) ? c_mul({value}, ph[{axis + 3}]) : {value};"
            if phased[axis] else "")


def _term(component: int, partner_axis: int, coefficient: str, tag: str,
          phased: Sequence[int], negate: Sequence[int]) -> str:
    own_axis = E_TERMS[component][2]
    down, _up, down_valid, _up_valid, _extent = _offdiag._TWO_WAY_NAMES["xyz"[partner_axis]]
    _down, own_up, _dv, own_valid, _extent = _offdiag._TWO_WAY_NAMES["xyz"[own_axis]]
    partner = f"g{partner_axis}"
    down_index = _index({partner_axis: down})
    up_index = _index({own_axis: own_up})
    corner_index = _index({own_axis: own_up, partner_axis: down})
    at = ("at_x", "at_y", "at_z")[partner_axis]
    parity_down = (f"    down_{tag} = ({at} ? c_mul_coefficient_left(-1.0f, down_{tag}) : down_{tag});"
                   if negate[partner_axis] else "")
    parity_corner = (f"    corner_{tag} = ({at} ? c_mul_coefficient_left(-1.0f, corner_{tag}) : corner_{tag});"
                     if negate[partner_axis] else "")
    lines = [
        f"    float2 down_{tag} = {down_valid} ? {partner}[{down_index}] : float2(0.0f, 0.0f);",
        _phase_down(partner_axis, phased, f"down_{tag}"), parity_down,
        f"    float2 near_{tag} = {partner}[ii] + down_{tag};",
        f"    float2 near_product_{tag} = c_mul_field_left(near_{tag}, {coefficient}[ii]);",
        f"    float2 far_a_{tag} = {own_valid} ? {partner}[{up_index}] : float2(0.0f, 0.0f);",
        f"    float2 corner_{tag} = ({own_valid} && {down_valid}) ? {partner}[{corner_index}] : float2(0.0f, 0.0f);",
        _phase_down(partner_axis, phased, f"corner_{tag}"), parity_corner,
        f"    float2 far_pair_{tag} = far_a_{tag} + corner_{tag};",
        f"    float2 far_product_{tag} = {own_valid} ? c_mul_field_left(far_pair_{tag}, {coefficient}[{up_index}]) : float2(0.0f, 0.0f);",
        _phase_up(own_axis, phased, f"far_product_{tag}"),
        f"    float2 term_{tag} = c_mul_coefficient_left(0.25f, near_product_{tag} + far_product_{tag});",
    ]
    return "\n".join(line for line in lines if line)


def _component_source(component: int, row_mask: Sequence[int], walls: Sequence[int],
                      phased: Sequence[int], negate: Sequence[int]) -> str:
    diagonal = f"c_mul_field_left(g{component}[ii], e{component}[ii])"
    active = [offset for offset in (0, 1) if row_mask[2 * component + offset]]
    if not active:
        return f"    float2 src{component} = {diagonal};"
    lines: List[str] = []
    terms: List[str] = []
    for offset in active:
        tag = f"{component}{offset}"
        lines.append(_term(component, _offdiag.TRANSVERSE_PARTNERS[component][offset],
                           _ROW_BUFFERS[2 * component + offset], tag, phased, negate))
        terms.append(tag)
    lines.append(f"    float2 total{component} = term_{terms[0]};")
    for tag in terms[1:]:
        lines.append(f"    total{component} = total{component} + term_{tag};")
    for axis in _offdiag.WALL_MASK_AXES[component]:
        if walls[axis]:
            lines.append(f"    total{component} = at_{'xyz'[axis]} ? float2(0.0f, 0.0f) : total{component};")
    lines.append(f"    float2 src{component} = {diagonal} + total{component};")
    return "\n".join(lines)


def complex_folded_offdiag_source(row_mask: Sequence[int], codes: Sequence[int],
                                  walls: Sequence[int], phased: Sequence[int],
                                  negate: Sequence[int], expansion: str,
                                  contract: str = shaders.CONTRACT_OFF) -> str:
    row_mask, codes = tuple(map(int, row_mask)), tuple(map(int, codes))
    walls, phased, negate = tuple(map(int, walls)), tuple(map(int, phased)), tuple(map(int, negate))
    if len(row_mask) != 6 or not any(row_mask):
        raise ValueError("row_mask must carry at least one of the six tensor slots")
    if any(value not in (0, 1) for value in (*row_mask, *walls, *phased, *negate)):
        raise ValueError("row masks, walls, phases, and parity flags must be 0 or 1")
    if len(codes) != 3 or any(code not in (0, 1, 2, 3) for code in codes):
        raise ValueError("codes must be three periodic/metallic/mirror values")
    if any(flag and code != templates.PERIODIC for flag, code in zip(phased, codes)):
        raise ValueError("a Bloch phase requires an unfolded periodic axis")
    if any(flag and code not in _folded_real.MIRROR_CODES for flag, code in zip(negate, codes)):
        raise ValueError("a mirror-parity flag requires a mirror boundary")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion), "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__GHOST_X__": _folded_real._folded_two_way_ghost("x", codes[0]),
        "__GHOST_Y__": _folded_real._folded_two_way_ghost("y", codes[1]),
        "__GHOST_Z__": _folded_real._folded_two_way_ghost("z", codes[2]),
        "__SRC0__": _component_source(0, row_mask, walls, phased, negate),
        "__SRC1__": _component_source(1, row_mask, walls, phased, negate),
        "__SRC2__": _component_source(2, row_mask, walls, phased, negate),
    })


def compile_complex_folded_offdiag(row_mask: Sequence[int], codes: Sequence[int],
                                   walls: Sequence[int], phased: Sequence[int],
                                   negate: Sequence[int], expansion: str,
                                   contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(complex_folded_offdiag_source(
        row_mask, codes, walls, phased, negate, expansion, contract)).complex_folded_offdiag_update_e


def metal_complex_folded_offdiag_coverage(
        fields: Any, pml: Any, residency: Any = None, probe: Any = None) -> Coverage:
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons, codes = _folded_complex._folded_complex_grid_reasons(
        fields, pml, grid, residency, probe)
    reasons = list(reasons)
    reasons.extend(_folded_complex._requires_a_fold(
        grid, codes, "the complex folded tensor row is an intersection product"))
    shape = tuple(getattr(grid, "shape", ()))
    rows = _offdiag.row_volumes_for(fields)
    if not any(value is not None for value in rows):
        reasons.append("no live off-diagonal chi1inv row is installed")
    reasons.extend(_offdiag._row_reasons(fields, shape))
    names = (tuple(name for name, _source, _axis in E_TERMS)
             + tuple("f_w_" + name for name, _source, _axis in E_TERMS)
             + tuple(source for _name, source, _axis in E_TERMS))
    for name in names:
        value = getattr(fields, name, None)
        if value is None:
            reasons.append(f"{name} is not allocated")
        else:
            reasons.extend(_complex._complex_volume_reasons(name, value, shape))
    if len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
        if pml is not None and bool(getattr(pml, "is_active", False)):
            reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), ("_h",)))
    reasons.extend(_folded_complex._stored_extent_reasons(codes, shape))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalComplexFoldedOffdiagPlan(KernelPlan):
    __slots__ = ("shape", "n_elem", "row_mask", "codes", "walls", "phased", "negate",
                 "expansion", "residency", "volumes")
    REPR_FIELDS = ("shape", "row_mask", "codes", "walls", "phased", "negate", "expansion")

    def __init__(self, shape: Sequence[int], row_mask: Sequence[int], codes: Sequence[int],
                 walls: Sequence[int], phased: Sequence[int], negate: Sequence[int],
                 expansion: str, residency: Residency, targets: Sequence[Any],
                 auxiliaries: Sequence[Any], sources: Sequence[Any], inverse: Sequence[Any],
                 rows: Sequence[Any], coefficients: Sequence[Any], phase_pack: Any,
                 functions: Mapping[str, Any], volumes: Sequence[str]) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = int(np.prod(self.shape))
        self.row_mask = tuple(int(value) for value in row_mask)
        self.codes, self.walls, self.phased, self.negate = (
            tuple(int(value) for value in values)
            for values in (codes, walls, phased, negate))
        self.expansion, self.residency = str(expansion), residency
        self.volumes = tuple(dict.fromkeys(volumes))
        bound_rows = tuple(inverse[index // 2] if value is None else value
                           for index, value in enumerate(rows))
        super().__init__(dict(functions), tuple(targets) + tuple(auxiliaries)
                         + tuple(sources) + tuple(inverse) + bound_rows
                         + tuple(coefficients) + (phase_pack, *self.shape, self.n_elem))


def _functions(row_mask, codes, walls, phased, negate, expansion, contract_variants):
    return {mode: compile_complex_folded_offdiag(
        row_mask, codes, walls, phased, negate, expansion, mode)
            for mode in contract_variants}


def plan_metal_complex_folded_offdiag(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,), probe: Any = None,
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalComplexFoldedOffdiagPlan]:
    if not metal_complex_folded_offdiag_coverage(fields, pml, residency, probe).covered:
        return None
    assert residency is not None and pml is not None
    record = probe if probe is not None else _folded_complex.load_expansion_probe()
    expansion = _folded_complex.expansion_from_probe(record)
    if expansion is None:  # predicate already named this refusal
        return None
    codes, _reasons = _folded_complex.folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - coverage already refuses
        return None
    kinds = _boundary_kinds(fields.grid, pml)
    phases = _complex.bloch_phase_table(fields.grid, kinds)
    phased_down, down = _complex.phase_arguments(phases, backward=True)
    phased_up, up = _complex.phase_arguments(phases, backward=False)
    if phased_down != phased_up:
        raise ValueError("forward and backward phase flags disagree")
    phase_pack = np.asarray([complex(real, imag) for real, imag in (*down, *up)],
                            dtype=np.complex64)
    rows = _offdiag.row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)
    walls = _offdiag.wall_mask_axes(fields.grid)
    negate = _folded_real.negated_axes(codes, _folded_real.mirror_ghost_weights(fields.grid))
    target_names = tuple(name for name, _source, _axis in E_TERMS)
    source_names = tuple(source for _name, source, _axis in E_TERMS)
    targets = [_complex._complex_mirror(residency, name, getattr(fields, name))
               for name in target_names]
    auxiliaries = [_complex._complex_mirror(residency, "f_w_" + name,
                                             getattr(fields, "f_w_" + name))
                   for name in target_names]
    sources = [_complex._complex_mirror(residency, name, getattr(fields, name))
               for name in source_names]
    inverse = [residency.mirror(f"inv_eps_{name}:complex_folded_offdiag",
                                fields.inverse_epsilon_for(name), constant=True)
               for name in target_names]
    mirrored_rows = [None if value is None else residency.mirror(
        f"chi1inv_offdiag:{row}:{partner}:complex_folded", value, constant=True)
        for (row, partner), value in zip(ROW_SLOTS, rows)]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}_h:complex_folded",
                                     getattr(pml, f"{stem}_{axis}_h"), constant=True)
                    for axis in "xyz" for stem in ("kps", "kms")]
    mirrored_phases = residency.mirror("complex_folded_offdiag:phases", phase_pack,
                                       constant=True, dtype=np.complex64)
    selected = dict(functions or _functions(row_mask, codes, walls, phased_down,
                                              negate, expansion, contract_variants))
    return MetalComplexFoldedOffdiagPlan(
        fields.grid.shape, row_mask, codes, walls, phased_down, negate, expansion,
        residency, targets, auxiliaries, sources, inverse, mirrored_rows, coefficients,
        mirrored_phases, selected,
        target_names + tuple("f_w_" + name for name in target_names) + source_names)


def _arm_gate(context: Any) -> bool:
    try:
        return bool(_folded_complex._has_real_fold(context.fields.grid)
                    and getattr(context.pml, "is_active", False)
                    and any(value is not None for value in _offdiag.row_volumes_for(context.fields)))
    except Exception:  # noqa: BLE001
        return False


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return (metal_complex_folded_offdiag_coverage(
        context.fields, context.pml, context.residency, context.extra.get("probe"))
            if slot == SLOT else Coverage(False, (f"complex folded tensor row cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexFoldedOffdiagPlan]:
    return (plan_metal_complex_folded_offdiag(
        context.fields, context.pml, context.residency, context.contract_variants,
        context.extra.get("probe")) if slot == SLOT else None)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return (arms.register(
        FAMILY, SLOT, LABEL, _arm_coverage, _arm_plan, gate=_arm_gate,
        prefix=f"{LABEL}: ", noun="complex folded tensor PML constitutive",
        wired=True),)


ARMS = register_arms()
