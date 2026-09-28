"""Complex/Bloch no-PML tensor-row ``update_E`` on Metal.

This is the Metal counterpart of Triton's complex no-PML off-diagonal family.
It owns the non-elementwise electric constitutive step when complex64 storage has
live real off-diagonal inverse-permittivity rows and the absorber is inactive.
The two shifts in every tensor term are deliberately separate: partner-axis DOWN
is phased at its wrapped field lane, while the product shifted UP the row's own
axis is phased after multiplication by the real tensor coefficient.

The arm is selected only by the experimental Metal composer.  Its direct native
gate establishes the tensor arithmetic; the whole-step gate establishes the
no-PML complex curl/null/tensor-row residency composition.  This is still not
production driver dispatch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import Coverage, _boundary_kinds, _inverse_epsilon_reasons
from . import shaders, templates
from .complex_fields import (
    _complex_mirror,
    _complex_volume_reasons,
    bloch_phase_table,
    expansion_from_probe,
    load_expansion_probe,
    phase_arguments,
)
from .complex_no_pml_conductive import _base_reasons
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .offdiag_update_e import (
    E_TERMS,
    ROW_SLOTS,
    TRANSVERSE_PARTNERS,
    WALL_MASK_AXES,
    _base_address,
    _two_way_ghost,
    row_volumes_for,
    wall_mask_axes,
)
from .plans import KernelPlan

FAMILY = "complex_no_pml_offdiag_update_e"
SLOT = "update_E"
BINDING_COUNT = 25

if BINDING_COUNT > MAX_BUFFER_BINDINGS:  # pragma: no cover - platform invariant
    raise RuntimeError("complex no-PML tensor row exceeds Metal's binding ceiling")

__all__ = [
    "BINDING_COUNT", "FAMILY", "SLOT", "MetalComplexNoPmlOffdiagPlan",
    "complex_no_pml_offdiag_source", "compile_complex_no_pml_offdiag",
    "metal_complex_no_pml_offdiag_coverage", "plan_metal_complex_no_pml_offdiag",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__
__HELPERS__

kernel void complex_no_pml_offdiag_update_e(
    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device const float2* g0      [[buffer(3)]],
    device const float2* g1      [[buffer(4)]],
    device const float2* g2      [[buffer(5)]],
    device const float*  e0      [[buffer(6)]],
    device const float*  e1      [[buffer(7)]],
    device const float*  e2      [[buffer(8)]],
    device const float*  u01     [[buffer(9)]],
    device const float*  u02     [[buffer(10)]],
    device const float*  u11     [[buffer(11)]],
    device const float*  u12     [[buffer(12)]],
    device const float*  u21     [[buffer(13)]],
    device const float*  u22     [[buffer(14)]],
    constant uint&       nx      [[buffer(15)]],
    constant uint&       ny      [[buffer(16)]],
    constant uint&       nz      [[buffer(17)]],
    constant uint&       n_elem  [[buffer(18)]],
    constant float2&     dpx     [[buffer(19)]],
    constant float2&     dpy     [[buffer(20)]],
    constant float2&     dpz     [[buffer(21)]],
    constant float2&     upx     [[buffer(22)]],
    constant float2&     upy     [[buffer(23)]],
    constant float2&     upz     [[buffer(24)]],
    uint idx [[thread_position_in_grid]])
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
__SRC0__
__SRC1__
__SRC2__
    f0[ii] = src0;
    f1[ii] = src1;
    f2[ii] = src2;
}
"""

_STRIDE = {0: "{} * nyz", 1: "{} * nzi", 2: "{}"}
_HOME = ("i", "j", "k")
_NAMES = {
    0: ("di", "ui", "dvx", "uvx", "i", "nxi", "dpx", "upx"),
    1: ("dj", "uj", "dvy", "uvy", "j", "nyi", "dpy", "upy"),
    2: ("dk", "uk", "dvz", "uvz", "k", "nzi", "dpz", "upz"),
}
_ROW_BUFFERS = ("u01", "u02", "u11", "u12", "u21", "u22")


def _index(shifts: Mapping[int, str]) -> str:
    return " + ".join(_STRIDE[axis].format(shifts.get(axis, _HOME[axis]))
                       for axis in range(3))


def _phase_down(axis: int, phased: bool, value: str) -> str:
    if not phased:
        return ""
    _down, _up, _dv, _uv, coordinate, _extent, phase, _up_phase = _NAMES[axis]
    return f"    {value} = ({coordinate} == 0) ? c_mul({value}, {phase}) : {value};"


def _phase_up(axis: int, phased: bool, value: str) -> str:
    if not phased:
        return ""
    _down, _up, _dv, _uv, coordinate, extent, _phase, up_phase = _NAMES[axis]
    return f"    {value} = ({coordinate} == {extent} - 1) ? c_mul({value}, {up_phase}) : {value};"


def _term(component: int, partner_axis: int, coefficient: str, tag: str,
          phased: Sequence[int]) -> str:
    """One ordered complex row product from ``stepping._offdiagonal_terms``."""
    own_axis = E_TERMS[component][2]
    down, _up, dvalid, _uvalid, _coordinate, _extent, _phase, _up_phase = _NAMES[partner_axis]
    _down, own_up, _dvalid, own_valid, _coord, _extent, _phase, _up_phase = _NAMES[own_axis]
    partner = f"g{partner_axis}"
    down_index = _index({partner_axis: down})
    up_index = _index({own_axis: own_up})
    corner_index = _index({own_axis: own_up, partner_axis: down})
    lines = [
        f"    float2 down_{tag} = {dvalid} ? {partner}[{down_index}] : float2(0.0f, 0.0f);",
        _phase_down(partner_axis, bool(phased[partner_axis]), f"down_{tag}"),
        f"    float2 near_{tag} = {partner}[ii] + down_{tag};",
        f"    float2 near_product_{tag} = c_mul_field_left(near_{tag}, {coefficient}[ii]);",
        f"    float2 far_a_{tag} = {own_valid} ? {partner}[{up_index}] : float2(0.0f, 0.0f);",
        f"    float2 far_b_{tag} = ({own_valid} && {dvalid}) ? {partner}[{corner_index}] : float2(0.0f, 0.0f);",
        _phase_down(partner_axis, bool(phased[partner_axis]), f"far_b_{tag}"),
        f"    float2 far_pair_{tag} = far_a_{tag} + far_b_{tag};",
        f"    float2 far_product_{tag} = {own_valid} ? c_mul_field_left(far_pair_{tag}, {coefficient}[{up_index}]) : float2(0.0f, 0.0f);",
        _phase_up(own_axis, bool(phased[own_axis]), f"far_product_{tag}"),
        f"    float2 term_{tag} = c_mul_coefficient_left(0.25f, near_product_{tag} + far_product_{tag});",
    ]
    return "\n".join(line for line in lines if line)


def _component_source(component: int, row_mask: Sequence[int], walls: Sequence[int],
                      phased: Sequence[int]) -> str:
    diagonal = f"c_mul_field_left(g{component}[ii], e{component}[ii])"
    active = [offset for offset in (0, 1) if row_mask[2 * component + offset]]
    if not active:
        return f"    float2 src{component} = {diagonal};"
    lines = []
    terms = []
    for offset in active:
        tag = f"{component}{offset}"
        lines.append(_term(component, TRANSVERSE_PARTNERS[component][offset],
                           _ROW_BUFFERS[2 * component + offset], tag, phased))
        terms.append(tag)
    lines.append(f"    float2 total{component} = term_{terms[0]};")
    for tag in terms[1:]:
        lines.append(f"    total{component} = total{component} + term_{tag};")
    for axis in WALL_MASK_AXES[component]:
        if walls[axis]:
            lines.append(f"    total{component} = at_{'xyz'[axis]} ? float2(0.0f, 0.0f) : total{component};")
    lines.append(f"    float2 src{component} = {diagonal} + total{component};")
    return "\n".join(lines)


def complex_no_pml_offdiag_source(row_mask: Sequence[int], codes: Sequence[int],
                                  walls: Sequence[int], phased: Sequence[int],
                                  expansion: str,
                                  contract: str = shaders.CONTRACT_OFF) -> str:
    """Emit a specialized complex no-PML tensor constitutive shader."""
    row_mask = tuple(int(value) for value in row_mask)
    codes = tuple(int(value) for value in codes)
    walls = tuple(int(value) for value in walls)
    phased = tuple(int(value) for value in phased)
    if len(row_mask) != len(ROW_SLOTS):
        raise ValueError("row_mask must carry six tensor-row flags")
    if not any(row_mask):
        raise ValueError("all-dead tensor rows belong to the elementwise family")
    if any(value not in (0, 1) for value in (*row_mask, *walls, *phased)):
        raise ValueError("row masks, walls, and phase flags must be 0 or 1")
    if len(codes) != 3 or any(value not in (templates.PERIODIC, templates.METALLIC)
                              for value in codes):
        raise ValueError("codes must be three periodic/metallic values")
    if any(flag and code != templates.PERIODIC for flag, code in zip(phased, codes)):
        raise ValueError("a Bloch phase requires a periodic boundary")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__GHOST_X__": _two_way_ghost("x", codes[0]),
        "__GHOST_Y__": _two_way_ghost("y", codes[1]),
        "__GHOST_Z__": _two_way_ghost("z", codes[2]),
        "__SRC0__": _component_source(0, row_mask, walls, phased),
        "__SRC1__": _component_source(1, row_mask, walls, phased),
        "__SRC2__": _component_source(2, row_mask, walls, phased),
    })


def compile_complex_no_pml_offdiag(row_mask: Sequence[int], codes: Sequence[int],
                                   walls: Sequence[int], phased: Sequence[int],
                                   expansion: str,
                                   contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(complex_no_pml_offdiag_source(
        row_mask, codes, walls, phased, expansion, contract)).complex_no_pml_offdiag_update_e


def _offdiag_reasons(fields: Any, shape: Sequence[int]) -> List[str]:
    reasons: List[str] = []
    rows = row_volumes_for(fields)
    if not any(value is not None for value in rows):
        reasons.append("no live off-diagonal chi1inv row is installed")
    outputs = list(getattr(fields, name, None) for name, _source, _axis in E_TERMS)
    sources = list(getattr(fields, source, None) for _name, source, _axis in E_TERMS)
    written = {address for value in (*outputs, *sources)
               if (address := _base_address(value)) is not None}
    for (row, partner), value in zip(ROW_SLOTS, rows):
        if value is None:
            continue
        label = f"chi1inv_offdiag:{row}:{partner}"
        if str(getattr(value, "dtype", None)) != "float32":
            reasons.append(f"{label} dtype {getattr(value, 'dtype', None)} is not float32")
        if tuple(getattr(value, "shape", ())) != tuple(shape):
            reasons.append(f"{label} shape {tuple(getattr(value, 'shape', ()))!r} != grid shape {tuple(shape)!r}")
        if not bool(getattr(getattr(value, "flags", None), "c_contiguous", False)):
            reasons.append(f"{label} is not C-contiguous")
        address = _base_address(value)
        if address is not None and address in written:
            reasons.append(f"{label} aliases a written E or D volume")
    return reasons


def metal_complex_no_pml_offdiag_coverage(
        fields: Any, pml: Any, residency: Any = None, probe: Any = None) -> Coverage:
    """Conservative predicate for the complex no-absorber tensor row."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons = list(_base_reasons(fields, pml, grid, residency, probe))
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if states:
        reasons.append("a polarization is registered; this tensor row reads D directly")
    shape = tuple(getattr(grid, "shape", ()))
    names = tuple(name for name, _source, _axis in E_TERMS) + tuple(
        source for _name, source, _axis in E_TERMS)
    for name in names:
        value = getattr(fields, name, None)
        if value is None:
            reasons.append(f"{name} is not allocated")
        else:
            reasons.extend(_complex_volume_reasons(name, value, shape))
    if len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    reasons.extend(_offdiag_reasons(fields, shape))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalComplexNoPmlOffdiagPlan(KernelPlan):
    """Persistent MPS mirrors for one specialized complex tensor-row update."""

    __slots__ = ("shape", "n_elem", "row_mask", "codes", "walls", "phased",
                 "expansion", "residency", "volumes")
    REPR_FIELDS = ("shape", "row_mask", "codes", "walls", "phased", "expansion")

    def __init__(self, shape: Sequence[int], row_mask: Sequence[int], codes: Sequence[int],
                 walls: Sequence[int], phased: Sequence[int], expansion: str,
                 residency: Residency, targets: Sequence[Any], sources: Sequence[Any],
                 inverse: Sequence[Any], rows: Sequence[Any], phase_down: Sequence[Any],
                 phase_up: Sequence[Any], functions: Mapping[str, Any],
                 volumes: Sequence[str]) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = int(np.prod(self.shape))
        self.row_mask = tuple(int(value) for value in row_mask)
        self.codes, self.walls, self.phased = (tuple(int(value) for value in values)
                                                for values in (codes, walls, phased))
        self.expansion, self.residency = str(expansion), residency
        self.volumes = tuple(dict.fromkeys(volumes))
        bound_rows = tuple(inverse[index // 2] if value is None else value
                           for index, value in enumerate(rows))
        super().__init__(dict(functions), tuple(targets) + tuple(sources)
                         + tuple(inverse) + bound_rows
                         + (self.shape[0], self.shape[1], self.shape[2], self.n_elem)
                         + tuple(phase_down) + tuple(phase_up))


def _functions(row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
               phased: Sequence[int], expansion: str,
               contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_complex_no_pml_offdiag(
        row_mask, codes, walls, phased, expansion, mode) for mode in contract_variants}


def plan_metal_complex_no_pml_offdiag(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,), probe: Any = None,
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalComplexNoPmlOffdiagPlan]:
    """Build the covered complex/Bloch tensor-row plan, or return ``None``."""
    if not metal_complex_no_pml_offdiag_coverage(fields, pml, residency, probe).covered:
        return None
    assert residency is not None
    expansion = expansion_from_probe(probe if probe is not None else load_expansion_probe())
    if expansion is None:  # predicate already reported this refusal
        return None
    grid = fields.grid
    kinds = _boundary_kinds(grid, None)
    codes = tuple(templates.METALLIC if kind == "metallic" else templates.PERIODIC
                  for kind in kinds)
    phases = bloch_phase_table(grid, kinds)
    phased_down, phase_down = phase_arguments(phases, backward=True)
    phased_up, phase_up = phase_arguments(phases, backward=False)
    if phased_down != phased_up:
        raise ValueError("forward and backward phase flags disagree")
    rows = row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)
    target_names = tuple(name for name, _source, _axis in E_TERMS)
    source_names = tuple(source for _name, source, _axis in E_TERMS)
    targets = [_complex_mirror(residency, name, getattr(fields, name)) for name in target_names]
    sources = [_complex_mirror(residency, name, getattr(fields, name)) for name in source_names]
    inverse = [residency.mirror(f"inv_eps_{name}:complex_no_pml_offdiag",
                                fields.inverse_epsilon_for(name), constant=True)
               for name in target_names]
    mirrored_rows = [
        None if value is None else residency.mirror(
            f"chi1inv_offdiag:{row}:{partner}:complex_no_pml", value, constant=True)
        for (row, partner), value in zip(ROW_SLOTS, rows)
    ]
    selected = dict(functions or _functions(
        row_mask, codes, wall_mask_axes(grid), phased_down, expansion, contract_variants))
    return MetalComplexNoPmlOffdiagPlan(
        grid.shape, row_mask, codes, wall_mask_axes(grid), phased_down, expansion,
        residency, targets, sources, inverse, mirrored_rows, phase_down, phase_up,
        selected, target_names + source_names,
    )


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return (metal_complex_no_pml_offdiag_coverage(
        context.fields, context.pml, context.residency, context.extra.get("probe"))
            if slot == SLOT else Coverage(False, (f"complex tensor row cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexNoPmlOffdiagPlan]:
    return (plan_metal_complex_no_pml_offdiag(
        context.fields, context.pml, context.residency, context.contract_variants,
        context.extra.get("probe")) if slot == SLOT else None)


def _arm_gate(context: Any) -> bool:
    return not bool(getattr(context.pml, "is_active", False))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return (arms.register(
        family=FAMILY, slot=SLOT, label="complex no-PML off-diagonal",
        coverage=_arm_coverage, plan=_arm_plan, gate=_arm_gate,
        prefix="complex no-PML off-diagonal: ", noun="complex no-PML tensor constitutive",
        wired=True,
    ),)


ARMS = register_arms()
