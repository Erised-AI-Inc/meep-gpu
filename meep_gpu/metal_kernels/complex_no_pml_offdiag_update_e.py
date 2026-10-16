"""Complex/Bloch no-PML tensor-row ``update_E`` on Metal.

**THE ROW IS MEEP'S DOUBLE ROW, PER PART, IN SOFTWARE BINARY64 (ruling R7).** MEEP
steps the two parts of a complex field separately with the same real coefficients
(``DOCMP``, update_eh.cpp:149), so each part's row is :mod:`.offdiag_update_e`'s:
``widen(gs*us)`` per part, each live term's float32 pair sum per part wall-masked
and added as ``scale_pow2(widen(sum), -2)`` where the cell's chunk keeps the term,
one rounding per part (``stepping._offdiagonal_terms``, ``_rounded_row``), every
double operation through :mod:`.softfp64` with ``fz = true``. The row text is
:mod:`.complex_folded_offdiag_update_e`'s, with no fold. A component with no live
row keeps the float32 ``gs*us``.

**THE BLOCH CONNECTION IS THE ARRAY PATH'S.** On a phased periodic axis the shifted
reads are plain wraps and the connection lands on the SAME-INDEX operand's plane 0
(``stepping._bloch_ghost_operand``): the partner pair's own word ``g[i]`` on the
partner axis's plane 0 takes ``eikna`` (``dp{a}``, the backward side of
:func:`.complex_fields.phase_arguments`), and the formed product
``(g[i] + g[i-sx]) * u[i]`` on the component's own axis's plane 0 takes
``conj(eikna)`` (``up{a}``, the forward side), each by MEEP's ``connect_phase``: a
copy, a per-part negation, or the unfused ``c_mul(word, phase)``
(:func:`.complex_fields.connection`). The far product is the plain up-shift of the
unconnected product.

**PRESENCE IS PER CHUNK (KERNELS.md §2).** The chunk record binds at
:data:`RECORD_BUFFER` and the six 27-bit presence words last at
:data:`PRESENCE_BUFFER`; the kernel reads them through
:func:`.offdiag_update_e.presence_block`, and the host builds them from the array
path's presence (:func:`.offdiag_update_e.presence_for`). A run without an absorber
steps one chunk unless the driver installed a division, so its words are all-kept.

THERE IS NO PER-CHUNK CONSTITUTIVE SELECT HERE. A constitutive sub-step without an
absorber carries no ``dsigw`` layer in any chunk, so ``E = src`` is the only form;
the record serves the presence labels only.

This is the Metal counterpart of Triton's complex no-PML off-diagonal family. It
owns the non-elementwise electric constitutive step when complex64 storage has live
real off-diagonal inverse-permittivity rows and the absorber is inactive. The weld
that lifts this body is :mod:`.complex_no_pml_offdiag_fused_electric_pair`.

The arm is selected only by the experimental Metal composer. Its direct native gate
establishes the tensor arithmetic. This is still not production driver dispatch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import Coverage, _boundary_kinds, _inverse_epsilon_reasons
from . import chunk_bounds as _chunk_bounds, shaders, templates
from . import offdiag_update_e as _offdiag
from .complex_fields import (
    CONNECT_KINDS,
    CONNECT_NEGATE,
    CONNECT_NONE,
    _complex_mirror,
    _complex_volume_reasons,
    bloch_phase_table,
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
#: 15 volumes + 4 scalars + six phase pairs + the chunk record + the presence words.
BINDING_COUNT = 27
#: The chunk record's binding index: one past the last phase pair.
RECORD_BUFFER = 25
#: The presence words' binding index: last.
PRESENCE_BUFFER = 26

if BINDING_COUNT > MAX_BUFFER_BINDINGS:  # pragma: no cover - platform invariant
    raise RuntimeError("complex no-PML tensor row exceeds Metal's binding ceiling")

__all__ = [
    "BINDING_COUNT", "FAMILY", "PRESENCE_BUFFER", "RECORD_BUFFER", "SLOT",
    "MetalComplexNoPmlOffdiagPlan", "complex_no_pml_offdiag_source",
    "compile_complex_no_pml_offdiag", "metal_complex_no_pml_offdiag_coverage",
    "plan_metal_complex_no_pml_offdiag",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__
__SOFTFP64__
__HELPERS__
__RECORD_STRUCT__
__PRESENCE_STRUCT__

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
    __RECORD_BINDING__,
    __PRESENCE_BINDING__,
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
__PRESENCE__
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


def _connect(axis: int, kinds: Sequence[int], value: str, above: bool) -> str:
    """MEEP's ``connect_phase`` of one same-index operand on plane 0 of ``axis``.

    ``kinds[axis]`` is :func:`.complex_fields.connection`'s: no line for a copy, a
    per-part negation, or the unfused product ``c_mul(word, phase)``
    (``part_products.connect_parts``). ``above`` names the word above the stored
    window (``eikna``, bound as ``dp{a}``, the partner pair's read) and its negation
    the word below it (``conj(eikna)``, ``up{a}``, the formed product's).
    """
    kind = int(kinds[axis])
    if kind == CONNECT_NONE:
        return ""
    _down, _up, _dv, _uv, coordinate, _extent, down_phase, up_phase = _NAMES[axis]
    phase = down_phase if above else up_phase
    connected = f"-{value}" if kind == CONNECT_NEGATE else f"c_mul({value}, {phase})"
    return f"    {value} = ({coordinate} == 0) ? {connected} : {value};"


def _term(component: int, partner_axis: int, coefficient: str, tag: str,
          phased: Sequence[int]) -> str:
    """One partner's OFFDIAG pair sum per part, in float32, as the array path forms it.

    ``pair = connect(g, partner, above) + shift_down(g, partner)``, ``product =
    u * pair`` per part, ``here = connect(product, own, below)``,
    ``summed = here + shift_up(product, own)`` (``stepping._offdiagonal_terms``): the
    shifts are plain wraps and the connections :func:`_connect`'s on plane 0.
    ``phased`` is the per-axis connection kind triple. The four displacement loads
    keep the home, partner-down, own-up and corner index shapes of :func:`_index`,
    which the weld's needles redirect.
    """
    own_axis = E_TERMS[component][2]
    down, _up, dvalid, _uvalid, _coordinate, _extent, _dp, _upp = _NAMES[partner_axis]
    _down, own_up, _dvalid, own_valid, _coord, _oextent, _odp, _oupp = _NAMES[own_axis]
    partner = f"g{partner_axis}"
    down_index = _index({partner_axis: down})
    up_index = _index({own_axis: own_up})
    corner_index = _index({own_axis: own_up, partner_axis: down})
    lines = [
        f"    float2 home_{tag} = {partner}[ii];",
        _connect(partner_axis, phased, f"home_{tag}", True),
        f"    float2 down_{tag} = {dvalid} ? {partner}[{down_index}] : float2(0.0f, 0.0f);",
        f"    float2 near_{tag} = home_{tag} + down_{tag};",
        f"    float2 near_product_{tag} = c_mul_field_left(near_{tag}, {coefficient}[ii]);",
        _connect(own_axis, phased, f"near_product_{tag}", False),
        f"    float2 far_a_{tag} = {own_valid} ? {partner}[{up_index}] : float2(0.0f, 0.0f);",
        _connect(partner_axis, phased, f"far_a_{tag}", True),
        f"    float2 corner_{tag} = ({own_valid} && {dvalid}) ? {partner}[{corner_index}] : float2(0.0f, 0.0f);",
        f"    float2 far_pair_{tag} = far_a_{tag} + corner_{tag};",
        f"    float2 far_product_{tag} = {own_valid} ? c_mul_field_left(far_pair_{tag}, {coefficient}[{up_index}]) : float2(0.0f, 0.0f);",
        f"    float2 sum_{tag} = near_product_{tag} + far_product_{tag};",
    ]
    return "\n".join(line for line in lines if line)


def _component_source(component: int, row_mask: Sequence[int], walls: Sequence[int],
                      phased: Sequence[int]) -> str:
    """One component's ``src{c}``: the float32 diagonal, or the double row per part.

    The row text is :mod:`.complex_folded_offdiag_update_e`'s: each term's float32
    pair sum wall-masked per part, ``widen(gs*us)`` per part, each kept term scaled by
    MEEP's double ``0.25`` and added left to right, one rounding per part.
    """
    diagonal = f"c_mul_field_left(g{component}[ii], e{component}[ii])"
    active = [offset for offset in (0, 1) if row_mask[2 * component + offset]]
    if not active:
        return f"    float2 src{component} = {diagonal};"
    c = component
    lines: List[str] = []
    tags: List[str] = []
    for offset in active:
        tag = f"{component}{offset}"
        lines.append(_term(component, TRANSVERSE_PARTNERS[component][offset],
                           _ROW_BUFFERS[2 * component + offset], tag, phased))
        tags.append(tag)
    # The wall mask zeroes each term's float32 pair sum (both parts) before its scale.
    for tag in tags:
        for axis in WALL_MASK_AXES[component]:
            if walls[axis]:
                lines.append(f"    sum_{tag} = at_{'xyz'[axis]} ? float2(0.0f, 0.0f) : sum_{tag};")
    lines.append(f"    float2 od_d{c} = {diagonal};")
    lines.append(f"    ulong od_re{c} = sd_widen_f(od_d{c}.x, true);")
    lines.append(f"    ulong od_im{c} = sd_widen_f(od_d{c}.y, true);")
    for tag in tags:
        for part, word in (("re", "x"), ("im", "y")):
            lines.append(
                f"    if (od_kept_{tag}) {{ od_{part}{c} = sd_add(od_{part}{c}, "
                f"sd_scale_pow2(sd_widen_f(sum_{tag}.{word}, true), {_offdiag.QUARTER}, true), "
                f"true); }}")
    lines.append(f"    float2 src{c} = float2(sd_narrow_f(od_re{c}, true), "
                 f"sd_narrow_f(od_im{c}, true));")
    return "\n".join(lines)


def complex_no_pml_offdiag_source(row_mask: Sequence[int], codes: Sequence[int],
                                  walls: Sequence[int], phased: Sequence[int],
                                  contract: str = shaders.CONTRACT_OFF) -> str:
    """Emit a specialized complex no-PML tensor constitutive shader.

    ``phased`` is the per-axis connection kind triple from
    :func:`.complex_fields.phase_arguments` (:data:`.complex_fields.CONNECT_KINDS`),
    ``codes`` the PERIODIC/METALLIC ghost triple, ``walls`` the metallic-wall
    declaration triple. Every kind is emitted as MEEP connects it (:func:`_connect`);
    a connection on an axis that does not wrap is refused by name.
    """
    row_mask = tuple(int(value) for value in row_mask)
    codes = tuple(int(value) for value in codes)
    walls = tuple(int(value) for value in walls)
    phased = tuple(int(value) for value in phased)
    if len(row_mask) != len(ROW_SLOTS):
        raise ValueError("row_mask must carry six tensor-row flags")
    if not any(row_mask):
        raise ValueError("all-dead tensor rows belong to the elementwise family")
    if any(value not in (0, 1) for value in (*row_mask, *walls)):
        raise ValueError("row masks and walls must be 0 or 1")
    if len(phased) != 3 or any(kind not in CONNECT_KINDS for kind in phased):
        raise ValueError(f"phased must be three connection kinds of {CONNECT_KINDS}, "
                         f"got {phased!r}")
    if len(codes) != 3 or any(value not in (templates.PERIODIC, templates.METALLIC)
                              for value in codes):
        raise ValueError("codes must be three periodic/metallic values")
    if any(kind != CONNECT_NONE and code != templates.PERIODIC
           for kind, code in zip(phased, codes)):
        raise ValueError("a Bloch connection requires a periodic boundary")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_BINDING__": _chunk_bounds.binding(RECORD_BUFFER),
        **_offdiag.presence_substitutions(row_mask, codes, PRESENCE_BUFFER),
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
                                   contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(complex_no_pml_offdiag_source(
        row_mask, codes, walls, phased, contract)).complex_no_pml_offdiag_update_e


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
        fields: Any, pml: Any, residency: Any = None,
        chunk_bounds: Any = None) -> Coverage:
    """Conservative predicate for the complex no-absorber tensor row.

    The complex no-PML base clauses, no polarization, the complex volumes, the
    inverse epsilon and the live rows' clauses, and last, on an otherwise admitted
    run, the per-chunk presence words (:func:`.offdiag_update_e.presence_reasons`
    through ``chunk_bounds``, the record the composer binds).
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons = list(_base_reasons(fields, pml, grid, residency))
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
    # (p) The per-chunk presence words, on the codes the kernel bakes, only on an
    #     otherwise admitted run.
    if not reasons:
        reasons.extend(_offdiag.presence_reasons(fields, pml, _codes(grid),
                                                 chunk_bounds))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def _codes(grid: Any) -> Tuple[int, int, int]:
    """The PERIODIC/METALLIC ghost triple the kernel bakes, from the array path's kinds."""
    kinds = _boundary_kinds(grid, None)
    return tuple(templates.METALLIC if kind == "metallic" else templates.PERIODIC
                 for kind in kinds)  # type: ignore[return-value]


class MetalComplexNoPmlOffdiagPlan(KernelPlan):
    """Persistent MPS mirrors for one specialized complex tensor-row update.

    The chunk record binds at :data:`RECORD_BUFFER` and the presence words last at
    :data:`PRESENCE_BUFFER`, each required with no default and checked by
    :func:`.chunk_bounds.plan_words` and :func:`.offdiag_update_e.presence_plan_words`.
    """

    __slots__ = ("shape", "n_elem", "row_mask", "codes", "walls", "phased",
                 "residency", "volumes", "chunk_bounds", "presence")
    REPR_FIELDS = ("shape", "row_mask", "codes", "walls", "phased")

    def __init__(self, shape: Sequence[int], row_mask: Sequence[int], codes: Sequence[int],
                 walls: Sequence[int], phased: Sequence[int],
                 residency: Residency, targets: Sequence[Any], sources: Sequence[Any],
                 inverse: Sequence[Any], rows: Sequence[Any], phase_down: Sequence[Any],
                 phase_up: Sequence[Any], functions: Mapping[str, Any],
                 volumes: Sequence[str], *, chunk_bounds: Any, presence: Any) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = int(np.prod(self.shape))
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, "MetalComplexNoPmlOffdiagPlan")
        self.presence = _offdiag.presence_plan_words(
            presence, "MetalComplexNoPmlOffdiagPlan")
        self.row_mask = tuple(int(value) for value in row_mask)
        self.codes, self.walls, self.phased = (tuple(int(value) for value in values)
                                                for values in (codes, walls, phased))
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        bound_rows = tuple(inverse[index // 2] if value is None else value
                           for index, value in enumerate(rows))
        device = getattr(residency, "device", "mps")
        super().__init__(dict(functions), tuple(targets) + tuple(sources)
                         + tuple(inverse) + bound_rows
                         + (self.shape[0], self.shape[1], self.shape[2], self.n_elem)
                         + tuple(phase_down) + tuple(phase_up)
                         + (_chunk_bounds.tensor(self.chunk_bounds, device),
                            _offdiag.presence_tensor(self.presence, device)))


def _functions(row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
               phased: Sequence[int],
               contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_complex_no_pml_offdiag(
        row_mask, codes, walls, phased, mode) for mode in contract_variants}


def plan_metal_complex_no_pml_offdiag(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None, *, chunk_bounds: Any,
        ) -> Optional[MetalComplexNoPmlOffdiagPlan]:
    """Build the covered complex/Bloch tensor-row plan, or return ``None``.

    ``chunk_bounds`` is the record of the division the run steps with (the composer's
    ``StepContext.chunk_bounds``); required, with no default. The presence words are
    built against it (:func:`.offdiag_update_e.presence_for`).
    """
    if not metal_complex_no_pml_offdiag_coverage(fields, pml, residency,
                                                 chunk_bounds).covered:
        return None
    assert residency is not None
    grid = fields.grid
    kinds = _boundary_kinds(grid, None)
    codes = _codes(grid)
    presence = _offdiag.presence_for(fields, pml, codes, chunk_bounds)
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
        row_mask, codes, wall_mask_axes(grid), phased_down, contract_variants))
    return MetalComplexNoPmlOffdiagPlan(
        grid.shape, row_mask, codes, wall_mask_axes(grid), phased_down,
        residency, targets, sources, inverse, mirrored_rows, phase_down, phase_up,
        selected, target_names + source_names,
        chunk_bounds=chunk_bounds, presence=presence,
    )


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return (metal_complex_no_pml_offdiag_coverage(
        context.fields, context.pml, context.residency,
        getattr(context, "chunk_bounds", None))
            if slot == SLOT else Coverage(False, (f"complex tensor row cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexNoPmlOffdiagPlan]:
    return (plan_metal_complex_no_pml_offdiag(
        context.fields, context.pml, context.residency, context.contract_variants,
        chunk_bounds=context.chunk_bounds)
            if slot == SLOT else None)


def _arm_gate(context: Any) -> bool:
    return not bool(getattr(context.pml, "is_active", False))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    # Wired: the row is the array path's double row per part (ruling R7, software
    # binary64), with the array path's Bloch connection and per-chunk presence.
    return (arms.register(
        family=FAMILY, slot=SLOT, label="complex no-PML off-diagonal",
        coverage=_arm_coverage, plan=_arm_plan, gate=_arm_gate,
        prefix="complex no-PML off-diagonal: ", noun="complex no-PML tensor constitutive",
        wired=True,
    ),)


ARMS = register_arms()
