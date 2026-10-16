"""Complex folded tensor-row ``update_E`` with the active-PML stored-E tail.

**THE ROW IS MEEP'S DOUBLE ROW, PER PART, IN SOFTWARE BINARY64 (ruling R7).** MEEP
steps the two parts of a complex field separately with the same real coefficients
(``DOCMP``, update_eh.cpp:149), so each part's row is :mod:`.offdiag_update_e`'s:
``widen(gs*us)`` per part, each live term's float32 pair sum per part wall-masked and
added as ``scale_pow2(widen(sum), -2)`` where the cell's chunk keeps the term, one
rounding per part (``stepping._offdiagonal_terms``, ``_rounded_row``). Presence is the
real family's 27-bit words (:func:`.offdiag_update_e.presence_block`), bound last.

**THE BLOCH CONNECTION IS THE ARRAY PATH'S.** On a phased periodic axis the shifted
reads are plain wraps and the connection lands on the SAME-INDEX operand's plane 0
(``stepping._bloch_ghost_operand``): the partner pair's own word ``g[i]`` on the
partner axis's plane 0 takes ``eikna`` (the word above the window), and the formed
product ``(g[i] + g[i-sx]) * u[i]`` on the component's own axis's plane 0 takes
``conj(eikna)`` (the word below it), each by MEEP's ``connect_phase``: a copy, a
per-part negation, or the unfused ``complex<realnum>(phase) * word``
(:func:`.complex_fields.connection`). The far product is the plain up-shift of the
unconnected product.

**THE PML TAIL IS THE ARRAY PATH'S PER-CHUNK SELECT**, KM5's complex constitutive
tail (:mod:`.complex_fields`) character for character: the chunk record binds last
and, on a cell whose chunk carries the component's own (``dsigw``) layer, ``prev`` is
read before ``f_w`` is written and ``E = E + ((kps*src) - (kms*prev))`` per part;
everywhere else ``E = src`` and ``f_w`` is neither read nor written.

This is distinct from the no-PML complex tensor shader: it stores the row product
in ``f_w_E`` on the accumulation's cells.  The six connection phases are packed into
one immutable ``float2`` buffer (``ph[a]``: ``eikna`` for the partner reads on axis
``a``; ``ph[a + 3]``: ``conj(eikna)`` for the own-axis product), keeping the
signature at 31 bindings, the ceiling (the record, then the presence words last).
Mirror parity is a coefficient-left complex multiplication on the partner DOWN ghost
only.

The native tensor gate and the experimental composer separately cover its folded
complex curl/fill/tensor-row residency composition. This does not enable production
driver dispatch.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import Coverage, _boundary_kinds, _coefficient_reasons, _inverse_epsilon_reasons
from . import chunk_bounds as _chunk_bounds, shaders, templates
from . import complex_fields as _complex
from . import folded_complex as _folded_complex
from . import folded_offdiag_update_e as _folded_real
from . import offdiag_update_e as _offdiag
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .plans import KernelPlan

FAMILY = "complex_folded_offdiag_update_e"
SLOT = "update_E"
LABEL = "complex folded off-diagonal PML E"
BINDING_COUNT = 31
#: The chunk record's binding index: one past ``n_elem``.
RECORD_BUFFER = 29
#: The presence words' binding index: last, at Metal's ceiling.
PRESENCE_BUFFER = 30
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
__SOFTFP64__
__HELPERS__
__RECORD_STRUCT__
__PRESENCE_STRUCT__
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
    constant uint& n_elem [[buffer(28)]], __RECORD_BINDING__,
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
    // --- the chunk record: each component's own-axis membership at this cell -----
__BOUNDS__
__PRESENCE__
    float kp_0 = kp0[i], km_0 = km0[i];
    float kp_1 = kp1[j], km_1 = km1[j];
    float kp_2 = kp2[k], km_2 = km2[k];
    const bool dw0 = __DW0__, dw1 = __DW1__, dw2 = __DW2__;
    // The PML tail is complex_fields' per-chunk select: on a dsigw cell `prev` is
    // read BEFORE `fw` is written and f + ((kps*src) - (kms*prev)) is formed per
    // part; elsewhere the word is src itself and fw is not touched.
    float2 prev0 = dw0 ? w0[ii] : float2(0.0f, 0.0f);
__SRC0__
    if (dw0) { w0[ii] = src0; }
    float2 a0 = src0;
    if (dw0) { a0 = f0[ii] + (c_mul_coefficient_left(kp_0, src0) - c_mul_coefficient_left(km_0, prev0)); }
    f0[ii] = a0;
    float2 prev1 = dw1 ? w1[ii] : float2(0.0f, 0.0f);
__SRC1__
    if (dw1) { w1[ii] = src1; }
    float2 a1 = src1;
    if (dw1) { a1 = f1[ii] + (c_mul_coefficient_left(kp_1, src1) - c_mul_coefficient_left(km_1, prev1)); }
    f1[ii] = a1;
    float2 prev2 = dw2 ? w2[ii] : float2(0.0f, 0.0f);
__SRC2__
    if (dw2) { w2[ii] = src2; }
    float2 a2 = src2;
    if (dw2) { a2 = f2[ii] + (c_mul_coefficient_left(kp_2, src2) - c_mul_coefficient_left(km_2, prev2)); }
    f2[ii] = a2;
}
"""

_STRIDE = {0: "{} * nyz", 1: "{} * nzi", 2: "{}"}
_HOME = ("i", "j", "k")
_ROW_BUFFERS = ("u01", "u02", "u11", "u12", "u21", "u22")


def _index(shifts: Mapping[int, str]) -> str:
    return " + ".join(_STRIDE[axis].format(shifts.get(axis, _HOME[axis]))
                       for axis in range(3))


def _connect(axis: int, kinds: Sequence[int], value: str, phase: str) -> str:
    """MEEP's ``connect_phase`` of one same-index operand on plane 0 of ``axis``.

    ``kinds[axis]`` is :func:`.complex_fields.connection`'s: no line for a copy, a
    per-part negation, or the unfused product ``c_mul(word, phase)``
    (``part_products.connect_parts``). ``phase`` names the packed pair.
    """
    kind = int(kinds[axis])
    if kind == _complex.CONNECT_NONE:
        return ""
    coordinate = ("i", "j", "k")[axis]
    connected = f"-{value}" if kind == _complex.CONNECT_NEGATE else f"c_mul({value}, {phase})"
    return f"    {value} = ({coordinate} == 0) ? {connected} : {value};"


def _term(component: int, partner_axis: int, coefficient: str, tag: str,
          phased: Sequence[int], negate: Sequence[int]) -> str:
    """One partner's OFFDIAG pair sum per part, in float32, as the array path forms it.

    ``pair = connect(g, partner, above) + shift_down(g, partner)``, ``product =
    u * pair`` per part, ``here = connect(product, own, below)``,
    ``summed = here + shift_up(product, own)`` (``stepping._offdiagonal_terms``); the
    shifts are plain wraps, the mirror parity rides on the partner DOWN ghost, and
    the connections are :func:`_connect`'s on plane 0. ``phased`` is the per-axis
    connection kind triple.
    """
    own_axis = E_TERMS[component][2]
    down, _up, down_valid, _up_valid, _extent = _offdiag._TWO_WAY_NAMES["xyz"[partner_axis]]
    _down, own_up, _dv, own_valid, _extent = _offdiag._TWO_WAY_NAMES["xyz"[own_axis]]
    partner = f"g{partner_axis}"
    down_index = _index({partner_axis: down})
    up_index = _index({own_axis: own_up})
    corner_index = _index({own_axis: own_up, partner_axis: down})
    at = ("at_x", "at_y", "at_z")[partner_axis]
    above = f"ph[{partner_axis}]"
    below = f"ph[{own_axis + 3}]"
    parity_down = (f"    down_{tag} = ({at} ? c_mul_coefficient_left(-1.0f, down_{tag}) : down_{tag});"
                   if negate[partner_axis] else "")
    parity_corner = (f"    corner_{tag} = ({at} ? c_mul_coefficient_left(-1.0f, corner_{tag}) : corner_{tag});"
                     if negate[partner_axis] else "")
    lines = [
        f"    float2 home_{tag} = {partner}[ii];",
        _connect(partner_axis, phased, f"home_{tag}", above),
        f"    float2 down_{tag} = {down_valid} ? {partner}[{down_index}] : float2(0.0f, 0.0f);",
        parity_down,
        f"    float2 near_{tag} = home_{tag} + down_{tag};",
        f"    float2 near_product_{tag} = c_mul_field_left(near_{tag}, {coefficient}[ii]);",
        _connect(own_axis, phased, f"near_product_{tag}", below),
        f"    float2 far_a_{tag} = {own_valid} ? {partner}[{up_index}] : float2(0.0f, 0.0f);",
        _connect(partner_axis, phased, f"far_a_{tag}", above),
        f"    float2 corner_{tag} = ({own_valid} && {down_valid}) ? {partner}[{corner_index}] : float2(0.0f, 0.0f);",
        parity_corner,
        f"    float2 far_pair_{tag} = far_a_{tag} + corner_{tag};",
        f"    float2 far_product_{tag} = {own_valid} ? c_mul_field_left(far_pair_{tag}, {coefficient}[{up_index}]) : float2(0.0f, 0.0f);",
        f"    float2 sum_{tag} = near_product_{tag} + far_product_{tag};",
    ]
    return "\n".join(line for line in lines if line)


def _component_source(component: int, row_mask: Sequence[int], walls: Sequence[int],
                      phased: Sequence[int], negate: Sequence[int]) -> str:
    """One component's ``src{c}``: the float32 diagonal, or the double row per part."""
    diagonal = f"c_mul_field_left(g{component}[ii], e{component}[ii])"
    active = [offset for offset in (0, 1) if row_mask[2 * component + offset]]
    if not active:
        return f"    float2 src{component} = {diagonal};"
    c = component
    lines: List[str] = []
    tags: List[str] = []
    for offset in active:
        tag = f"{component}{offset}"
        lines.append(_term(component, _offdiag.TRANSVERSE_PARTNERS[component][offset],
                           _ROW_BUFFERS[2 * component + offset], tag, phased, negate))
        tags.append(tag)
    # The wall mask zeroes each term's float32 pair sum (both parts) before its scale.
    for tag in tags:
        for axis in _offdiag.WALL_MASK_AXES[component]:
            if walls[axis]:
                lines.append(f"    sum_{tag} = at_{'xyz'[axis]} ? float2(0.0f, 0.0f) : sum_{tag};")
    # The row per part: widen(gs*us), each kept term's sum scaled by MEEP's double
    # 0.25 and added left to right, one rounding per part.
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


def complex_folded_offdiag_source(row_mask: Sequence[int], codes: Sequence[int],
                                  walls: Sequence[int], phased: Sequence[int],
                                  negate: Sequence[int],
                                  contract: str = shaders.CONTRACT_OFF) -> str:
    row_mask, codes = tuple(map(int, row_mask)), tuple(map(int, codes))
    walls, phased, negate = tuple(map(int, walls)), tuple(map(int, phased)), tuple(map(int, negate))
    if len(row_mask) != 6 or not any(row_mask):
        raise ValueError("row_mask must carry at least one of the six tensor slots")
    if any(value not in (0, 1) for value in (*row_mask, *walls, *negate)):
        raise ValueError("row masks, walls and parity flags must be 0 or 1")
    if len(phased) != 3 or any(kind not in _complex.CONNECT_KINDS for kind in phased):
        raise ValueError(f"phased must be three connection kinds of "
                         f"{_complex.CONNECT_KINDS}, got {phased!r}")
    if len(codes) != 3 or any(code not in (0, 1, 2, 3) for code in codes):
        raise ValueError("codes must be three periodic/metallic/mirror values")
    if any(kind != _complex.CONNECT_NONE and code != templates.PERIODIC
           for kind, code in zip(phased, codes)):
        raise ValueError("a Bloch connection requires an unfolded periodic axis")
    if any(flag and code not in _folded_real.MIRROR_CODES for flag, code in zip(negate, codes)):
        raise ValueError("a mirror-parity flag requires a mirror boundary")
    flags = shaders.constitutive_membership_flags("E")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(), "__GUARD__": templates.GUARD,
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_BINDING__": _chunk_bounds.binding(RECORD_BUFFER),
        **_offdiag.presence_substitutions(row_mask, codes, PRESENCE_BUFFER),
        "__BOUNDS__": shaders.constitutive_bounds_block("E"),
        "__DW0__": flags[0], "__DW1__": flags[1], "__DW2__": flags[2],
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
                                   negate: Sequence[int],
                                   contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(complex_folded_offdiag_source(
        row_mask, codes, walls, phased, negate, contract)).complex_folded_offdiag_update_e


def metal_complex_folded_offdiag_coverage(
        fields: Any, pml: Any, residency: Any = None,
        chunk_bounds: Any = None) -> Coverage:
    """May the Metal complex folded tensor row step ``update_E`` for this pair?

    The folded complex grid clauses, a fold, a live row slot, the certified rows'
    clauses, complex volumes, the inverse-epsilon and half-integer coefficient tables,
    the stored extent, and the per-chunk presence words
    (:func:`.offdiag_update_e.presence_reasons` on the folded codes, through
    ``chunk_bounds``, the record the composer binds).
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons, codes = _folded_complex._folded_complex_grid_reasons(
        fields, pml, grid, residency)
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
    # (p) The per-chunk presence words, on an otherwise admitted run.
    if not reasons and codes is not None:
        reasons.extend(_offdiag.presence_reasons(fields, pml, codes, chunk_bounds))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalComplexFoldedOffdiagPlan(KernelPlan):
    __slots__ = ("shape", "n_elem", "row_mask", "codes", "walls", "phased", "negate", "residency",
                 "volumes", "chunk_bounds", "presence")
    REPR_FIELDS = ("shape", "row_mask", "codes", "walls", "phased", "negate")

    def __init__(self, shape: Sequence[int], row_mask: Sequence[int], codes: Sequence[int],
                 walls: Sequence[int], phased: Sequence[int], negate: Sequence[int],
                 residency: Residency, targets: Sequence[Any],
                 auxiliaries: Sequence[Any], sources: Sequence[Any], inverse: Sequence[Any],
                 rows: Sequence[Any], coefficients: Sequence[Any], phase_pack: Any,
                 functions: Mapping[str, Any], volumes: Sequence[str], *,
                 chunk_bounds: Any, presence: Any) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = int(np.prod(self.shape))
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, "MetalComplexFoldedOffdiagPlan")
        self.presence = _offdiag.presence_plan_words(
            presence, "MetalComplexFoldedOffdiagPlan")
        self.row_mask = tuple(int(value) for value in row_mask)
        self.codes, self.walls, self.phased, self.negate = (
            tuple(int(value) for value in values)
            for values in (codes, walls, phased, negate))
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        bound_rows = tuple(inverse[index // 2] if value is None else value
                           for index, value in enumerate(rows))
        super().__init__(dict(functions), tuple(targets) + tuple(auxiliaries)
                         + tuple(sources) + tuple(inverse) + bound_rows
                         + tuple(coefficients) + (phase_pack, *self.shape, self.n_elem,
                                                  _chunk_bounds.tensor(
                                                      self.chunk_bounds,
                                                      getattr(residency, "device", "mps")),
                                                  _offdiag.presence_tensor(
                                                      self.presence,
                                                      getattr(residency, "device", "mps"))))


def _functions(row_mask, codes, walls, phased, negate, contract_variants):
    return {mode: compile_complex_folded_offdiag(
        row_mask, codes, walls, phased, negate, mode)
            for mode in contract_variants}


def plan_metal_complex_folded_offdiag(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None, *, chunk_bounds: Any,
        ) -> Optional[MetalComplexFoldedOffdiagPlan]:
    """The plan, or None when refused.

    ``chunk_bounds`` is the record of the division the run steps with (the
    composer's ``StepContext.chunk_bounds``); required, with no default. The
    presence words are built against it (:func:`.offdiag_update_e.presence_for`).
    """
    if not metal_complex_folded_offdiag_coverage(fields, pml, residency,
                                                 chunk_bounds).covered:
        return None
    assert residency is not None and pml is not None
    codes, _reasons = _folded_complex.folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - coverage already refuses
        return None
    presence = _offdiag.presence_for(fields, pml, codes, chunk_bounds)
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
                                              negate, contract_variants))
    return MetalComplexFoldedOffdiagPlan(
        fields.grid.shape, row_mask, codes, walls, phased_down, negate,
        residency, targets, auxiliaries, sources, inverse, mirrored_rows, coefficients,
        mirrored_phases, selected,
        target_names + tuple("f_w_" + name for name in target_names) + source_names,
        chunk_bounds=chunk_bounds, presence=presence)


def _arm_gate(context: Any) -> bool:
    try:
        return bool(_folded_complex._has_real_fold(context.fields.grid)
                    and getattr(context.pml, "is_active", False)
                    and any(value is not None for value in _offdiag.row_volumes_for(context.fields)))
    except Exception:  # noqa: BLE001
        return False


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return (metal_complex_folded_offdiag_coverage(
        context.fields, context.pml, context.residency,
        getattr(context, "chunk_bounds", None))
            if slot == SLOT else Coverage(False, (f"complex folded tensor row cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexFoldedOffdiagPlan]:
    return (plan_metal_complex_folded_offdiag(
        context.fields, context.pml, context.residency, context.contract_variants,
        chunk_bounds=context.chunk_bounds)
            if slot == SLOT else None)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return (arms.register(
        FAMILY, SLOT, LABEL, _arm_coverage, _arm_plan, gate=_arm_gate,
        prefix=f"{LABEL}: ", noun="complex folded tensor PML constitutive",
        wired=True),)


ARMS = register_arms()
