"""Complex/Bloch no-PML stored-E constitutive update on Metal.

This is the complex64 sibling of :mod:`.no_pml_stored_e`.  It preserves ordered
complex ``D - P0 - P1 ...`` and multiplies the resulting field by real inverse
epsilon through the probe-licensed ``float2`` expansion.  Bloch is admitted: the
operation is pointwise and therefore has no phase-bearing neighbor access.

The native gate establishes the ordered stored-E update and its handoff to the
rotating ADE recurrence, including a zero-pole control. Its complete-step
composition with the complex conductive curls and no-PML-null magnetic side is
separately exercised before the arm is wired in the experimental composer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import (
    COVERED_SUSCEPTIBILITY_KINDS, ELECTRIC_COMPONENTS, Coverage, _call,
    _inverse_epsilon_reasons, _susceptibility_reasons,
)
from . import shaders, templates
from .ade_update_p import _physical_name
from .complex_fields import (
    _complex_volume_reasons, _expansion_reasons, expansion_from_probe,
    load_expansion_probe,
)
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source

FAMILY = "complex_no_pml_stored_e"
MAX_POLES = 8
E_TERMS = (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz"))

__all__ = [
    "E_TERMS", "FAMILY", "MAX_POLES", "MetalComplexStoredEPlan",
    "complex_stored_e_source", "compile_complex_stored_e",
    "metal_complex_stored_e_coverage", "plan_metal_complex_stored_e",
]

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
__HELPERS__
kernel void complex_stored_e_component(
    device float2*       e_out [[buffer(0)]],
    device const float2* d_in  [[buffer(1)]],
    device const float*  inv_e [[buffer(2)]],
    device const float2* p0    [[buffer(3)]],
    device const float2* p1    [[buffer(4)]],
    device const float2* p2    [[buffer(5)]],
    device const float2* p3    [[buffer(6)]],
    device const float2* p4    [[buffer(7)]],
    device const float2* p5    [[buffer(8)]],
    device const float2* p6    [[buffer(9)]],
    device const float2* p7    [[buffer(10)]],
    constant uint& n_elem      [[buffer(11)]],
    uint idx [[thread_position_in_grid]])
{
__BODY__
}
"""


def complex_stored_e_source(pole_count: int, expansion: str,
                            contract: str = shaders.CONTRACT_OFF) -> str:
    if not 0 <= int(pole_count) <= MAX_POLES:
        raise ValueError(f"pole_count must be in [0, {MAX_POLES}], got {pole_count!r}")
    chain = "\n".join(f"    source = source - p{index}[idx];"
                      for index in range(int(pole_count)))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": "\n".join((templates.GUARD, "    float2 source = d_in[idx];",
                                  chain,
                                  "    e_out[idx] = c_mul_field_left(source, inv_e[idx]);")),
    })


def compile_complex_stored_e(pole_count: int, expansion: str,
                             contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(complex_stored_e_source(
        pole_count, expansion, contract)).complex_stored_e_component


def _poles(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    states = tuple(getattr(fields, "polarizations", ()) or ())
    return {component: tuple(state for state in states
                             if _call(state, "drives", component, default=False))
            for component in ELECTRIC_COMPONENTS}


def metal_complex_stored_e_coverage(fields: Any, pml: Any, residency: Any = None,
                                    probe: Any = None) -> Coverage:
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons: List[str] = []
    reasons.extend(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    if not bool(getattr(fields, "force_complex_fields", False) or
                _call(grid, "has_bloch", default=False)):
        reasons.append("complex64 storage is required; real storage belongs to no-PML stored E")
    if pml is not None and bool(getattr(pml, "is_active", False)):
        reasons.append("an active PML layer requires the split stored-E update")
    if bool(getattr(fields, "_pml_active", False)):
        reasons.append("Fields remains in PML storage mode while the layer is inert")
    if not bool(getattr(fields, "stores_E", False)):
        reasons.append("E is recomputed from D rather than stored")
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append("an off-diagonal chi1inv row is installed")
    if bool(getattr(fields, "has_nonlinearity", False)):
        reasons.append("chi2/chi3 is installed")
    if (_call(grid, "has_symmetry", default=False) or
            any(_call(grid, "is_mirrored", axis, default=False) for axis in range(3))):
        reasons.append("a mirror plane is active")
    if bool(getattr(grid, "cylindrical", False)):
        reasons.append("cylindrical coordinates are not carried")
    if bool(getattr(grid, "bfast_active", False)):
        reasons.append("BFAST is active")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append("special_kz is nonzero")
    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return Coverage(False, tuple(dict.fromkeys(reasons)))
    if 2 * int(np.prod(shape)) >= 2 ** 31:
        reasons.append("the grid exceeds the complex int32 word-index range")
    reasons.extend(_susceptibility_reasons(fields))
    for component, displacement in E_TERMS:
        for name in (component, displacement):
            array = getattr(fields, name, None)
            if array is None:
                reasons.append(f"{name} is not allocated")
            else:
                reasons.extend(_complex_volume_reasons(name, array, shape))
    reasons.extend(_inverse_epsilon_reasons(fields, shape))
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
                else:
                    reasons.extend(_complex_volume_reasons(
                        f"polarization {index} {label}[{component}]", array, shape))
        scratch = getattr(state, "_scratch", None)
        if scratch is None:
            reasons.append(f"polarization {index} scratch is not allocated")
        else:
            reasons.extend(_complex_volume_reasons(f"polarization {index} scratch", scratch, shape))
    for component, states in _poles(fields).items():
        if len(states) > MAX_POLES:
            reasons.append(f"{component} is driven by {len(states)} poles, more than MAX_POLES={MAX_POLES}")
    reasons.extend(_expansion_reasons(probe))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


@dataclass(frozen=True)
class _Entry:
    component: str
    target: Any
    displacement: Any
    inverse_epsilon: Any
    states: Tuple[Any, ...]


class MetalComplexStoredEPlan:
    performs_device_work = True
    replaces_sub_steps = ("update_E",)

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], functions: Mapping[Tuple[str, int, str], Any],
                 tensors: Mapping[int, Any], expansion: str) -> None:
        self.entries, self.residency = tuple(entries), residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self._functions, self._tensors, self.expansion = dict(functions), dict(tensors), expansion
        self.launches = self.runs = 0
        self.launches_per_run = len(self.entries)

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        mode = shaders.CONTRACT_OFF if contract is None else contract
        for entry in self.entries:
            poles = tuple(state.P[entry.component] for state in entry.states)
            function = self._functions[(mode, len(poles), self.expansion)]
            slots = poles + (entry.displacement,) * (MAX_POLES - len(poles))
            function(self._tensors[id(entry.target)], self._tensors[id(entry.displacement)],
                     self._tensors[id(entry.inverse_epsilon)],
                     *(self._tensors[id(value)] for value in slots), int(entry.target.size))
            self.launches += 1
        self.runs += 1


def plan_metal_complex_stored_e(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,), probe: Any = None,
        functions: Optional[Mapping[Tuple[str, int, str], Any]] = None,
        ) -> Optional[MetalComplexStoredEPlan]:
    if not metal_complex_stored_e_coverage(fields, pml, residency, probe).covered:
        return None
    assert residency is not None
    expansion = expansion_from_probe(probe if probe is not None else load_expansion_probe())
    if expansion is None:
        return None
    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: set[Tuple[str, int, str]] = set()
    order = _poles(fields)
    for component, displacement_name in E_TERMS:
        target, displacement = getattr(fields, component), getattr(fields, displacement_name)
        inverse = fields.inverse_epsilon_for(component)
        for name, host, constant, dtype in ((component, target, False, np.complex64),
                                            (displacement_name, displacement, False, np.complex64),
                                            (f"inv_eps_{component}:no_pml", inverse, True, np.float32)):
            tensors[id(host)] = residency.mirror(name, host, constant=constant, dtype=dtype)
            volumes.append(name)
        states = order[component]
        entries.append(_Entry(component, target, displacement, inverse, states))
        required.update((mode, len(states), expansion) for mode in contract_variants)
    for state_index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for component in tuple(_call(state, "driven", default=()) or ()):
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                name = _physical_name(state_index, label, host)
                tensors[id(host)] = residency.mirror(name, host, dtype=np.complex64)
                volumes.append(name)
        scratch = state._scratch
        name = _physical_name(state_index, "scratch", scratch)
        tensors[id(scratch)] = residency.mirror(name, scratch, dtype=np.complex64)
        volumes.append(name)
    selected = dict(functions or {})
    for key in required:
        if key not in selected:
            selected[key] = compile_complex_stored_e(key[1], key[2], key[0])
    return MetalComplexStoredEPlan(entries, residency, volumes, selected, tensors, expansion)


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return (metal_complex_stored_e_coverage(context.fields, context.pml,
                                             context.residency, context.extra.get("complex_probe"))
            if slot == "update_E" else Coverage(False, (f"complex stored E cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexStoredEPlan]:
    return (plan_metal_complex_stored_e(context.fields, context.pml, context.residency,
                                        context.contract_variants, context.extra.get("complex_probe"))
            if slot == "update_E" else None)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, "update_E", "complex no-PML stored E", _arm_coverage,
                          _arm_plan, prefix="complex no-PML stored E: ",
                          noun="complex no-PML stored-E constitutive", wired=True),)


ARMS = register_arms()
