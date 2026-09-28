"""Metal ARM S: no-PML stored-E constitutive update for real float32 fields.

This is the Metal counterpart of ``triton_kernels.no_pml_stored_e``.  It owns
``update_E`` when E is stored, the absorber is inactive, and dispersion changes
the source from D to ordered ``D - P0 - P1 ...``.  The body is deliberately
per-component: an all-component eight-pole signature needs 34 buffers, above
Metal's measured 31-buffer ceiling.  Three launches retain the engine's order
without inventing a packed-pointer representation.

Registration affects only the experimental Metal composer; production dispatch
remains disabled.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import (
    COVERED_SUSCEPTIBILITY_KINDS,
    ELECTRIC_COMPONENTS,
    Coverage,
    _call,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
    _volume_reasons,
)
from . import shaders, templates
from .ade_update_p import _physical_name
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source

FAMILY = "no_pml_stored_e"
MAX_POLES = 8
E_TERMS = (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz"))

__all__ = [
    "E_TERMS", "FAMILY", "MAX_POLES", "MetalStoredEPlan", "stored_e_source",
    "compile_stored_e", "metal_stored_e_coverage", "plan_metal_stored_e",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void stored_e_component(
    device float*       e_out [[buffer(0)]],
    device const float* d_in  [[buffer(1)]],
    device const float* inv_e [[buffer(2)]],
    device const float* p0    [[buffer(3)]],
    device const float* p1    [[buffer(4)]],
    device const float* p2    [[buffer(5)]],
    device const float* p3    [[buffer(6)]],
    device const float* p4    [[buffer(7)]],
    device const float* p5    [[buffer(8)]],
    device const float* p6    [[buffer(9)]],
    device const float* p7    [[buffer(10)]],
    constant uint& n_elem     [[buffer(11)]],
    uint idx [[thread_position_in_grid]])
{
__BODY__
}
"""


def stored_e_source(pole_count: int,
                    contract: str = shaders.CONTRACT_OFF) -> str:
    """Emit one bounded, left-associated component subtraction chain."""
    if not 0 <= int(pole_count) <= MAX_POLES:
        raise ValueError(f"pole_count must be in [0, {MAX_POLES}], got {pole_count!r}")
    chain = "\n".join(f"    source = source - p{index}[idx];"
                      for index in range(int(pole_count)))
    body = "\n".join((templates.GUARD, "    float source = d_in[idx];", chain,
                      "    e_out[idx] = source * inv_e[idx];"))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__BODY__": body,
    })


def compile_stored_e(pole_count: int,
                     contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(stored_e_source(pole_count, contract)).stored_e_component


def _poles(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    states = tuple(getattr(fields, "polarizations", ()) or ())
    return {component: tuple(state for state in states
                             if _call(state, "drives", component, default=False))
            for component in ELECTRIC_COMPONENTS}


def metal_stored_e_coverage(fields: Any, pml: Any,
                            residency: Any = None) -> Coverage:
    """Conservative coverage for the real no-PML stored-E update."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons: List[str] = []
    reasons.extend(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    if bool(getattr(fields, "force_complex_fields", False) or
            _call(grid, "has_bloch", default=False)):
        reasons.append("complex64 storage is not carried by this float32 body")
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
    if int(np.prod(shape)) >= 2 ** 31:
        reasons.append("the grid exceeds the kernel's int32 index range")
    reasons.extend(_susceptibility_reasons(fields))
    names = tuple(name for term in E_TERMS for name in term)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_layout_reasons(fields, shape, names))
    reasons.extend(_inverse_epsilon_reasons(fields, shape))
    for state_index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        kind = getattr(getattr(state, "susceptibility", None), "kind", None)
        if kind not in COVERED_SUSCEPTIBILITY_KINDS:
            reasons.append(f"polarization {state_index} kind {kind!r} is not covered")
        for component in tuple(_call(state, "driven", default=()) or ()):
            if component not in ELECTRIC_COMPONENTS:
                reasons.append(f"polarization {state_index} drives {component!r}")
                continue
            for label in ("P", "P_prev"):
                array = (getattr(state, label, {}) or {}).get(component)
                if array is None:
                    reasons.append(f"polarization {state_index} {label}[{component}] is not allocated")
                else:
                    reasons.extend(_volume_reasons(
                        f"polarization {state_index} {label}[{component}]", array, shape))
        scratch = getattr(state, "_scratch", None)
        if scratch is None:
            reasons.append(f"polarization {state_index} scratch is not allocated")
        else:
            reasons.extend(_volume_reasons(
                f"polarization {state_index} scratch", scratch, shape))
    for component, states in _poles(fields).items():
        if len(states) > MAX_POLES:
            reasons.append(f"{component} is driven by {len(states)} poles, more than MAX_POLES={MAX_POLES}")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


@dataclass(frozen=True)
class _Entry:
    component: str
    target: Any
    displacement: Any
    inverse_epsilon: Any
    states: Tuple[Any, ...]


class MetalStoredEPlan:
    """Persistent mirrors with per-launch resolution of rotating P arrays."""

    performs_device_work = True
    replaces_sub_steps = ("update_E",)

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], functions: Mapping[Tuple[str, int], Any],
                 tensor_by_host: Mapping[int, Any]) -> None:
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self._functions = dict(functions)
        self._tensor_by_host = dict(tensor_by_host)
        self.launches = 0
        self.runs = 0
        self.launches_per_run = len(self.entries)

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        mode = shaders.CONTRACT_OFF if contract is None else contract
        for entry in self.entries:
            poles = tuple(state.P[entry.component] for state in entry.states)
            function = self._functions[(mode, len(poles))]
            # Metal needs every declared buffer bound.  The source chain makes the
            # unused source aliases unreachable; binding D avoids null-pointer ABI.
            slots = poles + (entry.displacement,) * (MAX_POLES - len(poles))
            function(self._tensor_by_host[id(entry.target)],
                     self._tensor_by_host[id(entry.displacement)],
                     self._tensor_by_host[id(entry.inverse_epsilon)],
                     *(self._tensor_by_host[id(value)] for value in slots),
                     int(entry.displacement.size))
            self.launches += 1
        self.runs += 1


def plan_metal_stored_e(fields: Any, pml: Any, residency: Optional[Residency] = None,
                        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
                        functions: Optional[Mapping[Tuple[str, int], Any]] = None,
                        ) -> Optional[MetalStoredEPlan]:
    """Build the real no-PML stored-E plan, or return ``None`` on refusal."""
    if not metal_stored_e_coverage(fields, pml, residency).covered:
        return None
    assert residency is not None
    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: set[Tuple[str, int]] = set()
    for component, displacement_name in E_TERMS:
        target = getattr(fields, component)
        displacement = getattr(fields, displacement_name)
        inverse = fields.inverse_epsilon_for(component)
        for name, host, constant in (
                (component, target, False), (displacement_name, displacement, False),
                (f"inv_eps_{component}:no_pml", inverse, True)):
            tensors[id(host)] = residency.mirror(name, host, constant=constant)
            volumes.append(name)
        states = _poles(fields)[component]
        entries.append(_Entry(component, target, displacement, inverse, states))
        required.update((mode, len(states)) for mode in contract_variants)
    # Register all physical ADE arrays under ADE's stable names.  update_P then
    # obtains the same mirror instead of creating a private stale copy.
    for state_index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for component in tuple(_call(state, "driven", default=()) or ()):
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                tensors[id(host)] = residency.mirror(_physical_name(state_index, label, host), host)
                volumes.append(_physical_name(state_index, label, host))
        scratch = state._scratch
        tensors[id(scratch)] = residency.mirror(_physical_name(state_index, "scratch", scratch), scratch)
        volumes.append(_physical_name(state_index, "scratch", scratch))
    selected = dict(functions or {})
    for key in required:
        # ``dict.setdefault`` evaluates its default eagerly.  Compiling here even
        # when a gate supplied a host function defeats the optional-device boundary
        # (and on MPS makes a host-only unit test launch native compilation).
        if key not in selected:
            selected[key] = compile_stored_e(key[1], key[0])
    return MetalStoredEPlan(entries, residency, volumes, selected, tensors)


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return (metal_stored_e_coverage(context.fields, context.pml, context.residency)
            if slot == "update_E" else Coverage(False, (f"stored E cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalStoredEPlan]:
    return (plan_metal_stored_e(context.fields, context.pml, context.residency,
                                context.contract_variants)
            if slot == "update_E" else None)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, "update_E", "no-PML stored E", _arm_coverage,
                          _arm_plan, prefix="no-PML stored E: ",
                          noun="no-PML stored-E constitutive", wired=True),)


ARMS = register_arms()
