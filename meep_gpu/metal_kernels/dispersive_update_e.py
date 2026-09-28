"""Real PML dispersive ``update_E`` for the experimental Metal composer.

For each electric component this transcribes the array path exactly: ordered
``D - P0 - ...``, inverse epsilon, a pre-store ``f_w`` load, then the two distinct
PML accumulations ``(E + kps*src) - kms*prev``. The family is wired only in the
experimental Metal composer after dedicated MPS byte and full-step residency
evidence with :mod:`.ade_update_p`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import (
    COVERED_SUSCEPTIBILITY_KINDS, ELECTRIC_COMPONENTS, Coverage, _call,
    _coefficient_reasons, _inverse_epsilon_reasons, _susceptibility_reasons,
    _volume_reasons,
)
from . import shaders, templates
from .ade_update_p import _physical_name
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source

FAMILY = "dispersive_update_e"
MAX_POLES = 8
E_TERMS = (("Ex", "Dx", 0, "x"), ("Ey", "Dy", 1, "y"), ("Ez", "Dz", 2, "z"))

__all__ = [
    "E_TERMS", "FAMILY", "MAX_POLES", "MetalDispersiveEPlan",
    "dispersive_e_source", "compile_dispersive_e", "metal_dispersive_e_coverage",
    "plan_metal_dispersive_e", "_plan_metal_dispersive_e_covered",
]

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
kernel void dispersive_e_component(
    device float*       e_out  [[buffer(0)]],
    device float*       fw     [[buffer(1)]],
    device const float* d_in   [[buffer(2)]],
    device const float* inv_e  [[buffer(3)]],
    device const float* p0     [[buffer(4)]],
    device const float* p1     [[buffer(5)]],
    device const float* p2     [[buffer(6)]],
    device const float* p3     [[buffer(7)]],
    device const float* p4     [[buffer(8)]],
    device const float* p5     [[buffer(9)]],
    device const float* p6     [[buffer(10)]],
    device const float* p7     [[buffer(11)]],
    device const float* kps    [[buffer(12)]],
    device const float* kms    [[buffer(13)]],
    constant uint& nx           [[buffer(14)]],
    constant uint& ny           [[buffer(15)]],
    constant uint& nz           [[buffer(16)]],
    constant uint& n_elem       [[buffer(17)]],
    uint idx [[thread_position_in_grid]])
{
__BODY__
}
"""


def dispersive_e_source(pole_count: int, axis: int,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    if not 0 <= int(pole_count) <= MAX_POLES:
        raise ValueError(f"pole_count must be in [0, {MAX_POLES}], got {pole_count!r}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    coordinate = ("i", "j", "k")[axis]
    chain = "\n".join(f"    source = source - p{index}[idx];"
                      for index in range(int(pole_count)))
    body = "\n".join((templates.GUARD, templates.DECODE_IJK,
                       "    float prev = fw[idx];", "    float source = d_in[idx];",
                       chain, "    float src = source * inv_e[idx];", "    fw[idx] = src;",
                       "    float value = e_out[idx];",
                       f"    value = value + kps[{coordinate}] * src;",
                       f"    value = value - kms[{coordinate}] * prev;",
                       "    e_out[idx] = value;"))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract), "__BODY__": body,
    })


def compile_dispersive_e(pole_count: int, axis: int,
                         contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(dispersive_e_source(
        pole_count, axis, contract)).dispersive_e_component


def _poles(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    states = tuple(getattr(fields, "polarizations", ()) or ())
    return {component: tuple(state for state in states
                             if _call(state, "drives", component, default=False))
            for component in ELECTRIC_COMPONENTS}


def metal_dispersive_e_coverage(fields: Any, pml: Any,
                                residency: Any = None) -> Coverage:
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons: List[str] = []
    reasons.extend(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    if bool(getattr(fields, "force_complex_fields", False) or _call(grid, "has_bloch", default=False)):
        reasons.append("complex64 storage is not carried by this float32 body")
    if pml is None or not bool(getattr(pml, "is_active", False)):
        reasons.append("an active PML layer is required for the split stored-E update")
    if not bool(getattr(fields, "_pml_active", False)):
        reasons.append("Fields is not in PML storage mode")
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
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if not states:
        reasons.append("no susceptibility is registered; ordinary constitutive owns update_E")
    for component, displacement, _axis, _name in E_TERMS:
        for name in (component, "f_w_" + component, displacement):
            array = getattr(fields, name, None)
            if array is None:
                reasons.append(f"{name} is not allocated")
            else:
                reasons.extend(_volume_reasons(name, array, shape))
    reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and bool(getattr(pml, "is_active", False)):
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), ("_h",)))
    for index, state in enumerate(states):
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
                    reasons.extend(_volume_reasons(f"polarization {index} {label}[{component}]", array, shape))
        scratch = getattr(state, "_scratch", None)
        if scratch is None:
            reasons.append(f"polarization {index} scratch is not allocated")
        else:
            reasons.extend(_volume_reasons(f"polarization {index} scratch", scratch, shape))
    for component, driven in _poles(fields).items():
        if len(driven) > MAX_POLES:
            reasons.append(f"{component} is driven by {len(driven)} poles, more than MAX_POLES={MAX_POLES}")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


@dataclass(frozen=True)
class _Entry:
    component: str
    axis: int
    target: Any
    auxiliary: Any
    displacement: Any
    inverse_epsilon: Any
    kps: Any
    kms: Any
    states: Tuple[Any, ...]


class MetalDispersiveEPlan:
    performs_device_work = True
    replaces_sub_steps = ("update_E",)

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], functions: Mapping[Tuple[str, int, int], Any],
                 tensors: Mapping[int, Any]) -> None:
        self.entries, self.residency = tuple(entries), residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self._functions, self._tensors = dict(functions), dict(tensors)
        self.launches = self.runs = 0
        self.launches_per_run = len(self.entries)

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        mode = shaders.CONTRACT_OFF if contract is None else contract
        for entry in self.entries:
            poles = tuple(state.P[entry.component] for state in entry.states)
            function = self._functions[(mode, len(poles), entry.axis)]
            slots = poles + (entry.displacement,) * (MAX_POLES - len(poles))
            shape = entry.target.shape
            function(self._tensors[id(entry.target)], self._tensors[id(entry.auxiliary)],
                     self._tensors[id(entry.displacement)], self._tensors[id(entry.inverse_epsilon)],
                     *(self._tensors[id(value)] for value in slots),
                     self._tensors[id(entry.kps)], self._tensors[id(entry.kms)],
                     *map(int, shape), int(entry.target.size))
            self.launches += 1
        self.runs += 1


def plan_metal_dispersive_e(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, int, int], Any]] = None,
        ) -> Optional[MetalDispersiveEPlan]:
    """Build the ordinary (unfolded) PML stored-E plan, or refuse it.

    The binding construction intentionally lives in
    :func:`_plan_metal_dispersive_e_covered`.  A mirror fold changes neither this
    pointwise recurrence nor its bindings; it changes the validity predicate that
    establishes the stored extent.  Keeping the builder shared means the folded
    admission cannot acquire a subtly different pole order, PML coefficient
    lattice, or residency-name allocation.
    """
    if not metal_dispersive_e_coverage(fields, pml, residency).covered:
        return None
    return _plan_metal_dispersive_e_covered(
        fields, pml, residency, contract_variants, functions)


def _plan_metal_dispersive_e_covered(
        fields: Any, pml: Any, residency: Residency,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, int, int], Any]] = None,
        ) -> MetalDispersiveEPlan:
    """Bind the pointwise stored-E kernel after a family-specific predicate passed.

    This is deliberately private: only a predicate that proves the exact array
    shapes, real storage, active PML, supported pole state, and shared residency
    may call it.  The folded-dispersive family is the sole second caller, and its
    device gate proves this builder over an actual folded stored extent.
    """
    assert residency is not None and pml is not None
    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: set[Tuple[str, int, int]] = set()
    order = _poles(fields)
    for component, displacement_name, axis, axis_name in E_TERMS:
        target, auxiliary = getattr(fields, component), getattr(fields, "f_w_" + component)
        displacement, inverse = getattr(fields, displacement_name), fields.inverse_epsilon_for(component)
        kps, kms = getattr(pml, f"kps_{axis_name}_h"), getattr(pml, f"kms_{axis_name}_h")
        bindings = ((component, target, False), ("f_w_" + component, auxiliary, False),
                    (displacement_name, displacement, False),
                    (f"inv_eps_{component}", inverse, True),
                    (f"pml:kps_{axis_name}_h", kps, True),
                    (f"pml:kms_{axis_name}_h", kms, True))
        for name, host, constant in bindings:
            tensors[id(host)] = residency.mirror(name, host, constant=constant)
            volumes.append(name)
        states = order[component]
        entries.append(_Entry(component, axis, target, auxiliary, displacement, inverse, kps, kms, states))
        required.update((mode, len(states), axis) for mode in contract_variants)
    for state_index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for component in tuple(_call(state, "driven", default=()) or ()):
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                name = _physical_name(state_index, label, host)
                tensors[id(host)] = residency.mirror(name, host)
                volumes.append(name)
        scratch = state._scratch
        name = _physical_name(state_index, "scratch", scratch)
        tensors[id(scratch)] = residency.mirror(name, scratch)
        volumes.append(name)
    selected = dict(functions or {})
    for key in required:
        if key not in selected:
            selected[key] = compile_dispersive_e(key[1], key[2], key[0])
    return MetalDispersiveEPlan(entries, residency, volumes, selected, tensors)


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return (metal_dispersive_e_coverage(context.fields, context.pml, context.residency)
            if slot == "update_E" else Coverage(False, (f"dispersive E cannot fill {slot}",)))


def _arm_plan(context: Any, slot: str) -> Optional[MetalDispersiveEPlan]:
    return (plan_metal_dispersive_e(context.fields, context.pml, context.residency,
                                    context.contract_variants)
            if slot == "update_E" else None)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, "update_E", "dispersive PML E", _arm_coverage,
                          _arm_plan, prefix="dispersive PML E: ",
                          noun="PML dispersive stored-E constitutive", wired=True),)


ARMS = register_arms()
