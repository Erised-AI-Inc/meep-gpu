"""Lorentz/Drude polarization recurrence on PyTorch/MPS.

This family carries the complete ``update_P`` slot for float32 and complex64
storage, with or without an active split-field absorber.  The arithmetic is the
array path's three-pass, left-associated recurrence::

    ((P*c_now) + (c_prev*P_prev)) + (c_drive*(sigma*W))

``W`` is selected through :meth:`Fields.drive_field`: stored E without PML and
``f_w_E*`` under PML.  Confusing those pointers is especially dangerous because
the values agree outside the absorber and diverge only inside it.

Unlike an ordinary :class:`.plans.KernelPlan`, the arguments cannot be frozen.
``PolarizationState.update`` rotates ``P``, ``P_prev`` and one shared scratch
buffer after *each component*.  This plan therefore mirrors physical arrays once
but resolves their current semantic roles immediately before every launch, then
applies the reference rotation only after that launch succeeds.  Caching semantic
pointers would advance the wrong buffer from launch two onward.

Registration affects only the experimental Metal composer.  Production driver
dispatch remains unchanged.
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
    _volume_reasons,
)
from . import shaders, templates
from .complex_fields import _complex_volume_reasons
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source

FAMILY = "ade_update_p"

__all__ = [
    "FAMILY",
    "MetalAdeUpdatePPlan",
    "ade_source",
    "compile_ade",
    "metal_ade_component_coverage",
    "metal_ade_update_p_coverage",
    "plan_metal_ade_update_p",
]


_ADE_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

kernel void ade_update_p_step(
    device __VALUE_TYPE__*       p_out    [[buffer(0)]],
    device const __VALUE_TYPE__* p_now    [[buffer(1)]],
    device const __VALUE_TYPE__* p_prev   [[buffer(2)]],
__SIGMA_DECL__
    device const __VALUE_TYPE__* drive    [[buffer(__DRIVE_BUFFER__)]],
__SIGMA_SCALAR_DECL__
    constant float&              c_now    [[buffer(__CNOW_BUFFER__)]],
    constant float&              c_prev   [[buffer(__CPREV_BUFFER__)]],
    constant float&              c_drive  [[buffer(__CDRIVE_BUFFER__)]],
    constant uint&               n_elem   [[buffer(__NELEM_BUFFER__)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
__BODY__
}
"""


def ade_source(dtype: str, sigma_is_volume: bool,
               contract: str = shaders.CONTRACT_OFF) -> str:
    """Specialize storage width, sigma binding and contraction policy."""
    if dtype not in ("float32", "complex64"):
        raise ValueError(f"dtype must be 'float32' or 'complex64', got {dtype!r}")
    if sigma_is_volume:
        sigma_decl = "    device const float* sigma    [[buffer(3)]],"
        sigma_scalar_decl = ""
        drive_buffer, c_now, c_prev, c_drive, n_elem = 4, 5, 6, 7, 8
        sigma_value = "sigma[idx]"
    else:
        sigma_decl = ""
        sigma_scalar_decl = (
            "    constant float& sigma    [[buffer(4)]],")
        drive_buffer, c_now, c_prev, c_drive, n_elem = 3, 5, 6, 7, 8
        sigma_value = "sigma"
    value_type = "float2" if dtype == "complex64" else "float"
    if dtype == "complex64":
        helpers = templates.complex_helpers("FMA_V1")
        body = """    float2 p = p_now[idx];
    float2 q = p_prev[idx];
    float2 w = drive[idx];
    float s = __SIGMA_VALUE__;
    float2 a = c_mul_field_left(p, c_now);
    float2 b = c_mul_coefficient_left(c_prev, q);
    float2 sw = c_mul_coefficient_left(s, w);
    float2 d = c_mul_coefficient_left(c_drive, sw);
    p_out[idx] = (a + b) + d;"""
    else:
        helpers = ""
        body = """    float p = p_now[idx];
    float q = p_prev[idx];
    float w = drive[idx];
    float s = __SIGMA_VALUE__;
    p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));"""
    body = body.replace("__SIGMA_VALUE__", sigma_value)
    return templates.substitute(_ADE_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": helpers,
        "__VALUE_TYPE__": value_type,
        "__SIGMA_DECL__": sigma_decl,
        "__DRIVE_BUFFER__": str(drive_buffer),
        "__SIGMA_SCALAR_DECL__": sigma_scalar_decl,
        "__CNOW_BUFFER__": str(c_now),
        "__CPREV_BUFFER__": str(c_prev),
        "__CDRIVE_BUFFER__": str(c_drive),
        "__NELEM_BUFFER__": str(n_elem),
        "__BODY__": body,
    })


def compile_ade(dtype: str, sigma_is_volume: bool,
                contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(
        ade_source(dtype, sigma_is_volume, contract)).ade_update_p_step


def _storage_dtype(fields: Any) -> str:
    grid = getattr(fields, "grid", None)
    complex_storage = bool(getattr(fields, "force_complex_fields", False)
                           or getattr(grid, "has_bloch", False))
    return "complex64" if complex_storage else "float32"


def _drive_reasons(fields: Any, pml: Any, component: str) -> List[str]:
    reasons: List[str] = []
    active = bool(pml is not None and getattr(pml, "is_active", False))
    if active and not bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "an active PML layer is installed but Fields is not in PML storage "
            "mode; drive_field would not select f_w")
    if not active and bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "the layer is inactive but Fields remains in PML storage mode; "
            "drive_field would select a stale f_w array")
    if not active and not bool(getattr(fields, "stores_E", False)):
        reasons.append(
            "E is not stored: a live polarization requires update_E to refresh "
            "the no-PML drive before update_P")

    expected_name = ("f_w_" if active else "") + component
    expected = getattr(fields, expected_name, None)
    if expected is None:
        reasons.append(f"expected drive {expected_name} is not allocated")
    reader = getattr(fields, "drive_field", None)
    if not callable(reader):
        reasons.append("fields does not expose drive_field()")
        return reasons
    try:
        actual = reader(component)
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        reasons.append(f"drive_field({component!r}) raised {exc!r}")
        return reasons
    if actual is not expected:
        reasons.append(
            f"drive_field({component!r}) is not the expected {expected_name} "
            "array for this absorber state")
    return reasons


def _coefficient_reasons(state: Any) -> List[str]:
    reasons: List[str] = []
    coefficients = getattr(state, "_coefficients", None)
    try:
        values = tuple(coefficients)
    except Exception as exc:  # noqa: BLE001 - malformed is a refusal
        return [f"the coefficient triple is unreadable ({exc!r})"]
    if len(values) != 3:
        return ["the (c_now, c_prev, c_drive) coefficient triple is missing"]
    for name, value in zip(("c_now", "c_prev", "c_drive"), values):
        try:
            number = float(value)
        except Exception:  # noqa: BLE001 - non-numeric is a refusal
            reasons.append(f"{name}={value!r} is not a float")
            continue
        if not np.isfinite(number):
            reasons.append(f"{name}={number!r} is not finite")
    return reasons


def metal_ade_component_coverage(fields: Any, pml: Any, state: Any,
                                 component: str) -> Coverage:
    """Backend-independent numerical clauses for one pole/component."""
    reasons: List[str] = []
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    if component not in ELECTRIC_COMPONENTS:
        return Coverage(False, (f"component {component!r} is outside "
                                f"{ELECTRIC_COMPONENTS}",))
    if not _call(state, "drives", component, default=False):
        reasons.append(f"this susceptibility does not drive {component}")
    kind = getattr(getattr(state, "susceptibility", None), "kind", None)
    if kind not in COVERED_SUSCEPTIBILITY_KINDS:
        reasons.append(
            f"kind {kind!r} is outside {COVERED_SUSCEPTIBILITY_KINDS}")
    reasons.extend(_coefficient_reasons(state))

    dt = getattr(grid, "dt", None)
    if dt is None:
        reasons.append("grid carries no dt")
    elif getattr(getattr(state, "grid", None), "dt", dt) != dt:
        reasons.append("the state's coefficients were built for a different dt")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return Coverage(False, tuple(reasons))
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if total >= 2 ** 31:
        reasons.append(f"{total} cells exceeds the kernel's int32 index range")

    dtype = _storage_dtype(fields)
    volume_reasons = (_complex_volume_reasons if dtype == "complex64"
                      else _volume_reasons)
    buffers = {
        "P": (getattr(state, "P", {}) or {}).get(component),
        "P_prev": (getattr(state, "P_prev", {}) or {}).get(component),
        "_scratch": getattr(state, "_scratch", None),
    }
    for name, array in buffers.items():
        if array is None:
            reasons.append(f"{name}[{component}] is not allocated")
        else:
            reasons.extend(volume_reasons(
                f"{name}[{component}]", array, shape))

    reasons.extend(_drive_reasons(fields, pml, component))
    reader = getattr(fields, "drive_field", None)
    drive = None
    if callable(reader):
        try:
            drive = reader(component)
        except Exception:  # already named above
            pass
    if drive is not None:
        reasons.extend(volume_reasons(
            f"drive_field({component!r})", drive, shape))

    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    if sigma is None:
        reasons.append(f"sigma[{component}] is missing")
    elif getattr(sigma, "shape", ()):
        reasons.extend(_volume_reasons(f"sigma[{component}]", sigma, shape))
    else:
        try:
            number = float(sigma)
        except Exception:  # noqa: BLE001 - neither scalar nor volume
            reasons.append(
                f"sigma[{component}]={sigma!r} is neither a scalar nor a volume")
        else:
            if not np.isfinite(number):
                reasons.append(f"sigma[{component}]={number!r} is not finite")
    return Coverage(not reasons, tuple(reasons))


def _states(fields: Any) -> Tuple[Any, ...]:
    try:
        return tuple(getattr(fields, "polarizations", ()) or ())
    except Exception:  # unreadable is handled by the aggregate predicate
        return ()


def metal_ade_update_p_coverage(fields: Any, pml: Any,
                                residency: Any = None) -> Coverage:
    """All-or-nothing verdict for the complete ``update_P`` sub-step."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons = list(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    try:
        states = tuple(getattr(fields, "polarizations", ()) or ())
    except Exception as exc:  # noqa: BLE001 - unreadable is not empty
        reasons.append(f"the polarization list is unreadable ({exc!r})")
        return Coverage(False, tuple(dict.fromkeys(reasons)))
    driven_total = 0
    for index, state in enumerate(states):
        driven = getattr(state, "driven", None)
        try:
            components = tuple(driven()) if callable(driven) else ()
        except Exception as exc:  # noqa: BLE001 - malformed is a refusal
            reasons.append(f"polarization {index}: driven() raised {exc!r}")
            continue
        driven_total += len(components)
        for component in components:
            verdict = metal_ade_component_coverage(
                fields, pml, state, component)
            reasons.extend(
                f"polarization {index} {component}: {reason}"
                for reason in verdict.reasons)
    if driven_total == 0:
        reasons.append("no driven component exists; update_P is a no-op")
    reasons = list(dict.fromkeys(reasons))
    return Coverage(not reasons, tuple(reasons))


@dataclass(frozen=True)
class _Entry:
    state: Any
    component: str
    drive: Any
    sigma: Any
    sigma_is_volume: bool
    dtype: str


class MetalAdeUpdatePPlan:
    """Persistent physical mirrors plus live semantic pointer resolution."""

    __slots__ = (
        "entries", "residency", "volumes", "launches", "runs",
        "launches_per_run", "_functions", "_tensor_by_host",
    )

    performs_device_work = True
    replaces_sub_steps = ("update_P",)

    def __init__(self, entries: Sequence[_Entry], residency: Any,
                 volumes: Sequence[str], functions: Mapping[Tuple[str, bool, str], Any],
                 tensor_by_host: Mapping[int, Any]) -> None:
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.launches = 0
        self.runs = 0
        self.launches_per_run = len(self.entries)
        self._functions = dict(functions)
        self._tensor_by_host = dict(tensor_by_host)

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        mode = shaders.CONTRACT_OFF if contract is None else contract
        for entry in self.entries:
            state, component = entry.state, entry.component
            p = state.P[component]
            p_prev = state.P_prev[component]
            scratch = state._scratch
            function = self._functions.get(
                (mode, entry.sigma_is_volume, entry.dtype))
            if function is None:
                raise KeyError(
                    f"this ADE plan holds no {(mode, entry.sigma_is_volume, entry.dtype)!r} "
                    "variant")
            out_tensor = self._tensor_by_host[id(scratch)]
            now_tensor = self._tensor_by_host[id(p)]
            prev_tensor = self._tensor_by_host[id(p_prev)]
            drive_tensor = self._tensor_by_host[id(entry.drive)]
            c_now, c_prev, c_drive = state._coefficients
            common = (
                float(c_now), float(c_prev), float(c_drive), int(p.size))
            if entry.sigma_is_volume:
                function(
                    out_tensor, now_tensor, prev_tensor,
                    self._tensor_by_host[id(entry.sigma)], drive_tensor, *common)
            else:
                function(
                    out_tensor, now_tensor, prev_tensor, drive_tensor,
                    float(entry.sigma), *common)
            self.launches += 1
            state.P[component] = scratch
            state.P_prev[component] = p
            state._scratch = p_prev
        self.runs += 1

    def __repr__(self) -> str:
        return (f"MetalAdeUpdatePPlan(entries={len(self.entries)}, "
                f"variants={self.variants!r})")


def _physical_name(state_index: int, label: str, array: Any) -> str:
    return f"ade:{state_index}:{label}:{id(array):x}"


def plan_metal_ade_update_p(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, bool, str], Any]] = None,
        ) -> Optional[MetalAdeUpdatePPlan]:
    """Build the complete all-pole plan, returning ``None`` on any refusal."""
    if not metal_ade_update_p_coverage(fields, pml, residency).covered:
        return None
    assert residency is not None
    states = _states(fields)
    entries: List[_Entry] = []
    tensor_by_host: Dict[int, Any] = {}
    volumes: List[str] = []
    required: set[Tuple[str, bool, str]] = set()

    for state_index, state in enumerate(states):
        components = tuple(state.driven())
        physical: List[Tuple[str, Any]] = []
        for component in components:
            physical.extend((
                (f"P:{component}", state.P[component]),
                (f"P_prev:{component}", state.P_prev[component]),
            ))
        physical.append(("scratch", state._scratch))
        for label, host in physical:
            if id(host) in tensor_by_host:
                continue
            dtype = np.complex64 if str(host.dtype) == "complex64" else np.float32
            name = _physical_name(state_index, label, host)
            tensor_by_host[id(host)] = residency.mirror(name, host, dtype=dtype)
            volumes.append(name)

        dtype = _storage_dtype(fields)
        for component in components:
            drive = fields.drive_field(component)
            drive_name = ("f_w_" if (pml is not None and pml.is_active) else "") + component
            tensor_by_host[id(drive)] = residency.mirror(
                drive_name, drive,
                dtype=np.complex64 if dtype == "complex64" else np.float32)
            volumes.append(drive_name)
            sigma = state.sigma[component]
            volume = bool(getattr(sigma, "shape", ()))
            if volume:
                sigma_name = _physical_name(state_index, f"sigma:{component}", sigma)
                tensor_by_host[id(sigma)] = residency.mirror(
                    sigma_name, sigma, constant=True, dtype=np.float32)
                volumes.append(sigma_name)
            entries.append(_Entry(state, component, drive, sigma, volume, dtype))
            for mode in contract_variants:
                required.add((mode, volume, dtype))

    selected = dict(functions or {})
    for mode, volume, dtype in sorted(required):
        if (mode, volume, dtype) not in selected:
            selected[(mode, volume, dtype)] = compile_ade(dtype, volume, mode)
    return MetalAdeUpdatePPlan(
        entries, residency, volumes, selected, tensor_by_host)


def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != "update_P":
        return Coverage(False, (f"ADE cannot fill slot {slot!r}",))
    return metal_ade_update_p_coverage(
        context.fields, context.pml, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalAdeUpdatePPlan]:
    if slot != "update_P":
        return None
    return plan_metal_ade_update_p(
        context.fields, context.pml, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return (arms.register(
        family=FAMILY,
        slot="update_P",
        label="ADE update_P",
        coverage=_arm_coverage,
        plan=_arm_plan,
        prefix="ADE update_P: ",
        noun="Lorentz/Drude polarization recurrence",
        wired=True,
    ),)


ARMS = register_arms()
