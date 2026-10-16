"""Lorentz/Drude polarization recurrence on PyTorch/MPS.

This family carries the complete ``update_P`` slot for float32 and complex64
storage, with or without an active split-field absorber.  The arithmetic is the
array path's, which is MEEP's isotropic update (``PolarizationState.update``,
susceptibility.cpp:251-256) on MEEP's four float32 constants
(``PolarizationState.constants``, ``Susceptibility.realnum_constants``)::

    gamma1inv * (((two_minus_od * P) - (gamma1 * P_prev)) + (omega0dtsqr * (sigma * W)))

every operation one float32 operation, contraction off. Complex storage steps the
real update on each stored part, MEEP's ``DOCMP2`` two real passes: every product is
a real constant (or the real sigma) times one part, and the sums are per part.

THE DRIVE IS SELECTED PER CHUNK, ON THE DEVICE. ``W`` is MEEP's
``w = f_w ? f_w : f`` (update_pols.cpp:44, :meth:`Fields.drive_field`): the stored E
without an absorber, and under one ``f_w_E*`` on the cells whose chunk carries the
component's own (``dsigw``) layer and the stored E everywhere else. The kernel binds
both arrays and the chunk record and reads ``w = dw ? fw[idx] : drive[idx]``, with
``dw`` the component's own-axis membership on the half-integer lattice -- the line
the base constitutive kernel selects its form on (``shaders.constitutive_bounds_block
("E")``), so the drive and the field ``update_E`` stored are chosen by one predicate.
On a plain cell the stored E IS the constitutive product (``update_E`` writes ``E =
src`` there and leaves ``f_w`` untouched), which is why the stored word is the drive.
The plan therefore never asks ``Fields.drive_field`` for a pointer: a component split
between the two forms composes a pooled host buffer there, which no device plan can
hold.

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
from . import chunk_bounds as _chunk_bounds
from . import shaders, templates
from .complex_fields import _complex_volume_reasons
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source

FAMILY = "ade_update_p"

#: MEEP's four ``update_P`` constants, in ``dispersion.RealnumConstants`` order: the
#: names the kernel binds them under and the order the plan passes them in.
CONSTANT_NAMES: Tuple[str, str, str, str] = (
    "two_minus_od", "gamma1", "omega0dtsqr", "gamma1inv")

#: Each electric component's own axis: the axis whose half-integer membership line
#: chooses the drive under an absorber (``stepping.E_CONSTITUTIVE_TERMS``).
COMPONENT_AXIS: Dict[str, int] = {"Ex": 0, "Ey": 1, "Ez": 2}

__all__ = [
    "COMPONENT_AXIS",
    "CONSTANT_NAMES",
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

__RECORD_STRUCT__
kernel void ade_update_p_step(
    device __VALUE_TYPE__*       p_out        [[buffer(0)]],
    device const __VALUE_TYPE__* p_now        [[buffer(1)]],
    device const __VALUE_TYPE__* p_prev       [[buffer(2)]],
__SIGMA_DECL__
    device const __VALUE_TYPE__* drive        [[buffer(__DRIVE_BUFFER__)]],
__SIGMA_SCALAR_DECL__
    constant float&              two_minus_od [[buffer(5)]],
    constant float&              gamma1       [[buffer(6)]],
    constant float&              omega0dtsqr  [[buffer(7)]],
    constant float&              gamma1inv    [[buffer(8)]],
    constant uint&               n_elem       [[buffer(9)]],
__SELECT_DECL__
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
__BODY__
}
"""

#: The drive-select bindings, after ``n_elem``: the auxiliary drive, the two extents
#: the linear index decodes against, and the chunk record (last).
_SELECT_DECL = """    device const __VALUE_TYPE__* fw           [[buffer(10)]],
    constant uint&               ny           [[buffer(11)]],
    constant uint&               nz           [[buffer(12)]],
    __RECORD_BINDING__,"""

#: The chunk record's binding index under the drive select: last, after ``nz``.
RECORD_BUFFER = 13


def _drive_lines(value_type: str, select_axis: Optional[int]) -> List[str]:
    """The drive read: the stored E, or under an absorber MEEP's per-chunk select."""
    if select_axis is None:
        return [f"    {value_type} w = drive[idx];"]
    return [
        templates.DECODE_IJK,
        "",
        "    // --- the chunk record: this component's own-axis membership -----------",
        _chunk_bounds.prologue(((True, int(select_axis)),)),
        "",
        "    // MEEP's w = f_w ? f_w : f, per chunk (update_pols.cpp:44): f_w where the",
        "    // component's own (dsigw) chunk keeps one, the stored E elsewhere.",
        f"    const bool dw = {_chunk_bounds.membership_name(True, int(select_axis))};",
        f"    {value_type} w = dw ? fw[idx] : drive[idx];",
    ]


def ade_source(dtype: str, sigma_is_volume: bool,
               contract: str = shaders.CONTRACT_OFF,
               select_axis: Optional[int] = None) -> str:
    """Specialize storage width, sigma binding, contraction policy and the drive.

    ``select_axis`` is ``None`` for a run without an absorber, where the drive is the
    stored E, and the component's own axis (0, 1, 2) under one, where the drive is
    selected per chunk on that axis's half-integer membership line.
    """
    if dtype not in ("float32", "complex64"):
        raise ValueError(f"dtype must be 'float32' or 'complex64', got {dtype!r}")
    if select_axis is not None and select_axis not in (0, 1, 2):
        raise ValueError(f"select_axis must be None, 0, 1 or 2, got {select_axis!r}")
    if sigma_is_volume:
        sigma_decl = "    device const float*          sigma        [[buffer(3)]],"
        sigma_scalar_decl = ""
        drive_buffer = 4
        sigma_value = "sigma[idx]"
    else:
        sigma_decl = ""
        sigma_scalar_decl = (
            "    constant float&              sigma        [[buffer(4)]],")
        drive_buffer = 3
        sigma_value = "sigma"
    value_type = "float2" if dtype == "complex64" else "float"
    head = [f"    {value_type} p = p_now[idx];",
            f"    {value_type} q = p_prev[idx];",
            *_drive_lines(value_type, select_axis),
            f"    float s = {sigma_value};"]
    if dtype == "complex64":
        # MEEP's DOCMP2: the real update on each stored part. A float times a float2
        # is one rounded float32 product per part, and float2 sums are per part.
        tail = [
            "    // MEEP's DOCMP2 (susceptibility.cpp:251-256): the real update on each",
            "    // stored part -- every product a real word times one part.",
            "    float2 a = two_minus_od * p;",
            "    float2 b = gamma1 * q;",
            "    float2 sw = s * w;",
            "    float2 d = omega0dtsqr * sw;",
            "    p_out[idx] = gamma1inv * ((a - b) + d);",
        ]
    else:
        tail = [
            "    p_out[idx] = gamma1inv * (((two_minus_od * p) - (gamma1 * q)) + "
            "(omega0dtsqr * (s * w)));",
        ]
    select = select_axis is not None
    return templates.substitute(_ADE_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__RECORD_STRUCT__": (_chunk_bounds.struct_source() + "\n") if select else "",
        "__VALUE_TYPE__": value_type,
        "__SIGMA_DECL__": sigma_decl,
        "__DRIVE_BUFFER__": str(drive_buffer),
        "__SIGMA_SCALAR_DECL__": sigma_scalar_decl,
        "__SELECT_DECL__": (templates.substitute(_SELECT_DECL, {
            "__VALUE_TYPE__": value_type,
            "__RECORD_BINDING__": _chunk_bounds.binding(RECORD_BUFFER),
        }) if select else ""),
        "__BODY__": "\n".join(head + tail),
    })


def compile_ade(dtype: str, sigma_is_volume: bool,
                contract: str = shaders.CONTRACT_OFF,
                select_axis: Optional[int] = None) -> Any:
    return compile_source(
        ade_source(dtype, sigma_is_volume, contract, select_axis)).ade_update_p_step


def _storage_dtype(fields: Any) -> str:
    grid = getattr(fields, "grid", None)
    complex_storage = bool(getattr(fields, "force_complex_fields", False)
                           or getattr(grid, "has_bloch", False))
    return "complex64" if complex_storage else "float32"


def _layer_active(pml: Any) -> bool:
    return bool(pml is not None and getattr(pml, "is_active", False))


def _drive_reasons(fields: Any, pml: Any, component: str) -> List[str]:
    """The arrays the drive reads, and the storage state that makes them MEEP's ``w``.

    Under an absorber the kernel reads ``f_w_<c>`` on the component's own-layer cells
    and the stored ``<c>`` elsewhere; without one it reads the stored ``<c>``. It does
    not consult :meth:`Fields.drive_field`: that call composes a pooled host buffer for
    a component split between the two forms, which is exactly what the select reads in
    place.
    """
    reasons: List[str] = []
    active = _layer_active(pml)
    if active and not bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "an active PML layer is installed but Fields is not in PML storage "
            "mode; the drive select would read no f_w")
    if not active and bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "the layer is inactive but Fields remains in PML storage mode; "
            "drive_field would select a stale f_w array")
    if not active and not bool(getattr(fields, "stores_E", False)):
        reasons.append(
            "E is not stored: a live polarization requires update_E to refresh "
            "the no-PML drive before update_P")
    names = (component,) + (("f_w_" + component,) if active else ())
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"drive array {name} is not allocated")
    return reasons


def _constants_reasons(state: Any) -> List[str]:
    """MEEP's four float32 constants (``PolarizationState.constants``), each a finite word."""
    reasons: List[str] = []
    constants = getattr(state, "constants", None)
    try:
        values = tuple(constants)
    except Exception as exc:  # noqa: BLE001 - malformed is a refusal
        return [f"the realnum constants are unreadable ({exc!r})"]
    if len(values) != len(CONSTANT_NAMES):
        return [f"the four realnum constants {CONSTANT_NAMES} are missing"]
    for name, value in zip(CONSTANT_NAMES, values):
        try:
            number = float(value)
        except Exception:  # noqa: BLE001 - non-numeric is a refusal
            reasons.append(f"{name}={value!r} is not a float")
            continue
        if not np.isfinite(number):
            reasons.append(f"{name}={number!r} is not finite")
        elif float(np.float32(number)) != number:
            reasons.append(
                f"{name}={number!r} is not a float32 word; the kernel binds MEEP's "
                f"realnum constant and would round it")
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
    reasons.extend(_constants_reasons(state))

    dt = getattr(grid, "dt", None)
    if dt is None:
        reasons.append("grid carries no dt")
    elif getattr(getattr(state, "grid", None), "dt", dt) != dt:
        reasons.append("the state's constants were built for a different dt")

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
    names = (component,) + (("f_w_" + component,) if _layer_active(pml) else ())
    for name in names:
        array = getattr(fields, name, None)
        if array is not None:
            reasons.extend(volume_reasons(f"drive array {name}", array, shape))

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
    drive: Any               # the stored E component
    auxiliary: Any           # f_w_<component> under an absorber, else None
    sigma: Any
    sigma_is_volume: bool
    dtype: str
    select_axis: Optional[int]


class MetalAdeUpdatePPlan:
    """Persistent physical mirrors plus live semantic pointer resolution.

    Under an absorber the chunk record binds last (:data:`RECORD_BUFFER`), checked
    against the plan's shape by :func:`.chunk_bounds.plan_words`.
    """

    __slots__ = (
        "entries", "residency", "volumes", "launches", "runs",
        "launches_per_run", "chunk_bounds", "shape", "_functions",
        "_tensor_by_host", "_record_tensor",
    )

    performs_device_work = True
    replaces_sub_steps = ("update_P",)

    def __init__(self, entries: Sequence[_Entry], residency: Any,
                 volumes: Sequence[str],
                 functions: Mapping[Tuple[str, bool, str, Optional[int]], Any],
                 tensor_by_host: Mapping[int, Any], shape: Sequence[int],
                 chunk_bounds: Optional[np.ndarray] = None,
                 record_tensor: Any = None) -> None:
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.launches = 0
        self.runs = 0
        self.launches_per_run = len(self.entries)
        self.shape = tuple(int(n) for n in shape)
        self.chunk_bounds = chunk_bounds
        self._functions = dict(functions)
        self._tensor_by_host = dict(tensor_by_host)
        self._record_tensor = record_tensor

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
            key = (mode, entry.sigma_is_volume, entry.dtype, entry.select_axis)
            function = self._functions.get(key)
            if function is None:
                raise KeyError(f"this ADE plan holds no {key!r} variant")
            out_tensor = self._tensor_by_host[id(scratch)]
            now_tensor = self._tensor_by_host[id(p)]
            prev_tensor = self._tensor_by_host[id(p_prev)]
            drive_tensor = self._tensor_by_host[id(entry.drive)]
            constants = tuple(float(value) for value in state.constants)
            common = (*constants, int(p.size))
            select: Tuple[Any, ...] = ()
            if entry.select_axis is not None:
                select = (self._tensor_by_host[id(entry.auxiliary)],
                          int(self.shape[1]), int(self.shape[2]),
                          self._record_tensor)
            if entry.sigma_is_volume:
                function(
                    out_tensor, now_tensor, prev_tensor,
                    self._tensor_by_host[id(entry.sigma)], drive_tensor,
                    *common, *select)
            else:
                function(
                    out_tensor, now_tensor, prev_tensor, drive_tensor,
                    float(entry.sigma), *common, *select)
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
        functions: Optional[Mapping[Tuple[str, bool, str, Optional[int]], Any]] = None,
        *, chunk_bounds: Any,
        ) -> Optional[MetalAdeUpdatePPlan]:
    """Build the complete all-pole plan, returning ``None`` on any refusal.

    ``chunk_bounds`` is the record of the division the run steps with (the composer's
    ``StepContext.chunk_bounds``) and has no default. Under an absorber it is checked
    against the grid by name (:func:`.chunk_bounds.plan_words`) and bound, because the
    drive is selected per chunk; without one the drive is the stored E and the record
    is not bound.
    """
    if not metal_ade_update_p_coverage(fields, pml, residency).covered:
        return None
    assert residency is not None
    active = _layer_active(pml)
    shape = tuple(int(n) for n in fields.grid.shape)
    record = (_chunk_bounds.plan_words(chunk_bounds, shape, "plan_metal_ade_update_p")
              if active else None)
    states = _states(fields)
    entries: List[_Entry] = []
    tensor_by_host: Dict[int, Any] = {}
    volumes: List[str] = []
    required: set[Tuple[str, bool, str, Optional[int]]] = set()
    dtype = _storage_dtype(fields)
    mirror_dtype = np.complex64 if dtype == "complex64" else np.float32

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
            host_dtype = np.complex64 if str(host.dtype) == "complex64" else np.float32
            name = _physical_name(state_index, label, host)
            tensor_by_host[id(host)] = residency.mirror(name, host, dtype=host_dtype)
            volumes.append(name)

        for component in components:
            # The SAME residency names the constitutive plans register, so a
            # composition that holds both binds one tensor per array.
            drive = getattr(fields, component)
            tensor_by_host[id(drive)] = residency.mirror(
                component, drive, dtype=mirror_dtype)
            volumes.append(component)
            auxiliary = None
            select_axis = None
            if active:
                auxiliary = getattr(fields, "f_w_" + component)
                tensor_by_host[id(auxiliary)] = residency.mirror(
                    "f_w_" + component, auxiliary, dtype=mirror_dtype)
                volumes.append("f_w_" + component)
                select_axis = COMPONENT_AXIS[component]
            sigma = state.sigma[component]
            volume = bool(getattr(sigma, "shape", ()))
            if volume:
                sigma_name = _physical_name(state_index, f"sigma:{component}", sigma)
                tensor_by_host[id(sigma)] = residency.mirror(
                    sigma_name, sigma, constant=True, dtype=np.float32)
                volumes.append(sigma_name)
            entries.append(_Entry(state, component, drive, auxiliary, sigma, volume,
                                  dtype, select_axis))
            for mode in contract_variants:
                required.add((mode, volume, dtype, select_axis))

    selected = dict(functions or {})
    for key in sorted(required, key=repr):
        if key not in selected:
            mode, volume, key_dtype, select_axis = key
            selected[key] = compile_ade(key_dtype, volume, mode, select_axis)
    record_tensor = (_chunk_bounds.tensor(record, str(getattr(residency, "device", "mps")))
                     if record is not None else None)
    return MetalAdeUpdatePPlan(
        entries, residency, volumes, selected, tensor_by_host, shape,
        chunk_bounds=record, record_tensor=record_tensor)


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
        context.contract_variants, chunk_bounds=context.chunk_bounds)


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
