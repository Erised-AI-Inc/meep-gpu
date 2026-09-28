"""Real no-PML conductive curl sub-steps for PyTorch/MPS.

This is the Metal counterpart of
:mod:`meep_gpu.triton_kernels.no_pml_conductive`.  It owns the third branch of
``stepping._apply_curl`` when the absorber is inactive and a curl target has a
conductivity::

    field *= condfac
    field -= curl
    field *= condinv

The three assignments are deliberately retained as three assignments in the
shader.  They are the array path's three float32 round points; flattening them
into a single algebraic expression changes numerical results.  Conductivity is
read per target inside ``_apply_curl``.  The compile-time specialization and the
bindings therefore carry three flags, one for each target in ``step_B`` or
``step_D``.  A D-only conductivity still admits the ``step_B`` plan, but all of
its flags are false and it emits the ordinary direct tail for that sub-step.

The native MPS gate now establishes this B/D sub-step pair, including one-sided
conductivity. The family remains unwired pending complete driver-residency
composition evidence.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, ELECTRIC_COMPONENTS, _boundary_kinds
from ..triton_kernels.no_pml import SUB_STEPS
from ..triton_kernels.no_pml_conductive import (
    conductive_no_pml_targets,
    conductive_plain_curl_coverage as shared_conductive_coverage,
)
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .no_pml_curl import _FieldsView, _GridView
from .plans import KernelPlan

FAMILY = "no_pml_conductive"

__all__ = [
    "FAMILY",
    "SUB_STEPS",
    "MetalConductivePlainCurlPlan",
    "compile_conductive_plain_curl",
    "conductive_plain_curl_source",
    "metal_conductive_plain_curl_coverage",
    "plan_metal_conductive_plain_curl",
    "plan_metal_conductive_plain_curl_from_arrays",
]


_CONDUCTIVE_PLAIN_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void no_pml_conductive_curl_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device const float* g0      [[buffer(3)]],
    device const float* g1      [[buffer(4)]],
    device const float* g2      [[buffer(5)]],
    device const float* e0      [[buffer(6)]],
    device const float* e1      [[buffer(7)]],
    device const float* e2      [[buffer(8)]],
    device const float* cf0     [[buffer(9)]],
    device const float* cf1     [[buffer(10)]],
    device const float* cf2     [[buffer(11)]],
    device const float* ci0     [[buffer(12)]],
    device const float* ci1     [[buffer(13)]],
    device const float* ci2     [[buffer(14)]],
    constant uint&      nx      [[buffer(15)]],
    constant uint&      ny      [[buffer(16)]],
    constant uint&      nz      [[buffer(17)]],
    constant uint&      n_elem  [[buffer(18)]],
    constant float&     dtdx    [[buffer(19)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii = int(idx);
    int k = ii % nzi;
    int plane = ii / nzi;
    int j = plane % nyi;
    int i = plane / nyi;

    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    float a   = __A__;
    float b   = __B__;
    float c   = __C__;
    float a_y = vy ? __A_Y__ : 0.0f;
    float a_z = vz ? __A_Z__ : 0.0f;
    float b_x = vx ? __B_X__ : 0.0f;
    float b_z = vz ? __B_Z__ : 0.0f;
    float c_x = vx ? __C_X__ : 0.0f;
    float c_y = vy ? __C_Y__ : 0.0f;

    float curl0 = dtdx * ((c_y - c) + (b - b_z));
    float curl1 = dtdx * ((a_z - a) + (c - c_x));
    float curl2 = dtdx * ((b_x - b) + (a - a_y));

    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

__TAIL0__
__TAIL1__
__TAIL2__
}
"""


def _operand(component: int, index: str, derive: bool) -> str:
    value = f"g{component}[{index}]"
    return f"{value} * e{component}[{index}]" if derive else value


def _tail(index: int, conductive: bool) -> str:
    if not conductive:
        return f"    f{index}[ii] = f{index}[ii] - curl{index};"
    return f"""    float value{index} = f{index}[ii];
    value{index} = value{index} * cf{index}[ii];
    value{index} = value{index} - curl{index};
    value{index} = value{index} * ci{index}[ii];
    f{index}[ii] = value{index};"""


def conductive_plain_curl_source(codes: Sequence[int], backward: bool, derive: bool,
                                 conductive: Sequence[bool],
                                 contract: str = shaders.CONTRACT_OFF) -> str:
    """Specialize the real conductive direct curl and its three target tails."""
    codes = tuple(int(code) for code in codes)
    flags = tuple(bool(flag) for flag in conductive)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if len(flags) != 3:
        raise ValueError("conductive must carry one flag per curl target")
    return templates.substitute(_CONDUCTIVE_PLAIN_CURL_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__MASK__": templates.ownership_mask(codes, backward),
        "__A__": _operand(0, "ii", derive),
        "__B__": _operand(1, "ii", derive),
        "__C__": _operand(2, "ii", derive),
        "__A_Y__": _operand(0, "oy", derive),
        "__A_Z__": _operand(0, "oz", derive),
        "__B_X__": _operand(1, "ox", derive),
        "__B_Z__": _operand(1, "oz", derive),
        "__C_X__": _operand(2, "ox", derive),
        "__C_Y__": _operand(2, "oy", derive),
        "__TAIL0__": _tail(0, flags[0]),
        "__TAIL1__": _tail(1, flags[1]),
        "__TAIL2__": _tail(2, flags[2]),
    })


def compile_conductive_plain_curl(codes: Sequence[int], backward: bool, derive: bool,
                                  conductive: Sequence[bool],
                                  contract: str = shaders.CONTRACT_OFF) -> Any:
    """Compile one fully specialized conductive direct curl shader."""
    return compile_source(conductive_plain_curl_source(
        codes, backward, derive, conductive, contract)).no_pml_conductive_curl_step


def metal_conductive_plain_curl_coverage(fields: Any, pml: Any,
                                         sub_step: str = "step_B",
                                         residency: Any = None) -> Coverage:
    """Numerical conductive clauses plus Metal backend and residency clauses."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields has no grid",))
    shared = shared_conductive_coverage(
        _FieldsView(fields, _GridView(grid)), pml, sub_step)
    reasons = list(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(shared.reasons)
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalConductivePlainCurlPlan(KernelPlan):
    """Persistent-mirror plan for one no-PML conductive curl sub-step."""

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "derive",
                 "bc", "conductive", "residency", "volumes")

    REPR_FIELDS = ("sub_step", "shape", "bc", "derive", "conductive")

    def __init__(self, sub_step: str, shape: Sequence[int], dtdx: float,
                 codes: Sequence[int], derive: bool, conductive: Sequence[bool],
                 residency: Residency, targets: Sequence[Any], sources: Sequence[Any],
                 inverse: Sequence[Any], condfac: Sequence[Any], condinv: Sequence[Any],
                 functions: Dict[str, Any], volumes: Sequence[str]) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.derive = bool(derive)
        self.bc = tuple(int(code) for code in codes)
        self.conductive = tuple(bool(flag) for flag in conductive)
        self.residency = residency
        self.volumes = tuple(volumes)
        super().__init__(
            functions,
            tuple(targets) + tuple(sources) + tuple(inverse)
            + tuple(condfac) + tuple(condinv)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem, self.dtdx),
        )


def _functions(codes: Sequence[int], backward: bool, derive: bool,
               conductive: Sequence[bool],
               contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {
        mode: compile_conductive_plain_curl(
            codes, backward, derive, conductive, mode)
        for mode in contract_variants
    }


def _resolve_sources(fields: Any, sub_step: str, residency: Residency):
    spec = SUB_STEPS[sub_step]
    derive = bool(sub_step == "step_B" and not fields.stores_E)
    if derive:
        source_names = tuple(spec["displacement"])
        inverse = [
            residency.mirror(f"inv_eps_{name}:no_pml_conductive",
                             fields.inverse_epsilon_for(name), constant=True)
            for name in ELECTRIC_COMPONENTS
        ]
    else:
        source_names = tuple(spec["sources"])
        inverse = [residency.mirror(name, getattr(fields, name))
                   for name in source_names]
    return derive, source_names, inverse


def plan_metal_conductive_plain_curl(
        fields: Any, pml: Any, sub_step: str, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Dict[str, Any]] = None,
        ) -> Optional[MetalConductivePlainCurlPlan]:
    """Build an admitted no-PML conductive curl plan from engine objects."""
    if not metal_conductive_plain_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    assert residency is not None
    grid = fields.grid
    spec = SUB_STEPS[sub_step]
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, None))
    target_names = tuple(spec["targets"])
    targets = [residency.mirror(name, getattr(fields, name)) for name in target_names]
    derive, source_names, inverse = _resolve_sources(fields, sub_step, residency)
    sources = [residency.mirror(name, getattr(fields, name)) for name in source_names]
    conductive = conductive_no_pml_targets(fields, sub_step)
    condfac = [
        (residency.mirror(f"condfac_{name}:no_pml_conductive",
                          fields.condfac_for(name), constant=True)
         if conductive[index] else targets[index])
        for index, name in enumerate(target_names)
    ]
    condinv = [
        (residency.mirror(f"condinv_{name}:no_pml_conductive",
                          fields.condinv_for(name), constant=True)
         if conductive[index] else targets[index])
        for index, name in enumerate(target_names)
    ]
    return MetalConductivePlainCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, derive, conductive,
        residency, targets, sources, inverse, condfac, condinv,
        (functions if functions is not None else _functions(
            codes, bool(spec["backward"]), derive, conductive, contract_variants)),
        target_names + source_names,
    )


def plan_metal_conductive_plain_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], codes: Sequence[int], dtdx: float,
        residency: Residency, derive: bool = False,
        conductive: Sequence[bool] = (True, True, True),
        functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        ) -> MetalConductivePlainCurlPlan:
    """Build a gate-owned plan without running coverage."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    flags = tuple(bool(flag) for flag in conductive)
    if len(flags) != 3:
        raise ValueError("conductive must carry one flag per curl target")
    spec = SUB_STEPS[sub_step]
    target_names = tuple(spec["targets"])
    source_names = (tuple(spec["displacement"])
                    if derive else tuple(spec["sources"]))
    targets = [residency.mirror(name, arrays[name]) for name in target_names]
    sources = [residency.mirror(name, arrays[name]) for name in source_names]
    inverse = ([residency.mirror(f"inv_eps_{name}:no_pml_conductive",
                                 arrays["inv_eps_" + name], constant=True)
                for name in ELECTRIC_COMPONENTS]
               if derive else sources)
    condfac = [
        (residency.mirror(f"condfac_{name}:no_pml_conductive",
                          arrays["condfac_" + name], constant=True)
         if flags[index] else targets[index])
        for index, name in enumerate(target_names)
    ]
    condinv = [
        (residency.mirror(f"condinv_{name}:no_pml_conductive",
                          arrays["condinv_" + name], constant=True)
         if flags[index] else targets[index])
        for index, name in enumerate(target_names)
    ]
    selected = (functions if functions is not None else _functions(
        codes, bool(spec["backward"]), derive, flags, contract_variants))
    shape = arrays[target_names[0]].shape
    return MetalConductivePlainCurlPlan(
        sub_step, shape, dtdx, codes, derive, flags, residency,
        targets, sources, inverse, condfac, condinv, selected,
        target_names + source_names,
    )


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return metal_conductive_plain_curl_coverage(
        context.fields, context.pml, slot, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalConductivePlainCurlPlan]:
    return plan_metal_conductive_plain_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants)


def _arm_gate(context: Any) -> bool:
    from ..triton_kernels.launch import absorber_inactive  # noqa: PLC0415

    return absorber_inactive(context.pml)


def register_arms() -> Tuple[Any, ...]:
    """Register conductive direct curls after complete no-PML composition proof."""
    from . import arms  # noqa: PLC0415

    return tuple(
        arms.register(
            family=FAMILY,
            slot=slot,
            label="conductive no-PML curl",
            coverage=_arm_coverage,
            plan=_arm_plan,
            prefix="conductive no-PML curl: ",
            noun="conductive no-PML curl",
            gate=_arm_gate,
            wired=True,
        )
        for slot in SUB_STEPS
    )


ARMS = register_arms()
