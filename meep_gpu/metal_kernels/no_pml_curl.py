"""Real float32 curl sub-steps for an inactive absorber on PyTorch/MPS.

This is the Metal counterpart of :mod:`meep_gpu.triton_kernels.no_pml`.  It
implements the direct branch of ``stepping._apply_curl``: after the shared Yee
stencil and ownership mask, each target is updated as ``target -= curl``.  There
is no split-field auxiliary and no PML coefficient.

The non-backend predicate is reused from the Triton family through explicit grid
and fields views whose only changed fact is the array-module name.  This avoids a
second transcription of the sixteen numerical clauses while leaving the Metal
backend and mirror-residency requirements visible here.  Production dispatch
remains disabled; registration affects only the experimental Metal composer.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, ELECTRIC_COMPONENTS, _boundary_kinds
from ..triton_kernels.no_pml import (
    SUB_STEPS,
    plain_curl_coverage as shared_plain_curl_coverage,
)
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .plans import KernelPlan

FAMILY = "no_pml_curl"

__all__ = [
    "FAMILY",
    "SUB_STEPS",
    "MetalPlainCurlPlan",
    "compile_plain_curl",
    "metal_plain_curl_coverage",
    "plain_curl_source",
    "plan_metal_plain_curl",
    "plan_metal_plain_curl_from_arrays",
]


_PLAIN_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void no_pml_curl_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device const float* g0      [[buffer(3)]],
    device const float* g1      [[buffer(4)]],
    device const float* g2      [[buffer(5)]],
    device const float* e0      [[buffer(6)]],
    device const float* e1      [[buffer(7)]],
    device const float* e2      [[buffer(8)]],
    constant uint&      nx      [[buffer(9)]],
    constant uint&      ny      [[buffer(10)]],
    constant uint&      nz      [[buffer(11)]],
    constant uint&      n_elem  [[buffer(12)]],
    constant float&     dtdx    [[buffer(13)]],
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

    f0[ii] = f0[ii] - curl0;
    f1[ii] = f1[ii] - curl1;
    f2[ii] = f2[ii] - curl2;
}
"""


def _operand(component: int, index: str, derive: bool) -> str:
    value = f"g{component}[{index}]"
    return f"{value} * e{component}[{index}]" if derive else value


def plain_curl_source(codes: Sequence[int], backward: bool, derive: bool,
                      contract: str = shaders.CONTRACT_OFF) -> str:
    """Specialize the direct curl for direction, storage and three boundaries."""
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    return templates.substitute(_PLAIN_CURL_TEMPLATE, {
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
    })


def compile_plain_curl(codes: Sequence[int], backward: bool, derive: bool,
                       contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(
        plain_curl_source(codes, backward, derive, contract)).no_pml_curl_step


class _CupyName:
    """The single backend fact replaced before calling the shared predicate."""

    __name__ = "cupy"


class _GridView:
    __slots__ = ("_grid", "xp")

    def __init__(self, grid: Any) -> None:
        self._grid = grid
        self.xp = _CupyName()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._grid, name)


class _FieldsView:
    __slots__ = ("_fields", "grid")

    def __init__(self, fields: Any, grid: Any) -> None:
        self._fields = fields
        self.grid = grid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fields, name)


def metal_plain_curl_coverage(fields: Any, pml: Any,
                              sub_step: str = "step_B",
                              residency: Any = None) -> Coverage:
    """Apply Triton's numerical clauses plus Metal backend and residency clauses."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields has no grid",))

    shared_grid = _GridView(grid)
    shared_fields = _FieldsView(fields, shared_grid)
    shared = shared_plain_curl_coverage(shared_fields, pml, sub_step)
    reasons = list(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(shared.reasons)
    # Preserve clause order while preventing a proxy-visible fact and a Metal fact
    # from producing the same sentence twice.
    reasons = list(dict.fromkeys(reasons))
    return Coverage(not reasons, tuple(reasons))


class MetalPlainCurlPlan(KernelPlan):
    """Persistent-mirror plan for one direct curl sub-step."""

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "derive",
                 "bc", "residency", "volumes")

    REPR_FIELDS = ("sub_step", "shape", "bc", "derive")

    def __init__(self, sub_step: str, shape: Sequence[int], dtdx: float,
                 codes: Sequence[int], derive: bool, residency: Residency,
                 targets: Sequence[Any], sources: Sequence[Any],
                 inverse: Sequence[Any], functions: Dict[str, Any],
                 volumes: Sequence[str]) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.derive = bool(derive)
        self.bc = tuple(int(code) for code in codes)
        self.residency = residency
        self.volumes = tuple(volumes)
        super().__init__(
            functions,
            tuple(targets) + tuple(sources) + tuple(inverse)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem, self.dtdx),
        )


def _functions(codes: Sequence[int], backward: bool, derive: bool,
               contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_plain_curl(codes, backward, derive, mode)
            for mode in contract_variants}


def plan_metal_plain_curl(fields: Any, pml: Any, sub_step: str,
                          residency: Optional[Residency] = None,
                          contract_variants: Sequence[str] = (
                              shaders.CONTRACT_OFF,),
                          ) -> Optional[MetalPlainCurlPlan]:
    """Build from engine objects, returning ``None`` for every refused case."""
    if not metal_plain_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    assert residency is not None  # admitted by the predicate above
    grid = fields.grid
    spec = SUB_STEPS[sub_step]
    kinds = _boundary_kinds(grid, None)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    derive = bool(sub_step == "step_B" and not fields.stores_E)
    target_names = tuple(spec["targets"])
    if derive:
        source_names = tuple(spec["displacement"])
        inverse_names = tuple(ELECTRIC_COMPONENTS)
        inverse = [
            residency.mirror(f"inv_eps_{name}:no_pml", fields.inverse_epsilon_for(name),
                             constant=True)
            for name in inverse_names
        ]
    else:
        source_names = tuple(spec["sources"])
        inverse_names = source_names
        inverse = [residency.mirror(name, getattr(fields, name))
                   for name in inverse_names]
    targets = [residency.mirror(name, getattr(fields, name)) for name in target_names]
    sources = [residency.mirror(name, getattr(fields, name)) for name in source_names]
    volumes = target_names + source_names
    return MetalPlainCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, derive, residency,
        targets, sources, inverse,
        _functions(codes, bool(spec["backward"]), derive, contract_variants),
        volumes,
    )


def plan_metal_plain_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], codes: Sequence[int], dtdx: float,
        residency: Residency, derive: bool = False,
        functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        ) -> MetalPlainCurlPlan:
    """Build from a gate-owned array dictionary without running the predicate."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    spec = SUB_STEPS[sub_step]
    target_names = tuple(spec["targets"])
    source_names = (tuple(spec["displacement"])
                    if derive else tuple(spec["sources"]))
    targets = [residency.mirror(name, arrays[name]) for name in target_names]
    sources = [residency.mirror(name, arrays[name]) for name in source_names]
    inverse = ([residency.mirror(f"inv_eps_{name}:no_pml",
                                 arrays["inv_eps_" + name], constant=True)
                for name in ELECTRIC_COMPONENTS]
               if derive else sources)
    shape = arrays[target_names[0]].shape
    selected = (functions if functions is not None else
                _functions(codes, bool(spec["backward"]), derive,
                           contract_variants))
    return MetalPlainCurlPlan(
        sub_step, shape, dtdx, codes, derive, residency, targets, sources,
        inverse, selected, target_names + source_names,
    )


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return metal_plain_curl_coverage(
        context.fields, context.pml, slot, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalPlainCurlPlan]:
    return plan_metal_plain_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants)


def _arm_gate(context: Any) -> bool:
    from ..triton_kernels.launch import absorber_inactive  # noqa: PLC0415

    return absorber_inactive(context.pml)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return tuple(
        arms.register(
            family=FAMILY,
            slot=slot,
            label="no-PML curl",
            coverage=_arm_coverage,
            plan=_arm_plan,
            prefix="no-PML curl: ",
            noun="no-PML direct curl",
            gate=_arm_gate,
            wired=True,
        )
        for slot in SUB_STEPS
    )


ARMS = register_arms()
