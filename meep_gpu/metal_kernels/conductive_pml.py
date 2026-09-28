"""Real conductive split-field PML curls for PyTorch/MPS.

This family implements the active-PML branch of ``stepping._apply_curl`` for a
conductive target.  Unlike no-PML conductivity it carries the ``f_cond`` history
and selects one of MEEP's four transverse-PML cases per cell.  The branch masks
use exact ``!= 1.0f`` comparisons, matching the array path; approximate tests
would partition near-unity PML coefficients differently.

The direct native-MPS gate verifies every branch, history-preservation case, and
mutation.  The whole-step gate separately proves the two mixed B/H/D/E
compositions in which one conductive PML curl shares persistent mirrors with the
ordinary PML curl and ordinary constitutive pair; the family is therefore wired
into the experimental composer.  This is still not production driver dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from ..triton_kernels.conductivity import (
    CONDUCTIVE_SUB_STEPS,
    conductive_pml_curl_coverage as shared_conductive_pml_coverage,
    conductive_targets,
)
from ..triton_kernels.coverage import Coverage, _boundary_kinds
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .launch import SUB_STEPS
from .no_pml_curl import _FieldsView, _GridView
from .plans import KernelPlan

FAMILY = "conductive_pml"

__all__ = [
    "FAMILY",
    "MetalConductivePmlCurlPlan",
    "compile_conductive_pml_curl",
    "conductive_pml_curl_source",
    "metal_conductive_pml_curl_coverage",
    "plan_metal_conductive_pml_curl",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__
kernel void conductive_pml_curl_step(
    device float*       f0 [[buffer(0)]], device float*       f1 [[buffer(1)]], device float*       f2 [[buffer(2)]],
    device float*       u0 [[buffer(3)]], device float*       u1 [[buffer(4)]], device float*       u2 [[buffer(5)]],
    device float*       c0 [[buffer(6)]], device float*       c1 [[buffer(7)]], device float*       c2 [[buffer(8)]],
    device const float* cf0 [[buffer(9)]], device const float* cf1 [[buffer(10)]], device const float* cf2 [[buffer(11)]],
    device const float* ci0 [[buffer(12)]], device const float* ci1 [[buffer(13)]], device const float* ci2 [[buffer(14)]],
    device const float* g0 [[buffer(15)]], device const float* g1 [[buffer(16)]], device const float* g2 [[buffer(17)]],
    device const float* kmx [[buffer(18)]], device const float* sinvx [[buffer(19)]],
    device const float* kmy [[buffer(20)]], device const float* sinvy [[buffer(21)]],
    device const float* kmz [[buffer(22)]], device const float* sinvz [[buffer(23)]],
    constant uint& nx [[buffer(24)]], constant uint& ny [[buffer(25)]], constant uint& nz [[buffer(26)]],
    constant uint& n_elem [[buffer(27)]], constant float& dtdx [[buffer(28)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii = int(idx), k = ii % nzi, plane = ii / nzi;
    int j = plane % nyi, i = plane / nyi;
    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__
    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;
    float a = g0[ii], b = g1[ii], c = g2[ii];
    float a_y = vy ? g0[oy] : 0.0f, a_z = vz ? g0[oz] : 0.0f;
    float b_x = vx ? g1[ox] : 0.0f, b_z = vz ? g1[oz] : 0.0f;
    float c_x = vx ? g2[ox] : 0.0f, c_y = vy ? g2[oy] : 0.0f;
    float curl0 = dtdx * ((c_y - c) + (b - b_z));
    float curl1 = dtdx * ((a_z - a) + (c - c_x));
    float curl2 = dtdx * ((b_x - b) + (a - a_y));
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];
__TAIL0__
__TAIL1__
__TAIL2__
}
"""


def _tail(index: int, conductive: bool, km1: str, si1: str,
          km2: str, si2: str) -> str:
    if not conductive:
        return f"""    float u{index}_previous = u{index}[ii];
    float u{index}_new = ((u{index}_previous * {km1}) - curl{index}) * {si1};
    float f{index}_new = (((f{index}[ii] * {km2}) + u{index}_new) - u{index}_previous) * {si2};
    u{index}[ii] = u{index}_new;
    f{index}[ii] = f{index}_new;"""
    return f"""    float f{index}_previous = f{index}[ii];
    float u{index}_previous = u{index}[ii];
    float c{index}_previous = c{index}[ii];
    bool dsig{index} = ({km1} != 1.0f) || ({si1} != 1.0f);
    bool dsigu{index} = ({km2} != 1.0f) || ({si2} != 1.0f);
    if (dsig{index} && dsigu{index}) {{
        float c{index}_new = ((c{index}_previous * cf{index}[ii]) - curl{index}) * ci{index}[ii];
        float u{index}_new = (((u{index}_previous * {km1}) + c{index}_new) - c{index}_previous) * {si1};
        float f{index}_new = (((f{index}_previous * {km2}) + u{index}_new) - u{index}_previous) * {si2};
        c{index}[ii] = c{index}_new; u{index}[ii] = u{index}_new; f{index}[ii] = f{index}_new;
    }} else if (dsigu{index}) {{
        float u{index}_new = ((u{index}_previous * cf{index}[ii]) - curl{index}) * ci{index}[ii];
        float f{index}_new = (((f{index}_previous * {km2}) + u{index}_new) - u{index}_previous) * {si2};
        u{index}[ii] = u{index}_new; f{index}[ii] = f{index}_new;
    }} else if (dsig{index}) {{
        float c{index}_new = ((c{index}_previous * cf{index}[ii]) - curl{index}) * ci{index}[ii];
        float f{index}_new = (((f{index}_previous * {km1}) + c{index}_new) - c{index}_previous) * {si1};
        c{index}[ii] = c{index}_new; f{index}[ii] = f{index}_new;
    }} else {{
        f{index}[ii] = ((f{index}_previous * cf{index}[ii]) - curl{index}) * ci{index}[ii];
    }}"""


def conductive_pml_curl_source(codes: Sequence[int], backward: bool,
                               conductive: Sequence[bool],
                               contract: str = shaders.CONTRACT_OFF) -> str:
    """Emit one PML/conductive specialization, including its target flags."""
    codes = tuple(int(code) for code in codes)
    flags = tuple(bool(flag) for flag in conductive)
    if len(codes) != 3 or len(flags) != 3:
        raise ValueError("codes and conductive must each be triples")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__MASK__": templates.ownership_mask(codes, backward),
        "__TAIL0__": _tail(0, flags[0], "km_y", "si_y", "km_z", "si_z"),
        "__TAIL1__": _tail(1, flags[1], "km_z", "si_z", "km_x", "si_x"),
        "__TAIL2__": _tail(2, flags[2], "km_x", "si_x", "km_y", "si_y"),
    })


def compile_conductive_pml_curl(codes: Sequence[int], backward: bool,
                                conductive: Sequence[bool],
                                contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(conductive_pml_curl_source(
        codes, backward, conductive, contract)).conductive_pml_curl_step


def metal_conductive_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                       residency: Any = None) -> Coverage:
    """The established numerical PML-conductivity clauses plus Metal clauses."""
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    shared = shared_conductive_pml_coverage(
        _FieldsView(fields, _GridView(grid)), pml, sub_step)
    reasons = list(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(shared.reasons)
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalConductivePmlCurlPlan(KernelPlan):
    """Persistent-mirror plan for one active-PML conductive curl sub-step."""

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "conductive", "residency", "volumes")
    REPR_FIELDS = ("sub_step", "shape", "bc", "conductive")

    def __init__(self, sub_step, shape, dtdx, codes, conductive, residency,
                 targets, auxiliaries, history, condfac, condinv, sources,
                 coefficients, functions, volumes) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in codes)
        self.conductive = tuple(bool(flag) for flag in conductive)
        self.residency = residency
        self.volumes = tuple(volumes)
        super().__init__(functions, tuple(targets) + tuple(auxiliaries) + tuple(history)
                         + tuple(condfac) + tuple(condinv) + tuple(sources)
                         + tuple(coefficients) + (self.shape[0], self.shape[1],
                                                  self.shape[2], self.n_elem, self.dtdx))


def _functions(codes, backward, conductive, variants):
    return {mode: compile_conductive_pml_curl(codes, backward, conductive, mode)
            for mode in variants}


def plan_metal_conductive_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Dict[str, Any]] = None,
        ) -> Optional[MetalConductivePmlCurlPlan]:
    """Build an admitted real conductive PML curl plan, otherwise ``None``."""
    if not metal_conductive_pml_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    assert residency is not None
    grid = fields.grid
    spec = SUB_STEPS[sub_step]
    targets = tuple(CONDUCTIVE_SUB_STEPS[sub_step])
    codes = tuple(1 if kind == "metallic" else 0 for kind in _boundary_kinds(grid, pml))
    target_tensors = [residency.mirror(name, getattr(fields, name)) for name in targets]
    auxiliary_tensors = [residency.mirror("fu_" + name, getattr(fields, "fu_" + name))
                         for name in targets]
    conductive = conductive_targets(fields, sub_step)
    history = [
        (residency.mirror("f_cond_" + name, getattr(fields, "f_cond_" + name))
         if conductive[index] else target_tensors[index])
        for index, name in enumerate(targets)]
    condfac = [
        (residency.mirror(f"condfac_{name}:pml", fields.condfac_for(name), constant=True)
         if conductive[index] else target_tensors[index])
        for index, name in enumerate(targets)]
    condinv = [
        (residency.mirror(f"condinv_{name}:pml", fields.condinv_for(name), constant=True)
         if conductive[index] else target_tensors[index])
        for index, name in enumerate(targets)]
    sources = [residency.mirror(name, getattr(fields, name)) for name in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                                     getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                                     constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    selected = functions if functions is not None else _functions(
        codes, spec["backward"], conductive, contract_variants)
    return MetalConductivePmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, conductive, residency,
        target_tensors, auxiliary_tensors, history, condfac, condinv, sources,
        coefficients, selected,
        targets + tuple("fu_" + name for name in targets)
        + tuple("f_cond_" + name for name in targets) + tuple(spec["sources"]),
    )


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return metal_conductive_pml_curl_coverage(
        context.fields, context.pml, slot, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalConductivePmlCurlPlan]:
    return plan_metal_conductive_pml_curl(
        context.fields, context.pml, slot, context.residency, context.contract_variants)


def _arm_gate(context: Any) -> bool:
    return bool(getattr(context.pml, "is_active", False))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return tuple(arms.register(
        family=FAMILY, slot=slot, label="conductive PML curl",
        coverage=_arm_coverage, plan=_arm_plan, prefix="conductive PML curl: ",
        noun="conductive PML curl", gate=_arm_gate, wired=True,
    ) for slot in CONDUCTIVE_SUB_STEPS)


ARMS = register_arms()
