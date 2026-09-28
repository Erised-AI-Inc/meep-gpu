"""Complex/Bloch no-PML conductive curl sub-steps for PyTorch/MPS.

The complex conductive path combines three facts that must remain separate in the
implementation: the complex Yee stencil and Bloch rotation, the inactive-absorber
branch, and the per-target conductive tail.  Its tail is the array path's ordered
three-pass recurrence, with a real coefficient applied through the probe-licensed
``float2`` field-left expansion::

    value = value * condfac
    value = value - curl
    value = value * condinv

This is a distinct family rather than an option on the lossless complex curl:
the plan binds six additional real volumes and a component without sigma must not
load either one. The native MPS byte gate establishes this B/D sub-step pair; its
complete-step composition is separately exercised with the stored-E and null
companions before the arm is wired in the experimental composer.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    COVERED_BOUNDARIES,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _susceptibility_reasons,
    _volume_reasons,
)
from ..triton_kernels.no_pml_conductive import conductive_no_pml_targets
from . import shaders, templates
from .complex_fields import (
    _complex_mirror,
    _complex_volume_reasons,
    _expansion_reasons,
    _phase_block,
    bloch_phase_table,
    expansion_from_probe,
    load_expansion_probe,
    phase_arguments,
)
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .plans import KernelPlan

FAMILY = "complex_no_pml_conductive"
SUB_STEPS = {
    "step_B": {"targets": ("Bx", "By", "Bz"), "sources": ("Ex", "Ey", "Ez"),
               "backward": False},
    "step_D": {"targets": ("Dx", "Dy", "Dz"), "sources": ("Hx", "Hy", "Hz"),
               "backward": True},
}
SOURCE_ACCESSOR = {"step_B": "get_E", "step_D": "get_H"}

__all__ = [
    "FAMILY",
    "SUB_STEPS",
    "MetalComplexConductiveNoPmlCurlPlan",
    "complex_conductive_no_pml_curl_source",
    "compile_complex_conductive_no_pml_curl",
    "metal_complex_conductive_no_pml_curl_coverage",
    "plan_metal_complex_conductive_no_pml_curl",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__
__HELPERS__

kernel void complex_no_pml_conductive_curl_step(
    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device const float2* g0      [[buffer(3)]],
    device const float2* g1      [[buffer(4)]],
    device const float2* g2      [[buffer(5)]],
    device const float*  cf0     [[buffer(6)]],
    device const float*  cf1     [[buffer(7)]],
    device const float*  cf2     [[buffer(8)]],
    device const float*  ci0     [[buffer(9)]],
    device const float*  ci1     [[buffer(10)]],
    device const float*  ci2     [[buffer(11)]],
    constant uint&       nx      [[buffer(12)]],
    constant uint&       ny      [[buffer(13)]],
    constant uint&       nz      [[buffer(14)]],
    constant uint&       n_elem  [[buffer(15)]],
    constant float&      dtdx    [[buffer(16)]],
    constant float2&     px      [[buffer(17)]],
    constant float2&     py      [[buffer(18)]],
    constant float2&     pz      [[buffer(19)]],
    uint idx [[thread_position_in_grid]])
{
__GUARD__
__DECODE__
    int nxi = int(nx);
    int nyz = nyi * nzi;

    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    float2 a   = g0[ii];
    float2 b   = g1[ii];
    float2 c   = g2[ii];
    float2 a_y = vy ? g0[oy] : float2(0.0f, 0.0f);
    float2 a_z = vz ? g0[oz] : float2(0.0f, 0.0f);
    float2 b_x = vx ? g1[ox] : float2(0.0f, 0.0f);
    float2 b_z = vz ? g1[oz] : float2(0.0f, 0.0f);
    float2 c_x = vx ? g2[ox] : float2(0.0f, 0.0f);
    float2 c_y = vy ? g2[oy] : float2(0.0f, 0.0f);

__PHASE_X__
__PHASE_Y__
__PHASE_Z__

    float2 t0 = ((c_y - c) + (b - b_z));
    float2 t1 = ((a_z - a) + (c - c_x));
    float2 t2 = ((b_x - b) + (a - a_y));
    float2 curl0 = c_mul_coefficient_left(dtdx, t0);
    float2 curl1 = c_mul_coefficient_left(dtdx, t1);
    float2 curl2 = c_mul_coefficient_left(dtdx, t2);

    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

__TAIL0__
__TAIL1__
__TAIL2__
}
"""


def _tail(index: int, conductive: bool) -> str:
    if not conductive:
        return f"    f{index}[ii] = f{index}[ii] - curl{index};"
    return f"""    float2 value{index} = f{index}[ii];
    value{index} = c_mul_field_left(value{index}, cf{index}[ii]);
    value{index} = value{index} - curl{index};
    value{index} = c_mul_field_left(value{index}, ci{index}[ii]);
    f{index}[ii] = value{index};"""


def complex_conductive_no_pml_curl_source(
        codes: Sequence[int], backward: bool, phased: Sequence[int],
        conductive: Sequence[bool], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> str:
    """Emit the fully specialized complex no-PML conductive curl source."""
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    flags = tuple(bool(flag) for flag in conductive)
    if len(codes) != 3 or len(phased) != 3 or len(flags) != 3:
        raise ValueError("codes, phased, and conductive must each be triples")
    for axis, (code, flag) in enumerate(zip(codes, phased)):
        if flag and code != templates.PERIODIC:
            raise ValueError(f"axis {axis} carries a phase but is not periodic")
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__PHASE_X__": _phase_block("x", bool(phased[0]), backward),
        "__PHASE_Y__": _phase_block("y", bool(phased[1]), backward),
        "__PHASE_Z__": _phase_block("z", bool(phased[2]), backward),
        "__MASK__": templates.ownership_mask(
            codes, backward, zero=templates.COMPLEX_ZERO),
        "__TAIL0__": _tail(0, flags[0]),
        "__TAIL1__": _tail(1, flags[1]),
        "__TAIL2__": _tail(2, flags[2]),
    })


def compile_complex_conductive_no_pml_curl(
        codes: Sequence[int], backward: bool, phased: Sequence[int],
        conductive: Sequence[bool], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(complex_conductive_no_pml_curl_source(
        codes, backward, phased, conductive, expansion,
        contract)).complex_no_pml_conductive_curl_step


def _base_reasons(fields: Any, pml: Any, grid: Any, residency: Any,
                  probe: Any) -> List[str]:
    """Shared complex curl facts, with inactive PML and conductivity admitted."""
    reasons = list(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    if not (bool(getattr(fields, "force_complex_fields", False))
            or bool(_call(grid, "has_bloch", default=False))):
        reasons.append("complex64 storage is required; real storage belongs to real curls")
    if pml is not None and bool(getattr(pml, "is_active", False)):
        reasons.append("an active PML layer requires the split-field conductive curl")
    if bool(getattr(fields, "_pml_active", False)):
        reasons.append("Fields remains in PML storage mode while the layer is inert")
    # UNREADABLE IS A REFUSAL, NEVER A CRASH. `_boundary_kinds` returns None for a
    # grid the array path will not resolve (coverage.py:105-121), and the triple is
    # read TWICE below — once for the covered ghost rule and once for per-axis Bloch
    # legality — so iterating it would raise where the contract says refuse. This is
    # the shared base of four families (the plain and conductive complex no-PML
    # curls, the complex no-PML off-diagonal E, and the fused conductive pair), so
    # the crash reached all four; the refusal now does too.
    kinds = _boundary_kinds(grid, None)
    if kinds is None:
        reasons.append(
            "the per-axis boundary rule is unreadable, so neither the covered "
            "ghost rule nor per-axis Bloch legality can be asked of this grid")
        kinds = ()
    for axis, kind in enumerate(kinds):
        if kind not in COVERED_BOUNDARIES:
            reasons.append(f"axis {axis} boundary {kind!r} is not covered")
    if (_call(grid, "has_symmetry", default=False)
            or any(_call(grid, "is_mirrored", axis, default=False) for axis in range(3))):
        reasons.append("a mirror plane is active")
    if bool(getattr(grid, "cylindrical", False)):
        reasons.append("cylindrical coordinates are not carried")
    if bool(getattr(grid, "bfast_active", False)):
        reasons.append("BFAST is active")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append("special_kz beta is nonzero")
    if not bool(getattr(fields, "stores_E", False)):
        reasons.append("E is recomputed from D rather than stored")
    phase_reader = getattr(grid, "bloch_phase", None)
    if not callable(phase_reader):
        reasons.append("grid.bloch_phase is missing or not callable")
    else:
        k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
        for axis, kind in enumerate(kinds):
            try:
                phase = phase_reader(axis)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"grid.bloch_phase({axis}) raised {exc!r}")
                continue
            if phase is not None and kind != "periodic":
                reasons.append(f"axis {axis} carries a Bloch phase but is {kind!r}")
            if kind == "metallic" and float(k_point[axis]) != 0.0:
                reasons.append(f"axis {axis} is metallic with nonzero Bloch k")
    if bool(getattr(fields, "has_nonlinearity", False)):
        reasons.append("chi2/chi3 is installed")
    reasons.extend(_susceptibility_reasons(fields))
    reasons.extend(_expansion_reasons(probe))
    return reasons


def metal_complex_conductive_no_pml_curl_coverage(
        fields: Any, pml: Any, sub_step: str, residency: Any = None,
        probe: Any = None) -> Coverage:
    """Whether one complex conductive curl sub-step is covered."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons = _base_reasons(fields, pml, grid, residency, probe)
    reader = getattr(fields, "condfac_for", None)
    inverse_reader = getattr(fields, "condinv_for", None)
    if not callable(reader) or not callable(inverse_reader):
        reasons.append("fields does not expose condfac_for/condinv_for")
    else:
        any_conductive = False
        for target in CURL_TARGETS:
            try:
                any_conductive |= reader(target) is not None
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
        if not any_conductive:
            reasons.append("no curl target carries a conductivity")

    spec = SUB_STEPS[sub_step]
    shape = tuple(getattr(grid, "shape", ()))
    for name in spec["targets"]:
        array = getattr(fields, name, None)
        if array is None:
            reasons.append(f"{name} is not allocated")
        else:
            reasons.extend(_complex_volume_reasons(name, array, shape))
    accessor = getattr(fields, SOURCE_ACCESSOR[sub_step], None)
    if not callable(accessor):
        reasons.append(f"fields.{SOURCE_ACCESSOR[sub_step]} is missing or not callable")
    else:
        for name in spec["sources"]:
            try:
                source = accessor(name)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"fields.{SOURCE_ACCESSOR[sub_step]}({name!r}) raised {exc!r}")
                continue
            if source is None:
                reasons.append(f"{name} is not served by fields.{SOURCE_ACCESSOR[sub_step]}")
            else:
                reasons.extend(_complex_volume_reasons(name, source, shape))
    if callable(reader) and callable(inverse_reader):
        flags = conductive_no_pml_targets(fields, sub_step)
        for index, target in enumerate(spec["targets"]):
            try:
                condfac, condinv = reader(target), inverse_reader(target)
            except Exception:  # already reported in all-target loop
                continue
            if (condfac is None) != (condinv is None):
                reasons.append(f"{target} has only one of condfac/condinv")
            elif flags[index]:
                reasons.extend(_volume_reasons(f"condfac[{target}]", condfac, shape))
                reasons.extend(_volume_reasons(f"condinv[{target}]", condinv, shape))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalComplexConductiveNoPmlCurlPlan(KernelPlan):
    """Persistent-mirror plan for a complex no-PML conductive curl sub-step."""

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "phased", "phase_values", "conductive", "expansion",
                 "residency", "volumes")
    REPR_FIELDS = ("sub_step", "shape", "bc", "phased", "conductive", "expansion")

    def __init__(self, sub_step: str, shape: Sequence[int], dtdx: float,
                 codes: Sequence[int], phased: Sequence[int], phase_values, conductive,
                 expansion: str, residency: Residency, targets, sources, condfac, condinv,
                 functions: Dict[str, Any], volumes: Sequence[str]) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.conductive = tuple(bool(flag) for flag in conductive)
        self.expansion = str(expansion)
        self.residency = residency
        self.volumes = tuple(volumes)
        super().__init__(functions, tuple(targets) + tuple(sources) + tuple(condfac)
                         + tuple(condinv) + (self.shape[0], self.shape[1], self.shape[2],
                                              self.n_elem, self.dtdx) + self.phase_values)


def _functions(codes, backward, phased, conductive, expansion, variants):
    return {mode: compile_complex_conductive_no_pml_curl(
        codes, backward, phased, conductive, expansion, mode) for mode in variants}


def _source_binding(fields: Any, sub_step: str, name: str) -> Tuple[str, Any]:
    """Return a source's canonical mirror name and its host storage.

    In a no-PML step, ``Fields.get_H("Hx")`` returns the live ``Bx`` allocation.
    It must therefore share the B mirror written by the preceding B curl, rather
    than creating a second device copy under the logical H name.  Stored H is a
    PML-only state and this family refuses active PML, but the identity check keeps
    the canonicalization correct for a test double or future derived-field mode.
    """
    source = getattr(fields, SOURCE_ACCESSOR[sub_step])(name)
    if sub_step == "step_D":
        primary_name = "B" + name[1]
        primary = getattr(fields, primary_name, None)
        if source is primary:
            return primary_name, source
    return name, source


def plan_metal_complex_conductive_no_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,), probe: Any = None,
        functions: Optional[Dict[str, Any]] = None,
        ) -> Optional[MetalComplexConductiveNoPmlCurlPlan]:
    """Build one covered complex/Bloch conductive curl plan, otherwise ``None``."""
    if not metal_complex_conductive_no_pml_curl_coverage(
            fields, pml, sub_step, residency, probe).covered:
        return None
    assert residency is not None
    expansion = expansion_from_probe(probe if probe is not None else load_expansion_probe())
    if expansion is None:
        return None
    grid = fields.grid
    spec = SUB_STEPS[sub_step]
    kinds = _boundary_kinds(grid, None)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    phased, phase_values = phase_arguments(
        bloch_phase_table(grid, kinds), backward=bool(spec["backward"]))
    targets = [_complex_mirror(residency, name, getattr(fields, name))
               for name in spec["targets"]]
    source_bindings = [_source_binding(fields, sub_step, name) for name in spec["sources"]]
    sources = [_complex_mirror(residency, name, source)
               for name, source in source_bindings]
    conductive = conductive_no_pml_targets(fields, sub_step)
    condfac = [
        (residency.mirror(f"condfac_{name}:complex_no_pml", fields.condfac_for(name),
                          constant=True) if conductive[index] else targets[index])
        for index, name in enumerate(spec["targets"])
    ]
    condinv = [
        (residency.mirror(f"condinv_{name}:complex_no_pml", fields.condinv_for(name),
                          constant=True) if conductive[index] else targets[index])
        for index, name in enumerate(spec["targets"])
    ]
    selected = functions if functions is not None else _functions(
        codes, bool(spec["backward"]), phased, conductive, expansion, contract_variants)
    return MetalComplexConductiveNoPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, phased, phase_values,
        conductive, expansion, residency, targets, sources, condfac, condinv,
        selected, tuple(spec["targets"]) + tuple(name for name, _ in source_bindings),
    )


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return metal_complex_conductive_no_pml_curl_coverage(
        context.fields, context.pml, slot, context.residency,
        context.extra.get("probe"))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexConductiveNoPmlCurlPlan]:
    return plan_metal_complex_conductive_no_pml_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants, context.extra.get("probe"))


def _arm_gate(context: Any) -> bool:
    return not bool(getattr(context.pml, "is_active", False))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return tuple(arms.register(
        family=FAMILY, slot=slot, label="complex conductive no-PML curl",
        coverage=_arm_coverage, plan=_arm_plan,
        prefix="complex conductive no-PML curl: ",
        noun="complex conductive no-PML curl", gate=_arm_gate, wired=True,
    ) for slot in SUB_STEPS)


ARMS = register_arms()
