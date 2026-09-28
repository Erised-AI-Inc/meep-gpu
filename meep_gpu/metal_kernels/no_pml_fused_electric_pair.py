"""The lossless no-PML ``step_D`` welded into stored-E ``update_E``, deposit carried.

THE LOSSLESS SIBLING of :mod:`.no_pml_conductive_fused_electric_pair`, and the
second Metal product on :data:`meep_gpu.deposit_repair.PLAIN_PATH` — the repair
that inverts ``update_E``'s plain overwrite (``stepping.py:1019-1022``) rather than
the split-field recurrence. The two modules are one construction over the two
certified no-absorber curl arms, split the way the arms themselves are split:
:func:`.no_pml_conductive._any_curl_conductivity` is the partition, the conductive
arm requires a sigma on a curl target by name and the lossless arm refuses every
sigma by name, so exactly one of the two families answers for any run.

THE CELL, from ``results/fusion_matrix_metal_2026-08-31_plainrepair``::

    D->E   no-PML curl x no-PML stored E    1 seam-instance
           examples:material-dispersion.py
           verdict: FITS -- 15..18 pointers against a 30 ceiling; NOT BUILT

That row is lossless (no conductivity is installed, so the driver takes the plain
per-source injection loop at ``driver.py:3307``), periodic on all three axes, with
TWO Lorentz poles per component and ONE non-integrated electric source injected
INSIDE this seam — which is why :data:`CARRIES_DEPOSIT_REPAIR` is the product
rather than a refinement of it, and why this family serves 1 of its 1 seam-instance
where its conductive sibling serves 0 of 2 (see that module's priced refusal).

THE SEAM, PASS BY PASS — the conductive sibling's disposition, minus the rescale:

* **the electric injection — CARRIED, through the deposit repair**, saved and
  restored through :data:`REPAIR_PATHS` = the PLAIN path alone;
* **the scaled conductive injection — STRUCTURALLY UNREACHABLE here, and the
  clause still runs.** The lossless curl arm refuses every D-target conductivity
  by name, and ``FdtdDriver._inject_electric_through_conductivity`` skips a target
  whose ``condinv_for`` is None (``driver.py:3366-3367``) — but the clause is
  consulted anyway (imported from the sibling, one home), so a drift in either
  fact surfaces as a refusal rather than a silent admission;
* **``zero_metal_D`` — CARRIED INLINE** on the register (the corpus row is
  all-periodic, where the mask emits nothing; the walled specialisations are gated
  regardless);
* **the two fills — INERT**: both halves refuse a mirror plane, restated by name.

EVERYTHING IS TRANSCRIBED: the curl half from
:data:`.no_pml_curl._PLAIN_CURL_TEMPLATE`'s own decode, ghost, load, curl and mask
text (``no_pml_curl.py:47-141``), one target per dispatch, every emitted line
asserted to be a line of the certified all-component emitter's own output; the
constitutive half from :func:`.no_pml_stored_e.stored_e_source`
(``no_pml_stored_e.py:76-88``). What moves is :data:`LIFT_EDITS` — the certified
``float source = d_in[idx];`` reload becomes the register — plus the one split the
lossless tail needs: the certified ``f0[ii] = f0[ii] - curl0;`` is named into a
register before its store so the constitutive half and the wall clear can read it.
A float32 stored to a ``device float*`` and reloaded is bit-identical to the
register, so the split is byte-neutral by construction, and the gate's equivalence
leg measures it anyway.

PER COMPONENT for symmetry with the sibling and the certified stored-E family
(:data:`BINDINGS_PER_COMPONENT` = 19 of the 31-binding ceiling with all eight pole
slots declared); one dispatch per component IS the fusion.

NOT WIRED, composed only through ``launch.FUSED_PAIR_ARMS``. MEASURED through
``plan_step`` with ``fuse=False`` under ``MEEP_GPU_SUBNORMAL_POLICY=flush`` on
``metal_composition_matrix.real_stored_e_no_pml(with_polarization=True)``:
``step_D`` comes back ``'no-PML curl'`` and ``update_E`` ``'no-PML stored E'`` —
the two labels this family's row declares. ``meep_gpu.fastpath.plan_fast_path``
still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from ..triton_kernels.no_pml import SUB_STEPS as _NO_PML_SUB_STEPS
from . import shaders, templates
from .ade_update_p import _physical_name
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .fused_dispersive_pair import _TARGETS, _ZERO_METAL_ROWS
from .no_pml_conductive_fused_electric_pair import (
    _scaled_conductive_injection_reasons,
)
from .no_pml_curl import metal_plain_curl_coverage, plain_curl_source
from .no_pml_stored_e import (
    E_TERMS,
    MAX_POLES,
    _poles,
    metal_stored_e_coverage,
    stored_e_source,
)
from .. import deposit_repair as _deposit_repair

FAMILY = "no_pml_fused_electric_pair"

#: The whole demand of this cell carries an electric source inside the seam, so the
#: flag is the product. See the conductive sibling's constant for the wiring the
#: flag claims; both travel with :data:`REPAIR_PATHS` and the plan's own
#: ``repair_paths`` attribute, which the installer reads.
CARRIES_DEPOSIT_REPAIR = True

#: The PLAIN repair alone: both halves refuse an active absorber, so the recurrence
#: this seam inverts is always ``update_E``'s pure overwrite.
REPAIR_PATHS: Tuple[str, ...] = (_deposit_repair.PLAIN_PATH,)

SLOT = "step_D"

#: Driver call sites one run performs (``driver.py:3292-3306``); the two fills are
#: refused, not carried.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: READ from the no-absorber arm table: step_D's sources are the B volumes on this
#: branch (H is deliberately unallocated; ``no_pml.py:178-181``).
_CURL_SPEC = _NO_PML_SUB_STEPS[SLOT]
CURL_TARGETS: Tuple[str, ...] = tuple(_CURL_SPEC["targets"])
CURL_SOURCES: Tuple[str, ...] = tuple(_CURL_SPEC["sources"])

#: The per-component signature as data: 14 pointers (D target, three B sources, the
#: E target, the inverse epsilon, eight pole slots) plus five separate scalars.
POINTERS_PER_COMPONENT = 14
BINDINGS_PER_COMPONENT = POINTERS_PER_COMPONENT + 5

#: The lift, as data. The second row is the lossless tail's REGISTER SPLIT — not an
#: arithmetic change: the certified line's right-hand side is computed once, named,
#: and stored, so the wall clear and the constitutive half can read the register.
LIFT_EDITS: Tuple[Tuple[str, str, str], ...] = (
    ("    float source = d_in[idx];",
     "    float source = value{axis};",
     "THE SEAM: the register the curl tail just stored, not a reload of the "
     "displacement"),
    ("    f{axis}[ii] = f{axis}[ii] - curl{axis};",
     "    float value{axis} = f{axis}[ii] - curl{axis};\n"
     "__ZERO_METAL__\n"
     "    f{axis}[ii] = value{axis};",
     "the certified lossless tail, split so the stepped value has a name; the "
     "store is KEPT — the next timestep's curl and update_P read the D volume"),
)

__all__ = [
    "BINDINGS_PER_COMPONENT", "CARRIES_DEPOSIT_REPAIR", "CURL_SOURCES",
    "CURL_TARGETS", "FAMILY", "LIFT_EDITS", "POINTERS_PER_COMPONENT",
    "REPAIR_PATHS", "REPLACES", "SLOT",
    "MetalNoPmlFusedElectricPairPlan",
    "compile_no_pml_fused_electric_pair",
    "no_pml_fused_electric_pair_coverage",
    "no_pml_fused_electric_pair_source",
    "plan_metal_no_pml_fused_electric_pair",
    "refuted_over_ceiling_source", "zero_metal_mask",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void no_pml_fused_electric_pair_component(
    device float*       f__T__  [[buffer(0)]],
    device const float* g0      [[buffer(1)]],
    device const float* g1      [[buffer(2)]],
    device const float* g2      [[buffer(3)]],
    device float*       e_out   [[buffer(4)]],
    device const float* inv_e   [[buffer(5)]],
    device const float* p0      [[buffer(6)]],
    device const float* p1      [[buffer(7)]],
    device const float* p2      [[buffer(8)]],
    device const float* p3      [[buffer(9)]],
    device const float* p4      [[buffer(10)]],
    device const float* p5      [[buffer(11)]],
    device const float* p6      [[buffer(12)]],
    device const float* p7      [[buffer(13)]],
    constant uint&      nx      [[buffer(14)]],
    constant uint&      ny      [[buffer(15)]],
    constant uint&      nz      [[buffer(16)]],
    constant uint&      n_elem  [[buffer(17)]],
    constant float&     dtdx    [[buffer(18)]],
    uint idx [[thread_position_in_grid]])
{
__BODY__
}
"""


def zero_metal_mask(axis: int, zero_metal: Sequence[bool]) -> str:
    """``stepping.zero_metal_D`` for ONE target: the D off-diagonal, imported."""
    lines = [f"    value{axis} = {flag} ? 0.0f : value{axis};"
             for row_target, wall_axis, flag in _ZERO_METAL_ROWS
             if row_target == axis and bool(zero_metal[wall_axis])]
    return "\n".join(lines) or "    // no walled axis clears this target"


def _per_component_curl(codes: Sequence[int], axis: int,
                        zero_metal: Sequence[bool]) -> List[str]:
    """The certified plain curl text, restricted to ONE target, every line anchored."""
    spec = _TARGETS[axis]
    certified = plain_curl_source(codes, True, False)
    lines = [
        templates.GUARD,
        "",
        "    int nxi = int(nx), nyi = int(ny), nzi = int(nz);",
        "    int nyz = nyi * nzi;",
        "    int ii = int(idx);",
        "    int k = ii % nzi;",
        "    int plane = ii / nzi;",
        "    int j = plane % nyi;",
        "    int i = plane / nyi;",
        "",
        "    int si = i - 1, sj = j - 1, sk = k - 1;",
        "    bool vx = true, vy = true, vz = true;",
        templates.ghost("x", codes[0], True),
        templates.ghost("y", codes[1], True),
        templates.ghost("z", codes[2], True),
        "",
        "    int ox = si * nyz + j * nzi + k;",
        "    int oy = i * nyz + sj * nzi + k;",
        "    int oz = i * nyz + j * nzi + sk;",
        "",
        *spec["loads"],
        "",
        spec["curl"],
        "",
        "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);",
    ]
    mask = [line for line in templates.ownership_mask(codes, True).splitlines()
            if line.strip().startswith(f"curl{axis} ")]
    lines.extend(mask or [f"    // no metallic axis masks curl{axis}"])
    lines.append("")
    # THE LOSSLESS TAIL SPLIT, from LIFT_EDITS as data. The certified line must be
    # a line of the certified emitter's own output before it is edited.
    old, new, _why = LIFT_EDITS[1]
    certified_tail = old.format(axis=axis)
    assert certified_tail in certified, certified_tail
    lines.append(new.format(axis=axis).replace(
        "__ZERO_METAL__", zero_metal_mask(axis, zero_metal)))
    for line in lines:
        text = line.strip()
        if (not text or text.startswith("//") or "value" in line
                or line == templates.GUARD):
            continue
        assert line in certified, (
            f"lifted curl line is not the certified plain emitter's own: {line!r}")
    return lines


def _constitutive_chain(axis: int, pole_count: int) -> List[str]:
    """The certified stored-E body with the seam edit applied, every line anchored."""
    certified = stored_e_source(pole_count)
    old, new, why = LIFT_EDITS[0]
    assert old in certified, (
        f"the certified stored-E body no longer carries {old!r}; the seam has no "
        f"anchor")
    lines = [f"    // {why}.", new.format(axis=axis)]
    for index in range(int(pole_count)):
        line = f"    source = source - p{index}[idx];"
        assert line in certified, line
        lines.append(line)
    store = "    e_out[idx] = source * inv_e[idx];"
    assert store in certified
    lines.append(store)
    return lines


def no_pml_fused_electric_pair_source(
        codes: Sequence[int], axis: int, pole_count: int,
        zero_metal: Sequence[bool], contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, target, poles, walls, mode)."""
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    if not 0 <= int(pole_count) <= MAX_POLES:
        raise ValueError(f"pole_count must be in [0, {MAX_POLES}], got {pole_count!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    body = _per_component_curl(codes, axis, zero_metal)
    body.append("")
    body.append("    // --- stored-E update_E (stepping.py:981-984, :993) ---------")  # stepping.py live lines for the frozen device-text citation(s) in this string: 981-984->1010-1013, 993->1022
    body.extend(_constitutive_chain(axis, int(pole_count)))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__T__": str(axis),
        "__BODY__": "\n".join(body),
    })


def compile_no_pml_fused_electric_pair(
        codes: Sequence[int], axis: int, pole_count: int,
        zero_metal: Sequence[bool], contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(no_pml_fused_electric_pair_source(
        codes, axis, pole_count, zero_metal, contract)
    ).no_pml_fused_electric_pair_component


def refuted_over_ceiling_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The shipped per-component signature padded one binding PAST the ceiling."""
    source = no_pml_fused_electric_pair_source(
        (0, 0, 0), 0, MAX_POLES, (False, False, False), contract)
    pad = "".join(
        f"    device const float* pad{slot:<4}[[buffer({slot})]],\n"
        for slot in range(BINDINGS_PER_COMPONENT, MAX_BUFFER_BINDINGS + 1))
    anchor = "    uint idx [[thread_position_in_grid]])"
    assert anchor in source
    source = source.replace(anchor, pad + anchor)
    touch = " + ".join(
        f"pad{slot}[0]"
        for slot in range(BINDINGS_PER_COMPONENT, MAX_BUFFER_BINDINGS + 1))
    marker = "    e_out[idx] = source * inv_e[idx];"
    assert marker in source
    return source.replace(
        marker, f"    e_out[idx] = source * inv_e[idx] + ({touch}) * 0.0f;")


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def no_pml_fused_electric_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                        residency: Any = None) -> Coverage:
    """May one dispatch per component span plain ``step_D`` -> wall -> ``update_E``?"""
    reasons: List[str] = []

    curl = metal_plain_curl_coverage(fields, pml, SLOT, residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = metal_stored_e_coverage(fields, pml, residency)
    if not electric.covered:
        reasons.extend(f"stored-E half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM, CARRIED — pml and repair_paths forwarded, as on the sibling.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3294-3299)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR,
        pml=pml,
        repair_paths=REPAIR_PATHS))

    # THE CONDUCTIVE RESCALE CLAUSE, consulted even though the lossless curl half
    # already refuses every D-target conductivity by name: one home for the clause,
    # and a drift in either fact surfaces as a refusal rather than an admission.
    reasons.extend(_scaled_conductive_injection_reasons(fields, sources))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    mirrored = getattr(grid, "is_mirrored", None)
    for axis in range(3):
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entry:
    component: str
    displacement_name: str
    axis: int
    pole_count: int
    displacement: Any
    target: Any
    inverse_epsilon: Any
    states: Tuple[Any, ...]


class MetalNoPmlFusedElectricPairPlan:
    """Three dispatches that perform three driver passes, D never leaving a register."""

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = len(E_TERMS)

    #: Read by ``launch._install_fused_pair``; mirrors the module constant.
    repair_paths = REPAIR_PATHS

    __slots__ = ("entries", "residency", "volumes", "codes", "zero_metal", "shape",
                 "dtdx", "_functions", "_tensors", "_magnetic", "launches", "runs")

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], codes: Sequence[int],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 magnetic: Sequence[Any],
                 functions: Mapping[Tuple[str, int], Any],
                 tensors: Mapping[int, Any]) -> None:
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        self._magnetic = tuple(magnetic)
        self._functions = dict(functions)
        self._tensors = dict(tensors)
        self.launches = 0
        self.runs = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        """One complete D->E seam; pole pointers resolved per launch, never cached."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        for entry in self.entries:
            key = (mode, entry.axis)
            function = self._functions.get(key)
            if function is None:
                raise KeyError(
                    f"this plan holds no {mode!r} variant (it was built with "
                    f"{self.variants})")
            poles = tuple(state.P[entry.component] for state in entry.states)
            if len(poles) != entry.pole_count:
                raise RuntimeError(
                    f"{entry.component} resolved {len(poles)} poles, not the "
                    f"compiled count {entry.pole_count}")
            # Dead pole slots bind the inverse epsilon — constant, never written —
            # not the D target this kernel writes; see the sibling's run().
            slots = poles + (entry.inverse_epsilon,) * (MAX_POLES - len(poles))
            function(
                self._tensors[id(entry.displacement)],
                *(self._tensors[id(value)] for value in self._magnetic),
                self._tensors[id(entry.target)],
                self._tensors[id(entry.inverse_epsilon)],
                *(self._tensors[id(value)] for value in slots),
                self.shape[0], self.shape[1], self.shape[2],
                int(entry.displacement.size), self.dtdx,
            )
            self.launches += 1
        self.runs += 1

    def __repr__(self) -> str:
        return (f"MetalNoPmlFusedElectricPairPlan(shape={self.shape}, "
                f"poles={tuple(e.pole_count for e in self.entries)}, "
                f"bc={self.codes}, zero_metal={self.zero_metal})")


def plan_metal_no_pml_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, int], Any]] = None,
        ) -> Optional[MetalNoPmlFusedElectricPairPlan]:
    """Build the fused plan, or ``None`` when the seam is refused."""
    if not no_pml_fused_electric_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, None))
    walls = zero_metal_axes(grid)
    order = _poles(fields)

    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: set = set()

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        tensors[id(host)] = residency.mirror(name, host, constant=constant)
        volumes.append(name)
        return host

    magnetic = tuple(bind(name, getattr(fields, name)) for name in CURL_SOURCES)
    for (component, displacement_name), axis in zip(E_TERMS, range(3)):
        displacement = bind(displacement_name, getattr(fields, displacement_name))
        target = bind(component, getattr(fields, component))
        inverse = bind(f"inv_eps_{component}:no_pml",
                       fields.inverse_epsilon_for(component), constant=True)
        states = order[component]
        entries.append(_Entry(component, displacement_name, axis, len(states),
                              displacement, target, inverse, states))
        required.update((mode, axis) for mode in contract_variants)

    for state_index, state in enumerate(tuple(getattr(fields, "polarizations", ())
                                              or ())):
        for component in tuple(state.driven()):
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                bind(_physical_name(state_index, label, host), host)
        scratch = state._scratch
        bind(_physical_name(state_index, "scratch", scratch), scratch)

    selected = dict(functions or {})
    by_axis = {entry.axis: entry.pole_count for entry in entries}
    for key in required:
        if key not in selected:
            selected[key] = compile_no_pml_fused_electric_pair(
                codes, key[1], by_axis[key[1]], walls, key[0])

    return MetalNoPmlFusedElectricPairPlan(
        entries, residency, volumes, codes, walls, grid.shape,
        grid.dt / grid.dx, magnetic, selected, tensors)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"no-PML fused electric pair cannot fill {slot}",))
    return no_pml_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalNoPmlFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_no_pml_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False`` — enumerable, never selectable."""
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "no-PML fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="no-PML fused electric D/E pair: ",
                          noun="no-PML fused D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
