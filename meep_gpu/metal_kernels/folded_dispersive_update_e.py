"""Mirror-folded real PML dispersive ``update_E`` for the Metal composer.

A mirror changes the stored extent, not this sub-step's arithmetic: MEEP forms
``(D - sum(P)) * chi1inv`` at each stored cell and applies the two stored-E PML
accumulations without any neighbour shift or parity operation.  This module owns
the *intersection admission*—a real active mirror fold with live supported
electric poles—and deliberately reuses the ordinary family's one binding builder.
The shared builder is important: the pole order, half-integer PML tables, and
persistent-residency identities are part of the numerical contract, not incidental
implementation details.

The native gate certifies the exact folded stored extent, and the experimental
Metal composer separately measures the folded B/D/H/fill/E/ADE residency seam.
This is not production driver dispatch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import (
    COVERED_SUSCEPTIBILITY_KINDS,
    ELECTRIC_COMPONENTS,
    Coverage,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _susceptibility_reasons,
    _volume_reasons,
)
from . import dispersive_update_e as _base
from . import shaders, symmetry as _symmetry
from .coverage import _residency_declaration_reasons
from .device import Residency

FAMILY = "folded_dispersive_update_e"
SLOT = "update_E"
LABEL = "folded dispersive PML E"

__all__ = [
    "FAMILY",
    "LABEL",
    "SLOT",
    "folded_dispersive_e_coverage",
    "plan_folded_dispersive_e",
    "register_arms",
]


def _poles(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    """The same live-pole partition the shared plan resolves at every launch."""
    return _base._poles(fields)


def folded_dispersive_e_coverage(fields: Any, pml: Any,
                                  residency: Any = None) -> Coverage:
    """Whether the shared pointwise plan may operate on a folded stored volume.

    ``symmetry._folded_constitutive_grid_reasons`` is deliberately reused rather
    than weakening the ordinary dispersive predicate: it carries all fold-specific
    axis classification, active-PML, real-storage, Cartesian, k=0, nonlinear,
    BFAST, and beta clauses.  The remaining checks are precisely the ordinary
    dispersive body's live volumes and pole ABI, evaluated against ``grid.shape``
    which is already the folded stored shape.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons: List[str] = list(
        _symmetry._folded_constitutive_grid_reasons(fields, pml, grid))
    reasons.extend(_residency_declaration_reasons(residency))

    if not bool(getattr(fields, "_pml_active", False)):
        reasons.append("Fields is not in PML storage mode")
    if not bool(getattr(fields, "stores_E", False)):
        reasons.append("E is recomputed from D rather than stored")
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append(
            "an off-diagonal chi1inv row is installed; folded tensor dispersion "
            "requires its own fused stencil")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return Coverage(False, tuple(dict.fromkeys(reasons)))
    if int(np.prod(shape)) >= 2 ** 31:
        reasons.append("the grid exceeds the kernel's int32 index range")

    reasons.extend(_susceptibility_reasons(fields))
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if not states:
        reasons.append(
            "no susceptibility is registered; folded ordinary constitutive owns "
            "update_E")

    for component, displacement, _axis, _name in _base.E_TERMS:
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
                    reasons.append(
                        f"polarization {index} {label}[{component}] is not allocated")
                else:
                    reasons.extend(_volume_reasons(
                        f"polarization {index} {label}[{component}]", array, shape))
        scratch = getattr(state, "_scratch", None)
        if scratch is None:
            reasons.append(f"polarization {index} scratch is not allocated")
        else:
            reasons.extend(_volume_reasons(f"polarization {index} scratch", scratch,
                                            shape))

    for component, driven in _poles(fields).items():
        if len(driven) > _base.MAX_POLES:
            reasons.append(
                f"{component} is driven by {len(driven)} poles, more than "
                f"MAX_POLES={_base.MAX_POLES}")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def plan_folded_dispersive_e(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, int, int], Any]] = None,
        ) -> Optional[_base.MetalDispersiveEPlan]:
    """Build the shared pointwise plan after the folded predicate has passed."""
    if not folded_dispersive_e_coverage(fields, pml, residency).covered:
        return None
    assert residency is not None and pml is not None
    return _base._plan_metal_dispersive_e_covered(
        fields, pml, residency, contract_variants, functions)


def _has_fold_and_poles(context: Any) -> bool:
    """Consult this intersection only where both of its defining facts are live."""
    try:
        fields = getattr(context, "fields", None)
        grid = getattr(fields, "grid", None)
        return bool(grid is not None and _symmetry._has_real_fold(grid)
                    and tuple(getattr(fields, "polarizations", ()) or ()))
    except Exception:  # noqa: BLE001 - unreadable state falls safely to the array path
        return False


def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"folded dispersive E cannot fill {slot}",))
    return folded_dispersive_e_coverage(context.fields, context.pml, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[_base.MetalDispersiveEPlan]:
    if slot != SLOT:
        return None
    return plan_folded_dispersive_e(context.fields, context.pml, context.residency,
                                    context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415 - arm registration is import-time by design

    return (arms.register(
        FAMILY, SLOT, LABEL, _arm_coverage, _arm_plan,
        prefix=f"{LABEL}: ", noun="folded PML dispersive stored-E constitutive",
        gate=_has_fold_and_poles, wired=True),)


ARMS = register_arms()
