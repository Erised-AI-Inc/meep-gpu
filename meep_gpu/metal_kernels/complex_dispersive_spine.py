"""Complex/Bloch PML B/D/H spine for the pole-aware constitutive product.

The PML curls and magnetic constitutive recurrence do not read a polarization
state: they consume stored E/H/B/D and their split-field histories.  The original
complex family nevertheless refuses every registered polarization because its
shared predicate also owns ordinary ``update_E``, where the source changes from
``D`` to ordered ``D - P0 - ...``.  Reusing that broad refusal would leave a
complex dispersive composer with only E/ADE device plans and silently force B/D/H
through the array path.

This module narrows that exception without weakening it.  It exposes the already
byte-gated complex curl and H bodies through a view that hides only the two
polarization-presence flags.  Each prospective B/D/H plan is then conjoined with
the full complex dispersive-E predicate.  Thus it is selectable only when the
same live, supported polarization state is also eligible for the pole-aware E
plan and ADE successor; an unsupported pole, conductivity, tensor row, or altered
geometry refuses the entire product by name.

No production dispatch is enabled here.  Registration affects the experimental
Metal composer only.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import complex_dispersive_update_e as constitutive
from . import complex_fields as base
from . import shaders
from .device import Residency

FAMILY = "complex_dispersive_spine"

__all__ = [
    "FAMILY",
    "metal_complex_dispersive_pml_curl_coverage",
    "metal_complex_dispersive_pml_magnetic_coverage",
    "plan_metal_complex_dispersive_pml_curl",
    "plan_metal_complex_dispersive_pml_magnetic",
]


class _PolarizationScopeView:
    """Hide only the E-source-changing fact from a B/D/H predicate.

    The view deliberately leaves every pole object, material flag, conductivity
    table, geometry property, and volume reachable through the underlying fields.
    The companion E predicate validates the poles themselves.  Masking more than
    the two presence flags would turn a reusable arithmetic body into an unsafe
    general-purpose admission bypass.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        if name == "polarizations":
            return ()
        if name == "has_polarizations":
            return False
        return getattr(object.__getattribute__(self, "_fields"), name)


def _merge(shared: Coverage, companion: Coverage) -> Coverage:
    """Require both local B/D/H arithmetic and the full pole-aware E product."""
    reasons = tuple(dict.fromkeys(shared.reasons + companion.reasons))
    return Coverage(shared.covered and companion.covered, reasons)


def _companion(fields: Any, pml: Any, residency: Any, probe: Any) -> Coverage:
    return constitutive.metal_complex_dispersive_e_coverage(
        fields, pml, residency, probe)


def metal_complex_dispersive_pml_curl_coverage(
        fields: Any, pml: Any, sub_step: str, residency: Any = None,
        probe: Any = None) -> Coverage:
    """Coverage for a complex PML curl beside a live pole-aware E/ADE pair."""
    if sub_step not in base.CURL_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(base.CURL_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    shared = base.complex_pml_curl_coverage(
        _PolarizationScopeView(fields), pml, sub_step, residency, probe)
    return _merge(shared, _companion(fields, pml, residency, probe))


def metal_complex_dispersive_pml_magnetic_coverage(
        fields: Any, pml: Any, residency: Any = None,
        probe: Any = None) -> Coverage:
    """Coverage for ``update_H`` beside a live pole-aware E/ADE pair."""
    shared = base.complex_constitutive_coverage(
        _PolarizationScopeView(fields), pml, "H", residency, probe)
    return _merge(shared, _companion(fields, pml, residency, probe))


def plan_metal_complex_dispersive_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,), probe: Any = None,
        ) -> Optional[Any]:
    """Build one B/D plan only when the complete complex dispersive product fits."""
    if not metal_complex_dispersive_pml_curl_coverage(
            fields, pml, sub_step, residency, probe).covered:
        return None
    return base.plan_complex_pml_curl(
        _PolarizationScopeView(fields), pml, sub_step, residency,
        contract_variants, probe)


def plan_metal_complex_dispersive_pml_magnetic(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,), probe: Any = None,
        ) -> Optional[Any]:
    """Build ``update_H`` only when the complete complex dispersive product fits."""
    if not metal_complex_dispersive_pml_magnetic_coverage(
            fields, pml, residency, probe).covered:
        return None
    return base.plan_complex_constitutive(
        _PolarizationScopeView(fields), pml, "H", residency,
        contract_variants, probe)


def _curl_coverage(context: Any, slot: str) -> Coverage:
    return metal_complex_dispersive_pml_curl_coverage(
        context.fields, context.pml, slot, context.residency,
        context.extra.get("complex_probe"))


def _curl_plan(context: Any, slot: str) -> Optional[Any]:
    return plan_metal_complex_dispersive_pml_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants, context.extra.get("complex_probe"))


def _magnetic_coverage(context: Any, slot: str) -> Coverage:
    if slot != "update_H":
        return Coverage(False, (f"complex dispersive PML magnetic cannot fill {slot}",))
    return metal_complex_dispersive_pml_magnetic_coverage(
        context.fields, context.pml, context.residency,
        context.extra.get("complex_probe"))


def _magnetic_plan(context: Any, slot: str) -> Optional[Any]:
    if slot != "update_H":
        return None
    return plan_metal_complex_dispersive_pml_magnetic(
        context.fields, context.pml, context.residency,
        context.contract_variants, context.extra.get("complex_probe"))


def register_arms() -> Tuple[Any, ...]:
    """Register the three pole-aware spine slots as one disjoint product."""
    from . import arms  # noqa: PLC0415

    curls = tuple(
        arms.register(FAMILY, slot, "complex dispersive PML curl", _curl_coverage,
                      _curl_plan, prefix="complex dispersive PML curl: ",
                      noun="complex PML dispersive curl", wired=True)
        for slot in base.CURL_SUB_STEPS)
    magnetic = arms.register(
        FAMILY, "update_H", "complex dispersive PML magnetic", _magnetic_coverage,
        _magnetic_plan, prefix="complex dispersive PML magnetic: ",
        noun="complex PML dispersive magnetic constitutive", wired=True)
    return curls + (magnetic,)


ARMS = register_arms()
