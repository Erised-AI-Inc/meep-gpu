"""Live Metal planner census for one lifted MEEP simulation.

The MEEP corpus is a capability corpus: a slot is covered only when the live
Metal planner selects a plan for that exact lifted configuration.  This module
does not recreate predicates or turn a near match into a coverage claim.  The
separate Metal gates establish arithmetic correctness; this census establishes
which of those gated arms the production planner can actually reach.
"""

from __future__ import annotations

import json
from typing import Any


CORE_SLOTS = ("step_B", "step_D", "update_H", "update_E")


def slot_names(fields: Any) -> tuple[str, ...]:
    """Return the standing 4-or-5 slot denominator for this lifted row."""
    polarizations = tuple(getattr(fields, "polarizations", ()) or ())
    return CORE_SLOTS + (("update_P",) if polarizations else ())


def _configuration(driver: Any) -> dict[str, Any]:
    fields, pml, grid = driver.fields, driver.pml, driver.grid

    def ask(name: str, *args: Any) -> Any:
        attribute = getattr(grid, name, None)
        if not callable(attribute):
            return attribute
        try:
            return attribute(*args)
        except Exception:  # noqa: BLE001 - configuration must not stop a census
            return None

    return {
        "shape": [int(value) for value in getattr(grid, "shape", ()) or ()],
        "pml_active": bool(getattr(pml, "is_active", False)),
        "complex": bool(getattr(fields, "force_complex_fields", False)),
        "folded": [bool(ask("is_mirrored", axis)) for axis in range(3)],
        "metallic": [bool(ask("is_metallic", axis)) for axis in range(3)],
        "cylindrical": bool(getattr(grid, "cylindrical", False)),
        "beta": float(getattr(grid, "beta", 0.0) or 0.0),
        "bfast": bool(getattr(grid, "bfast_active", False)),
        "offdiagonal": bool(getattr(fields, "has_offdiagonal_epsilon", False)),
        "nonlinear": bool(getattr(fields, "has_nonlinearity", False)),
        "conductivity": bool(getattr(fields, "has_conductivity", False)),
        "stores_e": bool(getattr(fields, "stores_E", False)),
        "polarizations": len(tuple(getattr(fields, "polarizations", ()) or ())),
    }


def evaluate(driver: Any) -> dict[str, Any]:
    """Ask the unmodified Metal planner which slots it selects for ``driver``."""
    from meep_gpu.metal_kernels import device, launch, subnormal  # noqa: PLC0415

    fields, pml = driver.fields, driver.pml
    sources = tuple(getattr(driver, "_sources", ()) or ())
    slots = slot_names(fields)
    record: dict[str, Any] = {
        "slots": list(slots),
        "configuration": _configuration(driver),
        "subnormal_policy": subnormal.mps_policy_report(),
    }
    try:
        plan = launch.plan_step(fields, pml, residency=device.Residency(),
                                sources=sources)
    except Exception as exc:  # noqa: BLE001 - report a row-level planner refusal
        record.update({
            "planner_error": f"{type(exc).__name__}: {exc}"[:600],
            "selected": {},
            "replaces": [],
            "reasons": {},
            "residency_covered": False,
        })
        return record

    record.update({
        "planner_error": None,
        "selected": {slot: plan.selected[slot]
                     for slot in slots if slot in plan.selected},
        "replaces": [slot for slot in plan.replaces if slot in slots],
        "reasons": {slot: list(reasons)[:8]
                    for slot, reasons in plan.reasons.items() if slot in slots},
        "residency_covered": bool(plan.residency and plan.residency.covered),
        "residency_reasons": list(
            getattr(plan.residency, "reasons", ()) or ())[:8],
    })
    return record


def summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate the explicit coverage status for already measured rows."""
    measured = [record for record in records if record.get("measured")]
    slots = [slot for record in measured for slot in record.get("slots", ())]
    selected = [
        slot for record in measured for slot in record.get("slots", ())
        if slot in (record.get("selected") or {})
    ]
    return {
        "rows": len(records),
        "measured_rows": len(measured),
        "slots": len(slots),
        "selected_slots": len(selected),
        "unselected_slots": len(slots) - len(selected),
        "expected_slots": 759,
        "slot_denominator_matches": len(slots) == 759,
    }


def json_line(record: dict[str, Any]) -> str:
    """Serialize a row deterministically for append-only campaign output."""
    return json.dumps(record, sort_keys=True, default=str) + "\n"
