"""Complex/Bloch lossless no-PML curl sub-steps for PyTorch/MPS.

The body is the all-lossless specialization of
:mod:`.complex_no_pml_conductive`, not a second transcription of the complex
Yee stencil.  That is intentional: both products must make exactly the same
wrap-plane phase rotation, ownership decision, and coefficient-left curl
multiply.  Their only numerical distinction is the tail.  Here every tail is
specialized to ``target -= curl``; the six conductivity buffer positions remain
in the ABI but are dead in the emitted body.  A later native gate may replace
that conservative ABI with a narrower one only after it has compared the two
specializations' device bytes.

The family is wired into the experimental Metal composer. Its native MPS gate
establishes the B/D sub-step bytes and the canonical derived-H mirror binding;
the whole-step gate separately establishes the complete driver-residency path.
Production dispatch and performance remain separate obligations.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.coverage import CURL_TARGETS, Coverage, _boundary_kinds
from . import shaders
from .complex_fields import (
    _complex_mirror,
    _complex_volume_reasons,
    bloch_phase_table,
    expansion_from_probe,
    load_expansion_probe,
    phase_arguments,
)
from .complex_no_pml_conductive import (
    SOURCE_ACCESSOR,
    SUB_STEPS,
    MetalComplexConductiveNoPmlCurlPlan,
    _base_reasons,
    _source_binding,
    complex_conductive_no_pml_curl_source,
    compile_complex_conductive_no_pml_curl,
)
from .device import Residency

FAMILY = "complex_no_pml_curl"

__all__ = [
    "FAMILY",
    "SUB_STEPS",
    "MetalComplexNoPmlCurlPlan",
    "complex_no_pml_curl_source",
    "compile_complex_no_pml_curl",
    "metal_complex_no_pml_curl_coverage",
    "plan_metal_complex_no_pml_curl",
]


class MetalComplexNoPmlCurlPlan(MetalComplexConductiveNoPmlCurlPlan):
    """The lossless ABI-specialized form of the complex no-PML curl plan."""


def complex_no_pml_curl_source(
        codes: Sequence[int], backward: bool, phased: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> str:
    """Emit the complex curl with all three conductivity tails compiled out."""
    return complex_conductive_no_pml_curl_source(
        codes, backward, phased, (False, False, False), expansion, contract)


def compile_complex_no_pml_curl(
        codes: Sequence[int], backward: bool, phased: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """Compile the all-lossless specialization of the shared complex stencil."""
    return compile_complex_conductive_no_pml_curl(
        codes, backward, phased, (False, False, False), expansion, contract)


def metal_complex_no_pml_curl_coverage(
        fields: Any, pml: Any, sub_step: str, residency: Any = None,
        probe: Any = None) -> Coverage:
    """Whether one complex/Bloch lossless curl sub-step is covered."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    reasons: List[str] = _base_reasons(fields, pml, grid, residency, probe)
    spec = SUB_STEPS[sub_step]
    shape = tuple(getattr(grid, "shape", ()))
    for name in spec["targets"]:
        array = getattr(fields, name, None)
        if array is None:
            reasons.append(f"{name} is not allocated")
        else:
            reasons.extend(_complex_volume_reasons(name, array, shape))
    accessor_name = SOURCE_ACCESSOR[sub_step]
    accessor = getattr(fields, accessor_name, None)
    if not callable(accessor):
        reasons.append(f"fields.{accessor_name} is missing or not callable")
    else:
        for name in spec["sources"]:
            try:
                source = accessor(name)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"fields.{accessor_name}({name!r}) raised {exc!r}")
                continue
            if source is None:
                reasons.append(f"{name} is not served by fields.{accessor_name}")
            else:
                reasons.extend(_complex_volume_reasons(name, source, shape))
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for")
    else:
        for target in CURL_TARGETS:
            try:
                if reader(target) is not None:
                    reasons.append(
                        f"conductivity is installed on {target}; "
                        "the conductive complex curl owns that product")
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def _functions(codes: Sequence[int], backward: bool, phased: Sequence[int],
               expansion: str, variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_complex_no_pml_curl(
        codes, backward, phased, expansion, mode) for mode in variants}


def plan_metal_complex_no_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,), probe: Any = None,
        functions: Optional[Dict[str, Any]] = None,
        ) -> Optional[MetalComplexNoPmlCurlPlan]:
    """Build a covered lossless complex curl plan, otherwise return ``None``."""
    if not metal_complex_no_pml_curl_coverage(
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
    # The source specialization has no coefficient reads.  Binding the target
    # placeholders keeps one stable, gateable ABI with its conductive sibling.
    dead_coefficients = tuple(targets)
    selected = functions if functions is not None else _functions(
        codes, bool(spec["backward"]), phased, expansion, contract_variants)
    return MetalComplexNoPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, phased, phase_values,
        (False, False, False), expansion, residency, targets, sources,
        dead_coefficients, dead_coefficients, selected,
        tuple(spec["targets"]) + tuple(name for name, _ in source_bindings),
    )


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return metal_complex_no_pml_curl_coverage(
        context.fields, context.pml, slot, context.residency,
        context.extra.get("probe"))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexNoPmlCurlPlan]:
    return plan_metal_complex_no_pml_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants, context.extra.get("probe"))


def _arm_gate(context: Any) -> bool:
    return not bool(getattr(context.pml, "is_active", False))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return tuple(arms.register(
        family=FAMILY, slot=slot, label="complex no-PML curl",
        coverage=_arm_coverage, plan=_arm_plan,
        prefix="complex no-PML curl: ", noun="complex no-PML curl",
        gate=_arm_gate, wired=True,
    ) for slot in SUB_STEPS)


ARMS = register_arms()
