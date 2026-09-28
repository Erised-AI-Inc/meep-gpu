#!/usr/bin/env python3
"""Native MPS byte gate for real conductive active-PML B/D curl sub-steps.

This is the active-absorber counterpart of the no-PML conductive gate.  It tests
the branch that owns both the split-field ``fu`` recurrence and MEEP's per-target
``f_cond`` recurrence.  The gate is deliberately narrower than a complete driver
step: its claim is the B/D arithmetic and persistent mirror continuity, not
production dispatch or cross-sub-step fusion.

Every product compares float32 *words*, never tolerances, against ``stepping``.
The mutations are compiled from altered Metal source and must visibly diverge;
the refusal leg protects the inverse predicate boundary so this family cannot
silently overlap a lossless ordinary-PML plan.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import conductive_pml as family  # noqa: E402
from meep_gpu.metal_kernels import launch, shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import Residency, compile_source  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402


def _words(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value)
    if array.dtype != np.float32:
        raise TypeError(f"expected float32, got {array.dtype}")
    return array.reshape(-1).view(np.uint32)


def _differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


class _Tally:
    """The gate headline is exact words, not a misleading row count."""

    def __init__(self) -> None:
        self.compared = 0
        self.differing = 0

    def compare(self, actual: Any, reference: Any) -> int:
        self.compared += int(_words(actual).size)
        differing = _differing(actual, reference)
        self.differing += differing
        return differing


TALLY = _Tally()


def _sigma(shape: Sequence[int]) -> np.ndarray:
    """A nonconstant f32 loss volume: all four PML cases are reachable."""
    line = np.linspace(np.float32(0.06), np.float32(0.22), int(shape[0]),
                       dtype=np.float32)[:, None, None]
    return np.broadcast_to(line, tuple(shape)).copy()


def build(*, sides: Sequence[str], mixed: bool, boundaries: Sequence[str],
          seed: int) -> Tuple[Fields, PML]:
    """One real active-PML state with selected B and/or D conductivity."""
    grid = Grid(resolution=8.0, cell_size=(1.25, 1.0, 0.75), courant=0.35,
                boundaries=tuple(boundaries), xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = np.random.default_rng(seed)
    for name in (
            "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
            "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx",
            "fu_Dy", "fu_Dz"):
        getattr(fields, name)[...] = rng.uniform(
            -0.3, 0.3, grid.shape).astype(np.float32)

    sigma = _sigma(grid.shape)
    if "B" in sides:
        fields.set_b_conductivity({
            "Bx": sigma if not mixed else sigma,
            "By": sigma if not mixed else None,
            "Bz": sigma if not mixed else None,
        })
    if "D" in sides:
        fields.set_d_conductivity({
            "Dx": sigma if not mixed else None,
            "Dy": sigma if not mixed else None,
            "Dz": sigma if not mixed else sigma,
        })
    for target in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        history = getattr(fields, "f_cond_" + target, None)
        if history is not None:
            history[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, PML(grid=grid, thickness=1)


def _watched(fields: Fields, sub_step: str) -> Tuple[str, ...]:
    names: List[str] = []
    for target in family.CONDUCTIVE_SUB_STEPS[sub_step]:
        names.extend((target, "fu_" + target))
        if getattr(fields, "f_cond_" + target, None) is not None:
            names.append("f_cond_" + target)
    return tuple(names)


def _sub_steps(sides: Sequence[str]) -> Tuple[str, ...]:
    return tuple(slot for side, slot in (("B", "step_B"), ("D", "step_D"))
                 if side in sides)


def execute(*, sides: Sequence[str], mixed: bool, boundaries: Sequence[str],
            seed: int, cycles: int,
            functions: Mapping[str, Mapping[str, Any]] | None = None,
            record_tally: bool = True,
            ) -> Dict[str, Any]:
    """Run the selected conductive sub-steps over one shared MPS residency."""
    reference, reference_pml = build(sides=sides, mixed=mixed,
                                     boundaries=boundaries, seed=seed)
    actual, actual_pml = build(sides=sides, mixed=mixed,
                                boundaries=boundaries, seed=seed)
    slots = _sub_steps(sides)
    residency = Residency()
    plans = {
        slot: family.plan_metal_conductive_pml_curl(
            actual, actual_pml, slot, residency,
            functions=None if functions is None else functions.get(slot))
        for slot in slots
    }
    assert all(plan is not None for plan in plans.values()), (
        "declared conductive product was refused", slots, plans)
    watched = {slot: _watched(actual, slot) for slot in slots}
    before = {name: np.array(getattr(actual, name), copy=True)
              for names in watched.values() for name in names}
    rows: List[Dict[str, Any]] = []
    residency.sync_in()
    for cycle in range(1, cycles + 1):
        for slot in slots:
            plan = plans[slot]
            assert plan is not None
            plan.run()
            import torch  # noqa: PLC0415

            torch.mps.synchronize()
            getattr(stepping, slot)(reference, reference_pml)
            residency.sync_out(watched[slot])
            compare = TALLY.compare if record_tally else _differing
            differing = sum(compare(getattr(actual, name), getattr(reference, name))
                            for name in watched[slot])
            moved = sum(_differing(before[name], getattr(reference, name))
                        for name in watched[slot])
            kit.assert_moved(moved, f"{sides}/{slot}/cycle-{cycle}")
            row = {
                "cycle": cycle,
                "slot": slot,
                "differing_words": differing,
                "moved_words": moved,
                "words_compared": int(sum(_words(getattr(actual, name)).size
                                          for name in watched[slot])),
                "reference_subnormal_words": int(sum(
                    subnormal.census(getattr(reference, name))
                    for name in watched[slot])),
                "launches": plan.launches,
                "conductive": list(plan.conductive),
            }
            rows.append(row)
            if differing or row["reference_subnormal_words"]:
                return {"passed": False, "rows": rows,
                        "reason": "different or subnormal reference words"}
    residency.verify()
    launches = {slot: plans[slot].launches for slot in slots if plans[slot] is not None}
    return {
        "passed": (all(row["differing_words"] == 0 for row in rows)
                   and all(count == cycles for count in launches.values())),
        "rows": rows,
        "launches": launches,
        "residency_mirrors": len(residency.names),
    }


def _mutated_sources() -> Dict[str, str]:
    """Each source edit has a live effect on Bx in the selected product."""
    source = family.conductive_pml_curl_source(
        (1, 0, 1), False, (True, True, True))
    if "cf0[ii]" not in source or "ci0[ii]" not in source:
        raise LookupError("conductive coefficient operands")
    # ``cf0`` / ``ci0`` occur in all four PML branches.  Swap every occurrence:
    # a one-occurrence mutation would leave three branches compiling against an
    # undefined temporary and would test Metal's diagnostics, not the recurrence.
    swapped = source.replace("cf0[ii]", "TEMP_CF0").replace(
        "ci0[ii]", "cf0[ii]").replace("TEMP_CF0", "ci0[ii]")
    return {
        "condfac_condinv_swapped": swapped,
        "conductivity_history_store_dropped": kit.needle(
            source,
            "c0[ii] = c0_new; u0[ii] = u0_new; f0[ii] = f0_new;",
            "u0[ii] = u0_new; f0[ii] = f0_new;",
        ),
        "curl_grouping_flattened": kit.needle(
            source,
            "float curl0 = dtdx * ((c_y - c) + (b - b_z));",
            "float curl0 = dtdx * (c_y - c + b - b_z);",
        ),
        "first_pml_branch_disabled": kit.needle(
            source,
            "bool dsig0 = (km_y != 1.0f) || (si_y != 1.0f);",
            "bool dsig0 = false;",
        ),
    }


def leg_products(payload: Dict[str, Any], out: str) -> None:
    """Full, B-only, D-only, and per-target split conductivity products."""
    products: Tuple[Tuple[str, Tuple[str, ...], bool, Tuple[str, ...]], ...] = (
        ("two_sided_periodic", ("B", "D"), False,
         ("periodic", "periodic", "periodic")),
        ("b_only_metallic", ("B",), False,
         ("metallic", "periodic", "metallic")),
        ("d_only_mixed", ("D",), False,
         ("periodic", "metallic", "periodic")),
        ("per_target_mixed", ("B", "D"), True,
         ("metallic", "periodic", "metallic")),
    )
    rows: List[Dict[str, Any]] = []
    payload["legs"]["products"] = rows
    for index, (label, sides, mixed, boundaries) in enumerate(products, 1):
        result = execute(sides=sides, mixed=mixed, boundaries=boundaries,
                         seed=30100 + index, cycles=4)
        row = {"label": label, "sides": list(sides), "mixed": mixed,
               "boundaries": list(boundaries), **result}
        rows.append(row)
        payload["legs"]["products"] = rows
        kit.save(payload, out)
        differences = sum(item["differing_words"] for item in result["rows"])
        moved = sum(item["moved_words"] for item in result["rows"])
        compared = sum(item["words_compared"] for item in result["rows"])
        kit.log(f"[product] {index}/{len(products)} {label:<22} "
                f"moved={moved:<7} differing={differences:<4} words={compared}")
        assert result["passed"], row


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Faults in the active-PML conductive recurrence must be byte-visible."""
    rows: List[Dict[str, Any]] = []
    payload["legs"]["mutations"] = rows
    for index, (label, source) in enumerate(_mutated_sources().items(), 1):
        function = compile_source(source).conductive_pml_curl_step
        result = execute(
            sides=("B",), mixed=False, boundaries=("metallic", "periodic", "metallic"),
            seed=30200 + index, cycles=2,
            functions={"step_B": {shaders.CONTRACT_OFF: function}},
            record_tally=False,
        )
        differing = sum(item["differing_words"] for item in result["rows"])
        # ``execute`` stops at the *first* mismatch so it preserves the smallest
        # discriminating counterexample.  The successful path has ``launches``;
        # the caught path records it in its final per-cycle row instead.
        launches = result.get("launches", {}).get(
            "step_B", result["rows"][-1]["launches"])
        row = {"label": label, "differing_words": differing,
               "launches": launches, "caught": differing > 0,
               "result": result}
        rows.append(row)
        payload["legs"]["mutations"] = rows
        kit.save(payload, out)
        kit.log(f"[mutation] {index}/4 {label:<34} launches={launches} "
                f"differing={differing}")
        assert launches >= 1, row
        assert differing > 0, row


def leg_refusals(payload: Dict[str, Any], out: str) -> None:
    """The conductive arm owns only its target side and rejects the incumbent."""
    fields, pml = build(sides=("B",), mixed=False,
                        boundaries=("periodic", "periodic", "periodic"), seed=30301)
    b = family.metal_conductive_pml_curl_coverage(fields, pml, "step_B", Residency())
    d = family.metal_conductive_pml_curl_coverage(fields, pml, "step_D", Residency())
    ordinary = launch.plan_pml_curl(fields, pml, "step_B", Residency())
    inactive = PML(grid=fields.grid, thickness=0)
    inactive_verdict = family.metal_conductive_pml_curl_coverage(
        fields, inactive, "step_B", Residency())
    row = {
        "b_covered": b.covered,
        "d_reasons": list(d.reasons),
        "ordinary_plan": ordinary is not None,
        "inactive_reasons": list(inactive_verdict.reasons),
    }
    payload["legs"]["refusals"] = row
    kit.save(payload, out)
    kit.log("[refusal] B admitted; D-only, ordinary-PML overlap, and inactive-PML "
            "controls refused")
    assert b.covered, b.reasons
    assert not d.covered and any("no step_D target" in reason for reason in d.reasons)
    assert ordinary is None
    assert not inactive_verdict.covered
    assert any("active PML" in reason for reason in inactive_verdict.reasons)


def main(argv: Sequence[str]) -> int:
    parser = kit.argument_parser(__doc__ or "")
    args = parser.parse_args(list(argv))
    out = os.path.abspath(args.out)
    started = time.time()
    payload: Dict[str, Any] = {
        "gate": "metal_conductive_pml",
        "family": family.FAMILY,
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
    }
    kit.save(payload, out)
    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device available")
    reasons.extend(subnormal.mps_policy_reasons())
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    kit.provenance(
        os.path.dirname(out),
        {
            "conductive_pml.py": str(Path(family.__file__).resolve()),
            "stepping.py": str((API_ROOT / "meep_gpu/stepping.py").resolve()),
            "gate": str(Path(__file__).resolve()),
        },
        kernel_sources={
            f"{codes}/{backward}/{conductive}": family.conductive_pml_curl_source(
                codes, backward, conductive)
            for codes, backward, conductive in (
                ((0, 0, 0), False, (True, True, True)),
                ((1, 0, 1), False, (True, False, False)),
                ((0, 1, 0), True, (False, False, True)),
            )
        },
        name="provenance_conductive_pml.json",
    )
    ran = kit.run_legs(
        (("products", leg_products), ("mutations", leg_mutations),
         ("refusals", leg_refusals)),
        payload, out, kit.wanted_legs(args.legs))
    payload["totals"] = {
        "uint32_words_compared": TALLY.compared,
        "uint32_words_differing": TALLY.differing,
    }
    kit.save(payload, out)
    return kit.summarize(
        payload, out,
        claim=("the real conductive active-PML B/D curl kernels reproduce the "
               "array path word for word across side-specific and per-target "
               "conductivity products, with the PML and conductivity histories "
               "preserved across launches"),
        scope=("real float32 Cartesian fields, active PML, periodic/metallic "
               "boundaries, no fold, no Bloch phase, no poles, no nonlinear "
               "material, and no driver dispatch claim"),
        stated_weakness=("behavioural MPS evidence only: compile_shader exposes "
                         "no generated AIR or ISA; this is a B/D sub-step gate, "
                         "not a complete driver or throughput measurement"),
        started=started, legs_run=ran, compared=TALLY.compared,
        certified=TALLY.differing == 0,
        extra={"products": len(payload["legs"].get("products", [])),
               "mutations": len(payload["legs"].get("mutations", []))},
    )


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
