"""Certify the public ``plan_step`` route for the conductive-PML curl product.

The standalone conductive gate already owns the kernel product, its four branch
families, mutations and recurrence budgets. This probe owns the integration seam:
one real ``2d_cond_pml`` simulation must select the ordinary B curl, conductive D
curl and both ordinary constitutive products, while ordinary, dispersive and
no-PML controls retain their existing product classes. Every mutable field and
auxiliary is compared as uint32 after every complete driver step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for path in (HERE, API_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

CASES = (
    ("2d_cond_pml", {"res": 40, "n": 16}, 12,
     {"step_B": "PmlCurlPlan", "update_H": "ConstitutivePlan",
      "step_D": "ConductivePmlCurlPlan", "update_E": "ConstitutivePlan"}),
    ("2d_pml", {"res": 40, "n": 16}, 6,
     {"step_B": "PmlCurlPlan", "update_H": "ConstitutivePlan",
      "step_D": "PmlCurlPlan", "update_E": "ConstitutivePlan"}),
    ("2d_dispersive", {"res": 40, "n": 16}, 6,
     {"step_B": "PmlCurlPlan", "update_H": "ConstitutivePlan",
      "step_D": "PmlCurlPlan", "update_E": "DispersiveConstitutivePlan",
      "update_P": "list"}),
    # The no-PML control. Since the null-constitutive family was wired
    # (planner recert 2026-08-14), the certified selection on a plain grid is
    # the curl pair PLUS a null plan on each constitutive sub-step: under an
    # inactive absorber stepping.update_H/update_E return before reading an
    # array, and the planner now says so instead of leaving the slots empty.
    # This pins the COMPLETE new shape — a kernel displacing a null here, or a
    # null displacing a curl, both fail this dict.
    ("2d_plain", {"res": 40, "n": 16}, 6,
     {"step_B": "PlainCurlPlan", "step_D": "PlainCurlPlan",
      "update_H": "NullConstitutivePlan", "update_E": "NullConstitutivePlan"}),
)

STATE_NAMES = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz",
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
    "f_cond_Bx", "f_cond_By", "f_cond_Bz",
    "f_cond_Dx", "f_cond_Dy", "f_cond_Dz",
)


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def state(driver) -> Dict[str, Any]:
    arrays = {name: getattr(driver.fields, name) for name in STATE_NAMES
              if getattr(driver.fields, name, None) is not None}
    for index, pole in enumerate(driver.fields.polarizations):
        for component in pole.driven():
            arrays[f"P{index}_{component}"] = pole.P[component]
            arrays[f"Pprev{index}_{component}"] = pole.P_prev[component]
    return arrays


def compare(cp, left, right) -> Dict[str, Any]:
    a = np.ascontiguousarray(cp.asnumpy(left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(cp.asnumpy(right)).view(np.uint32).ravel()
    return {
        "bit_identical": bool(np.array_equal(a, b)),
        "differing_floats": int(np.count_nonzero(a != b)),
        "total_floats": int(a.size),
    }


def plan_types(plan) -> Dict[str, str]:
    return {
        name: ("list" if isinstance(entry, (list, tuple)) else type(entry).__name__)
        for name, entry in plan.plans.items()
    }


def run_case(cp, name: str, kwargs: Dict[str, Any], steps: int,
             expected: Dict[str, str]) -> Dict[str, Any]:
    import cases  # noqa: PLC0415

    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference = lift_simulation(cases.BUILDERS[name](**kwargs), prefer_gpu=True, gpu_id=0)
    candidate = lift_simulation(cases.BUILDERS[name](**kwargs), prefer_gpu=True, gpu_id=0)
    names = ("step_B", "update_H", "step_D", "update_E", "update_P")
    originals = {sub_step: getattr(driver_module, sub_step) for sub_step in names}
    try:
        plan = plan_step(candidate.fields, candidate.pml)
        selected = plan_types(plan)
        row: Dict[str, Any] = {
            "case": name,
            "shape": [int(value) for value in candidate.shape],
            "steps": steps,
            "replaces": list(plan.replaces),
            "selected": selected,
            "expected": expected,
            "refusals": {key: list(value) for key, value in plan.reasons.items()},
            "per_step": [],
        }
        if selected != expected:
            raise AssertionError(
                f"{name}: central product classes differ: selected={selected}, "
                f"expected={expected}, refusals={plan.reasons}")

        def replacement(sub_step):
            def wrapper(fields, pml=None):
                entry = plan.plans.get(sub_step) if fields is candidate.fields else None
                if entry is None:
                    return originals[sub_step](fields, pml)
                if sub_step == "update_P":
                    for pole_plan in entry:
                        pole_plan.run(fields.drive_field)
                else:
                    entry.run()
                return None
            return wrapper

        for sub_step in names:
            setattr(driver_module, sub_step, replacement(sub_step))

        started = time.time()
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            cp.cuda.runtime.deviceSynchronize()
            reference_state = state(reference)
            candidate_state = state(candidate)
            if set(reference_state) != set(candidate_state):
                raise AssertionError(
                    f"{name} step {step}: state inventory differs: "
                    f"reference={sorted(reference_state)}, candidate={sorted(candidate_state)}")
            parts = {key: compare(cp, candidate_state[key], reference_state[key])
                     for key in sorted(reference_state)}
            same = all(part["bit_identical"] for part in parts.values())
            point = {
                "step": step,
                "bit_identical": same,
                "differing_floats": sum(part["differing_floats"]
                                          for part in parts.values()),
                "total_floats": sum(part["total_floats"] for part in parts.values()),
                "differing_arrays": sorted(
                    key for key, part in parts.items() if not part["bit_identical"]),
            }
            row["per_step"].append(point)
            log(f"case {name} step {step}/{steps} identical={same} "
                f"ndiff={point['differing_floats']} ({time.time() - started:.1f} s)")
            if not same:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        row["bit_identical"] = True
        row["first_divergent_step"] = None
        return row
    finally:
        for sub_step, function in originals.items():
            setattr(driver_module, sub_step, function)
        reference.close()
        candidate.close()
        cp.get_default_memory_pool().free_all_blocks()


def environment(cp) -> Dict[str, Any]:
    import triton  # noqa: PLC0415

    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "triton": triton.__version__,
        "device": properties["name"].decode(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    payload: Dict[str, Any] = {"environment": environment(cp), "cases": []}
    save(payload, args.out)
    for name, kwargs, steps, expected in CASES:
        log(f"starting {name}: expected={expected}")
        payload["cases"].append(run_case(cp, name, kwargs, steps, expected))
        save(payload, args.out)
    payload["summary"] = {
        "status": "passed",
        "cases_exact": f"{len(payload['cases'])}/{len(CASES)}",
        "complete_steps_exact": f"{sum(row['steps'] for row in payload['cases'])}/"
                                f"{sum(row['steps'] for row in payload['cases'])}",
        "conductive_steps_exact": "12/12",
        "controls": [row["case"] for row in payload["cases"] if row["case"] != "2d_cond_pml"],
    }
    save(payload, args.out)
    log(f"COMPOSITION GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
