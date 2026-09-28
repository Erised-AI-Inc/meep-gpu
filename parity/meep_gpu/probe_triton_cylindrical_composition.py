"""Complete-driver CUDA gate for strict real m=0 Dcyl Triton composition.

The standalone cylindrical gate certifies the radial-prefix curl kernel and the
ordinary constitutive kernel separately.  This probe certifies their central-plan
composition through real ``FdtdDriver.step`` calls.  It is intentionally limited to
the predicate's strict m=0, real-storage, r-high/z-PML configuration; nonzero m,
dispersion and complex storage are not silently broadened here.

Pair fusion IS now part of what this certifies.  Until 2026-09-02 every case
selected the unfused curl+constitutive composition and this probe pinned that one
shape.  The dispatch expansion that put ``fill_B``/``fill_D`` into
``fastpath.DRIVER_SLOTS`` released the cylindrical real fused pair, so the arms
fuse here now; each case is pinned to the exact composition it produces, which is
a stronger claim than the unfused shape it replaced.  See
:data:`EXPECTED_BY_CASE`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)


# name, cell_size, resolution, optional source, complete steps
CASES = (
    ("r_high_z_pml", (2.4, 0.0, 3.2), 10.0, None, 8),
    ("r_high_z_pml_tall", (1.8, 0.0, 4.0), 10.0, None, 8),
    ("r_high_z_pml_ez_axis", (2.4, 0.0, 3.2), 10.0, "Ez", 10),
)

#: The composition ``plan_step`` produces, pinned PER CASE.
#:
#: Until 2026-09-02 every case selected the unfused shape (``CylindricalCurlPlan``
#: + ``ConstitutivePlan``) and one table covered all three. The dispatch expansion
#: that put ``fill_B``/``fill_D`` into :data:`fastpath.DRIVER_SLOTS` released the
#: cylindrical real fused pair, so the arms now fuse: the curl slot carries the
#: fused pair plan and its constitutive partner is the ``NoopPlan`` that pair
#: absorbed. Asserting the FUSED composition is a stronger claim than asserting
#: the unfused one, so each case is pinned to the exact composition it produces —
#: never to "either shape is acceptable".
#:
#: The Ez-source case does NOT fuse its electric arm: an electric source lands
#: inside the ``step_D``/``update_E`` seam, so that arm routes through the
#: deposit-repair bracket instead. Pinning the two shapes separately is what keeps
#: that distinction asserted rather than averaged away.
EXPECTED_BY_CASE = {
    "r_high_z_pml": {
        "step_B": "CylindricalRealFusedMagneticPairPlan",
        "update_H": "NoopPlan",
        "step_D": "CylindricalRealFusedElectricPairPlan",
        "update_E": "NoopPlan",
    },
    "r_high_z_pml_tall": {
        "step_B": "CylindricalRealFusedMagneticPairPlan",
        "update_H": "NoopPlan",
        "step_D": "CylindricalRealFusedElectricPairPlan",
        "update_E": "NoopPlan",
    },
    "r_high_z_pml_ez_axis": {
        "step_B": "CylindricalRealFusedMagneticPairPlan",
        "update_H": "NoopPlan",
        "step_D": "LeadingRepairPlan",
        "update_E": "TrailingRepairPlan",
    },
}

#: The generic cylindrical fusion routes, which must refuse BY NAME for a
#: cylindrical reason. Pinned as names so a route that disappears fails here
#: rather than satisfying an "all(...)" over an empty set vacuously.
CYLINDRICAL_REFUSALS = ("fused_pair_B", "fused_pair_D", "fused_pair_dispersive_D")

if {case[0] for case in CASES} != set(EXPECTED_BY_CASE):
    raise SystemExit(
        "every case must pin its own composition; unpinned or stale: %s"
        % sorted({case[0] for case in CASES} ^ set(EXPECTED_BY_CASE)))
STATE_NAMES = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz",
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")


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


def source(component: str) -> Dict[str, Any]:
    return {
        "component": component,
        "frequency": 0.75,
        "center": (0.0, 0.0, 0.0),
        "size": (0.0, 0.0, 0.0),
        "amplitude": 0.65,
    }


def build_driver(cp, cell, resolution: float, component: str | None, seed: int):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=resolution, cylindrical=True, m=0,
        force_complex_fields=False, courant=0.31,
        boundaries={"z": "metallic"}, prefer_gpu=True, gpu_id=0,
    )
    driver.setup_pml({"x": {"high": 4}, "z": 4})
    shape = tuple(int(value) for value in driver.shape)
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.35 + 0.20 * np.sin(index * np.float32(0.041))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    if component is not None:
        driver.add_source(source(component))
    rng = np.random.default_rng(seed)
    for name in PRIMARY_NAMES:
        values = np.ascontiguousarray(
            rng.uniform(-0.20, 0.20, size=shape).astype(np.float32))
        driver.set_field(name, cp.asarray(values))
    return driver


def state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name) for name in STATE_NAMES
            if getattr(driver.fields, name, None) is not None}


def compare(cp, left, right) -> Dict[str, Any]:
    a = np.ascontiguousarray(cp.asnumpy(left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(cp.asnumpy(right)).view(np.uint32).ravel()
    differing = int(np.count_nonzero(a != b))
    return {"bit_identical": differing == 0, "differing_floats": differing,
            "total_floats": int(a.size)}


def plan_types(plan) -> Dict[str, str]:
    return {name: type(entry).__name__ for name, entry in plan.plans.items()}


def install(driver_module, plan, owner):
    """Install the candidate plan only for its driver, leaving dispatch untouched."""
    names = ("step_B", "update_H", "step_D", "update_E", "update_P")
    originals = {name: getattr(driver_module, name) for name in names}

    def replacement(name):
        def wrapper(fields, pml=None):
            entry = plan.plans.get(name) if fields is owner else None
            if entry is None:
                return originals[name](fields, pml)
            if name == "update_P":
                for pole_plan in entry:
                    pole_plan.run(fields.drive_field)
                return None
            entry.run()
            return None
        return wrapper

    for name in names:
        setattr(driver_module, name, replacement(name))

    def undo():
        for name, function in originals.items():
            setattr(driver_module, name, function)
    return undo


def run_case(cp, name, cell, resolution, component, steps) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference = build_driver(cp, cell, resolution, component, 20260811)
    candidate = build_driver(cp, cell, resolution, component, 20260811)
    control = (build_driver(cp, cell, resolution, None, 20260811)
               if component is not None else None)
    undo = lambda: None
    try:
        plan = plan_step(candidate.fields, candidate.pml, fuse=True,
                         sources=tuple(candidate._sources), num_warps=1)
        selected = plan_types(plan)
        expected = EXPECTED_BY_CASE[name]
        if selected != expected:
            raise AssertionError(
                f"{name}: selected={selected}, expected={expected}, reasons={plan.reasons}")
        # The generic cylindrical routes must still refuse BY NAME for a cylindrical
        # reason — the claim this leg has always made, now pinned to those route
        # names so a vanished route fails instead of passing vacuously. The
        # product-specific routes enumerated since the 2026-09-02 consult expansion
        # refuse for their own named reasons (real storage, no chi2/chi3, ...), so
        # what is required of them is that none refuses SILENTLY.
        for key in CYLINDRICAL_REFUSALS:
            reasons = plan.reasons.get(key)
            if not reasons or not all("cylindrical" in reason.lower() for reason in reasons):
                raise AssertionError(
                    f"{name}: expected a named cylindrical refusal at {key}, got {reasons!r}")
        silent = sorted(key for key, reasons in plan.reasons.items()
                        if key.startswith("fused_pair")
                        and not [item for item in reasons if item and item.strip()])
        if silent:
            raise AssertionError(f"{name}: fused-pair routes refused silently: {silent}")
        undo = install(driver_module, plan, candidate.fields)
        row: Dict[str, Any] = {
            "case": name, "shape": [int(value) for value in candidate.shape],
            "steps": steps, "source": component, "selected": selected,
            "refusals": {key: list(value) for key, value in plan.reasons.items()},
            "per_step": [], "source_effect_seen": component is None,
        }
        started = time.time()
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            if control is not None:
                control.step()
            cp.cuda.runtime.deviceSynchronize()
            reference_state, candidate_state = state(reference), state(candidate)
            if set(reference_state) != set(candidate_state):
                raise AssertionError(f"{name} step {step}: state inventory differs")
            comparisons = {key: compare(cp, candidate_state[key], reference_state[key])
                           for key in sorted(reference_state)}
            same = all(item["bit_identical"] for item in comparisons.values())
            if control is not None:
                row["source_effect_seen"] = row["source_effect_seen"] or any(
                    not compare(cp, reference_state[key], getattr(control.fields, key))["bit_identical"]
                    for key in PRIMARY_NAMES)
            point = {
                "step": step, "bit_identical": same,
                "differing_floats": sum(item["differing_floats"]
                                          for item in comparisons.values()),
                "source_effect_seen": row["source_effect_seen"],
            }
            row["per_step"].append(point)
            log(f"case {name} step {step}/{steps} identical={same} "
                f"ndiff={point['differing_floats']} ({time.time() - started:.1f}s)")
            if not same:
                raise AssertionError(f"{name} step {step}: uint32 state mismatch {comparisons}")
        if component is not None and not row["source_effect_seen"]:
            raise AssertionError(f"{name}: source did not separate from source-free control")
        return row
    finally:
        undo()
        reference.close()
        candidate.close()
        if control is not None:
            control.close()
        cp.get_default_memory_pool().free_all_blocks()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import cupy as cp  # noqa: PLC0415
    import triton  # noqa: PLC0415

    payload: Dict[str, Any] = {
        "environment": {
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy": cp.__version__, "triton": triton.__version__,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "cases": [],
    }
    save(payload, args.out)
    for item in CASES:
        row = run_case(cp, *item)
        payload["cases"].append(row)
        save(payload, args.out)
    total_steps = sum(row["steps"] for row in payload["cases"])
    payload["summary"] = {
        "status": "passed", "cases_exact": f"{len(payload['cases'])}/{len(CASES)}",
        "complete_steps_exact": f"{total_steps}/{total_steps}",
        "source_cases_nonvacuous": "1/1",
        "scope": "real m=0 Dcyl; r-high and z PML; axis/metallic/periodic boundary "
                 "triple; no dispersion. PAIR FUSION IS CERTIFIED, not excluded: since the\n"
                 "2026-09-02 dispatch expansion the composer selects the cylindrical real\n"
                 "fused pair, and each case is pinned to the exact composition it produces\n"
                 "(the fused pair plan at the curl slot, its NoopPlan partner at the\n"
                 "constitutive) rather than to one shared unfused table",
    }
    save(payload, args.out)
    log(f"CYLINDRICAL COMPOSITION GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
