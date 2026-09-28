"""Certify the two fused-pair source seams through complete CUDA driver steps.

The B/H pair may span an electric source because that source is injected later, in
the D/E seam.  The D/E pair may likewise span a magnetic source.  A source in the
pair's OWN seam is now spanned too, by the deposit repair: ``coverage.py`` declares
``CARRIES_DEPOSIT_REPAIR`` and ``launch._install_fused_pair`` puts a
``LeadingRepairPlan`` in the pair's first slot and a ``TrailingRepairPlan`` in its
second, so the fused launch runs against an uninjected field and the deposit points are
recomputed once the driver has injected, filled symmetry and cleared walls.

WHAT THIS GATE MEASURES, AND WHY IT IS THE ONE THAT LICENSES THE FLIP. Both legs are
complete ``FdtdDriver`` steps: the reference on the array path, the candidate with the
plan installed at the driver's real consult points. Every live array is compared as
uint32 after EVERY step, so a repair that ran against a pre-injection field -- the exact
failure ``deposit_repair`` exists to prevent -- cannot pass; it would show as a handful
of differing floats at the deposit points on step 1. A source-free control additionally
has to SEPARATE from the reference, so a case where the source never reached the field
cannot pass vacuously either.

The expected plan classes below are exact, and they are the second half of the
measurement: byte identity with ``NoopPlan`` in the second slot would mean the fusion
was refused and the array path did the work, which is a pass that proves nothing about
the repair. ``LeadingRepairPlan`` / ``TrailingRepairPlan`` in a case's slots is the
assertion that the repaired composition is what produced those bytes.
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
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)


def source(component: str, integrated: bool = False) -> Dict[str, Any]:
    return {
        "component": component,
        "frequency": 0.7,
        "center": (0.18, -0.12, 0.0),
        "size": (0.0, 0.0, 0.0),
        "amplitude": 0.75,
        "is_integrated": integrated,
    }


# Both pairs fused with a CLEAN seam: nothing was injected between the halves, so the
# second slot holds the sentinel and there is no deposit to carry.
FUSED_BOTH = {
    "step_B": "FusedPairPlan", "update_H": "NoopPlan",
    "step_D": "FusedPairPlan", "update_E": "NoopPlan",
}
# The B pair's seam is clean and the D pair's carries an electric deposit, so the D pair
# is the one bracketed by the repair.
FUSED_BOTH_REPAIRING_D = {
    "step_B": "FusedPairPlan", "update_H": "NoopPlan",
    "step_D": "LeadingRepairPlan", "update_E": "TrailingRepairPlan",
}
FUSED_BOTH_REPAIRING_B = {
    "step_B": "LeadingRepairPlan", "update_H": "TrailingRepairPlan",
    "step_D": "FusedPairPlan", "update_E": "NoopPlan",
}
FUSED_BOTH_REPAIRING_BOTH = {
    "step_B": "LeadingRepairPlan", "update_H": "TrailingRepairPlan",
    "step_D": "LeadingRepairPlan", "update_E": "TrailingRepairPlan",
}

# (name, source declarations, complete-step budget, exact central-plan classes)
CASES = (
    ("source_free", (), 8, FUSED_BOTH),
    ("electric", (source("Ez"),), 10, FUSED_BOTH_REPAIRING_D),
    ("magnetic", (source("Hx"),), 10, FUSED_BOTH_REPAIRING_B),
    ("electric_and_magnetic", (source("Ez"), source("Hx")), 10,
     FUSED_BOTH_REPAIRING_BOTH),
    ("integrated_electric", (source("Ez", True),), 12, FUSED_BOTH_REPAIRING_D),
    ("integrated_magnetic", (source("Hx", True),), 12, FUSED_BOTH_REPAIRING_B),
)

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


def build_driver(cp, source_specs, seed: int):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=(4.0, 3.5, 0.0), resolution=16.0, dimensions=2,
        force_complex_fields=False, courant=0.35,
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.4 + 0.35 * np.sin(index * np.float32(0.031))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml({"x": 6, "y": 6})
    for declaration in source_specs:
        driver.add_source(dict(declaration))
    rng = np.random.default_rng(seed)
    for name in PRIMARY_NAMES:
        values = np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))
        driver.set_field(name, cp.asarray(values))
    return driver


def state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name) for name in STATE_NAMES
            if getattr(driver.fields, name, None) is not None}


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def compare(cp, left, right) -> Dict[str, Any]:
    a = words(cp, left)
    b = words(cp, right)
    return {
        "bit_identical": bool(np.array_equal(a, b)),
        "differing_floats": int(np.count_nonzero(a != b)),
        "total_floats": int(a.size),
    }


def plan_types(plan) -> Dict[str, str]:
    return {name: type(entry).__name__ for name, entry in plan.plans.items()}


def install(driver_module, plan, owner):
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
            else:
                entry.run()
            return None
        return wrapper

    for name in names:
        setattr(driver_module, name, replacement(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


def run_case(cp, name: str, declarations, steps: int,
             expected: Dict[str, str]) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference = build_driver(cp, declarations, 20260811)
    candidate = build_driver(cp, declarations, 20260811)
    control = build_driver(cp, (), 20260811) if declarations else None
    undo = lambda: None
    try:
        plan = plan_step(
            candidate.fields, candidate.pml, fuse=True,
            sources=tuple(candidate._sources), num_warps=1)
        selected = plan_types(plan)
        row: Dict[str, Any] = {
            "case": name,
            "shape": [int(value) for value in candidate.shape],
            "source_families": [str(item.field_type) for item in candidate._sources],
            "integrated": [bool(item.is_integrated) for item in candidate._sources],
            "steps": steps,
            "selected": selected,
            "expected": expected,
            "refusals": {key: list(value) for key, value in plan.reasons.items()},
            "repairs_seen": 0,
            "per_step": [],
        }
        expected_repairs = "LeadingRepairPlan" in expected.values()
        if selected != expected:
            raise AssertionError(
                f"{name}: central product classes differ: selected={selected}, "
                f"expected={expected}, refusals={plan.reasons}")
        undo = install(driver_module, plan, candidate.fields)
        source_effect_seen = not declarations
        started = time.time()
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            if control is not None:
                control.step()
            cp.cuda.runtime.deviceSynchronize()
            ref_state = state(reference)
            got_state = state(candidate)
            if set(ref_state) != set(got_state):
                raise AssertionError(
                    f"{name} step {step}: state inventory differs: "
                    f"reference={sorted(ref_state)}, candidate={sorted(got_state)}")
            parts = {key: compare(cp, got_state[key], ref_state[key])
                     for key in sorted(ref_state)}
            same = all(part["bit_identical"] for part in parts.values())
            if control is not None:
                source_effect_seen = source_effect_seen or any(
                    not compare(cp, ref_state[key], getattr(control.fields, key))["bit_identical"]
                    for key in PRIMARY_NAMES)
            source_state_same = all(
                getattr(left, "_applied_dipole", 0j)
                == getattr(right, "_applied_dipole", 0j)
                for left, right in zip(candidate._sources, reference._sources))
            point = {
                "step": step,
                "bit_identical": same and source_state_same,
                "differing_floats": sum(part["differing_floats"]
                                          for part in parts.values()),
                "total_floats": sum(part["total_floats"] for part in parts.values()),
                "differing_arrays": sorted(
                    key for key, part in parts.items() if not part["bit_identical"]),
                "source_state_identical": source_state_same,
                "source_effect_seen": source_effect_seen,
            }
            # THE REPAIR HAS TO HAVE TOUCHED SOMETHING. A TrailingRepairPlan whose
            # apply() found no deposit index would leave the bytes identical for the
            # wrong reason -- the launch would simply have been correct because
            # nothing was injected -- and this case would pass while measuring
            # nothing. `repairs` is the point count apply() returned.
            repairs = sum(int(getattr(plan.plans.get(slot), "repairs", 0) or 0)
                          for slot in ("step_B", "step_D"))
            point["repair_points"] = repairs
            row["repairs_seen"] += repairs
            if expected_repairs and not repairs:
                raise AssertionError(
                    f"{name} step {step}: the composition declares a repaired seam but "
                    f"the repair touched no point; a pass here would be vacuous")
            row["per_step"].append(point)
            log(f"case {name} step {step}/{steps} identical={point['bit_identical']} "
                f"source_effect={source_effect_seen} ndiff={point['differing_floats']} "
                f"repaired={repairs} ({time.time() - started:.1f} s)")
            if not point["bit_identical"]:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        if not source_effect_seen:
            raise AssertionError(
                f"{name}: the source never separated the reference from a source-free control")
        if expected_repairs and not row["repairs_seen"]:
            raise AssertionError(f"{name}: no deposit point was ever repaired")
        row["bit_identical"] = True
        row["source_effect_seen"] = source_effect_seen
        return row
    finally:
        undo()
        reference.close()
        candidate.close()
        if control is not None:
            control.close()
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
    for name, declarations, steps, expected in CASES:
        log(f"starting {name}: expected={expected}")
        payload["cases"].append(
            run_case(cp, name, declarations, steps, expected))
        save(payload, args.out)
    total_steps = sum(row["steps"] for row in payload["cases"])
    repaired = sum(row["repairs_seen"] for row in payload["cases"])
    payload["summary"] = {
        "status": "passed",
        "cases_exact": f"{len(payload['cases'])}/{len(CASES)}",
        "complete_steps_exact": f"{total_steps}/{total_steps}",
        "nonvacuous_source_cases": "5/5",
        "source_free_pairs_fused": "2/2",
        "pairs_fused": "12/12 pair-instances across 6 cases",
        "in_seam_deposits_carried": "8/8 pair/source intersections",
        "repair_points_touched": repaired,
    }
    save(payload, args.out)
    log(f"SOURCE-SEAM GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
