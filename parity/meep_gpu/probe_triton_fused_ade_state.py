"""CUDA gate for one-launch, one-susceptibility Triton ADE updates."""

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


# name, boundaries, susceptibility kind, sigma form, steps
CASES = (
    ("scalar_all_periodic", ("periodic", "periodic", "periodic"),
     "lorentzian", "scalar_all", 12),
    ("volume_subset_periodic", ("periodic", "periodic", "periodic"),
     "lorentzian", "volume_subset", 12),
    ("drude_all_metallic_x", ("metallic", "periodic", "periodic"),
     "drude", "volume_all", 12),
)

FIELD_NAMES = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz",
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically.

    THE STAMP IS HERE AND NOT AT THE CALL SITES, for the reason already written
    down on the chain gate: ``main`` calls this five times as the cases land, and
    a stamp bolted onto the last call would leave every partial artifact — the
    one an aborted run leaves behind, which is the artifact a failure is read
    from — unattributable.

    ``gate_provenance.stamp`` writes ``imported_source_sha256`` (every repo
    module THIS PROCESS imported, enumerated from ``sys.modules``) and
    ``canonical_verdict``. Before 2026-08-19 this gate wrote neither, which is
    why its record carried no readable verdict and could not be welded.

    The policy stamp is RE-READ rather than carried. Taken once at install time
    it records every strip counter at zero, because nothing had compiled yet;
    re-reading at each checkpoint makes the counters describe the run.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

    if "subnormal_policy" in payload:
        from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

        payload["subnormal_policy"] = _policy.policy_stamp()
    _stamp_provenance(payload)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_driver(cp, boundaries, kind: str, sigma_form: str, seed: int):
    from meep_gpu.dispersion import DRUDE, LORENTZIAN, Susceptibility  # noqa: PLC0415
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=(4.0, 3.5, 0.0), resolution=16.0, dimensions=2,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.55 + 0.30 * np.sin(index * np.float32(0.027))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml({"x": 6, "y": 6})

    if sigma_form == "scalar_all":
        sigma = 0.37
    else:
        base = np.ascontiguousarray(
            (0.22 + 0.31 * (0.5 + 0.5 * np.sin(index * np.float32(0.019))))
            .astype(np.float32))
        volume = cp.asarray(base)
        sigma = ({"Ex": volume, "Ey": 0.0, "Ez": volume * cp.float32(0.73)}
                 if sigma_form == "volume_subset"
                 else {"Ex": volume, "Ey": volume * cp.float32(0.81),
                       "Ez": volume * cp.float32(0.63)})
    driver.add_susceptibility(
        Susceptibility(
            frequency=0.72 if kind == "lorentzian" else 0.61,
            gamma=0.06 if kind == "lorentzian" else 0.04,
            kind=LORENTZIAN if kind == "lorentzian" else DRUDE),
        sigma)

    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))))
    state = driver.fields.polarizations[0]
    for component in state.driven():
        state.P[component][...] = cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.08, 0.08, size=shape).astype(np.float32)))
        state.P_prev[component][...] = cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.08, 0.08, size=shape).astype(np.float32)))
    return driver


def state(driver) -> Dict[str, Any]:
    result = {name: getattr(driver.fields, name) for name in FIELD_NAMES
              if getattr(driver.fields, name, None) is not None}
    for index, pole in enumerate(driver.fields.polarizations):
        for component in pole.driven():
            result[f"P{index}_{component}"] = pole.P[component]
            result[f"Pprev{index}_{component}"] = pole.P_prev[component]
        if pole._scratch is not None:
            result[f"Pscratch{index}"] = pole._scratch
    return result


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def compare_states(cp, left, right) -> Dict[str, Any]:
    a = state(left)
    b = state(right)
    if set(a) != set(b):
        raise AssertionError(
            f"state inventory differs: {sorted(a)} versus {sorted(b)}")
    differing = {}
    total = 0
    for name in sorted(a):
        wa = words(cp, a[name])
        wb = words(cp, b[name])
        count = int(np.count_nonzero(wa != wb))
        total += count
        if count:
            differing[name] = count
    return {"bit_identical": not differing, "differing_floats": total,
            "differing_arrays": differing}


def plan_types(plan) -> Dict[str, str]:
    types = {name: type(value).__name__ for name, value in plan.plans.items()}
    if "update_P" in plan.plans:
        types["update_P_entries"] = ",".join(
            type(value).__name__ for value in plan.plans["update_P"])
    return types


def install(driver_module, routes):
    names = ("step_B", "update_H", "step_D", "update_E", "update_P")
    originals = {name: getattr(driver_module, name) for name in names}

    def replacement(name):
        def wrapper(fields, pml=None):
            plan = next((candidate for owner, candidate in routes
                         if fields is owner), None)
            if plan is None or name not in plan.plans:
                return originals[name](fields, pml)
            entry = plan.plans[name]
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


def run_case(cp, name, boundaries, kind, sigma_form, steps) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference = build_driver(cp, boundaries, kind, sigma_form, 20260811)
    separate = build_driver(cp, boundaries, kind, sigma_form, 20260811)
    fused = build_driver(cp, boundaries, kind, sigma_form, 20260811)
    undo = lambda: None
    try:
        separate_plan = plan_step(
            separate.fields, separate.pml, fuse=True,
            sources=tuple(separate._sources), num_warps=1, fuse_ade=False)
        fused_plan = plan_step(
            fused.fields, fused.pml, fuse=True,
            sources=tuple(fused._sources), num_warps=1, fuse_ade=True)
        separate_types = plan_types(separate_plan)
        fused_types = plan_types(fused_plan)
        common = {
            "step_B": "FusedPairPlan", "update_H": "NoopPlan",
            "step_D": "DispersiveFusedPairPlan", "update_E": "NoopPlan",
            "update_P": "list",
        }
        expected_separate = dict(common, update_P_entries="AdeUpdatePPlan")
        expected_fused = dict(common, update_P_entries="FusedAdeStatePlan")
        if separate_types != expected_separate:
            raise AssertionError(
                f"{name}: separate products {separate_types}, expected "
                f"{expected_separate}; reasons={separate_plan.reasons}")
        if fused_types != expected_fused:
            raise AssertionError(
                f"{name}: fused products {fused_types}, expected "
                f"{expected_fused}; reasons={fused_plan.reasons}")
        undo = install(driver_module, (
            (separate.fields, separate_plan), (fused.fields, fused_plan)))
        row: Dict[str, Any] = {
            "case": name, "shape": list(reference.shape), "steps": steps,
            "driven": list(fused.fields.polarizations[0].driven()),
            "separate": separate_types, "fused": fused_types, "per_step": [],
        }
        started = time.time()
        for step in range(1, steps + 1):
            reference.step()
            separate.step()
            fused.step()
            cp.cuda.runtime.deviceSynchronize()
            fused_vs_array = compare_states(cp, fused, reference)
            separate_vs_array = compare_states(cp, separate, reference)
            fused_vs_separate = compare_states(cp, fused, separate)
            point = {
                "step": step, "fused_vs_array": fused_vs_array,
                "separate_vs_array": separate_vs_array,
                "fused_vs_separate": fused_vs_separate,
            }
            row["per_step"].append(point)
            identical = all(item["bit_identical"] for item in (
                fused_vs_array, separate_vs_array, fused_vs_separate))
            log(f"case {name} step {step}/{steps} identical={identical} "
                f"fused_ndiff={fused_vs_array['differing_floats']} "
                f"separate_ndiff={separate_vs_array['differing_floats']} "
                f"({time.time() - started:.1f} s)")
            if not identical:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        row["bit_identical"] = True
        return row
    finally:
        undo()
        reference.close()
        separate.close()
        fused.close()
        cp.get_default_memory_pool().free_all_blocks()


def environment(cp) -> Dict[str, Any]:
    import triton  # noqa: PLC0415

    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "cupy": cp.__version__, "triton": triton.__version__,
        "device": properties["name"].decode(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep' — the policy every record in "
             "triton_kernels/fingerprints.json is cut under. The 2026-08-11 job "
             "and both recerts of this gate installed NOTHING, which left CuPy "
             "flushing (it appends -ftz=true unconditionally) beside a natively "
             "keeping Triton: a mixed configuration attributable to no policy at "
             "all, which is the debt three existing welds already carry.")
    args = parser.parse_args(argv)

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep
    # and quietly did not is a process whose bytes mean nothing. It is also what
    # enforces the CUPY_CACHE_DIR token rule — CuPy's cache key is computed ABOVE
    # the seam the -ftz strip installs at, so a directory shared with a flush run
    # would serve flushed binaries under this record's name.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    policy_record = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: {policy_record.get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    backends.guard_kernel_compilation(cp)
    payload: Dict[str, Any] = {"environment": environment(cp),
                               "subnormal_policy": policy_record, "cases": []}
    save(payload, args.out)
    for case in CASES:
        log(f"starting {case[0]}: kind={case[2]} sigma={case[3]} "
            f"boundaries={case[1]}")
        payload["cases"].append(run_case(cp, *case))
        save(payload, args.out)
    total = sum(row["steps"] for row in payload["cases"])
    payload["summary"] = {
        "status": "passed", "cases_exact": f"{len(CASES)}/{len(CASES)}",
        "complete_steps_exact": f"{total}/{total}",
        "oracles_per_step": "array path + separate per-component Triton ADE",
        "scope": "one susceptibility; all/subset components; scalar/volume sigma; Lorentz/Drude",
    }
    save(payload, args.out)
    log(f"FUSED ADE-STATE GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
