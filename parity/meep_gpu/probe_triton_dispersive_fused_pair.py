"""CUDA gate for the fused dispersive D-curl / electric-constitutive pair.

THE SOURCE CASE. Until 2026-08-27 every case here was source-free, because the product
refused any row carrying an electric source: the driver injects it between ``step_D`` and
``update_E``, which is inside the seam this pair closes. ``dispersive_fused_pair`` now
declares ``CARRIES_DEPOSIT_REPAIR``, so ``launch._install_fused_pair`` brackets its launch
with ``deposit_repair.LeadingRepairPlan`` / ``TrailingRepairPlan`` and the row is carried
instead. ``one_pole_electric_source`` is the case that measures it, and it is the ONLY
evidence that entitles this product to the declaration: the fused driver runs against the
array path and against the separate Triton products, byte for byte over every field AND
every polarization array, with the plan classes pinned so that a pass cannot come from a
refused fusion, and with the repaired-point count required to be non-zero so that it
cannot come from a repair that found nothing to do.
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


def electric_source(component: str, integrated: bool = False) -> Dict[str, Any]:
    """A point current in the D/E seam — the deposit the repair has to carry."""
    return {
        "component": component,
        "frequency": 0.7,
        "center": (0.18, -0.12, 0.0),
        "size": (0.0, 0.0, 0.0),
        "amplitude": 0.75,
        "is_integrated": integrated,
    }


# name, boundary declarations, per-pole (frequency, gamma, Ex/Ey/Ez sigma), steps,
# source declarations injected inside this pair's seam
CASES = (
    ("one_pole_periodic", ("periodic", "periodic", "periodic"),
     ((0.62, 0.05, (0.40, 0.40, 0.40)),), 12, ()),
    ("two_pole_anisotropic", ("periodic", "periodic", "periodic"),
     ((0.58, 0.04, (0.35, 0.00, 0.25)),
      (0.91, 0.08, (0.20, 0.45, 0.00))), 12, ()),
    ("three_pole_metallic_x", ("metallic", "periodic", "periodic"),
     ((0.55, 0.03, (0.30, 0.25, 0.20)),
      (0.83, 0.07, (0.20, 0.15, 0.00)),
      (1.05, 0.09, (0.10, 0.00, 0.00))), 12, ()),
    # THE DEPOSIT CASE. Two poles so the pole-aware constitutive half is doing real
    # work, and a metallic x so the inline wall clear is live in the same seam the
    # repair spans.
    ("two_pole_electric_source", ("metallic", "periodic", "periodic"),
     ((0.58, 0.04, (0.35, 0.00, 0.25)),
      (0.91, 0.08, (0.20, 0.45, 0.00))), 12, (electric_source("Ez"),)),
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
    why its record carried no readable verdict and could not be welded: the
    record's ``source_sha256`` was transcribed by hand from a staged tree chosen
    by eye, and "did it release?" was recoverable only by knowing this gate's
    private spelling of the answer (``summary.status``).

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


def build_driver(cp, boundaries, pole_specs, seed: int, source_specs=()):
    from meep_gpu.dispersion import Susceptibility  # noqa: PLC0415
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
    for frequency, gamma, sigma_values in pole_specs:
        sigma = {
            component: cp.full(shape, value, dtype=cp.float32)
            for component, value in zip(("Ex", "Ey", "Ez"), sigma_values)
        }
        driver.add_susceptibility(
            Susceptibility(frequency=frequency, gamma=gamma), sigma)
    for declaration in source_specs:
        driver.add_source(dict(declaration))
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))))
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
    return {name: type(value).__name__ for name, value in plan.plans.items()}


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


def run_case(cp, name, boundaries, pole_specs, steps,
             source_specs=()) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference = build_driver(cp, boundaries, pole_specs, 20260811, source_specs)
    separate = build_driver(cp, boundaries, pole_specs, 20260811, source_specs)
    fused = build_driver(cp, boundaries, pole_specs, 20260811, source_specs)
    undo = lambda: None
    try:
        separate_plan = plan_step(
            separate.fields, separate.pml, fuse=False,
            sources=tuple(separate._sources), num_warps=1)
        fused_plan = plan_step(
            fused.fields, fused.pml, fuse=True,
            sources=tuple(fused._sources), num_warps=1)
        separate_types = plan_types(separate_plan)
        fused_types = plan_types(fused_plan)
        expected_separate = {
            "step_B": "PmlCurlPlan", "update_H": "ConstitutivePlan",
            "step_D": "PmlCurlPlan", "update_E": "DispersiveConstitutivePlan",
            "update_P": "list",
        }
        # WITH AN ELECTRIC SOURCE the dispersive pair still owns both slots, but the
        # objects in them are the repair's: the LeadingRepairPlan saves E and f_w_E at
        # the deposit points and then launches DispersiveFusedPairPlan, and the
        # TrailingRepairPlan recomputes them after the driver has injected. Pinning
        # NoopPlan here for a source-carrying row would accept a REFUSED fusion as a
        # pass. The B/H pair's seam is clean either way — nothing magnetic is declared.
        expected_fused = {
            "step_B": "FusedPairPlan", "update_H": "NoopPlan",
            "step_D": ("LeadingRepairPlan" if source_specs
                       else "DispersiveFusedPairPlan"),
            "update_E": "TrailingRepairPlan" if source_specs else "NoopPlan",
            "update_P": "list",
        }
        # plan_types sees the update_P value as the concrete list installed by plan_step.
        if separate_types != expected_separate:
            raise AssertionError(
                f"{name}: separate products {separate_types}, expected {expected_separate}; "
                f"reasons={separate_plan.reasons}")
        if fused_types != expected_fused:
            raise AssertionError(
                f"{name}: fused products {fused_types}, expected {expected_fused}; "
                f"reasons={fused_plan.reasons}")
        undo = install(driver_module, (
            (separate.fields, separate_plan), (fused.fields, fused_plan)))
        leading = fused_plan.plans["step_D"]
        # The repair wraps the product rather than replacing it; the counts belong to
        # the launch either way, so read through the wrapper instead of past it.
        product = getattr(leading, "inner", leading)
        row: Dict[str, Any] = {
            "case": name, "shape": list(reference.shape), "steps": steps,
            "pole_counts": list(product.counts),
            "sources": [str(item.field_type) for item in fused._sources],
            "repairs_seen": 0,
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
            # A repair that touched no point would make this case byte-identical for a
            # reason that has nothing to do with the repair, so the count is asserted
            # rather than merely recorded.
            repairs = int(getattr(leading, "repairs", 0) or 0)
            row["repairs_seen"] += repairs
            point = {
                "step": step, "fused_vs_array": fused_vs_array,
                "separate_vs_array": separate_vs_array,
                "fused_vs_separate": fused_vs_separate,
                "repair_points": repairs,
            }
            row["per_step"].append(point)
            identical = all(item["bit_identical"] for item in (
                fused_vs_array, separate_vs_array, fused_vs_separate))
            log(f"case {name} step {step}/{steps} identical={identical} "
                f"fused_ndiff={fused_vs_array['differing_floats']} "
                f"separate_ndiff={separate_vs_array['differing_floats']} "
                f"repaired={repairs} ({time.time() - started:.1f} s)")
            if not identical:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
            if source_specs and not repairs:
                raise AssertionError(
                    f"{name} step {step}: the seam carries a deposit but the repair "
                    f"touched no point; a pass here would be vacuous")
        if source_specs and not row["repairs_seen"]:
            raise AssertionError(f"{name}: no deposit point was ever repaired")
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
    for name, boundaries, poles, steps, sources in CASES:
        log(f"starting {name}: poles={len(poles)} boundaries={boundaries} "
            f"sources={len(sources)}")
        payload["cases"].append(
            run_case(cp, name, boundaries, poles, steps, sources))
        save(payload, args.out)
    total = sum(row["steps"] for row in payload["cases"])
    repaired = sum(row["repairs_seen"] for row in payload["cases"])
    payload["summary"] = {
        "status": "passed", "cases_exact": f"{len(CASES)}/{len(CASES)}",
        "complete_steps_exact": f"{total}/{total}",
        "oracles_per_step": "array path + separate Triton products",
        "pole_products": "one pole isotropic; two pole anisotropic; three pole metallic",
        "in_seam_deposit_cases": "1/1 (two_pole_electric_source)",
        "repair_points_touched": repaired,
    }
    save(payload, args.out)
    log(f"DISPERSIVE FUSED-PAIR GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
