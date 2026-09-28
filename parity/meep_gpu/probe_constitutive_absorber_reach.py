"""Has the field REACHED the absorber in the window the lifted leg compares?

WHY THIS EXISTS. The lifted corpus guard control — the same cases with
``--fmad=false`` STRIPPED — is supposed to diverge, and on the synthetic gate the
unguarded leg diverges on 240 of 240 cases. On the corpus it did not. Before
explaining that with an argument, measure it.

THE HYPOTHESIS THIS TESTS. The constitutive sub-step is

    a = f[idx] + kps * src;   f[idx] = a - kms * prev;

and ``kps``/``kms`` are ``kappa +/- sigma`` off the PML profile. Outside the
absorber ``sigma == 0``, so **kps == kms == 1.0 exactly**, and then
``fma(1.0f, src, f)`` and ``add.rn(f, mul.rn(1.0f, src))`` are the same float32:
one rounding either way, because ``1.0f * src`` is exact. The contraction can
therefore only be visible in cells where the coefficient is NOT 1 — i.e. inside
the layer — and only if the operand there is nonzero.

A corpus case runs 2 warm-up steps plus the leg's 8: if the source pulse has not
propagated from the centre to the boundary in 10 steps, every cell where the
coefficient bites still holds exactly zero, ``kps * 0`` is 0 under both
compilations, and the guard is unexercised no matter how many cells are compared.

THIS IS THE SAME FAILURE CLASS as the one the disposition's §1.4 records for the
curl's lifted leg (all six cases at Courant 0.5, where the dtdx scaling is exact,
so the leg could not distinguish the guard at all) — a leg that looks like
evidence for the arithmetic and is evidence for something else. The difference is
that this one is measured here rather than found later.

Run (the GPU host, one GPU)::

    CUDA_VISIBLE_DEVICES=0 python -u probe_constitutive_absorber_reach.py \\
        --out results/<dir>/absorber_reach.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)

#: The cheap end of the lifted leg's case list — the question is about the
#: physics of the window, not about size, and these run in seconds.
CASES = [
    ("2d_pml", {"res": 40, "n": 16}),
    ("3d_pml", {"res": 15, "n": 6}),
    ("2d_pml_geom", {"res": 40, "n": 16}),
]
COURANTS = (0.5, 0.35)

#: What the lifted leg does: 2 warm-up steps in ``driver.run`` then 8 whole steps.
WARMUP_STEPS = 2
COMPARED_STEPS = 8

SIDES = {
    "H": (("Hx", "Bx", 0), ("Hy", "By", 1), ("Hz", "Bz", 2)),
    "E": (("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2)),
}


def log(message: str) -> None:
    print(message, flush=True)


def load_curl_leg():
    path = os.path.join(HERE, "validate_pml_kernel_on_lifted_cases.py")
    spec = importlib.util.spec_from_file_location("validate_pml_lifted", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def one_case(name: str, kwargs: Dict[str, Any], courant: float) -> Dict[str, Any]:
    import cupy

    from meep_gpu import backends, stepping
    from meep_gpu.cuda_kernels import constitutive_kernels as kernels
    from meep_gpu.from_meep import lift_simulation
    import cases

    backends.guard_kernel_compilation(cupy)
    simulation = cases.BUILDERS[name](**kwargs)
    simulation.Courant = float(courant)
    driver = lift_simulation(simulation, prefer_gpu=True, gpu_id=0)
    driver.run(num_steps=WARMUP_STEPS)
    fields, pml = driver.fields, driver.pml

    record: Dict[str, Any] = {
        "case": name, "kwargs": kwargs, "courant": float(courant),
        "shape": [int(s) for s in driver.shape],
        "warmup_steps": WARMUP_STEPS, "compared_steps": COMPARED_STEPS,
        "sides": {},
    }

    for side, terms in SIDES.items():
        tables = kernels.constitutive_tables_for(side, pml)
        per_component: Dict[str, Any] = {}
        for target, source, axis in terms:
            axis_name = "xyz"[axis]
            kps = cupy.asnumpy(tables[f"kps_{axis_name}"]).ravel()
            kms = cupy.asnumpy(tables[f"kms_{axis_name}"]).ravel()
            # Cells along this axis where the coefficient is NOT exactly one, i.e.
            # where sigma > 0 and the contraction could be visible at all.
            biting = (kps != np.float32(1.0)) | (kms != np.float32(1.0))
            per_component[target] = {
                "axis": axis_name,
                "coefficient_cells": int(kps.size),
                "cells_where_coefficient_is_not_one": int(biting.sum()),
                "kps_min": float(kps.min()), "kps_max": float(kps.max()),
                "kms_min": float(kms.min()), "kms_max": float(kms.max()),
            }
        record["sides"][side] = {"per_component": per_component}

    # Now advance exactly what the lifted leg advances and ask whether the
    # operand is nonzero WHERE the coefficient bites.
    for _ in range(COMPARED_STEPS):
        stepping.step_B(fields, pml)
        stepping.update_H(fields, pml)
        stepping.step_D(fields, pml)
        stepping.update_E(fields, pml)
    cupy.cuda.runtime.deviceSynchronize()

    for side, terms in SIDES.items():
        tables = kernels.constitutive_tables_for(side, pml)
        for target, source, axis in terms:
            axis_name = "xyz"[axis]
            kps = tables[f"kps_{axis_name}"].reshape(-1)
            kms = tables[f"kms_{axis_name}"].reshape(-1)
            biting = (kps != cupy.float32(1.0)) | (kms != cupy.float32(1.0))
            volume = getattr(fields, source)
            auxiliary = getattr(fields, "f_w_" + target)
            # Broadcast the per-axis mask over the volume along its own axis.
            shape = [1, 1, 1]
            shape[axis] = biting.size
            mask = biting.reshape(tuple(shape))
            live = cupy.logical_and(mask, volume != 0)
            live_aux = cupy.logical_and(mask, auxiliary != 0)
            entry = record["sides"][side]["per_component"][target]
            entry.update({
                "cells_in_the_biting_region": int(cupy.broadcast_to(
                    mask, volume.shape).sum()),
                "biting_cells_with_a_NONZERO_source": int(live.sum()),
                "biting_cells_with_a_NONZERO_auxiliary": int(live_aux.sum()),
                "max_abs_source_in_the_biting_region": float(
                    cupy.abs(volume * mask).max()),
            })

    for side in SIDES:
        components = record["sides"][side]["per_component"]
        record["sides"][side]["guard_is_exercisable"] = bool(any(
            entry["biting_cells_with_a_NONZERO_source"] > 0
            or entry["biting_cells_with_a_NONZERO_auxiliary"] > 0
            for entry in components.values()))
    record["guard_is_exercisable"] = bool(
        any(record["sides"][side]["guard_is_exercisable"] for side in SIDES))

    driver.close()
    cupy.get_default_memory_pool().free_all_blocks()
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    curl_leg = load_curl_leg()
    if curl_leg.CASES_DIR not in sys.path:
        sys.path.insert(0, curl_leg.CASES_DIR)

    results: Dict[str, Any] = {
        "check": "constitutive_absorber_reach",
        "_question": (
            "In the window the lifted leg compares (2 warm-up + 8 whole steps), "
            "is there any cell where the constitutive coefficient is NOT exactly "
            "1.0 AND the operand is nonzero? If not, --fmad=false cannot be "
            "distinguished on these cases, because 1.0f*x is exact and fma(1,x,f) "
            "== f+x bit for bit."),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cases": [],
    }
    plan = [(name, kwargs, courant) for courant in COURANTS for name, kwargs in CASES]
    for index, (name, kwargs, courant) in enumerate(plan, 1):
        started = time.time()
        try:
            record = one_case(name, kwargs, courant)
        except Exception as exc:  # noqa: BLE001 - a failure is a row
            record = {"case": name, "kwargs": kwargs, "courant": float(courant),
                      "error": f"{type(exc).__name__}: {exc}"[:2000]}
        record["seconds"] = round(time.time() - started, 2)
        results["cases"].append(record)
        if "error" in record:
            log(f"[reach] {index}/{len(plan)} {name} C={courant}: FAILED {record['error']}")
        else:
            summary = {
                side: sum(entry["biting_cells_with_a_NONZERO_source"]
                          for entry in record["sides"][side]["per_component"].values())
                for side in SIDES}
            log(f"[reach] {index}/{len(plan)} {name} {kwargs} C={courant} "
                f"shape={record['shape']}: guard_exercisable="
                f"{record['guard_is_exercisable']} "
                f"nonzero_source_cells_where_coefficient_bites={summary} "
                f"({record['seconds']} s)")
        with open(args.out + ".tmp", "w") as handle:
            json.dump(results, handle, indent=2, default=str)
        os.replace(args.out + ".tmp", args.out)

    good = [c for c in results["cases"] if "error" not in c]
    results["summary"] = {
        "cases": len(good),
        "cases_where_the_guard_is_exercisable": sum(
            1 for c in good if c["guard_is_exercisable"]),
        "_reading": (
            "0 exercisable means the lifted corpus leg CANNOT be evidence for the "
            "contraction guard on this sub-step, whatever its verdict says. It "
            "remains evidence for the transcription — indexing, sub-lattice, "
            "grouping, the auxiliary recurrence — against the driver's own tables "
            "at corpus scale. The guard evidence is the synthetic gate's "
            "unguarded control."),
    }
    with open(args.out + ".tmp", "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    os.replace(args.out + ".tmp", args.out)
    log(f"[done] {results['summary']['cases_where_the_guard_is_exercisable']}"
        f"/{results['summary']['cases']} cases can exercise the contraction guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
