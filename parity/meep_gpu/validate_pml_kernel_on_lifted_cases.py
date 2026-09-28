"""Bit-identity on the REAL benchmark cases, not on synthetic arrays.

The gate in ``probe_fused_kernel_bit_identity.py`` sweeps shapes, boundary
combinations and coefficient tables the kernel must handle. This asks a
narrower but more direct question: on the cases the benchmark actually times —
``2d_pml``, ``3d_pml``, ``2d_pml_geom``, lifted from ``mp.Simulation`` through
``from_meep`` exactly as a user's run is — do the two paths agree byte for byte
over consecutive sub-steps, on the DRIVER'S OWN arrays, with the DRIVER'S OWN
PML table, Courant number and resolved boundaries?

It also records what those resolved boundaries ARE, per case, which is the
evidence behind the boundary-kind hazard: the complex ``_pml`` kernels hard-code
"X and Y metallic, Z periodic", and a real case whose axes resolve otherwise is
where that assumption stops being harmless.

THE COURANT NUMBER IS A GATE AXIS HERE, AND UNTIL 2026-08-15 THIS LEG HAD ONLY
ONE VALUE OF IT. All six cases of the 2026-08-09/10 run had ``Courant = 0.5``,
so every one recorded ``dtdx_float32_exact: true`` — and the kernel module's own
docstring says what that means: "a dtdx that is a power of two hides the whole
effect (0.5 scaling is exact, so contracted and uncontracted round identically);
the divergence only shows with a dtdx that is not exactly representable, which is
the realistic case." So the leg that reads as "and it works on real corpus
problems up to 2560²" could not distinguish ``--fmad=false`` from the default
options at all. It was real evidence for the ghost rules, the metallic mask and
the coefficient lattice at corpus scale, and NOT independent evidence for the
arithmetic guard — which had therefore never been exercised above 64×48×1
(disposition §1.4).

Every case now runs at BOTH: 0.5 (the recorded regime, kept as the control) and
:data:`INEXACT_COURANT`, which is the synthetic gate's own second value so the
two legs stay commensurable. ``dtdx = grid.dt / grid.dx = courant``, and the
summary counts the two classes separately, because a pass built only out of
exact-dtdx cases is the thing this change exists to stop being quotable.

Protocol per case: freeze one driver, snapshot every array the sub-step touches,
run N array sub-steps, snapshot the result, restore, run N kernel sub-steps,
compare. One driver and one restore means the two legs cannot differ in their
inputs — which two separately-lifted drivers could, and silently.

Run (the GPU host, one GPU)::

    CUDA_VISIBLE_DEVICES=0 python -u validate_pml_kernel_on_lifted_cases.py \\
        --out results/pml_kernel_on_lifted_cases.json

One flushed line per case, incremental JSON (the progress-reporting rule).
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
# cases.py is the byte-for-byte copy the throughput harness uses, so this and
# the benchmark drive the identical problems.
CASES_DIR = os.path.join(HERE, "results", "fused_pml_throughput_hand_2026-08-09",
                         "scripts")

SUB_STEP_ARRAYS = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                   "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                   "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

# Covered cases at two sizes each: a small one that runs in a second and a
# larger one where the launch geometry differs (more blocks than SMs).
SHAPES = [
    ("2d_pml", {"res": 40, "n": 16}),
    ("2d_pml", {"res": 160, "n": 16}),
    ("3d_pml", {"res": 15, "n": 6}),
    ("3d_pml", {"res": 30, "n": 6}),
    ("2d_pml_geom", {"res": 40, "n": 16}),
    ("2d_pml_geom", {"res": 160, "n": 16}),
]

#: MEEP's default, exact in binary, and the value every case of the 2026-08-09/10
#: run used. Kept as the CONTROL, not dropped: it is the regime the recorded
#: verdict describes, and a leg that quietly replaced it would make the old and
#: new artifacts incomparable.
EXACT_COURANT = 0.5

#: Not exactly representable in float32 — the double nearest 0.35 is
#: 0.34999999999999998 — and the same second value the synthetic gate sweeps
#: (``probe_fused_kernel_bit_identity.PML_DTDX``), so the corpus leg and the
#: synthetic leg are asking about the same regime at two scales. Well under the
#: 2-D and 3-D CFL bounds, and LOWER than the default, so no case is destabilized
#: by being asked the question.
INEXACT_COURANT = 0.35

COURANTS = (EXACT_COURANT, INEXACT_COURANT)
PLAN = [(name, kwargs, courant) for courant in COURANTS for name, kwargs in SHAPES]
STEPS = 8


def log(message: str) -> None:
    print(message, flush=True)


def load_probe():
    path = os.path.join(HERE, "probe_fused_kernel_bit_identity.py")
    spec = importlib.util.spec_from_file_location("probe_bit_identity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def one_case(name: str, kwargs: Dict[str, Any], probe,
             courant: float = EXACT_COURANT) -> Dict[str, Any]:
    import cupy

    from meep_gpu import backends, stepping
    from meep_gpu.cuda_kernels import step_curl_kernels as kernels
    from meep_gpu.from_meep import lift_simulation
    import cases

    backends.guard_kernel_compilation(cupy)
    simulation = cases.BUILDERS[name](**kwargs)
    # dtdx IS the Courant number: Grid sets dt = courant/resolution and
    # dx = 1/resolution, and from_meep reads sim.Courant straight through. Setting
    # it on the built simulation is the whole lever, and it needs no change to
    # cases.py — which lives in an untracked results/ directory and is the same
    # byte-for-byte copy the throughput harness drives.
    simulation.Courant = float(courant)
    driver = lift_simulation(simulation, prefer_gpu=True, gpu_id=0)
    # Two real steps first, so the arrays hold a simulation's values and every
    # lazily-allocated buffer exists before either leg runs.
    driver.run(num_steps=2)

    fields, pml, grid = driver.fields, driver.pml, driver.grid
    resolved = kernels.real_pml_boundary_kinds(grid)
    # BOTH sub-steps, because this leg runs both kernels in one loop below and the
    # predicate has been per-sub-step since 2026-08-15. A case admitted at step_B
    # and refused at step_D is a real configuration (a D conductivity is one), and
    # comparing it here would be comparing the array path against a kernel this
    # predicate never admitted — so the leg requires both and records each.
    per_sub_step = {sub_step: kernels.covers_real_pml_curl(fields, pml, grid, sub_step)
                    for sub_step in ("step_B", "step_D")}
    covered = all(verdict for verdict, _ in per_sub_step.values())
    reason = next((why for verdict, why in per_sub_step.values() if not verdict),
                  "covered")
    record: Dict[str, Any] = {
        "case": name, "kwargs": kwargs,
        "coverage_by_sub_step": {sub_step: {"covered": bool(verdict), "reason": why}
                                 for sub_step, (verdict, why) in per_sub_step.items()},
        "courant": float(courant),
        "shape": [int(s) for s in driver.shape],
        "resolved_boundaries": list(resolved),
        "dtdx": float(grid.dt / grid.dx),
        "dtdx_float32_exact": bool(float(np.float32(grid.dt / grid.dx))
                                   == float(grid.dt / grid.dx)),
        "pml_thickness_by_face": [list(f) for f in pml.thickness_by_face],
        "covered": covered, "coverage_reason": reason,
        "steps": STEPS,
    }
    if not covered:
        driver.close()
        return record

    baseline = {name_: getattr(fields, name_).copy() for name_ in SUB_STEP_ARRAYS}

    for _ in range(STEPS):
        stepping.step_B(fields, pml)
        stepping.step_D(fields, pml)
    array_result = {n: getattr(fields, n).copy() for n in SUB_STEP_ARRAYS}

    for n, value in baseline.items():
        getattr(fields, n)[...] = value

    tables_b = kernels.real_pml_curl_tables(pml, half_integer=True)
    tables_d = kernels.real_pml_curl_tables(pml, half_integer=False)
    # THE GRID, NOT ``resolved``: a folded axis resolves to "mirror" at both
    # terminations and the two need different codes, so the split is asked of the
    # grid (``real_curl_boundary_codes``) rather than of the kind strings.
    codes = kernels.real_curl_boundary_codes(grid)
    dtdx = grid.dt / grid.dx
    for _ in range(STEPS):
        kernels._step_B_fused_pml_real(fields, tables_b, codes, dtdx)
        kernels._step_D_fused_pml_real(fields, tables_d, codes, dtdx)
    cupy.cuda.runtime.deviceSynchronize()

    parts = {n: probe.bit_compare(getattr(fields, n), array_result[n])
             for n in SUB_STEP_ARRAYS}
    record["vs_array_path"] = probe.combine(parts)

    # LIVENESS, or the agreement is agreement about nothing. Two traps here, and
    # the first version of this check fell into the second:
    #
    #  1. Zero == zero is bit-identical. If the sub-step moved nothing, the
    #     comparison passes for the wrong reason, so at least one target must
    #     have changed and the largest magnitude in play is recorded.
    #  2. NOT EVERY COMPONENT IS LIVE. A 2-D Ez (TM) run carries Ez, Hx and Hy
    #     and holds the other half of the Yee cell at exactly zero forever, so
    #     "every target moved" is false on a perfectly healthy case. The
    #     criterion is therefore how MANY moved, reported per component, with
    #     "at least one" as the gate.
    targets = [n for n in SUB_STEP_ARRAYS if not n.startswith(("E", "H"))]
    sources = [n for n in SUB_STEP_ARRAYS if n.startswith(("E", "H"))]
    changed = {n: not bool((array_result[n] == baseline[n]).all()) for n in targets}
    record["targets_changed"] = changed
    record["targets_changed_count"] = f"{sum(changed.values())}/{len(targets)}"
    record["any_target_changed"] = any(changed.values())
    record["max_abs_after_array_substeps"] = {
        n: float(abs(array_result[n]).max()) for n in SUB_STEP_ARRAYS}
    record["largest_target_magnitude"] = max(
        record["max_abs_after_array_substeps"][n] for n in targets)
    record["sources_untouched"] = all(
        bool((array_result[n] == baseline[n]).all()) for n in sources)

    driver.close()
    cupy.get_default_memory_pool().free_all_blocks()
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("flush", "keep", "match_meep"),
                        help="INSTALL this float32 subnormal policy before the "
                             "first compile, and STAMP the artifact with it. "
                             "Omitted, nothing is installed — CuPy compiles "
                             "natively with its own -ftz=true — and the artifact "
                             "records that rather than leaving the question "
                             "unanswerable, which is what the 2026-08-09/10 "
                             "record silently did (disposition §1.1).")
    parser.add_argument("--import-meep-for-host-policy", action="store_true",
                        help="import MEEP before installing the policy, so the "
                             "HOST half of 'flush' is attainable. Measured: "
                             "without it a flush install RAISES, because "
                             "mp.set_zero_subnormals is the only route to this "
                             "process's FTZ/DAZ bits.")
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if CASES_DIR not in sys.path:
        sys.path.insert(0, CASES_DIR)
    probe = load_probe()

    # BEFORE the first kernel compiles, and before the probe's own observer would
    # have anything to see: install-then-compile and compile-then-install are
    # measurably different runs (probe.install_subnormal_policy_for_run).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", ".."))
    binary_observer = probe.install_nvrtc_binary_observer()
    meep_import = (probe.import_meep_for_host_policy()
                   if args.import_meep_for_host_policy else {"requested": False})
    policy_install = (probe.install_subnormal_policy_for_run(args.subnormal_policy,
                                                            repo_root)
                      if args.subnormal_policy else None)

    results: Dict[str, Any] = {
        "check": "pml_kernel_on_lifted_cases",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sub_steps_per_case": STEPS,
        "courants": list(COURANTS),
        "_why_two_courants": (
            "dtdx = courant, and at 0.5 the scaling is exact in float32 so a "
            "contracted and an uncontracted stencil round identically. A leg run "
            "only at 0.5 cannot distinguish --fmad=false from the default options "
            "and is not evidence for the arithmetic guard at any scale."),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR", "<unset>"),
        "subnormal_policy_requested": args.subnormal_policy,
        "subnormal_policy_install": policy_install,
        "subnormal_policy": probe.subnormal_policy_stamp(repo_root),
        "meep_import_for_host_policy": meep_import,
        "nvrtc_binary_observer": binary_observer,
        "cases": [],
    }
    started = time.time()
    for index, (name, kwargs, courant) in enumerate(PLAN, 1):
        case_started = time.time()
        try:
            record = one_case(name, kwargs, probe, courant)
        except Exception as exc:  # noqa: BLE001 - a failure is a row
            record = {"case": name, "kwargs": kwargs, "courant": float(courant),
                      "error": f"{type(exc).__name__}: {exc}"[:2000]}
        record["seconds"] = round(time.time() - case_started, 2)
        results["cases"].append(record)
        if "error" in record:
            log(f"[lifted] {index}/{len(PLAN)} {name} {kwargs} C={courant}: "
                f"FAILED {record['error']}")
        elif not record["covered"]:
            log(f"[lifted] {index}/{len(PLAN)} {name} {kwargs} C={courant}: NOT COVERED "
                f"({record['coverage_reason']})")
        else:
            verdict = record["vs_array_path"]
            log(f"[lifted] {index}/{len(PLAN)} {name} {kwargs} C={courant} "
                f"shape={record['shape']} bc={'/'.join(record['resolved_boundaries'])} "
                f"dtdx={record['dtdx']:.6f} (f32-exact={record['dtdx_float32_exact']}): "
                f"identical={verdict['bit_identical']} over {STEPS} sub-steps "
                f"(differing={verdict['differing_floats']}/{verdict['total_floats']}, "
                f"maxulp={verdict.get('max_ulp', 0)}) "
                f"targets_moved={record['targets_changed_count']} "
                f"maxabs={record['largest_target_magnitude']:.3e} "
                f"sources_untouched={record['sources_untouched']} "
                f"({record['seconds']} s)")
        with open(args.out + ".tmp", "w") as handle:
            json.dump(results, handle, indent=2, default=str)
        os.replace(args.out + ".tmp", args.out)

    covered: List[Dict[str, Any]] = [c for c in results["cases"]
                                     if c.get("covered") and "error" not in c]
    identical = [c for c in covered if c["vs_array_path"]["bit_identical"]]
    inexact = [c for c in covered if not c["dtdx_float32_exact"]]
    inexact_identical = [c for c in inexact if c["vs_array_path"]["bit_identical"]]
    results["summary"] = {
        "covered": len(covered),
        "identical": len(identical),
        # The half that is evidence for the ARITHMETIC GUARD rather than for the
        # ghost rules. A run whose inexact-dtdx count is zero has not exercised
        # --fmad=false at corpus scale, whatever its total says.
        "covered_at_inexact_dtdx": len(inexact),
        "identical_at_inexact_dtdx": len(inexact_identical),
        "pass": bool(covered) and len(identical) == len(covered)
        and bool(inexact) and len(inexact_identical) == len(inexact)
        and all(c["any_target_changed"] and c["sources_untouched"]
                and c["largest_target_magnitude"] > 0.0 for c in covered),
    }
    # Re-read AFTER the run: the stamp taken before it describes intent, this one
    # describes what the process actually compiled under.
    results["subnormal_policy"] = probe.subnormal_policy_stamp(repo_root)
    results["nvrtc_binaries"] = probe.nvrtc_binary_report()
    log(f"[observer] nvrtc calls={results['nvrtc_binaries']['nvrtc_calls_observed']} "
        f"distinct_binaries={results['nvrtc_binaries']['distinct_binaries']} "
        f"ftz_true_reached_nvrtc="
        f"{results['nvrtc_binaries']['any_ftz_true_reached_nvrtc']}")
    results["elapsed_seconds"] = round(time.time() - started, 2)
    with open(args.out + ".tmp", "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    os.replace(args.out + ".tmp", args.out)
    log(f"[done] policy={results['subnormal_policy'].get('policy')} "
        f"{results['summary']['identical']}/{results['summary']['covered']} "
        f"lifted cases bit-identical over {STEPS} sub-steps "
        f"({results['summary']['identical_at_inexact_dtdx']}/"
        f"{results['summary']['covered_at_inexact_dtdx']} of them at an INEXACT "
        f"dtdx, which is the half that tests the arithmetic guard); "
        f"pass={results['summary']['pass']} ({results['elapsed_seconds']} s)")
    return 0 if results["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
