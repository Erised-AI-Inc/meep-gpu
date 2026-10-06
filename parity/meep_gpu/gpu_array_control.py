#!/usr/bin/env python
"""The array-path control as its OWN PROCESS: one leg, every kernel vetoed.

OPT-IN (``--array-control row`` in ``timing_ladder.py``). Every ``bench_fused_products``
row already times an ``array`` leg beside its fused leg, forced by the kill switch
(``MEEP_GPU_FUSED=0``); the ladder's default control is read off those legs. This
script is the standalone form: a fresh process that lifts ONE leg of the same case
through the harness's own ``run_leg`` and times it with the harness's own
``prime`` / ``freeze`` / ``size_chunk`` / ``window`` -- imported from the harness tree,
not re-spelt -- with the launch witnesses installed. The case's builder is the one
``timing_cases.resolve`` returns, the same object the GPU and MEEP rows build from.

HOW THE ARRAY PATH IS FORCED is an argument and is written into the row:

* ``dispatch`` (default)  ``MEEP_GPU_DISPATCH=0``, the opt-out a user types now that
  dispatch is on by default; ``MEEP_GPU_FUSED`` and ``MEEP_GPU_FUSE_ARMS`` unset.
* ``fused``               the harness's own ``ARRAY_ENV`` (the kill switch).

FLOORS. The launch witnesses must be installed; the warm pass and every timed window
must launch NOTHING; the frozen plan must name no dispatched table; the spread over
windows must be within the harness's gate. A row failing any is
``TIMED-NOT-REPORTABLE`` and says which.

Monitors stay ATTACHED, as the ladder's GPU rows have them. ``--smoke`` lifts the NumPy
reference and exercises the plumbing only; such a row is marked and is never a timing.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback
from typing import Any, Dict, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import timing_cases  # noqa: E402

HARNESS_MARKER = "bench_fused_products.py"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness-root", default=None, dest="harness_root")
    parser.add_argument("--case", default="pml_3d")
    parser.add_argument("--table", default=None, choices=("triton", "cuda"),
                        help="the NVIDIA kernel table whose GPU rows this case matches")
    parser.add_argument("--res", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--target-seconds", type=float, default=2.0,
                        dest="target_seconds")
    parser.add_argument("--step-cap", type=int, default=4000, dest="step_cap")
    parser.add_argument("--warm-steps", type=int, default=6, dest="warm_steps")
    parser.add_argument("--forced-by", default="dispatch", dest="forced_by",
                        choices=("dispatch", "fused"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    harness_root = os.path.abspath(args.harness_root
                                   or timing_cases.find_harness_root(HERE) or HERE)
    if not os.path.isfile(os.path.join(harness_root, HARNESS_MARKER)):
        raise SystemExit(f"REFUSING: no {HARNESS_MARKER} under {harness_root}")
    builder, builder_facts = timing_cases.resolve(args.case, harness_root, args.table)
    timing_cases.put_on_path(harness_root)
    import bench_fused_products as harness  # noqa: PLC0415

    e2e, route = harness.e2e, harness.route
    if args.smoke:
        e2e.PREFER_GPU = False
    else:
        refusal = e2e.nvidia_host_refusal("gpu_array_control.py")
        if refusal:
            raise SystemExit(refusal)
    route.BACKEND = "triton"
    if args.forced_by == "dispatch":
        environment: Dict[str, Optional[str]] = {
            "MEEP_GPU_DISPATCH": "0", "MEEP_GPU_FUSED": None,
            "MEEP_GPU_FUSE_ARMS": None, "MEEP_GPU_BACKEND_PREFERENCE": None}
    else:
        environment = dict(route.ARRAY_ENV)

    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "rows.jsonl")
    counter = harness.NvidiaWitness(e2e.LaunchCounters())
    harness.WITNESSES = counter.install()
    harness.say(f"ARRAY CONTROL case={args.case} res={args.res} forced_by="
                f"{args.forced_by} env={environment} witnesses="
                f"{harness.WITNESSES or 'NONE'}")
    stamp = harness.provenance("triton")
    stamp["sha256"]["array_control"] = harness._sha256(os.path.abspath(__file__))

    started = time.time()
    row: Dict[str, Any] = {
        "case": args.case, "kind": "array_control", "drive_table": None,
        "forced_by": args.forced_by, "environment_of_the_leg": environment,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "box_before": harness.box_state(), "monitors_mode": "attached",
        "witnesses_installed": list(harness.WITNESSES),
        "smoke": bool(args.smoke), "prefer_gpu": bool(e2e.PREFER_GPU),
        "argv": list(sys.argv[1:]) if argv is None else list(argv),
        "provenance": stamp, "builder": builder_facts,
    }
    if not timing_cases.is_builtin(args.case):
        row["timing_case"] = timing_cases.case_identity(args.case)
    leg = None
    try:
        leg = e2e.run_leg(args.case, builder, "array", environment,
                          counter, args.gpu, args.res)
        driver = leg["driver"]
        row["grid_shape"] = leg["grid_shape"]
        row["lift_s"] = leg["lift_s"]
        row["array_module"] = getattr(driver.xp, "__name__", None)
        harness.prime(args.case, leg, counter, args.warm_steps)
        plan = leg["plan"]
        row["plan"] = plan
        row["active_step_path"] = leg.get("active_step_path")
        row["warm_launches_per_step"] = leg["warm_launches_per_step"]
        row["subnormal_policy"] = harness.policy_brief()
        row["monitors_attached"] = {
            name: len(getattr(driver, name, []) or [])
            for name in harness.MONITOR_LISTS}
        frozen = harness.freeze(driver)
        sizing = harness.size_chunk(leg, frozen, harness.sync_for(driver),
                                    args.target_seconds, args.step_cap)
        chunk = sizing["steps_per_window"]
        row["sizing"] = sizing
        windows = [harness.window(args.case, leg, frozen, counter, chunk, index)
                   for index in range(args.repeats)]
        row["windows"] = windows
        seconds = [w["seconds"] for w in windows]
        per_step = statistics.median(seconds) / float(chunk)
        cells = 1
        for count in leg["grid_shape"]:
            cells *= int(count)
        launches = sum(int(w["launches"]) for w in windows)
        tables = list((plan.get("composition") or {}).get("tables_dispatched") or [])
        row["cells"] = cells
        row["per_leg"] = {"array": {
            "windows": len(seconds), "steps_per_window": chunk,
            "median_seconds": statistics.median(seconds),
            "median_seconds_per_step": per_step,
            "spread": harness.spread(seconds),
            "launches_per_step": launches / float(chunk * len(windows)),
            "mcell_steps_per_s": cells / per_step / 1e6,
        }}
        row["tables_dispatched"] = tables
        row["dispatched_slots"] = plan.get("dispatched_slots")
        row["non_finite"] = harness.non_finite(driver)
        row["moved_words"] = harness.moved_words(driver, frozen)
        row["floors"] = {
            "launch_witness_installed": bool(harness.WITNESSES),
            "warm_pass_launched_nothing": leg["warm_launches_per_step"] == 0.0,
            "windows_launched_nothing": launches == 0,
            "no_table_dispatched": not tables and not plan.get("dispatched_slots"),
            "spread_within_gate": harness.spread(seconds) <= harness.SPREAD_GATE,
            "non_finite_zero": row["non_finite"] == 0,
            "moved_words_positive": row["moved_words"] > 0,
            "gpu_driver": not args.smoke,
        }
        row["reportable"] = all(row["floors"].values())
        row["verdict"] = "TIMED" if row["reportable"] else "TIMED-NOT-REPORTABLE"
        harness.say(f"{args.case}/array control: {row['verdict']} "
                    f"{row['per_leg']['array']['mcell_steps_per_s']:.2f} Mcell-steps/s "
                    f"({1e3 * per_step:.4f} ms/step, spread "
                    f"{row['per_leg']['array']['spread']:.3f}, launches {launches}, "
                    f"tables {tables or 'none'}, lift {leg['lift_s']} s)"
                    + ("" if row["reportable"] else "  FLOORS FAILED: "
                       f"{[k for k, v in row['floors'].items() if not v]}"))
    except Exception as error:  # noqa: BLE001 - the row lands whatever happened
        row["verdict"] = "ERROR"
        row["error"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc()
        harness.say(f"{args.case}/array control: ERROR {row['error']}")
    finally:
        row["box_after"] = harness.box_state()
        row["elapsed_s"] = round(time.time() - started, 1)
        if leg is not None:
            try:
                leg["driver"].close()
            except Exception:  # noqa: BLE001
                pass
        harness.append_jsonl(rows_path, row)
    return 0 if row.get("reportable") else 1


if __name__ == "__main__":
    raise SystemExit(main())
