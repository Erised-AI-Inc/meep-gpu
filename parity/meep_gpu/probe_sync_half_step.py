#!/usr/bin/env python3
"""Compare the three legs INSIDE ``synchronize_magnetic_fields``, sub-step by sub-step.

The route gate compares only the state AFTER the whole half-step, so a divergence
that enters at one sub-step and a divergence that enters at another are the same
row to it. This probe snapshots every leg's full state between every pass of the
half-step -- the curl, the source inject, the near fill, the wall clear, the far
fill, the constitutive, and the average -- and reports the FIRST canonical stage at
which any pair of legs disagrees, per array, with the subnormal forensics the byte
comparator already carries.

The passes are observed rather than re-implemented: the driver's own module-level
array functions and ``FastPathPlan.dispatch`` are wrapped, so what this records is
the sequence the driver actually ran, dispatches included.

WHAT IT MEASURED, 2026-09-11, on the GPU host RTX A6000s (GPUs 1 and 6, Triton table,
the shipped composition). ``folded_dispersive_2d`` and ``folded_2d`` behave
IDENTICALLY at every sub-step, at resolution 20 and 30, after the full 1600-step
ladder:

* the ``unfused`` leg -- certified single arms, no fused product -- is byte-equal
  to the ``array`` leg at EVERY stage, entry to average;
* the ``fused`` leg differs from both at ``curl_B`` (5 arrays) and stays different
  through the fills, then RE-CONVERGES exactly at ``constitutive_H``. That is the
  fused pair working: its ``step_B`` launch has already performed ``update_H``, so
  it carries an advanced H through the stages where the other two have not reached
  it yet. The difference is the substitution, not a defect -- and it doubles as this
  probe's arming control, because a comparator that could not see it would report
  the same all-clear on a seam that was genuinely broken.

So a report of a kernel-vs-array divergence at this site is not a statement about
any sub-step of the half-step on these shapes. Re-run this before believing one.

Run (the GPU host, one pinned GPU, the route lane's environment)::

    CUDA_VISIBLE_DEVICES=<idx> python -u parity/meep_gpu/probe_sync_half_step.py \\
        --gpu-id 0 --case folded_dispersive_2d --out <dir>/probe.json

Rule 7: one flushed line per leg per chunk and per stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# The root is the nearest ancestor that holds both the package and the harness
# (``meep_gpu/`` and ``parity/meep_gpu/``): the repository root of this layout,
# and the same directory in any tree the harness is copied into whole.
_API = HERE
while _API != os.path.dirname(_API) and not (
        os.path.isdir(os.path.join(_API, "meep_gpu"))
        and os.path.isdir(os.path.join(_API, "parity", "meep_gpu"))):
    _API = os.path.dirname(_API)
if not os.path.isdir(os.path.join(_API, "parity", "meep_gpu")):  # pragma: no cover
    raise SystemExit(f"cannot locate the repository root above {HERE}: no ancestor "
                     f"holds both meep_gpu/ and parity/meep_gpu/")
for path in (os.path.join(_API, "parity", "meep_gpu"), _API):
    if path not in sys.path:
        sys.path.insert(0, path)


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


#: driver module function name -> the canonical sub-step it performs.
ARRAY_PASSES = {
    "step_B": "curl_B",
    "fill_symmetry_bc_B": "fill_near_B",
    "zero_metal_B": "zero_metal_B",
    "fill_folded_far_ghosts_B": "fill_far_B",
    "update_H": "constitutive_H",
}
#: dispatch slot name -> the same canonical sub-step.
SLOT_PASSES = {
    "step_B": "curl_B",
    "fill_B": "fill_near_B",
    "fill_folded_far_ghosts_B": "fill_far_B",
    "update_H": "constitutive_H",
    "update_H_synchronize": "constitutive_H",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="folded_dispersive_2d")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    import gate_dispatch_end_to_end as e2e
    import gate_dispatch_fused_route as route

    # AN NVIDIA PROBE, REFUSED OFF NVIDIA: its legs lift ``prefer_gpu=True``, which
    # on an Apple GPU resolves Metal drivers this probe's Triton counter cannot see
    # (it would write a stage-by-stage artifact reading 0 launches).
    refusal = e2e.nvidia_host_refusal("probe_sync_half_step.py")
    if refusal:
        print(refusal.replace("; pass --smoke to exercise the harness on the NumPy "
                              "reference", ""), file=sys.stderr)
        return 2

    counter = e2e.TritonLaunchCounter()
    try:
        counter.install()
    except Exception as exc:  # noqa: BLE001
        say(f"no Triton launch counter ({exc!r})")
    e2e.PREFER_GPU = True

    case = arguments.case
    builder = route.CASES[case]
    spec = route.drive_table()[case]

    say(f"{case}: lifting three legs")
    legs = {
        "fused": e2e.run_leg(case, builder, "fused",
                             route._fuse_env(spec["arms"],
                                             spec.get("reached_by", "opt-in")),
                             counter, arguments.gpu_id, arguments.resolution),
        "unfused": e2e.run_leg(case, builder, "unfused", route.unfused_env(),
                               counter, arguments.gpu_id, arguments.resolution),
        "array": e2e.run_leg(case, builder, "array", route.ARRAY_ENV, counter,
                             arguments.gpu_id, arguments.resolution),
    }
    report = {"case": case, "steps": arguments.steps,
              "grid_shape": legs["fused"]["grid_shape"],
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    pre = {label: e2e.collect_state(leg["driver"]) for label, leg in legs.items()}
    ok_fa = e2e.compare_state(pre["fused"], pre["array"])
    ok_ua = e2e.compare_state(pre["unfused"], pre["array"])
    report["precondition"] = {"fused_vs_array": ok_fa["identical"],
                              "unfused_vs_array": ok_ua["identical"],
                              "arrays": ok_fa["arrays"],
                              "words": ok_fa["words_compared"]}
    say(f"{case}: precondition {report['precondition']}")
    if not (ok_fa["identical"] and ok_ua["identical"]):
        say("REFUSING: the legs did not lift to the same state")
        return 2

    # --- step the ladder, alternating, exactly as the gate does ---------------
    taken = 0
    for point in route.ladder_for(arguments.steps):
        for leg in legs.values():
            route.step_leg(case, leg, counter, point - taken)
        taken = point
        snap = {label: e2e.collect_state(leg["driver"]) for label, leg in legs.items()}
        fa = e2e.compare_state(snap["fused"], snap["array"])
        ua = e2e.compare_state(snap["unfused"], snap["array"])
        say(f"{case}: checkpoint {point}/{arguments.steps} -> "
            f"fused{'==' if fa['identical'] else '!='}array "
            f"unfused{'==' if ua['identical'] else '!='}array")
        if not (fa["identical"] and ua["identical"]):
            report["diverged_during_step"] = {"steps": point,
                                              "fused": fa["differences"][:4],
                                              "unfused": ua["differences"][:4]}
            say("the legs diverged during step(); the half-step probe below is "
                "about a state that already differs")
            break
    report["state_before_sync_identical"] = {
        "fused_vs_array": e2e.compare_state(
            e2e.collect_state(legs["fused"]["driver"]),
            e2e.collect_state(legs["array"]["driver"]))["identical"],
        "unfused_vs_array": e2e.compare_state(
            e2e.collect_state(legs["unfused"]["driver"]),
            e2e.collect_state(legs["array"]["driver"]))["identical"]}
    say(f"{case}: state immediately before the half-step: "
        f"{report['state_before_sync_identical']}")

    # --- instrument the half-step ---------------------------------------------
    from meep_gpu import driver as driver_mod
    from meep_gpu import fastpath as fastpath_mod

    trace: list = []          # [(stage, how, state)] for the leg being driven
    current = {"driver": None}

    def snapshot(stage: str, how: str) -> None:
        driver = current["driver"]
        if driver is None:
            return
        trace.append((stage, how, e2e.collect_state(driver)))

    originals = {name: getattr(driver_mod, name) for name in ARRAY_PASSES}

    def wrap_array(name: str):
        original = originals[name]

        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            snapshot(ARRAY_PASSES[name], f"array:{name}")
            return out
        return wrapper

    original_dispatch = fastpath_mod.FastPathPlan.dispatch

    def wrapped_dispatch(self, slot, fields):
        answer = original_dispatch(self, slot, fields)
        if answer and slot in SLOT_PASSES:
            snapshot(SLOT_PASSES[slot], f"dispatch:{slot}")
        return answer

    traces = {}
    launches = {}
    for label, leg in legs.items():
        driver = leg["driver"]
        e2e._apply_env(leg)
        trace.clear()
        current["driver"] = driver
        snapshot("entry", "before the half-step")
        for name in ARRAY_PASSES:
            setattr(driver_mod, name, wrap_array(name))
        fastpath_mod.FastPathPlan.dispatch = wrapped_dispatch
        snap = counter.snapshot()
        try:
            driver.synchronize_magnetic_fields()
        finally:
            for name, original in originals.items():
                setattr(driver_mod, name, original)
            fastpath_mod.FastPathPlan.dispatch = original_dispatch
        launches[label] = e2e.TritonLaunchCounter.delta(snap, counter.snapshot())
        snapshot("averaged", "after the half-step returned")
        current["driver"] = None
        traces[label] = list(trace)
        say(f"{case}/{label}: half-step ran "
            f"{[f'{stage}<-{how}' for stage, how, _ in traces[label]]} "
            f"({launches[label]['total']} launches)")

    report["launches_during_sync"] = {k: v["total"] for k, v in launches.items()}
    report["passes"] = {label: [f"{stage}<-{how}" for stage, how, _ in tr]
                        for label, tr in traces.items()}

    # --- compare the legs at every canonical stage they share -----------------
    def last_state_at(tr, stage):
        found = None
        for name, _how, state in tr:
            if name == stage:
                found = state
        return found

    stages = ["entry", "curl_B", "fill_near_B", "zero_metal_B", "fill_far_B",
              "constitutive_H", "averaged"]
    comparisons = []
    for stage in stages:
        row = {"stage": stage}
        for pair in (("fused", "array"), ("unfused", "array"),
                     ("fused", "unfused")):
            a = last_state_at(traces[pair[0]], stage)
            b = last_state_at(traces[pair[1]], stage)
            key = f"{pair[0]}_vs_{pair[1]}"
            if a is None or b is None:
                row[key] = ("stage absent on "
                            + ("both" if a is None and b is None else
                               pair[0] if a is None else pair[1]))
                continue
            verdict = e2e.compare_state(a, b)
            row[key] = {"identical": verdict["identical"],
                        "arrays_differing": verdict["arrays_differing"],
                        "differences": verdict["differences"][:4]}
        comparisons.append(row)
        summary = {k: (v if isinstance(v, str)
                       else ("==" if v["identical"]
                             else f"DIFFERS in {v['arrays_differing']} arrays"))
                   for k, v in row.items() if k != "stage"}
        say(f"{case}: stage {stage:>15} -> {summary}")
    report["stages"] = comparisons

    for leg in legs.values():
        leg["driver"].restore_magnetic_fields()
        leg["driver"].close()

    first_bad = next((row["stage"] for row in comparisons
                      if any(isinstance(v, dict) and not v["identical"]
                             for k, v in row.items() if k != "stage")), None)
    report["first_divergent_stage"] = first_bad
    say(f"{case}: FIRST DIVERGENT STAGE = {first_bad!r}")
    if arguments.out:
        os.makedirs(os.path.dirname(arguments.out), exist_ok=True)
        with open(arguments.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=str)
        say(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
