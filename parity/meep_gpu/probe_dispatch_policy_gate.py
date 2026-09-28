"""The dispatch subnormal-policy gate, exercised on the real seam, one state per process.

WHAT THE END-TO-END GATE ALREADY SHOWS, and what it does not. Re-running
``gate_dispatch_end_to_end.py`` on this tree with nothing installing a policy is
the headline: the shipped path is byte-identical to the kill-switched array path
across every dispatching case, where before the same command diverged on six of
them. That measures the HAPPY PATH. It does not measure what the gate does to a
process that made the policy decision itself, or that made it WRONG — and those
are the legs where a guard is usually discovered to be decorative.

So each case below puts the process into one state, lifts a real
``mp.Simulation`` through ``lift_simulation``, takes one real ``driver.step()``
and reads the driver's own artifact. One state per PROCESS is not tidiness: a
subnormal policy locks on first install (CuPy's cache key is computed above the
strip seam, so a swap would serve one policy's binaries under the other's name),
so two states in one process is a state neither of them.

    preinstalled_keep   the caller installed the certified policy first
                        -> DISPATCHES, and does not install a second time
    preinstalled_flush  the caller installed the other policy first
                        -> REFUSES by name, array path, no exception
    partial_install     the caller drove host+CuPy and left Triton alone
                        -> REFUSES naming triton: the split with an extra step
    shipped_census      nothing installed; after one dispatching step, read the
                        PTX BOTH executors actually generate

WHY shipped_census REVERSES ``probe_ftz_ptx_census``'S PHASES. That probe reads
the CuPy leg first (kill-switched) and the Triton leg second, which was right for
attributing the DEFECT: the two executors were independent and either order gave
the same answer. It is wrong for measuring the FIX, whose whole mechanism is that
the plan freeze installs a policy before either executor compiles — read in the
old order the CuPy phase runs before any plan exists and reports the pre-install
binaries, which is a statement about a configuration this tree no longer has. The
phases are the same functions, imported and called in the other order, with
CuPy's in-process memo dropped between them so the array sub-steps actually
recompile and can be censused rather than served from memory.

Rule 7: one flushed line per step, and the artifact is saved as each phase lands.

Run (the GPU host, ONE pinned GPU), one case per process::

    CUDA_VISIBLE_DEVICES=<idx> python -u parity/meep_gpu/probe_dispatch_policy_gate.py \\
        --case preinstalled_flush --out results/<tag>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)

from probe_ftz_ptx_census import (  # noqa: E402
    build_case, environment, run_cupy_phase, run_triton_phase, say,
)

CASES = ("preinstalled_keep", "preinstalled_flush", "partial_install",
         "shipped_census")


def save(artifact: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=1, default=str)
        handle.flush()


def one_dispatching_step(case: str, res: Optional[int], gpu_id: int) -> Dict[str, Any]:
    """Lift a real simulation, take ONE real step, and read the driver's artifact.

    The step is the point. ``plan_fast_path`` runs at the driver's configuration
    freeze, which IS the first step, so a probe that only built a driver would
    never reach the rung it is here to measure.
    """
    import meep_gpu  # noqa: PLC0415

    os.environ["MEEP_GPU_DISPATCH"] = "1"
    os.environ.pop("MEEP_GPU_FUSED", None)
    sim, _monitors, _until = build_case(case, res)
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=gpu_id)
    say(f"lifted {tuple(int(v) for v in driver.shape)}; stepping once")
    driver.step()
    try:
        driver.xp.cuda.runtime.deviceSynchronize()
    except Exception:  # noqa: BLE001 - a NumPy host has nothing to synchronize
        pass
    report = driver.fast_path_report() or {}
    subnormal = report.get("subnormal") or {}
    return {
        "step_path": getattr(driver, "active_step_path", None),
        "decision": report.get("decision"),
        "refused_because": report.get("refused_because"),
        "gate": subnormal.get("gate"),
        "stamp": subnormal.get("stamp"),
        "dispatched_slots": sorted(
            name for name, entry in (report.get("slots") or {}).items()
            if entry.get("state") == "dispatched"),
    }


def preinstall(policy: str, executors: Optional[tuple] = None) -> Dict[str, Any]:
    """Install a policy the way a caller would, BEFORE anything lifts or compiles.

    MEEP first, on every leg: ``install_host_policy('flush')`` refuses outright
    without it, and a leg that imported MEEP only when it needed to would differ
    from its siblings in more than the policy.
    """
    import meep as mp  # noqa: PLC0415

    from meep_gpu import backends  # noqa: PLC0415
    from meep_gpu import subnormal_policy as sp  # noqa: PLC0415

    record: Dict[str, Any] = {
        "requested": policy,
        "executors": list(executors) if executors else ["host", "cupy", "triton"],
        "meep": str(getattr(mp, "__version__", "unknown")),
        "host_flushing_after_meep_import": bool(backends.subnormals_flushed()),
    }
    kwargs: Dict[str, Any] = {"strict": True}
    if executors is not None:
        kwargs["executors"] = executors
    try:
        record["stamp"] = sp.install_subnormal_policy(policy, **kwargs)
        record["installed"] = True
    except Exception as exc:  # noqa: BLE001 - a refusal is a measurement here
        record["installed"] = False
        record["refused_with"] = f"{type(exc).__name__}: {exc}"
        if policy == "flush" and executors is None:
            # MEASURED, and the reason the "uniform flush" state cannot simply be
            # asked for on this host: CuPy's CUB reduction accelerators are fixed
            # at import and keep subnormals, so flush needs CUPY_ACCELERATORS=''
            # set before CuPy is imported. Fall back to the host leg alone, which
            # is still a process with a DIFFERENT policy installed — the state
            # this case exists to put in front of the gate.
            say("flush refused for all executors; installing it on the host alone")
            record["fallback"] = "host only"
            record["stamp"] = sp.install_subnormal_policy(
                policy, strict=False, executors=("host",))
            record["installed"] = True
    record["policy_is_installed"] = sp.policy_is_installed()
    record["policy_in_force"] = (sp.get_subnormal_policy()
                                 if sp.policy_is_installed() else None)
    return record


def expect(artifact: Dict[str, Any], name: str, condition: bool, detail: Any = None) -> bool:
    artifact.setdefault("checks", {})[name] = {"pass": bool(condition), "detail": detail}
    say(f"CHECK {name}: {'PASS' if condition else 'FAIL'}"
        + (f" -- {detail}" if detail is not None else ""))
    return bool(condition)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--sim-case", default="pml_2d",
                        help="which built simulation to lift (probe_ftz_ptx_census's set)")
    parser.add_argument("--res", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    artifact_path = os.path.join(args.out, f"{args.case}.json")
    artifact: Dict[str, Any] = {
        "case": args.case, "sim_case": args.sim_case,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    say(f"case {args.case} on {args.sim_case}")
    if args.case == "preinstalled_keep":
        artifact["preinstall"] = preinstall("keep")
        save(artifact, artifact_path)
        artifact["environment"] = environment()
        artifact["run"] = one_dispatching_step(args.sim_case, args.res, args.gpu)
        run, gate = artifact["run"], artifact["run"]["gate"] or {}
        ok = expect(artifact, "dispatched", run["decision"] == "dispatched",
                    run["refused_because"])
        ok &= expect(artifact, "kernels_ran", bool(run["dispatched_slots"]),
                     run["dispatched_slots"])
        ok &= expect(artifact, "did_not_reinstall",
                     "dispatch" not in str(gate.get("installed_by")),
                     gate.get("installed_by"))
        ok &= expect(artifact, "policy_in_force_is_the_certified_one",
                     gate.get("policy_in_force") == "keep", gate.get("policy_in_force"))

    elif args.case == "preinstalled_flush":
        artifact["preinstall"] = preinstall("flush")
        save(artifact, artifact_path)
        artifact["environment"] = environment()
        artifact["run"] = one_dispatching_step(args.sim_case, args.res, args.gpu)
        run, gate = artifact["run"], artifact["run"]["gate"] or {}
        reason = str(run["refused_because"])
        ok = expect(artifact, "refused", run["decision"] == "refused", reason)
        ok &= expect(artifact, "fell_back_to_the_array_path",
                     run["step_path"] == "array", run["step_path"])
        ok &= expect(artifact, "no_slot_dispatched", not run["dispatched_slots"],
                     run["dispatched_slots"])
        ok &= expect(artifact, "reason_names_both_policies",
                     "'flush'" in reason and "'keep'" in reason, reason[:200])
        ok &= expect(artifact, "reason_is_actionable", "fresh process" in reason)

    elif args.case == "partial_install":
        artifact["preinstall"] = preinstall("keep", executors=("host", "cupy"))
        save(artifact, artifact_path)
        # THE PRECONDITION IS THE CASE. Measured once and caught here: with a
        # CUPY_CACHE_DIR lacking the keep token the caller's own install is
        # REFUSED, the module rolls the process back to nothing installed, and
        # what follows measures the ordinary shipped path instead of the split
        # this case exists to construct. A case that quietly measures a different
        # state is worse than one that fails.
        from meep_gpu import subnormal_policy as sp  # noqa: PLC0415

        state = {"policy_is_installed": sp.policy_is_installed(),
                 "triton_report": sp.executor_report("triton"),
                 "cupy_report_attained": sp.executor_report("cupy").get("attained")}
        artifact["precondition"] = state
        if not (state["policy_is_installed"] and not state["triton_report"]
                and state["cupy_report_attained"]):
            expect(artifact, "precondition_the_split_was_constructed", False, state)
            artifact["all_checks_passed"] = False
            artifact["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save(artifact, artifact_path)
            say("ABORT: the host+CuPy-only install did not take, so there is no "
                "split to put in front of the gate")
            return 1
        expect(artifact, "precondition_the_split_was_constructed", True, state)
        artifact["environment"] = environment()
        artifact["run"] = one_dispatching_step(args.sim_case, args.res, args.gpu)
        run = artifact["run"]
        reason = str(run["refused_because"])
        ok = expect(artifact, "refused", run["decision"] == "refused", reason)
        ok &= expect(artifact, "fell_back_to_the_array_path",
                     run["step_path"] == "array", run["step_path"])
        ok &= expect(artifact, "reason_names_the_undriven_executor",
                     "triton" in reason, reason[:200])
        ok &= expect(artifact, "no_slot_dispatched", not run["dispatched_slots"],
                     run["dispatched_slots"])

    else:  # shipped_census
        artifact["environment"] = environment()
        artifact["preinstall"] = {"installed": False,
                                  "note": "nothing calls install_subnormal_policy; "
                                          "the gate is what installs, at the freeze"}
        say("phase 1/2: Triton dispatch (this is what installs the policy)")
        artifact["triton"] = run_triton_phase(args.sim_case, args.res, args.gpu,
                                              None, args.out, artifact, artifact_path)
        # DROP THE IN-PROCESS MEMO between the phases. The dispatching step above
        # already ran the array sub-steps it did NOT dispatch, so their kernels are
        # memoized and the census below would attribute zero compiles to them and
        # report an empty CuPy leg as agreement. Dropping it makes them recompile —
        # under the policy now in force, which is the thing being read.
        import cupy  # noqa: PLC0415

        cupy._util.clear_memo()
        # AND A COLD DISK CACHE, for the same reason one layer down. The memo
        # drop makes CuPy look the kernel up again; the DISK cache then serves it
        # without an NVRTC call and the census reads nothing. Measured: with only
        # the memo dropped the whole-step CuPy census fell to 7 audited f32
        # instructions against the 19 a cold process compiles. A fresh directory
        # is used rather than emptying the one the gate chose, so nothing this
        # probe made is deleted; it still carries the policy token, because that
        # is the rule for a keep-policy cache directory.
        phase2_cache = os.path.join(args.out, "cupy_cache_phase2-ftz_stripped")
        os.makedirs(phase2_cache, exist_ok=True)
        os.environ["CUPY_CACHE_DIR"] = phase2_cache
        artifact["phase2_cupy_cache_dir"] = phase2_cache
        say("phase 2/2: CuPy array path (kill switch on), memo dropped, cache cold")
        artifact["cupy"] = run_cupy_phase(args.sim_case, args.res, args.gpu,
                                          None, args.out)
        from meep_gpu import subnormal_policy as sp  # noqa: PLC0415

        artifact["stamp"] = sp.policy_stamp()
        cupy_whole = artifact["cupy"]["whole_step"]
        triton_whole = artifact["triton"]["whole_step"]
        artifact["verdict"] = {
            "cupy_audited": cupy_whole["audited"], "cupy_with_ftz": cupy_whole["with_ftz"],
            "triton_audited": triton_whole["audited"],
            "triton_with_ftz": triton_whole["with_ftz"],
        }
        say(f"VERDICT whole_step: cupy {cupy_whole['with_ftz']}/{cupy_whole['audited']} ftz, "
            f"triton {triton_whole['with_ftz']}/{triton_whole['audited']} ftz")
        ok = expect(artifact, "both_executors_measured",
                    cupy_whole["audited"] > 0 and triton_whole["audited"] > 0,
                    artifact["verdict"])
        ok &= expect(artifact, "cupy_keeps_subnormals", cupy_whole["with_ftz"] == 0,
                     f"{cupy_whole['with_ftz']} of {cupy_whole['audited']} carry .ftz")
        ok &= expect(artifact, "triton_keeps_subnormals", triton_whole["with_ftz"] == 0,
                     f"{triton_whole['with_ftz']} of {triton_whole['audited']} carry .ftz")
        ok &= expect(artifact, "the_two_executors_agree",
                     (cupy_whole["with_ftz"] == 0) == (triton_whole["with_ftz"] == 0))

    artifact["all_checks_passed"] = bool(ok)
    artifact["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(artifact, artifact_path)
    say(f"DONE case={args.case} all_checks_passed={ok} artifact={artifact_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
