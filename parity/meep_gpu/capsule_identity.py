"""The capsule's own test: a fresh lift equals ``load(capsule)``, word for word, at every step.

Three subcommands, each its own process:

``lift``     Where MEEP is installed. Lifts one case of a cases file (``build(mp, label)``
             or ``BUILDERS[label]``) on the NumPy reference under
             ``capsule.recording_construction``, writes its capsule (with the MEEP
             simulation, so the chunk division is recorded), then steps the lifted
             driver to N while recording every source current. At every scheduled step
             it records the digest of every array the capsule pins and of
             ``collect_state``; at step 0 and at N it saves the pinned arrays (a superset
             of ``collect_state``'s) for a word compare.
``load``     In a process where ``meep`` cannot be imported (an import blocker is
             installed before anything else). Loads the capsule, optionally with the
             recorded currents injected, and does the same at the same steps.
             ``--perturb-current`` scales one recorded current of source 0 (the
             earliest within ``2**-10`` of its largest) by ``1 + 2**-20``: the
             discriminating control, which must differ.
``compare``  Imports neither the package nor MEEP. Checks the two step schedules are the
             same, compares the digests step by step, and compares the saved arrays word
             for word (uint32 views, so the sign of zero counts) with a planted one-bit
             flip per saved step that the comparer must report.

N defaults to the cases file's ``full_timing(mp, sim)["steps_planned"]`` (Table 2's
step rule) and the schedule is every step to 20 plus the file's ``checkpoints(N)``.
Each leg prints one flushed line per scheduled step, appends one JSONL row per step to
``hashes.jsonl``, and writes ``manifest.json`` and ``DONE`` last.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import struct
import sys
import time
from typing import Any, Dict, List, Optional

import numpy

FORMAT = "capsule_identity/1"
PERTURBATION = 2.0 ** -20


class _Progress:
    def __init__(self, path: str, tag: str):
        self.path, self.tag, self.t0 = path, tag, time.time()

    def say(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} +{time.time() - self.t0:8.2f}s [{self.tag}] {text}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class _MeepBlocker:
    """A meta-path finder that makes ``import meep`` fail in this process."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == "meep" or name.startswith("meep."):
            raise ImportError("MEEP is blocked in a capsule load leg")
        return None


def _words(array: numpy.ndarray) -> numpy.ndarray:
    raw = numpy.ascontiguousarray(array).reshape(-1).view(numpy.uint8)
    return raw.view(numpy.uint32) if raw.size % 4 == 0 else raw


def _digest(arrays: Dict[str, numpy.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = numpy.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_cases(path: str) -> Any:
    directory = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, directory)
    try:
        spec = importlib.util.spec_from_file_location("capsule_identity_cases", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(directory)


def schedule_for(n_steps: int, checkpoints: Optional[List[int]] = None, every: int = 20) -> List[int]:
    """Every step 0..min(every, N), plus the given checkpoints, plus N."""
    steps = set(range(0, min(every, n_steps) + 1))
    steps.update(int(step) for step in (checkpoints or ()) if 0 <= int(step) <= n_steps)
    steps.add(n_steps)
    return sorted(steps)


def _query_time(key: str) -> float:  # The first argument of a recorded query: its time.
    return struct.unpack("<d", bytes.fromhex(key.split("|")[0]))[0]


def perturb_earliest_current(table: Any) -> Dict[str, Any]:
    """Scale one recorded value of source 0 by ``1 + 2**-20``: the control that must differ.

    The value chosen is the EARLIEST query whose magnitude is at least ``2**-10`` of the
    source's largest, so the change reaches the fields early and is carried through the
    rest of the run; the largest value alone can sit at the last step (a Gaussian whose
    peak is the run's end).
    """
    entry = table.sources[0]
    name = "current" if entry["calls"]["current"] else "dipole"
    calls = entry["calls"][name]
    largest = max(abs(value) for value in calls.values())
    key = min((key for key, value in calls.items() if abs(value) >= largest * 2.0 ** -10),
              key=_query_time)
    calls[key] = calls[key] * (1.0 + PERTURBATION)
    return {"source": 0, "function": name, "time": _query_time(key), "scale": "1 + 2**-20",
            "relative_to_largest": abs(calls[key]) / largest}


def _step_rows(driver: Any, schedule: List[int], out: str, progress: _Progress,
               collect_state: Any, capsule: Any) -> List[Dict[str, Any]]:
    rows = []
    last = schedule[-1]
    for target in schedule:
        started = time.time()
        while driver.step_count < target:
            driver.step()
        state = capsule.state_arrays(driver)
        collected = collect_state(driver)
        row = {"step": target, "state_sha256": _digest(state), "state_arrays": len(state),
               "state_words": int(sum(_words(array).size for array in state.values())),
               "collect_sha256": _digest(collected), "collect_arrays": len(collected),
               "seconds": round(time.time() - started, 4)}
        rows.append(row)
        with open(os.path.join(out, "hashes.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if target in (0, last):
            # The pinned arrays only: collect_state's arrays are among them, and its own
            # digest is compared at every scheduled step.
            numpy.savez_compressed(os.path.join(out, f"state_{target:05d}.npz"),
                                   **{f"state:{name}": array for name, array in state.items()})
        progress.say(f"step {target} ({schedule.index(target) + 1}/{len(schedule)}) state "
                     f"{row['state_sha256'][:16]} arrays {row['state_arrays']} words "
                     f"{row['state_words']} collect {row['collect_sha256'][:16]} ({row['seconds']:.2f} s)")
    return rows


def _environment() -> Dict[str, Any]:
    return {"python": sys.executable, "python_version": platform.python_version(),
            "numpy": numpy.__version__, "machine": platform.machine(), "system": platform.system(),
            "niceness": os.nice(0),
            "environ": {key: os.environ.get(key) for key in (
                "PYTHONPATH", "MEEP_GPU_DISPATCH", "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX",
                "OMP_NUM_THREADS")}}


def _start(out: str, tag: str) -> _Progress:
    os.makedirs(out, exist_ok=True)
    for marker in ("DONE", "manifest.json", "hashes.jsonl"):
        if os.path.exists(os.path.join(out, marker)):
            raise SystemExit(f"REFUSING: {out} already holds {marker}")
    return _Progress(os.path.join(out, "progress.log"), tag)


def _finish(out: str, manifest: Dict[str, Any], rc: int) -> int:
    manifest["rc"] = rc
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    with open(os.path.join(out, "DONE"), "w", encoding="utf-8") as handle:
        handle.write(f"rc={rc}\n")
    return rc


def lift(args: argparse.Namespace) -> int:
    out = os.path.abspath(args.out)
    progress = _start(out, f"lift/{args.case}")
    manifest: Dict[str, Any] = {"format": FORMAT, "leg": "lift", "case": args.case,
                                "cases_file": os.path.abspath(args.cases_file),
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **_environment()}
    rc = 1
    try:
        import meep as mp  # noqa: PLC0415

        mp.verbosity(0)
        import meep_gpu  # noqa: PLC0415
        from meep_gpu import capsule  # noqa: PLC0415

        from parity.meep_gpu.gate_dispatch_end_to_end import collect_state  # noqa: PLC0415

        cases = _load_cases(args.cases_file)
        builder = getattr(cases, "build", None)
        sim = builder(mp, args.case) if builder is not None else cases.BUILDERS[args.case](mp)
        with capsule.recording_construction() as recorder:
            driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
        if driver.gpu is not None:
            raise RuntimeError(f"REFUSING: the lift resolved driver.gpu = {driver.gpu!r}")
        if args.steps is not None:
            n_steps = int(args.steps)
            manifest["steps_rule"] = "--steps"
        else:
            n_steps = int(cases.full_timing(mp, sim)["steps_planned"])
            manifest["steps_rule"] = "cases.full_timing(mp, sim)['steps_planned']"
        checkpoints = cases.checkpoints(n_steps) if hasattr(cases, "checkpoints") else []
        schedule = schedule_for(n_steps, checkpoints)
        info = capsule.write_capsule(os.path.join(out, "capsule.capsule"), driver, recorder, sim=sim)
        manifest.update({
            "meep": {"version": mp.__version__, "file": os.path.realpath(mp.__file__),
                     "single_precision": bool(mp.is_single_precision())},
            "meep_gpu_file": os.path.realpath(meep_gpu.__file__),
            "capsule": {"file_sha256": info.file_sha256, "document_sha256": info.document_sha256,
                        "arrays": info.arrays, "array_bytes": info.array_bytes,
                        "file_bytes": os.path.getsize(info.path),
                        "construction_calls": [name for name, _ in recorder.log_for(driver).calls]},
            "n_steps": n_steps, "schedule": schedule,
        })
        progress.say(f"lifted {args.case}: stored {list(driver.grid.shape)}, capsule "
                     f"{os.path.getsize(info.path)} bytes, {info.arrays} arrays, calls "
                     f"{manifest['capsule']['construction_calls']}; N = {n_steps}, {len(schedule)} steps scheduled")
        with capsule.recording_currents(driver) as currents:
            manifest["rows"] = _step_rows(driver, schedule, out, progress, collect_state, capsule)
        table = currents.table()
        manifest["currents_sha256"] = table.write(os.path.join(out, "currents.json"))
        manifest["currents_recorded"] = [{name: len(calls) for name, calls in entry["calls"].items()}
                                         for entry in table.sources]
        driver.close()
        rc = 0
        progress.say("lift leg done")
    except BaseException as exc:  # noqa: BLE001 - the leg's record names the failure
        manifest["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        progress.say(f"ERROR {manifest['error']}")
    return _finish(out, manifest, rc)


def load(args: argparse.Namespace) -> int:
    if "meep" in sys.modules:
        raise SystemExit("REFUSING: meep is already imported in this process; run the load leg in its own")
    blocker = _MeepBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        return _load(args)
    finally:
        sys.meta_path.remove(blocker)


def _load(args: argparse.Namespace) -> int:
    out = os.path.abspath(args.out)
    progress = _start(out, f"load/{os.path.basename(os.path.dirname(os.path.abspath(args.capsule)))}")
    manifest: Dict[str, Any] = {"format": FORMAT, "leg": "load", "capsule_path": os.path.abspath(args.capsule),
                                "currents": os.path.abspath(args.currents) if args.currents else None,
                                "perturb_current": bool(args.perturb_current),
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **_environment()}
    rc = 1
    try:
        with open(args.schedule_from, encoding="utf-8") as handle:
            schedule = [int(step) for step in json.load(handle)["schedule"]]
        import meep_gpu  # noqa: PLC0415
        from meep_gpu import capsule  # noqa: PLC0415

        from parity.meep_gpu.gate_dispatch_end_to_end import collect_state  # noqa: PLC0415

        stored = capsule.read_capsule(args.capsule)
        table = None
        if args.currents:
            table = capsule.CurrentsTable.read(args.currents)
            if args.perturb_current:
                manifest["perturbed"] = perturb_earliest_current(table)
        elif args.perturb_current:
            raise RuntimeError("--perturb-current needs --currents")
        driver = stored.build_driver(prefer_gpu=False, currents=table)
        manifest.update({"meep_gpu_file": os.path.realpath(meep_gpu.__file__),
                         "capsule": {"file_sha256": stored.file_sha256,
                                     "document_sha256": stored.document_sha256},
                         "load_report": stored.load_report, "schedule": schedule})
        progress.say(f"loaded {args.capsule}: report {stored.load_report}")
        manifest["rows"] = _step_rows(driver, schedule, out, progress, collect_state, capsule)
        manifest["meep_imported"] = "meep" in sys.modules
        if manifest["meep_imported"]:
            raise RuntimeError("REFUSING: meep is in sys.modules after a load leg")
        driver.close()
        rc = 0
        progress.say("load leg done; meep never imported")
    except BaseException as exc:  # noqa: BLE001 - the leg's record names the failure
        manifest["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        progress.say(f"ERROR {manifest['error']}")
    return _finish(out, manifest, rc)


def _compare_dump(reference: str, candidate: str) -> Dict[str, Any]:
    with numpy.load(reference) as left, numpy.load(candidate) as right:
        names_left, names_right = sorted(left.files), sorted(right.files)
        result: Dict[str, Any] = {"keys_equal": names_left == names_right,
                                  "only_reference": sorted(set(names_left) - set(names_right)),
                                  "only_candidate": sorted(set(names_right) - set(names_left)),
                                  "arrays": 0, "words": 0, "differing_words": 0,
                                  "differing_arrays": {}, "meta_mismatch": {}}
        candidates = {}
        for name in sorted(set(names_left) & set(names_right)):
            a, b = left[name], right[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                result["meta_mismatch"][name] = [str(a.shape), a.dtype.str, str(b.shape), b.dtype.str]
                continue
            wa, wb = _words(a), _words(b)
            differing = int(numpy.count_nonzero(wa != wb))
            result["arrays"] += 1
            result["words"] += int(wa.size)
            result["differing_words"] += differing
            if differing:
                result["differing_arrays"][name] = differing
            agreeing = numpy.flatnonzero((wb != 0) & (wa == wb))
            if agreeing.size:
                candidates[name] = (wa, wb, agreeing)
        if candidates:
            name = max(candidates, key=lambda key: candidates[key][1].size)
            wa, wb, agreeing = candidates[name]
            planted = numpy.array(wb, copy=True)
            index = int(agreeing[agreeing.size // 2])
            planted[index] ^= planted.dtype.type(1)
            before = int(numpy.count_nonzero(wa != wb))
            after = int(numpy.count_nonzero(wa != planted))
            result["planted"] = {"array": name, "flat_word": index, "before": before, "after": after,
                                 "caught": after == before + 1}
        else:
            result["planted"] = {"array": None, "caught": False}
    return result


def compare(args: argparse.Namespace) -> int:
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    progress = _Progress(out + ".progress.log", f"compare/{os.path.basename(out)}")
    manifests = {}
    for side, directory in (("reference", args.reference), ("candidate", args.candidate)):
        with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as handle:
            manifests[side] = json.load(handle)
        if manifests[side].get("rc") != 0:
            raise SystemExit(f"{directory} did not finish rc 0: {manifests[side].get('error')}")
    reference_rows = {row["step"]: row for row in manifests["reference"]["rows"]}
    candidate_rows = {row["step"]: row for row in manifests["candidate"]["rows"]}
    report: Dict[str, Any] = {"format": FORMAT, "reference": os.path.abspath(args.reference),
                              "candidate": os.path.abspath(args.candidate),
                              "schedule_equal": sorted(reference_rows) == sorted(candidate_rows),
                              "steps": [], "dumps": {}}
    first_state_difference = None
    for step in sorted(reference_rows):
        left, right = reference_rows[step], candidate_rows.get(step)
        row = {"step": step,
               "state_equal": right is not None and left["state_sha256"] == right["state_sha256"],
               "collect_equal": right is not None and left["collect_sha256"] == right["collect_sha256"]}
        if not row["state_equal"] and first_state_difference is None:
            first_state_difference = step
        report["steps"].append(row)
    report["steps_compared"] = len(report["steps"])
    report["steps_state_equal"] = sum(row["state_equal"] for row in report["steps"])
    report["steps_collect_equal"] = sum(row["collect_equal"] for row in report["steps"])
    report["first_state_difference"] = first_state_difference
    progress.say(f"digests: state equal at {report['steps_state_equal']} of {report['steps_compared']} "
                 f"scheduled steps, collect_state equal at {report['steps_collect_equal']} of "
                 f"{report['steps_compared']}; first difference {first_state_difference}")
    for step in (0, max(reference_rows)):
        name = f"state_{step:05d}.npz"
        dump = _compare_dump(os.path.join(args.reference, name), os.path.join(args.candidate, name))
        report["dumps"][str(step)] = dump
        progress.say(f"words at step {step}: {dump['differing_words']} of {dump['words']} differ over "
                     f"{dump['arrays']} pinned arrays; keys equal {dump['keys_equal']}; planted flip "
                     f"caught {dump['planted']['caught']}")
    dumps = report["dumps"].values()
    report["identical"] = bool(
        report["schedule_equal"]
        and report["steps_state_equal"] == report["steps_compared"]
        and report["steps_collect_equal"] == report["steps_compared"]
        and all(dump["keys_equal"] and not dump["meta_mismatch"] and dump["differing_words"] == 0
                for dump in dumps))
    report["planted_caught"] = all(dump["planted"]["caught"] for dump in dumps)
    report["expect"] = args.expect
    passed = report["planted_caught"] and (report["identical"] if args.expect == "identical"
                                           else not report["identical"])
    report["passed"] = passed
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    progress.say(f"identical {report['identical']} (expected {args.expect}); planted caught "
                 f"{report['planted_caught']}; PASSED {passed}")
    return 0 if passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    leg = sub.add_parser("lift")
    leg.add_argument("--cases-file", required=True)
    leg.add_argument("--case", required=True)
    leg.add_argument("--steps", type=int, default=None)
    leg.add_argument("--out", required=True)
    loading = sub.add_parser("load")
    loading.add_argument("--capsule", required=True)
    loading.add_argument("--schedule-from", required=True, help="the lift leg's manifest.json")
    loading.add_argument("--currents", default=None)
    loading.add_argument("--perturb-current", action="store_true")
    loading.add_argument("--out", required=True)
    comparing = sub.add_parser("compare")
    comparing.add_argument("--reference", required=True)
    comparing.add_argument("--candidate", required=True)
    comparing.add_argument("--expect", choices=("identical", "differ"), default="identical")
    comparing.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"lift": lift, "load": load, "compare": compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
