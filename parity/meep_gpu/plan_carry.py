"""Which route legs and L3 rows still hold on a new tree, and the proof that they do.

What a run is evidence of. A route leg (one process of a route gate) and an L3 row (one
process of ``l3_row.py``) certify the plans they dispatched, executed by the bytes their
import record names. So a run holds on a new tree when three things hold there:

1. its CLOSURE: every file of its import record (``imported_source_sha256``) at the
   recorded bytes, the rule ``artifact_drift.py`` applies to every gate artifact;
2. its PLANNER'S DATA, or its plans: the data the planner read and the import record
   does not name (``plan_key.plan_inputs``: the timing records and the views of the
   ledgers) is byte-equal, or moved only where its device does not read it
   (``ledger_scope.py``), or, where only the timing records moved, the plans the run
   recorded (``plan_key`` digests) are the plans a plan pass on the new tree makes;
3. its DEVICE: the device class it stamped is the one being certified.

Why the timing records alone are offered a plan proof. The timing records exist to change
WHO holds a slot and nothing else (``meep_gpu/dispatch_preference.py``: arbitration only),
and the plan key records exactly who held each slot and how often it launched, so a
re-cut record that leaves a run's plans unchanged leaves its evidence unchanged. A route
leg's plans read them only through its default-preference control (every other leg runs
the shape rule, which opens no record; that is ``fastpath._apply_preference``, in the
closure), and an L3 row through its whole plan. The ledgers also decide admission, but a
ledger re-bind can change any plan of any leg on a device that reads it, and the plan
passes ask only the default's plan, so a moved view of such a ledger re-runs the run: no
plan proof is offered for it. A run on an NVIDIA device reads the Triton and hand-CUDA
ledgers and one on an Apple GPU the Metal ledger, whatever the run recorded beside them, so
a view of the other platform's ledger is not compared at all.

What "moved" means. Within a ledger its device reads, a run reads a slice, and a timing
record of another architecture or vendor is read only as ``ledger_scope.py`` says. A move
outside that is SET ASIDE: the proof names it (``set_aside``), with the digest it recorded,
why, and the slice's digest, and it voids nothing. A digest that cannot be set aside is
compared whole, as it always was, and a verdict names why (``compared_whole``). With no
``repo`` given the comparison is on digests alone.

The states. A run is CARRIED when it holds:

* ``CARRY-DIGEST``: closure, planner's data and device all byte-equal;
* ``CARRY-SLICE``: closure and device equal, and every planner's input that moved set aside:
  a whole-file digest moved and nothing this device reads did;
* ``CARRY-SHAPE-RULE``: closure and the device's ledgers equal, timing records moved, and
  no plan of the run read them (a route leg without a recorded default-preference plan);
* ``CARRY-PLAN``: closure and the device's ledgers equal, timing records moved, and a plan
  pass on the new tree, itself on the new tree's bytes and records and on the same device, of the
  same leg at the same resolution, in the same environment and recording the default on
  every case it is asked about, recorded the same plan digest for every case or row the
  run recorded one for.

The device is read and never assumed: a cut that names no device carries nothing. An L3
row carries only when it held against its primary set (``holds_dispatched`` or
``holds_array``); a row that differed, even as expected, re-runs.

Otherwise it RE-RUNS, by name: ``RERUN-CLOSURE`` (the moved files), ``RERUN-INPUTS`` (the
moved views of ledgers its device reads), ``RERUN-DEVICE``, ``RERUN-PLAN`` (the cases whose
plan changed),
``RERUN-UNRELEASED`` (the run did not release, or the row did not hold),
``RERUN-UNKEYED`` (the run recorded no plan key or no plan inputs: it ran before plan
keying), ``OWED-PLAN`` (timing records moved and no admissible plan pass was given: run
one, or re-run), ``MISSING`` (a required leg or a planned row is absent, or held only by a
leg of other facts), ``DUPLICATE`` (one leg or row found in two sources: neither is taken)
and ``RERUN-WITH-CAMPAIGN`` (a Metal leg that would carry alone, in a campaign that
re-runs: the Metal campaign runs as one item and its band witness feeds every other leg).

Standard library, ``artifact_drift``, ``ledger_scope`` and ``plan_key``; imported by no gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import artifact_drift
import ledger_scope
import plan_key

SCHEMA = "meep_gpu.plan_carry/1"

CARRY_DIGEST = "CARRY-DIGEST"
CARRY_SLICE = "CARRY-SLICE"
CARRY_SHAPE_RULE = "CARRY-SHAPE-RULE"
CARRY_PLAN = "CARRY-PLAN"
RERUN_CLOSURE = "RERUN-CLOSURE"
RERUN_INPUTS = "RERUN-INPUTS"
RERUN_DEVICE = "RERUN-DEVICE"
RERUN_PLAN = "RERUN-PLAN"
RERUN_UNRELEASED = "RERUN-UNRELEASED"
RERUN_UNKEYED = "RERUN-UNKEYED"
OWED_PLAN = "OWED-PLAN"
MISSING = "MISSING"
DUPLICATE = "DUPLICATE"
RERUN_WITH_CAMPAIGN = "RERUN-WITH-CAMPAIGN"
CARRIED = (CARRY_DIGEST, CARRY_SLICE, CARRY_SHAPE_RULE, CARRY_PLAN)

#: What a proof says of the ledgers: only those a run on its device reads are compared.
DEVICE_LEDGERS_EQUAL = "equal over what a run on this device reads of the ledgers"
#: The control whose one-step plan is a route leg's only plan that reads timing records.
DEFAULT_CONTROL = "default-preference"
#: That control's passing verdict (``gate_dispatch_fused_route.DEFAULT_PREFERENCE_RECORDED``,
#: spelled here so the rules import no gate; a test holds the two equal).
DEFAULT_RECORDED = "DEFAULT-PREFERENCE-RECORDED"
#: The L3 verdicts against the primary set that a carried row may hold.
L3_HOLDING = ("holds_dispatched", "holds_array")
#: Recorded L3 environment names a plan row is not compared on: the repository and MEEP
#: paths (the import record and the MEEP images are compared instead), the card's index
#: (the device class is compared instead) and the per-row dispatch log.
L3_ENV_NOT_COMPARED = ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "MEEP_GPU_DISPATCH_LOG")
#: Recorded L3 environment names that hold a per-leg directory: compared by the last
#: component, which names the cache's policy (``cupy_ftz_stripped``, ``cupy_plain``, ...).
L3_ENV_BY_NAME_ONLY = ("CUPY_CACHE_DIR", "TRITON_CACHE_DIR")
#: The facts that name which L3 leg a row belongs to, as the row records them.
L3_LEG_FACTS = ("table", "mode", "shipped_default", "sets", "primary", "image_set")
#: The environment facts a plan pass must share with the run it is compared against.
ENVIRONMENT_FACTS = ("python", "meep", "cupy", "triton", "numpy", "torch", "machine",
                     "metal_frontend")
#: The facts that name WHICH leg a route process ran (its arguments as the gate records them
#: in ``provenance``): a plan pass stands for a leg only when they are the leg's.
LEG_FACTS = ("backend", "install_policy_requested", "harness_installs_nothing",
             "policy_leg_role", "admitted_policy_installed", "uncertified_policy_installed",
             "hardware_scope", "leg_table_preference", "band_witness_leg", "resolution")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_files(directory: Path) -> Dict[str, str]:
    """Every regular file under ``directory`` (``__pycache__`` aside), keyed relative."""
    directory = Path(directory)
    out: Dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            out[path.relative_to(directory).as_posix()] = sha256_file(path)
    return out


def device_of(payload: Mapping[str, Any]) -> Optional[str]:
    """``cc86`` from a stamped compute capability, or the Apple GPU architecture."""
    stamped = payload.get("stamped_device")
    if isinstance(stamped, Mapping) and stamped.get("compute_capability"):
        return "cc" + str(stamped["compute_capability"]).replace(".", "")
    provenance = payload.get("provenance")
    for holder in (payload, provenance if isinstance(provenance, Mapping) else {}):
        apple = holder.get("apple_gpu") if isinstance(holder, Mapping) else None
        if isinstance(apple, Mapping) and apple.get("architecture"):
            return str(apple["architecture"])
    return None


def leg_of(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The leg-naming facts of a route process (:data:`LEG_FACTS`, and whether Triton was
    withheld from it)."""
    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    out = {name: provenance.get(name) for name in LEG_FACTS if name in provenance}
    out["triton_withheld"] = provenance.get("triton_withheld") is not None
    return out


def environment_of(payload: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    return {name: provenance.get(name) for name in ENVIRONMENT_FACTS
            if provenance.get(name) is not None}


def platforms_recorded(payload: Mapping[str, Any]) -> List[str]:
    """The platforms a run says it planned or launched on: an L3 row's ``table`` and its
    plan key's ``platform``, a route leg's gate and ``provenance.backend``."""
    provenance = payload.get("provenance")
    backend = provenance.get("backend") if isinstance(provenance, Mapping) else None
    key_block = payload.get(plan_key.KEY_FIELD)
    described = key_block.get("plan") if isinstance(key_block, Mapping) else None
    plan = described.get("selected_plan") if isinstance(described, Mapping) else None
    found = {ledger_scope.platform_of_table(payload.get("table")),
             ledger_scope.platform_of_table(backend),
             plan.get("platform") if isinstance(plan, Mapping) else None,
             "metal" if payload.get("gate") == "dispatch_metal_route" else None}
    return sorted(platform for platform in found if platform)


def triton_version_of(payload: Mapping[str, Any]) -> Optional[str]:
    """The Triton a run recorded: an L3 row's ``plan_toolchain``, a route leg's
    ``provenance``. None when it recorded neither, or two that disagree, so the slice keeps
    the whole list."""
    stated = {str(block["triton"])
              for block in (payload.get(plan_key.TOOLCHAIN_FIELD), payload.get("provenance"))
              if isinstance(block, Mapping) and block.get("triton") is not None}
    return stated.pop() if len(stated) == 1 else None


def compare_inputs(recorded: Any, live: Mapping[str, Any], device: Optional[str],
                   platforms: Iterable[str] = (), *, repo: Optional[Path] = None,
                   payload: Optional[Mapping[str, Any]] = None,
                   cache: Optional[Dict[str, Any]] = None
                   ) -> Tuple[Dict[str, List[str]], Dict[str, Any], Dict[str, Any]]:
    """``(moved, set_aside, compared_whole)``. ``moved`` is ``plan_key.inputs_moved`` over the
    ledgers a run on ``device`` reads, and those of any platform the run records
    (``ledger_scope.reads_ledger``, :func:`platforms_recorded`). A recorded statement holding
    no ledger map moves them all, as it always did: it says nothing about any of them. Given
    ``repo`` and ``payload``, what remains goes through ``ledger_scope.ledger_moved`` and
    ``ledger_scope.timing_record_moved``: ``set_aside`` is what they excuse and
    ``compared_whole`` is why the rest stands. ``plan_key`` read the ledger on ``live``, so
    where it holds no digest, or the tool's own read of the file agrees with the run and
    ``plan_key``'s did not, nothing is set aside."""
    platforms = tuple(platforms)
    moved = dict(plan_key.inputs_moved(recorded, live))
    statement = recorded.get("ledgers") if isinstance(recorded, Mapping) else None
    if "ledgers" in moved and isinstance(statement, Mapping):
        read = [name for name in moved.pop("ledgers")
                if ledger_scope.reads_ledger(name, device, platforms)]
        if read:
            moved["ledgers"] = read
    set_aside: Dict[str, Any] = {}
    compared_whole: Dict[str, Any] = {}
    if repo is None or payload is None or device is None:
        return moved, set_aside, compared_whole
    imports = payload.get("imported_source_sha256")
    imports = imports if isinstance(imports, Mapping) else {}
    if "ledgers" in moved and isinstance(statement, Mapping):
        kept = []
        views = live.get("ledgers")
        views = views if isinstance(views, Mapping) else {}
        for name in moved.pop("ledgers"):
            verdict = ledger_scope.ledger_moved(
                name, statement.get(name), form="view", device=device, imports=imports,
                triton_version=triton_version_of(payload), root=repo, cache=cache)
            if verdict.why == "digest_equal":
                verdict = ledger_scope.Verdict(True, "recorder_read_otherwise",
                                               {"live": views.get(name)})
            elif not verdict.moved and not isinstance(views.get(name), str):
                verdict = ledger_scope.Verdict(True, "live_ledger_unreadable",
                                               {"file": name, "live": views.get(name)})
            if verdict.moved:
                kept.append(name)
                compared_whole[name] = {"why": verdict.why, "recorded": statement.get(name),
                                        **verdict.proof}
            else:
                set_aside[name] = {"recorded": statement.get(name), "why": verdict.why,
                                   "slice_sha256": verdict.proof.get("slice_sha256")}
        if kept:
            moved["ledgers"] = kept
    records = recorded.get("timing_records") if isinstance(recorded, Mapping) else None
    if "timing_records" in moved and isinstance(records, Mapping):
        kept = []
        for name in moved.pop("timing_records"):
            verdict = ledger_scope.timing_record_moved(
                name, device, records, live.get("timing_records"), repo, platforms, cache)
            if verdict.moved:
                kept.append(name)
                compared_whole[name] = {"why": verdict.why, **verdict.proof}
            else:
                set_aside[name] = {"recorded": records.get(name), "why": verdict.why,
                                   "slice_sha256": None}
        if kept:
            moved["timing_records"] = kept
    return moved, set_aside, compared_whole


def inputs_moved(recorded: Any, live: Mapping[str, Any], device: Optional[str],
                 platforms: Iterable[str] = (), *, repo: Optional[Path] = None,
                 payload: Optional[Mapping[str, Any]] = None,
                 cache: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """What moved for a run on ``device``: the first answer of :func:`compare_inputs`."""
    return compare_inputs(recorded, live, device, platforms, repo=repo, payload=payload,
                          cache=cache)[0]


_NOTHING_SET_ASIDE: Mapping[str, Any] = {"set_aside": {}, "compared_whole": {}}


def _with_set_aside(proof: Dict[str, Any], scope: Mapping[str, Any]) -> Dict[str, Any]:
    if scope["set_aside"]:
        proof["set_aside"] = dict(scope["set_aside"])
    return proof


def _common(payload: Mapping[str, Any], repo: Path, live: Mapping[str, Any],
            device: Optional[str], cache: Dict[str, Any]
            ) -> Tuple[Optional[Dict[str, Any]], Dict[str, List[str]], Dict[str, Any]]:
    """The rules every run shares. ``(re-run verdict or None, moved planner inputs, the
    comparison's ``set_aside`` and ``compared_whole``)``."""
    keyed = (isinstance(payload.get(plan_key.INPUTS_FIELD), Mapping)
             and isinstance(payload.get("imported_source_sha256"), Mapping)
             and payload.get("imported_source_sha256"))
    if not keyed:
        return ({"state": RERUN_UNKEYED,
                 "why": ("the run records no plan_inputs or no import record: it ran before "
                         "plan keying, so neither its closure nor its planner's data can be "
                         "compared")}, {}, _NOTHING_SET_ASIDE)
    moved = artifact_drift.artifact_moved(payload, repo, cache)
    if moved:
        return ({"state": RERUN_CLOSURE, "moved": moved,
                 "why": artifact_drift.refusal("the run", moved, repo)}, {}, _NOTHING_SET_ASIDE)
    seen = device_of(payload)
    if device is None:
        return ({"state": RERUN_DEVICE,
                 "why": ("no device was named for this cut: the device is read and never "
                         "assumed, so nothing carries without one")}, {}, _NOTHING_SET_ASIDE)
    if seen != device:
        return ({"state": RERUN_DEVICE,
                 "why": f"the run stamped device {seen!r}, not {device!r}"}, {},
                _NOTHING_SET_ASIDE)
    inputs, set_aside, compared_whole = compare_inputs(
        payload.get(plan_key.INPUTS_FIELD), live, device, platforms_recorded(payload),
        repo=repo, payload=payload, cache=cache)
    if inputs.get("ledgers"):
        reasons = sorted({str(compared_whole[name]["why"]) for name in inputs["ledgers"]
                          if name in compared_whole})
        return ({"state": RERUN_INPUTS, "moved": inputs["ledgers"],
                 "compared_whole": compared_whole,
                 "why": ("views of the ledgers this device reads moved (entries other than "
                         f"driver_dispatch): {inputs['ledgers']}; a ledger can change the "
                         "admission of any plan on a device that reads it, and no plan "
                         "proof is offered for it"
                         + (f" (compared whole: {reasons})" if reasons else ""))},
                inputs, _NOTHING_SET_ASIDE)
    return None, inputs, {"set_aside": set_aside, "compared_whole": compared_whole}


# ---------------------------------------------------------------------------------------
# Plan passes
# ---------------------------------------------------------------------------------------

def plan_pass_problems(summary: Mapping[str, Any], repo: Path, live: Mapping[str, Any],
                       device: Optional[str], run: Mapping[str, Any],
                       cache: Dict[str, Any], cases: Iterable[str] = ()) -> List[str]:
    """Why a plan pass cannot stand for the tree in hand, compared with ``run``; [] if it can.

    The pass must be a plan pass, on bytes the tree holds, on the tree's planner data
    exactly, on the device being certified, of the same leg (:data:`LEG_FACTS`) and in the
    run's environment; and on every one of ``cases`` (the cases the run recorded a default
    plan for) that the pass ran, its row must have recorded the default: a row that did not
    (the preference ran in another mode, a record read wrongly) states no default plan.
    """
    problems: List[str] = []
    if summary.get("mode") != "plan-only" and not summary.get("plan_only"):
        problems.append("not a plan pass (mode is not plan-only)")
    moved = artifact_drift.artifact_moved(summary, repo, cache)
    if moved or not summary.get("imported_source_sha256"):
        problems.append("the plan pass ran on other bytes: "
                        + (artifact_drift.describe(moved, repo) if moved else "no import record"))
    inputs = inputs_moved(summary.get(plan_key.INPUTS_FIELD), live, device,
                          platforms_recorded(summary), repo=repo, payload=summary, cache=cache)
    if inputs:
        problems.append(f"the plan pass read other planner data than the tree's: {inputs}")
    if device is None or device_of(summary) != device:
        problems.append(f"the plan pass stamped device {device_of(summary)!r}, not {device!r}")
    failed = set(str(case) for case in summary.get("failures") or ())
    verdicts = {str(row.get("case")): row.get("verdict") for row in summary.get("rows") or ()
                if isinstance(row, Mapping)}
    unrecorded = sorted(case for case in cases
                        if case in failed or (case in verdicts
                                              and verdicts[case] != DEFAULT_RECORDED))
    if unrecorded:
        problems.append(f"the plan pass did not record the default on {unrecorded}: "
                        f"{ {case: verdicts.get(case) for case in unrecorded} }")
    want, have = leg_of(run), leg_of(summary)
    differ = sorted(name for name in set(want) | set(have) if want.get(name) != have.get(name))
    if differ:
        problems.append(f"the plan pass ran another leg: "
                        f"{ {name: (want.get(name), have.get(name)) for name in differ} }")
    want, have = environment_of(run), environment_of(summary)
    differ = sorted(name for name in set(want) | set(have) if want.get(name) != have.get(name))
    if differ:
        problems.append(f"the plan pass ran in another environment: "
                        f"{ {name: (want.get(name), have.get(name)) for name in differ} }")
    return problems


def read_plan_pass(directory: Path) -> Optional[Dict[str, Any]]:
    path = Path(directory) / "plans.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------
# Route legs
# ---------------------------------------------------------------------------------------

def default_digests(artifact: Mapping[str, Any]) -> Dict[str, str]:
    """``{case: digest}`` of the default-preference plans a leg (or plan pass) recorded."""
    keys = artifact.get("plan_keys")
    out: Dict[str, str] = {}
    if isinstance(keys, Mapping):
        for case, entry in keys.items():
            if isinstance(entry, Mapping) and isinstance(entry.get(DEFAULT_CONTROL), str):
                out[str(case)] = entry[DEFAULT_CONTROL]
    return out


def judge_route_leg(artifact: Mapping[str, Any], repo: Path, live: Mapping[str, Any],
                    device: Optional[str], plan_pass: Optional[Mapping[str, Any]] = None,
                    cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The state of one route leg (its ``gate.json``) on the tree at ``repo``."""
    cache = {} if cache is None else cache
    release = artifact.get("release")
    if not (isinstance(release, Mapping) and release.get("released") is True):
        return {"state": RERUN_UNRELEASED,
                "why": f"the leg did not release: {release!r}"[:600]}
    if not isinstance(artifact.get("plan_keys"), Mapping):
        return {"state": RERUN_UNKEYED,
                "why": "the leg records no plan_keys: it ran before plan keying"}
    verdict, inputs, scope = _common(artifact, repo, live, device, cache)
    if verdict is not None:
        return verdict
    recorded = default_digests(artifact)
    if not inputs:
        return {"state": CARRY_SLICE if scope["set_aside"] else CARRY_DIGEST,
                "proof": _with_set_aside(
                    {"closure": "every file of the import record at its bytes",
                     "plan_inputs": DEVICE_LEDGERS_EQUAL, "device": device_of(artifact),
                     "plans": len(recorded)}, scope)}
    if not recorded:
        return {"state": CARRY_SHAPE_RULE, "moved": inputs.get("timing_records"),
                "proof": _with_set_aside(
                    {"closure": "every file of the import record at its bytes",
                     "ledgers": DEVICE_LEDGERS_EQUAL, "device": device_of(artifact),
                     "why": ("no plan of this leg read a timing record: it recorded no "
                             "default-preference plan, and every other plan ran the "
                             "shape rule")}, scope)}
    if plan_pass is None:
        return {"state": OWED_PLAN, "moved": inputs.get("timing_records"),
                "cases": sorted(recorded), "compared_whole": scope["compared_whole"],
                "why": ("timing records moved and no plan pass was given: run the gate "
                        "with --plan-only under this leg's arguments, or re-run the leg")}
    problems = plan_pass_problems(plan_pass, repo, live, device, artifact, cache,
                                  sorted(recorded))
    if problems:
        return {"state": OWED_PLAN, "moved": inputs.get("timing_records"),
                "cases": sorted(recorded), "compared_whole": scope["compared_whole"],
                "why": "the plan pass cannot stand for this tree: " + "; ".join(problems)}
    found = default_digests(plan_pass)
    changed = sorted(case for case, value in recorded.items() if found.get(case) != value)
    if changed:
        return {"state": RERUN_PLAN, "cases": changed,
                "moved": inputs.get("timing_records"),
                "why": (f"the default's plan changed on {len(changed)} of {len(recorded)} "
                        f"cases: {changed}")}
    return {"state": CARRY_PLAN, "moved": inputs.get("timing_records"),
            "proof": _with_set_aside(
                {"closure": "every file of the import record at its bytes",
                 "ledgers": DEVICE_LEDGERS_EQUAL, "device": device_of(artifact),
                 "plans": {case: recorded[case] for case in sorted(recorded)},
                 "plan_pass_sha256": plan_pass.get("_sha256")}, scope)}


# ---------------------------------------------------------------------------------------
# L3 rows
# ---------------------------------------------------------------------------------------

def l3_leg_differences(row: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    """``{fact: (the row's, the target's)}`` for every :data:`L3_LEG_FACTS` fact that differs
    between an L3 row and the leg it is to be carried into (``sets`` compared as a set)."""
    out: Dict[str, Any] = {}
    for name in L3_LEG_FACTS:
        have, want = row.get(name), target.get(name)
        if name == "sets":
            have = sorted(have) if isinstance(have, list) else have
            want = sorted(want) if isinstance(want, list) else want
        if have != want:
            out[name] = (have, want)
    return out


def l3_environment(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The recorded environment of an L3 row as a plan row is compared on it."""
    env = row.get("env")
    env = env if isinstance(env, Mapping) else {}
    out: Dict[str, Any] = {}
    for name, value in env.items():
        if name in L3_ENV_NOT_COMPARED:
            continue
        if name in L3_ENV_BY_NAME_ONLY and isinstance(value, str):
            value = os.path.basename(value.rstrip("/"))
        out[str(name)] = value
    return out


def l3_plan_row_problems(row: Mapping[str, Any], plan_row: Mapping[str, Any], repo: Path,
                         live: Mapping[str, Any], device: Optional[str],
                         cache: Dict[str, Any]) -> List[str]:
    """Why a plan pass's row cannot stand for ``row`` on the tree in hand; [] if it can.

    It must be a plan-pass row, on bytes the tree holds and the tree's planner data, on the
    device being certified, of the same leg (:data:`L3_LEG_FACTS`), on the same MEEP
    (images and version) and in the same environment (:func:`l3_environment`) and
    toolchain (``plan_key.toolchain``).
    """
    problems: List[str] = []
    if not plan_row.get("plan_only") or plan_row.get("error"):
        problems.append(f"not a plan-pass row: {plan_row.get('error') or 'not plan only'}")
    moved = artifact_drift.artifact_moved(plan_row, repo, cache)
    if moved or not plan_row.get("imported_source_sha256"):
        problems.append("the plan row ran on other bytes: "
                        + (artifact_drift.describe(moved, repo) if moved else "no import record"))
    if inputs_moved(plan_row.get(plan_key.INPUTS_FIELD), live, device,
                    platforms_recorded(plan_row), repo=repo, payload=plan_row, cache=cache):
        problems.append("the plan row read other planner data than the tree's")
    if device is None or device_of(plan_row) != device:
        problems.append(f"the plan row stamped device {device_of(plan_row)!r}, not {device!r}")
    differ = l3_leg_differences(plan_row, row)
    if differ:
        problems.append(f"the plan row is of another leg: {differ}")
    row_meep, plan_meep = row.get("meep") or {}, plan_row.get("meep") or {}
    for name in ("images", "version", "single_precision"):
        if row_meep.get(name) != plan_meep.get(name):
            problems.append(f"the plan row loaded another MEEP ({name}: "
                            f"{plan_meep.get(name)!r}, the row's {row_meep.get(name)!r})")
    want, have = l3_environment(row), l3_environment(plan_row)
    differ = sorted(name for name in set(want) | set(have) if want.get(name) != have.get(name))
    if differ:
        problems.append(f"the plan row ran in another environment: "
                        f"{ {name: (want.get(name), have.get(name)) for name in differ} }")
    want, have = row.get(plan_key.TOOLCHAIN_FIELD), plan_row.get(plan_key.TOOLCHAIN_FIELD)
    if not isinstance(want, Mapping) or want != have:
        problems.append(f"the plan row ran under another toolchain: {have!r}, the row's "
                        f"{want!r}")
    return problems


def judge_l3_row(row: Mapping[str, Any], repo: Path, live: Mapping[str, Any],
                 device: Optional[str], plan_row: Optional[Mapping[str, Any]] = None,
                 cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The state of one L3 row (its line in ``rows.jsonl``) on the tree at ``repo``.

    ``plan_row`` is the same label's row of a plan pass (``l3_corpus.py --plan-only``). A
    row is carried only when it held against its primary set (:data:`L3_HOLDING`): a row
    that differed, errored or only planned is not evidence of a holding row, and re-runs.
    """
    cache = {} if cache is None else cache
    if row.get("error") or row.get("plan_only"):
        return {"state": RERUN_UNRELEASED,
                "why": f"the row is not a judged row: {row.get('error') or 'plan only'}"}
    if row.get("l3_verdict") not in L3_HOLDING:
        return {"state": RERUN_UNRELEASED,
                "why": (f"the row did not hold against its primary set: l3_verdict "
                        f"{row.get('l3_verdict')!r}, not one of {list(L3_HOLDING)}")}
    recorded = plan_key.key_digest(row.get(plan_key.KEY_FIELD))
    if recorded is None:
        return {"state": RERUN_UNKEYED,
                "why": "the row records no plan_key of this form: it ran before plan keying"}
    verdict, inputs, scope = _common(row, repo, live, device, cache)
    if verdict is not None:
        return verdict
    if not inputs:
        return {"state": CARRY_SLICE if scope["set_aside"] else CARRY_DIGEST,
                "proof": _with_set_aside(
                    {"closure": "every file of the import record at its bytes",
                     "plan_inputs": DEVICE_LEDGERS_EQUAL, "device": device_of(row),
                     "plan": recorded}, scope)}
    if plan_row is None:
        return {"state": OWED_PLAN, "moved": inputs.get("timing_records"),
                "compared_whole": scope["compared_whole"],
                "why": ("timing records moved and the plan pass holds no row for this "
                        "label: run l3_corpus.py --plan-only on the new tree, or re-run it")}
    problems = l3_plan_row_problems(row, plan_row, repo, live, device, cache)
    if problems:
        return {"state": OWED_PLAN, "moved": inputs.get("timing_records"),
                "compared_whole": scope["compared_whole"],
                "why": "the plan row cannot stand for this tree: " + "; ".join(problems)}
    found = plan_key.key_digest(plan_row.get(plan_key.KEY_FIELD))
    if found != recorded:
        return {"state": RERUN_PLAN, "moved": inputs.get("timing_records"),
                "why": f"the plan changed: recorded {recorded}, the plan pass {found}"}
    return {"state": CARRY_PLAN, "moved": inputs.get("timing_records"),
            "proof": _with_set_aside(
                {"closure": "every file of the import record at its bytes",
                 "ledgers": DEVICE_LEDGERS_EQUAL, "device": device_of(row),
                 "plan": recorded}, scope)}


# ---------------------------------------------------------------------------------------
# Re-checking a carry record (what a reader of the composed evidence runs)
# ---------------------------------------------------------------------------------------

def verify_route_carry(carry: Mapping[str, Any], run: Path, repo: Path,
                       base: Optional[Path] = None) -> Tuple[Dict[str, Any], List[str]]:
    """``(carried legs, problems)``: re-judge every leg a route carry record carries.

    Every carried leg must be present in ``run`` with exactly the files the record names
    (bytes by sha256), and must judge CARRIED again on the tree at ``repo`` (a CARRY-PLAN
    leg against the plan pass the record names, whose ``plans.json`` must still have the
    recorded sha256; a relative plan-pass path is read below ``base``). A carried leg that
    no longer judges CARRIED is a problem by name. A record that names no device carries
    nothing. A CARRY-PLAN leg is returned with the plan pass's default-preference rows
    (``plan_pass_rows``): the default as the live tree makes it, which the re-cut states
    for the leg in place of the leg's own statement.
    """
    problems: List[str] = []
    carried: Dict[str, Any] = {}
    if carry.get("schema") != SCHEMA or carry.get("kind") != "route":
        return {}, [f"not a route carry record of schema {SCHEMA}"]
    if not carry.get("device"):
        return {}, ["carry_record_names_no_device: the device is read and never assumed"]
    live = plan_key.plan_inputs(str(repo))
    cache: Dict[str, Any] = {}
    for leg, entry in sorted((carry.get("legs") or {}).items()):
        if entry.get("state") not in CARRIED:
            continue
        directory = Path(run) / leg
        if not (directory / "gate.json").is_file():
            problems.append(f"carried_leg_absent: {leg} is carried by the record and "
                            f"{directory} holds no gate.json")
            continue
        files = tree_files(directory)
        if files != entry.get("files"):
            changed = sorted(set(files) ^ set(entry.get("files") or {})
                             | {n for n in files if files.get(n) != (entry.get("files") or {}).get(n)})
            problems.append(f"carried_leg_bytes_moved: {leg}: {changed[:6]}")
            continue
        artifact = json.loads((directory / "gate.json").read_text(encoding="utf-8"))
        plan_pass = None
        if entry.get("state") == CARRY_PLAN:
            where = entry.get("plan_pass")
            path = Path(where) if where else None
            if path is not None and not path.is_absolute() and base is not None:
                path = Path(base) / path
            summary_path = (path / "plans.json") if path else None
            if summary_path is None or not summary_path.is_file():
                problems.append(f"carried_leg_plan_pass_absent: {leg}: {where!r}")
                continue
            if sha256_file(summary_path) != (entry.get("proof") or {}).get("plan_pass_sha256"):
                problems.append(f"carried_leg_plan_pass_moved: {leg}: {summary_path}")
                continue
            plan_pass = json.loads(summary_path.read_text(encoding="utf-8"))
            plan_pass["_sha256"] = sha256_file(summary_path)
        again = judge_route_leg(artifact, Path(repo), live, carry.get("device"), plan_pass,
                                cache)
        if again.get("state") not in CARRIED:
            problems.append(f"carried_leg_no_longer_holds: {leg}: {again.get('state')}: "
                            f"{again.get('why')}")
            continue
        carried[leg] = dict(again, source=entry.get("source"))
        if plan_pass is not None:
            carried[leg]["plan_pass_rows"] = [
                row for row in plan_pass.get("rows") or ()
                if isinstance(row, Mapping) and row.get("control") == DEFAULT_CONTROL]
    return carried, problems
