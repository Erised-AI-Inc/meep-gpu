#!/usr/bin/env python3
"""L3, one corpus row: lifted and DISPATCHED on this host's GPU, judged against the stored
digests of the reference MEEP's words. MEEP is imported (the lift needs it) and is never
stepped. ``l3_corpus.py`` runs one of these per row, each in a process of its own.

    python -s -u l3_row.py SPEC.json

SPEC (written by ``l3_corpus.py``):

* ``label``: the corpus row;
* ``table``: ``triton`` (the NVIDIA default precedence), ``cuda``
  (``MEEP_GPU_BACKEND_PREFERENCE=cuda``) or ``metal`` (the Apple GPU), see :data:`TABLES`;
* ``mode``: the subnormal baseline, ``keep`` or ``flush``;
* ``shipped_default``: when true the caller installs no subnormal policy, and the host FPU
  is recorded at every compared step but not enforced (informational);
* ``refs``: ``{set: row directory}``, each holding the reference's ``digests.json`` and
  ``planes.npy``; ``ref_modes``: ``{set: the mode that set's MEEP ran in}``;
* ``primary``: the set the row's top-level figures describe;
* ``image_set`` and ``meep_images``: the set whose MEEP the lift must load, and its
  ``{image: [sha256, ...]}``;
* ``cases_file``: the corpus module; ``out_json``: where the result goes.

Every judging function is the tree's own, called unmodified:
``meep_identity_tier_c.load_cases`` / ``build_case`` (the corpus module's build,
``after_lift``, ``steps_planned``), ``meep_identity.compared_steps`` / ``cell_classes`` /
``step_digests`` / ``read_digests`` / ``compare_planes`` / ``flip_word`` /
``flipped_planes`` / ``digest_words`` / ``subnormal_state``,
``meep_gpu.install_subnormal_policy``, ``meep_gpu.lift_simulation`` (``prefer_gpu=True``,
``MEEP_GPU_DISPATCH`` unset) and ``sweep_corpus_lift_parity.dispatch_facts``.

ORDER: NumPy, then MEEP; the loaded MEEP images against the image set's; the subnormal
policy, then the host FPU measured in it; the row built by the corpus module; the lift on
this host's GPU (``driver.gpu`` must be the table's); the row's own ``after_lift`` check;
N from the row's ``steps_planned``, which must equal every judged set's N and schedule,
and the compared cells and storage dtype equal to every set's; then the driver alone
steps. At every compared step the FPU is measured again, the twelve components' stored
and ``get_array`` words are digested ONCE (with x-plane digests up to step 20) and
compared with each set's MEEP digests (:func:`judge`). At each set's planted-flip step one
word is flipped by one ulp in a copy: it must change that component's digest and exactly
the plane the word lies in, or the comparison is not shown able to fail. The dispatch
record is read after the first step (the configuration freeze is inside ``step()``) and
at the end, with the plan's launch counters.

Prints one flushed progress line per compared step and, last, one line ``L3-JSON {...}``
with the row's result, which it also writes to ``out_json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

MARK = "L3-JSON "
STARTED = time.time()

#: What each table asks of the lift: the driver it must build, the executors the subnormal
#: policy is installed on (``None``: the package's default set) and the backend preference
#: the run's environment must carry.
TABLES: Dict[str, Dict[str, Any]] = {
    "triton": {"driver_gpu": "cuda", "executors": None, "preference": None},
    "cuda": {"driver_gpu": "cuda", "executors": None, "preference": "cuda"},
    "metal": {"driver_gpu": "metal", "executors": ("host", "mps"), "preference": None},
}

#: The variables a row records as it found them.
RECORDED_ENV = ("MEEP_GPU_BACKEND_PREFERENCE", "MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED",
                "MEEP_GPU_FUSE_ARMS", "MEEP_GPU_SUBNORMAL_POLICY", "MEEP_GPU_HARDWARE",
                "MEEP_GPU_METAL_RESIDENCY", "MEEP_GPU_METAL_BAND_WITNESS",
                "MEEP_GPU_KERNEL_TABLE", "CUPY_ACCELERATORS", "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS", "CUPY_CACHE_DIR", "TRITON_CACHE_DIR", "PYTHONPATH",
                "PYTHONNOUSERSITE")

#: The most differing steps a set keeps in full; the count is always kept.
DIFFERING_KEPT = 40
#: The most differing planes a differing component names.
PLANES_KEPT = 16


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def plain(value: Any) -> Any:
    """JSON-safe copy of a report (NumPy scalars, tuples, sets, objects)."""
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------------------
# Launch counters
# ---------------------------------------------------------------------------------------

def counter_leaf_total(value: Any) -> int:
    """The sum of every integer leaf of a launch-counter block (dispatches, plan launches
    and programs per dispatch together): the figure the first L3 records called launches."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, dict):
        return sum(counter_leaf_total(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(counter_leaf_total(item) for item in value)
    return 0


def counter_sums(counters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The per-slot launch counters summed: the kernel module's dispatches, the plans' own
    launch counts (a slot whose plan does not count its launches, the absorbed slot of a
    fused pair among them, is named in ``plan_launches_unknown_slots``), and the backend
    and arm of every slot that launched."""
    dispatches = plan_launches = 0
    unknown: List[str] = []
    backends: Dict[str, Any] = {}
    arms: Dict[str, Any] = {}
    for slot, entry in (counters or {}).items():
        if not isinstance(entry, dict):
            continue
        dispatches += int(entry.get("dispatches") or 0)
        count = entry.get("plan_launches")
        if count is None:
            unknown.append(slot)
        else:
            plan_launches += int(count)
        if int(entry.get("dispatches") or 0) > 0:
            backends[slot] = entry.get("backend")
            arms[slot] = entry.get("arm")
    return {"dispatches": dispatches, "plan_launches": plan_launches,
            "plan_launches_unknown_slots": unknown,
            "launched_slot_backends": backends, "launched_slot_arms": arms}


# ---------------------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------------------

def new_tally() -> Dict[str, Any]:
    """One reference set's running counts over a row."""
    return {"stored_equal": 0, "reader_equal": 0, "digests": 0, "first": None,
            "differing": [], "differing_count": 0}


def new_flip(document: Mapping[str, Any]) -> Dict[str, Any]:
    """The planted-flip control a set's record names, not yet run."""
    spec = document["planted_flip"]
    return {"step": int(spec["step"]), "component": spec["component"], "word": int(spec["word"]),
            "plane": int(spec["plane"]), "caught": False, "placed": False}


def compare_step(document: Mapping[str, Any], recorded_planes: Any, plane_steps: Sequence[int],
                 found: Any, target: int, tally: Dict[str, Any], flip: Dict[str, Any],
                 flipped_planes_of: Optional[Callable[[str, int], Any]]) -> Dict[str, Any]:
    """Judge one compared step against one reference set; returns the differing components.

    ``found`` is the step's :class:`meep_gpu.meep_identity.StepDigests`. Every component's
    stored and reader digests are compared with the set's; a differing component at a step
    that carries plane digests names its differing x-planes. At the set's planted-flip
    step one word of the found words is flipped by one ulp: the flipped digest must differ
    from MEEP's (``caught``), and the flipped plane digests, from ``flipped_planes_of``,
    must differ in exactly the plane the record names (``placed``).
    """
    from meep_gpu import meep_identity as identity  # noqa: PLC0415

    key = str(int(target))
    with_planes = int(target) in plane_steps
    bad: Dict[str, Any] = {}
    for component in identity.COMPONENTS:
        same_stored = found.stored[component] == document["stored"][key][component]
        same_reader = found.reader[component] == document["reader"][key][component]
        tally["stored_equal"] += bool(same_stored)
        tally["reader_equal"] += bool(same_reader)
        tally["digests"] += 1
        if not (same_stored and same_reader):
            entry: Dict[str, Any] = {"stored_equal": bool(same_stored),
                                     "reader_equal": bool(same_reader)}
            if with_planes and found.planes is not None:
                index = identity.COMPONENTS.index(component)
                entry["planes_differing"] = identity.compare_planes(
                    recorded_planes[list(plane_steps).index(int(target)), index],
                    found.planes[index])[:PLANES_KEPT]
            bad[component] = entry
    if bad:
        tally["differing_count"] += 1
        if len(tally["differing"]) < DIFFERING_KEPT:
            tally["differing"].append({"step": int(target), "components": bad})
        if tally["first"] is None:
            tally["first"] = {"step": int(target), "components": bad}
    if int(target) == int(flip["step"]):
        component = flip["component"]
        words = identity.flip_word(found.words[component], flip["word"])
        flip["caught"] = identity.digest_words(words) != document["stored"][key][component]
        if with_planes and flipped_planes_of is not None:
            planes = identity.compare_planes(
                recorded_planes[list(plane_steps).index(int(target)),
                                identity.COMPONENTS.index(component)],
                flipped_planes_of(component, flip["word"]))
            flip["planes_differing"] = planes
            flip["placed"] = planes == [int(flip["plane"])]
    return bad


def judge(driver: Any, documents: Mapping[str, Mapping[str, Any]],
          planes_of: Mapping[str, Any], masks: Mapping[str, Any], steps: Sequence[int], *,
          mode: str, enforce_fpu: bool, fpu_state: Callable[[], Mapping[str, Any]],
          after_first_step: Optional[Callable[[Any], Any]] = None,
          progress: Optional[Callable[[str], None]] = None, label: str = "") -> Dict[str, Any]:
    """Step ``driver`` through the compared ``steps`` and judge every set at each of them.

    The words are digested once per compared step and compared with every set in
    ``documents``. ``fpu_state`` is measured at every compared step; with ``enforce_fpu``
    a mode other than ``mode`` stops the row by name. ``after_first_step`` is called once,
    after the first ``driver.step()``."""
    from meep_gpu import meep_identity as identity  # noqa: PLC0415

    names = list(documents)
    tallies = {name: new_tally() for name in names}
    flips = {name: new_flip(documents[name]) for name in names}
    plane_steps = {name: [int(step) for step in documents[name]["planes"]["steps"]]
                   for name in names}
    fpu_modes: List[List[Any]] = []
    facts_first = None
    first_called = False

    def flipped(component: str, word: int) -> Any:
        return identity.flipped_planes(driver.get_field(component, cell_centered=False),
                                       masks[component], word)

    for number, target in enumerate(steps, 1):
        while driver.step_count < target:
            driver.step()
            if not first_called:
                first_called = True
                if after_first_step is not None:
                    facts_first = after_first_step(driver)
        now = fpu_state()
        fpu_modes.append([int(target), now["mode"]])
        if enforce_fpu and now["mode"] != mode:
            raise RuntimeError(f"the FPU is {now['mode']} at step {target}, not {mode}")
        with_planes = any(int(target) in plane_steps[name] for name in names)
        found = identity.step_digests(driver, masks, planes=with_planes)
        for name in names:
            compare_step(documents[name], planes_of[name], plane_steps[name], found, target,
                         tallies[name], flips[name], flipped)
        if progress is not None:
            progress(f"{label} {number}/{len(steps)} step {target}: "
                     + "; ".join(f"{name} stored {tallies[name]['stored_equal']} reader "
                                 f"{tallies[name]['reader_equal']} of {tallies[name]['digests']}"
                                 for name in names)
                     + f" ({time.time() - STARTED:.1f} s)")
    return {"per_set": {name: dict(tallies[name], planted_flip=flips[name]) for name in names},
            "fpu_checks": len(fpu_modes), "fpu_modes": fpu_modes,
            "fpu_steps_in_mode": sum(1 for _step, seen in fpu_modes if seen == mode),
            "dispatch_after_first_step": facts_first}


# ---------------------------------------------------------------------------------------
# Verdicts (the driver reads these too)
# ---------------------------------------------------------------------------------------

def set_verdict(row: Mapping[str, Any], name: str) -> str:
    """``holds``, ``differs`` or ``error`` for one reference set.

    A row holds against a set when it compared at least one digest, every stored and every
    reader digest is equal to the set's, the set's planted flip was caught and placed, and
    the FPU was measured at every compared step."""
    if row.get("error"):
        return "error"
    tally = (row.get("per_set") or {}).get(name) or {}
    digests = tally.get("digests") or 0
    flip = tally.get("planted_flip") or {}
    held = (digests > 0 and tally.get("stored_equal") == digests
            and tally.get("reader_equal") == digests
            and bool(flip.get("caught")) and bool(flip.get("placed"))
            and row.get("fpu_checks") == row.get("compared_steps"))
    return "holds" if held else "differs"


def dispatch_state(row: Mapping[str, Any]) -> str:
    """``dispatched``, ``array`` or ``error``: whether compiled kernels stepped the row.

    Dispatched means the step path read ``fused`` at the end, the plan named dispatched
    slots, the kernel modules counted at least one dispatch, and the plans counted at
    least one launch (or do not count their own, as an absorbed slot does not)."""
    if row.get("error"):
        return "error"
    end = row.get("dispatch_at_end") or {}
    slots = (end.get("dispatch") or {}).get("dispatched_slots") or []
    sums = row.get("launch_sums") or {}
    plan_ok = (sums.get("plan_launches") or 0) > 0 or bool(sums.get("plan_launches_unknown_slots"))
    if (end.get("step_path") == "fused" and slots and (sums.get("dispatches") or 0) > 0
            and plan_ok):
        return "dispatched"
    return "array"


def row_verdict(row: Mapping[str, Any], primary: str) -> str:
    """One word for the row against the primary set: ``holds_dispatched``,
    ``holds_array`` (bit-identical, but the plan was refused and the array path stepped),
    ``differs`` or ``error``."""
    verdict = set_verdict(row, primary)
    if verdict != "holds":
        return verdict
    return "holds_dispatched" if dispatch_state(row) == "dispatched" else "holds_array"


# ---------------------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------------------

def run(spec: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy  # noqa: F401 - NumPy before MEEP, as every reference run

    import meep as mp

    mp.verbosity(0)
    import meep_gpu
    from meep_gpu import meep_identity as identity
    from parity.meep_gpu import meep_identity_tier_c as tier_c
    from parity.meep_gpu.meep_identity_smoke import load_tool
    from parity.meep_gpu.sweep_corpus_lift_parity import dispatch_facts

    label, mode, table_name = spec["label"], spec["mode"], spec["table"]
    table = TABLES[table_name]
    shipped = bool(spec.get("shipped_default"))
    sets = list(spec["refs"])
    primary = spec["primary"]
    result: Dict[str, Any] = {
        "label": label, "mode": mode, "table": table_name, "shipped_default": shipped,
        "sets": sets, "primary": primary, "image_set": spec.get("image_set"),
        "env": {name: os.environ.get(name) for name in RECORDED_ENV},
        "meep": {"version": str(mp.__version__),
                 "single_precision": bool(mp.is_single_precision()),
                 "file": os.path.realpath(mp.__file__)},
        "package_file": os.path.realpath(meep_gpu.__file__)}
    preference = os.environ.get("MEEP_GPU_BACKEND_PREFERENCE")
    if preference != table["preference"]:
        result["error"] = (f"table {table_name!r} runs with MEEP_GPU_BACKEND_PREFERENCE="
                           f"{table['preference']!r}; this process has {preference!r}")
        return result
    check = load_tool("check_install")
    images = {os.path.basename(path): check.sha256_of(path)
              for path in check.loaded_image_paths()
              if os.path.basename(path).startswith(("libmeep", "_meep"))}
    result["meep"]["images"] = images
    expected = spec.get("meep_images") or {}
    result["meep_images_equal_reference"] = bool(expected) and all(
        images.get(name) in digests for name, digests in expected.items())
    if not result["meep_images_equal_reference"]:
        result["error"] = (f"the MEEP images loaded here {images} are not the "
                           f"{spec.get('image_set')} set's {expected}")
        return result
    if not mp.is_single_precision():
        result["error"] = "this MEEP is not a single-precision build"
        return result

    # THE POLICY FIRST, MEEP imported: the host lever is MEEP's own switch.
    if shipped:
        result["installed_policy"] = "nothing installed by the caller (the shipped default)"
    else:
        if table["executors"] is None:
            installed = meep_gpu.install_subnormal_policy(mode)
        else:
            installed = meep_gpu.install_subnormal_policy(mode, executors=table["executors"])
        result["installed_policy"] = plain(dict(installed))
    state = identity.subnormal_state()
    result["subnormals_at_start"] = state
    if not shipped and state["mode"] != mode:
        result["error"] = f"the host FPU is {state['mode']} after installing {mode!r}"
        return result
    if mode == "flush" and os.environ.get("OMP_NUM_THREADS") != "1":
        result["error"] = "flush is a reference only at OMP_NUM_THREADS=1"
        return result

    documents: Dict[str, Any] = {}
    planes_of: Dict[str, Any] = {}
    for name in sets:
        ref = spec["refs"][name]
        document, recorded_planes = identity.read_digests(
            os.path.join(ref, "digests.json"), os.path.join(ref, "planes.npy"))
        want = spec["ref_modes"][name]
        if document["case"] != label or document["mode"] != want:
            result["error"] = (f"the {name} record is case {document['case']!r} mode "
                               f"{document['mode']!r}, not {label!r} {want!r}")
            return result
        documents[name], planes_of[name] = document, recorded_planes

    cases = tier_c.load_cases(spec["cases_file"])
    sim = tier_c.build_case(cases, mp, label)
    lift_started = time.time()
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=True)
    result["lift_seconds"] = round(time.time() - lift_started, 3)
    try:
        result["driver_gpu"] = driver.gpu
        if driver.gpu != table["driver_gpu"]:
            result["error"] = (f"the lift resolved driver.gpu = {driver.gpu!r}, not "
                               f"{table['driver_gpu']!r}")
            return result
        result["grid_shape"] = [int(value) for value in driver.grid.shape]
        result["cylindrical"] = bool(getattr(driver.grid, "cylindrical", False))
        result["case_facts"] = plain(cases.after_lift(mp, sim, driver))
        n_steps = int(cases.steps_planned(mp, sim))
        steps = identity.compared_steps(n_steps)
        masks = {name: identity.cell_classes(driver.grid, name)["compared"]
                 for name in identity.COMPONENTS}
        cells = {name: int(masks[name].sum()) for name in identity.COMPONENTS}
        storage = ("complex64" if driver.get_field("Ez", cell_centered=False).dtype.kind == "c"
                   else "float32")
        for name, document in documents.items():
            if n_steps != int(document["n_steps"]) or steps != [int(s) for s in document["steps"]]:
                result["error"] = (f"N = {n_steps} and {len(steps)} compared steps here; the "
                                   f"{name} record holds N = {document['n_steps']} and "
                                   f"{len(document['steps'])}")
                return result
            if cells != {c: int(document["compared_cells"][c]) for c in identity.COMPONENTS}:
                result["error"] = (f"the lifted driver compares {cells} cells; the {name} "
                                   f"record was cut over {document['compared_cells']}")
                return result
            if storage != document["storage"]:
                result["error"] = f"storage {storage}; the {name} record's is {document['storage']}"
                return result
        say(f"{label} {mode} {table_name}: lifted on {driver.gpu} in {result['lift_seconds']} s, "
            f"stored {result['grid_shape']}, {len(steps)} compared steps to N = {n_steps}; "
            f"sets {sets}")

        def first_facts(lifted: Any) -> Any:
            facts = plain(dispatch_facts(lifted))
            say(f"{label}: after step 1 step_path={facts['step_path']} "
                f"table={facts['dispatch']['table']} dispatched "
                f"{len(facts['dispatch']['dispatched_slots'])}/{facts['dispatch']['slots']} slots"
                + (f" refused_because={facts['dispatch']['refused_because']!r}"
                   if facts['dispatch']['refused_because'] else ""))
            return facts

        step_started = time.time()
        judged = judge(driver, documents, planes_of, masks, steps, mode=mode,
                       enforce_fpu=not shipped, fpu_state=identity.subnormal_state,
                       after_first_step=first_facts, progress=say, label=label)
        facts_end = plain(dispatch_facts(driver))
        report = plain(driver.fast_path_report() or {})
        counters = report.get("launch_counters")
        result.update(judged)
        result.update({
            "n_steps": n_steps, "compared_steps": len(steps), "storage": storage,
            "subnormals_at_end": identity.subnormal_state(),
            "step_seconds": round(time.time() - step_started, 3),
            "dispatch_at_end": facts_end,
            "launch_counters": counters, "launch_sums": counter_sums(counters),
            "counter_leaf_total": counter_leaf_total(counters),
            "residency_syncs": report.get("residency_syncs"),
            "composition": report.get("composition"),
            "fast_path_report": report,
        })
        top = result["per_set"][primary]
        result.update({"digests": top["digests"], "stored_equal": top["stored_equal"],
                       "reader_equal": top["reader_equal"], "planted_flip": top["planted_flip"],
                       "first_difference": top["first"],
                       "differing_step_count": top["differing_count"]})
    finally:
        driver.close()
    result["seconds"] = round(time.time() - STARTED, 3)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as handle:
        spec = json.load(handle)
    try:
        result = run(spec)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent as the row's error
        traceback.print_exc()
        sys.stderr.flush()
        result = {"label": spec.get("label"), "mode": spec.get("mode"),
                  "table": spec.get("table"), "error": f"{type(exc).__name__}: {exc}",
                  "seconds": round(time.time() - STARTED, 3)}
    text = json.dumps(result, sort_keys=True, default=str)
    with open(spec["out_json"], "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(MARK + text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
