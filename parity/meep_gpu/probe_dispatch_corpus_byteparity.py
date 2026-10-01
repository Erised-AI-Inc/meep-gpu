"""ADVERSARIAL: the 28 corpus rows the Triton board says DISPATCH, run twice, word for word.

WHAT NO GATE HAS DONE. ``gate_dispatch_fused_route.py`` drove three HAND-BUILT
cases (``pml_2d``, ``conductive_2d``, ``dispersive_2d``) through the driver seam
and byte-compared them. The board
``results/fusion_matrix_triton_2026-08-29_final/`` then published
``served_in_dispatch: 55`` across 28 CORPUS rows — a different set of
configurations, none of which any device run has ever stepped with the fused
route live. This probe steps them.

THE PROTOCOL, per row, one device, one process::

    capture mp.Simulation (twice, same builder)
      lift driver A                       lift driver B
           |                                    |
      snapshot A0 -------- word-equal? ---- snapshot B0     (PRECONDITION)
           |                                    |
    MEEP_GPU_DISPATCH=1                MEEP_GPU_DISPATCH=1
                                        MEEP_GPU_FUSED=0
           |                                    |
      run(num_steps=N) ---- word-equal? --- run(num_steps=N) (THE CLAIM,
           at every rung of a doubling ladder)                 at each rung)

ONE VARIABLE separates the legs: the kill switch.

NON-VACUITY. A leg that fell back to the array path matches trivially. Every
dispatch leg must show ``active_step_path == "fused"``, a nonzero per-slot
dispatch count with nonzero ``programs_per_dispatch``, and a nonzero count on an
INDEPENDENT Triton-level launch counter installed on ``JITFunction.run`` — which
must read ZERO on the kill-switch leg.

ARMED (``--arm``). Three planted defects, each run on the same row set the clean
comparison passed on, each of which MUST be caught:

  ``drop_fill``      ``fill_symmetry_bc_D`` becomes a no-op.
  ``skip_wall``      ``zero_metal_B`` becomes a no-op.
  ``poison_wall``    ``zero_metal_B`` writes a nonzero constant into the wall plane
                     instead of zero — the discriminating mutation for a pass that
                     sits BETWEEN the two consults of a fused pair.
  ``withhold``       every ABSORBED consult answers False, so the array
                     constitutive half runs on top of the fused launch's.

A planted defect that the comparison does NOT catch is reported as an
INSTRUMENT FAILURE, and the clean result above it is then worth nothing.

Progress reporting: one flushed line per row per leg, and one JSONL row appended per row as
it lands.

Run (the GPU host, one pinned GPU)::

    CUDA_VISIBLE_DEVICES=6 python -u parity/meep_gpu/probe_dispatch_corpus_byteparity.py \\
        --corpus <meep>/python --out results/<stamp> --steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gate_dispatch_end_to_end as e2e  # noqa: E402

_PROGRESS: Optional[str] = None


def say(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if _PROGRESS:
        with open(_PROGRESS, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


# ---------------------------------------------------------------------------
# THE ROW SET: recomputed here from the shipped ladder, never transcribed
# ---------------------------------------------------------------------------

def dispatching_rows(board: Path, census: Path) -> List[Dict[str, Any]]:
    """The board's SERVED-IN-DISPATCH instances, recomputed from the shipped ladder.

    Never read off the board's own aggregate: the point is to ask
    ``fastpath.released_fused_arms`` about each row's configuration directly, so a
    board whose number went stale cannot silently narrow what this probe drives.
    """
    import dispatch_reachability as reach  # noqa: PLC0415

    def load(path: Path) -> List[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    record = load(census / "examples.jsonl") + load(census / "tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load(census / "tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    record = [r for r in record if r.get("measured")]
    configurations = {f"{r['leg']}:{r['row']}": r["configuration"] for r in record}

    matrix = json.loads(board.read_text())
    out: List[Dict[str, Any]] = []
    for entry in matrix["seam_instances"]:
        if not entry["predicate_admits"] or not entry["product"]:
            continue
        arm = reach.arm_label_for_product(entry["product"])
        if arm is None:
            continue
        verdict = reach.released_for_row(arm, configurations[entry["row"]])
        if not verdict["dispatches"]:
            continue
        out.append({"row": entry["row"], "seam": entry["seam"], "arm": arm,
                    "product": entry["product"],
                    "live_in_seam_passes": entry["live_in_seam_passes"],
                    "in_seam_source": entry.get("in_seam_source"),
                    "run_shape": verdict["run_shape"]})
    return out


# ---------------------------------------------------------------------------
# Builders: a corpus row -> (sim, monitors, until), the shape run_leg wants
# ---------------------------------------------------------------------------

def example_builder(corpus: Path, name: str) -> Callable[..., Tuple[Any, list, float]]:
    script = str(corpus / "examples" / name)

    def build(mp, res=None):  # noqa: ANN001, ARG001
        from parity.meep_gpu.sweep_corpus_lift_parity import capture_simulation  # noqa: PLC0415
        record, sim, restore = capture_simulation(script)
        if sim is None:
            raise RuntimeError(f"{name}: no mp.Simulation captured "
                               f"({record.get('outcome')})")
        restore()
        return sim, [], 0.0

    return build


def test_builder(corpus: Path, module_name: str, class_name: str,
                 method_name: str, timeout_s: float) -> Callable[..., Tuple[Any, list, float]]:
    module_path = str(corpus / "tests" / module_name)

    def build(mp, res=None):  # noqa: ANN001, ARG001
        from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415
        namespace = harness.build_child_namespace()
        module = namespace["_import_module"](module_path)
        record, sim, restore, _case = namespace["_run_case"](
            module, module_path, class_name, method_name, timeout_s)
        restore()
        if sim is None:
            raise RuntimeError(f"{class_name}.{method_name}: no mp.Simulation "
                               f"captured ({record.get('outcome')})")
        return sim, [], 0.0

    return build


def case_index(corpus: Path, timeout_s: float) -> Dict[str, Callable]:
    """``row label -> builder``, resolved against the corpus on disk.

    A test row's module is found by scanning the corpus's own test files for the
    class, rather than by a name convention: MEEP's test modules do not all follow
    one.
    """
    index: Dict[str, Callable] = {}
    for path in sorted((corpus / "examples").glob("*.py")):
        index[f"examples:{path.name}"] = example_builder(corpus, path.name)
    import ast  # noqa: PLC0415
    for path in sorted((corpus / "tests").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and item.name.startswith("test"):
                    label = f"tests:{node.name}.{item.name}"
                    index.setdefault(label, test_builder(
                        corpus, path.name, node.name, item.name, timeout_s))
    return index


def resolve(label: str, index: Dict[str, Callable]) -> Tuple[Optional[Callable], str]:
    """A row label -> its builder. Parameterised test rows carry a trailing suffix."""
    if label in index:
        return index[label], label
    if label.startswith("tests:"):
        stem = label.split(":", 1)[1]
        # ``TestX.test_y_1_10_7`` is ``test_y`` parameterised; strip trailing
        # ``_<int>`` groups until a real case name is left.
        parts = stem.split("_")
        while len(parts) > 1 and parts[-1].isdigit():
            parts = parts[:-1]
            candidate = f"tests:{'_'.join(parts)}"
            if candidate in index:
                return index[candidate], candidate
    return None, label


# ---------------------------------------------------------------------------
# The comparison. Mine, so it can be armed.
# ---------------------------------------------------------------------------

def compare(a: Dict[str, numpy.ndarray],
            b: Dict[str, numpy.ndarray]) -> Dict[str, Any]:
    """Every array on both drivers, compared as raw words. Never a tolerance."""
    names = sorted(set(a) | set(b))
    differing: Dict[str, Any] = {}
    words_total = 0
    words_differing = 0
    for name in names:
        if name not in a or name not in b:
            differing[name] = {"present_on": "A" if name in a else "B"}
            continue
        wa, wb = e2e._as_words(a[name]), e2e._as_words(b[name])
        if wa.shape != wb.shape or wa.dtype != wb.dtype:
            differing[name] = {"shape": [list(wa.shape), list(wb.shape)]}
            continue
        words_total += int(wa.size)
        bad = int(numpy.count_nonzero(wa != wb))
        if bad:
            words_differing += bad
            entry: Dict[str, Any] = {"words_differing": bad, "words": int(wa.size)}
            entry.update(e2e._subnormal_forensics(a[name], b[name]))
            differing[name] = entry
    return {"arrays_compared": len(names), "words_compared": words_total,
            "words_differing": words_differing,
            "arrays_differing": len(differing),
            "detail": dict(sorted(differing.items())[:12])}


# ---------------------------------------------------------------------------
# The planted defects
# ---------------------------------------------------------------------------

def plant_drop_fill(side: str = "D"):
    """``fill_symmetry_bc_{side}`` becomes a no-op on BOTH legs' driver module."""
    import meep_gpu.driver as drv  # noqa: PLC0415
    name = f"fill_symmetry_bc_{side}"
    real = getattr(drv, name)
    state = {"suppressed": 0}

    def mutated(fields):  # noqa: ANN001
        state["suppressed"] += 1

    setattr(drv, name, mutated)
    return (lambda: setattr(drv, name, real)), state


def plant_skip_wall(side: str = "B"):
    """``zero_metal_{side}`` becomes a no-op."""
    import meep_gpu.driver as drv  # noqa: PLC0415
    name = f"zero_metal_{side}"
    real = getattr(drv, name)
    state = {"suppressed": 0}

    def mutated(fields):  # noqa: ANN001
        state["suppressed"] += 1

    setattr(drv, name, mutated)
    return (lambda: setattr(drv, name, real)), state


POISON = numpy.float32(0.5)


def plant_poison_wall(side: str = "B"):
    """``zero_metal_{side}`` writes a NONZERO constant into the wall plane.

    The discriminating mutation for a pass that runs BETWEEN a fused pair's two
    consults: the array leg's constitutive half reads the poisoned plane, the
    fused leg's already read the plane inside its launch. If the comparison cannot
    see this, it cannot see anything the in-seam passes do.
    """
    import meep_gpu.driver as drv  # noqa: PLC0415
    name = f"zero_metal_{side}"
    real = getattr(drv, name)
    targets = ("Bx", "By", "Bz") if side == "B" else ("Dx", "Dy", "Dz")
    state = {"calls": 0, "written": []}

    def mutated(fields):  # noqa: ANN001
        real(fields)
        state["calls"] += 1
        for component in targets:
            array = getattr(fields, component, None)
            if array is None:
                continue
            array[0, :, :] = POISON
            if component not in state["written"]:
                state["written"].append(component)

    setattr(drv, name, mutated)
    return (lambda: setattr(drv, name, real)), state


def plant_withhold():
    """Every ABSORBED consult answers False (the leading repair slot excluded)."""
    from meep_gpu import fastpath as fp  # noqa: PLC0415
    real = fp.FastPathPlan.dispatch
    state = {"withheld": 0, "slots": {}, "skipped_leading": {}}

    def patched(self, slot, fields):  # noqa: ANN001
        if slot in getattr(self, "slots", ()):
            plan = self.step_plan.plans.get(slot)
            absorbed_by = getattr(plan, "absorbed_by", None)
            if absorbed_by is not None:
                if absorbed_by is getattr(plan, "inner", None):
                    state["skipped_leading"][slot] = \
                        state["skipped_leading"].get(slot, 0) + 1
                else:
                    state["withheld"] += 1
                    state["slots"][slot] = state["slots"].get(slot, 0) + 1
                    return False
        return real(self, slot, fields)

    fp.FastPathPlan.dispatch = patched
    return (lambda: setattr(fp.FastPathPlan, "dispatch", real)), state


def plant_skip_repair():
    """``deposit_repair.apply`` becomes a no-op returning 0.

    THE ARMING FOR THE DEPOSIT PATH. 27 of the board's 55 dispatching instances
    carry an in-seam source, and the only thing standing between the fused
    launch's pre-injection constitutive half and a wrong answer is this call. If
    the comparison cannot see it missing, it cannot see a lost deposit.
    """
    from meep_gpu import deposit_repair as dr  # noqa: PLC0415
    real = dr.apply
    state = {"suppressed": 0}

    def mutated(fields, pml, sources, pair, saved):  # noqa: ANN001
        state["suppressed"] += 1
        return 0

    dr.apply = mutated
    # The plan classes captured the module attribute at import; patch the module
    # they read through.
    return (lambda: setattr(dr, "apply", real)), state


PLANTS: Dict[str, Callable] = {
    "drop_fill_D": lambda: plant_drop_fill("D"),
    "drop_fill_B": lambda: plant_drop_fill("B"),
    "skip_wall_B": lambda: plant_skip_wall("B"),
    "poison_wall_B": lambda: plant_poison_wall("B"),
    "poison_wall_D": lambda: plant_poison_wall("D"),
    "withhold": plant_withhold,
    "skip_repair": plant_skip_repair,
}


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------

def _plans_of(driver: Any) -> Dict[str, Any]:
    plan = getattr(driver, "_fast_path", None)
    step_plan = getattr(plan, "step_plan", None)
    return dict(getattr(step_plan, "plans", {}) or {})


def _slot_plan_classes(driver: Any) -> Dict[str, str]:
    """The CLASS in every slot. ``LeadingRepairPlan``/``TrailingRepairPlan`` in the
    two slots of a pair is the only proof the deposit path is the one running."""
    return {slot: type(p).__name__ for slot, p in sorted(_plans_of(driver).items())}


def _repairs(driver: Any) -> Dict[str, int]:
    """Points repaired, per leading-repair slot. Zero would mean the repair is empty
    — the ``continuous_src_time::dipole`` failure shape, one level up."""
    out: Dict[str, int] = {}
    for slot, p in sorted(_plans_of(driver).items()):
        if type(p).__name__ == "LeadingRepairPlan":
            out[slot] = int(getattr(p, "repairs", -1))
    return out


DISPATCH_ENV = {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
                "MEEP_GPU_FUSE_ARMS": None}
ARRAY_ENV = {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": "0",
             "MEEP_GPU_FUSE_ARMS": None}


def _reseed(seed: Optional[int]) -> None:
    """Put every RNG a MEEP source could draw from back to one state.

    A corpus row whose source is ``mp.CustomSource(src_func=lambda t:
    np.random.randn())`` injects a DIFFERENT current into each leg, and the two
    runs then part for a reason that has nothing to do with dispatch. Seeding
    immediately before each leg's chunk makes the two draw the same sequence;
    the ``array_array`` control below is what proves whether a row needed it.
    """
    if seed is None:
        return
    import random as _random  # noqa: PLC0415
    numpy.random.seed(seed)
    _random.seed(seed)


def run_row(label: str, builder: Callable, gpu_id: int, steps: int,
            ladder: List[int], plant: Optional[str],
            counter: Any, control: str = "off",
            seed: Optional[int] = None) -> Dict[str, Any]:
    restore_plant = None
    plant_state: Dict[str, Any] = {}
    a = b = None
    env_a = DISPATCH_ENV if control != "array_array" else ARRAY_ENV
    env_b = ARRAY_ENV if control != "dispatch_dispatch" else DISPATCH_ENV
    row: Dict[str, Any] = {"row": label, "plant": plant, "steps_requested": steps,
                           "control": control, "seed": seed}
    try:
        a = e2e.run_leg(label, builder, "legA", env_a, counter, gpu_id, None)
        b = e2e.run_leg(label, builder, "legB", env_b, counter, gpu_id, None)
        row["grid_shape"] = a["grid_shape"]

        pre = compare(e2e.collect_state(a["driver"]), e2e.collect_state(b["driver"]))
        row["precondition"] = {"words_differing": pre["words_differing"],
                               "arrays_differing": pre["arrays_differing"],
                               "arrays_compared": pre["arrays_compared"],
                               "words_compared": pre["words_compared"],
                               "detail": pre["detail"]}
        if pre["words_differing"] or pre["arrays_differing"]:
            row["verdict"] = "HARNESS: the two lifts did not start equal"
            return row

        if plant:
            restore_plant, plant_state = PLANTS[plant]()

        done = 0
        row["checkpoints"] = []
        row["first_divergent_checkpoint"] = None
        for rung in ladder:
            chunk = rung - done
            if chunk <= 0:
                continue
            _reseed(seed)
            e2e.step_leg(label, a, counter, chunk)
            _reseed(seed)
            e2e.step_leg(label, b, counter, chunk)
            done = rung
            result = compare(e2e.collect_state(a["driver"]),
                             e2e.collect_state(b["driver"]))
            row["checkpoints"].append({"steps": rung,
                                       "words_differing": result["words_differing"],
                                       "arrays_differing": result["arrays_differing"],
                                       "words_compared": result["words_compared"],
                                       "detail": result["detail"]})
            say(f"{label}: rung {rung} words_differing="
                f"{result['words_differing']} arrays={result['arrays_differing']}")
            if result["words_differing"] or result["arrays_differing"]:
                row["first_divergent_checkpoint"] = rung
                break

        row["dispatch_leg"] = {
            "active_step_path": a["active_step_path"],
            "plan": a["plan"],
            "triton_launches": a["triton_launches"],
            "steps": a["steps"],
        }
        row["array_leg"] = {
            "active_step_path": b["active_step_path"],
            "plan": b["plan"],
            "triton_launches": b["triton_launches"],
            "steps": b["steps"],
        }
        ran = e2e.kernels_actually_ran(a)
        row["kernels_actually_ran"] = ran
        row["array_leg_launched_nothing"] = (b["triton_launches"]["total"] == 0)
        row["plant_state"] = plant_state
        row["fused_slots"] = [slot for slot, arm_ in
                              (a["plan"].get("arms") or {}).items()
                              if isinstance(arm_, str) and "fused pair" in arm_]
        row["arms"] = a["plan"].get("arms")
        row["slot_plan_classes"] = _slot_plan_classes(a["driver"])
        row["deposit_repairs"] = _repairs(a["driver"])
        diverged = row["first_divergent_checkpoint"] is not None
        if control != "off":
            # A CONTROL IS NOT A CLAIM ABOUT DISPATCH. Both legs took the same
            # route, so a divergence here is the ROW's own nondeterminism (a
            # random source function, an unseeded draw) and every verdict about
            # this row from a one-variable comparison is void until it is seeded.
            row["kernels_actually_ran"] = e2e.kernels_actually_ran(a)
            row["verdict"] = ("CONTROL-IDENTICAL" if not diverged
                              else "CONTROL-DIVERGED: the row is nondeterministic")
            return row
        # NON-VACUITY FIRST. A comparison whose dispatch leg never launched a
        # kernel compares the array path against itself; reporting that as
        # "identical" is the exact green-that-means-nothing this probe exists to
        # refuse. A PLANTED run is judged the same way: a defect "caught" by a leg
        # that never dispatched proves nothing about the dispatched composition.
        if not ran["kernels_ran"]:
            row["verdict"] = ("VACUOUS: the dispatch leg did not run a fused kernel ("
                              + "; ".join(ran["failures"]) + ")")
        elif not row["array_leg_launched_nothing"]:
            row["verdict"] = ("VACUOUS: the kill-switch leg launched "
                              f"{b['triton_launches']['total']} Triton kernels")
        elif not row["fused_slots"]:
            row["verdict"] = "VACUOUS: no slot carries a fused arm label"
        elif plant:
            row["verdict"] = ("CAUGHT" if diverged
                              else "INSTRUMENT FAILURE: planted defect NOT caught")
        else:
            row["verdict"] = "IDENTICAL" if not diverged else "DIVERGED"
    except BaseException as exc:  # noqa: BLE001
        row["verdict"] = f"ERROR: {type(exc).__name__}: {exc}"[:900]
        import traceback  # noqa: PLC0415
        row["traceback"] = traceback.format_exc()[-2000:]
    finally:
        if restore_plant is not None:
            restore_plant()
        for leg in (a, b):
            if leg is None:
                continue
            try:
                leg["driver"].close()
            except BaseException:  # noqa: BLE001
                pass
    return row


def main() -> int:
    global _PROGRESS
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--board", default=None)
    parser.add_argument("--census", default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rows", default=None, help="comma list, or a prefix filter")
    parser.add_argument("--plant", default=None, choices=sorted(PLANTS))
    parser.add_argument("--case-timeout", type=float, default=900.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--control", default="off",
                        choices=("off", "array_array", "dispatch_dispatch"),
                        help="both legs take the SAME route: any divergence is the "
                             "row's own nondeterminism, not dispatch")
    parser.add_argument("--seed", type=int, default=None,
                        help="reseed numpy/random before EACH leg's chunk")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _PROGRESS = str(out / "progress.log")
    results = out / "rows.jsonl"

    api = Path(REPO_API)
    board = Path(args.board) if args.board else (
        api / "parity/meep_gpu/results/fusion_matrix_triton_2026-08-29_final/fusion_matrix.json")
    census = Path(args.census) if args.census else (
        api / "parity/meep_gpu/results/predicate_coverage_triton_2026-08-28")

    instances = dispatching_rows(board, census)
    labels = sorted({e["row"] for e in instances})
    say(f"SERVED IN DISPATCH recomputed: {len(instances)} instances across "
        f"{len(labels)} rows")
    (out / "dispatch_set.json").write_text(
        json.dumps({"instances": instances, "rows": labels}, indent=1))

    corpus = Path(args.corpus)
    index = case_index(corpus, args.case_timeout)
    if args.rows:
        wanted = [w.strip() for w in args.rows.split(",") if w.strip()]
        labels = [lab for lab in labels if any(lab == w or lab.startswith(w)
                                               for w in wanted)]
    if args.limit:
        labels = labels[:args.limit]

    ladder = e2e.checkpoint_schedule(args.steps, None)
    say(f"ladder: {ladder}")

    counter = e2e.TritonLaunchCounter()
    counter.install()
    say("Triton launch counter installed on JITFunction.run + launch_enter_hook")

    summary = {"identical": 0, "diverged": 0, "caught": 0,
               "instrument_failure": 0, "error": 0, "unresolved": 0, "vacuous": 0,
               "control_identical": 0, "control_diverged": 0}
    for n, label in enumerate(labels, 1):
        builder, resolved = resolve(label, index)
        if builder is None:
            say(f"[{n}/{len(labels)}] {label}: NO BUILDER (not in corpus)")
            summary["unresolved"] += 1
            with results.open("a") as h:
                h.write(json.dumps({"row": label, "verdict": "UNRESOLVED"}) + "\n")
            continue
        say(f"[{n}/{len(labels)}] {label} -> {resolved}"
            + (f"  PLANT={args.plant}" if args.plant else ""))
        started = time.time()
        row = run_row(label, builder, args.gpu, args.steps, ladder, args.plant,
                      counter, args.control, args.seed)
        row["resolved_from"] = resolved
        row["wall_s"] = round(time.time() - started, 1)
        with results.open("a") as h:
            h.write(json.dumps(row) + "\n")
            h.flush()
        verdict = row["verdict"]
        say(f"[{n}/{len(labels)}] {label}: {verdict} ({row['wall_s']} s)")
        if verdict == "IDENTICAL":
            summary["identical"] += 1
        elif verdict == "DIVERGED":
            summary["diverged"] += 1
        elif verdict == "CAUGHT":
            summary["caught"] += 1
        elif verdict.startswith("INSTRUMENT"):
            summary["instrument_failure"] += 1
        elif verdict.startswith("VACUOUS"):
            summary["vacuous"] += 1
        elif verdict == "CONTROL-IDENTICAL":
            summary["control_identical"] += 1
        elif verdict.startswith("CONTROL-DIVERGED"):
            summary["control_diverged"] += 1
        else:
            summary["error"] += 1
        say(f"    running: {summary}")

    (out / "summary.json").write_text(json.dumps(
        {"summary": summary, "rows": labels, "steps": args.steps,
         "plant": args.plant, "ladder": ladder, "control": args.control,
         "seed": args.seed}, indent=1))
    say(f"SUMMARY {summary}")
    bad = (summary["diverged"] or summary["instrument_failure"]
           or summary["vacuous"] or summary["error"])
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
