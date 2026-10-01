"""ADVERSARIAL: hunt a configuration that DISPATCHES a fused route whose LIVE
in-seam passes the winning arm does not carry.

THE SHAPE OF THE DEFECT BEING HUNTED. A fused pair occupies TWO driver slots
(``step_B``/``update_H`` or ``step_D``/``update_E``) and performs both in ONE
launch, at the FIRST consult. Between those two consults ``driver.step`` runs
four things unconditionally, outside every consult
(``driver.py:3293-3299`` / ``:3306-3312``)::

    <consult step_B>            <- the fused launch runs BOTH halves here
    for source in magnetic: source.inject(...)
    fill_symmetry_bc_B(fields)
    zero_metal_B(fields)
    fill_folded_far_ghosts_B(fields)
    <consult update_H>          <- NoopPlan, or the deposit repair

Anything in that window that MUTATES state which the constitutive half reads is
a pass the launch performed too early. Three mechanisms are supposed to cover
all four: the wall clear is carried INSIDE the fused kernel, the two fills are
excluded by ``FUSED_RELEASE_ENVELOPE``'s ``folded: absent`` rung, and the source
injection is carried by ``deposit_repair``'s two-consult plans.

WHAT THIS PROBE MEASURES, and why it is not the board's column. The board's
``live_in_seam_passes`` is a DECLARATION — ``metal_kernels/launch.live_sub_steps``
derives it from the source list and the grid. This probe wraps each of the four
passes and records whether the call actually CHANGED any word of its family's
arrays. A pass that is declared live and writes nothing is inert; a pass that
writes while the seam is fused is the defect. Both are measured, per
configuration, on the device.

Every configuration that dispatches a fused arm is then byte-compared, dispatch
ON against the kill switch, exactly as ``probe_dispatch_corpus_byteparity.py``
does — so a mutation the pass instrument might miss still has to survive a
word-for-word comparison.

Progress reporting: one flushed line per configuration, one JSONL row appended as it lands.

Run::

    CUDA_VISIBLE_DEVICES=6 python -u parity/meep_gpu/probe_dispatch_seam_sweep.py \\
        --out results/<stamp> --steps 64
"""

from __future__ import annotations

import argparse
import itertools
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
# The in-seam pass instrument: EFFECT, not declaration
# ---------------------------------------------------------------------------

IN_SEAM = {
    "fill_symmetry_bc_B": ("Bx", "By", "Bz"),
    "zero_metal_B": ("Bx", "By", "Bz"),
    "fill_folded_far_ghosts_B": ("Bx", "By", "Bz"),
    "fill_symmetry_bc_D": ("Dx", "Dy", "Dz"),
    "zero_metal_D": ("Dx", "Dy", "Dz"),
    "fill_folded_far_ghosts_D": ("Dx", "Dy", "Dz"),
}


def _fingerprint(fields: Any, names: Tuple[str, ...]) -> Tuple[int, ...]:
    """A cheap word-level fingerprint of one field family.

    ``uint32`` sum over the raw bytes: two arrays with the same sum can differ,
    but a pass that changes a word almost never leaves the sum fixed, and the
    byte comparison downstream is the real detector. This is a per-pass
    ATTRIBUTION, not the verdict.
    """
    out = []
    for name in names:
        array = getattr(fields, name, None)
        if array is None:
            out.append(0)
            continue
        raw = array.reshape(-1)
        try:
            words = raw.view(numpy.uint32) if raw.dtype == numpy.float32 \
                else raw.view(numpy.uint32)
        except Exception:  # noqa: BLE001
            out.append(-1)
            continue
        out.append(int(words.sum(dtype=numpy.uint64)))
    return tuple(out)


def instrument_in_seam_passes(max_steps: int = 8):
    """Wrap the four in-seam passes; record calls and WRITES. Returns (restore, state)."""
    import meep_gpu.driver as drv  # noqa: PLC0415

    state: Dict[str, Dict[str, int]] = {
        name: {"calls": 0, "wrote": 0, "instrumented_calls": 0}
        for name in IN_SEAM}
    reals: Dict[str, Any] = {}
    budget = {name: max_steps * 2 for name in IN_SEAM}

    def wrap(name: str, names: Tuple[str, ...]):
        real = getattr(drv, name)
        reals[name] = real

        def wrapped(fields):  # noqa: ANN001
            state[name]["calls"] += 1
            if budget[name] <= 0:
                return real(fields)
            budget[name] -= 1
            before = _fingerprint(fields, names)
            real(fields)
            after = _fingerprint(fields, names)
            state[name]["instrumented_calls"] += 1
            if before != after:
                state[name]["wrote"] += 1

        setattr(drv, name, wrapped)

    for name, names in IN_SEAM.items():
        if hasattr(drv, name):
            wrap(name, names)

    def restore() -> None:
        for name, real in reals.items():
            setattr(drv, name, real)

    return restore, state


def instrument_injects(driver: Any, max_calls: int = 32):
    """Wrap every source's ``inject``; record whether it CHANGED the field.

    A deposit of exactly zero is a leg that measures nothing — the
    ``continuous_src_time::dipole`` shape this campaign already caught once one
    level up. ``wrote`` counts the injections that actually moved a word.
    """
    state = {"calls": 0, "wrote": 0, "by_source": {}}
    restores: List[Callable[[], None]] = []
    for index, source in enumerate(getattr(driver, "_sources", ()) or ()):
        real = getattr(source, "inject", None)
        if real is None:
            continue
        kind = type(source).__name__
        family = ("Bx", "By", "Bz") if str(getattr(source, "field_type", "")) == "B" \
            else ("Dx", "Dy", "Dz")
        key = f"{index}:{kind}"
        state["by_source"][key] = {"calls": 0, "wrote": 0, "field_type":
                                   str(getattr(source, "field_type", ""))}

        def make(real=real, family=family, key=key):
            def wrapped(fields, t):  # noqa: ANN001
                state["calls"] += 1
                state["by_source"][key]["calls"] += 1
                if state["calls"] > max_calls:
                    return real(fields, t)
                before = _fingerprint(fields, family)
                out = real(fields, t)
                if _fingerprint(fields, family) != before:
                    state["wrote"] += 1
                    state["by_source"][key]["wrote"] += 1
                return out
            return wrapped

        try:
            source.inject = make()
        except Exception as exc:  # noqa: BLE001 - a __slots__ source cannot be wrapped
            state["by_source"][key]["not_wrappable"] = repr(exc)
            continue
        restores.append(lambda s=source, r=real: setattr(s, "inject", r))

    def restore() -> None:
        for r in restores:
            r()

    return restore, state


# ---------------------------------------------------------------------------
# The configuration sweep
# ---------------------------------------------------------------------------

def _gaussian(mp, fcen, df, integrated=False):
    return mp.GaussianSource(fcen, fwidth=df, is_integrated=integrated)


BOUNDARIES = {
    # name -> (boundary_layers factory, k_point)
    "pml_all": (lambda mp: [mp.PML(1.0)], None),
    "pml_y_metal_x": (lambda mp: [mp.PML(1.0, direction=mp.Y)], None),
    "pml_x_metal_y": (lambda mp: [mp.PML(1.0, direction=mp.X)], None),
    "no_pml_metal_all": (lambda mp: [], None),
    "pml_y_periodic_x": (lambda mp: [mp.PML(1.0, direction=mp.Y)], "zero_k"),
}

SOURCES = {
    "none": None,
    "electric_Ez": ("Ez", (-2.5, 0.0), False, False),
    "electric_Ez_on_wall": ("Ez", ("wall_x_low", 0.0), False, False),
    "electric_Ez_integrated": ("Ez", (-2.5, 0.0), True, False),
    "magnetic_Hz": ("Hz", (-2.5, 0.0), False, False),
    "magnetic_Hz_on_wall": ("Hz", ("wall_x_low", 0.0), False, False),
    "magnetic_Hx": ("Hx", (-2.5, 0.5), False, False),
    "both_Ez_Hz": ("both", (-2.5, 0.0), False, False),
    "electric_plane_Ez": ("Ez", (-2.5, 0.0), False, True),
}

MATERIALS = {
    "plain": lambda mp: mp.Medium(epsilon=12),
    "conductive": lambda mp: mp.Medium(epsilon=6, D_conductivity=0.4),
    "lorentz": lambda mp: mp.Medium(
        epsilon=2.25,
        E_susceptibilities=[mp.LorentzianSusceptibility(
            frequency=0.4, gamma=0.02, sigma=1.1)]),
}

FOLDS = {"none": None, "mirror_y": "Y"}


def build_case(mp, boundary: str, source: str, material: str, fold: str, res: int):
    cell = mp.Vector3(8, 6, 0)
    layers_of, kpoint = BOUNDARIES[boundary]
    layers = layers_of(mp)
    geometry = [mp.Block(mp.Vector3(2.0, 1.0, mp.inf), center=mp.Vector3(),
                         material=MATERIALS[material](mp))]
    sources = []
    spec = SOURCES[source]
    if spec is not None:
        component, (cx, cy), integrated, plane = spec
        if cx == "wall_x_low":
            cx = -4.0  # exactly on the low-X cell face: a PEC wall when unPMLed
        size = mp.Vector3(0, 4) if plane else mp.Vector3()
        src = _gaussian(mp, 0.3, 0.3, integrated)
        comps = ["Ez", "Hz"] if component == "both" else [component]
        for name in comps:
            sources.append(mp.Source(src, component=getattr(mp, name),
                                     center=mp.Vector3(cx, cy), size=size))
    kwargs: Dict[str, Any] = dict(cell_size=cell, boundary_layers=layers,
                                  geometry=geometry, sources=sources,
                                  resolution=res)
    if kpoint == "zero_k":
        kwargs["k_point"] = mp.Vector3()
    if FOLDS[fold] is not None:
        kwargs["symmetries"] = [mp.Mirror(getattr(mp, FOLDS[fold]))]
    return mp.Simulation(**kwargs), [], 0.0


def case_builder(boundary: str, source: str, material: str, fold: str, res: int):
    def build(mp, r=None):  # noqa: ANN001, ARG001
        return build_case(mp, boundary, source, material, fold, res)
    return build


DISPATCH_ENV = {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
                "MEEP_GPU_FUSE_ARMS": None}
ARRAY_ENV = {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": "0",
             "MEEP_GPU_FUSE_ARMS": None}


def compare(a: Dict[str, numpy.ndarray], b: Dict[str, numpy.ndarray]) -> Dict[str, Any]:
    names = sorted(set(a) | set(b))
    differing: Dict[str, Any] = {}
    words_differing = 0
    for name in names:
        if name not in a or name not in b:
            differing[name] = {"present_on": "A" if name in a else "B"}
            continue
        wa, wb = e2e._as_words(a[name]), e2e._as_words(b[name])
        if wa.shape != wb.shape:
            differing[name] = {"shape": [list(wa.shape), list(wb.shape)]}
            continue
        bad = int(numpy.count_nonzero(wa != wb))
        if bad:
            words_differing += bad
            entry = {"words_differing": bad}
            entry.update(e2e._subnormal_forensics(a[name], b[name]))
            differing[name] = entry
    return {"words_differing": words_differing,
            "arrays_differing": len(differing),
            "detail": dict(sorted(differing.items())[:8])}


def run_case(name: str, builder, gpu_id: int, steps: int,
             counter: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {"case": name, "steps": steps}
    a = b = None
    restore_passes = restore_inject = None
    try:
        a = e2e.run_leg(name, builder, "dispatch", DISPATCH_ENV, counter, gpu_id, None)
        b = e2e.run_leg(name, builder, "array", ARRAY_ENV, counter, gpu_id, None)
        row["grid_shape"] = a["grid_shape"]
        pre = compare(e2e.collect_state(a["driver"]), e2e.collect_state(b["driver"]))
        row["precondition_words_differing"] = pre["words_differing"]
        if pre["words_differing"]:
            row["verdict"] = "HARNESS: lifts differ"
            return row

        restore_passes, pass_state = instrument_in_seam_passes(steps)
        restore_inject, inject_state = instrument_injects(a["driver"])

        e2e.step_leg(name, a, counter, steps)
        restore_inject()
        restore_inject = None
        e2e.step_leg(name, b, counter, steps)
        restore_passes()
        restore_passes = None

        result = compare(e2e.collect_state(a["driver"]),
                         e2e.collect_state(b["driver"]))
        row["words_differing"] = result["words_differing"]
        row["arrays_differing"] = result["arrays_differing"]
        row["detail"] = result["detail"]
        row["in_seam_passes"] = pass_state
        row["injects"] = inject_state
        plans = (getattr(getattr(a["driver"], "_fast_path", None), "step_plan", None)
                 and getattr(getattr(a["driver"], "_fast_path", None),
                             "step_plan").plans or {})
        row["deposit_repairs"] = {
            slot: int(getattr(p_, "repairs", -1))
            for slot, p_ in sorted(plans.items())
            if type(p_).__name__ == "LeadingRepairPlan"}
        plan = a["plan"]
        row["decision"] = plan.get("decision")
        row["refused_because"] = plan.get("refused_because")
        row["arms"] = plan.get("arms")
        row["dispatched_slots"] = plan.get("dispatched_slots")
        row["run_shape"] = plan.get("run_shape")
        row["fusion"] = (a["plan"].get("environment") or {})
        row["active_step_path"] = a["active_step_path"]
        row["triton_launches"] = a["triton_launches"]["total"]
        row["array_leg_launches"] = b["triton_launches"]["total"]
        fused = [slot for slot, arm in (plan.get("arms") or {}).items()
                 if isinstance(arm, str) and "fused pair" in arm]
        row["fused_slots"] = fused
        row["slot_plan_classes"] = {
            slot: type(p).__name__ for slot, p in sorted(
                (getattr(getattr(a["driver"], "_fast_path", None), "step_plan", None)
                 and getattr(getattr(a["driver"], "_fast_path", None),
                             "step_plan").plans or {}).items())}
        # THE HUNT. A pass that WROTE between the two consults of a fused pair,
        # and is not the wall clear the kernel carries.
        if fused:
            seams = {"B" if slot in ("step_B", "update_H") else "D" for slot in fused}
            suspicious = []
            for side in sorted(seams):
                for pass_name in (f"fill_symmetry_bc_{side}",
                                  f"fill_folded_far_ghosts_{side}"):
                    if pass_state.get(pass_name, {}).get("wrote"):
                        suspicious.append(pass_name)
            row["uncovered_live_passes"] = suspicious
        else:
            row["uncovered_live_passes"] = []

        if not fused:
            row["verdict"] = "NO FUSED ARM (not a dispatching configuration)"
        elif row["words_differing"] or row["arrays_differing"]:
            row["verdict"] = "DIVERGED"
        elif row["uncovered_live_passes"]:
            row["verdict"] = ("UNCOVERED LIVE PASS (identical anyway): "
                              + ",".join(row["uncovered_live_passes"]))
        elif a["active_step_path"] != "fused" or not a["triton_launches"]["total"]:
            row["verdict"] = "VACUOUS: dispatch leg launched nothing"
        else:
            row["verdict"] = "IDENTICAL"
    except BaseException as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        row["verdict"] = f"ERROR: {type(exc).__name__}: {exc}"[:500]
        row["traceback"] = traceback.format_exc()[-1500:]
    finally:
        if restore_inject is not None:
            restore_inject()
        if restore_passes is not None:
            restore_passes()
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
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--res", type=int, default=16)
    parser.add_argument("--only", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _PROGRESS = str(out / "progress.log")
    results = out / "cases.jsonl"

    counter = e2e.TritonLaunchCounter()
    counter.install()

    combos = [
        (b, s, m, f)
        for b, s, m, f in itertools.product(BOUNDARIES, SOURCES, MATERIALS, FOLDS)
    ]
    names = [f"{b}|{s}|{m}|fold_{f}" for b, s, m, f in combos]
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        keep = [(n, c) for n, c in zip(names, combos)
                if any(w in n for w in wanted)]
    else:
        keep = list(zip(names, combos))
    if args.limit:
        keep = keep[:args.limit]

    say(f"{len(keep)} configurations; steps={args.steps} res={args.res}")
    tally: Dict[str, int] = {}
    for n, (name, (b, s, m, f)) in enumerate(keep, 1):
        say(f"[{n}/{len(keep)}] {name}")
        started = time.time()
        row = run_case(name, case_builder(b, s, m, f, args.res), args.gpu,
                       args.steps, counter)
        row["wall_s"] = round(time.time() - started, 1)
        row["axes"] = {"boundary": b, "source": s, "material": m, "fold": f}
        with results.open("a") as h:
            h.write(json.dumps(row) + "\n")
            h.flush()
        key = row["verdict"].split(":")[0].split("(")[0].strip()
        tally[key] = tally.get(key, 0) + 1
        say(f"[{n}/{len(keep)}] {name}: {row['verdict']} ({row['wall_s']} s)")
        say(f"    running: {tally}")

    (out / "summary.json").write_text(json.dumps(
        {"tally": tally, "cases": len(keep), "steps": args.steps,
         "res": args.res}, indent=1))
    say(f"SUMMARY {tally}")
    bad = sum(v for k, v in tally.items()
              if k.startswith(("DIVERGED", "UNCOVERED", "VACUOUS", "ERROR", "HARNESS")))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
