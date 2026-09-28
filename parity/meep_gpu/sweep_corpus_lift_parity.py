"""Lift and field-compare MEEP's whole example corpus, one script at a time.

Three numbers about this corpus are routinely collapsed into one, and they are not
the same number:

* **accepted** — ``gpu_compatibility(sim)`` returned supported. A pre-flight verdict
  read off declarations. This is what ``survey_meep_examples.py`` measures, and it is
  the only one of the three that has ever been measured corpus-wide. The lift stage
  asks the gate again and writes its answer as ``accepted``; the survey's older answer
  rides along as ``accepted_by_survey``. **The two are not interchangeable** — the gate
  moves, they already disagree in both directions, and only ``accepted`` is a fact
  about this run. Quote that one; the progress line prints it with the survey's beside
  it when they differ.
* **lifts** — ``lift_simulation(sim)`` actually built an ``FdtdDriver``. Strictly
  harder: the lift initializes MEEP's grid, rasterizes the geometry, reads the
  permittivity back per Yee point and checks it, and every one of those can refuse a
  simulation the declaration analysis accepted.
* **parity-measured** — both the lifted driver and CPU MEEP stepped the SAME
  ``mp.Simulation`` for a bounded time and the fields were compared.

This sweep measures the second and third. Stage ``lift`` runs over every script that
builds a Simulation — including the ones the survey REFUSED, because confirming a
refusal still holds at lift time is free information. Stage ``parity`` runs over
everything the lift stage got a driver from.

**The capture is the survey's, not a second one.** This module execs
``survey_meep_examples.CHILD_PREAMBLE`` and calls its ``_capture``, so the object
stepped here is the object that survey scored, with the script's own geometry,
sources, boundaries and monitors attached at the moment it asked to run. The one
extra move is ``restore()``: the survey leaves ``Simulation.run`` and ``init_sim``
stubbed, and this sweep needs both of them real.

**Cost is capped, and every cap is recorded.** The corpus spans a 0-D dispersion
sweep and a 3-D photonic crystal at resolution 100; run as written, several scripts
are hours each. Two knobs, both written into every row: ``resolution`` is reduced
until the estimated grid fits ``--cell-cap``, and the step count is bounded by
``--max-steps``. A script whose grid does not fit the cap even at ``--res-floor`` is
recorded ``TOO-EXPENSIVE`` and not run — an honest outcome, neither a pass nor a
failure. **A parity number from a capped row is a number about the capped run**, and
the row says so; nothing here may be quoted as if it came from the script as written.

**The run must reach the source, and that is not a free choice.** The first version
of this sweep ran a flat 200 steps and reported disagreements of 1e-01 on four
scripts. They were artifacts of the sweep: a MEEP Gaussian at ``fwidth=0.1`` peaks at
t = 50, so a 200-step run holds only the source's exponential turn-on tail, and the
relative L2 there measures how two codes round a number near the storage floor.
Measured on ``cyl-ellipsoid.py`` at resolution 30 — 1.25e-01 at 200 steps, 1.11e-01
at 2000, **1.02e-05 at 6000**. ``until`` is therefore ``max(--steps, the steps needed
to reach the earliest source's peak)``, bounded by ``--max-steps``; a run that hits
the ceiling first is recorded ``signal_reached=False`` and its number is about the
turn-on tail, not the stepper. Read that field before quoting any row.

What a parity row does NOT claim: that the script's own published result is
reproduced. Both sides step the same captured object for the same short time and
their fields are compared. Scripts captured at ``run_k_points`` or ``solve_cw`` are
stepped with a plain ``run(until=...)`` like every other row.

Rule 7: the parent prints one flushed line per script and appends its JSONL row as it
lands; each child appends its own phase markers to a shared progress log, so a run
that is inside a slow lift is distinguishable from a hung one by ``tail`` alone.

Usage (from ``the repository root``)::

    python -u -m parity.meep_gpu.sweep_corpus_lift_parity \\
        --stage lift --out parity/meep_gpu/results/corpus_lift_parity
    python -u -m parity.meep_gpu.sweep_corpus_lift_parity \\
        --stage parity --out parity/meep_gpu/results/corpus_lift_parity

**The lifted driver is the NUMPY REFERENCE unless ``--dispatch`` is given.** The
columns this sweep has published were measured on a ``prefer_gpu=False`` lift, and
that is what the default still lifts: the NumPy reference, host arrays and the array
path, which consults no kernel table on any host whatever the environment says. The
parent also hands every child ``MEEP_GPU_DISPATCH=0``; on a reference lift that pin
decides nothing and is kept so ``dispatch_enable`` reads the same in every default
row.

``--dispatch`` measures the shipped GPU route instead: the children lift
``prefer_gpu=True`` -- this host's GPU, CUDA through CuPy or Metal kernels over
NumPy host arrays on an Apple GPU -- with ``MEEP_GPU_DISPATCH`` removed, so kernels
dispatch wherever this host certifies them. Removing the variable alone does NOT
dispatch: a ``prefer_gpu=False`` lift never plans. The parent refuses ``--dispatch``
on a host with no GPU route before any child is spawned.

**Every row says what stepped it, read off the driver and not off the flags**:
``prefer_gpu`` (what the lift asked for), ``driver_gpu`` (``"cuda"``, ``"metal"`` or
``None`` for the reference) and, on a parity row, ``step_path`` (``"fused"`` or
``"array"``) with ``dispatch`` (the freeze record's ``reference_driver``,
``decision``, ``table``, the slots that dispatched and the refusal reason). The
parent stamps ``dispatch_enable`` beside them. A row without ``driver_gpu`` predates
these fields, and its step path cannot be read back from it.

``--one`` is the child mode and is not meant to be called by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from parity.meep_gpu.survey_meep_examples import CHILD_PREAMBLE, _as_text, scratch_workdir

# MEEP's Cartesian component names, and the cylindrical family they map onto. The
# driver spells a Dcyl run's components in its own Cartesian names (from_meep.py:463
# — mp.Er -> "Ex", mp.Ep -> "Ey"), so the key here is always the driver's name and
# the value is the mp.* constant to ask CPU MEEP for.
CARTESIAN_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

# An import-time failure inside meep_gpu is not a result about a script. The tree is
# under concurrent edit; a child that dies this way is retried rather than recorded.
IMPORT_FAILURE_MARKERS = (
    "SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError",
    "NameError: name", "AttributeError: partially initialized",
)


def looks_like_import_failure(stderr_text: str) -> bool:
    """True when a child died importing ``meep_gpu`` rather than measuring anything.

    Deliberately narrow: the marker must appear in a traceback frame that names a
    ``meep_gpu/`` file, so a script's own ``ModuleNotFoundError`` (the corpus has
    several — ``gdspy``, ``nlopt``, ``PyMieScatt``) is still a real result.
    """
    if "meep_gpu/" not in stderr_text:
        return False
    return any(marker in stderr_text for marker in IMPORT_FAILURE_MARKERS)


# --- child: capture one script, then lift it (and optionally step it both ways) ------


def capture_simulation(script: str):
    """The survey's own capture, reused verbatim: ``(record, sim, restore)``."""
    namespace: dict = {}
    exec(compile(CHILD_PREAMBLE, "<survey-child>", "exec"), namespace)  # noqa: S102
    return namespace["_capture"](script)


class ProgressLog:
    """Append-only phase markers, flushed, shared by every child of a run."""

    def __init__(self, path: str | None, script: str) -> None:
        self.path = path
        self.script = script
        self.started = time.time()

    def __call__(self, message: str) -> None:
        if not self.path:
            return
        line = f"[{time.strftime('%H:%M:%S')}] {self.script:<40} +{time.time() - self.started:6.1f}s  {message}\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def source_signal_time(mp, sim) -> float:
    """The simulation time by which at least one source is actually radiating.

    A fixed step budget is not enough on its own, and measuring that cost this sweep
    its first set of numbers. MEEP's narrow-band Gaussians peak LATE — ``fwidth=0.1``
    puts the peak at t = 50 — and before the peak the cell holds only the source's
    exponential turn-on tail. Comparing two codes there compares their rounding of a
    number near the storage floor, not their stepping: measured on
    ``cyl-ellipsoid.py`` at resolution 30, the relative L2 reads 1.25e-01 at 200
    steps, 1.11e-01 at 2000 and **1.02e-05 at 6000**, the point at which the pulse
    itself finally exists. The 200-step number was an artifact of this sweep, and
    would have been published as an engine defect.

    Returns MEEP's own peak time for the earliest-peaking source: ``start_time +
    width * cutoff`` for a Gaussian (which is how ``mp.GaussianSource`` builds the
    C++ ``gaussian_src_time`` window), a few periods for a continuous source, and the
    start time for anything whose envelope this cannot know (``CustomSource``).
    """
    times: list[float] = []
    for source in (getattr(sim, "sources", ()) or ()):
        src = getattr(source, "src", None)
        if src is None:
            continue
        start = float(getattr(src, "start_time", 0.0) or 0.0)
        if not math.isfinite(start):
            start = 0.0
        if isinstance(src, mp.GaussianSource):
            width = float(getattr(src, "width", 0.0) or 0.0)
            cutoff = float(getattr(src, "cutoff", 5.0) or 5.0)
            times.append(start + width * cutoff)
        elif isinstance(src, mp.ContinuousSource):
            frequency = abs(complex(getattr(src, "frequency", 1.0) or 1.0))
            width = float(getattr(src, "width", 0.0) or 0.0)
            times.append(start + max(3.0 * width, 3.0 / frequency if frequency else 3.0))
        else:
            times.append(start)
    finite = [value for value in times if math.isfinite(value)]
    return min(finite) if finite else 0.0


def plan_run(mp, sim, facts: dict, cell_cap: int, steps: int, res_floor: float,
             force_resolution: float | None = None, max_steps: int = 12_000) -> dict:
    """Choose the resolution and stopping time for a bounded parity run.

    Returns the plan AND its justification, because a parity number is only readable
    beside the cap that produced it. ``resolution_used < resolution_original`` means
    the row is about a coarser grid than the script's; ``too_expensive`` means the
    grid does not fit even at the floor and nothing was run.
    """
    cell = list(facts.get("cell_size") or [0.0, 0.0, 0.0])
    extents = [abs(float(value)) for value in cell]
    original = float(sim.resolution)
    active = [value for value in extents if value > 0.0]
    ndim = len(active) or 1

    def cells_at(resolution: float) -> int:
        total = 1
        for value in extents:
            total *= max(1, int(round(value * resolution)))
        return total

    # Mirror folding takes BOTH cell-count parities now (grid.py folds an odd count
    # exactly as MEEP's halve() does, window shift included), so the capping no
    # longer needs to steer toward even counts. The folded axes and the
    # ``even_fold_satisfied`` flag are still recorded — a row's parity is grid
    # geometry a reader may want — but they constrain nothing.
    folded_axes: list[int] = []
    for symmetry in (getattr(sim, "symmetries", ()) or ()):
        direction = getattr(symmetry, "direction", None)
        if direction is None:
            continue
        try:
            axis = int(direction)
        except Exception:
            continue
        if 0 <= axis <= 2 and extents[axis] > 0.0:
            folded_axes.append(axis)

    def even_on_folded(resolution: float) -> bool:
        return all(int(round(extents[axis] * resolution)) % 2 == 0 for axis in folded_axes)

    n_original = cells_at(original)
    plan = {
        "resolution_original": original,
        "cells_estimated_original": n_original,
        "cell_cap": cell_cap,
        "step_budget": steps,
        "folded_axes": sorted(set(folded_axes)),
    }
    if force_resolution is not None:
        # A stated resolution, cap and floor ignored. This exists because the sweep
        # measured a parity that DEPENDS on the grid — the same script agrees at
        # 1e-06 on one resolution and disagrees at 1e-01 on another — and pinning
        # that down needs the resolution as an axis rather than a side effect of the
        # cell cap. Rows produced this way say so.
        plan.update({
            "too_expensive": False,
            "resolution_used": float(force_resolution),
            "cells_estimated_used": cells_at(float(force_resolution)),
            "resolution_capped": float(force_resolution) != original,
            "resolution_forced": True,
            "even_fold_satisfied": even_on_folded(float(force_resolution)),
        })
        plan.update(_timing(mp, sim, float(force_resolution), steps, max_steps))
        return plan
    if cells_at(res_floor) > cell_cap:
        plan.update({
            "too_expensive": True,
            "resolution_used": None,
            "cells_estimated_used": cells_at(res_floor),
            "res_floor": res_floor,
            "until": None,
        })
        return plan

    target = original
    if n_original > cell_cap:
        scale = (cell_cap / n_original) ** (1.0 / ndim)
        target = max(res_floor, math.floor(original * scale))
    # Walk down from the target to the first resolution that fits the cap (the
    # rounding in cells_at can leave the scaled target a hair over it). The old
    # walk also required an even count on every folded axis; that accommodation is
    # retired — an odd folded count lifts now.
    chosen = None
    candidate = float(int(target)) if target >= 1 else target
    while candidate >= res_floor and (candidate > target - 40):
        if cells_at(candidate) <= cell_cap:
            chosen = candidate
            break
        candidate -= 1.0
    if chosen is None:
        chosen = min(target, original)
    plan.update({
        "too_expensive": False,
        "resolution_used": float(chosen),
        "cells_estimated_used": cells_at(chosen),
        "resolution_capped": float(chosen) != original,
        "even_fold_satisfied": even_on_folded(chosen),
    })
    plan.update(_timing(mp, sim, float(chosen), steps, max_steps))
    return plan


def _timing(mp, sim, resolution: float, steps: int, max_steps: int) -> dict:
    """How long to run: at least ``steps``, and far enough for the source to radiate.

    The step count is what bounds cost, so it is still a ceiling (``max_steps``). A run
    that hits the ceiling before the source peaks is recorded ``signal_reached=False``
    and its parity number is about the source's turn-on tail, not the stepper.
    """
    dt = float(sim.Courant) / resolution
    signal_time = source_signal_time(mp, sim)
    wanted = max(int(steps), int(math.ceil(signal_time / dt)) if dt > 0 else int(steps))
    used = min(wanted, int(max_steps))
    return {
        "source_signal_time": signal_time,
        "steps_to_reach_signal": int(math.ceil(signal_time / dt)) if dt > 0 else None,
        "steps_planned": used,
        "max_steps": int(max_steps),
        "signal_reached": bool(used * dt >= signal_time),
        # HALF A STEP SHORT OF THE BOUNDARY, deliberately. MEEP steps while
        # `round_time() < until` and this driver steps `ceil((until - t)/dt - tol)`
        # times; at `until = N*dt` exactly, whether N*dt lands a bit above or below
        # MEEP's accumulated time is a property of dt's binary representation, so the
        # two take N and N+1 steps on some resolutions and N and N on others. That is
        # a whole class of false disagreement — measured on `oblique-planewave.py` at
        # resolution 29, MEEP reached t=3.465517 against the driver's t=3.448276,
        # exactly one dt, and the fields read 1.12e-01 apart while agreeing at
        # 1.3e-06 on the resolutions where the rounding happened to match. At
        # `(N - 0.5)*dt` both codes take N steps for every dt. `time_matched` on each
        # row is the guard that caught this and stays as the guard.
        "until": (used - 0.5) * dt,
    }


def compare_fields(mp, numpy, result, sim, cylindrical: bool) -> dict:
    """Complex relative L2 over the whole volume, in MEEP's own ``get_array`` layout.

    Every component is stacked, INCLUDING the ones MEEP returns identically zero:
    dropping those would hide the failure mode where the lifted run puts energy in a
    component MEEP leaves empty. Per-component numbers are kept beside the headline
    so a single bad polarization is visible rather than diluted.
    """
    if cylindrical:
        wanted = {"Ex": mp.Er, "Ey": mp.Ep, "Ez": mp.Ez, "Hx": mp.Hr, "Hy": mp.Hp, "Hz": mp.Hz}
    else:
        wanted = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez, "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz}
    per_component: dict = {}
    shapes: dict = {}
    ours_stack, theirs_stack = [], []
    for name in CARTESIAN_COMPONENTS:
        try:
            ours = numpy.asarray(result.get_array(name), dtype=numpy.complex128)
        except BaseException as exc:  # noqa: BLE001 - a component that cannot be read is a result.
            per_component[name] = f"lift-side error: {type(exc).__name__}: {exc}"[:300]
            continue
        try:
            theirs = numpy.asarray(sim.get_array(component=wanted[name]), dtype=numpy.complex128)
        except BaseException as exc:  # noqa: BLE001
            per_component[name] = f"meep-side error: {type(exc).__name__}: {exc}"[:300]
            continue
        shapes[name] = [list(ours.shape), list(theirs.shape)]
        if ours.shape != theirs.shape:
            per_component[name] = f"SHAPE MISMATCH {ours.shape} vs {theirs.shape}"
            continue
        ours_stack.append(ours.ravel())
        theirs_stack.append(theirs.ravel())
        reference_norm = float(numpy.linalg.norm(theirs))
        if reference_norm == 0.0:
            per_component[name] = {
                "meep_reference": "identically zero",
                "lifted_norm": float(numpy.linalg.norm(ours)),
            }
        else:
            per_component[name] = float(numpy.linalg.norm(ours - theirs) / reference_norm)
    out = {"per_component": per_component, "shapes": shapes,
           "components_compared": len(ours_stack)}
    if not ours_stack:
        out["parity_rel_l2"] = None
        out["parity_note"] = "no component could be compared"
        return out
    ours_all = numpy.concatenate(ours_stack)
    theirs_all = numpy.concatenate(theirs_stack)
    denominator = float(numpy.linalg.norm(theirs_all))
    if denominator == 0.0:
        out["parity_rel_l2"] = None
        out["parity_note"] = "CPU MEEP's whole volume is identically zero; no ratio exists"
        out["lifted_norm"] = float(numpy.linalg.norm(ours_all))
        return out
    out["parity_rel_l2"] = float(numpy.linalg.norm(ours_all - theirs_all) / denominator)
    out["meep_volume_norm"] = denominator
    return out


def dispatch_facts(driver) -> dict:
    """What stepped this driver, read off the driver's own freeze record.

    ``step_path`` is ``driver.active_step_path`` and the rest is the record the
    freeze published, so a row can be told apart from one lifted the other way
    without reading the command line that made it.
    """
    report = driver.fast_path_report() or {}
    slots = report.get("slots") or {}
    return {
        "step_path": driver.active_step_path,
        "dispatch": {
            "reference_driver": report.get("reference_driver"),
            "decision": report.get("decision"),
            "table": report.get("table"),
            "dispatched_slots": sorted(name for name, entry in slots.items()
                                       if entry.get("state") == "dispatched"),
            "slots": len(slots),
            "refused_because": report.get("refused_because"),
        },
    }


def child_lift(script: str, out_json: str, progress: ProgressLog, *,
               prefer_gpu: bool = False) -> None:
    """Capture, ask the gate, then actually build the driver. Nothing is stepped."""
    record, sim, restore = capture_simulation(script)
    record["stage"] = "lift"
    record["prefer_gpu"] = bool(prefer_gpu)
    progress(f"captured: outcome={record.get('outcome')} has_sim={record.get('has_simulation')}")
    if sim is None:
        record["lifts"] = None
        record["lift_note"] = "the script never built an mp.Simulation to lift"
        Path(out_json).write_text(json.dumps(record), encoding="utf-8")
        return
    restore()  # run/init_sim are stubs until here, and the lift calls init_sim.
    import meep_gpu

    try:
        verdict = meep_gpu.gpu_compatibility(sim)
        record["accepted"] = bool(verdict.supported)
        record["accept_reasons"] = list(verdict.reasons)
    except BaseException as exc:  # noqa: BLE001
        record["accepted"] = None
        record["accept_error"] = f"{type(exc).__name__}: {exc}"[:600]
    progress(f"gate: accepted={record.get('accepted')}")
    started = time.time()
    try:
        driver = meep_gpu.lift_simulation(
            sim, prefer_gpu=prefer_gpu,
            progress_cb=lambda done, total: (
                progress(f"lift sampling {done}/{total}") if total and done % max(1, total // 4) == 0 else None
            ),
        )
        record["lifts"] = True
        # Nothing is stepped here, so no configuration freezes and there is no step
        # path to record; which engine the lift resolved is a fact already.
        record["driver_gpu"] = driver.gpu
        record["grid_shape"] = [int(value) for value in driver.shape]
        record["grid_cells"] = int(math.prod(int(value) for value in driver.shape))
        record["sigma_lift"] = repr(getattr(driver, "sigma_lift", None))[:200]
        driver.close()
    except BaseException as exc:  # noqa: BLE001 - a failed lift is the headline result.
        record["lifts"] = False
        record["lift_error_type"] = type(exc).__name__
        record["lift_error"] = str(exc)[:1500]
    record["lift_s"] = round(time.time() - started, 2)
    progress(f"lift: lifts={record.get('lifts')} driver_gpu={record.get('driver_gpu')!r} "
             f"({record['lift_s']} s)")
    Path(out_json).write_text(json.dumps(record), encoding="utf-8")


def child_parity(script: str, out_json: str, progress: ProgressLog, *,
                 cell_cap: int, steps: int, res_floor: float,
                 force_resolution: float | None = None, max_steps: int = 12_000,
                 prefer_gpu: bool = False) -> None:
    """Step the lifted driver and CPU MEEP over the same object and compare fields."""
    record, sim, restore = capture_simulation(script)
    record["stage"] = "parity"
    record["prefer_gpu"] = bool(prefer_gpu)
    progress(f"captured: outcome={record.get('outcome')}")
    if sim is None:
        record["parity"] = "NO-SIMULATION"
        Path(out_json).write_text(json.dumps(record), encoding="utf-8")
        return
    restore()
    import meep as mp
    import numpy
    import meep_gpu

    try:
        mp.verbosity(0)
    except Exception:
        pass
    facts = record.get("facts") or {}
    plan = plan_run(mp, sim, facts, cell_cap, steps, res_floor, force_resolution, max_steps)
    record["capped"] = plan
    if plan["too_expensive"]:
        record["parity"] = "TOO-EXPENSIVE"
        record["parity_rel_l2"] = None
        progress(f"TOO-EXPENSIVE: {plan['cells_estimated_used']} cells at res_floor={res_floor}")
        Path(out_json).write_text(json.dumps(record), encoding="utf-8")
        return
    already_initialized = getattr(sim, "fields", None) is not None
    plan["already_initialized"] = already_initialized
    if plan.get("resolution_capped") and not already_initialized:
        sim.resolution = plan["resolution_used"]
    elif plan.get("resolution_capped"):
        # MEEP built the grid already, so the resolution field is decorative now.
        plan["resolution_used"] = plan["resolution_original"]
        plan["resolution_capped"] = False
        plan["cap_not_applied"] = "the script had already initialized MEEP's grid"
        plan.update(_timing(mp, sim, float(plan["resolution_original"]), steps, max_steps))
    until = float(plan["until"])
    cylindrical = (facts.get("dimensions_attr") == -2) or bool(facts.get("is_cylindrical"))
    record["cylindrical"] = cylindrical
    progress(f"plan: res {plan['resolution_original']} -> {plan['resolution_used']}, "
             f"~{plan['cells_estimated_used']} cells, until={until:.4g} "
             f"({plan['steps_planned']} steps, source peaks at t={plan['source_signal_time']:.4g}, "
             f"signal_reached={plan['signal_reached']})")

    started = time.time()
    try:
        result = meep_gpu.run_on_gpu(
            sim, until=until, prefer_gpu=prefer_gpu,
            lift_progress_cb=lambda done, total: (
                progress(f"lift sampling {done}/{total}") if total and done % max(1, total // 4) == 0 else None
            ),
            progress_cb=lambda done, total: (
                progress(f"stepping {done}/{total}") if total and done % max(1, total // 4) == 0 else None
            ),
        )
    except BaseException as exc:  # noqa: BLE001
        record["parity"] = "LIFT-OR-STEP-FAILED"
        record["parity_rel_l2"] = None
        record["parity_error_type"] = type(exc).__name__
        record["parity_error"] = str(exc)[:1500]
        record["lifted_step_s"] = round(time.time() - started, 2)
        progress(f"FAILED (lift/step): {type(exc).__name__}")
        Path(out_json).write_text(json.dumps(record), encoding="utf-8")
        return
    record["lifted_step_s"] = round(time.time() - started, 2)
    record["driver_steps"] = int(result.steps)
    record["driver_meep_time"] = float(result.meep_time)
    record["grid_shape"] = [int(value) for value in result.driver.shape]
    record["grid_cells"] = int(math.prod(int(value) for value in result.driver.shape))
    # WHAT STEPPED IT, read off the driver before anything else can refreeze it.
    record["driver_gpu"] = result.driver.gpu
    record.update(dispatch_facts(result.driver))
    progress(f"lifted run done: {result.steps} steps, {record['lifted_step_s']} s, "
             f"driver_gpu={record['driver_gpu']!r} step_path={record['step_path']} "
             f"table={record['dispatch']['table']!r} "
             f"slots {len(record['dispatch']['dispatched_slots'])}/{record['dispatch']['slots']}")

    started = time.time()
    try:
        sim.run(until=until)  # THE SAME OBJECT, now stepped by CPU MEEP.
    except BaseException as exc:  # noqa: BLE001
        record["parity"] = "CPU-MEEP-FAILED"
        record["parity_rel_l2"] = None
        record["parity_error_type"] = type(exc).__name__
        record["parity_error"] = str(exc)[:1500]
        result.close()
        progress(f"FAILED (CPU MEEP): {type(exc).__name__}")
        Path(out_json).write_text(json.dumps(record), encoding="utf-8")
        return
    record["cpu_meep_s"] = round(time.time() - started, 2)
    record["cpu_meep_time"] = float(sim.round_time())
    # A step-count disagreement makes any L2 meaningless, so it is recorded rather
    # than left for the number to absorb — this is the guard that caught the
    # step-boundary ambiguity `_timing` now avoids. Compared as INTEGER step counts
    # off MEEP's own `fields.t`, not as times: `round_time()` is accumulated in
    # float32, so two runs that took the identical 200 steps report times differing
    # at 5.8e-08 and a time-based check calls them mismatched.
    record["cpu_meep_steps"] = int(sim.fields.t)
    record["time_matched"] = bool(record["cpu_meep_steps"] == record["driver_steps"])
    progress(f"CPU MEEP done: {record['cpu_meep_steps']} steps vs driver "
             f"{record['driver_steps']} ({record['cpu_meep_s']} s)")

    try:
        record.update(compare_fields(mp, numpy, result, sim, cylindrical))
        record["parity"] = "MEASURED" if record.get("parity_rel_l2") is not None else "NO-COMPARABLE-FIELD"
    except BaseException as exc:  # noqa: BLE001
        record["parity"] = "COMPARE-FAILED"
        record["parity_rel_l2"] = None
        record["parity_error_type"] = type(exc).__name__
        record["parity_error"] = str(exc)[:1500]
    finally:
        result.close()
    progress(f"parity={record.get('parity')} rel_l2={record.get('parity_rel_l2')}")
    Path(out_json).write_text(json.dumps(record), encoding="utf-8")


# --- parent: one subprocess per script, partial results as they land -----------------


def load_survey(path: Path) -> dict:
    """``{script: accepted}`` from a survey JSONL, for scripts that built a Simulation."""
    rows: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("has_simulation"):
            rows[row["script"]] = row.get("supported")
    return rows


_ACCEPT_NAMES = {True: "accepted", False: "refused", None: "unknown"}


def accept_flag(record: dict) -> str:
    """Render the gate verdict for the progress line: this run's, then the survey's.

    ``accepted`` is what ``gpu_compatibility(sim)`` returned in THIS run;
    ``accepted_by_survey`` is the verdict copied from the survey JSONL, which may
    predate the current gate — the two already disagree on several scripts, in both
    directions. Printing the survey's flag alone reports a stale gate as if it were
    live, so the live verdict leads and a survey it has overtaken is shown beside it,
    labelled. The parity stage never asks the gate, so there the survey's verdict is
    all there is and the label says so.
    """
    surveyed = _ACCEPT_NAMES.get(record.get("accepted_by_survey"), "unknown")
    if "accepted" not in record:  # the parity stage, or a child that died before the gate
        return f"{'survey: ' + surveyed:<26}"
    live = _ACCEPT_NAMES.get(record.get("accepted"), "unknown")
    text = live if live == surveyed else f"{live} (survey: {surveyed})"
    return f"{text:<26}"


def already_done(jsonl: Path) -> dict:
    if not jsonl.exists():
        return {}
    out = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["script"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=os.path.join(os.environ.get("MGPU_SITE_MEEP_SOURCE", os.path.join(os.path.expanduser("~"), "meep")), "python", "examples"))
    parser.add_argument("--survey", default="parity/meep_gpu/results/survey_recovery_stock/survey.jsonl")
    parser.add_argument("--out", default="parity/meep_gpu/results/corpus_lift_parity")
    parser.add_argument("--stage", choices=("lift", "parity"), default="lift")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cell-cap", type=int, default=120_000,
                        help="parity stage: largest estimated grid to run, in cells")
    parser.add_argument("--steps", type=int, default=200,
                        help="parity stage: time steps to run (until = steps * Courant / resolution)")
    parser.add_argument("--res-floor", type=float, default=4.0,
                        help="parity stage: resolution below which a script is TOO-EXPENSIVE instead")
    parser.add_argument("--max-steps", type=int, default=12_000,
                        help="parity stage: ceiling on the step count (cost bound)")
    parser.add_argument("--force-resolution", type=float, default=None,
                        help="parity stage: run at this resolution, ignoring the cap and the floor")
    parser.add_argument("--only", default=None, help="comma-separated script names")
    parser.add_argument("--resume", action="store_true", help="skip scripts already in the JSONL")
    parser.add_argument("--retries", type=int, default=3,
                        help="retries for a child that died importing meep_gpu (the tree is under edit)")
    parser.add_argument("--retry-wait", type=float, default=45.0)
    parser.add_argument("--dispatch", action="store_true",
                        help="measure the shipped GPU route: lift prefer_gpu=True "
                             "(this host's GPU) with MEEP_GPU_DISPATCH unset, so "
                             "kernels dispatch wherever this host certifies them. "
                             "The default lifts the NumPy reference "
                             "(prefer_gpu=False), which never dispatches and is "
                             "what the columns have always meant. Refused on a "
                             "host with no GPU route")
    # Child mode.
    parser.add_argument("--one", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--progress-log", default=None)
    args = parser.parse_args()

    if args.one:
        progress = ProgressLog(args.progress_log, Path(args.one).name)
        if args.stage == "lift":
            child_lift(args.one, args.out_json, progress, prefer_gpu=args.dispatch)
        else:
            child_parity(args.one, args.out_json, progress, cell_cap=args.cell_cap,
                         steps=args.steps, res_floor=args.res_floor,
                         force_resolution=args.force_resolution, max_steps=args.max_steps,
                         prefer_gpu=args.dispatch)
        return 0

    host_gpu = None
    if args.dispatch:
        # A HOST FACT, ANSWERED BEFORE ANYTHING IS WRITTEN: prefer_gpu=True raises on a
        # host with no GPU route, and every child would record that as its own
        # LIFT-FAILED row, a sweep of one host fact repeated per script.
        from meep_gpu import backends  # noqa: PLC0415

        host_gpu = backends.available_gpu()
        if host_gpu is None:
            print("REFUSING: --dispatch lifts prefer_gpu=True and this host has no GPU "
                  "route (meep_gpu.backends.available_gpu() is None; missing: "
                  f"{', '.join(backends.missing_dependencies())}). Run without "
                  "--dispatch for the NumPy reference.", file=sys.stderr, flush=True)
            return 2

    examples = Path(args.examples)
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    per_script = out_root / f"per_script_{args.stage}"
    per_script.mkdir(exist_ok=True)
    jsonl = out_root / f"{args.stage}.jsonl"
    log_path = out_root / f"{args.stage}.log"
    progress_log = out_root / f"{args.stage}.progress.log"
    work = scratch_workdir(examples, out_root)

    survey = load_survey(Path(args.survey).resolve())
    if args.stage == "lift":
        names = sorted(survey)
    else:
        lift_rows = already_done(out_root / "lift.jsonl")
        if not lift_rows:
            print("parity stage needs lift.jsonl; run --stage lift first", flush=True)
            return 2
        names = sorted(name for name, row in lift_rows.items() if row.get("lifts") is True)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        names = [name for name in names if name in wanted]
    done = already_done(jsonl) if args.resume else {}
    if not args.resume:
        jsonl.write_text("", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")
        progress_log.write_text("", encoding="utf-8")
    names = [name for name in names if name not in done]

    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    # THE ENABLE EVERY CHILD GETS. What the child LIFTS decides whether anything can
    # dispatch (the default is the NumPy reference, which never plans); the variable
    # is a veto on a GPU driver only. --dispatch removes it so the prefer_gpu=True
    # children measure the shipped default; the default keeps the 0 the columns have
    # carried, redundant on a reference lift.
    if args.dispatch:
        environment.pop("MEEP_GPU_DISPATCH", None)
    else:
        environment["MEEP_GPU_DISPATCH"] = "0"
    dispatch_enable = environment.get("MEEP_GPU_DISPATCH")

    def say(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()

    say(f"stage {args.stage}: {len(names)} scripts (skipping {len(done)} already recorded)")
    say("dispatch: " + (f"--dispatch: children lift prefer_gpu=True (this host's GPU: "
                        f"{host_gpu}) with MEEP_GPU_DISPATCH unset, the shipped "
                        "default, kernels wherever this host certifies them"
                        if args.dispatch else
                        "children lift prefer_gpu=False, the NumPy reference (no "
                        "kernel table consulted), with MEEP_GPU_DISPATCH=0: the array "
                        "path, the columns' historical meaning"))
    if args.stage == "parity":
        say(f"caps: <= {args.cell_cap} cells, {args.steps} steps, resolution floor {args.res_floor}, "
            f"per-script timeout {args.timeout:.0f} s")
    run_started = time.time()
    for index, name in enumerate(names, start=1):
        script = examples / name
        record_path = per_script / f"{script.stem}.json"
        if record_path.exists():
            record_path.unlink()
        case_start = time.time()
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--one", str(script), "--out-json", str(record_path), "--stage", args.stage,
            "--progress-log", str(progress_log), "--cell-cap", str(args.cell_cap),
            "--steps", str(args.steps), "--res-floor", str(args.res_floor),
            "--max-steps", str(args.max_steps),
        ]
        if args.force_resolution is not None:
            command += ["--force-resolution", str(args.force_resolution)]
        if args.dispatch:
            command += ["--dispatch"]
        record = None
        for attempt in range(args.retries + 1):
            status, stderr_text, returncode = "ok", "", None
            try:
                completed = subprocess.run(
                    command, cwd=str(work), env=environment, timeout=args.timeout,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
                )
                stderr_text = _as_text(completed.stderr)
                returncode = completed.returncode
            except subprocess.TimeoutExpired as expired:
                status = "timeout"
                stderr_text = _as_text(expired.stderr)
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                break
            if status != "timeout" and looks_like_import_failure(stderr_text) and attempt < args.retries:
                # The engine tree is under concurrent edit; a transient half-written
                # module is not a fact about this script. Wait and try again.
                say(f"     {name}: meep_gpu failed to import, retry {attempt + 1}/{args.retries} "
                    f"in {args.retry_wait:.0f} s")
                time.sleep(args.retry_wait)
                continue
            record = {
                "script": name, "stage": args.stage,
                "outcome": "timeout" if status == "timeout" else "child_died",
                "returncode": returncode, "stderr_tail": stderr_text[-2000:],
            }
            if args.stage == "lift":
                record["lifts"] = None
                record["lift_note"] = (
                    f"the lift did not complete within the {args.timeout:.0f} s timeout"
                    if status == "timeout" else "the child process died; see stderr_tail"
                )
            else:
                record["parity"] = "TIMEOUT" if status == "timeout" else "CHILD-DIED"
                record["parity_rel_l2"] = None
            if stderr_text:
                (per_script / f"{script.stem}.stderr.txt").write_text(stderr_text, encoding="utf-8")
            break
        assert record is not None
        record["accepted_by_survey"] = survey.get(name)
        record["dispatch_enable"] = {"MEEP_GPU_DISPATCH": dispatch_enable}
        # A child that died before its lift wrote nothing about the engine; the row
        # still says what the lift was asked for.
        record.setdefault("prefer_gpu", bool(args.dispatch))
        record["wall_s"] = round(time.time() - case_start, 2)
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if args.stage == "lift":
            if record.get("lifts") is True:
                verdict = f"LIFTS   grid {record.get('grid_shape')}"
            elif record.get("lifts") is False:
                verdict = f"LIFT-FAILED {record.get('lift_error_type')}: {str(record.get('lift_error'))[:90]}"
            else:
                verdict = f"NO-RESULT ({record.get('outcome')})"
        else:
            if record.get("parity_rel_l2") is not None:
                verdict = (f"L2 = {record['parity_rel_l2']:.3e}  "
                           f"({record.get('grid_cells')} cells, {record.get('driver_steps')} steps, "
                           f"step path {record.get('step_path')}"
                           f"{' on ' + str((record.get('dispatch') or {}).get('table')) if record.get('step_path') == 'fused' else ''})")
            else:
                verdict = f"{record.get('parity')} {str(record.get('parity_error') or '')[:80]}"
        say(f"  {index:>2}/{len(names)} {name:<42} {accept_flag(record)}  {verdict} "
            f" [{record['wall_s']:.1f} s]")
    say(f"stage {args.stage} complete in {time.time() - run_started:.1f} s -> {jsonl}")
    summarize(jsonl, args.stage, say)
    return 0


def summarize(jsonl: Path, stage: str, say) -> None:
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    say("")
    say(f"rows: {len(rows)}")
    if stage == "lift":
        lifts = [r for r in rows if r.get("lifts") is True]
        failed = [r for r in rows if r.get("lifts") is False]
        unknown = [r for r in rows if r.get("lifts") is None]
        say(f"LIFTS       : {len(lifts)}")
        say(f"LIFT-FAILED : {len(failed)}")
        say(f"no result   : {len(unknown)}")
        # THIS RUN'S gate verdict, not the survey's. `accepted` is what
        # gpu_compatibility(sim) answered here; `accepted_by_survey` is a copy of an
        # older survey's answer and the two disagree — measured on
        # results/corpus_final_numpy_2026-08-06/lift.jsonl: 49 live against 43
        # surveyed over 57 rows, six scripts (disc_extraction_efficiency,
        # disc_radiation_pattern, gaussian-beam, oblique-planewave, oblique-source,
        # zone_plate) accepted live and refused by the survey. Quoting the survey's
        # figure published a gate six scripts behind the one that ran. The live count
        # is stated here so the sweep's own summary is the source for it.
        gated = [r for r in rows if "accepted" in r]
        say(f"gate ACCEPTED (this run): {sum(1 for r in gated if r.get('accepted') is True)}"
            f"  of {len(gated)} rows the gate answered for")
        say(f"gate REFUSED  (this run): {sum(1 for r in gated if r.get('accepted') is False)}")
        stale = [r for r in gated if r.get("accepted") != r.get("accepted_by_survey")]
        if stale:
            say(f"gate has MOVED since the survey on {len(stale)} script(s): "
                + ", ".join(f"{r['script']} ({_ACCEPT_NAMES.get(r.get('accepted_by_survey'))}"
                            f" -> {_ACCEPT_NAMES.get(r.get('accepted'))})" for r in stale))

        def verdict(row: dict):
            """The live verdict where the gate answered here, the survey's otherwise."""
            return row["accepted"] if "accepted" in row else row.get("accepted_by_survey")

        gap = [r for r in rows if verdict(r) is True and r.get("lifts") is not True]
        say(f"accepted by the gate but does NOT lift: {len(gap)}")
        for row in gap:
            say(f"    {row['script']:<42} {row.get('lift_error_type') or row.get('outcome')}: "
                f"{str(row.get('lift_error') or row.get('lift_note'))[:140]}")
        held = [r for r in rows if verdict(r) is False and r.get("lifts") is False]
        say(f"refused by the gate and refused at lift time (refusal holds): {len(held)}")
    else:
        measured = [r for r in rows if r.get("parity_rel_l2") is not None]
        say(f"parity measured : {len(measured)}")
        for row in sorted(measured, key=lambda r: r["parity_rel_l2"]):
            say(f"    {row['script']:<42} {row['parity_rel_l2']:.3e}"
                f"{'  (resolution capped)' if (row.get('capped') or {}).get('resolution_capped') else ''}")
        for label in ("TOO-EXPENSIVE", "TIMEOUT", "LIFT-OR-STEP-FAILED", "CPU-MEEP-FAILED",
                      "COMPARE-FAILED", "NO-COMPARABLE-FIELD", "CHILD-DIED"):
            hit = [r for r in rows if r.get("parity") == label]
            if hit:
                say(f"{label}: {len(hit)}")
                for row in hit:
                    say(f"    {row['script']:<42} {str(row.get('parity_error') or '')[:120]}")


if __name__ == "__main__":
    raise SystemExit(main())
