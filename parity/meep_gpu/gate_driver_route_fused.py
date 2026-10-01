"""THE DRIVER-ROUTE GATE FOR A FUSED PAIR: one launch spanning two of the driver's own consults.

WHAT CLAUSE (8) SAYS AND WHY THIS FILE EXISTS. ``fastpath._decide`` refuses every
cross-sub-step fused product with "a fused cross-sub-step product won {slots}; no
fused arm has been driven through the driver seam". That sentence was
unfalsifiable until 2026-08-29: ``_decide`` passed ``fuse=False`` as a LITERAL, so
no fused product could reach ``FastPathPlan.dispatch`` at all and no measurement
could ever change the refusal's truth value. :data:`fastpath.FUSE_ARMS_SWITCH` is
the opt-in that breaks that loop — default-off, byte-for-byte the old behaviour
when unset — and this gate is the measurement it exists for.

THE ARM. ``fused pair B``: the ordinary unfolded PML real-float32 magnetic pair,
``triton_kernels.launch.FusedPairPlan`` installed by ``_install_fused_pair`` at
``FUSED_PAIR_ARMS["B"] = ("PML", "ordinary")``. One Triton launch performs
``step_B``, the metallic wipe, and ``update_H``; it takes BOTH driver slots, the
leading one holding the pair (or ``deposit_repair.LeadingRepairPlan``) and the
absorbed one holding a ``NoopPlan`` (or ``TrailingRepairPlan``).

It is the only fused arm reachable in this phase. The fold rung (6b) fires AFTER
clause (8) and refuses any mirror-folded grid whole, so every folded pair stays
unreachable until that refusal is narrowed — which is a later phase and is
asserted STANDING here, not worked around.

THE PROTOCOL, per case, in ONE process on ONE device — THREE legs, not two::

    lift A (fused)        lift B (unfused)       lift C (array)
    DISPATCH=1            DISPATCH=1             DISPATCH=1
    FUSE_ARMS=fused pair B                       FUSED=0
         |                     |                      |
         +---- byte-equal? ----+---- byte-equal? -----+   PRECONDITION
         |                     |                      |
      for step in 1..12:  one COMPLETE driver.step() each, then
         +--- A vs C: every array, every word, as uint32 ---+   THE CLAIM
         +--- B vs C: the same, for the unfused baseline ---+   THE CONTROL
         then synchronize_magnetic_fields() on each and compare AGAIN, and
         restore_magnetic_fields() and compare a third time.

THE THIRD LEG IS THE POINT, and it is what the 2026-08-15 driver-route gate could
not have. A/C alone says the fused answer is right. A/B — same dispatcher, same
device, same policy, fusion the only difference — is what makes the LAUNCH COUNT
a substitution proof rather than a number: the B pair costs two Triton launches
per magnetic-half consult unfused and ONE fused, so the drop is exactly one per
consult, counted twice over (by kernel name from Triton's own entry points, and
by slot from the dispatcher's own counters).

THE SECOND CONSULT SITE. ``driver.step`` is not the whole seam:
``synchronize_magnetic_fields`` consults ``step_B`` and ``update_H`` again, with
the same three in-seam array passes between them, reading a plan it deliberately
does not re-build. Any flux or energy monitor reaches it. A gate that drove only
``step()`` would leave half the seam unmeasured, so every step here enters it —
and because ``restore_magnetic_fields`` puts the pre-sync arrays back exactly, the
comparison is taken WHILE SYNCHRONIZED, where a difference is still visible, and
again after the restore.

THE SEAM COMPOSITION NO GATE HAD RUN. The driver runs ``fill_symmetry_bc_B``,
``zero_metal_B`` and ``fill_folded_far_ghosts_B`` UNCONDITIONALLY after the
leading consult. The fused kernel already performed the metallic wipe, so the
array pass runs a second time ON TOP of it. That composition is what every
byte-identical step below actually measures, and the plan's own ``zero_metal``
binding is recorded per case so a reader can see whether the wipe was live.

WHAT MAKES A PASS NON-VACUOUS. Byte identity is half a verdict; a leg that fell
back matches trivially. Each of these is a FAILURE, not a caveat:

* the record's slots show ``fused pair B`` at BOTH ``step_B`` and ``update_H``,
  with ``absorbed_by`` wired on the trailing one;
* ``driver.active_step_path`` reads ``"fused"``;
* the dispatcher's per-slot launch counters are nonzero and
  ``programs_per_dispatch`` is nonzero (a Triton launch at ``grid=(0,)`` counts in
  every counter and computes nothing — that is what the warm pass does on
  purpose, and it is also this gate's null control);
* an independently installed Triton-level counter, on ``JITFunction.run`` and
  ``CompiledKernel.launch_enter_hook``, is nonzero on A and B and ZERO on C;
* the fused kernel appears by NAME in A's launches and NOT in B's, and the two
  separate kernels appear in B's and not in A's;
* the launch drop A vs B is exactly one per magnetic-half consult.

THE NULL CONTROL. "The same run with the fused route withheld MUST diverge." The
route is withheld the way the warm pass withholds one: the fused plan's launch
grid is emptied to ``(0,)`` after the freeze, so the kernel still LAUNCHES — every
counter still counts it, the record still reads dispatched — and computes
nothing. If the comparator still calls that byte-identical, then nothing in the
compared state depends on the fused kernel's output and every green row above is
worthless. It is an assertion, not a diagnostic.

POLICY. ``fastpath`` rung 8b installs ``CERTIFICATION_SUBNORMAL_POLICY`` ("keep")
itself and refuses any process that installed anything else, so the dispatch route
supports exactly one policy and the honest second leg is to MEASURE that: a
``flush`` run must be REFUSED BY NAME at rung 8b, never silently dispatched. The
harness installs nothing — which is precisely the missing evidence
``fastpath.DISPATCH_BY_DEFAULT`` names.

Progress reporting: one flushed line per case per leg, and a JSONL row appended as each case
lands, so an interrupted run keeps everything up to the failure.

Run (the GPU host, ONE pinned GPU)::

    CUDA_VISIBLE_DEVICES=<idx> python -u parity/meep_gpu/gate_driver_route_fused.py \\
        --out results/driver_route_fused_<stamp>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy

# RESOLVED BY NAME, not by parent depth. A ``parents[N]`` in a file that is later
# moved resolves to a directory that exists and is wrong, which this tree has paid
# for once already.
HERE = os.path.dirname(os.path.abspath(__file__))
# The root is the nearest ancestor that holds both the package and the harness
# (``meep_gpu/`` and ``parity/meep_gpu/``): the repository root of this layout,
# and the same directory in any tree the harness is copied into whole.
_probe = HERE
while _probe != os.path.dirname(_probe) and not (
        os.path.isdir(os.path.join(_probe, "meep_gpu"))
        and os.path.isdir(os.path.join(_probe, "parity", "meep_gpu"))):
    _probe = os.path.dirname(_probe)
REPO_API = _probe
if not os.path.isdir(os.path.join(REPO_API, "parity", "meep_gpu")):  # pragma: no cover
    raise SystemExit(f"cannot locate the repository root above {HERE}: no ancestor "
                     f"holds both meep_gpu/ and parity/meep_gpu/")
for path in (REPO_API, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

# THE COMPARATOR IS THE RELEASED ONE, imported rather than re-derived. Its
# collector enumerates the ``Fields`` dataclass instead of a hand-written array
# list, its word view is the uint32 spelling every byte claim in this tree is made
# in, and its negative control is already proven to fire. A second copy here would
# be a second thing to keep correct.
from gate_dispatch_end_to_end import (  # noqa: E402
    TritonLaunchCounter, collect_state, compare_state, comparator_negative_control,
)
import gate_dispatch_end_to_end as _e2e  # noqa: E402


# ---------------------------------------------------------------------------
# Progress (the progress-reporting rule)
# ---------------------------------------------------------------------------

_PROGRESS_PATH: Optional[str] = None

#: The arm this gate certifies, spelled exactly as the composer writes it into
#: ``TritonStepPlan.selected`` (``launch.py``: ``f"fused pair {pair_name}"``).
ARM = "fused pair B"

#: The two driver slots that pair takes. ``step_B`` is the LEADING consult — the
#: one whose ``dispatch`` call performs the launch — and ``update_H`` is ABSORBED.
LEADING_SLOT = "step_B"
ABSORBED_SLOT = "update_H"

#: Both ordinary pairs are opted into, and the reason is the per-label refusal
#: rather than an appetite for coverage. Clause (8) refuses the WHOLE plan when
#: ANY fused label in it was not opted into — "each half was measured" is not "the
#: step was measured" — so on a configuration where the composer also wins
#: ``fused pair D`` (an Hz-driven run has no in-seam ELECTRIC source, and it is the
#: electric source that bars the D pair), opting into B alone would not produce a
#: half-fused step: it would produce a REFUSAL, and the case would measure the
#: fallback. Both labels ride the same weld (``triton_fused_electric_device_gate``,
#: kernels ``fused_curl_constitutive_B``/``_D``).
#:
#: WHICH PAIRS ACTUALLY COMPOSED IS MEASURED PER CASE, never assumed, and every
#: count below is derived from that measurement.
OPT_IN = ("fused pair B", "fused pair D")

#: Which two driver slots each ordinary pair occupies, and which kernel it
#: launches. ``step_B``'s pair is additionally consulted by
#: ``synchronize_magnetic_fields``; ``step_D``'s is not.
PAIRS: Dict[str, Dict[str, Any]] = {
    "fused pair B": {"slots": ("step_B", "update_H"),
                     "kernel": "fused_curl_constitutive_B",
                     "consults_per_step": 2},   # step() and synchronize()
    "fused pair D": {"slots": ("step_D", "update_E"),
                     "kernel": "fused_curl_constitutive_D",
                     "consults_per_step": 1},   # step() only
}

#: Complete driver steps compared word for word. The house standard: long enough
#: that a per-sub-step difference accumulates past the first step's rounding and
#: short enough to run three legs of every case on one device.
DEFAULT_STEPS = 12


def say(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if _PROGRESS_PATH:
        with open(_PROGRESS_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


# ---------------------------------------------------------------------------
# The cases. Every one is chosen to reach ``fused pair B`` — or, for the fold, to
# be refused by a rung this phase does not touch.
# ---------------------------------------------------------------------------

def _gaussian(mp, fcen, df):
    return mp.GaussianSource(frequency=fcen, fwidth=df)


def case_pml_2d(mp, res=20):
    """The canonical shape: unfolded 2-D PML, real float32, ELECTRIC source.

    An Ez source is not in the B pair's seam (``deposit_repair.in_seam_sources``
    filters on ``field_type``), so the absorbed slot holds a bare ``NoopPlan`` and
    this case measures the CLEAN two-slot composition.
    """
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(_gaussian(mp, 0.25, 0.3), component=mp.Ez,
                         center=mp.Vector3(-3.5, 0))]
    return mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                         geometry=geometry, sources=sources, resolution=res)


def case_pml_2d_magnetic_source(mp, res=20):
    """THE SOURCE SEAM: an Hz source lands INSIDE the B pair's seam.

    ``coverage.CARRIES_DEPOSIT_REPAIR`` is True for this product, so the pair is
    bracketed by ``LeadingRepairPlan``/``TrailingRepairPlan`` rather than by a
    ``NoopPlan``: the launch accumulates before the driver injects, and the
    trailing consult is the only point in the step where the injected field is
    final and the pre-injection accumulation is still known.
    """
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(_gaussian(mp, 0.25, 0.3), component=mp.Hz,
                         center=mp.Vector3(-3.5, 0))]
    return mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                         geometry=geometry, sources=sources, resolution=res)


def case_pml_3d(mp, res=12):
    """The same arm in 3-D — a different launch shape and a third axis of walls."""
    cell = mp.Vector3(4, 4, 4)
    geometry = [mp.Sphere(radius=0.8, center=mp.Vector3(),
                          material=mp.Medium(epsilon=9))]
    sources = [mp.Source(_gaussian(mp, 0.4, 0.4), component=mp.Ez,
                         center=mp.Vector3(-1.2, 0, 0))]
    return mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                         geometry=geometry, sources=sources, resolution=res)


def case_folded_2d(mp, res=20):
    """THE FOLD, and it is here to be REFUSED.

    ``mp.Mirror`` on Y. Rung (6b) fires after clause (8), so opting into the arm
    does not reach it: the ladder must still refuse the whole plan and name the
    fold. Narrowing that refusal is a later phase; this case is what makes "the
    fold refusal still stands" a measurement rather than an assumption.
    """
    cell = mp.Vector3(8, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(_gaussian(mp, 0.25, 0.3), component=mp.Ez,
                         center=mp.Vector3(-2.5, 0))]
    return mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                         geometry=geometry, sources=sources, resolution=res,
                         symmetries=[mp.Mirror(mp.Y)])


CASES = {
    "pml_2d": case_pml_2d,
    "pml_2d_magnetic_source": case_pml_2d_magnetic_source,
    "pml_3d": case_pml_3d,
    "folded_2d": case_folded_2d,
}

CASE_INTENT = {
    "pml_2d": "fused pair B, clean seam: the absorbed slot holds a NoopPlan",
    "pml_2d_magnetic_source": "fused pair B with an IN-SEAM magnetic source: the "
                              "deposit repair brackets the pair",
    "pml_3d": "fused pair B in 3-D — a different launch shape",
    "folded_2d": "EXPECTED REFUSAL: rung 6b still refuses the fold, and opting "
                 "into the arm must not reach past it",
}

#: Cases whose planner verdict must be a REFUSAL naming the fold. Asserted, so a
#: build that quietly started dispatching folded grids fails here.
MUST_REFUSE_FOLD = {"folded_2d"}


# ---------------------------------------------------------------------------
# One leg
# ---------------------------------------------------------------------------

def _env_for(leg: str) -> Dict[str, Optional[str]]:
    """The ONE variable that separates each leg from the others."""
    if leg == "fused":
        return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
                "MEEP_GPU_FUSE_ARMS": ",".join(OPT_IN)}
    if leg == "unfused":
        return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
                "MEEP_GPU_FUSE_ARMS": None}
    if leg == "array":
        return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": "0",
                "MEEP_GPU_FUSE_ARMS": None}
    raise ValueError(leg)


def _apply_env(env: Dict[str, Optional[str]]) -> None:
    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def lift(case: str, builder, leg: str, gpu_id: int,
         res: Optional[int], prefer_gpu: bool) -> Dict[str, Any]:
    import meep as mp  # noqa: PLC0415
    import meep_gpu  # noqa: PLC0415

    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    sim = builder(mp) if res is None else builder(mp, res)
    started = time.time()
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=prefer_gpu, gpu_id=gpu_id)
    say(f"{case}/{leg}: lifted {tuple(int(v) for v in driver.shape)} "
        f"({time.time() - started:.1f} s)")
    return {"leg": leg, "sim": sim, "driver": driver, "env": _env_for(leg),
            "grid_shape": [int(v) for v in driver.shape],
            "launches": {"total": 0, "hook_calls": 0, "zero_grid_launches": 0,
                         "by_kernel": {}},
            "per_step_launches": []}


def _accumulate(leg: Dict[str, Any], chunk: Dict[str, Any]) -> None:
    """Add ONE chunk's launches. Accumulated per chunk, never as one long delta.

    The legs are stepped in turn, so a leg's count "since its first chunk" also
    contains every launch the other legs made in between — measured on the
    two-leg gate, where it made the kill-switch leg report thousands of launches
    it never made.
    """
    running = leg["launches"]
    for key in ("total", "hook_calls", "zero_grid_launches"):
        running[key] += chunk[key]
    for name, count in chunk["by_kernel"].items():
        running["by_kernel"][name] = running["by_kernel"].get(name, 0) + count


def _plan_of(driver: Any) -> Any:
    return getattr(driver, "_fast_path", None)


def _plan_summary(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if report is None:
        return {"report": None}
    slots = report.get("slots", {})
    dispatched = [name for name, entry in slots.items()
                  if entry.get("state") == "dispatched"]
    return {
        "decision": report.get("decision"),
        "step_path": report.get("step_path"),
        "refused_because": report.get("refused_because"),
        # True on a prefer_gpu=False lift (``--smoke``): the NumPy reference, whose
        # freeze consults no kernel table. The array-leg control reads it.
        "reference_driver": report.get("reference_driver"),
        "fusion": report.get("fusion"),
        "dispatched_slots": dispatched,
        "arms": {name: slots[name].get("arm") for name in dispatched},
        "array_slot_reasons": {name: slots[name].get("reason")
                               for name in slots if name not in dispatched},
        "families": {arm: {"family": entry.get("family"), "gate": entry.get("gate"),
                           "recorded_utc": entry.get("recorded_utc"),
                           "host": entry.get("host"),
                           "subnormal_policy": entry.get("subnormal_policy")}
                     for arm, entry in (report.get("families") or {}).items()},
        "composition": report.get("composition"),
        "run_shape": report.get("run_shape"),
        "subnormal": report.get("subnormal"),
        "launch_counters": report.get("launch_counters"),
        "built_not_dispatched": report.get("built_not_dispatched"),
    }


def freeze(leg: Dict[str, Any]) -> Dict[str, Any]:
    """Provoke the freeze THROUGH THE DRIVER, by taking one real step.

    NOT ``plan_fast_path`` called from here. The whole complaint clause (8) makes
    is that fused products have only ever been installed by a harness reaching
    into module globals, so a gate that planned the configuration itself and then
    poked the result into ``driver._fast_path`` would reproduce exactly the defect
    it is meant to close. ``driver.step`` freezes on its first call
    (``driver.py``: ``if self._fast_path_stale: self._fast_path = plan_fast_path(...)``),
    so one step is the driver's own freeze and nothing here substitutes for it.

    THE COST IS ONE STEP, and it is why the compared window starts at step 2. The
    plan-time WARM PASS launches every dispatchable slot's kernel once at
    ``grid=(0,)`` inside that first freeze, and it warms a different number of
    kernels on the fused leg (one pair) than on the unfused leg (two separate
    products). Counting the freeze step would put that difference into the
    substitution arithmetic, where it is noise; the freeze step is stepped on
    every leg, compared like any other, and excluded from the launch accounting.
    """
    _apply_env(leg["env"])
    driver = leg["driver"]
    driver.step()
    leg["freeze_step_taken"] = True
    leg["active_step_path"] = driver.active_step_path
    leg["plan"] = _plan_summary(driver.fast_path_report())
    return leg["plan"]


def _pair_shape(leg: Dict[str, Any]) -> Dict[str, Any]:
    """What the composer actually installed in the two slots, read off the objects.

    The record STATES which arm won each slot; this READS the plan objects, so a
    label that no longer matches its wiring is visible. ``absorbed_by`` is the
    whole two-slot mechanism and is the thing a regression would break silently.
    """
    plan = _plan_of(leg["driver"])
    out: Dict[str, Any] = {"plan_is_none": plan is None}
    if plan is None:
        return out
    plans = getattr(getattr(plan, "step_plan", None), "plans", {}) or {}
    for slot in (LEADING_SLOT, ABSORBED_SLOT):
        entry = plans.get(slot)
        out[slot] = {
            "class": type(entry).__name__ if entry is not None else None,
            "absorbed_by": (type(getattr(entry, "absorbed_by", None)).__name__
                            if getattr(entry, "absorbed_by", None) is not None
                            else None),
        }
    leading = plans.get(LEADING_SLOT)
    inner = getattr(leading, "inner", leading)
    for attribute in ("pair", "zero_metal", "bc", "shape", "block", "num_warps"):
        value = getattr(inner, attribute, None)
        if value is not None:
            out[attribute] = list(value) if isinstance(value, tuple) else value
    out["slots_claimed"] = list(getattr(plan, "slots", ()) or ())
    out["committed_after_freeze"] = sorted(getattr(plan, "_committed", set()) or set())
    return out


def withhold_the_fused_route(leg: Dict[str, Any]) -> Dict[str, Any]:
    """THE NULL CONTROL: empty the fused plan's launch grid so it computes nothing.

    Not a monkeypatch of the dispatcher and not a deleted slot — every identity in
    the plan survives, ``absorbed_by`` still points where it pointed, the
    ``_committed`` bookkeeping still runs, the record still reads dispatched and
    every launch counter still counts the launch. The single thing removed is the
    ARITHMETIC, through the same ``_grid = (0,)`` mechanism ``fastpath.warm_plan``
    uses on purpose at plan time.

    If the comparison still says byte-identical after this, then nothing in the
    compared state depends on the fused kernel and the gate measured nothing.
    """
    plan = _plan_of(leg["driver"])
    plans = getattr(getattr(plan, "step_plan", None), "plans", {}) or {}
    leading = plans.get(LEADING_SLOT)
    inner = getattr(leading, "inner", leading)
    original = getattr(inner, "_grid", None)
    if not isinstance(original, tuple) or len(original) != 1:
        return {"withheld": False,
                "why": f"the leading plan's _grid is {original!r}, not a 1-tuple"}
    inner._grid = (0,)
    return {"withheld": True, "plan_class": type(inner).__name__,
            "grid_was": list(original), "grid_now": [0]}


def step_and_sync(leg: Dict[str, Any], counter: TritonLaunchCounter,
                  enter_second_site: bool) -> Dict[str, Any]:
    """ONE complete driver step, then the second consult site, on this leg."""
    _apply_env(leg["env"])
    driver = leg["driver"]

    before = counter.snapshot()
    driver.step()
    after_step = counter.snapshot()
    step_chunk = TritonLaunchCounter.delta(before, after_step)
    _accumulate(leg, step_chunk)
    stepped = collect_state(driver)

    sync_chunk = {"total": 0, "hook_calls": 0, "zero_grid_launches": 0,
                  "by_kernel": {}}
    synchronized: Optional[Dict[str, numpy.ndarray]] = None
    restored: Optional[Dict[str, numpy.ndarray]] = None
    committed: List[int] = []
    if enter_second_site:
        # THE SECOND CONSULT SITE, entered explicitly. A DFT flux monitor
        # accumulates without it; ``flux_in_box``/``field_energy_in_box`` are what
        # reach it, and calling it directly is the same two consults with the same
        # three in-seam passes between them, on a plan it does NOT re-build.
        driver.synchronize_magnetic_fields()
        after_sync = counter.snapshot()
        sync_chunk = TritonLaunchCounter.delta(after_step, after_sync)
        _accumulate(leg, sync_chunk)
        synchronized = collect_state(driver)
        plan = _plan_of(driver)
        committed = sorted(getattr(plan, "_committed", set()) or set())
        driver.restore_magnetic_fields()
        restored = collect_state(driver)

    leg["per_step_launches"].append(
        {"step": step_chunk["total"], "sync": sync_chunk["total"],
         "by_kernel_step": dict(step_chunk["by_kernel"]),
         "by_kernel_sync": dict(sync_chunk["by_kernel"])})
    leg["active_step_path"] = driver.active_step_path
    return {"stepped": stepped, "synchronized": synchronized,
            "restored": restored, "committed_after_sync": committed}


# ---------------------------------------------------------------------------
# The verdict clauses
# ---------------------------------------------------------------------------

def composed_pairs(leg: Dict[str, Any]) -> Tuple[str, ...]:
    """Which ordinary pairs the composer actually installed, READ off the record."""
    arms = (leg["plan"] or {}).get("arms") or {}
    return tuple(name for name, spec in PAIRS.items()
                 if all(arms.get(slot) == name for slot in spec["slots"]))


def composition_is_the_fused_pair(leg: Dict[str, Any]) -> Dict[str, Any]:
    """Clause 1: the certified arm holds BOTH its slots, wired as a pair."""
    plan = leg["plan"]
    shape = leg["pair_shape"]
    failures: List[str] = []
    arms = plan.get("arms") or {}
    for slot in (LEADING_SLOT, ABSORBED_SLOT):
        if arms.get(slot) != ARM:
            failures.append(f"{slot} carries arm {arms.get(slot)!r}, not {ARM!r}")
    absorbed = (shape.get(ABSORBED_SLOT) or {}).get("absorbed_by")
    if not absorbed:
        failures.append("the absorbed slot names no absorbed_by; the two slots are "
                        "not wired as a pair")
    unfilled = [slot for slot in (LEADING_SLOT, ABSORBED_SLOT)
                if slot not in (plan.get("dispatched_slots") or [])]
    if unfilled:
        failures.append(f"the record does not call {unfilled} dispatched")
    return {"ok": not failures, "failures": failures,
            "arms": arms, "absorbed_by": absorbed,
            "pairs_composed": list(composed_pairs(leg)),
            "driver_slots_not_dispatched": unfilled}


def kernels_actually_ran(leg: Dict[str, Any]) -> Dict[str, Any]:
    """Clause 3: non-vacuity, per slot and at the Triton level."""
    plan = leg["plan"]
    counters = plan.get("launch_counters") or {}
    failures: List[str] = []
    if leg["active_step_path"] != "fused":
        failures.append(f"active_step_path reads {leg['active_step_path']!r}")
    if not plan.get("dispatched_slots"):
        failures.append("the record names no dispatched slot")
    for slot in plan.get("dispatched_slots", []):
        entry = counters.get(slot) or {}
        if not entry.get("dispatches"):
            failures.append(f"{slot} is recorded dispatched but its kernel ran "
                            f"{entry.get('dispatches')} times")
        if entry.get("programs_per_dispatch") == 0:
            failures.append(f"{slot} launched over an EMPTY grid: 0 programs")
    if not leg["launches"]["total"]:
        failures.append("no Triton kernel launched at all (JITFunction.run counter)")
    if not leg["launches"]["hook_calls"]:
        failures.append("no Triton launcher hook fired")
    return {"kernels_ran": not failures, "failures": failures,
            "slot_dispatches": {slot: (counters.get(slot) or {}).get("dispatches")
                                for slot in plan.get("dispatched_slots", [])},
            "programs_per_dispatch": {
                slot: (counters.get(slot) or {}).get("programs_per_dispatch")
                for slot in plan.get("dispatched_slots", [])},
            "triton_launches": leg["launches"]}


#: The two kernels an ordinary pair replaces. Matched as SUBSTRINGS of the
#: launched kernel's name, because Triton reports the JIT function's name and this
#: gate must not pin a spelling it does not own.
REPLACED_KERNELS = ("pml_curl_step", "constitutive_step")


def _named(leg: Dict[str, Any], needle: str) -> int:
    return sum(count for name, count in leg["launches"]["by_kernel"].items()
               if needle in name)


def substitution_proof(fused: Dict[str, Any], unfused: Dict[str, Any],
                       pairs: Tuple[str, ...], steps: int) -> Dict[str, Any]:
    """Clause 4: the device launch count DROPS by exactly one per fused launch.

    THE ARITHMETIC IS DERIVED, NOT ASSUMED. Each pair's leading consult performs
    one launch where the unfused composition performs two, so the expected number
    of fused launches is ``steps x consults_per_step`` summed over the pairs the
    composer ACTUALLY installed (read off the record, per case), and the expected
    total drop equals that same number — one net launch removed per fused launch,
    because each replaces exactly two.

    TWO INDEPENDENT COUNTS, the way the hand-CUDA timing gate proved its
    substitution, because each alone has a way to be wrong. Counting the TOTALS
    alone would pass if the composer had swapped one kernel for some other single
    kernel. Counting by NAME alone would pass if the fused kernel ran ALONGSIDE
    the two it replaces. So: the totals must drop by the derived amount, the fused
    kernels must appear on the fused leg exactly the derived number of times and
    ZERO times on the unfused leg, and each replaced kernel must be short by
    exactly the derived number on the fused leg.
    """
    failures: List[str] = []
    expected_fused_launches = sum(steps * PAIRS[name]["consults_per_step"]
                                  for name in pairs)
    if not expected_fused_launches:
        failures.append("no ordinary fused pair composed, so there is no "
                        "substitution to prove")

    per_kernel: Dict[str, Dict[str, int]] = {}
    for name in PAIRS:
        kernel = PAIRS[name]["kernel"]
        expected = (steps * PAIRS[name]["consults_per_step"]
                    if name in pairs else 0)
        got_fused, got_unfused = _named(fused, kernel), _named(unfused, kernel)
        per_kernel[kernel] = {"fused": got_fused, "unfused": got_unfused,
                              "expected_on_fused": expected}
        if got_fused != expected:
            failures.append(f"{kernel} launched {got_fused} times on the fused "
                            f"leg, expected {expected}")
        if got_unfused:
            failures.append(f"{kernel} launched {got_unfused} times on the UNFUSED "
                            f"leg, which is then not a control")

    drop = unfused["launches"]["total"] - fused["launches"]["total"]
    if drop != expected_fused_launches:
        failures.append(f"the launch drop is {drop}, not {expected_fused_launches} "
                        "— each fused launch replaces exactly two, so the net drop "
                        "is one per fused launch")

    replaced: Dict[str, Dict[str, int]] = {}
    for needle in REPLACED_KERNELS:
        got_fused, got_unfused = _named(fused, needle), _named(unfused, needle)
        replaced[needle] = {"fused": got_fused, "unfused": got_unfused,
                            "removed": got_unfused - got_fused,
                            "expected_removed": expected_fused_launches}
        if got_unfused - got_fused != expected_fused_launches:
            failures.append(f"{needle} dropped by {got_unfused - got_fused} on the "
                            f"fused leg, expected {expected_fused_launches}")
    return {"proved": not failures, "failures": failures,
            "pairs_composed": list(pairs), "steps": steps,
            "expected_fused_launches": expected_fused_launches,
            "launch_drop": drop,
            "totals": {"fused": fused["launches"]["total"],
                       "unfused": unfused["launches"]["total"]},
            "fused_kernel_launches": per_kernel,
            "replaced_kernel_launches": replaced,
            "by_kernel": {"fused": fused["launches"]["by_kernel"],
                          "unfused": unfused["launches"]["by_kernel"]}}


def array_leg_is_clean(leg: Dict[str, Any]) -> Dict[str, Any]:
    """The oracle leg must be the array path and must launch NOTHING.

    A GPU driver's oracle leg (every gate run) must have been refused at the kill
    switch, by name. A ``--smoke`` leg is lifted ``prefer_gpu=False``, the NumPy
    reference, which never plans and reads no kill switch: its record carries
    ``reference_driver`` instead, and it is accepted only while the run lifts the
    reference on purpose (``_e2e.PREFER_GPU`` False, which is ``--smoke``).
    """
    failures: List[str] = []
    plan = leg["plan"] or {}
    reference = bool(plan.get("reference_driver"))
    if leg["active_step_path"] != "array":
        failures.append(f"active_step_path reads {leg['active_step_path']!r}")
    if leg["launches"]["total"]:
        failures.append(f"{leg['launches']['total']} Triton launches on a leg that "
                        "must launch none")
    reason = plan.get("refused_because") or ""
    if reference:
        if _e2e.PREFER_GPU:
            failures.append("the leg is a NumPy reference driver (prefer_gpu=False) "
                            "on a run that lifts GPU drivers; only a --smoke run "
                            "lifts the reference")
    elif "kill switch" not in reason:
        failures.append(f"the refusal does not name the kill switch: {reason!r}")
    return {"clean": not failures, "failures": failures, "refused_because": reason,
            "reference_driver": reference}


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(name: str, gpu_id: int, counter: TritonLaunchCounter,
             steps: int, res: Optional[int], prefer_gpu: bool) -> Dict[str, Any]:
    builder = CASES[name]
    row: Dict[str, Any] = {
        "case": name, "intent": CASE_INTENT[name], "arm": ARM,
        "steps_planned": steps,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    legs = {leg: lift(name, builder, leg, gpu_id, res, prefer_gpu)
            for leg in ("fused", "unfused", "array")}
    row["grid_shape"] = legs["fused"]["grid_shape"]

    # THE PRECONDITION IS TAKEN FIRST, on drivers that have not stepped. Three
    # separately lifted drivers that never shared a starting state would make
    # every comparison below meaningless, and a freeze cannot be what establishes
    # they agree.
    states = {leg: collect_state(legs[leg]["driver"]) for leg in legs}
    pre_ac = compare_state(states["fused"], states["array"])
    pre_bc = compare_state(states["unfused"], states["array"])
    row["precondition"] = {"fused_vs_array": pre_ac, "unfused_vs_array": pre_bc}
    row["precondition_identical"] = pre_ac["identical"] and pre_bc["identical"]
    if not row["precondition_identical"]:
        row["verdict"] = "HARNESS-FAILURE"
        row["why"] = "the legs were lifted to different initial states"
        for leg in legs.values():
            leg["driver"].close()
        return row
    say(f"{name}: precondition OK — {pre_ac['arrays']} arrays, "
        f"{pre_ac['words_compared']} words identical on all three legs")

    for leg in legs.values():
        freeze(leg)
        leg["pair_shape"] = _pair_shape(leg)
    row["plans"] = {leg: legs[leg]["plan"] for leg in legs}
    row["pair_shape"] = legs["fused"]["pair_shape"]

    # The freeze step is itself a complete driver step and is compared like any
    # other; only its LAUNCHES are excluded (see :func:`freeze`).
    after_freeze = {leg: collect_state(legs[leg]["driver"]) for leg in legs}
    row["freeze_step"] = {
        "fused_vs_array": compare_state(after_freeze["fused"],
                                        after_freeze["array"])["identical"],
        "unfused_vs_array": compare_state(after_freeze["unfused"],
                                          after_freeze["array"])["identical"],
    }

    # THE FOLD CASE: the ladder must refuse, and it must refuse for the FOLD —
    # not because the arm was never opted into and not because something else
    # upstream said no.
    if name in MUST_REFUSE_FOLD:
        refusal = legs["fused"]["plan"].get("refused_because") or ""
        row["fold_refusal"] = refusal
        row["fold_refusal_names_the_fold"] = "mirror-folded" in refusal
        row["fusion_block"] = legs["fused"]["plan"].get("fusion")
        row["verdict"] = ("PASS-REFUSED-BY-THE-FOLD"
                          if row["fold_refusal_names_the_fold"]
                          else "FOLD-REFUSAL-MISSING")
        row["why"] = ("rung 6b fires after clause 8, so opting into the arm must "
                      "not reach past the fold; this case asserts that refusal is "
                      "still standing and narrowing it is a later phase")
        for leg in legs.values():
            leg["driver"].close()
        return row

    row["composition"] = composition_is_the_fused_pair(legs["fused"])
    if not row["composition"]["ok"]:
        row["verdict"] = "COMPOSITION-NOT-FUSED"
        row["why"] = ("the fused leg did not compose the pair, so nothing below "
                      "would be about fusion")
        row["refused_because"] = legs["fused"]["plan"].get("refused_because")
        for leg in legs.values():
            leg["driver"].close()
        return row

    row["per_step"] = []
    row["first_divergent_step"] = None
    for index in range(1, steps + 1):
        taken = {leg: step_and_sync(legs[leg], counter, enter_second_site=True)
                 for leg in ("fused", "unfused", "array")}
        entry: Dict[str, Any] = {"step": index}
        for label, left, right in (
                ("fused_vs_array", "fused", "array"),
                ("unfused_vs_array", "unfused", "array"),
                ("fused_vs_unfused", "fused", "unfused")):
            entry[label] = {
                phase: {"identical": verdict["identical"],
                        "arrays_differing": verdict["arrays_differing"],
                        "words": verdict["words_compared"],
                        "differences": verdict["differences"][:4]}
                for phase, verdict in (
                    (phase, compare_state(taken[left][phase], taken[right][phase]))
                    for phase in ("stepped", "synchronized", "restored"))}
        entry["committed_after_sync"] = {
            leg: taken[leg]["committed_after_sync"] for leg in taken}
        row["per_step"].append(entry)
        agreed = all(entry[label][phase]["identical"]
                     for label in ("fused_vs_array", "unfused_vs_array",
                                   "fused_vs_unfused")
                     for phase in ("stepped", "synchronized", "restored"))
        if not agreed and row["first_divergent_step"] is None:
            row["first_divergent_step"] = index
        say(f"{name}: step {index}/{steps} -> "
            f"{'IDENTICAL' if agreed else 'DIFFERS'} "
            f"(fused {legs['fused']['launches']['total']} launches, "
            f"unfused {legs['unfused']['launches']['total']}, "
            f"array {legs['array']['launches']['total']})")

    # THE SECOND CONSULT SITE'S BOOKKEEPING. The pair commits at the leading
    # consult and the absorbed one discards it, so a plan that has finished a sync
    # must hold nothing; a leaked entry means the two slots would stop answering
    # together on some later step.
    row["committed_is_empty_after_every_sync"] = all(
        not entry["committed_after_sync"]["fused"] for entry in row["per_step"])

    row["substitution"] = substitution_proof(
        legs["fused"], legs["unfused"], composed_pairs(legs["fused"]), steps)
    row["evidence"] = kernels_actually_ran(legs["fused"])
    row["unfused_evidence"] = kernels_actually_ran(legs["unfused"])
    row["array_control"] = array_leg_is_clean(legs["array"])
    row["comparator_control"] = comparator_negative_control(
        collect_state(legs["fused"]["driver"]))
    row["per_step_launches"] = {leg: legs[leg]["per_step_launches"] for leg in legs}
    row["launch_totals"] = {leg: legs[leg]["launches"] for leg in legs}

    for leg in legs.values():
        leg["driver"].close()

    # ------------------------------------------------------------------
    # THE NULL CONTROL, as its own pair of drivers: the same case, the fused
    # route WITHHELD, and it must diverge.
    # ------------------------------------------------------------------
    row["null_control"] = run_null_control(name, builder, gpu_id, counter,
                                           res, prefer_gpu)

    agree = row["first_divergent_step"] is None
    control = row["comparator_control"]
    if not agree:
        row["verdict"] = "DIVERGENCE"
    elif not (control.get("armed") and control.get("comparator_saw_it")
              and control.get("words_mismatched") == 1):
        row["verdict"] = "COMPARATOR-BLIND"
        row["why"] = ("the legs matched, but the comparator did not report a "
                      "one-ULP perturbation of the state it just compared")
    elif not row["null_control"]["diverged"]:
        row["verdict"] = "NULL-CONTROL-FAILED"
        row["why"] = ("withholding the fused kernel's arithmetic changed NOTHING "
                      "in the compared state, so the byte identity above is not "
                      "evidence that the fused kernel computed it")
    elif not row["evidence"]["kernels_ran"]:
        row["verdict"] = "VACUOUS-PASS"
    elif not row["substitution"]["proved"]:
        row["verdict"] = "SUBSTITUTION-UNPROVEN"
    elif not row["array_control"]["clean"]:
        row["verdict"] = "CONTROL-FAILURE"
    elif not row["committed_is_empty_after_every_sync"]:
        row["verdict"] = "SECOND-SITE-BOOKKEEPING"
    else:
        row["verdict"] = "PASS-DISPATCHED-FUSED"
    return row


def run_null_control(name: str, builder, gpu_id: int,
                     counter: TritonLaunchCounter, res: Optional[int],
                     prefer_gpu: bool) -> Dict[str, Any]:
    """The same case with the fused kernel's arithmetic withheld. MUST diverge.

    Fresh drivers rather than the stepped ones: the withholding has to be in force
    from the first step, and a plan whose grid is emptied mid-run would be a
    different experiment.
    """
    say(f"{name}: NULL CONTROL — withholding the fused route")
    fused = lift(name, builder, "fused", gpu_id, res, prefer_gpu)
    array = lift(name, builder, "array", gpu_id, res, prefer_gpu)
    freeze(fused)
    freeze(array)
    out: Dict[str, Any] = {"withholding": withhold_the_fused_route(fused)}
    if not out["withholding"]["withheld"]:
        out["diverged"] = False
        out["why"] = "the fused plan's launch grid could not be emptied"
        fused["driver"].close()
        array["driver"].close()
        return out

    before = compare_state(collect_state(fused["driver"]),
                           collect_state(array["driver"]))
    out["precondition_identical"] = before["identical"]
    diverged_at = None
    for index in (1, 2, 3):
        taken_f = step_and_sync(fused, counter, enter_second_site=False)
        taken_a = step_and_sync(array, counter, enter_second_site=False)
        verdict = compare_state(taken_f["stepped"], taken_a["stepped"])
        if not verdict["identical"] and diverged_at is None:
            diverged_at = index
            out["first_divergence"] = {
                "step": index, "arrays_differing": verdict["arrays_differing"],
                "differences": verdict["differences"][:4]}
    out["diverged"] = diverged_at is not None
    out["diverged_at_step"] = diverged_at
    # The record must still say DISPATCHED and the counters must still count the
    # launch: what was removed is arithmetic, not the route.
    out["still_reads_dispatched"] = fused["plan"].get("step_path") == "fused"
    out["launches_while_withheld"] = fused["launches"]["total"]
    out["active_step_path"] = fused.get("active_step_path")
    say(f"{name}: NULL CONTROL -> "
        f"{'DIVERGED at step %s' % diverged_at if diverged_at else 'NO DIVERGENCE'} "
        f"({out['launches_while_withheld']} launches while withheld)")
    fused["driver"].close()
    array["driver"].close()
    return out


# ---------------------------------------------------------------------------
# The policy leg
# ---------------------------------------------------------------------------

def policy_probe(policy: str, gpu_id: int, res: Optional[int],
                 prefer_gpu: bool) -> Dict[str, Any]:
    """ONE policy, in THIS process, which must therefore be a fresh one.

    Run through ``--policy-probe``; see :func:`policy_leg` for why it is a
    subprocess and not a third leg of the main run.
    """
    import meep_gpu  # noqa: PLC0415
    from meep_gpu import fastpath  # noqa: PLC0415

    installed_error = None
    if policy != fastpath.CERTIFICATION_SUBNORMAL_POLICY:
        # INSTALLED BY THE HARNESS, deliberately: it is the only way to reach rung
        # 8b's "this process already installed a different policy" branch. With
        # nothing installed the rung installs the certified policy ITSELF, which
        # is the keep leg and is the evidence DISPATCH_BY_DEFAULT asks for.
        try:
            meep_gpu.install_subnormal_policy(policy)
        except Exception as exc:  # noqa: BLE001
            installed_error = repr(exc)
    leg = lift("policy", CASES["pml_2d"], "fused", gpu_id, res, prefer_gpu)
    summary = freeze(leg)
    out = {
        "policy": policy,
        "harness_installed": None if policy == "keep" else policy,
        "install_error": installed_error,
        "decision": summary.get("decision"),
        "step_path": summary.get("step_path"),
        "refused_because": summary.get("refused_because"),
        "subnormal": summary.get("subnormal"),
        "dispatched_slots": summary.get("dispatched_slots"),
        "arms": summary.get("arms"),
    }
    leg["driver"].close()
    return out


def policy_leg(out_dir: str, gpu_id: int, res: Optional[int],
               prefer_gpu: bool) -> Dict[str, Any]:
    """BOTH policies, and what the dispatch route actually supports.

    Rung 8b installs ``CERTIFICATION_SUBNORMAL_POLICY`` and refuses any process
    that installed anything else, so this route supports exactly ONE policy. The
    honest measurement of "both policies" for such a family is therefore: keep
    dispatches with the HARNESS INSTALLING NOTHING, and flush is REFUSED BY NAME
    rather than silently dispatched. Reporting only the keep leg would leave the
    flush answer unmeasured, and the flush answer is the one that would be a wrong
    number at runtime.

    EACH IN A FRESH PROCESS, because the thing being measured is a property of a
    process. A policy cannot be swapped in place — binaries compiled under the
    first are still reachable and CuPy's cache key is computed above the seam the
    strip installs at — which is exactly what rung 8b's refusal says, so measuring
    it by installing one policy after another in the run that already dispatched
    would be measuring the swap, not the gate.
    """
    import subprocess  # noqa: PLC0415

    out: Dict[str, Any] = {}
    for policy in ("keep", "flush"):
        command = [sys.executable, "-u", os.path.abspath(__file__),
                   "--policy-probe", policy, "--gpu-id", str(gpu_id),
                   "--out", out_dir]
        if res is not None:
            command += ["--resolution", str(res)]
        if not prefer_gpu:
            command += ["--smoke"]
        finished = subprocess.run(command, capture_output=True, text=True)
        payload: Dict[str, Any] = {"rc": finished.returncode,
                                   "stderr_tail": finished.stderr[-2000:]}
        for line in finished.stdout.splitlines():
            if line.startswith("POLICY_PROBE_JSON "):
                payload.update(json.loads(line[len("POLICY_PROBE_JSON "):]))
        out[policy] = payload
        say(f"policy/{policy}: {payload.get('decision')} "
            f"({payload.get('refused_because') or 'dispatched'})")
    out["keep_dispatches"] = out["keep"].get("decision") == "dispatched"
    out["keep_installed_by_dispatch_not_the_harness"] = (
        out["keep"].get("harness_installed") is None
        and ((out["keep"].get("subnormal") or {}).get("gate") or {}
             ).get("installed_by") is not None)
    out["flush_is_refused_by_name"] = (
        out["flush"].get("decision") == "refused"
        and "subnormal policy" in (out["flush"].get("refused_because") or "").lower())
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#: The files whose bytes this gate's verdict is about. Digested, not transcribed.
DIGESTED = (
    "meep_gpu/fastpath.py",
    "meep_gpu/driver.py",
    "meep_gpu/fields.py",
    "meep_gpu/stepping.py",
    "meep_gpu/deposit_repair.py",
    "meep_gpu/triton_kernels/launch.py",
    "meep_gpu/triton_kernels/kernels.py",
    "meep_gpu/triton_kernels/coverage.py",
    "meep_gpu/triton_kernels/fingerprints.json",
    "parity/meep_gpu/gate_driver_route_fused.py",
    "parity/meep_gpu/gate_dispatch_end_to_end.py",
)


def _provenance(gpu_id: int) -> Dict[str, Any]:
    digests = {}
    for name in DIGESTED:
        path = os.path.join(REPO_API, name)
        with open(path, "rb") as handle:
            digests[name] = hashlib.sha256(handle.read()).hexdigest()
    record: Dict[str, Any] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_id": gpu_id,
        "source_sha256": digests,
        "arm": ARM,
    }
    for module in ("meep", "cupy", "triton", "numpy"):
        try:
            record[f"{module}_version"] = __import__(module).__version__
        except Exception as exc:  # noqa: BLE001
            record[f"{module}_version"] = f"unavailable: {exc!r}"
    try:
        import cupy  # noqa: PLC0415

        device = cupy.cuda.Device(gpu_id)
        record["device"] = {
            "name": device.attributes and cupy.cuda.runtime.getDeviceProperties(
                gpu_id)["name"].decode(),
            "compute_capability": device.compute_capability,
        }
    except Exception as exc:  # noqa: BLE001
        record["device"] = {"unreadable": repr(exc)}
    record["cupy_cache_dir"] = os.environ.get("CUPY_CACHE_DIR")
    return record


def main() -> int:
    global _PROGRESS_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="NOT A GATE RUN. Lift the NumPy reference "
                             "(prefer_gpu=False) so the harness's own plumbing can "
                             "be shaken out on a laptop; the reference consults no "
                             "kernel table and nothing about dispatch is measured. "
                             "Without it the gate refuses a host with no CUDA "
                             "device.")
    parser.add_argument("--policy-probe", default=None, choices=("keep", "flush"),
                        help="internal: freeze one configuration under one policy "
                             "in THIS (fresh) process and print the record")
    arguments = parser.parse_args()
    prefer_gpu = not arguments.smoke
    # AN NVIDIA GATE, REFUSED OFF NVIDIA -- before the policy-probe branch too, whose
    # child lifts with the same prefer_gpu: on an Apple host prefer_gpu=True resolves
    # the Metal table, which this gate does not grade.
    refusal = (_e2e.nvidia_host_refusal("gate_driver_route_fused.py")
               if prefer_gpu else None)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    if arguments.policy_probe:
        payload = policy_probe(arguments.policy_probe, arguments.gpu_id,
                               arguments.resolution, prefer_gpu)
        print("POLICY_PROBE_JSON " + json.dumps(payload, default=str), flush=True)
        return 0

    os.makedirs(arguments.out, exist_ok=True)
    _PROGRESS_PATH = os.path.join(arguments.out, "progress.log")
    _e2e._PROGRESS_PATH = _PROGRESS_PATH
    _e2e.PREFER_GPU = prefer_gpu
    rows_path = os.path.join(arguments.out, "cases.jsonl")

    names = (list(CASES) if arguments.cases == "all"
             else [n.strip() for n in arguments.cases.split(",") if n.strip()])
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown cases: {unknown}; known: {sorted(CASES)}", file=sys.stderr)
        return 2

    counter = TritonLaunchCounter()
    try:
        counter.install()
        say("Triton launch counter installed on JITFunction.run and "
            "CompiledKernel.launch_enter_hook")
    except Exception as exc:  # noqa: BLE001
        if not arguments.smoke:
            say(f"REFUSING: the Triton launch counter could not be installed "
                f"({exc!r}); without it a fallback that matched could not be told "
                "from a dispatch")
            return 3
        say(f"smoke run: no Triton launch counter ({exc!r})")

    provenance = _provenance(arguments.gpu_id)
    provenance["smoke"] = arguments.smoke
    provenance["steps"] = arguments.steps
    say(f"provenance: {provenance['host']} {provenance.get('device')} "
        f"triton={provenance['triton_version']} cupy={provenance['cupy_version']}")

    rows: List[Dict[str, Any]] = []
    for name in names:
        try:
            row = run_case(name, arguments.gpu_id, counter, arguments.steps,
                           arguments.resolution, prefer_gpu)
        except Exception as exc:  # noqa: BLE001 - one case's abort is not the run's
            import traceback  # noqa: PLC0415

            row = {"case": name, "verdict": "CASE-RAISED", "error": repr(exc),
                   "traceback": traceback.format_exc()}
            say(f"{name}: RAISED {exc!r}")
        rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
        say(f"{name}: {row['verdict']}")

    policy: Dict[str, Any]
    try:
        policy = policy_leg(arguments.out, arguments.gpu_id,
                            arguments.resolution, prefer_gpu)
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        policy = {"error": repr(exc), "traceback": traceback.format_exc()}
        say(f"policy leg RAISED {exc!r}")

    passing = {"PASS-DISPATCHED-FUSED", "PASS-REFUSED-BY-THE-FOLD"}
    verdicts = {row["case"]: row.get("verdict") for row in rows}
    dispatched = [name for name, v in verdicts.items()
                  if v == "PASS-DISPATCHED-FUSED"]
    summary = {
        "provenance": provenance,
        "arm": ARM,
        "verdicts": verdicts,
        "dispatched_cases": dispatched,
        "policy": policy,
        "pass": (all(v in passing for v in verdicts.values())
                 and bool(dispatched)
                 and bool(policy.get("keep_dispatches"))
                 and bool(policy.get("flush_is_refused_by_name"))),
        "what_this_licenses": (
            "ONE arm — 'fused pair B' — driven through driver.step's step_B/update_H "
            "consults AND through synchronize_magnetic_fields' second pair of "
            "consults, on unfolded real-float32 PML grids, under the 'keep' policy "
            "installed by fastpath rung 8b with the harness installing nothing. It "
            "does NOT license flipping any default, does NOT reach a folded grid "
            "(rung 6b still refuses and this run asserts that refusal standing), and "
            "does NOT cover fused pair D, the dispersive fused pair or the fused ADE "
            "state, none of which were composed here."),
    }
    with open(os.path.join(arguments.out, "gate.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "cases": rows}, handle, indent=2, default=str)
    say(f"WROTE {os.path.join(arguments.out, 'gate.json')}")
    say(f"VERDICTS: {verdicts}")
    say(f"PASS={summary['pass']}")
    # A leg that measured zero must exit non-zero: a green run with no dispatching
    # case is the shape of pass this project has caught repeatedly.
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
