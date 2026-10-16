"""A run's FIRST ``update_H`` through the Metal composition: the leg two gates share.

WHY THE OTHER LEGS NEVER SAW IT. Every other leg of ``gate_metal_pml`` and
``gate_metal_fused_magnetic_pair`` seeds a mid-run state, which carries
``Fields.h_separated`` (H was started from B long ago), so neither the array path nor the
composition takes MEEP's start of H (``stepping.separate_h_from_b``; update_eh.cpp:165-177)
on it. A run is frozen BEFORE its first step, with ``h_separated`` False; there the array
path starts H and ``f_w`` from B on the dsigw cells, and the Metal kernels step the forms
alone, so the composition carries the start around the slot
(``metal_kernels.launch.StartOfHPlan``). Found by the corpus run of 2026-10-12
(``ex.differential_cross_section``: 240 H words at step 1, sign of zero only).

THE CASE. All +0, no source, Bloch k = 0 on every axis, resolution 8 and a 4-cell layer: the
outermost cells of both lattices carry a negative ``kms``, so the first ``step_B`` lands B on
-0 there (``+0 * kms``), and on a dsigw cell whose own ``kms`` is negative the start keeps
that -0 in H where the forms alone store +0. The leg asserts that the case is not vacuous --
negative coefficients present, and the array path's first H differing from the forms-alone H
on a nonzero count of words -- before it scores anything.

THE SCORE. The composition is built by ``launch.plan_step`` and bracketed by
``launch.wrap_for_residency``, exactly as the dispatch ladder builds it, and stepped slot by
slot in the driver's order for ``STEPS`` complete steps against the array path's own sub-steps
on a copy, every stored volume compared as uint32 words after every step:

* SHIPPED -- 0 differing words at every step, and the slot's start ran exactly once;
* REVERT  -- the same composition with the start stripped from the slot (the occupant the
  composer wrapped, run bare, which is what the composition was before the start was
  carried) MUST differ, and on the first step's H words first.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: Complete steps per score. The defect lives in step 1; step 2 shows the start does not
#: run twice (a second start would re-copy B over an H that has moved off it).
STEPS = 2

#: Every stored volume a complete step can touch.
STATE: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

NEGATIVE_ZERO = 0x80000000


def _words(array: Any) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def first_step_case(dimensions: int = 3) -> Tuple[Any, Any]:
    """A run frozen before its first step: the triple, all +0, ``h_separated`` False."""
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    cell = (2.0, 2.0, 2.0) if dimensions == 3 else (2.0, 2.0, 0.0)
    grid = Grid(resolution=8.0, cell_size=cell, dimensions=dimensions, courant=0.5,
                k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=tuple((4, 4) if grid.shape[axis] > 8 else (0, 0)
                                         for axis in range(3)))
    assert fields.h_separated is False, "a fresh Fields must not carry the start"
    return fields, pml


def _snapshot(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def _differing(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, int]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return {name: count for name in sorted(left)
            if (count := int(np.count_nonzero(_words(left[name]) != _words(right[name]))))}


def _array_steps(steps: int, start: bool = True) -> list:
    """The array path's own sub-steps; ``start=False`` is the forms alone (non-vacuity)."""
    from meep_gpu import stepping

    fields, pml = first_step_case()
    if not start:
        fields.h_separated = True
    out = []
    for _ in range(steps):
        stepping.step_B(fields, pml)
        stepping.update_H(fields, pml)
        stepping.step_D(fields, pml)
        stepping.update_E(fields, pml)
        out.append(_snapshot(fields))
    return out


def _composed_steps(fuse_labels: Sequence[str], steps: int,
                    strip_start: bool) -> Dict[str, Any]:
    """The composition the dispatch ladder builds, stepped slot by slot."""
    from meep_gpu import stepping
    from meep_gpu.metal_kernels import launch

    array_pass: Dict[str, Callable[[Any, Any], None]] = {
        "step_B": stepping.step_B, "update_H": stepping.update_H,
        "step_D": stepping.step_D, "update_E": stepping.update_E}
    fields, pml = first_step_case()
    residency = launch.Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=(),
                            fuse=bool(fuse_labels),
                            fuse_labels=tuple(fuse_labels) or None)
    launch.wrap_for_residency(plan, residency)
    occupant = plan.plans.get("update_H")
    carrier = getattr(launch, "StartOfHPlan", None)
    wrapped = carrier is not None and isinstance(occupant, carrier)
    if strip_start and wrapped:
        plan.plans["update_H"] = occupant.inner
    out = []
    for _ in range(steps):
        for name in ("step_B", "update_H", "step_D", "update_E"):
            entry = plan.plans.get(name)
            if entry is not None:
                entry.run()
            else:
                residency.sync_out()
                array_pass[name](fields, pml)
                residency.sync_in()
        residency.sync_out()
        out.append(_snapshot(fields))
    return {"snapshots": out, "selected": dict(plan.selected),
            "replaces": list(plan.replaces), "wrapped": wrapped,
            "after_launch": getattr(occupant, "after_launch", None),
            "starts": getattr(occupant, "starts", None) if wrapped else None,
            "h_separated_after": bool(fields.h_separated)}


def run_leg(fuse_labels: Sequence[str], expect_update_h: str,
            log: Optional[Callable[[str], None]] = None,
            steps: int = STEPS) -> Dict[str, Any]:
    """The shipped score and the planted revert on one composition. Raises on a failure."""
    say = log or (lambda message: print(message, flush=True))
    started = time.perf_counter()
    fields, pml = first_step_case()
    negative_kms = {axis: int(np.count_nonzero(np.asarray(getattr(pml, f"kms_{axis}")) < 0))
                    for axis in "xyz"}
    assert sum(negative_kms.values()), f"VACUOUS: no negative kms in this layer: {negative_kms}"
    want = _array_steps(steps)
    alone = _array_steps(1, start=False)[0]
    start_words = {name: count for name, count in _differing(want[0], alone).items()
                   if name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")}
    assert sum(start_words.values()) > 0, (
        f"VACUOUS: the start changes no H word on this case: {start_words}")
    negative_h = sum(int(np.count_nonzero(_words(want[0][c]) == NEGATIVE_ZERO))
                     for c in ("Hx", "Hy", "Hz"))

    shipped = _composed_steps(fuse_labels, steps, strip_start=False)
    assert shipped["selected"].get("update_H") == expect_update_h, shipped["selected"]
    # The WORDS are scored first, so a composition that does not carry the start fails
    # here on its differing words, not on the wrapper's absence.
    differing = [_differing(got, ref) for got, ref in zip(shipped["snapshots"], want)]
    per_step = [sum(row.values()) for row in differing]
    say(f"[first_update_h] {expect_update_h}: shipped differing words per step {per_step} "
        f"{differing[0] if per_step[0] else ''}".rstrip())
    assert all(count == 0 for count in per_step), (
        f"the composition differs from the array path on the run's first update_H: "
        f"{differing}")
    assert shipped["wrapped"], "the composer did not carry the start around update_H"
    shipped_ok = shipped["starts"] == 1
    say(f"[first_update_h] {expect_update_h}: starts={shipped['starts']}, "
        f"after_launch={shipped['after_launch']} ({time.perf_counter() - started:.1f}s)")

    reverted = _composed_steps(fuse_labels, steps, strip_start=True)
    caught = _differing(reverted["snapshots"][0], want[0])
    caught_ok = any(caught.get(c) for c in ("Hx", "Hy", "Hz"))
    say(f"[first_update_h] {expect_update_h}: revert (start stripped) differing at step 1 "
        f"{caught} ({time.perf_counter() - started:.1f}s)")
    row = {
        "fuse_labels": list(fuse_labels),
        "selected": shipped["selected"], "replaces": shipped["replaces"],
        "after_launch": shipped["after_launch"], "starts": shipped["starts"],
        "negative_kms_cells": negative_kms,
        "start_changes_h_words": start_words,
        "oracle_negative_zero_h_words_step1": negative_h,
        "shipped_differing_words_per_step": per_step,
        "revert_differing_words_step1": caught,
        "shipped_identical": shipped_ok, "revert_caught": caught_ok,
        "passed": bool(shipped_ok and caught_ok),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    assert shipped_ok, f"the shipped composition differs from the array path: {row}"
    assert caught_ok, f"the planted revert was not caught: {row}"
    return row
