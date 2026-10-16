#!/usr/bin/env python3
"""ABSENT COMPONENTS ON THE DISPATCH ROUTE: perfect-conductor 2-D runs, dispatched, word for word.

WHAT THIS MEASURES. MEEP allocates no array for a field component nothing requires
(a 2-D run's other polarization: ``fields::_require_component``, fields.cpp:552-572),
never steps it (``if (f[ec][cmp])``, update_eh.cpp:150) and reads it as 0.0
(monitor.cpp:160-164). The lift records that set (``FdtdDriver.set_absent_components``)
and the array path leaves those components at +0. Every kernel table's ``update_E``
steps all three electric components, so on a perfect-conductor cell, where MEEP stores
``chi1inv = -1e-20``, a dispatched ``update_E`` forms ``-1e-20 * (+0) = -0`` in the
absent component's E (or, inside the absorber, its ``f_w``). On a folded grid the same
product is -0 on the planes the D fills write with an odd parity, conductor or none:
the array path still steps the absent component's D, and ``fill_symmetry_bc_D`` /
``fill_folded_far_ghosts_D`` write those planes as ``parity * image``. ``fastpath``
clears both sets after every dispatched ``update_E`` (``fastpath.AbsentComponentClear``).
This gate
is the end-to-end measurement of that seam on whichever table this host's GPU serves:
the hand-CUDA and Triton tables on a CUDA host, the Metal table on an Apple GPU.

THE PROTOCOL, per case, in ONE process on ONE device: two legs lifted from identical
``mp.Simulation`` declarations with ``lift_simulation(prefer_gpu=True)``, byte-equal
before a step is taken::

    dispatch   MEEP_GPU_DISPATCH=1                      the claim
    array      MEEP_GPU_DISPATCH=1  MEEP_GPU_FUSED=0    the oracle (the kill switch)

stepped alternately along a x4 checkpoint ladder, every array on ``Fields`` and the PML
compared word for word at every checkpoint (``gate_dispatch_end_to_end.compare_state``),
and each differing word CLASSIFIED: in an absent electric component's E or ``f_w`` on a
cell whose inverse permittivity carries the sign bit, or on a plane a D fill writes for
the component (the two signatures of the defect this gate exists for), or anywhere else
(a different defect).

WHAT MAKES A PASS NON-VACUOUS, every clause a failure if false:

* the dispatch leg's ``active_step_path`` reads ``"fused"`` and ``update_E`` is among its
  dispatched slots, with a consult count equal to the steps taken, so the slot whose
  arithmetic is in question really ran from a kernel plan;
* the launch witness of every table the record says served (``composition.
  tables_dispatched``) counted launches: ``e2e.kernels_actually_ran`` on a CUDA host,
  the Metal route gate's two in-package counts on an Apple GPU;
* the array leg is the array path, refused at the kill switch, and launched nothing;
* the comparator sees a one-word perturbation of the state it called identical;
* every case carries absent electric components with conductor cells or fill planes
  under them, so the words in question exist (``absent_signbit_cells``,
  ``absent_fill_plane_cells``).

VERDICTS. ``PASS-DISPATCHED``: 0 differing words at every checkpoint with the clauses
above. ``ABSENT-WORDS-DIFFER``: every differing word is an absent component's E or
``f_w`` word on a sign-bit cell or a fill plane -- what a tree without the clear produces,
and the reproduction. ``DIVERGENCE``: a differing word anywhere else. ``CLEAR-REFUSED``: the
planner refused the plan because the clear could not be written, which takes dispatch
away from the run. ``NOT-EXERCISED``: the dispatch leg did not run ``update_E`` from a
kernel (the table has no arm for the case: complex storage, a slot no consulted product
admitted), so the case measured nothing about this seam. ``CONTROL-FAILURE``,
``COMPARATOR-BLIND``, ``HARNESS-FAILURE`` and ``ERROR`` as in the end-to-end gate.

EXIT STATUS 0 when no case reads a failing verdict (anything but ``PASS-DISPATCHED`` and
``NOT-EXERCISED``) and at least one case reads ``PASS-DISPATCHED``; 1 otherwise.
``NOT-EXERCISED`` is neither a pass nor a failure: it says which table cannot reach the
seam on that case, and its row carries the planner's reason.

Progress reporting: one flushed line per leg per checkpoint, and each case's row is
appended to ``cases.jsonl`` as it lands.

Run (one GPU; on a CUDA host pin the table with ``MEEP_GPU_KERNEL_TABLE``)::

    CUDA_VISIBLE_DEVICES=<idx> MEEP_GPU_KERNEL_TABLE=triton python -u \\
        parity/meep_gpu/gate_dispatch_absent_components.py --out <dir>
    MEEP_GPU_SUBNORMAL_POLICY=flush python -u \\
        parity/meep_gpu/gate_dispatch_absent_components.py --out <dir>      # Apple GPU

``--plan-only`` lifts the dispatch leg, takes one step and records the planner's verdict.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
# RESOLVED BY NAME, NEVER BY DEPTH, as the Metal route gate resolves it.
_API = HERE
while _API != os.path.dirname(_API) and not (
        os.path.isdir(os.path.join(_API, "meep_gpu"))
        and os.path.isdir(os.path.join(_API, "parity", "meep_gpu"))):
    _API = os.path.dirname(_API)
if not os.path.isdir(os.path.join(_API, "parity", "meep_gpu")):  # pragma: no cover
    raise SystemExit(f"cannot locate the repository root above {HERE}: no ancestor "
                     f"holds both meep_gpu/ and parity/meep_gpu/")
for _path in (_API, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_dispatch_end_to_end as e2e  # noqa: E402

E_COMPONENTS = ("Ex", "Ey", "Ez")

#: The text ``fastpath._finish`` refuses a plan with when the clear cannot be written,
#: read off the record by substring, as the other gates read their refusals.
CLEAR_REFUSAL = "the absent components' words cannot be cleared"

DISPATCH_ENV: Dict[str, Optional[str]] = {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None}
ARRAY_ENV: Dict[str, Optional[str]] = {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": "0"}


def say(message: str) -> None:
    e2e.say(message)


# ---------------------------------------------------------------------------
# The cases: 2-D single-polarization runs with perfect-conductor cells
# ---------------------------------------------------------------------------

def _metal_cavity(mp, res: int, *, component: Any, folded: bool):
    """python/examples/metal-cavity-ldos.py's cell at w = 0.2.

    A 4 x 4 cell with a 1.0 PML, an ``mp.metal`` box of side ``a + 2t`` (a = 1,
    t = 0.1) hollowed to side ``a`` and slit at ``x = a/2`` (height w), and a dipole at
    the centre with the example's ``fcen = sqrt(0.5)/a``, ``df = 0.2``. ``until`` runs
    past the pulse peak (``5 / df``), so the fields are normal-magnitude numbers and
    not the exactly-zero turn-on tail. The example's resolution is 50; 20 keeps two
    pixels of shell and four of slit.
    """
    a, t, w, dpml = 1.0, 0.1, 0.2, 1.0
    sxy = 2.0 + 2 * dpml
    geometry = [
        mp.Block(mp.Vector3(a + 2 * t, a + 2 * t, mp.inf), material=mp.metal),
        mp.Block(mp.Vector3(a, a, mp.inf), material=mp.air),
        mp.Block(center=mp.Vector3(a / 2), size=mp.Vector3(2 * t, w, mp.inf),
                 material=mp.air),
    ]
    fcen, df = math.sqrt(0.5) / a, 0.2
    sources = [mp.Source(src=mp.GaussianSource(fcen, fwidth=df), component=component,
                         center=mp.Vector3())]
    sim = mp.Simulation(cell_size=mp.Vector3(sxy, sxy), geometry=geometry,
                        boundary_layers=[mp.PML(dpml)], sources=sources,
                        symmetries=[mp.Mirror(mp.Y)] if folded else [],
                        resolution=res)
    flux = sim.add_flux(fcen, df, 5, mp.FluxRegion(center=mp.Vector3(1.2, 0),
                                                   size=mp.Vector3(0, 1.6)))
    return sim, [flux], 30.0


def case_metal_cavity_ldos(mp, res=20):
    """ex.metal-cavity-ldos's 2-D configuration as the example declares it: Ez, Mirror(Y)."""
    return _metal_cavity(mp, res, component=mp.Ez, folded=True)


def case_metal_cavity_unfolded(mp, res=20):
    """The same cavity without the mirror: the fold-free route every table can serve."""
    return _metal_cavity(mp, res, component=mp.Ez, folded=False)


def case_metal_cavity_te(mp, res=20):
    """The cavity driven by Hz: TE allocated, so Ez (and Dz, Hx, Hy, Bx, By) absent."""
    return _metal_cavity(mp, res, component=mp.Hz, folded=False)


def case_metal_wall_in_pml(mp, res=20):
    """Conductor strips that cross the absorber: the -0 lands in ``f_w``, not only in E."""
    sim = mp.Simulation(
        cell_size=mp.Vector3(6, 4), resolution=res, boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(mp.Vector3(0.4, mp.inf, mp.inf), center=mp.Vector3(1.0, 0),
                           material=mp.metal),
                  mp.Block(mp.Vector3(mp.inf, 0.4, mp.inf), center=mp.Vector3(0, -1.5),
                           material=mp.metal)],
        sources=[mp.Source(mp.GaussianSource(0.4, fwidth=0.3), component=mp.Ez,
                           center=mp.Vector3(-1.5, 0))])
    flux = sim.add_flux(0.4, 0.3, 5, mp.FluxRegion(center=mp.Vector3(-0.5, 0),
                                                   size=mp.Vector3(0, 2)))
    return sim, [flux], 25.0


def case_metal_dispersive(mp, res=20):
    """A conductor beside a Lorentzian slab under PML: stored E, ``update_P``, E -> P."""
    lorentz = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.4, gamma=0.02, sigma=1.1)])
    sim = mp.Simulation(
        cell_size=mp.Vector3(6, 4), resolution=res, boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(mp.Vector3(0.6, 1.6, mp.inf), center=mp.Vector3(1.2, 0),
                           material=mp.metal),
                  mp.Block(mp.Vector3(1.0, mp.inf, mp.inf), center=mp.Vector3(-0.5, 0),
                           material=lorentz)],
        sources=[mp.Source(mp.GaussianSource(0.4, fwidth=0.3), component=mp.Ez,
                           center=mp.Vector3(-1.8, 0))])
    return sim, [], 25.0


def case_metal_bloch(mp, res=20):
    """A conductor in a Bloch-periodic cell: complex64 storage beside the conductor."""
    sim = mp.Simulation(
        cell_size=mp.Vector3(2, 6), resolution=res,
        boundary_layers=[mp.PML(1.0, direction=mp.Y)],
        geometry=[mp.Block(mp.Vector3(0.6, 0.6, mp.inf), center=mp.Vector3(0, -0.6),
                           material=mp.metal)],
        sources=[mp.Source(mp.GaussianSource(0.25, fwidth=0.3), component=mp.Ez,
                           center=mp.Vector3(0, 0.9))],
        k_point=mp.Vector3(0.3, 0, 0))
    return sim, [], 25.0


def case_folded_periodic_dielectric(mp, res=20):
    """No conductor: a 2-D cell folded on two mirrors over periodic axes (k_point = 0).

    ``k_point = 0`` makes every axis periodic under the PML (``use_bloch``), so each
    folded axis stores MEEP's far ghost slot, which ``fill_folded_far_ghosts_D`` writes
    with the parity of a component normal to the plane, -1 on an even plane: Dx and Dy
    of the absent TE polarization hold -0 on their far planes from the first step, and
    every inverse permittivity is positive. The words in question are the fill planes
    alone (the shape of ``tests:TestGetPoint.test_get_point``, a folded lift row).
    """
    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 4), resolution=res, boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(mp.Vector3(1.0, 1.0, mp.inf), center=mp.Vector3(),
                           material=mp.Medium(epsilon=4.0))],
        sources=[mp.Source(mp.GaussianSource(0.4, fwidth=0.3), component=mp.Ez,
                           center=mp.Vector3())],
        symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y)], k_point=mp.Vector3())
    flux = sim.add_flux(0.4, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0.9, 0),
                                                   size=mp.Vector3(0, 1.2)))
    return sim, [flux], 25.0


def case_metal_no_pml(mp, res=20):
    """A conductor, no absorber, no susceptibility: E is derived, so ``update_E`` stores nothing.

    The control of the set: there is no absent word for a kernel to write, and the clear
    is built empty. The flux plane reads the absent components through the monitor
    readers (``dft._raw_factors`` under a held residency).
    """
    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 4), resolution=res, boundary_layers=[],
        geometry=[mp.Block(mp.Vector3(0.8, 0.8, mp.inf), center=mp.Vector3(0.6, 0.6),
                           material=mp.metal)],
        sources=[mp.Source(mp.GaussianSource(0.3, fwidth=0.3), component=mp.Ez,
                           center=mp.Vector3(-1.0, -1.0))])
    flux = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0.6, -0.4),
                                                   size=mp.Vector3(1.6, 0)))
    return sim, [flux], 25.0


CASES = {
    "metal_cavity_ldos": case_metal_cavity_ldos,
    "metal_cavity_unfolded": case_metal_cavity_unfolded,
    "metal_cavity_te": case_metal_cavity_te,
    "metal_wall_in_pml": case_metal_wall_in_pml,
    "metal_dispersive": case_metal_dispersive,
    "metal_bloch": case_metal_bloch,
    "folded_periodic_dielectric": case_folded_periodic_dielectric,
    "metal_no_pml": case_metal_no_pml,
}

#: Whether the case carries stored absent E words for a dispatched ``update_E`` to write.
#: ``metal_no_pml`` derives E, so its clear is built empty and ``update_E`` is not
#: required to dispatch there.
STORES_ABSENT_E = {name: name != "metal_no_pml" for name in CASES}

CASE_INTENT = {
    "metal_cavity_ldos": "the corpus row ex.metal-cavity-ldos, 2-D, folded on Y",
    "metal_cavity_unfolded": "the same cell unfolded: the plain PML route",
    "metal_cavity_te": "TE polarization: Ez absent instead of Ex, Ey",
    "metal_wall_in_pml": "conductor inside the absorber: absent f_w words",
    "metal_dispersive": "dispersive constitutive and update_P beside the conductor",
    "metal_bloch": "complex64 storage beside the conductor",
    "folded_periodic_dielectric": "no conductor; the D fills' far planes, folded on X and Y",
    "metal_no_pml": "derived E: nothing stored, nothing to clear",
}


# ---------------------------------------------------------------------------
# Classification of differing words
# ---------------------------------------------------------------------------

def absent_signbit_masks(driver: Any) -> Dict[str, numpy.ndarray]:
    """``component -> boolean cell mask`` of sign-bit inverse permittivity, absent E only."""
    from meep_gpu import to_numpy  # noqa: PLC0415

    fields = driver.fields
    masks: Dict[str, numpy.ndarray] = {}
    for component in getattr(fields, "absent_components", ()) or ():
        if component not in E_COMPONENTS:
            continue
        stored = getattr(fields, component, None)
        if stored is None:
            continue
        inverse = numpy.asarray(to_numpy(fields.inverse_epsilon_for(component)))
        masks[component] = numpy.ascontiguousarray(
            numpy.broadcast_to(numpy.signbit(inverse), tuple(stored.shape)))
    return masks


def absent_fill_plane_masks(driver: Any) -> Dict[str, numpy.ndarray]:
    """``component -> boolean cell mask`` of the planes the D fills write, absent E only.

    ``fastpath._absent_fill_planes``, the clear's own second set, so the classification
    and the clear cannot disagree about which cells it names.
    """
    from meep_gpu import fastpath  # noqa: PLC0415

    fields = driver.fields
    masks: Dict[str, numpy.ndarray] = {}
    for component in getattr(fields, "absent_components", ()) or ():
        if component not in E_COMPONENTS:
            continue
        stored = getattr(fields, component, None)
        if stored is None:
            continue
        shape = tuple(int(n) for n in stored.shape)
        mask = numpy.zeros(int(numpy.prod(shape)), dtype=bool)
        mask[numpy.asarray(fastpath._absent_fill_planes(  # noqa: SLF001
            fields, component, shape), dtype=numpy.int64)] = True
        masks[component] = mask.reshape(shape)
    return masks


def _absent_component_of(name: str, masks: Dict[str, numpy.ndarray]) -> Optional[str]:
    """The absent component an E or ``f_w`` state path belongs to, or ``None``."""
    leaf = name.rsplit(".", 1)[-1]
    for component in masks:
        if leaf in (component, "f_w_" + component):
            return component
    return None


def classify(left: Dict[str, numpy.ndarray], right: Dict[str, numpy.ndarray],
             masks: Dict[str, numpy.ndarray],
             planes: Optional[Dict[str, numpy.ndarray]] = None) -> Dict[str, Any]:
    """Every differing word, split into the defect's two signatures and everything else.

    A word on a cell that is both a sign-bit cell and a fill plane counts as sign-bit.
    """
    planes = planes or {}
    on_signbit = 0
    on_plane = 0
    absent_elsewhere = 0
    outside = 0
    arrays: Dict[str, Dict[str, int]] = {}
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            outside += max(int(a.size), int(b.size))
            arrays[name] = {"shape_or_dtype": 1}
            continue
        wa, wb = e2e._as_words(a), e2e._as_words(b)  # noqa: SLF001
        differ = wa != wb
        count = int(numpy.count_nonzero(differ))
        if not count:
            continue
        component = _absent_component_of(name, masks)
        if component is None or masks[component].shape != a.shape or wa.dtype != numpy.uint32:
            outside += count
            arrays[name] = {"words": count, "outside": count}
            continue
        cells = masks[component].reshape(-1)
        per_cell = wa.size // cells.size
        mask = numpy.repeat(cells, per_cell) if per_cell > 1 else cells
        plane_cells = planes.get(component)
        plane = (numpy.zeros_like(mask) if plane_cells is None else
                 (numpy.repeat(plane_cells.reshape(-1), per_cell) if per_cell > 1
                  else plane_cells.reshape(-1)))
        hit = int(numpy.count_nonzero(differ & mask))
        planed = int(numpy.count_nonzero(differ & plane & ~mask))
        on_signbit += hit
        on_plane += planed
        absent_elsewhere += count - hit - planed
        arrays[name] = {"words": count, "absent_signbit": hit, "absent_fill_plane": planed,
                        "absent_elsewhere": count - hit - planed}
    return {"words_differing": on_signbit + on_plane + absent_elsewhere + outside,
            "absent_signbit_words": on_signbit,
            "absent_fill_plane_words": on_plane,
            "absent_other_words": absent_elsewhere,
            "outside_absent_words": outside,
            "arrays": arrays}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

class _Host:
    """The launch witness and the leg clauses of this host's kernel tables."""

    def __init__(self, gpu: str) -> None:
        self.gpu = gpu
        self.metal_route = None
        if gpu == "metal":
            import gate_dispatch_metal_route as metal_route  # noqa: PLC0415

            self.metal_route = metal_route
            self.counter: Any = metal_route.MetalLaunchCounter()
            self.witnesses = {"metal": "KernelPlan.launches and CountingFunction "
                                       "wrappers (gate_dispatch_metal_route)"}
            return
        self.counter = e2e.LaunchCounters()
        self.witnesses = {}
        try:
            self.counter.triton.install()
            self.witnesses["triton"] = "JITFunction.run and launch_enter_hook"
        except Exception as exc:  # noqa: BLE001 - recorded, and the clause asks the served table
            self.witnesses["triton"] = f"not installed: {exc!r}"
        before = e2e._cuda_compile_entry()  # noqa: SLF001
        self.counter.cuda.install()
        live, why = e2e.cuda_witness_is_live(self.counter.cuda, before)
        self.witnesses["cuda"] = "compile-cache proxy and memo wrap" if live else f"not live: {why}"

    def step(self, case: str, leg: Dict[str, Any], steps: int) -> None:
        if self.metal_route is not None:
            self.metal_route.step_leg(case, leg, self.counter, steps)
        else:
            e2e.step_leg(case, leg, self.counter, steps)

    def kernels_ran(self, leg: Dict[str, Any]) -> Dict[str, Any]:
        if self.metal_route is None:
            return e2e.kernels_actually_ran(leg)
        failures: List[str] = []
        plan = leg["plan"]
        counters = plan.get("launch_counters") or {}
        if leg["active_step_path"] != "fused":
            failures.append(f"active_step_path reads {leg['active_step_path']!r}")
        for slot in plan.get("dispatched_slots", []):
            if not (counters.get(slot) or {}).get("dispatches"):
                failures.append(f"{slot} is recorded dispatched and its consult ran "
                                f"{(counters.get(slot) or {}).get('dispatches')} times")
        launches = leg.get("metal_launches") or {}
        if not launches.get("total"):
            failures.append("no Metal plan booked a launch (KernelPlan.launches)")
        if not launches.get("function_calls"):
            failures.append("no compiled Metal function was invoked (CountingFunction)")
        if leg.get("unwitnessed_owners"):
            failures.append(f"plans with no witnessed function table: "
                            f"{leg['unwitnessed_owners']}")
        served = list((plan.get("composition") or {}).get("tables_dispatched") or [])
        if served and served != ["metal"]:
            failures.append(f"the record says {served} served, not the Metal table")
        return {"kernels_ran": not failures, "failures": failures,
                "metal_launches": launches, "tables_dispatched": served}

    def array_leg_clean(self, leg: Dict[str, Any]) -> Dict[str, Any]:
        if self.metal_route is not None:
            return self.metal_route.array_leg_is_clean(leg)
        return e2e.killswitch_leg_is_clean(leg)

    def require(self, driver: Any, what: str) -> None:
        gpu = getattr(driver, "gpu", None)
        if gpu != self.gpu:
            raise SystemExit(f"{what}: lifted a {gpu!r} driver on a {self.gpu!r} host; every "
                             "leg is this host's GPU driver (lift_simulation(prefer_gpu=True))")


def update_e_evidence(leg: Dict[str, Any]) -> Dict[str, Any]:
    """Did ``update_E`` run from a kernel plan at every step of the dispatch leg?"""
    plan = leg["plan"]
    counters = plan.get("launch_counters") or {}
    entry = counters.get("update_E") or {}
    dispatched = "update_E" in (plan.get("dispatched_slots") or [])
    consults = entry.get("dispatches")
    return {"update_E_dispatched": dispatched,
            "update_E_arm": (plan.get("arms") or {}).get("update_E"),
            "update_E_consults": consults,
            "steps": leg.get("steps"),
            "every_step": bool(dispatched and consults == leg.get("steps")),
            "array_slot_reason": (plan.get("array_slot_reasons") or {}).get("update_E")}


def absent_block(driver: Any) -> Optional[Dict[str, Any]]:
    """The record's ``absent_components`` block (with the live ``clears_run``), if any."""
    report = driver.fast_path_report() or {}
    block = report.get("absent_components")
    return dict(block) if isinstance(block, dict) else None


def _lift(host: _Host, name: str, label: str, env: Dict[str, Optional[str]],
          gpu_id: int, res: Optional[int]) -> Dict[str, Any]:
    leg = e2e.run_leg(name, CASES[name], label, env, host.counter, gpu_id, res)
    host.require(leg["driver"], f"{name}/{label}")
    return leg


def run_case(host: _Host, name: str, gpu_id: int, res: Optional[int],
             max_steps: Optional[int]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"case": name, "intent": CASE_INTENT[name],
                           "stores_absent_e": STORES_ABSENT_E[name],
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    say(f"{name}: lifting both legs")
    a = _lift(host, name, "dispatch", DISPATCH_ENV, gpu_id, res)
    b = _lift(host, name, "array", ARRAY_ENV, gpu_id, res)
    row["grid_shape"] = a["grid_shape"]
    try:
        masks = absent_signbit_masks(a["driver"])
        planes = absent_fill_plane_masks(a["driver"])
        row["absent_components"] = list(a["driver"].fields.absent_components)
        row["absent_signbit_cells"] = {c: int(m.sum()) for c, m in masks.items()}
        row["absent_fill_plane_cells"] = {c: int(m.sum()) for c, m in planes.items()}
        pre = e2e.compare_state(e2e.collect_state(a["driver"]), e2e.collect_state(b["driver"]))
        row["precondition_identical"] = pre["identical"]
        if not pre["identical"]:
            row["verdict"] = "HARNESS-FAILURE"
            row["why"] = "the two legs were lifted to different initial states"
            return row
        say(f"{name}: precondition OK -- {pre['arrays']} arrays, {pre['words_compared']} "
            f"words; absent {row['absent_components']}, sign-bit cells "
            f"{row['absent_signbit_cells']}, fill-plane cells "
            f"{row['absent_fill_plane_cells']}")

        total = max(1, int(round(a["until"] / float(a["driver"].grid.dt))))
        if max_steps is not None:
            total = min(total, max_steps)
        ladder = e2e.checkpoint_schedule(total, None)
        row["total_steps"] = total
        row["checkpoints"] = []
        row["first_divergent_checkpoint"] = None
        taken = 0
        for point in ladder:
            host.step(name, a, point - taken)
            host.step(name, b, point - taken)
            taken = point
            left, right = e2e.collect_state(a["driver"]), e2e.collect_state(b["driver"])
            sorted_words = classify(left, right, masks, planes)
            entry = {"steps": point, "words_compared": int(sum(
                         e2e._as_words(v).size for v in left.values())),  # noqa: SLF001
                     **{k: v for k, v in sorted_words.items() if k != "arrays"},
                     "arrays": dict(list(sorted_words["arrays"].items())[:12])}
            row["checkpoints"].append(entry)
            if entry["words_differing"] and row["first_divergent_checkpoint"] is None:
                row["first_divergent_checkpoint"] = point
            say(f"{name}: checkpoint {point}/{total}: differing "
                f"{entry['words_differing']} / {entry['words_compared']} words "
                f"(absent sign-bit {entry['absent_signbit_words']}, absent fill-plane "
                f"{entry['absent_fill_plane_words']}, absent other "
                f"{entry['absent_other_words']}, outside {entry['outside_absent_words']})")

        finals = (e2e.collect_state(a["driver"]), e2e.collect_state(b["driver"]))
        row["state"] = e2e.compare_state(*finals)
        row["observables"] = e2e.compare_spectra(e2e.read_spectra(a["driver"]),
                                                 e2e.read_spectra(b["driver"]))
        row["comparator_control"] = e2e.comparator_negative_control(finals[0])
        row["dispatch_leg"] = a["plan"]
        row["array_leg"] = b["plan"]
        row["evidence"] = host.kernels_ran(a)
        row["update_E"] = update_e_evidence(a)
        row["absent_clear"] = absent_block(a["driver"])
        row["array_leg_control"] = host.array_leg_clean(b)
        row["steps"] = {"dispatch": a["steps"], "array": b["steps"]}
        row["wall_s"] = {"dispatch": a["wall_s"], "array": b["wall_s"]}
    finally:
        a["driver"].close()
        b["driver"].close()

    last = row["checkpoints"][-1]
    differing = sum(c["words_differing"] for c in row["checkpoints"])
    outside = sum(c["outside_absent_words"] + c["absent_other_words"]
                  for c in row["checkpoints"])
    spectra_equal = row["observables"]["identical"]
    control = row["comparator_control"]
    exercised = (a["active_step_path"] == "fused"
                 and (row["update_E"]["every_step"] or not STORES_ABSENT_E[name]))
    if row["steps"]["dispatch"] != row["steps"]["array"]:
        row["verdict"] = "HARNESS-FAILURE"
        row["why"] = f"the legs took {row['steps']} steps"
    elif not row["array_leg_control"]["clean"]:
        row["verdict"] = "CONTROL-FAILURE"
    elif CLEAR_REFUSAL in str(a["plan"].get("refused_because") or ""):
        row["verdict"] = "CLEAR-REFUSED"
        row["why"] = a["plan"].get("refused_because")
    elif not exercised and differing:
        row["verdict"] = "DIVERGENCE"
        row["why"] = "the dispatch leg ran no update_E kernel and still differs"
    elif not exercised:
        row["verdict"] = "NOT-EXERCISED"
        row["why"] = (f"path {a['active_step_path']!r}, update_E {row['update_E']}; "
                      f"refused_because {a['plan'].get('refused_because')!r}")
    elif not row["evidence"]["kernels_ran"]:
        row["verdict"] = "VACUOUS-PASS" if not differing else "DIVERGENCE"
    elif outside or not spectra_equal:
        row["verdict"] = "DIVERGENCE"
    elif differing:
        row["verdict"] = "ABSENT-WORDS-DIFFER"
    elif not (control.get("armed") and control.get("comparator_saw_it")
              and control.get("words_mismatched") == 1):
        row["verdict"] = "COMPARATOR-BLIND"
    elif STORES_ABSENT_E[name] and not (any(row["absent_signbit_cells"].values())
                                        or any(row["absent_fill_plane_cells"].values())):
        row["verdict"] = "NOT-EXERCISED"
        row["why"] = ("no absent electric component has a conductor cell or a fill plane "
                      "under it")
    else:
        row["verdict"] = "PASS-DISPATCHED"
    row["final_words_differing"] = last["words_differing"]
    return row


def plan_only_case(host: _Host, name: str, gpu_id: int, res: Optional[int]) -> Dict[str, Any]:
    """Lift the dispatch leg, take one step, and record the planner's verdict."""
    leg = _lift(host, name, "dispatch", DISPATCH_ENV, gpu_id, res)
    driver = leg["driver"]
    try:
        e2e._apply_env(leg)  # noqa: SLF001
        driver.step()
        return {"case": name, "intent": CASE_INTENT[name],
                "grid_shape": leg["grid_shape"],
                "absent_components": list(driver.fields.absent_components),
                "absent_signbit_cells": {c: int(m.sum()) for c, m
                                         in absent_signbit_masks(driver).items()},
                "absent_fill_plane_cells": {c: int(m.sum()) for c, m
                                            in absent_fill_plane_masks(driver).items()},
                "active_step_path": driver.active_step_path,
                "plan": e2e._plan_summary(driver.fast_path_report()),  # noqa: SLF001
                "absent_clear": absent_block(driver),
                "verdict": "PLAN-ONLY"}
    finally:
        driver.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="results directory (cases are appended)")
    parser.add_argument("--cases", default="all", help="comma-separated names, or 'all'")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap every case's step count (default: its own until)")
    parser.add_argument("--plan-only", action="store_true")
    arguments = parser.parse_args(argv)

    names = (list(CASES) if arguments.cases == "all"
             else [n.strip() for n in arguments.cases.split(",") if n.strip()])
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown cases {unknown}; known {sorted(CASES)}", file=sys.stderr)
        return 2
    from meep_gpu import backends  # noqa: PLC0415

    gpu = backends.available_gpu()
    if gpu not in ("cuda", "metal"):
        print(f"REFUSING: no GPU on this host (available_gpu() is {gpu!r}); every leg "
              "is a prefer_gpu=True driver", file=sys.stderr)
        return 2
    os.makedirs(arguments.out, exist_ok=True)
    e2e._PROGRESS_PATH = os.path.join(arguments.out, "progress.log")  # noqa: SLF001
    rows_path = os.path.join(arguments.out, "cases.jsonl")
    host = _Host(gpu)
    provenance = {"gpu": gpu, "witnesses": host.witnesses,
                  "environment": e2e.environment_snapshot(),
                  "policy": e2e._policy_stamp(),  # noqa: SLF001
                  "argv": sys.argv, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(arguments.out, "provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=1, default=str)
    say(f"gpu={gpu} witnesses={host.witnesses} cases={names}")

    rows = []
    for index, name in enumerate(names, 1):
        say(f"=== case {index}/{len(names)}: {name} ===")
        started = time.time()
        try:
            row = (plan_only_case(host, name, arguments.gpu_id, arguments.resolution)
                   if arguments.plan_only else
                   run_case(host, name, arguments.gpu_id, arguments.resolution,
                            arguments.max_steps))
        except BaseException as exc:  # noqa: BLE001 - a raising case is a recorded row
            import traceback  # noqa: PLC0415

            row = {"case": name, "verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:2000],
                   "traceback": traceback.format_exc()[-4000:]}
        row["case_wall_s"] = round(time.time() - started, 2)
        rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        say(f"=== case {index}/{len(names)}: {name} -> {row.get('verdict')} "
            f"({row['case_wall_s']} s) ===")

    verdicts = {row["case"]: row.get("verdict") for row in rows}
    summary = {"provenance": provenance, "verdicts": verdicts,
               "words": {row["case"]: [row.get("final_words_differing"),
                                       (row.get("checkpoints") or [{}])[-1].get("words_compared")]
                         for row in rows}}
    with open(os.path.join(arguments.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    say(f"SUMMARY {json.dumps(verdicts)}")
    if arguments.plan_only:
        return 0
    return exit_status(verdicts)


def exit_status(verdicts: Dict[str, Any]) -> int:
    """0 when nothing failed and something passed; ``NOT-EXERCISED`` is neither."""
    failing = [case for case, verdict in verdicts.items()
               if verdict not in ("PASS-DISPATCHED", "NOT-EXERCISED")]
    passed = [case for case, verdict in verdicts.items() if verdict == "PASS-DISPATCHED"]
    return 0 if passed and not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
