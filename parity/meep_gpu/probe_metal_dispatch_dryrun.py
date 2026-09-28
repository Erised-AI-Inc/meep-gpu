#!/usr/bin/env python3
"""THE DRY RUN: every Metal route-gate case driven through the real driver seam,
BEFORE the release rows that cite it are typed.

WHY IT EXISTS, AND IT IS AN ORDERING FACT RATHER THAN A CONVENIENCE. The Metal
release rows (``meep_gpu.metal_dispatch.METAL_RELEASED_FUSED_ARMS`` and
``METAL_FUSED_RELEASE_ARM_AXES``) are the only TYPED statements in the whole
release, and they have to be written BEFORE the campaign that proves them: the
campaign's own gate reads them back and refuses to release a leg that did not
drive every ``(arm, case)`` pair they name. Every source edit of a round must land
before the round's campaign starts, so a row discovered wrong afterwards is not
repointed — it is a second full Metal fleet recut (~73 minutes, 62 ledger rebinds)
and a second campaign. This probe is what makes that discovery cost ten minutes
instead.

WHAT IT MEASURES, and it is deliberately the same measurement the gate makes minus
the release machinery: for each case, one driver lifted by
``meep_gpu.lift_simulation(sim, prefer_gpu=False)`` runs with the SHIPPED Metal
composer's plans installed at ``driver._fast_path``, and a second driver lifted
from an identical declaration runs the array path; every stored array on ``Fields``
and the PML is byte-compared as uint32 at every rung of a checkpoint ladder.

WHY A SHIM AND NOT ``plan_fast_path``. The rung-3 branch that routes a NumPy engine
with an MPS device into ``meep_gpu.metal_dispatch`` is a separate edit batch on
``fastpath.py``; until it lands, ``plan_fast_path`` refuses this host at the backend
rung and there is nothing to measure. The shim is the same pattern
``probe_metal_complex_driver_step.MetalShim`` and ``gate_metal_fused_hd_pair.Shim``
already use, and it answers the driver's OWN consults — ``driver.step`` reads
``self._fast_path`` once per step and calls ``fast.dispatch(slot, fields)`` at seven
sites. The shipped planner is untouched. AFTER the fastpath batch lands this probe
keeps its value: it is the cheap re-runnable check before every future envelope
widening, where the rows are typed first again.

THE TWO FACTS THAT MAKE IT NON-VACUOUS, because a shim that quietly declined
everything would produce a perfectly green byte comparison — the oracle IS the
array path:

* every dispatch is COUNTED, both halves (answered True, answered False), and a
  case whose counts are empty is reported as MEASURED-NOTHING rather than as a
  pass;
* the comparator is armed per case (``gate_dispatch_end_to_end.comparator_negative
  _control``): one word of the state just declared identical is moved by one ULP
  and the comparison must report exactly that word.

THE POLICY. The Metal composer's own backend clause consults
``metal_kernels.subnormal.mps_policy_reasons``, and this arm64 host resolves
``keep`` (MEEP's ``set_zero_subnormals`` is the ``#if HAVE_IMMINTRIN_H`` no-op
here), so with nothing installed the composer selects NOTHING on any case. This
probe installs ``flush`` on ``("host", "mps")`` before it lifts anything — the host
through ``subnormal_policy``'s fenv lever, the device natively — which is exactly
what the dispatch ladder's rung 8bM does for itself.

Rule 7: one flushed line per case per rung, and ``cases.jsonl`` is appended as each
case lands, so an interrupted run keeps everything up to the failure.

Run (this Mac; MEEP and torch coexist in one process, KMP_DUPLICATE_LIB_OK=TRUE)::

    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_metal_dispatch_dryrun.py \\
        --out parity/meep_gpu/results/_scratch/metal_dryrun_<stamp> \\
        --cases bloch_2d,folded_complex_2d --max-steps 192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

# RESOLVED BY NAME, NEVER BY DEPTH. A ``parents[N]`` walk is correct only while
# this file stays at the depth it was written at, and a moved harness resolving a
# wrong root is a trap this campaign has already paid for once.
HERE = os.path.dirname(os.path.abspath(__file__))
# The root is the nearest ancestor that holds both the package and the harness
# (``meep_gpu/`` and ``parity/meep_gpu/``): the repository root of this layout,
# and the same directory in any tree the harness is copied into whole.
_API = HERE
while _API != os.path.dirname(_API) and not (
        os.path.isdir(os.path.join(_API, "meep_gpu"))
        and os.path.isdir(os.path.join(_API, "parity", "meep_gpu"))):
    _API = os.path.dirname(_API)
if not os.path.isdir(os.path.join(_API, "parity", "meep_gpu")):  # pragma: no cover
    raise SystemExit(f"cannot locate the repository root above {HERE}: no ancestor "
                     f"holds both meep_gpu/ and parity/meep_gpu/")
for path in (HERE, _API):
    if path not in sys.path:
        sys.path.insert(0, path)

import gate_dispatch_end_to_end as e2e  # noqa: E402  the byte comparator and the cases
import gate_dispatch_fused_route as route  # noqa: E402  the five extra cases + the ladder
import gate_dispatch_metal_route as metal_route  # noqa: E402  the Metal-only cases


def say(message: str) -> None:
    e2e.say(message)


# ---------------------------------------------------------------------------
# The cases, and what each one is here to answer
# ---------------------------------------------------------------------------

#: Every case the Metal route gate drives, plus its two controls. The BUILDERS are
#: the two dispatch gates' own, imported rather than re-declared: a route-gate case
#: is a NAME the release binds, and this probe's whole job is to measure the case
#: the gate will drive — not a lookalike of it.
#: THE METAL ROUTE GATE'S OWN CASE TABLE, not the shared one: the two off-diagonal
#: cases live in ``gate_dispatch_metal_route`` because that module is bound by the
#: Metal driver record alone (the shared module's bytes are bound by the CUDA
#: driver record too). The gate imports this probe LAZILY, inside a function, so
#: the module-level import here is not a cycle.
CASES: Dict[str, Any] = dict(metal_route.CASES)

#: The order the probe walks, and the names ``metal_dispatch.METAL_RELEASED_FUSED_ARMS``
#: spells. ``cylindrical`` rather than ``cylindrical_m0``: the Metal release cites
#: the end-to-end gate's Dcyl builder, which carries no flux monitor — so on that
#: case the observable comparison is vacuous and is reported as such rather than
#: counted.
METAL_CASES: Tuple[str, ...] = (
    "pml_2d", "magnetic_seam_2d", "conductive_2d", "dispersive_2d", "pml_1d",
    "pml_3d_diagonal", "folded_2d", "folded_dispersive_2d", "cylindrical",
    "special_kz_2d", "bloch_2d", "folded_complex_2d", "pml_3d", "no_pml_2d",
    "offdiag_2d", "folded_offdiag_2d", "folded_3d", "folded_dispersive_3d",
    "complex_nobloch_2d", "cylindrical_m1",
)

#: What each case is here to answer, so a case that stops exercising what it was
#: chosen for is visible in the artifact rather than silently redundant.
CASE_INTENT: Dict[str, str] = {
    "pml_2d": "both ordinary pairs, electric deposit on the D seam",
    "magnetic_seam_2d": "both ordinary pairs, magnetic deposit on the B seam",
    "conductive_2d": "the ordinary magnetic pair beside the conductive electric pair",
    "dispersive_2d": "the dispersive electric pair with a LIVE update_P slot",
    "pml_1d": "the ordinary pairs on one live axis",
    "pml_3d_diagonal": "the ordinary pairs in 3-D on an axis-aligned block",
    "folded_2d": "both folded pairs plus the mirror fill in both fill slots",
    "folded_dispersive_2d": "the folded pairs over a susceptibility, update_P live",
    "cylindrical": "the cylindrical m=0 electric pair on a Dcyl grid",
    "special_kz_2d": "the beta electric pair on real storage carrying a nonzero beta",
    "bloch_2d": "the complex pairs; selects NOTHING without the complex expansion probe",
    "folded_complex_2d": "the folded complex pairs; needs the folded-complex probe",
    "pml_3d": "THE ENVELOPE CONTROL (a sphere): off-diagonal chi1inv rows",
    "no_pml_2d": "THE NO-FUSED-ARM CONTROL: no absorber, so no pair is selected",
    "offdiag_2d": "the ordinary magnetic pair on a 2-D off-diagonal grid; the D "
                  "seam keeps the offdiag singles",
    "folded_offdiag_2d": "the folded magnetic pair on a folded off-diagonal grid, "
                         "mirror fill in both fill slots; the D seam keeps the "
                         "folded offdiag singles",
    "folded_3d": "both folded pairs in 3-D with the mirror fill in both fill slots",
    "folded_dispersive_3d": "the folded magnetic pair in 3-D over one pole; the D "
                            "seam keeps its singles",
    "complex_nobloch_2d": "both complex pairs under force_complex_fields with no "
                          "k_point; needs the complex expansion probe",
    "cylindrical_m1": "the cylindrical complex electric pair at m = 1; needs the "
                      "cylindrical-complex expansion probe",
}


# ---------------------------------------------------------------------------
# The shim
# ---------------------------------------------------------------------------

class DryRunShim:
    """A ``FastPathPlan``-shaped object that answers the driver's seven consults.

    THE CONTRACT IS ``fastpath.FastPathPlan.dispatch``'S, reproduced here rather
    than imported because importing it would need the plan dataclass and the whole
    ladder that builds it. The three clauses that matter for correctness are the
    ones copied verbatim in behaviour:

    * THE IDENTITY GUARD — a consult whose ``fields`` is not the object the plans
      were built from is DECLINED, because the plans hold device mirrors bound to
      those arrays by identity;
    * ``update_P`` TAKES NO DRIVE FIELD ON THIS TRACK. The Triton slot holds a LIST
      of per-susceptibility plans each taking ``fields.drive_field``; the Metal
      composer fills it with ONE ``MetalAdeUpdatePPlan`` whose ``run()`` takes a
      contract. Handing it a drive field is a ``KeyError`` on the dispatch path,
      which is where exceptions PROPAGATE — measured, and it is the load-bearing
      half of the fastpath batch's ``dispatch`` edit;
    * THE SPLIT FILL — a fill plan that declares ``run_near`` launches its near pass
      at the near consult and its far pass at the driver's own far site, and one
      that does not (``MirrorGhostFillPlan``) fuses both into the near launch and
      answers the far consult True having launched nothing.

    ``run_near``/``run_far`` ARE REACHED THROUGH ``hasattr``, which is why
    ``launch.SyncedPlan`` answers them through ``__getattr__``: a wrapper defining
    them as real methods would make ``hasattr`` True for a fill plan that has none.
    """

    __slots__ = ("fields", "plans", "dispatched", "declined", "far_dispatched",
                 "residency")

    def __init__(self, fields: Any, plans: Dict[str, Any], residency: Any) -> None:
        self.fields = fields
        self.plans = {name: plan for name, plan in plans.items() if plan is not None}
        self.residency = residency
        self.dispatched: Dict[str, int] = {}
        self.declined: Dict[str, int] = {}
        self.far_dispatched: Dict[str, int] = {}

    def dispatch(self, slot: str, fields: Any) -> bool:
        from meep_gpu import fastpath as fp  # noqa: PLC0415 - late, after the policy

        if slot in fp.FAR_FILL_PASSES.values() or slot in (
                "fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D"):
            return self._far(slot, fields)
        if slot == fp.SYNC_UPDATE_H_PASS:
            # The second consult site's by-name channel. A pair spanning past the
            # magnetic half-step must decline it; a magnetic-half pair must not.
            owner = "update_H"
            plan = self.plans.get(owner)
            if plan is None:
                self.declined[slot] = self.declined.get(slot, 0) + 1
                return False
            slot = owner
        plan = self.plans.get(slot)
        if plan is None or fields is not self.fields:
            self.declined[slot] = self.declined.get(slot, 0) + 1
            return False
        if slot in ("fill_B", "fill_D") and hasattr(plan, "run_near"):
            plan.run_near()
        else:
            plan.run()
        self.dispatched[slot] = self.dispatched.get(slot, 0) + 1
        return True

    def _far(self, slot: str, fields: Any) -> bool:
        owner = "fill_B" if slot.endswith("_B") else "fill_D"
        plan = self.plans.get(owner)
        if plan is None or fields is not self.fields:
            self.declined[slot] = self.declined.get(slot, 0) + 1
            return False
        runner = getattr(plan, "run_far", None)
        if runner is None:
            # The near launch already wrote the far planes; nothing left to run.
            self.far_dispatched[slot] = self.far_dispatched.get(slot, 0) + 1
            return True
        runner()
        self.far_dispatched[slot] = self.far_dispatched.get(slot, 0) + 1
        return True

    @property
    def plan_launches(self) -> int:
        """What the PLANS counted, which is a different witness from the shim's tallies.

        The shim counts consults it answered True to; each ``KernelPlan`` counts its
        own ``run``. Two counts of one event, and a shim reporting a dispatch it did
        not perform disagrees with the plans.
        """
        from meep_gpu.metal_kernels import launch  # noqa: PLC0415

        seen: List[int] = []
        total = 0
        for plan in self.plans.values():
            owner = launch.declaring_plan(plan)
            if id(owner) in seen:
                continue
            seen.append(id(owner))
            total += int(getattr(owner, "launches", 0) or 0)
        return total


# ---------------------------------------------------------------------------
# The composition
# ---------------------------------------------------------------------------

def compose_metal_step(driver: Any, sources: Any, fuse: bool = True,
                       fuse_labels: Optional[Sequence[str]] = None
                       ) -> Tuple[Any, Any, Dict[str, Any]]:
    """Plan the step TWICE and install the residency bracket. The ladder's rung 5M.

    THE SECOND PASS IS NOT REDUNDANT and the derivation of the synced set is the
    load-bearing half. ``gate_metal_whole_step.covered_passes`` states the rule: a
    FILL plan occupies ONE slot and may replace TWO passes, because the driver runs
    the near symmetry fill and the folded far-ghost fill either side of the wall
    clear. "Live minus the slots the composer filled" would declare the far pass
    array-path, bracket a pass that never ran on the host, and report a composition
    weaker than the one executed.

    The composition window declares ``flush`` for the composer, exactly as
    ``metal_dispatch._composition_window`` does: ``mps_policy_reasons`` gates the
    whole Metal backend clause and cannot read an intention that has not been
    installed yet.

    ``fuse_labels`` IS THE RELEASE'S OWN SHAPE and ``None`` is not it. The ladder
    hands the composer the ADMITTED label set, so a product the release does not
    cover is never installed and its seam keeps the separate certified arms. With
    ``None`` — every product offered — this probe measures a composition WIDER than
    the one the campaign will drive, which is the right default for a dry run (it
    answers "is this product correct at the seam at all") and the wrong one for
    modelling a release row. ``--fuse-labels`` is how the second question is asked,
    and MEASURED 2026-09-10 it is a different composition on two cases: with every
    product offered, ``bloch_2d`` fuses both complex pairs and ``folded_complex_2d``
    both folded complex pairs, while the release admits only the magnetic one of
    each.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy  # noqa: PLC0415
    from meep_gpu.metal_kernels import device, launch  # noqa: PLC0415

    fields, pml = driver.fields, driver.pml
    residency = device.Residency()
    with declaring_run_policy("flush"):
        scout = launch.plan_step(fields, pml, residency=device.Residency(),
                                 sources=sources, fuse=fuse,
                                 fuse_labels=fuse_labels)
        covered = set()
        for slot, product in scout.plans.items():
            covered.add(slot)
            covered.update(getattr(launch.declaring_plan(product),
                                   "replaces_sub_steps", ()) or ())
        live = launch.live_sub_steps(fields, pml, sources) or ()
        synced = tuple(name for name in live if name not in covered)
        plan = launch.plan_step(fields, pml, residency=residency, sources=sources,
                                synced=synced, fuse=fuse, fuse_labels=fuse_labels)
    launch.wrap_for_residency(plan, residency)
    detail = {
        "fuse": bool(fuse),
        "fuse_labels": None if fuse_labels is None else list(fuse_labels),
        "selected": dict(getattr(plan, "selected", {}) or {}),
        "filled": sorted(plan.plans),
        "live": list(live),
        "synced": list(synced),
        "scouted_passes": sorted(covered),
        "mirrors": len(residency.names),
        "reasons": {name: list(value) for name, value
                    in sorted((getattr(plan, "reasons", {}) or {}).items())},
    }
    return plan, residency, detail


def install_shim(driver: Any, sources: Any, fuse: bool = True,
                 fuse_labels: Optional[Sequence[str]] = None
                 ) -> Tuple[Optional[DryRunShim], Dict[str, Any]]:
    """Compose and seat the shim in the driver's own freeze slot.

    ``_fast_path_stale`` is cleared so the driver does not re-plan over it on the
    next step; that flag and ``_fast_path`` are the engine's freeze slot, and this
    is the same seat every Metal composition probe in this tree uses.
    """
    plan, residency, detail = compose_metal_step(driver, sources, fuse=fuse,
                                                 fuse_labels=fuse_labels)
    if not plan.plans:
        driver._fast_path = None            # noqa: SLF001 - the engine's freeze slot
        driver._fast_path_stale = False     # noqa: SLF001
        return None, detail
    shim = DryRunShim(driver.fields, dict(plan.plans), residency)
    driver._fast_path = shim                # noqa: SLF001
    driver._fast_path_stale = False         # noqa: SLF001
    return shim, detail


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def _lift(case: str, res: Optional[int]) -> Tuple[Any, Any, float]:
    import meep as mp  # noqa: PLC0415
    import meep_gpu  # noqa: PLC0415

    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    builder = CASES[case]
    sim, monitors, until = builder(mp) if res is None else builder(mp, res)
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
    return driver, monitors, until


def run_case(case: str, res: Optional[int], max_steps: Optional[int],
             checkpoints: Optional[str], fuse: bool = True,
             fuse_labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case": case, "intent": CASE_INTENT.get(case, "?"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # THE ENABLE IS IRRELEVANT HERE and the environment is pinned to say so: the
    # shim is seated directly, so no ladder rung is consulted on either leg and a
    # stale switch from a sibling process cannot change what this measures.
    for name in ("MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED", "MEEP_GPU_FUSE_ARMS"):
        os.environ.pop(name, None)

    started = time.time()
    fused_driver, _monitors, until = _lift(case, res)
    array_driver, _array_monitors, _until = _lift(case, res)
    row["grid_shape"] = [int(v) for v in fused_driver.shape]
    row["lift_s"] = round(time.time() - started, 2)
    say(f"{case}: lifted {tuple(row['grid_shape'])} x2 ({row['lift_s']} s)")

    pre = e2e.compare_state(e2e.collect_state(fused_driver),
                            e2e.collect_state(array_driver))
    row["precondition"] = {"identical": pre["identical"], "arrays": pre["arrays"],
                           "words": pre["words_compared"]}
    if not pre["identical"]:
        row["verdict"] = "HARNESS-FAILURE"
        row["why"] = ("the two legs were lifted to DIFFERENT initial states, so no "
                      "downstream comparison is about the composition")
        fused_driver.close()
        array_driver.close()
        return row

    shim, detail = install_shim(fused_driver, fused_driver._sources,  # noqa: SLF001
                                fuse=fuse, fuse_labels=fuse_labels)
    row["composition"] = detail
    row["fused_slots"] = sorted(shim.plans) if shim else []
    if shim is None:
        row["verdict"] = "SELECTED-NOTHING"
        row["why"] = ("the shipped composer filled no slot on this configuration; "
                      "the refusals are recorded under composition.reasons")
        say(f"{case}: SELECTED NOTHING -- "
            f"{sorted(detail['reasons'])[:3]}")
        fused_driver.close()
        array_driver.close()
        return row
    say(f"{case}: composed {len(shim.plans)} slots "
        f"{ {s: detail['selected'].get(s) for s in sorted(shim.plans)} } "
        f"mirrors={detail['mirrors']} synced={detail['synced']}")

    dt = float(fused_driver.grid.dt)
    total = max(1, int(round(until / dt)))
    if max_steps is not None:
        total = min(total, max_steps)
    ladder = (route.ladder_for(total) if checkpoints is None
              else sorted({min(total, int(v)) for v in checkpoints.split(",")
                           if v.strip()}))
    row["total_steps"] = total
    row["checkpoints_planned"] = ladder
    row["checkpoints"] = []
    row["first_divergent_checkpoint"] = None

    taken = 0
    for point in ladder:
        chunk = point - taken
        fused_driver.run(num_steps=chunk)
        array_driver.run(num_steps=chunk)
        taken = point
        verdict = e2e.compare_state(e2e.collect_state(fused_driver),
                                    e2e.collect_state(array_driver))
        entry = {"steps": point, "identical": verdict["identical"],
                 "arrays": verdict["arrays"], "words": verdict["words_compared"],
                 "arrays_differing": verdict["arrays_differing"],
                 "differences": verdict["differences"][:4]}
        row["checkpoints"].append(entry)
        outcome = ("IDENTICAL" if verdict["identical"]
                   else f"DIFFERS in {verdict['arrays_differing']} arrays")
        say(f"{case}: checkpoint {point}/{total} -> {outcome} "
            f"({verdict['arrays']} arrays, {verdict['words_compared']} words)")
        if not verdict["identical"] and row["first_divergent_checkpoint"] is None:
            row["first_divergent_checkpoint"] = point
            row["first_divergence"] = entry

    finals = e2e.collect_state(fused_driver)
    row["final_state"] = e2e.compare_state(finals, e2e.collect_state(array_driver))
    row["observables"] = e2e.compare_spectra(e2e.read_spectra(fused_driver),
                                             e2e.read_spectra(array_driver))
    row["comparator_control"] = e2e.comparator_negative_control(finals)
    row["evidence"] = {
        "dispatched": dict(sorted(shim.dispatched.items())),
        "far_dispatched": dict(sorted(shim.far_dispatched.items())),
        "declined": dict(sorted(shim.declined.items())),
        "plan_launches": shim.plan_launches,
        "residency_syncs": {"in": shim.residency.syncs_in,
                            "out": shim.residency.syncs_out},
        "mirrors": len(shim.residency.names),
    }
    # THE RESIDENCY VERDICT, read after the last sync_out. It is not consulted for
    # correctness in per-launch-sync mode -- the bracket is what makes the host
    # authoritative -- so it is a VACUITY floor rather than a claim: a non-empty
    # registry whose mirrors all agree with their hosts.
    try:
        row["residency_verify"] = dict(shim.residency.verify())
    except Exception as exc:  # noqa: BLE001 - an unreadable verdict is recorded
        row["residency_verify"] = {"unreadable": repr(exc)}

    fused_driver.close()
    array_driver.close()

    agree = (row["final_state"]["identical"] and row["observables"]["identical"]
             and row["first_divergent_checkpoint"] is None)
    control = row["comparator_control"]
    ran = sum(shim.dispatched.values())
    if not ran:
        row["verdict"] = "MEASURED-NOTHING"
        row["why"] = ("the shim answered no consult, so the byte comparison is the "
                      "array path against itself")
    elif not (control.get("armed") and control.get("comparator_saw_it")
              and control.get("words_mismatched") == 1):
        row["verdict"] = "COMPARATOR-BLIND"
        row["why"] = ("the legs matched, but the comparator did not report a "
                      "one-ULP perturbation of the state it just compared")
    elif not agree:
        row["verdict"] = "DIVERGENCE"
    else:
        row["verdict"] = "IDENTICAL"
    say(f"{case}: {row['verdict']} :: dispatched={row['evidence']['dispatched']} "
        f"plan_launches={row['evidence']['plan_launches']} "
        f"syncs={row['evidence']['residency_syncs']}")
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _provenance(policy: Dict[str, Any]) -> Dict[str, Any]:
    import hashlib  # noqa: PLC0415
    import platform  # noqa: PLC0415

    package = os.path.join(_API, "meep_gpu")
    digests: Dict[str, str] = {}
    for name in ("fastpath.py", "driver.py", "fields.py", "metal_dispatch.py",
                 "subnormal_policy.py", "metal_kernels/launch.py",
                 "metal_kernels/device.py", "metal_kernels/subnormal.py"):
        path = os.path.join(package, name)
        try:
            with open(path, "rb") as handle:
                digests[name] = hashlib.sha256(handle.read()).hexdigest()
        except OSError as exc:
            digests[name] = f"unreadable: {exc!r}"
    record: Dict[str, Any] = {
        "probe": "metal_dispatch_dryrun",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "machine": platform.machine(),
        "python": sys.version.split()[0],
        "source_sha256": digests,
        "this_script_sha256": hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "installed_policy": policy,
        "expansion_probes": {
            name: os.environ.get(name) for name in (
                "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
                "MEEP_GPU_METAL_EXPANSION_PROBE",
                "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
                "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE")},
    }
    for module in ("meep", "numpy", "torch"):
        try:
            record[module] = __import__(module).__version__
        except Exception as exc:  # noqa: BLE001
            record[module] = f"unavailable: {exc!r}"
    try:
        from meep_gpu.metal_kernels import device, subnormal  # noqa: PLC0415

        record["metal_frontend"] = device.metal_frontend_version()
        record["mps_policy_report"] = subnormal.mps_policy_report()
    except Exception as exc:  # noqa: BLE001
        record["metal_frontend"] = f"unreadable: {exc!r}"
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="results directory")
    parser.add_argument("--cases", default="all",
                        help=f"comma-separated case names, or 'all' "
                             f"({', '.join(METAL_CASES)})")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap the ladder; the case's own budget is the default")
    parser.add_argument("--checkpoints", default=None,
                        help="comma-separated CUMULATIVE step counts to compare at")
    parser.add_argument("--fuse-labels", default=None,
                        help="comma-separated fused labels to OFFER the composer, "
                             "which is the release's own shape: a product outside "
                             "the offer is never installed and its seam keeps the "
                             "separate certified arms. Unset offers every product.")
    parser.add_argument("--no-fuse", action="store_true",
                        help="compose with fuse=False: the separate certified arms "
                             "through the same seam, which is the substitution "
                             "baseline rather than the claim")
    arguments = parser.parse_args(argv)

    os.makedirs(arguments.out, exist_ok=True)
    e2e._PROGRESS_PATH = os.path.join(arguments.out, "progress.log")  # noqa: SLF001
    e2e.PREFER_GPU = False
    rows_path = os.path.join(arguments.out, "cases.jsonl")

    offered = (None if arguments.fuse_labels is None
               else tuple(v.strip() for v in arguments.fuse_labels.split(",")
                          if v.strip()))
    names = (list(METAL_CASES) if arguments.cases == "all"
             else [n.strip() for n in arguments.cases.split(",") if n.strip()])
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown cases: {unknown}; known: {sorted(CASES)}", file=sys.stderr)
        return 2

    # THE POLICY FIRST, AND MEEP BEFORE IT. ``install_host_policy`` drives the host
    # FPU through MEEP's own ``set_zero_subnormals`` before falling back to the fenv
    # lever, and a request it cannot verify is reported UNATTAINABLE -- so a run
    # that imports MEEP afterwards would be measuring a policy it never established.
    import meep  # noqa: PLC0415,F401
    import meep_gpu  # noqa: PLC0415

    try:
        policy = dict(meep_gpu.install_subnormal_policy(
            "flush", executors=("host", "mps")))
        say(f"installed the flush policy on host + mps: "
            f"{ {k: policy.get(k) for k in ('policy', 'attained', 'executors')} }")
    except BaseException as exc:  # noqa: BLE001 - a refusal is the whole answer
        say(f"REFUSING: install_subnormal_policy('flush', executors=('host','mps')) "
            f"raised {exc!r}; the Metal composer refuses every arm under a resolved "
            f"'keep' and this probe would measure nothing")
        return 4

    provenance = _provenance(policy)
    with open(os.path.join(arguments.out, "provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=1, default=str)
    say(f"host={provenance['host']} torch={provenance.get('torch')} "
        f"meep={provenance.get('meep')} frontend={provenance.get('metal_frontend')}")

    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(names, 1):
        say(f"=== case {index}/{len(names)}: {name} ===")
        started = time.time()
        try:
            row = run_case(name, arguments.resolution, arguments.max_steps,
                           arguments.checkpoints, fuse=not arguments.no_fuse,
                           fuse_labels=offered)
        except BaseException as exc:  # noqa: BLE001 - a raising case is a recorded row
            row = {"case": name, "verdict": "ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:2000],
                   "traceback": traceback.format_exc()[-4000:]}
        row["case_wall_s"] = round(time.time() - started, 2)
        rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as handle:  # rule 7, incremental
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
        say(f"=== case {index}/{len(names)}: {name} -> {row.get('verdict')} "
            f"({row['case_wall_s']} s) ===")

    summary = {
        "probe": "metal_dispatch_dryrun",
        "purpose": ("drive the shipped Metal composer's fused products through the "
                    "real driver.step consults and byte-compare against the array "
                    "path, BEFORE the release rows that cite these cases are typed"),
        "provenance": provenance,
        "fuse": not arguments.no_fuse,
        "fuse_labels_offered": None if offered is None else list(offered),
        "verdicts": {r["case"]: r.get("verdict") for r in rows},
        "first_divergent_checkpoint": {r["case"]: r.get("first_divergent_checkpoint")
                                       for r in rows},
        "selected": {r["case"]: (r.get("composition") or {}).get("selected")
                     for r in rows},
        "dispatched": {r["case"]: (r.get("evidence") or {}).get("dispatched")
                       for r in rows},
        "identical_cases": [r["case"] for r in rows if r.get("verdict") == "IDENTICAL"],
        "selected_nothing": [r["case"] for r in rows
                             if r.get("verdict") == "SELECTED-NOTHING"],
        "failures": [r["case"] for r in rows
                     if r.get("verdict") not in ("IDENTICAL", "SELECTED-NOTHING")],
        "what_this_licenses": (
            "that the shipped Metal composer's products, installed at the driver's "
            "own freeze slot with the residency bracket, compute the array path's "
            "bytes on the cases listed IDENTICAL. It licenses NOTHING about the "
            "dispatch ladder, the release table, or a case it did not run: it is "
            "the pre-flight for typing the release rows, not the gate that proves "
            "them."),
        "rows": rows,
    }
    with open(os.path.join(arguments.out, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    say(f"SUMMARY {json.dumps(summary['verdicts'])}")
    if summary["failures"]:
        say(f"FAILURES: {summary['failures']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
