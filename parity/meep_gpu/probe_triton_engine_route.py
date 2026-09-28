"""Close the loop: the ENGINE plan route, on a real lifted simulation.

The shared gate (``probe_fused_kernel_bit_identity.py --track triton``) drives
``plan_from_arrays`` — bare device arrays the harness constructed. That is the
right thing for a gate, because it can then build the discriminating boundary and
coefficient combinations directly. But it leaves one seam untested: the route the
ENGINE would take, ``plan_pml_curl(fields, pml, sub_step)``, which resolves the
boundary kinds from the Grid, picks the Yee sub-lattice from the sub-step, reads
``grid.dt / grid.dx`` for the Courant and passes through the coverage predicate.

Every one of those is a place a silent half-cell or wrong-wrap error lives, and
none of them is exercised by handing the kernel a table the harness chose. So this
lifts a REAL ``mp.Simulation`` from the benchmark's own case list, steps it with
``stepping.step_B``/``step_D`` and with the plan, and compares bytes — target and
auxiliary, over several consecutive sub-steps.

It is a small probe on purpose. The gate is the gate; this only asserts that the
thing the gate certified is the thing the engine would build.

WHAT IT RELEASES ON, added 2026-08-31, and it had none before that. The artifact
stated ``probe``/``cases``/``summary`` and no verdict in any of the seven spellings
``gate_provenance.read_verdict`` knows, so a run that passed and a run nobody could
read were the same artifact, and ``recut_composition_records.released()`` scored it
``no verdict key`` and refused to bind a weld from it. It now writes
``summary.status`` — the spelling four of its sibling composition probes use — and
stamps through ``gate_provenance``, which normalises that into ``canonical_verdict``
and records the sha256 of every repo module the process imported.

The release rests on the ARITHMETIC and on non-vacuity: every covered case
bit-identical to the array path on both curl sub-steps over ``STEPS``; every composed
case bit-identical over ``WHOLE_STEPS`` complete driver steps with every state array
compared as uint32; the builder agreeing with its own named-sub-step predicate on
every case; no case erroring; and at least one case on each side of the coverage
boundary, so a run whose case list failed to lift cannot report 0/0 and release.

``refusals_returned_none`` IS RECORDED AND IS NOT A RELEASE CLAUSE. It counts the
cases where the plain PML curl builder declines BOTH curls, and since the conductive
PML family landed it is 3/4 by design: ``2d_cond_pml`` carries a D-side conductivity
only, so ``pml_curl_coverage`` refuses it in AGGREGATE and admits its ``step_B``, and
the composition then serves all four of its sub-steps byte-identically. Reading that
as a failure would refuse a run in which nothing is wrong; see ``one_case``, where the
measurement is written out.

Usage::

    TRITON_LIBCUDA_PATH=$HOME/triton_libcuda_stub CUDA_VISIBLE_DEVICES=<free> \
    PYTHONPATH=<repo> python -u probe_triton_engine_route.py \
        --out results/triton_engine_route.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# ``cases.py`` is the benchmark's case list, byte-identical in both result dirs;
# whichever copy is present is the same file (checked by md5 when it was copied).
for path in (_REPO_ROOT,
             # `gate_provenance` sits beside this file. Direct execution already puts
             # this directory on sys.path[0]; naming it means an IMPORTING caller (the
             # campaign shell drives some probes that way) gets the stamp too, rather
             # than an ImportError inside save().
             _HERE,
             os.path.join(_HERE, "results", "fused_pml_throughput_triton_2026-08-09",
                          "scripts"),
             os.path.join(_HERE, "results", "cupy_throughput_2026-08-09", "scripts")):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

STEPS = 6
#: How many whole ``FdtdDriver.step`` calls the composed leg compares.
WHOLE_STEPS = 6
#: Covered configurations from the benchmark's own case list, plus the refusals
#: that must return None rather than a plan.
#:
#: ``2d_dispersive`` MOVED HERE from the refusal list by the composability pass.
#: The curl refused any registered susceptibility until this round; it no longer
#: does, because the curl differences the STORED E and H and never reads a
#: polarization. It is the case the widening exists for, and it is the one this
#: probe has to show taking the path rather than being argued into it.
COVERED_CASES = (("2d_pml", {"res": 40, "n": 16}),
                 ("3d_pml", {"res": 15, "n": 6}),
                 ("2d_pml_geom", {"res": 40, "n": 16}),
                 ("2d_dispersive", {"res": 40, "n": 16}))
REFUSAL_CASES = (("2d_plain", {"res": 40, "n": 16}),
                 ("2d_symmetry", {"res": 40, "n": 16}),
                 ("2d_cond_pml", {"res": 40, "n": 16}),
                 ("cylindrical", {"res": 80, "r": 8.0, "z": 8.0}))
#: Every array a whole step may write, plus each pole's P and P_prev. A composed
#: gate that compares the primaries only is blind to a dropped auxiliary store —
#: measured at 120/120 uncaught on the curl (plan §12.3).
WHOLE_STEP_ARRAYS = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)


def log(message: str) -> None:
    print(message, flush=True)


def bit_compare(a_dev, b_dev) -> Dict[str, Any]:
    import cupy as cp  # noqa: PLC0415

    a = np.ascontiguousarray(cp.asnumpy(a_dev)).ravel()
    b = np.ascontiguousarray(cp.asnumpy(b_dev)).ravel()
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    return {"bit_identical": bool(np.array_equal(ua, ub)),
            "differing_floats": int(np.count_nonzero(ua != ub)),
            "total_floats": int(ua.size)}


def whole_step_state(driver, cp) -> Dict[str, Any]:
    """Every array the step may have written, including each pole's P and P_prev."""
    state = {name: getattr(driver.fields, name) for name in WHOLE_STEP_ARRAYS
             if getattr(driver.fields, name, None) is not None}
    for index, pole in enumerate(driver.fields.polarizations):
        for component in pole.driven():
            state[f"P{index}_{component}"] = pole.P[component]
            state[f"Pprev{index}_{component}"] = pole.P_prev[component]
    return state


def composed_whole_step(name: str, kwargs: Dict[str, Any], cp) -> Dict[str, Any]:
    """N whole ``FdtdDriver.step`` calls, array path vs the composed Triton plan.

    Two drivers lifted from the SAME ``mp.Simulation`` builder, one untouched and
    one whose covered sub-steps are served by :func:`plan_step`. Every state array
    is compared as bytes after every step, sources and monitors included, because
    this is the leg that has to show the composition surviving a real script's
    step order rather than a harness's.

    THE SUBSTITUTION IS SCOPED TO ONE ``Fields``. The seam is the driver module's
    globals, which every driver in the process shares; an unscoped patch sends the
    reference driver through a plan holding the other driver's device pointers and
    every array diverges at step 1 for a reason that has nothing to do with the
    kernels. That was measured, not imagined.
    """
    import cases  # noqa: PLC0415

    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference = lift_simulation(cases.BUILDERS[name](**kwargs), prefer_gpu=True,
                                gpu_id=0)
    fused = lift_simulation(cases.BUILDERS[name](**kwargs), prefer_gpu=True, gpu_id=0)
    substep_names = ("step_B", "step_D", "update_H", "update_E", "update_P")
    originals = {n: getattr(driver_module, n) for n in substep_names}
    try:
        plan = plan_step(fused.fields, fused.pml)
        out: Dict[str, Any] = {
            "replaces": list(plan.replaces),
            "refusals": {k: list(v) for k, v in plan.reasons.items()},
            "driver_step_path": fused.active_step_path,
        }

        def make(sub_step):
            def wrapper(fields, pml=None):
                entry = plan.plans.get(sub_step) if fields is fused.fields else None
                if entry is None:
                    return originals[sub_step](fields, pml)
                if sub_step == "update_P":
                    for sub_plan in entry:
                        sub_plan.run(fields.drive_field)
                    return None
                entry.run()
                return None
            return wrapper

        for sub_step in substep_names:
            setattr(driver_module, sub_step, make(sub_step))

        per_step = []
        for step in range(1, WHOLE_STEPS + 1):
            reference.step()
            fused.step()
            cp.cuda.runtime.deviceSynchronize()
            ref_state = whole_step_state(reference, cp)
            fused_state = whole_step_state(fused, cp)
            parts = {key: bit_compare(fused_state[key], ref_state[key])
                     for key in sorted(ref_state)}
            per_step.append({
                "step": step,
                "bit_identical": all(p["bit_identical"] for p in parts.values()),
                "differing_floats": sum(p["differing_floats"] for p in parts.values()),
                "total_floats": sum(p["total_floats"] for p in parts.values()),
                "differing_arrays": sorted(k for k, v in parts.items()
                                           if not v["bit_identical"]),
            })
        out["per_step"] = per_step
        out["bit_identical"] = all(s["bit_identical"] for s in per_step)
        out["first_divergent_step"] = next(
            (s["step"] for s in per_step if not s["bit_identical"]), None)
        return out
    finally:
        for sub_step, function in originals.items():
            setattr(driver_module, sub_step, function)
        reference.close()
        fused.close()
        cp.get_default_memory_pool().free_all_blocks()


def one_case(name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import cases  # noqa: PLC0415
    import cupy as cp  # noqa: PLC0415

    from meep_gpu import backends, stepping  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        ade_update_p_coverage,
        constitutive_coverage,
        pml_curl_coverage,
        plan_pml_curl,
        plan_step,
    )
    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    driver = lift_simulation(cases.BUILDERS[name](**kwargs), prefer_gpu=True, gpu_id=0)
    try:
        verdict = pml_curl_coverage(driver.fields, driver.pml)
        plan = plan_step(driver.fields, driver.pml)
        out: Dict[str, Any] = {
            "case": name, "kwargs": kwargs,
            "shape": [int(s) for s in driver.shape],
            "covered": bool(verdict.covered),
            "reasons": list(verdict.reasons),
            # The COMPOSED verdict: which sub-steps this configuration dispatches
            # and, for every one it does not, the named reason it fell out.
            "replaces": list(plan.replaces),
            "refusals": {k: list(v) for k, v in plan.reasons.items()},
            "constitutive": {
                side: list(constitutive_coverage(driver.fields, driver.pml, side).reasons)
                for side in ("H", "E")},
            "ade": {
                f"{index}:{component}":
                    list(ade_update_p_coverage(driver.fields, state, component).reasons)
                for index, state in enumerate(driver.fields.polarizations)
                for component in state.driven()},
        }
        # THE BUILDER, ASKED PER CURL SUB-STEP, ON EVERY CASE.
        #
        # WHY THE AGGREGATE ANSWER IS NOT THE PER-SUB-STEP ANSWER, and it is not a
        # subtlety this probe may keep collapsing. `pml_curl_coverage(fields, pml)`
        # with no `sub_step` takes the CONSERVATIVE AGGREGATE path — coverage.py's
        # clause 8 tests all six of CURL_TARGETS and says so in as many words:
        # "Conductivity changes only the curl whose primary targets carry it ... When
        # no sub-step is supplied this retains the conservative aggregate verdict used
        # by reports; builders and composers always ask for one named sub-step."
        # `plan_pml_curl` is a builder, so it asks for its own named sub-step
        # (launch.py:1351-1352, `return None` iff THAT sub-step's verdict refuses).
        #
        # MEASURED ON `2d_cond_pml`, on a lifted `mp.Simulation` rather than argued
        # from the source: its medium carries `D_conductivity=0.4`, so
        # `fields.condfac_for` answers non-None on Dx/Dy/Dz and None on Bx/By/Bz. The
        # aggregate verdict refuses, naming exactly the three D targets; the
        # `step_B` verdict names NO conductivity reason at all; the `step_D` verdict
        # names all three. So the builder returns a PLAN on step_B and None on
        # step_D, and `all(... is None)` over both is False for a configuration the
        # composition serves in full.
        #
        # THAT IS WHY `plan_is_none` BELOW IS KEPT EXACTLY AS IT WAS AND A SECOND,
        # SHARPER COUNTER IS ADDED BESIDE IT rather than silently redefined. The old
        # counter still means what it always meant — "both curls declined" — and its
        # numerator moving is a fact about the TREE (the conductive PML family landed
        # between the 2026-08-11 certified run and now), not about this run. Quietly
        # changing what a counter counts under an unchanged name is the drift this
        # whole apparatus exists to prevent.
        #
        # THE NEW COUNTER IS STRICTLY STRONGER THAN THE OLD, in two ways. It is an
        # EQUIVALENCE, so it also catches the direction `all(... is None)` never
        # looked at — a builder returning None where its own predicate ADMITS, which
        # is a silent coverage loss rather than a wrong step. And it is asked on ALL
        # EIGHT cases rather than on the four the probe happens to list as refusals.
        per_sub_step = {}
        for curl in ("step_B", "step_D"):
            named = pml_curl_coverage(driver.fields, driver.pml, curl)
            per_sub_step[curl] = {
                "covered": bool(named.covered),
                "plan_is_none": plan_pml_curl(driver.fields, driver.pml, curl) is None,
                "reasons": list(named.reasons),
            }
        out["per_sub_step"] = per_sub_step
        out["builder_matches_its_predicate"] = all(
            row["plan_is_none"] == (not row["covered"])
            for row in per_sub_step.values())

        if not verdict.covered:
            # Unchanged in meaning and in value: the same two builder calls, over the
            # same two sub-steps, collapsed the same way — now read off the rows
            # above instead of asking the builder a second time.
            out["plan_is_none"] = all(row["plan_is_none"]
                                      for row in per_sub_step.values())
            return out

        fields, pml = driver.fields, driver.pml
        # Nonzero everywhere, including the auxiliaries: a zero fu makes fu*kms an
        # exact zero whatever kms is, which would hide a mis-indexed coefficient.
        rng = np.random.default_rng(20260809)
        names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")
        seeded = {}
        for field_name in names:
            array = getattr(fields, field_name)
            host = rng.uniform(-1.0, 1.0, size=array.shape).astype(np.float32)
            array[...] = cp.asarray(np.ascontiguousarray(host))
            seeded[field_name] = array.copy()

        plans = {s: plan_pml_curl(fields, pml, s) for s in ("step_B", "step_D")}
        out["dtdx_plan"] = plans["step_B"].dtdx
        out["dtdx_grid"] = float(driver.grid.dt / driver.grid.dx)
        out["bc"] = list(plans["step_B"].bc)

        # Kernel leg first, on the seeded state; then restore and run the array
        # leg on the SAME seed. Sources are never written by either, so the two
        # legs see identical inputs at every step.
        kernel_state: List[Dict[str, Any]] = []
        for _ in range(STEPS):
            for sub_step in ("step_B", "step_D"):
                plans[sub_step].run()
            cp.cuda.runtime.deviceSynchronize()
            kernel_state.append({n: getattr(fields, n).copy()
                                 for n in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                                           "fu_Bx", "fu_By", "fu_Bz",
                                           "fu_Dx", "fu_Dy", "fu_Dz")})
        for field_name, value in seeded.items():
            getattr(fields, field_name)[...] = value

        per_step = []
        for step in range(STEPS):
            stepping.step_B(fields, pml)
            stepping.step_D(fields, pml)
            cp.cuda.runtime.deviceSynchronize()
            parts = {n: bit_compare(kernel_state[step][n], getattr(fields, n))
                     for n in kernel_state[step]}
            per_step.append({
                "step": step + 1,
                "bit_identical": all(p["bit_identical"] for p in parts.values()),
                "differing_floats": sum(p["differing_floats"] for p in parts.values()),
                "total_floats": sum(p["total_floats"] for p in parts.values()),
                "per_array": {k: v["bit_identical"] for k, v in parts.items()},
            })
        out["per_step"] = per_step
        out["bit_identical"] = all(s["bit_identical"] for s in per_step)
        out["first_divergent_step"] = next(
            (s["step"] for s in per_step if not s["bit_identical"]), None)
        return out
    finally:
        driver.close()
        cp.get_default_memory_pool().free_all_blocks()


def one_case_with_whole_step(name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The curl leg, then — when anything is covered — the composed whole-step leg."""
    import cupy as cp  # noqa: PLC0415

    out = one_case(name, kwargs)
    if out.get("replaces"):
        try:
            out["whole_step"] = composed_whole_step(name, kwargs, cp)
        except Exception as exc:  # noqa: BLE001
            import traceback  # noqa: PLC0415
            out["whole_step"] = {"error": f"{type(exc).__name__}: {exc}"[:600],
                                 "traceback": traceback.format_exc()[-1500:],
                                 "bit_identical": False}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    results: Dict[str, Any] = {
        "probe": "triton_engine_route",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": STEPS, "cases": [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    def save():
        # STAMPED BEFORE EVERY WRITE, like the fourteen probes already wired to it.
        # `stamp` records the sha256 of every module under `meep_gpu` and
        # `parity/meep_gpu` that THIS PROCESS imported, and normalises whatever
        # verdict shape the payload carries into `canonical_verdict`. Until this
        # landed the artifact could say neither what it imported nor whether it
        # released, so `recut_composition_records.released()` read it as
        # "no verdict key" and correctly refused to bind a weld from it.
        from gate_provenance import stamp  # noqa: PLC0415

        stamp(results)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(results, handle, indent=2)
        os.replace(tmp, args.out)

    for name, kwargs in COVERED_CASES + REFUSAL_CASES:
        started = time.time()
        try:
            case = one_case_with_whole_step(name, kwargs)
        except Exception as exc:  # noqa: BLE001
            import traceback
            case = {"case": name, "kwargs": kwargs, "bit_identical": False,
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                    "traceback": traceback.format_exc()[-1500:]}
        case["seconds"] = round(time.time() - started, 2)
        results["cases"].append(case)
        save()
        whole = case.get("whole_step") or {}
        log(f"[engine] {name}: covered={case.get('covered')} "
            f"replaces={case.get('replaces')} "
            f"curl_identical={case.get('bit_identical')} "
            f"whole_step_identical={whole.get('bit_identical')} "
            f"first_divergent_step={case.get('first_divergent_step')} "
            f"plan_is_none={case.get('plan_is_none')} "
            f"reason={(case.get('reasons') or ['-'])[0][:70]} "
            f"{case.get('error', '') or whole.get('error', '')} ({case['seconds']} s)")

    cases = results["cases"]
    covered = [c for c in cases if c.get("covered")]
    refused = [c for c in cases if c.get("covered") is False]
    composed = [c for c in cases if c.get("whole_step")]
    matched = [c for c in cases if c.get("builder_matches_its_predicate")]
    # THE COUNTER NAMES ARE THE WELD RECORD'S FIELD NAMES, and that is load-bearing
    # rather than cosmetic. `recut_composition_records.claim_disagreements` joins a
    # curated block to a fresh summary BY FIELD NAME; while this probe spelled them
    # `covered_identical` and `whole_step_identical`, the record's
    # `covered_curl_identical` and `covered_whole_step_identical` shared a name with
    # nothing and so were compared against the run by NOTHING. Two claims move from
    # uncompared to compared here, which is the check getting stronger — and the
    # first thing it reports is that `covered_whole_step_identical` has been stale
    # since the whole-step leg started running on every composed case rather than on
    # the covered four.
    results["summary"] = {
        "covered_curl_identical":
            f"{sum(bool(c.get('bit_identical')) for c in covered)}/{len(covered)}",
        "covered_whole_step_identical":
            f"{sum(bool(c['whole_step'].get('bit_identical')) for c in composed)}"
            f"/{len(composed)}",
        "builder_matches_its_predicate": f"{len(matched)}/{len(cases)}",
        "refusals_returned_none":
            f"{sum(bool(c.get('plan_is_none')) for c in refused)}/{len(refused)}",
        "replaces_by_case": {c["case"]: c.get("replaces") for c in cases},
    }

    # THE RELEASE CONDITION, STATED RATHER THAN INFERRED FROM THE COUNTERS.
    #
    # This probe had none, and inferring one was the trap: "every counter is n/n"
    # would refuse this run, because `refusals_returned_none` is 3/4 BY DESIGN since
    # the conductive PML family landed (see `one_case`, where the 2d_cond_pml
    # measurement is written out). A rule that reads a stale counter as a failure is
    # not more conservative than no rule — it is wrong in the direction that blocks
    # correct work, and it would have to be relaxed the first time someone looked.
    #
    # WHAT THE RELEASE RESTS ON is the arithmetic: every covered case bit-identical
    # on the curl legs, every composed case bit-identical over WHOLE_STEPS complete
    # driver steps with every array compared as uint32, and the builder agreeing with
    # its own named-sub-step predicate on every case. `refusals_returned_none` is
    # RECORDED and deliberately NOT a release clause: it is a fact about which cases
    # the plain family still declines, and the tree is free to move it by growing a
    # product without that being a regression in anything this probe measures.
    #
    # THE VACUITY CLAUSES ARE NOT DECORATION. A run whose case list failed to lift
    # would otherwise report 0/0 on every counter and release on all of them.
    reasons: List[str] = []
    errored = [c["case"] for c in cases if c.get("error")]
    if errored:
        reasons.append(f"cases raised before measuring: {errored}")
    whole_errored = [c["case"] for c in cases
                     if (c.get("whole_step") or {}).get("error")]
    if whole_errored:
        reasons.append(f"whole-step legs raised: {whole_errored}")
    if len(cases) != len(COVERED_CASES + REFUSAL_CASES):
        reasons.append(f"{len(cases)} cases ran of "
                       f"{len(COVERED_CASES + REFUSAL_CASES)} declared")
    if not covered:
        reasons.append("no case exercised the covered curl; a run that measured "
                       "nothing may not release")
    if not composed:
        reasons.append("no case composed a plan; the whole-step leg measured nothing")
    if not refused:
        reasons.append("no case fell outside the plain PML curl family; the builder's "
                       "refusal side measured nothing")
    for case in covered:
        if not case.get("bit_identical"):
            reasons.append(f"{case['case']}: the curl legs are not bit-identical to "
                           f"the array path (first divergent step "
                           f"{case.get('first_divergent_step')})")
        if len(case.get("per_step") or []) != STEPS:
            reasons.append(f"{case['case']}: the curl leg ran "
                           f"{len(case.get('per_step') or [])} of {STEPS} sub-steps")
    for case in composed:
        whole = case["whole_step"]
        if not whole.get("bit_identical"):
            reasons.append(f"{case['case']}: the composed whole step is not "
                           f"bit-identical to the array path (first divergent step "
                           f"{whole.get('first_divergent_step')})")
        if len(whole.get("per_step") or []) != WHOLE_STEPS:
            reasons.append(f"{case['case']}: the whole-step leg ran "
                           f"{len(whole.get('per_step') or [])} of {WHOLE_STEPS} steps")
    for case in cases:
        if not case.get("builder_matches_its_predicate"):
            reasons.append(f"{case['case']}: plan_pml_curl and pml_curl_coverage "
                           f"disagree per sub-step: {case.get('per_sub_step')}")
        # A SUB-STEP THE COMPOSITION DROPPED WITHOUT SAYING WHY. Empty today on every
        # case, and kept because an empty reason tuple beside a missing sub-step is a
        # silent fall-back — the one shape a `replaces` list cannot report.
        for sub_step, named in (case.get("refusals") or {}).items():
            if not named:
                reasons.append(f"{case['case']}: {sub_step} left the composition with "
                               f"no named reason")
    results["summary"]["status"] = "passed" if not reasons else "failed"
    results["summary"]["reasons"] = reasons

    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    log(f"[engine] summary {json.dumps(results['summary'])}")
    return 0 if results["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
