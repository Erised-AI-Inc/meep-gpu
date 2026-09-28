"""What a WHOLE beta step looks like if these curls were wired, and what it costs.

The byte gate certifies two sub-steps in isolation. This probe asks the question a
gate cannot: what happens across ONE COMPLETE DRIVER STEP — which sub-steps stay on
the array path, which mirrors go stale, and whether the residency invariant can be
held at all. It measures rather than argues, and it is deliberately allowed to
report a REFUSAL as its answer.

THE THREE MEASUREMENTS:

1. **The residency verdict, per storage family, and they DIFFER.** A REAL beta run
   can take all four sub-steps on the device — the curls from this family, the
   constitutive pair from the CERTIFIED kernel under this family's restated
   predicate — so the mirror set survives a complete step with no sync. A COMPLEX
   beta run cannot: no complex constitutive product is certified on this backend,
   so ``update_H``/``update_E`` run on the array path, write ``Hx``/``Ex`` behind
   the mirrors' backs, and the residency clause REFUSES the composition unless the
   caller declares those sub-steps synced. That asymmetry is the tranche's real
   composition result and it is measured here in both directions.

2. **The seam.** A source injects between ``step_B`` and ``update_H``
   (``fill_B``/``fill_D``), and no Metal product carries either slot. With a source
   declared, the residency verdict must refuse a held mirror set for BOTH storage
   families — the same refusal for a different reason, and the probe records both
   so a reader cannot mistake one for the other.

3. **Disjointness, in both directions.** Every WIRED arm and both of this family's
   unwired arms are asked for a verdict on every case: no configuration may be
   admitted by two products. Clause 12 is inverted (shipped predicates refuse
   ``beta != 0``; these require it), so the expected answer is "no overlap
   anywhere" — and running it is what makes that a measurement rather than a
   reading of the clause text.

    python -u probe_metal_special_kz_composition.py \\
        --out results/metal_special_kz_2026-08-15/composition.json
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import gate_metal_special_kz as gate  # noqa: E402
import metal_gate_kit as kit  # noqa: E402
import probe_metal_beta_expansion as expansion_probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import arms, coverage, device, special_kz  # noqa: E402
from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels import subnormal, templates  # noqa: E402

log, save = kit.log, kit.save

#: Which sub-steps are LIVE for a source-free beta run. `live_sub_steps` derives
#: this from the grid and the declared sources; it is restated here as the probe's
#: own expectation so a change in either is a visible disagreement rather than a
#: silently updated answer.
#:
#: THE WALL PASSES ARE THE SECOND COMPOSITION COST, and this probe found them
#: rather than assuming them: a METALLIC axis puts `zero_metal_B`/`zero_metal_D`
#: in the live set (stepping.py:2211/:2238, driver.py:3167/:3169), and those write
#: stored cell 0 of B and D on the array path — behind the mirrors. So a walled
#: beta run cannot hold its mirrors either, for a reason that has nothing to do
#: with the complex constitutive gap, and the two are recorded separately.
SOURCE_FREE_LIVE = ("step_B", "update_H", "step_D", "update_E")
WALL_PASSES = ("zero_metal_B", "zero_metal_D")


def expected_live(grid) -> tuple:
    """The live set this probe expects, derived from the grid's own walls."""
    walled = any(metal_launch.zero_metal_axes(grid))
    order = ("step_B", "zero_metal_B", "update_H", "step_D", "zero_metal_D",
             "update_E")
    live = set(SOURCE_FREE_LIVE) | (set(WALL_PASSES) if walled else set())
    return tuple(name for name in order if name in live)


def _plans_for(case: Dict[str, Any], fields, pml, residency, probe):
    """Every device plan a wired version of this family would build for one case."""
    plans: Dict[str, Any] = {}
    for sub_step in ("step_B", "step_D"):
        plans[sub_step] = (
            special_kz.plan_beta_bloch_pml_curl(fields, pml, sub_step, residency,
                                                probe=probe)
            if case["complex"] else
            special_kz.plan_beta_pml_curl(fields, pml, sub_step, residency))
    if not case["complex"]:
        # The CERTIFIED constitutive kernel under this family's restated
        # predicate: no new kernel, no new sub-step.
        for slot, side in (("update_H", "H"), ("update_E", "E")):
            plans[slot] = special_kz.plan_beta_run_constitutive(
                fields, pml, side, residency)
    return plans


def leg_residency(payload: Dict[str, Any], out: str) -> None:
    """The verdict on holding one mirror set across a complete step, per family."""
    rows: List[Dict[str, Any]] = []
    probe = expansion_probe.measure()
    started = time.time()
    for case in gate.CASES:
        residency = device.Residency()
        grid, fields, pml = gate.build(case)
        plans = _plans_for(case, fields, pml, residency, probe)
        planned = tuple(name for name, plan in plans.items() if plan is not None)
        live = metal_launch.live_sub_steps(fields, pml, sources=())
        verdict = coverage.residency_coverage(residency.names, planned, live)
        # ...and the same question once the array-path sub-steps are DECLARED
        # bracketed by an explicit sync, which is the only honest way to compose
        # the complex family today.
        unplanned = tuple(name for name in (live or ()) if name not in planned)
        declared = coverage.residency_coverage(residency.names, planned, live,
                                               synced=unplanned)
        row = {"case": case["name"], "complex": case["complex"],
               "planned": list(planned), "live": list(live or ()),
               "mirrors": len(residency.names),
               "held_without_sync": bool(verdict.covered),
               "reasons": list(verdict.reasons)[:3],
               "held_with_declared_sync": bool(declared.covered),
               "must_be_synced": list(unplanned)}
        rows.append(row)
        payload["legs"]["residency"] = rows
        save(payload, out)
        log(f"[residency] {case['name']:<28} planned={len(planned)}/4 "
            f"held={verdict.covered} held_with_sync={declared.covered} "
            f"({time.time() - started:.1f}s)")
        expected = expected_live(grid)
        assert live == expected, (
            f"the live set for a source-free beta run is {live}, not {expected}: "
            f"the probe's expectation and live_sub_steps disagree, and one of "
            f"them is wrong")
        walled = any(metal_launch.zero_metal_axes(grid))
        needed = set(WALL_PASSES) if walled else set()
        if case["complex"]:
            needed |= {"update_H", "update_E"}
        row["refusal_causes"] = sorted(
            ({"complex constitutive is not a Metal product"}
             if case["complex"] else set())
            | ({"a metallic axis puts the wall passes on the array path"}
               if walled else set()))
        assert set(unplanned) == needed, (unplanned, needed)
        assert bool(verdict.covered) == (not needed), verdict.reasons
        assert declared.covered, declared.reasons


def leg_whole_step(payload: Dict[str, Any], out: str) -> None:
    """The FOUR STEP_ORDER sub-steps, device against array path, mirrors verified.

    The REAL family runs all four on the device. The COMPLEX family runs its two
    curls there and its constitutive pair on the array path WITH an explicit sync
    at each crossing — which is the composition the residency leg says is the only
    legal one, executed rather than described.

    NOT A DRIVER STEP, and the row says so. A metallic axis also makes
    ``zero_metal_B``/``zero_metal_D`` live (``expected_live``), and neither side
    runs them here: this is a like-for-like comparison of the four sub-steps a
    plan can fill, which is what attributes a divergence to a kernel. The residency
    leg is where the wall passes are accounted for, as slots that MUST be synced
    because no Metal product carries them — calling this "one complete step" would
    have let a reader take the wall passes as covered.
    """
    rows: List[Dict[str, Any]] = []
    probe = expansion_probe.measure()
    started = time.time()
    for case in gate.CASES:
        residency = device.Residency()
        grid, fields, pml = gate.build(case)
        before = gate.snapshot(fields)
        plans = _plans_for(case, fields, pml, residency, probe)

        order = ("step_B", "update_H", "step_D", "update_E")
        for slot in order:
            plan = plans.get(slot)
            if plan is not None:
                plan.run()
                continue
            # The array path takes this sub-step, so the mirrors must be brought
            # out first and back in after: the declared sync, performed.
            residency.sync_out()
            getattr(stepping, slot)(fields, pml)
            residency.sync_in()
        residency.sync_out()
        # THE MIRROR CHECK BELONGS HERE, before the host arrays are rewound for
        # the oracle: verify() compares device against HOST, and running the array
        # path first would make every mirror look stale for the trivial reason
        # that the host had moved on.
        stale = residency.verify()
        got = gate.snapshot(fields)

        gate.restore(fields, before)
        for slot in order:
            getattr(stepping, slot)(fields, pml)
        oracle = gate.snapshot(fields)

        moved = sum(kit.differing(oracle[n], before[n]) for n in oracle)
        bad = {n: kit.differing(got[n], oracle[n]) for n in oracle}
        bad = {n: v for n, v in bad.items() if v}
        row = {"case": case["name"], "complex": case["complex"],
               "device_sub_steps": [s for s in order if plans.get(s) is not None],
               "array_path_sub_steps": [s for s in order if plans.get(s) is None],
               # RUN BY NEITHER SIDE, and named so the row cannot be read as a
               # driver step: the wall passes are live on a metallic axis and no
               # Metal product carries them. The residency leg is where that is
               # accounted for.
               "live_slots_this_leg_does_not_run": [
                   name for name in expected_live(grid) if name not in order],
               "mirrors": len(residency.names), "moved": moved,
               "differing": bad, "stale_mirrors": stale,
               "syncs_out": residency.syncs_out, "syncs_in": residency.syncs_in}
        rows.append(row)
        payload["legs"]["whole_step"] = rows
        save(payload, out)
        log(f"[whole_step] {case['name']:<28} device={len(row['device_sub_steps'])} "
            f"array={len(row['array_path_sub_steps'])} moved={moved} "
            f"{'IDENTICAL' if not bad else bad} stale={stale} "
            f"({time.time() - started:.1f}s)")
        kit.assert_moved(moved, f"{case['name']} whole step")
        assert residency.names, (
            "VACUOUS: an empty mirror registry verifies trivially")
        assert not stale, stale
        assert not bad, row


def leg_seam(payload: Dict[str, Any], out: str) -> None:
    """A DECLARED source makes the seam live, and no Metal product carries it.

    With a source the driver injects between ``step_B`` and ``update_H``, so
    ``fill_B``/``fill_D`` join the live set. Neither has a Metal arm, so the
    residency verdict must refuse a held mirror set for BOTH families — the same
    refusal for a different reason than the complex family's, and both are recorded
    so they cannot be confused.
    """
    class _Source:
        field_type = "D"

    rows: List[Dict[str, Any]] = []
    probe = expansion_probe.measure()
    for case in gate.CASES:
        residency = device.Residency()
        grid, fields, pml = gate.build(case)
        plans = _plans_for(case, fields, pml, residency, probe)
        planned = tuple(name for name, plan in plans.items() if plan is not None)
        live = metal_launch.live_sub_steps(fields, pml, sources=(_Source(),))
        verdict = coverage.residency_coverage(residency.names, planned, live)
        undeclared = metal_launch.live_sub_steps(fields, pml, sources=None)
        row = {"case": case["name"], "complex": case["complex"],
               "live_with_a_source": list(live or ()),
               "held": bool(verdict.covered),
               "reasons": list(verdict.reasons)[:3],
               "unarmed_slots": dict(metal_launch.NO_ARM_REASONS),
               "live_when_sources_undeclared": undeclared}
        rows.append(row)
        payload["legs"]["seam"] = rows
        save(payload, out)
        log(f"[seam] {case['name']:<28} live={live} held={verdict.covered}")
        assert "fill_D" in (live or ()), live
        assert not verdict.covered, (
            "a mirror set was held across an UNARMED source seam: fill_D writes "
            "D on the array path and no Metal product carries that slot")
        assert undeclared is None, (
            "live_sub_steps answered a question it cannot answer: with no declared "
            "sources, 'no seam work' is an inference from its own ignorance")
        assert "fill_B" in row["unarmed_slots"] and "fill_D" in row["unarmed_slots"], (
            "the seam slots lost their named reasons: a reader must be able to "
            "tell 'no product carries this' from 'nobody asked'")


def leg_disjointness(payload: Dict[str, Any], out: str) -> None:
    """No configuration admitted by two products, measured in BOTH directions.

    THE TABLE IS MADE WHOLE FIRST (``gate.assert_table_is_whole``), and every arm is
    consulted — wired, unwired-registered and unregistered. Both corrections were
    forced by measurement rather than taste:

    * ``arms.registered()`` is populated by IMPORT SIDE EFFECTS. This probe's own
      import set never reached ``offdiag_update_e``, so its ``update_E`` arm was
      absent from the table this leg claims to sweep;
    * skipping ``wired=False`` rows dropped ``complex_fields`` entirely — which is
      the single arm most likely to collide with this family, since both are complex
      curls on ``step_B``/``step_D``. An unwired-vs-unwired overlap would have been
      invisible to this leg and to the gate's.
    """
    rows: List[Dict[str, Any]] = []
    probe = expansion_probe.measure()
    table = gate.assert_table_is_whole(payload)
    # `complex_fields` reads its OWN key and its OWN pattern set; with only this
    # family's probe bound it refused every configuration and every complex row
    # below was a comparison against a product that could not admit anything.
    complex_probe, complex_source = gate.complex_family_probe()
    payload["complex_probe_source"] = (complex_source if complex_probe is not None
                                       else None)
    payload["complex_probe_reason"] = (None if complex_probe is not None
                                       else complex_source)
    # `beta_probe` is the key this family's arms read (special_kz.py:1634); nothing
    # reads `"probe"`, which is what this bound before. Under the wrong key every
    # complex row below reported ZERO admitters and passed the `<= 1` assertion
    # vacuously — a disjointness measured over a product that had been disabled.
    extra = {"beta_probe": probe, "complex_probe": complex_probe}
    log(f"[disjointness] arm table: {table['registered_families']}")
    log(f"[disjointness] unregistered arms swept: {table['unregistered_arms']}")
    log(f"[disjointness] complex_fields licence: "
        f"{complex_source if complex_probe is not None else 'NONE'}")
    matrix = list(gate.CASES) + [
        dict(gate.case_named("r1_kz2d_periodic"), name="beta_zero_control",
             beta=0.0),
        # THE COMPLEX CONTROL. Without it the complex half of this sweep is
        # vacuous: `complex_fields` refuses every beta run, so "no complex
        # co-admission" would be measured on a table where the only other
        # complex product refuses everything it was asked. On a beta = 0
        # complex run it ADMITS, which is what makes the beta rows a
        # disjointness measurement rather than an absence.
        dict(gate.case_named("c2_marquee_inplane_bloch"),
             name="complex_beta_zero_control", beta=0.0),
    ]
    for case in matrix:
        residency = device.Residency()
        grid, fields, pml = gate.build(case, beta=case.get("beta"))
        context = arms.StepContext(fields, pml, residency,
                                   (templates.CONTRACT_OFF,), sources=(),
                                   extra=extra)
        admitters: Dict[str, List[str]] = {
            slot: gate.admitters_for(slot, context)
            for slot in ("step_B", "step_D", "update_H", "update_E")}
        row = {"case": case["name"], "beta": case.get("beta"),
               "complex": case["complex"], "admitters": admitters}
        rows.append(row)
        payload["legs"]["disjointness"] = rows
        save(payload, out)
        log(f"[disjointness] {case['name']:<28} {admitters}")
        for slot, names in admitters.items():
            assert len(names) <= 1, (
                f"{case['name']}: {slot} is co-admitted by {names}; two products "
                f"on one slot is the over-covering dispatch the composer fails "
                f"closed on, and it would become live the moment this family is "
                f"wired")
        if case["name"] == "beta_zero_control":
            assert admitters["step_B"] == ["wired:pml_curl/PML"], admitters
        elif case["name"] == "complex_beta_zero_control":
            # THE VACUITY CONTROL for every complex row above: the other complex
            # curl on this slot must ADMIT something, or "exactly one admitter"
            # on the beta rows is an absence of coverage rather than disjointness.
            if complex_probe is None:
                kit.predicted_null(row, complex_source)
                save(payload, out)
            else:
                assert any("complex_fields" in name
                           for name in admitters["step_B"]), (admitters,
                                                              complex_source)
        else:
            # THE SOLE ADMITTER MUST BE THIS FAMILY. The assertion here read
            # `.startswith("unregistered:")`, which pinned the arm's REGISTRATION
            # TAG rather than its identity and was therefore a statement about a
            # tranche rather than about coverage: when a later round registered and
            # wired these arms, the tag became "wired:" and the row failed on the
            # wiring working. What the disjointness sweep actually needs is that
            # exactly one product admits (asserted above) and that it is the beta
            # family (asserted here) — the tag is recorded in the artifact either
            # way, so a reader can still see when it changes.
            # A NON-EMPTY LIST IS PART OF THE ASSERTION, not a precondition for it:
            # `len(names) <= 1` above is satisfied best by a sweep that admits
            # nothing, so the beta rows need a floor or a disabled product reads as
            # a clean disjointness result.
            assert len(admitters["step_B"]) == 1 and (
                "special_kz" in admitters["step_B"][0]), admitters
            assert len(admitters["step_D"]) == 1 and (
                "special_kz" in admitters["step_D"][0]), admitters


LEGS = (("residency", leg_residency), ("whole_step", leg_whole_step),
        ("seam", leg_seam), ("disjointness", leg_disjointness))
REQUIRED = tuple(name for name, _ in LEGS)


def main() -> int:
    parser = kit.argument_parser(__doc__)
    arguments = parser.parse_args()
    started = time.time()
    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    environment = kit.environment_stamp()
    payload: Dict[str, Any] = {
        "environment": environment,
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
        "question": ("what one COMPLETE beta step costs if these curls were wired: "
                     "which sub-steps stay on the array path, which mirrors go "
                     "stale, and whether the residency invariant can be held"),
    }
    save(payload, arguments.out)
    if not environment.get("mps_available"):
        return kit.cannot_certify(payload, arguments.out,
                                  ["no MPS device on this host"])

    ran = kit.run_legs(LEGS, payload, arguments.out,
                       kit.wanted_legs(arguments.legs))
    missing = [name for name in REQUIRED if name not in ran]
    if missing:
        return kit.cannot_certify(payload, arguments.out,
                                  [f"required legs did not run: {missing}"])
    payload["summary"] = {
        "status": "measured",
        "legs_run": ran,
        "headline": ("a REAL beta run composes end to end on the device; a COMPLEX "
                     "one cannot hold its mirrors across the constitutive pair, "
                     "because no complex constitutive product is certified on this "
                     "backend — the composition gap is the complex tranche's, not "
                     "beta's"),
        "elapsed_s": round(time.time() - started, 1),
    }
    save(payload, arguments.out)
    log(f"COMPOSITION PROBE MEASURED in {payload['summary']['elapsed_s']}s -> "
        f"{arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
