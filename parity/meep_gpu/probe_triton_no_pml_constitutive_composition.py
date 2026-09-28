"""Whole-step composition probe for the NO-PML CONSTITUTIVE family.

The sub-step gate (``gate_triton_no_pml_constitutive.py``) certifies the family's
own claim — ``update_H`` and ``update_E`` move no byte on an inert layer. This
probe asks the only question that decides whether that claim is WORTH anything:
does it take a corpus row WHOLE-STEP?

The arithmetic answer is arranged rather than computed, and it is the reason this
family exists. A no-PML, non-dispersive run's heavy step is five slots::

    step_B   -> no_pml.plain_curl_step            (certified, job 2315)
    update_H -> no_pml_constitutive               (this family: a null)
    step_D   -> no_pml.plain_curl_step            (certified, job 2315)
    update_E -> no_pml_constitutive               (this family: a null)
    update_P -> array-path no-op, stepping.py:1425-1426 (no polarizations)

Before this family, ``no_pml`` covered two of those five and the row stayed on the
array path for the other three. The measured consequence, from
``results/predicate_coverage_2026-08-12``: four rows print as
``no-PML  unadmitted: update_H, update_E`` — they already hold
``plain_curl@step_B`` and ``plain_curl@step_D``, and the two constitutive nulls are
exactly what closes them.

THREE LEGS, and which of them can run is decided by the host, never by a flag:

* ``census``      — pure predicate arithmetic over every slot of a whole step, for a
                    set of representative rows, reported MODULO THE BACKEND CLAUSE so
                    it is not vacuous on a laptop (every kernel predicate opens with
                    "array module is not cupy"). Produces the whole-step verdict per
                    row and the reason for every uncovered slot. Runs anywhere.
* ``constitutive`` — the half of the composition that is genuinely runnable without a
                    device: substitute ONLY ``update_H``/``update_E`` with this
                    family's null plans on a real ``FdtdDriver``, leave the curls on
                    the array path, run a stated number of COMPLETE steps — sources,
                    boundary fills, metal zeroing and all — and byte-compare every
                    live array after every step against an unsubstituted twin. With a
                    source-free non-vacuity control and a negative control (the null
                    forced onto an active-PML driver, which must diverge). Runs
                    anywhere; this is what the laptop certifies.
* ``whole_step``  — DEVICE ONLY. The same loop with ``step_B``/``step_D`` additionally
                    served by ``no_pml.plan_plain_curl``, so all four heavy slots are
                    off the array path at once. Requires CuPy AND Triton; on a host
                    without them the leg is RECORDED AS SKIPPED with its reason and
                    the artifact says in so many words that no whole-step byte-identity
                    claim has been made. It is not skipped silently and it is not
                    faked.

DISCIPLINE (each clause is here because a defect made it necessary):

* uint32 byte compare only, never ``allclose``; every live array of both drivers,
  including ``fu_*``, ``f_cond_*``, ``f_bfast_*`` and every polarization P/P_prev.
* a NON-POWER-OF-TWO Courant in every case, with a power-of-two control pair. On the
  constitutive half this is a PREDICTED NULL with its reason recorded (nothing this
  family does rounds); it is load-bearing on the curl, which is why the whole_step
  leg carries it too.
* PER-SLOT CALL COUNTERS on every substituted slot of every case, asserted equal to
  the step budget. A monkeypatch that fell through leaves both drivers on the array
  path and every step passes vacuously — that is not hypothetical, it is the defect
  the gate's engine leg caught by counting 48 calls for 24 steps.
* the step budget is STATED in the artifact, per case and in total.
* seeds carry normals, a planted signed-zero plane and dense subnormals, and NO inf
  or NaN (a NaN's sign and payload are IEEE-unspecified, so a raw-word comparison
  over one is version-sensitive). An in-run census is a PASS condition: a census of
  zero in any class is VACUOUS, not passed.
* the subnormal policy is stamped from the shared machinery
  (``gate_triton_complex.install_ftz_strip``), and the constitutive half records that
  the strip is INERT for it because nothing compiles.
* unbuffered per-case progress and an atomic JSON rewrite after every case.

Usage (laptop, NumPy — census + constitutive, the whole of what a laptop can hold)::

    python -u \\
        probe_triton_no_pml_constitutive_composition.py \\
        --out results/triton_no_pml_constitutive_<date>/composition.json

Usage (the GPU host, one clear device, adds the whole_step leg)::

    CUDA_VISIBLE_DEVICES=<one clear device> \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_no_pml_constitutive_composition.py \\
        --backend cupy --out results/.../composition_cupy.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _path in (_HERE, _REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# Shared machinery, never re-implemented: the atomic writer, the logger, the seeds,
# the census, the snapshot/diff comparator and the case builders all live in the
# family's own gate; the subnormal-policy machinery lives in the complex gate.
import gate_triton_complex as complex_gate  # noqa: E402
import gate_triton_no_pml_constitutive as gate  # noqa: E402

from meep_gpu import driver as driver_module  # noqa: E402
from meep_gpu.triton_kernels import no_pml as curl_family  # noqa: E402
from meep_gpu.triton_kernels import no_pml_constitutive as family  # noqa: E402

log = gate.log
save = gate.save
build = gate.build
snapshot = gate.snapshot
diff = gate.diff
census = gate.census
seed_arrays = gate.seed_arrays
SEED = gate.SEED
COURANT_NP2 = gate.COURANT_NP2
COURANT_P2 = gate.COURANT_P2
ELECTRIC_SOURCE = gate.ELECTRIC_SOURCE

install_ftz_strip = complex_gate.install_ftz_strip
ftz_strip_license_reasons = complex_gate.ftz_strip_license_reasons
policy_stamp = complex_gate.policy_stamp

#: The five slots of a heavy step, in the driver's own order (driver.py:3206 step_B,
#: :3212 update_H, :3215 step_D, :3225 update_E, :3226 update_P).
#: ``update_P`` is listed because a whole-step claim that quietly omits a slot is not
#: a whole-step claim.
#:
#: ON THE ROWS THIS FAMILY SERVES ``update_P`` IS AN ARRAY-PATH NO-OP OF ITS OWN
#: (stepping.py:1425-1426, the return before the loop when nothing is registered), and
#: it COUNTS toward the whole-step verdict — the record used to say "recorded rather
#: than counted as coverage" while the arithmetic at :whole counted it, and the two
#: now agree. Counting it is the right reading: "whole step" here means NO HEAVY WORK
#: IS LEFT ON THE ARRAY PATH, and a slot that returns before its first statement
#: leaves none. What that must never be allowed to mean is "a product covers it", so
#: every case records the slot's ``product`` string ('array-path no-op ...') and its
#: own :data:`HEAVY_SLOTS` column beside the whole-step verdict. An absorbing control
#: row therefore reads ``cov ['update_P']`` — one no-op slot and four refusals — and
#: is correctly NOT whole-step.
STEP_SLOTS: Tuple[str, ...] = ("step_B", "update_H", "step_D", "update_E", "update_P")

#: The four slots a PRODUCT has to cover. ``update_P`` is deliberately absent: on this
#: family's rows nothing is registered, so there is no polarization arithmetic for a
#: product to own. Reported beside ``whole_step`` so the two questions — "is any heavy
#: work left on the array path" and "how much of it did a kernel actually take" —
#: cannot be read off one number.
HEAVY_SLOTS: Tuple[str, ...] = ("step_B", "update_H", "step_D", "update_E")

#: Stated budget. Both legs run every case for exactly this many COMPLETE steps.
STEPS = 24


def _residual(reasons: Sequence[str]) -> List[str]:
    """Refusal reasons minus the backend clause.

    Every kernel predicate in the package opens with "array module is not cupy", so
    on a NumPy host that one clause refuses all of them and a raw census would report
    "nothing is covered" for reasons that have nothing to do with the configuration.
    Dropping exactly that reason — the way the predicate-coverage harness's
    ``residual_reasons`` does — asks the question the device would ask, and the
    artifact carries BOTH columns so the substitution is auditable.
    """
    return [reason for reason in reasons if "not cupy" not in reason]


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------

#: The four measured whole-step rows this family exists to close, plus the controls
#: that make the census falsifiable. ``expect_whole_step`` is a PREDICTION recorded
#: before the run, so a census that quietly stopped covering something fails rather
#: than merely reports.
ROWS: Tuple[Dict[str, Any], ...] = (
    {"name": "harminv_warnings_class", "cell": (1.2, 1.0, 1.4),
     "courant": COURANT_NP2, "salt": 41, "expect_whole_step": True,
     "row": "TestSimulation.test_harminv_warnings"},
    {"name": "amp_func_change_sources", "cell": (1.2, 1.0, 1.4),
     "courant": COURANT_NP2, "source": ELECTRIC_SOURCE, "salt": 42,
     "expect_whole_step": True,
     "row": "TestEigenModeSource.test_amp_func_change_sources"},
    {"name": "check_material_frequencies", "cell": (2.0, 2.0, 0.0), "dimensions": 2,
     "boundaries": {"x": "metallic", "y": "metallic"}, "courant": COURANT_NP2,
     "salt": 43, "expect_whole_step": True,
     "row": "TestMedium.test_check_material_frequencies"},
    # THE SOURCE IS LOAD-BEARING HERE AND WAS MISSING. On a single periodic voxel
    # every curl difference is a cell against ITSELF, so the curls are identically
    # zero and nothing can move the state but a source. Measured as shipped: 0 of 6
    # words moved across all 24 steps, so this row's "all four heavy slots
    # substituted, bit-identical" was comparing two frozen 6-word states — it passed
    # for the same reason an empty test passes. With the source, Dz moves.
    #
    # WHAT THIS ROW CAN AND CANNOT SHOW, stated so nobody reads more into it: it is a
    # degenerate-shape case. It demonstrates that the substitution stays byte-null
    # where n_a is 1 in every direction; it can NEVER be the evidence that the curls
    # ran, because on one voxel they have nothing to difference. The rows above carry
    # that, moving every word of Bx/By/Bz/Dx/Dy/Dz.
    {"name": "material_dispersion_user_material_1voxel", "cell": (0.0, 0.0, 0.1),
     "dimensions": 1, "courant": COURANT_NP2, "salt": 44,
     "source": ELECTRIC_SOURCE,
     "expect_whole_step": True,
     "moves_only_via": "the source; the curls are structurally zero on one voxel",
     "row": "TestMaterialDispersion.test_material_dispersion_with_user_material "
            "(the stores_E-False shape of it)"},
    {"name": "p2_courant_control", "cell": (1.2, 1.0, 1.4), "courant": COURANT_P2,
     "salt": 45, "expect_whole_step": True,
     "row": "power-of-two Courant control for the curl's association hazard"},
    # --- the controls. Each must NOT be whole-step, for a NAMED reason. ---
    {"name": "control_absorber_2d_conductivity", "cell": (1.2, 1.0, 1.4),
     "conductivity": 0.7, "courant": COURANT_NP2, "salt": 46,
     "expect_whole_step": False,
     "row": "TestAbsorber.test_absorber_2d — the constitutive nulls are covered, the "
            "CONDUCTIVE curl is not (no_pml clause 8), so the row does not close"},
    {"name": "control_active_pml", "cell": (1.2, 1.0, 1.4), "pml": 3,
     "courant": COURANT_NP2, "salt": 47, "expect_whole_step": False,
     "row": "an absorbing run: every slot here belongs to another product"},
    {"name": "control_no_pml_dispersive", "cell": (1.2, 1.0, 1.4),
     "dispersion": True, "courant": COURANT_NP2, "salt": 48,
     "expect_whole_step": False,
     "row": "material-dispersion.py — update_E is Arm S (not built) and update_P "
            "needs f_w_*, which no no-PML run allocates"},
)


# ---------------------------------------------------------------------------
# Leg 1 — the coverage census over every slot of a whole step
# ---------------------------------------------------------------------------

def slot_coverage(fields: Any, pml: Any) -> Dict[str, Dict[str, Any]]:
    """Which product covers each of the five slots, with the reason when none does."""
    out: Dict[str, Dict[str, Any]] = {}

    for sub_step in ("step_B", "step_D"):
        verdict = curl_family.plain_curl_coverage(fields, pml, sub_step)
        residual = _residual(verdict.reasons)
        out[sub_step] = {"product": "no_pml.plain_curl_step" if not residual else None,
                         "covered_raw": bool(verdict.covered),
                         "covered_modulo_backend": not residual,
                         "reasons": residual}

    for side, name in (("H", "update_H"), ("E", "update_E")):
        verdict = family.null_constitutive_coverage(fields, pml, side)
        # No backend clause exists on this predicate, so raw and modulo agree by
        # construction; both are reported so the two columns line up with the curl's.
        out[name] = {
            "product": "no_pml_constitutive (null)" if verdict.covered else None,
            "covered_raw": bool(verdict.covered),
            "covered_modulo_backend": bool(verdict.covered),
            "reasons": list(verdict.reasons)}

    if not bool(getattr(fields, "has_polarizations", False)):
        out["update_P"] = {
            "product": "array-path no-op (stepping.py:1425-1426)",
            "covered_raw": True, "covered_modulo_backend": True, "reasons": [],
            "note": "update_P returns before its loop when nothing is registered; no "
                    "kernel is needed and none is claimed"}
    else:
        from meep_gpu.triton_kernels.coverage import (  # noqa: PLC0415
            ELECTRIC_COMPONENTS, ade_update_p_coverage)

        reasons: List[str] = []
        for index, state in enumerate(fields.polarizations):
            for component in ELECTRIC_COMPONENTS:
                if not state.drives(component):
                    continue
                verdict = ade_update_p_coverage(fields, state, component)
                reasons.extend(f"pol{index}[{component}]: {reason}"
                               for reason in _residual(verdict.reasons))
        out["update_P"] = {
            "product": "ade_update_p" if not reasons else None,
            "covered_raw": not reasons, "covered_modulo_backend": not reasons,
            "reasons": reasons}
    return out


def run_census(results: Dict[str, Any], out_path: str, xp: Any,
               backend: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for spec in ROWS:
        fdtd = build(spec, xp)
        slots = slot_coverage(fdtd.fields, fdtd.pml)
        whole = all(slots[name]["covered_modulo_backend"] for name in STEP_SLOTS)
        constitutive_only = [name for name in ("update_H", "update_E")
                             if slots[name]["covered_modulo_backend"]]
        entry = {
            "name": spec["name"], "row": spec.get("row"),
            "shape": list(fdtd.grid.shape),
            "courant": repr(spec.get("courant", COURANT_NP2)),
            "np2_courant": bool(spec.get("courant", COURANT_NP2) != COURANT_P2),
            "slots": slots,
            "whole_step": bool(whole),
            "expected_whole_step": bool(spec["expect_whole_step"]),
            "constitutive_slots_covered": constitutive_only,
            # The two questions kept apart, so neither can be read off the other:
            # which HEAVY slots a product took, and whether the array path has any
            # work left at all (which counts update_P's own no-op, and says so).
            "heavy_slots_covered": [name for name in HEAVY_SLOTS
                                    if slots[name]["covered_modulo_backend"]],
            "update_P_is_array_path_noop": bool(
                "array-path no-op" in str(slots["update_P"].get("product"))),
            "ok": bool(whole == bool(spec["expect_whole_step"])),
        }
        if not entry["ok"]:
            entry["error"] = ("the whole-step verdict disagrees with the prediction "
                              "recorded before the run")
        cases.append(entry)
        log(f"[census:{backend}] {entry['name']:38s} whole_step={entry['whole_step']} "
            f"(expected {entry['expected_whole_step']}) "
            f"covered={[n for n in STEP_SLOTS if slots[n]['covered_modulo_backend']]} "
            f"ok={entry['ok']}")
        results.setdefault("census", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "heavy_slots": list(HEAVY_SLOTS),
            "note": ("``covered_modulo_backend`` drops the 'array module is not cupy' "
                     "clause every KERNEL predicate opens with; the null predicate has "
                     "no such clause, so its two columns agree by construction. "
                     "``whole_step`` counts all five slots INCLUDING update_P's own "
                     "array-path no-op — 'no heavy work left on the array path' — "
                     "while ``heavy_slots_covered`` is the four a product must own."),
            "cases": cases}
        save(results, out_path)
    return results["census"][backend]


# ---------------------------------------------------------------------------
# Leg 2 — the constitutive half of the composition, on a real driver
# ---------------------------------------------------------------------------

class _SlotSubstitution:
    """Replace named sub-steps on the DRIVER module's globals, for ONE driver.

    ``driver.py`` imports the sub-steps by name, so the swap has to be on the driver
    module's globals — patching ``stepping``'s would leave the driver holding the
    originals and the leg would silently measure nothing.

    Entered immediately around the substituted driver's ``step()`` and left
    immediately after, and every substituted function RAISES if handed a ``Fields``
    other than the one it was built for. A patch that leaked onto the reference
    driver would otherwise compare a null against a null and pass on every case.
    """

    def __init__(self, expected_fields: Any, plans: Dict[str, Any]) -> None:
        self.expected = expected_fields
        self.plans = plans
        self.counters: Dict[str, int] = {name: 0 for name in plans}
        self.leaks = 0
        self._originals: Dict[str, Any] = {}

    def _make(self, name: str):
        def substituted(fields, pml):
            del pml
            if fields is not self.expected:
                self.leaks += 1
                raise RuntimeError(
                    f"{name}'s substitution was invoked for a Fields it was not built "
                    "for: the patch has leaked and the leg is comparing like with like")
            self.plans[name].run()
            self.counters[name] += 1
        return substituted

    def __enter__(self) -> "_SlotSubstitution":
        for name in self.plans:
            self._originals[name] = getattr(driver_module, name)
            setattr(driver_module, name, self._make(name))
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, original in self._originals.items():
            setattr(driver_module, name, original)
        self._originals = {}
        return None


def _stepped_comparison(spec: Dict[str, Any], xp: Any, backend: str,
                        steps: int) -> Dict[str, Any]:
    """One case: reference driver vs a driver whose constitutive slots are null."""
    started = time.time()
    reference = build(spec, xp)
    substituted = build(spec, xp)
    plan = family.plan_null_constitutive_step(substituted.fields, substituted.pml)
    entry: Dict[str, Any] = {
        "name": spec["name"], "row": spec.get("row"),
        "shape": list(reference.grid.shape), "steps": steps,
        "courant": repr(spec.get("courant", COURANT_NP2)),
        "np2_courant": bool(spec.get("courant", COURANT_NP2) != COURANT_P2),
        "planned": list(plan.covered),
        "census_before": census(substituted.fields),
    }
    if set(plan.covered) != {"update_H", "update_E"}:
        entry["ok"] = False
        entry["error"] = ("the null plan did not cover both constitutive slots; this "
                          "case belongs to the census leg, not here")
        entry["refusals"] = {name: list(reasons)
                             for name, reasons in plan.refusals.items()}
        entry["seconds"] = round(time.time() - started, 3)
        return entry

    swap = _SlotSubstitution(substituted.fields, plan.plans)
    first_divergence = None
    total_words = 0
    for step in range(1, steps + 1):
        reference.step()
        with swap:
            substituted.step()
        comparison = diff(snapshot(reference.fields), snapshot(substituted.fields))
        total_words += comparison["total_words"]
        if not comparison["bit_identical"] and first_divergence is None:
            first_divergence = {"step": step,
                                "differing_words": comparison["differing_words"],
                                "per_array": comparison["per_array"]}
        if step % 8 == 0:
            log(f"[constitutive:{backend}] {spec['name']} step {step}/{steps} "
                f"diverged={first_divergence is not None}")
    entry["substitution_calls"] = dict(swap.counters)
    entry["substitution_leaks"] = swap.leaks
    entry["plan_runs"] = {name: p.runs for name, p in plan.plans.items()}
    entry["step_count"] = {"reference": reference.step_count,
                           "substituted": substituted.step_count}
    entry["words_compared"] = total_words
    entry["first_divergence"] = first_divergence
    entry["census_after"] = census(substituted.fields)
    counted = (reference.step_count == steps and substituted.step_count == steps
               and all(count == steps for count in swap.counters.values())
               and swap.leaks == 0)
    non_vacuous = all(entry["census_before"][key] > 0 for key in
                      ("nonzero_words", "signed_zero_words", "subnormal_words"))
    entry["step_and_call_counts_agree"] = bool(counted)
    entry["non_vacuous"] = bool(non_vacuous)
    entry["ok"] = bool(first_divergence is None and counted and non_vacuous
                       and total_words > 0)
    if not non_vacuous:
        entry["error"] = ("VACUOUS: the seeded state carried no nonzero / signed-zero "
                          "/ subnormal word")
    entry["seconds"] = round(time.time() - started, 3)
    return entry


def run_constitutive(results: Dict[str, Any], out_path: str, xp: Any,
                     backend: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    covered_rows = [spec for spec in ROWS if spec["expect_whole_step"]]
    for spec in covered_rows:
        entry = _stepped_comparison(spec, xp, backend, STEPS)
        cases.append(entry)
        log(f"[constitutive:{backend}] {entry['name']:38s} "
            f"identical={entry.get('first_divergence') is None} "
            f"calls={entry.get('substitution_calls')} ok={entry['ok']} "
            f"({entry['seconds']} s)")
        results.setdefault("constitutive", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "steps_per_case": STEPS, "total_steps": STEPS * len(cases),
            "cases": cases}
        save(results, out_path)

    # SOURCE NON-VACUITY. Two rows of the set differ only by an Ez gaussian; their
    # final states must differ, or "identical" above was measured on two runs that
    # were going to agree anyway.
    driven = build({"name": "nv_driven", "cell": (1.2, 1.0, 1.4),
                    "courant": COURANT_NP2, "source": ELECTRIC_SOURCE, "salt": 51}, xp)
    free = build({"name": "nv_free", "cell": (1.2, 1.0, 1.4),
                  "courant": COURANT_NP2, "salt": 51}, xp)
    for _ in range(STEPS):
        driven.step()
        free.step()
    effect = diff(snapshot(free.fields), snapshot(driven.fields))
    results["constitutive"][backend]["source_non_vacuity"] = {
        "differing_words": effect["differing_words"],
        "total_words": effect["total_words"],
        "ok": bool(effect["differing_words"] > 0),
        "note": ("same seed and step count, one with an Ez gaussian and one without: "
                 "the two states MUST differ")}
    log(f"[constitutive:{backend}] source non-vacuity: "
        f"{effect['differing_words']} words differ")

    # THE NEGATIVE CONTROL. Force the nulls into a driver whose layer IS active,
    # past the predicate that forbids it. The same loop must diverge; without this
    # row every "identical" above is unfalsified.
    control_spec = {"name": "negative_control_active_pml", "cell": (1.2, 1.0, 1.4),
                    "courant": COURANT_NP2, "pml": 3, "source": ELECTRIC_SOURCE,
                    "salt": 52}
    reference = build(control_spec, xp)
    forced = build(control_spec, xp)
    refusal = {side: family.null_constitutive_coverage(forced.fields, forced.pml, side)
               for side in ("H", "E")}
    plans = {"update_H": family.NullConstitutivePlan("H"),
             "update_E": family.NullConstitutivePlan("E")}
    swap = _SlotSubstitution(forced.fields, plans)
    divergence = None
    for step in range(1, STEPS + 1):
        reference.step()
        with swap:
            forced.step()
        comparison = diff(snapshot(reference.fields), snapshot(forced.fields))
        if not comparison["bit_identical"] and divergence is None:
            divergence = {"step": step,
                          "differing_words": comparison["differing_words"]}
    results["constitutive"][backend]["negative_control"] = {
        "predicate_refused": {side: not verdict.covered
                              for side, verdict in refusal.items()},
        "refusal_reasons": {side: list(verdict.reasons)
                            for side, verdict in refusal.items()},
        "first_divergence": divergence,
        "substitution_calls": dict(swap.counters),
        "ok": bool(divergence is not None
                   and all(not v.covered for v in refusal.values())
                   and all(count == STEPS for count in swap.counters.values())),
        "note": ("the nulls forced onto an ACTIVE-PML driver, past their own "
                 "predicate. Must diverge, and the predicate must have refused.")}
    log(f"[constitutive:{backend}] negative control: first divergence {divergence}, "
        f"predicate refused {[not v.covered for v in refusal.values()]}")
    save(results, out_path)
    return results["constitutive"][backend]


# ---------------------------------------------------------------------------
# Leg 3 — the whole step, DEVICE ONLY
# ---------------------------------------------------------------------------

def run_whole_step(results: Dict[str, Any], out_path: str, xp: Any,
                   backend: str) -> Dict[str, Any]:
    """All four heavy slots off the array path at once. Requires CuPy AND Triton.

    NOT SKIPPED SILENTLY. On a host without a device this records why, and the
    artifact's top-level claim block says that no whole-step byte-identity result
    exists yet. A leg that quietly disappears reads, later, as a leg that passed.
    """
    blockers: List[str] = []
    if cp is None or xp is not cp:
        blockers.append("no CuPy array module: no_pml.plain_curl_step launches "
                        "against device pointers and its predicate refuses NumPy at "
                        "its first clause")
    if not _TRITON_AVAILABLE:
        blockers.append("triton is not importable: there is no kernel to launch")
    if blockers:
        block = {"ran": 0, "ok": 0, "skipped": True, "blockers": blockers,
                 "claim": "NO whole-step byte-identity claim is made by this artifact",
                 "what_it_would_do": (
                     "substitute step_B/step_D with no_pml.plan_plain_curl and "
                     "update_H/update_E with no_pml_constitutive's null plans on the "
                     f"same driver, run {STEPS} complete steps per row, and "
                     "byte-compare every live array after every step, with per-slot "
                     "launch counters asserted equal to the budget")}
        results.setdefault("whole_step", {})[backend] = block
        log(f"[whole_step:{backend}] SKIPPED: {'; '.join(blockers)}")
        save(results, out_path)
        return block

    cases: List[Dict[str, Any]] = []
    for spec in [row for row in ROWS if row["expect_whole_step"]]:
        started = time.time()
        reference = build(spec, xp)
        substituted = build(spec, xp)
        curl_plan = curl_family.plan_plain_step(substituted.fields, substituted.pml)
        null_plan = family.plan_null_constitutive_step(substituted.fields,
                                                       substituted.pml)
        plans: Dict[str, Any] = {}
        plans.update({name: plan for name, plan in curl_plan.plans.items()})
        plans.update({name: plan for name, plan in null_plan.plans.items()})
        entry: Dict[str, Any] = {
            "name": spec["name"], "row": spec.get("row"), "steps": STEPS,
            "courant": repr(spec.get("courant", COURANT_NP2)),
            "np2_courant": bool(spec.get("courant", COURANT_NP2) != COURANT_P2),
            "slots_substituted": sorted(plans),
            "curl_refusals": {k: list(v) for k, v in curl_plan.refusals.items()},
            "null_refusals": {k: list(v) for k, v in null_plan.refusals.items()},
        }
        if set(plans) != {"step_B", "step_D", "update_H", "update_E"}:
            entry["ok"] = False
            entry["error"] = "not every heavy slot was covered; this is not whole-step"
            cases.append(entry)
            results.setdefault("whole_step", {})[backend] = {
                "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
                "cases": cases}
            save(results, out_path)
            continue
        swap = _SlotSubstitution(substituted.fields, plans)
        first_divergence = None
        totals = {"differing_words": 0, "total_words": 0, "comparisons": 0}
        # THE DENOMINATOR AND THE VACUITY GUARD, both of which this leg lacked while
        # carrying the family's largest claim. "Bit-identical over 24 steps" needs the
        # number of words that were actually compared, and — because TWO of the four
        # substituted slots are nulls — it needs proof that the run moved at all. Two
        # drivers that both sat still agree perfectly. The array-path driver's own
        # state before the first step against after the last is what rules that out.
        reference_initial = snapshot(reference.fields)
        for step in range(1, STEPS + 1):
            reference.step()
            with swap:
                substituted.step()
            comparison = diff(snapshot(reference.fields), snapshot(substituted.fields))
            totals["differing_words"] += comparison["differing_words"]
            totals["total_words"] += comparison["total_words"]
            totals["comparisons"] += 1
            if not comparison["bit_identical"] and first_divergence is None:
                first_divergence = {"step": step,
                                    "differing_words": comparison["differing_words"],
                                    "per_array": comparison["per_array"]}
            if step % 8 == 0:
                log(f"[whole_step:{backend}] {spec['name']} step {step}/{STEPS} "
                    f"diverged={first_divergence is not None}")
        entry["substitution_calls"] = dict(swap.counters)
        entry["substitution_leaks"] = swap.leaks
        entry["first_divergence"] = first_divergence
        counted = (all(count == STEPS for count in swap.counters.values())
                   and swap.leaks == 0
                   and substituted.step_count == STEPS)
        entry["step_and_call_counts_agree"] = bool(counted)
        entry["totals"] = totals
        moved = diff(reference_initial, snapshot(reference.fields))
        entry["reference_state_moved"] = {
            "differing_words": moved["differing_words"],
            "total_words": moved["total_words"],
            "arrays_moved": sorted(moved["per_array"]),
            "note": "the ARRAY-PATH driver's own state across the whole run. Two of "
                    "the four substituted slots are NULLS, so without this a case "
                    "where nothing moved would report 'all four slots off the array "
                    "path, bit-identical' and mean nothing."}
        entry["ok"] = bool(first_divergence is None and counted
                           and moved["differing_words"] > 0)
        entry["seconds"] = round(time.time() - started, 3)
        cases.append(entry)
        log(f"[whole_step:{backend}] {entry['name']:38s} "
            f"identical={first_divergence is None} calls={entry['substitution_calls']} "
            f"ok={entry['ok']} ({entry['seconds']} s)")
        results.setdefault("whole_step", {})[backend] = {
            "ran": len(cases), "ok": sum(int(c["ok"]) for c in cases),
            "skipped": False, "steps_per_case": STEPS,
            "total_steps": STEPS * len(cases), "cases": cases}
        save(results, out_path)
    return results["whole_step"][backend]


# ---------------------------------------------------------------------------
# Contract, provenance, main
# ---------------------------------------------------------------------------

def validate_payload(payload: Dict[str, Any], backend: str = "numpy",
                     require_all_legs: bool = True) -> Dict[str, Any]:
    """Raise unless every leg meets its contract — and unless every leg is here.

    ``require_all_legs`` closes the gap the probe's own docstring names: "a leg that
    quietly disappears reads, later, as a leg that passed". The whole_step guard used
    to fire only when the block EXISTED and lacked blockers, so a payload with no
    whole_step block at all — which is exactly what ``--legs census,constitutive``
    produces — passed with no statement about the device leg. Measured before the fix:
    deleting ``payload['whole_step']`` and deleting ``payload['whole_step']['numpy']``
    both returned ``status='passed'``.

    A whole_step block that records ``skipped`` with blockers is a perfectly good
    release artifact for this family on a laptop — the leg's absence is the thing that
    must be spoken, not the leg's success. False only for the suite's per-leg tests.
    """
    failures: List[str] = []

    present = [leg for leg in LEGS if payload.get(leg, {}).get(backend) is not None]
    absent = [leg for leg in LEGS if leg not in present]
    if require_all_legs and absent:
        failures.append(
            f"the artifact carries only {present} on backend {backend!r}: {absent} "
            "have no block at all, and a leg that is not in the artifact is not a leg "
            "that passed — a skipped whole_step leg must still RECORD itself skipped, "
            "with blockers")

    block = payload.get("census", {}).get(backend)
    if block is not None:
        if int(block.get("ran", 0)) <= 0 or int(block.get("ok", -1)) != int(
                block.get("ran", 0)):
            failures.append(f"census leg: {block.get('ok')}/{block.get('ran')}")
        whole = [case for case in block.get("cases", ()) if case.get("whole_step")]
        not_whole = [case for case in block.get("cases", ())
                     if not case.get("whole_step")]
        if not whole:
            failures.append("the census closed NO row whole-step: the family's whole "
                            "justification is unmeasured")
        if not not_whole:
            failures.append("the census closed EVERY row whole-step: with no negative "
                            "row the verdict is not falsifiable")
        for case in block.get("cases", ()):
            if bool(case.get("whole_step")) != bool(case.get("expected_whole_step")):
                failures.append(f"census row {case.get('name')} disagrees with its "
                                "recorded prediction")

    block = payload.get("constitutive", {}).get(backend)
    if block is not None:
        if int(block.get("ran", 0)) <= 0 or int(block.get("ok", -1)) != int(
                block.get("ran", 0)):
            failures.append(f"constitutive leg: {block.get('ok')}/{block.get('ran')}")
        courants = set()
        for case in block.get("cases", ()):
            courants.add(bool(case.get("np2_courant")))
            if case.get("first_divergence") is not None:
                failures.append(f"constitutive case {case.get('name')} diverged")
            if not case.get("step_and_call_counts_agree"):
                failures.append(f"constitutive case {case.get('name')}: step/call "
                                "counts disagree — the patch may not have taken")
            if not case.get("non_vacuous"):
                failures.append(f"constitutive case {case.get('name')} is VACUOUS")
        if courants != {True, False}:
            failures.append("the constitutive sweep must carry both a "
                            f"non-power-of-two Courant and its control; saw {courants}")
        source = block.get("source_non_vacuity")
        if not source or not source.get("ok"):
            failures.append("the source-free control did not differ from the driven run")
        control = block.get("negative_control")
        if not control or not control.get("ok"):
            failures.append("the negative control did not diverge: every 'identical' "
                            "verdict in this artifact is unfalsified")

    block = payload.get("whole_step", {}).get(backend)
    if block is not None and not block.get("skipped"):
        if int(block.get("ran", 0)) <= 0 or int(block.get("ok", -1)) != int(
                block.get("ran", 0)):
            failures.append(f"whole_step leg: {block.get('ok')}/{block.get('ran')}")
        for case in block.get("cases", ()):
            if case.get("first_divergence") is not None:
                failures.append(f"whole_step case {case.get('name')} diverged")
            if sorted(case.get("slots_substituted", ())) != [
                    "step_B", "step_D", "update_E", "update_H"]:
                failures.append(f"whole_step case {case.get('name')} did not substitute "
                                "all four heavy slots")
            # THE VACUITY GUARD. Half the substituted slots are nulls, so "identical"
            # is satisfied by two drivers that both sat still. Absent counts as a
            # failure, not as "not checked".
            moved = case.get("reference_state_moved")
            if not moved or int(moved.get("differing_words", 0)) <= 0:
                failures.append(
                    f"whole_step case {case.get('name')}: the ARRAY-PATH driver's own "
                    "state did not move across the run, so a four-slot substitution "
                    f"matching it certifies nothing ({moved})")
            # And the denominator: a claim of bit-identity needs the count of words it
            # held over, per step, not just the absence of a divergence record.
            totals = case.get("totals") or {}
            if int(totals.get("comparisons", 0)) != int(case.get("steps", -1)):
                failures.append(
                    f"whole_step case {case.get('name')}: {totals.get('comparisons')} "
                    f"comparisons for {case.get('steps')} steps — the per-step byte "
                    "comparison is the claim, so its count is part of the contract")
            if int(totals.get("total_words", 0)) <= 0:
                failures.append(f"whole_step case {case.get('name')} compared 0 words")
    elif block is not None and not block.get("blockers"):
        failures.append("the whole_step leg was skipped without recording why")

    if payload.get("policy") is None:
        failures.append("no subnormal-policy stamp")

    if failures:
        raise AssertionError("; ".join(failures))
    return {"status": "passed", "legs_validated": sorted(present),
            "legs_absent": sorted(absent),
            "all_legs_required": bool(require_all_legs),
            "whole_step_skipped": bool(
                (payload.get("whole_step", {}).get(backend) or {}).get("skipped"))}


def provenance(backend: str) -> Dict[str, Any]:
    import platform  # noqa: PLC0415

    return {
        "probe": "no-PML constitutive composition (whole-step question)",
        "backend": backend,
        "python": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": getattr(cp, "__version__", None),
        "triton": triton.__version__ if _TRITON_AVAILABLE else None,
        "family_sha256": gate.sha256_file(gate.SHIPPED_SOURCE),
        "curl_family_sha256": gate.sha256_file(os.path.join(
            _REPO_API, "meep_gpu", "triton_kernels", "no_pml.py")),
        "probe_sha256": gate.sha256_file(os.path.abspath(__file__)),
        "step_budget": {"per_case": STEPS,
                        "note": "identity is claimed for exactly this budget"},
        "byte_identity_claim": (
            "The constitutive half is measured HERE on whatever backend ran. The "
            "WHOLE-STEP claim requires the whole_step leg on a CUDA host; if that leg "
            "records skipped=True, no whole-step byte-identity claim exists."),
    }


LEGS = ("census", "constitutive", "whole_step")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default="results/no_pml_constitutive/composition.json")
    parser.add_argument("--legs", default=",".join(LEGS))
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    args = parser.parse_args(argv)

    if args.backend == "cupy":
        if cp is None:
            log("cupy is not importable; refusing to fabricate a device row")
            return 2
        xp: Any = cp
    else:
        xp = np

    requested = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
    unknown = [leg for leg in requested if leg not in LEGS]
    if unknown:
        log(f"unknown legs {unknown}; known: {LEGS}")
        return 2

    install_ftz_strip()

    out_path = os.path.abspath(args.out)
    results: Dict[str, Any] = {
        "provenance": provenance(args.backend),
        "policy": policy_stamp(args.backend),
        "legs_requested": requested,
        "slots": list(STEP_SLOTS),
    }
    results["policy"]["license_reasons"] = list(ftz_strip_license_reasons())
    results["policy"]["note_for_this_probe"] = (
        "the constitutive half compiles nothing, so the strip has nothing to strip "
        "on that leg; the whole_step leg DOES compile no_pml.plain_curl_step and is "
        "the leg the policy governs.")
    save(results, out_path)
    started = time.time()

    runners = {"census": run_census, "constitutive": run_constitutive,
               "whole_step": run_whole_step}
    for leg in requested:
        log(f"=== leg {leg} ({args.backend}) ===")
        runners[leg](results, out_path, xp, args.backend)

    # RE-STAMP: the startup call above could only ever have recorded zeros, because
    # ``policy_stamp`` reads live counters off the strip and no leg had run yet. This
    # probe's whole_step leg is the one the policy actually governs, so a frozen
    # ``nvrtc_calls: 0`` beside a leg that compiled the curl was the artifact claiming
    # the opposite of what happened. Both numbers kept and labelled.
    results["policy"]["at_startup"] = {
        "nvrtc_calls": results["policy"].get("nvrtc_calls"),
        "ftz_removed": results["policy"].get("ftz_removed"),
        "note": "taken before the first leg; structurally zero, measures nothing"}
    results["policy"]["measured_after_legs"] = policy_stamp(args.backend)
    results["policy"]["license_reasons"] = list(ftz_strip_license_reasons())

    summary: Dict[str, Any] = {}
    all_ok = True
    for leg in requested:
        block = results.get(leg, {}).get(args.backend, {})
        summary[leg] = {"ran": block.get("ran", 0), "ok": block.get("ok", 0),
                        "skipped": bool(block.get("skipped", False))}
        if summary[leg]["skipped"]:
            continue
        all_ok = all_ok and summary[leg]["ran"] > 0 and (
            summary[leg]["ok"] == summary[leg]["ran"])
    constitutive = results.get("constitutive", {}).get(args.backend, {})
    for key in ("source_non_vacuity", "negative_control"):
        extra = constitutive.get(key)
        if extra is not None:
            summary.setdefault("constitutive_extras", {})[key] = bool(extra["ok"])
            all_ok = all_ok and bool(extra["ok"])
    results["summary"] = summary
    try:
        results["contract"] = validate_payload(results, args.backend)
    except AssertionError as exc:
        results["contract"] = {"status": "failed", "failures": str(exc)}
        all_ok = False
        log(f"[contract] FAILED: {exc}")
    results["all_ok"] = bool(all_ok)
    results["seconds"] = round(time.time() - started, 3)
    save(results, out_path)
    log(f"=== summary: {json.dumps(summary, sort_keys=True)} all_ok={all_ok} "
        f"({results['seconds']} s) -> {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
