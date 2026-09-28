"""Complete-driver CUDA gate for the BFAST Triton composition.

The sub-step gate (``gate_triton_bfast.py``) certifies the kernel against an
in-file transcription; this complementary probe asks the composition question:
with ``step_B``/``update_H``/``step_D``/``update_E`` served by ``bfast_curl``'s
engine-route plans — the BFAST curls plus the bfast-run constitutive
delegations to the CERTIFIED kernels — installed by monkeypatch, production
dispatch untouched, ``plan_fast_path`` still returning None — does the
complete CuPy driver state reproduce the array-path driver EXACTLY after
every step, for a STATED number of steps, sources, PML, boundary fills,
metal zeroing, flux monitors and the magnetic-sync backup/restore included,
f_bfast STATES byte-compared throughout?

Seven cases, 80 complete steps (10+10+10+12+12+10+16 — ``TOTAL_BUDGET``, and
the artifact carries the per-case split), every one on a non-power-of-two
Courant,
each with a fresh ``Fields`` — the marginally stable IIR state
((-1)^n, stepping.py:868-874) makes cross-case reuse contaminating:

1. ``c1_marquee_refl_angular_quiet`` — the demand case's own shape scaled to
   probe size: dims=3, cell (1,1,96) cells with z-PML both faces (10 cells =
   1.0 unit), the two-media diagonal interface (n1=1.4 / n2=3.5),
   GaussianSource Ex near the low edge, k = (1.4 sin 35.7 deg, 0, 0),
   Courant (1-kx)/sqrt(3), 10 steps; PLUS a source-free control proving
   non-vacuity — the source case must diverge from its control BY STEP 2.
2. ``c2_marquee_refl_angular_loud`` — c1 with amplitudes scaled 1e4 so
   low-order mantissa bits are hot; the loud margin (per-step max |D|) is
   recorded.
3. ``c3_full_k_3d`` — 12x10x14, PML z, periodic x/y, k = (0.31, 0.17, 0.23)
   all distinct, RANDOM FIELDS AND RANDOM STATE, Courant 0.35, 10 steps.
   THE k-indexing needle: the marquee's P-pol content exercises only By/Dz,
   so this case is what makes the m1/m2 class byte-visible on all six
   targets at composition level.
4. ``c4_guard_2d`` — DECLARED dimensions=2, k = (0.4, 0, 0), random content
   (Dz/TM included), PML y, periodic x, Courant 0.35, 12 steps — the
   invariance-guard needle (have_p kills k2 = kx on the Ez sum).
5. ``c5_metallic_mask`` — metallic x boundary, PML z, k = (0.4, 0.25, 0),
   12 steps — the mask needle: wall-row STATE bytes.
6. ``c6_sync_flux`` — c1's grid + a flux monitor + a
   ``synchronize_magnetic_fields``/``restore_magnetic_fields`` ROUND TRIP
   after step 5 — pins in-place state mutation and the f_bfast
   backup/restore (driver.py:4126-4135) at the byte level. The sync's
   half-step runs the PATCHED step_B/update_H once more, so those slots'
   launch budgets are steps + 1.
7. ``c7_dft_flux_decimation`` — the standing decimation pattern from the
   nonlinear/offdiag tranches riding a BFAST run: c3's grid + Gaussian Ex +
   a flux monitor at the engine's automatically resolved decimation
   (recorded), 16 steps, with a source-free control.

An ARMED mutation leg re-runs c3 with the step_D plan's six scalars built for
the WRONG side (the D negation dropped — m2 at composition level; c3's
distinct k components are the needle). The leg counts the mutated plan's
launches and FAILS LOUDLY if it never ran or never diverged.

Requirements carried from prior defects: per-complete-step uint32 compare of
EVERY allocated primary and auxiliary INCLUDING the six f_bfast states;
non-power-of-two Courants everywhere (the marquee's own 0.105679... included);
the step budget STATED in the artifact; unbuffered per-case progress; atomic
JSON rewrite per case; own provenance record; and PER-SLOT launch counters on
EVERY patched sub-step of EVERY case — a monkeypatch that silently fell
through would leave candidate and reference both on the array path and every
step would pass vacuously, so each patched slot must prove it launched
exactly its budget.

SUBNORMAL POLICY. This probe runs UNDER THE STRIPPED IEEE-KEEP POLICY — the
ship configuration — installed via ``gate_triton_complex.install_ftz_strip``
BEFORE ``guard_kernel_compilation`` so the guard wraps the strip and both
apply. Every claim states the policy it was cut under. The BFAST family's
FTZ surface is unusually hot (the state's subnormals never decay), which is
the sub-step gate's dedicated subnormal leg; here the states evolve freely
across whole steps under the same policy.

Usage (the GPU host, one clear device; the cache dir must carry the policy token)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_bfast_composition.py \\
        --out results/triton_bfast_<date>/composition.json

Laptop (no CUDA/triton): ``--self-check`` builds every case's drivers on
NumPy, steps the array path two steps per case (the BFAST fold IS in that
path), verifies the marquee Courant against its (1 - kx)/sqrt(3) formula, and
says in so many words that the Triton legs were skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The policy machinery, atomic writer and logging live in the complex gate;
# the seeds, marquee numbers and provenance live in the bfast gate.
import gate_triton_bfast as bfast_gate  # noqa: E402
import gate_triton_complex as gate  # noqa: E402

SEED = gate.SEED
save = gate.save
log = gate.log

KX_MARQUEE = bfast_gate.KX_MARQUEE
COURANT_MARQUEE = bfast_gate.COURANT_MARQUEE
FULL_K = bfast_gate.FULL_K
K_METALLIC = bfast_gate.K_METALLIC
BFAST_ALL = bfast_gate.BFAST_ALL

STATE_NAMES = gate.ALL_STATE + BFAST_ALL
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
REQUIRED_STATE = set(STATE_NAMES)


def gaussian_source(component: str, amplitude: float = 1.0,
                    center=(0.0, 0.0, -3.6)) -> Dict[str, Any]:
    """The marquee's source shape: Gaussian, fcen 1.875 / df 1.25 (MEEP
    test_refl_angular's wvl 0.4-0.8 band), near the low z edge."""
    return {"component": component, "source_type": "gaussian",
            "frequency": 1.875, "fwidth": 1.25,
            "center": center, "size": (0.0, 0.0, 0.0),
            "amplitude": amplitude}


MARQUEE = {
    "dimensions": 3, "cell": (0.1, 0.1, 9.6), "resolution": 10.0,
    "boundaries": "periodic", "bfast": (KX_MARQUEE, 0.0, 0.0),
    "courant": COURANT_MARQUEE, "pml": {"z": 10}, "epsilon": "two_media",
}

#: Each case: the driver recipe plus the probe's own knobs. ``seed_scale``
#: scales the seeded primaries; ``deadline`` is the step by which a source
#: case must have separated from its control (None = by the end).
CASES: Tuple[Dict[str, Any], ...] = (
    {**MARQUEE, "name": "c1_marquee_refl_angular_quiet",
     "sources": (gaussian_source("Ex"),), "steps": 10, "seed_state": True,
     "seed_scale": 0.25, "sync_after": None, "flux": None, "deadline": 2},
    {**MARQUEE, "name": "c2_marquee_refl_angular_loud",
     "sources": (gaussian_source("Ex", amplitude=1.0e4),), "steps": 10,
     "seed_state": True, "seed_scale": 2.5e3, "sync_after": None,
     "flux": None, "deadline": 2, "record_loud_margin": True},
    {"name": "c3_full_k_3d", "dimensions": 3, "cell": (1.2, 1.0, 1.4),
     "resolution": 10.0, "boundaries": "periodic", "bfast": FULL_K,
     "courant": 0.35, "pml": {"z": 3}, "epsilon": "sine", "sources": (),
     "steps": 10, "seed_state": True, "seed_scale": 0.25,
     "sync_after": None, "flux": None, "deadline": None},
    {"name": "c4_guard_2d", "dimensions": 2, "cell": (3.0, 3.0, 0.0),
     "resolution": 12.0, "boundaries": "periodic", "bfast": (0.4, 0.0, 0.0),
     "courant": 0.35, "pml": {"y": 5}, "epsilon": "sine", "sources": (),
     "steps": 12, "seed_state": True, "seed_scale": 0.25,
     "sync_after": None, "flux": None, "deadline": None},
    {"name": "c5_metallic_mask", "dimensions": 3, "cell": (0.8, 0.6, 1.0),
     "resolution": 10.0,
     "boundaries": ("metallic", "periodic", "periodic"), "bfast": K_METALLIC,
     "courant": 0.35, "pml": {"z": 3}, "epsilon": "sine", "sources": (),
     "steps": 12, "seed_state": True, "seed_scale": 0.25,
     "sync_after": None, "flux": None, "deadline": None},
    {**MARQUEE, "name": "c6_sync_flux",
     "sources": (gaussian_source("Ex"),), "steps": 10, "seed_state": True,
     "seed_scale": 0.25, "sync_after": 5,
     "flux": {"fcen": 1.875, "df": 1.25, "nfreq": 3,
              "center": (0.0, 0.0, 2.0), "size": (0.1, 0.1, 0.0),
              "decimation_factor": 1},
     "deadline": 2},
    {"name": "c7_dft_flux_decimation", "dimensions": 3,
     "cell": (1.2, 1.0, 1.4), "resolution": 10.0, "boundaries": "periodic",
     "bfast": FULL_K, "courant": 0.35, "pml": {"z": 3}, "epsilon": "sine",
     "sources": (gaussian_source("Ex", center=(0.0, 0.0, -0.25)),),
     "steps": 16, "seed_state": True, "seed_scale": 0.25,
     "sync_after": None,
     "flux": {"fcen": 0.8, "df": 0.4, "nfreq": 5,
              "center": (0.0, 0.0, 0.35), "size": (1.2, 1.0, 0.0),
              "decimation_factor": 0},
     "deadline": None},
)
TOTAL_BUDGET = sum(case["steps"] for case in CASES)


def _epsilon_host(kind: str, shape) -> np.ndarray:
    if kind == "two_media":
        # The marquee's diagonal interface: n1 = 1.4 below z = 0, n2 = 3.5
        # above (test_refl_angular.py:63-66's geometry, scaled).
        eps = np.full(shape, np.float32(1.4 * 1.4), dtype=np.float32)
        eps[:, :, shape[2] // 2:] = np.float32(3.5 * 3.5)
        return eps
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))


def build_driver(xp, case: Dict[str, Any], sources, seed: int,
                 prefer_gpu: bool):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=case["cell"], resolution=case["resolution"],
        dimensions=case["dimensions"], force_complex_fields=False,
        courant=case["courant"], boundaries=case["boundaries"],
        k_point=(0.0, 0.0, 0.0), bfast_scaled_k=case["bfast"],
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    shape = tuple(int(v) for v in driver.shape)
    driver.set_epsilon(xp.asarray(_epsilon_host(case["epsilon"], shape)))
    driver.setup_pml(dict(case["pml"]))
    for declaration in sources:
        driver.add_source(dict(declaration))
    rng = np.random.default_rng(seed)
    scale = float(case.get("seed_scale", 0.25))
    for primary in PRIMARY_NAMES:
        host = scale * bfast_gate._seed_host_real(shape, rng)
        driver.set_field(primary, xp.asarray(
            np.ascontiguousarray(host.astype(np.float32))))
    if case.get("seed_state"):
        srng = np.random.default_rng(seed + 7)
        for name in BFAST_ALL:
            host = bfast_gate._seed_host_state(shape, srng)
            getattr(driver.fields, name)[...] = xp.asarray(host)
    return driver


def state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name) for name in STATE_NAMES
            if getattr(driver.fields, name, None) is not None}


def compare(cp, left, right) -> Dict[str, Any]:
    a = np.ascontiguousarray(cp.asnumpy(left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(cp.asnumpy(right)).view(np.uint32).ravel()
    return {"bit_identical": bool(np.array_equal(a, b)),
            "differing_floats": int(np.count_nonzero(a != b)),
            "total_floats": int(a.size)}


def build_plans(driver) -> Dict[str, Any]:
    """The four sub-step plans; every slot must admit (no dispersion or
    offdiag rides these cases)."""
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    plans = {
        "step_B": bfast_curl.plan_bfast_pml_curl(
            driver.fields, driver.pml, "step_B"),
        "update_H": bfast_curl.plan_bfast_run_constitutive(
            driver.fields, driver.pml, "H"),
        "step_D": bfast_curl.plan_bfast_pml_curl(
            driver.fields, driver.pml, "step_D"),
        "update_E": bfast_curl.plan_bfast_run_constitutive(
            driver.fields, driver.pml, "E"),
    }
    missing = sorted(slot for slot, plan in plans.items() if plan is None)
    if missing:
        raise AssertionError(f"plans refused for {missing}")
    return plans


class PlanCounter:
    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.launches = 0

    def run(self, guard: Optional[bool] = None) -> None:
        self.launches += 1
        self.plan.run(guard=guard)


def install(driver_module, plans: Dict[str, Any], owner):
    """Serve the patched sub-steps from the plans, for the owner fields only.
    Sources, boundary fills, ``zero_metal_*`` and the DFT accumulation stay
    the driver's own. Production dispatch is untouched. The sync half-step
    (driver.py:4160-4186) calls step_B/update_H through THESE module
    globals, so a c6 sync round trip launches the patched slots too."""
    names = tuple(plans)
    originals = {name: getattr(driver_module, name) for name in names}

    def wrapper(name):
        def wrapped(fields, pml=None):
            if fields is owner and name in plans:
                plans[name].run()
                return None
            return originals[name](fields, pml)
        return wrapped

    for name in names:
        setattr(driver_module, name, wrapper(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


def _expected_launches(case: Dict[str, Any], steps: int) -> Dict[str, int]:
    syncs = 1 if case.get("sync_after") else 0
    return {"step_B": steps + syncs, "update_H": steps + syncs,
            "step_D": steps, "update_E": steps}


def run_case(cp, case: Dict[str, Any], steps_override=None,
             mutate_step_d: bool = False) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    name = case["name"]
    steps = steps_override or case["steps"]
    sources = tuple(case["sources"])
    reference = build_driver(cp, case, sources, 20260812, prefer_gpu=True)
    candidate = build_driver(cp, case, sources, 20260812, prefer_gpu=True)
    control = (build_driver(cp, case, (), 20260812, prefer_gpu=True)
               if sources else None)
    monitors = []
    if case.get("flux"):
        for driver in (reference, candidate):
            monitors.append(driver.add_flux_monitor(**case["flux"]))
    undo = lambda: None  # noqa: E731
    try:
        plans = build_plans(candidate)
        if mutate_step_d:
            grid = candidate.fields.grid
            invariant = tuple(grid.is_invariant(axis) for axis in range(3))
            # The WRONG side's scalars: magnetic=True on step_D — the D
            # negation (stepping.py:913-914) dropped at composition level.
            plans["step_D"].ks = tuple(bfast_curl.bfast_curl_coefficients(
                grid.bfast_scaled_k, invariant, magnetic=True))
        missing = REQUIRED_STATE - set(state(candidate))
        if missing:
            raise AssertionError(f"{name}: state inventory is short: "
                                 f"{sorted(missing)}")
        counters = {slot: PlanCounter(plan) for slot, plan in plans.items()}
        undo = install(driver_module, counters, candidate.fields)
        row: Dict[str, Any] = {
            "case": name, "shape": [int(v) for v in candidate.shape],
            "boundaries": (case["boundaries"]
                           if isinstance(case["boundaries"], str)
                           else list(case["boundaries"])),
            "bfast": [repr(v) for v in case["bfast"]],
            "courant": repr(case["courant"]), "pml": dict(case["pml"]),
            "sources": [f"{d['component']} amplitude={d.get('amplitude')!r}"
                        for d in sources],
            "patched_sub_steps": sorted(plans),
            "steps": steps,
            "step_budget_note": "bit-identity is claimed for exactly this "
                                "many steps and no further",
            "per_step": [],
        }
        if monitors:
            row["resolved_decimation"] = int(monitors[0].decimation_factor)
        source_effect_first: Optional[int] = None
        loud_margin: List[str] = []
        started = time.time()
        first_divergence = None
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            if control is not None:
                control.step()
            if case.get("sync_after") == step:
                # The flux idiom's round trip on BOTH drivers: the patched
                # step_B/update_H serve the candidate's half-step, and the
                # marginally stable f_bfast_B* must come back EXACTLY
                # (driver.py:4126-4135; MEEP energy_and_flux.cpp:113/:130).
                for driver in (reference, candidate):
                    driver.synchronize_magnetic_fields()
                    driver.restore_magnetic_fields()
            cp.cuda.runtime.deviceSynchronize()
            ref_state = state(reference)
            got_state = state(candidate)
            if set(ref_state) != set(got_state):
                raise AssertionError(f"{name} step {step}: state inventory "
                                     f"differs")
            parts = {key: compare(cp, got_state[key], ref_state[key])
                     for key in sorted(ref_state)}
            if control is not None and source_effect_first is None:
                if any(not compare(cp, ref_state[key],
                                   getattr(control.fields, key))["bit_identical"]
                       for key in PRIMARY_NAMES):
                    source_effect_first = step
            if case.get("record_loud_margin"):
                loud_margin.append(repr(float(
                    cp.abs(candidate.fields.Dx).max())))
            point = {
                "step": step,
                "bit_identical": all(p["bit_identical"] for p in parts.values()),
                "differing_floats": sum(p["differing_floats"]
                                        for p in parts.values()),
                "total_floats": sum(p["total_floats"]
                                    for p in parts.values()),
                "differing_arrays": sorted(
                    key for key, p in parts.items() if not p["bit_identical"]),
            }
            row["per_step"].append(point)
            log(f"case {name} step {step}/{steps} "
                f"identical={point['bit_identical']} "
                f"source_effect_first={source_effect_first} "
                f"ndiff={point['differing_floats']} "
                f"({time.time() - started:.1f} s)")
            if not point["bit_identical"]:
                first_divergence = step
                if mutate_step_d:
                    break
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        if mutate_step_d:
            row["first_divergence"] = first_divergence
            row["sub_step_launches"] = {slot: counter.launches
                                        for slot, counter in counters.items()}
            return row
        if sources:
            deadline = case.get("deadline")
            if source_effect_first is None:
                raise AssertionError(
                    f"{name}: the source never separated the reference from "
                    f"its control — the case is vacuous")
            if deadline is not None and source_effect_first > deadline:
                raise AssertionError(
                    f"{name}: the source separated at step "
                    f"{source_effect_first}, past the stated deadline "
                    f"{deadline} — the early steps prove nothing about the "
                    f"source seam")
        expected = _expected_launches(case, steps)
        off_budget = {slot: (counter.launches, expected[slot])
                      for slot, counter in counters.items()
                      if counter.launches != expected[slot]}
        if off_budget:
            raise AssertionError(
                f"{name}: patched sub-steps launched off-budget "
                f"(got, expected): {off_budget} — a slot the monkeypatch "
                f"failed to serve makes the case vacuous")
        row["sub_step_launches"] = {slot: counter.launches
                                    for slot, counter in counters.items()}
        row["expected_launches"] = expected
        if sources:
            row["source_effect_first_step"] = source_effect_first
        if loud_margin:
            row["loud_margin_max_abs_Dx_per_step"] = loud_margin
        row["bit_identical"] = True
        return row
    finally:
        undo()
        reference.close()
        candidate.close()
        if control is not None:
            control.close()
        cp.get_default_memory_pool().free_all_blocks()


def run_armed_mutation(cp) -> Dict[str, Any]:
    """c3 with the step_D scalars built for the WRONG side — MUST diverge
    within the budget, launches counted (c3's distinct k components are what
    make the D-negation drop byte-visible on all six targets)."""
    case = next(c for c in CASES if c["name"] == "c3_full_k_3d")
    row = run_case(cp, case, mutate_step_d=True)
    launches = row["sub_step_launches"]["step_D"]
    if launches == 0:
        raise AssertionError("armed mutation DISARMED: the mutated step_D "
                             "plan never launched")
    unserved = sorted(slot for slot, n in row["sub_step_launches"].items()
                      if n == 0)
    if unserved:
        raise AssertionError(f"armed leg: patched sub-steps never launched: "
                             f"{unserved}")
    if row.get("first_divergence") is None:
        raise AssertionError(
            f"armed mutation NOT CAUGHT in {row['steps']} steps with "
            f"{launches} mutated launches — the per-step compare cannot see "
            f"the dropped D negation and the harness is not trustworthy")
    return {"case": row["case"],
            "mutation": "step_D scalars built for the magnetic side (the "
                        "D-side negation of stepping.py:913-914 dropped)",
            "launches": launches,
            "sub_step_launches": row["sub_step_launches"],
            "caught_at_step": row["first_divergence"], "caught": True}


# ---------------------------------------------------------------------------
# Laptop self-check — case builders + array path, no CUDA, no Triton
# ---------------------------------------------------------------------------

def self_check(out_path: str) -> int:
    """Build every case's driver on NumPy and step the ARRAY PATH two steps —
    which EXERCISES the BFAST fold (stepping.py:392-396/:475-478 run on every
    bfast step). The Triton/CuPy legs are skipped and the artifact says so."""
    payload: Dict[str, Any] = {
        "mode": "self-check (laptop, NumPy array path only)",
        "triton_legs": "SKIPPED: no CUDA/triton on this host — the plan "
                       "construction, kernel launches, and per-step "
                       "bit-identity claims are NOT exercised here",
        "total_step_budget": TOTAL_BUDGET,
        "marquee_courant_check": {
            "kx": repr(KX_MARQUEE),
            "courant": repr(COURANT_MARQUEE),
            "identity": "(1 - kx)/sqrt(3), MEEP test_refl_angular.py:50-52",
            "holds": abs(COURANT_MARQUEE
                         - (1.0 - KX_MARQUEE) / np.sqrt(3.0)) == 0.0,
        },
        "cases": [],
    }
    save(payload, out_path)
    failures = 0
    for case in CASES:
        started = time.time()
        row: Dict[str, Any] = {"case": case["name"], "steps_run": 0,
                               "declared_budget": case["steps"]}
        try:
            driver = build_driver(np, case, tuple(case["sources"]), 20260812,
                                  prefer_gpu=False)
        except Exception as exc:  # noqa: BLE001 - each failure is recorded
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
            log(f"[self-check] {case['name']}: BUILD FAILED {row['error']}")
            payload["cases"].append(row)
            save(payload, out_path)
            continue
        try:
            if not driver.fields.grid.bfast_active:
                raise AssertionError("the built grid carries no bfast; the "
                                     "case would not exercise the fold")
            inventory = sorted(state(driver))
            missing = REQUIRED_STATE - set(inventory)
            if missing:
                raise AssertionError(f"state inventory short: {sorted(missing)}")
            if case.get("flux"):
                monitor = driver.add_flux_monitor(**case["flux"])
                row["resolved_decimation"] = int(monitor.decimation_factor)
            for _ in range(2):
                driver.step()
                row["steps_run"] += 1
            total = sum(float(np.abs(np.asarray(getattr(driver.fields, n))).sum())
                        for n in PRIMARY_NAMES)
            state_total = sum(
                float(np.abs(np.asarray(getattr(driver.fields, n))).sum())
                for n in BFAST_ALL)
            if not np.isfinite(total) or not np.isfinite(state_total):
                raise AssertionError("array path produced non-finite state")
            if case.get("seed_state") and state_total == 0.0:
                raise AssertionError("the seeded f_bfast state is zero; the "
                                     "-2*state layer would be invisible")
            row.update({"shape": [int(v) for v in driver.shape],
                        "state_arrays": len(inventory),
                        "primary_l1_after_2_steps": total,
                        "f_bfast_l1_after_2_steps": state_total,
                        "ok": True})
            log(f"[self-check] {case['name']}: shape={tuple(driver.shape)} "
                f"2 array-path steps ok ({time.time() - started:.1f} s)")
        except Exception as exc:  # noqa: BLE001 - each failure is recorded
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
            log(f"[self-check] {case['name']}: FAILED {row['error']}")
        finally:
            driver.close()
        payload["cases"].append(row)
        save(payload, out_path)
    payload["summary"] = {
        "built": sum(1 for c in payload["cases"] if c.get("ok")),
        "failures": failures,
        "status": "passed" if failures == 0 else "FAILED",
    }
    save(payload, out_path)
    log(f"[self-check] {payload['summary']}")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def environment(cp) -> Dict[str, Any]:
    import triton  # noqa: PLC0415

    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "triton": triton.__version__,
        "device": properties["name"].decode(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--self-check", action="store_true",
                        help="laptop mode: build the cases on NumPy, no CUDA")
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check(args.out)

    try:
        import cupy as cp  # noqa: PLC0415
        import triton  # noqa: PLC0415, F401
    except ImportError as exc:
        log(f"SKIPPED cleanly: this probe's measurement legs need CUDA "
            f"(cupy + triton); this host has neither usable: {exc}. "
            f"Run --self-check for the laptop-safe case-builder validation.")
        return 75

    from meep_gpu import backends  # noqa: PLC0415

    # Strip FIRST, guard second (the demonstrated composition order).
    gate.install_ftz_strip()
    backends.guard_kernel_compilation(cp)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    bfast_gate.write_provenance(out_dir)

    payload: Dict[str, Any] = {
        "environment": environment(cp),
        "subnormal_policy": gate.policy_stamp("cupy"),  # refreshed at the end
        "marquee": {"kx": repr(KX_MARQUEE), "courant": repr(COURANT_MARQUEE)},
        "step_budget": {case["name"]: case["steps"] for case in CASES},
        "total_step_budget": TOTAL_BUDGET,
        "step_budget_note": "bit-identity is claimed per case for exactly the "
                            "stated budget and no further",
        "fresh_fields_note": "every case builds fresh drivers: the marginally "
                             "stable f_bfast IIR makes cross-case reuse "
                             "contaminating",
        "cases": [],
    }
    save(payload, args.out)
    for case in CASES:
        log(f"starting {case['name']}: bfast={case['bfast']} "
            f"courant={case['courant']!r} pml={case['pml']} "
            f"sources={[d['component'] for d in case['sources']]}")
        payload["cases"].append(run_case(cp, case))
        save(payload, args.out)
    payload["armed_mutation"] = run_armed_mutation(cp)
    save(payload, args.out)

    payload["subnormal_policy"] = gate.policy_stamp("cupy")
    residual = gate.ftz_strip_license_reasons()
    if residual:
        payload["summary"] = {"status": "FAILED",
                              "reason": "subnormal policy: "
                                        + "; ".join(residual)}
        save(payload, args.out)
        log(f"SUBNORMAL POLICY FAILURE AT CLOSE: {residual}")
        return 1

    ran = payload["cases"]
    total_steps = sum(row["steps"] for row in ran)
    payload["summary"] = {
        "status": "passed",
        "certified_under_subnormal_policy":
            payload["subnormal_policy"].get("policy"),
        "cases_exact": f"{len(ran)}/{len(ran)}",
        "complete_steps_exact": f"{total_steps}/{total_steps}",
        "source_cases_nonvacuous": f"{sum(1 for r in ran if r['sources'])}/"
                                   f"{sum(1 for r in ran if r['sources'])}",
        "sync_round_trip_case": "c6_sync_flux",
        "resolved_decimation_c7": next(
            (r.get("resolved_decimation") for r in ran
             if r["case"] == "c7_dft_flux_decimation"), None),
        "armed_mutation_caught_at_step":
            payload["armed_mutation"]["caught_at_step"],
    }
    save(payload, args.out)
    log(f"BFAST COMPOSITION GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
