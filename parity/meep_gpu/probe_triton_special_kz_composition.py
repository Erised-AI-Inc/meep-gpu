"""Complete-driver CUDA gate for the special_kz (grid.beta) Triton composition.

The sub-step gate (``gate_triton_special_kz.py``) certifies each kernel against
an in-file transcription; this complementary probe asks the composition
question: with ``step_B``/``update_H``/``step_D``/``update_E`` served by
``special_kz``' engine-route plans — the beta curls plus the beta-run
constitutive delegations to the CERTIFIED kernels — installed by monkeypatch,
production dispatch untouched, ``plan_fast_path`` still returning None — does
the complete CuPy driver state reproduce the array-path driver EXACTLY after
every step, for a STATED number of steps, sources, PML, boundary fills and
metal zeroing included?

Seven core cases (8 complete steps each — the 56-step core budget), one extra
dispersion case, and one 64-step long leg on each of the two marquee cases:

1. ``r1_kz2d_pml_xy``        — refl-angular-kz2d's beta (0.3321611318837033),
   2-D PML x+y, Courant 0.5, source-free. MARQUEE: also run 64 steps.
2. ``r2_electric_src``       — case 1 + an Ez Gaussian point source, proven
   non-vacuous against a source-free control.
3. ``r2_magnetic_src``       — case 1 + an Hy source (closes the B seam).
4. ``r3_metallic_bloch_sign``— metallic x / periodic y, the grating case's
   NEGATIVE beta (-0.6850526103319672), Courant 0.34 (non-power-of-two).
5. ``r4_odd_shape``          — 37x29x1 with the 1-cell invariant axis,
   beta = 0.2, second NP2 Courant (0.4375).
6. ``c1_complex_k0_src``     — beta = 0.2, in-plane k = 0, complex storage,
   PML both axes, Ez source with a COMPLEX amplitude vs a source-free control.
7. ``c2_special_kz_marquee`` — beta = -0.39073112848927377 + in-plane Bloch
   kx = 0.9205048534524404 (test_special_kz's own numbers), Courant 0.35,
   source-free. MARQUEE: also run 64 steps.
8. ``r7_lorentz_pole``       — one Lorentz pole + beta (R7): the beta curls
   and the beta-run update_H serve the step; ``update_P`` and the dispersive
   ``update_E`` stay the driver's OWN array path (a registered polarization is
   refused by the beta-run E-side predicate exactly as the shipped predicate
   refuses it — recorded as the expected refusal, not patched around).

An ARMED mutation leg re-runs case 7 (c2) with the step_D plan's beta
coefficient built for the WRONG side (the ±i swap, m2 at composition level).
The leg counts the mutated plan's launches and FAILS LOUDLY if it never ran or
never diverged.

Requirements carried from prior defects: per-complete-step uint32 compare of
EVERY allocated primary and auxiliary; non-power-of-two Courants in the sweep;
the step budget STATED in the artifact; unbuffered per-case progress; atomic
JSON rewrite per case; own provenance record; and PER-SLOT launch counters on
EVERY patched sub-step of EVERY case — a monkeypatch that silently fell
through for a slot (fields-identity mismatch, a driver call route not via the
module globals) would leave candidate and reference both on the array path
and every step would pass vacuously, so each patched slot must prove it
launched exactly once per complete step.

SUBNORMAL POLICY. This probe runs UNDER THE STRIPPED IEEE-KEEP POLICY — the
ship configuration — installed via ``gate_triton_complex.install_ftz_strip``
BEFORE ``guard_kernel_compilation`` so the guard wraps the strip and both
apply. A probe artifact handed in with ``--probe-artifact`` must certify that
same policy and carry the EXTENDED pattern set (it feeds
``special_kz.beta_expansion_from_probe``); every claim states the policy it
was cut under.

Usage (the GPU host, one clear device; the cache dir must carry the policy token)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_special_kz_composition.py \\
        --out results/triton_special_kz_<date>/composition.json

Laptop (no CUDA/triton): ``--self-check`` builds every case's drivers on NumPy,
steps the array path two steps per case (the beta term IS in that path), and
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

# The policy machinery, atomic writer and seeding live in the complex gate;
# the beta expansion measurement and provenance live in the beta gate.
import gate_triton_complex as gate  # noqa: E402
import gate_triton_special_kz as beta_gate  # noqa: E402

SEED = gate.SEED
save = gate.save
log = gate.log

BETA_KZ2D = beta_gate.BETA_KZ2D
BETA_GRATING = beta_gate.BETA_GRATING
BETA_SPECIAL_KZ = beta_gate.BETA_SPECIAL_KZ
KX_SPECIAL_KZ = beta_gate.KX_SPECIAL_KZ

STATE_NAMES = gate.ALL_STATE
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
REQUIRED_STATE = set(STATE_NAMES)


def cw_source(component: str, amplitude: complex) -> Dict[str, Any]:
    return {"component": component, "frequency": 0.7,
            "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
            "amplitude": amplitude}


def gaussian_source(component: str) -> Dict[str, Any]:
    return {"component": component, "source_type": "gaussian",
            "frequency": 0.8, "fwidth": 0.4,
            "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
            "amplitude": 1.0}


#: name, cell, resolution, boundaries, k_point, beta, courant, pml, sources,
#: steps, complex_storage, lorentz_pole, marquee.
CASES: Tuple[Tuple[Any, ...], ...] = (
    ("r1_kz2d_pml_xy", (3.0, 3.0, 0.0), 12.0, "periodic", (0.0, 0.0, 0.0),
     BETA_KZ2D, 0.5, {"x": 5, "y": 5}, (), 8, False, False, True),
    ("r2_electric_src", (3.0, 3.0, 0.0), 12.0, "periodic", (0.0, 0.0, 0.0),
     BETA_KZ2D, 0.5, {"x": 5, "y": 5}, (gaussian_source("Ez"),), 8, False,
     False, False),
    ("r2_magnetic_src", (3.0, 3.0, 0.0), 12.0, "periodic", (0.0, 0.0, 0.0),
     BETA_KZ2D, 0.5, {"x": 5, "y": 5}, (cw_source("Hy", 0.75),), 8, False,
     False, False),
    ("r3_metallic_bloch_sign", (3.0, 3.0, 0.0), 12.0,
     ("metallic", "periodic", "periodic"), (0.0, 0.0, 0.0), BETA_GRATING,
     0.34, {"x": 5, "y": 5}, (), 8, False, False, False),
    ("r4_odd_shape", (37.0, 29.0, 0.0), 1.0, "periodic", (0.0, 0.0, 0.0),
     0.2, 0.4375, {"x": 5, "y": 5}, (), 8, False, False, False),
    ("c1_complex_k0_src", (3.0, 3.0, 0.0), 12.0, "periodic", (0.0, 0.0, 0.0),
     0.2, 0.35, {"x": 5, "y": 5}, (cw_source("Ez", complex(0.75, 0.4)),), 8,
     True, False, False),
    ("c2_special_kz_marquee", (3.0, 3.0, 0.0), 12.0, "periodic",
     (KX_SPECIAL_KZ, 0.0, 0.0), BETA_SPECIAL_KZ, 0.35, {"x": 5, "y": 5}, (),
     8, True, False, True),
    ("r7_lorentz_pole", (3.0, 3.0, 0.0), 12.0, "periodic", (0.0, 0.0, 0.0),
     BETA_KZ2D, 0.4, {"x": 5, "y": 5}, (gaussian_source("Ez"),), 8, False,
     True, False),
)
MARQUEE_STEPS = 64
CORE_BUDGET = sum(case[9] for case in CASES if not case[11])  # 7 x 8 = 56


def build_driver(xp, name, cell, resolution, boundaries, k_point, beta,
                 courant, pml_spec, source_specs, complex_storage,
                 lorentz_pole, seed: int, prefer_gpu: bool):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=resolution, dimensions=2,
        force_complex_fields=complex_storage, courant=courant,
        boundaries=boundaries, k_point=k_point, beta=beta,
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(xp.asarray(epsilon))
    if lorentz_pole:
        from meep_gpu.dispersion import Susceptibility  # noqa: PLC0415

        driver.add_susceptibility(
            Susceptibility(frequency=1.1, gamma=1e-5, kind="lorentzian"),
            0.5)
    driver.setup_pml(dict(pml_spec))
    for declaration in source_specs:
        driver.add_source(dict(declaration))
    rng = np.random.default_rng(seed)
    for primary in PRIMARY_NAMES:
        if complex_storage:
            host = gate._seed_host_complex(tuple(shape), rng, scale=0.25)
        else:
            host = (0.25 * beta_gate._seed_host_real(tuple(shape), rng))
            host = np.ascontiguousarray(host.astype(np.float32))
        driver.set_field(primary, xp.asarray(host))
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


def build_plans(driver, record, complex_storage: bool,
                lorentz_pole: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The four sub-step plans (or three for the dispersion case), plus the
    expected refusals recorded rather than patched around."""
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    expected_refusals: Dict[str, Any] = {}
    if complex_storage:
        plans = {
            "step_B": special_kz.plan_beta_bloch_pml_curl(
                driver.fields, driver.pml, "step_B", probe=record),
            "update_H": special_kz.plan_beta_run_complex_constitutive(
                driver.fields, driver.pml, "H", probe=record),
            "step_D": special_kz.plan_beta_bloch_pml_curl(
                driver.fields, driver.pml, "step_D", probe=record),
            "update_E": special_kz.plan_beta_run_complex_constitutive(
                driver.fields, driver.pml, "E", probe=record),
        }
    else:
        plans = {
            "step_B": special_kz.plan_beta_pml_curl(
                driver.fields, driver.pml, "step_B"),
            "update_H": special_kz.plan_beta_run_constitutive(
                driver.fields, driver.pml, "H"),
            "step_D": special_kz.plan_beta_pml_curl(
                driver.fields, driver.pml, "step_D"),
            "update_E": special_kz.plan_beta_run_constitutive(
                driver.fields, driver.pml, "E"),
        }
    if lorentz_pole:
        # R7: a registered polarization means update_E's source is (D - sum P)
        # — the beta-run E-side predicate refuses it exactly as the shipped
        # one does, and the driver's own array path serves update_E/update_P.
        refused = plans.pop("update_E")
        if refused is not None:
            raise AssertionError(
                "r7: the beta-run E-side predicate ADMITTED a dispersive run "
                "— the ADE ownership split has been broken")
        expected_refusals["update_E"] = (
            "a susceptibility is registered: update_E belongs to the ADE "
            "path; served by the driver's own array path in this case")
    missing = sorted(slot for slot, plan in plans.items() if plan is None)
    if missing:
        raise AssertionError(f"plans refused for {missing}")
    return plans, expected_refusals


class PlanCounter:
    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.launches = 0

    def run(self, guard: Optional[bool] = None) -> None:
        self.launches += 1
        self.plan.run(guard=guard)


def install(driver_module, plans: Dict[str, Any], owner):
    """Serve the patched sub-steps from the plans, for the owner fields only.
    Sources, boundary fills, ``zero_metal_*`` and ``update_P`` stay the
    driver's own. Production dispatch is untouched."""
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


def run_case(cp, record, name, cell, resolution, boundaries, k_point, beta,
             courant, pml_spec, source_specs, steps, complex_storage,
             lorentz_pole, _marquee, steps_override=None) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    steps = steps_override or steps
    build = lambda sources: build_driver(  # noqa: E731
        cp, name, cell, resolution, boundaries, k_point, beta, courant,
        pml_spec, sources, complex_storage, lorentz_pole, 20260811,
        prefer_gpu=True)
    reference = build(source_specs)
    candidate = build(source_specs)
    control = build(()) if source_specs else None
    undo = lambda: None  # noqa: E731
    try:
        plans, expected_refusals = build_plans(candidate, record,
                                               complex_storage, lorentz_pole)
        missing = REQUIRED_STATE - set(state(candidate))
        if missing:
            raise AssertionError(f"{name}: state inventory is short: "
                                 f"{sorted(missing)}")
        # EVERY patched slot is launch-counted: an install() that silently
        # fell through for a slot would run the array path on both sides and
        # pass vacuously — each slot must launch exactly once per step.
        counters = {slot: PlanCounter(plan) for slot, plan in plans.items()}
        undo = install(driver_module, counters, candidate.fields)
        row: Dict[str, Any] = {
            "case": name, "shape": [int(v) for v in candidate.shape],
            "boundaries": boundaries if isinstance(boundaries, str)
            else list(boundaries),
            "k_point": list(k_point), "beta": repr(beta),
            "courant": repr(courant), "pml": dict(pml_spec),
            "sources": [f"{d['component']} amplitude={d.get('amplitude')!r}"
                        for d in source_specs],
            "patched_sub_steps": sorted(plans),
            "expected_refusals": expected_refusals,
            "steps": steps,
            "step_budget_note": "bit-identity is claimed for exactly this "
                                "many steps and no further",
            "per_step": [],
        }
        source_effect_seen = not source_specs
        started = time.time()
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            if control is not None:
                control.step()
            cp.cuda.runtime.deviceSynchronize()
            ref_state = state(reference)
            got_state = state(candidate)
            if set(ref_state) != set(got_state):
                raise AssertionError(f"{name} step {step}: state inventory "
                                     f"differs")
            parts = {key: compare(cp, got_state[key], ref_state[key])
                     for key in sorted(ref_state)}
            if control is not None:
                source_effect_seen = source_effect_seen or any(
                    not compare(cp, ref_state[key],
                                getattr(control.fields, key))["bit_identical"]
                    for key in PRIMARY_NAMES)
            point = {
                "step": step,
                "bit_identical": all(p["bit_identical"] for p in parts.values()),
                "differing_floats": sum(p["differing_floats"]
                                        for p in parts.values()),
                "total_floats": sum(p["total_floats"] for p in parts.values()),
                "differing_arrays": sorted(
                    key for key, p in parts.items() if not p["bit_identical"]),
                "source_effect_seen": source_effect_seen,
            }
            row["per_step"].append(point)
            log(f"case {name} step {step}/{steps} "
                f"identical={point['bit_identical']} "
                f"source_effect={source_effect_seen} "
                f"ndiff={point['differing_floats']} "
                f"({time.time() - started:.1f} s)")
            if not point["bit_identical"]:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        if not source_effect_seen:
            raise AssertionError(
                f"{name}: the source never separated the reference from its "
                f"control — the case is vacuous")
        off_budget = {slot: counter.launches
                      for slot, counter in counters.items()
                      if counter.launches != steps}
        if off_budget:
            raise AssertionError(
                f"{name}: patched sub-steps launched off-budget (expected "
                f"{steps} each): {off_budget} — a slot the monkeypatch failed "
                f"to serve makes the case vacuous")
        row["sub_step_launches"] = {slot: counter.launches
                                    for slot, counter in counters.items()}
        row["bit_identical"] = True
        row["source_effect_seen"] = source_effect_seen
        return row
    finally:
        undo()
        reference.close()
        candidate.close()
        if control is not None:
            control.close()
        cp.get_default_memory_pool().free_all_blocks()


def run_armed_mutation(cp, record) -> Dict[str, Any]:
    """c2 with the step_D beta coefficient built for the WRONG side (the ±i
    swap) — MUST diverge within the budget, launches counted."""
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    case = next(c for c in CASES if c[0] == "c2_special_kz_marquee")
    (name, cell, resolution, boundaries, k_point, beta, courant, pml_spec,
     source_specs, steps, complex_storage, lorentz_pole, _marquee) = case
    build = lambda: build_driver(  # noqa: E731
        cp, name, cell, resolution, boundaries, k_point, beta, courant,
        pml_spec, source_specs, complex_storage, lorentz_pole, 20260811,
        prefer_gpu=True)
    reference = build()
    candidate = build()
    undo = lambda: None  # noqa: E731
    try:
        plans, _refusals = build_plans(candidate, record, complex_storage,
                                       lorentz_pole)
        grid = candidate.fields.grid
        # The wrong side's words: magnetic=True on the D sub-step (+1j where
        # the transcription demands -1j) — stepping.py:798-799 inverted.
        wrong_words = special_kz.beta_curl_coefficients(
            grid.beta, grid.dt, magnetic=True, complex_storage=True)
        mutated = plans["step_D"]
        mutated.beta_words = tuple(tuple(float(w) for w in pair)
                                   for pair in wrong_words)
        counters = {slot: PlanCounter(plan) for slot, plan in plans.items()}
        counter = counters["step_D"]  # the armed slot
        undo = install(driver_module, counters, candidate.fields)
        first_divergence = None
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            cp.cuda.runtime.deviceSynchronize()
            parts = {key: compare(cp, getattr(candidate.fields, key),
                                  getattr(reference.fields, key))
                     for key in PRIMARY_NAMES}
            identical = all(p["bit_identical"] for p in parts.values())
            log(f"[armed] step {step}/{steps} identical={identical} "
                f"launches={counter.launches}")
            if not identical:
                first_divergence = step
                break
        if counter.launches == 0:
            raise AssertionError("armed mutation DISARMED: the mutated step_D "
                                 "plan never launched")
        unserved = sorted(slot for slot, c in counters.items()
                          if c.launches == 0)
        if unserved:
            raise AssertionError(f"armed leg: patched sub-steps never "
                                 f"launched: {unserved}")
        if first_divergence is None:
            raise AssertionError(
                f"armed mutation NOT CAUGHT in {steps} steps with "
                f"{counter.launches} mutated launches — the per-step compare "
                f"cannot see the ±i swap and the harness is not trustworthy")
        return {"case": name, "mutation": "step_D beta coefficient built for "
                                          "the magnetic side (±i swapped)",
                "launches": counter.launches,
                "sub_step_launches": {slot: c.launches
                                      for slot, c in counters.items()},
                "caught_at_step": first_divergence, "caught": True}
    finally:
        undo()
        reference.close()
        candidate.close()
        cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# Laptop self-check — case builders + array path, no CUDA, no Triton
# ---------------------------------------------------------------------------

def self_check(out_path: str) -> int:
    """Build every case's driver on NumPy and step the ARRAY PATH two steps —
    which EXERCISES the beta term (stepping.py:384-391/:467-474 run on every
    beta step). The Triton/CuPy legs are skipped and the artifact says so."""
    payload: Dict[str, Any] = {
        "mode": "self-check (laptop, NumPy array path only)",
        "triton_legs": "SKIPPED: no CUDA/triton on this host — the plan "
                       "construction, kernel launches, and per-step "
                       "bit-identity claims are NOT exercised here",
        "core_step_budget": CORE_BUDGET,
        "cases": [],
    }
    save(payload, out_path)
    failures = 0
    for case in CASES:
        (name, cell, resolution, boundaries, k_point, beta, courant, pml_spec,
         source_specs, steps, complex_storage, lorentz_pole, _marquee) = case
        started = time.time()
        row: Dict[str, Any] = {"case": name, "steps_run": 0,
                               "declared_budget": steps}
        try:
            driver = build_driver(np, name, cell, resolution, boundaries,
                                  k_point, beta, courant, pml_spec,
                                  source_specs, complex_storage, lorentz_pole,
                                  20260811, prefer_gpu=False)
        except Exception as exc:  # noqa: BLE001 - each failure is recorded
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
            log(f"[self-check] {name}: BUILD FAILED {row['error']}")
            payload["cases"].append(row)
            save(payload, out_path)
            continue
        try:
            if float(driver.fields.grid.beta) == 0.0:
                raise AssertionError("the built grid carries beta = 0; the "
                                     "case would not exercise the term")
            inventory = sorted(state(driver))
            missing = REQUIRED_STATE - set(inventory)
            if missing:
                raise AssertionError(f"state inventory short: {sorted(missing)}")
            for _ in range(2):
                driver.step()
                row["steps_run"] += 1
            total = sum(float(np.abs(np.asarray(getattr(driver.fields, n))).sum())
                        for n in PRIMARY_NAMES)
            if not np.isfinite(total):
                raise AssertionError("array path produced non-finite state")
            row.update({"shape": [int(v) for v in driver.shape],
                        "beta": repr(float(driver.fields.grid.beta)),
                        "state_arrays": len(inventory),
                        "polarizations": len(getattr(driver.fields,
                                                     "polarizations", ()) or ()),
                        "primary_l1_after_2_steps": total,
                        "ok": True})
            log(f"[self-check] {name}: shape={tuple(driver.shape)} "
                f"beta={row['beta']} 2 array-path steps ok "
                f"({time.time() - started:.1f} s)")
        except Exception as exc:  # noqa: BLE001 - each failure is recorded
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
            log(f"[self-check] {name}: FAILED {row['error']}")
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
    parser.add_argument("--probe-artifact", default=None,
                        help="an existing EXTENDED expansion probe JSON; "
                             "measured in-process when absent")
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
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    # Strip FIRST, guard second (the demonstrated composition order).
    gate.install_ftz_strip()
    backends.guard_kernel_compilation(cp)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    beta_gate.write_provenance(out_dir)

    if args.probe_artifact:
        with open(args.probe_artifact, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        policy_problems = gate.probe_record_policy_reasons(record)
    else:
        record = beta_gate.measure_beta_expansion_record(cp, "cupy")
        save(record, os.path.join(out_dir, "probe.json"))
        policy_problems = gate.ftz_strip_license_reasons()
    if policy_problems:
        payload = {"environment": environment(cp),
                   "subnormal_policy": gate.policy_stamp("cupy"),
                   "expansion_probe": record,
                   "summary": {"status": "REFUSED",
                               "certified_under_subnormal_policy": None,
                               "reason": "subnormal policy: "
                                         + "; ".join(policy_problems)}}
        save(payload, args.out)
        log(f"SUBNORMAL POLICY REFUSAL: {policy_problems}")
        return 1
    # THE VERDICT, not the boolean. 'measured' against 'environment_default' is
    # the difference between this run having discriminated the arm and having
    # inherited it, and ``non_discriminating`` names the patterns whose MEASURED
    # blindness took them out of the agreement test — what a reader needs to see
    # what did and did not contribute. Written on both paths, refusal included,
    # so a refusal carries its own reasons rather than one paraphrase of three.
    expansion_verdict = special_kz.beta_expansion_license(record)
    if expansion_verdict["expansion"] is None:
        payload = {"environment": environment(cp),
                   "subnormal_policy": gate.policy_stamp("cupy"),
                   "expansion_probe": record,
                   "expansion_license": expansion_verdict,
                   "summary": {"status": "REFUSED",
                               "reason": "the extended expansion probe "
                                         "licenses no single constexpr; the "
                                         "complex beta kernel may not launch — "
                                         + "; ".join(
                                             expansion_verdict["refusals"])}}
        save(payload, args.out)
        log(f"EXPANSION REFUSAL: {record.get('patterns')} "
            f"-> {expansion_verdict['refusals']}")
        return 1
    log(f"[expansion] arm={expansion_verdict['arm']} "
        f"basis={expansion_verdict['basis']} discriminating="
        f"{sorted(expansion_verdict['discriminating'])} non_discriminating="
        f"{expansion_verdict['non_discriminating']}")

    payload: Dict[str, Any] = {
        "environment": environment(cp),
        "expansion_patterns": record["patterns"],
        "expansion_license": expansion_verdict,
        "subnormal_policy": gate.policy_stamp("cupy"),  # refreshed at the end
        "step_budget": {case[0]: case[9] for case in CASES},
        "core_step_budget": CORE_BUDGET,
        "marquee_long_legs": {case[0]: MARQUEE_STEPS
                              for case in CASES if case[12]},
        "step_budget_note": "bit-identity is claimed per case for exactly the "
                            "stated budget and no further",
        "cases": [],
    }
    save(payload, args.out)
    for case in CASES:
        log(f"starting {case[0]}: beta={case[5]!r} k={case[4]} pml={case[7]} "
            f"sources={[d['component'] for d in case[8]]}")
        payload["cases"].append(run_case(cp, record, *case))
        save(payload, args.out)
    for case in CASES:
        if not case[12]:
            continue
        log(f"starting MARQUEE long leg {case[0]}: {MARQUEE_STEPS} steps")
        row = run_case(cp, record, *case, steps_override=MARQUEE_STEPS)
        row["case"] = row["case"] + "_long"
        payload["cases"].append(row)
        save(payload, args.out)
    payload["armed_mutation"] = run_armed_mutation(cp, record)
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
        "r7_expected_refusal_recorded": any(
            r.get("expected_refusals") for r in ran),
        "armed_mutation_caught_at_step":
            payload["armed_mutation"]["caught_at_step"],
    }
    save(payload, args.out)
    log(f"SPECIAL_KZ COMPOSITION GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
