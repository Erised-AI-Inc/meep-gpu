"""Complete-driver CUDA gate for the chi2/chi3 nonlinear Triton composition.

The sub-step gate (``gate_triton_nonlinear.py``) certifies the kernel against an
in-file transcription; this complementary probe asks the composition question:
with ``step_B``/``step_D`` served by the CERTIFIED curl kernel, ``update_H`` by
the CERTIFIED constitutive kernel, and ``update_E`` by the NEW nonlinear kernel
— installed by monkeypatch, production dispatch untouched, ``plan_fast_path``
still returning None — does the complete driver state reproduce the array-path
driver EXACTLY after every step, for a STATED number of steps, sources, PML,
metal zeroing and the nonlinear pole guard included?

THIS PROBE IS THE NARROWING MEASUREMENT. ``coverage._grid_reasons`` clause 10
(coverage.py:195-196) refuses a chi2/chi3 run at EVERY sub-step, while the
nonlinearity's arithmetic lives in ``update_E`` alone (stepping.py:975-979).
Before each case runs, the probe asserts that the certified curl and update_H
predicates refuse the run FOR THE CHI CLAUSE AND NOTHING ELSE — so the
composition that then reproduces the array path byte-for-byte is exactly the
evidence the wiring change needs to narrow the shared clause, the way plan
§13.1 narrowed the curl's dispersion refusal. The certified plans are built
through their ``from_arrays`` routes for that reason: their engine builders
still refuse, deliberately, until the narrowing lands.

The cases span the corpus demand and nothing else:

1. ``marquee_3rd_harm_quiet`` — 3rd-harm-1d.py's own configuration (1-D cell
   (0,0,100) at resolution 25 -> 2500 z-cells + PML both ends, uniform
   index 1, SCALAR chi3, Gaussian Ex source; the script's ``k`` is the chi3),
   at a quiet chi3, 8 complete steps. Built by ``from_meep.lift_simulation``
   when MEEP is importable (the corpus row's own route) and by direct driver
   construction otherwise — the artifact records which route ran.
2. ``marquee_3rd_harm_loud``  — the same configuration with the D field scaled
   so the measured ``nonlinear_margin`` expansion sits NEAR the pole guard's
   derived 1/3 bound (never past it), 64 complete steps — the large-u
   arithmetic byte-exercised through whole driver steps.
3. ``synthetic_3d_chi2_chi3`` — 3-D, PML on two axes with the third a bare
   metallic wall, Courant 0.35 (the mandate's home in this probe: dtdx enters
   the CURL kernels here), chi2 + chi3, Ez Gaussian source, seeded primaries,
   proven non-vacuous against a source-free control.
4. ``partial_ez``             — chi3 on Ez ONLY: Ex/Ey take the new kernel's
   compiled plain arm while Ez takes the Pade arm, over whole steps.
5. ``dft_flux_decimation``    — case 3 plus a flux monitor at the engine's
   AUTOMATICALLY resolved decimation; the resolved factor is recorded beside
   what MEEP's ``has_nonlinearities`` guard resolves for a nonlinear run
   (1, dft.cpp:207-210). RESOLVED 2026-08-12: the engine now mirrors that
   guard in ``_automatic_decimation_factor`` and both drivers resolve 1
   here. The historical measurement stands: before the fix the engine
   resolved 21 from source bandwidth alone (Gaussian fcen=0.8 df=0.4 on a
   chi2/chi3 run) — the discrepancy this case recorded for the wiring
   change. Flux spectra are compared bit-exactly at the end alongside the
   per-step field compare.

   NON-VACUITY, added 2026-08-12 after the certifying run was
   found to have compared two untouched buffers: at 16 steps against the
   factor of 21 the engine resolved then, NO sample ever accumulated and the
   recorded spectrum was ``[0, 0, 0, 0, 0]`` — "bit-exact" over exact zeros
   certifies nothing. This probe now adopts the off-diagonal tranche's
   remedy (``probe_triton_offdiag_composition.py``): the step budget is
   asserted at monitor-add time to be at least twice the engine's resolved
   decimation, ``flux_samples_in_budget`` is recorded, and an identically
   zero spectrum is REFUSED before the compare, in the device path and in
   the laptop self-check alike. THE CASE HAS NOT BEEN RE-RUN ON THE DEVICE
   since the fix, so its flux claim is uncertified; only the laptop
   self-check's array-path leg has exercised the new assertions.

Every composition case is LAUNCH-COUNTED: all four installed plans are
wrapped in :class:`PlanCounter` and the artifact records per-slot launches;
a slot whose count differs from the case's step budget fails the case loudly
(the monkeypatch stopped intercepting, or the driver's call structure
changed) — bit-identical rows without launch evidence certify nothing.

An ARMED mutation leg re-runs case 3 with the nonlinear plan's chi3 scalar arm
doubled — the plan-level analogue of a wrong material — launch-counted, with
DISARMED and NOT-CAUGHT failures loud. A refusal-enumeration leg reuses the
sub-step gate's ``run_refusals`` so the composition artifact carries the same
0-admissions record.

Requirements carried from prior defects: per-complete-step uint32 compare of
EVERY allocated stored volume; the step budget STATED in the artifact;
unbuffered per-case progress; atomic JSON rewrite per case; own provenance
record. SUBNORMAL POLICY: runs UNDER the stripped IEEE-keep policy via the
complex gate's ``install_ftz_strip`` (strip FIRST, compilation guard second),
and refuses to certify when the strip cannot be confirmed exercised.

Usage (the GPU host)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_nonlinear_composition.py \\
        --out results/triton_nonlinear_<date>/composition.json

Laptop (no CUDA/triton): ``--self-check`` builds every case's drivers on NumPy,
steps the ARRAY PATH through each case's full budget (the loud marquee's pole
margin is measured over all 64 steps), validates the narrowing assertion's
reason structure, and says in so many words that the Triton legs were skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gate_triton_complex as complex_gate  # noqa: E402  (policy machinery)
import gate_triton_nonlinear as gate  # noqa: E402

from meep_gpu import stepping  # noqa: E402

SEED = gate.SEED
save = gate.save
log = gate.log

#: 3rd-harm-1d.py's cell/PML/resolution family (KEY FACTS: 1-D (0,0,100),
#: res 25, PML, k=None -> metallic walls, Medium(index=1, chi3=k)).
MARQUEE_CELL_Z = 100.0
MARQUEE_RESOLUTION = 25.0
MARQUEE_DPML = 1.0
MARQUEE_FCEN = 1.0 / 3.0
MARQUEE_DF = MARQUEE_FCEN / 20.0
CHI3_QUIET = 0.02
CHI3_LOUD = 0.3
#: The loud marquee's INITIAL target expansion. The seeded state self-focuses
#: over the 64-step budget (measured on the array path: ~1.9x growth), so the
#: initial point is placed so the FINAL margin sits near the derived 1/3 bound
#: without crossing it — the self-check walks the whole budget and fails loudly
#: if the drift ever crosses.
LOUD_MARGIN_TARGET = 0.1

FIELD_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
               "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
AUX_STATE = ("fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
             "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")


def gaussian_source(component: str, frequency: float, fwidth: float,
                    center, amplitude: float = 1.0) -> Dict[str, Any]:
    return {"component": component, "source_type": "gaussian",
            "frequency": frequency, "fwidth": fwidth,
            "center": tuple(center), "size": (0.0, 0.0, 0.0),
            "amplitude": amplitude}


#: name, builder kwargs, steps, seed amplitude, margin target (None = leave).
CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "marquee_3rd_harm_quiet", "kind": "marquee", "chi3": CHI3_QUIET,
     "steps": 8, "seed_amplitude": 0.05, "margin_target": None},
    {"name": "marquee_3rd_harm_loud", "kind": "marquee", "chi3": CHI3_LOUD,
     "steps": 64, "seed_amplitude": 0.3, "margin_target": LOUD_MARGIN_TARGET},
    {"name": "synthetic_3d_chi2_chi3", "kind": "synthetic", "steps": 10,
     "seed_amplitude": 0.25, "margin_target": None,
     "chi2": 0.045, "chi3": 0.08},
    {"name": "partial_ez", "kind": "partial", "steps": 12,
     "seed_amplitude": 0.25, "margin_target": None, "chi3": 0.08},
    # The budget must clear 2x the engine's RESOLVED decimation or the flux
    # compare is vacuous — asserted at monitor-add time, in the device path
    # and the self-check both. 16 clears the factor of 1 a nonlinear run now
    # resolves (dft.cpp:207-210's guard, mirrored in the engine); it did NOT
    # clear the factor of 21 the engine resolved for the certifying run of this
    # case, which is why that run's spectrum was identically zero.
    {"name": "dft_flux_decimation", "kind": "synthetic", "steps": 16,
     "seed_amplitude": 0.25, "margin_target": None,
     "chi2": 0.045, "chi3": 0.08, "flux": True},
)


def build_marquee_via_meep(chi3: float, prefer_gpu: bool):
    """3rd-harm-1d.py's own construction, through from_meep.lift_simulation."""
    import meep as mp  # noqa: PLC0415

    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415

    sim = mp.Simulation(
        cell_size=mp.Vector3(0, 0, MARQUEE_CELL_Z),
        geometry=[],
        sources=[mp.Source(mp.GaussianSource(MARQUEE_FCEN, fwidth=MARQUEE_DF),
                           component=mp.Ex,
                           center=mp.Vector3(0, 0,
                                             -0.5 * MARQUEE_CELL_Z
                                             + MARQUEE_DPML))],
        boundary_layers=[mp.PML(MARQUEE_DPML)],
        default_material=mp.Medium(index=1, chi3=chi3),
        resolution=MARQUEE_RESOLUTION,
        dimensions=1,
    )
    return lift_simulation(sim, prefer_gpu=prefer_gpu)


def build_marquee_direct(xp, chi3: float, prefer_gpu: bool,
                         with_source: bool = True):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=(0.0, 0.0, MARQUEE_CELL_Z), resolution=MARQUEE_RESOLUTION,
        dimensions=1, boundaries=("periodic", "periodic", "metallic"),
        force_complex_fields=False,  # the script's own storage (k=None)
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    driver.set_epsilon(xp.ones(driver.shape, dtype=xp.float32))
    driver.set_chi3(chi3)
    driver.setup_pml({"z": int(round(MARQUEE_DPML * MARQUEE_RESOLUTION))})
    if with_source:
        driver.add_source(gaussian_source(
            "Ex", MARQUEE_FCEN, MARQUEE_DF,
            (0.0, 0.0, -0.5 * MARQUEE_CELL_Z + MARQUEE_DPML)))
    return driver


def build_synthetic(xp, prefer_gpu: bool, chi2: float, chi3: float,
                    partial: bool = False, with_source: bool = True):
    """3-D: PML on x and y, the z axis a bare metallic wall, Courant 0.35."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=(1.6, 1.6, 0.8), resolution=15.0, dimensions=3,
        boundaries="metallic", courant=0.35, force_complex_fields=False,
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (2.0 + 0.4 * np.sin(index * np.float32(0.041))).astype(np.float32))
    driver.set_epsilon(xp.asarray(epsilon))
    if partial:
        driver.set_chi3({"Ex": 0.0, "Ey": 0.0, "Ez": chi3})
    else:
        driver.set_chi2(chi2)
        driver.set_chi3(chi3)
    driver.setup_pml({"x": 4, "y": 4})
    if with_source:
        driver.add_source(gaussian_source("Ez", 0.8, 0.4, (0.0, 0.0, 0.0)))
    return driver


def build_case(case: Dict[str, Any], xp, prefer_gpu: bool,
               with_source: bool = True) -> Tuple[Any, str]:
    if case["kind"] == "marquee":
        if with_source:
            try:
                import meep  # noqa: F401, PLC0415

                return (build_marquee_via_meep(case["chi3"], prefer_gpu),
                        "from_meep.lift_simulation")
            except ImportError:
                pass
        return (build_marquee_direct(xp, case["chi3"], prefer_gpu,
                                     with_source=with_source),
                "direct FdtdDriver construction (meep not importable, or the "
                "source-free control, which the lifter refuses by design)")
    if case["kind"] == "partial":
        return (build_synthetic(xp, prefer_gpu, 0.0, case["chi3"],
                                partial=True, with_source=with_source),
                "direct FdtdDriver construction")
    return (build_synthetic(xp, prefer_gpu, case["chi2"], case["chi3"],
                            with_source=with_source),
            "direct FdtdDriver construction")


def seed_primaries(driver, xp, amplitude: float, seed: int = 20260812) -> None:
    rng = np.random.default_rng(seed)
    for name in PRIMARY_NAMES:
        host = np.ascontiguousarray(
            (amplitude * rng.uniform(-1.0, 1.0, size=driver.shape))
            .astype(np.float32))
        driver.set_field(name, xp.asarray(host))


def measured_margin(driver) -> Optional[float]:
    margin = stepping.nonlinear_margin(driver.fields, driver.pml)
    return None if margin is None else float(margin.expansion)


def scale_factor_for_margin(driver, target: float) -> float:
    """The float32 D-scale that puts the measured expansion near ``target``."""
    low, high = 0.0, 1.0
    originals = {name: getattr(driver.fields, "D" + axis).copy()
                 for name, axis in (("Dx", "x"), ("Dy", "y"), ("Dz", "z"))}

    def measure(factor: float) -> float:
        for axis in "xyz":
            getattr(driver.fields, "D" + axis)[...] = (
                originals["D" + axis] * driver.fields.grid.xp.float32(factor))
        return float(measured_margin(driver))

    while measure(high) < target and high < 1e9:
        high *= 4.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        value = measure(mid)
        if abs(value - target) <= 0.02 * target:
            break
        if value < target:
            low = mid
        else:
            high = mid
    else:
        mid = 0.5 * (low + high)
    final = measure(mid)
    if final >= 1.0 / 3.0:
        raise AssertionError(f"margin scaling overshot the pole bound: {final}")
    return float(mid)


def apply_margin_factor(driver, factor: float) -> None:
    xp = driver.fields.grid.xp
    for axis in "xyz":
        getattr(driver.fields, "D" + axis)[...] = (
            getattr(driver.fields, "D" + axis) * xp.float32(factor))


def state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name)
            for name in FIELD_STATE + AUX_STATE
            if getattr(driver.fields, name, None) is not None}


def compare(cp, left, right) -> Dict[str, Any]:
    a = np.ascontiguousarray(cp.asnumpy(left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(cp.asnumpy(right)).view(np.uint32).ravel()
    return {"bit_identical": bool(np.array_equal(a, b)),
            "differing_floats": int(np.count_nonzero(a != b)),
            "total_floats": int(a.size)}


# ---------------------------------------------------------------------------
# The narrowing assertion — the licensing measurement's precondition
# ---------------------------------------------------------------------------

def narrowing_reasons(fields, pml, drop_backend: bool = False) -> Dict[str, Any]:
    """Per certified sub-step predicate: the refusal reasons, split into the
    chi clause and everything else. The narrowing claim is that 'everything
    else' is EMPTY — the package-wide refusal is broader than the arithmetic."""
    from meep_gpu.triton_kernels import coverage  # noqa: PLC0415
    from meep_gpu.triton_kernels import nonlinear_update_e as nlmod  # noqa: PLC0415

    def split(reasons):
        reasons = list(reasons)
        if drop_backend:
            reasons = [r for r in reasons if "array module" not in r]
        chi = [r for r in reasons if "chi2/chi3" in r]
        other = [r for r in reasons if "chi2/chi3" not in r]
        return {"chi_clause": chi, "other": other}

    out = {
        "step_B": split(coverage.pml_curl_coverage(fields, pml, "step_B").reasons),
        "step_D": split(coverage.pml_curl_coverage(fields, pml, "step_D").reasons),
        "update_H": split(coverage.constitutive_coverage(fields, pml, "H").reasons),
    }
    nonlinear = nlmod.nonlinear_constitutive_coverage(fields, pml)
    reasons = list(nonlinear.reasons)
    if drop_backend:
        reasons = [r for r in reasons if "array module" not in r]
    out["update_E_nonlinear"] = {"covered": not reasons, "reasons": reasons}
    return out


def assert_narrowing(report: Dict[str, Any], case_name: str,
                     require_new_covered: bool) -> None:
    for sub in ("step_B", "step_D", "update_H"):
        if report[sub]["other"]:
            raise AssertionError(
                f"{case_name}: the certified {sub} predicate refuses this run "
                f"for MORE than the chi clause — the narrowing claim does not "
                f"hold here: {report[sub]['other']}")
        if not report[sub]["chi_clause"]:
            raise AssertionError(
                f"{case_name}: the certified {sub} predicate no longer refuses "
                f"the chi clause — coverage.py was narrowed underfoot; this "
                f"probe's install route must be rebuilt against plan_step")
    if require_new_covered and not report["update_E_nonlinear"]["covered"]:
        raise AssertionError(
            f"{case_name}: the nonlinear update_E predicate refused: "
            f"{report['update_E_nonlinear']['reasons']}")


# ---------------------------------------------------------------------------
# Plan installation — certified curls + update_H, the new update_E
# ---------------------------------------------------------------------------

def build_plans(driver) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import launch as launch_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import nonlinear_update_e as nlmod  # noqa: PLC0415

    fields, pml = driver.fields, driver.pml
    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, pml)
    codes = [1 if kind == "metallic" else 0 for kind in kinds]
    dtdx = grid.dt / grid.dx
    arrays = {name: getattr(fields, name)
              for name in FIELD_STATE + AUX_STATE
              if getattr(fields, name, None) is not None}
    for name in ("Ex", "Ey", "Ez"):
        arrays["inv_eps_" + name] = fields.inverse_epsilon_for(name)

    def curl_flat(suffix):
        return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
                for axis in "xyz" for stem in ("kms", "sinv")}

    def constitutive_flat(suffix):
        return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
                for axis in "xyz" for stem in ("kps", "kms")}

    plans = {
        # The CERTIFIED kernels, through their from_arrays routes: their
        # engine builders refuse the chi clause (asserted upstream), and this
        # composition is the measurement that licenses narrowing it.
        "step_B": launch_module.plan_from_arrays(
            "step_B", arrays, curl_flat("_h"), codes, float(dtdx)),
        "update_H": launch_module.plan_constitutive_from_arrays(
            "H", arrays, constitutive_flat("")),
        "step_D": launch_module.plan_from_arrays(
            "step_D", arrays, curl_flat(""), codes, float(dtdx)),
        # The NEW kernel, through its ENGINE route: its predicate must admit.
        "update_E": nlmod.plan_nonlinear_constitutive(fields, pml),
    }
    if plans["update_E"] is None:
        raise AssertionError(
            "the nonlinear update_E engine builder refused a run its "
            "predicate was asserted to admit: "
            + "; ".join(nlmod.nonlinear_constitutive_coverage(
                fields, pml).reasons))
    return plans


class PlanCounter:
    """Launch counting for EVERY composition case: all four installed plans
    are wrapped, the per-slot counts land in the artifact, and a slot whose
    count differs from the step budget fails the case — a case whose plans
    never ran measures nothing and must FAIL, not pass (the audit found the
    certification cases carried no per-case launch evidence at all)."""

    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.launches = 0

    def run(self, guard: Optional[bool] = None) -> None:
        self.launches += 1
        self.plan.run(guard=guard)


def install(driver_module, plans: Dict[str, Any], owner):
    """Serve the four sub-steps from the plans, for the owner fields only.
    Sources, ``zero_metal_*`` and the guards stay the driver's own. Production
    dispatch is untouched — this is a monkeypatch on the driver module."""
    names = ("step_B", "update_H", "step_D", "update_E")
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


# ---------------------------------------------------------------------------
# One composition case
# ---------------------------------------------------------------------------

def run_case(cp, case: Dict[str, Any],
             mutate_plan: Optional[Callable[[Dict[str, Any]], Any]] = None
             ) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    name = case["name"]
    steps = case["steps"]
    reference, route = build_case(case, cp, prefer_gpu=True)
    candidate, _ = build_case(case, cp, prefer_gpu=True)
    control, _ = build_case(case, cp, prefer_gpu=True, with_source=False)
    undo = lambda: None  # noqa: E731
    try:
        for driver in (reference, candidate, control):
            seed_primaries(driver, cp, case["seed_amplitude"])
        if case["margin_target"] is not None:
            factor = scale_factor_for_margin(reference, case["margin_target"])
            # reference's D is already scaled by the search; align the others.
            apply_margin_factor(candidate, factor)
            apply_margin_factor(control, factor)
        row: Dict[str, Any] = {
            "case": name, "construction": route,
            "shape": [int(v) for v in candidate.shape],
            "steps": steps,
            "step_budget_note": "bit-identity is claimed for exactly this "
                                "many steps and no further",
            "initial_margin": measured_margin(candidate),
            "per_step": [],
        }
        report = narrowing_reasons(candidate.fields, candidate.pml)
        assert_narrowing(report, name, require_new_covered=True)
        row["narrowing"] = {
            sub: {"chi_clause_count": len(report[sub]["chi_clause"]),
                  "other": report[sub]["other"]}
            for sub in ("step_B", "step_D", "update_H")}
        plans = build_plans(candidate)
        if mutate_plan is not None:
            plans["update_E"] = mutate_plan(plans)
        # EVERY case is launch-counted, not just the armed-mutation leg: the
        # per-slot counts are the evidence the Triton plans actually served
        # the sub-steps for THIS case's rows.
        counters = {slot: PlanCounter(plan) for slot, plan in plans.items()}
        row["plans"] = {slot: type(plan).__name__
                        for slot, plan in plans.items()}
        row["nonlinear_components"] = list(
            getattr(plans["update_E"], "plan", plans["update_E"]).nonlinear)
        if case.get("flux"):
            monitors = []
            for driver in (reference, candidate):
                monitor = driver.add_flux_monitor(
                    fcen=0.8, df=0.4, nfreq=5,
                    center=(0.0, 0.0, 0.2), size=(0.8, 0.8, 0.0),
                    decimation_factor=0)
                monitors.append(monitor)
            row["resolved_decimation"] = int(monitors[0].decimation_factor)
            row["meep_would_resolve"] = 1
            # NON-VACUITY: FluxMonitor.update accumulates only when
            # step % decimation_factor == 0 (dft.py), so a budget below the
            # resolved factor never touches the accumulators — measured on
            # the certifying run, where 16 steps against a factor of 21 left
            # the "bit-exact" compare running over two untouched zero
            # buffers. Same assertion as the off-diagonal tranche's.
            if steps < 2 * row["resolved_decimation"]:
                raise AssertionError(
                    f"{name}: budget {steps} accumulates fewer than two flux "
                    f"samples at the engine's resolved decimation factor "
                    f"{row['resolved_decimation']} (samples land on steps "
                    f"factor, 2*factor, ...) — the flux compare would be "
                    f"vacuous; raise the case's step budget")
            row["flux_samples_in_budget"] = steps // row["resolved_decimation"]
            row["decimation_note"] = (
                "RESOLVED 2026-08-12: _automatic_decimation_factor now "
                "mirrors MEEP's has_nonlinearities guard (dft.cpp:207-210 "
                "resolves 1 on a nonlinear run), so resolved_decimation "
                "should equal meep_would_resolve here. Historical record: "
                "before the fix this engine derived the factor from source "
                "bandwidth only and resolved 21 on this case (Gaussian "
                "fcen=0.8 df=0.4, chi2/chi3 installed) — the standing "
                "discrepancy this case carried for the wiring change. BOTH "
                "drivers here share the engine's resolved factor, so the "
                "byte comparison was unaffected either way")
        undo = install(driver_module, counters, candidate.fields)
        source_effect_seen = False
        started = time.time()
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            control.step()
            cp.cuda.runtime.deviceSynchronize()
            reference_state = state(reference)
            candidate_state = state(candidate)
            if set(reference_state) != set(candidate_state):
                raise AssertionError(
                    f"{name} step {step}: state inventory differs")
            parts = {key: compare(cp, candidate_state[key],
                                  reference_state[key])
                     for key in sorted(reference_state)}
            source_effect_seen = source_effect_seen or any(
                not compare(cp, reference_state[key],
                            getattr(control.fields, key))["bit_identical"]
                for key in PRIMARY_NAMES)
            point = {
                "step": step,
                "bit_identical": all(p["bit_identical"] for p in parts.values()),
                "differing_floats": sum(p["differing_floats"]
                                        for p in parts.values()),
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
            if mutate_plan is None and not point["bit_identical"]:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        # The launch evidence: each sub-step is called exactly once per
        # complete driver step (driver.py:3163-3182), so every slot's count
        # must equal the budget — 0 is the DISARMED monkeypatch, anything
        # else is a changed driver call structure; both invalidate the rows.
        row["plan_launches"] = {slot: counter.launches
                                for slot, counter in counters.items()}
        for slot, counter in counters.items():
            if counter.launches != steps:
                raise AssertionError(
                    f"{name}: plan slot {slot} launched {counter.launches} "
                    f"times over {steps} driver steps — the install seam "
                    f"stopped intercepting (DISARMED) or the driver's "
                    f"per-step call structure changed; bit-identical rows "
                    f"without launch evidence certify nothing")
        row["final_margin"] = measured_margin(candidate)
        if not source_effect_seen:
            raise AssertionError(
                f"{name}: the source never separated the reference from its "
                f"control — the case is vacuous")
        if case.get("flux"):
            spectra = [np.asarray(monitor.get_flux_spectrum())
                       for monitor in monitors]
            row["flux_spectra_identical"] = bool(
                spectra[0].tobytes() == spectra[1].tobytes())
            row["flux_spectrum"] = [float(v) for v in spectra[0]]
            # The vacuity tripwire the budget assertion predicts: a spectrum
            # of exact zeros means no sample ever accumulated, and equality
            # of two untouched buffers certifies nothing. Refused BEFORE the
            # identity is read as evidence.
            if not any(value != 0.0 for value in row["flux_spectrum"]):
                raise AssertionError(
                    f"{name}: the flux spectrum is identically zero — the "
                    f"monitors never accumulated a live sample and the "
                    f"bit-exact compare is vacuous")
            if not row["flux_spectra_identical"]:
                raise AssertionError(f"{name}: flux spectra differ between "
                                     f"the composed and array-path drivers")
        row["bit_identical"] = all(p["bit_identical"] for p in row["per_step"])
        row["source_effect_seen"] = source_effect_seen
        return row
    finally:
        undo()
        reference.close()
        candidate.close()
        control.close()
        cp.get_default_memory_pool().free_all_blocks()


def run_armed_mutation(cp) -> Dict[str, Any]:
    """Case 3 with the nonlinear plan's chi3 scalar arm doubled — MUST diverge
    within the budget, and the mutated plan MUST demonstrably have launched
    (run_case's uniform per-slot launch counting is the DISARMED tripwire:
    a mutated plan that never launched raises there before this leg reports)."""
    case = dict(CASES[2])

    def mutate(plans: Dict[str, Any]) -> Any:
        plan = plans["update_E"]
        doubled = tuple(2.0 * value for value in plan.chi3_scalars)
        if not any(doubled):
            raise AssertionError("armed mutation: no scalar chi3 arm to "
                                 "double — the leg is disarmed")
        plan.chi3_scalars = doubled
        return plan

    row = run_case(cp, case, mutate_plan=mutate)
    launches = row["plan_launches"]["update_E"]
    first_divergence = next((p["step"] for p in row["per_step"]
                             if not p["bit_identical"]), None)
    if first_divergence is None:
        raise AssertionError(
            f"armed mutation NOT CAUGHT in {case['steps']} steps with "
            f"{launches} mutated launches — the per-step compare "
            f"cannot see a doubled chi3 and the harness is not trustworthy")
    return {"case": case["name"],
            "mutation": "update_E plan chi3 scalar arm doubled",
            "launches": launches,
            "caught_at_step": first_divergence, "caught": True}


# ---------------------------------------------------------------------------
# Laptop self-check — case builders + array path, no CUDA, no Triton
# ---------------------------------------------------------------------------

def self_check(out_path: str) -> int:
    payload: Dict[str, Any] = {
        "mode": "self-check (laptop, NumPy array path only)",
        "triton_legs": "SKIPPED: no CUDA/triton on this host — the plan "
                       "construction, kernel launches, and per-step "
                       "bit-identity claims are NOT exercised here",
        "cases": [],
    }
    save(payload, out_path)
    failures = 0
    for case in CASES:
        started = time.time()
        row: Dict[str, Any] = {"case": case["name"], "steps_run": 0,
                               "declared_budget": case["steps"]}
        try:
            driver, route = build_case(case, np, prefer_gpu=False)
            row["construction"] = route
            seed_primaries(driver, np, case["seed_amplitude"])
            if case["margin_target"] is not None:
                factor = scale_factor_for_margin(driver,
                                                 case["margin_target"])
                row["margin_factor"] = factor
            row["initial_margin"] = measured_margin(driver)
            report = narrowing_reasons(driver.fields, driver.pml,
                                       drop_backend=True)
            assert_narrowing(report, case["name"], require_new_covered=True)
            row["narrowing_holds"] = True
            if case.get("flux"):
                monitor = driver.add_flux_monitor(
                    fcen=0.8, df=0.4, nfreq=5, center=(0.0, 0.0, 0.2),
                    size=(0.8, 0.8, 0.0), decimation_factor=0)
                row["resolved_decimation"] = int(monitor.decimation_factor)
                row["meep_would_resolve"] = 1
                # The resolution is backend-independent host arithmetic, so
                # THIS laptop check is where a budget-vs-factor drift is
                # caught before a device run ships a vacuous flux compare.
                if case["steps"] < 2 * row["resolved_decimation"]:
                    raise AssertionError(
                        f"declared budget {case['steps']} accumulates fewer "
                        f"than two flux samples at the engine's resolved "
                        f"decimation factor {row['resolved_decimation']} — "
                        f"the device flux compare would be vacuous")
                row["flux_samples_in_budget"] = (
                    case["steps"] // row["resolved_decimation"])
            for _ in range(case["steps"]):
                driver.step()
                row["steps_run"] += 1
            if case.get("flux"):
                spectrum = [float(value) for value in
                            np.asarray(monitor.get_flux_spectrum())]
                row["flux_spectrum"] = spectrum
                if not any(value != 0.0 for value in spectrum):
                    raise AssertionError(
                        "the flux spectrum is identically zero after the full "
                        "budget — the monitor never accumulated a live sample")
            row["final_margin"] = measured_margin(driver)
            if row["final_margin"] is not None and row["final_margin"] >= 1 / 3:
                raise AssertionError(
                    f"final margin {row['final_margin']:.3f} crossed the pole "
                    f"guard's bound — retune the case before the device run")
            if (case["margin_target"] is not None
                    and row["initial_margin"] < 0.5 * case["margin_target"]):
                raise AssertionError(
                    f"loud case landed at margin {row['initial_margin']:.3f}, "
                    f"far under its {case['margin_target']} target — the "
                    f"large-u leg would be vacuous")
            total = sum(float(np.abs(np.asarray(
                getattr(driver.fields, n))).sum()) for n in PRIMARY_NAMES)
            if not np.isfinite(total):
                raise AssertionError("array path produced non-finite state")
            row.update({"shape": [int(v) for v in driver.shape],
                        "primary_l1_after_budget": total, "ok": True})
            log(f"[self-check] {case['name']}: shape={tuple(driver.shape)} "
                f"margin {row['initial_margin']}->{row['final_margin']} "
                f"{row['steps_run']} array-path steps ok "
                f"({time.time() - started:.1f} s)")
            driver.close()
        except Exception as exc:  # noqa: BLE001 - each failure is recorded
            row["error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
            log(f"[self-check] {case['name']}: FAILED {row['error'][:160]}")
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
    complex_gate.install_ftz_strip()
    backends.guard_kernel_compilation(cp)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    gate.write_provenance(out_dir)

    payload: Dict[str, Any] = {
        "environment": environment(cp),
        "subnormal_policy": complex_gate.policy_stamp("cupy"),
        "step_budget": {case["name"]: case["steps"] for case in CASES},
        "step_budget_note": "bit-identity is claimed per case for exactly the "
                            "stated budget and no further",
        "narrowing_note": "each case asserts the certified sub-step "
                          "predicates refuse this run for the chi clause and "
                          "NOTHING else; the byte-identical composition is "
                          "the measurement that licenses narrowing "
                          "coverage.py's every-sub-step refusal",
        "cases": [],
    }
    save(payload, args.out)
    for case in CASES:
        log(f"starting {case['name']}: steps={case['steps']}")
        payload["cases"].append(run_case(cp, case))
        save(payload, args.out)
    payload["armed_mutation"] = run_armed_mutation(cp)
    save(payload, args.out)

    log("refusal enumeration (the sub-step gate's leg, into this artifact)")
    gate.run_refusals(payload, args.out)
    if not payload["refusals"]["pass"]:
        payload["summary"] = {"status": "FAILED",
                              "reason": "refusal enumeration failed"}
        save(payload, args.out)
        return 1

    payload["subnormal_policy"] = complex_gate.policy_stamp("cupy")
    residual = complex_gate.ftz_strip_license_reasons()
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
        "plan_launches_per_case": {row["case"]: row.get("plan_launches")
                                   for row in ran},
        "loud_final_margin": next(
            (row.get("final_margin") for row in ran
             if row["case"] == "marquee_3rd_harm_loud"), None),
        "resolved_decimation": next(
            (row.get("resolved_decimation") for row in ran
             if row["case"] == "dft_flux_decimation"), None),
        "armed_mutation_caught_at_step":
            payload["armed_mutation"]["caught_at_step"],
    }
    save(payload, args.out)
    log(f"NONLINEAR COMPOSITION GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
