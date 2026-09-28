"""Complete-driver CUDA gate for the tensor / off-diagonal epsilon Triton composition.

The sub-step gate (``gate_triton_offdiag.py``) certifies the kernel against an
in-file transcription; this complementary probe asks the composition question:
with ``step_B``/``step_D`` served by the CERTIFIED curl kernel, ``update_H`` by
the CERTIFIED constitutive kernel, and ``update_E`` by the NEW off-diagonal
kernel — installed by monkeypatch, production dispatch untouched,
``plan_fast_path`` still returning None — does the complete driver state
reproduce the array-path driver EXACTLY after every step, for a STATED number
of steps, sources, PML and metal zeroing included?

THIS PROBE IS THE PARTIAL-UNLOCK MEASUREMENT, and it differs from the
nonlinear probe's narrowing in exactly the way the coverage split differs. A
chi2/chi3 run is refused at EVERY sub-step by ``coverage._grid_reasons``; an
off-diagonal run is refused at ONE — ``constitutive_coverage(side="E")``'s
offdiag clause (coverage.py:367-370) — while the curl predicates and the H
side ADMIT it (coverage.py:26-31, :343-346; the laptop file pins this). So
where the nonlinear probe had to route the certified kernels through their
``from_arrays`` builders, this probe builds the three certified slots through
their ENGINE builders (``plan_pml_curl``, ``plan_constitutive("H")``) — which
must return real plans, today, on the engine's own offdiag objects — and only
``update_E`` comes from the new family's engine builder. Before each case the
probe asserts the exact split: the three certified predicates hold NO refusal,
the certified E predicate refuses FOR THE OFFDIAG CLAUSE AND NOTHING ELSE (the
disjointness seam), and the new predicate admits. The byte-identical
composition over whole steps is then the evidence the wiring change needs to
give ``plan_step`` its offdiag update_E branch — this kernel is the LAST
uncovered sub-step of such a run, not the first.

The cases span the demand and nothing else (tensors installed through
``FdtdDriver.set_epsilon_components(chi1inv_offdiagonal=...)`` — the same
route ``from_meep``'s epsilon_offdiag ingest uses, from_meep.py:1125-1126):

1. ``marquee_tensor_pml_loud`` — test_tensor_epsilon._CASE's 'pml' variant
   geometry (cell 1.1x1.3x1.5 at resolution 10 -> 11x13x15, 4-cell PML all
   axes, the case's own three CW point sources) with the uniform full oracle
   tensor, at LOUD seeded amplitude, 16 complete steps. REAL f32 storage —
   the oracle itself runs complex; Phase A's arm is the real-storage one, so
   this is the oracle's GEOMETRY, not its storage, and the artifact says so.
2. ``metallic_tensor``  — the same geometry with all-metallic walls: the
   wall-coupling mask live on every component's transverse axes through whole
   steps, 12 steps.
3. ``mixed_varying_courant035`` — mixed boundaries (periodic, metallic,
   periodic) at Courant 0.35 (the mandate's home in this probe: dtdx enters
   the CURL kernels here), SPATIALLY VARYING tensor volumes (the
   between-shifts association's byte-visible class), Ez Gaussian source,
   10 steps.
4. ``partial_row_anisotropic`` — a single-entry row (Ez<-Ex) with three
   DISTINCT diagonal inverse-epsilon volumes: two components take the new
   kernel's compiled plain arm while Ez takes the coupled arm, over whole
   steps, 12 steps.
5. ``reduced_2d`` — the 2-D reduced tensor case (160,160,1): the invariant
   axis's partner pair is g+g (MEEP's stride(d)=0 double-read), composed over
   whole steps, 8 steps. A real Grid collapses z only, so this is the
   driver-expressible form; the (1,160,160) leading-axis form is the sub-step
   gate's from_arrays sweep case.
6. ``dft_flux_decimation``  — case 3's geometry, rows and boundaries, with a
   flux monitor at the engine's automatically resolved decimation (21 on
   this configuration, recorded) and its own LONGER budget: 48 steps, so at
   least two decimated samples accumulate. Both are ASSERTED, not assumed —
   budget >= 2x the resolved factor at monitor-add time, spectrum nonzero at
   the end — because the original 16-step budget under the resolved factor
   of 21 was measured to accumulate NOTHING, leaving the 'bit-exact spectra'
   compare running over untouched zero buffers. Flux spectra compared
   bit-exactly at the end alongside the per-step field compare. The run is
   LINEAR — MEEP's has_nonlinearities decimation guard is not in play, unlike
   the nonlinear probe's standing-discrepancy note.

Every composition case is LAUNCH-COUNTED: all four installed plans are wrapped
in :class:`PlanCounter`, the artifact records per-slot launches, and a slot
whose count differs from the case's step budget fails the case loudly —
bit-identical rows without launch evidence certify nothing. Every case runs a
SOURCE-FREE control driver beside the pair and requires the source to separate
the reference from it (the sub-step gate's ``source_control_note`` delegates
the source-non-vacuity control HERE, because the sub-step itself has no
sources).

An ARMED mutation leg re-runs case 3 with the candidate's installed Ez<-Ex
coefficient volume doubled IN PLACE after the plans are built — the plan's
pointer serves the doubled bytes to the kernel while the array-path reference
keeps the true material; only the Triton kernel reads that volume on the
candidate, so the leg MUST be caught at STEP 1 (the coupling is live at the
first ``update_E``), launch-counted, with DISARMED and NOT-CAUGHT failures
loud. A refusal-enumeration leg reuses the sub-step gate's ``run_refusals`` so
the composition artifact carries the same 0-admissions record.

Requirements carried from prior defects: per-complete-step uint32 compare of
EVERY allocated stored volume; the step budget STATED in the artifact;
unbuffered per-case progress; atomic JSON rewrite per case; own provenance
record. SUBNORMAL POLICY: runs UNDER the stripped IEEE-keep policy via the
complex gate's ``install_ftz_strip`` (strip FIRST, compilation guard second),
and refuses to certify when the strip cannot be confirmed exercised.
Correctness only: no throughput or timing claims.

Usage (the GPU host)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_offdiag_composition.py \\
        --out results/triton_offdiag_<date>/composition.json

Laptop (no CUDA/triton): ``--self-check`` builds every case's drivers on
NumPy, steps the ARRAY PATH through each case's full budget, validates the
partial-unlock reason structure, proves the installed tensor separates each
case from a rows-dropped twin (the feature is live in whole steps), and says
in so many words that the Triton legs were skipped.
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
import gate_triton_offdiag as gate  # noqa: E402

from meep_gpu.triton_kernels import offdiag_update_e as odmod  # noqa: E402

SEED = gate.SEED
save = gate.save
log = gate.log
EPS_TENSOR = gate.EPS_TENSOR
INV_TENSOR = gate.INV_TENSOR

E_NAMES = ("Ex", "Ey", "Ez")
FIELD_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
               "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
AUX_STATE = ("fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
             "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")

#: test_tensor_epsilon._CASE's geometry and sources (the 'pml' oracle variant's
#: cell/resolution/PML; storage here is Phase A's real f32, recorded per case).
MARQUEE_CELL = (1.1, 1.3, 1.5)
MARQUEE_RESOLUTION = 10.0
MARQUEE_PML_CELLS = 4  # 0.4 length units at resolution 10, matching mp.PML(0.4)
MARQUEE_FREQUENCY = 1.0
MARQUEE_SOURCES = (
    ("Ex", (0.15, -0.25, -0.35), 0.7),
    ("Ey", (-0.15, 0.25, -0.15), 0.5),
    ("Ez", (0.05, 0.05, -0.25), 1.0),
)

#: name -> builder kind, budget, seeded amplitude, row install, extras.
CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "marquee_tensor_pml_loud", "kind": "marquee", "steps": 16,
     "seed_amplitude": 0.5, "rows": "uniform_full"},
    {"name": "metallic_tensor", "kind": "marquee", "steps": 12,
     "seed_amplitude": 0.25, "rows": "uniform_full", "boundaries": "metallic"},
    {"name": "mixed_varying_courant035", "kind": "synthetic", "steps": 10,
     "seed_amplitude": 0.25, "rows": "varying_full"},
    {"name": "partial_row_anisotropic", "kind": "partial", "steps": 12,
     "seed_amplitude": 0.25, "rows": "single_ez_ex"},
    {"name": "reduced_2d", "kind": "reduced", "steps": 8,
     "seed_amplitude": 0.25, "rows": "uniform_full"},
    # 48 steps, NOT case 3's 10: the engine resolves decimation 21 for this
    # source/monitor bandwidth pair, and the budget must cover at least two
    # accumulations (asserted in run_case) or the flux compare is vacuous.
    {"name": "dft_flux_decimation", "kind": "synthetic", "steps": 48,
     "seed_amplitude": 0.25, "rows": "varying_full", "flux": True},
)


# ---------------------------------------------------------------------------
# Case construction — real FdtdDriver objects, tensors through the engine API
# ---------------------------------------------------------------------------

def _uniform(shape, value) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def _tensor_install(driver, rows_kind: str) -> None:
    """Install the oracle tensor through ``set_epsilon_components`` — the same
    route from_meep's epsilon_offdiag ingest takes (from_meep.py:1125-1126).

    Diagonal 'epsilon' entries are EFFECTIVE permittivities (reciprocals of the
    inverse tensor's diagonal); the off-diagonal rows are the raw chi1inv
    entries, negative where the oracle tensor's inverse is."""
    shape = tuple(int(n) for n in driver.shape)
    volumes = {name: _uniform(shape, 1.0 / INV_TENSOR[i, i])
               for i, name in enumerate(E_NAMES)}
    if rows_kind == "uniform_full":
        rows = {row: {partner: _uniform(shape, INV_TENSOR[i, j])
                      for j, partner in enumerate(E_NAMES) if j != i}
                for i, row in enumerate(E_NAMES)}
    elif rows_kind == "varying_full":
        rng = np.random.default_rng(SEED + 101)
        rows = {row: {partner: (_uniform(shape, INV_TENSOR[i, j])
                                * rng.uniform(0.5, 1.5, size=shape)
                                .astype(np.float32)).astype(np.float32)
                      for j, partner in enumerate(E_NAMES) if j != i}
                for i, row in enumerate(E_NAMES)}
    elif rows_kind == "single_ez_ex":
        rows = {"Ez": {"Ex": _uniform(shape, INV_TENSOR[2, 0])}}
    else:
        raise ValueError(rows_kind)
    driver.set_epsilon_components(volumes, chi1inv_offdiagonal=rows)


def build_case(case: Dict[str, Any], prefer_gpu: bool,
               with_source: bool = True, with_rows: bool = True):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    kind = case["kind"]
    if kind == "marquee":
        driver = FdtdDriver(
            cell_size=MARQUEE_CELL, resolution=MARQUEE_RESOLUTION,
            dimensions=3, boundaries=case.get("boundaries"),
            force_complex_fields=False, prefer_gpu=prefer_gpu, gpu_id=0)
        pml_spec: Any = MARQUEE_PML_CELLS
        sources = [{"component": component, "frequency": MARQUEE_FREQUENCY,
                    "center": center, "size": (0.0, 0.0, 0.0),
                    "amplitude": amplitude}
                   for component, center, amplitude in MARQUEE_SOURCES]
    elif kind in ("synthetic",):
        driver = FdtdDriver(
            cell_size=(1.6, 1.6, 0.8), resolution=15.0, dimensions=3,
            boundaries=("periodic", "metallic", "periodic"), courant=0.35,
            force_complex_fields=False, prefer_gpu=prefer_gpu, gpu_id=0)
        pml_spec = {"x": 4, "z": 3}
        sources = [{"component": "Ez", "source_type": "gaussian",
                    "frequency": 0.8, "fwidth": 0.4,
                    "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                    "amplitude": 1.0}]
    elif kind == "partial":
        driver = FdtdDriver(
            cell_size=(0.8, 0.8, 0.8), resolution=15.0, dimensions=3,
            boundaries="metallic", courant=0.35,
            force_complex_fields=False, prefer_gpu=prefer_gpu, gpu_id=0)
        pml_spec = {"x": 3}
        sources = [{"component": "Ez", "source_type": "gaussian",
                    "frequency": 0.8, "fwidth": 0.4,
                    "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                    "amplitude": 1.0}]
    elif kind == "reduced":
        driver = FdtdDriver(
            cell_size=(6.4, 6.4, 0.0), resolution=25.0, dimensions=2,
            force_complex_fields=False, prefer_gpu=prefer_gpu, gpu_id=0)
        pml_spec = {"x": 6, "y": 6}
        sources = [{"component": "Ez", "source_type": "gaussian",
                    "frequency": 0.8, "fwidth": 0.4,
                    "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                    "amplitude": 1.0}]
    else:
        raise ValueError(kind)

    if with_rows:
        _tensor_install(driver, case["rows"])
    else:
        # The rows-dropped twin: the same diagonal install, no coupling.
        shape = tuple(int(n) for n in driver.shape)
        driver.set_epsilon_components(
            {name: _uniform(shape, 1.0 / INV_TENSOR[i, i])
             for i, name in enumerate(E_NAMES)})
    driver.setup_pml(pml_spec)
    if with_source:
        for source in sources:
            driver.add_source(source)
    return driver


def seed_primaries(driver, xp, amplitude: float, seed: int = 20260812) -> None:
    rng = np.random.default_rng(seed)
    for name in PRIMARY_NAMES:
        host = np.ascontiguousarray(
            (amplitude * rng.uniform(-1.0, 1.0, size=driver.shape))
            .astype(np.float32))
        driver.set_field(name, xp.asarray(host))


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
# The partial-unlock assertion — the licensing measurement's precondition
# ---------------------------------------------------------------------------

def unlock_reasons(fields, pml, drop_backend: bool = False) -> Dict[str, Any]:
    """Per sub-step: the certified predicates' refusal reasons for this run.

    The partial-unlock claim is that the three non-E sub-steps hold NO refusal
    at all (the curls and H side admit offdiag runs, coverage.py:26-31,
    :343-346), and the certified E side refuses for the offdiag clause and
    NOTHING else — the deliberate per-sub-step split this kernel completes."""
    from meep_gpu.triton_kernels import coverage  # noqa: PLC0415

    def strip(reasons):
        reasons = list(reasons)
        if drop_backend:
            reasons = [r for r in reasons if "array module" not in r]
        return reasons

    out: Dict[str, Any] = {
        "step_B": strip(coverage.pml_curl_coverage(fields, pml, "step_B").reasons),
        "step_D": strip(coverage.pml_curl_coverage(fields, pml, "step_D").reasons),
        "update_H": strip(coverage.constitutive_coverage(fields, pml, "H").reasons),
    }
    certified_e = strip(coverage.constitutive_coverage(fields, pml, "E").reasons)
    out["update_E_certified"] = {
        "offdiag_clause": [r for r in certified_e if "off-diagonal" in r],
        "other": [r for r in certified_e if "off-diagonal" not in r],
    }
    new = strip(odmod.offdiag_constitutive_coverage(fields, pml).reasons)
    out["update_E_offdiag"] = {"covered": not new, "reasons": new}
    return out


def assert_partial_unlock(report: Dict[str, Any], case_name: str,
                          require_new_covered: bool) -> None:
    for sub in ("step_B", "step_D", "update_H"):
        if report[sub]:
            raise AssertionError(
                f"{case_name}: the certified {sub} predicate refuses this "
                f"offdiag run — the partial-unlock premise (curls and H admit, "
                f"coverage.py:343-346) does not hold here: {report[sub]}")
    if not report["update_E_certified"]["offdiag_clause"]:
        raise AssertionError(
            f"{case_name}: the certified E predicate no longer refuses the "
            f"offdiag clause — the disjointness seam (coverage.py:367-370) "
            f"was narrowed underfoot; this probe's install route must be "
            f"rebuilt against plan_step")
    if report["update_E_certified"]["other"]:
        raise AssertionError(
            f"{case_name}: the certified E predicate refuses this run for "
            f"MORE than the offdiag clause — the unlock claim does not hold "
            f"here: {report['update_E_certified']['other']}")
    if require_new_covered and not report["update_E_offdiag"]["covered"]:
        raise AssertionError(
            f"{case_name}: the offdiag update_E predicate refused: "
            f"{report['update_E_offdiag']['reasons']}")


# ---------------------------------------------------------------------------
# Plan installation — certified ENGINE builders + the new update_E
# ---------------------------------------------------------------------------

def build_plans(driver) -> Dict[str, Any]:
    """All four sub-step plans from the engine's own objects.

    UNLIKE the nonlinear probe, the three certified slots come from their
    ENGINE builders, not from_arrays: their predicates admit the offdiag run,
    so a builder that returns None here is a certified-family regression, not
    an expected refusal — and the fact that they build is itself half the
    partial-unlock evidence."""
    from meep_gpu.triton_kernels import launch as launch_module  # noqa: PLC0415

    fields, pml = driver.fields, driver.pml
    plans = {
        "step_B": launch_module.plan_pml_curl(fields, pml, "step_B"),
        "update_H": launch_module.plan_constitutive(fields, pml, "H"),
        "step_D": launch_module.plan_pml_curl(fields, pml, "step_D"),
        "update_E": odmod.plan_offdiagonal_constitutive(fields, pml),
    }
    for slot, plan in plans.items():
        if plan is None:
            reasons: Any = "(see the slot's own predicate)"
            if slot == "update_E":
                reasons = "; ".join(odmod.offdiag_constitutive_coverage(
                    fields, pml).reasons)
            raise AssertionError(
                f"the {slot} engine builder refused a run its predicate was "
                f"asserted to admit: {reasons}")
    return plans


class PlanCounter:
    """Launch counting for EVERY composition case: all four installed plans
    are wrapped, the per-slot counts land in the artifact, and a slot whose
    count differs from the step budget fails the case — a case whose plans
    never ran measures nothing and must FAIL, not pass."""

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
             mutate: Optional[Callable[[Any, Dict[str, Any]], str]] = None
             ) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    name = case["name"]
    steps = case["steps"]
    reference = build_case(case, prefer_gpu=True)
    candidate = build_case(case, prefer_gpu=True)
    control = build_case(case, prefer_gpu=True, with_source=False)
    undo = lambda: None  # noqa: E731
    try:
        for driver in (reference, candidate, control):
            seed_primaries(driver, cp, case["seed_amplitude"])
        row: Dict[str, Any] = {
            "case": name,
            "shape": [int(v) for v in candidate.shape],
            "steps": steps,
            "step_budget_note": "bit-identity is claimed for exactly this "
                                "many steps and no further",
            "storage_note": "real f32 storage (Phase A's arm); the marquee "
                            "geometry is test_tensor_epsilon._CASE's, whose "
                            "CPU-MEEP oracle itself runs complex",
            "per_step": [],
        }
        report = unlock_reasons(candidate.fields, candidate.pml)
        assert_partial_unlock(report, name, require_new_covered=True)
        row["partial_unlock"] = {
            "certified_non_e_slots_admit": True,
            "certified_e_offdiag_clause_count": len(
                report["update_E_certified"]["offdiag_clause"]),
            "certified_e_other": report["update_E_certified"]["other"],
        }
        plans = build_plans(candidate)
        update_e = plans["update_E"]
        row["row_mask"] = list(update_e.row_mask)
        row["boundary_codes"] = list(update_e.boundary_codes)
        row["wall_axes"] = list(update_e.wall_axes)
        if mutate is not None:
            row["mutation"] = mutate(candidate, plans)
        # EVERY case is launch-counted, not just the armed-mutation leg: the
        # per-slot counts are the evidence the Triton plans actually served
        # the sub-steps for THIS case's rows.
        counters = {slot: PlanCounter(plan) for slot, plan in plans.items()}
        row["plans"] = {slot: type(plan).__name__
                        for slot, plan in plans.items()}
        if case.get("flux"):
            monitors = []
            for driver in (reference, candidate):
                monitor = driver.add_flux_monitor(
                    fcen=0.8, df=0.4, nfreq=5,
                    center=(0.0, 0.0, 0.1), size=(0.8, 0.8, 0.0),
                    decimation_factor=0)
                monitors.append(monitor)
            row["resolved_decimation"] = int(monitors[0].decimation_factor)
            # NON-VACUITY: FluxMonitor.update accumulates only when
            # step % decimation_factor == 0 (dft.py), so a budget below the
            # resolved factor never touches the accumulators — measured at
            # the original 16-step budget against a resolved factor of 21,
            # where the 'bit-exact spectra' were two untouched zero buffers.
            if steps < 2 * row["resolved_decimation"]:
                raise AssertionError(
                    f"{name}: budget {steps} accumulates fewer than two flux "
                    f"samples at the engine's resolved decimation factor "
                    f"{row['resolved_decimation']} (samples land on steps "
                    f"factor, 2*factor, ...) — the flux compare would be "
                    f"vacuous; raise the case's step budget")
            row["flux_samples_in_budget"] = steps // row["resolved_decimation"]
            row["decimation_note"] = (
                "linear run: MEEP's has_nonlinearities decimation guard is "
                "not in play; both drivers share the engine's resolved "
                "factor, recorded for the artifact")
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
            if mutate is None and not point["bit_identical"]:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        # The launch evidence: each sub-step is called exactly once per
        # complete driver step (driver.py step() order), so every slot's count
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
            # of two untouched buffers certifies nothing.
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
    """Case 3 with the candidate's installed Ez<-Ex coefficient volume doubled
    IN PLACE after the plans are built. The update_E plan's pointer serves the
    doubled bytes to the kernel; the array-path reference keeps the true
    material, and NOTHING ELSE on the candidate reads that volume (its own
    update_E is the monkeypatched slot) — so the defect is the kernel's input
    alone and MUST be caught at STEP 1, where the live coupling first lands in
    E. run_case's uniform per-slot launch counting is the DISARMED tripwire."""
    case = dict(CASES[2])

    def mutate(candidate, plans) -> str:
        rows = candidate.fields.chi1inv_offdiagonal_for("Ez")
        volume = rows.get("Ex")
        if volume is None or not bool(volume.any()):
            raise AssertionError("armed mutation: no live Ez<-Ex coefficient "
                                 "volume to double — the leg is disarmed")
        volume *= 2.0  # in place: the built plan's pointer serves these bytes
        return "candidate Ez<-Ex chi1inv volume doubled in place post-build"

    row = run_case(cp, case, mutate=mutate)
    launches = row["plan_launches"]["update_E"]
    first_divergence = next((p["step"] for p in row["per_step"]
                             if not p["bit_identical"]), None)
    if first_divergence is None:
        raise AssertionError(
            f"armed mutation NOT CAUGHT in {case['steps']} steps with "
            f"{launches} mutated launches — the per-step compare cannot see "
            f"a doubled coefficient and the harness is not trustworthy")
    if first_divergence != 1:
        raise AssertionError(
            f"armed mutation caught at step {first_divergence}, not step 1 — "
            f"a doubled live coefficient must be byte-visible at the first "
            f"update_E; a later catch means the coupling was dark at step 1 "
            f"and the case seeding is not what this leg assumes")
    return {"case": case["name"],
            "mutation": row["mutation"],
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
            driver = build_case(case, prefer_gpu=False)
            seed_primaries(driver, np, case["seed_amplitude"])
            report = unlock_reasons(driver.fields, driver.pml,
                                    drop_backend=True)
            assert_partial_unlock(report, case["name"],
                                  require_new_covered=True)
            row["partial_unlock_holds"] = True
            # The feature-liveness control: the same construction with the
            # rows dropped must separate BYTES within two steps, or the
            # tensor never entered the run and every later claim is vacuous.
            twin = build_case(case, prefer_gpu=False, with_rows=False)
            seed_primaries(twin, np, case["seed_amplitude"])
            for _ in range(2):
                driver.step()
                twin.step()
                row["steps_run"] += 1
            separated = any(
                np.asarray(getattr(driver.fields, component)).tobytes()
                != np.asarray(getattr(twin.fields, component)).tobytes()
                for component in E_NAMES)
            twin.close()
            if not separated:
                raise AssertionError(
                    "the installed tensor did not separate the run from its "
                    "rows-dropped twin in 2 steps — the coupling is dark in "
                    "whole steps and the composition would certify nothing")
            row["tensor_separates_from_diagonal_twin"] = True
            if case.get("flux"):
                monitor = driver.add_flux_monitor(
                    fcen=0.8, df=0.4, nfreq=5, center=(0.0, 0.0, 0.1),
                    size=(0.8, 0.8, 0.0), decimation_factor=0)
                row["resolved_decimation"] = int(monitor.decimation_factor)
                # The resolution is backend-independent host arithmetic, so
                # THIS laptop check is where a budget-vs-factor drift is
                # caught before a device run ships a vacuous flux compare.
                if case["steps"] < 2 * row["resolved_decimation"]:
                    raise AssertionError(
                        f"declared budget {case['steps']} accumulates fewer "
                        f"than two flux samples at the engine's resolved "
                        f"decimation factor {row['resolved_decimation']} — "
                        f"the device flux compare would be vacuous")
            while row["steps_run"] < case["steps"]:
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
            total = sum(float(np.abs(np.asarray(
                getattr(driver.fields, n))).sum()) for n in PRIMARY_NAMES)
            if not np.isfinite(total):
                raise AssertionError("array path produced non-finite state")
            row.update({"shape": [int(v) for v in driver.shape],
                        "primary_l1_after_budget": total, "ok": True})
            log(f"[self-check] {case['name']}: shape={tuple(driver.shape)} "
                f"{row['steps_run']} array-path steps ok, tensor separates "
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
        "partial_unlock_note": "each case asserts the certified curl and H "
                               "predicates ADMIT this run, the certified E "
                               "predicate refuses it for the offdiag clause "
                               "and nothing else (coverage.py:367-370), and "
                               "the new predicate admits; the byte-identical "
                               "composition is the measurement that licenses "
                               "plan_step's offdiag update_E branch",
        "courant_note": "cases 3, 4 and 6 of 6 run at Courant 0.35 "
                        "(non-power-of-two: the synthetic and partial "
                        "builders both cut it); dtdx enters the CURL kernels "
                        "in this probe",
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
    # Counted from the RECORDED per-step verdicts against the DECLARED case
    # table, so a deficit is displayable — the earlier f"{n}/{n}" form was
    # self-referential and could never show one. (Any failing case raises
    # before this summary, so a shortfall here means a harness defect.)
    identical_cases = sum(1 for row in ran if row.get("bit_identical"))
    identical_steps = sum(1 for row in ran for point in row["per_step"]
                          if point["bit_identical"])
    declared_steps = sum(case["steps"] for case in CASES)
    payload["summary"] = {
        "status": "passed",
        "certified_under_subnormal_policy":
            payload["subnormal_policy"].get("policy"),
        "cases_exact": f"{identical_cases}/{len(CASES)}",
        "complete_steps_exact": f"{identical_steps}/{declared_steps}",
        "plan_launches_per_case": {row["case"]: row.get("plan_launches")
                                   for row in ran},
        "resolved_decimation": next(
            (row.get("resolved_decimation") for row in ran
             if row["case"] == "dft_flux_decimation"), None),
        "armed_mutation_caught_at_step":
            payload["armed_mutation"]["caught_at_step"],
    }
    save(payload, args.out)
    log(f"OFFDIAG COMPOSITION GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
