"""Bit-identity for the REAL-STORAGE CONSTITUTIVE pair on the lifted corpus cases.

The sibling ``validate_pml_kernel_on_lifted_cases.py`` asks this question of the
curl pair. This asks it of ``update_H_pml_real`` and
``update_E_pml_real``: on the cases the benchmark actually times —
``2d_pml``, ``3d_pml``, ``2d_pml_geom``, lifted through ``from_meep`` exactly as
a user's run is — do the two paths agree byte for byte over consecutive whole
steps, on the DRIVER'S OWN arrays, with the DRIVER'S OWN PML tables, Courant
number and resolved boundaries?

=============================================================================
WHY THE WHOLE STEP RUNS AND NOT THE SUB-STEP ALONE
=============================================================================

``_apply_constitutive_pml`` writes ``fw[i] = src`` UNCONDITIONALLY. Hold B and D
fixed and ``f_w`` is a FIXED POINT from launch one: every later launch stores the
same value it stored before, ``prev`` equals ``fw``, and the history term stops
carrying history. A leg that ran ``update_H``/``update_E`` back to back on frozen
sources would therefore compare a recurrence that had stopped recurring — the
same vacuity the synthetic multi-step leg answers by scaling the source by 0.97
between launches (``probe_fused_kernel_bit_identity.one_constitutive_multi_step``).

On the corpus the faithful perturbation is the simulation itself. Both legs run
``step_B`` and ``step_D`` FROM THE ARRAY PATH, in MEEP's order
(``driver.FdtdDriver.step``: step_B, update_H, step_D, update_E), so B and D
evolve as a real run evolves them and ``f_w`` genuinely advances. Only the two
CONSTITUTIVE sub-steps are swapped between the legs, which is what makes a
divergence attributable to them — and because the curl reads H and E, a wrong
constitutive bit propagates into the very next sub-step instead of sitting in a
component nothing consumes.

THE CURL KERNELS ARE DELIBERATELY NOT DISPATCHED HERE. They are certified
separately and their own lifted leg already covers them; running both pairs at
once would make a divergence a two-suspect problem.

=============================================================================
WHAT IS COMPARED, AND THE VACUITY FLOOR
=============================================================================

Every CuPy array the driver holds (``vars(fields)``), byte for byte, AFTER EVERY
STEP rather than only at the end — so a divergence reports the first step and the
first array rather than a final tally. Liveness is recorded per case and gates
the pass:

* ``targets_moved``     — the constitutive sub-step wrote something. Zero == zero
  is bit-identical, so agreement without motion is agreement about nothing.
* ``auxiliary_advanced_after_first_step`` — ``f_w`` moved at some step AFTER the
  first. This is the floor that says the HISTORY term is live rather than merely
  written once, and it is the property the frozen-source arrangement above would
  silently fail. Zero-init is a fixed point of this sub-step; so is a frozen
  source.
* ``sources_moved``     — B and D advanced, which is what drives the above.

=============================================================================
THE GUARD IS AN AXIS HERE, AT CORPUS SCALE
=============================================================================

``--guard default_no_options`` recompiles the two kernels with NVRTC's default
options — i.e. with ``--fmad=false`` STRIPPED — through the probe's ``CupyShim``.
Both accumulations and the E side's ``D*inv_eps`` are FMA candidates, so the
control must DIVERGE; the pass criterion is that it does so on at least one
inexact-Courant case, and it is a FAILURE of the control if it agrees everywhere.

DO NOT CARRY THE CURL'S REASONING ABOUT COURANT 0.5 OVER TO THIS SUB-STEP. On the
curl, ``dtdx`` multiplies the stencil, so at 0.5 the scaling is exact in float32
and a contracted and an uncontracted expression round identically — which is why
that leg could not distinguish the guard at 0.5 at all. **This sub-step never
multiplies by dtdx.** Its coefficients are ``kps``/``kms`` off the absorber
profile, arbitrary floats at either Courant, so the contraction is visible in
both columns and the Courant is an axis here for a different reason: it changes
the PML tables the driver builds, not the exponent of a scale factor. The two
columns are reported separately regardless, so the measurement says which
happened rather than the docstring predicting it.

Run (the GPU host, one GPU)::

    CUDA_VISIBLE_DEVICES=0 python -u validate_constitutive_kernel_on_lifted_cases.py \\
        --out results/<dir>/constitutive_lifted.json --subnormal-policy keep

One flushed line per case, incremental JSON (the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)

#: The sibling curl leg owns the case list, the Courant pair and the cases.py
#: location. Imported rather than restated so the two legs cannot drift into
#: describing different problems while quoting the same case names.
_CURL_LEG_PATH = os.path.join(HERE, "validate_pml_kernel_on_lifted_cases.py")

#: Consecutive WHOLE steps per case. The synthetic multi-step leg runs 60; this
#: one runs 8 for the reason the curl's lifted leg does — a corpus case at
#: res=160 is a real simulation step, not a 13x11x9 launch, and the divergence
#: this leg exists to catch (a wrong f_w feeding the next curl) shows on step two.
#: The 60-step budget is carried by the synthetic leg, which is where the
#: measurement behind 60 was taken.
STEPS = 8

CONSTITUTIVE_TARGETS = ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")
CONSTITUTIVE_AUXILIARIES = tuple("f_w_" + name for name in CONSTITUTIVE_TARGETS)
CONSTITUTIVE_SOURCES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")

GUARDS = {
    "fmad_false": ("--fmad=false",),
    "default_no_options": (),
}


def log(message: str) -> None:
    print(message, flush=True)


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_curl_leg():
    return _load(_CURL_LEG_PATH, "validate_pml_lifted")


#: Attribute-name fragment that marks a SCRATCH buffer rather than state.
#:
#: THE ONE ASYMMETRY BETWEEN THE TWO LEGS THAT IS NOT A DEFECT. The array path
#: runs ``_apply_constitutive_pml`` through ``fields.scratch`` (stepping.py:2136)
#: and the kernel does not, so any scratch buffer left addressable on ``fields``
#: holds the array leg's last intermediate and the kernel leg's older one. That
#: is a difference in a temporary nothing reads, and comparing it would
#: manufacture a divergence the sub-step does not have.
#:
#: ``fields.scratch`` itself is a ``StepScratch`` OBJECT (fields.py:298, holding
#: its buffers in ``_slots``) so the isinstance filter already drops it; the name
#: filter is for ``_fmp_scratch`` and ``_fmp_scratch_by_component``
#: (fields.py:551-557), which ARE bare arrays on the instance. On the covered
#: configurations they stay ``None`` — ``displacement_minus_polarization`` returns
#: the D array itself where no susceptibility drives the component
#: (fields.py:1097-1098), which is exactly what the E-side predicate requires —
#: so this excludes nothing today. It is here so that a future covered case which
#: does allocate one cannot turn a temporary into a red gate.
_SCRATCH_MARKER = "scratch"


def device_state(fields, cupy) -> Dict[str, Any]:
    """Every CuPy array the driver holds as STATE, by attribute name.

    Read off ``vars(fields)`` — the INSTANCE dict — and not ``dir()``: the latter
    reaches properties that allocate (``get_H`` computes B/mu on demand) and one
    of them, ``inv_eps``, is a re-pointed view rather than storage. What this must
    capture is everything a sub-step can WRITE, so that the two legs provably
    start from the same bytes; capturing the read-only material volumes along with
    it costs one copy, removes the judgement call, and makes those volumes a free
    control — they must come back identical because neither leg writes them.

    Scratch is excluded by name (:data:`_SCRATCH_MARKER`) and for one reason only,
    written out there.
    """
    return {name: value for name, value in vars(fields).items()
            if isinstance(value, cupy.ndarray) and _SCRATCH_MARKER not in name}


def one_case(name: str, kwargs: Dict[str, Any], probe,
             courant: float, guard_label: str) -> Dict[str, Any]:
    import cupy

    from meep_gpu import backends, stepping
    from meep_gpu.cuda_kernels import constitutive_kernels as kernels
    from meep_gpu.cuda_kernels import coverage
    from meep_gpu.from_meep import lift_simulation
    import cases

    backends.guard_kernel_compilation(cupy)
    simulation = cases.BUILDERS[name](**kwargs)
    # dtdx IS the Courant number: Grid sets dt = courant/resolution and
    # dx = 1/resolution, and from_meep reads sim.Courant straight through.
    simulation.Courant = float(courant)
    driver = lift_simulation(simulation, prefer_gpu=True, gpu_id=0)
    # Two real steps first, so the arrays hold a simulation's values and every
    # lazily-allocated buffer exists before either leg runs.
    driver.run(num_steps=2)

    fields, pml, grid = driver.fields, driver.pml, driver.grid
    covered_H, reason_H = coverage.covers_real_pml_constitutive(fields, pml, grid, "H")
    covered_E, reason_E = coverage.covers_real_pml_constitutive(fields, pml, grid, "E")
    record: Dict[str, Any] = {
        "case": name, "kwargs": kwargs,
        "courant": float(courant),
        "guard_label": guard_label,
        "guard": list(GUARDS[guard_label]),
        "guard_is_primary": guard_label == "fmad_false",
        "shape": [int(s) for s in driver.shape],
        "dtdx": float(grid.dt / grid.dx),
        "dtdx_float32_exact": bool(float(np.float32(grid.dt / grid.dx))
                                   == float(grid.dt / grid.dx)),
        "pml_thickness_by_face": [list(f) for f in pml.thickness_by_face],
        "covered_H": covered_H, "coverage_reason_H": reason_H,
        "covered_E": covered_E, "coverage_reason_E": reason_E,
        "covered": bool(covered_H and covered_E),
        "steps": STEPS,
    }
    if not record["covered"]:
        driver.close()
        return record

    baseline = {key: value.copy() for key, value in device_state(fields, cupy).items()}
    record["compared_arrays"] = sorted(baseline)

    # ---- ARRAY LEG: MEEP's order, constitutive from stepping.py -------------
    array_steps: List[Dict[str, np.ndarray]] = []
    for _ in range(STEPS):
        stepping.step_B(fields, pml)
        stepping.update_H(fields, pml)
        stepping.step_D(fields, pml)
        stepping.update_E(fields, pml)
        cupy.cuda.runtime.deviceSynchronize()
        # Snapshotted to the HOST: eight steps of every volume would be gigabytes
        # of device memory on the 3-D cases, and the comparison is a byte compare
        # that ``probe.bit_compare`` runs host-side anyway.
        array_steps.append({key: cupy.asnumpy(getattr(fields, key))
                            for key in baseline})

    # Liveness, measured on the ARRAY leg — the oracle, never the kernel.
    first, last = array_steps[0], array_steps[-1]
    base_host = {key: cupy.asnumpy(value) for key, value in baseline.items()}
    moved_targets = {key: not bool(np.array_equal(last[key], base_host[key]))
                     for key in CONSTITUTIVE_TARGETS if key in base_host}
    moved_sources = {key: not bool(np.array_equal(last[key], base_host[key]))
                     for key in CONSTITUTIVE_SOURCES if key in base_host}
    auxiliary_advanced = any(
        not bool(np.array_equal(array_steps[step][key], array_steps[step - 1][key]))
        for step in range(1, STEPS)
        for key in CONSTITUTIVE_AUXILIARIES if key in base_host)
    record["targets_moved"] = moved_targets
    record["targets_moved_count"] = f"{sum(moved_targets.values())}/{len(moved_targets)}"
    record["sources_moved_count"] = f"{sum(moved_sources.values())}/{len(moved_sources)}"
    # THE VACUITY FLOOR for this sub-step: f_w moved at a step AFTER the first.
    # A frozen source makes f_w a fixed point from launch one, and the whole
    # recurrence would then be compared with its history term inert.
    record["auxiliary_advanced_after_first_step"] = bool(auxiliary_advanced)
    record["auxiliary_moved_on_first_step"] = any(
        not bool(np.array_equal(first[key], base_host[key]))
        for key in CONSTITUTIVE_AUXILIARIES if key in base_host)
    record["largest_target_magnitude"] = max(
        float(np.abs(last[key]).max()) for key in moved_targets) if moved_targets else 0.0
    # What the operands ACTUALLY held on a real corpus case, recorded rather than
    # assumed. The corpus is a normal-number regime; the subnormal band is the
    # SYNTHETIC gate's axis, and saying so with a count is what stops this leg
    # being quoted as band evidence it does not carry.
    record["operand_census"] = probe.operand_census(
        {key: base_host[key] for key in
         tuple(CONSTITUTIVE_TARGETS) + CONSTITUTIVE_AUXILIARIES + CONSTITUTIVE_SOURCES
         if key in base_host})

    # ---- restore, then the KERNEL LEG -------------------------------------
    for key, value in baseline.items():
        getattr(fields, key)[...] = value

    guard = GUARDS[guard_label]
    # THE CACHE CLEAR IS LOAD-BEARING, NOT HYGIENE. ``_get_kernel`` builds its
    # memo key from the MODULE's own ``_COMPILE_OPTIONS`` — always
    # ``('--fmad=false',)`` — while ``CupyShim`` substitutes the guard's options
    # at the ``cp.RawKernel`` call below it. So the key cannot tell the two guards
    # apart, and a guarded binary compiled earlier in this process would be served
    # to the unguarded leg under a key that claims it is guarded. Clearing after
    # installing the shim, per case, is what keeps the control honest; the probe's
    # own ``begin_guard`` does exactly the same thing for the same reason.
    kernels.cp = probe.CupyShim(guard)
    kernels._clear_kernel_cache()

    per_step: List[Dict[str, Any]] = []
    first_divergence: Optional[Dict[str, Any]] = None
    launch_error: Optional[str] = None
    try:
        for step in range(1, STEPS + 1):
            stepping.step_B(fields, pml)
            kernels.update_fused_pml_real("H", fields, pml)
            stepping.step_D(fields, pml)
            kernels.update_fused_pml_real("E", fields, pml)
            cupy.cuda.runtime.deviceSynchronize()
            expected = array_steps[step - 1]
            parts = {key: probe.bit_compare(getattr(fields, key), expected[key])
                     for key in baseline}
            verdict = probe.combine(parts)
            per_step.append({
                "step": step,
                "bit_identical": verdict["bit_identical"],
                "differing_floats": verdict["differing_floats"],
                "total_floats": verdict["total_floats"],
                "max_ulp": verdict.get("max_ulp", 0),
                "differing_arrays": sorted(
                    key for key, part in parts.items() if not part["bit_identical"]),
            })
            if not verdict["bit_identical"] and first_divergence is None:
                worst = sorted(
                    ((key, part) for key, part in parts.items()
                     if not part["bit_identical"]),
                    key=lambda item: -item[1]["differing_floats"])
                first_divergence = {
                    "step": step,
                    "arrays": [key for key, _ in worst],
                    "first_array": worst[0][0],
                    "detail": worst[0][1],
                }
                # STOP on the first divergence: the remaining steps compound an
                # error already located, and the question this leg answers is
                # WHERE it first appeared.
                break
    except Exception as exc:  # noqa: BLE001 - a launch failure is a row
        launch_error = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        kernels.cp = cupy
        kernels._clear_kernel_cache()

    record["launch_error"] = launch_error
    record["per_step"] = per_step
    record["identical_steps"] = sum(1 for s in per_step if s["bit_identical"])
    record["first_divergence"] = first_divergence
    record["bit_identical"] = bool(
        launch_error is None and len(per_step) == STEPS
        and all(s["bit_identical"] for s in per_step))

    driver.close()
    cupy.get_default_memory_pool().free_all_blocks()
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--guard", default="fmad_false", choices=sorted(GUARDS),
                        help="NVRTC options the two kernels compile under. "
                             "'default_no_options' STRIPS --fmad=false and is the "
                             "control: it must diverge at an inexact Courant.")
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("flush", "keep", "match_meep"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    curl_leg = load_curl_leg()
    if curl_leg.CASES_DIR not in sys.path:
        sys.path.insert(0, curl_leg.CASES_DIR)
    probe = curl_leg.load_probe()

    binary_observer = probe.install_nvrtc_binary_observer()
    meep_import = (probe.import_meep_for_host_policy()
                   if args.import_meep_for_host_policy else {"requested": False})
    policy_install = (probe.install_subnormal_policy_for_run(args.subnormal_policy,
                                                             REPO_API)
                      if args.subnormal_policy else None)

    plan = [(name, kwargs, courant)
            for courant in curl_leg.COURANTS for name, kwargs in curl_leg.SHAPES]
    results: Dict[str, Any] = {
        "check": "constitutive_kernel_on_lifted_cases",
        "kernels": ["update_H_pml_real", "update_E_pml_real"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "whole_steps_per_case": STEPS,
        "guard_label": args.guard,
        "guard": list(GUARDS[args.guard]),
        "courants": list(curl_leg.COURANTS),
        "_why_the_whole_step": (
            "fw[i] = src is unconditional, so a frozen source makes f_w a fixed "
            "point from launch one. step_B and step_D run from the ARRAY PATH in "
            "both legs so B and D evolve as a real run evolves them; only the two "
            "constitutive sub-steps are swapped."),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR", "<unset>"),
        "subnormal_policy_requested": args.subnormal_policy,
        "subnormal_policy_install": policy_install,
        "subnormal_policy": probe.subnormal_policy_stamp(REPO_API),
        "meep_import_for_host_policy": meep_import,
        "nvrtc_binary_observer": binary_observer,
        "cases": [],
    }
    started = time.time()
    for index, (name, kwargs, courant) in enumerate(plan, 1):
        case_started = time.time()
        try:
            record = one_case(name, kwargs, probe, courant, args.guard)
        except Exception as exc:  # noqa: BLE001 - a failure is a row
            record = {"case": name, "kwargs": kwargs, "courant": float(courant),
                      "guard_label": args.guard,
                      "error": f"{type(exc).__name__}: {exc}"[:2000]}
        record["seconds"] = round(time.time() - case_started, 2)
        results["cases"].append(record)
        if "error" in record:
            log(f"[clifted] {index}/{len(plan)} {name} {kwargs} C={courant} "
                f"guard={args.guard}: FAILED {record['error']}")
        elif not record["covered"]:
            log(f"[clifted] {index}/{len(plan)} {name} {kwargs} C={courant}: "
                f"NOT COVERED (H: {record['coverage_reason_H']} | "
                f"E: {record['coverage_reason_E']})")
        else:
            log(f"[clifted] {index}/{len(plan)} {name} {kwargs} C={courant} "
                f"guard={args.guard} shape={record['shape']} "
                f"dtdx={record['dtdx']:.6f} (f32-exact={record['dtdx_float32_exact']}): "
                f"identical={record['bit_identical']} "
                f"({record['identical_steps']}/{STEPS} steps) "
                f"targets_moved={record['targets_moved_count']} "
                f"sources_moved={record['sources_moved_count']} "
                f"fw_advanced={record['auxiliary_advanced_after_first_step']} "
                f"subnormals={record['operand_census'].get('subnormals', 0)} "
                f"maxabs={record['largest_target_magnitude']:.3e} "
                f"({record['seconds']} s)")
            if record["first_divergence"]:
                div = record["first_divergence"]
                log(f"[clifted]   FIRST DIVERGENCE step={div['step']} "
                    f"array={div['first_array']} "
                    f"differing={div['detail']['differing_floats']}"
                    f"/{div['detail']['total_floats']} "
                    f"max_ulp={div['detail'].get('max_ulp')} "
                    f"all_arrays={div['arrays']}")
        with open(args.out + ".tmp", "w") as handle:
            json.dump(results, handle, indent=2, default=str)
        os.replace(args.out + ".tmp", args.out)

    covered = [c for c in results["cases"] if c.get("covered") and "error" not in c]
    identical = [c for c in covered if c["bit_identical"]]
    inexact = [c for c in covered if not c["dtdx_float32_exact"]]
    inexact_identical = [c for c in inexact if c["bit_identical"]]
    live = [c for c in covered
            if c["auxiliary_advanced_after_first_step"]
            and c["largest_target_magnitude"] > 0.0
            and c["targets_moved_count"] != f"0/{len(CONSTITUTIVE_TARGETS)}"]
    results["summary"] = {
        "guard_label": args.guard,
        "covered": len(covered),
        "identical": len(identical),
        "covered_at_inexact_dtdx": len(inexact),
        "identical_at_inexact_dtdx": len(inexact_identical),
        "diverged_at_inexact_dtdx": len(inexact) - len(inexact_identical),
        "live": len(live),
        # THE PRIMARY GUARD must reach total agreement on every covered case.
        #
        # THE CONTROL IS SCORED ON WHETHER IT RAN, NOT ON WHETHER IT DIVERGED,
        # and that is a correction rather than a loosening. The first version
        # required the control to diverge — copied from the curl, where dtdx
        # multiplies the stencil in EVERY cell so stripping the guard moves the
        # whole volume. It does not transfer. Here the only non-unit factors are
        # kps/kms off the absorber profile, and `pml.py:686` builds the interior
        # as the exact identity kps == kms == 1.0; `fma(1.0f, src, f)` and
        # `f + 1.0f*src` are the same float32, so the contraction is invisible
        # everywhere except inside the layer. MEASURED: the control agreed on all
        # twelve cases at BOTH Courants, and `probe_constitutive_absorber_reach.py`
        # measures the reason — in the 2 warm-up + 8 compared steps the pulse has
        # not reached the absorber, so every cell where a coefficient could bite
        # still holds exactly zero.
        #
        # Requiring divergence here would therefore fail a healthy kernel for a
        # property of the fixture. The divergence count is REPORTED so the record
        # can state plainly that the corpus leg is evidence for the transcription
        # and NOT for the arithmetic guard; the guard evidence is the synthetic
        # gate's unguarded control, which diverges 240/240.
        "pass": (
            bool(covered) and len(live) == len(covered) and bool(inexact)
            and (len(identical) == len(covered) if args.guard == "fmad_false"
                 else True)),
        "_control_note": (
            "A default_no_options run scores on validity (covered, live, an "
            "inexact Courant present). Its divergence count is a MEASUREMENT: "
            "see diverged_at_inexact_dtdx and probe_constitutive_absorber_reach.py."
            if args.guard != "fmad_false" else
            "Primary guard: every covered case must be bit-identical."),
    }
    results["subnormal_policy"] = probe.subnormal_policy_stamp(REPO_API)
    results["nvrtc_binaries"] = probe.nvrtc_binary_report()
    results["elapsed_seconds"] = round(time.time() - started, 2)
    log(f"[observer] nvrtc calls={results['nvrtc_binaries']['nvrtc_calls_observed']} "
        f"distinct_binaries={results['nvrtc_binaries']['distinct_binaries']} "
        f"ftz_true_reached_nvrtc="
        f"{results['nvrtc_binaries']['any_ftz_true_reached_nvrtc']}")
    with open(args.out + ".tmp", "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    os.replace(args.out + ".tmp", args.out)
    log(f"[done] guard={args.guard} "
        f"policy={results['subnormal_policy'].get('policy')} "
        f"{results['summary']['identical']}/{results['summary']['covered']} "
        f"lifted cases bit-identical over {STEPS} whole steps "
        f"({results['summary']['identical_at_inexact_dtdx']}/"
        f"{results['summary']['covered_at_inexact_dtdx']} at an INEXACT dtdx); "
        f"live={results['summary']['live']}/{results['summary']['covered']}; "
        f"pass={results['summary']['pass']} ({results['elapsed_seconds']} s)")
    return 0 if results["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
