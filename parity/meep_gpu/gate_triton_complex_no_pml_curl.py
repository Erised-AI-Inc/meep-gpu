"""Byte gate for the COMPLEX NO-ABSORBER curl arm (``complex_no_pml_curl``).

DEVICE RESULT (2026-08-17). This gate released on the GPU host's RTX A6000 with the
``keep`` policy installed: 8/8 shipped product rows were byte-identical over
eight launches, all three controls and all three mutations diverged, and the
predicate partition agreed on every refusal row. The adversarial signed-zero
leg is recorded rather than asserted away: step_B remained identical, while
step_D differed by 86,450/281,250 words in 3-D and 3,552/11,250 in 2-D. The
tracked provenance is ``triton_complex_no_pml_curl_device_gate`` in
``meep_gpu/triton_kernels/fingerprints.json``.

WHAT IT MEASURES, AND WHY IT IS NOT THE GROUP-(J) PROBE AGAIN
------------------------------------------------------------
``probe_group_j_bodies.py`` measured a HAND-ASSEMBLED degenerate binding: it
built a ``ComplexPmlCurlPlan`` in the probe itself, bound ``PML(thickness=0)``'s
own coefficient tables, and wrapped the launch to zero its auxiliaries. That
settled whether the ARITHMETIC degenerates correctly (it does: 0 / 281250 on both
row shapes, three controls diverging). It settled nothing about the RESOLUTION —
which predicate admits, which builder binds, and whether the thing the tree
ships assembles the same launch.

This gate drives ``complex_no_pml_curl.plan_complex_no_pml_curl`` — the shipped
route, through the shipped predicate — and it therefore covers the two failure
modes this campaign's live defects have actually been: a pointer cached at build
time, and a plan built over a ``None``. Both no-absorber shapes the engine can
present are run, because they fail differently:

* ``PML(thickness=0)``, an inert layer object — the shape the group-(J) probe ran;
* ``pml=None``, no layer object at all — which the group-(J) probe never ran, and
  on which a builder that reaches for ``pml.kms_x`` raises rather than refuses.

LEGS, selected with ``--leg`` (default: all)

  product     the four slots (2 row shapes x {step_B, step_D}) x both no-absorber
              shapes, shipped plan vs ``stepping``, whole-inventory uint32.
  controls    PHASEOFF and STALESOURCE, which MUST diverge. A null beside a dead
              control is not evidence — group (G) measured IDENTICAL over four
              cycles and was wrong.
  adversarial the ``minus_zero_target_zero_curl`` row: the ONE pattern on which
              the degenerate binding is known to diverge from stepping.py:539
              (target ``-0.0``, curl ``+0.0``). Measured on device and RECORDED,
              never asserted away. Its word count is the release payload's, and
              ``validate_payload`` requires the field to be present rather than
              requiring it to be zero.
  mutations   deliberately broken copies of the SHIPPED builder, each of which
              must be CAUGHT. The load-bearing one is ``aux_zeroed_once``: it
              hoists the zeroing out of ``run`` to plan time, which is invisible
              on cycle 1 and wrong on every cycle after.
  refusals    the predicate's own answers on the adjacent configurations, so an
              over-broad clause is caught here and not in a later census.

EVERY MEASUREMENT PRIMITIVE IS IMPORTED from ``probe_residual_group_bodies``
unmodified — the uint32 whole-inventory comparison, the per-cycle bracketed
vacuity gate, the live signed-zero lattice and its restore, the every-volume
perturbation, and the complex seeding that writes BOTH halves. The 2026-08-15
adversarial audit found four defects in the superseded measurement and repaired
them there; running a copy is how one gets re-introduced.

THE POLICY TRAP, which cost the group-(J) leg a run. ``MEEP_GPU_SUBNORMAL_POLICY``
in the environment is NOT enough: the complex expansion licence is
policy-conditional and ``complex_fields._policy_in_force`` fails closed on an env
var, so every complex plan refuses for a harness reason wearing a coverage
reason's clothes. This gate calls ``install_subnormal_policy('keep')`` before the
first CuPy elementwise op or Triton compile, and records what it installed.

THE SINGLE COMMAND THE NEXT DEVICE VISIT RUNS::

    cd <repo>/parity/meep_gpu && \\
    MEEP_GPU_SUBNORMAL_POLICY=keep \\
    CUDA_VISIBLE_DEVICES=<one idle GPU> PYTHONPATH=<repo>:. \\
    python -u gate_triton_complex_no_pml_curl.py \\
        --out results/complex_no_pml_curl_<UTCSTAMP>/gate.json --cycles 8

Rule 7: one flushed line per case, and one fsync'd JSONL row per case written as
it lands, so an interrupted run keeps everything up to the failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_API = HERE.parents[1]
for _path in (str(HERE), str(REPO_API)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import probe_residual_group_bodies as P  # noqa: E402  (primitives, unmodified)
from meep_gpu import stepping  # noqa: E402

#: Group (J)'s two rows, transcribed from
#: ``results/group_j_2026-08-16/PROVENANCE.md``.
J_3D = dict(cell=(2.5, 2.5, 2.5), dimensions=3, boundaries="periodic",
            k_point=(0.23, -0.17, 0.35), complex_storage=True, offdiag=True,
            pml_thickness=0)
J_2D = dict(cell=(2.5, 2.5, 0.0), dimensions=2, boundaries="periodic",
            k_point=(0.3892, 0.1597, 0.0), complex_storage=True, offdiag=True,
            pml_thickness=0)

CYCLES = 8
SUB_STEP_FUNCTIONS = {"step_B": stepping.step_B, "step_D": stepping.step_D}


def log(handle, message: str) -> None:
    """One flushed line, to stdout AND to the run's own log (rule 7)."""
    print(message, flush=True)
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def emit(jsonl, row: Dict[str, Any]) -> None:
    """One fsync'd JSONL row per case, as it lands — never only at the end."""
    if jsonl is None:
        return
    jsonl.write(json.dumps(row, default=str) + "\n")
    jsonl.flush()
    os.fsync(jsonl.fileno())


# ---------------------------------------------------------------------------
# The shipped route, and the two controls as departures from it
# ---------------------------------------------------------------------------

def shipped_plan(fields: Any, pml: Any, sub_step: str, probe: Any,
                 drop_layer: bool = False):
    """``plan_complex_no_pml_curl`` and nothing else — the point of this gate.

    A ``None`` here is a GATE FAILURE, not a refusal to be tolerated: the
    predicate is exercised separately in the ``refusals`` leg, and a product leg
    that silently measured nothing is the vacuity this campaign keeps auditing
    for.
    """
    from meep_gpu.triton_kernels import complex_no_pml_curl as arm

    plan = arm.plan_complex_no_pml_curl(fields, None if drop_layer else pml,
                                        sub_step, probe=probe)
    if plan is None:
        raise RuntimeError(
            f"the SHIPPED builder refused {sub_step} on the target configuration "
            f"(drop_layer={drop_layer}); this leg measures nothing without a plan")
    return plan


def phase_off_plan(fields: Any, pml: Any, sub_step: str, probe: Any):
    """CONTROL: the Bloch rotation compiled away (PH* = 0).

    A live control on the exact feature that makes group (J) complex. On the
    hand-built binding it diverged 7350 / 281250, on the wrapped lanes only.
    """
    from meep_gpu.triton_kernels import complex_fields as cx

    plan = shipped_plan(fields, pml, sub_step, probe)
    inner = plan._inner
    unphased = cx.ComplexPmlCurlPlan(
        inner.sub_step, inner.shape, inner.dtdx, inner.bc,
        (0, 0, 0), (1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
        inner.expansion, inner.block,
        [], [], [], [], num_warps=inner.num_warps)
    # Re-bind the SHIPPED plan's own pointers; only the phase constexprs move.
    unphased._targets = inner._targets
    unphased._aux = inner._aux
    unphased._sources = inner._sources
    unphased._coefficients = inner._coefficients
    plan._inner = unphased
    return plan


def stale_source_plan(fields: Any, pml: Any, sub_step: str, probe: Any):
    """CONTROL: sources bound as a plan-time SNAPSHOT.

    Group (G)'s false-pass mechanism, re-armed. It diverged 93750 / 281250 on the
    hand-built binding and must diverge here too, or this gate cannot tell a live
    source binding from a dead one.
    """
    import cupy as cp
    from meep_gpu.triton_kernels import complex_fields as cx
    from meep_gpu.triton_kernels import complex_no_pml_curl as arm

    plan = shipped_plan(fields, pml, sub_step, probe)
    inner = plan._inner
    frozen = [array.copy() for array in arm.source_arrays(fields, sub_step)]
    inner._sources = tuple(cx.CupyPointer(cx._word_view(a)) for a in frozen)
    assert isinstance(frozen[0], cp.ndarray)
    return plan


def aux_never_zeroed_plan(fields: Any, pml: Any, sub_step: str, probe: Any):
    """MUTATION: the auxiliary is zeroed ONCE, at plan time, instead of per launch.

    Invisible on cycle 1 and wrong on every cycle after, which is precisely why
    it is here: a gate that ran a single cycle would certify it.
    """
    plan = shipped_plan(fields, pml, sub_step, probe)
    inner = plan._inner

    class _HoistedZeroing:
        sub_step = inner.sub_step

        def run(self, guard: Optional[bool] = None) -> None:
            inner.run(guard)

    plan._inner = _HoistedZeroing()
    plan._auxiliaries = ()          # nothing left for run() to zero
    return plan


def unit_columns_perturbed_plan(fields: Any, pml: Any, sub_step: str, probe: Any):
    """MUTATION: one synthesized coefficient column is 1.0 + 1 ulp, not 1.0.

    The binding's whole premise is that the columns are EXACTLY one. A column
    that is merely close must be caught, or the gate would license a near-unit
    binding as a degenerate one.
    """
    plan = shipped_plan(fields, pml, sub_step, probe)
    inner = plan._inner
    import cupy as cp
    from meep_gpu.triton_kernels import complex_fields as cx
    from meep_gpu.triton_kernels.launch import _flat

    nudged = cp.full(inner.shape[0], np.float32(np.nextafter(np.float32(1.0),
                                                             np.float32(2.0))),
                     dtype=cp.float32)
    coefficients = list(inner._coefficients)
    coefficients[0] = cx.CupyPointer(_flat(nudged))
    inner._coefficients = tuple(coefficients)
    return plan


# ---------------------------------------------------------------------------
# The adversarial row — measured and RECORDED, never asserted away
# ---------------------------------------------------------------------------

def leg_adversarial(probe: Any, handle) -> Dict[str, Any]:
    """The one pattern the degenerate binding is known to lose: (-0.0, +0.0).

    Seeded into the TARGET volumes directly, with the curl driven to exactly zero
    by making every source volume uniform (a uniform field has an identically
    zero curl, so no source-side trickery is needed and the arithmetic reaching
    the recurrence is the shipped one).

    The count is REPORTED. This leg has no pass/fail threshold on it, because the
    honest statement is a size, and hiding a known divergence behind an assertion
    is how a residual becomes a defect.
    """
    import cupy as cp

    out: Dict[str, Any] = {"leg": "adversarial", "cases": []}
    for tag, config in (("J3", J_3D), ("J2", J_2D)):
        for sub_step in ("step_B", "step_D"):
            reference_fields, pml, _ = P.build(cp, **config)
            candidate_fields, _, _ = P.build(cp, **config)
            shape = tuple(reference_fields.grid.shape)
            minus_zero = cp.empty(shape, dtype=cp.complex64)
            minus_zero.real = cp.full(shape, np.float32(-0.0), dtype=cp.float32)
            minus_zero.imag = cp.full(shape, np.float32(-0.0), dtype=cp.float32)
            uniform = cp.empty(shape, dtype=cp.complex64)
            uniform.real = cp.full(shape, np.float32(0.25), dtype=cp.float32)
            uniform.imag = cp.full(shape, np.float32(-0.125), dtype=cp.float32)
            for fields in (reference_fields, candidate_fields):
                from meep_gpu.triton_kernels.launch import SUB_STEPS

                for name in SUB_STEPS[sub_step]["targets"]:
                    getattr(fields, name)[...] = minus_zero
                for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
                    array = getattr(fields, name, None)
                    if array is not None:
                        array[...] = uniform
            try:
                plan = shipped_plan(candidate_fields, pml, sub_step, probe)
                SUB_STEP_FUNCTIONS[sub_step](reference_fields, pml)
                plan.run()
                verdict = P.compare(P.inventory(reference_fields),
                                    P.inventory(candidate_fields))
                row = {"case": f"{tag}_{sub_step}_minus_zero_target_zero_curl",
                       "differing": verdict["differing"],
                       "compared": verdict["words"],
                       "identical": verdict["identical"],
                       "first_difference": verdict.get("first_difference")}
            except BaseException as exc:  # noqa: BLE001
                row = {"case": f"{tag}_{sub_step}_minus_zero_target_zero_curl",
                       "error": f"{type(exc).__name__}: {exc}"[:300],
                       "traceback": traceback.format_exc()[-800:]}
            out["cases"].append(row)
            log(handle, f"  adversarial {row['case']}: "
                        f"{row.get('differing', 'ERROR')} / "
                        f"{row.get('compared', '?')} differing")
    return out


def leg_refusals(probe: Any, handle) -> Dict[str, Any]:
    """The predicate's own answers, so an over-broad clause is caught HERE."""
    import cupy as cp
    from meep_gpu.triton_kernels import complex_fields as cx
    from meep_gpu.triton_kernels import complex_no_pml_curl as arm

    out: Dict[str, Any] = {"leg": "refusals", "cases": []}
    checks = [
        ("target_J3_inert_layer", dict(J_3D), False, True),
        ("target_J2_inert_layer", dict(J_2D), False, True),
        ("active_layer_is_refused", dict(J_3D, pml_thickness=2), False, False),
        ("real_storage_is_refused",
         dict(J_3D, complex_storage=False, k_point=(0.0, 0.0, 0.0)), False, False),
    ]
    for name, config, drop, expected in checks:
        fields, pml, _ = P.build(cp, **config)
        for sub_step in ("step_B", "step_D"):
            verdict = arm.complex_no_pml_curl_coverage(
                fields, None if drop else pml, sub_step, probe=probe)
            incumbent = cx.complex_pml_curl_coverage(
                fields, None if drop else pml, sub_step, probe=probe)
            row = {"case": f"{name}:{sub_step}", "covered": verdict.covered,
                   "expected": expected, "agrees": verdict.covered == expected,
                   "reasons": list(verdict.reasons)[:6],
                   "incumbent_covered": incumbent.covered,
                   "both_admit": verdict.covered and incumbent.covered}
            out["cases"].append(row)
            log(handle, f"  refusals {row['case']}: covered={row['covered']} "
                        f"expected={expected} both_admit={row['both_admit']}")
    out["disjoint"] = not any(c["both_admit"] for c in out["cases"])
    out["all_agree"] = all(c["agrees"] for c in out["cases"])
    return out


# ---------------------------------------------------------------------------

def build_cases(probe: Any) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    clause = ("complex_fields._complex_grid_reasons clause 3 INVERTED "
              "(no active PML layer) — the arm under test")

    for tag, config in (("J3", J_3D), ("J2", J_2D)):
        def make(offset, config=config):
            import cupy as cp

            return P.build(cp, seed_offset=offset, **config)

        for sub_step in ("step_B", "step_D"):
            for shape_tag, drop in (("inert_layer", False), ("no_layer", True)):
                cases.append({
                    "name": f"{tag}_{sub_step}_{shape_tag}", "group": "J",
                    "sub_step": sub_step, "prediction": "IDENTICAL",
                    "clause": clause, "make": make,
                    "array_path": SUB_STEP_FUNCTIONS[sub_step],
                    "kernel_plan": (lambda f, p, s=sub_step, d=drop:
                                    shipped_plan(f, p, s, probe, drop_layer=d)),
                })

    def make3(offset):
        import cupy as cp

        return P.build(cp, seed_offset=offset, **J_3D)

    for sub_step in ("step_B", "step_D"):
        cases.append({
            "name": f"J3_{sub_step}_PHASEOFF_CONTROL", "group": "J",
            "sub_step": sub_step, "prediction": "DIVERGENT",
            "clause": "control: the Bloch rotation compiled away",
            "make": make3, "array_path": SUB_STEP_FUNCTIONS[sub_step],
            "kernel_plan": (lambda f, p, s=sub_step:
                            phase_off_plan(f, p, s, probe)),
        })
    cases.append({
        "name": "J3_step_B_STALESOURCE_CONTROL", "group": "J",
        "sub_step": "step_B", "prediction": "DIVERGENT",
        "clause": "control: sources bound as a plan-time snapshot (group G)",
        "make": make3, "array_path": stepping.step_B,
        "kernel_plan": lambda f, p: stale_source_plan(f, p, "step_B", probe),
    })
    for sub_step in ("step_B", "step_D"):
        cases.append({
            "name": f"J3_{sub_step}_MUTATION_aux_zeroed_once", "group": "J",
            "sub_step": sub_step, "prediction": "DIVERGENT",
            "clause": "mutation: the auxiliary zeroing hoisted to plan time",
            "make": make3, "array_path": SUB_STEP_FUNCTIONS[sub_step],
            "kernel_plan": (lambda f, p, s=sub_step:
                            aux_never_zeroed_plan(f, p, s, probe)),
        })
    cases.append({
        "name": "J3_step_B_MUTATION_unit_column_one_ulp", "group": "J",
        "sub_step": "step_B", "prediction": "DIVERGENT",
        "clause": "mutation: a synthesized coefficient column is 1.0 + 1 ulp",
        "make": make3, "array_path": stepping.step_B,
        "kernel_plan": lambda f, p: unit_columns_perturbed_plan(f, p, "step_B",
                                                                probe),
    })
    return cases


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The release contract. Refusals are BY NAME; nothing is inferred from absence."""
    reasons: List[str] = []
    if payload.get("subnormal_policy_installed") != "keep":
        reasons.append("the 'keep' subnormal policy was not INSTALLED (an env var "
                       "alone leaves _policy_in_force closed and every complex "
                       "plan refuses for a harness reason)")
    if payload.get("expansion") is None:
        reasons.append("no EXPANSION constexpr was licensed from a probe artifact")

    rows = payload.get("cases") or []
    by_name = {row.get("case"): row for row in rows}
    if not rows:
        reasons.append("no cases ran")
    for row in rows:
        if row.get("error"):
            reasons.append(f"{row.get('case')}: {row['error']}")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(f"{row.get('case')}: verdict {row.get('verdict')} != "
                           f"prediction {row.get('prediction')}")
    controls = [r for r in rows if "CONTROL" in str(r.get("case"))]
    if len(controls) < 3:
        reasons.append(f"{len(controls)} controls ran; a null beside a dead control "
                       f"is not evidence and this gate requires 3")
    if any(r.get("verdict") != "DIVERGENT" for r in controls):
        reasons.append("a control did not diverge; the round is void")
    mutations = [r for r in rows if "MUTATION" in str(r.get("case"))]
    if len(mutations) < 3:
        reasons.append(f"{len(mutations)} mutations ran; this gate requires 3")
    if any(r.get("verdict") != "DIVERGENT" for r in mutations):
        reasons.append("a mutation was NOT caught")

    adversarial = payload.get("adversarial") or {}
    if not adversarial.get("cases"):
        reasons.append("the adversarial (-0.0 target, +0.0 curl) row did not run; "
                       "its word count is REQUIRED in the payload and is reported, "
                       "not asserted to be zero")
    for row in adversarial.get("cases", []):
        if "differing" not in row:
            reasons.append(f"{row.get('case')}: no differing count recorded")

    refusals = payload.get("refusals") or {}
    if not refusals.get("disjoint"):
        reasons.append("two families admit the same slot; a slot admitted by two "
                       "is UNSELECTED and falls silently back to the array path")
    if not refusals.get("all_agree"):
        reasons.append("a predicate answer disagreed with its expectation")

    _ = by_name
    return {"released": not reasons, "reasons": reasons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="byte gate for the complex no-absorber curl arm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cycles", type=int, default=CYCLES)
    parser.add_argument("--leg", default="all",
                        choices=("all", "product", "adversarial", "refusals"))
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(out_path.parent / "driver.log", "a", encoding="utf-8")
    jsonl = open(out_path.parent / "cases.jsonl", "a", encoding="utf-8")

    started = time.time()
    payload: Dict[str, Any] = {"gate": "complex_no_pml_curl",
                               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                            time.gmtime()),
                               "cycles": args.cycles}
    # THE POLICY, INSTALLED before the first CuPy elementwise op or Triton compile.
    from meep_gpu.subnormal_policy import install_subnormal_policy

    install_subnormal_policy("keep")
    payload["subnormal_policy_installed"] = "keep"

    probe = P._load_expansion_probe()
    from meep_gpu.triton_kernels import complex_fields as cx

    payload["expansion"] = cx._resolve_expansion(probe)
    log(handle, f"policy installed: keep; expansion = {payload['expansion']}")
    if payload["expansion"] is None:
        log(handle, "REFUSED: no licensable expansion probe; a guess is not a "
                    "measurement")
        payload["cases"] = []
        payload["release"] = validate_payload(payload)
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        out_path.write_text(json.dumps(payload, indent=1, default=str))
        return 1

    cases: List[Dict[str, Any]] = []
    if args.leg in ("all", "product"):
        built = build_cases(probe)
        if args.only:
            built = [c for c in built if args.only in c["name"]]
        for index, case in enumerate(built, 1):
            log(handle, f"case {index}/{len(built)} {case['name']} "
                        f"(predict {case['prediction']})")
            row = P.run_case(cycles=args.cycles, **case)
            row["verdict"] = ("ERROR" if row.get("error")
                              else ("IDENTICAL" if row.get("identical")
                                    else "DIVERGENT"))
            cases.append(row)
            emit(jsonl, row)
            log(handle, f"  -> {row['verdict']} "
                        f"({row.get('differing_max', '?')} / "
                        f"{row.get('words_compared', '?')}) "
                        f"[{round(time.time() - started, 1)} s]")
    payload["cases"] = cases

    if args.leg in ("all", "adversarial"):
        payload["adversarial"] = leg_adversarial(probe, handle)
    if args.leg in ("all", "refusals"):
        payload["refusals"] = leg_refusals(probe, handle)

    payload["seconds"] = round(time.time() - started, 1)
    payload["release"] = validate_payload(payload)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
    out_path.write_text(json.dumps(payload, indent=1, default=str))
    log(handle, f"RELEASED={payload['release']['released']}")
    for reason in payload["release"]["reasons"]:
        log(handle, f"  refused: {reason}")
    handle.close()
    jsonl.close()
    return 0 if payload["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
