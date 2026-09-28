"""Group (J) — complex + off-diagonal + NO PML — put the closest certified bodies on it.

THE QUESTION. Group (J) is the one residual group that has never had a verdict.
The 2026-08-14 census established only that it is REACHABLE (two rows,
``test_material_grid.py::TestMaterialGrid.test_matgrid_3d`` and
``::test_subpixel_smoothing``, both carrying complex storage, an off-diagonal
chi1inv row and no absorber simultaneously) and that three sub-steps are
unadmitted on it: ``step_B``, ``step_D``, ``update_E``. Nobody has measured
whether the blocking clauses are OVER-BROAD (a certified body would have been
byte-exact) or LOAD-BEARING (a body genuinely diverges and a kernel is needed).

WHY THIS FILE EXISTS RATHER THAN NEW CASES IN THE CLOSURE PROBE. Every primitive
below — the uint32 whole-inventory comparison, the per-cycle bracketed vacuity
gate, the signed-zero lattice and its restore, the every-volume perturbation,
the complex seeding that writes BOTH halves — is IMPORTED from
``probe_residual_group_bodies``, unmodified. The 2026-08-15 adversarial audit
found four defects in the superseded measurement, all four repaired there; the
only way to be sure this round does not re-introduce one is to run the repaired
code rather than a copy of it. This file contributes the CASES and nothing else.

THE BINDING, STATED BEFORE THE RESULT
-------------------------------------
No certified product covers a complex run with NO absorber. The complex family
(``triton_kernels/complex_fields.py``) is split-field only, and the no-absorber
family (``triton_kernels/no_pml.py``) is real float32 only and carries no Bloch
phase — while both of (J)'s rows carry a nonzero k on a periodic wrap, which is
what forces their complex storage in the first place. So the closest body is the
complex one, and putting it on (J) requires a DEGENERATE NO-ABSORBER BINDING.
Two of its three parts are the shipped builder's own expressions; the third is a
substitution and is named as such:

1. **The coefficients are NOT substituted.** ``PML(thickness=0)`` already carries
   ``kms``/``sinv``/``kps`` tables that are EXACTLY 1.0 on both Yee sub-lattices
   (measured, not assumed — ``pml_coefficients_all_exactly_one`` in the artifact),
   so ``[getattr(pml, f"{stem}_{axis}{suffix}") ...]`` — verbatim
   ``plan_complex_pml_curl``'s line — is what gets bound.
2. **The sources are NOT substituted.** ``step_B`` reads the stored E, which
   exists here because the off-diagonal row forces ``stores_E``. ``step_D`` reads
   ``fields.get_H(...)``, the engine's own accessor, which without PML serves the
   B array itself (stepping.py:915-916 returns before storing any H).
3. **The auxiliary IS substituted**, because the array path has none: with the
   absorber gone ``_apply_curl`` takes stepping.py:537, one line, ``target -=
   curl``, and ``fu_*``/``f_w_*`` are never allocated. The faithful stand-in is an
   auxiliary that is ZERO at entry to every sub-step, which is what makes the
   split-field recurrence (stepping.py:1975-1982) collapse onto that one line:

       fu' = (0*kms - curl)*sinv = -curl ;  f' = (f*kms_u + fu' - 0)*sinv_u = f - curl

   with every coefficient exactly 1. Verified on the engine's own
   ``_apply_pml_update`` before the leg ran: 0 differing words of 93750 against a
   plain subtract. The zeroing happens immediately before each launch and NOT
   once at plan time — after one launch the auxiliary holds ``-curl``, and a
   second cycle run against it would measure the substitution rather than the
   body. The auxiliary is outside the compared inventory, so zeroing it cannot
   flatter the comparison.

That binding is as favourable to the kernel as the configuration allows. A
divergence under it is a statement about the BODY: no coefficient or pointer
choice recovers the array path, and the slot needs a kernel rather than a wider
predicate.

``update_E`` gets the same treatment with one honest caveat recorded in the case
itself: the census names ``null_constitutive`` as its closest predicate, but that
product's body is the NO-OP arm (it exists for the configuration where update_E
returns early), so measuring it would measure nothing. The body used instead is
``ComplexConstitutivePlan(side="E")``, which at least computes ``D * inv_eps``
— strictly the more favourable test.

CONTROLS. A null result with no live control proves nothing; group (G) measured
IDENTICAL over four cycles and was wrong. Three controls run beside the subjects,
each of which MUST diverge or the round is void:

* ``PHASEOFF`` — the same body and bindings with ``PHX=PHY=PHZ=0``, which compiles
  the Bloch rotation away. Both (J) rows carry k != 0 on a periodic wrap, so this
  must move the wrapped lane. It is a control on the exact feature that makes (J)
  complex.
* ``STALESOURCE`` — the sources bound as COPIES taken at plan time. ``perturb``
  moves every stored volume between cycles, so a plan holding a snapshot must
  diverge. This is group (G)'s false-pass mechanism, re-armed.

Usage (on a CUDA host, from ``the repository root``)::

    PYTHONPATH=. MEEP_GPU_SUBNORMAL_POLICY=keep python -u \\
        parity/meep_gpu/probe_group_j_bodies.py --out <dir> --cycles 8
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_API = _HERE.parents[1]
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from meep_gpu import stepping  # noqa: E402

# EVERY measurement primitive comes from the repaired closure harness. Importing
# rather than copying is what guarantees the four audited defects stay repaired.
from probe_residual_group_bodies import (  # noqa: E402
    CYCLES,
    _load_expansion_probe,
    build,
    cp,
    inventory,
    log,
    run_case,
    signed_zero_census,
)

#: (J)'s two census rows, transcribed from
#: ``results/predicate_coverage_2026-08-14_allnine/tests.jsonl``. The Courant is
#: the harness's deliberately non-power-of-two 0.35 rather than the rows' 0.5, so
#: no coefficient is exactly representable by accident (the closure probe's own
#: case discipline); everything else is the row's.
J_3D = dict(cell=(2.5, 2.5, 2.5), dimensions=3, boundaries="periodic",
            k_point=(0.23, -0.17, 0.35), complex_storage=True, offdiag=True,
            pml_thickness=0)
J_2D = dict(cell=(2.5, 2.5, 0.0), dimensions=2, boundaries="periodic",
            k_point=(0.3892, 0.1597, 0.0), complex_storage=True, offdiag=True,
            pml_thickness=0)

#: The features every (J) row must carry SIMULTANEOUSLY. Verified on the fixture
#: at build time rather than inherited from the census.
J_FEATURES = ("complex", "offdiag", "no_pml", "stores_E")


def j_feature_check(fields: Any, pml: Any) -> Dict[str, bool]:
    """Assert (J)'s four features on the fixture itself, one boolean each."""
    return {
        "complex": bool(getattr(fields, "force_complex_fields", False)),
        "offdiag": bool(getattr(fields, "has_offdiagonal_epsilon", False)),
        "no_pml": not bool(getattr(pml, "is_active", False)),
        "stores_E": bool(getattr(fields, "stores_E", False)),
        "has_bloch": bool(getattr(fields.grid, "has_bloch", False)),
    }


# ---------------------------------------------------------------------------
# The degenerate no-absorber binding
# ---------------------------------------------------------------------------

def _zero_aux_wrapper(plan: Any, auxiliaries: List[Any], label: str) -> Any:
    """Zero the substituted auxiliaries immediately before each launch.

    The array path has NO auxiliary without an absorber, so zero at entry is the
    faithful stand-in and it must hold on EVERY cycle, not just the first: after
    one launch the auxiliary holds ``-curl`` and the recurrence would no longer
    collapse onto ``target -= curl``. The auxiliaries are outside the compared
    inventory, so this cannot flatter the comparison — it only stops the
    substitution from being measured instead of the body.
    """

    class _ZeroAuxPlan:
        def run(self, guard: Optional[bool] = None) -> None:
            for array in auxiliaries:
                array.fill(0)
            plan.run(guard)

        def __repr__(self) -> str:
            return f"ZeroAux[{label}]({plan!r})"

    return _ZeroAuxPlan()


def _complex_curl_plan(fields: Any, pml: Any, sub_step: str, probe: Any,
                       phase_off: bool = False,
                       stale_sources: bool = False) -> Any:
    """The shipped complex curl BODY and PLAN CLASS on a no-absorber configuration.

    Only the auxiliary is substituted (see the module docstring). ``phase_off``
    and ``stale_sources`` are the two controls and are the only deliberate
    departures from what the shipped builder would bind.
    """
    from meep_gpu.triton_kernels import complex_fields as cx
    from meep_gpu.triton_kernels.kernels import DEFAULT_BLOCK
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, None)   # pml=None: no active layer
    phases = cx.bloch_phase_table(grid, kinds)
    phased, values = cx._phase_arguments(phases, backward=bool(spec["backward"]))
    if phase_off:
        phased = (0, 0, 0)
        values = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    expansion = cx._resolve_expansion(probe)
    if expansion is None:
        raise RuntimeError("no licensable expansion probe: the complex body cannot "
                           "be bound, and a guess would not be a measurement")

    shape = tuple(grid.shape)
    auxiliaries = [cp.zeros(shape, dtype=cp.complex64) for _ in spec["targets"]]
    # step_B reads the stored E (the off-diagonal row forces stores_E); step_D
    # reads get_H, which without PML serves the B array itself.
    sources = [fields.get_H(name) if name.startswith("H") else fields.get_E(name)
               for name in spec["sources"]]
    if stale_sources:
        sources = [array.copy() for array in sources]

    plan = cx.ComplexPmlCurlPlan(
        sub_step, shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        phased, values, expansion,
        DEFAULT_BLOCK,
        [getattr(fields, name) for name in spec["targets"]],
        auxiliaries,
        sources,
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
    )
    return _zero_aux_wrapper(plan, auxiliaries, f"complex_curl:{sub_step}")


def _complex_constitutive_plan(fields: Any, pml: Any, probe: Any) -> Any:
    """The shipped complex constitutive BODY at ``update_E`` on a no-absorber row.

    The census's closest predicate for this slot is ``null_constitutive``, whose
    body is the NO-OP arm; measuring that would measure nothing. This body at
    least forms ``D * inv_eps``, which is the diagonal half of what the array path
    computes (stepping.py:1014), so it is the strictly more favourable test.
    """
    from meep_gpu.triton_kernels import complex_fields as cx
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES
    from meep_gpu.triton_kernels.kernels import DEFAULT_BLOCK

    spec = CONSTITUTIVE_SIDES["E"]
    expansion = cx._resolve_expansion(probe)
    if expansion is None:
        raise RuntimeError("no licensable expansion probe: the complex body cannot "
                           "be bound, and a guess would not be a measurement")
    shape = tuple(fields.grid.shape)
    auxiliaries = [cp.zeros(shape, dtype=cp.complex64) for _ in spec["targets"]]
    plan = cx.ComplexConstitutivePlan(
        "E", shape, expansion, DEFAULT_BLOCK,
        [getattr(fields, name) for name in spec["targets"]],
        auxiliaries,
        [getattr(fields, name) for name in spec["sources"]],
        [fields.inverse_epsilon_for(name) for name in spec["targets"]],
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
    )
    return _zero_aux_wrapper(plan, auxiliaries, "complex_constitutive:E")


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

def build_cases(probe: Any) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    curl_clause = ("no active PML layer (this product implements the complex "
                   "split-field path only) + fu_<target> is not allocated "
                   "[complex_fields.py:1565, :1712-1714]")
    store_clause = ("fields.stores_E is True: update_E writes E[...] = "
                    "(D - sum P) * inv_eps (stepping.py:1022) — the STORE arm — and "
                    "an off-diagonal chi1inv row is installed "
                    "[no_pml_constitutive.py:586 clause set]")

    for tag, config in (("J3", J_3D), ("J2", J_2D)):
        def make(offset, config=config):
            return build(cp, seed_offset=offset, **config)

        for sub_step, fn in (("step_B", stepping.step_B), ("step_D", stepping.step_D)):
            cases.append({
                "name": f"{tag}_{sub_step}", "group": "J", "sub_step": sub_step,
                # The clause set is a PML/allocation clause and nothing about the
                # complex arithmetic or the off-diagonal row, which the curl
                # predicate admits outright (complex_fields.py:1694-1697). If the
                # degenerate binding lands on stepping.py:537 the clause is
                # over-broad at the body level.
                "prediction": "IDENTICAL",
                "clause": curl_clause,
                "make": make, "array_path": fn,
                "kernel_plan": (lambda f, p, s=sub_step:
                                _complex_curl_plan(f, p, s, probe)),
            })
        cases.append({
            "name": f"{tag}_update_E", "group": "J", "sub_step": "update_E",
            # Two independent structural gaps, either alone sufficient: the body
            # ACCUMULATES (f += kps*fw - kms*fw_prev) where the array path STORES
            # (stepping.py:1022), and it is element-wise where the array path adds
            # the off-diagonal row product (stepping.py:1016-1019) that no certified
            # complex body can form.
            "prediction": "DIVERGENT",
            "clause": store_clause,
            "make": make, "array_path": stepping.update_E,
            "kernel_plan": (lambda f, p: _complex_constitutive_plan(f, p, probe)),
        })

    # ---- the controls, on the 3-D row shape --------------------------------
    def make3(offset):
        return build(cp, seed_offset=offset, **J_3D)

    for sub_step, fn in (("step_B", stepping.step_B), ("step_D", stepping.step_D)):
        cases.append({
            "name": f"J3_{sub_step}_PHASEOFF_CONTROL", "group": "J",
            "sub_step": sub_step, "prediction": "DIVERGENT",
            "clause": ("control: k = (0.23, -0.17, 0.35) on three periodic wraps "
                       "MUST reach the wrapped lane"),
            "make": make3, "array_path": fn,
            "kernel_plan": (lambda f, p, s=sub_step:
                            _complex_curl_plan(f, p, s, probe, phase_off=True)),
        })
    cases.append({
        "name": "J3_step_B_STALESOURCE_CONTROL", "group": "J", "sub_step": "step_B",
        "prediction": "DIVERGENT",
        "clause": ("control: sources bound as a plan-time SNAPSHOT — group (G)'s "
                   "false-pass mechanism, re-armed"),
        "make": make3, "array_path": stepping.step_B,
        "kernel_plan": (lambda f, p: _complex_curl_plan(f, p, "step_B", probe,
                                                        stale_sources=True)),
    })
    return cases


# ---------------------------------------------------------------------------

def main(argv) -> int:
    parser = argparse.ArgumentParser(description="group (J) body measurement")
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--cycles", type=int, default=CYCLES)
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "groupj.progress.log"
    results_path = out / "groupj.json"

    results: Dict[str, Any] = {"cycles": args.cycles, "group": "J", "cases": []}
    if cp is None:
        results["skipped"] = "cupy is not importable on this host"
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        log(progress, "SKIPPED: cupy is not importable on this host")
        return 0
    try:
        import triton  # noqa: F401
        results["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        results["skipped"] = f"triton is not importable: {exc}"
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        log(progress, f"SKIPPED: triton is not importable: {exc}")
        return 0

    results["cupy"] = cp.__version__
    results["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

    # THE POLICY IS INSTALLED, NOT MERELY NAMED IN THE ENVIRONMENT, and it is
    # installed HERE — before the first CuPy elementwise op or Triton kernel
    # compiles, because neither executor can recall bytes it has already written.
    # It also has to be an INSTALL rather than a declaration: the complex family's
    # expansion licence is policy-conditional and `_policy_in_force` (complex_
    # fields.py:1182-1204) fails closed, so with nothing installed
    # `_resolve_expansion` returns None and every complex plan below would refuse
    # for a harness reason wearing a coverage reason's clothes (measured on the
    # laptop: MEEP_GPU_SUBNORMAL_POLICY=keep alone leaves it None).
    from meep_gpu.subnormal_policy import install_subnormal_policy  # noqa: PLC0415
    results["subnormal_policy_stamp"] = install_subnormal_policy("keep", strict=True)
    results["subnormal_policy"] = results["subnormal_policy_stamp"].get("policy")
    log(progress, f"subnormal policy installed: {results['subnormal_policy']!r} "
                  f"attained={results['subnormal_policy_stamp'].get('attained')}")

    probe = _load_expansion_probe()
    results["expansion_probe_loaded"] = probe is not None
    from meep_gpu.triton_kernels import complex_fields as _cx  # noqa: PLC0415
    results["expansion_constexpr"] = _cx._resolve_expansion(probe)
    results["expansion_refusals"] = [r[:200] for r in _cx._expansion_reasons(probe)]
    log(progress, f"expansion constexpr = {results['expansion_constexpr']!r} "
                  f"(refusals: {len(results['expansion_refusals'])})")

    # (1) REACHABILITY, VERIFIED NOT INHERITED. Both row shapes must carry all four
    #     of (J)'s features simultaneously on the fixture itself.
    features: Dict[str, Any] = {}
    for tag, config in (("J3", J_3D), ("J2", J_2D)):
        fields, pml, notes = build(cp, **config)
        flags = j_feature_check(fields, pml)
        features[tag] = {"shape": notes["shape"], "flags": flags,
                         "carries_all": all(flags[name] for name in J_FEATURES),
                         "minus0_at_seed": signed_zero_census(fields),
                         "volumes": sorted(inventory(fields)),
                         "pml_coefficients_all_exactly_one": all(
                             bool((cp.asarray(getattr(pml, f"{s}_{a}{suf}")) == 1.0).all())
                             for s in ("kms", "sinv", "kps")
                             for a in "xyz" for suf in ("", "_h"))}
        log(progress, f"fixture {tag} shape={notes['shape']} flags={flags} "
                      f"carries_all={features[tag]['carries_all']} "
                      f"minus0_at_seed={features[tag]['minus0_at_seed']} "
                      f"unit_pml_coefficients="
                      f"{features[tag]['pml_coefficients_all_exactly_one']}")
        del fields, pml
    results["fixtures"] = features

    # (2) THE BINDING'S PREMISE, on the engine's own recurrence, on this device.
    shape = tuple(int(n) for n in features["J3"]["shape"])
    rng = np.random.default_rng(4242)
    field = cp.asarray((rng.uniform(-1, 1, shape)
                        + 1j * rng.uniform(-1, 1, shape)).astype(np.complex64))
    curl = cp.asarray((rng.uniform(-1, 1, shape)
                       + 1j * rng.uniform(-1, 1, shape)).astype(np.complex64))
    plain = field.copy()
    plain -= curl                                        # stepping.py:537
    split = field.copy()
    one = cp.float32(1.0)
    stepping._apply_pml_update(split, curl, one, one, one, one,
                               cp.zeros(shape, dtype=cp.complex64))
    disagreement = int((cp.ascontiguousarray(plain).view(cp.float32).view(cp.uint32)
                        != cp.ascontiguousarray(split).view(cp.float32).view(cp.uint32)
                        ).sum())
    results["degenerate_binding_premise_words_differing"] = disagreement
    log(progress, "degenerate binding premise: unit-coefficient split-field "
                  f"recurrence vs plain subtract = {disagreement} differing words "
                  f"of {2 * int(np.prod(shape))}")
    del field, curl, plain, split

    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    cases = build_cases(probe)
    if wanted:
        cases = [c for c in cases if c["name"] in wanted]
    log(progress, f"groupj: {len(cases)} cases on {results['device']} "
                  f"(cupy {results['cupy']}, triton {results['triton']}, "
                  f"policy {results['subnormal_policy']})")

    for index, case in enumerate(cases, start=1):
        started = time.time()
        try:
            row = run_case(case["name"], case["group"], case["sub_step"],
                           case["prediction"], case["clause"], case["make"],
                           case["array_path"], case["kernel_plan"],
                           cycles=args.cycles)
        except BaseException as exc:  # noqa: BLE001
            row = {"case": case["name"], "group": case["group"],
                   "sub_step": case["sub_step"], "prediction": case["prediction"],
                   "error": f"{type(exc).__name__}: {exc}"[:400],
                   "traceback": traceback.format_exc()[-1500:],
                   "seconds": round(time.time() - started, 2),
                   "matches_prediction": case["prediction"] == "ERROR"}
        if row.get("error") and "matches_prediction" not in row:
            row["matches_prediction"] = case["prediction"] == "ERROR"
        results["cases"].append(row)
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        log(progress,
            f"case {index}/{len(cases)} {row['case']:<38} "
            f"verdict={row.get('verdict', 'ERROR')} "
            f"predicted={row.get('prediction')} "
            f"substep_moved>={row.get('substep_moved_min')} "
            f"kernel_moved>={row.get('kernel_moved_min')} "
            f"minus0>={row.get('signed_zero_words_min')} "
            f"maxdiff={row.get('differing_max')}/{row.get('words_compared')} "
            f"({row.get('seconds')} s)"
            + (f"  ERROR: {row['error']}" if row.get("error") else ""))

    controls = [r for r in results["cases"] if "CONTROL" in r["case"]]
    live = [r for r in controls if r.get("verdict") == "DIVERGENT"]
    results["controls_total"] = len(controls)
    results["controls_diverged"] = len(live)
    # THE ROUND'S OWN VALIDITY GATE. A null subject verdict beside a dead control
    # is not evidence; it is the group (G) failure repeated.
    results["round_valid"] = bool(controls) and len(live) == len(controls)
    results["surprises"] = [r["case"] for r in results["cases"]
                            if r.get("matches_prediction") is False]
    results["errors"] = [r["case"] for r in results["cases"] if r.get("error")]
    results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log(progress, f"--- {len(results['cases'])} cases, "
                  f"{len(results['surprises'])} surprises, "
                  f"{len(results['errors'])} errors, "
                  f"controls {len(live)}/{len(controls)} diverged, "
                  f"round_valid={results['round_valid']} ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
