"""Run the residual round's OWN predicates and OWN plan builders. Nothing patched.

WHY THIS EXISTS SEPARATELY FROM ``probe_residual_group_bodies.py``
-----------------------------------------------------------------
That probe answers a TRIAGE question — "would the certified body have been exact
on a configuration its clause refuses?" — and to ask it, it must put the body on a
configuration the shipped predicate says no to. It does that honestly, with an
``admitting`` context manager that replaces the INCUMBENT coverage function in
front of the INCUMBENT builder.

But the modules this round ships are not those objects. They are NEW predicates
(``nonlinear_run_pml_curl_coverage``, ``nonlinear_run_constitutive_coverage``,
``no_pml_ade_update_p_coverage``, ``folded_dispersive_constitutive_coverage``,
``folded_complex_offdiag_pml_curl_coverage``,
``folded_complex_offdiag_constitutive_coverage``) and NEW builders
(``plan_nonlinear_run_pml_curl``, ``plan_nonlinear_run_constitutive``,
``plan_no_pml_ade_update_p``, ``plan_folded_dispersive_constitutive``,
``plan_folded_complex_offdiag_pml_curl``,
``plan_folded_complex_offdiag_constitutive``). A triage measurement licenses a
CLAIM ABOUT THE BODY; it does not, on its own, license the OBJECT that will do the
admitting, because between the two sit the predicate's clause set and the
builder's array resolution — which is exactly where this campaign's live defects
have been (a stale pointer cached at build time; a plan built over a ``None``).

So every leg here:

* calls the NEW predicate, UNPATCHED, and requires it to ADMIT — a plan of
  ``None`` is a failed leg, not a skip. The admission is therefore executed
  evidence rather than a reading of the clause list;
* builds through the NEW builder, so the arrays the kernel receives are the ones
  the shipped resolution picked;
* carries the same case discipline as the triage probe, by reusing its
  primitives: uint32 over the whole stored inventory, per-cycle vacuity brackets
  on BOTH sides, a live ``+-0`` lattice, and a perturbation that reaches the
  sub-step's sources.

The PREDICATE BATTERY that runs first is the other half. An admission is only
worth anything if the neighbouring configurations are REFUSED, so each family is
also driven over the configurations it must not take — including the incumbent it
is disjoint from, called on the same objects.

Usage (on a CUDA host, from ``the repository root``)::

    PYTHONPATH=. python -u parity/meep_gpu/probe_residual_new_predicates.py \\
        --out <dir> [--cycles 8]
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
from typing import Any, Callable, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_API = _HERE.parents[1]
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu import stepping  # noqa: E402
from meep_gpu.triton_kernels import folded_complex as fcx  # noqa: E402
from meep_gpu.triton_kernels import folded_dispersive_update_e as fdisp  # noqa: E402
from meep_gpu.triton_kernels import no_pml_ade as ade  # noqa: E402
from meep_gpu.triton_kernels import nonlinear_update_e as nl  # noqa: E402

import parity.meep_gpu.probe_residual_group_bodies as bodies  # noqa: E402

try:
    import cupy as cp
except Exception:  # noqa: BLE001 - reported as a skip, never a crash
    cp = None

CYCLES = bodies.CYCLES

#: What a ``None`` plan means HERE. In the triage probe a None plan is a harness
#: fault (its predicate was patched to admit); here nothing is patched, so it is
#: the shipped predicate refusing — a FAILED leg, never a silent skip, because a
#: probe that skipped refusals would report "all legs identical" over an empty set.
REFUSED_MESSAGE = ("the NEW predicate REFUSED this configuration: "
                   "no identity claim is licensed for it")


def _reasons(verdict: Any) -> List[str]:
    """The verdict's reasons, minus the backend clause a NumPy host always fires."""
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# The configurations, named once and shared by both legs
# ---------------------------------------------------------------------------

def _configs(xp) -> Dict[str, Callable[[int], Tuple[Any, Any, Dict[str, Any]]]]:
    """Every fixture, as ``offset -> (fields, pml, notes)``.

    Named rather than inlined because the predicate battery and the identity legs
    must run on the SAME configurations: a predicate proved to admit one shape and
    a body measured on another is the gap this file was written to close.
    """
    return {
        # (A) chi2/chi3 live; curls and update_H.
        "chi": lambda o: bodies.build(xp, nonlinear=True, seed_offset=o),
        "linear": lambda o: bodies.build(xp, seed_offset=o),
        # (B/H) a pole with NO absorber, plain and conductive.
        "ade_no_pml": lambda o: bodies.build(xp, poles=1, pml_thickness=0,
                                             seed_offset=o),
        "ade_no_pml_conductive": lambda o: bodies.build(
            xp, poles=1, conductivity=0.7, magnetic_conductivity=0.5,
            pml_thickness=0, seed_offset=o),
        "ade_pml": lambda o: bodies.build(xp, poles=1, pml_thickness=2,
                                          seed_offset=o),
        # (C) fold + pole at update_E, in the round's 3-D fixture and in the shape
        #     of the rows it is claimed for.
        "folded_dispersive": lambda o: bodies.build(
            xp, cell=(1.6, 1.6, 1.6), symmetry=("X",), poles=1, pml_thickness=2,
            seed_offset=o),
        "folded_dispersive_2d": lambda o: bodies.build(
            xp, cell=(2.4, 2.4, 0.0), dimensions=2,
            boundaries={"x": "metallic", "y": "metallic", "z": "periodic"},
            symmetry=("Y",), poles=1, pml_thickness=2, seed_offset=o),
        "folded_no_pole": lambda o: bodies.build(
            xp, cell=(1.6, 1.6, 1.6), symmetry=("X",), pml_thickness=2,
            seed_offset=o),
        "unfolded_dispersive": lambda o: bodies.build(
            xp, poles=1, pml_thickness=2, seed_offset=o),
        "folded_offdiag_dispersive": lambda o: bodies.build(
            xp, cell=(1.6, 1.6, 1.6), symmetry=("X",), offdiag=True, poles=1,
            pml_thickness=2, seed_offset=o),
        # (D) complex + fold + off-diagonal, and the two corpus row shapes.
        "complex_fold_offdiag": lambda o: bodies.build(
            xp, cell=(1.6, 1.6, 1.6), symmetry=("X",), complex_storage=True,
            offdiag=True, pml_thickness=2, seed_offset=o),
        "complex_fold_offdiag_twofold": lambda o: bodies.build(
            xp, cell=(2.4, 2.4, 0.0), dimensions=2,
            boundaries={"x": "metallic", "y": "metallic", "z": "periodic"},
            symmetry=("X", "Y"), complex_storage=True, offdiag=True,
            pml_thickness=2, seed_offset=o),
        "complex_fold_offdiag_bloch": lambda o: bodies.build(
            xp, cell=(2.4, 2.4, 0.0), dimensions=2, symmetry=("Y",),
            boundaries="periodic", k_point=(3.5, 0.0, 0.0), complex_storage=True,
            offdiag=True, pml_thickness=2, seed_offset=o),
        "complex_fold_no_offdiag": lambda o: bodies.build(
            xp, cell=(1.6, 1.6, 1.6), symmetry=("X",), complex_storage=True,
            pml_thickness=2, seed_offset=o),
    }


# ---------------------------------------------------------------------------
# LEG 1 — the predicate battery
# ---------------------------------------------------------------------------

def predicate_checks(xp, probe_record) -> List[Dict[str, Any]]:
    """Every new predicate, on the configuration it takes and the ones it must not.

    Each row is (label, expected admission, thunk). The thunk returns a Coverage.
    A raise is recorded as its own outcome and counts as a mismatch: a predicate
    that cannot answer has not refused.
    """
    from meep_gpu.triton_kernels import coverage as cov
    from meep_gpu.triton_kernels import dispersive_update_e as disp
    from meep_gpu.triton_kernels import symmetry as sym

    configs = _configs(xp)
    made: Dict[str, Any] = {}

    def cfg(name):
        if name not in made:
            made[name] = configs[name](0)
        return made[name][0], made[name][1]

    checks: List[Tuple[str, bool, Callable[[], Any]]] = []

    # ---- (A) nonlinear run: curls + update_H ------------------------------
    for sub_step in ("step_B", "step_D"):
        checks.append((
            f"nonlinear_run_pml_curl({sub_step}) ADMITS a chi run", True,
            lambda s=sub_step: nl.nonlinear_run_pml_curl_coverage(*cfg("chi"), s)))
        checks.append((
            f"INCUMBENT pml_curl_coverage({sub_step}) REFUSES it", False,
            lambda s=sub_step: cov.pml_curl_coverage(*cfg("chi"), s)))
    checks.append((
        "nonlinear_run_constitutive(H) ADMITS a chi run", True,
        lambda: nl.nonlinear_run_constitutive_coverage(*cfg("chi"), "H")))
    checks.append((
        "nonlinear_run_constitutive(E) REFUSES it (the Pade sub-step)", False,
        lambda: nl.nonlinear_run_constitutive_coverage(*cfg("chi"), "E")))
    checks.append((
        "INCUMBENT constitutive_coverage(H) REFUSES it", False,
        lambda: cov.constitutive_coverage(*cfg("chi"), "H")))
    # Disjointness, both directions.
    checks.append((
        "nonlinear_run_pml_curl REFUSES a LINEAR run", False,
        lambda: nl.nonlinear_run_pml_curl_coverage(*cfg("linear"), "step_B")))
    checks.append((
        "INCUMBENT pml_curl_coverage ADMITS the linear run", True,
        lambda: cov.pml_curl_coverage(*cfg("linear"), "step_B")))
    # The narrowing this round added: an unallocated curl TARGET must be refused.
    checks.append((
        "nonlinear_run_pml_curl REFUSES an unallocated curl target", False,
        lambda: _with_missing_target("chi", configs, "Bx", "step_B")))

    # ---- (B/H) ADE with no absorber ---------------------------------------
    def _ade(config_name, expected_component="Ez"):
        fields, pml = cfg(config_name)
        state = fields.polarizations[0]
        return ade.no_pml_ade_update_p_coverage(fields, pml, state,
                                                expected_component)

    checks.append((
        "no_pml_ade ADMITS the absorber-free dispersive run", True,
        lambda: _ade("ade_no_pml")))
    checks.append((
        "no_pml_ade ADMITS the CONDUCTIVE absorber-free run", True,
        lambda: _ade("ade_no_pml_conductive")))
    checks.append((
        "no_pml_ade REFUSES an ACTIVE layer (the control's configuration)", False,
        lambda: _ade("ade_pml")))
    checks.append((
        "INCUMBENT ade_update_p_coverage ADMITS the active layer", True,
        lambda: _incumbent_ade(cfg("ade_pml"))))
    checks.append((
        "INCUMBENT ade_update_p_coverage REFUSES the absorber-free run", False,
        lambda: _incumbent_ade(cfg("ade_no_pml"))))

    # ---- (C) fold + dispersion at update_E --------------------------------
    checks.append((
        "folded_dispersive ADMITS fold+pole update_E", True,
        lambda: fdisp.folded_dispersive_constitutive_coverage(
            *cfg("folded_dispersive"))))
    checks.append((
        "folded_dispersive ADMITS the 2-D ROWSHAPE fold+pole", True,
        lambda: fdisp.folded_dispersive_constitutive_coverage(
            *cfg("folded_dispersive_2d"))))
    checks.append((
        "INCUMBENT folded_constitutive_coverage(E) REFUSES it", False,
        lambda: sym.folded_constitutive_coverage(*cfg("folded_dispersive"), "E")))
    checks.append((
        "INCUMBENT dispersive_constitutive_coverage REFUSES it", False,
        lambda: disp.dispersive_constitutive_coverage(*cfg("folded_dispersive"))))
    checks.append((
        "folded_dispersive REFUSES a POLE-FREE fold (disjointness)", False,
        lambda: fdisp.folded_dispersive_constitutive_coverage(
            *cfg("folded_no_pole"))))
    checks.append((
        "folded_dispersive REFUSES an UNFOLDED dispersive run (disjointness)", False,
        lambda: fdisp.folded_dispersive_constitutive_coverage(
            *cfg("unfolded_dispersive"))))
    checks.append((
        "folded_dispersive REFUSES the DIVERGENT offdiag composition (group G)", False,
        lambda: fdisp.folded_dispersive_constitutive_coverage(
            *cfg("folded_offdiag_dispersive"))))

    # ---- (D) complex + fold + off-diagonal --------------------------------
    for name, tag in (("complex_fold_offdiag", ""),
                      ("complex_fold_offdiag_twofold", " ROWSHAPE TWOFOLD"),
                      ("complex_fold_offdiag_bloch", " ROWSHAPE BLOCH")):
        for sub_step in ("step_B", "step_D"):
            checks.append((
                f"folded_complex_offdiag curl({sub_step}) ADMITS{tag}", True,
                lambda n=name, s=sub_step: fcx.folded_complex_offdiag_pml_curl_coverage(
                    *cfg(n), s, probe=probe_record)))
        checks.append((
            f"folded_complex_offdiag constitutive(H) ADMITS{tag}", True,
            lambda n=name: fcx.folded_complex_offdiag_constitutive_coverage(
                *cfg(n), "H", probe=probe_record)))
    checks.append((
        "folded_complex_offdiag constitutive(E) REFUSES (the row product)", False,
        lambda: fcx.folded_complex_offdiag_constitutive_coverage(
            *cfg("complex_fold_offdiag"), "E", probe=probe_record)))
    checks.append((
        "INCUMBENT folded_complex_pml_curl_coverage REFUSES the offdiag run", False,
        lambda: fcx.folded_complex_pml_curl_coverage(
            *cfg("complex_fold_offdiag"), "step_B", probe=probe_record)))
    checks.append((
        "folded_complex_offdiag curl REFUSES a run with NO offdiag row", False,
        lambda: fcx.folded_complex_offdiag_pml_curl_coverage(
            *cfg("complex_fold_no_offdiag"), "step_B", probe=probe_record)))

    rows: List[Dict[str, Any]] = []
    for label, expected, thunk in checks:
        row: Dict[str, Any] = {"check": label, "expected_admit": expected}
        try:
            verdict = thunk()
            row["admit"] = bool(verdict.covered)
            row["reasons"] = _reasons(verdict)
            # A NumPy-backend refusal is the host's, not the predicate's: the
            # admission question is asked with that one clause discounted.
            row["admit_modulo_backend"] = not row["reasons"]
        except BaseException as exc:  # noqa: BLE001
            row["admit"] = None
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            row["traceback"] = traceback.format_exc()[-800:]
        row["matches"] = row.get("admit_modulo_backend") == expected
        rows.append(row)
    return rows


def _with_missing_target(config_name, configs, target, sub_step):
    """The same configuration with one curl target unallocated."""
    fields, pml, _ = configs[config_name](0)
    setattr(fields, target, None)
    return nl.nonlinear_run_pml_curl_coverage(fields, pml, sub_step)


def _incumbent_ade(pair):
    """``coverage.ade_update_p_coverage`` on this configuration's first state.

    Its signature is ``(fields, state, component)`` — no ``pml``, because the
    incumbent reads the drive field's EXISTENCE off ``fields`` alone, which is
    precisely the clause ``no_pml_ade`` inverts.
    """
    from meep_gpu.triton_kernels import coverage as cov

    fields, _pml = pair
    return cov.ade_update_p_coverage(fields, fields.polarizations[0], "Ez")


# ---------------------------------------------------------------------------
# LEG 2 — identity legs through the NEW builders, predicates UNPATCHED
# ---------------------------------------------------------------------------

def _new_ade_plan(fields, pml):
    """Every state's certified plan, through THIS ROUND's builder, or None.

    ``plan_no_pml_ade_update_p`` is per state and takes the drive as a callable at
    run time, so the composite mirrors what a dispatcher would install.
    """
    states = tuple(getattr(fields, "polarizations", ()) or ())
    plans = [ade.plan_no_pml_ade_update_p(fields, pml, state) for state in states]
    if not plans or any(plan is None for plan in plans):
        return None
    drive = fields.drive_field

    class _Composite:
        def run(self, guard=None):
            for plan in plans:
                plan.run(drive, guard)

        def __repr__(self):
            return f"NewAdeComposite({[repr(p) for p in plans]})"

    return _Composite()


def build_cases(xp, probe_record) -> List[Dict[str, Any]]:
    configs = _configs(xp)
    cases: List[Dict[str, Any]] = []

    # (A) — the two curls and update_H of a nonlinear run.
    for sub_step, fn in (("step_B", stepping.step_B), ("step_D", stepping.step_D)):
        cases.append({
            "name": f"NEW_nonlinear_run_{sub_step}", "group": "A",
            "sub_step": sub_step, "prediction": "IDENTICAL",
            "builder": "nonlinear_update_e.plan_nonlinear_run_pml_curl",
            "make": configs["chi"], "array_path": fn,
            "kernel_plan": (lambda f, p, s=sub_step:
                            nl.plan_nonlinear_run_pml_curl(f, p, s)),
        })
    cases.append({
        "name": "NEW_nonlinear_run_update_H", "group": "A", "sub_step": "update_H",
        "prediction": "IDENTICAL",
        "builder": "nonlinear_update_e.plan_nonlinear_run_constitutive",
        "make": configs["chi"], "array_path": stepping.update_H,
        "kernel_plan": (lambda f, p: nl.plan_nonlinear_run_constitutive(f, p, "H")),
    })

    # (B/H) — update_P with no absorber, plain and conductive.
    for tag, config_name in (("", "ade_no_pml"),
                             ("_conductive", "ade_no_pml_conductive")):
        cases.append({
            "name": f"NEW_no_pml_ade_update_P{tag}", "group": "B/H",
            "sub_step": "update_P", "prediction": "IDENTICAL",
            "builder": "no_pml_ade.plan_no_pml_ade_update_p",
            "make": configs[config_name], "array_path": stepping.update_P,
            "kernel_plan": _new_ade_plan,
        })

    # (C) — folded dispersive update_E, in both shapes.
    for tag, config_name in (("", "folded_dispersive"),
                             ("_ROWSHAPE_2D", "folded_dispersive_2d")):
        cases.append({
            "name": f"NEW_folded_dispersive_update_E{tag}", "group": "C",
            "sub_step": "update_E", "prediction": "IDENTICAL",
            "builder": ("folded_dispersive_update_e."
                        "plan_folded_dispersive_constitutive"),
            "make": configs[config_name], "array_path": stepping.update_E,
            "kernel_plan": (lambda f, p:
                            fdisp.plan_folded_dispersive_constitutive(f, p)),
        })

    # (D) — complex + fold + off-diagonal, in all three shapes.
    for tag, config_name in (("", "complex_fold_offdiag"),
                             ("_ROWSHAPE_TWOFOLD", "complex_fold_offdiag_twofold"),
                             ("_ROWSHAPE_BLOCH", "complex_fold_offdiag_bloch")):
        for sub_step, fn in (("step_B", stepping.step_B),
                             ("step_D", stepping.step_D)):
            cases.append({
                "name": f"NEW_folded_complex_offdiag{tag}_{sub_step}", "group": "D",
                "sub_step": sub_step, "prediction": "IDENTICAL",
                "builder": "folded_complex.plan_folded_complex_offdiag_pml_curl",
                "make": configs[config_name], "array_path": fn,
                "kernel_plan": (lambda f, p, s=sub_step:
                                fcx.plan_folded_complex_offdiag_pml_curl(
                                    f, p, s, probe=probe_record)),
            })
        cases.append({
            "name": f"NEW_folded_complex_offdiag{tag}_update_H", "group": "D",
            "sub_step": "update_H", "prediction": "IDENTICAL",
            "builder": "folded_complex.plan_folded_complex_offdiag_constitutive",
            "make": configs[config_name], "array_path": stepping.update_H,
            "kernel_plan": (lambda f, p:
                            fcx.plan_folded_complex_offdiag_constitutive(
                                f, p, "H", probe=probe_record)),
        })

    return cases


# ---------------------------------------------------------------------------

def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--cycles", type=int, default=CYCLES)
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "new_predicates.progress.log"
    results_path = out / "new_predicates.json"

    results: Dict[str, Any] = {"cycles": args.cycles, "predicate_checks": [],
                               "cases": []}
    if cp is None:
        results["skipped"] = "cupy is not importable on this host"
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        bodies.log(progress, "SKIPPED: cupy is not importable on this host")
        return 0
    try:
        import triton  # noqa: F401
        results["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        results["skipped"] = f"triton is not importable: {exc}"
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        bodies.log(progress, f"SKIPPED: triton is not importable: {exc}")
        return 0

    results["cupy"] = cp.__version__
    results["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    probe_record = bodies._load_expansion_probe()
    results["expansion_probe_loaded"] = probe_record is not None

    # LEG 1 first: an identity leg is only interesting if the predicate that gates
    # it admits for the stated reason and refuses its neighbours.
    for row in predicate_checks(cp, probe_record):
        results["predicate_checks"].append(row)
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        bodies.log(progress,
                   f"PRED {'ok ' if row['matches'] else 'BAD'} {row['check']:<62} "
                   f"admit={row.get('admit_modulo_backend')} "
                   f"expect={row['expected_admit']} "
                   + (row.get("error") or json.dumps(row.get("reasons", []))[:220]))

    cases = build_cases(cp, probe_record)
    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    if wanted:
        cases = [c for c in cases if c["name"] in wanted]
    bodies.log(progress, f"new-predicate legs: {len(cases)} on {results['device']} "
                         f"(cupy {results['cupy']}, triton {results['triton']})")

    for index, case in enumerate(cases, start=1):
        started = time.time()
        try:
            row = bodies.run_case(
                case["name"], case["group"], case["sub_step"], case["prediction"],
                f"NEW BUILDER, predicate unpatched: {case['builder']}",
                case["make"], case["array_path"], case["kernel_plan"],
                cycles=args.cycles)
        except BaseException as exc:  # noqa: BLE001
            row = {"case": case["name"], "group": case["group"],
                   "sub_step": case["sub_step"], "prediction": case["prediction"],
                   "error": f"{type(exc).__name__}: {exc}"[:400],
                   "traceback": traceback.format_exc()[-1500:],
                   "matches_prediction": False,
                   "seconds": round(time.time() - started, 2)}
        row["builder"] = case["builder"]
        # run_case reports a None plan as "the shipped builder returned None even
        # with its predicate admitting". Here nothing is patched, so a None plan
        # means the NEW PREDICATE REFUSED — a failed leg, and the message must say
        # which of the two it is.
        if row.get("error", "").startswith("the shipped builder returned None"):
            row["error"] = REFUSED_MESSAGE
        results["cases"].append(row)
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        bodies.log(progress,
                   f"leg {index}/{len(cases)} {row['case']:<48} "
                   f"verdict={row.get('verdict', 'ERROR')} "
                   f"substep>={row.get('substep_moved_min')} "
                   f"kernel>={row.get('kernel_moved_min')} "
                   f"minus0>={row.get('signed_zero_words_min')} "
                   f"maxdiff={row.get('differing_max')}/{row.get('words_compared')} "
                   f"({row.get('seconds')} s)"
                   + (f"  ERROR: {row['error']}" if row.get("error") else ""))

    bad_predicates = [r["check"] for r in results["predicate_checks"]
                      if not r["matches"]]
    bad_legs = [r["case"] for r in results["cases"]
                if r.get("verdict") != "IDENTICAL"]
    results["predicate_mismatches"] = bad_predicates
    results["legs_not_identical"] = bad_legs
    results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    bodies.log(progress,
               f"--- {len(results['predicate_checks'])} predicate checks "
               f"({len(bad_predicates)} unexpected: {bad_predicates}); "
               f"{len(results['cases'])} identity legs "
               f"({len(bad_legs)} not IDENTICAL: {bad_legs}) ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
