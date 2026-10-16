"""CUDA byte gate for the complex64 no-absorber ADE ``update_P`` family.

The gate compares the shipped predicate and builder against ``stepping.update_P`` as
uint32 over the complete field and polarization inventory for eight cycles.

The four Group-I corpus rows are represented explicitly.  They share one
numerical configuration in the lifted census — shape ``(35, 32, 41)``, Bloch
``k=(0.4,-1.3,0.7)``, complex storage, electric and magnetic conductivity, one
Lorentz pole, and an inert ``mp.Absorber`` — but are kept as four named cases so
the released artifact maps one-to-one back to the corpus.  Additional products
exercise a volume sigma, Drude, a driven-component subset, and two poles.

Four word mutations (one per MEEP realnum word), the package's previous folded
three-term form planted back into the kernel (the inversion of KERNELS.md section 7.1),
and a wrong-drive mutation must diverge.  Predicate
refusals cover the adjacent real, active-PML and unstored-E classes.
Every case emits a flushed progress line and an fsync'd JSONL row as it lands.

Run only on one confirmed-idle
GPU, with private policy-labelled CuPy and Triton caches::

    python -u gate_triton_complex_ade.py --out <results-dir>/gate.json --cycles 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
API = HERE.parents[1]
for path in (str(HERE), str(API)):
    if path not in sys.path:
        sys.path.insert(0, path)

import probe_residual_group_bodies as P  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

CYCLES = 8
GROUP_I_ROWS = (
    "TestLoadDump.test_load_dump_chunk_layout_file_3d",
    "TestLoadDump.test_load_dump_chunk_layout_sim_3d",
    "TestLoadDump.test_load_dump_structure_3d",
    "TestLoadDump.test_load_dump_structure_sharded_3d",
)


def log(handle, message: str) -> None:
    print(message, flush=True)
    handle.write(message + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def emit(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def save(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=1, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_configuration(xp: Any, *, seed_offset: int = 0,
                        kind: str = "lorentzian", poles: int = 1,
                        volume_sigma: bool = False,
                        driven: Sequence[str] = ("Ex", "Ey", "Ez"),
                        active_pml: bool = False,
                        complex_storage: bool = True):
    """The exact lifted Group-I grid, with optional capability variations."""
    grid = Grid(
        resolution=15.0, cell_size=(2.3, 2.1, 2.7), dimensions=3,
        boundaries="periodic", courant=0.5,
        k_point=(0.4, -1.3, 0.7), xp=xp)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    shape = tuple(grid.shape)
    states = []
    for index in range(poles):
        sigma: Dict[str, Any] = {}
        for axis, component in enumerate(("Ex", "Ey", "Ez")):
            if component not in driven:
                sigma[component] = 0.0
            elif volume_sigma:
                host = np.linspace(
                    0.12 + 0.02 * index + 0.01 * axis,
                    0.42 + 0.02 * index + 0.01 * axis,
                    np.prod(shape), dtype=np.float32).reshape(shape)
                sigma[component] = xp.asarray(host)
            else:
                sigma[component] = 0.30 + 0.05 * index + 0.01 * axis
        state = PolarizationState(
            Susceptibility(frequency=1.0 + 0.2 * index, gamma=0.1, kind=kind),
            sigma, grid, np.complex64 if complex_storage else np.float32)
        fields.polarizations.append(state)
        states.append(state)

    gradient = np.linspace(0.55, 1.45, shape[0], dtype=np.float32)[:, None, None]
    fields.set_d_conductivity(xp.asarray(
        np.broadcast_to(np.float32(0.7) * gradient, shape).copy()))
    fields.set_b_conductivity(xp.asarray(
        np.broadcast_to(np.float32(0.5) * gradient, shape).copy()))
    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2 if active_pml else 0)
    rng = np.random.default_rng(P.SEED + seed_offset)
    P.seed_state(fields, xp, rng, amplitude=0.4)
    notes = {
        "shape": list(shape), "kind": kind, "poles": poles,
        "volume_sigma": volume_sigma, "driven": list(driven),
        "complex_storage": complex_storage,
        "pml_active": bool(pml.is_active),
        "k_point": list(grid.k_point),
        "conductivity": True, "magnetic_conductivity": True,
    }
    return fields, pml, notes


class StatePlans:
    """All susceptibility plans that together replace one ``update_P`` call."""

    def __init__(self, plans: Sequence[Any], fields: Any,
                 *, coefficient_mutation: Optional[int] = None,
                 wrong_drive: bool = False) -> None:
        # ``coefficient_mutation`` zeroes one of MEEP's four update_P words
        # (PolarizationState.constants) before every launch.
        self.plans = tuple(plans)
        self.fields = fields
        self.coefficient_mutation = coefficient_mutation
        self.wrong_drive = wrong_drive

    def run(self, guard=None) -> None:
        if self.coefficient_mutation is not None:
            for plan in self.plans:
                words = list(plan.state.constants)
                words[self.coefficient_mutation] = 0.0
                plan.state.constants = tuple(words)
        if self.wrong_drive:
            def drive(component):
                return getattr(self.fields, "D" + component[1])
        else:
            drive = self.fields.drive_field
        for plan in self.plans:
            plan.run(drive, guard)

    def __repr__(self) -> str:
        return f"StatePlans({self.plans!r})"


def build_plans(fields: Any, pml: Any, *, coefficient_mutation: Optional[int] = None,
                wrong_drive: bool = False, kernel: Any = None):
    from meep_gpu.triton_kernels import complex_ade as arm

    plans = []
    for state in fields.polarizations:
        plan = arm.plan_complex_ade_update_p(
            fields, pml, state)
        if plan is not None and kernel is not None:
            # The mutant through the plan's own kernel route: the predicate and the
            # builder stay the shipped ones.
            plan = arm.ComplexAdeUpdatePPlan(
                state, plan.components, plan.shape, plan.block, kernel=kernel)
        if plan is None:
            reasons = {
                component: list(arm.complex_ade_update_p_coverage(
                    fields, pml, state, component).reasons)
                for component in state.driven()
            }
            raise RuntimeError(f"shipped complex ADE builder refused: {reasons}")
        plans.append(plan)
    return StatePlans(plans, fields,
                      coefficient_mutation=coefficient_mutation,
                      wrong_drive=wrong_drive)


def make_factory(xp: Any, **config):
    return lambda offset: build_configuration(xp, seed_offset=offset, **config)


def case(name: str, prediction: str, make, builder, clause: str) -> Dict[str, Any]:
    return {
        "name": name, "group": "I", "sub_step": "update_P",
        "prediction": prediction, "clause": clause,
        "make": make, "array_path": stepping.update_P,
        "kernel_plan": builder,
    }


def build_cases(xp: Any) -> List[Dict[str, Any]]:
    target = make_factory(xp)
    rows = [case(
        "product_" + row.replace(".", "__"), "IDENTICAL", target,
        lambda f, p: build_plans(f, p),
        "one-to-one lifted Group-I corpus row") for row in GROUP_I_ROWS]
    rows.extend((
        case("product_volume_sigma", "IDENTICAL",
             make_factory(xp, volume_sigma=True),
             lambda f, p: build_plans(f, p),
             "f4 volume sigma left of complex W"),
        case("product_drude_subset", "IDENTICAL",
             make_factory(xp, kind="drude", driven=("Ex", "Ez")),
             lambda f, p: build_plans(f, p),
             "Drude recurrence and canonical driven subset"),
        case("product_two_poles", "IDENTICAL",
             make_factory(xp, poles=2),
             lambda f, p: build_plans(f, p),
             "two susceptibility plans in registration order"),
    ))
    for index, name in enumerate(("two_minus_od", "gamma1", "omega0dtsqr",
                                  "gamma1inv")):
        rows.append(case(
            f"MUTATION_drop_{name}", "DIVERGENT", target,
            lambda f, p, i=index: build_plans(
                f, p, coefficient_mutation=i),
            f"mutation: drop {name}"))
    rows.append(case(
        "MUTATION_previous_folded_three_term_form", "DIVERGENT", target,
        lambda f, p: build_plans(
            f, p, kernel=previous_folded_form_kernel()),
        "mutation (inverted): the package's previous folded three-term form"))
    rows.append(case(
        "MUTATION_drive_from_D_instead_of_stored_E", "DIVERGENT", target,
        lambda f, p: build_plans(
            f, p, wrong_drive=True),
        "mutation: bind D instead of Fields.drive_field"))
    return rows


#: The package's previous spelling, per part, planted back: gamma1inv folded into three
#: coefficients and the history ADDED with a negative coefficient. The same recurrence
#: in exact arithmetic and a different float32 number.
PREVIOUS_FORM = (
    ("""    a_re, a_im = _mul_field_left(p_re, p_im, two_minus_od)
    b_re, b_im = _mul_coefficient_left(gamma1, q_re, q_im)
    out_re = a_re - b_re
    out_im = a_im - b_im""",
     """    a_re, a_im = _mul_field_left(p_re, p_im, gamma1inv * two_minus_od)
    b_re, b_im = _mul_coefficient_left(0.0 - (gamma1inv * gamma1), q_re, q_im)
    out_re = a_re + b_re
    out_im = a_im + b_im"""),
    ("""    d_re, d_im = _mul_coefficient_left(omega0dtsqr, sw_re, sw_im)
    out_re = out_re + d_re
    out_im = out_im + d_im
    out_re, out_im = _mul_coefficient_left(gamma1inv, out_re, out_im)""",
     """    d_re, d_im = _mul_coefficient_left(gamma1inv * omega0dtsqr, sw_re, sw_im)
    out_re = out_re + d_re
    out_im = out_im + d_im"""),
)


def previous_folded_form_kernel() -> Any:
    """The shipped kernel's text with :data:`PREVIOUS_FORM` applied, compiled renamed."""
    import importlib.util  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    from meep_gpu.triton_kernels import complex_ade as arm  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(arm.complex_ade_update_p.fn))
    for old, new in PREVIOUS_FORM:
        if source.count(old) != 1:
            raise RuntimeError(f"the previous-form needle matched {source.count(old)} "
                               f"sites; the mutation was never armed")
        source = source.replace(old, new)
    source = source.replace("def complex_ade_update_p(",
                            "def mutant_complex_ade_previous_form(")
    header = ("import triton\nimport triton.language as tl\n"
              "from meep_gpu.triton_kernels.complex_fields import (\n"
              "    _mul_coefficient_left, _mul_field_left)\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_complex_ade_mutant.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    spec = importlib.util.spec_from_file_location("complex_ade_previous_form",
                                                  handle.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.mutant_complex_ade_previous_form


def refusal_leg(xp: Any) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import complex_ade as arm

    rows = []
    fixtures = (
        ("target", {}, True),
        ("active_pml", {"active_pml": True}, False),
        ("real_storage", {"complex_storage": False}, False),
    )
    for name, config, expected in fixtures:
        fields, pml, _ = build_configuration(xp, **config)
        state = fields.polarizations[0]
        verdict = arm.complex_ade_update_p_coverage(
            fields, pml, state, state.driven()[0])
        rows.append({"case": name, "covered": verdict.covered,
                     "expected": expected,
                     "agrees": verdict.covered == expected,
                     "reasons": list(verdict.reasons)})

    fields, pml, _ = build_configuration(xp)
    fields._stored_E = False
    state = fields.polarizations[0]
    verdict = arm.complex_ade_update_p_coverage(
        fields, pml, state, "Ex")
    rows.append({"case": "unstored_E", "covered": verdict.covered,
                 "expected": False, "agrees": not verdict.covered,
                 "reasons": list(verdict.reasons)})
    return {"cases": rows, "all_agree": all(row["agrees"] for row in rows)}


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rows = payload.get("cases") or []
    products = [row for row in rows if str(row.get("case", "")).startswith("product_")]
    mutations = [row for row in rows if "MUTATION" in str(row.get("case", ""))]
    if len(products) != 7:
        reasons.append(f"expected 7 products, found {len(products)}")
    if len(mutations) != 6:
        reasons.append(f"expected 6 mutations, found {len(mutations)}")
    for row in rows:
        if row.get("error"):
            reasons.append(f"{row.get('case')}: {row['error']}")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(
                f"{row.get('case')}: {row.get('verdict')} != {row.get('prediction')}")
        if row.get("vacuous"):
            reasons.append(f"{row.get('case')}: vacuous")
    if not (payload.get("refusals") or {}).get("all_agree"):
        reasons.append("a predicate refusal disagreed with its expectation")
    return {"released": not reasons, "reasons": reasons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cycles", type=int, default=CYCLES)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_log = (out.parent / "driver.log").open("a", encoding="utf-8")
    jsonl = (out.parent / "cases.jsonl").open("a", encoding="utf-8")
    started = time.time()
    payload: Dict[str, Any] = {
        "gate": "triton_complex_ade", "cycles": args.cycles,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "group_i_rows": list(GROUP_I_ROWS),
    }
    try:
        import cupy as cp
        import composition_policy
        import gate_triton_complex as G
        from meep_gpu import fastpath

        # ONE SUBNORMAL-POLICY AUTHORITY: THE CAMPAIGN'S. The certification round gates
        # this family under keep AND under flush (R13), and the fleet driver names the
        # campaign's policy in MEEP_GPU_SUBNORMAL_POLICY, which composition_policy reads
        # (keep or flush; anything else refused by name). Unset, it is the table's own
        # install default, so a bare run still certifies what dispatch installs -- and
        # the two may not drift apart. (The bare no-argument install_ftz_strip call
        # resolves MATCH_MEEP by measuring the host, which is 'flush' on x86.) MEASURED:
        # with the install default named here instead, the flush campaign's run refused
        # on its own flush cache token before the first product (FPC, 0 of 7).
        if composition_policy.DEFAULT_POLICY != fastpath.TABLE_SUBNORMAL_POLICY["triton"]:
            raise RuntimeError(
                f"composition_policy.DEFAULT_POLICY={composition_policy.DEFAULT_POLICY!r} "
                f"is not the Triton install default "
                f"{fastpath.TABLE_SUBNORMAL_POLICY['triton']!r}")
        policy = composition_policy.requested_policy()
        strip = G.install_ftz_strip(policy)
        payload["subnormal_policy_installed"] = policy
        payload["subnormal_policy_source"] = (
            "MEEP_GPU_SUBNORMAL_POLICY (composition_policy.requested_policy); unset reads "
            'the install default fastpath.TABLE_SUBNORMAL_POLICY["triton"]')
        payload["subnormal_policy_stamp"] = strip
        cases = build_cases(cp)
        rows = []
        for index, spec in enumerate(cases, 1):
            log(run_log, f"case {index}/{len(cases)} {spec['name']} "
                         f"predict={spec['prediction']}")
            try:
                row = P.run_case(cycles=args.cycles, **spec)
            except BaseException as exc:  # noqa: BLE001 - record and continue
                row = {"case": spec["name"], "prediction": spec["prediction"],
                       "verdict": "ERROR",
                       "error": f"{type(exc).__name__}: {exc}"[:500],
                       "traceback": traceback.format_exc()[-1800:]}
            rows.append(row)
            emit(jsonl, row)
            log(run_log, f"  -> {row.get('verdict', 'ERROR')} "
                         f"differing_max={row.get('differing_max', '?')} "
                         f"elapsed={time.time() - started:.1f}s")
            payload["cases"] = rows
            save(out, payload)
        payload["refusals"] = refusal_leg(cp)
    except BaseException as exc:  # noqa: BLE001 - artifact the setup failure
        payload["setup_error"] = f"{type(exc).__name__}: {exc}"
        payload["setup_traceback"] = traceback.format_exc()[-3000:]
        log(run_log, f"SETUP ERROR: {payload['setup_error']}")
    payload["seconds"] = round(time.time() - started, 2)
    payload["release"] = validate(payload)
    if payload.get("setup_error"):
        payload["release"]["released"] = False
        payload["release"]["reasons"].append(payload["setup_error"])
    save(out, payload)
    log(run_log, f"RELEASED={payload['release']['released']}")
    for reason in payload["release"]["reasons"]:
        log(run_log, f"  refused: {reason}")
    jsonl.close()
    run_log.close()
    return 0 if payload["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
