"""CUDA byte gate for complex, pole-aware, no-PML stored-E ``update_E``.

The four Group-I load/dump rows are represented one-to-one on their lifted
``(35, 32, 41)`` Bloch-periodic configuration.  Additional product rows cover
Drude, a driven-component subset, two poles, and a complete update_E/update_P
pair that rotates the P pointers between launches.

The shipped predicate and engine-object builder are used for every product.
Mutations reverse the two-pole order, pre-sum the poles, drop the final pole,
and freeze pointers across ADE rotation; a one-pole reversal is the control.
Every case is flushed to JSONL as it lands and the final payload is written
atomically.

Run only on one confirmed-idle
GPU, with private CuPy/Triton caches carrying the campaign policy's token
(MEEP_GPU_SUBNORMAL_POLICY, keep when unset)::

    python -u gate_triton_complex_no_pml_stored_e.py \
        --out <results-dir>/gate.json --cycles 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

HERE = Path(__file__).resolve().parent
API = HERE.parents[1]
for path in (str(HERE), str(API)):
    if path not in sys.path:
        sys.path.insert(0, path)

import gate_triton_complex_ade as ADE  # noqa: E402
import probe_residual_group_bodies as P  # noqa: E402
from meep_gpu import stepping  # noqa: E402

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


def make_factory(xp: Any, **config):
    return lambda offset: ADE.build_configuration(
        xp, seed_offset=offset, **config)


def shipped_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels import complex_no_pml_stored_e as arm

    plan = arm.plan_complex_stored_e(fields, pml)
    if plan is None:
        verdict = arm.complex_stored_e_coverage(fields, pml)
        raise RuntimeError(
            f"shipped complex stored-E builder refused: {verdict.reasons}")
    return plan


class EThenPPlan:
    """Run update_E, then rotate ADE state as one complete electric pair."""

    def __init__(self, inner: Any, fields: Any, pml: Any) -> None:
        self.inner = inner
        self.fields = fields
        self.pml = pml

    def run(self, guard=None) -> None:
        self.inner.run(guard)
        stepping.update_P(self.fields, self.pml)

    def __repr__(self) -> str:
        return f"EThenPPlan({self.inner!r})"


def array_e_then_p(fields: Any, pml: Any) -> None:
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


def _live_groups(fields: Any, *, reverse: bool = False,
                 drop: int = 0) -> List[Sequence[Any]]:
    from meep_gpu.triton_kernels import complex_no_pml_stored_e as arm

    order = arm.poles_per_component(fields)
    groups: List[Sequence[Any]] = []
    for component, _ in arm.E_TERMS:
        arrays = [state.P[component] for state in order[component]]
        if reverse:
            arrays.reverse()
        if drop:
            arrays = arrays[:-drop] if drop <= len(arrays) else []
        groups.append(tuple(arrays))
    return groups


class SummedComplexPoleBinding:
    """Mutation: pre-sum live P volumes before the D subtraction."""

    def __init__(self, fields: Any) -> None:
        from meep_gpu.triton_kernels import complex_no_pml_stored_e as arm

        self.fields = fields
        self.order = arm.poles_per_component(fields)
        self.counts = tuple(
            1 if self.order[component] else 0 for component, _ in arm.E_TERMS)
        self._live: Sequence[Sequence[Any]] = ()

    def arrays(self):
        from meep_gpu.triton_kernels import complex_no_pml_stored_e as arm

        groups = []
        for component, _ in arm.E_TERMS:
            states = self.order[component]
            if not states:
                groups.append(())
                continue
            total = states[0].P[component].copy()
            for state in states[1:]:
                total += state.P[component]
            groups.append((total,))
        # Retain the temporary device allocations until at least the next launch.
        self._live = tuple(groups)
        return self._live


def reordered_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels import complex_no_pml_stored_e as arm

    plan = shipped_plan(fields, pml)
    plan._poles = arm.StaticComplexPoleBinding(
        _live_groups(fields, reverse=True))
    return plan


def dropped_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels import complex_no_pml_stored_e as arm

    plan = shipped_plan(fields, pml)
    plan._poles = arm.StaticComplexPoleBinding(_live_groups(fields, drop=1))
    plan.counts = plan._poles.counts
    return plan


def presum_plan(fields: Any, pml: Any):
    plan = shipped_plan(fields, pml)
    plan._poles = SummedComplexPoleBinding(fields)
    plan.counts = plan._poles.counts
    return plan


def frozen_rotation_plan(fields: Any, pml: Any):
    from meep_gpu.triton_kernels import complex_no_pml_stored_e as arm

    plan = shipped_plan(fields, pml)
    plan._poles = arm.StaticComplexPoleBinding(_live_groups(fields))
    return EThenPPlan(plan, fields, pml)


def live_rotation_plan(fields: Any, pml: Any):
    return EThenPPlan(shipped_plan(fields, pml), fields, pml)


def case(name: str, prediction: str, make, builder, clause: str,
         *, rotate: bool = False) -> Dict[str, Any]:
    return {
        "name": name, "group": "I", "sub_step": "update_E",
        "prediction": prediction, "clause": clause, "make": make,
        "array_path": array_e_then_p if rotate else stepping.update_E,
        "kernel_plan": builder,
    }


def build_cases(xp: Any) -> List[Dict[str, Any]]:
    target = make_factory(xp)
    rows = [case(
        "product_" + row.replace(".", "__"), "IDENTICAL", target,
        lambda f, p: shipped_plan(f, p),
        "one-to-one lifted Group-I corpus row") for row in GROUP_I_ROWS]
    rows.extend((
        case("product_drude_subset", "IDENTICAL",
             make_factory(xp, kind="drude", driven=("Ex", "Ez")),
             lambda f, p: shipped_plan(f, p),
             "Drude and a canonical component subset"),
        case("product_two_poles", "IDENTICAL",
             make_factory(xp, poles=2),
             lambda f, p: shipped_plan(f, p),
             "two poles subtracted left-to-right in registration order"),
        case("product_live_pointer_rotation", "IDENTICAL",
             make_factory(xp, poles=2),
             lambda f, p: live_rotation_plan(f, p),
             "live P pointers after update_P rotation", rotate=True),
        case("MUTATION_reverse_pole_order", "DIVERGENT",
             make_factory(xp, poles=2),
             lambda f, p: reordered_plan(f, p),
             "mutation: reverse registration order"),
        case("MUTATION_presum_poles", "DIVERGENT",
             make_factory(xp, poles=2),
             lambda f, p: presum_plan(f, p),
             "mutation: D - (P0 + P1) instead of (D - P0) - P1"),
        case("MUTATION_drop_last_pole", "DIVERGENT",
             make_factory(xp, poles=2),
             lambda f, p: dropped_plan(f, p),
             "mutation: omit final registered pole"),
        case("MUTATION_freeze_rotating_pointers", "DIVERGENT",
             make_factory(xp, poles=2),
             lambda f, p: frozen_rotation_plan(f, p),
             "mutation: cache initial P pointers", rotate=True),
        case("CONTROL_reverse_one_pole", "IDENTICAL", target,
             lambda f, p: reordered_plan(f, p),
             "control: reversing one pole changes nothing"),
    ))
    return rows


def refusal_leg(xp: Any) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import complex_no_pml_stored_e as arm
    from meep_gpu.triton_kernels import no_pml_stored_e as real_arm

    rows = []
    fixtures = (
        ("target_with_conductivity", {}, True),
        ("active_pml", {"active_pml": True}, False),
        ("real_storage", {"complex_storage": False}, False),
        ("no_poles", {"poles": 0}, False),
    )
    for name, config, expected in fixtures:
        fields, pml, _ = ADE.build_configuration(xp, **config)
        verdict = arm.complex_stored_e_coverage(fields, pml)
        real = real_arm.stored_e_constitutive_coverage(fields, pml)
        rows.append({
            "case": name, "covered": verdict.covered, "expected": expected,
            "agrees": verdict.covered == expected,
            "real_arm_covered": real.covered,
            "exclusive": not (verdict.covered and real.covered),
            "reasons": list(verdict.reasons),
        })

    fields, pml, _ = ADE.build_configuration(xp)
    fields._stored_E = False
    verdict = arm.complex_stored_e_coverage(fields, pml)
    rows.append({"case": "unstored_E", "covered": verdict.covered,
                 "expected": False, "agrees": not verdict.covered,
                 "real_arm_covered": False, "exclusive": True,
                 "reasons": list(verdict.reasons)})

    return {
        "cases": rows,
        "all_agree": all(row["agrees"] for row in rows),
        "exclusive": all(row["exclusive"] for row in rows),
    }


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rows = payload.get("cases") or []
    products = [row for row in rows
                if str(row.get("case", "")).startswith("product_")]
    mutations = [row for row in rows
                 if "MUTATION" in str(row.get("case", ""))]
    controls = [row for row in rows
                if "CONTROL" in str(row.get("case", ""))]
    if len(products) != 7:
        reasons.append(f"expected 7 products, found {len(products)}")
    if len(mutations) != 4:
        reasons.append(f"expected 4 mutations, found {len(mutations)}")
    if len(controls) != 1:
        reasons.append(f"expected 1 control, found {len(controls)}")
    for row in rows:
        if row.get("error"):
            reasons.append(f"{row.get('case')}: {row['error']}")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(
                f"{row.get('case')}: {row.get('verdict')} != "
                f"{row.get('prediction')}")
        if row.get("vacuous"):
            reasons.append(f"{row.get('case')}: vacuous")
    # ABSENT IS NOT FAILED. Added 2026-08-17 after this exact confusion cost a
    # reader real time: a setup error aborts before ``refusals`` is ever
    # assigned, and ``payload.get("refusals") or {}`` then made every clause
    # below read False and emit its measured-sounding verdict. A run whose
    # policy install raised therefore announced "the real and complex stored-E
    # predicates overlap" — a specific, checkable, and entirely unmeasured
    # claim, which was read downstream as a disjointness defect and acted on.
    # A gate that cannot say "I did not look" will be believed when it guesses.
    refusals = payload.get("refusals")
    if refusals is None:
        reasons.append("the refusal leg DID NOT RUN, so predicate agreement and "
                       "real/complex exclusivity are UNMEASURED — not failed")
    else:
        if not refusals.get("all_agree"):
            reasons.append("a predicate answer disagreed with its expectation")
        if not refusals.get("exclusive"):
            reasons.append("the real and complex stored-E predicates overlap")
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
        "gate": "triton_complex_no_pml_stored_e", "cycles": args.cycles,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "group_i_rows": list(GROUP_I_ROWS),
    }
    try:
        import cupy as cp
        import composition_policy
        import gate_triton_complex as G
        from meep_gpu import fastpath

        # ONE SUBNORMAL-POLICY AUTHORITY, AND IT IS THIS LINE: THE CAMPAIGN'S.
        #
        # The certification round gates this family under keep AND under flush
        # (R13); the fleet driver names the campaign's policy in
        # MEEP_GPU_SUBNORMAL_POLICY and composition_policy.requested_policy reads it
        # (keep or flush, anything else refused by name). Unset, it is the table's own
        # install default -- asserted equal below, so a bare run cannot drift from
        # what dispatch installs. MEASURED: with the install default named here
        # whatever the campaign asked, the flush campaign's run refused on its own
        # flush cache token before the first product (FPC, 0 of 7).
        #
        # History of this line:
        #
        # Added 2026-08-17, replacing two competing authorities. This gate used
        # to install nothing and call the no-request path, which resolves
        # MATCH_MEEP by measuring the host — 'flush' on x86 — while every
        # complex arm carried a 'keep' certification constant. The
        # certification clause then refused the run, correctly: a kernel's bytes
        # were only ever compared against the array path under the policy its
        # own gate ran in, so nothing establishes what it emits under another.
        # An intermediate fix installed 'keep' here AND left the no-request call
        # below it — two authorities that agreed only when
        # MEEP_GPU_SUBNORMAL_POLICY happened to be exported. A run that is
        # correct only because of an unrecorded environment variable is worse
        # than one that fails.
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
                row = {"case": spec["name"],
                       "prediction": spec["prediction"], "verdict": "ERROR",
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
    except BaseException as exc:  # noqa: BLE001 - artifact setup failure
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
