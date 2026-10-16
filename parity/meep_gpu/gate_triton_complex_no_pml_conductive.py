"""Byte gate for the Triton complex64/Bloch conductive curls without active PML.

The eight primary rows are the four lifted Group-I ``TestLoadDump`` cases at
both ``step_B`` and ``step_D``.  All carry the exact lifted shape and Bloch
point, B and D conductivity, and a live Lorentz pole.  Additional product rows
pin D-only, B-only and per-component reach on both sub-steps, and a 2-D row
whose invariant z axis drives MEEP's one-partner curl forms.

The kernel steps each stored part with MEEP's forms: the Bloch connection of the
UNSHIFTED partner on plane 0, MEEP's left-to-right stencil with the chunk
record's partner presence, and the three-pass conductive tail. The gate is
adversarial, not a smoke test.  It must catch:

* PHASEOFF (control) — the Bloch connection compiled away;
* STALE_SOURCE (control) — the plan bound a copy rather than the live source;
* DROP_CONDUCTIVITY — the current sub-step took the lossless tail;
* D_SIGMA_LEAKS_TO_STEP_B — D coefficients were illegally applied to B;
* PREVIOUS_PAIRWISE_GROUPING — the previous package curl ``(g1s - g1) + (g2 - g2s)``
  (the inversion of the old MEEP-grouping mutation);
* CONNECTION_ON_THE_PREVIOUS_PLANE — the connection landed on the last plane, where
  the previous design rotated the shifted operand;
* FULL_BRACKET_OVER_AN_ABSENT_PARTNER — the presence word ignored (2-D row).

The planted leg seeds ``f = -0`` with an exact ``+0`` curl (uniform sources) and requires the
array path's words, the ``-0`` among them.

Every case is flushed and fsync'd to JSONL as it lands, and the summary is
atomically replaced.  Run this only on one confirmed-idle GPU with the installed
``keep`` policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
SUB_STEPS = {"step_B": stepping.step_B, "step_D": stepping.step_D}


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


def build_configuration(
    xp: Any, *, seed_offset: int = 0, conductivity: str = "both",
    kind: str = "lorentzian", poles: int = 1, active_pml: bool = False,
    complex_storage: bool = True, two_d: bool = False,
):
    """The lifted Group-I grid with independently selectable B/D sigma.

    ``two_d``: the same cell's x-y section on a 2-D grid, z invariant, so every curl
    component that differences along z takes MEEP's one-partner form.
    """
    grid = Grid(
        resolution=15.0, cell_size=(2.3, 2.1, 0.0) if two_d else (2.3, 2.1, 2.7),
        dimensions=2 if two_d else 3, boundaries="periodic", courant=0.5,
        k_point=(0.4, -1.3, 0.0) if two_d else (0.4, -1.3, 0.7), xp=xp)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    dtype = np.complex64 if complex_storage else np.float32
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(
                frequency=1.0 + 0.2 * index, gamma=0.1, kind=kind),
            {"Ex": 0.30, "Ey": 0.25, "Ez": 0.20}, grid, dtype)
        fields.polarizations.append(state)

    shape = tuple(grid.shape)
    host = np.linspace(
        0.12, 0.72, np.prod(shape), dtype=np.float32).reshape(shape)
    d_volume = xp.asarray(host)
    b_volume = xp.asarray((host * np.float32(0.75)).astype(np.float32))
    if conductivity in ("both", "d_only"):
        fields.set_d_conductivity(d_volume)
    elif conductivity == "mixed":
        fields.set_d_conductivity({"Dx": d_volume, "Dz": d_volume})
    if conductivity in ("both", "b_only"):
        fields.set_b_conductivity(b_volume)
    elif conductivity == "mixed":
        fields.set_b_conductivity({"By": b_volume})
    if conductivity not in {"both", "d_only", "b_only", "mixed", "none"}:
        raise ValueError(f"unknown conductivity selection {conductivity!r}")

    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2 if active_pml else 0)
    P.seed_state(fields, xp, np.random.default_rng(P.SEED + seed_offset),
                 amplitude=0.4)
    notes = {
        "shape": list(shape), "k_point": list(grid.k_point),
        "complex_storage": complex_storage, "conductivity": conductivity,
        "pml_active": bool(pml.is_active), "kind": kind, "poles": poles,
        "stores_E": bool(fields.stores_E), "dimensions": int(grid.dimensions),
    }
    return fields, pml, notes


def make_factory(xp: Any, **configuration):
    return lambda offset: build_configuration(
        xp, seed_offset=offset, **configuration)


def shipped_plan(fields: Any, pml: Any, sub_step: str):
    from meep_gpu.triton_kernels import complex_no_pml_conductive as arm

    verdict = arm.complex_conductive_no_pml_curl_coverage(
        fields, pml, sub_step)
    if not verdict.covered:
        raise RuntimeError(f"shipped predicate refused: {verdict.reasons}")
    plan = arm.plan_complex_conductive_no_pml_curl(
        fields, pml, sub_step)
    if plan is None:
        raise RuntimeError("shipped builder returned None after coverage admitted")
    return plan


def phase_off_plan(fields: Any, pml: Any, sub_step: str):
    """CONTROL: every axis's connection kind set to none (the stored word itself)."""
    plan = shipped_plan(fields, pml, sub_step)
    plan.connections = (0, 0, 0)
    plan.connection_values = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    return plan


def drop_conductivity_plan(
    fields: Any, pml: Any, sub_step: str, ):
    plan = shipped_plan(fields, pml, sub_step)
    plan.cond = (0, 0, 0)
    return plan


def leak_d_sigma_into_b_plan(
    fields: Any, pml: Any, _sub_step: str, ):
    """The exact forbidden cross-side reach: D sigma applied during step_B."""
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat

    plan = shipped_plan(fields, pml, "step_B")
    plan.cond = (1, 1, 1)
    plan._condfac = tuple(
        CupyPointer(_flat(fields.condfac_for(name)))
        for name in ("Dx", "Dy", "Dz"))
    plan._condinv = tuple(
        CupyPointer(_flat(fields.condinv_for(name)))
        for name in ("Dx", "Dy", "Dz"))
    return plan


# ---------------------------------------------------------------------------
# Kernel-text mutations: the shipped body, edited and compiled from a real file
# ---------------------------------------------------------------------------

_TEMPORARY: List[str] = []

#: name -> (needle, replacement). Each needle must occur in the shipped kernel
#: text, or the case refuses by name rather than launching the shipped kernel.
SOURCE_MUTATIONS: Dict[str, Any] = {
    # The inversion (KERNELS.md section 7.1): the previous package grouping.
    "PREVIOUS_PAIRWISE_GROUPING": (
        "    curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)",
        "    curl0_re = dtdx * ((c_y_re - c0_re) + (b0_re - b_z_re))"),
    # The previous design's plane: the last plane of the axis, where the rolled
    # operand used to carry the phase, instead of the unshifted operand's plane 0.
    "CONNECTION_ON_THE_PREVIOUS_PLANE": (
        "        c0_re = tl.where(at_y, r_re, c_re)\n"
        "        c0_im = tl.where(at_y, r_im, c_im)",
        "        c0_re = tl.where(j == ny - 1, r_re, c_re)\n"
        "        c0_im = tl.where(j == ny - 1, r_im, c_im)"),
    # The presence word ignored: the full bracket over an invariant axis's partner.
    "FULL_BRACKET_OVER_AN_ABSENT_PARTNER": (
        "    presence = tl.load(cb + RECORD_PRESENCE)",
        "    presence = 7"),
}


def shipped_kernel_source() -> str:
    """The shipped conductive curl kernel's text, read through its JIT handle."""
    from meep_gpu.triton_kernels import complex_no_pml_conductive as arm

    function = arm.complex_conductive_no_pml_curl_step
    return textwrap.dedent(inspect.getsource(getattr(function, "fn", function)))


def compile_mutated(name: str) -> Any:
    """Compile one mutated copy of the shipped kernel from a real file on disk.

    Triton reads a kernel's source through inspect, so exec'd text raises at first
    launch. THE MODULE NAME MUST NOT START WITH ``triton``: Triton's dependency
    finder leaves callees of a ``triton*`` module out of the kernel's cache key, so
    helper-only mutants would key alike and be served one binary (measured on the
    off-diagonal gate, 2026-10-08).
    """
    needle, replacement = SOURCE_MUTATIONS[name]
    source = shipped_kernel_source()
    if source.count(needle) != 1:
        raise RuntimeError(f"mutation {name}: the needle occurs {source.count(needle)} "
                           f"times in the shipped kernel, not once")
    from meep_gpu.triton_kernels import complex_fields as cx

    header = (
        "import triton\nimport triton.language as tl\n"
        "from meep_gpu.triton_kernels.complex_fields import _connect, _mul_field_left\n"
        "from meep_gpu.triton_kernels.forms import axis_present, curl_stencil\n"
        f"PERIODIC = tl.constexpr({int(getattr(cx.PERIODIC, 'value', cx.PERIODIC))})\n"
        f"METALLIC = tl.constexpr({int(getattr(cx.METALLIC, 'value', cx.METALLIC))})\n"
        "from meep_gpu.triton_kernels.kernels import RECORD_PRESENCE\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_condcurl_mutant.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source.replace(needle, replacement))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        f"condcurl_mutant_{len(_TEMPORARY)}", handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, "complex_conductive_no_pml_curl_step")


class CountingKernel:
    """Counts launches, so a mutant that never ran fails its row instead of passing."""

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        launcher = self.kernel[grid]

        def launch(*args, **kwargs):
            self.launches += 1
            return launcher(*args, **kwargs)

        return launch


def source_mutant_plan(fields: Any, pml: Any, sub_step: str, name: str, counters: Dict[str, Any]):
    plan = shipped_plan(fields, pml, sub_step)
    counter = CountingKernel(compile_mutated(name))
    counters[name] = counter
    plan._kernel = counter
    return plan


def stale_source_plan(
    fields: Any, pml: Any, sub_step: str, ):
    from meep_gpu.triton_kernels import complex_fields
    from meep_gpu.triton_kernels.complex_no_pml_curl import source_arrays

    plan = shipped_plan(fields, pml, sub_step)
    frozen = [array.copy() for array in source_arrays(fields, sub_step)]
    plan._sources = tuple(
        complex_fields.CupyPointer(complex_fields._word_view(array))
        for array in frozen)
    return plan


def case(
    name: str, sub_step: str, prediction: str, make, builder, clause: str,
) -> Dict[str, Any]:
    return {
        "name": name, "group": "I", "sub_step": sub_step,
        "prediction": prediction, "clause": clause, "make": make,
        "array_path": SUB_STEPS[sub_step], "kernel_plan": builder,
    }


#: Launch counters of the kernel-text mutants, by mutation name (filled as they run).
MUTANT_COUNTERS: Dict[str, Any] = {}


def build_cases(xp: Any) -> List[Dict[str, Any]]:
    target = make_factory(xp, conductivity="both", poles=1)
    two_d = make_factory(xp, conductivity="both", poles=1, two_d=True)
    cases: List[Dict[str, Any]] = []
    for row in GROUP_I_ROWS:
        for sub_step in SUB_STEPS:
            cases.append(case(
                "product_" + row.replace(".", "__") + "_" + sub_step,
                sub_step, "IDENTICAL", target,
                lambda f, p, s=sub_step:
                    shipped_plan(f, p, s),
                "one-to-one lifted Group-I curl slot with a live Lorentz pole"))

    # The current sub-step flags may all be false even though run-wide admission
    # selects this family.  Both directions are product claims, not mutations.
    for selection in ("d_only", "b_only", "mixed"):
        make = make_factory(xp, conductivity=selection, poles=2,
                            kind="drude" if selection == "mixed" else "lorentzian")
        for sub_step in SUB_STEPS:
            cases.append(case(
                f"product_{selection}_{sub_step}", sub_step, "IDENTICAL", make,
                lambda f, p, s=sub_step:
                    shipped_plan(f, p, s),
                "per-target conductivity reach with registered poles"))
    for sub_step in SUB_STEPS:
        cases.append(case(
            f"product_2d_{sub_step}", sub_step, "IDENTICAL", two_d,
            lambda f, p, s=sub_step:
                shipped_plan(f, p, s),
            "2-D: the invariant z axis takes MEEP's one-partner curl forms"))

    for sub_step in SUB_STEPS:
        cases.append(case(
            f"CONTROL_PHASEOFF_{sub_step}", sub_step, "DIVERGENT", target,
            lambda f, p, s=sub_step:
                phase_off_plan(f, p, s),
            "control: the Bloch connection compiled away"))
        cases.append(case(
            f"MUTATION_DROP_CONDUCTIVITY_{sub_step}", sub_step, "DIVERGENT",
            target,
            lambda f, p, s=sub_step:
                drop_conductivity_plan(f, p, s),
            "mutation: current-side conductive tail compiled away"))
    for name, factory, sub_steps in (
            ("PREVIOUS_PAIRWISE_GROUPING", target, ("step_B", "step_D")),
            ("CONNECTION_ON_THE_PREVIOUS_PLANE", target, ("step_B", "step_D")),
            ("FULL_BRACKET_OVER_AN_ABSENT_PARTNER", two_d, ("step_B", "step_D"))):
        for sub_step in sub_steps:
            cases.append(case(
                f"MUTATION_{name}_{sub_step}", sub_step, "DIVERGENT", factory,
                lambda f, p, s=sub_step, n=name:
                    source_mutant_plan(f, p, s, n, MUTANT_COUNTERS),
                f"kernel-text mutation: {name.lower().replace('_', ' ')}"))

    d_only = make_factory(xp, conductivity="d_only", poles=1)
    cases.append(case(
        "MUTATION_D_SIGMA_LEAKS_TO_STEP_B", "step_B", "DIVERGENT", d_only,
        lambda f, p:
            leak_d_sigma_into_b_plan(f, p, "step_B"),
        "mutation: D conductivity illegally reaches a B curl term"))
    cases.append(case(
        "CONTROL_STALE_SOURCE_step_B", "step_B", "DIVERGENT", target,
        lambda f, p:
            stale_source_plan(f, p, "step_B"),
        "control: source bound as a plan-time copy"))
    return cases


def planted_leg(xp: Any) -> Dict[str, Any]:
    """``f = -0`` on every target with uniform sources: an exact ``+0`` curl inside.

    MEEP's plain form ``f - curl`` keeps the ``-0``; the conductive tail
    ``((f * condfac) - curl) * condinv`` keeps it too for a positive coefficient. The
    shipped plan must store the array path's words, and the ``-0`` must be among
    them (a leg that stored no ``-0`` measured nothing). Run on the Group-I shape and
    on the 2-D section, each sub-step, with D-only sigma so ``step_B`` takes the plain
    tail.
    """
    rows: List[Dict[str, Any]] = []
    for label, two_d in (("3d", False), ("2d", True)):
        for sub_step, conductivity in (("step_B", "d_only"), ("step_D", "d_only"),
                                       ("step_B", "both")):
            reference_fields, pml, _ = build_configuration(
                xp, conductivity=conductivity, two_d=two_d)
            candidate_fields, _, _ = build_configuration(
                xp, conductivity=conductivity, two_d=two_d)
            shape = tuple(reference_fields.grid.shape)
            from meep_gpu.triton_kernels.launch import SUB_STEPS as TABLE

            for fields in (reference_fields, candidate_fields):
                minus = np.empty(shape, dtype=np.complex64)
                minus.real = np.float32(-0.0)
                minus.imag = np.float32(-0.0)
                uniform = np.empty(shape, dtype=np.complex64)
                uniform.real = np.float32(0.25)
                uniform.imag = np.float32(-0.125)
                for name in TABLE[sub_step]["targets"]:
                    getattr(fields, name)[...] = xp.asarray(minus)
                for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
                    array = getattr(fields, name, None)
                    if array is not None and name not in TABLE[sub_step]["targets"]:
                        array[...] = xp.asarray(uniform)
            row: Dict[str, Any] = {"case": f"planted_{label}_{sub_step}_{conductivity}"}
            try:
                plan = shipped_plan(candidate_fields, pml, sub_step)
                SUB_STEPS[sub_step](reference_fields, pml)
                plan.run()
                verdict = P.compare(P.inventory(reference_fields),
                                    P.inventory(candidate_fields))
                minus_zero = sum(
                    int((P.as_words(getattr(reference_fields, name)) == 0x80000000).sum())
                    for name in TABLE[sub_step]["targets"])
                row.update({"differing": verdict["differing"],
                            "compared": verdict["words"],
                            "identical": verdict["identical"],
                            "negative_zero_words_stored": minus_zero,
                            "first_difference": verdict.get("first_difference")})
            except BaseException as exc:  # noqa: BLE001 - the outcome is the row
                row.update({"error": f"{type(exc).__name__}: {exc}"[:400],
                            "traceback": traceback.format_exc()[-1500:]})
            rows.append(row)
    return {"cases": rows,
            "identical": all(row.get("identical") for row in rows),
            "every_row_stored_minus_zero": all(
                row.get("negative_zero_words_stored", 0) > 0 for row in rows)}


def refusal_leg(xp: Any) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import complex_no_pml_conductive as arm
    from meep_gpu.triton_kernels import complex_no_pml_curl as incumbent

    rows: List[Dict[str, Any]] = []
    fixtures = (
        ("target", {}, True),
        ("active_pml", {"active_pml": True}, False),
        ("real_storage", {"complex_storage": False}, False),
        ("no_conductivity", {"conductivity": "none"}, False),
    )
    for label, configuration, expected in fixtures:
        fields, pml, _ = build_configuration(xp, **configuration)
        for sub_step in SUB_STEPS:
            verdict = arm.complex_conductive_no_pml_curl_coverage(
                fields, pml, sub_step)
            old = incumbent.complex_no_pml_curl_coverage(
                fields, pml, sub_step)
            row = {
                "case": f"{label}:{sub_step}", "covered": verdict.covered,
                "expected": expected,
                "agrees": verdict.covered == expected,
                "reasons": list(verdict.reasons),
                "incumbent_covered": old.covered,
                "both_admit": verdict.covered and old.covered,
            }
            rows.append(row)

    # The public Susceptibility constructor correctly rejects unsupported
    # kinds, so plant the malformed state only after constructing a valid one.
    # This asks the predicate's defensive clause without inventing a public
    # configuration the engine itself would accept.
    from types import SimpleNamespace

    fields, pml, _ = build_configuration(xp)
    fields.polarizations[0].susceptibility = SimpleNamespace(kind="gyrotropic")
    for sub_step in SUB_STEPS:
        verdict = arm.complex_conductive_no_pml_curl_coverage(
            fields, pml, sub_step)
        rows.append({
            "case": f"unsupported_pole:{sub_step}",
            "covered": verdict.covered, "expected": False,
            "agrees": not verdict.covered, "reasons": list(verdict.reasons),
            "incumbent_covered": False, "both_admit": False,
        })
    return {
        "cases": rows,
        "all_agree": all(row["agrees"] for row in rows),
        "disjoint": not any(row["both_admit"] for row in rows),
    }


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rows = payload.get("cases") or []
    products = [row for row in rows
                if str(row.get("case", "")).startswith("product_")]
    mutations = [row for row in rows if "MUTATION" in str(row.get("case", ""))]
    controls = [row for row in rows if "CONTROL" in str(row.get("case", ""))]
    nulls = [row for row in rows if str(row.get("case", "")).startswith("NULL_")]
    if len(products) != 16:
        reasons.append(f"expected 16 products (8 corpus + 6 reach + 2 two-dimensional), "
                       f"found {len(products)}")
    if len(mutations) != 9:
        reasons.append(f"expected 9 mutations, found {len(mutations)}")
    if len(controls) != 3:
        reasons.append(f"expected 3 controls, found {len(controls)}")
    # NO DECLARED NULL REMAINS: the two NULL_WRONG_EXPANSION rows were the deleted
    # expansion licence's opposite arm (entry 89). A NULL_ row appearing again is a
    # case this verdict does not know how to read.
    if nulls:
        reasons.append(f"expected no declared nulls, found {len(nulls)}")
    for name, launches in (payload.get("mutant_launches") or {}).items():
        if not launches:
            reasons.append(f"the kernel-text mutant {name} never launched: DISARMED")
    planted = payload.get("planted") or {}
    if not planted.get("cases"):
        reasons.append("the planted signed-zero leg did not run")
    else:
        for row in planted["cases"]:
            if row.get("error"):
                reasons.append(f"{row.get('case')}: {row['error']}")
            elif not row.get("identical"):
                reasons.append(f"{row.get('case')}: {row.get('differing')} words differ "
                               f"from the array path")
            elif not row.get("negative_zero_words_stored"):
                reasons.append(f"{row.get('case')}: no -0 word was stored; the planted "
                               f"leg measured nothing")
    for row in rows:
        if row.get("error"):
            reasons.append(f"{row.get('case')}: {row['error']}")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(
                f"{row.get('case')}: {row.get('verdict')} != {row.get('prediction')}")
        if row.get("vacuous"):
            reasons.append(f"{row.get('case')}: vacuous")
    refusal = payload.get("refusals") or {}
    if not refusal.get("all_agree"):
        reasons.append("a coverage/refusal expectation disagreed")
    if not refusal.get("disjoint"):
        reasons.append("the new and incumbent predicates overlap")
    return {"released": not reasons, "reasons": reasons}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cycles", type=int, default=CYCLES)
    parser.add_argument("--subnormal-policy", default="keep", choices=("keep", "flush"),
                        help="the float32 subnormal policy to install before the first device compile (keep, the default, or flush: the certification round gates every family under both on the NVIDIA tables)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_log = (out.parent / "driver.log").open("a", encoding="utf-8")
    jsonl = (out.parent / "cases.jsonl").open("a", encoding="utf-8")
    started = time.time()
    payload: Dict[str, Any] = {
        "gate": "triton_complex_no_pml_conductive",
        "cycles": args.cycles, "group_i_rows": list(GROUP_I_ROWS),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import cupy as cp
        from meep_gpu.subnormal_policy import install_subnormal_policy

        install_subnormal_policy(args.subnormal_policy)
        payload["subnormal_policy_installed"] = args.subnormal_policy
        cases = build_cases(cp)
        rows = []
        for index, spec in enumerate(cases, 1):
            log(run_log, f"case {index}/{len(cases)} {spec['name']} "
                         f"predict={spec['prediction']}")
            try:
                row = P.run_case(cycles=args.cycles, **spec)
            except BaseException as exc:  # noqa: BLE001 - artifact and continue
                row = {
                    "case": spec["name"], "prediction": spec["prediction"],
                    "verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:500],
                    "traceback": traceback.format_exc()[-1800:],
                }
            rows.append(row)
            emit(jsonl, row)
            log(run_log, f"  -> {row.get('verdict', 'ERROR')} "
                         f"diff={row.get('differing_max', '?')} "
                         f"elapsed={time.time() - started:.1f}s")
            payload["cases"] = rows
            save(out, payload)
        payload["mutant_launches"] = {name: counter.launches
                                      for name, counter in MUTANT_COUNTERS.items()}
        payload["planted"] = planted_leg(cp)
        log(run_log, f"planted: identical={payload['planted']['identical']} "
                     f"-0 stored on every row={payload['planted']['every_row_stored_minus_zero']}")
        save(out, payload)
        payload["refusals"] = refusal_leg(cp)
    except BaseException as exc:  # noqa: BLE001 - setup failure is an artifact
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
