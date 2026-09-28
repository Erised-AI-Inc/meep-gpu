"""Byte gate for the NO-PML curl kernel (``no_pml.plain_curl_step``).

Legs, selected with ``--leg`` (default: all but ``budget`` and ``bench``):

  product   the shape x boundary x Courant x sub-step x DERIVE sweep, GUARDED and
            UNGUARDED, kernel vs ``no_pml_ref`` on CuPy. The unguarded run is the
            control that proves the gate bites.
  multistep N consecutive launches of the same plan, compared after every one.
  mutations one deliberately broken kernel per defect class, each of which must be
            CAUGHT. A mutation leg that reports a pass for a mutation it never
            applied is worse than no leg (plan section 13.4), so every leg records
            the sha256 of the source it compiled.
  engine    the ENGINE plan route on a real lifted mp.Simulation, plus the
            refusals, plus N whole driver steps.
  budget    the STEP BUDGET: 2d_plain run to --budget-steps with every state array
            compared bytewise after every step, one fsync'd jsonl row per step.
            Section 16 measured 2d_pml first differing at step 67 after a 60-step
            budget certified it, so a budget is a claim about N and nothing more.
  bench     throughput: array path vs DERIVE=0 vs DERIVE=1, sub-step and whole step.

Usage on the GPU host::

    TRITON_LIBCUDA_PATH=$HOME/triton_libcuda_stub CUDA_VISIBLE_DEVICES=<clear> \
    PYTHONPATH=<repo>:<scripts> python -u gate_no_pml.py --out-dir results/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
#: ``the repository root``, derived from this file's own location rather than an environment
#: variable, so the gate runs from a plain rsync of the tree on any host.
REPO_API = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
#: The shipped kernel. The mutation legs slice the ``@triton.jit`` body out of THIS
#: file, so a gate run measures the module the tree actually ships, never a copy.
SHIPPED_SOURCE = os.path.join(REPO_API, "meep_gpu", "triton_kernels", "no_pml.py")
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import no_pml_ref as ref  # noqa: E402

SHAPES = ((32, 24, 16), (64, 1, 64), (48, 48, 1), (17, 19, 23))
BOUNDARY_SETS = (
    ("periodic", "periodic", "periodic"),
    ("metallic", "metallic", "metallic"),
    ("metallic", "periodic", "metallic"),
    ("periodic", "metallic", "periodic"),
)
# 0.5 alone certifies a broken kernel: the FMA/association discrepancy vanishes
# where the scale factor is a power of two (plan section 9).
COURANTS = (0.5, 0.35, 0.2673)
SUB_STEPS = ("step_B", "step_D")
MULTISTEP = 8


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def make_arrays(cp, shape, sub_step: str, derive: int, rng) -> Dict[str, Any]:
    """Every array the sub-step reads or writes, seeded and non-degenerate.

    ``inv_eps`` is deliberately NOT 1.0 anywhere: the ``2d_plain`` benchmark case
    is a uniform vacuum, where ``D * inv_eps`` is bit-equal to ``D`` and a dropped
    derivation is invisible. A gate seeded from that case would certify it.
    """
    names = ["Bx", "By", "Bz", "Dx", "Dy", "Dz"]
    arrays = {n: cp.asarray(rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
              for n in names}
    if sub_step == "step_B" and not derive:
        for n in ("Ex", "Ey", "Ez"):
            arrays[n] = cp.asarray(rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
    if derive:
        for n in ("Ex", "Ey", "Ez"):
            arrays["inv_eps_" + n] = cp.asarray(
                rng.uniform(0.25, 1.0, size=shape).astype(np.float32))
    return arrays


def reference_sources(cp, arrays: Dict[str, Any], sub_step: str, derive: int):
    if sub_step == "step_D":
        # get_H is the B array itself without PML (fields.py:1184-1186).
        return {"Hx": arrays["Bx"], "Hy": arrays["By"], "Hz": arrays["Bz"]}
    if derive:
        return ref.derive_electric(
            cp, {n: arrays[n] for n in ("Dx", "Dy", "Dz")},
            {n: arrays["inv_eps_" + n] for n in ("Ex", "Ey", "Ez")})
    return {n: arrays[n] for n in ("Ex", "Ey", "Ez")}


TARGETS = {"step_B": ("Bx", "By", "Bz"), "step_D": ("Dx", "Dy", "Dz")}


def one_product_case(cp, shape, boundaries, dtdx, sub_step, derive, guard,
                     kernel=None) -> Dict[str, Any]:
    from meep_gpu.triton_kernels.no_pml import plan_plain_curl_from_arrays  # noqa: PLC0415

    rng = np.random.default_rng(ref.SEED + 3)
    arrays = make_arrays(cp, shape, sub_step, derive, rng)
    targets = TARGETS[sub_step]

    # Reference: the array-path transcription, on CuPy, from a copy.
    state = {n: arrays[n].copy() for n in targets}
    sources = reference_sources(cp, arrays, sub_step, derive)
    ref.reference_plain_step(cp, sources, state, np.float32(dtdx), sub_step,
                             tuple(boundaries), "array_order")

    codes = [1 if kind == "metallic" else 0 for kind in boundaries]
    plan = plan_plain_curl_from_arrays(sub_step, arrays, codes, dtdx,
                                       derive=derive, kernel=kernel)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    parts = {n: ref.bit_compare(arrays[n], state[n]) for n in targets}
    out = ref.combine(parts)
    # The sub-step must actually have moved something: two unchanged copies
    # compare identical for the wrong reason.
    out["all_targets_changed"] = True  # filled by the caller's before-snapshot
    return out


def leg_product(cp, guard: bool, kernel=None) -> Dict[str, Any]:
    from meep_gpu.triton_kernels.no_pml import plan_plain_curl_from_arrays  # noqa: PLC0415

    cases: List[Dict[str, Any]] = []
    started = time.time()
    total = (len(SHAPES) * len(BOUNDARY_SETS) * len(COURANTS)
             * len(SUB_STEPS) * 2)
    index = 0
    for shape in SHAPES:
        for boundaries in BOUNDARY_SETS:
            for dtdx in COURANTS:
                for sub_step in SUB_STEPS:
                    for derive in (0, 1):
                        index += 1
                        if sub_step == "step_D" and derive:
                            cases.append({"skipped": "step_D has no E derivation",
                                          "sub_step": sub_step, "derive": derive})
                            continue
                        walled = [a for a in range(3) if shape[a] == 1
                                  and boundaries[a] == "metallic"]
                        if walled:
                            # Grid refuses it (measured: the local transcription
                            # leg skips the same 32 cases), and the kernel masks
                            # the whole plane, so a target legitimately never
                            # moves and the comparison would be identical for the
                            # wrong reason.
                            cases.append({"skipped": f"axis {walled[0]} is one cell "
                                                     f"and declared metallic",
                                          "shape": list(shape),
                                          "boundaries": list(boundaries),
                                          "sub_step": sub_step, "derive": derive})
                            continue
                        rng = np.random.default_rng(ref.SEED + 3)
                        arrays = make_arrays(cp, shape, sub_step, derive, rng)
                        targets = TARGETS[sub_step]
                        before = {n: arrays[n].copy() for n in targets}
                        state = {n: arrays[n].copy() for n in targets}
                        sources = reference_sources(cp, arrays, sub_step, derive)
                        ref.reference_plain_step(cp, sources, state,
                                                 np.float32(dtdx), sub_step,
                                                 tuple(boundaries), "array_order")
                        codes = [1 if k == "metallic" else 0 for k in boundaries]
                        plan = plan_plain_curl_from_arrays(
                            sub_step, arrays, codes, dtdx, derive=derive,
                            kernel=kernel)
                        plan.run(guard=guard)
                        cp.cuda.runtime.deviceSynchronize()
                        verdict = ref.combine(
                            {n: ref.bit_compare(arrays[n], state[n]) for n in targets})
                        verdict["all_targets_changed"] = all(
                            not bool((arrays[n] == before[n]).all()) for n in targets)
                        verdict.update({"shape": list(shape),
                                        "boundaries": list(boundaries),
                                        "dtdx": dtdx, "sub_step": sub_step,
                                        "derive": derive})
                        verdict.pop("per_array", None)
                        cases.append(verdict)
                        log(f"  product {index}/{total} {shape} {boundaries} "
                            f"dtdx={dtdx} {sub_step} derive={derive} "
                            f"guard={guard}: identical={verdict['bit_identical']} "
                            f"ndiff={verdict['differing_floats']} "
                            f"({time.time() - started:.1f} s)")
    live = [c for c in cases if "skipped" not in c]
    identical = [c for c in live if c["bit_identical"]]
    moved = [c for c in live if c.get("all_targets_changed")]
    return {"guard": guard, "cases": cases,
            "identical": f"{len(identical)}/{len(live)}",
            "n_identical": len(identical), "n_live": len(live),
            "skipped": len(cases) - len(live),
            "all_targets_changed": f"{len(moved)}/{len(live)}",
            "worst_differing_floats": max([c["differing_floats"] for c in live] or [0]),
            "worst_max_ulp": max([c.get("max_ulp", 0) for c in live] or [0]),
            "wall_s": round(time.time() - started, 1)}


def leg_multistep(cp) -> Dict[str, Any]:
    """N consecutive launches. One launch can agree by accident; N cannot."""
    from meep_gpu.triton_kernels.no_pml import plan_plain_curl_from_arrays  # noqa: PLC0415

    out: List[Dict[str, Any]] = []
    for shape in SHAPES[:2]:
        for boundaries in (BOUNDARY_SETS[0], BOUNDARY_SETS[1]):
            for dtdx in (0.35,):
                for sub_step in SUB_STEPS:
                    for derive in ((0, 1) if sub_step == "step_B" else (0,)):
                        rng = np.random.default_rng(ref.SEED + 5)
                        arrays = make_arrays(cp, shape, sub_step, derive, rng)
                        targets = TARGETS[sub_step]
                        mirror = {n: arrays[n].copy() for n in arrays}
                        codes = [1 if k == "metallic" else 0 for k in boundaries]
                        plan = plan_plain_curl_from_arrays(sub_step, arrays, codes,
                                                           dtdx, derive=derive)
                        steps = []
                        for step in range(1, MULTISTEP + 1):
                            sources = reference_sources(cp, mirror, sub_step, derive)
                            ref.reference_plain_step(
                                cp, sources, {n: mirror[n] for n in targets},
                                np.float32(dtdx), sub_step, tuple(boundaries),
                                "array_order")
                            plan.run()
                            cp.cuda.runtime.deviceSynchronize()
                            verdict = ref.combine(
                                {n: ref.bit_compare(arrays[n], mirror[n])
                                 for n in targets})
                            verdict.pop("per_array", None)
                            verdict["step"] = step
                            steps.append(verdict)
                        row = {"shape": list(shape), "boundaries": list(boundaries),
                               "dtdx": dtdx, "sub_step": sub_step, "derive": derive,
                               "steps": steps,
                               "bit_identical": all(s["bit_identical"] for s in steps),
                               "first_divergent_step": next(
                                   (s["step"] for s in steps if not s["bit_identical"]),
                                   None)}
                        out.append(row)
                        log(f"  multistep {shape} {boundaries} {sub_step} "
                            f"derive={derive}: {MULTISTEP} steps identical="
                            f"{row['bit_identical']}")
    return {"cases": out,
            "identical": f"{sum(1 for r in out if r['bit_identical'])}/{len(out)}"}


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------
#
# Each entry rewrites ONE line of the kernel source and recompiles it. The plan
# object carries the mutated kernel explicitly (``kernel=``); dropping that
# argument disarms every leg silently, which is the harness defect section 13.4
# records — 4/4 real defects came back "identical" with the shipped kernel
# running underneath. The sha256 of the mutated source is recorded per leg so a
# leg that ran unmutated tables cannot report a pass.

MUTATIONS: Tuple[Tuple[str, str, str, str], ...] = (
    ("flatten_grouping",
     "curl0 = dtdx * ((c_y - c) + (b - b_z))",
     "curl0 = dtdx * (((c_y - c) + b) - b_z)",
     "C's left-to-right association of the stencil; must be caught at a "
     "non-power-of-two Courant and MAY legitimately survive at 0.5"),
    ("add_instead_of_subtract",
     "v0 = tl.load(f0 + idx, mask=live, other=0.0) - curl0",
     "v0 = tl.load(f0 + idx, mask=live, other=0.0) + curl0",
     "the plain branch's sign (stepping._apply_curl:508)"),
    ("drop_inverse_epsilon",
     "a = tl.load(g0 + idx, mask=live, other=0.0) * tl.load(e0 + idx, mask=live, other=0.0)",
     "a = tl.load(g0 + idx, mask=live, other=0.0)",
     "DERIVE forms E = D, not D * inv_eps; DERIVE=0 cases are legitimately "
     "unaffected"),
    ("derive_centre_only",
     "a_y = tl.load(g0 + oy, mask=vy, other=0.0) * tl.load(e0 + oy, mask=vy, other=0.0)",
     "a_y = tl.load(g0 + oy, mask=vy, other=0.0)",
     "the derivation applied at the cell but not at its neighbour — a "
     "half-derived stencil, smooth and wrong"),
    ("drop_ownership_mask",
     "curl0 = tl.where(at_x, 0.0, curl0)",
     "curl0 = curl0",
     "stepping._mask_non_owned_cells on the B side; periodic cases are "
     "legitimately unaffected"),
    ("periodic_ghost_on_metallic",
     "vx = live & (si >= 0) & (si < nx)",
     "vx = live",
     "a metallic axis served the wrapped plane instead of the zero ghost; "
     "periodic cases are legitimately unaffected"),
    ("swap_derive_operand_order",
     "a = tl.load(g0 + idx, mask=live, other=0.0) * tl.load(e0 + idx, mask=live, other=0.0)",
     "a = tl.load(e0 + idx, mask=live, other=0.0) * tl.load(g0 + idx, mask=live, other=0.0)",
     "EXPECTED-UNCAUGHT control: float multiplication is bitwise commutative, "
     "so this must come back 0 caught. A leg that reports it CAUGHT is "
     "measuring something other than the mutation."),
)


def compile_mutant(name: str, old: str, new: str, source_path: str):
    """Compile a one-line-mutated copy of the kernel FROM A REAL FILE ON DISK.

    Not ``exec``: Triton reads a kernel's text with ``inspect.getsource``, so an
    exec-ed ``@triton.jit`` body raises ``OSError: could not get source code`` at
    decoration. Measured here, and the sibling fold gate records the same thing
    (gate_triton_symmetry.compile_mutated) — worth stating twice because the
    failure is loud in one gate and would be a disarmed leg in another.
    """
    import importlib.util  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    if source.count(old) < 1:
        raise RuntimeError(f"mutation {name!r} anchor not found: {old!r}")
    mutated = source.replace(old, new, 1)
    if mutated == source:
        raise RuntimeError(f"mutation {name!r} changed nothing")
    digest = hashlib.sha256(mutated.encode()).hexdigest()
    handle = tempfile.NamedTemporaryFile("w", suffix=f"_mutant_{name}.py",
                                         delete=False, encoding="utf-8")
    handle.write(mutated)
    handle.close()
    spec = importlib.util.spec_from_file_location(f"no_pml_mutant_{name}",
                                                  handle.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.plain_curl_step, digest


def kernel_only_source(path: str) -> str:
    """Write the @triton.jit body alone to a temp file, importable without the package.

    ``no_pml.py`` imports ``.coverage``/``.launch`` relatively, which will not
    load outside the package; the mutation legs only need the kernel.
    """
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    # ANCHOR ON THE DEFINITION, NOT THE DECORATOR ALONE. A bare ``"@triton.jit"``
    # index was the anchor once, and MEASURED: adding a module comment that merely
    # MENTIONED the decorator moved the slice's start and silently changed what the
    # gate compiled. It still ran, still reported a verdict, and the verdict was
    # about different bytes. Both anchors are asserted unique below for the same
    # reason — a slice this gate is wrong about is a gate that certifies nothing.
    start_anchor = "\n@triton.jit\ndef plain_curl_step("
    end_anchor = ("# ---------------------------------------------------------------------------"
                  "\n# Coverage")
    for anchor in (start_anchor, end_anchor):
        if text.count(anchor) != 1:
            raise RuntimeError(
                f"kernel slice anchor is not unique in {path}: {anchor!r} occurs "
                f"{text.count(anchor)} times")
    start = text.index(start_anchor) + 1
    end = text.index(end_anchor)
    body = ("import triton\nimport triton.language as tl\n"
            "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n"
            + text[start:end])
    # A temp directory, not HERE: the recon build wrote this beside the gate and
    # left a stray module in the source tree that a later run would silently reuse.
    out = os.path.join(tempfile.mkdtemp(prefix="no_pml_gate_"), "no_pml_kernel_only.py")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(body)
    return out


def leg_mutations(cp, source_path: str) -> Dict[str, Any]:
    from meep_gpu.triton_kernels.no_pml import plan_plain_curl_from_arrays  # noqa: PLC0415

    kernel_src = kernel_only_source(source_path)
    shipped_digest = sha256_file(kernel_src)
    results: List[Dict[str, Any]] = []
    for name, old, new, why in MUTATIONS:
        kernel, digest = compile_mutant(name, old, new, kernel_src)
        if digest == shipped_digest:
            raise RuntimeError(f"mutation {name!r} compiled the shipped source")
        caught, live, detail = 0, 0, []
        for shape in SHAPES[:2]:
            for boundaries in BOUNDARY_SETS:
                for dtdx in COURANTS:
                    for sub_step in SUB_STEPS:
                        for derive in ((0, 1) if sub_step == "step_B" else (0,)):
                            if any(shape[a] == 1 and boundaries[a] == "metallic"
                                   for a in range(3)):
                                continue  # Grid refuses it; see leg_product.
                            rng = np.random.default_rng(ref.SEED + 3)
                            arrays = make_arrays(cp, shape, sub_step, derive, rng)
                            targets = TARGETS[sub_step]
                            state = {n: arrays[n].copy() for n in targets}
                            sources = reference_sources(cp, arrays, sub_step, derive)
                            ref.reference_plain_step(cp, sources, state,
                                                     np.float32(dtdx), sub_step,
                                                     tuple(boundaries), "array_order")
                            codes = [1 if k == "metallic" else 0 for k in boundaries]
                            plan = plan_plain_curl_from_arrays(
                                sub_step, arrays, codes, dtdx, derive=derive,
                                kernel=kernel)
                            plan.run()
                            cp.cuda.runtime.deviceSynchronize()
                            same = all(ref.bit_compare(arrays[n], state[n])["bit_identical"]
                                       for n in targets)
                            live += 1
                            if not same:
                                caught += 1
                            detail.append({"shape": list(shape),
                                           "boundaries": list(boundaries),
                                           "dtdx": dtdx, "sub_step": sub_step,
                                           "derive": derive, "caught": not same})
        row = {"mutation": name, "why": why, "sha256": digest,
               "caught": f"{caught}/{live}", "n_caught": caught, "n_live": live,
               "detail": detail}
        results.append(row)
        log(f"  mutation {name}: caught {caught}/{live}")
    return {"shipped_kernel_sha256": shipped_digest, "mutations": results}


# ---------------------------------------------------------------------------
# The engine route + the step budget
# ---------------------------------------------------------------------------

COVERED_CASES = (("2d_plain", {"res": 40, "n": 16}),)
REFUSAL_CASES = (("2d_pml", {"res": 40, "n": 16}),
                 ("2d_symmetry", {"res": 40, "n": 16}),
                 ("2d_cond_pml", {"res": 40, "n": 16}),
                 ("cylindrical", {"res": 80, "r": 8.0, "z": 8.0}),
                 ("2d_dispersive", {"res": 40, "n": 16}))

STATE_ARRAYS = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                "Hx", "Hy", "Hz")


def driver_state(driver) -> Dict[str, Any]:
    state = {n: getattr(driver.fields, n) for n in STATE_ARRAYS
             if getattr(driver.fields, n, None) is not None}
    for index, pole in enumerate(driver.fields.polarizations):
        for component in pole.driven():
            state[f"P{index}_{component}"] = pole.P[component]
            state[f"Pprev{index}_{component}"] = pole.P_prev[component]
    return state


def leg_engine(cp, steps: int, jsonl: Optional[str] = None) -> Dict[str, Any]:
    import cases  # noqa: PLC0415

    from meep_gpu import backends, driver as driver_module  # noqa: PLC0415
    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step, plain_curl_coverage  # noqa: PLC0415
    from meep_gpu.triton_kernels.no_pml import plan_plain_curl  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    out: Dict[str, Any] = {"refusals": [], "covered": []}

    for name, kwargs in REFUSAL_CASES:
        driver = lift_simulation(cases.BUILDERS[name](**kwargs), prefer_gpu=True,
                                 gpu_id=0)
        try:
            verdicts = {s: plain_curl_coverage(driver.fields, driver.pml, s)
                        for s in SUB_STEPS}
            plans = {s: plan_plain_curl(driver.fields, driver.pml, s)
                     for s in SUB_STEPS}
            central = plan_step(driver.fields, driver.pml)
            selected = {s: type(central.plans[s]).__name__
                        for s in SUB_STEPS if s in central.plans}
            row = {"case": name,
                   "covered": {s: bool(v.covered) for s, v in verdicts.items()},
                   "reasons": {s: list(v.reasons) for s, v in verdicts.items()},
                   "plan_is_none": {s: p is None for s, p in plans.items()},
                   "central_selected": selected,
                   "central_replaces": list(central.replaces),
                   "central_reasons": {s: list(v)
                                       for s, v in central.reasons.items()}}
            if any(row["covered"].values()) or not all(row["plan_is_none"].values()):
                raise AssertionError(f"no-PML product admitted refusal case: {row}")
            if any(kind == "PlainCurlPlan" for kind in selected.values()):
                raise AssertionError(f"central planner selected no-PML product: {row}")
            out["refusals"].append(row)
            log(f"  engine refusal {name}: plan_is_none={row['plan_is_none']} "
                f"central={selected} reasons[step_B]={row['reasons']['step_B'][:2]}")
        finally:
            driver.close()
            cp.get_default_memory_pool().free_all_blocks()

    for name, kwargs in COVERED_CASES:
        reference = lift_simulation(cases.BUILDERS[name](**kwargs), prefer_gpu=True,
                                    gpu_id=0)
        fused = lift_simulation(cases.BUILDERS[name](**kwargs), prefer_gpu=True,
                                gpu_id=0)
        names = ("step_B", "step_D")
        originals = {n: getattr(driver_module, n) for n in names}
        handle = open(jsonl, "a") if jsonl else None
        try:
            plan = plan_step(fused.fields, fused.pml)
            # Since the null-constitutive family was wired (planner recert
            # 2026-08-14), the certified selection on a no-PML grid is the
            # plain curl pair PLUS a NullConstitutivePlan on each constitutive
            # sub-step (stepping.update_H/update_E return before reading an
            # array under an inactive absorber). Pin the COMPLETE shape: the
            # curls must still be PlainCurlPlan — a null displacing a curl
            # fails here — and the constitutive slots must be exactly the
            # null product, so a kernel displacing a null fails too.
            null_names = ("update_H", "update_E")
            expected_replaces = set(names) | set(null_names)
            if set(plan.replaces) != expected_replaces:
                raise AssertionError(
                    f"central planner did not select the plain curl pair plus "
                    f"the null constitutive slots: replaces={plan.replaces}, "
                    f"reasons={plan.reasons}")
            if any(type(plan.plans[s]).__name__ != "PlainCurlPlan" for s in names):
                raise AssertionError(
                    f"central planner selected the wrong curl product: "
                    f"{ {s: type(plan.plans[s]).__name__ for s in names} }")
            if any(type(plan.plans[s]).__name__ != "NullConstitutivePlan"
                   for s in null_names):
                raise AssertionError(
                    f"central planner selected a non-null constitutive product "
                    f"on a no-PML grid: "
                    f"{ {s: type(plan.plans[s]).__name__ for s in null_names} }")
            row: Dict[str, Any] = {
                "case": name,
                "shape": [int(s) for s in fused.shape],
                "stores_E": bool(fused.fields.stores_E),
                "replaces": list(plan.replaces),
                "selected": {s: type(plan.plans[s]).__name__ for s in names},
                "derive": {s: plan.plans[s].derive for s in names},
                "refusals": {k: list(v) for k, v in plan.reasons.items()},
            }

            def make(sub_step):
                def wrapper(fields, pml=None):
                    entry = (plan.plans.get(sub_step)
                             if fields is fused.fields else None)
                    if entry is None:
                        return originals[sub_step](fields, pml)
                    entry.run()
                    return None
                return wrapper

            for sub_step in names:
                setattr(driver_module, sub_step, make(sub_step))

            first_divergent = None
            started = time.time()
            for step in range(1, steps + 1):
                reference.step()
                fused.step()
                cp.cuda.runtime.deviceSynchronize()
                a, b = driver_state(fused), driver_state(reference)
                parts = {k: ref.bit_compare(a[k], b[k]) for k in sorted(b)}
                same = all(p["bit_identical"] for p in parts.values())
                if not same and first_divergent is None:
                    first_divergent = step
                    row["first_divergence_detail"] = {
                        k: v for k, v in parts.items() if not v["bit_identical"]}
                if handle is not None:
                    handle.write(json.dumps({
                        "case": name, "step": step, "bit_identical": same,
                        "differing_floats": sum(p["differing_floats"]
                                                for p in parts.values()),
                        "total_floats": sum(p["total_floats"] for p in parts.values()),
                        "differing_arrays": sorted(k for k, v in parts.items()
                                                   if not v["bit_identical"]),
                    }) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                if step % 100 == 0 or step <= 5 or not same:
                    log(f"  engine {name} step {step}/{steps} identical={same} "
                        f"({time.time() - started:.1f} s)")
            row["steps"] = steps
            row["first_divergent_step"] = first_divergent
            row["bit_identical"] = first_divergent is None
            out["covered"].append(row)
        finally:
            if handle is not None:
                handle.close()
            for sub_step, function in originals.items():
                setattr(driver_module, sub_step, function)
            reference.close()
            fused.close()
            cp.get_default_memory_pool().free_all_blocks()
    return out


def leg_bench(cp, res: int, n: int, steps: int, warmup: int) -> Dict[str, Any]:
    """Array path vs DERIVE=0 vs DERIVE=1, sub-step and whole step, one device."""
    import cases  # noqa: PLC0415

    from meep_gpu import backends, driver as driver_module, stepping  # noqa: PLC0415
    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415
    from meep_gpu.triton_kernels.no_pml import plan_plain_curl, plan_plain_step  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    driver = lift_simulation(cases.BUILDERS["2d_plain"](res=res, n=n),
                             prefer_gpu=True, gpu_id=0)
    out: Dict[str, Any] = {"res": res, "n": n, "steps": steps,
                           "shape": [int(s) for s in driver.shape]}
    cells = int(np.prod(driver.shape))
    out["cells"] = cells
    try:
        plan = plan_plain_step(driver.fields, driver.pml)
        fields, pml = driver.fields, driver.pml

        def timed(fn, label):
            for _ in range(warmup):
                fn()
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                fn()
            cp.cuda.runtime.deviceSynchronize()
            dt = time.perf_counter() - t0
            rate = cells * steps / dt / 1e6
            log(f"  bench {label}: {dt * 1e3 / steps:.4f} ms/step  "
                f"{rate:.1f} Mcell-steps/s")
            return {"ms_per_call": dt * 1e3 / steps, "mcell_steps_per_s": rate}

        out["array_step_B"] = timed(lambda: stepping.step_B(fields, pml), "array step_B")
        out["array_step_D"] = timed(lambda: stepping.step_D(fields, pml), "array step_D")
        for sub_step, entry in plan.plans.items():
            out[f"triton_{sub_step}_derive{entry.derive}"] = timed(
                entry.run, f"triton {sub_step} derive={entry.derive}")
        # DERIVE=0 comparison for step_B needs stored E, which this run has not
        # got; report that rather than fabricate it.
        out["derive0_step_B"] = ("not measurable on 2d_plain: stores_E is False, "
                                 "so the array path derives and there is no stored "
                                 "E to bind")

        names = ("step_B", "step_D")
        originals = {n: getattr(driver_module, n) for n in names}
        try:
            out["whole_step_array"] = timed(driver.step, "whole step, array path")

            def make(sub_step):
                def wrapper(f, p=None):
                    entry = plan.plans.get(sub_step) if f is fields else None
                    if entry is None:
                        return originals[sub_step](f, p)
                    entry.run()
                    return None
                return wrapper
            for sub_step in names:
                setattr(driver_module, sub_step, make(sub_step))
            out["whole_step_triton"] = timed(driver.step, "whole step, Triton")
        finally:
            for sub_step, function in originals.items():
                setattr(driver_module, sub_step, function)
        a = out["whole_step_array"]["mcell_steps_per_s"]
        b = out["whole_step_triton"]["mcell_steps_per_s"]
        out["whole_step_speedup"] = b / a
        log(f"  whole-step speedup: {b / a:.3f}x")
    finally:
        driver.close()
        cp.get_default_memory_pool().free_all_blocks()
    return out


def environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "python": sys.version.split()[0],
                           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for module in ("triton", "cupy", "numpy"):
        try:
            env[module] = __import__(module).__version__
        except Exception as exc:  # noqa: BLE001
            env[module] = f"unavailable: {exc}"
    try:
        env["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        env["nvidia_smi"] = f"unavailable: {exc}"
    return env


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Raise unless every requested correctness leg meets its release contract.

    A gate that merely writes divergent numbers and exits zero is a recorder, not
    a gate. Keep this host-only so the laptop suite can mutation-test the harness
    without importing CuPy or Triton.
    """
    legs = set(payload.get("legs", ()))
    failures: List[str] = []

    if "product" in legs:
        guarded = payload.get("product_guarded", {})
        unguarded = payload.get("product_unguarded", {})
        live = int(guarded.get("n_live", 0))
        if (live <= 0 or int(guarded.get("n_identical", -1)) != live
                or int(guarded.get("worst_differing_floats", -1)) != 0
                or guarded.get("all_targets_changed") != f"{live}/{live}"):
            failures.append(f"guarded product did not pass every live case: {guarded}")
        control_live = int(unguarded.get("n_live", 0))
        if (control_live <= 0
                or int(unguarded.get("n_identical", control_live)) >= control_live):
            failures.append(f"unguarded product control caught no divergence: {unguarded}")

    if "multistep" in legs:
        cases = payload.get("multistep", {}).get("cases", ())
        if not cases or not all(bool(row.get("bit_identical")) for row in cases):
            failures.append("one or more multistep configurations diverged")

    if "mutations" in legs:
        rows = {row.get("mutation"): row
                for row in payload.get("mutations", {}).get("mutations", ())}
        real_defects = {
            "flatten_grouping", "add_instead_of_subtract", "drop_inverse_epsilon",
            "derive_centre_only", "drop_ownership_mask", "periodic_ghost_on_metallic",
        }
        missing = sorted((real_defects | {"swap_derive_operand_order"}) - set(rows))
        if missing:
            failures.append(f"mutation rows are missing: {missing}")
        for name in sorted(real_defects & set(rows)):
            if int(rows[name].get("n_caught", 0)) <= 0:
                failures.append(f"real mutation {name} caught no live case")
        sign = rows.get("add_instead_of_subtract", {})
        if int(sign.get("n_live", 0)) <= 0 or sign.get("n_caught") != sign.get("n_live"):
            failures.append(f"sign mutation was not caught universally: {sign}")
        commutative = rows.get("swap_derive_operand_order", {})
        if int(commutative.get("n_caught", -1)) != 0:
            failures.append(
                f"expected-uncaught commutativity control unexpectedly differed: {commutative}")

    if "engine" in legs:
        engine = payload.get("engine", {})
        covered = engine.get("covered", ())
        for row in covered:
            # Post null-family wiring (planner recert 2026-08-14): a no-PML
            # grid's certified plan is the plain curl pair PLUS the null
            # constitutive slots, so ``replaces`` carries all four names.
            # ``selected`` here records the CURL slots only (the leg
            # monkeypatches step_B/step_D); the run-time assertion above this
            # summary separately pins update_H/update_E to
            # NullConstitutivePlan, so this contract stays exact rather than
            # order-sensitive or weakened.
            if (not row.get("bit_identical") or row.get("first_divergent_step") is not None
                    or int(row.get("steps", 0)) < 12
                    or sorted(row.get("replaces", ())) != [
                        "step_B", "step_D", "update_E", "update_H"]
                    or row.get("selected") != {
                        "step_B": "PlainCurlPlan", "step_D": "PlainCurlPlan"}):
                failures.append(f"central no-PML route failed its exact contract: {row}")
        if not covered:
            failures.append("central no-PML route ran no covered case")
        refusals = engine.get("refusals", ())
        for row in refusals:
            if not all(bool(value) for value in row.get("plan_is_none", {}).values()):
                failures.append(f"direct no-PML builder admitted a refusal case: {row}")
            if "PlainCurlPlan" in row.get("central_selected", {}).values():
                failures.append(f"central planner selected no-PML for a refusal case: {row}")
        if not refusals:
            failures.append("central route ran no refusal case")

    if failures:
        raise AssertionError("; ".join(failures))
    return {
        "status": "passed",
        "legs": sorted(legs),
        "guarded_product": payload.get("product_guarded", {}).get("identical"),
        "unguarded_control": payload.get("product_unguarded", {}).get("identical"),
        "multistep": payload.get("multistep", {}).get("identical"),
        "engine_covered": len(payload.get("engine", {}).get("covered", ())),
        "engine_refusals": len(payload.get("engine", {}).get("refusals", ())),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join(HERE, "no_pml_gate"))
    parser.add_argument("--leg", action="append", default=None)
    parser.add_argument("--source", default=SHIPPED_SOURCE)
    parser.add_argument("--engine-steps", type=int, default=12)
    parser.add_argument("--budget-steps", type=int, default=3000)
    parser.add_argument("--bench-res", type=int, default=40)
    parser.add_argument("--bench-n", type=int, default=16)
    parser.add_argument("--bench-steps", type=int, default=200)
    parser.add_argument("--bench-warmup", type=int, default=20)
    args = parser.parse_args(argv)

    legs = args.leg or ["product", "multistep", "mutations", "engine"]
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "gate.json")
    payload: Dict[str, Any] = {"environment": environment(), "legs": legs,
                               "source_sha256": sha256_file(args.source)}
    save(payload, out_path)

    import cupy as cp  # noqa: PLC0415

    if "product" in legs:
        # THE PARAMETER IS enable_fp_fusion, NOT "is the guard on".
        # ``PmlCurlPlan.run(guard=...)`` (launch.py:170-189) forwards its argument
        # straight to ``enable_fp_fusion``, so guard=False is the GUARDED run and
        # guard=True is the unguarded control. Spelled out because this gate ran
        # once with the two labels swapped and reported a 56/144 "guarded" pass.
        log("LEG product (GUARDED: enable_fp_fusion=False)")
        payload["product_guarded"] = leg_product(cp, guard=False)
        save(payload, out_path)
        log("LEG product (UNGUARDED control: enable_fp_fusion=True)")
        payload["product_unguarded"] = leg_product(cp, guard=True)
        save(payload, out_path)
    if "multistep" in legs:
        log("LEG multistep")
        payload["multistep"] = leg_multistep(cp)
        save(payload, out_path)
    if "mutations" in legs:
        log("LEG mutations")
        payload["mutations"] = leg_mutations(cp, args.source)
        save(payload, out_path)
    if "engine" in legs:
        log("LEG engine")
        payload["engine"] = leg_engine(cp, args.engine_steps)
        save(payload, out_path)
    if "budget" in legs:
        log(f"LEG budget ({args.budget_steps} steps)")
        payload["budget"] = leg_engine(
            cp, args.budget_steps,
            jsonl=os.path.join(args.out_dir, "budget.jsonl"))
        save(payload, out_path)
    if "bench" in legs:
        log("LEG bench")
        payload["bench"] = leg_bench(cp, args.bench_res, args.bench_n,
                                     args.bench_steps, args.bench_warmup)
        save(payload, out_path)

    payload["validation"] = validate_payload(payload)
    save(payload, out_path)
    log(f"GATE PASSED: {payload['validation']}")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
