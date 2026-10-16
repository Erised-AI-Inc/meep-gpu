"""Byte gate for the COMPLEX NO-ABSORBER curl arm (``complex_no_pml_curl``).

DEVICE RESULT (2026-08-17), ON THE PREVIOUS BINDING. This gate released on the GPU
host's RTX A6000 with the ``keep`` policy installed: 8/8 shipped product rows were
byte-identical over eight launches, all three controls and all three mutations
diverged, and the predicate partition agreed on every refusal row. Its adversarial
signed-zero leg recorded the previous binding's residual (the recurrence on every
cell with a per-launch zeroed auxiliary): step_D differed by 86,450/281,250 words in
3-D and 3,552/11,250 in 2-D. step_B read identical only because that leg's
uniform-source loop overwrote the B targets it had just set to -0, so it never
planted step_B. The tracked provenance is
``triton_complex_no_pml_curl_device_gate`` in
``meep_gpu/triton_kernels/fingerprints.json``, re-cut at certification.

THE PRESENT BINDING (``complex_no_pml_curl``'s module docstring): the absorber-free
chunk record (no membership run, the grid's presence word) and unit columns, so the
kernel's dsig-only form ``((f * 1) - curl) * 1`` steps every cell, which IS MEEP's
plain ``f - curl`` on every word. The planted leg therefore REQUIRES the -0 target
with an exact +0 curl to come back identical, on both sub-steps, and the previous
binding is armed against it as a mutation that must be caught there.

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
  controls    PHASEOFF (every connection kind none) and STALESOURCE, which MUST
              diverge. A null beside a dead control is not evidence — group (G)
              measured IDENTICAL over four cycles and was wrong.
  mutations   deliberately broken bindings of the SHIPPED plan, each of which must
              be CAUGHT on the product leg: a record marking every cell a
              recurrence cell (the auxiliary then carries the previous launch's
              ``n``), the presence word with every bit set on the 2-D row (the full
              bracket over the invariant axis's partner), and a unit column one ulp
              off.
  nulls       ``aux_poisoned``: the placeholder auxiliary filled with NaN before
              every launch. The kernel reads it on recurrence cells only, of which
              the absorber-free record has none, so no byte may move.
  planted     the ``minus_zero_target_zero_curl`` rows: target ``-0.0`` with every
              source uniform (an exact ``+0`` curl inside the window), on both
              sub-steps and both row shapes. The shipped binding must store the
              array path's words with the ``-0`` among them; the previous binding
              (recurrence everywhere, auxiliary zeroed per launch) must diverge.
  refusals    the predicate's own answers on the adjacent configurations, so an
              over-broad clause is caught here and not in a later census.

EVERY MEASUREMENT PRIMITIVE IS IMPORTED from ``probe_residual_group_bodies``
unmodified — the uint32 whole-inventory comparison, the per-cycle bracketed
vacuity gate, the live signed-zero lattice and its restore, the every-volume
perturbation, and the complex seeding that writes BOTH halves. The 2026-08-15
adversarial audit found four defects in the superseded measurement and repaired
them there; running a copy is how one gets re-introduced.

THE POLICY IS INSTALLED, NOT NAMED. ``MEEP_GPU_SUBNORMAL_POLICY`` in the
environment declares a policy and installs nothing; this gate calls
``install_subnormal_policy('keep')`` before the first CuPy elementwise op or Triton
compile, and records what it installed.

THE SINGLE COMMAND THE NEXT DEVICE VISIT RUNS::

    cd <repo>/parity/meep_gpu && \\
    MEEP_GPU_SUBNORMAL_POLICY=keep \\
    CUDA_VISIBLE_DEVICES=<one idle GPU> PYTHONPATH=<repo>:. \\
    python -u gate_triton_complex_no_pml_curl.py \\
        --out results/complex_no_pml_curl_<UTCSTAMP>/gate.json --cycles 8

Progress reporting: one flushed line per case, and one fsync'd JSONL row per case written as
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
    """One flushed line, to stdout AND to the run's own log (progress reporting)."""
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

def shipped_plan(fields: Any, pml: Any, sub_step: str, drop_layer: bool = False):
    """``plan_complex_no_pml_curl`` and nothing else — the point of this gate.

    A ``None`` here is a GATE FAILURE, not a refusal to be tolerated: the
    predicate is exercised separately in the ``refusals`` leg, and a product leg
    that silently measured nothing is the vacuity this campaign keeps auditing
    for.
    """
    from meep_gpu.triton_kernels import complex_no_pml_curl as arm

    plan = arm.plan_complex_no_pml_curl(fields, None if drop_layer else pml,
                                        sub_step)
    if plan is None:
        raise RuntimeError(
            f"the SHIPPED builder refused {sub_step} on the target configuration "
            f"(drop_layer={drop_layer}); this leg measures nothing without a plan")
    return plan


def phase_off_plan(fields: Any, pml: Any, sub_step: str):
    """CONTROL: every axis's connection kind set to none (the stored word itself).

    A live control on the exact feature that makes group (J) complex. On the
    previous hand-built binding it diverged 7350 / 281250, on the wrapped lanes only.
    Only the connection constexprs move; the shipped plan's pointers are kept.
    """
    plan = shipped_plan(fields, pml, sub_step)
    plan._inner.connections = (0, 0, 0)
    plan._inner.connection_values = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    return plan


def stale_source_plan(fields: Any, pml: Any, sub_step: str):
    """CONTROL: sources bound as a plan-time SNAPSHOT.

    Group (G)'s false-pass mechanism, re-armed. It diverged 93750 / 281250 on the
    hand-built binding and must diverge here too, or this gate cannot tell a live
    source binding from a dead one.
    """
    import cupy as cp
    from meep_gpu.triton_kernels import complex_fields as cx
    from meep_gpu.triton_kernels import complex_no_pml_curl as arm

    plan = shipped_plan(fields, pml, sub_step)
    inner = plan._inner
    frozen = [array.copy() for array in arm.source_arrays(fields, sub_step)]
    inner._sources = tuple(cx.CupyPointer(cx._word_view(a)) for a in frozen)
    assert isinstance(frozen[0], cp.ndarray)
    return plan


def rebind_record(plan, edit) -> None:
    """Rebind the shipped plan's chunk record after ``edit(record)``, past its builder.

    ``launch.bind_chunk_bounds`` still checks the edited record, so a mutation can
    only plant a record the plan would accept from a harness.
    """
    from meep_gpu.chunk_map import RECORD_PRESENCE_WORD, record_index
    from meep_gpu.triton_kernels.launch import bind_chunk_bounds

    inner = plan._inner
    record = np.array(inner.chunk_bounds, dtype=np.int32, copy=True)
    edit(record, record_index, RECORD_PRESENCE_WORD, inner.shape)
    like = inner._targets[0].array
    inner.chunk_bounds, inner._cb = bind_chunk_bounds(
        record, inner.shape, like, "gate mutation")


def _every_cell_a_recurrence(record, record_index, presence_word, shape) -> None:
    for half in (False, True):
        for axis, n in enumerate(shape):
            if record[presence_word] >> axis & 1:
                record[record_index(half, axis, 0)] = 0
                record[record_index(half, axis, 1)] = n


def record_with_a_run_plan(fields: Any, pml: Any, sub_step: str):
    """MUTATION: a record marking every cell a recurrence cell on every present axis.

    The recurrence then runs on the plan's placeholder auxiliary, which carries the
    previous launch's ``n``: identical-looking on cycle 1, wrong on every cycle after.
    """
    plan = shipped_plan(fields, pml, sub_step)
    rebind_record(plan, _every_cell_a_recurrence)
    return plan


def presence_all_present_plan(fields: Any, pml: Any, sub_step: str):
    """MUTATION: every presence bit set. On the 2-D row the invariant z axis's partner
    then takes the full bracket, ``((g1s - g1) + g2) - g2``, which rounds twice more."""
    plan = shipped_plan(fields, pml, sub_step)

    def edit(record, record_index, presence_word, shape):
        record[presence_word] = 7

    rebind_record(plan, edit)
    return plan


class _AuxiliaryWrapped:
    """The shipped inner plan, with one thing done to its auxiliary before each launch."""

    def __init__(self, inner: Any, auxiliaries, value) -> None:
        self.sub_step = inner.sub_step
        self._inner = inner
        self._auxiliaries = auxiliaries
        self._value = value

    def run(self, guard: Optional[bool] = None) -> None:
        for auxiliary in self._auxiliaries:
            auxiliary.fill(self._value)
        self._inner.run(guard)


def aux_poisoned_plan(fields: Any, pml: Any, sub_step: str):
    """DECLARED NULL: the placeholder auxiliary is NaN at every launch.

    The kernel loads and stores ``fu`` on recurrence cells only (the auxiliary
    contract), and the absorber-free record has none, so no word may change.
    """
    plan = shipped_plan(fields, pml, sub_step)
    plan._inner = _AuxiliaryWrapped(plan._inner, plan._auxiliaries, float("nan"))
    return plan


def previous_binding_plan(fields: Any, pml: Any, sub_step: str):
    """MUTATION (the inversion): the previous binding, recurrence on every cell with
    the auxiliary zeroed before each launch. It loses ``(-0, +0)``: the planted leg's
    pattern, where it must be caught."""
    plan = shipped_plan(fields, pml, sub_step)
    rebind_record(plan, _every_cell_a_recurrence)
    plan._inner = _AuxiliaryWrapped(plan._inner, plan._auxiliaries, 0)
    return plan


def unit_columns_perturbed_plan(fields: Any, pml: Any, sub_step: str):
    """MUTATION: one synthesized coefficient column is 1.0 + 1 ulp, not 1.0.

    The binding's whole premise is that the columns are EXACTLY one. A column
    that is merely close must be caught, or the gate would license a near-unit
    binding as a degenerate one.
    """
    plan = shipped_plan(fields, pml, sub_step)
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
# The planted row — the pattern the previous binding lost, now required identical
# ---------------------------------------------------------------------------

PLANTED_BUILDERS = {"shipped": ("IDENTICAL", None),
                    "previous_binding": ("DIVERGENT", None)}


def leg_planted(handle) -> Dict[str, Any]:
    """Target ``-0.0`` with an exact ``+0`` curl: MEEP's plain form keeps the ``-0``.

    Seeded into the TARGET volumes directly, with every SOURCE volume uniform (a
    uniform field has an identically zero curl inside the window, so the arithmetic
    reaching the form is the shipped one; plane 0 of a phased axis connects to a
    rotated word and is not a planted cell). The targets are excluded from the
    uniform fill: the previous leg's fill overwrote the B targets, so it never
    planted ``step_B``.

    ``shipped`` rows must be IDENTICAL and store the ``-0``; ``previous_binding``
    rows (recurrence everywhere, auxiliary zeroed per launch) must DIVERGE.
    """
    import cupy as cp
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    builders = {"shipped": shipped_plan, "previous_binding": previous_binding_plan}
    out: Dict[str, Any] = {"leg": "planted", "cases": []}
    for tag, config in (("J3", J_3D), ("J2", J_2D)):
        for sub_step in ("step_B", "step_D"):
            for label, builder in builders.items():
                prediction = PLANTED_BUILDERS[label][0]
                reference_fields, pml, _ = P.build(cp, **config)
                candidate_fields, _, _ = P.build(cp, **config)
                shape = tuple(reference_fields.grid.shape)
                minus_zero = np.empty(shape, dtype=np.complex64)
                minus_zero.real = np.float32(-0.0)
                minus_zero.imag = np.float32(-0.0)
                uniform = np.empty(shape, dtype=np.complex64)
                uniform.real = np.float32(0.25)
                uniform.imag = np.float32(-0.125)
                targets = SUB_STEPS[sub_step]["targets"]
                for fields in (reference_fields, candidate_fields):
                    for name in targets:
                        getattr(fields, name)[...] = cp.asarray(minus_zero)
                    for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
                        array = getattr(fields, name, None)
                        if array is not None and name not in targets:
                            array[...] = cp.asarray(uniform)
                case = f"{tag}_{sub_step}_minus_zero_target_zero_curl:{label}"
                try:
                    plan = builder(candidate_fields, pml, sub_step)
                    SUB_STEP_FUNCTIONS[sub_step](reference_fields, pml)
                    plan.run()
                    verdict = P.compare(P.inventory(reference_fields),
                                        P.inventory(candidate_fields))
                    stored = sum(int((P.as_words(getattr(reference_fields, name))
                                      == 0x80000000).sum()) for name in targets)
                    row = {"case": case, "binding": label, "prediction": prediction,
                           "differing": verdict["differing"],
                           "compared": verdict["words"],
                           "identical": verdict["identical"],
                           "negative_zero_words_stored": stored,
                           "verdict": "IDENTICAL" if verdict["identical"] else "DIVERGENT",
                           "first_difference": verdict.get("first_difference")}
                except BaseException as exc:  # noqa: BLE001
                    row = {"case": case, "binding": label, "prediction": prediction,
                           "error": f"{type(exc).__name__}: {exc}"[:300],
                           "traceback": traceback.format_exc()[-800:]}
                out["cases"].append(row)
                log(handle, f"  planted {row['case']}: "
                            f"{row.get('differing', 'ERROR')} / "
                            f"{row.get('compared', '?')} differing, -0 stored "
                            f"{row.get('negative_zero_words_stored', '?')} "
                            f"(predict {prediction})")
    return out


def leg_refusals(handle) -> Dict[str, Any]:
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
                fields, None if drop else pml, sub_step)
            incumbent = cx.complex_pml_curl_coverage(
                fields, None if drop else pml, sub_step)
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

def build_cases() -> List[Dict[str, Any]]:
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
                                    shipped_plan(f, p, s, drop_layer=d)),
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
                            phase_off_plan(f, p, s)),
        })
    cases.append({
        "name": "J3_step_B_STALESOURCE_CONTROL", "group": "J",
        "sub_step": "step_B", "prediction": "DIVERGENT",
        "clause": "control: sources bound as a plan-time snapshot (group G)",
        "make": make3, "array_path": stepping.step_B,
        "kernel_plan": lambda f, p: stale_source_plan(f, p, "step_B"),
    })
    def make2(offset):
        import cupy as cp

        return P.build(cp, seed_offset=offset, **J_2D)

    for sub_step in ("step_B", "step_D"):
        cases.append({
            "name": f"J3_{sub_step}_MUTATION_record_with_a_run", "group": "J",
            "sub_step": sub_step, "prediction": "DIVERGENT",
            "clause": "mutation: a record marking every cell a recurrence cell",
            "make": make3, "array_path": SUB_STEP_FUNCTIONS[sub_step],
            "kernel_plan": (lambda f, p, s=sub_step:
                            record_with_a_run_plan(f, p, s)),
        })
        cases.append({
            "name": f"J2_{sub_step}_MUTATION_presence_all_present", "group": "J",
            "sub_step": sub_step, "prediction": "DIVERGENT",
            "clause": "mutation: every presence bit set on the 2-D row",
            "make": make2, "array_path": SUB_STEP_FUNCTIONS[sub_step],
            "kernel_plan": (lambda f, p, s=sub_step:
                            presence_all_present_plan(f, p, s)),
        })
        cases.append({
            "name": f"J3_{sub_step}_NULL_aux_poisoned", "group": "J",
            "sub_step": sub_step, "prediction": "IDENTICAL",
            "clause": "declared null: the placeholder auxiliary is NaN at every launch",
            "make": make3, "array_path": SUB_STEP_FUNCTIONS[sub_step],
            "kernel_plan": (lambda f, p, s=sub_step:
                            aux_poisoned_plan(f, p, s)),
        })
    cases.append({
        "name": "J3_step_B_MUTATION_unit_column_one_ulp", "group": "J",
        "sub_step": "step_B", "prediction": "DIVERGENT",
        "clause": "mutation: a synthesized coefficient column is 1.0 + 1 ulp",
        "make": make3, "array_path": stepping.step_B,
        "kernel_plan": lambda f, p: unit_columns_perturbed_plan(f, p, "step_B"),
    })
    return cases


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The release contract. Refusals are BY NAME; nothing is inferred from absence."""
    reasons: List[str] = []
    wanted = payload.get("subnormal_policy_requested", "keep")
    if payload.get("subnormal_policy_installed") != wanted:
        reasons.append(f"the {wanted!r} subnormal policy was not INSTALLED (an env var "
                       f"alone declares a policy and installs nothing)")

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
    if len(mutations) < 5:
        reasons.append(f"{len(mutations)} mutations ran; this gate requires 5")
    if any(r.get("verdict") != "DIVERGENT" for r in mutations):
        reasons.append("a mutation was NOT caught")
    nulls = [r for r in rows if "NULL" in str(r.get("case"))]
    if len(nulls) < 2:
        reasons.append(f"{len(nulls)} declared nulls ran; this gate requires 2")
    if any(r.get("verdict") != "IDENTICAL" for r in nulls):
        reasons.append("a declared null moved a byte: the auxiliary was read")

    planted = payload.get("planted") or {}
    if not planted.get("cases"):
        reasons.append("the planted (-0.0 target, +0.0 curl) leg did not run")
    shipped_rows = [r for r in planted.get("cases", []) if r.get("binding") == "shipped"]
    previous_rows = [r for r in planted.get("cases", [])
                     if r.get("binding") == "previous_binding"]
    if planted.get("cases") and (len(shipped_rows) < 4 or len(previous_rows) < 4):
        reasons.append(f"the planted leg ran {len(shipped_rows)} shipped and "
                       f"{len(previous_rows)} previous-binding rows; it requires 4 each")
    for row in planted.get("cases", []):
        if "differing" not in row:
            reasons.append(f"{row.get('case')}: no differing count recorded "
                           f"({row.get('error', 'no error recorded')})")
        elif row.get("verdict") != row.get("prediction"):
            reasons.append(f"{row.get('case')}: planted verdict {row.get('verdict')} != "
                           f"prediction {row.get('prediction')}")
        elif row.get("binding") == "shipped" and not row.get("negative_zero_words_stored"):
            reasons.append(f"{row.get('case')}: no -0 word was stored; the planted row "
                           f"measured nothing")

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
    parser.add_argument("--subnormal-policy", default="keep", choices=("keep", "flush"),
                        help="the float32 subnormal policy to install before the first device compile (keep, the default, or flush: the certification round gates every family under both on the NVIDIA tables)")
    parser.add_argument("--leg", default="all",
                        choices=("all", "product", "planted", "refusals"))
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

    payload["subnormal_policy_requested"] = args.subnormal_policy
    install_subnormal_policy(args.subnormal_policy)
    payload["subnormal_policy_installed"] = args.subnormal_policy

    log(handle, "policy installed: keep")

    cases: List[Dict[str, Any]] = []
    if args.leg in ("all", "product"):
        built = build_cases()
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

    if args.leg in ("all", "planted"):
        payload["planted"] = leg_planted(handle)
    if args.leg in ("all", "refusals"):
        payload["refusals"] = leg_refusals(handle)

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
