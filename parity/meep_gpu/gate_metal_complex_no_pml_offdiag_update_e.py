#!/usr/bin/env python3
"""Native MPS gate for complex/Bloch no-PML tensor-row ``update_E``.

RULING R7: SOFTWARE BINARY64. The kernel forms the array path's double row per part
(``stepping._offdiagonal_terms``, ``_rounded_row``) through ``metal_kernels.softfp64``,
connects a Bloch phase on the same-index operands' plane 0 as the array path does
(``stepping._bloch_ghost_operand``: a copy, a per-part negation or the unfused
product), and reads the per-chunk presence words; its arm is wired. Every plan is the
ENGINE builder's (``plan_metal_complex_no_pml_offdiag``) with the one-rank record of
the no-absorber division. The gate runs:

* ``product``: Bloch-phased, unphased, walled, collapsed and NEGATED-connection cases
  on GENERIC states (where a float32 row would differ) against
  ``stepping.update_E`` over ``--cycles`` launches;
* ``mutation_control``: each mutated case's own specialisation (read off the engine's
  plan), unmutated, through the same seam the mutants take: 0 words;
* ``mutation``: on those states, each must be caught -- the first term dropped, the
  partner connection dropped, the own-axis connection dropped, the partner connection
  on the shifted read (the pre-alignment placement), the float32 row (ruling R7's
  mutation, the pre-R7 spelling), and on the negated case the negation spelled as a
  product by the placeholder word the plan binds;
* ``engine``: every case admitted with no ruling clause, the arm wired, the builder's
  presence words all-kept (a no-absorber run is one chunk unless a division is
  installed; the words' per-chunk rule is the real family's and is gated there).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms, complex_no_pml_offdiag_update_e as family, offdiag_update_e as offdiag,
    shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

PARITY_ROOT = Path(__file__).resolve().parent
if str(PARITY_ROOT) not in sys.path:
    sys.path.insert(0, str(PARITY_ROOT))

from gate_chunk_records import layer_record  # noqa: E402


def _words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _different(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def _build(*, seed: int, cell_size: Sequence[float], phase: Sequence[float],
           boundaries: Sequence[str]) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=tuple(cell_size), courant=0.35,
                boundaries=tuple(boundaries), k_point=tuple(phase), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = np.full(shape, np.float32(2.25))
    inverse = np.full(shape, np.float32(1.0 / 2.25))
    rows = {
        "Ex": {"Ey": (0.04 * rng.standard_normal(shape)).astype(np.float32),
               "Ez": (0.03 * rng.standard_normal(shape)).astype(np.float32)},
        "Ey": {"Ez": (0.02 * rng.standard_normal(shape)).astype(np.float32),
               "Ex": (0.05 * rng.standard_normal(shape)).astype(np.float32)},
        "Ez": {"Ex": (0.01 * rng.standard_normal(shape)).astype(np.float32),
               "Ey": (0.06 * rng.standard_normal(shape)).astype(np.float32)},
    }
    fields.set_epsilon_volumes(
        {name: epsilon for name in ("Ex", "Ey", "Ez")},
        {name: inverse for name in ("Ex", "Ey", "Ez")}, rows)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        getattr(fields, name)[...] = (
            rng.uniform(-0.4, 0.4, shape).astype(np.float32)
            + 1j * rng.uniform(-0.4, 0.4, shape).astype(np.float32))
    return fields, PML(grid=grid, thickness=0)


def _run(*, seed: int, cycles: int, cell_size: Sequence[float], phase: Sequence[float],
         boundaries: Sequence[str], functions: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    reference, reference_pml = _build(
        seed=seed, cell_size=cell_size, phase=phase, boundaries=boundaries)
    actual, actual_pml = _build(
        seed=seed, cell_size=cell_size, phase=phase, boundaries=boundaries)
    names = ("Ex", "Ey", "Ez")
    before = {name: np.array(getattr(actual, name), copy=True) for name in names}
    residency = Residency()
    plan = family.plan_metal_complex_no_pml_offdiag(
        actual, actual_pml, residency, functions=functions,
        chunk_bounds=layer_record(actual.grid, None))
    if plan is None:
        return {"passed": False, "reason": "a declared complex tensor product was refused"}
    residency.sync_in()
    per_cycle = []
    import torch

    for cycle in range(1, cycles + 1):
        plan.run()
        torch.mps.synchronize()
        stepping.update_E(reference, reference_pml)
        residency.sync_out(names)
        difference = sum(_different(getattr(reference, name), getattr(actual, name))
                         for name in names)
        census = sum(subnormal.census(getattr(reference, name)) for name in names)
        per_cycle.append({"cycle": cycle, "differing_words": difference,
                          "reference_subnormals": census})
        if difference or census:
            break
    moved = sum(_different(before[name], getattr(actual, name)) for name in names)
    return {
        "passed": (len(per_cycle) == cycles
                   and all(row["differing_words"] == row["reference_subnormals"] == 0
                           for row in per_cycle)
                   and moved > 0 and plan.launches == len(per_cycle)),
        "per_cycle": per_cycle,
        "differing_words": per_cycle[-1]["differing_words"],
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "moved_words": moved, "launches": plan.launches,
        "mirrors": len(residency.names),
    }


def _case_source(*, seed: int, cell_size: Sequence[float], phase: Sequence[float],
                 boundaries: Sequence[str]) -> str:
    """The SHIPPED source of one case's own specialisation, read off the engine's plan
    (row mask, codes, walls, connection kinds): every mutant is this text
    with one defect planted, and this text unmutated is the control."""
    fields, pml = _build(seed=seed, cell_size=cell_size, phase=phase,
                         boundaries=boundaries)
    plan = family.plan_metal_complex_no_pml_offdiag(
        fields, pml, Residency(), contract_variants=(),
        chunk_bounds=layer_record(fields.grid, None))
    assert plan is not None, "a declared complex tensor product was refused"
    return family.complex_no_pml_offdiag_source(
        plan.row_mask, plan.codes, plan.walls, plan.phased)


def _needle(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"mutation needle {old!r} matches {text.count(old)} times")
    return text.replace(old, new)


def _compiled(source: str) -> Dict[str, Any]:
    return {shaders.CONTRACT_OFF: compile_source(source).complex_no_pml_offdiag_update_e}


#: Component 0's two-term double row per part, as the shipped source spells it.
_ROW0 = (
    "    ulong od_re0 = sd_widen_f(od_d0.x, true);\n"
    "    ulong od_im0 = sd_widen_f(od_d0.y, true);\n"
    "    if (od_kept_00) { od_re0 = sd_add(od_re0, sd_scale_pow2(sd_widen_f(sum_00.x, "
    "true), -2, true), true); }\n"
    "    if (od_kept_00) { od_im0 = sd_add(od_im0, sd_scale_pow2(sd_widen_f(sum_00.y, "
    "true), -2, true), true); }\n"
    "    if (od_kept_01) { od_re0 = sd_add(od_re0, sd_scale_pow2(sd_widen_f(sum_01.x, "
    "true), -2, true), true); }\n"
    "    if (od_kept_01) { od_im0 = sd_add(od_im0, sd_scale_pow2(sd_widen_f(sum_01.y, "
    "true), -2, true), true); }\n"
    "    float2 src0 = float2(sd_narrow_f(od_re0, true), sd_narrow_f(od_im0, true));")


def _mutants(phased_base: str, negated_base: str) -> Dict[str, Tuple[str, Mapping[str, Any]]]:
    """``label -> (case, functions)``. Each needle must match exactly once, or the leg
    is disarmed and raises. The phased case is ``two_axis_bloch`` (Ex <- Ey on the
    phased y axis, Ex's own x axis phased); the negated case puts -1 on x."""
    phased = {
        "first_tensor_row_dropped": _needle(
            _needle(phased_base,
                    "    if (od_kept_00) { od_re0 = sd_add(od_re0, sd_scale_pow2("
                    "sd_widen_f(sum_00.x, true), -2, true), true); }\n", ""),
            "    if (od_kept_00) { od_im0 = sd_add(od_im0, sd_scale_pow2("
            "sd_widen_f(sum_00.y, true), -2, true), true); }\n", ""),
        # The partner pair's same-index word on the partner axis's plane 0.
        "partner_connection_dropped": _needle(
            phased_base, "    home_00 = (j == 0) ? c_mul(home_00, dpy) : home_00;\n", ""),
        # The formed product on the component's own axis's plane 0.
        "own_connection_dropped": _needle(
            phased_base,
            "    near_product_00 = (i == 0) ? c_mul(near_product_00, upx) : "
            "near_product_00;\n", ""),
        # The pre-alignment placement: the phase on the shifted read instead.
        "partner_connection_on_the_shifted_read": _needle(
            phased_base,
            "    home_00 = (j == 0) ? c_mul(home_00, dpy) : home_00;\n"
            "    float2 down_00 = dvy ? g1[i * nyz + dj * nzi + k] : float2(0.0f, 0.0f);",
            "    float2 down_00 = dvy ? g1[i * nyz + dj * nzi + k] : float2(0.0f, 0.0f);\n"
            "    down_00 = (j == 0) ? c_mul(down_00, dpy) : down_00;"),
        # RULING R7's mutation, per part: component 0's row in float32.
        "float32_row": _needle(
            phased_base, _ROW0,
            "    float2 src0 = od_d0 + (c_mul_coefficient_left(0.25f, sum_00) + "
            "c_mul_coefficient_left(0.25f, sum_01));"),
    }
    negated = {
        # A phase of -1 is MEEP's negation; the plan binds the placeholder (1, 0) for
        # that kind, so a product by it is a dropped negation.
        "negation_spelled_as_a_product_by_the_placeholder": _needle(
            negated_base,
            "    near_product_00 = (i == 0) ? -near_product_00 : near_product_00;",
            "    near_product_00 = (i == 0) ? c_mul(near_product_00, upx) : "
            "near_product_00;"),
    }
    out = {label: ("phased", _compiled(source)) for label, source in phased.items()}
    out.update({label: ("negated", _compiled(source)) for label, source in negated.items()})
    return out


def leg_engine(products) -> Dict[str, Any]:
    """Every case admitted with no ruling clause; the arm wired; the builder's words
    all-kept and ``offdiag_update_e.presence_for``'s."""
    arm = next(spec for spec in arms.registered(family.SLOT)
               if spec.family == family.FAMILY)
    rows = []
    for index, (label, cell_size, phase, boundaries) in enumerate(products, 1):
        fields, pml = _build(seed=36000 + index, cell_size=cell_size, phase=phase,
                             boundaries=boundaries)
        residency = Residency()
        record = layer_record(fields.grid, None)
        verdict = family.metal_complex_no_pml_offdiag_coverage(
            fields, pml, residency, record)
        plan = family.plan_metal_complex_no_pml_offdiag(
            fields, pml, residency, contract_variants=(),
            chunk_bounds=record)
        words = offdiag.presence_for(fields, pml, family._codes(fields.grid), record)
        rows.append({"label": label, "covered": verdict.covered,
                     "reasons": list(verdict.reasons), "planned": plan is not None,
                     "arm_wired": arm.wired, "connection_kinds": (
                         list(plan.phased) if plan is not None else None),
                     "words": list(words),
                     "agrees": bool(verdict.covered and plan is not None and arm.wired
                                    and tuple(plan.presence) == tuple(words)
                                    == offdiag.ALL_KEPT)})
    return {"passed": all(row["agrees"] for row in rows), "rows": rows}


def _emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"{row['index']}/{row['total']} {row['label']}: "
          f"diff={row.get('differing_words', '-')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be positive")
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")

    products = (
        ("two_axis_bloch", (1.125, 0.875, 0.75), (0.2, -0.125, 0.0),
         ("periodic", "periodic", "periodic")),
        ("zero_phase_control", (1.125, 0.875, 0.75), (0.0, 0.0, 0.0),
         ("periodic", "periodic", "periodic")),
        ("metallic_xz_bloch_y", (1.125, 0.875, 0.75), (0.0, 0.125, 0.0),
         ("metallic", "periodic", "metallic")),
        ("collapsed_y", (1.125, 0.125, 0.75), (0.2, 0.0, 0.0),
         ("periodic", "periodic", "periodic")),
        # k_x * L_x = 1/2: a phase of -1 on x, MEEP's negation connection.
        ("negated_x_bloch_y", (1.125, 0.875, 0.75), (1.0 / 2.25, 0.125, 0.0),
         ("periodic", "periodic", "periodic")),
    )
    cases = {label: dict(cell_size=cell_size, phase=phase, boundaries=boundaries)
             for label, cell_size, phase, boundaries in products}
    seeds = {"phased": 37100, "negated": 37150}
    specs = {"phased": cases["two_axis_bloch"], "negated": cases["negated_x_bloch_y"]}
    bases = {state: _case_source(seed=seeds[state], **specs[state]) for state in specs}
    mutations = _mutants(bases["phased"], bases["negated"])
    total = len(products) + len(bases) + len(mutations) + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    started = time.perf_counter()
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        for label, cell_size, phase, boundaries in products:
            index += 1
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "cell_size": cell_size, "k_point": phase, "boundaries": boundaries,
                   "cycles": args.cycles,
                   **_run(seed=36000 + index, cycles=args.cycles, cell_size=cell_size,
                          phase=phase, boundaries=boundaries)}
            rows.append(row)
            _emit(handle, row)
        for state, source in bases.items():
            index += 1
            result = _run(seed=seeds[state], cycles=1, functions=_compiled(source),
                          **specs[state])
            row = {"index": index, "total": total, "leg": "mutation_control",
                   "label": f"shipped_{state}",
                   "differing_words": int(result.get("differing_words", -1)),
                   "launches": result.get("launches"),
                   "passed": bool(result.get("passed"))}
            rows.append(row)
            _emit(handle, row)
        for label, (state, functions) in mutations.items():
            index += 1
            result = _run(seed=seeds[state], cycles=1, functions=functions,
                          **specs[state])
            difference = int(result.get("differing_words", 0))
            row = {"index": index, "total": total, "leg": "mutation", "state": state,
                   "label": label, "differing_words": difference,
                   "launches": result.get("launches"), "caught": difference > 0,
                   "passed": result.get("launches") == 1 and difference > 0}
            rows.append(row)
            _emit(handle, row)
        index += 1
        row = {"index": index, "total": total, "leg": "engine", "label": "engine",
               **leg_engine(products)}
        rows.append(row)
        _emit(handle, row)
    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "product_rows": len(products), "mutation_rows": len(mutations),
        "rows": rows, "torch_version": torch.__version__,
        "metal_frontend": metal_frontend_version(), "jsonl": str(jsonl),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; artifact {args.out}",
          flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
