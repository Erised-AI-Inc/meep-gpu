#!/usr/bin/env python3
"""Native MPS byte gate for complex folded tensor-row PML ``update_E``.

This is a sub-step gate: it certifies the exact complex, Bloch, folded ghost,
off-diagonal Yee average, and stored-E PML arithmetic together.  It does not claim
that a whole FDTD step is device-resident.

RULING R7: SOFTWARE BINARY64. The row is MEEP's double row per part, formed with
``metal_kernels.softfp64``, with the per-chunk presence words, and the Bloch
connection on the same-index operand's plane 0 (``stepping._bloch_ghost_operand``).
Every plan is the ENGINE route's (``plan_metal_complex_folded_offdiag`` with the
chunk record of the division the oracle steps with). The gate runs:

* ``products``: Bloch-phased and unphased folds on GENERIC states (where a float32
  row would differ) against ``stepping.update_E`` over ``--cycles`` cycles;
* ``b_select``: the per-chunk tail and the fold on EXACT-ROW states (dyadic parts,
  k = 0: every float32 operation of the row is exact, so a differing word is the
  tail's or the fold's);
* ``mutations``: each mutant compiled from the CASE'S OWN specialisation (read off
  the engine's plan) with the unmutated specialisation as a 0-word control: the
  dropped ghost sign, the reversed and the previous accumulation groupings, the
  select removed (exact unphased state), and on a Bloch-phased generic state the
  partner and own-axis connections dropped and the float32 row.
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
PARITY_ROOT = Path(__file__).resolve().parent
if str(PARITY_ROOT) not in sys.path:
    sys.path.insert(0, str(PARITY_ROOT))

import metal_composition_matrix as matrix  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_folded_offdiag_update_e as family, shaders, subnormal,
)
from gate_chunk_records import layer_record  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402


COMPARED = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _different(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def _build(*, axes: str, mirror_phases: Sequence[int], boundaries: Any,
           k_point: Sequence[float], rows: Mapping[str, Sequence[str]],
           seed: int, exact: bool = False) -> Tuple[Fields, PML]:
    grid = Grid(
        resolution=8.0,
        cell_size=(2.0, 1.5, 1.25),
        courant=0.35,
        boundaries=boundaries,
        k_point=tuple(k_point),
        symmetry=tuple(Mirror(axis, int(phase))
                       for axis, phase in zip(axes, mirror_phases)),
        xp=np,
    )
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = {name: np.full(shape, value, np.float32)
               for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))}
    inverse = {name: np.full(shape, np.float32(1.0) / value, np.float32)
               for name, value in epsilon.items()}
    tensor_rows = {
        row: {partner: rng.uniform(-0.18, 0.18, shape).astype(np.float32)
              for partner in partners}
        for row, partners in rows.items()
    }
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=tensor_rows)
    folded = {"XYZ".index(axis) for axis in axes}
    pml = PML(grid=grid, thickness=tuple(
        (0, 2) if axis in folded else (2, 2) for axis in range(3)))
    for name in ("Dx", "Dy", "Dz", *COMPARED):
        getattr(fields, name)[...] = (
            rng.uniform(-0.4, 0.4, shape).astype(np.float32)
            + 1j * rng.uniform(-0.4, 0.4, shape).astype(np.float32))
    if exact:
        # Dyadic parts (seven bits), inverse epsilon (four), rows (four, never
        # zero): every float32 operation of the row is exact, so the double row's one
        # rounding is the identity and the two rows are one word.
        for name in ("Dx", "Dy", "Dz", *COMPARED):
            real = rng.integers(-64, 65, size=shape) / 64.0
            imag = rng.integers(-64, 65, size=shape) / 64.0
            getattr(fields, name)[...] = (real.astype(np.float32)
                                          + 1j * imag.astype(np.float32))
        for name in ("Ex", "Ey", "Ez"):
            fields.inverse_epsilon_for(name)[...] = (
                rng.integers(4, 9, size=shape) / 8.0).astype(np.float32)
        for row, partners in rows.items():
            for partner in partners:
                value = fields.chi1inv_offdiagonal_for(row)[partner]
                sign = np.where(rng.integers(0, 2, size=shape) == 0, 1.0, -1.0)
                value[...] = (sign * rng.integers(1, 9, size=shape) / 64.0
                              ).astype(np.float32)
    return fields, pml


def _run(*, axes: str, mirror_phases: Sequence[int], boundaries: Any,
         k_point: Sequence[float], rows: Mapping[str, Sequence[str]], seed: int,
         cycles: int, functions: Mapping[str, Any] | None = None,
         exact: bool = False) -> Dict[str, Any]:
    reference, reference_pml = _build(
        axes=axes, mirror_phases=mirror_phases, boundaries=boundaries,
        k_point=k_point, rows=rows, seed=seed, exact=exact)
    actual, actual_pml = _build(
        axes=axes, mirror_phases=mirror_phases, boundaries=boundaries,
        k_point=k_point, rows=rows, seed=seed, exact=exact)
    before = {name: np.array(getattr(actual, name), copy=True) for name in COMPARED}
    residency = Residency()
    plan = family.plan_metal_complex_folded_offdiag(
        actual, actual_pml, residency, functions=functions,
        chunk_bounds=layer_record(actual.grid, actual_pml))
    if plan is None:
        return {"passed": False, "reason": "a declared folded complex product was refused"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        stepping.update_E(reference, reference_pml)
        plan.run()
        residency.sync_out(COMPARED)
        difference = sum(_different(getattr(reference, name), getattr(actual, name))
                         for name in COMPARED)
        census = sum(subnormal.census(getattr(reference, name)) for name in COMPARED)
        per_cycle.append({"cycle": cycle, "differing_words": difference,
                          "reference_subnormals": census})
        if difference or census:
            break
    moved = sum(_different(before[name], getattr(actual, name)) for name in COMPARED)
    return {
        "passed": (len(per_cycle) == cycles
                   and all(row["differing_words"] == row["reference_subnormals"] == 0
                           for row in per_cycle)
                   and moved > 0 and plan.launches == len(per_cycle)),
        "per_cycle": per_cycle,
        "differing_words": per_cycle[-1]["differing_words"],
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "moved_words": moved,
        "launches": plan.launches,
        "mirrors": len(residency.names),
    }


def _needle(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"mutation needle {old!r} matches {text.count(old)} times")
    return text.replace(old, new)


def _case_source(*, axes: str, mirror_phases: Sequence[int], boundaries: Any,
                 k_point: Sequence[float], rows: Mapping[str, Sequence[str]],
                 seed: int) -> str:
    """The SHIPPED source of one case's own specialisation, read off the engine's plan
    (row mask, codes, walls, connection kinds, parity): every mutant is
    this text with one defect planted, and this text unmutated is the control."""
    fields, pml = _build(axes=axes, mirror_phases=mirror_phases, boundaries=boundaries,
                         k_point=k_point, rows=rows, seed=seed)
    plan = family.plan_metal_complex_folded_offdiag(
        fields, pml, Residency(), contract_variants=(),
        chunk_bounds=layer_record(fields.grid, pml))
    assert plan is not None, "a declared folded complex product was refused"
    return family.complex_folded_offdiag_source(
        plan.row_mask, plan.codes, plan.walls, plan.phased, plan.negate)


def _compiled(source: str) -> Dict[str, Any]:
    return {shaders.CONTRACT_OFF: compile_source(source).complex_folded_offdiag_update_e}


def _mutants(base: str, phased_base: str) -> Dict[str, Tuple[str, Mapping[str, Any]]]:
    """``label -> (state, functions)``: ``exact`` mutants on the unphased exact-row
    case, ``phased`` mutants on the Bloch-phased generic case. Each needle must match
    exactly once, or the leg is disarmed and raises."""
    flags = shaders.constitutive_membership_flags("E")
    tail = ("    if (dw0) { a0 = f0[ii] + (c_mul_coefficient_left(kp_0, src0) - "
            "c_mul_coefficient_left(km_0, prev0)); }")
    exact = {
        "folded_ghost_sign_dropped": _needle(
            base,
            "down_00 = (at_y ? c_mul_coefficient_left(-1.0f, down_00) : down_00);",
            "// dropped folded parity"),
        "pml_accumulations_reversed": _needle(
            base, tail,
            "    if (dw0) { a0 = f0[ii] + (c_mul_coefficient_left(km_0, src0) - "
            "c_mul_coefficient_left(kp_0, prev0)); }"),
        # INVERTED (KERNELS.md §7.1): the package's previous grouping.
        "previous_accumulation_grouping": _needle(
            base, tail,
            "    if (dw0) { a0 = (f0[ii] + c_mul_coefficient_left(kp_0, src0)) - "
            "c_mul_coefficient_left(km_0, prev0); }"),
        "select_removed": _needle(
            base, f"const bool dw0 = {flags[0]}, dw1 = {flags[1]}, dw2 = {flags[2]};",
            "const bool dw0 = true, dw1 = true, dw2 = true;"),
    }
    # The phased case: Ex <- Ey with the fold on y, a Bloch phase on x and z.
    phased = {
        # The partner pair's same-index word on the partner axis's plane 0 (Ey <- Ez
        # on the phased z axis).
        "partner_connection_dropped": _needle(
            phased_base, "    home_10 = (k == 0) ? c_mul(home_10, ph[2]) : home_10;\n",
            ""),
        # The formed product on the component's own axis's plane 0 (Ex on x).
        "own_connection_dropped": _needle(
            phased_base,
            "    near_product_00 = (i == 0) ? c_mul(near_product_00, ph[3]) : "
            "near_product_00;\n", ""),
        # The pre-alignment placement: the phase on the shifted read instead.
        "partner_connection_on_the_shifted_read": _needle(
            phased_base, "    home_10 = (k == 0) ? c_mul(home_10, ph[2]) : home_10;\n"
            "    float2 down_10 = dvz ? g2[i * nyz + j * nzi + dk] : float2(0.0f, 0.0f);",
            "    float2 down_10 = dvz ? g2[i * nyz + j * nzi + dk] : float2(0.0f, 0.0f);\n"
            "    down_10 = (k == 0) ? c_mul(down_10, ph[2]) : down_10;"),
        # RULING R7's mutation, per part: component 0's row in float32.
        "float32_row": _float32_row(phased_base),
    }
    out = {label: ("exact", _compiled(source)) for label, source in exact.items()}
    out.update({label: ("phased", _compiled(source)) for label, source in phased.items()})
    return out


def _float32_row(source: str) -> str:
    """Component 0's two-term double row per part replaced by the float32 row."""
    old = (
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
    new = ("    float2 src0 = od_d0 + (c_mul_coefficient_left(0.25f, sum_00) + "
           "c_mul_coefficient_left(0.25f, sum_01));")
    return _needle(source, old, new)


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

    all_rows = {"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"), "Ez": ("Ex", "Ey")}
    # The products leg: Bloch-phased and unphased folds on GENERIC states.
    products = (
        ("periodic_even_two_axis_bloch", "Y", (1,), "periodic", (0.2, 0.0, 0.125), all_rows),
        ("metallic_odd", "Y", (-1,), {"y": "metallic"}, (0.2, 0.0, 0.125),
         {"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}),
        ("two_axis_fold_bloch_z", "XY", (1, -1), "periodic", (0.0, 0.0, 0.125),
         {"Ex": ("Ey",), "Ey": ("Ex", "Ez")}),
        ("sparse_row_control", "Y", (1,), "periodic", (0.2, 0.0, 0.125),
         {"Ey": ("Ez",)}),
        ("periodic_even_unphased", "Y", (1,), "periodic", (0.0, 0.0, 0.0), all_rows),
    )
    # b_select runs the unphased folds on exact-row states.
    exact_cases = (
        ("periodic_even_all_rows", "Y", (1,), "periodic", (0.0, 0.0, 0.0), all_rows),
        ("metallic_odd", "Y", (-1,), {"y": "metallic"}, (0.0, 0.0, 0.0),
         {"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}),
        ("two_axis_fold", "XY", (1, -1), "periodic", (0.0, 0.0, 0.0),
         {"Ex": ("Ey",), "Ey": ("Ex", "Ez")}),
        ("sparse_row_control", "Y", (1,), "periodic", (0.0, 0.0, 0.0),
         {"Ey": ("Ez",)}),
    )
    exact_spec = dict(axes="Y", mirror_phases=(1,), boundaries="periodic",
                      k_point=(0.0, 0.0, 0.0),
                      rows={"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex")})
    phased_spec = dict(axes="Y", mirror_phases=(1,), boundaries="periodic",
                       k_point=(0.2, 0.0, 0.125),
                       rows={"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex")})
    base = _case_source(seed=37100, **exact_spec)
    phased_base = _case_source(seed=37150, **phased_spec)
    mutations = _mutants(base, phased_base)
    controls = (("exact", base), ("phased", phased_base))
    total = len(products) + len(exact_cases) + len(controls) + len(mutations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    started = time.perf_counter()
    index = 0
    with jsonl.open("w", encoding="utf-8") as handle:
        for label, axes, mirror_phases, boundaries, k_point, tensor_rows in products:
            index += 1
            row = {"index": index, "total": total, "leg": "products", "label": label,
                   "axes": axes, "mirror_phases": mirror_phases, "boundaries": boundaries,
                   "k_point": k_point, "rows": tensor_rows, "cycles": args.cycles,
                   **_run(axes=axes, mirror_phases=mirror_phases, boundaries=boundaries,
                          k_point=k_point, rows=tensor_rows, seed=36000 + index,
                          cycles=args.cycles)}
            rows.append(row)
            _emit(handle, row)
        for label, axes, mirror_phases, boundaries, k_point, tensor_rows in exact_cases:
            index += 1
            row = {"index": index, "total": total, "leg": "b_select", "label": label,
                   "axes": axes, "mirror_phases": mirror_phases, "boundaries": boundaries,
                   "k_point": k_point, "rows": tensor_rows, "cycles": args.cycles,
                   "exact_row_state": True,
                   **_run(axes=axes, mirror_phases=mirror_phases, boundaries=boundaries,
                          k_point=k_point, rows=tensor_rows, seed=37000 + index,
                          cycles=args.cycles, exact=True)}
            rows.append(row)
            _emit(handle, row)
        # The controls: the case's own specialisation, unmutated, compiled through
        # the same seam the mutants take, must be 0 words from the oracle.
        for state, source in controls:
            index += 1
            spec = exact_spec if state == "exact" else phased_spec
            seed = 37100 if state == "exact" else 37150
            result = _run(cycles=1, functions=_compiled(source),
                          exact=state == "exact", seed=seed, **spec)
            row = {"index": index, "total": total, "leg": "mutation_control",
                   "label": f"shipped_{state}",
                   "differing_words": int(result.get("differing_words", -1)),
                   "launches": result.get("launches"),
                   "passed": bool(result.get("passed"))}
            rows.append(row)
            _emit(handle, row)
        for label, (state, functions) in mutations.items():
            index += 1
            spec = exact_spec if state == "exact" else phased_spec
            seed = 37100 if state == "exact" else 37150
            result = _run(cycles=1, functions=functions,
                          exact=state == "exact", seed=seed, **spec)
            difference = int(result.get("differing_words", 0))
            row = {"index": index, "total": total, "leg": "mutation", "state": state,
                   "label": label,
                   "differing_words": difference, "launches": result.get("launches"),
                   "caught": difference > 0,
                   "passed": result.get("launches") == 1 and difference > 0}
            rows.append(row)
            _emit(handle, row)
    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "product_rows": len(products), "b_select_rows": len(exact_cases),
        "mutation_rows": len(mutations), "rows": rows,
        "environment": ENVIRONMENT, "torch_version": torch.__version__,
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
