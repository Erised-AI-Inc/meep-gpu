"""Byte gate for the Metal COMPLEX radial prefix scan — ``cyl_complex_rderiv_prefix``
against ``stepping.cylindrical_rderiv_prefix`` on complex64 storage.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: at every radial extent of the sixteen
``cylindrical complex`` H->D corpus rows on the Metal board, at both ``ir0`` values the
array path uses (0.0 with the zero wall row on the B side, 0.5 raw on the D side), over
four value classes, ONE dispatch of the device column-serial complex scan writes a
prefix volume that is BIT-IDENTICAL, as uint32 words, to
``stepping.cylindrical_rderiv_prefix`` CALLED on the same complex64 input — never
re-derived, never approximated, ``allclose`` nowhere.

WHAT IS NEW, and why this gate exists before the product that consumes the kernel: the
sixteen complex rows pay a host round trip on every ``step_D`` because the complex
family scans on the host; the two-launch H->D product removes it ONLY if a device
complex scan reproduces the oracle. And the port from the real scan carries one trap —
the divide spelling INVERTS (``/`` on float32, the reciprocal multiply on complex64,
because ``numpy``'s ``complex64 / float32`` is its complex divide loop) — which this
gate arms as a mutation that MUST be caught, beside the equivalences it measures.

THE LEGS

  1  bindings        the emitted source binds exactly eight buffers, compiles on both
                     sub-steps, spells the reciprocal multiply and never ``/`` on the
                     complex increment, multiplies with the field on the left, and
                     carries the contraction guard exactly once
  2  corpus_shapes   the thirteen (nr, nz) extents here ARE the sixteen rows' extents in
                     the census record, read from the record when it is present
  3  prefix          13 extents x 2 sub-steps x 4 value classes = 104 cases, the device
                     scan against the array path's own function, at BOTH expansion arms
                     (the probe-bound one is the shipped one; the other is a measured
                     equivalence for a real coefficient), with the subnormal-free
                     precondition CHECKED on every case and a movement floor
  4  absorption      the +-0 lattice: the oracle's prefix is +0.0 in EVERY word, and so
                     is the device's — the fact that makes the signed-zero refinement of
                     the divide spelling unobservable on the prefix. Recorded as a
                     predicted null-sign case, not as a discriminating comparison
  5  increment_sign  HOST measurement: numpy's complex-by-real divide and multiply
                     differ from the componentwise spellings on the INCREMENT's
                     signed-zero words and agree on the PREFIX, over the signed-zero
                     class at every extent
  6  extra_extents   the two withdraw_seam rows' extents (100, 1335), measured for the
                     ladder and NOT credited to any row
  7  mutations       ten armed defects through the launch-counted harness — the ``/``
                     spelling, a blocked scan, a -0.0 row zero, both off-by-ones, the
                     dropped wall row (must be caught); the literal numpy divide loop,
                     the componentwise multiply, the coefficient-left orientation and
                     the commuted accumulator (must NOT be caught, measured equivalences);
                     the fast-contraction variant (recorded)
  8  disarm          the shipped bytes through the same harness path must not diverge

WHAT THE GATE REFUSES TO INFER

* **A no-op agreeing with a no-op is trivially identical.** Every case must produce a
  floor of nonzero output words, and the +-0 lattice — whose output is all zeros by the
  oracle's own arithmetic — is recorded as predicted-null rather than counted as a
  discriminating pass.
* **"The other expansion arm is the same bits" is a measurement**, 104 cases of it.
* **A value class that entered the subnormal band is a REFUSAL**, not a comparison: MPS
  flushes natively, the oracle keeps, and this claim rides a checked precondition.
* **A mutation that never launched is DISARMED and fails; one whose needle matched
  nothing is NEEDLE-MISSED and fails** (``metal_gate_kit.MutationHarness``).

Rule 7: one flushed line per case, every row appended and fsynced as it lands.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (API_ROOT, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# THE POLICY IS SET BEFORE ANY `meep_gpu` MODULE IS REACHED, because the predicates
# read it at import. On MPS the flush is native and has no lever; setting `flush` puts
# the claim under a CHECKED precondition rather than a pretence.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import metal_composition_matrix as matrix  # noqa: E402
import metal_gate_kit as kit  # noqa: E402
import metal_value_classes as values  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    cylindrical_complex as cylc,
    cylindrical_complex_scan as scan,
    cylindrical_real as cyl,
    shaders,
    subnormal,
    templates,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, compile_source,
)

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words

#: THE SIXTEEN ROWS, keyed by the census record's own row name, with the board label
#: where it differs and the census grid shape ``(nr, ny, nz)``. Transcribed from
#: ``results/metal_coverage_2026-09-04_m0complex`` and cross-checked against it by leg
#: ``corpus_shapes`` whenever that record is on this host. ``ny`` is 1 on every row, so
#: the scan's independent unit is the ``nz`` radial columns.
CENSUS_RECORD = "metal_coverage_2026-09-04_m0complex"
CORPUS_ROWS: Tuple[Tuple[str, str, str, Tuple[int, int, int]], ...] = (
    # (census leg, census row name, board label, grid shape)
    ("examples", "dipole_in_vacuum_cyl_off_axis.py", "dipole_in_vacuum_cyl_off_axis.py",
     (150, 1, 300)),
    ("examples", "dipole_in_vacuum_cyl_on_axis.py", "dipole_in_vacuum_cyl_on_axis.py",
     (150, 1, 300)),
    ("examples", "disc_extraction_efficiency.py", "disc_extraction_efficiency.py",
     (250, 1, 115)),
    ("examples", "disc_radiation_pattern.py", "disc_radiation_pattern.py", (650, 1, 179)),
    ("examples", "extraction_eff_ldos.py", "extraction_eff_ldos.py", (520, 1, 143)),
    ("examples", "perturbation_theory.py", "perturbation_theory.py", (800, 1, 1)),
    ("examples", "planar_cavity_ldos.py", "planar_cavity_ldos.py", (462, 1, 497)),
    ("examples", "point_dipole_cyl.py", "point_dipole_cyl.py", (1050, 1, 115)),
    ("examples", "ring-cyl.py", "ring-cyl.py", (760, 1, 1)),
    ("tests", "TestLDOS.test_ldos_cyl", "TestLDOS.test_ldos_cyl", (163, 1, 175)),
    ("tests", "TestLDOS.test_ldos_ext_eff", "TestLDOS.test_ldos_ext_eff", (163, 1, 43)),
    ("tests", "TestRingCyl.test_ring_cyl", "TestRingCyl.test_ring_cyl", (80, 1, 1)),
    ("tests_param", "TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields__idx2",
     "TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_2_1", (120, 1, 120)),
    ("tests_param", "TestPMLCylindrical.test_pml_cyl__idx1",
     "TestPMLCylindrical.test_pml_cyl_1_1_0", (150, 1, 175)),
    ("tests_param", "TestPMLCylindrical.test_pml_cyl__idx2",
     "TestPMLCylindrical.test_pml_cyl_2_2_0", (150, 1, 175)),
    ("tests_param", "TestPMLCylindrical.test_pml_cyl__idx3",
     "TestPMLCylindrical.test_pml_cyl_3_3_0", (150, 1, 175)),
)

#: The two ``withdraw_seam`` rows. Their extents are measured (leg ``extra_extents``)
#: because 1335 is the largest radial ladder on the board; they are credited to NOTHING.
WITHDRAW_ROWS: Tuple[Tuple[str, str, Tuple[int, int, int]], ...] = (
    ("examples", "cylinder_cross_section.py", (100, 1, 222)),
    ("examples", "zone_plate.py", (1335, 1, 163)),
)

#: The distinct (nr, nz) extents of the sixteen rows, in the order first met. Thirteen.
SHAPES: Tuple[Tuple[int, int], ...] = tuple(dict.fromkeys(
    (shape[0], shape[2]) for _leg, _row, _label, shape in CORPUS_ROWS))

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: The four compared value classes, in scoring order. ``pm_zero_lattice`` is NOT one of
#: them: its oracle output is all +0.0 (leg ``absorption``), so it discriminates no
#: sign and moves no word.
UNIFORM = "uniform"
WIDE_DYNAMIC = "wide_dynamic"
CANCELLING = "cancelling"
SIGNED_ZERO_MIX = "signed_zero_mix"
VALUE_CLASSES: Tuple[str, ...] = (UNIFORM, WIDE_DYNAMIC, CANCELLING, SIGNED_ZERO_MIX)

#: The expansion arm measured beside the probe-bound one.
EXPANSION_ARMS: Tuple[str, ...] = tuple(templates.EXPANSION_ARMS)


# ---------------------------------------------------------------------------
# Inputs: value classes and the oracle
# ---------------------------------------------------------------------------

def make_input(shape: Tuple[int, int, int], value_class: str, seed: int) -> np.ndarray:
    """One complex64 source volume of one value class. Complex64 out, always."""
    rng = np.random.default_rng(seed)
    nr, ny, nz = shape
    if value_class == UNIFORM:
        re = rng.standard_normal(shape)
        im = rng.standard_normal(shape)
    elif value_class == WIDE_DYNAMIC:
        # Twenty-four decades, far above the subnormal band even after the weight
        # multiply (<= 1335x) and the running sum.
        scale = 10.0 ** rng.integers(-12, 13, shape)
        re = rng.standard_normal(shape) * scale
        im = rng.standard_normal(shape) * scale
    elif value_class == CANCELLING:
        # Alternating sign down the radial ladder, so the partial sums pass through
        # zero repeatedly — the worst case for a scan's rounding.
        sign = np.where(np.arange(nr).reshape(-1, 1, 1) % 2 == 0, 1.0, -1.0)
        re = rng.standard_normal(shape) * sign * 1e6
        im = rng.standard_normal(shape) * sign * 1e6
    elif value_class == SIGNED_ZERO_MIX:
        # Per plane: a quarter +0.0, a quarter -0.0, half physical. Exactly-zero
        # increments of BOTH signs arise, which is where the divide and multiply
        # spellings differ on the increment.
        def plane() -> np.ndarray:
            pick = rng.integers(0, 4, shape)
            phys = rng.standard_normal(shape).astype(np.float32)
            out = np.where(pick == 0, np.float32(0.0),
                           np.where(pick == 1, np.float32(-0.0), phys))
            return out.astype(np.float32)
        volume = np.empty(shape, dtype=np.complex64)
        volume.real = plane()
        volume.imag = plane()
        return volume
    else:
        raise ValueError(value_class)
    volume = np.empty(shape, dtype=np.complex64)
    volume.real = re.astype(np.float32)
    volume.imag = im.astype(np.float32)
    return volume


def scanned_source(sub_step: str, source: np.ndarray) -> np.ndarray:
    """What the array path scans: the source, EXTENDED by a zero wall row on B."""
    if not cyl.PREFIX_WALL_ROW[sub_step]:
        return source
    rows = source.shape[0]
    extended = np.zeros((rows + 1,) + source.shape[1:], dtype=source.dtype)
    extended[:rows] = source
    extended[rows] = 0  # stepping.py:360
    return extended


def reference_prefix(sub_step: str, source: np.ndarray) -> np.ndarray:
    """``stepping``'s own prefix — CALLED, never re-derived."""
    return stepping.cylindrical_rderiv_prefix(
        np, scanned_source(sub_step, source), cyl.PREFIX_IR0[sub_step])


class Case:
    """One (extent, sub-step, class) input with its oracle and its device tensors."""

    __slots__ = ("label", "shape", "sub_step", "value_class", "source", "expected",
                 "scan_shape", "weights", "divisor", "src_t", "w_t", "d_t")

    def __init__(self, shape: Tuple[int, int], sub_step: str, value_class: str,
                 seed: int) -> None:
        import torch  # noqa: PLC0415

        nr, nz = shape
        self.label = f"nr{nr}_nz{nz}/{sub_step}/{value_class}"
        self.shape = (nr, 1, nz)
        self.sub_step = sub_step
        self.value_class = value_class
        self.source = make_input(self.shape, value_class, seed)
        self.expected = reference_prefix(sub_step, self.source)
        self.scan_shape = (cyl.scan_rows(sub_step, nr), 1, nz)
        self.weights, self.divisor = scan.prefix_row_vectors(sub_step, nr)
        self.src_t = torch.from_numpy(np.ascontiguousarray(self.source).reshape(-1)
                                      ).to("mps")
        self.w_t = torch.from_numpy(np.ascontiguousarray(self.weights, dtype=np.float32)
                                    ).to("mps")
        self.d_t = torch.from_numpy(np.ascontiguousarray(self.divisor, dtype=np.float32)
                                    ).to("mps")

    def launch(self, function: Callable[..., Any]) -> np.ndarray:
        """One dispatch into a SENTINEL-filled output, so a kernel that never ran
        cannot compare equal."""
        import torch  # noqa: PLC0415

        sentinel = np.full(self.scan_shape, np.complex64(-7.5 - 7.5j), dtype=np.complex64)
        out = torch.from_numpy(sentinel.reshape(-1)).to("mps")
        function(*scan.scan_arguments(out, self.src_t, self.w_t, self.d_t,
                                      self.scan_shape))
        torch.mps.synchronize()
        got = out.cpu().numpy().reshape(self.scan_shape)
        assert got.shape == self.expected.shape, (self.label, got.shape,
                                                  self.expected.shape)
        return got


def build_cases(seed_base: int = 71000) -> List[Case]:
    cases: List[Case] = []
    for offset, (shape, sub_step, value_class) in enumerate(
            (shape, sub_step, value_class) for shape in SHAPES
            for sub_step in SUB_STEPS for value_class in VALUE_CLASSES):
        cases.append(Case(shape, sub_step, value_class, seed_base + offset))
    return cases


def bound_expansion() -> Tuple[Optional[str], str]:
    """``(probe-bound arm or None, the arm this gate ships its measurement at)``."""
    probe = cylc.load_expansion_probe()
    arm = cylc.expansion_from_probe(probe) if probe else None
    return arm, (arm or "FMA_V1")


_FUNCTIONS: Dict[Tuple[str, str, str], Any] = {}


def shipped_function(sub_step: str, expansion: str,
                     contract: str = shaders.CONTRACT_OFF) -> Any:
    key = (sub_step, expansion, contract)
    if key not in _FUNCTIONS:
        _FUNCTIONS[key] = scan.compile_cylindrical_complex_prefix(sub_step, expansion,
                                                                  contract)
    return _FUNCTIONS[key]


def _code_lines(source: str) -> str:
    """The source with ``//`` comment text removed — what the COMPILER sees."""
    kept = []
    for line in source.splitlines():
        head = line.split("//", 1)[0]
        if head.strip():
            kept.append(head)
    return "\n".join(kept)


def _compiles(source: str) -> Tuple[bool, str]:
    try:
        compile_source(source)
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        return False, str(exc)
    return True, ""


# ---------------------------------------------------------------------------
# LEG bindings — the signature and the spelling, compiled and pinned
# ---------------------------------------------------------------------------

def leg_bindings(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    rows: List[Dict[str, Any]] = []
    for sub_step in SUB_STEPS:
        source = scan.cylindrical_complex_prefix_source(sub_step, expansion)
        indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
        code = _code_lines(source)
        ok, error = _compiles(source)
        row = {
            "sub_step": sub_step,
            "expansion": expansion,
            "buffer_indices": indices,
            "bindings": len(indices),
            "declared_bindings": scan.BINDINGS,
            "ceiling": MAX_BUFFER_BINDINGS,
            "headroom": MAX_BUFFER_BINDINGS - len(indices),
            "compiled": ok,
            "compile_error": error,
            "float2_volumes": (code.count("device float2*") + code.count(
                "device const float2*")),
            "reciprocal_multiply_spelled": "return z * (1.0f / d);" in code,
            "true_divide_absent": ("/ divisor" not in code and "z / d" not in code),
            "fast_divide_absent": "fast::" not in code,
            # ONE definition (the certified helper) and TWO call sites (row 0, row i);
            # the coefficient-left helper is defined by the same helper block but is
            # never called on the weight.
            "field_left_multiply": (code.count("c_mul_field_left(") == 3
                                    and "c_mul_coefficient_left(w" not in code
                                    and "c_mul_coefficient_left(src" not in code),
            "contraction_guard_once": source.count(
                shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1,
            "no_math_mode_pragma": "math_mode" not in source and "denorm" not in source,
            "row_zero_positive": "float2 acc = float2(0.0f, 0.0f);" in code,
            "wall_row_read": ("float2(0.0f, 0.0f))" in code
                             if cyl.PREFIX_WALL_ROW[sub_step]
                             else "i < nxi - 1" not in code),
            "source_sha256": templates.source_sha256(source),
        }
        row["passed"] = bool(
            ok and indices == list(range(scan.BINDINGS))
            and row["float2_volumes"] == 2 and row["reciprocal_multiply_spelled"]
            and row["true_divide_absent"] and row["fast_divide_absent"]
            and row["field_left_multiply"] and row["contraction_guard_once"]
            and row["no_math_mode_pragma"] and row["row_zero_positive"]
            and row["wall_row_read"])
        rows.append(row)
        log(f"[bindings] {sub_step} bindings={len(indices)} compiled={ok} "
            f"passed={row['passed']}")
    payload["legs"]["bindings"] = rows
    save(payload, out)
    assert all(row["passed"] for row in rows), rows


# ---------------------------------------------------------------------------
# LEG corpus_shapes — the extents here are the census record's
# ---------------------------------------------------------------------------

def leg_corpus_shapes(payload: Dict[str, Any], out: str) -> None:
    record_dir = os.path.join(HERE, "results", CENSUS_RECORD)
    row: Dict[str, Any] = {"record": CENSUS_RECORD, "present": os.path.isdir(record_dir),
                           "rows": len(CORPUS_ROWS), "extents": len(SHAPES),
                           "shapes": [list(shape) for shape in SHAPES]}
    if not row["present"]:
        row["checked"] = 0
        row["finding"] = ("the census record is not on this host; the sixteen shapes "
                          "are as transcribed and were not cross-checked here")
        payload["legs"]["corpus_shapes"] = row
        save(payload, out)
        log(f"[corpus_shapes] record absent; {len(SHAPES)} extents transcribed")
        return
    census: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for leg in ("examples", "tests", "tests_param"):
        path = os.path.join(record_dir, f"{leg}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    census[(leg, str(entry.get("row")))] = entry
    mismatches: List[str] = []
    checked = 0
    for leg, name, _label, shape in CORPUS_ROWS + tuple(
            (leg, name, name, shape) for leg, name, shape in WITHDRAW_ROWS):
        entry = census.get((leg, name))
        if entry is None:
            mismatches.append(f"{leg}:{name} is not in the record")
            continue
        checked += 1
        recorded = tuple(int(n) for n in entry.get("grid_shape", ()))
        if recorded != tuple(shape):
            mismatches.append(f"{leg}:{name} shape {recorded} != {tuple(shape)}")
        selected = (entry.get("plan_step") or {}).get("selected") or {}
        for slot in ("update_H", "step_D"):
            if selected.get(slot) != "cylindrical complex":
                mismatches.append(f"{leg}:{name} {slot} is {selected.get(slot)!r}, "
                                  f"not 'cylindrical complex'")
    row.update(checked=checked, mismatches=mismatches, passed=not mismatches)
    payload["legs"]["corpus_shapes"] = row
    save(payload, out)
    log(f"[corpus_shapes] checked={checked} of {len(CORPUS_ROWS) + len(WITHDRAW_ROWS)} "
        f"mismatches={len(mismatches)}")
    assert not mismatches, mismatches


# ---------------------------------------------------------------------------
# LEG prefix — the device scan against the array path's own function
# ---------------------------------------------------------------------------

def _compare_case(case: Case, function: Callable[..., Any]) -> Tuple[int, int, np.ndarray]:
    got = case.launch(function)
    return differing(got, case.expected), int(words(case.expected).size), got


def leg_prefix(payload: Dict[str, Any], out: str, cases: Sequence[Case]) -> None:
    probe_arm, expansion = bound_expansion()
    other = tuple(arm for arm in EXPANSION_ARMS if arm != expansion)
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(cases, start=1):
        band_in = subnormal.census(case.source)
        band_ref = subnormal.census(case.expected)
        assert band_in == 0 and band_ref == 0, (
            f"{case.label}: the value class entered the subnormal band "
            f"(input {band_in}, reference {band_ref} words); a banded case is a "
            f"refusal on this executor, not a comparison")
        differ, compared, got = _compare_case(case, shipped_function(case.sub_step,
                                                                    expansion))
        moved = differing(got, np.zeros_like(got))
        kit.assert_moved(moved, f"{case.label} scan output is all zero",
                         floor=max(1, min(64, compared // 4)))
        others = {arm: _compare_case(case, shipped_function(case.sub_step, arm))[0]
                  for arm in other}
        row = {
            "case": case.label, "index": index, "total": len(cases),
            "shape": list(case.shape), "scan_shape": list(case.scan_shape),
            "sub_step": case.sub_step, "ir0": cyl.PREFIX_IR0[case.sub_step],
            "wall_row": bool(cyl.PREFIX_WALL_ROW[case.sub_step]),
            "value_class": case.value_class, "expansion": expansion,
            "compared": compared, "differing": differ, "nonzero_words": moved,
            "differing_other_arms": others,
            "subnormal_words_input": band_in, "subnormal_words_reference": band_ref,
            "reference_zero_sign_census": values.zero_sign_census([case.expected]),
            "digest": kit.state_digest({"prefix": case.expected}),
        }
        rows.append(row)
        payload["legs"]["prefix"] = rows
        save(payload, out)
        log(f"[prefix] case {index}/{len(cases)} {case.label:<40} compared={compared} "
            f"differing={differ} other_arms={others} ({time.time() - started:.1f}s)")
    payload["legs"]["prefix_summary"] = {
        "probe_bound_expansion": probe_arm, "measured_expansion": expansion,
        "cases": len(rows), "compared": sum(row["compared"] for row in rows),
        "differing": sum(row["differing"] for row in rows),
        "differing_other_arms": {
            arm: sum(row["differing_other_arms"][arm] for row in rows) for arm in other},
        "extents": len(SHAPES), "sub_steps": list(SUB_STEPS),
        "value_classes": list(VALUE_CLASSES),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG absorption — the +-0 lattice is all +0.0 on both sides
# ---------------------------------------------------------------------------

def leg_absorption(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    rows: List[Dict[str, Any]] = []
    for shape in SHAPES:
        for sub_step in SUB_STEPS:
            nr, nz = shape
            source = values.pm_zero_lattice_complex((nr, 1, nz))
            expected = reference_prefix(sub_step, source)
            census_in = values.zero_sign_census([source])
            census_out = values.zero_sign_census([expected])
            case = Case.__new__(Case)
            case.label = f"nr{nr}_nz{nz}/{sub_step}/pm_zero_lattice"
            case.shape = (nr, 1, nz)
            case.sub_step = sub_step
            case.value_class = "pm_zero_lattice"
            case.source = source
            case.expected = expected
            case.scan_shape = (cyl.scan_rows(sub_step, nr), 1, nz)
            case.weights, case.divisor = scan.prefix_row_vectors(sub_step, nr)
            import torch  # noqa: PLC0415
            case.src_t = torch.from_numpy(source.reshape(-1).copy()).to("mps")
            case.w_t = torch.from_numpy(np.ascontiguousarray(case.weights,
                                                             dtype=np.float32)).to("mps")
            case.d_t = torch.from_numpy(np.ascontiguousarray(case.divisor,
                                                             dtype=np.float32)).to("mps")
            differ, compared, got = _compare_case(case, shipped_function(sub_step,
                                                                        expansion))
            all_positive_zero = (census_out["positive_zero_words"] == compared
                                 and census_out["negative_zero_words"] == 0)
            row = kit.predicted_null({
                "case": case.label, "shape": list(case.shape), "sub_step": sub_step,
                "compared": compared, "differing": differ,
                "input_zero_sign_census": census_in,
                "reference_zero_sign_census": census_out,
                "reference_all_positive_zero": all_positive_zero,
            }, ("the accumulator starts at an exact +0.0 and IEEE addition never yields "
                "-0.0 from a +0.0 accumulator, so every signed-zero increment is "
                "absorbed: the oracle's prefix over a +-0 lattice is +0.0 in every word "
                "and the comparison discriminates no sign and moves no word"))
            rows.append(row)
            payload["legs"]["absorption"] = rows
            save(payload, out)
            log(f"[absorption] {case.label:<44} compared={compared} differing={differ} "
                f"reference +0={census_out['positive_zero_words']} "
                f"-0={census_out['negative_zero_words']}")
            assert all_positive_zero, row
            assert differ == 0, row
            assert census_in["negative_zero_words"] > 0, row


# ---------------------------------------------------------------------------
# LEG increment_sign — the refinement, measured on the host
# ---------------------------------------------------------------------------

def _increments(source: np.ndarray, ir0: float) -> Dict[str, np.ndarray]:
    """The increment three ways, with the array path's own row vectors."""
    weights, divisor = stepping._cylindrical_rderiv_weights(np, source.shape[0], ir0,
                                                            np.float32)
    # THE ARRAY PATH'S OWN SPELLING (stepping.py:1313-1315), numpy loops throughout.
    weighted = source * weights
    numpy_inc = np.zeros_like(source)
    numpy_inc[1:] = (weighted[1:] - weighted[:-1]) / divisor
    # THE PLAIN RECIPROCAL MULTIPLY, componentwise — what the kernel spells.
    recip = np.float32(1) / divisor
    recip_inc = np.zeros_like(source)
    diff = weighted[1:] - weighted[:-1]
    recip_inc.real[1:] = (diff.real * recip).astype(np.float32)
    recip_inc.imag[1:] = (diff.imag * recip).astype(np.float32)
    # THE COMPONENTWISE MULTIPLY feeding the reciprocal divide.
    comp_weighted = np.zeros_like(source)
    comp_weighted.real = (source.real * weights).astype(np.float32)
    comp_weighted.imag = (source.imag * weights).astype(np.float32)
    comp_inc = np.zeros_like(source)
    cdiff = comp_weighted[1:] - comp_weighted[:-1]
    comp_inc.real[1:] = (cdiff.real * recip).astype(np.float32)
    comp_inc.imag[1:] = (cdiff.imag * recip).astype(np.float32)
    return {"numpy": numpy_inc, "reciprocal": recip_inc,
            "componentwise_multiply": comp_inc}


def leg_increment_sign(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for shape in SHAPES:
        nr, nz = shape
        source = make_input((nr, 1, nz), SIGNED_ZERO_MIX, 83000 + nr)
        for sub_step in SUB_STEPS:
            scanned = scanned_source(sub_step, source)
            ir0 = cyl.PREFIX_IR0[sub_step]
            forms = _increments(scanned, ir0)
            prefixes = {name: np.cumsum(inc, axis=0) for name, inc in forms.items()}
            expected = reference_prefix(sub_step, source)
            row = {
                "shape": [nr, 1, nz], "sub_step": sub_step, "ir0": ir0,
                "words": int(words(forms["numpy"]).size),
                "increment_differing": {
                    name: differing(inc, forms["numpy"])
                    for name, inc in forms.items() if name != "numpy"},
                "prefix_differing": {
                    name: differing(prefix, expected) for name, prefix in prefixes.items()},
                "numpy_cumsum_is_the_oracle": differing(prefixes["numpy"], expected) == 0,
            }
            rows.append(row)
            payload["legs"]["increment_sign"] = rows
            save(payload, out)
            log(f"[increment_sign] nr{nr}_nz{nz} {sub_step} increment differs "
                f"{row['increment_differing']} prefix differs {row['prefix_differing']}")
            assert row["numpy_cumsum_is_the_oracle"], row
            assert all(count == 0 for count in row["prefix_differing"].values()), row
    total_inc = {name: sum(row["increment_differing"][name] for row in rows)
                 for name in ("reciprocal", "componentwise_multiply")}
    payload["legs"]["increment_sign_summary"] = {
        "cases": len(rows), "words": sum(row["words"] for row in rows),
        "increment_differing": total_inc,
        "prefix_differing": {name: sum(row["prefix_differing"][name] for row in rows)
                             for name in ("reciprocal", "componentwise_multiply")},
        "finding": (
            "numpy's complex64-by-float32 divide and multiply are its complex loops "
            "and differ from the componentwise spellings on the SIGN of exactly-zero "
            "increment words; the running sum from an exact +0.0 absorbs every one of "
            "them, so the two spellings are bit-identical on the PREFIX at every corpus "
            "extent. The kernel ships the reciprocal multiply the verdict named; the "
            "refinement is recorded here so a consumer of the INCREMENT would know"),
    }
    save(payload, out)
    # NON-VACUITY: the refinement must be REAL on the increment, or this leg says
    # nothing about the absorption.
    assert all(count > 0 for count in total_inc.values()), total_inc


# ---------------------------------------------------------------------------
# LEG extra_extents — the two withdraw rows' ladders, uncredited
# ---------------------------------------------------------------------------

def leg_extra_extents(payload: Dict[str, Any], out: str) -> None:
    _probe, expansion = bound_expansion()
    rows: List[Dict[str, Any]] = []
    for offset, (_leg, name, shape) in enumerate(WITHDRAW_ROWS):
        for sub_step in SUB_STEPS:
            case = Case((shape[0], shape[2]), sub_step, UNIFORM, 91000 + offset)
            differ, compared, got = _compare_case(case, shipped_function(sub_step,
                                                                        expansion))
            moved = differing(got, np.zeros_like(got))
            kit.assert_moved(moved, f"{name}/{sub_step} scan output is all zero", 64)
            row = {"row": name, "bucket": "withdraw_seam", "credited": False,
                   "shape": list(shape), "sub_step": sub_step, "compared": compared,
                   "differing": differ, "nonzero_words": moved,
                   "subnormal_words_reference": subnormal.census(case.expected)}
            rows.append(row)
            payload["legs"]["extra_extents"] = rows
            save(payload, out)
            log(f"[extra_extents] {name:<28} {sub_step} nr={shape[0]} compared={compared} "
                f"differing={differ}")


# ---------------------------------------------------------------------------
# LEG mutations — armed, launch-counted, each with its expected fate
# ---------------------------------------------------------------------------

_LITERAL_LOOP = """static inline float2 c_div_real(float2 z, float d) {
    float rat = 0.0f / d;
    float scl = 1.0f / (d + 0.0f * rat);
    return float2((z.x + z.y * rat) * scl, (z.y - z.x * rat) * scl);
}"""

_SHIPPED_DIVIDE = """static inline float2 c_div_real(float2 z, float d) {
    return z * (1.0f / d);
}"""

_BLOCKED = """    float2 carry = float2(0.0f, 0.0f);
    float2 local = float2(0.0f, 0.0f);
    for (int i = 1; i < nxi; ++i) {
        int o = i * nyz + base;
        float2 w = c_mul_field_left(__SRCI__, weights[i]);
        float2 inc = c_div_real(w - prev, divisor[i - 1]);
        local = local + inc;
        out[o] = carry + local;
        if ((i % 32) == 31) { carry = carry + local; local = float2(0.0f, 0.0f); }
        prev = w;
    }"""

_SERIAL = """    for (int i = 1; i < nxi; ++i) {
        int o = i * nyz + base;
        float2 w = c_mul_field_left(__SRCI__, weights[i]);
        // stepping.py:1286/:1301-1302 — the difference of two ALREADY-WEIGHTED rows,
        // then numpy's complex-by-real divide, which is the reciprocal multiply.
        float2 inc = c_div_real(w - prev, divisor[i - 1]);
        // STRICTLY SERIAL: numpy.cumsum is out[i] = out[i-1] + in[i], both planes.
        acc = acc + inc;
        out[o] = acc;
        prev = w;
    }"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 1286->1315, 1301-1302->1330-1331


def _with_reads(text: str, sub_step: str) -> str:
    src0, srci = scan.source_reads(sub_step)
    return text.replace("__SRC0__", src0).replace("__SRCI__", srci)


def mutation_table(sub_step: str) -> Tuple[Tuple[str, Callable[[str], str],
                                                 Optional[bool], str,
                                                 Optional[Tuple[str, ...]]], ...]:
    """``(label, transform, must_catch, why, classes)``.

    ``classes`` is the value-class SCOPE the verdict is taken over — ``None`` for all
    four. A mutation is scoped where a value class can ABSORB its edit: the dropped
    wall row touches one word pair per column, and on a single-column extent a
    wide-dynamic last row twenty-four decades below its neighbour vanishes in the
    subtraction, while a signed-zero-mix last row that is exactly zero makes the edit
    a no-op. Those classes are still RUN and their counts recorded beside the verdict,
    so a scoped mutation cannot hide a class it was not caught on.
    """
    src0, srci = scan.source_reads(sub_step)

    def componentwise(text: str) -> str:
        # The two CALL SITES only, spelled inline as (re*w, im*w); the certified helper
        # definition stays untouched so the mutant is a change of arithmetic and not
        # of names.
        text = kit.needle(text, f"c_mul_field_left({srci}, weights[i])",
                          f"float2(({srci}).x * weights[i], ({srci}).y * weights[i])")
        return kit.needle(text, f"c_mul_field_left({src0}, weights[0])",
                          f"float2(({src0}).x * weights[0], ({src0}).y * weights[0])")

    table: List[Tuple[str, Callable[[str], str], Optional[bool], str,
                      Optional[Tuple[str, ...]]]] = [
        ("true_divide_spelling",
         lambda s: kit.needle(s, "return z * (1.0f / d);", "return z / d;"), True,
         "the real scan's `/` carried onto the complex increment: numpy's complex64 / "
         "float32 is the reciprocal multiply, and the componentwise true divide is a "
         "different float32 number on ~25% of increments and, through the running "
         "sum, on most prefix words", None),
        ("literal_numpy_divide_loop",
         lambda s: kit.needle(s, _SHIPPED_DIVIDE, _LITERAL_LOOP), False,
         "numpy's complex divide loop spelled literally (rat/scl): differs from the "
         "reciprocal multiply only on the sign of exactly-zero increments, which the "
         "+0.0 accumulator absorbs — a MEASURED equivalence on the prefix", None),
        ("componentwise_weight_multiply", componentwise, False,
         "(re*w, im*w) in place of the certified field-left complex product: differs "
         "only on the sign of exactly-zero weighted words, absorbed by the running sum "
         "— a MEASURED equivalence on the prefix", None),
        ("coefficient_left_orientation",
         lambda s: kit.needle(kit.needle(
             s, f"c_mul_field_left({srci}, weights[i])",
             f"c_mul_coefficient_left(weights[i], {srci})"),
             f"c_mul_field_left({src0}, weights[0])",
             f"c_mul_coefficient_left(weights[0], {src0})"), False,
         "the weight on the left of the field: for a REAL coefficient the two "
         "orientations are the same four products and the same two adds, so this is "
         "an equivalence — kept field-left because that is the array path's order", None),
        ("blocked_scan_32",
         lambda s: kit.needle(s, _with_reads(_SERIAL, sub_step),
                              _with_reads(_BLOCKED, sub_step)), True,
         "a 32-row tiled scan with a carry: a legitimate alternative association of "
         "the same increments, and a different float32 number — the null control that "
         "makes 'column-serial' a licence rather than taste", None),
        ("commuted_accumulator",
         lambda s: kit.needle(s, "acc = acc + inc;", "acc = inc + acc;"), False,
         "float addition IS commutative; the control that shows the leg does not flag "
         "every edit", None),
        ("weights_off_by_one",
         lambda s: kit.needle(s, "weights[i]);", "weights[i - 1]);"), True,
         "the radial weight one row down: a half-cell error in (ir + ir0), in bounds "
         "for every i >= 1", None),
        ("divisor_off_by_one",
         lambda s: kit.needle(s, "divisor[i - 1]);", "divisor[max(i - 2, 0)]);"), True,
         "the divisor one row down, clamped in bounds: the row-constant (ir + ir0) - "
         "0.5 the difference lands on, shifted", None),
        ("negative_zero_row_zero",
         lambda s: kit.needle(s, "float2 acc = float2(0.0f, 0.0f);",
                              "float2 acc = float2(-0.0f, -0.0f);"), True,
         "row 0 written as -0.0: the array path's zeros_like row is +0.0 and this "
         "family compares words", None),
    ]
    if cyl.PREFIX_WALL_ROW[sub_step]:
        table.append((
            "wall_row_dropped",
            lambda s: kit.needle(s, srci, "src[min(i, nxi - 2) * nyz + base]"), True,
            "the B side's zero wall row replaced by the LAST VALID source row (in "
            "bounds, so the needle is deterministic): the wall row's own prefix entry "
            "is what Bz's last-row forward difference reads. SCOPED to the physical "
            "classes: the edit touches ONE word pair per column, and on a single-column "
            "extent a wide-dynamic last row twenty-four decades below its neighbour is "
            "absorbed by the subtraction while a signed-zero-mix last row that is "
            "exactly zero makes the edit a no-op (measured 49/52 unscoped in the "
            "_cyl3 cut, the three misses all on nz = 1 extents)",
            (UNIFORM, CANCELLING)))
    return tuple(table)


def leg_mutations(payload: Dict[str, Any], out: str, cases: Sequence[Case]) -> None:
    _probe, expansion = bound_expansion()
    harness = kit.MutationHarness(payload, out, key="mutations")
    labels = [label for label, *_rest in mutation_table("step_B")]
    labels += [label for label, *_rest in mutation_table("step_D") if label not in labels]
    for label in labels:
        per_step = {sub_step: dict(zip(("transform", "must_catch", "why", "classes"),
                                       entry[1:]))
                    for sub_step in SUB_STEPS
                    for entry in mutation_table(sub_step) if entry[0] == label}
        first = next(iter(per_step.values()))
        must_catch = first["must_catch"]
        why = first["why"]
        classes = first["classes"]
        scope = tuple(VALUE_CLASSES if classes is None else classes)
        functions: Dict[str, kit.Counter] = {}
        missed = False
        for sub_step, spec in per_step.items():
            shipped = scan.cylindrical_complex_prefix_source(sub_step, expansion)
            try:
                mutant = spec["transform"](shipped)
            except LookupError:
                missed = True
                continue
            assert mutant != shipped, label
            functions[sub_step] = kit.Counter(getattr(compile_source(mutant), scan.KERNEL))
        caught = ran = 0
        by_class: Dict[str, int] = {name: 0 for name in VALUE_CLASSES}
        ran_by_class: Dict[str, int] = {name: 0 for name in VALUE_CLASSES}
        by_step: Dict[str, int] = {}
        uncaught_out_of_scope: List[str] = []
        total_differing = total_words = 0
        for case in cases:
            counter = functions.get(case.sub_step)
            if counter is None:
                continue
            differ, compared, _got = _compare_case(case, counter)
            total_words += compared
            total_differing += differ
            ran_by_class[case.value_class] += 1
            if differ:
                by_class[case.value_class] += 1
                by_step[case.sub_step] = by_step.get(case.sub_step, 0) + 1
            if case.value_class in scope:
                ran += 1
                caught += int(bool(differ))
            elif not differ:
                uncaught_out_of_scope.append(case.label)
        launches = sum(counter.launches for counter in functions.values())
        verdict = kit.MutationHarness.verdict(missed, ran, launches, caught)
        harness.record(label, verdict, launches, caught, ran, must_catch, why, extra={
            "sub_steps": sorted(per_step), "scope_classes": list(scope),
            "caught_by_class": by_class, "ran_by_class": ran_by_class,
            "caught_by_sub_step": by_step, "differing_words": total_differing,
            "compared_words": total_words,
            "unscoped_cases_not_caught": uncaught_out_of_scope})

    # THE CONTRACTION GUARD, recorded: the shipped source rebuilt with the fast mode.
    # `acc + (w - prev) * r` is a multiply-add the compiler may fuse under fast
    # contraction, so unlike the real scan's measured-identical fast arm this one is
    # expected to move; whichever way it lands is recorded, not asserted.
    fast: Dict[str, kit.Counter] = {
        sub_step: kit.Counter(shipped_function(sub_step, expansion, shaders.CONTRACT_FAST))
        for sub_step in SUB_STEPS}
    caught = ran = 0
    total_differing = 0
    by_class = {name: 0 for name in VALUE_CLASSES}
    for case in cases:
        differ, _compared, _got = _compare_case(case, fast[case.sub_step])
        ran += 1
        total_differing += differ
        if differ:
            caught += 1
            by_class[case.value_class] += 1
    launches = sum(counter.launches for counter in fast.values())
    harness.record("contraction_fast", kit.MutationHarness.verdict(False, ran, launches,
                                                                  caught),
                   launches, caught, ran, None,
                   "the shipped source with the contraction guard removed; the multiply-"
                   "add acc + (w - prev) * r may fuse, so this measures whether the "
                   "guard is load-bearing on THIS kernel",
                   extra={"caught_by_class": by_class, "differing_words": total_differing})


# ---------------------------------------------------------------------------
# LEG disarm — the shipped bytes through the harness path
# ---------------------------------------------------------------------------

def leg_disarm(payload: Dict[str, Any], out: str, cases: Sequence[Case]) -> None:
    _probe, expansion = bound_expansion()
    counters = {sub_step: kit.Counter(shipped_function(sub_step, expansion))
                for sub_step in SUB_STEPS}
    total = 0
    for case in cases:
        differ, _compared, _got = _compare_case(case, counters[case.sub_step])
        total += differ
    launches = sum(counter.launches for counter in counters.values())
    row = {"launches": launches, "cases": len(cases), "differing": total,
           "passed": launches == len(cases) and total == 0}
    payload["legs"]["disarm"] = row
    save(payload, out)
    log(f"[disarm] launches={launches} differing={total}")
    assert row["passed"], row


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def main(argv: Sequence[str]) -> int:
    parser = kit.argument_parser(__doc__ or "")
    args = parser.parse_args(list(argv))
    started = time.time()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    payload: Dict[str, Any] = {
        "gate": "metal_cylindrical_complex_scan",
        "family": scan.FAMILY,
        "kernel": scan.KERNEL,
        "oracle": "meep_gpu.stepping.cylindrical_rderiv_prefix, called",
        "corpus_rows": [{"leg": leg, "row": row, "label": label, "shape": list(shape)}
                        for leg, row, label, shape in CORPUS_ROWS],
        "environment": kit.environment_stamp(),
        "probe_environment": ENVIRONMENT,
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
    }
    save(payload, out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device available")
    reasons.extend(subnormal.mps_policy_reasons())
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    probe_arm, expansion = bound_expansion()
    payload["expansion"] = {"probe_bound": probe_arm, "measured": expansion,
                            "other_arms": [arm for arm in EXPANSION_ARMS
                                           if arm != expansion]}
    save(payload, out)

    wanted = kit.wanted_legs(args.legs)
    cases: List[Case] = []
    if not wanted or {"prefix", "mutations", "disarm"} & set(wanted):
        log(f"[cases] building {len(SHAPES) * len(SUB_STEPS) * len(VALUE_CLASSES)} "
            f"cases: {len(SHAPES)} extents x {len(SUB_STEPS)} sub-steps x "
            f"{len(VALUE_CLASSES)} classes")
        cases = build_cases()

    legs: List[Tuple[str, Callable[[Dict[str, Any], str], None]]] = [
        ("bindings", leg_bindings),
        ("corpus_shapes", leg_corpus_shapes),
        ("prefix", lambda payload, out: leg_prefix(payload, out, cases)),
        ("absorption", leg_absorption),
        ("increment_sign", leg_increment_sign),
        ("extra_extents", leg_extra_extents),
        ("mutations", lambda payload, out: leg_mutations(payload, out, cases)),
        ("disarm", lambda payload, out: leg_disarm(payload, out, cases)),
    ]
    ran = kit.run_legs(legs, payload, out, wanted)

    kit.provenance(
        os.path.dirname(out),
        {"meep_gpu/metal_kernels/cylindrical_complex_scan.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                          "cylindrical_complex_scan.py"),
         "meep_gpu/metal_kernels/cylindrical_real.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels", "cylindrical_real.py"),
         "meep_gpu/metal_kernels/templates.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels", "templates.py"),
         "meep_gpu/metal_kernels/shaders.py":
             os.path.join(API_ROOT, "meep_gpu", "metal_kernels", "shaders.py"),
         "meep_gpu/stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
         "parity/meep_gpu/gate_metal_cylindrical_complex_scan.py":
             os.path.abspath(__file__)},
        kernel_sources={label: source for arm in EXPANSION_ARMS
                        for label, source in
                        scan.enumerate_cylindrical_complex_scan_sources(arm).items()},
        name="provenance.json")

    compared = 0
    certified = True
    prefix_rows = payload["legs"].get("prefix", ())
    for row in prefix_rows:
        compared += int(row["compared"]) * (1 + len(row["differing_other_arms"]))
        if row["differing"] or any(row["differing_other_arms"].values()):
            certified = False
    for row in payload["legs"].get("absorption", ()):
        compared += int(row["compared"])
        if row["differing"] or not row["reference_all_positive_zero"]:
            certified = False
    for row in payload["legs"].get("extra_extents", ()):
        compared += int(row["compared"])
        if row["differing"]:
            certified = False
    if "bindings" in payload["legs"] and not all(
            row["passed"] for row in payload["legs"]["bindings"]):
        certified = False
    if "corpus_shapes" in payload["legs"] and payload["legs"]["corpus_shapes"].get(
            "mismatches"):
        certified = False
    mutations = payload["legs"].get("mutations", [])
    armed = [row for row in mutations if row["must_catch"] is True]
    nulls = [row for row in mutations if row["must_catch"] is False]
    for row in armed:
        if row["caught"] != row["ran"] or row["ran"] == 0:
            certified = False
    for row in nulls:
        if row["caught"] != 0:
            certified = False
    if "disarm" in payload["legs"] and not payload["legs"]["disarm"]["passed"]:
        certified = False

    return kit.summarize(
        payload, out,
        claim=("ONE dispatch of the device column-serial complex scan reproduces "
               "stepping.cylindrical_rderiv_prefix on complex64 storage, as uint32 words, "
               "at every radial extent of the sixteen cylindrical complex H->D corpus "
               "rows, at both ir0 values, over four value classes and both expansion "
               "arms, with the reciprocal-multiply divide spelling that numpy's "
               "complex64 / float32 IS and the true divide caught as a defect"),
        scope=(f"{len(SHAPES)} extents x {len(SUB_STEPS)} sub-steps x "
               f"{len(VALUE_CLASSES)} classes = {len(prefix_rows)} device cases at the "
               f"probe-bound arm plus the other arm on every case; the +-0 lattice as "
               f"a predicted null-sign case; the two withdraw rows' extents uncredited"),
        stated_weakness=(
            "this certifies the SCAN alone: no curl, no constitutive, no complete step, "
            "no launch saved and no timing. The product that consumes it (the two-launch "
            "complex H->D pair) carries its own gate. The subnormal band is a REFUSAL on "
            "this executor, checked per case, never a comparison. The signed-zero "
            "refinement of the divide spelling is unobservable on the prefix and is "
            "recorded, not certified, for the increment"),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"cases": len(prefix_rows),
               "mutations_armed": len(armed),
               "mutations_caught_of_armed": sum(1 for row in armed
                                                if row["caught"] == row["ran"]),
               "mutations_null_controls": len(nulls),
               "mutation_verdicts": {row["mutation"]: row["verdict"] for row in mutations},
               "expansion": payload.get("expansion")})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
