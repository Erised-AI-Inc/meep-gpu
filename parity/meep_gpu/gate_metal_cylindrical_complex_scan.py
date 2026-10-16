"""Byte gate for the Metal COMPLEX radial scan — ``cyl_complex_rderiv_difference``
against ``stepping.cylindrical_rderiv_difference`` on complex64 storage.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: at every radial extent of the sixteen
``cylindrical complex`` H->D corpus rows on the Metal board, at both ``ir0`` values the
array path uses (0.0 with the zero wall row and a forward difference on the B side, 0.5
raw with a backward difference on the D side), over four value classes and two chunk
divisions (one radial chunk, and two with the cut where an r layer would start), ONE
dispatch of the device column-serial complex scan writes a radial-difference volume
that is BIT-IDENTICAL, as uint32 words, to ``stepping.cylindrical_rderiv_difference``
CALLED on the same complex64 input with the same chunk starts — never re-derived,
never approximated, ``allclose`` nowhere.

WHAT THE KERNEL IS: MEEP's ``f_rderiv_int`` summed per part in every structure chunk
from that chunk's first row, ``rinv`` multiplied (the double quotient stored as a
realnum), and the stencil's difference of two partial sums stored. The complex loop is
the real scan's with ``float2`` registers (``cylindrical_complex_scan.FLOAT2_EDITS``);
``float2 * float`` is one rounded product per part, which is what the array path's
per-part forms compute. No complex product and no divide remain.

THE LEGS

  1  bindings        nine buffers, compiles on both sub-steps, the rinv multiply on the
                     increment and no divide of a value, the per-chunk restart, the
                     contraction guard exactly once, the same text at every arm
  2  corpus_shapes   the thirteen (nr, nz) extents here ARE the sixteen rows' extents in
                     the census record, read from the record when it is present
  3  radial          13 extents x 2 sub-steps x 4 value classes x 2 divisions = 208
                     cases, the device scan against the array path's own function, the
                     subnormal-free precondition CHECKED on every case, a movement floor,
                     and on the two-chunk division the one-sum control (the reference
                     with one sum from the axis), which must differ
  4  absorption      the +-0 lattice: the oracle's difference is +0.0 in EVERY word, and
                     so is the device's. Recorded as a predicted null-sign case, not as a
                     discriminating comparison
  5  extra_extents   the two withdraw_seam rows' extents (100, 1335), measured for the
                     ladder and NOT credited to any row
  6  mutations       armed defects through the launch-counted harness — the restart
                     dropped or one row late, the quotient divided in the kernel, the
                     rounded increment stored, the parts swapped, both off-by-ones, the
                     short loop, the wall row given a value (must be caught); the
                     commuted add (must NOT be caught); a -0 start and the
                     fast-contraction variant (recorded)
  7  disarm          the shipped bytes through the same harness path must not diverge

WHAT THE GATE REFUSES TO INFER

* **A no-op agreeing with a no-op is trivially identical.** Every case must produce a
  floor of nonzero output words, and the +-0 lattice — whose output is all zeros by the
  oracle's own arithmetic — is recorded as predicted-null rather than counted as a
  discriminating pass.
* **A value class that entered the subnormal band is a REFUSAL**, not a comparison: MPS
  flushes natively, the oracle keeps, and this claim rides a checked precondition.
* **A mutation that never launched is DISARMED and fails; one whose needle matched
  nothing is NEEDLE-MISSED and fails** (``metal_gate_kit.MutationHarness``).

Progress reporting: one flushed line per case, every row appended and fsynced as it lands.
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

#: The two chunk divisions every case runs under: one radial chunk from the axis, and
#: two with the cut where an r layer would start (:func:`division_starts`).
ONE_CHUNK = "one_chunk"
TWO_CHUNK = "two_chunk"
DIVISIONS: Tuple[str, ...] = (ONE_CHUNK, TWO_CHUNK)

#: The four compared value classes, in scoring order. ``pm_zero_lattice`` is NOT one of
#: them: its oracle output is all +0.0 (leg ``absorption``), so it discriminates no
#: sign and moves no word.
UNIFORM = "uniform"
WIDE_DYNAMIC = "wide_dynamic"
CANCELLING = "cancelling"
SIGNED_ZERO_MIX = "signed_zero_mix"
VALUE_CLASSES: Tuple[str, ...] = (UNIFORM, WIDE_DYNAMIC, CANCELLING, SIGNED_ZERO_MIX)

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
        # increments of BOTH signs arise, which a +0 sum must absorb.
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


def division_starts(nr: int, division: str, sub_step: str) -> Optional[np.ndarray]:
    """The chunk starts of one division, as ``stepping._cylindrical_scan_starts`` gives
    them for the sub-step's target: ``None`` for one chunk, else one start per row,
    ``(nr, 1, 1)``. A chunk box ``[lo, hi]`` in doubled r owns ``lo < p <= hi``, so the
    cut row belongs to the UPPER chunk for Bz (half-integer, ``p = 2g + 1``) and to the
    LOWER chunk for Dz (integer, ``p = 2g``)."""
    if division == ONE_CHUNK:
        return None
    cut = nr - max(2, nr // 8)
    rows = np.arange(nr)
    upper = rows >= cut if sub_step == "step_B" else rows > cut
    return np.where(upper, cut, 0).astype(np.int64).reshape(-1, 1, 1)


def reference_radial(sub_step: str, source: np.ndarray,
                     starts: Optional[np.ndarray]) -> np.ndarray:
    """``stepping``'s own radial difference — CALLED, never re-derived."""
    return stepping.cylindrical_rderiv_difference(
        np, scanned_source(sub_step, source), cyl.PREFIX_IR0[sub_step], starts,
        int(source.shape[0]), backward=bool(cyl.SUB_STEPS[sub_step]["backward"]))


class Case:
    """One (extent, sub-step, class, division) input with its oracle and tensors."""

    __slots__ = ("label", "shape", "sub_step", "value_class", "division", "source",
                 "starts", "expected", "one_sum", "weights", "rinv", "src_t", "w_t",
                 "r_t", "s_t")

    def __init__(self, shape: Tuple[int, int], sub_step: str, value_class: str,
                 seed: int, division: str = TWO_CHUNK,
                 source: Optional[np.ndarray] = None) -> None:
        import torch  # noqa: PLC0415

        nr, nz = shape
        self.label = f"nr{nr}_nz{nz}/{sub_step}/{value_class}/{division}"
        self.shape = (nr, 1, nz)
        self.sub_step = sub_step
        self.value_class = value_class
        self.division = division
        self.source = (make_input(self.shape, value_class, seed) if source is None
                       else source)
        self.starts = division_starts(nr, division, sub_step)
        self.expected = reference_radial(sub_step, self.source, self.starts)
        self.one_sum = (self.expected if self.starts is None
                        else reference_radial(sub_step, self.source, None))
        self.weights, self.rinv = scan.prefix_row_vectors(sub_step, nr)
        line = cyl.scan_starts(sub_step, self.starts, nr)
        self.src_t = torch.from_numpy(np.ascontiguousarray(self.source).reshape(-1)
                                      ).to("mps")
        self.w_t = torch.from_numpy(np.ascontiguousarray(self.weights, dtype=np.float32)
                                    ).to("mps")
        self.r_t = torch.from_numpy(np.ascontiguousarray(self.rinv, dtype=np.float32)
                                    ).to("mps")
        self.s_t = torch.from_numpy(np.ascontiguousarray(line, dtype=np.int32)).to("mps")

    def launch(self, function: Callable[..., Any]) -> np.ndarray:
        """One dispatch into a SENTINEL-filled output, so a kernel that never ran
        cannot compare equal."""
        import torch  # noqa: PLC0415

        sentinel = np.full(self.shape, np.complex64(-7.5 - 7.5j), dtype=np.complex64)
        out = torch.from_numpy(sentinel.reshape(-1)).to("mps")
        function(*scan.scan_arguments(out, self.src_t, self.w_t, self.r_t, self.s_t,
                                      self.shape))
        torch.mps.synchronize()
        got = out.cpu().numpy().reshape(self.shape)
        assert got.shape == self.expected.shape, (self.label, got.shape,
                                                  self.expected.shape)
        return got


def build_cases(seed_base: int = 71000) -> List[Case]:
    cases: List[Case] = []
    for offset, (shape, sub_step, value_class) in enumerate(
            (shape, sub_step, value_class) for shape in SHAPES
            for sub_step in SUB_STEPS for value_class in VALUE_CLASSES):
        for division in DIVISIONS:
            cases.append(Case(shape, sub_step, value_class, seed_base + offset,
                              division))
    return cases


_FUNCTIONS: Dict[Tuple[str, str], Any] = {}


def shipped_function(sub_step: str,
                     contract: str = shaders.CONTRACT_OFF) -> Any:
    key = (sub_step, contract)
    if key not in _FUNCTIONS:
        _FUNCTIONS[key] = scan.compile_cylindrical_complex_scan(sub_step,
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

#: The increment line of each sub-step's loop: rinv MULTIPLIED, one product per part.
_INCREMENT = {"step_B": "float2 inc = rinv[g] * (w - w_prev);",
              "step_D": "float2 inc = rinv[g - 1] * (w - w_prev);"}


def leg_bindings(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for sub_step in SUB_STEPS:
        source = scan.cylindrical_complex_scan_source(sub_step)
        indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
        code = _code_lines(source)
        ok, error = _compiles(source)
        row = {
            "sub_step": sub_step,
            "buffer_indices": indices,
            "bindings": len(indices),
            "declared_bindings": scan.BINDINGS,
            "ceiling": MAX_BUFFER_BINDINGS,
            "headroom": MAX_BUFFER_BINDINGS - len(indices),
            "compiled": ok,
            "compile_error": error,
            "float2_volumes": (code.count("device float2*") + code.count(
                "device const float2*")),
            "rinv_multiplied": _INCREMENT[sub_step] in code,
            # The one `/` the compiler sees is the integer column decode.
            "no_divide_of_a_value": code.count("/") == 1 and "int j = int(idx) / nzi;"
                                    in code,
            "fast_math_absent": "fast::" not in code,
            "per_chunk_restart": ("if (s != owner) { acc = float2(0.0f, 0.0f); "
                                  "owner = s; }") in code,
            "difference_of_partial_sums": (
                "out[g * nyz + base] = nxt - acc;" if sub_step == "step_B"
                else "out[g * nyz + base] = acc - nxt;") in code,
            "contraction_guard_once": source.count(
                shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1,
            "no_math_mode_pragma": "math_mode" not in source and "denorm" not in source,
            "sum_starts_positive": "float2 acc = float2(0.0f, 0.0f);" in code,
            "wall_row_read": ("float2(0.0f, 0.0f)) * weights[g + 1]" in code
                             if cyl.PREFIX_WALL_ROW[sub_step]
                             else "weights[g + 1]" not in code),
            "source_sha256": templates.source_sha256(source),
        }
        row["passed"] = bool(
            ok and indices == list(range(scan.BINDINGS))
            and row["float2_volumes"] == 2 and row["rinv_multiplied"]
            and row["no_divide_of_a_value"] and row["fast_math_absent"]
            and row["per_chunk_restart"] and row["difference_of_partial_sums"]
            and row["contraction_guard_once"] and row["no_math_mode_pragma"]
            and row["sum_starts_positive"] and row["wall_row_read"])
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
# LEG radial — the device scan against the array path's own function
# ---------------------------------------------------------------------------

def _compare_case(case: Case, function: Callable[..., Any]) -> Tuple[int, int, np.ndarray]:
    got = case.launch(function)
    return differing(got, case.expected), int(words(case.expected).size), got


def leg_radial(payload: Dict[str, Any], out: str, cases: Sequence[Case]) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(cases, start=1):
        band_in = subnormal.census(case.source)
        band_ref = subnormal.census(case.expected)
        assert band_in == 0 and band_ref == 0, (
            f"{case.label}: the value class entered the subnormal band "
            f"(input {band_in}, reference {band_ref} words); a banded case is a "
            f"refusal on this executor, not a comparison")
        differ, compared, got = _compare_case(case, shipped_function(case.sub_step))
        moved = differing(got, np.zeros_like(got))
        kit.assert_moved(moved, f"{case.label} scan output is all zero",
                         floor=max(1, min(64, compared // 4)))
        control = differing(case.expected, case.one_sum)
        if case.division == TWO_CHUNK:
            kit.assert_moved(control, f"{case.label} one-sum control: one radial sum "
                             f"from the axis equals the two chunks' sums", floor=1)
        row = {
            "case": case.label, "index": index, "total": len(cases),
            "shape": list(case.shape), "sub_step": case.sub_step,
            "ir0": cyl.PREFIX_IR0[case.sub_step],
            "wall_row": bool(cyl.PREFIX_WALL_ROW[case.sub_step]),
            "value_class": case.value_class, "division": case.division,
            "chunk_starts": (sorted({int(s) for s in case.starts.reshape(-1)})
                             if case.starts is not None else [0]),
            "compared": compared, "differing": differ, "nonzero_words": moved,
            "one_sum_differing": control,
            "subnormal_words_input": band_in, "subnormal_words_reference": band_ref,
            "reference_zero_sign_census": values.zero_sign_census([case.expected]),
            "digest": kit.state_digest({"radial": case.expected}),
        }
        rows.append(row)
        payload["legs"]["radial"] = rows
        save(payload, out)
        log(f"[radial] case {index}/{len(cases)} {case.label:<52} compared={compared} "
            f"differing={differ} one_sum_differing={control} "
            f"({time.time() - started:.1f}s)")
    payload["legs"]["radial_summary"] = {
        "cases": len(rows), "compared": sum(row["compared"] for row in rows),
        "differing": sum(row["differing"] for row in rows),
        "two_chunk_cases": sum(1 for row in rows if row["division"] == TWO_CHUNK),
        "one_sum_differing_min_two_chunk": min(
            (row["one_sum_differing"] for row in rows if row["division"] == TWO_CHUNK),
            default=0),
        "extents": len(SHAPES), "sub_steps": list(SUB_STEPS),
        "value_classes": list(VALUE_CLASSES), "divisions": list(DIVISIONS),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG absorption — the +-0 lattice is all +0.0 on both sides
# ---------------------------------------------------------------------------

def leg_absorption(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for shape in SHAPES:
        for sub_step in SUB_STEPS:
            nr, nz = shape
            source = values.pm_zero_lattice_complex((nr, 1, nz))
            case = Case(shape, sub_step, "pm_zero_lattice", 0, TWO_CHUNK, source=source)
            census_in = values.zero_sign_census([source])
            census_out = values.zero_sign_census([case.expected])
            differ, compared, got = _compare_case(case, shipped_function(sub_step))
            all_positive_zero = (census_out["positive_zero_words"] == compared
                                 and census_out["negative_zero_words"] == 0)
            row = kit.predicted_null({
                "case": case.label, "shape": list(case.shape), "sub_step": sub_step,
                "compared": compared, "differing": differ,
                "input_zero_sign_census": census_in,
                "reference_zero_sign_census": census_out,
                "reference_all_positive_zero": all_positive_zero,
            }, ("every sum starts at an exact +0.0 on its chunk's first row and IEEE "
                "addition never yields -0.0 from a +0.0 accumulator, so each partial sum "
                "is +0.0 and so is the difference of two of them: the oracle's radial "
                "difference over a +-0 lattice is +0.0 in every word and the comparison "
                "discriminates no sign and moves no word"))
            rows.append(row)
            payload["legs"]["absorption"] = rows
            save(payload, out)
            log(f"[absorption] {case.label:<52} compared={compared} differing={differ} "
                f"reference +0={census_out['positive_zero_words']} "
                f"-0={census_out['negative_zero_words']}")
            assert all_positive_zero, row
            assert differ == 0, row
            assert census_in["negative_zero_words"] > 0, row


# ---------------------------------------------------------------------------
# LEG extra_extents — the two withdraw rows' ladders, uncredited
# ---------------------------------------------------------------------------

def leg_extra_extents(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for offset, (_leg, name, shape) in enumerate(WITHDRAW_ROWS):
        for sub_step in SUB_STEPS:
            case = Case((shape[0], shape[2]), sub_step, UNIFORM, 91000 + offset,
                        TWO_CHUNK)
            differ, compared, got = _compare_case(case, shipped_function(sub_step))
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

def _per_step(b_old: str, b_new: str, d_old: str, d_new: str
              ) -> Callable[[str, str], str]:
    """A transform whose needle differs by sub-step."""
    def transform(text: str, sub_step: str) -> str:
        if sub_step == "step_B":
            return kit.needle(text, b_old, b_new)
        return kit.needle(text, d_old, d_new)
    return transform


def _both(old: str, new: str) -> Callable[[str, str], str]:
    return lambda text, _sub_step: kit.needle(text, old, new)


#: ``(label, transform(text, sub_step), sub_steps, must_catch, why, classes,
#: divisions)``. ``classes`` and ``divisions`` are the SCOPE the verdict is taken over
#: (``None``: all). A mutation is scoped where a case can ABSORB its edit; the cases
#: outside the scope are still RUN and their counts recorded beside the verdict.
#: :data:`MULTI_COLUMN_ONLY` scopes by extent the same way.
MUTATIONS: Tuple[Tuple[str, Callable[[str, str], str], Tuple[str, ...],
                       Optional[bool], str, Optional[Tuple[str, ...]],
                       Optional[Tuple[str, ...]]], ...] = (
    ("sum_never_restarts",
     _both("{ acc = float2(0.0f, 0.0f); owner = s; }", "{ owner = s; }"), SUB_STEPS,
     True, "ONE radial sum from the axis instead of MEEP's f_rderiv_int per structure "
     "chunk: every row from the cut up reads another chunk's sum. Scoped to the "
     "two-chunk division, where it exists", None, (TWO_CHUNK,)),
    ("restart_one_row_late",
     _both("int s = starts[g];", "int s = starts[g > 0 ? g - 1 : 0];"), SUB_STEPS, True,
     "the restart one row after the chunk's first row (the bound-off-by-one class on "
     "the radial walk)", None, (TWO_CHUNK,)),
    ("quotient_divided_in_kernel",
     _per_step("float2 inc = rinv[g] * (w - w_prev);",
               "float2 inc = (w - w_prev) / (float(g + 1) - 0.5f);",
               "float2 inc = rinv[g - 1] * (w - w_prev);",
               "float2 inc = (w - w_prev) / ((float(g) + 0.5f) - 0.5f);"),
     SUB_STEPS, True,
     "the float32 divide where MEEP multiplies by rinv, the double quotient stored as a "
     "realnum: algebraically equal, a different float32 number", None, None),
    ("rounded_increment_stored",
     _per_step("out[g * nyz + base] = nxt - acc;", "out[g * nyz + base] = inc;",
               "out[g * nyz + base] = acc - nxt;", "out[g * nyz + base] = -inc;"),
     SUB_STEPS, True,
     "the increment stored where the stencil's difference of two partial sums is: "
     "(acc + inc) - acc is not inc once acc is not +0", None, None),
    ("difference_reversed",
     _per_step("out[g * nyz + base] = nxt - acc;", "out[g * nyz + base] = acc - nxt;",
               "out[g * nyz + base] = acc - nxt;", "out[g * nyz + base] = nxt - acc;"),
     SUB_STEPS, True, "forward on B and backward on D, each reversed", None, None),
    ("parts_swapped",
     _per_step("rinv[g] * (w - w_prev);", "rinv[g] * (w - w_prev).yx;",
               "rinv[g - 1] * (w - w_prev);", "rinv[g - 1] * (w - w_prev).yx;"),
     SUB_STEPS, True, "the real and imaginary parts of the increment exchanged",
     None, None),
    ("weights_off_by_one",
     _per_step("* weights[g + 1];", "* weights[g];",
               "float2 w = src[g * nyz + base] * weights[g];",
               "float2 w = src[g * nyz + base] * weights[g - 1];"),
     SUB_STEPS, True, "the radial weight one row down: a half-cell error in (ir + ir0)",
     None, None),
    ("rinv_off_by_one",
     _per_step("float2 inc = rinv[g] *", "float2 inc = rinv[g > 0 ? g - 1 : 0] *",
               "float2 inc = rinv[g - 1] *", "float2 inc = rinv[g > 1 ? g - 2 : 0] *"),
     SUB_STEPS, True, "rinv one row down, clamped in bounds", None, None),
    ("loop_stops_one_row_short",
     _per_step("for (int g = 0; g < nxi; ++g)", "for (int g = 0; g < nxi - 1; ++g)",
               "for (int g = 1; g < nxi; ++g)", "for (int g = 1; g < nxi - 1; ++g)"),
     SUB_STEPS, True, "the last row left holding the sentinel", None, None),
    ("column_decomposition_swapped",
     _both("int k = int(idx) % nzi;\n    int j = int(idx) / nzi;",
           "int j = int(idx) % nzi;\n    int k = int(idx) / nzi;"), SUB_STEPS, True,
     "the (phi, z) column decomposition swapped: ny is 1, so every thread but one "
     "returns at the `j >= nyi` guard (single-column extents are a structural null)",
     None, None),
    ("wall_row_carries_a_value",
     _both(": float2(0.0f, 0.0f)) * weights[g + 1];",
           ": float2(1.0f, 1.0f)) * weights[g + 1];"), ("step_B",), True,
     "the B side's zero wall row given a value (a literal, deterministic). SCOPED to the "
     "physical classes: the edit touches one word pair per column, and a wide-dynamic "
     "last row far above the literal absorbs it in the subtraction",
     (UNIFORM, CANCELLING), None),
    ("commuted_add", _both("float2 nxt = acc + inc;", "float2 nxt = inc + acc;"),
     SUB_STEPS, False, "float addition IS commutative; the control that shows the leg "
     "does not flag every edit", None, None),
    ("sum_starts_at_negative_zero",
     _both("    float2 acc = float2(0.0f, 0.0f);",
           "    float2 acc = float2(-0.0f, -0.0f);"), SUB_STEPS, None,
     "RECORDED ONLY: a -0 start reaches a stored word only where the first increment "
     "of a chunk is an exact +0 on the D side", None, None),
)


#: Mutations whose edit is a STRUCTURAL NULL on a single-column extent (nz = 1, so
#: the (phi, z) decomposition has one column and the swap maps it onto itself): three
#: corpus rows have one. Scored over the multi-column extents only, as the classes and
#: divisions are; the single-column cases are still run and recorded.
MULTI_COLUMN_ONLY = frozenset({"column_decomposition_swapped"})


def _columns(case: Case) -> int:
    return int(case.shape[1]) * int(case.shape[2])


def leg_mutations(payload: Dict[str, Any], out: str, cases: Sequence[Case]) -> None:
    harness = kit.MutationHarness(payload, out, key="mutations")
    for label, transform, sub_steps, must_catch, why, classes, divisions in MUTATIONS:
        class_scope = tuple(VALUE_CLASSES if classes is None else classes)
        division_scope = tuple(DIVISIONS if divisions is None else divisions)
        functions: Dict[str, kit.Counter] = {}
        missed = False
        for sub_step in sub_steps:
            shipped = scan.cylindrical_complex_scan_source(sub_step)
            try:
                mutant = transform(shipped, sub_step)
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
            if (case.value_class in class_scope and case.division in division_scope
                    and (label not in MULTI_COLUMN_ONLY or _columns(case) > 1)):
                ran += 1
                caught += int(bool(differ))
            elif not differ:
                uncaught_out_of_scope.append(case.label)
        launches = sum(counter.launches for counter in functions.values())
        verdict = kit.MutationHarness.verdict(missed, ran, launches, caught)
        harness.record(label, verdict, launches, caught, ran, must_catch, why, extra={
            "sub_steps": list(sub_steps), "scope_classes": list(class_scope),
            "scope_divisions": list(division_scope),
            "scope_multi_column_only": label in MULTI_COLUMN_ONLY,
            "caught_by_class": by_class, "ran_by_class": ran_by_class,
            "caught_by_sub_step": by_step, "differing_words": total_differing,
            "compared_words": total_words,
            "out_of_scope_cases_not_caught": uncaught_out_of_scope})

    # THE CONTRACTION GUARD, recorded: the shipped source rebuilt with the fast mode.
    # `acc + rinv * (w - w_prev)` is a multiply-add the compiler may fuse under fast
    # contraction; whichever way it lands is recorded, not asserted.
    fast: Dict[str, kit.Counter] = {
        sub_step: kit.Counter(shipped_function(sub_step, shaders.CONTRACT_FAST))
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
                   "add acc + rinv * (w - w_prev) may fuse, so this measures whether the "
                   "guard is load-bearing on THIS kernel",
                   extra={"caught_by_class": by_class, "differing_words": total_differing})


# ---------------------------------------------------------------------------
# LEG disarm — the shipped bytes through the harness path
# ---------------------------------------------------------------------------

def leg_disarm(payload: Dict[str, Any], out: str, cases: Sequence[Case]) -> None:
    counters = {sub_step: kit.Counter(shipped_function(sub_step))
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
        "oracle": "meep_gpu.stepping.cylindrical_rderiv_difference, called",
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

    wanted = kit.wanted_legs(args.legs)
    cases: List[Case] = []
    if not wanted or {"radial", "mutations", "disarm"} & set(wanted):
        log(f"[cases] building "
            f"{len(SHAPES) * len(SUB_STEPS) * len(VALUE_CLASSES) * len(DIVISIONS)} "
            f"cases: {len(SHAPES)} extents x {len(SUB_STEPS)} sub-steps x "
            f"{len(VALUE_CLASSES)} classes x {len(DIVISIONS)} divisions")
        cases = build_cases()

    legs: List[Tuple[str, Callable[[Dict[str, Any], str], None]]] = [
        ("bindings", leg_bindings),
        ("corpus_shapes", leg_corpus_shapes),
        ("radial", lambda payload, out: leg_radial(payload, out, cases)),
        ("absorption", leg_absorption),
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
        kernel_sources={label: source
                        for label, source in
                        scan.enumerate_cylindrical_complex_scan_sources().items()},
        name="provenance.json")

    compared = 0
    certified = True
    radial_rows = payload["legs"].get("radial", ())
    for row in radial_rows:
        compared += int(row["compared"])
        if row["differing"]:
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
               "stepping.cylindrical_rderiv_difference on complex64 storage, as uint32 "
               "words, at every radial extent of the sixteen cylindrical complex H->D "
               "corpus rows, at both ir0 values, over four value classes and two chunk "
               "divisions: MEEP's f_rderiv_int summed per part in each chunk, rinv "
               "multiplied, the difference of two partial sums stored"),
        scope=(f"{len(SHAPES)} extents x {len(SUB_STEPS)} sub-steps x "
               f"{len(VALUE_CLASSES)} classes x {len(DIVISIONS)} divisions = "
               f"{len(radial_rows)} device cases; the +-0 lattice as a predicted null-sign case; the "
               f"two withdraw rows' extents uncredited"),
        stated_weakness=(
            "this certifies the SCAN alone: no curl, no constitutive, no complete step, "
            "no launch saved and no timing. The product that consumes it (the two-launch "
            "complex H->D pair) carries its own gate. The subnormal band is a REFUSAL on "
            "this executor, checked per case, never a comparison. The chunk divisions "
            "are constructed (one cut per extent), not read from a lifted run; the "
            "run's own starts are the consuming products' gates' to bind"),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"cases": len(radial_rows),
               "mutations_armed": len(armed),
               "mutations_caught_of_armed": sum(1 for row in armed
                                                if row["caught"] == row["ran"]),
               "mutations_null_controls": len(nulls),
               "mutation_verdicts": {row["mutation"]: row["verdict"] for row in mutations}})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
