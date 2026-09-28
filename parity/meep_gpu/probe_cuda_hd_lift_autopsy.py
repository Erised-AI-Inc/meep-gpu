#!/usr/bin/env python3
"""Autopsy one lifted corpus row on which the H->D weld's lift leg diverged.

WHY A PROBE AND NOT A LEG. The gate's lift leg answers ONE question per row -- does the
weld step identically to the array path -- because that is the claim, and 129 rows x
four references is a different budget. When a row diverges, the question changes: WHICH
arrangement moved, and is the difference attributable to the weld at all? That is what
this probe answers, and it answers it the only way that can distinguish the two: it
drives the CERTIFIED SINGLES beside the weld on the same row, from the same state, and
reports the three-way comparison per differing word.

THE DISCRIMINATING SHAPE. On the sibling Metal backend the same leg found rows where
EVERY dispatched arrangement -- the certified singles, both compositions and the weld --
was bit-identical to every other, and ALL of them disagreed with the array path by one
or two units in the last place at magnitudes around 1e-32 inside the absorber. That is a
finding about the backend's agreement with the array path on those rows and NOT about
the weld, and no leg that compares only two engines can tell the two apart. So this
probe reports, per differing word:

* the two bit patterns and their signed-magnitude distance (the ulp distance for two
  words of the same sign);
* the IEEE class of each (zero / subnormal / normal / inf / nan);
* whether the weld agrees with the CERTIFIED SINGLES at that word, which is the whole
  question;
* the cell's coordinates and whether it lies inside the absorber.

It also censuses the reference's own stored state for subnormal words at every step, so
a row that entered the denormal band before the divergence is named rather than argued
about.

WHAT THIS PROBE DOES NOT DO: it does not change any verdict. A gate that does not
release is the finding; this exists so that finding can be reported with a cause rather
than as a number.

Usage (from ``parity/meep_gpu``)::

    python probe_cuda_hd_lift_autopsy.py --leg examples --target <script> \\
        --out results/<campaign>/autopsy/<row>.json [--steps 60]
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = str(next(parent for parent in Path(_HERE).parents
                     if (parent / "meep_gpu" / "cuda_kernels").is_dir()))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cupy as cp  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_fused_hd_pair as gate  # noqa: E402

from meep_gpu import stepping  # noqa: E402

#: How many differing words to transcribe per volume. A transcript is for reading; the
#: COUNT is the measurement and is reported whole.
TRANSCRIPT_CAP = 12


def _classify(word: int) -> str:
    exponent = (word >> 23) & 0xFF
    mantissa = word & 0x7FFFFF
    if exponent == 0:
        return "zero" if mantissa == 0 else "subnormal"
    if exponent == 0xFF:
        return "inf" if mantissa == 0 else "nan"
    return "normal"


def _words(array: Any) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(gate.to_host(array)).tobytes(),
                         dtype=np.uint32)


def _autopsy(reference: Dict[str, np.ndarray], other: Dict[str, np.ndarray],
             third: Optional[Dict[str, np.ndarray]], shape: Sequence[int],
             ) -> Dict[str, Any]:
    """Per volume: how many words differ, what class they are, and who agrees."""
    out: Dict[str, Any] = {}
    for name in sorted(reference):
        left, right = _words(reference[name]), _words(other[name])
        if left.shape != right.shape:
            out[name] = {"shape_mismatch": [int(left.size), int(right.size)]}
            continue
        where = np.flatnonzero(left != right)
        if where.size == 0:
            continue
        singles = _words(third[name]) if third is not None and name in third else None
        agrees_with_singles = (None if singles is None
                               else int(np.count_nonzero(right[where] == singles[where])))
        transcript = []
        for index in where[:TRANSCRIPT_CAP]:
            a, b = int(left[index]), int(right[index])
            coordinates = np.unravel_index(int(index), tuple(int(n) for n in shape)) \
                if int(np.prod([int(n) for n in shape])) == left.size else None
            transcript.append({
                "index": int(index),
                "cell": [int(c) for c in coordinates] if coordinates is not None else None,
                "array_path_word": a, "weld_word": b,
                "array_path_class": _classify(a), "weld_class": _classify(b),
                "signed_magnitude_distance": abs(a - b),
                "same_sign": (a >> 31) == (b >> 31),
                "array_path_value": float(np.frombuffer(
                    np.uint32(a).tobytes(), dtype=np.float32)[0]),
                "weld_value": float(np.frombuffer(
                    np.uint32(b).tobytes(), dtype=np.float32)[0]),
                "singles_word": (None if singles is None else int(singles[index])),
            })
        out[name] = {
            "differing_words": int(where.size),
            "words_in_volume": int(left.size),
            "array_path_subnormals": int(np.count_nonzero(
                ((left & np.uint32(0x7F800000)) == 0)
                & ((left & np.uint32(0x007FFFFF)) != 0))),
            "weld_subnormals": int(np.count_nonzero(
                ((right & np.uint32(0x7F800000)) == 0)
                & ((right & np.uint32(0x007FFFFF)) != 0))),
            "words_where_the_weld_agrees_with_the_certified_singles": agrees_with_singles,
            "transcript": transcript,
        }
    return out


def run(driver: Any, steps: int) -> Dict[str, Any]:
    """Drive THREE arrangements on one lifted row and compare per complete step.

    The array path is the run's own state -- it is what the run continues from -- and
    the weld and the certified singles are each driven from the SAME captured state and
    then discarded, so neither can contaminate the other or the reference.
    """
    family = gate._family()  # noqa: SLF001
    weld_plans, _state = gate._weld_plan_for(driver)  # noqa: SLF001
    weld_shim = gate.Shim(weld_plans)

    singles_shim: Optional[Any] = None
    try:
        singles = gate._certified_single_launchers(  # noqa: SLF001
            driver.fields, driver.grid, driver.pml, float(driver.grid.dt / driver.grid.dx))
        singles_shim = gate.Shim({"update_H": singles["update_H"],
                                  "step_D": singles["step_D"]})
    except Exception as error:  # noqa: BLE001
        singles_error = repr(error)[:400]
    else:
        singles_error = None

    detached = {"dft": list(getattr(driver, "_dft_monitors", ()) or ()),
                "flux": list(getattr(driver, "_flux_monitors", ()) or ())}
    driver._dft_monitors = []  # noqa: SLF001
    driver._flux_monitors = []  # noqa: SLF001

    shape = tuple(int(n) for n in driver.grid.shape)
    per_step: List[Dict[str, Any]] = []
    first: Optional[Dict[str, Any]] = None
    started = time.perf_counter()
    for step in range(steps):
        before = gate._capture(driver)  # noqa: SLF001
        driver._fast_path = weld_shim  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        driver.step()
        cp.cuda.runtime.deviceSynchronize()
        weld_state = gate._capture(driver)  # noqa: SLF001
        gate._restore(driver, before)  # noqa: SLF001

        singles_state = None
        if singles_shim is not None:
            driver._fast_path = singles_shim  # noqa: SLF001
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            singles_state = gate._capture(driver)  # noqa: SLF001
            gate._restore(driver, before)  # noqa: SLF001

        driver._fast_path = None  # noqa: SLF001
        driver.step()
        cp.cuda.runtime.deviceSynchronize()
        array_state = gate._capture(driver)  # noqa: SLF001

        subnormals = sum(gate._subnormal_words(v)  # noqa: SLF001
                         for v in array_state["volumes"].values())
        weld_moved = gate.compare(array_state["volumes"], weld_state["volumes"])
        singles_moved = ({} if singles_state is None
                         else gate.compare(array_state["volumes"],
                                           singles_state["volumes"]))
        per_step.append({
            "step": step + 1,
            "array_path_subnormal_words": subnormals,
            "weld_differing_volumes": weld_moved,
            "certified_singles_differing_volumes": singles_moved,
        })
        if weld_moved and first is None:
            first = {
                "step": step + 1,
                "weld_vs_array_path": _autopsy(
                    array_state["volumes"], weld_state["volumes"],
                    None if singles_state is None else singles_state["volumes"], shape),
                "certified_singles_vs_array_path": (
                    {} if singles_state is None
                    else _autopsy(array_state["volumes"], singles_state["volumes"],
                                  None, shape)),
                "the_weld_and_the_certified_singles_agree_everywhere":
                    (None if singles_state is None
                     else not gate.compare(weld_state["volumes"],
                                           singles_state["volumes"])),
            }
            break
    driver._dft_monitors = detached["dft"]  # noqa: SLF001
    driver._flux_monitors = detached["flux"]  # noqa: SLF001
    return {
        "steps_driven": len(per_step),
        "certified_singles_error": singles_error,
        "first_divergence": first,
        "per_step": per_step,
        "seconds": round(time.perf_counter() - started, 2),
        "reading": (
            "if `the_weld_and_the_certified_singles_agree_everywhere` is true and the "
            "singles ALSO differ from the array path at the same words, the finding is "
            "about this backend's agreement with the array path on this row and NOT "
            "about the weld. If the weld differs from the singles, it is the weld's."),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leg", choices=("examples", "tests"), required=True)
    parser.add_argument("--target", required=True,
                        help="the example script or the test module path")
    parser.add_argument("--cases", default="[]",
                        help="JSON list of test case ids (tests leg)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args(argv)

    import meep_gpu  # noqa: PLC0415

    payloads: List[Dict[str, Any]] = []
    if args.leg == "examples":
        from parity.meep_gpu.sweep_corpus_lift_parity import (  # noqa: PLC0415
            capture_simulation)

        record, sim, restore = capture_simulation(args.target)
        record["row"] = Path(args.target).name
        payloads.append(record)
        pairs = [] if sim is None else [(record, sim)]
        if sim is not None:
            restore()
    else:
        from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415

        namespace = harness.build_child_namespace()
        module = namespace["_import_module"](args.target)
        wanted = set(json.loads(args.cases))
        pairs = []
        for class_name, method_name in namespace["_enumerate_cases"](module):
            case_id = f"{class_name}.{method_name}"
            record, sim, restore, _case = namespace["_run_case"](
                module, args.target, class_name, method_name, 900.0)
            restore()
            if case_id not in wanted:
                continue
            record["row"] = case_id
            payloads.append(record)
            if sim is not None:
                pairs.append((record, sim))

    for record, sim in pairs:
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        family = gate._family()  # noqa: SLF001
        covered, reason = family.covers_fused_hd_pair(
            driver.fields, driver.pml, driver.grid,
            tuple(getattr(driver, "_sources", ()) or ()))
        record["predicate_admits"] = bool(covered)
        record["predicate_reason"] = reason
        if covered:
            record["autopsy"] = run(driver, args.steps)
        driver.close()

    for record in payloads:
        gate_provenance.stamp(record)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payloads, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    for record in payloads:
        autopsy = record.get("autopsy") or {}
        first = autopsy.get("first_divergence")
        print(f"{record.get('row')}: admits={record.get('predicate_admits')} "
              f"steps={autopsy.get('steps_driven')} "
              f"first_divergence_step={(first or {}).get('step')} "
              f"weld_agrees_with_singles="
              f"{(first or {}).get('the_weld_and_the_certified_singles_agree_everywhere')}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
